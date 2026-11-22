"""Aircraft defect deferral: may a (fictional) aircraft fly with a known defect, under what limits, for how long?

Maintenance-control decisions on an invented fleet (Halcyon H-320 / H-350, Corvid CR-9, Tarn T-72), invented
operators, stations and MEL item numbers: tech-log deferrals against the minimum equipment list, repeat defects,
ramp damage measured against repair-manual limits, repair-interval extensions and inspections after a hard landing,
bird strike or similar occurrence. Limits and procedures are invented, shaped like a real MEL / SRM / AMM, and no
real operator, registration or regulator text is used. Each schema draws a `regime` first so most cases sit near a
limit, carry signals that disagree (an engineer's confident note against the route weather), or miss a key fact.
"""
from __future__ import annotations

from datetime import timedelta

from .base import (
    Schema,
    Workflow,
    bullets,
    chance,
    choice,
    dumps,
    fmt_date,
    fmt_ts,
    ident,
    initials_name,
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
    show,
    table,
)

OPERATORS = ["Saltmarsh Air", "Kestrelford Regional", "Valewood Airways", "Copperline Air", "Bluestem Airlines",
             "Quillhaven Air Link", "Fernhill Aviation", "Wexmoor Airways", "Tamsley Air", "Corriston Air"]
# (code, name, kind); kind drives what can be fixed where
STATIONS = [
    ("HMR", "Harrowmere", "main_base"), ("ORB", "Orlan Bay", "main_base"),
    ("PEL", "Port Elden", "line_station"), ("KVM", "Kestrel Vale", "line_station"),
    ("BRK", "Brackenmoor", "contract_station"), ("LSW", "Lower Saltwick", "contract_station"),
    ("DHF", "Dunhollow Field", "outstation"), ("CYI", "Carraway Island", "outstation"),
]
STATION_SHORT = {"main_base": "main base", "line_station": "line station", "contract_station": "contract maintenance",
                 "outstation": "outstation, no engineer"}
STATION_KIND = {"main_base": "main base (hangar, stores, 24h engineers)",
                "line_station": "line station (own engineers, limited stores)",
                "contract_station": "contract maintenance (few spares)",
                "outstation": "outstation (no resident engineer)"}
# type: (name, class, seats, extended-range capable, cruise FL range)
TYPES = {
    "H-320": ("Halcyon H-320", "narrowbody twinjet", 174, False, (290, 390)),
    "H-350": ("Halcyon H-350", "widebody twinjet", 298, True, (310, 410)),
    "CR-9": ("Corvid CR-9", "regional jet", 88, False, (270, 370)),
    "T-72": ("Tarn T-72", "turboprop", 70, False, (170, 250)),
}


def _type(rng, weights=(4, 2, 3, 2)):
    return pick(rng, list(TYPES), list(weights))


def _registration(rng, operator):
    letters = "".join(w[0] for w in operator.split()[:2]).upper()
    fin = rng.randint(101, 489)
    return f"N{fin}{letters}", fin


def _station(rng, kinds=None):
    pool = [s for s in STATIONS if kinds is None or s[2] in kinds]
    return pick(rng, pool)


def _sectors(rng, origin, n):
    codes = [s[0] for s in STATIONS]
    out, cur = [], origin
    for _ in range(n):
        nxt = pick(rng, [c for c in codes if c != cur])
        out.append(f"{cur}-{nxt}")
        cur = nxt
    return out


# =========================================================================== MEL items (invented operator MEL)
MEL = {
    "21-51-02": dict(
        title="Air conditioning pack (one of two)", cat="C", days=10, m=True, o=True,
        proviso="other pack and APU operate normally; cruise at or below FL310",
        limits=("altitude", "apu_needed"),
        exact="PACK 2 FAULT caution in climb, pack tripped off, QRH reset unsuccessful. Cabin temp held on pack 1.",
        extra="PACK 2 FAULT then PACK 2 OVHT in climb, pack selected off per QRH. Cabin crew reported a brief "
              "burning smell at the aft galley, gone after about 5 min.",
        vague="Pack 2 u/s.",
        m_text="pack 2 flow control valve secured closed and placarded",
        o_text="crew briefed on FL310 limit", part="pack flow control valve",
        scope="one pack tripped or failed with a PACK FAULT indication", excl="pack overheat, smoke or fumes"),
    "30-21-01": dict(
        title="Engine intake anti-ice valve", cat="C", days=10, m=True, o=True,
        proviso="valve secured closed; no flight into known or forecast icing conditions",
        limits=("icing",),
        exact="ENG 1 ANTI ICE FAULT when selected ON during taxi test, valve position disagree.",
        extra="ENG 1 ANTI ICE FAULT in descent through cloud; engine 1 vibration rose to 2.9 units briefly, "
              "then back to normal.",
        vague="Eng 1 anti-ice inop.",
        m_text="valve secured closed, position checked", o_text="crew briefed on icing restriction",
        part="anti-ice valve assembly",
        scope="engine anti-ice valve fault or position disagree", excl="engine vibration or any other engine indication"),
    "34-41-01": dict(
        title="Weather radar", cat="C", days=10, m=False, o=True,
        proviso="no thunderstorms or other hazardous convective weather forecast along the route",
        limits=("convective",),
        exact="WX radar display blank both sides, all ranges. Transmitter fault flag on self test.",
        extra="WX radar blank both sides; captain's nav display also flickered twice in cruise.",
        vague="Radar not working properly.",
        m_text=None, o_text="crew briefed, route weather to be confirmed by dispatch", part="radar transceiver",
        scope="weather radar transmitter or display failure", excl="faults on other displays or navigation equipment"),
    "49-10-01": dict(
        title="Auxiliary power unit", cat="C", days=10, m=True, o=True,
        proviso="not dispatched on extended-range sectors; ground power and air start available at destination",
        limits=("etops", "ground_power"),
        exact="APU auto shutdown during start with APU FAULT light. Two further start attempts failed.",
        extra="APU auto shutdown during start; APU FIRE test also failed on the preflight test panel.",
        vague="APU u/s, see previous.",
        m_text="APU fuel valve closed, APU placarded inoperative", o_text="crew briefed, ground air requested",
        part="APU starter-generator",
        scope="APU that will not start or shuts itself down", excl="APU fire detection or extinguishing faults"),
    "28-42-01": dict(
        title="Fuel quantity indication (one tank)", cat="C", days=10, m=True, o=True,
        proviso="affected tank measured by dipstick before every departure; minimum turnaround 40 min",
        limits=("turnaround",),
        exact="LEFT INNER TANK quantity blank on fuel page. Other tanks normal.",
        extra="LEFT INNER tank quantity blank; FUEL IMBALANCE advisory later in cruise, cleared by crossfeed.",
        vague="FQI problem L wing.",
        m_text="left inner tank dipped, quantity entered in tech log", o_text="crew briefed on manual fuel monitoring",
        part="fuel quantity probe",
        scope="quantity indication lost for one tank", excl="fuel imbalance, a suspected leak or other fuel system faults"),
    "32-47-01": dict(
        title="Brake temperature indication (one wheel)", cat="C", days=10, m=True, o=False,
        proviso="brake temperature checked by hand after each landing; minimum turnaround 40 min",
        limits=("turnaround",),
        exact="BRAKE TEMP #3 shows amber XX after landing. Brakes felt normal.",
        extra="BRAKE TEMP #3 shows XX after landing; aircraft pulled slightly right during braking on rollout.",
        vague="Brake temp ind faulty.",
        m_text="#3 brake temperature checked by hand, within limits", o_text=None, part="brake temperature sensor",
        scope="brake temperature indication lost for one wheel", excl="abnormal braking, pulling or brake wear"),
    "23-71-01": dict(
        title="Cockpit voice recorder", cat="A", days=3, m=False, o=False,
        proviso="flight data recorder operates normally; repair within 3 calendar days",
        limits=("fdr_needed",),
        exact="CVR fails preflight test, no tone.",
        extra="CVR fails preflight test; RECORDER FAULT message also shown once during engine start.",
        vague="CVR test fail.",
        m_text=None, o_text=None, part="cockpit voice recorder unit",
        scope="cockpit voice recorder failing its test", excl="flight data recorder or other recorder faults"),
    "36-11-03": dict(
        title="Engine bleed air supply (one engine)", cat="B", days=3, m=True, o=True,
        proviso="APU bleed available; no flight into known or forecast icing; cruise at or below FL310",
        limits=("icing", "altitude", "apu_needed"),
        exact="ENG 2 BLEED FAULT in climb, bleed valve closed automatically. Reset unsuccessful.",
        extra="ENG 2 BLEED FAULT in climb, then a BLEED LEAK caution for about 30 s before it cleared.",
        vague="Bleed 2 problem.",
        m_text="engine 2 bleed valve secured closed", o_text="crew briefed on icing and FL310 limits",
        part="bleed pressure regulating valve",
        scope="engine bleed fault with the bleed valve closed", excl="bleed leak or overheat cautions"),
    "22-11-02": dict(
        title="Autopilot (one of two channels)", cat="C", days=10, m=False, o=True,
        proviso="approaches limited to CAT I minima",
        limits=("low_vis",),
        exact="AP 2 will not engage; AP 1 normal. AP 2 FAULT on test.",
        extra="AP 2 disconnected in cruise with a small uncommanded roll before the crew took over.",
        vague="AP2 inop.",
        m_text=None, o_text="crew briefed on CAT I limitation", part="flight control computer 2",
        scope="one autopilot channel that will not engage", excl="uncommanded control movement or disconnects in flight"),
}

LIMIT_TEXT = {
    "icing": {"none": "no icing forecast on the route", "possible": "light icing possible in descent, freezing level 4,000 ft",
              "forecast": "moderate icing forecast FL080-FL160 at destination"},
    "convective": {"none": "no significant weather", "isolated": "isolated CB forecast near destination after 1500Z",
                   "embedded": "embedded TS forecast along the second half of the route"},
    "low_vis": {"good": "destination visibility good", "marginal": "destination fog forecast, RVR 600-800 m",
                "below_cat1": "destination RVR 350 m, CAT II/III in use"},
}


def _conflict(limit, r):
    """none | marginal | conflict for one MEL limit against the planned flying."""
    if limit == "altitude":
        return "none" if r["planned_fl"] <= 310 else ("marginal" if r["planned_fl"] <= 350 else "conflict")
    if limit == "apu_needed":
        return {"ok": "none", "intermittent": "marginal", "deferred": "conflict"}[r["apu_status"]]
    if limit == "icing":
        return {"none": "none", "possible": "marginal", "forecast": "conflict", None: "marginal"}[r["icing"]]
    if limit == "convective":
        return {"none": "none", "isolated": "marginal", "embedded": "conflict", None: "marginal"}[r["convective"]]
    if limit == "etops":
        return "conflict" if r["extended_range"] else "none"
    if limit == "ground_power":
        return {True: "none", None: "marginal", False: "conflict"}[r["ground_power_dest"]]
    if limit == "turnaround":
        t = r["turnaround_min"]
        return "none" if t >= 50 else ("marginal" if t >= 38 else "conflict")
    if limit == "low_vis":
        return {"good": "none", "marginal": "marginal", "below_cat1": "conflict"}[r["low_vis"]]
    if limit == "fdr_needed":
        return {"ok": "none", "intermittent": "marginal", "fault": "conflict"}[r["fdr_status"]]
    return "none"


def _set_limit(limit, r, level, rng, typ):
    """Push the route variable behind `limit` to a given level (none / marginal / conflict)."""
    lo_fl, hi_fl = TYPES[typ][4]
    if limit == "altitude":
        r["planned_fl"] = {"none": min(310, max(lo_fl, 250 if lo_fl < 290 else 300)),
                           "marginal": pick(rng, [320, 330, 340]), "conflict": pick(rng, [370, 380, 390])}[level]
        if typ == "T-72":
            r["planned_fl"] = pick(rng, [190, 210, 230])
    elif limit == "apu_needed":
        r["apu_status"] = {"none": "ok", "marginal": "intermittent", "conflict": "deferred"}[level]
    elif limit == "icing":
        r["icing"] = {"none": "none", "marginal": "possible", "conflict": "forecast"}[level]
    elif limit == "convective":
        r["convective"] = {"none": "none", "marginal": "isolated", "conflict": "embedded"}[level]
    elif limit == "etops":
        r["extended_range"] = level == "conflict" and TYPES[typ][3]
    elif limit == "ground_power":
        r["ground_power_dest"] = {"none": True, "marginal": None, "conflict": False}[level]
    elif limit == "turnaround":
        r["turnaround_min"] = {"none": pick(rng, [55, 65, 80]), "marginal": near(rng, 40, 4, lo=35, integer=True),
                               "conflict": pick(rng, [25, 30, 35])}[level]
    elif limit == "low_vis":
        r["low_vis"] = {"none": "good", "marginal": "marginal", "conflict": "below_cat1"}[level]
    elif limit == "fdr_needed":
        r["fdr_status"] = {"none": "ok", "marginal": "intermittent", "conflict": "fault"}[level]


# =========================================================================== schema 1
class TechLogMelDeferral(Schema):
    name = "tech_log_mel_deferral"
    description = "A pre-departure tech-log defect that the line engineer proposes to defer under an MEL item."
    surfaces = {"json": 0.5, "tech_log": 0.25, "mcc_log": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        op = pick(rng, OPERATORS)
        typ = _type(rng)
        item_no = pick(rng, list(MEL))
        it = MEL[item_no]
        lims = [x for x in it["limits"] if x != "etops" or TYPES[typ][3]]
        stn = _station(rng)
        lo_fl, hi_fl = TYPES[typ][4]
        route = {"planned_fl": rng.randrange(lo_fl, hi_fl + 1, 10),
                 "icing": pick_w(rng, {"none": 6, "possible": 3, "forecast": 1}),
                 "convective": pick_w(rng, {"none": 6, "isolated": 3, "embedded": 1}),
                 "extended_range": TYPES[typ][3] and chance(rng, 0.5),
                 "ground_power_dest": pick_w(rng, {True: 7, None: 2, False: 1}),
                 "turnaround_min": pick(rng, [35, 45, 55, 70, 90]),
                 "low_vis": pick_w(rng, {"good": 7, "marginal": 2, "below_cat1": 1}),
                 "apu_status": pick_w(rng, {"ok": 8, "intermittent": 1, "deferred": 1}),
                 "fdr_status": pick_w(rng, {"ok": 9, "intermittent": 1}),
                 "night": chance(rng, 0.35)}
        if item_no == "49-10-01":
            route["apu_status"] = "ok"
        scope = pick_w(rng, {"exact": 5, "extra_symptom": 2, "vague": 2})
        m_status = pick_w(rng, {"done_signed": 4, "done_unsigned": 2, "not_done": 2}) if it["m"] else "n/a"
        o_status = pick_w(rng, {"briefed": 4, "not_yet": 2}) if it["o"] else "n/a"
        part = pick_w(rng, {"local_stores": 2, "base_stores": 4, "on_order": 3, "not_ordered": 1})
        eta = rng.randint(2, 12) if part == "on_order" else None
        base_night = pick(rng, [1, 2, 3, 5, 8, 12])
        remark = pick_w(rng, {"routine": 4, "cautious": 2, "none": 3})
        if reg == "clear":
            if chance(rng, 0.55):
                scope, remark = "exact", "routine"
                for lim in lims:
                    _set_limit(lim, route, "none", rng, typ)
                m_status = "done_signed" if it["m"] else "n/a"
                o_status = "briefed" if it["o"] else "n/a"
                part, eta, base_night = pick(rng, ["local_stores", "base_stores"]), None, pick(rng, [1, 2, 3])
            else:
                bad = pick(rng, ["scope", "route", "procedure"] if it["m"] else ["scope", "route"])
                if bad == "scope":
                    scope = "extra_symptom"
                elif bad == "route":
                    _set_limit(lims[0], route, "conflict", rng, typ)
                else:
                    m_status = "not_done"
        elif reg == "borderline":
            lim = pick(rng, lims)
            _set_limit(lim, route, "marginal", rng, typ)
            for other in lims:
                if other != lim:
                    _set_limit(other, route, pick(rng, ["none", "none", "marginal"]), rng, typ)
            if chance(rng, 0.5):
                part, eta = "on_order", near(rng, it["days"] - 1, 2, lo=1, integer=True)
            else:
                base_night = near(rng, it["days"] - 1, 2, lo=1, integer=True)
            if scope == "extra_symptom" and chance(rng, 0.5):
                scope = "vague"
        else:  # conflicting: a confident engineer against a fact that bites
            remark = "confident"
            twist = pick(rng, ["route", "scope", "related"] if "apu_needed" in lims else ["route", "scope"])
            if twist == "route":
                _set_limit(lims[0], route, "conflict", rng, typ)
                scope = "exact"
                m_status = "done_signed" if it["m"] else "n/a"
            elif twist == "scope":
                scope = "extra_symptom"
                for lim in lims:
                    _set_limit(lim, route, "none", rng, typ)
                m_status = "done_signed" if it["m"] else "n/a"
            else:
                route["apu_status"] = "deferred"
            if chance(rng, 0.4):   # a key fact goes missing
                route["icing" if "icing" in it["limits"] or chance(rng, 0.5) else "convective"] = None
        repair_days = {"local_stores": 1, "base_stores": base_night,
                       "on_order": max(eta or 0, base_night) + 1, "not_ordered": None}[part]
        conflicts = {lim: _conflict(lim, route) for lim in lims}
        return {"regime": reg, "operator": op, "aircraft_type": typ, "registration": _registration(rng, op)[0],
                "station": stn[0], "station_name": stn[1], "station_kind": stn[2], "mel_item": item_no,
                "mel_category": it["cat"], "interval_days": it["days"], "writeup_scope": scope,
                "maintenance_procedure": m_status, "operational_procedure": o_status, "route": route,
                "limit_conflicts": conflicts, "part_status": part, "part_eta_days": eta,
                "next_base_night_stop_days": base_night, "repair_opportunity_days": repair_days,
                "engineer_remark": remark, "mel_text_attached": chance(rng, 0.7), "minutes_to_departure": pick(rng, [25, 40, 55, 75, 110, 180]),
                "passengers_booked": rng.randint(20, TYPES[typ][2]),
                "sectors_today": _sectors(rng, stn[0], pick(rng, [1, 2, 3, 4])),
                "other_deferrals": pick(rng, [0, 0, 1, 2, 3]),
                "captain": initials_name(rng), "engineer": person(rng),
                "page": ident(rng, "TL", 6, ""), "deferral_ref": ident(rng, "DD", 5, "-"),
                "date": rand_ts(rng, hours=(4, 20))}

    # ------------------------------------------------------------------ text helpers
    def _entry(self, f):
        return MEL[f["mel_item"]][{"exact": "exact", "extra_symptom": "extra", "vague": "vague"}[f["writeup_scope"]]]

    def _weather(self, f, rng):
        r, it = f["route"], MEL[f["mel_item"]]
        out = []
        if "icing" in it["limits"] or chance(rng, 0.5):
            out.append(LIMIT_TEXT["icing"][r["icing"]] if r["icing"] else "icing forecast not yet received")
        if "convective" in it["limits"] or chance(rng, 0.5):
            out.append(LIMIT_TEXT["convective"][r["convective"]] if r["convective"] else "SIGWX chart not yet received")
        if "low_vis" in it["limits"] or r["low_vis"] != "good":
            out.append(LIMIT_TEXT["low_vis"][r["low_vis"]])
        return "; ".join(out)

    def _others(self, f, rng):
        pool = ["25-21-04 seat 14C recline (cat D)", "33-42-01 logo light (cat D)", "38-31-02 aft lav flush (cat C)",
                "25-31-06 galley 2 oven (cat D)", "31-61-01 cabin clock (cat D)", "33-24-02 reading light 22A (cat D)"]
        out = rng.sample(pool, f["other_deferrals"])
        if f["route"]["apu_status"] == "deferred":
            out.append(f"49-10-01 APU (cat C, day {rng.randint(2, 8)} of 10)")
        elif f["route"]["apu_status"] == "intermittent":
            out.append("APU slow start noted twice this week (no deferral raised)")
        if f["route"]["fdr_status"] == "intermittent":
            out.append("FDR data frame dropouts on last download (under review)")
        return out

    def _remark(self, f, rng):
        return {"routine": pick(rng, ["Standard deferral.", "Routine item, placard fitted.", ""]),
                "cautious": pick(rng, ["Would like MCC to confirm given the weather today.",
                                       "Crew not fully happy, please advise.", "Not sure this item covers everything written up."]),
                "confident": pick(rng, ["Seen this many times, standard deferral, good to go.",
                                        "Nothing unusual, aircraft fine to dispatch.",
                                        "Straightforward, no reason to delay the flight."]),
                "none": ""}[f["engineer_remark"]]

    def _parts(self, f):
        it = MEL[f["mel_item"]]
        s = {"local_stores": f"{it['part']} held in local stores",
             "base_stores": f"{it['part']} at main base stores",
             "on_order": f"{it['part']} on order, supplier ETA {f['part_eta_days']} day" + ("s" if f["part_eta_days"] != 1 else ""),
             "not_ordered": f"{it['part']}: no order raised yet"}[f["part_status"]]
        return s

    def _proc(self, f):
        it = MEL[f["mel_item"]]
        m = {"done_signed": f"(M) done: {it['m_text']}, signed", "done_unsigned": f"(M) {it['m_text']}; certification NOT signed",
             "not_done": "(M) required, not yet carried out", "n/a": None}[f["maintenance_procedure"]]
        o = {"briefed": f"(O) done: {it['o_text']}", "not_yet": "(O) required: crew not yet briefed", "n/a": None}[f["operational_procedure"]]
        return m, o

    def _required(self, f):
        it = MEL[f["mel_item"]]
        return {(True, True): "(M) and (O)", (True, False): "(M) only", (False, True): "(O) only",
                (False, False): "none (no (M) or (O) procedure for this item)"}[(it["m"], it["o"])]

    def _scope(self, f):
        it = MEL[f["mel_item"]]
        return f"applies to {it['scope']}; does not cover {it['excl']}, or a fault not yet identified"

    def render(self, f, rng, surface):
        it = MEL[f["mel_item"]]
        r = f["route"]
        name, cls, seats, _, _ = TYPES[f["aircraft_type"]]
        m, o = self._proc(f)
        wx = self._weather(f, rng)
        others = self._others(f, rng)
        remark = self._remark(f, rng)
        t0 = f["date"]
        std = t0 + timedelta(minutes=f["minutes_to_departure"])
        cruise = f"FL{r['planned_fl']}"
        if surface == "json":
            return dumps(prune({
                "aircraft": {"registration": f["registration"], "type": name, "operator": f["operator"],
                             "open_deferrals": others or None},
                "defect": {"tech_log_page": f["page"], "reported_by": f"Capt. {f['captain']}", "entry": self._entry(f)},
                "proposed_deferral": {"mel_item": f["mel_item"], "title": it["title"], "category": it["cat"],
                                      "interval_days": it["days"], "item_scope": self._scope(f),
                                      "procedures_required": self._required(f), "maintenance_procedure": m,
                                      "mel_conditions": it["proviso"] if f["mel_text_attached"] else None,
                                      "operational_procedure": o, "engineer": f["engineer"],
                                      "engineer_remarks": noisy(remark, rng) or None},
                "departure": {"station": f"{f['station']} {f['station_name']}", "station_type": STATION_SHORT[f['station_kind']],
                              "scheduled_utc": fmt_ts(std, style="iso"), "minutes_to_departure": f["minutes_to_departure"],
                              "sectors_today": f["sectors_today"], "planned_cruise": cruise,
                              "extended_range_sector": r["extended_range"] or None,
                              "night_departure": r["night"], "passengers_booked": f["passengers_booked"],
                              "turnaround_minutes": r["turnaround_min"] if "turnaround" in it["limits"] else None,
                              "ground_power_at_destination": (None if r["ground_power_dest"] is None
                                                              else ("yes" if r["ground_power_dest"] else "no"))
                              if "ground_power" in it["limits"] else None},
                "route_weather": wx or None,
                "parts": {"status": self._parts(f), "next_main_base_night_stop_in_days": f["next_base_night_stop_days"]},
            }))
        if surface == "tech_log":
            head = f"TECH LOG -- {f['operator']}   A/C {f['registration']} ({name})   Page {f['page']}"
            rows = kv_lines([("Station", f"{f['station']} {f['station_name']}, {STATION_KIND[f['station_kind']]}"),
                             ("Date", fmt_date(t0.date(), rng)), ("STD", std.strftime("%H:%MZ")),
                             ("Sectors", ", ".join(f["sectors_today"])), ("Planned cruise", cruise)], rng)
            body = [head, rows, "", f"DEFECT (Capt. {f['captain']}):", "  " + self._entry(f), "",
                    f"ACTION ({f['engineer']}, B1):",
                    f"  Deferred IAW MEL {f['mel_item']} {it['title']}, cat {it['cat']} ({it['days']} days). Ref {f['deferral_ref']}.",
                    f"  Item scope: {self._scope(f)}.", f"  Procedures required: {self._required(f)}"]
            body += [f"  {x}" for x in (m, o) if x]
            if f["mel_text_attached"]:
                body.append(f"  MEL remarks: {it['proviso']}.")
            if remark:
                body.append("  Remarks: " + noisy(remark, rng))
            body += ["", "Parts: " + self._parts(f),
                     f"Next main-base night stop: {f['next_base_night_stop_days']} day(s)",
                     "Route wx: " + (wx or "not attached")]
            if "ground_power" in it["limits"]:
                body.append("Ground power at destination: " + show(r["ground_power_dest"], rng,
                                                                   lambda v: "confirmed" if v else "not available"))
            if "turnaround" in it["limits"]:
                body.append(f"Scheduled turnaround: {r['turnaround_min']} min")
            if r["extended_range"]:
                body.append("Second sector is an extended-range (ETOPS) sector.")
            body.append("Open deferrals: " + ("; ".join(others) if others else "nil"))
            return "\n".join(body)
        # mcc_log
        t = t0
        lines = [f"MCC LOG {fmt_date(t0.date(), rng)} -- {f['registration']} {f['aircraft_type']} at {f['station']}"]

        def add(mins, text):
            nonlocal t
            t += timedelta(minutes=mins)
            lines.append(f"{t.strftime('%H:%M')}  {text}")
        add(0, f"{f['station']} ({f['engineer'].split()[-1]}) calls re tech log {f['page']}: \"{self._entry(f)}\"")
        add(rng.randint(2, 6), (f"Proposes MEL {f['mel_item']} ({it['title']}, cat {it['cat']}). "
                                + noisy(remark, rng, 0.1)).strip())
        add(1, f"MEL scope: {self._scope(f)}. Procedures required: {self._required(f)}.")
        if f["mel_text_attached"]:
            add(1, f"MEL text: {it['proviso']}.")
        if m or o:
            add(rng.randint(1, 4), "; ".join(x for x in (m, o) if x))
        add(rng.randint(1, 5), f"Dispatch: {', '.join(f['sectors_today'])}, {cruise}"
            + (", 2nd sector ETOPS" if r["extended_range"] else "")
            + f", {f['passengers_booked']} pax, STD {std.strftime('%H:%M')}.")
        add(rng.randint(1, 5), "Wx desk: " + (wx or "no route weather on file yet"))
        if "ground_power" in it["limits"]:
            add(1, "Handling agent: ground power/air start at destination "
                + show(r["ground_power_dest"], rng, lambda v: "available" if v else "NOT available"))
        if "turnaround" in it["limits"]:
            add(1, f"Ops: turnaround at next station {r['turnaround_min']} min")
        add(rng.randint(1, 4), "Stores: " + self._parts(f) + f"; next base night stop in {f['next_base_night_stop_days']} d.")
        if others:
            add(1, "Open items on aircraft: " + "; ".join(others))
        add(rng.randint(1, 3), "Awaiting MCC decision.")
        return "\n".join(lines)

    def questions(self):
        return {
            "dispatch_call": choice("What should maintenance control decide about releasing this aircraft?", {
                "defer_and_release": "Accept the deferral under the cited MEL item and release the aircraft for its "
                                     "planned flying.",
                "defer_with_restriction": "Accept the deferral but release the aircraft only for flights that respect "
                                          "the item's limits, changing the level, sector or timing as needed.",
                "fix_before_flight": "Do not defer; the defect has to be rectified before the aircraft flies again.",
                "swap_aircraft": "Take this aircraft off the planned flying and cover it with another airframe while "
                                 "the defect is dealt with.",
                "refer_to_engineering": "Hold the release and ask technical engineering for a ruling before anyone "
                                        "decides.",
            }),
            "mel_scope_fit": noul(
                "The defect written up in the tech log is fully covered by the MEL item the engineer cited.",
                yes="The entry describes exactly the failure that item addresses, with no other symptom reported.",
                no="The entry is too vague to tell, or reports symptoms or systems the item does not address."),
            "route_limit_conflict": score("How far do the MEL item's operating limits clash with the flying the "
                                          "aircraft is planned to do?", [
                "Not at all: the planned flights sit well inside the item's limits.",
                "Slightly: one limit needs watching, such as a forecast to recheck before departure.",
                "Materially: a planned sector would have to be changed or restricted to comply.",
                "Directly: the planned flying breaks a condition of the item as it stands.",
            ]),
            "rectification_pressure": score("How hard will it be to clear the defect before the deferral "
                                            "interval runs out?", [
                "Easy: the part and engineers are on the aircraft's routing well inside the interval.",
                "Manageable: the fix fits a planned night stop at a capable station.",
                "Tight: the interval ends close to the first realistic repair opportunity.",
                "At risk: no station with the part and staff is on the routing before the interval ends.",
            ]),
            "procedures_complete": noul(
                "Every maintenance and operational procedure the MEL item calls for has been completed and recorded "
                "for this departure.",
                yes="Each (M) and (O) procedure the item requires is recorded as done, with the (M) certification "
                    "signed; an item that requires none counts as complete.",
                no="A required procedure is not yet done, its certification is not signed, or the crew has not been "
                   "briefed."),
        }


# =========================================================================== schema 2
# what the crew must do if the fault comes back in flight (drives how much it matters)
RECUR_DRILL = {
    "21-31": "QRH: select the standby pressure controller; if that fails too, descend to FL100 and divert",
    "27-51": "QRH: land with flaps as locked, higher approach speed; a longer runway may mean diverting",
    "30-42": "QRH: keep below 250 kt under FL100; no further action",
    "32-61": "QRH: recycle the gear; if the disagree persists, land gear down and ferry to base for inspection",
    "36-12": "QRH: increase thrust or open the crossbleed; no further action",
    "73-21": "no crew action; maintenance message only",
    "22-31": "QRH: fly manual thrust for the rest of the approach",
    "34-22": "QRH: unreliable airspeed drill; continue only with two good air data sources, else divert",
}
REPEAT_SYSTEMS = {
    "21-31": ("cabin pressure controller", "CAB PRESS SYS 1 FAULT in climb, system 2 took over", "degraded"),
    "27-51": ("flap asymmetry detection", "FLAP ASYM caution on approach, flaps locked at 15", "diversion"),
    "30-42": ("windshield heat", "WSHLD HEAT L FAULT in cruise", "workload"),
    "32-61": ("landing gear proximity sensing", "GEAR DISAGREE after retraction, cleared on recycle", "degraded"),
    "36-12": ("bleed pressure regulation", "BLEED 1 LOW PRESS in descent, recovered with thrust increase", "workload"),
    "73-21": ("engine fuel control", "ENG 2 EEC minor fault message after landing", "nuisance"),
    "22-31": ("autothrottle servo", "A/THR disconnect on approach, re-engaged normally", "workload"),
    "34-22": ("air data module", "ADR 2 FAULT in climb, airspeed disagree for about 20 s", "diversion"),
}
ACTIONS = {
    "reset": ["circuit breaker reset, BITE test passed", "system reset, ops test good"],
    "cnd": ["could not duplicate on ground, no fault stored", "fault not reproduced, connectors checked"],
    "swap": ["units 1 and 2 swapped, ops test good", "swapped with serviceable unit from stores for fault isolation"],
    "part_unconfirmed": ["suspect sensor replaced, ops test good", "connector re-pinned and sensor replaced, ops test good"],
    "part_confirmed": ["fault reproduced on ground; failed unit replaced, bench report confirms internal failure",
                       "chafed wire found at frame 42 and repaired, fault no longer reproducible"],
}


class RepeatDefectReview(Schema):
    name = "repeat_defect_review"
    description = "A recurring defect on one aircraft, reviewed by reliability engineering after the latest occurrence."
    surfaces = {"json": 0.5, "reliability_table": 0.25, "engineering_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        op = pick(rng, OPERATORS)
        typ = _type(rng)
        ata = pick(rng, list(REPEAT_SYSTEMS))
        thr_n, thr_days = pick(rng, [(3, 10), (3, 15), (4, 30)])
        n = pick(rng, [2, 3, 3, 4, 5])
        span = rng.randint(4, thr_days + 8)
        mixed_code = chance(rng, 0.25)
        last_action = pick_w(rng, {"reset": 3, "cnd": 3, "swap": 2, "part_unconfirmed": 3, "part_confirmed": 2})
        recurred_after_fix = chance(rng, 0.3)
        consistent_reports = chance(rng, 0.7)
        if reg == "clear":
            if chance(rng, 0.5):
                n, span, mixed_code, last_action, recurred_after_fix = thr_n + 1, max(3, thr_days // 2), False, "cnd", True
            else:
                n, span, mixed_code, last_action, recurred_after_fix = 2, thr_days + 10, False, "part_confirmed", False
        elif reg == "borderline":
            n = thr_n
            span = near(rng, thr_days, 2, lo=2, integer=True)     # the first entry sits on the window edge
            if chance(rng, 0.4):
                mixed_code = True
        else:
            # a confident fix that the data does not bear out, or an alarming count that is really two faults
            if chance(rng, 0.5):
                last_action, recurred_after_fix, n = "part_unconfirmed", True, max(n, thr_n)
            else:
                n, mixed_code, consistent_reports = thr_n + 1, True, False
            span = rng.randint(max(3, thr_days - 4), thr_days + 3)
        start = rand_ts(rng).date()
        offs = sorted([0, span] + [rng.randint(1, max(1, span - 1)) for _ in range(max(0, n - 2))])[:n]
        f = {"regime": reg, "operator": op, "aircraft_type": typ, "registration": _registration(rng, op)[0],
             "ata": ata, "system": REPEAT_SYSTEMS[ata][0], "consequence_class": REPEAT_SYSTEMS[ata][2],
             "threshold_count": thr_n, "threshold_days": thr_days, "occurrences": n, "span_days": span,
             "mixed_subcodes": mixed_code, "last_action": last_action,
             "recurred_after_last_fix": recurred_after_fix, "reports_consistent": consistent_reports,
             "day_offsets": offs, "first_date": start,
             "next_base_overnight_days": pick(rng, [0, 1, 2, 4, 7]),
             "daily_sectors": pick(rng, [2, 4, 6, 8]),
             "controller_view": pick_w(rng, {"fixed": 3, "unsure": 3, "worried": 2, None: 2}),
             "reliability_engineer": person(rng), "controller": person(rng),
             "report_ref": ident(rng, "REL", 4, "-")}
        f["recurrence_gap_days"] = rng.randint(1, 3)   # the entry after the latest fix, when there is one
        return f

    def _entries(self, f, rng):
        ata, (_, symptom, _) = f["ata"], REPEAT_SYSTEMS[f["ata"]]
        n = f["occurrences"]
        acts = []
        for i in range(n):
            if i == n - 1:
                kind = f["last_action"]
            else:
                kind = pick(rng, ["reset", "cnd", "cnd", "swap", "part_unconfirmed"])
            acts.append(pick(rng, ACTIONS[kind]))
        rows = []
        for i, off in enumerate(f["day_offsets"]):
            code = ata
            sym = symptom
            if f["mixed_subcodes"] and i == 1:
                code = ata[:4] + str((int(ata[4]) + 1) % 10)
                sym = pick(rng, ["related caution on the ground during power transfer, no flight effect",
                                 "different message on the same panel, maintenance page only"])
            elif not f["reports_consistent"] and i % 2 == 1:
                sym = pick(rng, ["crew report: 'similar to previous', no message captured",
                                 "intermittent, crew unsure which caution appeared"])
            rows.append({"date": f["first_date"] + timedelta(days=off), "ata": code, "report": sym, "action": acts[i]})
        if f["recurred_after_last_fix"] and rows:
            last = rows[-1]["date"] + timedelta(days=f["recurrence_gap_days"])
            rows.append({"date": last, "ata": ata, "report": symptom + " (after latest rectification)",
                         "action": "deferred pending review"})
        for r in rows:
            r["day"] = (r["date"] - f["first_date"]).days
        return rows

    def _view(self, f, rng):
        return {"fixed": pick(rng, ["MCC believes the latest action fixed it.", "Line says the new part sorted it."]),
                "unsure": pick(rng, ["MCC not convinced either way.", "Controller wants a second opinion."]),
                "worried": pick(rng, ["MCC worried this will bite on a busy day.", "Crews starting to write it up with "
                                      "comments; controller uneasy."]),
                None: ""}[f["controller_view"]]

    def render(self, f, rng, surface):
        rows = self._entries(f, rng)
        rule = f"{f['threshold_count']} or more entries on the same 4-digit ATA within {f['threshold_days']} days"
        view = self._view(f, rng)
        if surface == "json":
            return dumps(prune({
                "aircraft": {"registration": f["registration"], "type": TYPES[f["aircraft_type"]][0],
                             "operator": f["operator"], "sectors_per_day": f["daily_sectors"]},
                "defect_history": [{"date": r["date"].isoformat(), "day": r["day"], "ata": r["ata"], "report": r["report"],
                                    "rectification": r["action"]} for r in rows],
                "programme": {"repeat_defect_rule": rule, "system": f"{f['ata']} {f['system']}",
                              "if_it_recurs_in_flight": RECUR_DRILL[f["ata"]]},
                "planning": {"next_main_base_overnight_in_days": f["next_base_overnight_days"],
                             "maintenance_control_view": view or None, "review_ref": f["report_ref"]},
            }))
        if surface == "reliability_table":
            tbl = table(["date", "day", "ATA", "report", "action"],
                        [[fmt_date(r["date"], rng, "iso"), r["day"], r["ata"], r["report"], r["action"]] for r in rows], rng)
            return "\n".join([
                f"REPEAT DEFECT REVIEW {f['report_ref']} -- {f['registration']} ({TYPES[f['aircraft_type']][0]}, {f['operator']})",
                f"System: {f['ata']} {f['system']}", f"Programme rule: {rule}",
                f"If it recurs in flight: {RECUR_DRILL[f['ata']]}", "", tbl, "",
                kv_lines([("Next base overnight", f"in {f['next_base_overnight_days']} day(s)"),
                          ("Utilisation", f"{f['daily_sectors']} sectors/day"),
                          ("MCC comment", view or None)], rng)])
        lines = [f"Subject: {f['registration']} {f['ata']} {f['system']} -- repeat defect review",
                 f"From: {f['reliability_engineer']}", "To: mcc-duty@ops.example", "",
                 f"{f['controller'].split()[0]},", "",
                 noisy(f"Pulling together the {f['ata']} history on {f['registration']} ahead of tonight's planning. "
                       f"Our rule is {rule}. {view}", rng), "",
                 f"If it comes back in flight -- {RECUR_DRILL[f['ata']]}.", ""]
        dstyle = pick(rng, ["iso", "us", "eu", "short"])
        lines += [f"{fmt_date(r['date'], style=dstyle)} (day {r['day']}): {r['report']} -> {r['action']}"
                  + (f" [{r['ata']}]" if r["ata"] != f["ata"] else "") for r in rows]
        nb = f["next_base_overnight_days"]
        when = "tonight" if nb == 0 else "tomorrow night" if nb == 1 else f"in {nb} days"
        lines += ["", f"It is next at a main base overnight {when} and flies "
                      f"{f['daily_sectors']} sectors a day until then.", "", f['reliability_engineer'].split()[0]]
        return "\n".join(lines)

    def questions(self):
        return {
            "repeat_defect_plan": choice("How should engineering handle this recurring defect now?", {
                "close_on_latest_fix": "Treat the latest rectification as the fix and return to normal monitoring: the "
                                       "failed part was confirmed and nothing has recurred since.",
                "targeted_monitoring": "Keep flying, with extra crew reporting and a data download after each sector, "
                                       "because the entries do not yet amount to a repetitive defect.",
                "structured_troubleshooting": "Plan a structured troubleshooting session at the next base overnight, "
                                              "because the entries meet the repeat rule and the cause is unconfirmed.",
                "withdraw_from_service": "Keep the aircraft on the ground until the root cause is found and cleared, "
                                         "because the fault came back after the latest fix and could force a diversion.",
            }),
            "root_cause_confidence": score("How well established is the root cause of the recurring fault?", [
                "Unknown: the actions so far are resets or no-fault-found.",
                "Suspected: there is a likely culprit that nobody has confirmed.",
                "Probable: the latest fix targets a component the evidence points to.",
                "Established: the fault was reproduced and the failed part confirmed.",
            ]),
            "inflight_consequence": score("If the fault came back in flight, how much would it matter?", [
                "Nuisance: a message with no crew action required.",
                "Workload: a checklist and some extra crew workload.",
                "Degraded: a system loses redundancy and the flight plan may change.",
                "Diversion likely: the crew would probably turn back or divert.",
            ]),
            "repeat_rule_met": noul(
                "These entries count as a repetitive defect under the operator's reliability programme.",
                yes="Enough entries fall inside the programme window and describe the same fault on the same system.",
                no="Too few entries fall inside the window, or they describe different faults or systems."),
            "operating_limitation_warranted": noul(
                "The aircraft should carry an operational limitation until the fault is cleared.",
                yes="The fault is not shown to be fixed (it came back after the latest action, or no failed part was "
                    "confirmed), and if it recurs in flight the drill degrades a system or may mean diverting.",
                no="The latest action confirmed and replaced the failed part with no recurrence since, or a recurrence "
                   "would only add a message or a checklist with no further action."),
        }


# =========================================================================== schema 3
DAMAGE = {
    # kind: (zone text, allowable depth mm, minimum distance to fastener row mm, typical cause)
    "dent": ("lower fuselage skin between frames 38 and 39, left side", 1.5, 20, "belt loader contact"),
    "scratch": ("forward cargo door surround, lower sill", 0.25, 10, "cargo container scrape"),
    "lightning_pitting": ("radome and forward fuselage skin", 0.4, 0, "lightning strike in descent"),
    "bird_strike_dent": ("left wing leading-edge slat 2", 2.0, 15, "bird strike on approach"),
    "composite_gouge": ("right wing-to-body fairing (composite panel)", 0.5, 25, "towbar contact"),
}


class StructuralDamageFinding(Schema):
    name = "structural_damage_finding"
    description = "Airframe damage found on the ramp and measured against the structural repair manual's limits."
    surfaces = {"json": 0.45, "damage_report": 0.3, "handover_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        op = pick(rng, OPERATORS)
        typ = _type(rng, (4, 2, 3, 1))
        kind = pick(rng, list(DAMAGE))
        zone, limit, min_dist, cause = DAMAGE[kind]
        depth = round(limit * rng.uniform(0.3, 1.3), 2)
        second_ratio = None if chance(rng, 0.6) else round(rng.uniform(0.85, 1.2), 2)
        dist = rng.randint(max(0, min_dist - 8), min_dist + 30) if min_dist else None
        on_frame = chance(rng, 0.12)
        corrosion = chance(rng, 0.1)
        cause_known = chance(rng, 0.6)
        ndt_here = chance(rng, 0.5)
        photos = chance(rng, 0.75)
        prior_repair_nearby = chance(rng, 0.15)
        dep_min = pick(rng, [35, 45, 90, 150, 300, 600])
        spare = chance(rng, 0.35)
        if reg == "clear":
            if chance(rng, 0.5):
                depth, second_ratio, on_frame, corrosion, prior_repair_nearby = round(limit * rng.uniform(0.2, 0.45), 2), None, False, False, False
                dist = min_dist + rng.randint(15, 40) if min_dist else None
                photos, cause_known = True, True
            else:
                depth = round(limit * rng.uniform(1.4, 2.2), 2)
                on_frame = chance(rng, 0.5)
        elif reg == "borderline":
            depth = near(rng, limit, limit * 0.12, lo=0.05)
            if min_dist and chance(rng, 0.5):
                dist = near(rng, min_dist, 4, lo=0, integer=True)
            if chance(rng, 0.4):
                second_ratio = pick(rng, [0.9, 1.1, 1.15])
        else:
            # a small-looking mark with a worrying context, or a measurement dispute under schedule pressure
            if chance(rng, 0.5):
                depth = round(limit * rng.uniform(0.4, 0.7), 2)
                on_frame, cause_known, prior_repair_nearby = chance(rng, 0.5), False, True
            else:
                depth = round(limit * rng.uniform(0.8, 0.95), 2)
                second_ratio = round(rng.uniform(1.12, 1.4), 2)
                dep_min, spare = pick(rng, [25, 35, 45]), False
            if chance(rng, 0.4):
                photos = False
        measured_by = pick_w(rng, {"b1_engineer": 5, "structures_inspector": 2, "ramp_lead": 2})
        second = round(depth * second_ratio, 2) if second_ratio else None
        pax = ("on_board" if dep_min <= 25 else pick(rng, ["boarding", "not_boarded"]) if dep_min <= 45
               else "not_boarded")
        f = {"regime": reg, "operator": op, "aircraft_type": typ, "registration": _registration(rng, op)[0],
             "damage_kind": kind, "zone": zone, "allowable_depth_mm": limit, "measured_depth_mm": depth,
             "second_measurement_mm": second, "length_mm": rng.randint(8, 140),
             "distance_to_fastener_row_mm": dist, "min_distance_mm": min_dist or None,
             "on_frame_or_stringer": on_frame, "corrosion_seen": corrosion, "cause": cause,
             "cause_known": cause_known, "ndt_available_here": ndt_here, "photos_attached": photos,
             "prior_repair_within_150mm": prior_repair_nearby, "minutes_to_departure": dep_min,
             "spare_aircraft": spare, "passenger_status": pax, "measured_by": measured_by,
             "station": _station(rng)[0], "finder": person(rng), "ref": ident(rng, "DMG", 5, "-"),
             "found_at": rand_ts(rng)}
        # the distance to the nearest fastener row is sometimes not taken (only matters where the manual limits it)
        f["position_recorded"] = not (min_dist and chance(rng, 0.2))
        return f

    WHO = {"b1_engineer": "B1 engineer (structures-authorised)", "structures_inspector": "structures inspector",
           "ramp_lead": "ramp lead with a depth gauge (not authorised for structural measurement)"}

    def _second(self, f):
        if not f["second_measurement_mm"]:
            return None
        d = round((f["second_measurement_mm"] / f["measured_depth_mm"] - 1) * 100)
        return f"{f['second_measurement_mm']} mm ({d:+d}% against the first reading)"

    def _dist(self, f):
        if f["min_distance_mm"] is None:
            return None
        return f"{f['distance_to_fastener_row_mm']} mm" if f["position_recorded"] else "not measured"

    def _cause_text(self, f):
        if f["cause_known"]:
            return f"reported {f['cause']}"
        return "found on walkaround, cause not known"

    def render(self, f, rng, surface):
        srm = f"SRM {pick(rng, ['53-10-11', '53-30-02', '57-40-05', '51-10-02'])} fig. {rng.randint(1, 9)}"
        mdist = f["min_distance_mm"]
        limit_txt = f"allowable depth {f['allowable_depth_mm']} mm" + (f", at least {mdist} mm from fastener rows" if mdist else "") \
            + ", not over a frame or stringer"
        who = self.WHO[f["measured_by"]]
        if surface == "json":
            return dumps(prune({
                "aircraft": {"registration": f["registration"], "type": TYPES[f["aircraft_type"]][0],
                             "operator": f["operator"], "station": f["station"]},
                "finding": {"ref": f["ref"], "type": f["damage_kind"].replace("_", " "), "location": f["zone"],
                            "circumstances": self._cause_text(f), "found_by": f["finder"],
                            "photos_attached": f["photos_attached"]},
                "measurements": {"depth_mm": f["measured_depth_mm"], "second_depth_reading": self._second(f),
                                 "length_mm": f["length_mm"], "distance_to_fastener_row": self._dist(f),
                                 "over_frame_or_stringer": f["on_frame_or_stringer"],
                                 "corrosion_or_cracking_seen": f["corrosion_seen"], "measured_by": who,
                                 "previous_repair_within_150mm": f["prior_repair_within_150mm"] or None},
                "manual_limits": {"reference": srm, "limits": limit_txt},
                "operations": {"minutes_to_departure": f["minutes_to_departure"], "passengers": f["passenger_status"].replace("_", " "),
                               "spare_aircraft_available": f["spare_aircraft"], "ndt_capability_on_station": f["ndt_available_here"]},
            }))
        if surface == "damage_report":
            rows = [("Report", f["ref"]), ("Aircraft", f"{f['registration']} {TYPES[f['aircraft_type']][0]}"),
                    ("Station", f["station"]), ("Found", fmt_ts(f["found_at"], rng)), ("Location", f["zone"]),
                    ("Damage", f["damage_kind"].replace("_", " ")), ("Circumstances", self._cause_text(f)),
                    ("Depth", f"{f['measured_depth_mm']} mm" + (f" (re-measured {self._second(f)})"
                                                               if f["second_measurement_mm"] else "")),
                    ("Length", f"{f['length_mm']} mm"),
                    ("Distance to nearest fastener row", self._dist(f) or "n/a (no fastener-row limit)"),
                    ("Over frame/stringer", "yes" if f["on_frame_or_stringer"] else "no"),
                    ("Corrosion/cracks", "seen" if f["corrosion_seen"] else "none seen"),
                    ("Previous repair nearby", "yes, within 150 mm" if f["prior_repair_within_150mm"] else None),
                    ("Measured by", who), ("Photos", "attached" if f["photos_attached"] else "none"),
                    ("Manual reference", f"{srm}: {limit_txt}")]
            ops = (f"Ops: next departure in {f['minutes_to_departure']} min, passengers {f['passenger_status'].replace('_', ' ')}, "
                   f"{'spare aircraft available' if f['spare_aircraft'] else 'no spare aircraft'}, "
                   f"NDT {'available' if f['ndt_available_here'] else 'not available'} on station.")
            return "AIRFRAME DAMAGE REPORT\n" + kv_lines(rows, rng) + "\n\n" + ops
        # handover_note: free text from the station to technical services
        s = [f"{f['registration']} at {f['station']}: {f['damage_kind'].replace('_', ' ')} on the {f['zone']}, "
             f"{self._cause_text(f)}.",
             f"{who.capitalize()} measured {f['measured_depth_mm']} mm deep, {f['length_mm']} mm long"
             + (f", {f['distance_to_fastener_row_mm']} mm from the nearest fastener row" if f["min_distance_mm"] and f["position_recorded"]
                else ", distance to the nearest fastener row not measured" if f["min_distance_mm"] else "")
             + ".",
             (f"A second reading gave {self._second(f)}." if f["second_measurement_mm"] else ""),
             ("It sits over a stringer." if f["on_frame_or_stringer"] else ""),
             ("Some corrosion visible at the edge." if f["corrosion_seen"] else ""),
             ("There is an old doubler repair just aft of it." if f["prior_repair_within_150mm"] else ""),
             (f"Photos to follow." if not f["photos_attached"] else "Photos attached."),
             f"Our manual reference is {srm} ({limit_txt}).",
             f"Flight is due out in {f['minutes_to_departure']} min, pax {f['passenger_status'].replace('_', ' ')}"
             + (", and there is a spare aircraft." if f["spare_aircraft"] else ", no spare tail today."),
             ("We have eddy-current kit here." if f["ndt_available_here"] else "No NDT capability on station.")]
        return f"From: {f['station']} line\nTo: Technical Services structures desk\n\n" + noisy(join_sentences(s), rng, 0.15)

    def questions(self):
        return {
            "damage_disposition": choice("How should the damage be dispositioned?", {
                "release_within_limits": "Record it and release the aircraft; the damage is inside the manual's limits.",
                "release_with_repeat_inspection": "Release the aircraft with a repeat inspection of the damage at a "
                                                  "set interval.",
                "temporary_repair_then_release": "Apply a time-limited repair from the manual now and schedule the "
                                                 "permanent repair.",
                "ground_pending_engineering": "Keep the aircraft on the ground until engineering or the manufacturer "
                                              "approves a repair.",
            }),
            "inspection_method": choice("What inspection does the area need before anyone signs it off?", {
                "detailed_visual_remeasure": "A detailed visual inspection with proper tools to re-measure depth and "
                                             "position.",
                "general_visual_enough": "A general visual check is enough; nothing suggests damage beyond what is seen.",
                "internal_structure_check": "Panels or linings must come off so the frames and stringers behind it "
                                            "can be examined.",
                "nondestructive_testing": "Eddy-current, ultrasonic or tap testing to rule out cracks or delamination "
                                          "below the surface.",
            }),
            "limit_margin": score("How much margin is there between the damage and the repair manual's allowable "
                                  "limits?", [
                "Ample: well inside every limit.",
                "Adequate: inside the limits with some room.",
                "Marginal: at or near a limit, or it depends on which reading is used.",
                "None: at least one limit is exceeded.",
            ]),
            "disposition_time_pressure": score("How pressing is a ruling for the aircraft's next departure?", [
                "Little: the aircraft is not due out for hours, or another aircraft can take the flight.",
                "Moderate: a departure later in the shift depends on it.",
                "High: the next departure is close and no other aircraft can cover it.",
                "Acute: passengers are boarding or on board and the flight is waiting on this ruling.",
            ]),
            "measurement_sufficient": noul(
                "The damage has been measured and located well enough to apply the repair manual's limits.",
                yes="Depth, size and position are recorded by a qualified person, with readings that agree.",
                no="A key dimension or position is missing, the readings disagree, or the measurer is not qualified."),
        }


# =========================================================================== schema 4
# how maintenance control abbreviates each item in extension paperwork
EXT_SHORT = {"21-51-02": "pack 2 inop", "30-21-01": "eng 1 A/I valve", "34-41-01": "wx radar u/s",
             "49-10-01": "APU inop", "28-42-01": "L inner FQI blank", "32-47-01": "#3 brake temp ind",
             "23-71-01": "CVR u/s", "36-11-03": "eng 2 bleed off", "22-11-02": "AP 2 inop"}

# the item's operating conditions as maintenance control abbreviates them on extension paperwork
EXT_COND = {"21-51-02": "max FL310, APU must work", "30-21-01": "no icing", "34-41-01": "no convective weather en route",
            "49-10-01": "no ETOPS, ground air at destination", "28-42-01": "dip tank each departure, 40 min turns",
            "32-47-01": "hand-check brake temp, 40 min turns", "23-71-01": "FDR must be serviceable",
            "36-11-03": "APU bleed needed, no icing, max FL310", "22-11-02": "CAT I approaches only"}

EXT_REASONS = {
    "supplier_backorder": "part backordered at the supplier; order placed {d} day(s) after the deferral",
    "part_lost_in_transit": "replacement shipped but lost in transit between stations",
    "wrong_part_supplied": "part received was the wrong dash number and had to be returned",
    "no_qualified_staff": "no engineer with the type rating available at the station",
    "planning_oversight": "rectification missed in the night-stop work package",
    "awaiting_tooling": "special tooling for the repair on loan from another operator, not yet arrived",
}


class DeferralExtensionRequest(Schema):
    name = "deferral_extension_request"
    description = "A request to extend the rectification interval of a deferred defect that is about to expire."
    surfaces = {"json": 0.5, "extension_form": 0.25, "mcc_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        op = pick(rng, OPERATORS)
        typ = _type(rng)
        item_no = pick(rng, [k for k in MEL if MEL[k]["cat"] in ("B", "C")] + ["23-71-01"])
        it = MEL[item_no]
        reason = pick(rng, list(EXT_REASONS))
        ordered_after = rng.randint(0, 6)
        hours_left = pick(rng, [6, 14, 30, 48])
        requested = pick(rng, [2, 3, 5, 7, 10])
        prior_ext = chance(rng, 0.22)
        related_open = chance(rng, 0.35)
        n_open = pick(rng, [0, 1, 2, 3, 5])
        can_reach_repair = pick_w(rng, {True: 4, False: 3, None: 1})
        eta_days = rng.randint(1, 9)
        notified = pick_w(rng, {"not_yet": 5, "drafted": 2, "sent": 1})
        if reg == "clear":
            if chance(rng, 0.5):
                reason, ordered_after, prior_ext, related_open, n_open = "supplier_backorder", 0, False, False, pick(rng, [0, 1])
                requested = max(1, min(requested, it["days"] // 2, eta_days + 1))
            else:
                reason, prior_ext = pick(rng, ["planning_oversight", "no_qualified_staff"]), chance(rng, 0.5)
                related_open, n_open = True, pick(rng, [3, 5])
        elif reg == "borderline":
            requested = near(rng, it["days"], 2, lo=1, integer=True)        # at the edge of the one-interval cap
            if reason == "supplier_backorder":
                ordered_after = pick(rng, [3, 4, 5])
            eta_days = near(rng, requested, 1.5, lo=1, integer=True)
        else:
            if chance(rng, 0.5):
                reason, related_open = "supplier_backorder", True         # a good reason on an aircraft carrying a lot
                n_open = pick(rng, [3, 4, 5])
            else:
                reason, can_reach_repair = "planning_oversight", True     # a poor reason, easy alternative
                related_open, n_open = False, 0
            if chance(rng, 0.4):
                eta_days = None
        opened = rand_ts(rng)
        return {"regime": reg, "operator": op, "aircraft_type": typ, "registration": _registration(rng, op)[0],
                "mel_item": item_no, "mel_category": it["cat"], "interval_days": it["days"], "reason": reason,
                "ordered_days_after_deferral": ordered_after, "hours_to_expiry": hours_left,
                "requested_extension_days": requested, "previous_extension": prior_ext,
                "related_open_item": related_open, "other_open_items": n_open,
                "can_reach_repair_station_before_expiry": can_reach_repair, "part_eta_days": eta_days,
                "authority_notice": notified, "opened": opened, "requester": person(rng),
                "dom": person(rng), "ref": ident(rng, "RIE", 4, "-"), "deferral_ref": ident(rng, "DD", 5, "-")}

    def _related(self, f):
        rel = {"21-51-02": "49-10-01 APU (cat C)", "36-11-03": "21-51-02 pack 1 (cat C)", "30-21-01": "30-11-02 wing anti-ice indication (cat C)",
               "34-41-01": "34-43-01 terrain display (cat C)", "49-10-01": "24-22-01 IDG 2 (cat B)",
               "28-42-01": "28-21-03 fuel pump L aft (cat C)", "32-47-01": "32-42-02 brake wear pin (cat C)",
               "22-11-02": "22-31-01 autothrottle (cat C)", "23-71-01": "31-31-01 FDR data frame (cat A)"}
        return rel.get(f["mel_item"]) if f["related_open_item"] else None

    def _reason(self, f):
        d = f["ordered_days_after_deferral"]
        txt = EXT_REASONS[f["reason"]].format(d=d)
        return txt.replace("0 day(s) after the deferral", "the day of the deferral").replace("1 day(s)", "1 day")

    RULES = [
        "one extension per deferral, categories B, C and D only, no longer than the original interval, and only "
        "for parts or tooling problems outside the operator's control (not planning or staffing, not a part ordered "
        "late, more than 3 days after deferral); DoM approval; authority notified within 24 h of approval",
        "MOE 2.19: a single extension of up to the original interval may be approved by the DoM for B/C/D items "
        "where rectification was prevented by a parts or tooling problem beyond the operator's control (a part "
        "ordered more than 3 days after deferral does not qualify); notify the authority within 24 h",
    ]

    def render(self, f, rng, surface):
        it = MEL[f["mel_item"]]
        expires = f["opened"] + timedelta(days=it["days"])
        rel = self._related(f)
        route_txt = {True: "a station with the part and engineers is on the current routing before expiry",
                     False: "current routing does not reach a repair-capable station before expiry",
                     None: "routing to a repair station not yet checked"}[f["can_reach_repair_station_before_expiry"]]
        limits = EXT_COND[f["mel_item"]]
        eta = show(f["part_eta_days"], rng, lambda v: f"{v} day(s)")
        if surface == "json":
            return dumps(prune({
                "request": {"ref": f["ref"], "requested_by": f["requester"],
                            "extension_requested_days": f["requested_extension_days"], "reason": self._reason(f),
                            "authority_notification": f["authority_notice"].replace("_", " ")},
                "deferral": {"ref": f["deferral_ref"], "aircraft": f"{f['registration']} ({TYPES[f['aircraft_type']][0]})",
                             "mel_item": f"{f['mel_item']} ({EXT_SHORT[f['mel_item']]})", "category": it["cat"],
                             "interval_days": it["days"], "opened_utc": fmt_ts(f["opened"], style="iso"),
                             "hours_to_expiry": f["hours_to_expiry"], "extended_before": f["previous_extension"],
                             "item_conditions": limits},
                "parts": {"eta": eta},
                "programme": {"extension_rule": pick(rng, self.RULES)},
                "aircraft_status": {"other_open_deferrals": f["other_open_items"], "related_open_deferral": rel,
                                    "routing": route_txt},
            }))
        if surface == "extension_form":
            rows = [("Request", f["ref"]), ("Aircraft", f"{f['registration']} {TYPES[f['aircraft_type']][0]}"),
                    ("Deferral", f"{f['deferral_ref']} / {f['mel_item']} {EXT_SHORT[f['mel_item']]}, cat {it['cat']}, {it['days']}-day limit"),
                    ("Deferred on", fmt_date(f["opened"].date(), rng)), ("Expires", fmt_ts(expires, rng)),
                    ("Time remaining", f"{f['hours_to_expiry']} h"), ("Extension requested", f"{f['requested_extension_days']} day(s)"),
                    ("Previously extended", "YES" if f["previous_extension"] else "no"), ("Reason", self._reason(f)),
                    ("Item conditions", limits),
                    ("Part ETA", eta), ("Other open deferrals", f["other_open_items"]),
                    ("Related open deferral", rel or "none"), ("Routing", route_txt),
                    ("Authority notified", f["authority_notice"].replace("_", " ")), ("Requested by", f["requester"])]
            return "REPAIR INTERVAL EXTENSION REQUEST\n" + kv_lines(rows, rng, skip_none=False) + \
                "\n\nProgramme: " + pick(rng, self.RULES) + "\nApproval: ______________ (DoM)"
        body = join_sentences([
            f"{f['dom'].split()[0]}, requesting an extension of {f['requested_extension_days']} day(s) on "
            f"{f['deferral_ref']} ({f['registration']}, {f['mel_item']} {EXT_SHORT[f['mel_item']]}, cat {it['cat']}).",
            f"It expires in {f['hours_to_expiry']} hours.", f"Item conditions while deferred: {limits}.",
            f"Reason: {self._reason(f)}.", f"Part ETA {eta}." if f["part_eta_days"] else "No firm part ETA yet.",
            "This item has already been extended once." if f["previous_extension"] else "",
            f"The aircraft also carries {f['other_open_items']} other open deferral(s)" + (f", including {rel}." if rel else "."),
            f"Planning says {route_txt}.",
            {"not_yet": "Authority not notified yet.", "drafted": "Authority notice drafted, not sent.",
             "sent": "Authority notice already sent."}[f["authority_notice"]],
            f"For reference, our programme: {pick(rng, self.RULES)}.",
        ])
        return "\n".join([f"Subject: Extension request {f['ref']} -- {f['registration']} MEL {f['mel_item']}",
                          f"From: {f['requester']} (MCC)", f"To: {f['dom']} (Director of Maintenance)", "",
                          noisy(body, rng), "", f['requester'].split()[0]])

    def questions(self):
        return {
            "extension_ruling": choice("How should the extension request be answered?", {
                "grant_as_requested": "Grant the extension for the full period requested.",
                "grant_shorter_period": "Grant an extension, but for less time than was asked for.",
                "refer_to_authority": "Decide nothing locally; seek the aviation authority's agreement first.",
                "refuse_extension": "Refuse; the item is fixed by the original expiry or the aircraft stops flying.",
            }),
            "routing_measure": choice("What should happen to the aircraft's flying while the item stays open?", {
                "no_change_to_flying": "Leave the aircraft's planned flying as it is; its current routing already "
                                       "reaches a station that can fix it in time, and no related item is open.",
                "park_until_repaired": "Take the aircraft out of the flying programme until the item is fixed: it "
                                       "cannot have an extension and cannot reach a repair station before expiry.",
                "restrict_sectors": "Keep it flying but away from sectors where the open items' limits bite, such "
                                    "as icing or long overwater legs, because a related system is also deferred.",
                "route_via_repair_station": "Re-plan the rotation so the aircraft passes a station with the part and "
                                            "engineers before the deadline, which the current routing does not do.",
            }),
            "delay_justification": score("How well justified is the failure to fix the defect within the original "
                                         "interval?", [
                "Unjustified: the delay comes from the operator's own planning.",
                "Weak: partly outside the operator's control, but it acted slowly.",
                "Sound: a genuine supply or resource problem, handled promptly.",
                "Compelling: a documented problem entirely outside the operator's control.",
            ]),
            "open_item_burden": score("How much do the aircraft's other open deferrals weigh against extending "
                                      "this one?", [
                "Not at all: none, or only cabin items.",
                "Slightly: a few items on unrelated systems.",
                "Considerably: several items, or one on a related system.",
                "Heavily: related systems are already degraded or near their own limits.",
            ]),
            "extension_eligible": noul(
                "This deferral is eligible for a one-time extension of its rectification interval.",
                yes="The item is category B or C, it has not been extended before, and the hold-up is a parts or "
                    "tooling problem outside the operator's control, with any part ordered promptly.",
                no="The item is category A, it has already been extended once, or the hold-up is the operator's own: "
                   "planning, staffing, or a part ordered late."),
        }


# =========================================================================== schema 5
EVENTS = ("hard_landing", "overweight_landing", "bird_strike", "lightning_strike", "severe_turbulence")


class OccurrenceInspectionCall(Schema):
    name = "occurrence_inspection_call"
    description = "After a hard landing, bird strike or similar event: which conditional inspection, if any, is due."
    surfaces = {"json": 0.5, "crew_report": 0.25, "mcc_worksheet": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        op = pick(rng, OPERATORS)
        typ = _type(rng)
        ev = pick(rng, EVENTS)
        g_thr = {"H-320": 1.75, "H-350": 1.7, "CR-9": 1.8, "T-72": 1.9}[typ]
        crew_level = pick_w(rng, {"mild": 3, "firm": 4, "hard": 3})
        g = round(rng.uniform(1.25, 2.3), 2) if ev in ("hard_landing", "overweight_landing") else None
        turb_g = (round(rng.uniform(1.4, 2.4), 2), round(rng.uniform(-0.8, 0.4), 2)) if ev == "severe_turbulence" else None
        sink = rng.randint(250, 900) if ev in ("hard_landing", "overweight_landing") else None
        lw_pct = round(rng.uniform(88, 104), 1)
        if ev == "overweight_landing":
            lw_pct = round(rng.uniform(100.5, 109), 1)
        data = pick_w(rng, {"full": 4, "partial": 3, "pending": 2, "unavailable": 1})
        findings = pick_w(rng, {"none": 5, "minor": 3, "notable": 1})
        injuries = ev == "severe_turbulence" and chance(rng, 0.4)
        engine_params = pick_w(rng, {"normal": 5, "brief_exceedance": 2, None: 1}) if ev == "bird_strike" else None
        strike_points = pick(rng, ["radome", "windshield", "engine 1 inlet", "left wing leading edge", "nose gear"]) \
            if ev == "bird_strike" else None
        if reg == "clear":
            if chance(rng, 0.5):
                crew_level, findings, data = "mild", "none", "full"
                g = round(g_thr - rng.uniform(0.35, 0.5), 2) if g else None
                sink = rng.randint(200, 330) if sink else None
                engine_params = "normal" if engine_params else None
                if ev == "overweight_landing":
                    lw_pct = round(rng.uniform(100.3, 101.5), 1)
            else:
                crew_level, data = "hard", "full"
                g = round(g_thr + rng.uniform(0.25, 0.5), 2) if g else None
                findings = pick(rng, ["minor", "notable"])
        elif reg == "borderline":
            if g:
                g = near(rng, g_thr, 0.08, lo=1.0)
            if turb_g:
                turb_g = (near(rng, 2.0, 0.15), turb_g[1])
            if ev == "overweight_landing":
                sink = near(rng, 360, 60, lo=150, integer=True)
            data = pick(rng, ["full", "partial"])
        else:
            # the crew and the recorder disagree, or the data is missing when it matters
            if chance(rng, 0.5):
                crew_level = "hard"
                g = round(g_thr - rng.uniform(0.15, 0.4), 2) if g else None
                findings = "none"
            else:
                crew_level = "mild"
                g = round(g_thr + rng.uniform(0.05, 0.3), 2) if g else None
                if turb_g:
                    turb_g = (round(rng.uniform(2.0, 2.4), 2), turb_g[1])
            if chance(rng, 0.4):
                data = pick(rng, ["partial", "pending"])
        f = {"regime": reg, "operator": op, "aircraft_type": typ, "registration": _registration(rng, op)[0],
             "event": ev, "crew_description": crew_level, "recorded_peak_g": g, "g_threshold": g_thr,
             "turbulence_g": turb_g, "sink_rate_fpm": sink, "landing_weight_pct_mlw": lw_pct,
             "flight_data": data, "walkaround_findings": findings, "cabin_injuries": injuries,
             "engine_parameters_after": engine_params, "strike_point": strike_points,
             "station": _station(rng), "next_departure_min": pick(rng, [40, 70, 120, 240, 720]),
             "captain": initials_name(rng), "controller": person(rng), "flight": f"{rng.randint(100, 2999)}",
             "at": rand_ts(rng)}
        # a partial download either thins out the event itself or only loses frames away from it
        f["partial_kind"] = pick_w(rng, {"low_rate_at_event": 1, "gaps_away_from_event": 2})
        return f

    # the operator's mandatory occurrence reporting list, as printed on the worksheet
    MOR_RULE = ("Mandatory occurrence report if: a hard-landing or turbulence trigger value is exceeded or the crew "
                "declares one; an overweight landing above 360 ft/min; a bird strike with damage or an engine "
                "exceedance; a lightning strike with any system effect or damage; any injury")

    def _reportable(self, f):
        ev, crew, find = f["event"], f["crew_description"], f["walkaround_findings"]
        if ev == "hard_landing":
            return crew == "hard" or (f["recorded_peak_g"] or 0) >= f["g_threshold"] or find == "notable"
        if ev == "overweight_landing":
            return (f["sink_rate_fpm"] or 0) > 360 or crew == "hard" or find == "notable"
        if ev == "severe_turbulence":
            tg = f["turbulence_g"]
            return crew == "hard" or f["cabin_injuries"] or tg[0] >= 2.0 or tg[1] <= -0.5
        if ev == "bird_strike":
            return find != "none" or f["engine_parameters_after"] == "brief_exceedance"
        return crew != "mild" or find == "notable"

    def _crew_text(self, f, rng):
        ev, lvl = f["event"], f["crew_description"]
        word = {"mild": pick(rng, ["a little firm", "positive but normal"]), "firm": "firm",
                "hard": pick(rng, ["hard", "very firm, passengers commented"])}[lvl]
        if ev == "hard_landing":
            return (f"Landing {word}. Single touchdown" + (", slight bounce." if lvl == "hard" and chance(rng, 0.5) else ".")
                    + (" Crew declared a hard landing." if lvl == "hard" else " Not declared as a hard landing."))
        if ev == "overweight_landing":
            return (f"Returned to {f['station'][0]} after a pressurisation caution, landed above max landing weight. "
                    f"Touchdown {word}." + (" Crew declared a hard landing." if lvl == "hard" else " Not declared as a hard landing."))
        if ev == "bird_strike":
            return (f"Bird strike on short final, impact heard near the {f['strike_point']}. "
                    + {"mild": "No abnormal indications.", "firm": "Brief smell in the cabin afterwards.",
                       "hard": "Loud bang, smell of burning in the cabin for a few minutes."}[lvl])
        if ev == "lightning_strike":
            return ("Lightning strike in descent at about 9,000 ft. "
                    + {"mild": "Flash, no bang, no indications.", "firm": "Flash and bang, standby compass swung.",
                       "hard": "Loud bang, both displays blanked for 2 s, then normal."}[lvl])
        return ("Turbulence in cruise, " + {"mild": "moderate for about a minute.", "firm": "moderate to severe, trolley moved.",
                                           "hard": "severe, unsecured items hit the ceiling."}[lvl]
                + (" One cabin crew member hurt her wrist." if f["cabin_injuries"] else ""))

    def _data_text(self, f):
        out = []
        if f["recorded_peak_g"] is not None:
            out.append(f"peak vertical acceleration {f['recorded_peak_g']} g")
        if f["sink_rate_fpm"] is not None:
            out.append(f"sink rate at touchdown {f['sink_rate_fpm']} ft/min")
        if f["turbulence_g"]:
            out.append(f"vertical acceleration {f['turbulence_g'][0]:+.2f} / {f['turbulence_g'][1]:+.2f} g")
        if f["event"] in ("hard_landing", "overweight_landing"):
            out.append(f"landing weight {f['landing_weight_pct_mlw']}% of maximum landing weight")
        if f["engine_parameters_after"]:
            out.append({"normal": "engine parameters normal after the strike",
                        "brief_exceedance": "engine 1 vibration exceeded advisory level for 40 s after the strike"}[f["engine_parameters_after"]])
        if f["event"] == "lightning_strike":
            out.append("no recorded parameter measures the strike itself; display and electrical bus logs only")
        return out

    def _manual(self, f):
        ev = f["event"]
        if ev in ("hard_landing", "overweight_landing"):
            s = f"AMM 05-51 trigger: crew report of a hard landing, or peak vertical acceleration of {f['g_threshold']} g or more"
            if ev == "overweight_landing":
                s += "; overweight landing inspection if sink rate exceeded 360 ft/min"
            return s
        if ev == "severe_turbulence":
            return "AMM 05-51 trigger: crew report of severe turbulence, or +2.0 g / -0.5 g exceeded"
        if ev == "bird_strike":
            return "AMM 05-51 trigger: any confirmed bird strike; engine inspection if ingestion suspected"
        return "AMM 05-51 trigger: any known or suspected lightning strike"

    def render(self, f, rng, surface):
        landing = f["event"] in ("hard_landing", "overweight_landing")
        partial = ({"low_rate_at_event": "partial download (touchdown frames at low sample rate)",
                    "gaps_away_from_event": "partial download (touchdown frames complete; gaps earlier in the flight)"}
                   if landing else
                   {"low_rate_at_event": "partial download (gaps around the event)",
                    "gaps_away_from_event": "partial download (event window complete; gaps elsewhere in the flight)"})[f["partial_kind"]]
        data_status = {"full": "full download available",
                       "partial": partial,
                       "pending": "download requested, not yet received", "unavailable": "recorder data not recoverable"}[f["flight_data"]]
        data = self._data_text(f) if f["flight_data"] in ("full", "partial") else []
        find = {"none": "nothing found on walkaround", "minor": pick(rng, ["paint scuffing and a loose access panel screw",
                                                                           "blood smear, no visible dent", "small burn mark near a static wick"]),
                "notable": pick(rng, ["hydraulic fluid weeping at the right main gear", "dent on the radome lower edge",
                                      "two missing static wicks and pitting on the tail cone"])}[f["walkaround_findings"]]
        if f["event"] in ("hard_landing", "overweight_landing", "severe_turbulence") and f["walkaround_findings"] == "minor":
            find = pick(rng, ["paint scuffing and a loose access panel screw", "tyre scrub marks, within limits"])
        stn = f["station"]
        if surface == "json":
            return dumps(prune({
                "event": {"type": f["event"].replace("_", " "), "flight": f["flight"], "aircraft": f"{f['registration']} ({TYPES[f['aircraft_type']][0]})",
                          "station": f"{stn[0]} ({STATION_SHORT[stn[2]]})", "crew_report": self._crew_text(f, rng),
                          "cabin_injuries": f["cabin_injuries"] if f["event"] == "severe_turbulence" else None},
                "flight_data": {"status": data_status, "values": data or None},
                "inspection": {"walkaround": find, "manual_trigger": self._manual(f)},
                "reporting": {"mandatory_occurrence_rule": self.MOR_RULE},
                "schedule": {"next_departure_minutes": f["next_departure_min"]},
            }))
        if surface == "crew_report":
            lines = [f"TECH LOG ENTRY -- flight {f['flight']}, {f['registration']}, Capt. {f['captain']}",
                     noisy(self._crew_text(f, rng), rng), "",
                     f"Engineering ({stn[0]}): {find}.", f"Flight data: {data_status}."]
            if data:
                lines += bullets(data, rng).split("\n")
            lines += ["", self._manual(f) + ".", self.MOR_RULE + ".", f"Aircraft due out again in {f['next_departure_min']} min."]
            return "\n".join(lines)
        rows = [("Controller", f["controller"]), ("Aircraft", f["registration"]), ("Event", f["event"].replace("_", " ")),
                ("Time", fmt_ts(f["at"], rng)), ("Station", stn[0]), ("Crew said", self._crew_text(f, rng)),
                ("Data", data_status)] + [("  -", d) for d in data] + [
                ("Walkaround", find), ("Manual", self._manual(f)), ("Reporting", self.MOR_RULE),
                ("Next departure", f"{f['next_departure_min']} min")]
        return "MCC OCCURRENCE WORKSHEET\n" + kv_lines(rows, rng, style="aligned")

    def questions(self):
        return {
            "inspection_call": choice("Which inspection should maintenance control call for before the next flight?", {
                "data_review_first": "Hold the aircraft while the recorded flight data is analysed, then decide.",
                "phase_one_inspection": "Carry out the first-phase conditional inspection of the areas most likely "
                                        "to be affected.",
                "phase_two_inspection": "Go straight to the extended second-phase inspection, including gear, "
                                        "attachments or engine internals as applicable.",
                "record_only": "Record the event; no conditional inspection is due.",
            }),
            "load_exceedance_likelihood": score("How likely is it that the aircraft took loads beyond what its "
                                                "structure or systems are designed for?", [
                "Remote: nothing points that way.",
                "Low: one indicator is close to its trigger.",
                "Plausible: indicators are at or around the trigger.",
                "Probable: indicators are clearly past the trigger or damage was found.",
            ]),
            "data_reliance": score("How far can the recorded flight data settle what actually happened in this event?", [
                "Not at all: there is no usable data, or none that bears on this kind of event.",
                "Somewhat: partial or low-rate data, or data that bears on the event only indirectly.",
                "Largely: a download covering the key parameters, with minor gaps.",
                "Fully: complete data on exactly the values the inspection trigger is defined by.",
            ]),
            "accounts_agree": noul(
                "The crew's description of the event and the recorded flight data tell the same story.",
                yes="The crew's words and the recorded values point to the same severity.",
                no="They point to different severities, or there is no data to compare with the crew report."),
            "authority_report_needed": noul(
                "This event must be filed as a mandatory occurrence report with the aviation authority.",
                yes="The event meets a condition on the operator's reporting list: a trigger value exceeded or "
                    "declared, damage or an engine exceedance after a strike, a system effect, or an injury.",
                no="None of the listed conditions is met: no exceedance or declaration, no damage, no system effect "
                   "and no injury."),
        }


WORKFLOW = Workflow(
    name="aircraft_defect_deferral",
    title="Aircraft defect deferral",
    description="Whether a fictional operator's aircraft may fly with a known defect or damage, under what limits "
                "and for how long.",
    domain="aviation maintenance control",
    schemas=[TechLogMelDeferral(), RepeatDefectReview(), StructuralDamageFinding(), DeferralExtensionRequest(),
             OccurrenceInspectionCall()],
    notes="Invented fleet, operators, stations, MEL items and manual references; decisions are maintenance-control "
          "rulings on defects, not security alerts, support threads, bills or agent runs.",
)
