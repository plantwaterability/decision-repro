"""Port berth allocation: how a (fictional) harbour authority's berth planners, pilots and harbour master decide
which vessel goes where and when, against the port's own written rules.

Berth requests checked against berth length and depth, deep-draft transits against an under-keel-clearance rule
and the tide, pilotage and towage requirements (exemption certificates, two-pilot cases, night limits), requests
to jump the anchorage queue for priority cargo, and berthing calls under wind, swell and visibility limits. Every
state quotes the extract of the port's rules that decides it, so each case is decidable in principle. Ports,
terminals, berths, vessels, agents and every limit are invented; flag states are real but decide nothing. Freight
exceptions after discharge (trucking delays, damage claims, reefer excursions) and customs live in other
workflows; nothing here is a conversation with a customer.

Each schema draws a `regime` first: 'clear' cases sit well inside or outside the rules, 'borderline' cases put a
length, draft, tide margin, notice period or wind forecast right at a limit, and 'conflicting' cases pair the
numbers with a contrary signal (an agent's optimistic sailing time, a draft survey that disagrees with the
declared draft, two forecasts that disagree). Every schema defines `intended(f)`, the answer the quoted rules
give; it documents the logic and drives the spread check.
"""
from __future__ import annotations

from datetime import timedelta

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
    rand_ts,
    regime,
    score,
    table,
)

# =========================================================================== shared pools and helpers
PORTS = ["Port of Marrowhaven", "Tarrow Estuary Port", "Kelderby Harbour", "Port of Ostwick", "Fennick Bay Port",
         "Harl Sound Terminals", "Port of Westermouth", "Caldmouth Harbour", "Port of Brackwater", "Lunsford Docks",
         "Port of Skelmar", "Innisvale Harbour"]
ADJ = ["Northern", "Silver", "Coral", "Amber", "Polar", "Golden", "Crimson", "Eastern", "Morning", "Iron", "Grand",
       "Winter", "Bright", "Western", "Steady", "Southern", "Jade", "Cobalt", "Quiet", "Swift", "Patient", "Hardy",
       "Emerald", "Scarlet", "Distant", "Rising"]
NOUN = ["Heron", "Meridian", "Tern", "Horizon", "Lantern", "Pioneer", "Harvest", "Spirit", "Venture", "Kestrel",
        "Mariner", "Fjord", "Dawn", "Cedar", "Pelican", "Compass", "Anchor", "Beacon", "Orchid", "Sable", "Trader",
        "Voyager", "Albatross", "Glacier", "Monsoon", "Juniper", "Petrel", "Cormorant"]
FLAGS = ["Panama", "Liberia", "Marshall Islands", "Malta", "Singapore", "Norway", "Netherlands", "Cyprus",
         "Antigua and Barbuda", "Bahamas", "Portugal", "Denmark"]
# vessel type: (label, LOA range m, draft range m, name prefixes)
VTYPES = {
    "container_feeder": ("container feeder", (140, 200), (8.0, 10.5), ["", "MV"]),
    "container_mainline": ("container ship", (255, 366), (12.0, 15.2), ["", "MV"]),
    "bulk": ("bulk carrier", (180, 235), (10.0, 13.5), ["MV"]),
    "tanker": ("product tanker", (150, 250), (9.0, 13.0), ["MT"]),
    "general": ("general cargo ship", (84, 160), (5.5, 9.0), ["MV"]),
    "car_carrier": ("car carrier", (180, 200), (8.5, 9.8), ["MV"]),
    "reefer": ("refrigerated cargo ship", (140, 170), (7.5, 9.5), ["MV"]),
}
TERMINALS = ["North Quay", "Container Terminal 2", "Deepwater Terminal", "Riverside Wharf", "East Basin",
             "Bulk Terminal", "Ro-Ro Terminal", "Liquid Bulk Jetty", "South Mole", "Fruit Terminal", "West Pier",
             "Inner Harbour"]
TERMINALS_BY_TYPE = {
    "container_feeder": ["Container Terminal 2", "North Quay", "East Basin", "Container Terminal 1"],
    "container_mainline": ["Container Terminal 2", "Deepwater Terminal", "Container Terminal 1"],
    "bulk": ["Bulk Terminal", "South Mole", "Deepwater Terminal", "West Pier"],
    "tanker": ["Liquid Bulk Jetty", "Oil Jetty 2", "Deepwater Terminal"],
    "general": ["Riverside Wharf", "Inner Harbour", "West Pier", "East Basin"],
    "car_carrier": ["Ro-Ro Terminal", "East Basin", "North Quay"],
    "reefer": ["Fruit Terminal", "Riverside Wharf", "North Quay"],
}
NIGHT_TERMINALS = ("Deepwater Terminal", "Liquid Bulk Jetty", "Oil Jetty 2", "Container Terminal 2")


def _terminal(rng, vtype, night=None):
    opts = TERMINALS_BY_TYPE[vtype]
    if night is not None:
        opts = [t for t in opts if (t in NIGHT_TERMINALS) == night] or opts
    return pick(rng, opts)


def _agency(rng):
    return f"{pick(rng, ORG_A)} {pick(rng, ['Shipping Agency', 'Maritime Agencies', 'Ship Services', 'Port Agency'])}"


def _vname(rng, vtype):
    pre = pick(rng, VTYPES[vtype][3])
    name = f"{pick(rng, ADJ)} {pick(rng, NOUN)}" if chance(rng, 0.7) else f"{pick(rng, NOUN)} {pick(rng, ['Bay', 'Star', 'Sky', 'Way', 'Point', 'Express'])}"
    return f"{pre} {name}".strip()


def _vessel(rng, vtype=None, weights=None):
    vtype = vtype or pick_w(rng, weights or {"container_feeder": 3, "container_mainline": 3, "bulk": 2, "tanker": 2,
                                             "general": 1, "car_carrier": 1, "reefer": 1})
    lab, (l0, l1), (d0, d1), _ = VTYPES[vtype]
    loa = between(rng, l0, l1, integer=True)
    return {"vtype": vtype, "vessel": _vname(rng, vtype), "flag": pick(rng, FLAGS), "loa_m": loa,
            "beam_m": int(round(loa * rng.uniform(0.135, 0.165))), "draft_m": round(between(rng, d0, d1), 1),
            "imo": f"9{rng.randint(100000, 999999)}"}


def _kv(pairs, rng):
    return kv_lines(pairs, rng, style=pick(rng, ["colon", "colon", "aligned", "dash"]))


def _std(rng, ref, clauses, json_ok=False):
    """The rules extract as prose, numbered clauses or bullets; for the json surface a clause list or a string."""
    style = pick(rng, ["prose", "numbered", "bullets", "prose"])
    if json_ok:
        return list(clauses) if style in ("numbered", "bullets") else f"{ref}: " + " ".join(clauses)
    if style == "numbered":
        return f"{ref}\n" + "\n".join(f"({chr(96 + i)}) {c}" for i, c in enumerate(clauses, 1))
    if style == "bullets":
        return f"{ref}:\n" + bullets(clauses, rng)
    return f"{ref}: " + " ".join(clauses)


def _tfmt(rng):
    """One time style per rendered document."""
    style = pick(rng, ["dmy", "iso", "hm_day", "short"])

    def f(t):
        if style == "iso":
            return t.strftime("%Y-%m-%d %H:%M")
        if style == "hm_day":
            return t.strftime("%H:%M on %d/%m")
        if style == "short":
            return t.strftime("%d %b %H%M")
        return t.strftime("%d.%m %H:%M")
    return f


def _plural(n, one, many=None):
    return f"{n} {one if n == 1 else (many or one + 's')}"


# =========================================================================== schema 1
class BerthAssignment(Schema):
    name = "berth_assignment"
    description = ("A ship's agent requests a berth for an arriving vessel: assign it, use another berth, queue it "
                   "at anchor or refer it, under the port's berthing rules.")
    surfaces = {"json": 0.5, "agent_email": 0.25, "berth_schedule": 0.25}

    LEN_D = {"comfortable": (35, 90), "tight": (20, 34), "disp": (10, 19), "no": (-15, 9)}
    DEP_D = {"comfortable": (1.5, 3.0), "tight": (1.0, 1.4), "disp": (0.6, 0.99), "no": (0.1, 0.59)}

    def sample_factors(self, rng):
        reg = regime(rng)
        v = _vessel(rng)
        target = pick(rng, {"clear": ["comfortable", "comfortable", "no"],
                            "borderline": ["tight", "disp", "disp", "tight", "no"],
                            "conflicting": ["comfortable", "tight", "disp", "no"]}[reg])
        bind = pick(rng, ["length", "depth", "length", "depth", "both"])
        other = pick(rng, ["comfortable", "tight"]) if target not in ("comfortable",) else "comfortable"
        lt = target if bind in ("length", "both") else other
        dt = target if bind in ("depth", "both") else other
        berth_len = int(round((v["loa_m"] + between(rng, *self.LEN_D[lt])) / 5.0) * 5)
        berth_depth = round(v["draft_m"] + between(rng, *self.DEP_D[dt]), 1)
        eta_h = between(rng, 5, 30, integer=True)
        # occupant of the requested berth: finish time relative to the 2 h availability allowance
        occupant = None
        if chance(rng, 0.85):
            if reg == "borderline":
                off = near(rng, 0, 1.5)
            elif reg == "clear":
                off = pick(rng, [between(rng, -8, -2), between(rng, -8, -2), between(rng, 2, 8)])
            else:
                off = pick(rng, [between(rng, -6, -0.5), between(rng, 0.5, 7)])
            finish_h = max(1.0, eta_h + 1 + off)
            unit = ("moves" if v["vtype"].startswith("container") else "vehicles" if v["vtype"] == "car_carrier"
                    else "pallets" if v["vtype"] == "reefer" else "t")
            rate = {"moves": pick(rng, [55, 70, 85, 110, 140]), "vehicles": pick(rng, [150, 200, 260]),
                    "pallets": pick(rng, [180, 240, 300]), "t": pick(rng, [600, 900, 1200, 1800])}[unit]
            occupant = {"name": _vname(rng, v["vtype"]), "unit": unit, "rate": rate,
                        "remaining": int(round(finish_h * rate, -1 if unit != "t" else -2)),
                        "declared_etd_h": max(2, eta_h + pick(rng, [-4, -3, -2, -1, 0, 1]))}
        liner = v["vtype"].startswith("container") or v["vtype"] == "car_carrier"
        has_window = chance(rng, 0.8 if liner else 0.15)
        win_off = None
        if has_window:
            win_off = (near(rng, pick(rng, [-6, 6]), 1.5, digits=1) if reg == "borderline"
                       else pick(rng, [between(rng, -3, 3, digits=1), between(rng, 7, 14, digits=1),
                                       between(rng, -12, -7, digits=1), between(rng, -2, 2, digits=1)]))
        notice = near(rng, 24, 6, lo=6, integer=True) if reg == "borderline" and chance(rng, 0.4) else pick(rng, [12, 24, 30, 36, 48, 48, 72, 96])
        alt = None
        req_term = _terminal(rng, v["vtype"])
        req_no = rng.randint(1, 12)
        if chance(rng, 0.6):
            at = pick(rng, ["comfortable", "comfortable", "tight", "disp", "no"])
            alt = {"name": f"{_terminal(rng, v['vtype'])} berth {pick(rng, [n for n in range(1, 14) if n != req_no])}",
                   "length_m": int(round((v["loa_m"] + between(rng, *self.LEN_D[at])) / 5.0) * 5),
                   "depth_m": round(v["draft_m"] + between(rng, *self.DEP_D[pick(rng, ['comfortable', 'tight', at])]), 1),
                   "free_from_h": pick(rng, [0, 0, max(0, eta_h - 3), eta_h + 1, eta_h + 5, eta_h + 12])}
        reefers = 0
        if v["vtype"].startswith("container"):
            reefers = pick(rng, [0, 20, 60, 120, 160, 240, 420, 560])
        perish_h = between(rng, 30, 160, integer=True) if v["vtype"] == "reefer" else None
        onward = between(rng, 6, 60, integer=True) if liner else None
        demurrage = (pick(rng, [12000, 18000, 26000, 38000, 45000]) if v["vtype"] in ("bulk", "tanker") else None)
        return {"regime": reg, "port": pick(rng, PORTS), **v, "eta_h": eta_h,
                "requested_berth": f"{req_term} berth {req_no}", "berth_length_m": berth_len,
                "berth_depth_m": berth_depth, "occupant": occupant, "has_window": has_window,
                "window_offset_h": win_off, "notice_h": notice,
                "competitor_waiting": chance(rng, {"clear": 0.4, "borderline": 0.5, "conflicting": 0.65}[reg]),
                "competitor": _vname(rng, pick(rng, list(VTYPES))), "alternative": alt, "reefer_units": reefers,
                "perishable_deadline_h": perish_h, "onward_slack_h": onward, "demurrage_usd_day": demurrage,
                "agent": _agency(rng), "agent_name": person(rng), "planner": person(rng),
                "now": rand_ts(rng, hours=(6, 20)), "agent_view": pick(rng, [None, None, "occupant_early", "will_fit",
                                                                            "urgent"])}

    # ------------------------------------------------------------------ the rules' logic
    @staticmethod
    def _len_status(loa, berth):
        return "fits" if loa + 20 <= berth else "disp" if loa + 10 <= berth else "no"

    @staticmethod
    def _dep_status(draft, depth):
        return "fits" if draft <= round(depth - 1.0, 2) else "disp" if draft <= round(depth - 0.6, 2) else "no"

    def _overstay(self, f):
        o = f["occupant"]
        if not o:
            return False
        return o["remaining"] / o["rate"] + 1 > f["eta_h"] + 2 + 1e-9

    def _window_holds(self, f):
        return bool(f["has_window"] and abs(f["window_offset_h"]) <= 6 and f["notice_h"] >= 24)

    def _alt_ok(self, f):
        a = f["alternative"]
        return bool(a and self._len_status(f["loa_m"], a["length_m"]) == "fits"
                    and self._dep_status(f["draft_m"], a["depth_m"]) == "fits" and a["free_from_h"] <= f["eta_h"] + 2)

    def intended(self, f):
        ls, ds = self._len_status(f["loa_m"], f["berth_length_m"]), self._dep_status(f["draft_m"], f["berth_depth_m"])
        rank = {"fits": 0, "disp": 1, "no": 2}
        fit = max(ls, ds, key=rank.get)
        over = self._overstay(f)
        win = self._window_holds(f)
        behind = f["competitor_waiting"] and not win
        alt = self._alt_ok(f)
        if fit == "no":
            plan = "assign_other_berth" if alt else "queue_at_anchorage"
        elif fit == "disp":
            plan = "assign_other_berth" if alt else "refer_to_harbour_master"
        elif over or behind:
            plan = "assign_other_berth" if alt else "queue_at_anchorage"
        else:
            plan = "assign_as_requested"
        if fit != "fits":
            constraint = "berth_length" if rank[ls] >= rank[ds] else "berth_depth"
        elif over:
            constraint = "occupant_overstay"
        elif behind:
            constraint = "earlier_arrival_ahead"
        else:
            constraint = "no_binding_constraint"
        if fit == "no":
            margin = 0
        elif fit == "disp":
            margin = 1
        else:
            spare_len = f["berth_length_m"] - (f["loa_m"] + 20)
            spare_dep = (f["berth_depth_m"] - 1.0) - f["draft_m"]
            margin = 2 if spare_len < 15 or spare_dep < 0.4 else 3
        pts = 0
        r = f["reefer_units"]
        pts += 2 if r >= 400 else 1 if r >= 150 else 0
        if f["perishable_deadline_h"] is not None:
            pts += 3 if f["perishable_deadline_h"] <= 48 else 2 if f["perishable_deadline_h"] <= 96 else 1
        if f["onward_slack_h"] is not None:
            pts += 2 if f["onward_slack_h"] < 12 else 1 if f["onward_slack_h"] < 24 else 0
        if f["demurrage_usd_day"] is not None and f["demurrage_usd_day"] >= 35000:
            pts += 1
        return {"berth_plan": plan, "main_constraint": constraint, "berth_fit_margin": margin,
                "alongside_pressure": min(3, pts), "window_priority_holds": "yes" if win else "no"}

    RULES = [
        "Length: a berth may take a vessel whose LOA plus 20 m does not exceed the berth length; down to LOA plus "
        "10 m only with a harbour master's dispensation.",
        "Depth: maximum arrival draft is the berth's dredged depth less 1.0 m; up to depth less 0.6 m only with a "
        "harbour master's dispensation.",
        "A berth counts as available if its occupant can finish cargo work and sail within 2 h of the arriving "
        "vessel's ETA at the pilot station, judged on the cargo remaining and the current working rate (allow 1 h "
        "to unberth after cargo work ends).",
        "Window priority: a liner vessel keeps priority for its agreed berthing window if it arrives within 6 h of "
        "the window start and gave ETA notice at least 24 h ahead; otherwise berths go in order of arrival at the "
        "pilot station.",
        "Where the requested berth is unavailable or unsuitable, the planner may allocate another berth that meets "
        "these rules; any allocation needing a dispensation goes to the harbour master.",
    ]

    # ------------------------------------------------------------------ rendering
    def _occ_text(self, f, tf):
        o = f["occupant"]
        if not o:
            return "berth empty"
        unit = o["unit"]
        rate = (f"{o['rate']} moves/h (all cranes)" if unit == "moves" else f"{o['rate']} {unit}/h"
                if unit != "t" else f"{o['rate']} t/h")
        etd = f["now"] + timedelta(hours=o["declared_etd_h"])
        return (f"{o['name']} alongside; {o['remaining']:,} {unit} remaining at {rate}; agent's declared ETD "
                f"{tf(etd)}")

    def _alt_text(self, f):
        a = f["alternative"]
        if not a:
            return None
        free = "free now" if a["free_from_h"] == 0 else f"free in about {a['free_from_h']} h"
        return f"{a['name']}: {a['length_m']} m long, {a['depth_m']} m dredged depth, {free}"

    def _window_text(self, f, tf):
        if not f["has_window"]:
            return "no agreed berthing window for this call"
        return f"agreed window starts {tf(f['now'] + timedelta(hours=f['eta_h'] - f['window_offset_h']))}"

    def _cargo_text(self, f):
        bits = []
        if f["reefer_units"]:
            bits.append(f"{f['reefer_units']} live reefer units on board")
        if f["perishable_deadline_h"] is not None:
            bits.append(f"chilled fruit; shipper's shelf-life deadline in {f['perishable_deadline_h']} h")
        if f["onward_slack_h"] is not None:
            bits.append(f"next port window {f['onward_slack_h']} h after planned sailing")
        if f["demurrage_usd_day"] is not None:
            bits.append(f"demurrage {money(f['demurrage_usd_day'], style='sym')}/day")
        return "; ".join(bits) or "general cargo, no deadline declared"

    def _view(self, f):
        return {"occupant_early": "The terminal tells us the vessel on the berth will be gone well before we arrive.",
                "will_fit": "Master is confident she will fit the berth without any problem.",
                "urgent": "Charterers are pressing hard for an immediate berth."}.get(f["agent_view"])

    def render(self, f, rng, surface):
        tf = _tfmt(rng)
        eta = f["now"] + timedelta(hours=f["eta_h"])
        comp = (f"{f['competitor']} has been at anchor since yesterday waiting for the same berth (no window)"
                if f["competitor_waiting"] else None)
        if surface == "json":
            return dumps(prune({
                "request": {"vessel": f["vessel"], "type": VTYPES[f["vtype"]][0], "flag": f["flag"], "imo": f["imo"],
                            "loa_m": f["loa_m"], "beam_m": f["beam_m"], "arrival_draft_m": f["draft_m"],
                            "eta_pilot_station": tf(eta), "eta_notice_hours": f["notice_h"],
                            "berth_requested": f["requested_berth"], "window": self._window_text(f, tf),
                            "cargo": self._cargo_text(f), "agent": f["agent"], "agent_remark": self._view(f)},
                "requested_berth": {"length_m": f["berth_length_m"], "dredged_depth_m": f["berth_depth_m"],
                                    "occupancy": self._occ_text(f, tf)},
                "anchorage": {"waiting_for_same_berth": comp},
                "other_berth_offered": self._alt_text(f),
                "planning": {"port": f["port"], "issued": tf(f["now"]), "rules": _std(rng, "Berthing rules", self.RULES, True)},
            }))
        if surface == "agent_email":
            body = [f"Please find below our berthing request for {f['vessel']} ({f['flag']}, IMO {f['imo']}), "
                    f"{VTYPES[f['vtype']][0]}, LOA {f['loa_m']} m, beam {f['beam_m']} m, arrival draft {f['draft_m']} m.",
                    f"ETA pilot station {tf(eta)} (this notice sent {f['notice_h']} h ahead).",
                    f"We request {f['requested_berth']}; {self._window_text(f, tf)}.",
                    f"Cargo: {self._cargo_text(f)}.", self._view(f) or ""]
            plan = _kv([("Requested berth", f"{f['requested_berth']}, {f['berth_length_m']} m, dredged {f['berth_depth_m']} m"),
                        ("Current occupancy", self._occ_text(f, tf)),
                        ("Anchorage", comp or "no vessel waiting for this berth"),
                        ("Other berth", self._alt_text(f) or "none suitable listed")], rng)
            return "\n".join([f"Subject: Berth request - {f['vessel']} ETA {tf(eta)}",
                              f"From: {f['agent_name']} <ops@{f['agent'].split()[0].lower()}-agency.example>",
                              f"To: berth planning, {f['port']}", "", "Dear planner,", "",
                              noisy(join_sentences(body), rng), "", f"Best regards,\n{f['agent_name']}\n{f['agent']}",
                              "", "--- planner's notes ---", plan, "", _std(rng, "Berthing rules", self.RULES)])
        rows = [[f["requested_berth"], f"{f['berth_length_m']} m", f"{f['berth_depth_m']} m",
                 self._occ_text(f, tf) if f["occupant"] else "empty"]]
        if f["alternative"]:
            a = f["alternative"]
            rows.append([a["name"], f"{a['length_m']} m", f"{a['depth_m']} m",
                         "free" if a["free_from_h"] == 0 else f"free in ~{a['free_from_h']} h"])
        head = (f"{f['port'].upper()} - BERTH PLAN, issued {tf(f['now'])} by {f['planner']}\n"
                f"Arrival: {f['vessel']} ({VTYPES[f['vtype']][0]}), LOA {f['loa_m']} m, draft {f['draft_m']} m, "
                f"ETA pilot station {tf(eta)}, notice {f['notice_h']} h, {self._window_text(f, tf)}. "
                f"Cargo: {self._cargo_text(f)}.")
        tail = [f"Anchorage queue: {comp}." if comp else "Anchorage queue: nobody waiting for this berth."]
        if self._view(f):
            tail.append(f"Agent says: {self._view(f)}")
        return "\n".join([head, "", table(["berth", "length", "depth", "occupancy"], rows, rng), "", *tail, "",
                          _std(rng, "Berthing rules", self.RULES)])

    def questions(self):
        return {
            "berth_plan": choice("How should the berth planner handle this arrival?", {
                "assign_as_requested": "Allocate the requested berth for the vessel's arrival.",
                "assign_other_berth": "Allocate a different berth that meets the rules and will be free in time.",
                "queue_at_anchorage": "Hold the vessel at anchor until a suitable berth becomes available.",
                "refer_to_harbour_master": "Send the request to the harbour master, because it needs a dispensation.",
            }),
            "main_constraint": choice("What, if anything, stands in the way of berthing the vessel where it asked?", {
                "berth_length": "The berth is too short for the vessel under the length rule.",
                "berth_depth": "The berth is too shallow for the vessel's arrival draft under the depth rule.",
                "occupant_overstay": "The vessel now on the berth will not have finished and sailed in time.",
                "earlier_arrival_ahead": "Another vessel that reached the pilot station first has the prior claim.",
                "no_binding_constraint": "Nothing: the berth suits the vessel and is free when it arrives.",
            }),
            "berth_fit_margin": score("How comfortably does the vessel fit the berth it asked for?", [
                "It does not fit within the rules, even with a dispensation.",
                "Only with a harbour master's dispensation on length or draft.",
                "Within the rules, but with little to spare on length or depth.",
                "Comfortably within the rules on both length and depth.",
            ]),
            "alongside_pressure": score("How pressing is it to get this vessel alongside without delay?", [
                "Little pressure: a wait costs time and not much else.",
                "Some: a delay costs money but breaks no deadline.",
                "High: a delay threatens an onward connection or a cargo deadline.",
                "Acute: perishable or temperature-controlled cargo or a fixed schedule is at risk within hours.",
            ]),
            "window_priority_holds": noul(
                "The vessel keeps its berthing-window priority under the port's rules.",
                yes="It has an agreed window, arrives within the allowed hours of it, and gave the required notice.",
                no="It has no window, arrives too far from it, or gave too little notice, so arrival order applies."),
        }


# =========================================================================== schema 2
CHANNELS = ["Outer Bar Channel", "Long Reach", "Fairway Approach", "Sandhead Cut", "North Channel", "Main Fairway",
            "Inner Reach"]


class TidalDraftWindow(Schema):
    name = "tidal_draft_window"
    description = ("A deep-draft vessel's inbound transit against the channel's under-keel-clearance rule and the "
                   "tidal window: go now, go under conditions, wait for the next tide, or change draft first.")
    surfaces = {"json": 0.5, "vts_log": 0.25, "passage_form": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        vtype = pick(rng, ["container_mainline", "bulk", "tanker"])
        v = _vessel(rng, vtype)
        v["vessel"] = v["vessel"]
        depth = round(between(rng, 11.5, 14.5), 1)
        tide = round(between(rng, 2.2, 5.0), 1)
        surge = pick_w(rng, {0.0: 4, 0.1: 2, 0.2: 2, 0.3: 1, 0.4: 1})
        swell = round(pick(rng, [between(rng, 0.4, 1.8), between(rng, 0.4, 1.8), near(rng, 2.0, 0.4)]), 1)
        if reg == "clear":
            margin = pick(rng, [between(rng, 0.9, 1.5), between(rng, 0.9, 1.5), between(rng, -0.9, -0.4)])
        elif reg == "borderline":
            margin = near(rng, pick(rng, [-0.1, 0.0, 0.15, 0.3]), 0.2)
        else:
            margin = near(rng, 0.3, 0.35)
        add = 0.5 if swell > 2.0 else 0.0
        declared = round(((depth + tide - surge - margin - 0.5 - add) / 1.10) * 20) / 20
        survey = None
        if chance(rng, 0.72):
            diff = {"clear": pick(rng, [0.0, 0.0, 0.05, -0.05]), "borderline": pick(rng, [0.0, 0.05, 0.1, -0.05]),
                    "conflicting": pick(rng, [0.15, 0.2, 0.25, 0.35, 0.0])}[reg]
            survey = round(declared + diff, 2)
        transit_h = pick(rng, [2.0, 2.5, 2.5, 3.0])
        if reg == "borderline" and chance(rng, 0.5):
            hours_left = round(near(rng, transit_h + 0.5, 1.0, lo=0.5), 1)
        elif reg == "clear":
            hours_left = pick(rng, [4.5, 5.0, 5.5, 6.0, 1.5])
        else:
            hours_left = round(between(rng, 1.0, 6.0), 1)
        eta = rand_ts(rng, hours=pick(rng, [(5, 14), (5, 14), (0, 23)]))
        sunset = pick(rng, [17, 18, 19, 20, 21])
        return {"regime": reg, "port": pick(rng, PORTS), **v, "draft_m": declared, "declared_draft_m": declared,
                "surveyed_draft_m": survey, "channel": pick(rng, CHANNELS), "charted_depth_m": depth,
                "tide_at_transit_m": tide, "neg_surge_m": surge, "swell_m": swell,
                "next_tide_delta_m": pick(rng, [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]), "transit_h": transit_h,
                "window_hours_left": hours_left, "eta": eta, "sunset_h": sunset, "sunrise_h": sunset - 13,
                "trim_note": pick(rng, [None, None, "trimmed 0.6 m by the stern", "on even keel", "slight hog reported"]),
                "pilot": person(rng), "vts_officer": person(rng), "master_note": pick(rng, [None, "insists", "flexible"])}

    # ------------------------------------------------------------------ the rule's logic
    @staticmethod
    def _req(draft, swell):
        return draft * 1.10 + 0.5 + (0.5 if swell > 2.0 else 0.0)

    def _avail(self, f, delta=0.0):
        return f["charted_depth_m"] + f["tide_at_transit_m"] + delta - f["neg_surge_m"]

    def _eff(self, f):
        s = f["surveyed_draft_m"]
        return max(f["declared_draft_m"], s) if s is not None else f["declared_draft_m"]

    def _night(self, f):
        start = f["eta"].hour + f["eta"].minute / 60 + 0.5
        end = start + f["transit_h"]
        return start < f["sunrise_h"] or end > f["sunset_h"]

    def intended(self, f):
        m_now = round(self._avail(f) - self._req(self._eff(f), f["swell_m"]), 3)
        m_decl = self._avail(f) - self._req(f["declared_draft_m"], f["swell_m"])
        m_next = round(self._avail(f, f["next_tide_delta_m"]) - self._req(self._eff(f), f["swell_m"]), 3)
        slack = f["window_hours_left"] - (0.5 + f["transit_h"])
        if f["surveyed_draft_m"] is None and abs(m_decl) < 0.3:
            dec = "verify_draft_first"
        elif slack < 0 or m_now < 0 or (m_now < 0.3 and self._night(f)):
            dec = "wait_next_tide" if m_next >= 0 else "lighten_or_retrim"
        elif m_now < 0.3:
            dec = "transit_with_conditions"
        else:
            dec = "transit_this_tide"
        ukc = 0 if m_now < 0 else 1 if m_now < 0.3 else 2 if m_now < 0.8 else 3
        wp = 0 if slack >= 3 else 1 if slack >= 1.5 else 2 if slack >= 0 else 3
        s = f["surveyed_draft_m"]
        reliable = s is not None and abs(s - f["declared_draft_m"]) <= 0.10 + 1e-9
        return {"transit_decision": dec, "ukc_margin": ukc, "window_pressure": wp,
                "meets_ukc_rule": "yes" if m_now >= 0 else "no", "draft_figure_reliable": "yes" if reliable else "no"}

    RULES = [
        "Minimum water in the channel = static draft plus 10% of draft, plus 0.5 m squat allowance, plus a further "
        "0.5 m when swell at the entrance exceeds 2.0 m.",
        "Available water = charted depth + predicted tide at the planned transit time - forecast negative surge.",
        "Plan on the greater of the declared draft and the latest draft survey; where no survey is on record and the "
        "declared draft leaves less than 0.3 m either side of the minimum, the pilot launch reads the draft marks "
        "before any transit decision.",
        "Where the minimum is met by less than 0.3 m: daylight transit only, 8 kn maximum, second pilot on board.",
        "The transit (pilot boarding 0.5 h plus passage time) must fit inside the tidal window; otherwise plan the "
        "next suitable tide, and if no tide in the next 24 h meets the minimum the vessel must lighten or re-trim.",
    ]

    def _night_txt(self, f):
        return f"sunrise {f['sunrise_h']:02d}:00, sunset {f['sunset_h']:02d}:00"

    def render(self, f, rng, surface):
        tf = _tfmt(rng)
        close = f["eta"] + timedelta(hours=f["window_hours_left"])
        surv = (f"{f['surveyed_draft_m']:.2f} m (pilot launch draft read)" if f["surveyed_draft_m"] is not None
                else None)
        next_t = (f"next high water about 12.4 h later, predicted "
                  f"{'+' if f['next_tide_delta_m'] >= 0 else ''}{f['next_tide_delta_m']:.1f} m against this tide")
        master = {"insists": "Master states the vessel is well within limits and wishes to proceed.",
                  "flexible": "Master content to wait if required."}.get(f["master_note"])
        if surface == "json":
            return dumps(prune({
                "vessel": {"name": f["vessel"], "type": VTYPES[f["vtype"]][0], "loa_m": f["loa_m"],
                           "declared_draft_m": f["declared_draft_m"], "draft_survey": surv, "trim": f["trim_note"],
                           "eta_pilot_station": tf(f["eta"]), "master_remark": master},
                "channel": {"name": f["channel"], "charted_depth_m": f["charted_depth_m"],
                            "passage_time_h": f["transit_h"]},
                "tide": {"predicted_at_transit_m": f["tide_at_transit_m"], "negative_surge_forecast_m": f["neg_surge_m"],
                         "window_closes": tf(close), "next_tide": next_t},
                "conditions": {"swell_at_entrance_m": f["swell_m"], "daylight": self._night_txt(f)},
                "rules": _std(rng, f"{f['port']} under-keel clearance rules", self.RULES, True),
            }))
        if surface == "vts_log":
            t = f["eta"] - timedelta(hours=rng.randint(2, 5))
            lines = [f"{f['port']} VTS log - {f['channel']}"]
            for gap, text in [
                (0, f"{f['vessel']} ({VTYPES[f['vtype']][0]}, LOA {f['loa_m']} m) reports declared draft "
                    f"{f['declared_draft_m']:.2f} m, ETA pilot station {tf(f['eta'])}."),
                (rng.randint(5, 30), f"Draft survey: {surv}." if surv else "No draft survey on record for this call."),
                (rng.randint(5, 20), f"Tide office: {f['tide_at_transit_m']} m predicted at planned transit, negative "
                                     f"surge {f['neg_surge_m']} m forecast, window for declared draft closes {tf(close)}; "
                                     f"{next_t}."),
                (rng.randint(5, 20), f"Swell at entrance {f['swell_m']} m. Charted depth {f['charted_depth_m']} m. "
                                     f"Passage {f['transit_h']} h. {self._night_txt(f)}."),
            ] + ([(rng.randint(3, 15), master)] if master else []) + (
                    [(rng.randint(3, 15), f"Master reports vessel {f['trim_note']}.")] if f["trim_note"] else []):
                t += timedelta(minutes=gap)
                lines.append(f"{t.strftime('%H:%M')}  {noisy(text, rng, 0.05)}")
            return "\n".join(lines) + "\n\n" + _std(rng, "UKC rules", self.RULES)
        rows = _kv([("Vessel", f"{f['vessel']} ({f['flag']})"), ("Type / LOA", f"{VTYPES[f['vtype']][0]} / {f['loa_m']} m"),
                    ("Declared draft", f"{f['declared_draft_m']:.2f} m"), ("Draft survey", surv or "none"),
                    ("Trim", f["trim_note"] or "not reported"), ("ETA pilot station", tf(f["eta"])),
                    ("Channel / charted depth", f"{f['channel']} / {f['charted_depth_m']} m"),
                    ("Predicted tide at transit", f"{f['tide_at_transit_m']} m"),
                    ("Negative surge forecast", f"{f['neg_surge_m']} m"), ("Swell at entrance", f"{f['swell_m']} m"),
                    ("Passage time", f"{f['transit_h']} h"), ("Tidal window closes", tf(close)),
                    ("Next tide", next_t), ("Daylight", self._night_txt(f)), ("Master", master)], rng)
        return f"DEEP-DRAFT PASSAGE PLAN - {f['port']}\nPilot: {f['pilot']}\n{rows}\n\n{_std(rng, 'UKC rules', self.RULES)}"

    def questions(self):
        return {
            "transit_decision": choice("What should be decided about this vessel's inbound transit?", {
                "transit_this_tide": "Bring the vessel in on this tide as planned.",
                "transit_with_conditions": "Bring it in on this tide under the rules' conditions: daylight, reduced speed and a second pilot.",
                "wait_next_tide": "Hold it at anchor and plan the transit on the next suitable tide.",
                "lighten_or_retrim": "Require it to reduce or re-trim its draft before any transit.",
                "verify_draft_first": "Have the draft marks read independently before deciding anything.",
            }),
            "ukc_margin": score("How much margin over the port's minimum water does the vessel have at the planned time?", [
                "None: the minimum is not met.",
                "Thin: met by less than 0.3 m.",
                "Adequate: between 0.3 m and 0.8 m to spare.",
                "Ample: more than 0.8 m to spare.",
            ]),
            "window_pressure": score("How tight is the time left in this tide's window for boarding and passage?", [
                "No pressure: three hours or more to spare.",
                "Some: an hour or two of slack.",
                "Tight: under 1.5 hours of slack after boarding and passage.",
                "Gone: boarding and passage no longer fit in this tide's window.",
            ]),
            "meets_ukc_rule": noul(
                "At the planned transit time the vessel meets the port's minimum-water rule.",
                yes="Available water, after any negative surge, covers draft plus all the rule's allowances on the draft the rule says to use.",
                no="Available water falls short of draft plus the allowances, on the draft the rule says to use."),
            "draft_figure_reliable": noul(
                "The declared draft can be relied on for planning this transit."),
        }


# =========================================================================== schema 3
class PilotageTugRequirement(Schema):
    name = "pilotage_tug_requirement"
    description = ("A planned vessel movement checked against compulsory pilotage, the exemption certificate, the "
                   "two-pilot and night rules and the towage table.")
    surfaces = {"json": 0.5, "pilot_booking": 0.25, "marine_ops_chat": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        # scenario: a big ship or laden tanker near the night limit, a master with an exemption, or neither
        scen = pick_w(rng, {"night": 4, "pec": 4, "plain": 3})
        if scen == "night":
            vtype = pick(rng, ["container_mainline", "container_mainline", "tanker"])
        elif scen == "pec":
            vtype = pick(rng, ["container_feeder", "general", "general", "reefer", "car_carrier", "bulk", "tanker"])
        else:
            vtype = pick(rng, list(VTYPES))
        v = _vessel(rng, vtype)
        if reg == "borderline" and chance(rng, 0.4):
            v["loa_m"] = near(rng, pick(rng, [250, 300, 200, 120]), 6, integer=True)
        laden = vtype == "tanker" and chance(rng, 0.7 if scen == "night" else 0.4)
        sunset = pick(rng, [17, 18, 19, 20, 21])
        pec = None
        if scen == "pec" or chance(rng, 0.25):
            lim = pick(rng, [x for x in (150, 200, 250, 300) if x >= v["loa_m"]] or [300])
            if reg != "clear" and chance(rng, 0.4):
                lim = pick(rng, [x for x in (150, 200, 250, 300) if abs(x - v["loa_m"]) <= 50] or [lim])
            pec = {"loa_limit": lim, "trips_12m": (near(rng, 6, 2, lo=0, integer=True) if reg == "borderline"
                                                   else pick(rng, [3, 8, 9, 12, 14, 20])),
                   "daylight_only": chance(rng, 0.4)}
        if scen == "night" and reg != "clear":
            hour = pick(rng, [sunset - 2, sunset - 1, sunset, sunset + 2, sunset - 14, 2, 4])
        elif scen == "night":
            hour = pick(rng, [1, 3, 23, 11, 13])
        else:
            hour = between(rng, 0, 23, integer=True)
        hour %= 24
        terminal = _terminal(rng, vtype, night=True if scen == "night" else None)
        wind = (near(rng, 20, 4, lo=3, integer=True) if reg == "borderline" and chance(rng, 0.5)
                else between(rng, 4, 30, integer=True))
        return {"regime": reg, "port": pick(rng, PORTS), **v, "laden_tanker": laden,
                "bow_thruster": chance(rng, 0.7 if vtype != "bulk" else 0.35),
                "movement": pick(rng, ["arrival", "arrival", "departure", "shift"]), "terminal": terminal,
                "night_terminal": terminal in NIGHT_TERMINALS,
                "hour": hour, "minute": pick(rng, [0, 15, 30, 45]), "sunset_h": sunset, "sunrise_h": sunset - 13,
                "master_pec": pec, "master": person(rng), "wind_kn": wind,
                "current": pick(rng, ["slack", "slack", "flood 1.5 kn", "ebb 1.0 kn", "ebb 2.5 kn", "spring ebb 3.5 kn"]),
                "dead_ship": chance(rng, 0.05), "senior_pilots_free": pick_w(rng, {0: 2, 1: 3, 2: 4}),
                "agent": _agency(rng), "booking": ident(rng, "PB", 6, "-"),
                "date": rand_ts(rng).date(), "duty_pilot": person(rng)}

    # ------------------------------------------------------------------ the rules' logic
    def _night(self, f):
        h = f["hour"] + f["minute"] / 60
        return h < f["sunrise_h"] or h + 1.5 > f["sunset_h"]

    def _needs_two(self, f):
        return f["loa_m"] >= 300 or f["draft_m"] >= 14.0 or f["dead_ship"]

    def _night_restricted(self, f):
        return self._night(f) and f["night_terminal"] and (f["loa_m"] > 250 or f["laden_tanker"])

    def _pec_ok(self, f):
        p = f["master_pec"]
        return bool(p and f["loa_m"] <= p["loa_limit"] and p["trips_12m"] >= 6 and not f["laden_tanker"]
                    and not (p["daylight_only"] and self._night(f)) and not self._needs_two(f))

    def _tugs(self, f):
        loa = f["loa_m"]
        n = 0 if loa < 120 else 1 if loa < 200 else 2 if loa < 280 else 3
        if not f["bow_thruster"] and loa >= 100:
            n += 1
        if f["wind_kn"] > 20:
            n += 1
        if f["laden_tanker"] and loa >= 150:
            n = max(n, 2)
        if f["dead_ship"]:
            n = max(n, 3)
        return min(4, n)

    def intended(self, f):
        if self._night_restricted(f):
            ruling = "reschedule_movement"
        elif self._needs_two(f):
            ruling = "two_pilots" if f["senior_pilots_free"] >= 1 else "reschedule_movement"
        elif self._pec_ok(f):
            ruling = "pec_exemption_accepted"
        else:
            ruling = "one_pilot"
        pts = (f["loa_m"] >= 280) + (2 if f["wind_kn"] > 28 else 1 if f["wind_kn"] > 20 else 0)
        pts += (not f["bow_thruster"]) + self._night(f) + ("3.5" in f["current"] or "2.5" in f["current"])
        pts += 2 * f["dead_ship"]
        diff = 0 if pts == 0 else 1 if pts == 1 else 2 if pts <= 3 else 3
        return {"pilotage_ruling": ruling, "towage_level": self._tugs(f), "handling_difficulty": diff,
                "pec_covers_movement": "yes" if self._pec_ok(f) else "no",
                "night_restriction_applies": "yes" if self._night_restricted(f) else "no"}

    def _rules(self, f):
        terms = ("this terminal is one of the berths covered" if f["night_terminal"]
                 else "this terminal is not one of the berths covered")
        return [
            "Pilotage is compulsory for vessels of 80 m LOA or more and for all laden tankers.",
            "A master holding this port's pilotage exemption certificate (PEC) may con the vessel without a pilot "
            "only if the vessel is within the certificate's LOA limit, the master has made at least 6 movements in "
            "the port in the last 12 months, the vessel is not a laden tanker, a daylight-only certificate is used "
            "between sunrise and sunset, and the movement does not need two pilots.",
            "Two pilots (one of them Class 1) are required for vessels of 300 m LOA or more, drafts of 14.0 m or "
            "more, and dead-ship tows; if no Class 1 pilot is free the movement is rescheduled.",
            f"Vessels over 250 m LOA and laden tankers move at the night-restricted berths only between sunrise and "
            f"sunset, allowing 1.5 h for the movement ({terms}).",
            "Tugs: under 120 m none; 120-199 m one; 200-279 m two; 280 m or more three. Add one with no working bow "
            "thruster (vessels of 100 m or more) and one when the forecast wind exceeds 20 kn; laden tankers of 150 m "
            "or more at least two; dead ships at least three; four at most.",
        ]

    def _pec_text(self, f):
        p = f["master_pec"]
        if not p:
            return "none held"
        return (f"PEC for this port, LOA limit {p['loa_limit']} m, {p['trips_12m']} movements in the last 12 months"
                + (", daylight only" if p["daylight_only"] else ""))

    def render(self, f, rng, surface):
        when = f"{f['hour']:02d}:{f['minute']:02d}"
        cargo = ("laden" if f["laden_tanker"] else "in ballast") if f["vtype"] == "tanker" else None
        if surface == "json":
            return dumps(prune({
                "movement": {"booking": f["booking"], "type": f["movement"], "terminal": f["terminal"],
                             "date": fmt_date(f["date"], style="iso"), "planned_time": when,
                             "daylight": f"sunrise {f['sunrise_h']:02d}:00, sunset {f['sunset_h']:02d}:00"},
                "vessel": {"name": f["vessel"], "type": VTYPES[f["vtype"]][0], "loa_m": f["loa_m"],
                           "draft_m": f["draft_m"], "cargo_state": cargo, "bow_thruster": f["bow_thruster"],
                           "under_own_power": not f["dead_ship"], "master": f["master"],
                           "master_exemption": self._pec_text(f)},
                "conditions": {"forecast_wind_kn": f["wind_kn"], "tidal_stream": f["current"]},
                "pilot_roster": {"class_1_pilots_free_at_time": f["senior_pilots_free"]},
                "rules": _std(rng, "Pilotage and towage directions", self._rules(f), True),
            }))
        if surface == "pilot_booking":
            rows = _kv([("Booking", f["booking"]), ("Agent", f["agent"]), ("Vessel", f"{f['vessel']} ({f['flag']})"),
                        ("Type", VTYPES[f["vtype"]][0] + (f", {cargo}" if cargo else "")),
                        ("LOA / draft", f"{f['loa_m']} m / {f['draft_m']} m"),
                        ("Bow thruster", "yes" if f["bow_thruster"] else "no / defective"),
                        ("Propulsion", "dead ship, towed" if f["dead_ship"] else "own power"),
                        ("Movement", f"{f['movement']} - {f['terminal']}"),
                        ("Requested time", f"{fmt_date(f['date'], rng)} {when}"),
                        ("Master's exemption", self._pec_text(f)), ("Wind forecast", f"{f['wind_kn']} kn"),
                        ("Tidal stream", f["current"])], rng)
            return (f"{f['port'].upper()} PILOT BOOKING\n{rows}\nDuty desk: Class 1 pilots free at that time: "
                    f"{f['senior_pilots_free']}. Daylight: sunrise {f['sunrise_h']:02d}:00, sunset "
                    f"{f['sunset_h']:02d}:00.\n\n" + _std(rng, "Pilotage and towage directions", self._rules(f)))
        msgs = [("ops", f"{f['vessel']} {f['movement']} {f['terminal']} booked for {when}. {VTYPES[f['vtype']][0]}"
                        f"{', ' + cargo if cargo else ''}, {f['loa_m']} m, draft {f['draft_m']} m"
                        + (", no bow thruster" if not f["bow_thruster"] else "") + (", DEAD SHIP" if f["dead_ship"] else "")),
                ("agent", f"Master {f['master']} holds: {self._pec_text(f)}. He'd like to bring her in himself."
                 if f["master_pec"] else "No exemption on board, pilot please."),
                ("duty pilot", f"Forecast {f['wind_kn']} kn, stream {f['current']}. I've got "
                               f"{_plural(f['senior_pilots_free'], 'Class 1 pilot')} free at that time."),
                ("ops", f"Sunset {f['sunset_h']:02d}:00, sunrise {f['sunrise_h']:02d}:00.")]
        out = [f"#marine-ops {f['port']} {fmt_date(f['date'], rng)}"]
        for who, text in msgs:
            out.append(f"{who}: {noisy(text, rng, 0.1)}")
        return "\n".join(out) + "\n\n" + _std(rng, "Pilotage and towage directions", self._rules(f))

    def questions(self):
        return {
            "pilotage_ruling": choice("How should the pilotage rules be applied to this movement?", {
                "pec_exemption_accepted": "Let the master take the vessel through without a pilot under the exemption certificate.",
                "one_pilot": "Board one pilot in the normal way.",
                "two_pilots": "Board two pilots, one of them Class 1.",
                "reschedule_movement": "Move the booking to a time when the rules can be met.",
            }),
            "towage_level": score("How much towage should the movement have?", [
                "No tug assistance.",
                "One tug.",
                "Two tugs.",
                "Three tugs.",
                "Four tugs.",
            ]),
            "handling_difficulty": score("How demanding is this movement for the pilot and tugs?", [
                "Routine.",
                "Some care needed on one count.",
                "Demanding on several counts.",
                "Difficult: size, weather, stream and vessel limitations stack up.",
            ]),
            "pec_covers_movement": noul(
                "The master's pilotage exemption certificate covers this movement.",
                yes="The master holds a PEC whose every condition (size, recent movements, cargo, daylight, single-pilot movement) is met.",
                no="There is no PEC, or one of its conditions fails for this movement."),
            "night_restriction_applies": noul(
                "The port's night-movement restriction applies to this movement as booked."),
        }


# =========================================================================== schema 4
CLAIMS = {
    "medical_relief": "urgent medical supplies consigned to the regional health authority",
    "reefer": "a large live reefer load",
    "perishable": "perishable cargo close to its shelf-life deadline",
    "window": "a liner berthing window",
    "crew_change": "an emergency crew change",
}
QUEUE_CATS = {"ordinary": "no priority", "window": "liner within its window", "passenger": "scheduled passenger call",
              "reefer": "reefer priority granted"}


class PriorityBerthingRequest(Schema):
    name = "priority_berthing_request"
    description = ("A ship's agent asks for priority over the anchorage queue on cargo or service grounds: grant, "
                   "partly grant, refuse, or refer, under the port's priority list.")
    surfaces = {"json": 0.5, "agent_letter": 0.25, "queue_board": 0.25}

    RANK = {"medical_relief": 1, "passenger": 2, "window": 3, "reefer": 4, "perishable": 4, "ordinary": 9}

    def sample_factors(self, rng):
        reg = regime(rng)
        claim = pick_w(rng, {"medical_relief": 1.5, "reefer": 2.5, "perishable": 2, "window": 2.5, "crew_change": 1.5})
        vt = {"medical_relief": "container_feeder", "reefer": pick(rng, ["container_feeder", "container_mainline"]),
              "perishable": "reefer", "window": pick(rng, ["container_mainline", "container_feeder", "car_carrier"]),
              "crew_change": pick(rng, ["bulk", "tanker", "general"])}[claim]
        v = _vessel(rng, vt)
        good = {"clear": chance(rng, 0.6), "borderline": chance(rng, 0.75), "conflicting": chance(rng, 0.5)}[reg]
        d = {"claim": claim}
        if claim == "reefer":
            claimed = near(rng, 150, 25, integer=True) if reg == "borderline" else pick(rng, [90, 180, 240, 320, 460])
            manifest = (claimed if good else int(claimed * rng.uniform(0.55, 0.85))) if chance(rng, 0.94) else None
            d.update(claimed=claimed, documented=manifest, genset_hours=pick(rng, [None, None, 18, 30, 60]))
        elif claim == "perishable":
            claimed = near(rng, 72, 14, integer=True) if reg == "borderline" else pick(rng, [36, 48, 60, 96, 120])
            documented = (claimed if good else claimed + pick(rng, [24, 36, 60])) if chance(rng, 0.92) else None
            d.update(claimed=claimed, documented=documented)
        elif claim == "window":
            claimed = near(rng, 6, 1.5) if reg == "borderline" else pick(rng, [1.5, 3.0, 4.5, 8.0, 11.0])
            documented = (claimed if good else claimed + pick(rng, [4.0, 7.0])) if chance(rng, 0.92) else None
            d.update(claimed=round(claimed, 1), documented=None if documented is None else round(documented, 1),
                     onward_slack_h=pick(rng, [8, 14, 30, 48]))
        elif claim == "medical_relief":
            claimed = pick(rng, [24, 36, 48, 72])
            documented = claimed if good else None
            d.update(claimed=claimed, documented=documented, boxes=pick(rng, [1, 2, 4, 12]),
                     total_boxes=pick(rng, [420, 900, 1600]))
        else:
            claimed = pick(rng, [10, 20, 30, 40])
            documented = claimed if good else None
            d.update(claimed=claimed, documented=documented)
        n_ahead = pick_w(rng, {1: 2, 2: 3, 3: 3, 4: 2})
        cats = ["ordinary"] + [pick_w(rng, {"ordinary": 6, "window": 2, "passenger": 1, "reefer": 1})
                               for _ in range(n_ahead - 1)]
        rng.shuffle(cats)
        queue = [{"vessel": _vname(rng, pick(rng, list(VTYPES))), "category": c,
                  "waiting_h": between(rng, 2, 40, integer=True)} for c in cats]
        return {"regime": reg, "port": pick(rng, PORTS), **v, **d, "queue_ahead": queue,
                "berth_hours": pick(rng, [6, 8, 10, 12, 16, 24]), "agent": _agency(rng),
                "agent_name": person(rng), "notice_h": pick(rng, [24, 36, 48]), "now": rand_ts(rng, hours=(6, 20)),
                "tone": pick(rng, ["formal", "formal", "pressing"]), "ref": ident(rng, "PR", 5, "-")}

    # ------------------------------------------------------------------ the rules' logic
    def _qualifies(self, f):
        c = f["claim"]
        if c == "reefer":
            return f["claimed"] >= 150
        if c == "perishable":
            return f["claimed"] <= 72
        if c == "window":
            return f["claimed"] <= 6
        if c == "medical_relief":
            return f["claimed"] <= 48
        return True

    def _docs_back(self, f):
        c, doc = f["claim"], f["documented"]
        if doc is None:
            return False
        if c == "reefer":
            return doc >= 0.9 * f["claimed"]
        if c in ("perishable", "window"):
            return doc <= f["claimed"] + 0.5
        return True

    def intended(self, f):
        q, docs = self._qualifies(f), self._docs_back(f)
        c = f["claim"]
        rank = self.RANK.get(c, 5)
        if c == "crew_change":
            ruling = "refer_to_harbour_master" if docs else "keep_queue_order"
        elif not (q and docs):
            ruling = "keep_queue_order"
        elif any(self.RANK[x["category"]] <= rank for x in f["queue_ahead"]):
            ruling = "move_up_partially"
        else:
            ruling = "grant_priority"
        total = len(f["queue_ahead"]) * f["berth_hours"]
        kd = 0 if total < 12 else 1 if total < 24 else 2 if total < 48 else 3
        if any(x["category"] != "ordinary" for x in f["queue_ahead"]):
            kd = 3 if total >= 24 else max(kd, 2)
        # time sensitivity on the documented facts (claimed-only figures count for little)
        doc = f["documented"]
        if c == "medical_relief":
            ts = 3 if doc is not None and doc <= 48 else 2 if doc is not None else 1
        elif c == "perishable":
            v = doc if doc is not None else None
            ts = 1 if v is None else 3 if v <= 48 else 2 if v <= 72 else 1 if v <= 120 else 0
        elif c == "reefer":
            units = doc or 0
            ts = 2 if units >= 400 else 1 if units >= 150 else 0
            if f.get("genset_hours") is not None and f["genset_hours"] <= 24:
                ts = 3
        elif c == "window":
            ts = 2 if f["onward_slack_h"] < 12 else 1
        else:
            ts = 2 if doc is not None and doc <= 20 else 1 if doc is not None else 0
        return {"priority_request_ruling": ruling, "knock_on_delay": kd, "cargo_time_sensitivity": ts,
                "claim_qualifies": "yes" if q else "no", "documents_back_claim": "yes" if docs else "no"}

    RULES = [
        "Berths are allocated in order of arrival at the pilot station, subject to this priority list.",
        "Category 1: vessels in distress, or carrying urgent medical or relief cargo for the region that is needed "
        "within 48 h, go first.",
        "Category 2: scheduled passenger calls on their published day.",
        "Category 3: liner vessels arriving within 6 h of an agreed berthing window.",
        "Category 4: vessels with 150 or more live reefer units, or perishable cargo with a documented shelf-life "
        "deadline within 72 h; they move ahead of vessels without priority but not ahead of categories 1-3 or an "
        "earlier category 4 vessel.",
        "Crew-change and bunkering emergencies are at the harbour master's discretion, on a written request with "
        "supporting documents.",
        "Every claim must be supported by the manifest, a terminal confirmation or a consignee or shipper "
        "declaration; unsupported or contradicted claims are not considered.",
    ]

    def _claim_text(self, f):
        c = f["claim"]
        if c == "reefer":
            return f"{f['claimed']} live reefer units on board"
        if c == "perishable":
            return f"chilled cargo whose shelf-life deadline falls in {f['claimed']} h"
        if c == "window":
            return f"arriving {f['claimed']} h from the start of our agreed berthing window"
        if c == "medical_relief":
            return (f"{f['boxes']} of {f['total_boxes']} containers are medical supplies for the regional health "
                    f"authority, needed within {f['claimed']} h")
        return f"an emergency crew change, relief crew flights in {f['claimed']} h"

    def _doc_text(self, f):
        c, doc = f["claim"], f["documented"]
        if doc is None:
            return {"reefer": "manifest reefer list not yet received", "perishable": "no shipper's declaration attached",
                    "window": "no terminal window confirmation attached",
                    "medical_relief": "no consignee declaration attached",
                    "crew_change": "no written request or flight bookings attached"}[c]
        return {"reefer": f"manifest reefer list shows {doc} live units",
                "perishable": f"shipper's declaration gives a shelf-life deadline in {doc} h",
                "window": f"terminal confirmation puts arrival {doc} h from the window start",
                "medical_relief": f"health authority's declaration attached (needed within {doc} h)",
                "crew_change": f"written request with flight bookings attached (flights in {doc} h)"}[c]

    def _extra(self, f):
        if f["claim"] == "reefer" and f.get("genset_hours") is not None:
            return f"vessel reports generator fuel for reefer power for about {f['genset_hours']} h at anchor"
        if f["claim"] == "window":
            return f"next port window {f['onward_slack_h']} h after planned sailing"
        return None

    def render(self, f, rng, surface):
        tf = _tfmt(rng)
        q = [[x["vessel"], QUEUE_CATS[x["category"]], f"{x['waiting_h']} h"] for x in f["queue_ahead"]]
        if surface == "json":
            return dumps(prune({
                "request": {"ref": f["ref"], "vessel": f["vessel"], "type": VTYPES[f["vtype"]][0], "loa_m": f["loa_m"],
                            "agent": f["agent"], "received": tf(f["now"]), "grounds": self._claim_text(f),
                            "supporting_documents": self._doc_text(f), "other_information": self._extra(f),
                            "expected_berth_time_h": f["berth_hours"]},
                "queue_ahead": [{"vessel": x["vessel"], "priority_status": QUEUE_CATS[x["category"]],
                                 "waiting_h": x["waiting_h"]} for x in f["queue_ahead"]],
                "rules": _std(rng, "Berthing priority rules", self.RULES, True),
            }))
        if surface == "agent_letter":
            push = (" We must stress that any delay will have serious consequences for our principals."
                    if f["tone"] == "pressing" else "")
            body = [f"On behalf of the owners of {f['vessel']} ({VTYPES[f['vtype']][0]}, LOA {f['loa_m']} m), now at "
                    f"anchor, we request priority berthing on the grounds of {CLAIMS[f['claim']]}: "
                    f"{self._claim_text(f)}.{push}",
                    f"Supporting documents: {self._doc_text(f)}.",
                    (f"Please note: {self._extra(f)}." if self._extra(f) else ""),
                    f"Expected time alongside about {f['berth_hours']} h."]
            return "\n".join([f"Subject: Priority berthing request {f['ref']} - {f['vessel']}",
                              f"From: {f['agent_name']}, {f['agent']}", f"Date: {tf(f['now'])}", "",
                              "To the Berth Planning Office,", "", noisy(join_sentences(body), rng), "",
                              f"Yours faithfully,\n{f['agent_name']}", "", "Queue ahead (planning office):",
                              table(["vessel", "status", "waiting"], q, rng), "",
                              _std(rng, "Berthing priority rules", self.RULES)])
        head = (f"{f['port'].upper()} ANCHORAGE QUEUE - {tf(f['now'])}\nRequest {f['ref']}: {f['vessel']} "
                f"({VTYPES[f['vtype']][0]}) asks for priority - {self._claim_text(f)}. Documents: {self._doc_text(f)}."
                + (f" Also: {self._extra(f)}." if self._extra(f) else "") + f" Berth time ~{f['berth_hours']} h.")
        return "\n".join([head, "", table(["ahead of requester", "status", "waiting"], q, rng), "",
                          _std(rng, "Berthing priority rules", self.RULES)])

    def questions(self):
        return {
            "priority_request_ruling": choice("How should the priority request be answered?", {
                "grant_priority": "Move the vessel to the front of the queue for the next suitable berth.",
                "move_up_partially": "Move it ahead of vessels without priority but behind those the rules rank equal or higher.",
                "keep_queue_order": "Refuse; the vessel keeps its place in arrival order.",
                "refer_to_harbour_master": "Pass the request to the harbour master, whose discretion the rules reserve for it.",
            }),
            "knock_on_delay": score("If the request were granted in full, how much delay would it push onto vessels already waiting?", [
                "Negligible: under half a day in total, all on vessels without priority.",
                "Modest: half a day to a day in total, all on vessels without priority.",
                "Substantial: one to two days in total, or less than a day falling partly on vessels with priority of their own.",
                "Severe: more than two days in total, or a day or more falling partly on vessels with priority of their own.",
            ]),
            "cargo_time_sensitivity": score("On the documented facts, how time-critical is the cargo or service behind the request?", [
                "Not time-critical.",
                "Somewhat: delay costs money or convenience.",
                "Clearly: a deadline or connection is at risk within a few days.",
                "Critical: harm or loss follows within hours to a day or two.",
            ]),
            "claim_qualifies": noul(
                "As the agent describes it, the request falls within one of the port's priority categories.",
                yes="Taking the agent's figures at face value, they meet a category's terms, or the request is of a kind the rules leave to the harbour master.",
                no="Even on the agent's own figures, the request misses the category's threshold or terms."),
            "documents_back_claim": noul(
                "The documents supplied back up the agent's description of the cargo or service."),
        }


# =========================================================================== schema 5
# vessel class: (description, mean wind kn, gust kn, swell m, visibility nm)
CLASSES = {
    "large": ("LOA 250 m or more, and all car carriers", 22, 30, 2.5, 1.0),
    "medium": ("LOA 150-249 m", 28, 36, 3.0, 0.5),
    "small": ("LOA under 150 m", 33, 42, 3.5, 0.3),
}
PARAMS = ["mean", "gust", "swell", "vis"]


class WeatherMovementCall(Schema):
    name = "weather_movement_call"
    description = ("A planned berthing or unberthing under a wind, swell or visibility forecast near the harbour "
                   "master's limits for the vessel's class.")
    surfaces = {"json": 0.5, "weather_brief": 0.25, "hm_log": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        v = _vessel(rng)
        cls = "large" if v["loa_m"] >= 250 or v["vtype"] == "car_carrier" else "medium" if v["loa_m"] >= 150 else "small"
        lim = CLASSES[cls][1:]
        gov = pick_w(rng, {"mean": 3, "gust": 3, "swell": 2, "vis": 2})
        if reg == "clear":
            r = pick(rng, [between(rng, 0.45, 0.78), between(rng, 0.45, 0.78), between(rng, 1.28, 1.6)])
        elif reg == "borderline":
            r = near(rng, pick(rng, [0.85, 1.0, 1.0, 1.2]), 0.06)
        else:
            r = between(rng, 0.7, 1.3)

        lm, lg, ls, lv = lim
        cap = min(0.8, r) * 0.95
        if gov == "mean":
            mean = lm * r
            gust = mean + rng.uniform(3, max(3.5, 0.9 * r * (lg - lm)))
        elif gov == "gust":
            gust = lg * r
            mean = gust - rng.uniform((lg - lm) * r + 1, (lg - lm) * r + 8)
        elif gov == "vis":
            mean = lm * rng.uniform(0.15, 0.4)
            gust = mean + rng.uniform(3, 6)
        else:
            mean = lm * rng.uniform(0.35, cap)
            gust = min(lg * cap, mean + rng.uniform(4, 10))
        sev = {"mean": int(round(mean)), "gust": int(round(max(gust, mean + 3))),
               "swell": round(ls * (r if gov == "swell" else rng.uniform(0.2, 0.6)), 1),
               "vis": round(lv / r, 1) if gov == "vis" else pick(rng, [2.0, 3.0, 4.0, 6.0, 8.0, 10.0])}
        # the second forecast source: agrees, or reads lower on the governing parameter (and its partner)
        other = dict(sev)
        dis = pick(rng, {"clear": [0, 0, 1], "borderline": [0, 1, 1, 2], "conflicting": [1, 2, 2]}[reg])
        if dis:
            k = rng.uniform(0.88, 0.93) if dis == 1 else rng.uniform(0.65, 0.8)
            if gov in ("mean", "gust"):
                other["mean"], other["gust"] = int(round(sev["mean"] * k)), int(round(sev["gust"] * k))
            elif gov == "swell":
                other["swell"] = round(sev["swell"] * k, 1)
            else:
                other["vis"] = round(sev["vis"] / k, 1)
        if chance(rng, 0.5):
            a, b = sev, other
        else:
            a, b = other, sev
        gale = chance(rng, 0.5) if r > 1.15 and gov in ("mean", "gust") else chance(rng, 0.05)
        nxt = pick(rng, [None, 3, 6, 9, 11, 14, 20, 30]) if r > 0.97 else None
        return {"regime": reg, "port": pick(rng, PORTS), **v, "vclass": cls, "governing": gov,
                "source_a": a, "source_b": b, "gale_warning": gale, "next_window_h": nxt,
                "movement": pick(rng, ["berthing", "unberthing", "berthing", "shifting berth"]),
                "terminal": pick(rng, TERMINALS), "start": rand_ts(rng, hours=(0, 23)),
                "duration_h": pick(rng, [1.5, 2.0, 2.5, 3.0]),
                "front_offset_h": pick(rng, [None, None, None, -5, -1, 1, 4, 7]),
                "hm": person(rng), "pilot": person(rng), "target_ratio": round(r, 3)}

    # ------------------------------------------------------------------ the rules' logic
    def _ratios(self, f):
        lim = CLASSES[f["vclass"]][1:]
        out = {}
        for i, p in enumerate(PARAMS):
            a, b = f["source_a"][p], f["source_b"][p]
            if p == "vis":
                out[p] = lim[i] / min(a, b)
            else:
                out[p] = max(a, b) / lim[i]
        return out

    def intended(self, f):
        rr = self._ratios(f)
        gov = max(PARAMS, key=lambda p: rr[p])
        r = rr[gov]
        if r <= 0.85:
            call = "proceed"
        elif r <= 1.0:
            call = "proceed_with_conditions"
        elif not f["gale_warning"] and f["next_window_h"] is not None and f["next_window_h"] <= 12:
            call = "postpone_to_window"
        else:
            call = "suspend_class_movements"
        margin = 0 if r <= 0.85 else 1 if r <= 1.0 else 2 if r <= 1.2 else 3
        a, b = f["source_a"], f["source_b"]
        spread = max(abs(a[p] - b[p]) / max(a[p], b[p]) for p in PARAMS)
        dis = 2 if spread >= 0.15 else 1 if spread >= 0.05 else 0
        fo = f["front_offset_h"]
        front = fo is not None and -1 <= fo <= f["duration_h"] + 1
        unc = 3 if dis == 2 and front else 2 if dis == 2 or front else dis
        names = {"mean": "mean_wind", "gust": "gusts", "swell": "swell", "vis": "visibility"}
        return {"movement_call": call, "governing_limit": "none_close" if r <= 0.85 else names[gov],
                "limit_margin": margin, "forecast_uncertainty": unc,
                "weather_window_holds": "yes" if r <= 1.0 else "no"}

    def _rules(self, f):
        d, mw, g, sw, vis = CLASSES[f["vclass"]]
        return [
            f"Limits for this vessel's class ({d}): mean wind {mw} kn, gusts {g} kn, swell at the boarding ground "
            f"{sw} m, visibility not below {vis} nm.",
            "Judge each limit on the more severe of the two forecast sources over the whole movement, from pilot "
            "boarding to all fast (or clear of the harbour).",
            "If every parameter stays below 85% of its limit (visibility above 1/0.85 of its limit), the movement "
            "proceeds normally.",
            "Between 85% and 100% of any limit: proceed only with one extra tug and the agreement of a senior pilot.",
            "Any limit exceeded: postpone to the next forecast window if it opens within 12 h and no gale warning is "
            "in force for the harbour; otherwise suspend movements of this class until the harbour master lifts it.",
        ]

    def _src(self, s):
        return (f"mean {s['mean']} kn, gusts {s['gust']} kn, swell {s['swell']} m, visibility {s['vis']} nm")

    def _front(self, f):
        fo = f["front_offset_h"]
        if fo is None:
            return "no frontal passage expected"
        if fo < 0:
            return f"cold front expected {abs(fo)} h before the planned start"
        return f"cold front expected {fo} h after the planned start" if fo else "cold front expected around the planned start"

    def render(self, f, rng, surface):
        tf = _tfmt(rng)
        if f["target_ratio"] <= 0.97:
            nxt = pick(rng, ["little change through the period", "slowly easing later in the day",
                             "steady through the movement and the following 12 h"])
        elif f["next_window_h"] is None:
            nxt = "no window inside the limits within 24 h"
        else:
            nxt = f"conditions forecast to ease inside the limits about {f['next_window_h']} h after the planned start"
        if surface == "json":
            return dumps(prune({
                "movement": {"vessel": f["vessel"], "type": VTYPES[f["vtype"]][0], "loa_m": f["loa_m"],
                             "kind": f["movement"], "terminal": f["terminal"], "planned_start": tf(f["start"]),
                             "expected_duration_h": f["duration_h"]},
                "forecast_for_movement": {"met_service": self._src(f["source_a"]), "port_model": self._src(f["source_b"]),
                                          "front": self._front(f), "outlook": nxt,
                                          "gale_warning_in_force": f["gale_warning"]},
                "rules": _std(rng, "Harbour master's weather limits", self._rules(f), True),
            }))
        if surface == "weather_brief":
            rows = _kv([("Vessel", f"{f['vessel']} ({VTYPES[f['vtype']][0]}, LOA {f['loa_m']} m)"),
                        ("Movement", f"{f['movement']} at {f['terminal']}, start {tf(f['start'])}, ~{f['duration_h']} h"),
                        ("Met service", self._src(f["source_a"])), ("Port model", self._src(f["source_b"])),
                        ("Synoptic", self._front(f)), ("Outlook", nxt),
                        ("Gale warning", "IN FORCE" if f["gale_warning"] else "none")], rng)
            return f"{f['port'].upper()} MARINE WEATHER BRIEF\n{rows}\n\n" + _std(rng, "HM weather limits", self._rules(f))
        t = f["start"] - timedelta(hours=rng.randint(2, 6))
        lines = [f"{f['port']} harbour master's log"]
        for gap, text in [(0, f"{f['vessel']} {f['movement']} {f['terminal']} planned {tf(f['start'])}, "
                              f"duration ~{f['duration_h']} h. LOA {f['loa_m']} m."),
                          (rng.randint(5, 40), f"Met service for the period: {self._src(f['source_a'])}."),
                          (rng.randint(2, 20), f"Port model: {self._src(f['source_b'])}."),
                          (rng.randint(2, 20), f"{self._front(f).capitalize()}; {nxt}."
                                               + (" Gale warning issued for the harbour." if f["gale_warning"] else "")),
                          (rng.randint(2, 20), f"Pilot {f['pilot']} asks for a decision.")]:
            t += timedelta(minutes=gap)
            lines.append(f"{t.strftime('%H:%M')}  {noisy(text, rng, 0.05)}")
        return "\n".join(lines) + "\n\n" + _std(rng, "Weather limits", self._rules(f))

    def questions(self):
        return {
            "movement_call": choice("What should be decided about the planned movement?", {
                "proceed": "Let it go ahead as planned.",
                "proceed_with_conditions": "Let it go ahead with an extra tug and a senior pilot's agreement.",
                "postpone_to_window": "Postpone it to the next forecast window within the limits.",
                "suspend_class_movements": "Suspend movements of vessels of this class until the harbour master lifts the suspension.",
            }),
            "governing_limit": choice("Which of the port's weather limits governs this decision?", {
                "mean_wind": "The mean wind speed limit.",
                "gusts": "The gust limit.",
                "swell": "The swell limit at the boarding ground.",
                "visibility": "The minimum visibility.",
                "none_close": "None: every parameter is comfortably inside its limit.",
            }),
            "limit_margin": score("How close does the forecast come to the limits for this vessel during the movement?", [
                "Well inside every limit.",
                "Close to a limit without exceeding it.",
                "Over a limit by up to a fifth.",
                "Over a limit by more than a fifth.",
            ]),
            "forecast_uncertainty": score("How far can the forecast be trusted for the movement period?", [
                "Well: the sources agree and conditions are settled.",
                "Reasonably: minor differences between the sources.",
                "Doubtfully: the sources disagree, or a front is due during the movement.",
                "Poorly: the sources disagree and a front is due during the movement.",
            ]),
            "weather_window_holds": noul(
                "Conditions stay within this vessel's limits for the whole movement.",
                yes="On the more severe forecast, wind, gusts, swell and visibility all stay within the class limits from boarding to all fast.",
                no="On the more severe forecast, at least one parameter breaches its limit during the movement."),
        }


WORKFLOW = Workflow(
    name="port_berth_allocation",
    title="Port berth allocation",
    description=("Berth, pilotage, towage and movement decisions in a port under draft, tide, weather, priority "
                 "cargo and congestion constraints, judged against written port rules."),
    domain="transport operations / port marine services",
    schemas=[BerthAssignment(), TidalDraftWindow(), PilotageTugRequirement(), PriorityBerthingRequest(),
             WeatherMovementCall()],
    notes="Fictional ports and rules quoted in each state. Not freight exception handling or customs; no customer "
          "conversation; agents' messages are evidence only.",
)
