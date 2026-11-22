"""Hot work and confined space permits: site-safety permit decisions on construction and utility work, judged
against the site's written permit procedure.

Every state quotes the governing procedure with case-specific limits (combustible clearance radius, fire-watch
durations, atmospheric action levels, retest intervals, rescue response times, extension limits), so the reader has
to apply the quoted numbers. Sites, contractors, procedures and people are fictional; the gas limits are typical
action levels, not advice. These are permit-to-work decisions by the site's permit issuer or safety officer: no
municipal building permits (building_permit_review), and no injury claims or insurance questions.

Each schema exposes `intended(f)`: the answer a careful reviewer would reach from the factors (choice key, "yes" /
"no", or a score level index). It is never shown to the model; it exists so the spread of answers across options
can be checked before labelling.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from .base import (
    Schema,
    Workflow,
    between,
    bullets,
    chance,
    choice,
    dumps,
    fmt_date,
    join_sentences,
    kv_lines,
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

# =========================================================================== shared pools
STEMS = ["Alder", "Basalt", "Cobalt", "Driftwood", "Ember", "Fernhill", "Ironbark", "Juniper", "Lakeshore", "Oakridge",
         "Pinecrest", "Quarry", "Redfern", "Saltmarsh", "Upland", "Valewood", "Westbrook", "Yarrow", "Ashgrove",
         "Bluestem", "Copperline", "Dunmore", "Elmstead", "Harrowgate", "Kingsfold", "Larkspur", "Moraine", "Nettlebed",
         "Orrin", "Rushford", "Stannard", "Wexcombe", "Whitlow", "Brackett", "Calloway", "Denholm", "Fairbairn",
         "Glenmoor", "Hatherley", "Lindell", "Quillhaven", "Pellbridge", "Marrowdale", "Corvina", "Halden"]
BUILDING_SITES = ["{s} Middle School addition", "{s} Public Library renovation", "{s} Hospital east wing",
                  "{s} Transit Center", "{s} Senior Housing", "{s} Community Recreation Center",
                  "{s} Science Building renovation", "{s} County Courthouse renovation", "{s} Office Tower, Phase {n}",
                  "{s} Cold Storage expansion", "{s} Medical Office Building", "{s} Student Housing, Phase {n}"]
UTILITY_SITES = ["{s} Wastewater Treatment Plant upgrade", "{s} Lift Station {n} rehabilitation",
                 "{s} Sewer Interceptor, Contract {n}", "{s} Reservoir outlet works", "{s} River bridge rehabilitation",
                 "{s} Water Treatment Plant filter rehabilitation", "{s} Storm Drain Improvements, Phase {n}",
                 "{s} Digester cleaning and recoating"]
SUB_SUFFIX = {"steel": ["Steel Erectors", "Welding & Fabrication", "Ironworks"],
              "mech": ["Mechanical", "Pipe & Boiler", "Refrigeration"],
              "roof": ["Roofing", "Roofing & Sheet Metal"],
              "elec": ["Electric", "Electrical Contractors"],
              "utility": ["Utilities", "Pipeline Services", "Tank Services", "Industrial Services", "Environmental"],
              "general": ["Builders", "Constructors", "Contracting"]}


def _site(rng, kind):
    stem = pick(rng, STEMS)
    return pick(rng, BUILDING_SITES if kind == "building" else UTILITY_SITES).format(s=stem, n=rng.randint(2, 9))


def _sub(rng, trade):
    return f"{pick(rng, STEMS)} {pick(rng, SUB_SUFFIX[trade])}"


def _permit_no(rng, prefix):
    return pick(rng, [f"{prefix}-{rng.randint(1000, 9999)}", f"{prefix} {rng.randint(100, 999)}",
                      f"{rng.randint(2025, 2027)}-{prefix}-{rng.randint(10, 999):03d}"])


def _ds(rng):
    return pick(rng, ["iso", "us", "long", "eu"])


def _clamp(x, lo=0, hi=3):
    return max(lo, min(hi, x))


def _hhmm(t):
    return t.strftime("%H:%M")


def pick_static(key, options):
    """A stable choice keyed on a case id (no rng draw), so the same case always reads the same."""
    return options[sum(map(ord, str(key))) % len(options)]


# confined spaces: (key, text, orientation, depth range or None, retrieval feasible, IDLH potential, has lines)
SPACES = [
    ("wet_well", "wet well at Lift Station {n}", "vertical", (5.0, 9.0), True, True, True),
    ("valve_vault", "valve vault VV-{n}", "vertical", (2.5, 4.5), True, False, True),
    ("storm_manhole", "storm sewer manhole MH-{n}", "vertical", (3.0, 6.0), True, False, True),
    ("sanitary_manhole", "sanitary sewer manhole SMH-{n}", "vertical", (3.0, 7.0), True, True, True),
    ("water_tank", "drained steel water tank T-{n} (side manway)", "horizontal", None, True, False, True),
    ("digester", "anaerobic digester No. {n} (sludge removed)", "vertical", (6.0, 10.0), True, True, True),
    ("box_girder", "box girder cell {n} of the river bridge", "horizontal", None, False, False, False),
    ("utility_tunnel", "utility tunnel section between vaults {n} and {m}", "horizontal", None, False, False, True),
    ("splice_vault", "precast electrical splice vault EV-{n}", "vertical", (2.0, 3.0), True, False, True),
    ("grit_chamber", "grit chamber No. {n} at the treatment plant", "vertical", (3.0, 5.0), True, True, True),
    ("crawl_tank", "crawl space beside an abandoned fuel-oil tank", "horizontal", None, False, False, False),
]


SPACE_SITES = {
    "wet_well": ["{s} Lift Station {n} rehabilitation", "{s} Sewer Rehabilitation, Contract {n}"],
    "valve_vault": ["{s} Water Main Replacement, Phase {n}", "{s} Reservoir outlet works"],
    "storm_manhole": ["{s} Storm Drain Improvements, Phase {n}", "{s} Main Street reconstruction"],
    "sanitary_manhole": ["{s} Sewer Interceptor, Contract {n}", "{s} Sewer Rehabilitation, Contract {n}"],
    "water_tank": ["{s} Water Storage Tank recoating", "{s} Water Treatment Plant filter rehabilitation"],
    "digester": ["{s} Wastewater Treatment Plant upgrade", "{s} Digester cleaning and recoating"],
    "box_girder": ["{s} River bridge rehabilitation"],
    "utility_tunnel": ["{s} Campus Utility Upgrade, Phase {n}", "{s} Hospital central plant expansion"],
    "splice_vault": ["{s} Campus Utility Upgrade, Phase {n}", "{s} Substation feeder relocation"],
    "grit_chamber": ["{s} Wastewater Treatment Plant upgrade"],
    "crawl_tank": ["{s} County Courthouse renovation", "{s} Public Library renovation"],
}


def _space_site(sp, rng):
    return pick(rng, SPACE_SITES[sp[0]]).format(s=pick(rng, STEMS), n=rng.randint(2, 9))


def _space_text(sp, rng):
    n = rng.randint(1, 48)
    return sp[1].format(n=n, m=n + 1)


# =========================================================================== schema 1: hot work permit request
class HotWorkPermitRequest(Schema):
    name = "hot_work_permit_request"
    description = "A request for a hot work permit, checked against the site's written hot work procedure."
    surfaces = {"json": 0.5, "permit_form": 0.3, "foreman_request": 0.2}

    # (text, flame kind, cold alternative or None, roofing?, trade)
    TASKS = [
        ("weld stair stringer connections on level {lvl}", "arc", None, False, "steel"),
        ("cut out redundant steel pipe supports with an oxy-fuel torch", "torch", "cut them with a portable band saw",
         False, "mech"),
        ("grind protruding rebar ends flush in the stair core", "grind", "cut them with a hydraulic rebar shear", False,
         "general"),
        ("braze copper refrigerant lines above the corridor ceiling", "braze",
         "use press-fit fittings listed for refrigerant", False, "mech"),
        ("torch-apply the modified-bitumen cap sheet at the parapets", "torch",
         "use the self-adhered cap sheet the specification allows as an alternate", True, "roof"),
        ("make exothermic grounding connections in the electrical room", "torch", "use listed compression connectors",
         False, "elec"),
        ("plasma-cut openings in the metal deck for roof drains", "plasma", None, False, "steel"),
        ("weld shear studs to the level {lvl} metal deck", "arc", None, False, "steel"),
        ("weld embed plates next to the timber formwork", "arc", None, False, "steel"),
        ("cut a new doorway through a steel-stud wall with a disc grinder", "grind",
         "cut it with snips and a reciprocating saw", False, "general"),
        ("torch-cut the old fuel-oil tank vent piping in the boiler room", "torch",
         "cold-cut it with a pneumatic pipe cutter", False, "mech"),
        ("weld a new pipe sleeve into the existing roof curb", "arc", None, True, "mech"),
    ]
    COMBUSTIBLES = ["stacked pallets and cardboard packaging", "foam insulation boards", "timber formwork and shoring",
                    "rolls of roofing felt", "drums of form-release oil", "sawdust and offcuts from the carpenters",
                    "plastic sheeting and dust partitions", "stored ceiling tiles in cartons"]
    HIDDEN = ["wood framing behind the gypsum wall", "polystyrene insulation in the wall cavity",
              "a timber roof deck under the insulation", "old cork insulation on the ducts above"]
    OCCUPANCY = {"vacant": "vacant shell, no other trades on the floor",
                 "offices": "occupied offices on the floor above",
                 "school": "occupied school wing next door, classes in session",
                 "hospital": "operating hospital wing on the other side of the fire wall"}

    def sample_factors(self, rng):
        reg = regime(rng)
        ti = rng.randrange(len(self.TASKS))
        radius = pick(rng, [10, 11, 15])
        base_watch = pick(rng, [30, 60])
        comb = pick_w(rng, {"none": 3, "removed": 3, "blanketed": 3, "unprotected": 2})
        hidden = chance(rng, 0.25)
        openings = pick_w(rng, {"none": 5, "covered": 3, "open": 1})
        sprinklers = pick_w(rng, {"in_service": 5, "impaired": 2, "not_installed": 2})
        extinguisher = chance(rng, 0.85)
        watch = pick_w(rng, {"dedicated": 6, "shared": 3, "none_named": 1})
        flammable = pick_w(rng, {"none": 6, "adjacent_painting": 2, "fuel_lines": 1})
        gas_test = pick_w(rng, {"not_done": 2, "zero": 3, "low": 1}) if flammable != "none" else None
        shift_end = chance(rng, 0.3)
        after_hours = chance(rng, 0.4)
        occ = pick_w(rng, {"vacant": 4, "offices": 3, "school": 2, "hospital": 1})
        wind = pick(rng, [8, 15, 22, 30, 38])
        task = self.TASKS[ti]
        required = base_watch + (60 if task[3] else 0)
        post = pick(rng, [0, 30, 60, 90, 120])
        if reg == "clear":
            if chance(rng, 0.5):
                comb, hidden, openings, sprinklers, extinguisher = pick(rng, ["none", "removed"]), False, "none", \
                    "in_service", True
                watch, flammable, gas_test, post, shift_end, wind = "dedicated", "none", None, required + 30, False, 10
            else:
                comb, openings = "unprotected", pick(rng, ["open", "covered"])
                if task[2] and chance(rng, 0.5):
                    hidden = True
        elif reg == "borderline":
            post = required + pick(rng, [-30, 0, 0, 30])
            post = max(0, post)
            if flammable != "none":
                gas_test = pick(rng, ["zero", "low"])
        else:  # conflicting: tidy preparation hiding a reason to refuse, or a strong plan with one gap
            if chance(rng, 0.5):
                ti = pick(rng, [i for i, t in enumerate(self.TASKS) if t[2]])
                task = self.TASKS[ti]
                required = base_watch + (60 if task[3] else 0)
                comb, openings, extinguisher, watch, post, hidden = "removed", "none", True, "dedicated", required, True
            else:
                comb, openings, extinguisher, hidden = "blanketed", "covered", True, False
                watch = pick(rng, ["shared", "dedicated"])
                shift_end, after_hours = True, False
        return {"regime": reg, "task": ti, "level": rng.randint(1, 6), "radius_m": radius,
                "required_watch_min": required, "proposed_watch_min": post, "combustibles": comb,
                "combustible_item": pick(rng, self.COMBUSTIBLES), "hidden_combustible": hidden,
                "hidden_item": pick(rng, self.HIDDEN), "openings": openings, "sprinklers": sprinklers,
                "extinguisher": extinguisher, "watch": watch, "flammable": flammable, "gas_test": gas_test,
                "ends_at_shift_end": shift_end, "after_hours_watch_arranged": after_hours, "occupancy": occ,
                "wind_kmh": wind, "site": self._occ_site(rng, occ), "sub": _sub(rng, task[4]),
                "foreman": person(rng), "permit": _permit_no(rng, "HW"), "date": rand_date(rng),
                "start_hour": pick(rng, [7, 8, 10, 13, 14])}

    OCC_SITES = {"hospital": ["{s} Hospital east wing", "{s} Medical Center imaging addition"],
                 "school": ["{s} Middle School addition", "{s} Elementary School modernization"],
                 "offices": ["{s} County Courthouse renovation", "{s} Office Tower, Phase {n}",
                             "{s} Medical Office Building", "{s} Public Library renovation"]}

    def _occ_site(self, rng, occ):
        if occ == "vacant":
            return _site(rng, "building")
        return pick(rng, self.OCC_SITES[occ]).format(s=pick(rng, STEMS), n=rng.randint(2, 9))

    # ------------------------------------------------------------------ answers
    def _prep_ok(self, f):
        flam_ok = f["flammable"] == "none" or f["gas_test"] == "zero"
        return f["combustibles"] != "unprotected" and f["openings"] != "open" and f["extinguisher"] and flam_ok

    def _watch_ok(self, f):
        late = f["ends_at_shift_end"] and not f["after_hours_watch_arranged"]
        return f["watch"] == "dedicated" and f["proposed_watch_min"] >= f["required_watch_min"] and not late

    def intended(self, f):
        task = self.TASKS[f["task"]]
        if task[2] and f["hidden_combustible"]:
            dec = "refuse_use_cold_method"
        elif not self._prep_ok(f):
            dec = "hold_until_area_prepared"
        elif not self._watch_ok(f) or f["sprinklers"] != "in_service":
            dec = "issue_with_conditions"
        else:
            dec = "issue_permit"
        r = {"torch": 2, "plasma": 1, "arc": 1, "grind": 1, "braze": 1}[task[1]]
        r += {"unprotected": 2, "blanketed": 1, "removed": 0, "none": 0}[f["combustibles"]]
        r += 1 if f["hidden_combustible"] else 0
        r += 1 if (f["flammable"] != "none" and f["gas_test"] != "zero") else 0
        r += 1 if (task[3] and f["wind_kmh"] >= 25) else 0
        r += 1 if f["openings"] == "open" else 0
        risk = 0 if r <= 1 else 1 if r == 2 else 2 if r == 3 else 3
        c = {"vacant": 0, "offices": 1, "school": 2, "hospital": 2}[f["occupancy"]]
        c += 1 if f["sprinklers"] != "in_service" else 0
        c += 1 if f["hidden_combustible"] else 0
        return {"permit_decision": dec, "fire_ignition_risk": risk, "fire_consequence": _clamp(c),
                "fire_watch_plan_compliant": "yes" if self._watch_ok(f) else "no",
                "area_preparation_complete": "yes" if self._prep_ok(f) else "no"}

    # ------------------------------------------------------------------ rendering
    def _procedure(self, f):
        base = f["required_watch_min"] - (60 if self.TASKS[f["task"]][3] else 0)
        return (f"Site procedure SSP-07 (hot work): before a permit is issued, combustibles within {f['radius_m']} m "
                f"are removed or covered with approved fire blankets; floor and wall openings within {f['radius_m']} m "
                f"are covered; a charged extinguisher is at the work; sprinklers are in service, or an impairment "
                f"permit and patrol are in place; a dedicated, trained fire watch stays during the work and for "
                f"{base} minutes after it ends ({base + 60} minutes for torch-applied roofing or work on roofs), then "
                f"makes a final check; flammable vapours are tested to 0% LEL. Hot work is not permitted where a "
                f"practicable cold method exists and the area contains combustible construction.")

    def _task_text(self, f):
        return self.TASKS[f["task"]][0].format(lvl=f["level"])

    def _lines(self, f):
        comb = {"none": "no combustibles within the radius", "removed": f"{f['combustible_item']} moved out of the area",
                "blanketed": f"{f['combustible_item']} left in place, covered with fire blankets",
                "unprotected": f"{f['combustible_item']} within about 6 m, not covered"}[f["combustibles"]]
        open_t = {"none": "no floor or wall openings nearby", "covered": "floor penetrations nearby covered with plywood",
                  "open": "an open floor penetration about 4 m away"}[f["openings"]]
        spr = {"in_service": "sprinklers in service", "impaired": "sprinkler zone impaired for a tie-in (impairment "
               "tag posted)", "not_installed": "sprinklers not yet installed on this floor"}[f["sprinklers"]]
        watch = {"dedicated": "dedicated fire watch named", "shared": "fire watch is the welder's helper, who also "
                 "feeds stock and cleans up", "none_named": "no fire watch named"}[f["watch"]]
        watch += f"; {f['proposed_watch_min']} min watch after work" if f["proposed_watch_min"] else "; no watch after work"
        if f["ends_at_shift_end"]:
            watch += ("; work runs to the end of the shift, after-hours watch arranged with security"
                      if f["after_hours_watch_arranged"] else "; work runs to the end of the shift when the site closes")
        flam = None
        if f["flammable"] != "none":
            src = "solvent-based painting in the adjacent room" if f["flammable"] == "adjacent_painting" else \
                "fuel-oil lines in the work area"
            flam = src + {"not_done": "; no gas test yet", "zero": "; gas test 0% LEL", "low": "; gas test 3% LEL"}[
                f["gas_test"]]
        hidden = f"behind the work: {f['hidden_item']}" if f["hidden_combustible"] else None
        return comb, open_t, spr, watch, flam, hidden

    def render(self, f, rng, surface):
        ds = _ds(rng)
        task = self.TASKS[f["task"]]
        comb, open_t, spr, watch, flam, hidden = self._lines(f)
        start = datetime(f["date"].year, f["date"].month, f["date"].day, f["start_hour"], pick(rng, [0, 30]))
        alt = f"cold alternative available: {task[2]}" if task[2] else "no practicable cold method for this task"
        wind = f"wind {f['wind_kmh']} km/h at roof level" if task[3] else None
        if surface == "json":
            return dumps(prune({
                "request": {"permit": f["permit"], "site": f["site"], "contractor": f["sub"], "foreman": f["foreman"],
                            "work": self._task_text(f), "start": start.strftime("%Y-%m-%dT%H:%M"),
                            "cold_method": alt},
                "area": {"combustibles": comb, "concealed": hidden, "openings": open_t, "sprinklers": spr,
                         "extinguisher": "6 kg ABC at the work" if f["extinguisher"] else "none at the work yet",
                         "flammable_vapour": flam, "weather": wind, "surroundings": self.OCCUPANCY[f["occupancy"]]},
                "fire_watch": watch,
                "procedure": self._procedure(f),
            }))
        if surface == "permit_form":
            rows = [("Permit", f["permit"]), ("Site", f["site"]), ("Contractor", f["sub"]),
                    ("Requested by", f["foreman"]), ("Date / start", f"{fmt_date(f['date'], rng, ds)} {_hhmm(start)}"),
                    ("Work", self._task_text(f)), ("Cold method", alt), ("Combustibles", comb),
                    ("Concealed combustibles", hidden or "none noted"), ("Openings", open_t), ("Sprinklers", spr),
                    ("Extinguisher", "yes" if f["extinguisher"] else "NO"), ("Vapours", flam or "none expected"),
                    ("Fire watch", watch), ("Surroundings", self.OCCUPANCY[f["occupancy"]])]
            if wind:
                rows.append(("Weather", wind))
            return "HOT WORK PERMIT -- REQUEST FOR ISSUE\n" + kv_lines(rows, rng) + "\n\n" + self._procedure(f)
        note = pick_static(f["permit"], ["We have done this many times without a problem.",
                                         "Need this today to stay on schedule.",
                                         "Happy to walk it with you before we start.",
                                         "Crew is standing by."])
        body = [f"Requesting a hot work permit for {fmt_date(f['date'], rng, ds)} from {_hhmm(start)} to "
                f"{self._task_text(f)}.", f"{comb[0].upper()}{comb[1:]}.", f"{open_t[0].upper()}{open_t[1:]}.",
                f"{spr[0].upper()}{spr[1:]}.", "Extinguisher on hand." if f["extinguisher"] else "",
                f"{flam[0].upper()}{flam[1:]}." if flam else "", f"Note: {hidden}." if hidden else "",
                f"{watch[0].upper()}{watch[1:]}.", f"{wind[0].upper()}{wind[1:]}." if wind else "",
                f"Area: {self.OCCUPANCY[f['occupancy']]}.", note]
        return (f"From: {f['foreman']} ({f['sub']})\nTo: site safety, {f['site']}\nSubject: hot work permit "
                f"{f['permit']}\n\n" + noisy(join_sentences(body), rng, 0.12) + "\n\n" + self._procedure(f))

    def questions(self):
        return {
            "permit_decision": choice("How should the permit issuer respond to this request?", {
                "hold_until_area_prepared": "Hold the permit until the area is prepared as the procedure requires.",
                "issue_permit": "Issue the permit as requested.",
                "issue_with_conditions": "Issue it with added conditions, such as a longer or dedicated fire watch or "
                                         "a sprinkler-impairment patrol.",
                "refuse_use_cold_method": "Refuse hot work and require the cold method instead.",
            }),
            "fire_ignition_risk": score("How likely is the work, as planned, to start a fire?", [
                "Low: little heat or spark near anything that can burn.",
                "Moderate: an ignition source with well-controlled surroundings.",
                "High: open flame or sparks near combustibles or vapours that are only partly controlled.",
                "Very high: open flame or sparks with exposed combustibles, openings or hidden fuel.",
            ]),
            "fire_consequence": score("If a fire did start, how serious would the consequences be?", [
                "Limited: a vacant area with fire protection working.",
                "Moderate: other people nearby, or reduced fire protection.",
                "Serious: vulnerable occupants nearby, or fire could spread unseen.",
                "Severe: vulnerable occupants and degraded protection or concealed fuel together.",
            ]),
            "fire_watch_plan_compliant": noul(
                "The proposed fire watch meets the procedure.",
                yes="A dedicated person watches during the work and for at least the required time afterwards, "
                    "including after the site closes.",
                no="The watcher has other duties, nobody is named, or the watch after work is too short or would "
                   "lapse when the site closes."),
            "area_preparation_complete": noul(
                "The work area is already prepared as the procedure requires for hot work to start."),
        }


# =========================================================================== schema 2: confined space entry permit
class ConfinedSpaceEntryPermit(Schema):
    name = "confined_space_entry_permit"
    description = "A permit-required confined space entry: atmosphere, isolation, monitoring and rescue against the procedure."
    surfaces = {"json": 0.5, "entry_permit": 0.3, "supervisor_radio_log": 0.2}

    WORK = {"inspection": ("visual inspection and photographs", False),
            "sludge": ("remove settled sludge and debris", True),
            "coating": ("apply a solvent-based epoxy lining", True),
            "welding": ("weld-repair a corroded pipe support", True),
            "pump": ("replace a submersible pump and guide rail", False)}

    def sample_factors(self, rng):
        reg = regime(rng)
        si = rng.randrange(len(SPACES))
        sp = SPACES[si]
        work = pick(rng, list(self.WORK))
        if sp[0] in ("box_girder", "crawl_tank", "utility_tunnel") and work in ("sludge", "pump"):
            work = pick(rng, ["inspection", "welding", "coating"])
        lim = {"lel": pick(rng, [5, 10]), "h2s": pick(rng, [5, 10]), "co": pick(rng, [25, 35])}
        o2 = round(between(rng, 20.2, 20.9), 1)
        lel, h2s, co = pick(rng, [0, 0, 1, 2]), pick(rng, [0, 0, 0, 1, 2]), pick(rng, [0, 0, 2, 5])
        bump = pick_w(rng, {"today": 6, "yesterday": 2, "not_recorded": 1})
        cal_ok = chance(rng, 0.88)
        continuous = chance(rng, 0.8)
        stratified = chance(rng, 0.7)
        vent = pick_w(rng, {"continuous_forced": 6, "pre_entry_only": 2, "natural": 1})
        iso = (pick_w(rng, {"positive": 5, "locked_valves": 4, "closed_unlocked": 1.5, "none_listed": 1}) if sp[6]
               else "not_applicable")
        attendant = pick_w(rng, {"dedicated": 7, "other_duties": 2, "none": 1})
        team = pick_w(rng, {"onsite_standby": 3, "fire_dept": 5})
        eta = pick(rng, [4, 7, 9, 12, 18, 25])
        max_rescue = pick(rng, [4, 6])
        rigged = chance(rng, 0.7)
        comms = pick_w(rng, {"radio": 5, "voice": 3, "none": 1})
        if reg == "clear":
            if chance(rng, 0.5):
                o2, lel, h2s, co, bump, cal_ok, continuous, stratified = 20.9, 0, 0, 0, "today", True, True, True
                vent, attendant, rigged, comms = "continuous_forced", "dedicated", True, "radio"
                iso = "positive" if sp[6] else "not_applicable"
                team = "onsite_standby" if sp[5] or not sp[4] else team
            else:
                pick_bad = pick(rng, ["h2s", "o2", "lel"])
                if pick_bad == "h2s":
                    h2s = lim["h2s"] + pick(rng, [2, 5, 9])
                elif pick_bad == "o2":
                    o2 = round(between(rng, 18.2, 19.3), 1)
                else:
                    lel = lim["lel"] + pick(rng, [1, 3, 6])
        elif reg == "borderline":
            g = pick(rng, ["h2s", "o2", "lel", "co"])
            if g == "h2s":
                h2s = max(0, lim["h2s"] + pick(rng, [-3, -2, -1, 0]))
            elif g == "o2":
                o2 = pick(rng, [19.4, 19.6, 19.8, 20.0])
            elif g == "lel":
                lel = max(0, lim["lel"] + pick(rng, [-3, -2, -1, 0]))
            else:
                co = max(0, lim["co"] + pick(rng, [-8, -4, -1, 0]))
        else:  # conflicting: clean readings from an instrument that cannot be trusted, or clean paperwork and no rescue
            if chance(rng, 0.5):
                o2, lel, h2s, co = 20.9, 0, 0, 0
                bump = pick(rng, ["yesterday", "not_recorded"])
                if sp[2] == "vertical":
                    stratified = False
            else:
                team, eta, rigged = "fire_dept", pick(rng, [12, 18, 25]), pick(rng, [False, True])
                bump, cal_ok, continuous = "today", True, True
        depth = round(between(rng, *sp[3]), 1) if sp[3] else None
        return {"regime": reg, "space": si, "space_text": _space_text(sp, rng), "orientation": sp[2], "depth_m": depth,
                "retrieval_feasible": sp[4], "idlh_potential": sp[5] or work in ("coating", "welding"),
                "work": work, "limits": lim, "o2": o2, "lel": lel, "h2s": h2s, "co": co, "bump": bump,
                "calibration_ok": cal_ok, "continuous": continuous, "stratified": stratified, "ventilation": vent,
                "isolation": iso, "attendant": attendant, "rescue_team": team, "rescue_eta_min": eta,
                "max_rescue_min": max_rescue, "retrieval_rigged": rigged, "comms": comms,
                "site": _space_site(sp, rng), "sub": _sub(rng, "utility"), "supervisor": person(rng),
                "permit": _permit_no(rng, "CSE"), "date": rand_date(rng), "start_hour": pick(rng, [7, 8, 9, 13])}

    # ------------------------------------------------------------------ answers
    def _ratio(self, f):
        lim = f["limits"]
        o2r = (20.9 - f["o2"]) / (20.9 - 19.5) if f["o2"] < 20.9 else 0.0
        return max(o2r, f["lel"] / lim["lel"], f["h2s"] / lim["h2s"], f["co"] / lim["co"])

    def _readings_reliable(self, f):
        return f["bump"] == "today" and f["calibration_ok"] and (f["orientation"] == "horizontal" or f["stratified"])

    def _monitoring_ok(self, f):
        return self._readings_reliable(f) and f["continuous"]

    def intended(self, f):
        m = self._ratio(f)
        generates = self.WORK[f["work"]][1]
        hz = 3 if m >= 1 else 2 if m >= 0.6 else 1 if (m >= 0.25 or generates) else 0
        if generates and m >= 0.25:
            hz = max(hz, 2)
        iso = {"positive": 3, "not_applicable": 3, "locked_valves": 2, "closed_unlocked": 1, "none_listed": 0}[
            f["isolation"]]
        if m >= 1 or iso <= 1 or f["attendant"] == "none":
            dec = "deny_entry"
        elif not self._readings_reliable(f):
            dec = "retest_before_deciding"
        elif (not f["continuous"] or (f["ventilation"] != "continuous_forced" and (generates or hz >= 1))
              or f["attendant"] == "other_duties" or iso == 2 or f["comms"] == "none"
              or self._rescue(f) != "current_arrangement_adequate"):
            dec = "authorize_with_added_controls"
        else:
            dec = "authorize_entry"
        return {"entry_decision": dec, "rescue_arrangement": self._rescue(f), "atmospheric_hazard_level": hz,
                "isolation_adequacy": iso, "monitoring_plan_compliant": "yes" if self._monitoring_ok(f) else "no"}

    def _rescue(self, f):
        onsite = f["rescue_team"] == "onsite_standby"
        if not f["retrieval_feasible"] and not onsite:
            return "rescue_plan_must_be_reworked"
        if f["idlh_potential"] and not onsite and f["rescue_eta_min"] > f["max_rescue_min"]:
            return "standby_team_required"
        if f["orientation"] == "vertical" and (f["depth_m"] or 0) > 1.5 and not f["retrieval_rigged"]:
            return "add_retrieval_system"
        return "current_arrangement_adequate"

    # ------------------------------------------------------------------ rendering
    def _procedure(self, f):
        lim = f["limits"]
        return (f"Site procedure SSP-09 (permit-required confined spaces): test before entry for oxygen, then "
                f"flammables, then toxics, at the top, middle and bottom of vertical spaces, with an instrument "
                f"bump-tested that day and in calibration. Entry only with O2 19.5-23.5%, LEL below {lim['lel']}%, "
                f"H2S below {lim['h2s']} ppm, CO below {lim['co']} ppm. Monitor continuously while occupied. Isolate "
                f"lines by blanking, disconnection or locked valves. A dedicated attendant with communications stays "
                f"at the entry. Non-entry retrieval for vertical entries deeper than 1.5 m. Where the atmosphere or "
                f"the work could become immediately dangerous, rescue must be able to start within "
                f"{f['max_rescue_min']} minutes.")

    def _texts(self, f):
        iso = {"positive": "inlet line blanked and pumps locked out", "locked_valves": "inlet valves closed and locked, "
               "no blank", "closed_unlocked": "inlet valves closed, not locked", "none_listed": "no isolation listed",
               "not_applicable": "no process lines into the space"}[f["isolation"]]
        vent = {"continuous_forced": "forced-air blower running throughout", "pre_entry_only": "blower used to purge "
                "before entry, then removed", "natural": "natural ventilation only (hatch open)"}[f["ventilation"]]
        mon = (f"{'continuous personal monitor on the entrant' if f['continuous'] else 'spot checks every hour'}; "
               f"bump test: {({'today': 'today', 'yesterday': 'yesterday', 'not_recorded': 'not recorded'})[f['bump']]}; "
               f"calibration {'current' if f['calibration_ok'] else 'overdue by 3 weeks'}")
        if f["orientation"] == "vertical":
            mon += "; tested at three levels" if f["stratified"] else "; tested at the hatch only"
        att = {"dedicated": "dedicated attendant", "other_duties": "attendant also running the pump and blower",
               "none": "no attendant named"}[f["attendant"]]
        res = ("on-site trained rescue team on standby" if f["rescue_team"] == "onsite_standby"
               else f"fire department on call, about {f['rescue_eta_min']} min away")
        if f["orientation"] == "vertical":
            res += "; tripod and winch rigged" if f["retrieval_rigged"] else "; tripod on site, not rigged"
        elif not f["retrieval_feasible"]:
            res += "; route has bends, no straight-line retrieval possible"
        comms = {"radio": "radios", "voice": "voice contact", "none": "no communication method stated"}[f["comms"]]
        return iso, vent, mon, att, res, comms

    def render(self, f, rng, surface):
        ds = _ds(rng)
        iso, vent, mon, att, res, comms = self._texts(f)
        work = self.WORK[f["work"]][0]
        where = f["space_text"] + (f", {f['depth_m']} m deep" if f["depth_m"] else "")
        readings = {"O2_pct": f["o2"], "LEL_pct": f["lel"], "H2S_ppm": f["h2s"], "CO_ppm": f["co"]}
        if surface == "json":
            return dumps(prune({
                "permit": {"number": f["permit"], "site": f["site"], "contractor": f["sub"],
                           "entry_supervisor": f["supervisor"], "space": where, "work": work,
                           "date": f["date"].isoformat()},
                "pre_entry_readings": readings,
                "controls": {"monitoring": mon, "ventilation": vent, "isolation": iso, "attendant": att,
                             "communications": comms, "rescue": res},
                "procedure": self._procedure(f),
            }))
        if surface == "entry_permit":
            rows = [["O2 (%)", f["o2"]], ["LEL (%)", f["lel"]], ["H2S (ppm)", f["h2s"]], ["CO (ppm)", f["co"]]]
            return "\n".join([f"CONFINED SPACE ENTRY PERMIT {f['permit']}", kv_lines([
                ("Site", f["site"]), ("Contractor", f["sub"]), ("Entry supervisor", f["supervisor"]),
                ("Date", fmt_date(f["date"], rng, ds)), ("Space", where), ("Work", work)], rng), "",
                "Pre-entry readings:", table(["Gas", "Reading"], rows, rng), "",
                kv_lines([("Monitoring", mon), ("Ventilation", vent), ("Isolation", iso), ("Attendant", att),
                          ("Communications", comms), ("Rescue", res)], rng), "", self._procedure(f)])
        t = datetime(f["date"].year, f["date"].month, f["date"].day, f["start_hour"], 0)
        log = [(t, f"{f['supervisor']}: set up at {where} for: {work}"),
               (t + timedelta(minutes=10), f"pre-entry test O2 {f['o2']}%, LEL {f['lel']}%, H2S {f['h2s']} ppm, CO "
                                           f"{f['co']} ppm"),
               (t + timedelta(minutes=14), mon), (t + timedelta(minutes=18), f"{vent}; {iso}"),
               (t + timedelta(minutes=22), f"{att}; {comms}"), (t + timedelta(minutes=25), res),
               (t + timedelta(minutes=27), "requesting authorisation to enter")]
        return (f"RADIO / FIELD LOG -- {f['site']} ({f['sub']}), permit {f['permit']}\n"
                + "\n".join(f"{_hhmm(x)}  {noisy(y, rng, 0.06)}" for x, y in log) + "\n\n" + self._procedure(f))

    def questions(self):
        return {
            "entry_decision": choice("What should the entry supervisor decide?", {
                "authorize_entry": "Authorise entry under the permit as written.",
                "authorize_with_added_controls": "Authorise entry once specific missing controls are added.",
                "deny_entry": "Do not allow entry: the atmosphere, isolation or attendance is unacceptable.",
                "retest_before_deciding": "Retest with a properly checked instrument before deciding, because the "
                                          "readings cannot be relied on.",
            }),
            "rescue_arrangement": choice("What does the rescue arrangement need?", {
                "add_retrieval_system": "Rig the non-entry retrieval system before anyone enters.",
                "current_arrangement_adequate": "Nothing more: the arrangement meets the procedure.",
                "rescue_plan_must_be_reworked": "A new rescue plan, because retrieval from outside cannot work here.",
                "standby_team_required": "A trained rescue team on standby at the entry.",
            }),
            "atmospheric_hazard_level": score("How hazardous is the atmosphere, given the readings and the work planned?", [
                "Low: readings clean and the work adds nothing.",
                "Watchful: clean readings, but the work or the space can change them.",
                "Elevated: readings close to a limit, or likely to rise during the work.",
                "Unacceptable: at least one reading at or beyond a limit.",
            ]),
            "isolation_adequacy": score("How adequate is the isolation of lines and energy into the space?", [
                "None: nothing is isolated.",
                "Weak: valves closed but not secured.",
                "Partial: valves locked, without positive isolation.",
                "Full: lines blanked or disconnected and locked out, or no lines enter the space.",
            ]),
            "monitoring_plan_compliant": noul(
                "The atmospheric testing and monitoring plan meets the procedure.",
                yes="The instrument was bump-tested that day and is in calibration, vertical spaces were tested at "
                    "several levels, and monitoring is continuous.",
                no="The bump test or calibration is not current, levels were not all tested, or monitoring is not "
                   "continuous."),
        }


# =========================================================================== schema 3: permit extension at shift change
class PermitShiftExtension(Schema):
    name = "permit_shift_extension"
    description = "A request to extend a hot work or confined space permit past its shift, with conditions that may have changed."
    surfaces = {"json": 0.5, "handover_sheet": 0.25, "night_supervisor_message": 0.25}

    # (key, text, applies to "hot" / "cs" / "both", material?)
    CHANGES = [
        ("new_crew", "the night crew replaces the day crew; two of them were not at the permit briefing", "both", False),
        ("watch_changed", "the fire watch is now the night labourer, who is also loading the hoist", "hot", True),
        ("solvent_adjacent", "painters started a solvent-based primer in the next room at 16:00", "both", True),
        ("wind_up", "wind has picked up to about 35 km/h at roof level", "hot", True),
        ("sprinkler_impairment", "the sprinkler zone was shut down at 15:30 for a tie-in", "hot", True),
        ("blower_moved", "the ventilation blower was taken to another manhole at 16:30", "cs", True),
        ("pump_running", "a gasoline-powered concrete pump is now running 10 m from the opening", "cs", True),
        ("lighting", "temporary lighting has been added for night work", "both", False),
        ("rain", "light rain has started", "both", False),
        ("visitor", "the owner's inspector will visit at 19:00", "both", False),
    ]
    HOT_WORK = ["welding the stair stringers on level 4", "torch-applying cap sheet at the north parapet",
                "plasma-cutting deck openings on level 2", "brazing refrigerant lines in the plant room"]
    CS_TASKS = ["inspection", "lining repair", "pump replacement", "pipe support welding", "debris removal"]

    def sample_factors(self, rng):
        reg = regime(rng)
        ptype = pick(rng, ["hot", "cs"])
        pool = [i for i, c in enumerate(self.CHANGES) if c[2] in (ptype, "both")]
        n = pick_w(rng, {0: 2, 1: 5, 2: 3})
        changes = sorted(rng.sample(pool, n))
        interval = pick(rng, [2, 4])
        since_test = pick(rng, [1, 2, 3, 4, 6]) if ptype == "cs" else pick(rng, [1, 3, 6])
        gas_test_on_permit = ptype == "cs" or chance(rng, 0.4)
        max_ext = pick(rng, [4, 6])
        requested = pick(rng, [2, 3, 3, 4, 4, 5, 6])
        issuer = pick_w(rng, {"original_on_site": 4, "night_issuer": 4, "none_until_morning": 1.5})
        handover = pick_w(rng, {"written_signed": 4, "written_unsigned": 2, "verbal": 3, "none": 1})
        crew_hours = pick(rng, [8, 8, 9, 10, 10])
        material_keys = [i for i in pool if self.CHANGES[i][3]]
        immaterial = [i for i in pool if not self.CHANGES[i][3]]
        if reg == "clear":
            if chance(rng, 0.5):
                changes, handover, issuer = [pick(rng, immaterial)] if chance(rng, 0.5) else [], "written_signed", \
                    "original_on_site"
                requested, since_test, crew_hours = min(requested, max_ext), 1, pick(rng, [9, 10])
                if self.CHANGES[changes[0]][0] == "new_crew" if changes else False:
                    changes = []
            else:
                changes = sorted(set(changes + [pick(rng, material_keys)]))
        elif reg == "borderline":
            requested = max_ext + pick(rng, [-1, 0, 0, 1])
            since_test = interval + pick(rng, [-1, 0, 1]) if ptype == "cs" else since_test
            crew_hours = pick(rng, [9, 10, 11])
            requested = min(requested, 14 - crew_hours + pick(rng, [0, 0, 1]))
        else:  # conflicting: a tidy written handover with one material change buried in it, or the reverse
            if chance(rng, 0.5):
                handover = "written_signed"
                changes = sorted(set([pick(rng, material_keys)] + ([pick(rng, immaterial)] if chance(rng, 0.5) else [])))
            else:
                handover = pick(rng, ["verbal", "none"])
                changes = [pick(rng, immaterial)] if chance(rng, 0.6) else []
        if any(self.CHANGES[i][0] == "new_crew" for i in changes):
            crew_hours = pick(rng, [0, 1, 1, 2])          # the night crew has only just started
        if ptype == "hot":
            work, site = pick(rng, self.HOT_WORK), _site(rng, "building")
        else:
            sp = pick(rng, SPACES)
            work, site = f"{pick(rng, self.CS_TASKS)} in the {_space_text(sp, rng)}", _space_site(sp, rng)
        return {"regime": reg, "permit_type": ptype, "work": work, "changes": changes, "retest_interval_h": interval,
                "hours_since_test": since_test, "gas_test_on_permit": gas_test_on_permit, "max_extension_h": max_ext,
                "requested_h": requested, "issuer": issuer, "handover": handover, "crew_hours_at_request": crew_hours,
                "site": site,
                "sub": _sub(rng, "steel" if ptype == "hot" else "utility"), "night_super": person(rng),
                "day_issuer": person(rng), "permit": _permit_no(rng, "HW" if ptype == "hot" else "CSE"),
                "date": rand_date(rng)}

    def _material(self, f):
        return any(self.CHANGES[i][3] for i in f["changes"])

    def _keys(self, f):
        return {self.CHANGES[i][0] for i in f["changes"]}

    def _retest(self, f):
        k = self._keys(f)
        if f["permit_type"] == "cs":
            return f["hours_since_test"] >= f["retest_interval_h"] or bool(k & {"new_crew", "solvent_adjacent",
                                                                                  "blower_moved", "pump_running"})
        return "solvent_adjacent" in k or (f["gas_test_on_permit"] and f["hours_since_test"] >= f["retest_interval_h"])

    def intended(self, f):
        k = self._keys(f)
        over_hours = f["crew_hours_at_request"] + f["requested_h"] > 14
        hq = {"written_signed": 3, "written_unsigned": 2, "verbal": 1, "none": 0}[f["handover"]]
        if f["issuer"] == "none_until_morning" or f["requested_h"] > f["max_extension_h"] or over_hours:
            rule = "refuse_and_stop_for_the_night"
        elif self._material(f):
            rule = "issue_new_permit"
        elif self._retest(f) or "new_crew" in k or hq <= 1:
            rule = "extend_after_retest_and_briefing"
        else:
            rule = "extend_as_is"
        risk = 1 if (f["permit_type"] == "cs" or "torch" in f["work"]) else 0
        risk += sum(1 for i in f["changes"] if self.CHANGES[i][3])
        risk += 1 if f["crew_hours_at_request"] >= 12 else 0
        risk += 1 if hq <= 1 else 0
        return {"extension_ruling": rule, "handover_quality": hq, "residual_risk": _clamp(risk),
                "conditions_changed_materially": "yes" if self._material(f) else "no",
                "retest_required": "yes" if self._retest(f) else "no"}

    def _procedure(self, f):
        return (f"Site procedure SSP-03 (permits at shift change): a permit expires at the end of the shift it was "
                f"issued for. The issuer, or an authorised night issuer, may extend it once by up to "
                f"{f['max_extension_h']} hours after a retest where testing applies and a re-briefing of the incoming "
                f"crew, if conditions are unchanged. A material change (new hazards nearby, changed fire watch or "
                f"protection, lost ventilation, new fume sources) voids the permit and needs a new permit. Atmospheric "
                f"and gas tests are repeated at least every {f['retest_interval_h']} hours and after any change of "
                f"crew. Nobody works more than 14 hours in a day.")

    def _change_lines(self, f):
        return [self.CHANGES[i][1] for i in f["changes"]] or ["no changes reported"]

    def _test_line(self, f):
        if f["permit_type"] == "cs" or f["gas_test_on_permit"]:
            return f"last gas test {f['hours_since_test']} h ago, clean"
        return "no gas testing on this permit"

    def render(self, f, rng, surface):
        ds = _ds(rng)
        kind = "hot work" if f["permit_type"] == "hot" else "confined space entry"
        issuer = {"original_on_site": f"{f['day_issuer']} (original issuer) still on site",
                  "night_issuer": "authorised night issuer on duty",
                  "none_until_morning": "no authorised issuer on site until 06:00"}[f["issuer"]]
        hand = {"written_signed": "written handover signed by both supervisors", "written_unsigned": "written handover, "
                "not signed by the incoming supervisor", "verbal": "verbal handover at the gate",
                "none": "no handover; the day supervisor had left"}[f["handover"]]
        if surface == "json":
            return dumps(prune({
                "permit": {"number": f["permit"], "type": kind, "site": f["site"], "contractor": f["sub"],
                           "work": f["work"], "issued_for": "day shift 07:00-17:00", "date": f["date"].isoformat()},
                "extension_request": {"requested_by": f["night_super"], "hours": f["requested_h"],
                                      "crew_hours_worked_so_far": f["crew_hours_at_request"]},
                "since_issue": {"changes": self._change_lines(f), "testing": self._test_line(f), "handover": hand},
                "issuer_availability": issuer,
                "procedure": self._procedure(f),
            }))
        if surface == "handover_sheet":
            return "\n".join([f"SHIFT HANDOVER -- {kind.upper()} PERMIT {f['permit']}", kv_lines([
                ("Site", f["site"]), ("Contractor", f["sub"]), ("Date", fmt_date(f["date"], rng, ds)),
                ("Work", f["work"]), ("Permit valid", "day shift 07:00-17:00"),
                ("Extension asked", f"{f['requested_h']} h, by {f['night_super']}"),
                ("Crew hours so far", f["crew_hours_at_request"]), ("Testing", self._test_line(f)),
                ("Handover", hand), ("Issuer", issuer)], rng), "", "Changes since issue:",
                bullets(self._change_lines(f), rng), "", self._procedure(f)])
        body = [f"Hi, it's {f['night_super'].split()[0]} on nights at {f['site']}.",
                f"We want to keep going on the {f['work']} under {f['permit']} for another {f['requested_h']} hours.",
                f"The crew has done {f['crew_hours_at_request']} hours today.",
                "Since this morning: " + "; ".join(self._change_lines(f)) + ".",
                f"{self._test_line(f)[0].upper()}{self._test_line(f)[1:]}.", f"{hand[0].upper()}{hand[1:]}.",
                f"{issuer[0].upper()}{issuer[1:]}.",
                pick_static(f["permit"], ["Can you sign it off?", "Need an answer in the next half hour.",
                                          "Everything else is as per the permit."])]
        return (f"[17:05] message to site safety ({kind} permit {f['permit']})\n\n"
                + noisy(join_sentences(body), rng, 0.15) + "\n\n" + self._procedure(f))

    def questions(self):
        return {
            "extension_ruling": choice("How should the extension request be decided?", {
                "extend_after_retest_and_briefing": "Extend once the retest and a briefing of the incoming crew are done.",
                "extend_as_is": "Extend the permit as it stands.",
                "issue_new_permit": "Do not extend; assess the work again and issue a new permit.",
                "refuse_and_stop_for_the_night": "Refuse, and stop the work until the day shift.",
            }),
            "handover_quality": score("How good was the handover between shifts?", [
                "None: nobody handed over.",
                "Thin: a verbal handover with nothing written.",
                "Fair: written, but not acknowledged by the incoming supervisor.",
                "Good: written and signed by both supervisors.",
            ]),
            "residual_risk": score("If the work continued tonight as requested, how much risk would remain?", [
                "Low: same conditions, rested crew, clear handover.",
                "Moderate: one factor has worsened.",
                "High: several factors have worsened.",
                "Very high: changed hazards with a tired crew or a poor handover.",
            ]),
            "conditions_changed_materially": noul(
                "Conditions have changed materially since the permit was issued.",
                yes="A change the procedure treats as material has happened: new nearby hazards, changed fire watch or "
                    "protection, lost ventilation or new fume sources.",
                no="Nothing has changed, or only things the procedure does not treat as material."),
            "retest_required": noul("A new atmospheric or gas test is required before work continues."),
        }


# =========================================================================== schema 4: field audit of an active permit
class ActivePermitFieldAudit(Schema):
    name = "active_permit_field_audit"
    description = "A safety officer's field audit of an active hot work or confined space permit that found deviations."
    surfaces = {"json": 0.5, "audit_checklist": 0.25, "safety_observation_report": 0.25}

    # (key, text, permit type, severity 0-3, stop trigger, deliberate defeat)
    DEVIATIONS = [
        ("not_posted", "the permit was not posted at the work location", "hot", 0, False, False),
        ("box_unticked", "the 'combustibles removed' box on the permit was left unticked", "hot", 0, False, False),
        ("ext_tag", "the extinguisher's inspection tag was 14 months old", "hot", 1, False, False),
        ("blanket_gap", "fire blankets left a pallet of packaging exposed about 5 m from the sparks", "hot", 2, False,
         False),
        ("watch_absent", "no fire watch present; the welder was working alone", "hot", 3, True, False),
        ("watch_busy", "the fire watch was also cutting stock at the saw", "hot", 2, False, False),
        ("expired", "hot work was still going 40 minutes after the permit expired", "hot", 2, True, False),
        ("sprinkler_closed", "a sprinkler valve was closed without an impairment tag", "hot", 3, True, True),
        ("attendant_away", "the attendant had left the entry to fetch tools from the van", "cs", 3, True, False),
        ("monitor_top", "the gas monitor was clipped to the attendant at the top, not to the entrant", "cs", 2, False,
         False),
        ("alarm_taped", "the continuous monitor's alarm was muted with tape", "cs", 3, True, True),
        ("log_behind", "the entry log was behind; one entrant had not signed in", "cs", 1, False, False),
        ("blower_off", "the ventilation blower had been switched off to cut the noise", "cs", 3, True, True),
        ("lock_removed", "another trade's worker had removed the lock from the inlet valve", "cs", 3, True, True),
        ("line_unclipped", "the winch line was rigged but not clipped to the entrant's harness", "cs", 2, False, False),
        ("wrong_number", "the rescue phone number on the permit was out of date", "cs", 1, False, False),
    ]

    def sample_factors(self, rng):
        reg = regime(rng)
        ptype = pick(rng, ["hot", "cs"])
        pool = [i for i, d in enumerate(self.DEVIATIONS) if d[2] == ptype]
        n = pick_w(rng, {1: 5, 2: 4, 3: 1})
        devs = sorted(rng.sample(pool, n))
        prior = pick_w(rng, {0: 4, 1: 3, 2: 2, 3: 1, 4: 1})
        same_type_before = prior > 0 and chance(rng, 0.35)
        corrected = chance(rng, 0.6)
        if reg == "clear":
            if chance(rng, 0.5):
                devs = [pick(rng, [i for i in pool if self.DEVIATIONS[i][3] == 0 or
                                   (self.DEVIATIONS[i][3] == 1 and ptype == "cs")])]
                prior, same_type_before = 0, False
            else:
                devs = sorted(set([pick(rng, [i for i in pool if self.DEVIATIONS[i][5]])] + devs[:1]))
                prior, same_type_before = pick(rng, [2, 3, 4]), True
        elif reg == "borderline":
            devs = [pick(rng, [i for i in pool if self.DEVIATIONS[i][3] in (1, 2)])]
            if chance(rng, 0.4):
                devs = sorted(set(devs + [pick(rng, [i for i in pool if self.DEVIATIONS[i][3] <= 1])]))
        else:  # conflicting: a crew with a clean record and one serious lapse, or a poor record and trivia
            if chance(rng, 0.5):
                devs, prior, same_type_before = [pick(rng, [i for i in pool if self.DEVIATIONS[i][4]])], 0, False
                corrected = True
            else:
                devs = [pick(rng, [i for i in pool if self.DEVIATIONS[i][3] <= 1])]
                prior, same_type_before = pick(rng, [3, 4]), chance(rng, 0.5)
        return {"regime": reg, "permit_type": ptype, "deviations": devs, "prior_findings_90d": prior,
                "same_type_before": same_type_before, "corrected_on_spot": corrected,
                "entrant_inside": ptype == "cs" and chance(rng, 0.7),
                "site": _site(rng, "building" if ptype == "hot" else "utility"),
                "sub": _sub(rng, pick(rng, ["steel", "mech"]) if ptype == "hot" else "utility"),
                "auditor": person(rng), "crew_lead": person(rng),
                "permit": _permit_no(rng, "HW" if ptype == "hot" else "CSE"), "date": rand_date(rng),
                "time": f"{rng.randint(8, 16):02d}:{pick(rng, ['05', '20', '35', '50'])}"}

    def intended(self, f):
        ds = [self.DEVIATIONS[i] for i in f["deviations"]]
        stop = any(d[4] for d in ds)
        n_stop = sum(1 for d in ds if d[4])
        deliberate = any(d[5] for d in ds)
        sev = max(d[3] for d in ds)
        p = f["prior_findings_90d"]
        rep = 0 if p == 0 else 1 if p == 1 else 2 if p <= 3 else 3
        if f["same_type_before"]:
            rep = _clamp(rep + 1)
        if stop and (deliberate or n_stop >= 2 or (rep >= 2 and f["same_type_before"])):
            resp = "revoke_permit_and_investigate"
        elif stop:
            resp = "suspend_until_reverified"
        elif sev >= 1:
            resp = "correct_on_the_spot"
        else:
            resp = "record_observation"
        if deliberate and rep >= 2:
            esc = "remove_individuals_from_site"
        elif stop or (rep >= 3 and sev >= 2):
            esc = "crew_stand_down_and_retraining"
        elif sev >= 2 or rep >= 2:
            esc = "written_notice_to_subcontractor"
        else:
            esc = "supervisor_coaching"
        return {"audit_response": resp, "followup_escalation": esc, "deviation_severity": sev,
                "repeat_pattern": rep, "stop_work_required": "yes" if stop else "no"}

    def _procedure(self, f):
        return ("Site procedure SSP-11 (permit audits): stop the work at once where there is no attendant at an "
                "occupied permit space, monitoring or ventilation is not operating, isolation has been removed, hot "
                "work is under way without a fire watch or with fire protection shut off, or work continues without "
                "a valid permit; work resumes only after the issuer re-verifies the permit. Deliberately defeating a "
                "safety device is grounds for removal from site. Other deviations are corrected on the spot and "
                "recorded; repeated findings are escalated to the subcontractor in writing.")

    def _history(self, f):
        p = f["prior_findings_90d"]
        h = "no permit findings against this crew in the last 90 days" if p == 0 else \
            ("one permit finding" if p == 1 else f"{p} permit findings") + " against this crew in the last 90 days"
        if f["same_type_before"]:
            h += ", including the same kind of lapse"
        return h

    def render(self, f, rng, surface):
        ds = _ds(rng)
        kind = "hot work" if f["permit_type"] == "hot" else "confined space entry"
        devs = [self.DEVIATIONS[i][1] for i in f["deviations"]]
        fixed = ("crew corrected what could be corrected while the auditor was there" if f["corrected_on_spot"]
                 else "crew lead disputed the findings; nothing changed during the visit")
        inside = "an entrant was inside the space at the time" if f["entrant_inside"] else None
        if surface == "json":
            return dumps(prune({
                "audit": {"permit": f["permit"], "type": kind, "site": f["site"], "contractor": f["sub"],
                          "auditor": f["auditor"], "crew_lead": f["crew_lead"],
                          "when": f"{f['date'].isoformat()} {f['time']}"},
                "findings": devs, "at_the_time": inside, "on_the_spot": fixed, "crew_history": self._history(f),
                "procedure": self._procedure(f),
            }))
        if surface == "audit_checklist":
            rows = [[str(k + 1), d, "FAIL"] for k, d in enumerate(devs)]
            ok_items = pick_static(f["permit"], [["permit signed by issuer", "PPE worn"],
                                                 ["isolation register checked", "barricade in place"],
                                                 ["work area tidy", "permit within validity"]])
            rows += [[str(len(rows) + k + 1), x, "pass"] for k, x in enumerate(ok_items)]
            return "\n".join([f"PERMIT FIELD AUDIT -- {kind.upper()}", kv_lines([
                ("Permit", f["permit"]), ("Site", f["site"]), ("Contractor", f["sub"]),
                ("Auditor", f["auditor"]), ("Date / time", f"{fmt_date(f['date'], rng, ds)} {f['time']}")], rng), "",
                table(["#", "Item", "Result"], rows, rng), "",
                f"Notes: {fixed}." + (f" {inside[0].upper()}{inside[1:]}." if inside else ""),
                f"History: {self._history(f)}.", "", self._procedure(f)])
        body = [f"During a walk-round at {f['time']} I audited {kind} permit {f['permit']} ({f['sub']}, crew lead "
                f"{f['crew_lead']}).", "I found that " + "; ".join(devs) + ".",
                f"{inside[0].upper()}{inside[1:]}." if inside else "", f"{fixed[0].upper()}{fixed[1:]}.",
                f"For context: {self._history(f)}."]
        return (f"SAFETY OBSERVATION REPORT -- {f['site']}\nBy {f['auditor']}, {fmt_date(f['date'], rng, ds)}\n\n"
                + noisy(join_sentences(body), rng, 0.1) + "\n\n" + self._procedure(f))

    def questions(self):
        return {
            "audit_response": choice("What should the auditor do about the permit now?", {
                "correct_on_the_spot": "Have the crew correct the items now and let the work continue.",
                "record_observation": "Record the observation; nothing needs to change before work continues.",
                "revoke_permit_and_investigate": "Revoke the permit and open an investigation.",
                "suspend_until_reverified": "Stop the work until the issuer re-verifies the permit.",
            }),
            "followup_escalation": choice("What follow-up is warranted with the crew and its employer?", {
                "crew_stand_down_and_retraining": "Stand the crew down for a briefing and retraining before more permit work.",
                "remove_individuals_from_site": "Remove the people responsible from the site.",
                "supervisor_coaching": "Coaching by the crew's own supervisor.",
                "written_notice_to_subcontractor": "A written notice to the subcontractor's management.",
            }),
            "deviation_severity": score("How serious is the worst deviation found?", [
                "Administrative: paperwork or posting only.",
                "Minor: a control weakened, no immediate danger.",
                "Serious: an important control missing or misapplied.",
                "Critical: people were exposed to a hazard with a key control absent or defeated.",
            ]),
            "repeat_pattern": score("How much does this crew's record suggest a repeating problem?", [
                "None: a clean recent record.",
                "Some: one earlier finding.",
                "Marked: several earlier findings, or a repeat of the same lapse.",
                "Persistent: frequent findings including the same lapse.",
            ]),
            "stop_work_required": noul(
                "The procedure requires the work to stop immediately.",
                yes="At least one finding is on the procedure's stop list.",
                no="None of the findings is on the stop list, so they can be corrected while work continues."),
        }


# =========================================================================== schema 5: re-entry after a gas alarm
class GasAlarmReentryReview(Schema):
    name = "gas_alarm_reentry_review"
    description = "A confined space evacuated on a gas alarm; whether and how entrants may go back in."
    surfaces = {"json": 0.5, "incident_log": 0.25, "supervisor_report": 0.25}

    GASES = {"h2s": ("H2S", "ppm", 10), "lel": ("flammable gas", "% LEL", 10), "co": ("CO", "ppm", 25),
             "o2": ("oxygen", "%", 19.5)}
    # (key, text, identified?, control-failure base 0-3)
    CAUSES = [
        ("sludge_release", "raking settled sludge released a pocket of gas", True, 0),
        ("valve_passing", "the inlet isolation valve was passing; flow and gas came in from upstream", True, 2),
        ("solvent_unplanned", "an entrant started using a solvent cleaner that was not on the permit", True, 2),
        ("adjacent_pump", "the plant operator restarted a pump in the adjoining station, pushing gas into the space",
         True, 2),
        ("sensor_fault", "the monitor failed its bump test after the alarm; sensor drift suspected", True, 0),
        ("unknown", "no cause found; readings fell back to zero within two minutes", False, 1),
        ("unplanned_purge", "the blower was shut off during a generator refuel and not restarted before work resumed",
         True, 2),
    ]
    CAUSES_BY_GAS = {"h2s": {0: 3, 1: 2, 3: 1.5, 4: 1.5, 5: 3}, "lel": {0: 2, 1: 1, 2: 2, 4: 1.5, 5: 3},
                     "co": {6: 3, 4: 1.5, 5: 3}, "o2": {6: 3, 4: 1.5, 5: 3, 1: 1}}
    SLUDGE_SPACES = ("wet_well", "sanitary_manhole", "digester", "grit_chamber")

    def sample_factors(self, rng):
        reg = regime(rng)
        gas = pick_w(rng, {"h2s": 4, "lel": 2.5, "co": 1.5, "o2": 2})
        ci = pick_w(rng, self.CAUSES_BY_GAS[gas])
        minutes = pick(rng, [1, 2, 3, 5, 8, 12])
        symptoms = pick_w(rng, {"none": 6, "headache": 2, "dizziness": 1, "collapse": 0.3})
        vent_on = chance(rng, 0.75)
        now = pick_w(rng, {"clean": 7, "still_elevated": 2})
        second = pick_w(rng, {"clean": 3, "none": 5, "elevated": 1})
        bump_after = pick_w(rng, {"passed": 5, "not_done": 3, "failed": 1})
        mult = pick(rng, [1.1, 1.5, 2.2, 3.5, 6.0])
        if reg == "clear":
            if chance(rng, 0.5):
                ci, minutes, symptoms, vent_on, now, bump_after, mult = 0, 1, "none", True, "clean", "passed", 1.2
            else:
                ci, minutes, symptoms, mult = pick(rng, [1, 3, 6]), pick(rng, [8, 12]), pick(rng, ["dizziness",
                                                                                                    "collapse"]), 5.0
        elif reg == "borderline":
            mult = pick(rng, [1.1, 1.4, 2.0])
            minutes = pick(rng, [2, 3, 4])
        else:  # conflicting: an alarming reading the instrument may have invented, or a calm crew after a real leak
            if chance(rng, 0.5):
                ci, bump_after = 4, "failed"
                second = pick(rng, ["clean", "none"])
            else:
                ci, symptoms, now = pick(rng, [1, 3]), "none", "clean"
        if ci not in self.CAUSES_BY_GAS[gas]:
            gas = pick(rng, [g for g, cs in self.CAUSES_BY_GAS.items() if ci in cs])
        cause = self.CAUSES[ci]
        names = self.SLUDGE_SPACES if cause[0] in ("sludge_release", "adjacent_pump") else \
            ("wet_well", "sanitary_manhole", "digester", "grit_chamber", "valve_vault", "storm_manhole", "water_tank",
             "utility_tunnel")
        sp = pick(rng, [x for x in SPACES if x[0] in names])
        if now == "still_elevated" and second == "clean":
            second = pick(rng, ["none", "elevated"])
        if cause[0] == "sensor_fault":
            bump_after = "failed"
        if cause[0] == "unplanned_purge":
            vent_on = False
        name, unit, limit = self.GASES[gas]
        if gas == "o2":
            peak = round(max(15.5, 19.5 - (mult - 1) * 1.2), 1)
        elif gas == "lel":
            peak = round(min(60, limit * mult), 0)
        else:
            peak = round(limit * mult, 0)
        return {"regime": reg, "space_text": _space_text(sp, rng), "gas": gas, "peak": peak, "multiple": mult,
                "minutes_exposed": minutes, "symptoms": symptoms, "cause": ci, "vent_on": vent_on,
                "readings_now": now, "second_instrument": second, "bump_after": bump_after,
                "entrants": pick(rng, [1, 2, 2, 3]), "site": _space_site(sp, rng), "sub": _sub(rng, "utility"),
                "supervisor": person(rng), "permit": _permit_no(rng, "CSE"), "date": rand_date(rng),
                "alarm_time": f"{rng.randint(8, 15):02d}:{rng.randint(0, 59):02d}"}

    def _sensor_confirmed(self, f):
        return self.CAUSES[f["cause"]][0] == "sensor_fault" and f["second_instrument"] == "clean"

    def intended(self, f):
        cause = self.CAUSES[f["cause"]]
        identified = cause[2] and (cause[0] != "sensor_fault" or self._sensor_confirmed(f))
        if self._sensor_confirmed(f):
            exp = 0
        else:
            m = f["multiple"]
            exp = 0 if (m < 1.3 and f["minutes_exposed"] <= 2) else 1 if m < 2 else 2 if m < 4 else 3
            if f["minutes_exposed"] >= 5:
                exp += 1
            if f["symptoms"] in ("dizziness", "headache"):
                exp = max(exp, 2)
            if f["symptoms"] == "collapse":
                exp = 3
        exp = _clamp(exp)
        depth = cause[3]
        if not f["vent_on"] and cause[0] != "unplanned_purge":
            depth += 1
        if f["minutes_exposed"] >= 5:
            depth += 1
        depth = _clamp(depth)
        medical = f["symptoms"] != "none" or (not self._sensor_confirmed(f) and f["multiple"] >= 2
                                              and f["minutes_exposed"] >= 3)
        if f["readings_now"] == "still_elevated" or not cause[2] or f["symptoms"] == "collapse" \
                or f["second_instrument"] == "elevated":
            dec = "keep_space_closed_pending_investigation"
        elif cause[0] == "sensor_fault":
            dec = "reenter_under_existing_permit" if self._sensor_confirmed(f) else "verify_with_second_instrument"
        elif f["bump_after"] != "passed" and f["second_instrument"] != "clean":
            dec = "verify_with_second_instrument"
        elif depth >= 1:
            dec = "cancel_and_reissue_with_new_controls"
        else:
            dec = "reenter_under_existing_permit"
        return {"reentry_decision": dec, "exposure_seriousness": exp, "control_failure_depth": depth,
                "alarm_cause_identified": "yes" if identified else "no",
                "medical_evaluation_needed": "yes" if medical else "no"}

    def _procedure(self, f):
        return ("Site procedure SSP-09, section 8 (alarms): any alarm suspends the permit and everyone leaves the "
                "space. Re-entry needs the cause identified and controlled, two clean readings 10 minutes apart from "
                "a bump-tested instrument, and a re-issued permit with revised controls wherever a planned control "
                "failed or the hazard was not foreseen. Entrants with symptoms, or exposed above twice the alarm "
                "level for more than a couple of minutes, are medically evaluated before returning to work.")

    def _peak_text(self, f):
        name, unit, limit = self.GASES[f["gas"]]
        if f["gas"] == "o2":
            return f"oxygen fell to {f['peak']}% (alarm at {limit}%)"
        return f"{name} peaked at {int(f['peak'])} {unit} (alarm at {limit} {unit})"

    def _texts(self, f):
        sym = {"none": "no symptoms reported", "headache": "one entrant reports a headache",
               "dizziness": "one entrant was dizzy on reaching the top", "collapse": "one entrant collapsed at the "
               "ladder and was hauled out by the winch"}[f["symptoms"]]
        now = ("readings at the hatch now O2 20.9%, LEL 0%, H2S 0 ppm, CO 0 ppm" if f["readings_now"] == "clean"
               else "readings at the hatch still above the alarm level an hour later")
        second = {"clean": "a second, freshly bump-tested monitor reads clean throughout the space",
                  "none": "no second instrument used yet",
                  "elevated": "a second monitor lowered into the space also reads above the alarm level"}[
            f["second_instrument"]]
        bump = {"passed": "the original monitor passed a bump test after the alarm", "not_done": "the original monitor "
                "has not been bump-tested since the alarm", "failed": "the original monitor failed a bump test after "
                "the alarm"}[f["bump_after"]]
        vent = "blower running at the time" if f["vent_on"] else "blower not running at the time"
        return sym, now, second, bump, vent

    def render(self, f, rng, surface):
        ds = _ds(rng)
        sym, now, second, bump, vent = self._texts(f)
        cause = self.CAUSES[f["cause"]][1]
        who = {1: "the entrant was", 2: "both entrants were"}.get(f["entrants"], f"all {f['entrants']} entrants were")
        out = f"{who} out after about {f['minutes_exposed']} min in the atmosphere"
        if surface == "json":
            return dumps(prune({
                "event": {"permit": f["permit"], "site": f["site"], "contractor": f["sub"], "space": f["space_text"],
                          "alarm": f"{f['date'].isoformat()} {f['alarm_time']}", "peak": self._peak_text(f),
                          "evacuation": out, "ventilation": vent},
                "entrants": sym,
                "since": {"cause_found": cause, "current_readings": now, "instrument_check": bump,
                          "second_instrument": second},
                "request": f"{f['supervisor']} asks to resume the work this shift",
                "procedure": self._procedure(f),
            }))
        if surface == "incident_log":
            h, mnt = map(int, f["alarm_time"].split(":"))
            t = datetime(f["date"].year, f["date"].month, f["date"].day, h, mnt)
            log = [(t, f"alarm on entrant monitor: {self._peak_text(f)}; {vent}"),
                   (t + timedelta(minutes=f["minutes_exposed"]), out),
                   (t + timedelta(minutes=f["minutes_exposed"] + 2), sym),
                   (t + timedelta(minutes=25), bump), (t + timedelta(minutes=35), f"cause: {cause}"),
                   (t + timedelta(minutes=50), second), (t + timedelta(minutes=60), now),
                   (t + timedelta(minutes=65), f"{f['supervisor']} requests re-entry to finish the job")]
            return (f"INCIDENT LOG -- permit {f['permit']}, {f['space_text']} ({f['site']})\n"
                    + "\n".join(f"{_hhmm(x)}  {y}" for x, y in log) + "\n\n" + self._procedure(f))
        body = [f"At {f['alarm_time']} on {fmt_date(f['date'], rng, ds)} the entrant monitor alarmed in the "
                f"{f['space_text']}: {self._peak_text(f)}.", f"{out[0].upper()}{out[1:]}; {vent}.",
                f"{sym[0].upper()}{sym[1:]}.", f"Cause: {cause}.", f"{bump[0].upper()}{bump[1:]}.",
                f"{second[0].upper()}{second[1:]}.", f"{now[0].upper()}{now[1:]}.",
                pick_static(f["permit"], ["We'd like to get back in and finish today.",
                                          "Crew is ready to go back in.",
                                          "I think it was a one-off; requesting re-entry."])]
        return (f"SUPERVISOR REPORT -- {f['sub']}\nTo: site safety\nRe: alarm under permit {f['permit']}\n\n"
                + noisy(join_sentences(body), rng, 0.12) + "\n\n" + self._procedure(f))

    def questions(self):
        return {
            "reentry_decision": choice("What should happen before anyone goes back into the space?", {
                "cancel_and_reissue_with_new_controls": "Cancel the permit and re-issue it with revised controls first.",
                "keep_space_closed_pending_investigation": "Keep the space closed until the cause is investigated and "
                                                           "the atmosphere is under control.",
                "reenter_under_existing_permit": "Allow re-entry under the existing permit once clean readings are "
                                                 "confirmed.",
                "verify_with_second_instrument": "Check the atmosphere with a second, bump-tested instrument before "
                                                 "deciding.",
            }),
            "exposure_seriousness": score("How serious was the entrants' exposure?", [
                "Negligible: brief and barely above the alarm level, or not real.",
                "Minor: above the alarm level briefly, no symptoms.",
                "Significant: well above the alarm level, or symptoms reported.",
                "Severe: far above it for minutes, or someone was overcome.",
            ]),
            "control_failure_depth": score("How badly did the planned controls fail?", [
                "Not at all: a foreseen transient that the controls handled.",
                "Partly: the plan underestimated a hazard.",
                "Clearly: a planned control was missing, bypassed or not applied.",
                "Badly: several controls failed or people stayed in too long.",
            ]),
            "alarm_cause_identified": noul(
                "The cause of the alarm has been identified.",
                yes="A specific source has been found, or an instrument fault has been confirmed by an independent "
                    "check.",
                no="No cause has been found, or the suspected instrument fault has not been confirmed."),
            "medical_evaluation_needed": noul(
                "The entrants need a medical evaluation before returning to work."),
        }


WORKFLOW = Workflow(
    name="hot_work_and_confined_space_permits",
    title="Hot work and confined space permits",
    description="Site-safety permit-to-work decisions for hot work and confined space entry against the site's written "
                "permit procedures.",
    domain="construction / site safety",
    schemas=[HotWorkPermitRequest(), ConfinedSpaceEntryPermit(), PermitShiftExtension(), ActivePermitFieldAudit(),
             GasAlarmReentryReview()],
    notes="Fictional sites, contractors and procedures; the procedure is quoted in every state. Permit-to-work only: no "
          "municipal building permits, injury claims or insurance questions.",
)
