"""Process management of change: chemical-plant change requests judged against a (fictional) site's written MOC standards.

Process-safety decisions at invented chemical plants: whether a replacement is truly "in kind" or needs a management
-of-change (MOC) review, what depth of hazard review a confirmed change needs, whether a modified unit is ready for
start-up after its pre-startup safety review (PSSR), whether a temporary change (an interlock bypass, a leak clamp, a
rental chiller) may be extended, and whether an organisational change (fewer board operators, a merged control
room, a retiring specialist) can be approved. Companies, units, tags and every site standard are invented, shaped
like real process-safety-management practice; the governing standard is quoted in each state and its limits vary
between sites. Each schema draws a `regime` first so most cases sit near a limit, carry signals that disagree (a
requester's "like for like" against the data sheet), or miss a fact.

Every schema also has `intended(f)`: the answer a careful reviewer would most likely give from what the state
shows. It is never rendered and never used as a label; it exists so that the answer spread of every question can
be audited before labelling.
"""
from __future__ import annotations

from datetime import date, timedelta

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
    maybe,
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
    show,
    table,
)

# --------------------------------------------------------------------------- shared pools
PLANT_A = ["Basalt", "Copperline", "Saltmarsh", "Driftwood", "Ironbark", "Fernhill", "Quarry Point", "Redfern",
           "Ashgrove", "Westbrook", "Bluestem", "Lakeshore", "Cinder Bay", "Halvorsen"]
PLANT_B = ["Chemical", "Chemicals", "Chemical Co.", "Process Industries", "Chemical Works", "Industrial Chemicals"]
TOWNS = ["Port Ellery", "Marlow Flats", "Kessock", "Brandon Reach", "Tolliver", "Granby Point", "Ashby Marsh",
         "New Hollis", "Carrick Bend", "Oldwharf"]
UNITS = {
    # unit: (process chemical, hazard class)
    "chlorine liquefaction": ("chlorine", "toxic"),
    "ammonia refrigeration": ("anhydrous ammonia", "toxic"),
    "HF alkylation": ("hydrofluoric acid", "toxic"),
    "amine treating": ("sour gas with hydrogen sulfide", "toxic_flammable"),
    "solvent recovery": ("toluene", "flammable"),
    "methanol tank farm": ("methanol", "flammable"),
    "hydrogen compression": ("hydrogen", "flammable"),
    "hydrogenation reactors": ("hydrogen and aldehyde feed", "flammable"),
    "polyol reactors": ("propylene oxide", "reactive"),
    "vinyl acetate storage": ("vinyl acetate monomer", "reactive"),
    "caustic evaporation": ("50% sodium hydroxide", "corrosive"),
    "cooling water treatment": ("sodium hypochlorite", "corrosive"),
}
HAZ = {"toxic": 3, "toxic_flammable": 3, "reactive": 3, "flammable": 2, "corrosive": 1}
DATE_STYLE = ["iso", "iso", "us", "long", "eu"]


def _company(rng):
    return f"{pick(rng, PLANT_A)} {pick(rng, PLANT_B)}"


def _unit(rng, weights=None):
    names = list(UNITS)
    u = pick(rng, names, weights)
    return u, UNITS[u][0], UNITS[u][1]


def _moc_no(rng, d):
    return pick(rng, [f"MOC-{d.year}-{rng.randint(1, 999):04d}", f"MOC {d.year % 100}-{rng.randint(100, 999)}",
                      f"CR-{rng.randint(10000, 99999)}"])


# =========================================================================== schema 1
# item: (tag prefix, original specification, [(proposed replacement, deviation 0-3, changes PSI?)], safety-related?)
ITEMS = {
    "pump_mechanical_seal": ("P-", "single cartridge seal, carbon vs silicon-carbide faces, FKM O-rings, flush plan 11", [
        ("identical part number from the same vendor", 0, False),
        ("vendor's superseded part number; vendor letter states same faces, elastomers and flush plan", 1, False),
        ("carbon vs tungsten-carbide faces, FKM O-rings, plan 11", 1, False),
        ("same seal with EPDM O-rings because stores had no FKM; compatibility with the process fluid not checked", 3, True),
        ("dual pressurised seal with a plan 53A barrier pot", 2, True)], False),
    "flange_gasket": ("V-", "spiral-wound 316/graphite gasket, Class 300", [
        ("spiral-wound 316/graphite, Class 300, second manufacturer to the same standard", 1, False),
        ("spiral-wound 316L/graphite, Class 300", 1, False),
        ("spiral-wound 316/PTFE-filler, Class 300 (lower temperature limit than graphite)", 2, False),
        ("compressed-fibre sheet gasket rated Class 150", 3, True),
        ("identical gasket from the same supplier", 0, False)], False),
    "pressure_relief_valve": ("PSV-", "conventional relief valve, orifice K, set 150 psig, rated 18,400 lb/h (required relieving rate 17,200 lb/h)", [
        ("rebuilt spare of the same model, orifice, set pressure and trim", 0, False),
        ("different vendor, orifice K, set 150 psig, rated 18,100 lb/h, conventional", 1, True),
        ("orifice J, set 150 psig, rated 11,800 lb/h", 3, True),
        ("same model set at 160 psig (vessel MAWP 165 psig)", 3, True),
        ("balanced-bellows version of the same valve, same set pressure and orifice", 2, True)], True),
    "level_transmitter": ("LT-", "guided-wave radar, 0-120 in range, SIL 2 certified, input to high-level trip", [
        ("stores spare of the same model and firmware", 0, False),
        ("same model, newer firmware; vendor states the SIL certificate is unchanged", 1, False),
        ("different radar model, 0-120 in, not SIL certified", 3, True),
        ("same model configured 0-100 in", 3, True)], True),
    "control_valve_trim": ("FV-", "Cv 45 equal-percentage trim, 316 with Stellite seat", [
        ("identical trim set", 0, False),
        ("Cv 45 equal-percentage trim with a hardened 410 stainless seat", 1, False),
        ("Cv 60 linear trim", 2, True)], False),
    "catalyst_charge": ("R-", "catalyst grade NK-12 from the usual supplier", [
        ("same grade from the same supplier, new lot", 0, False),
        ("equivalent grade from a second supplier; data sheet matches activity and particle size", 1, True),
        ("higher-activity grade NK-14 from the same supplier", 2, True)], False),
    "compressor_lubricant": ("C-", "synthetic PAO oil, ISO VG 46", [
        ("same product from the same supplier", 0, False),
        ("another brand of PAO ISO VG 46 meeting the same specification", 1, False),
        ("mineral oil ISO VG 68", 2, False)], False),
    "alarm_setpoint": ("TAH-", "high-temperature alarm at 180 C (safe operating limit 190 C)", [
        ("raise the alarm to 185 C to stop nuisance alarms", 2, True),
        ("raise the alarm to 195 C", 3, True)], True),
}
VERIFY = {"none": "none on file", "vendor_letter": "vendor equivalency letter", "datasheet": "data-sheet comparison by the requester",
          "engineering_calc": "engineering check with calculation attached"}
# units where an item type makes sense (absent = any unit)
ITEM_UNITS = {"catalyst_charge": ["hydrogenation reactors", "polyol reactors"],
              "compressor_lubricant": ["ammonia refrigeration", "hydrogen compression"]}
STORAGE_UNITS = ("methanol tank farm", "vinyl acetate storage")
ADDITIVE = {"polyol reactors": "initiator", "vinyl acetate storage": "polymerisation inhibitor", "amine treating": "antifoam",
            "cooling water treatment": "biocide", "caustic evaporation": "chelating agent", "hydrogenation reactors": "catalyst promoter"}
LEAKABLE = ("pump_mechanical_seal", "flange_gasket", "pressure_relief_valve", "control_valve_trim")
URGENCY = {"stores_spare": "replenishing stores, no equipment waiting", "turnaround": "planned for the next turnaround",
           "unit_down": "unit shut down awaiting this item", "leaking": "item leaking in service; cannot be isolated without a shutdown"}


class ReplacementInKindScreening(Schema):
    name = "replacement_in_kind_screening"
    description = "A maintenance replacement screened against the site's replacement-in-kind definition: no MOC, MOC, or emergency MOC."
    surfaces = {"json": 0.5, "work_request": 0.25, "screening_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        item = pick(rng, list(ITEMS))
        prefix, orig, variants, safety = ITEMS[item]
        vi = rng.randrange(len(variants))
        unit = pick(rng, ITEM_UNITS.get(item, list(UNITS)))
        chem, haz = UNITS[unit]
        verify = pick_w(rng, {"none": 2, "vendor_letter": 3, "datasheet": 2, "engineering_calc": 2})
        urgency = pick_w(rng, {"stores_spare": 2, "turnaround": 2, "unit_down": 3, "leaking": 5 if item in LEAKABLE else 0.01})
        claim = "RIK" if variants[vi][1] <= 1 or chance(rng, 0.45) else "MOC"
        if reg == "clear":
            want = pick(rng, [0, 3])
            choices = [i for i, v in enumerate(variants) if v[1] == want] or [vi]
            vi = pick(rng, choices)
            claim = "RIK" if want == 0 else "MOC"
            verify = "engineering_calc" if want == 3 else verify
            urgency = pick(rng, ["stores_spare", "turnaround"])
        elif reg == "borderline":
            choices = [i for i, v in enumerate(variants) if v[1] in (1, 2)] or [vi]
            vi = pick(rng, choices)
        else:
            # "like for like" on the work request against a data sheet that says otherwise, or an over-cautious MOC
            if chance(rng, 0.6):
                choices = [i for i, v in enumerate(variants) if v[1] >= 2] or [vi]
                vi, claim = pick(rng, choices), "RIK"
                urgency = "leaking" if (item in LEAKABLE and chance(rng, 0.7)) else pick(rng, ["unit_down", urgency])
            else:
                choices = [i for i, v in enumerate(variants) if v[1] == 0] or [vi]
                vi, claim = pick(rng, choices), "MOC"
        if urgency == "leaking" and item not in LEAKABLE:
            urgency = "unit_down"      # only pressure-retaining items leak; a failed instrument or spent oil stops the unit
        d = rand_date(rng)
        return {"regime": reg, "company": _company(rng), "site": pick(rng, TOWNS), "unit": unit, "chemical": chem,
                "hazard": haz, "item": item, "tag": f"{prefix}{rng.randint(100, 4999)}{pick(rng, ['', '', 'A', 'B'])}",
                "original": orig, "proposed": variants[vi][0], "deviation": variants[vi][1], "psi_change": variants[vi][2],
                "safety_related": safety, "verification": verify, "urgency": urgency, "requester_claim": claim,
                "requester": person(rng), "planner": person(rng), "date": d, "wo": f"WO-{rng.randint(100000, 999999)}",
                "emergency_hours": pick(rng, [48, 72, 72, 96])}

    def intended(self, f):
        dv, haz, safety = f["deviation"], HAZ[f["hazard"]], f["safety_related"]
        if dv == 0:
            cls = "replacement_in_kind"
        elif dv == 1:
            cls = "replacement_in_kind" if (f["verification"] in ("vendor_letter", "engineering_calc") and not f["psi_change"]) else "moc_checklist_review"
        elif dv == 2:
            cls = "moc_full_hazard_review" if (haz >= 3 and safety) else "moc_checklist_review"
        else:
            cls = "moc_full_hazard_review"
        if f["urgency"] == "leaking" and dv >= 2 and haz >= 2:
            cls = "emergency_moc"
        verif = dv if not (dv == 1 and (safety or f["psi_change"])) else 2
        sig = dv if not (dv == 2 and haz <= 1) else 1
        is_rik = cls == "replacement_in_kind"
        return {"change_classification": cls, "spec_deviation_significance": sig, "verification_needed": verif,
                "psi_update_required": "yes" if f["psi_change"] else "no",
                "requester_classification_correct": "yes" if (f["requester_claim"] == "RIK") == is_rik else "no"}

    def _policy(self, f):
        return (f"Site MOC standard S-10: a replacement in kind meets the design specification of the item it replaces - "
                f"materials, ratings, set points, range, capacity and certification - and needs no MOC; an alternative "
                f"claimed as equal or better counts only with a documented engineering or vendor confirmation and no change "
                f"to process safety information. Everything else, including any set-point or alarm change, is a change: a "
                f"checklist MOC if it touches no safeguard or highly hazardous service, a full hazard review otherwise. The "
                f"emergency MOC path is for changes needed at once to control a hazard; its review is completed within "
                f"{f['emergency_hours']} h.")

    def render(self, f, rng, surface):
        st = pick(rng, DATE_STYLE)
        haz_txt = f"{f['chemical']} service ({f['hazard'].replace('_', ' and ')})"
        claim = {"RIK": "replacement in kind - no MOC needed", "MOC": "submitted as a change (MOC)"}[f["requester_claim"]]
        if surface == "json":
            return dumps(prune({
                "request": {"work_order": f["wo"], "requested_by": f["requester"], "date": fmt_date(f["date"], style="iso"),
                            "unit": f["unit"], "service": haz_txt, "tag": f["tag"], "item": f["item"].replace("_", " "),
                            "requester_classification": claim, "urgency": URGENCY[f["urgency"]]},
                "specification": {"installed": f["original"], "proposed": f["proposed"]},
                "verification": VERIFY[f["verification"]],
                "standard": self._policy(f),
            }))
        if surface == "work_request":
            rows = [("Work order", f["wo"]), ("Site", f"{f['company']}, {f['site']}"), ("Unit", f["unit"]),
                    ("Service", haz_txt), ("Tag", f"{f['tag']} ({f['item'].replace('_', ' ')})"),
                    ("Installed", f["original"]), ("Replacement", f["proposed"]), ("Verification", VERIFY[f["verification"]]),
                    ("Urgency", URGENCY[f["urgency"]]), ("Requester's call", claim), ("Requested by", f["requester"]),
                    ("Date", fmt_date(f["date"], style=st))]
            return "MAINTENANCE WORK REQUEST - MOC SCREENING\n" + kv_lines(rows, rng) + "\n\n" + self._policy(f)
        s = [f"Need to replace {f['tag']} ({f['item'].replace('_', ' ')}) on the {f['unit']} unit, {haz_txt}.",
             f"Installed: {f['original']}.", f"What we can get: {f['proposed']}.",
             f"Checks done: {VERIFY[f['verification']]}.", f"Status: {URGENCY[f['urgency']]}.",
             {"RIK": pick(rng, ["I'm treating this as like-for-like, no MOC.", "This is a straight swap as far as I can see."]),
              "MOC": pick(rng, ["Raising it as an MOC to be safe.", "Putting it through MOC since the part number changed."])}[f["requester_claim"]]]
        return (f"From: {f['requester']}\nTo: {f['planner']} (MOC coordinator)\nDate: {fmt_date(f['date'], style=st)}\n"
                f"Subject: {f['wo']} - {f['tag']} replacement - MOC screening\n\n" + noisy(join_sentences(s), rng, 0.12)
                + "\n\n" + self._policy(f))

    def questions(self):
        return {
            "change_classification": choice("How should this replacement be classified under the site's MOC standard?", {
                "replacement_in_kind": "As a replacement in kind: install under the normal work order with no MOC.",
                "moc_checklist_review": "As a change, reviewed with a short checklist by engineering and operations.",
                "moc_full_hazard_review": "As a change needing a full hazard review before any work.",
                "emergency_moc": "Through the emergency MOC path: proceed now under interim controls, formal review within the grace period.",
            }),
            "spec_deviation_significance": score("How far does the proposed item depart from the installed design specification?", [
                "Not at all: the same item in every respect that matters.",
                "Slightly: different source or part number, equal or better on paper.",
                "Materially: a rating, material, capacity or behaviour differs.",
                "Critically: it falls short of a rating, set point, certification or compatibility the service needs.",
            ]),
            "verification_needed": score("How much engineering verification should happen before the item goes in?", [
                "None beyond checking the part number.",
                "A desk comparison of data sheets.",
                "An engineering calculation or compatibility check.",
                "A full design review, with vendor test data where needed.",
            ]),
            "psi_update_required": noul(
                "Process safety information must be updated because of this replacement.",
                yes="A data sheet, relief-sizing basis, instrument range, alarm list or drawing no longer matches what will be installed.",
                no="Everything recorded about the item stays true after the replacement."),
            "requester_classification_correct": noul(
                "The requester classified this replacement correctly under the standard."),
        }


# =========================================================================== schema 2
# kind: (description template, safeguard touched, new equipment or line?)
CHANGES = {
    "throughput": ("raise the throughput of the {unit} unit by {x}% above its current maximum", "none", False),
    "new_line": ("add a new tie-in line from tank T-{n} to the {unit} feed header", "none", True),
    "sif_setpoint": ("move the high-temperature trip of SIF-{n} from {a} C to {b} C", "sif", False),
    "sif_removal": ("remove the low-flow trip on pump P-{n}, keeping the low-flow alarm", "sif", False),
    "dosing_mode": ("replace batch addition of the {additive} with continuous metering", "none", True),
    "relief_reroute": ("reroute the discharge of PSV-{n} from atmosphere to the flare header", "relief", True),
    "valve_bypass": ("add a manual bypass around control valve FV-{n}", "bpcs", True),
    "inhibitor_swap": ("switch the corrosion inhibitor to a different supplier's product", "none", False),
    "fill_level": ("raise the maximum fill of tank T-{n} from 85% to 90%, moving the high-level alarm", "alarm", False),
    "startup_vacuum": ("run the {unit} column under vacuum during start-up to shorten warm-up", "none", False),
}
ENVELOPE = {"normal": "inside the normal operating window", "inside_sol": "outside the normal window but inside the safe operating limits",
            "beyond_sol": "beyond a current safe operating limit (equipment re-rating proposed)"}


class HazardReviewDepth(Schema):
    name = "hazard_review_depth"
    description = "A confirmed process change: which hazard-review method the site standard calls for, and where the review should focus."
    surfaces = {"json": 0.5, "moc_form": 0.25, "proposer_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        kind = pick(rng, list(CHANGES), [1, 1, 1.5, 1.5, 1, 1, 1, 1, 1, 1])
        unit, chem, haz = _unit(rng)
        cat = pick_w(rng, {1: 1, 2: 2, 3: 3, 4: 2, 5: 1})
        env = pick_w(rng, {"normal": 4, "inside_sol": 3, "beyond_sol": 1})
        novelty = pick_w(rng, {"routine_here": 2, "done_at_sister_plant": 3, "first_at_company": 2})
        relief_load = kind == "throughput" and chance(rng, 0.35)
        pha_age = pick(rng, [1, 2, 3, 4, 5, 6])
        proposed = pick(rng, ["checklist", "what-if", "HAZOP", "HAZOP + LOPA"])
        if reg == "clear":
            if chance(rng, 0.5):
                kind, env, cat, relief_load = pick(rng, ["inhibitor_swap", "fill_level"]), "normal", pick(rng, [1, 2]), False
                unit, chem, haz = _unit(rng, [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 2, 2])
                proposed = "checklist"
            else:
                kind, cat = pick(rng, ["sif_setpoint", "sif_removal"]), pick(rng, [4, 5])
                proposed = "HAZOP + LOPA"
        elif reg == "borderline":
            cat = pick(rng, [3, 4])
            env = pick(rng, ["normal", "inside_sol"])
        else:
            # the proposer argues for a light review on a change that touches a protection layer, or vice versa
            if chance(rng, 0.6):
                kind, proposed = pick(rng, ["relief_reroute", "throughput", "valve_bypass", "sif_setpoint"]), "checklist"
                relief_load = kind == "throughput"
            else:
                kind, env, cat, proposed = pick(rng, ["inhibitor_swap", "startup_vacuum", "fill_level"]), "normal", pick(rng, [2, 3]), "HAZOP"
        if unit in STORAGE_UNITS and kind in ("throughput", "startup_vacuum"):
            kind = pick(rng, ["fill_level", "new_line", "valve_bypass"])
        if kind == "startup_vacuum" and unit not in ("solvent recovery", "amine treating", "caustic evaporation", "HF alkylation"):
            unit = pick(rng, ["solvent recovery", "amine treating", "HF alkylation"])
            chem, haz = UNITS[unit]
        a = pick(rng, [165, 180, 185, 210, 240])
        tmpl = CHANGES[kind][0]
        desc = tmpl.format(unit=unit, x=pick(rng, [6, 8, 10, 12, 15, 20]), n=rng.randint(101, 499), a=a, b=a + pick(rng, [5, 7, 10]),
                           additive=ADDITIVE.get(unit, "neutralising additive"))
        d = rand_date(rng)
        return {"regime": reg, "company": _company(rng), "site": pick(rng, TOWNS), "unit": unit, "chemical": chem,
                "hazard": haz, "kind": kind, "description": desc, "pha_category": cat, "envelope": env, "novelty": novelty,
                "raises_relief_load": relief_load, "pha_age_years": pha_age, "proposed_method": proposed,
                "proposer": person(rng), "coordinator": person(rng), "date": d, "moc": _moc_no(rng, d),
                "inventory_change": {"fill_level": "+6% stored volume at maximum fill", "throughput": pick(rng, [None, "+10% hold-up"]),
                                     "new_line": pick(rng, ["+2 m3 in the new line", None]),
                                     "dosing_mode": pick(rng, ["-60% initiator hold-up in the dosing vessel", None])}.get(kind)}

    def _safeguard(self, f):
        sg = CHANGES[f["kind"]][1]
        if f["raises_relief_load"]:
            sg = "relief"
        return sg

    def intended(self, f):
        sg, new_eq, cat, env = self._safeguard(f), CHANGES[f["kind"]][2], f["pha_category"], f["envelope"]
        if sg == "sif":
            method = "hazop_with_lopa"
        elif new_eq or sg == "relief" or env == "beyond_sol" or cat >= 4:
            method = "hazop_affected_nodes"
        elif env == "inside_sol" or sg in ("alarm", "bpcs") or f["kind"] in ("dosing_mode", "startup_vacuum", "throughput"):
            method = "what_if_review"
        else:
            method = "checklist_review"
        reactive_kind = f["kind"] in ("throughput", "dosing_mode", "sif_setpoint", "startup_vacuum", "relief_reroute")
        if f["hazard"] == "reactive" and reactive_kind or f["kind"] == "relief_reroute" or f["raises_relief_load"]:
            focus = "runaway_or_overpressure"
        elif f["hazard"] in ("toxic", "toxic_flammable"):
            focus = "toxic_release"
        elif f["hazard"] in ("flammable", "reactive"):
            focus = "fire_or_explosion"
        else:
            focus = "operability_or_quality" if f["kind"] in ("inhibitor_swap", "startup_vacuum", "throughput", "dosing_mode", "fill_level") else "toxic_release"
        cons = {1: 0, 2: 0, 3: 1, 4: 2, 5: 3}[cat]
        if HAZ[f["hazard"]] >= 3 and cat >= 2:
            cons = min(3, cons + 1)
        imp = {"none": 0, "alarm": 1, "bpcs": 1, "relief": 2, "sif": 2}[sg]
        if f["kind"] == "sif_removal":
            imp = 3
        if env == "beyond_sol":
            imp = max(imp, 1)
        return {"review_method": method, "primary_hazard_focus": focus, "consequence_potential": cons,
                "safeguard_impact": imp, "sif_revalidation_needed": "yes" if sg == "sif" else "no"}

    def _policy(self):
        return ("MOC standard S-11 (hazard review level): checklist review for changes inside the normal operating window "
                "that alter no safeguard; what-if review for changes to operating conditions, procedures, alarms or control "
                "loops that stay inside safe operating limits where the node's consequence category is 3 or lower; HAZOP of "
                "the affected nodes for new lines or equipment, relief-path or relief-load changes, operation beyond a safe "
                "operating limit, or consequence category 4 or 5; HAZOP plus LOPA when a safety instrumented function is "
                "added, removed, bypassed or has its trip point changed.")

    def render(self, f, rng, surface):
        st = pick(rng, DATE_STYLE)
        haz_txt = f"{f['chemical']} ({f['hazard'].replace('_', ' and ')})"
        nov = {"routine_here": "done several times on this unit", "done_at_sister_plant": "done at a sister plant, first time here",
               "first_at_company": "first time anywhere in the company"}[f["novelty"]]
        relief = "process engineer notes the higher rate raises the relieving load on the unit's main relief valve" if f["raises_relief_load"] else None
        if surface == "json":
            return dumps(prune({
                "change": {"moc": f["moc"], "unit": f["unit"], "material": haz_txt, "description": f["description"],
                           "operating_envelope": ENVELOPE[f["envelope"]], "inventory_change": f["inventory_change"],
                           "experience": nov, "engineering_note": relief},
                "existing_pha": {"node_consequence_category": f"{f['pha_category']} of 5",
                                 "last_revalidation_years_ago": f["pha_age_years"]},
                "proposal": {"proposed_by": f["proposer"], "suggested_review": f["proposed_method"],
                             "date": fmt_date(f["date"], style="iso")},
                "standard": self._policy(),
            }))
        if surface == "moc_form":
            rows = [("MOC", f["moc"]), ("Site", f"{f['company']}, {f['site']}"), ("Unit", f["unit"]), ("Material", haz_txt),
                    ("Change", f["description"]), ("Operating envelope", ENVELOPE[f["envelope"]]),
                    ("Inventory", f["inventory_change"] or "no change"), ("Experience", nov),
                    ("PHA node consequence category", f"{f['pha_category']} (1 low - 5 catastrophic)"),
                    ("PHA last revalidated", f"{f['pha_age_years']} years ago"), ("Engineering note", relief),
                    ("Review suggested by originator", f["proposed_method"]), ("Originator", f["proposer"]),
                    ("Date", fmt_date(f["date"], style=st))]
            return "MANAGEMENT OF CHANGE - HAZARD REVIEW SCREENING\n" + kv_lines(rows, rng) + "\n\n" + self._policy()
        s = [f"For {f['moc']} we want to {f['description']}.", f"Material: {haz_txt}.",
             f"New conditions are {ENVELOPE[f['envelope']]}.", f"This has been {nov}.",
             (relief[0].upper() + relief[1:] + "." if relief else ""),
             f"The PHA puts this node at consequence category {f['pha_category']}; last revalidated {f['pha_age_years']} years ago.",
             f"I'd suggest a {f['proposed_method']} review" + pick(rng, [" - it's a small change.", " to keep it moving.", ".",
                                                                       " given what's involved."])]
        return (f"From: {f['proposer']}\nTo: {f['coordinator']} (PSM coordinator)\nDate: {fmt_date(f['date'], style=st)}\n"
                f"Subject: {f['moc']} - hazard review level\n\n" + noisy(join_sentences(s), rng, 0.12) + "\n\n" + self._policy())

    def questions(self):
        return {
            "review_method": choice("Which hazard-review method does the site standard call for on this change?", {
                "checklist_review": "A checklist review by engineering and operations.",
                "what_if_review": "A structured what-if review.",
                "hazop_affected_nodes": "A HAZOP of the nodes the change affects.",
                "hazop_with_lopa": "A HAZOP of the affected nodes plus a layer-of-protection analysis.",
            }),
            "primary_hazard_focus": choice("Which hazard should the review concentrate on first?", {
                "toxic_release": "A loss of containment releasing a toxic material.",
                "fire_or_explosion": "A flammable release that ignites.",
                "runaway_or_overpressure": "A runaway reaction or an overpressure the relief system must handle.",
                "operability_or_quality": "Operability and product quality; the safety exposure is minor.",
            }),
            "consequence_potential": score("How severe is the worst credible outcome if the change goes wrong?", [
                "Minor: a process upset or a small, contained spill.",
                "Moderate: an on-site injury or a reportable release kept on site.",
                "Major: serious injuries or a release reaching beyond the fence.",
                "Catastrophic: fatalities on or off site.",
            ]),
            "safeguard_impact": score("How much does the change alter the unit's layers of protection?", [
                "Not at all: no alarm, control loop, relief device or trip is touched.",
                "Slightly: an alarm, control loop or operating limit moves.",
                "Substantially: a relief device's duty or a trip point changes.",
                "Fundamentally: a protection layer is removed.",
            ]),
            "sif_revalidation_needed": noul(
                "The change requires a safety instrumented function's SIL verification to be redone.",
                yes="A safety instrumented function is added, removed, bypassed or has its trip point changed.",
                no="No safety instrumented function is touched by the change."),
        }


# =========================================================================== schema 3
SCOPES = {
    "new_equipment": ("install a new {eq} on the {unit} unit", True),
    "modified_psi": ("re-rate {eq} on the {unit} unit and fit a new relief valve sized for the new rating", True),
    "procedure_only": ("change the {unit} unit's start-up procedure to a staged warm-up; no hardware change", False),
    "rik_only": ("replace {eq} on the {unit} unit like for like", False),
    "temporary_psi": ("connect a temporary transfer hose and pump to {eq} on the {unit} unit", True),
}
EQUIPMENT = ["feed pump P-{n}C", "knock-out drum V-{n}", "plate heat exchanger E-{n}", "vent scrubber S-{n}",
             "dosing skid K-{n}", "storage tank T-{n}"]
A_ITEMS = ["hydrotest record for spool {n} missing", "gas detector GD-{n} not commissioned",
           "PSV-{n} not yet installed", "emergency shutdown valve XV-{n} stroke test not done",
           "safety shower at the new skid not connected", "fire-water monitor M-{n} obstructed by scaffolding"]
B_ITEMS = ["insulation incomplete on the new line", "paint touch-up on supports", "as-built P&ID update pending",
           "permanent nameplates not fitted", "housekeeping: scaffolding to remove", "spare-parts list not loaded to stores"]
HIDDEN_A = ["car seal missing on the PSV-{n} inlet block valve", "gas detector GD-{n} reading not yet on the control-room display",
            "high-level trip LSHH-{n} not function-tested", "relief valve PSV-{n} set-pressure certificate not received"]
FIELD_CHANGES = ["a 2 in globe valve was installed where the MOC approved a 2 in ball valve",
                 "a drain valve not on the approved P&ID was added at the low point",
                 "the line was routed through the pipe rack instead of at grade, adding 40 ft",
                 "a different pump model was fitted because the approved one was back-ordered"]


class PssrStartupReadiness(Schema):
    name = "pssr_startup_readiness"
    description = "A changed unit awaiting start-up: does it need a pre-startup safety review, and is it ready to start."
    surfaces = {"json": 0.5, "pssr_checklist": 0.3, "startup_meeting_notes": 0.2}

    def sample_factors(self, rng):
        reg = regime(rng)
        scope = pick_w(rng, {"new_equipment": 2.5, "modified_psi": 2.5, "procedure_only": 2, "rik_only": 1.5, "temporary_psi": 1})
        unit, chem, haz = _unit(rng)
        n_a = pick_w(rng, {0: 9, 1: 2, 2: 0.5})
        n_b = pick_w(rng, {0: 3, 1: 3, 2: 2, 4: 1})
        hidden = chance(rng, 0.12) and n_b > 0
        crew = rng.randint(4, 9)
        trained = pick_w(rng, {crew: 8, crew - 1: 2, max(1, crew - 3): 0.5})
        procs = pick_w(rng, {"approved": 8, "draft": 2, "not_updated": 0.7})
        field_change = chance(rng, 0.12)
        loss = pick_w(rng, {3000: 2, 8000: 2, 25000: 2, 60000: 2, 140000: 1, 300000: 1})
        commitment = pick_w(rng, {None: 5, "contract delivery": 2, "regulatory deadline": 1})
        days_late = pick(rng, [0, 0, 0, 2, 5, 12, 30])
        if reg == "clear":
            if chance(rng, 0.5):
                n_a, n_b, hidden, trained, procs, field_change = 0, 0, False, crew, "approved", False
            else:
                n_a, field_change = pick(rng, [1, 2]), False
        elif reg == "borderline":
            n_a, hidden, field_change = 0, chance(rng, 0.3), False
            n_b = max(n_b, 1)
            if not hidden and chance(rng, 0.25):
                trained = crew - 1
        else:
            # a clean-looking checklist under heavy schedule pressure with one buried blocker, or a field change
            loss, commitment, n_a = pick(rng, [140000, 300000]), "contract delivery", 0
            if chance(rng, 0.5):
                hidden, n_b = True, max(n_b, 2)
            else:
                field_change = True
        if procs == "not_updated" and scope == "rik_only":
            procs = "approved"
        n = lambda: rng.randint(101, 899)  # noqa: E731
        a_list = [pick(rng, A_ITEMS).format(n=n()) for _ in range(n_a)]
        b_list = [pick(rng, B_ITEMS) for _ in range(n_b)]
        if hidden:
            b_list[rng.randrange(len(b_list))] = pick(rng, HIDDEN_A).format(n=n())
        eq = pick(rng, EQUIPMENT).format(n=rng.randint(101, 899))
        d = rand_date(rng)
        return {"regime": reg, "company": _company(rng), "site": pick(rng, TOWNS), "unit": unit, "chemical": chem,
                "hazard": haz, "scope": scope, "description": SCOPES[scope][0].format(eq=eq, unit=unit),
                "a_items": a_list, "b_items": b_list, "hidden_a": hidden, "crew": crew, "trained": trained,
                "procedures": procs, "field_change": pick(rng, FIELD_CHANGES) if field_change else None,
                "loss_per_day": loss, "commitment": commitment, "days_late": days_late, "lead": person(rng),
                "manager": person(rng), "date": d, "moc": _moc_no(rng, d), "b_days": pick(rng, [30, 30, 60])}

    def intended(self, f):
        pssr = SCOPES[f["scope"]][1]
        blockers = len(f["a_items"]) + (1 if f["hidden_a"] else 0) + (1 if f["trained"] < f["crew"] else 0) + (
            1 if f["procedures"] != "approved" else 0)
        if f["field_change"]:
            auth = "return_to_moc_review"
        elif blockers:
            auth = "hold_until_items_closed"
        elif f["b_items"]:
            auth = "authorize_with_tracked_items"
        else:
            auth = "authorize_startup"
        gap = 3 if (f["field_change"] or blockers >= 2) else 2 if blockers == 1 else 1 if f["b_items"] else 0
        p = 0
        p += 2 if f["loss_per_day"] >= 140000 else 1 if f["loss_per_day"] >= 25000 else 0
        p += 1 if f["commitment"] else 0
        p += 1 if f["days_late"] >= 12 else 0
        return {"pssr_required": "yes" if pssr else "no", "startup_authorization": auth, "readiness_gap": min(3, gap),
                "schedule_pressure": min(3, p), "punch_item_misclassified": "yes" if f["hidden_a"] else "no"}

    def _policy(self, f):
        return (f"Site standard S-12: a pre-startup safety review is required before start-up of new facilities and of "
                f"modified facilities whose modification changes process safety information, temporary changes included. "
                f"Category A punch items (safety, environmental or code items) must be closed before start-up; category B "
                f"items may be closed within {f['b_days']} days after start-up, tracked by the PSSR lead. Every operator on "
                f"the start-up shift must be trained on new or changed procedures, and those procedures must be approved, "
                f"before start-up. Installed work that differs from the approved MOC goes back to MOC review.")

    def render(self, f, rng, surface):
        st, ms = pick(rng, DATE_STYLE), pick(rng, ["sym", "sym", "code"])
        procs = {"approved": "approved and issued", "draft": "in draft, not yet approved", "not_updated": "not yet updated"}[f["procedures"]]
        sched = (f"{money(f['loss_per_day'], style=ms)}/day of lost production"
                 + (f"; {f['commitment']} at stake" if f["commitment"] else "")
                 + (f"; {f['days_late']} days behind plan" if f["days_late"] else ""))
        fc = f"Field walk-down: {f['field_change']}." if f["field_change"] else None
        if surface == "json":
            return dumps(prune({
                "change": {"moc": f["moc"], "unit": f["unit"], "material": f["chemical"], "scope": f["description"]},
                "punch_list": {"category_a_open": f["a_items"] or None, "category_b_open": f["b_items"] or None},
                "readiness": {"startup_shift_trained": f"{f['trained']} of {f['crew']}", "procedures": procs,
                              "field_walkdown": f["field_change"] or "installation matches the approved MOC"},
                "schedule": {"impact": sched, "target_date": fmt_date(f["date"], style="iso")},
                "pssr_lead": f["lead"],
                "standard": self._policy(f),
            }))
        if surface == "pssr_checklist":
            rows = [["Scope", f["description"]], ["MOC", f["moc"]], ["Material", f["chemical"]],
                    ["A items open", "; ".join(f["a_items"]) or "none"], ["B items open", "; ".join(f["b_items"]) or "none"],
                    ["Start-up shift trained", f"{f['trained']}/{f['crew']}"], ["Procedures", procs],
                    ["Walk-down vs approved design", f["field_change"] or "matches"], ["Schedule", sched],
                    ["Target start-up", fmt_date(f["date"], style=st)]]
            return (f"PRE-STARTUP SAFETY REVIEW CHECKLIST - {f['company']}, {f['site']} - {f['unit']}\nPSSR lead: {f['lead']}\n\n"
                    + table(["item", "status"], rows, rng) + "\n\n" + self._policy(f))
        s = [f"Start-up meeting for {f['moc']} ({f['description']}).",
             f"Open A items: {', '.join(f['a_items']) if f['a_items'] else 'none'}.",
             f"Open B items: {', '.join(f['b_items']) if f['b_items'] else 'none'}.",
             f"{f['trained']} of {f['crew']} start-up shift operators signed off on training; procedures {procs}.",
             fc or "", f"Schedule: {sched}.",
             f"{f['manager']} (plant manager): " + pick(rng, ["we need this running this week.", "safety first, but let's not lose another day.",
                                                              "what's the earliest we can start?", "I'll back whatever the PSSR lead decides."])]
        return (f"MEETING NOTES - {fmt_date(f['date'] - timedelta(days=1), style=st)} - {f['unit']}\n\n"
                + noisy(join_sentences(s), rng, 0.1) + "\n\n" + self._policy(f))

    def questions(self):
        return {
            "pssr_required": noul(
                "A pre-startup safety review is required before this unit restarts.",
                yes="The work adds a facility or changes process safety information, temporary changes included.",
                no="The work changes no process safety information, such as a procedure-only change or a true like-for-like swap."),
            "startup_authorization": choice("What should the PSSR lead decide about start-up?", {
                "authorize_startup": "Authorise start-up; nothing is outstanding.",
                "authorize_with_tracked_items": "Authorise start-up and track the open items the standard allows after start-up.",
                "hold_until_items_closed": "Hold start-up until the blocking items are closed.",
                "return_to_moc_review": "Send the change back to MOC review, because what was built differs from what was approved.",
            }),
            "readiness_gap": score("How far is the unit from being ready to start up safely?", [
                "Ready: nothing outstanding.",
                "Nearly: only items that may be closed after start-up.",
                "Short: one item that must be closed first.",
                "Well short: several blockers, or the installation itself is in question.",
            ]),
            "schedule_pressure": score("How hard is the schedule pushing this start-up?", [
                "Little: the delay costs little and nothing external depends on it.",
                "Some: noticeable lost production, but no outside commitment.",
                "Heavy: large daily losses or an outside commitment at stake.",
                "Intense: large losses, an outside commitment and a slipping date together.",
            ]),
            "punch_item_misclassified": noul(
                "At least one open item listed as category B should really be category A."),
        }


# =========================================================================== schema 4
# kind: (description, class, base exposure)
TEMP = {
    "sif_bypass": ("bypass of high-level trip LSHH-{n} on V-{m} (the level transmitter failed)", "sif", 2),
    "interlock_jumper": ("jumper on high-pressure interlock PSHH-{n} on V-{m}", "sif", 3),
    "leak_clamp": ("leak-sealing enclosure clamp on the flange at FV-{n}", "clamp", 2),
    "alarm_shelved": ("chattering high-temperature alarm TAH-{n} suppressed", "alarm", 1),
    "rental_chiller": ("rental air-cooled chiller carrying the duty of E-{n}", "equipment", 1),
    "drum_storage": ("temporary storage of {k} drums of solvent inside the unit fence", "storage", 1),
    "temp_hose": ("temporary hose and pump for transfer from T-{n}", "equipment", 1),
}
# compensating measures by class of temporary change: (strong measures, routine measures)
MEASURES = {
    "sif": (["operator stationed at the vessel with a local gauge", "second independent transmitter alarming in the control room"],
            ["hourly operator rounds", "feed rate cut to 80%", "inventory reduced in the affected vessel"]),
    "clamp": (["fixed gas detector added beside the clamp", "daily clamp inspection with leak test"],
              ["hourly operator rounds", "area barricaded", "inventory reduced in the affected line"]),
    "alarm": (["second alarm configured on an independent sensor"], ["hourly check of the local temperature gauge", "hourly operator rounds"]),
    "equipment": (["hose and connections inspected and pressure-tested before each use"],
                  ["fire watch during transfers", "hourly operator rounds", "vendor service contract in place"]),
    "storage": (["drums in a bunded area with spill trays and fixed fire protection"], ["hourly operator rounds", "area barricaded"]),
}


class TemporaryChangeExtension(Schema):
    name = "temporary_change_extension"
    description = "A temporary change near its expiry with a request to extend it, judged against the site's temporary-change standard."
    surfaces = {"json": 0.5, "extension_request": 0.25, "shift_log_extract": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        kind = pick(rng, list(TEMP))
        unit, chem, haz = _unit(rng)
        max_days = pick(rng, [90, 120, 180])
        orig = pick(rng, [30, 45, 60, 90])
        ext_so_far = pick_w(rng, {0: 4, 1: 3, 2: 1})
        days_used = orig + ext_so_far * pick(rng, [15, 30, 30])
        req = pick(rng, [14, 21, 30, 45, 60, 90])
        strong, routine = MEASURES[TEMP[kind][1]]
        measures = rng.sample(strong, pick_w(rng, {0: 2, 1: 2, len(strong): 1})) + rng.sample(routine, rng.randint(0, len(routine)))
        incident = pick_w(rng, {None: 6, "near_miss": 2, "clamp_weeping": 1})
        demands = pick_w(rng, {0: 5, 1: 2, 3: 1})
        fix_eta = pick(rng, ["part on order, 3 weeks", "part on order, 10 weeks", "needs a shutdown, next one in 5 months",
                             "engineering design not started", "vendor service visit booked next week"])
        if reg == "clear":
            if chance(rng, 0.5):
                kind, ext_so_far, req, incident, demands = pick(rng, ["rental_chiller", "temp_hose", "alarm_shelved"]), 0, 14, None, 0
                strong, routine = MEASURES[TEMP[kind][1]]
                days_used, measures = orig, strong + routine[:1]
                fix_eta = "vendor service visit booked next week"
            else:
                kind, incident, demands, measures = pick(rng, ["sif_bypass", "interlock_jumper"]), "near_miss", 3, []
        elif reg == "borderline":
            req = max(7, max_days - days_used + pick(rng, [-10, -3, 0, 5, 12]))
        else:
            # a well-compensated bypass that has simply run too long, or a minor item with a near miss
            if chance(rng, 0.5):
                # a bypass with only routine cover; well run in one case, near its limits in another
                kind, ext_so_far = "sif_bypass", pick(rng, [1, 2])
                measures = [pick(rng, MEASURES["sif"][1])]
                if ext_so_far == 1:
                    days_used, req, incident, demands = orig + 30, 14, None, 0
                    max_days = max(max_days, days_used + 30)
                else:
                    days_used = max_days - pick(rng, [5, 10])
            else:
                kind, incident = pick(rng, ["drum_storage", "leak_clamp", "rental_chiller"]), pick(rng, ["near_miss", "clamp_weeping"])
                strong, routine = MEASURES[TEMP[kind][1]]
                measures = rng.sample(routine, 1)
        if incident == "clamp_weeping" and kind != "leak_clamp":
            incident = "near_miss"
        start = rand_date(rng)
        expiry = start + timedelta(days=days_used)
        desc = TEMP[kind][0].format(n=rng.randint(101, 899), m=rng.randint(101, 499), k=pick(rng, [12, 24, 40, 60]))
        return {"regime": reg, "company": _company(rng), "site": pick(rng, TOWNS), "unit": unit, "chemical": chem,
                "hazard": haz, "kind": kind, "description": desc, "max_days": max_days, "original_days": orig,
                "extensions_so_far": ext_so_far, "days_used": days_used, "requested_days": req, "measures": measures,
                "incident": incident, "demands": demands, "fix_eta": fix_eta, "start": start, "expiry": expiry,
                "owner": person(rng), "unit_manager": person(rng), "tc": f"TC-{start.year}-{rng.randint(10, 299):03d}"}

    def _strength(self, f):
        strong = MEASURES[TEMP[f["kind"]][1]][0]
        s = sum(1.5 if x in strong else 0.5 for x in f["measures"])
        return 0 if s == 0 else 1 if s < 1.5 else 2 if s < 3 else 3

    def intended(self, f):
        cls = TEMP[f["kind"]][1]
        total = f["days_used"] + f["requested_days"]
        within = total <= f["max_days"] and f["extensions_so_far"] < 2
        comp = self._strength(f)
        expo = TEMP[f["kind"]][2] + (1 if f["incident"] else 0) + (1 if f["demands"] >= 1 and cls == "sif" else 0)
        expo += 1 if (HAZ[f["hazard"]] >= 3 and cls in ("clamp", "sif")) else 0
        expo = max(0, min(3, expo - comp // 2))
        if cls in ("sif", "clamp") and comp == 0 and (f["incident"] or f["demands"] >= 3):
            dec = "restore_by_expiry"
        elif not within:
            dec = "restore_by_expiry" if cls == "sif" else "convert_to_permanent_change"
        elif comp < 2 and (expo >= 2 or f["incident"]):
            dec = "extend_with_added_safeguards"
        else:
            dec = "extend_as_requested"
        senior = f["extensions_so_far"] >= 1 or cls == "sif"
        return {"extension_decision": dec, "exposure_while_temporary": expo, "compensating_measures_strength": comp,
                "within_temporary_limits": "yes" if within else "no", "senior_approval_required": "yes" if senior else "no"}

    def _policy(self, f):
        return (f"Temporary-change standard S-9: a temporary change expires on its approved date. Total duration including "
                f"extensions may not exceed {f['max_days']} days and no more than two extensions are allowed. The unit manager "
                f"may approve a first extension; a second extension, or any extension of a safety-instrumented-function "
                f"bypass or interlock defeat, needs the plant manager. A change that must outlast these limits is either "
                f"removed or converted to a permanent change with a full MOC.")

    def render(self, f, rng, surface):
        st = pick(rng, DATE_STYLE)
        new_exp = f["expiry"] + timedelta(days=f["requested_days"])
        inc = {None: "none", "near_miss": "near miss logged during the period (see shift log)",
               "clamp_weeping": "clamp found weeping once; re-torqued"}[f["incident"]]
        dem = (f"{f['demands']} occasion(s) when the bypassed function would have been called on" if TEMP[f["kind"]][1] == "sif"
               else None)
        if surface == "json":
            return dumps(prune({
                "temporary_change": {"id": f["tc"], "unit": f["unit"], "material": f["chemical"], "description": f["description"],
                                     "started": fmt_date(f["start"], style="iso"), "current_expiry": fmt_date(f["expiry"], style="iso"),
                                     "extensions_granted": f["extensions_so_far"], "owner": f["owner"]},
                "request": {"extend_to": fmt_date(new_exp, style="iso"), "reason": f["fix_eta"]},
                "controls_in_place": f["measures"] or ["none recorded"],
                "experience": {"incidents": inc, "demands_on_bypassed_function": dem},
                "standard": self._policy(f),
            }))
        if surface == "extension_request":
            rows = [("Temporary change", f["tc"]), ("Unit", f"{f['unit']} ({f['chemical']})"), ("Description", f["description"]),
                    ("Started", fmt_date(f["start"], style=st)), ("Current expiry", fmt_date(f["expiry"], style=st)),
                    ("Extensions already granted", f["extensions_so_far"]), ("Requested new expiry", fmt_date(new_exp, style=st)),
                    ("Why the permanent fix is late", f["fix_eta"]),
                    ("Compensating measures", "; ".join(f["measures"]) or "none recorded"), ("Incidents", inc),
                    ("Demands on bypassed function", dem), ("Requested by", f["owner"])]
            return "TEMPORARY CHANGE - EXTENSION REQUEST\n" + kv_lines(rows, rng) + "\n\n" + self._policy(f)
        lines = [f"SHIFT LOG EXTRACT - {f['unit']} - {f['tc']} ({f['description']})"]
        t = f["start"]
        lines.append(f"{fmt_date(t, style=st)}  temporary change in service, expiry {fmt_date(f['start'] + timedelta(days=f['original_days']), style=st)}")
        for i in range(f["extensions_so_far"]):
            lines.append(f"{fmt_date(t + timedelta(days=f['original_days'] + 30 * i - 3), style=st)}  extension {i + 1} approved")
        if f["incident"] == "near_miss":
            lines.append(f"{fmt_date(t + timedelta(days=rng.randint(5, max(6, f['days_used'] - 5))), style=st)}  near miss: "
                         + pick(rng, ["high level reached 92% before the operator caught it", "pressure ran up to the alarm with the trip defeated",
                                      "drum found leaking at the bung", "hose coupling found loose at start of transfer"]))
        if f["incident"] == "clamp_weeping":
            lines.append(f"{fmt_date(t + timedelta(days=rng.randint(5, max(6, f['days_used'] - 5))), style=st)}  clamp weeping, re-torqued by contractor")
        if dem:
            lines.append(f"{fmt_date(f['expiry'] - timedelta(days=2), style=st)}  review: {dem}")
        lines.append(f"{fmt_date(f['expiry'] - timedelta(days=1), style=st)}  {f['owner']} requests extension to "
                     f"{fmt_date(new_exp, style=st)}: {f['fix_eta']}; controls: {', '.join(f['measures']) or 'none recorded'}")
        return "\n".join(lines) + "\n\n" + self._policy(f)

    def questions(self):
        return {
            "extension_decision": choice("How should the request to extend this temporary change be answered?", {
                "extend_as_requested": "Extend it to the requested date with the controls already in place.",
                "extend_with_added_safeguards": "Extend it only with stronger compensating measures added.",
                "convert_to_permanent_change": "Stop extending it; take it through a full MOC as a permanent change.",
                "restore_by_expiry": "Refuse; restore the original configuration by expiry, shutting down if necessary.",
            }),
            "exposure_while_temporary": score("How much extra risk does the plant carry while this temporary change stays in place?", [
                "Negligible: nothing safety-related is weakened.",
                "Low: a minor safeguard is weakened and well covered.",
                "Significant: a real protection is weakened or there has been a near miss.",
                "High: a key protection is defeated in hazardous service with little cover.",
            ]),
            "compensating_measures_strength": score("How well do the compensating measures make up for what the temporary change takes away?", [
                "Not at all: none are recorded.",
                "Weakly: routine rounds or checks only.",
                "Reasonably: specific measures aimed at the lost protection.",
                "Strongly: an independent measure of similar reliability is in place.",
            ]),
            "within_temporary_limits": noul(
                "Granting the requested extension keeps this temporary change within the standard's limits.",
                yes="Total duration including the extension stays within the maximum and the extension count stays within the cap.",
                no="The extension would take the total past the maximum or exceed the number of extensions allowed."),
            "senior_approval_required": noul(
                "Under the standard, this extension needs the plant manager's approval."),
        }


# =========================================================================== schema 5
ORG = {
    # kind: (description, main risk it creates)
    "board_staffing": ("reduce night-shift board operators on the {unit} console group from {a} to {b}", "upset_response_capacity"),
    "control_room_merge": ("merge the {unit} control room into the main control room, with one operator covering both consoles at night", "upset_response_capacity"),
    "field_rounds": ("combine field rounds for two units into one route, removing one field operator per shift", "upset_response_capacity"),
    "contracting": ("replace in-house instrument technicians with a contractor on a call-out basis", "loss_of_specialist_knowledge"),
    "retirement": ("retire the only engineer qualified on the SIS logic, with no successor named", "loss_of_specialist_knowledge"),
    "ert_reduction": ("reduce the site emergency response team from {a} to {b} per shift", "emergency_response_coverage"),
    "shift_pattern": ("move operators from 8-hour to 12-hour shifts with two-shift handovers", "fatigue_and_handover"),
}


class OrganisationalChangeReview(Schema):
    name = "organisational_change_review"
    description = "A staffing or organisational change at a chemical plant, reviewed under the site's organisational MOC standard."
    surfaces = {"json": 0.5, "org_moc_form": 0.25, "review_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        kind = pick(rng, list(ORG))
        unit, chem, haz = _unit(rng)
        min_board, ert_min = pick(rng, [2, 2, 1]), pick(rng, [4, 5, 6])
        board_now = max(min_board, 2)
        board_after = board_now - 1 if kind in ("board_staffing", "control_room_merge") else board_now
        ert_now = ert_min + pick(rng, [0, 1, 2])
        ert_after = ert_now - pick(rng, [1, 2]) if kind == "ert_reduction" else ert_now
        ops_in_ert = chance(rng, 0.35)
        if ops_in_ert and kind in ("board_staffing", "field_rounds", "control_room_merge"):
            ert_after = ert_now - 1
        sis_cover = "no" if kind == "retirement" and chance(rng, 0.6) else "yes"
        study = pick_w(rng, {"done_passed": 3, "done_marginal": 2, "not_done": 2})
        alarm_rate = round(pick_w(rng, {rng.uniform(0.3, 1.0): 4, rng.uniform(1.0, 2.0): 3, rng.uniform(2.0, 4.0): 2}), 1)
        upsets = pick(rng, [0, 1, 1, 2, 4, 6])
        cross = pick(rng, [40, 60, 75, 90, 100])
        overlap = pick(rng, [0, 0, 1, 3, 6])
        rationalised = chance(rng, 0.5)
        if reg == "clear":
            if chance(rng, 0.5):
                kind, study, cross, overlap, rationalised, alarm_rate = "shift_pattern", "done_passed", 100, 3, True, round(rng.uniform(0.3, 0.8), 1)
                board_after, ert_after, sis_cover, ops_in_ert = board_now, ert_now, "yes", False
            else:
                kind = "ert_reduction"
                ert_after, study = ert_min - 1, "not_done"
        elif reg == "borderline":
            if kind in ("board_staffing", "control_room_merge"):
                board_after = min_board
            ert_after = max(ert_after, ert_min) if chance(rng, 0.6) else ert_after
            study = pick(rng, ["done_marginal", "done_passed"])
        else:
            # big savings and a passed study, against a high alarm load; or a weak study with strong mitigations
            if chance(rng, 0.5):
                study, alarm_rate, upsets = "done_passed", round(rng.uniform(2.2, 4.0), 1), pick(rng, [4, 6])
            else:
                study, cross, overlap, rationalised = "not_done", 100, 6, True
        d = rand_date(rng)
        a, b = (board_now, board_after) if kind in ("board_staffing",) else (ert_now, ert_after)
        return {"regime": reg, "company": _company(rng), "site": pick(rng, TOWNS), "unit": unit, "chemical": chem,
                "hazard": haz, "kind": kind, "description": ORG[kind][0].format(unit=unit, a=a, b=b),
                "min_board": min_board, "board_after": board_after, "ert_min": ert_min, "ert_after": ert_after,
                "ops_in_ert": ops_in_ert, "sis_cover": sis_cover, "study": study, "alarm_rate": alarm_rate,
                "upsets_per_month": upsets, "cross_trained_pct": cross, "overlap_months": overlap,
                "alarms_rationalised": rationalised, "savings": pick(rng, [180000, 320000, 450000, 900000]),
                "sponsor": person(rng), "reviewer": person(rng), "date": d, "moc": _moc_no(rng, d),
                "feedback": pick(rng, ["operators' committee opposes the change", "operators broadly support it",
                                       "no formal feedback gathered", "union has asked for a joint review"])}

    def intended(self, f):
        met = f["board_after"] >= f["min_board"] and f["ert_after"] >= f["ert_min"] and f["sis_cover"] == "yes"
        transition_ok = f["cross_trained_pct"] >= 90 and (f["overlap_months"] >= 3 or f["kind"] not in ("retirement", "contracting"))
        if not met:
            dec = "reject_change"
        elif f["study"] == "not_done":
            dec = "require_staffing_study_first"
        elif f["study"] == "done_marginal" or not transition_ok or (f["alarm_rate"] > 2.0 and not f["alarms_rationalised"]):
            dec = "approve_with_transition_measures"
        else:
            dec = "approve_as_proposed"
        risk = ORG[f["kind"]][1]
        if f["ops_in_ert"] and f["kind"] in ("board_staffing", "field_rounds", "control_room_merge") and f["ert_after"] < f["ert_min"]:
            risk = "emergency_response_coverage"
        strain = 0
        if f["kind"] in ("board_staffing", "control_room_merge", "field_rounds", "ert_reduction"):
            strain = 1
            strain += 1 if f["alarm_rate"] > 1.0 else 0
            strain += 1 if (f["alarm_rate"] > 2.0 or f["upsets_per_month"] >= 4) else 0
        elif f["kind"] == "shift_pattern":
            strain = 1 if f["upsets_per_month"] >= 4 else 0
        if f["kind"] == "ert_reduction" and f["ert_after"] < f["ert_min"]:
            strain = max(strain, 2)
        ready = 0
        ready += 1 if f["cross_trained_pct"] >= 75 else 0
        ready += 1 if f["cross_trained_pct"] >= 90 and f["overlap_months"] >= 1 else 0
        ready += 1 if (f["alarms_rationalised"] and f["study"] == "done_passed") else 0
        return {"org_change_decision": dec, "main_risk": risk, "upset_handling_strain": min(3, strain),
                "transition_readiness": ready, "minimum_staffing_met": "yes" if met else "no"}

    def _policy(self, f):
        return (f"Organisational MOC standard S-15: at least {f['min_board']} qualified board operator(s) per console group "
                f"whenever a unit runs above 50% rate; an emergency response team of at least {f['ert_min']} on every shift "
                f"(operators who are team members count only if they can leave their post); a qualified SIS engineer reachable "
                f"within one hour at all times. A change to staffing, shift length or a key role needs a workload or staffing "
                f"study before approval, and a transition plan where skills or knowledge move.")

    def render(self, f, rng, surface):
        st, ms = pick(rng, DATE_STYLE), pick(rng, ["sym", "sym", "code"])
        study = {"done_passed": "workload study completed; upset simulations handled within targets",
                 "done_marginal": "workload study completed; two of five upset simulations exceeded response targets",
                 "not_done": "no workload study yet"}[f["study"]]
        after = (f"{f['board_after']} board operator(s) on nights, emergency team {f['ert_after']} per shift"
                 + (" (board operators are team members)" if f["ops_in_ert"] else "")
                 + f", SIS engineer cover {'in place' if f['sis_cover'] == 'yes' else 'not arranged'}")
        if surface == "json":
            return dumps(prune({
                "change": {"moc": f["moc"], "site": f"{f['company']}, {f['site']}", "unit": f["unit"],
                           "description": f["description"], "sponsor": f["sponsor"],
                           "annual_saving": money(f["savings"], style=ms)},
                "staffing": {"after_change": after, "workload_study": study},
                "operating_load": {"alarms_per_10_min_per_console": f["alarm_rate"], "upsets_per_month": f["upsets_per_month"],
                                   "alarm_rationalisation_done": f["alarms_rationalised"]},
                "transition": {"cross_trained_pct": f["cross_trained_pct"], "overlap_months": f["overlap_months"]},
                "workforce_feedback": f["feedback"],
                "standard": self._policy(f),
            }))
        if surface == "org_moc_form":
            rows = [("MOC", f["moc"]), ("Site", f"{f['company']}, {f['site']}"), ("Unit", f"{f['unit']} ({f['chemical']})"),
                    ("Change", f["description"]), ("Sponsor", f["sponsor"]), ("Annual saving", money(f["savings"], style=ms)),
                    ("Staffing after change", after), ("Workload study", study),
                    ("Alarm rate (per console, per 10 min)", f["alarm_rate"]), ("Upsets per month", f["upsets_per_month"]),
                    ("Alarm rationalisation", "done" if f["alarms_rationalised"] else "not done"),
                    ("Cross-trained", f"{f['cross_trained_pct']}%"), ("Overlap period", f"{f['overlap_months']} months"),
                    ("Workforce feedback", f["feedback"]), ("Date", fmt_date(f["date"], style=st))]
            return "ORGANISATIONAL CHANGE - MOC REVIEW FORM\n" + kv_lines(rows, rng) + "\n\n" + self._policy(f)
        s = [f"{f['sponsor']} proposes to {f['description']}, saving about {money(f['savings'], style=ms)} a year.",
             f"Staffing {after}.", f"{study[0].upper() + study[1:]}.",
             f"The console averages {f['alarm_rate']} alarms per 10 minutes and the unit sees about {f['upsets_per_month']} upsets a month"
             + ("; alarm rationalisation is complete." if f["alarms_rationalised"] else "; alarm rationalisation has not been done."),
             f"{f['cross_trained_pct']}% of the affected staff are cross-trained; planned overlap {f['overlap_months']} months.",
             f"Feedback: {f['feedback']}."]
        return (f"MEMO - {f['company']} {f['site']} - {fmt_date(f['date'], style=st)}\nTo: PSM committee\nFrom: {f['reviewer']}\n"
                f"Re: {f['moc']}\n\n" + noisy(join_sentences(s), rng, 0.1) + "\n\n" + self._policy(f))

    def questions(self):
        return {
            "org_change_decision": choice("How should the PSM committee decide on this organisational change?", {
                "approve_as_proposed": "Approve it as proposed.",
                "approve_with_transition_measures": "Approve it only with added transition measures such as overlap, training or alarm work.",
                "require_staffing_study_first": "Defer a decision until a workload or staffing study has been done.",
                "reject_change": "Reject it; it would leave the site below its written minimum staffing.",
            }),
            "main_risk": choice("What is the main process-safety risk this change creates?", {
                "upset_response_capacity": "Too few people at the board or in the field to handle an upset.",
                "loss_of_specialist_knowledge": "Losing specialist knowledge of the plant's safety systems.",
                "emergency_response_coverage": "Too few people to fill the emergency response roles.",
                "fatigue_and_handover": "Fatigue and errors at shift handover.",
            }),
            "upset_handling_strain": score("How much would the change strain the shift's ability to handle an upset?", [
                "None: the people handling upsets are unchanged.",
                "Slight: fewer people, but a light alarm load.",
                "Considerable: fewer people against a busy console or frequent upsets.",
                "Severe: thin cover against a heavy alarm load and frequent upsets.",
            ]),
            "transition_readiness": score("How well prepared is the transition to the new arrangement?", [
                "Unprepared: little cross-training and no overlap or alarm work.",
                "Partly: some cross-training, gaps elsewhere.",
                "Mostly: cross-training largely complete, with an overlap period.",
                "Fully: training, overlap, a passed study and rationalised alarms all in place.",
            ]),
            "minimum_staffing_met": noul(
                "After the change, the site still meets every minimum in its written staffing standard.",
                yes="Board operators, emergency team members and SIS engineering cover all stay at or above the standard's minimums.",
                no="At least one of those minimums would no longer be met."),
        }


WORKFLOW = Workflow(
    name="process_management_of_change",
    title="Chemical process management of change",
    description="Replacement-in-kind screening, hazard-review depth, pre-startup readiness, temporary-change extensions and organisational changes at chemical plants.",
    domain="process industry / process safety management",
    schemas=[ReplacementInKindScreening(), HazardReviewDepth(), PssrStartupReadiness(), TemporaryChangeExtension(),
             OrganisationalChangeReview()],
    notes="Invented companies, units, tags and site standards; decisions concern physical process changes under PSM-style "
          "rules, not IT change approval. Thresholds vary by site so the quoted standard must be read.",
)
