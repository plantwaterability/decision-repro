"""Repair and warranty adjudication: a manufacturer's service organisation rules on defect claims from its network.

Five schemas: coverage of a failed component against the written warranty terms; a returned-part teardown report and
the misuse exclusion's causation test; repair versus replacement under the service policy (and what happens to the
failed part); whether a technical service bulletin applies to a unit; and goodwill eligibility after expiry. The
products are appliances, HVAC equipment, vehicles, outboard engines, e-bikes, power tools and electronics from
invented manufacturers, and each state quotes only the clauses of the (fictional) warranty or service manual that
decide the case, with terms and thresholds sampled per case.

Constrained family (research/typed_synth_spec.md section 3, "warranty / returns claims"): this is defect adjudication
from the service side only. Every state is a technical artefact (dealer claim form, teardown report, field engineer's
note, bulletin check, field-office request); there is no customer conversation, no question about how to reply to
anyone, no churn or loyalty factor, and no reconciliation of a claim's labour or parts amounts against a bill.
`intended(f)` records the answer logic a careful warranty engineer would apply to the rendered facts; the builder
never uses it.
"""
from __future__ import annotations

from datetime import timedelta

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
)

MAKER_B = ["Appliances", "Climate Systems", "Motor Works", "Marine", "Electric Bikes", "Power Tools", "Electronics",
           "Home Products", "Outdoor Equipment"]


def _maker(rng, kind):
    return f"{pick(rng, ORG_A)} {kind}"


def _clamp(x, lo=0, hi=3):
    return max(lo, min(hi, x))


def _months(d0, d1):
    return (d1.year - d0.year) * 12 + (d1.month - d0.month) - (1 if d1.day < d0.day else 0)


DEALER_KIND = {"Climate Systems": ["Heating & Cooling", "HVAC Service"], "Appliances": ["Appliance Repair", "Service Centre"],
               "Home Products": ["Plumbing & Heating", "Service Centre"], "Motor Works": ["Motors", "Auto Service"],
               "Marine": ["Marine Service", "Boat Works"], "Electric Bikes": ["Cycle Works", "E-Bike Service"],
               "Electronics": ["Electronics Repair", "Service Centre"], "Power Tools": ["Tool Service", "Service Centre"],
               "Outdoor Equipment": ["Outdoor Power Service", "Service Centre"], "Power Systems": ["Generator Service", "Electrical Services"],
               "Mobility": ["Mobility Service", "Home Medical Equipment"]}
# product label -> manufacturer kind
PRODUCT_KIND = {"split heat pump outdoor unit": "Climate Systems", "gas furnace": "Climate Systems",
                "bottom-freezer refrigerator": "Appliances", "front-load washer": "Appliances", "chest freezer": "Appliances",
                "dishwasher": "Appliances", "50-gallon gas water heater": "Home Products", "compact SUV": "Motor Works",
                "90 hp outboard engine": "Marine", "cargo e-bike": "Electric Bikes", "15-inch laptop": "Electronics",
                "65-inch television": "Electronics", "power wheelchair": "Mobility", "standby generator": "Power Systems",
                "chainsaw": "Outdoor Equipment", "e-bike hub motor": "Electric Bikes", "smartphone": "Electronics",
                "pressure-washer pump": "Outdoor Equipment", "outboard lower unit": "Marine", "impact driver": "Power Tools",
                "laptop mainboard": "Electronics", "espresso machine boiler": "Appliances"}


def _dealer(rng, kind="Appliances"):
    return f"{pick(rng, ORG_A)} {pick(rng, DEALER_KIND.get(kind, ['Service Centre']))}"


# =========================================================================== schema 1: coverage determination
# family -> (product label, maker kind, component, base years, registered years, labour years, usage limit, unit)
FAMILIES = {
    "heat_pump": ("split heat pump outdoor unit", "Climate Systems", "compressor", 5, 10, 1, None, None),
    "furnace": ("gas furnace", "Climate Systems", "inducer motor", 5, 10, 1, None, None),
    "refrigerator": ("bottom-freezer refrigerator", "Appliances", "sealed-system compressor", 5, 10, 1, None, None),
    "washer": ("front-load washer", "Appliances", "direct-drive motor", 5, 10, 1, None, None),
    "water_heater": ("50-gallon gas water heater", "Home Products", "tank", 6, 9, 1, None, None),
    "vehicle": ("compact SUV", "Motor Works", "engine (internal lubricated parts)", 5, 5, 5, 100000, "km"),
    "outboard": ("90 hp outboard engine", "Marine", "powerhead", 3, 5, 3, None, None),
    "ebike": ("cargo e-bike", "Electric Bikes", "hub motor", 2, 3, 1, None, None),
    "laptop": ("15-inch laptop", "Electronics", "mainboard", 1, 2, 1, None, None),
}
# family -> origin -> evidence variants
EVIDENCE = {
    "heat_pump": {
        "manufacturing_defect": ["start winding open; insulation to ground 500 MOhm; contactor clean; charge within spec",
                                 "internal valve plate failure: no pumping, amps low, suction and discharge equalised"],
        "installation_or_modification": ["moisture in the filter drier, acid test positive: line set was not evacuated",
                                         "breaker sized 50 A against a 30 A nameplate maximum; terminals overheated"],
        "external_event": ["contactor and board burnt; surge marks at the disconnect after a logged lightning storm",
                           "fins crushed and line fractured: impact damage from a fallen limb"],
        "misuse_or_neglect": ["coil packed solid with cottonwood debris; head pressure 30% high; no maintenance in 6 years",
                              "unit buried in snow drift to the top of the coil for weeks; defrost never recovered"]},
    "furnace": {
        "manufacturing_defect": ["inducer bearing seized; motor windings intact; venting within spec",
                                 "inducer wheel cracked at the hub moulding; no debris in the housing"],
        "installation_or_modification": ["vent run 40% longer than the installation table allows; motor ran overloaded",
                                         "condensate trap omitted at install; water pooled in the inducer housing"],
        "external_event": ["housing scorched from a flue fire in the shared chimney", "flood line 30 cm up the cabinet"],
        "misuse_or_neglect": ["filter absent; inducer packed with lint; unit never serviced",
                              "drywall dust throughout the cabinet: run without a filter during renovation"]},
    "refrigerator": {
        "manufacturing_defect": ["compressor windings shorted to ground; start relay and overload normal",
                                 "refrigerant leak at the factory braze on the evaporator outlet; dye confirms"],
        "installation_or_modification": ["unit built into a cabinet with no rear ventilation gap; compressor cycling on thermal overload",
                                         "water line kinked behind the unit; ice maker flooded the compartment"],
        "external_event": ["relay and compressor burnt after a utility voltage swell (utility report attached)",
                           "cabinet punctured at the evaporator during a house move"],
        "misuse_or_neglect": ["condenser coil matted with pet hair; never cleaned; head pressure high",
                              "unit operated in an unheated garage below its rated ambient range"]},
    "washer": {
        "manufacturing_defect": ["stator winding shorted; rotor magnets intact; hall sensor normal",
                                 "rotor position sensor intermittent; drum and bearings normal"],
        "installation_or_modification": ["shipping bolts never removed; drum struck the cabinet and the motor mount cracked",
                                         "unit installed on a sprung timber floor without the stacking kit; violent vibration"],
        "external_event": ["control and motor driver burnt after a power surge (neighbouring units also damaged)",
                           "unit submerged during a basement flood"],
        "misuse_or_neglect": ["drum routinely overloaded with heavy bedding; suspension rods snapped",
                              "coins and an underwire jammed in the pump; motor stalled repeatedly"]},
    "water_heater": {
        "manufacturing_defect": ["tank leak at the bottom weld seam; anode 60% remaining; sediment light",
                                 "glass lining failure with pinhole rust spot; anode largely intact"],
        "installation_or_modification": ["closed system with no expansion tank; T&P valve discharging, tank bulged",
                                         "flue connected with the wrong vent pipe; heat damage at the top head"],
        "external_event": ["tank split by freezing in an unheated crawl space", "earthquake strap absent; tank shifted and inlet fractured"],
        "misuse_or_neglect": ["anode fully consumed and never replaced; heavy sediment; tank perforated",
                              "thermostat set to maximum and tank never flushed in hard-water area; scale throughout"]},
    "vehicle": {
        "manufacturing_defect": ["spun rod bearing on cylinder 3; oil level and condition normal; service history complete",
                                 "cracked piston ring land on cylinder 2; no detonation marks; oil analysis normal"],
        "installation_or_modification": ["ECU reflashed with an aftermarket performance tune; detonation damage on two pistons",
                                         "aftermarket intercooler fitted; boost logged above the factory limit"],
        "external_event": ["hydrolocked: water line on the air box after driving through flood water",
                           "oil pan holed by road debris; engine ran without oil"],
        "misuse_or_neglect": ["sludge under the valve cover; last oil change 28,000 km ago against a 10,000 km interval",
                              "repeated over-rev events logged in the ECU; bent valves"]},
    "outboard": {
        "manufacturing_defect": ["piston ring land cracked; oil injection working; water pump normal",
                                 "crankshaft bearing failure; correct oil in use; cooling normal"],
        "installation_or_modification": ["engine mounted too high on the transom; water pickup ventilating, powerhead overheated",
                                         "propeller pitched for 1,000 rpm over the rated range; over-rev damage"],
        "external_event": ["struck a submerged object: skeg bent, driveshaft twisted", "engine submerged when the boat swamped at the mooring"],
        "misuse_or_neglect": ["overheated: water-pump impeller not replaced in five seasons", "old ethanol fuel varnished the injectors; ran lean"]},
    "ebike": {
        "manufacturing_defect": ["hall sensor board failed; no heat discolouration; gears intact",
                                 "motor cable insulation chafed at the axle exit from the factory routing"],
        "installation_or_modification": ["controller firmware altered to raise the assist limit; windings overheated",
                                         "third-party higher-voltage battery fitted; controller and motor burnt"],
        "external_event": ["motor housing cracked in a fall; water entered through the crack", "bike submerged in a flooded underpass"],
        "misuse_or_neglect": ["gear set stripped; trip log shows loads 40% over the rated maximum",
                              "axle bearings seized with salt corrosion; never cleaned after winter use"]},
    "laptop": {
        "manufacturing_defect": ["GPU not detected; solder-joint fracture pattern typical of reflow defects; no liquid indicators tripped",
                                 "charging circuit failed at the power IC; no physical damage"],
        "installation_or_modification": ["third-party memory module installed; slot pins bent, short on the 3.3 V rail",
                                         "non-OEM power adapter with the wrong polarity used; input stage burnt"],
        "external_event": ["liquid contact indicators tripped; corrosion at the keyboard connector", "hinge torn and board cracked from a drop"],
        "misuse_or_neglect": ["fan and heatsink fully clogged with lint; thermal shutdowns logged for months",
                              "battery swollen after storage in a hot car for a summer"]},
}
ORIGINS = ["manufacturing_defect", "installation_or_modification", "external_event", "misuse_or_neglect"]
EVQ = ["technician's opinion only", "measurements", "measurements and photos", "measurements, photos and a fault-data download"]
DOCS = ["serial-plate photo", "proof of installation date", "maintenance records", "fault codes"]


class FailureCoverageDetermination(Schema):
    name = "failure_coverage_determination"
    description = "A service network's warranty claim for a failed component, judged on the failure evidence against the written warranty terms."
    surfaces = {"json": 0.5, "dealer_claim_form": 0.25, "field_engineer_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng, clear=0.25, borderline=0.45, conflicting=0.30)
        fam = pick(rng, list(FAMILIES))
        label, mk, comp, base_y, reg_y, lab_y, ulim, unit = FAMILIES[fam]
        reg_days = pick(rng, [60, 90])
        origin = pick_w(rng, {"manufacturing_defect": 5.5, "installation_or_modification": 2, "external_event": 1.8,
                              "misuse_or_neglect": 2.2})
        claimed = origin if chance(rng, 0.6) else "manufacturing_defect"
        registered = pick_w(rng, {"on time": 5, "late": 1.5, "never": 2})
        reg_day = (rng.randint(1, reg_days) if registered == "on time" else rng.randint(reg_days + 1, reg_days + 200)
                   if registered == "late" else None)
        commercial = chance(rng, 0.1) and fam not in ("vehicle",)
        term_y = 1 if commercial else (reg_y if registered == "on time" else base_y)
        if reg == "borderline":
            age = near(rng, term_y * 12, 3, lo=1, integer=True) if chance(rng, 0.6) else rng.randint(2, term_y * 12)
            if registered != "never" and chance(rng, 0.3):
                reg_day = near(rng, reg_days, 6, lo=1, integer=True)
        elif reg == "clear":
            age = (pick(rng, [rng.randint(1, lab_y * 12), rng.randint(2, max(3, term_y * 12 - 6))]) if chance(rng, 0.6)
                   else term_y * 12 + rng.randint(6, 30))
        else:
            age = pick(rng, [rng.randint(1, lab_y * 12), rng.randint(2, term_y * 12 + 12)])
        evq = pick_w(rng, {0: 1, 1: 2, 2: 3, 3: 2.6})
        docs = [d for d in DOCS if chance(rng, 0.7)]
        install_proof = "proof of installation date" in docs
        mfg_gap = rng.randint(1, 14)
        usage = None
        if ulim:
            usage = (near(rng, ulim, ulim * 0.06, integer=True) if reg == "borderline" and chance(rng, 0.4)
                     else int(rng.uniform(0.15, 1.15) * ulim))
        if reg == "conflicting":
            twist = pick(rng, ["coded_defect", "no_install_proof", "weak_evidence"])
            if twist == "coded_defect":
                origin, claimed = pick(rng, ORIGINS[1:]), "manufacturing_defect"
                evq = pick(rng, [2, 3])
            elif twist == "no_install_proof":
                install_proof = False
                docs = [d for d in docs if d != "proof of installation date"]
                age = rng.randint(max(1, term_y * 12 - 10), term_y * 12 - 1)
                mfg_gap = rng.randint(6, 14)
            else:
                evq, origin = 0, pick(rng, ORIGINS[1:])
        failed = rand_date(rng)
        install = failed - timedelta(days=int(age * 30.44))
        return {
            "regime": reg, "family": fam, "product": label, "component": comp, "maker": _maker(rng, mk),
            "base_years": base_y, "reg_years": reg_y, "labour_years": lab_y, "usage_limit": ulim, "usage_unit": unit,
            "usage": usage, "reg_days": reg_days, "registered": registered, "reg_day": reg_day, "commercial": commercial,
            "install_date": install, "fail_date": failed, "install_proof": install_proof, "mfg_gap_months": mfg_gap,
            "origin": origin, "claimed_origin": claimed, "evidence": pick(rng, EVIDENCE[fam][origin]), "evq": evq,
            "docs": docs, "dealer": _dealer(rng, mk), "tech": person(rng), "claim": ident(rng, "WC", 6, "-"),
            "model": f"{pick(rng, ['HX', 'TR', 'QS', 'MV', 'PL', 'ZR'])}-{rng.randint(100, 990)}",
            "serial": f"{rng.randint(10, 26):02d}{pick(rng, 'ABCDEFGHJK')}{rng.randint(100000, 999999)}",
            "lw": f"LW-{rng.randint(10, 99)}", "site": pick(rng, ["rental property", "restaurant", "home", "home", "home"]) if commercial else "home",
        }

    def _term_years(self, f):
        if f["commercial"]:
            return 1
        on_time = f["reg_day"] is not None and f["reg_day"] <= f["reg_days"]
        return f["reg_years"] if on_time else f["base_years"]

    def _age_months(self, f):
        a = _months(f["install_date"], f["fail_date"])
        return a if f["install_proof"] else a + f["mfg_gap_months"]

    def intended(self, f):
        term = self._term_years(f)
        age = self._age_months(f)
        in_term = age <= term * 12 and (f["usage"] is None or f["usage"] <= f["usage_limit"])
        conf = f["evq"]
        if not in_term:
            rul = "not_covered_term_expired"
        elif f["origin"] != "manufacturing_defect":
            rul = "not_covered_exclusion" if conf >= 2 else "inspect_before_ruling"
        elif conf == 0:
            rul = "inspect_before_ruling"
        else:
            rul = "covered_parts_and_labour" if age <= f["labour_years"] * 12 and not f["commercial"] else "covered_parts_only"
        nd = len(f["docs"])
        return {"coverage_ruling": rul, "failure_origin": f["origin"], "diagnostic_confidence": conf,
                "claim_documentation": 0 if nd <= 1 else nd - 1, "within_coverage_term": "yes" if in_term else "no"}

    def _rules(self, f):
        use = f" or {f['usage_limit']:,} {f['usage_unit']}, whichever first" if f["usage_limit"] else ""
        reg = (f"{f['reg_years']} if registered within {f['reg_days']} days" if f["reg_years"] != f["base_years"]
               else "registration does not change it")
        return [f"{f['lw']} §2: {f['component']}: {f['base_years']} yrs from installation ({reg}){use}; labour "
                f"{f['labour_years']} yr{'s' if f['labour_years'] > 1 else ''}. Commercial use of a residential product: 1 yr total. No proof of installation date: "
                "the term runs from the build date.",
                f"§6: excludes failures caused by installation faults or modifications, external events (surge, flood, fire, "
                "impact), misuse, or missed required maintenance."]

    def _reg_txt(self, f):
        if f["registered"] == "never" or f["reg_day"] is None:
            return "not registered"
        return f"registered {f['reg_day']} days after installation"

    def render(self, f, rng, surface):
        mfg = f["install_date"] - timedelta(days=int(f["mfg_gap_months"] * 30.44))
        inst = fmt_date(f["install_date"], rng) if f["install_proof"] else "per owner; no invoice"
        use = f"{f['usage']:,} {f['usage_unit']}" if f["usage"] is not None else None
        coded = f["claimed_origin"].replace("_", " ")
        if surface == "json":
            return dumps(prune({
                "claim": {"id": f["claim"], "dealer": f["dealer"], "technician": f["tech"], "coded_cause": coded},
                "unit": {"product": f"{f['maker']} {f['product']}", "model": f["model"], "serial": f["serial"],
                         "manufactured": mfg.strftime("%Y-%m"), "installed": inst, "registration": self._reg_txt(f),
                         "use": "commercial (" + f["site"] + ")" if f["commercial"] else "residential/private",
                         "odometer": use},
                "failure": {"component": f["component"], "failed_on": f["fail_date"].isoformat(), "findings": f["evidence"],
                            "evidence_basis": EVQ[f["evq"]]},
                "attachments": f["docs"] or ["none"], "terms": self._rules(f),
            }))
        if surface == "dealer_claim_form":
            rows = [("Claim", f["claim"]), ("Dealer", f["dealer"]), ("Unit", f"{f['maker']} {f['product']} {f['model']}"),
                    ("Serial / built", f"{f['serial']} / {mfg.strftime('%b %Y')}"), ("Installed", inst),
                    ("Registration", self._reg_txt(f)), ("Use", "commercial - " + f["site"] if f["commercial"] else "residential"),
                    ("Odometer", use), ("Failed", fmt_date(f["fail_date"], rng)), ("Component", f["component"]),
                    ("Cause code", coded), ("Findings", noisy(f["evidence"], rng, 0.1)), ("Evidence", EVQ[f["evq"]]),
                    ("Attached", ", ".join(f["docs"]) or "nothing")]
            rule = (f"Coverage ({f['lw']}): {f['component']} {f['base_years']}y"
                    + (f"/{f['reg_years']}y if registered <= {f['reg_days']}d" if f["reg_years"] != f["base_years"] else "")
                    + (f", {f['usage_limit']:,} {f['usage_unit']} cap" if f["usage_limit"] else "")
                    + f", labour {f['labour_years']}y; commercial use 1y; no install proof -> from build date; "
                    "§6 excludes installation/modification, external events, misuse, missed maintenance.")
            return "WARRANTY CLAIM - SERVICE NETWORK\n" + kv_lines(rows, rng) + "\n\n" + rule
        para = [f"Claim {f['claim']} from {f['dealer']}: {f['maker']} {f['product']} ({f['model']}, s/n {f['serial']}), "
                f"{f['component']} failed {fmt_date(f['fail_date'], rng)}.",
                f"Built {mfg.strftime('%B %Y')}; installation date {inst}; {self._reg_txt(f)}.",
                (f"Installed at a {f['site']}." if f["commercial"] else ""), (f"Odometer {use}." if use else ""),
                f"Dealer coded it as {coded}.", f"What I see: {f['evidence']} (basis: {EVQ[f['evq']]}).",
                f"Attached: {', '.join(f['docs']) or 'nothing'}."]
        return (f"Field engineer note - {f['tech']}\n\n" + noisy(join_sentences(para), rng, 0.08) + "\n\n"
                + bullets(self._rules(f), rng))

    def questions(self):
        return {
            "coverage_ruling": choice("How should this warranty claim be ruled?", {
                "covered_parts_and_labour": "Covered: parts and labour.",
                "covered_parts_only": "Covered for parts only; the labour coverage has ended.",
                "not_covered_term_expired": "Not covered: the failure falls outside the coverage term.",
                "not_covered_exclusion": "Not covered: an exclusion in the terms applies.",
                "inspect_before_ruling": "No ruling yet; the part or unit must be inspected first.",
            }),
            "failure_origin": choice("What most likely caused the failure?", {
                "manufacturing_defect": "A defect in materials or workmanship.",
                "installation_or_modification": "Faulty installation or a modification to the product.",
                "external_event": "An outside event such as a surge, flood, fire or impact.",
                "misuse_or_neglect": "Misuse, overloading or missed maintenance.",
            }),
            "diagnostic_confidence": score("How firmly does the evidence establish the cause of failure?", [
                "Barely: an opinion with nothing measured.",
                "Somewhat: measurements, but gaps or inconsistencies.",
                "Firmly: measurements and photos that agree.",
                "Conclusively: measurements, photos and logged data all point the same way.",
            ]),
            "claim_documentation": score("How complete is the claim's supporting documentation?", [
                "Bare: little or nothing attached.",
                "Thin: two supporting items.",
                "Good: three supporting items.",
                "Complete: serial photo, installation proof, maintenance records and fault codes.",
            ]),
            "within_coverage_term": noul(
                "The failure occurred within the coverage term that applies to this component.",
                yes="Counting from the right start date, and with registration and use taken into account, the failure falls inside the term.",
                no="The failure falls outside the applicable term, or past the usage limit."),
        }


# =========================================================================== schema 2: teardown and misuse
# product -> list of (failure mode, condition found or None, condition kind, causal strength)
TEARDOWN = {
    "chainsaw": [("piston and cylinder scored on the exhaust side", "fuel sample mixed at about 100:1 against the specified 50:1", "misuse", "strong"),
                 ("piston scored on the intake side", "air filter torn; fine dust throughout the intake tract", "misuse", "strong"),
                 ("ignition module open circuit", "bar and chain heavily worn", "misuse", "none"),
                 ("crankshaft seal leaking at the clutch side", None, None, "none"),
                 ("piston scuffed lightly", "carburettor limiter caps removed and mixture screws turned", "modification", "plausible")],
    "e-bike hub motor": [("stator windings discoloured; varnish burnt", "controller firmware altered to raise the assist limit", "modification", "strong"),
                         ("hall sensor board failed", "controller firmware altered to raise the assist limit", "modification", "none"),
                         ("planetary gear set stripped", "trip log shows loads well over the rated maximum", "misuse", "plausible"),
                         ("hall sensor board failed", None, None, "none")],
    "smartphone": [("display not powering; corrosion on the display connector", "liquid contact indicator tripped", "liquid", "strong"),
                   ("battery swollen", "liquid contact indicator tripped; no corrosion anywhere", "liquid", "none"),
                   ("charge port dead", "pocket lint compacted in the port", "misuse", "plausible"),
                   ("baseband chip failure", None, None, "none"),
                   ("display flickers; display flex cable torn", "screen previously replaced by a third-party shop with a non-OEM panel", "modification", "strong")],
    "pressure-washer pump": [("manifold cracked", "ice-expansion fracture pattern; unit stored without antifreeze", "freezing", "strong"),
                             ("manifold cracked", "casting porosity at the crack origin; no freeze pattern", None, "none"),
                             ("plunger seals worn through", "inlet filter missing; sand in the pump", "misuse", "strong"),
                             ("unloader valve stuck", "pump run dry; seals glazed", "misuse", "plausible"),
                             ("manifold burst from overpressure", "unloader spring replaced with a stiffer aftermarket spring", "modification", "strong")],
    "outboard lower unit": [("pinion and forward gear chipped", "water in the gear oil; fishing line cut the prop-shaft seal", "misuse", "strong"),
                            ("prop-shaft bearing failed", "gear oil low and dark; last change four seasons ago", "misuse", "plausible"),
                            ("shift-shaft seal leaking", None, None, "none"),
                            ("gearcase cracked", "skeg bent and scraped", "impact", "strong"),
                            ("forward gear teeth spalled", "non-OEM gear set fitted at an independent shop", "modification", "plausible")],
    "impact driver": [("armature windings burnt", "vents packed with drywall dust; run as a production tool", "misuse", "plausible"),
                      ("trigger switch failed", "housing cracked from a drop", "impact", "none"),
                      ("motor bearing seized", None, None, "none"),
                      ("anvil sheared", "used with an oversize breaker bar extension", "misuse", "strong"),
                      ("motor and trigger burnt", "speed-control board replaced with a non-OEM part", "modification", "strong")],
    "laptop mainboard": [("short at the keyboard connector", "liquid residue under the keyboard; indicator tripped", "liquid", "strong"),
                         ("GPU not detected; reflow-type solder fracture", "case screws replaced with non-OEM screws", "modification", "none"),
                         ("fan controller failed", None, None, "none"),
                         ("power rail burnt", "non-OEM charger with the wrong voltage in the box", "misuse", "plausible"),
                         ("short on the 3.3 V rail at the memory slot", "third-party memory upgrade installed; slot pins bent", "modification", "strong")],
    "espresso machine boiler": [("heating element failed open", "heavy scale; descaling indicator ignored for months", "misuse", "strong"),
                                ("boiler weld seeping", None, None, "none"),
                                ("pressure switch failed", "unit opened; non-OEM gasket fitted", "modification", "none"),
                                ("boiler deformed by overpressure", "safety valve replaced with a non-OEM valve rated higher", "modification", "strong")],
}
METHODS = ["visual inspection only", "visual and dimensional checks",
           "visual, dimensional and metallurgical or data-log analysis",
           "full root-cause analysis against a reference unit"]
SEALS = {"intact": "factory seals and fasteners intact", "broken": "tamper seal broken", "non_oem": "non-OEM fasteners found"}


class TeardownMisuseDetermination(Schema):
    name = "teardown_misuse_determination"
    description = "A returned-part teardown report: is the failure a covered defect, or does a misuse or tampering exclusion apply under the causation rule?"
    surfaces = {"json": 0.5, "teardown_report": 0.25, "lab_log": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng, clear=0.25, borderline=0.45, conflicting=0.30)
        prod = pick(rng, list(TEARDOWN))
        rows = TEARDOWN[prod]
        row = pick(rng, rows)
        mods = [r for r in rows if r[2] == "modification"]
        if mods and chance(rng, 0.3):
            row = pick(rng, mods)
        method = pick_w(rng, {0: 2, 1: 3, 2: 3, 3: 1.5})
        seal = pick_w(rng, {"intact": 6, "broken": 1.2, "non_oem": 1})
        if row[2] == "modification" and chance(rng, 0.7):
            seal = pick(rng, ["broken", "non_oem"])
        if reg == "clear":
            row = pick(rng, [r for r in rows if r[3] in ("strong",) or r[1] is None] + [r for r in rows if r[1] is None]
                       + [r for r in mods if r[3] == "strong"])
            method = pick(rng, [2, 3])
        elif reg == "borderline":
            cands = [r for r in rows if r[3] == "plausible"] or rows
            row = pick(rng, cands) if chance(rng, 0.55) else row
        else:
            cands = [r for r in rows if r[1] is not None and r[3] == "none"]
            row = pick(rng, cands) if cands and chance(rng, 0.65) else row
            if chance(rng, 0.35):
                seal = pick(rng, ["broken", "non_oem"])
        if row[2] == "modification" and any(w in row[1] for w in ("opened", "replaced", "fitted", "installed")):
            seal = pick(rng, ["broken", "non_oem"])
        return {
            "regime": reg, "product": prod, "failure": row[0], "condition": row[1], "condition_kind": row[2],
            "strength": row[3], "method": method, "seal": seal,
            "dealer_note": pick(rng, ["normal use per dealer", "dealer reports normal domestic use", "dealer: no abuse seen",
                                      "dealer: used on a job site", ""]),
            "maker": _maker(rng, PRODUCT_KIND[prod]), "rma": ident(rng, "RMA", 6, "-"), "analyst": person(rng), "lw": f"LW-{rng.randint(10, 99)}", "date": rand_date(rng),
            "serial": f"{pick(rng, 'KLMNPQRST')}{rng.randint(1000000, 9999999)}",
        }

    def _causal(self, f):
        if f["condition"] is None:
            return False
        return f["strength"] == "strong" or (f["strength"] == "plausible" and f["method"] >= 2)

    def intended(self, f):
        causal = self._causal(f)
        opened = f["seal"] != "intact" or f["condition_kind"] == "modification"
        if causal and f["condition_kind"] == "modification":
            det = "excluded_unauthorised_repair"
        elif causal:
            det = "excluded_misuse_causal"
        elif f["condition"] is not None and f["strength"] == "plausible":
            det = "further_analysis_required"
        elif f["condition"] is not None or opened:
            det = "covered_incidental_findings"
        elif f["method"] == 0:
            det = "further_analysis_required"
        else:
            det = "covered_defect_confirmed"
        st = f["strength"] if f["condition"] is not None else "none"
        link = 0 if st == "none" else (3 if f["method"] >= 2 else 2) if st == "strong" else (2 if f["method"] >= 2 else 1)
        return {"teardown_determination": det, "causal_link_strength": link, "analysis_depth": f["method"],
                "causation_test_met": "yes" if causal else "no", "unauthorised_opening": "yes" if opened else "no"}

    def _rules(self, f):
        return [f"{f['lw']} §7: not covered: misuse, overload beyond rating, wrong fuel, lubricant or charger, liquid or "
                "dust ingress, freezing, impact, or repair or modification by anyone outside the authorised network.",
                f"{f['lw']} §7.3: an exclusion applies only where the teardown shows the condition caused or contributed "
                "to the failure; incidental findings do not void coverage. A plausible but unconfirmed link needs deeper analysis."]

    def render(self, f, rng, surface):
        cond = f["condition"] or "no excluded condition found"
        if surface == "json":
            return dumps(prune({
                "return": {"rma": f["rma"], "product": f"{f['maker']} {f['product']}", "serial": f["serial"],
                           "dealer_note": f["dealer_note"] or None},
                "teardown": {"failure_mode": f["failure"], "other_findings": cond, "seals": SEALS[f["seal"]],
                             "method": METHODS[f["method"]], "analyst": f["analyst"]},
                "terms": self._rules(f),
            }))
        if surface == "teardown_report":
            rows = [("RMA", f["rma"]), ("Part", f"{f['maker']} {f['product']}, s/n {f['serial']}"), ("Received", fmt_date(f["date"], rng)),
                    ("Dealer note", f["dealer_note"] or None), ("Method", METHODS[f["method"]]), ("Failure mode", f["failure"]),
                    ("Other findings", cond), ("Seals / fasteners", SEALS[f["seal"]]), ("Analyst", f["analyst"])]
            return ("TEARDOWN REPORT\n" + kv_lines(rows, rng) + f"\n\nRule ({f['lw']} §7.3): exclusion only if the condition "
                    "caused or contributed to the failure; incidental findings don't void coverage; a plausible but "
                    "unconfirmed link needs deeper analysis.")
        t = f["date"]
        ev = [f"{f['rma']} received: {f['maker']} {f['product']} s/n {f['serial']}"]
        if f["dealer_note"]:
            ev.append(f"claim note: {f['dealer_note']}")
        ev += [f"seals: {SEALS[f['seal']]}", f"method: {METHODS[f['method']]}", f"failure mode: {f['failure']}",
               f"other findings: {cond}"]
        mins, lines = 8 * 60 + rng.randint(0, 40), []
        for e in ev:
            lines.append(f"{t.isoformat()} {mins // 60:02d}:{mins % 60:02d}  {e}")
            mins += rng.randint(15, 95)
        return f"LAB LOG ({f['analyst']})\n" + "\n".join(lines) + "\n\n" + "\n".join(self._rules(f))

    def questions(self):
        return {
            "teardown_determination": choice("What should the warranty lab conclude from this teardown?", {
                "covered_defect_confirmed": "A covered defect, with nothing found that would exclude it.",
                "covered_incidental_findings": "Covered: whatever else was found did not cause the failure.",
                "excluded_misuse_causal": "Excluded: misuse, neglect or an outside cause led to the failure.",
                "excluded_unauthorised_repair": "Excluded: an unauthorised repair or modification led to the failure.",
                "further_analysis_required": "No conclusion yet; deeper analysis is needed.",
            }),
            "causal_link_strength": score("How strongly do the findings tie an excluded condition to the failure?", [
                "Not at all: no excluded condition, or one unrelated to the failure.",
                "Weakly: a possible link that the analysis has not tested.",
                "Clearly: the link is evident, though not proven in depth.",
                "Conclusively: the analysis demonstrates the link.",
            ]),
            "analysis_depth": score("How thorough was the lab's analysis?", [
                "Cursory: a visual look only.",
                "Basic: visual and dimensional checks.",
                "Thorough: adds metallurgical or logged-data analysis.",
                "Complete: a full root-cause analysis against a reference unit.",
            ]),
            "causation_test_met": noul(
                "The teardown links an excluded condition to the failure, as the terms require before an exclusion can be applied.",
                yes="An excluded condition was found and the analysis shows it caused or contributed to the failure.",
                no="No excluded condition was found, or the one found is incidental or its link is unconfirmed."),
            "unauthorised_opening": noul(
                "There are signs that the unit was opened, repaired or modified outside the authorised network."),
        }


# =========================================================================== schema 3: repair vs replace
# (product, failed part, typical replacement cost, typical repair cost, downtime base, temporary measure)
RVR_PRODUCTS = [
    ("bottom-freezer refrigerator", "sealed-system compressor", 1400, 780, 2, "loan fridge"),
    ("front-load washer", "drum bearing assembly", 900, 420, 1, None),
    ("15-inch laptop", "mainboard", 1300, 520, 1, "loan laptop"),
    ("65-inch television", "display panel", 1100, 760, 0, None),
    ("split heat pump outdoor unit", "compressor", 4200, 2100, 3, "portable electric heaters"),
    ("gas furnace", "heat exchanger", 3200, 1750, 3, "portable electric heaters"),
    ("cargo e-bike", "hub motor", 3100, 640, 1, None),
    ("chest freezer", "compressor", 700, 380, 2, None),
    ("dishwasher", "control board and wash pump", 850, 330, 0, None),
    ("power wheelchair", "drive motor and gearbox", 6800, 1900, 3, "loan chair"),
]
NEW_MODES = ["a crack pattern not seen in the known-issues list", "a failure mode new to this part number",
             "an unusual burn pattern at the connector"]


class RepairVsReplace(Schema):
    name = "repair_vs_replace"
    description = "An authorised service centre's in-warranty remedy: repair, refurbished or new replacement, plus the handling of the failed part."
    surfaces = {"json": 0.5, "service_order": 0.25, "warranty_desk_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng, clear=0.25, borderline=0.45, conflicting=0.30)
        prod, part, repl0, rep0, dt_base, temp = pick(rng, RVR_PRODUCTS)
        pct = pick(rng, [50, 60, 70])
        n_rep = pick(rng, [3, 3, 4])
        lead_max = pick(rng, [10, 14, 21])
        new_days = pick(rng, [30, 60, 90])
        repl = round(repl0 * rng.uniform(0.85, 1.2), -1)
        scen = pick_w(rng, {"repair": 3.5, "cost": 2, "repeat": 1.5, "lead": 1.5, "undiagnosed": 1.2})
        ratio = rng.uniform(0.2, pct / 100 - 0.08)
        repeats = pick_w(rng, {0: 5, 1: 2.5, n_rep - 2: 1.5})
        lead = rng.randint(2, lead_max - 2)
        if scen == "cost":
            ratio = near(rng, pct / 100 + 0.04, 0.06) if reg == "borderline" else rng.uniform(pct / 100 + 0.05, 1.05)
        elif scen == "repeat":
            repeats = n_rep - 1 if reg != "borderline" else pick(rng, [n_rep - 2, n_rep - 1])
        elif scen == "lead":
            lead = near(rng, lead_max + 2, 3, lo=1, integer=True) if reg == "borderline" else rng.randint(lead_max + 3, lead_max + 40)
        elif reg == "borderline":
            ratio = near(rng, pct / 100 - 0.03, 0.05)
        if scen in ("cost", "repeat", "lead") and reg != "clear":
            if chance(rng, 0.35):
                repeats = max(repeats, pick(rng, [1, n_rep - 1]))
            if chance(rng, 0.3):
                lead = rng.randint(lead_max + 1, lead_max + 25)
        confirmed = scen != "undiagnosed"
        if reg == "conflicting" and chance(rng, 0.4):
            confirmed = False
        own_days = pick_w(rng, {rng.randint(5, new_days): 2, near(rng, new_days, 8, lo=1, integer=True): 1.5,
                                rng.randint(new_days + 1, 1000): 4})
        refurb = chance(rng, 0.7)
        part_cost = round(rep0 * rng.uniform(0.35, 0.75), -1)
        recover_min = pick(rng, [150, 250, 400])
        supplier_deal = chance(rng, 0.55)
        mode_new = chance(rng, 0.18)
        audit = chance(rng, 0.12)
        season = pick(rng, ["January", "February", "July", "August", "October", "April", "December"])
        return {
            "regime": reg, "product": prod, "part": part, "replacement_cost": repl, "repair_estimate": round(repl * ratio, -1),
            "pct_limit": pct, "repeat_limit": n_rep, "prior_repairs": repeats, "lead_days": lead, "lead_max": lead_max,
            "new_days": new_days, "owned_days": own_days, "refurb_stock": refurb, "diagnosis_confirmed": confirmed,
            "part_cost": part_cost, "recover_min": recover_min, "supplier_agreement": supplier_deal, "new_failure_mode": mode_new,
            "mode_text": pick(rng, NEW_MODES), "audit_sample": audit, "downtime_base": dt_base, "temporary_measure": temp if chance(rng, 0.5) else None,
            "month": season, "centre": _dealer(rng, PRODUCT_KIND[prod]), "tech": person(rng), "so": ident(rng, "SO", 6, "-"),
            "spm": f"SPM-{rng.randint(2, 9)}", "maker": _maker(rng, PRODUCT_KIND[prod]),
        }

    def intended(self, f):
        ratio = f["repair_estimate"] / f["replacement_cost"]
        meets = (ratio > f["pct_limit"] / 100 or f["prior_repairs"] + 1 >= f["repeat_limit"] or f["lead_days"] > f["lead_max"])
        if not f["diagnosis_confirmed"]:
            rem = "escalate_for_technical_review"
        elif meets:
            rem = "replace_with_new" if (f["owned_days"] <= f["new_days"] or not f["refurb_stock"]) else "replace_with_refurbished"
        else:
            rem = "repair_unit"
        if f["new_failure_mode"]:
            part = "send_to_product_engineering"
        elif f["audit_sample"]:
            part = "hold_for_quality_audit"
        elif f["supplier_agreement"] and f["part_cost"] >= f["recover_min"]:
            part = "return_to_supplier"
        else:
            part = "scrap_locally"
        via = 3 - (f["prior_repairs"] >= 1) - (f["prior_repairs"] >= 2) - (f["lead_days"] > f["lead_max"]) - (ratio > f["pct_limit"] / 100)
        heating = f["product"] in ("split heat pump outdoor unit", "gas furnace")
        winter = f["month"] in ("December", "January", "February")
        dt = f["downtime_base"]
        if heating and not winter:
            dt = 1
        if f["temporary_measure"]:
            dt -= 1
        return {"warranty_remedy": rem, "returned_part_handling": part, "repair_viability": _clamp(via),
                "downtime_pressure": _clamp(dt), "replacement_criteria_met": "yes" if meets else "no"}

    def _rules(self, f, short=False):
        if short:
            return (f"Replace if repair > {f['pct_limit']}% of replacement cost, if this is repair no. {f['repeat_limit']} "
                    f"of the same fault, or if parts take > {f['lead_max']} days; new unit within {f['new_days']} days of "
                    f"purchase or if no refurbished stock ({f['spm']} §4.2). Parts: new failure modes to engineering; audit "
                    f"samples held; >= {money(f['recover_min'], style='sym')} on a recovery agreement back to supplier (§9).")
        return [f"{f['spm']} §4.2: replace instead of repair if the estimate > {f['pct_limit']}% of replacement cost, if this "
                f"would be repair no. {f['repeat_limit']} of the same fault, or if parts take > {f['lead_max']} days; the "
                f"replacement is new within {f['new_days']} days of purchase or when no refurbished stock exists, else refurbished.",
                f"§9: unknown failure modes -> product engineering; audit samples held; parts >= {money(f['recover_min'], style='sym')} "
                "under a supplier-recovery agreement -> supplier; else scrap."]

    def _diag(self, f):
        return ("confirmed by test" if f["diagnosis_confirmed"]
                else "suspected from symptoms; not confirmed by test")

    def render(self, f, rng, surface):
        ms = pick(rng, ["sym", "sym", "code"])
        mode = f["mode_text"] if f["new_failure_mode"] else "matches a known-issues entry"
        tmp = f"{f['temporary_measure']} provided" if f["temporary_measure"] else None
        if surface == "json":
            return dumps(prune({
                "service_order": {"id": f["so"], "centre": f["centre"], "unit": f"{f['maker']} {f['product']}",
                                  "owned_days": f["owned_days"], "month": f["month"], "temporary_measure": tmp},
                "diagnosis": {"failed_part": f["part"], "status": self._diag(f), "failure_mode": mode,
                              "earlier_repairs_same_fault": f["prior_repairs"]},
                "costs": {"repair_estimate": money(f["repair_estimate"], style=ms), "replacement_cost": money(f["replacement_cost"], style=ms),
                          "failed_part_value": money(f["part_cost"], style=ms)},
                "logistics": {"parts_lead_days": f["lead_days"], "refurbished_stock": "available" if f["refurb_stock"] else "none",
                              "supplier_recovery_agreement": f["supplier_agreement"], "quality_audit_sample": f["audit_sample"] or None},
                "policy": self._rules(f),
            }))
        if surface == "service_order":
            rows = [("SO", f"{f['so']} - {f['centre']}"), ("Unit", f"{f['maker']} {f['product']} (owned {f['owned_days']} days)"),
                    ("Failed part", f"{f['part']} - {self._diag(f)}"), ("Failure mode", mode),
                    ("Earlier repairs, same fault", f["prior_repairs"]),
                    ("Repair est. / replacement", f"{money(f['repair_estimate'], style=ms)} / {money(f['replacement_cost'], style=ms)}"),
                    ("Parts lead time", f"{f['lead_days']} days"), ("Refurb stock", "yes" if f["refurb_stock"] else "no"),
                    ("Part value / recovery agreement", f"{money(f['part_cost'], style=ms)} / {'yes' if f['supplier_agreement'] else 'no'}"),
                    ("Audit sample", "yes" if f["audit_sample"] else None), ("Opened", f["month"]), ("Temporary measure", tmp)]
            return "SERVICE ORDER - WARRANTY REMEDY\n" + kv_lines(rows, rng) + "\n\n" + self._rules(f, short=True)
        para = [f"{f['so']}: {f['maker']} {f['product']}, owned {f['owned_days']} days; {f['part']} failed ({self._diag(f)}).",
                f"Failure mode {mode}.", ("No earlier repairs of this fault." if not f["prior_repairs"] else
                                          f"Same fault repaired {f['prior_repairs']} time{'s' if f['prior_repairs'] > 1 else ''} before."),
                f"Repair estimate {money(f['repair_estimate'], style=ms)} against {money(f['replacement_cost'], style=ms)} to replace; parts in "
                f"{f['lead_days']} days; refurbished stock {'available' if f['refurb_stock'] else 'not available'}.",
                f"Failed part worth {money(f['part_cost'], style=ms)}; supplier recovery agreement {'in place' if f['supplier_agreement'] else 'none'}.",
                ("Flagged as a quality-audit sample." if f["audit_sample"] else ""), f"Visit in {f['month']}.",
                (f"Owner has a {f['temporary_measure']}." if f["temporary_measure"] else "")]
        return (f"Warranty desk note - {f['tech']}\n\n" + noisy(join_sentences(para), rng, 0.08) + "\n\n" + bullets(self._rules(f), rng))

    def questions(self):
        return {
            "warranty_remedy": choice("What warranty remedy should the service centre carry out?", {
                "repair_unit": "Repair the unit by replacing the failed part.",
                "replace_with_refurbished": "Replace the unit with a refurbished one.",
                "replace_with_new": "Replace the unit with a new one.",
                "escalate_for_technical_review": "Hold the remedy and escalate for technical review of the diagnosis.",
            }),
            "returned_part_handling": choice("What should happen to the failed part?", {
                "scrap_locally": "Scrap it at the service centre.",
                "return_to_supplier": "Return it to the component supplier for cost recovery.",
                "send_to_product_engineering": "Send it to product engineering for failure analysis.",
                "hold_for_quality_audit": "Hold it for the quality audit.",
            }),
            "repair_viability": score("How sound would a repair be as a lasting remedy?", [
                "Poor: repeated failures, slow parts and poor economics.",
                "Doubtful: two strikes against it.",
                "Reasonable: one concern.",
                "Sound: economical, first occurrence, parts on hand.",
            ]),
            "downtime_pressure": score("How pressing is it to restore the unit's function?", [
                "Low: the owner can manage without it for a while.",
                "Moderate: an inconvenience; a wait of some days is acceptable.",
                "High: an essential function is lost; restore within about two days.",
                "Critical: health or property is at risk until it works again.",
            ]),
            "replacement_criteria_met": noul(
                "Under the service policy this unit qualifies for replacement rather than repair.",
                yes="The repair cost ratio, the repeat-repair count or the parts lead time crosses its stated limit.",
                no="None of the replacement triggers is met."),
        }


# =========================================================================== schema 4: service bulletin applicability
# (product, bulletin symptom, configuration condition, measurement name, unit, fix part, software)
BULLETINS = [
    ("compact SUV", "engine stalls at idle after a cold start", "1.5 L turbo engine", "idle air-valve deviation", "%", "throttle body", "ECU calibration"),
    ("split heat pump outdoor unit", "defrost cycle fails, coil ices in low ambient", "cold-climate kit", "defrost sensor offset", "K", "defrost sensor", "control firmware"),
    ("bottom-freezer refrigerator", "ice maker overfills and freezes the auger", "factory ice maker", "fill valve flow", "mL/s", "fill valve", "control firmware"),
    ("90 hp outboard engine", "high-speed misfire above 5,000 rpm", "digital throttle option", "ignition coil secondary resistance", "kOhm", "ignition coils", "ECM software"),
    ("standby generator", "fails to transfer on utility loss", "200 A transfer switch", "transfer relay pick-up time", "ms", "transfer relay", "controller firmware"),
    ("cargo e-bike", "assist cuts out under load on climbs", "dual-battery option", "connector contact resistance", "mOhm", "battery connector harness", "motor controller firmware"),
    ("front-load washer", "door lock error during spin", "steam models", "lock solenoid current", "mA", "door lock assembly", "control firmware"),
]
SYMPTOM_MATCH = ["unrelated symptom", "related symptom, different conditions", "same symptom, partly matching conditions",
                 "same symptom under the stated conditions"]


class ServiceBulletinApplicability(Schema):
    name = "service_bulletin_applicability"
    description = "Whether a technical service bulletin and its coverage extension apply to a unit in for service, from its identification data and diagnosis."
    surfaces = {"json": 0.5, "bulletin_check": 0.25, "technician_query": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng, clear=0.25, borderline=0.45, conflicting=0.30)
        prod, symptom, cfg, meas, unit, fixpart, sw = pick(rng, BULLETINS)
        s_lo = rng.randint(100000, 400000)
        s_hi = s_lo + rng.randint(20000, 90000)
        thr = round(rng.uniform(2, 40), 1)
        ext_y = pick(rng, [5, 7, 8])
        scen = pick_w(rng, {"inside": 5.5, "outside": 2, "edge": 1.5, "repeat": 1, "symptom_off": 1.4})
        serial = rng.randint(s_lo + 500, s_hi - 500)
        cfg_ok = True
        if scen == "outside":
            serial = pick(rng, [s_lo - rng.randint(5000, 60000), s_hi + rng.randint(5000, 60000)])
            cfg_ok = chance(rng, 0.7)
        elif scen == "edge":
            serial = pick(rng, [s_lo - rng.randint(1, 400), s_hi + rng.randint(1, 400)])
        if reg == "conflicting" and chance(rng, 0.35):
            cfg_ok = False
        sym = (pick(rng, [2, 3, 3]) if scen in ("inside", "edge", "repeat", "outside") else pick(rng, [0, 0, 1]))
        if reg == "borderline" and scen == "inside":
            sym = pick(rng, [1, 2, 3])
        done = scen == "repeat"
        mval = round(thr * (near(rng, 1.0, 0.12) if reg == "borderline" else pick(rng, [rng.uniform(0.4, 0.85), rng.uniform(1.15, 1.8)])), 1)
        age_y = round(rng.uniform(0.5, ext_y + 0.5), 1)
        if reg == "borderline" and chance(rng, 0.35):
            age_y = round(near(rng, ext_y, 0.4, lo=0.5), 1)
        id_issue = pick_w(rng, {None: 5, "data plate unreadable; serial read from the controller": 1.5,
                                "controller replaced in 2024; plate and controller serials differ": 1.3,
                                "data plate missing; serial taken from the owner's purchase paperwork": 0.9})
        return {
            "regime": reg, "product": prod, "symptom": symptom, "config": cfg, "measure": meas, "unit": unit, "fix_part": fixpart,
            "software": sw, "serial_lo": s_lo, "serial_hi": s_hi, "threshold": thr, "ext_years": ext_y, "scenario": scen,
            "serial": serial, "config_ok": cfg_ok, "symptom_match": sym, "bulletin_done": done, "measured": mval,
            "age_years": age_y, "id_issue": id_issue, "tsb": f"TSB-{rng.randint(21, 26)}-{rng.randint(100, 999)}",
            "maker": _maker(rng, PRODUCT_KIND[prod]), "dealer": _dealer(rng, PRODUCT_KIND[prod]), "tech": person(rng),
            "ro": ident(rng, "RO", 6, "-"),
        }

    def _inside(self, f):
        return f["serial_lo"] <= f["serial"] <= f["serial_hi"] and f["config_ok"]

    def intended(self, f):
        inside = self._inside(f)
        sym = f["symptom_match"]
        near_edge = min(abs(f["serial"] - f["serial_lo"]), abs(f["serial"] - f["serial_hi"])) <= 500
        if f["bulletin_done"] and sym >= 2:
            act = "investigate_repeat_after_bulletin"
        elif not inside:
            act = "request_technical_assistance" if (near_edge and f["config_ok"] and sym >= 2) or (f["id_issue"] and sym >= 2) else "diagnose_outside_bulletin"
        elif sym <= 1:
            act = "diagnose_outside_bulletin"
        elif f["measured"] > f["threshold"]:
            act = "perform_bulletin_part_replacement"
        else:
            act = "perform_bulletin_software_update"
        ext = inside and sym >= 2 and f["age_years"] <= f["ext_years"] and not f["bulletin_done"]
        idc = (3 if not f["id_issue"] else 1 if "differ" in f["id_issue"] else 0 if "paperwork" in f["id_issue"] else 2)
        if not f["config_ok"] and inside is False and f["serial_lo"] <= f["serial"] <= f["serial_hi"]:
            idc = min(idc, 2)
        return {"bulletin_action": act, "symptom_match": sym, "identification_certainty": idc,
                "inside_affected_population": "yes" if inside else "no", "extension_covers_repair": "yes" if ext else "no"}

    def _rules(self, f, short=False):
        if short:
            return (f"{f['tsb']}: s/n {f['serial_lo']}-{f['serial_hi']} with {f['config']}; symptom: {f['symptom']}. "
                    f"Measure {f['measure']}: > {f['threshold']} {f['unit']} -> replace {f['fix_part']} + update "
                    f"{f['software']}; otherwise update only. Coverage extended to {f['ext_years']} yrs from in-service.")
        return [f"{f['tsb']} applies to s/n {f['serial_lo']} to {f['serial_hi']} built with the {f['config']}, showing: "
                f"{f['symptom']}.",
                f"Procedure: measure {f['measure']}; above {f['threshold']} {f['unit']} replace the {f['fix_part']} and update "
                f"the {f['software']}, otherwise update the {f['software']} only. Repairs under this bulletin are covered for "
                f"{f['ext_years']} years from the in-service date. Units outside the range need a technical-assistance case."]

    def _unit(self, f):
        cfg = f"with {f['config']}" if f["config_ok"] else f"without the {f['config']}"
        return f"s/n {f['serial']}, {cfg}, in service {f['age_years']} years"

    def render(self, f, rng, surface):
        hist = (f"{f['tsb']} completed at a previous visit" if f["bulletin_done"] else "no campaign or bulletin history")
        if surface == "json":
            return dumps(prune({
                "repair_order": {"id": f["ro"], "dealer": f["dealer"], "unit": f"{f['maker']} {f['product']}",
                                 "identification": self._unit(f), "id_note": f["id_issue"], "history": hist},
                "diagnosis": {"symptom_reported": SYMPTOM_MATCH[f["symptom_match"]] + f" ({f['symptom']})" if f["symptom_match"] >= 2
                              else SYMPTOM_MATCH[f["symptom_match"]],
                              f["measure"].replace(" ", "_"): f"{f['measured']} {f['unit']}"},
                "bulletin": self._rules(f),
            }))
        if surface == "bulletin_check":
            rows = [("RO", f"{f['ro']} - {f['dealer']}"), ("Unit", f"{f['maker']} {f['product']}"), ("Identification", self._unit(f)),
                    ("ID note", f["id_issue"]), ("History", hist), ("Symptom vs bulletin", SYMPTOM_MATCH[f["symptom_match"]]),
                    (f"Measured {f['measure']}", f"{f['measured']} {f['unit']}")]
            return "BULLETIN APPLICABILITY CHECK\n" + kv_lines(rows, rng) + "\n\n" + self._rules(f, short=True)
        para = [f"RO {f['ro']}, {f['maker']} {f['product']}, {self._unit(f)}.", (f"Note: {f['id_issue']}." if f["id_issue"] else ""),
                f"History: {hist}.", f"Complaint vs {f['tsb']}: {SYMPTOM_MATCH[f['symptom_match']]}.",
                f"I measured {f['measure']} at {f['measured']} {f['unit']}.", "Does the bulletin apply here?"]
        return (f"Subject: {f['tsb']} applicability - {f['ro']}\nFrom: {f['tech']}, {f['dealer']}\nTo: technical assistance desk\n\n"
                + noisy(join_sentences(para), rng, 0.08) + "\n\n" + bullets(self._rules(f), rng))

    def questions(self):
        return {
            "bulletin_action": choice("What should the technician do about the bulletin on this repair order?", {
                "perform_bulletin_part_replacement": "Carry out the bulletin with the part replacement and software update.",
                "perform_bulletin_software_update": "Carry out the bulletin's software update only.",
                "diagnose_outside_bulletin": "Diagnose the fault normally; the bulletin does not apply.",
                "request_technical_assistance": "Open a technical-assistance case before acting.",
                "investigate_repeat_after_bulletin": "Investigate why the fault returned after the bulletin was already done.",
            }),
            "symptom_match": score("How closely does the reported fault match the bulletin's symptom?", [
                "Not at all.",
                "Loosely: a related fault under different conditions.",
                "Mostly: the same fault, with the conditions partly matching.",
                "Exactly: the same fault under the stated conditions.",
            ]),
            "identification_certainty": score("How certain is the identification of this unit against the bulletin's population?", [
                "Very uncertain.",
                "Doubtful: the identifying data conflict.",
                "Fairly certain: identified by a secondary source.",
                "Certain: plate data clear and consistent.",
            ]),
            "inside_affected_population": noul(
                "The unit falls inside the bulletin's affected population.",
                yes="Its serial number is within the stated range and it has the stated configuration.",
                no="Its serial number is outside the range or it lacks the stated configuration."),
            "extension_covers_repair": noul(
                "The bulletin's coverage extension pays for this repair."),
        }


# =========================================================================== schema 5: goodwill
# (product, component, typical parts cost, typical labour cost, usage unit or None)
GW_PRODUCTS = [
    ("compact SUV", "turbocharger", 1650, 720, "km"), ("compact SUV", "automatic transmission valve body", 1300, 900, "km"),
    ("split heat pump outdoor unit", "compressor", 1500, 900, None), ("bottom-freezer refrigerator", "sealed system", 650, 420, None),
    ("90 hp outboard engine", "powerhead", 3900, 1200, None), ("cargo e-bike", "battery pack", 780, 60, None),
    ("front-load washer", "drive motor", 340, 180, None), ("gas furnace", "control board and inducer", 520, 260, None),
]
MODS = {"compact SUV": ["aftermarket exhaust and intake fitted", "engine control unit re-tuned by an independent shop"],
        "split heat pump outdoor unit": ["third-party universal control board fitted", "refrigerant changed to a non-approved blend"],
        "bottom-freezer refrigerator": ["aftermarket ice-maker kit installed", "start device replaced with a universal kit"],
        "90 hp outboard engine": ["non-approved propeller and jack plate fitted", "thermostats removed"],
        "cargo e-bike": ["non-approved charger used", "aftermarket speed module fitted"],
        "front-load washer": ["third-party control board fitted", "drain pump replaced with a universal part"],
        "gas furnace": ["third-party thermostat wired directly to the control board", "gas valve converted without the approved kit"]}
MAINT = {"complete": "complete, on schedule", "minor_gaps": "complete, one interval overrun by about 15%",
         "major_gaps": "two scheduled services missed", "none": "no records on file; owner says serviced"}
KNOWN = {"bulletin": "matches a published service bulletin for this part", "field_fix": "matches an open field-quality case",
         "similar": "similar complaints logged, no reference issued", "none": "no known issue"}


class GoodwillRepairEligibility(Schema):
    name = "goodwill_repair_eligibility"
    description = "A service network's request for a goodwill contribution to an out-of-warranty repair, under the written goodwill guideline."
    surfaces = {"json": 0.5, "goodwill_request": 0.25, "field_office_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng, clear=0.25, borderline=0.45, conflicting=0.30)
        prod, comp, p0, l0, uunit = pick(rng, GW_PRODUCTS)
        m1, m2 = pick(rng, [6, 12]), pick(rng, [18, 24])
        mki = m2 + pick(rng, [6, 12, 24])
        lab = pick(rng, [50, 75])
        a1, a2 = pick(rng, [500, 750]), pick(rng, [1500, 2500])
        scen = pick_w(rng, {"band_a": 2.4, "band_b": 3.2, "known": 2, "beyond": 4.0, "records": 2.1, "disqualified": 0.6})
        months = {"band_a": rng.randint(1, m1), "band_b": rng.randint(m1 + 1, m2), "known": rng.randint(m2 + 1, mki),
                  "beyond": rng.randint(m2 + 1, m2 + 30), "records": rng.randint(1, m2), "disqualified": rng.randint(1, m2)}[scen]
        if reg == "borderline":
            months = {"band_a": m1 - pick(rng, [0, 1, 2]), "band_b": pick(rng, [m1 + 1, m1 + 2, m2 - 1, m2]),
                      "known": mki - pick(rng, [-2, -1, 0, 1]), "beyond": m2 + pick(rng, [1, 1, 2, 3])}.get(scen, months)
        known = (pick(rng, ["bulletin", "field_fix"]) if scen == "known"
                 else pick_w(rng, {"none": 4, "similar": 2, "field_fix": 1, "bulletin": 0.6}))
        if scen == "beyond" and known in ("bulletin", "field_fix"):
            known = "similar"
        maint = "none" if scen == "records" else pick_w(rng, {"complete": 5, "minor_gaps": 2.5, "major_gaps": 0.8})
        prior_gw = scen == "disqualified" and chance(rng, 0.5)
        modified = scen == "disqualified" and not prior_gw
        if reg == "conflicting":
            twist = pick_w(rng, {"known_but_gaps": 1.2, "near_but_modified": 0.7, "similar_only": 1})
            if twist == "known_but_gaps":
                known, maint = "bulletin", "major_gaps"
            elif twist == "near_but_modified":
                months, modified = rng.randint(1, m1), True
            else:
                known, months = "similar", rng.randint(m2 + 1, mki)
        km_lim = None
        km_past = None
        if uunit:
            km_lim = (pick(rng, [15000, 20000]), pick(rng, [30000, 40000]))
            km_past = int(rng.uniform(0.2, 1.15) * (km_lim[0] if months <= m1 else km_lim[1]))
        parts = round(p0 * rng.uniform(0.8, 1.3), -1)
        labour = round(l0 * rng.uniform(0.8, 1.3), -1)
        return {
            "regime": reg, "product": prod, "component": comp, "m1": m1, "m2": m2, "m_known": mki, "labour_pct": lab,
            "auth_centre": a1, "auth_region": a2, "months_past": max(1, months), "km_limits": km_lim, "km_past": km_past,
            "known_issue": known, "maintenance": maint, "prior_goodwill": prior_gw, "modified": modified,
            "parts": parts, "labour": labour, "dealer": _dealer(rng, PRODUCT_KIND[prod]), "manager": person(rng),
            "ref": ident(rng, "GW", 5, "-"), "gg": f"GG-{rng.randint(2, 9)}", "maker": _maker(rng, PRODUCT_KIND[prod]),
            "mod_text": pick(rng, MODS[prod]),
        }

    def _band(self, f):
        m = f["months_past"]
        kl = f["km_limits"]
        km = f["km_past"]
        in_a = m <= f["m1"] and (kl is None or km <= kl[0])
        in_b = m <= f["m2"] and (kl is None or km <= kl[1])
        if f["known_issue"] in ("bulletin", "field_fix") and m <= f["m_known"]:
            return "known"
        return "a" if in_a else "b" if in_b else None

    def intended(self, f):
        band = self._band(f)
        if f["modified"] or f["prior_goodwill"] or band is None:
            contr = "no_contribution"
        elif f["maintenance"] == "none":
            contr = "request_service_records_first"
        else:
            if f["maintenance"] == "major_gaps":
                band = {"known": "a", "a": "b", "b": None}.get(band)
            contr = {"known": "full_parts_and_labour", "a": "full_parts_partial_labour", "b": "half_parts_only",
                     None: "no_contribution"}[band]
        pay_band = band if contr != "request_service_records_first" else self._band(f)
        amt = {"known": f["parts"] + f["labour"], "a": f["parts"] + f["labour"] * f["labour_pct"] / 100,
               "b": f["parts"] * 0.5, None: 0}[pay_band]
        if contr == "no_contribution" or amt == 0:
            level = "no_approval_needed"
        elif amt <= f["auth_centre"]:
            level = "service_centre_manager"
        elif amt <= f["auth_region"]:
            level = "regional_technical_manager"
        else:
            level = "warranty_review_board"
        mc = {"complete": 3, "minor_gaps": 2, "major_gaps": 1, "none": 0}[f["maintenance"]]
        kn = {"bulletin": 3, "field_fix": 2, "similar": 1, "none": 0}[f["known_issue"]]
        window = self._band(f) is not None
        return {"goodwill_contribution": contr, "approval_level": level, "maintenance_compliance": mc,
                "known_issue_link": kn, "inside_goodwill_window": "yes" if window else "no"}

    def _rules(self, f):
        kl = f["km_limits"]
        ka = f" / {kl[0]:,} km" if kl else ""
        kb = f" / {kl[1]:,} km" if kl else ""
        return [f"{f['gg']} §3: up to {f['m1']} months{ka} past expiry: 100% parts and {f['labour_pct']}% labour; up to "
                f"{f['m2']} months{kb}: 50% parts, no labour; beyond that nothing, except a failure documented in a bulletin "
                f"or open field-quality case: 100% parts and labour up to {f['m_known']} months. Maintenance gaps of a missed "
                "service or more drop the case one band; no goodwill on modified units or twice for the same component.",
                f"§5: contributions up to {money(f['auth_centre'], style='sym')} service-centre manager; up to "
                f"{money(f['auth_region'], style='sym')} regional technical manager; above that, the warranty review board."]

    def _past(self, f):
        return f"{f['months_past']} months" + (f", {f['km_past']:,} km" if f["km_past"] is not None else "") + " past expiry"

    def render(self, f, rng, surface):
        ms = pick(rng, ["sym", "sym", "code"])
        hist = "goodwill already paid on this component once" if f["prior_goodwill"] else None
        mod = f["mod_text"] if f["modified"] else None
        if surface == "json":
            return dumps(prune({
                "request": {"ref": f["ref"], "dealer": f["dealer"], "unit": f"{f['maker']} {f['product']}",
                            "failed_component": f["component"], "warranty_status": self._past(f)},
                "repair": {"parts": money(f["parts"], style=ms), "labour": money(f["labour"], style=ms)},
                "unit_history": {"maintenance": MAINT[f["maintenance"]], "known_issue": KNOWN[f["known_issue"]],
                                 "modifications": mod, "goodwill_history": hist},
                "guideline": self._rules(f),
            }))
        if surface == "goodwill_request":
            rows = [("Ref", f["ref"]), ("Dealer", f"{f['dealer']} ({f['manager']})"), ("Unit", f"{f['maker']} {f['product']}"),
                    ("Component", f["component"]), ("Status", self._past(f)), ("Parts / labour", f"{money(f['parts'], style=ms)} / {money(f['labour'], style=ms)}"),
                    ("Maintenance", MAINT[f["maintenance"]]), ("Known issue", KNOWN[f["known_issue"]]), ("Modifications", mod or "none seen"),
                    ("Earlier goodwill", hist or "none")]
            return "GOODWILL REQUEST - SERVICE NETWORK\n" + kv_lines(rows, rng) + "\n\n" + self._rules(f)[0]
        para = [f"{f['dealer']} asks for goodwill on the {f['component']} of a {f['maker']} {f['product']}, {self._past(f)}.",
                f"Repair is {money(f['parts'], style=ms)} parts and {money(f['labour'], style=ms)} labour.",
                f"Service history: {MAINT[f['maintenance']]}.", f"Known-issue check: {KNOWN[f['known_issue']]}.",
                (f"Inspection noted: {mod}." if mod else ""), (f"Note: {hist}." if hist else "")]
        return (f"Field office note - {f['ref']}\n\n" + noisy(join_sentences(para), rng, 0.08) + "\n\n" + bullets(self._rules(f), rng))

    def questions(self):
        return {
            "goodwill_contribution": choice("What goodwill contribution does the guideline support?", {
                "full_parts_and_labour": "All parts and labour.",
                "full_parts_partial_labour": "All parts and part of the labour.",
                "half_parts_only": "Half the parts cost, no labour.",
                "no_contribution": "Nothing.",
                "request_service_records_first": "No decision until service records are supplied.",
            }),
            "approval_level": choice("Who has to approve the contribution the guideline supports?", {
                "service_centre_manager": "The service-centre manager, within local authority.",
                "regional_technical_manager": "The regional technical manager.",
                "warranty_review_board": "The warranty review board.",
                "no_approval_needed": "No one: nothing is payable.",
            }),
            "maintenance_compliance": score("How well has the unit been maintained according to its records?", [
                "Unknown: no records to judge from.",
                "Poor: scheduled services missed.",
                "Mostly kept: a small overrun.",
                "Fully kept: every service on schedule.",
            ]),
            "known_issue_link": score("How strongly does the failure match a documented product weakness?", [
                "Not at all.",
                "Weakly: similar complaints, nothing documented.",
                "Substantially: an open field-quality case covers it.",
                "Directly: a published bulletin covers it.",
            ]),
            "inside_goodwill_window": noul(
                "The unit is within the guideline's goodwill window after warranty expiry.",
                yes="Its time (and distance, where it applies) past expiry falls within a band, or within the known-issue window.",
                no="It is past every band and no documented known issue extends the window."),
        }


WORKFLOW = Workflow(
    name="repair_and_warranty_adjudication",
    title="Repair and warranty adjudication",
    description="A manufacturer's service organisation ruling on defect claims from its service network: coverage, teardown causation, repair or replace, service-bulletin applicability and goodwill, under written (fictional) warranty and service terms.",
    domain="repair and maintenance / product service",
    schemas=[FailureCoverageDetermination(), TeardownMisuseDetermination(), RepairVsReplace(),
             ServiceBulletinApplicability(), GoodwillRepairEligibility()],
    notes="Constrained family (warranty / returns): product-defect adjudication from the service side only. States are "
          "technical artefacts from the service network or the warranty lab; no customer conversation, no reply or churn "
          "questions, no loyalty factor, and no reconciliation of claim amounts against a bill.",
)
