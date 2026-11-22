"""Pipeline integrity dig decisions: in-line-inspection anomalies judged against a (fictional) operator's written criteria.

Integrity-engineering decisions on invented gas-transmission and hazardous-liquid pipelines: how to respond to a
metal-loss call from an in-line inspection (ILI) tool, to a dent from a geometry tool, to a cluster of corrosion
features that may interact, which repair to install once the pipe is exposed, and whether a scheduled excavation may
slip past its deadline. Operators, lines, tool vendors and every criterion are invented, shaped like a real integrity
management plan; numbers follow real engineering relations (modified-B31G failure pressure, hoop stress against
SMYS, the potential impact radius of a gas line) so a careful engineer can check them. Thresholds differ between
operators, so the criteria quoted in the state have to be read. Each schema draws a `regime` first so most cases
sit near a criterion, carry signals that disagree (a vendor's call against verification-dig bias), or miss a fact.

Every schema also has `intended(f)`: the answer a careful reviewer would most likely give from what the state
shows. It is never rendered and never used as a label; it exists so that the answer spread of every question can
be audited before labelling.
"""
from __future__ import annotations

import math
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

# --------------------------------------------------------------------------- shared pools and engineering helpers
OP_A = ["Tamarack", "Sandhill", "Coldwater", "Prairie Rose", "Ironhorse", "Larkspur", "Buffalo Grass", "Silver Sage",
        "Red Willow", "Kettle River", "Greyhawk", "Bluestem", "Juniper Flats", "Cutbank"]
OP_B = ["Gas Transmission", "Pipeline Co.", "Midstream", "Interstate Gas", "Pipe Line Partners"]
LIQ_B = ["Products Pipeline", "Crude Line Co.", "Liquids Pipeline"]
VENDORS = ["Tracerline", "PigSense", "Corvane Inspection", "Halden NDT", "Northpoint ILI", "Vantis Pipeline Services"]
PIPES = [(12.75, [0.250, 0.281, 0.330]), (16.0, [0.250, 0.281, 0.312]), (20.0, [0.281, 0.312, 0.375]),
         (24.0, [0.312, 0.375, 0.438]), (30.0, [0.375, 0.438, 0.500]), (36.0, [0.438, 0.500])]
SMYS = {"Grade B": 35000, "X42": 42000, "X52": 52000, "X60": 60000, "X65": 65000, "X70": 70000}
DESIGN_FACTOR = {1: 0.72, 2: 0.60, 3: 0.50, 4: 0.40}
PLACES = ["Coyote Draw", "Hinsdale", "Pawnee Flats", "Burnt Fork", "Lodgepole", "Antler Creek", "Sweetwater Bend",
          "Mill Iron", "Cedar Gap", "Box Elder", "Dutch Flats", "Sand Wash", "Ten Mile", "Bitter Creek", "Rock Ford"]
STRUCT_TYPES = ["homes", "a school", "a farmstead", "a gas station", "a church", "an apartment block", "a county road"]


def _operator(rng, liquid=False):
    return f"{pick(rng, OP_A)} {pick(rng, LIQ_B if liquid else OP_B)}"


def pf_mod_b31g(D, t, smys, depth_frac, L):
    """Predicted failure pressure (psig) of a metal-loss feature by the modified-B31G (0.85dL) method."""
    sflow = smys + 10000.0
    z = L * L / (D * t)
    M = math.sqrt(1 + 0.6275 * z - 0.003375 * z * z) if z <= 50 else 0.032 * z + 3.3
    dt = min(depth_frac, 0.95)
    s = sflow * (1 - 0.85 * dt) / (1 - 0.85 * dt / M)
    return 2 * s * t / D


def pir_ft(D, p):
    """Potential impact radius of a natural-gas line, ft (0.69 x d x sqrt(p))."""
    return 0.69 * D * math.sqrt(p)


def _pipe(rng, liquid=False):
    D, ts = pick(rng, PIPES)
    t = pick(rng, ts)
    year = pick_w(rng, {rng.randint(1952, 1969): 3, rng.randint(1970, 1989): 3, rng.randint(1990, 2016): 2})
    if year < 1970:
        grade = pick(rng, ["Grade B", "X42", "X52"])
        seam = pick_w(rng, {"LF-ERW": 3, "DSAW": 2, "seamless": 1, "flash-welded": 1})
        coating = pick(rng, ["coal tar enamel", "asphalt enamel", "polyethylene tape"])
    elif year < 1990:
        grade = pick(rng, ["X52", "X60", "X65"])
        seam = pick_w(rng, {"HF-ERW": 3, "DSAW": 2})
        coating = pick(rng, ["polyethylene tape", "coal tar enamel", "FBE"])
    else:
        grade = pick(rng, ["X60", "X65", "X70"])
        seam = pick_w(rng, {"HF-ERW": 2, "DSAW": 2})
        coating = pick(rng, ["FBE", "FBE", "three-layer polyethylene"])
    cls = pick_w(rng, {1: 5, 2: 3, 3: 2, 4: 0.3})
    smys = SMYS[grade]
    maop = int(round(DESIGN_FACTOR[cls] * 2 * smys * t / D * pick(rng, [1.0, 1.0, 0.95, 0.9, 0.85]), -1))
    if liquid:
        maop = int(round(0.72 * 2 * smys * t / D * pick(rng, [0.85, 0.9, 1.0]), -1))
    return {"D": D, "t": t, "year": year, "grade": grade, "smys": smys, "seam": seam, "coating": coating,
            "class": cls, "maop": maop}


def _line_name(rng):
    return pick(rng, [f"Line {rng.randint(100, 499)}", f"L-{rng.randint(1, 40)}{pick(rng, ['', 'A', 'B'])}",
                      f"{pick(rng, PLACES)} Lateral", f"Mainline {pick(rng, ['A', 'B', 'C'])}"])


def _length_for_fpr(D, t, smys, depth_frac, maop, target):
    """The feature length (in) whose modified-B31G failure-pressure ratio is closest to `target`."""
    best, bl = None, 2.0
    for L in [0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 3.6, 4.2, 5.0, 6.0, 7.5, 9.0, 11.0, 13.0, 16.0, 20.0, 25.0, 32.0, 40.0]:
        r = pf_mod_b31g(D, t, smys, depth_frac, L) / maop
        if best is None or abs(r - target) < best:
            best, bl = abs(r - target), L
    return bl


def _clock(rng, top=None):
    """An o'clock position string; top=True draws between 8:00 and 4:00 through 12."""
    if top is None:
        h = rng.randint(1, 12)
    elif top:
        h = pick(rng, [8, 9, 10, 11, 12, 1, 2, 3])
    else:
        h = pick(rng, [4, 5, 6, 7])
    return f"{h}:{pick(rng, ['00', '15', '30', '45'])}"


def _is_top(clock):
    h, m = clock.split(":")
    h = int(h) % 12
    minutes = h * 60 + int(m)
    return minutes >= 8 * 60 or minutes <= 4 * 60


DATE_STYLE = ["iso", "iso", "us", "long", "eu"]


# =========================================================================== schema 1
class MetalLossAnomalyResponse(Schema):
    name = "metal_loss_anomaly_response"
    description = "A corrosion (metal-loss) anomaly from an in-line inspection, to be placed in the integrity plan's response categories."
    surfaces = {"json": 0.5, "ili_feature_report": 0.25, "engineer_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        p = _pipe(rng)
        sched_depth, sched_fpr = pick(rng, [40, 50, 60]), pick(rng, [1.25, 1.39, 1.39])
        sched_days, next_ili = pick(rng, [180, 365]), pick(rng, [5, 7, 7, 10])
        tol = pick(rng, [8, 10, 10, 12])
        bias = pick_w(rng, {0: 5, rng.randint(3, 9): 2, -rng.randint(2, 6): 1})
        ext = chance(rng, 0.8)
        depth = rng.randint(18, 72)
        target = None
        if reg == "clear":
            if chance(rng, 0.5):
                depth, bias, target = rng.randint(12, 26), 0, sched_fpr + rng.uniform(0.2, 0.4)
            else:
                depth, target = rng.randint(76, 88), 1.05
        elif reg == "borderline":
            edge = pick(rng, ["depth_sched", "depth_imm", "fpr"])
            if edge == "depth_sched":
                depth = int(near(rng, sched_depth, 5, lo=10, hi=79, integer=True))
            elif edge == "depth_imm":
                depth = int(near(rng, 78, 4, lo=60, hi=85, integer=True))
            else:
                target = sched_fpr + rng.uniform(-0.07, 0.07)
        else:
            # a shallow-looking call that verification bias, tolerance or fast growth pushes over a line
            depth = int(near(rng, sched_depth - 6, 4, lo=10, hi=75, integer=True))
            bias = pick(rng, [rng.randint(5, 10), 0])
        if target is None:
            target = pick_w(rng, {rng.uniform(1.0, 1.1): 0.6, rng.uniform(1.1, sched_fpr): 1, rng.uniform(sched_fpr, 2.0): 3})
        L = _length_for_fpr(p["D"], p["t"], p["smys"], depth / 100, p["maop"], target)
        L = round(L * rng.uniform(0.9, 1.1), 1)
        pf = pf_mod_b31g(p["D"], p["t"], p["smys"], depth / 100, L)
        years_between = pick(rng, [5, 6, 7, 8])
        prior = maybe(rng, max(0, depth - int(round(rng.uniform(0.4, 4.5) * years_between))), 0.3)
        if reg == "conflicting" and prior is not None and chance(rng, 0.5):
            prior = max(0, depth - int(round(rng.uniform(2.8, 5.0) * years_between)))
        cp = round(pick_w(rng, {rng.uniform(-1.25, -0.88): 5, rng.uniform(-0.87, -0.70): 2}), 2)
        disc = rand_date(rng, date(2025, 1, 6), date(2027, 3, 1))
        return {"regime": reg, "operator": _operator(rng), "line": _line_name(rng), **p,
                "sched_depth": sched_depth, "sched_fpr": sched_fpr, "sched_days": sched_days, "next_ili_years": next_ili,
                "tool_tolerance": tol, "verified_bias": bias, "n_verification_digs": rng.randint(4, 22),
                "external": ext, "depth_pct": depth, "length_in": L, "width_in": round(rng.uniform(0.6, 9.0), 1),
                "clock": _clock(rng), "pf": int(round(pf, -1)), "prior_depth_pct": prior, "years_between_runs": years_between,
                "cp_volts": cp, "hca": chance(rng, 0.4), "odometer_ft": rng.randint(1200, 290000),
                "gw": rng.randint(100, 9900) * 10, "feature_id": rng.randint(1000, 99999), "vendor": pick(rng, VENDORS),
                "discovery": disc, "engineer": person(rng)}

    def _rate(self, f):
        if f["prior_depth_pct"] is None:
            return None
        prior = max(f["prior_depth_pct"], 10)       # below the reporting threshold reads as "about 10% or less"
        return max(0.0, (f["depth_pct"] - prior) / f["years_between_runs"])

    def intended(self, f):
        d_adj = f["depth_pct"] + max(0, f["verified_bias"])
        fpr = f["pf"] / f["maop"]
        imm = d_adj >= 80 or fpr < 1.10
        sched = d_adj >= f["sched_depth"] or fpr < f["sched_fpr"]
        rate = self._rate(f)
        rate_used = rate if rate is not None else 1.5           # an assumed rate when no prior match exists
        yrs80 = (80 - d_adj) / rate_used if rate_used > 0 else 99
        if imm:
            resp = "immediate_excavation"
        elif sched or yrs80 < f["next_ili_years"]:
            resp = "scheduled_excavation"
        elif fpr < f["sched_fpr"] + 0.06 or d_adj >= f["sched_depth"] - 4:
            resp = "detailed_assessment_first"
        else:
            resp = "monitor_next_inspection"
        burst = 3 if fpr < 1.10 else 2 if fpr < f["sched_fpr"] else 1 if fpr < f["sched_fpr"] + 0.15 else 0
        growth = 3 if yrs80 <= 1 else 2 if yrs80 <= 3 else 1 if yrs80 <= f["next_ili_years"] + 1 else 0
        if d_adj >= 80:
            growth = 3
        reported = f["depth_pct"]
        bump = max(f["tool_tolerance"], f["verified_bias"])
        crosses = (reported < f["sched_depth"] <= reported + bump) or (reported < 80 <= reported + bump)
        return {"response_category": resp, "burst_margin_concern": burst, "growth_time_pressure": growth,
                "tolerance_changes_category": "yes" if crosses else "no",
                "pressure_reduction_now": "yes" if imm else "no"}

    def _criteria(self, f):
        return (f"IMP 7.3 response criteria. Immediate (excavate now; reduce pressure to 80% of the operating pressure at "
                f"discovery until repaired): depth 80% of wall or more, or predicted failure pressure below 1.10 x MAOP. "
                f"Scheduled (excavate within {f['sched_days']} days of discovery): depth {f['sched_depth']}% or more, failure "
                f"pressure below {f['sched_fpr']:.2f} x MAOP, or projected to reach an immediate criterion before the next "
                f"assessment ({f['next_ili_years']} years). Otherwise monitor. Apply any depth bias found by verification digs.")

    def _growth_text(self, f):
        if f["prior_depth_pct"] is None:
            return f"not matched to the previous run ({f['years_between_runs']} years ago)"
        if f["prior_depth_pct"] < 10:
            return f"below the 10% reporting threshold in the run {f['years_between_runs']} years ago"
        return f"{f['prior_depth_pct']}% in the run {f['years_between_runs']} years ago"

    def _bias_text(self, f):
        b = f["verified_bias"]
        if b > 0:
            return f"tool under-called depth by {b}% of wall on average over {f['n_verification_digs']} verification digs"
        if b < 0:
            return f"tool over-called depth by {-b}% of wall on average over {f['n_verification_digs']} verification digs"
        return f"tool within stated tolerance over {f['n_verification_digs']} verification digs"

    def render(self, f, rng, surface):
        st = pick(rng, DATE_STYLE)
        kind = "external" if f["external"] else "internal"
        if surface == "json":
            return dumps(prune({
                "anomaly": {"feature_id": f["feature_id"], "type": f"{kind} metal loss", "depth_pct_wall": f["depth_pct"],
                            "length_in": f["length_in"], "width_in": f["width_in"], "clock": f["clock"],
                            "odometer_ft": f["odometer_ft"], "upstream_girth_weld": f["gw"],
                            "predicted_failure_pressure_psig_mod_b31g": f["pf"], "previous_run_depth": self._growth_text(f)},
                "pipe": {"line": f["line"], "operator": f["operator"], "od_in": f["D"], "wall_in": f["t"], "grade": f["grade"],
                         "seam": f["seam"], "installed": f["year"], "coating": f["coating"], "maop_psig": f["maop"],
                         "class_location": f["class"], "in_hca": f["hca"]},
                "tool": {"vendor": f["vendor"], "depth_tolerance": f"+/-{f['tool_tolerance']}% of wall at 80% certainty",
                         "field_verification": self._bias_text(f)},
                "corrosion_control": {"nearest_test_station_potential_v_cse": f["cp_volts"]},
                "discovered": fmt_date(f["discovery"], style="iso"),
                "criteria": self._criteria(f),
            }))
        if surface == "ili_feature_report":
            hdr = (f"{f['vendor']} - HIGH-RESOLUTION MFL FEATURE LISTING (extract) - {f['operator']} {f['line']}\n"
                   f"Pipe: {f['D']} in OD x {f['t']} in WT, {f['grade']}, {f['seam']}, {f['year']}, {f['coating']}; "
                   f"MAOP {f['maop']} psig; class {f['class']}; {'HCA' if f['hca'] else 'non-HCA'}")
            tbl = table(["feature", "odometer ft", "GW", "clock", "type", "depth %", "L in", "W in", "Pf psig"],
                        [[f["feature_id"], f"{f['odometer_ft']:,}", f["gw"], f["clock"], f"{kind} ML", f["depth_pct"],
                          f["length_in"], f["width_in"], f["pf"]]], rng)
            notes = bullets([f"Depth sizing tolerance +/-{f['tool_tolerance']}% WT (80% certainty)",
                             f"Previous run: {self._growth_text(f)}", f"Verification: {self._bias_text(f)}",
                             f"CP reading at nearest test station: {f['cp_volts']} V (CSE)",
                             f"Data received {fmt_date(f['discovery'], style=st)}"], rng)
            return f"{hdr}\n\n{tbl}\n\n{notes}\n\n{self._criteria(f)}"
        body = [f"Feature {f['feature_id']} on {f['line']} ({f['D']} x {f['t']} {f['grade']} {f['seam']}, {f['year']}, MAOP {f['maop']} psig, "
                f"class {f['class']}{', HCA' if f['hca'] else ''}): {kind} metal loss called at {f['depth_pct']}% wall, "
                f"{f['length_in']} in long, {f['width_in']} in wide at {f['clock']}.",
                f"Vendor's modified-B31G failure pressure is {f['pf']} psig.",
                f"Previous run: {self._growth_text(f)}.", f"Verification digs: {self._bias_text(f)}.",
                f"CP at the nearest test station reads {f['cp_volts']} V.",
                pick(rng, ["Operations would like to avoid a dig this season if we can.", "Landowner access is straightforward here.",
                           "This is in a hay field, access only after harvest.", ""])]
        return (f"Subject: {f['line']} ILI - feature {f['feature_id']} response call\nFrom: {f['engineer']}\n"
                f"Date: {fmt_date(f['discovery'] + timedelta(days=rng.randint(3, 20)), style=st)}\n\n"
                + noisy(join_sentences(body), rng, 0.1) + "\n\n" + self._criteria(f))

    def questions(self):
        return {
            "response_category": choice("How should the integrity plan respond to this anomaly?", {
                "immediate_excavation": "Excavate now, with an interim pressure reduction until it is repaired.",
                "scheduled_excavation": "Schedule an excavation within the plan's deadline for scheduled conditions.",
                "detailed_assessment_first": "Run a more detailed remaining-strength assessment of the tool's profile before committing to a dig.",
                "monitor_next_inspection": "Record it and reassess it with the next in-line inspection.",
            }),
            "burst_margin_concern": score("How thin is the anomaly's margin against burst at maximum allowable operating pressure?", [
                "Ample: the failure pressure sits well above the scheduled-response line.",
                "Modest: above the scheduled-response line, but not by much.",
                "Thin: inside the scheduled-response band.",
                "Critical: at or below the immediate-response line.",
            ]),
            "growth_time_pressure": score("How soon could corrosion growth carry this anomaly to the immediate-response criteria?", [
                "Not before the next scheduled inspection, with room to spare.",
                "Around the time of the next scheduled inspection.",
                "Within the next two or three years.",
                "Within a year, or it may already be there.",
            ]),
            "tolerance_changes_category": noul(
                "Allowing for the tool's sizing uncertainty, this anomaly could belong in a more severe response category than its reported depth suggests.",
                yes="The reported depth plus the stated tolerance, or the bias found in verification digs, crosses the next criterion line.",
                no="Even at the top of the tolerance band the anomaly stays in the category its reported depth gives."),
            "pressure_reduction_now": noul(
                "The segment needs an interim pressure reduction until this anomaly is dealt with."),
        }


# =========================================================================== schema 2
CONTEXT = {
    None: None,
    "road_crossing": "within 40 ft of a county road crossing",
    "drain_tile": "in a field where drainage tile was installed last autumn",
    "fibre_hdd": "near a fibre-optic directional bore completed two years ago",
    "rocky_ground": "in rocky ground with shallow bedrock",
    "creek_slope": "on a creek bank slope with past soil movement",
}


class DentAssessment(Schema):
    name = "dent_assessment"
    description = "A dent reported by a geometry (caliper) tool, judged against the plan's dent criteria and the line's service."
    surfaces = {"json": 0.5, "caliper_listing": 0.25, "integrity_review": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        liquid = chance(rng, 0.4)
        p = _pipe(rng, liquid)
        depth = round(pick_w(rng, {rng.uniform(0.8, 2.0): 3, rng.uniform(2.0, 6.0): 5, rng.uniform(6.0, 9.5): 1}), 1)
        top = chance(rng, 0.5)
        clock = _clock(rng, top)
        gw_dist = pick_w(rng, {round(rng.uniform(0.0, 3.0), 1): 2, round(rng.uniform(3.0, 120.0), 1): 5})
        seam_clock = _clock(rng) if p["seam"] in ("LF-ERW", "HF-ERW", "DSAW", "flash-welded") else None
        on_seam = seam_clock is not None and chance(rng, 0.12)
        coincident = pick_w(rng, {"none": 8, "low_confidence": 2, "metal_loss": 1})
        ml_depth = rng.randint(8, 24) if coincident != "none" else None
        new_since_last = pick_w(rng, {True: 3, False: 3, None: 2})
        context = pick_w(rng, {None: 3, "road_crossing": 1, "drain_tile": 1, "fibre_hdd": 1, "rocky_ground": 2, "creek_slope": 2})
        cycles = (rng.randint(150, 2200) if liquid else rng.randint(2, 40))
        length = round(pick(rng, [rng.uniform(2, 8), rng.uniform(8, 30)]), 1)
        if reg == "clear":
            if chance(rng, 0.5):
                depth, top, coincident, gw_dist, on_seam = round(rng.uniform(0.8, 1.8), 1), False, "none", round(rng.uniform(20, 120), 1), False
                length = round(rng.uniform(8, 30), 1)
                clock, new_since_last, context = _clock(rng, False), False, pick(rng, ["rocky_ground", None])
            else:
                top, coincident, depth = True, "metal_loss", round(rng.uniform(3.0, 8.0), 1)
                clock, new_since_last, context = _clock(rng, True), True, pick(rng, ["road_crossing", "drain_tile", "fibre_hdd"])
        elif reg == "borderline":
            depth = round(near(rng, pick(rng, [2.0, 6.0]), 0.5, lo=0.6), 1)
        else:
            # a clean signal on paper with a worrying context, a deep old dent that has never changed, or a modest
            # dent on a hard-cycling liquid line
            branch = pick(rng, ["context", "old_deep", "cycling"])
            if branch == "cycling":
                liquid = True
                p = _pipe(rng, True)
                seam_clock = _clock(rng) if p["seam"] != "seamless" else None
                cycles = rng.randint(700, 2200)
                depth, coincident, gw_dist, on_seam = round(rng.uniform(1.6, 4.0), 1), "none", round(rng.uniform(12, 120), 1), False
                length = round(rng.uniform(8, 30), 1)
            elif branch == "context":
                coincident, top, new_since_last = "low_confidence", True, True
                clock, context = _clock(rng, True), pick(rng, ["road_crossing", "drain_tile", "fibre_hdd"])
            else:
                depth, top, new_since_last, context = round(rng.uniform(5.5, 8.5), 1), False, False, "rocky_ground"
                clock = _clock(rng, False)
        if coincident != "none" and ml_depth is None:
            ml_depth = rng.randint(8, 24)
        # dent strain grows with depth and sharpness (short dents); some runs cannot resolve it
        strain_limit = pick(rng, [6.0, 6.0, 4.0])
        strain = round(max(0.3, 0.85 * depth * math.sqrt(10.0 / max(1.5, length)) * rng.uniform(0.7, 1.4)), 1)
        if reg == "borderline" and chance(rng, 0.4):
            strain = round(near(rng, strain_limit, 0.6, lo=0.5), 1)
        strain = maybe(rng, strain, 0.2)
        if on_seam:
            seam_clock = clock
        return {"regime": reg, "operator": _operator(rng, liquid), "liquid": liquid, "line": _line_name(rng), **p,
                "product": pick(rng, ["refined products", "crude oil", "NGL"]) if liquid else "natural gas",
                "dent_depth_pct_od": depth, "clock": clock, "length_in": length, "gw_dist_in": gw_dist,
                "seam_clock": seam_clock, "on_seam": on_seam, "coincident": coincident, "ml_depth": ml_depth,
                "strain_pct": strain, "new_since_last": new_since_last, "context": context, "cycles_per_year": cycles,
                "top_limit": pick(rng, [6.0, 6.0, 4.0]),
                "weld_zone_in": pick(rng, [1.0, 2.0, 3.0]), "strain_limit": strain_limit,
                "vendor": pick(rng, VENDORS), "feature_id": rng.randint(1000, 99999), "odometer_ft": rng.randint(1200, 290000),
                "run_date": rand_date(rng, date(2025, 1, 6), date(2027, 3, 1)), "engineer": person(rng)}

    def intended(self, f):
        top = _is_top(f["clock"])
        weld = f["gw_dist_in"] <= f["weld_zone_in"] or f["on_seam"]
        imm = f["coincident"] == "metal_loss" or (top and f["dent_depth_pct_od"] > f["top_limit"])
        sched = (f["coincident"] == "low_confidence" or (f["dent_depth_pct_od"] > 2 and weld)
                 or f["dent_depth_pct_od"] > 6 or (f["strain_pct"] or 0) > f["strain_limit"])
        fat_hi = f["liquid"] and f["cycles_per_year"] >= 600
        if imm:
            resp = "immediate_excavation"
        elif sched:
            resp = "scheduled_excavation"
        elif (f["liquid"] and f["cycles_per_year"] >= 400 and (f["dent_depth_pct_od"] > 1.5 or top)) or (
                f["strain_pct"] is None and f["dent_depth_pct_od"] > 2.5):
            resp = "engineering_assessment"
        else:
            resp = "monitor_next_inspection"
        sev = 3 if imm else 2 if sched else 1 if (f["dent_depth_pct_od"] > 2 or resp == "engineering_assessment") else 0
        fat = 0
        if f["liquid"]:
            fat = 1 + (f["cycles_per_year"] >= 600) + (f["cycles_per_year"] >= 1400 and f["dent_depth_pct_od"] > 2)
            if f["context"] == "rocky_ground" and not top:
                fat = max(1, fat - 1)       # a restrained bottom-side dent rerounds less under cycling
        elif f["dent_depth_pct_od"] > 4 and (weld or f["coincident"] != "none"):
            fat = 1
        tp_ctx = f["context"] in ("road_crossing", "drain_tile", "fibre_hdd")
        if f["context"] == "creek_slope" and f["length_in"] > 8:
            origin = "soil_movement_or_bending"
        elif f["new_since_last"] and (top or tp_ctx):
            origin = "third_party_excavation"
        elif f["new_since_last"] and f["length_in"] > 12:
            origin = "soil_movement_or_bending"
        elif f["new_since_last"] is False or (not top and f["context"] == "rocky_ground"):
            origin = "rock_or_construction_damage"
        elif tp_ctx and top:
            origin = "third_party_excavation"
        else:
            origin = "undetermined_needs_excavation"
        return {"dent_response": resp, "dent_origin": origin, "dent_severity": sev, "fatigue_concern": min(3, fat),
                "strain_exceeds_limit": "yes" if (f["strain_pct"] or 0) > f["strain_limit"] else "no"}

    def _criteria(self, f):
        return (f"IMP 7.4 dents. Immediate: any dent with an indication of metal loss, cracking or gouging; a top-side dent "
                f"(8:00 to 4:00 through 12:00) deeper than {f['top_limit']}% of OD. Scheduled (within one year): a dent deeper "
                f"than 2% of OD within {f['weld_zone_in']} in of a girth weld or on a longitudinal seam; any dent deeper than 6% "
                f"of OD; dent strain above {f['strain_limit']}%; an unconfirmed metal-loss indication in a dent. "
                f"{'For liquid lines with frequent pressure cycling, dents deeper than 2% of OD need a fatigue assessment. ' if f['liquid'] else ''}"
                f"Otherwise monitor.")

    def _fields(self, f):
        ml = {"none": "none", "low_confidence": f"possible metal loss, about {f['ml_depth']}% WT (low confidence)",
              "metal_loss": f"metal loss {f['ml_depth']}% WT within the dent"}[f["coincident"]]
        prev = {True: "not present in the previous caliper run", False: "present and unchanged since the baseline caliper run",
                None: "no previous caliper run on this line"}[f["new_since_last"]]
        return ml, prev

    def render(self, f, rng, surface):
        st = pick(rng, DATE_STYLE)
        ml, prev = self._fields(f)
        strain = show(f["strain_pct"], rng, lambda v: f"{v}%") if f["strain_pct"] is not None else "not computed (sensor spacing too coarse)"
        ctx = CONTEXT[f["context"]]
        seam = f"seam at {f['seam_clock']}" if f["seam_clock"] else "seamless pipe"
        if surface == "json":
            return dumps(prune({
                "dent": {"feature_id": f["feature_id"], "depth_pct_od": f["dent_depth_pct_od"], "clock": f["clock"],
                         "axial_length_in": f["length_in"], "distance_to_girth_weld_in": f["gw_dist_in"],
                         "on_longitudinal_seam": f["on_seam"], "coincident_mfl_signal": ml, "strain_pct": strain,
                         "history": prev, "odometer_ft": f["odometer_ft"]},
                "pipe": {"line": f["line"], "operator": f["operator"], "product": f["product"], "od_in": f["D"],
                         "wall_in": f["t"], "grade": f["grade"], "seam": f"{f['seam']} ({seam})", "maop_psig": f["maop"],
                         "pressure_cycles_per_year_over_20pct_maop": f["cycles_per_year"]},
                "location": ctx,
                "run": {"vendor": f["vendor"], "date": fmt_date(f["run_date"], style="iso")},
                "criteria": self._criteria(f),
            }))
        if surface == "caliper_listing":
            hdr = (f"{f['vendor']} GEOMETRY / IMU REPORT - {f['operator']} {f['line']} ({f['product']}) - run of {fmt_date(f['run_date'], style=st)}\n"
                   f"Pipe {f['D']} in x {f['t']} in {f['grade']} {f['seam']} ({seam}), MAOP {f['maop']} psig")
            tbl = table(["feature", "odometer ft", "clock", "depth %OD", "length in", "to GW in", "strain", "MFL"],
                        [[f["feature_id"], f"{f['odometer_ft']:,}", f["clock"], f["dent_depth_pct_od"], f["length_in"],
                          f["gw_dist_in"], strain, ml]], rng)
            extra = bullets([f"History: {prev}", f"Pressure cycles >20% MAOP: {f['cycles_per_year']}/yr",
                             f"On seam: {'yes' if f['on_seam'] else 'no'}", (f"Location: {ctx}" if ctx else None)], rng)
            return f"{hdr}\n\n{tbl}\n\n{extra}\n\n{self._criteria(f)}"
        s = [f"Dent {f['feature_id']} on {f['line']} ({f['product']}, {f['D']} in {f['seam']}, MAOP {f['maop']} psig): {f['dent_depth_pct_od']}% of OD at "
             f"{f['clock']}, {f['length_in']} in long, {f['gw_dist_in']} in from the nearest girth weld"
             + (", sitting on the longitudinal seam." if f["on_seam"] else (f"; seam at {f['seam_clock']}." if f["seam_clock"] else ".")),
             f"Coincident MFL: {ml}.", f"Strain: {strain}.", f"History: {prev}.",
             (f"Location is {ctx}." if ctx else ""),
             f"The line sees about {f['cycles_per_year']} pressure cycles a year of more than 20% of MAOP.",
             pick(rng, ["Field ops think it's an old rock dent.", "The landowner mentioned contractors working nearby last year.",
                        "No one-call tickets on record here in the last two years.", ""])]
        return (f"INTEGRITY REVIEW NOTE - {f['operator']}\nReviewer: {f['engineer']}\n\n" + noisy(join_sentences(s), rng, 0.1)
                + "\n\n" + self._criteria(f))

    def questions(self):
        return {
            "dent_response": choice("How should the integrity plan respond to this dent?", {
                "immediate_excavation": "Excavate and examine it now, with the plan's interim pressure reduction.",
                "scheduled_excavation": "Excavate it within the plan's deadline for scheduled conditions.",
                "engineering_assessment": "Carry out a strain or fatigue engineering assessment to set the response.",
                "monitor_next_inspection": "Record it and look at it again with the next geometry run.",
            }),
            "dent_origin": choice("What most likely caused this dent?", {
                "third_party_excavation": "Recent excavation or machinery working over or near the pipe.",
                "rock_or_construction_damage": "A rock or damage dating from construction, unchanged since.",
                "soil_movement_or_bending": "Ground movement bending or buckling the pipe.",
                "undetermined_needs_excavation": "Nothing on file points to a cause; only exposing the pipe will tell.",
            }),
            "dent_severity": score("Taken together, how severe is this dent as reported?", [
                "Benign: shallow, smooth and away from welds.",
                "Notable: deeper than routine but with nothing else attached.",
                "Serious: meets a scheduled-response condition.",
                "Critical: meets an immediate-response condition.",
            ]),
            "fatigue_concern": score("How much does pressure cycling threaten this dent with fatigue cracking?", [
                "Negligible: the line hardly cycles.",
                "Low: some cycling, on a dent unlikely to concentrate stress.",
                "Moderate: frequent cycling on a dent of real depth.",
                "High: heavy cycling on a deep or weld-affected dent.",
            ]),
            "strain_exceeds_limit": noul(
                "The dent's calculated strain exceeds the limit in the plan's dent criteria.",
                yes="A strain value is reported and it is above the plan's strain limit.",
                no="The reported strain is at or below the limit, or no strain was computed."),
        }


# =========================================================================== schema 3
class InteractingCorrosionCluster(Schema):
    name = "interacting_corrosion_cluster"
    description = "Neighbouring metal-loss features that may interact under the plan's rule, changing the assessed failure pressure."
    surfaces = {"json": 0.5, "assessment_worksheet": 0.3, "vendor_cluster_note": 0.2}

    def sample_factors(self, rng):
        reg = regime(rng)
        p = _pipe(rng)
        rule = pick(rng, ["6t", "1in", "6t"])
        thresh = round(6 * p["t"], 2) if rule == "6t" else 1.0
        n = rng.randint(2, 4)
        depths = [rng.randint(14, 62) for _ in range(n)]
        lens = [round(rng.uniform(0.6, 4.5), 1) for _ in range(n)]
        gaps = [round(pick_w(rng, {rng.uniform(0.1, thresh * 0.8): 3, near(rng, thresh, thresh * 0.25, lo=0.05): 3,
                                   rng.uniform(thresh * 1.3, thresh * 4): 3}), 2) for _ in range(n - 1)]
        circ = round(pick_w(rng, {rng.uniform(0.0, thresh * 0.9): 4, rng.uniform(thresh * 1.1, thresh * 5): 2}), 2)
        sched_fpr, sched_depth = pick(rng, [1.25, 1.39]), pick(rng, [50, 60])
        weld = pick_w(rng, {"none": 9, "girth_weld": 1, "seam": 1})
        tool = pick_w(rng, {"MFL-A": 3, "MFL-C": 1, "MFL-A + MFL-C": 2, "UT": 1})
        narrow_axial = chance(rng, 0.25)
        if reg == "clear":
            if chance(rng, 0.5):
                gaps = [round(rng.uniform(thresh * 2, thresh * 5), 2) for _ in gaps]
                circ = round(rng.uniform(thresh * 1.5, thresh * 5), 2)
                depths = [rng.randint(12, 30) for _ in depths]
                weld, narrow_axial = "none", False
            else:
                gaps = [round(rng.uniform(0.05, thresh * 0.5), 2) for _ in gaps]
                circ = round(rng.uniform(0.0, thresh * 0.5), 2)
                depths = [rng.randint(45, 72) for _ in depths]
                lens = [round(rng.uniform(2.0, 6.0), 1) for _ in lens]
        elif reg == "borderline":
            gaps[0] = round(near(rng, thresh, thresh * 0.12, lo=0.05), 2)
        else:
            # individually harmless features that only matter combined, or a cluster on a weld the tool sizes poorly
            if chance(rng, 0.5):
                depths = [rng.randint(30, 48) for _ in depths]
                gaps = [round(rng.uniform(0.05, thresh * 0.9), 2) for _ in gaps]
                circ = round(rng.uniform(0.0, thresh * 0.8), 2)
                lens = [round(rng.uniform(2.0, 5.0), 1) for _ in lens]
            elif chance(rng, 0.6):
                weld, narrow_axial = pick(rng, ["girth_weld", "seam"]), True
            else:
                narrow_axial, tool = True, "MFL-A"
        if weld == "seam" and p["seam"] == "seamless":
            weld = "girth_weld"
        interact = all(g <= thresh for g in gaps) and circ <= thresh
        comb = round(sum(lens) + sum(gaps), 1)

        def gov_fpr(ds):
            if interact:
                return pf_mod_b31g(p["D"], p["t"], p["smys"], max(ds) / 100, comb) / p["maop"]
            return min(pf_mod_b31g(p["D"], p["t"], p["smys"], d / 100, L) for d, L in zip(ds, lens)) / p["maop"]
        if reg == "borderline" or (reg == "conflicting" and chance(rng, 0.4)):
            aim = sched_fpr + rng.uniform(-0.05, 0.09)
        elif reg == "clear" and interact:
            aim = rng.uniform(1.02, 1.13)
        elif gov_fpr(depths) < 1.02:
            aim = rng.uniform(1.03, 1.15)
        else:
            aim = None
        if aim is not None:
            best = None
            for k in [x / 20 for x in range(6, 31)]:
                ds = [max(8, min(79, int(round(d * k)))) for d in depths]
                err = abs(gov_fpr(ds) - aim)
                if best is None or err < best[0]:
                    best = (err, ds)
            depths = best[1]
        worst = max(range(n), key=lambda i: pf_mod_b31g(p["D"], p["t"], p["smys"], 0, 1) - pf_mod_b31g(p["D"], p["t"], p["smys"], depths[i] / 100, lens[i]))
        pf_ind = pf_mod_b31g(p["D"], p["t"], p["smys"], depths[worst] / 100, lens[worst])
        comb_len = round(sum(lens) + sum(gaps), 1)
        pf_comb = pf_mod_b31g(p["D"], p["t"], p["smys"], max(depths) / 100, comb_len)
        return {"regime": reg, "operator": _operator(rng), "line": _line_name(rng), **p, "rule": rule,
                "threshold_in": thresh, "depths": depths, "lengths": lens, "axial_gaps": gaps, "circ_sep_in": circ,
                "interacts": interact, "pf_individual": int(round(pf_ind, -1)), "pf_combined": int(round(pf_comb, -1)),
                "combined_len": comb_len, "sched_fpr": sched_fpr, "sched_depth": sched_depth, "weld": weld, "tool": tool,
                "narrow_axial": narrow_axial, "tolerance": pick(rng, [8, 10, 10]), "vendor": pick(rng, VENDORS),
                "cluster_id": f"C-{rng.randint(100, 999)}", "odometer_ft": rng.randint(1200, 290000), "clock": _clock(rng),
                "engineer": person(rng), "run_date": rand_date(rng, date(2025, 1, 6), date(2027, 3, 1))}

    def intended(self, f):
        interact = f["interacts"]
        pf = f["pf_combined"] if interact else f["pf_individual"]
        fpr = pf / f["maop"]
        dmax = max(f["depths"])
        if dmax >= 80 or fpr < 1.10:
            resp = "immediate_excavation"
        elif dmax >= f["sched_depth"] or fpr < f["sched_fpr"] or f["weld"] != "none":
            resp = "scheduled_excavation"
        elif fpr < f["sched_fpr"] + 0.08:
            resp = "detailed_assessment_first"
        else:
            resp = "monitor_next_inspection"
        if f["weld"] != "none" or (f["narrow_axial"] and "MFL-C" not in f["tool"] and f["tool"] != "UT"):
            method = "field_measurement_required"
        elif interact and abs(fpr - f["sched_fpr"]) <= 0.08:
            method = "effective_area_profile"
        elif interact:
            method = "combined_feature_mod_b31g"
        else:
            method = "individual_features_mod_b31g"
        sev = 3 if (dmax >= 80 or fpr < 1.10) else 2 if (fpr < f["sched_fpr"] or dmax >= f["sched_depth"]) else 1 if fpr < f["sched_fpr"] + 0.15 else 0
        conf = 3
        if f["narrow_axial"] and "MFL-C" not in f["tool"] and f["tool"] != "UT":
            conf -= 2
        if f["weld"] != "none":
            conf -= 1
        if min(f["axial_gaps"]) and any(abs(g - f["threshold_in"]) <= 0.15 * f["threshold_in"] for g in f["axial_gaps"]):
            conf -= 1
        if f["tool"] == "UT":
            conf += 1
        return {"cluster_response": resp, "governing_assessment": method, "combined_severity": sev,
                "sizing_confidence": max(0, min(3, conf)), "interaction_rule_applies": "yes" if interact else "no"}

    def _criteria(self, f):
        rule = (f"six times the nominal wall thickness ({f['threshold_in']} in for this pipe)" if f["rule"] == "6t"
                else "1.0 in")
        return (f"IMP 7.3.2 interaction: metal-loss features interact when every axial gap between them and their "
                f"circumferential separation are no more than {rule}; interacting features are assessed as one feature "
                f"spanning their combined length at the maximum depth. Cluster categories: dig at once when the deepest point "
                f"reaches 80% of wall or the governing burst estimate is under 1.10 times MAOP; dig on the scheduled timetable when "
                f"the deepest point reaches {f['sched_depth']}% of wall, the governing burst estimate is under {f['sched_fpr']:.2f} "
                f"times MAOP, or any feature touches a girth weld or longitudinal seam; otherwise keep it on the watch list.")

    def _rows(self, f):
        rows, pos = [], 0.0
        for i, (d, L) in enumerate(zip(f["depths"], f["lengths"])):
            rows.append([f"{f['cluster_id']}-{i + 1}", f"{pos:.2f}", L, d])
            pos += L + (f["axial_gaps"][i] if i < len(f["axial_gaps"]) else 0)
        return rows

    def render(self, f, rng, surface):
        st = pick(rng, DATE_STYLE)
        weld = {"none": "clear of welds", "girth_weld": "the downstream feature runs into the girth weld",
                "seam": "the features lie along the longitudinal seam"}[f["weld"]]
        sizing = (f"{f['tool']}; depth tolerance +/-{f['tolerance']}% WT"
                  + ("; vendor flags narrow axially aligned features (possible axial slotting)" if f["narrow_axial"] else ""))
        if surface == "json":
            return dumps(prune({
                "cluster": {"id": f["cluster_id"], "odometer_ft": f["odometer_ft"], "clock": f["clock"],
                            "features": [{"id": r[0], "start_in": float(r[1]), "length_in": r[2], "depth_pct_wall": r[3]} for r in self._rows(f)],
                            "axial_gaps_in": f["axial_gaps"], "circumferential_separation_in": f["circ_sep_in"],
                            "weld_proximity": weld},
                "assessment": {"worst_single_feature_failure_pressure_psig": f["pf_individual"],
                               "failure_pressure_if_combined_psig": f["pf_combined"], "combined_length_in": f["combined_len"],
                               "method": "modified B31G"},
                "pipe": {"line": f["line"], "operator": f["operator"], "od_in": f["D"], "wall_in": f["t"], "grade": f["grade"],
                         "seam": f["seam"], "maop_psig": f["maop"], "class_location": f["class"]},
                "tool": {"vendor": f["vendor"], "sizing": sizing},
                "criteria": self._criteria(f),
            }))
        if surface == "assessment_worksheet":
            hdr = (f"INTEGRITY ASSESSMENT WORKSHEET - {f['operator']} {f['line']} - cluster {f['cluster_id']} - {f['engineer']}\n"
                   f"Pipe {f['D']} x {f['t']} in {f['grade']} {f['seam']} ({f['year']}), MAOP {f['maop']} psig, class {f['class']}\n"
                   f"Tool: {f['vendor']} {sizing}; run {fmt_date(f['run_date'], style=st)}")
            tbl = table(["feature", "start in", "length in", "depth %WT"], self._rows(f), rng)
            calc = kv_lines([("Axial gaps (in)", ", ".join(str(g) for g in f["axial_gaps"])),
                             ("Circumferential separation (in)", f["circ_sep_in"]), ("Weld proximity", weld),
                             ("Pf worst single feature (Mod. B31G)", f"{f['pf_individual']} psig"),
                             ("Pf if combined, L = {0} in (Mod. B31G)".format(f["combined_len"]), f"{f['pf_combined']} psig")], rng)
            return f"{hdr}\n\n{tbl}\n\n{calc}\n\n{self._criteria(f)}"
        s = [f"{f['vendor']} reports a cluster of {len(f['depths'])} metal-loss features ({f['cluster_id']}) at {f['odometer_ft']:,} ft on {f['line']}, "
             f"{f['clock']} o'clock: depths {', '.join(str(d) + '%' for d in f['depths'])}; lengths {', '.join(str(x) for x in f['lengths'])} in.",
             f"Axial gaps {', '.join(str(g) for g in f['axial_gaps'])} in; circumferential separation {f['circ_sep_in']} in; {weld}.",
             f"Sizing: {sizing}.",
             f"The vendor reports the worst single feature at {f['pf_individual']} psig failure pressure; treated as one {f['combined_len']} in "
             f"feature it would be {f['pf_combined']} psig.",
             f"Pipe: {f['D']} x {f['t']} in {f['grade']} {f['seam']}, MAOP {f['maop']} psig."]
        return "VENDOR CLUSTER NOTE\n\n" + noisy(join_sentences(s), rng, 0.1) + "\n\n" + self._criteria(f)

    def questions(self):
        return {
            "cluster_response": choice("How should the integrity plan respond to this cluster?", {
                "immediate_excavation": "Excavate now, with an interim pressure reduction until it is repaired.",
                "scheduled_excavation": "Excavate within the plan's deadline for scheduled conditions.",
                "detailed_assessment_first": "Refine the assessment before committing to a dig.",
                "monitor_next_inspection": "Record it and reassess with the next in-line inspection.",
            }),
            "governing_assessment": choice("Which assessment should govern the response to this cluster?", {
                "individual_features_mod_b31g": "Each feature on its own, by modified B31G.",
                "combined_feature_mod_b31g": "The features combined into one, by modified B31G.",
                "effective_area_profile": "An effective-area calculation on the measured depth profile of the combined feature.",
                "field_measurement_required": "None of the tool-based numbers; the cluster must be measured in the ditch.",
            }),
            "combined_severity": score("Assessed the way the plan requires, how severe is this cluster?", [
                "Minor: well inside every criterion.",
                "Watch: close to the scheduled-response line.",
                "Significant: meets a scheduled-response condition.",
                "Critical: meets an immediate-response condition.",
            ]),
            "sizing_confidence": score("How far can the tool's sizing of these features be trusted for this decision?", [
                "Hardly: the geometry or location is one the tool sizes poorly.",
                "Partly: a known weakness or a value right at a rule's edge.",
                "Reasonably: ordinary tool tolerance on ordinary features.",
                "Highly: a sizing method well suited to these features, with nothing near an edge.",
            ]),
            "interaction_rule_applies": noul(
                "Under the plan's interaction rule, these features must be assessed as one combined feature.",
                yes="Every axial gap and the circumferential separation are within the plan's interaction distance.",
                no="At least one gap, or the circumferential separation, exceeds the interaction distance."),
        }


# =========================================================================== schema 4
FEATURES = {
    "external_corrosion": "external corrosion",
    "internal_corrosion": "internal corrosion (UT from outside)",
    "dent_with_gouge": "dent with a gouge",
    "scc_colony": "stress-corrosion crack colony",
    "arc_burn": "arc burn",
    "plain_dent": "plain dent, no gouge or crack",
}


class FieldRepairSelection(Schema):
    name = "field_repair_selection"
    description = "An anomaly exposed and measured in the ditch; choose the repair the operator's repair manual allows."
    surfaces = {"json": 0.5, "dig_report": 0.3, "field_call_notes": 0.2}

    def sample_factors(self, rng):
        reg = regime(rng)
        liquid = chance(rng, 0.3)
        p = _pipe(rng, liquid)
        feat = pick_w(rng, {"external_corrosion": 5, "internal_corrosion": 1, "dent_with_gouge": 1, "scc_colony": 1,
                            "arc_burn": 1, "plain_dent": 1})
        depth = rng.randint(6, 78) if feat in ("external_corrosion", "internal_corrosion") else rng.randint(4, 40)
        length = round(pick_w(rng, {rng.uniform(1, 12): 6, rng.uniform(12, 60): 2, rng.uniform(60, 140): 0.7}), 1)
        on_gw = chance(rng, 0.15)
        on_lf = p["seam"] == "LF-ERW" and chance(rng, 0.35)
        ce = round(pick_w(rng, {rng.uniform(0.30, 0.40): 5, near(rng, 0.42, 0.02): 2, rng.uniform(0.44, 0.55): 1.5}), 2)
        ce_limit = pick(rng, [0.42, 0.45])
        sleeve_max_in = pick(rng, [48, 72, 96])
        outage = pick_w(rng, {"bypass_available": 2, "single_feed_winter": 2, "parallel_loop": 2, "shutdown_window_in_6_months": 1})
        flow = pick_w(rng, {"normal": 4, "low": 1, "none_isolated": 1})
        if reg == "clear":
            if chance(rng, 0.5):
                feat, depth, length, on_gw, on_lf = "external_corrosion", rng.randint(6, 16), round(rng.uniform(1, 8), 1), False, False
            elif chance(rng, 0.5):
                feat, depth, length = "scc_colony", rng.randint(15, 40), round(rng.uniform(sleeve_max_in + 10, sleeve_max_in + 60), 1)
            else:
                feat, depth, length, on_gw = pick(rng, ["scc_colony", "dent_with_gouge", "arc_burn"]), rng.randint(10, 40), round(rng.uniform(4, 40), 1), False
                ce, flow = round(rng.uniform(0.30, 0.39), 2), "normal"
        elif reg == "borderline":
            ce = round(near(rng, ce_limit, 0.02), 2)
            if feat == "external_corrosion" and chance(rng, 0.5):
                depth = int(near(rng, pick(rng, [20, 80]), 4, lo=5, hi=90, integer=True))
        else:
            # composite looks perfect, but the feature sits on a weld or seam; or a weld sleeve is wanted on high-CE pipe
            if chance(rng, 0.5):
                feat, depth, on_gw = "external_corrosion", rng.randint(30, 70), True
            else:
                feat, ce = pick(rng, ["dent_with_gouge", "internal_corrosion"]), round(rng.uniform(ce_limit + 0.01, 0.55), 2)
        if feat == "arc_burn":
            length = round(rng.uniform(0.4, 3.0), 1)
        elif feat in ("plain_dent", "dent_with_gouge"):
            length = round(rng.uniform(3, 24), 1)
        if feat == "plain_dent":
            depth = rng.randint(2, 7)      # %OD for a dent; reported as depth of dent
        ili_depth = max(0, depth + pick_w(rng, {rng.randint(-4, 4): 4, rng.randint(-10, 10): 3, rng.randint(-25, -11): 2,
                                                rng.randint(11, 20): 1}))
        ili_type = feat if chance(rng, 0.85) else pick(rng, [x for x in FEATURES if x != feat])
        if ili_type != feat:   # a different feature type was called, sized on its own scale
            ili_depth = rng.randint(1, 5) if ili_type == "plain_dent" else rng.randint(10, 45)
        pf = pf_mod_b31g(p["D"], p["t"], p["smys"], depth / 100, min(length, 40)) if feat in ("external_corrosion", "internal_corrosion") else None
        return {"regime": reg, "operator": _operator(rng, liquid), "liquid": liquid, "line": _line_name(rng), **p,
                "feature": feat, "field_depth_pct": depth, "length_in": length, "on_girth_weld": on_gw, "on_lf_erw_seam": on_lf,
                "ce": ce, "ce_limit": ce_limit, "sleeve_max_in": sleeve_max_in, "ili_depth_pct": ili_depth,
                "ili_feature": ili_type, "outage": outage, "flow": flow, "pf_field": int(round(pf, -1)) if pf else None,
                "pressure_now_pct": pick(rng, [80, 80, 70, 100]), "dig_seq": rng.randint(10, 299),
                "nde_method": pick(rng, ["laser scan + UT", "pit gauge + UT", "MPI + UT", "phased-array UT"]),
                "crew_lead": person(rng), "engineer": person(rng), "date": rand_date(rng, date(2025, 4, 1), date(2027, 6, 1))}

    def _allowed(self, f):
        feat, d = f["feature"], f["field_depth_pct"]
        weld_issue = f["on_girth_weld"] or f["on_lf_erw_seam"]
        recoat = feat == "external_corrosion" and d <= 20 and not weld_issue
        if feat == "arc_burn":
            recoat = False
        composite = feat == "external_corrosion" and 20 < d <= 80 and not weld_issue and f["length_in"] <= f["sleeve_max_in"]
        type_a = (feat in ("external_corrosion", "plain_dent") and d <= 80 and not f["on_lf_erw_seam"]
                  and f["length_in"] <= f["sleeve_max_in"])
        weld_ok = f["ce"] <= f["ce_limit"] and f["t"] >= 0.250 and f["flow"] != "none_isolated"
        type_b = weld_ok and f["length_in"] <= f["sleeve_max_in"] and not f["on_girth_weld"]
        return recoat, composite, type_a, type_b, weld_ok

    def intended(self, f):
        recoat, composite, type_a, type_b, weld_ok = self._allowed(f)
        if recoat:
            m = "recoat_only"
        elif composite:
            m = "composite_wrap"
        elif type_a:
            m = "steel_reinforcing_sleeve"
        elif type_b:
            m = "welded_pressure_sleeve"
        else:
            m = "cut_out_and_replace"
        feat, d = f["feature"], f["field_depth_pct"]
        if feat in ("scc_colony", "dent_with_gouge"):
            sev = 3 if d >= 25 or feat == "dent_with_gouge" else 2
        elif feat in ("external_corrosion", "internal_corrosion"):
            fpr = f["pf_field"] / f["maop"]
            sev = 3 if (d >= 80 or fpr < 1.10) else 2 if (d >= 50 or fpr < 1.39) else 1 if d > 20 else 0
        else:
            sev = 1 if feat == "arc_burn" else (1 if d > 4 else 0)
        diff = abs(f["ili_depth_pct"] - d)
        agree = 3 if diff <= 5 else 2 if diff <= 10 else 1 if diff <= 20 else 0
        if f["ili_feature"] != feat:
            agree = 0
        out = f["outage"] in ("bypass_available", "parallel_loop")
        return {"repair_method": m, "as_found_severity": sev, "ili_field_agreement": agree,
                "in_service_welding_ok": "yes" if weld_ok else "no", "cutout_outage_feasible": "yes" if out else "no"}

    def _manual(self, f):
        return (f"Repair manual RM-4: recoat only for external corrosion no deeper than 20% of wall, clear of welds. Composite "
                f"wrap for external metal loss deeper than 20% and no deeper than 80%, not on a girth weld or low-frequency ERW "
                f"seam, not for cracks, gouges, dents or internal corrosion. Type A steel reinforcing sleeve for external metal "
                f"loss up to 80% or plain dents, not on LF-ERW seams. Type B welded pressure sleeve for any defect, only where "
                f"in-service welding is allowed (carbon equivalent {f['ce_limit']:.2f} or less, wall 0.250 in or more, gas or "
                f"liquid flowing), not over a girth weld. Longest sleeve stocked: {f['sleeve_max_in']} in. Otherwise cut out and "
                f"replace. Use the least invasive repair the manual allows.")

    def render(self, f, rng, surface):
        st = pick(rng, DATE_STYLE)
        outage = {"bypass_available": "temporary bypass can be set up in about 3 days",
                  "single_feed_winter": "segment is the single feed to a town; shutdown not possible until spring",
                  "parallel_loop": "parallel loop line can carry full flow during a shutdown",
                  "shutdown_window_in_6_months": "next planned shutdown window is in about 6 months"}[f["outage"]]
        flow = {"normal": "normal flow", "low": "low flow (in-service welding cooling marginal)", "none_isolated": "segment isolated, no flow"}[f["flow"]]
        unit = "% OD" if f["feature"] == "plain_dent" else "% WT"
        iunit = "% OD" if f["ili_feature"] == "plain_dent" else "% WT"
        dig_id = f"DIG-{f['date'].year}-{f['dig_seq']:03d}"
        if surface == "json":
            return dumps(prune({
                "dig": {"id": dig_id, "date": fmt_date(f["date"], style="iso"), "crew_lead": f["crew_lead"],
                        "nde": f["nde_method"]},
                "as_found": {"feature": FEATURES[f["feature"]], "max_depth": f"{f['field_depth_pct']}{unit}",
                             "length_in": f["length_in"], "on_girth_weld": f["on_girth_weld"],
                             "on_lf_erw_seam": f["on_lf_erw_seam"], "failure_pressure_psig_mod_b31g": f["pf_field"]},
                "ili_call": {"feature": FEATURES[f["ili_feature"]], "depth": f"{f['ili_depth_pct']}{iunit}"},
                "pipe": {"line": f["line"], "operator": f["operator"], "od_in": f["D"], "wall_in": f["t"], "grade": f["grade"],
                         "seam": f["seam"], "carbon_equivalent": f["ce"], "maop_psig": f["maop"],
                         "pressure_now_pct_maop": f["pressure_now_pct"], "flow": flow},
                "outage_options": outage,
                "repair_manual": self._manual(f),
            }))
        if surface == "dig_report":
            rows = [("Dig", dig_id), ("Date", fmt_date(f["date"], style=st)), ("Line", f"{f['line']} ({f['operator']})"),
                    ("Pipe", f"{f['D']} x {f['t']} in {f['grade']} {f['seam']}, CE {f['ce']}"), ("MAOP", f"{f['maop']} psig"),
                    ("Pressure during dig", f"{f['pressure_now_pct']}% of MAOP"), ("Flow", flow),
                    ("ILI call", f"{FEATURES[f['ili_feature']]}, {f['ili_depth_pct']}{iunit}"),
                    ("As found", f"{FEATURES[f['feature']]}, max {f['field_depth_pct']}{unit}, {f['length_in']} in long ({f['nde_method']})"),
                    ("On girth weld", "yes" if f["on_girth_weld"] else "no"), ("On LF-ERW seam", "yes" if f["on_lf_erw_seam"] else "no"),
                    ("Failure pressure (as found)", f"{f['pf_field']} psig" if f["pf_field"] else None),
                    ("Outage options", outage)]
            return f"FIELD EXAMINATION REPORT\n{kv_lines(rows, rng)}\n\n{self._manual(f)}"
        s = [f"{f['crew_lead']} on {dig_id} ({f['line']}): pipe exposed, {f['nde_method']} done.",
             f"Found {FEATURES[f['feature']]}, max {f['field_depth_pct']}{unit}, {f['length_in']} in long"
             + (", running onto the girth weld" if f["on_girth_weld"] else "") + (", along the LF-ERW seam" if f["on_lf_erw_seam"] else "") + ".",
             f"ILI had it as {FEATURES[f['ili_feature']]} at {f['ili_depth_pct']}{iunit}.",
             (f"As-found failure pressure works out at {f['pf_field']} psig against MAOP {f['maop']}." if f["pf_field"] else f"MAOP {f['maop']} psig."),
             f"Mill cert CE {f['ce']}; wall {f['t']} in; {flow}; holding {f['pressure_now_pct']}% of MAOP.",
             f"Gas control says: {outage}.",
             pick(rng, ["Crew has composite kits on the truck.", "Welder on site is qualified for in-service sleeves.",
                        "We'd like to backfill before the weekend.", ""])]
        return (f"FIELD CALL NOTES - {f['engineer']} - {fmt_date(f['date'], style=st)}\n\n" + noisy(join_sentences(s), rng, 0.1)
                + "\n\n" + self._manual(f))

    def questions(self):
        return {
            "repair_method": choice("Which repair should be installed on this anomaly?", {
                "recoat_only": "Clean and recoat the pipe; no reinforcement is needed.",
                "composite_wrap": "Install a composite wrap repair.",
                "steel_reinforcing_sleeve": "Install a steel reinforcing sleeve that is not welded to the pipe.",
                "welded_pressure_sleeve": "Install a pressure-containing steel sleeve welded to the pipe in service.",
                "cut_out_and_replace": "Take the segment out of service, cut out the damaged pipe and replace it.",
            }),
            "as_found_severity": score("How severe is the anomaly as measured in the ditch?", [
                "Minor: shallow, no reinforcement needed.",
                "Moderate: needs reinforcement, with comfortable strength left.",
                "Serious: little strength margin, or a feature type the manual treats strictly.",
                "Critical: at or past immediate-response levels, or a crack or gouge that must be removed.",
            ]),
            "ili_field_agreement": score("How well did the in-line inspection call match what was found?", [
                "Poorly: wrong feature type or a depth far off.",
                "Loosely: depth off by more than the usual tolerance.",
                "Reasonably: within about ten percent of wall.",
                "Closely: right feature type and depth within a few percent.",
            ]),
            "in_service_welding_ok": noul(
                "The pipe and operating conditions allow a sleeve to be welded on while the line stays in service.",
                yes="Carbon equivalent and wall thickness are within the manual's limits and there is flow to cool the weld.",
                no="Carbon equivalent is above the limit, the wall is too thin, or there is no flow."),
            "cutout_outage_feasible": noul(
                "The segment could be taken out of service for a cut-out within the next few days."),
        }


# =========================================================================== schema 5
REASONS = {
    "landowner_refusal": ("landowner refuses access; court order for entry requested", True),
    "wetland_permit": ("state wetland crossing permit still pending", True),
    "frozen_ground": ("ground frozen to 4 ft; excavation in frost would need thawing equipment", True),
    "river_crossing": ("anomaly lies under a river bank; specialist contractor booked", True),
    "crop_season": ("landowner asks that the dig wait until harvest", False),
    "contractor_shortage": ("dig contractor fully booked on other projects", False),
    "budget": ("this year's integrity budget is spent", False),
}


# how long each cause plausibly holds a dig up (days); the ruling compares the request with this
NEED_DAYS = {"frozen_ground": 120, "wetland_permit": 90, "landowner_refusal": 75, "river_crossing": 90, "crop_season": 60,
             "contractor_shortage": 45, "budget": 180}


class DigDeferralRequest(Schema):
    name = "dig_deferral_request"
    description = "Operations asks to push a required excavation past its response deadline; the integrity engineer rules."
    surfaces = {"json": 0.5, "deferral_form": 0.25, "ops_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        p = _pipe(rng)
        depth = rng.randint(48, 76)
        target = rng.uniform(1.04, 1.38)
        k = pick(rng, [1.25, 1.39])
        reason = pick_w(rng, {"landowner_refusal": 2, "wetland_permit": 2, "frozen_ground": 2, "river_crossing": 2,
                              "crop_season": 1.5, "contractor_shortage": 1.5, "budget": 1})
        reduce_pct = pick_w(rng, {None: 3, 80: 3, 85: 2, 90: 2, 70: 1})
        deadline_days = pick(rng, [180, 365])
        days_left = pick_w(rng, {rng.randint(1, 14): 3, rng.randint(15, 45): 3, rng.randint(46, 120): 2})
        cls = p["class"]
        hca = pick_w(rng, {"yes": 2, "no": 5, "boundary_unresolved": 1} if cls < 3 else {"yes": 5, "no": 1, "boundary_unresolved": 1})
        stretch = pick_w(rng, {rng.uniform(0.5, 1.15): 4, rng.uniform(1.5, 2.6): 3})
        if reg == "clear":
            if chance(rng, 0.5):
                reason, reduce_pct, target, stretch = pick(rng, ["wetland_permit", "landowner_refusal"]), 80, rng.uniform(1.25, 1.45), rng.uniform(0.6, 1.0)
                hca = "no" if cls < 3 else hca
            else:
                reason, reduce_pct, hca = pick(rng, ["budget", "contractor_shortage"]), None, "yes"
        elif reg == "conflicting":
            # a sound reason but no real interim margin, or an in-house reason with a strong pressure cut
            if chance(rng, 0.5):
                reason, reduce_pct = pick(rng, ["river_crossing", "frozen_ground"]), pick(rng, [None, 90])
            else:
                reason, reduce_pct = pick(rng, ["contractor_shortage", "crop_season"]), 70
        L = _length_for_fpr(p["D"], p["t"], p["smys"], depth / 100, p["maop"], target)
        pf = pf_mod_b31g(p["D"], p["t"], p["smys"], depth / 100, L)
        extra = int(NEED_DAYS[reason] * stretch)
        disc = rand_date(rng, date(2025, 1, 6), date(2027, 1, 1))
        deadline = disc + timedelta(days=deadline_days)
        today = deadline - timedelta(days=days_left)
        structures = rng.randint(0, 3) if cls == 1 else rng.randint(4, 30) if cls == 2 else rng.randint(30, 160)
        return {"regime": reg, "operator": _operator(rng), "line": _line_name(rng), **p, "class": cls,
                "depth_pct": depth, "length_in": round(L, 1), "pf": int(round(pf, -1)), "k_required": k,
                "reason": reason, "reduce_to_pct": reduce_pct, "deadline_days": deadline_days, "discovery": disc,
                "deadline": deadline, "today": today, "days_left": days_left, "extra_days": extra, "hca": hca,
                "structures_in_pir": structures, "pir": int(round(pir_ft(p["D"], p["maop"]), -1)),
                "struct_type": pick(rng, STRUCT_TYPES), "patrols": pick(rng, ["weekly ground patrol", "daily drive-by",
                                                                              "monthly aerial patrol only", "leak survey every two weeks"]),
                "requester": person(rng), "engineer": person(rng), "feature_id": rng.randint(1000, 99999)}

    def intended(self, f):
        outside = REASONS[f["reason"]][1]
        p_int = f["maop"] * (f["reduce_to_pct"] / 100 if f["reduce_to_pct"] else 1.0)
        margin_ok = f["pf"] / p_int >= f["k_required"]
        conseq = 0 if (f["class"] == 1 and f["hca"] == "no") else 1
        if f["hca"] == "yes":
            conseq = 2
        if f["class"] >= 3 or (f["hca"] == "yes" and f["structures_in_pir"] > 20):
            conseq = 3
        if f["hca"] == "boundary_unresolved":
            conseq = max(conseq, 2)
        needed_extra = NEED_DAYS[f["reason"]]
        if not outside:
            ruling = "deny_dig_on_schedule"
        elif not margin_ok:
            ruling = "grant_with_pressure_restriction"      # a restriction (or a deeper one) is the price of the extension
        elif f["extra_days"] > needed_extra * 1.3 + 15:
            ruling = "grant_shorter_extension"
        else:
            ruling = "grant_as_requested"
        pressure = 3 if f["days_left"] <= 14 else 2 if f["days_left"] <= 45 else 1 if f["days_left"] <= 90 else 0
        notice = f["hca"] in ("yes", "boundary_unresolved") or f["extra_days"] > 120
        return {"deferral_ruling": ruling, "failure_consequence": conseq, "deadline_pressure": pressure,
                "interim_margin_adequate": "yes" if margin_ok else "no",
                "regulator_notice_required": "yes" if notice else "no"}

    def _rule(self, f):
        return (f"IMP 7.6 deferral: a response deadline may be extended only if (a) the cause is outside the operator's "
                f"control and documented, (b) an interim pressure restriction keeps the anomaly's predicted failure pressure "
                f"at least {f['k_required']:.2f} times the restricted operating pressure for the whole extension, and (c) the "
                f"regulator is notified before the deadline when the segment is in a high-consequence area or the extension "
                f"exceeds 120 days. Extensions are granted only for the time the cause needs.")

    def render(self, f, rng, surface):
        st = pick(rng, DATE_STYLE)
        new_date = f["deadline"] + timedelta(days=f["extra_days"])
        red = f"restrict to {f['reduce_to_pct']}% of MAOP" if f["reduce_to_pct"] else "no pressure restriction proposed"
        hca = {"yes": "inside an HCA", "no": "outside any HCA",
               "boundary_unresolved": "HCA boundary about 300 ft away; which side the anomaly is on is not yet resolved"}[f["hca"]]
        if surface == "json":
            return dumps(prune({
                "anomaly": {"feature_id": f["feature_id"], "depth_pct_wall": f["depth_pct"], "length_in": f["length_in"],
                            "predicted_failure_pressure_psig": f["pf"], "category": f"scheduled ({f['deadline_days']}-day)",
                            "discovered": fmt_date(f["discovery"], style="iso"), "deadline": fmt_date(f["deadline"], style="iso")},
                "request": {"submitted": fmt_date(f["today"], style="iso"), "by": f["requester"],
                            "reason": REASONS[f["reason"]][0], "new_date": fmt_date(new_date, style="iso"),
                            "interim_measures": [red, f["patrols"]]},
                "segment": {"line": f["line"], "operator": f["operator"], "od_in": f["D"], "wall_in": f["t"], "grade": f["grade"],
                            "maop_psig": f["maop"], "class_location": f["class"], "hca_status": hca,
                            "potential_impact_radius_ft": f["pir"],
                            "structures_in_impact_radius": f"{f['structures_in_pir']} (incl. {f['struct_type']})" if f["structures_in_pir"] else 0},
                "rule": self._rule(f),
            }))
        if surface == "deferral_form":
            rows = [("Feature", f["feature_id"]), ("Line", f"{f['line']} ({f['operator']})"),
                    ("Pipe", f"{f['D']} x {f['t']} in {f['grade']}, MAOP {f['maop']} psig, class {f['class']}"),
                    ("Anomaly", f"{f['depth_pct']}% WT x {f['length_in']} in, Pf {f['pf']} psig"),
                    ("Discovered", fmt_date(f["discovery"], style=st)), ("Deadline", fmt_date(f["deadline"], style=st)),
                    ("Request date", fmt_date(f["today"], style=st)), ("Proposed new date", fmt_date(new_date, style=st)),
                    ("Reason", REASONS[f["reason"]][0]), ("Interim measures", f"{red}; {f['patrols']}"),
                    ("HCA", hca), ("PIR", f"{f['pir']} ft, {f['structures_in_pir']} structures"),
                    ("Requested by", f["requester"])]
            return "DEADLINE DEFERRAL REQUEST\n" + kv_lines(rows, rng) + "\n\n" + self._rule(f)
        s = [f"We need to move the dig on feature {f['feature_id']} ({f['line']}) from {fmt_date(f['deadline'], style=st)} to "
             f"{fmt_date(new_date, style=st)}.", f"Reason: {REASONS[f['reason']][0]}.",
             f"In the meantime we propose to {red if f['reduce_to_pct'] else 'keep operating normally'} and keep a {f['patrols']}.",
             pick(rng, ["We see this as low risk.", "Happy to discuss.", "Gas control is fine with the restriction.",
                        "Marketing would prefer no restriction during the peak season."])]
        return (f"From: {f['requester']}\nTo: integrity engineering\nDate: {fmt_date(f['today'], style=st)}\n"
                f"Subject: Deferral - {f['line']} feature {f['feature_id']}\n\n" + noisy(join_sentences(s), rng, 0.1)
                + f"\n\n[integrity file: {f['depth_pct']}% WT x {f['length_in']} in; Pf {f['pf']} psig; MAOP {f['maop']} psig; "
                  f"class {f['class']}; {hca}; PIR {f['pir']} ft with {f['structures_in_pir']} structures; "
                  f"discovered {fmt_date(f['discovery'], style=st)}]\n\n" + self._rule(f))

    def questions(self):
        return {
            "deferral_ruling": choice("How should the integrity engineer rule on the deferral request?", {
                "grant_as_requested": "Grant the new date as requested with the interim measures proposed.",
                "grant_with_pressure_restriction": "Grant it only with a pressure restriction deep enough to meet the plan's margin.",
                "grant_shorter_extension": "Grant a shorter extension, covering only the time the cause actually needs.",
                "deny_dig_on_schedule": "Refuse; the excavation goes ahead by the original deadline.",
            }),
            "failure_consequence": score("If this anomaly failed before it was repaired, how serious would the consequences be?", [
                "Limited: open country with nothing inside the impact radius.",
                "Moderate: a few structures nearby, outside any high-consequence area.",
                "High: inside or possibly inside a high-consequence area.",
                "Severe: a densely built area or many occupied structures in the impact radius.",
            ]),
            "deadline_pressure": score("How close is the response deadline for this anomaly?", [
                "Distant: several months remain to dig on schedule.",
                "Approaching: two to three months remain.",
                "Near: a few weeks remain, barely enough to mobilise.",
                "Imminent: about two weeks or less remain.",
            ]),
            "interim_margin_adequate": noul(
                "The proposed interim measures keep the anomaly's failure pressure above the margin the plan requires during the extension.",
                yes="The failure pressure divided by the restricted operating pressure meets or beats the plan's required ratio.",
                no="There is no restriction, or the ratio at the restricted pressure falls short of the plan's figure."),
            "regulator_notice_required": noul(
                "The plan requires the regulator to be notified about this extension before the deadline."),
        }


WORKFLOW = Workflow(
    name="pipeline_integrity_dig_decision",
    title="Pipeline integrity dig decisions",
    description="Dig-now, schedule or monitor decisions on in-line-inspection anomalies, repair selection and deadline deferrals under a written integrity plan.",
    domain="energy / pipeline integrity engineering",
    schemas=[MetalLossAnomalyResponse(), DentAssessment(), InteractingCorrosionCluster(), FieldRepairSelection(),
             DigDeferralRequest()],
    notes="Invented operators, lines, tool vendors and plan criteria; failure pressures are computed with the modified-B31G "
          "relation from the stated pipe data, so the numbers in a state are mutually consistent.",
)
