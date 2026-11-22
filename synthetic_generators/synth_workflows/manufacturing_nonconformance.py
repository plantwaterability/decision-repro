"""Manufacturing nonconformance: what a plant's material review board does with parts that miss the drawing.

A contract manufacturer's quality function decides on nonconforming product: use-as-is, rework, repair, scrap
or return to supplier; how far containment must reach; whether a customer must approve or be told; whether a
temporary process deviation may run; and how deep a corrective action must go. Plants, parts, customers and
suppliers are fictional. Purchased material is judged on inspection evidence only: no bill, purchase order,
receipt matching, debit note or payment appears anywhere.
"""
from __future__ import annotations

from collections import Counter
from datetime import timedelta

from .base import (
    ORG_A,
    REAL_NAME_CLASHES,
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
    money,
    near,
    noisy,
    noul,
    person,
    pick,
    pick_w,
    prune,
    rand_date,
    rand_ts,
    regime,
    score,
    show,
    table,
)

CUSTOMER_SUFFIX = {"automotive": ["Motors", "Automotive", "Vehicle Systems"], "aerospace": ["Aerostructures", "Aviation"],
                   "medical": ["Medical Devices", "Surgical"], "industrial": ["Pumps", "Automation", "Hydraulics"],
                   "appliances": ["Appliances", "Home Products"]}


def _name(rng, suffixes):
    """ORG_A prefix + one of `suffixes`, skipping combinations that name a real business (as base.org does)."""
    a, i = rng.choice(ORG_A), rng.randrange(len(suffixes))
    for k in range(len(suffixes)):
        n = f"{a} {suffixes[(i + k) % len(suffixes)]}"
        if n not in REAL_NAME_CLASHES:
            return n
    return n


def _customer(rng, sector):
    return _name(rng, CUSTOMER_SUFFIX[sector])


def _people(rng, n):
    out, seen = [], []
    while len(out) < n:
        p = person(rng)
        if p.split()[0] not in seen:
            seen.append(p.split()[0])
            out.append(p)
    return out


def _plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def _money(x, rng, cur):
    return money(x, rng, cur, style=pick(rng, ["sym", "sym", "code"]))


def _cap(s):
    return s[0].upper() + s[1:] if s else s


DEFECT_CLASS = {1: "minor", 2: "major", 3: "critical"}


def _ordinal(n):
    return f"{n}" + ("th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th"))


# =========================================================================== schema 1
# (part, material, characteristic, nominal, +/- tolerance, sector, feature kind)
DIM_PARTS = [
    ("pump housing", "cast aluminium A356", "bearing bore diameter", 42.000, 0.025, "industrial", "bore"),
    ("gearbox output shaft", "4140 steel", "bearing journal diameter", 25.000, 0.013, "automotive", "shaft"),
    ("hydraulic manifold", "6061-T6 aluminium", "cartridge port depth", 18.50, 0.10, "industrial", "depth"),
    ("access panel bracket", "7075-T6 aluminium", "flange thickness", 3.20, 0.08, "aerospace", "thickness"),
    ("infusion pump chassis", "polycarbonate", "snap-fit wall thickness", 1.60, 0.05, "medical", "moulded"),
    ("steering column bracket", "HSLA steel", "mounting hole position (dia. zone)", 0.0, 0.20, "automotive", "position"),
    ("connector housing", "PBT-GF30", "latch height", 2.40, 0.05, "automotive", "moulded"),
    ("valve spool", "440C stainless steel", "land diameter", 12.000, 0.008, "industrial", "shaft"),
    ("stapler anvil", "17-4PH stainless steel", "forming pocket depth", 0.45, 0.02, "medical", "depth"),
]
# (feature kind, direction) -> ("rework" | "repair" | None, method)
FIX = {("bore", "under"): ("rework", "re-bore to size"), ("bore", "over"): ("repair", "fit a bush and re-bore"),
       ("shaft", "over"): ("rework", "regrind the diameter"), ("shaft", "under"): ("repair", "hard-chrome and regrind"),
       ("depth", "under"): ("rework", "deepen to drawing"), ("depth", "over"): (None, None),
       ("thickness", "over"): ("rework", "machine the face down"), ("thickness", "under"): (None, None),
       ("moulded", "over"): (None, None), ("moulded", "under"): (None, None),
       ("position", "over"): ("repair", "weld-fill and re-drill the hole")}
ENG = {"stack_up_ok": "tolerance stack-up shows the mating part still assembles and functions",
       "needs_test": "fit looks acceptable on paper; a functional test on the assembly is recommended",
       "not_assessed": None,
       "functional_risk": "stack-up shows a risk of loss of fit or reduced service life"}
LAST_GOOD = {"previous_lot_clean": "previous lot passed final inspection on this feature",
             "spc_drift": "SPC chart shows the feature drifting toward the limit over the last three lots",
             "no_spc": "no SPC on this feature; earlier lots were released on sample inspection only"}


class MrbDimensionalDisposition(Schema):
    name = "mrb_dimensional_disposition"
    description = "A lot with an out-of-tolerance dimension before the material review board: use, rework, repair, scrap or return."
    surfaces = {"json": 0.5, "ncr_form": 0.25, "mrb_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        part, mat, char, nom, tol, sector, feat = pick(rng, DIM_PARTS)
        direction = "over" if feat == "position" else pick(rng, ["over", "under"])
        cls = pick_w(rng, {"critical": 2, "major": 5, "minor": 3})
        excess = between(rng, 0.1, 0.8)                     # how far past the limit, as a share of the tolerance
        unc = pick(rng, [0.08, 0.12, 0.2])                  # gauge uncertainty, share of the tolerance
        eng = pick_w(rng, {"stack_up_ok": 3, "needs_test": 3, "not_assessed": 2, "functional_risk": 2})
        authority = "customer" if sector in ("automotive", "aerospace", "medical") and chance(rng, 0.8) else "in_house"
        origin = pick_w(rng, {"in_house": 4, "supplier_machined": 1})
        lot = pick(rng, [60, 120, 250, 400, 800])
        nc_share = between(rng, 0.05, 0.6)
        days_ship, remake = pick(rng, [2, 4, 7, 12, 20, 30]), pick(rng, [5, 10, 15, 25])
        last_good = pick_w(rng, {"previous_lot_clean": 3, "spc_drift": 2, "no_spc": 2})
        repeat = pick(rng, [0, 0, 0, 1, 2, 3])
        if reg == "clear":
            if chance(rng, 0.5):   # a straightforward rework
                part, mat, char, nom, tol, sector, feat = pick(rng, [x for x in DIM_PARTS if x[6] in
                                                                     ("bore", "shaft", "depth", "thickness")])
                direction = {"bore": "under", "shaft": "over", "depth": "under", "thickness": "over"}[feat]
                authority = "in_house" if sector == "industrial" else authority
                cls, excess, eng, days_ship, remake, last_good, repeat = "major", between(rng, 0.3, 0.6), "not_assessed", \
                    pick(rng, [20, 30]), 10, "previous_lot_clean", 0
            else:                  # far out on a critical feature
                cls, excess, eng = "critical", between(rng, 0.8, 1.6), "functional_risk"
                last_good = pick(rng, ["spc_drift", "no_spc"])
        elif reg == "borderline":
            excess = near(rng, unc, unc * 0.6, lo=0.02)      # the reading is barely past the limit
            eng = pick(rng, ["needs_test", "not_assessed", "stack_up_ok"])
            remake = near(rng, days_ship, 4, lo=2, integer=True)
        else:  # conflicting
            k = pick(rng, ["eng_vs_class", "tiny_but_customer", "supplier_but_late"])
            if k == "eng_vs_class":
                cls, eng, authority = "critical", "stack_up_ok", "customer"
            elif k == "tiny_but_customer":
                cls, excess, authority, days_ship = "minor", between(rng, 0.05, 0.2), "customer", pick(rng, [2, 3])
            else:
                origin, days_ship, remake = "supplier_machined", pick(rng, [3, 5]), pick(rng, [20, 30])
            if chance(rng, 0.4):
                eng = "not_assessed"
        if sector in ("industrial", "appliances") and authority == "customer" and chance(rng, 0.5):
            authority = "in_house"
        fix, method = FIX.get((feat, direction), (None, None))
        limit = nom + tol if direction == "over" else nom - tol
        worst = limit + excess * tol if direction == "over" else limit - excess * tol
        dp = 3 if tol < 0.05 else 2
        return {"regime": reg, "part": part, "material": mat, "characteristic": char, "nominal": nom, "tolerance": tol,
                "decimals": dp, "sector": sector, "feature": feat, "direction": direction, "char_class": cls,
                "excess_share_of_tol": round(excess, 3), "gauge_uncertainty_share": unc, "limit": round(limit, dp + 1),
                "worst_reading": round(worst, dp + 1), "engineering": eng, "design_authority": authority,
                "origin": origin, "lot_qty": lot, "nc_qty": max(1, int(lot * nc_share)),
                "fix_route": fix, "fix_method": method, "days_to_ship": days_ship, "remake_days": remake,
                "last_good": last_good, "repeats_90d": repeat, "customer": _customer(rng, sector),
                "supplier": (_name(rng, ["Moulding", "Plastics"] if feat == "moulded" else ["Precision", "Machining",
                                                                                              "Engineering"])
                             if origin != "in_house" else None),
                "part_cost": pick(rng, [18, 45, 120, 380, 900]), "currency": pick(rng, ["USD", "EUR", "GBP"]),
                "ncr": ident(rng, "NCR", 6, "-"), "part_no": f"{rng.randint(100, 999)}-{rng.randint(1000, 9999)}",
                "raised_by": None, "chair": None, "date": rand_date(rng)} | \
            dict(zip(("raised_by", "chair"), _people(rng, 2)))

    def _spec(self, f):
        d = f["decimals"]
        if f["feature"] == "position":
            return f"{f['tolerance']:.{d}f} mm max"
        return f"{f['nominal']:.{d}f} +/- {f['tolerance']:.{d}f} mm"

    def _reading(self, f):
        d = f["decimals"] + 1
        return f"{f['worst_reading']:.{d}f} mm"

    def _fix_text(self, f):
        if f["fix_route"] == "rework":
            return f"rework possible: {f['fix_method']}"
        if f["fix_route"] == "repair":
            return f"no rework to drawing; a repair is possible ({f['fix_method']})"
        return "no rework or repair route exists for this feature"

    def render(self, f, rng, surface):
        cost = _money(f["part_cost"], rng, f["currency"])
        unc = round(f["gauge_uncertainty_share"] * f["tolerance"], f["decimals"] + 1)
        verb = "moulded" if f["feature"] == "moulded" else "machined"
        src = (f"{verb} by supplier {f['supplier']}" if f["origin"] != "in_house" else f"{verb} in-house")
        auth = ("customer-owned drawing" if f["design_authority"] == "customer" else "our own design")
        auth_s = "the customer's" if f["design_authority"] == "customer" else "ours"
        if surface == "json":
            return dumps(prune({
                "ncr": {"id": f["ncr"], "raised": fmt_date(f["date"], style="iso"), "raised_by": f["raised_by"]},
                "part": {"number": f["part_no"], "name": f["part"], "material": f["material"], "customer": f["customer"],
                         "sector": f["sector"], "design_authority": auth, "source": src, "unit_cost": cost},
                "nonconformance": {"characteristic": f["characteristic"], "classification": f["char_class"],
                                   "specification": self._spec(f), "worst_reading": self._reading(f),
                                   "gauge_uncertainty": f"+/- {unc} mm", "nonconforming_qty": f["nc_qty"],
                                   "lot_qty": f["lot_qty"]},
                "assessment": {"engineering": ENG[f["engineering"]], "fix_options": self._fix_text(f),
                               "earlier_production": LAST_GOOD[f["last_good"]],
                               "same_issue_last_90_days": f["repeats_90d"]},
                "schedule": {"customer_ship_in_days": f["days_to_ship"], "remake_lead_time_days": f["remake_days"]},
            }))
        if surface == "ncr_form":
            rows = [("NCR", f["ncr"]), ("Date", fmt_date(f["date"], rng)), ("Raised by", f["raised_by"]),
                    ("Part", f"{f['part_no']} {f['part']} ({f['material']})"), ("Customer", f["customer"]),
                    ("Drawing", auth), ("Source", src), ("Characteristic", f"{f['characteristic']} [{f['char_class']}]"),
                    ("Spec", self._spec(f)), ("Worst reading", self._reading(f)), ("Gauge uncertainty", f"+/- {unc} mm"),
                    ("Qty", f"{f['nc_qty']} of {f['lot_qty']} nonconforming"),
                    ("Engineering review", show(ENG[f["engineering"]], rng)),
                    ("Fix options", self._fix_text(f)), ("Earlier lots", LAST_GOOD[f["last_good"]]),
                    ("Repeats (90 d)", f["repeats_90d"]),
                    ("Ship date", f"in {_plural(f['days_to_ship'], 'day')}; remake takes {f['remake_days']} days"),
                    ("Unit cost", cost)]
            return "NONCONFORMANCE REPORT\n" + kv_lines(rows, rng, style=pick(rng, ["colon", "aligned", "dash"]),
                                                         skip_none=False)
        s = [f"{f['ncr']} is on tomorrow's MRB list.",
             f"{f['nc_qty']} of {f['lot_qty']} {f['part']}s ({f['part_no']}, {f['customer']}) are out on the "
             f"{f['characteristic']}, a {f['char_class']} characteristic: spec {self._spec(f)}, worst reading "
             f"{self._reading(f)}, gauge uncertainty about +/- {unc} mm.",
             f"The parts were {src}; the drawing is {auth_s}.",
             (f"Engineering says the {ENG[f['engineering']]}." if ENG[f["engineering"]]
              else "Engineering has not looked at it yet."),
             _cap(self._fix_text(f)) + ".", _cap(LAST_GOOD[f["last_good"]]) + ".",
             (f"This is the {_ordinal(f['repeats_90d'] + 1)} time in 90 days we've seen this." if f["repeats_90d"] >= 2 else
              "We had one similar NCR last quarter." if f["repeats_90d"] == 1 else ""),
             f"Customer shipment is due in {_plural(f['days_to_ship'], 'day')} and a remake would take "
             f"{f['remake_days']} days."]
        return "\n".join([f"Subject: MRB -- {f['ncr']} {f['part']}", f"From: {f['raised_by']}", f"To: {f['chair']}", "",
                          f"{f['chair'].split()[0]},", "", noisy(join_sentences(s), rng, 0.12), "",
                          f["raised_by"].split()[0]])

    def questions(self):
        return {
            "mrb_disposition": choice("How should the material review board disposition the nonconforming parts?", {
                "repair_under_concession": "Apply a non-standard repair that leaves the parts usable but not to drawing, "
                                           "under a written concession.",
                "return_to_supplier": "Send the parts back to the supplier that made them.",
                "rework_to_drawing": "Rework the parts so that they fully meet the drawing, then re-inspect them.",
                "scrap": "Scrap the parts and make replacements.",
                "use_as_is": "Accept the parts as they are, with the deviation recorded.",
            }),
            "containment_reach": choice("How far should containment reach beyond the parts in hand?", {
                "all_stock_and_wip": "Every part in stock and in work in progress, whatever lot it came from.",
                "including_shipped_product": "Stock, work in progress and parts already shipped to the customer.",
                "same_run_other_lots": "Other lots made on the same set-up or run.",
                "this_lot_only": "Only the lot in which the problem was found.",
            }),
            "functional_effect": score("How much could the deviation affect the part's fit, function or service life?", [
                "None: well inside what the assembly can tolerate.",
                "Slight: a small loss of margin with no practical effect.",
                "Real: fit or service life could suffer in some assemblies.",
                "Serious: parts could fail to fit or fail in service.",
            ]),
            "schedule_squeeze": score("How much does the customer delivery date constrain the board's options?", [
                "Not at all: a remake would finish 5 or more days before the ship date.",
                "Slightly: a remake would finish within 4 days either side of the ship date, so overtime can close "
                "the gap.",
                "Strongly: a remake would miss the ship date by more than 4 days, and shipment is more than 3 days "
                "away.",
                "Completely: the parts ship in 3 days or less whatever is decided.",
            ]),
            "customer_approval_needed": noul(
                "The customer must approve the disposition before these parts can ship.",
                yes="The drawing is the customer's and no rework route brings the parts back to drawing, so any part "
                    "shipped would carry a departure from it.",
                no="The design is ours, or a rework route can bring the parts back to the drawing before they ship."),
        }


# =========================================================================== schema 2
# ANSI/ISO-style single sampling, general level II, AQL 1.0: (max lot size, sample size, accept, reject)
AQL = [(500, 50, 1, 2), (1200, 80, 2, 3), (3200, 125, 3, 4), (10000, 200, 5, 6), (35000, 315, 7, 8)]
MATERIALS = [
    ("M8x25 grade 8.8 hex bolts", "pcs", ["hardness_low", "plating_thin", "cert_mismatch", "thread_out"], "Fasteners"),
    ("6061-T6 aluminium bar, 50 mm", "bars", ["cert_mismatch", "hardness_low", "dimension_out"], "Metals"),
    ("controller PCB assemblies", "boards", ["wrong_revision", "solder_bridges", "cosmetic_scratches"], "Electronics"),
    ("EPDM door gaskets", "pcs", ["dimension_out", "hardness_low", "moulding_flash"], "Rubber"),
    ("zinc-plated mounting brackets", "pcs", ["plating_thin", "dimension_out", "cosmetic_scratches"], "Pressings"),
    ("PA66-GF30 resin, 25 kg bags", "bags", ["moisture_high", "cert_mismatch", "contamination"], "Polymers"),
]
# how a 100% sort would go, per defect: quick (visual / go-no-go), slow (a measurement per piece), or impossible
SORTABLE = {"hardness_low": "slow", "plating_thin": "slow", "cert_mismatch": "not_possible", "thread_out": "quick",
            "dimension_out": "quick", "wrong_revision": "quick", "solder_bridges": "slow", "cosmetic_scratches": "quick",
            "moulding_flash": "quick", "moisture_high": "not_possible", "contamination": "not_possible"}
# defect -> (description, seriousness 1..3, what we could do in-house)
RECV_DEFECTS = {
    "hardness_low": ("hardness below the specified minimum", 3, None),
    "plating_thin": ("zinc plating thinner than specified", 2, None),
    "cert_mismatch": ("heat or batch number on the material certificate does not match the goods", 2, "retest"),
    "thread_out": ("thread gauge failures (no-go gauge enters)", 2, None),
    "dimension_out": ("a controlled dimension out of tolerance", 2, None),
    "wrong_revision": ("boards built to revision B instead of revision C", 2, "rework"),
    "solder_bridges": ("solder bridges on connector pins", 3, "rework"),
    "cosmetic_scratches": ("scratches on visible faces", 1, "rework"),
    "moulding_flash": ("moulding flash on the sealing lip", 2, "rework"),
    "moisture_high": ("moisture content above the processing limit", 1, "rework"),
    "contamination": ("off-colour foreign pellets mixed into the bags", 2, None),
}
INHOUSE = {"wrong_revision": "rework the boards to revision C on our SMT line",
           "solder_bridges": "touch up the joints at our rework station",
           "cosmetic_scratches": "polish out scratches at the finishing bench",
           "moulding_flash": "trim the flash by hand", "moisture_high": "dry the resin for four hours before use",
           "cert_mismatch": "send samples to the lab to verify the material grade"}
MEASURE = {"second_inspector": "confirmed by a second inspector on a calibrated gauge",
           "single_check": "single inspector, one measurement pass",
           "gauge_overdue": "the gauge used is past its calibration date",
           "supplier_disagrees": "the supplier's outgoing inspection report shows the whole lot conforming"}


class ReceivingInspectionReject(Schema):
    name = "receiving_inspection_reject"
    description = "Purchased material that failed receiving inspection: accept on deviation, retest, sort, rework or return."
    surfaces = {"json": 0.45, "inspection_report": 0.3, "planner_chat": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        mat, unit, defects, sup_kind = pick(rng, MATERIALS)
        defect = pick(rng, defects)
        lot = pick(rng, [400, 900, 2500, 6000, 20000])
        _mx, n, ac, re = next(r for r in AQL if lot <= r[0])
        found = re + pick(rng, [0, 0, 1, 2, 5])
        measure = pick_w(rng, {"second_inspector": 3, "single_check": 3, "gauge_overdue": 1, "supplier_disagrees": 2})
        cover_days = pick(rng, [0, 1, 3, 7, 14, 30])        # days until the line runs short without this lot
        status = pick_w(rng, {"approved": 5, "conditional": 2, "new": 2})
        prior = pick(rng, [0, 0, 1, 2, 4])
        offer = pick_w(rng, {"none": 3, "sort_at_our_site": 2, "disputes": 2})
        replace_days = pick(rng, [3, 7, 14, 21])            # supplier lead time for a replacement lot
        if reg == "clear":
            if chance(rng, 0.5):   # decisive, serious and plenty of stock: send it back
                defect = pick(rng, [d for d in defects if RECV_DEFECTS[d][1] >= 2] or defects)
                found, measure, cover_days, replace_days, prior = re + pick(rng, [6, 10, 15]), "second_inspector", 30, \
                    7, pick(rng, [2, 4])
            else:                  # trivial, in-house fixable, line waiting
                defect = pick(rng, [d for d in defects if RECV_DEFECTS[d][2] == "rework"] or defects)
                found, measure, cover_days, replace_days, prior = re, "second_inspector", 1, 14, 0
        elif reg == "borderline":
            found = re                                      # rejected by a single piece
            cover_days = near(rng, replace_days, 3, lo=0, integer=True)
            measure = pick(rng, ["single_check", "second_inspector", "gauge_overdue"])
        else:  # conflicting
            if chance(rng, 0.5):   # heavy failure, but the measurement is in dispute
                found, measure, offer = re + pick(rng, [4, 8]), pick(rng, ["supplier_disagrees", "gauge_overdue"]), \
                    "disputes"
            else:                  # serious defect, the line is about to stop
                defect = pick(rng, [d for d in defects if RECV_DEFECTS[d][1] == 3] or defects)
                cover_days, replace_days = pick(rng, [0, 1]), pick(rng, [14, 21])
        sort = SORTABLE[defect]
        if offer == "sort_at_our_site" and sort == "not_possible":
            offer = "none"
        return {"regime": reg, "material": mat, "unit": unit, "defect": defect, "lot_qty": lot, "sample_size": n,
                "accept_no": ac, "reject_no": re, "found": found, "measurement": measure,
                "days_until_line_short": cover_days, "sort_feasibility": sort, "supplier_status": status,
                "rejections_12m": prior, "supplier_offer": offer, "replacement_days": replace_days,
                "supplier": _name(rng, [sup_kind]),
                "receipt": ident(rng, "RI", 6, "-"), "inspector": None, "planner": None, "date": rand_date(rng)} | \
            dict(zip(("inspector", "planner"), _people(rng, 2)))

    def _offer_text(self, f):
        return {"none": None, "sort_at_our_site": "supplier offers to send two sorters to our site",
                "disputes": "supplier disputes our result and asks for the parts to be re-measured"}[f["supplier_offer"]]

    def _supply_text(self, f):
        short = ("the line is already short of this material" if f["days_until_line_short"] == 0 else
                 f"without this lot the line runs short in {_plural(f['days_until_line_short'], 'day')}")
        return f"{short}; a replacement lot takes {f['replacement_days']} days from the supplier"

    def _sort_text(self, f):
        return {"quick": "a 100% sort is quick (visual or go/no-go)", "slow": "a 100% sort is possible but slow",
                "not_possible": "a piece-by-piece sort cannot catch this defect"}[f["sort_feasibility"]]

    def render(self, f, rng, surface):
        d = RECV_DEFECTS[f["defect"]][0]
        inh = INHOUSE.get(f["defect"])
        if surface == "json":
            return dumps(prune({
                "receipt": {"id": f["receipt"], "date": fmt_date(f["date"], style="iso"), "material": f["material"],
                            "supplier": f["supplier"], "lot_qty": f["lot_qty"], "unit": f["unit"]},
                "inspection": {"defect": d, "defect_class": DEFECT_CLASS[RECV_DEFECTS[f["defect"]][1]],
                               "sample_size": f["sample_size"], "accept_number": f["accept_no"],
                               "reject_number": f["reject_no"], "nonconforming_in_sample": f["found"],
                               "measurement": MEASURE[f["measurement"]]},
                "options": {"sorting": self._sort_text(f), "in_house_fix": inh,
                            "supplier_offer": self._offer_text(f)},
                "supplier_record": {"status": f["supplier_status"], "rejections_last_12_months": f["rejections_12m"]},
                "production": {"days_until_line_runs_short": f["days_until_line_short"],
                               "replacement_lot_lead_time_days": f["replacement_days"]},
            }))
        if surface == "inspection_report":
            rows = [("Receipt", f["receipt"]), ("Date", fmt_date(f["date"], rng)), ("Supplier", f"{f['supplier']} "
                    f"({f['supplier_status']}, {_plural(f['rejections_12m'], 'rejection')} in 12 months)"),
                    ("Material", f"{f['material']}, lot of {f['lot_qty']:,} {f['unit']}"),
                    ("Plan", f"n={f['sample_size']}, Ac={f['accept_no']}, Re={f['reject_no']}"),
                    ("Found", f"{f['found']} nonconforming"),
                    ("Defect", f"{d} ({DEFECT_CLASS[RECV_DEFECTS[f['defect']][1]]} defect)"),
                    ("Measurement", MEASURE[f["measurement"]]), ("Inspector", f["inspector"])]
            notes = bullets([_cap(self._sort_text(f)) + ".", (f"In-house option: {inh}." if inh else None),
                             (_cap(self._offer_text(f)) + ".") if self._offer_text(f) else None,
                             f"Planning: {self._supply_text(f)}."], rng)
            return "RECEIVING INSPECTION -- REJECT\n" + kv_lines(rows, rng, style=pick(rng, ["colon", "aligned"])) + \
                "\n\n" + notes
        i, p = f["inspector"].split()[0].lower(), f["planner"].split()[0].lower()
        msgs = [(i, f"{f['receipt']} from {f['supplier']} failed incoming: {d} "
                    f"({DEFECT_CLASS[RECV_DEFECTS[f['defect']][1]]} defect), {f['found']} in a sample of "
                    f"{f['sample_size']} (reject at {f['reject_no']})"),
                (p, ("ugh. " if f["days_until_line_short"] < f["replacement_days"] else "ok, ")
                    + ("we're already out on the line" if f["days_until_line_short"] == 0 else
                               f"we run short in {_plural(f['days_until_line_short'], 'day')} without it")
                    + f", and a replacement is {f['replacement_days']} days out"),
                (i, {"second_inspector": "double-checked with a second inspector, it's real",
                     "single_check": "measured once so far",
                     "gauge_overdue": "heads up, the gauge I used is past its cal date",
                     "supplier_disagrees": "their outgoing report says the whole lot passed"}[f["measurement"]]),
                (p, "can we sort it?"), (i, self._sort_text(f))]
        if inh:
            msgs.append((i, f"we could {inh}"))
        if self._offer_text(f):
            msgs.append((p, self._offer_text(f)))
        if f["rejections_12m"] >= 2:
            msgs.append((i, f"fyi that's their {_ordinal(f['rejections_12m'] + 1)} reject this year"))
        t = rng.randint(7, 15) * 60
        out = []
        for who, txt in msgs:
            t += rng.randint(1, 12)
            out.append(f"[{t // 60:02d}:{t % 60:02d}] {who}: {noisy(txt, rng, 0.1)}")
        return f"#receiving-quality  {fmt_date(f['date'], rng)}\n" + "\n".join(out)

    def questions(self):
        return {
            "receiving_decision": choice("What should quality do with the rejected lot?", {
                "accept_on_deviation": "Release the lot to production as it is, under a recorded deviation.",
                "hold_for_retest": "Keep the lot on hold until a re-measurement or lab test settles the result.",
                "return_lot_to_supplier": "Send the whole lot back to the supplier for replacement.",
                "rework_in_house": "Fix the material ourselves, then re-inspect it before use.",
                "sort_and_use_conforming": "Inspect every piece, use the good ones and set the bad ones aside.",
            }),
            "lot_condition": score("How poor is the lot as a whole, judging by the inspection result?", [
                "Marginal: exactly at the reject number with a minor defect, or a certificate mismatch only.",
                "Poor: at the reject number with a major defect, or one or two over it with a minor defect.",
                "Bad: at the reject number with a critical defect, one or two over with a major defect, or three or "
                "more over with a minor defect.",
                "Unusable: one or two over the reject number with a critical defect, or three or more over with a "
                "major or critical defect.",
            ]),
            "production_pressure": score("How hard is production pressing for this material?", [
                "Not at all: stock lasts well beyond the time a replacement lot takes.",
                "Somewhat: stock lasts about as long as a replacement lot takes.",
                "Strongly: the line runs short well before a replacement could arrive.",
                "Line-down: the line stops within a day without this lot.",
            ]),
            "result_reliable": noul(
                "The inspection result is reliable enough to act on.",
                yes="The measurement was checked on calibrated equipment and nothing credible contradicts it.",
                no="The gauge, the method or a conflicting measurement leaves the result open to doubt."),
            "corrective_request_warranted": noul(
                "The supplier should receive a formal corrective-action request over this lot.",
                yes="The result is sound (no overdue gauge, no conflicting supplier report) and this is at least the "
                    "supplier's third rejection in 12 months, or the sample is three or more over the reject number.",
                no="The result is in doubt, or it is a first or second rejection that sits within two pieces of the "
                   "reject number."),
        }


# =========================================================================== schema 3
# (part, sector, what the part does, issues that can escape on it)
ESCAPE_PARTS = [
    ("brake caliper bracket", "automotive", "safety", ["the heat-treat step was skipped for one furnace load",
                                                        "mounting holes drilled oversize after a tool change"]),
    ("seat-rail slider", "automotive", "safety", ["fillet welds undersized on one fixture",
                                                   "a missing weld on one of four studs"]),
    ("fuel-pump housing", "automotive", "function", ["thread depth short on one spindle",
                                                      "sealing-face flatness out of tolerance"]),
    ("cabin-seat pivot pin", "aerospace", "safety", ["hardness below specification after a furnace fault",
                                                      "the heat-treat step was skipped for one furnace load"]),
    ("infusion pump door latch", "medical", "safety", ["latch spring force below specification",
                                                        "a latch spring missing from some assemblies"]),
    ("dishwasher pump impeller", "appliances", "function", ["an out-of-balance condition after a gate change",
                                                             "a hub bore moulded undersize after a tool change"]),
    ("control-panel bezel", "industrial", "cosmetic", ["sink marks on the visible face",
                                                        "colour outside the approved range"]),
    ("fan hub", "industrial", "function", ["a missing thread-locking patch",
                                           "a bore machined undersize after a tool change"]),
]
FOUND_VIA = {"later_lot": "found while inspecting a later lot from the same set-up",
             "process_audit": "found in a process audit that uncovered an unrecorded tool change",
             "operator_report": "reported by an operator who had dropped the gauge used for release checks",
             "missing_records": "final-inspection records are missing for two shifts"}
ROLE_TEXT = {"safety": "safety-critical part", "function": "functional part (not safety-critical)",
             "cosmetic": "cosmetic part (appearance only)"}
AGREEMENT = {"notify_any_suspect": "notify the customer within 24 hours of any suspect product shipped",
             "notify_key_only": "notify the customer when a designated key characteristic is affected",
             "not_located": None}


class EscapedNonconformance(Schema):
    name = "escaped_nonconformance"
    description = "A nonconformance found after parts have already shipped: whether and how to tell the customer."
    surfaces = {"json": 0.5, "escalation_email": 0.25, "ncr_timeline": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        part, sector, role, issues = pick(rng, ESCAPE_PARTS)
        issue = pick(rng, issues)
        found_via = pick(rng, list(FOUND_VIA))
        shipped = pick(rng, [120, 400, 1500, 4000, 9000])
        stock = pick(rng, [0, 200, 800, 2000])
        tested = pick(rng, [0, 0, 20, 30, 50])
        failed = 0 if tested == 0 else pick(rng, [0, 0, 1, 2, 5])
        cust_days = pick(rng, [0, 1, 3, 7, 15])                  # days of stock the customer holds before assembly
        agreement = pick_w(rng, {"notify_any_suspect": 4, "notify_key_only": 3, "not_located": 1})
        key_char = chance(rng, 0.5 if role != "cosmetic" else 0.1)
        scorecard = pick_w(rng, {"green": 3, "yellow": 3, "red": 1})
        sales = pick(rng, [None, None, "they'll put us on controlled shipping if we report this",
                           "the account manager asks whether we can just replace them quietly at the next delivery",
                           "the customer is renewing the programme next month"])
        if reg == "clear":
            if chance(rng, 0.5):
                part, sector, role, issues = pick(rng, [x for x in ESCAPE_PARTS if x[2] == "safety"])
                issue = pick(rng, issues)
                tested, failed, agreement, key_char = 30, pick(rng, [4, 7]), "notify_any_suspect", True
            else:
                part, sector, role, issues = ESCAPE_PARTS[6]
                issue = pick(rng, issues)
                tested, failed, agreement, key_char, shipped = 50, 0, "notify_key_only", False, 120
        elif reg == "borderline":
            tested, failed = pick(rng, [20, 30]), pick(rng, [0, 1])
            agreement = pick(rng, ["notify_key_only", "not_located"])
        else:  # conflicting
            if chance(rng, 0.5):   # no failures found in testing, but a safety part and a strict agreement
                part, sector, role, issues = pick(rng, [x for x in ESCAPE_PARTS if x[2] == "safety"])
                issue = pick(rng, issues)
                tested, failed, agreement = pick(rng, [20, 50]), 0, "notify_any_suspect"
            else:                  # clear failures, but on a cosmetic feature with the relationship under strain
                part, sector, role, issues = ESCAPE_PARTS[6]
                issue = pick(rng, issues)
                tested, failed, scorecard = 30, pick(rng, [6, 9]), "red"
                sales = "they'll put us on controlled shipping if we report this"
        return {"regime": reg, "part": part, "sector": sector, "part_role": role, "issue": issue, "found_via": found_via,
                "qty_shipped": shipped, "qty_in_stock": stock, "retained_tested": tested, "retained_failed": failed,
                "customer_stock_days": cust_days, "agreement": agreement, "key_characteristic": key_char,
                "scorecard": scorecard, "sales_comment": sales, "customer": _customer(rng, sector),
                "days_since_found": pick(rng, [0, 1, 2, 4]), "ncr": ident(rng, "NCR", 6, "-"),
                "author": None, "manager": None, "found": rand_ts(rng, hours=(6, 22))} | \
            dict(zip(("author", "manager"), _people(rng, 2)))

    def _test_text(self, f):
        if not f["retained_tested"]:
            return "no retained samples tested yet"
        return f"{f['retained_failed']} of {f['retained_tested']} retained samples from the shipped lots failed"

    def _cust_text(self, f):
        if f["customer_stock_days"] == 0:
            return "customer builds just-in-time; parts are normally fitted the day they arrive"
        return f"customer holds about {_plural(f['customer_stock_days'], 'day')} of our parts before assembly"

    def render(self, f, rng, surface):
        agr = AGREEMENT[f["agreement"]]
        if surface == "json":
            return dumps(prune({
                "ncr": {"id": f["ncr"], "part": f["part"], "part_function": ROLE_TEXT[f["part_role"]],
                        "customer": f["customer"], "sector": f["sector"],
                        "issue": f["issue"], "how_found": FOUND_VIA[f["found_via"]],
                        "days_since_found": f["days_since_found"]},
                "exposure": {"suspect_qty_shipped": f["qty_shipped"], "suspect_qty_in_our_stock": f["qty_in_stock"],
                             "retained_sample_result": self._test_text(f), "customer_pipeline": self._cust_text(f)},
                "customer_terms": {"quality_agreement": agr or "agreement not located in the contract file",
                                   "designated_key_characteristic": f["key_characteristic"],
                                   "our_scorecard_status": f["scorecard"]},
                "notes": {"sales": f["sales_comment"]},
            }))
        if surface == "escalation_email":
            s = [f"We have a possible escape on the {f['part']} ({ROLE_TEXT[f['part_role']]}) for {f['customer']} "
                 f"({f['ncr']}): {f['issue']}.",
                 _cap(FOUND_VIA[f["found_via"]]) + ".",
                 f"Suspect parts: {f['qty_shipped']:,} shipped, {f['qty_in_stock']:,} still with us.",
                 _cap(self._test_text(f)) + ".", _cap(self._cust_text(f)) + ".",
                 (f"The quality agreement says we must {agr}." if agr else
                  "I can't find the quality agreement in the contract file."),
                 ("The feature is a designated key characteristic." if f["key_characteristic"] else
                  "The feature is not on their key-characteristic list."),
                 f"We are {f['scorecard']} on their supplier scorecard."]
            if f["sales_comment"]:
                s.append(f"Sales mentioned {f['sales_comment']}.")
            return "\n".join([f"Subject: Possible escape -- {f['part']} / {f['customer']}", f"From: {f['author']}",
                              f"To: {f['manager']}", "", noisy(join_sentences(s), rng, 0.12), "",
                              f["author"].split()[0]])
        t0 = f["found"]
        ev = [(t0, f"{FOUND_VIA[f['found_via']]}: {f['issue']}"),
              (t0 + timedelta(minutes=rng.randint(20, 90)), f"stock quarantined: {f['qty_in_stock']:,} pcs; "
                                                             f"shipped since last good check: {f['qty_shipped']:,} pcs"),
              (t0 + timedelta(hours=rng.randint(3, 20)), self._test_text(f)),
              (t0 + timedelta(hours=rng.randint(21, 30)), "contract file checked: " +
               (agr if agr else "quality agreement not located")),
              (t0 + timedelta(hours=rng.randint(31, 40)), self._cust_text(f))]
        if f["sales_comment"]:
            ev.append((t0 + timedelta(hours=rng.randint(41, 50)), f"sales: {f['sales_comment']}"))
        log = "\n".join(f"{fmt_ts(t, style='log')}  {noisy(e, rng, 0.05)}" for t, e in ev)
        head = (f"{f['ncr']}  {f['part']} ({ROLE_TEXT[f['part_role']]})  customer: {f['customer']}  key characteristic: "
                f"{'yes' if f['key_characteristic'] else 'no'}  scorecard: {f['scorecard']}")
        return head + "\n" + log

    def questions(self):
        return {
            "customer_notification": choice("How should the plant handle telling the customer about the escape?", {
                "formal_notice_with_sort_team": "Formal suspect-product notice plus a sort team at the customer's "
                                                "site: notice is required (the agreement covers it, or retained "
                                                "samples of a safety-critical part failed), samples failed, and the "
                                                "customer holds more than a day of our parts.",
                "formal_suspect_notice": "Formal written suspect-product notice: notice is required, but no retained "
                                         "sample failed or the customer holds a day or less of stock.",
                "informal_call": "Informal heads-up by phone: no notice is required, but retained samples failed.",
                "internal_record_only": "Internal record only: no notice is required and no retained sample failed.",
            }),
            "customer_side_containment": choice("What containment should be sought at the customer's end?", {
                "none_at_customer": "Nothing at the customer's end: no retained sample failed, and the part is "
                                    "neither safety-critical nor a key characteristic.",
                "quarantine_their_stock": "Ask the customer to set aside the suspect lots: no retained sample "
                                          "failed, but the part is safety-critical or a key characteristic.",
                "sort_their_stock_on_site": "Sort at the customer's site: retained samples failed and the customer "
                                            "holds more than a day of our parts.",
                "trace_into_built_products": "Trace into built assemblies: retained samples failed and the customer "
                                             "holds a day or less of stock, so suspect parts are already fitted.",
            }),
            "escape_likelihood": score("How likely is it that nonconforming parts actually reached the customer?", [
                "Very unlikely: tests of shipped lots found nothing and the cause came later.",
                "Possible: no failures found, but the testing was limited.",
                "Likely: failures turned up among retained samples of shipped lots.",
                "Near certain: a substantial share of shipped parts are affected.",
            ]),
            "worst_downstream_effect": score("What is the worst credible consequence at the customer's end or beyond?", [
                "Cosmetic: a look-and-feel complaint at most.",
                "Nuisance: assembly trouble or rework at the customer.",
                "Functional: products that fail or wear out early in use.",
                "Safety: a failure that could injure someone.",
                "Severe safety: a failure that could kill, or ground or recall a fleet of products.",
            ]),
            "notice_contractually_required": noul(
                "The quality agreement obliges the plant to tell the customer about this.",
                yes="The agreement's notification clause clearly covers this kind of suspect product.",
                no="The clause does not cover this feature, or no agreement can be found to require it."),
        }


# =========================================================================== schema 4
DEVIATIONS = {
    "alt_resin": ("use a second-source resin grade that is not yet qualified", "the approved resin is on allocation"),
    "backup_machine": ("run the part on a backup CNC with no capability study", "the main machining centre is down "
                       "for a spindle rebuild"),
    "sampled_leak_test": ("replace 100% leak testing with 1-in-20 sampling", "the leak tester is waiting for spare parts"),
    "cure_window": ("extend the adhesive open time before assembly from 24 h to 36 h", "assembly is a shift behind"),
    "regrind_limit": ("raise regrind content from 15% to 25%", "the virgin material allocation was cut"),
    "alt_plater": ("send parts to a plating subcontractor not on the approved list", "the approved plater closed "
                   "after a fire"),
}
DEV_PARTS = {"alt_resin": [("connector housing", "automotive"), ("infusion set clamp", "medical"),
                           ("cabin trim clip", "aerospace")],
             "regrind_limit": [("connector housing", "automotive"), ("cabin trim clip", "aerospace"),
                               ("pump impeller", "industrial")],
             "backup_machine": [("sensor bracket", "automotive"), ("valve body", "industrial"),
                                ("bone-screw driver shaft", "medical")],
             "sampled_leak_test": [("valve body", "industrial"), ("coolant pump housing", "automotive"),
                                   ("infusion set drip chamber", "medical")],
             "cure_window": [("sensor bracket", "automotive"), ("cabin trim clip", "aerospace"),
                             ("control-panel overlay", "industrial")],
             "alt_plater": [("sensor bracket", "automotive"), ("valve body", "industrial"),
                            ("seat-track bracket", "aerospace")]}
EVIDENCE = {"none": "no comparison data yet", "small_sample": "a small trial: {n} parts made and all passed inspection",
            "capability": "a capability study on {n} parts: Cpk {cpk} (requirement 1.33)",
            "full": "a full equivalence study: dimensions, material tests and a {n}-part capability run (Cpk {cpk})"}


class ProcessDeviationRequest(Schema):
    name = "process_deviation_request"
    description = "A request to run production temporarily outside the approved process: approve, limit, seek consent or refuse."
    surfaces = {"json": 0.5, "deviation_form": 0.25, "review_thread": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        dkey = pick(rng, list(DEVIATIONS))
        part, sector = pick(rng, DEV_PARTS[dkey])
        weeks = pick(rng, [1, 2, 4, 6, 10])
        qty = pick(rng, [800, 2500, 6000, 15000])
        evidence = pick_w(rng, {"none": 2, "small_sample": 3, "capability": 4, "full": 1})
        cpk = between(rng, 1.0, 1.8)
        n = {"none": 0, "small_sample": pick(rng, [5, 10, 12]), "capability": pick(rng, [30, 50]), "full": 125}[evidence]
        cls = pick_w(rng, {"critical": 2, "major": 5, "minor": 3})
        extensions = pick(rng, [0, 0, 1, 2, 4])
        check = pick_w(rng, {"100pct_inspection": 3, "extra_sampling": 3, "none": 2})
        consent_rule = "required" if sector in ("automotive", "aerospace", "medical") else "internal"
        if reg == "clear":
            if chance(rng, 0.5):
                evidence, n, cpk, cls, extensions, check, weeks = "full", 125, between(rng, 1.6, 2.0), "minor", 0, \
                    "extra_sampling", 2
            else:
                evidence, n, cls, extensions, check, weeks = "none", 0, "critical", pick(rng, [3, 4]), "none", 10
        elif reg == "borderline":
            evidence, n = "capability", pick(rng, [30, 50])
            cpk = near(rng, 1.33, 0.15, lo=0.8)
            extensions = pick(rng, [0, 1, 2])
        else:  # conflicting: strong data, but a critical feature and repeated extensions; or no data, trivial feature
            if chance(rng, 0.5):
                evidence, n, cpk, cls, extensions = "full", 125, between(rng, 1.5, 1.9), "critical", pick(rng, [3, 4])
            else:
                evidence, n, cls, check, weeks = "none", 0, "minor", "100pct_inspection", 1
        if dkey == "sampled_leak_test" and check == "100pct_inspection":
            check = "extra_sampling"
        return {"regime": reg, "deviation": dkey, "part": part, "sector": sector, "weeks": weeks, "qty": qty,
                "evidence": evidence, "evidence_n": n, "cpk": round(cpk, 2), "char_class": cls,
                "prior_extensions": extensions, "added_check": check, "consent_rule": consent_rule,
                "customer": _customer(rng, sector), "request_id": ident(rng, "DEV", 4, "-"),
                "requester": None, "reviewer": None, "date": rand_date(rng),
                "tone": pick(rng, ["neutral", "neutral", "pushy"])} | \
            dict(zip(("requester", "reviewer"), _people(rng, 2)))

    def _evidence_text(self, f):
        return EVIDENCE[f["evidence"]].format(n=f["evidence_n"], cpk=f["cpk"])

    def _check_text(self, f):
        return {"100pct_inspection": "100% inspection of the affected feature while the deviation runs",
                "extra_sampling": "doubled sampling frequency at final inspection",
                "none": None}[f["added_check"]]

    def _rule_text(self, f):
        return ("the customer's supplier manual requires written approval for process changes"
                if f["consent_rule"] == "required" else "the part is our own design; process changes are approved "
                                                        "internally")

    def render(self, f, rng, surface):
        what, why = DEVIATIONS[f["deviation"]]
        if surface == "json":
            return dumps(prune({
                "request": {"id": f["request_id"], "raised": fmt_date(f["date"], style="iso"),
                            "requested_by": f["requester"], "change": what, "reason": why,
                            "duration_weeks": f["weeks"], "parts_affected": f["qty"]},
                "part": {"name": f["part"], "customer": f["customer"], "sector": f["sector"],
                         "affected_characteristic_class": f["char_class"]},
                "evidence": {"equivalence": self._evidence_text(f), "added_containment": self._check_text(f)},
                "history": {"previous_extensions_of_this_deviation": f["prior_extensions"]},
                "rules": {"customer_change_approval": self._rule_text(f)},
            }))
        if surface == "deviation_form":
            rows = [("Deviation", f["request_id"]), ("Raised", fmt_date(f["date"], rng)),
                    ("Requested by", f["requester"]), ("Part / customer", f"{f['part']} / {f['customer']}"),
                    ("Change requested", what), ("Reason", why),
                    ("Duration", f"{_plural(f['weeks'], 'week')}, about {f['qty']:,} parts"),
                    ("Characteristic class", f["char_class"]), ("Equivalence evidence", self._evidence_text(f)),
                    ("Added containment", self._check_text(f)), ("Previous extensions", f["prior_extensions"]),
                    ("Customer approval rule", self._rule_text(f))]
            return "TEMPORARY PROCESS DEVIATION REQUEST\n" + kv_lines(rows, rng, style=pick(rng, ["colon", "aligned"]),
                                                                      skip_none=False)
        r, v = f["requester"].split()[0], f["reviewer"].split()[0]
        msgs = [(r, f"Need to {what} on the {f['part']} for {_plural(f['weeks'], 'week')} "
                    f"(~{f['qty']:,} parts) because {why}." + (" We cannot miss the next shipments." if f["tone"] == "pushy"
                                                                else "")),
                (v, f"What data do we have? The affected feature is {f['char_class']}."),
                (r, _cap(self._evidence_text(f)) + "."),
                (v, "Containment while it runs?"),
                (r, (_cap(self._check_text(f)) + ".") if self._check_text(f) else "Nothing extra planned."),
                (v, _cap(self._rule_text(f)) + ".")]
        if f["prior_extensions"]:
            msgs.append((v, "Note this deviation has already been extended " +
                         {1: "once.", 2: "twice."}.get(f["prior_extensions"], f"{f['prior_extensions']} times.")))
        return f"Review thread -- {f['request_id']} ({f['customer']})\n" + "\n".join(
            f"{who}: {noisy(txt, rng, 0.1)}" for who, txt in msgs)

    def questions(self):
        return {
            "deviation_ruling": choice("How should quality engineering rule on the deviation request?", {
                "approve_after_customer_consent": "Approve, but only once the customer has agreed in writing.",
                "approve_as_requested": "Approve the deviation as requested.",
                "approve_with_limits": "Approve for a shorter period or a smaller quantity, with extra checks while it runs.",
                "reject_deviation": "Refuse it; production must stay on the approved process.",
            }),
            "equivalence_strength": score("How well does the evidence show the deviated process performs like the "
                                          "approved one?", [
                "Not at all: no data.",
                "Weakly: a handful of parts, no statistics.",
                "Fairly: a capability study near the requirement.",
                "Convincingly: a full study comfortably above the requirement.",
            ]),
            "added_defect_risk": score("How much does the deviation raise the chance of defective parts reaching the "
                                       "customer?", [
                "Negligibly.",
                "Slightly, and the added checks would catch most of it.",
                "Noticeably, on a feature that matters.",
                "Substantially, on a critical feature with little to catch it.",
            ]),
            "customer_consent_required": noul(
                "The customer must agree before parts made under this deviation can ship.",
                yes="The customer's rules require approval of process changes on its parts.",
                no="The part is our own design, or the change falls outside what the customer's rules cover."),
            "masks_permanent_change": noul(
                "The deviation is standing in for a permanent change that should be qualified properly."),
        }


# =========================================================================== schema 5
RECURRING = {
    "burrs": ("burrs on cross-drilled holes", "machining", "fixture", ["Fixture 1", "Fixture 2", "Fixture 3", "Fixture 4"]),
    "short_shots": ("short shots on the clip moulding", "moulding", "cavity", ["Cav 1", "Cav 2", "Cav 3", "Cav 4",
                                                                                "Cav 5", "Cav 6"]),
    "weld_porosity": ("porosity in the bracket fillet weld", "welding", "cell", ["Cell A", "Cell B", "Cell C"]),
    "torque_low": ("fastening-station torque below specification", "assembly", "spindle", ["Sp 1", "Sp 2", "Sp 3",
                                                                                            "Sp 4"]),
    "label_misprint": ("misprinted batch codes on carton labels", "packing", "printer", ["PR-1", "PR-2"]),
}
# defect -> (pieces per occurrence lo, hi, scrap value per piece)
SCRAP = {"burrs": (5, 60, 6.0), "short_shots": (50, 800, 0.35), "weld_porosity": (3, 40, 12.0),
         "torque_low": (5, 60, 18.0), "label_misprint": (100, 1200, 0.9)}
PATTERNS = {"one_station": "tooling_or_fixture", "one_shift": "operator_training", "after_lot_change": "incoming_material",
            "everywhere": "process_parameters", "marginal_design": "design_change"}
LEAD_CAUSE = {"operator": "operator error; the operators have been reminded of the work instruction",
              "fixture": "worn fixture or tooling", "material": "a bad batch of material",
              "unknown": "cause not yet known"}


class RecurringDefectCapa(Schema):
    name = "recurring_defect_capa"
    description = "A defect that keeps coming back on a line: how deep the corrective action must go, and where."
    surfaces = {"json": 0.45, "defect_log": 0.3, "capa_request": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        dkey = pick(rng, list(RECURRING))
        occ = pick(rng, [2, 3, 4, 6, 8])
        pattern = pick_w(rng, {"one_station": 3, "one_shift": 4, "after_lot_change": 2, "everywhere": 2,
                               "marginal_design": 1})
        lead = pick_w(rng, {"operator": 4, "fixture": 2, "material": 2, "unknown": 2})
        detection = pick_w(rng, {"in_station": 3, "final_inspection": 4, "customer_reject": 2})
        retrained = chance(rng, 0.4)
        contained = chance(rng, 0.6)
        sites = pick(rng, [1, 1, 2, 3])                      # plants running the same process
        if reg == "clear":
            if chance(rng, 0.5):   # a single slip, caught in station
                occ, pattern, lead, detection, retrained, sites = 2, "one_shift", "operator", "in_station", False, 1
            else:                  # repeated, reaching the customer, systemic
                occ, pattern, detection, retrained, sites = pick(rng, [6, 8]), pick(rng, ["everywhere", "marginal_design"]), \
                    "customer_reject", True, pick(rng, [2, 3])
        elif reg == "borderline":
            occ = pick(rng, [3, 4])
            detection = pick(rng, ["in_station", "final_inspection"])
        else:  # conflicting: the line lead's cause does not fit the log
            lead = "operator"
            pattern = pick(rng, ["one_station", "after_lot_change", "everywhere", "marginal_design"])
            retrained = True
        lo, hi, value = SCRAP[dkey]
        qtys = [rng.randint(lo, hi) for _ in range(occ)]
        scrap = round(sum(qtys) * value, -1)
        d = {"regime": reg, "defect_key": dkey, "defect": RECURRING[dkey][0], "process": RECURRING[dkey][1],
                "occurrence_qty": qtys,
                "station_word": RECURRING[dkey][2], "stations": RECURRING[dkey][3], "occurrences_90d": occ,
                "pattern": pattern, "lead_cause": lead, "detection": detection, "retrained_before": retrained,
                "scrap_cost_90d": scrap, "currency": pick(rng, ["USD", "EUR", "GBP"]), "containment_in_place": contained,
                "sites_same_process": sites, "capa": ident(rng, "CAPA", 4, "-"), "lead": person(rng),
                "start": rand_date(rng), "cpk_note": round(between(rng, 0.7, 1.05), 2)}
        d["log_rows"] = self._log(d, rng)
        return d

    def _summary(self, f):
        """How tightly the occurrences cluster on one shift, operator, station and material lot."""
        rows, n, out = f["log_rows"], len(f["log_rows"]), []
        for i, label in ((1, "shift"), (3, "operator"), (2, f["station_word"]), (4, "material lot")):
            v, c = Counter(r[i] for r in rows).most_common(1)[0]
            out.append(f"{label} {v} ({c} of {n})")
        return "most frequent " + ", ".join(out)

    def _log(self, f, rng):
        """The occurrence log: the pattern is visible in which station, shift, operator and lot each row names."""
        st = f["stations"]
        bad_st, bad_shift, bad_op, new_lot = pick(rng, st), pick(rng, ["A", "B", "C"]), f"op{rng.randint(200, 299)}", \
            f"ML-{rng.randint(1000, 9999)}"
        old_lots = [f"ML-{rng.randint(1000, 9999)}" for _ in range(2)]
        ops = [f"op{rng.randint(100, 199)}" for _ in range(4)]
        rows, day = [], f["start"]
        for i in range(f["occurrences_90d"]):
            day = day + timedelta(days=rng.randint(2, max(3, 84 // f["occurrences_90d"])))
            p = f["pattern"]
            station = bad_st if p == "one_station" else pick(rng, st)
            shift = bad_shift if p == "one_shift" else pick(rng, ["A", "B", "C"])
            op = bad_op if p == "one_shift" else pick(rng, ops)
            lot = new_lot if p == "after_lot_change" else pick(rng, old_lots + [new_lot])
            rows.append([fmt_date(day, style="iso"), shift, station, op, lot, f["occurrence_qty"][i]])
        return rows

    def _detect_text(self, f):
        return {"in_station": "caught at the station by the in-process check",
                "final_inspection": "caught at final inspection", "customer_reject": "at least one caught by the "
                "customer at incoming inspection"}[f["detection"]]

    def render(self, f, rng, surface):
        rows = f["log_rows"]
        scrap = _money(f["scrap_cost_90d"], rng, f["currency"])
        extra = [f"Log summary: {self._summary(f)}."]
        if f["pattern"] == "marginal_design":
            extra.append(f"Capability on the affected feature is Cpk {f['cpk_note']} on every {f['station_word']}.")
        if f["retrained_before"]:
            extra.append("Operators were retrained on the work instruction after the first occurrences; the defect "
                         "came back.")
        if f["sites_same_process"] > 1:
            extra.append(f"The same process runs at {f['sites_same_process']} plants.")
        else:
            extra.append("No other plant runs this process.")
        if surface == "json":
            return dumps(prune({
                "capa_request": {"id": f["capa"], "raised_by": f["lead"], "defect": f["defect"],
                                 "process": f["process"], "occurrences_last_90_days": f["occurrences_90d"],
                                 "detection": self._detect_text(f), "scrap_cost_90_days": scrap,
                                 "containment_in_place": f["containment_in_place"]},
                "line_lead_view": LEAD_CAUSE[f["lead_cause"]],
                "occurrence_log": [{"date": r[0], "shift": r[1], f["station_word"]: r[2], "operator": r[3],
                                    "material_lot": r[4], "qty": r[5]} for r in rows],
                "notes": extra or None,
            }))
        if surface == "defect_log":
            return "\n".join([f"{f['capa']} -- {f['defect']} ({f['process']})", "",
                              table(["date", "shift", f["station_word"], "operator", "material lot", "qty"], rows, rng),
                              "", bullets([f"Detection: {self._detect_text(f)}.", f"Scrap cost, 90 days: {scrap}.",
                                           f"Line lead: {LEAD_CAUSE[f['lead_cause']]}.",
                                           "Containment in place." if f["containment_in_place"] else
                                           "No containment in place yet.", *extra], rng)])
        occ = "; ".join(f"{r[0]} shift {r[1]} {r[2]} {r[3]} lot {r[4]} ({r[5]} pcs)" for r in rows)
        form = kv_lines([("CAPA request", f["capa"]), ("Raised by", f["lead"]), ("Defect", f["defect"]),
                         ("Process", f["process"]), ("Occurrences (90 d)", f["occurrences_90d"]),
                         ("Detection", self._detect_text(f)), ("Scrap cost (90 d)", scrap),
                         ("Containment", "in place" if f["containment_in_place"] else "not yet"),
                         ("Suspected cause (line lead)", LEAD_CAUSE[f["lead_cause"]])], rng,
                        style=pick(rng, ["colon", "aligned"]))
        return form + "\n\nOccurrences: " + occ + ("\n\nNotes: " + noisy(join_sentences(extra), rng, 0.1) if extra else "")

    def questions(self):
        return {
            "capa_depth": choice("How deep should the corrective action go?", {
                "correct_and_close": "Correct the parts and close: two occurrences, both caught at the station.",
                "cross_site_capa": "Formal corrective action across every plant using the process: a formal-CAPA "
                                   "trigger (a customer reject, six or more occurrences, or four or more with a repeat "
                                   "after retraining) and the process runs at two or more plants.",
                "formal_root_cause_capa": "Formal root-cause corrective action at this plant: a formal-CAPA trigger, "
                                          "and no other plant runs the process.",
                "local_fix_with_check": "Local fix with a follow-up check: no formal-CAPA trigger, but three or more "
                                        "occurrences or one found only at final inspection.",
            }),
            "fix_direction": choice("Where does the evidence point the fix?", {
                "design_change": "The part design, which leaves too little margin for the process.",
                "incoming_material": "The incoming material or the supplier that provides it.",
                "operator_training": "The people running the process and how they are trained.",
                "process_parameters": "The process settings, which drift or are set wrongly on every station.",
                "tooling_or_fixture": "A specific tool, fixture, cavity or station.",
            }),
            "recurrence_outlook": score("Without new action, how likely is the defect to come back in the next quarter?", [
                "Unlikely: three or fewer occurrences, no earlier retraining, the stated cause fits more than half "
                "the log, and containment is in place.",
                "Possible: three or fewer occurrences and no earlier retraining, but the stated cause fits half the "
                "log or less, or there is no containment.",
                "Likely: four or more occurrences, or it came back after operators were retrained.",
                "Near certain: four or more occurrences and it came back after operators were retrained.",
            ]),
            "stated_cause_credibility": score("How well does the line lead's stated cause fit the occurrence log?", [
                "Not at all: no cause is stated, or no two occurrences share what the cause implies (operator error: "
                "one operator; worn fixture or tooling: one station; bad material: one material lot).",
                "Poorly: at least two but no more than half of the occurrences share it.",
                "Reasonably: more than half, but not all, share it.",
                "Fully: every occurrence shares it.",
            ]),
            "operator_error_explains": noul(
                "Operator error on its own explains the occurrences.",
                yes="Every occurrence has the same operator and shift, they do not also all share one station and "
                    "one material lot, and no capability note points at the design.",
                no="The occurrences spread across operators or shifts, or they also all share one station and one "
                   "material lot, or a capability note points at the design."),
        }


WORKFLOW = Workflow(
    name="manufacturing_nonconformance",
    title="Manufacturing nonconformance and material review",
    description="What a plant's quality function does with nonconforming product: disposition, containment, "
                "customer approval or notification, temporary deviations and corrective-action depth.",
    domain="manufacturing quality",
    schemas=[MrbDimensionalDisposition(), ReceivingInspectionReject(), EscapedNonconformance(),
             ProcessDeviationRequest(), RecurringDefectCapa()],
    notes="Quality decisions on parts and processes. Purchased material is judged on inspection evidence only; "
          "no invoice, purchase order, receipt matching, debit note or payment appears.",
)
