"""Waste management operations: day-to-day decisions at landfills, treatment facilities, recycling plants and
transfer stations, judged against (fictional) acceptance plans, receiving rules, hauler agreements, gas-management
plans and station procedures.

Five schemas: a generator's waste profile reviewed for landfill acceptance, a hazardous-waste shipment whose
manifest and fingerprint disagree with what arrived, an inbound recycling load inspected for contamination, a
landfill perimeter gas-probe reading against the gas plan's action level, and an incident on a transfer station's
tipping floor. Artefacts are profile sheets, technical-services emails, receiving logs, generator emails,
inspection tickets, floor chat, probe monitoring sheets and incident forms.

Facility permits and discharge or emission reports (environmental_compliance_reports) are out of scope: these are
operational acceptance, receiving and response calls. All facilities, boards, plans, limits and people are invented;
thresholds are sampled per case, and only the clauses that govern the case are quoted.

Each schema also has `intended(f)`: the designer's reading of the rule for that case. It is never rendered and
never a label; it exists so the answer-spread check can confirm every question moves with the factors.
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
    rand_date,
    rand_ts,
    regime,
    score,
    table,
)

SITE_A = ["Cinder Hill", "Mallow Creek", "Hesketh Road", "Brayford", "Tolland Ridge", "Kilnworth", "Saddleback",
          "Orme Valley", "Partridge Lane", "Wexley Moor", "Gannet Point", "Rookery Farm", "Sedgewick", "Fallow Fields"]


GEN_A = ["Arden", "Blakemore", "Corrigan", "Dovecote", "Easterly", "Fennimore", "Glenholme", "Harrowby", "Irvale",
         "Jessamy", "Kirkdale", "Lomond", "Mosswood", "Northam", "Pendleford", "Quarles"]
HAULER_B = ["Hauling", "Roll-Off Services", "Demolition", "Waste Services", "Junk Removal", "Skip Hire"]


def _site(rng, kind):
    return f"{pick(rng, SITE_A)} {kind}"


def _hazgen(rng):
    """An invented hazardous-waste generator."""
    return f"{pick(rng, GEN_A)} {pick(rng, ['Plating', 'Coatings', 'Chemicals', 'Laboratories', 'Circuit Boards', 'Metal Finishing', 'Pharmaceuticals'])}"


def _hauler(rng):
    return f"{pick(rng, GEN_A)} {pick(rng, HAULER_B)}"


def _first(name):
    return name.split()[0]


def _kv(pairs, rng, skip_none=True):
    """kv_lines without the snake_case 'equals' house style, which mangles labels such as 'Site / sector'."""
    return kv_lines(pairs, rng, style=pick(rng, ["colon", "colon", "aligned", "dash"]), skip_none=skip_none)


# =========================================================================== schema 1
# stream -> (analytes tested, liquid-prone, on the waste board's special list)
STREAMS = {
    "dewatered wastewater sludge": (["lead", "cadmium", "chromium"], True, True),
    "foundry sand": (["lead", "cadmium", "chromium"], False, False),
    "petroleum-contaminated soil": (["benzene", "lead"], False, False),
    "incinerator bottom ash": (["lead", "cadmium"], False, True),
    "auto shredder residue": (["lead", "cadmium", "PCB"], False, True),
    "paint booth filter waste": (["lead", "chromium"], True, False),
    "spent blasting grit": (["lead", "chromium"], False, False),
    "drilling mud and cuttings": (["benzene", "chromium"], True, False),
    "street sweepings": (["lead", "benzene"], False, False),
    "ceramic kiln dust": (["lead", "cadmium"], False, False),
}
LIMITS = {"lead": ([1.5, 2.0, 2.5], "mg/L leachate"), "cadmium": ([0.2, 0.3, 0.5], "mg/L leachate"),
          "chromium": ([2.0, 3.0], "mg/L leachate"), "benzene": ([0.2, 0.3], "mg/L leachate"),
          "PCB": ([10, 25], "mg/kg")}
STREAM_GEN = {"dewatered wastewater sludge": ["Water Reclamation District", "Wastewater Treatment Works"],
              "foundry sand": ["Foundry", "Castings"], "petroleum-contaminated soil": ["Fuel Terminal", "Remediation Contractors"],
              "incinerator bottom ash": ["Energy Recovery", "Resource Recovery Plant"],
              "auto shredder residue": ["Metal Recycling", "Auto Salvage"], "paint booth filter waste": ["Coatings", "Body Works"],
              "spent blasting grit": ["Shipyard", "Industrial Coatings"], "drilling mud and cuttings": ["Drilling", "Well Services"],
              "street sweepings": ["Public Works Department", "Highways Depot"], "ceramic kiln dust": ["Ceramics", "Tileworks"]}
PROCESS_CHANGES = ["switched coating supplier after the sample was taken", "added a second furnace line in March",
                   "now blends in material from a second plant", "changed the dewatering polymer"]


class WasteAcceptanceProfile(Schema):
    name = "waste_acceptance_profile"
    description = "A generator's special-waste profile reviewed against the landfill's waste acceptance plan."
    surfaces = {"json": 0.5, "profile_sheet": 0.25, "review_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        stream = pick(rng, list(STREAMS))
        names, liquid, listed = STREAMS[stream]
        lims = {a: pick(rng, LIMITS[a][0]) for a in names}
        vals = {a: round(lims[a] * rng.uniform(0.05, 0.9), 3 if lims[a] < 1 else 2) for a in names}
        paint = pick_w(rng, {"pass": 4, "fail": 3, None: 1}) if liquid else pick_w(rng, {"pass": 7, "fail": 1, None: 1.2})
        max_age = pick(rng, [6, 12])
        age = rng.randint(1, max_age + 4)
        sample = pick_w(rng, {"composite": 5, "grab": 3, "knowledge": 1})
        V = pick(rng, [500, 1000, 2000])
        tonnes = int(V * rng.uniform(0.1, 2.0))
        change = maybe(rng, pick(rng, PROCESS_CHANGES), 0.85)
        k = None
        if reg == "clear":
            if chance(rng, 0.5):
                vals = {a: round(lims[a] * rng.uniform(0.02, 0.3), 3 if lims[a] < 1 else 2) for a in names}
                paint, age, sample, change = "pass", rng.randint(1, max_age - 2), "composite", None
            else:
                a = pick(rng, names)
                vals[a] = round(lims[a] * rng.uniform(1.4, 4.0), 3 if lims[a] < 1 else 2)
                age, sample, change = rng.randint(1, max_age - 1), "composite", None
        elif reg == "borderline":
            k = pick(rng, ["limit", "age", "liquid"])
            if k == "limit":
                a = pick(rng, names)
                vals[a] = near(rng, lims[a], lims[a] * 0.12, lo=0.001, digits=3 if lims[a] < 1 else 2)
            elif k == "age":
                age = near(rng, max_age, 1, lo=1, integer=True)
            else:
                paint = pick(rng, ["fail", "pass"])
        else:
            k = pick(rng, ["clean_but_stale", "knowledge_only", "process_changed", "one_spike"])
            if k == "clean_but_stale":
                vals = {a: round(lims[a] * rng.uniform(0.05, 0.4), 3 if lims[a] < 1 else 2) for a in names}
                age = max_age + rng.randint(2, 10)
            elif k == "knowledge_only":
                sample = "knowledge"
            elif k == "process_changed":
                vals = {a: round(lims[a] * rng.uniform(0.05, 0.5), 3 if lims[a] < 1 else 2) for a in names}
                change = pick(rng, PROCESS_CHANGES)
            else:
                a = pick(rng, names)
                vals[a] = round(lims[a] * rng.uniform(1.05, 1.6), 3 if lims[a] < 1 else 2)
                sample = "grab"
        if sample == "knowledge":
            vals = {a: None for a in names}
        elif chance(rng, 0.08):
            vals[pick(rng, names)] = None
        return {"regime": reg, "case_kind": k, "generator": f"{pick(rng, GEN_A)} {pick(rng, STREAM_GEN[stream])}",
                "stream": stream, "analytes": names,
                "values": vals, "limits": lims, "paint_filter": paint, "liquid_prone": liquid, "listed": listed,
                "age_months": age, "max_age": max_age, "sample": sample, "n_grabs": rng.randint(3, 12),
                "tonnes": tonnes, "tonnes_rule": V, "change": change, "profile": ident(rng, "WP", 5, "-"),
                "landfill": _site(rng, pick(rng, ["Landfill", "Disposal Site", "Waste Management Facility"])),
                "reviewer": person(rng), "contact": person(rng), "date": rand_date(rng),
                "plan": pick(rng, ["WAP-3", "Waste Acceptance Plan rev 7", "SWAP-2"]),
                "note": pick(rng, ["same", "urgent", "none", "none"])}

    def _ratio(self, f):
        rs = [f["values"][a] / f["limits"][a] for a in f["analytes"] if f["values"][a] is not None]
        return max(rs) if rs else None

    def intended(self, f):
        r = self._ratio(f)
        missing = any(f["values"][a] is None for a in f["analytes"])
        data_ok = (f["age_months"] <= f["max_age"] and f["sample"] != "knowledge" and not missing
                   and f["change"] is None and f["paint_filter"] is not None)
        if r is not None and r >= 1:
            dec = "reject_profile"
        elif not data_ok:
            dec = "request_further_testing"
        elif f["paint_filter"] == "fail" or r >= 0.8:
            dec = "approve_with_handling_conditions"
        else:
            dec = "approve_profile"
        if f["listed"]:
            lvl = "agency_prior_approval"
        elif f["tonnes"] > f["tonnes_rule"]:
            lvl = "corporate_special_waste_review"
        elif (r is not None and r >= 0.5) or f["paint_filter"] == "fail":
            lvl = "site_engineer_signoff"
        else:
            lvl = "gate_acceptance"
        if r is None:
            lc = 2 if f["listed"] else 1
        else:
            lc = 0 if r < 0.25 else 1 if r < 0.6 else 2 if r < 1 else 3
        da = 3
        if f["age_months"] > f["max_age"]:
            da -= 1
        da -= {"composite": 0, "grab": 1, "knowledge": 2}[f["sample"]]
        if missing and f["sample"] != "knowledge":
            da -= 1
        if f["change"]:
            da -= 1
        if f["paint_filter"] is None:
            da -= 1
        return {"profile_decision": dec, "required_signoff": lvl, "leachability_concern": lc,
                "data_adequacy": max(0, da), "free_liquids_rule_met": "yes" if f["paint_filter"] == "pass" else "no"}

    def _rule(self, f):
        lim = "; ".join(f"{a} < {f['limits'][a]} {LIMITS[a][1].split()[0]}" for a in f["analytes"])
        return (f"{f['plan']}: accept only if {lim}; no free liquids (paint filter pass); analysis <= {f['max_age']} "
                f"months old and representative; > {f['tonnes_rule']:,} t/yr needs corporate review; board-listed streams "
                "need prior agency approval")

    def _sample_txt(self, f):
        return {"composite": f"composite of {f['n_grabs']} grabs", "grab": "single grab sample",
                "knowledge": "generator knowledge only, no lab data"}[f["sample"]]

    def _note(self, f, rng):
        t = {"same": "Process is the same as last year's approved profile.",
             "urgent": "We need the first loads in within two weeks, the storage pad is full.", "none": ""}[f["note"]]
        return noisy(t, rng, 0.15) if t else None

    def _val(self, f, a):
        v = f["values"][a]
        return "not analysed" if v is None else f"{v} {LIMITS[a][1].split()[0]}"

    def render(self, f, rng, surface):
        note = self._note(f, rng)
        pf = {None: "not tested", "pass": "pass", "fail": "fail (free liquid present)"}[f["paint_filter"]]
        if surface == "json":
            return dumps(prune({
                "profile": {"id": f["profile"], "generator": f["generator"], "stream": f["stream"],
                            "annual_tonnes": f["tonnes"], "board_special_list": f["listed"],
                            "process_change": f["change"], "generator_note": note},
                "analysis": {"results": {a: f["values"][a] for a in f["analytes"]}, "paint_filter": pf,
                             "sample": self._sample_txt(f), "age_months": f["age_months"]},
                "acceptance_rule": self._rule(f),
            }))
        if surface == "profile_sheet":
            rows = [[a, self._val(f, a), f"< {f['limits'][a]}"] for a in f["analytes"]]
            return "\n".join([
                f"SPECIAL WASTE PROFILE {f['profile']} -- {f['landfill']}",
                _kv([("Generator", f"{f['generator']} (contact {f['contact']})"), ("Waste", f["stream"]),
                          ("Est. annual quantity", f"{f['tonnes']:,} t"),
                          ("On board special-waste list", "yes" if f["listed"] else "no"),
                          ("Sampling", f"{self._sample_txt(f)}, {f['age_months']} months ago"),
                          ("Paint filter test", pf), ("Process change since sampling", f["change"] or "none declared")], rng),
                table(["analyte", "result", "plan limit"], rows, rng), f"Generator remarks: {note or '-'}",
                f"Plan: {f['plan']} -- no free liquids; analysis max {f['max_age']} months; > {f['tonnes_rule']:,} t/yr "
                "corporate review; board-listed streams need agency approval."])
        res = ", ".join(f"{a} {self._val(f, a)}" for a in f["analytes"])
        s = [f"Profile {f['profile']} from {f['generator']} is for {f['tonnes']:,} t/yr of {f['stream']}"
             + (" (a board-listed stream)." if f["listed"] else "."),
             f"Results: {res}; paint filter {pf}.",
             f"The data come from a {self._sample_txt(f)} taken {f['age_months']} months ago.",
             (f"Generator mentions they {f['change']}." if f["change"] else ""),
             (f"Their note: \"{note}\"" if note else "")]
        return "\n".join([f"Subject: Profile review {f['profile']} ({f['stream']})", f"From: {f['reviewer']}, technical services",
                          f"To: landfill manager, {f['landfill']}", "", join_sentences(s), "", f"Ref: {self._rule(f)}."])

    def questions(self):
        return {
            "profile_decision": choice("How should the landfill decide on this waste profile?", {
                "approve_profile": "Approve the profile for routine acceptance.",
                "approve_with_handling_conditions": "Approve it with conditions, such as solidifying first or retesting more often.",
                "reject_profile": "Reject the profile; the waste cannot come to this landfill.",
                "request_further_testing": "Hold the decision until current, representative analysis is supplied.",
            }),
            "required_signoff": choice("Under the acceptance plan, who has to sign off before loads arrive?", {
                "agency_prior_approval": "The state waste board, through prior approval.",
                "corporate_special_waste_review": "Corporate technical services' special-waste review.",
                "gate_acceptance": "No one beyond routine gate acceptance by the scale house.",
                "site_engineer_signoff": "The site engineer.",
            }),
            "leachability_concern": score("How close does the waste come to the plan's leachate limits?", [
                "Far: every result is a small fraction of its limit.",
                "Moderate: results are well inside the limits.",
                "Close: a result is near its limit, or the stream is high-risk with no numbers.",
                "Over: at least one result is at or above its limit.",
            ]),
            "data_adequacy": score("How adequate is the analytical package for an acceptance decision?", [
                "Inadequate: little that can be relied on.",
                "Weak: several gaps such as age, sampling or missing tests.",
                "Mostly adequate: one gap.",
                "Complete: current, representative and covering every test.",
            ]),
            "free_liquids_rule_met": noul(
                "The waste as it would be shipped meets the plan's free-liquids rule.",
                yes="The paint filter test was run and passed.",
                no="The test failed or was not run."),
        }


# =========================================================================== schema 2
CONTAINERS = {"drums": ("55-gal drums", "count"), "totes": ("IBC totes", "count"), "tanker": ("bulk tanker", "tons"),
              "rolloff": ("roll-off box of solids", "tons")}
BULK_COND = {"sound": "sound", "one leaking": "dome valve weeping", "one bulging": "dome valve weeping",
             "one unlabelled": "placard missing"}
LEAKY = {"one leaking", "one bulging", "dome valve weeping", "liquid seeping from the box"}
UNMARKED = {"one unlabelled", "placard missing"}
DECLARED = {"ignitable": "ignitable solvent waste", "acid": "corrosive (acidic) waste", "alkaline": "corrosive (alkaline) waste",
            "metals": "metal-bearing aqueous waste", "none": "non-hazardous oily water"}


class ManifestDiscrepancy(Schema):
    name = "hazwaste_manifest_discrepancy"
    description = "A hazardous-waste shipment whose count, weight or fingerprint differs from its manifest, at the receiving bay."
    surfaces = {"json": 0.5, "receiving_log": 0.25, "generator_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        cont = pick(rng, list(CONTAINERS))
        unit = CONTAINERS[cont][1]
        X = pick(rng, [5, 10])
        D = pick(rng, [10, 15, 20])
        flash_cut = pick(rng, [50, 55, 65])
        ph_lo, ph_hi = pick(rng, [2.0, 2.5]), pick(rng, [11.5, 12.0])
        ftype = pick_w(rng, {"aqueous": 4, "full": 2, "fuels": 3})
        permitted = {"aqueous": ["acid", "alkaline", "metals", "none"], "full": ["acid", "alkaline", "metals", "none", "ignitable"],
                     "fuels": ["ignitable", "none"]}[ftype]
        declared = pick(rng, permitted)
        man = rng.randint(4, 80) if unit == "count" else round(rng.uniform(4, 24), 2)
        rec = man
        if unit == "tons":
            rec = round(man * (1 + rng.uniform(-X, X) / 100 * 0.6), 2)
        flash = self._flash_for(declared, flash_cut, rng)
        ph = self._ph_for(declared, ph_lo, ph_hi, rng)
        cond = pick_w(rng, {"sound": 7, "one leaking": 1, "one bulging": 1, "one unlabelled": 1})
        days = rng.randint(1, D + 10)
        resp = pick_w(rng, {"no reply": 3, "promised corrected manifest": 2, "disputes our count": 1, "confirmed error, revised manifest sent": 1})
        retest = chance(rng, 0.4)
        k = None
        if reg == "clear":
            v = pick(rng, ["clean", "clean", "short_late", "not_permitted"])
            if v == "clean":
                rec, cond = man, "sound"
                if unit == "tons":
                    rec = round(man * (1 + rng.uniform(-X, X) / 100 * 0.3), 2)
            elif v == "not_permitted":
                if ftype == "fuels":
                    declared, flash, ph, retest = "none", None, round(rng.uniform(0.4, ph_lo - 0.5), 1), True
                else:
                    ftype, permitted = "aqueous", ["acid", "alkaline", "metals", "none"]
                    declared = pick(rng, ["metals", "none"])
                    flash, ph, retest = rng.randint(10, flash_cut - 10), round(rng.uniform(5.0, 9.0), 1), True
            else:
                if unit == "count":
                    rec = man - pick(rng, [1, 2, 3])
                else:
                    rec = round(man * (1 - rng.uniform(X * 2, X * 3) / 100), 2)
                days, resp = D + rng.randint(1, 8), "no reply"
        elif reg == "borderline":
            k = pick(rng, ["weight", "fingerprint", "deadline"])
            if k == "weight":
                cont = pick(rng, ["tanker", "rolloff"])
                unit = "tons"
                man = round(rng.uniform(4, 24), 2)
                rec = round(man * (1 + pick(rng, [-1, 1]) * near(rng, X, 1.5, lo=0.2) / 100), 2)
            elif k == "fingerprint":
                if chance(rng, 0.5):
                    flash = near(rng, flash_cut, 4, lo=10, integer=True)
                else:
                    ph = near(rng, ph_lo, 0.4, lo=0.1, digits=1) if chance(rng, 0.5) else near(rng, ph_hi, 0.4, hi=14, digits=1)
                retest = False
            else:
                if unit == "count":
                    rec = man + pick(rng, [-1, 1])
                days = near(rng, D, 2, lo=1, integer=True)
                resp = pick(rng, ["no reply", "promised corrected manifest"])
        else:
            k = pick(rng, ["undeclared_ignitable", "undeclared_corrosive", "count_ok_leaker"])
            if k == "undeclared_ignitable":
                declared = pick(rng, [p for p in permitted if p != "ignitable"])
                flash = rng.randint(12, flash_cut - 8)
                ph = self._ph_for(declared, ph_lo, ph_hi, rng)
                retest = True
            elif k == "undeclared_corrosive":
                declared = pick(rng, [p for p in permitted if p in ("metals", "none")])
                ph, flash = round(rng.uniform(0.5, ph_lo - 0.3), 1), self._flash_for("none", flash_cut, rng)
                retest = True
            else:
                rec, cond = man, pick(rng, ["one leaking", "one bulging"])
                if unit == "tons":
                    rec = round(man * (1 + rng.uniform(-X, X) / 100 * 0.3), 2)
        if unit == "count" and rec == man and reg != "clear" and chance(rng, 0.2):
            rec = man - 1
        if unit == "tons":
            cond = BULK_COND[cond]
            if cont == "rolloff" and cond == "dome valve weeping":
                cond = "liquid seeping from the box"
        if reg != "clear" and k != "fingerprint" and chance(rng, 0.25):
            # an undeclared property shows up in the fingerprint
            if ftype == "aqueous" or (ftype == "full" and declared != "ignitable"):
                flash, retest = rng.randint(10, flash_cut - 6), True
            else:
                ph, retest = round(rng.uniform(0.4, ph_lo - 0.3), 1), True
        return {"regime": reg, "case_kind": k, "container": cont, "unit": unit, "manifest_qty": man, "received_qty": rec,
                "declared": declared, "permitted": permitted, "flash_c": flash, "ph": ph, "condition": cond,
                "days_since_receipt": days, "generator_response": resp, "retest_done": retest,
                "weight_tol_pct": X, "resolve_days": D, "flash_cut": flash_cut, "ph_lo": ph_lo, "ph_hi": ph_hi,
                "generator": _hazgen(rng), "transporter": f"{pick(rng, SITE_A).split()[0]} {pick(rng, ['Haulage', 'Environmental Transport', 'Tank Lines'])}",
                "facility": _site(rng, {"aqueous": "Chemical Treatment Plant", "full": pick(rng, ["Treatment Centre", "Waste Treatment Facility"]),
                                        "fuels": "Fuel Blending Facility"}[ftype]),
                "manifest": ident(rng, "MF", 8, ""), "clerk": person(rng), "gen_contact": person(rng),
                "date": rand_date(rng), "rule": pick(rng, ["HW-14", "SWB Rule 9.2", "Receiving Rule R-5"])}

    @staticmethod
    def _flash_for(declared, cut, rng):
        if declared == "ignitable":
            return rng.randint(5, cut - 10)
        return None if chance(rng, 0.7) else rng.randint(cut + 15, 93)

    @staticmethod
    def _ph_for(declared, lo, hi, rng):
        if declared == "acid":
            return round(rng.uniform(0.3, lo - 0.3), 1)
        if declared == "alkaline":
            return round(rng.uniform(hi + 0.2, 13.8), 1)
        return round(rng.uniform(4.5, 9.5), 1)

    def _actual(self, f):
        a = set()
        if f["flash_c"] is not None and f["flash_c"] < f["flash_cut"]:
            a.add("ignitable")
        if f["ph"] <= f["ph_lo"]:
            a.add("acid")
        if f["ph"] >= f["ph_hi"]:
            a.add("alkaline")
        if not a:
            a.add(f["declared"] if f["declared"] in ("metals", "none") else "none")
        return a

    def _sig(self, f):
        if f["unit"] == "count":
            qty = f["received_qty"] != f["manifest_qty"]
        else:
            qty = abs(f["received_qty"] - f["manifest_qty"]) / f["manifest_qty"] * 100 > f["weight_tol_pct"]
        undeclared = self._actual(f) - {f["declared"]}
        type_mismatch = bool(undeclared - {"none", "metals"}) or (f["declared"] in ("ignitable", "acid", "alkaline")
                                                                 and f["declared"] not in self._actual(f))
        return qty, type_mismatch

    def intended(self, f):
        qty, mismatch = self._sig(f)
        sig = qty or mismatch
        actual = self._actual(f)
        covered = actual <= set(f["permitted"])
        near_cut = ((f["flash_c"] is not None and abs(f["flash_c"] - f["flash_cut"]) <= 4)
                    or abs(f["ph"] - f["ph_lo"]) <= 0.4 or abs(f["ph"] - f["ph_hi"]) <= 0.4)
        resolved = f["generator_response"] == "confirmed error, revised manifest sent"
        if not covered:
            act = "reject_shipment"
        elif near_cut and not f["retest_done"]:
            act = "hold_for_fingerprint_retest"
        elif not sig:
            act = "accept_note_minor_variance"
        elif f["days_since_receipt"] >= f["resolve_days"] and not resolved:
            act = "report_unresolved_discrepancy"
        else:
            act = "accept_and_reconcile"
        hz = 0
        und = actual - {f["declared"]}
        if "ignitable" in und:
            hz = 3
        elif und & {"acid", "alkaline"}:
            hz = 2
        elif qty:
            hz = 1
        if f["condition"] in LEAKY:
            hz = min(3, hz + 1)
        if not covered or (sig and f["days_since_receipt"] >= f["resolve_days"] and not resolved):
            rx = 3
        elif sig and f["days_since_receipt"] >= f["resolve_days"] - 3 and not resolved:
            rx = 2
        elif sig or f["condition"] in UNMARKED:
            rx = 1
        else:
            rx = 0
        return {"receiving_action": act, "handling_hazard": hz, "regulatory_exposure": rx,
                "significant_under_rule": "yes" if sig else "no", "permit_covers_actual_waste": "yes" if covered else "no"}

    def _rule_txt(self, f):
        return (f"{f['rule']}: significant = any container-count difference, bulk weight off by > {f['weight_tol_pct']}%, "
                f"or a waste type other than manifested; unresolved after {f['resolve_days']} days -> report to the board. "
                f"Receiving criteria: flash point < {f['flash_cut']} C = ignitable; pH <= {f['ph_lo']} or >= {f['ph_hi']} = corrosive")

    def _qty(self, f, v):
        return f"{v} {CONTAINERS[f['container']][0]}" if f["unit"] == "count" else f"{v} t"

    def _fp(self, f):
        fl = "no flash to 93 C" if f["flash_c"] is None else f"flash point {f['flash_c']} C"
        return f"{fl}, pH {f['ph']}" + (" (retest done)" if f["retest_done"] else "")

    def _permit(self, f):
        return ", ".join(DECLARED[p].replace(" waste", "") for p in f["permitted"])

    def render(self, f, rng, surface):
        if surface == "json":
            return dumps(prune({
                "shipment": {"manifest": f["manifest"], "generator": f["generator"], "transporter": f["transporter"],
                             "container": CONTAINERS[f["container"]][0], "manifest_qty": f["manifest_qty"],
                             "received_qty": f["received_qty"], "unit": f["unit"], "condition_on_arrival": f["condition"],
                             "manifested_as": DECLARED[f["declared"]]},
                "fingerprint": self._fp(f),
                "follow_up": {"days_since_receipt": f["days_since_receipt"], "generator_response": f["generator_response"]},
                "facility": {"name": f["facility"], "permit_accepts": self._permit(f)},
                "rule": self._rule_txt(f),
            }))
        if surface == "receiving_log":
            return "\n".join([
                f"RECEIVING LOG -- {f['facility']}  {fmt_date(f['date'], rng)}  clerk {f['clerk']}",
                _kv([("Manifest", f"{f['manifest']} ({f['generator']}, via {f['transporter']})"),
                          ("Manifested", f"{self._qty(f, f['manifest_qty'])}, {DECLARED[f['declared']]}"),
                          ("Received", self._qty(f, f["received_qty"])), ("Condition", f["condition"]),
                          ("Fingerprint", self._fp(f)), ("Permit accepts", self._permit(f)),
                          ("Days since receipt", f["days_since_receipt"]), ("Generator", f["generator_response"])], rng),
                self._rule_txt(f)])
        s = [f"Re manifest {f['manifest']}: it lists {self._qty(f, f['manifest_qty'])} of {DECLARED[f['declared']]}.",
             f"We received {self._qty(f, f['received_qty'])}; condition: {f['condition']}.",
             f"Our fingerprint shows {self._fp(f)}.",
             f"It is now {f['days_since_receipt']} days since receipt. Status on your side: {f['generator_response']}."]
        return "\n".join([f"Subject: Manifest {f['manifest']} -- receiving discrepancy", f"From: {f['clerk']}, {f['facility']}",
                          f"To: {f['gen_contact']}, {f['generator']}", "", join_sentences(s), "",
                          f"For reference: {self._rule_txt(f)}. Our permit accepts {self._permit(f)}."])

    def questions(self):
        return {
            "receiving_action": choice("What should the receiving facility do with this shipment now?", {
                "accept_and_reconcile": "Accept the waste and work the difference out with the generator in writing.",
                "accept_note_minor_variance": "Accept it and note a variance that is within tolerance.",
                "hold_for_fingerprint_retest": "Hold it in the receiving bay until the fingerprint is retested.",
                "reject_shipment": "Refuse the shipment and arrange its return or redirection.",
                "report_unresolved_discrepancy": "Report the discrepancy to the waste board as unresolved.",
            }),
            "handling_hazard": score("How much extra handling hazard does the difference create at the facility?", [
                "None: the waste is what was declared.",
                "Small: quantities differ but the waste is as declared.",
                "Significant: an undeclared corrosive property, or a damaged container.",
                "High: undeclared ignitable waste, or a hazardous mismatch in a failing container.",
            ]),
            "regulatory_exposure": score("How much regulatory exposure does the facility carry on this shipment?", [
                "None: nothing that needs resolving.",
                "Low: a difference that the resolution window covers.",
                "Rising: the resolution window is nearly used up.",
                "High: past the window unresolved, or waste the permit does not cover.",
            ]),
            "significant_under_rule": noul(
                "The difference counts as a significant discrepancy under the receiving rule.",
                yes="A container count differs, bulk weight is off by more than the tolerance, or the waste type differs.",
                no="Any difference is within the rule's tolerances and the waste is as manifested."),
            "permit_covers_actual_waste": noul("The facility's permit covers the waste as it actually arrived."),
        }


# =========================================================================== schema 3
PROHIBITED = {"none": 0, "bagged recyclables": 0, "hoses and cords": 1, "food waste": 0, "sharps": 2,
              "propane cylinder": 3, "lithium battery pack": 2}
STREAMS_MRF = ["single-stream residential", "commercial cardboard", "mixed paper", "multi-family recycling"]


class RecyclingLoadContamination(Schema):
    name = "recycling_load_contamination"
    description = "An inbound recycling load inspected at the materials recovery facility against the hauler agreement's contamination rules."
    surfaces = {"json": 0.5, "inspection_ticket": 0.25, "floor_chat": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        C = pick(rng, [10, 15, 20, 25])
        method = pick_w(rng, {"visual": 4, "camera": 2, "audit": 2})
        cont = round(rng.uniform(2, C + 12), 1)
        item = pick_w(rng, {"none": 3.5, "bagged recyclables": 2, "hoses and cords": 2.5, "food waste": 1, "sharps": 1.2,
                            "propane cylinder": 1, "lithium battery pack": 1.5})
        smoking = item == "lithium battery pack" and chance(rng, 0.3)
        prior = pick_w(rng, {0: 5, 1: 3, 2: 2, 3: 0.5})
        dispute = chance(rng, 0.3)
        photos = chance(rng, 0.7)
        k = None
        if reg == "clear":
            if chance(rng, 0.5):
                cont, item, method, prior, dispute = round(rng.uniform(1, C - 8), 1), "none", pick(rng, ["audit", "camera"]), 0, False
            else:
                cont, method, photos = round(rng.uniform(C + 8, C + 25), 1), "audit", True
        elif reg == "borderline":
            k = pick(rng, ["threshold", "fee_band", "third_strike"])
            if k == "threshold":
                cont = near(rng, C, 2.5, lo=0.5, digits=1)
            elif k == "fee_band":
                cont = near(rng, C - 5, 2, lo=0.5, digits=1)
            else:
                prior, cont = 2, near(rng, C, 4, lo=0.5, digits=1)
        else:
            k = pick(rng, ["visual_disputed", "clean_but_hazard", "audit_low_driver_blamed"])
            if k == "visual_disputed":
                method, dispute, photos = "visual", True, chance(rng, 0.4)
                cont = near(rng, C, 3, lo=0.5, digits=1)
            elif k == "clean_but_hazard":
                cont = round(rng.uniform(2, C - 6), 1)
                item = pick(rng, ["propane cylinder", "lithium battery pack", "sharps"])
                smoking = item == "lithium battery pack" and chance(rng, 0.5)
            else:
                method, dispute = "audit", True
                cont = round(rng.uniform(C + 1, C + 10), 1)
        return {"regime": reg, "case_kind": k, "hauler": f"{pick(rng, SITE_A).split()[0]} {pick(rng, ['Disposal', 'Waste Services', 'Recycling Haulers'])}",
                "route": f"R-{rng.randint(101, 489)}", "stream": pick(rng, STREAMS_MRF), "truck": pick(rng, ["rear loader", "side loader", "front loader"]),
                "tons": round(rng.uniform(3.5, 11.5), 2), "contamination_pct": cont, "method": method,
                "audit_kg": rng.randint(15, 80) if method == "audit" else None, "photos": photos, "item": item,
                "smoking": smoking, "prior_rejections_90d": prior, "dispute": dispute, "threshold_pct": C,
                "mrf": _site(rng, pick(rng, ["MRF", "Recycling Centre", "Materials Recovery Facility"])),
                "inspector": person(rng), "supervisor": person(rng), "ts": rand_ts(rng, hours=(6, 17)),
                "agreement": pick(rng, ["HDA s5", "Hauler Delivery Agreement 5.2", "Inbound Quality Schedule B"]),
                "ticket": ident(rng, "IN", 6, "-")}

    def intended(self, f):
        C, c = f["threshold_pct"], f["contamination_pct"]
        hazard = PROHIBITED[f["item"]] >= 2
        exceeds = c >= C
        if f["method"] == "visual" and abs(c - C) <= 3 and f["dispute"]:
            disp = "audit_sample_before_ruling"
        elif exceeds:
            disp = "reject_as_refuse"
        elif hazard:
            disp = "isolate_hazard_then_process"
        elif c >= C - 5:
            disp = "accept_with_contamination_fee"
        else:
            disp = "accept_standard"
        strikes = f["prior_rejections_90d"] + (1 if disp == "reject_as_refuse" else 0)
        if strikes >= 3:
            fu = "formal_contract_notice"
        elif f["item"] in ("bagged recyclables", "hoses and cords", "food waste") and (f["prior_rejections_90d"] >= 1 or c >= C - 5):
            fu = "cart_tagging_outreach"
        elif exceeds or hazard or c >= C - 5:
            fu = "written_warning"
        else:
            fu = "none_needed"
        rel = {"audit": 3, "camera": 2, "visual": 1}[f["method"]]
        if f["method"] == "audit" and f["audit_kg"] < 25:
            rel -= 1
        if f["method"] != "audit" and not f["photos"]:
            rel -= 1
        hz = PROHIBITED[f["item"]]
        if f["smoking"]:
            hz = 3
        return {"load_disposition": disp, "hauler_follow_up": fu, "estimate_reliability": max(0, rel),
                "worker_safety_hazard": hz, "exceeds_contract_threshold": "yes" if exceeds else "no"}

    def _rule(self, f):
        C = f["threshold_pct"]
        return (f"{f['agreement']}: contamination >= {C}% = reject, tipped as refuse at hauler cost; {C - 5}-{C}% = "
                f"contamination fee; hazards isolated before processing; 3 rejected loads per route in 90 days = formal notice")

    def _est(self, f):
        m = {"visual": "spotter's visual estimate", "camera": "optical camera estimate",
             "audit": f"weighed audit of a {f['audit_kg']} kg sample"}[f["method"]]
        return f"{f['contamination_pct']}% ({m})"

    def _item(self, f):
        if f["item"] == "none":
            return "none found"
        return f["item"] + (", smoking when found" if f["smoking"] else "")

    def render(self, f, rng, surface):
        disp_txt = "Driver says the contamination came from one commercial stop, not the route." if f["dispute"] else None
        if surface == "json":
            return dumps(prune({
                "load": {"ticket": f["ticket"], "hauler": f["hauler"], "route": f["route"], "stream": f["stream"],
                         "truck": f["truck"], "net_tons": f["tons"], "tipped": fmt_ts(f["ts"], style="iso")},
                "inspection": {"contamination": self._est(f), "photos_taken": f["photos"], "prohibited_items": self._item(f),
                               "inspector": f["inspector"]},
                "hauler_record": {"route_rejections_last_90_days": f["prior_rejections_90d"], "driver_statement": disp_txt},
                "agreement": self._rule(f),
            }))
        if surface == "inspection_ticket":
            body = _kv([("Ticket", f["ticket"]), ("Hauler / route", f"{f['hauler']} / {f['route']}"),
                             ("Stream", f["stream"]), ("Truck", f["truck"]), ("Net weight", f"{f['tons']} t"),
                             ("Contamination", self._est(f)), ("Photos", "yes" if f["photos"] else "no"),
                             ("Prohibited items", self._item(f)),
                             ("Route rejections (90 d)", f["prior_rejections_90d"]),
                             ("Driver statement", disp_txt or "none")], rng)
            return f"INBOUND LOAD INSPECTION -- {f['mrf']}  {fmt_ts(f['ts'], rng)}\n{body}\n{self._rule(f)}"
        msgs = [(f["inspector"], f"just tipped: {f['route']} ({f['hauler']}), {f['stream']}, {f['tons']} t"),
                (f["inspector"], f"contamination {self._est(f)}" + ("" if f["photos"] else ", no photos, camera's dead")),
                (f["inspector"], f"prohibited: {self._item(f)}")]
        if disp_txt:
            msgs.append((f["inspector"], "driver's saying it's all from one commercial stop"))
        msgs.append((f["supervisor"], f"how many rejections on that route lately?"))
        msgs.append((f["inspector"], f"{f['prior_rejections_90d']} in the last 90 days"))
        t = f["ts"]
        out = []
        for who, m in msgs:
            out.append(f"{t.strftime('%H:%M')} {_first(who)}: {noisy(m, rng, 0.1)}")
            t += timedelta(minutes=rng.randint(1, 4))
        return f"[tip floor channel -- {f['mrf']}]\n" + "\n".join(out) + f"\n(pinned) {self._rule(f)}"

    def questions(self):
        return {
            "load_disposition": choice("How should the facility dispose of this load?", {
                "accept_standard": "Accept the load and process it normally.",
                "accept_with_contamination_fee": "Accept the load and charge the agreement's contamination fee.",
                "audit_sample_before_ruling": "Pull and weigh a sample before ruling on the load.",
                "isolate_hazard_then_process": "Pull out the hazardous item safely, then process the load.",
                "reject_as_refuse": "Reject the load and tip it as refuse at the hauler's cost.",
            }),
            "hauler_follow_up": choice("What follow-up with the hauler does this load call for?", {
                "cart_tagging_outreach": "Ask the hauler to tag carts and educate households on the route.",
                "formal_contract_notice": "Issue a formal notice under the agreement.",
                "none_needed": "Nothing beyond the normal ticket.",
                "written_warning": "Send a written warning about this load.",
            }),
            "estimate_reliability": score("How reliable is the contamination estimate?", [
                "Weak: an unsupported glance at the pile.",
                "Fair: a visual or camera estimate with some backing.",
                "Good: a camera estimate with photos, or a small weighed sample.",
                "Strong: a properly sized weighed audit.",
            ]),
            "worker_safety_hazard": score("How much does the load endanger sorters and equipment?", [
                "None beyond normal.",
                "Some: material that wraps or jams equipment.",
                "Serious: sharps or batteries that could injure or ignite.",
                "Severe: a pressurised cylinder or a battery already smoking.",
            ]),
            "exceeds_contract_threshold": noul(
                "The load's contamination is at or above the agreement's rejection threshold.",
                yes="The estimate on the ticket is at or above the rejection figure.",
                no="The estimate is below the rejection figure."),
        }


# =========================================================================== schema 4
STRUCTS = {"occupied site office": True, "scale house": True, "row of homes": True, "unoccupied storage shed": False,
           "maintenance garage": True}
WELLFIELD = {"normal": None, "watered_in": "two nearby extraction wells watered in", "blower_down": "blower down",
             "new_cell": "new cell, collection not yet installed"}


class LandfillGasProbe(Schema):
    name = "landfill_gas_probe_exceedance"
    description = "A perimeter gas-probe reading at a landfill, judged against the site gas-management plan's action level."
    surfaces = {"json": 0.5, "monitoring_sheet": 0.25, "env_tech_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        A = pick(rng, [1.0, 1.25, 2.5])
        S = pick(rng, [150, 250, 300])
        ch4 = round(A * rng.uniform(0.1, 1.8), 2)
        prev = round(A * rng.uniform(0.0, 1.3), 2)
        o2 = round(rng.uniform(0.5, 4.0), 1)
        cal = pick_w(rng, {"calibrated today": 5, "calibration overdue": 2})
        purge = pick_w(rng, {"yes": 5, "no": 2, "not recorded": 2})
        baro = pick(rng, ["falling fast", "steady", "rising"])
        well = pick_w(rng, {"normal": 6, "watered_in": 1.5, "blower_down": 1.5, "new_cell": 1})
        struct = pick(rng, list(STRUCTS))
        dist = int(S * rng.uniform(0.3, 2.0))
        k = None
        if reg == "clear":
            cal, purge, o2 = "calibrated today", "yes", round(rng.uniform(0.3, 2.5), 1)
            if chance(rng, 0.5):
                ch4, prev, well = round(A * rng.uniform(0.0, 0.4), 2), round(A * rng.uniform(0.0, 0.4), 2), "normal"
            else:
                ch4, prev = round(A * rng.uniform(1.6, 4.0), 2), round(A * rng.uniform(1.2, 3.0), 2)
        elif reg == "borderline":
            k = pick(rng, ["action_level", "structure_distance", "prev_month"])
            if k == "action_level":
                ch4 = near(rng, A, A * 0.12, lo=0)
            elif k == "structure_distance":
                ch4 = round(A * rng.uniform(1.05, 1.6), 2)
                dist = near(rng, S, S * 0.12, lo=10, integer=True)
            else:
                ch4 = round(A * rng.uniform(1.05, 1.6), 2)
                prev = near(rng, A, A * 0.12, lo=0)
        else:
            k = pick(rng, ["air_leak_high", "baro_spike", "wellfield_down", "uncalibrated_low"])
            if k == "air_leak_high":
                ch4, o2 = round(A * rng.uniform(1.1, 2.0), 2), round(rng.uniform(9, 17), 1)
            elif k == "baro_spike":
                ch4, prev, baro = round(A * rng.uniform(1.05, 1.8), 2), round(A * rng.uniform(0, 0.5), 2), "falling fast"
                purge = pick(rng, ["no", "not recorded", "yes"])
            elif k == "wellfield_down":
                well = pick(rng, ["blower_down", "watered_in"])
                ch4 = round(A * rng.uniform(1.05, 2.5), 2)
            else:
                cal, ch4 = "calibration overdue", round(A * rng.uniform(0.6, 0.98), 2)
                prev = round(A * rng.uniform(1.0, 1.6), 2)
        return {"regime": reg, "case_kind": k, "site": _site(rng, pick(rng, ["Landfill", "Sanitary Landfill", "Recycling and Disposal Facility"])),
                "probe": f"GP-{rng.randint(1, 38):02d}", "ch4_pct": ch4, "prev_ch4_pct": prev, "co2_pct": round(rng.uniform(0.2, 9.0), 1),
                "o2_pct": o2, "calibration": cal, "purged": purge, "barometric": baro, "wellfield": well,
                "blower_hours": rng.randint(3, 40) if well == "blower_down" else None, "structure": struct,
                "structure_m": dist, "action_pct": A, "structure_radius_m": S, "tech": person(rng), "engineer": person(rng),
                "neighbour_probe": f"GP-{rng.randint(39, 60):02d}", "neighbour_ch4": round(max(0.0, ch4 * rng.uniform(0.0, 0.9)), 2),
                "depth_m": pick(rng, [3, 4.5, 6, 9]), "static_kpa": round(rng.uniform(-0.3, 0.6), 2),
                "date": rand_date(rng), "plan": pick(rng, ["LGMP rev 4", "Gas Management Plan s3", "LFG-MP 2"])}

    def _valid(self, f):
        return f["calibration"] == "calibrated today" and f["purged"] == "yes" and f["o2_pct"] < 6

    def intended(self, f):
        A = f["action_pct"]
        over = f["ch4_pct"] >= A
        valid = self._valid(f)
        exceed = over and valid
        if over and not valid:
            resp = "resample_to_confirm"
        elif exceed and f["prev_ch4_pct"] >= A:
            resp = "prepare_remediation_plan"
        elif exceed:
            resp = "tune_wellfield_and_resample"
        elif f["calibration"] != "calibrated today" and f["ch4_pct"] >= 0.6 * A:
            resp = "resample_to_confirm"
        else:
            resp = "routine_monitoring"
        r = f["ch4_pct"] / A
        mr = 0 if r < 0.5 else 1 if r < 1 else 2 if r < 2 else 3
        if f["wellfield"] != "normal" and mr in (1, 2):
            mr += 1
        if not valid and mr == 3:
            mr = 2
        rep = 3
        if f["calibration"] != "calibrated today":
            rep -= 1
        if f["purged"] != "yes":
            rep -= 1
        if f["o2_pct"] >= 6:
            rep -= 2
        occupied_close = STRUCTS[f["structure"]] and f["structure_m"] <= f["structure_radius_m"]
        return {"probe_response": resp, "migration_risk": mr, "reading_representativeness": max(0, rep),
                "action_level_exceeded": "yes" if exceed else "no",
                "structure_check_required": "yes" if (over and occupied_close) else "no"}

    def _rule(self, f):
        return (f"{f['plan']}: perimeter probe action level {f['action_pct']}% CH4; confirm within 48 h, adjust collection "
                f"within 7 days; check occupied structures within {f['structure_radius_m']} m within 24 h; two consecutive "
                "monthly exceedances = remediation plan")

    def _wf(self, f):
        w = WELLFIELD[f["wellfield"]]
        if f["wellfield"] == "blower_down":
            w += f" {f['blower_hours']} h"
        return w

    def render(self, f, rng, surface):
        if surface == "json":
            return dumps(prune({
                "probe": {"id": f["probe"], "site": f["site"], "date": f["date"].isoformat(), "depth_m": f["depth_m"],
                          "ch4_pct": f["ch4_pct"], "co2_pct": f["co2_pct"], "o2_pct": f["o2_pct"],
                          "static_pressure_kpa": f["static_kpa"], "previous_month_ch4_pct": f["prev_ch4_pct"]},
                "next_probe_along_boundary": {"id": f["neighbour_probe"], "ch4_pct": f["neighbour_ch4"]},
                "sampling": {"instrument": f["calibration"], "probe_purged": f["purged"], "barometric_trend": f["barometric"]},
                "surroundings": {"nearest_structure": f"{f['structure']}, {f['structure_m']} m", "wellfield": self._wf(f)},
                "plan": self._rule(f),
            }))
        if surface == "monitoring_sheet":
            return "\n".join([
                f"PERIMETER GAS MONITORING -- {f['site']}  {fmt_date(f['date'], rng)}  ({f['tech']})",
                table(["probe", "depth m", "CH4 %", "CO2 %", "O2 %", "static kPa", "last month CH4 %"],
                      [[f["probe"], f["depth_m"], f["ch4_pct"], f["co2_pct"], f["o2_pct"], f["static_kpa"], f["prev_ch4_pct"]],
                       [f["neighbour_probe"], "", f["neighbour_ch4"], "", "", "", ""]], rng),
                _kv([("Meter", f["calibration"]), ("Purged before reading", f["purged"]),
                          ("Barometer", f["barometric"]), ("Nearest structure", f"{f['structure']} at {f['structure_m']} m"),
                          ("Wellfield", self._wf(f) or "normal")], rng),
                f"{f['plan']}: action level {f['action_pct']}% CH4; occupied structures within {f['structure_radius_m']} m "
                "checked within 24 h of an exceedance; 2 consecutive monthly exceedances = remediation plan"])
        s = [f"{f['probe']} read {f['ch4_pct']}% CH4, {f['co2_pct']}% CO2 and {f['o2_pct']}% O2 today; last month it was {f['prev_ch4_pct']}% CH4.",
             f"Meter {f['calibration']}; probe purged: {f['purged']}; barometer {f['barometric']}.",
             f"Next probe along the boundary ({f['neighbour_probe']}) read {f['neighbour_ch4']}%.",
             f"Nearest structure is {'an' if f['structure'][0] in 'aeiou' else 'a'} {f['structure']} about {f['structure_m']} m away.",
             (f"Wellfield: {self._wf(f)}." if self._wf(f) else "Wellfield running normally.")]
        return "\n".join([f"Subject: {f['site']} -- {f['probe']} reading", f"From: {f['tech']}", f"To: {f['engineer']}", "",
                          join_sentences(s), "", f"Plan says: {self._rule(f)}.", "", _first(f["tech"])])

    def questions(self):
        return {
            "probe_response": choice("What does the gas plan call for at this probe?", {
                "prepare_remediation_plan": "Prepare a formal remediation plan for the gas migration.",
                "resample_to_confirm": "Resample with a valid method to confirm the reading.",
                "routine_monitoring": "Keep the probe on the routine monitoring schedule.",
                "tune_wellfield_and_resample": "Adjust the gas collection near the probe and resample.",
            }),
            "migration_risk": score("How likely is landfill gas to be migrating past the boundary at this probe?", [
                "Unlikely: readings low and collection working.",
                "Possible: some methane, below the action level.",
                "Likely: above the action level, or near it with collection impaired.",
                "Very likely: well above the action level or with collection down.",
            ]),
            "reading_representativeness": score("How representative of real probe conditions is the reading?", [
                "Not at all: the sample was mostly air or the method failed.",
                "Doubtful: more than one sampling problem.",
                "Mostly: one sampling weakness.",
                "Fully: calibrated meter, purged probe, normal oxygen.",
            ]),
            "action_level_exceeded": noul(
                "A valid reading at the probe is at or above the plan's action level.",
                yes="The reading is at or above the action level and was taken properly.",
                no="The reading is below the action level, or only an invalid reading exceeds it."),
            "structure_check_required": noul("Occupied structures near the probe must be checked for gas now."),
        }


# =========================================================================== schema 5
INCIDENTS = ["hot_load", "unknown_container", "gas_cylinder", "suspected_asbestos", "sharps_injury"]


class TransferStationIncident(Schema):
    name = "transfer_station_incident"
    description = "An incident on a transfer station's tipping floor, handled under the station operating procedure."
    surfaces = {"json": 0.5, "incident_form": 0.25, "supervisor_radio": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        kind = pick(rng, INCIDENTS, [3, 2, 2, 2, 1])
        T = pick(rng, [60, 70, 80])
        d = {"temp_c": None, "smoke": None, "flame": False, "move_min": None, "label": None, "leaking": False,
             "size": None, "valve_damaged": False, "cyl_type": None, "friable": None, "volume_m3": None, "injury": None}
        if kind == "hot_load":
            d.update(temp_c=int(T * rng.uniform(0.6, 2.2)), smoke=pick(rng, ["none", "light", "heavy"], [3, 3, 1]),
                     flame=chance(rng, 0.12), move_min=rng.randint(4, 25))
        elif kind == "unknown_container":
            d.update(label=pick(rng, ["no label", "label illegible", "'corrosive' label", "'flammable' label"]),
                     leaking=chance(rng, 0.3), size=pick(rng, ["5-gal pail", "55-gal drum"]))
        elif kind == "gas_cylinder":
            d.update(cyl_type=pick(rng, ["20 lb propane cylinder", "unmarked industrial gas cylinder", "helium balloon tank",
                                         "acetylene cylinder"]),
                     valve_damaged=chance(rng, 0.3))
        elif kind == "suspected_asbestos":
            d.update(friable=chance(rng, 0.45), volume_m3=round(rng.uniform(0.2, 6.0), 1))
        else:
            d.update(injury=pick(rng, ["first aid on site", "sent to clinic"]))
        hauler = pick_w(rng, {"commercial": 5, "self_haul": 3, "unknown": 2})
        prior = pick(rng, [0, 0, 0, 1, 2, 3]) if hauler == "commercial" else 0
        k = None
        if reg == "clear":
            if kind == "hot_load":
                if chance(rng, 0.5):
                    d.update(temp_c=int(T * rng.uniform(0.4, 0.7)), smoke="none", flame=False)
                else:
                    d.update(temp_c=int(T * rng.uniform(1.8, 3.0)), smoke="heavy", flame=True)
            elif kind == "unknown_container":
                d.update(leaking=True, label="'corrosive' label")
        elif reg == "borderline":
            k = "near_rule"
            if kind == "hot_load":
                d.update(temp_c=near(rng, T, 6, lo=30, integer=True), smoke=pick(rng, ["none", "light"]), flame=False,
                         move_min=near(rng, 10, 3, lo=2, integer=True))
            elif kind == "suspected_asbestos":
                d.update(volume_m3=near(rng, 1.0, 0.4, lo=0.1, digits=1))
        else:
            k = pick(rng, ["calm_report_bad_facts", "alarming_report_minor_facts"])
            if kind == "hot_load" and k == "calm_report_bad_facts":
                d.update(temp_c=int(T * rng.uniform(1.1, 1.6)), smoke="light", move_min=rng.randint(12, 25))
            elif kind == "unknown_container" and k == "calm_report_bad_facts":
                d.update(leaking=True, label=pick(rng, ["no label", "label illegible"]))
        return {"regime": reg, "case_kind": k, "kind": kind, **d, "hot_rule_c": T, "hauler": hauler,
                "hauler_name": _hauler(rng) if hauler == "commercial" else None, "prior_incidents": prior,
                "queue": rng.randint(0, 14), "sprinklers_ok": chance(rng, 0.85), "minutes_to_close": rng.randint(20, 400),
                "station": _site(rng, "Transfer Station"), "supervisor": person(rng), "operator": person(rng),
                "ts": rand_ts(rng, hours=(6, 17)), "sop": pick(rng, ["SOP-9", "Station Operating Procedure 4", "TS-OPS 12"]),
                "tone": ("calm" if k == "calm_report_bad_facts" else "alarmed" if k == "alarming_report_minor_facts"
                         else "plain" if reg == "clear" else pick(rng, ["calm", "alarmed", "plain"]))}

    def intended(self, f):
        kind, T = f["kind"], f["hot_rule_c"]
        if kind == "hot_load":
            if f["flame"] or f["smoke"] == "heavy" or (f["temp_c"] >= T and f["move_min"] > 10):
                resp = "call_fire_service"
            elif f["temp_c"] >= T or f["smoke"] == "light":
                resp = "isolate_and_monitor"
            elif f["temp_c"] >= 0.8 * T:
                resp = "isolate_and_monitor"
            else:
                resp = "note_and_continue"
        elif kind == "unknown_container":
            resp = "call_hazmat_responder" if (f["leaking"] or "'" in f["label"]) else "isolate_and_monitor"
        elif kind == "gas_cylinder":
            resp = "stop_tipping_and_clear_floor" if f["valve_damaged"] else "isolate_and_monitor"
        elif kind == "suspected_asbestos":
            if f["friable"] and f["volume_m3"] >= 1.0:
                resp = "call_hazmat_responder"
            elif f["friable"]:
                resp = "stop_tipping_and_clear_floor"
            else:
                resp = "isolate_and_monitor"
        else:
            resp = "note_and_continue"
        big = resp in ("call_fire_service", "call_hazmat_responder")
        if f["hauler"] != "commercial":
            cost = "handling_fee" if (f["hauler"] == "self_haul" and resp != "note_and_continue") else "no_charge"
        elif kind == "sharps_injury" or (kind == "hot_load" and resp == "note_and_continue"):
            cost = "no_charge" if f["prior_incidents"] < 2 else "handling_fee"
        elif f["prior_incidents"] >= 2 or (kind == "suspected_asbestos" and f["friable"]):
            cost = "refer_for_enforcement"
        elif big:
            cost = "full_cleanup_cost_to_hauler"
        else:
            cost = "handling_fee"
        hz = {"note_and_continue": 0, "isolate_and_monitor": 1, "stop_tipping_and_clear_floor": 2,
              "call_hazmat_responder": 2, "call_fire_service": 3}[resp]
        if kind == "hot_load" and resp == "isolate_and_monitor" and f["temp_c"] >= T and not f["sprinklers_ok"]:
            hz = 2
        if kind == "unknown_container" and f["leaking"] and "flammable" in f["label"]:
            hz = 3
        if kind == "sharps_injury":
            hz = 1 if f["injury"] == "sent to clinic" else 0
        dis = {"note_and_continue": 0, "isolate_and_monitor": 1, "stop_tipping_and_clear_floor": 2,
               "call_hazmat_responder": 2, "call_fire_service": 3}[resp]
        if f["queue"] >= 8 and dis in (1, 2):
            dis += 1
        if f["minutes_to_close"] < 45 and dis >= 2:
            dis -= 1
        report = (resp == "call_fire_service" or (kind == "sharps_injury" and f["injury"] == "sent to clinic")
                  or (kind == "unknown_container" and f["leaking"]) or (kind == "suspected_asbestos" and f["friable"])
                  or (kind == "gas_cylinder" and f["valve_damaged"] and f["cyl_type"] != "helium balloon tank"))
        return {"immediate_response": resp, "cost_recovery": cost, "floor_hazard": hz, "operating_disruption": dis,
                "authority_report_required": "yes" if report else "no"}

    def _rule(self, f):
        if f["kind"] == "hot_load":
            core = (f"hot load = surface >= {f['hot_rule_c']} C or smoke: push to the hot-load pad and wet down; "
                    "call the fire service for flame, heavy smoke, or a hot load that cannot be outside within 10 min")
        elif f["kind"] == "unknown_container":
            core = "unknown containers: isolate in the quarantine bay unopened; leaking or hazard-labelled ones go to the contracted hazmat responder"
        elif f["kind"] == "gas_cylinder":
            core = "cylinders: move to the cylinder cage; a damaged valve means stop tipping and clear the floor"
        elif f["kind"] == "suspected_asbestos":
            core = "suspected asbestos: wet and wrap if non-friable; friable stops tipping, and >= 1.0 m3 friable goes to the hazmat responder"
        else:
            core = "sharps: first aid, then clinic if the skin is broken; keep the item"
        return (f"{f['sop']}: {core}. Report to the county waste authority within 24 h any fire-service call-out, injury "
                "treated off site, or hazardous release")

    def _what(self, f):
        k = f["kind"]
        if k == "hot_load":
            s = f"hot load, IR gun {f['temp_c']} C, smoke {f['smoke']}" + (", open flame" if f["flame"] else "")
            return s + f"; loader estimates {f['move_min']} min to get it outside"
        if k == "unknown_container":
            return f"{f['size']} with {f['label']}" + (", leaking" if f["leaking"] else ", not leaking")
        if k == "gas_cylinder":
            return f"{f['cyl_type']} in the load" + (", valve damaged and hissing" if f["valve_damaged"] else ", valve intact")
        if k == "suspected_asbestos":
            return (f"suspected asbestos, about {f['volume_m3']} m3, " + ("crumbly lagging (friable)" if f["friable"] else "intact cement sheeting"))
        return f"operator stuck by a needle while hand-picking; {f['injury']}"

    def _hauler(self, f):
        if f["hauler"] == "commercial":
            return f"{f['hauler_name']} (commercial account, {f['prior_incidents']} prior incident(s) this year)"
        return "self-haul resident, still on site" if f["hauler"] == "self_haul" else "not identified"

    def render(self, f, rng, surface):
        tone = {"calm": "Probably nothing, but logging it.", "alarmed": "Need someone down here now!",
                "plain": ""}[f["tone"]]
        if surface == "json":
            return dumps(prune({
                "incident": {"station": f["station"], "time": fmt_ts(f["ts"], style="iso"), "type": f["kind"].replace("_", " "),
                             "details": self._what(f), "reported_by": f["operator"], "operator_comment": tone or None},
                "source": {"hauler": self._hauler(f)},
                "floor": {"trucks_queued": f["queue"], "sprinklers_in_service": f["sprinklers_ok"],
                          "minutes_to_closing": f["minutes_to_close"]},
                "procedure": self._rule(f),
            }))
        if surface == "incident_form":
            body = _kv([("Station", f["station"]), ("Time", fmt_ts(f["ts"], rng)), ("Reported by", f["operator"]),
                             ("What happened", self._what(f)), ("Load from", self._hauler(f)),
                             ("Trucks queued", f["queue"]), ("Sprinklers", "in service" if f["sprinklers_ok"] else "OUT OF SERVICE"),
                             ("Minutes to closing", f["minutes_to_close"])], rng)
            return f"TIPPING FLOOR INCIDENT FORM\n{body}\nOperator comment: {tone or '-'}\nProcedure: {self._rule(f)}"
        t = f["ts"]
        out = [f"{t.strftime('%H:%M')} {_first(f['operator']).upper()}: {self._what(f)}. {tone}".rstrip(),
               f"{(t + timedelta(minutes=1)).strftime('%H:%M')} {_first(f['supervisor']).upper()}: whose load?",
               f"{(t + timedelta(minutes=2)).strftime('%H:%M')} {_first(f['operator']).upper()}: {self._hauler(f)}",
               f"{(t + timedelta(minutes=3)).strftime('%H:%M')} {_first(f['supervisor']).upper()}: {f['queue']} trucks in the queue, "
               + ("sprinklers are up" if f["sprinklers_ok"] else "and the sprinklers are still out of service")
               + f", {f['minutes_to_close']} min to close"]
        return f"{f['station']} -- floor radio\n" + "\n".join(out) + f"\n[{self._rule(f)}]"

    def questions(self):
        return {
            "immediate_response": choice("What should the floor supervisor do first under the procedure?", {
                "call_fire_service": "Call the fire service.",
                "call_hazmat_responder": "Call the contracted hazardous-materials responder.",
                "isolate_and_monitor": "Move the item or load to its isolation area and watch it.",
                "note_and_continue": "Record it and carry on operating normally.",
                "stop_tipping_and_clear_floor": "Stop all tipping and clear people off the floor.",
            }),
            "cost_recovery": choice("How should the station recover the cost of this incident?", {
                "full_cleanup_cost_to_hauler": "Bill the hauler the full response and clean-up cost.",
                "handling_fee": "Charge the standard special-handling fee.",
                "no_charge": "No charge.",
                "refer_for_enforcement": "Refer the hauler to the county for enforcement.",
            }),
            "floor_hazard": score("How hazardous is the situation on the tipping floor?", [
                "Minimal: no ongoing danger.",
                "Contained: a hazard that is controlled once it is isolated.",
                "Serious: a hazard that needs the floor cleared or outside help.",
                "Severe: fire or a dangerous release in progress.",
            ]),
            "operating_disruption": score("How much will the response disrupt today's operation?", [
                "Negligible.",
                "Brief: a short pause on part of the floor.",
                "Substantial: tipping stops for a while and trucks back up.",
                "Major: the station is effectively shut for hours.",
            ]),
            "authority_report_required": noul(
                "The procedure requires a report to the county waste authority.",
                yes="The incident involved a fire-service call-out, an injury treated off site, or a hazardous release.",
                no="None of the reportable events occurred."),
        }


WORKFLOW = Workflow(
    name="waste_management_operations",
    title="Waste management operations",
    description="Landfill acceptance, hazardous-waste receiving, recycling contamination, landfill gas and transfer-station "
                "incidents, against fictional plans and procedures.",
    domain="waste collection, treatment and disposal (NAICS 562)",
    schemas=[WasteAcceptanceProfile(), ManifestDiscrepancy(), RecyclingLoadContamination(), LandfillGasProbe(),
             TransferStationIncident()],
    notes="Operational acceptance and response only; facility permits and discharge reports are covered by "
          "environmental_compliance_reports. All plans, limits and organisations are fictional.",
)
