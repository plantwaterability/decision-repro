"""Generator interconnection review: queue requests judged against a (fictional) transmission provider's written tariff.

Decisions a transmission provider's interconnection team takes on generator interconnection requests (solar, wind,
storage, hybrids, gas peakers): whether a new request is valid and complete before a cluster window closes, whether
a requested modification is material and what happens to the queue position, how a network-upgrade cost is shared
in a cluster, whether a request may enter Phase 2 on its commercial-readiness evidence, and what withdrawal penalty
applies. Providers, projects, substations and every tariff section are invented, shaped like real generator
interconnection procedures; thresholds vary between providers so the rule quoted in the state has to be read. Each
schema draws a `regime` first so most cases sit near a threshold, carry signals that disagree (a developer's
cover letter against the record), or miss a key fact.

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
    ident,
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
PROV_A = ["Tri-Basin", "High Plains", "Copperline", "Northern Tier", "Saltmarsh", "Ashgrove", "Bluestem", "Red Mesa",
          "Lakeshore", "Granite Range", "Pinecrest", "Yarrow Valley", "Cold River", "Sandhill", "Two Forks"]
PROV_B = ["Transmission Authority", "Grid Operator", "Transmission Cooperative", "Independent System Operator",
          "Interconnection Services", "Power Transmission Co."]
PLACE_A = ["Antelope", "Badger", "Cottonwood", "Dry Creek", "Eagle", "Foxtail", "Greasewood", "Hackberry", "Ironwood",
           "Jackrabbit", "Lone Pine", "Mesquite", "Nighthawk", "Owl Creek", "Prairie", "Quartz", "Rattlesnake",
           "Sagebrush", "Tumbleweed", "Vermilion", "Whitetail", "Yucca", "Bitterroot", "Chalk Bluff", "Saddle Horn"]
PLACE_B = ["Flats", "Ridge", "Draw", "Mesa", "Butte", "Valley", "Springs", "Basin", "Bend", "Hills", "Gulch", "Prairie"]
DEV_B = ["Renewables LLC", "Energy Partners", "Power Holdings", "Clean Power Co.", "Energy Ventures", "Generation LLC",
         "Power Development LLC", "Infrastructure Partners"]
PROJ_SUFFIX = {"solar_pv": ["Solar", "Solar Project", "Solar Energy Center"], "wind": ["Wind", "Wind Farm", "Wind Energy Center"],
               "battery_storage": ["Storage", "Battery Storage", "Energy Storage"],
               "solar_plus_storage": ["Solar + Storage", "Hybrid", "Solar and Storage"],
               "gas_peaker": ["Peaking Station", "Generating Station", "Peaker"]}
TECH = {
    # tech: (label, MW range, acres per MW used by most providers' site-control tables)
    "solar_pv": ("solar PV", (20, 400), 6.0),
    "wind": ("wind", (60, 500), 40.0),
    "battery_storage": ("battery storage (4 h)", (20, 300), 0.3),
    "solar_plus_storage": ("solar PV + battery hybrid", (50, 400), 6.5),
    "gas_peaker": ("gas combustion turbine", (90, 450), 0.15),
}
KV_FOR_MW = [(80, [69, 115]), (250, [115, 138, 230]), (600, [230, 345]), (9999, [345, 500])]


def _provider(rng):
    return f"{pick(rng, PROV_A)} {pick(rng, PROV_B)}"


def _place(rng):
    return f"{pick(rng, PLACE_A)} {pick(rng, PLACE_B)}"


def _developer(rng, place):
    return f"{place.split()[0]} {pick(rng, DEV_B)}" if chance(rng, 0.5) else f"{pick(rng, PLACE_A)} {pick(rng, DEV_B)}"


def _project(rng, tech, place=None):
    return f"{place or _place(rng)} {pick(rng, PROJ_SUFFIX[tech])}"


def _mw(rng, tech):
    lo, hi = TECH[tech][1]
    return int(round(rng.uniform(lo, hi) / 5.0) * 5)


def _kv(rng, mw, mismatch=False):
    for cap, kvs in KV_FOR_MW:
        if mw <= cap:
            return pick(rng, kvs) if not mismatch else pick(rng, [69, 115]) if mw > 250 else pick(rng, [345, 500])
    return 345


def _queue_id(rng, year):
    return pick(rng, [f"Q{year % 100}-{rng.randint(100, 999)}", f"GI-{year}-{rng.randint(1, 499):04d}",
                      f"J{rng.randint(1100, 2999)}"])


def _poi(rng, kv):
    if chance(rng, 0.6):
        return f"{pick(rng, PLACE_A)} {kv} kV substation"
    return f"tap on the {pick(rng, PLACE_A)}-{pick(rng, PLACE_A)} {kv} kV line"


def add_business_days(d: date, n: int) -> date:
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def business_days_between(a: date, b: date) -> int:
    n, d = 0, a
    while d < b:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return n


def _weekday(rng, start=date(2025, 1, 6), end=date(2027, 6, 30)):
    d = rand_date(rng, start, end)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


DATE_STYLE = ["iso", "iso", "us", "long", "eu"]


def _r(x, nd=0):
    """Round to `nd` digits (negative = tens, hundreds...) and return an int: money is whole dollars."""
    return int(round(x, nd))


def _musd(x):
    return f"${x / 1e6:,.1f}M" if x >= 1e6 else f"${x:,.0f}"


# =========================================================================== schema 1
SITE_DOCS = {
    "lease": "executed lease", "option": "executed option to lease", "deed": "recorded deed",
    "loi": "letter of intent with landowners (non-binding)", "unsigned_exclusivity": "exclusivity agreement, landowner signature pending",
}
ONE_LINE = {"stamped": "stamped by a licensed engineer", "preliminary": "preliminary, not stamped", "missing": "not provided"}
DYN_MODEL = {"validated": "validated library model with parameters", "user_written_only": "user-written model only, no library equivalent",
             "placeholder": "generic placeholder model (vendor not selected)", "missing": "not provided"}


class RequestCompletenessReview(Schema):
    name = "request_completeness_review"
    description = "A new generator interconnection request checked for validity and completeness before the cluster window closes."
    surfaces = {"json": 0.5, "intake_checklist": 0.25, "applicant_letter": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        tech = pick_w(rng, {"solar_pv": 4, "wind": 2, "battery_storage": 3, "solar_plus_storage": 3, "gas_peaker": 1})
        mw = _mw(rng, tech)
        kv_mismatch = chance(rng, 0.12)
        term_req = pick(rng, [3, 3, 5])
        cure_days = pick(rng, [10, 10, 15])
        dep_base, dep_per_mw, dep_cap = pick(rng, [35000, 50000, 75000]), pick(rng, [1000, 1500, 2000]), pick(rng, [250000, 400000, None])
        dep_req = dep_base + dep_per_mw * mw
        if dep_cap:
            dep_req = min(dep_req, dep_cap)
        acres_per_mw = round(TECH[tech][2] * pick(rng, [0.9, 1.0, 1.0, 1.1]), 2)
        req_acres = round(acres_per_mw * mw, 1)
        site_doc = pick_w(rng, {"lease": 4, "option": 5, "deed": 1, "loi": 0.7, "unsigned_exclusivity": 0.7})
        coverage = pick_w(rng, {round(rng.uniform(1.0, 1.6), 3): 5, round(rng.uniform(0.85, 0.999), 3): 3, round(rng.uniform(0.4, 0.84), 3): 1})
        term = pick_w(rng, {term_req + rng.randint(0, 25): 8, max(1, term_req - rng.randint(1, 2)): 1})
        paid = dep_req if chance(rng, 0.65) else int(round(dep_req - pick_w(rng, {rng.uniform(150, 2500): 2, rng.uniform(dep_req * 0.2, dep_req * 0.6): 1}), -1))
        one_line = pick_w(rng, {"stamped": 5, "preliminary": 3, "missing": 1})
        model = pick_w(rng, {"validated": 4, "user_written_only": 2, "placeholder": 2, "missing": 1})
        stage = pick_w(rng, {"initial_submission": 3, "cure_response": 1})
        today = _weekday(rng)
        bd_left = pick_w(rng, {rng.randint(cure_days + 2, cure_days + 30): 4, rng.randint(max(1, cure_days - 4), cure_days + 3): 3,
                               rng.randint(1, max(2, cure_days - 5)): 2})
        if reg == "clear":
            if chance(rng, 0.5):
                site_doc, coverage, term, paid, one_line, model, kv_mismatch = "lease", round(rng.uniform(1.05, 1.5), 3), term_req + rng.randint(5, 25), dep_req, "stamped", "validated", False
                stage = "initial_submission"
            else:
                site_doc, paid, model = pick(rng, ["loi", "unsigned_exclusivity"]), int(round(dep_req * rng.uniform(0.3, 0.6), -1)), pick(rng, ["missing", "placeholder"])
        elif reg == "borderline":
            coverage = round(near(rng, 1.0, 0.06, lo=0.9), 3)
            if chance(rng, 0.5):
                term = term_req
        else:
            # strong technical package and a confident cover letter, against one basic gap
            one_line, model = "stamped", "validated"
            gap = pick(rng, ["site", "deposit", "term"])
            if gap == "site":
                site_doc = pick(rng, ["loi", "unsigned_exclusivity"])
            elif gap == "deposit":
                paid = int(round(dep_req - rng.uniform(dep_req * 0.05, dep_req * 0.4), -1))
            else:
                term = max(1, term_req - 1)
        acres = round(req_acres * coverage, 1)
        close = add_business_days(today, bd_left)
        notice_deadline = today - timedelta(days=pick(rng, [0, 1, 3])) if stage == "cure_response" else None
        place = _place(rng)
        kv = _kv(rng, mw, kv_mismatch)
        return {"regime": reg, "provider": _provider(rng), "project": _project(rng, tech, place),
                "developer": _developer(rng, place), "tech": tech, "mw": mw, "poi_kv": kv, "poi": _poi(rng, kv),
                "kv_mismatch": kv_mismatch, "term_req_years": term_req, "cure_days": cure_days,
                "dep_base": dep_base, "dep_per_mw": dep_per_mw, "dep_cap": dep_cap, "dep_required": dep_req,
                "dep_paid": paid, "acres_per_mw": acres_per_mw, "req_acres": req_acres, "site_doc": site_doc,
                "site_acres": acres, "site_term_years": term, "one_line": one_line, "dyn_model": model, "stage": stage,
                "today": today, "window_close": close, "bd_left": bd_left, "notice_deadline": notice_deadline,
                "queue_id": _queue_id(rng, today.year), "reviewer": person(rng), "contact": person(rng),
                "cluster": f"Cluster {pick(rng, [5, 6, 7, 8, 9, 11])}", "letter_tone": pick(rng, ["confident", "apologetic", "plain"])}

    def _gravity(self, f):
        g = 0
        if f["site_doc"] in ("loi", "unsigned_exclusivity"):
            g = 3
        elif f["site_acres"] < f["req_acres"]:
            g = max(g, 2 if f["site_acres"] >= 0.9 * f["req_acres"] else 3)
        if f["site_term_years"] < f["term_req_years"]:
            g = max(g, 2)
        short = f["dep_required"] - f["dep_paid"]
        if short > 0:
            g = max(g, 1 if short <= 0.02 * f["dep_required"] else 3)
        g = max(g, {"stamped": 0, "preliminary": 1, "missing": 2}[f["one_line"]])
        g = max(g, {"validated": 0, "user_written_only": 1, "placeholder": 2, "missing": 2}[f["dyn_model"]])
        return g

    def intended(self, f):
        g = self._gravity(f)
        site_ok = (f["site_doc"] in ("lease", "option", "deed") and f["site_acres"] >= f["req_acres"]
                   and f["site_term_years"] >= f["term_req_years"])
        fits = f["cure_days"] <= f["bd_left"]
        if g == 0:
            v = "accept_as_valid"
        elif f["stage"] == "cure_response":
            v = "deem_withdrawn"
        elif fits:
            v = "issue_deficiency_notice"
        else:
            v = "notice_and_next_cluster"
        tq = 3
        tq -= {"stamped": 0, "preliminary": 1, "missing": 2}[f["one_line"]]
        tq -= {"validated": 0, "user_written_only": 1, "placeholder": 1, "missing": 2}[f["dyn_model"]]
        tq -= 1 if f["kv_mismatch"] else 0
        return {"validity_determination": v, "site_control_demonstrated": "yes" if site_ok else "no",
                "technical_package_quality": max(0, tq), "deficiency_gravity": g,
                "cure_fits_before_close": "yes" if fits else "no"}

    def _rules(self, f):
        cap = f", capped at {_musd(f['dep_cap'])}" if f["dep_cap"] else ""
        return (f"GIP s.3.2: a valid request needs (a) a study deposit of {_musd(f['dep_base'])} plus {_musd(f['dep_per_mw'])}/MW{cap}; "
                f"(b) exclusive site control by deed, lease or option over at least {f['acres_per_mw']} acres per MW for "
                f"{TECH[f['tech']][0]}, for a term of at least {f['term_req_years']} years; (c) a one-line diagram and dynamic "
                f"models usable in the provider's study software. s.3.4: gaps are listed in a deficiency notice and may be cured "
                f"within {f['cure_days']} business days; a request still deficient when its cure period ends is withdrawn. "
                f"A request cured after the window closes joins the next cluster.")

    def _letter(self, f, rng):
        s = [f"Please find our interconnection request for the {f['mw']} MW {TECH[f['tech']][0]} {f['project']} at the {f['poi']}."]
        if f["stage"] == "cure_response":
            s.append("This package responds to your deficiency notice.")
        if f["site_doc"] == "loi":
            s.append("Landowners have signed letters of intent; leases will follow shortly.")
        elif f["site_doc"] == "unsigned_exclusivity":
            s.append("The exclusivity agreement is agreed in principle and awaiting one landowner's signature.")
        if f["site_acres"] < f["req_acres"] and f["site_doc"] not in ("loi", "unsigned_exclusivity"):
            s.append(pick(rng, ["One more parcel is in negotiation and should close this month.",
                                "We believe the layout fits on the controlled acreage with a tighter row spacing."]))
        if f["dep_paid"] < f["dep_required"]:
            s.append(pick(rng, ["The deposit was wired per our calculation.", "Our bank deducted a wire fee; we can top up if needed.",
                                "We paid the deposit using last year's fee schedule."]))
        if f["dyn_model"] in ("placeholder", "missing"):
            s.append("Inverter vendor selection is ongoing; models will be provided once the vendor is chosen.")
        if f["letter_tone"] == "confident":
            s.append(pick(rng, ["We consider the request complete.", "We trust the request is in order for the current cluster."]))
        elif f["letter_tone"] == "apologetic":
            s.append("Apologies for the late submission; we would be grateful if it could still join the current cluster.")
        return s

    def render(self, f, rng, surface):
        st, ms = pick(rng, DATE_STYLE), pick(rng, ["sym", "sym", "code"])
        rules = self._rules(f)
        if surface == "json":
            return dumps(prune({
                "request": {"queue_id": f["queue_id"], "project": f["project"], "customer": f["developer"],
                            "technology": TECH[f["tech"]][0], "requested_mw": f["mw"], "point_of_interconnection": f["poi"],
                            "submission_type": f["stage"].replace("_", " "),
                            "cover_letter": noisy(join_sentences(self._letter(f, rng)), rng, 0.1)},
                "site_control": {"document": SITE_DOCS[f["site_doc"]], "acres_controlled": f["site_acres"],
                                 "term_years": f["site_term_years"]},
                "deposit": {"paid_usd": f["dep_paid"]},
                "technical": {"one_line_diagram": ONE_LINE[f["one_line"]], "dynamic_models": DYN_MODEL[f["dyn_model"]]},
                "dates": {"reviewed_on": f["today"].isoformat(), "cluster": f["cluster"],
                          "cluster_window_closes": f["window_close"].isoformat(),
                          "earlier_cure_deadline": f["notice_deadline"].isoformat() if f["notice_deadline"] else None},
                "tariff": rules,
            }))
        if surface == "intake_checklist":
            rows = [["Queue ID", f["queue_id"]], ["Customer", f["developer"]],
                    ["Project", f"{f['project']}, {f['mw']} MW {TECH[f['tech']][0]}"], ["POI", f["poi"]],
                    ["Submission", f["stage"].replace("_", " ")
                     + (f" (cure deadline was {fmt_date(f['notice_deadline'], style=st)})" if f["notice_deadline"] else "")],
                    ["Deposit received", money(f["dep_paid"], style=ms)],
                    ["Site control", f"{SITE_DOCS[f['site_doc']]}; {f['site_acres']} acres; {f['site_term_years']}-yr term"],
                    ["One-line", ONE_LINE[f["one_line"]]], ["Dynamic models", DYN_MODEL[f["dyn_model"]]],
                    ["Review date", fmt_date(f["today"], style=st)],
                    ["Window closes", f"{fmt_date(f['window_close'], style=st)} ({f['cluster']})"]]
            return (f"{f['provider'].upper()} - INTERCONNECTION REQUEST INTAKE CHECKLIST\n"
                    + table(["Item", "Entry"], rows, rng) + f"\n\nReviewer: {f['reviewer']}\n\n{rules}")
        letter = [f"From: {f['contact']} <{f['contact'].split()[0].lower()}@{f['developer'].split()[0].lower()}-energy.example>",
                  f"To: interconnection@{f['provider'].split()[0].lower()}-grid.example",
                  f"Date: {fmt_date(f['today'], style=st)}",
                  f"Subject: {f['queue_id']} - {f['project']} interconnection request", "",
                  noisy(join_sentences(self._letter(f, rng)), rng, 0.15), "",
                  f"Regards,\n{f['contact']}, {f['developer']}", "",
                  "---- intake note (provider) ----",
                  kv_lines([("Deposit received", money(f["dep_paid"], style=ms)),
                            ("Site control on file", f"{SITE_DOCS[f['site_doc']]}, {f['site_acres']} acres, {f['site_term_years']} yr"),
                            ("One-line", ONE_LINE[f["one_line"]]), ("Models", DYN_MODEL[f["dyn_model"]]),
                            ("Earlier cure deadline", fmt_date(f["notice_deadline"], style=st) if f["notice_deadline"] else None),
                            ("Window closes", f"{fmt_date(f['window_close'], style=st)} ({f['cluster']})")], rng),
                  "", rules]
        return "\n".join(letter)

    def questions(self):
        return {
            "validity_determination": choice("How should the provider treat this interconnection request?", {
                "accept_as_valid": "Deem it valid and complete and place it in the current cluster.",
                "issue_deficiency_notice": "Send a deficiency notice listing the gaps; the cure period can end before this window closes.",
                "notice_and_next_cluster": "Send a deficiency notice and plan for the next cluster, since a cure cannot finish before this window closes.",
                "deem_withdrawn": "Treat it as withdrawn, because its cure period has already run out with gaps still open.",
            }),
            "site_control_demonstrated": noul(
                "The request demonstrates the site control the tariff requires.",
                yes="The document is an executed deed, lease or option giving exclusive rights over at least the required acreage for at least the required term.",
                no="The document is not binding or exclusive, covers too little acreage, or runs for too short a term."),
            "technical_package_quality": score("How usable is the technical package for the cluster study?", [
                "Unusable: the diagram or the dynamic models are missing.",
                "Weak: placeholders or unsupported models that study engineers would have to chase.",
                "Workable: complete, with small gaps that can be settled by email.",
                "Study-ready: stamped diagram and validated models consistent with the requested size and voltage.",
            ]),
            "deficiency_gravity": score("Taken together, how serious are the request's gaps?", [
                "None: nothing required is missing.",
                "Minor: clerical items a short cure would settle.",
                "Substantial: an item the customer may struggle to supply in time.",
                "Fundamental: a basic requirement such as binding site control or the deposit is not met.",
            ]),
            "cure_fits_before_close": noul(
                "A cure period starting on the review date would end on or before the date the cluster window closes."),
        }


# =========================================================================== schema 2
MOD_TYPES = {
    "mw_decrease": "reduce the requested capacity",
    "mw_increase": "increase the requested capacity",
    "poi_same_substation": "move to a different breaker position in the same substation",
    "poi_new_location": "move the point of interconnection to a different substation",
    "equipment_model": "change the inverter / turbine model",
    "add_storage": "add battery storage behind the same point of interconnection",
    "cod_extension": "extend the commercial operation date",
    "gen_tie_route": "reroute the generator tie line (same point of interconnection)",
}
PHASES = {"pre_phase1": "Phase 1 study not yet tendered", "phase1_complete": "Phase 1 results issued, Phase 2 not started",
          "phase2_in_progress": "Phase 2 study under way", "facilities_study": "facilities study under way"}
SCREEN = {"no_impact": "screening shows no measurable change on any monitored element",
          "minor": "screening shows flow changes under 2% on two shared elements",
          "shared_upgrade": "screening shows the change alters loading on an upgrade shared with later-queued requests",
          "not_screened": "not yet screened"}


class MaterialModificationRequest(Schema):
    name = "material_modification_request"
    description = "A customer asks to change a queued interconnection request; the provider rules on materiality and queue position."
    surfaces = {"json": 0.5, "modification_form": 0.25, "engineering_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        tech = pick_w(rng, {"solar_pv": 4, "wind": 3, "battery_storage": 2, "solar_plus_storage": 2, "gas_peaker": 1})
        mw = _mw(rng, tech)
        mod = pick(rng, list(MOD_TYPES))
        phase = pick_w(rng, {"pre_phase1": 2, "phase1_complete": 3, "phase2_in_progress": 3, "facilities_study": 2})
        dec_limit = pick(rng, [40, 50, 60])
        cod_cap = pick(rng, [36, 48])
        later_shared = pick_w(rng, {0: 2, 1: 2, 3: 2, 6: 1})
        screen = pick_w(rng, {"no_impact": 2, "minor": 3, "shared_upgrade": 3, "not_screened": 2})
        dec_pct = inc_mw = storage_mw = sc_ratio = None
        controller = None
        if mod == "mw_decrease":
            limit_now = dec_limit if phase in ("pre_phase1", "phase1_complete") else 10
            dec_pct = int(near(rng, limit_now, max(4, limit_now * 0.35), lo=3, hi=80, integer=True))
        elif mod == "mw_increase":
            inc_mw = int(round(mw * rng.uniform(0.02, 0.25)))
        elif mod == "equipment_model":
            sc_ratio = round(pick_w(rng, {rng.uniform(0.85, 1.0): 3, rng.uniform(1.0, 1.06): 2, rng.uniform(1.06, 1.25): 2}), 2)
        elif mod == "add_storage":
            storage_mw = int(round(mw * rng.uniform(0.25, 1.0) / 5) * 5)
            controller = pick_w(rng, {"certified": 3, "proposed_uncertified": 2, "not_specified": 2})
        prior_ext = min(cod_cap, pick(rng, [0, 6, 12, 18, 24, 30, cod_cap - 6, cod_cap]))
        new_ext = (int(near(rng, cod_cap - prior_ext, 8, lo=3, integer=True)) if mod == "cod_extension"
                   else pick_w(rng, {0: 2, rng.randint(6, 30): 3}))
        if reg == "clear":
            if chance(rng, 0.5):
                mod, dec_pct, phase, screen, later_shared, new_ext = "mw_decrease", rng.randint(10, 25), "pre_phase1", "no_impact", 0, 0
                inc_mw = storage_mw = sc_ratio = controller = None
            else:
                mod, phase, screen, later_shared = "poi_new_location", pick(rng, ["phase2_in_progress", "facilities_study"]), "shared_upgrade", pick(rng, [3, 6])
                dec_pct = inc_mw = storage_mw = sc_ratio = controller = None
        elif reg == "conflicting":
            # the developer calls it a minor tweak; screening says it moves a shared upgrade -- or the reverse
            if chance(rng, 0.5):
                screen, later_shared = "shared_upgrade", pick(rng, [1, 3])
            else:
                screen, later_shared = "no_impact", 0
                if mod in ("mw_decrease", "cod_extension", "add_storage", "equipment_model"):
                    mod, inc_mw = "mw_increase", int(round(mw * rng.uniform(0.02, 0.1)))
                    dec_pct = storage_mw = sc_ratio = controller = None
        orig_cod = _weekday(rng, date(2027, 1, 4), date(2030, 12, 1))
        today = _weekday(rng)
        return {"regime": reg, "provider": _provider(rng), "queue_id": _queue_id(rng, today.year - rng.randint(1, 3)),
                "project": _project(rng, tech), "developer": _developer(rng, _place(rng)),
                "tech": tech, "mw": mw, "mod_type": mod, "phase": phase, "dec_limit_pct": dec_limit, "dec_pct": dec_pct,
                "inc_mw": inc_mw, "storage_mw": storage_mw, "controller": controller, "sc_ratio": sc_ratio,
                "later_shared": later_shared, "screen": screen, "cod_cap_months": cod_cap, "prior_ext_months": prior_ext,
                "new_ext_months": new_ext, "orig_cod": orig_cod, "today": today, "engineer": person(rng),
                "dev_claim": pick(rng, ["minor", "minor", "necessary", "neutral"]),
                "reason": pick(rng, ["supply-chain change at the module vendor", "a narrower offtake contract", "a county setback ordinance",
                                     "a landowner withdrawing a parcel", "a better transformer delivery slot", "investor requirements",
                                     "a revised wind resource study", "an updated environmental survey"])}

    def _status(self, f):
        """('permitted' | 'other'), plus whether a quick confirmation restudy is still needed."""
        m = f["mod_type"]
        early = f["phase"] in ("pre_phase1", "phase1_complete")
        if m == "mw_decrease":
            return ("permitted", False) if f["dec_pct"] <= (f["dec_limit_pct"] if early else 10) else ("other", False)
        if m == "equipment_model":
            return ("permitted", f["sc_ratio"] > 0.97) if f["sc_ratio"] <= 1.0 else ("other", False)
        if m == "add_storage":
            return ("permitted", f["controller"] != "certified") if f["controller"] != "not_specified" else ("other", False)
        if m == "cod_extension":
            return ("permitted", False) if f["prior_ext_months"] + f["new_ext_months"] <= f["cod_cap_months"] else ("other", False)
        return ("other", False)

    def intended(self, f):
        kind, confirm = self._status(f)
        affects = f["later_shared"] > 0 and f["screen"] in ("shared_upgrade", "not_screened", "minor")
        material = kind == "other" and affects
        early = f["phase"] in ("pre_phase1", "phase1_complete")
        if material:
            ruling = "move_to_next_cluster" if early else "deny_modification"
        elif kind == "permitted":
            ruling = "allow_keep_position" if (not confirm and f["screen"] == "no_impact") else "allow_after_restudy"
        else:   # an unlisted change that nothing later-queued can feel
            ruling = "allow_keep_position" if (f["later_shared"] == 0 or f["screen"] == "no_impact") else "allow_after_restudy"
        big = {"mw_increase": 2, "poi_new_location": 3, "poi_same_substation": 1, "equipment_model": 1, "add_storage": 1,
               "mw_decrease": 1, "cod_extension": 1, "gen_tie_route": 0}[f["mod_type"]]
        if f["later_shared"] == 0 or f["screen"] == "no_impact":
            imp = 0
        elif f["screen"] == "minor":
            imp = 1
        else:
            imp = min(3, 1 + (f["later_shared"] >= 3) + (big >= 2))
        effort = {"mw_decrease": 1, "cod_extension": 0, "gen_tie_route": 1, "equipment_model": 2, "add_storage": 2,
                  "mw_increase": 2, "poi_same_substation": 2, "poi_new_location": 3}[f["mod_type"]]
        if f["mod_type"] == "mw_decrease" and f["phase"] == "pre_phase1":
            effort = 0
        if f["mod_type"] == "equipment_model" and f["sc_ratio"] > 1.06:
            effort = 3
        total_ext = f["prior_ext_months"] + f["new_ext_months"]
        return {"modification_ruling": ruling, "modification_is_material": "yes" if material else "no",
                "later_queue_impact": imp, "restudy_effort": effort,
                "cod_within_extension_cap": "yes" if total_ext <= f["cod_cap_months"] else "no"}

    def _change_text(self, f):
        m = f["mod_type"]
        if m == "mw_decrease":
            return f"reduce from {f['mw']} MW to {int(round(f['mw'] * (1 - f['dec_pct'] / 100)))} MW"
        if m == "mw_increase":
            return f"increase from {f['mw']} MW to {f['mw'] + f['inc_mw']} MW"
        if m == "equipment_model":
            return f"switch to a new {'turbine' if f['tech'] == 'wind' else 'inverter'} model; short-circuit contribution {f['sc_ratio']:.2f}x the original"
        if m == "add_storage":
            return (f"add {f['storage_mw']} MW of battery storage; injection limited to {f['mw']} MW by plant controller "
                    f"({ {'certified': 'certified', 'proposed_uncertified': 'proposed, not yet certified', 'not_specified': 'control scheme not specified'}[f['controller']] })")
        return MOD_TYPES[m]

    def _cod(self, f):
        cur = f["orig_cod"] + timedelta(days=int(f["prior_ext_months"] * 30.4))
        new = cur + timedelta(days=int(f["new_ext_months"] * 30.4))
        return cur, new

    def _rules(self, f):
        return (f"GIP s.4.4.1 permits without loss of queue position: (a) a capacity decrease of up to {f['dec_limit_pct']}% before "
                f"Phase 2 starts, or up to 10% after; (b) a new inverter or turbine model whose short-circuit contribution and "
                f"dynamic response are no worse than the original; (c) storage added behind the same point of interconnection "
                f"if a certified plant controller keeps injection within the requested MW; (d) in-service date extensions "
                f"totalling no more than {f['cod_cap_months']} months beyond the original request. s.4.4.2: any other change is "
                f"material if it could affect the cost or timing of a later-queued request; a material change made before Phase 2 "
                f"starts moves the request to the next cluster, and after that it is refused. s.4.4.3: the provider may restudy, at "
                f"the customer's cost, to confirm a change has no such effect.")

    def render(self, f, rng, surface):
        st, ms = pick(rng, DATE_STYLE), pick(rng, ["sym", "sym", "code"])
        cur, new = self._cod(f)
        claim = {"minor": "We regard this as a minor adjustment that should not affect anyone else in the queue.",
                 "necessary": f"The change is necessary because of {f['reason']}.",
                 "neutral": f"The change follows {f['reason']}."}[f["dev_claim"]]
        if surface == "json":
            return dumps(prune({
                "request": {"queue_id": f["queue_id"], "project": f["project"], "customer": f["developer"],
                            "technology": TECH[f["tech"]][0], "queued_mw": f["mw"], "study_status": PHASES[f["phase"]]},
                "modification": {"type": f["mod_type"].replace("_", " "), "details": self._change_text(f),
                                 "customer_statement": claim,
                                 "in_service_date": {"original": f["orig_cod"].isoformat(), "current": cur.isoformat(),
                                                     "requested": new.isoformat() if f["new_ext_months"] else None}},
                "screening": {"result": SCREEN[f["screen"]], "later_queued_requests_sharing_upgrades": f["later_shared"],
                              "engineer": f["engineer"]},
                "tariff": self._rules(f),
            }))
        if surface == "modification_form":
            rows = [("Queue ID", f["queue_id"]), ("Project", f"{f['project']} ({f['mw']} MW {TECH[f['tech']][0]})"),
                    ("Customer", f["developer"]), ("Study status", PHASES[f["phase"]]),
                    ("Requested change", self._change_text(f)), ("Reason given", claim),
                    ("Original in-service date", fmt_date(f["orig_cod"], style=st)), ("Current in-service date", fmt_date(cur, style=st)),
                    ("Requested in-service date", fmt_date(new, style=st) if f["new_ext_months"] else "no change"),
                    ("Submitted", fmt_date(f["today"], style=st))]
            return (f"{f['provider']} - MODIFICATION REQUEST FORM (GIP s.4.4)\n" + kv_lines(rows, rng)
                    + f"\n\nProvider screening ({f['engineer']}): {SCREEN[f['screen']]}; {f['later_shared']} later-queued request(s) share "
                    f"upgrades with this one.\n\n{self._rules(f)}")
        para = [f"Re {f['queue_id']} ({f['project']}, {f['mw']} MW {TECH[f['tech']][0]}): the customer asks to {self._change_text(f)}.",
                f"Study status: {PHASES[f['phase']]}.", f"Customer's position: {claim}",
                f"In-service date: original {fmt_date(f['orig_cod'], style=st)}, currently {fmt_date(cur, style=st)}"
                + (f", requested {fmt_date(new, style=st)}." if f["new_ext_months"] else ", unchanged by this request."),
                f"Screening: {SCREEN[f['screen']]}. Later-queued requests sharing upgrades: {f['later_shared']}."]
        return (f"ENGINEERING MEMO - {f['provider']}\nFrom: {f['engineer']}\nDate: {fmt_date(f['today'], style=st)}\n\n"
                + noisy(join_sentences(para), rng, 0.1) + "\n\n" + self._rules(f))

    def questions(self):
        return {
            "modification_ruling": choice("How should the provider rule on this modification request?", {
                "allow_keep_position": "Allow the change; the request keeps its place in the current cluster.",
                "allow_after_restudy": "Allow it and keep the place, but only after a restudy confirms later requests are unaffected.",
                "move_to_next_cluster": "Treat the change as material and move the request to the next cluster.",
                "deny_modification": "Refuse the change; the request continues as originally submitted or withdraws.",
            }),
            "modification_is_material": noul(
                "Under the tariff, this change is a material modification.",
                yes="It is outside the permitted list and could affect the cost or timing of a later-queued request.",
                no="It is on the permitted list, or nothing queued later could be affected by it."),
            "later_queue_impact": score("How much could the change shift cost or timing onto requests queued after it?", [
                "None: nothing queued later shares facilities with it.",
                "Slight: small flow changes that leave shared upgrades unchanged.",
                "Noticeable: a shared upgrade could change in size or cost.",
                "Major: several later requests could need different upgrades or face delays.",
            ]),
            "restudy_effort": score("How much study work would the provider need to evaluate the change?", [
                "None: a records update.",
                "A desk check of a few contingencies.",
                "Power-flow and short-circuit cases rerun for the cluster.",
                "A full restudy including stability, possibly with neighbouring systems.",
            ]),
            "cod_within_extension_cap": noul(
                "The in-service date the customer now asks for stays within the tariff's limit on cumulative extensions."),
        }


# =========================================================================== schema 3
UPGRADES = [
    ("reconductor the {a}-{b} 115 kV line ({mi} mi)", (6e6, 38e6)),
    ("rebuild the {a}-{b} 138 kV line ({mi} mi)", (14e6, 70e6)),
    ("second 230/115 kV transformer at {a} substation", (12e6, 28e6)),
    ("replace three under-rated breakers at {a} 138 kV (short-circuit duty)", (2.5e6, 7e6)),
    ("new 345 kV line {a}-{b} ({mi} mi)", (90e6, 260e6)),
    ("series reactor on the {a}-{b} 230 kV line", (4e6, 11e6)),
    ("raise line clearances on the {a}-{b} 230 kV line ({mi} mi)", (3e6, 15e6)),
]
CLAIMS = {
    "below_threshold": "Our distribution factor on this element is below the allocation threshold.",
    "pre_existing": "This element is overloaded without the cluster; the upgrade is a reliability project, not a cluster cost.",
    "math_error": "Our share does not match the tariff's pro-rata formula applied to the study's own numbers.",
    "dispatch_assumption": "The study dispatched our storage at full discharge at light load, which our operating plan never does.",
    "affected_system": "The neighbouring utility's study may require its own upgrade instead; this allocation is premature.",
    "too_expensive": "The allocation makes the project uneconomic and we ask the provider to reconsider.",
}
EST_CLASS = {"planning": ("planning estimate", 50), "phase2": ("Phase 2 estimate", 30), "facilities": ("facilities-study estimate", 20)}


class NetworkUpgradeAllocation(Schema):
    name = "network_upgrade_allocation"
    description = "A customer disputes its share of a network upgrade assigned by the cluster study; the provider rules under the tariff."
    surfaces = {"json": 0.5, "dispute_letter": 0.25, "allocation_table": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        tech = pick_w(rng, {"solar_pv": 3, "wind": 3, "battery_storage": 3, "solar_plus_storage": 3})
        mw = _mw(rng, tech)
        thr = pick(rng, [3, 5, 5, 10])
        tmpl, (clo, chi) = pick(rng, UPGRADES)
        cost = _r(rng.uniform(clo, chi), -4)
        dfax = round(pick_w(rng, {rng.uniform(thr + 2, thr + 25): 4, near(rng, thr, 1.2, lo=0.5): 3, rng.uniform(0.5, thr * 0.7): 1.5}), 1)
        n_other = rng.randint(1, 4)
        others = [round(rng.uniform(3, 45), 1) for _ in range(n_other)]
        base_load = round(pick_w(rng, {rng.uniform(70, 94): 6, near(rng, 100, 3): 2.5, rng.uniform(101, 112): 1.5}), 1)
        post_load = round(base_load + rng.uniform(8, 35), 1)
        est = pick_w(rng, {"planning": 2, "phase2": 3, "facilities": 2})
        cost_change = pick_w(rng, {0: 2, round(rng.uniform(-10, 20)): 3, round(rng.uniform(21, 80)): 2})
        aff_pending = chance(rng, 0.3)
        aff_element = aff_pending and chance(rng, 0.65)
        claim = pick(rng, list(CLAIMS))
        if tech in ("battery_storage", "solar_plus_storage") and chance(rng, 0.35):
            claim = "dispatch_assumption"
        error = chance(rng, 0.2)
        if reg == "clear":
            if chance(rng, 0.5):
                dfax, base_load, error, aff_pending, aff_element = round(rng.uniform(thr + 8, thr + 25), 1), round(rng.uniform(70, 88), 1), False, False, False
                claim = pick(rng, ["too_expensive", "dispatch_assumption"])
            else:
                base_load, claim = round(rng.uniform(103, 112), 1), "pre_existing"
        elif reg == "borderline":
            edge = pick(rng, ["dfax", "base", "claim"])
            if edge == "dfax":
                dfax, claim = round(near(rng, thr, 0.8, lo=0.5), 1), "below_threshold"
            elif edge == "base":
                base_load, claim = round(near(rng, 99.5, 1.5), 1), "pre_existing"
        else:
            # the customer's stated ground is wrong, but a different flaw is visible in the record (or none at all)
            claim = pick(rng, ["below_threshold", "pre_existing", "affected_system"])
            if claim == "below_threshold":
                dfax = round(rng.uniform(thr + 1, thr + 6), 1)
                error = chance(rng, 0.6)
            elif claim == "pre_existing":
                base_load = round(rng.uniform(91, 99.4), 1)
            else:
                aff_pending, aff_element = True, False
        impact = round(mw * dfax / 100, 1)
        a, b = pick(rng, PLACE_A), pick(rng, PLACE_A)
        today = _weekday(rng)
        return {"regime": reg, "provider": _provider(rng), "queue_id": _queue_id(rng, today.year - 1),
                "project": _project(rng, tech),
                "developer": _developer(rng, _place(rng)), "tech": tech, "mw": mw, "threshold_pct": thr,
                "upgrade": tmpl.format(a=a, b=b, mi=rng.randint(6, 48)), "cost": cost, "dfax_pct": dfax,
                "impact_mw": impact, "others_impact_mw": others, "base_load_pct": base_load, "post_load_pct": post_load,
                "est_class": est, "cost_change_pct": cost_change, "aff_pending": aff_pending,
                "aff_on_this_element": aff_element, "claim": claim, "alloc_error": error,
                "error_factor": pick(rng, [1.18, 1.25, 1.4, 0.8]) if error else 1.0, "today": today,
                "cluster": f"Cluster {pick(rng, [5, 6, 7, 8, 9, 11])}", "counsel": person(rng), "analyst": person(rng)}

    def _shares(self, f):
        if f["dfax_pct"] < f["threshold_pct"]:
            eligible = f["others_impact_mw"]
            mine = 0.0
        else:
            eligible = f["others_impact_mw"] + [f["impact_mw"]]
            mine = f["impact_mw"]
        tot = sum(eligible)
        correct = f["cost"] * mine / tot if tot else 0
        # the study's number: what it should be, or a slip (including a below-threshold request charged anyway)
        if f["dfax_pct"] < f["threshold_pct"]:
            charged = f["cost"] * f["impact_mw"] / (sum(f["others_impact_mw"]) + f["impact_mw"])
        else:
            charged = correct * f["error_factor"]
        return _r(correct, -3), _r(charged, -3)

    def intended(self, f):
        correct, charged = self._shares(f)
        pre = f["base_load_pct"] >= 100
        below = f["dfax_pct"] < f["threshold_pct"]
        if pre or below:
            ruling = "remove_from_allocation"
        elif abs(charged - correct) > 0.03 * max(correct, 1):
            ruling = "correct_share_calculation"
        elif f["aff_pending"] and f["aff_on_this_element"]:
            ruling = "hold_for_affected_system_results"
        elif f["claim"] == "dispatch_assumption" and f["tech"] in ("battery_storage", "solar_plus_storage"):
            ruling = "refer_to_dispute_resolution"
        else:
            ruling = "uphold_allocation"
        per_kw = charged / (f["mw"] * 1000)
        exp = 0 if per_kw < 25 else 1 if per_kw < 75 else 2 if per_kw < 200 else 3
        firm = {"planning": 0, "phase2": 1, "facilities": 2}[f["est_class"]]
        firm += 1 if abs(f["cost_change_pct"]) <= 10 else (-1 if abs(f["cost_change_pct"]) > 40 else 0)
        if f["aff_pending"]:
            firm -= 1
        return {"allocation_ruling": ruling, "cost_burden": exp, "estimate_firmness": max(0, min(3, firm)),
                "meets_allocation_threshold": "no" if below else "yes",
                "upgrade_needed_without_cluster": "yes" if pre else "no"}

    def _rules(self, f):
        return (f"Tariff s.6.2: a network upgrade's cost is shared among cluster requests whose distribution factor on the "
                f"constrained element is at least {f['threshold_pct']}%, pro rata to each one's MW impact (distribution factor x "
                f"requested MW). s.6.3: an upgrade needed for an overload that exists in the base case without the cluster's "
                f"generation is not a cluster cost. s.6.5: allocations touching an affected system with a pending study are "
                f"provisional. s.6.8: disputes over study assumptions go to the tariff's dispute-resolution procedure.")

    def render(self, f, rng, surface):
        st, ms = pick(rng, DATE_STYLE), pick(rng, ["sym", "sym", "code"])
        correct, charged = self._shares(f)
        label, pm = EST_CLASS[f["est_class"]]
        others = [[f"other request {i + 1}", f"{x} MW"] for i, x in enumerate(f["others_impact_mw"])]
        aff = ("neighbouring utility's affected-system study pending" + (" (it monitors this element)" if f["aff_on_this_element"] else " (other elements)")
               if f["aff_pending"] else None)
        if surface == "json":
            return dumps(prune({
                "request": {"queue_id": f["queue_id"], "project": f["project"], "customer": f["developer"],
                            "technology": TECH[f["tech"]][0], "requested_mw": f["mw"], "cluster": f["cluster"]},
                "upgrade": {"description": f["upgrade"], "estimated_cost_usd": f["cost"],
                            "estimate_class": f"{label} (+/-{pm}%)", "change_vs_previous_study_pct": f["cost_change_pct"],
                            "element_loading_without_cluster_pct": f["base_load_pct"],
                            "element_loading_with_cluster_pct": f["post_load_pct"], "affected_system": aff},
                "allocation": {"this_request_distribution_factor_pct": f["dfax_pct"], "this_request_mw_impact": f["impact_mw"],
                               "other_requests_mw_impact": f["others_impact_mw"], "allocated_to_this_request_usd": charged},
                "dispute": {"filed_by": f["counsel"], "ground": CLAIMS[f["claim"]]},
                "tariff": self._rules(f),
            }))
        if surface == "dispute_letter":
            body = [f"Subject: Dispute of {f['cluster']} network-upgrade allocation - {f['queue_id']} {f['project']}",
                    f"From: {f['counsel']}, counsel for {f['developer']}", f"Date: {fmt_date(f['today'], style=st)}", "",
                    noisy(join_sentences([
                        f"We dispute the {money(charged, style=ms)} allocated to our {f['mw']} MW {TECH[f['tech']][0]} project for the "
                        f"{f['upgrade']}.", CLAIMS[f["claim"]],
                        pick(rng, ["We request the allocation be withdrawn.", "We ask the provider to correct the allocation.",
                                   "We reserve all rights under the tariff."])]), rng, 0.1), "",
                    "---- provider study record ----",
                    kv_lines([("Upgrade cost", f"{money(f['cost'], style=ms)} ({label}, +/-{pm}%; {f['cost_change_pct']:+d}% vs previous study)"),
                              ("Element loading", f"{f['base_load_pct']}% without the cluster, {f['post_load_pct']}% with it"),
                              ("This request", f"distribution factor {f['dfax_pct']}%, MW impact {f['impact_mw']}"),
                              ("Other requests' MW impact", ", ".join(str(x) for x in f["others_impact_mw"])),
                              ("Affected system", aff)], rng), "", self._rules(f)]
            return "\n".join(body)
        rows = [[f["queue_id"] + " (this request)", f"{f['dfax_pct']}%", f"{f['impact_mw']} MW", money(charged, style=ms)]]
        rows += [[o[0], "-", o[1], "-"] for o in others]
        return (f"{f['provider']} - {f['cluster']} NETWORK UPGRADE ALLOCATION (extract)\n"
                f"Upgrade: {f['upgrade']} - {money(f['cost'], style=ms)} ({label}, +/-{pm}%, {f['cost_change_pct']:+d}% vs previous study)\n"
                f"Constrained element loading: {f['base_load_pct']}% base case without cluster -> {f['post_load_pct']}% with cluster\n"
                + (f"Affected system: {aff}\n" if aff else "") + "\n"
                + table(["request", "dist. factor", "MW impact", "allocated"], rows, rng)
                + f"\n\nCustomer dispute ({f['counsel']}): {CLAIMS[f['claim']]}\n\n{self._rules(f)}")

    def questions(self):
        return {
            "allocation_ruling": choice("How should the provider rule on the disputed allocation?", {
                "uphold_allocation": "Uphold the allocation as the study issued it.",
                "correct_share_calculation": "Keep the request in the allocation but recompute its share under the pro-rata formula.",
                "remove_from_allocation": "Take the upgrade off this request, because the tariff does not make it a cost the request shares.",
                "hold_for_affected_system_results": "Treat the allocation as provisional until the neighbouring system's study is in.",
                "refer_to_dispute_resolution": "Send the disagreement over study assumptions to the tariff's dispute-resolution procedure.",
            }),
            "cost_burden": score("How heavy is the allocation the study assigned, for a project of this size?", [
                "Light: under about $25 per kW of requested capacity.",
                "Moderate: roughly $25 to $75 per kW.",
                "Heavy: roughly $75 to $200 per kW.",
                "Severe: over about $200 per kW, enough to threaten the project.",
            ]),
            "estimate_firmness": score("How firm is the cost figure behind the allocation?", [
                "Soft: an early estimate with a wide band, or one still moving a lot between studies.",
                "Tentative: a mid-stage estimate with open questions.",
                "Fairly firm: a detailed estimate that has held roughly steady.",
                "Firm: a facilities-level estimate, stable between studies and with nothing pending elsewhere.",
            ]),
            "meets_allocation_threshold": noul(
                "This request's contribution to the overload meets the tariff's threshold for sharing the upgrade cost.",
                yes="Its distribution factor on the constrained element is at or above the tariff's threshold.",
                no="Its distribution factor is below the threshold, so it does not share this upgrade."),
            "upgrade_needed_without_cluster": noul(
                "The upgrade would be needed even without the cluster's generation.",
                yes="The constrained element is already at or above its rating in the base case without the cluster.",
                no="The element is within its rating until the cluster's generation is added."),
        }


# =========================================================================== schema 4
EVIDENCE = {
    "executed_ppa": "power purchase agreement, executed by both parties",
    "ppa_seller_signed": "power purchase agreement signed by the customer; buyer signature pending board approval",
    "shortlist_current": "notice of selection to a utility's RFP shortlist",
    "shortlist_expired": "notice of selection to a utility's RFP shortlist (the shortlist's stated validity has lapsed)",
    "term_sheet": "non-binding term sheet with an offtaker",
    "none": "no commercial-readiness document submitted",
}


class PhaseTwoReadiness(Schema):
    name = "phase_two_readiness"
    description = "Before Phase 2, a request must post its readiness deposit and show commercial readiness; the provider decides if it stays."
    surfaces = {"json": 0.5, "readiness_review": 0.25, "customer_submission": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        tech = pick_w(rng, {"solar_pv": 4, "wind": 2, "battery_storage": 3, "solar_plus_storage": 3, "gas_peaker": 1})
        mw = _mw(rng, tech)
        dep_pct, dep_min = pick(rng, [5, 10, 10, 15]), pick(rng, [250000, 500000])
        share_req = pick(rng, [50, 60, 75])
        mult = pick(rng, [2, 3])
        alloc = _r(mw * 1000 * pick_w(rng, {rng.uniform(5, 60): 3, rng.uniform(60, 180): 3, rng.uniform(180, 420): 2}), -3)
        prev_est = _r(alloc / (1 + pick_w(rng, {rng.uniform(-0.15, 0.2): 3, near(rng, 0.25, 0.06): 2, rng.uniform(0.35, 1.2): 2})), -3)
        required = max(dep_min, _r(alloc * dep_pct / 100, -2))
        posted = pick_w(rng, {required: 6, _r(required * rng.uniform(0.9, 0.995), -2): 2, _r(required * rng.uniform(0.3, 0.85), -2): 1, 0: 1})
        short_cause = pick(rng, ["wire fee deducted by the bank", "amount computed on the pre-study estimate", "second tranche still in transit"])
        evidence = pick_w(rng, {"executed_ppa": 4, "ppa_seller_signed": 2, "shortlist_current": 3, "shortlist_expired": 1, "term_sheet": 2, "none": 1.5})
        share = rng.randint(20, 100) if evidence in ("executed_ppa", "ppa_seller_signed", "term_sheet") else None
        in_lieu = chance(rng, 0.25) and evidence not in ("executed_ppa",)
        permit = pick_w(rng, {"approved": 3, "pending": 4, "denied_appealing": 1, "county_moratorium": 1})
        withdrawals = pick(rng, [0, 0, 1, 2, 4, 7])
        days_left = pick(rng, [1, 3, 6, 9, 14])
        if reg == "clear":
            if chance(rng, 0.5):
                evidence, share, posted, permit, in_lieu = "executed_ppa", rng.randint(share_req + 5, 100), required, "approved", False
                prev_est = _r(alloc / rng.uniform(0.95, 1.1), -3)
            else:
                evidence, share, posted, in_lieu, permit = "none", None, 0, False, pick(rng, ["denied_appealing", "county_moratorium"])
        elif reg == "borderline":
            if share is not None:
                share = int(near(rng, share_req, 8, lo=10, hi=100, integer=True))
        else:
            # a well-funded developer with a strong-looking but defective document, or a thin file with the deposit in lieu
            if chance(rng, 0.5):
                evidence, share, posted, withdrawals = "ppa_seller_signed", rng.randint(share_req, 100), required, 0
            else:
                evidence, share, in_lieu, permit = pick(rng, ["term_sheet", "none", "shortlist_expired"]), None, True, "pending"
        lieu_amt = required * mult if in_lieu else None
        if lieu_amt and chance(rng, 0.25):
            lieu_amt = required * (mult - 1)
        return {"regime": reg, "provider": _provider(rng), "queue_id": _queue_id(rng, 2025),
                "project": _project(rng, tech),
                "developer": _developer(rng, _place(rng)), "tech": tech, "mw": mw, "dep_pct": dep_pct, "dep_min": dep_min,
                "share_req_pct": share_req, "lieu_multiple": mult, "alloc_usd": alloc, "prev_est_usd": prev_est,
                "dep_required": required, "dep_posted": posted, "short_cause": short_cause, "evidence": evidence,
                "contracted_share_pct": share, "in_lieu_posted_usd": lieu_amt, "permit": permit,
                "dev_withdrawals_3y": withdrawals, "days_left": days_left, "reviewer": person(rng),
                "contact": person(rng), "today": _weekday(rng)}

    def _evidence_cat(self, f):
        if f["evidence"] == "executed_ppa" and (f["contracted_share_pct"] or 0) >= f["share_req_pct"]:
            return "qualifying_offtake_contract"
        if f["evidence"] == "shortlist_current":
            return "qualifying_shortlist_selection"
        if f["in_lieu_posted_usd"] and f["in_lieu_posted_usd"] >= f["dep_required"] * f["lieu_multiple"]:
            return "deposit_in_lieu"
        return "nothing_qualifying"

    def intended(self, f):
        cat = self._evidence_cat(f)
        dep_ok = f["dep_posted"] >= f["dep_required"]
        small_short = 0 < f["dep_required"] - f["dep_posted"] <= 0.1 * f["dep_required"] and f["dep_posted"] > 0
        ambiguous = f["evidence"] == "ppa_seller_signed" or (f["in_lieu_posted_usd"] and cat == "nothing_qualifying")
        if cat != "nothing_qualifying" and dep_ok:
            entry = "admit_to_phase_two"
        elif cat != "nothing_qualifying" and small_short:
            entry = "issue_cure_notice"
        elif ambiguous and (dep_ok or small_short) and f["days_left"] >= 3:
            entry = "request_clarification"
        else:
            entry = "remove_from_cluster"
        risk = 0
        risk += {"qualifying_offtake_contract": 0, "qualifying_shortlist_selection": 0.5, "deposit_in_lieu": 1, "nothing_qualifying": 1.5}[cat]
        risk += {"approved": 0, "pending": 0.25, "denied_appealing": 1.5, "county_moratorium": 1.5}[f["permit"]]
        per_kw = f["alloc_usd"] / (f["mw"] * 1000)
        risk += 0 if per_kw < 60 else 0.5 if per_kw < 180 else 1.25
        risk += 0.5 if f["dev_withdrawals_3y"] >= 4 else 0
        risk += 0.5 if f["alloc_usd"] > f["prev_est_usd"] * 1.25 else 0
        strength = {"executed_ppa": 3, "ppa_seller_signed": 2, "shortlist_current": 2, "shortlist_expired": 1,
                    "term_sheet": 1, "none": 0}[f["evidence"]]
        if f["evidence"] in ("executed_ppa", "ppa_seller_signed") and (f["contracted_share_pct"] or 0) < f["share_req_pct"]:
            strength -= 1
        return {"phase_two_entry": entry, "readiness_evidence_category": cat,
                "withdrawal_likelihood": max(0, min(4, int(round(risk)))), "readiness_evidence_strength": max(0, strength),
                "deposit_amount_sufficient": "yes" if dep_ok else "no"}

    def _rules(self, f):
        return (f"GIP s.5.1: to enter Phase 2 a request must (a) post Readiness Deposit 2 of {f['dep_pct']}% of its Phase 1 "
                f"network-upgrade allocation, minimum {_musd(f['dep_min'])}; and (b) show commercial readiness by an offtake "
                f"contract executed by both parties for at least {f['share_req_pct']}% of requested MW, or a current selection on "
                f"a load-serving entity's RFP shortlist; or (c) in place of (b), post an extra deposit of {f['lieu_multiple']}x "
                f"Readiness Deposit 2. s.5.3: a deposit short by no more than 10% because of a payment error may be cured "
                f"within 5 business days of notice. Requests not meeting s.5.1 at the window's end are withdrawn.")

    def render(self, f, rng, surface):
        st, ms = pick(rng, DATE_STYLE), pick(rng, ["sym", "sym", "code"])
        ev = EVIDENCE[f["evidence"]] + (f", covering {f['contracted_share_pct']}% of capacity" if f["contracted_share_pct"] else "")
        permit = {"approved": "county conditional-use permit approved", "pending": "county permit hearing pending",
                  "denied_appealing": "county permit denied, appeal filed", "county_moratorium": "county moratorium on new utility-scale projects"}[f["permit"]]
        short = f["dep_required"] - f["dep_posted"]
        if surface == "json":
            return dumps(prune({
                "request": {"queue_id": f["queue_id"], "project": f["project"], "customer": f["developer"],
                            "technology": TECH[f["tech"]][0], "requested_mw": f["mw"]},
                "phase1": {"network_upgrade_allocation_usd": f["alloc_usd"], "pre_study_estimate_usd": f["prev_est_usd"]},
                "readiness": {"deposit_posted_usd": f["dep_posted"],
                              "deposit_note": f["short_cause"] if 0 < short and f["dep_posted"] else None,
                              "commercial_evidence": ev, "extra_deposit_posted_usd": f["in_lieu_posted_usd"],
                              "business_days_left_in_window": f["days_left"]},
                "project_status": {"permitting": permit, "customer_withdrawals_last_3_years": f["dev_withdrawals_3y"]},
                "tariff": self._rules(f),
            }))
        if surface == "readiness_review":
            rows = [("Queue ID", f["queue_id"]), ("Project", f"{f['project']} - {f['mw']} MW {TECH[f['tech']][0]}"),
                    ("Phase 1 NU allocation", money(f["alloc_usd"], style=ms)), ("Pre-study estimate", money(f["prev_est_usd"], style=ms)),
                    ("Readiness Deposit 2 posted", money(f["dep_posted"], style=ms) if f["dep_posted"] else "nothing received"),
                    ("Payment note", f["short_cause"] if 0 < short and f["dep_posted"] else None),
                    ("Commercial readiness", ev), ("Extra deposit in lieu", money(f["in_lieu_posted_usd"], style=ms) if f["in_lieu_posted_usd"] else "none"),
                    ("Permitting", permit), ("Customer's withdrawals (3 yrs)", f["dev_withdrawals_3y"]),
                    ("Business days left in window", f["days_left"])]
            return f"{f['provider']} - PHASE 2 READINESS REVIEW\nReviewer: {f['reviewer']}\n\n" + kv_lines(rows, rng) + "\n\n" + self._rules(f)
        s = [f"Attached is our Phase 2 readiness package for {f['queue_id']} ({f['project']}, {f['mw']} MW).",
             f"Commercial readiness: {ev}."]
        if f["dep_posted"]:
            s.append(f"We have wired {money(f['dep_posted'], style=ms)} as Readiness Deposit 2" + (f" ({f['short_cause']})." if 0 < short else "."))
        else:
            s.append("The readiness deposit will follow once our board approves the funding.")
        if f["in_lieu_posted_usd"]:
            s.append(f"We have also posted {money(f['in_lieu_posted_usd'], style=ms)} as the additional deposit in lieu of commercial-readiness evidence.")
        s.append(f"Permitting: {permit}.")
        if f["evidence"] == "ppa_seller_signed":
            s.append("The buyer's board meets next month and countersignature is expected then.")
        return (f"From: {f['contact']} ({f['developer']})\nSubject: Phase 2 readiness - {f['queue_id']}\n\n"
                + noisy(join_sentences(s), rng, 0.1)
                + f"\n\n[provider note: Phase 1 allocation {money(f['alloc_usd'], style=ms)}; pre-study estimate {money(f['prev_est_usd'], style=ms)}; "
                  f"{f['days_left']} business days left in the readiness window; customer has withdrawn {f['dev_withdrawals_3y']} request(s) in 3 years]\n\n"
                + self._rules(f))

    def questions(self):
        return {
            "phase_two_entry": choice("What should the provider do with this request at the Phase 2 readiness check?", {
                "admit_to_phase_two": "Admit it to the Phase 2 study.",
                "issue_cure_notice": "Notify the customer of a payment shortfall it may still cure under the tariff, then decide.",
                "request_clarification": "Ask for a corrected or completed document before the window closes; admission turns on it.",
                "remove_from_cluster": "Withdraw it from the cluster for failing the readiness requirements.",
            }),
            "readiness_evidence_category": choice("Which way, if any, does the file satisfy the commercial-readiness requirement?", {
                "qualifying_offtake_contract": "An offtake contract executed by both parties that meets the tariff's capacity share.",
                "qualifying_shortlist_selection": "A current selection on a load-serving entity's procurement shortlist.",
                "deposit_in_lieu": "No qualifying document, but the full extra deposit the tariff accepts in its place.",
                "nothing_qualifying": "Nothing on file satisfies the requirement.",
            }),
            "withdrawal_likelihood": score("How likely is this project to leave the queue before signing an interconnection agreement?", [
                "Very unlikely: contracted, permitted and affordable.",
                "Unlikely: one open item, otherwise on track.",
                "Even odds: real commercial or permitting doubts.",
                "Likely: several serious obstacles at once.",
                "Very likely: no buyer, blocked permits or costs that sink it.",
            ]),
            "readiness_evidence_strength": score("How strong is the commercial-readiness evidence as a commitment to build?", [
                "None: nothing binding on file.",
                "Weak: a non-binding or lapsed document.",
                "Moderate: a real selection or contract with a gap in signature or coverage.",
                "Strong: an executed contract covering the required capacity.",
            ]),
            "deposit_amount_sufficient": noul(
                "The readiness deposit posted is at least the amount the tariff requires.",
                yes="The amount posted equals or exceeds the tariff's percentage of the allocation, or its minimum if that is higher.",
                no="Nothing was posted, or the amount is below what the tariff formula requires."),
        }


# =========================================================================== schema 5
REASONS = {
    "cost_increase": "Network-upgrade costs assigned in the latest study make the project uneconomic.",
    "lost_offtake": "Our offtaker terminated the power purchase agreement.",
    "permit_denied": "The county denied our conditional-use permit.",
    "financing": "Our tax-equity partner has withdrawn.",
    "portfolio": "The project no longer fits our portfolio strategy.",
}
STAGES = {"before_phase1_results": "before Phase 1 results", "after_phase1_results": "after Phase 1 results, before Phase 2 results",
          "after_phase2_results": "after Phase 2 results"}


class WithdrawalPenaltyReview(Schema):
    name = "withdrawal_penalty_review"
    description = "A request has withdrawn from a cluster; the provider determines the withdrawal penalty under the tariff."
    surfaces = {"json": 0.5, "settlement_worksheet": 0.25, "withdrawal_notice": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        tech = pick_w(rng, {"solar_pv": 4, "wind": 3, "battery_storage": 3, "solar_plus_storage": 2})
        mw = _mw(rng, tech)
        stage = pick_w(rng, {"after_phase1_results": 3, "after_phase2_results": 2})
        alloc_prev = _r(mw * 1000 * rng.uniform(10, 200), -3)
        incr = pick_w(rng, {round(rng.uniform(-0.2, 0.18), 3): 5, near(rng, 0.25, 0.05, digits=3): 3, round(rng.uniform(0.35, 1.5), 3): 1})
        alloc = _r(alloc_prev * (1 + incr), -3)
        window_bd = pick(rng, [10, 15, 20])
        bd_after = pick_w(rng, {rng.randint(1, window_bd): 2, rng.randint(window_bd - 2, window_bd + 3): 2, rng.randint(window_bd + 4, 70): 4})
        shared = pick_w(rng, {0: 3, 1: 2, 3: 2, 6: 1})
        restudy = "none_needed" if shared == 0 else pick_w(rng, {"required": 3, "under_evaluation": 3, "none_needed": 1})
        reason = pick(rng, list(REASONS))
        requeue = pick_w(rng, {None: 6, "same_poi_new_request": 2, "affiliate_nearby_poi": 1})
        offtake = pick_w(rng, {"executed": 2, "terminated": 2, "never_had": 2})
        permit = pick_w(rng, {"approved": 2, "pending": 2, "denied": 1})
        if reg == "clear":
            if chance(rng, 0.5):
                stage, incr = "after_phase2_results", round(rng.uniform(-0.1, 0.1), 3)
                alloc = _r(alloc_prev * (1 + incr), -3)
                bd_after, shared, restudy, requeue = rng.randint(window_bd + 10, 60), 3, "required", None
            else:     # plainly inside the penalty-free window
                bd_after = rng.randint(1, max(1, window_bd - 4))
        elif reg == "conflicting":
            # the letter blames permits or offtake, but the record shows a cost jump and a fresh request at the same POI
            reason = pick(rng, ["permit_denied", "lost_offtake", "portfolio"])
            if chance(rng, 0.6):
                requeue = "same_poi_new_request"
            if chance(rng, 0.5):
                incr = round(rng.uniform(0.3, 1.2), 3)
                alloc = _r(alloc_prev * (1 + incr), -3)
        if reason == "lost_offtake":
            offtake = "terminated"
        if reason == "permit_denied":
            permit = "denied"
        study_cost = _r(rng.uniform(80000, 450000), -3)
        deposits = _r(study_cost + alloc * pick(rng, [0.05, 0.1, 0.15]) + rng.uniform(50000, 300000), -3)
        pen_mult, pen_pct = pick(rng, [2, 3]), pick(rng, [10, 15])
        return {"regime": reg, "provider": _provider(rng), "queue_id": _queue_id(rng, 2024),
                "project": _project(rng, tech),
                "developer": _developer(rng, _place(rng)), "tech": tech, "mw": mw, "stage": stage,
                "alloc_prev_usd": alloc_prev, "alloc_usd": alloc, "increase_frac": incr, "window_bd": window_bd,
                "bd_after_results": bd_after, "shared_upgrades": shared, "restudy": restudy, "reason": reason,
                "requeue": requeue, "requeue_days_later": rng.randint(3, 60) if requeue else None, "offtake": offtake,
                "permit": permit, "study_cost": study_cost, "deposits": deposits, "pen_mult": pen_mult, "pen_pct": pen_pct,
                "analyst": person(rng), "contact": person(rng), "results_date": _weekday(rng)}

    def intended(self, f):
        in_window = f["bd_after_results"] <= f["window_bd"]
        cost_exempt = f["increase_frac"] > 0.25
        if f["stage"] == "before_phase1_results" or (in_window and f["stage"] != "before_phase1_results") or cost_exempt:
            det = "no_penalty"
        elif f["restudy"] == "under_evaluation":
            det = "hold_pending_restudy"
        elif f["restudy"] == "none_needed":
            det = "half_penalty"
        else:
            det = "full_scheduled_penalty"
        if f["requeue"] == "same_poi_new_request" or (f["requeue"] and f["increase_frac"] < 0.1):
            driver = "tactical_requeue"
        elif f["increase_frac"] > 0.3:
            driver = "upgrade_cost_increase"
        elif f["offtake"] == "terminated" or f["reason"] == "financing":
            driver = "lost_offtake_or_financing"
        elif f["permit"] == "denied":
            driver = "permitting_or_site_failure"
        elif f["increase_frac"] > 0.15:
            driver = "upgrade_cost_increase"
        else:
            driver = "lost_offtake_or_financing" if f["offtake"] == "never_had" else "permitting_or_site_failure"
        harm = {0: 0, 1: 1, 3: 2, 6: 3}[f["shared_upgrades"]]
        if f["restudy"] == "none_needed":
            harm = min(harm, 1)
        if f["stage"] == "before_phase1_results" or in_window or f["increase_frac"] > 0.35:
            strength = 3
        elif cost_exempt:
            strength = 2
        elif f["increase_frac"] > 0.15 or f["bd_after_results"] <= f["window_bd"] + 3:
            strength = 1
        else:
            strength = 0
        return {"penalty_determination": det, "withdrawal_driver": driver, "harm_to_remaining": harm,
                "exemption_case_strength": strength, "inside_election_window": "yes" if in_window else "no"}

    def _rules(self, f):
        return (f"GIP s.7.2: no penalty for withdrawal before Phase 1 results. After Phase 1 results the penalty is the greater "
                f"of {f['pen_mult']}x the customer's study costs or {f['pen_pct']}% of its current network-upgrade allocation "
                f"({2 * f['pen_pct']}% after Phase 2 results), capped at deposits held. s.7.3: no penalty if the allocation rose more than 25% over the previous study, or if "
                f"the customer withdraws within {f['window_bd']} business days after results are posted. s.7.4: if the withdrawal "
                f"requires no restudy and shifts no cost to other requests, the penalty is halved; the provider may hold refunds "
                f"until that is known.")

    def render(self, f, rng, surface):
        st, ms = pick(rng, DATE_STYLE), pick(rng, ["sym", "sym", "code"])
        withdrawn_on = add_business_days(f["results_date"], f["bd_after_results"])
        restudy = {"required": "restudy of remaining cluster requests required", "under_evaluation": "restudy need under evaluation",
                   "none_needed": "no restudy needed"}[f["restudy"]]
        rq = None
        if f["requeue"]:
            rq = (f"an affiliate of the customer filed a new request at {'the same' if f['requeue'] == 'same_poi_new_request' else 'a nearby'} "
                  f"point of interconnection {f['requeue_days_later']} days after the withdrawal")
        off = {"executed": "PPA executed and in force", "terminated": "PPA terminated by the offtaker", "never_had": "no offtake contract on file"}[f["offtake"]]
        if surface == "json":
            return dumps(prune({
                "request": {"queue_id": f["queue_id"], "project": f["project"], "customer": f["developer"],
                            "technology": TECH[f["tech"]][0], "requested_mw": f["mw"]},
                "withdrawal": {"stage": STAGES[f["stage"]], "results_posted": f["results_date"].isoformat(),
                               "notice_received": withdrawn_on.isoformat(), "stated_reason": REASONS[f["reason"]]},
                "costs": {"previous_study_allocation_usd": f["alloc_prev_usd"], "current_allocation_usd": f["alloc_usd"],
                          "study_costs_usd": f["study_cost"], "deposits_held_usd": f["deposits"]},
                "cluster_effects": {"shared_upgrades_with_other_requests": f["shared_upgrades"], "restudy": restudy},
                "record": {"offtake": off, "permit": f["permit"], "later_filings": rq},
                "tariff": self._rules(f),
            }))
        if surface == "settlement_worksheet":
            rows = [["Stage at withdrawal", STAGES[f["stage"]]], ["Results posted", fmt_date(f["results_date"], style=st)],
                    ["Withdrawal notice", fmt_date(withdrawn_on, style=st)],
                    ["Allocation: previous / current", f"{money(f['alloc_prev_usd'], style=ms)} / {money(f['alloc_usd'], style=ms)}"],
                    ["Study costs to date", money(f["study_cost"], style=ms)], ["Deposits held", money(f["deposits"], style=ms)],
                    ["Shared upgrades", f["shared_upgrades"]], ["Restudy", restudy], ["Offtake", off], ["Permit", f["permit"]],
                    ["Later filings", rq or "none found"]]
            return (f"{f['provider']} - WITHDRAWAL SETTLEMENT WORKSHEET - {f['queue_id']} ({f['project']}, {f['mw']} MW)\n"
                    f"Analyst: {f['analyst']}\n\n" + table(["item", "value"], rows, rng) + f"\n\n{self._rules(f)}")
        s = [f"{f['developer']} hereby withdraws interconnection request {f['queue_id']} ({f['project']}).",
             REASONS[f["reason"]], pick(rng, ["We request a full refund of all deposits.", "Please confirm the refund amount.",
                                                "We expect the tariff's exemptions to apply."])]
        return (f"Date: {fmt_date(withdrawn_on, style=st)}\nFrom: {f['contact']}, {f['developer']}\nSubject: Notice of withdrawal - {f['queue_id']}\n\n"
                + noisy(join_sentences(s), rng, 0.1)
                + "\n\n---- provider file ----\n"
                + bullets([f"stage: {STAGES[f['stage']]}; results posted {fmt_date(f['results_date'], style=st)}",
                           f"allocation {money(f['alloc_prev_usd'], style=ms)} in the previous study, {money(f['alloc_usd'], style=ms)} now",
                           f"study costs {money(f['study_cost'], style=ms)}; deposits held {money(f['deposits'], style=ms)}",
                           f"{f['shared_upgrades']} shared upgrade(s); {restudy}", f"offtake: {off}; permit: {f['permit']}",
                           rq], rng)
                + "\n\n" + self._rules(f))

    def questions(self):
        return {
            "penalty_determination": choice("What withdrawal penalty should the provider apply?", {
                "no_penalty": "No penalty: refund the deposits less actual study costs.",
                "half_penalty": "Half the scheduled penalty, because nobody else is restudied or charged more.",
                "full_scheduled_penalty": "The full penalty the tariff schedules for the stage at which it withdrew.",
                "hold_pending_restudy": "Hold the refund until it is known whether a restudy or cost shift follows.",
            }),
            "withdrawal_driver": choice("What most likely drove this withdrawal, reading the whole record?", {
                "upgrade_cost_increase": "A jump in the network-upgrade costs assigned to the project.",
                "lost_offtake_or_financing": "Losing, or never landing, a buyer or the financing.",
                "permitting_or_site_failure": "A permit refusal or a problem with the site.",
                "tactical_requeue": "A tactical exit to re-enter the queue in a better position.",
            }),
            "harm_to_remaining": score("How much does the withdrawal shift cost or delay onto the requests that remain?", [
                "None: it shared nothing with them.",
                "Slight: one shared item, little to restudy.",
                "Material: shared upgrades must be reassigned and restudied.",
                "Heavy: many requests face new costs or a delayed study.",
            ]),
            "exemption_case_strength": score("How strong is the customer's case for escaping the penalty?", [
                "None: no exemption comes close to applying.",
                "Weak: a cost rise or timing argument that falls short of the tariff's terms.",
                "Arguable: an exemption applies on a reasonable reading of the figures.",
                "Clear: the record plainly meets an exemption or a no-penalty stage.",
            ]),
            "inside_election_window": noul(
                "The customer gave notice within the tariff's penalty-free window after study results were posted.",
                yes="The notice date falls within the stated number of business days after results were posted.",
                no="The notice came after that window closed."),
        }


WORKFLOW = Workflow(
    name="generator_interconnection_review",
    title="Generator interconnection queue review",
    description="Validity, modification, cost-allocation, readiness and withdrawal decisions on generator interconnection requests under a written tariff.",
    domain="energy / transmission planning and interconnection",
    schemas=[RequestCompletenessReview(), MaterialModificationRequest(), NetworkUpgradeAllocation(), PhaseTwoReadiness(),
             WithdrawalPenaltyReview()],
    notes="Invented transmission providers, projects and tariff sections; thresholds differ by provider so the quoted rule "
          "must be read. Deposits and penalties are deposits held against studies, never bills paid or rejected.",
)
