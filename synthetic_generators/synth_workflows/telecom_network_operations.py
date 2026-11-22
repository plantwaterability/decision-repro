"""Telecom network operations: carrier-side decisions about the network itself, judged against (fictional) regulator
rules, municipal ordinances, an industry portability code and the carrier's own engineering procedures.

Five schemas: whether a network outage must be reported to the regulator and how fast, an engineering review of a
proposed small-cell site against a city's wireless ordinance, triage of an uplink interference report, a sector's
busy-hour load against the capacity-planning trigger, and an inter-carrier number-porting dispute judged on the
records. Artefacts are NOC tickets, incident-bridge notes, site review forms, zoning emails, interference tickets,
engineer chat, planning dashboards and dispute forms.

This is NOT customer support: no subscriber conversation is answered, no churn or retention question is asked, and
no question asks whether anything is malicious. Regulators, cities, carriers, codes and numbers are invented; every
threshold is sampled per case, and only the clauses that govern the case are quoted.

Each schema also has `intended(f)`: the designer's reading of the rule for that case. It is never rendered and
never a label; it exists so the answer-spread check can confirm every question moves with the factors.
"""
from __future__ import annotations

import math
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

CARRIER_A = ["Brightwater", "Corvel", "Daleside", "Eastmere", "Fenwick", "Galloway", "Hollin", "Kestrin", "Marrowby",
             "Nordvale", "Pemberly", "Ravelstone", "Silverbeck", "Tarnwick", "Velden", "Wyncote", "Ashmere", "Coldharbour"]
CARRIER_B = ["Mobile", "Telecom", "Wireless", "Networks", "Connect"]
CITIES = ["Port Allard", "Kingsmere", "Elsworth", "Ravenna Falls", "Hadley Cross", "Lowmoor", "Brackenford", "Tilney",
          "Cardew", "Ostley", "Fairhaven Bay", "Durrant", "Wickmoor", "Ellery Springs"]


def _carrier(rng):
    return f"{pick(rng, CARRIER_A)} {pick(rng, CARRIER_B)}"


def _first(name):
    return name.split()[0]


def _kv(pairs, rng, skip_none=True):
    """kv_lines without the snake_case 'equals' house style, which mangles labels such as 'Site / sector'."""
    return kv_lines(pairs, rng, style=pick(rng, ["colon", "colon", "aligned", "dash"]), skip_none=skip_none)


# =========================================================================== schema 1
ELEMENTS = {
    "fibre cut on the metro ring": "voice and data", "IMS core node failure": "VoLTE voice",
    "packet gateway fault": "mobile data", "commercial power loss at a cluster of cell sites": "all services",
    "DNS resolver outage": "mobile data", "transport router misconfiguration": "all services",
    "SMS centre fault": "SMS", "microwave backhaul link failure": "all services",
}
COUNT_METHODS = {"registrations": ("HSS registrations that dropped", 3), "traffic": ("cell traffic drop, extrapolated", 2),
                 "care_calls": ("scaled up from care-line call volume", 1)}
EMERG = {"unaffected": "emergency calls unaffected", "rerouted": "emergency calls rerouted to the backup answering point",
         "failed_some": "emergency calls failing for some callers", "failed_all": "emergency calls failing in the area"}


class OutageReportability(Schema):
    name = "outage_reportability"
    description = "A network outage judged against the regulator's outage-reporting thresholds and its emergency-call clause."
    surfaces = {"json": 0.5, "noc_ticket": 0.25, "bridge_notes": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        N = pick(rng, [5000, 10000, 20000, 50000])
        D = pick(rng, [30, 45, 60])
        E = pick(rng, [10, 15, 30])
        el = pick(rng, list(ELEMENTS))
        method = pick_w(rng, {"registrations": 3, "traffic": 3, "care_calls": 2, None: 1})
        subs = int(N * rng.uniform(0.2, 3.0))
        dur = int(D * rng.uniform(0.3, 3.0))
        emerg = pick_w(rng, {"unaffected": 5, "rerouted": 2, "failed_some": 1.6, "failed_all": 0.8})
        if ELEMENTS[el] in ("mobile data", "SMS"):
            emerg = "unaffected"
        k = None
        if reg == "clear":
            if chance(rng, 0.5):
                subs, dur, emerg = int(N * rng.uniform(0.05, 0.4)), int(D * rng.uniform(0.2, 0.7)), "unaffected"
                method = pick(rng, ["registrations", "traffic"])
            else:
                subs, dur = int(N * rng.uniform(2.0, 6.0)), int(D * rng.uniform(1.5, 5.0))
                method = "registrations"
        elif reg == "borderline":
            k = pick(rng, ["subs", "duration", "emergency"])
            if k == "subs":
                subs = int(near(rng, N, N * 0.15))
                dur = int(D * rng.uniform(1.0, 2.5))
            elif k == "duration":
                dur = near(rng, D, 8, lo=5, integer=True)
                subs = int(N * rng.uniform(1.0, 3.0))
            else:
                el = pick(rng, ["IMS core node failure", "fibre cut on the metro ring", "commercial power loss at a cluster of cell sites"])
                emerg = pick(rng, ["failed_some", "rerouted"])
                dur = near(rng, E, 5, lo=3, integer=True)
                subs = int(N * rng.uniform(0.1, 0.9))
        else:
            k = pick(rng, ["weak_count_near", "small_but_emergency", "noc_says_under"])
            if k == "weak_count_near":
                method = pick(rng, ["care_calls", None])
                subs = int(N * rng.uniform(0.6, 1.1))
                dur = int(D * rng.uniform(1.2, 3.0))
            elif k == "small_but_emergency":
                el = pick(rng, ["IMS core node failure", "transport router misconfiguration"])
                subs = int(N * rng.uniform(0.05, 0.4))
                emerg = pick(rng, ["failed_some", "failed_all"])
                dur = E + rng.randint(1, 40)
            else:
                subs = int(N * rng.uniform(1.0, 1.6))
                dur = int(D * rng.uniform(1.0, 1.5))
                method = pick(rng, ["registrations", "traffic"])
        if method is None:
            subs = None
        start = rand_ts(rng)
        return {"regime": reg, "case_kind": k, "carrier": _carrier(rng), "region": pick(rng, CITIES),
                "element": el, "services": ELEMENTS[el], "subs": subs, "method": method,
                "duration_min": dur, "ongoing": chance(rng, 0.25), "emergency": emerg,
                "critical_site": pick_w(rng, {None: 6, "regional hospital": 1, "airport": 1, "emergency operations centre": 1}),
                "subs_rule": N, "dur_rule": D, "emerg_rule": E, "std_window_h": pick(rng, [24, 72]),
                "emerg_window_min": pick(rng, [60, 120]), "start": start,
                "hours_since": round(rng.uniform(0.5, 20), 1), "ticket": ident(rng, "NOC", 6, "-"),
                "lead": person(rng), "duty_mgr": person(rng),
                "opinion": "under" if k == "noc_says_under" else pick(rng, ["under", "unsure", "none", "none"]),
                "rule_id": pick(rng, ["NRO Outage Rule 7.3", "NRO-OR 4", "Network Reliability Order 12"])}

    def intended(self, f):
        subs, dur = f["subs"], f["duration_min"]
        meets = subs is not None and subs >= f["subs_rule"] and dur >= f["dur_rule"]
        emerg = f["emergency"] in ("failed_some", "failed_all") and dur >= f["emerg_rule"]
        weak = f["method"] in ("care_calls", None)
        if emerg:
            ob = "expedited_emergency_notice"
        elif meets and not (weak and subs < 1.2 * f["subs_rule"]):
            ob = "standard_notice"
        elif dur >= f["dur_rule"] and (subs is None or (weak and subs >= 0.6 * f["subs_rule"])):
            ob = "refine_count_then_decide"
        elif meets:
            ob = "standard_notice"
        else:
            ob = "no_external_report"
        if subs is None:
            imp = 2 if dur >= 2 * f["dur_rule"] else 1
        else:
            ratio = subs * dur / (f["subs_rule"] * f["dur_rule"])
            imp = 0 if ratio < 0.25 else 1 if ratio < 1 else 2 if ratio < 4 else 3
        if f["critical_site"] or f["emergency"] in ("failed_some", "failed_all"):
            imp = min(3, imp + 1)
        rel = 0 if f["method"] is None else COUNT_METHODS[f["method"]][1]
        if f["ongoing"] and rel > 0:
            rel -= 1
        return {"reporting_obligation": ob, "user_impact_scale": imp, "count_reliability": rel,
                "meets_reporting_threshold": "yes" if meets else "no",
                "emergency_calls_impaired": "yes" if f["emergency"] in ("failed_some", "failed_all") else "no"}

    def _rule(self, f):
        return (f"{f['rule_id']}: report outages lasting >= {f['dur_rule']} min that affect >= {f['subs_rule']:,} "
                f"subscribers within {f['std_window_h']} h; emergency-call failures lasting >= {f['emerg_rule']} min "
                f"within {f['emerg_window_min']} min")

    def _count(self, f):
        if f["subs"] is None:
            return "not yet estimated"
        return f"~{f['subs']:,} ({COUNT_METHODS[f['method']][0]})"

    def _dur(self, f):
        return f"{f['duration_min']} min" + (" so far, still ongoing" if f["ongoing"] else "")

    def _opinion(self, f, rng):
        t = {"under": "Ops view is that this stays under the reporting line.",
             "unsure": ("Not sure the count is solid yet; care-line numbers are still coming in."
                        if f["method"] in ("care_calls", None) else "Count may still move once the core team reconciles it."),
             "none": ""}[f["opinion"]]
        return noisy(t, rng, 0.15) if t else None

    def render(self, f, rng, surface):
        op = self._opinion(f, rng)
        if surface == "json":
            return dumps(prune({
                "incident": {"ticket": f["ticket"], "carrier": f["carrier"], "region": f["region"],
                             "cause": f["element"], "services_down": f["services"],
                             "started": fmt_ts(f["start"], style="iso"), "duration": self._dur(f)},
                "impact": {"subscribers": f["subs"], "count_method": COUNT_METHODS[f["method"]][0] if f["method"] else None,
                           "emergency_calls": EMERG[f["emergency"]], "critical_sites": f["critical_site"]},
                "status": {"hours_since_discovery": f["hours_since"], "filed_with_regulator": False,
                           "incident_lead_note": op},
                "rule": self._rule(f),
            }))
        if surface == "noc_ticket":
            body = _kv([("Ticket", f["ticket"]), ("Carrier / region", f"{f['carrier']} / {f['region']}"),
                             ("Cause", f["element"]), ("Services", f["services"]),
                             ("Start", fmt_ts(f["start"], rng)), ("Duration", self._dur(f)),
                             ("Subscribers affected", self._count(f)), ("Emergency calls", EMERG[f["emergency"]]),
                             ("Critical sites", f["critical_site"] or "none identified"),
                             ("Hours since discovery", f["hours_since"]), ("Regulator filing", "not filed")], rng)
            return f"NOC INCIDENT TICKET -- SEV 1\n{body}\nLead note: {op or '-'}\nRef: {self._rule(f)}"
        s = [f"Bridge opened for {f['element']} in {f['region']} ({f['services']} affected).",
             f"Duration {self._dur(f)}.", f"Count: {self._count(f)}.", EMERG[f["emergency"]].capitalize() + ".",
             (f"{f['critical_site'].capitalize()} in the affected footprint." if f["critical_site"] else ""),
             f"Discovered {f['hours_since']} h ago; nothing filed with the regulator yet.", op or ""]
        return "\n".join([f"Incident bridge summary -- {f['ticket']} ({f['carrier']})",
                          f"Lead: {f['lead']}  Duty manager: {f['duty_mgr']}", "", join_sentences(s), "",
                          f"Reporting rule: {self._rule(f).split(': ', 1)[1]}"])

    def questions(self):
        return {
            "reporting_obligation": choice("What does the carrier owe the regulator for this outage?", {
                "expedited_emergency_notice": "An early notice under the emergency-call clause, plus word to the affected answering points.",
                "no_external_report": "No filing; record it internally only.",
                "refine_count_then_decide": "Firm up the affected-subscriber count before deciding whether to file.",
                "standard_notice": "A standard outage notice within the rule's ordinary filing window.",
            }),
            "user_impact_scale": score("How large is the outage's impact on users, taking size, length and what was lost together?", [
                "Small: few users for a short time.",
                "Moderate: noticeable but below the reporting scale.",
                "Large: at or beyond the reporting scale.",
                "Very large: many times the scale, or emergency services or a critical site affected.",
            ]),
            "count_reliability": score("How reliable is the count of affected subscribers?", [
                "None: there is no count yet.",
                "Weak: a rough extrapolation from indirect signs.",
                "Fair: a reasonable network-based estimate that may still move.",
                "Firm: taken directly from network registration data.",
            ]),
            "meets_reporting_threshold": noul(
                "On the best available count, the outage meets the regulator's size-and-duration reporting threshold.",
                yes="Both the subscriber count and the duration are at or above the rule's figures.",
                no="The count or the duration falls short, or there is no count to compare."),
            "emergency_calls_impaired": noul(
                "Callers in the affected area could not reliably reach emergency services during the outage."),
        }


# =========================================================================== schema 2
STRUCTURES = {"existing utility pole": True, "existing streetlight": True, "new steel pole": False, "rooftop": True}


class SmallCellSiting(Schema):
    name = "small_cell_siting_review"
    description = "An engineering review of a proposed small-cell site against a city's wireless-facility ordinance before submission."
    surfaces = {"json": 0.5, "site_review_form": 0.25, "zoning_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        S = pick(rng, [6, 10, 15, 30])
        cap = pick(rng, [10.5, 12.0, 15.0])
        struct = pick(rng, list(STRUCTURES))
        existing = round(rng.uniform(7.5, 11.5), 1) if STRUCTURES[struct] else None
        if struct == "rooftop":
            existing = round(rng.uniform(9.0, 14.0), 1)
        add = round(rng.uniform(0.4, 2.5), 1)
        height = round((existing or rng.uniform(8.0, 12.5)) + (add if existing else 0), 1)
        setback = round(S * rng.uniform(0.5, 2.0), 1)
        heritage = chance(rng, 0.32)
        rf = maybe(rng, round(rng.uniform(5, 70), 1), 0.1)
        load = maybe(rng, int(rng.uniform(55, 98)), 0.1)
        consent = pick_w(rng, {"granted": 6, "pending": 3, "refused": 1})
        drop = round(rng.uniform(0.4, 3.5), 1)
        util = rng.randint(45, 98)
        k = None
        if reg == "clear":
            if chance(rng, 0.5):
                consent, heritage, rf, load = "granted", chance(rng, 0.35), round(rng.uniform(4, 30), 1), int(rng.uniform(50, 85))
                setback = round(S * rng.uniform(1.5, 3.0), 1)
                height = min(height, cap - 0.5)
            else:
                consent = pick(rng, ["refused", "granted"])
                if consent == "granted":
                    rf = round(rng.uniform(105, 160), 1)
        elif reg == "borderline":
            k = pick(rng, ["setback", "height", "rf", "load"])
            if k == "setback":
                setback = near(rng, S, S * 0.12, lo=1, digits=1)
            elif k == "height":
                height = near(rng, cap, 0.8, digits=1)
            elif k == "rf":
                rf = near(rng, 100, 12, lo=1, digits=1)
            else:
                load = near(rng, 100, 8, lo=40, integer=True)
        else:
            k = pick(rng, ["strong_need_short_setback", "heritage_tall", "weak_need_clean", "refused_but_best"])
            if k == "strong_need_short_setback":
                setback, drop, util = round(S * rng.uniform(0.7, 0.98), 1), round(rng.uniform(2.5, 4.5), 1), rng.randint(88, 99)
            elif k == "heritage_tall":
                heritage, height = True, round(cap + rng.uniform(0.3, 2.0), 1)
            elif k == "weak_need_clean":
                drop, util, consent = round(rng.uniform(0.2, 0.8), 1), rng.randint(30, 55), "granted"
                setback = round(S * rng.uniform(1.2, 2.5), 1)
            else:
                consent, drop, util = "refused", round(rng.uniform(2.0, 4.0), 1), rng.randint(85, 99)
        if existing is not None and height - existing > 3.0:
            existing = round(height - rng.uniform(0.5, 2.5), 1)
        return {"regime": reg, "case_kind": k, "carrier": _carrier(rng), "city": pick(rng, CITIES),
                "candidate": f"{ident(rng, 'SC', 4, '-')}{pick(rng, 'ABC')}", "structure": struct,
                "existing_m": existing, "height_m": height, "setback_m": setback, "heritage": heritage,
                "rf_pct": rf, "load_pct": load, "consent": consent, "drop_pct": drop, "util_pct": util,
                "alternatives": pick(rng, [0, 1, 1, 2, 3]), "objections": pick(rng, [0, 0, 1, 3, 7, 15]),
                "shot_clock_days": rng.randint(5, 85), "setback_rule": S, "height_cap": cap,
                "minor_mod_pct": pick(rng, [10, 15]), "ground_cabinet": chance(rng, 0.35),
                "street": f"{rng.randint(10, 990)} {pick(rng, ['Mill', 'Station', 'Church', 'Park', 'Orchard', 'Quarry', 'Elm'])} "
                          f"{pick(rng, ['St', 'Rd', 'Ave', 'Lane'])}",
                "engineer": person(rng), "acq": person(rng), "date": rand_date(rng),
                "code": pick(rng, ["WFC 17.40", "Wireless Facilities Code ch. 9", "Small Cell Ord. 2.12"])}

    def _need(self, f):
        n = (2 if f["drop_pct"] >= 2.5 else 1 if f["drop_pct"] >= 1.5 else 0) + (1 if f["util_pct"] >= 85 else 0)
        if f["alternatives"] >= 2 and n > 0:
            n -= 0  # alternatives do not lessen the need itself
        return min(3, n)

    def intended(self, f):
        S, cap = f["setback_rule"], f["height_cap"]
        need = self._need(f)
        setback_ok = f["setback_m"] >= S
        height_ok = f["height_m"] <= cap
        rf_bad = f["rf_pct"] is not None and f["rf_pct"] >= 100
        load_bad = f["load_pct"] is not None and f["load_pct"] > 100
        if f["consent"] == "refused" or rf_bad:
            dec = "drop_candidate" if (f["alternatives"] > 0 or f["consent"] == "refused") else "redesign_before_submitting"
        elif not setback_ok:
            dec = "seek_variance" if (f["setback_m"] >= 0.7 * S and need >= 2) else "redesign_before_submitting"
        elif not height_ok:
            dec = "seek_variance" if need >= 2 else "redesign_before_submitting"
        elif load_bad:
            dec = "redesign_before_submitting"
        elif f["heritage"]:
            dec = "submit_with_concealment"
        else:
            dec = "submit_as_designed"
        existing = STRUCTURES[f["structure"]]
        minor = (existing and f["existing_m"] is not None and not f["ground_cabinet"]
                 and f["height_m"] <= f["existing_m"] * (1 + f["minor_mod_pct"] / 100))
        if f["heritage"]:
            track = "heritage_board_review"
        elif not setback_ok or not height_ok:
            track = "discretionary_hearing"
        elif minor:
            track = "minor_modification_exemption"
        else:
            track = "administrative_permit"
        rf = f["rf_pct"]
        rfc = 3 if rf is None else 0 if rf < 20 else 1 if rf < 50 else 2 if rf < 85 else 3
        return {"siting_decision": dec, "approval_track": track, "rf_exposure_concern": rfc,
                "coverage_need_strength": need, "meets_setback_rule": "yes" if setback_ok else "no"}

    def _rule(self, f):
        return (f"{f['code']}: >= {f['setback_rule']} m from any dwelling; max {f['height_cap']} m overall; "
                f"heritage districts need concealment and board review; existing structures may grow <= "
                f"{f['minor_mod_pct']}% in height without a permit if no ground equipment is added")

    def _rows(self, f, rng):
        return [("Structure", f["structure"] + (f", existing {f['existing_m']} m" if f["existing_m"] else "")),
                ("Proposed height", f"{f['height_m']} m"), ("Nearest dwelling", f"{f['setback_m']} m"),
                ("Ground cabinet", "yes" if f["ground_cabinet"] else "no"),
                ("Heritage district", "yes" if f["heritage"] else "no"),
                ("RF exposure (predicted max, % of limit)", "study pending" if f["rf_pct"] is None else f"{f['rf_pct']}%"),
                ("Structural loading", "analysis pending" if f["load_pct"] is None else f"{f['load_pct']}% of capacity"),
                ("Pole/property owner consent", f["consent"]),
                ("Sector drop rate / busy-hour load", f"{f['drop_pct']}% / {f['util_pct']}%"),
                ("Other candidates in search ring", f["alternatives"]), ("Objections logged", f["objections"])]

    def render(self, f, rng, surface):
        if surface == "json":
            return dumps(prune({
                "candidate": {"id": f["candidate"], "address": f"{f['street']}, {f['city']}", "structure": f["structure"],
                              "existing_height_m": f["existing_m"], "proposed_height_m": f["height_m"],
                              "ground_cabinet": f["ground_cabinet"], "heritage_district": f["heritage"],
                              "nearest_dwelling_m": f["setback_m"]},
                "engineering": {"rf_pct_of_limit": f["rf_pct"], "structural_load_pct": f["load_pct"],
                                "owner_consent": f["consent"]},
                "need": {"sector_drop_rate_pct": f["drop_pct"], "busy_hour_load_pct": f["util_pct"],
                         "other_candidates": f["alternatives"]},
                "city": {"objections_logged": f["objections"], "ordinance": self._rule(f)},
            }))
        if surface == "site_review_form":
            body = _kv(self._rows(f, rng), rng)
            return (f"SMALL CELL SITE REVIEW -- {f['candidate']} ({f['carrier']})\n{f['street']}, {f['city']}  "
                    f"reviewed {fmt_date(f['date'], rng)} by {f['engineer']}\n{body}\nOrdinance: {self._rule(f)}")
        s = [f"Candidate {f['candidate']} at {f['street']} is {'an' if f['structure'][0] in 'aeiou' else 'a'} {f['structure']}"
             + (f" ({f['existing_m']} m today)" if f["existing_m"] else "") + f", proposed at {f['height_m']} m overall.",
             f"Nearest dwelling is {f['setback_m']} m away" + (", and the block is in the heritage district." if f["heritage"] else "."),
             ("RF study still pending." if f["rf_pct"] is None else f"RF predicts {f['rf_pct']}% of the exposure limit at the nearest accessible point."),
             ("Structural analysis not back yet." if f["load_pct"] is None else f"Pole loading comes out at {f['load_pct']}%."),
             f"Owner consent is {f['consent']}. "
             + ("No ground cabinet." if not f["ground_cabinet"] else "Design adds a ground cabinet."),
             f"The sector is at {f['drop_pct']}% drops and {f['util_pct']}% busy-hour load; "
             f"{f['alternatives']} other candidate(s) in the ring; {f['objections']} objection(s) on file so far."]
        return "\n".join([f"Subject: {f['candidate']} -- ready for zoning?", f"From: {f['acq']} (site acquisition)",
                          f"To: {f['engineer']} (RF)", "", join_sentences(s), "",
                          f"City rule ({f['code']}): " + self._rule(f).split(": ", 1)[1] + ".", "", _first(f["acq"])])

    def questions(self):
        return {
            "siting_decision": choice("What should the carrier do with this candidate?", {
                "drop_candidate": "Drop this candidate and move to another in the search ring.",
                "redesign_before_submitting": "Change the design (height, position or equipment) before applying.",
                "seek_variance": "Apply as designed and ask the city for a variance from the ordinance.",
                "submit_as_designed": "Submit the application as designed.",
                "submit_with_concealment": "Submit with a concealed or stealth design to suit the district.",
            }),
            "approval_track": choice("Which approval route would the proposal, as designed, go through?", {
                "administrative_permit": "An ordinary staff-level permit under the ordinance.",
                "discretionary_hearing": "A public hearing, because it departs from a by-right limit.",
                "heritage_board_review": "Review by the heritage board.",
                "minor_modification_exemption": "No permit: it qualifies as a minor change to an existing structure.",
            }),
            "rf_exposure_concern": score("How much concern does the predicted RF exposure raise?", [
                "Negligible: a small fraction of the limit.",
                "Low: comfortably below the limit.",
                "Notable: close enough that design changes could matter.",
                "High: at or over the limit, or no prediction to rely on.",
            ]),
            "coverage_need_strength": score("How strong is the network case for a site here?", [
                "Weak: the sector is performing well.",
                "Moderate: some strain on one measure.",
                "Strong: clear strain on drops or load.",
                "Pressing: heavy strain on both drops and load.",
            ]),
            "meets_setback_rule": noul(
                "The proposed site meets the ordinance's setback from dwellings.",
                yes="The measured distance to the nearest dwelling is at least the ordinance figure.",
                no="The nearest dwelling is closer than the ordinance allows."),
        }


# =========================================================================== schema 3
BANDS = ["Band 12 (700 MHz)", "Band 5 (850 MHz)", "Band 2 (1900 MHz)", "Band 66 (AWS)", "n41 (2.5 GHz)", "n77 (3.7 GHz)"]
SOURCES = {"booster": "consumer signal booster", "camera": "wireless camera transmitter", "ballast": "lighting ballast",
           "licensee": "adjacent-channel licensee's site", "pim": "passive intermodulation at our own site", "unknown": "unknown"}


class InterferenceTriage(Schema):
    name = "interference_complaint_triage"
    description = "An uplink interference report on a carrier sector, triaged under the interference-handling procedure."
    surfaces = {"json": 0.5, "interference_ticket": 0.25, "engineer_chat": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        T = pick(rng, [3, 5, 6])
        rise = round(rng.uniform(0.5, 14), 1)
        pattern = pick_w(rng, {"constant": 1, "intermittent": 1, "business hours only": 1, "tracks our own traffic": 2})
        sectors = pick(rng, [1, 1, 2, 3, 5])
        pim = pick_w(rng, {"pass": 3, "fail": 1.6, "not_run": 3})
        work = pick_w(rng, {None: 5, "antenna swap 3 days ago": 1, "new diplexer installed last week": 1, "jumpers replaced": 1})
        src = pick(rng, list(SOURCES))
        df = pick_w(rng, {"located": 2, "bearing_only": 2, "none": 2, "not_done": 2})
        owner = pick_w(rng, {"cooperating": 2, "refused": 1, "not_contacted": 2})
        k = None
        if reg == "clear":
            if chance(rng, 0.5):
                rise, src, df, pim, work = round(rng.uniform(0.3, T * 0.4), 1), "unknown", "not_done", "not_run", None
            else:
                rise, pim, pattern, sectors, work = round(T + rng.uniform(4, 10), 1), "fail", "tracks our own traffic", 1, pick(rng, ["antenna swap 3 days ago", "jumpers replaced"])
                src = "pim"
        elif reg == "borderline":
            k = pick(rng, ["threshold", "partial_df", "located_owner"])
            if k == "threshold":
                rise = near(rng, T, 1.0, lo=0.1, digits=1)
                if chance(rng, 0.4):
                    pattern, sectors = "tracks our own traffic", 1
                    work = pick(rng, ["antenna swap 3 days ago", "new diplexer installed last week", "jumpers replaced"])
            elif k == "located_owner":
                rise, df, pim = near(rng, T + 1, 1.5, lo=0.1, digits=1), "located", "pass"
                owner, src = pick(rng, ["refused", "refused", "cooperating"]), pick(rng, ["booster", "camera", "ballast"])
                pattern = pick(rng, ["constant", "intermittent", "business hours only"])
            else:
                rise = round(T + rng.uniform(0.5, 6), 1)
                df, pim = "bearing_only", pick(rng, ["pass", "not_run"])
                src = pick(rng, ["booster", "camera", "ballast", "unknown"])
        else:
            k = pick(rng, ["external_but_work", "located_refused", "licensee", "pim_pass_but_pattern"], [1, 1.6, 1.6, 1])
            if k == "external_but_work":
                rise, sectors, pattern = round(T + rng.uniform(1, 8), 1), pick(rng, [3, 5]), "constant"
                work, pim, src = pick(rng, ["antenna swap 3 days ago", "jumpers replaced"]), "pass", pick(rng, ["booster", "camera"])
                df = pick(rng, ["bearing_only", "none"])
            elif k == "located_refused":
                rise, df, owner, src = round(T + rng.uniform(1, 8), 1), "located", "refused", pick(rng, ["booster", "camera", "ballast"])
                pim, pattern = "pass", pick(rng, ["constant", "intermittent", "business hours only"])
            elif k == "licensee":
                rise, df, src, pim = round(T + rng.uniform(0.5, 6), 1), "located", "licensee", "pass"
                owner = pick(rng, ["cooperating", "not_contacted"])
            else:
                rise, pattern, sectors, pim, work = round(T + rng.uniform(0.5, 5), 1), "tracks our own traffic", 1, "pass", None
                src, df = "unknown", "not_done"
        if src == "pim" and pim != "fail" and chance(rng, 0.5):
            src = "unknown"
        if df in ("none", "not_done") and src in ("licensee",):
            src = "unknown"
        if src == "pim":
            df = "not_done"
        ul = int(min(85, max(0, (rise - 1.5) * rng.uniform(3, 7))))
        return {"regime": reg, "case_kind": k, "carrier": _carrier(rng), "market": pick(rng, CITIES),
                "site": ident(rng, "SITE", 5, "-"), "sector": pick(rng, "ABG"), "band": pick(rng, BANDS),
                "rise_db": rise, "pattern": pattern, "sectors": sectors, "pim": pim, "work": work, "suspect": src,
                "df": df, "owner": owner if df == "located" else None, "ul_drop_pct": ul,
                "access_fail_pct": round(min(9.0, max(0.2, rise * rng.uniform(0.1, 0.5))), 1),
                "complainant": pick(rng, ["KPI alarm", "neighbouring licensee", "enterprise account's field team", "drive-test vendor"]),
                "threshold_db": T, "sustain_min": pick(rng, [15, 30, 60]), "ticket": ident(rng, "RFI", 6, "-"),
                "engineer": person(rng), "tech": person(rng), "ts": rand_ts(rng),
                "proc": pick(rng, ["SIP-2", "RF-OPS 11", "IHP-4"]), "distance_m": rng.randint(120, 2400)}

    def _internal(self, f):
        return f["pim"] == "fail" or (f["pattern"] == "tracks our own traffic" and f["sectors"] == 1
                                      and (f["work"] is not None or f["pim"] == "not_run"))

    def intended(self, f):
        exceeds = f["rise_db"] >= f["threshold_db"]
        internal = self._internal(f)
        if internal and (exceeds or f["pim"] == "fail"):
            route = "dispatch_tower_crew"
        elif not exceeds:
            route = "monitor_only"
        elif f["df"] == "located" and f["suspect"] == "licensee":
            route = "coordinate_with_licensee"
        elif f["df"] == "located" and f["owner"] == "refused":
            route = "refer_to_spectrum_authority"
        else:
            route = "field_hunt"
        r = f["rise_db"]
        deg = 0 if r < 2 else 1 if r < 5 else 2 if r < 9 else 3
        if f["sectors"] >= 3 and deg < 3:
            deg += 1
        if f["pim"] == "fail" or (f["df"] == "located" and f["suspect"] != "unknown"):
            conf = 3
        elif f["df"] == "bearing_only" or (internal and f["pim"] == "not_run"):
            conf = 2 if f["suspect"] != "unknown" else 1
        elif f["suspect"] != "unknown":
            conf = 1
        else:
            conf = 0
        return {"interference_route": route, "service_degradation": deg, "source_identification_confidence": conf,
                "exceeds_action_threshold": "yes" if exceeds else "no",
                "source_inside_own_network": "yes" if internal else "no"}

    def _rule(self, f):
        return (f"{f['proc']}: act on uplink noise rise >= {f['threshold_db']} dB over baseline sustained "
                f"{f['sustain_min']} min; rule out own-site PIM before hunting externally")

    def _owner(self, f):
        return {None: None, "cooperating": "cooperating", "refused": "refused access",
                "not_contacted": "not contacted yet"}[f["owner"]]

    def _df(self, f):
        return {"located": f"source located about {f['distance_m']} m from the site",
                "bearing_only": "bearing only, no fix yet", "none": "no usable bearing",
                "not_done": "direction finding not done"}[f["df"]]

    def render(self, f, rng, surface):
        if surface == "json":
            return dumps(prune({
                "report": {"ticket": f["ticket"], "raised_by": f["complainant"], "market": f["market"],
                           "site": f"{f['site']}-{f['sector']}", "band": f["band"]},
                "measurements": {"ul_noise_rise_db": f["rise_db"], "pattern": f["pattern"],
                                 "sectors_affected": f["sectors"], "ul_throughput_drop_pct": f["ul_drop_pct"],
                                 "access_failures_pct": f["access_fail_pct"]},
                "investigation": {"pim_test": f["pim"].replace("_", " "), "recent_site_work": f["work"],
                                  "direction_finding": self._df(f), "suspected_source": SOURCES[f["suspect"]],
                                  "owner_response": self._owner(f)},
                "procedure": self._rule(f),
            }))
        if surface == "interference_ticket":
            body = _kv([("Ticket", f["ticket"]), ("Raised by", f["complainant"]),
                             ("Site / sector / band", f"{f['site']}-{f['sector']} / {f['band']}"),
                             ("UL noise rise", f"{f['rise_db']} dB ({f['pattern']})"),
                             ("Sectors affected", f["sectors"]),
                             ("KPI", (f"UL throughput -{f['ul_drop_pct']}%" if f["ul_drop_pct"] else "UL throughput unchanged")
                                     + f", access failures {f['access_fail_pct']}%"),
                             ("PIM test", f["pim"].replace("_", " ")), ("Recent site work", f["work"] or "none logged"),
                             ("DF", self._df(f)), ("Suspected source", SOURCES[f["suspect"]]),
                             ("Owner response", self._owner(f))], rng)
            return f"INTERFERENCE TICKET  {fmt_ts(f['ts'], rng)}\n{body}\n{self._rule(f)}"
        msgs = [(f["engineer"], f"seeing UL noise up {f['rise_db']} dB on {f['site']}-{f['sector']} {f['band']}, {f['pattern']}, "
                                f"{f['sectors']} sector(s) showing it")]
        msgs.append((f["engineer"], (f"throughput down {f['ul_drop_pct']}% on the uplink" if f["ul_drop_pct"] else "no uplink throughput hit")
                     + f", access fails {f['access_fail_pct']}%"))
        msgs.append((f["tech"], {"pass": "PIM test passed on all ports", "fail": "PIM test failed on port 2",
                                 "not_run": "haven't been up to run PIM yet"}[f["pim"]]))
        if f["work"]:
            msgs.append((f["engineer"], f"fwiw {f['work']}"))
        msgs.append((f["tech"], self._df(f) + ("" if f["suspect"] == "unknown" else f", looks like a {SOURCES[f['suspect']]}")))
        if f["owner"]:
            msgs.append((f["tech"], {"cooperating": "owner says they'll switch it off", "refused": "owner refused to let us in or switch it off",
                                     "not_contacted": "haven't reached the owner"}[f["owner"]]))
        t = f["ts"]
        out = []
        for who, m in msgs:
            out.append(f"[{t.strftime('%H:%M')}] {_first(who).lower()}: {noisy(m, rng, 0.1)}")
            t += timedelta(minutes=rng.randint(2, 30))
        return f"#rf-interference ({f['ticket']}, raised by {f['complainant']})\n" + "\n".join(out) + f"\n-- {self._rule(f)}"

    def questions(self):
        return {
            "interference_route": choice("How should the interference report be routed?", {
                "coordinate_with_licensee": "Work it out with the neighbouring licensee whose site is the source.",
                "dispatch_tower_crew": "Send a tower crew to fix a fault in the carrier's own equipment.",
                "field_hunt": "Send an interference hunter with direction-finding gear to find the source.",
                "monitor_only": "Log it and keep watching; it does not warrant action yet.",
                "refer_to_spectrum_authority": "Refer it to the spectrum authority for enforcement.",
            }),
            "service_degradation": score("How much is the interference hurting service?", [
                "Barely: within normal variation.",
                "Somewhat: a measurable dip on one sector.",
                "Clearly: users on the sector are getting worse service.",
                "Badly: heavy degradation or several sectors hit.",
            ]),
            "source_identification_confidence": score("How confidently has the source been identified?", [
                "Not at all: no evidence of where it comes from.",
                "Weakly: a guess with little to back it up.",
                "Partly: a bearing or a test points one way, without a fix.",
                "Firmly: located on the ground or confirmed by a test.",
            ]),
            "exceeds_action_threshold": noul(
                "The measured noise rise meets the procedure's action threshold.",
                yes="The rise over baseline is at or above the procedure's figure.",
                no="The rise is below the procedure's figure."),
            "source_inside_own_network": noul("The interference most likely comes from the carrier's own equipment."),
        }


# =========================================================================== schema 4
class CapacityUpgradeTrigger(Schema):
    name = "capacity_upgrade_trigger"
    description = "A sector's busy-hour load and backhaul use against the capacity-planning trigger: what upgrade, if any, goes in the plan."
    surfaces = {"json": 0.5, "planning_dashboard": 0.25, "planner_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        U = pick(rng, [70, 75, 80, 85])
        kk = pick(rng, [3, 4])
        bh_trig = pick(rng, [70, 80])
        base = U * rng.uniform(0.65, 1.15)
        weeks = [round(min(99, base * rng.uniform(0.93, 1.07)), 1) for _ in range(6)]
        anomaly = None
        bh = round(rng.uniform(35, 90), 1)
        sinr = pick_w(rng, {"normal": 4, "poor": 1.5})
        users = pick(rng, ["growing", "flat"])
        growth = round(rng.uniform(1, 14), 1)
        spare = chance(rng, 0.45)
        event = None
        hist = pick(rng, [3, 12, 24])
        k = None
        if reg == "clear":
            if chance(rng, 0.5):
                weeks = [round(U * rng.uniform(0.45, 0.7), 1) for _ in range(6)]
                bh, growth, sinr = round(rng.uniform(25, bh_trig * 0.6), 1), round(rng.uniform(0.5, 3), 1), "normal"
            else:
                weeks = [round(min(99, U * rng.uniform(1.04, 1.2)), 1) for _ in range(6)]
                sinr, users, growth = "normal", "growing", round(rng.uniform(6, 14), 1)
        elif reg == "borderline":
            k = pick(rng, ["weeks", "weeks", "weeks", "backhaul", "backhaul"])
            if k == "weeks":
                weeks = [round(near(rng, U, 4), 1) for _ in range(6)]
                if chance(rng, 0.3):
                    anomaly = rng.randrange(6)
                    weeks[anomaly] = round(min(99, U * rng.uniform(1.05, 1.25)), 1)
            else:
                bh = near(rng, bh_trig, 4, digits=1)
                weeks = [round(U * rng.uniform(0.62, 1.0), 1) for _ in range(6)]
        else:
            k = pick(rng, ["counter_glitch", "interference", "event", "backhaul_hidden"])
            if k == "counter_glitch":
                weeks = [round(U * rng.uniform(0.75, 0.97), 1) for _ in range(6)]
                high = rng.sample(range(6), kk)
                for i in high:
                    weeks[i] = round(min(99, U * rng.uniform(1.02, 1.15)), 1)
                anomaly = pick(rng, high)
            elif k == "interference":
                weeks = [round(min(99, U * rng.uniform(0.98, 1.15)), 1) for _ in range(6)]
                sinr, users, growth = "poor", "flat", round(rng.uniform(0, 2), 1)
            elif k == "event":
                event = f"stadium concert in {rng.randint(3, 8)} weeks"
                weeks = [round(U * rng.uniform(0.7, 0.95), 1) for _ in range(6)]
                hist = 3
            else:
                weeks = [round(U * rng.uniform(0.6, 0.85), 1) for _ in range(6)]
                bh = round(bh_trig + rng.uniform(3, 15), 1)
        if anomaly is None and chance(rng, 0.12):
            anomaly = rng.randrange(6)
        nb = [round(U * rng.uniform(0.35, 1.05), 1) for _ in range(2)]
        return {"regime": reg, "case_kind": k, "carrier": _carrier(rng), "market": pick(rng, CITIES),
                "sector": f"{ident(rng, 'S', 5, '')}-{pick(rng, ['A', 'B', 'G'])}", "tech": pick(rng, ["LTE", "5G NR"]),
                "neighbour_load": nb, "bh_users": rng.randint(60, 420), "busy_hour": pick(rng, ["18:00", "19:00", "20:00", "21:00", "12:00"]),
                "weeks": weeks, "anomaly_week": anomaly, "backhaul_p95": bh, "sinr": sinr, "users": users,
                "growth_qtr_pct": growth, "spare_spectrum": spare, "event": event, "history_months": hist,
                "prb_trigger": U, "weeks_needed": kk, "bh_trigger": bh_trig, "planner": person(rng),
                "date": rand_date(rng), "rule_id": pick(rng, ["CPG-4", "Capacity Planning Rule 2.1", "NPL-7"])}

    def _valid(self, f):
        return [v for i, v in enumerate(f["weeks"]) if i != f["anomaly_week"]]

    def intended(self, f):
        U = f["prb_trigger"]
        valid = self._valid(f)
        hits = sum(v >= U for v in valid)
        hits_raw = sum(v >= U for v in f["weeks"])
        radio = hits >= f["weeks_needed"]
        bh_hit = f["backhaul_p95"] >= f["bh_trigger"]
        interf = f["sinr"] == "poor" and f["users"] == "flat"
        if f["anomaly_week"] is not None and hits_raw >= f["weeks_needed"] and not radio:
            neck = "counter_data_error"
        elif interf and max(valid) >= 0.9 * U:
            neck = "interference_not_demand"
        elif bh_hit and not radio:
            neck = "backhaul_transport"
        elif radio:
            neck = "radio_capacity"
        else:
            neck = "radio_capacity" if max(valid) / U >= f["backhaul_p95"] / f["bh_trigger"] else "backhaul_transport"
        if neck == "counter_data_error":
            act = "monitor_only"
        elif neck == "interference_not_demand":
            act = "optimise_parameters"
        elif bh_hit and neck == "backhaul_transport":
            act = "upgrade_backhaul"
        elif radio and sum(f["neighbour_load"]) / 2 < U - 25 and f["growth_qtr_pct"] < 8:
            act = "optimise_parameters"
        elif radio:
            act = "add_carrier" if f["spare_spectrum"] else "densify_new_site"
        elif max(valid) >= U - 5 and f["growth_qtr_pct"] >= 6:
            act = "optimise_parameters"
        else:
            act = "monitor_only"
        recent = sum(valid[-3:]) / len(valid[-3:])
        g = f["growth_qtr_pct"] / 100
        if radio and recent >= U:
            tim = 3
        elif recent >= U:
            tim = 2
        else:
            q = math.log(U / recent) / math.log(1 + g) if g > 0 else 99
            tim = 2 if q <= 1 else 1 if q <= 4 else 0
        if f["event"] and tim < 2:
            tim += 1
        conf = 3 - (1 if f["anomaly_week"] is not None else 0) - (1 if f["event"] else 0) - (1 if f["history_months"] == 3 else 0)
        spread = max(valid) - min(valid)
        if spread > 0.2 * U:
            conf -= 1
        return {"upgrade_action": act, "binding_constraint": neck, "congestion_timing": tim,
                "forecast_confidence": max(0, conf), "trigger_met": "yes" if radio else "no"}

    def _rule(self, f):
        return (f"{f['rule_id']}: upgrade when busy-hour PRB use >= {f['prb_trigger']}% in {f['weeks_needed']} of "
                f"the last 6 weeks, or backhaul 95th percentile >= {f['bh_trigger']}%")

    def _anom(self, f):
        if f["anomaly_week"] is None:
            return None
        return f"week {f['anomaly_week'] + 1}: counters reset after a software load, value suspect"

    def render(self, f, rng, surface):
        if surface == "json":
            return dumps(prune({
                "sector": {"id": f["sector"], "market": f["market"], "technology": f["tech"],
                           "spare_spectrum_licensed": f["spare_spectrum"]},
                "load": {"busy_hour_prb_pct_last_6_weeks": f["weeks"], "data_note": self._anom(f),
                         "busy_hour": f["busy_hour"], "busy_hour_users": f["bh_users"],
                         "neighbour_sectors_prb_pct": f["neighbour_load"],
                         "backhaul_p95_pct": f["backhaul_p95"], "avg_sinr": f["sinr"], "active_users_trend": f["users"]},
                "forecast": {"traffic_growth_pct_per_quarter": f["growth_qtr_pct"], "history_months": f["history_months"],
                             "known_event": f["event"]},
                "planning_rule": self._rule(f),
            }))
        if surface == "planning_dashboard":
            rows = [[f"W{i + 1}", f"{v}%" + (" *" if i == f["anomaly_week"] else "")] for i, v in enumerate(f["weeks"])]
            return "\n".join([
                f"CAPACITY DASHBOARD  {f['sector']} ({f['tech']}, {f['market']})  as of {fmt_date(f['date'], rng)}",
                table(["week", "busy-hour PRB"], rows, rng),
                ("* " + self._anom(f)) if f["anomaly_week"] is not None else "",
                _kv([("Busy hour", f"{f['busy_hour']}, ~{f['bh_users']} active users"),
                          ("Neighbour sectors PRB", " / ".join(f"{v}%" for v in f["neighbour_load"])),
                          ("Backhaul p95", f"{f['backhaul_p95']}%"), ("Avg SINR", f["sinr"]),
                          ("Active users", f["users"]), ("Growth / quarter", f"{f['growth_qtr_pct']}% ({f['history_months']} months of data)"),
                          ("Spare spectrum", "yes" if f["spare_spectrum"] else "no"), ("Known event", f["event"])], rng),
                self._rule(f)]).replace("\n\n", "\n")
        s = [f"Sector {f['sector']} ({f['tech']}) in {f['market']} ran {', '.join(str(v) for v in f['weeks'])}% busy-hour PRB over the last six weeks.",
             (self._anom(f).capitalize() + "." if f["anomaly_week"] is not None else ""),
             f"Busy hour is around {f['busy_hour']} with ~{f['bh_users']} active users ({f['users']}).",
             f"The two neighbouring sectors run {f['neighbour_load'][0]}% and {f['neighbour_load'][1]}% PRB.",
             f"Backhaul p95 is {f['backhaul_p95']}%, SINR {f['sinr']}.",
             f"Growth is running {f['growth_qtr_pct']}% a quarter on {f['history_months']} months of history"
             + (f"; {f['event']}." if f["event"] else "."),
             "Spare licensed spectrum is available here." if f["spare_spectrum"] else "No spare spectrum in this market."]
        return "\n".join([f"MEMO -- capacity review, {f['sector']}", f"From: {f['planner']}, RAN planning", "",
                          join_sentences(s), "", f"Rule: {self._rule(f)}"])

    def questions(self):
        return {
            "upgrade_action": choice("What should the capacity plan do for this sector?", {
                "add_carrier": "Add a carrier on spare spectrum at the existing site.",
                "densify_new_site": "Add a new sector or small cell to take load off this one.",
                "monitor_only": "No upgrade now; keep watching the sector.",
                "optimise_parameters": "Tune parameters (tilt, load balancing, neighbour settings) instead of building.",
                "upgrade_backhaul": "Increase the backhaul capacity serving the site.",
            }),
            "binding_constraint": choice("What is really limiting this sector at busy hour?", {
                "backhaul_transport": "The transport link behind the site.",
                "counter_data_error": "Nothing real: the high readings come from bad counter data.",
                "interference_not_demand": "Poor radio conditions inflating resource use, not real demand.",
                "radio_capacity": "Genuine demand on the radio resources.",
            }),
            "congestion_timing": score("How soon will users feel congestion at busy hour on this sector?", [
                "Not within the planning year.",
                "Within the next few quarters.",
                "Within about a quarter, before a normal build could land.",
                "Already: the sector is congested at busy hour now.",
            ]),
            "forecast_confidence": score("How much confidence does the load forecast deserve?", [
                "Little: the data are short, distorted or event-driven.",
                "Limited: one significant weakness in the data.",
                "Reasonable: decent history with minor noise.",
                "High: long, clean and consistent history.",
            ]),
            "trigger_met": noul(
                "The sector meets the planning rule's busy-hour radio trigger.",
                yes="Enough valid weeks are at or above the PRB figure the rule sets.",
                no="Too few valid weeks reach the PRB figure, once suspect data are set aside."),
        }


# =========================================================================== schema 5
REASONS = {"account_mismatch": "account number does not match", "pin_mismatch": "port-out PIN incorrect",
           "name_mismatch": "subscriber name does not match", "outstanding_balance": "outstanding balance on the account",
           "contract_term": "number is in a minimum-term contract", "number_inactive": "number not active on our network",
           "no_authorisation": "no subscriber authorisation supplied"}
ALWAYS_OK = {"account_mismatch", "pin_mismatch", "number_inactive", "no_authorisation"}


class NumberPortDispute(Schema):
    name = "number_port_dispute"
    description = "A rejected number-port request disputed between carriers, judged on the records against the portability code."
    surfaces = {"json": 0.5, "dispute_form": 0.25, "carrier_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        name_ok = chance(rng, 0.4)  # whether this code version lets a name mismatch ground a rejection
        reason = pick(rng, list(REASONS))
        acct = pick_w(rng, {"match": 5, "one_digit": 2, "different": 1})
        pin = pick_w(rng, {"match": 5, "wrong": 2, "not_supplied": 1})
        name = pick_w(rng, {"exact": 5, "abbreviated": 2, "different": 1})
        auth = pick_w(rng, {"signed_loa": 4, "recorded_verbal": 2, "none": 1.5, None: 1})
        active = chance(rng, 0.9)
        window = pick(rng, [1, 2])
        multi = chance(rng, 0.3)
        if multi:
            window = pick(rng, [3, 4])
        days = rng.randint(1, 9)
        k = None
        if reg == "clear":
            if chance(rng, 0.5):
                reason = pick(rng, ["outstanding_balance", "contract_term"])
                acct, pin, name, auth = "match", "match", "exact", "signed_loa"
            else:
                reason = pick(rng, ["account_mismatch", "pin_mismatch"])
                acct, pin = ("different", "match") if reason == "account_mismatch" else ("match", "wrong")
        elif reg == "borderline":
            k = pick(rng, ["one_digit", "name_variant", "deadline"])
            if k == "one_digit":
                reason, acct = "account_mismatch", pick(rng, ["one_digit", "match"])
            elif k == "name_variant":
                reason, name = "name_mismatch", pick(rng, ["abbreviated", "different"])
            else:
                reason = pick(rng, ["outstanding_balance", "contract_term", "name_mismatch"])
                days = near(rng, window, 1, lo=1, integer=True)
        else:
            k = pick(rng, ["valid_reason_wrong_label", "auth_missing_but_verbal", "inactive_disputed"])
            if k == "valid_reason_wrong_label":
                reason, acct = "outstanding_balance", pick(rng, ["one_digit", "different"])
            elif k == "auth_missing_but_verbal":
                reason, auth = "no_authorisation", pick(rng, ["recorded_verbal", None])
            else:
                reason, active = "number_inactive", chance(rng, 0.4)
        gaining, losing = _carrier(rng), _carrier(rng)
        while losing.split()[0] == gaining.split()[0]:
            losing = _carrier(rng)
        main = f"+1 {rng.randint(201, 989)} 555 01{rng.randint(0, 99):02d}"
        return {"regime": reg, "case_kind": k, "gaining": gaining, "losing": losing,
                "number": main if not multi else f"{rng.randint(4, 40)} numbers, main {main}",
                "multi_line": multi, "reason": reason, "acct": acct, "pin": pin, "name": name, "auth": auth,
                "active": active, "name_ground_allowed": name_ok, "window_bd": window, "days_elapsed": days,
                "business_line": chance(rng, 0.45) or multi, "service_lost": chance(rng, 0.3),
                "comp_per_day": pick(rng, [10, 15, 25]), "case_id": ident(rng, "PAB", 5, "-"),
                "subscriber": f"Subscriber {ident(rng, 'S', 6, '-')}", "analyst": person(rng), "date": rand_date(rng),
                "code": pick(rng, ["NPC-2", "Portability Code 2026", "NP Rules s.4"])}

    def _valid(self, f):
        r = f["reason"]
        if r == "name_mismatch":
            return f["name_ground_allowed"] and f["name"] == "different"
        if r not in ALWAYS_OK:
            return False
        if r == "account_mismatch":
            return f["acct"] != "match"
        if r == "pin_mismatch":
            return f["pin"] != "match"
        if r == "number_inactive":
            return not f["active"]
        return f["auth"] == "none"

    def intended(self, f):
        valid = self._valid(f)
        if valid:
            ruling = "uphold_rejection"
        elif f["auth"] in (None, "none"):
            ruling = "request_missing_records"
        elif f["days_elapsed"] > f["window_bd"]:
            ruling = "order_release_with_compensation"
        else:
            ruling = "order_release"
        mism = (f["acct"] != "match") + (f["pin"] != "match") + (f["name"] == "different") + (f["name"] == "abbreviated") * 0.5
        rc = 3 if mism == 0 else 2 if mism <= 0.5 else 1 if mism <= 1.5 else 0
        over = max(0, f["days_elapsed"] - f["window_bd"])
        harm = (1 if f["business_line"] else 0) + (1 if f["service_lost"] else 0) + (1 if over >= 2 else 0)
        if over == 0 and harm > 0:
            harm -= 1
        return {"dispute_ruling": ruling, "record_consistency": rc, "subscriber_harm": min(3, harm),
                "rejection_ground_valid": "yes" if valid else "no",
                "authorisation_on_file": "yes" if f["auth"] in ("signed_loa", "recorded_verbal") else "no"}

    def _rule(self, f):
        grounds = "number not active, account number or PIN mismatch, no subscriber authorisation"
        if f["name_ground_allowed"]:
            grounds += ", subscriber name clearly different"
        return (f"{f['code']} cl.4: a losing carrier may reject only for: {grounds}. Balances and contract terms "
                f"are not grounds. Ports complete within {f['window_bd']} business day(s); a late port earns "
                f"{f['comp_per_day']} credits per day")

    def _records(self, f):
        return {"account_number": {"match": "matches", "one_digit": "differs by one digit", "different": "different"}[f["acct"]],
                "pin": {"match": "correct", "wrong": "wrong", "not_supplied": "not supplied"}[f["pin"]],
                "name": {"exact": "matches", "abbreviated": "abbreviated form of the account name", "different": "different name"}[f["name"]],
                "authorisation": {"signed_loa": "signed letter of authorisation", "recorded_verbal": "recorded verbal consent",
                                  "none": "none on file", None: "not provided to the board"}[f["auth"]],
                "number_status_at_losing_carrier": "active" if f["active"] else "inactive"}

    def render(self, f, rng, surface):
        rec = self._records(f)
        if surface == "json":
            return dumps(prune({
                "dispute": {"case": f["case_id"], "gaining_carrier": f["gaining"], "losing_carrier": f["losing"],
                            "numbers": f["number"], "subscriber": f["subscriber"],
                            "line_type": "business multi-line" if f["multi_line"] else ("business" if f["business_line"] else "residential"),
                            "business_days_since_request": f["days_elapsed"], "subscriber_without_service": f["service_lost"]},
                "rejection": {"reason_given": REASONS[f["reason"]]},
                "records_compared": rec,
                "code": self._rule(f),
            }))
        if surface == "dispute_form":
            body = _kv([("Case", f["case_id"]), ("Gaining / losing", f"{f['gaining']} / {f['losing']}"),
                             ("Number(s)", f["number"]), ("Subscriber", f["subscriber"]),
                             ("Rejection reason", REASONS[f["reason"]]),
                             ("Account number in request", rec["account_number"]), ("PIN", rec["pin"]),
                             ("Name", rec["name"]), ("Authorisation", rec["authorisation"]),
                             ("Number status", rec["number_status_at_losing_carrier"]),
                             ("Business days since request", f["days_elapsed"]),
                             ("Subscriber out of service", "yes" if f["service_lost"] else "no")], rng)
            return f"PORTABILITY ADMINISTRATION BOARD -- DISPUTE\n{body}\n{self._rule(f)}"
        s = [f"We are disputing the rejection of our port request for {f['number']} ({f['subscriber']}).",
             f"{f['losing']} rejected it with \"{REASONS[f['reason']]}\".",
             "Against their records, our request's account number "
             + {"match": "matches", "one_digit": "differs by one digit", "different": "does not match"}[f["acct"]]
             + ", the PIN " + {"match": "was correct", "wrong": "was wrong", "not_supplied": "was not supplied"}[f["pin"]]
             + ", and the name " + {"exact": "matches", "abbreviated": "is an abbreviated form of the account name",
                                    "different": "is a different name"}[f["name"]] + ".",
             f"Authorisation: {rec['authorisation']}.",
             f"It has been {f['days_elapsed']} business day(s) since the request"
             + (" and the customer has no working service." if f["service_lost"] else ".")]
        return "\n".join([f"Subject: Port dispute {f['case_id']} -- {f['losing']}", f"From: porting desk, {f['gaining']}",
                          f"To: {f['analyst']}, Portability Administration Board", "", join_sentences(s), "",
                          f"Relevant code: {self._rule(f)}."])

    def questions(self):
        return {
            "dispute_ruling": choice("How should the board rule on the port dispute?", {
                "order_release": "Order the losing carrier to release the number now.",
                "order_release_with_compensation": "Order the release now and award per-day compensation for the delay.",
                "request_missing_records": "Ask for the missing records before ruling.",
                "uphold_rejection": "Uphold the rejection; the gaining carrier must submit a corrected request.",
            }),
            "record_consistency": score("How well do the port request details agree with the losing carrier's records?", [
                "Poorly: several details conflict.",
                "Partly: one clear conflict.",
                "Largely: only a minor variant, such as an abbreviated name.",
                "Fully: every detail agrees.",
            ]),
            "subscriber_harm": score("How much harm is the delay doing to the subscriber?", [
                "Little: a residential line still working on the old carrier.",
                "Some: inconvenience or a short delay.",
                "Significant: a business line or lost service.",
                "Severe: a business without service well past the deadline.",
            ]),
            "rejection_ground_valid": noul(
                "The losing carrier's rejection rests on a ground the code permits, and the records bear it out.",
                yes="The stated reason is a permitted ground and the compared records support it.",
                no="The reason is not a permitted ground, or the records do not support it."),
            "authorisation_on_file": noul("A valid subscriber authorisation for the port is on file."),
        }


WORKFLOW = Workflow(
    name="telecom_network_operations",
    title="Telecom network operations",
    description="Carrier-side network decisions (outage reporting, siting, interference, capacity, porting disputes) "
                "against fictional rules.",
    domain="telecommunications carriers (NAICS 517)",
    schemas=[OutageReportability(), SmallCellSiting(), InterferenceTriage(), CapacityUpgradeTrigger(), NumberPortDispute()],
    notes="Network side only: no subscriber conversations, no churn, no maliciousness questions. Regulators, cities, "
          "carriers and codes are fictional.",
)
