"""Environmental compliance reports: what a (fictional) facility must report, to whom, and how seriously to treat it.

Discharge-permit exceedances, continuous-emission-monitor deviations, spills and releases, corrective-action plans
and the monthly monitoring report before it is certified. Facilities, permits, the state agency ("Department of
Environmental Stewardship") and every limit, threshold and reporting clause are invented; they follow the usual
shape of a water or air permit. Each schema draws a `regime` first, so most cases sit near a limit or a reporting
threshold, carry a lab or monitor quality flag that undercuts the headline number, or leave the timing unclear.
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

AGENCY = "Department of Environmental Stewardship"
# name: (sector, receiving water and its use, sensitivity 0-3, typical water parameters)
FACILITIES = {
    "Pellbridge Water Reclamation Plant": ("municipal wastewater", "Tidemill Creek (shellfish beds 2 km downstream)", 3,
                                           ["ammonia", "tss", "ecoli", "bod", "chlorine"]),
    "Driftwood Poultry Processing": ("food processing", "Millrace River (designated trout water)", 2,
                                     ["bod", "tss", "ammonia", "ph"]),
    "Copperline Metal Finishing": ("metal finishing", "Heron Brook (designated trout water)", 2, ["zinc", "ph", "tss"]),
    "Saltmarsh Paper Mill": ("pulp and paper", "Saltmarsh Estuary (tidal, commercial fishing)", 2, ["bod", "tss", "ph"]),
    "Stonewick Quarry": ("aggregate quarry", "Quarry Run (drains to an industrial canal)", 1, ["tss", "ph"]),
    "Juniper Hollow Dairy": ("dairy processing", "Hollow Brook (small stream, low summer flow)", 2, ["bod", "ammonia", "tss"]),
}
# parameter: (label, unit, limit, basis, toxic-listed)
PARAMS = {
    "ammonia": ("ammonia (as N)", "mg/L", 4.0, "daily maximum", True),
    "tss": ("total suspended solids", "mg/L", 30.0, "monthly average", False),
    "ecoli": ("E. coli", "cfu/100 mL", 410.0, "daily maximum", False),
    "bod": ("BOD5", "mg/L", 45.0, "weekly average", False),
    "chlorine": ("total residual chlorine", "mg/L", 0.019, "daily maximum", True),
    "zinc": ("total zinc", "ug/L", 120.0, "daily maximum", True),
    "ph": ("pH", "SU", 9.0, "instantaneous maximum", False),
}


def _fmt(v, param):
    if param == "chlorine":
        return f"{v:.3f}"
    if param in ("ecoli", "zinc"):
        return f"{int(round(v))}"
    return f"{v:.1f}"


# =========================================================================== schema 1
class DischargeExceedance(Schema):
    name = "discharge_exceedance"
    description = "A wastewater discharge result over its permit limit, as first reported by the lab."
    surfaces = {"json": 0.5, "lab_result_email": 0.25, "compliance_log": 0.25}

    CAUSES = {"storm_inflow": "heavy rain overnight, flow through treatment 2.6x normal",
              "aeration_failure": "one aeration blower tripped for about 9 hours",
              "dosing_pump": "chemical dosing pump lost prime during the night shift",
              "cleaning_discharge": "tank cleaning wash-water sent to treatment without notice",
              "unknown": None}
    SECTOR_CAUSES = {"aggregate quarry": ["storm_inflow", "dosing_pump", "unknown"],
                     "metal finishing": ["storm_inflow", "dosing_pump", "cleaning_discharge", "unknown"]}
    QC = {"none": None, "holding_time": "sample analysed 4 h past its holding time",
          "temperature": "sample received at 11 C (should be at or below 6 C)",
          "duplicate": "field duplicate differs by 38%", "blank": "method blank showed low-level contamination"}

    def sample_factors(self, rng):
        reg = regime(rng)
        fac = pick(rng, list(FACILITIES))
        sector, water, sens, plist = FACILITIES[fac]
        param = pick(rng, plist)
        label, unit, limit, basis, toxic = PARAMS[param]
        ratio = round(rng.uniform(1.03, 2.8), 2)
        others = [round(limit * rng.uniform(0.3, 0.9), 3) for _ in range(pick(rng, [2, 3, 4]))]
        qc = pick_w(rng, {"none": 6, "holding_time": 1, "temperature": 1, "duplicate": 1, "blank": 1})
        resample = pick_w(rng, {"below": 4, "above": 2, "pending": 2, "not_taken": 1})
        cause = pick(rng, self.SECTOR_CAUSES.get(sector, list(self.CAUSES)))
        prior = pick(rng, [0, 0, 1, 2, 4])
        hours = pick(rng, [1, 3, 8, 20, 40])
        clause_shown = chance(rng, 0.65)
        if reg == "clear":
            if chance(rng, 0.5):   # a real, large exceedance of a listed parameter
                ratio, qc, resample = round(rng.uniform(2.0, 3.5), 2), "none", pick(rng, ["above", "pending"])
                if not toxic and chance(rng, 0.5):
                    param = pick(rng, [p for p in plist if PARAMS[p][4]] or [param])
            else:                  # a marginal averaged parameter with a clean resample
                param = pick(rng, [p for p in plist if "average" in PARAMS[p][3]] or [param])
                ratio, qc, resample, prior = round(rng.uniform(1.02, 1.1), 2), "none", "below", 0
        elif reg == "borderline":
            ratio = near(rng, 1.06, 0.06, lo=1.005)
            if chance(rng, 0.4):
                qc = pick(rng, ["holding_time", "temperature", "duplicate"])
        else:
            # a big number the lab's own QC undercuts, or a small one on a sensitive water with a bad history
            if chance(rng, 0.5):
                ratio, qc = round(rng.uniform(1.8, 3.0), 2), pick(rng, ["holding_time", "temperature", "blank"])
                resample = "below"
            else:
                ratio, prior = round(rng.uniform(1.03, 1.2), 2), pick(rng, [3, 4, 5])
                cause = "unknown"
            if chance(rng, 0.4):
                resample = "pending"
        label, unit, limit, basis, toxic = PARAMS[param]
        value = round(limit * ratio, 3)
        if param == "ph":
            value = round(9.0 + (ratio - 1) * 1.5, 2)
        others = [round(limit * rng.uniform(0.3, 0.9), 3) for _ in range(len(others))]
        return {"regime": reg, "facility": fac, "sector": sector, "receiving_water": water, "water_sensitivity": sens,
                "parameter": param, "limit": limit, "basis": basis, "toxic_listed": toxic, "value": value,
                "exceed_ratio": ratio, "other_results_this_period": others, "lab_qc_flag": qc,
                "resample": resample, "cause": cause, "exceedances_12mo": prior, "hours_since_result": hours,
                "permit_clause_shown": clause_shown, "permit": ident(rng, "WD", 5, "-"),
                "lab": pick(rng, ["Pellbridge Analytical", "Quillhaven Environmental Labs", "Wexmoor Water Testing"]),
                "officer": person(rng), "operator": person(rng), "sampled": rand_ts(rng, hours=(6, 11))}

    CLAUSE = (f"Permit Part II.D: report orally to the {AGENCY} within 24 hours any noncompliance that may endanger "
              "health or the environment, any upset or bypass, and any violation of a daily maximum for a parameter "
              "listed in Part III; report all other noncompliance with the monthly monitoring report.")

    def _period_avg(self, f):
        """For an averaged limit: the period average including this result, the number the permit judges."""
        label, unit, limit, basis, _ = PARAMS[f["parameter"]]
        if "average" not in basis:
            return None
        vals = list(f["other_results_this_period"]) + [f["value"]]
        return f"{_fmt(sum(vals) / len(vals), f['parameter'])} {unit} ({basis} so far, including this result)"

    def _resample(self, f):
        label, unit, limit, _, _ = PARAMS[f["parameter"]]
        if f["parameter"] == "ph":     # pH resamples stay inside the plausible 6-10 band
            lo_v, hi_v = "8.3", f"{min(9.8, f['value'] + 0.1):.1f}"
        else:
            lo_v, hi_v = _fmt(limit * 0.55, f["parameter"]), _fmt(limit * 1.3, f["parameter"])
        return {"below": f"resample {lo_v} {unit}",
                "above": f"resample {hi_v} {unit}", "pending": "resample taken, result pending",
                "not_taken": "no resample taken"}[f["resample"]]

    def render(self, f, rng, surface):
        p = f["parameter"]
        label, unit, limit, basis, toxic = PARAMS[p]
        val = f"{_fmt(f['value'], p)} {unit}"
        lim = f"{_fmt(limit, p)} {unit} ({basis})"
        others = [f"{_fmt(x, p)}" for x in f["other_results_this_period"]]
        qc = self.QC[f["lab_qc_flag"]]
        cause = self.CAUSES[f["cause"]]
        water = f["receiving_water"]
        t = f["sampled"]
        if surface == "json":
            return dumps(prune({
                "facility": {"name": f["facility"], "sector": f["sector"], "permit": f["permit"], "receiving_water": water},
                "result": {"parameter": label, "value": val, "permit_limit": lim, "sampled": fmt_ts(t, style="iso"),
                           "lab": f["lab"], "lab_qc_note": qc,
                           "other_results_same_period": others if "average" in basis else None,
                           "period_average_with_this_result": self._period_avg(f),
                           "part_iii_listed": toxic or None},
                "follow_up": {"resample": self._resample(f), "suspected_cause": cause,
                              "hours_since_result_received": f["hours_since_result"]},
                "history": {"exceedances_last_12_months": f["exceedances_12mo"]},
                "permit_reporting_clause": self.CLAUSE if f["permit_clause_shown"] else None,
            }))
        if surface == "lab_result_email":
            body = [f"Subject: RESULT OVER LIMIT -- {f['facility']} outfall 001, {label}",
                    f"From: reporting@{f['lab'].split()[0].lower()}labs.example", f"To: {f['officer']}", "",
                    f"Sample collected {fmt_ts(t, rng)} at outfall 001.", "",
                    kv_lines([("Parameter", label), ("Result", val), ("Permit limit (per your COC)", lim),
                              ("QC", qc or "all QC within acceptance limits")], rng),
                    ""]
            if "average" in basis:
                body.append(f"Earlier results this period: {', '.join(others)} {unit}. Period average with this "
                            f"result: {self._period_avg(f)}.")
            body += ["", "Please call if you need a re-run.", f"{f['lab']} client services"]
            tail = [f"--- forwarded by {f['officer']}, {f['hours_since_result']} h after receipt ---",
                    noisy(join_sentences([
                        f"Operator ({f['operator']}) says: {cause}." if cause else "Operators have no explanation yet.",
                        self._resample(f).capitalize() + ".",
                        f"We have had {f['exceedances_12mo']} exceedance(s) in the last 12 months." if f["exceedances_12mo"] else "",
                        f"Discharge goes to {water}."]), rng)]
            if f["permit_clause_shown"]:
                tail.append(self.CLAUSE)
            return "\n".join(body + [""] + tail)
        rows = [("Facility", f"{f['facility']} ({f['permit']})"), ("Outfall", "001"), ("Receiving water", water),
                ("Parameter", label), ("Result", val), ("Limit", lim), ("Sampled", fmt_ts(t, rng)),
                ("Lab QC", qc or "none flagged"), ("Other results", ", ".join(others) + f" {unit}" if "average" in basis else None),
                ("Period average", self._period_avg(f)),
                ("Resample", self._resample(f)), ("Cause", show(cause, rng)),
                ("Exceedances (12 mo)", f["exceedances_12mo"]), ("Result received", f"{f['hours_since_result']} h ago")]
        out = "COMPLIANCE LOG -- possible permit exceedance\n" + kv_lines(rows, rng)
        if f["permit_clause_shown"]:
            out += "\n\n" + self.CLAUSE
        return out

    def questions(self):
        return {
            "agency_notice": choice("How should the facility report this result to the agency?", {
                "document_as_invalid_sample": "Record the result as invalid under the lab's quality rules and do "
                                              "not report it as an exceedance.",
                "monthly_report_only": "Report it in the regular monthly monitoring report, with no separate notice.",
                "twenty_four_hour_notice": "Give the agency notice within 24 hours, followed by the written report "
                                           "the permit requires.",
                "immediate_notice_and_downstream_warning": "Notify the agency at once and warn downstream users such "
                                                           "as shellfish harvesters or a water intake.",
            }),
            "operational_response": choice("What should the plant do about the discharge now?", {
                "document_and_monitor": "Record the event and continue routine monitoring.",
                "increased_sampling": "Step up sampling while the cause is investigated.",
                "process_correction": "Correct the process fault that caused it and confirm with a check sample.",
                "curtail_discharge": "Reduce or hold the discharge, for example in storage, until results are back "
                                     "under the limit.",
            }),
            "harm_potential": score("How much harm could this discharge do to the receiving water?", [
                "Negligible: a marginal result on a robust water.",
                "Limited: a modest excess of a conventional parameter.",
                "Considerable: a toxic parameter, a large excess, or a sensitive water.",
                "Serious: a large excess of a toxic parameter into a sensitive water.",
            ]),
            "cause_understanding": score("How well is the cause of the high result understood?", [
                "Unknown: nobody can yet say why it happened.",
                "Suspected: there is a plausible explanation, not yet checked.",
                "Probable: the explanation fits the plant records.",
                "Confirmed: the cause is established and has been dealt with.",
            ]),
            "counts_as_violation": noul(
                "The result counts as a violation of the permit limit.",
                yes="It is a valid result that breaches the limit on the averaging basis the permit uses.",
                no="It fails the lab's quality rules, or it does not breach the limit once averaged as the permit "
                   "requires."),
        }


# =========================================================================== schema 2
UNITS = {
    "boiler_2": ("Boiler 2 (natural gas, 180 MMBtu/h)", "NOx", "lb/MMBtu", 0.10, "3-hour rolling average"),
    "kiln_1": ("Kiln 1 (cement)", "NOx", "lb/ton clinker", 2.8, "30-day rolling average"),
    "turbine_a": ("Combustion turbine A", "NOx", "ppmvd @ 15% O2", 9.0, "1-hour average"),
    "dryer_3": ("Rotary dryer 3", "CO", "ppmvd", 200.0, "1-hour average"),
    "boiler_5": ("Boiler 5 (coal, backup)", "SO2", "lb/MMBtu", 1.2, "3-hour rolling average"),
}
UNIT_SITE = {"boiler_2": "Saltmarsh Paper Mill", "kiln_1": "Stonewick Cement", "turbine_a": "Northholm Power Station",
             "dryer_3": "Brackenmoor Aggregates Dryer Plant", "boiler_5": "Northholm Power Station"}


class CemsEmissionsDeviation(Schema):
    name = "cems_emissions_deviation"
    description = "Continuous emission monitor readings over an air-permit limit around a startup, trip or monitor fault."
    surfaces = {"json": 0.5, "cems_excerpt": 0.25, "shift_log": 0.25}

    EVENTS = {"startup": "cold startup after a maintenance outage", "shutdown": "planned shutdown",
              "malfunction": "burner management fault tripped two burners", "normal": "normal operation at full load",
              "monitor_fault": "analyser sample line iced up"}

    def sample_factors(self, rng):
        reg = regime(rng)
        unit = pick(rng, list(UNITS))
        _, pol, u, limit, basis = UNITS[unit]
        event = pick_w(rng, {"startup": 3, "shutdown": 1, "malfunction": 6, "normal": 2, "monitor_fault": 2})
        hours_over = pick(rng, [1, 2, 3, 5, 9])
        peak_ratio = round(rng.uniform(1.05, 2.5), 2)
        startup_allow_h = pick(rng, [4, 6, 8])
        startup_h = rng.randint(2, 10)
        work_practice = pick_w(rng, {"followed": 5, "partly": 2, "not_documented": 2})
        cal = pick_w(rng, {"passed": 6, "failed_after": 2, "overdue": 1, "passed_late": 1})
        availability = round(rng.uniform(88, 99.5), 1)
        prior = pick(rng, [0, 0, 1, 2, 5])
        maint_history = pick_w(rng, {"clean": 7, "same_fault_before": 2, "deferred_repair": 2})
        if reg == "clear":
            arche = pick(rng, ["normal", "monitor_fault", "clean_malfunction"])
            if arche == "normal":
                event, peak_ratio, hours_over, cal, work_practice = "normal", round(rng.uniform(1.6, 2.4), 2), \
                    pick(rng, [5, 9]), "passed", "followed"
            elif arche == "monitor_fault":
                event, cal, peak_ratio, hours_over = "monitor_fault", "failed_after", round(rng.uniform(2.5, 4.0), 2), 2
            else:   # an abrupt failure with no history behind it
                event, maint_history, cal = "malfunction", "clean", "passed"
        elif reg == "borderline":
            if event == "startup":
                startup_h = near(rng, startup_allow_h, 1, lo=1, integer=True)
            peak_ratio = near(rng, 1.08, 0.07, lo=1.01)
            availability = near(rng, 95.0, 1.2, digits=1)
        else:
            if chance(rng, 0.5):   # a malfunction claim, often on a fault seen before or put off fixing
                event, maint_history = "malfunction", pick(rng, ["same_fault_before", "deferred_repair", "clean"])
            else:                  # a startup inside the allowance, but the work practices are not on paper
                event, startup_h, work_practice = "startup", max(1, startup_allow_h - 1), "not_documented"
            if chance(rng, 0.4):
                cal = pick(rng, ["overdue", "passed_late"])
        if event != "startup":
            startup_h = None
        return {"regime": reg, "unit": unit, "pollutant": pol, "units": u, "limit": limit, "basis": basis,
                "event": event, "hours_over_limit": hours_over, "peak_ratio": peak_ratio,
                "startup_allowance_h": startup_allow_h, "startup_duration_h": startup_h,
                "work_practices": work_practice, "calibration": cal, "quarter_data_availability_pct": availability,
                "deviations_this_year": prior, "maintenance_history": maint_history,
                "facility": UNIT_SITE[unit],
                "permit": ident(rng, "AP", 5, "-"), "operator": person(rng), "env_lead": person(rng),
                "start": rand_ts(rng)}

    def _hours(self, f, rng):
        """An hourly series around the event: values relative to the limit."""
        n_over = f["hours_over_limit"]
        pre = [round(rng.uniform(0.5, 0.9), 2) for _ in range(2)]
        over = [round(rng.uniform(1.0, f["peak_ratio"]), 2) for _ in range(n_over)]
        if over:
            over[len(over) // 2] = f["peak_ratio"]
        post = [round(rng.uniform(0.5, 0.95), 2) for _ in range(2)]
        seq = pre + over + post
        flags = [""] * len(seq)
        if f["event"] == "monitor_fault":
            for i in range(2, 2 + n_over):
                flags[i] = "M" if chance(rng, 0.5) else "OOC"
        return seq, flags

    def _cal(self, f):
        return {"passed": "daily calibration check passed", "failed_after": "calibration check after the event FAILED (drift 11%)",
                "overdue": "quarterly accuracy audit overdue by 12 days", "passed_late": "daily calibration check passed but was run 30 h late"}[f["calibration"]]

    def _history(self, f):
        if f["maintenance_history"] == "clean" and f["event"] == "malfunction":
            return "no earlier occurrence of this fault on record; burner management serviced on schedule at the last outage"
        return {"clean": None, "same_fault_before": "same burner fault logged twice in the last 6 months",
                "deferred_repair": "burner management repair was deferred at the last outage"}[f["maintenance_history"]]

    def _practice(self, f):
        return {"followed": "startup/shutdown work practices followed and logged",
                "partly": "work practices followed, but the fuel-switch step is not logged",
                "not_documented": "no work-practice checklist found for the event"}[f["work_practices"]]

    RULE = ("Permit condition 4.2: emissions during a startup or shutdown lasting up to {h} hours are excluded from "
            "the {basis} limit if the startup/shutdown work practices are followed and logged; all excluded periods and every deviation are "
            "reported in the semiannual report. Malfunctions are deviations unless the facility shows the failure "
            "was sudden, unavoidable and not caused by poor maintenance.")

    def render(self, f, rng, surface):
        name, pol, u, limit, basis = UNITS[f["unit"]]
        seq, flags = self._hours(f, rng)
        t0 = f["start"]
        rule = self.RULE.format(h=f["startup_allowance_h"], basis=basis)
        hist = self._history(f)
        if surface == "json":
            return dumps(prune({
                "source": {"facility": f["facility"], "permit": f["permit"], "unit": name,
                           "limit": f"{limit} {u}, {basis}"},
                "event": {"description": self.EVENTS[f["event"]], "began": fmt_ts(t0, style="iso"),
                          "startup_duration_hours": f["startup_duration_h"],
                          "work_practices": self._practice(f) if f["event"] in ("startup", "shutdown") else None},
                "cems": {"pollutant": pol, "hourly_values": [round(x * limit, 3) for x in seq],
                         "quality_flags": [x or "ok" for x in flags] if any(flags) else None,
                         "calibration": self._cal(f),
                         "quarter_data_availability_pct": f["quarter_data_availability_pct"]},
                "history": {"deviations_reported_this_year": f["deviations_this_year"], "maintenance": hist},
                "permit_condition": rule,
            }))
        if surface == "cems_excerpt":
            rows = [[(t0 + timedelta(hours=i - 2)).strftime("%m-%d %H:00"), f"{x * limit:.3g}", fl or "valid"]
                    for i, (x, fl) in enumerate(zip(seq, flags))]
            head = (f"CEMS HOURLY REPORT  {f['facility']} / {name}\n{pol} limit {limit} {u} ({basis})  "
                    f"permit {f['permit']}")
            tail = kv_lines([("Event", self.EVENTS[f["event"]]),
                             ("Startup duration", f"{f['startup_duration_h']} h" if f["startup_duration_h"] else None),
                             ("Work practices", self._practice(f) if f["event"] in ("startup", "shutdown") else None),
                             ("QA", self._cal(f)), ("Data availability this quarter", f"{f['quarter_data_availability_pct']}%"),
                             ("Deviations this year", f["deviations_this_year"]), ("Maintenance notes", hist)], rng)
            return head + "\n\n" + table(["hour", pol, "status"], rows, rng, "fixed") + "\n\n" + tail + "\n\n" + rule
        s = [f"{fmt_ts(t0, rng)} -- {name}: {self.EVENTS[f['event']]}."]
        over = [x for x in seq if x > 1.0]
        s.append(f"{pol} on the CEMS went over {limit} {u} for {len(over)} hour(s), peak {max(seq) * limit:.3g}.")
        if f["startup_duration_h"]:
            s.append(f"Startup took {f['startup_duration_h']} h.")
        if f["event"] in ("startup", "shutdown"):
            s.append(self._practice(f).capitalize() + ".")
        if any(flags):
            s.append("Analyser showed maintenance/out-of-control flags on some of those hours.")
        s.append(self._cal(f).capitalize() + ".")
        if hist:
            s.append(hist.capitalize() + ".")
        s.append(f"Quarter-to-date data availability {f['quarter_data_availability_pct']}%.")
        return (f"SHIFT LOG -- {f['facility']} (operator {f['operator']})\n" + noisy(join_sentences(s), rng) +
                f"\n\nEnv lead ({f['env_lead']}) to classify. Ref: {rule}")

    def questions(self):
        return {
            "deviation_class": choice("How should the environmental lead classify this event?", {
                "excess_emissions": "An excess-emissions deviation with no exclusion available.",
                "excluded_startup_shutdown": "An excluded startup or shutdown period under the permit condition.",
                "malfunction_deviation": "A malfunction: reported as a deviation, with a defence that the failure was "
                                         "sudden and unavoidable.",
                "monitor_data_problem": "A monitoring-data problem rather than a real emissions excess.",
            }),
            "excess_extent": score("How large were the excess emissions, taking size and duration together?", [
                "None: no valid hour was over the limit.",
                "Marginal: a few valid hours just over.",
                "Material: clearly over the limit for several hours.",
                "Gross: far over the limit or for a long period.",
            ]),
            "monitoring_trust": score("How far can the monitor's data for this period be trusted?", [
                "Not at all: the monitor was out of control.",
                "Doubtful: a quality check failed or was missed.",
                "Mostly: minor quality gaps only.",
                "Fully: every quality check passed on time.",
            ]),
            "malfunction_defence_available": noul(
                "The facility can show the exceedance came from a sudden, unavoidable equipment failure it could not "
                "reasonably have prevented.",
                yes="The failure was abrupt, not a known or recurring problem, and not the result of deferred "
                    "maintenance.",
                no="It was a startup, a normal-operation excess or a monitor fault, or the failure was foreseeable."),
            "monitor_needs_requalifying": noul(
                "The continuous monitor needs repair and a quality-assurance check before its data can be relied on again."),
        }


# =========================================================================== schema 3
# substance: (label, lb per gallon of solution, fraction that counts, reporting threshold lb, oil-like)
SUBSTANCES = {
    "diesel": ("diesel fuel", 7.1, 1.0, None, True),
    "hydraulic_oil": ("hydraulic oil", 7.3, 1.0, None, True),
    "hypochlorite": ("sodium hypochlorite solution 12.5%", 10.0, 0.125, 100, False),
    "sulfuric": ("sulfuric acid 93%", 15.3, 0.93, 1000, False),
    "ammonia_aq": ("aqueous ammonia 19%", 7.6, 0.19, 100, False),
    "glycol": ("ethylene glycol (coolant)", 9.3, 1.0, 5000, False),
}


class SpillReleaseNotification(Schema):
    name = "spill_release_notification"
    description = "A spill or release at a facility: which notifications are due and how bad it could get."
    surfaces = {"json": 0.45, "spill_form": 0.3, "radio_timeline": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        sub = pick(rng, list(SUBSTANCES))
        label, lbgal, frac, thr, oil = SUBSTANCES[sub]
        gallons = round(rng.uniform(3, 400), 0)
        spread = pick(rng, [0.1, 0.2, 0.35])
        reached = pick_w(rng, {"none": 4, "storm_drain": 2, "ditch": 1, "creek": 2})
        sheen = reached in ("ditch", "creek") and chance(rng, 0.7) if oil else False
        containment = pick_w(rng, {"held": 3, "partial": 3, "none": 2})
        discovery_gap_h = pick(rng, [0, 0, 1, 3, 6, None])
        hours_since_discovery = round(rng.uniform(0.2, 4), 1)
        land_window_h = pick(rng, [1, 24])
        people = pick_w(rng, {"none": 6, "one_exposed": 2, "evacuated_building": 1})
        rain = chance(rng, 0.3)
        cleanup = pick_w(rng, {"complete": 2, "ongoing": 4, "not_started": 1})
        if thr is None:
            thr_note = "oil: any sheen on water, or 25 gallons or more on land"
        else:
            thr_note = f"{thr} lb of the pure substance"
        if reg == "clear":
            if chance(rng, 0.5):
                gallons, reached, containment, people, cleanup, sheen = round(rng.uniform(2, 8), 0), "none", "held", \
                    "none", "complete", False
                hours_since_discovery = round(rng.uniform(0.2, 0.6), 1)
            else:
                reached, containment, cleanup = "creek", "none", "ongoing"
                gallons = round(max(gallons, (thr or 25) / (lbgal * frac) * rng.uniform(2, 5) if thr else 200), 0)
                sheen = oil
        elif reg == "borderline":
            if thr:
                gallons = round(near(rng, thr / (lbgal * frac), thr / (lbgal * frac) * 0.15, lo=1), 0)
            else:
                gallons = round(near(rng, 25, 5, lo=1), 0)
                reached = pick(rng, ["none", "storm_drain"])
        else:
            # a small spill that reached water, or a big one that stayed in containment; often an unclear start time
            if chance(rng, 0.5):
                gallons, reached, containment = round(rng.uniform(4, 15), 0), pick(rng, ["storm_drain", "creek"]), "none"
                sheen = oil and chance(rng, 0.6)
            else:
                gallons, reached, containment, cleanup = round(rng.uniform(150, 600), 0), "none", "held", "ongoing"
            discovery_gap_h = pick(rng, [None, 6, 9])
        window_h = 1 if reached != "none" else land_window_h
        if reg == "borderline":
            hours_since_discovery = near(rng, window_h * 0.8, window_h * 0.3, lo=0.1, digits=1)
        est_lo, est_hi = max(1, round(gallons * (1 - spread))), round(gallons * (1 + spread))
        pounds = round(gallons * lbgal * frac, 1)
        return {"regime": reg, "substance": sub, "gallons_best_estimate": gallons, "gallons_range": [est_lo, est_hi],
                "pounds_counted": pounds, "threshold_lb": thr, "threshold_note": thr_note, "oil": oil,
                "reached": reached, "sheen_seen": sheen, "containment": containment,
                "hours_before_discovery": discovery_gap_h, "hours_since_discovery": hours_since_discovery,
                "notification_window_h": window_h, "people": people, "rain_forecast": rain, "cleanup": cleanup,
                "facility": pick(rng, ["Saltmarsh Chemical Works", "Pellbridge Water Reclamation Plant",
                                       "Copperline Metal Finishing", "Saltwick Cold Storage", "Stonewick Quarry"]),
                "source": pick(rng, ["a cracked transfer hose", "an overfilled day tank", "a forklift puncturing a drum",
                                     "a failed pump seal", "a tote valve left open"]),
                "reporter": person(rng), "coordinator": person(rng), "discovered": rand_ts(rng),
                "ref": ident(rng, "SPL", 4, "-")}

    def _reached(self, f):
        base_ = {"none": "stayed on the paved yard / in the building", "storm_drain": "some entered a storm drain",
                 "ditch": "some ran into the roadside ditch", "creek": "some reached the creek behind the site"}[f["reached"]]
        if f["oil"] and f["reached"] in ("ditch", "creek"):
            base_ += "; sheen visible on the water" if f["sheen_seen"] else "; no sheen seen on the water"
        return base_

    def _people(self, f):
        return {"none": None, "one_exposed": "one worker splashed, rinsed at the eyewash and sent to the clinic",
                "evacuated_building": "the adjacent warehouse was evacuated for 40 minutes"}[f["people"]]

    def _start(self, f):
        g = f["hours_before_discovery"]
        return {None: "start time unknown; area last checked at the previous shift change", 0: "seen as it happened"}.get(
            g, f"probably began about {g} h before discovery")

    def _rule(self, f):
        return (f"Contingency plan 3.1: call the state spill line within {f['notification_window_h']} h of discovery for any "
                f"release to water (a storm drain counts as water) or any release at or above the threshold "
                f"({f['threshold_note']}); releases to water "
                f"at or above the threshold also go to the national response line; call 911 first if anyone is hurt "
                f"or the public is at risk.")

    def render(self, f, rng, surface):
        label = SUBSTANCES[f["substance"]][0]
        lo, hi = f["gallons_range"]
        qty = f"about {int(f['gallons_best_estimate'])} gal (range {lo}-{hi} gal)"
        if not f["oil"]:
            qty += f", about {f['pounds_counted']} lb of the pure substance at the best estimate"
        contain = {"held": "held by secondary containment", "partial": "partly held; containment berm overtopped at one corner",
                   "none": "no secondary containment at that location"}[f["containment"]]
        clean = {"complete": "absorbed and drummed, area washed down", "ongoing": "absorbent booms down, vacuum truck on the way",
                 "not_started": "cleanup crew not yet on site"}[f["cleanup"]]
        people = self._people(f)
        if surface == "json":
            return dumps(prune({
                "release": {"ref": f["ref"], "facility": f["facility"], "material": label, "quantity": qty,
                            "source": f["source"], "discovered": fmt_ts(f["discovered"], style="iso"),
                            "onset": self._start(f), "where_it_went": self._reached(f)},
                "response": {"containment": contain, "cleanup": clean, "people": people,
                             "rain_forecast_next_12h": f["rain_forecast"] or None},
                "status": {"hours_since_discovery": f["hours_since_discovery"], "agencies_called": "none yet"},
                "plan_reference": self._rule(f),
            }))
        if surface == "spill_form":
            rows = [("Ref", f["ref"]), ("Facility", f["facility"]), ("Reported by", f["reporter"]),
                    ("Discovered", fmt_ts(f["discovered"], rng)), ("Onset", self._start(f)), ("Material", label),
                    ("Quantity", qty), ("Source", f["source"]), ("Extent", self._reached(f)), ("Containment", contain),
                    ("Cleanup", clean), ("Injuries / evacuation", people or "none"),
                    ("Rain forecast", "yes, within 12 h" if f["rain_forecast"] else "no"),
                    ("Time since discovery", f"{f['hours_since_discovery']} h"), ("Agencies notified", "none yet")]
            return "INTERNAL SPILL REPORT\n" + kv_lines(rows, rng, style="aligned") + "\n\n" + self._rule(f)
        t = f["discovered"]
        msgs = [(0, f["reporter"], f"got a leak at {f['source'].replace('a ', 'the ', 1)}, looks like {label}")]
        msgs.append((rng.randint(3, 10), f["coordinator"], "how much and where is it going?"))
        msgs.append((rng.randint(2, 8), f["reporter"], f"{qty}, {self._reached(f)}. {contain}"))
        if people:
            msgs.append((rng.randint(1, 5), f["reporter"], people))
        msgs.append((rng.randint(5, 20), f["coordinator"], f"ok. cleanup: {clean}. " + self._start(f)))
        mins = int(f["hours_since_discovery"] * 60)
        lines, cur = [], t
        for dm, who, txt in msgs:
            cur += timedelta(minutes=dm)
            lines.append(f"{cur.strftime('%H:%M')} {who.split()[0]}: {noisy(txt, rng, 0.1)}")
        now = t + timedelta(minutes=mins)
        lines.append(f"{now.strftime('%H:%M')} [coordinator] no agency calls made yet"
                     + ("; rain forecast tonight" if f["rain_forecast"] else ""))
        return f"#site-ops radio log -- {f['facility']}\n" + "\n".join(lines) + "\n\n" + self._rule(f)

    def questions(self):
        return {
            "notification_route": choice("Which outside notifications should the facility make?", {
                "emergency_services_first": "Call emergency services first, then the state and national lines.",
                "internal_record_only": "None; record it internally, as it is below every reporting trigger.",
                "state_and_national_lines": "Call both the state spill line and the national response line.",
                "state_spill_line": "Call the state spill line only.",
            }),
            "threshold_reached": noul(
                "The amount released meets or exceeds the reporting threshold for this material.",
                yes="The best estimate of what counts toward the threshold, or the rule for oil on water, is met.",
                no="Even allowing for the estimate's range, the release stays below the threshold and no oil reached water."),
            "offsite_impact": score("How far could this release affect people or the environment beyond the site?", [
                "Not at all: it stayed contained on site.",
                "Slightly: a small amount may have left the site with little effect.",
                "Considerably: material reached a drain or waterway, or a person was exposed.",
                "Severely: a waterway or the public is affected now or will be once it rains.",
            ]),
            "notification_clock": score("How much of the notification window is left?", [
                "Plenty: the clock has barely started, or no call is due.",
                "Running down: well into the window with calls still to make.",
                "Nearly gone: the deadline passes within minutes.",
                "Missed: the window has already closed, counting from when the release could have been found.",
            ]),
            "cleanup_sufficient": noul(
                "The containment and cleanup so far are enough to stop any further release to the environment."),
        }


# =========================================================================== schema 4
ROOT_CAUSES = {
    "operator_error": "operators did not follow the treatment procedure during shift changes",
    "equipment_failure": "a worn pump in the treatment train was losing capacity",
    "design_capacity": "treatment capacity exceeded at current throughput",
    "upstream_change": "the load reaching treatment rose after an upstream change ({up})",
    "unknown": "cause not determined",
}
UPSTREAM = {"aggregate quarry": "a new pit area now drains to the settling ponds",
            "municipal wastewater": "a new industrial user discharging to the sewer"}
ACTIONS = {
    "retrain": "retrain operators on the procedure",
    "sop": "revise the standard operating procedure",
    "replace_part": "replace the failed component and add it to the PM schedule",
    "alarm": "add a high-level alarm and interlock",
    "capital": "design and build additional treatment capacity",
    "sampling": "increase sampling to weekly for three months",
}


class CorrectiveActionPlanReview(Schema):
    name = "corrective_action_plan_review"
    description = "A root-cause and corrective-action plan for repeat exceedances, reviewed before it goes to the agency."
    surfaces = {"json": 0.5, "capa_document": 0.25, "reviewer_markup": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        fac = pick(rng, list(FACILITIES))
        param = pick(rng, FACILITIES[fac][3])
        repeats = pick(rng, [2, 3, 4, 6])
        method = pick_w(rng, {"five_whys": 3, "fishbone": 2, "fault_tree": 1, "none": 2})
        cause = pick(rng, list(ROOT_CAUSES), [2, 3, 2, 2, 1])
        acts = rng.sample(list(ACTIONS), pick(rng, [1, 2, 3]))
        owners = chance(rng, 0.6)
        dates = chance(rng, 0.6)
        metric = chance(rng, 0.5)
        capital_budget = pick_w(rng, {"approved": 2, "requested": 2, "none": 2}) if "capital" in acts else None
        prior_enf = pick_w(rng, {"none": 4, "warning_letter": 3, "nov": 2, "consent_order": 1})
        author = pick(rng, ["plant engineer", "shift supervisor", "outside consultant"])
        if reg == "clear":
            if chance(rng, 0.5):
                method, cause, acts = "fault_tree", "equipment_failure", ["replace_part", "alarm", "sampling"]
                owners = dates = metric = True
                repeats, prior_enf = 2, "none"
            else:
                method, cause, acts, owners, dates, metric = "none", "operator_error", ["retrain"], False, False, False
                repeats, prior_enf = pick(rng, [4, 6]), pick(rng, ["nov", "consent_order"])
        elif reg == "borderline":
            method = pick(rng, ["five_whys", "fishbone"])
            owners, dates, metric = True, True, chance(rng, 0.5)
            repeats = 3
        else:
            # a capacity problem answered with retraining, or a thorough plan whose big fix has no money behind it
            if chance(rng, 0.5):
                cause, acts, method = "design_capacity", ["retrain", "sop"], "five_whys"
                owners = dates = True
            else:
                cause, acts, method = "upstream_change", ["capital", "sampling"], "fault_tree"
                capital_budget, owners, dates, metric = "none", True, True, True
        if "capital" in acts and capital_budget is None:
            capital_budget = pick(rng, ["approved", "requested", "none"])
        if "capital" not in acts:
            capital_budget = None
        return {"regime": reg, "facility": fac, "parameter": param, "exceedances_12mo": repeats,
                "rca_method": method, "stated_root_cause": cause, "actions": acts, "actions_have_owners": owners,
                "actions_have_dates": dates, "effectiveness_metric": metric, "capital_budget": capital_budget,
                "prior_enforcement": prior_enf, "author_role": author, "author": person(rng), "reviewer": person(rng),
                "doc": ident(rng, "CAP", 4, "-"), "date": rand_ts(rng).date()}

    def _action_rows(self, f, rng):
        rows = []
        d = f["date"]
        for k in f["actions"]:
            owner = person(rng) if f["actions_have_owners"] else None
            days = rng.randint(270, 540) if k == "capital" else rng.randint(14, 120)
            due = fmt_date(d + timedelta(days=days), style="iso") if f["actions_have_dates"] else None
            rows.append((ACTIONS[k], owner, due))
        return rows

    def _cause(self, f):
        sector = FACILITIES[f["facility"]][0]
        return ROOT_CAUSES[f["stated_root_cause"]].format(up=UPSTREAM.get(sector, "a new product line"))

    def _enf(self, f):
        return {"none": "no prior enforcement", "warning_letter": "warning letter received last year",
                "nov": "notice of violation issued for the same parameter 8 months ago",
                "consent_order": "operating under a consent order with stipulated penalties"}[f["prior_enforcement"]]

    def render(self, f, rng, surface):
        label = PARAMS[f["parameter"]][0]
        acts = self._action_rows(f, rng)
        method = {"five_whys": "5-whys", "fishbone": "fishbone diagram", "fault_tree": "fault-tree analysis with data review",
                  "none": "not stated"}[f["rca_method"]]
        metric = "success = no exceedance and monthly average below 70% of the limit for 6 months" if f["effectiveness_metric"] else None
        budget = {"approved": "capital budget approved", "requested": "capital request submitted, not yet approved",
                  "none": "no budget identified", None: None}[f["capital_budget"]]
        if surface == "json":
            return dumps(prune({
                "plan": {"id": f["doc"], "facility": f["facility"], "prepared_by": f"{f['author']} ({f['author_role']})",
                         "date": f["date"].isoformat()},
                "problem": {"parameter": label, "exceedances_last_12_months": f["exceedances_12mo"],
                            "enforcement_history": self._enf(f)},
                "analysis": {"method": method, "root_cause": self._cause(f)},
                "actions": [prune({"action": a, "owner": o, "due": d}) for a, o, d in acts],
                "verification": metric, "funding": budget,
            }))
        if surface == "capa_document":
            tbl = table(["action", "owner", "due"], [[a, o or "", d or ""] for a, o, d in acts], rng)
            return "\n".join([
                f"CORRECTIVE ACTION PLAN {f['doc']} -- {f['facility']}",
                kv_lines([("Prepared by", f"{f['author']} ({f['author_role']})"), ("Date", fmt_date(f["date"], rng)),
                          ("Parameter", label), ("Exceedances (12 months)", f["exceedances_12mo"]),
                          ("Enforcement history", self._enf(f))], rng), "",
                f"1. Root cause analysis ({method}): {self._cause(f)}.", "",
                "2. Corrective actions:", tbl, "",
                f"3. Verification: {metric or 'to be determined'}.", (f"4. Funding: {budget}." if budget else "")]).strip()
        comments = []
        if f["rca_method"] == "none":
            comments.append("no method shown for how the cause was found")
        if not f["actions_have_owners"]:
            comments.append("actions have no named owners")
        if not f["actions_have_dates"]:
            comments.append("no due dates")
        if not f["effectiveness_metric"]:
            comments.append("how will we know it worked?")
        if f["capital_budget"] in ("none", "requested"):
            comments.append("capital fix is not funded yet")
        comments = comments or [pick(rng, ["reads fine to me", "looks complete"])]
        return "\n".join([
            f"Subject: Review of {f['doc']} ({f['facility']}) before submittal",
            f"From: {f['reviewer']}", f"To: {f['author']}", "",
            noisy(join_sentences([
                f"I went through the plan for the {label} exceedances ({f['exceedances_12mo']} in the last year; {self._enf(f)}).",
                f"Root cause as written: {self._cause(f)} (method: {method}).",
                "Actions: " + "; ".join(a + (f" ({o}, {d})" if o and d else "") for a, o, d in acts) + ".",
                f"Funding: {budget}." if budget else ""]), rng),
            "", "My comments so far: " + "; ".join(comments) + ".", "", f['reviewer'].split()[0]])

    def questions(self):
        return {
            "plan_verdict": choice("What should the reviewer do with this corrective-action plan?", {
                "accept_as_written": "Accept the plan and send it as written: a technical root cause, actions that "
                                     "address it, and owners, dates, a success measure and any funding in place.",
                "accept_with_added_actions": "Accept it once specific gaps are filled: missing owners, due dates, a "
                                             "success measure, or approved funding for the capital fix.",
                "return_for_deeper_analysis": "Send it back: no method or cause is shown, the blame rests on operators "
                                              "alone, or the actions do not address the stated cause.",
                "escalate_to_management_and_counsel": "Escalate to senior management and legal counsel first: the "
                                                      "facility is already under a notice of violation or consent "
                                                      "order, the exceedances keep repeating, and the plan would not "
                                                      "stop them.",
            }),
            "enforcement_outlook": choice("What response from the agency should the facility plan for?", {
                "consent_order_or_penalty": "A consent order or a financial penalty.",
                "no_action_expected": "No enforcement; the agency will note the plan and move on.",
                "notice_of_violation": "A formal notice of violation requiring a response.",
                "warning_letter": "An informal warning letter.",
            }),
            "root_cause_depth": score("How deep does the root-cause analysis go?", [
                "Surface: no method shown, no cause found, or the cause given is simply that operators got it wrong.",
                "Shallow: a technical cause reached by a quick 5-whys, with no data shown.",
                "Reasonable: a technical cause from a fishbone review that sets out the possible causes.",
                "Thorough: a technical cause from a fault-tree analysis checked against the plant data.",
            ]),
            "recurrence_likelihood": score("How likely is the exceedance to recur if this plan is carried out as "
                                           "written?", [
                "Unlikely: the actions remove the cause.",
                "Possible: the actions help, but gaps remain.",
                "Likely: the actions do not match the cause.",
                "Near certain: nothing in the plan changes the conditions that caused it.",
            ]),
            "actions_verifiable": noul(
                "Every corrective action has an owner, a due date and a measurable test of whether it worked.",
                yes="Owners, dates and a success measure are all present for the actions.",
                no="At least one action lacks an owner, a date, or any way to check its effect."),
        }


# =========================================================================== schema 5
class MonitoringReportPrecheck(Schema):
    name = "monitoring_report_precheck"
    description = "A monthly discharge monitoring report checked before the responsible official certifies it."
    surfaces = {"json": 0.5, "dmr_table": 0.25, "compliance_memo": 0.25}

    FREQ = {"ammonia": 8, "tss": 8, "ecoli": 4, "bod": 8, "chlorine": 22, "zinc": 2, "ph": 22}
    MISSED = {"autosampler": "autosampler failed on one day", "ice": "sampling point frozen for two days",
              "lab_lost": "lab lost one bottle", "staffing": "no operator available on a holiday weekend"}

    def sample_factors(self, rng):
        reg = regime(rng)
        fac = pick(rng, list(FACILITIES))
        params = FACILITIES[fac][3]
        rows = {}
        for p in params:
            limit = PARAMS[p][2]
            req = self.FREQ[p]
            rows[p] = {"required": req, "taken": req, "avg_ratio": round(rng.uniform(0.25, 0.75), 2),
                       "max_ratio": round(rng.uniform(0.5, 0.95), 2), "nondetects": 0}
        focus = pick(rng, params)
        missed = None
        over = None
        if reg == "clear":
            if chance(rng, 0.5):
                pass
            else:
                over = focus
                rows[focus]["max_ratio"] = round(rng.uniform(1.5, 2.5), 2)
                rows[focus]["avg_ratio"] = round(rng.uniform(1.05, 1.4), 2)
        elif reg == "borderline":
            rows[focus]["avg_ratio"] = near(rng, 0.97, 0.04, hi=1.04)
            rows[focus]["max_ratio"] = near(rng, 1.0, 0.06)
            if chance(rng, 0.5):
                missed = pick(rng, list(self.MISSED))
                mp = pick(rng, params)
                rows[mp]["taken"] = rows[mp]["required"] - 1
        else:
            # a missed sample with a good excuse, or a clean-looking table resting on non-detects and a new lab
            missed = pick(rng, list(self.MISSED))
            mp = pick(rng, params)
            rows[mp]["taken"] = rows[mp]["required"] - pick(rng, [1, 2])
            if chance(rng, 0.5):
                rows[focus]["nondetects"] = pick(rng, [2, 3])
                rows[focus]["avg_ratio"] = near(rng, 0.92, 0.05)
        for r in rows.values():            # the month's maximum can never sit below its average
            r["max_ratio"] = max(r["max_ratio"], round(r["avg_ratio"] * rng.uniform(1.04, 1.3), 2))
        if over is None and any(r["max_ratio"] > 1.0 for r in rows.values()):
            over = next(p for p, r in rows.items() if r["max_ratio"] > 1.0)
        short = next((p for p, r in rows.items() if r["taken"] < r["required"]), None)
        days_to_due = pick(rng, [1, 3, 6, 12, 20])
        official = pick_w(rng, {"available": 5, "on_leave_delegate_on_file": 2, "on_leave_no_delegate": 1})
        new_lab = chance(rng, 0.2)
        trend = pick_w(rng, {"stable": 4, "rising": 3, "falling": 2})
        return {"regime": reg, "facility": fac, "permit": ident(rng, "WD", 5, "-"), "rows": rows, "focus": focus,
                "missed_reason": missed, "short_parameter": short, "parameter_over": over, "days_to_due": days_to_due,
                "official": official, "new_lab_this_month": new_lab, "trend_3mo": trend,
                "preparer": person(rng), "official_name": person(rng), "month": rand_ts(rng).date().replace(day=1)}

    def _table_rows(self, f):
        out = []
        for p, r in f["rows"].items():
            label, unit, limit, basis, _ = PARAMS[p]
            avg = _fmt(limit * r["avg_ratio"], p) if p != "ph" else f"min {6.1 + r['avg_ratio']:.1f}"
            mx = _fmt(limit * r["max_ratio"], p) if p != "ph" else f"{7.0 + 2.0 * r['max_ratio']:.1f}"
            lim = _fmt(limit, p)
            nd = f" ({r['nondetects']} ND counted as zero)" if r["nondetects"] else ""
            out.append([label, unit, f"{r['taken']}/{r['required']}", avg + nd, mx, f"{lim} ({basis})"])
        return out

    def _official(self, f):
        return {"available": "responsible official available to sign",
                "on_leave_delegate_on_file": "responsible official on leave; signature delegation letter on file",
                "on_leave_no_delegate": "responsible official on leave; no delegation on file"}[f["official"]]

    def render(self, f, rng, surface):
        rows = self._table_rows(f)
        month = f["month"].strftime("%B %Y")
        miss = self.MISSED.get(f["missed_reason"]) if f["missed_reason"] else None
        if miss and f["short_parameter"]:
            r = f["rows"][f["short_parameter"]]
            miss += (f" ({PARAMS[f['short_parameter']][0]}: {r['taken']} of the {r['required']} samples the permit "
                     f"requires)")
        trend = {"stable": "results similar to the last three months", "rising": "the tightest parameter has risen each of the last three months",
                 "falling": "results improving over the last three months"}[f["trend_3mo"]]
        lab = "new contract lab this month; its methods not yet audited" if f["new_lab_this_month"] else None
        if surface == "json":
            return dumps(prune({
                "report": {"facility": f["facility"], "permit": f["permit"], "period": month, "prepared_by": f["preparer"],
                           "due_in_days": f["days_to_due"]},
                "results": [{"parameter": r[0], "unit": r[1], "samples_taken_of_required": r[2], "average": r[3],
                             "maximum": r[4], "limit": r[5]} for r in rows],
                "notes": {"missed_sampling": miss, "trend": trend, "lab": lab},
                "signatory": self._official(f),
            }))
        if surface == "dmr_table":
            head = f"DRAFT DMR -- {f['facility']} ({f['permit']}), {month}\nPrepared by {f['preparer']}; due in {f['days_to_due']} days"
            tbl = table(["parameter", "unit", "n taken/req", "average", "max", "limit"], rows, rng)
            notes = bullets([x for x in (miss, trend, lab, self._official(f)) if x], rng)
            return head + "\n\n" + tbl + "\n\nNotes:\n" + notes
        s = [f"{f['official_name'].split()[0]}, the {month} monitoring report for {f['facility']} is drafted and due "
             f"in {f['days_to_due']} days."]
        for r in rows:
            s.append(f"{r[0]}: {r[2]} samples, average {r[3]}, max {r[4]} {r[1]} against {r[5]}.")
        if miss:
            s.append(f"Sampling gap: {miss}.")
        s.append(trend.capitalize() + ".")
        if lab:
            s.append(lab.capitalize() + ".")
        s.append(self._official(f).capitalize() + ".")
        return f"MEMO -- monthly report pre-check\nFrom: {f['preparer']}\n\n" + noisy(join_sentences(s), rng, 0.15)

    def questions(self):
        return {
            "report_action": choice("What should happen to the draft report before the deadline?", {
                "certify_and_submit": "Certify and submit it as prepared.",
                "correct_before_signing": "Correct errors in how results were calculated or reported, then certify.",
                "submit_with_explanation": "Submit it with a cover letter explaining missed samples or exceedances.",
                "request_deadline_extension": "Ask the agency for more time because the data or the signature "
                                              "cannot be finalised yet.",
            }),
            "sampling_frequency_met": noul(
                "Every parameter was sampled at least as often as the permit requires for the month.",
                yes="For each parameter the number of samples taken equals or exceeds the number required.",
                no="At least one parameter has fewer samples than the permit requires, whatever the reason."),
            "limit_exceeded_in_period": noul(
                "The facility exceeded a permit limit during this reporting period.",
                yes="At least one parameter breaks its limit on the permit's own basis: a daily or instantaneous "
                    "maximum above its limit, or a monthly or weekly average above its limit.",
                no="No parameter breaks its limit on its own basis; a single high reading under an averaged limit "
                   "is not a breach while the average stays under."),
            "trend_concern": score("How worrying are the results relative to the permit limits?", [
                "Not at all: everything sits well below the limits.",
                "Mildly: one parameter is creeping upward.",
                "Considerably: a parameter is close to its limit or depends on how non-detects are treated.",
                "Seriously: a limit is already breached or will be soon on the current trend.",
            ]),
            "submission_deadline_pressure": score("How tight is the time to get this report certified and filed?", [
                "Relaxed: weeks remain and the signatory is available.",
                "Comfortable: enough time, with small things to sort out.",
                "Tight: a few days, with open issues or signature trouble.",
                "Critical: due within a day or two with no one able to sign, or open data problems.",
            ]),
        }


WORKFLOW = Workflow(
    name="environmental_compliance_reports",
    title="Environmental compliance reports",
    description="What a fictional facility must report after an exceedance, deviation or spill, to whom, and how "
                "seriously to treat it.",
    domain="environmental health and safety compliance",
    schemas=[DischargeExceedance(), CemsEmissionsDeviation(), SpillReleaseNotification(),
             CorrectiveActionPlanReview(), MonitoringReportPrecheck()],
    notes="Invented facilities, permits, agency and reporting clauses; monitoring data and permit terms, not "
          "security alerts, support threads, vendor bills or agent runs.",
)
