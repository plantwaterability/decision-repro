"""Lab safety incident review: radiation, laser, cryogen and high-voltage incidents in research laboratories,
judged against the institution's WRITTEN safety rules.

Each case is an incident file a safety officer or committee would actually see: a dosimetry investigation, a laser
eye-exposure report, an oxygen-monitor log after a cryogen release, a high-voltage lockout breach, a radioactive
spill survey. The rule that applies is quoted in every state, and its numbers (investigation and reporting levels,
required optical density, alarm and evacuation oxygen levels, the stored-energy limit, removable-contamination
limits) vary between institutions and rule revisions, so every answer needs the facts compared with the rule
actually quoted.

Every institution, laboratory, rule number, person, threshold and reading is fictional. No real regulator is
named. Numbers are kept physically plausible (stored energy 0.5*C*V^2, cryogen expansion ratios of roughly 700:1
and 750:1, isotope half-lives, optical densities from the hazard evaluation).

Each schema draws a `regime` first. Borderline cases sit near the written threshold (a badge just over the
reporting level, eyewear one OD short, oxygen at 18.3% against an 18% evacuation level, stored energy a few
joules either side of the limit). Conflicting cases pair signals that disagree: a high badge with a corroborated
"left on the bench" story, a supervisor's "no harm done" beside a taped-over interlock, a clean wipe survey
counted on an instrument that cannot see the isotope. Each schema's `intended(f)` gives the careful reviewer's
answer from the factors; it is used only to audit answer spread and is never rendered.
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
    near,
    noisy,
    noul,
    org,
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

# =========================================================================== shared pools
SITE_POOLS = {
    "rad": (["Halvorsen Nuclear Physics Institute", "Whinmoor Radiochemistry Centre", "Ashby Fold Isotope Facility",
             "Greyholme Institute for Biophysics", "Thrapston Imaging Research Unit", "Sallowmere Neutron Science Centre",
             "Linmouth Ion Beam Centre", "Mirefield Accelerator Centre", "Calder Ridge Laboratory"],
            ["School of Chemistry", "Department of Biochemistry", "Institute of Nuclear Science",
             "Centre for Imaging Sciences", "Department of Molecular Biology", "Department of Medical Physics",
             "Radiopharmacy Unit", "Department of Physics", "Department of Pharmacology"]),
    "laser": (["Vantor Institute for Photonics", "Kelmscott Institute of Optics", "Wendle Laser Science Centre",
               "Ostrander Applied Physics Laboratory", "Fennick Institute of Quantum Matter", "Sandhaven Light Source",
               "Pellow Plasma Laboratory", "Calder Ridge Laboratory"],
              ["Department of Physics", "Department of Applied Optics", "School of Chemistry",
               "Department of Electrical Engineering", "Department of Materials", "Department of Astronomy and "
               "Instrumentation", "Ultrafast Spectroscopy Group"]),
    "cryo": (["Tarrant Low Temperature Facility", "Corrie Fields Cryogenics Laboratory", "Rookhope Magnet Laboratory",
              "Fennick Institute of Quantum Matter", "Elsby Underground Laboratory", "Sallowmere Neutron Science Centre",
              "Hollins Gate Instrumentation Laboratory", "Calder Ridge Laboratory"],
             ["Department of Physics", "School of Chemistry", "Department of Materials",
              "Department of Astronomy and Instrumentation", "School of Engineering", "Department of Biochemistry",
              "Structural Biology Unit"]),
    "hv": (["Mirefield Accelerator Centre", "Pellow Plasma Laboratory", "Linmouth Ion Beam Centre",
            "Brannock Materials Research Centre", "Ostrander Applied Physics Laboratory",
            "Hollins Gate Instrumentation Laboratory", "Sandhaven Light Source", "Wendle Laser Science Centre",
            "Calder Ridge Laboratory"],
           ["Department of Physics", "Department of Electrical Engineering", "School of Engineering",
            "Department of Applied Optics", "Accelerator Science Group", "Plasma Physics Group"]),
}
RAD_BOOKS = ["Radiation Safety Manual", "Local Rules for Radiation Work", "Radioisotope Users Handbook",
             "Radiation Protection Code", "Ionising Radiation Local Rules"]
LASER_BOOKS = ["Laser Safety Programme", "Laser Users Code", "Optical Radiation Safety Rules", "Laser Hazard Standard"]
CRYO_BOOKS = ["Cryogen Handling Standard", "Oxygen Deficiency Hazard Rules", "Low Temperature Safety Code",
              "Cryogenic Safety Procedure"]
HV_BOOKS = ["Electrical Safety Rules", "High Voltage Work Standard", "Electrical Work Code", "Hazardous Energy Procedure"]


def _site(rng, domain):
    insts, depts = SITE_POOLS[domain]
    if chance(rng, 0.5):
        return pick(rng, insts)
    return f"{org(rng, 'school')}, {pick(rng, depts)}"


def _room(rng):
    return (f"{pick(rng, list('BCDEGHKLMNPRSTW'))}{rng.randint(1, 48)}-"
            f"{pick(rng, ['0', '1', '2', '3', 'B'])}{rng.randint(1, 60):02d}")


def _rule(rng, books):
    book = pick(rng, books)
    abbr = "".join(w[0] for w in book.split() if w[0].isupper())
    return f"{book} {abbr}-{rng.randint(2, 9)}.{rng.randint(1, 14)} (rev. {rng.randint(2019, 2026)})"


def _yn(v):
    return "yes" if v else "no"


def _fmt(x, dec):
    return f"{x:.{dec}f}"


# =========================================================================== schema 1
WB_TASKS = [
    ("dispensing F-18 tracer doses for small-animal PET", "F-18", "2-6 GBq per batch"),
    ("eluting the Tc-99m generator and preparing labelling kits", "Tc-99m", "up to 12 GBq per elution"),
    ("loading the Cs-137 source into the detector calibration stand", "Cs-137", "a 370 MBq sealed source"),
    ("Ga-68 peptide labelling at the shielded synthesis unit", "Ga-68", "0.8-1.5 GBq per run"),
    ("I-125 protein iodination in the fume hood", "I-125", "37-74 MBq per labelling"),
    ("handling activated foils after neutron irradiations", "activation products", "a few MBq per foil set"),
    ("swapping targets at the cyclotron target station", "activation products", "residual activity on holders"),
    ("loading samples into the Co-60 gamma irradiator", "Co-60", "a self-shielded irradiator"),
    ("preparing Lu-177 labelled compounds", "Lu-177", "1-4 GBq per synthesis"),
    ("work on the Am-Be neutron moderator test rig", "Am-Be", "a 37 GBq sealed source"),
]
EXT_TASKS = [
    ("P-32 labelling of DNA probes", "P-32", "a 185 MBq stock vial"),
    ("drawing F-18 doses into syringes", "F-18", "1-4 GBq per session"),
    ("Tc-99m kit preparation", "Tc-99m", "2-10 GBq per session"),
    ("Y-90 aliquoting for a microsphere study", "Y-90", "a 3 GBq stock vial"),
    ("Ga-68 peptide labelling and QC sampling", "Ga-68", "about 1 GBq per run"),
    ("sorting and loading I-125 seeds", "I-125", "batches of 15-30 MBq"),
    ("Lu-177 dispensing", "Lu-177", "2-5 GBq per batch"),
    ("P-32 orthophosphate metabolic labelling", "P-32", "a 370 MBq shipment"),
]
ROLES = ["PhD student", "postdoctoral researcher", "research technician", "radiochemist", "visiting researcher",
         "staff scientist", "lab manager", "master's student", "radiopharmacy technician", "beamline scientist"]
MINOR_LAPSES = [
    ("survey log has no entries for several of the sessions", "any"),
    ("no bench paper or drip tray under the work area", "any"),
    ("end-of-session hand and foot survey not recorded", "any"),
    ("shielding block on the bench but pushed aside during dispensing", "any"),
    ("ring dosimeter worn on the non-dominant hand, away from the vial", "ext"),
    ("waste container left unshielded beside the work position", "any"),
    ("work started before the survey meter's daily source check", "any"),
    ("ring dosimeter worn over the glove instead of under it", "ext"),
]
MAJOR_LAPSES = [
    ("stock vial handled without tongs or a vial shield", "any"),
    ("whole stock activity used instead of the aliquot set in the protocol", "any"),
    ("dispensing done at the open bench instead of behind the L-block", "any"),
    ("activity handled per session above the user's authorised limit", "any"),
    ("sealed source carried by hand between rooms without its transport pot", "wb"),
    ("syringe shield not used for any of the draws", "any"),
]
STORY_PLACES = [
    "on the lab coat hanging next to the {iso} waste drum",
    "clipped to the fume-hood sash above the {iso} stock",
    "on the shelf beside the shielded source safe",
    "on the bench next to the lead castle",
    "in the pocket of a coat left in the source store",
    "on the hook by the hot-cell door",
    "on top of the {iso} waste shielding",
]
NO_STORY = [
    "Nothing different this period; the usual sessions of {task}.",
    "Workload was heavier than usual, {n} sessions instead of the normal handful.",
    "I can't think of anything unusual. I wear the dosimeter for every session.",
    "Same routine as every period as far as I remember.",
    "There were a couple of long days in the {task} campaign, otherwise normal.",
    "I don't know why it is high. I followed the protocol each time.",
]


class DosimeterInvestigation(Schema):
    name = "dosimeter_investigation"
    description = ("A personal or extremity dosimeter reading above the institution's investigation level: real dose "
                   "or artefact, whether it is reportable, and how the investigation closes.")
    surfaces = {"json": 0.5, "dosimetry_report": 0.25, "rso_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        badge = pick_w(rng, {"whole_body": 3, "extremity_ring": 2})
        period = pick_w(rng, {"monthly": 3, "quarterly": 2})
        if badge == "whole_body":
            inv = pick(rng, [0.3, 0.4, 0.5]) if period == "monthly" else pick(rng, [1.0, 1.25, 1.5])
            report = round(inv * pick(rng, [4, 5, 6, 8]), 2)
            limit = pick(rng, [20, 20, 20, 50])
            dec = 2
            task, iso, act = pick(rng, WB_TASKS)
        else:
            inv = pick(rng, [4.0, 5.0, 6.5]) if period == "monthly" else pick(rng, [12.0, 15.0])
            report = round(inv * pick(rng, [4, 5, 8]), 1)
            limit = 500
            dec = 1
            task, iso, act = pick(rng, EXT_TASKS)
        if reg == "clear":
            kind = pick_w(rng, {"benign_real": 1, "serious_real": 1.3, "artefact": 1.5})
            real = kind != "artefact"
        elif reg == "borderline":
            kind, real = "border", chance(rng, 0.7)
        else:
            kind, real = "conflict", chance(rng, 0.4)
        if kind == "benign_real":
            pos = "just_over_inv"
        elif kind == "serious_real":
            pos = "over_report"
        elif kind == "artefact":
            pos = pick(rng, ["mid", "over_report", "near_report"])
        elif kind == "border":
            pos = pick_w(rng, {"near_report": 6, "over_report": 1, "just_over_inv": 1, "mid": 1})
        else:
            pos = pick_w(rng, {"near_report": 2, "over_report": 4, "mid": 1, "just_over_inv": 1})
        reading = {"just_over_inv": inv * rng.uniform(1.03, 1.6),
                   "mid": rng.uniform(inv * 1.6, report * 0.85),
                   "near_report": near(rng, report * 1.08, report * 0.14),
                   "over_report": report * rng.uniform(1.15, 2.6)}[pos]
        reading = round(max(inv * 1.02, reading), dec)

        mech = None if real else pick_w(rng, {"stored_near_source": 6, "processing_fault": 2, "unexplained": 2})
        if kind == "artefact":
            mech = pick(rng, ["stored_near_source", "stored_near_source", "processing_fault"])
        story, corroborated, contradicted = "none", False, False
        if mech == "stored_near_source":
            story = "stored"
            corroborated = True if kind == "artefact" else chance(rng, 0.65 if reg == "borderline" else 0.4)
        elif real and reg == "conflicting" and chance(rng, 0.5):
            story = "stored"   # an account the evidence does not back
            contradicted = chance(rng, 0.6)
        # task-based estimate (survey rate x logged hours)
        if real:
            ratio = rng.uniform(0.55, 1.5)
            if reg == "conflicting" and chance(rng, 0.35):
                ratio = rng.uniform(0.27, 0.44)      # the log under-records the work
        else:
            ratio = rng.uniform(0.04, 0.22)
            if reg == "conflicting" and chance(rng, 0.4):
                ratio = rng.uniform(0.28, 0.44)
        est = round(reading * ratio, dec)
        if real and story == "none" and reg != "clear" and chance(rng, 0.2):
            est = None                               # survey log too incomplete for an estimate
        hours = rng.randint(6, 48) if badge == "whole_body" else rng.randint(3, 20)
        cw = None
        if chance(rng, 0.65):
            cw = round(reading * (rng.uniform(0.35, 0.95) if real else rng.uniform(0.02, 0.12)), dec)
        control = round(rng.uniform(0.0, 0.03 if badge == "whole_body" else 0.2), dec)
        service_note = None
        if mech == "processing_fault":
            if chance(rng, 0.5):
                control = round(reading * rng.uniform(0.5, 0.95), dec)
                service_note = "control badge in the same shipment also read high"
            else:
                service_note = pick(rng, ["irregular glow curve on the reading element; result flagged as questionable",
                                          "reader fault logged during this batch; result marked provisional",
                                          "chip found cracked on unpacking; reading may be unreliable"])
        if real and pos in ("near_report", "over_report") and chance(rng, 0.45):
            prev = [round(inv * x, dec) for x in (rng.uniform(0.2, 0.4), rng.uniform(0.45, 0.65),
                                                   rng.uniform(0.7, 0.98))]
        else:
            prev = [round(inv * rng.uniform(0.02, 0.45), dec) for _ in range(3)]
        rising = prev[0] < prev[1] < prev[2] and prev[2] > 0.5 * inv
        if reg != "clear" and chance(rng, 0.15):
            ytd = round(max(0.0, limit - reading * rng.uniform(0.6, 1.4)), dec)
        else:
            ytd = round(limit * rng.uniform(0.01, 0.3), dec)
        okey = "wb" if badge == "whole_body" else "ext"
        minors = [t for t, a in MINOR_LAPSES if a in ("any", okey)]
        majors = [t for t, a in MAJOR_LAPSES if a in ("any", okey)]
        if kind == "benign_real":
            n_min, n_maj = 0, 0
        elif kind == "serious_real":
            n_min, n_maj = pick(rng, [0, 1]), pick(rng, [1, 2])
        elif kind == "artefact":
            n_min, n_maj = 0, 0
        elif reg == "borderline":
            n_min, n_maj = pick(rng, [0, 0, 1, 2]), pick(rng, [0, 0, 0, 1])
        else:
            n_min, n_maj = pick(rng, [0, 1, 1, 2]), pick(rng, [0, 0, 1])
        if not real:
            n_maj = 0                                # a shielding lapse would explain a real dose instead
        lapses_minor = rng.sample(minors, n_min)
        lapses_major = rng.sample(majors, n_maj)
        if story == "stored":
            lapses_minor.append("dosimeter left in the lab between sessions instead of on the dosimeter rack")
        prior_warning = chance(rng, 0.2)
        place = pick(rng, STORY_PLACES).format(iso=iso)
        return {
            "regime": reg, "kind": kind, "true_real": real, "artefact_mechanism": mech,
            "badge_type": badge, "period": period, "investigation_level_msv": inv, "report_level_msv": report,
            "annual_limit_msv": limit, "decimals": dec, "task": task, "isotope": iso, "activity": act,
            "reading_msv": reading, "task_estimate_msv": est, "logged_hours": hours, "coworker_msv": cw,
            "control_msv": control, "service_note": service_note, "previous_msv": prev, "rising_trend": rising,
            "ytd_before_msv": ytd, "story": story, "corroborated": corroborated, "contradicted": contradicted,
            "story_place": place,
            "lapses_minor": lapses_minor, "lapses_major": lapses_major, "prior_warning": prior_warning,
            "role": pick(rng, ROLES), "worker": person(rng), "rso": person(rng), "witness": person(rng),
            "site": _site(rng, "rad"), "rule_ref": _rule(rng, RAD_BOOKS), "case_id": ident(rng, "DI", 4),
            "badge_id": ident(rng, "BX", 6, ""), "date": rand_date(rng), "sessions": rng.randint(4, 22),
            "hours_away": pick(rng, [40, 52, 64, 70, 88]),
        }

    # ------------------------------------------------------------------ evidence -> judgement
    def _assess(self, f):
        s = 0
        rd = f["reading_msv"]
        if f["task_estimate_msv"] is not None:
            r = f["task_estimate_msv"] / rd
            s += 2 if r >= 0.45 else (0 if r >= 0.25 else -2)
        if f["story"] == "stored":
            s -= 3 if f["corroborated"] else (0 if f["contradicted"] else 1)
        if f["service_note"]:
            s -= 3
        if f["coworker_msv"] is not None:
            c = f["coworker_msv"] / rd
            s += 1 if c >= 0.35 else (-1 if c < 0.15 else 0)
        if f["rising_trend"]:
            s += 1
        real = s >= 0
        if real:
            dor = rd
        elif f["task_estimate_msv"] is not None:
            dor = f["task_estimate_msv"]
        elif f["coworker_msv"] is not None:
            dor = f["coworker_msv"]
        else:
            dor = 0.5 * f["investigation_level_msv"]
        return real, dor

    def intended(self, f):
        real, dor = self._assess(f)
        inv, rep, lim, ytd = (f["investigation_level_msv"], f["report_level_msv"], f["annual_limit_msv"],
                              f["ytd_before_msv"])
        if dor < inv:
            sig = 0
        elif dor < 0.6 * rep and ytd + dor < 0.5 * lim:
            sig = 1
        elif dor <= rep and ytd + dor <= lim:
            sig = 2
        else:
            sig = 3
        reportable = dor > rep or ytd + dor > lim
        pts = len(f["lapses_minor"]) + 2 * len(f["lapses_major"])
        if pts and f["prior_warning"]:
            pts += 1
        fault = 0 if pts == 0 else 1 if pts == 1 else 2 if pts <= 3 else 3
        if f["service_note"] and not real:
            out = "dosimetry_recheck"
        elif fault == 3:
            out = "committee_review"
        elif sig == 3:
            out = "committee_review" if fault >= 2 else "suspend_radiation_work"
        elif sig == 2 and fault >= 2:
            out = "suspend_radiation_work"
        elif fault >= 1 or sig == 2:
            out = "practice_changes_and_monitoring"
        else:
            out = "close_with_note"
        return {"dose_significance": sig, "reading_reflects_real_exposure": "yes" if real else "no",
                "regulator_report_required": "yes" if reportable else "no", "review_outcome": out,
                "work_practice_fault": fault}

    # ------------------------------------------------------------------ rendering
    def _rule_text(self, f):
        kind = "whole-body" if f["badge_type"] == "whole_body" else "extremity (ring)"
        return (f"Investigation level {f['investigation_level_msv']} mSv per {f['period']} {kind} badge. Report to the "
                f"regulator within 30 days if the dose of record for one badge exceeds {f['report_level_msv']} mSv or "
                f"the year-to-date dose exceeds the annual limit of {f['annual_limit_msv']} mSv. If the investigation "
                f"shows the badge did not record the wearer's dose, the RSO assigns an estimated dose (task-based or "
                f"co-worker) and that estimate becomes the dose of record.")

    def _period_label(self, f):
        d = f["date"]
        if f["period"] == "monthly":
            return d.strftime("%B %Y")
        return f"Q{(d.month - 1) // 3 + 1} {d.year}"

    def _statement(self, f, rng):
        if f["story"] == "stored":
            return pick(rng, [f"I think I left the dosimeter {f['story_place']} over the long weekend.",
                              f"The dosimeter may have been {f['story_place']} for a few days while I was away.",
                              f"I took the coat off in a hurry and I believe the dosimeter stayed {f['story_place']}."])
        return pick(rng, NO_STORY).format(task=f["task"], n=f["sessions"])

    def _follow_up(self, f, rng):
        if f["story"] != "stored":
            return None
        if f["corroborated"]:
            rate = round(f["reading_msv"] * 1000 / f["hours_away"], 1)
            return (f"{f['witness']} confirms seeing the dosimeter there; survey at that spot reads about {rate} "
                    f"uSv/h, and it was away from the wearer for roughly {f['hours_away']} h.")
        if f["contradicted"]:
            return pick(rng, [f"Colleagues say the wearer had the dosimeter on as usual that week; the spot described "
                              f"reads {round(rng.uniform(0.08, 0.25), 2)} uSv/h (background).",
                              "The dosimeter rack sign-out sheet shows it signed out and worn on every working day in "
                              "question."])
        return pick(rng, ["Could not be confirmed either way; nobody remembers where the dosimeter was and the spot "
                          "was not surveyed.",
                          "No witness either way; the spot described was cleaned up before anyone surveyed it."])

    def _basis(self, f):
        if f["task_estimate_msv"] is None:
            return None
        if f["badge_type"] == "whole_body":
            rate = round(f["task_estimate_msv"] * 1000 / f["logged_hours"], 1)
            return f"survey ~{rate} uSv/h at the work position x {f['logged_hours']} h logged"
        return f"measured finger dose rate on the vial x {f['logged_hours']} h of logged handling"

    def _lapses(self, f):
        return f["lapses_major"] + f["lapses_minor"]

    def render(self, f, rng, surface):
        d = f["decimals"]
        est = f["task_estimate_msv"]
        btype = "whole body (chest)" if f["badge_type"] == "whole_body" else "extremity ring"
        statement = noisy(self._statement(f, rng), rng, 0.15)
        follow = self._follow_up(f, rng)
        lapses = self._lapses(f)
        if surface == "json":
            return dumps(prune({
                "case": {"id": f["case_id"], "site": f["site"], "opened": fmt_date(f["date"], style="iso"),
                         "radiation_safety_officer": f["rso"]},
                "rule": {"reference": f["rule_ref"], "text": self._rule_text(f)},
                "badge": {"id": f["badge_id"], "type": btype, "wear_period": self._period_label(f),
                          "reading_msv": f["reading_msv"], "control_badge_msv": f["control_msv"],
                          "dosimetry_service_note": f["service_note"],
                          "previous_three_periods_msv": f["previous_msv"],
                          "year_to_date_before_this_badge_msv": f["ytd_before_msv"]},
                "work": {"wearer_role": f["role"], "tasks": f["task"], "source": f["activity"],
                         "task_based_estimate_msv": est, "estimate_basis": self._basis(f),
                         "estimate_note": None if est is not None else "survey log too incomplete for an estimate",
                         "co_worker_same_tasks_msv": f["coworker_msv"]},
                "interview": {"wearer_statement": statement, "follow_up": follow},
                "walkthrough": {"findings": lapses or ["no deviations from the protocol noted"],
                                "previous_written_warning": f["prior_warning"]},
            }))
        if surface == "dosimetry_report":
            rows = [[f"previous {i + 1}", _fmt(v, d)] for i, v in enumerate(f["previous_msv"])]
            rows += [[f"this badge ({self._period_label(f)})", _fmt(f["reading_msv"], d)],
                     ["control badge", _fmt(f["control_msv"], d)]]
            head = kv_lines([
                ("Investigation", f["case_id"]), ("Site", f["site"]), ("Rule", f["rule_ref"]),
                ("Wearer", f"{f['role']}, badge {f['badge_id']} ({btype})"),
                ("Service note", f["service_note"]),
                ("Year to date before this badge", f"{_fmt(f['ytd_before_msv'], d)} mSv"),
                ("Work this period", f"{f['task']} ({f['activity']})"),
                ("Task-based estimate", f"{est} mSv ({self._basis(f)})" if est is not None
                 else "not available, survey log incomplete"),
                ("Co-worker on same tasks", f"{_fmt(f['coworker_msv'], d)} mSv" if f["coworker_msv"] is not None
                 else "no co-worker on these tasks"),
                ("Previous written warning", _yn(f["prior_warning"])),
            ], rng)
            parts = [f"DOSIMETRY INVESTIGATION REPORT", head, "", f"Rule: {self._rule_text(f)}", "",
                     table(["period", "mSv"], rows, rng), "", f"Wearer's statement: \"{statement}\""]
            if follow:
                parts.append(f"Follow-up: {follow}")
            parts += ["Walkthrough:", bullets(lapses or ["no deviations from the protocol noted"], rng)]
            return "\n".join(parts)
        # rso_memo
        prev = ", ".join(_fmt(v, d) for v in f["previous_msv"])
        para = [
            f"Badge {f['badge_id']} ({btype}) worn by a {f['role']} read {_fmt(f['reading_msv'], d)} mSv for "
            f"{self._period_label(f)}, above our investigation level.",
            f"The previous three periods read {prev} mSv; the control badge read {_fmt(f['control_msv'], d)} mSv.",
            f"Dosimetry service: {f['service_note']}." if f["service_note"] else "",
            f"Work this period was {f['task']} with {f['activity']}.",
            (f"From the survey log I estimate {est} mSv for that work ({self._basis(f)})." if est is not None
             else "The survey log is too incomplete to build a task-based estimate."),
            (f"A co-worker on the same tasks read {_fmt(f['coworker_msv'], d)} mSv." if f["coworker_msv"] is not None
             else "Nobody else did these tasks this period."),
            f"Year to date before this badge: {_fmt(f['ytd_before_msv'], d)} mSv.",
            f"In interview the wearer said: \"{statement}\"",
            follow or "",
        ]
        find = ("Walkthrough found: " + "; ".join(lapses) + ".") if lapses else "Walkthrough found no deviations."
        warn = "There is a previous written warning on file." if f["prior_warning"] else "No previous warnings on file."
        return "\n".join([f"MEMO -- {f['site']}", f"From: {f['rso']}, Radiation Safety Officer",
                          f"Re: dosimetry investigation {f['case_id']}", "",
                          noisy(join_sentences(para), rng, 0.1), "", find + " " + warn, "",
                          f"Applicable rule ({f['rule_ref']}): {self._rule_text(f)}"])

    def questions(self):
        return {
            "dose_significance": score(
                "How significant is the dose of record for this badge against the institution's stated levels?", [
                    "Negligible: once the investigation is taken into account, it is below the investigation level.",
                    "Minor: above the investigation level but well short of the reporting level and a small share "
                    "of the annual limit.",
                    "Significant: approaching the reporting level, or a sizeable share of the annual limit.",
                    "Serious: above the reporting level, or it puts the annual limit at risk.",
                ]),
            "reading_reflects_real_exposure": noul(
                "The badge reading reflects dose the wearer actually received.",
                yes="The reading fits the logged work, the survey data or co-workers' badges, and nothing credible "
                    "shows the badge was exposed while not being worn.",
                no="Credible evidence, such as a corroborated account of the badge left near a source, a processing "
                   "flag or a task estimate far below the reading, shows it did not record the wearer's dose."),
            "regulator_report_required": noul("The written rule requires this badge to be reported to the regulator."),
            "review_outcome": choice("How should the radiation safety officer close this investigation?", {
                "close_with_note": "Close it with a note on file; the wearer's radiation work continues unchanged.",
                "committee_review": "Refer the case to the radiation safety committee to review the wearer's "
                                    "authorisation.",
                "dosimetry_recheck": "Ask the dosimetry service to re-read or re-process the badge before reaching "
                                     "any conclusion.",
                "practice_changes_and_monitoring": "Keep the wearer on radiation work with specific practice changes "
                                                   "or retraining and closer dose monitoring.",
                "suspend_radiation_work": "Take the wearer off radiation work until the investigation is complete.",
            }),
            "work_practice_fault": score("How far did the wearer's handling practice fall short of the written rules?", [
                "Not at all: practice matched the rules and the dose came with the work itself.",
                "Slightly: one minor lapse, such as record-keeping or badge storage.",
                "Clearly: a significant shielding or handling lapse, or several minor ones.",
                "Seriously: several significant lapses, or lapses repeated after a written warning.",
            ]),
        }


# =========================================================================== schema 2
# (description, wavelength nm, output, class, required OD from the SOP hazard evaluation)
LASERS = [
    ("Q-switched Nd:YAG", 1064, "450 mJ per pulse at 10 Hz", "4", 7),
    ("frequency-doubled Nd:YAG", 532, "180 mJ per pulse at 10 Hz", "4", 7),
    ("Ti:sapphire regenerative amplifier", 800, "4 mJ, 35 fs pulses at 1 kHz", "4", 7),
    ("mode-locked Ti:sapphire oscillator", 790, "1.8 W average", "4", 5),
    ("argon-ion laser", 514, "2 W CW", "4", 5),
    ("DPSS green laser", 532, "5 W CW", "4", 6),
    ("helium-neon laser", 633, "21 mW CW", "3B", 3),
    ("violet diode module", 405, "300 mW CW", "3B", 4),
    ("CO2 laser", 10600, "60 W CW", "4", 5),
    ("erbium fibre amplifier", 1550, "2 W CW", "4", 3),
    ("KrF excimer laser", 248, "300 mJ per pulse at 20 Hz", "4", 6),
    ("fourth-harmonic Nd:YAG", 266, "40 mJ per pulse at 10 Hz", "4", 6),
    ("pulsed dye laser", 590, "30 mJ per pulse at 10 Hz", "4", 6),
    ("ytterbium fibre laser", 1070, "400 W CW", "4", 8),
    ("tapered amplifier for a cold-atom trap", 780, "1.5 W CW", "4", 5),
    ("1064 nm fibre amplifier for an optical dipole trap", 1064, "20 W CW", "4", 6),
    ("red alignment diode", 650, "45 mW CW", "3B", 3),
    ("picosecond Nd:YVO4 laser", 1064, "8 W average, 10 ps pulses", "4", 7),
    ("optical parametric amplifier (signal)", 1300, "0.5 mJ, 100 fs at 1 kHz", "4", 6),
]
ENG_ISSUES = {
    "interlock_taped": "door interlock switch found taped over; no bypass permit on file",
    "interlock_faulty": "door interlock switch failed closed and did not cut the shutter when the door opened",
    "no_enclosure": "periscope section of the beam path at chest-to-eye height had no cover",
    "shutter_failed": "safety shutter stuck open after a controller reset",
    "beam_block_missing": "the beam dump behind the sample stage had been removed and not replaced",
}
PROC_ISSUES = {
    "full_power_alignment": "alignment done at full output instead of the reduced alignment power",
    "eyewear_removed": "eyewear taken off to see the beam spot, against the SOP",
    "no_beam_blocks": "beam blocks not placed behind the optic being adjusted",
    "jewellery": "wristwatch worn at the table; the beam reflected off its face",
    "eye_level": "worker bent down with eyes at beam height to read a label",
}
AUTH_ISSUES = {
    "training_lapsed": "laser user training lapsed {m} months ago; not on the current user list",
    "untrained_student": "new student, training booked but not yet done; working alone at the table",
    "visitor_unbriefed": "visiting engineer let into the controlled area during a run without a briefing",
}
NONE_EVENTS = [
    "a kinematic mount's retaining screw sheared and the mirror tilted upward",
    "a newly fitted lens had an uncoated back face that sent a ghost reflection out of the plane",
    "a beam-splitter cube shifted in its holder when the table was bumped",
    "an optical post slipped in its holder after a building vibration and re-pointed a turning mirror",
]
ACTIVITIES = ["aligning a turning mirror", "replacing a beam-splitter", "re-pointing the beam onto a new sample",
              "adjusting the periscope", "running a routine measurement", "cleaning the final optic",
              "swapping a sample in the beam", "checking the beam profile"]


class LaserEyeExposure(Schema):
    name = "laser_eye_exposure"
    description = ("A suspected eye exposure from a Class 3B or 4 research laser: was the eyewear adequate, how likely "
                   "is an injury, which control failed, and when may the lab restart.")
    surfaces = {"json": 0.5, "incident_form": 0.25, "lso_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        lname, wl, output, lclass, req = pick(rng, LASERS)
        cause = pick_w(rng, {"engineering": 2.5, "procedure": 3, "ppe": 3, "authorisation": 2, "none": 1.8})
        eng = proc = auth = None
        unforeseen = None
        if cause == "engineering":
            eng = pick(rng, list(ENG_ISSUES))
            if lclass == "3B" and eng in ("interlock_taped", "interlock_faulty"):
                eng = pick(rng, ["no_enclosure", "beam_block_missing", "shutter_failed"])
        elif cause == "procedure":
            proc = pick(rng, list(PROC_ISSUES))
        elif cause == "authorisation":
            auth = pick(rng, list(AUTH_ISSUES))
        elif cause == "none":
            unforeseen = pick(rng, NONE_EVENTS)
        # eyewear
        if cause == "ppe":
            status = pick_w(rng, {"under": 3, "wrong_band": 2})
        elif proc == "eyewear_removed":
            status = "removed"
        elif auth == "visitor_unbriefed":
            status = pick(rng, ["none", "rated"])
        else:
            status = pick_w(rng, {"rated": 4, "none": 1})
        od = None
        if status in ("rated", "removed"):
            od = req + (0 if reg == "borderline" and chance(rng, 0.5) else pick(rng, [1, 1, 2]))
        elif status == "under":
            od = req - (1 if reg == "borderline" else pick(rng, [1, 2, 3]))
        elif status == "wrong_band":
            od = pick(rng, [0.5, 1, 1.5, 2])
        if od is not None and od < 0.5:
            od = 0.5
        band = None
        if status == "wrong_band":
            centre = pick(rng, [c for c in (266, 405, 532, 633, 800, 1064, 10600) if abs(c - wl) > 60])
            lo, hi = (int(centre * 0.95), int(centre * 1.04))
            band = (f"OD {req + pick(rng, [0, 1])}+ {lo}-{hi} nm; maker's chart gives OD {od} at {wl} nm")
        # conflicting: a second failure or a reassuring supervisor
        if reg == "conflicting":
            extra = pick(rng, ["eng", "proc", "auth", "ppe_note"])
            if extra == "eng" and eng is None:
                eng = pick(rng, ["no_enclosure", "beam_block_missing"])
            elif extra == "proc" and proc is None and status != "removed":
                proc = pick(rng, ["no_beam_blocks", "jewellery", "eye_level"])
            elif extra == "auth" and auth is None:
                auth = "training_lapsed"
        if reg == "clear" and cause == "none":
            status = "rated"
            od = req + 1
        path = pick_w(rng, {"specular": 4, "direct": 3, "diffuse": 2, "skin_only": 1})
        if proc == "full_power_alignment" or eng == "no_enclosure":
            path = pick(rng, ["direct", "specular"])
        if reg == "clear":
            path = pick(rng, ["direct", "skin_only"]) if cause != "none" else pick(rng, ["diffuse", "skin_only"])
        adequate = status == "rated" and od is not None and od >= req
        prot = 2 if adequate else (1 if status in ("under",) and od >= req - 1 else 0)
        power_mw = {"helium-neon laser": 21, "red alignment diode": 45}.get(lname)
        e = {"direct": 3, "specular": 2, "diffuse": 1, "skin_only": 0}[path] - prot - (1 if power_mw else 0)
        if adequate and path in ("direct", "specular"):
            e = max(e, 1)                    # beam reached the eyewear: well attenuated, not impossible
        e = max(0, min(3, e))
        exam = "pending"
        if chance(rng, 0.45):
            if e >= 3 and chance(rng, 0.8) or e == 2 and chance(rng, 0.3):
                exam = "lesion"
            else:
                exam = "normal"
        corrected = chance(rng, 0.45)
        lab_in_use = chance(rng, 0.6)
        return {
            "regime": reg, "cause": cause, "laser": lname, "wavelength_nm": wl, "output": output,
            "laser_class": lclass, "required_od": req, "eyewear_status": status, "eyewear_od": od,
            "eyewear_band_note": band, "eyewear_adequate": adequate, "eng_issue": eng, "proc_issue": proc,
            "auth_issue": auth, "unforeseen": unforeseen, "path": path, "exposure_index": e, "exam": exam,
            "corrected_since": corrected, "lab_in_use": lab_in_use,
            "prior_incidents_12m": pick_w(rng, {0: 6, 1: 3, 2: 1}),
            "lapsed_months": rng.randint(2, 30), "align_mw": pick(rng, [1, 2, 5, 5, 10]),
            "activity": pick(rng, ACTIVITIES), "hours_since": pick(rng, [1, 3, 6, 20, 30, 52]),
            "supervisor_view": pick(rng, ["no harm done, the beam only glanced off", "a one-off, the student is careful",
                                          "the lab is otherwise well run", None, None]),
            "worker": person(rng), "lso": person(rng), "pi": person(rng), "site": _site(rng, "laser"), "room": _room(rng),
            "rule_ref": _rule(rng, LASER_BOOKS), "report_id": ident(rng, "LSR", 4), "when": rand_ts(rng, hours=(8, 22)),
        }

    def _symptoms(self, f):
        e, wl = f["exposure_index"], f["wavelength_nm"]
        if 400 <= wl < 700:
            return ["no symptoms reported", "a brief bright flash, no after-effects",
                    "a green-purple afterimage that took about 20 minutes to fade",
                    "a dark spot in the centre of vision that has not cleared"][e]
        if 700 <= wl < 1400:
            return ["no symptoms reported", "no symptoms; only noticed the beam on the IR viewer card afterwards",
                    "a faint flash and a popping sound, vision seems normal since",
                    "a faint flash and a blurred patch in one eye since"][e]
        return ["no symptoms reported", "mild irritation the next morning",
                "gritty, painful eye starting a few hours later",
                "severe eye pain and light sensitivity within hours, still present"][e]

    def _primary(self, f):
        if f["auth_issue"]:
            return "user_authorisation"
        if f["eng_issue"]:
            return "engineering_controls"
        if f["proc_issue"]:
            return "written_procedure"
        if f["eyewear_status"] in ("under", "wrong_band"):
            return "eye_protection"
        return "no_control_failure"

    def intended(self, f):
        e, exam = f["exposure_index"], f["exam"]
        if exam == "lesion":
            inj = 3
        elif exam == "normal":
            inj = 1 if e >= 2 else 0
        else:
            inj = e
        prim = self._primary(f)
        persists = (not f["corrected_since"]) and f["lab_in_use"] and prim != "no_control_failure"
        if inj == 3 and prim != "no_control_failure":
            restart = "keep_locked_out"
        elif f["eng_issue"] == "interlock_taped" or (f["prior_incidents_12m"] >= 2 and prim != "no_control_failure"):
            restart = "keep_locked_out"
        elif prim == "user_authorisation":
            restart = "resume_restricted_users"
        elif prim == "no_control_failure":
            restart = "resume_now" if inj <= 1 else "resume_after_fixes"
        elif f["corrected_since"]:
            restart = "resume_now"
        else:
            restart = "resume_after_fixes"
        if exam == "pending" and inj >= 2:
            timing = 3
        elif persists:
            timing = 2
        elif exam == "pending" and inj == 1:
            timing = 2
        elif prim != "no_control_failure" or inj >= 1:
            timing = 1
        else:
            timing = 0
        return {"eyewear_adequate": "yes" if f["eyewear_adequate"] else "no", "injury_likelihood": inj,
                "primary_control_failure": prim, "lab_restart_condition": restart, "lso_response_timing": timing}

    def _rule_text(self, f):
        return (f"Class 3B/4 work only by registered users. Eyewear must give at least OD {f['required_od']} at "
                f"{f['wavelength_nm']} nm (SOP hazard evaluation). Interlocks may not be defeated without a written "
                f"bypass permit. Align at {f['align_mw']} mW or less with beam blocks in place. Report any suspected "
                f"eye exposure to the LSO at once; eye examination within 24 h.")

    def _eyewear_text(self, f):
        s = f["eyewear_status"]
        if s == "none":
            return "no eyewear worn"
        if s == "wrong_band":
            return f"eyewear labelled {f['eyewear_band_note']}"
        if s == "removed":
            return f"eyewear rated OD {f['eyewear_od']}+ at {f['wavelength_nm']} nm, pushed up onto the forehead at the time"
        return f"eyewear rated OD {f['eyewear_od']}+ at {f['wavelength_nm']} nm, worn"

    def _findings(self, f):
        out = []
        if f["eng_issue"]:
            out.append(ENG_ISSUES[f["eng_issue"]])
        if f["proc_issue"]:
            out.append(PROC_ISSUES[f["proc_issue"]])
        if f["auth_issue"]:
            out.append(AUTH_ISSUES[f["auth_issue"]].format(m=f["lapsed_months"]))
        if f["eyewear_status"] == "under":
            out.append("eyewear was the group's older set, bought for a previous laser on this table")
        elif f["eyewear_status"] == "wrong_band":
            out.append("eyewear was taken from the shared rack by the door")
        if f["unforeseen"]:
            out.append(f"immediate trigger: {f['unforeseen']}")
        return out

    def _path_text(self, f):
        return {"direct": "direct beam toward the face", "specular": "specular reflection off a mount or tool",
                "diffuse": "diffuse reflection from a card or beam dump", "skin_only": "beam crossed the hand; eyes "
                "were out of the beam plane"}[f["path"]]

    def _status_now(self, f):
        if f["corrected_since"]:
            fix = "the corrective actions listed in the report are complete and checked"
        else:
            fix = "nothing has been changed yet"
        use = "the group is still using the laser" if f["lab_in_use"] else "the PI has locked the laser out"
        return f"{fix}; {use}"

    def _exam_text(self, f):
        return {"pending": f"not yet examined ({f['hours_since']} h after the event)",
                "normal": "eye examination normal",
                "lesion": _lesion_text(f)}[f["exam"]]

    def render(self, f, rng, surface):
        sym = self._symptoms(f)
        finds = self._findings(f)
        exam = self._exam_text(f)
        if surface == "json":
            return dumps(prune({
                "report": {"id": f["report_id"], "site": f["site"], "room": f["room"],
                           "time": fmt_ts(f["when"], style="iso"), "reported_by": f["worker"],
                           "activity": f["activity"]},
                "laser": {"type": f["laser"], "wavelength_nm": f["wavelength_nm"], "output": f["output"],
                          "class": f["laser_class"]},
                "rule": {"reference": f["rule_ref"], "text": self._rule_text(f)},
                "exposure": {"path": self._path_text(f), "eyewear": self._eyewear_text(f), "symptoms": sym,
                             "eye_examination": exam},
                "investigation": {"findings": finds or ["no further deviation identified"],
                                  "supervisor_comment": f["supervisor_view"],
                                  "laser_incidents_in_lab_last_12_months": f["prior_incidents_12m"],
                                  "status_now": self._status_now(f)},
            }))
        if surface == "incident_form":
            body = kv_lines([
                ("Report", f["report_id"]), ("Site / room", f"{f['site']} / {f['room']}"),
                ("Date and time", fmt_ts(f["when"], rng)), ("Person involved", f["worker"]),
                ("Laser", f"{f['laser']}, {f['wavelength_nm']} nm, {f['output']}, Class {f['laser_class']}"),
                ("Task at the time", f["activity"]), ("Exposure route", self._path_text(f)),
                ("Eye protection", self._eyewear_text(f)), ("Symptoms", sym), ("Eye examination", exam),
                ("Laser incidents in this lab, last 12 months", f["prior_incidents_12m"]),
                ("Status now", self._status_now(f)),
            ], rng)
            return "\n".join([f"LASER INCIDENT REPORT ({f['rule_ref']})", body, "",
                              "Investigator findings:", bullets(finds or ["no further deviation identified"], rng),
                              (f"PI comment: \"{f['supervisor_view']}\"" if f["supervisor_view"] else ""), "",
                              f"Rule: {self._rule_text(f)}"]).strip()
        # lso_email
        open_line = pick(rng, [f"Following up on the laser report from {fmt_ts(f['when'], rng)} in {f['room']}.",
                               f"Summary of what I found on the {f['laser']} incident in {f['room']}.",
                               f"Notes from my visit to {f['room']} after the {f['laser']} report."])
        para = [open_line,
                f"{f['worker']} was {f['activity']} on the {f['laser']} ({f['wavelength_nm']} nm, {f['output']}, "
                f"Class {f['laser_class']}).",
                f"Route: {self._path_text(f)}. Eye protection: {self._eyewear_text(f)}.",
                f"Symptoms: {sym}. Examination: {exam}."]
        para += [x[0].upper() + x[1:] + "." for x in finds]
        if not finds:
            para.append("I found no further deviation.")
        para.append(f"This lab has had {f['prior_incidents_12m']} other laser incident{'' if f['prior_incidents_12m'] == 1 else 's'} in the last 12 months.")
        para.append(f"Current status: {self._status_now(f)}.")
        reply = (f"\n\n> {f['pi']} wrote: \"{f['supervisor_view'][0].upper() + f['supervisor_view'][1:]}.\""
                 if f["supervisor_view"] else "")
        return (f"Subject: Laser incident {f['report_id']} -- {f['site']}\nFrom: {f['lso']} (Laser Safety Officer)\n"
                f"To: laser-safety-committee@{pick(rng, ['lab', 'research', 'safety'])}.example.org\n\n"
                + noisy(join_sentences(para), rng, 0.12) + reply
                + f"\n\nFor reference, {f['rule_ref']}: {self._rule_text(f)}")

    def questions(self):
        return {
            "eyewear_adequate": noul(
                "The eye protection in use gave the protection the written rule requires at this laser's wavelength.",
                yes="Eyewear was being worn at the time and its optical density at the operating wavelength meets or "
                    "exceeds the required value.",
                no="No eyewear, eyewear taken off, the wrong band, or an optical density below the required value "
                   "at the operating wavelength."),
            "injury_likelihood": score("How likely is it that this exposure injured the person's eye?", [
                "Very unlikely: no credible route for harmful light to reach the eye.",
                "Unlikely: a brief or well-attenuated exposure; injury cannot be ruled out entirely.",
                "Likely: an unprotected or under-protected exposure with symptoms consistent with harm.",
                "Confirmed or near certain: an examination found damage, or classic injury signs persist.",
            ]),
            "primary_control_failure": choice("Which failed control should the review treat as the primary cause?", {
                "engineering_controls": "An engineering control was missing, defeated or faulty, such as an "
                                        "interlock, beam enclosure, shutter or beam dump.",
                "eye_protection": "The eyewear worn was the wrong type or had too low an optical density at the "
                                  "operating wavelength.",
                "no_control_failure": "Every required control was in place and working; the exposure came from an "
                                      "event the controls could not reasonably anticipate.",
                "user_authorisation": "The person was not trained or authorised for this laser, or was let in "
                                      "without the required briefing.",
                "written_procedure": "The written alignment or operating procedure was not followed, for example "
                                     "full-power alignment, no beam blocks or eyewear taken off.",
            }),
            "lab_restart_condition": choice("On what terms may the laser lab resume work?", {
                "keep_locked_out": "Keep the laser locked out until the laser safety committee completes a full "
                                   "review.",
                "resume_after_fixes": "Resume once the named deficiencies are corrected and checked by the laser "
                                      "safety officer.",
                "resume_now": "Resume now; nothing has to be fixed first.",
                "resume_restricted_users": "Resume for fully authorised users only; the person involved works "
                                           "under supervision until trained.",
            }),
            "lso_response_timing": score("How quickly does the laser safety officer need to act on this report?", [
                "Routine: record it and review at the next scheduled laser-safety inspection.",
                "Soon: follow up within the week, before the group's next booked laser session.",
                "Today: before the laser is switched on again, because a hazard or open question remains.",
                "Immediately: get the person to an eye examination now and lock the laser out.",
            ]),
        }


def _lesion_text(f):
    """Examination finding text for a confirmed injury, consistent with the wavelength band."""
    wl = f["wavelength_nm"]
    if wl < 400 or wl >= 1400:
        return "examination found corneal epithelial damage"
    return "examination found a small retinal burn" + (" near the fovea" if f["exposure_index"] >= 3 else
                                                      " outside the central macula")


# =========================================================================== schema 3
CRYO_EVENTS = {
    "transfer_spill": dict(cryo="LN2", litres=(4, 60), room=(35, 260), dur=(0.05, 0.25)),
    "autofill_overflow": dict(cryo="LN2", litres=(20, 180), room=(50, 320), dur=(1.0, 6.0)),
    "relief_venting": dict(cryo=None, litres=(15, 120), room=(30, 220), dur=(0.5, 4.0)),
    "magnet_quench": dict(cryo="LHe", litres=(60, 650), room=(120, 1600), dur=(0.02, 0.1)),
    "lhe_fill_loss": dict(cryo="LHe", litres=(8, 50), room=(35, 220), dur=(0.2, 1.0)),
    "dewar_in_lift": dict(cryo="LN2", litres=(2, 14), room=(4, 9), dur=(0.05, 0.3)),
}
GAS_PER_LITRE = {"LN2": 0.694, "LHe": 0.757}       # m3 of gas at ~20 C per litre of liquid
CAUSE_TEXT = {
    ("transfer_spill", "handling"): ["receiving dewar overfilled while the operator stepped away",
                                     "transfer made into an unrated vacuum flask that cracked",
                                     "fill hose pulled out of the neck before the valve was shut"],
    ("transfer_spill", "equipment"): ["transfer hose split at the bayonet fitting", "phase separator jammed open"],
    ("autofill_overflow", "equipment"): ["autofill solenoid valve stuck open overnight",
                                         "level sensor iced up and never signalled full"],
    ("autofill_overflow", "handling"): ["autofill switched to manual fill and left running at the end of the day"],
    ("relief_venting", "equipment"): ["pressure-relief valve iced open on the supply dewar",
                                      "burst disc on the storage dewar ruptured"],
    ("relief_venting", "design"): ["routine top-up of a storage dewar from the bulk line, with the usual "
                                   "cool-down and flash losses venting into the room"],
    ("magnet_quench", "design"): ["superconducting magnet quenched during a field ramp within its rating"],
    ("magnet_quench", "equipment"): ["quench line coupling had been left disconnected after maintenance"],
    ("magnet_quench", "handling"): ["magnet ramped above its rated field during an unauthorised test"],
    ("lhe_fill_loss", "handling"): ["transfer line forced into the cryostat neck and the seal blew out"],
    ("lhe_fill_loss", "design"): ["normal flash losses during a routine helium fill"],
    ("dewar_in_lift", "handling"): ["operator rode in the lift with a pressurised dewar against the transport rule"],
}
EVENT_TEXT = {
    "transfer_spill": "liquid nitrogen spill during a dewar transfer",
    "autofill_overflow": "liquid nitrogen autofill overflow",
    "relief_venting": "gas venting from a storage dewar",
    "magnet_quench": "superconducting magnet quench",
    "lhe_fill_loss": "liquid helium loss during a cryostat fill",
    "dewar_in_lift": "nitrogen release from a dewar in a passenger lift",
}


class CryogenOdhEvent(Schema):
    name = "cryogen_odh_event"
    description = ("A cryogen release in an enclosed lab space (spill, overflow, venting, magnet quench): oxygen "
                   "hazard, the response against the written rule, cause, and whether the room rating holds.")
    surfaces = {"json": 0.45, "monitor_log": 0.3, "safety_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        ev = pick_w(rng, {"transfer_spill": 3, "autofill_overflow": 2, "relief_venting": 2, "magnet_quench": 3,
                          "lhe_fill_loss": 2, "dewar_in_lift": 1})
        spec = CRYO_EVENTS[ev]
        cryo = spec["cryo"] or pick(rng, ["LN2", "LHe"])
        alarm = pick(rng, [19.5, 19.5, 19.0])
        evac = pick(rng, [18.0, 18.0, 17.5, 18.5])
        reentry = pick(rng, [alarm, 20.0, 20.5])
        # cause of the release
        cause_w = {"transfer_spill": {"handling": 6, "equipment": 4}, "autofill_overflow": {"equipment": 6, "handling": 4},
                   "relief_venting": {"equipment": 6, "design": 4}, "magnet_quench": {"design": 6, "equipment": 3, "handling": 1.5},
                   "lhe_fill_loss": {"handling": 4, "design": 6}, "dewar_in_lift": {"handling": 1}}[ev]
        cause = pick_w(rng, cause_w)
        cause_text = pick(rng, CAUSE_TEXT[(ev, cause)])
        # ventilation, quench line, monitor
        if ev == "dewar_in_lift":
            ach_design, vent = 0.5, "running"
        else:
            ach_design = pick(rng, [2, 4, 6, 6, 8, 10])
            vent = pick_w(rng, {"running": 6, "off": 2, "undersized": 2})
        ach = ach_design if vent == "running" else (0.0 if vent == "off" else round(ach_design * rng.uniform(0.15, 0.35), 1))
        f_room = 1.0
        if ev == "magnet_quench":
            f_room = 1.0 if cause == "equipment" else round(rng.uniform(0.08, 0.2), 2)
        mon = "none" if ev == "dewar_in_lift" else pick_w(rng, {"ok": 5, "overdue": 2, "misplaced": 2, "none": 1})
        # steer the minimum oxygen the release could produce
        if reg == "clear":
            target = pick(rng, [between(rng, 19.0, 20.4), between(rng, 10.5, 13.5)])
        elif reg == "borderline":
            target = pick(rng, [near(rng, evac, 0.7), near(rng, 14.0, 0.8), near(rng, evac - 1.0, 0.6)])
        else:
            target = between(rng, 12.0, 19.8)
        eff = max(0.02, 1 - target / 20.9)
        dur = between(rng, *spec["dur"])
        dil = 1 / (1 + ach * dur)
        vroom = between(rng, *spec["room"])
        k = GAS_PER_LITRE[cryo] * f_room * dil
        litres = eff * vroom / k
        lo, hi = spec["litres"]
        if not lo <= litres <= hi:
            litres = min(hi, max(lo, litres))
            vroom = min(spec["room"][1], max(spec["room"][0], litres * k / eff))
        litres, vroom = int(round(litres)), int(round(vroom))
        o2_est = round(20.9 * (1 - min(0.85, litres * k / vroom)), 1)
        measured = None
        if mon == "ok":
            measured = o2_est + rng.uniform(-0.3, 0.4)
        elif mon == "overdue":
            measured = o2_est + rng.uniform(-0.4, 1.8)
        elif mon == "misplaced":
            measured = o2_est + rng.uniform(1.0, 3.0)
        portable = None
        if mon == "none":
            portable = round(min(20.9, o2_est + rng.uniform(0.5, 2.5)), 1)
        if measured is not None:
            measured = round(min(20.9, measured), 1)
        alarmed = measured is not None and measured < alarm
        # people and their response
        if ev == "dewar_in_lift":
            people = "one"
        elif ev in ("autofill_overflow", "relief_venting"):
            people = pick_w(rng, {"none_overnight": 5, "one": 2, "two": 2})
        else:
            people = pick_w(rng, {"one": 4, "two": 4, "none_overnight": 1})
        transfer = ev in ("transfer_spill", "lhe_fill_loss")
        lone_limit = pick(rng, [25, 50, 100])
        transfer_vol = None
        if transfer:
            transfer_vol = int(near(rng, lone_limit, lone_limit * 0.5, lo=litres + 5)) if reg == "borderline" else \
                int(max(litres + 5, rng.uniform(0.3, 2.5) * lone_limit))
        if ev == "lhe_fill_loss":
            transfer_vol = max(transfer_vol, int(litres * 2.5))   # losses are a fraction of the transfer
        lone_violation = transfer and people == "one" and transfer_vol > lone_limit
        bad = {"clear": 0.5, "borderline": 0.3, "conflicting": 0.45}[reg]
        continued = silenced = reentered_early = no_check = False
        if people != "none_overnight" and alarmed:
            continued = chance(rng, bad * 0.6)
            silenced = chance(rng, bad * 0.4)
        if people != "none_overnight":
            reentered_early = chance(rng, bad * 0.5)
        elif alarmed:
            no_check = chance(rng, bad)
        if reg == "clear" and not (lone_violation or continued or silenced or reentered_early or no_check) \
                and chance(rng, 0.3) and people != "none_overnight":
            reentered_early = True
        posted = pick_w(rng, {0: 6, 1: 3, 2: 1})
        fix = pick_w(rng, {"repaired_verified": 4, "temporary": 2, "not_fixed": 4})
        still = None
        if fix == "not_fixed":
            if ev in ("relief_venting", "autofill_overflow") and o2_est < alarm and chance(rng, 0.75):
                still = "release"
            elif ev == "magnet_quench" and cause == "equipment" and chance(rng, 0.7):
                still = "quench_line"
            elif vent == "off" and chance(rng, 0.5):
                still = "vent_off"
        cal_int = pick(rng, [6, 12])
        cal_age = rng.randint(1, cal_int - 1) if mon != "overdue" else cal_int + rng.randint(2, 14)
        symptoms = None
        x = min(o2_est, measured) if measured is not None else o2_est
        if people != "none_overnight" and x < 17.0:
            symptoms = pick(rng, ["feeling light-headed and short of breath", "a headache afterwards",
                                  "feeling dizzy near the door"])
        return {
            "regime": reg, "event": ev, "cryogen": cryo, "release_cause": cause, "cause_text": cause_text,
            "litres_released": litres, "room_m3": vroom, "ach_design": ach_design, "ventilation": vent,
            "ach_actual": ach, "quench_fraction_into_room": f_room, "release_minutes": int(round(dur * 60)) or 1,
            "o2_estimate": o2_est, "monitor": mon, "measured_min_o2": measured, "portable_o2": portable,
            "alarm_level": alarm, "evac_level": evac, "reentry_level": reentry, "alarmed": alarmed,
            "people": people, "transfer_volume_l": transfer_vol, "lone_limit_l": lone_limit,
            "lone_violation": lone_violation, "continued_after_alarm": continued, "alarm_silenced": silenced,
            "reentered_early": reentered_early, "responder_entered_unchecked": no_check, "symptoms": symptoms,
            "posted_odh_class": posted, "fix_status": fix, "still_hazardous": still,
            "cal_interval_months": cal_int, "cal_age_months": cal_age, "site": _site(rng, "cryo"), "room": _room(rng),
            "rule_ref": _rule(rng, CRYO_BOOKS), "event_id": ident(rng, "ODH", 4), "when": rand_ts(rng),
            "worker": person(rng), "responder": person(rng), "author": person(rng),
            "room_kind": pick(rng, ["NMR suite", "cryostat lab", "detector assembly hall", "magnet test bay",
                                    "sample-prep room", "beamline hutch", "low-temperature lab"])
            if ev != "dewar_in_lift" else "passenger lift",
        }

    def _hazard_x(self, f):
        m = f["measured_min_o2"]
        return min(f["o2_estimate"], m) if m is not None else f["o2_estimate"]

    def intended(self, f):
        x, ev = self._hazard_x(f), f["evac_level"]
        hz = 0 if x >= ev else 1 if x >= ev - 1.0 else 2 if x >= 14.0 else 3
        ok = not (f["lone_violation"] or f["continued_after_alarm"] or f["alarm_silenced"] or f["reentered_early"]
                  or f["responder_entered_unchecked"] or f["event"] == "dewar_in_lift")
        c = f["release_cause"]
        if c == "handling":
            rc = "handling_error"
        elif c == "equipment":
            rc = "equipment_fault"
        elif f["ventilation"] != "running":
            rc = "ventilation_shortfall"
        elif f["monitor"] != "ok":
            rc = "monitoring_shortfall"
        else:
            rc = "design_basis_event"
        implied = 0 if x >= ev else 1 if x >= 14.0 else 2
        understated = implied > f["posted_odh_class"]
        if f["still_hazardous"]:
            t = 3
        elif f["fix_status"] == "not_fixed":
            t = 2 if hz >= 1 or understated else 1
        elif f["fix_status"] == "temporary":
            t = 1
        else:
            t = 1 if understated else 0
        return {"oxygen_hazard": hz, "response_followed_rule": "yes" if ok else "no", "main_cause": rc,
                "odh_rating_understated": "yes" if understated else "no", "room_safety_timing": t}

    def _rule_text(self, f):
        return (f"Fixed O2 monitors alarm at {f['alarm_level']}%: stop transfers and leave. Below {f['evac_level']}% "
                f"evacuate; no re-entry until the reading is back to {f['reentry_level']}% or more. No lone working "
                f"during transfers over {f['lone_limit_l']} L. Dewars travel in lifts unaccompanied. Monitors are "
                f"calibrated every {f['cal_interval_months']} months. Room ratings: ODH-0 = worst credible release "
                f"stays at or above {f['evac_level']}%; ODH-1 = 14-{f['evac_level']}% (monitor, training, buddy rule); "
                f"ODH-2 = below 14% (also interlocked extract and personal monitors).")

    def _monitor_text(self, f):
        cr = f["cryogen"]
        place_bad = "near the ceiling" if cr == "LN2" else "at knee height"
        place_ok = "at about 1 m, near the floor-level pooling zone" if cr == "LN2" else "high on the wall near the ceiling"
        m = f["monitor"]
        if m == "none":
            return "no fixed O2 monitor in this space"
        if m == "misplaced":
            return f"fixed monitor mounted {place_bad}; calibrated {f['cal_age_months']} months ago"
        if m == "overdue":
            return f"fixed monitor {place_ok}; last calibrated {f['cal_age_months']} months ago"
        return f"fixed monitor {place_ok}; calibrated {f['cal_age_months']} months ago"

    def _vent_text(self, f, rng):
        v = f["ventilation"]
        if v == "running":
            return f"extract running at its design {f['ach_design']} air changes/h"
        if v == "off":
            return pick(rng, ["extract fan switched off to cut vibration for a measurement",
                              "extract fan off; breaker tripped earlier in the week and not reset",
                              "extract switched off by the building system's night setback"])
        return f"extract running but measured at {f['ach_actual']} air changes/h against a design of {f['ach_design']}"

    def _response(self, f):
        out = []
        if f["people"] == "none_overnight" and not f["alarmed"]:
            out.append("room unoccupied and no alarm raised; the release was found on the next walk-round")
        elif f["people"] == "none_overnight":
            out.append("room unoccupied; alarm went to the site control room")
            out.append(f"responder {'entered before checking the remote O2 reading' if f['responder_entered_unchecked'] else 'checked the remote reading and waited for it to recover before entering'}")
        else:
            who = "one person present" if f["people"] == "one" else "two people present"
            if f["transfer_volume_l"]:
                who += f", transferring {f['transfer_volume_l']} L"
            out.append(who)
            if f["alarmed"]:
                out.append(("carried on to finish the fill after the alarm" if f["transfer_volume_l"]
                            else "carried on working in the room after the alarm") if f["continued_after_alarm"]
                           else "stopped work when the alarm sounded and left the room")
                if f["alarm_silenced"]:
                    out.append("alarm was silenced at the panel before leaving")
            else:
                out.append("no alarm sounded, so work continued until the release was noticed")
            if f["reentered_early"]:
                out.append(f"re-entered to close a valve while the reading was still below {f['reentry_level']}%")
            else:
                out.append("did not re-enter until the reading recovered")
        if f["symptoms"]:
            out.append(f"one person reported {f['symptoms']}")
        return out

    def _fix_text(self, f):
        st = f["still_hazardous"]
        if st == "vent_off":
            return "extract still off; people are working in the room and the dewars are still stored there"
        if st == "release":
            return "release not yet stopped at source; the dewar is still venting into the room"
        if st == "quench_line":
            return "quench line still disconnected and the magnet has been ramped back to field"
        return {"repaired_verified": "cause corrected and the room re-checked by the safety office",
                "temporary": "temporary measure in place; permanent fix still on order",
                "not_fixed": "no corrective action taken yet; room back in normal use"}[f["fix_status"]]

    def render(self, f, rng, surface):
        rule = self._rule_text(f)
        gas = round(f["litres_released"] * GAS_PER_LITRE[f["cryogen"]], 1)
        quench = None
        if f["event"] == "magnet_quench":
            quench = ("quench line intact; most gas vented outdoors" if f["quench_fraction_into_room"] < 1
                      else "quench line found disconnected; all gas released into the room")
        resp = self._response(f)
        fix = self._fix_text(f)
        vent = self._vent_text(f, rng)
        meas = (f"{f['measured_min_o2']}%" if f["measured_min_o2"] is not None
                else f"no fixed reading; responder's portable meter read {f['portable_o2']}% on entry")
        if surface == "json":
            return dumps(prune({
                "event": {"id": f["event_id"], "site": f["site"], "location": f"{f['room']} ({f['room_kind']})",
                          "time": fmt_ts(f["when"], style="iso"), "type": EVENT_TEXT[f["event"]],
                          "cryogen": f["cryogen"], "litres_liquid_released": f["litres_released"],
                          "release_over_minutes": f["release_minutes"], "what_happened": f["cause_text"],
                          "quench_line": quench},
                "room": {"volume_m3": f["room_m3"], "ventilation": vent,
                         "posted_odh_rating": f"ODH-{f['posted_odh_class']}", "oxygen_monitor": self._monitor_text(f)},
                "oxygen": {"lowest_monitor_reading": meas,
                           "safety_office_estimate_min_o2_pct": f["o2_estimate"],
                           "estimate_basis": f"{gas} m3 of gas into {f['room_m3']} m3 with ventilation as found"},
                "response": resp,
                "rule": {"reference": f["rule_ref"], "text": rule},
                "follow_up": {"corrective_action": fix},
            }))
        if surface == "monitor_log":
            t = f["when"]
            lines = []
            base_o2 = 20.9
            if f["measured_min_o2"] is not None:
                m = f["measured_min_o2"]
                pts = [base_o2, round((base_o2 + m) / 2, 1), m, min(20.9, round((m + base_o2) / 2 + 0.3, 1)), 20.8]
                for i, v in enumerate(pts):
                    tag = ""
                    if v < f["evac_level"]:
                        tag = "  EVACUATE"
                    elif v < f["alarm_level"]:
                        tag = "  LOW ALARM"
                    lines.append(f"{fmt_ts(t + timedelta(minutes=[0, 3, 7, 25, 55][i]), style='log')}  "
                                 f"O2-{f['room']}  {v:.1f}%{tag}")
            else:
                lines.append(f"{fmt_ts(t, style='log')}  (no fixed O2 channel for this space)")
            notes = [f"{fmt_ts(t, style='log')}  {EVENT_TEXT[f['event']]}: {f['cause_text']}"] + \
                    [f"{fmt_ts(t + timedelta(minutes=10 + 4 * i), style='log')}  {r}" for i, r in enumerate(resp)]
            facts = kv_lines([
                ("Event", f["event_id"]), ("Site", f["site"]), ("Location", f"{f['room']} ({f['room_kind']})"),
                ("Cryogen released", f"{f['litres_released']} L {f['cryogen']} over ~{f['release_minutes']} min"),
                ("Quench line", quench), ("Room volume", f"{f['room_m3']} m3"),
                ("Ventilation", vent), ("Monitor", self._monitor_text(f)),
                ("Lowest reading", meas),
                ("Safety office estimate of minimum O2", f"{f['o2_estimate']}% ({gas} m3 gas, ventilation as found)"),
                ("Posted rating", f"ODH-{f['posted_odh_class']}"), ("Corrective action", fix),
            ], rng)
            return "\n".join(["O2 MONITORING EXPORT + CONTROL ROOM NOTES"] + lines + [""] + notes +
                             ["", facts, "", f"{f['rule_ref']}: {rule}"])
        # safety_memo
        para = [f"On {fmt_date(f['when'].date(), rng)} there was a {EVENT_TEXT[f['event']]} in {f['room']} "
                f"({f['room_kind']}, {f['room_m3']} m3) at {f['site']}: {f['cause_text']}.",
                f"About {f['litres_released']} L of {f['cryogen']} was released over roughly {f['release_minutes']} "
                f"minutes, around {gas} m3 of gas.",
                (quench[0].upper() + quench[1:] + ".") if quench else "",
                f"Ventilation: {vent}. Monitoring: {self._monitor_text(f)}.",
                f"Lowest reading: {meas}. Our estimate of the minimum oxygen level with the ventilation as found is "
                f"{f['o2_estimate']}%.",
                "Response: " + "; ".join(resp) + ".",
                f"The room is posted as ODH-{f['posted_odh_class']}.",
                f"Corrective action: {fix}."]
        return (f"MEMO -- oxygen deficiency event {f['event_id']}\nFrom: {f['author']}, safety office\n\n"
                + noisy(join_sentences(para), rng, 0.1) + f"\n\nRule ({f['rule_ref']}): {rule}")

    def questions(self):
        return {
            "oxygen_hazard": score("How dangerous was the oxygen deficiency this release could have produced?", [
                "Low: oxygen stayed at or above the evacuation level throughout, even by the worst estimate.",
                "Elevated: oxygen fell to within about one point below the evacuation level in part of the room.",
                "High: oxygen fell clearly below the evacuation level, where judgement and coordination suffer.",
                "Life-threatening: oxygen fell to around 14% or less, where people can collapse within minutes.",
            ]),
            "response_followed_rule": noul(
                "Everyone involved responded to the release as the written cryogen rule requires.",
                yes="Work stopped at the alarm, people left and re-entered only at the levels the rule sets, and "
                    "the lone-working and transport conditions were respected.",
                no="Someone carried on through an alarm, silenced it, re-entered before the stated recovery "
                   "reading, entered without checking, or broke the lone-working or lift-transport condition."),
            "main_cause": choice("What was the main cause of this event becoming a hazard?", {
                "design_basis_event": "A foreseeable event the room was designed for occurred, and the ventilation "
                                      "and monitoring worked as intended.",
                "equipment_fault": "A piece of equipment failed, such as a relief valve, autofill controller, hose "
                                   "or quench line.",
                "handling_error": "The person handling the cryogen did something the procedure forbids, such as "
                                  "overfilling, leaving a fill unattended or riding with a dewar.",
                "monitoring_shortfall": "Oxygen monitoring was missing, overdue for calibration or mounted where it "
                                        "could not see the hazard.",
                "ventilation_shortfall": "The room's ventilation was off or delivering far less than its design for "
                                         "the cryogen kept there.",
            }),
            "odh_rating_understated": noul(
                "The room's posted oxygen-deficiency rating understates the hazard this release showed."),
            "room_safety_timing": score("How soon must the safety office act on this room?", [
                "At the next routine oxygen-deficiency review of the building.",
                "Before the next scheduled cryogen delivery or fill in this room.",
                "Before anyone works in the room again.",
                "Now: the room is still unsafe or people are still exposed.",
            ]),
        }


# =========================================================================== schema 4
# (equipment, rated kV, capacitance uF, current-limited supply?)
HV_EQUIP = [
    ("drift-chamber bias supply and filter box", 3.0, 0.1, True),
    ("photomultiplier HV crate", 2.0, 0.02, True),
    ("klystron modulator pulse-forming network", 28.0, 2.0, False),
    ("capacitor bank for the pulsed magnet", 8.0, 500.0, False),
    ("flashlamp capacitor bank on the Nd:YAG", 1.5, 400.0, False),
    ("ion-source extraction supply", 35.0, 0.05, False),
    ("electrostatic deflector supply", 120.0, 0.002, True),
    ("X-ray tube generator", 60.0, 0.01, False),
    ("Pockels-cell driver", 4.0, 0.05, True),
    ("plasma-source DC supply", 15.0, 10.0, False),
    ("gas-detector test bench supply", 5.0, 0.2, True),
    ("electron-gun cathode supply", 90.0, 0.004, False),
    ("Marx generator stage", 40.0, 0.1, False),
    ("silicon tracker bias rack", 0.5, 2.0, True),
    ("RF amplifier anode supply", 12.0, 8.0, False),
    ("electrostatic precipitator test supply", 25.0, 0.02, False),
    ("streak-camera sweep unit", 6.0, 0.03, True),
]
GAP_EVIDENCE = {
    "training_gap": ["HV qualification expired {m} months ago", "the worker has never completed the HV qualification course",
                     "trained on low-voltage work only; not on the HV authorised list"],
    "procedure_gap": ["the group's local procedure (last revised {y}) has no grounding-hook step for this unit",
                      "the local procedure still describes the old supply and its built-in dump relay"],
    "supervision_pressure": ["supervisor asked for the fix before the {slot} beam slot and said the permit could follow",
                             "work was assigned by the group leader without raising a permit, to save the run time"],
    "individual_shortcut": ["worker says they have done the job many times and skipped the steps to save time",
                            "worker knew the procedure and admits cutting corners at the end of a long shift"],
    "equipment_design": ["bleeder resistor found open-circuit, so the capacitors held charge",
                         "dump relay stuck open after the supply was switched off"],
}
OUTCOMES = {
    "near_miss": "no contact; a colleague stopped the work",
    "near_miss_meter": "no contact; the problem was found at the next check",
    "near_miss_caught": "no contact; the meter check caught the charge before anything was touched",
    "minor_shock": "brief shock to the fingers, no burn",
    "shock_burn": "shock with a small burn on the hand; seen at occupational health",
    "arc_flash": "arc between the tool and the terminal; no injury, scorch marks on the tool",
}


class HighVoltageLockoutBreach(Schema):
    name = "high_voltage_lockout_breach"
    description = ("Work on high-voltage research equipment where the lockout and grounding rule was or may have been "
                   "breached: energy hazard, whether the procedure was followed, the main gap, and the review outcome.")
    surfaces = {"json": 0.5, "near_miss_form": 0.25, "committee_minute": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        e_rule = pick(rng, [5, 10, 10, 20, 50])
        if reg == "borderline":
            cands = [q for q in HV_EQUIP if 0.5 * (q[2] * 1e-6) * (q[1] * 1e3) ** 2 >= e_rule * 1.1
                     and 0.3 * 0.5 * (q[2] * 1e-6) * (q[1] * 1e3) ** 2 <= e_rule * 0.9]
            eq = pick(rng, cands or HV_EQUIP)
        else:
            eq = pick(rng, HV_EQUIP)
        name, kv_rated, c_uf, current_limited = eq
        c = c_uf * 1e-6
        e_rated = 0.5 * c * (kv_rated * 1e3) ** 2
        if reg == "borderline" and e_rated >= e_rule * 1.1:
            e_target = min(e_rated, max(0.3 * e_rated, near(rng, e_rule, e_rule * 0.35)))
            kv = math.sqrt(2 * e_target / c) / 1e3
        else:
            kv = kv_rated * rng.uniform(0.6, 1.0)
        kv = round(min(kv_rated, kv), 2 if kv_rated < 5 else 1)
        energy = float(f"{0.5 * c * (kv * 1e3) ** 2:.3g}")
        if energy >= 100:
            energy = int(energy)
        gap = pick_w(rng, {"training_gap": 2, "procedure_gap": 1.5, "supervision_pressure": 2,
                           "individual_shortcut": 3.5, "equipment_design": 4})
        flags = [gap]
        if reg == "conflicting":
            flags.append(pick(rng, [g for g in ("individual_shortcut", "equipment_design", "supervision_pressure",
                                                "training_gap") if g != gap]))
        # steps performed
        isolated, lock, test, ground, second = True, True, True, True, True
        bypass = "none"
        all_done = reg == "conflicting" and gap in ("training_gap", "supervision_pressure") and chance(rng, 0.5)
        if (gap != "equipment_design" or len(flags) > 1) and not all_done:
            if reg == "borderline":
                skip = pick_w(rng, {"lock": 0.3, "test": 0.3, "ground": 5, "second": 3})
                n_skip = 1
            else:
                skip = pick_w(rng, {"isolate": 1, "lock": 2, "test": 2, "ground": 3, "second": 2, "bypass": 2})
                n_skip = pick(rng, [1, 2, 2, 3])
            opts = ["lock", "test", "ground", "second"]
            skips = [skip] + rng.sample([o for o in opts if o != skip], max(0, n_skip - 1))
            if "isolate" in skips:
                isolated, test = False, False
            if "lock" in skips:
                lock = False
            if "test" in skips:
                test = False
            if "ground" in skips:
                ground = False
            if "second" in skips:
                second = False
            if "bypass" in skips:
                bypass = pick(rng, ["no_permit", "expired_permit"])
        if "procedure_gap" in flags:
            ground = False
        if gap == "supervision_pressure" and chance(rng, 0.5):
            bypass = "no_permit"
        if gap != "equipment_design" and chance(rng, 0.15) and bypass == "none":
            bypass = "permit"
        if reg == "clear" and gap == "equipment_design":
            isolated, lock, test, ground, second, bypass = True, True, True, True, True, "none"
        followed = (isolated and lock and test and bypass in ("none", "permit")
                    and (energy <= e_rule or (ground and second)))
        # what was found on contact
        if not isolated:
            contact = pick_w(rng, {"live": 3, "unknown": 1})
        elif not test:
            contact = pick_w(rng, {"residual": 2, "unknown": 2})
        elif "equipment_design" in flags:
            contact = pick_w(rng, {"caught": 3, "residual": 2})
        elif not ground:
            contact = pick_w(rng, {"verified_zero": 3, "residual": 2})   # charge recovers on ungrounded capacitors
        else:
            contact = "verified_zero"
        v_res = None
        if contact == "residual":
            v_res = round(kv * 1e3 * (rng.uniform(0.02, 0.1) if test else rng.uniform(0.1, 0.4)))
        elif contact == "caught":
            v_res = round(kv * 1e3 * rng.uniform(0.3, 0.9))
        if contact == "caught":
            outcome = "near_miss_caught"
        elif contact == "verified_zero":
            outcome = pick(rng, ["near_miss", "near_miss_meter"])
        elif contact == "unknown":
            outcome = pick(rng, ["near_miss", "arc_flash", "minor_shock"])
        else:
            outcome = pick(rng, ["minor_shock", "shock_burn", "arc_flash", "near_miss"])
        widespread = pick_w(rng, {"individual": 5, "group": 3, "institution": 1.5})
        if gap == "equipment_design":
            widespread = pick_w(rng, {"individual": 6, "group": 2})
        return {
            "regime": reg, "equipment": name, "kv": kv, "capacitance_uf": c_uf, "stored_energy_j": energy,
            "current_limited": current_limited, "energy_rule_j": e_rule, "gap": gap, "gap_flags": flags,
            "isolated": isolated, "personal_lock": lock, "absence_test": True if test else
            (None if reg == "conflicting" and chance(rng, 0.4) else False),
            "grounding_hook": ground, "second_person": second, "bypass": bypass, "procedure_followed": followed,
            "contact_state": contact, "residual_v": v_res, "event_outcome": outcome, "practice_spread": widespread,
            "gap_detail": {g: pick(rng, GAP_EVIDENCE[g]).format(m=rng.randint(3, 30), y=rng.randint(2009, 2018),
                                                                slot=pick(rng, ["18:00", "overnight", "Monday"]))
                           for g in flags},
            "spread_detail": {"individual": "the same supply model is used only on this bench",
                              "group": "two more supplies of the same model are in use in the group"}[widespread]
            if gap == "equipment_design" else
            {"individual": "no sign that others do the same",
                              "group": pick(rng, ["two others in the group say they skip the same step",
                                                  "the group's shared checklist omits the step"]),
                              "institution": pick(rng, ["the same local procedure template is used by several "
                                                        "departments", "the institution's HV refresher course skips "
                                                                       "this step"])}[widespread],
            "equipment_note": (("grounding hook's cable was broken inside its sleeve, so the hook did not discharge "
                                "the bank") if contact == "residual" and ground and test else
                               pick(rng, GAP_EVIDENCE["equipment_design"])),
            "worker": person(rng), "supervisor": person(rng), "chair": person(rng), "site": _site(rng, "hv"),
            "room": _room(rng), "rule_ref": _rule(rng, HV_BOOKS), "report_id": ident(rng, "HV", 4),
            "permit_id": ident(rng, "BYP", 3), "when": rand_ts(rng, hours=(7, 23)),
            "task": pick(rng, ["replacing a failed HV cable", "swapping a blown fuse in the supply",
                               "re-terminating a capacitor lead", "fitting a new divider board",
                               "cleaning arcing marks off an insulator", "measuring the output ripple",
                               "replacing a cooling fan inside the HV enclosure"]),
        }

    def intended(self, f):
        cs = f["contact_state"]
        if cs in ("verified_zero", "caught"):
            hz = 0
        else:
            e = f["stored_energy_j"] if cs in ("live", "unknown") else 0.5 * f["capacitance_uf"] * 1e-6 * f["residual_v"] ** 2
            if e < 0.25:
                hz = 1
            elif e <= 50:
                hz = 2
            else:
                hz = 3
        flags = f["gap_flags"]
        prim = next(g for g in ("training_gap", "procedure_gap", "supervision_pressure", "individual_shortcut",
                                "equipment_design") if g in flags)
        followed = f["procedure_followed"] and f["absence_test"] is not None
        contact = f["event_outcome"] in ("minor_shock", "shock_burn", "arc_flash")
        if (hz == 3 and contact) or (f["bypass"] in ("no_permit", "expired_permit") and hz >= 2) \
                or f["event_outcome"] == "shock_burn" and hz >= 2:
            out = "committee_investigation"
        elif prim in ("procedure_gap", "equipment_design", "supervision_pressure") \
                or f["practice_spread"] == "institution":
            out = "group_stand_down"
        elif prim == "training_gap" or (hz >= 1 and not followed) or f["practice_spread"] == "group":
            out = "retrain_and_reauthorise"
        else:
            out = "coaching_and_close"
        if f["practice_spread"] == "institution":
            scope = 3
        elif prim == "equipment_design" and followed:
            scope = 0
        elif prim in ("procedure_gap", "supervision_pressure") or f["practice_spread"] == "group":
            scope = 2
        else:
            scope = 1
        return {"energy_hazard": hz, "lockout_procedure_followed": "yes" if followed else "no", "main_gap": prim,
                "review_outcome": out, "retraining_reach": scope}

    def _rule_text(self, f):
        return (f"(a) Any work inside an HV enclosure: isolate, apply your own lock and tag, and test for absence of "
                f"voltage with a rated meter. (b) Where stored energy can exceed {f['energy_rule_j']} J: also discharge "
                f"through the dump resistor, apply a grounding hook to every capacitor terminal before contact, and "
                f"have a second qualified person present. (c) Interlocks may be bypassed only under a written permit "
                f"signed by the electrical safety coordinator for the named person and date.")

    def _steps(self, f):
        t = f["absence_test"]
        return [("Supply isolated", "yes" if f["isolated"] else "no, the wrong isolator was switched off"),
                ("Personal lock applied", "yes" if f["personal_lock"] else "tag only, no lock"),
                ("Absence-of-voltage test", "not recorded" if t is None else _yn(t)),
                ("Grounding hook applied", _yn(f["grounding_hook"])),
                ("Second qualified person present", _yn(f["second_person"])),
                ("Interlock bypass", {"none": "none", "permit": f"yes, under permit {f['permit_id']} (valid for this "
                                                                  f"person and date)",
                                      "no_permit": "yes, no permit raised",
                                      "expired_permit": f"yes, permit {f['permit_id']} expired the week before"}[f["bypass"]])]

    def _contact_text(self, f):
        cs = f["contact_state"]
        if cs == "verified_zero":
            return "meter read 0 V on all terminals before contact"
        if cs == "caught":
            return (f"absence-of-voltage test found {f['residual_v']} V still on the terminals; the worker stopped and "
                    f"discharged it with the grounding hook before touching anything")
        if cs == "residual":
            return f"{f['residual_v']} V found on the capacitor terminals afterwards"
        if cs == "live":
            return "supply output was still energised when the work started"
        return "terminal voltage at the time is not known; no reading was taken"

    def _energy_text(self, f, rng):
        e = f["stored_energy_j"]
        lim = "current-limited supply" if f["current_limited"] else "not current-limited"
        return f"{f['kv']:g} kV, {f['capacitance_uf']:g} uF ({e:g} J stored at that voltage; {lim})"

    def render(self, f, rng, surface):
        steps = self._steps(f)
        details = [f["equipment_note"] if g == "equipment_design" else v for g, v in f["gap_detail"].items()]
        if surface == "json":
            return dumps(prune({
                "report": {"id": f["report_id"], "site": f["site"], "room": f["room"],
                           "time": fmt_ts(f["when"], style="iso"), "worker": f["worker"], "task": f["task"]},
                "equipment": {"name": f["equipment"], "operating_kv": f["kv"], "capacitance_uf": f["capacitance_uf"],
                              "stored_energy_j": f["stored_energy_j"], "current_limited": f["current_limited"]},
                "rule": {"reference": f["rule_ref"], "text": self._rule_text(f)},
                "lockout_record": {k.lower().replace(" ", "_").replace("-", "_"): v for k, v in steps},
                "what_happened": {"contact": self._contact_text(f), "result": OUTCOMES[f["event_outcome"]]},
                "investigation": {"findings": details, "same_practice_elsewhere": f["spread_detail"]},
            }))
        if surface == "near_miss_form":
            head = kv_lines([("Report", f["report_id"]), ("Site / room", f"{f['site']} / {f['room']}"),
                             ("When", fmt_ts(f["when"], rng)), ("Worker", f["worker"]), ("Task", f["task"]),
                             ("Equipment", f"{f['equipment']}: {self._energy_text(f, rng)}")], rng)
            return "\n".join([f"HV INCIDENT / NEAR-MISS REPORT", head, "", "Lockout record:",
                              kv_lines(steps, rng, style="dash"), "",
                              f"What happened: {self._contact_text(f)}; {OUTCOMES[f['event_outcome']]}.",
                              "Investigator notes:", bullets(details + [f["spread_detail"]], rng), "",
                              f"{f['rule_ref']}: {self._rule_text(f)}"])
        # committee_minute
        undone = [k.lower() for k, v in steps
                  if v == "no" or v.startswith("no,") or v in ("tag only, no lock", "not recorded")]
        byp = dict(steps)["Interlock bypass"]
        para = [f"Item {rng.randint(3, 9)}: {f['report_id']}, {f['site']}.",
                f"{f['worker']} was {f['task']} on the {f['equipment']} ({self._energy_text(f, rng)}).",
                ("The lockout record shows these steps not done or not recorded: " + ", ".join(undone) + "."
                 if undone else "The lockout record shows every listed step done."),
                f"Interlock bypass: {byp}.",
                f"Contact: {self._contact_text(f)}. Result: {OUTCOMES[f['event_outcome']]}.",
                "The investigator reports that " + "; and ".join(details) + ".",
                f"On wider practice: {f['spread_detail']}."]
        if f["supervisor"] and chance(rng, 0.5):
            para.append(f"{f['supervisor']} (group leader) told the committee the worker is experienced and careful.")
        return (f"MINUTES -- Electrical Safety Committee, chaired by {f['chair']}\n\n"
                + noisy(join_sentences(para), rng, 0.08) + f"\n\nRule under review ({f['rule_ref']}): "
                + self._rule_text(f))

    def questions(self):
        return {
            "energy_hazard": score("How severe was the electrical hazard the worker was exposed to?", [
                "Minimal: the parts touched were verified discharged, so no harmful energy was present.",
                "Low: a small charge or a current-limited supply that could sting or startle but not injure badly.",
                "Serious: enough stored energy to burn or deliver a dangerous shock (up to about 50 J).",
                "Potentially lethal: a charged circuit holding more than about 50 J, or a supply of that size live.",
            ]),
            "lockout_procedure_followed": noul(
                "The worker completed every lockout and grounding step the written rule requires for this equipment.",
                yes="Isolation, personal lock, a recorded absence-of-voltage test and, above the stated energy limit, "
                    "grounding and a second person were all in place, and any bypass was under a valid permit.",
                no="At least one step the rule requires at this energy level was missing or unrecorded, or an "
                   "interlock was bypassed without a valid permit."),
            "main_gap": choice("What does the evidence show as the main reason this incident happened?", {
                "equipment_design": "The equipment lacked a safeguard it should have had, such as a working bleeder "
                                    "resistor, dump relay or interlock.",
                "individual_shortcut": "A trained worker knowingly skipped required steps, for example to save time.",
                "procedure_gap": "The local written procedure was out of date or left out a step the institutional "
                                 "rule requires.",
                "supervision_pressure": "A supervisor or schedule pushed the work ahead without the required permit "
                                        "or planning.",
                "training_gap": "The worker had not been trained or qualified for high-voltage work on this "
                                "equipment.",
            }),
            "review_outcome": choice("How should the institution respond to this incident?", {
                "coaching_and_close": "Coach the worker, record the event and close it; no wider action is needed.",
                "committee_investigation": "Refer it to the institutional safety committee for a formal investigation.",
                "group_stand_down": "Pause high-voltage work in the group until the procedure, equipment or "
                                    "supervision problem is fixed.",
                "retrain_and_reauthorise": "Withdraw the worker's high-voltage authorisation until retraining and a "
                                           "supervised check are complete.",
            }),
            "retraining_reach": score("How widely should corrective training reach after this incident?", [
                "No one: the procedure was followed and the cause lies in the equipment.",
                "The individual worker involved.",
                "The whole group or team that works on this equipment.",
                "Every high-voltage user across the institution.",
            ]),
        }


# =========================================================================== schema 5
# name: (half-life text, half-life hours, emission, activity range MBq, removable limits dpm/100 cm2,
#        meter can see fixed contamination, volatile)
SPILL_ISOTOPES = {
    "P-32": ("14.3 d", 343, "high-energy beta", (2, 150), [1000, 2000], True, False),
    "S-35": ("87.4 d", 2098, "low-energy beta", (5, 200), [1000, 2000], False, False),
    "H-3": ("12.3 y", 107800, "very low-energy beta", (20, 800), [10000, 20000], False, False),
    "C-14": ("5,730 y", 5.0e7, "low-energy beta", (1, 60), [1000, 2000], False, False),
    "I-125": ("59.4 d", 1426, "low-energy photons", (1, 40), [100, 200], True, True),
    "I-131": ("8.0 d", 192, "beta and gamma", (5, 300), [100, 200], True, True),
    "F-18": ("110 min", 1.83, "positron / 511 keV gamma", (100, 3000), [1000, 2000], True, False),
    "Tc-99m": ("6.0 h", 6.0, "140 keV gamma", (50, 1500), [1000, 2000], True, False),
    "Cr-51": ("27.7 d", 665, "320 keV gamma", (2, 80), [1000, 2000], True, False),
    "Ga-68": ("68 min", 1.13, "positron / 511 keV gamma", (100, 1000), [1000, 2000], True, False),
    "Ca-45": ("163 d", 3912, "medium-energy beta", (1, 40), [1000, 2000], True, False),
}
SPILL_SPOTS = ["floor below the bench", "fridge door handle", "sink rim", "pipette barrel", "waste bin lid",
               "keyboard", "chair seat", "drawer handle", "water bath lid", "shared timer", "vortex mixer",
               "glove dispenser by the door", "label printer", "ice bucket"]
CENTRE_SPOTS = {"tray": ["drip-tray liner", "bench top beside the tray"],
                "fume_hood": ["fume-hood work surface", "fume-hood sash handle"],
                "open_bench": ["bench top at the spill centre", "bench edge by the rack"],
                "floor": ["floor at the spill centre", "floor by the bench leg"],
                "centrifuge": ["centrifuge bowl", "centrifuge lid seal"]}
NICE_MBQ = [1, 2, 5, 10, 20, 25, 37, 50, 74, 100, 185, 200, 370, 500, 740, 1000, 1850, 2000, 3700, 5000]
OUTSIDE_SPOTS = ["corridor floor outside the door", "office door handle", "lift button panel", "write-up desk"]
SPILL_HOW = {
    "tray": "a tube cracked inside the absorbent drip tray",
    "fume_hood": "a vial tipped over inside the fume hood",
    "open_bench": "a tube slipped from a rack onto the open bench",
    "floor": "a stock vial was dropped and broke on the floor",
    "centrifuge": "a tube leaked in the centrifuge and spread when the lid was opened",
}


class ContaminationSpillSurvey(Schema):
    name = "contamination_spill_survey"
    description = ("An open-source radioactive spill and its decontamination surveys: can the area be released, how "
                   "far the contamination reaches, personnel follow-up and regulator notification.")
    surfaces = {"json": 0.45, "survey_record": 0.3, "elog_thread": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        iso = pick(rng, list(SPILL_ISOTOPES))
        hl_text, hl_h, emis, (alo, ahi), limits, fixed_ok, volatile = SPILL_ISOTOPES[iso]
        limit = pick(rng, limits)
        activity = round(math.exp(rng.uniform(math.log(alo), math.log(ahi))), 1)
        loc = pick_w(rng, {"tray": 2, "fume_hood": 2, "open_bench": 3, "floor": 2, "centrifuge": 1.5})
        method_invalid = False
        if reg == "clear":
            mode = pick(rng, ["clean", "bad"])
            r = rng.uniform(0.08, 0.6) if mode == "clean" else rng.uniform(15, 80)
            outside = mode == "bad" and chance(rng, 0.5)
        elif reg == "borderline":
            r = near(rng, 1.0, 0.4, lo=0.3)
            outside = chance(rng, 0.1)
        else:
            r = pick(rng, [rng.uniform(0.2, 0.9), rng.uniform(1.5, 8)])
            outside = chance(rng, 0.25)
            if emis in ("low-energy beta", "very low-energy beta") and chance(rng, 0.5):
                method_invalid = True
        if emis in ("low-energy beta", "very low-energy beta") and reg == "borderline" and chance(rng, 0.2):
            method_invalid = True
        init_r = r * rng.uniform(3, 15)
        n_spots = rng.randint(4, 6)
        spots = [pick(rng, CENTRE_SPOTS[loc])] + rng.sample(SPILL_SPOTS, n_spots - 1)
        # post-decon wipe values: the worst at r * limit, others scattered below it
        vals = [r * limit] + [r * limit * rng.uniform(0.05, 0.9) for _ in range(n_spots - 1)]
        if r > 1 and chance(rng, 0.5):
            vals[1] = r * limit * rng.uniform(0.6, 0.95)
        wipes = [[s, int(round(v, -1)) if v >= 50 else "<50"] for s, v in zip(spots, vals)]
        if outside:
            wipes.append([pick(rng, OUTSIDE_SPOTS), int(round(limit * rng.uniform(1.2, 6), -1))])
        fixed_limit = pick(rng, [100, 200, 300])
        fixed = None
        if fixed_ok:
            fixed = int(near(rng, fixed_limit, fixed_limit * 0.6, lo=10)) if reg != "clear" else (
                int(fixed_limit * rng.uniform(0.1, 0.6)) if r < 1 else int(fixed_limit * rng.uniform(2, 8)))
        skin_limit = pick(rng, [100, 200])
        persons = []
        for i in range(pick(rng, [1, 1, 2])):
            if reg == "clear" and r < 1:
                v = "not detected"
            else:
                v = pick_w(rng, {"not detected": 4, "low": 3, "high": 3})
            if v == "low":
                v = int(skin_limit * rng.uniform(0.2, 0.9))
            elif v == "high":
                v = int(skin_limit * rng.uniform(1.1, 6))
            persons.append(v)
        wound = chance(rng, 0.08)
        if reg == "borderline":
            raw = near(rng, activity, activity * 0.4)
            a_report = min(NICE_MBQ, key=lambda v: abs(math.log(v) - math.log(max(0.5, raw))))
        else:
            a_report = pick(rng, [37, 50, 100, 370])
        area_closed = chance(rng, 0.5)
        corridor_secured = outside and chance(rng, 0.5)
        return {
            "regime": reg, "isotope": iso, "half_life": hl_text, "half_life_h": hl_h, "emission": emis,
            "volatile": volatile, "activity_mbq": activity, "location": loc, "limit_dpm": limit,
            "post_ratio": round(r, 3), "initial_ratio": round(init_r, 2), "wipes": wipes, "spread_outside": outside,
            "method_invalid": method_invalid, "fixed_cpm": fixed, "fixed_limit_cpm": fixed_limit,
            "skin_limit_cpm": skin_limit, "persons_post_wash": persons, "wound": wound, "report_mbq": a_report,
            "area_closed": area_closed, "corridor_secured": corridor_secured,
            "hours_since": pick(rng, [1, 2, 4, 6, 20]), "decon_rounds": pick(rng, [1, 2, 2, 3]),
            "site": _site(rng, "rad"), "room": _room(rng), "rule_ref": _rule(rng, RAD_BOOKS), "spill_id": ident(rng, "SP", 4),
            "user": person(rng), "rso": person(rng), "colleague": person(rng), "when": rand_ts(rng, hours=(8, 20)),
        }

    def _intake(self, f):
        return (f["volatile"] and f["location"] != "fume_hood") or f["wound"]

    def intended(self, f):
        r, limit = f["post_ratio"], f["limit_dpm"]
        over = [w for w in f["wipes"] if isinstance(w[1], int) and w[1] > limit]
        fixed_over = f["fixed_cpm"] is not None and f["fixed_cpm"] > f["fixed_limit_cpm"]
        short = f["half_life_h"] <= 24
        skin_over = any(isinstance(p, int) and p > f["skin_limit_cpm"] for p in f["persons_post_wash"])
        intake = self._intake(f)
        person_need = skin_over or intake
        if f["spread_outside"]:
            ext = 3
        elif f["method_invalid"]:
            ext = 1 if f["initial_ratio"] <= 10 else 2
        elif r <= 1 and not fixed_over:
            ext = 0
        elif len(over) <= 2 and r <= 5:
            ext = 1
        else:
            ext = 2
        if f["spread_outside"]:
            rel = "call_in_rso_team"
        elif f["method_invalid"]:
            rel = "decontaminate_and_resurvey"
        elif r <= 1 and not fixed_over:
            rel = "release_now"
        elif short:
            rel = "restrict_for_decay"
        elif r <= 1 and fixed_over:
            rel = "seal_and_label_fixed"
        elif r > 20:
            rel = "call_in_rso_team"
        else:
            rel = "decontaminate_and_resurvey"
        contained = f["location"] in ("tray", "fume_hood")
        reportable = (f["activity_mbq"] > f["report_mbq"] and not contained) or f["spread_outside"] or intake
        if (f["spread_outside"] and not f["corridor_secured"]) or person_need:
            t = 3
        elif rel != "release_now" and not f["area_closed"]:
            t = 2
        elif rel != "release_now":
            t = 1
        else:
            t = 0
        return {"area_release": rel, "contamination_extent": ext, "personnel_follow_up_needed":
                "yes" if person_need else "no", "spill_response_timing": t,
                "regulator_notice_required": "yes" if reportable else "no"}

    def _rule_text(self, f):
        if f["isotope"] == "H-3":   # tritium betas are invisible to survey meters: skin is checked by swab + LSC
            skin = (f"any post-wash skin swab above {10 * f['skin_limit_cpm']} dpm (H-3 is checked by swab and liquid "
                    f"scintillation, not with a meter)")
        else:
            skin = f"any skin reading above {f['skin_limit_cpm']} cpm above background after two washes"
        return (f"Release for unrestricted use only when removable contamination is at or below {f['limit_dpm']} "
                f"dpm/100 cm2 for {f['isotope']} and fixed contamination is at or below {f['fixed_limit_cpm']} cpm above "
                f"background at 1 cm. H-3, C-14 and S-35 wipes must be counted by liquid scintillation. Personnel: "
                f"{skin}, any contaminated "
                f"wound, or a volatile iodine spill outside a fume hood needs RSO follow-up (dose or bioassay). Report "
                f"to the regulator within 24 h if more than {f['report_mbq']} MBq is released outside a tray, fume "
                f"hood or glovebox, contamination is found outside the controlled area, or an intake is suspected.")

    def _persons_text(self, f):
        out = []
        for i, p in enumerate(f["persons_post_wash"]):
            who = f["user"] if i == 0 else f["colleague"]
            side = "right" if i == 0 else "left"
            if p == "not detected":
                out.append(f"{who}: " + ("post-wash hand swabs at background (liquid scintillation)" if f["isotope"] == "H-3"
                                         else "hands and shoes not detected above background after washing"))
            elif f["isotope"] == "H-3":
                out.append(f"{who}: post-wash swab of the {side} hand, {10 * p} dpm (liquid scintillation)")
            else:
                out.append(f"{who}: {p} cpm above background on the {side} hand after two washes")
        if f["wound"]:
            out.append(f"{f['user']} had a small cut under a torn glove on the hand that held the tube")
        return out

    def _count_note(self, f):
        if f["method_invalid"]:
            return "post-decontamination wipes counted on the GM pancake meter in the lab"
        if f["emission"] in ("low-energy beta", "very low-energy beta"):
            return "wipes counted by liquid scintillation"
        return _count_text(f)

    def _status(self, f):
        a = "spill area roped off and posted" if f["area_closed"] else "spill bench still in shared use by the group"
        if f["spread_outside"]:
            a += "; corridor " + ("taped off" if f["corridor_secured"] else "still open to foot traffic")
        return a

    def _wipe_rows(self, f):
        rows = []
        for spot, v in f["wipes"]:
            if f["method_invalid"]:
                rows.append([spot, "not detected (GM)"])
            else:
                rows.append([spot, str(v)])
        return rows

    def render(self, f, rng, surface):
        persons = self._persons_text(f)
        init_max = int(round(f["initial_ratio"] * f["limit_dpm"], -1))
        fixed = (f"{f['fixed_cpm']} cpm above background (worst spot)" if f["fixed_cpm"] is not None
                 else "not measurable with the lab meter for this isotope")
        how = SPILL_HOW[f["location"]]
        if surface == "json":
            return dumps(prune({
                "spill": {"id": f["spill_id"], "site": f["site"], "room": f["room"],
                          "time": fmt_ts(f["when"], style="iso"), "isotope": f["isotope"],
                          "half_life": f["half_life"], "activity_spilled_mbq": f["activity_mbq"],
                          "what_happened": how, "reported_by": f["user"]},
                "surveys": {"initial_max_removable_dpm_100cm2": init_max,
                            "decontamination_rounds": f["decon_rounds"],
                            "post_decon_wipes_dpm_100cm2": {w[0]: ("not detected (GM)" if f["method_invalid"]
                                                                   else w[1]) for w in f["wipes"]},
                            "counting": self._count_note(f), "fixed_contamination": fixed},
                "personnel": persons,
                "area_status": self._status(f),
                "rule": {"reference": f["rule_ref"], "text": self._rule_text(f)},
            }))
        if surface == "survey_record":
            head = kv_lines([("Spill", f["spill_id"]), ("Site / room", f"{f['site']} / {f['room']}"),
                             ("Time", fmt_ts(f["when"], rng)), ("Isotope", f"{f['isotope']} ({f['half_life']}, "
                                                                          f"{f['emission']})"),
                             ("Activity spilled", f"{f['activity_mbq']} MBq"), ("How", how),
                             ("Initial survey max (removable)", f"{init_max} dpm/100 cm2"),
                             ("Decontamination rounds", f["decon_rounds"]), ("Counting", self._count_note(f)),
                             ("Fixed contamination", fixed), ("Area", self._status(f))], rng)
            return "\n".join(["CONTAMINATION SURVEY RECORD", head, "",
                              f"Post-decontamination wipes (dpm/100 cm2, {f['hours_since']} h after the spill):",
                              table(["location", "result"], self._wipe_rows(f), rng), "",
                              "Personnel:", bullets(persons, rng), "", f"{f['rule_ref']}: {self._rule_text(f)}"])
        # elog_thread
        t = f["when"]
        wl = ", ".join(f"{s} {v}" for s, v in self._wipe_rows(f))
        msgs = [(f["user"], f"{f['isotope']} spill in {f['room']}, about {f['activity_mbq']} MBq: {how}. Gloves off, "
                            f"area marked, called RSO."),
                (f["rso"], f"Initial wipes up to {init_max} dpm/100cm2. Doing {f['decon_rounds']} round{'' if f['decon_rounds'] == 1 else 's'} of decon."),
                (f["rso"], f"After decon ({self._count_note(f)}): {wl}."),
                (f["rso"], f"Fixed: {fixed}."),
                (f["rso"], "Personnel survey: " + "; ".join(persons) + "."),
                (f["colleague"], f"Status: {self._status(f)}.")]
        out = []
        for i, (who, text) in enumerate(msgs):
            ts = t + timedelta(minutes=15 + 35 * i)
            out.append(f"[{ts.strftime('%H:%M')}] {who}: {noisy(text, rng, 0.08)}")
        return (f"e-log {f['site']} / radiation work / {fmt_date(t.date(), rng)}\n" + "\n".join(out)
                + f"\n\nPinned rule ({f['rule_ref']}): {self._rule_text(f)}")

    def questions(self):
        return {
            "area_release": choice("What should happen to the spill area now?", {
                "call_in_rso_team": "Bring in the radiation safety team for a full-scale decontamination; this is "
                                    "beyond what the lab should handle itself.",
                "decontaminate_and_resurvey": "Decontaminate further and re-survey with a valid method before "
                                              "release.",
                "release_now": "Release the area for normal use; the valid surveys are within the release limits.",
                "restrict_for_decay": "Close and post the area and let the short-lived activity decay before "
                                      "re-surveying.",
                "seal_and_label_fixed": "Seal and label the remaining fixed contamination, record it on the area "
                                        "map, and release the rest.",
            }),
            "contamination_extent": score("How extensive is the remaining contamination?", [
                "None above the limits: every valid survey is below the stated release levels.",
                "Localised: one or two spots above the limits, inside the spill zone.",
                "Extensive: many spots, or readings far above the limits, within the lab.",
                "Beyond the lab: contamination found in corridors, other rooms or on items taken out.",
            ]),
            "personnel_follow_up_needed": noul(
                "Someone involved needs radiation safety follow-up under the written personnel rule.",
                yes="A post-wash skin reading is above the stated personnel limit, a wound was contaminated, or a "
                    "volatile iodine spill happened outside a fume hood.",
                no="Every post-wash personnel survey is at or below the stated limit and no wound or intake "
                   "condition applies."),
            "spill_response_timing": score("How quickly must the radiation safety office act on this spill?", [
                "At the next routine lab audit: the area is releasable and nobody needs follow-up.",
                "Before the area's next scheduled use, with the area already closed off.",
                "Today, before the end of the shift, because the area is still in use.",
                "Immediately: contamination is spreading beyond the lab or a person still needs follow-up.",
            ]),
            "regulator_notice_required": noul(
                "This spill meets the written criteria for a report to the regulator.",
                yes="At least one stated criterion is met: activity above the quoted level released outside "
                    "containment, contamination outside the controlled area, or a suspected intake.",
                no="None of the stated criteria is met; the spill is recorded and handled locally."),
        }


def _count_text(f):
    """Counting-method wording for isotopes the lab instruments can see."""
    if f["isotope"] in ("I-125",):
        return "wipes counted in the gamma well counter"
    return "wipes counted in the gamma well counter" if f["half_life_h"] < 24 else "wipes counted on the GM pancake"


WORKFLOW = Workflow(
    name="lab_safety_incident_review",
    title="Lab safety incident review",
    description="Radiation, laser, cryogen and high-voltage incidents in research laboratories, judged against the "
                "institution's written safety rules.",
    domain="physics / astronomy / research operations",
    schemas=[DosimeterInvestigation(), LaserEyeExposure(), CryogenOdhEvent(), HighVoltageLockoutBreach(),
             ContaminationSpillSurvey()],
    notes="Physical safety events in research labs (dose, eye exposure, oxygen deficiency, electrical energy, "
          "contamination) against quoted institutional rules; no security, SIEM or access-control content, no "
          "real institutions or regulators.",
)
