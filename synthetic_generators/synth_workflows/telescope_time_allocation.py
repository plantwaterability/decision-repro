"""Telescope time allocation: how a (fictional) observatory ranks, checks, schedules and renews observing time.

Five decisions from an observatory's research-operations cycle: a proposal at the time allocation committee
(TAC), the technical feasibility review of an exposure-time request, a queue observer's call on one observation
block tonight, a target-of-opportunity (ToO) request to interrupt a scheduled programme, and the continuation
review of a multi-semester programme.

Every observatory, telescope, instrument, programme, target designation, person and policy here is invented.
Telescopes carry tree names, instruments glacial-landform names (SERAC, MORAINE, CIRQUE, ...). The rules quoted
in each state (grade bands, airmass limits, overhead rules, dome-closure limits, ToO trigger criteria,
compensation and continuation policies) are written for this generator and VARY between cases, so an answer
needs the stated rule, not a remembered one. Numbers are kept physically plausible: minimum airmass follows
sec|latitude - declination|, semester visibility comes from a small hour-angle model, S/N scales as sqrt(time)
for background-limited exposures, and site seeing statistics follow a log-normal around the site median.

Each schema draws a `regime` first. Borderline cases sit near the written threshold (a mean grade 0.05 from a
band boundary, a minimum airmass of 1.48 against a 1.5 limit, humidity one point under the closure limit);
conflicting cases pair signals that disagree (strong grades with an overdue prior-use report, an exposure-time
calculation run at better seeing than the proposal's own constraint, a rank-A block on its last night in
marginal seeing) or leave a key fact out. Every schema has `intended(f)`: the careful reviewer's answer per
question, used only to check that answers spread across the options; it is never rendered.
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
    near,
    noisy,
    noul,
    org,
    person,
    pick,
    pick_w,
    prune,
    regime,
    score,
    table,
)

# =========================================================================== shared pools
# (name, code, latitude deg, altitude m, median seeing arcsec, photometric share of usable time)
OBSERVATORIES = [
    ("Monte Arvela Observatory", "MAO", -29.6, 2380, 0.82, 0.68),
    ("Sierra Calva Observatory", "SCO", 31.2, 2790, 0.95, 0.62),
    ("Norrbacken Observatory", "NBO", 28.6, 2350, 0.78, 0.64),
    ("Kaimalu Ridge Observatory", "KRO", 19.9, 3650, 0.66, 0.70),
    ("Llano Quemado Observatory", "LQO", -24.4, 2900, 0.71, 0.76),
    ("Ostrander Peak Observatory", "OPO", 32.6, 3180, 0.90, 0.58),
    ("Cerro Vientos Observatory", "CVO", -30.9, 2450, 0.86, 0.66),
    ("Mirador Alto Observatory", "MAA", -33.1, 2100, 1.05, 0.55),
    ("Pic de Sarrat Observatory", "PSO", 42.8, 2860, 1.10, 0.45),
    ("Hallen Mesa Observatory", "HMO", 35.4, 2600, 1.00, 0.57),
    ("Barranco Seco Observatory", "BSO", 28.9, 2420, 0.80, 0.63),
    ("Kestrel Butte Observatory", "KBO", 34.1, 2750, 1.02, 0.60),
    ("Tolvanen Plateau Observatory", "TPO", -22.9, 3050, 0.74, 0.72),
    ("Altiplano Verde Observatory", "AVO", -27.3, 2640, 0.79, 0.70),
]
TELESCOPES = [("Ulmo", 4.0), ("Quillay", 2.5), ("Lenga", 3.6), ("Rauli", 1.8), ("Canelo", 6.5), ("Tepa", 2.2),
              ("Rowan", 4.2), ("Linden", 3.5), ("Larch", 2.4), ("Sorrel", 1.5), ("Coigue", 8.1), ("Aspen", 3.0),
              ("Maqui", 2.0), ("Hawthorn", 5.0)]
INSTRUMENTS = {
    "SERAC": "high-resolution echelle spectrograph",
    "NUNATAK": "optical spectropolarimeter",
    "TARN": "low-resolution long-slit spectrograph",
    "ARETE": "near-infrared spectrograph",
    "MORAINE": "wide-field optical imager",
    "DRUMLIN": "near-infrared imager",
    "FIRN": "high-speed photometer",
    "CIRQUE": "optical integral-field spectrograph",
    "ESKER": "fibre-fed multi-object spectrograph",
}
# (science, instruments, (bright, faint) magnitude range, band, kind)
TOPICS = [
    ("precise radial velocities of transiting sub-Neptune hosts", ["SERAC"], (9.0, 13.0), "V", "spec"),
    ("chemical abundances of metal-poor halo giants", ["SERAC"], (11.0, 14.5), "V", "spec"),
    ("Rossiter-McLaughlin measurements of warm Jupiters", ["SERAC"], (9.5, 12.5), "V", "spec"),
    ("magnetic field mapping of chemically peculiar stars", ["NUNATAK"], (6.5, 10.5), "V", "spec"),
    ("lithium abundances of open-cluster dwarfs", ["SERAC", "ESKER"], (12.0, 15.5), "V", "spec"),
    ("the binary fraction of runaway OB stars", ["SERAC", "TARN"], (8.0, 12.0), "V", "spec"),
    ("line-profile variability in Be-star discs", ["SERAC", "NUNATAK"], (6.0, 10.0), "V", "spec"),
    ("metal pollution in cool white dwarf atmospheres", ["TARN", "SERAC"], (15.0, 17.5), "g", "spec"),
    ("spectroscopic classification of young supernovae", ["TARN"], (17.0, 20.0), "r", "spec"),
    ("redshifts of lensed galaxies behind group-scale lenses", ["TARN", "CIRQUE"], (21.0, 23.5), "r", "spec"),
    ("nebular-phase spectra of stripped-envelope supernovae", ["TARN"], (19.5, 22.0), "r", "spec"),
    ("spectral typing of brown dwarfs in young moving groups", ["ARETE"], (14.0, 17.0), "J", "spec"),
    ("near-infrared spectra of embedded protostars", ["ARETE"], (12.5, 16.0), "K", "spec"),
    ("emission-line abundances of faint planetary nebulae", ["TARN", "CIRQUE"], (15.5, 19.0), "V", "spec"),
    ("reverberation mapping of low-luminosity active nuclei", ["TARN"], (15.0, 17.5), "V", "spec"),
    ("metal absorption systems toward bright quasars", ["SERAC"], (16.0, 18.0), "r", "spec"),
    ("outburst spectroscopy of dwarf novae", ["TARN"], (14.0, 17.5), "V", "spec"),
    ("surface colours of trans-Neptunian objects", ["MORAINE"], (21.5, 23.5), "r", "img"),
    ("time-delay monitoring of lensed quasars", ["MORAINE"], (18.0, 20.5), "r", "img"),
    ("dipper variability in young stellar objects", ["MORAINE", "DRUMLIN"], (13.0, 16.5), "i", "img"),
    ("faint tidal features around nearby spiral galaxies", ["MORAINE"], (24.0, 26.0), "r", "img"),
    ("transit timing variations of hot Jupiters", ["FIRN", "MORAINE"], (10.5, 14.0), "V", "img"),
    ("stellar occultations by small outer Solar System bodies", ["FIRN"], (13.0, 16.5), "V", "img"),
    ("rotational variability of L/T brown dwarfs", ["DRUMLIN"], (14.0, 16.5), "J", "img"),
    ("eclipsing binaries in intermediate-age open clusters", ["MORAINE", "FIRN"], (13.0, 17.0), "V", "img"),
    ("gas kinematics of ram-pressure stripped cluster galaxies", ["CIRQUE"], (16.0, 18.5), "r", "spec"),
    ("resolved stellar populations in compact ellipticals", ["CIRQUE"], (15.0, 17.5), "r", "spec"),
    ("ionised outflows in nearby Seyfert nuclei", ["CIRQUE"], (13.5, 16.0), "V", "spec"),
    ("velocity dispersions of ultra-faint dwarf galaxy members", ["ESKER"], (19.0, 21.5), "g", "spec"),
    ("membership and rotation in globular-cluster outskirts", ["ESKER"], (16.5, 19.5), "V", "spec"),
    ("a redshift survey of an intermediate-redshift galaxy cluster", ["ESKER"], (20.0, 22.0), "r", "spec"),
    ("rotation periods of open-cluster K dwarfs", ["ESKER", "MORAINE"], (14.5, 17.5), "V", "spec"),
    ("pulsation modes of rapidly oscillating Ap stars", ["FIRN", "SERAC"], (8.0, 11.5), "V", "img"),
    ("dust colours of active asteroids", ["MORAINE"], (19.0, 21.5), "r", "img"),
]
TIME_DOMAIN_WORDS = ("variab", "monitor", "timing", "radial velocities", "pulsation", "reverberation", "rotation periods",
                     "Rossiter", "eclipsing", "occultation", "dipper", "outburst")
TARGET_PREFIXES = ["", "", "", "KRS ", "VSC ", "HLX ", "MCQ ", "SDR ", "OGV ", "TPS "]
SKY = ["photometric", "clear", "thin cirrus", "patchy cloud", "thick cloud"]
SKY_REQ = {"photometric": 0, "clear": 1, "thin cloud OK": 2}
MOON_REQ_ILLUM = {"dark": 30, "grey": 70, "any": 100}
MOON_TIME_FRAC = {"dark": 0.30, "grey": 0.60, "any": 1.0}


def _site(rng):
    return pick(rng, OBSERVATORIES)


def _tel(rng):
    return pick(rng, TELESCOPES)


def _topic(rng):
    t = pick(rng, TOPICS)
    return t, pick(rng, t[1])


def _cycle(rng):
    return f"{pick(rng, [2026, 2027, 2028])}{pick(rng, ['A', 'B'])}"


def _semester_span(cycle):
    y = int(cycle[:4])
    return (date(y, 2, 1), date(y, 7, 31)) if cycle[-1] == "A" else (date(y, 8, 1), date(y + 1, 1, 31))


def _deadline(rng, cycle):
    y = int(cycle[:4])
    return date(y - 1, 9, rng.randint(12, 30)) if cycle[-1] == "A" else date(y, 3, rng.randint(10, 31))


def _prop_id(rng, cycle, code):
    n = rng.randint(1, 260)
    return pick(rng, [f"{cycle}-{code}-{n:03d}", f"{code}/{cycle}/{n:03d}", f"{code}{cycle[2:]}-{n:03d}"])


def _jname(rng, ra, dec):
    h, m = int(ra), int((ra - int(ra)) * 60)
    a = abs(dec)
    d, dm = int(a), int((a - int(a)) * 60)
    return f"{pick(rng, TARGET_PREFIXES)}J{h:02d}{m:02d}{'+' if dec >= 0 else '-'}{d:02d}{dm:02d}"


def _ra_str(ra):
    h, m = int(ra), int(round((ra - int(ra)) * 60))
    if m == 60:
        h, m = (h + 1) % 24, 0
    return f"{h:02d}h{m:02d}m"


def _dec_str(dec):
    a = abs(dec)
    d, m = int(a), int(round((a - int(a)) * 60))
    if m == 60:
        d, m = d + 1, 0
    return f"{'+' if dec >= 0 else '-'}{d:02d}d{m:02d}'"


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _seeing_frac(limit, median):
    """Share of usable time with seeing at or below `limit` (log-normal around the site median)."""
    return _phi(math.log(limit / median) / 0.32)


def _ha_limit(lat, dec, xlim):
    """Half-width in hours of the hour-angle range where the target sits at airmass <= xlim."""
    z = math.acos(1.0 / xlim)
    phi, d = math.radians(lat), math.radians(dec)
    c = (math.cos(z) - math.sin(phi) * math.sin(d)) / (math.cos(phi) * math.cos(d))
    if c >= 1:
        return 0.0
    if c <= -1:
        return 12.0
    return math.degrees(math.acos(c)) / 15.0


def _semester_hours(lat, dec, ra, xlim, sem, moon_frac, night_len=9.0):
    """Night hours over a semester with the target below the airmass limit (no weather losses)."""
    ha = _ha_limit(lat, dec, xlim)
    if ha <= 0:
        return 0.0
    centre = 14.7 if sem == "A" else 2.7          # sidereal time at local midnight, mid-semester
    tot = 0.0
    for k in range(37):                           # one sample every 5 nights
        mid = centre - 6.0 + 12.0 * k / 36.0
        hm = ((mid - ra + 12.0) % 24.0) - 12.0
        lo, hi = max(-ha, hm - night_len / 2), min(ha, hm + night_len / 2)
        tot += max(0.0, hi - lo) * 5
    return round(tot * moon_frac, 1)


def _pl(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def _clamp(x, lo=0, hi=3):
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- a small photon-counting ETC
# Source and sky photons from a zero-point flux, the collecting area and the instrument throughput, plus read noise
# per frame. S/N is per resolution element for spectrographs and per frame stack for imagers.
F0_BAND = {"g": 1000.0, "V": 1000.0, "r": 1000.0, "i": 1000.0, "J": 200.0, "K": 50.0}   # photons/s/cm2/A at mag 0
SKY_OPT = {"dark": 21.7, "grey": 20.7, "any": 19.7}                                      # mag/arcsec2 by moon
SKY_NIR = {"J": 16.0, "K": 13.0}                                                          # mag/arcsec2, any moon
# instrument -> (A per resolution element or filter width, end-to-end efficiency, read noise e-, imager)
INSTR_ETC = {"SERAC": (0.1, 0.08, 4.0, False), "NUNATAK": (0.08, 0.04, 4.0, False), "TARN": (5.0, 0.20, 4.0, False),
             "CIRQUE": (2.0, 0.12, 4.0, False), "ESKER": (1.0, 0.10, 4.0, False), "ARETE": (4.0, 0.15, 10.0, False),
             "MORAINE": (1500.0, 0.30, 5.0, True), "DRUMLIN": (1600.0, 0.30, 12.0, True),
             "FIRN": (900.0, 0.25, 3.0, True)}
NIR_INSTRUMENTS = ("ARETE", "DRUMLIN")
SN_LEVELS = {"spec": [10, 15, 20, 30, 50, 80, 100, 150], "img": [20, 50, 100, 200, 300, 500, 1000]}


def _sn_cap(kind, topic):
    """Highest S/N a proposal on this topic would plausibly ask for: faint targets are not observed to S/N 1000."""
    lo = topic[2][0]
    if kind == "spec":
        return 30 if lo >= 19 else 100 if lo >= 15 else 150
    return 100 if lo >= 18 else 1000


def _etc_band(inst, band):
    return band if inst not in NIR_INSTRUMENTS or band in ("J", "K") else "J"


def _etc_snr(mag, band, aperture_m, inst, seeing, moon, t_s, n_frames):
    dl, eff, rn, imager = INSTR_ETC[inst]
    area = math.pi * (aperture_m * 50.0) ** 2 * 0.85
    f0 = F0_BAND.get(band, 1000.0)
    a_ap = 2.3 * seeing ** 2
    src = f0 * 10 ** (-0.4 * mag) * area * dl * eff
    sky = f0 * 10 ** (-0.4 * SKY_NIR.get(band, SKY_OPT[moon])) * a_ap * area * dl * eff
    npix = a_ap / 0.0625 if imager else 3.0 * max(2.0, 2.0 * seeing / 0.25)
    sig = src * t_s
    return sig / math.sqrt(sig + sky * t_s + n_frames * npix * rn ** 2)


def _etc_mag_for(sn, band, aperture_m, inst, seeing, moon, t_s, n_frames):
    """The magnitude at which the ETC gives `sn` (S/N falls monotonically with magnitude)."""
    lo, hi = -2.0, 32.0
    for _ in range(32):
        mid = (lo + hi) / 2
        if _etc_snr(mid, band, aperture_m, inst, seeing, moon, t_s, n_frames) > sn:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# =========================================================================== schema 1
GRADE_SCALES = {
    "low_best": dict(lo=1.0, hi=5.0, width=4.0, step=0.1, tol=0.10,
                     text="1.0 (outstanding) to 5.0 (not competitive); lower is better"),
    "high_best": dict(lo=1.0, hi=10.0, width=9.0, step=0.5, tol=0.25,
                      text="1 (weak) to 10 (exceptional); higher is better"),
}
COMMENTS_GOOD = ["Compelling case with a clear observing strategy.", "Well-motivated sample; the team can clearly deliver.",
                 "Timely, and likely to produce a high-impact result.", "Strong case; target selection is well argued.",
                 "Excellent use of this instrument's unique capability.", "Clear, testable prediction; I rank it near the top."]
COMMENTS_MID = ["Interesting, but the sample looks marginal for the stated goal.",
                "Sound but incremental relative to existing data.",
                "The link between the observations and the science goal is only partly explained.",
                "Solid team; the case would benefit from a sharper hypothesis.",
                "Reasonable request, though similar programmes have run before."]
COMMENTS_POOR = ["The case does not explain why this telescope is needed.", "Unclear what new result the data would give.",
                 "Overlaps heavily with published work.", "The analysis plan is too vague to judge.",
                 "Sample selection is not justified."]
INFEASIBLE_REASONS = [
    "{k} of the {n} targets never reach the call's airmass limit from this site",
    "the requested {inst} mode is not offered this cycle",
    "the faintest targets need roughly three times the requested time, more than the dark time available",
    "the requested time resolution is beyond what {inst} can read out",
]
CHAIR_REMARKS = [
    "The primary reviewer spoke strongly in favour.", "One member felt the request looked generous for the stated S/N.",
    "No discussion beyond the grades.", "Discussion focused on whether the sample is large enough.",
    "A member noted the team's earlier data from this telescope are not yet published.",
    "The panel liked the science but asked whether a smaller telescope could do it.",
    "Two members argued the grades understate the case.", "Brief discussion; no objections raised.",
]


class ProposalTacRanking(Schema):
    name = "proposal_tac_ranking"
    description = ("A proposal before the time allocation committee: reviewer grades, technical review, prior-use and "
                   "archive checks against the call's written rules.")
    surfaces = {"json": 0.5, "tac_minutes": 0.25, "grade_sheet": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        site, tel = _site(rng), _tel(rng)
        topic, inst = _topic(rng)
        cycle = _cycle(rng)
        scale = pick(rng, ["low_best", "low_best", "high_best"])
        sc = GRADE_SCALES[scale]
        u = sc["width"] / 4.0
        sgn = 1 if scale == "low_best" else -1
        if scale == "low_best":
            b1 = round(rng.uniform(1.6, 2.0), 2)
            b2 = round(b1 + rng.uniform(0.5, 0.8), 2)
            b3 = round(b2 + rng.uniform(0.5, 0.8), 2)
        else:
            b1 = round(rng.uniform(7.6, 8.4), 1)
            b2 = round(b1 - rng.uniform(1.1, 1.6), 1)
            b3 = round(b2 - rng.uniform(1.0, 1.4), 1)
        cuts = [b1, b2, b3]
        bad = [sgn * c for c in cuts]                 # "badness": lower is better on both scales
        if reg == "clear":
            band = pick(rng, [1, 2, 3, 4], [0.4, 0.25, 0.2, 0.15])
            if band == 1:
                mb = bad[0] - rng.uniform(0.15, 0.5) * u
            elif band == 4:
                mb = bad[2] + rng.uniform(0.15, 0.8) * u
            else:
                lo, hi = bad[band - 2], bad[band - 1]
                mb = rng.uniform(lo + 0.3 * (hi - lo), hi - 0.3 * (hi - lo))
        else:
            mb = sgn * pick(rng, cuts, [0.5, 0.3, 0.2]) + near(rng, 0, 0.12 * u)
        mean_t = sgn * mb
        spread = pick_w(rng, {"clear": {"agree": 6, "divided": 2, "split": 1},
                              "borderline": {"agree": 3, "divided": 4, "split": 2},
                              "conflicting": {"agree": 2, "divided": 3, "split": 4}}[reg])
        n_rev = pick(rng, [3, 4, 4, 5, 5, 6])
        sd = {"agree": rng.uniform(0.06, 0.16), "divided": rng.uniform(0.3, 0.5), "split": rng.uniform(0.06, 0.18)}[spread] * u
        g = [rng.gauss(mean_t, sd) for _ in range(n_rev)]
        if spread == "split":
            g[rng.randrange(n_rev)] = mean_t + pick(rng, [-1, 1]) * rng.uniform(1.3, 2.0) * u
        shift = mean_t - sum(g) / n_rev
        grades = [round(min(sc["hi"], max(sc["lo"], round((x + shift) / sc["step"]) * sc["step"])), 1) for x in g]
        mean = round(sum(grades) / n_rev, 2)

        tech = pick_w(rng, {"feasible": 5, "feasible_trim": 1.3, "feasible_underestimated": 2, "infeasible": 0.8,
                            "not_reviewed": 1})
        prior = pick_w(rng, {"none": 1.6, "on_time": 2.4, "coi_missing": 1.0, "late": 2.0, "missing": 2.2})
        arch = pick_w(rng, {"none": 1.8, "minor": 1.0, "major_shallow": 1.4, "major_dup": 6.0, "major_timedomain": 1.2})
        if reg == "conflicting":
            # strong grades that run into a rule, or weak-looking grades on a squeaky-clean proposal
            twist = pick(rng, ["prior", "archive", "tech", "clean"])
            if twist == "prior":
                prior = pick(rng, ["late", "missing", "coi_missing"])
            elif twist == "archive":
                arch = pick(rng, ["major_dup", "major_dup", "major_shallow", "major_timedomain"])
            elif twist == "tech":
                tech = pick(rng, ["infeasible", "feasible_trim", "not_reviewed"])
            else:
                tech, prior, arch = "feasible", pick(rng, ["none", "on_time"]), "none"
        flexible = chance(rng, 0.45)
        if flexible:
            seeing_max, transp = pick(rng, [1.2, 1.5, 2.0]), "thin cloud OK"
        else:
            seeing_max = pick(rng, [0.5, 0.6, 0.7, 0.8, 1.0, 1.2])
            transp = pick(rng, ["photometric", "clear"]) if seeing_max >= 1.2 else pick(rng, ["photometric", "clear", "thin cloud OK"])
            flexible = False
        n_t = pick(rng, [1, 2, 3, 4, 5, 6, 8, 10, 12])
        time_domain = any(w in topic[0] for w in TIME_DOMAIN_WORDS)
        if arch == "major_timedomain" and not time_domain:
            arch = "major_shallow"               # the new-epochs exception only exists for time-domain science
        if arch == "none":
            k_arch = 0
        elif arch == "minor":
            k_arch = max(1, rng.randint(1, max(1, n_t // 2))) if n_t >= 2 else 0
        else:
            k_arch = rng.randint(n_t // 2 + 1, n_t)
        if arch == "minor" and k_arch == 0:
            arch = "none"
        depth = (None if arch == "none" else round(rng.uniform(0.25, 0.6), 2) if arch == "major_shallow"
                 else round(rng.uniform(0.85, 1.35), 2))
        dl = _deadline(rng, cycle)
        if prior == "on_time":
            rep = dl - timedelta(days=pick(rng, [rng.randint(1, 6), rng.randint(20, 240)]))
        elif prior == "late":
            rep = dl + timedelta(days=pick(rng, [rng.randint(1, 4), rng.randint(8, 45)]))
        else:
            rep = None
        thesis = chance(rng, 0.3)
        return {
            "regime": reg, "observatory": site[0], "obs_code": site[1], "site_median_seeing": site[4],
            "telescope": tel[0], "aperture_m": tel[1], "instrument": inst, "science": topic[0], "cycle": cycle,
            "proposal_id": _prop_id(rng, cycle, site[1]), "pi": person(rng), "institution": org(rng, "school"),
            "scale": scale, "cuts": cuts, "grades": grades, "mean_grade": mean, "spread_kind": spread,
            "comments": [self._comment(rng, x, scale, cuts) for x in grades],
            "tech_review": tech, "infeasible_reason": pick(rng, INFEASIBLE_REASONS),
            "trim_pct": rng.randint(20, 45), "instrument_status": pick_w(rng, {"offered": 8, "shared_risk": 2}),
            "hours_requested": max(2, min(60, int(round(n_t * pick(rng, [0.4, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0]))))),
            "seeing_max": seeing_max, "transparency": transp, "moon": pick(rng, ["dark", "grey", "any"]),
            "flexible_conditions": flexible, "n_targets": n_t, "time_domain": time_domain,
            "prior_status": prior, "prior_rule": pick(rng, ["not_considered", "downgrade"], [0.3, 0.7]),
            "prior_programme": _prop_id(rng, f"{int(cycle[:4]) - 1}{pick(rng, ['A', 'B'])}", site[1]),
            "prior_hours": pick(rng, [4, 6, 8, 10, 12, 18, 24]), "coi_name": person(rng),
            "proposal_deadline": dl, "report_received": rep,
            "archive_case": arch, "archive_targets": k_arch, "archive_depth_ratio": depth,
            "archive_rule": pick(rng, ["returned", "trim"], [0.3, 0.7]),
            "thesis": thesis, "thesis_rule": chance(rng, 0.6), "student": person(rng),
            "chair_remark": pick(rng, CHAIR_REMARKS),
            "oversubscription": round(rng.uniform(1.6, 5.5), 1),
        }

    @staticmethod
    def _comment(rng, x, scale, cuts):
        good = x <= cuts[0] if scale == "low_best" else x >= cuts[0]
        poor = x > cuts[2] if scale == "low_best" else x < cuts[2]
        if chance(rng, 0.15):
            return None
        return pick(rng, COMMENTS_GOOD if good else COMMENTS_POOR if poor else COMMENTS_MID)

    # ------------------------------------------------------------------ rule logic
    def _dup(self, f):
        return f["archive_case"] == "major_dup"

    def _band(self, f):
        m, c = f["mean_grade"], f["cuts"]
        band = 1 + (sum(m > x for x in c) if f["scale"] == "low_best" else sum(m < x for x in c))
        if f["thesis"] and f["thesis_rule"] and band > 1:
            if abs(m - c[band - 2]) <= GRADE_SCALES[f["scale"]]["tol"] + 1e-9:
                band -= 1
        if f["prior_status"] in ("late", "missing") and f["prior_rule"] == "downgrade":
            band = min(4, band + 1)
        return band

    def intended(self, f):
        a = {}
        noncompliant = f["prior_status"] in ("late", "missing")
        dup = self._dup(f)
        band = self._band(f)
        if f["tech_review"] == "infeasible" or (noncompliant and f["prior_rule"] == "not_considered") or \
                (dup and f["archive_rule"] == "returned"):
            a["allocation_band"] = "not_eligible"
        elif band == 1:
            trimmed = f["tech_review"] == "feasible_trim" or f["archive_case"] == "minor" or dup
            a["allocation_band"] = "reduced_or_band_two" if trimmed else "band_one_full"
        elif band == 2:
            a["allocation_band"] = "reduced_or_band_two"
        elif band == 3:
            a["allocation_band"] = "poor_weather_queue" if f["flexible_conditions"] else "no_time_resubmit"
        else:
            a["allocation_band"] = "no_time_resubmit"
        g = f["grades"]
        r = (max(g) - min(g)) / GRADE_SCALES[f["scale"]]["width"]
        a["merit_agreement"] = 0 if r >= 0.36 else 1 if r >= 0.2 else 2 if r >= 0.09 else 3
        fc = {"feasible": 3, "feasible_trim": 3, "feasible_underestimated": 2, "not_reviewed": 2, "infeasible": 0}[f["tech_review"]]
        if f["tech_review"] != "infeasible":
            if f["seeing_max"] <= 0.75 * f["site_median_seeing"]:
                fc -= 1
            if f["instrument_status"] == "shared_risk":
                fc -= 1
        a["feasibility_confidence"] = _clamp(fc)
        a["prior_use_compliant"] = "no" if noncompliant else "yes"
        a["archive_duplication"] = "yes" if dup else "no"
        return a

    # ------------------------------------------------------------------ rendering
    def _rules(self, f, rng):
        c, sc = f["cuts"], GRADE_SCALES[f["scale"]]
        op = "<=" if f["scale"] == "low_best" else ">="
        fmt = "{:.2f}" if f["scale"] == "low_best" else "{:.1f}"
        bands = (f"Band 1 (priority) mean {op} {fmt.format(c[0])}; Band 2 (as time allows) {op} {fmt.format(c[1])}; "
                 f"Band 3 (poor-weather queue) {op} {fmt.format(c[2])}; otherwise no time.")
        filler = "Band 3 programmes must accept seeing of 1.2\" or worse AND thin cloud, otherwise they are not scheduled."
        prior = (f"A PI of a programme in the last four semesters must file its report before the proposal deadline "
                 f"({fmt_date(f['proposal_deadline'], style='iso')}), or the proposal is "
                 + ("not considered" if f["prior_rule"] == "not_considered" else "moved down one band")
                 + " (applies to the PI's own programmes only).")
        archive = ("Targets with same-mode public archive data at the requested depth are removed; "
                   + ("if that removes over half the targets, no time is given."
                      if f["archive_rule"] == "returned" else "hours are cut to match.")
                   + " New epochs are allowed for time-domain science.")
        out = {"grading": bands, "poor_weather": filler, "prior_use": prior, "archive": archive}
        if f["thesis_rule"]:
            out["thesis"] = f"Thesis proposals within {sc['tol']:.2f} of a band boundary go in the higher band."
        return out

    def _tech_text(self, f):
        t = f["tech_review"]
        if t == "feasible":
            return "Feasible as proposed; exposure-time inputs and overheads checked."
        if t == "feasible_trim":
            return (f"Feasible. The requested S/N is reached with about {100 - f['trim_pct']}% of the science time; "
                    f"the request could be cut by {f['trim_pct']}%.")
        if t == "feasible_underestimated":
            return ("Feasible, but acquisition overheads were left out; the corrected total is about "
                    f"{round(f['hours_requested'] * 1.25, 1)} h.")
        if t == "infeasible":
            k = max(1, f["n_targets"] // 2)
            return "Not feasible: " + f["infeasible_reason"].format(k=k, n=max(f["n_targets"], k), inst=f["instrument"]) + "."
        return "Technical review not completed before the TAC meeting."

    def _prior_text(self, f):
        p = f["prior_status"]
        if p == "none":
            return "No programme as PI in the previous four semesters."
        if p == "coi_missing":
            return (f"Applicant was co-investigator on {f['prior_programme']} (PI {f['coi_name']}), whose report is "
                    "outstanding. No earlier programme as PI.")
        head = f"PI of {f['prior_programme']} ({f['prior_hours']} h). "
        if p == "missing":
            return head + "End-of-programme report: not received."
        return head + f"End-of-programme report received {fmt_date(f['report_received'], style='iso')}."

    def _archive_text(self, f):
        if f["archive_case"] == "none":
            return "No public archive data in the same mode for any target."
        k, n = f["archive_targets"], f["n_targets"]
        base_ = (f"{k} of {n} targets {'has' if k == 1 else 'have'} public archive data in the same mode" if n > 1
                 else "The single target has public archive data in the same mode")
        d = f["archive_depth_ratio"]
        depth = (f", reaching about {int(round(d * 100))}% of the requested S/N" if d < 1 else
                 ", at or beyond the requested depth")
        if f["archive_case"] == "major_dup" and f["time_domain"]:
            just = pick_text(f, [" The archive already holds a multi-epoch series at the requested cadence.",
                                 " Archived epochs cover the cadence the proposal asks for; PI does not discuss them."])
        else:
            just = {"major_timedomain": " PI: new epochs are needed to measure the variability the programme is about.",
                    "major_dup": pick_text(f, ["", " PI: the archive data predate a detector upgrade.",
                                               " PI does not discuss the archive data."]),
                    }.get(f["archive_case"], "")
        return base_ + depth + "." + just

    def _conditions(self, f):
        return f"seeing <= {f['seeing_max']}\", {f['transparency']}, {f['moon']} time"

    def render(self, f, rng, surface):
        rules = self._rules(f, rng)
        sc = GRADE_SCALES[f["scale"]]
        thesis = (f"yes (student {f['student']})" if f["thesis"] else None)
        if surface == "json":
            return dumps(prune({
                "proposal": {"id": f["proposal_id"], "cycle": f["cycle"], "pi": f["pi"], "institution": f["institution"],
                             "science": f["science"], "telescope": f"{f['telescope']} {f['aperture_m']} m",
                             "instrument": f"{f['instrument']} ({INSTRUMENTS[f['instrument']]})",
                             "instrument_status": None if f["instrument_status"] == "offered"
                             else "shared-risk (still in commissioning)",
                             "hours_requested": f["hours_requested"], "targets": f["n_targets"],
                             "conditions": self._conditions(f), "thesis": thesis},
                "reviews": {"scale": sc["text"], "grades": f["grades"], "mean": f["mean_grade"],
                            "comments": [c for c in f["comments"] if c] or None},
                "technical_review": self._tech_text(f),
                "checks": {"prior_use": self._prior_text(f), "archive": self._archive_text(f)},
                "site": {"observatory": f["observatory"], "median_seeing_arcsec": f["site_median_seeing"]},
                "call_rules": rules,
            }))
        if surface == "tac_minutes":
            gr = ", ".join(str(x) for x in f["grades"])
            para = [f"{f['proposal_id']} -- {f['pi']} ({f['institution']}): {f['science']}.",
                    f"{f['hours_requested']} h on {f['telescope']}/{f['instrument']}, {_pl(f['n_targets'], 'target')}, "
                    f"{self._conditions(f)}" + (" (instrument is shared-risk this cycle)" if f["instrument_status"] != "offered" else "") + ".",
                    f"Grades ({sc['text']}): {gr}; mean {f['mean_grade']}.",
                    "Technical: " + self._tech_text(f), "Prior use: " + self._prior_text(f),
                    "Archive: " + self._archive_text(f)]
            if f["thesis"]:
                para.append(f"Supports the thesis of {f['student']}.")
            para.append(f["chair_remark"])
            hdr = (f"{f['observatory']} -- TAC {f['cycle']}, minutes (extract). Oversubscription this cycle: "
                   f"{f['oversubscription']}x. Site median seeing {f['site_median_seeing']}\".")
            return hdr + "\n\n" + noisy(" ".join(para), rng, 0.1) + "\n\nRules applied (call for proposals):\n" + \
                bullets(list(rules.values()), rng)
        # grade_sheet
        rows = [[f"R{i + 1}", x, c or "-"] for i, (x, c) in enumerate(zip(f["grades"], f["comments"]))]
        head = kv_lines([("Proposal", f["proposal_id"]), ("PI", f"{f['pi']}, {f['institution']}"),
                         ("Science", f["science"]), ("Instrument", f"{f['instrument']} on the {f['aperture_m']} m {f['telescope']}"
                                                     + (" [shared-risk]" if f["instrument_status"] != "offered" else "")),
                         ("Request", f"{f['hours_requested']} h, {_pl(f['n_targets'], 'target')}, {self._conditions(f)}"),
                         ("Thesis", thesis or "no"), ("Site median seeing", f"{f['site_median_seeing']}\"")], rng)
        return "\n".join([f"GRADE SHEET -- {f['observatory']}, cycle {f['cycle']}", head, "",
                          f"Scale: {sc['text']}", table(["reviewer", "grade", "comment"], rows, rng),
                          f"Mean: {f['mean_grade']}", "",
                          "Technical review: " + self._tech_text(f), "Prior use: " + self._prior_text(f),
                          "Archive: " + self._archive_text(f), "", "Call rules:",
                          bullets(list(rules.values()), rng)])

    def questions(self):
        return {
            "allocation_band": choice("Where should the committee place this proposal under the call's rules?", {
                "band_one_full": "Schedule the full request in the top band, with priority over other programmes.",
                "no_time_resubmit": "Give it no time this cycle, but leave the team free to resubmit an improved version.",
                "not_eligible": "Set it aside without ranking, because it breaks an eligibility rule of the call such as "
                                "an unfiled report, an infeasible plan or a request the archive already covers.",
                "poor_weather_queue": "Accept it only into the poor-weather queue, where it runs when the sky is unfit for "
                                      "ranked programmes.",
                "reduced_or_band_two": "Schedule it, but either in the second band or with the hours trimmed to what is "
                                       "actually justified.",
            }),
            "merit_agreement": score("How far do the reviewers agree about the proposal's scientific merit?", [
                "Split: the grades span a large part of the scale or one reviewer sharply dissents.",
                "Divided: the grades differ by more than a typical band width.",
                "Broadly aligned: small differences that would not change the band on their own.",
                "Unanimous: every grade is essentially the same.",
            ]),
            "feasibility_confidence": score("How confident can the committee be that the programme can be carried out as "
                                            "proposed?", [
                "Not at all: the technical review found a blocking problem.",
                "Doubtful: open technical questions, an instrument still being commissioned or conditions the site "
                "rarely delivers.",
                "Reasonable: workable, with a correction or an unchecked detail.",
                "High: checked and comfortably within what the site and instrument deliver.",
            ]),
            "prior_use_compliant": noul(
                "The applicant has met the reporting obligations the call attaches to earlier allocations.",
                yes="The applicant held no earlier programme as PI, or filed its report before the proposal deadline.",
                no="A report on an earlier programme the applicant led is missing or arrived after the deadline."),
            "archive_duplication": noul(
                "Most of the requested targets already have public archive data that would serve the proposal's purpose."),
        }


def pick_text(f, options):
    """Deterministic choice of a phrasing from the factors (render-time text that must not consume rng)."""
    key = sum(ord(ch) for ch in str(f.get("proposal_id", f.get("programme_id", "")))) % len(options)
    return options[key]


# =========================================================================== schema 2


class ExposureFeasibilityReview(Schema):
    name = "exposure_feasibility_review"
    description = ("Technical review of an observing request: exposure-time calculator output, airmass and semester "
                   "visibility, site statistics and overheads against the call's rules.")
    surfaces = {"json": 0.5, "tech_review_form": 0.3, "etc_printout": 0.2}

    def sample_factors(self, rng):
        reg = regime(rng)
        site, tel = _site(rng), _tel(rng)
        topic, inst = _topic(rng)
        cycle = _cycle(rng)
        lat = site[2]
        xlim = pick(rng, [1.5, 1.6, 1.7, 1.8, 2.0, 2.2])
        zmax = math.degrees(math.acos(1.0 / xlim))
        vis = pick_w(rng, {"clear": {"ample": 4, "never": 3, "edge": 2, "low": 1},
                           "borderline": {"ample": 1, "low": 5, "edge": 4},
                           "conflicting": {"ample": 2, "low": 3, "edge": 3, "never": 1}}[reg])
        if vis == "never":
            zt = zmax + rng.uniform(3, 12)
        elif vis == "low":
            zt = near(rng, zmax - 1.0, 3.0)
        else:
            zt = rng.uniform(2, max(5, zmax - 14))
        cands = [d for d in (lat - zt, lat + zt) if -84 < d < 84]
        dec = round(pick(rng, cands), 2)
        zt = abs(lat - dec)
        min_x = round(1.0 / math.cos(math.radians(zt)), 2) if zt < 85 else 11.5
        sem = cycle[-1]
        centre = 14.7 if sem == "A" else 2.7
        dra = rng.uniform(0, 4.0) if vis in ("ample", "never") else rng.uniform(4.0, 8.0) if vis == "low" \
            else rng.uniform(10.3, 12.0)
        ra = round((centre + pick(rng, [-1, 1]) * dra) % 24, 2)
        moon = pick_w(rng, {"dark": 3, "grey": 3, "any": 2} if vis != "edge" else {"dark": 5, "grey": 3, "any": 1})
        vis_h = _semester_hours(lat, dec, ra, xlim, sem, MOON_TIME_FRAC[moon])
        if vis_h == 0 and min_x <= xlim:
            min_x = round(xlim + 0.01, 2)          # keep the rounded minimum airmass consistent with zero hours

        kind = "img" if INSTR_ETC[inst][3] else "spec"
        sub_opts = [300, 600, 900, 1200, 1800] if kind == "spec" else [30, 60, 120, 180, 300]
        t_sub = pick(rng, sub_opts)
        n_exp = pick(rng, [2, 3, 4, 6]) if kind == "spec" else pick(rng, [6, 10, 15, 20, 30])
        n_blocks = pick(rng, [2, 3, 4, 6, 8, 10, 12, 16, 20, 24])
        sn_req = pick(rng, SN_LEVELS[kind])
        rho = {"clear": pick(rng, [rng.uniform(0.55, 0.72), rng.uniform(0.98, 1.2), rng.uniform(1.45, 1.9)]),
               "borderline": pick(rng, [near(rng, 1.0, 0.12), near(rng, 0.8, 0.08), near(rng, 1.3, 0.1)]),
               "conflicting": rng.uniform(1.05, 1.6)}[reg]
        seeing_max = pick(rng, [0.6, 0.7, 0.8, 1.0, 1.2, 1.5])
        etc_seeing, etc_moon = seeing_max, moon
        if reg == "conflicting" and chance(rng, 0.65):
            etc_seeing = round(seeing_max * rng.uniform(0.5, 0.75), 1)     # ETC run at better seeing than the constraint
        elif chance(rng, 0.12) and moon != "dark":
            etc_moon = "dark"
        transp = pick(rng, ["photometric", "clear", "thin cloud OK"])
        acq_min = pick(rng, [8, 10, 12, 15]) if kind == "spec" else pick(rng, [3, 4, 5, 6])
        readout_s = pick(rng, [20, 30, 45, 60, 90]) if kind == "spec" else pick(rng, [5, 10, 15, 20])
        oh_ratio = {"clear": pick(rng, [rng.uniform(0.98, 1.3), rng.uniform(0.15, 0.55)]),
                    "borderline": near(rng, 0.9, 0.18, lo=0.3),
                    "conflicting": pick(rng, [rng.uniform(0.2, 0.6), rng.uniform(0.95, 1.2)])}[reg]
        if vis in ("edge", "low") and vis_h > 0:
            # the request is sized near what the semester offers: the borderline a technical reviewer checks
            target_h = min(vis_h * rng.uniform(0.65, 2.2), rng.uniform(30, 110))
            max_exp = max(1, int((3.0 * 3600 - acq_min * 60) / (t_sub + readout_s)))   # blocks of at most ~3 h
            n_exp = max(1, min(40, max_exp, int(round((target_h * 3600.0 / n_blocks - acq_min * 60)
                                                       / (t_sub + readout_s)))))
            # requested hours per block as the PI counts them (science plus the overheads the PI claims)
            per_block_req = n_exp * t_sub + (acq_min * 60 + readout_s * n_exp) * oh_ratio
            n_blocks = max(2, min(80, int(round(target_h * 3600.0 / per_block_req))))
        # the ETC: fix the plan and the S/N the regime calls for, invert for the target magnitude, and pick a topic
        # for this instrument whose magnitude range contains it (else another S/N level, another telescope, and
        # last a shorter exposure plan)
        topic, tel, sn_req, mag, n_exp, t_sub, fallback = self._fit_target(
            rng, inst, topic, tel, sn_req, rho, etc_seeing, etc_moon, n_blocks, n_exp, t_sub, kind, readout_s, acq_min)
        req_oh_s = n_blocks * (acq_min * 60 + readout_s * n_exp)
        claimed_oh_h = round(req_oh_s * oh_ratio / 3600.0, 2)
        requested_h = round(n_blocks * n_exp * t_sub / 3600.0 + claimed_oh_h, 1)
        band = _etc_band(inst, topic[3])
        scope = "per block" if self._per_block(topic) else "combined over all blocks"
        t_s, frames = self._exposure(topic, n_blocks, n_exp, t_sub)
        sn_etc = max(2, int(round(_etc_snr(mag, band, tel[1], inst, etc_seeing, etc_moon, t_s, frames))))
        return {
            "regime": reg, "observatory": site[0], "obs_code": site[1], "latitude": lat, "site_median_seeing": site[4],
            "photometric_frac": site[5], "telescope": tel[0], "aperture_m": tel[1], "instrument": inst,
            "science": topic[0], "band": band, "kind": kind, "cycle": cycle, "etc_fallback": fallback,
            "proposal_id": _prop_id(rng, cycle, site[1]), "pi": person(rng), "reviewer": person(rng),
            "target": _jname(rng, ra, dec), "ra_h": ra, "dec_deg": dec, "airmass_limit": xlim, "min_airmass": min_x,
            "visibility_case": vis, "semester_hours_visible": vis_h, "moon": moon, "seeing_max": seeing_max,
            "transparency": transp, "mag": mag, "n_blocks": n_blocks, "n_exp": n_exp, "t_sub_s": t_sub,
            "sn_scope": scope, "sn_required": sn_req, "sn_etc": sn_etc, "etc_seeing": etc_seeing, "etc_moon": etc_moon,
            "acq_min": acq_min, "readout_s": readout_s, "overhead_ratio": round(oh_ratio, 3),
            "overheads_claimed_h": claimed_oh_h, "hours_requested": requested_h,
            "etc_version": f"{pick(rng, ['ETC', 'ExpCalc', 'SNcalc'])} {rng.randint(2, 7)}.{rng.randint(0, 9)}",
            "pi_note": pick(rng, ["", "The ETC numbers are conservative.", "We have observed similar targets before.",
                                  "Timing within the semester is flexible.", "Blocks can be split across nights.", ""]),
        }

    # ------------------------------------------------------------------ physics
    @staticmethod
    def _per_block(topic):
        return any(w in topic[0] for w in TIME_DOMAIN_WORDS)

    def _exposure(self, topic, n_blocks, n_exp, t_sub):
        """(integration seconds, frames) behind the quoted S/N: one block for time-domain work, all blocks otherwise."""
        if self._per_block(topic):
            return n_exp * t_sub, n_exp
        return n_blocks * n_exp * t_sub, n_blocks * n_exp

    def _fit_target(self, rng, inst, topic, tel, sn_req, rho, seeing, moon, n_blocks, n_exp, t_sub, kind,
                    readout_s, acq_min):
        topics = [t for t in TOPICS if inst in t[1]]
        others = [x for x in TELESCOPES if x != tel]
        rng.shuffle(others)
        sns = [x for x in SN_LEVELS[kind] if x != sn_req]

        def scan(tels, levels, plans):
            out = []
            for tl in tels:
                for sr in levels:
                    for ne, ts in plans:
                        for t in topics:
                            if sr > _sn_cap(kind, t):
                                continue
                            t_s, frames = self._exposure(t, n_blocks, ne, ts)
                            m = _etc_mag_for(sr * rho, _etc_band(inst, t[3]), tl[1], inst, seeing, moon, t_s, frames)
                            if t[2][0] - 0.3 <= m <= t[2][1] + 0.3:
                                out.append((t, tl, sr, round(m, 1), ne, ts))
            return out

        plan = [(n_exp, t_sub)]
        subs = [300, 600, 900, 1200, 1800] if kind == "spec" else [10, 30, 60, 120, 180, 300]
        max_block = 3.0 * 3600 - acq_min * 60
        short = [(ne, ts) for ts in subs for ne in (1, 2, 3, 4, 6, 8, 10, 15, 20)
                 if ne * (ts + readout_s) <= max_block and ne * ts < n_exp * t_sub]
        for how, tels, levels, plans in (("none", [tel], [sn_req], plan), ("sn_level", [tel], sns, plan),
                                         ("telescope", others, [sn_req], plan), ("telescope_and_sn", others, sns, plan),
                                         ("shorter_plan", [tel], [sn_req] + sns, short)):
            cands = scan(tels, levels, plans)
            if cands:
                t, tl, sr, m, ne, ts = pick(rng, cands)
                return t, tl, sr, m, ne, ts, how
        # nothing fits: keep the plan and let the physics set the S/N for a magnitude in the topic's own range
        return topic, tel, sn_req, round(rng.uniform(*topic[2]), 1), n_exp, t_sub, "clamped"

    def _derived(self, f):
        rho = f["sn_etc"] / f["sn_required"]
        rho_eff = rho * min(1.0, f["etc_seeing"] / f["seeing_max"])
        if f["etc_moon"] == "dark" and f["moon"] != "dark":
            rho_eff *= 0.8
        req_oh_h = f["n_blocks"] * (f["acq_min"] * 60 + f["readout_s"] * f["n_exp"]) / 3600.0
        science_h = f["n_blocks"] * f["n_exp"] * f["t_sub_s"] / 3600.0
        needed = science_h / max(rho_eff, 0.05) ** 2 + req_oh_h      # background-limited: S/N ~ sqrt(t)
        return rho_eff, req_oh_h, needed

    def _sky_prob(self, f):
        ps = _seeing_frac(f["seeing_max"], f["site_median_seeing"])
        pt = {"photometric": f["photometric_frac"], "clear": 0.85, "thin cloud OK": 1.0}[f["transparency"]]
        return ps, pt, MOON_TIME_FRAC[f["moon"]]

    def intended(self, f):
        rho_eff, req_oh_h, needed = self._derived(f)
        a = {}
        reaches = f["min_airmass"] <= f["airmass_limit"]
        observable = reaches and f["semester_hours_visible"] >= f["hours_requested"]
        if not reaches or f["semester_hours_visible"] < needed:
            a["time_request_verdict"] = "infeasible_as_proposed"
        else:
            r = f["hours_requested"] / needed
            a["time_request_verdict"] = "trim_request" if r > 1.25 else "add_time" if r < 0.8 else "adequate_as_requested"
        a["sn_margin"] = 0 if rho_eff < 0.8 else 1 if rho_eff < 1.0 else 2 if rho_eff < 1.3 else 3
        a["observable_in_semester"] = "yes" if observable else "no"
        ps, pt, pm = self._sky_prob(f)
        p = ps * pt * pm
        a["sky_constraint_cost"] = 0 if p >= 0.4 else 1 if p >= 0.18 else 2 if p >= 0.07 else 3
        a["overheads_included"] = "yes" if f["overheads_claimed_h"] >= 0.9 * req_oh_h else "no"
        return a

    # ------------------------------------------------------------------ rendering
    def _site_stats(self, f):
        ps, pt, pm = self._sky_prob(f)
        out = {f"seeing <= {f['seeing_max']}\"": f"{int(round(ps * 100))}% of usable time"}
        if f["transparency"] != "thin cloud OK":
            out[f"{f['transparency']} sky"] = f"{int(round(pt * 100))}% of usable time"
        if f["moon"] != "any":
            out[f"{f['moon']} time"] = f"{int(round(pm * 100))}% of nights"
        return out

    def _rules_text(self, f):
        return [f"Targets must reach airmass {f['airmass_limit']} or better during the night.",
                f"Requested time = science time + overheads: acquisition {f['acq_min']} min per block plus "
                f"{f['readout_s']} s readout per exposure.",
                "Exposure times must reach the stated S/N under the proposal's own seeing and moon constraints."]

    def _plan(self, f):
        return f"{f['n_blocks']} blocks x {f['n_exp']} x {f['t_sub_s']} s"

    def render(self, f, rng, surface):
        stats = self._site_stats(f)
        mag = f"{f['band']} = {f['mag']}"
        sn_unit = ("per resolution element" if f["kind"] == "spec" else "per frame stack") + ", " + f["sn_scope"]
        if surface == "json":
            return dumps(prune({
                "request": {"proposal": f["proposal_id"], "pi": f["pi"], "science": f["science"],
                            "instrument": f"{f['instrument']} on {f['telescope']} {f['aperture_m']} m",
                            "target": {"name": f["target"], "ra": _ra_str(f["ra_h"]), "dec": _dec_str(f["dec_deg"]),
                                       "magnitude": mag},
                            "exposure_plan": self._plan(f), "overheads_claimed_h": f["overheads_claimed_h"],
                            "hours_requested": f["hours_requested"],
                            "constraints": {"seeing_max_arcsec": f["seeing_max"], "sky": f["transparency"],
                                            "moon": f["moon"]},
                            "pi_note": f["pi_note"] or None},
                "etc": {"tool": f["etc_version"], "assumed_seeing_arcsec": f["etc_seeing"], "assumed_moon": f["etc_moon"],
                        "sn_required": f["sn_required"], "sn_predicted": f["sn_etc"], "sn_quoted": sn_unit},
                "visibility": {"site_latitude_deg": f["latitude"], "minimum_airmass": f["min_airmass"],
                               "semester": f["cycle"],
                               "night_hours_within_airmass_limit": f"{f['semester_hours_visible']} h with the moon "
                                                                   "constraint applied, before weather losses"},
                "site_statistics": stats,
                "call_rules": self._rules_text(f),
            }))
        if surface == "tech_review_form":
            body = kv_lines([
                ("Proposal", f["proposal_id"]), ("PI", f["pi"]), ("Technical reviewer", f["reviewer"]),
                ("Instrument", f"{f['instrument']} ({INSTRUMENTS[f['instrument']]}), {f['telescope']} {f['aperture_m']} m"),
                ("Science", f["science"]),
                ("Target", f"{f['target']} ({_ra_str(f['ra_h'])} {_dec_str(f['dec_deg'])}), {mag}"),
                ("Constraints", f"seeing <= {f['seeing_max']}\", {f['transparency']}, {f['moon']} time"),
                ("Exposure plan", self._plan(f)),
                ("Overheads claimed", f"{f['overheads_claimed_h']} h"),
                ("Total requested", f"{f['hours_requested']} h"),
                ("ETC", f"S/N {f['sn_etc']} {sn_unit} (required {f['sn_required']}); run at seeing {f['etc_seeing']}\", "
                        f"{f['etc_moon']} sky"),
                ("Minimum airmass", f"{f['min_airmass']} (site latitude {f['latitude']:+.1f} deg)"),
                ("Hours within airmass limit", f"{f['semester_hours_visible']} h in {f['cycle']} ({f['moon']} time, before weather)"),
                ("Site statistics", "; ".join(f"{k}: {v}" for k, v in stats.items())),
            ], rng)
            return (f"TECHNICAL FEASIBILITY REVIEW -- {f['observatory']}\n{body}\n\nCall rules:\n"
                    + bullets(self._rules_text(f), rng) + (f"\n\nPI remark: {f['pi_note']}" if f["pi_note"] else ""))
        # etc_printout
        lines = [f"== {f['etc_version']} / {f['instrument']} @ {f['telescope']} {f['aperture_m']} m ({f['obs_code']}) ==",
                 f"target            {f['target']}   {mag}",
                 f"seeing            {f['etc_seeing']} arcsec      sky: {f['etc_moon']}",
                 f"exposure          {self._plan(f)}",
                 f"S/N               {f['sn_etc']} {sn_unit}",
                 "", "== visibility ==",
                 f"site lat {f['latitude']:+.1f}  RA {_ra_str(f['ra_h'])}  Dec {_dec_str(f['dec_deg'])}  "
                 f"min airmass {f['min_airmass']}",
                 f"{f['cycle']}: {f['semester_hours_visible']} h at airmass <= {f['airmass_limit']} ({f['moon']} time, no weather)",
                 "", f"-- proposal {f['proposal_id']} (PI {f['pi']}) --",
                 f"required S/N {f['sn_required']}; constraints: seeing <= {f['seeing_max']}\", {f['transparency']}, "
                 f"{f['moon']}",
                 f"overheads claimed {f['overheads_claimed_h']} h; total requested {f['hours_requested']} h",
                 "site: " + "; ".join(f"{k} {v}" for k, v in stats.items())]
        return "\n".join(lines) + "\n\nCall rules:\n" + bullets(self._rules_text(f), rng)

    def questions(self):
        return {
            "time_request_verdict": choice("What should the technical review say about the hours requested?", {
                "adequate_as_requested": "The request is about right for the stated goals and can stand.",
                "add_time": "The request is too small to reach the stated goals and should be increased.",
                "infeasible_as_proposed": "The programme cannot be done as proposed this semester, whatever the hours.",
                "trim_request": "The request is larger than the stated goals need and should be cut.",
            }),
            "sn_margin": score("How much margin does the exposure plan leave against the required signal-to-noise under "
                               "the proposal's own conditions?", [
                "Short: the S/N achieved under the stated conditions falls clearly below the requirement.",
                "Marginal: just below the requirement, within about a fifth.",
                "Adequate: meets the requirement with a modest margin.",
                "Generous: exceeds the requirement by a wide margin.",
            ]),
            "observable_in_semester": noul(
                "The target can be observed within the call's airmass limit for at least the requested time this semester.",
                yes="It reaches the airmass limit and the night hours available within it cover the requested time.",
                no="It never reaches the limit, or the hours available within the limit are fewer than requested."),
            "sky_constraint_cost": score("How hard will the requested sky conditions be to obtain at this site?", [
                "Easy: the combined conditions occur in a large share of the time.",
                "Moderate: they occur in roughly a fifth to two fifths of the time.",
                "Demanding: they occur in under a fifth of the time.",
                "Rare: the combination occurs only in a small fraction of the time.",
            ]),
            "overheads_included": noul(
                "The requested hours include the acquisition and readout overheads the call prescribes."),
        }


# =========================================================================== schema 3
class QueueBlockExecution(Schema):
    name = "queue_block_execution"
    description = ("A queue observer's call on one observation block tonight: sky and weather against the block's "
                   "constraints and the dome limits, programme rank and the target's remaining window.")
    surfaces = {"json": 0.45, "queue_tool": 0.3, "night_log_chat": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        site, tel = _site(rng), _tel(rng)
        topic, inst = _topic(rng)
        cycle = _cycle(rng)
        s0, s1 = _semester_span(cycle)
        night = s0 + timedelta(days=rng.randrange((s1 - s0).days))
        seeing_max = pick(rng, [0.6, 0.7, 0.8, 1.0, 1.2, 1.5])
        sky_req = pick(rng, list(SKY_REQ))
        moon_req = pick(rng, ["dark", "grey", "any"])
        airmass_max = pick(rng, [1.3, 1.5, 1.8, 2.0])
        hum_lim, gust_lim, dew_lim = pick(rng, [85, 88, 90, 92, 95]), pick(rng, [15, 18, 20, 22]), pick(rng, [1.5, 2.0, 2.5])
        # ---- dome weather
        close = chance(rng, {"clear": 0.4, "borderline": 0.42, "conflicting": 0.36}[reg])
        trig = pick(rng, ["humidity", "gust", "dew"])
        tight = reg != "clear"
        hum = round(hum_lim - (rng.uniform(1.0, 3.4) if tight else rng.uniform(8, 40)), 0)
        gust = round(gust_lim - (rng.uniform(0.3, 2) if tight and chance(rng, 0.5) else rng.uniform(4, 12)), 1)
        dew = round(dew_lim + (rng.uniform(0.1, 0.8) if tight and chance(rng, 0.5) else rng.uniform(2, 9)), 1)
        precip = False
        if close:
            if trig == "humidity":
                hum = min(100, round(hum_lim + (rng.uniform(0, 1.5) if tight else rng.uniform(3, 8)), 0))
                dew = round(min(dew, rng.uniform(dew_lim + 0.2, dew_lim + 2.0)), 1)
            elif trig == "gust":
                gust = round(gust_lim + (rng.uniform(0, 1.0) if tight else rng.uniform(3, 8)), 1)
            else:
                dew = round(dew_lim - (rng.uniform(0, 0.4) if tight else rng.uniform(0.6, 1.4)), 1)
            if not tight and chance(rng, 0.2):
                precip = True
        hum = max(5, hum)
        wind = round(max(2.0, gust * rng.uniform(0.55, 0.75)), 1)
        # ---- sky versus the block's constraints
        cls = pick_w(rng, {"clear": {3: 4, 0: 3, 1: 1}, "borderline": {2: 5, 1: 3, 3: 1},
                           "conflicting": {3: 2, 2: 3, 1: 3, 0: 1}}[reg])
        allowed = SKY_REQ[sky_req]
        sky_idx = rng.randint(0, allowed)
        if cls == 3:
            s_ratio = rng.uniform(0.55, 0.88)
        elif cls == 2:
            s_ratio = rng.uniform(0.92, 1.1)
        elif cls == 1:
            how = pick(rng, ["seeing", "sky", "moon"])
            s_ratio = rng.uniform(1.12, 1.45) if how == "seeing" else rng.uniform(0.6, 0.95)
            if how == "sky":
                sky_idx = min(4, allowed + 1)
        else:
            s_ratio = rng.uniform(1.55, 2.4) if chance(rng, 0.6) else rng.uniform(0.9, 1.3)
            if s_ratio < 1.5:
                sky_idx = min(4, allowed + 2)
        seeing = round(seeing_max * s_ratio, 2)
        if close and chance(rng, 0.5):
            sky_idx = max(sky_idx, 2)
        trend = pick(rng, ["improving", "stable", "worsening"])
        step = {"improving": -1, "stable": 0, "worsening": 1}[trend]
        dimm = [round(seeing - step * rng.uniform(0.04, 0.12) * k + rng.uniform(-0.02, 0.02), 2) for k in (2, 1)] + [seeing]
        moon_bad_wanted = cls == 1 and s_ratio < 1.0 and sky_idx <= allowed
        illum = rng.randint(0, 100)
        moon_alt = rng.randint(-40, 70)
        sep = rng.randint(35, 160)
        if moon_bad_wanted:
            if moon_req == "any":
                moon_alt, sep = rng.randint(15, 60), rng.randint(12, 28)
            else:
                illum, moon_alt = rng.randint(MOON_REQ_ILLUM[moon_req] + 5, 100), rng.randint(10, 60)
        elif moon_req != "any" and moon_alt > 0 and illum > MOON_REQ_ILLUM[moon_req]:
            moon_alt = -rng.randint(5, 40)
        if moon_alt <= 0:
            illum = min(illum, rng.randint(5, 60))        # a nearly full moon is up most of the night
        timing = chance(rng, 0.12)
        season = pick_w(rng, {1: 0.8, 2: 0.7, 4: 1, 7: 1.2, 12: 1.5, 25: 1.5, 45: 1.5, 80: 2, 120: 1.5})
        # target geometry: local sidereal time from the date, hour angle west of the meridian for a target that is
        # setting for the season, airmass and hours left from the site latitude
        lat = site[2]
        lh = pick(rng, [21, 22, 23, 23, 0, 1, 2, 3]) if season > 12 else pick(rng, [20, 21, 21, 22, 23])
        lmin = rng.randrange(60)
        t_mid = (lh - 24 if lh >= 12 else lh) + lmin / 60.0
        lst = ((night.timetuple().tm_yday - 264) * 0.0657 + t_mid) % 24
        dec = max(-85.0, min(85.0, lat + rng.uniform(-35, 35)))
        hal = min(6.0, _ha_limit(lat, dec, airmass_max))
        if hal < 1.0:
            dec = lat + rng.uniform(-10, 10)
            hal = min(6.0, _ha_limit(lat, dec, airmass_max))
        ha = rng.uniform(max(0.2, hal * 0.4), max(0.3, hal - 0.3)) if season <= 12 else rng.uniform(-hal + 0.3, hal * 0.5)
        ra = (lst - ha) % 24
        phi, dec_r = math.radians(lat), math.radians(dec)
        cosz = math.sin(phi) * math.sin(dec_r) + math.cos(phi) * math.cos(dec_r) * math.cos(math.radians(15 * ha))
        airmass_now = round(1.0 / max(cosz, 0.05), 2)
        hours_left = round(max(0.2, min(hal - ha, rng.uniform(5.0, 6.0) - t_mid)), 1)
        sem_left = max(1, (s1 - night).days)
        rank = pick_w(rng, {"A": 4, "B": 4, "C": 2})
        completion = rng.randint(0, 95)
        relax_seeing = None
        relax_sky = None
        if chance(rng, 0.5):
            if chance(rng, 0.65):
                relax_seeing = round(seeing_max * pick(rng, [1.2, 1.25, 1.3, 1.5]), 1)
            else:
                relax_sky = min(4, allowed + 1)
        if reg != "clear" and seeing > seeing_max and sky_idx <= allowed and chance(rng, 0.45):
            # the PI's advance relaxation just covers tonight's seeing: in spec for charging, out of spec nominally
            relax_seeing, relax_sky = math.ceil(seeing * rng.uniform(1.02, 1.2) * 10) / 10, None
        backup = chance(rng, 0.75)
        b_seeing = pick(rng, [1.0, 1.2, 1.5, 2.0, 2.5])
        b_sky = pick(rng, [1, 2, 3])
        return {
            "regime": reg, "observatory": site[0], "obs_code": site[1], "telescope": tel[0], "aperture_m": tel[1],
            "instrument": inst, "science": topic[0], "cycle": cycle, "night": night,
            "block_id": f"{_prop_id(rng, cycle, site[1])}/OB{rng.randint(1, 40):02d}", "rank": rank,
            "completion_pct": completion, "pi": person(rng), "observer": person(rng), "operator": person(rng),
            "target": _jname(rng, ra, dec), "hour_angle": round(ha, 2),
            "seeing_max": seeing_max, "sky_req": sky_req, "moon_req": moon_req, "airmass_max": airmass_max,
            "relax_seeing": relax_seeing, "relax_sky": relax_sky,
            "hum_lim": hum_lim, "gust_lim": gust_lim, "dew_lim": dew_lim,
            "humidity": hum, "wind": wind, "gust": gust, "dew_spread": dew, "precip": precip,
            "seeing": seeing, "dimm": dimm, "trend": trend, "sky_idx": sky_idx,
            "moon_illum": illum, "moon_alt": moon_alt, "moon_sep": sep, "min_moon_sep": 30,
            "airmass_now": airmass_now, "hours_left_tonight": hours_left, "timing_critical": timing,
            "timing_window": f"{rng.randint(0, 4):02d}:{rng.randrange(0, 60, 5):02d}",
            "season_nights_left": season, "semester_nights_left": sem_left,
            "backup_available": backup, "backup_block": _prop_id(rng, cycle, site[1]),
            "backup_rank": pick(rng, ["A", "B", "B", "C"]), "backup_seeing": b_seeing, "backup_sky": b_sky,
            "local_time": f"{lh:02d}:{lmin:02d}",
        }

    # ------------------------------------------------------------------ rule logic
    def _dome_close(self, f):
        return (f["humidity"] >= f["hum_lim"] or f["gust"] >= f["gust_lim"] or f["dew_spread"] <= f["dew_lim"]
                or f["precip"])

    def _moon_ok(self, f):
        up = f["moon_alt"] > 0
        if up and f["moon_sep"] < f["min_moon_sep"]:
            return False
        return (not up) or f["moon_illum"] <= MOON_REQ_ILLUM[f["moon_req"]]

    def _violations(self, f, seeing_max=None, sky_allowed=None):
        sm = seeing_max or f["seeing_max"]
        sa = SKY_REQ[f["sky_req"]] if sky_allowed is None else sky_allowed
        return {"seeing": f["seeing"] / sm, "sky_over": f["sky_idx"] - sa, "moon_bad": not self._moon_ok(f),
                "airmass_bad": f["airmass_now"] > f["airmass_max"]}

    def _fit(self, f):
        v = self._violations(f)
        n = (v["seeing"] > 1.0) + (v["sky_over"] > 0) + v["moon_bad"] + v["airmass_bad"]
        if n == 0:
            return 3 if v["seeing"] <= 0.9 else 2
        if n == 1 and v["seeing"] <= 1.1 and v["sky_over"] <= 0:
            return 2 if not (v["moon_bad"] or v["airmass_bad"]) else 1
        if n == 1 and v["seeing"] <= 1.5 and v["sky_over"] <= 1:
            return 1
        return 0

    def _within(self, f):
        v = self._violations(f)
        return v["seeing"] <= 1.0 and v["sky_over"] <= 0 and not v["moon_bad"] and not v["airmass_bad"]

    def _within_relaxed(self, f):
        if f["relax_seeing"] is None and f["relax_sky"] is None:
            return False
        v = self._violations(f, seeing_max=f["relax_seeing"], sky_allowed=f["relax_sky"])
        return v["seeing"] <= 1.0 and v["sky_over"] <= 0 and not v["moon_bad"] and not v["airmass_bad"]

    def _pressure(self, f):
        if f["timing_critical"] or f["season_nights_left"] <= 2:
            return 3
        if f["season_nights_left"] <= 10 or (f["semester_nights_left"] <= 7 and f["completion_pct"] < 80):
            return 2
        if f["season_nights_left"] <= 40 or f["semester_nights_left"] <= 21:
            return 1
        return 0

    def _backup_fits(self, f):
        return f["backup_available"] and f["seeing"] <= f["backup_seeing"] and f["sky_idx"] <= f["backup_sky"]

    def intended(self, f):
        a = {}
        close = self._dome_close(f)
        fit, within, relaxed, pressure = self._fit(f), self._within(f), self._within_relaxed(f), self._pressure(f)
        if close:
            call = "close_dome"
        elif within:
            call = "execute_as_specified"
        elif relaxed or (fit >= 1 and pressure == 3 and f["rank"] == "A"):
            call = "execute_relaxed_flagged"
        elif self._backup_fits(f):
            call = "switch_to_backup"
        else:
            call = "wait_and_reassess"
        a["block_call"] = call
        a["conditions_fit"] = fit
        a["window_pressure"] = pressure
        a["dome_must_close"] = "yes" if close else "no"
        a["time_chargeable"] = "yes" if (within or relaxed) else "no"
        return a

    # ------------------------------------------------------------------ rendering
    def _relax_text(self, f):
        if f["relax_seeing"]:
            return f"PI accepts seeing up to {f['relax_seeing']}\" if the data are flagged"
        if f["relax_sky"] is not None:
            return f"PI accepts {SKY[f['relax_sky']]} if the data are flagged"
        return None

    def _moon_text(self, f):
        if f["moon_alt"] <= 0:
            return f"moon below horizon ({f['moon_illum']}% illuminated)"
        return f"moon {f['moon_illum']}% illuminated, altitude {f['moon_alt']} deg, {f['moon_sep']} deg from target"

    def _season_text(self, f):
        s = f"target sets for the season in ~{_pl(f['season_nights_left'], 'night')}"
        if f["timing_critical"]:
            s += f"; phase-critical window tonight centred {f['timing_window']} (next comparable window in 5+ weeks)"
        return s

    def _rules(self, f):
        return {"dome": (f"Close at relative humidity >= {f['hum_lim']}%, wind gusts >= {f['gust_lim']} m/s, dew-point "
                         f"spread <= {f['dew_lim']} C, or any precipitation."),
                "moon": (f"dark = moon down or <= 30% illuminated; grey = down or <= 70%; any moon must be "
                         f"{f['min_moon_sep']} deg or more from the target."),
                "charging": ("Time is charged to the programme when the block runs within its constraints or within "
                             "limits the PI approved in advance; blocks run outside those are flagged and not charged.")}

    def render(self, f, rng, surface):
        rules = self._rules(f)
        sky_now = SKY[f["sky_idx"]]
        backup = (f"{f['backup_block']} (rank {f['backup_rank']}): seeing <= {f['backup_seeing']}\", sky up to "
                  f"{SKY[f['backup_sky']]}" if f["backup_available"] else None)
        if surface == "json":
            return dumps(prune({
                "block": {"id": f["block_id"], "rank": f["rank"], "programme_completion_pct": f["completion_pct"],
                          "pi": f["pi"], "science": f["science"], "instrument": f"{f['instrument']} / {f['telescope']} {f['aperture_m']} m",
                          "constraints": {"seeing_max_arcsec": f["seeing_max"], "sky": f["sky_req"], "moon": f["moon_req"],
                                          "airmass_max": f["airmass_max"]},
                          "pi_relaxation": self._relax_text(f)},
                "now": {"local_time": f["local_time"], "night": f["night"].isoformat(),
                        "dimm_seeing_last_30min": f["dimm"], "sky": sky_now, "humidity_pct": f["humidity"],
                        "wind_ms": f["wind"], "gust_ms": f["gust"], "dew_point_spread_c": f["dew_spread"],
                        "precipitation": f["precip"] or None, "moon": self._moon_text(f)},
                "target": {"name": f["target"], "airmass_now": f["airmass_now"],
                           "hours_left_tonight_within_limit": f["hours_left_tonight"], "season": self._season_text(f)},
                "queue": {"semester_nights_left": f["semester_nights_left"],
                          "backup_block": backup or "none suitable for this instrument tonight"},
                "rules": rules,
            }))
        if surface == "queue_tool":
            lines = [f"QUEUE  {f['obs_code']} {f['aperture_m']}m {f['telescope']} / {f['instrument']}   night "
                     f"{fmt_date(f['night'], style='iso')}   now {f['local_time']}",
                     f"Block       {f['block_id']}  rank {f['rank']}  PI {f['pi']}  programme {f['completion_pct']}% complete",
                     f"Target      {f['target']}  airmass {f['airmass_now']}  ({f['hours_left_tonight']} h left within limit)",
                     f"Constraint  seeing <= {f['seeing_max']}\"  sky {f['sky_req']}  moon {f['moon_req']}  "
                     f"airmass <= {f['airmass_max']}"]
            if self._relax_text(f):
                lines.append(f"            note: {self._relax_text(f)}")
            lines += [f"DIMM        {' '.join(str(x) for x in f['dimm'])} arcsec (last 30 min)",
                      f"Sky         {sky_now}" + ("   PRECIPITATION reported" if f["precip"] else ""),
                      f"Weather     RH {f['humidity']:.0f}%  wind {f['wind']} m/s gust {f['gust']}  dew-pt spread {f['dew_spread']} C",
                      f"Moon        {self._moon_text(f)}",
                      f"Season      {self._season_text(f)}; {f['semester_nights_left']} nights left in {f['cycle']}",
                      f"Backup      {backup or 'none suitable for this instrument tonight'}", "",
                      "Limits: " + rules["dome"], "Moon: " + rules["moon"], "Charging: " + rules["charging"]]
            return "\n".join(lines)
        # night_log_chat
        obs, op = f["observer"].split()[0].lower(), f["operator"].split()[0].lower()
        msgs = [(op, f"weather station: RH {f['humidity']:.0f}%, wind {f['wind']} gusting {f['gust']}, dew-point spread "
                     f"{f['dew_spread']}" + (", and we have drops on the camera" if f["precip"] else "")),
                (obs, f"DIMM reading {', '.join(str(x) for x in f['dimm'])} over the last half hour, sky looks {sky_now}"),
                (obs, f"next up is {f['block_id']} (rank {f['rank']}, {f['completion_pct']}% done), wants seeing <= "
                      f"{f['seeing_max']}\", {f['sky_req']}, {f['moon_req']} moon, airmass <= {f['airmass_max']}"),
                (obs, f"{f['target']} at airmass {f['airmass_now']}, {f['hours_left_tonight']} h left; {self._season_text(f)}"),
                (op, self._moon_text(f))]
        if self._relax_text(f):
            msgs.append((obs, "README on the block says: " + self._relax_text(f)))
        msgs.append((op, f"backup in the queue: {backup}" if backup else "no backup block for this instrument tonight"))
        msgs.append((op, pick(rng, ["your call on the block", "what do you want to do?", "standing by",
                                    "let me know before I slew"])))
        hh, mm = f["local_time"].split(":")
        t = int(hh) * 60 + int(mm) - 12
        out = []
        for who, text in msgs:
            t += rng.randint(1, 3)
            out.append(f"[{(t // 60) % 24:02d}:{t % 60:02d}] {who}: {noisy(text, rng, 0.08)}")
        return (f"#nightops-{f['obs_code'].lower()} ({f['telescope']} {f['aperture_m']} m, night of "
                f"{fmt_date(f['night'], rng)})\n" + "\n".join(out) + "\n\nOps limits: " + rules["dome"] + "\n" +
                rules["moon"] + "\n" + rules["charging"] + f"\nSemester {f['cycle']}: {f['semester_nights_left']} nights left.")

    def questions(self):
        return {
            "block_call": choice("What should the observer do with this block right now?", {
                "close_dome": "Close the dome and stop observing until the weather is back inside the operating limits.",
                "execute_as_specified": "Start the block now; the sky meets every one of its stated constraints.",
                "execute_relaxed_flagged": "Start the block now outside its nominal constraints and flag the data as "
                                           "taken in relaxed conditions.",
                "switch_to_backup": "Leave this block and run a backup block whose constraints suit the current sky.",
                "wait_and_reassess": "Keep the dome open but hold this block and reassess conditions after a short wait.",
            }),
            "conditions_fit": score("How well does the current sky match the block's constraints?", [
                "Far outside: several constraints are missed, or one by a wide margin.",
                "Outside: one constraint is clearly missed.",
                "Marginal: everything is at, or a whisker beyond, the limits.",
                "Comfortable: every constraint is met with room to spare.",
            ]),
            "window_pressure": score("How pressing is it to get this block done tonight rather than on a later night?", [
                "Not pressing: the target stays well placed for many weeks.",
                "Some pressure: a few weeks of the target's season or the semester remain.",
                "Pressing: the target sets for the season, or the semester ends, within days.",
                "Tonight or never: a timed event tonight, or the last usable night of the target's season.",
            ]),
            "dome_must_close": noul(
                "The weather requires the dome to be closed now under the observatory's operating limits.",
                yes="Humidity, gusts, dew-point spread or precipitation has reached a written closure limit.",
                no="Every weather reading is still inside the closure limits, however poor the sky may be."),
            "time_chargeable": noul(
                "Running this block now would be charged against the programme's allocated hours.",
                yes="The current conditions are within the block's constraints or within limits the PI approved in advance.",
                no="The current conditions fall outside both the constraints and any relaxation the PI approved."),
        }


# =========================================================================== schema 4
# kind -> (label, [(criterion, comparator, thresholds, unit, additive?)], rarity 1-3, data-needed-within hours)
TOO_KINDS = {
    "young_sn": ("a very young supernova", [
        ("days since last non-detection", "le", [2, 3, 4], "d", False),
        ("discovery magnitude", "le", [18.0, 18.5, 19.0], "mag", True),
        ("host redshift", "le", [0.02, 0.04, 0.06], "", False)], 2, [12, 24, 48]),
    "grb_afterglow": ("an optical gamma-ray-burst afterglow", [
        ("hours since the burst", "le", [4, 8, 12], "h", False),
        ("position uncertainty", "le", [2.0, 3.0, 5.0], "arcsec", False),
        ("afterglow magnitude", "le", [19.5, 20.0, 21.0], "mag", True)], 2, [6, 12, 24]),
    "kilonova_candidate": ("a candidate optical counterpart to a gravitational-wave event", [
        ("distance estimate", "le", [120, 160, 200], "Mpc", False),
        ("credible region containing the candidate", "le", [90, 90, 95], "%", False),
        ("fade rate", "ge", [0.3, 0.4, 0.5], "mag/day", False)], 3, [12, 24]),
    "comet_outburst": ("an outburst of a distant comet", [
        ("brightening", "ge", [1.5, 2.0, 2.5], "mag", False),
        ("solar elongation", "ge", [45, 50, 60], "deg", False),
        ("days since outburst", "le", [3, 4, 5], "d", False)], 1, [24, 48, 72]),
    "microlensing_anomaly": ("a planetary anomaly in a microlensing event", [
        ("hours to predicted caustic exit", "le", [12, 18, 24], "h", False),
        ("source magnitude (I)", "le", [17.0, 17.5, 18.0], "mag", True),
        ("model probability of a planetary anomaly", "ge", [50, 60, 70], "%", False)], 2, [6, 12]),
    "fuor_outburst": ("an outburst of a young eruptive star", [
        ("rise amplitude", "ge", [2.5, 3.0, 3.5], "mag", False),
        ("rise time", "le", [60, 90, 120], "d", False),
        ("distance", "le", [600, 800, 1000], "pc", False)], 1, [72, 120]),
    "tde_candidate": ("a candidate tidal disruption event", [
        ("offset from galaxy nucleus", "le", [0.3, 0.4, 0.5], "arcsec", False),
        ("g-r colour", "le", [-0.1, 0.0, 0.1], "mag", True),
        ("days since first detection", "le", [10, 14, 21], "d", False)], 2, [24, 48, 96]),
    "nova_nearby": ("a classical nova in a nearby galaxy", [
        ("days since discovery", "le", [2, 3, 4], "d", False),
        ("current magnitude", "le", [16.5, 17.0, 17.5], "mag", True),
        ("host distance", "le", [1.0, 2.0, 3.5], "Mpc", False)], 1, [24, 48]),
    "fast_blue_transient": ("a fast-rising blue optical transient", [
        ("rise time to peak", "le", [3, 4, 5], "d", False),
        ("peak absolute magnitude", "le", [-19.0, -18.5, -18.0], "mag", True),
        ("days since peak", "le", [2, 3, 4], "d", False)], 3, [12, 24]),
}
COVERAGE = {"none": "No other facility has reported follow-up yet.",
            "one": "One other facility reports photometry; no spectrum yet.",
            "several": "Two other facilities have already posted spectra and a classification."}


def _crit_value(rng, thr, cmp_, additive, mode):
    """A measured value against threshold `thr`: comfortably passing, near the line, or failing."""
    sgn = 1 if cmp_ == "ge" else -1                                  # direction in which a value is 'better'
    mag = abs(thr) if thr else 1.0
    if mode == "pass":
        d = rng.uniform(0.3, 1.2) if additive else mag * rng.uniform(0.15, 0.55)
    elif mode == "near":
        d = rng.uniform(-0.15, 0.15) if additive else mag * rng.uniform(-0.08, 0.08)
    else:
        d = -(rng.uniform(0.25, 1.0) if additive else mag * rng.uniform(0.15, 0.6))
    v = thr + sgn * d
    if not additive and isinstance(thr, int):
        v = int(round(v))
    return round(v, 2)


class TooInterruptRequest(Schema):
    name = "too_interrupt_request"
    description = ("A target-of-opportunity request to interrupt tonight's scheduled programme, against the approved "
                   "ToO programme's written trigger criteria and the observatory's interrupt and compensation rules.")
    surfaces = {"json": 0.5, "too_email": 0.25, "duty_log": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        site, tel = _site(rng), _tel(rng)
        cycle = _cycle(rng)
        kind = pick(rng, list(TOO_KINDS))
        label, crits, rarity, within_opts = TOO_KINDS[kind]
        thresholds = [pick(rng, c[2]) for c in crits]
        modes = ["pass"] * len(crits)
        missing = None
        if reg == "clear":
            if chance(rng, 0.5):
                modes[rng.randrange(len(crits))] = "fail"
        elif reg == "borderline":
            modes[rng.randrange(len(crits))] = "near"
            if chance(rng, 0.15):
                modes[rng.randrange(len(crits))] = "fail"
        else:
            r = rng.random()
            if r < 0.4:
                missing = rng.randrange(len(crits))
            elif r < 0.7:
                modes[rng.randrange(len(crits))] = "near"
        values = [_crit_value(rng, t, c[1], c[4], m) for t, c, m in zip(thresholds, crits, modes)]
        if missing is not None:
            values[missing] = None
        approved = pick(rng, [6, 8, 10, 12, 16, 20, 24])
        used = round(approved * rng.uniform(0.1, 0.98), 1)
        request = pick(rng, [0.5, 0.75, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        if reg == "conflicting" and chance(rng, 0.4):
            used = round(approved - request * rng.uniform(0.3, 0.8), 1)
        disp_rank = pick_w(rng, {"A": 4, "B": 4, "C": 2})
        disp_mode = pick_w(rng, {"queue": 6, "visitor": 4})
        timed = chance(rng, 0.2 if reg != "conflicting" else 0.35)
        return {
            "regime": reg, "observatory": site[0], "obs_code": site[1], "telescope": tel[0], "aperture_m": tel[1],
            "instrument": pick(rng, ["TARN", "SERAC", "MORAINE", "ARETE", "CIRQUE", "DRUMLIN"]), "cycle": cycle,
            "kind": kind, "event_label": label, "event_name": self._event_name(rng, kind, cycle),
            "criteria": [c[0] for c in crits], "comparators": [c[1] for c in crits], "units": [c[3] for c in crits],
            "thresholds": thresholds, "values": values, "missing_index": missing,
            "rarity": rarity, "needed_within_h": pick(rng, within_opts),
            "too_programme": f"{_prop_id(rng, cycle, site[1])}-ToO", "too_pi": person(rng), "duty": person(rng),
            "approved_h": approved, "used_h": used, "request_h": request,
            "displaced_id": _prop_id(rng, cycle, site[1]), "displaced_pi": person(rng), "displaced_rank": disp_rank,
            "displaced_mode": disp_mode, "displaced_timed": timed,
            "displaced_season_nights": pick(rng, [5, 10, 15, 25, 40, 70]),
            "visitor_cap_h": pick(rng, [1.0, 1.5, 2.0]), "absorb_below_h": pick(rng, [0.5, 1.0]),
            "coverage": pick_w(rng, {"none": 4, "one": 3, "several": 2}),
            "pi_tone": pick(rng, ["measured", "measured", "urgent"]),
            "date": _semester_span(cycle)[0] + timedelta(days=rng.randrange(170)),
        }

    @staticmethod
    def _event_name(rng, kind, cycle):
        yy = cycle[2:4]
        letters = "".join(pick(rng, "abcdefghijklmnopqrstuvwxyz") for _ in range(3))
        if kind == "grb_afterglow":
            return f"GRB {yy}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}{pick(rng, 'ABC')}"
        if kind == "comet_outburst":
            return f"C/20{yy} {pick(rng, 'ABCDEFGHJKLMNOPQRSTUVWXY')}{rng.randint(1, 9)}"
        if kind == "microlensing_anomaly":
            return f"OPL-20{yy}-BLG-{rng.randint(1, 1800):04d}"
        if kind == "kilonova_candidate":
            return f"{pick(rng, ['VSC', 'KRS', 'HLX'])}{yy}{letters} (in the skymap of TRD-{yy}{rng.randint(1000, 9999)})"
        return f"{pick(rng, ['VSC', 'KRS', 'HLX', 'SDR', 'TPS'])}{yy}{letters}"

    # ------------------------------------------------------------------ rule logic
    def _met(self, f):
        for v, t, c in zip(f["values"], f["thresholds"], f["comparators"]):
            if v is None:
                return False
            if c == "le" and v > t:
                return False
            if c == "ge" and v < t:
                return False
        return True

    def _remaining(self, f):
        return round(f["approved_h"] - f["used_h"], 1)

    def _young(self, f):
        """Very early data: an age-type criterion at <= 40% of its limit."""
        for name, v, t in zip(f["criteria"], f["values"], f["thresholds"]):
            if v is not None and any(w in name for w in ("days since", "hours since", "hours to")) and t:
                return v <= 0.4 * t
        return False

    def intended(self, f):
        a = {}
        met = self._met(f)
        rem = self._remaining(f)
        if not met:
            ruling = "decline"
        elif f["needed_within_h"] > 24:
            ruling = "defer_to_queue"
        elif f["displaced_timed"] or f["request_h"] > rem or \
                (f["displaced_mode"] == "visitor" and f["request_h"] > f["visitor_cap_h"]):
            ruling = "partial_interrupt"
        else:
            ruling = "interrupt_now"
        a["interrupt_ruling"] = ruling
        a["trigger_criteria_met"] = "yes" if met else "no"
        c = {"C": 0, "B": 1, "A": 2}[f["displaced_rank"]] + (1 if f["displaced_mode"] == "visitor" else 0)
        if f["displaced_timed"]:
            c = 3
        elif f["request_h"] <= 0.75:
            c -= 1
        a["displacement_cost"] = _clamp(c)
        s = f["rarity"] + (1 if self._young(f) else 0) - {"none": 0, "one": 0, "several": 2}[f["coverage"]]
        a["science_return"] = _clamp(s)
        loss = f["request_h"]
        protected = f["displaced_mode"] == "visitor" or f["displaced_rank"] == "A"
        if f["displaced_timed"]:
            comp = "director_decides"
        elif not protected or loss < f["absorb_below_h"]:
            comp = "none_needed"
        elif loss > 3.0:
            comp = "director_decides"
        elif f["displaced_season_nights"] >= 21:
            comp = "return_this_semester"
        else:
            comp = "carry_to_next_semester"
        a["compensation"] = comp
        return a

    # ------------------------------------------------------------------ rendering
    def _crit_lines(self, f):
        sym = {"le": "<=", "ge": ">="}
        return [f"{n} {sym[c]} {t}{(' ' + u) if u else ''}" for n, c, t, u in
                zip(f["criteria"], f["comparators"], f["thresholds"], f["units"])]

    def _measured(self, f):
        out = {}
        for n, v, u in zip(f["criteria"], f["values"], f["units"]):
            out[n] = "not reported" if v is None else f"{v}{(' ' + u) if u else ''}"
        return out

    def _policy(self, f):
        return [
            "Rapid-response interrupts are allowed only when every trigger criterion of the approved ToO programme is "
            "met and the science needs data within 24 hours; otherwise an accepted trigger is queued for the next "
            "suitable night.",
            f"A visitor-mode programme may be interrupted for at most {f['visitor_cap_h']} h per night; a timed "
            "observation of the displaced programme must be protected.",
            "An interrupt cannot exceed the hours left in the ToO programme's allocation.",
            "Compensation (Director's policy): the loss of a timed observation always goes to the Director. "
            f"Otherwise time lost by a visitor-mode programme or a rank-A queue block is returned: losses under "
            f"{f['absorb_below_h']} h are absorbed; up to 3 h is returned later this semester if the displaced target "
            "stays observable for three more weeks, otherwise carried to the next semester; more than 3 h goes to the "
            "Director. Other queue blocks are simply re-queued.",
        ]

    def _displaced_text(self, f):
        s = (f"{f['displaced_id']} (PI {f['displaced_pi']}), rank {f['displaced_rank']}, "
             f"{'visitor mode, observer on site' if f['displaced_mode'] == 'visitor' else 'queue mode'}")
        if f["displaced_timed"]:
            s += pick_text({"proposal_id": f["displaced_id"]},
                           ["; tonight includes a timed transit egress that recurs only in five weeks",
                            "; a phase-critical eclipse is scheduled tonight",
                            "; a timed occultation chord is planned tonight"])
        return s + f"; displaced target observable for ~{f['displaced_season_nights']} more nights"

    def render(self, f, rng, surface):
        measured = self._measured(f)
        if surface == "json":
            return dumps(prune({
                "request": {"event": f["event_name"], "type": f["event_label"], "too_programme": f["too_programme"],
                            "pi": f["too_pi"], "hours_requested": f["request_h"],
                            "instrument": f"{f['instrument']} on {f['telescope']} {f['aperture_m']} m",
                            "science_needs_data_within_h": f["needed_within_h"], "pi_message": self._pi_line(f)},
                "event_measurements": measured,
                "too_programme": {"trigger_criteria": self._crit_lines(f), "approved_hours": f["approved_h"],
                                  "hours_used": f["used_h"]},
                "displaced": {"programme": self._displaced_text(f)},
                "other_follow_up": COVERAGE[f["coverage"]],
                "policy": self._policy(f),
            }))
        if surface == "too_email":
            urgent = "We need to move fast on this. " if f["pi_tone"] == "urgent" else ""
            body = [f"Subject: ToO trigger {f['too_programme']} -- {f['event_name']}", f"From: {f['too_pi']}",
                    f"To: duty astronomer, {f['observatory']}", "",
                    "Dear duty astronomer,", "",
                    noisy(join_sentences([
                        f"I am triggering our ToO programme {f['too_programme']} on {f['event_name']}, {f['event_label']}.",
                        urgent + f"We request {f['request_h']} h with {f['instrument']} tonight.",
                        f"Our science case needs data within {f['needed_within_h']} hours.",
                        COVERAGE[f["coverage"]]]), rng),
                    "", "Measurements against our approved trigger criteria:",
                    bullets([f"{k}: {v}" for k, v in measured.items()], rng),
                    "", "Approved criteria: " + "; ".join(self._crit_lines(f)) + ".",
                    f"Our programme has used {f['used_h']} of {f['approved_h']} h.", "",
                    "Best regards,", f["too_pi"], "",
                    "-- duty astronomer's notes --",
                    "Displaced tonight: " + self._displaced_text(f) + ".", "Policy:", bullets(self._policy(f), rng)]
            return "\n".join(body)
        # duty_log
        sym = {"le": "<=", "ge": ">="}
        req = [f"{sym[c]} {t}{(' ' + u) if u else ''}" for c, t, u in zip(f["comparators"], f["thresholds"], f["units"])]
        rows = [[k, v, r] for (k, v), r in zip(measured.items(), req)]
        head = (f"DUTY LOG {f['obs_code']} {fmt_date(f['date'], rng)} -- ToO request received\n"
                + kv_lines([("Event", f"{f['event_name']} ({f['event_label']})"), ("ToO programme", f["too_programme"]),
                            ("Requested", f"{f['request_h']} h, {f['instrument']}, data needed within {f['needed_within_h']} h"),
                            ("ToO hours", f"{f['used_h']} used of {f['approved_h']}"),
                            ("Scheduled tonight", self._displaced_text(f)),
                            ("Other facilities", COVERAGE[f["coverage"]])], rng))
        return head + "\n\n" + table(["criterion", "reported", "required"], rows, rng) + "\n\nPolicy:\n" + \
            bullets(self._policy(f), rng)

    def _pi_line(self, f):
        return pick_text({"proposal_id": f["event_name"]}, [
            "Early data are essential; please interrupt as soon as possible.",
            "This looks like exactly the event our programme was approved for.",
            "Happy to take a shorter slot if the full time is not possible.",
            "We would like spectra before the event fades below reach."])

    def questions(self):
        return {
            "interrupt_ruling": choice("How should the duty astronomer rule on the interrupt request?", {
                "decline": "Turn the request down; the event does not qualify under the programme's approved trigger "
                           "conditions.",
                "defer_to_queue": "Accept the trigger but queue it for the next suitable night instead of interrupting "
                                  "tonight.",
                "interrupt_now": "Interrupt tonight's programme for the full requested time.",
                "partial_interrupt": "Interrupt tonight, but only for a shorter time or a slot that protects the "
                                     "displaced programme's critical part.",
            }),
            "trigger_criteria_met": noul(
                "The event satisfies every trigger condition written into the approved ToO programme.",
                yes="Every listed condition is met by the reported measurements.",
                no="At least one condition is failed or cannot be checked because the measurement was not reported."),
            "displacement_cost": score("How much would an interrupt of the requested length cost the programme it "
                                       "displaces?", [
                "Negligible: a low-ranked or easily repeated queue block.",
                "Modest: the lost time can be made up with little scientific loss.",
                "Significant: a high-ranked or visitor programme loses time it cannot easily recover.",
                "Severe: a timed, non-repeatable observation would be lost.",
            ]),
            "science_return": score("How much would observing the event now add scientifically?", [
                "Little: common event, or others already have the key data.",
                "Some: useful data on a fairly common type of event.",
                "Substantial: early or uncommon data that few others will get.",
                "Exceptional: a rare event at a stage where the data are unique.",
            ]),
            "compensation": choice("If the interrupt went ahead for the full requested time, how should the displaced "
                                   "programme be compensated under the Director's policy?", {
                "carry_to_next_semester": "Credit the lost time to the next semester, because the target will not "
                                          "stay observable long enough this semester.",
                "director_decides": "Refer the loss to the Director, because it is too large or a timed observation is "
                                    "lost.",
                "none_needed": "No compensation; the loss is small enough to absorb or the programme is simply "
                               "re-queued.",
                "return_this_semester": "Return the lost time later in this semester while the target is still "
                                        "observable.",
            }),
        }


# =========================================================================== schema 5
PI_CLAIMS = ["We are on track to complete the full sample within the approved allocation.",
             "The programme will finish on schedule; the remaining targets are the easier ones.",
             "Completion is expected within the remaining semesters, weather permitting.",
             "We may need a modest extension, but the core sample will be done.",
             "Progress was slowed by weather; we expect to catch up next semester."]
DELAY_REASONS = ["observing blocks were not prepared in time", "the team's postdoc left mid-programme",
                 "the pipeline needed rewriting for the new detector", "targets were reprioritised after a pilot result"]


class PriorUseContinuation(Schema):
    name = "prior_use_continuation"
    description = ("Annual continuation review of a multi-semester programme: time used, data reduced and released, "
                   "publications and reports against the observatory's continuation rules.")
    surfaces = {"json": 0.5, "progress_form": 0.25, "panel_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        site, tel = _site(rng), _tel(rng)
        topic, inst = _topic(rng)
        cycle = _cycle(rng)
        n_sem = pick(rng, [4, 5, 6])
        k = pick(rng, [2, 3, 4]) if n_sem > 4 else pick(rng, [2, 3])
        total = pick(rng, [60, 80, 100, 120, 150, 200, 240, 300, 400])
        alloc = round(total * k / n_sem)
        exec_min = pick(rng, [60, 65, 70, 75])
        obs_loss = round(alloc * rng.uniform(0.03, 0.3))
        # four separate axes -- time use, completion pace, output, obligations -- aligned in clear cases, near
        # their written lines in borderline cases, and pulling apart in conflicting ones
        if reg == "clear":
            good = chance(rng, 0.5)
            ax = {"use": "ok" if good else "low", "pace": "ok" if good else "low",
                  "out": pick(rng, [2, 3]) if good else pick(rng, [0, 0, 1]), "obl": good or chance(rng, 0.2)}
        elif reg == "borderline":
            ax = {"use": pick(rng, ["near", "ok", "ok"]), "pace": pick(rng, ["near", "ok", "ok", "low"]),
                  "out": pick(rng, [0, 1, 1, 2, 2, 3]), "obl": chance(rng, 0.5)}
        else:
            ax = {"use": pick(rng, ["ok", "ok", "low", "near"]), "pace": pick(rng, ["ok", "ok", "low", "near"]),
                  "out": pick(rng, [0, 1, 2, 2, 3]), "obl": chance(rng, 0.45)}
        mn = exec_min / 100
        eff = {"ok": rng.uniform(mn + 0.06, 0.98), "low": rng.uniform(0.3, mn - 0.1),
               "near": near(rng, mn, 0.06, lo=0.2, hi=0.99)}[ax["use"]]
        executed = round((alloc - obs_loss) * eff, 1)
        pi_side = round(alloc - obs_loss - executed, 1)
        # completion pace: remaining hours (incl. returned weather losses) against the need at the rate so far
        targets = pick(rng, [12, 20, 30, 40, 60, 80, 120])
        ratio_t = {"ok": rng.uniform(0.95, 1.8), "low": rng.uniform(0.35, 0.62),
                   "near": near(rng, pick(rng, [0.9, 1.15, 0.65]), 0.08, lo=0.3)}[ax["pace"]]
        rem = total - alloc + obs_loss
        needed = rem / ratio_t
        done = max(1, min(targets - 1, int(round(executed * targets / (needed + executed)))))
        # output: papers per 100 executed hours (half the bar in the first year) and the reduced share
        bar = 0.5 if k == 2 else 1.0
        lvl = ax["out"]
        if lvl == 0:
            papers, in_prep = 0, pick(rng, [0, 0, 1])
            reduced = rng.uniform(0.08, 0.38) if chance(rng, 0.6) else rng.uniform(0.4, 0.58)
        elif lvl == 1:
            papers, in_prep = (0, rng.randint(0, 3)) if chance(rng, 0.6) else (1, rng.randint(0, 2))
            reduced = rng.uniform(0.6, 0.95) if k > 2 else rng.uniform(0.5, 0.74)
        else:
            rate = rng.uniform(1.0, 2.0) if lvl == 2 else rng.uniform(2.7, 4.5)
            papers, in_prep = max(1, int(round(rate * bar * executed / 100))), rng.randint(0, 4)
            reduced = rng.uniform(0.55, 1.0)
        window = pick(rng, [6, 9, 12])                 # months from acquisition to archive release
        max_over = k * 6 - window - 1                   # a release cannot be later than the programme is old
        if ax["obl"]:
            report, release_over = "on_time", 0
        else:
            report = pick_w(rng, {"late": 3, "missing": 2, "on_time": 2})
            if max_over >= 1 and (report == "on_time" or chance(rng, 0.5)):
                release_over = rng.randint(1, max_over)
            else:
                release_over = 0
                if report == "on_time":
                    report = "late"
        review = _deadline(rng, cycle) + timedelta(days=rng.randint(25, 70))
        return {
            "regime": reg, "axes": ax, "observatory": site[0], "obs_code": site[1], "telescope": tel[0],
            "aperture_m": tel[1], "instrument": inst, "science": topic[0], "cycle": cycle,
            "programme_id": f"{pick(rng, ['LP', 'LTP', 'KP'])}-{site[1]}-{rng.randint(1, 40):02d}", "pi": person(rng),
            "institution": org(rng, "school"), "n_semesters": n_sem, "semesters_done": k, "total_h": total,
            "allocated_h": alloc, "executed_h": executed, "obs_loss_h": obs_loss, "pi_side_unused_h": max(0.0, pi_side),
            "exec_min_pct": exec_min, "targets_total": targets, "targets_done": done,
            "reduced_pct": int(round(reduced * 100)), "papers": papers, "papers_in_prep": in_prep,
            "report_status": report, "report_due": review - timedelta(days=rng.randint(20, 60)),
            "report_late_days": rng.randint(3, 50), "release_overdue_months": release_over,
            "release_window_months": window, "review_date": review,
            "pi_claim": pick(rng, PI_CLAIMS), "delay_reason": pick(rng, DELAY_REASONS) if pi_side > 3 else None,
        }

    # ------------------------------------------------------------------ rule logic
    def _eff(self, f):
        base_ = f["allocated_h"] - f["obs_loss_h"]
        return f["executed_h"] / base_ if base_ > 0 else 1.0

    def _completion_ratio(self, f):
        hpt = f["executed_h"] / max(1, f["targets_done"])
        needed = (f["targets_total"] - f["targets_done"]) * hpt
        rem = f["total_h"] - f["allocated_h"] + f["obs_loss_h"]
        return rem / needed if needed > 0 else 9.9

    def _productivity(self, f):
        per100 = f["papers"] / max(1.0, f["executed_h"]) * 100
        bar = 0.5 if f["semesters_done"] == 2 else 1.0       # first-year programmes are held to half the rate
        if f["papers"] == 0:
            if f["semesters_done"] == 2:     # first year: reduced data, not papers, is the expectation
                p = 2 if f["reduced_pct"] >= 75 else 1 if f["reduced_pct"] >= 50 else 0
            else:
                p = 1 if f["reduced_pct"] >= 60 else 0
        else:
            p = 3 if per100 >= 2.5 * bar else 2 if per100 >= 1.0 * bar else 1
        if f["reduced_pct"] < 40:
            p -= 1
        return _clamp(p)

    def _reporting_ok(self, f):
        return f["report_status"] == "on_time" and f["release_overdue_months"] == 0

    def intended(self, f):
        a = {}
        eff = self._eff(f)
        fails = eff < f["exec_min_pct"] / 100
        prod = self._productivity(f)
        ratio = self._completion_ratio(f)
        rep_ok = self._reporting_ok(f)
        a["reporting_obligations_met"] = "yes" if rep_ok else "no"
        a["productivity"] = prod
        cred = 3 if ratio >= 1.15 else 2 if ratio >= 0.9 else 1 if ratio >= 0.65 else 0
        if fails:
            cred -= 1
        a["completion_claim_credibility"] = _clamp(cred)
        issues = []
        if fails:
            issues.append((1.0 + (f["exec_min_pct"] / 100 - eff) * 6, "low_time_use"))
        if prod == 0 or (prod == 1 and f["semesters_done"] >= 3):
            issues.append((1.6 if prod == 0 else 1.0, "slow_publication"))
        rep_sev = {"on_time": 0.0, "late": 0.8, "missing": 1.8}[f["report_status"]] + \
            (0.5 + f["release_overdue_months"] / 6 if f["release_overdue_months"] else 0.0)
        if rep_sev > 0:
            issues.append((rep_sev, "reporting_lapses"))
        if ratio < 0.8:
            issues.append((1.0 + (0.8 - ratio) * 4, "unrealistic_completion"))
        a["main_shortfall"] = max(issues)[1] if issues else "none_significant"
        no_output = f["papers"] == 0 and f["reduced_pct"] < (50 if f["semesters_done"] == 2 else 60)   # rule (e)
        if (fails and no_output) or (f["report_status"] == "missing" and f["release_overdue_months"] >= 6):
            dec = "terminate"
        elif no_output:
            dec = "suspend_one_semester"
        elif fails or ratio < 0.75:                       # rules (c) and (d)
            dec = "continue_reduced"
        elif not rep_ok:
            dec = "continue_conditional"
        else:
            dec = "continue_as_planned"
        a["continuation_decision"] = dec
        return a

    # ------------------------------------------------------------------ rendering
    def _rules(self, f):
        return [f"(a) The annual progress report is due {fmt_date(f['report_due'], style='iso')}.",
                f"(b) Reduced data must be released to the archive within {f['release_window_months']} months of "
                "being taken.",
                f"(c) At least {f['exec_min_pct']}% of the allocated time must have been executed, not counting weather "
                "or technical losses, which are carried forward.",
                "(d) Projection: hours executed per completed target so far, times the targets still to do, must not "
                "exceed 4/3 of the hours still to come (approved total minus allocated to date, plus carried-forward "
                "losses).",
                "(e) No output means no refereed paper and under 60% of the data reduced (under 50% at the "
                "first-year review, after two semesters).",
                "Outcomes: failing (c) together with (e), or a missing report together with a release six or more "
                "months late, ends the programme; (e) alone pauses it for a semester; failing (c) or (d) reduces the "
                "hours or the target list; failing (a) or (b) makes continuation conditional. Where several apply: "
                "end > pause > reduce > conditional; otherwise it continues as planned.",
                "Output is otherwise judged against hours executed; at the first-year review reduced data, not yet "
                "refereed papers, are expected."]

    def _report_text(self, f):
        s = f["report_status"]
        if s == "on_time":
            return f"received {fmt_date(f['report_due'] - timedelta(days=3), style='iso')}"
        if s == "late":
            return f"received {fmt_date(f['report_due'] + timedelta(days=f['report_late_days']), style='iso')}"
        return "not received"

    def _release_text(self, f):
        m = f["release_overdue_months"]
        if m == 0:
            return "releases up to date"
        return f"reduced data from the first semester are {m} month{'s' if m > 1 else ''} past the release deadline"

    def render(self, f, rng, surface):
        rules = self._rules(f)
        if surface == "json":
            return dumps(prune({
                "programme": {"id": f["programme_id"], "pi": f["pi"], "institution": f["institution"],
                              "science": f["science"], "instrument": f"{f['instrument']} on {f['telescope']} {f['aperture_m']} m",
                              "approved": f"{f['total_h']} h over {f['n_semesters']} semesters",
                              "semesters_completed": f["semesters_done"]},
                "time_accounting_h": {"allocated_to_date": f["allocated_h"], "executed": f["executed_h"],
                                      "lost_weather_or_technical": f["obs_loss_h"],
                                      "unused_on_programme_side": f["pi_side_unused_h"]},
                "progress": {"targets_total": f["targets_total"], "targets_completed": f["targets_done"],
                             "data_reduced_pct": f["reduced_pct"], "refereed_papers": f["papers"],
                             "papers_in_preparation": f["papers_in_prep"], "pi_statement": f["pi_claim"],
                             "pi_explanation": f["delay_reason"]},
                "obligations": {"annual_report": self._report_text(f), "archive_release": self._release_text(f)},
                "continuation_rules": rules,
            }))
        if surface == "progress_form":
            body = kv_lines([
                ("Programme", f"{f['programme_id']} -- {f['science']}"), ("PI", f"{f['pi']} ({f['institution']})"),
                ("Instrument", f"{f['instrument']}, {f['telescope']} {f['aperture_m']} m"),
                ("Approved", f"{f['total_h']} h / {f['n_semesters']} semesters; {f['semesters_done']} completed"),
                ("Allocated to date", f"{f['allocated_h']} h"), ("Executed", f"{f['executed_h']} h"),
                ("Lost to weather/technical", f"{f['obs_loss_h']} h"),
                ("Not used (programme side)", f"{f['pi_side_unused_h']} h"
                 + (f" -- {f['delay_reason']}" if f["delay_reason"] else "")),
                ("Targets completed", f"{f['targets_done']} of {f['targets_total']}"),
                ("Data reduced", f"{f['reduced_pct']}%"),
                ("Refereed papers", f["papers"]), ("In preparation", f["papers_in_prep"]),
                ("Annual report", self._report_text(f)), ("Archive release", self._release_text(f)),
            ], rng)
            return (f"CONTINUATION REVIEW -- {f['observatory']}, {fmt_date(f['review_date'], rng)}\n{body}\n\n"
                    f"PI statement: \"{f['pi_claim']}\"\n\nRules:\n" + "\n".join(rules))
        para = [f"{f['programme_id']} ({f['pi']}, {f['institution']}) studies {f['science']} with "
                f"{f['instrument']} and was approved for {f['total_h']} h over {f['n_semesters']} semesters.",
                f"After {f['semesters_done']} semesters it has been allocated {f['allocated_h']} h, of which "
                f"{f['executed_h']} h were executed and {f['obs_loss_h']} h lost to weather or technical problems.",
                (f"The team explains the unused time: {f['delay_reason']}." if f["delay_reason"] else ""),
                f"{f['targets_done']} of {f['targets_total']} targets are complete and {f['reduced_pct']}% of the data "
                f"are reduced.",
                f"Output so far: {_pl(f['papers'], 'refereed paper')}, {f['papers_in_prep']} in preparation.",
                f"Annual report {self._report_text(f)}; {self._release_text(f)}.",
                f"The PI writes: \"{f['pi_claim']}\""]
        return (f"MEMO to the TAC chair -- continuation of {f['programme_id']}\n\n" + noisy(join_sentences(para), rng, 0.1)
                + "\n\nContinuation rules:\n" + bullets(rules, rng))

    def questions(self):
        return {
            "continuation_decision": choice("What should the committee decide about the programme's next allocation?", {
                "continue_as_planned": "Release the next allocation in full, as originally approved.",
                "continue_conditional": "Release the next allocation only once an outstanding obligation, such as a "
                                        "report or archive release, is met by a set date.",
                "continue_reduced": "Continue with fewer hours or a narrower target list that matches what the team has "
                                    "shown it can use and finish.",
                "suspend_one_semester": "Pause the programme for a semester until the team shows the data already taken "
                                        "are being used.",
                "terminate": "End the programme and return its remaining hours to the general pool.",
            }),
            "main_shortfall": choice("What is the most serious weakness in the programme's record so far?", {
                "low_time_use": "Too little of the allocated time was used, for reasons on the team's side.",
                "none_significant": "No significant weakness; the programme is broadly on course.",
                "reporting_lapses": "Progress reports or archive releases are late or missing.",
                "slow_publication": "Data are being taken but little has been reduced or published.",
                "unrealistic_completion": "The remaining allocation cannot plausibly finish the stated sample.",
            }),
            "productivity": score("How productive has the programme been with the time it has already used?", [
                "Poor: little reduced data and no publications to show for it.",
                "Below expectations: some output, but thin for the hours used.",
                "Solid: output in line with the hours used.",
                "Excellent: output well above what the hours used would suggest.",
            ]),
            "completion_claim_credibility": score("How credible is the claim that the programme will finish its "
                                                  "sample within the approved time?", [
                "Not credible: at the rate so far, far more time would be needed.",
                "Doubtful: it would need clearly better efficiency than achieved so far.",
                "Plausible: achievable at the current rate, with little to spare.",
                "Credible: the remaining time comfortably covers the remaining targets.",
            ]),
            "reporting_obligations_met": noul(
                "The programme is up to date with its reporting and archive-release obligations.",
                yes="The annual report arrived by its due date and no archive release is overdue.",
                no="The report is late or missing, or a required archive release is overdue."),
        }


WORKFLOW = Workflow(
    name="telescope_time_allocation",
    title="Telescope time allocation",
    description=("How a fictional observatory ranks proposals, checks technical feasibility, executes queue blocks, "
                 "rules on target-of-opportunity interrupts and renews multi-semester programmes, against written "
                 "TAC and operations rules."),
    domain="physics / astronomy / research operations",
    schemas=[ProposalTacRanking(), ExposureFeasibilityReview(), QueueBlockExecution(), TooInterruptRequest(),
             PriorUseContinuation()],
    notes=("Observatory research operations: committee rankings, technical reviews, night operations and programme "
           "renewals. No support threads, security alerts, bills or AI-agent runs. All facilities, instruments, "
           "programmes, people and rules are invented."),
)
