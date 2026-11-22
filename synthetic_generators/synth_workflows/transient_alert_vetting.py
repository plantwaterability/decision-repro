"""Transient alert vetting: astrophysical transients and gravitational-wave-style candidates, judged from written
pipeline outputs against written vetting rules.

A vetting shift reads what the pipelines wrote and decides what the source most likely is, whether to spend
telescope time or issue a notice, and how exposed a notice is to later retraction. The artefacts are an optical
survey alert with its light curve and cross-matches, a compact-binary candidate from an interferometer network
with its false-alarm rate, per-detector SNRs and data-quality flags, a high-energy neutrino track with a catalogued
source in its error region, a public alert that new information has undermined, and a candidate counterpart
inside a gravitational-wave sky map.

Everything is fictional: the surveys, the interferometer network and its sites, the neutrino telescopes, the
brokers, pipelines, catalogues, telescopes and instruments, every designation, every person and every number.
The numbers are kept physically plausible: false-alarm rates per year, per-detector SNRs that add in quadrature
to the network SNR, distance moduli from a flat Lambda-CDM cosmology, in-band signal durations from the chirp
mass, chance-coincidence counts from source densities times region areas. The vetting guides quoted in the
states are invented, and their thresholds VARY from case to case (survey, network, observing cycle), so every
answer needs the case's own rule.

Each schema draws a `regime` first. Borderline cases sit near the quoted threshold: a FAR within a factor of two
of the significant-alert line, a real-bogus score at the cut, a parallax near 3 sigma, a host's normalised
offset at the association limit, a source at the edge of the 90% region. Conflicting cases pair signals that
point different ways (a loud candidate with a glitch inside its in-band duration, a supernova-like light curve
on top of a catalogued star, a kilonova-like fade from a source seen before the merger, a shifter's confident
opinion against the evidence) or leave a key fact missing.

Every schema also defines `intended(f)`: the careful vetter's answer to each question from the factors. It is
never rendered; it exists for the answer-spread audit (no question may collapse onto one answer).
"""
from __future__ import annotations

import math
from datetime import timedelta

from .base import (
    Schema,
    Workflow,
    between,
    chance,
    choice,
    dumps,
    dumps_pretty,
    fmt_date,
    fmt_ts,
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
    rand_ts,
    regime,
    score,
    show,
    table,
)

# =========================================================================== shared pools (all fictional)
# (survey name, alert prefix, bands, single-epoch 5-sigma depth in mag)
SURVEYS = [
    ("Quillon Sky Survey", "QSS", ("g", "r"), 20.8), ("Halcyon Wide-Field Survey", "HWF", ("g", "r", "i"), 21.3),
    ("Corvid Transient Search", "CTS", ("g", "r"), 20.5), ("Sable Ridge Synoptic Survey", "SRS", ("g", "r", "i"), 21.6),
    ("Pampas Rapid Imaging Survey", "PRI", ("g", "r"), 20.9), ("Tarn Valley Deep Survey", "TVD", ("g", "r", "i"), 22.0),
    ("Northfell Survey Telescope", "NFT", ("g", "r"), 20.6), ("Kestrel Deep Synoptic Survey", "KDS", ("g", "r", "i"), 21.8),
    ("Lowmoor Twilight Survey", "LTS", ("g", "r"), 20.2), ("Saltire Southern Survey", "SSV", ("g", "r", "i"), 21.1),
    ("Wrenfield Transient Monitor", "WTM", ("g", "r"), 20.4), ("Obsidian Plains Survey", "OPS", ("g", "r", "i"), 21.4),
]
BROKERS = ["Lanternfish", "Wayfarer", "Cormorant", "Halyard", "Sextant", "Tessellate", "Driftnet", "Quern",
           "Plumbline", "Fieldfare", "Gannetry", "Lodestar"]
# (instrument on telescope, practical limiting magnitude for a classification spectrum)
SPECTROGRAPHS = [
    ("SERAC on the 2.6 m Halcon telescope", 19.6), ("MORAINE on the 3.5 m Ostrander telescope", 20.3),
    ("CIRQUE on the 4.1 m Norbacken telescope", 20.8), ("TARN on the 2.2 m Vientos telescope", 19.2),
    ("ESKER on the 1.8 m Calva telescope", 18.7), ("DRUMLIN on the 3.0 m Kaimalu telescope", 20.0),
    ("NUNATAK on the 2.4 m Quemado telescope", 19.4), ("ARETE on the 5.0 m Sarnath telescope", 21.2),
    ("CORRIE on the 3.6 m Brannock telescope", 20.5), ("SCREE on the 2.0 m Ilvaine telescope", 19.0),
]
GALAXY_CATS = ["QGC", "SWG", "HGS", "MRG", "PGX", "DGC2", "OWS", "KGS", "TLG", "BRG"]
STAR_CATS = ["HAC-3", "TSC2", "Varden Astrometric Catalogue", "OPC-II", "Lindqvist Parallax Survey"]
VAR_CATS = ["Varden Variable Index", "Thornbury Variable Catalogue", "Kelvane Variable Index", "Hollin Periodic Variables"]
# deep imagers that tile GW sky maps (practical single-visit depth in r)
TILERS = [("the wide-field imager on the 4.1 m Norbacken telescope", 23.4), ("the mosaic camera on the 2.6 m Halcon telescope", 22.6),
          ("GW tiling with the 3.5 m Ostrander telescope", 23.2), ("stacked visits of the Kessrin 1.2 m Schmidt array", 22.4),
          ("deep tiling with the 3.6 m Brannock telescope", 23.3), ("the prime-focus camera on the 5.0 m Sarnath telescope", 24.0)]
SHIFT_SIGNOFFS = ["vetting desk", "transient desk", "night vetter", "RRT", "alert desk"]

GW_NETWORKS = [("Ternion Gravitational-Wave Network", "TN"), ("Quadrant Interferometer Array", "QIA"),
               ("Harrowgate Interferometer Consortium", "HIC"), ("Longwave Strain Network", "LSN"),
               ("Equinox Detector Network", "EDN"), ("Meridional Baseline Array", "MBA")]
GW_SITES = [("AR", "Arvela"), ("BM", "Brenmoor"), ("KS", "Kessrin"), ("TV", "Talvara"), ("NH", "Norhavn"),
            ("OG", "Ostgaard")]
GW_PIPELINES = ["SPINDLE", "ORRERY", "LATTICE", "PELICAN", "CASCADE", "TOROID", "WAVERIDER", "SIEVE", "QUILL"]
GW_CYCLES = ["run C4", "run C5a", "run C5b", "cycle 7", "cycle 8", "run D2"]
DOWN_REASONS = ["maintenance", "lock loss", "high ground motion", "calibration", "commissioning"]
# glitch families: (description, duration range s, low-frequency range Hz, high-frequency range Hz)
GLITCHES = [
    ("short broadband blip", (0.05, 0.3), (25, 40), (200, 450)),
    ("scattered-light arches", (4.0, 40.0), (10, 16), (35, 70)),
    ("loud broadband burst", (0.4, 2.0), (18, 30), (600, 1500)),
    ("narrow-band whistle", (0.2, 1.0), (250, 400), (600, 1100)),
    ("ground-motion transient", (2.0, 12.0), (8, 12), (25, 40)),
    ("low-frequency knock", (0.3, 1.5), (15, 25), (60, 120)),
]

NU_OBS = [("Fathom Neutrino Observatory", "FNO"), ("Glacier Deep Array", "GDA"), ("Benthos Deep-Sea Telescope", "BDT"),
          ("Hadal Cherenkov Array", "HCA"), ("Rimeshelf Neutrino Telescope", "RNT")]
GAMMA_CATS = ["3LGC", "2KHX", "4QGS", "1HRC"]
RADIO_CATS = ["QRC", "TRS", "KRAS"]

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _letters(rng, n):
    return "".join(rng.choice(LETTERS) for _ in range(n))


def _survey_desig(rng, prefix):
    return f"{prefix}{rng.choice([25, 26, 27])}{_letters(rng, rng.choice([3, 3, 4]))}"


def _radec(rng):
    return rng.uniform(0, 360), math.degrees(math.asin(rng.uniform(-0.95, 0.85)))


def _jname(ra, dec, short=False):
    h = ra / 15.0
    hh, mm = int(h), int((h - int(h)) * 60)
    ss = min(59.99, ((h - hh) * 60 - mm) * 60)
    d = abs(dec)
    dd, dm = int(d), int((d - int(d)) * 60)
    ds = min(59.9, ((d - dd) * 60 - dm) * 60)
    sgn = "+" if dec >= 0 else "-"
    if short:
        return f"J{hh:02d}{mm:02d}.{int(ss / 6)}{sgn}{dd:02d}{dm:02d}"
    return f"J{hh:02d}{mm:02d}{ss:05.2f}{sgn}{dd:02d}{dm:02d}{ds:04.1f}"


def _galaxy_name(rng, ra, dec):
    cat = rng.choice(GALAXY_CATS)
    if chance(rng, 0.5):
        return f"{cat} {rng.randrange(10000, 2999999)}"
    return f"{cat} {_jname(ra + rng.uniform(-0.001, 0.001), dec)}"


def _sso_name(rng):
    return f"20{rng.choice([18, 19, 20, 21, 22, 23, 24, 25])} {rng.choice('ABCDEFGHJKLMNOPQRSTUVWXY')}" \
           f"{rng.choice('ABCDEFGHJKLMNOPQRSTUVWXYZ')}{rng.randint(2, 290)}"


def _dl_mpc(z):
    """Luminosity distance (Mpc), flat Lambda-CDM with H0 = 70, Omega_m = 0.3 (midpoint rule)."""
    n, s = 60, 0.0
    for i in range(n):
        zz = (i + 0.5) * z / n
        s += 1.0 / math.sqrt(0.3 * (1 + zz) ** 3 + 0.7)
    return (1 + z) * 299792.458 / 70.0 * s * z / n


def _z_of_dl(d):
    lo, hi = 0.0, 3.0
    for _ in range(45):
        mid = (lo + hi) / 2
        if _dl_mpc(mid) < d:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _mu(d_mpc):
    return 5 * math.log10(d_mpc * 1e5)


def _sig(x, n=2):
    return float(f"{x:.{n}g}")


def _far(x):
    return f"{x:.1e}"


def _per_text(per_year):
    """A rate per year written the way alert guides write it."""
    days = 365.25 / per_year
    if days >= 360:
        return f"1 per {round(days / 365.25)} yr" if days >= 700 else "1 per year"
    if days >= 1.5:
        return f"1 per {round(days)} days"
    return f"{round(per_year / 365.25)} per day" if per_year >= 700 else "1 per day"


def _first(name):
    return name.split()[0].lower()


def _chat(lines, t0, rng):
    out, t = [], t0
    for who, text in lines:
        t += timedelta(minutes=rng.randint(1, 9))
        out.append(f"[{t.strftime('%H:%M')}] {_first(who)}: {noisy(text, rng, 0.08)}")
    return "\n".join(out)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _t_before(f_hz, tau20):
    """Seconds before merger at which a leading-order chirp passes f_hz, given its time in band from 20 Hz."""
    return tau20 * (f_hz / 20.0) ** (-8.0 / 3.0)


def _secs(t):
    """A duration in seconds, at a precision that suits its size."""
    t = abs(t)
    if t >= 1:
        return f"{t:.1f} s"
    if t >= 0.01:
        return f"{t:.2f} s"
    return f"{t * 1000:.1f} ms"


def _f_at(t_before, tau20):
    return 20.0 * (tau20 / t_before) ** 0.375


# =========================================================================== schema 1
class OpticalSurveyAlert(Schema):
    name = "optical_survey_alert"
    description = ("A new transient from a wide-field optical survey: what kind of source it is, and whether it "
                   "earns follow-up under the programme's written vetting rules.")
    surfaces = {"json": 0.5, "broker_packet": 0.25, "vetting_chat": 0.25}

    # ------------------------------------------------------------------ sampling
    def _extragalactic(self, rng, seed, reg, f):
        youth, depth, spec_lim = f["youth_days"], f["depth"], f["spec_limit"]
        if seed == "nuclear":
            phase = "slow"
            M, gr, rate = between(rng, -20.6, -18.6), between(rng, -0.45, -0.18), between(rng, 0.01, 0.1)
            z = between(rng, 0.03, 0.15, digits=4)
            htype = pick_w(rng, {"spiral": 2, "elliptical": 2, "lenticular": 1, "barred spiral": 1})
            f["prior_flares"] = pick(rng, [0, 0, 1, 2, 3])
            t_latest = between(rng, 0.6, youth) if chance(rng, 0.3) else between(rng, youth + 0.5, youth + 14)
        else:
            phase = pick_w(rng, {"young": 4, "prepeak": 3, "fading": 3})
            if seed == "sn_ia":
                htype = pick_w(rng, {"spiral": 4, "barred spiral": 2, "elliptical": 2, "lenticular": 1, "dwarf": 1})
                z = between(rng, 0.02, 0.11, digits=4)
                if phase == "young":
                    M, gr, rate = between(rng, -18.3, -17.0), between(rng, 0.0, 0.35), between(rng, 0.25, 0.6)
                elif phase == "prepeak":
                    M, gr, rate = between(rng, -19.5, -18.6), between(rng, -0.12, 0.15), between(rng, 0.05, 0.2)
                else:
                    M, gr, rate = between(rng, -19.2, -18.4), between(rng, 0.2, 0.9), -between(rng, 0.03, 0.09)
            else:
                htype = pick_w(rng, {"spiral": 4, "barred spiral": 2, "irregular": 2, "dwarf": 2})
                z = between(rng, 0.006, 0.06, digits=4)
                if phase == "young":
                    M, gr, rate = between(rng, -17.6, -16.0), between(rng, -0.4, -0.1), between(rng, 0.25, 0.7)
                elif phase == "prepeak":
                    M, gr, rate = between(rng, -18.0, -16.0), between(rng, -0.3, 0.05), between(rng, 0.05, 0.2)
                else:
                    M, gr, rate = between(rng, -17.6, -15.5), between(rng, 0.1, 0.6), -between(rng, 0.02, 0.07)
            t_latest = between(rng, 0.6, youth) if phase == "young" else between(rng, youth + 0.6, youth + 15)
        offset_scale = None
        m_target = None
        knobs = []
        if reg == "borderline":
            pool = ["youth", "dlr", "speclim", "season"] + (["absmag", "absmag"] if seed != "nuclear" else ["nucoffset"])
            knobs = [pick(rng, pool)] + ([pick(rng, pool)] if chance(rng, 0.4) else [])
            if "absmag" in knobs:
                M = near(rng, pick(rng, [-18.7, -17.8]), 0.3)
                gr = near(rng, -0.05, 0.12)
                htype = pick(rng, ["spiral", "barred spiral"])
            if "youth" in knobs:
                t_latest = near(rng, youth, 0.6, lo=0.4)
            if "dlr" in knobs:
                offset_scale = near(rng, f["dlr_max"], 0.35, lo=0.3)
            if "speclim" in knobs:
                m_target = near(rng, spec_lim, 0.3)
            if "season" in knobs:
                f["season_left"] = near(rng, 10, 3, lo=2, integer=True)
            if "nucoffset" in knobs:
                f["prior_flares"] = 0
        f["knobs"] = knobs
        m = M + _mu(_dl_mpc(z))
        if m_target is None and (m > depth - 0.15 or m < 16.3):
            m_target = depth - between(rng, 0.2, 1.4) if m > depth - 0.15 else between(rng, 16.4, 17.6)
        if m_target is not None:
            m_target = min(m_target, depth - 0.1)
            z = round(_clamp(_z_of_dl(10 ** ((m_target - M) / 5) / 1e5), 0.003, 0.35), 4)
            m = M + _mu(_dl_mpc(z))
        size = round(_clamp(between(rng, 3, 10) * (0.04 / z) ** 0.7, 1.2, 45), 1)
        if seed == "nuclear":
            offset = between(rng, 0.03, 0.35)
            if "nucoffset" in knobs:
                offset = near(rng, 0.5, 0.15, lo=0.1)
        else:
            d = offset_scale if offset_scale is not None else (
                between(rng, 0.1, 0.8 * f["dlr_max"]) if reg == "clear" else between(rng, 0.1, f["dlr_max"] + 1.2))
            offset = round(d * size, 2)
        zkind = pick_w(rng, {"spec": 4, "photo": 5})
        f["host"] = {"name": _galaxy_name(rng, f["ra"], f["dec"]), "type": htype, "z": z, "z_kind": zkind,
                     "z_err": None if zkind == "spec" else round(between(rng, 0.006, 0.02) * (1 + z), 3),
                     "size_arcsec": size, "offset_arcsec": round(offset, 2)}
        f.update(phase=phase, gr=round(gr, 2), rate=round(rate, 3), t_latest=round(t_latest, 2), m_latest=round(m, 2),
                 true_abs_mag=round(M, 2))
        f["rb"] = between(rng, max(f["rb_cut"] + 0.08, 0.62), 0.99)
        if reg == "clear":
            f["season_left"] = rng.randint(25, 150)
        if reg == "conflicting":
            pool = ["star_lowplx", "sso_multi", "rival", "rb_low_multi", "missing_z", "missing_colour", "opinion"]
            f["decoys"] = [pick(rng, pool)]
            if chance(rng, 0.45):
                extra = pick(rng, pool)
                if extra not in f["decoys"]:
                    f["decoys"].append(extra)
        elif reg == "borderline" and chance(rng, 0.25):
            f["decoys"] = [pick(rng, ["rival", "opinion"])]
        for dcy in f["decoys"]:
            if dcy == "star_lowplx":
                f["star"] = {"catalogue": pick(rng, STAR_CATS), "sep_arcsec": between(rng, 0.2, 0.9),
                             "parallax_mas": between(rng, 0.1, 0.9), "plx_snr": between(rng, 0.3, 2.4),
                             "g_mag": between(rng, 20.5, 22.8)}
            elif dcy == "sso_multi":
                f["sso"] = {"name": _sso_name(rng), "sep_arcsec": between(rng, 0.5, f["sso_radius"]),
                            "pred_mag": round(f["m_latest"] + rng.gauss(0, 0.4), 1)}
                f["force_multi"] = True
            elif dcy == "rival":
                h = f["host"]
                dn = h["offset_arcsec"] / h["size_arcsec"]
                rsize = round(between(rng, 1.2, 8.0), 1)
                f["rival"] = {"name": _galaxy_name(rng, f["ra"], f["dec"]), "size_arcsec": rsize,
                              "offset_arcsec": round(max(2.0, between(rng, 0.6, 1.9) * dn * rsize), 2)}
            elif dcy == "rb_low_multi":
                f["rb"] = between(rng, f["rb_cut"] - 0.25, f["rb_cut"] - 0.02)
                f["force_multi"] = True
            elif dcy == "missing_z":
                f["host"]["z"] = None
            elif dcy == "missing_colour":
                f["gr"] = None
        if f["rival"] is None and chance(rng, 0.3):
            # a harmless second galaxy, far out in normalised terms
            h = f["host"]
            dn = max(0.2, h["offset_arcsec"] / h["size_arcsec"])
            rsize = round(between(rng, 1.0, 4.0), 1)
            f["rival"] = {"name": _galaxy_name(rng, f["ra"], f["dec"]), "size_arcsec": rsize,
                          "offset_arcsec": round(between(rng, 3.0, 6.0) * dn * rsize + 2.0, 2)}

    def _galactic(self, rng, reg, f):
        q = between(rng, 19.0, 22.8)
        amp = between(rng, 2.0, 5.5)
        rising = chance(rng, 0.5)
        f.update(m_latest=round(max(15.0, q - amp), 2), gr=round(between(rng, -0.5, 0.15), 2),
                 rate=round(between(rng, 1.2, 4.5) if rising else -between(rng, 1.0, 3.0), 2),
                 t_latest=between(rng, 0.3, 8.0), phase="outburst", rb=between(rng, max(f["rb_cut"] + 0.05, 0.6), 0.99))
        if chance(rng, 0.85):
            f["star"] = {"catalogue": pick(rng, STAR_CATS), "sep_arcsec": between(rng, 0.05, 0.45),
                         "parallax_mas": between(rng, 0.4, 6.0), "plx_snr": between(rng, 5, 40), "g_mag": round(q, 2)}
        knobs = []
        if reg == "borderline":
            knobs = [pick(rng, ["plx", "plx", "rate", "rb"])]
            if "plx" in knobs and f["star"]:
                f["star"]["plx_snr"] = near(rng, 3.0, 1.0, lo=0.4)
                f["star"]["parallax_mas"] = round(f["star"]["plx_snr"] * between(rng, 0.12, 0.4), 2)
            if "rate" in knobs:
                f["rate"] = round(near(rng, 1.0, 0.3, lo=0.4) * (1 if f["rate"] > 0 else -1), 2)
            if "rb" in knobs:
                f["rb"] = near(rng, f["rb_cut"], 0.05, hi=0.99)
        f["knobs"] = knobs
        if reg == "conflicting":
            f["decoys"] = [pick(rng, ["host_decoy", "host_decoy", "opinion"])]
            if chance(rng, 0.5) and f["star"]:
                f["star"]["plx_snr"] = between(rng, 0.5, 2.6)
                f["star"]["parallax_mas"] = round(f["star"]["plx_snr"] * between(rng, 0.15, 0.4), 2)
                f["decoys"].append("star_lowplx")
        if "host_decoy" in f["decoys"] or (reg == "borderline" and chance(rng, 0.25)):
            z = between(rng, 0.02, 0.12, digits=4)
            size = round(between(rng, 1.5, 5.0), 1)
            f["host"] = {"name": _galaxy_name(rng, f["ra"], f["dec"]), "type": pick(rng, ["spiral", "barred spiral", "dwarf"]),
                         "z": z, "z_kind": "photo", "z_err": round(between(rng, 0.008, 0.02), 3),
                         "size_arcsec": size, "offset_arcsec": round(between(rng, 0.9, 2.6) * size, 2)}

    def _moving(self, rng, reg, f):
        kind = pick_w(rng, {"asteroid": 5, "artefact": 4})
        f.update(m_latest=between(rng, 18.0, f["depth"] - 0.1), rate=0.0, phase=kind, t_latest=between(rng, 0.7, 6.0),
                 single_night=True, n_per_night=pick(rng, [1, 2, 2, 3]))
        f["gr"] = round(between(rng, 0.35, 0.75), 2) if kind == "asteroid" and chance(rng, 0.6) else None
        knobs = []
        if kind == "asteroid":
            r = f["sso_radius"]
            sep = between(rng, 0.2, 0.6 * r)
            if reg == "borderline":
                sep, knobs = near(rng, r, 1.0, lo=0.2), ["sso_sep"]
            f["sso"] = {"name": _sso_name(rng), "sep_arcsec": round(sep, 1),
                        "pred_mag": round(f["m_latest"] + rng.gauss(0, 0.3), 1)}
            f["rb"] = between(rng, f["rb_cut"] + 0.02, 0.98)
            if f["n_per_night"] >= 2:
                f["scatter"] = between(rng, 0.8, 3.5)
        else:
            if reg == "clear":
                f["rb"] = between(rng, 0.05, f["rb_cut"] - 0.12)
            elif reg == "borderline":
                f["rb"], knobs = near(rng, f["rb_cut"], 0.04), ["rb"]
            else:
                f["rb"] = between(rng, f["rb_cut"] - 0.2, f["rb_cut"] - 0.01)
            if chance(rng, 0.6):
                f["bright_star"] = {"mag": between(rng, 7.5, 10.5, digits=1), "sep_arcsec": between(rng, 6, 28, digits=0)}
            if f["n_per_night"] >= 2:
                f["scatter"] = between(rng, 0.05, 0.3)
        f["knobs"] = knobs
        if reg == "conflicting" or (reg == "borderline" and chance(rng, 0.3)):
            f["decoys"] = ["host_decoy", "opinion"] if chance(rng, 0.6) else ["opinion"]
        if "host_decoy" in f["decoys"]:
            z = between(rng, 0.02, 0.1, digits=4)
            size = round(between(rng, 2.0, 8.0), 1)
            f["host"] = {"name": _galaxy_name(rng, f["ra"], f["dec"]), "type": pick(rng, ["spiral", "barred spiral"]),
                         "z": z, "z_kind": pick(rng, ["spec", "photo"]), "z_err": round(between(rng, 0.008, 0.02), 3),
                         "size_arcsec": size, "offset_arcsec": round(between(rng, 0.3, 1.4) * size, 2)}

    def _lightcurve(self, rng, f):
        bands = f["bands"]
        nights = [0.0]
        if not f.get("single_night"):
            want = pick(rng, [2, 3, 3, 4]) if f.get("force_multi") else pick(rng, [1, 2, 3, 3, 4])
            off = 0.0
            while len(nights) < want:
                off += pick(rng, [1, 1, 1, 2, 3]) + rng.uniform(-0.06, 0.06)
                if f["t_latest"] - off < 0.15:
                    break
                nights.append(off)
            if f.get("force_multi") and len(nights) < 2:
                f["t_latest"] = round(f["t_latest"] + 1.1, 2)
                nights.append(1.0 + rng.uniform(-0.05, 0.05))
        rows = []
        for k, off in enumerate(nights):
            per = f.get("n_per_night") if f.get("single_night") else (2 if chance(rng, 0.55) else 1)
            for j in range(per):
                t = f["t_latest"] - off - j * rng.uniform(0.012, 0.035)
                if t < 0.05:
                    continue
                band = bands[(k + j) % len(bands)]
                m = f["m_latest"] + f["rate"] * (f["t_latest"] - t)
                if band == "g" and f["gr"] is not None:
                    m += f["gr"]
                elif band == "i" and f["gr"] is not None:
                    m -= 0.5 * f["gr"]
                m += rng.gauss(0, 0.03)
                if m > f["depth"] + 0.1:
                    continue
                err = round(0.02 + 0.25 * 10 ** (0.4 * (m - f["depth"])), 2)
                rows.append([round(t, 2), band, round(m, 2), err])
        if not rows:
            rows = [[round(f["t_latest"], 2), "r", round(f["m_latest"], 2), 0.05]]
        rows.sort(key=lambda r: r[0])
        rows = rows[-6:]
        f["lc"] = rows
        f["n_det"] = len(rows)
        # nights counted from the night offsets that survived (rows within ~0.5 d of each other share a night)
        night_ids = []
        for r in rows:
            nid = round(f["t_latest"] - r[0])
            if nid not in night_ids:
                night_ids.append(nid)
        f["n_nights"] = 1 if f.get("single_night") else len(night_ids)
        if f["n_nights"] == 1 and f["n_det"] >= 2 and f.get("scatter") is None:
            f["scatter"] = between(rng, 0.05, 0.25)

    def sample_factors(self, rng):
        reg = regime(rng)
        sname, prefix, bands, depth = pick(rng, SURVEYS)
        spec_name, spec_lim = pick(rng, SPECTROGRAPHS)
        ra, dec = _radec(rng)
        seed = pick_w(rng, {"sn_ia": 24, "cc_sn": 24, "nuclear": 16, "galactic": 17, "moving_or_artefact": 19})
        f = {"regime": reg, "seed_class": seed, "survey": sname, "alert_id": _survey_desig(rng, prefix),
             "bands": list(bands), "depth": depth, "broker": pick(rng, BROKERS), "spectrograph": spec_name,
             "spec_limit": spec_lim, "ra": round(ra, 5), "dec": round(dec, 5),
             "rb_cut": pick(rng, [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]), "youth_days": pick(rng, [2, 3, 4]),
             "dlr_max": pick(rng, [1.5, 2.0, 2.5, 3.0]), "sso_radius": pick(rng, [3, 5, 8]),
             "season_left": rng.randint(3, 150), "history_years": between(rng, 1.5, 6.5, digits=1),
             "host": None, "rival": None, "star": None, "sso": None, "bright_star": None, "prior_flares": 0,
             "scatter": None, "decoys": [], "knobs": [], "vetter": person(rng), "rules_rev": f"{rng.randint(1, 4)}.{rng.randint(0, 9)}",
             "night": rand_ts(rng, hours=(1, 7))}
        if seed in ("sn_ia", "cc_sn", "nuclear"):
            self._extragalactic(rng, seed, reg, f)
        elif seed == "galactic":
            self._galactic(rng, reg, f)
        else:
            self._moving(rng, reg, f)
        self._lightcurve(rng, f)
        # the last non-detection: normally to survey depth; for a young nuclear flare (which rises slowly) the
        # previous visit was shallow, so its limit is only a little fainter than the first detection
        # a colour needs both bands; a positional scatter needs two detections
        if len({r[1] for r in f["lc"]}) < 2:
            f["gr"] = None
        if f["n_det"] < 2:
            f["scatter"] = None
        f["nondet_lim"] = round(f["depth"] + rng.uniform(-0.3, 0.2), 1)
        if seed == "nuclear":
            first = f["lc"][0]
            was = first[2] + max(0.0, f["rate"]) * first[0]
            f["nondet_lim"] = round(min(f["nondet_lim"], was + between(rng, 0.2, 0.8)), 1)
        f["nondet_poor"] = (pick(rng, ["poor transparency", "nucleus subtraction noise", "bright Moon nearby",
                                       "twilight", "thin cloud"])
                            if f["nondet_lim"] < f["depth"] - 0.6 else None)
        h = f["host"]
        f["abs_mag"] = (round(f["m_latest"] - _mu(_dl_mpc(h["z"])), 2) if h and h.get("z") else None)
        # the comment in the packet: right in clear cases, often wrong in conflicting ones
        cls = self._cls(f)
        if "opinion" in f["decoys"] or (reg == "conflicting" and chance(rng, 0.5)):
            others = [c for c in ["sn", "nuclear", "galactic", "moving", "artefact"] if not cls.startswith(c[:4])
                      and (f["host"] is not None or c not in ("sn", "nuclear"))]
            f["opinion"] = pick(rng, others)
        elif reg == "clear" or chance(rng, 0.5):
            f["opinion"] = {"sn_ia": "sn", "cc_sn": "sn", "nuclear": "nuclear", "galactic": "galactic",
                            "moving_or_artefact": "moving" if f["sso"] else "artefact"}[cls]
        else:
            f["opinion"] = "unsure"
        return f

    # ------------------------------------------------------------------ the careful vetter
    @staticmethod
    def _slope(f):
        """Visible brightening rate in mag/day (positive = brightening) from one band seen on two nights, or None."""
        best = None
        for band in f["bands"]:
            pts = [r for r in f["lc"] if r[1] == band]
            if len(pts) >= 2 and pts[-1][0] - pts[0][0] >= 0.5:
                sl = (pts[0][2] - pts[-1][2]) / (pts[-1][0] - pts[0][0])
                if best is None or abs(sl) > abs(best):
                    best = sl
        return best

    @classmethod
    def _cls(cls, f):
        if f["n_nights"] == 1:
            if f["sso"] and f["sso"]["sep_arcsec"] <= f["sso_radius"]:
                return "moving_or_artefact"
            if f["rb"] < f["rb_cut"]:
                return "moving_or_artefact"
            if f["scatter"] is not None and f["scatter"] >= 0.5:
                return "moving_or_artefact"
        sl = cls._slope(f)
        fast = sl is not None and abs(sl) >= 1.0
        st = f["star"]
        near_star = st is not None and st["sep_arcsec"] <= 1.0
        if near_star and st["plx_snr"] >= 3:
            return "galactic"
        if fast:
            return "galactic"
        h = f["host"]
        if h is None:
            if near_star:
                return "galactic"
            if f["n_nights"] == 1:
                return "moving_or_artefact"
            return "cc_sn" if (f["gr"] is not None and f["gr"] < -0.05) else "sn_ia"
        if h["offset_arcsec"] <= 0.5 and (f["prior_flares"] > 0 or (f["gr"] is not None and f["gr"] <= -0.15)):
            return "nuclear"
        if h["type"] in ("elliptical", "lenticular"):
            return "sn_ia"
        M = f["abs_mag"]
        young_rising = f["t_latest"] <= f["youth_days"] and f["rate"] > 0
        if M is not None and M <= -18.7:
            return "sn_ia"
        if M is not None and M >= -17.8 and not young_rising:
            return "cc_sn"
        if f["gr"] is None:
            return "sn_ia" if (M is not None and M < -18.25) else "cc_sn"
        return "cc_sn" if f["gr"] < -0.05 else "sn_ia"

    def _conf(self, f, cls):
        c = 3
        c -= min(2, len([d for d in f["decoys"] if d not in ("opinion", "rival")]))
        if f["n_det"] <= (2 if cls in ("sn_ia", "cc_sn", "nuclear") else 1):
            c -= 1
        h, st = f["host"], f["star"]
        if cls in ("sn_ia", "cc_sn"):
            early = h is not None and h["type"] in ("elliptical", "lenticular")
            if f["abs_mag"] is None or f["gr"] is None:
                c -= 1
            elif not early and -18.95 < f["abs_mag"] < -17.55 and abs(f["gr"] + 0.05) < 0.12:
                c -= 1
        elif cls == "galactic":
            if st is None or 2.0 <= st["plx_snr"] < 4.5:
                c -= 1
        elif cls == "moving_or_artefact":
            if abs(f["rb"] - f["rb_cut"]) < 0.05 or (f["sso"] and abs(f["sso"]["sep_arcsec"] - f["sso_radius"]) < 1.0):
                c -= 1
        elif cls == "nuclear" and f["prior_flares"] == 0:
            c -= 1
        return _clamp(c, 0, 3)

    def intended(self, f):
        cls = self._cls(f)
        conf = self._conf(f, cls)
        h = f["host"]
        secure = False
        if cls not in ("galactic", "moving_or_artefact") and h is not None:
            d = h["offset_arcsec"] / h["size_arcsec"]
            rv = f["rival"]
            secure = d <= f["dlr_max"] and not (rv and rv["offset_arcsec"] / rv["size_arcsec"] < 2 * d)
        young = f["t_latest"] <= f["youth_days"]
        bright = f["m_latest"] <= f["spec_limit"]
        sl = self._slope(f)
        rising = (sl > 0) if sl is not None else young
        fading = sl is not None and sl < 0
        if cls == "moving_or_artefact":
            fu = "no_followup"
        elif cls == "galactic":
            fu = "no_followup" if conf >= 2 else "await_survey_epochs"
        elif conf <= 1 or f["n_nights"] < 2:
            fu = "await_survey_epochs"
        elif bright and (young or cls == "nuclear"):
            fu = "spectrum_tonight"
        else:
            fu = "photometric_monitoring"
        if cls in ("moving_or_artefact", "galactic"):
            p = 0
        else:
            if cls == "nuclear":
                p = 2 if (young and rising and bright) else 1
            elif young and rising:
                p = 3 if bright else 2
            elif fading and f["m_latest"] > f["spec_limit"] - 0.5:
                p = 2
            else:
                p = 1
            if f["season_left"] <= 10:
                p = min(3, p + 1)
        cls_key = {"sn_ia": "thermonuclear_supernova", "cc_sn": "core_collapse_supernova",
                   "nuclear": "nuclear_transient", "galactic": "galactic_stellar_outburst",
                   "moving_or_artefact": "moving_object_or_artefact"}[cls]
        return {"likely_source_class": cls_key, "followup_call": fu, "classification_confidence": conf,
                "host_association_secure": "yes" if secure else "no", "observing_window_pressure": p}

    # ------------------------------------------------------------------ rendering
    def _rules(self, f):
        a = (f"promote only with real-bogus >= {f['rb_cut']:.2f} or detections on two or more nights; a one-night "
             f"candidate within {f['sso_radius']}\" of a predicted solar-system object, or whose same-night "
             f"positions differ by 0.5\" or more, is a moving object; a catalogued "
             f"counterpart counts as stellar when its parallax is significant at 3 sigma or more; accept a host when "
             f"offset / semi-major axis <= {f['dlr_max']} and no other galaxy has under twice that normalised offset")
        b = (f"classification spectra ({f['spectrograph']}) go to extragalactic candidates seen on two or more nights, "
             f"brighter than {f['spec_limit']} mag and either young (last non-detection <= {f['youth_days']} d before the "
             f"latest detection) or nuclear; other credible extragalactic candidates get photometric monitoring")
        return a, b

    def _note(self, f, rng):
        h = f["host"]
        hn = h["name"] if h else "the field"
        texts = {
            "sn": [f"Looks like a normal SN in {hn} to me.", "SN-like rise, I'd treat it as a supernova.",
                   f"Probably a supernova on the disc of {hn}.", "Classic SN shape so far."],
            "nuclear": ["Right on the nucleus, could be a TDE.", "Nuclear position, AGN flicker or TDE?",
                        "Sits in the core of the galaxy, worth a nuclear-transient flag."],
            "galactic": ["Point source underneath, I'd call it a stellar flare.", "Smells like a CV outburst.",
                         "Probably a Galactic star doing something."],
            "moving": ["Single night only, probably a rock.", "Might be an asteroid, check the motion.",
                       "Could be a slow mover."],
            "artefact": ["Subtraction residual?", "Dipole-ish residual in the difference image.",
                         "Possibly a bad pixel column or ghost."],
            "unsure": ["No strong view, flagging for the call.", "Not sure what this is yet.", "Leaving this one for the call."],
        }
        return noisy(pick(rng, texts[f["opinion"]]), rng, 0.15)

    def _xmatch_lines(self, f, rng):
        out = []
        h, rv, st, so, bs = f["host"], f["rival"], f["star"], f["sso"], f["bright_star"]
        if h:
            z = ("z unknown" if h["z"] is None else
                 (f"z_spec {h['z']:.4f}" if h["z_kind"] == "spec" else f"photo-z {h['z']:.3f} +/- {h['z_err']}"))
            out.append(f"host {h['name']} ({h['type']}): offset {h['offset_arcsec']}\", semi-major axis "
                       f"{h['size_arcsec']}\", {z}")
        else:
            out.append("no catalogued galaxy within 30\"")
        if rv:
            out.append(f"other galaxy {rv['name']}: offset {rv['offset_arcsec']}\", semi-major axis {rv['size_arcsec']}\"")
        if st:
            out.append(f"{st['catalogue']} source at {st['sep_arcsec']}\": g = {st['g_mag']}, parallax "
                       f"{st['parallax_mas']} mas ({st['plx_snr']:.1f} sigma)")
        else:
            out.append("no stellar counterpart catalogued within 2\"")
        if so:
            out.append(f"predicted solar-system object {so['name']} at {so['sep_arcsec']}\" (V ~ {so['pred_mag']})")
        if bs:
            out.append(f"V = {bs['mag']} star {int(bs['sep_arcsec'])}\" away")
        hist = (f"{f['prior_flares']} earlier flare{'s' if f['prior_flares'] > 1 else ''} at this position in {f['history_years']} yr of survey data"
                if f["prior_flares"] else f"no earlier detections in {f['history_years']} yr of survey data")
        out.append(hist)
        return out

    def render(self, f, rng, surface):
        rules_a, rules_b = self._rules(f)
        note = self._note(f, rng)
        lc = f["lc"]
        last = lc[-1]
        nondet_lim = f["nondet_lim"]
        poor = f" ({f['nondet_poor']})" if f["nondet_poor"] else ""
        if surface == "json":
            h, rv, st, so = f["host"], f["rival"], f["star"], f["sso"]
            return dumps(prune({
                "alert": {"id": f["alert_id"], "survey": f["survey"], "broker": f["broker"], "ra_deg": f["ra"],
                          "dec_deg": f["dec"], "real_bogus": round(f["rb"], 2), "n_detections": f["n_det"],
                          "nights_with_detections": f["n_nights"],
                          "position_scatter_arcsec": None if f["scatter"] is None else round(f["scatter"], 2)},
                "photometry": {"last_nondetection": {"days_before_latest": round(f["t_latest"], 2), "limit_mag": nondet_lim,
                                                     "conditions": f["nondet_poor"]},
                               "detections": [{"t_days": r[0], "band": r[1], "mag": r[2], "err": r[3]} for r in lc],
                               "latest_g_minus_r": f["gr"], "abs_mag_at_host_z": f["abs_mag"]},
                "crossmatch": {
                    "host_galaxy": None if h is None else {"name": h["name"], "type": h["type"], "offset_arcsec": h["offset_arcsec"],
                                                           "semi_major_arcsec": h["size_arcsec"], "redshift": h["z"],
                                                           "redshift_kind": h["z_kind"] if h["z"] is not None else None,
                                                           "redshift_err": h["z_err"] if h["z"] is not None else None},
                    "other_galaxy": None if rv is None else {"name": rv["name"], "offset_arcsec": rv["offset_arcsec"],
                                                             "semi_major_arcsec": rv["size_arcsec"]},
                    "stellar_counterpart": None if st is None else {"catalogue": st["catalogue"], "sep_arcsec": st["sep_arcsec"],
                                                                    "g_mag": st["g_mag"], "parallax_mas": st["parallax_mas"],
                                                                    "parallax_snr": round(st["plx_snr"], 1)},
                    "solar_system_object": None if so is None else {"designation": so["name"], "sep_arcsec": so["sep_arcsec"],
                                                                    "predicted_v": so["pred_mag"]},
                    "bright_star_nearby": None if f["bright_star"] is None else
                    f"V={f['bright_star']['mag']} at {int(f['bright_star']['sep_arcsec'])} arcsec",
                    "history": (f"{f['prior_flares']} earlier flare{'s' if f['prior_flares'] > 1 else ''} in "
                                f"{f['history_years']} yr" if f["prior_flares"]
                                else f"no earlier detections in {f['history_years']} yr")},
                "programme": {"vetting_rules": rules_a, "followup_rules": rules_b,
                              "nights_before_solar_conjunction": f["season_left"]},
                "vetter_comment": note,
            }))
        if surface == "broker_packet":
            rows = [f"  {'t(d)':>6}  band  mag     err", f"  {0.0:6.2f}  -     >{nondet_lim:.1f}   (last non-detection{poor})"]
            rows += [f"  {r[0]:6.2f}  {r[1]}     {r[2]:.2f}   {r[3]:.2f}" for r in lc]
            head = [f"=== {f['broker']} broker | {f['alert_id']} | {f['survey']} ===",
                    f"position  {f['ra']:.5f} {f['dec']:+.5f}",
                    f"rb        {f['rb']:.2f}",
                    f"detected  {f['n_det']}x on {f['n_nights']} night(s)"
                    + (f", positional scatter {f['scatter']:.2f}\"" if f["scatter"] is not None else "")]
            mid = [f"g-r (latest) {show(f['gr'], rng)}",
                   f"M at host z {show(f['abs_mag'], rng)}"]
            xm = ["xmatch:"] + [f"  {x}" for x in self._xmatch_lines(f, rng)]
            tail = [f"programme ({f['survey']} vetting rules v{f['rules_rev']}): {rules_a}.", f"follow-up: {rules_b}.",
                    f"field lost to the Sun in {f['season_left']} nights", f"comment: {note}"]
            return "\n".join(head + ["photometry (t since last non-detection):"] + rows + mid + xm + tail)
        # vetting_chat
        a, b = f["vetter"], person(rng)
        first = lc[0]
        lines = [(a, f"{f['alert_id']} ({f['survey']}, via {f['broker']}): rb {f['rb']:.2f}, {f['n_det']} det(s) on "
                     f"{f['n_nights']} night(s)" + (f", scatter {f['scatter']:.2f}\"" if f["scatter"] is not None else "")),
                 (a, f"last non-det >{nondet_lim:.1f}{poor} at {f['t_latest']:.1f} d before the latest point; "
                     + (f"{first[1]} {first[2]:.2f} at t={first[0]:.2f} d -> {last[1]} {last[2]:.2f} at t={last[0]:.2f} d"
                        if len(lc) > 1 else f"single detection {last[1]} {last[2]:.2f} at t={last[0]:.2f} d")
                     + (f", g-r {f['gr']:+.2f}" if f["gr"] is not None else ", no colour yet")),
                 (b, "xmatch: " + "; ".join(self._xmatch_lines(f, rng)))]
        if f["abs_mag"] is not None:
            lines.append((b, f"that puts it at M ~ {f['abs_mag']:.1f} if the host redshift is right"))
        lines.append((a, note))
        lines.append((b, f"rules for this field (v{f['rules_rev']}): {rules_a}"))
        lines.append((a, f"and {rules_b}. Field has {f['season_left']} nights before conjunction."))
        return f"#vetting-{f['survey'].split()[0].lower()}\n" + _chat(lines, f["night"], rng)

    def questions(self):
        return {
            "likely_source_class": choice("What is this transient most likely to be?", {
                "core_collapse_supernova": "A core-collapse supernova: a massive star exploding in a star-forming host, "
                                           "usually fainter than a thermonuclear event and blue when young.",
                "galactic_stellar_outburst": "An outburst from a star in our own Galaxy, such as a flare star or a "
                                             "cataclysmic variable.",
                "moving_object_or_artefact": "Not a stationary astrophysical transient: a solar-system object or a "
                                             "spurious image-subtraction residual.",
                "nuclear_transient": "A transient in the host's nucleus, such as a tidal disruption event or a "
                                     "flaring active nucleus.",
                "thermonuclear_supernova": "A thermonuclear (type Ia-like) supernova, peaking near absolute magnitude "
                                           "-19 in any type of host.",
            }),
            "followup_call": choice("What follow-up should the vetting shift assign to this candidate?", {
                "await_survey_epochs": "Spend no extra telescope time yet; let the survey's next visits confirm or "
                                       "clarify it.",
                "no_followup": "Spend no follow-up time on it under this programme.",
                "photometric_monitoring": "Put it on multi-band photometric monitoring, without a spectrum for now.",
                "spectrum_tonight": "Request a classification spectrum on tonight's schedule.",
            }),
            "classification_confidence": score("How firmly do the data support a single source class?", [
                "Guesswork: the data barely constrain what this is.",
                "Tentative: one class is favoured, but a rival class is nearly as plausible.",
                "Fair: one class is clearly favoured, with one loose end.",
                "Firm: independent lines of evidence all point to the same class.",
            ]),
            "host_association_secure": noul(
                "The transient can be securely tied to the catalogued host galaxy.",
                yes="It is an extragalactic source within the stated normalised-offset limit of that galaxy, with no "
                    "other galaxy at a comparable normalised offset.",
                no="It is stellar or spurious, has no catalogued host, sits beyond the offset limit, or another "
                   "galaxy is an equally good host."),
            "observing_window_pressure": score(
                "How soon does follow-up have to happen for this candidate to keep its scientific value?", [
                    "No deadline: nothing worth having is lost by waiting.",
                    "Within the next week or so: it evolves slowly or is already past its early phase.",
                    "Within a night or two, before the early phase passes or it fades toward the spectrograph's limit.",
                    "Tonight: it is in its first days and bright enough to catch, or its field is about to be lost "
                    "to the Sun.",
                ]),
        }


# =========================================================================== schema 2
class GwCandidateVetting(Schema):
    name = "gw_candidate_vetting"
    description = ("A compact-binary candidate from a gravitational-wave interferometer network: public notice or "
                   "not, retraction exposure, data-quality overlap and counterpart-search value.")
    surfaces = {"json": 0.5, "pipeline_log": 0.25, "rrt_chat": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        net, code = pick(rng, GW_NETWORKS)
        sites = rng.sample(GW_SITES, pick(rng, [3, 3, 4]))
        src = pick_w(rng, {"bns": 5, "nsbh": 3, "bbh": 2.5})
        trials = pick(rng, [2, 3, 4, 4, 5, 6])
        thr_days = pick(rng, [30, 30, 60, 100, 182])
        thr_sig = 365.25 / thr_days
        thr_low_txt, thr_low = pick(rng, [("2 per day", 730.5), ("1 per day", 365.25), ("1 per 2 days", 182.6)])
        single = chance(rng, {"clear": 0.0, "borderline": 0.15, "conflicting": 0.3}[reg])
        # trials-adjusted false-alarm rate, steered by regime
        band = None
        if reg == "clear":
            band = pick(rng, ["sig", "sig", "mid", "none"])
        elif reg == "borderline":
            band = pick_w(rng, {"edge_sig": 6, "edge_low": 3, "sig": 1})
        else:
            band = pick_w(rng, {"sig": 6, "mid": 3, "edge_sig": 1})
        if band == "sig":
            far_adj = thr_sig * 10 ** -between(rng, 1.0, 7.0)
        elif band == "mid":
            far_adj = 10 ** between(rng, math.log10(thr_sig) + 0.35, math.log10(thr_low) - 0.3)
        elif band == "none":
            far_adj = thr_low * 10 ** between(rng, 0.3, 1.3)
        elif band == "edge_sig":
            far_adj = thr_sig * 10 ** near(rng, 0.0, 0.3)
        else:
            far_adj = thr_low * 10 ** near(rng, 0.0, 0.25)
        far_raw = _sig(max(1e-8, far_adj / trials), 2)
        far_adj = far_raw * trials
        snr = _clamp(7.6 - 1.15 * math.log10(far_adj) + rng.gauss(0, 0.6), 6.0, 30.0)
        if far_adj < 1e-6:
            snr = between(rng, 13.0, 30.0)
        snr = round(snr, 1)
        # which sites were observing, and how the SNR splits between them (quadrature)
        n_obs = len(sites) if chance(rng, 0.55) else max(2, len(sites) - 1)
        if single:
            n_obs = pick(rng, [1, 2, 2])
        observing = sites[:n_obs]
        down = {s[0]: pick(rng, DOWN_REASONS) for s in sites[n_obs:]}
        if single:
            w = [1.0] + [between(rng, 0.005, 0.06) for _ in observing[1:]]
            tot = sum(w)
            det_snr = {s[0]: round(snr * math.sqrt(x / tot), 1) for s, x in zip(observing, w)}
        else:
            # per-detector SNRs add in quadrature to the network SNR; the second detector usually clears 4
            others = [between(rng, 1.5, max(1.6, min(5.0, 0.4 * snr))) for _ in observing[2:]]
            rem = snr ** 2 - sum(o ** 2 for o in others)
            half = math.sqrt(rem / 2)
            if half >= 4.1 and chance(rng, 0.85):
                s2 = between(rng, 4.05, half)
            else:
                s2 = half * between(rng, 0.7, 1.0)
            s1 = math.sqrt(max(0.0, rem - s2 ** 2))
            vals = [s1, s2] + others
            det_snr = {s[0]: round(v, 1) for s, v in zip(observing, vals)}
        contributing = [k for k, v in det_snr.items() if v >= 3.0] or [observing[0][0]]
        # source parameters
        if src == "bns":
            m1, m2 = between(rng, 1.25, 2.0), between(rng, 1.1, 1.6)
        elif src == "nsbh":
            m1, m2 = between(rng, 4.0, 14.0), between(rng, 1.15, 1.9)
        else:
            m1 = between(rng, 9.0, 60.0)
            m2 = between(rng, 5.0, m1)
        mtot = m1 + m2
        mc = (m1 * m2) ** 0.6 / mtot ** 0.2
        tau = 159.0 * (1.21 / mc) ** (5 / 3)
        f_hi = 1500 if src == "bns" else int(min(1400, 2 * 4400 / mtot))
        p_fore = between(rng, 5.0, 20.0)
        p_astro = p_fore / (p_fore + far_adj)
        split = {"bns": (between(rng, 0.85, 0.98), 0.0), "nsbh": (between(rng, 0.05, 0.25), 0.0),
                 "bbh": (0.0, between(rng, 0.0, 0.04))}[src]
        if src == "bns":
            pb, pn = split[0], (1 - split[0]) * 0.8
        elif src == "nsbh":
            pn, pb = 1 - split[0], split[0] * 0.2
        else:
            pb, pn = 0.0, split[1]
        pbbh = max(0.0, 1 - pb - pn)
        cls_p = {"BNS": round(p_astro * pb, 3), "NSBH": round(p_astro * pn, 3), "BBH": round(p_astro * pbbh, 3)}
        cls_p["Terrestrial"] = round(max(0.0, 1 - sum(cls_p.values())), 3)
        has_ns = round({"bns": between(rng, 0.97, 1.0), "nsbh": between(rng, 0.6, 0.99), "bbh": between(rng, 0.0, 0.05)}[src], 2)
        base_area = {1: 2600, 2: 450, 3: 80, 4: 45}[min(4, len(contributing))]
        area = int(_clamp(base_area * (12 / snr) ** 1.5 * math.exp(rng.gauss(0, 0.35)), 10, 5000))
        horizon = 150.0 * (mc / 1.2) ** (5 / 6)
        dist = int(_clamp(horizon * (10 / snr) * between(rng, 0.6, 1.4), 40, 3000))
        dist_err = int(dist * between(rng, 0.2, 0.4))
        # data-quality issue near the trigger
        glitch = None
        if chance(rng, {"clear": 0.4, "borderline": 0.65, "conflicting": 0.95}[reg]):
            gname, (dlo, dhi), (flo1, flo2), (fhi1, fhi2) = pick(rng, GLITCHES)
            gdur = between(rng, dlo, dhi)
            g_lo, g_hi = int(between(rng, flo1, flo2)), int(between(rng, fhi1, fhi2))
            place = pick_w(rng, {"clear": {"inside": 1, "away": 1},
                                 "borderline": {"edge": 3, "time_only": 2, "inside": 1, "away": 1},
                                 "conflicting": {"inside": 4, "time_only": 1, "edge": 1}}[reg])
            # the chirp's frequency range that the glitch band shares, if any
            lo, hi = max(g_lo, 20.0), min(g_hi, f_hi * 0.9)
            if place == "inside" and lo >= hi:
                place = "time_only"
            if place == "inside":
                dt = -_t_before(between(rng, lo, hi), tau)
            elif place == "time_only":
                # inside the in-band duration, but the chirp is at another frequency at that moment
                if g_lo > 30:
                    ft = between(rng, 20.5, min(g_lo * 0.7, f_hi * 0.6))
                else:
                    ft = between(rng, min(g_hi * 1.5, f_hi * 0.8), f_hi * 0.85) if g_hi * 1.5 < f_hi * 0.85 else 20.5
                dt = -_t_before(ft, tau)
            elif place == "away":
                dt = pick(rng, [-tau - gdur / 2 - between(rng, 4, 60), between(rng, 2.0, 40.0) + gdur / 2])
            else:
                # near an edge: just before the chirp enters the band, or where its frequency meets the glitch band
                if chance(rng, 0.5) or lo >= hi:
                    dt = -tau - gdur / 2 + near(rng, 0.0, max(0.6, 0.12 * tau))
                else:
                    edge_f = lo if lo > 21 else hi
                    dt = -_t_before(edge_f, tau) + near(rng, 0.0, max(0.05, 0.4 * gdur + 0.1 * _t_before(edge_f, tau)))
            gdet = pick(rng, observing)[0]
            glitch = {"det": gdet, "kind": gname, "dt_s": round(dt, 4 if abs(dt) < 1 else 1), "dur_s": round(gdur, 2),
                      "f_lo": g_lo, "f_hi": g_hi, "place": place,
                      "subtracted": chance(rng, 0.25), "recovered_snr": None}
            if glitch["subtracted"]:
                glitch["recovered_snr"] = round(snr * between(rng, 0.9, 1.02), 1)
        t0 = rand_ts(rng)
        opinion = pick(rng, ["clean", "worried_dq", "worried_single", "excited", "neutral"])
        if opinion == "worried_dq" and glitch is None:
            opinion = "neutral"
        if reg == "conflicting":
            opinion = "clean" if glitch else "excited"
        return {"regime": reg, "network": net, "net_code": code, "sites": [s[0] for s in sites],
                "site_names": {s[0]: s[1] for s in sites}, "observing": [s[0] for s in observing], "down": down,
                "pipeline": pick(rng, GW_PIPELINES), "cycle": pick(rng, GW_CYCLES), "source_seed": src,
                "trials": trials, "thr_days": thr_days, "thr_low_text": thr_low_txt, "thr_low": thr_low,
                "far_raw": far_raw, "band": band, "snr": snr, "det_snr": det_snr, "single_detector": single,
                "contributing": contributing, "m1": round(m1, 2), "m2": round(m2, 2), "chirp_mass": round(mc, 3),
                "in_band_s": round(tau, 1) if tau >= 1 else round(tau, 2), "f_hi": f_hi, "p_astro": round(p_astro, 3),
                "class_probs": cls_p, "has_ns": has_ns, "area90": area, "distance": dist, "distance_err": dist_err,
                "glitch": glitch, "t0": t0, "opinion": opinion,
                "event_id": f"{code}{t0.strftime('%y%m%d')}{_letters(rng, 2)}", "shifter": person(rng)}

    @staticmethod
    def _overlap(f):
        """Does the glitch's time-frequency box touch the chirp track?"""
        g = f["glitch"]
        if not g:
            return False
        tau = f["in_band_s"]
        t_lo, t_hi = g["dt_s"] - g["dur_s"] / 2, g["dt_s"] + g["dur_s"] / 2
        if t_hi < -tau or t_lo > 0.0:
            return False
        f_a = _f_at(min(tau, -t_lo), tau) if t_lo < 0 else f["f_hi"]
        f_b = f["f_hi"] if t_hi >= -1e-3 else min(f["f_hi"], _f_at(-t_hi, tau))
        return f_a <= g["f_hi"] and f_b >= g["f_lo"]

    @staticmethod
    def _single(f):
        return sum(1 for v in f["det_snr"].values() if v >= 4.0) == 1

    def intended(self, f):
        far_adj = f["far_raw"] * f["trials"]
        thr_sig = 365.25 / f["thr_days"]
        sig, low = far_adj < thr_sig, far_adj < f["thr_low"]
        ov = self._overlap(f)
        blocking = ov and not f["glitch"]["subtracted"]
        if not low:
            nd = "no_public_notice"
        elif blocking:
            nd = "hold_for_vetting"
        elif sig and self._single(f):
            nd = "hold_for_vetting"
        elif sig:
            nd = "issue_significant_notice"
        else:
            nd = "issue_low_significance_notice"
        lm = math.log10(far_adj / thr_sig)
        if not low:
            r = 4 if far_adj > 3 * f["thr_low"] else 3
        elif lm >= 0:
            r = 2 if lm < 0.3 else 3
        else:
            r = 2 if lm > -0.3 else (1 if lm > -3 else 0)
        if self._single(f):
            r = max(r + 1, 2)
        if blocking:
            r = max(r + 1, 3)
        elif ov:
            r = max(r, 1)
        r = min(4, r)
        if f["has_ns"] < 0.1 or f["p_astro"] < 0.2:
            v = 0
        else:
            v = 3 if (f["area90"] <= 250 and f["distance"] <= 250) else (
                2 if (f["area90"] <= 1000 and f["distance"] <= 500) else 1)
            if blocking:
                v = max(0, v - 1)
        return {"notice_decision": nd, "retraction_likelihood": r, "significant_threshold_met": "yes" if sig else "no",
                "glitch_overlaps_signal": "yes" if ov else "no", "counterpart_search_value": v}

    def _guide(self, f):
        return (f"{f['network']} alert guide ({f['cycle']}): significant public alert if FAR x {f['trials']} (trials "
                f"factor) is below {_per_text(365.25 / f['thr_days'])}; low-significance public alert if FAR x "
                f"{f['trials']} is below {f['thr_low_text']}; a candidate with SNR 4 or more in only one detector is never "
                f"announced as significant without human vetting; any candidate with a data-quality issue overlapping the signal "
                f"in time and frequency is held for vetting unless the issue was subtracted and the candidate recovered.")

    @staticmethod
    def _track(f):
        tau = f["in_band_s"]
        fs = [x for x in (30, 50, 80, 150, 300, 600, 1000) if x < 0.8 * f["f_hi"]]
        if len(fs) > 4:
            fs = [fs[round(i * (len(fs) - 1) / 3)] for i in range(4)]
        pts = []
        for x in fs:
            t = _t_before(x, tau)
            pts.append(f"{x} Hz at -{_secs(t)}")
        return ", ".join(pts) if pts else "merger within a second of entering the band"

    def _dq_text(self, f):
        g = f["glitch"]
        if not g:
            return None
        rel = f"{_secs(g['dt_s'])} {'before' if g['dt_s'] < 0 else 'after'} merger"
        s = (f"{g['det']}: {g['kind']}, {g['f_lo']}-{g['f_hi']} Hz, centred {rel}, lasting {g['dur_s']} s")
        if g["subtracted"]:
            s += f"; removed by glitch subtraction, candidate recovered at network SNR {g['recovered_snr']}"
        else:
            s += "; not subtracted"
        return s

    def _opinion(self, f, rng):
        return noisy(pick(rng, {
            "clean": ["Data look clean to me, I'd send it.", "Nothing odd in the spectrograms.", "Looks like a solid one."],
            "worried_dq": ["I don't love the DQ near this one.", "Can someone look at that detector before we send anything?"],
            "worried_single": ["Mostly one detector carrying this, careful.", "The SNR is lopsided between sites."],
            "excited": ["This is a nice one, let's move.", "Observers will want this fast."],
            "neutral": ["Following the guide on this one.", "Over to the guide."],
        }[f["opinion"]]), rng, 0.1)

    def render(self, f, rng, surface):
        guide = self._guide(f)
        dq = self._dq_text(f)
        snrs = ", ".join(f"{k} {v}" for k, v in f["det_snr"].items())
        status = {s: ("observing" if s in f["observing"] else f"not observing ({f['down'][s]})") for s in f["sites"]}
        dur = f"{f['in_band_s']} s from 20 Hz to merger, up to ~{f['f_hi']} Hz; track: {self._track(f)}"
        if surface == "json":
            return dumps(prune({
                "candidate": {"id": f["event_id"], "network": f["network"], "pipeline": f["pipeline"],
                              "merger_time_utc": fmt_ts(f["t0"], style="iso"), "far_per_year": f["far_raw"],
                              "trials_factor": f["trials"], "network_snr": f["snr"], "detector_snr": f["det_snr"]},
                "detectors": status,
                "source": {"p_astro": f["p_astro"], "classification": f["class_probs"], "has_ns": f["has_ns"],
                           "chirp_mass_msun": f["chirp_mass"], "signal_in_band": dur},
                "skymap": {"area90_deg2": f["area90"], "distance_mpc": f["distance"], "distance_err_mpc": f["distance_err"]},
                "data_quality": {"issues": [dq] if dq else [], "summary": None if dq else "no flags within +/-60 s"},
                "guide": guide,
                "shifter_note": self._opinion(f, rng),
            }))
        if surface == "pipeline_log":
            ts = fmt_ts(f["t0"], style="log")
            cp = ", ".join(f"{k} {v}" for k, v in f["class_probs"].items())
            lines = [f"{ts} [{f['pipeline']}] candidate {f['event_id']} ({f['network']})",
                     f"  FAR {_far(f['far_raw'])} /yr  ({_far(f['far_raw'] / 3.156e7)} Hz)   network SNR {f['snr']}   [{snrs}]",
                     f"  chirp mass {f['chirp_mass']} Msun   in band: {dur}",
                     f"  p_astro {f['p_astro']}  ({cp})  HasNS {f['has_ns']}",
                     f"  skymap: 90% area {f['area90']} deg2, distance {f['distance']} +/- {f['distance_err']} Mpc",
                     "  detectors: " + "; ".join(f"{k} {v}" for k, v in status.items()),
                     "  DQ: " + (dq if dq else "no flags within +/-60 s of merger"),
                     f"  trials factor for this cycle: {f['trials']}",
                     "", "GUIDE: " + guide]
            return "\n".join(lines)
        a, b, c = f["shifter"], person(rng), person(rng)
        lines = [(a, f"{f['event_id']} from {f['pipeline']}: FAR {_far(f['far_raw'])}/yr, trials factor {f['trials']}, "
                     f"network SNR {f['snr']} ({snrs})"),
                 (b, "detector states: " + "; ".join(f"{k} {v}" for k, v in status.items())),
                 (a, f"p_astro {f['p_astro']}, " + ", ".join(f"{k} {v}" for k, v in f["class_probs"].items())
                  + f", HasNS {f['has_ns']}, Mc {f['chirp_mass']}; {dur}"),
                 (c, f"skymap {f['area90']} deg2 at {f['distance']} +/- {f['distance_err']} Mpc"),
                 (b, ("DQ flag: " + dq) if dq else "no DQ flags around the time"),
                 (c, self._opinion(f, rng)),
                 (a, "pasting the guide: " + guide)]
        return f"#rrt-{f['net_code'].lower()}\n" + _chat(lines, f["t0"], rng)

    def questions(self):
        return {
            "notice_decision": choice("What should the rapid-response team do about a public notice for this candidate?", {
                "hold_for_vetting": "Send nothing public yet; hold the candidate for a human data-quality and detector "
                                    "check first.",
                "issue_low_significance_notice": "Issue a public notice labelled low-significance, with its caveats "
                                                 "stated.",
                "issue_significant_notice": "Issue a public notice as a significant candidate now.",
                "no_public_notice": "Issue no public notice; the candidate meets no public-alert criterion.",
            }),
            "retraction_likelihood": score(
                "If a public notice went out for this candidate, how likely is it that it would later have to be "
                "retracted?", [
                    "Very unlikely: far past the significance line, in clean data from several detectors.",
                    "Unlikely: comfortably significant, with at most a minor caveat.",
                    "Possible: close to the line, or carrying one real data-quality or single-detector caveat.",
                    "Likely: short of the significant line, or significant only in data with a problem on top of it.",
                    "Very likely: a background-level candidate, or one resting on compromised data.",
                ]),
            "significant_threshold_met": noul(
                "After the trials factor, the candidate's false-alarm rate is below the network's significant-alert "
                "threshold.",
                yes="The pipeline FAR multiplied by the stated trials factor is below the significant-alert rate in "
                    "the quoted guide.",
                no="Once multiplied by the trials factor, the FAR is at or above that rate."),
            "glitch_overlaps_signal": noul(
                "A data-quality problem in at least one detector overlaps the signal in both time and frequency."),
            "counterpart_search_value": score("How worthwhile is an electromagnetic counterpart search for this candidate?", [
                "Not worthwhile: no neutron star is likely involved, or the candidate is probably terrestrial.",
                "Marginal: a counterpart is possible, but the sky area or distance makes a search a long shot.",
                "Worthwhile: a neutron star is likely, and the area and distance suit wide-field instruments.",
                "Compelling: a likely neutron-star merger, nearby and well localised.",
            ]),
        }


# =========================================================================== schema 3
class NeutrinoTrackCoincidence(Schema):
    name = "neutrino_track_coincidence"
    description = ("A high-energy neutrino track alert with a catalogued source near its error region: association "
                   "strength, localisation quality, and how to spend follow-up time and word the circular.")
    surfaces = {"json": 0.45, "alert_notice": 0.3, "desk_table": 0.25}

    KINDS = {
        # kind: (label, density range per deg2 of comparable sources)
        "blazar_flare": ("gamma-ray blazar", (0.003, 0.02)),
        "optical_sn": ("optical supernova from a survey", (0.15, 0.8)),
        "nuclear_flare": ("nuclear optical flare (TDE-like)", (0.002, 0.02)),
        "radio_agn": ("radio-bright active nucleus", (0.25, 1.2)),
    }

    def sample_factors(self, rng):
        reg = regime(rng)
        obs, code = pick(rng, NU_OBS)
        stream = pick_w(rng, {"high-purity": 5, "broad": 4})
        sig_n = between(rng, 0.42, 0.9) if stream == "high-purity" else between(rng, 0.12, 0.5)
        energy = int(_clamp(180 * math.exp(rng.gauss(0, 0.6)), 60, 1500))
        topo = pick_w(rng, {"through-going track": 6, "starting track": 2, "track clipping the detector edge": 2})
        thr = pick(rng, [0.05, 0.1, 0.2])
        tile_max = pick(rng, [10, 20, 30])
        kind = pick_w(rng, {"blazar_flare": 4, "optical_sn": 3, "nuclear_flare": 2, "radio_agn": 2})
        refined = True
        density_mult = None
        if reg == "clear":
            strong = chance(rng, 0.55)
            radius = between(rng, 0.35, 1.2) if strong else between(rng, 0.5, 4.0)
            u = between(rng, 0.1, 0.6) if strong else between(rng, 1.4, 3.0)
            if strong:
                kind = pick(rng, ["blazar_flare", "nuclear_flare"])
            active = strong or chance(rng, 0.4)
            topo = "through-going track" if strong else topo
        elif reg == "borderline":
            radius = between(rng, 0.6, 3.6)
            if chance(rng, 0.42):
                u = near(rng, 1.0, 0.25, lo=0.2)
                refined = chance(rng, 0.55)
            else:
                u = between(rng, 0.15, 0.8)
                kind = pick(rng, ["blazar_flare", "nuclear_flare", "blazar_flare", "optical_sn"])
                density_mult = near(rng, 0.75, 0.45, lo=0.3)
            active = chance(rng, 0.75)
        else:
            radius = between(rng, 0.4, 3.0)
            refined = chance(rng, 0.7)
            twist = pick(rng, ["inactive_rare", "active_common", "outside_bright", "lowsig_perfect",
                               "aligned_but_doubted"])
            if twist == "inactive_rare":
                kind, u, active = pick(rng, ["blazar_flare", "nuclear_flare"]), between(rng, 0.1, 0.7), False
            elif twist == "active_common":
                kind, u, active = pick(rng, ["optical_sn", "radio_agn"]), between(rng, 0.1, 0.8), True
            elif twist == "outside_bright":
                kind, u, active = "blazar_flare", between(rng, 1.15, 1.8), True
            elif twist == "lowsig_perfect":
                kind, u, active = pick(rng, ["blazar_flare", "nuclear_flare"]), between(rng, 0.1, 0.6), True
                stream, sig_n = "broad", between(rng, 0.1, 0.28)
            else:
                kind, u, active = pick(rng, ["blazar_flare", "nuclear_flare"]), between(rng, 0.1, 0.7), True
            if twist in ("inactive_rare", "lowsig_perfect", "aligned_but_doubted"):
                density_mult = between(rng, 0.1, 0.6)
        if kind == "radio_agn" and reg != "conflicting":
            active = active and chance(rng, 0.3)
        label, (dlo, dhi) = self.KINDS[kind]
        density = between(rng, dlo, dhi, digits=4)
        if density_mult is not None:
            density = round(max(0.0005, density_mult * thr / (math.pi * radius ** 2)), 4)
        area = round(math.pi * radius ** 2, 1)
        ra, dec = _radec(rng)
        # the candidate source sits offset_deg from the best-fit direction, at a random position angle
        pa = rng.uniform(0, 2 * math.pi)
        sdec = _clamp(dec + u * radius * math.sin(pa), -89.9, 89.9)
        sra = (ra + u * radius * math.cos(pa) / max(0.05, math.cos(math.radians(sdec)))) % 360.0
        t0 = rand_ts(rng)
        # activity wording, by kind and whether the source was active at the arrival time
        if kind == "blazar_flare":
            if active:
                pre, post = rng.randint(3, 40), rng.randint(5, 60)
                act = (f"gamma-ray flux {between(rng, 3, 12, digits=1)}x its catalogue average from {pre} d before to "
                       f"{post} d after the arrival" + pick(rng, ["", " (still ongoing)"]))
            else:
                act = pick(rng, [f"last gamma-ray flare ended {rng.randint(60, 400)} d before the arrival; quiescent since",
                                 f"steady gamma-ray flux in the {rng.randint(1, 6)} months around the arrival"])
            name = f"{pick(rng, GAMMA_CATS)} {_jname(sra, sdec, short=True)}"
        elif kind == "optical_sn":
            if active:
                act = (f"type {pick(rng, ['II', 'IIn', 'Ic-BL', 'Ib'])} supernova at z = {between(rng, 0.01, 0.08, digits=3)}; "
                       f"estimated explosion {rng.randint(2, 25)} d before the arrival")
            else:
                act = (f"supernova at z = {between(rng, 0.01, 0.08, digits=3)}; estimated explosion "
                       f"{rng.randint(15, 90)} d after the arrival")
            name = f"{pick(rng, SURVEYS)[1]}{rng.choice([25, 26])}{_letters(rng, 3)}"
        elif kind == "nuclear_flare":
            if active:
                act = f"nuclear optical flare peaked {rng.randint(20, 160)} d before the arrival, still {between(rng, 1.2, 3.0, digits=1)} mag above baseline"
            else:
                act = f"nuclear flare peaked {between(rng, 1.5, 4.0, digits=1)} yr before the arrival; back at baseline"
            name = f"{pick(rng, SURVEYS)[1]}{rng.choice([22, 23, 24, 25])}{_letters(rng, 3)}"
        else:
            if active:
                act = (f"radio flux up {rng.randint(30, 160)}% in a flare that began {rng.randint(20, 200)} d before "
                       f"the arrival")
            else:
                act = pick(rng, [f"radio flux steady within 10% over {rng.randint(3, 9)} yr",
                                 f"radio flux steady within 15% in monitoring {rng.randint(1, 4)} months either side of the arrival"])
            name = f"{pick(rng, RADIO_CATS)} {_jname(sra, sdec)}"
        return {"regime": reg, "observatory": obs, "obs_code": code, "stream": stream,
                "signalness": maybe(rng, round(sig_n, 2), 0.08 if reg == "conflicting" else 0.0),
                "energy_tev": energy, "topology": topo, "radius90": round(radius, 2), "area90": area,
                "refined": refined, "refine_eta_h": rng.randint(2, 6), "kind": kind, "kind_label": label,
                "source_name": name, "offset_deg": round(u * radius, 2), "u": round(u, 3), "active": active,
                "activity": act, "density": density, "thr": thr, "tile_max": tile_max,
                "ra": round(ra, 2), "dec": round(dec, 2), "t0": t0,
                "alert_id": f"{code}-{t0.strftime('%y%m%d')}{pick(rng, ['A', 'A', 'B'])}", "desk": person(rng),
                "twist": twist if reg == "conflicting" else None,
                "show_nexp": chance(rng, 0.5)}

    @staticmethod
    def _derived(f):
        inside = f["offset_deg"] <= f["radius90"]
        nexp = f["density"] * f["area90"]
        s = f["signalness"]
        return inside, nexp, s

    def intended(self, f):
        inside, nexp, s = self._derived(f)
        u = f["offset_deg"] / f["radius90"]
        rare = nexp < f["thr"]
        if not inside:
            strength = 1 if (u <= 1.3 and f["active"] and not f["refined"]) else 0
        else:
            strength = 1 + int(f["active"]) + int(rare)
            if s is None or s < 0.3:
                strength -= 1
            if not f["refined"] and u > 0.75:
                strength = min(strength, 2)
        strength = _clamp(strength, 0, 3)
        chance_ok = not (inside and f["active"] and rare)
        r = f["radius90"]
        q = 3 if r <= 0.7 else (2 if r <= 1.5 else (1 if r <= 3.0 else 0))
        if not f["refined"]:
            q -= 1
        if "edge" in f["topology"]:
            q -= 1
        q = _clamp(q, 0, 3)
        edge = 0.75 <= u <= 1.3
        if not f["refined"] and (f["area90"] > f["tile_max"] or edge):
            plan = "await_refined_localisation"
        elif strength == 3 or (strength == 2 and rare and f["active"]):
            plan = "target_candidate_source"
        elif f["area90"] <= f["tile_max"] and (s is not None and s >= 0.3):
            plan = "tile_error_region"
        else:
            plan = "archival_search_only"
        if not f["refined"] and edge:
            word = "defer_statement"
        elif strength == 3:
            word = "claim_likely_counterpart"
        elif inside and (f["active"] or rare):
            word = "list_as_coincident"
        else:
            word = "omit_source"
        return {"association_strength": strength, "followup_plan": plan,
                "chance_alignment_plausible": "yes" if chance_ok else "no", "localisation_quality": q,
                "circular_wording": word}

    def _guide(self, f):
        return (f"{f['obs_code']} follow-up guide: name a source as a likely counterpart only if it lies inside the 90% "
                f"region, was active when the neutrino arrived, and fewer than {f['thr']} comparable sources are expected "
                f"in the region by chance; wide-field tiling is offered for 90% regions up to {f['tile_max']} deg2; "
                f"initial (unrefined) regions can move by about one radius once the refined scan is in.")

    def _loc_text(self, f):
        return ("refined scan" if f["refined"]
                else f"initial reconstruction; refined scan expected in ~{f['refine_eta_h']} h")

    def render(self, f, rng, surface):
        nexp = round(f["density"] * f["area90"], 3)
        sig = f["signalness"]
        if surface == "json":
            return dumps(prune({
                "alert": {"id": f["alert_id"], "observatory": f["observatory"], "stream": f["stream"],
                          "arrival_utc": fmt_ts(f["t0"], style="iso"), "signalness": sig, "energy_tev": f["energy_tev"],
                          "topology": f["topology"],
                          "localisation": {"ra_deg": f["ra"], "dec_deg": f["dec"], "radius90_deg": f["radius90"],
                                           "area90_deg2": f["area90"], "status": self._loc_text(f)}},
                "candidate_source": {"name": f["source_name"], "type": f["kind_label"], "offset_deg": f["offset_deg"],
                                     "activity": f["activity"],
                                     "comparable_density_per_deg2": f["density"],
                                     "expected_in_region_by_chance": nexp if f["show_nexp"] else None},
                "guide": self._guide(f),
                "desk_note": noisy(self._desk_note(f, rng), rng, 0.1),
            }))
        if surface == "alert_notice":
            lines = [f"{f['observatory'].upper()} TRACK ALERT {f['alert_id']} ({f['stream']} stream)",
                     f"Arrival: {fmt_ts(f['t0'], rng)} UTC",
                     f"Direction: RA {f['ra']:.2f}, Dec {f['dec']:+.2f} deg; 90% containment radius {f['radius90']} deg "
                     f"({f['area90']} deg2), {self._loc_text(f)}",
                     f"Signalness: {show(sig, rng)}   Estimated energy: {f['energy_tev']} TeV   Event: {f['topology']}",
                     "",
                     f"Follow-up desk note: {f['source_name']} ({f['kind_label']}) lies {f['offset_deg']} deg from the "
                     f"best-fit direction. Activity: {f['activity']}. Sky density of comparable sources: "
                     f"{f['density']} per deg2" + (f" (about {nexp} expected in the region)." if f["show_nexp"] else "."),
                     "", self._guide(f), "", noisy(self._desk_note(f, rng), rng, 0.1)]
            return "\n".join(lines)
        rows = [["alert", f"{f['alert_id']} ({f['observatory']})"], ["stream / signalness", f"{f['stream']} / {show(sig, rng)}"],
                ["energy", f"{f['energy_tev']} TeV"], ["event", f["topology"]],
                ["90% radius / area", f"{f['radius90']} deg / {f['area90']} deg2"], ["localisation", self._loc_text(f)],
                ["candidate", f"{f['source_name']} ({f['kind_label']})"], ["offset from best fit", f"{f['offset_deg']} deg"],
                ["activity", f["activity"]], ["comparable sources", f"{f['density']} per deg2"]]
        if f["show_nexp"]:
            rows.append(["expected in region", str(nexp)])
        return "\n".join([f"Follow-up desk sheet, {fmt_date(f['t0'].date(), rng)}", "", table(["item", "value"], rows, rng),
                          "", self._guide(f), "", f"{f['desk'].split()[0]}: " + noisy(self._desk_note(f, rng), rng, 0.1)])

    def _desk_note(self, f, rng):
        if f["twist"] == "aligned_but_doubted":
            return pick(rng, ["Probably nothing, these turn up in every region.",
                              "I wouldn't read much into it, we see a source like this near most alerts.",
                              "Feels like a coincidence to me."])
        if f["twist"]:
            return pick(rng, ["This has to be the source, it's the obvious one in the field.",
                              "I'd put it straight in the circular as the counterpart.",
                              "Bright and right there, I think this is it.",
                              "Not convinced either way."])
        return pick(rng, ["Flagging it for the circular discussion.", "Worth a look before the circular goes.",
                          "Please check against the guide before we name anything.", "No strong opinion from the desk."])

    def questions(self):
        return {
            "association_strength": score("How strong is the case that the neutrino came from the candidate source?", [
                "None: the source is outside the error region or unrelated in time.",
                "Weak: a coincidence of a kind that chance alone would often produce.",
                "Suggestive: positional and timing hints, but chance is not excluded on every count.",
                "Compelling: inside the region, active at the arrival time, and rare enough that chance is unlikely.",
            ]),
            "followup_plan": choice("How should follow-up telescope time be used for this alert?", {
                "archival_search_only": "Search archives and catalogues only; commit no new telescope time.",
                "await_refined_localisation": "Commit nothing yet; wait for the refined localisation before choosing "
                                              "targets.",
                "target_candidate_source": "Concentrate follow-up on the named candidate source with spectroscopy and "
                                           "multi-band monitoring.",
                "tile_error_region": "Tile the whole 90% region with wide-field imaging without singling out the named "
                                     "source.",
            }),
            "chance_alignment_plausible": noul(
                "A chance alignment accounts for the candidate source lying near the neutrino direction as well as a "
                "physical link does.",
                yes="Comparable sources are common enough that one is expected in a region this size, the source lies "
                    "outside the region, or it was not active when the neutrino arrived.",
                no="It lies inside the region, was active at the arrival time, and comparable sources are rare enough "
                   "that chance is unlikely."),
            "localisation_quality": score("How well does the alert pin down where the neutrino came from?", [
                "Poor: a large region, or an initial estimate likely to move substantially.",
                "Rough: a few degrees across, or caveated by reconstruction systematics.",
                "Good: about a degree or so, from a refined reconstruction.",
                "Excellent: well under a degree, from a refined reconstruction of a clean track.",
            ]),
            "circular_wording": choice("How should the candidate source be presented in the follow-up circular?", {
                "claim_likely_counterpart": "Name it as a likely counterpart and quote the chance-coincidence "
                                            "probability.",
                "defer_statement": "Say nothing about it until the refined localisation shows whether it is inside "
                                   "the region.",
                "list_as_coincident": "List it as a coincident source of interest without claiming an association.",
                "omit_source": "Leave it out; it adds nothing beyond what chance would put there.",
            }),
        }


# =========================================================================== schema 4
class AlertRetractionReview(Schema):
    name = "alert_retraction_review"
    description = ("A public transient or gravitational-wave alert that new information has called into question: "
                   "keep, update, hold or retract under the written retraction policy.")
    surfaces = {"json": 0.5, "shift_elog": 0.25, "board_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        kind = pick_w(rng, {"gw": 4, "optical": 3, "neutrino": 3})
        f = {"regime": reg, "kind": kind, "status": None, "numbers": {}, "t_now": rand_ts(rng)}
        if kind == "gw":
            net, code = pick(rng, GW_NETWORKS)
            rt_txt, rt = pick(rng, [("1 per 6 months", 2.0), ("1 per 30 days", 12.2), ("1 per 60 days", 6.1),
                                    ("1 per week", 52.2)])
            far0 = _sig(rt * 10 ** -between(rng, 0.5, 4.5), 2)
            info = pick_w(rng, {"offline_far": 4, "glitch_found": 4, "calibration": 1.5})
            f.update(issuer=net, alert_id=f"{code}{f['t_now'].strftime('%y%m%d')}{_letters(rng, 2)}",
                     retract_thr=rt, retract_txt=rt_txt, far_orig=far0, info=info,
                     src=pick(rng, ["BNS", "NSBH", "BBH"]), area0=rng.randint(30, 2500))
            if info == "offline_far":
                if reg == "clear":
                    far1 = rt * 10 ** between(rng, 0.6, 3.0) if chance(rng, 0.6) else far0 * between(rng, 0.1, 1.5)
                elif reg == "borderline":
                    far1 = rt * 10 ** near(rng, 0.0, 0.25)
                else:
                    far1 = rt * 10 ** -between(rng, 0.05, 0.6)
                f["far_new"] = _sig(far1, 2)
            elif info == "glitch_found":
                f["status"] = ("confirmed" if reg == "clear" and chance(rng, 0.7) else
                               pick(rng, ["preliminary", "confirmed"]) if reg == "conflicting" else
                               pick_w(rng, {"preliminary": 3, "confirmed": 2.5}))
                f["explains"] = chance(rng, 0.8)
                f["glitch_det"] = pick(rng, GW_SITES)[0]
                f["glitch_kind"] = pick(rng, GLITCHES)[0]
            else:
                f["area_new"] = int(f["area0"] * (between(rng, 1.3, 2.5) if chance(rng, 0.55) else between(rng, 0.95, 1.08)))
        elif kind == "optical":
            sname, prefix, _, _ = pick(rng, SURVEYS)
            info = pick_w(rng, {"sso": 3.5, "bad_reference": 3.5, "spectrum": 1.5, "forced_phot": 1.5})
            n = pick(rng, [2, 3, 3, 4])
            f.update(issuer=sname, alert_id=_survey_desig(rng, prefix), info=info, n_dets=n,
                     claim=pick(rng, ["a fast-rising young transient", "a possible fast blue optical transient",
                                      "a candidate orphan afterglow", "a rapidly fading red transient"]),
                     mag0=between(rng, 18.4, 20.9, digits=1))
            if info == "sso":
                if reg == "clear":
                    k = n if chance(rng, 0.6) else max(0, n - 2)
                elif reg == "borderline":
                    k = n - 1
                else:
                    k = n
                f["sso_matched"] = k
                f["sso_name"] = _sso_name(rng)
                f["sso_off_arcsec"] = between(rng, 3.0, 12.0, digits=1)
            elif info == "bad_reference":
                f["status"] = ("confirmed" if reg == "clear" else
                               pick_w(rng, {"preliminary": 3, "confirmed": 2}) if reg == "borderline" else
                               pick(rng, ["preliminary", "confirmed"]))
            elif info == "spectrum":
                f["reclass"] = pick(rng, ["an ordinary type II supernova at z = 0.03", "a dwarf-nova outburst in the Galaxy",
                                          "a flaring active nucleus at z = 0.21", "a type Ia supernova two weeks before peak"])
        else:
            obs, code = pick(rng, NU_OBS)
            floor = pick(rng, [0.1, 0.15, 0.2])
            s0 = between(rng, 0.35, 0.85)
            info = pick_w(rng, {"reprocessed": 4, "artifact": 4, "direction": 2})
            f.update(issuer=obs, alert_id=f"{code}-{f['t_now'].strftime('%y%m%d')}A", floor=floor, sig0=s0, info=info,
                     radius0=between(rng, 0.4, 2.5))
            if info == "reprocessed":
                if reg == "clear":
                    s1 = between(rng, 0.01, floor * 0.6) if chance(rng, 0.6) else s0 + between(rng, -0.05, 0.05)
                elif reg == "borderline":
                    s1 = near(rng, floor, 0.04, lo=0.01)
                else:
                    s1 = floor + between(rng, 0.01, 0.06)
                f["sig_new"] = round(_clamp(s1, 0.01, 0.95), 2)
            elif info == "artifact":
                f["status"] = ("confirmed" if reg == "clear" and chance(rng, 0.7) else
                               pick_w(rng, {"preliminary": 3, "confirmed": 2}))
                f["artifact_kind"] = pick(rng, ["a coincident down-going muon bundle", "a calibration light flash",
                                                "a noisy string firing in a burst", "a timing glitch in one readout crate"])
            else:
                f["shift_deg"] = round(f["radius0"] * (between(rng, 0.6, 2.2) if chance(rng, 0.6) else between(rng, 0.05, 0.3)), 2)
                f["radius_new"] = round(f["radius0"] * between(rng, 0.7, 1.4), 2)
        # who is following the alert, and since when
        f["issued_h"] = round(pick_w(rng, {between(rng, 0.5, 12): 5, between(rng, 12, 36): 3, between(rng, 36, 72): 2}), 1)
        groups = pick(rng, [0, 1, 2, 3, 5, 8, 12, 20])
        if f["issued_h"] > 40:
            on = pick(rng, [0, 0, 0, 1])
        else:
            on = rng.randint(0, min(groups, 6))
        on = min(on, groups)
        f.update(groups=groups, on_target=on, queued=rng.randint(0, groups - on) if groups > on else 0,
                 big_facility=on > 0 and chance(rng, 0.4), circulars=rng.randint(0, groups + 1),
                 big_name=pick(rng, ["the 8.2 m Sarnath telescope", "the 6.5 m Calva-North telescope",
                                     "the 10 m Kaimalu dish", "the Brannock X-ray satellite"]),
                 raised_by=person(rng), shift=person(rng))
        f["pushy"] = reg == "conflicting" or chance(rng, 0.25)
        return f

    def _assess(self, f):
        """(criteria_met, evidence level 0-3, preliminary_instrumental_or_partial, update_type)."""
        k, i = f["kind"], f["info"]
        if k == "gw":
            if i == "offline_far":
                x = f["far_new"]
                if x > f["retract_thr"]:
                    return True, 3 if x > 3 * f["retract_thr"] else 2, False, False
                return False, 1 if x > 3 * f["far_orig"] else 0, False, x > 3 * f["far_orig"]
            if i == "glitch_found":
                if not f["explains"]:
                    return False, 0, False, False
                if f["status"] == "confirmed":
                    return True, 3, False, False
                return False, 2, True, False
            material = f["area_new"] > 1.25 * f["area0"]
            return False, 1 if material else 0, False, material
        if k == "optical":
            if i == "sso":
                if f["sso_matched"] == f["n_dets"]:
                    return True, 3, False, False
                if f["sso_matched"] == f["n_dets"] - 1 and f["sso_matched"] > 0:
                    return False, 2, True, False
                return False, 1 if f["sso_matched"] > 0 else 0, False, f["sso_matched"] > 0
            if i == "bad_reference":
                return (True, 3, False, False) if f["status"] == "confirmed" else (False, 2, True, False)
            if i == "spectrum":
                return False, 1, False, True
            return False, 0, False, False
        if i == "reprocessed":
            s1 = f["sig_new"]
            if s1 < f["floor"]:
                return True, 3 if s1 < f["floor"] / 2 else 2, False, False
            dropped = f["sig0"] - s1 > 0.15
            return False, 1 if dropped else 0, False, dropped
        if i == "artifact":
            return (True, 3, False, False) if f["status"] == "confirmed" else (False, 2, True, False)
        moved = f["shift_deg"] > 0.5 * f["radius0"]
        return False, 1 if moved else 0, False, moved

    def intended(self, f):
        met, ev, prelim, upd = self._assess(f)
        if met:
            dec = "retract_alert"
        elif prelim:
            dec = "hold_pending_reanalysis"
        elif upd:
            dec = "issue_update"
        else:
            dec = "keep_unchanged"
        on, q = f["on_target"], f["queued"]
        if on >= 3 or (on >= 1 and f["big_facility"]):
            cost = 3
        elif on >= 1:
            cost = 2
        elif q >= 1 and f["issued_h"] <= 36:
            cost = 1
        else:
            cost = 0
        moving_target = f["info"] in ("calibration", "direction")
        engaged = on >= 1 or (q >= 1 and f["issued_h"] <= 36)
        partners = ((dec in ("retract_alert", "hold_pending_reanalysis") and engaged)
                    or (dec == "issue_update" and moving_target and engaged))
        return {"alert_update_decision": dec, "evidence_against_strength": ev,
                "retraction_criteria_met": "yes" if met else "no", "followup_cost_of_delay": cost,
                "partners_need_direct_notice": "yes" if partners else "no"}

    def _policy(self, f):
        k = f["kind"]
        if k == "gw":
            return (f"Retraction policy ({f['issuer']}): retract when the offline FAR after trials is above "
                    f"{f['retract_txt']}, or when the detector-characterisation group confirms an instrumental origin. "
                    f"Revised significance, sky map or classification that stays within public thresholds is an update, "
                    f"not a retraction.")
        if k == "optical":
            return (f"Retraction policy ({f['issuer']}): retract when every detection is explained by a catalogued "
                    f"solar-system object, or when re-subtraction against a new reference confirms an image artefact. "
                    f"A new classification is an update, not a retraction.")
        return (f"Retraction policy ({f['issuer']}): retract when reprocessing puts the signalness below {f['floor']}, or "
                f"when the detector group confirms the event as an instrumental or atmospheric-background artefact. A "
                f"revised direction or a lower signalness above that line is an update.")

    def _as_issued(self, f):
        k = f["kind"]
        if k == "gw":
            return (f"{f['src']} candidate, FAR {_far(f['far_orig'])}/yr after trials, 90% area {f['area0']} deg2")
        if k == "optical":
            return f"{f['claim']}, r ~ {f['mag0']}, {f['n_dets']} detections"
        return f"track, signalness {f['sig0']:.2f}, 90% radius {f['radius0']:.2f} deg"

    def _new_info(self, f):
        k, i = f["kind"], f["info"]
        st = f["status"]
        stx = {"preliminary": "preliminary: one analyst, not yet reviewed",
               "confirmed": "confirmed after review by the responsible group", None: None}[st]
        if k == "gw":
            if i == "offline_far":
                return f"offline reanalysis gives FAR {_far(f['far_new'])}/yr after trials (was {_far(f['far_orig'])}/yr)", stx
            if i == "glitch_found":
                g = (f"{f['glitch_kind']} in {f['glitch_det']}" + (" overlapping the signal's in-band track; re-running the "
                     "search with the glitch gated removes the candidate" if f["explains"] else
                     " found 40 s after the merger, outside the signal's in-band duration; candidate unchanged when it is gated"))
                return g, stx
            return (f"calibration correction for one detector moves the 90% area from {f['area0']} to {f['area_new']} deg2", stx)
        if k == "optical":
            if i == "sso":
                if f["sso_matched"] == f["n_dets"]:
                    s = f"all {f['n_dets']} detections lie within 1.5\" of the predicted path of {f['sso_name']}"
                elif f["sso_matched"] > 0:
                    s = (f"{f['sso_matched']} of {f['n_dets']} detections match the predicted path of {f['sso_name']}; the "
                         f"other lies {f['sso_off_arcsec']}\" off it")
                else:
                    s = f"no solar-system object passes within 30\" at any detection epoch (checked against {f['sso_name']})"
                return s, stx
            if i == "bad_reference":
                return "the reference image for this field has a known bad column near the position; residual may be an artefact", stx
            if i == "spectrum":
                return f"classification spectrum shows {f['reclass']}", stx
            return "forced photometry recovers the source on two further nights at consistent brightness", stx
        if i == "reprocessed":
            return f"reprocessing with the updated reconstruction gives signalness {f['sig_new']:.2f} (was {f['sig0']:.2f})", stx
        if i == "artifact":
            return f"detector group flags {f['artifact_kind']} at the event time", stx
        return (f"refined direction moved by {f['shift_deg']} deg; new 90% radius {f['radius_new']} deg "
                f"(was {f['radius0']:.2f} deg)", stx)

    def _comment(self, f, rng):
        met, ev, prelim, upd = self._assess(f)
        if f["pushy"]:
            if met:
                return pick(rng, ["I'd keep it up; observers are already invested.", "Let's not retract yet, it may still be real."])
            return pick(rng, ["This is dead, we should retract now.", "Pull it before more people waste time.",
                              "Retract and move on."])
        return pick(rng, ["Raising for the board's decision.", "Over to the alert board.",
                          "Please decide per the policy.", "Flagging this for a decision."])

    def render(self, f, rng, surface):
        new, stx = self._new_info(f)
        g, c = f["groups"], f["circulars"]
        follow = (f"{g} group{'s' if g != 1 else ''} announced follow-up; {f['on_target']} on target now"
                  + (f" (including {f['big_name']})" if f["big_facility"] else "")
                  + f"; {f['queued']} with observations queued for later nights; "
                  + (f"{c} circular{'s' if c != 1 else ''} cite{'' if c != 1 else 's'} the alert"))
        comment = noisy(self._comment(f, rng), rng, 0.1)
        if surface == "json":
            return dumps(prune({
                "alert": {"id": f["alert_id"], "issued_by": f["issuer"], "issued_hours_ago": f["issued_h"],
                          "as_issued": self._as_issued(f)},
                "new_information": {"finding": new, "status": stx},
                "follow_up": {"groups_announced": f["groups"], "on_target_now": f["on_target"],
                              "large_facility_on_target": f["big_name"] if f["big_facility"] else None,
                              "queued_for_later_nights": f["queued"], "circulars_citing": f["circulars"]},
                "policy": self._policy(f),
                "raised_by": {"name": f["raised_by"], "comment": comment},
            }))
        if surface == "shift_elog":
            lines = [f"[{fmt_ts(f['t_now'], style='log')} UTC] shift {f['shift']}",
                     f"Re: public alert {f['alert_id']} ({f['issuer']}), issued {f['issued_h']} h ago: {self._as_issued(f)}.",
                     f"New: {new}." + (f" Status: {stx}." if stx else ""),
                     f"Follow-up: {follow}.", "", self._policy(f), "",
                     f"{f['raised_by']} adds: \"{comment}\""]
            return "\n".join(lines)
        return "\n".join([
            f"Subject: {f['alert_id']} -- new information, decision needed",
            f"From: {f['raised_by']}", "To: alert board", "",
            "Board,", "",
            noisy(join_sentences([f"Our alert {f['alert_id']} went out {f['issued_h']} hours ago ({self._as_issued(f)}).",
                                  f"Since then: {new}.", f"Status: {stx}." if stx else "",
                                  f"Follow-up so far: {follow}."]), rng, 0.08),
            "", f"For reference: {self._policy(f)}", "", comment, "", f["raised_by"].split()[0]])

    def questions(self):
        return {
            "alert_update_decision": choice("What should the alert team do with the public alert now?", {
                "hold_pending_reanalysis": "Post a short notice that the alert is under review, and decide once the "
                                           "reanalysis is complete.",
                "issue_update": "Keep the alert standing but publish revised numbers, classification or localisation.",
                "keep_unchanged": "Leave the alert as issued; nothing material has changed.",
                "retract_alert": "Retract the alert publicly.",
            }),
            "evidence_against_strength": score(
                "How strongly does the new information count against the alert as it was issued?", [
                    "Not at all: it is neutral or even supports the alert.",
                    "Somewhat: it revises the numbers or the picture, but the alert's basis stands.",
                    "Seriously: it points to a non-astrophysical origin that is not yet established.",
                    "Decisively: it meets the written grounds for retraction on established evidence.",
                ]),
            "retraction_criteria_met": noul(
                "The new information meets the written retraction policy as it stands.",
                yes="A quoted retraction condition is satisfied: a revised number crosses the stated line, or an "
                    "instrumental or solar-system origin is confirmed in the way the policy requires.",
                no="No condition is fully satisfied: the numbers stay on the right side of the line, or the "
                   "explanation is still preliminary or only partial."),
            "followup_cost_of_delay": score(
                "How costly is each further hour without a decision, for the observers following this alert?", [
                    "Negligible: no one is observing it or planning to soon.",
                    "Modest: some groups have observations queued for later nights.",
                    "Considerable: one or more groups are on target tonight.",
                    "Severe: several groups, or a large facility, are spending time on it right now.",
                ]),
            "partners_need_direct_notice": noul(
                "Observers who committed time to this alert should be contacted directly before the public update "
                "goes out."),
        }


# =========================================================================== schema 5
class CounterpartCandidateTriage(Schema):
    name = "counterpart_candidate_triage"
    description = ("One optical candidate found inside a gravitational-wave sky map: how kilonova-like it is, whether "
                   "its host distance fits, and how to prioritise it.")
    surfaces = {"json": 0.45, "observer_report": 0.3, "pretty_json": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        net, code = pick(rng, GW_NETWORKS)
        sname, prefix, _, sdepth = pick(rng, SURVEYS)
        D = int(between(rng, 60, 340))
        sigma = int(D * between(rng, 0.18, 0.35))
        fade_thr = pick(rng, [0.3, 0.4, 0.5])
        red_thr = pick(rng, [0.1, 0.15, 0.2])
        M_lo, M_hi = pick(rng, [(-17.0, -14.5), (-16.5, -14.0), (-17.5, -15.0)])
        seed = pick_w(rng, {"kn": 3.5, "young_sn": 2.3, "agn": 2.3, "stellar": 2.1, "far_sn": 1.0})
        mu_gw = _mu(D)
        host_kind = pick_w(rng, {"spec": 4, "photo": 3.5, "none": 2.5})
        nsig_target = between(rng, 0.0, 1.5)
        pre_det = cat_var = False
        if seed == "kn":
            fade = between(rng, fade_thr + 0.15, 1.4)
            red = between(rng, red_thr + 0.08, 0.7)
            M = between(rng, M_lo + 0.3, M_hi - 0.3)
            miss = pick_w(rng, {None: 5, "fade": 1.3, "red": 1.7, "mag": 1.3})
            if miss == "fade":
                fade = between(rng, 0.05, fade_thr - 0.08)
            elif miss == "red":
                red = between(rng, -0.03, red_thr - 0.05)
            elif miss == "mag":
                M = pick(rng, [between(rng, M_lo - 1.5, M_lo - 0.3), between(rng, M_hi + 0.3, M_hi + 1.2)])
        elif seed == "young_sn":
            fade = between(rng, -0.25, 0.1)
            red = between(rng, -0.05, 0.06)
            M = between(rng, -18.5, -16.0)
            nsig_target = between(rng, 0.0, 2.6)
        elif seed == "agn":
            fade = between(rng, 0.0, 0.3)
            red = between(rng, -0.05, 0.1)
            M = between(rng, -19.0, -16.0)
            cat_var = chance(rng, 0.7)
            pre_det = not cat_var or chance(rng, 0.5)
            host_kind = pick(rng, ["spec", "photo"])
        elif seed == "stellar":
            fade = between(rng, 0.1, 0.8)
            red = between(rng, -0.1, 0.05)
            M = between(rng, M_lo - 0.5, M_hi + 0.5)
            pre_det = chance(rng, 0.7)
            cat_var = not pre_det or chance(rng, 0.3)
            host_kind = "none"
        else:
            fade = between(rng, -0.1, 0.12)
            red = between(rng, -0.03, 0.08)
            M = between(rng, -17.0, -15.0)
            host_kind = "spec"
            nsig_target = between(rng, 3.5, 8.0)
        # regime steering
        knobs = []
        if reg == "borderline":
            knobs = [pick(rng, ["fade", "red", "mag", "nsig", "map"])]
            if "fade" in knobs:
                fade = near(rng, fade_thr, 0.08)
            if "red" in knobs:
                red = near(rng, red_thr, 0.05)
            if "mag" in knobs:
                M = near(rng, pick(rng, [M_lo, M_hi]), 0.25)
            if "nsig" in knobs:
                nsig_target = near(rng, pick(rng, [2.0, 3.0]), 0.3, lo=0.0)
                if host_kind == "none":
                    host_kind = "spec"
        twist = None
        if reg == "conflicting":
            twist = pick(rng, ["kn_but_pre", "kn_but_pre", "kn_but_far", "kn_but_map", "dull_but_fits", "missing_g"])
            if twist.startswith("kn_"):
                fade, red, M = between(rng, fade_thr + 0.1, 1.2), between(rng, red_thr + 0.05, 0.6), between(rng, M_lo + 0.2, M_hi - 0.2)
            if twist == "kn_but_pre":
                pre_det = True
            elif twist == "kn_but_far":
                host_kind, nsig_target = "spec", between(rng, 2.6, 6.0)
            elif twist == "dull_but_fits":
                fade, red = between(rng, -0.05, fade_thr - 0.12), between(rng, -0.04, red_thr - 0.06)
                host_kind, nsig_target = "spec", between(rng, 0.0, 0.9)
                pre_det = cat_var = False
        # keep the candidate within reach of the instruments that find GW counterparts: if it would be fainter than
        # r ~ 22.6 at this distance, the event is placed nearer (the GW distance and its error scale together)
        if M + _mu(D) > 22.6:
            d_new = max(40, int(10 ** ((between(rng, 20.6, 22.6) - M) / 5 - 5)))
            sigma, D = max(8, int(sigma * d_new / D)), d_new
        mu_gw = _mu(D)
        # host distance from the targeted number of sigma (either side of D)
        host = None
        if host_kind != "none":
            sgn = pick(rng, [-1, 1])
            if host_kind == "spec":
                Dh = _clamp(D + sgn * nsig_target * sigma, 15, 3000)
                z = round(_z_of_dl(Dh), 4)
                host = {"name": _galaxy_name(rng, *_radec(rng)), "z": z, "z_kind": "spec", "z_err": None,
                        "dist": round(_dl_mpc(z)), "dist_err": None, "offset_arcsec": between(rng, 0.5, 15, digits=1)}
            else:
                z0 = _z_of_dl(D)
                zerr = round(between(rng, 0.008, 0.02), 3)
                derr = _dl_mpc(z0 + zerr) - _dl_mpc(z0)
                Dh = _clamp(D + sgn * nsig_target * math.sqrt(sigma ** 2 + derr ** 2), 15, 3000)
                z = round(_z_of_dl(Dh), 3)
                host = {"name": _galaxy_name(rng, *_radec(rng)), "z": z, "z_kind": "photo", "z_err": zerr,
                        "dist": round(_dl_mpc(z)), "dist_err": round(_dl_mpc(z + zerr) - _dl_mpc(z)),
                        "offset_arcsec": between(rng, 0.5, 15, digits=1)}
            if seed == "agn":
                host["offset_arcsec"] = between(rng, 0.0, 0.3)
        # photometry: two or three epochs after the merger
        t1 = between(rng, 0.5, 30.0, digits=1)
        r1 = round(M + mu_gw, 2)
        gr1 = between(rng, -0.35, 0.3)
        g_missing = twist == "missing_g" or (reg != "clear" and chance(rng, 0.08))
        epochs = []
        dts = [0.0, between(rng, 0.7, 1.3)] + ([between(rng, 1.8, 2.6)] if chance(rng, 0.4) else [])
        fade_r = fade - red
        for k, dt in enumerate(dts):
            g = round(r1 + gr1 + fade * dt + rng.gauss(0, 0.03), 2)
            r = round(r1 + fade_r * dt + rng.gauss(0, 0.03), 2)
            epochs.append({"t_hours": round(t1 + 24 * dt, 1), "g": None if (g_missing and k > 0) else g, "r": r})
        # pre-merger imaging
        pre_days = between(rng, 0.4, 20.0, digits=1)
        shallow = reg == "conflicting" and chance(rng, 0.3) and not pre_det
        pre_lim = (round(r1 - between(rng, 0.2, 1.0), 1) if shallow
                   else round(min(r1 + between(rng, 0.6, 2.5), max(r1 + 0.5, 23.5)), 1))
        if r1 <= sdepth - 0.2:
            found_by = sname
        else:
            found_by = pick(rng, [t for t, dep in TILERS if dep >= r1 + 0.3] or [TILERS[-1][0]])
        arch = None
        if pre_det:
            arch = pick(rng, [f"detected at r = {round(r1 + between(rng, 0.8, 2.8), 1)} in the {rng.choice([2023, 2024, 2025])} reference stack",
                              f"faint source at r = {round(r1 + between(rng, 1.0, 3.0), 1)} in images {rng.randint(5, 60)} d before the merger",
                              f"{rng.randint(2, 9)} detections in archival survey data over {rng.randint(1, 5)} yr"])
        cv = None
        if cat_var:
            cv = (f"listed in the {pick(rng, VAR_CATS)} as " + ("a known active nucleus with 0.3 mag variability" if seed == "agn"
                                                               else pick(rng, ["a dwarf nova", "a flare star", "an irregular variable"])))
        # sky-map position: credible level of the candidate in the initial and the latest map
        if reg == "borderline" and knobs == ["map"]:
            c_upd = near(rng, 90, 4, lo=50, hi=99.5)
        elif twist == "kn_but_map":
            c_upd = between(rng, 92, 99.5)
        else:
            c_upd = between(rng, 20, 88) if chance(rng, 0.6) else (between(rng, 91, 96.5) if chance(rng, 0.8)
                                                                    else between(rng, 97.5, 99.8))
        c_init = _clamp(c_upd + rng.gauss(0, 12), 5, 99.8) if chance(rng, 0.7) else c_upd
        if twist == "kn_but_map":
            c_init = between(rng, 40, 85)
        return {"regime": reg, "seed": seed, "twist": twist, "knobs": knobs, "network": net,
                "gw_id": f"{code}{rng.choice([25, 26, 27])}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}{_letters(rng, 2)}",
                "distance": D, "distance_sigma": sigma, "area90": rng.randint(20, 1500), "survey": found_by,
                "cand_id": _survey_desig(rng, prefix), "epochs": epochs, "fade_g": round(fade, 3), "redden": round(red, 3),
                "abs_mag": round(r1 - mu_gw, 2), "host": host, "pre_days": pre_days, "pre_limit": pre_lim,
                "archival": arch, "catalogued_var": cv, "c_init": round(c_init, 1), "c_upd": round(c_upd, 1),
                "fade_thr": fade_thr, "red_thr": red_thr, "M_lo": M_lo, "M_hi": M_hi, "observer": person(rng),
                "spectrograph": pick(rng, SPECTROGRAPHS)[0], "date": rand_ts(rng).date()}

    def _measured(self, f):
        e = f["epochs"]
        a, b = e[0], e[-1]
        dt = (b["t_hours"] - a["t_hours"]) / 24.0
        if a["g"] is None or b["g"] is None or dt <= 0:
            return None, None
        fg = (b["g"] - a["g"]) / dt
        red = ((b["g"] - b["r"]) - (a["g"] - a["r"])) / dt
        return fg, red

    def intended(self, f):
        h = f["host"]
        D, s = f["distance"], f["distance_sigma"]
        if h is None:
            dist = 1
        elif h["z_kind"] == "spec":
            n = abs(h["dist"] - D) / s
            dist = 3 if n <= 1 else (2 if n <= 2 else (1 if n <= 3 else 0))
        else:
            n = abs(h["dist"] - D) / math.sqrt(s ** 2 + h["dist_err"] ** 2)
            dist = 2 if n <= 2 else (1 if n <= 3 else 0)
        pre = bool(f["archival"] or f["catalogued_var"])
        fg, red = self._measured(f)
        tests = 0
        if fg is not None and fg >= f["fade_thr"]:
            tests += 1
        if red is not None and red >= f["red_thr"]:
            tests += 1
        if f["M_lo"] <= f["abs_mag"] <= f["M_hi"]:
            tests += 1
        kn = tests
        loc_ok = f["c_upd"] <= 90
        if pre or dist == 0 or f["c_upd"] > 97:
            pr = "rule_out"
        elif kn >= 2 and dist >= 1 and loc_ok and (dist >= 2 or kn == 3):
            pr = "top_priority_spectroscopy"
        elif kn >= 1 and dist >= 1:
            pr = "keep_photometric_monitoring"
        else:
            pr = "deprioritise"
        return {"candidate_priority": pr, "distance_consistency": dist, "kilonova_likeness": kn,
                "pre_existing_source": "yes" if pre else "no", "inside_latest_skymap": "yes" if loc_ok else "no"}

    def _rules(self, f):
        return (f"Counterpart triage rules: kilonova-like means g fades by at least {f['fade_thr']} mag/day, g-r reddens "
                f"by at least {f['red_thr']} mag/day, and the absolute magnitude at the GW distance lies between "
                f"{f['M_lo']} and {f['M_hi']}. Rule out sources present before the merger, hosts more than 3 sigma "
                f"from the GW distance, and positions beyond the 97% credible level of the latest map.")

    def _host_text(self, h):
        if h is None:
            return "no catalogued host within 30\""
        if h["z_kind"] == "spec":
            return f"{h['name']}, z_spec {h['z']:.4f} ({h['dist']} Mpc), offset {h['offset_arcsec']}\""
        return (f"{h['name']}, photo-z {h['z']:.3f} +/- {h['z_err']} ({h['dist']} +/- {h['dist_err']} Mpc), "
                f"offset {h['offset_arcsec']}\"")

    def _pre_text(self, f):
        parts = [f"deepest pre-merger image {f['pre_days']} d before: nothing to r > {f['pre_limit']}"
                 if not f["archival"] else f"pre-merger: {f['archival']}"]
        if f["catalogued_var"]:
            parts.append(f["catalogued_var"])
        return "; ".join(parts)

    def render(self, f, rng, surface):
        ep = [{"hours_after_merger": e["t_hours"], "g": e["g"], "r": e["r"]} for e in f["epochs"]]
        h = f["host"]
        obj = {
            "gw_event": {"id": f["gw_id"], "network": f["network"], "distance_mpc": f["distance"],
                         "distance_sigma_mpc": f["distance_sigma"], "area90_deg2": f["area90"]},
            "candidate": {"id": f["cand_id"], "found_by": f["survey"], "photometry": ep,
                          "abs_mag_r_first_epoch_at_gw_distance": f["abs_mag"],
                          "credible_level_pct": {"initial_map": f["c_init"], "latest_map": f["c_upd"]}},
            "host": None if h is None else {"name": h["name"], "redshift": h["z"], "redshift_kind": h["z_kind"],
                                            "redshift_err": h["z_err"], "distance_mpc": h["dist"],
                                            "distance_err_mpc": h["dist_err"], "offset_arcsec": h["offset_arcsec"]},
            "history": {"last_pre_merger_image_days": f["pre_days"],
                        "pre_merger_limit_r": None if f["archival"] else f["pre_limit"],
                        "archival_detection": f["archival"], "variable_catalogue": f["catalogued_var"]},
            "rules": self._rules(f),
        }
        if surface == "json":
            return dumps(prune(obj))
        if surface == "pretty_json":
            obj["observer_note"] = noisy(self._note(f, rng), rng, 0.1)
            return dumps_pretty(prune(obj), rng)
        rows = [[f"+{e['t_hours']} h", show(e["g"], rng), e["r"]] for e in f["epochs"]]
        return "\n".join([
            f"Counterpart report -- {f['cand_id']} in the {f['gw_id']} map ({f['network']})",
            f"Observer: {f['observer']}, {fmt_date(f['date'], rng)}", "",
            kv_lines([("GW distance", f"{f['distance']} +/- {f['distance_sigma']} Mpc"),
                      ("90% area", f"{f['area90']} deg2"),
                      ("Credible level at position", f"{f['c_init']}% (initial map), {f['c_upd']}% (latest map)"),
                      ("Host", self._host_text(h)),
                      ("History", self._pre_text(f)),
                      ("M_r at GW distance (first epoch)", f"{f['abs_mag']}"),
                      ("Found by", f["survey"])], rng, style=pick(rng, ["colon", "aligned", "dash"])),
            "", table(["epoch", "g", "r"], rows, rng), "", self._rules(f), "",
            "Note: " + noisy(self._note(f, rng), rng, 0.1)])

    def _note(self, f, rng):
        if f["regime"] == "conflicting":
            return pick(rng, ["This is our best candidate so far, I'd push for a spectrum.",
                              "Colours look promising.", "Nothing special, I'd drop it.",
                              f"Spectrum possible with {f['spectrograph']} if we decide tonight."])
        return pick(rng, ["Adding to the shared list.", f"{f['spectrograph']} is available tonight.",
                          "Next epoch scheduled for tomorrow.", "Please rank against the rules."])

    def questions(self):
        return {
            "candidate_priority": choice("How should this counterpart candidate be prioritised?", {
                "deprioritise": "Keep it on the list at low priority: nothing excludes it, but nothing makes it stand out.",
                "keep_photometric_monitoring": "Keep monitoring it photometrically and decide on spectroscopy after the "
                                               "next epochs.",
                "rule_out": "Rule it out as the counterpart and stop spending time on it.",
                "top_priority_spectroscopy": "Make it the top spectroscopic target tonight.",
            }),
            "distance_consistency": score(
                "How consistent is the candidate's host distance with the gravitational-wave distance?", [
                    "Inconsistent: the host lies more than about 3 sigma from the GW distance.",
                    "Weak: there is no host redshift, or the host sits 2 to 3 sigma away.",
                    "Consistent within about 2 sigma, but resting on a photometric redshift or near the edge.",
                    "Strong: a spectroscopic host redshift within about 1 sigma of the GW distance.",
                ]),
            "kilonova_likeness": score("How kilonova-like is the candidate's behaviour?", [
                "Not at all: it passes none of the quoted tests.",
                "Slightly: it passes one of the quoted tests.",
                "Largely: it passes two of the quoted tests.",
                "Fully: fast fading, rapid reddening and the right luminosity at the GW distance.",
            ]),
            "pre_existing_source": noul(
                "The candidate was already present, or variable, before the merger time.",
                yes="Pre-merger imaging shows a source at the position, or a variable source is catalogued there.",
                no="Nothing was seen at the position before the merger and no variable source is catalogued there."),
            "inside_latest_skymap": noul(
                "The candidate lies inside the 90% credible region of the most recent sky map."),
        }


WORKFLOW = Workflow(
    name="transient_alert_vetting",
    title="Transient alert vetting",
    description=("Vetting astrophysical transient and gravitational-wave-style candidates from written pipeline "
                 "outputs: source class, follow-up, public notices and retraction exposure."),
    domain="physics / astronomy / research operations",
    schemas=[OpticalSurveyAlert(), GwCandidateVetting(), NeutrinoTrackCoincidence(), AlertRetractionReview(),
             CounterpartCandidateTriage()],
    notes=("Clear of the security_incidents exclusion: nothing here is a machine or account alert, there is no "
           "'is it real / true positive' yes-no question paired with a close / monitor / investigate / contain "
           "disposition, and the decisions are astronomy-specific (source class among supernova, nuclear, Galactic "
           "and moving-object options; spectroscopy versus photometric monitoring; significant versus "
           "low-significance public notices under a trials-factor rule; circular wording; keep / update / hold / "
           "retract under a written retraction policy). All surveys, networks, telescopes and numbers are fictional."),
)
