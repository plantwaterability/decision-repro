"""Detector data-quality flags: is a run, or a range of luminosity blocks, from a particle or nuclear physics
experiment good, bad or only partly usable for physics, judged against the experiment's WRITTEN data-quality
criteria?

Every experiment, detector, facility, run, person, policy and number is fictional. Detectors carry bird names
and laboratories invented place names; no real collaboration's policy is quoted. The criteria (dead-channel
limits, good-run luminosity tolerances, live-time floors, PMT coverage, sign-off rules, shift procedures) are
written into every state and VARY between cases, so an answer needs the case's own rule, not a memorised one.

Each schema draws a `regime` first. Clear cases sit far from the written limits. Borderline cases put a measured
value within a few percent of a limit (2.07% dead channels against a 2% rule, 94.6% live time against 95%, a
GPS hold-over drift a few ns under the tolerance). Conflicting cases pair the numbers with a shifter's or
analyst's opinion that points the other way, or leave a key measurement out. Every schema has `intended(f)`:
the answer a careful data-quality reviewer would give from the factors. It is used only by the answer-spread
audit and is never rendered.
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
    join_sentences,
    kv_lines,
    near,
    noisy as _base_noisy,
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

# =========================================================================== shared pools
# (detector, facility, centre-of-mass energy in TeV) -- proton-proton colliders, all invented
COLLIDERS = [
    ("SKUA", "Vantor Ring", 13.6), ("FULMAR", "Mirefield Collider", 14.0), ("MERGANSER", "Calder Ridge Collider", 13.0),
    ("SANDERLING", "Kessick Hadron Ring", 13.6), ("LAPWING", "Ostmark Collider", 27.0), ("REDSHANK", "Lindholm Ring", 13.6),
    ("PIPIT", "Saltcote Collider", 14.0), ("GREBE", "Wrenfield Ring", 13.6), ("NIGHTJAR", "Ardmore Hadron Collider", 14.0),
]

# subsystem key: (display name, group, channel noun)
SUBSYS = {
    "pixel": ("pixel tracker", "tracker", "modules"),
    "strip": ("silicon strip tracker", "tracker", "modules"),
    "ecal": ("electromagnetic calorimeter", "calorimeter", "crystals"),
    "hcal": ("hadronic calorimeter", "calorimeter", "towers"),
    "fcal": ("forward calorimeter", "calorimeter", "cells"),
    "muon_dt": ("muon drift chambers", "muon", "wire channels"),
    "muon_trig": ("muon trigger chambers", "muon", "strips"),
    "l1_calo": ("level-1 calorimeter trigger", "trigger", "trigger towers"),
    "l1_muon": ("level-1 muon trigger", "trigger", "sector links"),
    "solenoid": ("solenoid magnet", "magnet", "field"),
}


def _partition(rng, sub):
    s = pick(rng, ["+", "-"])
    if sub == "pixel":
        return pick(rng, [f"barrel layer {rng.randint(1, 4)} ladder {rng.randint(1, 64)}", f"forward disk {s}{rng.randint(1, 3)}"])
    if sub == "strip":
        return pick(rng, [f"outer barrel layer {rng.randint(1, 6)} rod {rng.randint(1, 88)}", f"endcap {s} wheel {rng.randint(1, 9)}"])
    if sub == "ecal":
        return pick(rng, [f"barrel supermodule {s}{rng.randint(1, 18):02d}", f"endcap quadrant {s}{rng.randint(1, 4)}"])
    if sub == "hcal":
        return pick(rng, [f"barrel wedge {s}{rng.randint(1, 18)}", f"endcap readout box {s}{rng.randint(1, 36)}"])
    if sub == "fcal":
        return f"forward {s} ring {rng.randint(1, 13)}"
    if sub == "muon_dt":
        return f"wheel {pick(rng, ['-2', '-1', '0', '+1', '+2'])} sector {rng.randint(1, 12)}"
    if sub == "muon_trig":
        return f"endcap station {s}{rng.randint(1, 4)} ring {rng.randint(1, 3)}"
    if sub == "l1_calo":
        return f"processor crate {rng.randint(0, 17)}"
    if sub == "l1_muon":
        return f"{pick(rng, ['barrel', 'overlap', 'endcap'])} sector processor {rng.randint(1, 12)}"
    return "main solenoid"


def _region(partition: str) -> str:
    if partition.startswith("forward"):
        return "forward"
    if "endcap" in partition or "disk" in partition:
        return "endcap"
    return "barrel"


def _policy_id(rng, prefix="DQ"):
    return f"{prefix}-{rng.randint(2025, 2027)}.{rng.randint(1, 9)}"


def _first(name: str) -> str:
    return name.split()[0]


def _v(rng, *wordings):
    """One of several equivalent wordings of a written rule (same numbers, same meaning)."""
    return pick(rng, list(wordings))


def noisy(text: str, rng, p: float = 0.2) -> str:
    """base.noisy, but a leading acronym or code (GPS, HV, LB, K7) keeps its capitals."""
    out = _base_noisy(text, rng, p)
    head = text.split(" ", 1)[0]
    if out and text and out[0] != text[0] and sum(c.isupper() or c.isdigit() for c in head) >= 2:
        out = text[0] + out[1:]
    return out


# =========================================================================== schema 1
class CollisionRunSignoff(Schema):
    name = "collision_run_signoff"
    description = ("A collider run with an open detector defect, signed off against the written good-run criteria: "
                   "good, good minus blocks, restricted lists, bad, or hold.")
    surfaces = {"json": 0.5, "signoff_sheet": 0.25, "shift_elog": 0.25}

    DEFECTS = {
        "hv_trip": dict(subs=["pixel", "strip", "muon_dt"], limits=[1.0, 2.0, 3.0], kind="channel", digits=2),
        "dead_channels": dict(subs=["ecal", "hcal", "muon_dt", "strip"], limits=[0.5, 1.0, 2.0], kind="channel", digits=2),
        "noisy_channels": dict(subs=["ecal", "hcal", "fcal", "strip"], limits=[0.5, 1.0, 2.0], kind="channel", digits=2),
        "calib_drift": dict(subs=["ecal", "hcal", "fcal"], limits=[0.5, 1.0, 1.5], kind="calib", digits=2),
        "alignment_shift": dict(subs=["pixel", "strip", "muon_dt"], limits=[5, 10, 15, 20], kind="align", digits=0),
        "timing_shift": dict(subs=["l1_calo", "l1_muon"], limits=[1.5, 2.0, 2.5, 3.0], kind="timing", digits=1),
        "readout_errors": dict(subs=["pixel", "strip", "ecal", "hcal", "muon_dt", "muon_trig"],
                               limits=[0.1, 0.25, 0.5, 1.0], kind="readout", digits=2),
        "magnet_ramp": dict(subs=["solenoid"], limits=[0.0], kind="magnet", digits=2),
    }
    RECOVERABLE = ("calib_drift", "alignment_shift", "noisy_channels")
    LABEL = {"hv_trip": "HV trip", "dead_channels": "dead channels", "noisy_channels": "noisy channels",
             "calib_drift": "response drift", "alignment_shift": "alignment shift",
             "timing_shift": "trigger timing shift", "readout_errors": "corrupted readout fragments",
             "magnet_ramp": "solenoid below nominal field"}
    # alignment tolerances differ by an order of magnitude between pixels, strips and muon chambers (um)
    ALIGN = {"pixel": ([3.0, 5.0, 8.0], 1), "strip": ([10.0, 15.0, 20.0], 1), "muon_dt": ([100, 150, 200], 0)}
    MINOR = [
        "{n} noisy cells in the forward calorimeter masked online; tolerable",
        "one muon trigger chamber at reduced efficiency, below the channel limit; tolerable",
        "beam-spot fit missing for {n} blocks; no physics impact",
        "pixel module {m} masked since the previous run; already in the reference",
        "secondary luminometer low for {n} blocks; primary luminometer fine",
        "{n} hadronic calorimeter channels with pedestal shifts, corrected online; tolerable",
        "one strip module with elevated leakage current, still inside limits; tolerable",
    ]

    def _minor(self, rng):
        if not chance(rng, 0.55):
            return None
        return pick(rng, self.MINOR).format(n=rng.randint(2, 9), m=rng.randint(100, 1899))

    @staticmethod
    def _share(a, b, n, dur_h, tau_h):
        """Fraction of the fill's delivered luminosity in blocks a..b (1-based, inclusive): luminosity decays
        exponentially through a fill, so early blocks weigh more."""
        x0, x1 = (a - 1) / n * dur_h, b / n * dur_h
        return (math.exp(-x0 / tau_h) - math.exp(-x1 / tau_h)) / (1 - math.exp(-dur_h / tau_h))

    def sample_factors(self, rng):
        reg = regime(rng)
        det, fac, ecm = pick(rng, COLLIDERS)
        defect = pick_w(rng, {"hv_trip": 2, "dead_channels": 1.5, "noisy_channels": 3.5, "calib_drift": 3,
                              "alignment_shift": 2.5, "timing_shift": 1.5, "readout_errors": 1.5, "magnet_ramp": 1})
        spec = self.DEFECTS[defect]
        sub = pick(rng, spec["subs"])
        limit = pick(rng, spec["limits"])
        digits = spec["digits"]
        if defect == "alignment_shift":
            limits, digits = self.ALIGN[sub]
            limit = pick(rng, limits)
        tol = pick(rng, [1.0, 2.0, 3.0, 5.0])
        signoff_pct = pick(rng, [5, 10, 15, 20])
        dur_h = between(rng, 3, 16, digits=1)
        lb_s = pick(rng, [60, 60, 46, 93])
        n_lb = int(dur_h * 3600 / lb_s)
        peak = between(rng, 0.9, 2.4)
        tau = between(rng, 10, 18, digits=1)
        nominal_field = None
        if defect == "magnet_ramp":
            nominal_field = pick(rng, [2.0, 3.8, 4.0])
            mag = round(nominal_field * rng.uniform(0.25, 0.97), 2)
        else:
            if reg == "clear":
                r = rng.uniform(0.15, 0.6) if chance(rng, 0.47) else rng.uniform(1.8, 4.5)
            elif reg == "borderline":
                r = near(rng, 1.0, 0.25, lo=0.6)
            else:
                r = rng.uniform(0.55, 1.9)
            mag = round(limit * r, digits)
            if mag <= 0:
                mag = round(limit * 0.3, digits) or 1
        # luminosity share of the affected blocks
        if reg == "clear":
            target = tol * (rng.uniform(0.1, 0.6) if chance(rng, 0.5) else rng.uniform(2.5, 14))
        elif reg == "borderline":
            target = near(rng, tol, 0.45 * tol, lo=0.05)
        else:
            target = tol * rng.uniform(0.3, 9)
        target = min(target, 80.0)
        a = 1 if defect == "magnet_ramp" else rng.randint(1, max(1, int(n_lb * 0.9)))
        b = a
        while b < n_lb and 100 * self._share(a, b, n_lb, dur_h, tau) < target:
            b += 1
        while a > 1 and 100 * self._share(a, b, n_lb, dur_h, tau) < target:
            a -= 1
        share = self._share(a, b, n_lb, dur_h, tau)
        avg = peak * tau / dur_h * (1 - math.exp(-dur_h / tau))
        delivered = round(avg * 36 * dur_h, 1)
        eff = between(rng, 0.91, 0.975, digits=3)
        missing = defect != "magnet_ramp" and ((reg == "conflicting" and chance(rng, 0.4)) or chance(rng, 0.04))
        shown_mag = None if missing else mag
        bad = True if defect == "magnet_ramp" else (None if missing else mag > limit)
        if reg == "conflicting" and bad is not None:
            stance = "fine" if bad else "bad"
        elif reg == "conflicting":
            stance = pick(rng, ["fine", "bad"])
        else:
            stance = pick_w(rng, {"neutral": 6, "consistent": 3})
        start = rand_ts(rng)
        return {
            "regime": reg, "detector": det, "facility": fac, "ecm_tev": ecm,
            "run": rng.randint(380000, 529999), "fill": rng.randint(8000, 11999), "start": start,
            "dur_h": dur_h, "lb_s": lb_s, "n_lb": n_lb, "peak_lumi_e34": peak, "lumi_lifetime_h": tau,
            "delivered_pb": delivered, "recorded_pb": round(delivered * eff, 1),
            "pileup": int(round(peak * rng.uniform(20, 27))),
            "defect": defect, "subsystem": sub, "partition": _partition(rng, sub),
            "limit": limit, "tol_pct": tol, "signoff_pct": signoff_pct, "magnitude": shown_mag,
            "nominal_field": nominal_field, "drift_dir": pick(rng, ["below", "above"]),
            "extent_hint": pick(rng, ["narrow", "wide"]) if missing else None,
            "lb_first": a, "lb_last": b, "affected_lumi_pct": round(100 * share, 2),
            "affected_pb": round(delivered * share, 2), "recover_ok": chance(rng, 0.85),
            "noisy_count": rng.randint(12, 480),
            "expert_signoff": pick_w(rng, {"signed": 5, "pending": 3, "not_requested": 2}),
            "signoff_wait_h": rng.randint(2, 30),
            "shifter_stance": stance, "shifter": person(rng), "expert": person(rng),
            "minor_defect": self._minor(rng),
            "policy_version": _policy_id(rng),
        }

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _bad(f):
        if f["defect"] == "magnet_ramp":
            return True
        if f["magnitude"] is None:
            return None
        return f["magnitude"] > f["limit"]

    def _level_text(self, f):
        v = f["magnitude"]
        if v is None:
            return None
        noun = SUBSYS[f["subsystem"]][2]
        return {
            "hv_trip": f"{v:.2f}% of {noun} without high voltage",
            "dead_channels": f"{v:.2f}% of {noun} dead",
            "noisy_channels": f"{v:.2f}% of {noun} above the noise-occupancy limit",
            "calib_drift": f"response {v:.2f}% {f['drift_dir']} the reference",
            "alignment_shift": f"{v:g} um displacement relative to the alignment constants",
            "timing_shift": f"level-1 timing offset of {v:.1f} ns",
            "readout_errors": f"{v:.2f}% of events with corrupted fragments",
            "magnet_ramp": f"field at {v:.2f} T against a nominal {f['nominal_field']} T",
        }[f["defect"]]

    def _rules(self, f, rng):
        L, T, E = f["limit"], f["tol_pct"], f["signoff_pct"]
        kind = self.DEFECTS[f["defect"]]["kind"]
        sub = SUBSYS[f["subsystem"]][0]
        first = {
            "channel": _v(rng, f"A block is BAD for a subsystem when more than {L:g}% of its channels or modules are off, dead or noisy.",
                          f"Blocks with over {L:g}% of a subsystem's channels or modules off, dead or noisy are BAD for that subsystem."),
            "calib": _v(rng, f"A calorimeter response shift of more than {L:g}% from the reference makes a block BAD until corrected.",
                        f"Blocks where the calorimeter response sits more than {L:g}% away from its reference are BAD until corrected."),
            "align": _v(rng, f"A displacement of the {sub} of more than {L:g} um from its alignment constants makes a block BAD until realigned.",
                        f"Blocks in which the {sub} has moved more than {L:g} um from its alignment constants are BAD until realigned."),
            "timing": _v(rng, f"A level-1 trigger timing offset of more than {L:g} ns makes a block BAD.",
                         f"Blocks taken with the level-1 timing more than {L:g} ns off are BAD."),
            "readout": _v(rng, f"A block in which more than {L:g}% of events carry corrupted readout fragments is BAD.",
                          f"Blocks with over {L:g}% of events carrying corrupted readout fragments are BAD."),
            "magnet": _v(rng, "Every block recorded with the solenoid below its nominal field is BAD for physics.",
                         "Blocks taken while the solenoid was below nominal field are BAD for physics."),
        }[kind]
        return [
            first,
            _v(rng, f"A run enters the all-physics good-run list when BAD blocks carry no more than {T:g}% of its delivered "
                    f"luminosity; those blocks are left out.",
               f"If the BAD blocks hold {T:g}% or less of the delivered luminosity, the run goes on the all-physics "
               f"good-run list without them."),
            _v(rng, f"Above {T:g}%, a run with a tracker, trigger or magnet defect is BAD as a whole; a run with a calorimeter "
                    f"or muon defect may enter only the restricted lists of analyses that do not use that system.",
               f"Beyond {T:g}%: tracker, trigger or magnet defects make the whole run BAD; calorimeter or muon defects allow "
               f"only the restricted lists for analyses that do not use the affected system."),
            _v(rng, "Defects an offline correction can fix (a response calibration derived from monitoring data, a realignment "
                    "of a geometry that stayed put, masking a fixed set of noisy channels) are left out now and re-enter after "
                    "the physics reprocessing.",
               "Recoverable defects (a calibration derived from monitoring data, realignment of a geometry that stayed put, "
               "masking a fixed set of noisy channels) stay out until the physics reprocessing applies the fix."),
            _v(rng, f"No run is flagged while a defect with BAD blocks covering more than {E:g}% of its luminosity lacks the "
                    f"subsystem expert's sign-off.",
               f"A defect whose BAD blocks exceed {E:g}% of the run's luminosity needs the subsystem expert's sign-off "
               f"before the run can be flagged."),
        ]

    def _detail(self, f, rng):
        d, ok = f["defect"], f["recover_ok"]
        if d == "calib_drift":
            return (pick(rng, ["laser monitoring ran throughout the run", "the monitoring laser recorded every block of the run"])
                    if ok else pick(rng, ["laser monitoring was off for the whole run after a power-supply fault",
                                          "no monitoring-laser data were recorded during this run"]))
        if d == "alignment_shift":
            return ("the displacement appeared in one step after the magnet cycle and then stayed constant" if ok else
                    "the displacement kept changing through the run (thermal drift of the support structure)")
        if d == "noisy_channels":
            return (f"the same {f['noisy_count']} channels are noisy in every affected block" if ok else
                    "noise comes in coherent bursts across whole readout crates, hitting different channels each time")
        if d == "hv_trip":
            return pick(rng, ["the modules tripped and were only re-ramped after the listed blocks",
                              "HV channels tripped on a beam-loss spike and recovered automatically after the listed blocks"])
        if d == "dead_channels":
            return pick(rng, ["a front-end power board failed; the channels gave no signal in the listed blocks",
                              "an optical link went dark; the channels were read out as empty"])
        if d == "timing_shift":
            return pick(rng, ["a firmware reload shifted the trigger timing; restored after the listed blocks",
                              "a clock-phase jump after a resync; retimed by the trigger on-call afterwards"])
        if d == "readout_errors":
            return pick(rng, ["one readout link sent corrupted fragments; the events are kept with that fragment empty",
                              "a desynchronised readout board corrupted fragments until it was reset"])
        return pick(rng, ["stable beams were declared before the solenoid finished ramping",
                          "the solenoid was still ramping after a fast discharge earlier in the fill"])

    def _comment(self, f, rng):
        part, st = f["partition"], f["shifter_stance"]
        bad = self._bad(f)
        if st == "consistent":
            st = "bad" if bad else ("fine" if bad is False else "neutral")
        if st == "fine":
            txt = pick(rng, ["Histograms compatible with the reference, I would call it green.",
                             f"Looks OK to me, the {part} hole is small.",
                             f"Nothing worrying from the DQ side, {part} came back by itself.",
                             "Only a cosmetic issue; physics objects should be fine.",
                             f"Compared with run {f['run'] - rng.randint(3, 40)}, no visible difference in the physics plots."])
        elif st == "bad":
            txt = pick(rng, [f"I would mark the whole run bad, {part} looked awful.",
                             f"Recommend BAD, not comfortable with {part}.",
                             "Not sure we can trust anything from this run.",
                             f"Physics objects around {part} look off; my vote is bad."])
        else:
            txt = pick(rng, [f"{part} showed the problem from LB {f['lb_first']} to LB {f['lb_last']}; on-call informed.",
                             f"Flagged {part} in the online DQ; details in the defect report.",
                             f"Reported the {part} issue to the subsystem on-call, see defect entry."])
        return noisy(txt, rng, 0.15)

    def _signoff_text(self, f):
        s = f["expert_signoff"]
        if s == "signed":
            return f"signed by {f['expert']} (defect confirmed as reported)"
        if s == "pending":
            return f"requested from {f['expert']} {f['signoff_wait_h']} h ago, no answer yet"
        return "not requested"

    def _measured(self, f):
        lv = self._level_text(f)
        if lv is not None:
            return lv
        return {"narrow": "not computed yet (shifter: a handful of channels)",
                "wide": "not computed yet (shifter: most of the partition looked affected)"}[f["extent_hint"]]

    # ------------------------------------------------------------------ render
    def render(self, f, rng, surface):
        sub_name = SUBSYS[f["subsystem"]][0]
        rules = self._rules(f, rng)
        detail = self._detail(f, rng)
        comment = self._comment(f, rng)
        lbs = f"LB {f['lb_first']}-{f['lb_last']}"
        nblk = f["lb_last"] - f["lb_first"] + 1
        if surface == "json":
            return dumps(prune({
                "run": {"detector": f["detector"], "facility": f["facility"], "number": f["run"], "fill": f["fill"],
                        "collisions": f"pp at {f['ecm_tev']} TeV", "start_utc": fmt_ts(f["start"], style="iso"),
                        "duration_hours": f["dur_h"], "lumiblocks": f["n_lb"], "lumiblock_seconds": f["lb_s"],
                        "delivered_pb": f["delivered_pb"], "recorded_pb": f["recorded_pb"], "mean_pileup": f["pileup"]},
                "defect": {"subsystem": sub_name, "partition": f["partition"], "type": self.LABEL[f["defect"]],
                           "measured": self._measured(f), "blocks": lbs, "n_blocks": nblk,
                           "luminosity_pb": f["affected_pb"], "luminosity_share_pct": f["affected_lumi_pct"],
                           "detail": detail},
                "other_defects": [f["minor_defect"]] if f["minor_defect"] else None,
                "review": {"shifter": f["shifter"], "shifter_comment": comment,
                           "expert_signoff": self._signoff_text(f)},
                "criteria": {"version": f["policy_version"], "rules": rules},
            }))
        if surface == "signoff_sheet":
            head = kv_lines([
                ("Run / fill", f"{f['run']} / {f['fill']}"),
                ("Collisions", f"pp, {f['ecm_tev']} TeV ({f['facility']})"),
                ("Start", fmt_ts(f["start"], rng)),
                ("Length", f"{f['dur_h']} h, {f['n_lb']} blocks of {f['lb_s']} s"),
                ("Delivered / recorded", f"{f['delivered_pb']} / {f['recorded_pb']} pb^-1"),
                ("Mean pile-up", f["pileup"]),
            ], rng)
            defects = [f"{sub_name}, {f['partition']}: {self.LABEL[f['defect']]}; {self._measured(f)}; "
                       f"{lbs} ({nblk} blocks, {f['affected_pb']} pb^-1). {detail[0].upper() + detail[1:]}."]
            if f["minor_defect"]:
                defects.append(f["minor_defect"][0].upper() + f["minor_defect"][1:] + ".")
            return "\n".join([
                f"DATA-QUALITY SIGN-OFF -- {f['detector']} run {f['run']}", head, "",
                "Open defects:", "\n".join(f"  {i}. {d}" for i, d in enumerate(defects, 1)), "",
                f"Shifter ({f['shifter']}): \"{comment}\"",
                f"Expert sign-off: {self._signoff_text(f)}", "",
                f"Criteria in force ({f['policy_version']}):",
                "\n".join(f"  ({chr(97 + i)}) {r}" for i, r in enumerate(rules)),
            ])
        # shift_elog
        t0 = f["start"] + timedelta(seconds=f["lb_s"] * (f["lb_first"] - 1))
        t1 = f["start"] + timedelta(seconds=f["lb_s"] * f["lb_last"])
        t2 = t1 + timedelta(minutes=rng.randint(20, 240))
        role = pick(rng, ["DQ shifter", "data-quality shift", "DQM shift"])
        entries = [
            (t0, f"{f['shifter']} ({role})",
             f"Run {f['run']}: {sub_name} {f['partition']} -- {self.LABEL[f['defect']]} starting at LB {f['lb_first']}. "
             f"{detail[0].upper() + detail[1:]}."),
            (t1, f"{f['shifter']} ({role})",
             f"Issue over at LB {f['lb_last']}. Measured: {self._measured(f)}. Affected luminosity "
             f"{f['affected_pb']} pb^-1 of {f['delivered_pb']} pb^-1 delivered. {comment}"),
        ]
        if f["expert_signoff"] == "signed":
            entries.append((t2, f"{f['expert']} ({SUBSYS[f['subsystem']][1]} expert)",
                            pick(rng, ["Confirmed the defect as reported, numbers above are right.",
                                       "Checked offline: defect confirmed, extent as logged."])))
        elif f["expert_signoff"] == "pending":
            entries.append((t2, f"{f['shifter']} ({role})", f"Asked {f['expert']} for the expert sign-off, waiting."))
        if f["minor_defect"]:
            entries.append((t2 + timedelta(minutes=5), f"{f['shifter']} ({role})", f"Also: {f['minor_defect']}."))
        log = "\n".join(f"[{fmt_ts(t, style='log')}] {who}: {txt}" for t, who, txt in entries)
        return (f"e-log :: {f['detector']} :: run {f['run']} (fill {f['fill']}, pp {f['ecm_tev']} TeV, {f['n_lb']} x {f['lb_s']} s blocks)\n"
                f"{log}\n\nPinned: criteria {f['policy_version']}\n" + bullets(rules, rng))

    def questions(self):
        return {
            "run_flag": choice("How should this run be flagged for physics?", {
                "bad_whole_run": "Mark the whole run bad for physics.",
                "good_full_run": "Mark the whole run good for every physics analysis.",
                "good_minus_lumiblocks": "Mark it good but leave the affected luminosity blocks out of the good-run list.",
                "hold_for_expert": "Leave the run unflagged until the missing measurement or expert sign-off is in.",
                "restricted_list_only": "Release it only to analyses that do not use the affected subsystem.",
            }),
            "defect_impact": score("How much does the defect degrade physics objects reconstructed in the affected blocks?", [
                "Negligible: below the level any analysis would notice.",
                "Minor: a small loss of efficiency or resolution that the standard corrections absorb.",
                "Significant: one class of physics objects is biased or lost in those blocks.",
                "Severe: most analyses could not use those blocks at all.",
            ]),
            "flag_certainty": score("How firmly does the evidence on file support a final flag right now?", [
                "Unsettled: a key measurement is missing or the evidence contradicts itself.",
                "Tentative: the evidence leans one way, but checks the criteria require are outstanding.",
                "Firm: the evidence supports a flag, with minor caveats.",
                "Conclusive: the measurements, the expert review and the written criteria all point the same way.",
            ]),
            "criteria_met_full_range": noul(
                "The run satisfies the written good-run criteria over its full range of luminosity blocks.",
                yes="Every block passes each written threshold, so no block has to be left out.",
                no="At least one block fails a threshold, or the measurement needed to show it passes is missing."),
            "recoverable_offline": noul(
                "The recorded defect can be corrected offline in the physics reprocessing."),
        }

    def intended(self, f):
        bad = self._bad(f)
        lumi = f["affected_lumi_pct"]
        grp = SUBSYS[f["subsystem"]][1]
        needs_sig = bad is True and lumi > f["signoff_pct"] and f["expert_signoff"] != "signed"
        if bad is None or needs_sig:
            flag = "hold_for_expert"
        elif not bad:
            flag = "good_full_run"
        elif lumi <= f["tol_pct"]:
            flag = "good_minus_lumiblocks"
        elif grp in ("tracker", "trigger", "magnet"):
            flag = "bad_whole_run"
        else:
            flag = "restricted_list_only"
        if f["defect"] == "magnet_ramp":
            imp = 3
        elif f["magnitude"] is None:
            imp = 2 if f["extent_hint"] == "wide" else 1
        else:
            r = f["magnitude"] / f["limit"]
            imp = 0 if r < 0.5 else 1 if r <= 1.0 else 2 if r < 2.5 else 3
            if f["subsystem"] == "fcal":
                imp = max(0, imp - 1)
        c = 3
        if bad is None:
            c -= 2
            if f["expert_signoff"] != "signed":
                c -= 1
        if needs_sig:
            c -= 1
        if f["magnitude"] is not None and f["defect"] != "magnet_ramp" and abs(f["magnitude"] - f["limit"]) <= 0.12 * f["limit"]:
            c -= 1
        if bad is True and abs(lumi - f["tol_pct"]) <= 0.15 * f["tol_pct"]:
            c -= 1
        if (f["shifter_stance"] == "fine" and bad is True) or (f["shifter_stance"] == "bad" and bad is False):
            c -= 1
        rec = f["defect"] in self.RECOVERABLE and f["recover_ok"]
        return {"run_flag": flag, "defect_impact": imp, "flag_certainty": max(0, c),
                "criteria_met_full_range": "yes" if bad is False else "no",
                "recoverable_offline": "yes" if rec else "no"}


# =========================================================================== schema 2
FIXED_TARGET = [  # (experiment, facility, hall)
    ("DUNLIN", "Ostrander Electron Accelerator", "Hall 2"), ("KITTIWAKE", "Ravensmoor Continuous Beam Facility", "Hall C"),
    ("AVOCET", "Brannock Electron Facility", "End Station B"), ("STINT", "Tarnholm Accelerator Centre", "Experimental Area 4"),
    ("WAGTAIL", "Holtby Linear Accelerator", "Hall A2"), ("BITTERN", "Ravensmoor Continuous Beam Facility", "Hall D"),
    ("TWITE", "Carrow Electron Ring", "North Hall"), ("ROSEFINCH", "Ostrander Electron Accelerator", "Hall 3"),
]
TARGETS = {  # key: (label, cryogenic)
    "LH2": ("liquid hydrogen", True), "LD2": ("liquid deuterium", True), "C12": ("carbon foil", False),
    "He3": ("polarised helium-3 cell", False), "Ca40": ("calcium-40 foil", False), "Pb208": ("lead-208 foil", False),
}


class FixedTargetRunQuality(Schema):
    name = "fixed_target_run_quality"
    description = ("A production run from a fixed-target nuclear-physics experiment checked against the written "
                   "run-selection rules for a cross-section measurement.")
    surfaces = {"json": 0.5, "run_summary": 0.25, "shift_log": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        exp, fac, hall = pick(rng, FIXED_TARGET)
        tgt = pick_w(rng, {"LH2": 4, "LD2": 3, "C12": 1.5, "He3": 1, "Ca40": 1, "Pb208": 1})
        cryo = TARGETS[tgt][1]
        S, ct = pick(rng, [90, 92, 95]), pick(rng, [2, 3, 5])
        LT = pick(rng, [90, 92, 95])
        LTF = LT - pick(rng, [8, 10, 12])
        D, A = pick(rng, [7, 14, 21, 30]), pick(rng, [0.3, 0.5, 1.0])
        B = pick(rng, [0.5, 1.0, 1.5])
        M = pick(rng, [0.05, 0.1, 0.2])
        # which quantity carries the case: clean, one near a limit, one over a limit, or the configuration
        # each input drawn on its own: runs often carry one problem, sometimes two; calibrations fall due often
        dist = {"current": {"safe": 60, "near": 18, "mild": 14, "gross": 8},
                "live": {"safe": 55, "near": 22, "mild": 15, "gross": 8},
                "bcm_age": {"safe": 46, "near": 38, "mild": 10, "gross": 6},
                "bcm_agree": {"safe": 74, "near": 18, "mild": 5, "gross": 3},
                "density": {"safe": 58, "near": 20, "mild": 14, "gross": 8},
                "config": {"safe": 84, "near": 8, "gross": 8}}
        items = [k for k in dist if cryo or k != "density"]
        if reg == "clear":
            modes = {k: "safe" for k in items}
            if chance(rng, 0.55):
                modes[pick(rng, items)] = "gross"
        else:
            modes = {k: pick_w(rng, dist[k]) for k in items}
            if reg == "conflicting" and all(m == "safe" for m in modes.values()):
                k = pick(rng, items)
                modes[k] = "gross" if k == "config" else pick(rng, ["mild", "gross"])
        stable = {"safe": between(rng, S + 1.5, 99.8, digits=1), "near": near(rng, S, 2.5, hi=99.9, digits=1),
                  "mild": between(rng, S - 9.5, S - 1, digits=1), "gross": between(rng, S - 30, S - 11, digits=1)}[modes["current"]]
        live = {"safe": between(rng, LT + 0.8, 99.4, digits=1), "near": near(rng, LT, 1.5, hi=99.5, digits=1),
                "mild": between(rng, LTF + 0.5, LT - 0.6, digits=1), "gross": between(rng, LTF - 14, LTF - 0.6, digits=1)}[modes["live"]]
        age = {"safe": rng.randint(1, max(2, D - 2)), "near": near(rng, D + 1, 3, lo=1, integer=True),
               "mild": rng.randint(D + 1, D + 6), "gross": rng.randint(D + 7, 3 * D + 10)}[modes["bcm_age"]]
        agree = {"safe": between(rng, 0.03, 0.7 * A), "near": near(rng, A, 0.25 * A, lo=0.02),
                 "mild": between(rng, 1.1 * A, 1.8 * A), "gross": between(rng, 2.2 * A, 5 * A)}[modes["bcm_agree"]]
        if cryo:
            dens = {"safe": between(rng, 0.1 * B, 0.75 * B), "near": near(rng, B, 0.2 * B, lo=0.05),
                    "mild": between(rng, 1.1 * B, 1.9 * B), "gross": between(rng, 2.2 * B, 4 * B)}[modes["density"]]
        else:
            dens = None
        mom_dev = {"safe": between(rng, 0.0, 0.6 * M, digits=3), "near": near(rng, M, 0.3 * M, lo=0.0, digits=3),
                   "mild": between(rng, 1.3 * M, 3 * M, digits=3), "gross": between(rng, 1.3 * M, 4 * M, digits=3)}[modes["config"]]
        tgt_mismatch = modes["config"] == "gross" and chance(rng, 0.5)
        if tgt_mismatch:
            mom_dev = between(rng, 0.0, 0.5 * M, digits=3)
        age_missing = reg == "conflicting" and modes.get("bcm_age") != "safe" and chance(rng, 0.4)
        p_plan = between(rng, 0.8, 6.0, digits=3)
        p_rec = round(p_plan * (1 + pick(rng, [1, -1]) * mom_dev / 100), 4)
        current = pick(rng, [15, 20, 30, 40, 50, 60, 75]) if tgt != "He3" else pick(rng, [10, 12, 15])
        beam_min = rng.randint(35, 125)
        charge = round(current * beam_min / 60 * 3.6 * min(1.0, stable / 100 + 0.04), 1)
        goal = round(charge * pick(rng, [4, 6, 9, 14, 25]) * rng.uniform(0.8, 1.25), -1)
        if chance(rng, 0.2):
            collected = round(goal * rng.uniform(1.0, 1.2), 1)
        else:
            collected = round(max(0.0, goal - charge * pick(rng, [1.2, 2, 3, 5, 9, 20]) * rng.uniform(0.8, 1.2)), 1)
        crew_view = None
        if reg == "conflicting":
            crew_view = "fine"
        elif chance(rng, 0.3):
            crew_view = pick(rng, ["fine", "worried"])
        return {
            "regime": reg, "experiment": exp, "facility": fac, "hall": hall, "run": rng.randint(1200, 48999),
            "start": rand_ts(rng), "beam_minutes": beam_min, "energy_gev": pick(rng, [2.2, 4.4, 6.6, 8.8, 10.6]),
            "current_ua": current, "target": tgt, "target_len_cm": pick(rng, [4, 10, 15, 20]) if cryo else None,
            "target_mismatch": tgt_mismatch, "setting": f"K{rng.randint(1, 24)}", "angle_deg": between(rng, 8, 48, digits=1),
            "p_plan": p_plan, "p_rec": p_rec, "mom_dev_pct": mom_dev,
            "current_band_pct": ct, "stable_req": S, "stable_pct": stable, "trips": rng.randint(0, 28),
            "lt_min": LT, "lt_floor": LTF, "live_pct": live, "bcm_max_days": D, "bcm_tol_pct": A,
            "bcm_age_days": None if age_missing else age, "bcm_agree_pct": agree,
            "density_lim": B, "density_rms_pct": dens, "mom_tol_pct": M,
            "charge_mc": charge, "goal_mc": goal, "collected_mc": collected,
            "crew_view": crew_view, "shift_leader": person(rng), "coordinator": person(rng),
            "modes": modes, "policy_version": _policy_id(rng, "RS"),
        }

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _charge_ok(f):
        return f["bcm_age_days"] is not None and f["bcm_age_days"] <= f["bcm_max_days"] and f["bcm_agree_pct"] <= f["bcm_tol_pct"]

    @staticmethod
    def _mismatch(f):
        return f["target_mismatch"] or f["mom_dev_pct"] > f["mom_tol_pct"]

    def _rules(self, f, rng):
        S, ct, LT, LTF, D, A, M = (f["stable_req"], f["current_band_pct"], f["lt_min"], f["lt_floor"], f["bcm_max_days"],
                                   f["bcm_tol_pct"], f["mom_tol_pct"])
        B = f["density_lim"]
        return [
            _v(rng, f"(a) Beam current within +/-{ct}% of nominal for at least {S}% of beam-on time. Runs missing this by up "
                    f"to 10 points join the production sample only with the current-excursion cut; runs missing it by more are excluded.",
               f"(a) Current stability: at least {S}% of beam-on time within +/-{ct}% of nominal. A shortfall of up to 10 "
               f"points means production only with the current-excursion cut; a larger shortfall excludes the run."),
            _v(rng, f"(b) DAQ live time of at least {LT}%. Between {LTF}% and {LT}% the run joins only with the measured "
                    f"dead-time correction; below {LTF}% it is excluded.",
               f"(b) Live time: {LT}% or better. From {LTF}% up to {LT}% the measured dead-time correction must be applied; "
               f"under {LTF}% the run is excluded."),
            _v(rng, f"(c) Charge normalisation: the beam-charge monitors calibrated within the last {D} days and agreeing with "
                    f"each other within {A:g}%. Otherwise the run is for systematic studies only until recalibrated.",
               f"(c) Beam charge: monitor calibration no older than {D} days and the two monitors within {A:g}% of each "
               f"other; failing either, the run serves systematic studies only until recalibrated."),
            _v(rng, f"(d) Cryogenic targets only: luminosity-monitor RMS below {B:g}%; above it the run is for systematic "
                    f"studies only, above {2 * B:g}% it is excluded.",
               f"(d) Applies to cryogenic targets: a luminosity-monitor RMS above {B:g}% limits the run to systematic "
               f"studies, above {2 * B:g}% it is excluded."),
            _v(rng, f"(e) Spectrometer momentum within {M:g}% of the run plan and the target-ladder readback matching the "
                    f"logged target; otherwise the run is held until the run coordinator resolves it from the logbook.",
               f"(e) Configuration: momentum readback within {M:g}% of the planned setting and the ladder readback equal to "
               f"the logged target; any mismatch holds the run until the run coordinator settles it from the logbook."),
            _v(rng, "Where several rules apply, the most restrictive outcome wins; a hold under (e) comes first.",
               "If more than one rule bites, the most restrictive outcome applies, and an (e) hold takes precedence."),
        ]

    @staticmethod
    def _readback(f):
        if f["target_mismatch"]:
            return "empty cell" if TARGETS[f["target"]][1] else "dummy frame"
        return f["target"]

    def _crew_note(self, f, rng):
        v = f["crew_view"]
        if v == "fine":
            return noisy(pick(rng, ["Smooth run, nothing special.", "Good run from our side.",
                                    "Quiet shift, run looked fine on the online plots.",
                                    "No problems noticed, beam was decent."]), rng, 0.15)
        if v == "worried":
            return noisy(pick(rng, ["Not happy with this one, please check before using it.",
                                    "Several things looked odd during this run.",
                                    "Recommend a careful look at this run."]), rng, 0.15)
        return None

    # ------------------------------------------------------------------ render
    def render(self, f, rng, surface):
        tlabel = TARGETS[f["target"]][0]
        tgt = f"{f['target']} ({tlabel}{', ' + str(f['target_len_cm']) + ' cm' if f['target_len_cm'] else ''})"
        rules = self._rules(f, rng)
        crew = self._crew_note(f, rng)
        readback = self._readback(f)
        age_txt = show(f["bcm_age_days"], rng, lambda v: f"{v} days ago")
        if surface == "json":
            return dumps(prune({
                "run": {"experiment": f["experiment"], "facility": f["facility"], "hall": f["hall"], "number": f["run"],
                        "started": fmt_ts(f["start"], style="iso"), "beam_on_minutes": f["beam_minutes"],
                        "beam": {"particle": "electrons", "energy_gev": f["energy_gev"], "nominal_current_ua": f["current_ua"]},
                        "target": {"logged": tgt, "ladder_readback": readback},
                        "kinematics": {"setting": f["setting"], "angle_deg": f["angle_deg"],
                                       "planned_momentum_gev": f["p_plan"], "recorded_momentum_gev": f["p_rec"]}},
                "quality": {"current_in_band_pct": f["stable_pct"], "beam_trips": f["trips"],
                            "daq_live_time_pct": f["live_pct"],
                            "charge_monitors": {"last_calibration": None if f["bcm_age_days"] is None else age_txt,
                                                "calibration_record": "not found" if f["bcm_age_days"] is None else None,
                                                "monitor_disagreement_pct": f["bcm_agree_pct"]},
                            "target_density_rms_pct": f["density_rms_pct"]},
                "statistics": {"run_charge_mc": f["charge_mc"], "setting_goal_mc": f["goal_mc"],
                               "setting_collected_before_run_mc": f["collected_mc"]},
                "shift_note": crew,
                "selection_rules": {"version": f["policy_version"], "text": rules},
            }))
        if surface == "run_summary":
            rows = [("Run", f"{f['run']}  ({f['experiment']}, {f['hall']}, {f['facility']})"),
                    ("Start", fmt_ts(f["start"], rng)), ("Beam on", f"{f['beam_minutes']} min"),
                    ("Beam", f"e- {f['energy_gev']} GeV, nominal {f['current_ua']} uA"),
                    ("Target (logged)", tgt), ("Target ladder readback", readback),
                    ("Setting", f"{f['setting']}: theta = {f['angle_deg']} deg"),
                    ("Momentum set / readback", f"{f['p_plan']:.3f} / {f['p_rec']:.4f} GeV/c"),
                    ("Current in +/-band", f"{f['stable_pct']}% of beam-on time ({f['trips']} trip{'' if f['trips'] == 1 else 's'})"),
                    ("Live time", f"{f['live_pct']}%"),
                    ("Charge monitors", f"calibrated {age_txt}; disagreement {f['bcm_agree_pct']}%"),
                    ("Lumi-monitor RMS", None if f["density_rms_pct"] is None else f"{f['density_rms_pct']}%"),
                    ("Charge this run", f"{f['charge_mc']} mC"),
                    ("Setting charge", f"{f['collected_mc']} of {f['goal_mc']:.0f} mC goal before this run")]
            body = kv_lines(rows, rng, style=pick(rng, ["aligned", "colon"]))
            tail = f"\n\nShift leader note: {crew}" if crew else ""
            return (f"=== END-OF-RUN SUMMARY ===\n{body}{tail}\n\nRun selection rules {f['policy_version']}:\n"
                    + "\n".join(rules))
        # shift_log
        t = f["start"]
        t_end = t + timedelta(minutes=f["beam_minutes"] + rng.randint(2, 12))
        step = max(3, f["beam_minutes"] // 5)
        lines = [(t, f"Run {f['run']} started. {tlabel} target{', ' + str(f['target_len_cm']) + ' cm' if f['target_len_cm'] else ''}, "
                     f"setting {f['setting']} ({f['angle_deg']} deg, {f['p_plan']:.3f} GeV/c), {f['current_ua']} uA requested.")]
        t += timedelta(minutes=rng.randint(step // 2 + 1, step))
        if f["stable_pct"] < f["stable_req"] - 5:
            lines.append((t, f"Beam unstable, {f['trips']} trips; current within band only {f['stable_pct']}% of beam-on time."))
        elif f["trips"] >= 8:
            lines.append((t, f"Beam tripping a lot ({f['trips']} trips so far), current in band {f['stable_pct']}% of the time."))
        else:
            lines.append((t, f"Beam ok-ish, {f['trips']} trips; current within band {f['stable_pct']}% of beam-on time."))
        t += timedelta(minutes=rng.randint(step // 2 + 1, step))
        lines.append((t, f"DAQ live time running at {f['live_pct']}%."))
        if f["density_rms_pct"] is not None:
            t += timedelta(minutes=rng.randint(step // 2 + 1, step))
            lines.append((t, f"Lumi monitor RMS {f['density_rms_pct']}% (target loop {pick(rng, ['steady', 'slightly noisy', 'fan speed raised once'])})."))
        t += timedelta(minutes=rng.randint(step // 2 + 1, step))
        lines.append((t, f"Ladder readback says '{readback}'. Spectrometer momentum readback {f['p_rec']:.4f} GeV/c."))
        lines.append((t_end, f"Run ended after {f['beam_minutes']} min of beam, {f['charge_mc']} mC. "
                         f"Charge monitors last calibrated: {age_txt}, disagreement {f['bcm_agree_pct']}%."
                         + (f" {crew}" if crew else "")))
        log = "\n".join(f"{fmt_ts(x, style='log')}  [{_first(f['shift_leader']).lower()}] {noisy(y, rng, 0.08)}" for x, y in lines)
        stats = (f"Setting {f['setting']}: goal {f['goal_mc']:.0f} mC, {f['collected_mc']} mC collected before this run.")
        return (f"{f['experiment']} shift log, {f['hall']} ({f['facility']})\n{log}\n\n{stats}\n\nRules ({f['policy_version']}):\n"
                + bullets(rules, rng))

    def questions(self):
        return {
            "run_classification": choice("How should this run be classified for the cross-section analysis?", {
                "exclude": "Exclude it from the analysis altogether.",
                "good_with_correction": "Accept it into the production sample with the correction the rules prescribe.",
                "production_good": "Accept it into the production sample as it stands.",
                "recheck_logbook": "Hold it until the run coordinator settles what configuration was actually used.",
                "systematics_only": "Keep it out of the production sample but use it for systematic studies.",
            }),
            "followup_owner": choice("Which group has to act before this run's status can be finalised?", {
                "beam_and_charge_monitors": "The beam-instrumentation group: current stability or the charge-monitor calibration needs attention.",
                "daq_group": "The data-acquisition group: the dead time needs to be understood or corrected.",
                "none_required": "No one: the run needs nothing beyond the normal sign-off.",
                "run_coordination": "The run coordinator: the logged configuration has to be reconciled with the readbacks.",
                "target_group": "The cryogenic-target group: the target density was not stable enough.",
            }),
            "normalisation_reliability": score("How reliable is the run's normalisation (beam charge, live time and target density)?", [
                "Unreliable: at least one input cannot be trusted for this run.",
                "Doubtful: an input needs a correction whose size is uncertain.",
                "Sound with corrections: the inputs are understood and any corrections are small.",
                "Solid: charge, live time and target density are all comfortably inside their limits.",
            ]),
            "statistical_weight": score("How much does this run matter for its kinematic setting's statistics?", [
                "Negligible: the setting already has its planned charge, or the run adds almost nothing.",
                "Modest: a small share of what the setting still needs.",
                "Substantial: a sizeable share of the setting's remaining charge.",
                "Critical: this run alone covers a large part of what the setting still lacks.",
            ]),
            "charge_record_trustworthy": noul(
                "The accumulated beam charge recorded for this run can be trusted for normalisation.",
                yes="The charge monitors are within their calibration period and agree within the written tolerance.",
                no="The calibration is too old or not on record, or the two monitors disagree by more than the tolerance."),
        }

    def intended(self, f):
        S, LT, LTF = f["stable_req"], f["lt_min"], f["lt_floor"]
        B = f["density_lim"]
        dens = f["density_rms_pct"]  # None for solid targets: rule (d) does not apply
        charge_ok = self._charge_ok(f)
        if self._mismatch(f):
            cls = "recheck_logbook"
        elif f["stable_pct"] < S - 10 or f["live_pct"] < LTF or (dens is not None and dens > 2 * B):
            cls = "exclude"
        elif not charge_ok or (dens is not None and dens > B):
            cls = "systematics_only"
        elif f["live_pct"] < LT or f["stable_pct"] < S:
            cls = "good_with_correction"
        else:
            cls = "production_good"
        # owner: the most serious failing item
        if self._mismatch(f):
            owner = "run_coordination"
        else:
            sev = {}
            if f["stable_pct"] < S:
                sev["beam_and_charge_monitors"] = (S - f["stable_pct"]) / 10
            if not charge_ok:
                a = f["bcm_age_days"]
                s = 0.75 if a is None else max((a - f["bcm_max_days"]) / f["bcm_max_days"] + 0.5 if a > f["bcm_max_days"] else 0,
                                               (f["bcm_agree_pct"] - f["bcm_tol_pct"]) / f["bcm_tol_pct"] + 0.5
                                               if f["bcm_agree_pct"] > f["bcm_tol_pct"] else 0)
                sev["beam_and_charge_monitors"] = max(sev.get("beam_and_charge_monitors", 0), s)
            if f["live_pct"] < LT:
                sev["daq_group"] = (LT - f["live_pct"]) / (LT - LTF)
            if dens is not None and dens > B:
                sev["target_group"] = (dens - B) / B + 0.5
            owner = max(sev, key=sev.get) if sev else "none_required"
        # normalisation reliability: a gross failure of any input makes it unreliable; a moderate charge or
        # density problem makes it doubtful; a measured dead-time correction keeps it sound
        a, D, A = f["bcm_age_days"], f["bcm_max_days"], f["bcm_tol_pct"]
        gross = (f["live_pct"] < LTF or (dens is not None and dens > 2 * B) or a is None or a > 2 * D
                 or f["bcm_agree_pct"] > 2 * A)
        if gross:
            n = 0
        elif not charge_ok or (dens is not None and dens > B):
            n = 1
        elif f["live_pct"] < LT:
            n = 2
        else:
            n = 3
        if n == 3:
            close = (f["live_pct"] - LT < 0.1 * (100 - LT) or (f["bcm_age_days"] or 0) >= 0.9 * f["bcm_max_days"]
                     or f["bcm_agree_pct"] >= 0.85 * f["bcm_tol_pct"] or (dens is not None and dens >= 0.85 * B))
            if close:
                n = 2
        # statistical weight
        missing = f["goal_mc"] - f["collected_mc"]
        if missing <= 0 or f["charge_mc"] < 0.03 * missing:
            w = 0
        else:
            frac = f["charge_mc"] / missing
            w = 1 if frac < 0.15 else 2 if frac < 0.4 else 3
        return {"run_classification": cls, "followup_owner": owner, "normalisation_reliability": max(0, n),
                "statistical_weight": w, "charge_record_trustworthy": "yes" if charge_ok else "no"}


# =========================================================================== schema 3
NU_DETECTORS = [  # (name, lab, technology, inner PMTs, beam)
    ("SHEARWATER", "Elsby Underground Laboratory", "water Cherenkov", 9850, "Kessick neutrino beam"),
    ("GUILLEMOT", "Brannock Deep Site", "liquid scintillator", 17600, "Ostmark neutrino beam"),
    ("WHIMBREL", "Holtby Mine Laboratory", "water-based liquid scintillator", 4960, "Vantor neutrino beam"),
    ("MURRELET", "Carrow Deep Laboratory", "water Cherenkov", 12400, "Mirefield neutrino beam"),
    ("EIDER", "Tolland Underground Facility", "liquid scintillator", 8192, "Calder neutrino beam"),
    ("SCOTER", "Penrith Deep Laboratory", "water Cherenkov", 7300, "Lindholm neutrino beam"),
]
MARKERS = {
    "water Cherenkov": ["Michel-electron spectrum endpoint", "through-going muon light yield"],
    "liquid scintillator": ["Po-214 alpha peak", "neutron-capture gamma peak"],
    "water-based liquid scintillator": ["neutron-capture gamma peak", "through-going muon light yield"],
}
SOURCES = ["Am-Be neutron source", "Co-60 gamma source", "laser diffuser ball", "Ni-Cf gamma source", "Cs-137 source"]


class NeutrinoRunQuality(Schema):
    name = "neutrino_run_quality"
    description = ("A day-long run of an underground neutrino detector: which analysis samples may use it under the "
                   "written run-selection criteria.")
    surfaces = {"json": 0.5, "run_sheet": 0.25, "shift_chat": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        det, lab, tech, npmt, beam = pick(rng, NU_DETECTORS)
        P = pick(rng, [96.5, 97.0, 98.0, 98.5])
        x = pick(rng, [1.2, 1.3, 1.5])
        t = pick(rng, [0.5, 1.0, 1.5])
        dmax = pick(rng, [50, 100, 200, 500])
        q = pick(rng, [2, 5, 10])
        dur = between(rng, 8, 24, digits=1)
        live = round(dur * rng.uniform(0.93, 0.995), 2)
        clear, border, confl = reg == "clear", reg == "borderline", reg == "conflicting"
        # PMT coverage: the minimum operating fraction, and how long the detector sat below the requirement
        cm = (pick_w(rng, {"safe": 60, "bad": 40}) if clear else
              pick_w(rng, {"near": 75, "safe": 25}) if border else pick_w(rng, {"near": 40, "bad": 30, "safe": 30}))
        pmt_min = {"safe": between(rng, P + 0.3, 99.9, digits=2), "near": near(rng, P, 0.35, hi=99.95, digits=2),
                   "bad": between(rng, P - 6, P - 0.4, digits=2)}[cm]
        low_cov_h = 0.0
        if pmt_min < P:
            low_cov_h = round(live * (rng.uniform(0.02, 0.3) if chance(rng, 0.72) else rng.uniform(0.55, 0.95)), 1)
        # low-energy trigger rate and flashers
        lm = (pick_w(rng, {"safe": 60, "bad": 40}) if clear else
              pick_w(rng, {"near": 55, "safe": 45}) if border else pick_w(rng, {"near": 40, "safe": 60}))
        ratio = {"safe": rng.uniform(0.85, 0.95 * x), "near": near(rng, x, 0.12), "bad": rng.uniform(1.3 * x, 3 * x)}[lm]
        flasher = None
        if confl and chance(rng, 0.35):
            flasher = "unmasked"
        elif chance(rng, 0.15):
            flasher = pick(rng, ["masked", "masked", "unmasked"])
        # energy scale
        if clear:
            drift = pick(rng, [between(rng, 0.0, 0.8 * t), between(rng, 0.0, 0.8 * t), between(rng, 1.1 * t, 1.9 * t),
                               between(rng, 2.1 * t, 4 * t)])
        elif border:
            drift = near(rng, t, 0.25 * t, lo=0.0) if chance(rng, 0.55) else near(rng, 2 * t, 0.3 * t)
        else:
            drift = between(rng, 0.0, 3 * t)
        scale_missing = confl and chance(rng, 0.3)
        # timing: GPS losses, the hold-over drift, and how much of the run they cover
        has_loss = chance(rng, 0.55 if clear else 0.85)
        gps_loss, hold, beam_off, hold_missing = 0, None, False, False
        if has_loss:
            frac = near(rng, q, 0.4 * q, lo=0.2) if border else between(rng, 0.2 * q, 4 * q)
            gps_loss = max(3, int(round(frac / 100 * dur * 60)))
            if clear:
                hold = between(rng, 0.05 * dmax, 0.6 * dmax, digits=0) if chance(rng, 0.35) else between(rng, 1.5 * dmax, 6 * dmax, digits=0)
            elif border:
                hold = near(rng, dmax, 0.15 * dmax, lo=1, digits=0)
            else:
                hold = between(rng, 0.3 * dmax, 3 * dmax, digits=0)
                hold_missing = chance(rng, 0.35)
            beam_off = chance(rng, 0.1)
        src_frac = 0.0
        u = rng.random()
        if u < 0.08:
            src_frac = between(rng, 0.55, 0.95)
        elif u < 0.14:
            src_frac = near(rng, 0.5, 0.08, lo=0.3, hi=0.7)
        elif u < 0.28:
            src_frac = between(rng, 0.05, 0.3)
        view = "clean" if confl else (pick(rng, ["clean", "worried"]) if chance(rng, 0.25) else None)
        ref_rate = between(rng, 4, 40, digits=1)
        return {
            "regime": reg, "detector": det, "lab": lab, "technology": tech, "n_pmt": npmt, "beam": beam,
            "run": rng.randint(30000, 99999), "start": rand_ts(rng), "duration_h": dur, "live_h": live,
            "pmt_req_pct": P, "pmt_min_pct": pmt_min, "low_cov_hours": low_cov_h,
            "lowe_limit_x": x, "lowe_ref_hz": ref_rate, "lowe_rate_hz": round(ref_rate * ratio, 1),
            "flasher": flasher, "flasher_pmt": rng.randint(1, npmt),
            "radon_bq_m3": round(rng.uniform(8, 40) * (ratio if ratio > 1.1 else 1.0), 1),
            "marker": pick(rng, MARKERS[tech]), "scale_tol_pct": t,
            "scale_drift_pct": None if scale_missing else round(drift, 2),
            "gps_loss_min": gps_loss, "holdover_ns": None if hold_missing else hold, "holdover_max_ns": dmax,
            "beam_cut_pct": q, "beam_off_during_loss": beam_off,
            "spills": int(dur * 3600 / pick(rng, [1.2, 1.33, 2.2]) * rng.uniform(0.6, 0.97)),
            "source": pick(rng, SOURCES), "source_frac": round(src_frac, 2),
            "shift_view": view, "shifter": person(rng), "run_coordinator": person(rng),
            "policy_version": _policy_id(rng, "RQ"),
        }

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _spill_ok(f):
        if not f["gps_loss_min"] or f["beam_off_during_loss"]:
            return True
        if f["holdover_ns"] is None:
            return False
        return f["holdover_ns"] <= f["holdover_max_ns"]

    @staticmethod
    def _loss_pct(f):
        return 100 * f["gps_loss_min"] / (f["duration_h"] * 60)

    @staticmethod
    def _lowe_level(f):
        r = (f["lowe_rate_hz"] / f["lowe_ref_hz"]) / f["lowe_limit_x"]
        lv = 0 if r <= 0.8 else 1 if r <= 1.0 else 2 if r <= 1.6 else 3
        if f["flasher"] == "unmasked":
            lv = max(lv, 2)
        return lv

    @staticmethod
    def _scale_level(f):
        d = f["scale_drift_pct"]
        if d is None:
            return 2
        t = f["scale_tol_pct"]
        return 0 if d <= 0.5 * t else 1 if d <= t else 2 if d <= 2 * t else 3

    def _rules(self, f, rng):
        P, x, t, d, q = f["pmt_req_pct"], f["lowe_limit_x"], f["scale_tol_pct"], f["holdover_max_ns"], f["beam_cut_pct"]
        return [
            _v(rng, f"R1 Coverage: sub-runs with fewer than {P}% of inner PMTs operating are cut; if the cut removes more "
                    f"than half of the run's live time, the run is excluded.",
               f"R1 Coverage: any sub-run below {P}% operating inner PMTs is removed; a run losing more than half its live "
               f"time this way is excluded."),
            _v(rng, f"R2 Low-energy samples: the low-energy trigger rate stays within {x}x its reference, and no flashing "
                    f"PMT is left unmasked; otherwise the run is excluded from the low-energy samples only.",
               f"R2 Low energy: a low-energy trigger rate above {x}x reference, or an unmasked flashing PMT, removes the "
               f"run from the low-energy samples (and only from those)."),
            _v(rng, f"R3 Energy scale: the {f['marker']} within +/-{t}% of reference; drifts up to {2 * t:g}% are absorbed "
                    f"by the time-dependent calibration; beyond that the run is excluded.",
               f"R3 Energy scale, tracked with the {f['marker']}: within +/-{t}% nothing is needed, up to {2 * t:g}% the "
               f"time-dependent calibration corrects it, beyond {2 * t:g}% the run is excluded."),
            _v(rng, f"R4 Beam sample: every spill needs a GPS timestamp; during a GPS loss the rubidium hold-over is accepted "
                    f"while its accumulated drift stays below {d} ns. Spills without a valid time are dropped; if they span "
                    f"more than {q}% of the run, the whole run leaves the beam sample.",
               f"R4 Beam timing: spills are matched by GPS time; while GPS is lost, rubidium hold-over counts as valid "
               f"until its drift reaches {d} ns. Untimed spills are dropped, and if they cover more than {q}% of the run "
               f"the run leaves the beam sample."),
            _v(rng, "R5 Calibration: periods with a source deployed are cut; a run with a source in the detector for more "
                    "than half its live time is a calibration run, not a physics run.",
               "R5 Calibration: source-deployment periods are removed; with a source in for over half the live time the "
               "run counts as a calibration run."),
            _v(rng, "Every exclusion that applies is made.", "Exclusions are cumulative."),
        ]

    def _cov_text(self, f):
        s = f"minimum {f['pmt_min_pct']}% of {f['n_pmt']} inner PMTs operating"
        if f["low_cov_hours"]:
            s += f"; below {f['pmt_req_pct']}% for {f['low_cov_hours']} h in total (HV crate trips)"
        return s

    def _gps_text(self, f, rng):
        if not f["gps_loss_min"]:
            return "GPS locked for the whole run"
        s = f"GPS lock lost for {f['gps_loss_min']} min"
        if f["beam_off_during_loss"]:
            s += " (accelerator down for maintenance over that period, no spills delivered)"
        if f["holdover_ns"] is None:
            return s + "; hold-over drift estimate not available"
        return s + f"; rubidium hold-over drift estimated at {f['holdover_ns']:.0f} ns"

    def _flasher_text(self, f):
        if f["flasher"] is None:
            return None
        return (f"PMT {f['flasher_pmt']} identified as flashing; masked from the second sub-run"
                if f["flasher"] == "masked" else f"PMT {f['flasher_pmt']} flashing, not yet masked")

    def _source_text(self, f):
        if not f["source_frac"]:
            return None
        hrs = round(f["live_h"] * f["source_frac"], 1)
        return f"{f['source']} deployed for {hrs} h of {f['live_h']} h live time"

    def _view_text(self, f, rng):
        if f["shift_view"] == "clean":
            return noisy(pick(rng, ["Run looked clean to me.", "Nothing unusual on the shift plots.",
                                    "Good run, no alarms worth mentioning.", "Physics run, all green on the overview."]), rng, 0.15)
        if f["shift_view"] == "worried":
            return noisy(pick(rng, ["Some odd rates during the night, please double-check.",
                                    "I'm not sure about this run.", "Several alarms, would not trust it blindly."]), rng, 0.15)
        return None

    def render(self, f, rng, surface):
        rules = self._rules(f, rng)
        scale = (f"{f['scale_drift_pct']:.2f}% from reference" if f["scale_drift_pct"] is not None
                 else pick(rng, ["calibration fit failed for this run", "not yet computed", "not available"]))
        if surface == "json":
            return dumps(prune({
                "run": {"detector": f["detector"], "laboratory": f["lab"], "technology": f["technology"],
                        "number": f["run"], "start_utc": fmt_ts(f["start"], style="iso"),
                        "duration_hours": f["duration_h"], "live_hours": f["live_h"], "beam": f["beam"],
                        "spills_delivered": f["spills"]},
                "detector_status": {"inner_pmts": f["n_pmt"], "min_pmts_operating_pct": f["pmt_min_pct"],
                                    "hours_below_coverage_requirement": f["low_cov_hours"] or None,
                                    "flasher": self._flasher_text(f), "calibration_source": self._source_text(f)},
                "rates": {"low_energy_trigger_hz": f["lowe_rate_hz"], "low_energy_reference_hz": f["lowe_ref_hz"],
                          "radon_cover_gas_bq_m3": f["radon_bq_m3"]},
                "energy_scale": {"marker": f["marker"], "drift": scale},
                "timing": {"gps": self._gps_text(f, rng)},
                "shift_comment": self._view_text(f, rng),
                "criteria": {"version": f["policy_version"], "rules": rules},
            }))
        if surface == "run_sheet":
            rows = [["live time", f"{f['live_h']} h of {f['duration_h']} h"],
                    ["PMT coverage", self._cov_text(f)],
                    ["low-energy trigger", f"{f['lowe_rate_hz']} Hz (ref {f['lowe_ref_hz']} Hz)"],
                    ["radon, cover gas", f"{f['radon_bq_m3']} Bq/m3"],
                    [f["marker"], scale], ["timing", self._gps_text(f, rng)],
                    ["spills delivered", f"{f['spills']:,}"]]
            if self._flasher_text(f):
                rows.append(["flasher", self._flasher_text(f)])
            if self._source_text(f):
                rows.append(["calibration", self._source_text(f)])
            view = self._view_text(f, rng)
            return "\n".join([f"{f['detector']} RUN SHEET -- run {f['run']} ({f['lab']}, {f['technology']})",
                              f"Started {fmt_ts(f['start'], rng)}; beam: {f['beam']}", "",
                              table(["item", "value"], rows, rng), ""]
                             + ([f"Shifter remark: {view}", ""] if view else [])
                             + [f"Selection criteria {f['policy_version']}:"] + rules)
        # shift_chat
        sh, rc = f["shifter"], f["run_coordinator"]
        msgs = [(sh, f"run {f['run']} closed after {f['duration_h']} h, {f['live_h']} h live. {self._cov_text(f)}"),
                (sh, f"low-E trigger averaged {f['lowe_rate_hz']} Hz vs {f['lowe_ref_hz']} Hz reference, radon {f['radon_bq_m3']} Bq/m3")]
        if self._flasher_text(f):
            msgs.append((sh, self._flasher_text(f)))
        msgs.append((rc, f"what did the {f['marker']} do?"))
        msgs.append((sh, scale))
        msgs.append((rc, "and timing?"))
        msgs.append((sh, self._gps_text(f, rng)))
        if self._source_text(f):
            msgs.append((sh, self._source_text(f)))
        view = self._view_text(f, rng)
        if view:
            msgs.append((sh, view))
        t = f["start"] + timedelta(hours=f["duration_h"])
        out = []
        for who, txt in msgs:
            t += timedelta(minutes=rng.randint(1, 9))
            out.append(f"{t.strftime('%H:%M')} <{_first(who).lower()}> {noisy(txt, rng, 0.08)}")
        return (f"#{f['detector'].lower()}-runcoord\n" + "\n".join(out) + f"\n\n(pinned) {f['policy_version']}:\n"
                + "\n".join(rules))

    def questions(self):
        return {
            "sample_assignment": choice("Which analysis samples should this run go into?", {
                "all_physics_samples": "Every physics sample, including the low-energy and beam-neutrino samples.",
                "calibration_only": "None of the physics samples; it is treated as a calibration run.",
                "exclude_beam_and_low_energy": "Every physics sample except both the beam-neutrino and the low-energy samples.",
                "exclude_beam_sample": "Every physics sample except the beam-neutrino sample.",
                "exclude_entirely": "No sample at all; the run fails a criterion that applies to every analysis.",
                "exclude_low_energy": "Every physics sample except the low-energy samples.",
            }),
            "low_energy_background_concern": score("How much does instrumental or radon background threaten the low-energy samples in this run?", [
                "None: low-energy rates sit at their reference.",
                "Slight: a small excess that the standard cuts absorb.",
                "Marked: an excess or an unmasked flasher large enough to contaminate the low-energy samples.",
                "Severe: the low-energy rate is dominated by background or an instrumental source.",
            ]),
            "energy_scale_stability": score("How stable was the detector's energy scale over the run?", [
                "Stable: the calibration marker sits on its reference.",
                "Minor drift, inside the stated tolerance.",
                "Drift beyond the tolerance that needs the time-dependent correction, or not yet verified.",
                "Unstable: the drift is beyond what the calibration can absorb.",
            ]),
            "pmt_coverage_adequate": noul(
                "Enough photomultipliers were operating to meet the written coverage criterion throughout the run.",
                yes="The lowest operating fraction recorded during the run is at or above the stated minimum.",
                no="At some point the operating fraction fell below the stated minimum, so sub-runs are cut."),
            "spill_timing_valid": noul(
                "Every delivered beam spill in this run carries a valid time under the written timing rule."),
        }

    def intended(self, f):
        cov_ok = f["pmt_min_pct"] >= f["pmt_req_pct"]
        le = self._lowe_level(f)
        sc = self._scale_level(f)
        spill = self._spill_ok(f)
        beam_out = not spill and self._loss_pct(f) > f["beam_cut_pct"]
        if f["source_frac"] > 0.5:
            samp = "calibration_only"
        elif f["low_cov_hours"] > 0.5 * f["live_h"] or sc == 3:
            samp = "exclude_entirely"
        elif beam_out and le >= 2:
            samp = "exclude_beam_and_low_energy"
        elif beam_out:
            samp = "exclude_beam_sample"
        elif le >= 2:
            samp = "exclude_low_energy"
        else:
            samp = "all_physics_samples"
        return {"sample_assignment": samp, "low_energy_background_concern": le, "energy_scale_stability": sc,
                "pmt_coverage_adequate": "yes" if cov_ok else "no", "spill_timing_valid": "yes" if spill else "no"}


# =========================================================================== schema 4
# analysis: (name, statistically limited?, barrel only?, {subsystem: dependence 0-3}, groups, objects text).
# The objects text is what the request states; the dependence values follow from it.
ANALYSES = [
    ("high-mass dimuon resonance search", True, False,
     {"muon_dt": 3, "muon_trig": 3, "strip": 2, "pixel": 2, "ecal": 1, "hcal": 1, "fcal": 0}, ["Exotics", "Searches"],
     "two high-momentum muons measured in the muon chambers and the tracker; calorimeters only for isolation"),
    ("diphoton resonance search", True, True,
     {"ecal": 3, "pixel": 1, "strip": 1, "hcal": 1, "muon_dt": 0, "muon_trig": 0, "fcal": 0}, ["Higgs", "Exotics"],
     "two isolated photons in the barrel electromagnetic calorimeter; tracks only for conversions and the vertex"),
    ("inclusive jet cross-section measurement", False, False,
     {"hcal": 3, "ecal": 2, "fcal": 1, "strip": 1, "pixel": 1, "muon_dt": 0, "muon_trig": 0}, ["Jets and QCD", "Standard Model"],
     "jets from the hadronic and electromagnetic calorimeters; tracks and forward calorimeter only in the pile-up correction"),
    ("top-quark pair cross-section in the lepton-plus-jets channel", False, True,
     {"pixel": 3, "strip": 2, "ecal": 2, "hcal": 2, "muon_dt": 2, "muon_trig": 2, "fcal": 1}, ["Top Physics"],
     "one isolated electron or muon and four jets, two of them b-tagged with pixel-tracker vertices; barrel region only"),
    ("long-lived particle search with displaced vertices", True, True,
     {"pixel": 3, "strip": 3, "ecal": 1, "hcal": 1, "muon_dt": 1, "muon_trig": 1, "fcal": 0}, ["Exotics", "Searches"],
     "displaced vertices built from pixel and strip tracks in the barrel; calorimeters and muon chambers only in vetoes"),
    ("missing-momentum plus jet search", True, False,
     {"hcal": 3, "ecal": 2, "fcal": 2, "strip": 1, "pixel": 1, "muon_dt": 1, "muon_trig": 1}, ["Searches", "Exotics"],
     "one energetic jet and large missing momentum summed over all calorimeters, forward included; muons only as a veto"),
    ("W-boson mass measurement in the muon channel", False, True,
     {"muon_dt": 3, "strip": 3, "pixel": 2, "muon_trig": 2, "ecal": 1, "hcal": 1, "fcal": 1}, ["Electroweak", "Standard Model"],
     "one barrel muon whose momentum comes from the strip tracker and muon chambers; calorimeters only for the recoil"),
    ("B-meson lifetime measurement", False, True,
     {"pixel": 3, "strip": 3, "muon_trig": 2, "muon_dt": 2, "ecal": 0, "hcal": 0, "fcal": 0}, ["Heavy Flavour"],
     "displaced dimuon vertices from the pixel and strip tracker in the barrel, triggered and identified by the muon system"),
    ("vector-boson-fusion Higgs search", True, False,
     {"fcal": 3, "hcal": 2, "ecal": 2, "strip": 1, "pixel": 1, "muon_dt": 1, "muon_trig": 1}, ["Higgs"],
     "two forward tagging jets reaching into the forward calorimeter, plus central decay products"),
    ("dielectron resonance search", True, False,
     {"ecal": 3, "pixel": 2, "strip": 2, "hcal": 1, "muon_dt": 0, "muon_trig": 0, "fcal": 0}, ["Exotics", "Searches"],
     "two electrons: electromagnetic-calorimeter clusters matched to pixel and strip tracks"),
    ("tau-lepton pair search", True, True,
     {"hcal": 2, "ecal": 2, "pixel": 2, "strip": 2, "muon_dt": 1, "muon_trig": 1, "fcal": 1}, ["Higgs", "Searches"],
     "two hadronic tau decays (tracks plus calorimeter deposits) in the barrel and missing momentum; muons only as a veto"),
    ("Z-boson transverse-momentum measurement", False, False,
     {"muon_dt": 2, "muon_trig": 2, "strip": 2, "ecal": 2, "pixel": 1, "hcal": 1, "fcal": 0}, ["Electroweak", "Standard Model"],
     "Z bosons decaying to muon or electron pairs, each lepton reconstructed in its own system plus the tracker"),
]
DEFECT_NATURE = {
    "muon_dt": ["chambers at reduced high voltage", "a readout board dropping hits"],
    "muon_trig": ["trigger chambers with gas-mixture trouble", "a timing-misconfigured trigger sector"],
    "ecal": ["a block of crystals with stale laser corrections", "a supermodule with noisy readout"],
    "hcal": ["towers with a gain miscalibration", "a readout box masked for noise"],
    "fcal": ["a forward ring with elevated noise", "forward cells with a pedestal shift"],
    "pixel": ["pixel modules with HV off", "a pixel layer with desynchronised readout"],
    "strip": ["strip modules with HV off", "a strip rod with a cooling problem and reduced efficiency"],
}


class DefectOverrideRequest(Schema):
    name = "defect_override_request"
    description = ("An analysis team asks to use luminosity blocks the data-quality flags exclude, for one analysis "
                   "only, against the written override policy.")
    surfaces = {"json": 0.5, "request_email": 0.25, "review_form": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        det, fac, ecm = pick(rng, COLLIDERS)
        an, stat, central, use, groups, objects = pick(rng, ANALYSES)
        sub = pick_w(rng, {s: {0: 0.6, 1: 1.2, 2: 2.2, 3: 1.2}[use[s]] for s in sorted(use)})
        role = use[sub]
        if reg == "clear":
            # aligned: either a harmless system with a decent gain, or the signal system itself
            if chance(rng, 0.5):
                sub = pick(rng, [s for s in sorted(use) if use[s] <= 1] or sorted(use))
            else:
                sub = pick(rng, [s for s in sorted(use) if use[s] == 3] or sorted(use))
            role = use[sub]
        part = _partition(rng, sub)
        region = _region(part)
        min_pct = pick(rng, [0.5, 1.0, 2.0])
        route_pct = pick(rng, [1.0, 2.0, 3.0])
        dataset_fb = between(rng, 25, 310, digits=1)
        bar = min_pct / 2 if stat else min_pct
        if reg == "borderline":
            gain_pct = near(rng, bar, 0.3 * bar, lo=0.05)
        elif reg == "clear":
            gain_pct = between(rng, 2.0 * bar, 6 * bar) if chance(rng, 0.55) else between(rng, 0.1 * bar, 0.4 * bar)
        else:
            gain_pct = between(rng, 0.25 * bar, 2.5 * bar)
        gain_fb = round(dataset_fb * gain_pct / 100, 2)
        gain_pct = round(100 * gain_fb / dataset_fb, 2)
        validation = pick_w(rng, {0: 1, 1: 2, 2: 3.5, 3: 3})
        claim = validation
        if reg == "conflicting":
            claim = 3
            validation = pick(rng, [0, 1, 1, 2])
        return {
            "regime": reg, "detector": det, "facility": fac, "ecm_tev": ecm, "analysis": an, "stat_limited": stat,
            "central_only": central, "subsystem": sub, "role": role, "partition": part, "region": region,
            "nature": pick(rng, DEFECT_NATURE[sub]), "group": pick(rng, groups), "objects": objects,
            "contact": person(rng),
            "dq_coordinator": person(rng), "n_runs": rng.randint(2, 40), "first_run": rng.randint(380000, 520000),
            "dataset_fb": dataset_fb, "gain_fb": gain_fb, "gain_pct": gain_pct, "min_pct": min_pct,
            "route_pct": route_pct, "validation": validation, "claimed_validation": claim,
            "syst_proposed_pct": None if validation < 2 else between(rng, 0.2, 3.0, digits=1),
            "frozen": chance(rng, 0.2), "expert_dispute": chance(rng, 0.22),
            "deadline_days": pick(rng, [None, 5, 12, 21, 40, 75]), "date": rand_ts(rng).date(),
            "policy_version": _policy_id(rng, "DQO"),
        }

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _dependence(f):
        if f["central_only"] and f["region"] != "barrel":
            return 0
        return f["role"]

    @staticmethod
    def _worthwhile(f):
        return f["gain_pct"] >= f["min_pct"] or (f["stat_limited"] and f["gain_pct"] >= f["min_pct"] / 2)

    def _rules(self, f, rng):
        m, r = f["min_pct"], f["route_pct"]
        return [
            _v(rng, "1. No override for a subsystem that the analysis' signal objects depend on in the affected region.",
               "1. Overrides are refused when the analysis' key signal objects rely on the subsystem in the affected region."),
            _v(rng, "2. Otherwise, if the analysis uses the subsystem in the affected region, a quantified validation on the "
                    "affected blocks is required. An added systematic uncertainty is required when physics objects are "
                    "reconstructed directly with the subsystem, or when the validation has no closure or bias test.",
               "2. Where the analysis uses the subsystem in the affected region at all, the team must attach a quantified "
               "validation on the affected blocks; it must also add a systematic uncertainty if objects are reconstructed "
               "directly with that subsystem or if the validation lacks a closure or bias test."),
            _v(rng, f"3. Overrides adding less than {m:g}% to the analysis dataset ({m / 2:g}% for statistically limited "
                    f"analyses) are not granted when the analysis uses the affected subsystem at all.",
               f"3. If the analysis uses the subsystem at all, the gain must reach {m:g}% of its dataset ({m / 2:g}% when "
               f"the analysis is statistically limited); smaller gains are not granted."),
            _v(rng, f"4. Routing: the data-quality coordinator alone may approve overrides under {r:g}% where the subsystem "
                    "is used at most indirectly; disputed defect assessments go to the data-quality review board; any "
                    "change to a dataset already frozen for a result goes to physics coordination; everything else to the "
                    "physics group conveners.",
               f"4. Who signs: physics coordination for anything touching a dataset frozen for a result; the data-quality "
               f"review board when the defect assessment is disputed; the data-quality coordinator alone for gains under "
               f"{r:g}% with at most indirect use; the physics group conveners otherwise."),
        ]

    def _studies(self, f, rng):
        v = f["validation"]
        if v == 0:
            return []
        out = [pick(rng, ["control plots of the affected blocks against the good runs",
                          "event displays from a few affected runs", "occupancy maps of the affected region"])]
        if v >= 2:
            out.append(pick(rng, [f"data/simulation comparison of {pick(rng, ['muon', 'jet', 'photon', 'electron', 'track'])} "
                                  f"efficiency in the affected blocks, agreement within {between(rng, 0.3, 2.5, digits=1)}%",
                                  "tag-and-probe efficiency in affected vs unaffected blocks, ratio quoted with uncertainty"]))
        if v >= 3:
            out.append(pick(rng, ["closure test of the full selection on the affected blocks",
                                  "signal-injection test showing no bias from the affected blocks"]))
        return out

    def _claim_text(self, f):
        return {0: "no validation yet, plots to follow", 1: "first look shows nothing odd",
                2: "validated with a quantified comparison", 3: "fully validated, no bias seen"}[f["claimed_validation"]]

    def _context(self, f):
        bits = [f"selection: {f['objects']}"]
        bits.append("the analysis is limited by statistics" if f["stat_limited"] else "systematic uncertainties dominate the measurement")
        if f["frozen"]:
            bits.append("the dataset was frozen for a preliminary result already shown")
        if f["expert_dispute"]:
            bits.append(f"the {SUBSYS[f['subsystem']][0]} experts dispute the original BAD assessment")
        return bits

    def render(self, f, rng, surface):
        sub_name = SUBSYS[f["subsystem"]][0]
        runs = f"runs {f['first_run']}-{f['first_run'] + f['n_runs'] * rng.randint(3, 9)} ({f['n_runs']} runs)"
        studies = self._studies(f, rng)
        rules = self._rules(f, rng)
        ctx = self._context(f)
        dl = f"approval deadline in {f['deadline_days']} days" if f["deadline_days"] else None
        if surface == "json":
            return dumps(prune({
                "request": {"id": f"{f['policy_version'].split('-')[0]}-REQ-{f['first_run'] % 9973}",
                            "submitted": fmt_date(f["date"], style="iso"), "analysis": f["analysis"],
                            "physics_group": f["group"], "contact": f["contact"], "deadline": dl,
                            "summary": f"Use {sub_name} BAD blocks in {runs} for this analysis only; {self._claim_text(f)}.",
                            "analysis_context": ctx},
                "defect": {"detector": f["detector"], "subsystem": sub_name, "partition": f["partition"],
                           "region": f["region"], "nature": f["nature"]},
                "luminosity": {"analysis_dataset_fb": f["dataset_fb"], "gain_fb": f["gain_fb"]},
                "evidence": {"studies_attached": studies or None,
                             "proposed_systematic_pct": f["syst_proposed_pct"]},
                "policy": {"version": f["policy_version"], "rules": rules},
            }))
        if surface == "request_email":
            body = join_sentences([
                f"On behalf of the {f['analysis']} team ({f['group']} group) I'd like to request a data-quality override.",
                f"We would use the blocks flagged BAD for {f['nature']} in the {sub_name} ({f['partition']}) in {runs}.",
                f"That adds {f['gain_fb']} fb^-1 to our {f['dataset_fb']} fb^-1 dataset.",
                f"Status: {self._claim_text(f)}.",
                ("Context: " + "; ".join(ctx) + ".") if ctx else "",
                (f"We'd propose a {f['syst_proposed_pct']}% systematic to cover it." if f["syst_proposed_pct"] else ""),
                (f"We have an {dl}." if dl else ""),
            ])
            att = ("\n\nAttached:\n" + bullets(studies, rng)) if studies else "\n\nAttached: nothing yet."
            return (f"Subject: DQ override request -- {f['analysis']} ({f['detector']})\nFrom: {f['contact']}\n"
                    f"To: dq-coordination@{f['detector'].lower()}.example.org\nDate: {fmt_date(f['date'], rng)}\n\n"
                    f"Dear {_first(f['dq_coordinator'])},\n\n{noisy(body, rng, 0.12)}{att}\n\nThanks,\n{f['contact']}\n\n"
                    f"--- policy {f['policy_version']} (quoted by the DQ office) ---\n" + "\n".join(rules))
        rows = [("Analysis", f"{f['analysis']} ({f['group']})"), ("Contact", f["contact"]),
                ("Detector / subsystem", f"{f['detector']} / {sub_name}, {f['partition']} ({f['region']})"),
                ("Defect", f["nature"]), ("Runs", runs),
                ("Luminosity gain", f"{f['gain_fb']} fb^-1 on a {f['dataset_fb']} fb^-1 dataset"),
                ("Requester's validation statement", self._claim_text(f)),
                ("Studies attached", "; ".join(studies) if studies else "none"),
                ("Proposed systematic", show(f["syst_proposed_pct"], rng, lambda v: f"{v}%")),
                ("Context", "; ".join(ctx) if ctx else None), ("Deadline", dl)]
        return ("DQ OVERRIDE REVIEW FORM\n" + kv_lines(rows, rng) + f"\n\nPolicy {f['policy_version']}:\n" + "\n".join(rules))

    def questions(self):
        return {
            "override_ruling": choice("How should the override request be decided?", {
                "approve": "Approve it; the blocks enter this analysis' good-run list.",
                "approve_with_systematic": "Approve it only with an added systematic uncertainty covering the defect.",
                "defer_pending_validation": "Defer it until the team delivers the validation the policy requires.",
                "reject": "Reject it; the defect matters too much for this analysis or the gain does not justify the exception.",
            }),
            "approval_route": choice("Who has to sign off on this override under the written routing rule?", {
                "dq_coordinator": "The data-quality coordinator alone; it is a routine, small override.",
                "dq_review_board": "The data-quality review board, because the defect assessment itself is disputed.",
                "physics_coordination": "Physics coordination, because a dataset frozen for a result would change.",
                "physics_group_conveners": "The physics group conveners together with the data-quality coordinator.",
            }),
            "analysis_dependence": score("How strongly does this analysis depend on the defective subsystem in the affected region?", [
                "Not at all: the analysis does not use that system there.",
                "Indirectly: the system enters only through vetoes, isolation or missing-momentum corrections.",
                "Directly: a selected physics object is reconstructed with the affected system.",
                "Critically: the analysis' key signal object relies on the affected region.",
            ]),
            "validation_strength": score("How strong is the validation the team actually attached for the affected blocks?", [
                "None: nothing is attached.",
                "Anecdotal: a few plots without a quantified comparison.",
                "Partial: a quantified comparison, but no closure or bias test.",
                "Thorough: quantified comparisons plus a closure or signal-injection test.",
            ]),
            "luminosity_gain_worthwhile": noul(
                "The luminosity the override would add is large enough to count under the written policy.",
                yes="The gain reaches the policy's minimum share of the dataset (or the lower bar for statistically limited analyses).",
                no="The gain falls below the applicable minimum share of the dataset."),
        }

    def intended(self, f):
        dep = self._dependence(f)
        worth = self._worthwhile(f)
        v = f["validation"]
        if dep == 3:
            ruling = "reject"
        elif dep >= 1 and v < 2:
            ruling = "defer_pending_validation"
        elif dep >= 1 and not worth:
            ruling = "reject"
        elif dep == 2 or (dep == 1 and v == 2):
            ruling = "approve_with_systematic"
        else:
            ruling = "approve"
        if f["frozen"]:
            route = "physics_coordination"
        elif f["expert_dispute"]:
            route = "dq_review_board"
        elif dep <= 1 and f["gain_pct"] < f["route_pct"]:
            route = "dq_coordinator"
        else:
            route = "physics_group_conveners"
        return {"override_ruling": ruling, "approval_route": route, "analysis_dependence": dep,
                "validation_strength": v, "luminosity_gain_worthwhile": "yes" if worth else "no"}


# =========================================================================== schema 5
class OnlineShiftAnomaly(Schema):
    name = "online_shift_anomaly"
    description = ("An anomaly during data taking, logged by the control-room shift: what the written shift procedures "
                   "call for, how much data is being lost and how fast the expert is needed.")
    surfaces = {"json": 0.5, "elog_thread": 0.25, "handover_note": 0.25}

    KINDS = {"partition_dropout": 2.6, "hot_trigger_tower": 1.6, "hv_trip": 2.0, "deadtime_creep": 1.8,
             "cooling_alarm": 1.2, "beam_background": 1.1}

    def sample_factors(self, rng):
        reg = regime(rng)
        det, fac, ecm = pick(rng, COLLIDERS)
        kind = pick_w(rng, self.KINDS)
        f = {"regime": reg, "detector": det, "facility": fac, "ecm_tev": ecm, "kind": kind,
             "run": rng.randint(380000, 529999), "fill": rng.randint(8000, 11999), "time": rand_ts(rng),
             "shift_leader": person(rng), "shifter": person(rng), "policy_version": _policy_id(rng, "SHIFT")}
        f.update({"resync_limit": pick(rng, [2, 3]), "wait_min": pick(rng, [2, 3, 5]),
                  "mask_limit_pct": pick(rng, [5, 8, 10]), "hv_frac_limit": pick(rng, [2, 5, 10]),
                  "dead_limit_pct": pick(rng, [5, 8, 10]), "hw_count": pick(rng, [3, 3, 3, 4]), "hw_window": pick(rng, [10, 20, 30])})
        b = reg == "borderline"
        if kind == "partition_dropout":
            sub = pick(rng, ["ecal", "hcal", "strip", "pixel", "muon_dt", "muon_trig", "fcal"])
            f["subsystem"], f["component"] = sub, _partition(rng, sub)
            f["auto_recovered"] = chance(rng, 0.15)
            f["resyncs_tried"] = 0 if f["auto_recovered"] else (
                f["resync_limit"] - pick(rng, [0, 1]) if b else rng.randint(0, f["resync_limit"]))
            f["blocks_daq"] = chance(rng, 0.35)
            f["minutes_out"] = rng.randint(1, 4) if f["auto_recovered"] else rng.randint(3, 45)
        elif kind == "hot_trigger_tower":
            f["subsystem"], f["component"] = "l1_calo", f"trigger tower ({rng.randint(-28, 28):+d}, {rng.randint(1, 72)})"
            lim = f["mask_limit_pct"]
            f["rate_share_pct"] = (near(rng, lim, 0.25 * lim, lo=0.5, digits=1) if b else
                                   pick(rng, [between(rng, 0.5, 0.7 * lim, digits=1), between(rng, 1.3 * lim, 6 * lim, digits=1)]))
            f["l1_rate_khz"] = between(rng, 60, 110, digits=0)
        elif kind == "hv_trip":
            sub = pick(rng, ["strip", "pixel", "muon_dt", "muon_trig"])
            f["subsystem"], f["component"] = sub, _partition(rng, sub)
            lim = f["hv_frac_limit"]
            f["hv_frac_pct"] = (near(rng, lim, 0.3 * lim, lo=0.1, digits=1) if b else
                                pick(rng, [between(rng, 0.1, 0.6 * lim, digits=1), between(rng, 1.4 * lim, 4 * lim, digits=1)]))
            f["repeat_trip"] = chance(rng, 0.3)
        elif kind == "deadtime_creep":
            f["subsystem"], f["component"] = pick(rng, ["strip", "pixel", "ecal", "muon_dt"]), "central DAQ"
            lim = f["dead_limit_pct"]
            f["deadtime_pct"] = (near(rng, lim, 0.25 * lim, lo=0.5, digits=1) if b else
                                 pick(rng, [between(rng, 0.45 * lim, 0.85 * lim, digits=1), between(rng, 1.3 * lim, 6 * lim, digits=1)]))
            f["minutes_above"] = pick(rng, [4, 7, 12, 18, 35]) if not b else pick(rng, [8, 9, 11, 12])
        elif kind == "cooling_alarm":
            sub = pick(rng, ["pixel", "strip", "ecal", "hcal"])
            f["subsystem"], f["component"] = sub, _partition(rng, sub)
            f["interlock_c"] = pick(rng, [30, 35, 40, 45])
            f["glitch"] = chance(rng, 0.35)
            f["temp_c"] = round(f["interlock_c"] - (between(rng, 6, 12) if f["glitch"] else between(rng, 0.8, 4)), 1)
            f["trend_c_per_10min"] = 0.0 if f["glitch"] else between(rng, 0.3, 1.5)
        else:  # beam_background
            f["subsystem"], f["component"] = "strip", pick(rng, ["forward muon chambers", "beam-condition monitors", "inner tracker"])
            f["stable_beams"] = chance(rng, 0.4)
            f["tracker_hv_on"] = chance(rng, 0.5)
            f["occupancy_x"] = between(rng, 1.5, 9, digits=1)
        # what the crew did: the procedure's step, or a plausible departure from it
        intended = self._response(f)
        follow = chance(rng, 0.25) if reg == "conflicting" else chance(rng, 0.6)
        f["crew_action"] = intended if follow else self._departure(f, intended, rng)
        f["crew_says"] = pick(rng, [None, "no harm done", "all back to normal", "seemed the safest thing"]) if not follow else None
        # earlier e-log mentions of the same component or a neighbour, with their fills
        fills, used = [], []
        n_prev = pick(rng, [1, 2, 3, 4, 5, 5, 6, 7])
        for _ in range(n_prev):
            same = chance(rng, 0.8)
            back = near(rng, f["hw_window"], f["hw_window"] * 0.4, lo=1, integer=True) if chance(rng, 0.5) else rng.randint(1, 2 * f["hw_window"])
            if back in used:
                continue
            used.append(back)
            fills.append({"fill": f["fill"] - back, "same_component": same})
        fills.sort(key=lambda x: -x["fill"])
        f["previous"] = fills
        f["neighbour"] = _partition(rng, f["subsystem"]) if f["subsystem"] != "solenoid" else "main solenoid"
        return f

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _response(f):
        k = f["kind"]
        if k == "partition_dropout":
            if f["auto_recovered"]:
                return "continue_and_log"
            return "resync_partition" if f["resyncs_tried"] < f["resync_limit"] else "stop_restart_run"
        if k == "hot_trigger_tower":
            return "mask_and_continue" if f["rate_share_pct"] > f["mask_limit_pct"] else "continue_and_log"
        if k == "hv_trip":
            return "safe_state_and_call" if (f["repeat_trip"] or f["hv_frac_pct"] > f["hv_frac_limit"]) else "continue_and_log"
        if k == "deadtime_creep":
            return "stop_restart_run" if (f["deadtime_pct"] > f["dead_limit_pct"] and f["minutes_above"] >= 10) else "continue_and_log"
        if k == "cooling_alarm":
            return "continue_and_log" if f["glitch"] else "safe_state_and_call"
        return "safe_state_and_call" if (not f["stable_beams"] and f["tracker_hv_on"]) else "continue_and_log"

    @staticmethod
    def _departure(f, intended, rng):
        alt = {
            "partition_dropout": ["stop_restart_run", "continue_and_log", "resync_partition"],
            "hot_trigger_tower": ["mask_and_continue", "continue_and_log", "stop_restart_run"],
            "hv_trip": ["continue_and_log", "safe_state_and_call", "stop_restart_run"],
            "deadtime_creep": ["stop_restart_run", "continue_and_log"],
            "cooling_alarm": ["continue_and_log", "safe_state_and_call"],
            "beam_background": ["continue_and_log", "safe_state_and_call"],
        }[f["kind"]]
        opts = [a for a in alt if a != intended]
        return pick(rng, opts)

    def _hw_needed(self, f):
        n = 1 + sum(1 for p in f["previous"] if p["same_component"] and f["fill"] - p["fill"] <= f["hw_window"])
        return n >= f["hw_count"]

    def _procedures(self, f, rng):
        k = f["kind"]
        rule = {
            "partition_dropout": _v(
                rng, f"If a partition goes out of sync and does not recover by itself within {f['wait_min']} min, issue a "
                     f"resync; after {f['resync_limit']} failed resyncs, stop and restart the run with the partition included.",
                f"Out-of-sync partition: allow {f['wait_min']} min for self-recovery, then resync; once {f['resync_limit']} "
                f"resyncs have failed, stop the run and restart it with the partition back in."),
            "hot_trigger_tower": _v(
                rng, f"A single trigger tower supplying more than {f['mask_limit_pct']}% of the level-1 rate may be masked "
                     "by the shift leader; below that, leave it and log it for the trigger expert.",
                f"Hot trigger towers: the shift leader masks a tower only if it alone feeds more than "
                f"{f['mask_limit_pct']}% of the level-1 rate; otherwise it is logged for the trigger expert."),
            "hv_trip": _v(
                rng, "One automatic HV recovery per channel per fill is allowed. If a channel trips again, or trips cover "
                     f"more than {f['hv_frac_limit']}% of a sub-detector, put it in standby and call the on-call expert.",
                "HV trips: each channel gets one automatic recovery per fill. A second trip of the same channel, or trips "
                f"covering over {f['hv_frac_limit']}% of the sub-detector, mean standby plus a call to the on-call expert."),
            "deadtime_creep": _v(
                rng, f"If dead time stays above {f['dead_limit_pct']}% for more than 10 minutes, stop and restart the run.",
                f"Dead time above {f['dead_limit_pct']}% that persists beyond 10 minutes: stop and restart the run."),
            "cooling_alarm": _v(
                rng, "On a cooling alarm, put the affected sub-detector's low voltage in standby and call the expert; never "
                     "wait for the hardware interlock. A single-sensor reading contradicted by its neighbours is logged as "
                     "a sensor fault instead.",
                "Cooling alarms: low voltage of the affected sub-detector to standby and call the expert, without waiting "
                "for the interlock -- unless neighbouring sensors contradict a lone reading, which is logged as a sensor fault."),
            "beam_background": _v(
                rng, "Silicon tracker HV stays in standby until stable beams are declared; if it is on outside stable beams, "
                     "ramp it down and call the tracker expert. Background excursions in stable beams are logged for the "
                     "machine coordinator.",
                "Outside stable beams the silicon tracker HV must be in standby; if it is found on, ramp it down and call "
                "the tracker expert. In stable beams, background excursions are only logged for the machine coordinator."),
        }[k]
        return [rule, _v(rng, f"Any anomaly seen {f['hw_count']} or more times on the same component within {f['hw_window']} "
                              f"fills (this occurrence included) requires a hardware intervention request to run coordination.",
                         f"A hardware intervention request goes to run coordination once the same component shows the same "
                         f"anomaly {f['hw_count']} times within {f['hw_window']} fills, counting this one.")]

    def _situation(self, f):
        k, comp = f["kind"], f["component"]
        sub = SUBSYS[f["subsystem"]][0]
        if k == "partition_dropout":
            s = f"{sub} partition {comp} out of sync"
            s += (f", recovered by itself after {f['minutes_out']} min" if f["auto_recovered"]
                  else f", still out after {f['minutes_out']} min; " + {0: "no resync issued yet", 1: "1 resync issued so far"}.get(
                      f["resyncs_tried"], f"{f['resyncs_tried']} resyncs issued so far, none successful"))
            if f["blocks_daq"]:
                s += "; the partition is holding the whole DAQ busy (no events recorded while it is out)"
            return s
        if k == "hot_trigger_tower":
            return (f"{comp} firing hot: {f['rate_share_pct']}% of the level-1 rate "
                    f"(total L1 {f['l1_rate_khz']:.0f} kHz)")
        if k == "hv_trip":
            return (f"HV trip in {sub} {comp}: {f['hv_frac_pct']}% of the sub-detector's channels off"
                    + ("; the same channel tripped earlier this fill and was recovered automatically" if f["repeat_trip"] else "; first trip this fill"))
        if k == "deadtime_creep":
            return (f"dead time {f['deadtime_pct']}% for the last {f['minutes_above']} min, busy coming from the {sub} readout")
        if k == "cooling_alarm":
            return (f"cooling alarm on {sub} {comp}: {f['temp_c']} C (interlock at {f['interlock_c']} C), "
                    + ("neighbouring sensors flat at nominal" if f["glitch"]
                       else f"rising {f['trend_c_per_10min']} C per 10 min, neighbouring sensors also rising"))
        return (f"background in {comp} at {f['occupancy_x']}x the usual occupancy; "
                + ("stable beams declared" if f["stable_beams"] else "beams NOT yet stable (injection/adjust)")
                + ("; silicon tracker HV is ON" if f["tracker_hv_on"] else "; silicon tracker HV in standby"))

    ACTION_TEXT = {
        "continue_and_log": "kept the run going and logged it",
        "resync_partition": "issued a resync of the partition",
        "mask_and_continue": "masked it in the trigger and kept running",
        "stop_restart_run": "stopped and restarted the run",
        "safe_state_and_call": "put the sub-detector in standby/safe state and called the on-call expert",
    }

    def _prev_text(self, f):
        if not f["previous"]:
            return "no earlier e-log entries for this component"
        return "; ".join(f"fill {p['fill']} ({f['component'] if p['same_component'] else f['neighbour']})"
                         for p in f["previous"])

    def render(self, f, rng, surface):
        sit = self._situation(f)
        act = self.ACTION_TEXT[f["crew_action"]]
        procs = self._procedures(f, rng)
        says = f" ({f['crew_says']})" if f["crew_says"] else ""
        if surface == "json":
            return dumps(prune({
                "context": {"detector": f["detector"], "facility": f["facility"], "run": f["run"], "fill": f["fill"],
                            "time_utc": fmt_ts(f["time"], style="iso"), "shift_leader": f["shift_leader"]},
                "anomaly": {"type": f["kind"], "subsystem": SUBSYS[f["subsystem"]][0], "component": f["component"],
                            "observed": sit},
                "shift_response": {"logged_action": act, "crew_remark": f["crew_says"]},
                "history": {"earlier_elog_mentions": self._prev_text(f)},
                "procedures": {"version": f["policy_version"], "text": procs},
            }))
        if surface == "elog_thread":
            t = f["time"]
            entries = [(t, f["shifter"], sit), (t + timedelta(minutes=rng.randint(2, 9)), f["shift_leader"], f"We {act}{says}.")]
            entries.append((t + timedelta(minutes=rng.randint(10, 30)), f["shifter"],
                            "Searched the e-log: " + self._prev_text(f) + "."))
            body = "\n".join(f"#{rng.randint(10000, 99999)} {fmt_ts(x, style='log')} {who}: {noisy(y, rng, 0.08)}" for x, who, y in entries)
            return (f"{f['detector']} e-log, control room -- run {f['run']}, fill {f['fill']}\n{body}\n\n"
                    f"Shift procedures {f['policy_version']} (relevant sections):\n" + bullets(procs, rng))
        lines = [f"SHIFT HANDOVER -- {f['detector']} ({f['facility']}), {fmt_ts(f['time'], rng)}",
                 f"Run {f['run']} / fill {f['fill']}. Shift leader {f['shift_leader']}.", "",
                 "What happened: " + noisy(sit, rng, 0.08) + ".",
                 f"What we did: {act}{says}.",
                 "History: " + self._prev_text(f) + ".", "",
                 f"From the procedures ({f['policy_version']}):"] + [f"  > {p}" for p in procs]
        return "\n".join(lines)

    def questions(self):
        return {
            "procedure_response": choice("What response do the written shift procedures call for at this point?", {
                "continue_and_log": "Keep the run going and log the anomaly for the offline data-quality review.",
                "mask_and_continue": "Mask the offending trigger element and keep running.",
                "resync_partition": "Resynchronise the affected partition without stopping the run.",
                "safe_state_and_call": "Put the affected sub-detector in a safe state and call the on-call expert.",
                "stop_restart_run": "Stop the run and restart it to bring the affected component back.",
            }),
            "data_loss_rate": score("While the anomaly lasts, how much of the physics data being recorded is compromised?", [
                "None: the physics data are unaffected.",
                "A trickle: a small, localised part of the detector or of the rate.",
                "A significant share: a sub-detector region or a noticeable fraction of the rate is lost.",
                "Most of it: the run is recording little usable physics.",
            ]),
            "expert_call_timing": score("How soon does the on-call subsystem expert need to be involved?", [
                "Next working day: an e-log entry is enough.",
                "Before the end of this shift.",
                "Within the hour, while the fill continues.",
                "Immediately: detector hardware or the fill is at stake.",
            ]),
            "crew_followed_procedure": noul(
                "The shift crew's logged response followed the written shift procedures.",
                yes="What the crew did is the step the procedure prescribes for what they saw.",
                no="The crew skipped, reversed or went beyond the step the procedure prescribes."),
            "hardware_request_required": noul(
                "The written rules require a hardware intervention request for this component."),
        }

    def intended(self, f):
        resp = self._response(f)
        k = f["kind"]
        if k == "partition_dropout":
            loss = 3 if f["blocks_daq"] and not f["auto_recovered"] else (1 if f["auto_recovered"] else 2)
            call = 0 if f["auto_recovered"] else (2 if f["resyncs_tried"] >= f["resync_limit"] else 1)
        elif k == "hot_trigger_tower":
            s = f["rate_share_pct"]
            loss = 0 if s <= 0.5 * f["mask_limit_pct"] else 1 if s <= 15 else 2 if s <= 35 else 3
            call = 1 if s > f["mask_limit_pct"] else 0
        elif k == "hv_trip":
            h = f["hv_frac_pct"]
            loss = 1 if h <= f["hv_frac_limit"] else 2 if h <= 3 * f["hv_frac_limit"] else 3
            if f["repeat_trip"] or h > f["hv_frac_limit"]:
                call = 3 if h > 3 * f["hv_frac_limit"] else 2
            else:
                call = 0
        elif k == "deadtime_creep":
            d = f["deadtime_pct"]
            loss = 0 if d < 3 else 1 if d <= 12 else 2 if d <= 35 else 3
            call = 1 if d > f["dead_limit_pct"] else 0
        elif k == "cooling_alarm":
            loss = 0
            call = 1 if f["glitch"] else 3
        else:
            loss = 0 if not f["stable_beams"] else (1 if f["occupancy_x"] < 5 else 2)
            call = 3 if (not f["stable_beams"] and f["tracker_hv_on"]) else 0
        return {"procedure_response": resp, "data_loss_rate": loss, "expert_call_timing": call,
                "crew_followed_procedure": "yes" if f["crew_action"] == resp else "no",
                "hardware_request_required": "yes" if self._hw_needed(f) else "no"}


WORKFLOW = Workflow(
    name="detector_data_quality_flags",
    title="Detector data-quality flags",
    description=("Whether a run or a range of luminosity blocks from a particle or nuclear physics experiment is good, "
                 "bad or partly usable for physics, judged against the experiment's written data-quality criteria."),
    domain="physics / astronomy / research operations",
    schemas=[CollisionRunSignoff(), FixedTargetRunQuality(), NeutrinoRunQuality(), DefectOverrideRequest(),
             OnlineShiftAnomaly()],
    notes=("Fictional detectors and laboratories; the criteria are quoted in every state and vary between cases. "
           "Physical-detector data quality only: no security alerts, no support threads, no bills, no agent runs."),
)
