"""Structural inspection rating: condition states and follow-up actions for existing bridge and building elements,
judged against the asset owner's written inspection-manual criteria.

Every state quotes the manual excerpt that governs it, so each case is decidable in principle; the thresholds vary
by owner and manual edition, so the reader has to apply the quoted numbers rather than remember a standard. The
assets, owners, manuals and people are fictional. These are in-service condition assessments for the asset owner;
municipal plan review, permit inspections and occupancy decisions belong to building_permit_review and are not
generated here.

Each schema exposes `intended(f)`: the answer a careful reviewer would reach from the factors (choice key, "yes" /
"no", or a score level index). It is never shown to the model; it exists so the spread of answers across options
can be checked before labelling.
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
    dumps_pretty,
    fmt_date,
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
    regime,
    score,
    show,
    table,
)

# =========================================================================== shared pools
COUNTY_STEMS = ["Alder", "Basalt", "Cobalt", "Driftwood", "Ember", "Fernhill", "Ironbark", "Juniper", "Lakeshore",
                "Oakridge", "Pinecrest", "Quarry", "Redfern", "Saltmarsh", "Upland", "Valewood", "Westbrook",
                "Yarrow", "Ashgrove", "Bluestem", "Copperline", "Dunmore", "Elmstead", "Harrowgate", "Kingsfold",
                "Larkspur", "Moraine", "Nettlebed", "Orrin", "Rushford", "Stannard", "Wexcombe", "Whitlow",
                "Brackett", "Calloway", "Denholm", "Fairbairn", "Glenmoor", "Hatherley", "Lindell"]
CITY_STEMS = ["Quillhaven", "Pellbridge", "Marrowdale", "Sutter Falls", "Corvina", "Easton Mills", "Halden",
              "Brightwater", "Kessel Point", "Norwood Crossing", "Ostrava Springs", "Tamberlane", "Wrexley",
              "Ambleside Park", "Cinder Bay", "Larchmont Heights", "Fallow Creek", "Ridgemont", "Sorrel Bend"]
ROADS = ["Brenner Road", "Hollis Pike", "Old Quarry Road", "Mill Race Road", "Cedar Hollow Road", "Kessler Avenue",
         "Station Street", "Tolland Road", "Ridgeline Drive", "Ferry Landing Road", "Sawmill Road", "Linden Avenue",
         "Canal Street", "Fairground Road", "Dutch Hollow Road", "Granger Lane", "Ostrander Road", "Wickham Avenue",
         "Black Rock Road", "Haskins Road", "Orchard Valley Road", "Cobb Hill Road", "Turnpike Road", "Pumphouse Road",
         "Lockkeeper Lane", "Gristmill Road", "Tannery Street", "Beacon Hill Road", "Shad Run Road", "Kiln Road"]
WATERS = ["Tanner Creek", "Little Sorrel River", "Holloway Brook", "Ash Run", "Mink Creek", "Bear Wallow Branch",
          "Sugar Kettle Creek", "Pell River", "Otter Slide Creek", "Wolf Pen Run", "Fenwick Drain", "Lost Mill Creek",
          "Spruce Fork", "Halfmoon Creek", "North Branch Sable River", "Cutler Brook", "Stony Lonesome Creek",
          "Cranberry Run", "Weir Creek", "Middle Fork Tamsin River", "Blackwater Slough", "Heron Creek",
          "Gravel Bar Creek", "Painted Post Brook", "Saltlick Run", "Upper Marrow River"]
RAILS = ["a freight rail spur", "an active short-line railroad", "a two-track rail line", "a rail siding to the grain "
         "elevator", "a commuter rail line"]
PATHS = ["a shared-use greenway path", "a pedestrian underpass path", "a canal towpath", "a park trail",
         "a school walking route"]


def _county(rng):
    return f"{pick(rng, COUNTY_STEMS)} County"


def _route(rng):
    r = rng.random()
    if r < 0.35:
        return f"County Route {rng.randint(2, 98)}"
    if r < 0.55:
        return f"State Route {rng.randint(101, 989)}"
    return pick(rng, ROADS)


def _crossing(rng, kind):
    if kind == "water":
        return pick(rng, WATERS)
    if kind == "rail":
        return pick(rng, RAILS)
    if kind == "path":
        return pick(rng, PATHS)
    return pick(rng, [f"County Route {rng.randint(2, 98)}", f"State Route {rng.randint(101, 989)}",
                      pick(rng, ROADS), "an interstate on-ramp", "a two-lane local road"])


def _bridge_id(rng):
    return pick(rng, [f"{rng.randint(1, 88):02d}-{rng.randint(1000, 9999)}", ident(rng, "BR", 5, "-"),
                      f"{rng.randint(1000000, 9999999)}", ident(rng, "STR", 4, "-")])


def _ds(rng):
    return pick(rng, ["iso", "us", "long", "eu"])


def _clamp(x, lo=0, hi=3):
    return max(lo, min(hi, x))


def pick_w_static(f, options):
    """A stable choice keyed on the case's own ids (no rng draw): the same case always reads the same."""
    key = sum(map(ord, str(f.get("bridge_id", "")) + str(f.get("location", ""))))
    return options[key % len(options)]


# =========================================================================== schema 1: concrete deck
class BridgeDeckCondition(Schema):
    name = "bridge_deck_condition"
    description = "A routine inspection of a reinforced concrete bridge deck, rated against the owner's manual criteria."
    surfaces = {"json": 0.5, "inspection_report": 0.3, "team_leader_email": 0.2}

    SOUNDING = {
        "chain_drag_full": "chain drag and hammer sounding over the full deck",
        "chain_drag_one_lane": "chain drag on one lane only (traffic control limited to a single lane closure)",
        "gpr_with_cores": "ground-penetrating radar survey with two verification cores",
        "hammer_spot": "hammer sounding at suspect areas only",
        "visual_only": "visual only; no sounding (no traffic control available on the day)",
    }
    SOFFIT = {
        "sound": "soffit sound, light map cracking",
        "efflorescence": "efflorescence at transverse cracks on the soffit, no spalls",
        "spalls_bars": "soffit spalls with exposed bottom bars, no loose material remaining",
        "loose_concrete": "soffit spall with hollow-sounding, loose concrete around it",
        "sip_forms": "stay-in-place metal forms; soffit concrete not visible",
    }

    def _cs(self, d, crack, staining, rebar, full_depth, soffit, a, b):
        if full_depth or rebar == "exposed_section_loss" or d > b:
            return 4
        if d > a or rebar == "exposed_no_loss" or (crack > 1.0 and staining) or soffit in ("spalls_bars", "loose_concrete"):
            return 3
        if d > 0 or crack > 0.3:
            return 2
        return 1

    def sample_factors(self, rng):
        reg = regime(rng)
        a = pick(rng, [2, 3, 5])
        b = pick(rng, [10, 15, 20])
        under = pick_w(rng, {"water": 3, "road": 3, "rail": 1, "path": 1})
        ws = pick_w(rng, {"bare concrete": 4, "asphalt overlay": 3, "latex-modified concrete overlay": 2,
                          "thin epoxy overlay": 1})
        method = pick_w(rng, {"chain_drag_full": 5, "gpr_with_cores": 2, "chain_drag_one_lane": 2, "hammer_spot": 1,
                              "visual_only": 1})
        crack = round(between(rng, 0.1, 0.8), 1)
        staining = False
        rebar, full_depth, soffit, loose_removed = "none", False, pick_w(rng, {"sound": 4, "efflorescence": 3,
                                                                               "sip_forms": 2}), False
        if reg == "clear":
            if chance(rng, 0.5):
                d = pick(rng, [0.0, 0.0, round(between(rng, 0.1, a * 0.5), 1)])
                crack = round(between(rng, 0.1, 0.5), 1)
                method = pick(rng, ["chain_drag_full", "gpr_with_cores"])
            else:
                d = round(between(rng, b * 1.15, b * 2.0), 1)
                rebar = pick(rng, ["exposed_section_loss", "exposed_no_loss", "exposed_section_loss"])
                full_depth = chance(rng, 0.4)
                soffit = pick(rng, ["spalls_bars", "loose_concrete", "loose_concrete", "efflorescence"])
                crack, staining = round(between(rng, 0.8, 2.0), 1), True
        elif reg == "borderline":
            thr = pick(rng, [a, b])
            d = near(rng, thr, max(0.7, thr * 0.22), lo=0.0, digits=1)
            if chance(rng, 0.45):
                crack = near(rng, 1.0, 0.35, lo=0.1, digits=1)
                staining = crack >= 0.8 and chance(rng, 0.55)
            rebar = pick_w(rng, {"none": 7, "exposed_no_loss": 3}) if thr == a else \
                pick_w(rng, {"none": 5, "exposed_no_loss": 3, "exposed_section_loss": 1})
            soffit = pick_w(rng, {"sound": 3, "efflorescence": 2, "sip_forms": 2, "spalls_bars": 1, "loose_concrete": 3})
            loose_removed = chance(rng, 0.3)
            if soffit == "loose_concrete":
                under = pick_w(rng, {"water": 1, "road": 2, "rail": 1, "path": 1})
        else:  # conflicting: the proposal and the data disagree; the soffit or the survey method muddies it
            d = round(between(rng, 0.4, b * 1.3), 1)
            soffit = pick_w(rng, {"loose_concrete": 6, "sound": 2, "sip_forms": 1, "spalls_bars": 1})
            loose_removed = chance(rng, 0.4)
            if soffit == "loose_concrete":
                under = pick_w(rng, {"water": 1, "road": 2, "rail": 1, "path": 1})
            method = pick_w(rng, {"chain_drag_one_lane": 3, "visual_only": 2, "hammer_spot": 2, "chain_drag_full": 2,
                                  "gpr_with_cores": 1})
            if chance(rng, 0.3):
                ws = "asphalt overlay"
            rebar = pick_w(rng, {"none": 5, "exposed_no_loss": 3, "exposed_section_loss": 1})
            full_depth = chance(rng, 0.16)
        # split the distressed area into its parts; without sounding, delaminations are simply not known
        delam_share = between(rng, 0.45, 0.8)
        spall = round(d * between(rng, 0.08, 0.3), 1)
        patch = round(max(0.0, d - spall - d * delam_share), 1)
        delam = round(max(0.0, d - spall - patch), 1)
        if method == "visual_only":
            delam = None
        coverage = {"chain_drag_full": 100, "gpr_with_cores": 100, "visual_only": 0,
                    "chain_drag_one_lane": pick(rng, [45, 50, 55]), "hammer_spot": pick(rng, [10, 15, 25])}[method]
        d_seen = round(spall + patch + (delam or 0.0), 1)
        # history: a growth rate, and sometimes a weaker survey method last time (inflating the apparent change)
        years = pick_w(rng, {2: 6, 1: 2, 3: 1, 4: 1})
        growth = pick_w(rng, {0.2: 3, 0.8: 3, 2.0: 3, 4.0: 2}) * between(rng, 0.7, 1.3)
        prev_d = round(max(0.0, d_seen - growth * years), 1)
        prev_method_weaker = method in ("chain_drag_full", "gpr_with_cores") and chance(rng, 0.25)
        cs = self._cs(d_seen, crack, staining, rebar, full_depth, soffit, a, b)
        prev_cs = self._cs(prev_d, max(0.1, crack - 0.2), False, "none", False, "sound", a, b)
        if reg == "clear":
            proposed = cs
        elif reg == "borderline":
            proposed = cs if chance(rng, 0.6) else _clamp(cs + pick(rng, [-1, 1]), 1, 4)
        else:
            proposed = _clamp(cs - 1, 1, 4) if chance(rng, 0.75) else _clamp(cs + 1, 1, 4)
        insp = rand_date(rng)
        return {"regime": reg, "owner": _county(rng), "bridge_id": _bridge_id(rng), "carries": _route(rng),
                "under_kind": under, "crosses": _crossing(rng, under), "year_built": rng.randint(1952, 2006),
                "deck_area_m2": between(rng, 260, 4400, integer=True), "adt": pick(rng, [450, 1200, 3800, 9100,
                                                                                         16500, 31000]),
                "truck_pct": between(rng, 3, 19, integer=True), "wearing_surface": ws,
                "cs2_max_pct": a, "cs3_max_pct": b, "manual_rev": pick(rng, ["rev. 3", "rev. 4", "2023 ed.",
                                                                              "2025 ed."]),
                "sounding": method, "area_sounded_pct": coverage, "delam_pct": delam, "spall_pct": spall,
                "patch_pct": patch, "distressed_seen_pct": d_seen, "max_crack_mm": crack, "crack_staining": staining,
                "rebar": rebar, "full_depth_hole": full_depth, "soffit": soffit, "loose_removed": loose_removed,
                "years_since_last": years, "prev_distressed_pct": prev_d, "prev_cs": prev_cs,
                "prev_method_weaker": prev_method_weaker, "proposed_cs": proposed, "team_leader": person(rng),
                "inspection_date": insp, "hours_on_deck": pick(rng, [2, 3, 4, 5, 6])}

    # ------------------------------------------------------------------ answers
    def _critical(self, f):
        return f["full_depth_hole"] or (f["soffit"] == "loose_concrete" and f["under_kind"] != "water"
                                        and not f["loose_removed"])

    def _pace(self, f):
        rate = (f["distressed_seen_pct"] - f["prev_distressed_pct"]) / f["years_since_last"]
        lvl = 0 if rate < 0.5 else 1 if rate < 1.5 else 2 if rate < 3.0 else 3
        cs = self._cs_now(f)
        if cs - f["prev_cs"] >= 2:
            lvl = max(lvl, 2)
        if f["prev_method_weaker"]:
            lvl = max(0, lvl - 1)
        return lvl

    def _cs_now(self, f):
        return self._cs(f["distressed_seen_pct"], f["max_crack_mm"], f["crack_staining"], f["rebar"],
                        f["full_depth_hole"], f["soffit"], f["cs2_max_pct"], f["cs3_max_pct"])

    def intended(self, f):
        cs, pace = self._cs_now(f), self._pace(f)
        crit = self._critical(f)
        if crit:
            work = "emergency_make_safe"
        elif cs == 4:
            work = "priority_repair"
        elif cs == 3:
            work = "priority_repair" if pace >= 3 or (f["rebar"] != "none" and f["truck_pct"] >= 14) else "programmed_repair"
        elif cs == 2:
            work = "preventive_maintenance" if (f["distressed_seen_pct"] >= 0.5 or f["max_crack_mm"] > 0.3 or pace >= 2) \
                else "monitor_only"
        else:
            work = "monitor_only"
        sufficient = (f["sounding"] == "gpr_with_cores" or
                      (f["sounding"] == "chain_drag_full" and f["wearing_surface"] != "asphalt overlay"))
        return {"deck_condition_state": cs - 1, "deterioration_pace": pace, "deck_work_recommendation": work,
                "critical_finding_report": "yes" if crit else "no",
                "field_data_sufficient": "yes" if sufficient else "no"}

    # ------------------------------------------------------------------ rendering
    def _manual(self, f):
        a, b = f["cs2_max_pct"], f["cs3_max_pct"]
        return (f"{f['owner']} Structure Inspection Manual ({f['manual_rev']}), RC deck. Distressed area = "
                f"delaminations + spalls + unsound patches, as % of deck area. CS1: no distressed area, cracks up to "
                f"0.3 mm. CS2: distressed area up to {a}%, no exposed bars, cracks up to 1.0 mm. CS3: distressed area "
                f"over {a}% and up to {b}%, exposed bars without section loss, soffit spalling, or cracks over 1.0 mm "
                f"with rust staining. CS4: distressed area over {b}%, bars with section loss, or any full-depth hole. "
                f"Critical finding (report to the owner within 24 h): a full-depth hole, or loose concrete above a "
                f"road, railway or path that the team did not remove. Rate from the data collected; note limits.")

    def _rebar_text(self, f):
        return {"none": "no exposed reinforcement",
                "exposed_no_loss": "top mat exposed in two spalls, bars rusted but full section",
                "exposed_section_loss": "exposed top bars with visible section loss (necking) in the wheel path"}[f["rebar"]]

    def _soffit_text(self, f):
        t = self.SOFFIT[f["soffit"]]
        if f["soffit"] == "loose_concrete":
            t += (" -- loose pieces hand-scaled by the team, remaining edges sound" if f["loose_removed"]
                  else " -- not removed (no lift access over the traffic below)" if f["under_kind"] != "water"
                  else " -- not removed")
        return t

    def _notes(self, f, rng):
        s = []
        cs, p = self._cs_now(f), f["proposed_cs"]
        if p < cs:
            s.append(pick(rng, [f"Deck looks typical for a {2026 - f['year_built']}-year-old structure; proposing CS{p}.",
                                f"Overall the deck rides fine, I'd keep it at CS{p}.",
                                f"Proposing CS{p}; the distress is scattered rather than concentrated.",
                                f"No change in ride quality since last time, so CS{p}.",
                                f"I'd hold it at CS{p} until the repair contract is let."]))
        elif p > cs:
            s.append(pick(rng, [f"Proposing CS{p} to get it on the repair list.",
                                f"Conservatively proposing CS{p} given the truck traffic."]))
        else:
            s.append(pick(rng, [f"Proposing CS{p}.", f"Rating proposed: CS{p}.", f"My rating: CS{p}."]))
        if f["sounding"] in ("chain_drag_one_lane", "hammer_spot"):
            s.append(pick(rng, ["Could not get a full closure.", "Traffic control was not available for the full deck.",
                                "Only part of the deck could be sounded safely."]))
        elif f["sounding"] == "visual_only":
            s.append(pick(rng, ["No closure available, so no sounding this time.",
                                "Walked the shoulders only; nothing was sounded."]))
        if f["wearing_surface"] == "asphalt overlay" and f["sounding"] in ("chain_drag_full", "chain_drag_one_lane",
                                                                          "hammer_spot"):
            s.append(pick(rng, ["Sounding through the asphalt is muffled in places.",
                                "Chain drag over the overlay was hard to read."]))
        if f["prev_method_weaker"]:
            s.append("Last cycle's numbers came from a visual survey only.")
        if f["soffit"] == "loose_concrete" and f["under_kind"] != "water" and not f["loose_removed"]:
            s.append(f"Loose soffit concrete is over {f['crosses']}.")
        if f["full_depth_hole"]:
            s.append("Daylight visible through one spall near the joint.")
        return noisy(join_sentences(s), rng)

    def render(self, f, rng, surface):
        ds = _ds(rng)
        prev_date = f["inspection_date"] - timedelta(days=365 * f["years_since_last"] + rng.randint(-20, 20))
        notes = self._notes(f, rng)
        delam = None if f["delam_pct"] is None else f["delam_pct"]
        if surface == "json":
            return dumps(prune({
                "structure": {"id": f["bridge_id"], "owner": f["owner"], "carries": f["carries"],
                              "crosses": f["crosses"], "year_built": f["year_built"], "deck_area_m2": f["deck_area_m2"],
                              "adt": f["adt"], "truck_pct": f["truck_pct"], "wearing_surface": f["wearing_surface"]},
                "inspection": {"type": "routine", "date": f["inspection_date"].isoformat(),
                               "team_leader": f["team_leader"], "sounding": self.SOUNDING[f["sounding"]],
                               "area_sounded_pct": f["area_sounded_pct"], "proposed_condition_state": f"CS{f['proposed_cs']}"},
                "findings": {"delaminated_pct": delam if delam is not None else "not sounded",
                             "spalled_pct": f["spall_pct"], "unsound_patches_pct": f["patch_pct"],
                             "max_crack_width_mm": f["max_crack_mm"],
                             "rust_staining_at_cracks": f["crack_staining"], "reinforcement": self._rebar_text(f),
                             "full_depth_hole": f["full_depth_hole"], "soffit": self._soffit_text(f)},
                "previous": {"date": prev_date.isoformat(), "distressed_pct": f["prev_distressed_pct"],
                             "condition_state": f"CS{f['prev_cs']}",
                             "survey": "visual only" if f["prev_method_weaker"] else "same method"},
                "notes": notes,
                "manual_excerpt": self._manual(f),
            }))
        if surface == "inspection_report":
            head = kv_lines([
                ("Structure", f"{f['bridge_id']} -- {f['carries']} over {f['crosses']}"), ("Owner", f["owner"]),
                ("Built", f["year_built"]), ("Deck area", f"{f['deck_area_m2']:,} m2"),
                ("ADT / trucks", f"{f['adt']:,} / {f['truck_pct']}%"), ("Wearing surface", f["wearing_surface"]),
                ("Inspection", f"routine, {fmt_date(f['inspection_date'], rng, ds)}, TL {f['team_leader']}"),
                ("Sounding", f"{self.SOUNDING[f['sounding']]} ({f['area_sounded_pct']}% of deck)"),
            ], rng)
            rows = [["Delaminated (% deck)", pick(rng, ["not sounded", "--", "n/a"]) if delam is None else delam],
                    ["Spalled (% deck)", f["spall_pct"]], ["Unsound patches (% deck)", f["patch_pct"]],
                    ["Max crack width", f"{f['max_crack_mm']} mm" + (" w/ rust staining" if f["crack_staining"] else "")],
                    ["Reinforcement", self._rebar_text(f)],
                    ["Full-depth hole", "YES" if f["full_depth_hole"] else "none"],
                    ["Soffit", self._soffit_text(f)],
                    ["Previous", f"{fmt_date(prev_date, rng, ds)}: {f['prev_distressed_pct']}% distressed, "
                                 f"CS{f['prev_cs']}" + (" (visual survey)" if f["prev_method_weaker"] else "")]]
            return "\n".join([f"ROUTINE INSPECTION -- RC DECK", head, "", table(["Item", "Recorded"], rows, rng), "",
                              f"Proposed rating: CS{f['proposed_cs']}", f"Remarks: {notes}", "",
                              f"Criteria -- {self._manual(f)}"])
        # team_leader_email
        lines = [f"Subject: {f['bridge_id']} deck -- inspection {fmt_date(f['inspection_date'], rng, ds)}",
                 f"From: {f['team_leader']}", f"To: bridge management, {f['owner']}", "",
                 pick(rng, ["Hi,", "Hello,", "Morning,"]), "",
                 f"Finished the routine on {f['carries']} over {f['crosses']} ({f['year_built']}, "
                 f"{f['deck_area_m2']:,} m2, {f['wearing_surface']}, ADT {f['adt']:,}, {f['truck_pct']}% trucks). "
                 f"Survey: {self.SOUNDING[f['sounding']]}.", "",
                 bullets([f"delaminations: " + (pick(rng, ["not sounded", "n/a (no sounding)", "not measured"])
                                                if delam is None else f"{delam}% of deck"),
                          f"spalls: {f['spall_pct']}%, unsound patches: {f['patch_pct']}%",
                          f"widest crack {f['max_crack_mm']} mm" + (", rust staining" if f["crack_staining"] else ""),
                          self._rebar_text(f), "full-depth hole found" if f["full_depth_hole"] else None,
                          f"soffit: {self._soffit_text(f)}",
                          f"last cycle ({f['years_since_last']} yr ago): {f['prev_distressed_pct']}% distressed, "
                          f"CS{f['prev_cs']}"], rng), "", notes, "",
                 f"For reference -- {self._manual(f)}", "", f["team_leader"].split()[0]]
        return "\n".join(lines)

    def questions(self):
        return {
            "deck_condition_state": score("Under the quoted manual, what condition state does the deck warrant?", [
                "Condition state 1 (good): nothing the manual counts as distress.",
                "Condition state 2 (fair): minor distress within the manual's fair limits.",
                "Condition state 3 (poor): distress past the fair limits but short of every severe criterion.",
                "Condition state 4 (severe): at least one of the manual's severe criteria is met.",
            ]),
            "deterioration_pace": score("How quickly is the deck deteriorating since the last inspection?", [
                "Stable: no meaningful change once survey differences are allowed for.",
                "Slow: a small increase in distressed area, in line with normal ageing.",
                "Noticeable: distress is growing faster than normal ageing would explain.",
                "Rapid: distress is spreading fast enough to change the repair plan.",
            ]),
            "deck_work_recommendation": choice("What work should the bridge engineer recommend for the deck?", {
                "emergency_make_safe": "Make the deck or the area below safe immediately, before any planned repair.",
                "monitor_only": "No work now; look at it again at the next routine inspection.",
                "preventive_maintenance": "Seal cracks and the deck surface as routine maintenance to slow the decay.",
                "priority_repair": "Patch or repair within about 90 days, ahead of the normal programme.",
                "programmed_repair": "Put patching or an overlay into next season's planned repair programme.",
            }),
            "critical_finding_report": noul(
                "This inspection produced a critical finding that must be reported to the owner within 24 hours.",
                yes="A full-depth hole was found, or loose concrete remains above a road, railway or path.",
                no="Nothing found meets the manual's critical-finding definition, including loose concrete over water "
                   "or loose concrete the team removed."),
            "field_data_sufficient": noul(
                "The survey data collected is enough to finalise the deck rating without returning to site."),
        }


# =========================================================================== schema 2: steel girder section loss
class SteelGirderSectionLoss(Schema):
    name = "steel_girder_section_loss"
    description = "Corrosion and section loss found on a steel girder bridge: condition state, load-rating review and response."
    surfaces = {"json": 0.5, "inspection_report": 0.25, "engineer_memo": 0.25}

    # (location text, primary member?, critical region?, plate kind, at a support?)
    LOCATIONS = [
        ("web of {g} in the end panel at the {sup} bearing", True, True, "web", True),
        ("bearing stiffener of {g} at {sup}", True, True, "web", True),
        ("bottom flange of {g} near midspan", True, True, "flange", False),
        ("bottom flange of {g} at quarter span", True, False, "flange", False),
        ("top flange of {g} under the deck drain outlet", True, False, "flange", False),
        ("web of {g} at mid-depth under a leaking scupper", True, False, "web", False),
        ("end diaphragm between {g} and the next girder at {sup}", False, False, "light", True),
        ("intermediate cross-frame next to {g}", False, False, "light", False),
        ("lower lateral bracing gusset at {g}", False, False, "light", False),
        ("connection angle of the end diaphragm at {g}, {sup}", False, False, "light", True),
    ]
    THICK = {"web": [9.5, 11.1, 12.7, 14.3], "flange": [15.9, 19.1, 22.2, 25.4, 31.8], "light": [7.9, 9.5, 11.1, 12.7]}
    CAUSES_SUPPORT = ["leaking expansion joint above", "debris and bird waste packed on the bearing seat",
                      "failed joint seal letting deck runoff onto the bearing area", "clogged bearing-seat drainage"]
    CAUSES_SPAN = ["failed deck drain", "trough drain discharging onto the steel", "salt spray from traffic below",
                   "deck cracking letting brine through onto the girder"]
    METHODS = {
        "ut_cleaned": "ultrasonic thickness gauge after cleaning to bare metal",
        "pit_gauge": "pit-depth gauge and calipers after wire brushing",
        "ut_through_coating": "ultrasonic gauge through the coating and scale (not cleaned)",
        "visual_estimate": "visual estimate from photographs",
    }

    def sample_factors(self, rng):
        reg = regime(rng)
        t = pick(rng, [10, 15, 20])
        loc_i = rng.randrange(len(self.LOCATIONS))
        method = pick_w(rng, {"ut_cleaned": 4, "pit_gauge": 2, "ut_through_coating": 2, "visual_estimate": 1})
        readings = pick(rng, [3, 4, 6, 8, 12, 16])
        arms_length = chance(rng, 0.8)
        perforation, crack, pack = False, "none", pick_w(rng, {"none": 6, "present": 3, "distorting": 1})
        rf = round(between(rng, 0.95, 1.9), 2)
        if reg == "clear":
            if chance(rng, 0.5):
                loss = pick(rng, [0.0, 0.0, round(between(rng, 0.5, 3.0), 1)])
                method, readings, arms_length, pack, rf = "ut_cleaned", pick(rng, [8, 12, 16]), True, "none", \
                    round(between(rng, 1.35, 1.9), 2)
            else:
                loc_i = pick(rng, [0, 1, 2])
                loss = round(between(rng, t + 8, t + 25), 1)
                perforation = chance(rng, 0.5)
                rf = round(between(rng, 0.92, 1.15), 2)
                method = pick(rng, ["ut_cleaned", "pit_gauge"])
        elif reg == "borderline":
            loss = near(rng, t, 4.0, lo=0.0, digits=1)
            rf = near(rng, 1.1, 0.2, lo=0.85, digits=2)
            if chance(rng, 0.5):
                loc_i = pick(rng, [0, 1, 2])
            if chance(rng, 0.25):
                crack = "indication_unconfirmed"
        else:  # conflicting: heavy loss on a secondary member, or a crack call on a clean-looking girder
            if chance(rng, 0.5):
                loc_i = rng.randrange(6, len(self.LOCATIONS))
                loss = round(between(rng, t + 5, t + 35), 1)
                perforation = chance(rng, 0.4)
            else:
                loss = round(between(rng, 1.0, t * 0.7), 1)
                crack = pick(rng, ["indication_unconfirmed", "confirmed"])
                loc_i = pick(rng, [0, 2, 3, 5])
            method = pick_w(rng, {"ut_through_coating": 3, "visual_estimate": 2, "ut_cleaned": 2, "pit_gauge": 1})
        loc, primary, critical, plate, at_support = self.LOCATIONS[loc_i]
        nominal = pick(rng, self.THICK[plate])
        if loss == 0:
            corrosion = pick(rng, ["coating_failure", "coating_failure", "none"])
        else:
            corrosion = "active"
        measured = round(nominal * (1 - loss / 100), 1)
        loss = round(100 * (1 - measured / nominal), 1)        # report the loss the rounded reading implies
        gtype = pick(rng, ["rolled beam", "welded plate girder", "riveted plate girder"])
        built = {"rolled beam": rng.randint(1934, 1988), "welded plate girder": rng.randint(1958, 1990),
                 "riveted plate girder": rng.randint(1925, 1961)}[gtype]
        span = between(rng, 12, 27, integer=True) if gtype == "rolled beam" else between(rng, 20, 55, integer=True)
        cause = pick(rng, self.CAUSES_SUPPORT if at_support else self.CAUSES_SPAN)
        return {"regime": reg, "owner": _county(rng), "bridge_id": _bridge_id(rng), "carries": _route(rng),
                "crosses": _crossing(rng, pick(rng, ["water", "water", "road", "rail"])),
                "year_built": built, "girder_type": gtype, "span_m": span, "girders": rng.randint(4, 9),
                "location": loc.format(g=f"G{rng.randint(1, 6)}", sup=pick(rng, ["abutment 1", "abutment 2", "pier 1",
                                                                                 "pier 2"])),
                "loc_index": loc_i, "primary": primary, "critical_region": critical, "loss_trigger_pct": t,
                "nominal_mm": nominal, "measured_min_mm": measured, "loss_pct": loss, "corrosion": corrosion,
                "method": method, "readings": readings, "arms_length": arms_length, "perforation": perforation,
                "crack": crack, "pack_rust": pack, "operating_rf": rf,
                "rf_assumed_loss": pick(rng, ["no section loss", "no section loss", "2% general loss"]),
                "cause": cause,
                "adt": pick(rng, [380, 900, 2600, 7400, 14800, 26000]), "inspector": person(rng),
                "inspection_date": rand_date(rng)}

    def _cs(self, f):
        if f["primary"] and (f["loss_pct"] >= f["loss_trigger_pct"] or f["perforation"] or f["crack"] == "confirmed"
                             or f["pack_rust"] == "distorting"):
            return 4
        if f["loss_pct"] >= 2.0 or (not f["primary"] and (f["loss_pct"] > 0 or f["perforation"])) \
                or f["pack_rust"] != "none":
            return 3
        if f["corrosion"] != "none":
            return 2
        return 1

    def _confidence(self, f):
        lvl = {"ut_cleaned": 3, "pit_gauge": 2, "ut_through_coating": 1, "visual_estimate": 0}[f["method"]]
        if f["readings"] <= 4 and f["method"] in ("ut_cleaned", "pit_gauge"):
            lvl -= 1
        if not f["arms_length"]:
            lvl -= 1
        return _clamp(lvl)

    def _rf_after(self, f):
        k = 1.0 if f["critical_region"] else 0.4
        return f["operating_rf"] * (1 - k * f["loss_pct"] / 100) if f["primary"] else f["operating_rf"]

    def intended(self, f):
        cs, conf = self._cs(f), self._confidence(f)
        rating_review = f["primary"] and (cs == 4 or (f["critical_region"] and f["loss_pct"] >= 2.0
                                                      and f["operating_rf"] < 1.20))
        posting = f["primary"] and ((cs == 4 and f["critical_region"] and self._rf_after(f) < 1.0)
                                    or (f["crack"] == "confirmed" and f["critical_region"]))
        near_trigger = abs(f["loss_pct"] - f["loss_trigger_pct"]) <= 5 and f["primary"]
        if posting or (f["perforation"] and f["critical_region"]):
            resp = "urgent_repair_or_shoring"
        elif conf <= 1 and (near_trigger or f["crack"] == "indication_unconfirmed"):
            resp = "clean_and_regauge"
        elif cs == 4 or (cs == 3 and f["loss_pct"] >= 5):
            resp = "programmed_strengthening"
        else:
            resp = "spot_coat_and_monitor"
        return {"girder_condition_state": cs - 1, "measurement_confidence": conf, "girder_response": resp,
                "load_rating_review": "yes" if rating_review else "no", "posting_needed_now": "yes" if posting else "no"}

    def _manual(self, f):
        t = f["loss_trigger_pct"]
        return (f"{f['owner']} Bridge Inspection Manual, steel members. CS1: coating sound, no corrosion. CS2: "
                f"corrosion or coating failure, no measurable section loss. CS3: measurable section loss under {t}% of "
                f"original thickness; any section loss or perforation in a secondary member; pack rust without "
                f"distortion. CS4: section loss of {t}% or more, a perforation, a crack, or distorting pack rust in a "
                f"primary member. Load rating review: any CS4 in a primary member, and any measurable loss in a "
                f"critical region (bearing web or stiffener, midspan bottom flange) where the operating rating factor "
                f"is below 1.20. Thickness readings for CS decisions shall be taken on cleaned metal.")

    def _cond_text(self, f):
        if f["corrosion"] == "none":
            return "coating intact, no rust"
        if f["corrosion"] == "coating_failure":
            return "coating failed with light surface rust; no measurable loss after brushing"
        return pick_w_static(f, ["active corrosion with scale and pitting", "laminar rust scale over a pitted area",
                                 "active corrosion, scale removed for measurement"])

    def _crack_text(self, f):
        return {"none": None,
                "indication_unconfirmed": "linear indication in the paint at a weld toe nearby; dye penetrant not done",
                "confirmed": "crack at the weld toe confirmed by dye penetrant, about 30 mm long"}[f["crack"]]

    def render(self, f, rng, surface):
        ds = _ds(rng)
        common = {
            "structure": {"id": f["bridge_id"], "owner": f["owner"], "carries": f["carries"], "crosses": f["crosses"],
                          "built": f["year_built"], "superstructure": f"{f['girder_type']}, {f['girders']} girders",
                          "span_m": f["span_m"], "adt": f["adt"]},
            "finding": {"location": f["location"], "condition": self._cond_text(f), "nominal_thickness_mm": f["nominal_mm"],
                        "min_measured_mm": f["measured_min_mm"] if f["loss_pct"] > 0 else None,
                        "section_loss_pct": f["loss_pct"], "perforation": f["perforation"] or None,
                        "pack_rust": None if f["pack_rust"] == "none" else
                        ("between plates, connection visibly bowed" if f["pack_rust"] == "distorting"
                         else "between plates, no distortion"),
                        "crack": self._crack_text(f), "probable_cause": f["cause"]},
            "measurement": {"method": self.METHODS[f["method"]], "readings": f["readings"],
                            "access": "arm's length from a snooper truck" if f["arms_length"]
                            else "from the bank with binoculars and a long-range camera"},
            "load_rating": {"operating_rating_factor": f["operating_rf"], "rating_assumed": f["rf_assumed_loss"]},
        }
        if surface == "json":
            common["inspection"] = {"date": f["inspection_date"].isoformat(), "inspector": f["inspector"]}
            common["manual_excerpt"] = self._manual(f)
            return dumps(prune(common))
        if surface == "inspection_report":
            fd = prune(common["finding"])
            rows = [(k.replace("_", " ").capitalize(), v) for k, v in fd.items()]
            return "\n".join([
                "BRIDGE INSPECTION -- STEEL SUPERSTRUCTURE FINDING",
                kv_lines([("Structure", f"{f['bridge_id']} ({f['carries']} over {f['crosses']})"),
                          ("Owner", f["owner"]), ("Superstructure", common["structure"]["superstructure"]),
                          ("Span", f"{f['span_m']} m"), ("Built", f["year_built"]),
                          ("Date", fmt_date(f["inspection_date"], rng, ds)), ("Inspector", f["inspector"])], rng), "",
                kv_lines(rows, rng), "",
                kv_lines([("Measurement", common["measurement"]["method"]), ("Readings", f["readings"]),
                          ("Access", common["measurement"]["access"]),
                          ("Operating RF (current rating)", f"{f['operating_rf']} (assumed {f['rf_assumed_loss']})")],
                         rng), "", "Manual: " + self._manual(f)])
        # engineer_memo
        para = [f"{f['bridge_id']}, {f['carries']} over {f['crosses']} ({f['girder_type']}, built {f['year_built']}).",
                f"During the {fmt_date(f['inspection_date'], rng, ds)} inspection {f['inspector']} found corrosion "
                f"on the {f['location']}, probably from the {f['cause']}.",
                (f"Minimum thickness {f['measured_min_mm']} mm against {f['nominal_mm']} mm nominal "
                 f"({f['loss_pct']}% loss)" if f["loss_pct"] > 0 else "No measurable thickness loss")
                + f", by {self.METHODS[f['method']]}, {f['readings']} readings, "
                + ("at arm's length." if f["arms_length"] else "viewed from the bank."),
                "The web is perforated in one spot." if f["perforation"] else "",
                ("Pack rust is bowing the connection plates." if f["pack_rust"] == "distorting" else
                 "Some pack rust between plates." if f["pack_rust"] == "present" else ""),
                (self._crack_text(f).capitalize() + ".") if f["crack"] != "none" else "",
                f"The current rating gives an operating factor of {f['operating_rf']}, assuming {f['rf_assumed_loss']}.",
                pick(rng, ["Inspector's view: not urgent.", "Inspector would like a decision before winter.",
                           "Inspector flagged it for the engineer.", ""])]
        return ("MEMO -- steel girder finding for review\n\n" + noisy(join_sentences(para), rng, 0.12)
                + "\n\nCriteria: " + self._manual(f))

    def questions(self):
        return {
            "girder_condition_state": score("What condition state does the quoted manual assign to this member?", [
                "Condition state 1: coating sound, no corrosion.",
                "Condition state 2: corrosion without measurable section loss.",
                "Condition state 3: section loss or pack rust short of the severe criteria.",
                "Condition state 4: a severe criterion is met in a primary member.",
            ]),
            "measurement_confidence": score("How far can the reported section loss be relied on?", [
                "Not at all: an estimate without measurement.",
                "Weakly: measured, but through coating or rust, or from a distance.",
                "Reasonably: measured on prepared metal with limited coverage.",
                "Fully: many readings on cleaned metal at arm's length.",
            ]),
            "girder_response": choice("What should the owner do about this finding now?", {
                "clean_and_regauge": "Blast-clean the area and re-measure before deciding, because the numbers are not reliable enough.",
                "programmed_strengthening": "Schedule a bolted plate repair or replacement in the planned programme.",
                "spot_coat_and_monitor": "Spot-paint, fix the drainage source and look again at the next routine inspection.",
                "urgent_repair_or_shoring": "Shore or repair within weeks, with interim protection until then.",
            }),
            "load_rating_review": noul(
                "The manual requires the bridge's load rating to be reviewed because of this finding.",
                yes="The finding is a severe condition in a primary member, or measurable loss in a critical region "
                    "of a bridge rated below the stated factor.",
                no="The loss is in a secondary member, or is light and away from critical regions on an adequately "
                   "rated bridge."),
            "posting_needed_now": noul("The bridge needs a reduced weight limit posted before any repair is made."),
        }


# =========================================================================== schema 3: culvert
class CulvertConditionRating(Schema):
    name = "culvert_condition_rating"
    description = "A road culvert's barrel, joints and roadway above, rated against the owner's culvert criteria."
    surfaces = {"json": 0.5, "inspection_form": 0.25, "field_notes": 0.25}

    TYPES = {"csp": ("corrugated steel pipe", True), "csp_arch": ("corrugated steel pipe-arch", True),
             "plate_arch": ("structural plate steel arch", True), "cap": ("corrugated aluminium pipe", True),
             "rcp": ("reinforced concrete pipe", False), "box": ("precast concrete box", False),
             "cip_box": ("cast-in-place concrete box", False), "masonry": ("stone masonry arch", False)}

    SPANS = {"csp": (0.6, 3.6), "csp_arch": (1.2, 3.6), "plate_arch": (3.0, 6.0), "cap": (0.6, 2.4), "rcp": (0.6, 2.7),
             "box": (1.2, 3.6), "cip_box": (1.8, 4.2), "masonry": (1.0, 3.0)}

    def sample_factors(self, rng):
        reg = regime(rng)
        typ = pick(rng, list(self.TYPES))
        flexible = self.TYPES[typ][1]
        x, y = pick(rng, [(3, 8), (5, 10), (4, 8)])
        lo, hi = self.SPANS[typ]
        span = round(pick_w(rng, {between(rng, lo, min(hi, 1.15)): 4 if lo < 1.15 else 0, between(rng, max(lo, 1.2), hi): 6}), 1)
        fill = round(pick(rng, [between(rng, 0.4, 1.0), between(rng, 1.0, 3.0), between(rng, 3.0, 7.0)]), 1)
        barrel = "sound"
        shape = round(between(rng, 0, x * 0.7), 1) if flexible else None
        joints = pick_w(rng, {"tight": 6, "minor_separation": 3, "open_soil_loss": 1})
        road = pick_w(rng, {"none": 7, "cracking": 2, "dip": 1})
        dip_mm = None
        blockage = pick(rng, [0, 5, 10, 20, 35, 55])
        if reg == "clear":
            if chance(rng, 0.5):
                barrel, joints, road = pick(rng, ["sound", "minor"]), "tight", "none"
                shape = round(between(rng, 0, x * 0.5), 1) if flexible else None
                blockage = pick(rng, [0, 5, 10])
            else:
                barrel = "severe"
                joints = pick(rng, ["open_soil_loss", "minor_separation"])
                road = pick(rng, ["dip", "sinkhole"])
                shape = round(between(rng, y + 1, y + 6), 1) if flexible else None
        elif reg == "borderline":
            if flexible:
                shape = near(rng, pick_w(rng, {x: 2, y: 1}), 1.5, lo=0.0, digits=1)
            barrel = pick_w(rng, {"sound": 1, "minor": 5, "moderate": 3, "severe": 1})
            joints = pick_w(rng, {"tight": 5, "minor_separation": 3, "open_soil_loss": 1})
            road = pick_w(rng, {"none": 6, "cracking": 2, "dip": 1})
        else:  # conflicting: a tidy barrel with a dip in the road, or a bad barrel under deep fill and no surface sign
            twist = pick_w(rng, {"looks_bad": 4, "road_only": 2, "deep_bad_barrel": 2, "joint_loss": 2})
            if twist == "looks_bad":      # silted, overtopped and rusty, but structurally only fair
                barrel, joints, road = "minor", "tight", "none"
                blockage = pick(rng, [35, 45, 55, 65])
                if flexible:
                    shape = round(between(rng, 0.5, x * 0.9), 1)
            elif twist == "road_only":      # a tidy barrel, but the pavement over the line has cracked
                barrel, joints, road = pick(rng, ["minor", "sound"]), pick(rng, ["tight", "minor_separation"]), "cracking"
                if flexible:
                    shape = round(between(rng, 0.5, x * 0.8), 1)
            elif twist == "deep_bad_barrel":   # a bad barrel under deep fill with no surface sign
                barrel, road, fill = "severe", "none", round(between(rng, 3.5, 7.5), 1)
                joints = pick(rng, ["tight", "minor_separation"])
            else:                          # a tidy barrel with soil coming in at a joint
                barrel, joints, road = pick(rng, ["minor", "sound"]), "open_soil_loss", pick(rng, ["none", "cracking"])
            if flexible and shape is None:
                shape = round(between(rng, 0, y), 1)
        if road == "dip":
            dip_mm = pick(rng, [15, 25, 40, 60, 90])
        walkable = span >= 1.2
        viewed = "full length" if walkable else pick(rng, ["from both ends only", "from both ends only",
                                                           "pole camera from the inlet"])
        if not walkable and chance(rng, 0.25):
            viewed = "CCTV crawler, full length"
        overtopping = blockage >= 35 and chance(rng, 0.5)
        f = {"regime": reg, "owner": _county(rng), "culvert_id": ident(rng, pick(rng, ["CUL", "C", "STR"]), 5, "-"),
                "road": _route(rng), "stream": pick(rng, WATERS + ["an unnamed tributary", "a roadside ditch outfall"]),
                "type": typ, "flexible": flexible, "span_m": span, "length_m": between(rng, 12, 64, integer=True),
                "fill_m": fill, "aadt": pick(rng, [220, 650, 1900, 5200, 11800, 24000]), "year_installed":
                rng.randint(1948, 2004), "cs2_shape_pct": x, "cs3_shape_pct": y, "shape_change_pct": shape,
                "barrel": barrel, "joints": joints, "joint_gap_mm": pick(rng, [10, 20, 30, 45, 70]),
                "road_surface": road, "dip_mm": dip_mm, "blockage_pct": blockage, "overtopping_marks": overtopping,
                "outlet_scour": pick_w(rng, {"none": 5, "minor": 3, "undermining_end": 1}),
                "viewed": viewed, "walkable": walkable, "inspector": person(rng), "inspection_date": rand_date(rng)}
        f["prev_cs"] = max(1, self._cs(f) - pick(rng, [0, 0, 1, 1, 2]))
        return f

    def _barrel_text(self, f):
        if f["type"] == "masonry":
            return {"sound": "masonry sound, mortar joints full", "minor": "shallow mortar loss in the arch ring "
                    "joints", "moderate": "deep mortar loss and two displaced stones in the arch ring, no fill visible",
                    "severe": "stones missing from the invert and lower arch ring, fill washing out"}[f["barrel"]]
        if f["flexible"]:
            return {"sound": "galvanising intact, no rust through", "minor": "surface rust and light pitting along "
                    "the invert", "moderate": "heavy pitting with a few small perforations in the invert, backfill "
                    "not visible through them", "severe": "invert rusted through over about a third of the length, "
                    "backfill washing in through the holes"}[f["barrel"]]
        return {"sound": "concrete sound, hairline cracks only", "minor": "hairline cracking with light "
                "efflorescence", "moderate": "spalls with exposed bars in the top slab and invert, no loose material",
                "severe": "invert worn through to the bottom bars, bars broken in two places"}[f["barrel"]]

    def _joint_text(self, f):
        return {"tight": "joints tight", "minor_separation": f"joint separation up to {f['joint_gap_mm']} mm, no soil "
                "visible", "open_soil_loss": f"joint open {f['joint_gap_mm']} mm with fines and soil cones inside the "
                "barrel"}[f["joints"]]

    def _road_text(self, f):
        if f["road_surface"] == "dip":
            return f"{f['dip_mm']} mm dip in the travel lane over the culvert line"
        return {"none": "no pavement distress over the line", "cracking": "transverse pavement cracking over the "
                "culvert line, no dip", "sinkhole": "open sinkhole at the shoulder over the barrel"}[f["road_surface"]]

    def _cs(self, f):
        sh = f["shape_change_pct"]
        if (f["barrel"] == "severe" or f["joints"] == "open_soil_loss" or f["road_surface"] in ("dip", "sinkhole")
                or (sh is not None and sh > f["cs3_shape_pct"])):
            return 4
        if f["barrel"] == "moderate" or f["joints"] == "minor_separation" or (sh is not None and sh > f["cs2_shape_pct"]):
            return 3
        if f["barrel"] == "minor" or f["road_surface"] == "cracking" or (sh is not None and sh > 1.0):
            return 2
        return 1

    def _settlement(self, f):
        r = 0
        if f["joints"] == "open_soil_loss" or (f["barrel"] == "severe"):
            r = 2
        elif f["joints"] == "minor_separation" or f["barrel"] == "moderate":
            r = 1
        if f["road_surface"] == "sinkhole" or (f["road_surface"] == "dip" and f["dip_mm"] >= 40):
            r = 3
        elif f["road_surface"] == "dip":
            r = max(r, 2)
        elif f["road_surface"] == "cracking":
            r = max(r, 1)
        if r in (1, 2) and f["fill_m"] < 1.0:
            r += 1
        if r >= 2 and f["fill_m"] >= 4.0 and f["road_surface"] == "none":
            r -= 1
        return _clamp(r)

    def intended(self, f):
        cs, risk = self._cs(f), self._settlement(f)
        sh = f["shape_change_pct"]
        if f["road_surface"] == "sinkhole" or (f["road_surface"] == "dip" and f["dip_mm"] >= 40) or risk == 3:
            act = "restrict_road_over_culvert"
        elif cs == 4 and (f["barrel"] == "severe" or (sh is not None and sh > f["cs3_shape_pct"])):
            act = "programme_replacement"
        elif f["joints"] in ("open_soil_loss", "minor_separation") and f["barrel"] in ("sound", "minor"):
            act = "seal_joints_and_fill_voids"
        elif f["barrel"] in ("moderate", "severe"):
            act = "line_or_pave_invert"
        else:
            act = "clean_and_monitor"
        soil_loss = f["joints"] == "open_soil_loss" or f["barrel"] == "severe" or f["road_surface"] in ("dip", "sinkhole")
        partial_view = f["viewed"] in ("from both ends only", "pole camera from the inlet")
        if soil_loss and f["road_surface"] != "sinkhole":
            follow = "void_survey_over_barrel"
        elif partial_view:
            follow = "cctv_survey"
        elif f["blockage_pct"] >= 35 or f["overtopping_marks"]:
            follow = "hydraulic_review"
        else:
            follow = "routine_cycle_only"
        shortened = cs >= 3 or f["road_surface"] != "none"
        return {"culvert_condition_state": cs - 1, "roadway_settlement_risk": risk, "culvert_action": act,
                "follow_up_investigation": follow, "interval_shortened": "yes" if shortened else "no"}

    def _manual(self, f):
        x, y = f["cs2_shape_pct"], f["cs3_shape_pct"]
        return (f"{f['owner']} Culvert Inspection Guide. CS1: barrel sound, joints tight. CS2: surface corrosion, "
                f"pitting or hairline cracks; metal shape within {x}% of design; pavement cracking without a dip. "
                f"CS3: perforations or spalls without loss of backfill, joints open without soil entering, metal "
                f"shape change {x}-{y}%. CS4: backfill loss through holes or joints, invert worn through, shape change "
                f"over {y}%, or a pavement dip or sinkhole attributable to the culvert. Interval: 48 months at CS1-CS2 "
                f"with no roadway distress; 12 months otherwise.")

    def render(self, f, rng, surface):
        ds = _ds(rng)
        tname = self.TYPES[f["type"]][0]
        shape = None if f["shape_change_pct"] is None else f"{f['shape_change_pct']}% (rise/span vs design)"
        if surface == "json":
            return dumps(prune({
                "culvert": {"id": f["culvert_id"], "owner": f["owner"], "road": f["road"], "stream": f["stream"],
                            "type": tname, "span_m": f["span_m"], "length_m": f["length_m"], "fill_over_top_m": f["fill_m"],
                            "installed": f["year_installed"], "aadt": f["aadt"]},
                "inspection": {"date": f["inspection_date"].isoformat(), "inspector": f["inspector"],
                               "barrel_viewed": f["viewed"], "previous_condition_state": f"CS{f['prev_cs']}"},
                "findings": {"barrel": self._barrel_text(f), "shape_change": shape, "joints": self._joint_text(f),
                             "roadway_above": self._road_text(f), "waterway_blocked_pct": f["blockage_pct"],
                             "overtopping_debris_line_on_road": f["overtopping_marks"] or None,
                             "outlet": {"none": "outlet apron intact", "minor": "minor scour below the outlet apron",
                                        "undermining_end": "end section undermined by about 0.4 m"}[f["outlet_scour"]]},
                "manual_excerpt": self._manual(f),
            }))
        if surface == "inspection_form":
            return "\n".join([
                f"CULVERT INSPECTION -- {f['owner'].upper()}",
                kv_lines([("Culvert", f["culvert_id"]), ("Road / stream", f"{f['road']} / {f['stream']}"),
                          ("Type", f"{tname}, {f['span_m']} m span, {f['length_m']} m long"),
                          ("Cover", f"{f['fill_m']} m"), ("AADT", f"{f['aadt']:,}"),
                          ("Date", fmt_date(f["inspection_date"], rng, ds)), ("Inspector", f["inspector"]),
                          ("Viewed", f["viewed"]), ("Previous CS", f["prev_cs"])], rng), "",
                table(["Element", "Observation"], [["Barrel", self._barrel_text(f)],
                                                  ["Shape", show(shape, rng) if f["flexible"] else "n/a (rigid)"],
                                                  ["Joints", self._joint_text(f)],
                                                  ["Roadway", self._road_text(f)],
                                                  ["Waterway", f"{f['blockage_pct']}% blocked by sediment/debris"
                                                   + ("; debris line on the road shoulder" if f["overtopping_marks"] else "")]],
                      rng), "", "Guide criteria: " + self._manual(f)])
        obs = [f"Culvert {f['culvert_id']} under {f['road']} ({tname}, {f['span_m']} m, {f['fill_m']} m cover).",
               f"Viewed {f['viewed']}.", self._barrel_text(f).capitalize() + ".",
               f"Shape change {f['shape_change_pct']}%." if f["shape_change_pct"] is not None else "",
               self._joint_text(f).capitalize() + ".", self._road_text(f).capitalize() + ".",
               f"About {f['blockage_pct']}% of the opening silted or blocked." if f["blockage_pct"] else "",
               "Debris line on the road shoulder from the last storm." if f["overtopping_marks"] else "",
               pick(rng, [f"Was CS{f['prev_cs']} last time.", f"Last cycle: CS{f['prev_cs']}.", ""])]
        return (f"Field notes -- {f['inspector']}, {fmt_date(f['inspection_date'], rng, ds)}\n"
                + noisy(join_sentences(obs), rng, 0.15) + "\n\nRef: " + self._manual(f))

    def questions(self):
        return {
            "culvert_condition_state": score("What condition state does the culvert warrant under the quoted guide?", [
                "CS1: barrel sound and joints tight.",
                "CS2: minor surface deterioration only.",
                "CS3: deterioration that has not yet let backfill escape.",
                "CS4: backfill is being lost, or the barrel or road above shows failure.",
            ]),
            "roadway_settlement_risk": score("How likely is the road over the culvert to settle or collapse before a repair?", [
                "Remote: nothing suggests soil is moving.",
                "Low: openings exist, but no sign that soil is escaping.",
                "Material: soil is escaping or the pavement has started to move.",
                "High: a dip or hole shows the fill is already failing under traffic.",
            ]),
            "culvert_action": choice("What maintenance or repair action fits these findings?", {
                "clean_and_monitor": "Clear sediment and debris and keep it on the normal inspection cycle.",
                "line_or_pave_invert": "Line the barrel or pave the invert to restore the deteriorated section.",
                "programme_replacement": "Plan replacement of the culvert; repair would not restore it.",
                "restrict_road_over_culvert": "Close or narrow the road over the culvert now, until it is made safe.",
                "seal_joints_and_fill_voids": "Seal the joints and grout any voids behind them.",
            }),
            "follow_up_investigation": choice("What further investigation should be ordered before the next decision?", {
                "cctv_survey": "A camera survey of the parts of the barrel nobody has seen.",
                "hydraulic_review": "A hydraulic check of whether the opening still passes design flows.",
                "routine_cycle_only": "Nothing beyond the normal inspection cycle.",
                "void_survey_over_barrel": "Probe or radar the fill over the barrel to find voids.",
            }),
            "interval_shortened": noul(
                "The guide requires this culvert to go on the shortened 12-month inspection interval.",
                yes="The culvert rates CS3 or worse, or there is any distress in the road above it.",
                no="It rates CS1 or CS2 and the road above shows no distress."),
        }


# =========================================================================== schema 4: parking structure
class ParkingStructureCondition(Schema):
    name = "parking_structure_condition"
    description = "A periodic condition assessment of a parking structure, classified under the city's written rule."
    surfaces = {"json": 0.5, "assessment_report": 0.3, "engineer_letter": 0.2}

    FINDINGS = {
        "soffit_spall": ("a slab soffit spall with hollow-sounding concrete around it, {area} m2, over {where}",
                         ["drive aisle", "parking stalls", "the pedestrian route to the stair", "a closed storage bay"]),
        "pt_tendon": ("an exposed post-tensioning tendon at the slab edge, {wires} of 7 wires broken, over {where}",
                      ["drive aisle", "parking stalls", "the pedestrian route to the stair", "a closed storage bay"]),
        "tee_connector": ("broken flange-to-flange weld connectors between double tees ({n} of {m}), {mm} mm "
                          "differential deflection, at {where}", ["drive aisle", "parking stalls", "the ramp"]),
        "column_base": ("a column base spall with exposed vertical bars, about {loss}% bar section loss, at {where}",
                        ["drive aisle", "parking stalls", "the pedestrian route to the stair"]),
        "barrier_cable": ("{cables} slack with corroded anchorages, at a {drop} m edge drop by {where}",
                          ["parking stalls", "the ramp", "the roof-level perimeter"]),
    }

    def sample_factors(self, rng):
        reg = regime(rng)
        system = pick(rng, ["cast-in-place post-tensioned slabs", "precast double tees with topping",
                            "conventionally reinforced flat slab"])
        kinds = ["soffit_spall", "column_base", "barrier_cable"] + (["pt_tendon"] if "post-tensioned" in system else []) \
            + (["tee_connector"] if "double tees" in system else [])
        kind = pick(rng, kinds)
        where = pick(rng, self.FINDINGS[kind][1])
        severity = pick_w(rng, {"low": 3, "mid": 4, "high": 3})
        made_safe = pick_w(rng, {"none": 4, "loose_removed": 3, "cordoned": 2, "shored": 1})
        hc = pick(rng, [None, 4, 12, 22, 38, 55])
        cl_ratio = round(between(rng, 0.2, 2.6), 2)
        if reg == "clear":
            if chance(rng, 0.5):
                severity, made_safe, hc, cl_ratio = "low", "none", pick(rng, [2, 5, 8]), round(between(rng, 0.2, 0.6), 2)
            else:
                severity, made_safe = "high", pick(rng, ["none", "none", "loose_removed"])
                structural = [k for k in kinds if k in ("pt_tendon", "tee_connector", "column_base")]
                if structural and chance(rng, 0.6):
                    kind = pick(rng, structural)
                    where = pick(rng, [w for w in self.FINDINGS[kind][1] if w != "a closed storage bay"])
                if where == "a closed storage bay":
                    where = "drive aisle"
        elif reg == "borderline":
            severity = pick(rng, ["mid", "mid", "low", "high"])
            cl_ratio = near(rng, 1.0, 0.35, lo=0.1)
        else:  # conflicting: a frightening finding already made safe, or a mild-looking one in a busy spot
            if chance(rng, 0.5):
                severity, made_safe = "high", pick(rng, ["loose_removed", "cordoned", "shored"])
            else:
                severity, made_safe = "mid", "none"
                where = pick(rng, [w for w in self.FINDINGS[kind][1] if w != "a closed storage bay"])
            if hc is None:
                hc = pick(rng, [8, 45])
        # numbers for the finding text
        params = {"area": {"low": 0.2, "mid": 0.8, "high": 2.4}[severity] * between(rng, 0.7, 1.4),
                  "wires": {"low": 1, "mid": pick(rng, [2, 3]), "high": 7}[severity],
                  "n": {"low": 1, "mid": pick(rng, [2, 3]), "high": pick(rng, [4, 5, 6])}[severity],
                  "m": pick(rng, [8, 10, 12]), "mm": {"low": 4, "mid": 9, "high": 22}[severity],
                  "loss": {"low": 5, "mid": 15, "high": 35}[severity], "drop": pick(rng, [3.0, 3.2, 6.5, 9.6])}
        params["area"] = round(params["area"], 1)
        params["cables"] = "one vehicle-barrier cable" if params["n"] == 1 else f"{params['n']} vehicle-barrier cables"
        if kind != "soffit_spall" and made_safe == "loose_removed" and kind not in ("pt_tendon", "column_base"):
            made_safe = "none"
        return {"regime": reg, "city": pick(rng, CITY_STEMS), "garage": f"{pick(rng, ROADS).split()[0]} Street Garage",
                "system": system, "levels": rng.randint(3, 8), "spaces": pick(rng, [180, 320, 540, 780, 1100, 1400]),
                "use": pick_w(rng, {"public, 24 h": 4, "employee, weekdays": 3, "residential tenants": 2}),
                "built": rng.randint(1971, 2009), "finding_kind": kind, "where": where, "severity": severity,
                "finding_params": params, "made_safe": made_safe, "top_deck_delam_pct": round(between(rng, 0.5, 14), 1),
                "halfcell_pct_below_350mv": hc, "chloride_to_threshold": cl_ratio, "rust_staining":
                pick_w(rng, {"none": 3, "some": 4, "widespread": 2}), "prev_class": pick(rng, ["serviceable",
                "serviceable, repair in cycle", "repair within 90 days"]), "engineer": person(rng),
                "firm": f"{pick(rng, COUNTY_STEMS)} {pick(rng, ['Structural', 'Engineering', 'Restoration Engineers', 'Consulting'])}",
                "assessment_date": rand_date(rng), "cycle_years": pick(rng, [3, 5])}

    def _imminent(self, f):
        k, s, w = f["finding_kind"], f["severity"], f["where"]
        occupied = w != "a closed storage bay"
        if k == "barrier_cable":
            return s != "low" and f["finding_params"]["drop"] >= 3.0
        if k == "pt_tendon":
            return s == "high" or (s == "mid" and occupied)
        if k == "soffit_spall":
            return occupied and s != "low"
        if k == "tee_connector":
            return s == "high"
        return s == "high"  # column base

    def _underlying(self, f):
        """Classification index of the deterioration itself once any imminent hazard is removed (0..2)."""
        s = f["severity"]
        base = {"low": 0, "mid": 1, "high": 2}[s]
        if base == 0 and (f["chloride_to_threshold"] >= 1.0 or f["top_deck_delam_pct"] >= 8):
            base = 1
        if f["finding_kind"] in ("pt_tendon", "barrier_cable") and base == 0:
            base = 1
        return base

    def _corrosion(self, f):
        hc, cl = f["halfcell_pct_below_350mv"], f["chloride_to_threshold"]
        lvl = 0 if cl < 0.6 else 1 if cl < 1.0 else 2 if cl < 1.8 else 3
        if hc is not None:
            lvl2 = 0 if hc < 10 else 1 if hc < 25 else 2 if hc < 45 else 3
            lvl = round((lvl + lvl2) / 2 + 0.01)
        if f["rust_staining"] == "widespread":
            lvl += 1
        return _clamp(lvl)

    def _exposure(self, f):
        w = f["where"]
        lvl = {"drive aisle": 2, "parking stalls": 2, "the pedestrian route to the stair": 3, "a closed storage bay": 0,
               "the ramp": 2, "the roof-level perimeter": 2}[w]
        if f["use"] == "public, 24 h":
            lvl += 1
        elif f["use"] == "employee, weekdays" and lvl > 0:
            lvl -= 0
        if f["made_safe"] in ("cordoned", "shored"):
            lvl -= 2
        elif f["made_safe"] == "loose_removed":
            lvl -= 1
        if f["severity"] == "low":
            lvl -= 1
        return _clamp(lvl)

    def intended(self, f):
        imm = self._imminent(f)
        hazard_cleared = f["made_safe"] in ("loose_removed", "shored") and f["finding_kind"] in ("soffit_spall",
                                                                                              "column_base", "pt_tendon")
        under = self._underlying(f)
        if imm and not hazard_cleared and f["made_safe"] != "cordoned":
            cls = "unsafe"
        elif imm:
            cls = "repair_within_90_days"
        else:
            cls = ["serviceable", "serviceable_repair_in_cycle", "repair_within_90_days"][under]
        if imm and f["made_safe"] == "none":
            prot = "close_level_or_structure" if (f["finding_kind"] in ("tee_connector", "column_base", "pt_tendon")
                                                  and f["severity"] == "high") else "cordon_and_shore_area"
        elif imm and f["made_safe"] == "loose_removed":
            prot = "remove_loose_and_monitor"
        elif imm:
            prot = "cordon_and_shore_area"
        elif f["finding_kind"] == "soffit_spall" or f["severity"] == "mid":
            prot = "remove_loose_and_monitor"
        else:
            prot = "none_required"
        return {"assessment_classification": cls, "immediate_protection": prot, "corrosion_activity": self._corrosion(f),
                "public_exposure": self._exposure(f), "department_notification": "yes" if imm else "no"}

    def _finding_text(self, f):
        tmpl = self.FINDINGS[f["finding_kind"]][0]
        return tmpl.format(where=f["where"], **f["finding_params"])

    def _made_safe_text(self, f):
        return {"none": "no protective measures taken yet",
                "loose_removed": "loose concrete hand-scaled during the assessment; remaining surface sounded solid",
                "cordoned": "area taped off and cones placed by the garage operator the same day",
                "shored": "temporary shoring posts installed under the area the next morning"}[f["made_safe"]]

    def _rule(self, f):
        return (f"City of {f['city']} parking structure rule: classify as Serviceable; Serviceable - repair within "
                f"cycle (could become unsafe if not repaired before the next {f['cycle_years']}-year assessment); "
                f"Repair within 90 days (could become unsafe within a year; protect affected areas now); or Unsafe "
                f"(imminent threat to people; protect immediately). Notify the Department within 24 h of finding "
                f"any imminent threat, even if it is made safe the same day. Chloride at bar depth is reported as a "
                f"ratio to the 0.20%-by-cement-mass corrosion threshold.")

    def render(self, f, rng, surface):
        ds = _ds(rng)
        hc = f["halfcell_pct_below_350mv"]
        tests = {"top_deck_delaminated_pct": f["top_deck_delam_pct"],
                 "halfcell_readings_more_negative_than_minus_350mv_pct": hc if hc is not None else "not tested this cycle",
                 "chloride_at_bar_depth_vs_threshold": f["chloride_to_threshold"],
                 "rust_staining": f["rust_staining"]}
        if surface == "json":
            return dumps(prune({
                "structure": {"name": f["garage"], "city": f["city"], "system": f["system"], "levels": f["levels"],
                              "spaces": f["spaces"], "use": f["use"], "built": f["built"]},
                "assessment": {"engineer": f"{f['engineer']}, PE ({f['firm']})", "date": f["assessment_date"].isoformat(),
                               "previous_classification": f["prev_class"]},
                "governing_finding": {"description": self._finding_text(f), "action_taken": self._made_safe_text(f)},
                "testing": tests,
                "rule_excerpt": self._rule(f),
            }))
        if surface == "assessment_report":
            return "\n".join([
                f"PERIODIC CONDITION ASSESSMENT -- {f['garage'].upper()}",
                kv_lines([("City", f["city"]), ("Structure", f"{f['system']}, {f['levels']} levels, {f['spaces']} spaces"),
                          ("Use", f["use"]), ("Built", f["built"]), ("Engineer", f"{f['engineer']}, PE"),
                          ("Firm", f["firm"]), ("Date", fmt_date(f["assessment_date"], rng, ds)),
                          ("Previous filing", f["prev_class"])], rng), "",
                "Governing finding:", f"  {self._finding_text(f)}", f"  Action taken: {self._made_safe_text(f)}", "",
                table(["Test", "Result"], [["Top deck delaminated", f"{f['top_deck_delam_pct']}% of area"],
                                           ["Half-cell < -350 mV", show(hc, rng, lambda v: f"{v}% of readings")],
                                           ["Chloride / threshold", f["chloride_to_threshold"]],
                                           ["Rust staining", f["rust_staining"]]], rng), "",
                "Classification: (to be filed)", "", self._rule(f)])
        para = [f"Re: {f['garage']} ({f['system']}, {f['levels']} levels, {f['use']}).",
                f"Our assessment on {fmt_date(f['assessment_date'], rng, ds)} found {self._finding_text(f)}.",
                self._made_safe_text(f).capitalize() + ".",
                f"Top deck sounding shows {f['top_deck_delam_pct']}% delaminated; chloride at bar depth is "
                f"{f['chloride_to_threshold']} times the threshold"
                + (f" and {hc}% of half-cell readings are below -350 mV." if hc is not None else "; half-cell testing was not done this cycle."),
                f"Rust staining: {f['rust_staining']}.", f"The previous filing was '{f['prev_class']}'.",
                pick(rng, ["The owner has asked us to keep the garage fully open if at all possible.",
                           "The operator reports no complaints from users.", "We will issue the filing this week.", ""])]
        return (f"From: {f['engineer']}, PE, {f['firm']}\nTo: garage owner; cc Department of Buildings, {f['city']}\n\n"
                + noisy(join_sentences(para), rng, 0.1) + "\n\n" + self._rule(f))

    def questions(self):
        return {
            "assessment_classification": choice("Which classification should the engineer file for this structure?", {
                "repair_within_90_days": "Repair within 90 days: deterioration could become unsafe within a year.",
                "serviceable": "Serviceable: nothing that is or could soon become unsafe.",
                "serviceable_repair_in_cycle": "Serviceable, with repairs to be completed before the next assessment.",
                "unsafe": "Unsafe: an imminent threat to people remains in place.",
            }),
            "immediate_protection": choice("What protective measure is needed right now?", {
                "close_level_or_structure": "Close the affected level, or the whole structure, to users.",
                "cordon_and_shore_area": "Barricade the area and install or keep temporary shoring.",
                "none_required": "No immediate measure; it can wait for the planned repairs.",
                "remove_loose_and_monitor": "Remove loose material and watch the area until it is repaired.",
            }),
            "corrosion_activity": score("How active is reinforcement corrosion in the structure, judging by the test data?", [
                "Dormant: chloride well below threshold and potentials quiet.",
                "Possible: some readings or chloride levels approach the corrosion range.",
                "Probable: chloride at or above threshold, or many active potentials.",
                "Advanced: high chloride and widespread active potentials or staining.",
            ]),
            "public_exposure": score("How exposed are garage users to the governing finding as things stand today?", [
                "Not exposed: the area is closed off or out of use.",
                "Slightly: users pass rarely or the hazard is minor.",
                "Regularly: users and vehicles are under or beside it every day.",
                "Constantly: a busy route in a public facility, with no protection in place.",
            ]),
            "department_notification": noul(
                "The rule requires the engineer to notify the Department within 24 hours of this finding.",
                yes="The finding was an imminent threat to people when found, whether or not it was made safe that day.",
                no="The conditions are deterioration that could become unsafe later, not an imminent threat."),
        }


# =========================================================================== schema 5: post-flood scour
class PostFloodScourAssessment(Schema):
    name = "post_flood_scour_assessment"
    description = "A bridge inspected after a flood under its scour plan of action: reopen, restrict or keep closed."
    surfaces = {"json": 0.5, "event_log": 0.25, "inspection_summary": 0.25}

    FOUNDATIONS = {"spread_gravel": ("spread footing on gravel", True), "spread_rock": ("spread footing on rock", False),
                   "h_piles": ("driven steel H-piles", False), "timber_piles": ("timber piles", True),
                   "shafts": ("drilled shafts", False), "unknown": ("unknown (no plans on file)", True)}

    def sample_factors(self, rng):
        reg = regime(rng)
        fnd = pick_w(rng, {"spread_gravel": 3, "spread_rock": 1, "h_piles": 2, "timber_piles": 1, "shafts": 1,
                           "unknown": 1})
        closure_stage = round(between(rng, 3.2, 6.8), 1)
        peak = round(closure_stage + between(rng, -1.2, 0.8), 2)
        und = pick_w(rng, {0: 3, 1: 3, 2: 2, 3: 1})
        debris = pick(rng, [0, 10, 20, 25, 35, 50])
        embank = pick_w(rng, {"none": 6, "slope_scarp": 2, "shoulder_undermined": 1})
        settlement = False
        probe = pick_w(rng, {"probing_rod": 3, "sonar_from_boat": 3, "diver": 1, "limited_high_flow": 2})
        if reg == "clear":
            if chance(rng, 0.5):
                und, debris, embank, peak = 0, pick(rng, [0, 10, 20, 30]), "none", round(closure_stage - between(rng, 0.6, 1.4), 2)
                probe = pick(rng, ["probing_rod", "sonar_from_boat", "diver"])
            else:
                und, settlement = 3, chance(rng, 0.6)
                peak = round(closure_stage + between(rng, 0.2, 1.0), 2)
        elif reg == "borderline":
            peak = near(rng, closure_stage, 0.15)
            und = pick_w(rng, {0: 2, 1: 4, 2: 3})
        else:  # conflicting: deep scour on piles, or a modest-looking one on gravel, with weak soundings
            if chance(rng, 0.5):
                fnd, und = pick(rng, ["h_piles", "shafts"]), 2
            else:
                fnd, und = pick(rng, ["spread_gravel", "timber_piles", "unknown"]), 1
            probe = pick(rng, ["limited_high_flow", "probing_rod"])
            debris = pick(rng, [20, 35, 50])
        # undermining geometry, consistent with the latent level
        fdesc, erodible = self.FOUNDATIONS[fnd]
        if fnd in ("h_piles", "timber_piles", "shafts"):
            exposure = {0: 0.0, 1: round(between(rng, 0.3, 1.0), 1), 2: round(between(rng, 1.2, 2.5), 1),
                        3: round(between(rng, 2.8, 4.5), 1)}[und]
            what = "shaft" if fnd == "shafts" else "pile"
            embed = pick(rng, [6, 7.5, 9, 11, 14]) if fnd != "timber_piles" else pick(rng, [5, 6, 7.5])
            measure = (f"{exposure} m of {what} length newly exposed below the cap (design embedment about {embed} m)"
                       if exposure else "bed at or above the bottom of the pile cap")
        else:
            measure = {0: "bed level at or above the top of footing", 1: "top of footing exposed, bed about "
                       f"{round(between(rng, 0.1, 0.4), 1)} m below it",
                       2: f"bed below the footing bottom at the upstream nose, undermined about "
                          f"{round(between(rng, 0.3, 0.9), 1)} m back (under 25% of the footing width)",
                       3: f"undermined about {round(between(rng, 1.0, 2.2), 1)} m back under the footing (over 25% of its "
                          f"width)"}[und]
        return {"regime": reg, "owner": _county(rng), "bridge_id": _bridge_id(rng), "road": _route(rng),
                "river": pick(rng, WATERS), "spans": rng.randint(2, 5), "pier": f"pier {rng.randint(1, 3)}",
                "foundation": fnd, "erodible": erodible, "undermining": und, "measure_text": measure,
                "settlement": settlement, "closure_stage_m": closure_stage, "watch_stage_m": round(closure_stage - 0.8, 1),
                "peak_stage_m": peak, "closed_during_event": chance(rng, 0.6), "debris_pct": debris,
                "embankment": embank, "probe": probe, "stage_now": pick(rng, ["falling", "falling", "near normal",
                                                                              "still high"]),
                "detour_km": pick(rng, [1.5, 4, 9, 16, 28, 42]), "aadt": pick(rng, [300, 900, 2400, 6200, 13500]),
                "emergency_route": pick_w(rng, {"none": 4, "school_bus": 2, "fire_ems": 2, "hospital_access": 1}),
                "inspector": person(rng), "event_date": rand_date(rng),
                "scour_code": "unknown foundation, plan of action on file" if fnd == "unknown"
                else "scour critical, plan of action on file"}

    def _und(self, f):
        u = f["undermining"]
        if f["settlement"]:
            u = 3
        return u

    def intended(self, f):
        u = self._und(f)
        weak_probe = f["probe"] == "limited_high_flow"
        if u >= 3 or (u == 2 and f["erodible"]):
            status = "keep_closed_pending_engineering"
        elif u == 2 or f["embankment"] == "shoulder_undermined":
            status = "reopen_with_restrictions"
        elif u == 1 or weak_probe or f["debris_pct"] >= 35:
            status = "reopen_with_follow_up_soundings"
        else:
            status = "reopen_normally"
        if u >= 2:
            cm = "emergency_scour_repair"
        elif u == 1 or f["embankment"] != "none":
            cm = "programmed_countermeasure"
        elif f["debris_pct"] >= 20:
            cm = "remove_debris"
        else:
            cm = "none_needed"
        p = {1.5: 0, 4: 0, 9: 1, 16: 1, 28: 2, 42: 2}[f["detour_km"]]
        if f["aadt"] >= 6000:
            p += 1
        p += {"none": 0, "school_bus": 0, "fire_ems": 1, "hospital_access": 1}[f["emergency_route"]]
        triggered = f["peak_stage_m"] >= f["closure_stage_m"] or f["debris_pct"] >= 35
        return {"bridge_status_after_flood": status, "countermeasure_need": cm, "foundation_undermining": u,
                "reopening_pressure": _clamp(p), "closure_trigger_reached": "yes" if triggered else "no"}

    def _poa(self, f):
        return (f"Plan of action ({f['owner']}): watch stage {f['watch_stage_m']} m - hourly visual checks; closure "
                f"stage {f['closure_stage_m']} m on the upstream gauge, or debris blocking a third or more of a span "
                f"opening - close the bridge. After the flood, inspect before reopening; any undermining of a spread "
                f"footing or of an unknown foundation keeps the bridge closed pending an engineer's evaluation.")

    def _probe_text(self, f):
        return {"probing_rod": "probing rod from the deck and the bank", "sonar_from_boat": "sonar from a boat",
                "diver": "dive team, hands-on at the footing",
                "limited_high_flow": "probing rod only; current too strong to reach the upstream nose"}[f["probe"]]

    def _embank_text(self, f):
        return {"none": "approach embankments intact", "slope_scarp": "scarp in the downstream embankment slope, "
                "roadway shoulder intact", "shoulder_undermined": "embankment eroded back under the paved shoulder"}[
            f["embankment"]]

    def _route_text(self, f):
        return {"none": "not on a designated emergency or school route", "school_bus": "on two school bus routes",
                "fire_ems": "primary fire/EMS route to the east side", "hospital_access": "shortest route to the regional "
                "hospital"}[f["emergency_route"]]

    def render(self, f, rng, surface):
        ds = _ds(rng)
        ev = f["event_date"]
        if surface == "json":
            return dumps(prune({
                "bridge": {"id": f["bridge_id"], "road": f["road"], "river": f["river"], "spans": f["spans"],
                           "foundation_at_" + f["pier"].replace(" ", "_"): self.FOUNDATIONS[f["foundation"]][0],
                           "scour_status": f["scour_code"], "aadt": f["aadt"]},
                "event": {"date": ev.isoformat(), "peak_stage_m": f["peak_stage_m"],
                          "bridge_closed_during_event": f["closed_during_event"], "river_now": f["stage_now"]},
                "post_flood_inspection": {"inspector": f["inspector"], "method": self._probe_text(f),
                                          "scour_at_" + f["pier"].replace(" ", "_"): f["measure_text"],
                                          "pier_settlement_or_tilt": "survey shows 18 mm settlement at the pier cap"
                                          if f["settlement"] else "none found",
                                          "debris_blocking_pct_of_span": f["debris_pct"],
                                          "embankments": self._embank_text(f)},
                "closure_impact": {"detour_km": f["detour_km"], "route_role": self._route_text(f)},
                "plan_of_action": self._poa(f),
            }))
        if surface == "event_log":
            t0 = ev
            rows = [(t0, f"stage {f['watch_stage_m']} m reached, watch started" if f["peak_stage_m"] >= f["watch_stage_m"]
                     else "river rising, below the watch stage; spot checks only"),
                    (t0, f"peak stage {f['peak_stage_m']} m" + (" - bridge closed" if f["closed_during_event"]
                                                                  else " - bridge left open")),
                    (t0 + timedelta(days=1), f"post-flood inspection by {f['inspector']} ({self._probe_text(f)})"),
                    (t0 + timedelta(days=1), f"{f['pier']}: {f['measure_text']}"),
                    (t0 + timedelta(days=1), "settlement 18 mm at pier cap (level survey)" if f["settlement"]
                     else "no settlement or tilt found"),
                    (t0 + timedelta(days=1), f"debris raft blocking ~{f['debris_pct']}% of a span" if f["debris_pct"]
                     else "no debris accumulation"),
                    (t0 + timedelta(days=1), self._embank_text(f))]
            log = "\n".join(f"{fmt_date(d, rng, 'iso')}  {t}" for d, t in rows)
            return "\n".join([f"EVENT LOG -- {f['bridge_id']} {f['road']} over {f['river']}",
                              f"Foundation at {f['pier']}: {self.FOUNDATIONS[f['foundation']][0]}; {f['scour_code']}",
                              "", log, "",
                              f"River now: {f['stage_now']}. Closure detour {f['detour_km']} km; AADT {f['aadt']:,}; "
                              f"{self._route_text(f)}.", "", self._poa(f)])
        para = [f"{f['bridge_id']} ({f['road']} over {f['river']}, {f['spans']} spans) after the "
                f"{fmt_date(ev, rng, ds)} flood. Peak gauge {f['peak_stage_m']} m; the bridge was "
                + ("closed" if f["closed_during_event"] else "not closed") + " during the event.",
                f"{f['inspector']} inspected by {self._probe_text(f)}. At {f['pier']} "
                f"({self.FOUNDATIONS[f['foundation']][0]}): {f['measure_text']}.",
                "Level survey shows 18 mm settlement at the pier cap." if f["settlement"] else "",
                f"Debris raft blocking about {f['debris_pct']}% of one span." if f["debris_pct"] else "",
                self._embank_text(f).capitalize() + ".", f"The river is {f['stage_now']}.",
                f"Closure means a {f['detour_km']} km detour for {f['aadt']:,} vehicles a day; the bridge is "
                f"{self._route_text(f)}."]
        return ("POST-FLOOD INSPECTION SUMMARY\n\n" + noisy(join_sentences(para), rng, 0.1) + "\n\n" + self._poa(f))

    def questions(self):
        return {
            "bridge_status_after_flood": choice("What should the owner do with the bridge now?", {
                "keep_closed_pending_engineering": "Keep it closed until an engineer has evaluated the foundation.",
                "reopen_normally": "Reopen to normal traffic with no special follow-up.",
                "reopen_with_follow_up_soundings": "Reopen, and sound the piers again after the river drops or the next high flow.",
                "reopen_with_restrictions": "Reopen with a lane closure or weight limit until repairs are made.",
            }),
            "countermeasure_need": choice("What scour work does the finding call for?", {
                "emergency_scour_repair": "Place emergency riprap or grout bags at the pier before the next high water.",
                "none_needed": "No scour work is needed.",
                "programmed_countermeasure": "Design a permanent countermeasure for the planned works programme.",
                "remove_debris": "Clear the debris raft; the foundation itself needs no work.",
            }),
            "foundation_undermining": score("How badly is the pier foundation undermined?", [
                "Not at all: the bed is at or above the foundation's protected level.",
                "Exposed: the top of the foundation or a little of the piles is exposed, nothing undermined.",
                "Partly undermined: material is lost under or along the foundation, support still largely intact.",
                "Seriously undermined: support is lost over a large area, or the pier has moved.",
            ]),
            "reopening_pressure": score("How much pressure is there to reopen the bridge quickly?", [
                "Little: a short detour on roads of the same standard.",
                "Some: a noticeable detour for local traffic.",
                "Considerable: a long detour for heavy traffic, or an emergency route lost.",
                "Severe: a long detour that also cuts an emergency or hospital route for many people.",
            ]),
            "closure_trigger_reached": noul(
                "The plan of action's closure trigger was reached during this flood.",
                yes="The peak gauge reading reached the closure stage, or debris blocked a third or more of a span.",
                no="The river stayed below the closure stage and debris stayed below the blockage trigger."),
        }


WORKFLOW = Workflow(
    name="structural_inspection_rating",
    title="Structural inspection rating",
    description="Condition ratings and follow-up actions for existing bridge and building elements against the "
                "owner's written inspection-manual criteria.",
    domain="construction / civil engineering asset inspection",
    schemas=[BridgeDeckCondition(), SteelGirderSectionLoss(), CulvertConditionRating(), ParkingStructureCondition(),
             PostFloodScourAssessment()],
    notes="Fictional assets, owners and manuals; the governing manual excerpt is quoted in every state. In-service "
          "condition assessments only: no municipal plan review, permit inspections or occupancy decisions "
          "(building_permit_review), and no billing or payment of repair contracts.",
)
