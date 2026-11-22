"""Construction RFI triage: requests for information from a contractor, judged for whether the contract documents
already answer them, whether the answer implies a design change, how urgent they are, and which party responds.

Every state quotes the project's written RFI procedure or coordination clause with case-specific terms (response
periods, required RFI contents, order-of-precedence rules, field-verification duties). Projects, firms, drawings and
people are fictional. This is document control on a construction project: no pricing of change orders (see
construction_change_order_review), no municipal plan review, and nothing about invoices or payment.

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
    fmt_date,
    join_sentences,
    kv_lines,
    money,
    noisy,
    noul,
    person,
    pick,
    pick_w,
    prune,
    rand_date,
    regime,
    score,
    table,
)

# =========================================================================== shared pools
STEMS = ["Alder", "Basalt", "Cobalt", "Driftwood", "Ember", "Fernhill", "Ironbark", "Juniper", "Lakeshore", "Oakridge",
         "Pinecrest", "Quarry", "Redfern", "Saltmarsh", "Upland", "Valewood", "Westbrook", "Yarrow", "Ashgrove",
         "Bluestem", "Copperline", "Dunmore", "Elmstead", "Harrowgate", "Kingsfold", "Larkspur", "Moraine", "Nettlebed",
         "Orrin", "Rushford", "Stannard", "Wexcombe", "Whitlow", "Brackett", "Calloway", "Denholm", "Fairbairn",
         "Glenmoor", "Hatherley", "Lindell", "Quillhaven", "Pellbridge", "Marrowdale", "Corvina", "Halden"]
BUILDINGS = ["{s} Middle School addition", "{s} Public Library renovation", "{s} Fire Station No. {n}",
             "{s} Transit Center", "{s} Senior Housing", "{s} Community Recreation Center", "{s} Police Headquarters",
             "{s} Health Clinic expansion", "{s} Science Building renovation", "{s} County Courthouse renovation",
             "{s} Aquatic Center", "{s} Medical Office Building", "{s} Student Housing, Phase {n}",
             "{s} Elementary School modernization", "{s} City Hall seismic retrofit", "{s} Laboratory Building",
             "{s} Performing Arts Center", "{s} Bus Maintenance Facility", "{s} Veterans Clinic",
             "{s} Water Department operations building"]
DESIGN_SUFFIX = ["Architects", "Design Studio", "Architecture + Planning", "Partners Architects", "Collaborative"]
GC_SUFFIX = ["Constructors", "Builders", "Contracting", "Construction Co.", "General Contractors", "Construction Group",
             "Building Company", "Builders & Engineers"]
PARTIES = {"architect": "the architect", "structural": "the structural engineer of record",
           "mep": "the mechanical/electrical engineer", "owner": "the owner", "contractor": "the contractor"}


def _project(rng):
    stem = pick(rng, STEMS)
    name = pick(rng, BUILDINGS).format(s=stem, n=rng.randint(2, 6))
    arch = f"{pick(rng, STEMS)} {pick(rng, DESIGN_SUFFIX)}"
    gc = f"{pick(rng, STEMS)} {pick(rng, GC_SUFFIX)}"
    return name, arch, gc


def _rfi_no(rng):
    return pick(rng, [f"RFI-{rng.randint(12, 480):04d}", f"RFI #{rng.randint(12, 480)}", f"{rng.randint(12, 480):03d}"])


def _ds(rng):
    return pick(rng, ["iso", "us", "long", "eu"])


def _clamp(x, lo=0, hi=3):
    return max(lo, min(hi, x))


def _rfi_label(no):
    return no if no.startswith("RFI") else f"RFI {no}"


def _working_to_calendar(days):
    return days + 2 * (days // 5)


# =========================================================================== schema 1: drawing conflict RFI
class DrawingConflictRfi(Schema):
    name = "drawing_conflict_rfi"
    description = "An RFI about conflicting or missing information in the drawings: who answers, how, and how urgently."
    surfaces = {"json": 0.5, "rfi_form": 0.3, "contractor_email": 0.2}

    # (question, party, docs note when the documents answer it, note when they do not, design-change base 0-3,
    #  activity, fabricated item with a lead time?)
    SCEN = [
        ("Beam B12 at grid D/4-5: framing plan S-202 shows W16x31 but section 3/S-401 shows W18x35. Which governs?",
         "structural", "General note 4 on S-001 says larger-scale details govern over plans.",
         "The general notes say nothing about precedence, and the beam schedule lists a third size, W16x36.", 1,
         "steel shop drawings for level 2", True),
        ("Top of footing F7 is 98.45 on S-101 but 98.15 on section 2/S-301. Which elevation is correct?",
         "structural", "The footing schedule on S-102 lists F7 at 98.15 and is referenced as governing on S-101.",
         "The footing schedule repeats 98.45 and the column base detail assumes 98.15.", 1, "footing excavation", False),
        ("Lap splice length for #8 vertical bars in shear wall SW-2 is not shown on S-501. Please advise.",
         "structural", "The lap splice table in general notes S-002 covers #8 bars in walls.",
         "The lap table on S-002 stops at #7 bars.", 1, "shear wall rebar placement", False),
        ("A 300 mm duct must pass through the web of beam B7 at grid C/6; no penetration is shown. Is this acceptable?",
         "structural", "Detail 9/S-502 gives a standard web-opening detail for openings up to 350 mm in this beam series.",
         "No web-opening detail exists, and B7 is a moment-frame beam.", 2, "duct installation, level 3", False),
        ("Slab depression for the lobby terrazzo is shown on A-101 but not on S-201. Should the slab be depressed?",
         "structural", "Structural note 12 says to provide depressions shown on architectural drawings, 50 mm max.",
         "The terrazzo needs 75 mm, more than the 50 mm structural note 12 allows without review.", 2,
         "level 1 slab pour", False),
        ("Embed plate EP-4 for the canopy conflicts with the post-tensioning tendon profile at grid A/2. Can it move?",
         "structural", "The PT shop drawings already reviewed show EP-4 shifted 150 mm east, clear of the tendons.",
         "Moving EP-4 changes the canopy connection, and no alternative location is detailed.", 2,
         "level 2 PT slab pour", False),
        ("Door D214 swings into the column at grid E/3 as drawn on A-201. Please advise.", "architect",
         "Enlarged plan 4/A-401 shows D214 reversed to swing the other way, clear of the column.",
         "The enlarged plans repeat the conflict, and reversing the swing blocks the corridor.", 1,
         "door frame installation", True),
        ("A fire damper above the hard ceiling in corridor 2C has no access panel on the reflected ceiling plan A-601.",
         "architect", "Spec 08 31 13 requires access panels at every fire damper 'whether shown or not'.",
         "Spec 08 31 13 only covers access panels 'where shown', and none is shown in this corridor.", 1,
         "ceiling close-in, level 2", False),
        ("Corridor C2 ceiling is 2.70 m on A-501 but the duct main on M-201 sits with its bottom at 2.55 m.", "mep",
         "Note 6 on M-001 says ducts may be flattened to a 4:1 aspect ratio to suit ceilings without further review.",
         "The duct is already at its flattest ratio and no other routing is shown.", 2, "ductwork rough-in, level 2",
         False),
        ("Panel LP-2B is shown on E-201 in a wall that is only 100 mm deep. Should it be surface mounted?", "mep",
         "Panel schedule note 3 says panels in shallow walls are to be surface mounted.",
         "Nothing addresses it, and a surface panel would project into the egress corridor.", 1,
         "electrical rough-in", False),
        ("The finish colour for the exposed atrium concrete is not given. Spec 03 35 00 says 'colour by owner'.",
         "owner", "Owner selection memo No. 4 already fixed the atrium colour as Pigment C-12.",
         "No owner selection has been issued for this item.", 0, "atrium wall pours", False),
        ("The owner's IT standard calls for a different equipment cabinet than spec 27 11 00. Which should be used?",
         "owner", "Addendum 2 adopted the owner's IT standard for all telecom rooms.",
         "The addendum is silent and the two cabinets differ in size and power.", 1, "telecom room fit-out", True),
        ("Where can the tower crane be placed, and can it bear on the podium slab?", "contractor",
         "Spec 01 50 00 assigns crane location and support to the contractor, with loads on the permanent structure "
         "to be submitted for review.", "", 0, "crane erection", False),
        ("What excavation support is required along the east property line?", "contractor",
         "Spec 31 50 00 makes excavation support a delegated design by the contractor's engineer.", "", 0,
         "mass excavation", False),
        ("May the level 3 slab be poured in three segments instead of two? Construction joints are not shown.",
         "structural", "Note 18 on S-002 allows construction joints at mid-span of any bay without further approval.",
         "S-002 requires construction joint locations to be approved by the structural engineer.", 1,
         "level 3 slab pour", False),
    ]

    def sample_factors(self, rng):
        reg = regime(rng)
        si = rng.randrange(len(self.SCEN))
        q, party, in_note, out_note, base, activity, fabricated = self.SCEN[si]
        if party == "contractor":
            status = "means_methods"
        else:
            status = pick_w(rng, {"in_docs": 35, "needs_design": 65})
        change = base + pick(rng, [-1, 0, 0, 1]) if status == "needs_design" else 0
        response_days = pick(rng, [5, 7, 10])
        lead = pick(rng, [10, 15, 21, 28]) if fabricated else 0
        start_in = pick(rng, [3, 7, 12, 20, 35, 60])
        stopped = chance(rng, 0.12)
        claim = pick_w(rng, {"none": 4, "possible": 3, "cost_and_time": 2})
        proposal = chance(rng, 0.6)
        if reg == "clear":
            if chance(rng, 0.5):
                start_in, stopped, claim = pick(rng, [45, 60, 90]), False, "none"
                if status == "needs_design":
                    change = 0
            else:
                start_in, stopped = pick(rng, [2, 3]), chance(rng, 0.5)
                if status == "needs_design":
                    change = 3
        elif reg == "borderline":
            start_in = _working_to_calendar(response_days) + lead + pick(rng, [-3, -1, 0, 2, 4])
        else:  # conflicting: the contractor's own impact claim points the other way from the documents
            claim = "cost_and_time" if status != "needs_design" or change <= 1 else "none"
        return {"regime": reg, "scen": si, "party": party, "status": status, "design_change": _clamp(change),
                "response_days": response_days, "lead_days": lead, "start_in_days": max(1, start_in),
                "work_stopped": stopped, "contractor_impact": claim, "proposal": proposal,
                "project": _project(rng), "rfi": _rfi_no(rng), "author": person(rng), "date": rand_date(rng),
                "spec_ref": pick(rng, ["01 26 00", "01 31 19", "00 72 00 3.2"])}

    def _urgency(self, f):
        if f["work_stopped"]:
            return 3
        slack = f["start_in_days"] - f["lead_days"]
        window = _working_to_calendar(f["response_days"])
        if slack <= window:
            return 2
        if slack <= 2 * window:
            return 1
        return 0

    def intended(self, f):
        st = f["status"]
        party = "contractor_responsibility" if st == "means_methods" else {
            "structural": "structural_engineer", "mep": "mep_engineer", "architect": "architect", "owner": "owner_decision"}[
            f["party"]]
        if st == "in_docs":
            handling = "answer_with_reference"
        elif st == "means_methods":
            handling = "return_as_means_and_methods"
        elif f["design_change"] >= 2:
            handling = "open_potential_change"
        else:
            handling = "issue_clarification"
        return {"responding_party": party, "rfi_handling": handling,
                "answer_in_documents": "yes" if st in ("in_docs", "means_methods") else "no",
                "design_change_likelihood": f["design_change"], "response_urgency": self._urgency(f)}

    def _procedure(self, f):
        return (f"RFI procedure (Section {f['spec_ref']}): the design team responds within {f['response_days']} working "
                f"days of a complete RFI. RFIs answered by the Contract Documents are returned with the reference. "
                f"Questions of construction means, methods and temporary works are the contractor's. An RFI response "
                f"does not change the contract; any response the contractor believes changes cost or time must be "
                f"raised as a change request.")

    def _review_note(self, f):
        _, _, in_note, out_note, _, _, _ = self.SCEN[f["scen"]]
        return in_note if f["status"] in ("in_docs", "means_methods") else out_note

    def _timing(self, f, rng, ds):
        start = f["date"] + timedelta(days=f["start_in_days"])
        t = f"{self.SCEN[f['scen']][5]} scheduled to start {fmt_date(start, rng, ds)}"
        if f["lead_days"]:
            t += f"; procurement and fabrication lead time about {f['lead_days']} days"
        if f["work_stopped"]:
            t = f"crew stopped on {self.SCEN[f['scen']][5]} pending the answer"
        return t

    def _impact(self, f):
        return {"none": "no cost or schedule impact anticipated", "possible": "possible cost impact",
                "cost_and_time": "contractor anticipates cost and time impact"}[f["contractor_impact"]]

    def _proposal(self, f):
        if not f["proposal"]:
            return None
        st = f["status"]
        if st == "means_methods":
            return "Contractor proposes its own scheme and asks the designer to confirm it."
        return pick_static(f, ["Contractor will proceed on its own reading of the drawings unless directed "
                               "otherwise by the date needed.",
                               "Contractor asks for a sketch or written direction.",
                               "Contractor suggests the more conservative of the options."])

    def render(self, f, rng, surface):
        ds = _ds(rng)
        proj, arch, gc = f["project"]
        q = self.SCEN[f["scen"]][0]
        if surface == "json":
            return dumps(prune({
                "rfi": {"number": f["rfi"], "project": proj, "from": f"{f['author']}, {gc}", "to": arch,
                        "submitted": f["date"].isoformat(), "question": q, "proposed_solution": self._proposal(f),
                        "contractor_impact_statement": self._impact(f)},
                "schedule": self._timing(f, rng, "iso"),
                "design_team_check": self._review_note(f),
                "procedure": self._procedure(f),
            }))
        if surface == "rfi_form":
            return "\n".join([
                f"REQUEST FOR INFORMATION {f['rfi'].replace('RFI-', '').replace('RFI #', '')}",
                kv_lines([("Project", proj), ("To", arch), ("From", f"{f['author']} ({gc})"),
                          ("Date", fmt_date(f["date"], rng, ds)), ("Schedule", self._timing(f, rng, ds)),
                          ("Impact", self._impact(f))], rng),
                "", "Question:", q, "", "Proposed solution:", self._proposal(f) or "(none given)", "",
                "Coordinator's check of the documents: " + self._review_note(f), "", self._procedure(f)])
        body = [q, self._proposal(f) or "", f"Timing: {self._timing(f, rng, ds)}.", f"Impact: {self._impact(f)}."]
        return (f"Subject: {f['rfi']} -- {proj}\nFrom: {f['author']} <{f['author'].split()[0].lower()}@"
                f"{gc.split()[0].lower()}-build.example>\nTo: RFI coordinator, {arch}\n\n"
                + noisy(join_sentences(body), rng, 0.12)
                + f"\n\n> Coordinator's note: {self._review_note(f)}\n\n{self._procedure(f)}")

    def questions(self):
        return {
            "responding_party": choice("Which party should prepare the response to this RFI?", {
                "architect": "The architect, because it concerns the architectural design.",
                "contractor_responsibility": "Nobody on the design team: it is the contractor's own means and methods.",
                "mep_engineer": "The mechanical or electrical engineer.",
                "owner_decision": "The owner, because it needs an owner selection or decision.",
                "structural_engineer": "The structural engineer of record.",
            }),
            "rfi_handling": choice("How should the design team handle this RFI?", {
                "answer_with_reference": "Answer by pointing to the drawing, note or specification that already covers it.",
                "issue_clarification": "Issue a clarification or sketch that does not change scope, cost or time.",
                "open_potential_change": "Answer it and log it as a potential change to scope or cost.",
                "return_as_means_and_methods": "Return it: the question is the contractor's to resolve.",
            }),
            "answer_in_documents": noul(
                "The contract documents already contain the answer to this RFI.",
                yes="A drawing, note, schedule or specification section answers it, or assigns the question to the "
                    "contractor.",
                no="The documents conflict without a stated resolution, or are silent, so the design team must decide."),
            "design_change_likelihood": score("How likely is the response to change the design?", [
                "Remote: the answer simply confirms what is drawn.",
                "Possible: a minor clarification that may adjust a detail.",
                "Likely: an element will be resized, relocated or redetailed.",
                "Near certain: the design must change in a way that affects other work.",
            ]),
            "response_urgency": score("How urgently does the contractor need an answer?", [
                "Not urgent: the work is weeks beyond the normal response period.",
                "Soon: the answer is needed shortly after the normal response period.",
                "Pressing: the work, or a fabrication order, falls inside the normal response period.",
                "Immediate: crews are already waiting on the answer.",
            ]),
        }


def pick_static(f, options):
    """A stable choice keyed on the case's ids (no rng draw), so the same case always reads the same."""
    key = sum(map(ord, str(f.get("rfi", "")) + str(f.get("author", ""))))
    return options[key % len(options)]


# =========================================================================== schema 2: MEP coordination clash
class MepCoordinationClash(Schema):
    name = "mep_coordination_clash"
    description = "A clash found in building-services coordination: who resolves it, whether to hold work, and cost exposure."
    surfaces = {"json": 0.5, "clash_report": 0.25, "meeting_minutes": 0.25}

    # (services item, what it clashes with -- "{beam}" means the structural member itself, kind, gravity line?)
    CLASHES = [
        ("600x400 supply duct", "{beam}", "duct", False),
        ("200 mm sprinkler main", "750x350 return duct", "pipe", False),
        ("150 mm sanitary line (gravity, 1% slope)", "{beam}", "gravity", True),
        ("300 mm cable tray", "chilled-water supply and return pair", "tray", False),
        ("VAV box VB-3-12", "the corridor ceiling at 2.60 m", "equipment", False),
        ("100 mm storm leader (gravity)", "{beam}", "gravity", True),
        ("450 mm round exhaust duct", "{beam}", "duct", False),
        ("fire-alarm conduit bank", "recessed light fixtures in the lobby soffit", "conduit", False),
        ("grease exhaust duct (fire-rated enclosure)", "{beam}", "duct", False),
        ("medical-gas header", "the smoke damper SD-14 access zone", "pipe", False),
        ("250 mm heating hot-water main", "{beam}", "pipe", False),
        ("busway BD-2", "the 900x450 main exhaust duct", "tray", False),
    ]
    BEAMS = {"steel": "steel beam W21x50", "pt": "post-tensioned concrete beam PB-{n}", "transfer": "transfer girder TG-{n}"}
    STRUCT = {"steel": "steel beam above, with a web that could take an opening near mid-depth",
              "pt": "post-tensioned concrete beam above (no coring without tendon scanning and engineer approval)",
              "transfer": "transfer girder above (no penetrations permitted)"}

    def sample_factors(self, rng):
        reg = regime(rng)
        ci = rng.randrange(len(self.CLASHES))
        item, against, kind, gravity = self.CLASHES[ci]
        fix = pick_w(rng, {"reroute": 4, "resize": 3, "penetration": 2, "ceiling": 2})
        if gravity and fix == "resize":
            fix = pick(rng, ["penetration", "ceiling"])
        shortfall = {"reroute": pick(rng, [-120, -60, -20, 10]), "resize": pick(rng, [20, 40, 70, 95]),
                     "penetration": pick(rng, [120, 180, 260]), "ceiling": pick(rng, [110, 160, 240])}[fix]
        alt_route = fix == "reroute"
        struct = pick(rng, ["steel", "pt", "transfer"])
        if fix == "penetration":
            struct = "steel"
        elif fix == "ceiling":
            struct = pick(rng, ["pt", "transfer"])
        ceiling_margin = pick(rng, [0, 50, 100, 200])
        installed_near = chance(rng, 0.35)
        days_to_install = pick(rng, [2, 5, 9, 15, 30, 45])
        close_in_days = days_to_install + pick(rng, [10, 20, 30])
        response_days = pick(rng, [5, 7, 10])
        if reg == "clear":
            if chance(rng, 0.5):
                fix, shortfall, alt_route, days_to_install, installed_near = "reroute", -150, True, 40, False
            else:
                fix, shortfall, struct, days_to_install = "ceiling", 260, "transfer", 3
        elif reg == "borderline":
            if fix in ("reroute", "resize"):
                shortfall = pick(rng, [-10, 0, 15, 30])
                alt_route = shortfall <= 0 and chance(rng, 0.7)
                fix = "reroute" if alt_route else "resize"
            days_to_install = _working_to_calendar(response_days) + pick(rng, [-2, 0, 2])
        else:  # conflicting: the trade foreman says it will fit; the numbers say otherwise, or the reverse
            if chance(rng, 0.5):
                fix, shortfall, alt_route = pick(rng, ["penetration", "ceiling"]), pick(rng, [130, 190]), False
                struct = "steel" if fix == "penetration" else pick(rng, ["pt", "transfer"])
            else:
                fix, shortfall, alt_route = "reroute", pick(rng, [-40, -80]), True
                installed_near = True
        if gravity and fix == "resize":
            fix = "penetration"
            struct = "steel"
        alt_route = fix == "reroute"
        against = against.format(beam=self.BEAMS[struct].format(n=rng.randint(1, 6)))
        return {"regime": reg, "clash": ci, "item": item, "against": against, "kind": kind, "gravity": gravity,
                "fix": fix, "shortfall_mm": shortfall, "alt_route": alt_route, "structure": struct,
                "ceiling_margin_mm": ceiling_margin, "installed_near": installed_near,
                "days_to_install": days_to_install, "close_in_days": close_in_days, "response_days": response_days,
                "foreman_view": pick(rng, ["will fit", "needs a design answer", "no opinion"]),
                "level": rng.randint(1, 5), "grid": f"{pick(rng, 'ABCDEFGH')}/{rng.randint(1, 9)}",
                "project": _project(rng), "clash_id": f"CL-{rng.randint(100, 999)}", "coordinator": person(rng),
                "date": rand_date(rng)}

    def intended(self, f):
        res = {"reroute": "reroute_within_coordination", "resize": "engineer_resize_or_reroute",
               "penetration": "structural_penetration_review", "ceiling": "architect_ceiling_revision"}[f["fix"]]
        sh = f["shortfall_mm"]
        sev = 0 if sh <= 0 else 1 if sh <= 50 else 2 if sh <= 150 else 3
        if f["gravity"] and sev < 3 and sh > 0:
            sev += 1
        cost = {"reroute": 0, "resize": 1, "penetration": 2, "ceiling": 2}[f["fix"]]
        if f["installed_near"] and f["fix"] != "reroute":
            cost += 1
        if f["fix"] == "ceiling" and f["ceiling_margin_mm"] >= sh:
            cost -= 1
        hold = f["fix"] != "reroute" and (f["installed_near"] or
                                          f["days_to_install"] <= _working_to_calendar(f["response_days"]) + 3)
        return {"coordination_resolution": res, "space_conflict_severity": sev, "cost_exposure": _clamp(cost),
                "coordination_clause_covers": "yes" if f["fix"] == "reroute" else "no",
                "hold_installation": "yes" if hold else "no"}

    def _clause(self, f):
        return (f"Section 01 31 00 (Coordination): the contractor prepares coordinated services drawings; routing "
                f"adjustments within the space shown are the contractor's responsibility at no cost. Changes to duct or "
                f"pipe sizes, gravity slopes, structural members or ceiling heights need the engineer's or architect's "
                f"approval. Design team review period: {f['response_days']} working days.")

    def _facts(self, f):
        sh = f["shortfall_mm"]
        space = (f"short by {sh} mm in the zone shown" if sh > 0 else
                 f"fits with {abs(sh)} mm to spare once offset" if sh < 0 else "exactly fills the zone shown")
        route = ("a parallel bay 1.2 m away has clear space for the full run" if f["alt_route"]
                 else "no alternative route found in the model")
        return {"space": space, "alternative_route": route, "structure": self.STRUCT[f["structure"]],
                "ceiling": (f"architectural ceiling can drop up to {f['ceiling_margin_mm']} mm and still meet the "
                            f"room's minimum" if f["ceiling_margin_mm"] else "ceiling already at the room's minimum height"),
                "status": (f"installation starts in {f['days_to_install']} days; ceiling close-in in "
                           f"{f['close_in_days']} days")
                + ("; same run already installed in the adjacent bay" if f["installed_near"] else "")}

    def render(self, f, rng, surface):
        ds = _ds(rng)
        proj, arch, gc = f["project"]
        fx = self._facts(f)
        title = f"{f['item']} vs {f['against']}, level {f['level']}, grid {f['grid']}"
        foreman = {"will fit": "trade foreman: 'we can make it fit in the field'",
                   "needs a design answer": "trade foreman wants a design answer before installing",
                   "no opinion": None}[f["foreman_view"]]
        if surface == "json":
            return dumps(prune({
                "clash": {"id": f["clash_id"], "project": proj, "description": title,
                          "found": f["date"].isoformat(), "by": f"{f['coordinator']} ({gc} coordination)"},
                "geometry": {"space": fx["space"], "alternative_route": fx["alternative_route"],
                             "structure_in_the_way": fx["structure"], "ceiling": fx["ceiling"]},
                "site": {"status": fx["status"], "field_comment": foreman},
                "coordination_clause": self._clause(f),
            }))
        if surface == "clash_report":
            rows = [["Clash", f"{f['clash_id']}: {title}"], ["Space", fx["space"]], ["Alternative", fx["alternative_route"]],
                    ["Structure", fx["structure"]], ["Ceiling", fx["ceiling"]], ["Status", fx["status"]]]
            if foreman:
                rows.append(["Field", foreman])
            return "\n".join([f"CLASH DETECTION REPORT -- {proj}", f"Run {fmt_date(f['date'], rng, ds)}, "
                              f"{f['coordinator']}", "", table(["Field", "Detail"], rows, rng), "", self._clause(f)])
        att = [f"{f['coordinator']} (GC coordination)", f"{person(rng)} ({arch})", f"{person(rng)} (mechanical sub)"]
        items = [f"{f['clash_id']} {title}: {fx['space']}; {fx['alternative_route']}.",
                 f"Structure at the clash: {fx['structure']}.", f"{fx['ceiling'][0].upper()}{fx['ceiling'][1:]}.",
                 f"{fx['status'][0].upper()}{fx['status'][1:]}.", (foreman[0].upper() + foreman[1:] + ".") if foreman else None]
        return "\n".join([f"COORDINATION MEETING -- {proj}, {fmt_date(f['date'], rng, ds)}",
                          "Attendees: " + "; ".join(att), "", bullets(items, rng), "", self._clause(f)])

    def questions(self):
        return {
            "coordination_resolution": choice("How should this clash be resolved?", {
                "architect_ceiling_revision": "The architect revises the ceiling or adds a soffit locally.",
                "engineer_resize_or_reroute": "The services engineer approves a resized or rerouted system.",
                "reroute_within_coordination": "The trades reroute within the space already shown, as part of coordination.",
                "structural_penetration_review": "The structural engineer reviews an opening through the member.",
            }),
            "space_conflict_severity": score("How severe is the space conflict?", [
                "None: it fits once offset within the zone shown.",
                "Slight: a few centimetres short, with ordinary ways round it.",
                "Significant: well short of the space, and options are limited.",
                "Severe: far short, or a fixed-slope line with nowhere to go.",
            ]),
            "cost_exposure": score("How much is the owner's cost exposed by the resolution?", [
                "None: the contractor absorbs it as coordination.",
                "Small: a minor design adjustment with a modest cost.",
                "Moderate: redesign or structural work with real cost.",
                "Large: redesign plus removal of work already in place.",
            ]),
            "coordination_clause_covers": noul(
                "Resolving this clash falls within the contractor's coordination duty under the quoted clause.",
                yes="It can be solved by routing within the space shown, with no change of size, slope, structure or "
                    "ceiling height.",
                no="It needs a change the clause reserves to the engineer or the architect."),
            "hold_installation": noul(
                "Installation in the affected area should be held until the design team responds."),
        }


# =========================================================================== schema 3: spec / drawing discrepancy
class SpecDrawingDiscrepancy(Schema):
    name = "spec_drawing_discrepancy"
    description = "An RFI about a conflict between specification and drawings, read against the order-of-precedence clause."
    surfaces = {"json": 0.5, "rfi_form": 0.25, "rfi_log_entry": 0.25}

    # (topic, spec says, drawing says, more stringent: "spec" / "drawing" / None, life safety 0-3,
    #  unit cost difference, unit, quantity range, substitution?)
    PAIRS = [
        ("gypsum board at 1-hour corridor walls", "16 mm Type X board", "13 mm regular board on wall type W3",
         "spec", 3, 6.5, "m2", (400, 2200), False),
        ("roof insulation", "two layers, 150 mm total, R-30", "a single 100 mm layer on detail 2/A-511", "spec", 0,
         14.0, "m2", (600, 3000), False),
        ("slab-on-grade concrete", "32 MPa", "28 MPa in general note 3 on S-001", "spec", 1, 9.0, "m3", (80, 600), False),
        ("stair guard height", "1070 mm", "1000 mm on detail 6/A-521", "spec", 3, 85.0, "m", (40, 160), False),
        ("stair door hardware", "lever locksets", "panic devices in the hardware schedule", "drawing", 3, 650.0, "door",
         (6, 24), False),
        ("corridor paint sheen", "eggshell", "semi-gloss on the finish schedule", None, 0, 0.4, "m2", (800, 4000), False),
        ("laboratory flooring", "sheet vinyl with welded seams", "rubber tile on the finish plan", None, 1, 22.0, "m2",
         (150, 900), False),
        ("podium waterproofing", "fluid-applied membrane", "sheet membrane on detail 4/A-560", None, 1, 18.0, "m2",
         (300, 1500), False),
        ("anchor rods", "ASTM F1554 Grade 55", "Grade 36 on the column base detail", "spec", 2, 40.0, "ea", (60, 300),
         False),
        ("sprinkler heads in public areas", "concealed pendent heads", "standard pendent heads on FP-201", "spec", 0,
         45.0, "ea", (80, 400), False),
        ("toilet partitions", "the named manufacturer's phenolic model 'Series 40'", "the same model on A-601",
         None, 0, 120.0, "compartment", (10, 40), True),
        ("exterior sealant", "the specified silicone 'Product 790'", "the same product in the joint schedule",
         None, 1, 3.5, "m", (500, 2500), True),
    ]
    SECTIONS = ["09 21 16", "07 22 00", "03 30 00", "05 52 13", "08 71 00", "09 91 23", "09 65 00", "07 14 16",
                "05 05 19", "21 13 13", "10 21 13", "07 92 00"]
    PRECEDENCE = {"specs_govern": "where drawings and specifications conflict, the specifications govern",
                  "drawings_govern": "where drawings and specifications conflict, the drawings govern",
                  "stringent_governs": "where documents conflict, the more stringent (higher quality or greater "
                                       "quantity) requirement governs",
                  "silent": "conflicts shall be referred to the architect, whose decision is final"}

    def sample_factors(self, rng):
        reg = regime(rng)
        pi = rng.randrange(len(self.PAIRS))
        prec = pick_w(rng, {"specs_govern": 3, "drawings_govern": 3, "stringent_governs": 3, "silent": 2.5})
        p = self.PAIRS[pi]
        if reg == "clear":
            if chance(rng, 0.6):         # a plain conflict on a non-safety item that the clause settles
                pi = pick(rng, [i for i, x in enumerate(self.PAIRS) if x[4] <= 1 and not x[8]])
                p = self.PAIRS[pi]
                prec = pick(rng, ["specs_govern", "drawings_govern"] + (["stringent_governs"] if p[3] else []))
            else:
                prec = "silent"
        elif reg == "conflicting":
            # precedence picks the weaker, code-relevant requirement: a careful reader will not just apply it
            safety = [i for i, x in enumerate(self.PAIRS) if x[4] >= 2]
            pi = pick(rng, safety)
            p = self.PAIRS[pi]
            prec = "drawings_govern" if p[3] == "spec" else "specs_govern"
        lo, hi = p[7]
        section = self.SECTIONS[pi]
        qty = max(1, round(rng.randint(lo, hi) * pick(rng, [0.25, 0.6, 1.0, 2.5, 6.0])))
        sub_reason = pick(rng, ["the manufacturer has discontinued it", "lead time is now 22 weeks",
                                "the contractor would prefer a cheaper equal product"]) if p[8] else None
        return {"regime": reg, "pair": pi, "precedence": prec, "quantity": qty, "sub_reason": sub_reason,
                "project": _project(rng), "rfi": _rfi_no(rng), "author": person(rng), "date": rand_date(rng),
                "spec_section": section,
                "response_days": pick(rng, [5, 7, 10]),
                "contractor_reading": pick(rng, ["spec", "drawing", "asks"])}

    def _governing(self, f):
        p = self.PAIRS[f["pair"]]
        if p[8]:
            return "substitution"
        prec = f["precedence"]
        if prec == "silent":
            return None
        if prec == "specs_govern":
            return "spec"
        if prec == "drawings_govern":
            return "drawing"
        return p[3]          # stringent: None when the two are not comparable

    def intended(self, f):
        p = self.PAIRS[f["pair"]]
        gov = self._governing(f)
        code_conflict = gov in ("spec", "drawing") and p[4] >= 2 and p[3] is not None and gov != p[3]
        if gov == "substitution":
            g = "not_a_conflict_substitution"
        elif gov is None or code_conflict:
            g = "design_team_must_decide"
        else:
            g = "specification_governs" if gov == "spec" else "drawing_governs"
        delta = p[5] * f["quantity"]
        cost = 0 if delta < 2000 else 1 if delta < 15000 else 2 if delta < 75000 else 3
        if g == "not_a_conflict_substitution":
            route = "redirect_to_substitution_process"
        elif g in ("specification_governs", "drawing_governs"):
            route = "answer_from_documents"
        elif cost >= 2:
            route = "route_as_potential_change"
        else:
            route = "design_clarification"
        resolves = gov in ("spec", "drawing")
        return {"governing_requirement": g, "response_route": route,
                "precedence_resolves": "yes" if resolves else "no", "life_safety_relevance": p[4],
                "cost_delta_exposure": cost}

    def _question(self, f):
        p = self.PAIRS[f["pair"]]
        if p[8]:
            return (f"Spec {f['spec_section']} names {p[1]} for the {p[0]}, but {f['sub_reason']}. May we provide an "
                    f"equivalent product from another manufacturer?")
        lead = {"spec": "We intend to follow the specification unless directed otherwise.",
                "drawing": "We priced and intend to follow the drawing.",
                "asks": "Please confirm which governs."}[f["contractor_reading"]]
        return f"Spec {f['spec_section']} calls for {p[1]} for the {p[0]}; the drawings show {p[2]}. {lead}"

    def _clause(self, f):
        return (f"General Conditions 1.2 (order of precedence): {self.PRECEDENCE[f['precedence']]}. RFI responses "
                f"within {f['response_days']} working days; requests to use products other than those specified follow "
                f"the substitution procedure in Section 01 25 00, not the RFI process.")

    def render(self, f, rng, surface):
        ds = _ds(rng)
        proj, arch, gc = f["project"]
        p = self.PAIRS[f["pair"]]
        qty = f"about {f['quantity']:,} {p[6]}" + ("" if p[6] in ("m2", "m3", "m", "ea") else "s")
        if surface == "json":
            return dumps(prune({
                "rfi": {"number": f["rfi"], "project": proj, "from": f"{f['author']}, {gc}", "to": arch,
                        "submitted": f["date"].isoformat(), "question": self._question(f), "quantity_affected": qty},
                "contract": {"precedence_and_substitution_clause": self._clause(f)},
            }))
        if surface == "rfi_form":
            return "\n".join([f"{_rfi_label(f['rfi'])} -- {proj}",
                              kv_lines([("From", f"{f['author']} ({gc})"), ("To", arch),
                                        ("Date", fmt_date(f["date"], rng, ds)),
                                        ("Spec section", f["spec_section"]), ("Quantity affected", qty)], rng),
                              "", noisy(self._question(f), rng, 0.1), "", self._clause(f)])
        prior = [[f"RFI-{rng.randint(1, 11):04d}", pick(rng, ["door hardware clarification", "slab edge dimension",
                                                               "roof drain location", "paint colour"]), "answered"],
                 [f["rfi"], noisy(self._question(f), rng, 0.08), "open"]]
        return "\n".join([f"RFI LOG (extract) -- {proj}", f"Contractor: {gc}; design team: {arch}", "",
                          table(["No.", "Subject", "Status"], prior, rng), "", f"Quantity affected: {qty}.", "",
                          self._clause(f)])

    def questions(self):
        return {
            "governing_requirement": choice("Which requirement governs the work in question?", {
                "design_team_must_decide": "Neither by rule: the design team has to decide, because the clause does not "
                                           "settle it or its answer would fall short of a safety requirement.",
                "drawing_governs": "The drawing's requirement governs.",
                "not_a_conflict_substitution": "There is no real conflict; the contractor is asking to substitute a product.",
                "specification_governs": "The specification's requirement governs.",
            }),
            "response_route": choice("How should the response be routed?", {
                "answer_from_documents": "Answer by citing the precedence clause and the governing document.",
                "design_clarification": "Have the design team issue a clarification with no change in contract sum.",
                "redirect_to_substitution_process": "Return it to be resubmitted as a substitution request.",
                "route_as_potential_change": "Have the design team decide and log the answer as a potential change.",
            }),
            "precedence_resolves": noul(
                "The contract's order-of-precedence clause settles which of the two requirements applies.",
                yes="The clause names which document governs, or one requirement is clearly the more stringent under "
                    "a more-stringent rule.",
                no="The clause refers conflicts to the architect, the two requirements cannot be ranked, or there is no "
                   "conflict between documents at all."),
            "life_safety_relevance": score("How much does the discrepancy matter for life safety or code compliance?", [
                "Not at all: appearance or preference only.",
                "Slightly: durability or performance, not safety.",
                "Moderately: structural or code performance is involved.",
                "Highly: fire separation, egress or fall protection depends on it.",
            ]),
            "cost_delta_exposure": score("How large is the cost difference between the two readings?", [
                "Negligible: under a couple of thousand.",
                "Minor: a few thousand to around fifteen thousand.",
                "Significant: tens of thousands.",
                "Major: well over seventy-five thousand.",
            ]),
        }


# =========================================================================== schema 4: existing conditions in renovation
class ExistingConditionsRfi(Schema):
    name = "existing_conditions_rfi"
    description = "An RFI raised when demolition in a renovation exposes conditions that differ from the drawings."
    surfaces = {"json": 0.5, "rfi_form": 0.25, "superintendent_email": 0.25}

    # (finding, minor variant, major variant, route, hazard, may be cleared by analysis, design change minor/major)
    FINDINGS = [
        ("existing slab under the new rooftop unit", "the slab only 140 mm thick where the drawings assume 150 mm",
         "the slab only 100 mm thick where the drawings assume 150 mm", "structural", False, True, (1, 2)),
        ("new stair opening at grid C/3", "an unrecorded steel angle lintel along one edge",
         "an unrecorded steel beam running across the opening", "structural", False, False, (1, 3)),
        ("ceiling space on level 2", "pipe insulation of a type listed in the owner's survey as non-asbestos",
         "pipe insulation that looks like the suspect material excluded from the owner's survey", "hazmat", True, False,
         (0, 0)),
        ("wall W-14 scheduled for demolition", "abandoned circuits, tested dead",
         "live circuits not on the drawings, feeding the floor above", "mep", True, False, (0, 1)),
        ("north wall joist bearings", "minor rot at two joist ends, sound timber beyond",
         "rotted joist ends at eight bearings, joists deflecting", "structural", True, False, (1, 3)),
        ("new elevator pit", "the existing footing projecting 80 mm into the pit",
         "the existing footing projecting 400 mm into the pit", "structural", False, True, (1, 2)),
        ("corridor door openings", "the existing frames 10 mm narrower than the drawings show",
         "the existing openings 50 mm narrower than the drawings show", "architect", False, False, (0, 1)),
        ("ceiling plenum in the east wing", "the plenum 30 mm shallower than the survey shows",
         "the plenum 150 mm shallower than the survey shows", "mep", False, False, (0, 2)),
        ("masonry wall to remain at the lobby", "the wall 12 mm out of plumb, within the finish tolerance",
         "the wall 60 mm out of plumb", "architect", False, False, (0, 1)),
        ("slab to be removed in the old kitchen", "two capped floor drains, as expected",
         "an abandoned buried fuel-oil tank", "hazmat", True, False, (0, 0)),
        ("exterior wall behind the plaster", "brick backup, as drawn, with some soft mortar",
         "hollow clay tile backup where the drawings show brick; the new anchors were designed for brick",
         "structural", False, True, (0, 2)),
    ]

    def sample_factors(self, rng):
        reg = regime(rng)
        fi = rng.randrange(len(self.FINDINGS))
        major = chance(rng, 0.55)
        fv = chance(rng, 0.4)
        concealed = chance(rng, 0.6)
        ordered = chance(rng, 0.45)
        critical = chance(rng, 0.45)
        crew_idle = chance(rng, 0.3)
        days_needed = pick(rng, [2, 5, 10, 20, 40])
        if reg == "clear":
            if chance(rng, 0.5):
                major, critical, crew_idle, days_needed = False, False, False, 30
            else:
                fi = pick(rng, [1, 3, 4, 9])
                major, critical, crew_idle, days_needed = True, True, True, 2
        elif reg == "conflicting":
            # a field-verify note on the drawing, but the condition was hidden until demolition; or the reverse
            fv, concealed, ordered = True, chance(rng, 0.5), True
            major = True
        return {"regime": reg, "finding": fi, "major": major, "field_verify_note": fv, "concealed": concealed,
                "ordered_to_drawing": ordered,
                "critical_path": critical, "crew_idle": crew_idle, "days_until_needed": days_needed,
                "response_days": pick(rng, [5, 7, 10]), "project": _project(rng), "rfi": _rfi_no(rng),
                "super": person(rng), "date": rand_date(rng), "level": rng.randint(1, 4)}

    def intended(self, f):
        name, minor_t, major_t, route, hazard, analysis, (dc_min, dc_maj) = self.FINDINGS[f["finding"]]
        major = f["major"]
        hazmat = route == "hazmat"
        if not major and not hazard and dc_min == 0:
            r = "contractor_field_resolution"
        else:
            r = {"structural": "structural_engineer", "mep": "mep_engineer", "hazmat": "owner_hazmat_consultant",
                 "architect": "architect"}[route]
        if hazmat and not major:
            r = "owner_hazmat_consultant" if f["finding"] == 2 else "contractor_field_resolution"
        dc = dc_maj if major else dc_min
        if r == "contractor_field_resolution":
            cost = "no_cost_impact"
        elif major and hazmat:
            cost = "owner_change"
        elif f["field_verify_note"] and not f["concealed"] and f["ordered_to_drawing"]:
            cost = "contractor_bears_field_verify"
        elif analysis and major:
            cost = "undetermined_pending_design"
        elif dc == 0 and not hazmat:
            cost = "no_cost_impact"
        else:
            cost = "owner_change"
        stop = (hazard and major) or (major and route == "structural" and dc >= 3)
        sched = 0
        if f["crew_idle"]:
            sched += 2
        if f["critical_path"]:
            sched += 1
        if f["days_until_needed"] <= _working_to_calendar(f["response_days"]):
            sched += 1
        if hazmat and major:
            sched += 1
        return {"rfi_route": r, "cost_responsibility": cost, "work_stoppage_needed": "yes" if stop else "no",
                "design_change_scope": dc, "schedule_exposure": _clamp(sched)}

    def _clause(self, f):
        return (f"Section 01 73 00 (existing conditions): existing-condition drawings come from record documents and a "
                f"limited survey. Where a dimension is marked FV, the contractor verifies it before ordering or "
                f"fabricating; conditions concealed until demolition that differ from the drawings are handled as "
                f"changes. Suspect hazardous materials: stop work in the area and notify the owner. RFI responses "
                f"within {f['response_days']} working days.")

    def _text(self, f):
        name, minor_t, major_t = self.FINDINGS[f["finding"]][:3]
        return name, (major_t if f["major"] else minor_t)

    def _status(self, f):
        s = [("the existing-conditions drawing for this area is marked FV (field verify)" if f["field_verify_note"]
              else "the existing-conditions drawing for this area carries no FV note"),
             ("the area was concealed until demolition" if f["concealed"]
              else "the area was open to view from the start of the job"),
             ("new material for this area was already ordered to the drawing" if f["ordered_to_drawing"]
              else "nothing has been ordered for this area yet")]
        sched = [("crew standing by" if f["crew_idle"] else "crew working elsewhere"),
                 ("area on the critical path" if f["critical_path"] else "area has float"),
                 f"answer needed within {f['days_until_needed']} days"]
        return "; ".join(s), "; ".join(sched)

    def render(self, f, rng, surface):
        ds = _ds(rng)
        proj, arch, gc = f["project"]
        where, what = self._text(f)
        verify, sched = self._status(f)
        if surface == "json":
            return dumps(prune({
                "rfi": {"number": f["rfi"], "project": proj, "from": f"{f['super']}, superintendent, {gc}",
                        "to": arch, "submitted": f["date"].isoformat(),
                        "finding": f"{where} (level {f['level']}): {what}",
                        "question": "Please advise how to proceed."},
                "context": {"verification": verify, "schedule": sched},
                "contract_clause": self._clause(f),
            }))
        if surface == "rfi_form":
            return "\n".join([f"{_rfi_label(f['rfi'])} -- EXISTING CONDITION", kv_lines([
                ("Project", proj), ("From", f"{f['super']} ({gc})"), ("To", arch),
                ("Date", fmt_date(f["date"], rng, ds)), ("Location", f"{where}, level {f['level']}")], rng),
                "", f"Found: {what}.", "Question: how should we proceed?", "",
                f"Verification: {verify}.", f"Schedule: {sched}.", "", self._clause(f)])
        body = [f"While demolishing at the {where} (level {f['level']}) we found {what}.",
                f"{verify[0].upper()}{verify[1:]}.", f"{sched[0].upper()}{sched[1:]}.",
                pick(rng, ["Need direction.", "How do you want us to proceed?", "Please advise asap.",
                           "Let us know if you want to walk it."])]
        return (f"Subject: {f['rfi']} {where} -- {proj}\nFrom: {f['super']} ({gc})\nTo: {arch}\n\n"
                + noisy(join_sentences(body), rng, 0.15) + "\n\n" + self._clause(f))

    def questions(self):
        return {
            "rfi_route": choice("Who should answer this RFI?", {
                "architect": "The architect.",
                "contractor_field_resolution": "No designer: the contractor can resolve it in the field within tolerances.",
                "mep_engineer": "The mechanical or electrical engineer.",
                "owner_hazmat_consultant": "The owner's hazardous-materials consultant.",
                "structural_engineer": "The structural engineer.",
            }),
            "cost_responsibility": choice("Who is likely to bear any added cost from this condition?", {
                "contractor_bears_field_verify": "The contractor, who was required to verify this before relying on it.",
                "no_cost_impact": "Nobody: the fix adds no cost worth a change.",
                "owner_change": "The owner, through a change: the condition was concealed or outside the contract.",
                "undetermined_pending_design": "Unclear until the designer checks whether the existing work is adequate.",
            }),
            "work_stoppage_needed": noul(
                "Work in the affected area must stop until the RFI is answered.",
                yes="Continuing would expose people to a hazard or remove or load something whose adequacy is unknown.",
                no="Work can continue around the condition safely while the answer is prepared."),
            "design_change_scope": score("How much design change is the condition likely to need?", [
                "None: build as drawn.",
                "A field adjustment to a detail.",
                "Redesign of one element.",
                "Redesign that reaches other trades or the approved permit drawings.",
            ]),
            "schedule_exposure": score("How exposed is the schedule while this RFI is open?", [
                "Barely: work continues elsewhere with time in hand.",
                "Somewhat: the answer is needed within the normal response period.",
                "Considerably: crews are waiting or the critical path is affected.",
                "Severely: crews idle on the critical path, or a long abatement or redesign ahead.",
            ]),
        }


# =========================================================================== schema 5: RFI quality screening
class RfiQualityScreening(Schema):
    name = "rfi_quality_screening"
    description = "An incoming RFI screened by the design team's coordinator before routing, against the RFI procedure."
    surfaces = {"json": 0.5, "rfi_form": 0.25, "coordinator_log": 0.25}

    # kind -> [(question, references, contractor's proposed solution)]
    KINDS = {
        "genuine": [
            ("Detail 5/A-512 shows the parapet coping lapping the membrane, but the roofing spec requires the membrane "
             "over the coping. Which sequence applies?", "A-512 detail 5; Spec 07 54 23",
             "Run the membrane over the coping as the spec says."),
            ("The storefront head on A-421 has no blocking shown behind the new canopy anchors. What backing is required?",
             "A-421 detail 3; Spec 08 41 13", "Add a 3 mm steel backing plate at each anchor."),
            ("Plumbing riser P-3 conflicts with the shear wall opening on S-301. Can the riser shift 300 mm north?",
             "S-301; P-201", "Shift the riser 300 mm north into the chase."),
            ("The roof drain sump on P-501 sits on a joist line shown on S-401. Can the sump move 400 mm?",
             "P-501; S-401", "Move the sump 400 mm east between joists."),
            ("Window sill height on A-301 is 750 mm, but the guard note requires 900 mm above floor. Which applies?",
             "A-301; A-001 note 7", "Add a guard rail at the affected windows."),
        ],
        "in_docs": [
            ("What fire rating is required for the corridor doors?", "Spec 08 11 13; A-601",
             "Provide 20-minute doors as listed in the door schedule."),
            ("What concrete cover is required for the footing bars?", "S-001",
             "Use 75 mm cover per general note 9."),
            ("What paint colour is required in stair 2?", "A-701 finish schedule", "Use P-4 as scheduled."),
        ],
        "duplicate": [
            ("Please confirm the slab edge dimension at grid A/1-3 (asked before as RFI-{n:04d}, answered).", "S-201",
             "Use the dimension given in the earlier response."),
            ("What is the top-of-steel elevation at the canopy? (Asked before as RFI-{n:04d}, answered.)", "S-501",
             "Follow the earlier response."),
        ],
        "substitution": [
            ("Can we use another manufacturer's acoustic ceiling tile instead of the specified one? It is cheaper and "
             "in stock.", "Spec 09 51 13", "Use the alternate tile; data sheet attached."),
            ("May we use PVC in lieu of the specified cast-iron storm piping above the ceiling?", "Spec 22 14 13; P-301",
             "Use PVC; it is lighter and faster to install."),
        ],
        "contractor_work": [
            ("Please provide coordinated dimensions for all sleeves through the level 2 slab.", "S-202; M-202; P-202",
             "Design team to issue a sleeve layout."),
            ("Please lay out the hangers for the mechanical piping in the boiler room.", "M-401",
             "Engineer to provide hanger locations."),
            ("Please tell us how to sequence the trades in the core so the drywall can start.", "A-201",
             "Architect to advise a sequence."),
        ],
    }
    ELEMENTS = ["question", "references", "proposal", "impact", "needed_by"]

    def sample_factors(self, rng):
        reg = regime(rng)
        kind = pick_w(rng, {"genuine": 4, "in_docs": 2, "duplicate": 1, "substitution": 1, "contractor_work": 2})
        requires_proposal = chance(rng, 0.6)
        refs, proposal, impact = chance(rng, 0.75), chance(rng, 0.55), chance(rng, 0.7)
        needed_by = pick_w(rng, {"date": 8, "asap": 2})
        sub_date = rand_date(rng)
        response_days = pick(rng, [5, 7, 10])
        lead = pick(rng, [3, 6, 9, 12, 16, 25])
        volume_ratio = round(pick_w(rng, {between(rng, 0.6, 1.3): 5, between(rng, 1.4, 2.2): 3,
                                          between(rng, 2.5, 4.0): 2}), 1)
        returned_share = pick(rng, [5, 10, 20, 35, 50])
        reserve = chance(rng, 0.3)
        delay_claim_text = chance(rng, 0.2)
        if reg == "clear":
            if chance(rng, 0.5):
                kind, refs, proposal, impact, needed_by, lead = "genuine", True, True, True, "date", 25
                volume_ratio, returned_share, reserve, delay_claim_text = 0.9, 5, False, False
            else:
                refs, proposal, needed_by = False, False, "asap"
                volume_ratio, returned_share, reserve = 3.1, 50, True
        elif reg == "borderline":
            # one required element missing, or a needed-by date just inside / outside the response period
            lead = _working_to_calendar(response_days) + pick(rng, [-2, -1, 0, 1, 2])
            refs, proposal = True, pick(rng, [True, False])
        else:  # conflicting: a tidy, complete form wrapped around a question that is not a design question
            kind = pick(rng, ["duplicate", "substitution", "contractor_work", "in_docs"])
            refs, proposal, impact, needed_by = True, True, True, "date"
        return {"regime": reg, "kind": kind, "variant": rng.randrange(len(self.KINDS[kind])),
                "dup_no": rng.randint(20, 140),
                "requires_proposal": requires_proposal, "has_refs": refs, "has_proposal": proposal,
                "has_impact": impact, "needed_by": needed_by, "lead_days": lead, "response_days": response_days,
                "volume_ratio": volume_ratio, "returned_share_pct": returned_share, "reserve_rights": reserve,
                "delay_claim_text": delay_claim_text, "project": _project(rng), "rfi": _rfi_no(rng),
                "author": person(rng), "date": sub_date}

    CHECK = {"genuine": "the documents do not settle this; it needs a design answer",
             "in_docs": {0: "Spec 08 11 13 and the door schedule on A-601 list 20-minute corridor doors",
                         1: "general note 9 on S-001 gives 75 mm cover for footing bars",
                         2: "the finish schedule on A-701 lists P-4 for stair 2"},
             "duplicate": "RFI-{n:04d} answered the same question",
             "substitution": "the named product is specified with no 'or equal' clause",
             "contractor_work": "no design information is missing; the drawings show the elements involved"}

    def _check(self, f):
        c = self.CHECK[f["kind"]]
        c = c[f["variant"]] if isinstance(c, dict) else c
        return c.format(n=f["dup_no"])

    def _complete(self, f):
        ok = f["has_refs"] and f["needed_by"] == "date"
        if f["requires_proposal"]:
            ok = ok and f["has_proposal"]
        return ok

    def intended(self, f):
        k = f["kind"]
        if k == "duplicate" or k == "in_docs":
            act = "close_citing_existing_answer"
        elif k == "substitution":
            act = "redirect_to_substitution_request"
        elif k == "contractor_work":
            act = "return_as_contractor_responsibility"
        elif not self._complete(f):
            act = "return_incomplete"
        else:
            act = "route_for_design_response"
        n = sum([f["has_refs"], f["has_proposal"], f["has_impact"], f["needed_by"] == "date"])
        comp = 0 if n <= 1 else 1 if n == 2 else 2 if n == 3 else 3
        if comp == 3 and k != "genuine":
            comp = 2
        sig = 0
        sig += 0 if f["volume_ratio"] < 1.4 else 1 if f["volume_ratio"] < 2.5 else 2
        sig += 1 if f["returned_share_pct"] >= 35 else 0
        sig += 1 if (f["reserve_rights"] or f["delay_claim_text"]) else 0
        reasonable = f["needed_by"] == "date" and f["lead_days"] >= _working_to_calendar(f["response_days"])
        return {"screening_action": act, "rfi_completeness": comp, "claim_building_signal": _clamp(sig),
                "response_clock_started": "yes" if self._complete(f) else "no",
                "requested_date_reasonable": "yes" if reasonable else "no"}

    def _question(self, f):
        return self.KINDS[f["kind"]][f["variant"]][0].format(n=f["dup_no"])

    def _procedure(self, f):
        req = "the question, drawing/spec references, the contractor's proposed solution, and the date a response is " \
              "needed" if f["requires_proposal"] else "the question, drawing/spec references and the date a " \
                                                      "response is needed (a proposed solution is encouraged)"
        return (f"RFI procedure (01 26 00): each RFI must state {req}. Incomplete RFIs are returned and the "
                f"{f['response_days']}-working-day response period starts only when a complete RFI is received. "
                f"Product substitutions go through Section 01 25 00. Coordination of trades and field dimensions is the "
                f"contractor's work.")

    def _fields(self, f, rng, ds):
        need = (fmt_date(f["date"] + timedelta(days=f["lead_days"]), rng, ds) if f["needed_by"] == "date"
                else pick(rng, ["ASAP", "asap", "Immediately"]))
        entry = self.KINDS[f["kind"]][f["variant"]]
        refs = entry[1] if f["has_refs"] else None
        prop = entry[2] if f["has_proposal"] else None
        imp = pick_static(f, ["No cost impact.", "Possible cost impact.", "Schedule impact if not answered by the date "
                                                                            "requested."]) if f["has_impact"] else None
        return need, refs, prop, imp

    def _log(self, f):
        s = (f"This contractor has submitted {f['volume_ratio']}x the project's average monthly RFI count; "
             f"{f['returned_share_pct']}% of its RFIs so far were returned as answered by the documents.")
        if f["reserve_rights"]:
            s += " Its RFIs now close with 'contractor reserves all rights to cost and time'."
        if f["delay_claim_text"]:
            s += " Its last schedule narrative attributes 23 days of delay to 'late RFI responses'."
        return s

    def render(self, f, rng, surface):
        ds = _ds(rng)
        proj, arch, gc = f["project"]
        need, refs, prop, imp = self._fields(f, rng, ds)
        if surface == "json":
            return dumps(prune({
                "rfi": {"number": f["rfi"], "project": proj, "from": f"{f['author']}, {gc}", "received":
                        f["date"].isoformat(), "question": self._question(f), "references": refs,
                        "proposed_solution": prop, "impact_statement": imp, "response_needed_by": need},
                "contractor_rfi_history": self._log(f), "coordinator_check": self._check(f),
                "procedure": self._procedure(f),
            }))
        if surface == "rfi_form":
            return "\n".join([f"{_rfi_label(f['rfi'])} -- {proj}", kv_lines([
                ("From", f"{f['author']} ({gc})"), ("To", arch), ("Received", fmt_date(f["date"], rng, ds)),
                ("Response needed by", need), ("References", refs or "--"), ("Impact", imp or "--")], rng),
                "", "Question: " + self._question(f), "Proposed solution: " + (prop or "--"), "",
                "Coordinator's check: " + self._check(f) + ".", "History: " + self._log(f), "", self._procedure(f)])
        rows = [["Received", fmt_date(f["date"], rng, ds)], ["From", f"{f['author']} ({gc})"],
                ["Question", self._question(f)], ["Refs", refs or "none"], ["Proposal", prop or "none"],
                ["Impact", imp or "not stated"], ["Needed by", need], ["Check", self._check(f)]]
        return "\n".join([f"RFI SCREENING LOG -- {proj} ({arch})", f"Entry {f['rfi']}", "",
                          table(["Field", "Entry"], rows, rng), "", "History: " + self._log(f), "",
                          self._procedure(f)])

    def questions(self):
        return {
            "screening_action": choice("What should the coordinator do with this RFI?", {
                "close_citing_existing_answer": "Close it by pointing to the earlier response or the document that "
                                                "already answers it.",
                "redirect_to_substitution_request": "Return it to be resubmitted as a substitution request.",
                "return_as_contractor_responsibility": "Return it: the work asked for is the contractor's own.",
                "return_incomplete": "Return it as incomplete under the procedure.",
                "route_for_design_response": "Send it to the design team for a response.",
            }),
            "rfi_completeness": score("How complete and usable is the RFI as submitted?", [
                "Poor: most of the required content is missing.",
                "Patchy: a key element such as references or a date is missing.",
                "Adequate: the required content is there, if thin.",
                "Complete: clear question, references, proposal, impact and a real date.",
            ]),
            "claim_building_signal": score("How strongly does this contractor's RFI pattern suggest it is building a "
                                           "record for a claim?", [
                "Not at all: ordinary volume and quality.",
                "Weakly: one unusual feature.",
                "Moderately: several features point that way.",
                "Strongly: high volume, many needless RFIs and explicit reservation of rights.",
            ]),
            "response_clock_started": noul(
                "The design team's contractual response period for this RFI is running.",
                yes="The RFI contains every element the quoted procedure requires, so the period runs from receipt.",
                no="A required element is missing (for example a real needed-by date), so the period has not started."),
            "requested_date_reasonable": noul(
                "The date by which the contractor wants an answer leaves the design team its full response period."),
        }


WORKFLOW = Workflow(
    name="construction_rfi_triage",
    title="Construction RFI triage",
    description="Requests for information on construction projects: whether the documents already answer them, "
                "design-change implications, urgency, and which party responds.",
    domain="construction / document control",
    schemas=[DrawingConflictRfi(), MepCoordinationClash(), SpecDrawingDiscrepancy(), ExistingConditionsRfi(),
             RfiQualityScreening()],
    notes="Fictional projects, firms and drawing references; the governing RFI procedure, coordination or precedence "
          "clause is quoted in every state. No change-order pricing, no municipal plan review, no invoices or payment.",
)
