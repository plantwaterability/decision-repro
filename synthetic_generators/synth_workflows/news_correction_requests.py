"""News correction requests: how a (fictional) local news outlet's standards desk rules on requests to correct,
clarify, re-attribute or remove published material, against the outlet's written corrections policy.

Disputed facts in news stories, complaints about reviews and opinion columns, misquotation and off-the-record
disputes, requests to unpublish or anonymise archive stories, and errors in breaking stories and push alerts.
The reader's or subject's message is EVIDENCE inside an editorial-standards case: no question asks how to reply
to anyone, how satisfied they are or whether a person should take over; every question is a standards decision
(correction type, placement, print correction, quote ruling, archive treatment, distribution of a fix). The policy
extract that decides each case is quoted in the state.

Outlets, places, businesses, people, productions and every policy are invented. Beats are deliberately neutral
(council budgets, transport, schools, sport, local business, reviews, courts in plain terms); no partisan politics
and no real person, outlet or work appears, and no protected attribute decides anything.

Each schema draws a `regime` first: 'clear' cases have aligned evidence, 'borderline' cases sit on a policy line
(a figure off by about 5%, a story just past the anonymisation age, a reader report that is not yet
confirmation), and 'conflicting' cases pair evidence that points both ways (official documents against the
reporter's notes, an off-record claim raised only after publication). Every schema defines `intended(f)`, the
answer the quoted policy gives; it documents the logic and drives the spread check.
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
    ident,
    join_sentences,
    kv_lines,
    money,
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
)

# =========================================================================== shared pools and helpers
PLACES = ["Marrowgate", "Kelderby", "Ostwick", "Fennick Bay", "Harlow Fen", "Brackwater", "Lantern Bay", "Tarrow Vale",
          "Innisvale", "Skelmar", "Westermouth", "Caldmouth", "Lunsford", "Ashlin", "Corriston", "Pellbridge"]
MASTHEADS = ["Courier", "Ledger", "Sentinel", "Gazette", "Examiner", "Record", "Observer", "Herald", "Chronicle",
             "Mercury", "Advertiser", "Dispatch"]
STREETS = ["Mill Lane", "Station Road", "Elm Street", "Quarry Hill", "Harbour Road", "Church Walk", "Fernside Avenue",
           "Bridge Street", "Orchard Way", "Kiln Road", "Market Row", "Castle Terrace", "Tannery Lane", "Wharf Street"]
BIZ_A = ["Saffron", "Ember", "Juniper", "Copper Pot", "Harbourside", "Old Mill", "Fig Tree", "Lantern", "Bluebell",
         "Corner House", "Salt & Smoke", "Greenhouse", "Ropewalk", "Larder"]
BIZ_B = ["Kitchen", "Bistro", "Grill", "Noodle Bar", "Dining Room", "Trattoria", "Cafe", "Bakery", "Deli", "Tap Room"]


def _outlet(rng):
    return f"The {pick(rng, PLACES)} {pick(rng, MASTHEADS)}"


def _biz(rng):
    return f"{pick(rng, BIZ_A)} {pick(rng, BIZ_B)}"


def _kv(pairs, rng):
    return kv_lines(pairs, rng, style=pick(rng, ["colon", "colon", "aligned", "dash"]))


def _policy(rng, ref, clauses, json_ok=False):
    """The policy extract as prose, numbered clauses or bullets; on the json surface a clause list or a string."""
    style = pick(rng, ["prose", "numbered", "bullets", "prose"])
    if json_ok:
        return list(clauses) if style in ("numbered", "bullets") else f"{ref}: " + " ".join(clauses)
    if style == "numbered":
        return f"{ref}\n" + "\n".join(f"{i}. {c}" for i, c in enumerate(clauses, 1))
    if style == "bullets":
        return f"{ref}:\n" + bullets(clauses, rng)
    return f"{ref}: " + " ".join(clauses)


def _loc(location):
    return location if location.startswith("paragraph") else f"the {location}"


def _ago(days):
    if days == 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 14:
        return f"{days} days ago"
    if days < 60:
        return f"{days // 7} weeks ago"
    return f"{days // 30} months ago"


# =========================================================================== schema 1
# figure facts: (beat, headline with {v}, what is disputed, value style, range)
FIGURES = [
    ("council", "Council approves {v} for road repairs", "the road-repair budget", "money_m", (1.2, 9.5)),
    ("planning", "Plan for {v} homes on former mill site goes to committee", "the number of homes in the plan", "int", (40, 600)),
    ("courts", "Landlord fined {v} over unsafe wiring in flats", "the fine imposed", "money", (400, 9000)),
    ("arts", "Harbour festival draws {v} visitors", "festival attendance", "int", (900, 25000)),
    ("business", "Packaging plant to cut {v} jobs", "the number of job losses", "int", (12, 420)),
    ("environment", "River clean-up removes {v} tonnes of waste", "the tonnage removed", "int", (3, 80)),
    ("health", "Physiotherapy waits reach {v} weeks", "the waiting time", "int", (4, 22)),
    ("fire", "Warehouse fire causes {v} in damage", "the damage estimate", "money", (20000, 900000)),
    ("weather", "Storm leaves {v} homes without power", "the number of homes without power", "int", (300, 9000)),
    ("transport", "Town-centre parking charges to rise {v}", "the size of the parking-charge rise", "pct", (5, 40)),
    ("schools", "District to spend {v} on school roofs", "the school-roof budget", "money_m", (0.6, 4.5)),
    ("sport", "Leisure centre membership passes {v}", "the membership figure", "int", (800, 6000)),
]
DATES = [
    ("transport", "Bridge Street bridge to close from {v}", "the start date of the bridge closure"),
    ("community", "Library reopens on {v} after refit", "the library's reopening date"),
    ("transport", "New bus timetable starts {v}", "the start date of the new timetable"),
    ("planning", "Public hearing on depot plan set for {v}", "the date of the public hearing"),
    ("business", "Bakery opens second branch on {v}", "the opening date"),
    ("schools", "Pupils return {v} after boiler repairs", "the date pupils return"),
]
ROLES = [("chair of the allotment society", "community", "{n} steps down after 12 years as allotment chair"),
         ("head teacher at {p} Primary", "schools", "{p} Primary head praises pupils after inspection"),
         ("owner of {b}", "business", "{b} marks ten years on {s}"),
         ("coach of {p} Rovers under-14s", "sport", "{p} Rovers under-14s reach county final"),
         ("director of the {p} food bank", "community", "Food bank opens second site in {p}"),
         ("site manager at the {p} recycling centre", "environment", "Recycling centre extends weekend hours"),
         ("organiser of the {p} harbour festival", "arts", "Harbour festival confirms new date")]
TITLES = [("chief executive", "chief operating officer"), ("head teacher", "deputy head teacher"),
          ("chair", "vice-chair"), ("director", "interim director"), ("senior planning officer", "planning officer"),
          ("founder", "co-owner")]
CONTEXTS = [
    ("crime", "Burglaries up 50% in {p}", "reported a 50% rise in burglaries",
     "the rise was from 4 to 6 reported cases over the year"),
    ("council", "Council's arts spending doubles", "said arts spending had doubled",
     "half of this year's total is a one-off grant that will not recur"),
    ("transport", "Bus fares to rise by a third", "said fares would rise by a third",
     "the rise applies only to the peak single fare; day tickets are unchanged"),
    ("health", "Flu jabs cut staff sick days by 40%, local study finds", "presented a 40% cut as the study's finding",
     "the study was small and its authors describe the result as preliminary"),
    ("housing", "Rents climb 18% in a year", "said rents climbed 18%",
     "the figure covers new listings only, not existing tenancies"),
    ("schools", "Half of pupils miss reading target", "said half of pupils missed the target",
     "the figure is for one year group at two schools, not the whole district"),
]
TYPOS = [("Wednesday", "Wedensday"), ("accommodation", "accomodation"), ("separate", "seperate"),
         ("committee", "comittee"), ("library", "libary"), ("February", "Febuary"), ("necessary", "neccessary"),
         ("occasion", "occassion")]
NAMED_KINDS = ("name_spelling", "wrong_person", "title", "attribution")


def _fmt_val(style, v):
    if style == "money_m":
        return f"${v:.1f} million"
    if style == "money":
        return money(int(round(v, -2 if v > 5000 else -1)), style="sym")
    if style == "pct":
        return f"{int(round(v))}%"
    return f"{int(round(v)):,}"


def _misspell(name):
    first, last = name.split(" ", 1)
    if len(last) > 4:
        i = len(last) // 2
        last = last[:i] + last[i + 1:] if last[i] != last[i - 1] else last[:i] + last[i] + last[i:]
    return f"{first} {last}"


class FactualErrorClaim(Schema):
    name = "factual_error_claim"
    description = ("A reader or the subject of a story says a published fact is wrong: correct, clarify, fix "
                   "silently, let it stand or hold it, under the outlet's corrections policy.")
    surfaces = {"json": 0.5, "complaint_email": 0.25, "standards_ticket": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        kind = pick_w(rng, {"figure": 3, "date": 1.5, "name_spelling": 1.5, "wrong_person": 1, "title": 1,
                            "attribution": 1, "typo": 1, "context": 2})
        place = pick(rng, PLACES)
        d = {"kind": kind, "magnitude": None, "reputational": False}
        if kind == "figure":
            beat, head, what, style, (lo, hi) = pick(rng, FIGURES)
            correct = between(rng, lo, hi)
            mag = (near(rng, 0.05, 0.02, lo=0.005) if reg == "borderline" and chance(rng, 0.6)
                   else pick(rng, [0.02, 0.15, 0.3, 0.45, 0.6]) if reg == "clear" else between(rng, 0.03, 0.5))
            published = correct * (1 + mag * pick(rng, [-1, 1]))
            d.update(beat=beat, headline=head.format(v=_fmt_val(style, published)), what=what,
                     published=_fmt_val(style, published), claimed=_fmt_val(style, correct), magnitude=round(mag, 3))
        elif kind == "date":
            beat, head, what = pick(rng, DATES)
            base = rand_ts(rng).date()
            wrong = base + timedelta(days=pick(rng, [-7, -1, 1, 2, 7, 14]))
            d.update(beat=beat, headline=head.format(v=fmt_date(wrong, style="long")), what=what,
                     published=fmt_date(wrong, style="long"), claimed=fmt_date(base, style="long"))
        elif kind == "context":
            beat, head, what, ctx = pick(rng, CONTEXTS)
            d.update(beat=beat, headline=head.format(p=place), what=what, published=what, claimed=ctx)
        elif kind == "typo":
            word, typo = pick(rng, TYPOS)
            beat = pick(rng, ["council", "community", "transport", "schools"])
            d.update(beat=beat, headline=pick(rng, ["Residents back plan for new crossing on {s}",
                                                    "Community hall refit finished ahead of schedule",
                                                    "Allotment waiting list closes for the year"]).format(s=pick(rng, STREETS)),
                     what="the spelling of a word", published=typo, claimed=word)
        else:
            role, beat, head = pick(rng, ROLES)
            n1, n2 = person(rng), person(rng)
            role = role.format(p=place, b=_biz(rng))
            headline = head.format(n=n1, p=place, b=_biz(rng), s=pick(rng, STREETS))
            if kind == "name_spelling":
                d.update(published=_misspell(n1), claimed=n1, what=f"the spelling of the {role}'s name")
            elif kind == "wrong_person":
                beat, headline = pick(rng, [(beat, headline), ("courts", f"Two fined after fight outside {place} pub"),
                                            ("fire", f"Chip-pan fire spreads to neighbouring flat on {pick(rng, STREETS)}")])
                d.update(published=n2, claimed=n1, what=f"who the {role} is" if beat not in ("courts", "fire")
                         else "the name of a person named in the story", reputational=beat in ("courts", "fire"))
            elif kind == "title":
                t1, t2 = pick(rng, TITLES)
                d.update(published=f"{n1}, {t1}", claimed=f"{n1}, {t2}", what=f"{n1}'s job title")
            else:
                d.update(published=f"a statement attributed to {n1}", claimed=f"the statement was issued by {n2}",
                         what="who made a statement quoted in the story", reputational=chance(rng, 0.5))
            d.update(beat=beat, headline=headline)
        # where the disputed point sits in the story
        loc = pick_w(rng, {"headline": 3, "first paragraph": 4, "paragraph 4": 2, "paragraph 9": 1.5,
                           "photo caption": 1, "fact box": 1})
        if kind in ("typo",):
            loc = pick(rng, ["paragraph 3", "paragraph 7", "photo caption"])
        if kind == "context" and loc in ("photo caption", "fact box"):
            loc = "first paragraph"
        if loc == "headline" and kind not in ("figure", "date", "context"):
            loc = "first paragraph"
        cent = {"headline": 3, "first paragraph": 2, "paragraph 4": 1, "paragraph 9": 0, "photo caption": 0,
                "fact box": 1}.get(loc, 0)
        # evidence, steered by whether the complainant is right and by regime
        right = chance(rng, 0.68) if kind not in ("typo",) else True
        if reg == "clear":
            doc = pick(rng, ["official", "official", "secondary"]) if right else pick(rng, [None, "screenshot"])
            notes = pick(rng, ["support_complainant", "unavailable"]) if right else "support_published"
            check = "confirms_complainant" if right else "confirms_published"
        elif reg == "borderline":
            doc = pick(rng, ["secondary", None, "screenshot", "official"]) if right else pick(rng, [None, "screenshot", "secondary"])
            notes = pick(rng, ["unavailable", "ambiguous", "support_complainant" if right else "support_published"])
            check = pick(rng, [None, "inconclusive", "confirms_complainant" if right else "confirms_published"])
        else:
            doc = "official" if right else pick(rng, ["secondary", "official"])
            notes = "support_published" if right else pick(rng, ["support_complainant", "ambiguous"])
            check = pick(rng, ["inconclusive", None, "confirms_complainant" if right else "confirms_published"])
        if kind == "typo":
            doc, notes, check = None, "unavailable", "confirms_complainant"
        days = pick(rng, [0, 1, 2, 4, 9, 20, 45])
        return {"regime": reg, "outlet": _outlet(rng), "place": place, **d, "location": loc, "centrality": cent,
                "complainant_right": right, "doc": doc, "notes": notes, "check": check,
                "in_print": chance(rng, 0.72), "days_since": days,
                "complainant_role": pick(rng, ["the person named", "a reader", "a spokesperson for the organisation",
                                               "a source in the story"] if kind in NAMED_KINDS
                                         else ["a reader", "a spokesperson for the organisation", "a reader",
                                               "a local councillor's office"]),
                "complainant": person(rng), "reporter": person(rng), "editor": person(rng),
                "tone": pick(rng, ["polite", "polite", "angry", "brief"]), "ticket": ident(rng, "STD", 5, "-"),
                "received": rand_ts(rng, hours=(7, 22))}

    # ------------------------------------------------------------------ the policy's logic
    @staticmethod
    def _support(f):
        pts = {"official": 2, "secondary": 1, "screenshot": 0, None: 0}[f["doc"]]
        pts += {"support_complainant": 1, "support_published": -1, "unavailable": 0, "ambiguous": 0}[f["notes"]]
        pts += {"confirms_complainant": 2, "confirms_published": -2, "inconclusive": 0, None: 0}[f["check"]]
        return pts

    def _outcome(self, f):
        sup = self._support(f)
        if f["kind"] == "typo":
            return "fix_without_note"
        if f["kind"] == "figure" and f["magnitude"] < 0.05:
            return "story_stands"
        if sup >= 2:
            return "clarification" if f["kind"] == "context" else "full_correction"
        if sup <= -1:
            return "story_stands"
        return "hold_pending_verification" if f["centrality"] >= 2 else "story_stands"

    def intended(self, f):
        out = self._outcome(f)
        sup = self._support(f)
        sig = min(3, f["centrality"] + (1 if f["reputational"] else 0))
        cs = 0 if sup <= -1 else 1 if sup == 0 else 2 if sup <= 2 else 3
        prt = (f["in_print"] and out in ("full_correction", "clarification")
               and (f["centrality"] >= 2 or f["kind"] in NAMED_KINDS))
        if out not in ("full_correction", "clarification"):
            place = "no_note"
        else:
            place = "top_of_story" if (f["centrality"] >= 2 or f["reputational"]) else "foot_of_story"
        return {"correction_type": out, "error_significance": sig, "claim_support": cs,
                "print_correction_required": "yes" if prt else "no", "note_placement": place}

    RULES = [
        "We correct factual errors with a note saying what was wrong and what is right; facts are never changed "
        "silently.",
        "Typing errors that do not change the meaning may be fixed without a note; errors in names, titles, figures "
        "and dates always get a note.",
        "A figure within 5% of the source's figure is a rounding difference, not an error.",
        "Where the facts were accurate but the wording or a missing comparison could mislead, add a clarification.",
        "Rule only on documents, the reporter's notes or the desk's own check. If these do not settle a point in the "
        "headline or a main point, hold any change pending verification; if the point is minor and unsettled, the "
        "story stands.",
        "Notes go at the top of the online story when the error is in the headline or the main point, or could harm "
        "a named person's reputation; otherwise at the foot.",
        "Items that appeared in print also get the correction or clarification in the next print edition when the "
        "point is in the headline or a main point or concerns a named person.",
    ]

    def _doc_text(self, f):
        return {"official": pick_doc(f, "official"), "secondary": pick_doc(f, "secondary"),
                "screenshot": "a screenshot of a social-media post supporting their version", None: None}[f["doc"]]

    def _notes_text(self, f):
        return {"support_complainant": "reporter's notes match the complainant's version",
                "support_published": "reporter's notes match what was published",
                "unavailable": "reporter's notes not available (reporter on leave)",
                "ambiguous": "reporter's notes are ambiguous on this point"}[f["notes"]]

    def _check_text(self, f):
        return {"confirms_complainant": "desk check against the primary source supports the complainant",
                "confirms_published": "desk check against the primary source supports the story as published",
                "inconclusive": "desk check inconclusive: primary source not yet obtained",
                None: None}[f["check"]]

    def _message(self, f):
        k = f["kind"]
        if k == "context":
            body = (f"Your story {f['what']}, which is technically true but misleading: {f['claimed']}. Readers will "
                    f"come away with the wrong impression.")
        elif k == "typo":
            body = f"Small thing, but '{f['claimed']}' is spelt '{f['published']}' in {_loc(f['location'])}."
        elif k in ("figure", "date"):
            body = f"Your story gives {f['what']} as {f['published']}. It is {f['claimed']}."
        elif k == "name_spelling":
            body = f"The story spells the name as {f['published']}; the correct spelling is {f['claimed']}."
        elif k == "wrong_person":
            body = f"The story names {f['published']} in connection with {f['what']}. That is wrong; it should be {f['claimed']}."
        elif k == "title":
            body = f"The story describes {f['published']}. The correct description is {f['claimed']}."
        else:
            body = f"The story includes {f['published']}. In fact {f['claimed']}."
        lead = {"polite": "I hope you can help with something in a recent article.",
                "angry": "I am frankly astonished this got past your editors.",
                "brief": ""}[f["tone"]]
        return join_sentences([lead, body])

    def render(self, f, rng, surface):
        story = {"headline": f["headline"], "beat": f["beat"], "published": _ago(f["days_since"]),
                 "also_in_print": f["in_print"], "disputed_text_location": f["location"]}
        evidence = {"supplied_by_complainant": self._doc_text(f), "reporter_notes": self._notes_text(f),
                    "desk_check": self._check_text(f) or "not yet done"}
        if surface == "json":
            return dumps(prune({
                "request": {"ticket": f["ticket"], "from": f"{f['complainant']} ({f['complainant_role']})",
                            "message": noisy(self._message(f), rng, 0.15), "received": f["received"].strftime("%Y-%m-%dT%H:%M")},
                "story": {**story, "outlet": f["outlet"], "reporter": f["reporter"]},
                "evidence": evidence,
                "policy": _policy(rng, "Corrections policy", self.RULES, True),
            }))
        if surface == "complaint_email":
            return "\n".join([
                f"Subject: Error in \"{f['headline']}\"", f"From: {f['complainant']}", f"To: corrections@{f['outlet'].split()[1].lower()}{f['outlet'].split()[2].lower()}.example",
                "", noisy(self._message(f), rng, 0.2), "", f"{f['complainant']}", "",
                "--- standards desk notes ---",
                _kv([("Complainant", f["complainant_role"]), ("Story", f"{f['headline']} ({_ago(f['days_since'])}, "
                                                                       f"{'print and online' if f['in_print'] else 'online only'})"),
                     ("Disputed text", f["location"]), ("Supplied", self._doc_text(f) or "nothing"),
                     ("Reporter's notes", self._notes_text(f)), ("Desk check", self._check_text(f) or "not yet done")], rng),
                "", _policy(rng, "Corrections policy", self.RULES)])
        rows = _kv([("Ticket", f["ticket"]), ("Outlet", f["outlet"]), ("Story", f["headline"]),
                    ("Published", f"{_ago(f['days_since'])}; " + ("print and online" if f["in_print"] else "online only")),
                    ("Reporter", f["reporter"]), ("Complainant", f"{f['complainant']}, {f['complainant_role']}"),
                    ("Disputed text", f["location"]), ("Complaint", self._message(f)),
                    ("Evidence supplied", self._doc_text(f) or "none"), ("Reporter's notes", self._notes_text(f)),
                    ("Desk check", self._check_text(f) or "not yet done"), ("Assigned to", f["editor"])], rng)
        return f"STANDARDS DESK TICKET\n{rows}\n\n{_policy(rng, 'Corrections policy', self.RULES)}"

    def questions(self):
        return {
            "correction_type": choice("How should the standards desk deal with this request?", {
                "full_correction": "Change the story and publish a correction note saying what was wrong and what is right.",
                "clarification": "Keep the facts but add a clarification that removes the misleading impression.",
                "fix_without_note": "Fix the text quietly; the policy does not require a note for this kind of slip.",
                "story_stands": "Make no change; the story is accurate or the point does not warrant one.",
                "hold_pending_verification": "Change nothing yet and verify the point before ruling.",
            }),
            "error_significance": score("How much does the disputed point matter to the story?", [
                "Incidental: a detail few readers would notice.",
                "A supporting detail.",
                "One of the story's main points.",
                "Central: the headline or core claim, or a named person's reputation.",
            ]),
            "claim_support": score("How strongly does the evidence back the complainant's version?", [
                "It points the other way.",
                "Not at all yet: nothing on file settles it.",
                "Partly: some evidence supports it.",
                "Strongly: documents or the desk's own check confirm it.",
            ]),
            "print_correction_required": noul(
                "The policy requires a correction in the print edition as well as online.",
                yes="The item ran in print, a correction or clarification is due, and the point is in the headline or a main point or concerns a named person.",
                no="The item was online only, no correction or clarification is due, or the point is minor and concerns no named person."),
            "note_placement": choice("Where, if anywhere, should a note about this go in the online story?", {
                "top_of_story": "At the top of the article, above the text.",
                "foot_of_story": "At the foot of the article.",
                "no_note": "Nowhere: the ruling calls for no published note.",
            }),
        }


def pick_doc(f, kind):
    """Deterministic wording of the complainant's supporting document for a factual claim (no rng: the
    wording follows the fact type)."""
    k = f["kind"]
    if kind == "official":
        return {"figure": "the published council or court record showing their figure",
                "date": "the official notice giving their date", "context": "the source report with the full figures",
                "name_spelling": "a copy of the organisation's register showing the spelling",
                "wrong_person": "the official record naming the person concerned",
                "title": "the organisation's published staff list",
                "attribution": "the statement as issued, on the issuing body's letterhead"}.get(k, "an official record")
    return {"figure": "a press release from the organisation", "date": "an email from the organiser",
            "context": "a summary prepared by the organisation", "name_spelling": "an email signature",
            "wrong_person": "a letter from the person's solicitor", "title": "an email from the organisation's office",
            "attribution": "an email from the communications office"}.get(k, "an email from the organisation")


# =========================================================================== schema 2
# piece: (label, subject builder, statements by kind: [(text, severity 0-3)])
PIECES = {
    "restaurant": ("restaurant review", lambda rng: _biz(rng), {
        "checkable_fact": [("{s} failed its most recent food-hygiene inspection", 3),
                           ("the tasting menu at {s} costs $95 a head", 1),
                           ("{s} has no step-free entrance", 2), ("{s} opened only in March", 0)],
        "pure_opinion": [("the risotto at {s} was bland and joyless", 1), ("the dining room feels tired", 1),
                         ("it is the least memorable meal I have eaten this year", 2)],
        "opinion_on_stated_facts": [("at $38 for a main course, {s} is overpriced", 1),
                                    ("with two staff on a full Saturday, service was painfully slow", 1)],
        "implied_accusation": [("one suspects the kitchen at {s} cuts corners when nobody is watching", 3),
                               ("{s} seems more interested in fleecing visitors than feeding locals", 3)]}),
    "theatre": ("theatre review", lambda rng: f"the {pick(rng, PLACES)} Players' new production", {
        "checkable_fact": [("{s} runs over three hours with no interval", 1),
                           ("{s} was cut from a two-week run to one week after poor sales", 2)],
        "pure_opinion": [("the second act of {s} drags", 1), ("{s} is muddled and overlong", 1)],
        "opinion_on_stated_facts": [("with ticket prices at $45, {s} asks a lot of its audience", 1)],
        "implied_accusation": [("the casting of {s} reads like a list of favours to friends", 2)]}),
    "transport_column": ("opinion column", lambda rng: f"the {pick(rng, ['number 12', 'number 4', 'X7', 'number 31'])} bus service", {
        "checkable_fact": [("{s} has lost a third of its riders since 2023", 1),
                           ("{s} now runs only once an hour after 6pm", 1)],
        "pure_opinion": [("{s} is a shambles", 1), ("the new timetable for {s} is a muddle", 0)],
        "opinion_on_stated_facts": [("with buses an hour apart in the evening, {s} is useless for shift workers", 1)],
        "implied_accusation": [("the operator of {s} is quietly running it down so it can be withdrawn", 2)]}),
    "music": ("gig review", lambda rng: f"the band {pick(rng, ['Paper Lanterns', 'The Tidelines', 'North Window', 'Copper Field', 'The Late Ferries'])}", {
        "checkable_fact": [("{s} played for barely 40 minutes", 1), ("{s} cancelled their two previous shows", 1)],
        "pure_opinion": [("{s} sounded tired and under-rehearsed", 1), ("the new songs from {s} are forgettable", 1)],
        "opinion_on_stated_facts": [("for a $30 ticket, a 40-minute set from {s} was poor value", 1)],
        "implied_accusation": [("{s} were plainly too drunk to finish the set", 3)]}),
    "parks_column": ("opinion column", lambda rng: f"the redesign of {pick(rng, PLACES)} Park", {
        "checkable_fact": [("{s} removed 60 mature trees", 2), ("{s} went 40% over budget", 2)],
        "pure_opinion": [("{s} has produced a soulless square of paving", 1), ("{s} is a missed opportunity", 0)],
        "opinion_on_stated_facts": [("having gone 40% over budget, {s} is a poor use of public money", 1)],
        "implied_accusation": [("{s} looks like a favour to the contractor", 3)]}),
}
INTERESTS = {
    "restaurant": "the reviewer's sister co-owns a rival restaurant on the same street",
    "theatre": "the critic was paid to advise a competing company this season",
    "transport_column": "the columnist sits on the board of a community-transport group bidding to replace the route",
    "music": "the reviewer manages a band that was dropped from the same bill",
    "parks_column": "the columnist's firm lost the tender for the redesign",
}


class OpinionPieceDispute(Schema):
    name = "opinion_piece_dispute"
    description = ("A complaint about a review or opinion column: correct a factual premise, offer a reply, take no "
                   "action, or add a disclosure note.")
    surfaces = {"json": 0.5, "complaint_letter": 0.25, "editors_thread": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        piece = pick(rng, list(PIECES))
        label, subj_fn, stmts = PIECES[piece]
        subject = subj_fn(rng)
        skind = pick(rng, list(stmts))
        text, sev = pick(rng, stmts[skind])
        truth = None
        if skind in ("checkable_fact", "opinion_on_stated_facts"):
            truth = pick_w(rng, {"accurate": 4, "wrong": 4, "outdated": 2} if reg != "clear"
                           else {"accurate": 1, "wrong": 1})
        elif skind == "implied_accusation":
            truth = pick(rng, ["supported", "unsupported", "unsupported"])
        interest = chance(rng, {"clear": 0.3, "borderline": 0.4, "conflicting": 0.55}[reg])
        disclosed = interest and chance(rng, 0.35)
        complainant = pick(rng, ["subject", "subject", "reader", "competitor"] if piece in ("restaurant", "theatre", "music")
                           else ["subject", "subject", "reader"])
        return {"regime": reg, "outlet": _outlet(rng), "piece": piece, "piece_label": label, "subject": subject,
                "statement": text.format(s=subject), "statement_kind": skind, "severity": sev, "truth": truth,
                "interest": interest, "interest_disclosed": disclosed, "complainant_type": complainant,
                "complainant": person(rng), "writer": person(rng), "editor": person(rng),
                "complaint_angle": pick(rng, ["unfair", "inaccurate", "biased", "damaging"]),
                "days_since": pick(rng, [1, 2, 3, 6, 10, 21]), "ticket": ident(rng, "OP", 5, "-")}

    def _premise_wrong(self, f):
        return f["truth"] in ("wrong", "outdated", "unsupported")

    def intended(self, f):
        undisclosed = f["interest"] and not f["interest_disclosed"]
        sk = f["statement_kind"]
        if undisclosed:
            out = "editors_note_disclosure"
        elif sk != "pure_opinion" and self._premise_wrong(f):
            out = "correct_factual_premise"
        elif f["complainant_type"] == "subject":
            out = "offer_published_reply"
        else:
            out = "no_action_fair_comment"
        sev = f["severity"]
        harm = sev if self._premise_wrong(f) or sk == "implied_accusation" and f["truth"] != "supported" else max(0, sev - 1)
        if undisclosed:
            merit = 3
        elif sk in ("checkable_fact", "implied_accusation") and f["truth"] in ("wrong", "unsupported"):
            merit = 3
        elif self._premise_wrong(f):
            merit = 2
        elif f["complainant_type"] == "subject":
            merit = 1
        else:
            merit = 0
        return {"opinion_complaint_outcome": out, "statement_kind": sk, "reputational_harm": harm,
                "complaint_merit": merit, "undisclosed_interest": "yes" if undisclosed else "no"}

    RULES = [
        "Reviews and opinion pieces are held to facts, not views: we correct factual statements in them that are "
        "wrong or out of date, and we do not correct opinions.",
        "A statement that implies undisclosed facts (that a business cheats, that someone was drunk) is treated as a "
        "factual claim and must be supported by evidence the writer holds; if it is not, it is corrected.",
        "Writers must disclose any personal, family or financial interest in their subject; an interest found "
        "undisclosed after publication is set out in an editor's note at the top, which also corrects any factual "
        "error in the piece.",
        "A person or business that is the subject of a harsh but honest judgment is offered a published reply of up "
        "to 250 words; complaints from others about a judgment get no action beyond the ordinary letters page.",
    ]

    def _facts_text(self, f):
        sk, t = f["statement_kind"], f["truth"]
        if sk == "pure_opinion":
            return "no factual statement at issue"
        if sk == "implied_accusation":
            return ("the writer has a recent inspection report and two former staff on record" if t == "supported"
                    else "the writer holds no evidence beyond the visit itself")
        return {"accurate": "the desk has confirmed the stated fact is accurate",
                "wrong": "the desk's check shows the stated fact is wrong",
                "outdated": "the stated fact was true last year but no longer is"}[t]

    def _interest_text(self, f):
        if not f["interest"]:
            return "no interest declared or found"
        return INTERESTS[f["piece"]] + ("; this was disclosed in the piece's footnote" if f["interest_disclosed"]
                                         else "; nothing was disclosed in the piece")

    def _who(self, f):
        subj = {"restaurant": f"the owner of {f['subject']}", "theatre": "the production company",
                "transport_column": "the bus operator", "music": "the band's manager",
                "parks_column": "the council's parks department"}[f["piece"]]
        return {"subject": subj, "reader": "a reader", "competitor": "a competing business"}[f["complainant_type"]]

    def _complaint(self, f):
        angle = {"unfair": "This review was grossly unfair.", "inaccurate": "Your writer got the facts wrong.",
                 "biased": "Your writer clearly had an axe to grind.",
                 "damaging": "This piece has done real damage."}[f["complaint_angle"]]
        return join_sentences([angle, f"In particular the line that \"{f['statement']}\" should never have been "
                                      f"published.", "I expect it to be corrected or removed."])

    def render(self, f, rng, surface):
        if surface == "json":
            return dumps(prune({
                "complaint": {"ref": f["ticket"], "from": f"{f['complainant']}, {self._who(f)}",
                              "text": noisy(self._complaint(f), rng, 0.15)},
                "piece": {"type": f["piece_label"], "subject": f["subject"], "writer": f["writer"],
                          "published": _ago(f["days_since"]), "disputed_passage": f["statement"], "outlet": f["outlet"]},
                "desk_findings": {"facts": self._facts_text(f), "writer_interests": self._interest_text(f)},
                "policy": _policy(rng, "Opinion and reviews policy", self.RULES, True),
            }))
        if surface == "complaint_letter":
            prep = "on" if "column" in f["piece_label"] else "of"
            return "\n".join([f"Subject: Complaint about your {f['piece_label']} {prep} {f['subject']}",
                              f"From: {f['complainant']}", "", noisy(self._complaint(f), rng, 0.2), "",
                              f"{f['complainant']}", "", "--- desk findings ---",
                              _kv([("Ref", f["ticket"]), ("Complainant", self._who(f)), ("Writer", f["writer"]),
                                   ("Published", _ago(f["days_since"])), ("Facts", self._facts_text(f)),
                                   ("Writer's interests", self._interest_text(f))], rng),
                              "", _policy(rng, "Opinion and reviews policy", self.RULES)])
        prep = "on" if "column" in f["piece_label"] else "of"
        msgs = [(f["editor"], f"Complaint {f['ticket']} about the {f['piece_label']} {prep} {f['subject']} by {f['writer']} "
                              f"({_ago(f['days_since'])}), from {self._who(f)}. They object to: \"{f['statement']}\"."),
                ("standards", f"Checked: {self._facts_text(f)}."),
                ("standards", f"Interests: {self._interest_text(f)}."),
                (f["writer"], pick(rng, ["I stand by every word.", "It's a review, not a news report.",
                                         "Happy to look at the wording if needed."]))]
        out = [f"#{f['outlet'].split()[1].lower()}-standards"]
        for who, text in msgs:
            out.append(f"{who.split()[0]}: {noisy(text, rng, 0.1)}")
        return "\n".join(out) + "\n\n" + _policy(rng, "Opinion and reviews policy", self.RULES)

    def questions(self):
        return {
            "opinion_complaint_outcome": choice("What should the outlet do about this complaint?", {
                "correct_factual_premise": "Correct the factual statement the piece got wrong or could not support.",
                "offer_published_reply": "Make no correction, but offer the subject a published reply.",
                "no_action_fair_comment": "Take no action; the passage is honest opinion and the complaint gives no ground under the policy.",
                "editors_note_disclosure": "Add an editor's note disclosing the writer's interest, correcting any error with it.",
            }),
            "statement_kind": choice("What kind of statement is the disputed passage?", {
                "checkable_fact": "A plain statement of fact that can be checked.",
                "pure_opinion": "A judgment or taste that cannot be true or false.",
                "opinion_on_stated_facts": "A judgment resting on a fact stated alongside it.",
                "implied_accusation": "A judgment that implies wrongdoing or facts the piece does not state.",
            }),
            "reputational_harm": score("How much harm could the passage do to the person or business concerned?", [
                "Negligible.",
                "Mild: an unflattering view.",
                "Real: readers could think worse of them on a matter of fact.",
                "Serious: it suggests misconduct, danger or dishonesty.",
            ]),
            "complaint_merit": score("How much merit does the complaint have under the policy?", [
                "None.",
                "Slight: no correction, but a right of reply.",
                "Substantial: part of it is borne out.",
                "Fully justified.",
            ]),
            "undisclosed_interest": noul(
                "The writer had an interest in the subject that the policy required them to disclose and they did not.",
                yes="A personal, family or financial interest exists and the piece did not disclose it.",
                no="No such interest exists, or it was disclosed in the piece."),
        }


# =========================================================================== schema 3
# quote: (speaker role, what was said, tidied version, materially changed version, change severity 2-3,
#         the context that framing dropped)
QUOTES = [
    ("leisure centre manager",
     "Honestly, we may have to look again at weekend opening at the pool if the grant isn't renewed.",
     "We may have to look again at weekend opening at the pool if the grant isn't renewed.",
     "We will have to close the pool at weekends.", 3,
     "the remark answered a question about what would happen if the grant were cut, but ran under a headline about planned closures"),
    ("head teacher",
     "I think most parents have been supportive of the uniform change, though some families have struggled with the cost.",
     "Most parents have been supportive of the uniform change, though some families have struggled with the cost.",
     "Parents have been fully supportive of the uniform change.", 2,
     "the story placed the remark straight after complaints about cost, as though it dismissed them"),
    ("highways officer",
     "The Station Road repairs were held up by the contractor, not by the council's budget.",
     "The Station Road repairs were held up by the contractor, not the council's budget.",
     "The Station Road repairs were held up by the council's budget.", 3,
     "the remark was given as one reason among several, but printed as the whole explanation"),
    ("bakery owner",
     "We're not closing - we're moving to a smaller unit across the square in the spring.",
     "We're not closing; we're moving to a smaller unit across the square in the spring.",
     "We're closing in the spring.", 3,
     "the story used the remark under a headline about shops leaving the high street"),
    ("food bank director",
     "Demand is up about a fifth on last winter, mostly from working families.",
     "Demand is up about a fifth on last winter, mostly from working families.",
     "Demand has doubled since last winter.", 2,
     "the remark came before a caveat about a one-off spike after the factory closure, which was left out"),
    ("youth football coach",
     "If pitch fees go up again, a couple of our smaller teams might struggle to carry on.",
     "If pitch fees go up again, a couple of our smaller teams might struggle to carry on.",
     "Our teams will fold if pitch fees go up.", 3,
     "the coach was asked about the worst case; the story ran it under 'Clubs warn of collapse'"),
    ("charity shop manager",
     "It's been a difficult year, but the shop will stay open.",
     "It's been a difficult year, but the shop will stay open.",
     "It's been a disastrous year.", 2,
     "the second half of the sentence, about staying open, was cut from the printed quote"),
    ("planning consultant",
     "The scheme meets most of the design guidance; the height is the main sticking point.",
     "The scheme meets most of the design guidance; height is the main sticking point.",
     "The scheme meets the design guidance.", 2,
     "the remark was printed as an endorsement next to the developer's statement"),
]
RECORDS = {"recording": "audio recording of the interview", "email": "email exchange (the interview was by email)",
           "shorthand": "reporter's shorthand notes made during the call", "typed_later": "notes typed up the next day",
           "memory": "no recording or notes; the reporter's recollection"}


class QuoteAttributionDispute(Schema):
    name = "quote_attribution_dispute"
    description = ("A source says a published quote misrepresents them or broke an off-the-record agreement: correct "
                   "it, add context, stand by it, remove it or refer it.")
    surfaces = {"json": 0.5, "source_email": 0.25, "standards_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        qi = rng.randrange(len(QUOTES))
        objection = pick_w(rng, {"misquote": 3, "context": 2, "off_record": 3.5, "regret": 2})
        record = pick_w(rng, {"recording": 3, "email": 1, "shorthand": 3, "typed_later": 2, "memory": 2})
        # what the published version actually is (the truth the record can show)
        if objection == "misquote":
            shown = pick(rng, ["material", "material", "tidy"] if reg != "clear" else ["material", "exact"])
        elif objection == "context":
            shown = pick(rng, ["context", "context", "tidy"])
        else:
            shown = pick(rng, ["exact", "tidy"])
        offrec = None
        if objection == "off_record" or (reg == "conflicting" and chance(rng, 0.4)):
            offrec = {"when": pick(rng, ["before", "before", "before", "after"]),
                      "evidence": pick(rng, ["pre_interview_email", "reporter_notes", "reporter_notes", "none"])}
            if reg == "clear":
                offrec = pick(rng, [{"when": "before", "evidence": "pre_interview_email"},
                                    {"when": "after", "evidence": "none"}])
        partial = record == "recording" and chance(rng, 0.2)
        return {"regime": reg, "outlet": _outlet(rng), "quote_index": qi, "objection": objection,
                "record": record, "recording_partial": partial, "shown": shown, "off_record": offrec,
                "speaker": person(rng), "place": pick(rng, PLACES), "reporter": person(rng),
                "editor": person(rng), "days_since": pick(rng, [1, 2, 3, 5, 8, 14, 30]),
                "central": chance(rng, 0.5), "reason": pick(rng, ["employer", "neighbours", "misrepresents",
                                                                  "context"]),
                "ticket": ident(rng, "QD", 5, "-")}

    def _valid_offrec(self, f):
        o = f["off_record"]
        return bool(o and o["when"] == "before" and o["evidence"] != "none")

    def _published(self, f):
        role, said, tidy, material, sev, ctx = QUOTES[f["quote_index"]]
        return {"exact": said, "tidy": tidy, "material": material, "context": tidy}[f["shown"]]

    def intended(self, f):
        role, said, tidy, material, sev, ctx = QUOTES[f["quote_index"]]
        shown = f["shown"]
        if self._valid_offrec(f):
            ruling = "remove_quote_breach"
        elif f["record"] == "memory" and f["objection"] in ("misquote", "context"):
            ruling = "refer_standards_editor"
        elif shown == "material":
            ruling = "correct_wording"
        elif shown == "context":
            ruling = "add_context_note"
        else:
            ruling = "stand_by_quote"
        strength = {"recording": 3, "email": 3, "shorthand": 2, "typed_later": 1, "memory": 0}[f["record"]]
        if f["recording_partial"]:
            strength -= 1
        change = {"exact": 0, "tidy": 0, "context": 2, "material": sev}[shown]
        if shown == "tidy" and said != tidy:
            change = 1 if chance_free(said, tidy) else 0
        regret = (shown in ("exact", "tidy") and not self._valid_offrec(f) and f["record"] != "memory")
        return {"quote_ruling": ruling, "record_strength": strength, "meaning_change": change,
                "offrecord_agreement_valid": "yes" if self._valid_offrec(f) else "no",
                "objection_is_regret": "yes" if regret else "no"}

    RULES = [
        "Words inside quotation marks must be what was said; removing filler words and fixing grammar is allowed, "
        "changes that alter meaning are errors and are corrected.",
        "Where a quote is accurate but its placement or framing drops context that changes how it reads, add a "
        "clarifying note giving the context.",
        "We do not remove or alter accurate, on-the-record quotes because a source regrets them.",
        "Off-the-record or background terms bind us only if agreed before the remark was made and shown by an email, "
        "a recording or the reporter's own notes; if such an agreement was breached, the quote is removed and a note "
        "says so.",
        "Where there is no recording or notes and the reporter and source disagree about wording or context, the "
        "standards editor decides.",
    ]

    def _source_says(self, f):
        role, said, tidy, material, sev, ctx = QUOTES[f["quote_index"]]
        o = f["objection"]
        if o == "misquote":
            return f"I never said that. What I said was: \"{said}\""
        if o == "context":
            return f"The words are roughly right but the story twisted them: {ctx}."
        if o == "off_record":
            when = (f["off_record"] or {}).get("when", "before")
            return ("I told your reporter at the start that this part was off the record." if when == "before"
                    else "That comment was obviously off the record and should never have been used.")
        return {"employer": "My employer is unhappy about the quote and I need it taken down.",
                "neighbours": "Since the story ran I have had neighbours knocking on my door. Please remove my quote.",
                "misrepresents": "The quote makes me look foolish and I want it removed.",
                "context": "I would like the quote removed; I did not expect it to be used so prominently."}[f["reason"]]

    def _record_text(self, f):
        role, said, tidy, material, sev, ctx = QUOTES[f["quote_index"]]
        r = f["record"]
        if r == "memory":
            return RECORDS[r] + "; the reporter is sure the quote is right"
        base = RECORDS[r] + (" (partly inaudible)" if f["recording_partial"] else "")
        if f["shown"] == "context":
            return f"{base}: words match the published quote; {ctx}"
        return f"{base}: the source said \"{said}\""

    def _offrec_text(self, f):
        o = f["off_record"]
        if not o:
            return None
        ev = {"pre_interview_email": "an email sent before the interview asking that remarks about this be 'background only'",
              "reporter_notes": "the reporter's notes mark the passage 'OTR'", "none": "no record of any agreement"}[o["evidence"]]
        return f"off-record claim raised {'as agreed before the remark' if o['when'] == 'before' else 'only after publication'}; {ev}"

    def render(self, f, rng, surface):
        role = QUOTES[f["quote_index"]][0]
        pub = self._published(f)
        if surface == "json":
            return dumps(prune({
                "dispute": {"ref": f["ticket"], "source": f"{f['speaker']}, {role} in {f['place']}",
                            "objection": noisy(self._source_says(f), rng, 0.15)},
                "story": {"outlet": f["outlet"], "reporter": f["reporter"], "published": _ago(f["days_since"]),
                          "quote_as_published": pub,
                          "quote_role_in_story": "central" if f["central"] else "peripheral"},
                "record": {"what_exists": self._record_text(f), "ground_rules": self._offrec_text(f)},
                "policy": _policy(rng, "Quotation policy", self.RULES, True),
            }))
        if surface == "source_email":
            return "\n".join([f"Subject: Your story quoting me ({_ago(f['days_since'])})", f"From: {f['speaker']}", "",
                              noisy(f"I am the {role} you quoted as saying: \"{pub}\" {self._source_says(f)}", rng, 0.2),
                              "", f["speaker"], "", "--- desk notes ---",
                              _kv([("Ref", f["ticket"]), ("Reporter", f["reporter"]),
                                   ("Quote's role", "central to the story" if f["central"] else "one of several voices"),
                                   ("Record", self._record_text(f)), ("Ground rules", self._offrec_text(f) or "none raised")], rng),
                              "", _policy(rng, "Quotation policy", self.RULES)])
        para = [f"{f['speaker']} ({role}, {f['place']}) disputes a quote in {f['reporter']}'s story published "
                f"{_ago(f['days_since'])}: \"{pub}\"", f"The source's position: {self._source_says(f)}",
                f"Record: {self._record_text(f)}.", (f"Ground rules: {self._offrec_text(f)}." if f["off_record"] else ""),
                f"The quote is {'central to' if f['central'] else 'a minor part of'} the story."]
        return (f"MEMO to {f['editor']} - quote dispute {f['ticket']}\n\n{noisy(join_sentences(para), rng, 0.1)}\n\n"
                + _policy(rng, "Quotation policy", self.RULES))

    def questions(self):
        return {
            "quote_ruling": choice("How should the standards desk rule on the disputed quote?", {
                "correct_wording": "Correct the quote to what was actually said, with a note.",
                "add_context_note": "Keep the quote and add a note supplying the context it was given in.",
                "stand_by_quote": "Stand by the quote as published.",
                "remove_quote_breach": "Remove the quote and note that it breached an agreement made before it was said.",
                "refer_standards_editor": "Refer the dispute to the standards editor, as the policy requires when there is no record.",
            }),
            "record_strength": score("How strong is the outlet's record of what was actually said?", [
                "None: memory only.",
                "Weak: notes written up after the event.",
                "Fair: contemporaneous notes, or a recording with gaps.",
                "Strong: a full recording or the written exchange itself.",
            ]),
            "meaning_change": score("How far does the published quote, as presented, depart from what the source meant?", [
                "Not at all.",
                "Trivially: tidying that keeps the meaning.",
                "Materially: a qualifier or context that changes how it reads is gone.",
                "Fundamentally: it says something the source did not say.",
            ]),
            "offrecord_agreement_valid": noul(
                "An off-the-record or background agreement covering the quote was made before it was said and can be shown.",
                yes="The agreement predates the remark and an email, recording or the reporter's notes show it.",
                no="No agreement was claimed, it was raised only afterwards, or nothing shows it."),
            "objection_is_regret": noul(
                "The record shows the quote is accurate, and the objection is in substance regret at having said it."),
        }


def chance_free(a, b):
    """True when two versions of a quote differ by more than punctuation (deterministic, no rng)."""
    strip = lambda s: "".join(ch for ch in s.lower() if ch.isalnum() or ch == " ").split()  # noqa: E731
    return strip(a) != strip(b)


# =========================================================================== schema 4
STORIES = {
    "minor_charge": ("an arrest or charge for a minor offence", ["{n}, {a}, charged after scuffle outside {v}",
                                                                 "Driver charged after minor collision on {s}",
                                                                 "{n}, {a}, arrested over criminal damage at {v}"]),
    "serious_charge": ("a charge for a serious offence", ["{n} charged with fraud over charity accounts",
                                                          "{n} accused of assault at {v}"]),
    "youth": ("an incident involving a person under 18", ["Teenagers cautioned after graffiti at {p} station",
                                                          "Pupil, 16, named in court over vandalism at {v}"]),
    "business_failure": ("a business failure", ["{b} closes owing suppliers ${m}",
                                                "{n}'s {b} goes into administration"]),
    "human_interest": ("a feature about a private person", ["{n} on rebuilding life after redundancy",
                                                            "{n}'s long road back to running"]),
    "official": ("a public official acting in office", ["Councillor {n} criticised over {p} depot contract",
                                                        "{p} harbour master {n} defends berth charges"]),
}
OUTCOMES = {"dropped": "charges dropped", "acquitted": "acquitted at trial", "sealed": "convicted, record since sealed",
            "convicted": "convicted (fine)", "pending": "case still pending", None: None}


class ArchiveUnpublishRequest(Schema):
    name = "archive_unpublish_request"
    description = ("A person asks for an old story about them to be removed, anonymised or updated, judged against "
                   "the outlet's archive policy.")
    surfaces = {"json": 0.5, "request_letter": 0.25, "archive_review_form": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        stype = pick_w(rng, {"minor_charge": 3, "serious_charge": 2, "youth": 1.5, "business_failure": 1.2,
                             "human_interest": 1.5, "official": 1.5})
        court = stype in ("minor_charge", "serious_charge", "youth")
        if reg == "borderline":
            years = near(rng, pick(rng, [5, 10] if court else [10]), 1.2, lo=0.5, digits=1)
        elif court:
            years = pick(rng, [1, 2, 3, 4, 6, 8, 11, 14])
        else:
            years = pick(rng, [2, 4, 8, 11, 12, 14, 16])
        outcome = None
        if court:
            outcome = pick_w(rng, {"dropped": 3.5, "acquitted": 3, "sealed": 1.5, "convicted": 1.5, "pending": 0.8})
        docs = court and chance(rng, {"clear": 0.75, "borderline": 0.7, "conflicting": 0.5}[reg])
        harm = pick_w(rng, {"none": 2, "claimed": 3, "documented": 3.5, "safety": 0.7})
        legal = chance(rng, 0.04)
        if reg == "conflicting" and chance(rng, 0.5):
            harm = pick(rng, ["documented", "documented", "safety"])
            docs = False if court else docs
        update_exists = court and outcome in ("dropped", "acquitted") and chance(rng, 0.15)
        n = person(rng)
        head = pick(rng, STORIES[stype][1]).format(n=n, a=pick(rng, [19, 23, 31, 44, 52]), v=pick(rng, ["the Crown and Anchor", "the station car park", "a takeaway on " + pick(rng, STREETS), "the leisure centre"]),
                                                   s=pick(rng, STREETS), p=pick(rng, PLACES), b=_biz(rng),
                                                   m=f"{pick(rng, [40, 85, 120, 300])},000")
        return {"regime": reg, "outlet": _outlet(rng), "story_type": stype, "headline": head, "person": n,
                "years_since": years, "outcome": outcome, "outcome_documented": bool(docs), "harm": harm,
                "legal_order": legal, "update_exists": update_exists,
                "requester": pick(rng, ["the person named", "the person named", "their solicitor", "a family member"]),
                "reason": pick(rng, ["job", "search", "children", "fresh_start"]), "editor": person(rng),
                "ref": ident(rng, "ARC", 5, "-")}

    def _anon_criteria(self, f):
        if f["story_type"] == "youth":
            return True
        return (f["story_type"] == "minor_charge" and f["years_since"] > 5
                and f["outcome"] in ("dropped", "acquitted", "sealed"))

    def intended(self, f):
        st = f["story_type"]
        safety = f["harm"] == "safety"
        anon = self._anon_criteria(f)
        not_conv = f["outcome"] in ("dropped", "acquitted")
        if f["legal_order"] or safety:
            ruling = "unpublish"
        elif st == "official":
            ruling = "keep_unchanged"
        elif anon and (st == "youth" or f["outcome_documented"]):
            ruling = "anonymise_name"
        elif not_conv and f["outcome_documented"] and not f["update_exists"]:
            ruling = "add_outcome_update"
        elif f["years_since"] > 10 and st != "official" and f["harm"] == "documented":
            ruling = "remove_from_search"
        else:
            ruling = "keep_unchanged"
        pi = {"official": 3, "serious_charge": 2, "business_failure": 2, "minor_charge": 1, "youth": 0,
              "human_interest": 0}[st]
        if f["outcome"] in ("convicted",) and st == "serious_charge":
            pi = 3
        if f["years_since"] > 10 and st != "official":
            pi = max(0, pi - 1)
        harm = {"none": 0, "claimed": 1, "documented": 2, "safety": 3}[f["harm"]]
        return {"archive_request_ruling": ruling, "public_interest_now": pi, "ongoing_harm": harm,
                "meets_anonymisation_criteria": "yes" if anon else "no",
                "outcome_documented": "yes" if f["outcome_documented"] else "no"}

    RULES = [
        "We do not unpublish, except where a court order or other legal requirement compels it, or where police or "
        "a court have documented a credible threat to someone's safety.",
        "Stories about public officials acting in their public role are not altered.",
        "Where a story reported an arrest or charge and the person was not convicted, we add an update with the "
        "outcome once it is documented by a court record.",
        "On request we replace a name with initials for anyone under 18 at the time, and for minor offences "
        "reported more than 5 years ago where the charge was dropped, the person was acquitted or the record has "
        "been sealed, once the outcome is documented.",
        "Where none of the above applies, a story more than 10 years old about a private person may be removed from "
        "search-engine indexing if continuing harm is documented.",
    ]

    def _harm_text(self, f):
        return {"none": "no harm described beyond embarrassment",
                "claimed": "says the story costs them job offers (no documents)",
                "documented": "has supplied a former employer's letter and screenshots of harassing messages that cite the story",
                "safety": "has supplied a police reference and a court protection order relating to threats prompted by the story"}[f["harm"]]

    def _outcome_text(self, f):
        if f["outcome"] is None:
            return None
        doc = "court record supplied" if f["outcome_documented"] else "no court record supplied"
        upd = "; story already carries an update" if f["update_exists"] else ""
        return f"{OUTCOMES[f['outcome']]} ({doc}){upd}"

    def _age(self, f):
        y = f["years_since"]
        return "a year ago" if y == 1 else f"{y:g} years ago" if y >= 1 else "under a year ago"

    def _harm_letter(self, f):
        me = f["requester"] == "the person named"
        who = "me" if me else "them"
        return {"none": "It is embarrassing." if me else "It is a source of embarrassment.",
                "claimed": f"It has cost {who} job offers, though there is nothing in writing to prove it.",
                "documented": "Enclosed are a former employer's letter and screenshots of messages from people who "
                              f"found the story and used it against {who}.",
                "safety": "Enclosed are a police reference and a court protection order; the threats began after "
                          "this story circulated."}[f["harm"]]

    def render(self, f, rng, surface):
        st_label = STORIES[f["story_type"]][0]
        if surface == "json":
            return dumps(prune({
                "request": {"ref": f["ref"], "from": f["requester"], "asks_for": "removal of the story",
                            "harm_described": self._harm_text(f),
                            "legal_order": "court order attached requiring removal" if f["legal_order"] else None},
                "story": {"headline": f["headline"], "published": self._age(f), "type": st_label,
                          "outcome": self._outcome_text(f), "outlet": f["outlet"]},
                "policy": _policy(rng, "Archive policy", self.RULES, True),
            }))
        if surface == "request_letter":
            ask = {"job": "It comes up every time an employer searches my name.",
                   "search": "It is the first thing anyone finds about me online.",
                   "children": "My children's friends have started finding it.",
                   "fresh_start": "I have moved on and deserve a fresh start."}[f["reason"]]
            body = [f"I am writing as {f['requester']} about your story \"{f['headline']}\", published {self._age(f)}.",
                    ask if f["requester"] == "the person named" else
                    {"job": "It comes up every time an employer searches the name.",
                     "search": "It is the first thing anyone finds online.",
                     "children": "Their children's friends have started finding it.",
                     "fresh_start": "They have moved on and deserve a fresh start."}[f["reason"]], f"For the record, the matter ended: {self._outcome_text(f)}." if f["outcome"] else "",
                    self._harm_letter(f),
                    "A court order requiring removal is attached." if f["legal_order"] else "",
                    "I ask that you take the story down."]
            return "\n".join([f"Subject: Request to remove archive story ({f['ref']})", "",
                              noisy(join_sentences(body), rng, 0.2), "", "Yours sincerely", "",
                              _policy(rng, "Archive policy", self.RULES)])
        rows = _kv([("Ref", f["ref"]), ("Story", f["headline"]), ("Type", st_label), ("Published", self._age(f)),
                    ("Outcome", self._outcome_text(f) or "not a court story"), ("Requester", f["requester"]),
                    ("Harm", self._harm_text(f)), ("Legal order", "yes, attached" if f["legal_order"] else "none"),
                    ("Reviewer", f["editor"])], rng)
        return f"ARCHIVE REVIEW FORM - {f['outlet']}\n{rows}\n\n{_policy(rng, 'Archive policy', self.RULES)}"

    def questions(self):
        return {
            "archive_request_ruling": choice("How should the outlet treat the archived story?", {
                "keep_unchanged": "Leave the story as it is.",
                "add_outcome_update": "Keep the story and add an update recording how the case ended.",
                "anonymise_name": "Replace the person's name with initials and note the change.",
                "remove_from_search": "Keep the story online but ask search engines not to index it.",
                "unpublish": "Take the story down.",
            }),
            "public_interest_now": score("How much public interest does the story still serve with the name attached?", [
                "Little or none.",
                "Some, fading with time.",
                "Considerable.",
                "High: it records conduct in public office or serious proven wrongdoing.",
            ]),
            "ongoing_harm": score("How much continuing harm does the evidence show the story causing?", [
                "None beyond embarrassment.",
                "Claimed but not documented.",
                "Documented harm to work or private life.",
                "A documented threat to safety.",
            ]),
            "meets_anonymisation_criteria": noul(
                "The story meets the policy's criteria for replacing the name with initials.",
                yes="The person was under 18 at the time, or it is a minor offence over 5 years old with no conviction or a sealed record.",
                no="None of the anonymisation criteria applies."),
            "outcome_documented": noul(
                "The requester has supplied a court record showing how the case ended."),
        }


# =========================================================================== schema 5
# event: (label, headline, the detail that was wrong, base safety effect 0-3, wrong/right value pools)
EVENTS = [
    ("boil_water", "Boil-water notice issued for parts of {p}", "the streets covered by the notice", 3,
     [f"{a} and {b}" for a, b in [("Mill Lane", "Kiln Road"), ("Elm Street", "Orchard Way"), ("Bridge Street", "Wharf Street"),
                                  ("Church Walk", "Market Row")]]),
    ("road_closure", "{r} closed after lorry spill", "which road is closed", 2,
     ["Station Road", "Harbour Road", "Quarry Hill", "Fernside Avenue", "Tannery Lane"]),
    ("school_closure", "{x} closed today after heating failure", "which school is closed", 1,
     [f"{p} {k}" for p in ("St Anne's", "Hillside", "Riverside", "Northfield") for k in ("Primary", "Junior School")]),
    ("evacuation", "Residents in zone {z} told to leave as river rises", "the evacuation zone", 3,
     ["zone A", "zone B", "zone C", "zone D"]),
    ("shelter", "Emergency shelter opens at {x}", "where the shelter is", 3,
     ["the leisure centre", "St Mary's church hall", "the community college", "the old drill hall"]),
    ("power", "Power expected back by {t}", "the restoration time", 1, ["6pm", "8pm", "midnight", "tomorrow morning"]),
    ("meeting", "Public meeting on depot plan moved to {t}", "the meeting time", 0, ["6pm", "6.30pm", "7pm", "7.30pm"]),
    ("parade", "Carnival parade closes {r} from {t}", "the closure start time", 1, ["10am", "11am", "noon", "1pm"]),
]
OFFICIAL = {"boil_water": "the water company's notice", "road_closure": "the police traffic update",
            "school_closure": "the school's own notice to parents", "evacuation": "the council's emergency notice",
            "shelter": "the council's emergency notice", "power": "the network operator's outage update",
            "meeting": "the council's meeting notice", "parade": "the organisers' published route notice"}
CONFIRM = {"official": "the official notice", "reporter_scene": "our reporter at the scene",
           "two_sources": "two independent residents with photos", "single_reader": "a single reader's message",
           "none": "nobody yet; a reader has queried it"}


class BreakingStoryError(Schema):
    name = "breaking_story_error"
    description = ("An error in a live or breaking story, possibly also sent as a push alert or posted to social "
                   "channels: verify, then fix in the right channels at the right speed.")
    surfaces = {"json": 0.5, "live_desk_chat": 0.25, "error_log": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        ei = rng.randrange(len(EVENTS))
        key, head, what, base, pool = EVENTS[ei]
        wrong, right = rng.sample(pool, 2)
        conf = pick_w(rng, {"official": 3, "reporter_scene": 2, "two_sources": 1.5, "single_reader": 2.5, "none": 1}
                      if reg != "clear" else {"official": 3, "single_reader": 1})
        direction = pick(rng, ["omits_affected", "adds_unaffected"]) if base >= 2 else "neutral"
        push = chance(rng, 0.5)
        social = push or chance(rng, 0.6)
        return {"regime": reg, "outlet": _outlet(rng), "event": key, "event_index": ei, "what": what,
                "wrong": wrong, "right": right, "base_safety": base, "direction": direction,
                "confirmation": conf, "push_sent": push, "push_reach": pick(rng, [4000, 12000, 30000, 85000]) if push else None,
                "social_posted": social, "syndicated": chance(rng, 0.4), "partners": pick(rng, [2, 3, 5, 8]),
                "minutes_since": pick(rng, [6, 12, 20, 35, 55, 90]),
                "views": pick(rng, [800, 2500, 6000, 15000, 40000]),
                "event_in_min": pick(rng, [None, None, 20, 45, 90, 240]), "editor": person(rng),
                "reporter": person(rng), "producer": person(rng), "place": pick(rng, PLACES),
                "street": pick(rng, STREETS),
                "published": rand_ts(rng, hours=(6, 21)), "ref": ident(rng, "LIVE", 4, "-")}

    def _confirmed(self, f):
        return f["confirmation"] in ("official", "reporter_scene", "two_sources")

    def _effect(self, f):
        e = f["base_safety"]
        if f["direction"] == "adds_unaffected":
            e -= 1
        return max(0, e)

    def intended(self, f):
        eff = self._effect(f)
        if not self._confirmed(f):
            ch = "verify_first"
        elif f["push_sent"] and eff >= 2:
            ch = "corrective_push_alert"
        elif f["social_posted"]:
            ch = "story_and_social"
        else:
            ch = "story_log_only"
        speed = eff
        if f["event_in_min"] is not None and f["event_in_min"] <= 60:
            speed += 1
        if f["push_sent"] and (f["push_reach"] or 0) >= 30000:
            speed += 1
        return {"fix_channel": ch, "fix_speed": min(3, speed), "public_safety_effect": eff,
                "error_confirmed": "yes" if self._confirmed(f) else "no",
                "partner_notice_due": "yes" if self._confirmed(f) and f["syndicated"] else "no"}

    RULES = [
        "An error in a live story is confirmed when the official source, our reporter on the scene or two "
        "independent sources contradict the published detail; a single reader's report is verified first, within "
        "15 minutes, before anything is changed.",
        "Confirmed errors are corrected in the story and logged with the time in its update log.",
        "If the wrong detail went out on our social posts, each post is corrected or replied to.",
        "If it went out in a push alert and could lead readers to act unsafely or miss a safety instruction, send a "
        "corrective push alert as well; otherwise the story, log and social posts are enough.",
        "Stories carried by syndication partners get a correction notice to every partner once the error is confirmed.",
    ]

    def _conf(self, f):
        return OFFICIAL[f["event"]] if f["confirmation"] == "official" else CONFIRM[f["confirmation"]]

    def _published(self, f):
        """What went out: a swap (the affected place left out) or an over-inclusion (an unaffected one added)."""
        if f["direction"] != "adds_unaffected":
            return f["wrong"]
        if f["event"] == "evacuation":
            return f"zones {f['right'].replace('zone ', '')} and {f['wrong'].replace('zone ', '')}"
        if f["event"] == "shelter":
            return f"{f['right']} and {f['wrong']}"
        if f["event"] == "boil_water":
            return f"{f['right'].replace(' and ', ', ')}, {f['wrong']}"
        return f"{f['right']} and {f['wrong']}"

    def _headline(self, f):
        key, head, what, base, pool = EVENTS[f["event_index"]]
        pub = self._published(f)
        if f["event"] == "evacuation" and f["direction"] == "adds_unaffected":
            return f"Residents in {pub} told to leave as river rises"
        road = f["street"] if f["event"] == "parade" else pub
        return head.format(p=f["place"], r=road, x=pub, z=pub.replace("zone ", ""), t=pub)

    def _channels(self, f):
        bits = ["live story"]
        if f["social_posted"]:
            bits.append("social posts")
        if f["push_sent"]:
            bits.append(f"push alert to {f['push_reach']:,} devices")
        if f["syndicated"]:
            bits.append(f"{f['partners']} syndication partner sites")
        return ", ".join(bits)

    def _timing(self, f):
        if f["event_in_min"] is None:
            return None
        return {"meeting": f"the meeting starts in {f['event_in_min']} min", "parade": f"the closure starts in {f['event_in_min']} min",
                "power": f"the next restoration update is due in {f['event_in_min']} min"}.get(
            f["event"], f"the next official update is due in {f['event_in_min']} min")

    def render(self, f, rng, surface):
        claim = f"published {f['what']} as {self._published(f)}; the correct detail is {f['right']}" + (
            " only" if f["direction"] == "adds_unaffected" else "")
        if surface == "json":
            return dumps(prune({
                "story": {"ref": f["ref"], "headline": self._headline(f), "outlet": f["outlet"],
                          "published_minutes_ago": f["minutes_since"], "views_so_far": f["views"],
                          "went_out_on": self._channels(f)},
                "suspected_error": {"detail": claim, "raised_by": self._conf(f), "timing": self._timing(f)},
                "policy": _policy(rng, "Live coverage corrections policy", self.RULES, True),
            }))
        if surface == "live_desk_chat":
            t = f["published"] + timedelta(minutes=f["minutes_since"])
            msgs = [(f["producer"], f"Possible error in \"{self._headline(f)}\": we {claim}."),
                    (f["producer"], f"Source for that: {self._conf(f)}."),
                    (f["editor"], f"What has it gone out on?"),
                    (f["producer"], f"{self._channels(f)}. {f['views']:,} views in {f['minutes_since']} min."
                                    + (f" Also {self._timing(f)}." if self._timing(f) else ""))]
            out = [f"#live-desk {f['outlet']}"]
            for who, text in msgs:
                t += timedelta(minutes=rng.randint(0, 2))
                out.append(f"[{t.strftime('%H:%M')}] {who.split()[0]}: {noisy(text, rng, 0.1)}")
            return "\n".join(out) + "\n\n" + _policy(rng, "Live corrections policy", self.RULES)
        rows = _kv([("Ref", f["ref"]), ("Story", self._headline(f)), ("Published", f"{f['minutes_since']} min ago"),
                    ("Suspected error", claim), ("Raised by", self._conf(f)),
                    ("Channels", self._channels(f)), ("Views", f"{f['views']:,}"), ("Timing", self._timing(f))], rng)
        return f"LIVE ERROR LOG - {f['outlet']}\n{rows}\n\n{_policy(rng, 'Live corrections policy', self.RULES)}"

    def questions(self):
        return {
            "fix_channel": choice("How should the desk handle the suspected error now?", {
                "verify_first": "Leave the text for now and verify the report before changing anything.",
                "story_log_only": "Correct the story and its update log; nothing else carried the error.",
                "story_and_social": "Correct the story and log, and correct the social posts that carried the error.",
                "corrective_push_alert": "Correct the story, log and posts, and send a corrective push alert.",
            }),
            "fix_speed": score("How quickly does the correction (or the verification) need to happen?", [
                "With the next routine update of the story.",
                "Within the hour.",
                "Within 15 minutes.",
                "Immediately, ahead of anything else on the desk.",
            ]),
            "public_safety_effect": score("How likely is the error to lead readers into unsafe or costly choices?", [
                "Not at all: an inconvenience at most.",
                "Minor: some readers may waste a trip or some time.",
                "Real: readers could make a costly or risky choice.",
                "Serious: readers could be put in danger.",
            ]),
            "error_confirmed": noul(
                "The error is confirmed under the policy.",
                yes="The official source, the reporter on scene or two independent sources contradict the published detail.",
                no="Only a single reader's report or an unanswered query so far."),
            "partner_notice_due": noul(
                "Syndication partners must now be sent a correction notice."),
        }


WORKFLOW = Workflow(
    name="news_correction_requests",
    title="News correction requests",
    description=("Editorial-standards decisions on requests to correct, clarify, re-attribute or remove published "
                 "material, against a written corrections policy."),
    domain="media / newsroom editorial standards",
    schemas=[FactualErrorClaim(), OpinionPieceDispute(), QuoteAttributionDispute(), ArchiveUnpublishRequest(),
             BreakingStoryError()],
    notes="Fictional outlets and policies; neutral local beats; no reply-to-reader, satisfaction or hand-off "
          "questions (not a customer-support reskin).",
)
