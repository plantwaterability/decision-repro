"""Library material reconsideration: how a (fictional) public library system handles challenges to items in its
collection, judged against its written reconsideration, selection, placement and currency policies.

Formal requests at intake (eligibility, completeness, the informal step, the re-review bar, items outside the
collection), committee decisions after reading a challenged work, requests to move an item between age
collections, challenges to nonfiction as outdated or wrong, and appeals to the library board. Each state quotes the
policy extract that decides it.

Viewpoint-neutral by construction: every library, branch, title, author, review journal and policy is invented,
and challenged content is described only in neutral generic terms (frightening imagery, strong language, a death
scene, outdated dosage tables, unsafe canning times). No position on any political, religious or social question
is encoded, challengers come from every direction (parents, patrons, teachers, staff), and no protected attribute
decides anything. Decisions turn on process and on the written selection, placement and currency criteria.

Each schema draws a `regime` first: 'clear' cases sit squarely inside one policy outcome, 'borderline' cases put a
date, a count or an age band right at a policy line (a form one item short, an appeal a day past the deadline, a
health title just past its currency limit, reviewers straddling two age bands), and 'conflicting' cases pair
signals that disagree (high circulation against weak reviews, a publisher band against two reviewers, a committee
member who read only excerpts). Every schema defines `intended(f)`, the answer the quoted policy gives; it documents
the logic and drives the spread check.
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
    table,
)

# =========================================================================== shared pools and helpers
LIBRARIES = ["Fenwick County Library District", "Ashgrove Public Library", "Tarrow Vale Libraries",
             "Kelderby Regional Library", "Marrowgate City Library", "Ostwick Township Library",
             "Innisvale Free Library", "Lantern Bay Public Library", "Brackwater Library Service",
             "Corriston Community Libraries"]
BRANCHES = ["Central", "Northside", "Riverside", "Mill Street", "Eastgate", "Hilltop", "Harbour", "Old Town",
            "Westfield", "Orchard Road"]
PLACES = ["Marrowgate", "Kelderby", "Ostwick", "Fennick", "Harlow", "Brackwater", "Innisvale", "Skelmar", "Tarrow"]
FIC_A = ["Lantern", "Hollow", "Silver", "Winter", "Paper", "Quiet", "Copper", "Salt", "Midnight", "Orchard",
         "Thistle", "Harbour", "Willow", "Ember", "Glass", "Cinder", "Bramble", "Velvet"]
FIC_B = ["Fox", "Garden", "River", "Kite", "Lighthouse", "Crow", "Bridge", "Boat", "Map", "Moth", "Clock", "Sparrow",
         "Island", "Forest", "Moon", "Door", "Hare", "Tide"]
FIRSTS = ["Mira", "Tobias", "Juno", "Ezra", "Pip", "Nell", "Otto", "Wren", "Ines", "Kit", "Bram", "Lottie"]
COLLECTIONS = ["picture books", "early readers", "juvenile", "teen", "adult"]
BANDS = {"picture books": "ages 0-5", "early readers": "ages 5-7", "juvenile": "ages 8-12", "teen": "ages 13-17",
         "adult": "18+"}
CONCERNS = {
    "picture books": ["frightening imagery of a storm and a lost child", "a depiction of a pet's death",
                      "crude toilet humour"],
    "early readers": ["a scene where a character is bullied", "mild swearing", "a frightening monster on several pages"],
    "juvenile": ["a battle scene with casualties", "a parent's death and grief", "strong language in dialogue",
                 "a character running away from home"],
    "teen": ["references to drug use", "a self-harm storyline", "strong language throughout", "a violent war setting",
             "a romantic scene"],
    "adult": ["graphic violence", "explicit sexual content", "a detailed depiction of a crime",
              "strong language throughout"],
}


def _title(rng):
    a, b = pick(rng, FIC_A), pick(rng, FIC_B)
    return pick(rng, [f"The {a} {b}", f"{b}s of {pick(rng, PLACES)}", f"A {b} Called {pick(rng, FIRSTS)}",
                      f"When the {b} Came Home", f"{pick(rng, FIRSTS)} and the {a} {b}", f"The Last {b} of {pick(rng, PLACES)}"])


NF_TITLES = ["The Story of {p}'s Canals", "Rivers of the World", "All About {b}s", "How Bridges Are Built",
             "A Year on the {p} Farm", "The Great Storm of 1953", "Kings, Castles and Coins", "Inside the Lighthouse",
             "The {p} Mills: A History", "Wonders of the Night Sky"]
JOURNALS_BY_BAND = {0: ["Picture Book Digest", "Children's Book Monthly", "The Young Readers' Bulletin"],
                    1: ["Picture Book Digest", "Children's Book Monthly", "The Young Readers' Bulletin"],
                    2: ["Children's Book Monthly", "The Young Readers' Bulletin", "Shelf Notes Review"],
                    3: ["Teen Shelf Review", "Shelf Notes Review", "Library Selection Quarterly"],
                    4: ["Shelf Notes Review", "Library Selection Quarterly", "The Reading Room Journal"]}


def _journals(rng, band, nonfiction=False, k=3):
    pool = list(JOURNALS_BY_BAND[band]) + (["The Nonfiction Index"] if nonfiction else [])
    rng.shuffle(pool)
    return pool[:k]


def _nf_title(rng):
    return pick(rng, NF_TITLES).format(p=pick(rng, PLACES), b=pick(rng, FIC_B))


def _kv(pairs, rng):
    return kv_lines(pairs, rng, style=pick(rng, ["colon", "colon", "aligned", "dash"]))


def _policy(rng, ref, clauses, json_ok=False):
    """The policy extract as prose, numbered clauses or bullets; on the json surface a clause list or a string."""
    style = pick(rng, ["prose", "numbered", "bullets", "prose"])
    if json_ok:
        return list(clauses) if style in ("numbered", "bullets") else f"{ref}: " + " ".join(clauses)
    if style == "numbered":
        return f"{ref}\n" + "\n".join(f"{chr(64 + i)}. {c}" for i, c in enumerate(clauses, 1))
    if style == "bullets":
        return f"{ref}:\n" + bullets(clauses, rng)
    return f"{ref}: " + " ".join(clauses)


def _plural(n, one, many=None):
    return f"{n} {one if n == 1 else (many or one + 's')}"


# =========================================================================== schema 1
OTHER_ITEMS = {"display": "the 'Summer Reads' display table at the {b} branch",
               "program": "the reading list used at the {b} branch's after-school story club",
               "database": "an online encyclopedia database the library subscribes to",
               "meeting_room": "a community group's booking of the {b} branch meeting room"}


class FormalChallengeIntake(Schema):
    name = "formal_challenge_intake"
    description = ("A formal Request for Reconsideration arrives: open a review, return the form, offer the informal "
                   "step, close it without review, or route it elsewhere, under the reconsideration policy.")
    surfaces = {"json": 0.5, "reconsideration_form": 0.25, "branch_email": 0.25}

    FIELDS = ["name and card number or address", "title and author", "specific passages or pages",
              "whether the whole work was read", "the action requested"]

    def sample_factors(self, rng):
        reg = regime(rng)
        coll = pick(rng, COLLECTIONS)
        other = pick(rng, list(OTHER_ITEMS)) if chance(rng, 0.14) else None
        card, resident = chance(rng, 0.8), chance(rng, 0.75)
        if chance(rng, {"clear": 0.2, "borderline": 0.2, "conflicting": 0.4}[reg]):
            card, resident = False, False
        present = [chance(rng, 0.85) for _ in self.FIELDS]
        if reg == "borderline":
            present = [True] * 5
            present[rng.randrange(5)] = chance(rng, 0.2)
        elif reg == "clear":
            present = [True] * 5 if chance(rng, 0.6) else [chance(rng, 0.5) for _ in self.FIELDS]
        read_whole = pick(rng, ["yes", "no", "partly"]) if present[3] else None
        informal = chance(rng, {"clear": 0.75, "borderline": 0.6, "conflicting": 0.5}[reg])
        last_review = None
        if chance(rng, 0.55):
            last_review = (near(rng, 3.0, 0.6, lo=0.3, digits=1) if reg == "borderline"
                           else pick(rng, [0.5, 1.0, 1.5, 2.0, 4.0, 5.5]))
        new_grounds = last_review is not None and chance(rng, 0.3)
        days = near(rng, 5, 2, lo=0, integer=True) if reg == "borderline" else pick(rng, [0, 1, 2, 3, 4, 6, 8])
        return {"regime": reg, "library": pick(rng, LIBRARIES), "branch": pick(rng, BRANCHES),
                "title": _title(rng), "author": person(rng), "collection": coll,
                "concern": pick(rng, CONCERNS[coll]), "other_item": other, "cardholder": card, "resident": resident,
                "requester": person(rng), "requester_role": pick(rng, ["parent", "patron", "teacher", "grandparent",
                                                                       "member of a local reading group"]),
                "fields_present": present, "read_whole": read_whole, "informal_offered": informal,
                "last_review_years": last_review, "new_grounds": new_grounds,
                "working_days_since_receipt": days, "acknowledged": chance(rng, 0.25),
                "holiday_in_window": chance(rng, 0.25), "received": rand_date(rng),
                "manager": person(rng), "ref": ident(rng, "RFR", 4, "-")}

    def _eligible(self, f):
        return f["cardholder"] or f["resident"]

    def _moratorium(self, f):
        return (f["other_item"] is None and f["last_review_years"] is not None and f["last_review_years"] < 3
                and not f["new_grounds"])

    def intended(self, f):
        n = sum(f["fields_present"])
        if f["other_item"]:
            step = "route_other_procedure"
        elif not self._eligible(f) or self._moratorium(f):
            step = "close_without_review"
        elif not f["informal_offered"]:
            step = "offer_informal_conversation_first"
        elif n < 5:
            step = "return_for_completion"
        else:
            step = "open_committee_review"
        comp = 3 if n == 5 else 2 if n == 4 else 1 if n == 3 else 0
        if f["acknowledged"]:
            pressure = 0
        else:
            left = 5 - f["working_days_since_receipt"]
            pressure = 0 if left >= 3 else 1 if left == 2 else 2 if left >= 0 else 3
        return {"intake_next_step": step, "request_completeness": comp, "response_deadline_pressure": pressure,
                "eligibility_met": "yes" if self._eligible(f) else "no",
                "moratorium_applies": "yes" if self._moratorium(f) else "no"}

    RULES = [
        "Displays, programmes, databases and meeting-room bookings are not collection items; concerns about them "
        "follow the separate Programmes and Spaces procedure.",
        "Any resident of the service area or any cardholder may file a request; others are thanked and the request is "
        "closed without review.",
        "A title reconsidered within the last 3 years is not reviewed again unless the request raises grounds the "
        "earlier review did not consider; such requests are closed without review.",
        "Before a formal request is processed, the requester must have been offered a conversation with the branch "
        "manager.",
        "A complete form gives the requester's name and card number or address, the title and author, the specific "
        "passages or pages, whether the whole work was read, and the action requested; incomplete forms are returned "
        "for completion.",
        "Every request is acknowledged in writing within 5 working days of receipt; the item stays on the shelf "
        "during any review.",
    ]

    def _item(self, f):
        if f["other_item"]:
            return OTHER_ITEMS[f["other_item"]].format(b=f["branch"])
        return f"\"{f['title']}\" by {f['author']} ({f['collection']} collection, {f['branch']} branch)"

    def _form_rows(self, f):
        pres = dict(zip(self.FIELDS, f["fields_present"]))
        read = {"yes": "Yes", "no": "No - read the pages a friend flagged", "partly": "About half"}.get(f["read_whole"])
        return [("Requester", f"{f['requester']} ({f['requester_role']})" if pres[self.FIELDS[0]] else f"{f['requester']} (no card number or address given)"),
                ("Card / residence", ("cardholder" if f["cardholder"] else "no library card") + ", "
                 + ("lives in the service area" if f["resident"] else "lives outside the service area")),
                ("Item", self._item(f) if pres[self.FIELDS[1]] else "title given, author left blank"),
                ("Passages / pages", f"concern: {f['concern']}; pages listed" if pres[self.FIELDS[2]] and not f["other_item"]
                 else ("concern described, no pages given" if not f["other_item"] else "n/a")),
                ("Read the whole work?", read or "left blank"),
                ("Action requested", pick_action(f) if pres[self.FIELDS[4]] else "left blank")]

    def _history(self, f):
        if f["last_review_years"] is None:
            return "no earlier reconsideration of this item"
        g = ("this request raises a concern the earlier review did not address" if f["new_grounds"]
             else "this request raises the same concern")
        y = f["last_review_years"]
        return f"last reconsidered {'a year' if y == 1 else f'{y:g} years'} ago (retained); {g}"

    def _timing(self, f):
        d = f["working_days_since_receipt"]
        hol = " (the branch was closed one extra day for a public holiday, already excluded)" if f["holiday_in_window"] else ""
        return (f"received {_plural(d, 'working day')} ago{hol}; "
                + ("acknowledgement sent" if f["acknowledged"] else "not yet acknowledged"))

    def render(self, f, rng, surface):
        informal = ("branch manager offered a conversation; requester asked to proceed formally" if f["informal_offered"]
                    else "no conversation with the branch manager has been offered yet")
        if surface == "json":
            return dumps(prune({
                "request": {"ref": f["ref"], "form": {k: v for k, v in self._form_rows(f)}},
                "status": {"informal_step": informal, "timing": self._timing(f), "history": self._history(f)},
                "library": {"name": f["library"], "policy": _policy(rng, "Reconsideration policy", self.RULES, True)},
            }))
        if surface == "reconsideration_form":
            return (f"{f['library'].upper()} - REQUEST FOR RECONSIDERATION ({f['ref']})\n"
                    f"{_kv(self._form_rows(f), rng)}\n\nOffice use: {informal}. {self._timing(f).capitalize()}. "
                    f"History: {self._history(f)}.\n\n" + _policy(rng, "Reconsideration policy", self.RULES))
        rows = "; ".join(f"{k}: {v}" for k, v in self._form_rows(f))
        body = [f"A reconsideration form ({f['ref']}) came in at {f['branch']}.", f"Form says - {rows}.",
                f"Status: {informal}; {self._timing(f)}.", f"History: {self._history(f)}.",
                "What's the next step?"]
        return "\n".join([f"Subject: Reconsideration request {f['ref']}", f"From: {f['manager']} ({f['branch']} branch manager)",
                          "To: Collection services", "", noisy(join_sentences(body), rng, 0.12), "",
                          _policy(rng, "Reconsideration policy", self.RULES)])

    def questions(self):
        return {
            "intake_next_step": choice("What is the correct next step with this request?", {
                "open_committee_review": "Accept it and send the item to the reconsideration committee.",
                "return_for_completion": "Return the form to the requester to complete the missing parts.",
                "offer_informal_conversation_first": "Offer the requester a conversation with the branch manager before anything else.",
                "close_without_review": "Acknowledge it and close it without a review, as the policy directs.",
                "route_other_procedure": "Send it to the separate procedure for things that are not collection items.",
            }),
            "request_completeness": score("How complete is the request form?", [
                "Two or more required parts are missing.",
                "Two required parts are missing.",
                "One required part is missing.",
                "Complete.",
            ]),
            "response_deadline_pressure": score("How close is the library to missing the policy's acknowledgement deadline?", [
                "Not pressing: already acknowledged, or three or more working days left.",
                "Two working days left.",
                "Due today or tomorrow.",
                "Already overdue.",
            ]),
            "eligibility_met": noul(
                "The requester is eligible to file a formal request under the policy.",
                yes="The requester holds a card or lives in the service area.",
                no="The requester neither holds a card nor lives in the service area."),
            "moratorium_applies": noul(
                "The policy's bar on reviewing a recently reconsidered title applies to this request."),
        }


def pick_action(f):
    """The action the requester asks for (deterministic from the factors)."""
    if f["other_item"]:
        return "remove it"
    return {"picture books": "move it out of the children's area", "early readers": "remove it",
            "juvenile": "move it to the teen collection", "teen": "move it to the adult collection",
            "adult": "remove it from the library"}[f["collection"]]


# =========================================================================== schema 2
class CommitteeReviewDecision(Schema):
    name = "committee_review_decision"
    description = ("The reconsideration committee has examined a challenged work: retain it, relocate it, withdraw "
                   "it, or defer, under the selection and placement criteria.")
    surfaces = {"json": 0.5, "committee_minutes": 0.25, "review_worksheet": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        coll = pick(rng, COLLECTIONS)
        idx = COLLECTIONS.index(coll)
        nonfic = chance(rng, 0.3)
        n_rev = pick_w(rng, {0: 1.5, 1: 2, 2: 3, 3: 2})
        fav = sum(chance(rng, 0.75) for _ in range(n_rev))
        award = pick_w(rng, {None: 6, "regional shortlist": 2, "state book award": 1})
        shift = pick_w(rng, {0: 6, 1: 2, -1: 2})
        if reg == "clear":
            if chance(rng, 0.5):
                n_rev, fav, shift = 3, 3, 0
            else:
                n_rev, fav, award = 0, 0, None
        rev_idx = min(4, max(0, idx + shift))
        accuracy = pick_w(rng, {"none": 6, "minor": 2, "significant": 2}) if nonfic else "none"
        members = pick(rng, [4, 5, 5, 6])
        read_full = members if chance(rng, 0.75) else members - pick(rng, [1, 1, 2])
        if reg == "conflicting":
            pick_c = pick(rng, ["popular_weak", "member_skimmed", "reviews_vs_band"])
            if pick_c == "popular_weak":
                n_rev, fav, award = pick(rng, [0, 1]), 0, None
            elif pick_c == "member_skimmed":
                read_full = members - 1
            else:
                rev_idx = min(4, max(0, idx + pick(rng, [-1, 1])))
        share = pick(rng, [2, 4, 8, 15, 30, 60])
        return {"regime": reg, "library": pick(rng, LIBRARIES), "title": _nf_title(rng) if nonfic else _title(rng),
                "author": person(rng), "collection": coll, "review_collection": COLLECTIONS[rev_idx], "nonfiction": nonfic,
                "concern": pick(rng, CONCERNS[coll]) if not nonfic else pick(rng, ["outdated statistics",
                                                                                     "a disputed historical account",
                                                                                     "an error-strewn timeline"]),
                "reviews": n_rev, "favourable": min(fav, n_rev), "award": award, "accuracy": accuracy,
                "checkouts_2y": pick(rng, [3, 11, 26, 48, 95, 160]), "members": members, "read_in_full": read_full,
                "challenged_pages_pct": share, "chair": person(rng), "branch": pick(rng, BRANCHES),
                "meeting": rand_date(rng), "journals": _journals(rng, rev_idx, nonfic), "ref": ident(rng, "RFR", 4, "-")}

    def _criteria_met(self, f):
        return (f["favourable"] >= 1 or f["award"] is not None) and f["accuracy"] != "significant"

    def intended(self, f):
        complete = f["read_in_full"] >= f["members"]
        if not complete:
            out = "defer_review"
        elif not self._criteria_met(f):
            out = "withdraw"
        elif f["review_collection"] != f["collection"] and f["reviews"] >= 2:
            out = "relocate"
        else:
            out = "retain_in_place"
        fit = (2 if f["favourable"] >= 2 else 1 if f["favourable"] == 1 else 0) + (1 if f["award"] else 0)
        if f["accuracy"] == "significant":
            fit = 0
        elif f["accuracy"] == "minor":
            fit = max(0, fit - 1)
        fit = min(3, fit)
        if f["reviews"] == 0:
            place = 1
        else:
            d = abs(COLLECTIONS.index(f["review_collection"]) - COLLECTIONS.index(f["collection"]))
            place = 3 if d == 0 and f["reviews"] >= 2 else 2 if d == 0 else 1 if d == 1 and f["reviews"] < 2 else 0
        return {"committee_outcome": out, "selection_criteria_fit": fit, "audience_placement_fit": place,
                "whole_work_considered": "yes" if complete else "no",
                "challenged_passages_central": "yes" if f["challenged_pages_pct"] >= 25 else "no"}

    RULES = [
        "The committee judges the work as a whole, not isolated passages; no decision may be issued until every "
        "voting member has read the whole work, otherwise the review is deferred.",
        "Selection criteria: at least one favourable review in a professional review journal or a recognised award, "
        "and, for nonfiction, no significant errors of fact. Circulation alone neither keeps nor removes a title.",
        "A work that fails the selection criteria is withdrawn.",
        "A work that meets the criteria is retained where it is; it is relocated when at least two professional "
        "reviews place its audience in a different age collection from the one it sits in.",
        "Age collections: picture books (0-5), early readers (5-7), juvenile (8-12), teen (13-17), adult (18+).",
    ]

    def _reviews_text(self, f):
        if f["reviews"] == 0:
            return "no professional reviews found"
        neg = f["reviews"] - f["favourable"]
        j = ", ".join(f["journals"][:f["reviews"]])
        return (f"{_plural(f['reviews'], 'professional review')} ({j}): {f['favourable']} favourable"
                + (f", {neg} unfavourable" if neg else "") + f"; reviewers place it at {BANDS[f['review_collection']]}")

    def _accuracy_text(self, f):
        if not f["nonfiction"]:
            return None
        return {"none": "fact-check by the subject librarian found no errors",
                "minor": "fact-check found two minor slips (a date and a spelling)",
                "significant": "fact-check found several significant factual errors in the main argument"}[f["accuracy"]]

    def _reading(self, f):
        if f["read_in_full"] >= f["members"]:
            return f"all {f['members']} voting members read the whole work"
        return (f"{f['read_in_full']} of {f['members']} voting members read the whole work; "
                f"{_plural(f['members'] - f['read_in_full'], 'member')} read only the challenged pages")

    def render(self, f, rng, surface):
        item = f"\"{f['title']}\" by {f['author']}"
        if surface == "json":
            return dumps(prune({
                "case": {"ref": f["ref"], "item": item, "type": "nonfiction" if f["nonfiction"] else "fiction",
                         "shelved_in": f"{f['collection']} ({BANDS[f['collection']]})", "challenge": f["concern"],
                         "challenged_pages_share_pct": f["challenged_pages_pct"]},
                "evidence": {"reviews": self._reviews_text(f), "awards": f["award"], "fact_check": self._accuracy_text(f),
                             "checkouts_last_2_years": f["checkouts_2y"]},
                "committee": {"chair": f["chair"], "reading": self._reading(f)},
                "policy": _policy(rng, "Selection and reconsideration policy", self.RULES, True),
            }))
        if surface == "committee_minutes":
            para = [f"Reconsideration committee, {fmt_date(f['meeting'], rng)}, chair {f['chair']}.",
                    f"Item: {item}, shelved in {f['collection']}. Challenge: {f['concern']} "
                    f"(about {f['challenged_pages_pct']}% of the pages).", f"Reading: {self._reading(f)}.",
                    f"Reviews: {self._reviews_text(f)}.", f"Awards: {f['award'] or 'none'}.",
                    (f"Fact-check: {self._accuracy_text(f)}." if f["nonfiction"] else ""),
                    f"Circulation: {f['checkouts_2y']} checkouts in two years.",
                    pick(rng, ["Discussion was lengthy.", "Members were divided.", "One member asked for more time.", ""])]
            return (f"MINUTES - {f['library']}\n\n{noisy(join_sentences(para), rng, 0.08)}\n\n"
                    + _policy(rng, "Selection and reconsideration policy", self.RULES))
        rows = [["Item", item], ["Shelved in", f"{f['collection']} ({BANDS[f['collection']]})"],
                ["Challenge", f"{f['concern']} (~{f['challenged_pages_pct']}% of pages)"],
                ["Reviews", self._reviews_text(f)], ["Awards", f["award"] or "none"],
                ["Fact-check", self._accuracy_text(f) or "n/a (fiction)"],
                ["Checkouts (2 yrs)", f["checkouts_2y"]], ["Reading", self._reading(f)]]
        return (f"REVIEW WORKSHEET {f['ref']} - {f['library']}\n" + table(["criterion", "finding"], rows, rng)
                + "\n\n" + _policy(rng, "Selection and reconsideration policy", self.RULES))

    def questions(self):
        return {
            "committee_outcome": choice("What should the committee decide?", {
                "retain_in_place": "Keep the work where it is.",
                "relocate": "Keep the work but move it to the age collection the reviews point to.",
                "withdraw": "Withdraw the work from the collection.",
                "defer_review": "Issue no decision yet; the review is not complete under the policy.",
            }),
            "selection_criteria_fit": score("How well does the work meet the selection criteria?", [
                "Fails them.",
                "Barely meets them.",
                "Meets them well.",
                "Meets them strongly, with recognition beyond a single review.",
            ]),
            "audience_placement_fit": score("How well does the current shelving match the audience the evidence points to?", [
                "Mismatched: well-supported evidence points to another collection.",
                "Doubtful: the evidence is thin or leans elsewhere.",
                "Matches on the evidence available.",
                "Clearly matches: several reviews agree with the current collection.",
            ]),
            "whole_work_considered": noul(
                "Every voting member has considered the work as a whole, as the policy requires before a decision.",
                yes="All voting members read the whole work.",
                no="At least one voting member read only part of it."),
            "challenged_passages_central": noul(
                "The challenged passages make up a substantial part of the work rather than a few pages."),
        }


# =========================================================================== schema 3
SPECIAL = {"parenting": "parenting and teachers' shelf", "local_history": "local history room"}


class RelocationRequest(Schema):
    name = "relocation_request"
    description = ("A request to move an item from one collection to another, judged against the placement rules on "
                   "publisher and reviewer age bands and consistent shelving.")
    surfaces = {"json": 0.5, "catalogue_record": 0.25, "staff_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        cur = pick(rng, COLLECTIONS)
        ci = COLLECTIONS.index(cur)
        special = None
        if chance(rng, 0.14):
            special = pick(rng, list(SPECIAL))
        pub = min(4, max(0, ci + pick_w(rng, {0: 4, 1: 2, -1: 2})))
        r1 = min(4, max(0, pub + pick_w(rng, {0: 4, 1: 2, -1: 2})))
        r2 = None if chance(rng, 0.3) else min(4, max(0, pub + pick_w(rng, {0: 4, 1: 2, -1: 2})))
        if reg == "conflicting":
            if chance(rng, 0.5):                     # both reviewers depart from the publisher, the same way
                r1 = r2 = min(4, max(0, pub + pick(rng, [-1, 1])))
            else:                                    # reviewers split either side of the publisher's band
                pub = min(3, max(1, pub))
                r1, r2 = pub + 1, pub - 1
        elif reg == "clear":
            if chance(rng, 0.5):
                pub = r1 = r2 = ci
            else:
                pub = r1 = r2 = min(4, max(0, ci + pick(rng, [-1, 1])))
        others = pick(rng, [0, 1, 2, 3, 4, 5])
        same = between(rng, 0, others, integer=True) if others else 0
        local_rec = chance(rng, 0.55)
        new_staff = local_rec and chance(rng, 0.5)
        if new_staff and others and chance(rng, 0.7):
            same = pick(rng, [0, 0, 1]) if others > 1 else 0     # a fresh local record often strays from the rest
        return {"regime": reg, "library": pick(rng, LIBRARIES), "branch": pick(rng, BRANCHES), "title": _title(rng),
                "author": person(rng), "current": cur, "special": special, "pub_idx": pub, "r1_idx": r1, "r2_idx": r2,
                "reading_grade": max(0, min(12, pub * 3 + between(rng, -1, 2, integer=True))),
                "other_copies": others, "other_copies_same": same, "local_record": local_rec,
                "record_by_new_staff": new_staff, "journals": _journals(rng, ci, False, 2),
                "requester": person(rng), "requester_role": pick(rng, ["parent", "teacher", "children's librarian",
                                                                       "patron", "branch volunteer"]),
                "requested": None, "ref": ident(rng, "MV", 4, "-"),
                "format": pick(rng, ["picture book", "graphic novel", "illustrated novel", "chapter book", "novel",
                                     "nonfiction title"])}

    def _target(self, f):
        if f["special"]:
            return f["special"]
        if f["r2_idx"] is not None and f["r1_idx"] == f["r2_idx"]:
            return COLLECTIONS[f["r1_idx"]]
        return COLLECTIONS[f["pub_idx"]]

    def intended(self, f):
        t = self._target(f)
        if t == f["current"]:
            ruling = "keep_current_location"
        elif t in SPECIAL:
            ruling = "move_to_non_age_collection"
        elif COLLECTIONS.index(t) > COLLECTIONS.index(f["current"]):
            ruling = "move_to_older_collection"
        else:
            ruling = "move_to_younger_collection"
        bands = [f["pub_idx"], f["r1_idx"]] + ([f["r2_idx"]] if f["r2_idx"] is not None else [])
        spread = max(bands) - min(bands)
        agree = 3 if spread == 0 and len(bands) == 3 else 2 if spread == 0 else 1 if spread == 1 else 0
        if f["other_copies"] == 0:
            cons = 1
        else:
            r = f["other_copies_same"] / f["other_copies"]
            cons = 3 if r == 1 else 2 if r > 0.5 else 1 if r > 0 else 0
        err = f["other_copies"] > 0 and f["other_copies_same"] / f["other_copies"] < 0.5 and f["local_record"]
        return {"placement_ruling": ruling, "designation_agreement": agree, "consistency_with_comparables": cons,
                "designation_supports_move": "yes" if t != f["current"] else "no",
                "cataloguing_error_likely": "yes" if err else "no"}

    RULES = [
        "Age collections: picture books (0-5), early readers (5-7), juvenile (8-12), teen (13-17), adult (18+).",
        "Shelve a title in the collection matching the publisher's age band, unless two professional reviews agree "
        "on a different band, in which case follow the reviews.",
        "Works written for parents or teachers go to the parenting and teachers' shelf, and works of local interest "
        "to the local history room, whatever their format.",
        "Copies of the same title should be shelved alike across branches; a copy shelved differently from most "
        "others, under a locally created catalogue record, is checked for a cataloguing error.",
        "Requests to move an item are decided on these rules, not on the requester's view of the content.",
    ]

    def _bands(self, f):
        out = [f"publisher: {BANDS[COLLECTIONS[f['pub_idx']]]}",
               f"{f['journals'][0]}: {BANDS[COLLECTIONS[f['r1_idx']]]}"]
        out.append(f"{f['journals'][1]}: {BANDS[COLLECTIONS[f['r2_idx']]]}" if f["r2_idx"] is not None
                   else f"{f['journals'][1]}: not reviewed")
        return out

    def _special_note(self, f):
        return {"parenting": "the book is written for parents, explaining how to talk to children about a new sibling",
                "local_history": "the book is a history of the town's mills, with old photographs"}.get(f["special"])

    def _holdings(self, f):
        if f["other_copies"] == 0:
            return "no other copies in the system"
        diff = f["other_copies"] - f["other_copies_same"]
        return (f"{_plural(f['other_copies'], 'other copy', 'other copies')} in the system: "
                f"{f['other_copies_same']} shelved in {f['current']}, {diff} elsewhere")

    def _record(self, f):
        if not f["local_record"]:
            return "catalogue record copied from the national bibliographic vendor"
        return "locally created catalogue record" + (" (entered by a staff member in their first month)"
                                                     if f["record_by_new_staff"] else "")

    def render(self, f, rng, surface):
        item = f"\"{f['title']}\" by {f['author']}"
        ask = pick(rng, ["move it somewhere more suitable", "move it to a more appropriate section",
                         "find a better place for it"])
        if surface == "json":
            return dumps(prune({
                "request": {"ref": f["ref"], "from": f"{f['requester']} ({f['requester_role']})", "asks": ask,
                            "item": item, "format": f["format"], "currently_in": f"{f['current']} at {f['branch']}"},
                "evidence": {"age_bands": self._bands(f), "reading_level": f"grade {f['reading_grade']}",
                             "content_note": self._special_note(f), "holdings": self._holdings(f), "record": self._record(f)},
                "policy": _policy(rng, "Placement rules", self.RULES, True),
            }))
        if surface == "catalogue_record":
            rows = _kv([("Title", f["title"]), ("Author", f["author"]), ("Format", f["format"]),
                        ("Location", f"{f['branch']} - {f['current']}"), ("Record", self._record(f)),
                        ("Audience notes", "; ".join(self._bands(f))), ("Reading level", f"grade {f['reading_grade']}"),
                        ("Summary note", self._special_note(f)), ("Holdings", self._holdings(f))], rng)
            return (f"CATALOGUE RECORD + MOVE REQUEST {f['ref']} ({f['library']})\n{rows}\n"
                    f"Request from {f['requester']} ({f['requester_role']}): {ask}.\n\n"
                    + _policy(rng, "Placement rules", self.RULES))
        body = [f"{f['requester']} ({f['requester_role']}) has asked us to {ask}: {item}, a {f['format']} now in "
                f"{f['current']} at {f['branch']}.", "Age bands - " + "; ".join(self._bands(f)) + ".",
                f"Reading level grade {f['reading_grade']}.", (self._special_note(f) or "").capitalize() + ("." if f["special"] else ""),
                f"{self._holdings(f).capitalize()}.", f"{self._record(f).capitalize()}."]
        return "\n".join([f"Subject: Move request {f['ref']}", "From: collection services", "",
                          noisy(join_sentences(body), rng, 0.1), "", _policy(rng, "Placement rules", self.RULES)])

    def questions(self):
        return {
            "placement_ruling": choice("Where should the item be shelved?", {
                "keep_current_location": "Where it is now.",
                "move_to_older_collection": "In an older age collection.",
                "move_to_younger_collection": "In a younger age collection.",
                "move_to_non_age_collection": "In a collection not defined by age, such as the parenting shelf or the local history room.",
            }),
            "designation_agreement": score("How far do the publisher's and reviewers' age bands agree?", [
                "They spread across more than one neighbouring band.",
                "They differ by one band.",
                "The available sources agree, but there are only two.",
                "All three sources agree.",
            ]),
            "consistency_with_comparables": score("How consistent is the current location with where the system's other copies sit?", [
                "Inconsistent: every other copy is shelved elsewhere.",
                "Mostly inconsistent, or nothing to compare with.",
                "Mostly consistent.",
                "Fully consistent with every other copy.",
            ]),
            "designation_supports_move": noul(
                "Under the placement rules, the item belongs in a different collection from the one it is in.",
                yes="The rules' deciding band, or a parenting or local-history purpose, points to another collection.",
                no="The rules point to the collection it is already in."),
            "cataloguing_error_likely": noul(
                "The current location is more likely a cataloguing error than a deliberate placement."),
        }


# =========================================================================== schema 4
# subject: (label, currency limit in years or None, physical-harm subject?, titles)
# subject: (label, currency limit in years or None, physical-harm subject?, [(title, the challenger's claim)])
SUBJECTS = {
    "health": ("health and medicine", 5, True, [
        ("Managing Childhood Fevers", "its dosage table for children's fever medicine no longer matches current guidance"),
        ("The Family Guide to Common Medicines", "it lists a cough remedy for toddlers that is no longer advised"),
        ("First Aid at Home", "its advice on treating burns contradicts current first-aid guidance"),
        ("Living Well with Diabetes", "its insulin-adjustment advice is out of date")]),
    "law": ("law and legal self-help", 3, False, [
        ("Renting Without Tears", "the notice periods for ending a tenancy changed last year"),
        ("Your Small-Claims Case", "the court fees and claim limits it gives have changed"),
        ("Wills and Probate Explained", "the probate thresholds it quotes are out of date")]),
    "finance": ("personal finance and tax", 2, False, [
        ("The Household Tax Handbook", "the tax thresholds and allowances in it are out of date"),
        ("Pensions Made Simple", "the pension contribution limits it gives have changed"),
        ("Saving for Your First Home", "the savings scheme it recommends has closed to new savers")]),
    "tech": ("computing and technology", 3, False, [
        ("Mastering Your Home Network", "the router software it describes has been replaced"),
        ("Spreadsheets for Everyone", "the menus and screenshots no longer match the current program"),
        ("Getting Started with Photo Editing", "the editing program it teaches is no longer sold")]),
    "travel": ("travel guides", 5, False, [
        ("Coastal Walks of the Northern Isles", "two of the walks cross land that has since been closed to the public"),
        ("The Lakes and Valleys Guidebook", "the ferry and bus times it gives no longer run"),
        ("City Breaks on a Budget", "half the hostels it lists have closed")]),
    "preserving": ("home food preservation", 10, True, [
        ("The Complete Home Canning Book", "its processing times for low-acid vegetables are shorter than current tested times"),
        ("Pickles, Jams and Preserves", "it recommends sealing jars with wax, which current guidance advises against"),
        ("Bottling the Harvest", "its method for bottling tomatoes skips the acid step now required")]),
    "electrical": ("home electrical work", 5, True, [
        ("Wiring Your Home Safely", "it describes wiring practices that current regulations no longer allow"),
        ("The DIY Electrics Manual", "it tells readers to work on circuits current guidance reserves for electricians")]),
    "history": ("local and general history", None, False, [
        ("A History of the {p} Mills", "its account of the mill strike is one-sided"),
        ("Voices from the Harbour", "it gets the date of the harbour fire wrong"),
        ("The Railway Comes to {p}", "it misnames the engineer who built the viaduct")]),
}


class AccuracyCurrencyChallenge(Schema):
    name = "accuracy_currency_challenge"
    description = ("A nonfiction title is challenged as outdated or wrong: retain, add a current title, replace with "
                   "a newer edition, withdraw, or move to a historical collection, under the currency policy.")
    surfaces = {"json": 0.5, "weeding_worksheet": 0.25, "challenge_letter": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        subj = pick_w(rng, {"health": 2, "law": 1.5, "finance": 1.5, "tech": 1, "travel": 1.5, "preserving": 1.5,
                            "electrical": 1, "history": 1})
        label, limit, phys, titles = SUBJECTS[subj]
        title, claim = pick(rng, titles)
        if limit is None:
            age = pick(rng, [3, 9, 16, 25, 40])
        elif reg == "borderline":
            age = max(1, near(rng, limit + 0.5, 2.0, integer=True))
        elif reg == "clear":
            age = pick(rng, [max(1, limit - 2), max(1, limit - 2), limit + 1, limit + 2, limit + 4, limit + 7])
        else:
            age = between(rng, 1, limit * 2 + 3, integer=True)
        past = limit is not None and age > limit
        if reg == "clear":
            verify = "superseded" if past else "accurate"
        elif reg == "conflicting":
            verify = pick(rng, ["accurate", "superseded"]) if past else pick(rng, ["superseded", "wrong_from_start"])
        else:
            verify = pick_w(rng, {"accurate": 3, "superseded": 3, "wrong_from_start": 1})
        if subj == "history":
            verify = pick(rng, ["accurate", "accurate", "wrong_from_start"])
        newer = chance(rng, 0.45)
        hist = chance(rng, 0.2) and subj in ("travel", "history", "tech", "preserving", "law")
        return {"regime": reg, "library": pick(rng, LIBRARIES), "branch": pick(rng, BRANCHES), "subject": subj,
                "title": title.format(p=pick(rng, PLACES)), "claim": claim, "author": person(rng),
                "pub_year": 2026 - age, "age_years": age, "verification": verify, "newer_edition": newer,
                "current_title_available": newer or chance(rng, 0.5), "historical_value": hist,
                "checkouts_2y": pick(rng, [0, 2, 7, 15, 34, 70]), "challenger": person(rng),
                "challenger_role": pick(rng, ["patron", "nurse", "retired solicitor", "local tour guide", "teacher",
                                              "staff member", "allotment society member"]),
                "librarian": person(rng), "ref": ident(rng, "WD", 4, "-")}

    def intended(self, f):
        label, limit, phys, _ = SUBJECTS[f["subject"]]
        past = limit is not None and f["age_years"] > limit
        contradicts = f["verification"] in ("superseded", "wrong_from_start")
        if f["historical_value"] and (past or contradicts) and not (phys and contradicts):
            out = "move_to_reference_history"
        elif contradicts or past:
            out = "replace_with_newer_edition" if f["newer_edition"] else "withdraw"
        elif limit is not None and f["age_years"] >= limit - 1 and f["current_title_available"]:
            out = "retain_and_add_current"
        else:
            out = "retain_as_is"
        if f["verification"] == "wrong_from_start":
            prob = "wrong_from_start"
        elif f["verification"] == "superseded":
            prob = "superseded_guidance"
        elif past or (limit is not None and f["age_years"] >= limit - 1):
            prob = "dated_but_sound"
        else:
            prob = "nothing_wrong"
        if not contradicts:
            harm = 0
        elif phys:
            harm = 3
        elif f["subject"] in ("law", "finance"):
            harm = 2
        else:
            harm = 1
        over = 0 if limit is None or f["age_years"] <= limit else f["age_years"] - limit
        gap = 0 if over == 0 else 1 if over <= 1 else 2 if over <= 3 else 3
        return {"currency_outcome": out, "currency_problem": prob, "misinformation_harm": harm, "currency_gap": gap,
                "newer_edition_available": "yes" if f["newer_edition"] else "no"}

    RULES = [
        "Currency limits from publication: health and medicine 5 years; law 3; personal finance and tax 2; computing "
        "3; travel 5; home food preservation 10; home electrical work 5; history and literature have no limit.",
        "A title past its limit, or whose content now contradicts current authoritative sources, is replaced by its "
        "newer edition where one exists, and otherwise withdrawn.",
        "Titles with documented local or historical value move to the non-circulating reference and local history "
        "collection instead, unless following their content could cause physical harm.",
        "A title within its limit and still accurate is kept; where it is within a year of its limit and a current "
        "title on the subject is available, keep it and add the current title.",
        "Interpretations in history and literature are not grounds for withdrawal; demonstrable factual errors are.",
    ]

    def _verify_text(self, f):
        return {"accurate": "the subject librarian checked current authoritative sources: the challenged content still matches them",
                "superseded": "the subject librarian checked current authoritative sources: the challenged content was right when published but has since been superseded",
                "wrong_from_start": "the subject librarian checked: the challenged content was already wrong when the book was published"}[f["verification"]]

    def render(self, f, rng, surface):
        label = SUBJECTS[f["subject"]][0]
        item = f"\"{f['title']}\" by {f['author']} ({f['pub_year']})"
        newer = "a newer edition is in print" if f["newer_edition"] else "no newer edition exists"
        cur = ("a current title on the subject is available to buy" if f["current_title_available"]
               else "no suitable current title has been identified")
        hist = ("the local history librarian notes the book documents the area as it was and is cited in local research"
                if f["historical_value"] else None)
        if surface == "json":
            return dumps(prune({
                "challenge": {"ref": f["ref"], "from": f"{f['challenger']} ({f['challenger_role']})",
                              "claim": f["claim"]},
                "item": {"title": item, "subject": label, "branch": f["branch"], "checkouts_last_2_years": f["checkouts_2y"]},
                "findings": {"verification": self._verify_text(f), "editions": newer, "alternatives": cur,
                             "historical_value": hist},
                "policy": _policy(rng, "Collection currency policy", self.RULES, True),
            }))
        if surface == "weeding_worksheet":
            rows = _kv([("Ref", f["ref"]), ("Item", item), ("Subject", label), ("Branch", f["branch"]),
                        ("Checkouts (2 yrs)", f["checkouts_2y"]), ("Challenge", f["claim"]),
                        ("Raised by", f"{f['challenger']} ({f['challenger_role']})"),
                        ("Verification", self._verify_text(f)), ("Editions", newer), ("Alternatives", cur),
                        ("Historical value", hist or "none noted"), ("Librarian", f["librarian"])], rng)
            return f"WEEDING / CURRENCY WORKSHEET - {f['library']}\n{rows}\n\n" + _policy(rng, "Collection currency policy", self.RULES)
        body = [f"I borrowed {item} from the {f['branch']} branch and was concerned to find that {f['claim']}.",
                pick(rng, ["Someone could easily rely on it.", "Surely this should not be on the shelves.",
                           "Please could someone look at it.", ""])]
        notes = _kv([("Verification", self._verify_text(f)), ("Editions", newer), ("Alternatives", cur),
                     ("Historical value", hist or "none noted"), ("Checkouts (2 yrs)", f["checkouts_2y"])], rng)
        return "\n".join([f"Subject: Out-of-date book at {f['branch']}", f"From: {f['challenger']} ({f['challenger_role']})", "",
                          noisy(join_sentences(body), rng, 0.2), "", "--- collection services notes ---", notes, "",
                          _policy(rng, "Collection currency policy", self.RULES)])

    def questions(self):
        return {
            "currency_outcome": choice("What should happen to the title?", {
                "retain_as_is": "Keep it as it is.",
                "retain_and_add_current": "Keep it and buy a current title on the subject alongside it.",
                "replace_with_newer_edition": "Replace it with its newer edition.",
                "withdraw": "Withdraw it from the collection.",
                "move_to_reference_history": "Move it to the non-circulating reference and local history collection.",
            }),
            "currency_problem": choice("What, if anything, is wrong with the title's content?", {
                "superseded_guidance": "It was right when published but current sources have moved on.",
                "wrong_from_start": "It contained the error when it was published.",
                "dated_but_sound": "It is old, but what it says still holds.",
                "nothing_wrong": "Nothing: it is recent enough and accurate.",
            }),
            "misinformation_harm": score("How much harm could a reader suffer by relying on the challenged content?", [
                "None: the content is accurate.",
                "Inconvenience: wasted time or a wrong turn.",
                "Financial or legal cost.",
                "Physical harm.",
            ]),
            "currency_gap": score("How far past the policy's currency limit for its subject is the title?", [
                "Not past it, or no limit applies.",
                "Up to a year past.",
                "One to three years past.",
                "More than three years past.",
            ]),
            "newer_edition_available": noul("A newer edition of this title exists."),
        }


# =========================================================================== schema 5
GROUNDS = {
    "procedural": "procedural_error", "merits": "disagrees_with_merits", "new_info": "new_information",
    "out_of_scope": "outside_the_decision",
}


class ReconsiderationAppeal(Schema):
    name = "reconsideration_appeal"
    description = ("An appeal to the library board against a reconsideration decision: uphold, remand, modify or "
                   "dismiss, under the appeal procedure.")
    surfaces = {"json": 0.5, "appeal_letter": 0.25, "board_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        decision = pick(rng, ["retained", "retained", "relocated", "withdrawn"])
        ground = pick_w(rng, {"procedural": 3, "merits": 3, "new_info": 2, "out_of_scope": 1})
        days = near(rng, 30, 3, lo=1, integer=True) if reg == "borderline" else pick(rng, [6, 12, 19, 25, 34, 45])
        defect = pick_w(rng, {"none": 4, "late_decision": 2, "member_not_read": 2, "undisclosed_conflict": 1,
                              "no_notice_of_appeal_right": 1})
        if ground == "procedural" and reg != "conflicting" and defect == "none" and chance(rng, 0.6):
            defect = pick(rng, ["late_decision", "member_not_read", "undisclosed_conflict"])
        info_new = ground == "new_info" and chance(rng, 0.6)
        policy_breach = chance(rng, {"clear": 0.1, "borderline": 0.15, "conflicting": 0.3}[reg])
        past = pick_w(rng, {"consistent": 5, "mixed": 2, "inconsistent": 2})
        return {"regime": reg, "library": pick(rng, LIBRARIES), "title": _title(rng), "author": person(rng),
                "collection": pick(rng, COLLECTIONS), "decision": decision, "ground": ground,
                "days_after_letter": days, "defect": defect, "late_by_days": pick(rng, [2, 5, 12, 20]),
                "info_new": info_new, "policy_breach": policy_breach, "past_decisions": past,
                "appellant": person(rng), "appellant_role": pick(rng, ["original requester", "original requester",
                                                                       "parent", "patron"]),
                "board_chair": person(rng), "director": person(rng), "letter_date": rand_date(rng),
                "ref": ident(rng, "APL", 3, "-")}

    def _timely(self, f):
        return f["days_after_letter"] <= 30

    def _material_defect(self, f):
        return f["defect"] in ("member_not_read", "undisclosed_conflict")

    def intended(self, f):
        g = f["ground"]
        if not self._timely(f) or g == "out_of_scope":
            act = "dismiss_appeal"
        elif g == "procedural" and self._material_defect(f):
            act = "remand_to_committee"
        elif g == "new_info" and f["info_new"]:
            act = "remand_to_committee"
        elif f["policy_breach"]:
            act = "modify_decision"
        else:
            act = "uphold_decision"
        pd = {"none": 0, "late_decision": 1, "no_notice_of_appeal_right": 1, "member_not_read": 2,
              "undisclosed_conflict": 3}[f["defect"]]
        if f["policy_breach"]:
            cons = 0
        else:
            cons = {"consistent": 3, "mixed": 2, "inconsistent": 1}[f["past_decisions"]]
        return {"board_action": act, "appeal_ground": GROUNDS[g], "process_defects": pd,
                "decision_consistency": cons, "appeal_timely": "yes" if self._timely(f) else "no"}

    RULES = [
        "An appeal must reach the board within 30 days of the date on the committee's decision letter.",
        "The board does not vote again on the merits of the work.",
        "It sends a decision back to the committee where a procedural defect could have affected the outcome (a "
        "voting member who did not read the whole work, an undisclosed conflict of interest) or where the appeal "
        "brings information the committee could not have considered.",
        "It changes a decision only where the decision itself breaches the written selection or placement criteria.",
        "Otherwise it upholds the decision; a late decision or a defective notice is noted but does not by itself "
        "change the outcome.",
        "Complaints about other matters (staff conduct, opening hours) go to the director, and the appeal is dismissed.",
    ]

    def _appeal_text(self, f):
        g = f["ground"]
        if g == "procedural":
            return {"none": "The committee did not follow its own procedure; the process was rushed and unfair.",
                    "late_decision": f"The decision came {f['late_by_days']} days after the policy's deadline.",
                    "member_not_read": "One committee member admitted at the meeting to having read only the passages I listed.",
                    "undisclosed_conflict": "A committee member is related to someone involved in publishing the book, and said nothing.",
                    "no_notice_of_appeal_right": "The decision letter never told me I could appeal."}[f["defect"]]
        if g == "merits":
            return "The committee simply got it wrong. Anyone who reads the book can see it does not belong there."
        if g == "new_info":
            return ("Since the decision a revised edition has been published that drops the passages in question; "
                    "the committee could not have known." if f["info_new"]
                    else "The committee ignored the review in the regional paper that I attached to my original form.")
        return "Staff at the branch were dismissive when I raised this, and the branch keeps changing its opening hours."

    def _findings(self, f):
        d = {"none": "the file shows the written procedure was followed",
             "late_decision": f"the decision was issued {f['late_by_days']} days after the 45-day deadline; all members read the work",
             "member_not_read": "the minutes record that one voting member read only the challenged pages",
             "undisclosed_conflict": "a voting member's close relative works for the book's publisher; it was not declared",
             "no_notice_of_appeal_right": "the decision letter omitted the paragraph on the right of appeal"}[f["defect"]]
        b = (" The committee's written reasons conflict with the placement criteria: they moved the title against "
             "two agreeing reviews." if f["policy_breach"] else "")
        p = {"consistent": "similar challenges in the last five years were decided the same way",
             "mixed": "similar challenges were decided both ways", "inconsistent":
             "two similar recent challenges were decided the other way"}[f["past_decisions"]]
        return f"Clerk's file note: {d}.{b} Precedent: {p}."

    def render(self, f, rng, surface):
        item = f"\"{f['title']}\" by {f['author']}"
        received = f["letter_date"] + timedelta(days=f["days_after_letter"])
        if surface == "json":
            return dumps(prune({
                "appeal": {"ref": f["ref"], "appellant": f"{f['appellant']} ({f['appellant_role']})",
                           "grounds": self._appeal_text(f),
                           "decision_letter_dated": fmt_date(f["letter_date"], style="iso"),
                           "appeal_received": fmt_date(received, style="iso")},
                "decision_appealed": {"item": item, "collection": f["collection"], "committee_decision": f["decision"]},
                "file": self._findings(f),
                "policy": _policy(rng, "Board appeal procedure", self.RULES, True),
            }))
        if surface == "appeal_letter":
            ds = pick(rng, ["long", "us", "eu", "iso"])
            return "\n".join([f"To the Board of {f['library']}", f"Received: {fmt_date(received, style=ds)} (stamp)", "",
                              noisy(f"I appeal against the decision dated {fmt_date(f['letter_date'], style=ds)} that "
                                    f"{item} be {f['decision']}. {self._appeal_text(f)}", rng, 0.2), "",
                              f"{f['appellant']}", "", self._findings(f), "",
                              _policy(rng, "Board appeal procedure", self.RULES)])
        para = [f"Appeal {f['ref']} from {f['appellant']} ({f['appellant_role']}) against the decision that {item} "
                f"({f['collection']}) be {f['decision']}.",
                f"Decision letter dated {fmt_date(f['letter_date'], rng)}; appeal received {fmt_date(received, rng)}.",
                f"The appellant says: \"{self._appeal_text(f)}\"", self._findings(f)]
        return (f"MEMO to {f['board_chair']}, Board chair, from {f['director']}, Director\n\n{join_sentences(para)}\n\n"
                + _policy(rng, "Board appeal procedure", self.RULES))

    def questions(self):
        return {
            "board_action": choice("What should the board do with the appeal?", {
                "uphold_decision": "Uphold the committee's decision.",
                "remand_to_committee": "Send the decision back to the committee to reconsider.",
                "modify_decision": "Change the decision because it breaches the written criteria.",
                "dismiss_appeal": "Dismiss the appeal as out of time or not about the decision.",
            }),
            "appeal_ground": choice("What does the appeal, at bottom, rely on?", {
                "procedural_error": "A failure to follow the written procedure.",
                "disagrees_with_merits": "Disagreement with the committee's judgment of the work.",
                "new_information": "Information the appellant says the committee did not consider.",
                "outside_the_decision": "Matters other than the decision itself.",
            }),
            "process_defects": score("How serious were the departures from the written procedure, on the file?", [
                "None.",
                "Minor: a late or incomplete letter, without effect on the decision.",
                "Significant: a step that bears on the decision was skipped.",
                "Serious: the decision may have been tainted by an undeclared interest.",
            ]),
            "decision_consistency": score("How consistent is the committee's decision with the policy and with past decisions?", [
                "It conflicts with the written criteria themselves.",
                "It is within the criteria but out of line with recent similar decisions.",
                "Broadly in line; precedent is mixed.",
                "Fully consistent with the criteria and with precedent.",
            ]),
            "appeal_timely": noul(
                "The appeal reached the board within the policy's deadline.",
                yes="It was received within 30 days of the decision letter's date.",
                no="It was received more than 30 days after the decision letter's date."),
        }


WORKFLOW = Workflow(
    name="library_material_reconsideration",
    title="Library material reconsideration",
    description=("Challenges to library materials judged against a written reconsideration policy: intake, "
                 "committee decisions, relocation, currency and board appeals."),
    domain="public institutions / public libraries",
    schemas=[FormalChallengeIntake(), CommitteeReviewDecision(), RelocationRequest(), AccuracyCurrencyChallenge(),
             ReconsiderationAppeal()],
    notes="Fictional libraries, titles and authors; viewpoint-neutral content descriptions; decisions turn on "
          "process and written criteria, never on a viewpoint or a protected attribute.",
)
