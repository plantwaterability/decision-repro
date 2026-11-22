"""Museum acquisition and provenance review: purchases, gifts, loans, disposals, restitution claims and export
licences, judged against a museum's written policy and the export laws of FICTIONAL source states.

Six schemas: a purchase proposal's due diligence, a gift offer, an incoming exhibition loan with a wartime
provenance gap, a deaccession proposal, a restitution claim, and a national export-licence review. Every state quotes
the policy clauses or statutes it is judged against, with the variable parts sampled per case (the cut-off year, the
seizure-immunity notice period, the proceeds rule, the value and age thresholds), so each answer comes from setting
the provenance facts against the rule.

Each schema draws a `regime` first ('clear' / 'borderline' / 'conflicting'). Borderline cases sit on a line (an
object first documented abroad in the cut-off year itself, a provenance note that can be published a week before the
immunity deadline); conflicting cases pair an attractive object with a weak paper trail, or a sympathetic claim with
an uncertain identification. `intended(f)` records the answer logic a careful registrar or committee would apply to
the RENDERED facts; the builder does not use it.

Museums, dealers, auction houses, collectors, claimants and source states are fictional. Claims are judged on
evidence of title, theft, forced sale and illicit export, never on who the claimant is; no protected attribute is a
factor.
"""
from __future__ import annotations

import re
from datetime import timedelta

from .base import (
    Schema,
    Workflow,
    bullets,
    chance,
    choice,
    dumps,
    fmt_date,
    ident,
    join_sentences,
    kv_lines,
    lognormal_amount,
    money,
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

MUSEUMS = ["Halloway Museum of Art", "Corran Valley Museum", "Westerly Collection", "Museum of the Northern Coast",
           "Sallowmere Gallery Trust", "Fenmoor Museum of World Cultures", "Larkspur Art Institute",
           "Tidemark Historical Museum", "Greywater Museum of Antiquities", "Brightwell Gallery Trust",
           "Marrowby Civic Museum", "Quillon House Collection"]
DEALERS = ["Ashdown & Vell Fine Art", "Galerie Morvane", "Tessaly Antiquities", "Orme Street Gallery", "Cardew Oriental Art",
           "Haldane Tribal Art", "Lysander Old Masters", "Weir & Calloway", "Galleria Sorbel", "Pennick Antiques"]
AUCTIONS = ["Brannock's", "Halsey & Moor", "Auktionshaus Vielmann", "Thorne Auctioneers", "Castellane Ventes"]
# fictional source states: (name, adjective, year its cultural-property export law took effect)
STATES = [("Kaldoria", "Kaldorian", 1934), ("Sarnova", "Sarnovan", 1956), ("Tamaresh", "Tamareshi", 1962),
          ("Orsenia", "Orsenian", 1971), ("Velenmark", "Velenish", 1948), ("Qareth", "Qarethi", 1983),
          ("Bastovia", "Bastovian", 1925), ("Ilmaran", "Ilmarani", 1966)]
# (object, category, approximate date text)
OBJECTS = [
    ("bronze oil lamp", "archaeological", "2nd century BCE"), ("terracotta female figure", "archaeological", "c. 600 BCE"),
    ("limestone stele fragment with inscription", "archaeological", "1st century CE"),
    ("painted ceramic bowl", "archaeological", "c. 1200 BCE"), ("gold diadem", "archaeological", "4th century BCE"),
    ("marble head of a youth", "archaeological", "2nd century CE"), ("glass perfume flask", "archaeological", "3rd century CE"),
    ("bronze helmet", "archaeological", "5th century BCE"), ("inscribed clay tablet", "archaeological", "c. 1800 BCE"),
    ("mosaic panel with birds", "archaeological", "4th century CE"),
    ("carved wooden mask", "ethnographic", "late 19th century"), ("beaded ceremonial belt", "ethnographic", "early 20th century"),
    ("woven funerary textile", "ethnographic", "15th century"), ("carved ancestor figure", "ethnographic", "19th century"),
    ("river landscape, oil on panel", "painting", "1640s"), ("portrait of a merchant, oil on canvas", "painting", "1780s"),
    ("illuminated manuscript leaf", "manuscript", "c. 1450"), ("harbour at dusk, oil on canvas", "painting", "1890s"),
    ("still life with lemons, oil on canvas", "painting", "1920s"), ("ink drawing of a cathedral", "drawing", "1830s"),
    ("bronze dancer, cast sculpture", "sculpture", "1910s"), ("silver tea service", "decorative", "1760s"),
    ("lacquered writing box", "decorative", "18th century"), ("porcelain vase with dragons", "decorative", "17th century"),
    ("tapestry fragment with hunters", "decorative", "16th century"),
]
EUROPEAN_ART = {"painting", "drawing", "sculpture", "manuscript", "decorative"}
# plausible measurements: object -> (prefix, min cm, max cm); paintings and drawings get width x height
SIZE = {"bronze oil lamp": ("l.", 6, 15), "terracotta female figure": ("h.", 12, 40),
        "limestone stele fragment with inscription": ("h.", 30, 90), "painted ceramic bowl": ("diam.", 10, 28),
        "gold diadem": ("w.", 14, 30), "marble head of a youth": ("h.", 20, 45), "glass perfume flask": ("h.", 6, 18),
        "bronze helmet": ("h.", 20, 30), "inscribed clay tablet": ("h.", 5, 15), "mosaic panel with birds": ("w.", 60, 180),
        "carved wooden mask": ("h.", 20, 45), "beaded ceremonial belt": ("l.", 60, 110),
        "woven funerary textile": ("l.", 80, 200), "carved ancestor figure": ("h.", 30, 120),
        "bronze dancer, cast sculpture": ("h.", 25, 70), "silver tea service": ("h. of pot", 12, 30),
        "lacquered writing box": ("w.", 20, 45), "porcelain vase with dragons": ("h.", 25, 60),
        "tapestry fragment with hunters": ("w.", 60, 220), "illuminated manuscript leaf": ("x", 20, 45),
        "ink drawing of a cathedral": ("x", 20, 60)}


def _clamp(x, lo=0, hi=3):
    return max(lo, min(hi, x))


def _object(rng, categories=None):
    pool = [o for o in OBJECTS if categories is None or o[1] in categories]
    obj, cat, date_text = pick(rng, pool)
    st = pick(rng, STATES)
    culture = st[1] if cat in ("archaeological", "ethnographic") else pick(rng, ["Flemish", "Northern", "Lowland",
                                                                                 "Southern", "Alpine", "Baltic"])
    prefix, lo, hi = SIZE.get(obj, ("x", 30, 140))
    a = rng.randint(lo * 10, hi * 10) / 10
    if prefix == "x":
        dims = f"{round(a * 10)} x {round(a * 10 * rng.uniform(0.6, 1.4))} mm"
    else:
        dims = f"{prefix} {a:g} cm"
    return {"object": f"{culture} {obj}", "category": cat, "date_text": date_text, "dims": dims, "state": st[0],
            "state_adj": st[1], "state_law_year": st[2]}


def _label(f):
    """The catalogue line: object, measurements, date."""
    return f"{f['object']} ({f['dims']}), {f['date_text']}"


def _yrs(n):
    return f"{n} year{'' if n == 1 else 's'}"


# =========================================================================== schema 1
DOC_QUALITY = {"originals": "original dated invoices and period photographs",
               "copies": "photocopies of invoices; originals said to be lost",
               "mixed": "one original invoice, the rest the seller's typed list",
               "statement": "a signed statement from the seller, with no supporting papers"}
LICENCE = {"valid": "an export licence from {state} for this object, matching its description and photograph",
           "mismatch": "an export licence from {state} whose description and dimensions do not match the object",
           "none": "no export licence", "not_needed": "no licence; the seller says the object left {state} before its law applied"}
REGISTRY = {"clear": "stolen-object register searched: no match", "not_done": "stolen-object register not yet searched",
            "possible_match": "stolen-object register search: a possible match with an object reported stolen in {year}"}
RED_FLAGS = ["fresh soil traces in crevices noted by the conservator", "the price is well below comparable sales",
             "the seller's 'old family collection' cannot be named", "the invoice is dated before the dealer's firm existed",
             "the object appeared on the market only two years ago"]


class PurchaseProposalDueDiligence(Schema):
    name = "purchase_proposal_due_diligence"
    description = "A curator proposes buying an object from a dealer or at auction; the acquisitions committee weighs its provenance and paperwork."
    surfaces = {"json": 0.5, "acquisition_proposal": 0.25, "provenance_research_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        o = _object(rng)
        cutoff = pick(rng, [1970, 1970, 1970, 1972, 1983])
        seller = pick_w(rng, {"dealer": 5, "auction": 3, "private": 2})
        first_doc = pick(rng, [1931, 1952, 1964, 1968, 1975, 1988, 2004, None])
        doc_q = pick_w(rng, {"originals": 3, "copies": 3, "mixed": 2, "statement": 2})
        gap = pick_w(rng, {0: 3, 6: 2, 15: 2, 30: 1})
        licence = pick_w(rng, {"none": 4, "valid": 2, "mismatch": 2, "not_needed": 2})
        registry = pick_w(rng, {"clear": 5, "not_done": 3, "possible_match": 1})
        flag = pick(rng, RED_FLAGS) if chance(rng, 0.3) else None
        if reg == "clear":
            if chance(rng, 0.5):
                first_doc, doc_q, gap, registry, flag = pick(rng, [1931, 1952, 1964]), "originals", 0, "clear", None
                licence = pick(rng, ["not_needed", "valid"])
            else:
                first_doc, doc_q, licence, flag = pick(rng, [2004, None]), "statement", "none", pick(rng, RED_FLAGS)
        elif reg == "borderline":
            first_doc = cutoff + pick(rng, [-2, -1, 0, 1])
            doc_q = pick(rng, ["copies", "mixed", "originals"])
        else:
            twist = pick(rng, ["masterpiece_thin_paper", "good_paper_register_hit", "licence_but_gap", "licence_mismatch"])
            if twist == "masterpiece_thin_paper":
                first_doc, doc_q, registry = pick(rng, [1952, 1964]), "statement", "clear"
            elif twist == "good_paper_register_hit":
                first_doc, doc_q, registry = pick(rng, [1952, 1964]), "originals", "possible_match"
            elif twist == "licence_but_gap":
                licence, first_doc, gap = "valid", pick(rng, [1988, 2004]), 30
            else:
                licence, first_doc, doc_q, registry = "mismatch", pick(rng, [1975, 1988]), pick(rng, ["originals", "copies"]), "clear"
        if first_doc is not None:
            gap = min(gap, max(0, 2021 - first_doc))
        price = int(round(lognormal_amount(rng, 85000, 1.0), -3))
        return {"regime": reg, **o, "cutoff": cutoff, "seller_type": seller,
                "seller": pick(rng, DEALERS) if seller == "dealer" else pick(rng, AUCTIONS) if seller == "auction" else person(rng),
                "first_doc_year": first_doc, "doc_quality": doc_q, "gap_years": gap, "licence": licence,
                "registry": registry, "registry_year": rng.randint(1985, 2019), "red_flag": flag, "price": price,
                "warranty": pick(rng, ["full title warranty", "limited warranty for 12 months", "no warranty"]),
                "museum": pick(rng, MUSEUMS), "curator": person(rng), "proposal_id": ident(rng, "ACQ", 4, "-"),
                "meeting": rand_date(rng)}

    def _reliability(self, f):
        r = {"originals": 3, "copies": 2, "mixed": 1, "statement": 0}[f["doc_quality"]]
        if f["red_flag"] in ("the invoice is dated before the dealer's firm existed",
                             "the seller's 'old family collection' cannot be named"):
            r -= 1
        return _clamp(r)

    def _meets(self, f):
        documented = f["first_doc_year"] is not None and f["first_doc_year"] < f["cutoff"] and self._reliability(f) >= 1
        return documented or f["licence"] == "valid"

    def _gap(self, f):
        if f["first_doc_year"] is None or (not self._meets(f) and f["first_doc_year"] >= f["cutoff"]):
            return 3
        g = f["gap_years"]
        return 0 if g == 0 else 1 if g < 10 else 2 if g < 25 else 3

    def _policy(self, f):
        return [f"Acquisitions policy 4.2: the museum acquires an object only if documents show it was outside its likely "
                f"country of modern origin before {f['cutoff']}, or was exported lawfully under a licence of that country.",
                "4.3: the museum will not acquire where there is reasonable cause to think an object was stolen; a "
                "possible register match goes to the ethics committee.",
                f"4.4: {f['state']}'s cultural-property law (from {f['state_law_year']}) requires a licence to export "
                "objects of this kind.",
                "4.5: missing documents may be made a condition of purchase; the committee may decline any proposal."]

    def _chain(self, f):
        if f["first_doc_year"] is None:
            return ["no ownership history before the current seller"]
        out = [f"{f['first_doc_year']}: first recorded, " + pick_from(f, ["in a private collection abroad",
                                                                      "with a dealer abroad", "in a sale catalogue abroad"])]
        if f["gap_years"]:
            out.append(f"{f['first_doc_year'] + 1}-{f['first_doc_year'] + f['gap_years']}: no record")
        out.append(f"{min(2025, max(f['first_doc_year'] + f['gap_years'] + 1, 2008))}: offered by {f['seller']}")
        return out

    def render(self, f, rng, surface):
        lic = LICENCE[f["licence"]].format(state=f["state"])
        reg_t = REGISTRY[f["registry"]].format(year=f["registry_year"])
        chain = self._chain(f)
        if surface == "json":
            return dumps(prune({
                "museum": f["museum"],
                "proposal": {"id": f["proposal_id"], "object": _label(f),
                             "seller": f"{f['seller']} ({f['seller_type']})", "price": f["price"],
                             "proposed_by": f["curator"]},
                "provenance": {"chain": chain, "documents": DOC_QUALITY[f["doc_quality"]], "export_licence": lic,
                               "register_search": reg_t, "seller_warranty": f["warranty"],
                               "concerns_noted": f["red_flag"]},
                "policy": self._policy(f),
            }))
        if surface == "acquisition_proposal":
            pitch = noisy(join_sentences([
                f"I propose that the {f['museum']} acquire this {f['object']} ({f['date_text']}; {f['dims']}) from {f['seller']} for "
                f"{money(f['price'], rng)}.", pick(rng, ["It would fill a clear gap in our holdings.",
                                                        "It is the finest example offered in a decade.",
                                                        "It complements the gallery redisplay planned for next year."]),
                f"Provenance: {'; '.join(chain)}.", f"Documents: {DOC_QUALITY[f['doc_quality']]}.",
                f"Export: {lic}.", f"{reg_t.capitalize()}.", (f"Note: {f['red_flag']}." if f["red_flag"] else "")]), rng, 0.1)
            return (f"ACQUISITION PROPOSAL {f['proposal_id']} - committee of {fmt_date(f['meeting'], rng)}\n"
                    f"From: {f['curator']}, curator\n\n{pitch}\n\nSeller warranty: {f['warranty']}.\n\n" + "\n".join(self._policy(f)))
        rows = [["Object", _label(f)], ["Seller", f"{f['seller']} ({f['seller_type']})"]]
        rows += [["Provenance", c] for c in chain]
        rows += [["Documents", DOC_QUALITY[f["doc_quality"]]], ["Export", lic], ["Register", reg_t],
                 ["Warranty", f["warranty"]], ["Concerns", f["red_flag"] or "none noted"]]
        return (f"PROVENANCE RESEARCH MEMO - {f['proposal_id']} ({f['museum']})\n\n" + table(["Topic", "Finding"], rows, rng)
                + "\n\n" + "\n".join(self._policy(f)))

    def questions(self):
        return {
            "acquisition_recommendation": choice("What should the acquisitions committee do with this proposal?", {
                "approve": "Approve the purchase as proposed.",
                "approve_subject_to_documents": "Approve only once named missing documents or checks are supplied.",
                "decline": "Decline the purchase.",
                "refer_to_ethics_committee": "Refer it to the ethics committee because of a possible theft or a serious unexplained gap.",
            }),
            "principal_concern": choice("What is the main concern about this object, if any?", {
                "documents_unreliable": "The papers offered cannot be relied on.",
                "export_licence_problem": "The export licence is missing or does not match the object.",
                "no_concern": "Nothing material: the record supports acquisition.",
                "possible_theft": "A possible match with an object reported stolen.",
                "provenance_gap": "A gap or late first record in the ownership history.",
            }),
            "provenance_gap_severity": score("How serious are the gaps in the object's ownership history?", [
                "None: the history is continuous.",
                "Minor: a short gap well clear of the cut-off.",
                "Significant: a long gap or uncertain dating near the cut-off.",
                "Critical: nothing documents the object before the cut-off year.",
            ]),
            "paperwork_reliability": score("How far can the provenance documents be relied on?", [
                "Not at all: assertions only, or papers that contradict themselves.",
                "Weakly: mostly the seller's own lists.",
                "Fairly: copies or partial originals.",
                "Fully: original, dated, consistent documents.",
            ]),
            "meets_cutoff_rule": noul(
                "The object satisfies the policy's cut-off rule.",
                yes="Reliable documents place it outside its country of origin before the cut-off year, or a matching export licence exists.",
                no="It is first documented abroad in or after the cut-off year, or only by the seller's word, and no matching licence exists."),
        }

    def intended(self, f):
        meets, rel, gap = self._meets(f), self._reliability(f), self._gap(f)
        if f["registry"] == "possible_match":
            concern = "possible_theft"
        elif not meets and f["licence"] == "mismatch":
            concern = "export_licence_problem"
        elif not meets:
            concern = "provenance_gap" if rel >= 1 else "documents_unreliable"
        elif rel <= 1:
            concern = "documents_unreliable"
        elif gap >= 2:
            concern = "provenance_gap"
        else:
            concern = "no_concern"
        if f["registry"] == "possible_match":
            rec = "refer_to_ethics_committee"
        elif not meets:
            rec = "decline" if rel == 0 or gap == 3 else "refer_to_ethics_committee"
        elif rel <= 1 or f["registry"] == "not_done" or gap == 2:
            rec = "approve_subject_to_documents"
        elif gap == 3:
            rec = "refer_to_ethics_committee"
        else:
            rec = "approve"
        return {"acquisition_recommendation": rec, "principal_concern": concern, "provenance_gap_severity": gap,
                "paperwork_reliability": rel, "meets_cutoff_rule": "yes" if meets else "no"}


def pick_from(f, options):
    """Deterministic render-time flavour keyed on the case id (does not consume the rng)."""
    key = f.get("proposal_id") or f.get("case_id") or f.get("offer_id") or ""
    return options[sum(map(ord, key)) % len(options)]


# =========================================================================== schema 2
CONDITIONS = {"none": "no conditions", "credit_line": "a credit line naming the donor",
              "permanent_display": "the object must be on permanent display",
              "never_deaccession": "the museum may never deaccession the object",
              "display_with_family_items": "it must always be shown together with the donor's other family pieces"}
FIT = {"core": "a core area of the collecting plan", "adjacent": "an area the plan names as secondary",
       "outside": "an area outside the museum's collecting plan", "duplicate": "an area where the museum already holds several close examples"}


class GiftOfferReview(Schema):
    name = "gift_offer_review"
    description = "A donor offers an object as a gift or bequest, with conditions and an appraisal; the museum decides whether to accept."
    surfaces = {"json": 0.5, "donor_letter": 0.25, "registrar_assessment": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        o = _object(rng)
        cond = pick_w(rng, {"none": 3, "credit_line": 3, "permanent_display": 2, "never_deaccession": 2,
                            "display_with_family_items": 1})
        fit = pick_w(rng, {"core": 3, "adjacent": 3, "outside": 2, "duplicate": 2})
        prov = pick_w(rng, {"complete": 4, "minor_gap": 3, "unknown_before_donor": 2})
        care_cost = pick(rng, [500, 1500, 4000, 9000, 18000, 45000])
        budget = pick(rng, [10000, 20000, 40000])
        appraiser = pick_w(rng, {"independent": 5, "donor_relative": 1, "dealer_selling_to_donor": 1, "museum_staff": 1})
        if reg == "clear":
            if chance(rng, 0.5):
                cond, fit, prov, care_cost, appraiser = pick(rng, ["none", "credit_line"]), "core", "complete", pick(rng, [500, 2000]), "independent"
            else:
                cond, fit, care_cost = pick(rng, ["permanent_display", "never_deaccession"]), pick(rng, ["outside", "duplicate"]), 60000
        elif reg == "borderline":
            care_cost = int(near(rng, budget, budget * 0.2, lo=200))
            fit = pick(rng, ["adjacent", "core"])
        else:
            twist = pick(rng, ["great_object_bad_terms", "fine_terms_unknown_past"])
            if twist == "great_object_bad_terms":
                fit, prov, cond = "core", "complete", pick(rng, ["permanent_display", "never_deaccession", "display_with_family_items"])
            else:
                fit, cond, prov = "core", "credit_line", "unknown_before_donor"
        value = int(round(lognormal_amount(rng, 40000, 1.0), -2))
        return {"regime": reg, **o, "conditions": cond, "fit": fit, "provenance": prov, "care_cost": care_cost,
                "care_budget": budget, "appraiser": appraiser, "appraised_value": value,
                "kind": pick(rng, ["gift", "bequest"]), "donor": person(rng), "registrar": person(rng),
                "museum": pick(rng, MUSEUMS), "offer_id": ident(rng, "GFT", 4, "-"), "date": rand_date(rng),
                "acquired_by_donor": pick(rng, [1958, 1967, 1979, 1994, 2006])}

    def _fit_score(self, f):
        return {"core": 3, "adjacent": 2, "duplicate": 1, "outside": 0}[f["fit"]]

    def _burden(self, f):
        b = 0 if f["care_cost"] < f["care_budget"] * 0.5 else 1 if f["care_cost"] <= f["care_budget"] else 2
        if f["conditions"] in ("permanent_display", "never_deaccession", "display_with_family_items"):
            b += 1
        return _clamp(b)

    def _conditions_ok(self, f):
        return f["conditions"] in ("none", "credit_line")

    def _prov_text(self, f):
        return {"complete": f"documented since its {f['date_text']} creation or discovery through to the donor's purchase in {f['acquired_by_donor']}",
                "minor_gap": f"documented except for about ten years before the donor bought it in {f['acquired_by_donor']}",
                "unknown_before_donor": f"nothing is known before the donor bought it in {f['acquired_by_donor']}"}[f["provenance"]]

    def _appraiser_text(self, f):
        return {"independent": "an accredited appraiser with no link to the donor or the museum",
                "donor_relative": "the donor's brother-in-law, a licensed appraiser",
                "dealer_selling_to_donor": "the dealer who originally sold the object to the donor",
                "museum_staff": "a curator on the museum's own staff"}[f["appraiser"]]

    def _policy(self, f):
        return ["Gifts policy 6.1: the museum accepts gifts that fit its collecting plan and that it can care for within "
                f"its annual care budget for new gifts ({money(f['care_budget'])}).",
                "6.2: the museum accepts no conditions on display, grouping or future disposal; a credit line is acceptable.",
                "6.3: gifts face the same provenance standard as purchases; appraisals for the donor's tax purposes must "
                "come from an appraiser independent of both the donor and the museum."]

    def render(self, f, rng, surface):
        if surface == "json":
            return dumps(prune({
                "museum": f["museum"],
                "offer": {"id": f["offer_id"], "type": f["kind"], "donor": f["donor"],
                          "object": _label(f), "donor_conditions": CONDITIONS[f["conditions"]]},
                "assessment": {"collection_fit": FIT[f["fit"]], "provenance": self._prov_text(f),
                               "estimated_first_year_care_cost": f["care_cost"],
                               "appraisal": {"value": f["appraised_value"], "by": self._appraiser_text(f)}},
                "policy": self._policy(f),
            }))
        if surface == "donor_letter":
            body = noisy(join_sentences([
                f"I would like to offer the {f['museum']} my {f['object']} ({f['date_text']}; {f['dims']}) as a {f['kind']}.",
                f"I bought it in {f['acquired_by_donor']}." if f["provenance"] != "complete" else
                f"It came to me with its full history, bought in {f['acquired_by_donor']}.",
                {"none": "I ask nothing in return.", "credit_line": "I would only ask that the label carry my family's name.",
                 "permanent_display": "My one condition is that it is always on display.",
                 "never_deaccession": "I must insist that it never be sold or given away.",
                 "display_with_family_items": "It must always be shown with the other family pieces I gave earlier."}[f["conditions"]],
                f"It has been valued at {money(f['appraised_value'], rng)} by " + {
                    "independent": "an accredited appraiser with no link to me or to the museum",
                    "donor_relative": "my brother-in-law, who is a licensed appraiser",
                    "dealer_selling_to_donor": "the dealer who sold it to me",
                    "museum_staff": "one of your own curators"}[f["appraiser"]] + "."]), rng)
            notes = kv_lines([("Collection fit", FIT[f["fit"]]), ("Provenance research", self._prov_text(f)),
                              ("First-year care cost", money(f["care_cost"], rng))], rng)
            return (f"Letter received {fmt_date(f['date'], rng)} ({f['offer_id']})\n\n{body}\n\n{f['donor']}\n\n"
                    f"[registrar notes]\n{notes}\n\n" + "\n".join(self._policy(f)))
        rows = [("Offer", f"{f['offer_id']} - {f['kind']} from {f['donor']}"), ("Object", _label(f)),
                ("Conditions", CONDITIONS[f["conditions"]]), ("Fit", FIT[f["fit"]]), ("Provenance", self._prov_text(f)),
                ("Care cost (year 1)", money(f["care_cost"], rng)),
                ("Appraisal", f"{money(f['appraised_value'], rng)} by {self._appraiser_text(f)}")]
        return f"REGISTRAR'S ASSESSMENT ({f['registrar']})\n" + kv_lines(rows, rng) + "\n\n" + "\n".join(self._policy(f))

    def questions(self):
        return {
            "gift_decision": choice("How should the museum respond to this offer?", {
                "accept": "Accept the gift on the terms offered.",
                "accept_if_terms_change": "Accept only if the donor drops the conditions the policy does not allow.",
                "decline": "Decline the offer.",
                "defer_for_research": "Hold the decision until the object's history before the donor is researched.",
            }),
            "collection_fit_strength": score("How well does the object fit the museum's collecting plan?", [
                "Not at all: outside the plan.",
                "Weakly: it largely duplicates what the museum holds.",
                "Reasonably: a secondary area of the plan.",
                "Strongly: a core area of the plan.",
            ]),
            "care_burden": score("How heavy a burden would the gift place on the museum?", [
                "Light: modest care costs and no strings.",
                "Noticeable: costs that use a real share of the budget.",
                "Heavy: costs near or over budget, or binding conditions.",
                "Severe: costs over budget together with binding conditions.",
            ]),
            "conditions_within_policy": noul(
                "The donor's conditions are ones the gifts policy allows.",
                yes="There are no conditions, or only a credit line.",
                no="The donor requires permanent display, fixed grouping or a ban on future disposal."),
            "appraisal_independent": noul("The appraisal comes from someone independent of both the donor and the museum."),
        }

    def intended(self, f):
        fit, burden, ok = self._fit_score(f), self._burden(f), self._conditions_ok(f)
        if fit == 0 or f["care_cost"] > f["care_budget"] or (fit == 1 and burden >= 2):
            dec = "decline"
        elif f["provenance"] == "unknown_before_donor":
            dec = "defer_for_research"
        elif not ok:
            dec = "accept_if_terms_change"
        else:
            dec = "accept"
        return {"gift_decision": dec, "collection_fit_strength": fit, "care_burden": burden,
                "conditions_within_policy": "yes" if ok else "no",
                "appraisal_independent": "yes" if f["appraiser"] == "independent" else "no"}


# =========================================================================== schema 3
LENDERS = {"private": "a private collector", "foreign_museum": "a foreign public museum",
           "foundation": "a family foundation", "dealer": "a dealer's gallery"}
WARTIME = {"documented": "ownership is documented without a break from 1925 to 1960",
           "partial": "the owner from 1930 to 1938 is known; the next record is a sale in 1949",
           "gap": "no record exists between 1932 and 1951",
           "forced_sale_known": "a 1938 sale by a family forced to leave their home is recorded; no settlement with the family is known"}


class IncomingExhibitionLoan(Schema):
    name = "incoming_exhibition_loan"
    description = "A loan request for an exhibition where the object's 1933-1945 history or a register entry needs checking before contract."
    surfaces = {"json": 0.5, "registrar_checklist": 0.25, "lender_correspondence": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        o = _object(rng, EUROPEAN_ART)
        lender = pick_w(rng, {"private": 3, "foreign_museum": 3, "foundation": 2, "dealer": 1})
        war = pick_w(rng, {"documented": 4, "partial": 3, "gap": 2, "forced_sale_known": 1})
        register = pick_w(rng, {"none": 6, "claim_listed": 1, "inquiry_listed": 1})
        notice_weeks = pick(rng, [4, 6, 8, 12])
        weeks_to_arrival = pick(rng, [3, 6, 9, 14, 20, 30])
        lender_publish = chance(rng, 0.75)
        if reg == "clear":
            if chance(rng, 0.5):
                war, register, lender_publish, weeks_to_arrival = "documented", "none", True, notice_weeks + pick(rng, [8, 16])
            else:
                war, register = pick(rng, ["forced_sale_known", "gap"]), "claim_listed"
        elif reg == "borderline":
            weeks_to_arrival = notice_weeks + pick(rng, [-1, 0, 1, 2])
            war = pick(rng, ["partial", "documented"])
        else:
            twist = pick(rng, ["prestige_loan_open_gap", "documented_but_no_time"])
            if twist == "prestige_loan_open_gap":
                lender, war, lender_publish = "foreign_museum", pick(rng, ["gap", "partial"]), False
            else:
                war, register, weeks_to_arrival = "documented", "none", notice_weeks - pick(rng, [1, 2, 3])
        return {"regime": reg, **o, "lender_type": lender, "wartime": war, "register": register,
                "notice_weeks": notice_weeks, "weeks_to_arrival": max(1, weeks_to_arrival),
                "lender_agrees_publication": lender_publish, "museum": pick(rng, MUSEUMS), "registrar": person(rng),
                "lender_contact": person(rng), "loan_id": ident(rng, "LN", 5, "-"),
                "exhibition": pick(rng, ["Light on the Water", "Faces of Trade", "The Quiet Interior", "Northern Workshops",
                                         "Gold and Ash", "Cities in Ink", "Makers and Merchants", "Winter Pieces"]),
                "sent": rand_date(rng)}

    def _risk(self, f):
        r = {"documented": 0, "partial": 1, "gap": 2, "forced_sale_known": 3}[f["wartime"]]
        if f["register"] == "claim_listed":
            r = 3
        elif f["register"] == "inquiry_listed":
            r = max(r, 2)
        return r

    def _immunity_ok(self, f):
        return f["weeks_to_arrival"] >= f["notice_weeks"] and f["lender_agrees_publication"] and self._risk(f) <= 1

    def _pressure(self, f):
        slack = f["weeks_to_arrival"] - f["notice_weeks"]
        return 3 if slack < 0 else 2 if slack <= 1 else 1 if slack <= 6 else 0

    def _rules(self, f):
        return [f"Seizure-immunity law: a loan from abroad is protected from seizure only if the borrower publishes the "
                f"object's provenance at least {f['notice_weeks']} weeks before it arrives and has found no evidence that "
                "it was unlawfully taken.",
                "Loans policy 8.2: any gap in an object's history between 1933 and 1945 must be researched and disclosed; "
                "an object with an unresolved claim or a known forced sale is not borrowed until the matter is resolved."]

    def render(self, f, rng, surface):
        reg_t = {"none": "it has no entry on the lost-art registers",
                 "claim_listed": "it is listed on a lost-art register as claimed by heirs",
                 "inquiry_listed": "a register lists an open inquiry about a closely similar work"}[f["register"]]
        arrival = f["sent"] + timedelta(weeks=f["weeks_to_arrival"])
        if surface == "json":
            return dumps(prune({
                "loan": {"id": f["loan_id"], "borrower": f["museum"], "exhibition": f["exhibition"],
                         "object": _label(f), "lender": LENDERS[f["lender_type"]],
                         "today": fmt_date(f["sent"], style="iso"), "planned_arrival": fmt_date(arrival, style="iso")},
                "provenance": {"1933_1945": WARTIME[f["wartime"]], "registers": reg_t,
                               "lender_allows_publication": f["lender_agrees_publication"]},
                "rules": self._rules(f),
            }))
        if surface == "registrar_checklist":
            items = [f"Object: {f['object']} ({f['date_text']}; {f['dims']}), lent by {LENDERS[f['lender_type']]}",
                     f"Exhibition: {f['exhibition']}; arrival {fmt_date(arrival, rng)} ({f['weeks_to_arrival']} weeks from today)",
                     f"1933-1945: {WARTIME[f['wartime']]}", f"Registers: {reg_t}",
                     f"Lender permits publishing provenance: {'yes' if f['lender_agrees_publication'] else 'no'}"]
            return (f"LOAN CHECKLIST {f['loan_id']} - {f['museum']} (prepared {fmt_date(f['sent'], rng)})\n"
                    + bullets(items, rng) + "\n\n" + "\n".join(self._rules(f)))
        msgs = [(f["registrar"], f"For {f['exhibition']} we hope to borrow the {f['object']}. Our research shows "
                                  f"{WARTIME[f['wartime']]}, and {reg_t}."),
                (f["lender_contact"], ("We are happy for you to publish the provenance."
                                       if f["lender_agrees_publication"] else "We would prefer the provenance not be published.")
                 + f" The work can travel to arrive on {fmt_date(arrival, rng)}.")]
        thread = "\n\n".join(f"{fmt_date(f['sent'], rng)} - {w}:\n{noisy(t, rng, 0.1)}" for w, t in msgs)
        return f"Re: loan request {f['loan_id']}\n\n{thread}\n\n" + "\n".join(self._rules(f))

    def questions(self):
        return {
            "loan_handling": choice("How should the borrowing museum proceed with this loan?", {
                "proceed": "Go ahead with the loan contract as planned.",
                "proceed_with_provenance_note": "Go ahead, publishing the provenance with an explicit note on the open years.",
                "research_before_contract": "Pause the contract until the open questions in the history are researched.",
                "withdraw_object": "Drop the object from the exhibition.",
            }),
            "open_issue": choice("What is the main unresolved issue, if any?", {
                "claim_or_forced_sale": "A recorded claim or forced sale affecting the object.",
                "immunity_timing": "Too little time, or no permission, to publish the provenance before arrival.",
                "no_open_issue": "Nothing material is unresolved.",
                "wartime_gap": "Unexplained years in the object's history between 1933 and 1945.",
            }),
            "spoliation_risk": score("How likely is it that the object was taken unlawfully between 1933 and 1945?", [
                "Remote: the history through those years is documented.",
                "Low: a partial gap with nothing pointing to loss.",
                "Real: an unexplained gap covering those years, or an open inquiry.",
                "High: a known forced sale or a registered claim.",
            ]),
            "immunity_deadline_pressure": score("How tight is the time left to meet the seizure-immunity notice period?", [
                "Comfortable: months to spare.",
                "Manageable: a few weeks to spare.",
                "Tight: a week or less to spare.",
                "Missed: the notice period can no longer be met for the planned arrival.",
            ]),
            "immunity_obtainable": noul(
                "The loan can obtain seizure immunity for its planned arrival date.",
                yes="The provenance can be published in time, the lender allows it, and nothing suggests an unlawful taking.",
                no="The notice period cannot be met, the lender refuses publication, or there is evidence of a possible unlawful taking."),
        }

    def intended(self, f):
        risk, ok = self._risk(f), self._immunity_ok(f)
        if risk == 3:
            issue = "claim_or_forced_sale"
        elif risk == 2:
            issue = "wartime_gap"
        elif f["weeks_to_arrival"] < f["notice_weeks"] or not f["lender_agrees_publication"]:
            issue = "immunity_timing"
        else:
            issue = "no_open_issue"
        if risk == 3:
            handling = "withdraw_object"
        elif risk == 2:
            handling = "research_before_contract"
        elif not ok:
            handling = "research_before_contract" if f["weeks_to_arrival"] < f["notice_weeks"] else "withdraw_object"
        elif risk == 1:
            handling = "proceed_with_provenance_note"
        else:
            handling = "proceed"
        return {"loan_handling": handling, "open_issue": issue, "spoliation_risk": risk,
                "immunity_deadline_pressure": self._pressure(f), "immunity_obtainable": "yes" if ok else "no"}


# =========================================================================== schema 4
# reason -> (curatorial case text, base strength 0-3)
DEACC_REASONS = {
    "duplicate": ("the museum holds four near-identical examples in better condition", 2),
    "poor_condition": ("the object has decayed beyond economic repair, per the conservator's report", 3),
    "repairable_condition": ("the object needs costly conservation the museum has postponed", 1),
    "outside_scope": ("the object lies outside the collecting plan adopted after a merger", 2),
    "hazardous": ("the object contains asbestos and radioactive paint that cannot be stabilised", 3),
    "forgery": ("technical analysis shows the object is a later copy, not the work it was catalogued as", 3),
    "storage_pressure": ("storage space is needed for the new archive", 0),
    "market_value": ("prices for comparable pieces are at a peak and a sale would raise funds", 0),
}
METHODS = {"transfer": "transfer to another public museum", "auction": "sale at public auction",
           "trustee_sale": "private sale to a trustee who has expressed interest", "destroy": "documented destruction"}
PROCEEDS = {"none": "no proceeds (transfer or destruction)", "acquisitions": "proceeds to the acquisitions fund",
            "direct_care": "proceeds to conservation of the existing collection",
            "operating": "proceeds to cover the operating deficit"}
RESTRICTION = {"none": "no donor restriction", "no_sale": "the deed of gift forbids sale",
               "fund_purchase": "bought with a donor fund whose terms send any proceeds back to the fund",
               "bequest_silent": "a bequest whose will is silent on disposal"}


class DeaccessionProposal(Schema):
    name = "deaccession_proposal"
    description = "A curator proposes removing an object from the collection; the board weighs the case, the method and the use of any proceeds against the disposals policy."
    surfaces = {"json": 0.5, "deaccession_form": 0.25, "board_paper": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        o = _object(rng)
        reason = pick(rng, list(DEACC_REASONS), [3, 3, 2, 3, 2, 2, 1, 1])
        significance = pick_w(rng, {"minor": 4, "moderate": 3, "major": 2})
        method = pick_w(rng, {"transfer": 3, "auction": 5, "trustee_sale": 1, "destroy": 1})
        proceeds = "none" if method in ("transfer", "destroy") else pick_w(rng, {"acquisitions": 4, "direct_care": 3, "operating": 2})
        restriction = pick_w(rng, {"none": 3, "no_sale": 3, "fund_purchase": 2, "bequest_silent": 2})
        min_years = pick(rng, [5, 10, 15])
        held = pick(rng, [2, 4, 8, 12, 25, 60])
        offered_first = chance(rng, 0.7)
        if reg == "clear":
            if chance(rng, 0.5):
                reason, significance, method, proceeds, restriction = pick(rng, ["poor_condition", "duplicate", "hazardous"]), "minor", "transfer", "none", "none"
                held, offered_first = pick(rng, [25, 60]), True
            else:
                reason, significance, method, proceeds = pick(rng, ["storage_pressure", "market_value"]), "major", "auction", "operating"
        elif reg == "borderline":
            held = min_years + pick(rng, [-1, 0, 1])
            reason = pick(rng, ["duplicate", "outside_scope", "repairable_condition"])
        else:
            twist = pick(rng, ["good_case_bad_plan", "weak_case_clean_plan"])
            if twist == "good_case_bad_plan":
                reason = pick(rng, ["duplicate", "outside_scope", "poor_condition"])
                method, proceeds = pick(rng, [("trustee_sale", "acquisitions"), ("auction", "operating")])
                restriction = pick(rng, ["no_sale", "no_sale", "fund_purchase", "none"])
            else:
                reason, method, proceeds, restriction, offered_first = pick(rng, ["repairable_condition", "storage_pressure"]), "transfer", "none", "none", True
                held = pick(rng, [25, 60])
        if method in ("transfer", "destroy"):
            proceeds = "none"
        return {"regime": reg, **o, "reason": reason, "significance": significance, "method": method,
                "proceeds": proceeds, "restriction": restriction, "min_years": min_years, "years_held": held,
                "offered_to_public_museums": offered_first, "museum": pick(rng, MUSEUMS), "curator": person(rng),
                "case_id": ident(rng, "DEA", 4, "-"), "estimate": int(round(lognormal_amount(rng, 30000, 1.1), -2)),
                "meeting": rand_date(rng)}

    def _case(self, f):
        return DEACC_REASONS[f["reason"]][1]

    def _trust_risk(self, f):
        r = {"minor": 0, "moderate": 1, "major": 2}[f["significance"]]
        if f["method"] == "trustee_sale":
            r += 2
        elif f["method"] == "auction" and f["significance"] == "major":
            r += 1
        if f["proceeds"] == "operating":
            r += 1
        return _clamp(r)

    def _proceeds_ok(self, f):
        return f["proceeds"] in ("none", "acquisitions", "direct_care")

    def _restriction_blocks(self, f):
        sale = f["method"] in ("auction", "trustee_sale")
        return (f["restriction"] == "no_sale" and sale) or (f["restriction"] == "fund_purchase" and sale
                                                           and f["proceeds"] != "acquisitions")

    def _policy(self, f):
        return ["Disposals policy 9.1: an object may be removed only for a documented curatorial reason: duplication, "
                "irreparable condition, a hazard, misattribution, or falling outside the collecting plan. Space and money "
                "are not reasons.",
                f"9.2: nothing is removed within {f['min_years']} years of acquisition; objects are first offered to public "
                "museums; no sale to trustees, staff or their families; donor restrictions are honoured.",
                "9.3: proceeds go only to new acquisitions or the direct care of the collection, never to operating costs."]

    def render(self, f, rng, surface):
        case_text = DEACC_REASONS[f["reason"]][0]
        offered = ("offered to public museums for 60 days; no taker" if f["offered_to_public_museums"]
                   else "not yet offered to other public museums")
        if surface == "json":
            return dumps(prune({
                "museum": f["museum"],
                "proposal": {"id": f["case_id"], "object": _label(f),
                             "proposed_by": f["curator"], "reason": case_text,
                             "significance": f"{f['significance']} within the collection",
                             "years_in_collection": f["years_held"]},
                "disposal": {"method": METHODS[f["method"]], "estimate": f["estimate"] if f["method"] in ("auction", "trustee_sale") else None,
                             "proceeds": PROCEEDS[f["proceeds"]], "public_offer": offered, "donor_terms": RESTRICTION[f["restriction"]]},
                "policy": self._policy(f),
            }))
        if surface == "deaccession_form":
            rows = [("Case", f["case_id"]), ("Object", _label(f)), ("Reason", case_text),
                    ("Significance", f["significance"]), ("Held for", _yrs(f["years_held"])),
                    ("Method", METHODS[f["method"]]), ("Proceeds", PROCEEDS[f["proceeds"]]), ("Public offer", offered),
                    ("Donor terms", RESTRICTION[f["restriction"]]), ("Proposed by", f["curator"])]
            return "DEACCESSION FORM\n" + kv_lines(rows, rng) + "\n\n" + "\n".join(self._policy(f))
        para = noisy(join_sentences([
            f"The curator proposes removing the {f['object']} ({f['date_text']}; {f['dims']}) from the collection.",
            f"Reason given: {case_text}.", f"It has been in the collection for {_yrs(f['years_held'])} and is of "
                                           f"{f['significance']} significance.",
            f"Proposed method: {METHODS[f['method']]}" + (f", estimate {money(f['estimate'], rng)}." if f["method"] in ("auction", "trustee_sale") else "."),
            f"Proceeds: {PROCEEDS[f['proceeds']]}.", f"Public museums: {offered}.", f"Donor terms: {RESTRICTION[f['restriction']]}."]), rng, 0.1)
        return (f"BOARD PAPER {f['case_id']} - {f['museum']}, meeting {fmt_date(f['meeting'], rng)}\n\n{para}\n\n"
                + "\n".join(self._policy(f)))

    def questions(self):
        return {
            "deaccession_decision": choice("What should the board decide about the proposed disposal?", {
                "approve": "Approve the disposal as proposed.",
                "approve_with_changes": "Approve removal only with a changed method or proceeds plan that meets the policy.",
                "decline": "Refuse the disposal.",
                "return_to_curator": "Send it back: the curatorial case for removal has not been made.",
            }),
            "curatorial_case_strength": score("How strong is the curatorial case for removing the object?", [
                "None: the reasons are space or money.",
                "Weak: a problem that could be solved without disposal.",
                "Sound: a recognised reason, though a judgement call.",
                "Compelling: the object is irreparable, hazardous or not what it was thought to be.",
            ]),
            "public_trust_risk": score("How much would the disposal as proposed risk public trust in the museum?", [
                "Little: a minor object leaving by an open route.",
                "Some: a notable object, handled openly.",
                "Considerable: an important object sold, or proceeds used doubtfully.",
                "Severe: an insider sale or money diverted to running costs for a major object.",
            ]),
            "proceeds_plan_compliant": noul(
                "The plan for any proceeds complies with the disposals policy.",
                yes="There are no proceeds, or they go to acquisitions or the direct care of the collection.",
                no="Proceeds would cover operating costs."),
            "donor_terms_prevent_plan": noul("The donor's terms prevent the disposal method proposed."),
        }

    def intended(self, f):
        case, risk = self._case(f), self._trust_risk(f)
        early = f["years_held"] < f["min_years"] and f["reason"] not in ("hazardous", "forgery")
        method_problem = (f["method"] == "trustee_sale" or self._restriction_blocks(f) or not self._proceeds_ok(f)
                          or (f["method"] in ("auction", "trustee_sale") and not f["offered_to_public_museums"]))
        if case <= 1:
            dec = "return_to_curator"
        elif early:
            dec = "decline"
        elif method_problem:
            dec = "approve_with_changes"
        else:
            dec = "approve"
        return {"deaccession_decision": dec, "curatorial_case_strength": case, "public_trust_risk": risk,
                "proceeds_plan_compliant": "yes" if self._proceeds_ok(f) else "no",
                "donor_terms_prevent_plan": "yes" if self._restriction_blocks(f) else "no"}


# =========================================================================== schema 5
CLAIM_BASIS = {"theft": "taken from the family home during an occupation and never returned",
               "confiscation": "confiscated by the authorities of the day from its owner",
               "forced_sale": "sold by the owner, who was being forced out of the country, at a fraction of its value",
               "illegal_export": "excavated and exported from {state} after its {law} cultural-property law without a licence",
               "lawful_sale": "lost to the family during the upheavals of that time"}
EVIDENCE = {"heirs": {3: "a pre-war insurance inventory and a police report, both naming the object",
                      2: "a post-war claim file and a family photograph showing the object",
                      1: "affidavits written decades later from family recollection",
                      0: "the family's account, with no supporting documents"},
            "ministry": {3: "excavation records and a customs seizure report naming the object",
                         2: "a site survey photograph showing the object in place before it disappeared",
                         1: "statements by local witnesses recorded many years later",
                         0: "the ministry's assertion, with no supporting documents"}}
IDENT = {3: "the claimant's photograph matches the object, including a repaired chip and an inventory number",
         2: "the description and dimensions match, but no photograph survives",
         1: "the description fits this object and several similar ones",
         0: "the claimant describes an object of this type with no distinguishing details"}


class RestitutionClaimAssessment(Schema):
    name = "restitution_claim_assessment"
    description = "A claim for the return of an object in the collection, by a fictional state's ministry or a former owner's heirs, assessed against the museum's restitution policy."
    surfaces = {"json": 0.5, "claim_letter": 0.25, "provenance_report": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        o = _object(rng)
        basis = pick(rng, list(CLAIM_BASIS), [2, 1, 3, 2, 2])
        ev = pick_w(rng, {3: 3, 2: 3, 1: 2, 0: 2})
        ident_lvl = pick_w(rng, {3: 3, 2: 3, 1: 2, 0: 1})
        acq_year = pick(rng, [1951, 1963, 1978, 1992, 2003, 2011])
        diligence = pick_w(rng, {"registers_checked": 3, "dealer_assurance_only": 3, "none_recorded": 2})
        if reg == "clear":
            if chance(rng, 0.5):
                basis, ev, ident_lvl = pick(rng, ["theft", "confiscation", "illegal_export", "forced_sale", "forced_sale"]), 3, 3
            else:
                basis, ev = "lawful_sale", pick(rng, [2, 3])
        elif reg == "borderline":
            if chance(rng, 0.5):
                ident_lvl = pick(rng, [0, 1, 2])
            else:
                ev = pick(rng, [0, 1, 2])
        else:
            twist = pick(rng, ["strong_story_weak_match", "good_faith_buyer_strong_claim"])
            if twist == "strong_story_weak_match":
                ev, ident_lvl = pick(rng, [2, 3]), pick(rng, [0, 0, 1])
            else:
                ev, ident_lvl, diligence = pick(rng, [2, 3]), 3, "registers_checked"
                basis = pick(rng, ["forced_sale", "theft", "illegal_export"])
        if basis == "illegal_export":
            o = _object(rng, {"archaeological", "ethnographic"})
        return {"regime": reg, **o, "basis": basis, "evidence": ev, "identification": ident_lvl,
                "acquired_year": acq_year, "diligence": diligence, "claimant_type": "ministry" if basis == "illegal_export" else "heirs",
                "claimant": person(rng), "museum": pick(rng, MUSEUMS), "case_id": ident(rng, "RST", 4, "-"),
                "acquired_from": pick(rng, DEALERS + AUCTIONS), "received": rand_date(rng),
                "loss_year": pick(rng, [1936, 1938, 1940, 1942, 1944]) if basis != "illegal_export" else None}

    def _basis_key(self, f):
        if f["evidence"] == 0 or f["basis"] == "lawful_sale":
            return "no_established_basis"
        return {"theft": "theft_or_confiscation", "confiscation": "theft_or_confiscation",
                "forced_sale": "forced_sale", "illegal_export": "illegal_export"}[f["basis"]]

    def _sale_record(self, f):
        if f["basis"] != "lawful_sale":
            return None
        return (f"a dealer's ledger shows the family sold the object at a normal market price in {f['loss_year']} and "
                "was paid in full")

    def _good_faith(self, f):
        return f["diligence"] == "registers_checked" or (f["diligence"] == "dealer_assurance_only" and f["acquired_year"] < 1970)

    def _policy(self, f):
        return ["Restitution policy 11.1: the museum returns an object where the evidence shows it was stolen or confiscated "
                "from its owner, or exported from its country of origin in breach of that country's law.",
                "11.2: where an owner sold under duress, the museum seeks a just and fair solution with the heirs, which "
                "may be return, a payment, or shared ownership and display.",
                "11.3: identification must be reasonably certain; where it is not, the museum commissions independent "
                "research before deciding. Good-faith acquisition by the museum does not defeat a proven claim."]

    def _story(self, f):
        text = CLAIM_BASIS[f["basis"]].format(state=f["state"], law=f["state_law_year"])
        if f["claimant_type"] == "heirs":
            return f"The heirs of its former owner say it was {text} in {f['loss_year']}."
        return f"The Ministry of Culture of {f['state']} says it was {text}."

    def _story_first_person(self, f):
        text = CLAIM_BASIS[f["basis"]].format(state=f["state"], law=f["state_law_year"])
        if f["claimant_type"] == "heirs":
            return f"It belonged to our family until {f['loss_year']}, when it was {text}."
        return f"It was {text}."

    def render(self, f, rng, surface):
        acq = (f"The museum bought it in {f['acquired_year']} from {f['acquired_from']}; "
               + {"registers_checked": "loss registers were searched at the time with no result",
                  "dealer_assurance_only": "the file holds only the dealer's assurance of good title",
                  "none_recorded": "no checks are recorded"}[f["diligence"]] + ".")
        if surface == "json":
            return dumps(prune({
                "museum": f["museum"],
                "claim": {"id": f["case_id"], "object": _label(f),
                          "claimant": ("Ministry of Culture of " + f["state"]) if f["claimant_type"] == "ministry"
                          else f"heirs represented by {f['claimant']}", "account": self._story(f),
                          "evidence": EVIDENCE[f["claimant_type"]][f["evidence"]],
                          "identification": IDENT[f["identification"]]},
                "museum_record": {"acquisition": acq, "earlier_sale": self._sale_record(f)},
                "policy": self._policy(f),
            }))
        if surface == "claim_letter":
            sender = (f"Ministry of Culture of {f['state']}" if f["claimant_type"] == "ministry"
                      else f"{f['claimant']}, on behalf of the heirs")
            body = noisy(join_sentences([
                f"We write to request the return of the {f['object']} now in your collection.", self._story_first_person(f),
                f"In support we provide {EVIDENCE[f['claimant_type']][f['evidence']]}.",
                f"As to identification, {IDENT[f['identification']]}."]), rng)
            return (f"From: {sender}\nTo: Director, {f['museum']}\nReceived: {fmt_date(f['received'], rng)}\nRef: {f['case_id']}\n\n"
                    f"{body}\n\n[museum file] {acq}" + (f" Also on file: {self._sale_record(f)}." if self._sale_record(f) else "")
                    + "\n\n" + "\n".join(self._policy(f)))
        rows = [["Object", _label(f)], ["Claim", self._story(f)],
                ["Evidence offered", EVIDENCE[f["claimant_type"]][f["evidence"]]], ["Identification", IDENT[f["identification"]]],
                ["Museum acquisition", acq]] + ([["Earlier sale", self._sale_record(f)]] if self._sale_record(f) else [])
        return (f"PROVENANCE REPORT on claim {f['case_id']} ({f['museum']})\n\n" + table(["Topic", "Finding"], rows, rng)
                + "\n\n" + "\n".join(self._policy(f)))

    def questions(self):
        return {
            "claim_response": choice("How should the museum respond to the claim?", {
                "commission_further_research": "Commission independent research before deciding.",
                "decline_claim": "Decline the claim.",
                "negotiate_settlement": "Negotiate a just and fair solution such as payment or shared ownership.",
                "return_object": "Return the object to the claimant.",
            }),
            "loss_basis": choice("On the evidence, how did the object leave its rightful owner or country?", {
                "forced_sale": "A sale made under duress at an unfair price.",
                "illegal_export": "Export in breach of the source country's law.",
                "no_established_basis": "No wrongful loss is established.",
                "theft_or_confiscation": "Theft or confiscation.",
            }),
            "identification_certainty": score("How certain is it that the claimed object is this object?", [
                "Speculative: an object of the same type, nothing more.",
                "Uncertain: the description fits several objects.",
                "Probable: description and dimensions match.",
                "Certain: photographs and unique marks match.",
            ]),
            "claim_evidence_strength": score("How strong is the evidence of a wrongful loss?", [
                "None: assertion only.",
                "Weak: late recollection.",
                "Good: post-war claim records and photographs.",
                "Strong: contemporary documents naming the object.",
            ]),
            "museum_acted_in_good_faith": noul(
                "The museum acquired the object in good faith.",
                yes="It checked loss registers at the time, or relied on a dealer's assurance in an era before such checks were expected.",
                no="It made no checks, or relied only on a dealer's word when checks were already expected."),
        }

    def intended(self, f):
        basis = self._basis_key(f)
        if basis == "no_established_basis":
            resp = "decline_claim" if f["evidence"] == 0 or f["basis"] == "lawful_sale" else "commission_further_research"
        elif f["identification"] <= 1 or f["evidence"] == 1:
            resp = "commission_further_research"
        elif basis == "forced_sale":
            resp = "negotiate_settlement"
        else:
            resp = "return_object"
        return {"claim_response": resp, "loss_basis": basis, "identification_certainty": f["identification"],
                "claim_evidence_strength": f["evidence"], "museum_acted_in_good_faith": "yes" if self._good_faith(f) else "no"}


# =========================================================================== schema 6
HOME_STATES = ["Republic of Merrovia", "Kingdom of Estravel", "Federation of Calderra", "Republic of Linnmark"]
MODERN = [("bent-plywood armchair", "h.", 70, 95), ("poster for a shipping line, lithograph", "x", 60, 100),
          ("studio pottery vase", "h.", 18, 45), ("press photograph of the harbour strike, gelatin silver print", "x", 20, 40),
          ("abstract composition, oil on canvas", "x", 50, 160), ("silver cocktail shaker", "h.", 18, 30),
          ("enamelled radio cabinet", "w.", 35, 60), ("tapestry design, gouache on paper", "x", 30, 70)]
NOW_YEAR = 2025


def _age_from_date(text):
    """Approximate age in years of a date text such as '1640s', 'c. 1450', '17th century' or '1958'."""
    m = re.search(r"(\d+)(st|nd|rd|th) century", text)
    if m:
        return NOW_YEAR - ((int(m.group(1)) - 1) * 100 + 50)
    m = re.search(r"(\d{4})", text)
    return NOW_YEAR - int(m.group(1)) - (5 if text.strip().endswith("s") else 0)
CONNECTION = {3: "made in the country for a figure central to its history, and shown at a defining public event",
              2: "made locally and long associated with a regional family",
              1: "bought abroad by a local collector and kept here for decades", 0: "imported in recent years"}
AESTHETIC = {3: "an outstanding work by the leading maker of its period", 2: "a fine example of its type",
             1: "a competent, ordinary example", 0: "a damaged, much-restored example"}
STUDY = {3: "the only documented example of a technique", 2: "a notable source for scholars of the period",
         1: "of modest interest to specialists", 0: "of no particular research interest"}
HOLDINGS = {"many": "public collections hold many comparable examples", "some": "public collections hold some comparable examples",
            "few": "public collections hold one or two comparable examples", "none": "no public collection holds a comparable example"}


class ExportLicenceReview(Schema):
    name = "export_licence_review"
    description = "A sold object needs a licence to leave the country; the export reviewing committee decides whether it is a national treasure."
    surfaces = {"json": 0.5, "expert_adviser_statement": 0.25, "licence_application": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        home = pick(rng, HOME_STATES)
        age_line = pick(rng, [50, 75, 100])
        value_line = pick(rng, [50000, 100000, 250000])
        o = self._export_object(rng, None if chance(rng, 0.6) else pick(rng, [20, 40, 60, 80, 95]))
        value = int(round(lognormal_amount(rng, value_line * 1.3, 0.9), -3))
        conn, aes, study = (pick(rng, [0, 0, 0, 1, 1, 2, 3]), pick(rng, [0, 0, 1, 1, 2, 3]), pick(rng, [0, 0, 0, 1, 1, 2, 3]))
        holdings = pick_w(rng, {"many": 3, "some": 3, "few": 2, "none": 1})
        title = pick_w(rng, {"clear": 7, "unclear": 2})
        if reg == "clear":
            if chance(rng, 0.5):
                o = self._export_object(rng, None, oldest=True)
                value, conn, aes, study, holdings, title = value_line * 6, 3, 3, pick(rng, [2, 3]), pick(rng, ["few", "none"]), "clear"
            else:
                o = self._export_object(rng, pick(rng, [20, age_line - 15]))
                value = int(round(value_line * 0.4, -3))
                conn, aes, study = 0, pick(rng, [0, 1]), pick(rng, [0, 1])
        elif reg == "borderline":
            if chance(rng, 0.5):
                o = self._export_object(rng, age_line + pick(rng, [-2, -1, 0, 1, 3]))
            else:
                value = int(round(near(rng, value_line, value_line * 0.08), -3))
            conn, aes, study = pick(rng, [1, 2]), pick(rng, [1, 2, 3]), pick(rng, [0, 1, 2])
        else:
            twist = pick(rng, ["treasure_with_title_doubt", "famous_but_common"])
            if twist == "treasure_with_title_doubt":
                o = self._export_object(rng, None, oldest=True)
                value, aes, title = value_line * 3, 3, "unclear"
            else:
                o = self._export_object(rng, None, oldest=True)
                value, conn, aes, study, holdings = value_line * 2, 1, 2, 1, "many"
        return {"regime": reg, **o, "home_state": home, "age_line": age_line, "value_line": value_line,
                "age_years": _age_from_date(o["date_text"]), "value": value, "connection": conn, "aesthetic": aes, "study": study,
                "holdings": holdings, "title": title, "applicant": person(rng), "adviser": person(rng),
                "buyer": pick(rng, ["a foreign private collector", "a foreign museum", "an overseas dealer"]),
                "case_id": ident(rng, "EXP", 5, "-"), "date": rand_date(rng)}

    def _export_object(self, rng, age, oldest=False):
        """An object for export review; with `age`, a modern object dated to exactly that many years ago."""
        if age is None:
            o = _object(rng, EUROPEAN_ART)
            while oldest and _age_from_date(o["date_text"]) < 150:
                o = _object(rng, EUROPEAN_ART)
            return o
        obj, prefix, lo, hi = pick(rng, MODERN)
        a = rng.randint(lo * 10, hi * 10) / 10
        dims = (f"{round(a * 10)} x {round(a * 10 * rng.uniform(0.6, 1.4))} mm" if prefix == "x"
                else f"{prefix} {a:g} cm")
        return {"object": f"{pick(rng, ['locally made', 'nationally known', 'regional'])} {obj}", "category": "modern",
                "date_text": str(NOW_YEAR - age), "dims": dims, "state": None, "state_adj": None, "state_law_year": None}

    def _required(self, f):
        return f["age_years"] >= f["age_line"] and f["value"] >= f["value_line"]

    def _significance(self, f):
        c = (f["connection"], f["aesthetic"], f["study"])
        return 3 if max(c) == 3 else 2 if sum(c) >= 4 else 1 if sum(c) >= 2 else 0

    def _criterion(self, f):
        return self._significance(f) == 3

    def _rules(self, f):
        return [f"{f['home_state']} export rules 3.1: a licence is needed to export an object at least {f['age_line']} "
                f"years old and worth at least {money(f['value_line'])}.",
                "3.2: the committee recommends deferring the licence, so that a public institution here can offer a matching "
                "price, if the object is a national treasure: (a) so closely connected with the nation's history that its "
                "departure would be a misfortune; (b) of outstanding aesthetic importance; or (c) of outstanding significance "
                "for study.",
                "3.3: no licence is recommended until the applicant shows the object was lawfully in the country."]

    def render(self, f, rng, surface):
        title = ("the applicant's family has owned it since its purchase locally, with receipts" if f["title"] == "clear"
                 else "how the object entered the country is not documented")
        if surface == "json":
            return dumps(prune({
                "application": {"id": f["case_id"], "object": _label(f), "applicant": f["applicant"],
                                "buyer": f["buyer"], "agreed_price": f["value"], "object_age_years": f["age_years"],
                                "title_history": title},
                "expert_adviser": {"name": f["adviser"], "history": CONNECTION[f["connection"]],
                                   "aesthetic": AESTHETIC[f["aesthetic"]], "study": STUDY[f["study"]],
                                   "public_holdings": HOLDINGS[f["holdings"]]},
                "rules": self._rules(f),
            }))
        if surface == "expert_adviser_statement":
            body = noisy(join_sentences([
                f"I have examined the {f['object']} ({f['dims']}; {f['date_text']}, about {f['age_years']} years old), sold to "
                f"{f['buyer']} for {money(f['value'], rng)}.", f"Historically, it was {CONNECTION[f['connection']]}.",
                f"Aesthetically, it is {AESTHETIC[f['aesthetic']]}.", f"For research it is {STUDY[f['study']]}.",
                f"{HOLDINGS[f['holdings']].capitalize()}.", f"On title: {title}."]), rng, 0.1)
            return (f"EXPERT ADVISER'S STATEMENT - {f['case_id']}\n{f['adviser']}, {fmt_date(f['date'], rng)}\n\n{body}\n\n"
                    + "\n".join(self._rules(f)))
        rows = [("Application", f["case_id"]), ("Object", _label(f)),
                ("Age", f"about {f['age_years']} years"), ("Price agreed", money(f["value"], rng)),
                ("Buyer", f["buyer"]), ("Title", title), ("Adviser on history", CONNECTION[f["connection"]]),
                ("Adviser on quality", AESTHETIC[f["aesthetic"]]), ("Adviser on research value", STUDY[f["study"]]),
                ("Public holdings", HOLDINGS[f["holdings"]])]
        return "EXPORT LICENCE APPLICATION\n" + kv_lines(rows, rng) + "\n\n" + "\n".join(self._rules(f))

    def questions(self):
        return {
            "licence_recommendation": choice("What should the reviewing committee recommend?", {
                "defer_for_matching_offer": "Defer the licence so a public institution can try to match the price.",
                "grant_licence": "Recommend the licence be granted.",
                "no_licence_needed": "Advise that no licence is needed for this object.",
                "withhold_pending_title": "Recommend no licence until the applicant shows the object was lawfully here.",
            }),
            "national_significance": score("How significant is the object to the nation's heritage?", [
                "Negligible.",
                "Modest: of local or specialist interest.",
                "Considerable: a fine object with real ties or research value.",
                "Outstanding: it meets at least one national-treasure criterion in full.",
            ]),
            "public_holdings_gap": score("How poorly are comparable objects represented in public collections?", [
                "Well represented: many comparable examples.",
                "Fairly represented: some examples.",
                "Thinly represented: one or two examples.",
                "Unrepresented: no comparable example.",
            ]),
            "licence_required": noul(
                "This object needs an export licence under the rules.",
                yes="It is at least as old as the age line and worth at least the value line.",
                no="It falls below the age line or the value line."),
            "national_treasure_criterion_met": noul("The object meets at least one of the national-treasure criteria."),
        }

    def intended(self, f):
        req, crit = self._required(f), self._criterion(f)
        if not req:
            rec = "no_licence_needed"
        elif f["title"] == "unclear":
            rec = "withhold_pending_title"
        elif crit:
            rec = "defer_for_matching_offer"
        else:
            rec = "grant_licence"
        return {"licence_recommendation": rec, "national_significance": self._significance(f),
                "public_holdings_gap": {"many": 0, "some": 1, "few": 2, "none": 3}[f["holdings"]],
                "licence_required": "yes" if req else "no", "national_treasure_criterion_met": "yes" if crit else "no"}


WORKFLOW = Workflow(
    name="museum_acquisition_provenance_review",
    title="Museum acquisition and provenance review",
    description="How a museum should decide on purchases, gifts, loans, disposals, restitution claims and export licences "
                "under its written policy and the export laws of the source countries.",
    domain="arts, entertainment and recreation / museums and heritage",
    schemas=[PurchaseProposalDueDiligence(), GiftOfferReview(), IncomingExhibitionLoan(), DeaccessionProposal(),
             RestitutionClaimAssessment(), ExportLicenceReview()],
    notes="Museums, dealers, auction houses, people and source states are fictional; each state quotes the policy or "
          "statute it is judged against. Claims turn on evidence, never on who the claimant is.",
)
