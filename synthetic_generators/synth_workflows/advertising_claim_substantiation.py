"""Advertising claim substantiation: a brand's claims-clearance team decides whether an advertising claim is
substantiated, needs qualifying or rewording, or must be pulled, against a written substantiation standard.

Every state quotes the (fictional) standard's relevant rules -- what evidence an efficacy figure needs, how
comparisons and market-position claims must be based, when green claims need qualifiers, how endorsements must
disclose connections and typical results, and how reference prices and urgency claims are tested -- so each case
is decidable in principle from the state alone. Brands, products, competitors, tests and numbers are invented and
internally consistent (claimed figures against measured results, dates against relaunch dates, days at a
reference price against the rule).

This is advertising compliance, not customer service: no question asks how to reply to a consumer, and
complaints appear only as evidence of a live problem.

Each schema carries `intended(f)`: the answer the written standard points to for each question (choice key,
'yes'/'no', or a score level index). It is never rendered and never used as a label; it exists so the answer
spread can be checked for collapsed questions before a teacher labels the data.
"""
from __future__ import annotations

from datetime import date, timedelta

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
)

CURRENCIES = ["USD", "USD", "EUR", "GBP", "CAD", "AUD"]
SYM = {"USD": "$", "EUR": "€", "GBP": "£", "CAD": "CA$", "AUD": "A$"}
BRANDS = ["Lumora", "Velstar", "Brightwick", "Norvane", "Kestrella", "Pellucid", "Aurelle", "Tindall", "Zephra",
          "Oakveil", "Crestline", "Halden", "Mirabel", "Quorra", "Sundrop", "Truvale", "Embervale", "Nimbra", "Arvelle",
          "Bramwell", "Corvia", "Dellamere", "Glimmerly", "Hartwell", "Isolde", "Juniva", "Kindred & Co", "Larkspur",
          "Morrow", "Nettlefield", "Ottoline", "Prism & Pine", "Quayside", "Rowanberry", "Silvermist", "Tamberly"]
COMPETITORS = ["Duramax", "Glowell", "Purefield", "Vantix", "Corelle Home", "Brisko", "Ultrafresh", "Nordlite",
               "Spectra One", "Freshwave", "Topline", "Evergleam", "Maxivo", "Clearbrook"]
AGENCIES = ["Northlight Creative", "Paper Kite Agency", "Saltbox Media", "Tidemark Partners", "Foxglove Studio",
            "Redbrick Advertising", "Halftone & Co"]


def _st(rng):
    return pick(rng, ["sym", "sym", "code"])


def money_k(k, cur, style):
    """An amount held in thousands, in one house style per state."""
    s = SYM.get(cur, cur + " ")
    if style == "code":
        return f"{cur} {k * 1000:,.0f}"
    return f"{s}{k:,.0f}k" if k < 1000 else f"{s}{k / 1000:,.2f}m"


def price(x, cur):
    return f"{SYM.get(cur, cur + ' ')}{x:,.2f}"


def _ds(rng):
    return pick(rng, ["iso", "long", "us", "eu", "iso"])


def _first(name):
    return name.split()[0]


def _followers(k):
    return f"{k / 1000:.1f}m" if k >= 1000 else f"{k}k"


# =========================================================================== schema 1
# (category, product noun, claim template with {x}, unit, health-related?, typical claimed range)
EFFICACY = [
    ("laundry detergent", "Deep Clean liquid", "removes {x}% more stains than our previous formula", "%", False, (20, 60)),
    ("dishwasher tablets", "PowerTabs", "cuts drying time by {x}%", "%", False, (15, 45)),
    ("power bank", "Volt 20 power bank", "charges phones {x}% faster", "%", False, (20, 70)),
    ("running shoe", "Stride Pro shoe", "returns {x}% more energy than standard foam", "%", False, (10, 40)),
    ("mattress", "CoolSleep mattress", "keeps you {x}% cooler through the night", "%", False, (10, 35)),
    ("toothpaste", "Enamel+ toothpaste", "reduces plaque by {x}% in four weeks", "%", True, (20, 60)),
    ("shampoo", "Strength shampoo", "reduces hair breakage by {x}%", "%", False, (30, 80)),
    ("exterior paint", "DuraCoat paint", "lasts {x}% longer before repainting", "%", False, (25, 100)),
    ("air purifier", "PureAir 300", "removes {x}% of airborne allergens", "%", True, (90, 99.9)),
    ("fuel additive", "MileMax additive", "improves fuel economy by up to {x}%", "%", False, (5, 15)),
    ("cordless vacuum", "CycloPro vacuum", "picks up {x}% more pet hair", "%", False, (20, 60)),
    ("loft insulation", "ThermaRoll insulation", "cuts heating bills by {x}%", "%", False, (10, 30)),
    ("sunscreen", "SunGuard SPF50", "stays water-resistant for {x} minutes", "minutes", True, (40, 120)),
    ("protein bar", "Fuel+ bar", "helps muscles recover {x}% faster", "%", True, (10, 40)),
    ("screen protector", "Shield glass", "is {x}% more scratch resistant", "%", False, (50, 300)),
    ("e-bike battery", "RangeCell battery", "gives {x}% more range on a single charge", "%", False, (10, 40)),
    ("hand cream", "Repair hand cream", "boosts skin hydration by {x}% after one use", "%", True, (30, 120)),
    ("tyre", "GripLine tyre", "stops {x}% shorter on wet roads", "%", False, (5, 20)),
]
STUDY = {
    "independent_rct": "a randomised controlled trial run by an independent laboratory",
    "independent_lab": "standardised tests at an accredited independent laboratory",
    "in_house_lab": "tests in the company's own laboratory",
    "consumer_panel": "a consumer perception survey (participants rated how well it worked)",
    "none": "no test of the claim; marketing relied on the ingredient supplier's brochure",
}
MEDIA = {"tv": "national TV spot", "social": "paid social video", "packaging": "front-of-pack text",
         "in_store": "in-store shelf barker", "online_video": "online video pre-roll", "radio": "radio advert"}


class EfficacyClaimReview(Schema):
    name = "efficacy_claim_review"
    description = ("A quantified performance claim in a campaign: does the test evidence support the figure, and "
                   "who must clear it under the substantiation standard?")
    surfaces = {"json": 0.5, "copy_clearance_form": 0.25, "brand_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        cat, noun, tmpl, unit, health, (lo, hi) = pick(rng, EFFICACY)
        claimed = round(rng.uniform(lo, hi), 1 if hi < 20 else 0)
        study = pick_w(rng, {"independent_rct": 3, "independent_lab": 3, "in_house_lab": 3, "consumer_panel": 1.5,
                             "none": 0.8})
        tested = pick_w(rng, {"as_sold": 6, "earlier_formulation": 2, "prototype": 1})
        conditions = pick_w(rng, {"real_use": 4, "lab_conditions": 4, "different_use": 0.6})
        n = pick(rng, [12, 20, 30, 48, 60, 120, 240])
        rule = pick(rng, ["mean", "lower_bound"])
        rel = pick_w(rng, {"above": 3, "between": 3, "below": 4})
        if reg == "clear":
            if chance(rng, 0.5):
                study, tested, conditions, n, rel = "independent_rct", "as_sold", "real_use", 240, "below"
            else:
                study = pick(rng, ["none", "consumer_panel"])
        elif reg == "conflicting":
            # strong study on the wrong product, or a weak test whose numbers comfortably clear the claim
            if chance(rng, 0.5):
                study, tested, rel = "independent_rct", pick(rng, ["earlier_formulation", "prototype"]), "below"
            else:
                study, rel = pick(rng, ["in_house_lab", "consumer_panel"]), "below"
        # measured mean and 95% lower bound, placed relative to the claimed figure
        width = rng.uniform(0.08, 0.3) * (1.6 if n < 40 else 1.0)
        if rel == "above":          # the claim exceeds even the mean
            mean = claimed * rng.uniform(0.6, 0.93)
        elif rel == "between":      # claim sits between the lower bound and the mean
            mean = claimed * rng.uniform(1.03, 1.15)
            width = max(width, (mean - claimed) / mean + rng.uniform(0.03, 0.12))
        else:                       # claim at or under the lower bound
            mean = claimed * rng.uniform(1.15, 1.6)
            width = min(width, 1 - claimed / mean - 0.02)
        lower = mean * (1 - width)
        dec = 1 if hi < 20 else 0
        mean, lower = round(mean, dec), round(lower, dec)
        if study == "none":
            mean = lower = None
        media = pick(rng, list(MEDIA))
        spend = int(round(lognormal_amount(rng, 250, 0.9), -1))
        return {"regime": reg, "brand": pick(rng, BRANDS), "category": cat, "product": noun, "claim_template": tmpl,
                "unit": unit, "health_related": health, "claimed": claimed, "study": study, "product_tested": tested,
                "test_conditions": conditions, "sample_size": n, "measured_mean": mean, "ci_lower": lower,
                "figure_rule": rule, "media": media, "spend_k": spend, "spend_threshold_k": pick(rng, [250, 500, 750]),
                "currency": pick(rng, CURRENCIES), "brand_manager": person(rng), "reviewer": person(rng),
                "agency": pick(rng, AGENCIES), "campaign": ident(rng, "CMP", 4, "-"),
                "launch": rand_date(rng, date(2025, 3, 1), date(2027, 6, 30))}

    def strength(self, f):
        base = {"independent_rct": 4, "independent_lab": 3, "in_house_lab": 2, "consumer_panel": 1, "none": 0}[f["study"]]
        if base and f["product_tested"] != "as_sold":
            base -= 1
        if base >= 2 and f["sample_size"] < 30:
            base -= 1
        return max(0, base)

    def match(self, f):
        if f["study"] == "none":
            return 0
        mism = (f["product_tested"] != "as_sold") + (f["test_conditions"] != "real_use") + \
            (2 if f["test_conditions"] == "different_use" else 0)
        return max(0, 3 - mism)

    def within(self, f):
        if f["measured_mean"] is None:
            return False
        bound = f["measured_mean"] if f["figure_rule"] == "mean" else f["ci_lower"]
        return f["claimed"] <= bound

    def intended(self, f):
        s, m, ok = self.strength(f), self.match(f), self.within(f)
        if s <= 1 or (f["health_related"] and s < 3) or f["test_conditions"] == "different_use":
            ruling = "pull_claim"
        elif not ok:
            ruling = "revise_figure"
        elif m < 3:
            ruling = "approve_with_qualifier"
        else:
            ruling = "approve_as_written"
        if f["health_related"] and f["media"] in ("tv", "radio"):
            route = "regulator_preclearance"
        elif f["health_related"] or f["spend_k"] >= f["spend_threshold_k"]:
            route = "outside_counsel"
        elif f["study"] in ("in_house_lab", "consumer_panel", "none"):
            route = "claims_committee"
        else:
            route = "brand_compliance_signoff"
        return {"claim_ruling": ruling, "evidence_strength": s, "test_match": m,
                "figure_within_results": "yes" if ok else "no", "clearance_route": route}

    def _standard(self, f, cur, st):
        rule = ("must not exceed the average result measured" if f["figure_rule"] == "mean" else
                "must not exceed the lower end of the 95% confidence interval of the result")
        return {
            "CSS-2.1": "Objective performance claims need competent and reliable evidence: controlled tests on the "
                       "product as sold, under conditions matching the claim. Consumer perception surveys and "
                       "supplier literature do not substantiate performance figures.",
            "CSS-2.2": f"A stated figure {rule}.",
            "CSS-2.3": "Where tests were run only under laboratory conditions, the claim must say so; a test of a "
                       "different use cannot support the claim at all.",
            "CSS-2.4": "Health-related claims need at least an independent controlled test of the product as sold.",
            "CSS-7": f"Clearance: health claims in TV or radio need regulator pre-clearance; other health claims, and "
                     f"any campaign spending {money_k(f['spend_threshold_k'], cur, st)} or more, go to outside "
                     f"counsel; claims resting on in-house or survey data go to the claims committee; everything "
                     f"else is signed off by brand compliance.",
        }

    def _claim(self, f):
        x = f"{f['claimed']:g}"
        return f"{f['brand']} {f['product']} " + f["claim_template"].format(x=x) + "."

    def _evidence(self, f):
        e = [STUDY[f["study"]].capitalize() + "."]
        if f["study"] != "none":
            e.append({"as_sold": "The product tested was the version now on sale.",
                      "earlier_formulation": "The product tested was the previous version, changed last year.",
                      "prototype": "The product tested was a pre-production prototype."}[f["product_tested"]])
            e.append({"real_use": "Testing reproduced normal everyday use.",
                      "lab_conditions": "Testing used controlled laboratory conditions, not normal everyday use.",
                      "different_use": "Testing measured a different use from the one the claim describes."}[f["test_conditions"]])
            u = "" if f["unit"] == "%" else " " + f["unit"]
            pc = "%" if f["unit"] == "%" else ""
            e.append(f"Sample size {f['sample_size']}. Average result {f['measured_mean']:g}{pc}{u}; lower end of "
                     f"the 95% confidence interval {f['ci_lower']:g}{pc}{u}.")
        return e

    def render(self, f, rng, surface):
        cur, st, ds = f["currency"], _st(rng), _ds(rng)
        std = self._standard(f, cur, st)
        claim = self._claim(f)
        ev = self._evidence(f)
        if surface == "json":
            return dumps(prune({
                "claim": {"text": claim, "brand": f["brand"], "product_category": f["category"],
                          "media": MEDIA[f["media"]], "campaign": f["campaign"],
                          "media_spend": money_k(f["spend_k"], cur, st), "launch_date": f["launch"].isoformat()},
                "substantiation": ev, "submitted_by": f"{f['brand_manager']} ({f['agency']} creative)",
                "standard": std,
            }))
        if surface == "copy_clearance_form":
            return "\n".join([
                f"CLAIM CLEARANCE REQUEST -- {f['campaign']}",
                kv_lines([("Brand", f["brand"]), ("Category", f["category"]), ("Media", MEDIA[f["media"]]),
                          ("Media spend", money_k(f["spend_k"], cur, st)), ("Launch", fmt_date(f["launch"], rng, ds)),
                          ("Requested by", f["brand_manager"]), ("Agency", f["agency"])], rng),
                "", f"Claim copy: \"{claim}\"", "", "Substantiation provided:", bullets(ev, rng), "",
                "Standard (extract):", "\n".join(f"  {k} {v}" for k, v in std.items()),
            ])
        return "\n".join([
            f"Subject: Claim sign-off needed -- {f['brand']} {f['category']} ({f['campaign']})",
            f"From: {f['brand_manager']}", f"To: {f['reviewer']}", "",
            f"Hi {_first(f['reviewer'])},", "",
            noisy(join_sentences([f"{f['agency']} has the {MEDIA[f['media']]} ready to go for "
                                  f"{fmt_date(f['launch'], rng, ds)}, with {money_k(f['spend_k'], cur, st)} of media "
                                  f"behind it. The hero line is: \"{claim}\"", "Evidence we have:"] + ev), rng, 0.12),
            "", "For reference, the standard says: " + " ".join(f"{k}: {v}" for k, v in std.items()), "",
            "Can we clear it?", _first(f["brand_manager"]),
        ])

    def questions(self):
        return {
            "claim_ruling": choice("What should the clearance team do with this claim?", {
                "approve_as_written": "Clear the claim exactly as written.",
                "approve_with_qualifier": "Clear it only with a qualifier stating the test conditions.",
                "pull_claim": "Pull the claim; the evidence cannot support it.",
                "revise_figure": "Clear the claim only once the figure is lowered to what the results support.",
            }),
            "evidence_strength": score("How strong is the evidence behind the claim?", [
                "None: nothing that tests the claim.",
                "Anecdotal: perception data or a weak test.",
                "Limited: a controlled test with real gaps.",
                "Competent: a sound controlled test.",
                "Robust: an independent randomised trial on the product as sold.",
            ]),
            "test_match": score("How closely does what was tested match what the claim says?", [
                "Not at all: nothing relevant was tested, or the wrong use was.",
                "Loosely: the product or the conditions differ, and more than one way.",
                "Closely: one difference, such as lab conditions or an earlier formulation.",
                "Exactly: the product as sold, under the conditions the claim describes.",
            ]),
            "figure_within_results": noul(
                "The figure in the claim is within what the test results support under the standard.",
                yes="The claimed figure does not exceed the measure the standard sets (the average, or the lower "
                    "end of the confidence interval).",
                no="There are no usable results, or the claimed figure exceeds that measure."),
            "clearance_route": choice("Who must clear the claim under the standard's clearance rules?", {
                "brand_compliance_signoff": "Brand compliance can sign it off.",
                "claims_committee": "The internal claims committee.",
                "outside_counsel": "Outside counsel.",
                "regulator_preclearance": "The broadcast regulator, by pre-clearance.",
            }),
        }


# =========================================================================== schema 2
COMPARATIVE = [
    # (category, attribute, claim kind)
    ("kitchen roll", "absorbency", "head_to_head"), ("razor", "closeness of shave", "head_to_head"),
    ("dishwasher tablets", "grease removal", "head_to_head"), ("smartphone plan", "4G coverage", "head_to_head"),
    ("dog food", "palatability", "leading_brand"), ("battery", "run time in toys", "head_to_head"),
    ("coffee pods", "taste preference", "leading_brand"), ("deodorant", "48-hour freshness", "leading_brand"),
    ("oat drink", "unit sales", "number_one"), ("sparkling water", "unit sales", "number_one"),
    ("broadband service", "average download speed", "head_to_head"), ("toilet cleaner", "limescale removal",
                                                                       "leading_brand"),
    ("frozen pizza", "unit sales", "number_one"), ("car wax", "water beading after 30 days", "head_to_head"),
    ("face wipes", "make-up removal", "leading_brand"), ("cat litter", "odour control", "head_to_head"),
    ("energy drink", "value sales", "number_one"), ("bin bags", "tear resistance", "head_to_head"),
]
SCOPES = [("convenience stores in the north region", "nationwide"), ("supermarkets only", "across all retailers"),
          ("online sales", "in all channels"), ("the 12 months to March", "of all time"),
          ("the premium segment", "in the whole category")]


class ComparativeClaimReview(Schema):
    name = "comparative_claim_review"
    description = ("A claim comparing the product with a competitor or the market: is the comparison fair, current "
                   "and within its data?")
    surfaces = {"json": 0.5, "legal_review_note": 0.25, "copy_deck": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        cat, attr, kind = pick(rng, COMPARATIVE)
        named = kind == "head_to_head" or chance(rng, 0.2)
        like = chance(rng, 0.52) if kind != "number_one" else True
        sig = chance(rng, 0.8) if kind != "number_one" else True
        age = pick_w(rng, {3: 2, 5: 2, 9: 3, 15: 3, 26: 1})
        test_date = rand_date(rng, date(2023, 6, 1), date(2026, 6, 1))
        relaunch = None
        if kind != "number_one" and chance(rng, 0.25):
            relaunch = test_date + timedelta(days=rng.randint(20, 300))
        scope_broad = chance(rng, 0.25)
        if reg == "clear":
            if chance(rng, 0.5):
                like, sig, age, relaunch, scope_broad = True, True, 3, None, False
            else:
                like, sig = (False, False) if kind != "number_one" else (True, True)
                if kind == "number_one":
                    scope_broad, age = True, 26
        elif reg == "conflicting":
            # strong, significant test result -- on a competitor version since replaced, or a narrower dataset
            sig = True
            if kind != "number_one" and chance(rng, 0.6):
                relaunch = test_date + timedelta(days=rng.randint(20, 200))
            else:
                scope_broad = True
        review = test_date + timedelta(days=int(age * 30.4))
        if relaunch and relaunch > review:
            review = relaunch + timedelta(days=rng.randint(10, 60))
        return {"regime": reg, "brand": pick(rng, BRANDS), "competitor": pick(rng, COMPETITORS), "category": cat,
                "attribute": attr, "kind": kind, "competitor_named": named, "like_for_like": like,
                "significant": sig, "data_age_months": age, "test_date": test_date, "competitor_relaunch": relaunch,
                "review_date": review, "scope_pair": pick(rng, SCOPES), "scope_broader": scope_broad,
                "margin_pct": pick(rng, [8, 12, 15, 20, 25, 30, 40]), "sample": pick(rng, [60, 120, 200, 400, 800]),
                "share_pct": round(rng.uniform(18, 34), 1), "runner_up_pct": None,
                "complaint": chance(rng, 0.25), "reviewer": person(rng), "marketer": person(rng),
                "copy_id": ident(rng, "CD", 4, "-")}

    def superseded(self, f):
        return f["competitor_relaunch"] is not None and f["competitor_relaunch"] > f["test_date"]

    def intended(self, f):
        sup = self.superseded(f)
        age = f["data_age_months"]
        cur = 0 if (sup or age > 24) else 1 if age > 12 else 2 if age > 6 else 3
        pts = (not f["like_for_like"]) + sup + f["scope_broader"] + (not f["significant"]) + (age > 12)
        chal = min(3, pts)
        if not f["significant"] or not f["like_for_like"]:
            ruling = "withdraw_claim"
        elif sup or age > 12:
            ruling = "refresh_the_evidence"
        elif f["scope_broader"]:
            ruling = "narrow_the_claim"
        elif f["kind"] in ("number_one", "leading_brand"):
            ruling = "approve_with_basis_footnote"
        else:
            ruling = "approve"
        return {"comparison_ruling": ruling, "data_currency": cur, "challenge_success_likelihood": chal,
                "like_for_like_test": "yes" if f["like_for_like"] else "no",
                "competitor_version_superseded": "yes" if sup else "no"}

    STANDARD = {
        "CSS-4.1": "Comparisons must test equivalent products under the same conditions, and the advantage claimed "
                   "must be statistically significant.",
        "CSS-4.2": "Comparative evidence must reflect the products currently on sale: if the competitor's product "
                   "has changed since testing, or the data is more than 12 months old, refresh it before running.",
        "CSS-4.3": "A claim may not be wider than its data (channel, region, period, segment).",
        "CSS-4.4": "Claims against an unnamed 'leading brand', and market-position claims, must state their basis "
                   "in a footnote.",
    }

    def _claim(self, f):
        if f["kind"] == "head_to_head":
            return f"{f['margin_pct']}% better {f['attribute']} than {f['competitor']}"
        if f["kind"] == "leading_brand":
            return f"Beats the leading brand on {f['attribute']}"
        return f"The No.1 {f['category']} " + (f["scope_pair"][1] if f["scope_broader"] else "in " + f["scope_pair"][0])

    def _evidence(self, f, rng, ds):
        ev = []
        if f["kind"] == "number_one":
            ev.append(f"Retail panel data for {f['scope_pair'][0]}: {f['brand']} {f['share_pct']}% of {f['attribute']}, "
                      f"ahead of every other brand.")
            if not f["scope_broader"]:
                ev[-1] += " The claim copy limits itself to the same channel and period."
        else:
            vs = f["competitor"] if f["competitor_named"] else f"the leading brand ({f['competitor']})"
            ev.append(f"Test on {fmt_date(f['test_date'], rng, ds)}: {f['sample']} samples against {vs}; "
                      f"advantage of {f['margin_pct']}% on {f['attribute']}"
                      + ("." if f["significant"] else ", not statistically significant at the 95% level."))
            ev.append("Both products were tested in their standard versions under identical conditions."
                      if f["like_for_like"] else
                      f"The test compared {f['brand']}'s premium variant with {f['competitor']}'s value variant.")
            if f["scope_broader"]:
                ev.append(f"The test covered {f['scope_pair'][0]}; the copy makes the claim {f['scope_pair'][1]}.")
            if f["competitor_relaunch"]:
                ev.append(f"{f['competitor']} relaunched the product with a new formula on "
                          f"{fmt_date(f['competitor_relaunch'], rng, ds)}.")
        ev.append(f"Evidence is {f['data_age_months']} months old at review.")
        if f["complaint"]:
            ev.append(f"{f['competitor']}'s legal team has written objecting to a similar claim in another market.")
        return ev

    def render(self, f, rng, surface):
        ds = _ds(rng)
        claim = self._claim(f)
        ev = self._evidence(f, rng, ds)
        if surface == "json":
            return dumps(prune({
                "copy": {"id": f["copy_id"], "brand": f["brand"], "category": f["category"], "claim": claim,
                         "review_date": f["review_date"].isoformat()},
                "evidence": ev, "standard": self.STANDARD,
            }))
        if surface == "legal_review_note":
            return "\n".join([
                f"LEGAL REVIEW -- comparative claim -- {f['copy_id']}",
                f"Reviewer: {f['reviewer']}   Date: {fmt_date(f['review_date'], rng, ds)}", "",
                f"Proposed claim ({f['brand']} {f['category']}): \"{claim}\"", "",
                "Evidence file:", bullets(ev, rng), "",
                "Standard:", "\n".join(f"  {k} {v}" for k, v in self.STANDARD.items()),
            ])
        return "\n".join([
            f"COPY DECK {f['copy_id']} -- {f['brand']} {f['category']} spring burst",
            f"Owner: {f['marketer']}", "",
            f"HEADLINE: {claim.upper()}", f"Body: {noisy(pick(rng, ['Switch today.', 'See the difference.', 'Try it for yourself.']), rng, 0.1)}",
            "", "Support (from the marketer's notes):", "\n".join(f"- {noisy(x, rng, 0.08)}" for x in ev), "",
            "Clearance rules: " + " ".join(f"[{k}] {v}" for k, v in self.STANDARD.items()),
        ])

    def questions(self):
        return {
            "comparison_ruling": choice("What should happen to this comparative claim?", {
                "approve": "Run it as written.",
                "approve_with_basis_footnote": "Run it with a footnote stating the basis of the comparison.",
                "narrow_the_claim": "Narrow the wording to what the data covers.",
                "refresh_the_evidence": "Hold it until the comparison is re-run against current products or data.",
                "withdraw_claim": "Withdraw it; the comparison does not stand up.",
            }),
            "data_currency": score("How current is the evidence behind the comparison?", [
                "Out of date: over two years old, or the competitor has since changed its product.",
                "Ageing: one to two years old.",
                "Recent: six to twelve months old.",
                "Current: under six months old, against the products now on sale.",
            ]),
            "challenge_success_likelihood": score("If the competitor challenged the claim, how likely is the "
                                                  "challenge to succeed?", [
                "Unlikely.",
                "Possible.",
                "Likely.",
                "Almost certain.",
            ]),
            "like_for_like_test": noul(
                "The comparison set equivalent products against each other under the same conditions.",
                yes="Equivalent versions were measured the same way (or the claim rests on market data measured "
                    "identically for every brand).",
                no="Different tiers or variants, or different conditions, were compared."),
            "competitor_version_superseded": noul(
                "The competitor product used in the comparison has since been replaced by a new version."),
        }


# =========================================================================== schema 3
GREEN = {
    "recyclable": "Fully recyclable packaging",
    "recycled_content": "Made from {x}% recycled plastic",
    "carbon_neutral": "Carbon neutral {p}",
    "compostable": "Compostable {p}",
    "vague": "The eco-friendly choice",
    "co2_reduction": "{x}% less CO2 than our previous model",
}
VERIF = {"self_declared": "the company's own statement", "supplier_statement": "a supplier's declaration",
         "third_party_tested": "tests by an independent laboratory",
         "accredited_certification": "certification by an accredited scheme"}
GREEN_PRODUCTS = {
    "recyclable": ["shampoo", "yoghurt pots", "coffee capsules", "takeaway cups", "laundry liquid", "crisps",
                   "toothpaste", "pet food pouches", "hand soap", "ready meals"],
    "recycled_content": ["shampoo bottle", "phone case", "trainers", "bin liners", "water bottle", "garden chair",
                         "fleece jacket", "storage boxes", "toothbrush", "dish soap bottle"],
    "carbon_neutral": ["sparkling water", "kettle", "ground coffee", "trainers", "face cream", "notebooks", "oat drink",
                       "running jacket", "chocolate bar"],
    "compostable": ["coffee capsules", "laundry pods", "takeaway cups", "tea bags", "bin liners", "cutlery set",
                    "phone case", "nappies"],
    "vague": ["trainers", "dish soap", "kettle", "face cream", "notebooks", "toothbrush", "paint", "sofa", "jeans",
              "laundry liquid"],
    "co2_reduction": ["kettle", "dishwasher", "trainers", "phone case", "office chair", "washing machine", "e-bike",
                      "laptop", "jeans"],
}


class EnvironmentalClaimReview(Schema):
    name = "environmental_claim_review"
    description = ("A green claim on a product or its packaging: is it supported as worded, does it need a "
                   "qualifier, or must it be replaced or removed?")
    surfaces = {"json": 0.5, "sustainability_dossier": 0.25, "packaging_brief": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        ct = pick_w(rng, {"recyclable": 3.2, "recycled_content": 2.5, "carbon_neutral": 1.2, "compostable": 2.2,
                          "vague": 1.6, "co2_reduction": 2})
        thresh = pick(rng, [60, 70, 75])
        verif = pick_w(rng, {"self_declared": 1.2, "supplier_statement": 1.2, "third_party_tested": 3,
                             "accredited_certification": 3})
        f = {"regime": reg, "claim_type": ct, "access_threshold_pct": thresh, "verification": verif,
             "brand": pick(rng, BRANDS), "product": pick(rng, GREEN_PRODUCTS[ct]), "manager": person(rng),
             "analyst": person(rng), "ref": ident(rng, "ENV", 4, "-"), "date": rand_date(rng)}
        good = reg == "clear" and chance(rng, 0.5)
        bad = reg == "clear" and not good
        if ct == "recyclable":
            access = round(rng.uniform(thresh + 3, 95)) if good else round(rng.uniform(20, thresh - 25)) if bad \
                else int(near(rng, thresh + 5, 16, lo=15, hi=95, integer=True))
            if access == thresh and not good:
                access -= 1
            f.update(access_pct=access, whole_pack=good or chance(rng, 0.6))
        elif ct == "recycled_content":
            claimed = pick(rng, [30, 50, 75, 100])
            actual = claimed + rng.randint(0, 10) if good else (claimed * rng.uniform(0.2, 0.5) if bad else
                                                                claimed * rng.uniform(0.9, 1.2))
            f.update(claimed=claimed, actual_pct=min(100, int(round(actual))), whole_pack=good or chance(rng, 0.6))
        elif ct == "carbon_neutral":
            off = "verified_removals" if good else ("unverified" if bad else
                                                    pick_w(rng, {"verified_removals": 2, "verified_avoidance": 3,
                                                                 "unverified": 2}))
            f.update(offsets=off, reduction_plan=good or chance(rng, 0.55),
                     lifecycle_scope=pick(rng, ["full_lifecycle", "factory_only"]) if not good else "full_lifecycle")
        elif ct == "compostable":
            f.update(compost_cert=("home" if good else "none" if bad else
                                   pick_w(rng, {"home": 3, "industrial": 4, "none": 1})),
                     industrial_access_pct=rng.randint(8, 45))
        elif ct == "vague":
            f.update(specific_attribute=pick(rng, [None, "40% recycled cardboard packaging",
                                                   "refill packs that use 70% less plastic",
                                                   "a plant-based formula"]) if not bad else None)
        else:
            claimed = pick(rng, [15, 20, 25, 30, 40])
            actual = claimed + rng.randint(0, 8) if good else (claimed * rng.uniform(0.3, 0.6) if bad else
                                                               claimed * rng.uniform(0.85, 1.2))
            f.update(claimed=claimed, actual_pct=int(round(actual)),
                     lca_scope="cradle_to_grave" if good else pick(rng, ["cradle_to_grave", "cradle_to_gate"]),
                     reviewed=good or chance(rng, 0.65))
        if good:
            f["verification"] = pick(rng, ["third_party_tested", "accredited_certification"])
        elif bad and chance(rng, 0.6):
            f["verification"] = pick(rng, ["self_declared", "supplier_statement"])
        if ct == "co2_reduction":
            f["verification"] = (pick(rng, ["third_party_tested", "accredited_certification"]) if f["reviewed"]
                                 else pick(rng, ["self_declared", "supplier_statement"]))
        if reg == "conflicting" and ct != "co2_reduction":
            # strong certification of a claim worded more broadly than the certified fact, or the reverse
            f["verification"] = pick(rng, ["accredited_certification", "accredited_certification", "self_declared"])
            if ct in ("recyclable", "recycled_content") and chance(rng, 0.6):
                f["whole_pack"] = False
        f["claim_text"] = GREEN[ct].format(x=f.get("claimed", ""), p=f["product"])
        return f

    def assess(self, f):
        """-> (status, qualifier, overstatement level) under the standard."""
        ct = f["claim_type"]
        if ct == "recyclable":
            a, t = f["access_pct"], f["access_threshold_pct"]
            if a < t - 25:
                return "remove", None, 3
            if a >= t and f["whole_pack"]:
                return "ok", None, 0
            if a >= t:
                return "qualify", "scope", 1
            return "qualify", "availability", 2 if a < t - 10 or not f["whole_pack"] else 1
        if ct == "recycled_content":
            r = f["actual_pct"] / f["claimed"]
            if r < 0.6:
                return "remove", None, 3
            if r >= 1 and f["whole_pack"]:
                return "ok", None, 0
            if r >= 1:
                return "qualify", "scope", 1
            return "remove", None, 2
        if ct == "carbon_neutral":
            if f["offsets"] == "unverified" or not f["reduction_plan"]:
                return "remove", None, 3 if f["offsets"] == "unverified" else 2
            return "qualify", "method", 1 if f["offsets"] == "verified_removals" and f["lifecycle_scope"] == "full_lifecycle" else 2
        if ct == "compostable":
            c = f["compost_cert"]
            if c == "home":
                return "ok", None, 0
            if c == "industrial":
                return "qualify", "availability", 1 if f["industrial_access_pct"] >= 30 else 2
            return "remove", None, 3
        if ct == "vague":
            return ("replace", None, 2) if f["specific_attribute"] else ("remove", None, 3)
        r = f["actual_pct"] / f["claimed"]
        if r < 0.8:
            return "remove", None, 3 if r < 0.6 else 2
        if r >= 1 and f["lca_scope"] == "cradle_to_grave" and f["reviewed"]:
            return "ok", None, 0
        if r >= 1:
            return "qualify", "method", 1
        return "remove", None, 2

    def intended(self, f):
        status, q, over = self.assess(f)
        weak_proof = f["verification"] in ("self_declared", "supplier_statement")
        if status == "ok" and weak_proof:
            status, q = "qualify", "method"
            over = max(over, 1)
        ruling = {"ok": "approve", "qualify": "approve_with_qualifier", "replace": "replace_with_specific_claim",
                  "remove": "remove_claim"}[status]
        qual = {"ok": "no_qualifier_needed", "replace": "no_qualifier_can_fix",
                "remove": "no_qualifier_can_fix"}.get(status) or {"scope": "scope_qualifier",
                                                                  "availability": "availability_qualifier",
                                                                  "method": "method_qualifier"}[q]
        indep = {"self_declared": 0, "supplier_statement": 1, "third_party_tested": 2,
                 "accredited_certification": 3}[f["verification"]]
        return {"green_claim_ruling": ruling, "required_qualifier": qual, "overstatement_degree": over,
                "evidence_independence": indep, "unqualified_claim_supported": "yes" if ruling == "approve" else "no"}

    def _standard(self, f):
        return {
            "ENV-1": "Green claims must be specific; general terms such as 'eco-friendly' or 'green' are not allowed "
                     "unless replaced by the specific attribute that can be proved.",
            "ENV-2": f"'Recyclable' may be used without qualification only if collection is available to at least "
                     f"{f['access_threshold_pct']}% of households and every component is recyclable; otherwise "
                     f"say which part is recyclable or that facilities vary, and drop the claim where access is far "
                     f"below that level.",
            "ENV-3": "Content and reduction figures must not exceed the verified figure and must say which part of "
                     "the product or which lifecycle stages they cover.",
            "ENV-4": "Carbon-neutral claims need verified offsets plus a published reduction plan, and must say "
                     "that offsets are used. 'Compostable' alone means home-compostable; industrial-only "
                     "certification must be stated.",
            "ENV-5": "Evidence must be independent: a company's or supplier's own statement supports only a "
                     "claim that says how it was established.",
        }

    def _facts(self, f):
        ct = f["claim_type"]
        if ct == "recyclable":
            fx = [f"Kerbside collection for this material reaches {f['access_pct']}% of households in the market.",
                  "All components (body, cap, label) are recyclable." if f["whole_pack"] else
                  "One component (a cap, pump or film lid) is a mixed material and is not recyclable."]
        elif ct == "recycled_content":
            fx = [f"Verified recycled content: {f['actual_pct']}%.",
                  "The figure applies to the whole pack." if f["whole_pack"] else
                  "The figure applies to the main component only; the other parts and the packaging are virgin "
                  "plastic."]
        elif ct == "carbon_neutral":
            fx = [{"verified_removals": "Residual emissions offset with verified carbon-removal credits.",
                   "verified_avoidance": "Residual emissions offset with verified avoided-deforestation credits.",
                   "unverified": "Offsets bought through a broker; the projects are not on any recognised registry."}[f["offsets"]],
                  "A reduction plan with yearly targets is published." if f["reduction_plan"] else
                  "There is no published plan to cut the product's own emissions.",
                  "Footprint covers the full lifecycle." if f["lifecycle_scope"] == "full_lifecycle" else
                  "Footprint covers factory emissions only."]
        elif ct == "compostable":
            fx = [{"home": "Certified home-compostable by an accredited scheme.",
                   "industrial": f"Certified compostable in industrial facilities only; about "
                                 f"{f['industrial_access_pct']}% of households can use one.",
                   "none": "No composting certification; the material is 'plant-based'."}[f["compost_cert"]]]
        elif ct == "vague":
            fx = [f"The product's provable environmental feature: {f['specific_attribute']}."
                  if f["specific_attribute"] else "No specific environmental feature has been documented."]
        else:
            fx = [f"Lifecycle assessment shows a {f['actual_pct']}% reduction in CO2 per unit.",
                  "The assessment covers manufacture, use and disposal." if f["lca_scope"] == "cradle_to_grave" else
                  "The assessment covers manufacture only (raw materials to factory gate).",
                  "It was critically reviewed by an independent panel." if f["reviewed"] else
                  "It has not been independently reviewed."]
        fx.append(f"Source of evidence: {VERIF[f['verification']]}.")
        return fx

    def render(self, f, rng, surface):
        ds = _ds(rng)
        std = self._standard(f)
        fx = self._facts(f)
        if surface == "json":
            return dumps(prune({
                "claim": {"ref": f["ref"], "brand": f["brand"], "product": f["product"], "wording": f["claim_text"],
                          "placement": pick(rng, ["front of pack", "on-shelf label", "online product page",
                                                  "TV end-frame", "social campaign"])},
                "evidence": fx, "submitted_by": f["manager"], "standard": std,
            }))
        if surface == "sustainability_dossier":
            return "\n".join([
                f"ENVIRONMENTAL CLAIM DOSSIER {f['ref']} ({fmt_date(f['date'], rng, ds)})",
                f"{f['brand']} {f['product']} -- proposed wording: \"{f['claim_text']}\"",
                f"Compiled by {f['analyst']}", "", "Evidence summary:", bullets(fx, rng), "",
                "Applicable rules:", "\n".join(f"  {k} {v}" for k, v in std.items()),
            ])
        return "\n".join([
            f"PACK BRIEF -- {f['brand']} {f['product']} redesign",
            f"From {f['manager']} to the claims team", "",
            noisy(join_sentences([f"We want \"{f['claim_text']}\" on the new pack, big and front-of-pack.",
                                  pick(rng, ["Retailers are asking for it.", "Competitors already say similar things.",
                                             "This is the centrepiece of the relaunch.", ""])] + fx), rng, 0.12), "",
            "Rules to apply: " + " ".join(f"{k}: {v}" for k, v in std.items()),
        ])

    def questions(self):
        return {
            "green_claim_ruling": choice("What should happen to this environmental claim?", {
                "approve": "Approve the wording as it stands.",
                "approve_with_qualifier": "Approve it with a qualifier added.",
                "remove_claim": "Remove the claim; the evidence does not support it in any form.",
                "replace_with_specific_claim": "Replace the general claim with the specific feature that can be proved.",
            }),
            "required_qualifier": choice("What kind of qualifier does the claim need?", {
                "availability_qualifier": "One saying the benefit depends on facilities available to the consumer.",
                "method_qualifier": "One saying how the result is achieved or measured, such as offsets, the "
                                    "lifecycle stages covered or who established it.",
                "no_qualifier_can_fix": "None would do: the claim has to be replaced or removed.",
                "no_qualifier_needed": "None: the claim stands as worded.",
                "scope_qualifier": "One saying which part of the product or pack the claim covers.",
            }),
            "overstatement_degree": score("How far does the wording overstate what the evidence shows?", [
                "Not at all.",
                "Slightly: true, but missing a necessary limit.",
                "Materially: a consumer would be misled about the benefit.",
                "Grossly: the evidence does not support the claim at all.",
            ]),
            "evidence_independence": score("How independent is the evidence behind the claim?", [
                "Self-declared by the company.",
                "A supplier's own declaration.",
                "Tested by an independent laboratory.",
                "Certified under an accredited scheme.",
            ]),
            "unqualified_claim_supported": noul(
                "The claim can run as worded, with no qualifier, under the standard.",
                yes="The independent evidence meets every condition the standard sets for this kind of claim.",
                no="A condition fails, so the claim needs a qualifier, a replacement or removal."),
        }


# =========================================================================== schema 4
RESULT_PRODUCTS = [
    # (product, metric, unit, typical value range, testimonial phrase)
    ("weight-management app", "weight lost in 12 weeks", "kg", (2, 4), "I lost {v} in twelve weeks"),
    ("anti-ageing serum", "reduction in visible wrinkles", "%", (8, 15), "my fine lines are down {v}"),
    ("budgeting app", "savings in the first year", "USD", (300, 700), "I saved {v} in my first year"),
    ("language-learning app", "words learned in 30 days", "words", (150, 400), "{v} learned in a month"),
    ("hair-growth serum", "increase in hair density", "%", (5, 12), "my hair is {v} thicker"),
    ("home-workout programme", "gain in push-ups after 8 weeks", "reps", (4, 9), "{v} more push-ups after eight weeks"),
    ("sleep-tracking ring", "extra sleep per night", "minutes", (8, 20), "{v} more sleep every night"),
    ("acne cleanser", "fewer blemishes after 4 weeks", "%", (15, 30), "{v} fewer breakouts in a month"),
    ("cashback credit-card app", "cashback earned in a year", "USD", (60, 180), "{v} cashback in a year"),
    ("meal-kit subscription", "money saved on groceries per month", "USD", (20, 60), "I save {v} a month on food"),
    ("posture corrector", "less back pain reported", "%", (10, 25), "{v} less back pain"),
    ("typing course", "words-per-minute gain", "wpm", (8, 18), "typing {v} faster"),
]
ENDORSERS = {
    "paid_influencer": "a lifestyle creator paid a fee for the post",
    "gifted_creator": "a creator who received the product free",
    "affiliate": "a reviewer who earns commission on sales through a personal link",
    "employee": "a member of the brand's own staff",
    "incentivised_customer": "a customer who received a discount voucher in exchange for a review",
    "paid_expert": "a practising professional paid as a brand ambassador",
    "customer": "an ordinary customer who bought the product at full price and received nothing for the testimonial",
}
DISCLOSURE = {
    "none": "No mention of any relationship with the brand.",
    "hashtag_buried": "'#sp' appears as the 23rd of 25 hashtags, below the 'more' fold.",
    "caption_end": "The last line of a long caption reads 'thanks to {brand} for this'.",
    "upfront": "The first words of the caption are 'Paid partnership with {brand}'.",
    "label_and_verbal": "The platform's paid-partnership label is on, and the creator says 'this is sponsored' "
                        "in the opening seconds.",
}
PLATFORMS = ["short-form video", "24-hour story", "podcast read", "blog review", "TV testimonial ad",
             "photo carousel", "livestream"]


class TestimonialEndorsementReview(Schema):
    name = "testimonial_endorsement_review"
    description = ("An endorsement or testimonial in advertising: is the connection disclosed, are the results "
                   "typical, and is the endorser's account genuine?")
    surfaces = {"json": 0.5, "post_capture": 0.25, "endorsement_review_form": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        prod, metric, unit, (tlo, thi), phrase = pick(rng, RESULT_PRODUCTS)
        dec = 1 if unit in ("kg",) else 0
        etype = pick_w(rng, {"paid_influencer": 3, "gifted_creator": 2, "affiliate": 2, "employee": 1,
                             "incentivised_customer": 2, "paid_expert": 1, "customer": 3.5})
        disc = pick_w(rng, {"none": 2, "hashtag_buried": 2, "caption_end": 3, "upfront": 1.2, "label_and_verbal": 1.2})
        typical = round(rng.uniform(tlo, thi), dec)
        mult = pick_w(rng, {1.1: 2, 1.6: 2, 2.5: 2, 3.5: 2, 7: 1})
        use = pick_w(rng, {"verified": 3, "plausible": 3, "doubtful": 2, "scripted": 1})
        stmt = chance(rng, 0.35)
        if reg == "clear":
            if chance(rng, 0.5):
                disc, mult, use, stmt = "label_and_verbal", 1.1, "verified", True
            else:
                disc, mult, use = "none", 7, pick(rng, ["doubtful", "scripted"])
        elif reg == "conflicting":
            # flawless disclosure on an untypical result, or a genuine, typical story with no disclosure
            if chance(rng, 0.5):
                disc, mult, stmt, use = "label_and_verbal", pick(rng, [3.5, 7]), False, "verified"
            else:
                disc, mult, use = pick(rng, ["none", "hashtag_buried"]), 1.1, "verified"
        if etype == "customer":
            disc = "none"
            if use == "scripted":
                use = "doubtful"
        featured = round(typical * mult * rng.uniform(0.95, 1.05), dec)
        share = {1.1: rng.randint(40, 60), 1.6: rng.randint(18, 35), 2.5: rng.randint(6, 15), 3.5: rng.randint(2, 6),
                 7: rng.randint(0, 1)}[mult]
        return {"regime": reg, "brand": pick(rng, BRANDS), "product": prod, "metric": metric, "unit": unit,
                "phrase": phrase,
                "typical": typical, "featured": featured, "share_achieving_pct": share,
                "typical_statement": stmt, "endorser_type": etype, "disclosure": disc, "genuine_use": use,
                "platform": pick(rng, PLATFORMS), "handle": "@" + pick(rng, ["move", "glow", "daily", "real", "simply",
                                                                             "the", "just"]) +
                pick(rng, ["withmaya", "byjonah", "ellie", "_tom", "lifeoflena", "withraf", "sam.k", "dana_d"]),
                "followers_k": pick(rng, [4, 18, 55, 120, 480, 1300]), "reviewer": person(rng),
                "brief_owner": person(rng), "post_date": rand_date(rng), "ref": ident(rng, "END", 4, "-")}

    def intended(self, f):
        conn = f["endorser_type"] != "customer"
        clar = {"none": 0, "hashtag_buried": 1, "caption_end": 2, "upfront": 3, "label_and_verbal": 3}[f["disclosure"]]
        if not conn:
            clar = 3
        ratio = f["featured"] / f["typical"]
        gap = 0 if (ratio <= 1.25 or f["share_achieving_pct"] >= 40) else 1 if ratio <= 2 else 2 if ratio <= 5 else 3
        genuine = f["genuine_use"] in ("verified", "plausible")
        if not genuine:
            ruling = "pull_endorsement"
        elif conn and clar < 3:
            ruling = "fix_disclosure"
        elif gap >= 2 and not f["typical_statement"]:
            ruling = "add_typical_results_statement"
        else:
            ruling = "run_as_is"
        return {"endorsement_ruling": ruling, "typicality_gap": gap, "disclosure_clarity": clar,
                "material_connection": "yes" if conn else "no", "genuine_use": "yes" if genuine else "no"}

    STANDARD = {
        "END-1": "Any connection the audience would not expect -- payment, free product, commission, employment, "
                 "discounts for reviews -- must be disclosed clearly and up front, where it cannot be missed. "
                 "Disclosures at the end of captions or among hashtags are not clear.",
        "END-2": "Testimonials showing results well beyond what users generally achieve must state the typical "
                 "result alongside them.",
        "END-3": "Endorsers must have genuinely used the product as sold, and the testimonial must reflect their "
                 "honest experience; scripted results or reviews posted before real use must be pulled.",
    }

    def _use_text(self, f):
        return {"verified": "Order records show the endorser bought or received the product four months before posting "
                            "and used it throughout.",
                "plausible": "No usage records, but the post describes the routine in consistent, specific detail.",
                "doubtful": "The product was delivered to the endorser six days before the post, which describes "
                            "results 'after twelve weeks'.",
                "scripted": "The brand's brief supplied the script, including the result figure, before the product "
                            "was sent."}[f["genuine_use"]]

    def _fmt(self, v, unit):
        if unit == "USD":
            return f"${v:,.0f}"
        if unit == "%":
            return f"{v:g}%"
        return f"{v:g} {unit}"

    def render(self, f, rng, surface):
        ds = _ds(rng)
        disc = DISCLOSURE[f["disclosure"]].format(brand=f["brand"])
        feat, typ = self._fmt(f["featured"], f["unit"]), self._fmt(f["typical"], f["unit"])
        claim = (f"\"{pick(rng, ['Honestly life-changing', 'I could not believe it', 'Game changer', 'Still in shock'])}: "
                 + f["phrase"].format(v=feat) + f" with {f['brand']}!\"")
        res = (f"Brand study ({f['metric']}): the average user result is {typ}; {f['share_achieving_pct']}% of "
               f"users reach {feat} or more.")
        stmt = ("The ad shows 'typical result: " + typ + "' on screen." if f["typical_statement"] else
                "No typical-result statement appears.")
        who = ENDORSERS[f["endorser_type"]]
        if surface == "json":
            return dumps(prune({
                "endorsement": {"ref": f["ref"], "brand": f["brand"], "product": f["product"], "platform": f["platform"],
                                "endorser": f"{f['handle']} ({_followers(f['followers_k'])} followers)", "relationship": who,
                                "testimonial": claim, "posted": f["post_date"].isoformat()},
                "disclosure": disc, "results": {"evidence": res, "on_screen": stmt},
                "use_check": self._use_text(f), "standard": self.STANDARD,
            }))
        if surface == "post_capture":
            return "\n".join([
                f"CAPTURED POST -- {f['platform']} -- {fmt_date(f['post_date'], rng, ds)}",
                f"{f['handle']} · {_followers(f['followers_k'])} followers", "",
                noisy(claim.strip('"'), rng, 0.1), "", f"[Disclosure as captured] {disc}", f"[On-screen] {stmt}", "",
                f"Reviewer notes ({f['reviewer']}):",
                bullets([f"Endorser is {who}.", res, self._use_text(f)], rng), "",
                "Standard: " + " ".join(f"[{k}] {v}" for k, v in self.STANDARD.items()),
            ])
        return "\n".join([
            f"ENDORSEMENT REVIEW FORM {f['ref']}",
            kv_lines([("Brand / product", f"{f['brand']} {f['product']}"), ("Platform", f["platform"]),
                      ("Endorser", f"{f['handle']} -- {who}"), ("Testimonial", claim), ("Disclosure", disc),
                      ("Typical-result statement", "yes" if f["typical_statement"] else "no"),
                      ("Results evidence", res), ("Use check", self._use_text(f)),
                      ("Brief owner", f["brief_owner"])], rng), "",
            "Rules:", "\n".join(f"  {k} {v}" for k, v in self.STANDARD.items()),
        ])

    def questions(self):
        return {
            "endorsement_ruling": choice("What should happen to this endorsement?", {
                "add_typical_results_statement": "Keep it, but add the typical result alongside the featured one.",
                "fix_disclosure": "Keep it, but make the connection to the brand clear and up front.",
                "pull_endorsement": "Pull it; the endorsement does not reflect genuine use.",
                "run_as_is": "Let it run as it is.",
            }),
            "typicality_gap": score("How far does the featured result exceed what users typically achieve?", [
                "Not meaningfully: it is a common result.",
                "Somewhat: better than average, but not unusual.",
                "Well beyond: a result few users reach.",
                "Exceptional: almost no one achieves it.",
            ]),
            "disclosure_clarity": score("How clearly is the endorser's connection to the brand disclosed?", [
                "Not at all, although a connection exists.",
                "Hidden: technically present but easy to miss.",
                "Visible but late or vague.",
                "Clear and up front, or there is no connection to disclose.",
            ]),
            "material_connection": noul(
                "The endorser has a connection to the brand that the audience would not expect.",
                yes="Payment, free product, commission, employment or a review incentive links the endorser to the "
                    "brand.",
                no="The endorser is an ordinary customer with no such link."),
            "genuine_use": noul("The testimonial reflects the endorser's genuine use of the product as sold."),
        }


# =========================================================================== schema 5
PROMO_PRODUCTS = [("air fryer", 129.0), ("cordless drill", 89.0), ("sofa", 1299.0), ("running shoes", 110.0),
                  ("television", 649.0), ("mattress", 799.0), ("coffee machine", 249.0), ("winter coat", 180.0),
                  ("kids' bike", 159.0), ("garden furniture set", 499.0), ("headphones", 199.0),
                  ("vacuum cleaner", 279.0), ("perfume", 75.0), ("office chair", 329.0), ("stand mixer", 399.0),
                  ("electric toothbrush", 120.0)]
PROMO_TYPES = ["was_now", "up_to", "bogof", "lowest_ever", "urgency"]


class PricePromotionClaimReview(Schema):
    name = "price_promotion_claim_review"
    description = ("A price or promotion claim in advertising (was/now, 'up to', free offers, 'lowest ever', "
                   "deadlines): does it pass the standard's test, and how soon must it be fixed?")
    surfaces = {"json": 0.5, "promo_brief": 0.25, "compliance_ticket": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        pt = pick_w(rng, {"was_now": 4, "up_to": 2, "bogof": 2, "lowest_ever": 1.5, "urgency": 1.5})
        prod, base_price = pick(rng, PROMO_PRODUCTS)
        rule_days = pick(rng, [28, 30, 45])
        f = {"regime": reg, "promo_type": pt, "product": prod, "rule_days": rule_days,
             "retailer": pick(rng, BRANDS) + pick(rng, [" Home", " Stores", " Direct", " Outlet", ""]),
             "currency": pick(rng, CURRENCIES), "conditions_disclosed": chance(rng, 0.6),
             "has_conditions": pt in ("bogof", "up_to") or chance(rng, 0.3),
             "evidence_source": pick_w(rng, {"system_report": 4, "merchant_assertion": 5}),
             "manager": person(rng), "reviewer": person(rng), "ticket": ident(rng, "PRC", 5, "-")}
        good = reg == "clear" and chance(rng, 0.5)
        bad = reg == "clear" and not good
        if pt == "was_now":
            days = (rng.randint(rule_days + 10, 90) if good else rng.randint(0, rule_days // 3) if bad else
                    int(near(rng, rule_days, rule_days * 0.4, lo=0, hi=90, integer=True)))
            ref = round(base_price * rng.uniform(1.1, 1.5), 0) - 0.01
            now = round(ref * rng.uniform(0.5, 0.8), 0) - 0.01
            sold_share = pick(rng, [0, 2, 6, 12, 25]) if not good else pick(rng, [12, 25])
            length = rng.randint(7, 60)
            if good:
                length = min(length, days)
            f.update(days_at_ref=days, ref_price=ref, now_price=now, units_share_at_ref_pct=sold_share,
                     promo_length_days=length)
        elif pt == "up_to":
            maxd = pick(rng, [40, 50, 60, 70])
            share = (rng.randint(15, 40) if good else rng.randint(0, 3) if bad else
                     int(near(rng, 10, 6, lo=0, hi=40, integer=True)))
            f.update(max_discount_pct=maxd, share_at_max_pct=share, lines=rng.randint(80, 900))
        elif pt == "bogof":
            raised = 0 if good else (rng.randint(15, 40) if bad else pick(rng, [0, 0, 5, 12, 25]))
            f.update(price_rise_pct=raised, base_price=base_price)
        elif pt == "lowest_ever":
            gap = (None if good else rng.uniform(5, 20) if bad else pick(rng, [None, None, 2.0, 6.0, 15.0]))
            f.update(lower_price_seen_pct=None if gap is None else round(gap, 0),
                     history_months=pick(rng, [6, 12, 24, 36]))
        else:
            ext = 0 if good else (rng.randint(3, 6) if bad else pick(rng, [0, 0, 1, 2, 4]))
            f.update(extensions=ext)
        live = chance(rng, 0.3)
        f["live"] = live
        f["days_to_launch"] = None if live else pick_w(rng, {rng.randint(1, 7): 3, rng.randint(8, 30): 4,
                                                            rng.randint(31, 70): 3})
        f["complaints"] = rng.randint(1, 30) if live and chance(rng, 0.6) else 0
        return f

    def passes(self, f):
        pt = f["promo_type"]
        if pt == "was_now":
            return (f["days_at_ref"] >= f["rule_days"] and f["units_share_at_ref_pct"] >= 5 and
                    f["promo_length_days"] <= f["days_at_ref"])
        if pt == "up_to":
            return f["share_at_max_pct"] >= 10
        if pt == "bogof":
            return f["price_rise_pct"] == 0
        if pt == "lowest_ever":
            return f["lower_price_seen_pct"] is None
        return f["extensions"] == 0

    def integrity(self, f):
        pt = f["promo_type"]
        if pt == "was_now":
            r = f["days_at_ref"] / f["rule_days"]
            lvl = 3 if r >= 1.5 else 2 if r >= 1 else 1 if r >= 0.5 else 0
            if f["units_share_at_ref_pct"] < 5:
                lvl = max(0, lvl - 2)
            return lvl
        if pt == "up_to":
            r = f["share_at_max_pct"] / 10
            return 3 if r >= 2 else 2 if r >= 1 else 1 if r >= 0.5 else 0
        if pt == "bogof":
            p = f["price_rise_pct"]
            return 3 if p == 0 else 2 if p <= 5 else 1 if p <= 15 else 0
        if pt == "lowest_ever":
            g = f["lower_price_seen_pct"]
            return 3 if g is None else 1 if g <= 3 else 0
        e = f["extensions"]
        return 3 if e == 0 else 1 if e == 1 else 0

    def intended(self, f):
        ok, lvl, pt = self.passes(f), self.integrity(f), f["promo_type"]
        if not ok:
            if pt == "was_now":
                if lvl == 0:
                    ruling = "withdraw"
                elif f["days_at_ref"] >= f["rule_days"] and f["units_share_at_ref_pct"] >= 5:
                    ruling = "shorten_or_end"
                else:
                    ruling = "reword_price_claim"
            elif pt == "bogof":
                ruling = "withdraw" if f["price_rise_pct"] > 5 else "reword_price_claim"
            elif pt == "urgency":
                ruling = "shorten_or_end" if f["extensions"] <= 2 else "withdraw"
            else:
                ruling = "reword_price_claim"
        elif f["has_conditions"] and not f["conditions_disclosed"]:
            ruling = "disclose_conditions"
        else:
            ruling = "approve"
        if f["live"]:
            press = 3
        else:
            d = f["days_to_launch"]
            press = 2 if d <= 7 else 1 if d <= 30 else 0
        if f["evidence_source"] == "system_report":
            ev = "no_more_evidence"
        else:
            ev = {"was_now": "till_sales_records", "up_to": "range_discount_breakdown", "bogof": "price_change_log",
                  "lowest_ever": "price_change_log", "urgency": "price_change_log"}[pt]
        return {"promotion_ruling": ruling, "price_claim_integrity": lvl, "fix_pressure": press,
                "meets_price_claim_test": "yes" if ok else "no", "evidence_request": ev}

    def _standard(self, f):
        return {
            "PR-1": f"A 'was' or reference price must have been the genuine selling price for at least "
                    f"{f['rule_days']} of the 90 days before the promotion, with at least 5% of the period's units "
                    f"sold at it; the promotion may not run longer than the reference price did.",
            "PR-2": "'Up to X% off' requires at least 10% of the lines in the promotion to carry the maximum discount.",
            "PR-3": "'Free' and multi-buy offers require the base price to be unchanged for 28 days before the offer.",
            "PR-4": "'Lowest price ever' must be true across the whole price history. A stated deadline must be real: "
                    "a promotion extended past its advertised end may not repeat the deadline claim.",
            "PR-5": "Material conditions (minimum spend, exclusions, membership) must appear with the price claim.",
        }

    def _claim_and_facts(self, f, rng, ds):
        cur, pt = f["currency"], f["promo_type"]
        if pt == "was_now":
            claim = f"WAS {price(f['ref_price'], cur)}, NOW {price(f['now_price'], cur)} -- {f['product']}"
            facts = [f"Sold at {price(f['ref_price'], cur)} on {f['days_at_ref']} of the last 90 days.",
                     f"Units sold at that price: {f['units_share_at_ref_pct']}% of the period's units.",
                     f"Planned promotion length: {f['promo_length_days']} days."]
        elif pt == "up_to":
            claim = f"Up to {f['max_discount_pct']}% off {f['product']}s and more"
            facts = [f"{f['lines']} lines in the promotion; {f['share_at_max_pct']}% of them carry the full "
                     f"{f['max_discount_pct']}% discount."]
        elif pt == "bogof":
            claim = f"Buy one {f['product']}, get one FREE"
            facts = [f"Base price {price(f['base_price'], cur)}" +
                     (" unchanged for the past three months." if f["price_rise_pct"] == 0 else
                      f", raised {f['price_rise_pct']}% two weeks before the offer.")]
        elif pt == "lowest_ever":
            claim = f"Our lowest price EVER on this {f['product']}"
            facts = [f"Price history checked over {f['history_months']} months." +
                     ("" if f["lower_price_seen_pct"] is None else
                      f" It was sold {f['lower_price_seen_pct']:g}% cheaper during an earlier sale.")]
        else:
            claim = f"{f['product'].capitalize()} sale -- ENDS SUNDAY!"
            facts = [("First run of this promotion." if f["extensions"] == 0 else
                      f"The same 'ends Sunday' promotion has already been extended {f['extensions']} time(s).")]
        if f["has_conditions"]:
            facts.append("Offer excludes clearance lines and needs a loyalty-card sign-up; " +
                         ("terms shown next to the price." if f["conditions_disclosed"] else
                          "terms appear only on the website's terms page."))
        facts.append({"system_report": "Figures above come from the pricing system's audit report.",
                      "merchant_assertion": "Figures above are the category manager's own statement; no system "
                                            "report is attached."}[f["evidence_source"]])
        if f["live"]:
            facts.append("The promotion is already live" + (f"; {f['complaints']} complaints received so far."
                                                            if f["complaints"] else "."))
        else:
            facts.append(f"Media booked; launch in {f['days_to_launch']} days.")
        return claim, facts

    def render(self, f, rng, surface):
        ds = _ds(rng)
        std = self._standard(f)
        claim, facts = self._claim_and_facts(f, rng, ds)
        if surface == "json":
            return dumps(prune({
                "promotion": {"ticket": f["ticket"], "advertiser": f["retailer"], "headline": claim,
                              "channels": pick(rng, ["press and online", "TV and social", "in-store and email",
                                                     "online only", "radio and in-store"])},
                "facts": facts, "standard": std,
            }))
        if surface == "promo_brief":
            return "\n".join([
                f"PROMOTION BRIEF -- {f['retailer']} -- {f['product']}", f"Owner: {f['manager']}", "",
                f"Headline: {claim}", "", "Supporting facts:", "\n".join(f"- {noisy(x, rng, 0.08)}" for x in facts),
                "", "Pricing standard:", "\n".join(f"  {k} {v}" for k, v in std.items()),
            ])
        return "\n".join([
            f"[{f['ticket']}] Price claim review -- {f['retailer']}",
            f"Status: open    Assignee: {f['reviewer']}    Reporter: {f['manager']}", "",
            f"Claim: {claim}", "", "Description:", noisy(" ".join(facts), rng, 0.1), "",
            "Linked standard: " + " ".join(f"{k}: {v}" for k, v in std.items()),
        ])

    def questions(self):
        return {
            "promotion_ruling": choice("What should happen to this price claim?", {
                "approve": "Approve it as it stands.",
                "disclose_conditions": "Approve it once the conditions are shown with the price claim.",
                "reword_price_claim": "Reword the price claim so it is true, for example drop 'was' or lower the "
                                      "'up to' figure.",
                "shorten_or_end": "Shorten or end the promotion so the deadline or reference period is honest.",
                "withdraw": "Withdraw the promotion claim altogether.",
            }),
            "price_claim_integrity": score("How genuine is the price or deadline the claim relies on?", [
                "Fabricated: the reference, base price or deadline is not real.",
                "Doubtful: it falls well short of the standard's test.",
                "Adequate: it meets the test, but only just.",
                "Solid: it clears the test comfortably.",
            ]),
            "fix_pressure": score("How soon must any problem with the claim be fixed?", [
                "No rush: launch is more than a month away.",
                "Before the media goes out in the next few weeks.",
                "This week: launch is days away.",
                "Immediately: the claim is already live.",
            ]),
            "meets_price_claim_test": noul(
                "The price claim passes the standard's test for its type.",
                yes="The reference price, discount spread, base price, price history or deadline meets the rule "
                    "that applies.",
                no="The rule that applies to this kind of claim is not met."),
            "evidence_request": choice("What further evidence should the reviewer ask for?", {
                "no_more_evidence": "None: the figures come from the pricing system's own report.",
                "price_change_log": "The system log of price changes and promotion dates.",
                "range_discount_breakdown": "A line-by-line breakdown of discounts across the range.",
                "till_sales_records": "Till records of units sold at the reference price.",
            }),
        }


WORKFLOW = Workflow(
    name="advertising_claim_substantiation",
    title="Advertising claim substantiation",
    description="A claims-clearance team decides whether advertising claims (efficacy figures, comparisons, green "
                "claims, endorsements, price promotions) are substantiated, need qualifying, or must be pulled, "
                "against a written substantiation standard.",
    domain="advertising / marketing compliance",
    schemas=[EfficacyClaimReview(), ComparativeClaimReview(), EnvironmentalClaimReview(),
             TestimonialEndorsementReview(), PricePromotionClaimReview()],
    notes="Fictional brands, products, competitors and tests. Clearance decisions on advertising copy; no "
          "customer-service replies, no vendor bills, no procurement.",
)
