"""Insurance claims triage: property and casualty claims at first notice of loss and shortly after.

The insurer's own adjudication calls on six kinds of file:
- a personal-auto collision at first notice of loss;
- a homeowners water loss;
- a theft or burglary;
- a reserve review on an open claim (small commercial property fire, homeowners fire, auto bodily injury,
  premises liability);
- a subrogation / recovery referral;
- a travel claim (lost or delayed baggage, trip cancellation, trip interruption).

Decisions cover routing (fast track, desk, field, special investigations), coverage, fraud-indicator strength,
reserve adequacy, recovery potential and documentation sufficiency. No customer-service replies, no
payment-card alerting, no "is this alert real" questions.

Structure follows it_change_approval.py: each schema draws a `regime` first ('clear' / 'borderline' /
'conflicting') and lets it steer the factors. Every amount that depends on an overridden factor is derived
AFTER the regime block, so nothing stale is rendered.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from .base import (
    Schema,
    Workflow,
    chance,
    choice,
    dumps,
    dumps_pretty,
    fmt_date,
    fmt_ts,
    join_sentences,
    kv_lines,
    lognormal_amount,
    near,
    noisy,
    noul,
    org,
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

LOSS_WINDOW = (date(2025, 2, 3), date(2026, 9, 12))


# =========================================================================== private helpers
def _day(rng, start=LOSS_WINDOW[0], end=LOSS_WINDOW[1]) -> date:
    return rand_date(rng, start, end)


def _claim_no(rng, prefix: str, d: date) -> str:
    return f"{prefix}-{d.year % 100:02d}-{rng.randrange(100000, 1000000)}"


def _policy_no(rng, prefix: str) -> str:
    return f"{prefix} {rng.randrange(100, 1000)}-{rng.randrange(1000, 10000)}-{rng.randrange(10)}"


def _h(v):
    """snake_case factor value -> words (None stays None, so prune / show can treat it as missing)."""
    return None if v is None else str(v).replace("_", " ")


def _yn(v):
    return None if v is None else ("yes" if v else "no")


def _num(x):
    """A JSON-friendly amount: whole numbers as int (no stray '.0'), others rounded to cents."""
    return int(x) if float(x).is_integer() else round(x, 2)


def _iso(d):
    return None if d is None else d.isoformat()


SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "CAD": "CA$"}


def _money(x, cur: str = "USD", style: str = "sym") -> str:
    """Like base.money, but whole amounts never show cents and non-whole amounts always do, at any size."""
    if x is None:
        return "unknown"
    body = f"{abs(x):,.0f}" if float(x).is_integer() else f"{abs(x):,.2f}"
    neg = "-" if x < 0 else ""
    return f"{neg}{cur} {body}" if style == "code" or cur not in SYMBOLS else f"{neg}{SYMBOLS[cur]}{body}"


def _mfmt(rng, cur: str = "USD"):
    """One money style per rendered document (a real form does not switch styles mid-page)."""
    style = pick(rng, ["sym", "sym", "code"])
    return lambda x: _money(x, cur=cur, style=style)


def _dfmt(rng):
    style = pick(rng, ["iso", "iso", "us", "long", "eu"])
    return lambda d: fmt_date(d, style=style) if d is not None else None


def _vendor(rng, kinds) -> str:
    return f"{org(rng).split()[0]} {pick(rng, kinds)}"


def _handler(rng) -> str:
    return person(rng)


def _at(d, rng, day_offset: int = 0) -> datetime:
    """A business-hours timestamp `day_offset` days after date or datetime `d`."""
    d = d.date() if isinstance(d, datetime) else d
    d = d + timedelta(days=day_offset)
    return datetime(d.year, d.month, d.day, rng.randint(8, 17), rng.randrange(60))


def _diary(entries, rng) -> str:
    """Claim-system diary lines: timestamp, a short category tag, free text."""
    style = pick(rng, ["log", "log", "iso"])
    w = max(len(tag) for _, tag, _ in entries)
    return "\n".join(f"{fmt_ts(t, style=style)}  {tag.ljust(w)}  {text}" for t, tag, text in entries if text)


# =========================================================================== schema 1: auto collision FNOL
VEHICLES = [
    ("2012 compact sedan", 5400), ("2015 mid-size sedan", 9600), ("2016 minivan", 11800),
    ("2014 full-size SUV", 12900), ("2017 compact SUV", 14700), ("2018 electric hatchback", 16800),
    ("2019 mid-size sedan", 18200), ("2021 small crossover", 22600), ("2023 compact hatchback", 21400),
    ("2020 full-size pickup", 32500),
]
ROADS = ["State Route 41", "Harbor Blvd", "County Road 14", "Ashgrove Ave", "Millbrook Rd", "Lakeshore Dr",
         "Fernhill Rd", "Upland Pkwy", "Quarry Rd", "Elmstead St", "Old Mill Ln", "Route 7"]
# loss type -> (what happened, is another vehicle involved and traceable: "other" / "none" / "unknown")
AUTO_LOSSES = {
    "rear_ended_by_other": ("was stopped at a light on {road} when another car hit the rear", "other"),
    "rear_ended_other": ("ran into the back of the car ahead on {road} when traffic stopped suddenly", "other"),
    "left_turn_intersection": ("was turning left off {road} and collided with an oncoming car", "other"),
    "lane_change_sideswipe": ("was sideswiped by a car changing lanes on {road}", "other"),
    "parked_unattended": ("came back to the car parked on {road} and found the side pushed in, no note left", "none"),
    "single_vehicle_object": ("lost control on a bend on {road} and hit a utility pole", "none"),
    "animal_strike": ("hit a deer that ran out on {road}", "none"),
    "hit_and_run": ("was hit on {road} by a car that drove off before the plate could be read", "unknown"),
}
DAMAGE_AREA = {
    "rear_ended_by_other": "rear bumper, trunk lid and tail lamps",
    "rear_ended_other": "front bumper, hood, grille and radiator support",
    "left_turn_intersection": "front corner, fender and wheel",
    "lane_change_sideswipe": "doors and rear quarter panel on one side",
    "parked_unattended": "driver-side doors and mirror",
    "single_vehicle_object": "front end, hood and one headlamp",
    "animal_strike": "front bumper, headlamp and hood",
    "hit_and_run": "rear quarter panel and bumper corner",
}
FRONTAL = ("rear_ended_other", "left_turn_intersection", "single_vehicle_object", "animal_strike")
FAULT_LINES = {
    ("rear_ended_by_other", "other"): ["Other driver said at the scene they had looked down at their phone.",
                                       "Other driver apologised and handed over insurance details."],
    ("rear_ended_other", "self"): ["Admits following too closely.", "Says they saw the brake lights too late."],
    ("rear_ended_other", "other"): ["Says the car ahead braked hard for no reason.",
                                    "Says the car ahead cut in and stopped dead."],
    ("left_turn_intersection", "other"): ["Says the oncoming car came through on a red.",
                                          "Says the other car was speeding with no headlights on."],
    ("left_turn_intersection", "self"): ["Says they misjudged the gap."],
    ("left_turn_intersection", "disputed"): ["Not sure who had right of way; the light was changing."],
    ("lane_change_sideswipe", "other"): ["Says the other car drifted into their lane without signalling."],
    ("lane_change_sideswipe", "disputed"): ["The other driver says it was our driver who moved over."],
    ("parked_unattended", "other"): ["Did not see it happen."],
    ("single_vehicle_object", "self"): ["Says the road was wet and the car slid.",
                                        "Says a car coming the other way forced them wide on the bend."],
    ("animal_strike", "none"): ["The deer ran off into the trees."],
    ("hit_and_run", "other"): ["Could only say the other car was dark-coloured.", "Got two characters of the plate."],
}
DRIVER_PHRASE = {
    "listed_driver": "The insured's spouse, a listed driver,",
    "unlisted_household_member": "The insured's adult son, who lives at the address but is not listed on the policy,",
    "permissive_friend": "A friend who had borrowed the car",
    "excluded_driver": "The insured's partner",
}
HOURS = list(range(7, 21)) * 3 + [21, 22, 23, 0, 1, 2, 3, 4, 5, 6]


class AutoCollisionFnol(Schema):
    name = "auto_collision_fnol"
    description = ("A personal-auto collision at first notice of loss: route it, allocate fault, test own-damage "
                   "coverage and gauge fraud and total-loss odds.")
    surfaces = {"json": 0.4, "fnol_form": 0.25, "claim_notes": 0.2, "system_export": 0.15}

    def sample_factors(self, rng):
        reg = regime(rng)
        loss = pick_w(rng, {"rear_ended_by_other": 3, "rear_ended_other": 2, "left_turn_intersection": 3,
                            "lane_change_sideswipe": 2, "parked_unattended": 2, "single_vehicle_object": 2,
                            "animal_strike": 1, "hit_and_run": 2})
        vehicle, base_acv = pick(rng, VEHICLES)
        acv = int(round(base_acv * rng.uniform(0.85, 1.15), -2))
        tl_pct = pick(rng, [70, 75, 75, 80])
        band = pick_w(rng, {"small": 5, "medium": 3, "large": 2})
        ratio = rng.uniform(*{"small": (0.04, 0.25), "medium": (0.25, 0.55), "large": (0.55, 1.05)}[band])
        policy_age = pick(rng, [12, 35, 70, 150, 400, 900, 1700, 3300]) + rng.randint(0, 25)
        lag = pick(rng, [0, 0, 0, 1, 1, 2, 3, 5, 9])
        hour = pick(rng, HOURS)
        prior = pick(rng, [0, 0, 0, 1, 1, 2, 3])
        consistency = pick_w(rng, {"consistent": 6, "minor_questions": 2, "inconsistent": 1, None: 2})
        police = pick_w(rng, {"report_taken": 4, "info_exchange_only": 3, "none_called": 2, "pending": 1})
        cover = pick_w(rng, {"collision_and_comprehensive": 8, "comprehensive_only": 2, "liability_only": 2})
        status = pick_w(rng, {"active": 10, "premium_past_due": 2, "reinstated_after_lapse": 2})
        driver = pick_w(rng, {"named_insured": 8, "listed_driver": 3, "unlisted_household_member": 1,
                              "permissive_friend": 1, "excluded_driver": 1})
        use = pick_w(rng, {"personal": 10, "rideshare_app_on": 2, "food_delivery": 2})
        injuries = pick_w(rng, {"none": 6, "soreness_reported": 2, "er_visit": 1, None: 1})
        witness = pick_w(rng, {"independent": 2, "passenger_only": 2, "none": 4})
        reinstate_terms = pick_w(rng, {"no_lapse_in_coverage": 1, "lapse_recorded": 1, None: 1})
        grace_days = pick(rng, [10, 15, 20, 30])
        past_due = rng.randint(2, grace_days + 10)
        paid_after_loss = pick_w(rng, {True: 2, False: 1, None: 1})
        fault_view, citation, twist = None, None, None

        if reg == "clear":
            if chance(rng, 0.5):                      # routine, covered and well supported
                twist = "routine"
                loss = pick(rng, ["rear_ended_by_other", "rear_ended_by_other", "animal_strike"])
                ratio = rng.uniform(0.04, 0.2)
                policy_age, lag, prior, hour = rng.randint(700, 4000), pick(rng, [0, 0, 1]), 0, rng.randint(8, 18)
                consistency, cover, status, use, injuries = "consistent", "collision_and_comprehensive", "active", "personal", "none"
                police = "report_taken" if loss != "animal_strike" else "none_called"
                driver = pick(rng, ["named_insured", "listed_driver"])
                witness = "independent" if loss != "animal_strike" else "passenger_only"
            elif chance(rng, 0.6):                    # warning signs everywhere
                twist = "red_flags"
                loss = pick(rng, ["single_vehicle_object", "hit_and_run", "parked_unattended"])
                ratio = rng.uniform(0.6, 1.1)
                policy_age, lag, prior, hour = rng.randint(5, 28), rng.randint(6, 16), pick(rng, [2, 3]), pick(rng, [0, 1, 2, 3, 4])
                consistency, police, witness = "inconsistent", "none_called", "none"
                cover, status = "collision_and_comprehensive", "active"
            else:                                     # plainly outside the cover bought
                twist = pick(rng, ["no_cover", "excluded_driver", "lapse"])
                injuries, consistency, status, cover = "none", "consistent", "active", "collision_and_comprehensive"
                if twist == "no_cover":
                    cover = "liability_only"
                    loss = pick(rng, ["rear_ended_other", "single_vehicle_object", "left_turn_intersection"])
                elif twist == "excluded_driver":
                    driver = "excluded_driver"
                else:
                    status, reinstate_terms = "reinstated_after_lapse", "lapse_recorded"
        elif reg == "borderline":
            dims = rng.sample(["total_loss", "new_policy", "late_report", "driver_use", "soft_injury", "past_due"], 2)
            for dim in dims:
                if dim == "total_loss":
                    ratio = near(rng, tl_pct / 100, 0.12, lo=0.35)
                elif dim == "new_policy":
                    policy_age = near(rng, 45, 25, lo=6, integer=True)
                elif dim == "late_report":
                    lag = near(rng, 5, 3, lo=2, integer=True)
                elif dim == "driver_use":
                    if chance(rng, 0.5):
                        driver = pick(rng, ["unlisted_household_member", "permissive_friend"])
                    else:
                        use = pick(rng, ["rideshare_app_on", "food_delivery"])
                elif dim == "soft_injury":
                    injuries, ratio = "soreness_reported", rng.uniform(0.04, 0.15)
                else:
                    status = "premium_past_due"
                    past_due = near(rng, grace_days, 4, lo=1, integer=True)
                    paid_after_loss = pick(rng, [True, None])
            twist = "+".join(sorted(dims))
        else:
            twist = pick(rng, ["clean_insured_bad_damage", "story_vs_citation", "minor_impact_injury",
                               "reinstatement_unclear", "tenured_hit_and_run"])
            blanks = ["consistency", "injuries", "police"]
            if twist == "clean_insured_bad_damage":
                policy_age, prior, police, status = rng.randint(1500, 5000), 0, "report_taken", "active"
                cover, consistency = "collision_and_comprehensive", pick(rng, ["inconsistent", "minor_questions"])
                blanks = ["injuries"]
            elif twist == "story_vs_citation":
                loss = pick(rng, ["left_turn_intersection", "lane_change_sideswipe", "rear_ended_other"])
                fault_view, police, citation = "other", "report_taken", "insured_cited"
                witness = pick(rng, ["none", "passenger_only"])
                blanks = ["consistency", "injuries"]
            elif twist == "minor_impact_injury":
                loss = pick(rng, ["rear_ended_by_other", "rear_ended_other", "lane_change_sideswipe"])
                ratio, injuries, witness = rng.uniform(0.03, 0.12), "er_visit", "none"
                blanks = ["consistency", "police"]
            elif twist == "reinstatement_unclear":
                status, reinstate_terms = "reinstated_after_lapse", None
                consistency, prior, policy_age = "consistent", 0, rng.randint(800, 3000)
                blanks = ["injuries", "police"]
            else:
                loss, witness, police = "hit_and_run", "none", pick(rng, ["pending", "report_taken"])
                policy_age, prior, ratio = rng.randint(1500, 5000), 0, rng.uniform(0.3, 0.7)
                consistency = "minor_questions"
                blanks = ["injuries"]
            if chance(rng, 0.4):                      # and one fact is not on file yet
                blank = pick(rng, blanks)
                if blank == "consistency":
                    consistency = None
                elif blank == "injuries":
                    injuries = None
                else:
                    police = "pending"

        # ---- derived after the regime block
        estimate = _num(round(acv * ratio, 2))
        drivable = chance(rng, 0.92 if ratio < 0.25 else 0.5 if ratio < 0.55 else 0.12)
        airbags = loss in FRONTAL and ratio > 0.4 and chance(rng, 0.6)
        if fault_view is None:
            fault_view = {"rear_ended_by_other": "other", "rear_ended_other": pick(rng, ["self", "self", "other"]),
                          "left_turn_intersection": pick(rng, ["other", "disputed", "self"]),
                          "lane_change_sideswipe": pick(rng, ["other", "disputed"]), "parked_unattended": "other",
                          "single_vehicle_object": "self", "animal_strike": "none", "hit_and_run": "other"}[loss]
        if police != "report_taken":
            citation = None
        elif citation is None:
            if loss == "rear_ended_by_other":
                citation = pick(rng, ["other_cited", "none"])
            elif loss == "rear_ended_other":
                citation = pick(rng, ["insured_cited", "none"])
            elif loss == "left_turn_intersection":
                citation = pick(rng, {"other": ["other_cited", "none", "insured_cited"],
                                      "self": ["insured_cited", "none"]}.get(fault_view, ["none"]))
            elif loss == "lane_change_sideswipe":
                citation = pick(rng, ["none", "none", "other_cited", "insured_cited"])
            else:
                citation = "none"
        other_kind = AUTO_LOSSES[loss][1]
        other_driver = person(rng) if other_kind == "other" else None
        other_carrier = None
        if other_driver:
            other_carrier = pick_w(rng, {"known": 7, "uninsured": 1, "not_yet_known": 2})
            if other_carrier == "known":
                other_carrier = org(rng, "insurer")
        if other_kind != "other" and witness == "independent" and loss in ("single_vehicle_object", "animal_strike"):
            witness = "passenger_only"
        if loss == "parked_unattended":            # nobody was in the car
            injuries = "none" if injuries is not None else None
            witness = "none" if witness == "passenger_only" else witness
        photos = pick(rng, [0, 3, 4, 6, 8, 12])
        if consistency is not None and photos == 0:   # a damage review needs photos
            photos = pick(rng, [3, 5, 8])
        loss_date = _day(rng)
        loss_at = datetime(loss_date.year, loss_date.month, loss_date.day, hour, rng.randrange(60))
        lapse_on = reinstated_on = None
        if status == "reinstated_after_lapse":
            policy_age = max(policy_age, rng.randint(90, 900))
            lapse_on = loss_date - timedelta(days=rng.randint(4, 25))
            reinstated_on = loss_date + timedelta(days=rng.randint(0, 3))
        return {
            "regime": reg, "twist": twist, "loss_type": loss, "road": pick(rng, ROADS), "vehicle": vehicle,
            "acv": acv, "total_loss_threshold_pct": tl_pct, "repair_ratio": round(ratio, 3), "estimate": estimate,
            "drivable": drivable, "airbags_deployed": airbags, "policy_age_days": policy_age,
            "report_lag_days": lag, "loss_at": loss_at, "reported_on": loss_date + timedelta(days=lag),
            "inception": loss_date - timedelta(days=policy_age), "prior_claims_36m": prior,
            "damage_consistency": consistency, "police": police, "citation": citation, "cover": cover,
            "status": status, "lapse_on": lapse_on, "reinstated_on": reinstated_on,
            "reinstatement_terms": reinstate_terms if status == "reinstated_after_lapse" else None,
            "grace_days": grace_days if status == "premium_past_due" else None,
            "days_past_due_at_loss": past_due if status == "premium_past_due" else None,
            "paid_after_loss": paid_after_loss if status == "premium_past_due" else None,
            "driver": driver, "use": use, "injuries": injuries, "witness": witness, "fault_view": fault_view,
            "other_driver": other_driver, "other_carrier": other_carrier,
            "deductible": pick(rng, [250, 500, 500, 1000, 1000, 2000]),
            "photos": photos, "channel": pick(rng, ["phone", "mobile app", "agent", "web"]),
            "claim_no": _claim_no(rng, "AU", loss_date), "policy_no": _policy_no(rng, "PA"), "insured": person(rng),
            "police_ref": f"{loss_date.year % 100}-{rng.randrange(10000, 99999)}",
        }

    # ------------------------------------------------------------------ text pieces
    def _account(self, f, rng):
        who = DRIVER_PHRASE.get(f["driver"]) or pick(rng, ["Insured", "The insured", "The named insured"])
        s = [f"{who} {AUTO_LOSSES[f['loss_type']][0].format(road=f['road'])}."]
        s.append(pick(rng, FAULT_LINES.get((f["loss_type"], f["fault_view"]), [""])))
        if f["use"] == "rideshare_app_on":
            s.append("The ride-hail app was on and the driver was waiting for a fare.")
        elif f["use"] == "food_delivery":
            s.append("Was out delivering food orders at the time.")
        s.append({"none": pick(rng, ["No injuries.", "Nobody hurt."]),
                  "soreness_reported": "Mentions neck soreness the next morning; no treatment yet.",
                  "er_visit": "Driver went to the emergency room that night and was released.",
                  None: ""}[f["injuries"]])
        if f["witness"] == "independent":
            s.append("A pedestrian stopped and gave a phone number.")
        elif f["witness"] == "passenger_only":
            s.append("Only a passenger saw it.")
        return noisy(join_sentences(s), rng, 0.15)

    def _damage_review(self, f, rng):
        area, c = DAMAGE_AREA[f["loss_type"]], f["damage_consistency"]
        if c is None:
            return None
        if c == "consistent":
            return f"Damage to the {area} fits the described impact."
        if c == "minor_questions":
            return f"Damage to the {area} mostly fits, but " + pick(rng, [
                "the rust at the break lines looks older than the loss date.",
                "one long scrape runs the wrong way for this impact.",
                "the mirror damage looks like a separate event."])
        bad = ["there are two separate impact points.", "dirt and rust in the creases suggest the damage is old."]
        if f["loss_type"] in ("rear_ended_by_other", "rear_ended_other", "left_turn_intersection",
                              "lane_change_sideswipe", "hit_and_run"):
            bad += ["the impact height is well above a car bumper.",
                    "paint transfer colour does not match the other vehicle described."]
        elif f["loss_type"] == "animal_strike":
            bad += ["no hair or blood in the grille, and the dent is narrow like a post strike."]
        elif f["loss_type"] == "single_vehicle_object":
            bad += ["the dent is far wider than a pole strike would leave."]
        return "Damage does not match the account: " + pick(rng, bad)

    COVER_TEXT = {"collision_and_comprehensive": "collision and comprehensive (own damage from any collision, plus "
                                                 "theft, fire, glass and animal strikes)",
                  "comprehensive_only": "comprehensive only (theft, fire, glass and animal strikes; no collision cover)",
                  "liability_only": "liability only (no cover for damage to the insured's own vehicle)"}
    EXCLUSIONS = ("own-damage exclusions: any driver named as excluded on the policy; use for hire, ride-hail or "
                  "delivery; a loss while the policy is lapsed (premium unpaid beyond the grace period)")

    def _policy_status(self, f):
        if f["status"] == "premium_past_due":
            return (f"premium {f['days_past_due_at_loss']} days past due at date of loss "
                    f"(grace period {f['grace_days']} days)")
        if f["status"] == "reinstated_after_lapse":
            return "reinstated after lapse for non-payment"
        return "active"

    def render(self, f, rng, surface):
        account = self._account(f, rng)
        review = self._damage_review(f, rng)
        ld = f["loss_at"].date()
        coll = f["cover"] == "collision_and_comprehensive"
        if surface in ("json", "system_export"):
            policy = {"policy_number": f["policy_no"], "status": self._policy_status(f), "inception_date": _iso(f["inception"]),
                      "physical_damage_cover": self.COVER_TEXT[f["cover"]], "exclusions": self.EXCLUSIONS,
                      "deductible": f["deductible"] if f["cover"] != "liability_only" else None,
                      "driver_status": f["driver"], "use_at_time_of_loss": f["use"],
                      "lapsed_on": _iso(f["lapse_on"]), "reinstated_on": _iso(f["reinstated_on"]),
                      "reinstatement_terms": f["reinstatement_terms"],
                      "payment_received_after_loss": f["paid_after_loss"]}
            vehicle = {"description": f["vehicle"], "estimated_acv": f["acv"],
                       "photo_estimate": f["estimate"] if f["photos"] else None,
                       "total_loss_threshold_pct": f["total_loss_threshold_pct"], "drivable": f["drivable"],
                       "airbags_deployed": f["airbags_deployed"], "photos_received": f["photos"]}
            parties = {"other_driver": f["other_driver"],
                       "other_carrier": f["other_carrier"] if f["other_driver"] else None,
                       "police": f["police"], "police_report_no": f["police_ref"] if f["police"] == "report_taken" else None,
                       "citation": f["citation"], "witness": f["witness"]}
            if surface == "json":
                return dumps(prune({
                    "claim": {"claim_number": f["claim_no"], "line": "personal auto", "reported_via": f["channel"],
                              "date_of_loss": _iso(ld), "time_of_loss": f["loss_at"].strftime("%H:%M"),
                              "date_reported": _iso(f["reported_on"]), "loss_description": account},
                    "policy": policy, "vehicle": vehicle, "parties": parties,
                    "review": {"injuries_reported": f["injuries"], "damage_review": review,
                               "prior_claims_36m": f["prior_claims_36m"]},
                }))
            return dumps_pretty(prune({
                "claimHeader": {"claimId": f["claim_no"], "lossDateTime": f["loss_at"].strftime("%Y-%m-%dT%H:%M"),
                                "reportedDate": _iso(f["reported_on"]), "channel": f["channel"],
                                "insuredName": f["insured"], "narrative": account},
                "policySnapshot": policy, "vehicleAppraisal": dict(vehicle, reviewerComment=review),
                "liabilityInfo": parties,
                "flags": {"injuryIndicator": f["injuries"], "priorClaims36m": f["prior_claims_36m"]},
            }), rng)
        m, d = _mfmt(rng), _dfmt(rng)
        if surface == "fnol_form":
            head = kv_lines([
                ("Claim no", f["claim_no"]), ("Policy", f["policy_no"]), ("Insured", f["insured"]),
                ("Policy status", self._policy_status(f)), ("Policy inception", d(f["inception"])),
                ("Lapse / reinstated", f"{d(f['lapse_on'])} / {d(f['reinstated_on'])}" if f["lapse_on"] else None),
                ("Reinstatement terms", show(f["reinstatement_terms"], rng, _h) if f["lapse_on"] else None),
                ("Coverage", self.COVER_TEXT[f["cover"]]), ("Exclusions", self.EXCLUSIONS),
                ("Deductible", m(f["deductible"]) if f["cover"] != "liability_only" else "n/a"),
                ("Date / time of loss", f"{d(ld)} {f['loss_at'].strftime('%H:%M')}"),
                ("Reported", f"{d(f['reported_on'])} via {f['channel']}"),
                ("Driver", _h(f["driver"])), ("Use at time of loss", _h(f["use"])),
                ("Vehicle", f"{f['vehicle']}, ACV approx {m(f['acv'])}"),
                ("Drivable", _yn(f["drivable"])), ("Airbags deployed", _yn(f["airbags_deployed"])),
                ("Other driver", f["other_driver"]),
                ("Other carrier", show(f["other_carrier"], rng, _h) if f["other_driver"] else None),
                ("Police", _h(f["police"]) + (f" #{f['police_ref']}" if f["police"] == "report_taken" else "")),
                ("Citation", _h(f["citation"])), ("Witness", _h(f["witness"])),
                ("Injuries", show(f["injuries"], rng, _h)), ("Prior claims (36 mo)", f["prior_claims_36m"]),
            ], rng)
            appraisal = ("Photo estimate: " + (m(f["estimate"]) + f"; total-loss threshold {f['total_loss_threshold_pct']}% of ACV"
                                               if f["photos"] else "no photos received yet"))
            rep = pick(rng, ["Caller had photos and the other party's details ready.", "Caller was calm and clear.",
                             "Caller asked twice whether this would raise the premium.",
                             "Caller wants the car repaired at a shop near work.", ""])
            if f["report_lag_days"] >= 4:
                rep = join_sentences([rep, pick(rng, ["Says they were travelling and could not call sooner.",
                                                      "Did not say why the report came in late."])])
            return "\n".join(x for x in [
                "FIRST NOTICE OF LOSS -- PERSONAL AUTO", head, "", "Caller narrative:", account, "",
                appraisal + (f". Reviewer: {review}" if review else "."),
                ("Rep notes: " + rep) if rep else None] if x is not None)
        # claim_notes: the diary a desk adjuster would open
        t0 = _at(f["reported_on"], rng)
        if t0 <= f["loss_at"] + timedelta(minutes=30):
            t0 = f["loss_at"] + timedelta(minutes=rng.randint(40, 300))
        cov = [f"Policy {f['policy_no']} {self._policy_status(f)}; inception {d(f['inception'])}."]
        if f["lapse_on"]:
            cov.append(f"Lapsed {d(f['lapse_on'])}, reinstated {d(f['reinstated_on'])}; terms "
                       f"{show(f['reinstatement_terms'], rng, _h)}.")
        if f["status"] == "premium_past_due":
            cov.append("Payment received after the loss: " + show(f["paid_after_loss"], rng, _yn) + ".")
        cov.append(f"Cover: {self.COVER_TEXT[f['cover']]}" + (f", ded {m(f['deductible'])}." if f["cover"] != "liability_only" else "."))
        cov.append(self.EXCLUSIONS[0].upper() + self.EXCLUSIONS[1:] + ".")
        cov.append(f"Driver: {_h(f['driver'])}; use: {_h(f['use'])}.")
        liab = []
        if f["other_driver"]:
            liab.append(f"Other driver {f['other_driver']}, carrier {show(f['other_carrier'], rng, _h)}.")
        liab.append({"report_taken": f"Police report {f['police_ref']} requested.", "info_exchange_only": "Drivers exchanged details; no police attendance.",
                     "none_called": "Police not called.", "pending": "Police report number not yet provided."}[f["police"]])
        if f["citation"] and f["citation"] != "none":
            liab.append(f"Report shows {_h(f['citation'])}.")
        liab.append(f"Witness: {_h(f['witness'])}.")
        entries = [
            (t0, "INTAKE", f"FNOL by {f['channel']}, loss {d(ld)} {f['loss_at'].strftime('%H:%M')}. {account}"),
            (t0 + timedelta(minutes=rng.randint(10, 50)), "COVERAGE", " ".join(cov)),
            (max(t0 + timedelta(minutes=55), _at(t0, rng)), "APPRAISAL",
             (f"Photo estimate {m(f['estimate'])} on {f['vehicle']} (ACV about {m(f['acv'])}, TL threshold "
              f"{f['total_loss_threshold_pct']}%). {f['photos']} photos. Drivable: {_yn(f['drivable'])}; "
              f"airbags: {_yn(f['airbags_deployed'])}. " + (review or "Damage review not done yet."))
             if f["photos"] else f"No photos yet; {f['vehicle']}, ACV about {m(f['acv'])}. Drivable: {_yn(f['drivable'])}."),
            (_at(t0, rng, 1), "LIABILITY", " ".join(liab)),
            (_at(t0, rng, 2), "HISTORY", f"{f['prior_claims_36m']} prior claim(s) in 36 months."),
        ]
        return f"Claim {f['claim_no']} -- {f['insured']}\n" + _diary(entries, rng)

    def questions(self):
        return {
            "claim_routing": choice("Where should this auto claim be routed after first notice?", {
                "fast_track_payment": "Settle straight from the photo estimate with no adjuster assigned; the loss is "
                                      "small, covered and unremarkable.",
                "desk_adjuster": "Assign a desk adjuster to confirm the facts and coverage by phone and settle from "
                                 "documents.",
                "field_appraisal": "Send an appraiser or field adjuster to inspect the vehicle or the scene in person.",
                "coverage_review_first": "Hold all handling until an open coverage question, such as a lapse, an "
                                         "unlisted driver or business use, is resolved.",
                "siu_referral": "Refer the claim to the special investigations unit for a formal fraud review.",
            }),
            "misrepresentation_signals": score("How strong are the signs that this loss was staged, inflated or misreported?", [
                "None: the facts hang together and nothing is out of pattern.",
                "Faint: one soft indicator that probably has an innocent explanation.",
                "Notable: several indicators that need answers before any payment.",
                "Strong: a pattern that justifies a formal investigation.",
            ]),
            "total_loss_likelihood": score("How likely is the vehicle to end up declared a total loss?", [
                "Remote: the repair cost is a small share of the vehicle's value.",
                "Possible: only if the teardown finds a lot of hidden damage.",
                "Likely: the estimate sits close to the threshold and supplements usually push it over.",
                "Near certain: the repair cost already meets or passes the total-loss threshold.",
            ]),
            "fault_allocation": choice("How is fault for this accident most likely to be allocated?", {
                "other_party_at_fault": "The other driver will probably be held responsible for all or nearly all of it.",
                "insured_at_fault": "The insured's driver will probably be held responsible for all or nearly all of it.",
                "shared_fault": "Both drivers will probably share responsibility in some proportion.",
                "no_recoverable_party": "No identifiable other party can be held responsible, so the loss stays on "
                                        "the insured's own cover.",
                "undetermined": "The file is too thin or too contradictory to allocate fault yet.",
            }),
            "own_damage_cover_applies": noul(
                "The policy's own-damage coverage responds to this loss.",
                yes="The policy was in force on the date of loss, the needed coverage (collision or comprehensive) "
                    "was bought, and neither the driver nor the use is excluded.",
                no="Coverage had lapsed, the needed coverage was not bought, or the driver or the use at the time "
                   "falls under an exclusion."),
        }


# =========================================================================== schema 2: home water damage
WATER_SOURCES = {
    # key: (policyholder's description, nature)
    "supply_line_burst": ("the braided supply line to the {fixture} burst", "sudden"),
    "washer_hose_failure": ("the washing machine inlet hose split", "sudden"),
    "water_heater_rupture": ("the water heater tank split at the base", "sudden"),
    "slow_leak_under_sink": ("a drain joint under the {fixture} had been leaking", "gradual"),
    "shower_pan_seepage": ("water had been getting through the shower pan into the floor below", "gradual"),
    "roof_leak_storm": ("rain came in through the roof", "storm"),
    "sewer_backup": ("the sewer line backed up through the basement floor drain", "sewer"),
    "surface_water_flood": ("heavy rain flooded the yard and water came in under the basement door", "flood"),
}
FIXTURES = ["kitchen sink", "upstairs toilet", "bathroom vanity", "refrigerator", "utility sink"]
ROOMS = {
    "indoor": ["kitchen", "hallway", "finished basement", "downstairs bathroom", "living room", "laundry room",
               "bedroom below", "dining room"],
    "storm": ["upstairs bedroom", "upstairs hallway", "attic", "living room", "home office", "stairwell"],
    "below_grade": ["finished basement", "basement bathroom", "basement laundry", "basement guest room", "storage room"],
}
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}
PLUMBER_TEXT = {
    "sudden_failure_confirmed": ["Failure point is clean and looks recent; no sign of earlier leaking around it.",
                                 "Break is fresh; the framing around it is dry and sound."],
    "long_term_leak_indicated": ["Corrosion and rot around the joint point to leaking over weeks or longer.",
                                 "Subfloor delaminated and stained well beyond the wet area; leak was not new."],
    "mixed_findings": ["The failure itself looks recent, but staining under it suggests an earlier slow drip as well.",
                       "Break looks fresh; some of the subfloor rot looks older than a couple of weeks."],
    "pending": ["Plumber booked; no report yet."],
}
ROOF_TEXT = {"wind_opening": "Adjuster photos show lifted and missing shingles along the ridge.",
             "worn_roof": "Roof is about 25 years old; granule loss and cracked flashing, no fresh wind marks.",
             "both": "Several shingles freshly lifted, but the flashing around the chimney is badly worn as well."}


class HomeWaterDamage(Schema):
    name = "home_water_damage"
    description = ("A homeowners water-damage claim shortly after first notice: coverage position, cause "
                   "documentation, cost growth and whether the policyholder mitigated.")
    surfaces = {"json": 0.4, "claimant_email": 0.2, "adjuster_diary": 0.2, "mitigation_report": 0.2}

    def sample_factors(self, rng):
        reg = regime(rng)
        source = pick_w(rng, {"supply_line_burst": 3, "washer_hose_failure": 2, "water_heater_rupture": 2,
                              "slow_leak_under_sink": 2, "shower_pan_seepage": 1, "roof_leak_storm": 2,
                              "sewer_backup": 2, "surface_water_flood": 1})
        discovery = pick_w(rng, {"as_it_happened": 4, "returned_after_days_away": 2, "noticed_stains_earlier": 2})
        days_away = rng.randint(2, 21)
        plumber = pick_w(rng, {"sudden_failure_confirmed": 3, "long_term_leak_indicated": 2, "mixed_findings": 1,
                               "pending": 2, None: 1})
        roof = pick(rng, ["wind_opening", "worn_roof", "both"])
        mit_hours = pick(rng, [2, 4, 6, 10, 18, 30, 52, 80, 120])
        mold = pick_w(rng, {"none": 5, "localized": 3, "widespread": 1})
        rooms = pick(rng, [1, 1, 2, 2, 3, 4, 5])
        occupancy = pick_w(rng, {"owner_occupied": 8, "tenant_occupied": 2, "vacant": 1})
        vacant_days = rng.randint(20, 120)
        sewer_limit = pick_w(rng, {None: 3, 5000: 2, 10000: 2, 25000: 1})
        moisture_map = chance(rng, 0.6)
        photos = pick(rng, [0, 6, 14, 22, 40])
        rebuild_missing = chance(rng, 0.3)
        agent_said_covered = False
        twist = None

        if reg == "clear":
            if chance(rng, 0.55):
                twist = "clean_sudden"
                source = pick(rng, ["supply_line_burst", "washer_hose_failure", "water_heater_rupture"])
                discovery, plumber, mit_hours, mold, rooms = "as_it_happened", "sudden_failure_confirmed", pick(rng, [2, 3, 5]), "none", pick(rng, [1, 2])
                occupancy, moisture_map, photos, rebuild_missing = "owner_occupied", True, pick(rng, [14, 22, 40]), False
            else:
                twist = "clearly_excluded"
                source = pick(rng, ["surface_water_flood", "slow_leak_under_sink", "shower_pan_seepage"])
                discovery = "noticed_stains_earlier"
                plumber = "long_term_leak_indicated" if source != "surface_water_flood" else None
                mit_hours, mold = pick(rng, [80, 120, 160]), pick(rng, ["localized", "widespread"])
        elif reg == "borderline":
            twist = pick(rng, ["away_near_14_days", "sewer_near_limit", "vacancy_near_60", "slow_drying"])
            if twist == "away_near_14_days":
                source = pick(rng, ["supply_line_burst", "slow_leak_under_sink", "washer_hose_failure"])
                discovery, days_away = "returned_after_days_away", near(rng, 14, 5, lo=6, integer=True)
                plumber = pick(rng, ["pending", "mixed_findings", None])
            elif twist == "sewer_near_limit":
                source, sewer_limit = "sewer_backup", pick(rng, [5000, 10000, 25000])
            elif twist == "vacancy_near_60":
                occupancy, vacant_days = "vacant", near(rng, 60, 14, lo=30, integer=True)
                source = pick(rng, ["supply_line_burst", "water_heater_rupture"])
                discovery = "returned_after_days_away"
                days_away = min(vacant_days, rng.randint(5, 30))
            else:
                mit_hours, mold = near(rng, 48, 20, lo=12, integer=True), pick(rng, ["none", "localized"])
                rooms = pick(rng, [2, 3, 4])
        else:
            twist = pick(rng, ["says_sudden_plumber_disagrees", "fresh_break_old_rot", "roof_wind_and_wear",
                               "sewer_agent_promise", "prompt_but_mold"])
            if twist == "says_sudden_plumber_disagrees":
                source, discovery, plumber = pick(rng, ["supply_line_burst", "washer_hose_failure"]), "as_it_happened", "long_term_leak_indicated"
            elif twist == "fresh_break_old_rot":
                source, plumber = pick(rng, ["slow_leak_under_sink", "supply_line_burst"]), "mixed_findings"
                discovery = pick(rng, ["noticed_stains_earlier", "returned_after_days_away"])
            elif twist == "roof_wind_and_wear":
                source, roof = "roof_leak_storm", "both"
            elif twist == "sewer_agent_promise":
                source, sewer_limit, agent_said_covered = "sewer_backup", None, True
            else:
                source, discovery, mit_hours, mold = pick(rng, ["supply_line_burst", "water_heater_rupture"]), "as_it_happened", pick(rng, [2, 4]), "widespread"
            if chance(rng, 0.4):
                rebuild_missing = True

        # ---- derived after the regime block
        nature = WATER_SOURCES[source][1]
        if nature in ("sewer", "flood", "storm"):
            plumber = None if source != "sewer_backup" else pick_w(rng, {None: 2, "pending": 1})
        if nature in ("sewer", "flood", "storm") and discovery == "noticed_stains_earlier" and nature != "storm":
            discovery = "as_it_happened"             # a backup or a flood is not a slow stain
        pool = ROOMS["storm" if nature == "storm" else "below_grade" if nature in ("sewer", "flood") else "indoor"]
        rooms = min(rooms, len(pool))
        room_names = rng.sample(pool, rooms)
        if occupancy != "vacant":
            vacant_days = None
        if discovery != "returned_after_days_away":
            days_away = None
        mitigation = _num(round(lognormal_amount(rng, 1600 + 1400 * rooms, 0.35) * (1.25 if mold != "none" else 1.0), 2))
        rebuild = int(round(lognormal_amount(rng, 5200 * rooms, 0.4) * (1.3 if mold == "widespread" else 1.0), -1))
        if twist == "sewer_near_limit":        # total lands near the endorsement limit
            share = near(rng, 1.0, 0.25, lo=0.55)
            total = sewer_limit * share
            mitigation = _num(round(total * rng.uniform(0.3, 0.45), 2))
            rebuild = int(round(total - mitigation, -1))
        contents = pick_w(rng, {None: 3, "some": 2})
        contents = int(round(lognormal_amount(rng, 2500, 0.6), -1)) if contents else None
        loss_date = _day(rng)
        return {
            "regime": reg, "twist": twist, "source": source, "nature": nature, "fixture": pick(rng, FIXTURES),
            "rooms_affected": rooms, "room_names": room_names, "discovery": discovery,
            "days_away": days_away, "plumber_finding": plumber, "roof_condition": roof if source == "roof_leak_storm" else None,
            "mitigation_hours": mit_hours, "mold": mold, "occupancy": occupancy, "vacant_days": vacant_days,
            "sewer_endorsement_limit": sewer_limit, "agent_said_covered": agent_said_covered,
            "moisture_map": moisture_map, "photos": photos, "mitigation_invoice": mitigation,
            "rebuild_estimate": None if rebuild_missing else rebuild, "contents_claimed": contents,
            "deductible": pick(rng, [1000, 1000, 2500, 5000]), "mold_sublimit": pick(rng, [5000, 10000]),
            "prior_water_claims_5y": pick(rng, [0, 0, 0, 1, 2]), "year_built": pick(rng, [1958, 1974, 1988, 1996, 2004, 2015]),
            "loss_date": loss_date, "reported_on": loss_date + timedelta(days=pick(rng, [0, 1, 1, 2, 4, 9])),
            "claim_no": _claim_no(rng, "HO", loss_date), "policy_no": _policy_no(rng, "HO"),
            "insured": person(rng), "adjuster": _handler(rng),
            "vendor": _vendor(rng, ["Restoration", "Dry-Out Services", "Water Recovery"]),
        }

    def _account(self, f, rng):
        what = WATER_SOURCES[f["source"]][0].format(fixture=f["fixture"])
        names = f["room_names"]
        where = (f"the {names[0]}" if len(names) == 1 else f"the {names[0]} and the {names[1]}" if len(names) == 2
                 else f"the {names[0]}, the {names[1]} and {NUMBER_WORDS.get(len(names) - 2, 'several')} other room(s)")
        nature, disc, occ = f["nature"], f["discovery"], f["occupancy"]
        we = {"owner_occupied": "we", "tenant_occupied": "the tenants", "vacant": "I"}[occ]
        i_ = {"owner_occupied": "I", "tenant_occupied": "the tenant", "vacant": "I"}[occ]
        cap = lambda x: x[0].upper() + x[1:]  # noqa: E731
        if occ == "vacant":
            s = [f"The house has been empty since the tenants moved out about {f['vacant_days']} days ago.",
                 "When I went by to check on it"
                 + (f" (my last visit was {f['days_away']} days earlier)" if f["days_away"] else "")
                 + f" I found water through {where}.", f"It turned out {what}."]
        elif disc == "returned_after_days_away":
            s = [f"{cap(we)} got back after {f['days_away']} days away and found water through {where}.",
                 f"It turned out {what}."]
        elif nature == "sewer":
            s = [f"During heavy rain {what}.", f"The water got into {where}."]
        elif nature == "flood":
            s = [cap(what) + ".", f"The water got into {where}."]
        elif nature == "storm":
            if disc == "noticed_stains_earlier":
                s = [f"{cap(we)} saw a small stain on the ceiling after a storm in the spring and did not think much of it.",
                     f"In the storm last week {what} and into {where}."]
            else:
                s = [f"During the storm last week {what} and into {where}."]
        elif disc == "noticed_stains_earlier":
            s = [f"{cap(we)} had noticed a stain on the ceiling a few weeks back and thought it was old.",
                 f"Last weekend it started dripping and {we} found that {what}.", f"The water had reached {where}."]
        elif nature == "gradual":
            s = [f"{cap(we)} noticed the floor was soft and damp and found that {what}.", f"It had got into {where}."]
        else:
            s = [f"{cap(i_)} heard water running, found that {what}, and shut off the main valve straight away.",
                 f"The water reached {where}."]
        if occ == "tenant_occupied":
            s.insert(0, "The property is rented out and the tenant called me.")
        if f["mitigation_hours"] >= 36:
            s.append(pick(rng, ["The drying company could not come until later.", "We tried to dry it ourselves first.",
                                "It took a while to find someone to come out."]))
        if f["agent_said_covered"]:
            s.append("When I bought the policy the agent told me backups were covered.")
        return noisy(join_sentences(s), rng, 0.2)

    DESK_LIMIT = 30000

    def _terms(self, f):
        sewer = f"${f['sewer_endorsement_limit']:,}" if f["sewer_endorsement_limit"] else "not purchased"
        return (f"Policy terms: deductible ${f['deductible']:,}; continuous or repeated leakage over 14 days excluded; "
                f"vacancy condition 60 days; flood and surface water excluded; interior rain damage covered only "
                f"through a storm-created opening; sewer backup endorsement {sewer}; mould sublimit "
                f"${f['mold_sublimit']:,}. Handling guideline: claims totalling under ${self.DESK_LIMIT:,} with a cause "
                f"report, photos and a rebuild estimate on file may be paid without an inspection.")

    def _plumber(self, f, rng):
        if f["roof_condition"]:
            return ROOF_TEXT[f["roof_condition"]]
        if f["plumber_finding"] is None:
            return None
        return pick(rng, PLUMBER_TEXT[f["plumber_finding"]])

    def render(self, f, rng, surface):
        account = self._account(f, rng)
        cause = self._plumber(f, rng)
        sewer = (f["sewer_endorsement_limit"] if f["sewer_endorsement_limit"] else "not purchased")
        if surface == "json":
            return dumps(prune({
                "claim": {"claim_number": f["claim_no"], "policy_number": f["policy_no"], "date_of_loss": _iso(f["loss_date"]),
                          "date_reported": _iso(f["reported_on"]), "policyholder_account": account},
                "policy": {"form": "homeowners special form", "deductible": f["deductible"],
                           "seepage_exclusion": "continuous or repeated leakage over 14 days",
                           "vacancy_condition_days": 60, "mold_sublimit": f["mold_sublimit"],
                           "sewer_backup_endorsement": sewer, "flood": "excluded",
                           "rain_damage": "covered only through a storm-created opening",
                           "inspection_guideline": f"claims under {self.DESK_LIMIT} with a cause report, photos and a "
                                                   f"rebuild estimate may be paid without inspection"},
                "property": {"occupancy": f["occupancy"], "days_vacant": f["vacant_days"], "year_built": f["year_built"],
                             "rooms_affected": f["rooms_affected"]},
                "mitigation": {"vendor": f["vendor"], "started_hours_after_discovery": f["mitigation_hours"],
                               "moisture_map": f["moisture_map"], "mold_observed": f["mold"],
                               "invoice": f["mitigation_invoice"]},
                "evidence": {"cause_report": cause, "photos": f["photos"],
                             "rebuild_estimate": f["rebuild_estimate"] if f["rebuild_estimate"] else "pending",
                             "contents_claimed": f["contents_claimed"],
                             "prior_water_claims_5y": f["prior_water_claims_5y"]},
            }))
        m, d = _mfmt(rng), _dfmt(rng)
        if surface == "claimant_email":
            extras = [f"The drying company ({f['vendor']}) invoice is {m(f['mitigation_invoice'])}."]
            if f["rebuild_estimate"]:
                extras.append(f"A contractor quoted {m(f['rebuild_estimate'])} to put the rooms back.")
            else:
                extras.append("I am still waiting on a contractor quote for the repairs.")
            if f["contents_claimed"]:
                extras.append(f"We also lost furniture and boxes worth about {m(f['contents_claimed'])}.")
            if f["plumber_finding"] == "pending":
                extras.append("The plumber is booked but has not written anything up yet.")
            elif cause:
                extras.append(("The plumber wrote: " if not f["roof_condition"] else "Your adjuster noted: ") + cause)
            if f["mold"] != "none":
                extras.append("There is mould " + ("in a couple of spots" if f["mold"] == "localized" else "on several walls now") + ".")
            if f["photos"]:
                extras.append(f"I have attached {f['photos']} photos.")
            extras.append(f"The drying company came out about {f['mitigation_hours']} hours after we found the water.")
            return "\n".join([
                f"Subject: Claim {f['claim_no']} -- water damage on {d(f['loss_date'])}",
                f"From: {f['insured']}", "To: property-claims@carrier.example", "",
                "Hello,", "", account, "", noisy(join_sentences(extras), rng, 0.15), "",
                "Please let me know what happens next.", f["insured"].split()[0], "",
                f"[claims system: {self._terms(f)}]"])
        if surface == "adjuster_diary":
            t0 = _at(f["reported_on"], rng)
            # the vendor's paperwork reaches the file after the vendor has actually been out
            on_site = f["loss_date"] + timedelta(days=f["mitigation_hours"] // 24 + 1)
            m_at = max(_at(t0, rng, 1), _at(on_site, rng))
            cov = [f"Special form, ded {m(f['deductible'])}. Sewer backup endorsement: "
                   + (m(f["sewer_endorsement_limit"]) if f["sewer_endorsement_limit"] else "none") + "."]
            if f["occupancy"] == "vacant":
                cov.append(f"Dwelling vacant {f['vacant_days']} days at loss; vacancy condition is 60 days.")
            if f["nature"] == "gradual" or (f["nature"] == "sudden" and (f["discovery"] != "as_it_happened"
                                                                          or f["plumber_finding"] in ("long_term_leak_indicated", "mixed_findings"))):
                cov.append("Need to rule out the 14-day seepage exclusion.")
            if f["nature"] == "flood":
                cov.append("Surface water: check flood exclusion.")
            if f["nature"] == "storm":
                cov.append("Interior rain damage is covered only through a storm-created opening; check roof condition.")
            entries = [
                (t0, "FNOL", f"Water loss reported {d(f['reported_on'])}, DOL {d(f['loss_date'])}. PH states: {account}"),
                (t0 + timedelta(minutes=rng.randint(20, 90)), "COVERAGE", " ".join(cov)),
                (m_at, "MITIGATION",
                 f"{f['vendor']} on site {f['mitigation_hours']} h after discovery. Invoice {m(f['mitigation_invoice'])}. "
                 f"Moisture map {'received' if f['moisture_map'] else 'not received'}. Mould: {f['mold']}."),
                (_at(m_at, rng, 1), "CAUSE", cause or "No cause report on file."),
                (_at(m_at, rng, pick(rng, [2, 3, 5])), "ESTIMATE",
                 (f"Rebuild estimate {m(f['rebuild_estimate'])}" if f["rebuild_estimate"] else "Rebuild estimate outstanding")
                 + (f"; contents list {m(f['contents_claimed'])}" if f["contents_claimed"] else "")
                 + f". {f['photos']} photos. Prior water claims 5y: {f['prior_water_claims_5y']}."),
            ]
            return f"{f['claim_no']} / adjuster {f['adjuster']}\n" + _diary(entries, rng) + "\n" + self._terms(f)
        # mitigation_report: the drying vendor's job report
        lines = [f"WATER MITIGATION JOB REPORT -- {f['vendor']}",
                 kv_lines([("Customer", f["insured"]), ("Insurer claim", f["claim_no"]),
                           ("Date of loss", d(f["loss_date"])),
                           ("Arrived", f"{f['mitigation_hours']} h after the homeowner found the water"),
                           ("Source (per homeowner)", WATER_SOURCES[f["source"]][0].format(fixture=f["fixture"])),
                           ("Rooms", ", ".join(f["room_names"])), ("Occupancy", _h(f["occupancy"])),
                           ("Visible mould", f["mold"])], rng), ""]
        if f["moisture_map"]:
            hi = rng.randint(28, 45) if f["mitigation_hours"] < 24 else rng.randint(35, 60)
            lines += ["Moisture readings (dry standard 15%):",
                      table(["area", "first reading", "final reading"],
                            [[r, f"{hi - 3 * i}%", f"{rng.randint(12, 17)}%"] for i, r in enumerate(f["room_names"][:3])], rng), ""]
        else:
            lines.append("Moisture map: not attached.")
        days = 3 + (1 if f["mitigation_hours"] > 24 else 0) + (2 if f["mold"] == "widespread" else 0)
        lines.append(f"Equipment: {2 * f['rooms_affected'] + 1} air movers, {max(1, f['rooms_affected'] // 2)} dehumidifier(s), {days} days.")
        lines.append(f"Invoice total: {m(f['mitigation_invoice'])}")
        tech = []
        if f["plumber_finding"] == "long_term_leak_indicated":
            tech.append("Tech note: framing under the source is dark and soft, looks like it has been wet a long time.")
        elif f["plumber_finding"] == "mixed_findings":
            tech.append("Tech note: fresh water on top, but some older staining on the joists.")
        if f["mold"] == "widespread":
            tech.append("Mould remediation will need a separate protocol and containment.")
        if not f["rebuild_estimate"]:
            tech.append("Rebuild not in our scope; homeowner getting quotes.")
        if tech:
            lines.append(noisy(" ".join(tech), rng, 0.2))
        lines += ["", f"[insurer file note: {self._terms(f)}]"]
        return "\n".join(lines)

    def questions(self):
        return {
            "water_coverage_position": choice("How does the homeowners policy most likely respond to this water loss?", {
                "covered_sudden_discharge": "Covered as a sudden and accidental discharge, or rain through a fresh "
                                            "storm opening, with only the deductible taken off.",
                "covered_to_sublimit": "Covered only up to the sewer backup endorsement limit, which may fall short "
                                       "of the loss.",
                "partly_covered": "Covered in part: a fresh failure with older seepage or roof wear alongside it, or "
                                  "widespread mould beyond the mould sublimit, has to be carved out.",
                "excluded": "Not covered: flood or surface water, leakage over 14 days, a worn roof with no storm "
                            "opening, a sewer backup with no endorsement, or a home vacant more than 60 days.",
                "cause_not_established": "Cannot be decided yet: no cause report is in and the water may have run "
                                         "for longer than 14 days unnoticed.",
            }),
            "cause_documentation": score("How well does the file establish the cause and extent of the water damage?", [
                "Thin: no cause report and no photos; little beyond the policyholder's account and the drying invoice.",
                "Partial: photos are in, but no plumber or adjuster report documents the cause.",
                "Workable: a cause report is on file, but the moisture map, photos or rebuild estimate is missing.",
                "Complete: a cause report, moisture readings, photos and an itemised rebuild estimate are all on file.",
            ]),
            "cost_growth_risk": score("How likely is the final cost to grow well beyond the figures now on file?", [
                "Low: the damage is contained and the numbers look final.",
                "Moderate: some tear-out or contents still has to be priced.",
                "High: several rooms, slow drying or a missing rebuild estimate point to growth.",
                "Very high: mould, saturated structure or time out of the home will add substantially.",
            ]),
            "mitigation_duty_met": noul(
                "The policyholder took reasonable steps to limit further damage once the water was found.",
                yes="Drying began within 48 hours of finding the water, and no earlier stain or leak had been noticed "
                    "and left.",
                no="Drying began more than 48 hours after the water was found, or an earlier stain or leak was "
                   "noticed and left alone."),
            "inspection_before_payment": noul(
                "An adjuster should inspect the property in person before any payment is made.",
                yes="The claim falls outside the no-inspection guideline: the total is at or above the limit, a cause "
                    "report, photos or the rebuild estimate is missing, or mould is widespread.",
                no="The total is under the guideline limit and the cause report, photos and rebuild estimate are all on "
                   "file, with no widespread mould."),
        }


# =========================================================================== schema 3: theft / burglary
THEFT_ITEMS = {
    # category -> [(description, typical value)]
    "electronics": [("laptop", 1400), ("gaming console and games", 650), ("65-inch television", 1100),
                    ("tablet", 800), ("camera body and two lenses", 2600), ("noise-cancelling headphones", 380)],
    "jewelry": [("diamond engagement ring", 6500), ("gold chain necklace", 1800), ("pair of diamond earrings", 1900),
                ("gold bracelet", 1200)],
    "watches": [("automatic wristwatch", 4200), ("smartwatch", 450)],
    "cash": [("cash kept in a drawer", 600), ("cash in an envelope", 1500)],
    "bikes": [("carbon road bike", 3800), ("electric bike", 2400), ("children's bikes (2)", 500)],
    "tools": [("cordless power tool set", 1300), ("mechanic's tool chest", 2900), ("mitre saw", 700)],
    "apparel": [("designer handbag", 2200), ("leather jacket", 650), ("ski gear", 900)],
}
THEFT_SETTINGS = {
    "home_burglary": (["electronics", "jewelry", "watches", "cash", "apparel"], "house"),
    "apartment_burglary": (["electronics", "jewelry", "cash", "apparel"], "apartment"),
    "vehicle_break_in": (["electronics", "apparel", "tools"], "parked car"),
    "garage_or_shed": (["bikes", "tools"], "detached garage"),
    "storage_unit": (["tools", "bikes", "electronics", "apparel"], "rented storage unit"),
}
PROOFS = ["receipt", "card_statement", "photo", "none"]


class TheftBurglaryClaim(Schema):
    name = "theft_burglary_claim"
    description = ("A homeowners or renters theft claim with a police report and an inventory: handling path, "
                   "fraud indicators, proof of ownership and how much the policy would pay.")
    surfaces = {"json": 0.45, "police_and_inventory": 0.3, "statement_summary": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        setting = pick_w(rng, {"home_burglary": 4, "apartment_burglary": 2, "vehicle_break_in": 2,
                               "garage_or_shed": 2, "storage_unit": 1})
        entry = pick_w(rng, {"forced": 5, "none_visible": 2, "admits_unlocked": 1, None: 1})
        police = pick_w(rng, {"same_day": 5, "late": 2, "not_filed": 1})
        police_lag = rng.randint(3, 12)
        n_items = pick(rng, [2, 3, 3, 4, 5])
        proof_w = {"receipt": 3, "card_statement": 2, "photo": 2, "none": 2}
        policy_age = pick(rng, [25, 60, 110, 300, 800, 1600, 3200]) + rng.randint(0, 30)
        limit_raised_days = pick_w(rng, {None: 8, "recent": 1})
        prior = pick(rng, [0, 0, 0, 0, 1, 2])
        growth = pick_w(rng, {0: 6, 1: 2, 3: 1})          # items added after the first call
        security = pick_w(rng, {"alarm_triggered": 2, "alarm_not_set": 2, "camera_footage": 2, "none": 4})
        arrears = chance(rng, 0.12)
        basis = pick_w(rng, {"replacement_cost": 3, "actual_cash_value": 2})
        scheduled = chance(rng, 0.15)
        heavy = None                                        # force a category into the inventory
        twist = None

        if reg == "clear":
            if chance(rng, 0.55):
                twist = "clean"
                entry, police, prior, growth, arrears = "forced", "same_day", 0, 0, False
                policy_age, limit_raised_days = rng.randint(900, 4000), None
                proof_w = {"receipt": 5, "card_statement": 3, "photo": 1, "none": 0.2}
                security = pick(rng, ["alarm_triggered", "camera_footage"])
                setting = pick(rng, ["home_burglary", "apartment_burglary", "garage_or_shed"])
            else:
                twist = "suspicious"
                entry, police, police_lag = pick(rng, ["none_visible", None]), "late", rng.randint(6, 15)
                policy_age, limit_raised_days, prior, growth = rng.randint(20, 70), "recent", pick(rng, [1, 2]), 3
                proof_w = {"receipt": 0.3, "card_statement": 0.5, "photo": 1, "none": 5}
                security, heavy = pick(rng, ["alarm_not_set", "none"]), "jewelry"
                setting = pick(rng, ["home_burglary", "apartment_burglary"])
        elif reg == "borderline":
            twist = pick(rng, ["jewelry_near_limit", "partial_proof", "newish_policy", "police_a_few_days"])
            if twist == "jewelry_near_limit":
                setting, heavy, scheduled = pick(rng, ["home_burglary", "apartment_burglary"]), "jewelry_small", False
            elif twist == "partial_proof":
                proof_w = {"receipt": 2, "card_statement": 1, "photo": 2, "none": 3}
            elif twist == "newish_policy":
                policy_age = near(rng, 90, 40, lo=20, integer=True)
            else:
                police, police_lag = "late", near(rng, 3, 2, lo=1, integer=True)
        else:
            twist = pick(rng, ["documented_but_grew", "tenured_no_entry", "unattended_car_no_force",
                               "footage_contradicts", "receipts_but_new_limit"])
            if twist == "documented_but_grew":
                proof_w, growth = {"receipt": 5, "card_statement": 2, "photo": 1, "none": 0.3}, 3
            elif twist == "tenured_no_entry":
                policy_age, prior, entry, police = rng.randint(2000, 5000), 0, "none_visible", "same_day"
            elif twist == "unattended_car_no_force":
                setting, entry = pick(rng, ["vehicle_break_in", "storage_unit"]), pick(rng, ["none_visible", "admits_unlocked"])
                proof_w = {"receipt": 4, "card_statement": 2, "photo": 1, "none": 0.5}
            elif twist == "footage_contradicts":
                security, entry = "camera_footage", "forced"
            else:
                limit_raised_days, proof_w = "recent", {"receipt": 5, "card_statement": 3, "photo": 1, "none": 0.3}
                policy_age = rng.randint(700, 3000)
            if chance(rng, 0.35) and twist != "unattended_car_no_force":
                entry = None                                 # the entry evidence is simply not recorded

        # ---- inventory and amounts, derived after the regime block
        cats = list(THEFT_SETTINGS[setting][0])
        items = []
        if heavy in ("jewelry", "jewelry_small"):
            desc, val = pick(rng, THEFT_ITEMS["jewelry"])
            if heavy == "jewelry_small":
                val = near(rng, 1500, 700, lo=600)
            items.append({"item": desc, "category": "jewelry",
                          "claimed": int(round(val * (rng.uniform(0.8, 1.3) if heavy == "jewelry" else 1), -1))})
        while len(items) < n_items:
            cat = pick(rng, cats)
            desc, val = pick(rng, THEFT_ITEMS[cat])
            if any(it["item"] == desc or (cat == "cash" and it["category"] == "cash") for it in items):
                continue
            items.append({"item": desc, "category": cat, "claimed": int(round(val * rng.uniform(0.6, 1.5), -1))})
        for it in items:
            it["proof"] = (pick_w(rng, {"bank_withdrawal_record": 1, "none": 2}) if it["category"] == "cash"
                           else pick_w(rng, proof_w))
            it["age_years"] = pick(rng, [0.5, 1, 2, 3, 5, 8]) if it["category"] != "cash" else None
        if twist == "receipts_but_new_limit" or twist == "suspicious":
            for it in items[: max(1, len(items) // 2)]:
                it["age_years"] = pick(rng, [0.1, 0.2])        # much of it bought just before the loss
        claimed_total = sum(it["claimed"] for it in items)
        first_call_items = max(1, len(items) - growth) if growth else len(items)
        contents_limit = pick(rng, [30000, 50000, 75000, 100000])
        prev_limit = None
        loss_date = _day(rng)
        raised_on = None
        if limit_raised_days == "recent":
            raised_days = near(rng, 30, 20, lo=3, integer=True)
            raised_on = loss_date - timedelta(days=raised_days)
            prev_limit = contents_limit // 2
            policy_age = max(policy_age, raised_days + 10)
        forced_condition = setting in ("vehicle_break_in", "storage_unit")
        police_filed = None if police == "not_filed" else loss_date + timedelta(days=0 if police == "same_day" else police_lag)
        return {
            "regime": reg, "twist": twist, "setting": setting, "entry_evidence": entry, "police": police,
            "police_filed_on": police_filed, "items": items, "claimed_total": claimed_total,
            "first_call_item_count": first_call_items, "inventory_growth": growth, "security": security,
            "policy_age_days": policy_age, "inception": loss_date - timedelta(days=policy_age),
            "contents_limit": contents_limit, "limit_raised_on": raised_on, "previous_limit": prev_limit,
            "prior_theft_claims_5y": prior, "premium_in_arrears": arrears, "settlement_basis": basis,
            "jewelry_scheduled": scheduled, "special_limit_jewelry": 1500, "special_limit_cash": 250,
            "forced_entry_condition": forced_condition, "deductible": pick(rng, [500, 1000, 1000, 2500]),
            "loss_date": loss_date, "reported_on": loss_date + timedelta(days=pick(rng, [0, 1, 2, 3, 6])),
            "claim_no": _claim_no(rng, "TH", loss_date), "policy_no": _policy_no(rng, pick(rng, ["HO", "RN"])),
            "insured": person(rng), "adjuster": _handler(rng),
            "police_ref": f"PD-{loss_date.year}-{rng.randrange(10000, 99999)}",
            "footage_note": pick(rng, ["footage shows a figure at the side door at 02:14",
                                       "footage covers the driveway only and shows no one arriving",
                                       "camera was offline from 22:00 to 06:00"]),
        }

    def _account(self, f, rng):
        where = THEFT_SETTINGS[f["setting"]][1]
        s = [pick(rng, [f"Insured says the {where} was broken into on {f['loss_date'].strftime('%A')}.",
                        f"Insured reports a theft from the {where}."])]
        s.append({"forced": pick(rng, ["The back door lock was snapped.", "A window was smashed to get in."]),
                  "none_visible": "There are no marks on the doors or windows; insured cannot say how they got in.",
                  "admits_unlocked": "Insured admits it may have been left unlocked.",
                  None: ""}[f["entry_evidence"]])
        s.append({"alarm_triggered": "The alarm went off and the monitoring company called the insured.",
                  "alarm_not_set": "There is an alarm but it was not set that night.",
                  "camera_footage": f"There is a doorbell camera on the house; {f['footage_note']}.", "none": ""}[f["security"]])
        if f["inventory_growth"]:
            s.append(f"On the first call the insured listed {f['first_call_item_count']} item(s); the written "
                     f"inventory has {len(f['items'])}.")
        return noisy(join_sentences(s), rng, 0.15)

    def render(self, f, rng, surface):
        account = self._account(f, rng)
        if surface == "json":
            return dumps(prune({
                "claim": {"claim_number": f["claim_no"], "loss_type": f"theft - {_h(f['setting'])}",
                          "date_of_loss": _iso(f["loss_date"]), "date_reported": _iso(f["reported_on"]),
                          "insured_account": account},
                "police": ({"report_number": f["police_ref"], "filed_on": _iso(f["police_filed_on"]),
                            "forced_entry_noted": {"forced": True, "none_visible": False,
                                                   "admits_unlocked": False}.get(f["entry_evidence"])}
                           if f["police"] != "not_filed" else {"filed": False}),
                "policy": {"policy_number": f["policy_no"], "inception_date": _iso(f["inception"]),
                           "contents_limit": f["contents_limit"], "limit_raised_on": _iso(f["limit_raised_on"]),
                           "previous_contents_limit": f["previous_limit"], "deductible": f["deductible"],
                           "settlement_basis": f["settlement_basis"],
                           "special_limits": {"jewelry_and_watches_theft": f["special_limit_jewelry"],
                                              "cash": f["special_limit_cash"]},
                           "jewelry_scheduled": f["jewelry_scheduled"],
                           "theft_condition": "visible forced entry required" if f["forced_entry_condition"] else None},
                "inventory": [{"item": it["item"], "claimed": it["claimed"], "proof": it["proof"],
                               "age_years": it["age_years"]} for it in f["items"]],
                "history": {"prior_theft_claims_5y": f["prior_theft_claims_5y"],
                            "premium_in_arrears": f["premium_in_arrears"],
                            "items_on_first_call": f["first_call_item_count"]},
            }))
        m, d = _mfmt(rng), _dfmt(rng)
        rows = [[it["item"], m(it["claimed"]), _h(it["proof"]), "-" if it["age_years"] is None else f"{it['age_years']} yr"]
                for it in f["items"]]
        inv = table(["item", "claimed", "proof", "age"], rows, rng)
        if surface == "police_and_inventory":
            pol = (kv_lines([("Report", f["police_ref"]), ("Filed", d(f["police_filed_on"])),
                             ("Offence", "burglary" if f["setting"] in ("home_burglary", "apartment_burglary") else "theft"),
                             ("Point of entry", {"forced": "forced", "none_visible": "no visible forced entry",
                                                 "admits_unlocked": "possibly unlocked", None: "not recorded"}[f["entry_evidence"]]),
                             ("Suspects", "none identified")], rng)
                   if f["police"] != "not_filed" else "No police report filed as of this review.")
            pol_head = "POLICE REPORT SUMMARY" if f["police"] != "not_filed" else "POLICE REPORT"
            pol_text = kv_lines([
                ("Policy", f"{f['policy_no']} since {d(f['inception'])}"),
                ("Contents limit", m(f["contents_limit"]) + (f" (raised from {m(f['previous_limit'])} on {d(f['limit_raised_on'])})" if f["limit_raised_on"] else "")),
                ("Deductible", m(f["deductible"])), ("Settlement basis", _h(f["settlement_basis"])),
                ("Special limits", f"jewellery/watches theft {m(f['special_limit_jewelry'])}, cash {m(f['special_limit_cash'])}"
                                   + ("; jewellery scheduled" if f["jewelry_scheduled"] else "")),
                ("Theft condition", "visible forced entry required" if f["forced_entry_condition"] else None),
                ("Prior theft claims (5y)", f["prior_theft_claims_5y"]),
                ("Premium in arrears", "yes" if f["premium_in_arrears"] else None),
            ], rng)
            return "\n".join([f"Claim {f['claim_no']} -- theft, DOL {d(f['loss_date'])}, reported {d(f['reported_on'])}", "",
                              pol_head, pol, "", "POLICY", pol_text, "",
                              f"INVENTORY (claimed total {m(f['claimed_total'])}; {f['first_call_item_count']} item(s) given on first call)",
                              inv, "", "Account: " + account])
        # statement_summary: the adjuster's write-up of a recorded phone statement
        s = [f"Recorded statement taken from {f['insured']} on {d(f['reported_on'] + timedelta(days=rng.randint(1, 5)))} by {f['adjuster']}.",
             account]
        if f["police"] == "same_day":
            s.append("Police were called the same day.")
        elif f["police"] == "late":
            s.append(f"Police report made {(f['police_filed_on'] - f['loss_date']).days} day(s) after the loss"
                     + pick(rng, [".", "; insured says they were in shock.", "; no reason given."]))
        else:
            s.append("No police report; insured did not think it was worth it.")
        s.append(f"Policy in force since {d(f['inception'])}"
                 + (f"; contents limit raised to {m(f['contents_limit'])} on {d(f['limit_raised_on'])}." if f["limit_raised_on"] else "."))
        if f["prior_theft_claims_5y"]:
            s.append(f"{f['prior_theft_claims_5y']} earlier theft claim(s) in five years.")
        if f["forced_entry_condition"]:
            s.append("Policy covers theft from vehicles and storage only with visible forced entry.")
        return "\n".join([f"STATEMENT SUMMARY -- {f['claim_no']}", noisy(join_sentences(s), rng, 0.1), "",
                          f"Inventory claimed ({m(f['claimed_total'])}, {_h(f['settlement_basis'])}, ded {m(f['deductible'])}):",
                          inv])

    def questions(self):
        return {
            "theft_claim_routing": choice("How should this theft claim be handled from here?", {
                "desk_settlement": "Value and settle it from the documents on file: a police report, proof for most "
                                   "of the value, and at most one fraud indicator.",
                "pend_for_documents": "Pend it for the police report or proof of ownership: no police report was "
                                      "filed, or most of the claimed value has no proof.",
                "statement_and_verification": "Take a recorded statement and check the account with neighbours, "
                                              "footage or retailers: two to four fraud indicators are present.",
                "siu_referral": "Refer it to the special investigations unit: five or more fraud indicators together.",
            }),
            "fraud_indicator_strength": score("How strong are the indicators that this theft claim is false or inflated?", [
                "Negligible: none of the usual indicators (a policy under 90 days old, a recently raised limit, a "
                "growing inventory, no visible forced entry, a late or missing police report, earlier theft claims, "
                "goods bought just before the loss).",
                "Slight: one or two of those indicators.",
                "Considerable: three or four of them.",
                "Pronounced: five or more of them together.",
            ]),
            "ownership_proof_strength": score("How well does the file prove the insured owned the claimed items and what they were worth?", [
                "Unsupported: the values rest on the insured's word alone.",
                "Weak: proof covers only a minority of the claimed value.",
                "Fair: most of the high-value items have some proof.",
                "Strong: receipts, statements or photos back nearly every item.",
            ]),
            "payable_share": choice("If the insured's account is accepted, how much of the claimed total would the policy pay?", {
                "close_to_full": "Close to the full amount claimed, less only the deductible.",
                "cut_by_special_limits": "Materially less, because special limits for jewellery, watches or cash "
                                         "cap part of the claim.",
                "cut_by_depreciation": "Materially less, because the policy pays actual cash value and older items "
                                       "would be depreciated.",
                "little_or_nothing": "Little or nothing, because the loss is under the deductible or a theft "
                                     "condition of the policy is not met.",
            }),
            "account_consistency": noul(
                "The insured's account of the theft holds together across the documents on file.",
                yes="Dates, entry evidence, the police report and the inventory agree with one another.",
                no="The inventory grew, the dates shift, or the physical evidence does not fit the story."),
        }


# =========================================================================== schema 4: reserve review
BUSINESSES = ["bakery", "hardware store", "dental office", "print shop", "yoga studio", "florist", "bike shop",
              "tailor's shop", "coffee roaster"]
RESERVE_LINES = {
    # line: (description template, limit options, typical valuation, authority options)
    "commercial_property_fire": ("Fire at the insured {biz}: building, contents and business income",
                                 [350000, 500000, 750000, 1000000], 140000, [50000, 100000, 150000]),
    "homeowners_fire": ("Kitchen fire at the insured dwelling: structure, contents and additional living expense",
                        [250000, 350000, 450000, 600000], 70000, [25000, 50000, 75000]),
    "auto_bodily_injury": ("Third-party injury claim after the insured driver rear-ended the claimant",
                           [25000, 50000, 100000, 250000], 21000, [15000, 25000, 50000]),
    "premises_liability": ("Customer slip-and-fall at the insured {biz}",
                           [300000, 500000, 1000000], 42000, [25000, 50000, 100000]),
}
# developments after (or before) the last valuation: (key, text template, multiplier on the valuation)
RESERVE_DEVS = {
    "commercial_property_fire": [
        ("supplement", "Contractor supplement for smoke damage found in the adjoining unit", (1.15, 1.35)),
        ("bi_extended", "Reopening pushed back {n} weeks; business income period extended", (1.2, 1.5)),
        ("public_adjuster", "Insured retained a public adjuster, whose estimate is {figure}", (1.1, 1.3)),
        ("code_upgrade", "Building department requires a full electrical code upgrade before reoccupancy", (1.15, 1.4)),
        ("repairs_complete", "Repairs complete; final invoices came in under the estimate", (0.78, 0.92)),
    ],
    "homeowners_fire": [
        ("ale_extended", "Rebuild waiting on permits; temporary housing extended {n} weeks", (1.1, 1.3)),
        ("contents_list", "Contents inventory received, totalling {figure}", (1.2, 1.6)),
        ("public_adjuster", "Insured retained a public adjuster, whose estimate is {figure}", (1.1, 1.3)),
        ("repairs_complete", "Rebuild complete; final invoice below the estimate", (0.8, 0.93)),
    ],
    "auto_bodily_injury": [
        ("demand", "Demand letter received for {figure}", (1.2, 1.7)),
        ("surgery", "Treating surgeon recommends a lumbar fusion", (1.9, 3.0)),
        ("released", "Claimant released from treatment; specials closed", (0.7, 0.88)),
        ("suit", "Suit filed; defence counsel assigned", (1.3, 1.8)),
    ],
    "premises_liability": [
        ("demand", "Demand letter received for {figure}", (1.2, 1.7)),
        ("surgery", "MRI shows a torn meniscus; surgery scheduled", (1.6, 2.4)),
        ("surveillance", "Surveillance shows the claimant playing five-a-side football", (0.55, 0.8)),
        ("suit", "Suit filed; defence counsel assigned", (1.3, 1.8)),
    ],
}


class ReserveReview(Schema):
    name = "reserve_review"
    description = ("A supervisor's reserve review on an open claim (small commercial fire, homeowners fire, auto "
                   "bodily injury or premises liability): adjust the reserve, judge its adequacy and the file's "
                   "valuation.")
    surfaces = {"json": 0.45, "reserve_worksheet": 0.3, "supervisor_diary": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        line = pick_w(rng, {"commercial_property_fire": 3, "homeowners_fire": 2, "auto_bodily_injury": 3,
                            "premises_liability": 2})
        tmpl, limits, typical, auths = RESERVE_LINES[line]
        limit = pick(rng, limits)
        authority = pick(rng, auths)
        valuation = min(lognormal_amount(rng, typical, 0.45, 0), limit * 0.9)
        devs = RESERVE_DEVS[line]
        dev = pick(rng, devs + [None])
        after_val = chance(rng, 0.6)                       # did it happen after the last valuation?
        mult = 1.0
        coverage_question = chance(rng, 0.12)
        adequacy = rng.uniform(0.7, 1.25)                 # (paid + reserve) / likely ultimate
        val_age = pick(rng, [9, 21, 40, 75, 130, 210])
        adjuster_view = pick_w(rng, {"adequate": 3, "watching": 2, "none": 2})
        twist = None

        if reg == "clear":
            twist = pick(rng, ["well_set", "badly_short", "over_reserved"])
            if twist == "well_set":
                dev, adequacy, val_age = None, rng.uniform(0.98, 1.1), rng.randint(5, 30)
            elif twist == "badly_short":
                dev = pick(rng, [x for x in devs if x[2][0] >= 1.2])
                after_val, adequacy, val_age = True, 1.0, rng.randint(120, 260)
            else:
                dev = pick(rng, [x for x in devs if x[2][1] < 1.0] or devs)
                after_val, adequacy, val_age = True, 1.0, rng.randint(30, 120)
        elif reg == "borderline":
            twist = pick(rng, ["slightly_short", "near_authority", "dev_partly_priced"])
            if twist == "slightly_short":
                adequacy = near(rng, 0.9, 0.08)
            elif twist == "near_authority":
                adequacy = near(rng, 0.88, 0.06)
                authority = None                           # set from the numbers below
            else:
                dev = pick(rng, devs)
                after_val = True
                adequacy = 1.0
                val_age = rng.randint(15, 60)
        else:
            twist = pick(rng, ["adjuster_says_fine", "claimant_figure_vs_own", "stale_but_quiet", "old_dev_new_val"])
            if twist == "adjuster_says_fine":
                dev = pick(rng, [x for x in devs if x[2][0] >= 1.15])
                after_val, adequacy, adjuster_view = True, 1.0, "adequate"
            elif twist == "claimant_figure_vs_own":
                dev = pick(rng, [x for x in devs if x[0] in ("public_adjuster", "demand", "contents_list")] or devs)
                after_val, adequacy = True, 1.0
            elif twist == "stale_but_quiet":
                dev, val_age, adequacy = None, rng.randint(180, 320), rng.uniform(0.95, 1.1)
                adjuster_view = "adequate"
            else:
                dev = pick(rng, [x for x in devs if x[2][0] >= 1.15])
                after_val, adequacy, val_age = False, rng.uniform(0.95, 1.08), rng.randint(10, 40)
            if chance(rng, 0.3):
                coverage_question = True

        # ---- derived after the regime block
        if dev is not None:
            mult = rng.uniform(*dev[2])
        dev_key = dev[0] if dev else None
        ultimate = min(valuation * (mult if (dev is not None and after_val) else 1.0), limit)
        reserve_reflects_dev = False
        if twist == "dev_partly_priced":
            incurred = valuation + (ultimate - valuation) * rng.uniform(0.3, 0.7)
            reserve_reflects_dev = True
        elif dev is not None and after_val and (reg in ("clear", "conflicting") and twist != "stale_but_quiet"):
            incurred = valuation * rng.uniform(0.97, 1.03)      # reserve still sits at the old valuation
        else:
            incurred = ultimate * adequacy
            reserve_reflects_dev = dev is not None and after_val
        incurred = min(incurred, limit)
        casualty = line in ("auto_bodily_injury", "premises_liability")
        if casualty:          # liability claims pay little before settlement (medical payments, an advance)
            paid_share = 0.0 if chance(rng, 0.4) else rng.uniform(0.01, 0.08)
        elif dev_key == "repairs_complete":
            paid_share = rng.uniform(0.8, 0.95)
        else:
            paid_share = rng.uniform(0.1, 0.55)
        paid = round(min(incurred, ultimate) * paid_share, 2)
        outstanding = int(max(round(incurred - paid, -2), 500))
        paid = _num(paid)
        incurred = _num(round(paid + outstanding, 2))
        if authority is None:
            authority = min(RESERVE_LINES[line][3], key=lambda a: abs(a - ultimate))
        figure = None
        if dev_key in ("demand", "public_adjuster", "contents_list"):
            figure = (int(round(ultimate * rng.uniform(1.2, 2.0), -3)) if dev_key != "contents_list"
                      else int(round(ultimate * 0.35, -2)))
            if dev_key == "demand" and chance(rng, 0.3):
                figure = int(limit)                        # a policy-limits demand
        weeks = rng.randint(4, 14)
        dev_text = dev[1].format(n=weeks, figure=_money(figure) if figure else "") if dev else None
        dev_age = (rng.randint(3, max(4, val_age - 2)) if after_val else val_age + rng.randint(5, 40)) if dev else None
        # when was the reserve last moved? after the development if it already reflects it; on the valuation date
        # if it matches the valuation; otherwise some time before the valuation
        if reserve_reflects_dev:
            since_change = rng.randint(1, max(1, dev_age - 1))
        elif abs(incurred / valuation - 1) < 0.04 and twist != "stale_but_quiet":
            since_change = val_age
        else:
            since_change = val_age + rng.randint(5, 90)
        claim_age = max(val_age + rng.randint(30, 400), (dev_age or 0) + 20, since_change + 15)
        as_of = _day(rng, date(2025, 9, 1), date(2026, 9, 25))
        loss_date = as_of - timedelta(days=claim_age)
        litigation = "suit_filed" if dev_key == "suit" else pick_w(rng, {"none": 3, "attorney_letter": 2})
        if dev_key in ("demand", "surgery") and litigation == "none" and chance(rng, 0.75):
            litigation = "attorney_letter"
        if not casualty:
            litigation = "public_adjuster" if dev_key == "public_adjuster" else "none"
        return {
            "regime": reg, "twist": twist, "line": line, "business": pick(rng, BUSINESSES), "policy_limit": limit,
            "adjuster_authority": authority, "last_valuation": int(round(valuation, -2)), "valuation_age_days": val_age,
            "development": dev_key, "development_text": dev_text, "development_after_valuation": after_val if dev else None,
            "development_age_days": dev_age, "claimant_figure": figure, "likely_ultimate": int(round(ultimate, -2)),
            "paid": paid, "outstanding_reserve": outstanding, "total_incurred": incurred,
            "needed_outstanding": int(round(max(ultimate - paid, 0), -2)),
            "expense_reserve": int(round(lognormal_amount(rng, 9000 if litigation == "suit_filed" else 3500, 0.4), -2)),
            "days_since_reserve_change": since_change, "claim_age_days": claim_age, "as_of": as_of,
            "loss_date": loss_date, "litigation": litigation,
            "reservation_of_rights": coverage_question, "adjuster_view": adjuster_view,
            "claim_no": _claim_no(rng, {"commercial_property_fire": "CP", "homeowners_fire": "HO",
                                        "auto_bodily_injury": "BI", "premises_liability": "GL"}[line], loss_date),
            "adjuster": _handler(rng), "supervisor": _handler(rng), "insured": person(rng),
        }

    def _effect(self, f):
        if not f["development"]:
            return None
        if not f["development_after_valuation"]:
            return "already reflected in the latest valuation"
        delta = f["likely_ultimate"] - f["last_valuation"]
        return f"handler's estimate of the effect on total indemnity: {'+' if delta >= 0 else '-'}{_money(abs(delta))}"

    def _devline(self, f, d):
        if not f["development"]:
            return None
        return (f"{d(f['as_of'] - timedelta(days=f['development_age_days']))}: {f['development_text']} "
                f"({self._effect(f)})")

    def _view(self, f, rng):
        return {"adequate": pick(rng, ["Reserve is adequate as set.", "Reserve looks right to me; no change needed."]),
                "watching": pick(rng, ["Holding the reserve until the next report comes in.",
                                       "Will revisit after the next medical update."
                                       if f["line"] in ("auto_bodily_injury", "premises_liability")
                                       else "Will revisit once the contractor's final figures are in."]),
                "none": None}[f["adjuster_view"]]

    def render(self, f, rng, surface):
        desc = RESERVE_LINES[f["line"]][0].format(biz=f["business"])
        val_on = f["as_of"] - timedelta(days=f["valuation_age_days"])
        changed_on = f["as_of"] - timedelta(days=f["days_since_reserve_change"])
        if surface == "json":
            return dumps(prune({
                "claim": {"claim_number": f["claim_no"], "line": f["line"], "description": desc,
                          "date_of_loss": _iso(f["loss_date"]), "review_date": _iso(f["as_of"]),
                          "handler": f["adjuster"], "handler_incurred_authority": f["adjuster_authority"]},
                "coverage": {"policy_limit": f["policy_limit"], "reservation_of_rights": f["reservation_of_rights"]},
                "financials": {"paid_indemnity": f["paid"], "outstanding_indemnity_reserve": f["outstanding_reserve"],
                               "total_incurred_indemnity": f["total_incurred"], "expense_reserve": f["expense_reserve"],
                               "last_reserve_change": _iso(changed_on)},
                "valuation": {"latest_estimate_total_indemnity": f["last_valuation"], "as_of": _iso(val_on),
                              "handler_comment": self._view(f, rng)},
                "developments": ([{"date": _iso(f["as_of"] - timedelta(days=f["development_age_days"])),
                                   "event": f["development_text"], "effect": self._effect(f)}]
                                 if f["development"] else None),
                "claimant": {"representation": f["litigation"]},
            }))
        m, d = _mfmt(rng), _dfmt(rng)
        if surface == "reserve_worksheet":
            fin = table(["", "paid", "outstanding", "incurred"],
                        [["indemnity", m(f["paid"]), m(f["outstanding_reserve"]), m(f["total_incurred"])],
                         ["expense", "--", m(f["expense_reserve"]), "--"]], rng)
            head = kv_lines([("Claim", f["claim_no"]), ("Insured", f["insured"]), ("Loss", desc),
                             ("Date of loss", d(f["loss_date"])), ("Review date", d(f["as_of"])),
                             ("Policy limit", m(f["policy_limit"])),
                             ("Reservation of rights", "issued" if f["reservation_of_rights"] else "none"),
                             ("Representation", _h(f["litigation"])),
                             ("Handler / authority", f"{f['adjuster']} / {m(f['adjuster_authority'])} incurred")], rng)
            val = (f"Latest handler valuation of total indemnity: {m(f['last_valuation'])} as of {d(val_on)}. "
                   f"Reserve last changed {d(changed_on)}.")
            dl = self._devline(f, d)
            view = self._view(f, rng)
            return "\n".join(x for x in ["RESERVE REVIEW WORKSHEET", head, "", fin, "", val,
                                         ("Development since opening: " + dl) if dl else "No new developments logged.",
                                         ("Handler comment: " + view) if view else None] if x)
        # supervisor_diary
        t = datetime(f["as_of"].year, f["as_of"].month, f["as_of"].day, rng.randint(8, 11), rng.randrange(60))
        entries = []
        if f["development"] and not f["development_after_valuation"]:
            dt = f["as_of"] - timedelta(days=f["development_age_days"])
            entries.append((datetime(dt.year, dt.month, dt.day, 10, 5), "FILE",
                            f"{f['development_text']} ({self._effect(f)})."))
        entries.append((datetime(val_on.year, val_on.month, val_on.day, 15, 20), "EVAL",
                        f"Handler evaluation: total indemnity {m(f['last_valuation'])}. " + (self._view(f, rng) or "")))
        if f["development"] and f["development_after_valuation"]:
            dt = f["as_of"] - timedelta(days=f["development_age_days"])
            entries.append((datetime(dt.year, dt.month, dt.day, 11, 40), "FILE",
                            f"{f['development_text']} ({self._effect(f)})."))
        entries.append((t, "SUPV REVIEW",
                        f"{f['supervisor']}: diary review. Paid {m(f['paid'])}, outstanding {m(f['outstanding_reserve'])}, "
                        f"incurred {m(f['total_incurred'])} against a {m(f['policy_limit'])} limit. Handler authority "
                        f"{m(f['adjuster_authority'])}. Reserve last moved {d(changed_on)}."
                        + (" ROR letter out on coverage." if f["reservation_of_rights"] else "")
                        + (f" Claimant: {_h(f['litigation'])}." if f["litigation"] != "none" else "")))
        entries.sort(key=lambda e: e[0])
        return f"{f['claim_no']} -- {desc}\n" + _diary(entries, rng)

    def questions(self):
        return {
            "reserve_action": choice("What should the claims supervisor do with the indemnity reserve on this file?", {
                "lower_reserve": "Reduce it; more is set aside than the evidence on file supports.",
                "hold_reserve": "Leave it unchanged and look again at the next diary date.",
                "raise_within_authority": "Raise it by an amount the handling adjuster can approve alone.",
                "raise_and_escalate": "Raise it past the adjuster's authority and send the file to a manager or the "
                                      "large-loss unit.",
            }),
            "reserve_adequacy": score("How well does the current reserve cover what this claim is likely to cost from here?", [
                "Badly short: total incurred is under 80% of the latest valuation plus the effect of any later "
                "development.",
                "Short: total incurred is 80% to 95% of that figure.",
                "Sufficient: total incurred is 95% to 110% of that figure.",
                "Excessive: total incurred is more than 110% of that figure.",
            ]),
            "cost_uncertainty": score("How uncertain is the final cost of this claim at this stage?", [
                "Settled: repairs are complete or treatment has ended, so what remains is known.",
                "Narrow: no open demand, surgery, dispute, extension or representation is on file.",
                "Wide: an open demand, surgery, public adjuster estimate, extended repairs or housing, or an "
                "attorney letter.",
                "Open-ended: suit has been filed or a reservation of rights is out on coverage.",
            ]),
            "valuation_is_current": noul(
                "The file holds a current valuation that supports the reserve now carried.",
                yes="A recent evaluation accounts for every material development on file.",
                no="The last evaluation predates new information, or key parts of the loss have not been valued."),
            "excess_letter_warranted": noul(
                "The policyholder should be told in writing that the claim may exceed the policy limit."),
        }


# =========================================================================== schema 5: subrogation / recovery
SUBRO = {
    # scenario: (what happened, typical paid, kind)
    "auto_rear_end": ("Insured's car was struck from behind while stopped at a light", 9000, "auto"),
    "auto_intersection": ("Intersection collision; each driver says the other had the red light", 12000, "auto"),
    "appliance_failure": ("Water damage after the {appliance} failed", 26000, "product"),
    "contractor_fire": ("Fire that started where a plumbing contractor had been soldering inside a wall", 90000, "contractor"),
    "neighbor_unit_leak": ("Water came through the ceiling from the unit above after its toilet supply line failed", 14000, "neighbor"),
    "municipal_main_break": ("A city water main broke and flooded the insured's basement", 22000, "public"),
}
APPLIANCES = ["dishwasher", "refrigerator ice-maker line", "water heater", "washing machine"]


class SubrogationRecovery(Schema):
    name = "subrogation_recovery"
    description = ("A paid claim referred to the recovery unit: whether and how to pursue the responsible party, "
                   "how much could come back and how soon the file must move.")
    surfaces = {"json": 0.45, "referral_memo": 0.3, "file_notes": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        scen = pick(rng, list(SUBRO))
        kind = SUBRO[scen][2]
        paid = None                                    # drawn from the FINAL scenario after the regime block
        liability = pick_w(rng, {"clear": 3, "probable": 3, "contested": 2, "weak": 1})
        fault_pct = pick(rng, [0, 0, 0, 10, 20, 30, 50])
        rp_cover = pick_w(rng, {"known_carrier": 6, "uninsured": 1, "unknown": 2})
        rp_limit = None
        evidence = pick_w(rng, {"retained_by_carrier": 3, "held_by_third_party": 2, "discarded": 1, None: 1})
        days_since = pick(rng, [60, 150, 300, 500, 650])
        deadline_days = 730 if kind != "public" else pick(rng, [90, 180])
        waiver = None
        response = pick_w(rng, {"accepted_liability": 2, "accepted_partial": 2, "denied": 2, "no_response": 2,
                                "not_yet_notified": 2})
        twist = None

        if reg == "clear":
            if chance(rng, 0.55):
                twist = "easy_recovery"
                scen, kind = "auto_rear_end", "auto"
                paid = lognormal_amount(rng, 9000, 0.4, 2)
                liability, fault_pct, rp_cover, response, days_since = "clear", 0, "known_carrier", pick(rng, ["accepted_liability", "no_response"]), rng.randint(40, 200)
            else:
                twist = "dead_end"
                liability, evidence = "weak", "discarded"
                rp_cover = pick(rng, ["unknown", "uninsured"])
                paid = lognormal_amount(rng, 5000, 0.4, 2)
                response = "denied"
        elif reg == "borderline":
            twist = pick(rng, ["limits_near_paid", "comparative_fault", "deadline_months", "small_vs_expert"])
            if twist == "limits_near_paid":
                scen, kind, rp_cover = pick(rng, ["auto_rear_end", "auto_intersection"]), "auto", "known_carrier"
                rp_limit = pick(rng, [25000, 50000])
                paid = near(rng, rp_limit, rp_limit * 0.25, lo=8000)
            elif twist == "comparative_fault":
                scen, kind, liability, fault_pct = "auto_intersection", "auto", "contested", pick(rng, [20, 30, 40, 50])
            elif twist == "deadline_months":
                days_since = None                                  # set from the deadline below
            else:
                scen = pick(rng, ["appliance_failure", "contractor_fire"])
                kind = SUBRO[scen][2]
                paid = lognormal_amount(rng, 9000, 0.3, 2)
        else:
            twist = pick(rng, ["clear_liability_lost_evidence", "clear_liability_uninsured", "accepted_but_deadline",
                               "waiver_unknown", "public_notice_near"])
            if twist == "clear_liability_lost_evidence":
                scen = pick(rng, ["appliance_failure", "contractor_fire"])
                kind, liability, evidence = SUBRO[scen][2], "probable", "discarded"
            elif twist == "clear_liability_uninsured":
                scen, kind, liability, rp_cover = "auto_rear_end", "auto", "clear", "uninsured"
            elif twist == "accepted_but_deadline":
                response, days_since = "accepted_partial", deadline_days - rng.randint(10, 40)
            elif twist == "waiver_unknown":
                scen = pick(rng, ["contractor_fire", "neighbor_unit_leak"])
                kind, liability, waiver = SUBRO[scen][2], "clear", "unknown"
            else:
                scen, kind, liability = "municipal_main_break", "public", "probable"

        # ---- derived after the regime block (deadlines, limits and costs follow the final scenario)
        if kind == "public":
            deadline_days = pick(rng, [90, 180]) if deadline_days == 730 else deadline_days
            rp_cover = "public_entity"
            if days_since is None or days_since > deadline_days or twist == "public_notice_near":
                days_since = deadline_days - rng.randint(5, 40)
        else:
            deadline_days = 730 if kind != "product" else pick(rng, [730, 1095])
        if days_since is None:
            days_since = deadline_days - near(rng, 90, 50, lo=15, integer=True)
        days_since = min(days_since, deadline_days - 3)
        if kind == "auto":
            evidence = "not_needed" if evidence in ("retained_by_carrier", None) else "photos_only"
            if rp_limit is None and rp_cover == "known_carrier":
                rp_limit = pick(rng, [25000, 50000, 100000, 250000])
            if scen == "auto_rear_end" and twist not in ("comparative_fault",):
                fault_pct = 0 if liability in ("clear", "probable") else fault_pct
        elif kind == "contractor":
            rp_cover = "known_carrier" if rp_cover == "unknown" and chance(rng, 0.5) else rp_cover
            rp_limit = 1000000 if rp_cover == "known_carrier" else None
            fault_pct = 0
            if waiver is None:
                waiver = pick_w(rng, {"none_found": 3, "contract_waiver": 1, "unknown": 1})
        elif kind == "neighbor":
            rp_limit = 300000 if rp_cover == "known_carrier" else None
            fault_pct = 0
            if waiver is None:
                waiver = pick_w(rng, {"none_found": 3, "association_bylaw_waiver": 1, "unknown": 1})
        elif kind == "product":
            rp_cover = "manufacturer" if evidence != "discarded" else pick(rng, ["manufacturer", "unknown"])
            fault_pct = 0
        else:
            fault_pct = 0
        if twist == "comparative_fault":
            liability = "contested"
        expert = {"auto": 0, "product": round(rng.uniform(4000, 12000), -2), "contractor": round(rng.uniform(6000, 18000), -2),
                  "neighbor": round(rng.uniform(0, 1500), -2), "public": round(rng.uniform(0, 2500), -2)}[kind]
        expert = int(expert)
        arb = kind == "auto" and rp_cover == "known_carrier" and chance(rng, 0.8)
        if rp_cover in ("uninsured", "unknown") and response in ("accepted_liability", "accepted_partial"):
            response = pick(rng, ["no_response", "denied", "not_yet_notified"])
        offer_pct = (100 - fault_pct) if fault_pct else pick(rng, [60, 70, 75, 80])
        if paid is None:
            paid = lognormal_amount(rng, SUBRO[scen][1], 0.5, 2)
        paid = _num(round(paid, 2))
        as_of = _day(rng, date(2025, 8, 1), date(2026, 9, 25))
        loss_date = as_of - timedelta(days=days_since)
        return {
            "regime": reg, "twist": twist, "scenario": scen, "kind": kind, "appliance": pick(rng, APPLIANCES),
            "paid": paid, "insured_deductible": pick(rng, [500, 1000, 1000, 2500]), "liability": liability,
            "insured_fault_pct": fault_pct, "offer_pct": offer_pct, "responsible_party_cover": rp_cover, "responsible_party_limit": rp_limit,
            "evidence": evidence, "deadline_kind": "notice of claim" if kind == "public" else "statute of limitations",
            "deadline_days": deadline_days, "days_since_loss": days_since, "days_left": deadline_days - days_since,
            "waiver": waiver, "expert_cost_estimate": expert, "arbitration_eligible": arb,
            "rp_response": response if kind != "product" or evidence != "discarded" else "not_yet_notified",
            "as_of": as_of, "loss_date": loss_date,
            "claim_no": _claim_no(rng, "RC", loss_date), "insured": person(rng), "handler": _handler(rng),
            "rp_carrier": org(rng, "insurer"), "rp_name": person(rng) if kind in ("auto", "neighbor") else None,
            "contractor": _vendor(rng, ["Plumbing", "Mechanical", "Heating & Plumbing"]),
            "maker": _vendor(rng, ["Appliances", "Home Products"]),
        }

    def _party(self, f):
        k = f["kind"]
        if k == "auto":
            return f"other driver {f['rp_name']}"
        if k == "product":
            return f"the {f['appliance']} maker, {f['maker']}"
        if k == "contractor":
            return f"the contractor, {f['contractor']}"
        if k == "neighbor":
            return f"the upstairs owner, {f['rp_name']}"
        return "the city water department"

    def _cover_text(self, f, m):
        c = f["responsible_party_cover"]
        if c == "known_carrier":
            return f"insured with {f['rp_carrier']}" + (f", limit {m(f['responsible_party_limit'])}" if f["responsible_party_limit"] else "")
        return {"uninsured": "no insurance found", "unknown": "insurance not yet identified",
                "public_entity": "public body (self-insured)", "manufacturer": "manufacturer, insurer not yet known"}[c]

    def _evidence(self, f):
        e, k = f["evidence"], f["kind"]
        if e is None:
            return None
        if k == "auto":
            return {"not_needed": "photos and police report on file",
                    "photos_only": "photos only; vehicle already repaired"}.get(e)
        return {
            "product": {"retained_by_carrier": f"failed {f['appliance']} part retained in our evidence store",
                        "held_by_third_party": "failed part held by the restoration contractor",
                        "discarded": "failed part thrown away during clean-up"},
            "contractor": {"retained_by_carrier": "burnt wall section and the contractor's torch retained",
                           "held_by_third_party": "fire department holds the scene samples",
                           "discarded": "debris cleared before our origin expert could attend"},
            "neighbor": {"retained_by_carrier": "failed supply line retained in our evidence store",
                         "held_by_third_party": "the upstairs owner's plumber kept the failed line",
                         "discarded": "failed line thrown away by the upstairs owner's plumber"},
            "public": {"retained_by_carrier": "photos of the break and the city's repair crew on file",
                       "held_by_third_party": "the city holds the removed pipe section",
                       "discarded": "no photos of the break were taken"},
        }[k][e]

    def render(self, f, rng, surface):
        what = SUBRO[f["scenario"]][0].format(appliance=f["appliance"])
        resp = {"accepted_liability": "accepted liability in full",
                "accepted_partial": f"offered to accept {f['offer_pct']}% liability",
                "denied": "denied liability", "no_response": "no response to the notice letter",
                "not_yet_notified": "not yet put on notice"}[f["rp_response"]]
        ev = self._evidence(f)
        if surface == "json":
            return dumps(prune({
                "recovery": {"file_number": f["claim_no"], "referred_on": _iso(f["as_of"]), "loss": what,
                             "date_of_loss": _iso(f["loss_date"]), "handler": f["handler"]},
                "payments": {"paid_by_carrier": f["paid"], "insured_deductible": f["insured_deductible"],
                             "estimated_expert_cost": f["expert_cost_estimate"] or None},
                "responsible_party": {"who": self._party(f), "insurance": f["responsible_party_cover"],
                                      "carrier": f["rp_carrier"] if f["responsible_party_cover"] == "known_carrier" else None,
                                      "liability_limit": f["responsible_party_limit"], "response": f["rp_response"],
                                      "inter_insurer_arbitration_member": f["arbitration_eligible"] if f["kind"] == "auto" else None},
                "liability": {"assessment": f["liability"], "insured_share_pct": f["insured_fault_pct"] if f["kind"] == "auto" else None,
                              "evidence": ev, "waiver_of_subrogation": f["waiver"]},
                "deadline": {"type": f["deadline_kind"],
                             "expires_on": _iso(f["loss_date"] + timedelta(days=f["deadline_days"]))},
            }))
        m, d = _mfmt(rng), _dfmt(rng)
        expires = f["loss_date"] + timedelta(days=f["deadline_days"])
        if surface == "referral_memo":
            s = [f"Referring {f['claim_no']} ({f['insured']}) to recovery.", f"{what} on {d(f['loss_date'])}.",
                 f"We have paid {m(f['paid'])}; the insured's deductible was {m(f['insured_deductible'])}.",
                 f"Responsible party: {self._party(f)}, {self._cover_text(f, m)}; {resp}."]
            s.append({"clear": "Liability looks clear.", "probable": "Liability looks likely but will be argued.",
                      "contested": "Liability is disputed.", "weak": "Our liability case is thin."}[f["liability"]])
            if f["kind"] == "auto" and f["insured_fault_pct"]:
                s.append(f"Our own driver may carry about {f['insured_fault_pct']}% of the fault.")
            if ev:
                s.append(f"Evidence: {ev}.")
            if f["waiver"] and f["waiver"] != "none_found":
                s.append({"contract_waiver": "The repair contract has a waiver-of-subrogation clause.",
                          "association_bylaw_waiver": "The association bylaws may waive claims between unit owners.",
                          "unknown": "We have not yet seen the contract / bylaws to check for a waiver."}[f["waiver"]])
            if f["expert_cost_estimate"]:
                s.append(f"An expert would cost roughly {m(f['expert_cost_estimate'])}.")
            if f["arbitration_eligible"]:
                s.append("Both carriers are members of the inter-insurer arbitration forum.")
            s.append(f"The {f['deadline_kind']} runs out on {d(expires)}.")
            return "\n".join([f"MEMO -- subrogation referral", f"From: {f['handler']}", f"Date: {d(f['as_of'])}", "",
                              noisy(join_sentences(s), rng, 0.1)])
        # file_notes
        t = datetime(f["as_of"].year, f["as_of"].month, f["as_of"].day, rng.randint(9, 15), rng.randrange(60))
        rows = kv_lines([("Paid", m(f["paid"])), ("Deductible", m(f["insured_deductible"])),
                         ("Responsible party", self._party(f)), ("Their cover", self._cover_text(f, m)),
                         ("Response", resp), ("Liability", f["liability"]),
                         ("Insured fault", f"{f['insured_fault_pct']}%" if f["kind"] == "auto" else None),
                         ("Evidence", ev), ("Waiver", _h(f["waiver"])),
                         ("Expert estimate", m(f["expert_cost_estimate"]) if f["expert_cost_estimate"] else None),
                         ("Arbitration member", _yn(f["arbitration_eligible"]) if f["kind"] == "auto" else None),
                         (f"{f['deadline_kind'].capitalize()} expires", d(expires))], rng, style=pick(rng, ["aligned", "dash"]))
        opener = (datetime(f["loss_date"].year, f["loss_date"].month, f["loss_date"].day, 10, 0)
                  + timedelta(days=rng.randint(1, 4)))
        return "\n".join([f"RECOVERY FILE {f['claim_no']}",
                          _diary([(opener, "OPEN", f"{what}. Insured {f['insured']}."),
                                  (t, "REFERRAL", f"Referred to recovery by {f['handler']}.")], rng), "", rows])

    def questions(self):
        return {
            "recovery_prospect": score("How likely is the carrier to recover a meaningful part of what it paid?", [
                "Poor: little realistic chance of getting money back.",
                "Fair: a recovery is possible but depends on facts still in dispute.",
                "Good: a recovery is likely, though perhaps not in full.",
                "Strong: liability and a source of payment are both clear.",
            ]),
            "deadline_pressure": score("How soon does the recovery file need action to protect the carrier's rights?", [
                "No pressure: more than a year before any limitation or notice deadline, and evidence is secure.",
                "This quarter: the deadline is months away but should be diarised now.",
                "Within weeks: a deadline falls inside about three months, or key evidence sits with someone else.",
                "Immediately: a deadline falls within a month, or the key evidence could be lost.",
            ]),
            "subrogation_path": choice("What should the recovery unit do with this file?", {
                "close_no_pursuit": "Close the recovery file; the chances or the amount do not justify pursuing it.",
                "send_demand": "Send a demand for the paid amount and the deductible to the responsible party or "
                               "their insurer.",
                "inter_insurer_arbitration": "File the claim with the inter-insurer arbitration forum against the "
                                             "other carrier.",
                "expert_exam_first": "Retain an expert and arrange a joint examination of the evidence before any "
                                     "demand.",
                "refer_to_counsel": "Refer the file to subrogation counsel to sue, because a deadline is close or "
                                    "the claim has been refused.",
            }),
            "recoverable_share": choice("If the recovery is pursued, how much of the paid amount could realistically come back?", {
                "full_amount": "Essentially all of it, plus the insured's deductible.",
                "reduced_by_shared_fault": "A reduced share, because the insured's own fault would be offset against it.",
                "capped_by_available_limits": "Only up to the responsible party's insurance limit or collectable "
                                              "assets, which fall short of the payment.",
                "little_or_none": "Little or nothing, because liability is weak, the party is unidentified, or a "
                                  "waiver or deadline bars the claim.",
            }),
            "recovery_worth_pursuing": noul(
                "Pursuing this recovery is worth what it will cost.",
                yes="The likely recovery comfortably exceeds the expert, legal and staff costs of pursuing it.",
                no="The expected recovery is small or doubtful relative to what pursuing it would cost."),
        }


# =========================================================================== schema 6: travel / baggage
DESTINATIONS = ["Lisbon", "Kyoto", "Reykjavik", "Cape Town", "Honolulu", "Dubrovnik", "Marrakesh", "Vancouver",
                "Cancun", "Hanoi", "Queenstown", "Seville"]
REASONS = {
    # reason: (claimant's words, plan category)
    "own_illness": ("I came down with pneumonia and my doctor told me not to fly", "listed"),
    "family_illness": ("my mother was admitted to hospital", "listed"),
    "storm_at_destination": ("a hurricane hit the destination and the resort closed", "listed_if_unforeseen"),
    "job_loss": ("I was laid off from my job", "listed"),
    "work_conflict": ("my manager cancelled my leave for a project deadline", "not_listed"),
    "changed_plans": ("we decided the trip was not a good idea after all", "not_listed"),
    "carrier_strike": ("the airline's pilots went on strike", "listed_if_unforeseen"),
}
BAG_ITEMS = [("clothing and shoes", 900), ("toiletries", 120), ("laptop", 1400), ("camera", 1100),
             ("gold necklace", 1300), ("suit", 600), ("hiking boots", 220), ("children's clothes", 350),
             ("prescription sunglasses", 380), ("gifts", 500)]
TRAVEL_DOCS = {
    "baggage_lost": ["airline property irregularity report", "airline's final settlement letter", "receipts for items",
                     "boarding pass and bag tag"],
    "baggage_delayed": ["airline property irregularity report", "bag delivery confirmation with time",
                        "receipts for essentials", "boarding pass"],
    "trip_cancellation": ["booking invoice", "cancellation confirmation", "supplier refund statement",
                          "proof of the reason (medical certificate, notice, etc.)"],
    "trip_interruption": ["booking invoice", "new return ticket receipt", "proof of the reason (medical certificate, notice, etc.)",
                          "supplier refund statement"],
}


class TravelClaim(Schema):
    name = "travel_claim"
    description = ("A travel-insurance claim for lost or delayed baggage or a cancelled or cut-short trip: "
                   "handling path, covered reason, documentation and fraud signals.")
    surfaces = {"json": 0.45, "online_claim_form": 0.3, "claimant_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        kind = pick_w(rng, {"baggage_lost": 3, "baggage_delayed": 2, "trip_cancellation": 3, "trip_interruption": 2})
        cur = pick_w(rng, {"USD": 5, "EUR": 2, "GBP": 2, "CAD": 1})
        reason = pick_w(rng, {"own_illness": 3, "family_illness": 2, "storm_at_destination": 2, "job_loss": 1,
                              "work_conflict": 2, "changed_plans": 2, "carrier_strike": 1})
        preex = pick_w(rng, {"none": 5, "stable": 2, "treatment_changed_in_lookback": 1, None: 1})
        bought_after_deposit = pick(rng, [1, 5, 12, 20, 35, 60])
        event_known_at_purchase = False
        bought_days_before_event = pick(rng, [2, 9, 25, 60, 120])
        cfar = chance(rng, 0.12)
        delay_hours = pick(rng, [6, 11, 14, 22, 30, 50, 96])
        pir = pick_w(rng, {True: 4, False: 2, None: 1})
        docs_missing_n = pick_w(rng, {0: 4, 1: 3, 2: 2})
        prior = pick(rng, [0, 0, 0, 1, 2])
        refund = pick_w(rng, {"refund_received": 2, "credit_offered": 2, "pending": 2, "not_requested": 1,
                              "denied_in_writing": 1})
        valuables_in_checked = chance(rng, 0.3)
        doc_oddity = None
        twist = None

        if reg == "clear":
            if chance(rng, 0.55):
                twist = "clean"
                reason = pick(rng, ["own_illness", "family_illness"])
                preex, bought_after_deposit, bought_days_before_event = "none", pick(rng, [1, 5, 10]), pick(rng, [60, 120])
                docs_missing_n, prior, pir, delay_hours = 0, 0, True, pick(rng, [30, 50])
                refund, valuables_in_checked = pick(rng, ["refund_received", "denied_in_writing"]), False
            else:
                twist = "not_covered_or_suspicious"
                if kind in ("trip_cancellation", "trip_interruption"):
                    reason, cfar = pick(rng, ["changed_plans", "work_conflict"]), False
                else:
                    pir, docs_missing_n, prior, valuables_in_checked = False, 2, 2, True
                    doc_oddity = "receipts_dated_after_trip"
        elif reg == "borderline":
            twist = pick(rng, ["preex_lookback", "storm_named_near_purchase", "delay_near_trigger", "limit_edge"])
            if twist == "preex_lookback":
                reason, preex = "own_illness", pick(rng, ["stable", "treatment_changed_in_lookback"])
                bought_after_deposit = near(rng, 14, 7, lo=1, integer=True)
                kind = pick(rng, ["trip_cancellation", "trip_interruption"])
            elif twist == "storm_named_near_purchase":
                reason = "storm_at_destination"
                kind = pick(rng, ["trip_cancellation", "trip_interruption"])
                event_known_at_purchase = None                # set from the dates below
            elif twist == "delay_near_trigger":
                kind, delay_hours = "baggage_delayed", near(rng, 12, 4, lo=4, integer=True)
            else:
                kind = pick(rng, ["baggage_lost", "trip_cancellation"])
                docs_missing_n = 1
        else:
            twist = pick(rng, ["documented_but_bought_late", "listed_reason_no_proof", "high_value_bag_no_receipts",
                               "refund_unanswered", "odd_medical_note"])
            if twist == "documented_but_bought_late":
                kind, bought_days_before_event, docs_missing_n = pick(rng, ["trip_cancellation", "trip_interruption"]), pick(rng, [1, 2, 3]), 0
                reason = pick(rng, ["own_illness", "family_illness", "storm_at_destination"])
            elif twist == "listed_reason_no_proof":
                kind, reason, docs_missing_n = pick(rng, ["trip_cancellation", "trip_interruption"]), pick(rng, ["own_illness", "job_loss"]), 2
            elif twist == "high_value_bag_no_receipts":
                kind, pir, valuables_in_checked, docs_missing_n, prior = "baggage_lost", True, True, 1, 0
            elif twist == "refund_unanswered":
                kind, refund, docs_missing_n = pick(rng, ["trip_cancellation", "trip_interruption"]), "pending", 0
            else:
                kind, reason, doc_oddity = pick(rng, ["trip_cancellation", "trip_interruption"]), "own_illness", "medical_note_dated_after"

        # ---- derived after the regime block
        baggage = kind.startswith("baggage")
        if twist == "storm_named_near_purchase":
            named_offset = near(rng, 0, 3, integer=True)     # >0: storm named before the plan was bought
            event_known_at_purchase = named_offset > 0
        else:
            named_offset = None
        if reason not in ("storm_at_destination", "carrier_strike"):
            event_known_at_purchase = False if event_known_at_purchase is not None else None
        if baggage:
            reason, preex, cfar = None, None, False
            # the airline answers first; the plan pays in excess of what the airline pays
            refund = (pick_w(rng, {"refund_received": 2, "pending": 3, "denied_in_writing": 1, "not_requested": 1})
                      if twist != "refund_unanswered" else "pending")
        trip_cost = int(round(lognormal_amount(rng, 3800, 0.5), -1))
        depart = _day(rng, date(2025, 3, 1), date(2026, 6, 20))
        nights = pick(rng, [5, 7, 9, 12, 14])
        items, claimed = [], 0.0
        bag_limit, per_item_limit = pick(rng, [1500, 2000, 2500]), 500
        if kind == "baggage_lost":
            n = pick(rng, [3, 4, 5, 6])
            chosen = rng.sample(BAG_ITEMS, n)
            if valuables_in_checked and not any(x[0] in ("laptop", "camera", "gold necklace") for x in chosen):
                chosen[0] = pick(rng, [x for x in BAG_ITEMS if x[0] in ("laptop", "camera", "gold necklace")])
            for desc, val in chosen:
                items.append({"item": desc, "claimed": int(round(val * rng.uniform(0.6, 1.4), -1)),
                              "receipt": chance(rng, 0.2 if twist == "high_value_bag_no_receipts" else 0.6)})
            claimed = sum(i["claimed"] for i in items)
        elif kind == "baggage_delayed":
            claimed = round(lognormal_amount(rng, 260, 0.5), 2)
        elif kind == "trip_cancellation":
            claimed = round(trip_cost * rng.uniform(0.6, 1.0), 2)
        else:
            claimed = round(trip_cost * rng.uniform(0.3, 0.6) + lognormal_amount(rng, 900, 0.4), 2)
        claimed = _num(round(claimed, 2))
        purchase = depart - timedelta(days=rng.randint(20, 150))
        deposit = purchase - timedelta(days=bought_after_deposit)
        event = (purchase + timedelta(days=bought_days_before_event)) if kind == "trip_cancellation" else \
                (depart + timedelta(days=rng.randint(1, max(1, nights - 3)))) if kind == "trip_interruption" else depart
        if kind == "trip_cancellation" and event >= depart:
            event = depart - timedelta(days=rng.randint(1, 5))
            purchase = min(purchase, event - timedelta(days=bought_days_before_event))
            deposit = purchase - timedelta(days=bought_after_deposit)
        if kind == "trip_interruption" and twist == "documented_but_bought_late":
            purchase = event - timedelta(days=bought_days_before_event)   # plan bought mid-trip
        named_on = (purchase - timedelta(days=named_offset)) if named_offset is not None else None
        docs = TRAVEL_DOCS[kind]
        missing = rng.sample(docs, min(docs_missing_n, len(docs)))
        pir_doc = "airline property irregularity report"
        if pir_doc in docs and pir is not True and pir_doc not in missing:
            missing.append(pir_doc)                        # no report was made, so none can be on file
        answer_doc = "airline's final settlement letter" if kind == "baggage_lost" else "supplier refund statement"
        if answer_doc in docs and refund in ("pending", "not_requested") and answer_doc not in missing:
            missing.append(answer_doc)                     # nothing to show until the airline / supplier answers
        if answer_doc in docs and refund == "refund_received" and answer_doc in missing:
            missing.remove(answer_doc)
        preex_related = preex in ("stable", "treatment_changed_in_lookback") and reason == "own_illness" and chance(rng, 0.6)
        filed_on = (event + timedelta(days=rng.randint(3, 30)) if kind == "trip_cancellation"
                    else depart + timedelta(days=nights + pick(rng, [2, 6, 15, 40, 85])))
        refund_amount = _num(round(claimed * rng.uniform(0.2, 0.5), 2)) if refund in ("refund_received", "credit_offered") else None
        if kind == "baggage_delayed" and refund == "refund_received":
            refund_amount = pick(rng, [50, 75, 100])       # an airline's interim-expense payment
        return {
            "regime": reg, "twist": twist, "kind": kind, "currency": cur, "destination": pick(rng, DESTINATIONS),
            "reason": reason, "preexisting": preex, "cfar_rider": cfar, "deposit_date": deposit,
            "purchase_date": purchase, "days_purchase_after_deposit": (purchase - deposit).days,
            "preex_waiver_window_days": 14, "event_date": event, "storm_named_on": named_on,
            "event_known_at_purchase": event_known_at_purchase, "depart": depart, "return": depart + timedelta(days=nights),
            "trip_cost": trip_cost, "claimed": claimed, "items": items, "bag_limit": bag_limit,
            "per_item_limit": per_item_limit, "valuables_in_checked_bag": valuables_in_checked if kind == "baggage_lost" else None,
            "delay_hours": delay_hours if kind == "baggage_delayed" else None, "delay_trigger_hours": 12,
            "pir_filed": pir if baggage else None, "docs_on_file": [x for x in docs if x not in missing],
            "docs_missing": missing, "supplier_refund": refund, "refund_amount": refund_amount,
            "prior_claims_3y": prior, "doc_oddity": doc_oddity,
            "filed_on": filed_on, "preexisting_related": preex_related,
            "claim_no": _claim_no(rng, "TR", depart), "plan_no": f"TP-{rng.randrange(1000000, 9999999)}",
            "claimant": person(rng), "plan": "comprehensive" + (" + cancel-for-any-reason rider" if cfar else ""),
            "airline": f"{org(rng).split()[0]} Air",
        }

    WORDING = ("Plan wording: trip cover pays for cancellation or interruption caused by sickness or injury of you or a "
               "family member, or your job loss; a storm or strike at the destination only if it was not known when "
               "the plan was bought; a pre-existing condition whose treatment changed in the 60 days before purchase "
               "is excluded unless the plan was bought within 14 days of the first trip payment; other reasons only "
               "with the cancel-for-any-reason rider. Baggage cover needs an airline property irregularity report; "
               "the delay benefit starts after 12 hours.")

    def _story(self, f, rng, m, d):
        k = f["kind"]
        if k == "baggage_lost":
            s = [f"My checked bag never arrived in {f['destination']} and the airline has now declared it lost."]
            if f["pir_filed"] is False:
                s.append("I did not fill in a form at the airport because the desk was closed.")
        elif k == "baggage_delayed":
            s = [f"My bag arrived {f['delay_hours']} hours after I landed in {f['destination']}, so I had to buy clothes and toiletries."]
        else:
            why = REASONS[f["reason"]][0]
            if f["preexisting_related"]:
                why = pick(rng, ["my heart condition flared up and I was admitted for two days",
                                 "my blood pressure spiked and the doctor said I could not travel"])
            s = [f"We had to {'cancel our trip to' if k == 'trip_cancellation' else 'come home early from'} "
                 f"{f['destination']} because {why}."]
            if f["reason"] != "own_illness":
                pass
            elif f["preexisting"] == "stable":
                s.append("I have had a heart condition for years, but it has been stable.")
            elif f["preexisting"] == "treatment_changed_in_lookback":
                s.append("My doctor had changed my blood pressure medication a couple of months before we booked.")
            if f["storm_named_on"] is not None:
                s.append(f"The storm was named on {d(f['storm_named_on'])}.")
        if f["doc_oddity"] == "medical_note_dated_after":
            s.append(f"The doctor's note is dated {d(f['event_date'] + timedelta(days=rng.randint(9, 20)))}.")
        elif f["doc_oddity"] == "receipts_dated_after_trip":
            s.append("Some receipts are dated after I got home because I replaced things later.")
        return s

    def render(self, f, rng, surface):
        cur = f["currency"]
        m, d = _mfmt(rng, cur), _dfmt(rng)
        story = self._story(f, rng, m, d)
        if surface == "json":
            return dumps(prune({
                "claim": {"claim_number": f["claim_no"], "type": f["kind"], "currency": cur,
                          "amount_claimed": f["claimed"], "claimant_statement": noisy(" ".join(story), rng, 0.15),
                          "date_filed": _iso(f["filed_on"])},
                "plan": {"plan_number": f["plan_no"], "plan": f["plan"], "purchased_on": _iso(f["purchase_date"]),
                         "first_trip_deposit_on": _iso(f["deposit_date"]),
                         "preexisting_waiver_if_bought_within_days": f["preex_waiver_window_days"] if not f["kind"].startswith("baggage") else None,
                         "insured_trip_cost": f["trip_cost"] if not f["kind"].startswith("baggage") else None,
                         "baggage_limit": f["bag_limit"] if f["kind"] == "baggage_lost" else None,
                         "per_item_limit": f["per_item_limit"] if f["kind"] == "baggage_lost" else None,
                         "delay_benefit_after_hours": f["delay_trigger_hours"] if f["kind"] == "baggage_delayed" else None},
                "trip": {"destination": f["destination"], "departure": _iso(f["depart"]), "return": _iso(f["return"]),
                         "event_date": _iso(f["event_date"]) if not f["kind"].startswith("baggage") else None,
                         "carrier": f["airline"]},
                "evidence": {"documents_received": f["docs_on_file"], "documents_outstanding": f["docs_missing"] or None,
                             "airline_report_filed": f["pir_filed"],
                             "items": [{"item": i["item"], "claimed": i["claimed"], "receipt": i["receipt"]} for i in f["items"]] or None,
                             **({"airline_compensation": {"refund_received": "paid"}.get(f["supplier_refund"], f["supplier_refund"]),
                                 "airline_paid_amount": f["refund_amount"]} if f["kind"].startswith("baggage") else
                                {"supplier_refund": f["supplier_refund"], "supplier_refund_amount": f["refund_amount"]})},
                "history": {"prior_claims_3y": f["prior_claims_3y"],
                            "preexisting_condition_declared": f["preexisting"] if f["preexisting"] != "none" else None},
                "plan_wording": self.WORDING,
            }))
        if surface == "online_claim_form":
            rows = [("Claim reference", f["claim_no"]), ("Plan", f"{f['plan_no']} ({f['plan']})"),
                    ("Plan purchased", d(f["purchase_date"])), ("First trip payment", d(f["deposit_date"])),
                    ("Claim type", _h(f["kind"])), ("Destination", f["destination"]),
                    ("Travel dates", f"{d(f['depart'])} to {d(f['return'])}"), ("Airline", f["airline"]),
                    ("Amount claimed", m(f["claimed"]))]
            if not f["kind"].startswith("baggage"):
                rows += [("Trip cost insured", m(f["trip_cost"])), ("Date of event", d(f["event_date"])),
                         ("Refund from supplier", show(f["supplier_refund"], rng, _h)
                          + (f" ({m(f['refund_amount'])})" if f["refund_amount"] else "")),
                         ("Pre-existing condition", show(f["preexisting"], rng, _h))]
            else:
                rows += [("Airline report (PIR) filed", show(f["pir_filed"], rng, _yn)),
                         ("Airline compensation", {"refund_received": "paid"}.get(f["supplier_refund"], _h(f["supplier_refund"]))
                          + (f" ({m(f['refund_amount'])})" if f["refund_amount"] else ""))]
                if f["kind"] == "baggage_delayed":
                    rows.append(("Hours without bag", f["delay_hours"]))
            rows.append(("Previous claims (3 yrs)", f["prior_claims_3y"]))
            out = ["TRAVEL CLAIM FORM (online submission)", kv_lines(rows, rng, skip_none=False), ""]
            if f["items"]:
                out += ["Items:", table(["item", "value", "receipt"],
                                        [[i["item"], m(i["claimed"]), "yes" if i["receipt"] else "no"] for i in f["items"]], rng), ""]
            out += ["What happened?", noisy(" ".join(story), rng, 0.2), "",
                    "Uploaded: " + (", ".join(f["docs_on_file"]) or "nothing yet")]
            if f["docs_missing"]:
                out.append("System check -- not yet uploaded: " + ", ".join(f["docs_missing"]))
            out += ["", self.WORDING]
            return "\n".join(out)
        # claimant_email
        if f["kind"] == "trip_cancellation":
            trip_line = (f"We were due to fly with {f['airline']} on {d(f['depart'])} and come back on {d(f['return'])}; "
                         f"I bought the plan on {d(f['purchase_date'])}.")
        elif f["kind"] == "trip_interruption":
            trip_line = (f"We flew out on {d(f['depart'])} and were due back on {d(f['return'])}, but came home on "
                         f"{d(f['event_date'] + timedelta(days=1))}; the plan was bought on {d(f['purchase_date'])}.")
        else:
            trip_line = (f"We travelled {d(f['depart'])} to {d(f['return'])} with {f['airline']}; the plan was bought on "
                         f"{d(f['purchase_date'])}.")
        extra = []
        if f["items"]:
            extra.append("I've listed what was in the bag: " + "; ".join(f"{i['item']} {m(i['claimed'])}" for i in f["items"]) + ".")
            nr = sum(1 for i in f["items"] if not i["receipt"])
            if nr:
                extra.append(f"I don't have receipts for {nr} of them.")
        if f["kind"].startswith("baggage"):
            extra.append({"refund_received": f"The airline paid me {m(f['refund_amount'])}." if f["refund_amount"] else "",
                          "pending": "The airline has not settled with me yet.",
                          "denied_in_writing": "The airline refused to pay anything; their letter is attached.",
                          "not_requested": "I haven't claimed from the airline."}.get(f["supplier_refund"], ""))
        elif f["supplier_refund"] == "credit_offered":
            extra.append(f"The tour company only offered a {m(f['refund_amount'])} credit for a future trip.")
        elif f["supplier_refund"] == "refund_received":
            extra.append(f"The hotel refunded {m(f['refund_amount'])}.")
        elif f["supplier_refund"] == "pending":
            extra.append("I've asked the airline and hotel for refunds but have not heard back.")
        elif f["supplier_refund"] == "denied_in_writing":
            extra.append("The operator refused any refund in writing, which I've attached.")
        elif f["supplier_refund"] == "not_requested":
            extra.append("I haven't asked the operator for anything as it was non-refundable.")
        if f["docs_missing"]:
            extra.append("I'll send the " + " and ".join(f["docs_missing"])
                         + (" when I have them." if len(f["docs_missing"]) > 1 else " when I have it."))
        if f["prior_claims_3y"]:
            extra.append(f"(I have claimed on my travel cover {f['prior_claims_3y']} time(s) before.)")
        return "\n".join([
            f"Subject: {f['plan_no']} -- claim for {m(f['claimed'])}",
            f"From: {f['claimant']}", "To: travel-claims@insurer.example",
            f"Date: {d(f['filed_on'])}", "",
            "Dear claims team,", "",
            noisy(join_sentences(story + [trip_line]), rng, 0.2),
            "", noisy(join_sentences(extra), rng, 0.15), "", "Kind regards,", f["claimant"], "",
            f"[claims system attachment -- {self.WORDING}]"])

    def questions(self):
        return {
            "claim_handling": choice("How should the insurer handle this travel claim?", {
                "pay_straight_through": "Approve and pay automatically; the claim is covered, documented and "
                                        "unremarkable.",
                "examiner_adjustment": "Have an examiner review and adjust the amount for limits, depreciation or "
                                       "refunds before paying.",
                "pend_for_documents": "Pend the claim until missing documents, such as a medical certificate, an "
                                      "airline report or receipts, arrive.",
                "refer_to_investigations": "Refer the claim to the investigations unit because its pattern "
                                           "suggests misrepresentation.",
                "decline_not_covered": "Decline it because the reason for the loss falls outside the plan's cover.",
            }),
            "documentation_sufficiency": score("How completely do the documents on file support the amount claimed?", [
                "Insufficient: the key documents for this type of claim are missing.",
                "Partial: some support is on file, but a required document is still outstanding.",
                "Adequate: the required documents are in, with small gaps in item-level proof.",
                "Complete: every required document and proof of value is on file.",
            ]),
            "fraud_signal_strength": score("How strong are the signs that this travel claim is false or exaggerated?", [
                "None: none of the usual signs (a plan bought days before the event or mid-trip, a document dated "
                "after the event, two or more earlier claims, valuables in a checked bag without receipts, no airline "
                "report for a lost bag).",
                "Weak: one of those signs.",
                "Moderate: two of them.",
                "Strong: three or more of them together.",
            ]),
            "covered_reason": noul(
                "The event behind this claim is a covered reason under the plan purchased.",
                yes="Under the plan wording shown, the cause is covered: a listed reason not caught by the "
                    "pre-existing-condition rule, an unforeseen storm or strike, any reason with the rider, or a "
                    "baggage loss with an airline report (and, for a delay, more than 12 hours).",
                no="The cause is not listed and there is no rider, the storm or strike was known at purchase, the "
                   "pre-existing-condition rule applies, or a baggage claim lacks the airline report or the 12 hours."),
            "await_supplier_response": noul(
                "The claim should wait until the airline or travel supplier has answered its own refund or "
                "compensation request."),
        }


WORKFLOW = Workflow(
    name="insurance_claims_triage",
    title="Insurance claims triage",
    description="Property and casualty claims at first notice of loss and shortly after: routing, coverage, fraud "
                "indicators, reserves, recovery and documentation.",
    domain="Insurance / P&C claims handling",
    schemas=[AutoCollisionFnol(), HomeWaterDamage(), TheftBurglaryClaim(), ReserveReview(), SubrogationRecovery(),
             TravelClaim()],
    notes="The insurer's own adjudication of a claim file; no replies to customers, no churn, no payment-card "
          "alerts, no 'is this alert real' questions. All names, carriers and ids are invented.",
)
