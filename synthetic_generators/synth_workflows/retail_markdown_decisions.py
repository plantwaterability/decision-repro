"""Retail markdown decisions: when to hold price, mark down, clear or exit stock, judged against written markdown,
margin and reference-pricing rules.

Five schemas: a weekly in-season markdown review, an end-of-season clearance exit, a 'was / now' reference-price
claim check, a short-dated fresh-food reduction at store level, and the readout of a markdown price test. The
artefacts are merchandising records (markdown dashboards, planner emails, clearance reports, price-history
extracts, store reduction sheets, test readouts), never a conversation with a shopper. Retailers, brands, stores,
partners and policies are invented. Every schema draws a `regime` first and lets it steer the factors, so most
cases sit near a written threshold, carry evidence that points two ways, or leave a figure out.

Each schema also has `intended(f)`: the designer's reading of the written rules for that case. It is never
rendered and never used as a label; it exists so the answer-spread check can confirm that every question's answer
moves with the sampled factors.
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
    lognormal_amount,
    maybe,
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

# =========================================================================== shared pools
BANNER_A = ["Kestrel", "Wrenfield", "Tillbrook", "Oakhaven", "Marlowe", "Bramblecote", "Hartwell", "Larkspur",
            "Fennimore", "Ravensworth", "Thistledown", "Brightwater", "Ashcombe", "Wexley", "Penrose", "Halden",
            "Quillon", "Sorrelby", "Elderbank", "Moorfield", "Calder & Rowe", "Linnet", "Dovecote", "Hollins",
            "Greystoke", "Amberley", "Fairlight", "Cresswell"]
BANNER_B = ["Home", "Outfitters", "Mercantile", "Living", "Goods", "Department Stores", "Supply Co.", "Trading",
            "Stores", "& Daughters"]
GROCER_B = ["Fresh", "Food Hall", "Grocers", "Market", "Local", "Foods", "Pantry", "Provisions"]
LINES = ["Northline", "Fieldcraft", "Loomwell", "Harrow & Pine", "Sunday Studio", "Ternfield", "Coastline Basics",
         "Brookhaven", "Alpenrose", "Hearthstone", "Driftline", "Meadowmere", "Stonefold", "Urban Fable",
         "Kindred Thread", "Pebble & Oak", "Lumen Home", "Trailwise", "Saltwick", "Greywell", "Copse & Co",
         "Marram", "Fellwalker", "Quaymark", "Wildhaven", "Ottery", "Bellcrest", "Rookery"]
COLOURS = ["charcoal", "sage", "navy", "oatmeal", "rust", "black", "ivory", "olive", "slate blue", "terracotta",
           "plum", "sand", "forest green", "dusty rose", "mustard", "stone", "teal", "burgundy", "heather grey",
           "cobalt", "ochre", "cream", "pine", "coral"]
# department -> (season label, items, ticket range)
DEPTS = {
    "outerwear": ("Autumn/Winter", ["parka", "rain shell", "quilted vest", "wool-blend overcoat", "softshell jacket",
                                    "puffer jacket", "fleece-lined anorak", "waxed field jacket"], (49, 289)),
    "knitwear": ("Autumn/Winter", ["merino crew sweater", "cable-knit cardigan", "cashmere-blend scarf",
                                   "rib beanie", "half-zip fleece", "roll-neck jumper"], (15, 149)),
    "cold_weather_footwear": ("Autumn/Winter", ["insulated snow boot", "leather chelsea boot", "waterproof hiker",
                                                "shearling slipper", "duck boot"], (39, 199)),
    "swimwear": ("Spring/Summer", ["one-piece swimsuit", "board shorts", "rash guard", "bikini set",
                                   "swim trunks", "cover-up dress"], (16, 89)),
    "outdoor_living": ("Spring/Summer", ["4-piece patio set", "solar string lights", "steel fire pit",
                                         "rope hammock", "glazed planter", "outdoor cushion set", "bistro table",
                                         "cantilever parasol"], (19, 899)),
    "summer_footwear": ("Spring/Summer", ["slide sandal", "canvas espadrille", "sport sandal", "boat shoe",
                                          "woven mule"], (14, 99)),
    "activewear": ("Spring/Summer", ["running tight", "seamless sports top", "training short",
                                     "quarter-zip pullover", "packable wind jacket"], (18, 99)),
    "garden": ("Spring/Summer", ["cordless hedge trimmer", "raised bed kit", "hose reel cart", "seed-starting kit",
                                 "compost tumbler"], (12, 249)),
    "holiday_decor": ("Holiday", ["7 ft pre-lit tree", "glass ornament set", "light-up reindeer",
                                  "wooden advent calendar", "door wreath", "tabletop village"], (9, 349)),
    "back_to_school": ("Back to School", ["laptop backpack", "insulated lunch bag", "binder set",
                                          "uniform polo 3-pack", "pencil case", "desk lamp"], (6, 89)),
    "seasonal_appliances": ("Summer", ["tower fan", "portable air conditioner", "dehumidifier",
                                       "ice-cream maker", "evaporative cooler"], (29, 499)),
    "heating": ("Winter", ["oil-filled radiator", "ceramic space heater", "heated throw", "electric blanket"],
                (29, 179)),
}
BULKY = {"outdoor_living", "seasonal_appliances", "garden", "heating", "holiday_decor"}


def _banner(rng):
    return f"{pick(rng, BANNER_A)} {pick(rng, BANNER_B)}"


def _grocer(rng):
    return f"{pick(rng, BANNER_A)} {pick(rng, GROCER_B)}"


def _product(rng, dept):
    item = pick(rng, DEPTS[dept][1])
    return f"{pick(rng, LINES)} {pick(rng, COLOURS)} {item}"


def _price(rng, lo, hi):
    """A retail ticket ending in .99 (or .00 for big-ticket items)."""
    v = rng.uniform(lo, hi)
    if v >= 200 and chance(rng, 0.5):
        return float(int(round(v, -1)))
    return round(int(v) + 0.99, 2)


def _usd(x, rng):
    return money(x, rng, style=pick(rng, ["sym", "sym", "code"]))


def _dept_label(d):
    return d.replace("_", " ")


# season -> (month, earliest day, latest day) of the first selling week
SEASON_START = {"Autumn/Winter": (8, 10, 31), "Spring/Summer": (2, 10, 28), "Holiday": (10, 15, 31),
                "Back to School": (6, 20, 30), "Summer": (4, 15, 30), "Winter": (10, 1, 25)}


def _season_date(rng, season, week):
    from datetime import date
    m, d0, d1 = SEASON_START[season]
    start = date(pick(rng, [2025, 2026]), m, rng.randint(d0, d1))
    return start + timedelta(days=7 * week + rng.randint(0, 3))


# =========================================================================== schema 1
VIEW_TEXT = {
    "push": ["We should not carry this into the last weeks at full ticket.",
             "Two competitors are already promoting similar styles.",
             "I would rather take the hit now than chase it with deeper cuts later.",
             "Stores keep asking for a price move on this one.",
             "Rate of sale has been soft since launch, so I don't expect it to recover on its own."],
    "wait": ["Rate of sale picked up after the window refresh, so there is an argument for waiting a week.",
             "Weather has been against us; one normal week could close part of the gap.",
             "The style gets an email feature next week, which may help without a price move.",
             "Not fully convinced it needs a markdown yet, but flagging it per the cadence.",
             "Size curve is broken in the small stores; a transfer might do more than a price cut."],
    "neutral": ["Standard cadence step per the markdown calendar.", "Submitting per the weekly review.",
                "Numbers attached; happy to discuss at the review.",
                "Following the department markdown calendar.", "Routine proposal, no special circumstances."],
}
CONTEXT_NOTES = ["A cold snap in the north lifted sales for two days last week.",
                 "An unusually warm weekend hit the south.", "Online returns on this style run above the department.",
                 "Fit complaints have come in on the smallest size.",
                 "A main competitor launched a similar style at a lower price.",
                 "It was featured in last week's email with little response.",
                 "Window displays were changed on Monday.", "Two large stores had the style out of stock for a week.",
                 "", "", ""]
PROMOS = ["Friends & Family 25% off", "Spring Savings Weekend", "Members' Week", "Labor Day Sale",
          "Anniversary Sale", "Back-to-School Weekend", "Midsummer Sale preview", "Loyalty Double Points Days",
          "Holiday Preview Event", "Clearance Weekend (online)"]


class InSeasonMarkdownReview(Schema):
    name = "in_season_markdown_review"
    description = ("A weekly in-season markdown review of one style: hold the price, take the next step, go deep, "
                   "rebalance stock between stores, or send it to clearance.")
    surfaces = {"json": 0.5, "markdown_dashboard": 0.2, "planner_email": 0.15, "review_sheet": 0.15}

    def sample_factors(self, rng):
        reg = regime(rng)
        dept = pick(rng, list(DEPTS))
        season, _items, (plo, phi) = DEPTS[dept]
        ticket = _price(rng, plo, phi)
        season_weeks = pick(rng, [10, 12, 13, 14, 16, 18])
        trigger = pick(rng, [6, 8, 10, 12])
        wos_mult = pick(rng, [1.5, 1.6, 1.75, 2.0])
        promo_block = pick(rng, [10, 14, 21])
        rebalance_pct = pick(rng, [35, 40, 45])
        floor = pick(rng, [34, 36, 38, 40, 42, 45])
        tiers = pick(rng, [(8000, 30000, 100000), (10000, 40000, 125000), (15000, 50000, 150000),
                           (5000, 25000, 80000)])
        if reg == "clear":
            kind = pick(rng, ["on_plan", "far_behind", "stranded"])
        elif reg == "borderline":
            kind = pick_w(rng, {"near_trigger": 3, "near_wos": 2})
        else:
            kind = pick(rng, ["promo_conflict", "imbalance", "late_surge", "receipts_due"])
        cap = season_weeks - 2
        trend = pick_w(rng, {"steady": 4, "slowing": 4, "accelerating": 2})
        imbalance = round(rng.uniform(8, rebalance_pct - 4)) if chance(rng, 0.85) else round(rng.uniform(rebalance_pct, 60))
        promo_days = pick(rng, [None, None, None, None, None, None, None, 6, 12, 18, 25, 33])
        on_order_share = pick(rng, [0, 0, 0, rng.uniform(0.03, 0.1)])
        if kind == "on_plan":
            wl, gap, ratio = rng.randint(3, cap), rng.uniform(-6, trigger * 0.4), rng.uniform(0.7, 1.15)
        elif kind == "far_behind":
            wl, gap, ratio = rng.randint(3, min(8, cap)), rng.uniform(trigger + 8, trigger + 22), rng.uniform(1.9, 3.2)
            trend = pick(rng, ["steady", "slowing"])
        elif kind == "stranded":
            wl, gap, ratio = pick(rng, [1, 2]), rng.uniform(trigger, trigger + 18), rng.uniform(1.9, 4.5)
        elif kind == "near_trigger":
            wl, gap, ratio = rng.randint(3, min(10, cap)), near(rng, trigger + 1, 3), rng.uniform(1.0, wos_mult * 1.05)
        elif kind == "near_wos":
            wl, gap, ratio = rng.randint(3, min(10, cap)), rng.uniform(0, trigger - 1), near(rng, wos_mult, 0.25)
        elif kind == "promo_conflict":
            wl, gap, ratio = rng.randint(4, min(10, cap)), rng.uniform(trigger, trigger + 8), rng.uniform(1.4, 2.1)
            promo_days = rng.randint(3, promo_block)
        elif kind == "imbalance":
            wl, gap, ratio = rng.randint(4, min(10, cap)), rng.uniform(trigger - 3, trigger + 6), rng.uniform(1.2, 1.9)
            imbalance = round(rng.uniform(rebalance_pct, rebalance_pct + 20))
        elif kind == "late_surge":
            wl, gap, ratio = rng.randint(3, min(9, cap)), rng.uniform(trigger, trigger + 8), rng.uniform(1.25, 1.9)
            trend = "accelerating"
        else:  # receipts_due
            wl, gap, ratio = rng.randint(3, min(9, cap)), rng.uniform(-2, trigger - 2), rng.uniform(1.6, 2.4)
            on_order_share = rng.uniform(0.15, 0.35)
        week = season_weeks - wl
        plan_st = max(8, min(85, round(82 * week / season_weeks + rng.uniform(-3, 3))))
        actual_st = max(2, min(95, round(plan_st - gap)))
        week_frac = week / season_weeks
        current_md = 0 if week_frac < 0.35 else pick(rng, [0, 0, 20, 25, 30])
        if current_md == 0:
            proposed = pick(rng, [20, 25, 30, 40]) if kind != "far_behind" else pick(rng, [30, 40, 50])
        else:
            proposed = min(60, current_md + pick(rng, [10, 15, 20]))
        # the markdown value lands in a sign-off band (sometimes right at a threshold); stock follows from it
        t1, t2, t3 = tiers
        band = pick_w(rng, {"planner": 25, "buyer": 30, "dmm": 27, "top": 18})
        lo, hi = {"planner": (t1 * 0.25, t1), "buyer": (t1, t2), "dmm": (t2, t3), "top": (t3, t3 * 2.5)}[band]
        md_target = near(rng, pick(rng, [t1, t2, t3]), 0.06 * t2) if chance(rng, 0.2) else rng.uniform(lo, hi)
        step_value = ticket * (proposed - current_md) / 100
        on_hand = max(60, int(round(md_target / step_value)))
        received = int(round(on_hand / max(0.05, 1 - actual_st / 100)))
        sold = received - on_hand
        buy = int(round(received / (1 - on_order_share)))
        on_order = buy - received
        shape = {"steady": [1.04, 0.98, 1.01, 0.97], "slowing": [1.35, 1.18, 1.0, 0.86],
                 "accelerating": [0.72, 0.84, 0.98, 1.22]}[trend]
        weekly_target = (on_hand + on_order) / max(0.5, ratio * wl)
        avg3 = sum(shape[1:]) / 3
        series = [max(1, int(round(weekly_target * m / avg3 * rng.uniform(0.96, 1.04)))) for m in shape]
        wos = round((on_hand + on_order) / (sum(series[1:]) / 3), 1)
        md_value = int(round(on_hand * step_value, -1))
        mband = pick_w(rng, {"below": 3, "near": 4, "above": 4})
        delta = {"below": rng.uniform(-6, -0.3), "near": rng.uniform(-1, 1.5), "above": rng.uniform(1.5, 9)}[mband]
        proj = maybe(rng, round(floor + delta, 1), 0.08)
        return {
            "regime": reg, "kind": kind, "banner": _banner(rng), "dept": dept, "season": season,
            "product": _product(rng, dept), "style_id": ident(rng, "STY", 6, "-"), "ticket": ticket,
            "review_date": _season_date(rng, season, week), "season_weeks": season_weeks, "week": week,
            "weeks_left": wl,
            "plan_st": plan_st, "actual_st": actual_st, "units_on_hand": on_hand, "units_on_order": on_order,
            "weekly_units_last4": series, "wos": wos, "trend": trend,
            "stores": max(6, min(480, int((on_hand + sold) / rng.uniform(12, 45)))),
            "imbalance_pct": imbalance, "promo_days": promo_days,
            "promo_name": pick(rng, PROMOS) if promo_days is not None else None,
            "current_md": current_md, "proposed_md": proposed, "md_value": md_value, "projected_mm": proj,
            "planner": person(rng), "planner_view": pick_w(rng, {"push": 3, "wait": 3, "neutral": 3}),
            "context_note": pick(rng, CONTEXT_NOTES),
            "policy_id": f"MD-{rng.randint(2, 9)}.{rng.randint(1, 12)}", "trigger_pts": trigger, "wos_mult": wos_mult,
            "promo_block_days": promo_block, "rebalance_pct": rebalance_pct, "margin_floor": floor,
            "signoff_tiers": list(tiers),
        }

    # ------------------------------------------------------------------ the designer's reading of the rules
    def intended(self, f):
        gap = f["plan_st"] - f["actual_st"]
        wl = f["weeks_left"]
        ratio = f["wos"] / max(wl, 1)
        promo_block = f["promo_days"] is not None and f["promo_days"] <= f["promo_block_days"]
        triggered = gap >= f["trigger_pts"] or ratio > f["wos_mult"]
        if wl <= 2 and ratio >= 1.8:
            action = "move_to_clearance"
        elif promo_block and ratio < 2.6:
            action = "hold_current_price"
        elif f["imbalance_pct"] >= f["rebalance_pct"] and wl >= 4 and ratio < 2.2:
            action = "rebalance_before_markdown"
        elif triggered and (gap >= f["trigger_pts"] + 8 or ratio >= 2.2) and wl <= 7:
            action = "jump_to_deep_markdown"
        elif triggered and not (f["trend"] == "accelerating" and gap < f["trigger_pts"] + 4 and ratio < 1.75):
            action = "take_next_markdown_step"
        else:
            action = "hold_current_price"
        t1, t2, t3 = f["signoff_tiers"]
        p = f["projected_mm"]
        v = f["md_value"]
        lvl = 0 if v <= t1 else 1 if v <= t2 else 2 if v <= t3 else 3
        if p is None or p < f["margin_floor"]:
            lvl = min(3, lvl + 1)
        signoff = ["planner_within_guidelines", "buyer_signoff", "dmm_signoff", "gmm_and_finance"][lvl]
        gap_lvl = 0 if gap <= 1 else 1 if gap < f["trigger_pts"] else 2 if gap < f["trigger_pts"] + 8 else 3
        if wl <= 2 and ratio >= 1.3:
            press = 3
        else:
            press = 0 if ratio < 1.15 else 1 if ratio < 1.5 else 2 if ratio < 2.2 else 3
        return {"markdown_action": action, "signoff_level": signoff, "sell_through_gap": gap_lvl,
                "season_exit_pressure": press,
                "margin_floor_respected": "yes" if (p is not None and p >= f["margin_floor"]) else "no"}

    # ------------------------------------------------------------------ rendering
    def _policy(self, f, rng):
        t1, t2, t3 = f["signoff_tiers"]
        return [
            f"Take the next markdown step when sell-through trails plan by {f['trigger_pts']} pts or more, or when "
            f"weeks of supply exceed {f['wos_mult']}x the weeks left to the floor set.",
            f"No new permanent markdown in the {f['promo_block_days']} days before a promotional event.",
            f"Rebalance between stores before marking down when {f['rebalance_pct']}% or more of the stock sits in "
            f"overstocked stores and 4+ weeks remain.",
            f"Maintained-margin floor for the department: {f['margin_floor']}%.",
            f"Sign-off by markdown value: planner up to {_usd(t1, rng)}, buyer up to {_usd(t2, rng)}, DMM up to "
            f"{_usd(t3, rng)}, GMM and finance above that. A projection below the margin floor, or no projection "
            f"at all, moves sign-off up one level.",
        ]

    def _comment(self, f, rng):
        s = [pick(rng, VIEW_TEXT[f["planner_view"]])]
        if f["context_note"]:
            s.append(f["context_note"])
        return noisy(join_sentences(s), rng, 0.12)

    def render(self, f, rng, surface):
        price_now = round(f["ticket"] * (1 - f["current_md"] / 100), 2)
        promo = (f"{f['promo_name']} starts in {f['promo_days']} days" if f["promo_days"] is not None
                 else "none scheduled")
        proj_txt = "not provided" if f["projected_mm"] is None else f"{f['projected_mm']}%"
        comment = self._comment(f, rng)
        if surface == "json":
            return dumps(prune({
                "style": {"id": f["style_id"], "description": f["product"], "department": f["dept"],
                          "banner": f["banner"], "ticket_price": f["ticket"], "current_price": price_now,
                          "current_markdown_pct": f["current_md"]},
                "calendar": {"season": f["season"], "week_of_season": f["week"], "season_weeks": f["season_weeks"],
                             "weeks_to_floor_set": f["weeks_left"], "review_date": f["review_date"].isoformat(),
                             "next_promo_event": ({"name": f["promo_name"], "starts_in_days": f["promo_days"]}
                                                  if f["promo_days"] is not None else None)},
                "performance": {"sell_through_to_date_pct": f["actual_st"],
                                "planned_sell_through_to_date_pct": f["plan_st"],
                                "units_on_hand": f["units_on_hand"], "units_on_order": f["units_on_order"],
                                "weekly_units_last_4": f["weekly_units_last4"],
                                "weeks_of_supply_incl_on_order": f["wos"], "stores_carrying": f["stores"],
                                "share_of_stock_in_overstocked_stores_pct": f["imbalance_pct"]},
                "proposal": {"submitted_by": f["planner"], "proposed_markdown_pct": f["proposed_md"],
                             "permanent_markdown_value_usd": f["md_value"],
                             "projected_season_margin_pct": f["projected_mm"], "planner_comment": comment},
                "policy": {"id": f["policy_id"], "rules": self._policy(f, rng)},
            }))
        if surface == "markdown_dashboard":
            rows = [["sell-through to date", f"{f['actual_st']}%"], ["plan to date", f"{f['plan_st']}%"],
                    ["units on hand", f"{f['units_on_hand']:,}"], ["units on order", f"{f['units_on_order']:,}"],
                    ["weekly units, last 4 wks", ", ".join(str(x) for x in f["weekly_units_last4"])],
                    ["WOS incl. on order", f["wos"]], ["weeks to floor set", f["weeks_left"]],
                    ["stock in overstocked stores", f"{f['imbalance_pct']}%"], ["stores carrying", f["stores"]],
                    ["next promo", promo]]
            head = (f"MARKDOWN REVIEW  {f['banner']}  wk {f['week']}/{f['season_weeks']} {f['season']}  "
                    f"({fmt_date(f['review_date'], rng)})\n"
                    f"{f['style_id']}  {f['product']}  [{_dept_label(f['dept'])}]\n"
                    f"Ticket {_usd(f['ticket'], rng)} | now {_usd(price_now, rng)}"
                    + (f" ({f['current_md']}% off)" if f["current_md"] else " (full price)"))
            prop = (f"Proposal from {f['planner']}: {f['proposed_md']}% off ticket. Markdown value "
                    f"{_usd(f['md_value'], rng)}. Projected season margin {proj_txt}.\nComment: {comment}")
            return "\n".join([head, "", table(["metric", "value"], rows, rng), "", prop, "",
                              f"Policy {f['policy_id']}:", bullets(self._policy(f, rng), rng)])
        if surface == "planner_email":
            s = [f"{f['product']} ({f['style_id']}) is at {f['actual_st']}% sell-through against a plan of "
                 f"{f['plan_st']}% in week {f['week']} of {f['season_weeks']}.",
                 f"We have {f['units_on_hand']:,} units on hand"
                 + (f" and {f['units_on_order']:,} still to arrive" if f["units_on_order"] else "")
                 + f", which is {f['wos']} weeks of supply at the last three weeks' rate"
                 f" (weekly units {', '.join(str(x) for x in f['weekly_units_last4'])}).",
                 f"The floor set is {f['weeks_left']} week{'s' if f['weeks_left'] != 1 else ''} away.",
                 f"{f['imbalance_pct']}% of the stock sits in stores running well above the chain's weeks of supply.",
                 (f"{f['promo_name']} starts in {f['promo_days']} days." if f["promo_days"] is not None
                  else "Nothing promotional is scheduled for the department."),
                 f"I'm proposing {f['proposed_md']}% off ticket (now {_usd(price_now, rng)}"
                 + (f", {f['current_md']}% off" if f["current_md"] else ", full price") + "), "
                 f"about {_usd(f['md_value'], rng)} of permanent markdown; projected season margin "
                 f"{proj_txt} against the {f['margin_floor']}% floor.", comment]
            return "\n".join([f"Subject: Markdown proposal {f['style_id']} -- wk {f['week']} review",
                              f"From: {f['planner']}", f"To: buying-{f['dept'].replace('_', '-')}@{f['banner'].split()[0].lower()}.example",
                              "", "Hi all,", "", noisy(join_sentences(s), rng, 0.1), "",
                              f"(Policy {f['policy_id']} applies; rules pasted below for reference.)",
                              bullets(self._policy(f, rng), rng), "", f["planner"].split()[0]])
        # review_sheet
        body = kv_lines([
            ("Style", f"{f['style_id']} {f['product']}"), ("Department", _dept_label(f["dept"])),
            ("Season week", f"{f['week']} of {f['season_weeks']}"), ("Weeks to floor set", f["weeks_left"]),
            ("Ticket / current", f"{_usd(f['ticket'], rng)} / {_usd(price_now, rng)}"),
            ("Sell-through actual vs plan", f"{f['actual_st']}% vs {f['plan_st']}%"),
            ("On hand / on order", f"{f['units_on_hand']:,} / {f['units_on_order']:,}"),
            ("Weeks of supply", f["wos"]), ("Unit trend (4 wks)", " > ".join(str(x) for x in f["weekly_units_last4"])),
            ("Overstocked-store share", f"{f['imbalance_pct']}%"), ("Promo calendar", promo),
            ("Proposed markdown", f"{f['proposed_md']}% off ticket"), ("Markdown value", _usd(f["md_value"], rng)),
            ("Projected margin", f["projected_mm"] and f"{f['projected_mm']}%"), ("Planner", f["planner"]),
        ], rng, skip_none=False)
        return (f"{f['banner'].upper()} -- MARKDOWN REVIEW SHEET\n{body}\nNotes: {comment}\n\n"
                f"{f['policy_id']} rules:\n" + "\n".join(f"  {i}. {r}" for i, r in enumerate(self._policy(f, rng), 1)))

    def questions(self):
        return {
            "markdown_action": choice("What should the markdown review do with this style this week?", {
                "hold_current_price": "Keep the current price and look at the style again at next week's review.",
                "take_next_markdown_step": "Take the next step on the markdown cadence at its normal depth.",
                "jump_to_deep_markdown": "Skip ahead to a deep markdown so the stock clears well before the floor set.",
                "rebalance_before_markdown": "Move stock from overstocked stores to stores that are selling it, "
                                             "and defer the price decision.",
                "move_to_clearance": "Send the style to final clearance now and stop treating it as current stock.",
            }),
            "signoff_level": choice("Who has to sign off the markdown exactly as the planner proposed it?", {
                "planner_within_guidelines": "Nobody beyond the planner; it sits inside the standing guidelines.",
                "buyer_signoff": "The department buyer.",
                "dmm_signoff": "The divisional merchandise manager.",
                "gmm_and_finance": "The general merchandise manager together with finance.",
            }),
            "sell_through_gap": score("How far is the style's sell-through behind where the plan expected it to be by now?", [
                "Not behind: on plan or ahead of it.",
                "Slightly behind: inside the range the markdown rule tolerates.",
                "Clearly behind: past the markdown trigger.",
                "Far behind: well past the trigger, with much of the buy still unsold.",
            ]),
            "season_exit_pressure": score("How much pressure is there to move the stock before the floor set?", [
                "Little: the stock on hand will sell through in the weeks remaining at the current rate.",
                "Some: a modest surplus that one more week of waiting would not make much worse.",
                "High: at the current rate a large part of the stock will still be there at the floor set.",
                "Acute: the floor set is almost here, or the stock would take more than twice the time left to sell.",
            ]),
            "margin_floor_respected": noul(
                "The planner's proposed markdown keeps the style's projected margin at or above the department floor.",
                yes="The projection shown is at or above the floor written in the policy.",
                no="The projection is below the floor, or no projection is given to check against it."),
        }


# =========================================================================== schema 2
class ClearanceExitDecision(Schema):
    name = "clearance_exit_decision"
    description = ("Leftover clearance stock at the end of a season: mark down again, sell to an off-price buyer, "
                   "pack and hold, return to the vendor, or donate.")
    surfaces = {"json": 0.5, "clearance_report": 0.2, "exit_memo": 0.15, "pretty_json": 0.15}

    CARRY = {"core_carryover": "core style, carries over next season unchanged",
             "colour_continues": "style continues next season, this colourway is dropped",
             "discontinued": "discontinued fashion style",
             "dated_edition": "dated edition (year-specific packaging or print)"}
    CONDITION = {"new_in_package": "new, in original packaging", "mixed_returns": "mix of new and customer-returned units",
                 "damaged_packaging": "much of it in damaged or shop-worn packaging"}

    def sample_factors(self, rng):
        reg = regime(rng)
        dept = pick(rng, list(DEPTS))
        _season, _items, (plo, phi) = DEPTS[dept]
        ticket = _price(rng, plo, phi)
        imu = rng.uniform(0.5, 0.68)
        cost = round(ticket * (1 - imu), 2)
        units = max(40, int(lognormal_amount(rng, 900, 0.9)))
        clearance_off = pick(rng, [50, 60, 65, 70])
        floor_off = pick(rng, [70, 75, 80])
        weekly_st = round(rng.uniform(1, 22), 1)
        weeks_window = pick(rng, [0, 1, 1, 2, 3, 4, 6])
        carry = pick_w(rng, {"core_carryover": 3, "colour_continues": 3, "discontinued": 3, "dated_edition": 2})
        cond = pick_w(rng, {"new_in_package": 8, "mixed_returns": 2, "damaged_packaging": 1})
        dc_util = rng.randint(58, 97)
        dc_cap = pick(rng, [88, 90, 92])
        storage = round(rng.uniform(1.2, 9) if dept in BULKY else rng.uniform(0.08, 0.9), 2)
        months_next = rng.randint(6, 10)
        bid = maybe(rng, rng.randint(5, 45), 0.25)
        restr = pick(rng, [None, 6, 12, 12, 18, 24])
        launch_m = rng.randint(3, 24)
        rtv = chance(rng, 0.35)
        rtv_cap = rng.randint(5, 15)          # % of season receipts returnable
        receipts = int(units * rng.uniform(4, 14))
        rtv_used_units = int(receipts * rtv_cap / 100 * rng.uniform(0, 0.9)) if rtv else 0
        rtv_fee = pick(rng, [10, 15, 20, 25])
        donation_ok = chance(rng, 0.8)
        if reg == "clear":
            k = pick(rng, ["rtv", "damaged", "carry", "bid"])
            if k == "rtv":
                rtv, rtv_used_units, cond, rtv_fee = True, 0, "new_in_package", pick(rng, [10, 15])
                rtv_cap = max(rtv_cap, int(units / receipts * 100) + 3)
            elif k == "damaged":
                cond, bid, rtv, weeks_window = "damaged_packaging", None, False, pick(rng, [0, 1])
                carry = pick(rng, ["discontinued", "dated_edition"])
            elif k == "carry":
                carry, cond, dc_util, rtv = "core_carryover", "new_in_package", rng.randint(60, dc_cap - 8), False
                storage = round(rng.uniform(0.05, 0.4), 2)
            else:
                bid, restr, rtv, carry = rng.randint(30, 48), None, False, pick(rng, ["discontinued", "dated_edition"])
        elif reg == "borderline":
            k = pick(rng, ["dc", "launch", "bid_vs_hold", "rtv_cap"])
            if k == "dc":
                carry, cond = pick(rng, ["core_carryover", "colour_continues"]), "new_in_package"
                dc_util = near(rng, dc_cap, 2, lo=60, hi=99, integer=True)
            elif k == "launch":
                restr = pick(rng, [6, 12, 18])
                launch_m = near(rng, restr, 1.5, lo=2, integer=True)
                bid = rng.randint(22, 42)
            elif k == "bid_vs_hold":
                carry, cond, dc_util = "colour_continues", "new_in_package", rng.randint(60, dc_cap - 3)
                bid, restr = rng.randint(18, 34), None
            else:
                rtv, cond = True, "new_in_package"
                rtv_used_units = int(receipts * rtv_cap / 100) - int(units * rng.uniform(0.6, 1.3))
                rtv_used_units = max(0, rtv_used_units)
        else:
            k = pick(rng, ["best_bid_blocked", "hold_but_full", "rtv_but_used", "markdown_no_time"])
            if k == "best_bid_blocked":
                bid, restr = rng.randint(32, 48), pick(rng, [12, 18])
                launch_m = rng.randint(3, restr - 1)
            elif k == "hold_but_full":
                carry, cond, dc_util = "core_carryover", "new_in_package", rng.randint(dc_cap + 1, 99)
            elif k == "rtv_but_used":
                rtv, cond = True, pick(rng, ["mixed_returns", "new_in_package"])
            else:
                weekly_st, weeks_window, clearance_off = round(rng.uniform(12, 24), 1), 0, pick(rng, [50, 60])
        return {
            "regime": reg, "banner": _banner(rng), "dept": dept, "product": _product(rng, dept),
            "style_id": ident(rng, "STY", 6, "-"), "ticket": ticket, "unit_cost": cost, "units_left": units,
            "clearance_off_pct": clearance_off, "deepest_allowed_off_pct": floor_off,
            "clearance_weekly_sell_through_pct": weekly_st, "weeks_in_clearance": rng.randint(1, 8),
            "weeks_to_floor_set": weeks_window, "carryover": carry, "condition": cond,
            "dc_utilisation_pct": dc_util, "dc_hold_cap_pct": dc_cap, "storage_cost_unit_month": storage,
            "months_to_next_season": months_next, "offprice_bid_pct_of_cost": bid,
            "offprice_buyer": f"{pick(rng, BANNER_A)} Off-Price Partners", "brand_resale_restriction_months": restr,
            "months_since_launch": launch_m, "rtv_agreement": rtv, "rtv_cap_pct_of_receipts": rtv_cap,
            "season_receipts_units": receipts, "rtv_units_already_returned": rtv_used_units,
            "rtv_restocking_fee_pct": rtv_fee, "donation_partner_accepts": donation_ok,
            "coordinator": person(rng), "vendor": f"{pick(rng, LINES)} Brands",
            "report_date": rand_date(rng),
        }

    def _recoveries(self, f):
        """Recovery of cost (fraction) for every route the rules permit."""
        cost, units = f["unit_cost"], f["units_left"]
        r = {}
        if f["clearance_off_pct"] < f["deepest_allowed_off_pct"] and f["weeks_to_floor_set"] > 0:
            nxt = min(f["deepest_allowed_off_pct"], f["clearance_off_pct"] + 10)
            price = f["ticket"] * (1 - nxt / 100)
            share = min(1.0, f["clearance_weekly_sell_through_pct"] / 100 * 1.6 * f["weeks_to_floor_set"])
            if f["condition"] == "damaged_packaging":
                share *= 0.5
            r["further_markdown"] = share * price / cost
        if f["offprice_bid_pct_of_cost"] is not None and self._offprice_ok(f):
            r["sell_to_offprice_buyer"] = f["offprice_bid_pct_of_cost"] / 100
        if self._hold_ok(f):
            mult = 0.7 if f["carryover"] == "core_carryover" else 0.5
            r["pack_and_hold"] = max(0.0, (f["ticket"] * mult * 0.8 - f["storage_cost_unit_month"]
                                           * f["months_to_next_season"]) / cost)
        if f["rtv_agreement"] and f["condition"] != "damaged_packaging":
            room = f["season_receipts_units"] * f["rtv_cap_pct_of_receipts"] / 100 - f["rtv_units_already_returned"]
            eligible = max(0.0, min(units, room))
            if f["condition"] == "mixed_returns":
                eligible *= 0.6
            if eligible > 0:
                r["return_to_vendor"] = (1 - f["rtv_restocking_fee_pct"] / 100) * eligible / units
        return r

    def _offprice_ok(self, f):
        restr = f["brand_resale_restriction_months"]
        return restr is None or f["months_since_launch"] >= restr

    def _hold_ok(self, f):
        return (f["carryover"] in ("core_carryover", "colour_continues") and f["condition"] == "new_in_package"
                and f["dc_utilisation_pct"] < f["dc_hold_cap_pct"])

    def intended(self, f):
        r = self._recoveries(f)
        best = max(r.values()) if r else 0.0
        route = max(r, key=r.get) if r and best >= 0.05 else "donate_or_recycle"
        rec = 0 if best < 0.1 else 1 if best < 0.33 else 2 if best < 0.66 else 3
        w = f["weeks_to_floor_set"]
        window = 0 if w >= 4 else 1 if w >= 2 else 2 if w == 1 else 3
        return {"exit_route": route, "recovery_outlook": rec, "selling_window_left": window,
                "pack_and_hold_viable": "yes" if self._hold_ok(f) else "no",
                "offprice_sale_permitted": "yes" if self._offprice_ok(f) else "no"}

    def _terms(self, f, rng):
        restr = f["brand_resale_restriction_months"]
        out = [(f"{f['vendor']} agreement: no sale to off-price resellers within {restr} months of launch."
                if restr else f"{f['vendor']} agreement: no restriction on off-price resale."),
               (f"Return-to-vendor allowance: up to {f['rtv_cap_pct_of_receipts']}% of season receipts "
                f"({f['season_receipts_units']:,} units), unopened only, {f['rtv_restocking_fee_pct']}% restocking fee; "
                f"{f['rtv_units_already_returned']:,} units already returned this season."
                if f["rtv_agreement"] else "No return-to-vendor allowance for this style."),
               f"Pack-and-hold: only sellable carryover stock, and only while the DC is under "
               f"{f['dc_hold_cap_pct']}% full.",
               f"Deepest clearance price the system allows: {f['deepest_allowed_off_pct']}% off ticket."]
        return out

    def render(self, f, rng, surface):
        carry, cond = self.CARRY[f["carryover"]], self.CONDITION[f["condition"]]
        bid = (f"{f['offprice_bid_pct_of_cost']}% of cost" if f["offprice_bid_pct_of_cost"] is not None
               else "no bid received")
        window = ("the floor set has already happened" if f["weeks_to_floor_set"] == 0
                  else f"{f['weeks_to_floor_set']} week(s) until the space is reset")
        if surface in ("json", "pretty_json"):
            obj = prune({
                "style": {"id": f["style_id"], "description": f["product"], "department": f["dept"],
                          "banner": f["banner"], "lifecycle": carry, "condition": cond,
                          "ticket_price": f["ticket"], "unit_cost": f["unit_cost"],
                          "months_since_launch": f["months_since_launch"]},
                "clearance": {"units_left": f["units_left"], "current_pct_off": f["clearance_off_pct"],
                              "weeks_in_clearance": f["weeks_in_clearance"],
                              "last_week_sell_through_pct": f["clearance_weekly_sell_through_pct"],
                              "weeks_to_floor_set": f["weeks_to_floor_set"]},
                "storage": {"dc_utilisation_pct": f["dc_utilisation_pct"],
                            "storage_cost_per_unit_month": f["storage_cost_unit_month"],
                            "months_to_next_season": f["months_to_next_season"]},
                "offers": {"offprice_buyer": f["offprice_buyer"], "offprice_bid": bid,
                           "donation_partner_accepts_category": f["donation_partner_accepts"]},
                "terms": self._terms(f, rng),
                "prepared_by": f["coordinator"], "report_date": f["report_date"].isoformat(),
            })
            return dumps(obj) if surface == "json" else dumps_pretty(obj, rng)
        if surface == "clearance_report":
            rows = [["units left", f"{f['units_left']:,}"], ["unit cost / ticket",
                                                              f"{_usd(f['unit_cost'], rng)} / {_usd(f['ticket'], rng)}"],
                    ["now", f"{f['clearance_off_pct']}% off"], ["sell-thru last wk", f"{f['clearance_weekly_sell_through_pct']}%"],
                    ["weeks to floor set", f["weeks_to_floor_set"]], ["lifecycle", carry], ["condition", cond],
                    ["DC full", f"{f['dc_utilisation_pct']}%"],
                    ["storage /unit/month", _usd(f["storage_cost_unit_month"], rng)],
                    ["months to next season", f["months_to_next_season"]], ["off-price bid", bid],
                    ["launched", f"{f['months_since_launch']} months ago"],
                    ["donation partner", "accepts category" if f["donation_partner_accepts"] else "does not take this category"]]
            return "\n".join([f"END-OF-SEASON CLEARANCE  {f['banner']}  {fmt_date(f['report_date'], rng)}",
                              f"{f['style_id']}  {f['product']}", "", table(["item", "value"], rows, rng), "",
                              "Terms on file:", bullets(self._terms(f, rng), rng)])
        s = [f"{f['units_left']:,} units of {f['product']} ({f['style_id']}) are left at {f['clearance_off_pct']}% off "
             f"after {f['weeks_in_clearance']} weeks in clearance; last week sold {f['clearance_weekly_sell_through_pct']}% "
             f"of what was on hand, and {window}.",
             f"It is a {carry}; stock is {cond}.",
             f"Cost is {_usd(f['unit_cost'], rng)} a unit against a {_usd(f['ticket'], rng)} ticket.",
             f"{f['offprice_buyer']}: {bid}." if f["offprice_bid_pct_of_cost"] is not None
             else f"{f['offprice_buyer']} has not bid.",
             f"The DC is {f['dc_utilisation_pct']}% full; storage runs about {_usd(f['storage_cost_unit_month'], rng)} "
             f"per unit per month and next season is {f['months_to_next_season']} months away.",
             f"Launched {f['months_since_launch']} months ago.",
             "Our donation partner takes this category." if f["donation_partner_accepts"]
             else "Our donation partner does not take this category."]
        return "\n".join([f"MEMO -- clearance exit, {f['style_id']}", f"From: {f['coordinator']}",
                          f"Date: {fmt_date(f['report_date'], rng)}", "", noisy(join_sentences(s), rng, 0.1), "",
                          "Relevant terms:", bullets(self._terms(f, rng), rng)])

    def questions(self):
        return {
            "exit_route": choice("Which exit route should the remaining units take?", {
                "further_markdown": "Take one more clearance markdown and keep selling them in stores and online.",
                "sell_to_offprice_buyer": "Sell the lot to the off-price buyer at the bid on the table.",
                "pack_and_hold": "Pull the units and store them to sell again next season.",
                "return_to_vendor": "Send the eligible units back to the vendor under the return allowance.",
                "donate_or_recycle": "Take the units off sale and donate or recycle them.",
            }),
            "recovery_outlook": score("How much of the units' cost can the best permitted route realistically get back?", [
                "Almost nothing: under a tenth of cost.",
                "Some: roughly a tenth to a third of cost.",
                "Much of it: a third to two thirds of cost.",
                "Most or all of it: two thirds of cost or more.",
            ]),
            "selling_window_left": score("How little selling time is left before the units must leave the sales floor?", [
                "Ample: a month or more before the space is reset.",
                "Limited: two or three weeks.",
                "Short: about a week.",
                "None: the floor set has already taken the space.",
            ]),
            "pack_and_hold_viable": noul(
                "Packing and holding these units for next season is allowed and sensible.",
                yes="The style carries over, the units are sellable as new, and the DC is under its holding cap.",
                no="The style or colour will not sell next season, the units are not sellable as new, or the DC "
                   "is at or over its holding cap."),
            "offprice_sale_permitted": noul(
                "The vendor agreement lets the retailer sell these units to an off-price reseller now."),
        }


# =========================================================================== schema 3
CHANNELS = ["in-store shelf sign", "weekly circular", "website product page", "email campaign", "end-cap sign",
            "app push offer", "printed catalogue"]
SPECIALS = {
    "stockout_days": "out of stock in the warehouse for part of the period while still ticketed at the 'was' price",
    "online_only_history": "the 'was' price was charged online only; stores carried it at a lower everyday price",
    "intro_price_switch": "launched at the 'was' price, then moved to a lower everyday price after a few weeks",
}


class ReferencePriceClaimCheck(Schema):
    name = "reference_price_claim_check"
    description = ("A 'was / now' markdown claim checked against the price history and the retailer's written "
                   "reference-pricing rule: approve, correct, drop the comparison, or refer.")
    surfaces = {"json": 0.5, "price_history_extract": 0.25, "compliance_ticket": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        dept = pick(rng, list(DEPTS))
        _s, _i, (plo, phi) = DEPTS[dept]
        was = _price(rng, max(plo, 12), phi)
        disc = rng.uniform(0.2, 0.55)
        now = max(1.99, round(int(was * (1 - disc)) + 0.99, 2))
        actual_save = (was - now) / was * 100
        lookback = pick(rng, [90, 90, 120])
        required = pick(rng, [28, 30, 45])
        recent_req = pick(rng, [30, 45, 60])
        coverage_req = pick(rng, [50, 60, 75])
        days_at_was = rng.randint(required, min(lookback, required * 3))
        last_at_was = rng.randint(0, recent_req - 1)
        coverage = 100
        special = None
        printed_kind = pick_w(rng, {"exact": 3, "rounded_down": 2, "rounded_up": 5})
        alt_price = None
        if reg == "clear":
            if chance(rng, 0.5):
                printed_kind = pick(rng, ["exact", "rounded_down"])
            else:
                days_at_was = rng.randint(0, max(1, required // 4))
                last_at_was = rng.randint(recent_req, lookback + 60)
        elif reg == "borderline":
            k = pick(rng, ["days", "recent", "coverage", "rounding"])
            if k == "days":
                days_at_was = near(rng, required, 4, lo=1, hi=lookback, integer=True)
            elif k == "recent":
                last_at_was = near(rng, recent_req, 5, lo=0, integer=True)
            elif k == "coverage":
                coverage = near(rng, coverage_req, 8, lo=10, hi=100, integer=True)
            else:
                printed_kind = pick(rng, ["rounded_up", "rounded_down"])
                if chance(rng, 0.5):
                    days_at_was = near(rng, required, 6, lo=1, hi=lookback, integer=True)
        else:
            special = pick(rng, list(SPECIALS))
            if special == "stockout_days":
                days_at_was = rng.randint(required, required + 15)
            elif special == "online_only_history":
                coverage = 0
            elif special == "intro_price_switch":
                days_at_was = rng.randint(max(3, required // 3), required - 1)
                last_at_was = rng.randint(recent_req - 10, lookback)
        days_at_was = max(0, min(days_at_was, lookback - last_at_was))
        stockout = rng.randint(max(1, days_at_was - required + 1), max(2, days_at_was - required + 12)) \
            if special == "stockout_days" else 0
        if days_at_was < required or last_at_was >= recent_req or coverage < coverage_req or special:
            if chance(rng, 0.55):
                alt_price = round(int((was + now) / 2 + rng.uniform(-3, 3)) + 0.99, 2)
                if alt_price <= now + 1:
                    alt_price = None
        actual_floor = int(actual_save)
        if printed_kind == "exact":
            printed = actual_floor
        elif printed_kind == "rounded_down":
            printed = max(5, 5 * (actual_floor // 5))
        else:
            printed = actual_floor + pick(rng, [1, 2, 3, 5])
        scope = pick_w(rng, {"all stores and online": 5, "all stores": 3, "online only": 2})
        if special == "online_only_history":
            scope = "all stores and online"
        return {
            "regime": reg, "banner": _banner(rng), "dept": dept, "product": _product(rng, dept),
            "sku": ident(rng, "SKU", 7, ""), "channel": pick(rng, CHANNELS), "claim_scope": scope,
            "was_price": was, "now_price": now, "printed_save_pct": printed, "lookback_days": lookback,
            "required_days": required, "recency_days": recent_req, "coverage_required_pct": coverage_req,
            "days_at_was": days_at_was, "days_since_last_at_was": last_at_was, "store_coverage_pct": coverage,
            "units_share_at_was_pct": min(90, round(days_at_was / lookback * 100 * rng.uniform(0.25, 1.1))),
            "stockout_days": stockout, "special": special, "supported_alt_price": alt_price,
            "alt_price_days": rng.randint(required, lookback) if alt_price else None,
            "claim_start": rand_date(rng), "hours_to_print": pick(rng, [6, 20, 48, 96, 120]),
            "submitted_by": person(rng), "policy_ref": f"PR-{rng.randint(10, 49)}",
        }

    def _meets(self, f):
        return (f["days_at_was"] >= f["required_days"] and f["days_since_last_at_was"] < f["recency_days"]
                and f["store_coverage_pct"] >= f["coverage_required_pct"])

    def intended(self, f):
        actual = (f["was_price"] - f["now_price"]) / f["was_price"] * 100
        accurate = f["printed_save_pct"] <= actual
        meets = self._meets(f)
        sp = f["special"]
        ambiguous = (sp == "stockout_days" and f["days_at_was"] - f["stockout_days"] < f["required_days"]) or \
                    (sp == "online_only_history" and f["claim_scope"] != "online only")
        if ambiguous:
            ruling = "refer_to_pricing_counsel"
        elif meets and accurate and sp is None:
            ruling = "approve_as_written"
        elif meets or f["supported_alt_price"]:
            ruling = "correct_and_approve"
        else:
            ruling = "drop_comparison"
        ratio = f["days_at_was"] / f["required_days"]
        stale = f["days_since_last_at_was"] >= f["recency_days"]
        if ratio < 0.35 or (stale and f["days_since_last_at_was"] > f["lookback_days"]):
            hist = 0
        elif not meets:
            hist = 1
        elif ratio < 1.5 or f["units_share_at_was_pct"] < 8 or f["store_coverage_pct"] < 90:
            hist = 2
        else:
            hist = 3
        over = f["printed_save_pct"] - actual
        if hist == 0:
            mislead = 3
        elif hist == 1:
            mislead = 2 if (over > 0 or ratio < 0.8 or stale) else 1
        else:
            mislead = 1 if over > 0 else 0
        if sp == "online_only_history" and f["claim_scope"] != "online only":
            mislead = max(mislead, 2)
        return {"claim_ruling": ruling, "history_support": hist, "shopper_mislead_risk": mislead,
                "meets_reference_rule": "yes" if meets else "no",
                "savings_figure_accurate": "yes" if accurate else "no"}

    def _rule(self, f):
        return (f"Policy {f['policy_ref']}: a 'was' price may be shown only if it was the offered price for at least "
                f"{f['required_days']} of the {f['lookback_days']} days before the sale, was charged at some point in "
                f"the last {f['recency_days']} days, and applied in at least {f['coverage_required_pct']}% of the stores "
                f"where the claim runs. Printed savings may be rounded down, never up.")

    def _notes(self, f, rng):
        s = []
        if f["special"]:
            s.append(f"Note from pricing: {SPECIALS[f['special']]}.")
        if f["special"] == "stockout_days":
            s.append(f"{f['stockout_days']} of the {f['days_at_was']} days fall in that stock-out.")
        if f["supported_alt_price"]:
            s.append(f"The item also sat at {money(f['supported_alt_price'])} for {f['alt_price_days']} days in the window.")
        s.append(pick(rng, ["Merchant says the 'was' price is the true regular price.",
                            "Marketing wants the bigger savings number for the headline.",
                            "Copy was built from the item master price.", "",
                            "Same claim ran last season without comment."]))
        return noisy(join_sentences(s), rng, 0.1)

    def render(self, f, rng, surface):
        notes = self._notes(f, rng)
        claim = f"Was {money(f['was_price'])}, now {money(f['now_price'])} -- save {f['printed_save_pct']}%"
        if surface == "json":
            return dumps(prune({
                "claim": {"text": claim, "channel": f["channel"], "scope": f["claim_scope"], "sku": f["sku"],
                          "item": f["product"], "banner": f["banner"], "starts": f["claim_start"].isoformat(),
                          "hours_to_print_deadline": f["hours_to_print"], "submitted_by": f["submitted_by"]},
                "price_history": {"lookback_days": f["lookback_days"], "days_offered_at_was_price": f["days_at_was"],
                                  "days_since_last_charged_at_was_price": f["days_since_last_at_was"],
                                  "share_of_stores_at_was_price_pct": f["store_coverage_pct"],
                                  "share_of_units_sold_at_was_price_pct": f["units_share_at_was_pct"]},
                "rule": self._rule(f), "notes": notes or None,
            }))
        if surface == "price_history_extract":
            rows = [["days at was price (window)", f"{f['days_at_was']} / {f['lookback_days']}"],
                    ["last charged at was price", f"{f['days_since_last_at_was']} days before start"],
                    ["stores at was price", f"{f['store_coverage_pct']}%"],
                    ["units sold at was price", f"{f['units_share_at_was_pct']}%"]]
            return "\n".join([f"PRICE HISTORY EXTRACT -- {f['sku']} {f['product']}",
                              f"Claim: \"{claim}\" ({f['channel']}, {f['claim_scope']}), from {fmt_date(f['claim_start'], rng)}",
                              "", table(["measure", "value"], rows, rng), "", notes or "No notes.", "", self._rule(f)])
        body = kv_lines([("Ticket", ident(rng, "PCR", 5, "-")), ("Submitted by", f["submitted_by"]),
                         ("Banner", f["banner"]), ("Item", f"{f['product']} ({f['sku']})"),
                         ("Claim copy", claim), ("Channel / scope", f"{f['channel']} / {f['claim_scope']}"),
                         ("Print deadline", f"{f['hours_to_print']} h"),
                         ("Days at was price", f"{f['days_at_was']} of last {f['lookback_days']}"),
                         ("Last at was price", f"{f['days_since_last_at_was']} days ago"),
                         ("Store coverage at was price", f"{f['store_coverage_pct']}%"),
                         ("Units at was price", f"{f['units_share_at_was_pct']}% of window units")], rng)
        return f"PRICING COMPLIANCE REVIEW\n{body}\n\nNotes: {notes or 'none'}\n\n{self._rule(f)}"

    def questions(self):
        return {
            "claim_ruling": choice("How should pricing compliance rule on this 'was / now' claim?", {
                "approve_as_written": "Approve the claim exactly as it is written.",
                "correct_and_approve": "Approve it only after the figures are corrected, such as a 'was' price the "
                                       "history supports or a savings number that is not rounded up.",
                "drop_comparison": "Run the sale price alone, with no 'was' price or savings claim.",
                "refer_to_pricing_counsel": "Send it to pricing counsel, because the case turns on a point the "
                                            "written rule does not settle.",
            }),
            "history_support": score("How well does the price history support the advertised 'was' price?", [
                "Not at all: the item was rarely or never sold at that price in the window.",
                "Weakly: offered at it too briefly, too long ago, or in too few stores to count.",
                "Adequately: the rule is met, though with little to spare.",
                "Strongly: offered at it for most of the window, widely, with real sales.",
            ]),
            "shopper_mislead_risk": score("How likely is the claim to give shoppers a false idea of the saving?", [
                "Unlikely: the comparison is genuine and the number is right.",
                "Slight: a small technical flaw that barely changes the impression.",
                "Real: the saving looks noticeably bigger than shoppers would actually get.",
                "High: the 'was' price is essentially fictitious.",
            ]),
            "meets_reference_rule": noul(
                "The 'was' price satisfies the retailer's written reference-pricing rule.",
                yes="The days-at-price, recency and store-coverage conditions are all met.",
                no="At least one of those conditions is not met."),
            "savings_figure_accurate": noul(
                "The printed savings percentage follows from the 'was' and 'now' prices without being rounded up."),
        }


# =========================================================================== schema 4
FRESH = {
    "bakery": (["sourdough loaf", "seeded bloomer", "croissant 4-pack", "cinnamon buns", "rye batch loaf",
                "custard tarts 2-pack"], "best_before", False),
    "ready_meals": (["chicken tikka masala", "macaroni cheese", "lasagne for two", "vegetable curry",
                     "shepherd's pie", "beef chilli & rice"], "use_by", True),
    "fresh_meat": (["chicken thighs 1 kg", "beef mince 500 g", "pork sausages", "lamb leg steaks",
                    "chicken breast fillets"], "use_by", True),
    "dairy": (["fresh whole milk 2 L", "greek-style yoghurt", "double cream", "soft cheese tub",
               "fresh custard"], "use_by", True),
    "deli": (["houmous tub", "coleslaw", "pasta salad", "sliced ham", "smoked salmon slices"], "use_by", True),
    "produce": (["salad bag mixed leaves", "punnet strawberries", "ripe avocados 2-pack", "baby spinach",
                 "prepared fruit pot"], "best_before", True),
    "sandwiches": (["egg mayo sandwich", "chicken club wrap", "falafel wrap", "BLT sandwich"], "use_by", True),
}


class ShortDatedReduction(Schema):
    name = "short_dated_reduction"
    description = ("A store's reduction check on short-dated fresh food: keep the price, step it down the ladder, "
                   "or withdraw the stock for donation or waste.")
    surfaces = {"json": 0.5, "reduction_sheet": 0.25, "handheld_log": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        cat = pick(rng, list(FRESH))
        items, dtype, chilled = FRESH[cat]
        close = pick(rng, [21, 22, 22, 23])
        gates = pick(rng, [(12, 16, 19), (13, 17, 19.5), (14, 17, 20), (12, 15, 18)])
        partner_rule_meat = chance(rng, 0.5)
        temp_limit = pick(rng, [5.0, 5.0, 8.0])
        now = round(rng.uniform(11, close - 0.5) * 4) / 4
        offset = pick_w(rng, {0: 6, 1: 3, 2: 1})
        current = 0
        temp = round(rng.uniform(1.5, temp_limit - 0.6), 1) if chilled else None
        units = rng.randint(3, 40)
        hours_today = close - now
        rate = rng.uniform(0.3, 4)
        forecast = rate * (hours_today + 12 * offset)
        partner_today = chance(rng, 0.75)
        pickup = round(rng.uniform(17, close - 0.25) * 4) / 4 if partner_today else None
        expired = False
        if reg == "clear":
            k = pick(rng, ["selling", "glut", "breach"])
            if k == "selling":
                forecast = units * rng.uniform(1.1, 2.0)
            elif k == "glut":
                offset, forecast = 0, units * rng.uniform(0.15, 0.4)
                partner_today = chance(rng, 0.4)
                now = round(rng.uniform(max(gates[2], 17), close - 0.5) * 4) / 4
            else:
                if chilled:
                    temp = round(rng.uniform(temp_limit + 1, temp_limit + 6), 1)
                else:
                    expired, offset = True, -1
        elif reg == "borderline":
            k = pick(rng, ["ratio", "gate", "temp"])
            if k == "ratio":
                forecast = units / near(rng, 1.2, 0.35, lo=0.6)
            elif k == "gate":
                g = pick(rng, list(gates))
                now = round(near(rng, g, 0.5, lo=10, hi=close - 0.5) * 4) / 4
                offset = 0
                forecast = units * rng.uniform(0.35, 0.8)
            else:
                if chilled:
                    temp = round(near(rng, temp_limit, 0.6), 1)
                forecast = units * rng.uniform(0.4, 1.3)
        else:
            k = pick(rng, ["deep_but_early", "donation_vs_reduce", "tomorrow_glut"])
            if k == "deep_but_early":
                offset, now = 0, round(rng.uniform(11, gates[1] - 0.5) * 4) / 4
                forecast = units * rng.uniform(0.15, 0.4)
            elif k == "donation_vs_reduce":
                offset, partner_today = 0, True
                now = round(rng.uniform(max(17.0, close - 3), close - 0.75) * 4) / 4
                pickup = min(close - 0.25, now + pick(rng, [0.25, 0.5, 0.75, 1.0]))
                forecast = units * rng.uniform(0.15, 0.45)
            else:
                offset, forecast = 1, units * rng.uniform(0.3, 0.6)
                now = round(rng.uniform(17, close - 0.5) * 4) / 4
        if chilled and reg != "clear" and chance(rng, 0.18):
            temp = round(rng.uniform(temp_limit + 0.3, temp_limit + 4), 1)
        hours_today = close - now
        if offset == 0 and now >= gates[0]:
            current = pick(rng, [0, 0, 25] if now < gates[1] else [0, 25, 50])
        return {
            "regime": reg, "banner": _grocer(rng), "store": f"#{rng.randint(100, 2999)}", "category": cat,
            "item": pick(rng, items), "date_type": dtype, "chilled": chilled, "date_offset_days": offset,
            "expired_yesterday": expired, "time_now": now, "close_hour": close, "ladder_gates": list(gates),
            "units_on_display": units, "forecast_full_price_units": round(max(0.3, forecast), 1),
            "current_reduction_pct": current, "chiller_max_c_last_4h": temp, "chill_limit_c": temp_limit,
            "partner_collecting_today": partner_today, "partner_pickup": pickup,
            "partner_takes_raw_meat": partner_rule_meat, "partner": f"{pick(rng, BANNER_A)} Community Pantry",
            "colleague": pick(rng, ["night manager", "fresh team leader", "duty manager", "section supervisor"]),
            "colleague_view": pick(rng, ["keep", "reduce", "bin", None]),
            "waste_vs_target": pick(rng, ["well over target", "slightly over target", "on target", "under target"]),
            "check_date": rand_date(rng),
        }

    def _safe(self, f):
        if f["expired_yesterday"]:
            return False
        t = f["chiller_max_c_last_4h"]
        return t is None or t <= f["chill_limit_c"]

    def intended(self, f):
        g1, g2, g3 = f["ladder_gates"]
        now, units, fc = f["time_now"], f["units_on_display"], f["forecast_full_price_units"]
        ratio = units / max(fc, 0.3)
        safe = self._safe(f)
        hours = f["close_hour"] - now + 14 * max(0, f["date_offset_days"])
        steps = ["keep_price", "reduce_first_step", "reduce_second_step", "reduce_final_step"]
        allowed = 0
        if f["date_offset_days"] == 0:
            allowed = 3 if now >= g3 else 2 if now >= g2 else 1 if now >= g1 else 0
        elif f["date_offset_days"] == 1:
            allowed = 1 if now >= 18 else 0
        cur = {0: 0, 25: 1, 50: 2, 75: 3}[f["current_reduction_pct"]]
        donate_ok = self._donation_open(f)
        if not safe:
            step = "withdraw_from_sale"
        elif ratio <= 1.15:
            step = steps[cur]
        elif (f["date_offset_days"] == 0 and donate_ok and ratio > 2.2 and f["partner_pickup"] is not None
              and f["partner_pickup"] - now <= 1.0 and f["close_hour"] - now <= 3):
            step = "withdraw_from_sale"
        else:
            want = cur + 1 if ratio <= 1.9 else 3
            step = steps[max(cur, min(want, allowed))]
        unsold = max(0.0, (units - fc) / units)
        risk = 0 if unsold <= 0.1 else 1 if unsold <= 0.35 else 2 if unsold <= 0.7 else 3
        if not f["expired_yesterday"]:
            tl = 0 if hours >= 10 else 1 if hours >= 4 else 2 if hours >= 1.5 else 3
        else:
            tl = 3
        return {"reduction_step": step, "unsold_stock_risk": risk, "time_left_to_sell": tl,
                "safe_to_keep_selling": "yes" if safe else "no",
                "donation_route_open": "yes" if donate_ok else "no"}

    def _donation_open(self, f):
        if not f["partner_collecting_today"] or f["partner_pickup"] is None or f["partner_pickup"] <= f["time_now"]:
            return False
        if f["expired_yesterday"] and f["date_type"] == "use_by":
            return False
        t = f["chiller_max_c_last_4h"]
        if t is not None and t > f["chill_limit_c"]:
            return False
        if f["category"] == "fresh_meat" and not f["partner_takes_raw_meat"]:
            return False
        return True

    @staticmethod
    def _hhmm(h):
        return f"{int(h):02d}:{int(round((h % 1) * 60)):02d}"

    def _date_txt(self, f):
        label = {"use_by": "use by", "best_before": "best before"}[f["date_type"]]
        if f["expired_yesterday"]:
            return f"{label} yesterday"
        return f"{label} " + {0: "today", 1: "tomorrow", 2: "in 2 days"}[f["date_offset_days"]]

    def _policy(self, f):
        g1, g2, g3 = f["ladder_gates"]
        h = self._hhmm
        return [f"Date-day ladder: 25% off from {h(g1)}, 50% from {h(g2)}, 75% from {h(g3)}; never reduce ahead of a step's time.",
                "Items dated tomorrow may take the first step only after 18:00. Reduce only when stock exceeds expected full-price sales.",
                f"Use-by items come off sale at close on the date. Any chilled item whose chiller read above "
                f"{f['chill_limit_c']} C is withdrawn, not reduced.",
                f"Donation: {f['partner']} collects at the posted time; chilled items only with a compliant chiller "
                f"log; " + ("raw meat accepted." if f["partner_takes_raw_meat"] else "no raw meat."),
                "When the partner collects within the hour and the store closes within three hours, deep surplus goes "
                "to donation rather than the 75% step."]

    def _colleague(self, f):
        v = f["colleague_view"]
        if v is None:
            return ""
        return {"keep": f"The {f['colleague']} would rather keep full price; tomorrow's delivery is short.",
                "reduce": f"The {f['colleague']} wants it reduced now to get the waste figure down ({f['waste_vs_target']}).",
                "bin": f"The {f['colleague']} suggests binning it to save time at close."}[v]

    def render(self, f, rng, surface):
        h = self._hhmm
        temp = None if f["chiller_max_c_last_4h"] is None else f"{f['chiller_max_c_last_4h']} C"
        if not (f["partner_collecting_today"] and f["partner_pickup"]):
            partner = "not collecting today"
        elif f["partner_pickup"] <= f["time_now"]:
            partner = f"today's collection already made at {h(f['partner_pickup'])}"
        else:
            partner = f"collecting at {h(f['partner_pickup'])}"
        coll = self._colleague(f)
        if surface == "json":
            return dumps(prune({
                "check": {"store": f"{f['banner']} {f['store']}", "date": f["check_date"].isoformat(),
                          "time": h(f["time_now"]), "store_closes": h(f["close_hour"])},
                "item": {"description": f["item"], "category": f["category"], "date_code": self._date_txt(f),
                         "units_on_display": f["units_on_display"],
                         "current_reduction_pct": f["current_reduction_pct"],
                         "forecast_full_price_sales_before_date_end": f["forecast_full_price_units"],
                         "chiller_max_last_4h": temp},
                "donation": {"partner": f["partner"], "status": partner},
                "store_waste_week": f["waste_vs_target"], "colleague_note": coll or None,
                "policy": self._policy(f),
            }))
        if surface == "reduction_sheet":
            rows = [[f["item"], self._date_txt(f), f["units_on_display"], f["forecast_full_price_units"],
                     f"{f['current_reduction_pct']}%", temp or "ambient"]]
            return "\n".join([f"REDUCTION CHECK  {f['banner']} {f['store']}  {fmt_date(f['check_date'], rng)} "
                              f"{h(f['time_now'])} (close {h(f['close_hour'])})", "",
                              table(["item", "date", "units", "fcst full-price", "reduced", "chiller max"], rows, rng),
                              "", f"Donation partner: {f['partner']}, {partner}.",
                              f"Waste this week: {f['waste_vs_target']}.", coll, "", "Rules:",
                              bullets(self._policy(f), rng)])
        lines = [f"{h(f['time_now'])} SCAN {f['item']} | {self._date_txt(f)} | qty {f['units_on_display']} | "
                 f"red {f['current_reduction_pct']}%",
                 f"{h(f['time_now'])} FCST full-price units to date end: {f['forecast_full_price_units']}"]
        if temp:
            lines.append(f"{h(f['time_now'])} CHILLER max(4h) {temp} (limit {f['chill_limit_c']} C)")
        lines.append(f"{h(f['time_now'])} PARTNER {f['partner']}: {partner}")
        if coll:
            lines.append(f"{h(f['time_now'])} NOTE {noisy(coll, rng, 0.15)}")
        return (f"handheld log -- {f['banner']} {f['store']} ({f['category'].replace('_', ' ')}), closes "
                f"{h(f['close_hour'])}\n" + "\n".join(lines) + "\n\nstore policy:\n" + "\n".join(
                    f"- {p}" for p in self._policy(f)))

    def questions(self):
        return {
            "reduction_step": choice("What should the store do with this stock at this check?", {
                "keep_price": "Leave it at its current price for now.",
                "reduce_first_step": "Ticket it at the first step of the ladder, about a quarter off.",
                "reduce_second_step": "Ticket it at the middle step, about half off.",
                "reduce_final_step": "Ticket it at the final step, about three quarters off.",
                "withdraw_from_sale": "Take it off sale now, for the donation partner or for waste.",
            }),
            "unsold_stock_risk": score("How much of the stock is likely to be left unsold by the end of its date at the current price?", [
                "Little or none.",
                "A small share.",
                "A large share.",
                "Nearly all of it.",
            ]),
            "time_left_to_sell": score("How little time is left to sell the stock before its date runs out?", [
                "Plenty: a full trading day or more.",
                "Some: several trading hours today.",
                "Little: under about four trading hours.",
                "Next to none: under about ninety minutes, or the date has passed.",
            ]),
            "safe_to_keep_selling": noul(
                "The stock may stay on sale under the store's date and temperature rules.",
                yes="The item is inside its date and any chiller reading is within the limit.",
                no="The date has passed or the chiller log shows a reading above the limit."),
            "donation_route_open": noul(
                "The stock can still go to the donation partner today under the partner's rules."),
        }


# =========================================================================== schema 5
CAVEATS = {
    "sample_too_small": "few test stores or too few weeks to separate the effect from noise",
    "control_not_comparable": "control stores differ materially from the test stores",
    "stock_limited_sales": "test stores ran out of key sizes, so sales were capped",
    "calendar_distortion": "a local event or weather spell hit one group of stores during the test",
}


class MarkdownTestReadout(Schema):
    name = "markdown_test_readout"
    description = ("The readout of a markdown price test in a set of stores: roll it out, roll it out to similar "
                   "stores, extend the test, or drop it.")
    surfaces = {"json": 0.5, "test_readout": 0.25, "analyst_email": 0.25}

    EVENTS = {"a heatwave in the test region in week 2": 2, "a school holiday in the test region": 1,
              "a competitor closing a store near two test stores": 1, "a local festival near the largest test store": 1}

    def sample_factors(self, rng):
        reg = regime(rng)
        dept = pick(rng, list(DEPTS))
        md = pick(rng, [15, 20, 25, 30])
        imu = rng.uniform(0.55, 0.68)
        breakeven = imu / (imu - md / 100) - 1          # unit lift at which margin dollars stay flat
        n_test, weeks = pick(rng, [12, 20, 30, 45]), pick(rng, [3, 4, 6])
        control, oos, event = "matched", rng.uniform(0, 10), None
        if reg == "clear":
            k = pick(rng, ["win", "lose"])
            n_flaws = 0
        elif reg == "borderline":
            k = "near"
            n_flaws = pick(rng, [0, 1, 1])
        else:
            k = pick(rng, ["win", "win", "win", "lose"])
            n_flaws = pick(rng, [1, 1, 2])
        for fl in rng.sample(["small", "control", "oos", "event"], n_flaws):
            if fl == "small":
                n_test, weeks = pick(rng, [4, 6, 8]), pick(rng, [1, 2])
            elif fl == "control":
                control = pick(rng, ["larger", "other_climate"])
            elif fl == "oos":
                oos = rng.uniform(15, 35)
            else:
                event = pick(rng, list(self.EVENTS))
        if event is None and chance(rng, 0.12):
            event = pick(rng, ["a long weekend in week 1", "a regional storm closing three test stores for a day"])
        factor = {"win": rng.uniform(1.25, 2.1), "lose": rng.uniform(0.2, 0.8), "near": rng.uniform(0.85, 1.2)}[k]
        lift = breakeven * 100 * factor
        cannib = rng.uniform(0, 60)       # neighbouring styles' margin loss, as % of the tested styles' gain
        ticket = _price(rng, *DEPTS[dept][2])
        cost = ticket * (1 - imu)
        base_units = rng.uniform(4, 40)
        gm_control = base_units * (ticket - cost)
        gm_test = base_units * (1 + lift / 100) * (ticket * (1 - md / 100) - cost)
        gain = gm_test - gm_control
        adj = -gain * cannib / 100 if gain > 0 else -gm_control * rng.uniform(0, 0.04)
        return {
            "regime": reg, "outcome_kind": k, "banner": _banner(rng), "dept": dept,
            "product_group": f"{pick(rng, LINES)} {pick(rng, DEPTS[dept][1])} range", "markdown_pct": md,
            "test_stores": n_test, "control_stores": n_test + pick(rng, [0, 2, 6, 10]), "weeks_run": weeks,
            "unit_lift_pct": round(lift, 1),
            "lift_interval_pts": round(max(4.0, 60 / (n_test * weeks) ** 0.5 * rng.uniform(1.6, 2.6)), 1),
            "ticket": ticket, "unit_cost": round(cost, 2), "gm_per_store_week_control": round(gm_control, 0),
            "gm_per_store_week_test": round(gm_test, 0), "adjacent_gm_change_per_store_week": round(adj, 0),
            "test_oos_pct": round(oos, 1), "control_match": control, "calendar_event": event,
            "cluster_split": chance(rng, 0.2), "season_weeks_left": rng.randint(2, 12), "analyst": person(rng),
            "analyst_view": pick(rng, ["enthusiastic", "cautious", "neutral"]), "readout_date": rand_date(rng),
            "test_id": ident(rng, "PT", 4, "-"),
        }

    def _flaws(self, f):
        """(caveat, severity): 1 = a caveat worth stating, 2 = one that undermines the read."""
        fl = []
        sw = f["test_stores"] * f["weeks_run"]
        if sw < 20:
            fl.append(("sample_too_small", 2 if sw < 10 else 1))
        if f["control_match"] != "matched":
            fl.append(("control_not_comparable", 2 if f["control_match"] == "other_climate" else 1))
        if f["test_oos_pct"] >= 15:
            fl.append(("stock_limited_sales", 2 if f["test_oos_pct"] >= 25 else 1))
        if f["calendar_event"] in self.EVENTS:
            fl.append(("calendar_distortion", self.EVENTS[f["calendar_event"]]))
        return fl

    def intended(self, f):
        fl = self._flaws(f)
        caveat = max(fl, key=lambda x: x[1])[0] if fl else "no_material_caveat"
        quality = 3 - min(3, sum(w for _, w in fl))
        net = f["gm_per_store_week_test"] - f["gm_per_store_week_control"] + f["adjacent_gm_change_per_store_week"]
        rel = net / max(1.0, f["gm_per_store_week_control"])
        effect = 0 if rel < -0.08 else 1 if rel < -0.02 else 2 if rel <= 0.02 else 3 if rel <= 0.08 else 4
        gain = f["gm_per_store_week_test"] - f["gm_per_store_week_control"]
        cann = gain > 0 and -f["adjacent_gm_change_per_store_week"] >= 0.25 * gain
        if rel <= 0.0:
            call = "drop_the_markdown"
        elif quality <= 1 and f["season_weeks_left"] >= 4:
            call = "extend_the_test"
        elif quality <= 1:
            call = "roll_out_to_matching_clusters" if rel > 0.05 else "drop_the_markdown"
        elif f["cluster_split"] or f["control_match"] == "other_climate" or quality == 2:
            call = "roll_out_to_matching_clusters"
        else:
            call = "roll_out_chainwide"
        return {"rollout_call": call, "primary_caveat": caveat, "evidence_quality": quality,
                "margin_effect": effect, "cannibalisation_material": "yes" if cann else "no"}

    def _match_txt(self, f):
        return {"matched": "matched to test stores on last year's sales (within 3%)",
                "larger": "on average about 20% larger than the test stores",
                "other_climate": "mostly in a cooler climate zone than the test stores"}[f["control_match"]]

    def _view(self, f, rng):
        return {"enthusiastic": pick(rng, ["Strong result, I'd roll it everywhere next week.",
                                           "Looks like a clear win to me."]),
                "cautious": pick(rng, ["Encouraging, but I'm not sure it would hold chain-wide.",
                                       "I'd want another read before committing."]),
                "neutral": pick(rng, ["Results below for the review.", "Summary attached."])}[f["analyst_view"]]

    def render(self, f, rng, surface):
        split = ("Lift was concentrated in the warmer-climate test stores; the rest were close to flat."
                 if f["cluster_split"] else None)
        view = self._view(f, rng)
        if surface == "json":
            return dumps(prune({
                "test": {"id": f["test_id"], "banner": f["banner"], "products": f["product_group"],
                         "department": f["dept"], "markdown_pct": f["markdown_pct"], "test_stores": f["test_stores"],
                         "control_stores": f["control_stores"], "weeks_run": f["weeks_run"],
                         "control_selection": self._match_txt(f), "readout_date": f["readout_date"].isoformat()},
                "results": {"unit_lift_vs_control_pct": f["unit_lift_pct"],
                            "lift_90pct_interval_pts": f["lift_interval_pts"],
                            "gross_margin_per_store_week_control": f["gm_per_store_week_control"],
                            "gross_margin_per_store_week_test": f["gm_per_store_week_test"],
                            "adjacent_styles_margin_change_per_store_week": f["adjacent_gm_change_per_store_week"],
                            "test_store_out_of_stock_pct": f["test_oos_pct"], "segment_note": split},
                "context": {"calendar_event": f["calendar_event"], "season_weeks_left": f["season_weeks_left"]},
                "analyst": {"name": f["analyst"], "comment": view},
            }))
        if surface == "test_readout":
            rows = [["units vs control", f"{f['unit_lift_pct']:+}% (+/- {f['lift_interval_pts']} pts, 90%)"],
                    ["GM $/store/wk control", money(f["gm_per_store_week_control"])],
                    ["GM $/store/wk test", money(f["gm_per_store_week_test"])],
                    ["adjacent styles GM change", money(f["adjacent_gm_change_per_store_week"])],
                    ["test-store OOS", f"{f['test_oos_pct']}%"]]
            return "\n".join([f"PRICE TEST READOUT {f['test_id']} -- {f['banner']}",
                              f"{f['markdown_pct']}% off {f['product_group']} ({_dept_label(f['dept'])})",
                              f"{f['test_stores']} test / {f['control_stores']} control stores, {f['weeks_run']} weeks; "
                              f"controls {self._match_txt(f)}.", "", table(["measure", "result"], rows, rng), "",
                              f"Calendar: {f['calendar_event'] or 'nothing unusual logged'}." + (f" {split}" if split else ""),
                              f"Season weeks left: {f['season_weeks_left']}.", f"Analyst ({f['analyst']}): {view}"])
        s = [f"Readout for {f['test_id']}: {f['markdown_pct']}% off the {f['product_group']} in {f['test_stores']} stores "
             f"for {f['weeks_run']} week(s), against {f['control_stores']} control stores {self._match_txt(f)}.",
             f"Units were {f['unit_lift_pct']:+}% vs control (90% interval +/- {f['lift_interval_pts']} pts).",
             f"Gross margin per store-week was {money(f['gm_per_store_week_test'])} in test vs "
             f"{money(f['gm_per_store_week_control'])} in control; neighbouring styles moved "
             f"{money(f['adjacent_gm_change_per_store_week'])} per store-week.",
             f"Test stores were out of stock {f['test_oos_pct']}% of the time.",
             f"Worth knowing: {f['calendar_event']}." if f["calendar_event"] else "", split or "",
             f"{f['season_weeks_left']} weeks of season left.", view]
        return "\n".join([f"Subject: {f['test_id']} markdown test results", f"From: {f['analyst']}", "",
                          noisy(join_sentences(s), rng, 0.1)])

    def questions(self):
        return {
            "rollout_call": choice("What should merchandising do with the tested markdown?", {
                "roll_out_chainwide": "Take the markdown in every store carrying the range.",
                "roll_out_to_matching_clusters": "Take it only in stores like the ones where it worked.",
                "extend_the_test": "Keep the test running to get a cleaner read before deciding.",
                "drop_the_markdown": "Do not take the markdown; keep the current price.",
            }),
            "primary_caveat": choice("Which weakness most limits how far the test result can be trusted?", {
                "sample_too_small": "Too few stores or weeks to tell the effect from noise.",
                "control_not_comparable": "The control stores are not a fair comparison for the test stores.",
                "stock_limited_sales": "Stock-outs in the test stores capped what the markdown could sell.",
                "calendar_distortion": "An outside event or weather spell distorted one group of stores.",
                "no_material_caveat": "No weakness stands out; the read is sound.",
            }),
            "evidence_quality": score("How far can the test read be trusted to predict the result across the chain?", [
                "Hardly at all.",
                "Only loosely.",
                "Reasonably, with one caveat.",
                "Well: a clean, adequately sized test.",
            ]),
            "margin_effect": score("What does the markdown do to gross margin dollars once the neighbouring styles are counted?", [
                "Clearly destroys margin dollars.",
                "Slightly reduces margin dollars.",
                "Roughly neutral.",
                "Slightly adds margin dollars.",
                "Clearly adds margin dollars.",
            ]),
            "cannibalisation_material": noul(
                "Losses on neighbouring styles offset a material part of the tested styles' margin gain.",
                yes="The neighbouring styles' margin drop is about a quarter of the tested styles' gain or more.",
                no="There is no gain to offset, or the neighbours' drop is small next to it."),
        }


WORKFLOW = Workflow(
    name="retail_markdown_decisions",
    title="Retail markdown decisions",
    description="When to hold price, mark down, clear or exit retail stock, against written markdown, margin and "
                "reference-pricing rules.",
    domain="retail trade / merchandising and pricing",
    schemas=[InSeasonMarkdownReview(), ClearanceExitDecision(), ReferencePriceClaimCheck(), ShortDatedReduction(),
             MarkdownTestReadout()],
    notes="Merchandising records only (dashboards, planner emails, clearance reports, price histories, reduction "
          "sheets, test readouts); no shopper conversations, no vendor bills or purchase-order matching.",
)
