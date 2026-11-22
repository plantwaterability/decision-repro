"""Hotel overbooking and walks: how far a (fictional) hotel oversells a night, and which reservation it relocates
("walks") when it is oversold, judged against written revenue-management and guest-priority rules.

Five schemas: setting the overbooking limit for a future night, choosing which arriving reservation to relocate
tonight, reviewing a group block's wash before its cut-off, closing an oversold room type during the day, and the
audit after a night on which guests were relocated. The artefacts are hotel-operations records (revenue
dashboards, arrivals sheets, group pickup reports, duty-manager logs, night-audit reports), never a conversation
with a guest. Guest priority uses only loyalty tier, rate type, stay length, arrival time, guarantee status and
past relocations: nothing about who the guest is. Hotels, cities, programmes and policies are invented. Each
schema draws a `regime` first so most cases sit near a policy line or pit one rule against another.

Each schema also has `intended(f)`: the designer's reading of the written rules for that case. It is never
rendered and never a label; it exists so the answer-spread check can confirm every question moves with the factors.
"""
from __future__ import annotations

import math

from .base import (
    Schema,
    Workflow,
    bullets,
    chance,
    choice,
    dumps,
    dumps_pretty,
    fmt_date,
    ident,
    join_sentences,
    kv_lines,
    last_name,
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

HOTEL_A = ["Harrowgate", "Larchmont", "Quayside", "Sable Point", "Tamarind", "Westerly", "Copperbeech", "Meridale",
           "Fairhaven", "Kittering", "Redwater", "Belcourt", "Linden Square", "Alderbrook", "Marisol", "Old Mill",
           "Brightholme", "Cavendale", "Wrexley", "Ottermere", "Saltings", "Hollowell"]
HOTEL_B = ["Hotel", "Grand Hotel", "Suites", "Inn & Conference Centre", "Harbour Hotel", "Lodge", "Plaza",
           "House Hotel"]
CITIES = ["Port Elsworth", "Granby", "Kellford", "Ashby Bay", "Marlow Heights", "Dunmere", "Calderwick", "Sandhaven",
          "New Ilford", "Brackwater", "Easthollow", "Pellham"]
PROGRAMME = "Lumen Rewards"
TIERS = ["Member", "Silver", "Gold", "Platinum", "Ambassador"]
EVENTS = ["the regional medical congress", "the marathon weekend", "the boat show", "the university graduation",
          "a stadium concert", "the trade fair", "the food festival", "the national swim championships"]


def _hotel(rng):
    return f"{pick(rng, HOTEL_A)} {pick(rng, HOTEL_B)}"


def _hhmm(h):
    h = max(0.0, min(23.98, h))
    return f"{int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}"


# =========================================================================== schema 1
class OverbookingLimitSetting(Schema):
    name = "overbooking_limit_setting"
    description = ("Revenue management sets the overbooking limit for a future night: raise it, keep it, lower it, "
                   "or stop overbooking.")
    surfaces = {"json": 0.5, "rm_dashboard": 0.25, "rm_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        rooms = rng.randint(90, 620)
        ooo = rng.randint(0, max(1, rooms // 40))
        ns = round(rng.uniform(2, 9), 1)
        canc = round(rng.uniform(1, 7), 1)
        group = rng.randint(0, 35)
        event = None
        compression = False
        reloc = -1
        trend = pick_w(rng, {1.0: 6, 1.5: 2, 0.6: 2})
        sd_pct = rng.uniform(0.25, 0.6)
        current = rng.randint(0, max(2, int(rooms * 0.06)))
        if reg == "clear":
            k = pick(rng, ["roomy", "tight"])
            if k == "roomy":
                ns, canc, group, reloc, trend = round(rng.uniform(6, 10), 1), round(rng.uniform(4, 8), 1), rng.randint(0, 15), rng.randint(60, 120), 1.0
                current = rng.randint(0, max(1, int(rooms * 0.02)))
            else:
                event, compression, reloc = pick(rng, EVENTS), True, rng.randint(0, 6)
                current = rng.randint(max(2, int(rooms * 0.03)), max(3, int(rooms * 0.07)))
        elif reg == "borderline":
            k = pick(rng, ["near_safe", "reloc"])
            if k == "reloc":
                reloc = None  # set after the safe level is known
        else:
            k = pick(rng, ["event_no_compression", "groups", "trend"])
            if k == "event_no_compression":
                event, compression = pick(rng, EVENTS), chance(rng, 0.5)
            elif k == "groups":
                group = rng.randint(45, 75)
            else:
                trend = pick(rng, [1.5, 1.7, 0.5, 0.6])
        avail = rooms - ooo
        proj = avail + current
        eff_ns = ns * (1 - group / 100 * 0.7) * (0.6 if compression else 1.0)
        exp_ns = proj * eff_ns / 100
        exp_cx = proj * canc * trend / 100
        sd = max(1.0, (exp_ns + exp_cx) * sd_pct)
        safe = max(0, math.floor(exp_ns + exp_cx - sd))
        if compression:
            safe = safe // 2
        if reg == "borderline" and k == "near_safe":
            current = max(0, int(round(near(rng, safe, 2.5))))
        elif not (reg == "clear"):
            current = max(0, int(round(exp_ns + exp_cx + sd * rng.uniform(-2.4, 1.6))))
        if reloc is None:
            reloc = max(0, int(round(near(rng, 2 * max(1, safe, current), max(2, safe)))))
        elif reloc == -1:
            reloc = rng.randint(max(0, safe), max(4, 2 * max(safe, current)) * 3)
        on_books = avail + current - rng.randint(0, max(1, current + 3))
        return {
            "regime": reg, "hotel": _hotel(rng), "city": pick(rng, CITIES), "rooms": rooms, "out_of_order": ooo,
            "days_out": pick(rng, [3, 5, 7, 10, 14, 21, 28]), "night": rand_date(rng), "on_books": on_books,
            "current_limit": current, "no_show_pct": ns, "late_cancel_pct": canc, "recent_cancel_trend": trend,
            "group_share_pct": group, "city_event": event, "compression_flag": compression,
            "relocation_rooms_5km": reloc, "forecast_sd_rooms": round(sd, 1),
            "expected_no_shows": round(exp_ns, 1), "expected_late_cancels": round(exp_cx, 1),
            "walk_cost": rng.randint(180, 520), "adr": rng.randint(110, 420), "analyst": person(rng),
        }

    def _safe(self, f):
        s = max(0, math.floor(f["expected_no_shows"] + f["expected_late_cancels"] - f["forecast_sd_rooms"]))
        if f["compression_flag"]:
            s //= 2
        if f["relocation_rooms_5km"] < 2 * s:
            s = 0
        return s

    def intended(self, f):
        safe = self._safe(f)
        cur = f["current_limit"]
        if safe == 0 and cur > 0:
            adj = "stop_overbooking"
        elif safe >= cur + 2:
            adj = "raise_limit"
        elif safe < cur - 1:
            adj = "lower_limit"
        else:
            adj = "keep_current_limit"
        if f["city_event"]:
            drv = "event_compression"
        elif f["group_share_pct"] >= 40:
            drv = "group_mix"
        elif f["recent_cancel_trend"] != 1.0:
            drv = "recent_cancellation_trend"
        else:
            drv = "day_of_week_history"
        exp = f["expected_no_shows"] + f["expected_late_cancels"]
        z = (cur - exp) / f["forecast_sd_rooms"]
        risk = 0 if z < -1.5 else 1 if z < -0.5 else 2 if z < 0.5 else 3
        rel_sd = f["forecast_sd_rooms"] / max(1.0, exp)
        pts = (1 if rel_sd > 0.45 else 0) + (1 if f["city_event"] else 0) + (1 if f["days_out"] >= 14 else 0) \
            + (1 if f["recent_cancel_trend"] != 1.0 else 0)
        reliab = max(0, 3 - pts)
        return {"limit_adjustment": adj, "no_show_driver": drv, "walk_risk_at_current_limit": risk,
                "forecast_reliability": reliab,
                "relocation_cover_adequate": "yes" if f["relocation_rooms_5km"] >= 2 * cur else "no"}

    def _policy(self):
        return ["Overbooking limit: expected no-shows plus expected late cancellations, minus one standard deviation "
                "of the forecast, rounded down.",
                "Halve that figure on nights flagged for city-wide compression.",
                "Set the limit to zero when comparable rooms available within 5 km number fewer than twice the limit.",
                "Change the limit only when the policy figure differs from it by two rooms or more, except to stop."]

    def _trend_txt(self, f):
        t = f["recent_cancel_trend"]
        if t == 1.0:
            return "in line with history"
        return f"running at {t:g}x the historical rate over the last four weeks (already in the expectation)"

    def render(self, f, rng, surface):
        ev = (f"{f['city_event']}" + (" (city-wide compression flagged)" if f["compression_flag"] else
                                      " (not flagged as compression)")) if f["city_event"] else None
        if surface == "json":
            return dumps(prune({
                "night": {"hotel": f["hotel"], "city": f["city"], "date": f["night"].isoformat(),
                          "days_out": f["days_out"], "event": ev},
                "inventory": {"rooms": f["rooms"], "out_of_order": f["out_of_order"], "on_books": f["on_books"],
                              "current_overbooking_limit": f["current_limit"],
                              "group_share_of_on_books_pct": f["group_share_pct"]},
                "forecast": {"no_show_rate_pct": f["no_show_pct"], "late_cancel_rate_pct": f["late_cancel_pct"],
                             "late_cancel_trend": self._trend_txt(f),
                             "expected_no_shows": f["expected_no_shows"],
                             "expected_late_cancellations": f["expected_late_cancels"],
                             "forecast_sd_rooms": f["forecast_sd_rooms"]},
                "relocation": {"comparable_rooms_within_5km": f["relocation_rooms_5km"],
                               "est_cost_per_walk_usd": f["walk_cost"], "adr_usd": f["adr"]},
                "policy": self._policy(),
            }))
        if surface == "rm_dashboard":
            rows = [["rooms / OOO", f"{f['rooms']} / {f['out_of_order']}"], ["on the books", f["on_books"]],
                    ["current OB limit", f["current_limit"]], ["group share", f"{f['group_share_pct']}%"],
                    ["no-show rate", f"{f['no_show_pct']}%"], ["late cancel rate", f"{f['late_cancel_pct']}%"],
                    ["exp. no-shows", f["expected_no_shows"]], ["exp. late cancels", f["expected_late_cancels"]],
                    ["forecast SD (rooms)", f["forecast_sd_rooms"]],
                    ["comparable rooms <5 km", f["relocation_rooms_5km"]], ["walk cost / ADR", f"${f['walk_cost']} / ${f['adr']}"]]
            return "\n".join([f"OVERBOOKING REVIEW  {f['hotel']}, {f['city']}  night of {fmt_date(f['night'], rng)} "
                              f"({f['days_out']} days out)", "", table(["metric", "value"], rows, rng), "",
                              f"Event: {ev or 'none'}. Late cancellations: {self._trend_txt(f)}.", "",
                              "Revenue policy:", bullets(self._policy(), rng)])
        s = [f"Night of {fmt_date(f['night'], rng)}, {f['days_out']} days out: {f['on_books']} on the books for "
             f"{f['rooms'] - f['out_of_order']} sellable rooms ({f['out_of_order']} out of order), current limit "
             f"{f['current_limit']}.",
             f"Groups are {f['group_share_pct']}% of the books.",
             f"Forecast: {f['expected_no_shows']} no-shows and {f['expected_late_cancels']} late cancellations, SD "
             f"{f['forecast_sd_rooms']} rooms; late cancellations are {self._trend_txt(f)}.",
             f"Event in town: {ev}." if ev else "Nothing special in town that night.",
             f"Partners report {f['relocation_rooms_5km']} comparable rooms within 5 km; a walk costs about ${f['walk_cost']}."]
        return "\n".join([f"RM note -- {f['hotel']} ({f['analyst']})", "", noisy(join_sentences(s), rng, 0.08), "",
                          "Policy:", bullets(self._policy(), rng)])

    def questions(self):
        return {
            "limit_adjustment": choice("What should revenue management do with the overbooking limit for this night?", {
                "raise_limit": "Raise the limit to the level the policy supports.",
                "keep_current_limit": "Leave the limit where it is.",
                "lower_limit": "Lower the limit to the level the policy supports.",
                "stop_overbooking": "Set the limit to zero and sell no rooms beyond capacity.",
            }),
            "no_show_driver": choice("Which factor should most shape the no-show and cancellation assumption for this night?", {
                "day_of_week_history": "The usual pattern for this day of the week and season.",
                "group_mix": "The unusual share of group rooms on the books.",
                "event_compression": "The event in town, which changes how guests cancel and show up.",
                "recent_cancellation_trend": "The recent shift in late-cancellation behaviour.",
            }),
            "walk_risk_at_current_limit": score("If the current limit is kept and sold out, how likely is the hotel to have to relocate guests?", [
                "Very unlikely.",
                "Unlikely.",
                "About even.",
                "Likely.",
            ]),
            "forecast_reliability": score("How much can the no-show and cancellation forecast be trusted for this night?", [
                "Hardly at all.",
                "Somewhat.",
                "Reasonably.",
                "Well.",
            ]),
            "relocation_cover_adequate": noul(
                "Comparable rooms nearby are enough to back the current overbooking limit under the policy.",
                yes="Comparable rooms within 5 km number at least twice the current limit.",
                no="They number fewer than twice the current limit."),
        }


# =========================================================================== schema 2
RATES = {"ota_prepaid": "online travel agency, prepaid", "opaque": "opaque / wholesale rate",
         "bar_flex": "best available rate, flexible", "corporate": "corporate negotiated rate",
         "group": "group block", "award": f"{PROGRAMME} award stay", "crew": "airline crew contract"}
LETTERS = "abcde"


class WalkSelectionTonight(Schema):
    name = "walk_selection_tonight"
    description = ("The hotel is oversold tonight; choose which arriving reservation to relocate under the guest-"
                   "priority rules, and what relocation package applies.")
    surfaces = {"json": 0.5, "arrivals_sheet": 0.25, "duty_manager_log": 0.25}

    def _cand(self, rng, cls, now):
        """A candidate whose priority class (lower = relocate first) is `cls`; 9 = protected. Arrives after `now`."""
        c = {"tier": pick(rng, [None, None, "Member", "Silver"]), "nights": pick(rng, [1, 1, 2, 3]),
             "eta": round(rng.uniform(now + 0.25, 23.75) * 4) / 4, "guaranteed": True, "relocated_before": False}
        if cls == 0:
            c.update(rate="bar_flex", guaranteed=False,
                     eta=round(rng.uniform(max(18.5, now + 0.25), 23.75) * 4) / 4)
        elif cls == 1:
            c.update(rate=pick(rng, ["ota_prepaid", "opaque"]), nights=1)
        elif cls == 2:
            c.update(rate=pick(rng, ["bar_flex", "ota_prepaid", "opaque"]))
            if c["rate"] != "bar_flex":
                c["nights"] = pick(rng, [2, 3])
        elif cls == 3:
            c.update(rate="corporate", tier=pick(rng, [None, "Member", "Silver"]))
        elif cls == 4:
            c.update(rate="group")
        elif cls == 5:
            c.update(rate=pick(rng, ["award", "bar_flex", "corporate"]), tier="Gold")
            if c["rate"] == "award":
                c["tier"] = pick(rng, ["Silver", "Gold"])
        else:
            k = pick(rng, ["crew", "platinum", "relocated"])
            if k == "crew":
                c.update(rate="crew", tier=None)
            elif k == "platinum":
                c.update(rate=pick(rng, ["bar_flex", "ota_prepaid", "corporate"]), tier=pick(rng, ["Platinum", "Ambassador"]), nights=1)
            else:
                c.update(rate=pick(rng, ["ota_prepaid", "opaque"]), nights=1, relocated_before=True)
        c["res"] = ident(rng, "R", 7, "")
        c["guest"] = last_name(rng)
        return c

    def sample_factors(self, rng):
        reg = regime(rng)
        if reg == "clear":
            low = pick(rng, [0, 1, 1])
            others = [pick(rng, [3, 4, 5, 9]) for _ in range(4)]
            classes = [low] + others
        elif reg == "borderline":
            low = pick(rng, [1, 2, 3])
            classes = [low, low] + [pick(rng, [low + 1, 4, 5, 9]) for _ in range(3)]
        else:
            k = pick(rng, ["protected_cheap", "all_protected", "multi_night"])
            if k == "protected_cheap":
                classes = [9, 9] + [pick(rng, [2, 3, 4, 5]) for _ in range(3)]
            elif k == "all_protected":
                classes = [9, 9, 9, 9, 9]
            else:
                classes = [2, 3, 4, 5, 9]
        now = round(rng.uniform(13, 19.5) * 4) / 4
        cands = [self._cand(rng, c, now) for c in classes]
        if reg == "conflicting" and classes == [2, 3, 4, 5, 9]:
            cands[0]["nights"] = pick(rng, [3, 4, 5])
            cands[0]["rate"] = "ota_prepaid"
        if reg == "borderline":
            # the tie is broken by stay length or arrival time
            if chance(rng, 0.5):
                cands[1]["nights"] = cands[0]["nights"]
                cands[1]["eta"] = min(23.75, max(now + 0.25, cands[0]["eta"] + pick(rng, [-1.5, -0.75, 0.75, 1.5])))
        rng.shuffle(cands)
        hold = round(min(23.5, now + rng.uniform(1.0, 6)) * 4) / 4
        our_stars = pick(rng, [3, 3.5, 4, 4, 4.5])
        km_limit = pick(rng, [3, 5, 8])
        partner_stars = our_stars + pick(rng, [0, 0, 0.5, -0.5])
        partner_km = round(rng.uniform(0.5, km_limit * 1.4), 1)
        if chance(rng, 0.35):
            partner_km = round(near(rng, km_limit, 1.2, lo=0.3), 1)
        return {"regime": reg, "classes_sampled": classes, "hotel": _hotel(rng), "city": pick(rng, CITIES),
                "candidates": cands, "oversold_by": pick(rng, [1, 1, 2]), "time_now": now,
                "partner": _hotel(rng), "partner_stars": partner_stars, "our_stars": our_stars,
                "partner_km": partner_km, "km_limit": km_limit, "partner_hold_until": hold,
                "partner_room_type": pick(rng, ["equivalent king", "equivalent king", "smaller double"]),
                "duty_manager": person(rng), "date": rand_date(rng)}

    @staticmethod
    def _protected(c):
        return c["rate"] == "crew" or c["tier"] in ("Platinum", "Ambassador") or c["relocated_before"]

    @staticmethod
    def _class(c):
        if WalkSelectionTonight._protected(c):
            return 9
        if not c["guaranteed"] and c["eta"] > 18:
            return 0
        if c["rate"] in ("ota_prepaid", "opaque") and c["nights"] == 1:
            return 1
        if c["tier"] == "Gold" or c["rate"] == "award":
            return 5
        if c["rate"] in ("bar_flex", "ota_prepaid", "opaque"):
            return 2
        if c["rate"] == "corporate":
            return 3
        return 4

    def _chosen(self, f):
        cs = f["candidates"]
        order = sorted(range(len(cs)), key=lambda i: (self._class(cs[i]), cs[i]["nights"], -cs[i]["eta"]))
        return order, cs[order[0]]

    def intended(self, f):
        order, ch = self._chosen(f)
        cs = f["candidates"]
        cls = [self._class(c) for c in cs]
        first = cls[order[0]]
        if first == 9:
            comp = "gm_exception"
        elif ch["nights"] > 1:
            comp = "enhanced_with_return_night"
        elif ch["tier"] in ("Silver", "Gold") or ch["rate"] == "corporate":
            comp = "standard_plus_points"
        else:
            comp = "standard_package"
        same = sum(1 for c in cls if c == first)
        cheap_protected = any(self._protected(c) and c["rate"] in ("ota_prepaid", "opaque") and c["nights"] == 1 for c in cs)
        if first == 9:
            tension = 3
        elif ch["nights"] > 2 or (cheap_protected and first >= 2):
            tension = 2
        elif same > 1 or cheap_protected:
            tension = 1
        else:
            tension = 0
        mins = (min(ch["eta"], f["partner_hold_until"]) - f["time_now"]) * 60
        dead = 0 if mins > 240 else 1 if mins > 120 else 2 if mins > 30 else 3
        comparable = (f["partner_stars"] >= f["our_stars"] and f["partner_km"] <= f["km_limit"]
                      and f["partner_room_type"] == "equivalent king")
        return {"reservation_to_relocate": f"relocate_reservation_{LETTERS[order[0]]}", "walk_compensation": comp,
                "rule_tension": tension, "relocation_deadline": dead,
                "partner_hotel_comparable": "yes" if comparable else "no"}

    def _policy(self, f):
        return ["Never relocate: airline crew under contract, Platinum or Ambassador members, or guests we have "
                "relocated in the last 12 months.",
                "Relocate first: non-guaranteed reservations arriving after 18:00; then one-night OTA-prepaid or "
                "opaque stays; then other transient leisure stays; then corporate rates; then group block guests; "
                "then award stays and Gold members.",
                "Within a level: shorter stays first, then the latest arrival.",
                "Package: first night paid at the partner hotel, transport both ways, one call. Add 5,000 points for "
                "Silver or Gold members and corporate guests. Guests staying more than one night also get transport "
                "back next day and a free night on a future stay. Relocating a protected guest needs the GM, who "
                "sets the package.",
                f"The partner hotel must be rated at least as high as ours, within {f['km_limit']} km, with an "
                f"equivalent room."]

    def _row(self, c, i):
        return [LETTERS[i].upper(), c["res"], c["guest"], RATES[c["rate"]], c["tier"] or "-", c["nights"],
                _hhmm(c["eta"]), "yes" if c["guaranteed"] else "no", "yes" if c["relocated_before"] else "no"]

    def render(self, f, rng, surface):
        cs = f["candidates"]
        partner = (f"{f['partner']}: {f['partner_stars']:g} stars (ours {f['our_stars']:g}), {f['partner_km']} km, "
                   f"{f['partner_room_type']}, holding rooms until {_hhmm(f['partner_hold_until'])}")
        if surface == "json":
            return dumps(prune({
                "situation": {"hotel": f["hotel"], "city": f["city"], "date": f["date"].isoformat(),
                              "time_now": _hhmm(f["time_now"]), "oversold_by_rooms": f["oversold_by"]},
                "candidates": [{"label": LETTERS[i].upper(), "reservation": c["res"], "guest": c["guest"],
                                "rate": RATES[c["rate"]], "loyalty_tier": c["tier"], "nights": c["nights"],
                                "eta": _hhmm(c["eta"]), "guaranteed": c["guaranteed"],
                                "relocated_by_us_last_12m": c["relocated_before"]} for i, c in enumerate(cs)],
                "partner_hotel": partner, "policy": self._policy(f),
            }))
        if surface == "arrivals_sheet":
            return "\n".join([
                f"WALK CANDIDATES  {f['hotel']}, {f['city']}  {fmt_date(f['date'], rng)} at {_hhmm(f['time_now'])}",
                f"Oversold by {f['oversold_by']} room(s). First relocation to decide.", "",
                table(["", "res", "guest", "rate", "tier", "nts", "ETA", "guar.", "walked <12m"],
                      [self._row(c, i) for i, c in enumerate(cs)], rng), "",
                f"Partner: {partner}.", "", "Guest-priority rules:", bullets(self._policy(f), rng)])
        lines = [f"{_hhmm(f['time_now'])} oversold by {f['oversold_by']}; reviewing arrivals for relocation."]
        for i, c in enumerate(cs):
            lines.append(f"  {LETTERS[i].upper()}. {c['res']} {c['guest']}: {RATES[c['rate']]}, "
                         f"{c['tier'] or 'no tier'}, {c['nights']} night(s), ETA {_hhmm(c['eta'])}"
                         + ("" if c["guaranteed"] else ", NOT guaranteed")
                         + (", relocated by us earlier this year" if c["relocated_before"] else ""))
        lines.append(f"Partner: {partner}.")
        return "\n".join([f"DUTY MANAGER LOG -- {f['hotel']} -- {f['duty_manager']}"] + lines +
                         ["", "Rules:", bullets(self._policy(f), rng)])

    def questions(self):
        return {
            "reservation_to_relocate": choice("Which reservation should the hotel relocate first under its rules?", {
                "relocate_reservation_a": "Reservation A, the first one listed.",
                "relocate_reservation_b": "Reservation B, the second one listed.",
                "relocate_reservation_c": "Reservation C, the third one listed.",
                "relocate_reservation_d": "Reservation D, the fourth one listed.",
                "relocate_reservation_e": "Reservation E, the fifth one listed.",
            }),
            "walk_compensation": choice("Which relocation package applies to the guest who should be relocated?", {
                "standard_package": "The standard package only: first night, transport and a call.",
                "standard_plus_points": "The standard package plus the loyalty points bonus.",
                "enhanced_with_return_night": "The standard package plus return transport and a free future night.",
                "gm_exception": "A package set by the general manager, because the guest is normally protected.",
            }),
            "rule_tension": score("How much do the hotel's priority rules pull against each other in this choice?", [
                "Not at all: one reservation is plainly first.",
                "A little: a tie-breaker settles it.",
                "Noticeably: the rules skip an obvious candidate or land on a long stay.",
                "Severely: every candidate is protected.",
            ]),
            "relocation_deadline": score("How soon must the relocation be arranged?", [
                "No rush: more than four hours before the guest arrives or the partner's hold lapses.",
                "Within two to four hours.",
                "Within the next two hours.",
                "Within half an hour.",
            ]),
            "partner_hotel_comparable": noul(
                "The partner hotel meets the policy's standard for a comparable relocation.",
                yes="It is rated at least as high, within the distance limit, with an equivalent room.",
                no="It is rated lower, too far away, or offers a lesser room."),
        }


# =========================================================================== schema 3
GROUPS = [("dental association annual meeting", "association"), ("wedding party", "wedding"),
          ("under-14 football tournament teams", "sports"), ("sales kickoff", "corporate"),
          ("film production crew", "production"), ("alumni reunion", "social"),
          ("engineering society conference", "association"), ("choir festival delegation", "social"),
          ("software user conference", "corporate"), ("cycling tour riders", "sports")]
SEG_WASH = {"association": 12, "wedding": 22, "sports": 15, "corporate": 18, "production": 8, "social": 25}


class GroupBlockWashReview(Schema):
    name = "group_block_wash_review"
    description = ("A group room block running behind before its cut-off: hold it, release part, release all "
                   "unpicked rooms, or ask the group to reduce the block.")
    surfaces = {"json": 0.5, "pickup_report": 0.25, "pretty_json": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        gname, seg = pick(rng, GROUPS)
        block = rng.randint(20, 160)
        pickup_pct = rng.uniform(65, 100)
        cutoff = rng.randint(3, 35)
        attr = pick(rng, [10, 15, 20, 25])
        clause_days = pick(rng, [None, 14, 21, 30])
        clause_pct = pick(rng, [50, 60, 70])
        hist = pick(rng, [None, rng.randint(55, 100)])
        pace = rng.randint(85, 140)
        rate_gap = rng.randint(-5, 40)            # % the forecast public rate is above the group rate
        if reg == "clear":
            if chance(rng, 0.5):
                pickup_pct, hist, pace, cutoff = rng.uniform(85, 98), rng.randint(95, 102), rng.randint(80, 95), rng.randint(5, 30)
            else:
                pickup_pct, hist, pace = rng.uniform(25, 45), rng.randint(30, 50), rng.randint(115, 140)
                clause_days, cutoff = pick(rng, [21, 30]), rng.randint(3, 20)
                clause_pct = 60
        elif reg == "borderline":
            k = pick(rng, ["clause_date", "clause_pct", "attrition"])
            if k == "clause_date":
                clause_days = pick(rng, [14, 21, 30])
                cutoff = near(rng, clause_days, 3, lo=1, integer=True)
                pickup_pct = rng.uniform(30, clause_pct - 5)
            elif k == "clause_pct":
                clause_days = pick(rng, [14, 21, 30])
                cutoff = rng.randint(2, clause_days)
                pickup_pct = near(rng, clause_pct, 6, lo=10, hi=99)
            else:
                pickup_pct = near(rng, 100 - attr, 4, lo=10, hi=99)
        else:
            k = pick(rng, ["strong_demand_no_clause", "planner_promises", "low_demand_behind"])
            if k == "strong_demand_no_clause":
                clause_days, pace, pickup_pct = None, rng.randint(118, 140), rng.uniform(30, 55)
                hist = rng.randint(55, 70)
            elif k == "planner_promises":
                pickup_pct, hist, pace = rng.uniform(35, 55), rng.randint(90, 100), rng.randint(110, 130)
                clause_days, cutoff = pick(rng, [21, 30]), rng.randint(3, 15)
            else:
                pickup_pct, pace, hist = rng.uniform(30, 50), rng.randint(78, 92), rng.randint(55, 70)
                clause_days, cutoff = pick(rng, [21, 30]), rng.randint(3, 15)
        picked = max(1, min(block, int(round(block * pickup_pct / 100))))
        return {
            "regime": reg, "hotel": _hotel(rng), "city": pick(rng, CITIES), "group": f"{last_name(rng)} {gname}",
            "segment": seg, "block_rooms_peak": block, "picked_up": picked, "days_to_cutoff": cutoff,
            "attrition_allowance_pct": attr, "early_release_days": clause_days, "early_release_below_pct": clause_pct,
            "last_year_final_pickup_pct": hist, "segment_wash_pct": SEG_WASH[seg], "transient_pace_pct": pace,
            "public_rate_premium_pct": rate_gap, "rooming_list": pick(rng, ["not received", "partial", "received"]),
            "planner_note": pick(rng, ["Registrations always come in late for us.", "We expect a late surge after "
                                       "the programme is published.", "Numbers are lower than hoped this year.", "", ""]),
            "sales_manager": person(rng), "arrival": rand_date(rng),
        }

    def _expected_final_pct(self, f):
        h = f["last_year_final_pickup_pct"]
        return h if h is not None else 100 - f["segment_wash_pct"]

    def intended(self, f):
        block, picked = f["block_rooms_peak"], f["picked_up"]
        pickup = picked / block * 100
        exp_final = max(pickup, min(100, self._expected_final_pct(f)))
        unpicked = block - picked
        unused = (block - block * exp_final / 100) / unpicked if unpicked else 0
        wash = 0 if unused < 0.25 else 1 if unused < 0.5 else 2 if unused < 0.75 else 3
        pace = f["transient_pace_pct"]
        disp = 0 if pace < 92 else 1 if pace < 105 else 2 if pace < 120 else 3
        if f["public_rate_premium_pct"] >= 30 and disp < 3 and pace >= 100:
            disp += 1
        permitted = (f["early_release_days"] is not None and f["days_to_cutoff"] <= f["early_release_days"]
                     and pickup < f["early_release_below_pct"])
        attrition = exp_final < 100 - f["attrition_allowance_pct"]
        if permitted and disp >= 2 and wash >= 3:
            act = "release_all_unpicked"
        elif permitted and disp >= 2 and wash >= 1:
            act = "release_partial_keep_buffer"
        elif not permitted and disp >= 2 and wash >= 2:
            act = "negotiate_block_reduction"
        else:
            act = "hold_full_block"
        return {"block_action": act, "expected_wash": wash, "displacement_cost": disp,
                "early_release_permitted": "yes" if permitted else "no",
                "attrition_charge_likely": "yes" if attrition else "no"}

    def _contract(self, f):
        out = [f"Attrition: the group owes for rooms below {100 - f['attrition_allowance_pct']}% of the block at cut-off."]
        if f["early_release_days"] is None:
            out.append("The contract has no early-release clause; the hotel holds the block until cut-off.")
        else:
            out.append(f"Early release: from {f['early_release_days']} days before cut-off, the hotel may release "
                       f"unpicked rooms if pickup is below {f['early_release_below_pct']}% of the block.")
        out.append("Hotel practice: ask the planner to reduce the block when the contract does not allow release and "
                   "the rooms are needed.")
        return out

    def render(self, f, rng, surface):
        pickup = round(f["picked_up"] / f["block_rooms_peak"] * 100)
        hist = (f"{f['last_year_final_pickup_pct']}% final pickup last year" if f["last_year_final_pickup_pct"] is not None
                else f"first year; {f['segment']} groups typically wash {f['segment_wash_pct']}%")
        note = f["planner_note"]
        if surface in ("json", "pretty_json"):
            obj = prune({
                "group": {"name": f["group"], "segment": f["segment"], "hotel": f["hotel"], "city": f["city"],
                          "arrival": f["arrival"].isoformat(), "rooming_list": f["rooming_list"],
                          "planner_note": note or None},
                "block": {"peak_night_rooms": f["block_rooms_peak"], "picked_up": f["picked_up"],
                          "pickup_pct": pickup, "days_to_cutoff": f["days_to_cutoff"], "history": hist},
                "demand": {"transient_pace_vs_last_year_pct": f["transient_pace_pct"],
                           "public_rate_above_group_rate_pct": f["public_rate_premium_pct"]},
                "contract": self._contract(f),
            })
            return dumps(obj) if surface == "json" else dumps_pretty(obj, rng)
        rows = [["peak-night block", f["block_rooms_peak"]], ["picked up", f"{f['picked_up']} ({pickup}%)"],
                ["days to cut-off", f["days_to_cutoff"]], ["history", hist], ["rooming list", f["rooming_list"]],
                ["transient pace vs LY", f"{f['transient_pace_pct']}%"],
                ["public rate vs group rate", f"{f['public_rate_premium_pct']:+d}%"]]
        return "\n".join([f"GROUP PICKUP REPORT  {f['hotel']}, {f['city']}", f"{f['group']} ({f['segment']}), arriving "
                          f"{fmt_date(f['arrival'], rng)}; sales: {f['sales_manager']}", "",
                          table(["item", "value"], rows, rng), "",
                          (f"Planner: \"{noisy(note, rng, 0.1)}\"" if note else "No word from the planner."), "",
                          "Contract terms:", bullets(self._contract(f), rng)])

    def questions(self):
        return {
            "block_action": choice("What should the hotel do with the group's room block now?", {
                "hold_full_block": "Keep the whole block for the group until cut-off.",
                "release_partial_keep_buffer": "Release part of the unpicked rooms and keep a buffer for late pickup.",
                "release_all_unpicked": "Release every room the group has not picked up.",
                "negotiate_block_reduction": "Ask the group's planner to agree a smaller block now.",
            }),
            "expected_wash": score("How much of the still-unpicked block is likely to go unused?", [
                "Little: most of it will still be picked up.",
                "Some of it.",
                "Most of it.",
                "Nearly all of it.",
            ]),
            "displacement_cost": score("How much other business is the held block likely to turn away?", [
                "None: the hotel will not fill those rooms anyway.",
                "A little.",
                "A fair amount, at better rates.",
                "A lot: strong demand at much higher rates.",
            ]),
            "early_release_permitted": noul(
                "The contract allows the hotel to release unpicked rooms today.",
                yes="There is an early-release clause, today is inside its window, and pickup is below its trigger.",
                no="There is no such clause, today is too early, or pickup is at or above the trigger."),
            "attrition_charge_likely": noul(
                "On the group's own pickup record, its final pickup will fall short of the contract minimum."),
        }


# =========================================================================== schema 4
ROOM_TYPES = [("Standard King", "Deluxe King"), ("Standard Double", "Deluxe Double"),
              ("Deluxe King", "Junior Suite"), ("Standard Twin", "Deluxe Twin")]
OOO_STATUS = {"ready_by": "repair done, awaiting inspection; engineering says ready by",
              "parts": "waiting on a part; no firm time", "drying": "carpet drying; expected ready by"}


class OversoldRoomTypePlan(Schema):
    name = "oversold_room_type_plan"
    description = ("One room type is oversold for tonight: decide which lever closes the gap (free upgrades, returning "
                   "out-of-order rooms, expected no-shows) or whether relocations must be booked now.")
    surfaces = {"json": 0.5, "front_office_log": 0.25, "shift_handover": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        rtype, up = pick(rng, ROOM_TYPES)
        oversold = rng.randint(1, 7)
        upgrades = rng.randint(0, 4)
        arrivals = rng.randint(15, 90)
        ns_rate = round(rng.uniform(1.5, 7), 1)
        now = round(rng.uniform(10, 19.5) * 4) / 4
        ooo = []
        for _ in range(pick(rng, [0, 1, 1, 2, 3])):
            st = pick(rng, list(OOO_STATUS))
            ooo.append({"room": str(rng.randint(2, 14) * 100 + rng.randint(1, 40)), "status": st,
                        "eta": round(rng.uniform(13, 22) * 4) / 4 if st != "parts" else None})
        if reg == "clear":
            if chance(rng, 0.5):
                upgrades = oversold + rng.randint(0, 3)
            else:
                oversold, upgrades, ooo, now = rng.randint(5, 9), rng.randint(0, 1), [], round(rng.uniform(16, 19.5) * 4) / 4
                ns_rate = round(rng.uniform(1, 2.5), 1)
        elif reg == "borderline":
            upgrades = max(0, oversold - rng.randint(1, 3))
            ns_rate = round(rng.uniform(2, 6), 1)
            for o in ooo:
                if o["eta"] is not None:
                    o["eta"] = round(near(rng, 18, 1.0, lo=12, hi=22.5) * 4) / 4
        else:
            k = pick(rng, ["ooo_saves", "early_in_day", "noshow_reliance"])
            if k == "ooo_saves":
                upgrades = max(0, oversold - 2)
                ooo = [{"room": str(rng.randint(2, 14) * 100 + rng.randint(1, 40)), "status": "ready_by",
                        "eta": round(rng.uniform(13, 17.5) * 4) / 4} for _ in range(2)]
            elif k == "early_in_day":
                now = round(rng.uniform(10, 13) * 4) / 4
                upgrades = max(0, oversold - rng.randint(2, 4))
            else:
                upgrades, ooo = max(0, oversold - rng.randint(1, 3)), []
                arrivals = rng.randint(60, 120)
                ns_rate = round(rng.uniform(4, 8), 1)
        return {"regime": reg, "hotel": _hotel(rng), "room_type": rtype, "upgrade_type": up,
                "oversold_by": oversold, "free_upgrade_rooms": upgrades, "arrivals_of_type": arrivals,
                "no_show_rate_pct": ns_rate, "ooo_rooms": ooo, "time_now": now,
                "late_checkouts_of_type": rng.randint(0, 5), "duty_manager": person(rng),
                "date": rand_date(rng)}

    def _counts(self, f):
        ready = sum(1 for o in f["ooo_rooms"] if o["eta"] is not None and o["eta"] <= 18 and o["status"] != "parts")
        exp_ns = f["arrivals_of_type"] * f["no_show_rate_pct"] / 100
        eff_ns = math.floor(exp_ns * (0.5 if f["time_now"] < 18 else 1.0))
        return ready, eff_ns

    def intended(self, f):
        ready, eff_ns = self._counts(f)
        o, u = f["oversold_by"], f["free_upgrade_rooms"]
        if o <= u:
            plan = "upgrade_chain"
        elif o <= u + ready:
            plan = "return_ooo_rooms_to_sale"
        elif o <= u + ready + eff_ns:
            plan = "wait_for_no_shows"
        else:
            plan = "arrange_walks_now"
        short = o - u - ready - eff_ns
        proj = 0 if short <= 0 else 1 if short == 1 else 2 if short <= 3 else 3
        hrs = 19 - f["time_now"]
        ttr = 0 if hrs >= 6 else 1 if hrs >= 3 else 2 if hrs >= 1 else 3
        return {"shortfall_plan": plan, "projected_shortfall": proj, "time_to_resolve": ttr,
                "ooo_rooms_sellable_tonight": "yes" if ready >= 1 else "no",
                "walk_unavoidable": "yes" if short >= 1 else "no"}

    def _policy(self):
        return ["Free upgrades go up one category only.",
                "Out-of-order rooms go back on sale only once engineering has signed them off; count only rooms "
                "expected ready by 18:00.",
                "Before 18:00 count at most half the forecast no-shows; after 18:00 count all of them.",
                "Relocations are booked by 19:00 so guests can be told before they arrive."]

    def _ooo_lines(self, f):
        out = []
        for o in f["ooo_rooms"]:
            txt = OOO_STATUS[o["status"]]
            out.append(f"Room {o['room']}: {txt}" + (f" {_hhmm(o['eta'])}" if o["eta"] is not None else ""))
        return out or ["No out-of-order rooms of this type."]

    def render(self, f, rng, surface):
        if surface == "json":
            return dumps(prune({
                "tonight": {"hotel": f["hotel"], "date": f["date"].isoformat(), "time_now": _hhmm(f["time_now"])},
                "oversold_type": {"type": f["room_type"], "oversold_by_rooms": f["oversold_by"],
                                  "arrivals_due": f["arrivals_of_type"], "no_show_rate_pct": f["no_show_rate_pct"],
                                  "approved_late_checkouts": f["late_checkouts_of_type"]},
                "options": {"free_rooms_one_category_up": {f["upgrade_type"]: f["free_upgrade_rooms"]},
                            "out_of_order_rooms": self._ooo_lines(f)},
                "policy": self._policy(),
            }))
        if surface == "front_office_log":
            return "\n".join([f"FRONT OFFICE -- {f['hotel']} -- {fmt_date(f['date'], rng)} {_hhmm(f['time_now'])}",
                              kv_lines([("Oversold type", f"{f['room_type']} by {f['oversold_by']}"),
                                        ("Arrivals due (type)", f["arrivals_of_type"]),
                                        ("No-show rate (type)", f"{f['no_show_rate_pct']}%"),
                                        (f"Free {f['upgrade_type']} rooms", f["free_upgrade_rooms"]),
                                        ("Approved late check-outs", f["late_checkouts_of_type"])], rng),
                              "Out of order:", bullets(self._ooo_lines(f), rng), "", "Standing rules:",
                              bullets(self._policy(), rng)])
        s = [f"Handing over at {_hhmm(f['time_now'])}: {f['room_type']} is oversold by {f['oversold_by']} for tonight "
             f"with {f['arrivals_of_type']} arrivals still due (no-show rate about {f['no_show_rate_pct']}%).",
             f"We have {f['free_upgrade_rooms']} free {f['upgrade_type']} rooms for upgrades.",
             " ".join(x.rstrip(".") + "." for x in self._ooo_lines(f)),
             f"{f['late_checkouts_of_type']} late check-outs were approved on this type."]
        return "\n".join([f"SHIFT HANDOVER -- {f['hotel']} -- {f['duty_manager']}", "", noisy(join_sentences(s), rng, 0.08),
                          "", "Rules:", bullets(self._policy(), rng)])

    def questions(self):
        return {
            "shortfall_plan": choice("Which step is needed to cover tonight's oversold rooms?", {
                "upgrade_chain": "Free upgrades to the next category cover it on their own.",
                "return_ooo_rooms_to_sale": "Free upgrades plus out-of-order rooms returned to sale cover it.",
                "wait_for_no_shows": "In-house options fall short; it rests on expected no-shows, so wait for the "
                                     "checkpoint.",
                "arrange_walks_now": "Even counting no-shows the hotel is short; book relocations now.",
            }),
            "projected_shortfall": score("After every in-house option and the no-shows the rules allow, how short is the hotel likely to be tonight?", [
                "Not short at all.",
                "One room short.",
                "Two or three rooms short.",
                "Four or more rooms short.",
            ]),
            "time_to_resolve": score("How little time is left before the 19:00 relocation checkpoint?", [
                "Plenty: six hours or more.",
                "Some: three to six hours.",
                "Little: one to three hours.",
                "Almost none: under an hour, or already past.",
            ]),
            "ooo_rooms_sellable_tonight": noul(
                "Out-of-order rooms of this type can be counted as sellable for tonight under the rules.",
                yes="At least one is expected repaired and signed off by 18:00.",
                no="None is expected ready by 18:00, or the repair has no firm time."),
            "walk_unavoidable": noul(
                "Even after every option the rules allow, at least one guest will have to be relocated tonight."),
        }


# =========================================================================== schema 5
class WalkNightAudit(Schema):
    name = "walk_night_audit"
    description = ("The audit after a night on which guests were relocated: compliant, compliant with notes, a "
                   "deviation needing coaching, or a breach to escalate.")
    surfaces = {"json": 0.5, "night_audit_report": 0.25, "rm_review_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        rooms = rng.randint(100, 500)
        exp = round(rooms * rng.uniform(0.03, 0.09), 1)
        sd = round(max(1.5, exp * rng.uniform(0.25, 0.5)), 1)
        policy_lim = max(1, math.floor(exp - sd))
        limit = policy_lim + pick(rng, [0, 0, 0, 1, 2, 4])
        miss = rng.uniform(-0.5, 2.5)
        prio_ok, comp, comparable, rewalk, approval = True, "full", True, False, True
        if reg == "clear":
            if chance(rng, 0.5):
                limit, miss = policy_lim, rng.uniform(1.6, 3)
            else:
                limit, miss = policy_lim + rng.randint(4, 8), rng.uniform(0, 1)
                prio_ok, comp, approval = False, pick(rng, ["partial", "none"]), False
        elif reg == "borderline":
            limit = policy_lim + pick(rng, [0, 1])
            miss = near(rng, 1.0, 0.5)
            comp = pick(rng, ["full", "full", "partial"])
        else:
            k = pick(rng, ["good_limit_bad_priority", "bad_limit_good_handling", "rewalk"])
            if k == "good_limit_bad_priority":
                limit, prio_ok = policy_lim, False
                miss = rng.uniform(1.4, 3)
            elif k == "bad_limit_good_handling":
                limit, miss, approval = policy_lim + rng.randint(2, 5), rng.uniform(0, 1), chance(rng, 0.5)
            else:
                rewalk, prio_ok = True, False
                miss = rng.uniform(1.2, 2.5)
        actual = min(max(0, int(round(exp - miss * sd))), limit - 1)   # a night with relocations: wash < limit
        walks = limit - actual
        multi = sum(1 for _ in range(walks) if chance(rng, 0.3))
        if chance(rng, 0.2):
            comparable = False
        return {"regime": reg, "hotel": _hotel(rng), "city": pick(rng, CITIES), "night": rand_date(rng),
                "rooms": rooms, "expected_wash": exp, "forecast_sd": sd, "limit_set": limit,
                "actual_wash": actual, "walks": walks, "walked_multi_night": min(multi, walks),
                "approval_cap": policy_lim + 2, "rm_director_approved": approval,
                "priority_followed": prio_ok, "rewalked_guest": rewalk,
                "kept_lower_priority": pick(rng, ["one-night OTA-prepaid guests", "opaque-rate guests",
                                                  "non-guaranteed late arrivals"]),
                "walked_higher_priority": pick(rng, ["a Gold member", "a corporate-rate guest on a 3-night stay",
                                                     "an award-stay guest"]),
                "compensation": comp, "partner_comparable": comparable, "auditor": person(rng),
                "night_manager_note": pick(rng, ["Arrivals came in all at once after a delayed flight.",
                                                 "Fewer no-shows than on any comparable night this year.",
                                                 "We chose guests who were friendliest about moving.", "", ""])}

    def _policy_limit(self, f):
        return max(1, math.floor(f["expected_wash"] - f["forecast_sd"]))

    def intended(self, f):
        pol = self._policy_limit(f)
        over = f["limit_set"] > pol
        needs_appr = f["limit_set"] > f["approval_cap"]
        appr_gap = needs_appr and not f["rm_director_approved"]
        miss_sd = abs(f["expected_wash"] - f["actual_wash"]) / f["forecast_sd"]
        miss = 0 if miss_sd < 0.5 else 1 if miss_sd < 1.2 else 2 if miss_sd < 2.2 else 3
        pts = (1 if f["walks"] >= 3 else 0) + (1 if f["walked_multi_night"] else 0) \
            + {"full": 0, "partial": 1, "none": 2}[f["compensation"]] + (0 if f["partner_comparable"] else 1) \
            + (1 if f["rewalked_guest"] else 0)
        impact = 0 if pts == 0 else 1 if pts == 1 else 2 if pts <= 3 else 3
        prio = f["priority_followed"] and not f["rewalked_guest"]
        if f["rewalked_guest"] or (appr_gap and f["compensation"] == "none") or (over and not prio and f["compensation"] != "full"):
            finding = "breach_escalate"
        elif over or not prio or appr_gap:
            finding = "deviation_needs_coaching"
        elif f["compensation"] != "full" or not f["partner_comparable"]:
            finding = "compliant_with_notes"
        else:
            finding = "compliant"
        return {"audit_finding": finding, "forecast_miss_size": miss, "guest_impact_severity": impact,
                "walks_were_avoidable": "yes" if over else "no", "priority_order_followed": "yes" if prio else "no"}

    def _policy(self, f):
        return ["Overbooking limit: expected no-shows plus late cancellations, minus one forecast standard deviation.",
                f"A limit above {f['approval_cap']} rooms needs the revenue director's approval.",
                "Relocate in the order of the guest-priority list; never relocate a guest we relocated in the last 12 months.",
                "Every relocated guest gets the full package: first night paid at a comparable partner hotel, "
                "transport both ways, one call."]

    def _facts(self, f):
        s = [f"Limit set: {f['limit_set']} rooms (revenue director approval: {'yes' if f['rm_director_approved'] else 'no'}).",
             f"Forecast at the time: {f['expected_wash']} rooms of no-shows and late cancellations, SD {f['forecast_sd']}.",
             f"On the night: {f['actual_wash']} rooms washed; {f['walks']} guest(s) relocated, "
             f"{f['walked_multi_night']} of them on multi-night stays."]
        if not f["priority_followed"]:
            s.append(f"{f['walked_higher_priority'].capitalize()} was relocated while {f['kept_lower_priority']} were kept.")
        if f["rewalked_guest"]:
            s.append("One relocated guest had also been relocated by us four months earlier.")
        s.append({"full": "All relocated guests received the full package.",
                  "partial": "Transport back was not arranged for every relocated guest.",
                  "none": "No package was given; guests were sent to the partner on their own."}[f["compensation"]])
        s.append("The partner hotel was comparable." if f["partner_comparable"]
                 else "The partner hotel was rated a level below ours.")
        return s

    def render(self, f, rng, surface):
        facts = self._facts(f)
        note = f["night_manager_note"]
        if surface == "json":
            return dumps(prune({
                "night": {"hotel": f["hotel"], "city": f["city"], "date": f["night"].isoformat(), "rooms": f["rooms"]},
                "overbooking": {"limit_set": f["limit_set"], "revenue_director_approved": f["rm_director_approved"],
                                "forecast_wash_rooms": f["expected_wash"], "forecast_sd_rooms": f["forecast_sd"],
                                "actual_wash_rooms": f["actual_wash"]},
                "relocations": {"guests_relocated": f["walks"], "multi_night_stays": f["walked_multi_night"],
                                "findings": facts[3:]},
                "night_manager_note": note or None, "policy": self._policy(f),
            }))
        if surface == "night_audit_report":
            parts = [f"NIGHT AUDIT -- RELOCATIONS  {f['hotel']}, {f['city']}  {fmt_date(f['night'], rng)}",
                     f"Auditor: {f['auditor']}", "", bullets(facts, rng), ""]
            if note:
                parts += [f"Night manager: {note}", ""]
            return "\n".join(parts + ["Policy:", bullets(self._policy(f), rng)])
        return "\n".join([f"MEMO -- review of relocations on {fmt_date(f['night'], rng)} ({f['hotel']})", "",
                          noisy(join_sentences(facts + ([f"The night manager wrote: \"{note}\""] if note else [])), rng, 0.08),
                          "", "Policy:", bullets(self._policy(f), rng)])

    def questions(self):
        return {
            "audit_finding": choice("What should the audit conclude about how the night was handled?", {
                "compliant": "Handled within policy; nothing to act on.",
                "compliant_with_notes": "Within policy, with minor points to note.",
                "deviation_needs_coaching": "A departure from policy that calls for coaching.",
                "breach_escalate": "A breach of policy to escalate to the general manager.",
            }),
            "forecast_miss_size": score("How far did the no-show and cancellation forecast miss on the night?", [
                "Hardly at all.",
                "Within normal error.",
                "Well outside normal error.",
                "Far outside anything the forecast allowed for.",
            ]),
            "guest_impact_severity": score("How heavily were relocated guests affected?", [
                "Lightly.",
                "Moderately.",
                "Heavily.",
                "Severely.",
            ]),
            "walks_were_avoidable": noul(
                "A limit set within policy would have avoided some or all of the relocations.",
                yes="The limit was set above the level the policy allowed on the forecast at the time.",
                no="The limit was within policy; the relocations came from the forecast missing on the night."),
            "priority_order_followed": noul(
                "The guests relocated were the ones the priority rules put first."),
        }


WORKFLOW = Workflow(
    name="hotel_overbooking_and_walk",
    title="Hotel overbooking and walks",
    description="How far a hotel oversells a night and which reservation it relocates when oversold, against written "
                "revenue-management and guest-priority rules.",
    domain="accommodation / hotel revenue management and front office",
    schemas=[OverbookingLimitSetting(), WalkSelectionTonight(), GroupBlockWashReview(), OversoldRoomTypePlan(),
             WalkNightAudit()],
    notes="Fictional hotels. Guest priority uses loyalty tier, rate type, stay length, arrival time, guarantee and past "
          "relocations only. Operations records, not guest conversations; no complaint handling.",
)
