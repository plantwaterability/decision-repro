"""Property management operations: process decisions on residential buildings and leases, judged against a
(fictional) maintenance standard, tenancy code, rent ordinance, lease clauses, insurance conditions and management
agreement.

Five schemas: a maintenance work order to prioritise and route, a move-out security-deposit deduction list to check
against the move-in record and the code, a lease-renewal rent increase to check against written caps and notice
rules, a common-area incident to handle, and contractor bids to award under the management agreement. Artefacts are
work-order tickets, resident portal messages, deduction statements, renewal worksheets, incident reports, manager
emails and bid tabulations.

There is NO tenant screening here (tenant_screening covers applications). Residents and owners appear as role IDs
or invented names, and no decision turns on who a resident is: answers follow from the building, the record and
the written rule. Codes, ordinances, buildings, companies and people are fictional; thresholds are sampled per case
and only the governing clauses are quoted.

Each schema also has `intended(f)`: the designer's reading of the rule for that case. It is never rendered and
never a label; it exists so the answer-spread check can confirm every question moves with the factors.
"""
from __future__ import annotations

from datetime import date, timedelta

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
    ident,
    join_sentences,
    kv_lines,
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
    table,
)

BLDG_A = ["Alderley", "Bexhill", "Carraway", "Dunstan", "Ellery", "Foxglove", "Greyfriars", "Hawthorn Row", "Ivybridge",
          "Juniper Hill", "Kingsbury", "Linden", "Maple Terrace", "Norwood", "Oriel", "Primrose", "Quayside", "Rosemont",
          "Sefton", "Thistle", "Union Mill", "Verity", "Willowmere", "Yewtree"]
BLDG_B = ["Apartments", "Court", "House", "Lofts", "Gardens", "Residences", "Commons"]
MGMT = ["Cornerstone Residential", "Hearthside Property Management", "Lantern Realty Services", "Brookfield Lane Management",
        "Parapet Property Services", "Gable & Rowe Management"]
VENDOR_A = ["Ardent", "Beacon", "Crestline", "Dependable", "Evergreen", "Foursquare", "Guardian", "Highmark", "Ironwood",
            "Keystroke", "Landmark", "Mainstay", "Northpoint", "Overland", "Precision", "Reliance", "Summit Ridge", "Trueline"]


def _bldg(rng):
    return f"{pick(rng, BLDG_A)} {pick(rng, BLDG_B)}"


def _unit(rng):
    return f"{rng.randint(1, 12)}{pick(rng, 'ABCDEF')}"


def _resident(rng):
    return f"Resident {ident(rng, 'R', 4, '-')}"


def _first(name):
    return name.split()[0]


def _kv(pairs, rng, skip_none=True):
    """kv_lines without the snake_case 'equals' house style, which mangles labels such as 'Site / sector'."""
    return kv_lines(pairs, rng, style=pick(rng, ["colon", "colon", "aligned", "dash"]), skip_none=skip_none)


def _date_in(rng, months):
    return date(pick(rng, [2025, 2026, 2027]), pick(rng, months), rng.randint(1, 28))


def _usd(x, rng):
    return money(round(x), rng, style=pick(rng, ["sym", "sym", "code"]))


COLD = [11, 12, 1, 2, 3]
MILD = [4, 5, 9, 10]
WARM = [6, 7, 8]


# =========================================================================== schema 1
# issue -> (text, base class, resource)
ISSUES = {
    "no_heat": ("No heat in the unit", None, "outside_specialist_vendor"),
    "active_leak": ("Water coming through the ceiling", None, "superintendent_first_check"),
    "no_hot_water": ("No hot water", "urgent", None),
    "entry_lock": ("Entry door lock broken", None, "in_house_technician"),
    "smoke_alarm": ("Smoke alarm problem", None, "in_house_technician"),
    "gas_odour": ("Smell of gas in the kitchen", "emergency", "utility_emergency_line"),
    "fridge": ("Refrigerator not cooling", "urgent", "outside_specialist_vendor"),
    "toilet_clog": ("Toilet clogged", None, "in_house_technician"),
    "dishwasher": ("Dishwasher not draining", "routine", "in_house_technician"),
    "mold": ("Mold spots on bathroom ceiling", "routine", "superintendent_first_check"),
    "cabinet": ("Kitchen cabinet door hanging off", "cosmetic", "in_house_technician"),
    "paint": ("Scuffed and chipped paint in hallway of unit", "cosmetic", "in_house_technician"),
    "breaker": ("Breaker keeps tripping, outlets dead in one room", None, "outside_specialist_vendor"),
    "window_crack": ("Cracked window pane", "routine", "in_house_technician"),
}
CLASS_ORDER = ["cosmetic", "routine", "urgent", "emergency"]


class WorkOrderPriority(Schema):
    name = "work_order_priority"
    description = "A resident's maintenance request to be classed and routed under the company maintenance standard."
    surfaces = {"json": 0.5, "work_order": 0.25, "portal_message": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        issue = pick_w(rng, {i: (2.0 if i == "gas_odour" else 1.0) for i in ISSUES})
        H = pick(rng, [18, 20])
        E = pick(rng, [2, 4])
        U = pick(rng, [24, 48])
        R = pick(rng, [5, 7])
        d = {"indoor_c": None, "outdoor_c": None, "spreading": None, "near_electrical": None, "second_bath": None,
             "alarm_state": None, "burning_smell": None, "lock_state": None, "building_wide": False}
        months = COLD + MILD if issue == "no_heat" else COLD + MILD + WARM
        day = _date_in(rng, months)
        if issue == "no_heat":
            out = rng.randint(-12, 8) if day.month in COLD else rng.randint(4, 16)
            d.update(outdoor_c=out, indoor_c=rng.randint(11, 21), building_wide=chance(rng, 0.3))
        elif issue == "active_leak":
            d.update(spreading=chance(rng, 0.45), near_electrical=chance(rng, 0.3))
        elif issue == "toilet_clog":
            d.update(second_bath=chance(rng, 0.4))
        elif issue == "smoke_alarm":
            d.update(alarm_state=pick(rng, ["chirping (low battery)", "removed from ceiling", "not sounding on test"]))
        elif issue == "breaker":
            d.update(burning_smell=chance(rng, 0.3))
        elif issue == "entry_lock":
            d.update(lock_state=pick(rng, ["will not lock", "key sticks but locks"]))
        elif issue == "no_hot_water":
            d.update(building_wide=chance(rng, 0.35))
        tone = pick(rng, ["calm", "urgent", "plain"])
        repeat = pick_w(rng, {0: 6, 1: 2, 2: 1, 3: 0.7})
        k = None
        if reg == "clear":
            repeat = 0
            tone = "plain"
        elif reg == "borderline":
            k = "near_rule"
            if issue == "no_heat":
                d.update(indoor_c=near(rng, H, 1.5, integer=True))
        else:
            k = pick(rng, ["tone_mismatch", "repeat_job"])
            if k == "tone_mismatch":
                cls = self._class(issue, d, H)
                tone = "urgent" if cls in ("cosmetic", "routine") else "calm"
            else:
                repeat = pick(rng, [2, 3])
        hours = round(rng.uniform(0.05, 1.4) * {"emergency": E, "urgent": U, "routine": R * 24, "cosmetic": 30 * 24}[self._class(issue, d, H)], 1)
        if reg == "borderline" and chance(rng, 0.5):
            target = {"emergency": E, "urgent": U, "routine": R * 24, "cosmetic": 30 * 24}[self._class(issue, d, H)]
            hours = near(rng, target, target * 0.15, lo=0.2, digits=1)
        return {"regime": reg, "case_kind": k, "issue": issue, **d, "heat_min_c": H, "emerg_h": E, "urgent_h": U,
                "routine_days": R, "tone": tone, "repeat_60d": repeat, "hours_elapsed": hours,
                "permission": pick_w(rng, {"granted": 4, "not given": 2, "not asked": 3}),
                "building": _bldg(rng), "unit": _unit(rng), "resident": _resident(rng), "date": day,
                "wo": ident(rng, "WO", 6, "-"), "coordinator": person(rng), "mgmt": pick(rng, MGMT),
                "std": pick(rng, ["MS-2", "Maintenance Standard 4.1", "Service Levels Policy SL-3"])}

    @staticmethod
    def _class(issue, d, H):
        base = ISSUES[issue][1]
        if base:
            return base
        if issue == "no_heat":
            return "emergency" if (d["indoor_c"] < H and d["outdoor_c"] < 10) else "urgent"
        if issue == "active_leak":
            return "emergency" if (d["spreading"] or d["near_electrical"]) else "urgent"
        if issue == "entry_lock":
            return "emergency" if d["lock_state"] == "will not lock" else "routine"
        if issue == "smoke_alarm":
            return "urgent" if d["alarm_state"].startswith("chirping") else "emergency"
        if issue == "breaker":
            return "emergency" if d["burning_smell"] else "urgent"
        if issue == "toilet_clog":
            return "routine" if d["second_bath"] else "urgent"
        return "routine"

    def intended(self, f):
        cls = self._class(f["issue"], f, f["heat_min_c"])
        res = ISSUES[f["issue"]][2]
        if f["issue"] == "no_hot_water":
            res = "outside_specialist_vendor" if f["building_wide"] else "in_house_technician"
        if f["repeat_60d"] >= 2 and res == "in_house_technician":
            res = "outside_specialist_vendor"
        hab = {"cosmetic": 0, "routine": 1, "urgent": 2, "emergency": 3}[cls]
        if f["issue"] in ("entry_lock", "smoke_alarm", "gas_odour") and hab == 3:
            hab = 2  # a safety emergency, but the home itself is still livable
        if f["issue"] == "mold":
            hab = 1
        if f["building_wide"] and hab < 3:
            hab += 1
        target = {"emergency": f["emerg_h"], "urgent": f["urgent_h"], "routine": f["routine_days"] * 24,
                  "cosmetic": 30 * 24}[cls]
        r = f["hours_elapsed"] / target
        dl = 0 if r < 0.25 else 1 if r < 0.75 else 2 if r < 1 else 3
        return {"priority_class": cls, "assigned_resource": res, "habitability_impact": hab, "deadline_pressure": dl,
                "entry_without_notice_permitted": "yes" if (cls == "emergency" or f["permission"] == "granted") else "no"}

    def _detail(self, f):
        i = f["issue"]
        if i == "no_heat":
            return (f"thermostat reads {f['indoor_c']} C inside, {f['outdoor_c']} C outside"
                    + ("; several units on the same riser report the same" if f["building_wide"] else ""))
        if i == "active_leak":
            return ("spreading across the ceiling" if f["spreading"] else "slow drip into a bucket") + (
                ", dripping onto a light fixture" if f["near_electrical"] else "")
        if i == "toilet_clog":
            return "second bathroom works" if f["second_bath"] else "only toilet in the unit"
        if i == "smoke_alarm":
            return f["alarm_state"]
        if i == "breaker":
            return "faint burning smell near the panel" if f["burning_smell"] else "no smell, no scorch marks"
        if i == "entry_lock":
            return f["lock_state"]
        if i == "no_hot_water":
            return "whole building affected" if f["building_wide"] else "only this unit"
        return None

    def _rule(self, f):
        return (f"{f['std']}: emergency (life, safety, no heat below {f['heat_min_c']} C when under 10 C outside, active "
                f"leak spreading or near electrics, unit cannot be secured) {f['emerg_h']} h; urgent {f['urgent_h']} h; "
                f"routine {f['routine_days']} days; cosmetic at next scheduled visit. Entry needs notice unless the "
                "resident consents or it is an emergency")

    def _msg(self, f, rng):
        base = ISSUES[f["issue"]][0]
        det = self._detail(f)
        core = base + (f" ({det})" if det else "")
        if f["tone"] == "urgent":
            t = f"{core}!! Please send someone ASAP, this is unacceptable."
        elif f["tone"] == "calm":
            t = f"Hi, just letting you know: {core[0].lower() + core[1:]}. No rush, whenever someone is around."
        else:
            t = f"{core}."
        if f["repeat_60d"]:
            t += f" This is the {['', 'second', 'third', 'fourth'][f['repeat_60d']]} time in two months."
        return noisy(t, rng, 0.15)

    def render(self, f, rng, surface):
        msg = self._msg(f, rng)
        if surface == "json":
            return dumps(prune({
                "request": {"work_order": f["wo"], "building": f["building"], "unit": f["unit"], "submitted_by": f["resident"],
                            "category": f["issue"].replace("_", " "), "resident_message": msg,
                            "submitted": f["date"].isoformat(), "hours_open": f["hours_elapsed"]},
                "history": {"same_issue_last_60_days": f["repeat_60d"]},
                "access": {"permission_to_enter": f["permission"]},
                "standard": self._rule(f),
            }))
        if surface == "work_order":
            body = _kv([("WO", f["wo"]), ("Property", f"{f['building']} #{f['unit']}"), ("Requested by", f["resident"]),
                             ("Category", f["issue"].replace("_", " ")), ("Details", self._detail(f)),
                             ("Opened", f"{fmt_date(f['date'], rng)}, {f['hours_elapsed']} h ago"),
                             ("Repeat (60 d)", f["repeat_60d"]), ("Permission to enter", f["permission"]),
                             ("Status", "unassigned")], rng)
            return f"WORK ORDER -- {f['mgmt']}\n{body}\nResident wrote: \"{msg}\"\nPolicy: {self._rule(f)}"
        return "\n".join([f"Resident portal -- {f['building']}, unit {f['unit']}", f"From: {f['resident']}  ({fmt_date(f['date'], rng)})",
                          f"Subject: {ISSUES[f['issue']][0]}", "", msg, "",
                          f"[staff note, {_first(f['coordinator'])}] open {f['hours_elapsed']} h; permission to enter: {f['permission']}; "
                          f"same issue {f['repeat_60d']}x in 60 days.", f"[{self._rule(f)}]"])

    def questions(self):
        return {
            "priority_class": choice("Which priority class does the maintenance standard put this request in?", {
                "cosmetic": "Cosmetic: fix at the next scheduled visit or turnover.",
                "emergency": "Emergency: respond within the standard's emergency window.",
                "routine": "Routine: schedule within the routine window.",
                "urgent": "Urgent: respond within the urgent window.",
            }),
            "assigned_resource": choice("Who should be sent to this request first?", {
                "in_house_technician": "The in-house maintenance technician.",
                "outside_specialist_vendor": "An outside specialist (licensed trade or appliance service).",
                "superintendent_first_check": "The superintendent, to find the source before anyone is dispatched.",
                "utility_emergency_line": "The utility's emergency line, while the unit is cleared.",
            }),
            "habitability_impact": score("How much does the problem affect the home's livability?", [
                "None: purely cosmetic.",
                "Minor: an inconvenience, the home works normally.",
                "Significant: an essential service or security is impaired.",
                "Severe: the home is unsafe or unlivable until fixed.",
            ]),
            "deadline_pressure": score("How close is the request to missing its response target?", [
                "Plenty of time: most of the window remains.",
                "On track: a good part of the window remains.",
                "Tight: the window is nearly used up.",
                "Missed: the response target has already passed.",
            ]),
            "entry_without_notice_permitted": noul(
                "Staff may enter the unit without giving the usual advance notice.",
                yes="The resident has consented to entry, or the problem is an emergency under the standard.",
                no="There is no consent on record and the problem is not an emergency."),
        }


# =========================================================================== schema 2
# item -> (label, damage text, wear text, cost range, depreciable kind)
ITEMS = {
    "carpet": ("carpet replacement, living room", "pet urine stains and odour through to the pad", "traffic-lane wear",
               (900, 2400), "carpet"),
    "paint": ("full interior repaint", "crayon and marker on two walls", "faded paint and light scuffs", (450, 1500), "paint"),
    "cleaning": ("cleaning", "oven, fridge and bathroom left heavily soiled", "light dust on shelves", (150, 450), None),
    "holes": ("drywall repair", "fist-sized hole in bedroom wall", "a few small nail holes from pictures", (80, 400), None),
    "blinds": ("blind replacement", "bent and snapped slats", "sun-faded blinds", (60, 250), None),
    "door": ("bedroom door replacement", "door off its hinges, frame split", "door rubs slightly", (250, 650), None),
}


class DepositDeductionReview(Schema):
    name = "deposit_deduction_review"
    description = "A move-out deduction list from a security deposit, checked against the move-in record and the tenancy code."
    surfaces = {"json": 0.5, "deduction_statement": 0.25, "manager_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        N = pick(rng, [14, 21, 30])
        life_c = pick(rng, [5, 7, 10])
        life_p = pick(rng, [2, 3, 5])
        n = rng.randint(2, 4)
        keys = rng.sample(list(ITEMS), n)
        tenancy = rng.randint(1, 9)
        items = []
        for key in keys:
            lo, hi = ITEMS[key][3]
            items.append({"key": key, "amount": int(rng.uniform(lo, hi)),
                          "nature": pick_w(rng, {"damage": 5, "wear": 3}),
                          "move_in": pick_w(rng, {"good": 6, "noted same issue": 1.3, "not recorded": 2}),
                          "photos": chance(rng, 0.7), "age_years": tenancy + rng.randint(0, 5) if ITEMS[key][4] else None})
        days = rng.randint(3, N + 6)
        k = None
        if reg == "clear":
            days = rng.randint(3, N - 4)
            tenancy = pick(rng, [1, 1, 2])
            for it in items:
                it.update(nature="damage", move_in="good", photos=True)
                if it["age_years"] is not None:
                    it["age_years"] = tenancy
            if chance(rng, 0.4):
                days = N + rng.randint(3, 12)
        elif reg == "borderline":
            k = pick(rng, ["deadline", "depreciation"])
            if k == "deadline":
                days = near(rng, N, 2, lo=1, integer=True)
            else:
                dep = [it for it in items if it["age_years"] is not None]
                if dep:
                    it = dep[0]
                    life = life_c if it["key"] == "carpet" else life_p
                    it.update(nature="damage", age_years=max(1, near(rng, life, 1, integer=True)))
                    tenancy = min(tenancy, it["age_years"])
        else:
            k = pick(rng, ["wear_billed", "no_move_in_record", "pre_existing"])
            biggest = max(items, key=lambda it: it["amount"])
            if k == "wear_billed":
                (biggest if chance(rng, 0.5) else pick(rng, items))["nature"] = "wear"
            elif k == "no_move_in_record":
                biggest.update(nature="damage", move_in="not recorded", photos=False)
            else:
                biggest.update(move_in="noted same issue")
        unpaid = pick(rng, [0, 0, 0, rng.randint(200, 1800)])
        move_out = rand_date(rng)
        return {"regime": reg, "case_kind": k, "items": items, "unpaid_rent": unpaid,
                "deposit": pick(rng, [800, 1000, 1200, 1500, 1800, 2400, 3000]), "move_out": move_out,
                "statement_days": days, "deadline_days": N, "life_carpet": life_c, "life_paint": life_p,
                "tenancy_years": tenancy, "former_resident": f"Former resident {ident(rng, 'FR', 4, '-')}",
                "building": _bldg(rng), "unit": _unit(rng), "manager": person(rng), "supervisor": person(rng),
                "mgmt": pick(rng, MGMT), "code": pick(rng, ["RTC s.12", "Residential Tenancy Code 12", "Tenancy Deposit Rule 4"])}

    def _allow(self, f, it):
        if it["nature"] == "wear" or it["move_in"] == "noted same issue":
            return 0.0
        kind = ITEMS[it["key"]][4]
        if kind:
            life = f["life_carpet"] if kind == "carpet" else f["life_paint"]
            return it["amount"] * max(0.0, 1 - it["age_years"] / life)
        return float(it["amount"])

    def intended(self, f):
        met = f["statement_days"] <= f["deadline_days"]
        items = f["items"]
        biggest = max(items, key=lambda it: it["amount"])
        weak_big = biggest["move_in"] == "not recorded" and not biggest["photos"] and biggest["nature"] == "damage"
        wear = any(it["nature"] == "wear" or it["move_in"] == "noted same issue" for it in items)
        unprorated = any(ITEMS[it["key"]][4] and it["nature"] == "damage" and it["move_in"] != "noted same issue"
                         and self._allow(f, it) < 0.85 * it["amount"] for it in items)
        if not met:
            ruling = "return_full_deposit"
        elif weak_big:
            ruling = "request_evidence"
        elif wear:
            ruling = "remove_wear_items"
        elif unprorated:
            ruling = "reduce_to_prorated"
        else:
            ruling = "approve_as_proposed"
        good = sum(1 for it in items if it["move_in"] != "not recorded" and it["photos"])
        frac = good / len(items)
        ev = 3 if frac == 1 else 2 if frac >= 0.6 else 1 if frac >= 0.3 else 0
        claimed = sum(it["amount"] for it in items)
        allowed = sum(self._allow(f, it) for it in items)
        over = (claimed - allowed) / claimed
        oc = 0 if over < 0.05 else 1 if over < 0.25 else 2 if over < 0.6 else 3
        return {"deduction_ruling": ruling, "evidence_strength": ev, "overcharge_extent": oc,
                "statement_deadline_met": "yes" if met else "no",
                "largest_item_is_damage": "yes" if (biggest["nature"] == "damage" and biggest["move_in"] != "noted same issue") else "no"}

    def _rule(self, f):
        return (f"{f['code']}: itemised statement within {f['deadline_days']} days of move-out or the whole deposit is "
                f"returned; no charge for normal wear or conditions noted at move-in; carpet prorated over "
                f"{f['life_carpet']} years, paint over {f['life_paint']} years")

    def _desc(self, it):
        return ITEMS[it["key"]][1 if it["nature"] == "damage" else 2]

    def _row(self, f, it, rng):
        age = f", {it['age_years']} yr{'s' if it['age_years'] != 1 else ''} old" if it["age_years"] else ""
        return [ITEMS[it["key"]][0], self._desc(it) + age, _usd(it["amount"], rng),
                it["move_in"], "yes" if it["photos"] else "no"]

    def render(self, f, rng, surface):
        sent = f["move_out"] + timedelta(days=f["statement_days"])
        if surface == "json":
            return dumps(prune({
                "tenancy": {"unit": f"{f['building']} #{f['unit']}", "resident": f["former_resident"],
                            "years": f["tenancy_years"], "deposit_usd": f["deposit"], "move_out": f["move_out"].isoformat(),
                            "statement_to_be_sent": sent.isoformat()},
                "proposed_deductions": [{"item": ITEMS[it["key"]][0], "finding": self._desc(it), "amount_usd": it["amount"],
                                         "item_age_years": it["age_years"], "move_in_report": it["move_in"],
                                         "move_out_photos": it["photos"]} for it in f["items"]],
                "unpaid_rent_usd": f["unpaid_rent"] or None,
                "code": self._rule(f),
            }))
        if surface == "deduction_statement":
            rows = [self._row(f, it, rng) for it in f["items"]]
            return "\n".join([
                f"DRAFT DEPOSIT STATEMENT -- {f['mgmt']}",
                f"{f['building']} #{f['unit']} / {f['former_resident']} / moved out {fmt_date(f['move_out'], rng)}; "
                f"deposit {_usd(f['deposit'], rng)}; statement to go out {fmt_date(sent, rng)}",
                table(["item", "finding", "amount", "move-in report", "photos"], rows, rng),
                (f"Unpaid rent: {_usd(f['unpaid_rent'], rng)}" if f["unpaid_rent"] else "No unpaid rent."),
                self._rule(f)])
        lines = [f"- {ITEMS[it['key']][0]}: {self._desc(it)}"
                 + (f" ({it['age_years']} yr{'s' if it['age_years'] != 1 else ''} old)" if it["age_years"] else "") + f", {_usd(it['amount'], rng)}; move-in report: "
                 f"{it['move_in']}; " + ("photos taken" if it["photos"] else "no photos") for it in f["items"]]
        return "\n".join([f"Subject: Deposit deductions -- {f['building']} #{f['unit']}", f"From: {f['manager']}",
                          f"To: {f['supervisor']}", "",
                          f"{f['former_resident']} moved out on {fmt_date(f['move_out'], rng)} after {f['tenancy_years']} year(s); "
                          f"deposit {_usd(f['deposit'], rng)}. I plan to send the statement on {fmt_date(sent, rng)}. Proposed:",
                          "\n".join(lines), (f"Plus unpaid rent of {_usd(f['unpaid_rent'], rng)}." if f["unpaid_rent"] else ""),
                          "", f"Code: {self._rule(f)}."]).replace("\n\n\n", "\n\n")

    def questions(self):
        return {
            "deduction_ruling": choice("What should the reviewer do with the proposed deductions?", {
                "approve_as_proposed": "Approve the deductions as listed.",
                "reduce_to_prorated": "Cut replacement charges down to the item's remaining useful life.",
                "remove_wear_items": "Strike the charges for normal wear or conditions already there at move-in.",
                "request_evidence": "Ask for move-in records or photos before any charge is made.",
                "return_full_deposit": "Return the whole deposit; the code no longer allows deductions.",
            }),
            "evidence_strength": score("How well does the record support the charges?", [
                "Poorly: most charges have no move-in record or photos.",
                "Partly: a few charges are documented.",
                "Mostly: most charges are documented.",
                "Fully: every charge has a move-in record and photos.",
            ]),
            "overcharge_extent": score("Setting the statement deadline aside, how much of the proposed total goes beyond "
                                       "what the code allows for these items?", [
                "Nothing or almost nothing.",
                "A modest part.",
                "A large part.",
                "Most or all of it.",
            ]),
            "statement_deadline_met": noul(
                "The itemised statement goes out within the code's deadline.",
                yes="The planned send date is within the deadline counted from move-out.",
                no="The planned send date is after the deadline."),
            "largest_item_is_damage": noul("The largest proposed charge is for damage beyond normal wear and tear."),
        }


# =========================================================================== schema 3
FEES = {"amenity fee": True, "trash valet fee": True, "optional parking space": False, "pest control fee": True}


class RenewalRentIncrease(Schema):
    name = "renewal_rent_increase_check"
    description = "A drafted lease-renewal offer checked against the rent ordinance, the lease's own cap and the notice rules."
    surfaces = {"json": 0.5, "renewal_worksheet": 0.25, "leasing_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        X = pick(rng, [3, 4, 5, 7])
        Y = pick(rng, [1995, 2005, 2010])
        built = rng.randint(1962, 2024)
        lease_cap = pick_w(rng, {None: 3.5, 3: 1, 5: 1.5, 6: 1})
        pt = pick_w(rng, {"none": 5, "pending": 2, "approved": 2.5})
        P = pick(rng, [1.5, 2.0, 3.0])
        D = pick(rng, [60, 90])
        current = int(round(rng.uniform(950, 3400), -1))
        fee = maybe(rng, pick(rng, list(FEES)), 0.65)
        fee_amt = pick(rng, [25, 35, 50, 75]) if fee else None
        pct = round(rng.uniform(1.0, 12.0), 1)
        months_since = rng.randint(10, 26)
        notice = rng.randint(D - 30, D + 45)
        days_to_deadline = rng.randint(-10, 60)
        k = None
        if reg == "clear":
            if chance(rng, 0.5):
                built, lease_cap, pt, fee = pick(rng, [Y - rng.randint(5, 30), Y + rng.randint(1, 12)]), None, "none", None
                pct, notice, months_since = round(rng.uniform(1.0, X - 1.0), 1), D + rng.randint(5, 40), rng.randint(12, 24)
            else:
                built, pct = Y - rng.randint(5, 30), round(X + rng.uniform(3, 8), 1)
        elif reg == "borderline":
            k = pick(rng, ["cap_edge", "notice_edge", "fee_tips_over"])
            if k == "cap_edge":
                pct = near(rng, X, 0.6, lo=0.5, digits=1)
                built = Y - rng.randint(1, 30)
            elif k == "notice_edge":
                notice = near(rng, D, 4, lo=10, integer=True)
                days_to_deadline = rng.randint(-5, 10)
            else:
                built, fee = Y - rng.randint(1, 30), pick(rng, ["amenity fee", "trash valet fee"])
                fee_amt = pick(rng, [35, 50, 75])
                pct = round(X - rng.uniform(0.2, 1.2), 1)
        else:
            k = pick(rng, ["exempt_building_high", "lease_cap_lower", "pass_through_pending", "twelve_month"])
            if k == "exempt_building_high":
                built, lease_cap, pct = Y + rng.randint(0, 12), None, round(rng.uniform(8, 14), 1)
            elif k == "lease_cap_lower":
                built, lease_cap = Y - rng.randint(1, 30), 3
                pct = round(rng.uniform(3.3, X if X > 3.4 else 4.0), 1)
            elif k == "pass_through_pending":
                built, pt = Y - rng.randint(1, 30), "pending"
                pct = round(X + rng.uniform(0.5, P), 1)
            else:
                months_since, pct = rng.randint(7, 11), round(rng.uniform(1.0, X), 1)
        proposed = int(round(current * (1 + pct / 100)))
        lease_end = _date_in(rng, list(range(1, 13)))
        # a notice cannot be served before today: the longest possible notice is the rule period plus the days left
        notice = min(notice, D + days_to_deadline)
        return {"regime": reg, "case_kind": k, "built": built, "ord_cap": X, "ord_year": Y, "lease_cap": lease_cap,
                "pass_through": pt, "pt_pct": P, "notice_days_rule": D, "current": current, "proposed": proposed,
                "fee": fee, "fee_amt": fee_amt, "months_since_last": months_since, "notice_days": notice,
                "days_to_notice_deadline": days_to_deadline, "lease_end": lease_end, "building": _bldg(rng),
                "unit": _unit(rng), "resident": _resident(rng), "leasing": person(rng), "manager": person(rng),
                "market_view": pick(rng, ["supports", "none", "none"]), "mgmt": pick(rng, MGMT),
                "ord": pick(rng, ["Rent Stabilisation Ordinance s4", "City Rent Ordinance 9.2", "RSO 14"])}

    def _caps(self, f):
        ord_applies = f["built"] < f["ord_year"]
        ord_eff = None
        if ord_applies:
            ord_eff = f["ord_cap"] + (f["pt_pct"] if f["pass_through"] == "approved" else 0)
        return ord_applies, ord_eff

    def _total_pct(self, f):
        extra = f["fee_amt"] if (f["fee"] and FEES[f["fee"]]) else 0
        return (f["proposed"] + extra - f["current"]) / f["current"] * 100

    def intended(self, f):
        ord_applies, ord_eff = self._caps(f)
        lc = f["lease_cap"]
        if not ord_applies and lc is None:
            gov, cap = "no_written_cap", None
        elif lc is not None and (not ord_applies or lc < ord_eff):
            gov, cap = "lease_clause_cap", lc
        elif f["pass_through"] == "approved":
            gov, cap = "ordinance_cap_plus_pass_through", ord_eff
        else:
            gov, cap = "ordinance_cap", ord_eff
        tot = self._total_pct(f)
        within = cap is None or tot <= cap + 1e-9
        notice_ok = f["notice_days"] >= f["notice_days_rule"] and f["months_since_last"] >= 12
        if not within and f["pass_through"] == "pending" and gov == "ordinance_cap":
            act = "await_pass_through_ruling"
        elif not within:
            act = "reduce_to_cap"
        elif not notice_ok:
            act = "reissue_with_later_effective_date"
        else:
            act = "send_as_drafted"
        gap = 0
        if not within:
            gap += 1 + (1 if tot - cap > 2 else 0)
        if f["notice_days"] < f["notice_days_rule"]:
            gap += 1
        if f["months_since_last"] < 12:
            gap += 1
        dd = f["days_to_notice_deadline"]
        tp = 0 if dd > 30 else 1 if dd > 7 else 2 if dd >= 0 else 3
        return {"renewal_offer_action": act, "governing_limit": gov, "compliance_gap": min(3, gap),
                "notice_timing_pressure": tp, "increase_within_cap": "yes" if within else "no"}

    def _rule(self, f):
        s = (f"{f['ord']}: buildings first occupied before {f['ord_year']} capped at {f['ord_cap']}% per renewal "
             f"(+ any Rent Board-approved capital pass-through); mandatory recurring fees count as rent; one increase per "
             f"12 months; {f['notice_days_rule']} days' written notice")
        if f["lease_cap"] is not None:
            s += f". Lease rider 4.3: renewal increase capped at {f['lease_cap']}%"
        return s

    def _pt(self, f):
        return {"none": "none", "pending": f"capital-improvement petition pending (+{f['pt_pct']}% requested)",
                "approved": f"capital-improvement pass-through approved (+{f['pt_pct']}%)"}[f["pass_through"]]

    def _fee(self, f, rng):
        if not f["fee"]:
            return None
        return f"new {f['fee']} {_usd(f['fee_amt'], rng)}/month"

    def render(self, f, rng, surface):
        mv = "Leasing says the market supports a bigger increase here." if f["market_view"] == "supports" else None
        if surface == "json":
            return dumps(prune({
                "unit": {"building": f["building"], "unit": f["unit"], "resident": f["resident"],
                         "building_first_occupied": f["built"], "lease_ends": f["lease_end"].isoformat()},
                "draft_offer": {"current_rent_usd": f["current"], "proposed_rent_usd": f["proposed"],
                                "new_monthly_charge": self._fee(f, rng), "notice_days_before_effective": f["notice_days"],
                                "months_since_last_increase": f["months_since_last"]},
                "context": {"capital_pass_through": self._pt(f), "days_left_to_serve_notice": f["days_to_notice_deadline"],
                            "leasing_note": mv},
                "rules": self._rule(f),
            }))
        if surface == "renewal_worksheet":
            body = _kv([("Unit", f"{f['building']} #{f['unit']} ({f['resident']})"),
                             ("Building first occupied", f["built"]), ("Lease ends", fmt_date(f["lease_end"], rng)),
                             ("Current rent", _usd(f["current"], rng)), ("Proposed rent", _usd(f["proposed"], rng)),
                             ("New charges", self._fee(f, rng) or "none"), ("Last increase", f"{f['months_since_last']} months ago"),
                             ("Notice as drafted", f"{f['notice_days']} days before effective date"),
                             ("Days left to serve notice", f["days_to_notice_deadline"]),
                             ("Capital pass-through", self._pt(f))], rng)
            return f"RENEWAL WORKSHEET -- {f['mgmt']}\n{body}\n{('Note: ' + mv) if mv else ''}\nRules: {self._rule(f)}".replace("\n\n", "\n")
        s = [f"Renewal draft for {f['building']} #{f['unit']} ({f['resident']}): {_usd(f['current'], rng)} to "
             f"{_usd(f['proposed'], rng)}" + (f", plus a {self._fee(f, rng)}." if f["fee"] else "."),
             f"Lease ends {fmt_date(f['lease_end'], rng)}; the letter as drafted gives {f['notice_days']} days' notice, "
             f"and we have {f['days_to_notice_deadline']} days left to serve it.",
             f"Last increase was {f['months_since_last']} months ago. Building first occupied {f['built']}.",
             f"Pass-through: {self._pt(f)}.", mv or ""]
        return "\n".join([f"Subject: Renewal check -- #{f['unit']}", f"From: {f['leasing']}", f"To: {f['manager']}", "",
                          join_sentences(s), "", f"Rules: {self._rule(f)}."])

    def questions(self):
        return {
            "renewal_offer_action": choice("What should happen to the drafted renewal offer?", {
                "await_pass_through_ruling": "Hold the part above the cap until the Rent Board rules on the pending petition.",
                "reduce_to_cap": "Cut the increase to the governing cap before it goes out.",
                "reissue_with_later_effective_date": "Reissue it with a later effective date so the notice and timing rules are met.",
                "send_as_drafted": "Send the offer as drafted.",
            }),
            "governing_limit": choice("Which written limit governs this increase?", {
                "lease_clause_cap": "The cap in the lease rider.",
                "no_written_cap": "None: neither the ordinance nor the lease caps it.",
                "ordinance_cap": "The ordinance's percentage cap.",
                "ordinance_cap_plus_pass_through": "The ordinance cap plus the approved capital pass-through.",
            }),
            "compliance_gap": score("How far does the draft depart from the written rules?", [
                "Not at all.",
                "One small departure.",
                "A clear breach, or two departures.",
                "Several breaches at once.",
            ]),
            "notice_timing_pressure": score("How much time is left to serve a valid notice for the planned effective date?", [
                "Over a month.",
                "One to four weeks.",
                "A week or less.",
                "None: the date has passed, so the effective date must move.",
            ]),
            "increase_within_cap": noul(
                "The total increase, counting any new mandatory recurring charge, stays within the governing cap.",
                yes="No cap applies, or the total increase is at or under the governing cap.",
                no="The total increase is over the governing cap."),
        }


# =========================================================================== schema 4
INCIDENT_TYPES = {
    "wet_lobby_slip": ("slipped on the freshly mopped lobby floor", "cleaning contractor"),
    "icy_walkway_fall": ("fell on ice on the front walkway", "snow and ice contractor"),
    "paving_trip": ("tripped on a raised paving slab in the courtyard", None),
    "elevator_entrapment": ("was trapped in the elevator between floors", "elevator maintenance contractor"),
    "gate_car_damage": ("had their car hit by the closing garage gate", "gate service contractor"),
    "ceiling_tile_fall": ("was hit by a falling ceiling tile in the corridor", None),
    "loose_handrail": ("fell when the stair handrail came loose", None),
}
INJURY = ["none", "first aid on site", "clinic visit", "emergency room"]


class CommonAreaIncident(Schema):
    name = "common_area_incident"
    description = "An injury or damage incident in a building's common area, handled under the incident standard and insurance conditions."
    surfaces = {"json": 0.5, "incident_report": 0.25, "site_manager_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        t = pick(rng, list(INCIDENT_TYPES))
        day = _date_in(rng, COLD) if t == "icy_walkway_fall" else _date_in(rng, list(range(1, 13)))
        injury = pick_w(rng, {"none": 2, "first aid on site": 3, "clinic visit": 2, "emergency room": 1})
        if t == "gate_car_damage":
            injury = "none"
        dmg = int(rng.uniform(300, 6000)) if t == "gate_car_damage" else pick(rng, [0, 0, 0, int(rng.uniform(50, 900))])
        X = pick(rng, [1000, 2500])
        known = pick_w(rng, {"open work order": 2, "reported to the front desk": 1.5, "none": 3})
        known_days = rng.randint(1, 30) if known != "none" else None
        sign = pick_w(rng, {"yes": 2, "no": 3, "unknown": 1})
        cctv = chance(rng, 0.7)
        R = pick(rng, [7, 14, 30])
        since = rng.randint(0, 20)
        saved = cctv and chance(rng, 0.25)
        letter = chance(rng, 0.1)
        present = chance(rng, 0.5)
        k = None
        if reg == "clear":
            if chance(rng, 0.5):
                injury, dmg, known, known_days, letter, present = "none", 0, "none", None, False, False
            else:
                injury, letter = "emergency room", chance(rng, 0.5)
        elif reg == "borderline":
            k = pick(rng, ["damage_edge", "footage_edge", "injury_edge"])
            if k == "damage_edge":
                t, injury = "gate_car_damage", "none"
                dmg = int(near(rng, X, X * 0.12))
            elif k == "footage_edge":
                cctv, saved = True, False
                since = near(rng, R, 2, lo=0, integer=True)
            else:
                injury = pick(rng, ["first aid on site", "clinic visit"])
        else:
            k = pick(rng, ["minor_but_known", "bad_injury_unknown_hazard", "contractor_blamed"])
            if k == "minor_but_known":
                injury, known, known_days, sign = "first aid on site", "open work order", rng.randint(10, 40), "no"
            elif k == "bad_injury_unknown_hazard":
                injury, known, known_days, sign = pick(rng, ["clinic visit", "emergency room"]), "none", None, "yes"
            else:
                t = pick(rng, ["wet_lobby_slip", "icy_walkway_fall", "elevator_entrapment"])
                injury = pick(rng, ["clinic visit", "emergency room", "first aid on site"])
                dmg = 0
        if t == "icy_walkway_fall" and day.month not in COLD:
            day = _date_in(rng, COLD)
        if t == "gate_car_damage":
            injury = "none"
        return {"regime": reg, "case_kind": k, "type": t, "date": day, "injury": injury, "damage_usd": dmg,
                "damage_rule": X, "known": known, "known_days": known_days, "sign": sign, "cctv": cctv,
                "retention_days": R, "days_since": since, "footage_saved": saved, "attorney_letter": letter,
                "hazard_present": present, "notice_days_rule": pick(rng, [5, 10, 30]),
                "person_role": pick(rng, ["resident", "visitor", "delivery driver"]), "who_id": _resident(rng),
                "building": _bldg(rng), "manager": person(rng), "regional": person(rng), "mgmt": pick(rng, MGMT),
                "policy": pick(rng, ["Policy condition 7", "GL policy cond. 4(b)", "Owner's liability policy s.9"])}

    def intended(self, f):
        contractor = INCIDENT_TYPES[f["type"]][1]
        due = f["injury"] in ("clinic visit", "emergency room") or f["damage_usd"] > f["damage_rule"] or f["attorney_letter"]
        if f["attorney_letter"] or f["injury"] == "emergency room":
            h = "refer_to_owner_and_counsel"
        elif contractor and due:
            h = "tender_to_contractor"
        elif due:
            h = "notify_insurer_and_preserve"
        elif f["hazard_present"]:
            h = "cordon_and_repair_urgently"
        else:
            h = "log_and_repair"
        inj = INJURY.index(f["injury"]) if f["type"] != "gate_car_damage" else (2 if f["damage_usd"] > f["damage_rule"] else 1)
        le = inj
        if f["known"] != "none" and (f["known_days"] or 0) >= 3:
            le += 1
        if f["sign"] == "yes":
            le -= 1
        le = max(0, min(3, le))
        reportable = f["injury"] != "none" or f["damage_usd"] > 0 or f["attorney_letter"]
        if not f["cctv"] or f["footage_saved"] or not reportable:
            ev = 0
        else:
            left = f["retention_days"] - f["days_since"]
            ev = 1 if left > 7 else 2 if left > 1 else 3
        return {"incident_handling": h, "liability_exposure": le, "evidence_preservation_urgency": ev,
                "insurer_notice_due": "yes" if due else "no",
                "hazard_known_beforehand": "yes" if f["known"] != "none" else "no"}

    def _rule(self, f):
        return (f"{f['policy']}: notify the insurer within {f['notice_days_rule']} days of any injury treated by a clinic "
                f"or hospital, property damage over {f['damage_rule']:,} USD, or any claim letter. Incident standard: "
                "cordon live hazards, preserve CCTV for any reportable incident")

    def _what(self, f):
        who = f"A {f['person_role']} ({f['who_id']})" if f["person_role"] == "resident" else f"A {f['person_role']}"
        return f"{who} {INCIDENT_TYPES[f['type']][0]}"

    def _cctv(self, f):
        if not f["cctv"]:
            return "no camera covers the spot"
        if f["footage_saved"]:
            return "footage already exported"
        return f"camera covers the spot; recorder keeps {f['retention_days']} days; not yet exported"

    def _known(self, f):
        if f["known"] == "none":
            return "no prior report found"
        return f"{f['known']} about it {f['known_days']} days before"

    def render(self, f, rng, surface):
        if surface == "json":
            return dumps(prune({
                "incident": {"building": f["building"], "date": f["date"].isoformat(), "what": self._what(f),
                             "injury": f["injury"] if f["type"] != "gate_car_damage" else None,
                             "property_damage_usd": f["damage_usd"] or None, "warning_sign_in_place": f["sign"],
                             "hazard_still_present": f["hazard_present"], "days_since": f["days_since"]},
                "background": {"prior_report": self._known(f), "contractor_involved": INCIDENT_TYPES[f["type"]][1],
                               "claim_letter_received": f["attorney_letter"]},
                "evidence": {"cctv": self._cctv(f)},
                "rules": self._rule(f),
            }))
        if surface == "incident_report":
            body = _kv([("Building", f["building"]), ("Date", fmt_date(f["date"], rng)), ("What happened", self._what(f)),
                             ("Injury", f["injury"] if f["type"] != "gate_car_damage" else "n/a"),
                             ("Damage", _usd(f["damage_usd"], rng) if f["damage_usd"] else "none reported"),
                             ("Warning sign", f["sign"]), ("Hazard still there", "yes" if f["hazard_present"] else "no, fixed"),
                             ("Prior report", self._known(f)), ("Contractor on duty", INCIDENT_TYPES[f["type"]][1] or "none"),
                             ("CCTV", self._cctv(f)), ("Claim letter", "received" if f["attorney_letter"] else "none"),
                             ("Days since incident", f["days_since"])], rng)
            return f"COMMON AREA INCIDENT REPORT -- {f['mgmt']}\n{body}\n{self._rule(f)}"
        s = [f"{self._what(f)} on {fmt_date(f['date'], rng)} ({f['days_since']} days ago).",
             (f"Injury: {f['injury']}." if f["type"] != "gate_car_damage" else "")
             + (f" Damage estimate {_usd(f['damage_usd'], rng)}." if f["damage_usd"] else ""),
             f"Warning sign: {f['sign']}. Prior report: {self._known(f)}.",
             (f"The {INCIDENT_TYPES[f['type']][1]} was responsible for that area." if INCIDENT_TYPES[f["type"]][1] else ""),
             f"CCTV: {self._cctv(f)}.", ("A claim letter has arrived." if f["attorney_letter"] else ""),
             ("The hazard is still there." if f["hazard_present"] else "The hazard has been fixed.")]
        return "\n".join([f"Subject: Incident at {f['building']}", f"From: {f['manager']}", f"To: {f['regional']}", "",
                          join_sentences(s), "", f"Per {self._rule(f)}.", "", _first(f["manager"])])

    def questions(self):
        return {
            "incident_handling": choice("How should management handle this incident now?", {
                "cordon_and_repair_urgently": "Cordon off the hazard and get it repaired urgently; no claim steps needed.",
                "log_and_repair": "Log it and fix anything outstanding through normal maintenance.",
                "notify_insurer_and_preserve": "Notify the owner's insurer and preserve the evidence.",
                "refer_to_owner_and_counsel": "Refer it to the owner and legal counsel as a likely claim.",
                "tender_to_contractor": "Tender the claim to the responsible contractor and its insurer, copying the owner's insurer.",
            }),
            "liability_exposure": score("How exposed is the owner to a liability claim?", [
                "Minimal: no real harm and nothing pointing at the owner.",
                "Low: minor harm or a hazard properly signed.",
                "Moderate: real harm, or a hazard the owner knew about.",
                "High: serious harm from a known hazard left unsigned.",
            ]),
            "evidence_preservation_urgency": score("How urgently must the CCTV footage be secured?", [
                "Not urgent: nothing needs preserving, no footage exists, or it is already saved.",
                "Some urgency: more than a week before it is overwritten.",
                "Urgent: it will be overwritten within days.",
                "Critical: it is due to be overwritten today or already has been.",
            ]),
            "insurer_notice_due": noul(
                "The owner's insurance conditions require the insurer to be notified of this incident.",
                yes="Someone needed clinic or hospital care, damage is over the policy figure, or a claim letter came in.",
                no="None of the policy's notice triggers is met."),
            "hazard_known_beforehand": noul("Management knew about the hazard before the incident."),
        }


# =========================================================================== schema 5
PROJECTS = {"roof replacement": (60000, 240000, "tear-off and disposal"), "boiler replacement": (40000, 180000, "flue liner"),
            "parking lot resurfacing": (25000, 120000, "line striping"), "elevator modernisation": (150000, 420000, "cab interior"),
            "exterior painting": (20000, 90000, "lift rental"), "hallway carpet replacement": (8000, 40000, "furniture moving"),
            "fire alarm panel upgrade": (15000, 60000, "monitoring tie-in")}
TRADES = {"roof replacement": ["Roofing", "Roofing & Sheet Metal"], "boiler replacement": ["Mechanical", "Heating"],
          "parking lot resurfacing": ["Paving", "Asphalt"], "elevator modernisation": ["Elevator", "Lift Services"],
          "exterior painting": ["Painting", "Decorating"], "hallway carpet replacement": ["Flooring", "Carpet"],
          "fire alarm panel upgrade": ["Fire & Security", "Electrical"]}


class VendorBidSelection(Schema):
    name = "vendor_bid_selection"
    description = "Contractor bids for building work, compared before award under the property management agreement."
    surfaces = {"json": 0.5, "bid_tabulation": 0.25, "manager_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        proj = pick(rng, list(PROJECTS))
        lo, hi, excl = PROJECTS[proj]
        A = pick(rng, [5000, 10000, 25000])
        kreq = pick(rng, [2, 3])
        L = pick(rng, [1, 2])  # USD millions
        B = pick(rng, [25000, 50000, 100000])
        base = rng.uniform(lo, hi)
        n = rng.randint(2, 4)
        names = rng.sample(VENDOR_A, n)
        bids = []
        for nm in names:
            bids.append({"vendor": f"{nm} {pick(rng, TRADES[proj] + ['Contracting', 'Building Services'])}",
                         "price": int(round(base * rng.uniform(0.85, 1.2), -2)), "scope": "full",
                         "licence": pick_w(rng, {"valid": 8, "expired": 1, "not provided": 1}),
                         "insurance_m": pick_w(rng, {L: 6, L + 1: 2, 0.5: 1}), "ins_ok_dates": chance(rng, 0.9),
                         "warranty": pick(rng, [1, 2, 5, 10]), "refs": pick_w(rng, {"good": 5, "mixed": 2, "none": 1})})
        pref = rng.randrange(n)
        related = chance(rng, 0.1)
        k = None
        if reg == "clear":
            for b in bids:
                b.update(licence="valid", insurance_m=L, ins_ok_dates=True, scope="full")
            if chance(rng, 0.5):
                pref = min(range(n), key=lambda i: bids[i]["price"])
                if n < kreq:
                    bids.append(dict(bids[0], vendor=f"{pick(rng, [v for v in VENDOR_A if v not in names])} Contracting",
                                     price=int(round(bids[0]["price"] * 1.1, -2))))
            related = False
        elif reg == "borderline":
            k = pick(rng, ["count_edge", "best_value_edge"])
            if k == "count_edge":
                bids[rng.randrange(len(bids))]["insurance_m"] = 0.5 if chance(rng, 0.5) else L
            else:
                cheapest = min(range(n), key=lambda i: bids[i]["price"])
                pref = (cheapest + 1) % n
                bids[pref]["price"] = int(round(bids[cheapest]["price"] * rng.uniform(1.03, 1.15), -2))
                bids[pref]["warranty"] = 10
        else:
            k = pick(rng, ["cheap_partial_scope", "pref_uninsured", "related_party"])
            if k == "cheap_partial_scope":
                cheapest = min(range(n), key=lambda i: bids[i]["price"])
                bids[cheapest]["scope"] = f"excludes {excl}"
            elif k == "pref_uninsured":
                bids[pref].update(ins_ok_dates=False)
            else:
                related = True
        if chance(rng, 0.2):
            bids[rng.randrange(len(bids))]["scope"] = f"excludes {excl}"
            if chance(rng, 0.4):
                bids[rng.randrange(len(bids))]["scope"] = f"excludes {excl} and permits"
        return {"regime": reg, "case_kind": k, "project": proj, "bids": bids, "preferred": pref, "related_party": related,
                "bids_above": A, "bids_required": kreq, "insurance_min_m": L, "owner_approval_above": B,
                "building": _bldg(rng), "manager": person(rng), "owner": f"{pick(rng, BLDG_A)} Holdings LLC",
                "date": rand_date(rng), "mgmt": pick(rng, MGMT),
                "pma": pick(rng, ["PMA s6.2", "Management Agreement 6.2", "Owner Agreement sched. C"])}

    def _qual(self, f, b):
        return b["licence"] == "valid" and b["insurance_m"] >= f["insurance_min_m"] and b["ins_ok_dates"]

    def intended(self, f):
        bids = f["bids"]
        pref = bids[f["preferred"]]
        qual = [b for b in bids if self._qual(f, b)]
        top = max(b["price"] for b in bids)
        count_ok = top < f["bids_above"] or len(qual) >= f["bids_required"]
        comparable = all(b["scope"] == "full" for b in qual)
        lowest = min(qual, key=lambda b: b["price"]) if qual else None
        if not count_ok:
            award = "gather_more_bids"
        elif not comparable:
            award = "rebid_on_common_scope"
        elif lowest is pref:
            award = "award_lowest_qualified"
        elif self._qual(f, pref) and pref["price"] <= lowest["price"] * 1.15 and pref["warranty"] >= lowest["warranty"] + 3:
            award = "award_best_value_alternative"
        else:
            award = "award_lowest_qualified"
        partial = sum(1 for b in bids if b["scope"] != "full")
        spread = (max(b["price"] for b in bids) - min(b["price"] for b in bids)) / min(b["price"] for b in bids)
        if partial == 0:
            comp = 2 if spread > 0.25 else 3
        elif partial == 1 and len(bids) >= 3:
            comp = 1
        else:
            comp = 0
        risk = 0
        if pref["licence"] != "valid":
            risk += 2
        if pref["insurance_m"] < f["insurance_min_m"] or not pref["ins_ok_dates"]:
            risk += 1
        if pref["refs"] == "none":
            risk += 1
        elif pref["refs"] == "mixed" and risk == 0:
            risk += 1
        if f["related_party"]:
            risk += 1
        return {"bid_award": award, "bid_comparability": comp, "preferred_vendor_risk": min(3, risk),
                "bid_count_rule_met": "yes" if count_ok else "no",
                "owner_approval_required": "yes" if (pref["price"] > f["owner_approval_above"] or f["related_party"]) else "no"}

    def _rule(self, f):
        return (f"{f['pma']}: work over {f['bids_above']:,} USD needs {f['bids_required']} written bids from licensed vendors "
                f"carrying >= {f['insurance_min_m']}M USD liability cover through the job; award the lowest qualified bid unless "
                f"a best-value reason is documented; owner's written approval over {f['owner_approval_above']:,} USD or for "
                "any related-party vendor")

    def _ins(self, b):
        return f"{b['insurance_m']}M" + ("" if b["ins_ok_dates"] else ", expires before job start")

    def _note(self, f):
        p = f["bids"][f["preferred"]]
        s = f"I recommend {p['vendor']}: {p['warranty']}-year warranty and we know their crews."
        if f["related_party"]:
            s += " Disclosure: the owner of that firm is my brother-in-law."
        return s

    def render(self, f, rng, surface):
        note = self._note(f)
        if surface == "json":
            return dumps(prune({
                "project": {"building": f["building"], "work": f["project"], "owner": f["owner"]},
                "bids": [{"vendor": b["vendor"], "price_usd": b["price"], "scope": b["scope"], "licence": b["licence"],
                          "liability_cover": self._ins(b), "warranty_years": b["warranty"], "references": b["refs"]}
                         for b in f["bids"]],
                "manager_recommendation": note,
                "agreement": self._rule(f),
            }))
        if surface == "bid_tabulation":
            rows = [[b["vendor"], _usd(b["price"], rng), b["scope"], b["licence"], self._ins(b), f"{b['warranty']} yr", b["refs"]]
                    for b in f["bids"]]
            return "\n".join([f"BID TABULATION -- {f['project']}, {f['building']} ({fmt_date(f['date'], rng)})",
                              table(["vendor", "price", "scope", "licence", "cover", "warranty", "refs"], rows, rng),
                              f"Manager: {note}", self._rule(f)])
        lines = [f"- {b['vendor']}: {_usd(b['price'], rng)}, scope: {b['scope']}, licence {b['licence']}, cover {self._ins(b)}, "
                 f"{b['warranty']}-yr warranty, references {b['refs']}" for b in f["bids"]]
        return "\n".join([f"MEMO to {f['owner']} / {f['mgmt']}", f"From: {f['manager']}, property manager",
                          f"Re: {f['project']} at {f['building']}", "", "Bids received:", "\n".join(lines), "", note, "",
                          f"Agreement: {self._rule(f)}."])

    def questions(self):
        return {
            "bid_award": choice("How should the property manager proceed with the award?", {
                "award_best_value_alternative": "Award to a qualified bid other than the cheapest, with a documented best-value reason.",
                "award_lowest_qualified": "Award to the lowest-priced qualified bid.",
                "gather_more_bids": "Get more qualifying bids before awarding.",
                "rebid_on_common_scope": "Ask the vendors to rebid on one common scope.",
            }),
            "bid_comparability": score("How directly can the bids be compared?", [
                "Not at all: the scopes differ a lot.",
                "Partly: one bid leaves out part of the work.",
                "Mostly: same scope, but prices far apart.",
                "Fully: same scope and prices in a normal range.",
            ]),
            "preferred_vendor_risk": score("How much risk does the manager's recommended vendor carry?", [
                "Low: licensed, insured, good references, no conflict.",
                "Some: one weak point.",
                "Significant: a gap in licence or cover plus another concern.",
                "High: several serious gaps or a conflict on top of them.",
            ]),
            "bid_count_rule_met": noul(
                "The file holds the number of qualifying written bids the agreement requires for this job.",
                yes="The job is under the bid threshold, or enough bids are licensed and properly insured.",
                no="Too few bids qualify once licence and insurance are checked."),
            "owner_approval_required": noul("The owner's written approval is needed before this award."),
        }


WORKFLOW = Workflow(
    name="property_management_operations",
    title="Property management operations",
    description="Work orders, deposit deductions, renewal increases, common-area incidents and contractor bids, against "
                "fictional codes, ordinances and agreements.",
    domain="real estate property management (NAICS 531)",
    schemas=[WorkOrderPriority(), DepositDeductionReview(), RenewalRentIncrease(), CommonAreaIncident(),
             VendorBidSelection()],
    notes="No tenant screening and no protected attributes; residents are role IDs. All rules and parties are fictional.",
)
