"""Wholesale credit-hold release: whether a distributor's credit team releases a customer order held for credit
limit, past-due terms, missing terms or an external credit event, judged against written credit policy.

Five schemas: an order that takes the account over its limit, an order held for a past-due balance, a new
customer's first order on open terms, a seasonal extended-dating request, and an order held after an outside
credit trigger (bureau score drop, lien, insurer cut, change of owner). The artefacts are credit-desk records
(order-hold queue entries, account summaries with aging, credit applications, analyst notes, sales requests).

Scope is the ORDER-RELEASE decision only. Balances appear as account totals and aging buckets; nothing here
reconciles, approves or rejects a bill, matches documents to deliveries, or processes a payment. Distributors,
customers, insurers and policies are invented. Each schema draws a `regime` first so most cases sit near a policy
line, pair a good history with a bad signal, or leave a figure out.

Each schema also has `intended(f)`: the designer's reading of the written policy for that case. It is never
rendered and never a label; it exists so the answer-spread check can confirm every question moves with the factors.
"""
from __future__ import annotations

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
    table,
)

# =========================================================================== shared pools
DIST_A = ["Saltmarsh", "Copperline", "Ironbark", "Bluestem", "Granite Ridge", "Fernhill", "Tidemark", "Oakridge",
          "Northgate", "Redfern", "Westbrook", "Upland", "Harrowgate", "Millbrook", "Stonebridge", "Cedarvale",
          "Ashford", "Brookside", "Kettering", "Lynwood"]
# sector -> (distributor suffix, customer kinds, customer-name endings, goods, perishable share)
SECTORS = {
    "foodservice": ("Foodservice", ["restaurant group", "cafe", "catering company", "school caterer", "hotel kitchen"],
                    ["Bistro Group", "Kitchens", "Catering", "Eatery", "Dining Co."],
                    ["frozen proteins", "dry goods", "fresh produce", "dairy"], 0.6),
    "electrical": ("Electrical Supply", ["electrical contractor", "solar installer", "facilities contractor"],
                   ["Electrical", "Electric Co.", "Power Services", "Solar"],
                   ["copper wire", "switchgear", "LED fixtures", "conduit", "inverters"], 0.0),
    "building": ("Building Supply", ["general contractor", "roofing contractor", "lumber yard", "kitchen fitter"],
                 ["Construction", "Builders", "Roofing", "Joinery", "Homes"],
                 ["lumber", "roofing shingles", "drywall", "custom-cut trusses", "windows made to order"], 0.0),
    "auto_parts": ("Auto Parts Distribution", ["independent repair shop", "fleet depot", "body shop"],
                   ["Motors", "Auto Care", "Garage", "Fleet Services"],
                   ["brake components", "batteries", "filters", "tyres"], 0.0),
    "beverage": ("Beverage Distributors", ["bar and grill", "convenience store group", "event caterer", "wine shop"],
                 ["Taproom", "Bar & Grill", "Wines", "Convenience"],
                 ["craft beer kegs", "spirits", "soft drinks", "wine"], 0.3),
    "horticulture": ("Horticultural Supply", ["garden centre", "landscaping firm", "nursery"],
                     ["Garden Centre", "Landscapes", "Nurseries", "Greenhouses"],
                     ["bedding plants", "fertiliser", "compost", "garden tools"], 0.4),
    "janitorial": ("Janitorial Supply", ["cleaning contractor", "facilities firm"],
                   ["Cleaning Services", "Facilities", "Maintenance"],
                   ["paper towels", "floor chemicals", "trash liners"], 0.0),
    "pet": ("Pet Supply Distribution", ["independent pet store", "grooming chain", "veterinary practice"],
            ["Pet Supplies", "Pets", "Grooming", "Vets"], ["dry dog food", "cat litter", "aquarium supplies"], 0.0),
    "plumbing": ("Plumbing Wholesale", ["plumbing contractor", "HVAC installer", "property maintenance firm"],
                 ["Plumbing", "Heating", "Mechanical", "Property Services"],
                 ["water heaters", "copper fittings", "PEX pipe", "boilers"], 0.0),
}
TERMS = ["Net 30", "Net 30", "Net 45", "Net 60", "1% 10 Net 30"]
PERISHABLE = {"frozen proteins", "fresh produce", "dairy", "bedding plants", "craft beer kegs"}
KIND_ENDINGS = {
    "restaurant group": ["Bistro Group", "Dining Co.", "Restaurants"], "cafe": ["Cafe", "Coffee House"],
    "catering company": ["Catering", "Event Catering"], "school caterer": ["School Meals", "Catering Services"],
    "hotel kitchen": ["Hotel", "Inn"], "electrical contractor": ["Electrical", "Electric Co."],
    "solar installer": ["Solar", "Solar Installations"], "facilities contractor": ["Facilities", "Building Services"],
    "general contractor": ["Construction", "Builders"], "roofing contractor": ["Roofing", "Roofing & Cladding"],
    "lumber yard": ["Timber", "Lumber"], "kitchen fitter": ["Kitchens", "Joinery"],
    "independent repair shop": ["Motors", "Auto Repair"], "fleet depot": ["Fleet Services", "Fleet Maintenance"],
    "body shop": ["Body Shop", "Collision Repair"], "bar and grill": ["Bar & Grill", "Taproom"],
    "convenience store group": ["Convenience", "Quick Stop"], "event caterer": ["Events", "Event Catering"],
    "wine shop": ["Wines", "Wine Merchants"], "garden centre": ["Garden Centre", "Home & Garden"],
    "landscaping firm": ["Landscapes", "Landscaping"], "nursery": ["Nurseries", "Greenhouses"],
    "cleaning contractor": ["Cleaning Services", "Cleaning Co."], "facilities firm": ["Facilities", "Maintenance"],
    "independent pet store": ["Pet Supplies", "Pets"], "grooming chain": ["Grooming", "Pet Spa"],
    "veterinary practice": ["Vets", "Animal Clinic"], "plumbing contractor": ["Plumbing", "Plumbing & Drains"],
    "HVAC installer": ["Heating & Cooling", "Mechanical"], "property maintenance firm": ["Property Services", "Maintenance"],
    "hardware store": ["Hardware", "Hardware & Supply"], "pool and spa store": ["Pools & Spas", "Pool Centre"],
    "pool service company": ["Pool Services", "Pool Care"], "gift shop": ["Gifts", "Gift House"],
    "home store": ["Home Store", "Interiors"], "farm supply store": ["Farm Supply", "Country Store"],
    "outdoor power dealer": ["Outdoor Power", "Mower Centre"],
}


def _sector(rng):
    return pick(rng, list(SECTORS))


def _distributor(rng, sec):
    return f"{pick(rng, DIST_A)} {SECTORS[sec][0]}"


def _customer(rng, kind):
    return f"{last_name(rng)} {pick(rng, KIND_ENDINGS[kind])}"


def _goods_type(goods):
    if goods.startswith(("custom", "windows")):
        return "custom"
    return "perishable" if goods in PERISHABLE else "stock"


def _usd(x, rng):
    return money(round(x), rng, style=pick(rng, ["sym", "sym", "code"]))


def _acct(rng):
    return ident(rng, "C", 6, "")


REP_NOTES = ["This is their busiest month and they have always come good.", "Big job for them, please don't lose it.",
             "Customer says the past-due is a timing issue with their own receivables.",
             "They will take the order to a competitor if we hold it.", "Long-standing account, never a write-off.",
             "Owner has promised to call the credit team today.", "", ""]


# =========================================================================== schema 1
class OverLimitOrderRelease(Schema):
    name = "over_limit_order_release"
    description = ("An order held because it takes the account over its credit limit: release, part-ship, ask "
                   "for cash in advance on the excess, hold for the past-due, or decline.")
    surfaces = {"json": 0.5, "hold_queue": 0.25, "analyst_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        sec = _sector(rng)
        limit = max(5000, round(lognormal_amount(rng, 60000, 0.8), -3))
        allow = pick(rng, [10, 15, 20])
        dbt_cap = pick(rng, [10, 15])
        a1, a2, a3 = pick(rng, [(10, 25, 50), (5, 20, 40), (10, 30, 60)])
        dbt = rng.randint(0, 35)
        oldest = pick(rng, [0, 0, 0, 12, 25, 40, 55, 75, 95])
        bureau = maybe(rng, rng.randint(20, 95), 0.15)
        security = pick_w(rng, {None: 6, "personal_guarantee": 2, "letter_of_credit": 1, "credit_insurance": 1})
        over = rng.uniform(0.02, 0.6)
        splittable = chance(rng, 0.55)
        if reg == "clear":
            if chance(rng, 0.5):
                over, dbt, oldest, bureau = rng.uniform(0.01, allow / 100 * 0.7), rng.randint(0, dbt_cap - 3), 0, rng.randint(65, 95)
            else:
                over, dbt, oldest, bureau = rng.uniform(0.35, 0.9), rng.randint(30, 60), pick(rng, [75, 95, 120]), rng.randint(10, 30)
        elif reg == "borderline":
            k = pick(rng, ["allowance", "authority", "dbt"])
            if k == "allowance":
                over = max(0.005, near(rng, allow / 100, 0.03))
                oldest = pick(rng, [0, 0, 12, 25, 35])
            elif k == "authority":
                over = max(0.005, near(rng, pick(rng, [a1, a2, a3]) / 100, 0.03))
            else:
                dbt, over = near(rng, dbt_cap, 3, lo=0, integer=True), rng.uniform(0.02, allow / 100)
                oldest = pick(rng, [0, 0, 25])
        else:
            k = pick(rng, ["secured_but_slow", "good_payer_big_order", "past_due_small"])
            if k == "secured_but_slow":
                security, dbt, over = pick(rng, ["letter_of_credit", "credit_insurance"]), rng.randint(18, 40), rng.uniform(0.15, 0.45)
            elif k == "good_payer_big_order":
                dbt, oldest, bureau, over = rng.randint(0, 6), 0, rng.randint(70, 95), rng.uniform(0.3, 0.7)
            else:
                oldest, over = rng.randint(61, 100), rng.uniform(0.03, 0.25)
        after = round(limit * (1 + over), -1)
        order = round(min(after * 0.6, max(500, after * rng.uniform(0.08, 0.4))), -1)
        before = after - order
        past_due = 0 if oldest == 0 else round(before * rng.uniform(0.03, 0.3), -1)
        if reg == "conflicting" and oldest > 60 and chance(rng, 0.5):
            past_due = round(limit * rng.uniform(0.02, 0.08), -1)
        open_orders = round(before * rng.uniform(0, 0.25), -1)
        current = max(0, before - past_due - open_orders)
        sec_amt = int(round((after - limit) * rng.uniform(0.6, 1.8), -2)) if security in ("letter_of_credit", "credit_insurance") else None
        ship = pick(rng, [0, 1, 2, 3, 5, 8, 12])
        goods = pick(rng, SECTORS[sec][3])
        special = _goods_type(goods)
        kind = pick(rng, SECTORS[sec][1])
        return {
            "regime": reg, "sector": sec, "distributor": _distributor(rng, sec), "customer": _customer(rng, kind),
            "customer_kind": kind, "account": _acct(rng), "years_customer": rng.randint(1, 25),
            "terms": pick(rng, TERMS), "credit_limit": int(limit), "open_current": int(current),
            "past_due": int(past_due), "oldest_past_due_days": oldest, "open_orders": int(open_orders),
            "order_value": int(order),
            "order_ref": ident(rng, "SO", 7, ""), "goods": goods, "goods_type": special, "ship_in_days": ship,
            "splittable": splittable, "dbt_12m": dbt, "bureau_score": bureau, "security": security,
            "security_amount": sec_amt, "policy_allowance_pct": allow, "policy_dbt_cap": dbt_cap,
            "authority_pct": [a1, a2, a3], "rep": person(rng), "rep_note": pick(rng, REP_NOTES),
            "analyst": person(rng), "hold_date": rand_date(rng),
        }

    def _exposure(self, f):
        before = f["open_current"] + f["past_due"] + f["open_orders"]
        return before, before + f["order_value"]

    def _risk(self, f):
        d = f["dbt_12m"]
        p = (0 if d <= 5 else 1 if d <= 15 else 2 if d <= 30 else 3)
        o = f["oldest_past_due_days"]
        p += 0 if o <= 30 else 1 if o <= 60 else 2
        b = f["bureau_score"]
        p += 1 if b is None else 0 if b >= 70 else 1 if b >= 50 else 2 if b >= 30 else 3
        return 0 if p <= 1 else 1 if p == 2 else 2 if p <= 4 else 3 if p <= 6 else 4

    def intended(self, f):
        before, after = self._exposure(f)
        over = after / f["credit_limit"] - 1
        pd60 = f["past_due"] > 0 and f["oldest_past_due_days"] > 60
        allow_ok = (over <= f["policy_allowance_pct"] / 100 and f["dbt_12m"] <= f["policy_dbt_cap"]
                    and not (f["past_due"] > 0 and f["oldest_past_due_days"] > 30))
        a1, a2, a3 = f["authority_pct"]
        lvl = 0 if over <= a1 / 100 else 1 if over <= a2 / 100 else 2 if over <= a3 / 100 else 3
        if pd60:
            lvl = max(lvl, 1)
        risk = self._risk(f)
        secured = f["security_amount"] is not None and f["security_amount"] >= after - f["credit_limit"]
        room = f["credit_limit"] - before
        if risk >= 4:
            dec = "decline_order"
        elif pd60 and f["past_due"] >= 0.03 * f["credit_limit"]:
            dec = "hold_pending_past_due"
        elif allow_ok:
            dec = "release_in_full"
        elif secured and risk <= 2:
            dec = "release_in_full"
        elif f["splittable"] and room >= 0.3 * f["order_value"]:
            dec = "release_partial"
        else:
            dec = "require_cash_in_advance"
        s = f["ship_in_days"]
        press = 0 if s >= 7 else 1 if s >= 3 else 2
        if s <= 2 and f["goods_type"] in ("perishable", "custom"):
            press = 3
        return {"hold_decision": dec,
                "release_authority": ["credit_analyst", "credit_manager", "finance_director", "credit_committee"][lvl],
                "account_payment_risk": risk, "ship_date_pressure": press,
                "within_overlimit_allowance": "yes" if allow_ok else "no"}

    def _policy(self, f, rng):
        a1, a2, a3 = f["authority_pct"]
        return [f"Temporary over-limit allowance: up to {f['policy_allowance_pct']}% over the limit, only for accounts "
                f"averaging {f['policy_dbt_cap']} days beyond terms or less with nothing more than 30 days past due.",
                f"Release authority by exposure over the limit after the order: analyst up to {a1}%, credit manager up "
                f"to {a2}%, finance director up to {a3}%, credit committee beyond that.",
                "Any release on an account with a balance more than 60 days past due needs at least the credit manager.",
                "Security on file (guarantee, letter of credit, insured cover) may support a release up to its amount."]

    def _security_txt(self, f, rng):
        s = f["security"]
        if s is None:
            return "none on file"
        if s == "personal_guarantee":
            return "personal guarantee from the owner (unsecured, not valued)"
        label = "standby letter of credit" if s == "letter_of_credit" else "credit insurance cover"
        return f"{label} for {_usd(f['security_amount'], rng)}"

    def render(self, f, rng, surface):
        before, after = self._exposure(f)
        note = noisy(f["rep_note"], rng, 0.15) if f["rep_note"] else None
        if surface == "json":
            return dumps(prune({
                "order": {"ref": f["order_ref"], "value_usd": f["order_value"], "goods": f["goods"],
                          "goods_type": f["goods_type"], "requested_ship_in_days": f["ship_in_days"],
                          "can_ship_in_parts": f["splittable"], "hold_reason": "credit limit",
                          "held_on": f["hold_date"].isoformat()},
                "account": {"number": f["account"], "customer": f["customer"], "business": f["customer_kind"],
                            "distributor": f["distributor"], "years_as_customer": f["years_customer"],
                            "terms": f["terms"], "credit_limit_usd": f["credit_limit"]},
                "exposure": {"open_balance_current_usd": f["open_current"], "past_due_usd": f["past_due"],
                             "oldest_past_due_days": f["oldest_past_due_days"] or None,
                             "unshipped_orders_usd": f["open_orders"], "after_this_order_usd": after},
                "credit": {"avg_days_beyond_terms_12m": f["dbt_12m"], "bureau_score": f["bureau_score"],
                           "security": self._security_txt(f, rng)},
                "sales_note": ({"from": f["rep"], "text": note} if note else None),
                "policy": self._policy(f, rng),
            }))
        if surface == "hold_queue":
            rows = [["credit limit", _usd(f["credit_limit"], rng)], ["open, current", _usd(f["open_current"], rng)],
                    ["past due", _usd(f["past_due"], rng) + (f" (oldest {f['oldest_past_due_days']} d)" if f["past_due"] else "")],
                    ["unshipped orders", _usd(f["open_orders"], rng)], ["this order", _usd(f["order_value"], rng)],
                    ["exposure after order", _usd(after, rng)], ["avg days beyond terms", f["dbt_12m"]],
                    ["bureau score", "no file" if f["bureau_score"] is None else f["bureau_score"]],
                    ["security", self._security_txt(f, rng)]]
            return "\n".join([
                f"CREDIT HOLD QUEUE -- {f['distributor']}  ({fmt_date(f['hold_date'], rng)})",
                f"{f['order_ref']}  {f['customer']} ({f['account']}, {f['customer_kind']}, {f['terms']}, "
                f"customer {f['years_customer']} yrs)",
                f"Order: {f['goods']} ({f['goods_type']}), ship in {f['ship_in_days']} day(s), "
                f"{'can' if f['splittable'] else 'cannot'} ship in parts.", "", table(["item", "value"], rows, rng), "",
                (f"Sales ({f['rep']}): {note}" if note else "No note from sales."), "", "Policy:",
                bullets(self._policy(f, rng), rng)])
        s = [f"{f['order_ref']} for {f['customer']} ({f['customer_kind']}) is on credit hold: {_usd(f['order_value'], rng)} "
             f"of {f['goods']} would take exposure to {_usd(after, rng)} against a {_usd(f['credit_limit'], rng)} limit.",
             f"The account has {_usd(f['open_current'], rng)} current, {_usd(f['open_orders'], rng)} in unshipped orders"
             + (f" and {_usd(f['past_due'], rng)} past due, the oldest {f['oldest_past_due_days']} days." if f["past_due"]
                else " and nothing past due."),
             f"Twelve-month average is {f['dbt_12m']} days beyond terms ({f['terms']}); bureau score "
             + ("not available." if f["bureau_score"] is None else f"{f['bureau_score']}."),
             f"Security: {self._security_txt(f, rng)}.",
             f"Requested ship in {f['ship_in_days']} day(s); goods are {f['goods_type']}; the order "
             f"{'can' if f['splittable'] else 'cannot'} be split.",
             f"Sales says: \"{note}\"" if note else ""]
        return "\n".join([f"ANALYST NOTE -- {f['analyst']}", f"Account {f['account']} / {f['distributor']}", "",
                          join_sentences(s), "", "Policy reference:", bullets(self._policy(f, rng), rng)])

    def questions(self):
        return {
            "hold_decision": choice("What should the credit team do with the held order?", {
                "release_in_full": "Release the whole order on the account's normal terms.",
                "release_partial": "Ship the part of the order that fits under the limit and hold the rest.",
                "require_cash_in_advance": "Ship once the amount over the limit is paid in advance.",
                "hold_pending_past_due": "Keep the order held until the past-due balance is brought current.",
                "decline_order": "Decline the order on credit terms.",
            }),
            "release_authority": choice("Who has the authority to release the order in full under the policy?", {
                "credit_analyst": "The credit analyst working the queue.",
                "credit_manager": "The credit manager.",
                "finance_director": "The finance director.",
                "credit_committee": "The credit committee.",
            }),
            "account_payment_risk": score("How risky is the account's payment behaviour?", [
                "Very low: pays on time, sound outside score.",
                "Low: minor slippage only.",
                "Moderate: regular late payment or a weak outside score.",
                "High: well behind terms with aged balances.",
                "Severe: seriously delinquent on every measure.",
            ]),
            "ship_date_pressure": score("How much pressure does the requested ship date put on the credit decision?", [
                "None: it ships a week or more out.",
                "Some: it ships within the week.",
                "High: it misses its truck unless released within a day or two.",
                "Critical: perishable or made-to-order goods are ready and lose value if held.",
            ]),
            "within_overlimit_allowance": noul(
                "The order falls inside the policy's temporary over-limit allowance.",
                yes="Exposure after the order is within the allowance and the account meets its payment conditions.",
                no="Exposure goes past the allowance, or the account fails a payment condition."),
        }


# =========================================================================== schema 2
class PastDueTermsHold(Schema):
    name = "past_due_terms_hold"
    description = ("An order held because the account is past due beyond terms: release, release against a payment "
                   "commitment, keep holding, or move the account to prepayment.")
    surfaces = {"json": 0.5, "aging_summary": 0.25, "credit_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        sec = _sector(rng)
        limit = max(5000, round(lognormal_amount(rng, 50000, 0.8), -3))
        rule_x, rule_y = pick(rng, [2500, 5000]), pick(rng, [5, 10])
        rule_amt = max(rule_x, limit * rule_y / 100)
        b1 = round(limit * rng.uniform(0.02, 0.2), -1)
        target = rule_amt * pick(rng, [0.6, 1.3, 1.8, 2.5])
        dispute, raised, documented = 0, None, None
        broken = pick(rng, [0, 0, 0, 1, 2, 3])
        commit = None
        seasonal = chance(rng, 0.2)
        if reg == "clear":
            k = pick(rng, ["small", "deep"])
            target = rule_amt * (rng.uniform(0.3, 0.8) if k == "small" else rng.uniform(2.5, 5))
            if k == "deep":
                broken = pick(rng, [2, 3, 4])
        elif reg == "borderline":
            target = near(rng, rule_amt, rule_amt * 0.15, lo=200)
        else:
            k = pick(rng, ["dispute_late", "dispute_ok", "promise_breaker", "seasonal_payer"])
            target = rule_amt * rng.uniform(1.2, 2.2)
            if k in ("dispute_late", "dispute_ok"):
                dispute = round(target * rng.uniform(0.25, 0.6), -1)
                raised = rng.randint(31, 70) if k == "dispute_late" else rng.randint(3, 30)
                documented = chance(rng, 0.8) if k == "dispute_ok" else chance(rng, 0.5)
            elif k == "promise_breaker":
                broken = pick(rng, [2, 3, 4])
                commit = True
            else:
                seasonal, broken = True, 0
                commit = True
        if dispute == 0 and chance(rng, 0.45):
            dispute = round(target * rng.uniform(0.1, 0.5), -1)
            raised = near(rng, 24, 9, lo=1, integer=True)
            documented = chance(rng, 0.7)
        over30 = round(target, -1)
        split = [rng.uniform(0.3, 1), rng.uniform(0, 0.8), rng.uniform(0, 0.6)]
        if reg == "borderline" or chance(rng, 0.3):
            split = [rng.uniform(0.6, 1), rng.uniform(0, 0.25), rng.uniform(0, 0.12)]
        if reg == "clear" and broken >= 2:
            split[2] += 1.2
        tot = sum(split)
        b2, b3 = round(over30 * split[0] / tot, -1), round(over30 * split[1] / tot, -1)
        b4 = max(0, over30 - b2 - b3)
        if commit is None and chance(rng, 0.7):
            commit = True
        commit_amt = int(round(over30 * rng.uniform(0.4, 1.1), -1)) if commit else None
        kind = pick(rng, SECTORS[sec][1])
        return {
            "regime": reg, "sector": sec, "distributor": _distributor(rng, sec), "customer": _customer(rng, kind),
            "customer_kind": kind, "account": _acct(rng), "terms": pick(rng, TERMS),
            "credit_limit": int(limit), "current_balance": int(round(limit * rng.uniform(0.1, 0.5), -1)),
            "past_due_1_30": int(b1), "past_due_31_60": int(b2), "past_due_61_90": int(b3),
            "past_due_over_90": int(b4), "disputed_amount": int(dispute) or None, "dispute_raised_days_after_delivery": raised,
            "dispute_documented": documented, "rule_floor": rule_x, "rule_pct": rule_y,
            "commitment_amount": commit_amt, "commitment_in_days": rng.randint(1, 21) if commit else None,
            "broken_commitments_12m": broken, "seasonal_pattern": seasonal,
            "order_value": int(round(lognormal_amount(rng, limit * 0.12, 0.5), -1)), "order_ref": ident(rng, "SO", 7, ""),
            "ship_in_days": pick(rng, [1, 2, 4, 7, 10]), "years_customer": rng.randint(1, 30),
            "collector": person(rng), "hold_date": rand_date(rng),
        }

    def _over30(self, f):
        return f["past_due_31_60"] + f["past_due_61_90"] + f["past_due_over_90"]

    def _excludable(self, f):
        return bool(f["disputed_amount"]) and f["dispute_raised_days_after_delivery"] is not None \
            and f["dispute_raised_days_after_delivery"] <= 30 and bool(f["dispute_documented"])

    def intended(self, f):
        over30 = self._over30(f)
        rule = max(f["rule_floor"], f["credit_limit"] * f["rule_pct"] / 100)
        counted = over30 - (f["disputed_amount"] if self._excludable(f) else 0)
        trig = counted > rule
        b = f["broken_commitments_12m"]
        if f["commitment_amount"] is None:
            rel = 0
        else:
            rel = 3 if b == 0 else 2 if b == 1 else 1 if b == 2 else 0
            if f["commitment_in_days"] > 14 and rel > 0:
                rel -= 1
        if not trig:
            out = "release_order"
        elif rel >= 2 and f["commitment_amount"] >= counted - rule:
            out = "release_on_commitment"
        elif f["past_due_over_90"] >= 0.4 * over30 and b >= 2:
            out = "move_to_prepay_terms"
        else:
            out = "keep_on_hold"
        share90 = f["past_due_over_90"] / over30 if over30 else 0
        if counted <= rule:
            ser = 0
        elif share90 >= 0.3 or counted > 0.25 * f["credit_limit"]:
            ser = 3
        elif f["past_due_61_90"] + f["past_due_over_90"] > 0.3 * over30 or counted > 2 * rule:
            ser = 2
        else:
            ser = 1
        return {"terms_hold_outcome": out, "past_due_seriousness": ser, "commitment_reliability": rel,
                "dispute_excludable": "yes" if self._excludable(f) else "no",
                "hold_rule_still_triggered": "yes" if trig else "no"}

    def _policy(self, f, rng):
        return [f"Orders hold automatically when the balance more than 30 days past due exceeds the greater of "
                f"{_usd(f['rule_floor'], rng)} or {f['rule_pct']}% of the credit limit.",
                "A disputed amount is left out of that test only if the dispute was raised within 30 days of delivery "
                "and is documented (claim form plus evidence).",
                "A held order may be released against a written payment commitment that brings the counted balance "
                "under the rule, from a customer who has kept its commitments.",
                "Accounts with repeated broken commitments and balances over 90 days move to cash-in-advance terms."]

    def _facts(self, f, rng):
        s = []
        if f["disputed_amount"]:
            s.append(f"{_usd(f['disputed_amount'], rng)} of the past-due balance is disputed; the dispute was raised "
                     f"{f['dispute_raised_days_after_delivery']} days after delivery and is "
                     + ("documented with a claim form and photos." if f["dispute_documented"] else "not documented beyond an email."))
        if f["commitment_amount"]:
            s.append(f"The customer has committed in writing to pay {_usd(f['commitment_amount'], rng)} within "
                     f"{f['commitment_in_days']} days.")
        else:
            s.append("No payment commitment has been received.")
        s.append(f"Broken payment commitments in the last 12 months: {f['broken_commitments_12m']}.")
        if f["seasonal_pattern"]:
            s.append("History shows the account runs late every off-season and catches up when trade picks up.")
        return s

    def render(self, f, rng, surface):
        facts = self._facts(f, rng)
        if surface == "json":
            return dumps(prune({
                "order": {"ref": f["order_ref"], "value_usd": f["order_value"], "ship_in_days": f["ship_in_days"],
                          "hold_reason": "past due beyond terms", "held_on": f["hold_date"].isoformat()},
                "account": {"number": f["account"], "customer": f["customer"], "business": f["customer_kind"],
                            "distributor": f["distributor"], "terms": f["terms"], "credit_limit_usd": f["credit_limit"],
                            "years_as_customer": f["years_customer"]},
                "aging_usd": {"current": f["current_balance"], "1_30": f["past_due_1_30"], "31_60": f["past_due_31_60"],
                              "61_90": f["past_due_61_90"], "over_90": f["past_due_over_90"]},
                "notes": facts, "policy": self._policy(f, rng),
            }))
        if surface == "aging_summary":
            rows = [[_usd(f["current_balance"], rng), _usd(f["past_due_1_30"], rng), _usd(f["past_due_31_60"], rng),
                     _usd(f["past_due_61_90"], rng), _usd(f["past_due_over_90"], rng)]]
            return "\n".join([
                f"ACCOUNT AGING -- {f['customer']} ({f['account']}), {f['distributor']}",
                f"Limit {_usd(f['credit_limit'], rng)}; terms {f['terms']}; held order {f['order_ref']} "
                f"{_usd(f['order_value'], rng)}, ships in {f['ship_in_days']} day(s)", "",
                table(["current", "1-30", "31-60", "61-90", "90+"], rows, rng), "", bullets(facts, rng), "",
                "Credit policy:", bullets(self._policy(f, rng), rng)])
        body = [f"Order {f['order_ref']} ({_usd(f['order_value'], rng)}) for {f['customer']} is on hold for past-due terms.",
                f"Aging: {_usd(f['past_due_1_30'], rng)} at 1-30 days, {_usd(f['past_due_31_60'], rng)} at 31-60, "
                f"{_usd(f['past_due_61_90'], rng)} at 61-90 and {_usd(f['past_due_over_90'], rng)} over 90; "
                f"limit {_usd(f['credit_limit'], rng)}."] + facts
        return "\n".join([f"Subject: Hold on {f['order_ref']} -- {f['customer']}", f"From: {f['collector']}", "",
                          noisy(join_sentences(body), rng, 0.08), "", "Policy extract:", bullets(self._policy(f, rng), rng)])

    def questions(self):
        return {
            "terms_hold_outcome": choice("What should happen to the order held for past-due terms?", {
                "release_order": "Release it; the account does not breach the hold rule once the policy is applied.",
                "release_on_commitment": "Release it against the customer's written payment commitment and track it.",
                "keep_on_hold": "Keep it on hold until enough of the past-due balance is paid.",
                "move_to_prepay_terms": "Keep holding and move the account to cash-in-advance terms.",
            }),
            "past_due_seriousness": score("How serious is the account's past-due position?", [
                "Minor: under the hold rule once the policy is applied.",
                "Moderate: over the rule, mostly in the 31-60 day bucket.",
                "Serious: well over the rule, or a large share past 60 days.",
                "Severe: a big share over 90 days, or a large part of the limit.",
            ]),
            "commitment_reliability": score("How far can the customer's payment commitment be relied on?", [
                "Not at all: none given, or recent ones broken repeatedly.",
                "Weakly: given, but the record of keeping them is poor or the date is far off.",
                "Reasonably: given, with one lapse on record or a date some way off.",
                "Fully: specific and soon, from a customer who keeps its commitments.",
            ]),
            "dispute_excludable": noul(
                "Policy lets a disputed amount be left out of the past-due test for this account.",
                yes="There is a dispute, raised within 30 days of delivery and documented.",
                no="There is no dispute, or it was raised late or is undocumented."),
            "hold_rule_still_triggered": noul(
                "After any allowed exclusion, the balance more than 30 days past due still triggers the hold rule."),
        }


# =========================================================================== schema 3
REGISTRY = {"active": "active, trading name matches", "renamed": "active; changed name 4 months ago",
            "mismatch": "no record under the trading name; a similar name is registered at another address"}


class FirstOrderOnTerms(Schema):
    name = "first_order_on_terms"
    description = ("A new customer's first order, held because no credit line exists yet: open terms as asked, at a "
                   "lower limit, prepay first orders, ask for more, or decline.")
    surfaces = {"json": 0.5, "credit_application": 0.25, "pretty_json": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        sec = _sector(rng)
        max_new = pick(rng, [10000, 15000, 25000])
        req = round(max_new * rng.uniform(0.4, 2.0), -3)
        years = round(rng.uniform(0.3, 25), 1)
        registry = pick_w(rng, {"active": 8, "renamed": 1, "mismatch": 1})
        n_ref = 3
        refs = []
        quality = pick_w(rng, {"good": 5, "mixed": 3, "weak": 2})
        if reg == "clear":
            quality = pick(rng, ["good", "weak"])
        for _ in range(n_ref):
            if quality == "good":
                refs.append({"responded": chance(rng, 0.95), "highest": int(round(req * rng.uniform(0.6, 2), -2)),
                             "pays": pick(rng, ["prompt", "prompt", "prompt", "prompt", "prompt", "slow"])})
            elif quality == "mixed":
                refs.append({"responded": chance(rng, 0.65), "highest": int(round(req * rng.uniform(0.25, 1.2), -2)),
                             "pays": pick(rng, ["prompt", "prompt", "slow"])})
            else:
                refs.append({"responded": chance(rng, 0.4), "highest": int(round(req * rng.uniform(0.1, 0.6), -2)),
                             "pays": pick(rng, ["prompt", "slow", "very slow"])})
        signals = {"ship_to_mismatch": chance(rng, 0.12), "webmail": chance(rng, 0.2), "rush": chance(rng, 0.2),
                   "resale_goods": chance(rng, 0.15)}
        if reg == "clear":
            if quality == "good":
                years, registry = rng.uniform(3, 25), "active"
                signals = {k: False for k in signals}
                req = round(max_new * rng.uniform(0.4, 0.95), -3)
            else:
                years = rng.uniform(0.3, 1.5)
        elif reg == "borderline":
            years = near(rng, 2, 0.6, lo=0.3)
            if chance(rng, 0.5):
                req = round(near(rng, max_new, max_new * 0.15), -3)
        else:
            k = pick(rng, ["sham", "good_but_flags", "young_strong"])
            if k == "sham":
                for key in rng.sample(list(signals), 3):
                    signals[key] = True
                registry = pick(rng, ["mismatch", "renamed", "active"])
            elif k == "good_but_flags":
                for key in rng.sample(list(signals), 2):
                    signals[key] = True
            else:
                years = rng.uniform(0.5, 1.8)
                for r in refs:
                    r["responded"], r["pays"] = True, "prompt"
        order = round(min(req * rng.uniform(0.4, 1.4), req * 1.5), -2)
        bureau = pick(rng, [None, rng.randint(25, 60), rng.randint(60, 92)])
        kind = pick(rng, SECTORS[sec][1])
        return {
            "regime": reg, "sector": sec, "distributor": _distributor(rng, sec), "applicant": _customer(rng, kind),
            "kind": kind, "years_trading": round(years, 1), "registry": registry,
            "trade_refs": refs, "bank_ref": pick(rng, ["satisfactory", "satisfactory", "account under a year old",
                                                       "declined to comment", None]),
            "bureau_score": bureau, "requested_terms": pick(rng, ["Net 30", "Net 30", "Net 45", "Net 60"]),
            "requested_limit": int(req), "first_order_value": int(order), "goods": pick(rng, SECTORS[sec][3]),
            "financials_provided": chance(rng, 0.35), "guarantee_signed": chance(rng, 0.4),
            "signals": signals, "policy_max_new": max_new, "app_ref": ident(rng, "CA", 5, "-"),
            "rep": person(rng), "received": rand_date(rng),
            "resale_goods_label": pick(rng, ["copper cable drums", "generators", "cordless tool kits", "spirits cases"]),
        }

    def _good_refs(self, f):
        return [r for r in f["trade_refs"] if r["responded"] and r["pays"] == "prompt"
                and r["highest"] >= 0.5 * f["requested_limit"]]

    def intended(self, f):
        good = self._good_refs(f)
        mins = f["years_trading"] >= 2 and len(good) >= 2 and f["registry"] != "mismatch"
        sig = sum(f["signals"].values()) + (1 if f["registry"] == "mismatch" else 0)
        strong_combo = f["signals"]["ship_to_mismatch"] and f["signals"]["resale_goods"] and f["signals"]["rush"]
        sham = 0 if sig == 0 else 1 if sig == 1 else 2 if sig == 2 and not strong_combo else 3
        silent = sum(1 for r in f["trade_refs"] if not r["responded"])
        if sig >= 2:
            ver = "confirm_business_identity"
        elif silent >= 2:
            ver = "contact_references_directly"
        elif f["requested_limit"] > f["policy_max_new"] and not f["financials_provided"]:
            ver = "obtain_financial_statements"
        else:
            ver = "no_further_verification"
        if sham >= 3:
            dec = "decline_application"
        elif sham == 2 or (silent >= 2 and f["years_trading"] >= 2):
            dec = "request_more_information"
        elif mins and f["requested_limit"] <= f["policy_max_new"] and f["first_order_value"] <= f["requested_limit"]:
            dec = "open_terms_as_requested"
        elif mins:
            dec = "open_terms_lower_limit"
        else:
            dec = "prepay_first_orders"
        pts = (f["years_trading"] >= 2) + (len(good) >= 2) + (f["bank_ref"] == "satisfactory") \
            + ((f["bureau_score"] or 0) >= 60) + bool(f["financials_provided"])
        strength = 0 if pts <= 1 else 1 if pts == 2 else 2 if pts == 3 else 3
        return {"first_order_terms": dec, "verification_step": ver, "credit_file_strength": strength,
                "sham_buyer_signals": sham, "meets_open_terms_minimums": "yes" if mins else "no"}

    def _policy(self, f, rng):
        return ["Open terms need two years' trading and at least two responding trade references that pay "
                "promptly with a highest credit of at least half the requested limit.",
                f"New accounts start at no more than {_usd(f['policy_max_new'], rng)}; above that, financial "
                f"statements are required before a higher line is considered.",
                "A first order larger than the granted line is prepaid.",
                "Unresolved signs of a sham buyer (delivery to an address that is not the business, a registry "
                "mismatch, rush freight on resaleable goods, webmail contact) are verified before any terms are set."]

    def _signals_txt(self, f):
        s = []
        sg = f["signals"]
        if sg["ship_to_mismatch"]:
            s.append("Ship-to address is a self-storage facility, not the registered premises.")
        if sg["webmail"]:
            s.append("All contact is from a free webmail address.")
        if sg["rush"]:
            s.append("Next-day freight requested on the first order.")
        if sg["resale_goods"]:
            s.append(f"First order is mostly {f['resale_goods_label']}.")
        return s

    def _refs_rows(self, f, rng):
        out = []
        for i, r in enumerate(f["trade_refs"], 1):
            if r["responded"]:
                out.append([f"ref {i}", "responded", _usd(r["highest"], rng), r["pays"]])
            else:
                out.append([f"ref {i}", "no response", "-", "-"])
        return out

    def render(self, f, rng, surface):
        sigs = self._signals_txt(f)
        if surface in ("json", "pretty_json"):
            obj = prune({
                "application": {"ref": f["app_ref"], "applicant": f["applicant"], "business": f["kind"],
                                "years_trading": f["years_trading"], "registry_check": REGISTRY[f["registry"]],
                                "requested_terms": f["requested_terms"], "requested_limit_usd": f["requested_limit"],
                                "received": f["received"].isoformat(), "distributor": f["distributor"]},
                "first_order": {"value_usd": f["first_order_value"], "goods": f["goods"]},
                "references": {"trade": [{"responded": r["responded"],
                                          "highest_credit_usd": r["highest"] if r["responded"] else None,
                                          "payment": r["pays"] if r["responded"] else None} for r in f["trade_refs"]],
                               "bank": f["bank_ref"] or "not returned"},
                "file": {"bureau_score": f["bureau_score"] if f["bureau_score"] is not None else "thin file",
                         "financial_statements": f["financials_provided"],
                         "personal_guarantee_signed": f["guarantee_signed"]},
                "observations": sigs or None, "policy": self._policy(f, rng),
            })
            return dumps(obj) if surface == "json" else dumps_pretty(obj, rng)
        head = kv_lines([("Application", f["app_ref"]), ("Applicant", f"{f['applicant']} ({f['kind']})"),
                         ("Years trading", f["years_trading"]), ("Registry", REGISTRY[f["registry"]]),
                         ("Terms / limit requested", f"{f['requested_terms']} / {_usd(f['requested_limit'], rng)}"),
                         ("First order", f"{_usd(f['first_order_value'], rng)} of {f['goods']}"),
                         ("Bank reference", f["bank_ref"]),
                         ("Bureau", "thin file" if f["bureau_score"] is None else f["bureau_score"]),
                         ("Financials", "provided" if f["financials_provided"] else "not provided"),
                         ("Guarantee", "signed" if f["guarantee_signed"] else "not signed")], rng, skip_none=False)
        return "\n".join([f"CREDIT APPLICATION -- {f['distributor']}", head, "",
                          table(["reference", "status", "highest credit", "pays"], self._refs_rows(f, rng), rng), "",
                          ("Observations:\n" + bullets(sigs, rng)) if sigs else "No observations recorded.",
                          f"Submitted by sales: {f['rep']}", "", "Policy:", bullets(self._policy(f, rng), rng)])

    def questions(self):
        return {
            "first_order_terms": choice("On what basis should the new customer's first order be released?", {
                "open_terms_as_requested": "Open an account on the terms and limit requested and ship.",
                "open_terms_lower_limit": "Open terms, but at a lower limit than requested.",
                "prepay_first_orders": "Ship on cash in advance until the account has a payment record.",
                "request_more_information": "Hold the order and ask for the missing or doubtful information first.",
                "decline_application": "Decline credit and do not ship on terms.",
            }),
            "verification_step": choice("What should the credit team verify before setting terms?", {
                "confirm_business_identity": "The applicant's identity and delivery address, through the registry "
                                             "and a call to its listed number.",
                "contact_references_directly": "The trade references, by contacting them directly.",
                "obtain_financial_statements": "The applicant's financial statements.",
                "no_further_verification": "Nothing more; the file is enough.",
            }),
            "credit_file_strength": score("How strong is the applicant's credit file?", [
                "Very weak.",
                "Weak.",
                "Reasonable.",
                "Strong.",
            ]),
            "sham_buyer_signals": score("How strongly do the details suggest a buyer that intends to take goods and not pay?", [
                "Not at all.",
                "Faintly: one detail that usually has an innocent reason.",
                "Noticeably: several details that need checking together.",
                "Strongly: the classic pattern of a sham buyer.",
            ]),
            "meets_open_terms_minimums": noul(
                "The applicant meets the policy's minimum requirements for open terms.",
                yes="Two years' trading, two qualifying trade references, and a registry record that matches.",
                no="It falls short on trading history, references or the registry check."),
        }


# =========================================================================== schema 4
SEASONAL = {
    "horticulture": ("spring", ["garden centre", "nursery", "hardware store"]),
    "pool": ("summer", ["pool and spa store", "pool service company"]),
    "holiday": ("holiday", ["gift shop", "home store", "garden centre"]),
    "snow": ("winter", ["hardware store", "farm supply store", "outdoor power dealer"]),
}
SEASON_DIST = {"horticulture": "Horticultural Supply", "pool": "Pool & Leisure Wholesale",
               "holiday": "Seasonal Decor Wholesale", "snow": "Outdoor Power Distribution"}
SEASON_GOODS = {"horticulture": ["bedding plants", "seed racks", "fertiliser", "garden tools"],
                "pool": ["pool chemicals", "pumps and filters", "pool toys"],
                "holiday": ["artificial trees", "lights", "ornaments"],
                "snow": ["snow blowers", "ice melt", "snow shovels"]}


class SeasonalDatingRequest(Schema):
    name = "seasonal_dating_request"
    description = ("A seasonal pre-book order held because the customer asks for extended dating: approve, cap at "
                   "the programme date, approve with security, or decline.")
    surfaces = {"json": 0.5, "dating_request_form": 0.25, "credit_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        line = pick(rng, list(SEASONAL))
        limit = max(10000, round(lognormal_amount(rng, 80000, 0.7), -3))
        prog_days = pick(rng, [120, 150, 180])
        insured_days = pick(rng, [120, 150, 180])
        min_order = pick(rng, [5000, 7500, 10000])
        req_days = prog_days + pick(rng, [-20, 0, 15, 30, 45, 60])
        order = round(limit * rng.uniform(0.4, 1.2), -2)
        exposure = round(limit * rng.uniform(0.05, 0.5), -2)
        insurer = round((exposure + order) * rng.uniform(0.7, 1.5), -3)
        history = [pick(rng, ["on time", "on time", "10 days late", "25 days late", "45 days late"]) for _ in range(3)]
        seasons = rng.randint(0, 8)
        on_time_deadline = chance(rng, 0.85)
        pd60 = pick(rng, [0, 0, 0, 0, round(limit * rng.uniform(0.01, 0.08), -2)])
        if reg == "clear":
            if chance(rng, 0.5):
                history, seasons, pd60, on_time_deadline = ["on time"] * 3, rng.randint(3, 8), 0, True
                insurer = round((exposure + order) * rng.uniform(1.2, 1.8), -3)
                insured_days = max(insured_days, req_days)
            else:
                history = [pick(rng, ["45 days late", "60 days late", "paid in part, rest 70 days late"]) for _ in range(3)]
                req_days = prog_days + pick(rng, [45, 60])
        elif reg == "borderline":
            k = pick(rng, ["insurer", "period", "min"])
            if k == "insurer":
                insurer = round(near(rng, exposure + order, (exposure + order) * 0.08), -3)
            elif k == "period":
                req_days = near(rng, insured_days, 10, integer=True)
            else:
                order = round(near(rng, min_order, min_order * 0.12), -2)
        else:
            k = pick(rng, ["good_uninsured", "late_but_insured", "new_customer"])
            if k == "good_uninsured":
                history = ["on time"] * 3
                req_days = insured_days + pick(rng, [15, 30])
            elif k == "late_but_insured":
                history = [pick(rng, ["25 days late", "45 days late"]) for _ in range(3)]
                insurer, insured_days = round((exposure + order) * 1.4, -3), req_days + 30
            else:
                seasons, history = 0, []
        kind = pick(rng, SEASONAL[line][1])
        if seasons == 0:
            history = []
        else:
            history = history[:min(3, seasons)]
        return {
            "regime": reg, "line": line, "season": SEASONAL[line][0],
            "distributor": f"{pick(rng, DIST_A)} {SEASON_DIST[line]}",
            "customer": _customer(rng, kind), "kind": kind, "account": _acct(rng), "credit_limit": int(limit),
            "current_exposure": int(exposure), "order_value": int(order), "goods": pick(rng, SEASON_GOODS[line]),
            "programme_days": prog_days, "requested_days": req_days, "order_before_deadline": on_time_deadline,
            "min_order": min_order, "seasons_on_programme": seasons, "history": history, "past_due_over_60": int(pd60),
            "insurer_limit": int(insurer), "insured_max_days": insured_days, "insurer": f"{pick(rng, DIST_A)} Trade Credit",
            "rep": person(rng), "rep_note": pick(rng, ["A competitor is offering them later dating.",
                                                       "They expanded to a second site this year.",
                                                       "Weather killed their season last year.", "", ""]),
            "request_date": rand_date(rng), "order_ref": ident(rng, "PB", 6, "-"),
        }

    def _track(self, f):
        h = f["history"]
        if not h:
            return 1
        bad = sum(1 for x in h if x not in ("on time", "10 days late"))
        very = sum(1 for x in h if x in ("45 days late", "60 days late", "paid in part, rest 70 days late"))
        if bad == 0:
            return 3
        if very == 0 and bad == 1:
            return 2
        return 1 if very <= 1 else 0

    def _qualifies(self, f):
        return (f["order_before_deadline"] and f["seasons_on_programme"] >= 1 and not f["past_due_over_60"]
                and f["order_value"] >= f["min_order"])

    def intended(self, f):
        peak = f["current_exposure"] + f["order_value"]
        insured = f["insurer_limit"] >= peak and f["requested_days"] <= f["insured_max_days"]
        track = self._track(f)
        q = self._qualifies(f)
        ratio = peak / f["credit_limit"]
        conc = 0 if ratio <= 1.0 else 1 if ratio <= 1.25 else 2 if ratio <= 1.6 else 3
        if not insured:
            conc = min(3, conc + 1)
        if not q and track <= 1:
            dec = "decline_extended_terms"
        elif f["requested_days"] <= f["programme_days"] and q:
            dec = "approve_requested_dating"
        elif insured and track >= 2:
            dec = "approve_requested_dating"
        elif track >= 2:
            dec = "approve_with_added_security"
        elif q:
            dec = "approve_programme_dating_only"
        else:
            dec = "decline_extended_terms"
        return {"dating_decision": dec, "peak_exposure_concern": conc, "seasonal_track_record": track,
                "insurance_cover_holds": "yes" if insured else "no", "qualifies_for_programme": "yes" if q else "no"}

    def _policy(self, f, rng):
        return [f"Early-order programme: orders placed by the programme deadline, of at least {_usd(f['min_order'], rng)}, "
                f"from accounts with one or more seasons on the programme and nothing over 60 days past due, are due "
                f"{f['programme_days']} days after shipment.",
                "Dating beyond the programme needs insured cover for the whole peak exposure and a credit period "
                "inside the insurer's maximum, or added security such as a guarantee.",
                "Customers with poor seasonal payment records are kept to standard terms."]

    def render(self, f, rng, surface):
        peak = f["current_exposure"] + f["order_value"]
        hist = "; ".join(f"season -{i}: {x}" for i, x in enumerate(f["history"], 1)) or "no previous seasons"
        note = f["rep_note"]
        if surface == "json":
            return dumps(prune({
                "order": {"ref": f["order_ref"], "value_usd": f["order_value"], "goods": f["goods"],
                          "placed_before_programme_deadline": f["order_before_deadline"],
                          "requested_due_days_after_shipment": f["requested_days"]},
                "account": {"number": f["account"], "customer": f["customer"], "business": f["kind"],
                            "distributor": f["distributor"], "credit_limit_usd": f["credit_limit"],
                            "current_exposure_usd": f["current_exposure"], "peak_exposure_usd": peak,
                            "past_due_over_60_usd": f["past_due_over_60"],
                            "seasons_on_programme": f["seasons_on_programme"], "seasonal_payment_history": f["history"]},
                "credit_insurance": {"insurer": f["insurer"], "approved_limit_usd": f["insurer_limit"],
                                     "max_credit_period_days": f["insured_max_days"]},
                "sales_note": note or None, "policy": self._policy(f, rng),
            }))
        if surface == "dating_request_form":
            body = kv_lines([("Pre-book", f["order_ref"]), ("Customer", f"{f['customer']} ({f['account']}, {f['kind']})"),
                             ("Order", f"{_usd(f['order_value'], rng)} of {f['goods']} for the {f['season']} season"),
                             ("Placed before deadline", "yes" if f["order_before_deadline"] else "no"),
                             ("Dating requested", f"{f['requested_days']} days from shipment"),
                             ("Programme dating", f"{f['programme_days']} days"),
                             ("Credit limit", _usd(f["credit_limit"], rng)),
                             ("Exposure now / at peak", f"{_usd(f['current_exposure'], rng)} / {_usd(peak, rng)}"),
                             ("Over 60 days past due", _usd(f["past_due_over_60"], rng)),
                             ("Insurer", f"{f['insurer']}: {_usd(f['insurer_limit'], rng)}, max {f['insured_max_days']} days"),
                             ("Seasonal history", hist), ("Sales comment", note or "-")], rng)
            return f"EXTENDED DATING REQUEST\n{body}\n\nPolicy:\n" + bullets(self._policy(f, rng), rng)
        s = [f"{f['customer']} wants {f['requested_days']}-day dating on a {_usd(f['order_value'], rng)} {f['season']} "
             f"pre-book of {f['goods']}; the programme gives {f['programme_days']} days.",
             f"Peak exposure would be {_usd(peak, rng)} against a {_usd(f['credit_limit'], rng)} limit.",
             f"{f['insurer']} covers {_usd(f['insurer_limit'], rng)} for up to {f['insured_max_days']} days.",
             f"History: {hist}.", "The order came in after the programme deadline." if not f["order_before_deadline"] else "",
             f"{_usd(f['past_due_over_60'], rng)} is over 60 days past due." if f["past_due_over_60"] else "",
             f"Sales: {note}" if note else ""]
        return "\n".join([f"CREDIT MEMO -- {f['order_ref']}", f"Prepared for: credit manager, {f['distributor']}", "",
                          noisy(join_sentences(s), rng, 0.08), "", "Policy:", bullets(self._policy(f, rng), rng)])

    def questions(self):
        return {
            "dating_decision": choice("How should the credit team answer the extended-dating request?", {
                "approve_requested_dating": "Approve the dating the customer asked for.",
                "approve_programme_dating_only": "Approve the order, but only at the standard programme dating.",
                "approve_with_added_security": "Approve the requested dating only once added security is in place.",
                "decline_extended_terms": "Keep the order on standard terms; no extended dating.",
            }),
            "peak_exposure_concern": score("How concerning is the account's exposure at the peak of the season?", [
                "Not concerning: within the limit and fully insured.",
                "Mild: slightly over the limit, or partly uninsured.",
                "Real: well over the limit, or not insured for this period.",
                "Serious: far over the limit with no cover to fall back on.",
            ]),
            "seasonal_track_record": score("How well has the customer paid off past seasonal programmes?", [
                "Badly: repeatedly very late or only in part.",
                "Unevenly, or there is no record to go on.",
                "Mostly well: one modest lapse.",
                "Very well: on time every season.",
            ]),
            "insurance_cover_holds": noul(
                "The credit insurance would cover this exposure on the requested dating.",
                yes="The insurer's approved limit covers the peak exposure and the requested period is within its maximum.",
                no="The peak exposure is above the insured limit, or the requested period is longer than the insurer allows."),
            "qualifies_for_programme": noul(
                "The order qualifies for the standard early-order dating programme."),
        }


# =========================================================================== schema 5
TRIGGERS = {
    "score_drop": "outside credit score fell",
    "tax_lien": "tax lien filed against the customer",
    "insurer_cut": "credit insurer cut its cover on the customer",
    "ownership_change": "the business was sold to a new owner",
    "supplier_lawsuit": "another supplier sued the customer for non-payment",
    "slow_pay_bureau": "trade bureau shows the customer paying other suppliers more slowly",
}


class CreditTriggerHold(Schema):
    name = "credit_trigger_hold"
    description = ("Orders held after an outside credit event on the customer (score drop, lien, insurer cut, change "
                   "of owner): release, cap, require security, or stop shipping.")
    surfaces = {"json": 0.5, "alert_record": 0.25, "analyst_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        sec = _sector(rng)
        trig = pick(rng, list(TRIGGERS))
        limit = max(5000, round(lognormal_amount(rng, 60000, 0.8), -3))
        exposure = round(limit * rng.uniform(0.3, 0.95), -2)
        order = round(limit * rng.uniform(0.05, 0.35), -2)
        our_dbt = rng.randint(0, 30)
        mag = pick_w(rng, {"small": 3, "near": 3, "large": 3})
        if reg == "clear":
            mag = pick(rng, ["small", "huge"])
            our_dbt = rng.randint(0, 6) if mag == "small" else rng.randint(20, 45)
        elif reg == "borderline":
            mag = "near"
        else:
            mag = pick(rng, ["large", "huge"])
            our_dbt = rng.randint(0, 5)
        m = {"small": rng.uniform(0.2, 0.7), "near": near(rng, 1.0, 0.15, lo=0.6), "large": rng.uniform(1.3, 1.9),
             "huge": rng.uniform(2.0, 4.0)}[mag]
        detail = {}
        if trig == "score_drop":
            drop = min(60, max(3, int(round(20 * m))))
            before = rng.randint(max(45, drop + 15), 98)
            detail = {"score_before": before, "score_after": before - drop}
        elif trig == "tax_lien":
            detail = {"lien_usd": int(round(limit * 0.10 * m, -2))}
        elif trig == "insurer_cut":
            cut = min(100, int(round(25 * m)))
            detail = {"cover_before": int(round(limit * rng.uniform(0.8, 1.2), -3)), "cut_pct": cut}
        elif trig == "supplier_lawsuit":
            detail = {"claim_usd": int(round(limit * 0.10 * m, -2))}
        elif trig == "slow_pay_bureau":
            jump = max(2, int(round(15 * m)))
            b0 = rng.randint(2, 12)
            detail = {"dbt_before": b0, "dbt_after": b0 + jump}
        else:
            detail = {"new_owner_guarantee_offered": chance(rng, 0.4)}
        goods = pick(rng, SECTORS[sec][3])
        gtype = _goods_type(goods)
        kind = pick(rng, SECTORS[sec][1])
        return {
            "regime": reg, "sector": sec, "distributor": _distributor(rng, sec), "customer": _customer(rng, kind),
            "kind": kind, "account": _acct(rng), "years_customer": rng.randint(1, 20),
            "trigger": trig, "trigger_detail": detail, "magnitude": round(m, 2), "credit_limit": int(limit),
            "exposure": int(exposure), "order_value": int(order), "goods": goods, "goods_type": gtype,
            "ship_in_days": pick(rng, [0, 1, 2, 4, 7, 10]), "our_dbt": our_dbt, "order_ref": ident(rng, "SO", 7, ""),
            "analyst": person(rng), "alert_date": rand_date(rng),
            "rep_note": pick(rng, ["Customer says the news is old and already sorted.",
                                   "They have paid us on time for years.", "", ""]),
        }

    def _material(self, f):
        t, d = f["trigger"], f["trigger_detail"]
        lim = f["credit_limit"]
        if t == "score_drop":
            return d["score_before"] - d["score_after"] >= 20 or d["score_after"] < 40
        if t == "tax_lien":
            return d["lien_usd"] > 0.10 * lim
        if t == "insurer_cut":
            return d["cut_pct"] > 25
        if t == "supplier_lawsuit":
            return d["claim_usd"] > 0.10 * lim
        if t == "slow_pay_bureau":
            return d["dbt_after"] - d["dbt_before"] > 15 and d["dbt_after"] > 20
        return True

    def intended(self, f):
        mat = self._material(f)
        t = f["trigger"]
        m = f["magnitude"]
        if not mat:
            sev = 1 if m >= 0.85 else 0
        elif m >= 2.0 or f["our_dbt"] > 20 or (t == "insurer_cut" and f["trigger_detail"]["cut_pct"] >= 100):
            sev = 3
        else:
            sev = 2
        after = f["exposure"] + f["order_value"]
        if not mat:
            resp = "release_and_note"
        elif sev == 3 and f["our_dbt"] > 20:
            resp = "stop_shipments"
        elif f["goods_type"] == "custom" or sev == 3:
            resp = "require_security_or_prepayment"
        elif after > 0.9 * f["credit_limit"]:
            resp = "limit_to_current_exposure"
        else:
            resp = "release_this_order_pending_review"
        if not mat:
            info = "nothing_further"
        elif t == "ownership_change":
            info = "new_owner_guarantee"
        elif t in ("insurer_cut", "slow_pay_bureau"):
            info = "insurer_or_bureau_clarification"
        else:
            info = "updated_financial_statements"
        s = f["ship_in_days"]
        press = 0 if s >= 7 else 1 if s >= 3 else 2
        if s <= 1 and f["goods_type"] in ("perishable", "custom"):
            press = 3
        return {"trigger_response": resp, "information_request": info, "deterioration_severity": sev,
                "dispatch_pressure": press, "trigger_material_under_policy": "yes" if mat else "no"}

    def _event_txt(self, f, rng):
        t, d = f["trigger"], f["trigger_detail"]
        if t == "score_drop":
            return f"outside credit score fell from {d['score_before']} to {d['score_after']}"
        if t == "tax_lien":
            return f"tax lien of {_usd(d['lien_usd'], rng)} filed against the customer"
        if t == "insurer_cut":
            return (f"credit insurer cancelled its {_usd(d['cover_before'], rng)} cover" if d["cut_pct"] >= 100
                    else f"credit insurer cut its {_usd(d['cover_before'], rng)} cover by {d['cut_pct']}%")
        if t == "supplier_lawsuit":
            return f"another supplier sued for {_usd(d['claim_usd'], rng)} unpaid"
        if t == "slow_pay_bureau":
            return f"trade bureau shows days beyond terms with other suppliers up from {d['dbt_before']} to {d['dbt_after']}"
        return ("the business was sold; the new owner has offered a personal guarantee"
                if d["new_owner_guarantee_offered"] else "the business was sold; nothing yet from the new owner")

    def _policy(self, f, rng):
        return ["A trigger is material if: score drops 20+ points or below 40; a lien or supplier claim exceeds 10% of "
                "our limit; insured cover is cut by more than 25%; other-supplier days beyond terms rise 15+ days to "
                "over 20; or the business changes hands.",
                "Immaterial triggers: the analyst releases and notes the file.",
                "Material triggers: no growth in open-terms exposure until updated information or security is on file; "
                "made-to-order goods need security or prepayment.",
                "Where our own account is also slipping badly, shipments stop pending review."]

    def render(self, f, rng, surface):
        ev = self._event_txt(f, rng)
        note = f["rep_note"]
        if surface == "json":
            return dumps(prune({
                "alert": {"date": f["alert_date"].isoformat(), "event": ev, "account": f["account"]},
                "account": {"customer": f["customer"], "business": f["kind"], "distributor": f["distributor"],
                            "years_as_customer": f["years_customer"], "credit_limit_usd": f["credit_limit"],
                            "exposure_usd": f["exposure"], "avg_days_beyond_terms_with_us": f["our_dbt"]},
                "held_order": {"ref": f["order_ref"], "value_usd": f["order_value"], "goods": f["goods"],
                               "goods_type": f["goods_type"], "ship_in_days": f["ship_in_days"]},
                "sales_note": note or None, "policy": self._policy(f, rng),
            }))
        if surface == "alert_record":
            body = kv_lines([("Monitoring alert", f["alert_date"].isoformat()), ("Account", f"{f['account']} {f['customer']}"),
                             ("Event", ev), ("Credit limit", _usd(f["credit_limit"], rng)),
                             ("Current exposure", _usd(f["exposure"], rng)),
                             ("Days beyond terms with us", f["our_dbt"]),
                             ("Held order", f"{f['order_ref']}, {_usd(f['order_value'], rng)}, {f['goods']} ({f['goods_type']})"),
                             ("Ship in", f"{f['ship_in_days']} day(s)")], rng)
            return f"CREDIT MONITORING ALERT -- {f['distributor']}\n{body}\n" + (f"Sales: {note}\n" if note else "") + \
                "\nPolicy:\n" + bullets(self._policy(f, rng), rng)
        s = [f"Monitoring flagged {f['customer']} ({f['kind']}, customer {f['years_customer']} years): {ev}.",
             f"Exposure is {_usd(f['exposure'], rng)} on a {_usd(f['credit_limit'], rng)} limit; they run {f['our_dbt']} "
             f"days beyond terms with us.",
             f"Held: {f['order_ref']} for {_usd(f['order_value'], rng)} of {f['goods']} ({f['goods_type']}), due to ship in "
             f"{f['ship_in_days']} day(s).", f"Sales adds: {note}" if note else ""]
        return "\n".join([f"ANALYST NOTE -- {f['analyst']}", "", noisy(join_sentences(s), rng, 0.08), "", "Policy:",
                          bullets(self._policy(f, rng), rng)])

    def questions(self):
        return {
            "trigger_response": choice("How should the credit team respond to the trigger for this held order?", {
                "release_and_note": "Release the order and note the event on file.",
                "release_this_order_pending_review": "Release this order, then review the account once updated "
                                                     "information arrives.",
                "limit_to_current_exposure": "Ship only as much as keeps exposure at today's level.",
                "require_security_or_prepayment": "Ship only against added security or prepayment.",
                "stop_shipments": "Stop shipping to the account until a full review is done.",
            }),
            "information_request": choice("What should the credit team ask for next?", {
                "updated_financial_statements": "Updated financial statements from the customer.",
                "insurer_or_bureau_clarification": "Clarification from the insurer or the trade bureau.",
                "new_owner_guarantee": "A guarantee or details from the new owner.",
                "nothing_further": "Nothing; the event does not call for more information.",
            }),
            "deterioration_severity": score("How badly does the event change the customer's credit standing?", [
                "Not at all.",
                "Slightly.",
                "Materially.",
                "Severely.",
            ]),
            "dispatch_pressure": score("How urgently does the held order need a credit answer?", [
                "Not urgently: it ships a week or more out.",
                "Somewhat: it ships within a few days.",
                "Urgently: it misses its dispatch within a day or two.",
                "Very urgently: perishable or made-to-order goods are waiting now.",
            ]),
            "trigger_material_under_policy": noul(
                "The triggering event is material under the credit policy's definition.",
                yes="The event meets one of the policy's materiality tests.",
                no="The event falls below every materiality test in the policy."),
        }


WORKFLOW = Workflow(
    name="wholesale_credit_hold_release",
    title="Wholesale credit-hold release",
    description="Whether a wholesale distributor releases a customer order held for credit limit, terms or a credit "
                "event, against written credit policy.",
    domain="wholesale trade / trade credit",
    schemas=[OverLimitOrderRelease(), PastDueTermsHold(), FirstOrderOnTerms(), SeasonalDatingRequest(),
             CreditTriggerHold()],
    notes="Order-release decisions only: balances appear as account totals and aging buckets. Nothing here "
          "reconciles, approves or rejects a bill, matches documents to deliveries, or processes a payment.",
)
