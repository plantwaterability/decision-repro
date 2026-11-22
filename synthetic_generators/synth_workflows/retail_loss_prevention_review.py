"""Retail loss-prevention review: shrink and exception-based case review from POS, inventory and incident records,
judged against written loss-prevention policy.

Five schemas: a point-of-sale exception report on one operator ID, an inventory count variance, a pattern of
returns on a loyalty account, a set of possibly linked theft incidents across stores, and a cash-office variance
history. The artefacts are LP records (exception reports, count variance reports, return histories, incident logs,
cash-office logs, analyst tickets). People appear ONLY as role IDs (operator OP-xxxx, key holder KH-xx, loyalty
account LY-xxxxxxxx, "Subject 1"); nothing names, describes or accuses an individual, and no protected attribute
appears anywhere. Every action is a policy step (coach, audit, open a case, restrict a return privilege, refer),
never a finding of guilt. Retailers, stores, partners and policies are invented. Each schema draws a `regime` first
so most cases sit near a written threshold, carry an innocent explanation next to a warning sign, or leave a key
record out.

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
LP_A = ["Hartfield", "Coldbrook", "Pemberton", "Ashdown", "Kingsmere", "Larchwood", "Sutton Vale", "Brayford",
        "Elmhurst", "Fallowfield", "Glenrock", "Hawksmoor", "Ivybridge", "Kirkdale", "Lowther", "Mapleton",
        "Norbury", "Orchardleigh", "Prestwick", "Redcliffe", "Thornbury", "Westcombe", "Ambleside", "Birchfield",
        "Cranmore", "Dunwell"]
LP_B = ["Superstore", "Supercenter", "Discount", "Department Store", "Value Stores", "Hypermarket", "General Store",
        "Megastore", "Outlet", "Market Hall"]
REGIONS = ["North Central", "Coastal South", "Metro East", "Valley West", "Lakes", "High Plains", "Tidewater",
           "Upstate", "Gulf", "Mountain", "Inland Empire", "Riverside"]


def _banner(rng):
    return f"{pick(rng, LP_A)} {pick(rng, LP_B)}"


def _store(rng):
    return f"#{rng.randint(100, 3999)}"


def _usd(x, rng):
    return money(x, rng, style=pick(rng, ["sym", "sym", "code"]))


def _op(rng):
    return f"OP-{rng.randint(1000, 9999)}"


# =========================================================================== schema 1
EXC = {
    "no_receipt_refund": ("refunds without a receipt", "refunds", (18, 95)),
    "post_void": ("post-voids of completed transactions", "post-voids", (12, 70)),
    "price_override": ("manual price overrides", "overrides", (4, 35)),
    "manual_discount": ("manual line discounts", "discounts", (3, 22)),
    "refund_to_gift_card": ("refunds loaded onto new gift cards", "gift-card refunds", (20, 120)),
}
CONTEXTS = {
    "product_recall": ("a supplier recall of a popular item ran during the period",
                       {"no_receipt_refund", "refund_to_gift_card"}),
    "pos_update": ("a POS software update caused scanner double-reads for about a week", {"post_void"}),
    "price_file_error": ("a price-file error left shelf labels below the system price on several lines",
                         {"price_override", "manual_discount"}),
    "desk_cover": ("the operator covered the service desk for most of the period",
                   {"no_receipt_refund", "post_void", "refund_to_gift_card"}),
    "damaged_stock_sale": ("a pallet of damaged-box stock was being sold off at the tills",
                           {"price_override", "manual_discount"}),
}
ROLES = ["cashier", "service desk associate", "self-checkout attendant", "shift lead", "front-end supervisor"]


class PosExceptionReview(Schema):
    name = "pos_exception_review"
    description = ("An exception-based POS report flags one operator ID for unusual refunds, voids, overrides or "
                   "discounts: close, coach, audit, open a formal case, or fix the store's controls.")
    surfaces = {"json": 0.5, "exception_report": 0.25, "review_ticket": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        etype = pick(rng, list(EXC))
        period = pick(rng, [14, 28])
        n_req = pick(rng, [8, 10, 12, 15])
        r_req = pick(rng, [3.0, 3.0, 4.0])
        v_req = pick(rng, [250, 400, 500, 750])
        lo, hi = EXC[etype][2]
        avg_val = rng.uniform(lo, hi)
        count = rng.randint(int(n_req * 0.6), int(n_req * 1.8))
        ratio = rng.uniform(1.5, 6)
        ctx = None
        repeat = pick(rng, [0, 0, 1, 1, 2])
        alone = rng.randint(10, 55)
        offshift = False
        audit = pick(rng, [None, None, (8, 10), (6, 10), (9, 12)])
        tenure = rng.randint(10, 320)
        prior = None
        retention = pick(rng, [14, 21, 30, 45, 60])
        oldest = rng.randint(2, retention)
        matching = [c for c, (_, types) in CONTEXTS.items() if etype in types]
        if reg == "clear":
            k = pick(rng, ["explained", "strong", "quiet"])
            if k == "explained":
                ctx = pick(rng, matching)
                count, ratio, repeat, audit = rng.randint(n_req, int(n_req * 1.6)), rng.uniform(2.0, 3.8), 0, pick(rng, [(10, 10), (9, 10), (12, 12)])
                alone = rng.randint(5, 35)
            elif k == "strong":
                count, ratio = rng.randint(int(n_req * 1.5), n_req * 3), rng.uniform(6, 11)
                repeat, alone = rng.randint(3, 7), rng.randint(70, 95)
                avg_val = rng.uniform((lo + hi) / 2, hi)
                audit = pick(rng, [None, (2, 10), (3, 10), (1, 8)])
            else:
                count, ratio, repeat = rng.randint(int(n_req * 0.3), int(n_req * 0.7)), rng.uniform(1.1, 2.2), 0
        elif reg == "borderline":
            k = pick(rng, ["count", "value", "ratio"])
            if k == "count":
                count = near(rng, n_req, 2, lo=2, integer=True)
                ratio = rng.uniform(r_req, r_req + 3)
            elif k == "value":
                count, ratio = rng.randint(n_req, int(n_req * 1.5)), rng.uniform(r_req, r_req + 3)
                avg_val = near(rng, v_req, v_req * 0.12) / count
            else:
                count, ratio = rng.randint(n_req, int(n_req * 1.5)), near(rng, r_req, 0.5, lo=1.2)
            if chance(rng, 0.3):
                ctx = pick(rng, matching)
            repeat = pick(rng, [0, 1, 2, 3])
        else:
            k = pick(rng, ["offshift", "ctx_but_repeat", "new_starter", "after_coaching"])
            count, ratio = rng.randint(n_req, n_req * 2), rng.uniform(r_req + 0.5, 8)
            if k == "offshift":
                offshift = True
                etype = pick(rng, ["no_receipt_refund", "price_override", "post_void"])
                lo, hi = EXC[etype][2]
                avg_val = rng.uniform(lo, hi)
            elif k == "ctx_but_repeat":
                ctx, repeat = pick(rng, matching), rng.randint(3, 5)
            elif k == "new_starter":
                tenure, repeat, audit = rng.randint(1, 6), 0, pick(rng, [(8, 10), (7, 10), None])
            else:
                prior = rng.randint(4, 12)
                count = near(rng, n_req, 3, lo=3, integer=True)
        store_avg = round(max(0.5, count / ratio), 1)
        value = int(round(count * avg_val))
        return {
            "regime": reg, "banner": _banner(rng), "store": _store(rng), "district": f"D{rng.randint(2, 48)}",
            "operator_id": _op(rng), "role": pick(rng, ROLES), "tenure_weeks": tenure, "exception_type": etype,
            "period_days": period, "count": count, "store_avg_per_operator": store_avg, "value_usd": value,
            "repeat_counterparty_max": repeat, "share_when_alone_pct": alone, "override_keyholder_offshift": offshift,
            "keyholder_id": f"KH-{rng.randint(10, 99)}", "context": ctx, "receipt_audit": audit,
            "prior_coaching_weeks_ago": prior, "cctv_retention_days": retention, "oldest_flag_age_days": oldest,
            "policy_count": n_req, "policy_ratio": r_req, "policy_value": v_req, "analyst": person(rng),
            "report_date": rand_date(rng), "alert_id": ident(rng, "EBR", 6, "-"),
            "manager_view": pick(rng, ["supportive", "concerned", "none", "none"]),
        }

    def _ratio(self, f):
        return f["count"] / max(0.5, f["store_avg_per_operator"])

    def intended(self, f):
        ratio = self._ratio(f)
        thr = f["count"] >= f["policy_count"] and ratio >= f["policy_ratio"] and f["value_usd"] >= f["policy_value"]
        ctx_fit = f["context"] is not None and f["exception_type"] in CONTEXTS[f["context"]][1]
        a = f["receipt_audit"]
        audit_good = a is not None and a[0] / a[1] >= 0.8
        audit_bad = a is not None and a[0] / a[1] <= 0.5
        strong = f["repeat_counterparty_max"] >= 3 or audit_bad or f["share_when_alone_pct"] >= 70
        if f["override_keyholder_offshift"]:
            outcome = "fix_store_controls"
        elif ctx_fit and not strong and (audit_good or ratio < f["policy_ratio"] * 1.4):
            outcome = "close_explained"
        elif thr and strong:
            outcome = "open_formal_case"
        elif thr:
            outcome = "audit_transactions"
        elif f["prior_coaching_weeks_ago"] is not None and f["count"] >= 0.8 * f["policy_count"]:
            outcome = "audit_transactions"
        elif ratio < 1.6 and not strong:
            outcome = "close_explained"
        else:
            outcome = "coach_and_monitor"
        left = f["cctv_retention_days"] - f["oldest_flag_age_days"]
        refundish = f["exception_type"] in ("no_receipt_refund", "refund_to_gift_card", "post_void")
        if left <= 3:
            ev = "video_footage"
        elif f["override_keyholder_offshift"]:
            ev = "override_and_schedule_logs"
        elif f["repeat_counterparty_max"] >= 2:
            ev = "tender_and_loyalty_history"
        elif refundish:
            ev = "returned_goods_check"
        else:
            ev = "video_footage"
        adj = ratio / (2.0 if ctx_fit else 1.0)
        anomaly = 0 if adj < 1.6 else 1 if adj < 3 else 2 if adj < 6 else 3
        if strong and anomaly < 2 and not ctx_fit:
            anomaly += 1
        press = 0 if left > 14 else 1 if left > 6 else 2 if left > 1 else 3
        return {"review_outcome": outcome, "evidence_priority": ev, "anomaly_strength": anomaly,
                "footage_deadline_pressure": press, "case_threshold_met": "yes" if thr else "no"}

    def _policy(self, f, rng):
        lab = EXC[f["exception_type"]][0]
        return [f"Formal case: at least {f['policy_count']} {lab} in {f['period_days']} days, at least "
                f"{f['policy_ratio']:g}x the store average for the role, and at least {_usd(f['policy_value'], rng)} in total.",
                "Below any of those: coaching by the store manager and a 30-day watch on the ID.",
                "Approvals must be keyed under the approver's own credentials; approvals under a key holder who was not "
                "on shift are a control failure, fixed with the store manager before any individual step.",
                f"CCTV is kept {f['cctv_retention_days']} days; secure footage of flagged transactions before it is overwritten.",
                "LP does not interview anyone before a formal case is opened."]

    def _facts(self, f, rng):
        s = []
        if f["context"]:
            s.append(f"Store note: {CONTEXTS[f['context']][0]}.")
        if f["override_keyholder_offshift"]:
            s.append(f"{int(f['count'] * rng.uniform(0.5, 0.9))} of the flagged transactions were approved under "
                     f"{f['keyholder_id']}, who is not on the schedule for those shifts.")
        a = f["receipt_audit"]
        if a:
            s.append(f"Sample audit: {a[0]} of {a[1]} flagged transactions matched goods or paperwork in the store.")
        else:
            s.append("No sample audit has been done yet.")
        if f["prior_coaching_weeks_ago"]:
            s.append(f"The ID was coached on this procedure {f['prior_coaching_weeks_ago']} weeks ago.")
        mv = {"supportive": "Store manager says the operator is reliable and was simply busy.",
              "concerned": "Store manager raised concerns about this register before the report.", "none": ""}
        s.append(mv[f["manager_view"]])
        return [x for x in s if x]

    def render(self, f, rng, surface):
        lab, short, _ = EXC[f["exception_type"]]
        facts = self._facts(f, rng)
        left = f["cctv_retention_days"] - f["oldest_flag_age_days"]
        if surface == "json":
            return dumps(prune({
                "alert": {"id": f["alert_id"], "banner": f["banner"], "store": f["store"], "district": f["district"],
                          "generated": f["report_date"].isoformat(), "exception": lab,
                          "period_days": f["period_days"]},
                "subject": {"operator_id": f["operator_id"], "role": f["role"], "tenure_weeks": f["tenure_weeks"]},
                "metrics": {"count": f["count"], "store_average_for_role": f["store_avg_per_operator"],
                            "total_value_usd": f["value_usd"],
                            "most_transactions_tied_to_one_card_or_loyalty_id": f["repeat_counterparty_max"],
                            "pct_when_no_other_staff_on_front_end": f["share_when_alone_pct"],
                            "oldest_flagged_transaction_days_ago": f["oldest_flag_age_days"]},
                "findings": facts, "policy": self._policy(f, rng),
            }))
        if surface == "exception_report":
            rows = [[short, f["count"], f["store_avg_per_operator"], _usd(f["value_usd"], rng),
                     f["repeat_counterparty_max"], f"{f['share_when_alone_pct']}%"]]
            return "\n".join([
                f"EXCEPTION REPORT {f['alert_id']} -- {f['banner']} {f['store']} ({f['district']})",
                f"Operator {f['operator_id']} ({f['role']}, {f['tenure_weeks']} wks)  period: last {f['period_days']} days"
                f"  run {fmt_date(f['report_date'], rng)}", "",
                table(["exception", "count", "store avg", "value", "max on one card/ID", "alone on front end"], rows, rng),
                "", f"Oldest flagged transaction: {f['oldest_flag_age_days']} days ago (CCTV kept {f['cctv_retention_days']} days).",
                "", bullets(facts, rng), "", "Policy:", "\n".join(f"  {p}" for p in self._policy(f, rng))])
        body = kv_lines([("Ticket", ident(rng, "LPR", 5, "-")), ("Alert", f["alert_id"]),
                         ("Assigned analyst", f["analyst"]), ("Store", f"{f['banner']} {f['store']}"),
                         ("Operator ID", f"{f['operator_id']} / {f['role']}"), ("Tenure", f"{f['tenure_weeks']} weeks"),
                         ("Exception", f"{f['count']} {short} in {f['period_days']} days (store avg {f['store_avg_per_operator']})"),
                         ("Value", _usd(f["value_usd"], rng)),
                         ("Footage window", f"{left} day(s) left on the oldest item"
                          if left > 0 else "oldest items already past retention")], rng)
        notes = noisy(join_sentences(facts + [
            f"{f['repeat_counterparty_max']} transactions sit on a single card or loyalty ID; "
            f"{f['share_when_alone_pct']}% happened with nobody else on the front end."]), rng, 0.1)
        return f"LP REVIEW TICKET\n{body}\n\nAnalyst notes: {notes}\n\nPolicy extract:\n" + bullets(self._policy(f, rng), rng)

    def questions(self):
        return {
            "review_outcome": choice("How should loss prevention close out its review of this exception report?", {
                "close_explained": "Close it: the recorded circumstances account for the activity.",
                "coach_and_monitor": "Have the store manager re-train on the procedure and watch the ID for 30 days.",
                "audit_transactions": "Pull the transaction detail and footage for the flagged items before going further.",
                "open_formal_case": "Open a formal loss-prevention case and hand it to the district LP manager.",
                "fix_store_controls": "Treat it as a store control failure and correct the approval process first.",
            }),
            "evidence_priority": choice("Which evidence should the analyst secure first?", {
                "video_footage": "The CCTV footage of the flagged transactions.",
                "override_and_schedule_logs": "The approval logs compared with the shift schedule.",
                "tender_and_loyalty_history": "The history of the card or loyalty ID that keeps recurring.",
                "returned_goods_check": "A check that the refunded or voided goods are physically back in the store.",
            }),
            "anomaly_strength": score("How far out of pattern is the flagged activity once the store's circumstances are allowed for?", [
                "Within the normal range for the role and period.",
                "Somewhat high, with plausible reasons on file.",
                "Clearly unusual and only partly explained.",
                "Extreme, with nothing on file to explain it.",
            ]),
            "footage_deadline_pressure": score("How soon will key footage be lost if nobody acts?", [
                "Not soon: everything flagged is well inside the retention period.",
                "Within about two weeks for the oldest items.",
                "Within a few days for the oldest items.",
                "Within a day, or some of it is already gone.",
            ]),
            "case_threshold_met": noul(
                "The activity meets the policy's thresholds for opening a formal case.",
                yes="The count, the multiple of the store average and the total value all reach the written thresholds.",
                no="At least one of the three falls short."),
        }


# =========================================================================== schema 2
CATS = [("fragrance", True, 48), ("razor blades", True, 19), ("infant formula", True, 29), ("premium spirits", True, 36),
        ("power tool batteries", True, 79), ("wireless earbuds", True, 59), ("cosmetics", True, 13),
        ("vitamins & supplements", True, 17), ("teeth whitening kits", True, 31), ("phone chargers & cables", True, 19),
        ("laundry pods", True, 15), ("allergy medicine", True, 12), ("paint sundries", False, 9),
        ("rice & grains", False, 4), ("pet food", False, 23), ("greeting cards", False, 4), ("frozen desserts", False, 6),
        ("stationery", False, 5), ("light bulbs", False, 8), ("garden hand tools", False, 15),
        ("kitchen utensils", False, 9), ("bottled water", False, 3)]
METHODS = {"third_party_scan": "third-party scan team", "two_person_count": "two-person count",
           "single_person_count": "single-person hand count", "partial_recount": "partial handheld recount"}
CONDITIONS = {"normal": "normal trading", "planogram_reset": "during a planogram reset of the aisle",
              "delivery_unbooked": "with a delivery on the dock not yet booked in"}


class InventoryShrinkVariance(Schema):
    name = "inventory_shrink_variance"
    description = ("A stock count shows a shortfall in one department; work out where it most likely came from and "
                   "what the policy requires before it is booked.")
    surfaces = {"json": 0.5, "variance_report": 0.25, "store_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        cat, high, unit_val = pick(rng, CATS)
        book = rng.randint(300, 6000)
        pct = rng.uniform(0.3, 4.5)
        method = pick_w(rng, {"third_party_scan": 4, "two_person_count": 4, "single_person_count": 1, "partial_recount": 1})
        cond = pick_w(rng, {"normal": 6, "planogram_reset": 1, "delivery_unbooked": 1})
        recv, xfer, dmg, packsz = rng.uniform(0, 0.3), rng.uniform(0, 0.15), rng.uniform(0, 0.2), False
        empties = pick(rng, [0, 0, 0, 1, 2, 3, 6])
        alarms = rng.randint(0, 25)
        backroom = rng.randint(5, 50)
        prev = round(rng.uniform(0.3, 6), 1)
        trig_v, trig_p, prot = pick(rng, [1500, 2500, 5000]), pick(rng, [3.0, 4.0, 5.0]), pick(rng, [2.0, 2.5, 3.0])
        if reg == "clear":
            k = pick(rng, ["paperwork", "theft", "tiny"])
            if k == "paperwork":
                recv, xfer, empties, method, cond = rng.uniform(0.5, 0.75), rng.uniform(0.15, 0.3), 0, "third_party_scan", "normal"
            elif k == "theft":
                high, recv, xfer, dmg = True, rng.uniform(0, 0.08), 0.0, rng.uniform(0, 0.06)
                cat, _, unit_val = pick(rng, [c for c in CATS if c[1]])
                empties, alarms, pct, prev, backroom = rng.randint(6, 25), rng.randint(15, 40), rng.uniform(4, 10), round(rng.uniform(3.5, 8), 1), rng.randint(5, 20)
                method = pick(rng, ["third_party_scan", "two_person_count"])
            else:
                pct, method, cond = rng.uniform(0.2, 0.9), "third_party_scan", "normal"
        elif reg == "borderline":
            k = pick(rng, ["trigger", "explained_half", "protect", "protect", "protect"])
            if k == "trigger":
                target = pick(rng, ["value", "pct"])
                if target == "pct":
                    pct = near(rng, trig_p, 0.5, lo=0.2)
                else:
                    pct = max(0.2, near(rng, trig_v, trig_v * 0.15) / (book * unit_val) * 100)
            elif k == "explained_half":
                recv, xfer = rng.uniform(0.3, 0.45), rng.uniform(0.05, 0.15)
            else:
                cat, high, unit_val = pick(rng, [c for c in CATS if c[1]])
                recv, xfer, dmg = rng.uniform(0, 0.15), 0.0, rng.uniform(0, 0.1)
                pct, prev = near(rng, prot * 1.45, 0.8, lo=0.5), round(near(rng, prot + 0.3, 0.6, lo=0.2), 1)
        else:
            k = pick(rng, ["empties_and_receiving", "backroom", "backroom", "bad_count", "damages", "damages"])
            if k == "empties_and_receiving":
                empties, recv = rng.randint(4, 10), rng.uniform(0.5, 0.75)
            elif k == "backroom":
                backroom, empties, alarms, recv, xfer = rng.randint(65, 90), 0, rng.randint(10, 25), rng.uniform(0, 0.1), 0.0
            elif k == "bad_count":
                method, cond, pct, empties = pick(rng, ["single_person_count", "partial_recount"]), pick(rng, ["planogram_reset", "delivery_unbooked"]), rng.uniform(4, 12), 0
            else:
                dmg, recv, xfer = rng.uniform(0.5, 0.8), rng.uniform(0, 0.15), 0.0
                packsz = chance(rng, 0.2)
        var_units = max(2, int(round(book * pct / 100)))
        recv_u, xfer_u, dmg_u = int(var_units * recv), int(var_units * xfer), int(var_units * dmg)
        return {
            "regime": reg, "banner": _banner(rng), "store": _store(rng), "category": cat, "high_theft_category": high,
            "unit_value": unit_val, "book_units": book, "variance_units": var_units,
            "variance_value": int(round(var_units * unit_val)), "variance_pct": round(var_units / book * 100, 1),
            "count_method": method, "count_conditions": cond, "receiving_discrepancy_units": recv_u,
            "unbooked_transfer_units": xfer_u, "damages_bin_unscanned_units": dmg_u, "pack_size_query": packsz,
            "empty_packaging_found": empties, "door_alarms_30d": alarms, "backroom_share_pct": backroom,
            "previous_count_variance_pct": prev, "weeks_since_last_count": rng.randint(6, 30),
            "recount_trigger_value": trig_v, "recount_trigger_pct": trig_p, "protect_threshold_pct": prot,
            "count_date": rand_date(rng), "lp_manager": person(rng), "count_ref": ident(rng, "CC", 6, "-"),
            "store_view": pick(rng, ["theft", "paperwork", "none", "none"]),
        }

    def _unreliable(self, f):
        return f["count_method"] in ("single_person_count", "partial_recount")

    def intended(self, f):
        v = f["variance_units"]
        expl = f["receiving_discrepancy_units"] + f["unbooked_transfer_units"]
        dmg = f["damages_bin_unscanned_units"]
        if self._unreliable(f) and f["count_conditions"] != "normal" and f["empty_packaging_found"] == 0:
            src = "count_error"
        elif expl >= 0.55 * v:
            src = "record_keeping_error"
        elif dmg >= 0.45 * v:
            src = "unrecorded_damage"
        elif f["backroom_share_pct"] >= 60 and f["empty_packaging_found"] <= 2:
            src = "internal_loss"
        elif f["empty_packaging_found"] >= 4 or (f["door_alarms_30d"] >= 15 and f["high_theft_category"]):
            src = "external_theft"
        elif self._unreliable(f):
            src = "count_error"
        else:
            src = "external_theft" if f["high_theft_category"] else "record_keeping_error"
        p, val = f["variance_pct"], f["variance_value"]
        mat = 0 if (p < 1 and val < 500) else 1 if (p < 2.5 and val < 2500) else 2 if (p < 5 and val < 8000) else 3
        rel = {"third_party_scan": 3, "two_person_count": 2, "single_person_count": 1, "partial_recount": 1}[f["count_method"]]
        if f["count_conditions"] != "normal":
            rel -= 1
        rel = max(0, rel)
        recount = val >= f["recount_trigger_value"] or p >= f["recount_trigger_pct"] or self._unreliable(f)
        unexpl_pct = max(0, v - expl - dmg) / f["book_units"] * 100
        protect = (f["high_theft_category"] and unexpl_pct >= f["protect_threshold_pct"]
                   and f["previous_count_variance_pct"] >= f["protect_threshold_pct"])
        return {"likely_loss_source": src, "variance_materiality": mat, "count_reliability": rel,
                "recount_before_booking": "yes" if recount else "no",
                "protective_display_warranted": "yes" if protect else "no"}

    def _policy(self, f, rng):
        return [f"A shortfall of {_usd(f['recount_trigger_value'], rng)} or more, or {f['recount_trigger_pct']}% or more of "
                f"book stock, is recounted before it is booked to shrink.",
                "Counts not done by a scan team or by two people are always recounted before booking.",
                f"High-theft categories move to locked or protected display when the shortfall not explained by "
                f"paperwork or damages reaches {f['protect_threshold_pct']}% of book stock at two counts in a row."]

    def _evidence(self, f):
        e = [f"Open receiving discrepancies on these lines: {f['receiving_discrepancy_units']} units.",
             f"Store-to-store transfers not yet booked: {f['unbooked_transfer_units']} units.",
             f"Unscanned items in the damages bin: {f['damages_bin_unscanned_units']} units.",
             f"Empty packaging recovered in the store last month: {f['empty_packaging_found']}.",
             f"Door alarms in 30 days: {f['door_alarms_30d']}.",
             f"Share of the shortfall on lines kept mainly in the stockroom: {f['backroom_share_pct']}%."]
        if f["pack_size_query"]:
            e.append("Item file query open: two lines may be set up with the wrong case size.")
        return e

    def render(self, f, rng, surface):
        theft_flag = "yes" if f["high_theft_category"] else "no"
        view = {"theft": "Store manager believes this is shoplifting.",
                "paperwork": "Store manager thinks it's a receiving problem.", "none": ""}[f["store_view"]]
        if surface == "json":
            return dumps(prune({
                "count": {"ref": f["count_ref"], "store": f"{f['banner']} {f['store']}", "date": f["count_date"].isoformat(),
                          "department": f["category"], "high_theft_category": f["high_theft_category"],
                          "method": METHODS[f["count_method"]], "conditions": CONDITIONS[f["count_conditions"]],
                          "weeks_since_previous_count": f["weeks_since_last_count"]},
                "variance": {"book_units": f["book_units"], "short_units": f["variance_units"],
                             "short_value_usd": f["variance_value"], "short_pct_of_book": f["variance_pct"],
                             "previous_count_short_pct": f["previous_count_variance_pct"]},
                "evidence": self._evidence(f), "store_comment": view or None, "policy": self._policy(f, rng),
            }))
        if surface == "variance_report":
            rows = [["book units", f["book_units"]], ["short units", f["variance_units"]],
                    ["short value", _usd(f["variance_value"], rng)], ["short % of book", f"{f['variance_pct']}%"],
                    ["previous count short %", f"{f['previous_count_variance_pct']}%"],
                    ["receiving discrepancies", f["receiving_discrepancy_units"]],
                    ["unbooked transfers", f["unbooked_transfer_units"]],
                    ["damages bin unscanned", f["damages_bin_unscanned_units"]],
                    ["empty packaging found", f["empty_packaging_found"]], ["door alarms (30d)", f["door_alarms_30d"]],
                    ["stockroom-held share of shortfall", f"{f['backroom_share_pct']}%"]]
            return "\n".join([f"COUNT VARIANCE {f['count_ref']}  {f['banner']} {f['store']}  {fmt_date(f['count_date'], rng)}",
                              f"Dept: {f['category']} (high-theft list: {theft_flag})  Method: {METHODS[f['count_method']]}, "
                              f"{CONDITIONS[f['count_conditions']]}", "", table(["line", "value"], rows, rng),
                              *(["Item file query open on case sizes."] if f["pack_size_query"] else []),
                              *([view] if view else []), "", "Policy:", bullets(self._policy(f, rng), rng)])
        s = [f"The {f['category']} count on {fmt_date(f['count_date'], rng)} ({METHODS[f['count_method']]}, "
             f"{CONDITIONS[f['count_conditions']]}) came up {f['variance_units']} units short of a book of "
             f"{f['book_units']}, about {_usd(f['variance_value'], rng)} or {f['variance_pct']}%.",
             f"Last count, {f['weeks_since_last_count']} weeks ago, was {f['previous_count_variance_pct']}% short."] + \
            self._evidence(f) + [view]
        return "\n".join([f"Subject: {f['store']} {f['category']} count variance", f"To: {f['lp_manager']}", "",
                          noisy(join_sentences(s), rng, 0.1), "", "For reference:", bullets(self._policy(f, rng), rng)])

    def questions(self):
        return {
            "likely_loss_source": choice("What most likely accounts for the shortfall?", {
                "record_keeping_error": "Paperwork or system errors such as receiving, transfer or item-set-up mistakes.",
                "external_theft": "Theft by shoppers or outside groups.",
                "internal_loss": "Loss inside the store's own operation, not yet tied to anyone.",
                "unrecorded_damage": "Damaged goods thrown away without being written off.",
                "count_error": "A faulty count that a careful recount would largely reverse.",
            }),
            "variance_materiality": score("How material is the shortfall for this store and department?", [
                "Trivial.",
                "Minor.",
                "Material.",
                "Serious.",
            ]),
            "count_reliability": score("How far can the count result be relied on?", [
                "Not at all.",
                "Weakly.",
                "Reasonably.",
                "Fully.",
            ]),
            "recount_before_booking": noul(
                "Policy requires a recount before this shortfall is booked to shrink.",
                yes="The shortfall reaches the value or percentage trigger, or the count method needs a recount.",
                no="It is under both triggers and the count was done by an approved method."),
            "protective_display_warranted": noul(
                "The department meets the policy's conditions for moving stock into locked or protected display."),
        }


# =========================================================================== schema 3
RET_CATS = {"small electrics": (25, 140), "apparel": (12, 80), "power tools": (40, 230), "beauty": (8, 55),
            "baby goods": (10, 90), "headphones": (30, 200), "home textiles": (15, 120), "toys": (10, 70)}
CONDS = ["resaleable", "resaleable", "resaleable", "opened, complete", "opened, complete"]


class ReturnPatternReview(Schema):
    name = "return_pattern_review"
    description = ("A loyalty account's return history is flagged: leave it, send a policy notice, limit or suspend "
                   "receiptless returns, or refer it to investigations.")
    surfaces = {"json": 0.5, "returns_history": 0.25, "pretty_json": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        window = pick(rng, [90, 120, 180])
        lim_n, lim_v = pick(rng, [3, 4, 5]), pick(rng, [200, 300, 500])
        cat = pick(rng, list(RET_CATS))
        lo, hi = RET_CATS[cat]
        n_rows = rng.randint(3, 8)
        n_none = rng.randint(1, n_rows)
        worn, serial, explanation, prior, arb = 0, "not_applicable", None, None, 0
        if cat in ("small electrics", "power tools", "headphones"):
            serial = pick(rng, ["matched", "not_captured"])
        if reg != "clear":
            worn = pick(rng, [0, 0, 1, 1, 2])
        if reg == "clear":
            k = pick(rng, ["light", "heavy"])
            if k == "light":
                n_rows, n_none = rng.randint(2, 4), rng.randint(0, max(0, lim_n - 2))
            else:
                n_rows = rng.randint(lim_n + 3, lim_n + 6)
                n_none = n_rows - rng.randint(0, 1)
                worn = rng.randint(2, 4)
                if serial != "not_applicable":
                    serial = pick(rng, ["mismatch_found", "matched"])
        elif reg == "borderline":
            n_rows = rng.randint(lim_n, lim_n + 3)
            n_none = near(rng, lim_n, 1, lo=1, hi=n_rows, integer=True)
        else:
            k = pick(rng, ["bracketing", "gift", "notice_repeat", "notice_repeat", "worn_but_under"])
            n_rows = rng.randint(lim_n + 2, lim_n + 5)
            if k == "bracketing":
                explanation, n_none = "online_orders", rng.randint(1, lim_n)
            elif k == "gift":
                explanation, n_none = "gift_receipts", rng.randint(max(1, lim_n - 1), lim_n + 1)
            elif k == "notice_repeat":
                n_none, prior = rng.randint(lim_n + 1, n_rows), rng.randint(20, 150)
            else:
                n_none, worn = rng.randint(1, lim_n), rng.randint(2, 3)
        if prior is None and chance(rng, 0.3):
            prior = rng.randint(15, 300)
        n_none = max(0, min(n_none, n_rows))
        other = n_rows - n_none
        # the other rows: receipted, or receiptless-looking rows the policy counts as receipted
        tags = ["none"] * n_none
        for _ in range(other):
            if explanation == "online_orders":
                tags.append(pick(rng, ["online order (not linked)", "online order (not linked)", "e-receipt"]))
            elif explanation == "gift_receipts":
                tags.append(pick(rng, ["gift receipt", "gift receipt", "e-receipt"]))
            else:
                tags.append(pick(rng, ["e-receipt", "paper receipt", "gift receipt"]))
        if explanation is None and reg != "clear" and "none" in tags and chance(rng, 0.2):
            tags[tags.index("none")] = "online order (not linked)"
        rng.shuffle(tags)
        rows = []
        for i, t in enumerate(tags):
            val = round(rng.uniform(lo, hi), 2)
            cond = pick(rng, CONDS)
            rows.append({"days_ago": 0, "store": _store(rng), "value": val, "receipt": t, "condition": cond,
                         "refunded_above_paid": False})
        for i in rng.sample(range(len(rows)), min(worn, len(rows))):
            rows[i]["condition"] = pick(rng, ["worn or used", "tags removed"])
        for r in rows:
            if r["condition"] in ("worn or used", "tags removed"):
                pass
        wcount = sum(1 for r in rows if r["condition"] in ("worn or used", "tags removed"))
        if wcount > worn and reg == "clear":
            for r in rows:
                if r["condition"] in ("worn or used", "tags removed") and wcount > worn:
                    r["condition"], wcount = "resaleable", wcount - 1
        none_rows = [r for r in rows if r["receipt"] == "none"]
        if none_rows and chance(rng, 0.3):
            for r in rng.sample(none_rows, min(len(none_rows), rng.randint(1, 2))):
                r["refunded_above_paid"] = True
        days = sorted(rng.sample(range(1, window), len(rows)), reverse=True)
        for r, d in zip(rows, days):
            r["days_ago"] = d
        return {
            "regime": reg, "banner": _banner(rng), "loyalty_id": f"LY-{rng.randint(10**7, 10**8 - 1)}",
            "account_age_months": rng.randint(2, 140), "category": cat, "window_days": window,
            "limit_count": lim_n, "limit_value": lim_v, "returns": rows, "serial_status": serial,
            "explanation": explanation, "prior_notice_days_ago": prior,
            "purchases_value_window": round(sum(r["value"] for r in rows) * rng.uniform(0.6, 3.5), 2),
            "vendor_risk_score": pick(rng, [None, rng.randint(5, 60), rng.randint(60, 98)]),
            "analyst": person(rng), "alert_id": ident(rng, "RTN", 6, "-"), "review_date": rand_date(rng),
        }

    def _true_none(self, f):
        return [r for r in f["returns"] if r["receipt"] == "none"]

    def intended(self, f):
        none = self._true_none(f)
        n, v = len(none), sum(r["value"] for r in none)
        breached = n > f["limit_count"] or v > f["limit_value"]
        worn = [r for r in f["returns"] if r["condition"] in ("worn or used", "tags removed")]
        arb = [r for r in f["returns"] if r["refunded_above_paid"]]
        recent_notice = f["prior_notice_days_ago"] is not None and f["prior_notice_days_ago"] <= 90
        if f["serial_status"] == "mismatch_found" or len(worn) >= 3:
            act = "refer_to_investigations"
        elif breached and recent_notice:
            act = "suspend_receiptless_returns"
        elif breached:
            act = "store_credit_only"
        elif n == f["limit_count"] or v >= 0.8 * f["limit_value"] or len(worn) >= 1:
            act = "send_policy_notice"
        else:
            act = "no_restriction"
        unlinked = any(r["receipt"] == "online order (not linked)" for r in f["returns"])
        if f["serial_status"] == "not_captured":
            chk = "verify_serial_numbers"
        elif unlinked:
            chk = "link_online_orders"
        elif 1 <= len(worn) <= 2:
            chk = "inspect_returned_items"
        else:
            chk = "no_check_needed"
        pts = (1 if breached else 0) + (1 if worn else 0) + (1 if len({r["store"] for r in f["returns"]}) >= 4 else 0) \
            + (2 if f["serial_status"] == "mismatch_found" else 0) + (1 if len(arb) >= 2 else 0) \
            + (1 if (f["vendor_risk_score"] or 0) >= 80 else 0)
        strength = 0 if pts == 0 else 1 if pts <= 2 else 2 if pts == 3 else 3
        loss = sum(r["value"] for r in worn) + sum(r["value"] * 0.3 for r in arb)
        if f["serial_status"] == "mismatch_found":
            loss += max(r["value"] for r in f["returns"])
        money_lvl = 0 if loss < 25 else 1 if loss < 100 else 2 if loss < 280 else 3
        return {"return_privilege_action": act, "check_before_restricting": chk, "abuse_pattern_strength": strength,
                "likely_loss_to_retailer": money_lvl, "receiptless_limit_breached": "yes" if breached else "no"}

    def _policy(self, f, rng):
        return [f"Receiptless returns are limited to {f['limit_count']} returns and {_usd(f['limit_value'], rng)} per "
                f"account in any rolling {f['window_days']} days; gift receipts and the account's own online orders "
                f"count as receipted.",
                "Over the limit: further receiptless refunds are store credit only, with ID, and a written notice is sent.",
                "A further breach within 90 days of a notice suspends receiptless returns for 90 days.",
                "At or near the limit, or with a worn item returned: a policy reminder may be sent.",
                "A serial-number mismatch, or three or more worn or tag-removed returns, goes to investigations.",
                "Receiptless refunds are paid at the lowest price in the last 30 days."]

    def _rows(self, f, rng):
        out = []
        for r in f["returns"]:
            out.append([f"-{r['days_ago']}d", r["store"], _usd(r["value"], rng), r["receipt"], r["condition"],
                        "yes" if r["refunded_above_paid"] else ""])
        return out

    def _facts(self, f):
        s = []
        if f["serial_status"] == "mismatch_found":
            s.append("One returned unit's serial number does not match any sale on record.")
        elif f["serial_status"] == "not_captured":
            s.append("Serial numbers were not captured on the returns.")
        elif f["serial_status"] == "matched":
            s.append("Serial numbers on the returned units match sales on record.")
        if f["prior_notice_days_ago"] is not None:
            s.append(f"A written policy notice was sent {f['prior_notice_days_ago']} days ago.")
        if f["vendor_risk_score"] is not None:
            s.append(f"Return-authorisation vendor score: {f['vendor_risk_score']}/100.")
        return s

    def render(self, f, rng, surface):
        sys_n = sum(1 for r in f["returns"] if r["receipt"] != "e-receipt" and r["receipt"] != "paper receipt")
        if surface in ("json", "pretty_json"):
            obj = prune({
                "alert": {"id": f["alert_id"], "banner": f["banner"], "raised": f["review_date"].isoformat(),
                          "system_receiptless_count": sys_n, "window_days": f["window_days"]},
                "account": {"loyalty_id": f["loyalty_id"], "age_months": f["account_age_months"],
                            "purchases_in_window_usd": f["purchases_value_window"], "main_category": f["category"]},
                "returns": [{"days_ago": r["days_ago"], "store": r["store"], "refund_usd": r["value"],
                             "receipt": r["receipt"], "condition": r["condition"],
                             "refunded_above_price_paid": r["refunded_above_paid"] or None} for r in f["returns"]],
                "checks": self._facts(f), "policy": self._policy(f, rng),
            })
            return dumps(obj) if surface == "json" else dumps_pretty(obj, rng)
        return "\n".join([
            f"RETURNS HISTORY  {f['loyalty_id']}  ({f['banner']}, alert {f['alert_id']})",
            f"Account age {f['account_age_months']} months; purchases in window {_usd(f['purchases_value_window'], rng)}; "
            f"system flagged {sys_n} receiptless returns in {f['window_days']} days.", "",
            table(["when", "store", "refund", "receipt", "condition", "above price paid"], self._rows(f, rng), rng), "",
            bullets(self._facts(f), rng), "", "Returns policy:", "\n".join(f"  - {p}" for p in self._policy(f, rng))])

    def questions(self):
        return {
            "return_privilege_action": choice("What should happen to this account's return privileges?", {
                "no_restriction": "Nothing; the account's returns are within policy.",
                "send_policy_notice": "Send the written returns-policy reminder and change nothing else.",
                "store_credit_only": "Make further receiptless refunds store credit only, with ID, and send the notice.",
                "suspend_receiptless_returns": "Suspend receiptless returns on the account for 90 days.",
                "refer_to_investigations": "Refer the account to the investigations team.",
            }),
            "check_before_restricting": choice("What should be checked before any restriction is applied?", {
                "verify_serial_numbers": "Match the returned units' serial numbers against sales.",
                "link_online_orders": "Link the returns to the account's own online orders.",
                "inspect_returned_items": "Inspect the condition of the returned items.",
                "no_check_needed": "Nothing further; the record is sufficient.",
            }),
            "abuse_pattern_strength": score("How strongly does the history point to abuse of the returns policy?", [
                "Not at all.",
                "Faintly.",
                "Clearly.",
                "Strongly.",
            ]),
            "likely_loss_to_retailer": score("How much has the retailer probably lost on these returns, beyond goods it can resell?", [
                "Next to nothing.",
                "A small amount.",
                "A noticeable amount.",
                "A large amount.",
            ]),
            "receiptless_limit_breached": noul(
                "The account has gone over the written receiptless-return limit for the window.",
                yes="Counting only returns the policy treats as receiptless, the count or value is over the limit.",
                no="Once gift receipts and the account's own online orders are set aside, it is within both limits."),
        }


# =========================================================================== schema 4
ORC_METHODS = ["concealment in a lined bag", "full trolley pushed out through the entrance",
               "shelf sweep into a holdall", "distraction at the counter while a second person takes stock",
               "fitting-room concealment", "security tags cut off in the aisle"]
ORC_TARGETS = ["fragrance", "razor blades", "power tools", "premium spirits", "infant formula", "designer handbags",
               "headphones", "cosmetics", "laundry pods", "allergy medicine"]
VEHICLES = ["grey estate car", "white panel van", "dark blue saloon", "silver hatchback", "black SUV",
            "red pickup", "green people carrier"]
STAFF = {"none": "no contact with staff", "verbal": "verbal threats to a colleague",
         "push": "a colleague pushed aside at the door", "weapon": "a blade shown when a colleague approached"}


class OrcIncidentLinkage(Schema):
    name = "orc_incident_linkage"
    description = ("Several theft incidents across stores may be the same group: log them, alert the region, refer "
                   "to police under the aggregation policy, or harden one store.")
    surfaces = {"json": 0.5, "incident_log": 0.25, "analyst_brief": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        n = rng.randint(3, 6)
        thr = pick(rng, [1000, 1500, 2500])
        span_max = pick(rng, [90, 120, 180])
        m0, t0, veh0 = pick(rng, ORC_METHODS), pick(rng, ORC_TARGETS), pick(rng, VEHICLES)
        plate0 = f"{pick(rng, 'ABDEFGHKLMNPRSTVWXY')}{rng.randint(1, 9)}{pick(rng, 'ABDEFGHKLMNPRSTVWXY')}"
        stores = [_store(rng) for _ in range(n)]
        if reg == "clear":
            mode = pick(rng, ["linked", "unlinked"])
        elif reg == "borderline":
            mode = pick(rng, ["partial", "linked_low_value"])
        else:
            mode = pick(rng, ["plate_only", "same_store", "linked_low_value", "partial"])
        incs = []
        day = rng.randint(0, 6)
        interval = rng.randint(3, 25)
        shrinking = chance(rng, 0.4)
        for i in range(n):
            linked = (i == 0) or mode in ("linked", "linked_low_value") or (mode == "partial" and chance(rng, 0.5)) \
                or (mode == "plate_only" and chance(rng, 0.6))
            if mode == "unlinked" and i > 0:
                linked = False
            meth = m0 if linked and mode != "plate_only" else pick(rng, ORC_METHODS)
            targ = t0 if linked and (mode != "partial" or chance(rng, 0.6)) else pick(rng, ORC_TARGETS)
            show_plate = linked and (mode in ("linked", "plate_only") or (mode == "partial" and chance(rng, 0.3)))
            val = rng.randint(150, 1400)
            if mode == "linked_low_value":
                val = rng.randint(90, max(120, int(thr / n * 1.3)))
            where = stores[0] if mode == "same_store" else (stores[i] if chance(rng, 0.85) else pick(rng, stores))
            incs.append({"store": where,
                         "days_ago": day, "time": f"{rng.randint(9, 21):02d}:{pick(rng, ['05', '15', '20', '35', '40', '50'])}",
                         "value": val, "people": pick(rng, [1, 2, 2, 3, 3, 4]), "method": meth, "target": targ,
                         "vehicle": veh0 if show_plate else pick(rng, [None, None, pick(rng, VEHICLES)]),
                         "plate": plate0 if show_plate else None,
                         "staff": pick_w(rng, {"none": 20, "verbal": 2, "push": 1, "weapon": 0.6}),
                         "police_report": chance(rng, 0.6), "video": pick(rng, ["clear", "clear", "partial", "none"])})
            day += max(1, int(interval * (0.7 if shrinking and i > 0 else 1.0) * rng.uniform(0.7, 1.3)))
            if shrinking:
                interval = max(1, int(interval * 0.7))
        if mode == "same_store":
            for inc in incs[1:]:
                inc["method"], inc["target"] = pick(rng, ORC_METHODS), pick(rng, ORC_TARGETS)
        return {"regime": reg, "mode": mode, "banner": _banner(rng), "region": pick(rng, REGIONS), "incidents": incs,
                "policy_threshold": thr, "policy_span_days": span_max, "analyst": person(rng),
                "brief_date": rand_date(rng), "case_ref": ident(rng, "ORC", 5, "-")}

    def _sim(self, a, b):
        return (a["method"] == b["method"]) + (a["target"] == b["target"]) + 2 * (a["plate"] is not None and a["plate"] == b["plate"])

    def _linked(self, f):
        incs = f["incidents"]
        return [incs[0]] + [x for x in incs[1:] if self._sim(incs[0], x) >= 2]

    def intended(self, f):
        incs = f["incidents"]
        linked = self._linked(f)
        val = sum(x["value"] for x in linked)
        span = max(x["days_ago"] for x in linked) - min(x["days_ago"] for x in linked)
        stores = {x["store"] for x in linked}
        agg = len(linked) >= 2 and val >= f["policy_threshold"] and len(stores) >= 2 and span <= f["policy_span_days"]
        plates = sum(1 for x in linked[1:] if x["plate"] and x["plate"] == incs[0]["plate"])
        frac = (len(linked) - 1) / (len(incs) - 1)
        if len(linked) == 1:
            conf = 0
        elif plates >= 2 or (plates >= 1 and frac >= 0.6):
            conf = 3
        elif frac >= 0.5 or plates >= 1:
            conf = 2
        else:
            conf = 1
        hits = {}
        for x in incs:
            hits[x["store"]] = hits.get(x["store"], 0) + 1
        if len(linked) >= 2 and agg:
            route = "law_enforcement_referral"
        elif len(linked) >= 2 and len(stores) >= 2:
            route = "regional_store_alert"
        elif max(hits.values()) >= 2:
            route = "store_protection_measures"
        else:
            route = "log_and_monitor"
        days = sorted(x["days_ago"] for x in linked)
        if len(days) >= 2:
            gaps = [b - a for a, b in zip(days, days[1:])]
            recent_gap = gaps[0]
            due = recent_gap - days[0]
            press = 3 if due <= 0 else 2 if due <= 3 else 1 if due <= 10 else 0
        else:
            press = 1 if days[0] <= 3 else 0
        safety = any(x["staff"] != "none" for x in incs)
        return {"case_route": route, "linkage_confidence": conf, "next_hit_pressure": press,
                "aggregation_threshold_met": "yes" if agg else "no", "staff_safety_risk": "yes" if safety else "no"}

    def _policy(self, f, rng):
        return [f"Refer to police as one case when linked incidents total {_usd(f['policy_threshold'], rng)} or more "
                f"across two or more stores within {f['policy_span_days']} days.",
                "Linked incidents below that: a regional alert to stores in the area.",
                "Repeat hits on one store without a wider link: that store's protection measures are reviewed.",
                "Staff never pursue or detain; where there has been any threat, stores brief colleagues on disengagement."]

    def _line(self, x, rng):
        veh = f"{x['vehicle']}" + (f", plate ending {x['plate']}" if x["plate"] else "") if x["vehicle"] else "no vehicle seen"
        return (f"{x['store']} {x['days_ago']}d ago {x['time']}: {x['people']} people, {x['method']}, {x['target']}, "
                f"{_usd(x['value'], rng)}; {veh}; {STAFF[x['staff']]}; police report {'yes' if x['police_report'] else 'no'}; "
                f"video {x['video']}")

    def render(self, f, rng, surface):
        incs = f["incidents"]
        if surface == "json":
            return dumps(prune({
                "case": {"ref": f["case_ref"], "banner": f["banner"], "region": f["region"],
                         "prepared": f["brief_date"].isoformat(), "analyst": f["analyst"]},
                "incidents": [{"store": x["store"], "days_ago": x["days_ago"], "time": x["time"],
                               "people": x["people"], "method": x["method"], "items_targeted": x["target"],
                               "loss_usd": x["value"], "vehicle": x["vehicle"], "partial_plate": x["plate"],
                               "staff_contact": STAFF[x["staff"]], "police_report": x["police_report"],
                               "video": x["video"]} for x in incs],
                "policy": self._policy(f, rng),
            }))
        if surface == "incident_log":
            rows = [[x["store"], f"{x['days_ago']}d", x["time"], x["people"], x["method"], x["target"],
                     _usd(x["value"], rng), (x["vehicle"] or "-") + (f" ..{x['plate']}" if x["plate"] else ""),
                     x["staff"]] for x in incs]
            return "\n".join([f"ORC INCIDENT LOG {f['case_ref']} -- {f['banner']}, {f['region']} region", "",
                              table(["store", "when", "time", "ppl", "method", "items", "loss", "vehicle", "staff"],
                                    rows, rng), "", "Staff contact key: " + "; ".join(f"{k} = {v}" for k, v in STAFF.items()),
                              "", "Policy:", bullets(self._policy(f, rng), rng)])
        s = [f"{len(incs)} incidents in the {f['region']} region are under review as a possible series."]
        lines = [noisy(self._line(x, rng), rng, 0.05) for x in incs]
        return "\n".join([f"BRIEF {f['case_ref']} ({fmt_date(f['brief_date'], rng)}) -- {f['analyst']}", "",
                          join_sentences(s), "", "\n".join(f"{i}. {ln}" for i, ln in enumerate(lines, 1)), "",
                          "Policy:", bullets(self._policy(f, rng), rng)])

    def questions(self):
        return {
            "case_route": choice("What should loss prevention do with these incidents now?", {
                "log_and_monitor": "Log them and keep watching; nothing yet ties them together.",
                "regional_store_alert": "Alert the stores in the region to the pattern.",
                "law_enforcement_referral": "Package the linked incidents and refer them to police as one case.",
                "store_protection_measures": "Review protection measures at the store that keeps being hit.",
            }),
            "linkage_confidence": score("How confident can the analyst be that the incidents are the work of one group?", [
                "Not at all: nothing ties them together.",
                "Slightly: a shared method or target, nothing more.",
                "Moderately: several shared features across most incidents.",
                "Highly: a vehicle and method recur across several incidents.",
            ]),
            "next_hit_pressure": score("How soon is the next hit likely, given the pattern so far?", [
                "Not soon: no sign of a continuing series.",
                "Within a few weeks.",
                "Within days.",
                "Any time now: the series is overdue or speeding up.",
            ]),
            "aggregation_threshold_met": noul(
                "The incidents that can be linked together meet the policy's police-referral threshold.",
                yes="Counting only linked incidents, the total value, store count and time span all meet the policy.",
                no="Too few incidents can be linked, or their total, spread or time span falls short."),
            "staff_safety_risk": noul(
                "The incidents show a risk to staff safety that should shape how stores respond."),
        }


# =========================================================================== schema 5
SCOPES = {"register_till": "register till", "service_desk_till": "service desk till",
          "cash_recycler": "self-checkout cash recycler", "store_safe": "store safe"}


class CashOfficeVariance(Schema):
    name = "cash_office_variance"
    description = ("A till or safe keeps counting short or over: no action, recount and procedure check, coaching, a "
                   "documented warning, or referral to LP.")
    surfaces = {"json": 0.5, "cash_office_log": 0.25, "manager_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        scope = pick_w(rng, {"register_till": 5, "service_desk_till": 2, "cash_recycler": 1, "store_safe": 1})
        x_thr, y_thr = pick(rng, [5, 10, 20]), pick(rng, [75, 100, 150])
        shared = chance(rng, 0.1) if scope != "store_safe" else chance(rng, 0.5)
        blind = chance(rng, 0.85)
        pickups = rng.randint(6, 20)
        signed = rng.randint(int(pickups * 0.8), pickups) if chance(rng, 0.85) else rng.randint(int(pickups * 0.4), pickups)
        fault = chance(rng, 0.12) or (scope == "cash_recycler" and chance(rng, 0.5))
        evs = []
        n = rng.randint(2, 7)
        style = pick_w(rng, {"mixed_small": 4, "shorts_round": 2, "shorts_odd": 2})
        if reg == "clear":
            k = pick(rng, ["noise", "round_shorts"])
            if k == "noise":
                style, shared, fault = "mixed_small", False, False
            else:
                style, shared, blind, fault = "shorts_round", False, True, False
                signed = pickups
                n = rng.randint(4, 7)
        elif reg == "borderline":
            style = pick(rng, ["shorts_odd", "shorts_round"])
            n = rng.randint(2, 4)
        else:
            k = pick(rng, ["shared_big", "fault_shorts", "no_blind"])
            style = "shorts_round"
            if k == "shared_big":
                shared = True
            elif k == "fault_shorts":
                fault = True
                style = pick(rng, ["mixed_small", "shorts_odd"])
            else:
                blind, signed = False, rng.randint(int(pickups * 0.3), int(pickups * 0.6))
                shared = chance(rng, 0.4)
        for _ in range(n):
            if style == "mixed_small":
                a = round(rng.uniform(-x_thr * 1.15, x_thr * 1.0), 2)
            elif style == "shorts_round":
                a = -float(pick(rng, [10, 20, 20, 20, 50, 50, 100] if y_thr > 75 else [10, 20, 20, 50]))
                if reg == "borderline" and chance(rng, 0.4):
                    a = -float(near(rng, y_thr, 15, lo=x_thr + 1, integer=True))
            else:
                a = -round(rng.uniform(x_thr * 0.6, x_thr * 3.5), 2)
            evs.append(a)
        if reg == "borderline" and style == "shorts_odd":
            evs[0] = -round(near(rng, x_thr, x_thr * 0.3, lo=0.5), 2)
        days = sorted(rng.sample(range(1, 90), n), reverse=True)
        ops = [_op(rng) for _ in range(rng.randint(2, 4) if shared else 1)]
        return {"regime": reg, "banner": _banner(rng), "store": _store(rng), "scope": scope, "operators": ops,
                "shared_drawer": shared, "blind_count": blind, "pickups_total": pickups,
                "pickups_dual_signed": signed, "equipment_fault_logged": fault,
                "events": [{"days_ago": d, "amount": a} for d, a in zip(days, evs)],
                "policy_minor": x_thr, "policy_major": y_thr, "tenure_weeks": rng.randint(2, 300),
                "manager": person(rng), "log_date": rand_date(rng),
                "manager_view": pick(rng, ["trusts", "suspects", "none"])}

    def _shorts(self, f):
        return [e["amount"] for e in f["events"] if e["amount"] < 0]

    def intended(self, f):
        sh = self._shorts(f)
        over_minor = [a for a in sh if -a > f["policy_minor"]]
        big = [a for a in sh if -a > f["policy_major"]]
        thr = len(over_minor) >= 3 or bool(big)
        overs = [e for e in f["events"] if e["amount"] > 0]
        round_bills = sum(1 for a in sh if float(-a).is_integer() and int(-a) % 10 == 0)
        dual_rate = f["pickups_dual_signed"] / f["pickups_total"]
        if f["equipment_fault_logged"] and (overs or round_bills < 2):
            acct = "equipment_or_system_fault"
        elif f["shared_drawer"]:
            acct = "shared_till_no_attribution"
        elif not f["blind_count"] or dual_rate < 0.7:
            acct = "cannot_tell_without_audit"
        else:
            acct = "single_operator_accountable"
        if acct in ("equipment_or_system_fault", "shared_till_no_attribution", "cannot_tell_without_audit"):
            resp = "recount_and_procedure_review" if (thr or len(over_minor) >= 2) else "no_action"
        elif round_bills >= 3 and not overs:
            resp = "refer_to_lp_investigation"
        elif thr:
            resp = "documented_warning"
        elif over_minor:
            resp = "coach_operator"
        else:
            resp = "no_action"
        if round_bills >= 3 and not overs:
            concern = 3
        elif len(over_minor) >= 3 or big or round_bills >= 2:
            concern = 2
        elif over_minor:
            concern = 1
        else:
            concern = 0
        comp = (1 if f["blind_count"] else 0) + (1 if dual_rate >= 0.9 else 0) + (0 if f["shared_drawer"] else 1)
        return {"variance_response": resp, "accountability_basis": acct, "pattern_concern": concern,
                "procedure_compliance": comp, "escalation_threshold_met": "yes" if thr else "no"}

    def _policy(self, f, rng):
        return [f"Variances of {_usd(f['policy_minor'], rng)} or less are counting noise and need no action.",
                f"Three shortages above {_usd(f['policy_minor'], rng)} in 90 days, or any single shortage above "
                f"{_usd(f['policy_major'], rng)}, lead to a documented warning.",
                "Three or more whole-note shortages with no overs in 90 days go to LP whatever their size.",
                "No individual step is taken on a till shared by several operators, or where counts were not blind "
                "and pickups not signed by two people; the procedure is fixed first.",
                "Variances on equipment with an open fault ticket are reviewed with the equipment log."]

    def render(self, f, rng, surface):
        evs = f["events"]
        view = {"trusts": "Store manager vouches for the operator(s).",
                "suspects": "Store manager thinks money is going missing deliberately.", "none": ""}[f["manager_view"]]
        meta = [f"Till / scope: {SCOPES[f['scope']]}",
                f"Operator IDs on it: {', '.join(f['operators'])}" + (" (shared)" if f["shared_drawer"] else ""),
                f"Closing counts blind: {'yes' if f['blind_count'] else 'no'}",
                f"Pickups signed by two people: {f['pickups_dual_signed']} of {f['pickups_total']}",
                "Open fault ticket on the equipment: " + ("yes" if f["equipment_fault_logged"] else "no")]
        if surface == "json":
            return dumps(prune({
                "location": {"banner": f["banner"], "store": f["store"], "scope": SCOPES[f["scope"]],
                             "log_date": f["log_date"].isoformat()},
                "controls": {"operator_ids": f["operators"], "shared_drawer": f["shared_drawer"],
                             "blind_closing_count": f["blind_count"], "pickups_total": f["pickups_total"],
                             "pickups_dual_signed": f["pickups_dual_signed"],
                             "open_equipment_fault": f["equipment_fault_logged"],
                             "operator_tenure_weeks": f["tenure_weeks"] if not f["shared_drawer"] else None},
                "variances_90d": [{"days_ago": e["days_ago"], "amount_usd": e["amount"]} for e in evs],
                "manager_comment": view or None, "policy": self._policy(f, rng),
            }))
        if surface == "cash_office_log":
            rows = [[f"-{e['days_ago']}d", f"{e['amount']:+.2f}"] for e in evs]
            return "\n".join([f"CASH OFFICE VARIANCE LOG  {f['banner']} {f['store']}  as of {fmt_date(f['log_date'], rng)}",
                              "\n".join(meta), "", table(["day", "variance USD"], rows, rng), "", view, "",
                              "Cash policy:", bullets(self._policy(f, rng), rng)])
        s = ["Variances over the last 90 days: " + ", ".join(f"{e['amount']:+.2f} ({e['days_ago']}d ago)" for e in evs) + ".",
             view]
        return "\n".join([f"Note to LP from {f['manager']}, {f['banner']} {f['store']}", "", "\n".join(meta), "",
                          noisy(join_sentences(s), rng, 0.1), "", "Policy (cash handling):",
                          bullets(self._policy(f, rng), rng)])

    def questions(self):
        return {
            "variance_response": choice("What should the store and LP do about these variances?", {
                "no_action": "Nothing; the variances are counting noise within policy.",
                "recount_and_procedure_review": "Recount and review the cash-handling procedure, with no individual step.",
                "coach_operator": "Coach the operator on cash handling and keep watching the till.",
                "documented_warning": "Issue the documented warning the policy sets for repeated shortages.",
                "refer_to_lp_investigation": "Refer the till's history to loss prevention for investigation.",
            }),
            "accountability_basis": choice("On what basis can the variances be attributed under the policy?", {
                "single_operator_accountable": "One operator ran the till under blind counts and signed pickups.",
                "shared_till_no_attribution": "The till was shared, so no single operator can be held to it.",
                "equipment_or_system_fault": "An equipment or system fault is the likeliest cause.",
                "cannot_tell_without_audit": "Counting controls were too weak to tell without an audit.",
            }),
            "pattern_concern": score("How concerning is the pattern of variances?", [
                "Not concerning: small amounts in both directions.",
                "Mildly concerning: a few shortages above the noise level.",
                "Concerning: repeated or large shortages.",
                "Highly concerning: whole-note shortages and no overs.",
            ]),
            "procedure_compliance": score("How well did the store follow the cash-handling procedure?", [
                "Poorly: none of the controls were in place.",
                "Partly: one control was in place.",
                "Mostly: two controls were in place.",
                "Fully: blind counts, dual-signed pickups and a single operator.",
            ]),
            "escalation_threshold_met": noul(
                "The variance history meets the policy's threshold for a documented warning.",
                yes="There are three shortages above the noise level in 90 days, or one above the large-shortage limit.",
                no="Neither condition is met."),
        }


WORKFLOW = Workflow(
    name="retail_loss_prevention_review",
    title="Retail loss-prevention review",
    description="Shrink and exception-based case review from POS, inventory and incident records, against written "
                "loss-prevention policy.",
    domain="retail trade / loss prevention",
    schemas=[PosExceptionReview(), InventoryShrinkVariance(), ReturnPatternReview(), OrcIncidentLinkage(),
             CashOfficeVariance()],
    notes="People appear only as role IDs or 'Subject n'; no names, descriptions or protected attributes of anyone "
          "reviewed, and every outcome is a policy step rather than a finding of guilt. No shopper conversations.",
)
