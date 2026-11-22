"""Shipment exception handling: what a (fictional) logistics control tower, terminal or claims desk does when
freight goes wrong in transit.

Late loads against delivery appointments, damage found at delivery, reefer temperature excursions, freight that has
stopped scanning, and dangerous-goods problems found at a terminal. The artefacts are operational records (TMS
exception records, carrier status feeds, over/short/damage reports, reefer downloads, trace logs, terminal DG
reports), never a conversation with a customer: nothing here asks what to reply to anyone. Shippers, carriers,
towns, policies and routing guides are invented; UN numbers and hazard classes are the public ones. Each schema
draws a `regime` first so most cases sit near a tolerance, pit a document against a claim, or miss a timestamp.
"""
from __future__ import annotations

from datetime import timedelta

from .base import (
    Schema,
    Workflow,
    bullets,
    chance,
    choice,
    dumps,
    fmt_date,
    fmt_ts,
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
    show,
    table,
)

TOWNS = ["Brackenmoor", "Pellbridge", "Wexmoor", "Dunhollow", "Corriston", "Quillhaven", "Marrowfield", "Tamsley",
         "Saltwick", "Orlan Bay", "Kestrelford", "Harrowmere"]
CARRIERS = ["Basalt Freight Lines", "Ironbark Transport", "Valewood Linehaul", "Quarry Road Carriers",
            "Saltwick Express", "Dunhollow Trucking", "Ashgrove Motor Freight", "Copperline Carriers"]
TOWER = "Wexmoor Logistics control tower"


def _lane(rng):
    a, b = rng.sample(TOWNS, 2)
    return a, b


def _usd(x, rng):
    return money(int(x), rng, style=pick(rng, ["sym", "sym", "code"]))


# =========================================================================== schema 1
RECEIVERS = {
    "assembly_line": ("Tamsley Auto Assembly plant", "just-in-time parts for the trim line"),
    "retail_dc": ("Corriston Home Goods distribution centre", "stock for a promotion that starts at the weekend"),
    "construction": ("Marrowfield hospital extension site", "steel beams; a crane and crew are booked for the lift"),
    "restock": ("Pellbridge Supply regional warehouse", "routine replenishment"),
    "hospital": ("Quillhaven General Hospital stores", "surgical consumables below reorder point"),
}
DELAY_CAUSES = {
    "tractor_breakdown": "tractor broke down (air system), roadside repair called",
    "weather_closure": "interstate closed by an ice storm, reopening time not given",
    "driver_hours": "driver out of legal driving hours after a long shipper wait",
    "missed_linehaul": "freight missed the scheduled linehaul out of the carrier terminal",
    "shipper_late_load": "load released 6 hours late at the shipper's dock",
    "no_update": None,
}


class TransitDelayRecovery(Schema):
    name = "transit_delay_recovery"
    description = "A truckload running late against a delivery appointment: how to recover and whether notice is due."
    surfaces = {"json": 0.5, "tms_exception": 0.25, "status_feed": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        rcv = pick(rng, list(RECEIVERS))
        cause = pick(rng, list(DELAY_CAUSES))
        mode = pick_w(rng, {"FTL": 5, "LTL": 3, "intermodal": 1})
        slip_h = round(rng.uniform(0.5, 20), 1)
        eta_confidence = pick_w(rng, {"firm": 3, "estimate": 4, "unknown": 2})
        tolerance_h = pick(rng, [1, 2, 4])
        hours_to_appt = pick(rng, [4, 8, 14, 26, 40])
        stock_h = rng.randint(2, 30) if rcv in ("assembly_line", "hospital") else None
        options = {"power_swap_h": pick(rng, [2, 4, 7, None]), "team_available": chance(rng, 0.4),
                   "expedite_quote_usd": int(round(rng.uniform(1800, 9000), -2)), "expedite_transit_h": rng.randint(5, 16)}
        rescheduled = chance(rng, 0.15)
        last_update_h = pick(rng, [0.5, 2, 5, 11])
        carrier_otp = round(rng.uniform(78, 98), 1)
        if reg == "clear":
            if chance(rng, 0.5):
                slip_h, rcv, eta_confidence = round(rng.uniform(0.3, 0.8), 1), "restock", "firm"
                stock_h, cause, rescheduled = None, pick(rng, ["driver_hours", "missed_linehaul"]), False
                hours_to_appt = pick(rng, [26, 40])
            else:
                rcv, slip_h, cause, eta_confidence = "assembly_line", round(rng.uniform(8, 18), 1), "tractor_breakdown", "firm"
                stock_h, hours_to_appt = rng.randint(2, 5), pick(rng, [4, 8])
        elif reg == "borderline":
            slip_h = near(rng, tolerance_h, 0.8, lo=0.2, digits=1)
            if stock_h is not None:
                stock_h = int(near(rng, hours_to_appt + slip_h, 3, lo=1, integer=True))
        else:
            # a confident carrier ETA against a cause that makes it doubtful, or a 'late' load the receiver has moved
            if chance(rng, 0.5):
                cause, eta_confidence, last_update_h = pick(rng, ["weather_closure", "no_update"]), "firm", pick(rng, [5, 11])
                slip_h = round(rng.uniform(0.5, 1.5), 1)
            else:
                rescheduled, slip_h = True, round(rng.uniform(3, 10), 1)
            if chance(rng, 0.4):
                options["power_swap_h"] = None
        if rcv not in ("assembly_line", "hospital"):
            stock_h = None
        a, b = _lane(rng)
        return {"regime": reg, "receiver": rcv, "cause": cause, "mode": mode, "projected_slip_h": slip_h,
                "eta_confidence": eta_confidence, "late_tolerance_h": tolerance_h, "hours_to_appointment": hours_to_appt,
                "receiver_stock_cover_h": stock_h, "recovery_options": options, "appointment_already_moved": rescheduled,
                "hours_since_carrier_update": last_update_h, "carrier_on_time_pct_90d": carrier_otp,
                "carrier": pick(rng, CARRIERS), "origin": a, "destination": b, "pallets": rng.randint(4, 26),
                "load_value_usd": int(round(rng.uniform(12000, 240000), -3)), "analyst": person(rng),
                "load": ident(rng, "LD", 7, ""), "now": rand_ts(rng)}

    RULE = ("Routing guide 7.3: the control tower sends the receiving site a formal late-delivery notice when a load's "
            "projected arrival misses its appointment by more than {tol} h, unless the appointment has been re-set by "
            "agreement.")

    def _eta_txt(self, f):
        s = f["projected_slip_h"]
        return {"firm": f"carrier ETA {s} h after the appointment", "estimate": f"about {s} h late (carrier estimate)",
                "unknown": "carrier cannot give an ETA yet"}[f["eta_confidence"]]

    def _options(self, f, rng):
        o = f["recovery_options"]
        out = [f"power swap possible in {o['power_swap_h']} h" if o["power_swap_h"] else "no spare tractor nearby",
               "team drivers available" if o["team_available"] else "no team drivers available",
               f"expedited carrier quote {_usd(o['expedite_quote_usd'], rng)}, {o['expedite_transit_h']} h transit from pickup"]
        return out

    def render(self, f, rng, surface):
        rname, need = RECEIVERS[f["receiver"]]
        cause = DELAY_CAUSES[f["cause"]]
        stock = f"receiver reports about {f['receiver_stock_cover_h']} h of stock on hand" if f["receiver_stock_cover_h"] else None
        moved = "receiving dock agreed a new appointment 6 h later" if f["appointment_already_moved"] else None
        rule = self.RULE.format(tol=f["late_tolerance_h"])
        if surface == "json":
            return dumps(prune({
                "load": {"id": f["load"], "mode": f["mode"], "carrier": f["carrier"], "lane": f"{f['origin']} -> {f['destination']}",
                         "pallets": f["pallets"], "value_usd": f["load_value_usd"]},
                "delivery": {"receiver": rname, "purpose": need, "hours_to_appointment": f["hours_to_appointment"],
                             "projection": self._eta_txt(f), "appointment_change": moved, "receiver_stock": stock},
                "exception": {"reported_cause": cause or "no reason given", "hours_since_last_carrier_update": f["hours_since_carrier_update"],
                              "carrier_on_time_90d_pct": f["carrier_on_time_pct_90d"]},
                "recovery_options": self._options(f, rng),
                "routing_guide": rule,
            }))
        if surface == "tms_exception":
            rows = [("Exception", f"LATE-RISK on {f['load']}"), ("Carrier", f"{f['carrier']} ({f['mode']})"),
                    ("Lane", f"{f['origin']} to {f['destination']}"), ("Receiver", rname), ("Freight", need),
                    ("Appointment in", f"{f['hours_to_appointment']} h"), ("Projection", self._eta_txt(f)),
                    ("Cause", show(cause, rng)), ("Last carrier update", f"{f['hours_since_carrier_update']} h ago"),
                    ("Receiver stock", stock), ("Appointment", moved), ("Carrier OTP (90d)", f"{f['carrier_on_time_pct_90d']}%"),
                    ("Load value", _usd(f["load_value_usd"], rng))]
            return (f"{TOWER.upper()} -- EXCEPTION RECORD\n" + kv_lines(rows, rng) + "\n\nRecovery options:\n"
                    + bullets(self._options(f, rng), rng) + "\n\n" + rule)
        ev = [("X3", f"picked up at {f['origin']}"), ("X6", "en route")]
        if cause:
            ev.append(("SD", cause))
        ev.append(("AG", self._eta_txt(f)))
        t = f["now"] - timedelta(hours=f["hours_since_carrier_update"])
        stamps = [t]
        for _ in ev[1:]:
            t -= timedelta(hours=rng.randint(1, 3), minutes=rng.randint(0, 59))
            stamps.append(t)
        lines = [f"{fmt_ts(ts, style='log')}  {code}  {txt}" for ts, (code, txt) in zip(reversed(stamps), ev)]
        lines.append(f"{fmt_ts(f['now'], style='log')}  --  no further carrier message ({f['hours_since_carrier_update']} h)")
        notes = [f"Receiver: {rname} ({need}); appointment in {f['hours_to_appointment']} h."]
        if stock:
            notes.append(stock[0].upper() + stock[1:] + ".")
        if moved:
            notes.append(moved[0].upper() + moved[1:] + ".")
        notes.append("Options: " + "; ".join(self._options(f, rng)) + ".")
        return (f"STATUS FEED {f['load']} ({f['carrier']}, {f['mode']})\n" + "\n".join(lines) + "\n\n"
                + noisy(join_sentences(notes), rng, 0.1) + "\n" + rule)

    def questions(self):
        return {
            "recovery_plan": choice("What should the control tower do about this load?", {
                "carrier_power_swap_or_team": "Have the carrier send a replacement tractor or a two-driver team to "
                                              "win back time.",
                "let_carrier_recover": "Leave recovery to the carrier on its current plan and keep watching.",
                "reschedule_delivery_appointment": "Keep the freight on the truck and book a later slot at the "
                                                   "receiving dock.",
                "split_ship_critical_units": "Pull the most critical units onto an expedited or air service and let "
                                             "the rest follow.",
                "transload_to_expedited_carrier": "Move the whole load to an expedited carrier at the nearest suitable dock.",
            }),
            "downstream_disruption": score("How badly will the late arrival disrupt the receiving site?", [
                "Hardly: the freight is routine stock with slack.",
                "Somewhat: plans shift, but nothing stops.",
                "Seriously: a booked crew, promotion or clinic is affected.",
                "Critically: a production line or patient supply stops without it.",
            ]),
            "recovery_window": score("How much time is left to act before the delay can no longer be recovered?", [
                "Ample: ordinary action can still get it there on time.",
                "Narrowing: a recovery decision is needed within the day.",
                "Short: only an expedited option chosen in the next few hours can still make it.",
                "Gone: nothing can now meet the need; only damage limitation remains.",
            ]),
            "late_notice_due": noul(
                "The routing guide requires a formal late-delivery notice to the receiving site for this load.",
                yes="A firm or estimated carrier ETA misses the appointment by more than the guide's tolerance, and "
                    "the slot has not been re-set by agreement.",
                no="The ETA is within the tolerance, the carrier cannot give an ETA yet, or the appointment has been "
                   "re-set by agreement."),
            "carrier_responsible": noul(
                "The delay is the carrier's responsibility rather than the weather's, the shipper's or the receiver's.",
                yes="The cause lies in the carrier's own operation: its tractor broke down or the freight missed the "
                    "carrier's own linehaul.",
                no="Weather closed the road, the shipper released the load late or held the driver until the driver "
                   "ran out of hours, or no cause has been given."),
        }


# =========================================================================== schema 2
COMMODITIES = {
    "televisions": ("65-inch televisions", 780, 58, "fragile, top-load only"),
    "pumps": ("industrial centrifugal pumps", 4200, 210, "crated"),
    "glass_doors": ("frameless glass shower doors", 390, 70, "A-frame racks"),
    "canned_goods": ("pallets of canned tomatoes", 1150, 1900, "shrink-wrapped pallets"),
    "furniture": ("flat-pack office desks", 310, 95, "cartons on pallets"),
}


class FreightDamageClaim(Schema):
    name = "freight_damage_claim"
    description = "Freight delivered damaged: who to recover the loss from and what to do with the goods."
    surfaces = {"json": 0.5, "osd_report": 0.25, "claims_worksheet": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        com = pick(rng, list(COMMODITIES))
        label, unit_val, unit_lb, pack = COMMODITIES[com]
        units = rng.randint(12, 60) if com != "canned_goods" else rng.randint(8, 20)
        damaged = max(1, int(units * rng.uniform(0.05, 0.5)))
        pod = pick_w(rng, {"specific_exception": 4, "subject_to_inspection": 2, "clean": 2})
        seal = pick_w(rng, {"intact": 5, "broken": 1, "mismatch": 1, "none_ltl": 2})
        loaded_by = pick_w(rng, {"shipper_load_count": 3, "carrier_handled": 3})
        packaging_ok = pick_w(rng, {"meets_tariff": 5, "questionable": 2, "deficient": 1})
        concealed_days = 0 if pod != "clean" else pick(rng, [0, 2, 4, 7, 12])
        photos = chance(rng, 0.7)
        released_value_per_lb = pick(rng, [None, 0.5, 2.0, 10.0])
        deductible = pick(rng, [2500, 5000, 10000])
        days_since = pick(rng, [2, 10, 45, 150, 250])
        cause = pick_w(rng, {"forklift": 2, "load_shift": 3, "water": 1, "crush": 3})
        if reg == "clear":
            if chance(rng, 0.5):
                pod, seal, packaging_ok, photos, concealed_days = "specific_exception", "intact", "meets_tariff", True, 0
                loaded_by, days_since = "carrier_handled", pick(rng, [2, 10])
            else:
                damaged, pod, packaging_ok = 1, "clean", "deficient"
                concealed_days = pick(rng, [12, 20])
        elif reg == "borderline":
            concealed_days = pick(rng, [4, 5, 6]) if pod == "clean" else 0
            if com != "canned_goods":
                damaged = max(1, int(round(deductible / unit_val * rng.uniform(0.8, 1.3))))
                damaged = min(damaged, units)
        else:
            # a clean POD with photos of obvious damage, or an exception noted on a shipper-loaded, poorly packed load
            if chance(rng, 0.5):
                pod, photos, concealed_days = "clean", True, pick(rng, [1, 3])
                cause = "crush"
            else:
                pod, loaded_by, packaging_ok, seal = "specific_exception", "shipper_load_count", "questionable", "intact"
            if chance(rng, 0.4):
                released_value_per_lb = pick(rng, [0.5, 2.0])
        if pod != "clean":
            concealed_days = 0
        days_since = max(days_since, concealed_days + rng.randint(0, 3))   # a late report cannot post-date today
        loss = damaged * unit_val
        weight_lb = damaged * unit_lb
        return {"regime": reg, "commodity": com, "units_shipped": units, "units_damaged": damaged,
                "loss_usd": loss, "damaged_weight_lb": weight_lb, "pod_notation": pod, "seal": seal,
                "loaded_by": loaded_by, "packaging": packaging_ok, "concealed_report_days": concealed_days,
                "photos": photos, "released_value_per_lb": released_value_per_lb, "policy_deductible_usd": deductible,
                "days_since_delivery": days_since, "damage_cause": cause, "carrier": pick(rng, CARRIERS),
                "lane": _lane(rng), "salvage_offer_pct": pick(rng, [None, 10, 20, 35]),
                "pro": ident(rng, "PRO", 8, ""), "clerk": person(rng), "delivered": rand_ts(rng).date()}

    def _pod(self, f):
        return {"specific_exception": f"POD signed with exception: '{f['units_damaged']} {self._unit(f)} damaged'",
                "subject_to_inspection": "POD signed 'subject to inspection'", "clean": "POD signed clean, no exceptions"}[f["pod_notation"]]

    def _unit(self, f):
        return "pallets" if f["commodity"] == "canned_goods" else "units"

    def _seal(self, f):
        return {"intact": "seal intact, number matches the bill of lading", "broken": "seal broken on arrival",
                "mismatch": "seal number does not match the bill of lading", "none_ltl": "LTL, no trailer seal"}[f["seal"]]

    def _cause(self, f):
        return {"forklift": "forklift tine punctures", "load_shift": "load shifted, pallets toppled toward the doors",
                "water": "water staining from a roof leak in the trailer", "crush": "crushed cartons on the bottom tier"}[f["damage_cause"]]

    def _liab(self, f, rng):
        r = f["released_value_per_lb"]
        return (f"bill of lading shows released value {_usd_c(r)}/lb" if r else "no released-value limit on the bill of lading")

    def render(self, f, rng, surface):
        label, unit_val, unit_lb, pack = COMMODITIES[f["commodity"]]
        a, b = f["lane"]
        conc = (f"damage reported by the receiver {f['concealed_report_days']} day(s) after delivery"
                if f["pod_notation"] == "clean" and f["concealed_report_days"] else None)
        packaging = {"meets_tariff": f"packaging: {pack}, meets the carrier tariff",
                     "questionable": f"packaging: {pack}, but non-stackable pallets were double-stacked",
                     "deficient": f"packaging: loose {label} without corner protection"}[f["packaging"]]
        loaded = {"shipper_load_count": "shipper load and count", "carrier_handled": "carrier-handled (cross-docked twice)"}[f["loaded_by"]]
        salvage = f"salvage buyer offers {f['salvage_offer_pct']}% of value" if f["salvage_offer_pct"] else None
        if surface == "json":
            return dumps(prune({
                "shipment": {"pro": f["pro"], "carrier": f["carrier"], "lane": f"{a} -> {b}", "commodity": label,
                             "units": f["units_shipped"], "loading": loaded, "delivered": f["delivered"].isoformat(),
                             "days_since_delivery": f["days_since_delivery"]},
                "damage": {"units_damaged": f["units_damaged"], "description": self._cause(f),
                           "value_usd": f["loss_usd"], "weight_lb": f["damaged_weight_lb"], "photos": f["photos"],
                           "late_report": conc},
                "documents": {"pod": self._pod(f), "seal": self._seal(f), "packaging": packaging,
                              "carrier_liability": self._liab(f, rng)},
                "insurance": {"cargo_policy": f"all-risk, deductible {f['policy_deductible_usd']} USD per occurrence, "
                                              "excludes insufficient packing and inherent vice"},
                "salvage": salvage,
            }))
        if surface == "osd_report":
            rows = [("PRO", f["pro"]), ("Carrier", f["carrier"]), ("Delivered", fmt_date(f["delivered"], rng)),
                    ("Commodity", f"{label} x {f['units_shipped']}"), ("Loading", loaded),
                    ("Over/Short/Damaged", f"D: {f['units_damaged']} {self._unit(f)}, {self._cause(f)}"),
                    ("POD", self._pod(f)), ("Seal", self._seal(f)), ("Photos", "yes" if f["photos"] else "none taken"),
                    ("Reported", conc or "at delivery"), ("Packaging", packaging.split(": ", 1)[1]),
                    ("Damaged value / weight", f"{_usd(f['loss_usd'], rng)} / {f['damaged_weight_lb']:,} lb"),
                    ("Carrier liability", self._liab(f, rng)),
                    ("Cargo policy", f"all-risk, deductible {_usd(f['policy_deductible_usd'], rng)}, excludes "
                                     "insufficient packing")]
            return "OVER / SHORT / DAMAGE REPORT\n" + kv_lines(rows, rng)
        lines = [f"CLAIMS DESK WORKSHEET -- {f['pro']} ({f['clerk']})",
                 table(["item", "value"], [["damaged value", _usd(f["loss_usd"], rng)],
                                           ["damaged weight", f"{f['damaged_weight_lb']:,} lb"],
                                           ["carrier limit", self._liab(f, rng)],
                                           ["policy deductible", _usd(f["policy_deductible_usd"], rng)],
                                           ["days since delivery", f["days_since_delivery"]],
                                           ["salvage", salvage or "no offer yet"]], rng),
                 "", noisy(join_sentences([f"{self._pod(f)}; {self._seal(f)}.", f"{self._cause(f).capitalize()}.",
                                            (conc[0].upper() + conc[1:] + ".") if conc else "",
                                            packaging[0].upper() + packaging[1:] + ".",
                                            f"Load was {loaded}.", "Photos on file." if f["photos"] else "No photos."]), rng),
                 "", "Carrier tariff: concealed damage must be reported within 5 days of delivery; claims filed within "
                     "9 months. Cargo policy: all-risk, excludes insufficient packing and inherent vice; notify "
                     "insurer within 30 days of discovery."]
        return "\n".join(lines)

    def questions(self):
        return {
            "recovery_route": choice("Where should the claims desk seek recovery for this loss, given that a carrier "
                                     "claim is viable only when damage was noted on the POD or reported within 5 "
                                     "days, packaging was not deficient, it is filed within 9 months, and it is not "
                                     "load-shift or crush damage on a shipper-load-and-count trailer with the seal "
                                     "intact?", {
                "absorb_loss": "Absorb the loss: no viable carrier claim, and the insurance would not pay.",
                "carrier_claim_and_insurer_notice": "Claim against the carrier and notify the insurer: the carrier "
                                                    "claim is viable but its released-value limit falls short of the "
                                                    "loss, and the insurance would pay the rest.",
                "file_cargo_insurance_claim": "Claim on the cargo insurance only: no viable carrier claim, but the "
                                              "insurance would pay.",
                "file_carrier_claim": "Claim against the carrier only: the carrier claim is viable and its limit "
                                      "covers the loss, or the insurance would not pay the shortfall.",
            }),
            "goods_disposition": choice("What should happen to the damaged goods?", {
                "accept_and_rework": "Accept them and repair, repack or rework them for sale or use.",
                "destroy_with_certificate": "Destroy them and keep a certificate of destruction for the claim.",
                "return_for_inspection": "Return them to the origin for inspection before deciding.",
                "sell_to_salvage": "Sell them to a salvage buyer and credit the proceeds against the claim.",
            }),
            "carrier_liability_evidence": score("How strongly do the documents support the carrier's liability?", [
                "Weakly: clean receipt, late report, or damage pointing at packing.",
                "Partly: some support, but the carrier has a ready defence.",
                "Well: an exception at delivery and sound packing.",
                "Conclusively: specific exceptions, photos, intact packaging, carrier-handled freight.",
            ]),
            "claim_deadline_risk": score("How much risk is there that a notice or filing deadline defeats recovery?", [
                "None: every deadline is well ahead.",
                "Low: a deadline is approaching but easily met.",
                "High: a notice window is about to close or was only just met.",
                "Fatal: a required notice or filing window has already passed.",
            ]),
            "insurance_would_pay": noul(
                "The cargo insurance policy would pay for this loss.",
                yes="Packaging was not deficient (so the insufficient-packing exclusion does not apply) and the "
                    "damaged value exceeds the deductible.",
                no="Packaging was deficient (loose goods without protection), or the damaged value is at or under "
                   "the deductible."),
        }


def _usd_c(x):
    return f"${x:.2f}"


# =========================================================================== schema 3
PRODUCTS = {
    "berries": ("fresh blueberries", 1.0, (0.0, 4.0), "quality: soft fruit and mould risk rises quickly above 5 C"),
    "seafood": ("frozen shrimp", -18.0, (-22.0, -15.0), "quality: refreeze damage if above -12 C for several hours"),
    "vaccines": ("vaccine consignment (2-8 C)", 5.0, (2.0, 8.0),
                 "stability data allow up to 72 h cumulative at 8-25 C; any freezing is a rejection"),
    "dairy": ("yoghurt and fresh cheese", 3.0, (1.0, 5.0), "food safety: above 7 C for more than 4 h is a rejection"),
    "flowers": ("cut roses", 2.0, (1.0, 4.0), "vase life drops sharply after 12 h above 10 C"),
}


class ReeferTemperatureExcursion(Schema):
    name = "reefer_temperature_excursion"
    description = "A refrigerated load whose temperature record shows an excursion: disposition, exposure and cover."
    surfaces = {"json": 0.5, "reefer_download": 0.25, "qa_receiving_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        prod = pick(rng, list(PRODUCTS))
        label, setp, (lo, hi), rule = PRODUCTS[prod]
        peak_over = round(rng.uniform(0.5, 9), 1)
        hours_out = round(rng.uniform(0.5, 30), 1)
        probe = pick_w(rng, {"return_air": 4, "supply_air": 2, "product_logger": 3})
        logger_agrees = pick_w(rng, {"agrees": 4, "milder": 3, "none_in_load": 2})
        cause = pick_w(rng, {"unit_failure": 3, "door_openings": 2, "setpoint_error": 2, "warm_loading": 2, "fuel_out": 1})
        setpoint_bol = setp
        setpoint_unit = setp if cause != "setpoint_error" else setp + pick(rng, [5, 8, 20])
        breakdown_h = round(rng.uniform(1, 30), 1) if cause in ("unit_failure", "fuel_out") else 0
        pulp_at_loading = round(setp + (rng.uniform(4, 9) if cause == "warm_loading" else rng.uniform(-0.5, 1.5)), 1)
        pulp_at_arrival = round(hi + rng.uniform(-1.5, 4), 1)
        policy_breakdown_h = pick(rng, [12, 24])
        value = int(round(rng.uniform(18000, 420000), -3))
        if reg == "clear":
            if chance(rng, 0.5):
                peak_over, hours_out, logger_agrees, cause = round(rng.uniform(0.3, 1.0), 1), round(rng.uniform(0.3, 1.2), 1), "milder", "door_openings"
                breakdown_h, pulp_at_arrival = 0, round(hi - rng.uniform(0.2, 1.5), 1)
                setpoint_unit = setp
            else:
                cause, breakdown_h = "unit_failure", round(rng.uniform(policy_breakdown_h + 4, policy_breakdown_h + 20), 1)
                peak_over, hours_out, logger_agrees = round(rng.uniform(8, 14), 1), breakdown_h, "agrees"
                pulp_at_arrival = round(hi + rng.uniform(5, 9), 1)
        elif reg == "borderline":
            if cause in ("unit_failure", "fuel_out"):
                breakdown_h = near(rng, policy_breakdown_h, 2, lo=1, digits=1)
                hours_out = breakdown_h
            pulp_at_arrival = near(rng, hi, 1.0, digits=1)
        else:
            # the unit's return-air trace says disaster while the product logger is mild, or a breakdown claim on a
            # load that was loaded warm
            if chance(rng, 0.5):
                probe, logger_agrees, peak_over = "return_air", "milder", round(rng.uniform(6, 12), 1)
            else:
                cause, pulp_at_loading = "warm_loading", round(setp + rng.uniform(5, 9), 1)
                breakdown_h = round(rng.uniform(2, 8), 1)
        # derived values follow the FINAL cause, so a case never mixes one cause's signals with another's
        frozen = setp < 0
        if cause == "setpoint_error":
            setpoint_unit = setp + (pick(rng, [8, 16]) if frozen else pick(rng, [5, 8]))
        else:
            setpoint_unit = setp
        if cause not in ("unit_failure", "fuel_out"):
            breakdown_h = 0
        if cause == "warm_loading":
            pulp_at_loading = round(max(pulp_at_loading, setp + rng.uniform(4, 9)), 1)
        else:
            pulp_at_loading = round(setp + rng.uniform(-0.5, 1.5), 1)
        if reg != "clear":
            hours_out = {"door_openings": round(rng.uniform(0.5, 4), 1), "setpoint_error": round(rng.uniform(10, 40), 1),
                         "warm_loading": round(rng.uniform(3, 12), 1),
                         "unit_failure": round(breakdown_h + rng.uniform(0.5, 3), 1),
                         "fuel_out": round(breakdown_h + rng.uniform(0.5, 3), 1)}[cause]
        if cause == "door_openings":
            peak_over = min(peak_over, round(rng.uniform(0.5, 3.5), 1))
        if frozen:
            pulp_at_arrival = round(lo + (pulp_at_arrival - lo) * 0.9, 1)
        value = int(round(rng.uniform(*{"berries": (40000, 120000), "seafood": (80000, 300000),
                                        "vaccines": (250000, 1800000), "dairy": (30000, 90000),
                                        "flowers": (40000, 150000)}[prod]), -3))
        return {"regime": reg, "product": prod, "setpoint_bol": setpoint_bol, "setpoint_unit": setpoint_unit,
                "range": [lo, hi], "peak_above_range_c": peak_over, "hours_out_of_range": hours_out,
                "probe": probe, "product_logger": logger_agrees, "cause_signal": cause,
                "unit_breakdown_h": breakdown_h, "pulp_at_loading_c": pulp_at_loading,
                "pulp_at_arrival_c": pulp_at_arrival, "policy_breakdown_hours": policy_breakdown_h,
                "load_value_usd": value, "carrier": pick(rng, CARRIERS), "lane": _lane(rng),
                "trailer": ident(rng, "RF", 5, ""), "qa": person(rng), "arrived": rand_ts(rng)}

    def _alarms(self, f):
        return {"unit_failure": [f"ALARM 63 high head pressure, unit shut down; restarted after {f['unit_breakdown_h']} h"],
                "fuel_out": [f"ALARM 00 low fuel; unit stopped; refuelled {f['unit_breakdown_h']} h later"],
                "door_openings": ["DOOR OPEN x6 (multi-stop delivery), longest 38 min"],
                "setpoint_error": [f"setpoint changed to {f['setpoint_unit']} C at pickup (bill of lading says {f['setpoint_bol']} C)"],
                "warm_loading": ["no alarms; unit running continuous at full capacity for the first 10 h"]}[f["cause_signal"]]

    def _logger(self, f):
        return {"agrees": "independent logger in the load shows the same pattern",
                "milder": "independent logger in the load peaked well below the unit's reading",
                "none_in_load": "no independent logger was packed with the load"}[f["product_logger"]]

    def render(self, f, rng, surface):
        label, setp, (lo, hi), rule = PRODUCTS[f["product"]]
        a, b = f["lane"]
        peak = round(hi + f["peak_above_range_c"], 1)
        probe = {"return_air": "return-air sensor", "supply_air": "supply-air sensor", "product_logger": "product probe"}[f["probe"]]
        cover = (f"cargo policy: loss from breakdown of refrigerating machinery covered if the breakdown lasts at least "
                 f"{f['policy_breakdown_hours']} consecutive hours; excludes improper pre-cooling and incorrect instructions")
        if surface == "json":
            return dumps(prune({
                "load": {"trailer": f["trailer"], "carrier": f["carrier"], "lane": f"{a} -> {b}", "product": label,
                         "value_usd": f["load_value_usd"], "setpoint_on_bill_of_lading_c": f["setpoint_bol"]},
                "temperature_record": {"source": probe, "range_c": [lo, hi], "peak_c": peak,
                                       "hours_out_of_range": f["hours_out_of_range"], "unit_events": self._alarms(f),
                                       "independent_logger": self._logger(f)},
                "pulp_temperatures_c": {"at_loading": f["pulp_at_loading_c"], "at_arrival": f["pulp_at_arrival_c"]},
                "product_rule": rule, "insurance": cover,
            }))
        if surface == "reefer_download":
            t0 = f["arrived"] - timedelta(hours=max(12, int(f["hours_out_of_range"]) + 10))
            n = 6
            rows = []
            for i in range(n):
                t = t0 + timedelta(hours=i * max(2, int(f["hours_out_of_range"] + 10) // n))
                frac = [0.0, 0.2, 1.0, 0.8, 0.3, 0.1][i]
                rows.append([t.strftime("%m-%d %H:%M"), f"{f['setpoint_unit']:.1f}",
                             f"{(hi - 0.5) + frac * (peak - hi + 0.5):.1f}"])
            return "\n".join([f"REEFER UNIT DOWNLOAD -- trailer {f['trailer']} ({f['carrier']}), {label}",
                              f"Sensor: {probe}; range {lo} to {hi} C", "",
                              table(["time", "setpoint C", "reading C"], rows, rng, "fixed"), "",
                              "Events:", bullets(self._alarms(f), rng), "",
                              kv_lines([("Out of range", f"{f['hours_out_of_range']} h, peak {peak} C"),
                                        ("Logger", self._logger(f)), ("Pulp at loading", f"{f['pulp_at_loading_c']} C"),
                                        ("Pulp at arrival", f"{f['pulp_at_arrival_c']} C"),
                                        ("BOL setpoint", f"{f['setpoint_bol']} C")], rng),
                              "", rule[0].upper() + rule[1:] + ".", cover[0].upper() + cover[1:] + "."])
        s = [f"Trailer {f['trailer']} arrived {fmt_ts(f['arrived'], rng)} with {label} from {a}.",
             f"Unit download ({probe}) shows {f['hours_out_of_range']} h outside {lo} to {hi} C, peaking at {peak} C.",
             self._alarms(f)[0][0].upper() + self._alarms(f)[0][1:] + ".", self._logger(f).capitalize() + ".",
             f"Our pulp readings at the dock: {f['pulp_at_arrival_c']} C. Shipper's loading record says {f['pulp_at_loading_c']} C.",
             f"Load value {_usd(f['load_value_usd'], rng)}."]
        return (f"QA RECEIVING NOTE ({f['qa']})\n" + noisy(join_sentences(s), rng) + f"\n\nProduct rule: {rule}.\n"
                f"Policy: {cover}.\nLoad on hold at door 14 pending decision.")

    def questions(self):
        return {
            "load_disposition": choice("What should the receiving site do with this load?", {
                "divert_to_secondary_outlet": "Accept it for a lower-grade use, such as processing, at a reduced value.",
                "receive_under_qa_hold": "Receive it into a quarantine area while quality assurance assesses it.",
                "reject_load": "Refuse the load and document the refusal for the claim.",
                "release_for_normal_receipt": "Receive it into stock as normal.",
            }),
            "product_exposure": score("How far did the product itself go beyond its allowed conditions?", [
                "Not at all: the product stayed within its limits.",
                "Briefly: a short excursion well within the product's tolerance.",
                "Substantially: close to or at the product's tolerance.",
                "Beyond tolerance: the product's own limits were clearly exceeded.",
            ]),
            "record_reliability": score("How reliable is the temperature record as evidence of what the product "
                                        "experienced?", [
                "Unreliable: sensors disagree and none measured the product.",
                "Doubtful: only air readings, or loggers that disagree.",
                "Fair: air readings backed by pulp temperatures.",
                "Strong: a product logger and the unit agree.",
            ]),
            "breakdown_cover_applies": noul(
                "The loss falls under the cargo policy's refrigeration-breakdown cover.",
                yes="The refrigeration machinery broke down for at least the policy's stated period and no exclusion "
                    "such as warm loading or wrong instructions applies.",
                no="There was no qualifying breakdown, it was too short, or an exclusion applies."),
            "carrier_at_fault": noul(
                "The excursion was caused by the carrier's handling of the unit rather than by how the load was "
                "prepared or instructed."),
        }


# =========================================================================== schema 4
TRACE_COMMODITIES = {
    "laptops": ("boxed laptops", 1150, True), "tires": ("passenger tyres", 140, True),
    "brackets": ("steel mounting brackets", 18, False), "supplements": ("nutritional supplements", 42, True),
    "valves": ("brass valves", 95, False), "spirits": ("cases of spirits", 260, True),
}


class NoScanTrace(Schema):
    name = "no_scan_trace"
    description = "Freight that has stopped scanning in a carrier network: keep tracing, reship, recover or claim."
    surfaces = {"json": 0.5, "trace_log": 0.25, "scan_history": 0.25}

    TRACE = {"searching": "carrier terminal is searching the dock and trailers",
             "osd_match": "carrier OS&D reports unlabelled freight matching the description at {t}",
             "delivered_no_pod": "carrier shows 'delivered' but has no signed proof of delivery",
             "misrouted": "carrier says the pallets were loaded to the wrong linehaul and went to {t}",
             "no_response": "no reply to two trace requests"}

    def sample_factors(self, rng):
        reg = regime(rng)
        com = pick(rng, list(TRACE_COMMODITIES))
        label, unit_val, attractive = TRACE_COMMODITIES[com]
        units = rng.randint(20, 400)
        value = units * unit_val
        hours_silent = pick(rng, [30, 52, 76, 110, 160])
        days_past_due = pick(rng, [0, 1, 2, 4, 7])
        trace = pick_w(rng, {"searching": 3, "osd_match": 2, "delivered_no_pod": 2, "misrouted": 2, "no_response": 2})
        pieces_total = rng.randint(2, 8)
        pieces_missing = rng.randint(1, pieces_total)
        need_days = pick(rng, [1, 3, 6, 14])
        replacement = pick_w(rng, {"in_stock_origin": 4, "partial_stock": 2, "none_4_weeks": 2})
        threshold = pick(rng, [50000, 100000])
        mode = pick_w(rng, {"LTL": 5, "partial_truckload": 2})
        if reg == "clear":
            if chance(rng, 0.5):
                trace, hours_silent, days_past_due = "osd_match", 30, 0
                com = pick(rng, ["brackets", "valves"])
                label, unit_val, attractive = TRACE_COMMODITIES[com]
                value = units * unit_val
            else:
                trace, hours_silent, days_past_due = "no_response", 160, 7
                com = pick(rng, ["laptops", "spirits"])
                label, unit_val, attractive = TRACE_COMMODITIES[com]
                units = rng.randint(100, 300)
                value = units * unit_val
        elif reg == "borderline":
            value = int(near(rng, threshold, threshold * 0.08, lo=1000))
            units = max(1, value // unit_val)
            value = units * unit_val
            hours_silent = pick(rng, [52, 76])
        else:
            # a 'delivered' scan with no POD on attractive freight, or a promising OS&D lead gone quiet for days
            if chance(rng, 0.5):
                trace, com = "delivered_no_pod", pick(rng, ["laptops", "tires", "spirits"])
            else:
                trace, hours_silent = "osd_match", pick(rng, [110, 160])
            label, unit_val, attractive = TRACE_COMMODITIES[com]
            value = units * unit_val
        a, b = _lane(rng)
        other_town = pick(rng, [t for t in TOWNS if t not in (a, b)])
        d = {"regime": reg, "commodity": com, "units": units, "declared_value_usd": value,
                "theft_attractive": attractive, "hours_since_last_scan": hours_silent, "days_past_due": days_past_due,
                "trace_status": trace, "pieces_total": pieces_total, "pieces_missing": pieces_missing,
                "consignee_need_days": need_days, "replacement_stock": replacement, "high_value_threshold_usd": threshold,
                "mode": mode, "carrier": pick(rng, CARRIERS), "origin": a, "destination": b, "other_town": other_town,
                "last_scan_place": pick(rng, [f"{a} terminal", "linehaul hub", f"{b} terminal"]),
                "analyst": person(rng), "bol": ident(rng, "BOL", 7, ""), "pro": ident(rng, "PRO", 8, ""),
                "shipped": rand_ts(rng)}
        d["replacement_transit_days"] = pick(rng, [1, 2, 3])      # drawn last: every other factor is unchanged
        return d

    def _trace(self, f):
        return self.TRACE[f["trace_status"]].format(t=f"{f['other_town']} terminal")

    def _stock(self, f):
        t = f"a replacement takes {f['replacement_transit_days']} day(s) in transit"
        return {"in_stock_origin": f"replacement stock available at origin, can ship today; {t}",
                "partial_stock": f"only part of the quantity is in stock for a replacement; {t}",
                "none_4_weeks": "no replacement stock for about 4 weeks"}[f["replacement_stock"]]

    def _protocol(self, f):
        return (f"Security protocol: shipments declared at {_usd_c0(f['high_value_threshold_usd'])} or more, or any "
                f"quantity of electronics, spirits or pharmaceuticals, move under the high-value protocol (sealed "
                f"trailer, no terminal dwell, 24 h escalation).")

    def render(self, f, rng, surface):
        label = TRACE_COMMODITIES[f["commodity"]][0]
        pieces = f"{f['pieces_missing']} of {f['pieces_total']} pallets unaccounted for"
        if surface == "json":
            return dumps(prune({
                "shipment": {"bol": f["bol"], "pro": f["pro"], "carrier": f"{f['carrier']} ({f['mode']})",
                             "lane": f"{f['origin']} -> {f['destination']}", "commodity": f"{label} x {f['units']}",
                             "declared_value_usd": f["declared_value_usd"], "status": pieces},
                "tracking": {"last_scan": f["last_scan_place"], "hours_since_last_scan": f["hours_since_last_scan"],
                             "days_past_due_date": f["days_past_due"], "carrier_trace": self._trace(f)},
                "need": {"consignee_needs_it_within_days": f["consignee_need_days"], "replacement": self._stock(f)},
                "policy": self._protocol(f) if chance(rng, 0.7) else None,
            }))
        if surface == "trace_log":
            t = f["shipped"]
            ev = [(0, f"shipped {f['bol']} ({label}, {f['pieces_total']} pallets) via {f['carrier']}"),
                  (rng.randint(20, 40), f"last scan at {f['last_scan_place']}"),
                  (f["hours_since_last_scan"] // 2, "trace request #1 sent"),
                  (rng.randint(8, 20), self._trace(f))]
            lines, cur = [], t
            for dh, txt in ev:
                cur += timedelta(hours=dh)
                lines.append(f"{fmt_ts(cur, rng, 'log')}  {txt}")
            tail = noisy(join_sentences([f"Status: {pieces}.",
                                         f"Now {f['hours_since_last_scan']} h without a scan, {f['days_past_due']} day(s) past due.",
                                         f"Declared value {_usd(f['declared_value_usd'], rng)}.",
                                         f"Receiving site needs it within {f['consignee_need_days']} day(s); {self._stock(f)}."]), rng, 0.1)
            return f"TRACE LOG -- {f['pro']} ({f['analyst']})\n" + "\n".join(lines) + "\n\n" + tail + "\n\n" + self._protocol(f)
        t = f["shipped"]
        scans = [[fmt_ts(t, style="log"), f["origin"], "picked up"],
                 [fmt_ts(t + timedelta(hours=6), style="log"), f"{f['origin']} terminal", "arrived at terminal"],
                 [fmt_ts(t + timedelta(hours=rng.randint(10, 30)), style="log"), f["last_scan_place"], "departed" if "hub" in f["last_scan_place"] else "loaded to linehaul"]]
        if f["trace_status"] == "delivered_no_pod":
            scans.append([fmt_ts(t + timedelta(hours=rng.randint(40, 60)), style="log"), f["destination"], "delivered (no signature captured)"])
        return "\n".join([f"SCAN HISTORY {f['pro']} -- {f['carrier']}", table(["time", "location", "event"], scans, rng), "",
                          kv_lines([("Freight", f"{label} x {f['units']}, declared {_usd(f['declared_value_usd'], rng)}"),
                                    ("Status", pieces), ("Hours since last scan", f["hours_since_last_scan"]),
                                    ("Past due", f"{f['days_past_due']} day(s)"), ("Carrier trace", self._trace(f)),
                                    ("Need", f"within {f['consignee_need_days']} day(s)"), ("Replacement", self._stock(f))], rng)])

    def questions(self):
        return {
            "trace_decision": choice("What should the control tower do next about the missing freight?", {
                "arrange_recovery_from_location": "Arrange pickup or re-delivery from where the carrier says the "
                                                  "freight has turned up.",
                "continue_trace": "Keep tracing with the carrier and check again at the next escalation point.",
                "declare_lost_and_claim": "Declare the freight lost and file a claim with the carrier.",
                "reship_replacement_now": "Send replacement stock now while the trace continues.",
            }),
            "found_intact_likelihood": score("How likely is the freight to be found intact?", [
                "Very unlikely: the carrier shows it delivered with no proof of delivery, or trace requests have gone "
                "unanswered for 96 hours or more since the last scan.",
                "Unlikely: trace requests unanswered for under 96 hours, or the terminal is still searching 96 hours "
                "or more after the last scan.",
                "Fairly likely: the terminal is searching under 96 hours after the last scan, or the carrier has a "
                "lead (an OS&D match or a misroute) but more than 76 hours have passed without a scan.",
                "Very likely: the carrier has a lead (an OS&D match or a misroute) and no more than 76 hours have "
                "passed without a scan.",
            ]),
            "replacement_urgency": score("How urgently must replacement stock move for the receiving site to be "
                                         "supplied in time?", [
                "Not urgent: the site's need date leaves 5 or more days beyond the replacement's transit time.",
                "Moderate: it leaves 2 to 4 days beyond the transit time.",
                "Pressing: it leaves a day or less, so a replacement has to leave now.",
                "Too late: there is no replacement stock for weeks, or the transit time is longer than the time "
                "left before the need date.",
            ]),
            "high_value_protocol_applies": noul(
                "This shipment falls under the high-value handling protocol.",
                yes="The declared value is at or above the protocol threshold, or the commodity is on its list.",
                no="The declared value is below the threshold and the commodity is not on the list."),
            "freight_still_in_network": noul(
                "The evidence suggests the freight is still somewhere in the carrier's network.",
                yes="The carrier has an OS&D match or says it was misrouted to another terminal, or the terminal is "
                    "still searching less than 96 hours after the last scan.",
                no="The carrier shows it delivered without proof of delivery, has not answered trace requests, or "
                   "has searched 96 hours or more without a lead."),
        }


def _usd_c0(x):
    return f"${x:,.0f}"


# =========================================================================== schema 5
DG = {
    "paint": ("UN1263", "Paint", "3", "II", "20 L steel pails"),
    "hcl": ("UN1789", "Hydrochloric acid", "8", "II", "4 x 4 L jerricans in fibreboard boxes"),
    "li_ion": ("UN3480", "Lithium ion batteries", "9", None, "fibreboard boxes of loose cells"),
    "flammable_nos": ("UN1993", "Flammable liquid, n.o.s. (contains toluene)", "3", "III", "200 L steel drums"),
    "sulfuric": ("UN1830", "Sulfuric acid", "8", "II", "1,000 L intermediate bulk container"),
}
DG_EXCEPTIONS = {
    "leak": "package found leaking on the dock",
    "papers_mismatch": "UN number on the shipping papers differs from the package marking",
    "missing_labels": "hazard labels missing on some packages",
    "segregation": "loaded next to incompatible freight on the inbound trailer",
    "undeclared": "found during a dock inspection in packages marked as general cargo",
    "damaged_no_leak": "outer packaging crushed, inner packaging appears intact",
}


class DangerousGoodsException(Schema):
    name = "dangerous_goods_exception"
    description = "A dangerous-goods problem found at a carrier terminal: handling, reporting and hazard."
    surfaces = {"json": 0.5, "terminal_report": 0.25, "dock_log": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        mat = pick(rng, list(DG))
        exc = pick(rng, list(DG_EXCEPTIONS))
        leaked_l = round(rng.uniform(0.1, 15), 1) if exc == "leak" else 0.0
        exposed = pick_w(rng, {"none": 6, "symptoms_first_aid": 2, "hospital": 1}) if exc == "leak" else "none"
        evacuated = exc == "leak" and chance(rng, 0.25)
        pkgs_affected, pkgs_total = rng.randint(1, 4), rng.randint(4, 40)
        shipper_history = pick_w(rng, {"clean": 5, "prior_errors": 3, "new_shipper": 2})
        next_mode = pick_w(rng, {"road": 6, "air": 2, "rail": 1})
        hours_to_departure = pick(rng, [1, 3, 6, 18])
        cleanup = pick_w(rng, {"contained": 4, "ongoing": 3, "not_started": 1}) if exc == "leak" else None
        if reg == "clear":
            if chance(rng, 0.5):
                exc, leaked_l, exposed, evacuated, cleanup = "missing_labels", 0.0, "none", False, None
                shipper_history, pkgs_affected = "clean", 1
            else:
                exc, leaked_l, exposed, evacuated, cleanup = "leak", round(rng.uniform(8, 25), 1), "hospital", True, "ongoing"
                mat = pick(rng, ["hcl", "sulfuric", "flammable_nos"])
        elif reg == "borderline":
            if exc == "leak":
                leaked_l, exposed = round(rng.uniform(0.05, 0.6), 2), pick(rng, ["none", "symptoms_first_aid"])
            else:
                exc = pick(rng, ["papers_mismatch", "damaged_no_leak", "segregation"])
        else:
            # trace wetting with a worker sent to hospital, or undeclared batteries headed for an air leg
            if chance(rng, 0.5):
                exc, leaked_l, exposed, cleanup = "leak", round(rng.uniform(0.05, 0.3), 2), "hospital", "contained"
                evacuated = False
            else:
                exc, mat, next_mode = "undeclared", "li_ion", "air"
                leaked_l, exposed, evacuated, cleanup = 0.0, "none", False, None
        return {"regime": reg, "material": mat, "exception": exc, "leaked_litres": leaked_l, "people_exposed": exposed,
                "evacuation": evacuated, "packages_affected": pkgs_affected, "packages_total": pkgs_total,
                "shipper_history": shipper_history, "next_leg_mode": next_mode, "hours_to_departure": hours_to_departure,
                "cleanup": cleanup, "terminal": pick(rng, TOWNS) + " terminal", "carrier": pick(rng, CARRIERS),
                "supervisor": person(rng), "handler": person(rng), "shipper": pick(rng, ["Pellbridge Coatings",
                                                                                         "Quillhaven Chemicals",
                                                                                         "Dunhollow Battery Recycling",
                                                                                         "Marrowfield Industrial Supply"]),
                "pro": ident(rng, "PRO", 8, ""), "found": rand_ts(rng)}

    def _ident(self, f):
        un, psn, cls, pg, pack = DG[f["material"]]
        return f"{un} {psn}, class {cls}" + (f", PG {pg}" if pg else "") + f"; {pack}"

    PARTNER = {"3": "drums of pool oxidiser (class 5.1)", "8": "sodium cyanide briquettes (class 6.1)",
               "9": "consumer fireworks (class 1.4G)"}

    def _what(self, f):
        exc = DG_EXCEPTIONS[f["exception"]]
        if f["exception"] == "segregation":
            exc = exc.replace("incompatible freight", self.PARTNER[DG[f["material"]][2]])
        return exc

    def _leg(self, f):
        return {"air": "air, via the forwarder's airport gateway", "road": "road linehaul", "rail": "rail, via the intermodal ramp"}[f["next_leg_mode"]]

    def _exposure(self, f):
        return {"none": None, "symptoms_first_aid": "one handler reported eye and throat irritation, treated with first aid",
                "hospital": "one handler taken to hospital for assessment after breathing fumes"}[f["people_exposed"]]

    def _history(self, f):
        return {"clean": None, "prior_errors": "two declaration errors from this shipper in the last quarter",
                "new_shipper": "first dangerous-goods shipment from this shipper"}[f["shipper_history"]]

    RULE = ("Carrier DG manual 5.4: any unintended release in transport needs a written incident report to the "
            "transport regulator within 30 days; phone the regulator immediately if anyone is hospitalised, an area "
            "is evacuated or a road is closed; undeclared dangerous goods found in air-bound freight are reported "
            "to the regulator within 24 h.")

    def render(self, f, rng, surface):
        exc = self._what(f)
        leak = f"about {f['leaked_litres']} L released" if f["leaked_litres"] else None
        clean = {"contained": "spill kit used, absorbent drummed", "ongoing": "spill kit deployed, area still wet",
                 "not_started": "area taped off, waiting for the spill contractor", None: None}[f["cleanup"]]
        exp = self._exposure(f)
        hist = self._history(f)
        evac = "dock area evacuated for 30 minutes" if f["evacuation"] else None
        if surface == "json":
            return dumps(prune({
                "shipment": {"pro": f["pro"], "shipper": f["shipper"], "carrier": f["carrier"], "declared_as": self._ident(f)
                             if f["exception"] != "undeclared" else "general cargo (not declared as dangerous goods)",
                             "packages": f"{f['packages_affected']} of {f['packages_total']} affected"},
                "exception": {"found_at": f["terminal"], "found": fmt_ts(f["found"], style="iso"), "what": exc,
                              "actual_contents": self._ident(f) if f["exception"] == "undeclared" else None,
                              "release": leak, "cleanup": clean, "people": exp, "evacuation": evac},
                "onward": {"next_leg": self._leg(f), "hours_to_departure": f["hours_to_departure"]},
                "shipper_record": hist,
                "manual": self.RULE if chance(rng, 0.7) else None,
            }))
        if surface == "terminal_report":
            rows = [("PRO", f["pro"]), ("Terminal", f["terminal"]), ("Found", fmt_ts(f["found"], rng)),
                    ("Shipper", f["shipper"]), ("Material", self._ident(f)),
                    ("Declared", "no -- shipped as general cargo" if f["exception"] == "undeclared" else "yes"),
                    ("Exception", exc), ("Packages affected", f"{f['packages_affected']} of {f['packages_total']}"),
                    ("Release", leak or "none"), ("Clean-up", clean), ("Injuries", exp or "none"), ("Evacuation", evac),
                    ("Next leg", f"{self._leg(f)}, departs in {f['hours_to_departure']} h"), ("Shipper history", hist)]
            return "DANGEROUS GOODS EXCEPTION REPORT\n" + kv_lines(rows, rng) + "\n\n" + self.RULE
        t = f["found"]
        entries = [(0, f"{f['handler']} flags {f['pro']}: {exc}"),
                   (rng.randint(2, 8), f"packages: {self._ident(f)}" + (" (NOT declared)" if f["exception"] == "undeclared" else "")),
                   (rng.randint(2, 6), leak or "no release seen")]
        if exp:
            entries.append((rng.randint(1, 5), exp))
        if evac:
            entries.append((1, evac))
        if clean:
            entries.append((rng.randint(5, 20), clean))
        if hist:
            entries.append((rng.randint(1, 5), hist))
        entries.append((rng.randint(2, 10), f"trailer for the {f['next_leg_mode']} leg leaves in {f['hours_to_departure']} h; awaiting supervisor call"))
        lines, cur = [], t
        for dm, txt in entries:
            cur += timedelta(minutes=dm)
            lines.append(f"{cur.strftime('%H:%M')}  {noisy(txt, rng, 0.08)}")
        return f"DOCK LOG -- {f['terminal']} (supervisor {f['supervisor']})\n" + "\n".join(lines)

    def questions(self):
        return {
            "dg_handling": choice("How should the terminal handle the affected freight?", {
                "correct_and_forward": "Correct and forward: missing labels or a papers mismatch from a shipper with "
                                       "a clean record, or a segregation problem fixed on reloading.",
                "emergency_response": "Call emergency responders and isolate the area: a release with someone taken "
                                      "to hospital, an evacuation, or clean-up still ongoing or not started.",
                "hold_for_shipper": "Hold for the shipper: missing labels or a papers mismatch from a shipper with "
                                    "prior errors or a first-time dangerous-goods shipper.",
                "overpack_and_forward": "Overpack and forward: crushed outer packaging with the inner intact, or a "
                                        "release whose clean-up is complete with no one hospitalised.",
                "return_to_shipper": "Return to the shipper: dangerous goods were shipped undeclared.",
            }),
            "regulator_reporting": choice("What reporting to the transport regulator does this event need?", {
                "immediate_call_and_written_report": "An immediate phone report followed by a written incident report.",
                "internal_record_only": "None; record it internally and deal with the shipper.",
                "undeclared_goods_report": "A report of undeclared dangerous goods within 24 hours.",
                "written_incident_report": "A written incident report within the 30-day period.",
            }),
            "hazard_to_people": score("How much hazard does the situation pose to workers or the public?", [
                "Negligible: nothing has been released and packages are intact.",
                "Low: a small, contained release or a minor packaging problem.",
                "Significant: an ongoing release, exposure symptoms, or a fire or reaction risk.",
                "Severe: people hurt or evacuated, or an uncontrolled release.",
            ]),
            "declaration_gap": score("How serious is the gap between what was declared and what is in the "
                                     "packages?", [
                "None: the declaration matches the contents.",
                "Clerical: a detail such as a label or a phone number is wrong or missing.",
                "Material: the identification or packing group differs, or segregation was not respected.",
                "Complete: dangerous goods moved undeclared.",
            ]),
            "fit_to_travel_on": noul(
                "Once the terminal's corrective step is done, the freight can travel on safely.",
                yes="The terminal can put it right itself: a labelling or papers slip from a shipper with a clean "
                    "record, a segregation fix on reloading, or a damaged or leaking package that can be overpacked "
                    "once clean-up is complete and no one was hospitalised.",
                no="The goods were undeclared, a labelling or papers error needs a shipper with prior errors or a "
                   "first-time shipper to confirm, or a release involved a hospitalisation, an evacuation or "
                   "unfinished clean-up."),
        }


WORKFLOW = Workflow(
    name="shipment_exception_handling",
    title="Shipment exception handling",
    description="What a fictional logistics control tower, terminal or claims desk does when freight is late, "
                "damaged, out of temperature, missing or a dangerous-goods problem.",
    domain="freight logistics operations",
    schemas=[TransitDelayRecovery(), FreightDamageClaim(), ReeferTemperatureExcursion(), NoScanTrace(),
             DangerousGoodsException()],
    notes="Operational records only (TMS exceptions, status feeds, OS&D reports, reefer downloads, trace logs, "
          "terminal DG reports); no customer conversation and no reply-to-customer, churn or bill questions.",
)
