"""Customs declarations: the trade-compliance desk of a customs broker, and the customs officer at the frontier,
deciding how an import or an export should be treated.

Every question is about CUSTOMS treatment: tariff classification, customs value, preferential origin, export
licensing, release or hold, penalties and disclosures. A commercial invoice appears only as a customs document.
Nothing here pays, holds or rejects a vendor bill, and nothing matches a bill against an order or a delivery.
Tariff codes, duty rates, trade agreements, control-list entries and most country names are invented; no country
is a protected attribute that drives a decision. Each schema draws a `regime` first and lets it steer the factors,
as the reference module does, so most cases sit near a threshold, carry signals that disagree, or lack a fact.
"""
from __future__ import annotations

import math
from datetime import timedelta

from .base import (
    Schema,
    Workflow,
    between,
    bullets,
    chance,
    choice,
    dumps,
    email_addr,
    fmt_date,
    fmt_ts,
    ident,
    join_sentences,
    kv_lines,
    lognormal_amount,
    maybe,
    money,
    near,
    noisy,
    noul,
    ORG_A,
    person,
    pick,
    pick_w,
    prune,
    rand_date,
    rand_ts,
    regime,
    score,
    show,
    table,
)

COUNTRIES = ["Norland", "Estavia", "Marovia", "Solenne", "Tarsland", "Belmora", "Kestria", "Veskaria", "Lunmark",
             "Pellagia", "Corvesse", "Halvern", "Drevania", "Quellis"]
CURRENCIES = {"USD": 5, "EUR": 4, "GBP": 2, "CAD": 1, "AUD": 1}


def _countries(rng, n):
    """n distinct country names: invented names most of the time, neutral 'Country X' labels otherwise."""
    if chance(rng, 0.2):
        return [f"Country {c}" for c in rng.sample("ABCDEFGHJK", n)]
    return rng.sample(COUNTRIES, n)


def _money_fmt(rng, cur):
    """One money style per rendered state (a real document does not switch styles mid-page)."""
    style = pick(rng, ["sym", "sym", "code", "plain"])
    return lambda x: money(x, cur=cur, style=style)


def _date_fmt(rng):
    """One date style per state. No month-first numeric style: expiry and validity dates decide several questions
    here, and '02/05/2026' would be ambiguous."""
    style = pick(rng, ["iso", "iso", "long", "eu"])
    return lambda d: fmt_date(d, style=style)


# base.org combinations that happen to name real, well-known businesses
_REAL_ORG_NAMES = {"Harbor Freight", "Keystone Foods", "Juniper Systems", "Juniper Networks", "Granite Group",
                   "Meridian Instruments"}


# name suffixes by the role an organisation plays, so a university is not called 'Logistics'
ROLE_SUFFIXES = {
    "maker": ["Industries", "Manufacturing", "Works", "Technologies", "Instruments", "Engineering", "Systems"],
    "importer": ["Outfitters", "Home Goods", "Brands", "Retail", "Distribution", "Supply Co.", "Trading"],
    "trader": ["Trading", "Holdings", "Distribution", "Supply Co.", "Commerce"],
    "forwarder": ["Logistics", "Freight", "Forwarding"],
    "lab": ["University", "Institute of Technology", "Research Centre"],
    "state_group": ["Industrial Group", "Heavy Industries"],
    "seafood": ["Seafoods", "Fisheries"],
    "shop": ["Online", "Direct", "Shop", "Store"],
    "courier": ["Express", "Parcel", "Courier"],
    "production": ["Films", "Productions", "Media"],
    "fashion": ["Fashion", "Apparel", "Studio"],
    "touring": ["Touring", "Live", "Music"],
}


def _names(rng, *roles):
    """One fictional name per role, with distinct first words (an importer and its supplier should not share one)."""
    out, seen = [], set()
    for role in roles:
        while True:
            first = rng.choice(ORG_A)
            name = f"{first} {rng.choice(ROLE_SUFFIXES[role])}"
            if first not in seen and name not in _REAL_ORG_NAMES:
                break
        seen.add(first)
        out.append(name)
    return out


def _code(rng, chapter, avoid=None):
    """An invented tariff subheading under a plausible chapter prefix."""
    while True:
        c = f"{chapter}{rng.randint(10, 99)}.{rng.randint(10, 99)}"
        if c != avoid:
            return c


def _h(v):
    return v.replace("_", " ") if isinstance(v, str) else v


def _domain(name):
    return "".join(ch for ch in name.lower().split()[0] if ch.isalpha()) + ".example"


def _yn(v):
    return "yes" if v else "no"


# =========================================================================== schema 1
CLASS_PRODUCTS = [
    {"key": "fitness_band", "kg": 0.09, "goods": "wrist-worn fitness band with heart-rate sensor, step counter and time display",
     "unit": "pcs", "per": "each", "price": 18.0,
     "a": ("91", "wrist-worn timekeeping instruments", 4.5),
     "b": ("85", "apparatus for transmitting and receiving data", 0.0),
     "cue_a": "the product page leads with the watch face, alarms and swappable straps",
     "cue_b": "the product page leads with heart-rate tracking and syncing to a phone app",
     "cue_mix": "the product page gives the clock and the fitness tracking equal billing"},
    {"key": "massage_cushion", "kg": 2.4, "goods": "heated back cushion with vibration massage and a wired remote",
     "unit": "pcs", "per": "each", "price": 34.0,
     "a": ("85", "electro-thermic appliances for domestic use", 2.7),
     "b": ("90", "massage apparatus", 0.0),
     "cue_a": "the heating pad is the main feature and the vibration motor is a small add-on",
     "cue_b": "the kneading rollers are the main feature and heat is an optional setting",
     "cue_mix": "the heat and massage functions are advertised as equally important"},
    {"key": "plush_backpack", "kg": 0.55, "goods": "children's backpack with a detachable plush animal",
     "unit": "pcs", "per": "each", "price": 9.5,
     "a": ("42", "rucksacks with an outer surface of textile", 12.0),
     "b": ("95", "stuffed toys", 0.0),
     "cue_a": "the plush animal is a small zip-off keyring on a full-size school bag",
     "cue_b": "the plush animal is full-size and the bag is a small pouch sewn into its back",
     "cue_mix": "the plush animal and the bag are about the same size"},
    {"key": "led_jacket", "kg": 0.6, "goods": "cycling jacket with a built-in rechargeable LED strip",
     "unit": "pcs", "per": "each", "price": 41.0,
     "a": ("62", "jackets of synthetic fibres", 12.0),
     "b": ("85", "electric lighting and signalling equipment for cycles", 2.7),
     "cue_a": "the LED strip clips into a sleeve pocket and the jacket is sold as outerwear",
     "cue_b": "it is marketed as a 'wearable light', the fabric acting mainly as a carrier for the LEDs",
     "cue_mix": "the catalogue lists it under both clothing and bike lights"},
    {"key": "ink_refill", "kg": 0.7, "goods": "printer refill kit: refillable cartridges, bottled ink and a syringe tool",
     "unit": "kits", "per": "per kit", "price": 12.0,
     "a": ("84", "parts and accessories of printers", 0.0),
     "b": ("32", "printing ink", 6.5),
     "cue_a": "most of the kit's value is in the refillable cartridges with reset chips",
     "cue_b": "most of the kit's value is in the four 100 ml ink bottles",
     "cue_mix": "cartridges and ink make up roughly equal shares of the kit's cost"},
    {"key": "food_flask", "kg": 0.45, "goods": "insulated steel food flask with built-in tea strainer",
     "unit": "pcs", "per": "each", "price": 7.8,
     "a": ("73", "table and kitchen articles of stainless steel", 3.2),
     "b": ("96", "vacuum flasks and other vacuum vessels", 6.0),
     "cue_a": "the factory specification describes a double wall that is not vacuum-sealed",
     "cue_b": "the factory specification describes a vacuum-sealed double wall",
     "cue_mix": "the listing says 'vacuum-style insulation' without saying whether the wall is evacuated"},
    {"key": "dash_cam", "kg": 0.35, "goods": "vehicle dash camera with GPS module and 3-inch screen",
     "unit": "pcs", "per": "each", "price": 29.0,
     "a": ("85", "video recording apparatus", 0.0),
     "b": ("90", "navigational instruments", 3.7),
     "cue_a": "it records continuous video; the GPS only stamps position and speed on the footage",
     "cue_b": "it gives turn-by-turn directions on the screen and recording is a secondary mode",
     "cue_mix": "the listing promotes the recording and the route guidance equally"},
    {"key": "glove_liners", "kg": 0.05, "goods": "knitted touchscreen glove liners with conductive fingertips",
     "unit": "pairs", "per": "per pair", "price": 2.4,
     "a": ("61", "gloves, knitted or crocheted", 13.0),
     "b": ("63", "other made-up textile articles", 7.0),
     "cue_a": "they are fully shaped five-finger gloves, worn alone or under ski gloves",
     "cue_b": "they are sold as loose fingertip sleeves in sets of ten, not as whole gloves",
     "cue_mix": "the photos show whole liners and loose fingertip pieces in the same pack"},
    {"key": "work_platform", "kg": 13.5, "goods": "folding aluminium multi-position ladder with work platform",
     "unit": "pcs", "per": "each", "price": 88.0,
     "a": ("76", "ladders and steps of aluminium", 5.0),
     "b": ("76", "structures and parts of structures of aluminium", 6.0),
     "cue_a": "it is used mainly as a ladder; the platform board is an accessory",
     "cue_b": "it is sold as a mobile work platform with guard rails, climbed only for access",
     "cue_mix": "it converts between ladder and platform modes and no main mode is stated"},
]
CLASS_BY_KEY = {p["key"]: p for p in CLASS_PRODUCTS}
PRICE_NOTES = {
    "volume_discount": "importer says a volume discount applies because this is a full-container order",
    "end_of_line": "supplier is clearing the previous model at a reduced price",
    "transfer_price": "intra-group transfer price set by the parent company",
    "launch_price": "supplier gave an introductory price to win a new customer",
    "seconds": "goods are B-grade stock with cosmetic defects",
    "see_attachment": "importer refers to an attached supplier letter; no letter is on file",
}
PRICE_REPLY = {
    "volume_discount": "It's lower than usual because we took a full container and got a volume discount.",
    "end_of_line": "The factory is clearing last year's model, so we got a good price.",
    "transfer_price": "The supplier is our sister company; group finance sets the transfer price.",
    "launch_price": "It's an introductory price, they want us as a customer.",
    "seconds": "These are B-grade units with small cosmetic marks, hence the price.",
    "see_attachment": "The supplier's letter explaining the price is attached.",
}
TRANSPORT_DOC = {"sea": "bill_of_lading", "air": "air_waybill", "road": "road_consignment_note"}
DOC_STATE = {"on_file": "on file", "final": "on file", "draft": "draft only", "missing": "not received"}


class ImportEntryReview(Schema):
    name = "import_entry_review"
    description = ("A customs broker's draft import entry before lodgement: a product that could sit under two "
                   "headings, a price set against comparable entries, and the document set.")
    surfaces = {"json": 0.45, "entry_worksheet": 0.3, "importer_reply": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        p = pick(rng, CLASS_PRODUCTS)
        declared = pick(rng, ["a", "b"])
        cue = pick_w(rng, {"declared": 4, "alternative": 3, "mixed": 3, None: 1})
        spec = chance(rng, 0.55)
        ruling = pick_w(rng, {"none": 6, "similar_declared": 1, "similar_alternative": 1, "pending": 1})
        tenure = pick_w(rng, {"first_entry": 2, "under_1y": 3, "established": 5})
        related = chance(rng, 0.2)
        expl = pick_w(rng, {None: 3, "volume_discount": 2, "end_of_line": 2, "launch_price": 1, "seconds": 1})
        gap = near(rng, 0.28, 0.14, lo=0.05, hi=0.5)
        comps = pick(rng, [4, 6, 11, 23, 58])
        incoterm = pick(rng, ["EXW", "FOB", "FOB", "CIF", "DAP"])
        freight_added = pick_w(rng, {True: 6, False: 3, None: 1})
        royalty = pick_w(rng, {"none": 7, "paid_separately": 2, None: 1})
        packing = pick_w(rng, {"on_file": 8, "missing": 2})
        transport = pick_w(rng, {"final": 6, "draft": 2, "missing": 1})
        mismatch = chance(rng, 0.15)
        amendments = pick(rng, [0, 0, 0, 1, 2, 4])
        if reg == "clear":
            if chance(rng, 0.5):
                cue, spec, ruling = "declared", True, pick(rng, ["none", "similar_declared"])
                gap, expl = between(rng, -0.08, 0.1), pick(rng, [None, "volume_discount"])
                royalty, related, freight_added = "none", False, True
                packing, transport, mismatch, tenure, amendments = "on_file", "final", False, "established", 0
            else:
                cue, spec, ruling = "alternative", False, pick(rng, ["none", "similar_alternative"])
                gap, expl, tenure = between(rng, 0.45, 0.62), pick(rng, [None, "see_attachment"]), "first_entry"
                royalty, transport = pick(rng, ["paid_separately", None]), pick(rng, ["draft", "missing"])
        elif reg == "borderline":
            cue = pick_w(rng, {"mixed": 4, "declared": 2, "alternative": 2, None: 1})
            gap = near(rng, 0.3, 0.08, lo=0.18, hi=0.42)
            if expl is None and chance(rng, 0.6):
                expl = pick(rng, ["volume_discount", "end_of_line", "launch_price", "seconds"])
        else:
            # a trusted importer with one alarming fact, or a prior ruling that the product facts contradict
            variant = pick(rng, ["trusted_importer_low_price", "ruling_vs_product", "related_party"])
            if variant == "trusted_importer_low_price":
                tenure, amendments, cue, spec = "established", 0, "declared", True
                gap, expl = between(rng, 0.34, 0.5), maybe(rng, pick(rng, ["end_of_line", "seconds"]), 0.4)
            elif variant == "ruling_vs_product":
                ruling, cue, gap = "similar_declared", "alternative", between(rng, -0.05, 0.15)
                tenure = pick(rng, ["under_1y", "established"])
            else:
                related, expl, tenure = True, "transfer_price", "established"
                gap, royalty = between(rng, 0.25, 0.4), pick(rng, ["paid_separately", "none", None])
            if chance(rng, 0.5):
                spec = False
        if expl == "transfer_price":
            related = True
        elif related and expl is None and chance(rng, 0.6):
            expl = "transfer_price"
        if incoterm in ("CIF", "DAP"):
            freight_added = None            # freight is already inside the invoice price
        if tenure == "first_entry":
            amendments = 0
        cur = pick_w(rng, CURRENCIES)
        comp = round(p["price"] * rng.uniform(0.85, 1.2), 2)
        unit_price = round(comp * (1 - gap), 2)
        qty = max(20, int(round(pick(rng, [5000, 12000, 30000, 75000]) / p["price"], -1)))
        total = round(unit_price * qty, 2)
        code_a = _code(rng, p["a"][0])
        code_b = _code(rng, p["b"][0], avoid=code_a)
        gross = max(5, int(round(qty * p["kg"] * rng.uniform(1.1, 1.3))))   # goods plus packaging
        mode = (pick_w(rng, {"sea": 3, "air": 5, "road": 1}) if gross < 800
                else pick_w(rng, {"sea": 6, "air": 1, "road": 2}))
        importer, supplier = _names(rng, "importer", "maker")
        return {
            "regime": reg, "product": p["key"], "declared_heading": declared, "code_a": code_a, "code_b": code_b,
            "function_cue": cue, "tech_sheet_on_file": spec, "ruling": ruling,
            "importer": importer, "importer_contact": person(rng), "importer_tenure": tenure,
            "entries_12m": {"first_entry": 0, "under_1y": rng.randint(1, 9), "established": rng.randint(14, 160)}[tenure],
            "amendments_12m": amendments, "supplier": supplier, "supplier_country": _countries(rng, 1)[0],
            "related_party": related, "price_explanation": expl, "currency": cur,
            "comparable_median": comp, "comparables_n": comps, "unit_price": unit_price,
            "value_gap_pct": round((1 - unit_price / comp) * 100, 1), "quantity": qty, "invoice_total": total,
            "incoterm": incoterm, "freight_amount": round(total * rng.uniform(0.03, 0.09), 2) if incoterm in ("EXW", "FOB") else None,
            "freight_added": freight_added, "royalty": royalty,
            "royalty_pct": pick(rng, [3, 5, 6, 8]) if royalty == "paid_separately" else None,
            "packing_list": packing, "transport_doc": transport, "weights_mismatch": mismatch,
            "gross_kg_packing": gross,
            "gross_kg_transport": int(round(gross * rng.uniform(1.07, 1.22))) if mismatch else gross,
            "mode": mode, "broker_ref": ident(rng, "BRK", 6, "-"),
            "handler": person(rng), "eta": rand_date(rng),
            "tone": pick(rng, ["cooperative", "cooperative", "hurried", "defensive"]),
        }

    def _view(self, f):
        p = CLASS_BY_KEY[f["product"]]
        d = f["declared_heading"]
        o = "b" if d == "a" else "a"
        codes = {"a": f["code_a"], "b": f["code_b"]}
        cue = {"declared": p["cue_" + d], "alternative": p["cue_" + o], "mixed": p["cue_mix"],
               None: None}[f["function_cue"]]
        ruling = {"none": None,
                  "similar_declared": f"a binding ruling on the previous model placed it under {codes[d]}",
                  "similar_alternative": f"a binding ruling on the previous model placed it under {codes[o]}",
                  "pending": "a binding-ruling application for this model is pending"}[f["ruling"]]
        return p, (codes[d], p[d][1], p[d][2]), (codes[o], p[o][1], p[o][2]), cue, ruling

    def render(self, f, rng, surface):
        p, dc, ac, cue, ruling = self._view(f)
        tkey = TRANSPORT_DOC[f["mode"]]
        royalty_txt = {"none": "none", None: None,
                       "paid_separately": f"{f['royalty_pct']}% of resale turnover, paid under a separate licence "
                                          f"agreement"}[f["royalty"]]
        if surface == "json":
            return dumps(prune({
                "entry": {"broker_ref": f["broker_ref"], "importer": f["importer"], "supplier": f["supplier"],
                          "supplier_country": f["supplier_country"], "transport_mode": f["mode"],
                          "eta": f["eta"].isoformat(), "prepared_by": f["handler"]},
                "classification": {"goods": p["goods"], "drafted_code": dc[0], "drafted_heading": dc[1],
                                   "drafted_duty_pct": dc[2], "alternative_code": ac[0], "alternative_heading": ac[1],
                                   "alternative_duty_pct": ac[2], "product_note": cue,
                                   "technical_sheet_on_file": f["tech_sheet_on_file"], "ruling_history": ruling},
                "valuation": {"basis": "CIF at the border", "currency": f["currency"], "quantity": f["quantity"],
                              "unit": p["unit"], "unit_price": f["unit_price"], "invoice_total": f["invoice_total"],
                              "incoterm": f["incoterm"], "freight_to_border": f["freight_amount"],
                              "freight_added_to_value": f["freight_added"],
                              "comparable_median_unit_price": f["comparable_median"],
                              "comparable_entries_12m": f["comparables_n"], "buyer_seller_related": f["related_party"],
                              "price_comment": PRICE_NOTES.get(f["price_explanation"]),
                              "licence_fee_to_brand_owner": royalty_txt},
                "documents": {"commercial_invoice": "on file", "packing_list": DOC_STATE[f["packing_list"]],
                              tkey: DOC_STATE[f["transport_doc"]],
                              "gross_kg_packing_list": f["gross_kg_packing"] if f["packing_list"] == "on_file" else None,
                              "gross_kg_" + tkey: f["gross_kg_transport"] if f["transport_doc"] != "missing" else None},
                "importer_history": {"first_entry_with_broker": f["importer_tenure"] == "first_entry",
                                     "entries_12m": f["entries_12m"],
                                     "post_entry_amendments_12m": f["amendments_12m"]},
            }))
        m, fd = _money_fmt(rng, f["currency"]), _date_fmt(rng)
        tname = _h(tkey)
        if f["incoterm"] in ("EXW", "FOB"):
            freight_line = f"{m(f['freight_amount'])} per forwarder, " + {
                True: "added to value", False: "not added to value", None: "not stated whether added"}[f["freight_added"]]
        else:
            freight_line = f"included in the {f['incoterm']} price"
        if surface == "entry_worksheet":
            hist = {"first_entry": " (first entry with us)", "under_1y": f" (client < 1 year, {f['entries_12m']} entries)",
                    "established": f" (client since before last year, {f['entries_12m']} entries in 12 mo)"}
            head = kv_lines([
                ("Broker ref", f["broker_ref"]), ("Importer", f["importer"] + hist[f["importer_tenure"]]),
                ("Supplier", f"{f['supplier']}, {f['supplier_country']}"),
                ("Transport", f"{f['mode']}, ETA {fd(f['eta'])}"), ("Goods", p["goods"]),
                ("Quantity", f"{f['quantity']:,} {p['unit']}"),
                ("Draft code", f"{dc[0]} {dc[1]} ({dc[2]}% duty)"),
                ("Other candidate", f"{ac[0]} {ac[1]} ({ac[2]}% duty)"),
                ("Product note", cue), ("Technical sheet", "on file" if f["tech_sheet_on_file"] else "not on file"),
                ("Ruling history", ruling),
            ], rng)
            val = kv_lines([
                ("Valuation basis", "CIF at the border"),
                ("Invoice price", f"{m(f['unit_price'])} {p['per']}, {f['incoterm']}, total {m(f['invoice_total'])}"),
                ("Freight to border", freight_line),
                ("Comparables last 12 months", f"median {m(f['comparable_median'])} {p['per']} across "
                                               f"{f['comparables_n']} entries"),
                ("Buyer and seller related", _yn(f["related_party"])),
                ("Price comment", PRICE_NOTES.get(f["price_explanation"])),
                ("Licence fee to brand owner", royalty_txt),
            ], rng, skip_none=pick(rng, [True, False]))
            docs = ["[x] commercial invoice",
                    f"[{'x' if f['packing_list'] == 'on_file' else ' '}] packing list"
                    + (f" ({f['gross_kg_packing']:,} kg gross)" if f["packing_list"] == "on_file" else ""),
                    f"[{'x' if f['transport_doc'] == 'final' else ' '}] {tname}"
                    + (" -- draft only" if f["transport_doc"] == "draft" else "")
                    + (f" ({f['gross_kg_transport']:,} kg gross)" if f["transport_doc"] != "missing" else "")]
            tail = f"Post-entry amendments for this importer, last 12 months: {f['amendments_12m']}"
            return ("IMPORT ENTRY -- PRE-LODGEMENT WORKSHEET\n" + head + "\n\nVALUE\n" + val + "\n\nDOCUMENTS\n"
                    + "\n".join(docs) + "\n\n" + tail)
        # importer_reply: the importer answers the broker's standard pre-lodgement questions
        c = f["importer_contact"]
        a1 = f"Product: {p['goods']}."
        a1 += (f" On your question about its main function: {cue}." if cue
               else " Not sure what more you need on the product, it's as described on the listing.")
        a1 += (" Factory spec sheet attached." if f["tech_sheet_on_file"] else " We don't have a technical sheet from the factory.")
        a2 = f"{m(f['unit_price'])} {p['per']}, {f['incoterm']}."
        a2 += " " + (PRICE_REPLY[f["price_explanation"]] if f["price_explanation"] else "That is simply the agreed price.")
        if f["related_party"] and f["price_explanation"] != "transfer_price":
            a2 += " For the record the supplier is part of our group."
        if f["incoterm"] in ("EXW", "FOB"):
            a2 += {True: f" Freight to your port was {m(f['freight_amount'])}, forwarder's bill attached.",
                   False: " Freight is billed to us separately by the forwarder.", None: ""}[f["freight_added"]]
        a3 = {"none": "We pay no royalties or licence fees on these.", None: "",
              "paid_separately": f"We pay the brand owner {f['royalty_pct']}% of our sales under a licence agreement, "
                                 f"separate from the goods price."}[f["royalty"]]
        have = ["invoice"] + (["packing list"] if f["packing_list"] == "on_file" else [])
        a4 = "Attached: " + ", ".join(have) + "."
        a4 += {"final": f" {tname.capitalize()} attached too.", "draft": f" The {tname} is still a draft from the carrier.",
               "missing": f" We haven't had the {tname} yet."}[f["transport_doc"]]
        close = {"cooperative": "Let me know if you need anything else.",
                 "hurried": "We need these in stock before month end, so please lodge as soon as you can.",
                 "defensive": "We've imported similar goods before without questions, so I'm not sure why this is needed."}
        answers = [f"1. {a1}", f"2. {a2}"] + ([f"3. {a3}"] if a3 else []) + [f"{4 if a3 else 3}. {a4}"]
        desk = kv_lines([("Drafted code", f"{dc[0]} ({dc[2]}%), alternative {ac[0]} ({ac[2]}%)"),
                         ("Comparables, 12 mo", f"n={f['comparables_n']}, median {m(f['comparable_median'])} {p['per']}"),
                         ("Gross weight", f"packing list {f['gross_kg_packing']:,} kg / {tname} {f['gross_kg_transport']:,} kg"
                          if f["packing_list"] == "on_file" and f["transport_doc"] != "missing" else None),
                         ("Importer", f"{_h(f['importer_tenure'])}, post-entry amendments in 12 mo: "
                                      f"{f['amendments_12m']}"),
                         ("Ruling history", ruling)], rng, style="dash")
        lines = [f"Subject: RE: {f['broker_ref']} -- questions before we lodge",
                 f"From: {c} <{email_addr(c, _domain(f['importer']))}>", f"To: {f['handler']}", "",
                 f"Hi {f['handler'].split()[0]},", "", "Answers below.", "",
                 noisy("\n".join(answers), rng, 0.15), "", close[f["tone"]], "", c.split()[0], "",
                 "---- desk data (added by broker) ----", desk]
        return "\n".join(lines)

    def questions(self):
        return {
            "heading_confidence": score("How confident can the desk be that the commodity code on the draft is the right one?", [
                "Low: the product note on the goods' main function points to the other candidate heading.",
                "Uncertain: there is no product note, it fits both headings equally, or it fits the drafted heading "
                "while a ruling on the previous model went the other way.",
                "Fair: the product note fits the drafted heading, but no technical sheet backs it up.",
                "High: the product note fits the drafted heading and the factory's technical sheet is on file.",
            ]),
            "undervaluation_concern": score("How strongly do the price and cost details suggest the customs value is understated?", [
                "Negligible: the price is in line with comparable entries and no cost element is left out.",
                "Mild: there is a gap or an omission, but a credible explanation covers it.",
                "Marked: the gap or a missing cost element needs a documented answer before lodging.",
                "Severe: the value looks materially understated and nothing credible explains it.",
            ]),
            "pre_lodgement_step": choice("What should the broker do with this draft entry before the goods arrive?", {
                "lodge_as_drafted": "Lodge the entry with the commodity code and value exactly as drafted.",
                "amend_then_lodge": "Change the commodity code or add the missing cost elements to the value, then lodge.",
                "query_importer": "Hold the draft and ask the importer for product or pricing evidence before lodging.",
                "provisional_value": "Lodge now but declare the value as provisional while a price element is confirmed.",
                "specialist_referral": "Pass the file to the broker's classification or valuation specialist before "
                                       "anything is lodged.",
            }),
            "primary_issue": choice("Which aspect of this entry most needs attention before it is lodged?", {
                "commodity_code": "The choice between the two candidate tariff headings.",
                "customs_value": "Whether the invoice price plus any added costs gives the correct customs value.",
                "document_set": "Gaps or disagreements in the shipping documents.",
                "nothing_material": "Nothing stands out; the entry looks in order.",
            }),
            "documents_sufficient": noul(
                "The documents on file are sufficient to lodge this entry.",
                yes="The invoice, the packing list and a final transport document are on file and agree on weights.",
                no="A needed document is missing or only a draft, or the documents disagree on the weight shipped."),
        }


# =========================================================================== schema 2
ORIGIN_GOODS = [
    {"key": "cookware", "goods": "stainless steel cookware sets", "rule": "rvc", "chapter": "73", "mfn": 6.5,
     "price": 38.0, "unit": "sets"},
    {"key": "led_drivers", "goods": "LED driver modules for commercial lighting", "rule": "rvc", "chapter": "85",
     "mfn": 3.7, "price": 14.0, "unit": "pcs"},
    {"key": "seat_covers", "goods": "polyester car seat covers", "rule": "rvc", "chapter": "87", "mfn": 4.5,
     "price": 11.0, "unit": "sets"},
    {"key": "brackets", "goods": "machined aluminium mounting brackets", "rule": "cth", "chapter": "76", "mfn": 6.0,
     "price": 4.2, "unit": "pcs", "input_ok": "aluminium bar stock", "input_bad": "cast aluminium bracket blanks",
     "work": "machining and anodising"},
    {"key": "cabinets", "goods": "flat-pack oak bedside cabinets", "rule": "cth", "chapter": "94", "mfn": 5.6,
     "price": 46.0, "unit": "pcs", "input_ok": "sawn oak boards", "input_bad": "pre-cut cabinet panels",
     "work": "cutting, jointing and finishing"},
    {"key": "dinnerware", "goods": "glazed stoneware dinnerware sets", "rule": "cth", "chapter": "69", "mfn": 8.0,
     "price": 22.0, "unit": "sets", "input_ok": "clay and glaze materials", "input_bad": "unglazed bisque-fired blanks",
     "work": "glazing and firing"},
    {"key": "sweaters", "goods": "knitted wool-blend sweaters", "rule": "double", "chapter": "61", "mfn": 12.0,
     "price": 16.0, "unit": "pcs", "input": "yarn", "work": "knitting and making-up"},
    {"key": "bed_linen", "goods": "woven cotton bed linen sets", "rule": "double", "chapter": "63", "mfn": 12.0,
     "price": 13.0, "unit": "sets", "input": "woven fabric", "work": "cutting and sewing"},
    {"key": "hake", "goods": "frozen hake fillets", "rule": "wholly", "chapter": "03", "mfn": 7.5, "price": 5.5,
     "unit": "kg", "input": "fish", "work": "filleting and freezing"},
]
ORIGIN_BY_KEY = {g["key"]: g for g in ORIGIN_GOODS}
# a neighbouring product from the same supplier: what a declaration covering the wrong goods names instead
ORIGIN_NEIGHBOUR = {"cookware": "cutlery", "led_drivers": "LED panels", "seat_covers": "floor mats",
                    "brackets": "aluminium profiles", "cabinets": "oak dining chairs", "dinnerware": "stoneware planters",
                    "sweaters": "knitted scarves", "bed_linen": "cotton towels", "hake": "cod fillets"}
PROOF_NAMES = {"invoice_statement": "exporter's origin statement on the commercial invoice",
               "supplier_declaration": "long-term supplier declaration",
               "certificate": "certificate of origin issued in the exporting country"}
PROOF_DEFECTS = {"invoice_statement": ["the exporter's registration number is missing",
                                       "the statement is not signed or dated"],
                 "certificate": ["the box naming the agreement is left blank",
                                 "the goods description reads only 'household goods'"]}


class PreferentialOriginClaim(Schema):
    name = "preferential_origin_claim"
    description = ("Checking a preferential-origin claim under a trade agreement: the product rule, the supplier's "
                   "origin evidence and how the goods travelled.")
    surfaces = {"json": 0.45, "origin_worksheet": 0.3, "broker_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        g = pick(rng, ORIGIN_GOODS)
        partner, home, third, hub = _countries(rng, 4)
        rule_status = pick_w(rng, {"comfortable": 3, "marginal": 3, "unclear": 2, "not_met": 1})
        proof_status = pick_w(rng, {"valid": 5, "expired": 2, "wrong_goods": 1, "defective": 1, "missing": 1})
        routing = pick_w(rng, {"direct": 5, "tranship_docs": 2, "tranship_no_docs": 1, "stored": 1})
        prior = pick_w(rng, {"none": 6, "passed": 2, "failed": 1})
        first_from_supplier = chance(rng, 0.3)
        if reg == "clear":
            if chance(rng, 0.5):
                rule_status, proof_status, routing = "comfortable", "valid", "direct"
                prior = pick(rng, ["none", "passed"])
            else:
                rule_status, proof_status = "not_met", pick(rng, ["missing", "wrong_goods"])
                routing = pick(rng, ["tranship_no_docs", "direct"])
        elif reg == "borderline":
            rule_status = pick_w(rng, {"marginal": 4, "unclear": 2, "comfortable": 1})
            proof_status = pick_w(rng, {"valid": 3, "expired": 3, "defective": 2})
            routing = pick_w(rng, {"direct": 2, "tranship_docs": 2, "stored": 1})
        else:
            variant = pick(rng, ["evidence_ok_rule_doubtful", "rule_ok_evidence_bad", "history_against"])
            if variant == "evidence_ok_rule_doubtful":
                proof_status, rule_status = "valid", pick(rng, ["not_met", "unclear"])
            elif variant == "rule_ok_evidence_bad":
                rule_status, proof_status = "comfortable", pick(rng, ["expired", "missing", "defective"])
            else:
                rule_status, proof_status, prior = "comfortable", "valid", "failed"
                routing = pick(rng, ["stored", "tranship_docs"])
        # the evidence type follows the status: only declarations expire; only statements/certificates are defective
        if proof_status == "expired":
            proof_type = "supplier_declaration"
        elif proof_status == "defective":
            proof_type = pick(rng, ["invoice_statement", "certificate"])
        elif proof_status == "wrong_goods":
            proof_type = pick(rng, ["supplier_declaration", "certificate"])
        else:
            proof_type = pick_w(rng, {"invoice_statement": 4, "supplier_declaration": 4, "certificate": 2})
        ship = rand_date(rng)
        thr = pick(rng, [40, 45, 50, 55]) if g["rule"] == "rvc" else None
        rvc = None
        rvc_basis = None
        if g["rule"] == "rvc":
            rvc_basis = "supplier cost breakdown"
            if rule_status == "comfortable":
                rvc = round(thr + rng.uniform(8, 20), 1)
            elif rule_status == "marginal":
                rvc = round(thr + rng.uniform(0.2, 3.0), 1)
            elif rule_status == "not_met":
                rvc = round(thr - rng.uniform(0.6, 7.0), 1)
            elif chance(rng, 0.5):
                rvc, rvc_basis = round(thr + rng.uniform(-2.0, 4.0), 1), "importer's estimate from list prices"
            else:
                rvc_basis = None
        days_expired = near(rng, 9, 9, lo=1, integer=True) if proof_status == "expired" else None
        if proof_type == "supplier_declaration":
            valid_to = ship - timedelta(days=days_expired) if days_expired else ship + timedelta(days=rng.randint(15, 300))
            valid_from = valid_to - timedelta(days=364)
        else:
            valid_to = valid_from = None
        cur = pick_w(rng, CURRENCIES)
        qty = int(round(pick(rng, [8000, 20000, 60000, 150000]) / g["price"], -1))
        value = round(qty * g["price"] * rng.uniform(0.85, 1.15), 2)
        pref = pick(rng, [0.0, 0.0, 0.0, round(g["mfn"] / 2, 1)])
        if prior != "none":
            first_from_supplier = False     # an earlier verification means the supplier's goods came through before
        orgs = _names(rng, "seafood" if g["key"] == "hake" else "maker", "importer")
        return {
            "regime": reg, "product": g["key"], "rule_type": g["rule"], "rule_status": rule_status,
            "rvc_threshold": thr, "rvc_pct": rvc, "rvc_basis": rvc_basis,
            "tolerance_pct": near(rng, 8.5, 1.4, lo=6.0, hi=9.9, digits=1) if rule_status == "marginal" else None,
            "partner": partner, "importing_country": home, "third_country": third, "hub": hub,
            "agreement": f"{partner}-{home} " + pick(rng, ["Economic Partnership Agreement", "Free Trade Agreement",
                                                            "Trade and Cooperation Agreement"]),
            "proof_type": proof_type, "proof_status": proof_status, "days_expired": days_expired,
            "declaration_from": valid_from, "declaration_to": valid_to,
            "defect": pick(rng, PROOF_DEFECTS[proof_type]) if proof_status == "defective" else None,
            "routing": routing, "hub_days": rng.randint(2, 40) if routing != "direct" else None,
            "hub_certificate": chance(rng, 0.5) if routing == "stored" else None,
            "prior_verification": prior, "first_shipment_from_supplier": first_from_supplier,
            "importer_insists": chance(rng, 0.35), "ship_date": ship,
            "arrival": ship + timedelta(days=rng.randint(6, 34)), "currency": cur, "quantity": qty,
            "customs_value": value, "mfn_rate": g["mfn"], "pref_rate": pref,
            "duty_saving": round(value * (g["mfn"] - pref) / 100, 2),
            "supplier": orgs[0], "importer": orgs[1], "broker_contact": person(rng),
            "entry_ref": ident(rng, "ENT", 7, ""), "code": _code(rng, g["chapter"]),
        }

    def _rule_text(self, f):
        g = ORIGIN_BY_KEY[f["product"]]
        if g["rule"] == "rvc":
            return f"regional value content of at least {f['rvc_threshold']}% of the ex-works price"
        if g["rule"] == "cth":
            return "change of tariff heading: every non-originating material must sit under a different heading"
        if g["rule"] == "double":
            return f"both the {g['input']} and the {g['work']} must be produced in a party to the agreement"
        return "wholly obtained: the fish must be taken by a vessel registered and flagged in a party to the agreement"

    def _rule_evidence(self, f):
        g, st, pa, th = ORIGIN_BY_KEY[f["product"]], f["rule_status"], f["partner"], f["third_country"]
        if g["rule"] == "rvc":
            if f["rvc_pct"] is None:
                return "supplier has not yet sent a cost breakdown"
            return f"RVC {f['rvc_pct']}% per {f['rvc_basis']}"
        if g["rule"] == "cth":
            return {"comfortable": f"{g['input_ok']} bought from {th}; {g['work']} done in {pa}",
                    "marginal": f"{g['input_ok']} bought from {th}; part of the run used {g['input_bad']} from {th}, "
                                f"worth {f['tolerance_pct']}% of the ex-works price (tolerance 10%)",
                    "unclear": f"supplier lists inputs from {th} but gives no tariff heading for them",
                    "not_met": f"{g['input_bad']} bought from {th}; only finishing and packing done in {pa}"}[st]
        if g["rule"] == "double":
            return {"comfortable": f"{g['input']} produced in {pa} by the supplier's own mill",
                    "marginal": f"{g['input']} from a mill in {th}, counted as originating under a cumulation clause; "
                                f"the mill's own declaration is only a draft",
                    "unclear": f"supplier has not said where the {g['input']} was made",
                    "not_met": f"{g['input']} bought from {th}; only the {g['work']} done in {pa}"}[st]
        return {"comfortable": f"catch certificate shows a vessel flagged in {pa}; {g['work']} in {pa}",
                "marginal": f"vessel flagged in {pa} but operated under charter by a company from {th}",
                "unclear": "catch certificate names the vessel but not its flag",
                "not_met": f"caught by a vessel flagged in {th}; {g['work']} in {pa}"}[st]

    def _proof_text(self, f, fd):
        name = PROOF_NAMES[f["proof_type"]]
        g = ORIGIN_BY_KEY[f["product"]]
        st = f["proof_status"]
        if st == "missing":
            return f"{name}: not received; supplier says it will follow"
        if st == "defective":
            return f"{name}: on file, but {f['defect']}"
        if st == "wrong_goods":
            return f"{name}: on file, but it covers {ORIGIN_NEIGHBOUR[g['key']]} only"
        if f["proof_type"] == "supplier_declaration":
            return f"{name}: covers {fd(f['declaration_from'])} to {fd(f['declaration_to'])}"
        return f"{name}: on file, wording as the agreement requires"

    def _routing_text(self, f):
        return {"direct": f"shipped direct from {f['partner']} to the port of entry",
                "tranship_docs": f"transhipped at a port in {f['hub']} after {f['hub_days']} days on the quay; "
                                 f"through bill of lading issued in {f['partner']}",
                "tranship_no_docs": f"transhipped in {f['hub']}; re-consigned there on a new bill of lading, "
                                    f"no documents from the hub",
                "stored": f"stored {f['hub_days']} days in a bonded warehouse in {f['hub']} before onward shipment; "
                          + ("warehouse certificate says the goods were not altered" if f["hub_certificate"]
                             else "no warehouse certificate on file")}[f["routing"]]

    def render(self, f, rng, surface):
        g = ORIGIN_BY_KEY[f["product"]]
        prior = {"none": None, "passed": "one earlier verification of this supplier's goods, passed",
                 "failed": "one earlier verification of this supplier's goods, preference withdrawn"}[f["prior_verification"]]
        if surface == "json":
            fd = _date_fmt(rng)
            return dumps(prune({
                "entry": {"ref": f["entry_ref"], "importer": f["importer"], "goods": g["goods"], "code": f["code"],
                          "quantity": f"{f['quantity']:,} {g['unit']}", "customs_value": f["customs_value"],
                          "currency": f["currency"], "shipped": f["ship_date"].isoformat(),
                          "arrival": f["arrival"].isoformat()},
                "claim": {"agreement": f["agreement"], "exporter": f"{f['supplier']} ({f['partner']})",
                          "standard_rate_pct": f["mfn_rate"], "preferential_rate_pct": f["pref_rate"],
                          "duty_saving": f["duty_saving"],
                          "importer_requests_preference": True if f["importer_insists"] else None},
                "origin_rule": {"product_rule": self._rule_text(f), "supplier_evidence": self._rule_evidence(f)},
                "proof": {"type": f["proof_type"], "detail": self._proof_text(f, fd)},
                "transport": {"route": self._routing_text(f)},
                "history": {"first_shipment_from_supplier": f["first_shipment_from_supplier"],
                            "earlier_verification": prior},
            }))
        m, fd = _money_fmt(rng, f["currency"]), _date_fmt(rng)
        if surface == "origin_worksheet":
            head = kv_lines([
                ("Entry", f["entry_ref"]), ("Importer", f["importer"]), ("Goods", f"{g['goods']} ({f['code']})"),
                ("Exporter", f"{f['supplier']}, {f['partner']}"), ("Agreement claimed", f["agreement"]),
                ("Shipped", fd(f["ship_date"])), ("Arrives", fd(f["arrival"])),
                ("Customs value", m(f["customs_value"])),
                ("Duty", f"standard {f['mfn_rate']}% vs preferential {f['pref_rate']}% "
                         f"(saving {m(f['duty_saving'])})"),
            ], rng)
            checks = table(["check", "finding"], [
                ["product rule", self._rule_text(f)],
                ["rule evidence", self._rule_evidence(f)],
                ["origin proof", self._proof_text(f, fd)],
                ["transport", self._routing_text(f)],
                ["supplier history", ("first shipment from this supplier" if f["first_shipment_from_supplier"]
                                      else "regular supplier") + (f"; {prior}" if prior else "")],
            ], rng, style="markdown")
            note = ("\n\nNote from account manager: importer expects preference as on earlier shipments."
                    if f["importer_insists"] else "")
            return "PREFERENCE CLAIM WORKSHEET\n" + head + "\n\n" + checks + note
        # broker_email: the broker's compliance desk writes to the importer before finalising the entry
        who = f["broker_contact"]
        para = [
            f"We are preparing entry {f['entry_ref']} for your {g['goods']} from {f['supplier']} "
            f"({f['partner']}), shipped {fd(f['ship_date'])}.",
            f"The preference under the {f['agreement']} would take duty from {f['mfn_rate']}% to "
            f"{f['pref_rate']}%, a saving of about {m(f['duty_saving'])} on a customs value of {m(f['customs_value'])}.",
        ]
        para2 = [f"The product rule is {self._rule_text(f)}.",
                 f"What the supplier has given us: {self._rule_evidence(f)}.",
                 f"Origin proof: {self._proof_text(f, fd)}.",
                 f"Routing: {self._routing_text(f)}."]
        extra = []
        if f["first_shipment_from_supplier"]:
            extra.append("This is the first shipment we have handled from this supplier.")
        if prior:
            extra.append(f"Our records show {prior}.")
        if f["importer_insists"]:
            extra.append("We note your request to claim preference as on previous shipments.")
        lines = [f"Subject: {f['entry_ref']} -- preference claim, {f['agreement']}",
                 f"From: {who} (trade compliance)", f"To: imports@{_domain(f['importer'])}", "",
                 "Hello,", "", join_sentences(para), "", noisy(" ".join(para2), rng, 0.1)]
        if extra:
            lines += ["", join_sentences(extra)]
        lines += ["", "We will confirm how the claim is to be declared before the entry is lodged.", "",
                  "Regards,", who]
        return "\n".join(lines)

    def questions(self):
        return {
            "qualifies_for_preference": noul(
                "The goods qualify for the preferential rate under the agreement being claimed.",
                yes="The product rule appears met, the origin proof covers this shipment and the routing keeps the "
                    "goods' originating status.",
                no="The rule looks unmet, the proof does not cover this shipment, or the routing breaks the transport "
                   "condition."),
            "routing_preserves_status": noul(
                "The way the goods travelled keeps them eligible under the agreement's transport condition."),
            "rule_margin": score("How comfortably do the goods satisfy the agreement's product-specific origin rule?", [
                "Not satisfied: the facts on file show the rule is missed.",
                "Unproven: the rule may be met, but the supplier has not shown it.",
                "Narrowly: the rule is met only just, or only by relying on a tolerance or cumulation.",
                "Comfortably: the rule is met with room to spare.",
            ]),
            "verification_exposure": score("If the authority later verified this claim with the exporter, how likely "
                                           "is the preference to be withdrawn?", [
                "Remote: the file would stand up to a verification request.",
                "Low: there are minor gaps that the exporter could probably fill.",
                "Real: a verification could well end with the duty being recovered.",
                "High: the claim would very probably fail on verification.",
            ]),
            "claim_handling": choice("How should the preference claim on this entry be handled?", {
                "claim_now": "Claim the preferential rate now on the evidence already held.",
                "claim_and_refresh_evidence": "Claim the preferential rate now and obtain corrected or renewed "
                                              "evidence from the supplier straight away.",
                "full_duty_then_refund": "Pay the standard rate now and apply for a refund if valid evidence "
                                         "arrives later.",
                "no_claim": "Declare the goods at the standard rate with no claim, because they do not qualify.",
            }),
        }


# =========================================================================== schema 3
DU_ITEMS = [
    {"item": "uncooled thermal imaging camera core", "param": "frame rate", "unit": "Hz", "thr": 25.0,
     "ctrl": "above", "entry": "6A-311", "civil": "roof inspection drones for insurers", "price": 2100.0},
    {"item": "5-axis vertical machining centre", "param": "positioning accuracy", "unit": "µm", "thr": 6.0,
     "ctrl": "at_or_below", "entry": "2B-104", "civil": "mould and die making", "price": 380000.0},
    {"item": "industrial frequency inverter", "param": "maximum output frequency", "unit": "Hz", "thr": 600.0,
     "ctrl": "above", "entry": "3A-227", "civil": "high-speed spindle drives for woodworking machines",
     "price": 4800.0},
    {"item": "carbon-fibre composite tubing", "param": "tensile modulus", "unit": "GPa", "thr": 230.0,
     "ctrl": "above", "entry": "1C-018", "civil": "bicycle frames and sporting goods", "price": 95.0},
    {"item": "benchtop bioreactor system", "param": "vessel volume", "unit": "L", "thr": 20.0, "ctrl": "above",
     "entry": "2B-352", "civil": "cell-culture teaching labs", "price": 31000.0},
    {"item": "high-speed video camera", "param": "maximum frame rate", "unit": "fps", "thr": 40000.0,
     "ctrl": "above", "entry": "6A-203", "civil": "crash-test and materials research", "price": 56000.0},
    {"item": "fixed-wing survey drone", "param": "flight endurance", "unit": "min", "thr": 60.0, "ctrl": "above",
     "entry": "9A-012", "civil": "agricultural mapping", "price": 14500.0},
    {"item": "laser rangefinder module", "param": "maximum range", "unit": "m", "thr": 3000.0, "ctrl": "above",
     "entry": "6A-008", "civil": "forestry survey instruments", "price": 1350.0},
]
DU_END_USERS = {
    "university_lab": "a university research laboratory",
    "oem_manufacturer": "an equipment maker that builds it into its own machines",
    "distributor": "a trading company that resells industrial equipment",
    "state_industrial_group": "a state-owned industrial group",
    "forwarder_only": "not identified; a freight forwarder is named as consignee",
}
DU_FLAGS = [
    "customer asked for the manufacturer's labels to be left off",
    "payment offered in advance from an account in a third country",
    "delivery address is a freight forwarder's warehouse, not the customer's site",
    "order quantity is several times the customer's usual volume",
    "customer declined installation and training",
    "customer was vague when asked what the goods would be used for",
    "routing via a trans-shipment hub requested with no commercial reason",
    "customer asked whether the performance limiter can be removed",
]
DU_EUS = {"signed_specific": "signed, names the site and the specific end use",
          "signed_generic": "signed, but says only '{generic}'",
          "unsigned_draft": "draft received, not yet signed",
          "missing": "not received"}
DU_SCREEN = {"no_match": "no match on any restricted-party list",
             "partial_match_cleared": "partial name match, reviewed and cleared as a different entity",
             "partial_match_open": "partial name match against a restricted-party entry, not yet resolved",
             "not_run": "not yet run"}
DU_LICENCE = {"none": "none", "general_registered": "registered to use the general export authorisation",
              "individual_valid": "individual licence for this customer and item, valid",
              "individual_expired": "individual licence for this customer and item, expired last quarter"}
DU_GROUPS = {"general_eligible": "covered by the general export authorisation",
             "standard": "not covered by the general authorisation",
             "heightened": "on the heightened diversion-risk list"}


def _eus_text(f):
    generic = "research use" if f["end_user_type"] == "university_lab" else "general industrial use"
    return DU_EUS[f["end_user_statement"]].format(generic=generic)


def _du_value(rng, it, pos):
    """The item's figure for its control parameter, placed deliberately relative to the list threshold."""
    thr, above = it["thr"], it["ctrl"] == "above"
    inside = pos in ("near_inside", "clearly_inside")
    if above:
        r = {"clearly_outside": (0.4, 0.75), "near_outside": (0.9, 0.985), "near_inside": (1.015, 1.12),
             "clearly_inside": (1.4, 2.4)}[pos]
    else:   # controlled when the figure is at or better (lower) than the threshold
        r = {"clearly_outside": (1.6, 3.0), "near_outside": (1.03, 1.15), "near_inside": (0.85, 0.99),
             "clearly_inside": (0.35, 0.6)}[pos]
    digits = 0 if thr >= 100 else (1 if thr >= 10 else 2)
    step = 10 ** -digits
    v = round(thr * rng.uniform(*r), digits)
    for _ in range(50):
        ctrl = v > thr if above else v <= thr
        if ctrl == inside:
            break
        v = round(v + (step if above == inside else -step), digits)
    return int(v) if digits == 0 else v


def _num(x):
    return f"{int(x):,}" if float(x).is_integer() else f"{x:g}"


class DualUseExport(Schema):
    name = "dual_use_export"
    description = ("An export order for an item that may be on a dual-use control list: licence position, "
                   "end-user assurance and whether the shipment can go.")
    surfaces = {"json": 0.45, "compliance_check": 0.3, "sales_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        it = pick(rng, DU_ITEMS)
        pos = pick_w(rng, {"clearly_outside": 2, "near_outside": 3, "near_inside": 3, "clearly_inside": 2})
        spec_src = pick_w(rng, {"manufacturer_datasheet": 5, "sales_brochure": 3, None: 1})
        maker_view = pick_w(rng, {"consistent": 6, "says_not_listed": 2, None: 2})
        group = pick_w(rng, {"general_eligible": 3, "standard": 4, "heightened": 2})
        eu = pick_w(rng, {"university_lab": 2, "oem_manufacturer": 3, "distributor": 3, "state_industrial_group": 1,
                          "forwarder_only": 1})
        eus = pick_w(rng, {"signed_specific": 4, "signed_generic": 3, "unsigned_draft": 2, "missing": 1})
        screen = pick_w(rng, {"no_match": 6, "partial_match_cleared": 2, "partial_match_open": 1, "not_run": 1})
        n_flags = pick(rng, [0, 0, 1, 1, 2])
        licence = pick_w(rng, {"none": 5, "general_registered": 3, "individual_valid": 1, "individual_expired": 1})
        years = pick(rng, [0, 0, 1, 3, 6, 11])
        if reg == "clear":
            if chance(rng, 0.5):
                pos, spec_src, maker_view, group = "clearly_outside", "manufacturer_datasheet", "consistent", pick(
                    rng, ["general_eligible", "standard"])
                eu, eus, screen, n_flags, years = pick(rng, ["oem_manufacturer", "university_lab"]), "signed_specific", \
                    "no_match", 0, pick(rng, [4, 7, 12])
            else:
                pos, group, eus, n_flags, years = "clearly_inside", "heightened", "missing", pick(rng, [2, 3]), 0
                eu, screen, licence = pick(rng, ["distributor", "forwarder_only"]), pick(
                    rng, ["partial_match_open", "not_run"]), "none"
                maker_view = pick(rng, ["consistent", None])
        elif reg == "borderline":
            pos = pick(rng, ["near_outside", "near_inside"])
            n_flags = pick(rng, [0, 1, 1])
            eus = pick_w(rng, {"signed_specific": 2, "signed_generic": 3, "unsigned_draft": 1})
        else:
            variant = pick(rng, ["unlisted_but_suspicious", "trusted_customer_listed_item", "maker_disagrees"])
            if variant == "unlisted_but_suspicious":
                pos, n_flags, eus = "clearly_outside", pick(rng, [2, 3]), pick(rng, ["signed_generic", "missing"])
            elif variant == "trusted_customer_listed_item":
                pos, years, eus, screen, n_flags = pick(rng, ["near_inside", "clearly_inside"]), pick(rng, [6, 11]), \
                    "signed_specific", "no_match", pick(rng, [0, 1])
                licence = pick(rng, ["individual_expired", "none", "general_registered"])
            else:
                pos, maker_view, spec_src = pick(rng, ["near_inside", "clearly_inside"]), "says_not_listed", \
                    "manufacturer_datasheet"
        value = _du_value(rng, it, pos) if spec_src else None
        inside = pos in ("near_inside", "clearly_inside")
        maker = None if maker_view is None else ("not listed" if maker_view == "says_not_listed" or not inside
                                                 else f"listed under {it['entry']}")
        flags = rng.sample(DU_FLAGS, n_flags)
        if eu == "forwarder_only" and DU_FLAGS[2] not in flags and chance(rng, 0.5):
            flags = flags + [DU_FLAGS[2]]
        qty = max(1, int(round(pick(rng, [15000, 60000, 180000, 500000]) / it["price"])))
        cur = pick_w(rng, CURRENCIES)
        eu_role = {"university_lab": "lab", "oem_manufacturer": "maker", "distributor": "trader",
                   "state_industrial_group": "state_group", "forwarder_only": "trader"}[eu]
        orgs = _names(rng, eu_role, "forwarder", "maker")
        if eu == "forwarder_only":
            consignee = f"{orgs[1]} (freight forwarder)"
        elif chance(rng, 0.65):
            consignee = orgs[0]
        else:
            consignee = f"{orgs[1]} (freight forwarder, for the end user)"
        return {
            "regime": reg, "item": it["item"], "tech_position": pos, "spec_value": value, "spec_source": spec_src,
            "maker_classification": maker, "destination": _countries(rng, 1)[0], "destination_group": group,
            "end_user_type": eu, "end_user_org": orgs[0] if eu != "forwarder_only" else None,
            "consignee": consignee,
            "stated_end_use": (it["civil"] if eus == "signed_specific" or chance(rng, 0.5) else None),
            "end_user_statement": eus, "screening": screen, "red_flags": flags, "licence_on_file": licence,
            "customer_years": years, "quantity": qty, "currency": cur,
            "order_value": round(qty * it["price"] * rng.uniform(0.9, 1.1), 2),
            "order_ref": ident(rng, "SO", 6, "-"), "check_ref": ident(rng, "EXC", 6, "-"),
            "sales_rep": person(rng), "exporter": orgs[2],
            "ship_by": rand_date(rng), "sales_pressure": chance(rng, 0.5),
        }

    def _item(self, f):
        return next(i for i in DU_ITEMS if i["item"] == f["item"])

    def _scope(self, it):
        if it["ctrl"] == "above":
            return f"{it['param']} above {_num(it['thr'])} {it['unit']}"
        return f"{it['param']} of {_num(it['thr'])} {it['unit']} or better (lower)"

    def render(self, f, rng, surface):
        it = self._item(f)
        spec = f"{_num(f['spec_value'])} {it['unit']}" if f["spec_value"] is not None else None
        customer = ("new customer" if f["customer_years"] == 0 else "customer for 1 year" if f["customer_years"] == 1
                    else f"customer for {f['customer_years']} years")
        if surface == "json":
            return dumps(prune({
                "order": {"ref": f["order_ref"], "exporter": f["exporter"], "item": it["item"],
                          "quantity": f["quantity"], "value": f["order_value"], "currency": f["currency"],
                          "requested_ship_date": f["ship_by"].isoformat()},
                "classification": {"candidate_entry": it["entry"], "entry_covers": self._scope(it),
                                   f"item_{it['param'].replace(' ', '_')}": spec,
                                   "spec_source": _h(f["spec_source"]) if f["spec_source"] else None,
                                   "manufacturer_classification": f["maker_classification"]},
                "parties": {"destination": f["destination"], "destination_status": DU_GROUPS[f["destination_group"]],
                            "consignee": f["consignee"], "end_user": f["end_user_org"],
                            "end_user_profile": DU_END_USERS[f["end_user_type"]],
                            "stated_end_use": f["stated_end_use"], "relationship": customer},
                "checks": {"restricted_party_screening": DU_SCREEN[f["screening"]],
                           "end_user_statement": _eus_text(f),
                           "red_flags": f["red_flags"] or None},
                "licensing": {"on_file": DU_LICENCE[f["licence_on_file"]]},
            }))
        m, fd = _money_fmt(rng, f["currency"]), _date_fmt(rng)
        if surface == "compliance_check":
            ts = rand_ts(rng, day=f["ship_by"] - timedelta(days=rng.randint(2, 12)), hours=(8, 18))
            head = kv_lines([
                ("Order", f"{f['order_ref']}: {f['quantity']:,} x {it['item']}, {m(f['order_value'])}"),
                ("Destination", f"{f['destination']} ({DU_GROUPS[f['destination_group']]})"),
                ("Consignee", f["consignee"]),
                ("End user", (f["end_user_org"] + ", " if f["end_user_org"] else "") + DU_END_USERS[f["end_user_type"]]),
                ("Stated end use", f["stated_end_use"]), ("Relationship", customer),
                ("Ship by", fd(f["ship_by"])),
            ], rng, style="aligned", skip_none=False)
            cls = [f"candidate entry {it['entry']}: {self._scope(it)}",
                   f"item figure: {show(spec, rng)}" + (f" (source: {_h(f['spec_source'])})" if f["spec_source"] else ""),
                   f"manufacturer's classification: {show(f['maker_classification'], rng)}"]
            out = [f"EXPORT COMPLIANCE CHECK {f['check_ref']}   run {fmt_ts(ts, rng)}", head, "",
                   "[CLASSIFICATION]", *("  " + c for c in cls), "",
                   f"[SCREENING]  {DU_SCREEN[f['screening']]}",
                   f"[END-USE]    statement {_eus_text(f)}",
                   f"[LICENCES]   {DU_LICENCE[f['licence_on_file']]}"]
            if f["red_flags"]:
                out += ["[FLAGS]", bullets(f["red_flags"], rng, marker="  -")]
            else:
                out.append("[FLAGS]      none raised by sales")
            return "\n".join(out)
        # sales_email: the account manager hands the order to the export-control desk
        rep = f["sales_rep"]
        s = [f"New order {f['order_ref']} from "
             + (f"{f['end_user_org']}" if f["end_user_org"] else "a buyer shipping via " + f["consignee"])
             + f" in {f['destination']}: {f['quantity']:,} x {it['item']}, {m(f['order_value'])}."]
        s.append("They're a new customer." if f["customer_years"] == 0
                 else "We've sold to them for a year." if f["customer_years"] == 1
                 else f"We've sold to them for {f['customer_years']} years.")
        s.append(f"They say it's for {f['stated_end_use']}." if f["stated_end_use"]
                 else "They haven't said much about the application.")
        s.append({"signed_specific": "Signed end-user statement attached.",
                  "signed_generic": f"They signed the end-user form but only wrote "
                                    f"'{'research use' if f['end_user_type'] == 'university_lab' else 'general industrial use'}'.",
                  "unsigned_draft": "End-user form is with them, I have a draft but no signature yet.",
                  "missing": "I haven't asked for an end-user statement yet."}[f["end_user_statement"]])
        if spec:
            s.append(f"Datasheet figure for {it['param']} is {spec}" + (" (from the brochure)." if f["spec_source"] == "sales_brochure" else "."))
        if f["maker_classification"]:
            s.append(f"The manufacturer's export sheet says {f['maker_classification']}.")
        if len(f["red_flags"]) == 1:
            s.append(f"One thing to mention: {f['red_flags'][0]}.")
        elif f["red_flags"]:
            s.append("A few things you should know: " + "; ".join(f["red_flags"]) + ".")
        if f["sales_pressure"]:
            s.append(f"They need it shipped by {fd(f['ship_by'])} or we risk losing the order.")
        foot = kv_lines([("Screening", DU_SCREEN[f["screening"]]), ("Licence on file", DU_LICENCE[f["licence_on_file"]]),
                         ("Candidate entry", f"{it['entry']} ({self._scope(it)})")], rng, style="dash")
        return "\n".join([f"Subject: {f['order_ref']} -- export check please",
                          f"From: {rep}", "To: export-control desk", "", "Hi team,", "",
                          noisy(" ".join(s), rng, 0.15), "", "Auto-filled by order system:", foot, "",
                          "Thanks,", rep.split()[0]])

    def questions(self):
        return {
            "licence_position": choice("What is the export licensing position for this order?", {
                "not_controlled": "The item falls outside the controlled entries, so no export licence is needed.",
                "general_authorisation": "The item is controlled, but the general export authorisation covers this "
                                         "destination and it is registered.",
                "individual_licence_needed": "The item is controlled and this export needs its own individual licence.",
                "classification_unresolved": "The technical information is not good enough to say whether the item "
                                             "is controlled.",
            }),
            "shipment_decision": choice("What should the export desk do with the shipment now?", {
                "release_now": "Let the shipment go on the current paperwork.",
                "complete_file_then_ship": "Ship once missing paperwork such as a signed end-user statement or a "
                                           "screening result is in, with no licence application.",
                "hold_for_licence": "Hold the shipment and apply for an export licence.",
                "escalate_to_export_control_officer": "Hold the shipment and put the case to the company's export "
                                                      "control officer to decide.",
                "decline_order": "Refuse the order and consider reporting the enquiry to the licensing authority.",
            }),
            "diversion_signal_strength": score("How strong are the signs that the goods could end up with an "
                                               "unauthorised user or use?", [
                "None: nothing in the file points away from the stated customer and use.",
                "Faint: one oddity with an innocent explanation.",
                "Clear: several warning signs that need answers before shipping.",
                "Strong: the pattern of signs is typical of attempts to divert goods.",
            ]),
            "end_user_assurance": score("How much comfort does the end-user evidence give that the goods will be "
                                        "used as stated?", [
                "None: the real end user or end use is unknown.",
                "Thin: a named buyer, but the end use is vague or unsigned.",
                "Reasonable: a signed, specific statement from a plausible user.",
                "Strong: a specific statement backed by a long, clean trading relationship.",
            ]),
            "catch_all_concern": noul(
                "The exporter has grounds to suspect a sensitive end use even if the item is not on a control list.",
                yes="Warning signs or end-user facts would make a careful exporter suspect a military or weapons-related "
                    "end use.",
                no="Nothing in the file suggests an end use beyond the stated civilian one."),
        }


# =========================================================================== schema 4
PCA_FINDINGS = {
    "misclassification": ("a product family was entered under a heading carrying a lower duty rate than the one "
                          "that applies", "classification"),
    "royalty_omitted": ("licence fees paid to the brand owner were left out of the customs value", "valuation"),
    "assist_omitted": ("moulds supplied free of charge to the overseas factory were not added to the customs value",
                       "valuation"),
    "origin_unsupported": ("preferential duty was claimed on entries with no valid supplier declaration on file",
                           "origin claims"),
    "exchange_rate": ("invoice prices were converted at an out-of-date exchange rate", "valuation"),
    "freight_omitted": ("inbound freight was left out of the customs value on ex-works purchases", "valuation"),
}
PCA_CAUSES = {
    "keying_slip": "a keying error by one entry clerk",
    "stale_master_data": "an out-of-date setting in the importer's master data, applied automatically to every "
                         "affected entry",
    "misread_guidance": "the team read the published guidance differently from the authority's current view",
    "supplier_data": "the team relied on the overseas supplier's paperwork without checking it",
    "unexplained": "no explanation found yet",
}
PCA_INTENT = {
    "second_invoice": "a second invoice for the same goods at a higher price was found in supplier correspondence",
    "split_fee_email": "an email from the buying team asks the supplier to bill part of the price separately "
                       "'to keep the entry value down'",
    "ignored_warning": "the broker flagged the same issue in writing fourteen months ago and nothing was changed",
}
PCA_CONTROLS = {
    "documented_and_trained": "written procedures, annual training and a monthly entry check",
    "documented_outdated": "written procedures last updated four years ago; no entry checks",
    "informal": "no written procedures; relies on one experienced clerk",
    "none": "no procedures and no review of broker entries",
}
PCA_OTHER_TOPIC = {"classification": "valuation", "valuation": "origin claims", "origin claims": "classification"}


class PostClearanceAudit(Schema):
    name = "post_clearance_audit"
    description = ("An internal post-clearance audit finding at an importer: how culpable the error looks, and "
                   "whether to correct, disclose or dig further.")
    surfaces = {"json": 0.45, "audit_memo": 0.3, "workpaper": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        finding = pick(rng, list(PCA_FINDINGS))
        tested = pick(rng, [30, 40, 60, 80, 120])
        rate = near(rng, 0.18, 0.12, lo=0.02, hi=0.6)
        cause = pick_w(rng, {"keying_slip": 2, "stale_master_data": 3, "misread_guidance": 2, "supplier_data": 2,
                             "unexplained": 1})
        intent = pick_w(rng, {None: 14, "ignored_warning": 3, "second_invoice": 1, "split_fee_email": 1})
        contact = pick_w(rng, {"none": 4, "unrelated_audit": 3, "covers_topic": 2, "query_on_entries": 2})
        controls = pick_w(rng, {"documented_and_trained": 2, "documented_outdated": 3, "informal": 3, "none": 1})
        corrected = pick_w(rng, {"yes": 3, "in_progress": 3, "no": 2})
        other_lines = pick_w(rng, {"clean": 3, "similar_pattern": 2, None: 3})
        limit = pick(rng, [5000, 10000, 20000])
        extrap_target = None
        if reg == "clear":
            if chance(rng, 0.5):
                rate, cause, intent, contact = between(rng, 0.02, 0.05), "keying_slip", None, "none"
                controls, corrected, other_lines = "documented_and_trained", "yes", "clean"
                extrap_target = limit * rng.uniform(0.05, 0.3)
            else:
                rate, intent, controls, corrected = between(rng, 0.35, 0.6), pick(
                    rng, ["second_invoice", "split_fee_email"]), pick(rng, ["informal", "none"]), "no"
                cause, extrap_target = "unexplained", limit * rng.uniform(8, 25)
        elif reg == "borderline":
            cause = pick(rng, ["stale_master_data", "misread_guidance", "supplier_data"])
            extrap_target = near(rng, limit, limit * 0.35, lo=limit * 0.4)
            contact = pick_w(rng, {"none": 3, "unrelated_audit": 3, "covers_topic": 2})
        else:
            variant = pick(rng, ["good_controls_bad_email", "innocent_but_audit_open", "few_errors_big_money"])
            if variant == "good_controls_bad_email":
                controls, intent, cause = "documented_and_trained", pick(rng, ["split_fee_email", "second_invoice"]), \
                    "supplier_data"
            elif variant == "innocent_but_audit_open":
                cause, intent, contact = pick(rng, ["keying_slip", "misread_guidance"]), None, pick(
                    rng, ["covers_topic", "query_on_entries"])
            else:
                rate, cause = between(rng, 0.02, 0.06), "stale_master_data"
                extrap_target = limit * rng.uniform(3, 12)
                other_lines = None
        if cause == "keying_slip":
            rate = between(rng, 0.01, 0.05)     # one clerk's slip touches a handful of entries, not a quarter
        errors = max(1, round(tested * rate))
        period = pick(rng, [12, 18, 24, 36])
        in_period = int(round(tested * rng.uniform(5, 30), -1))
        avg = lognormal_amount(rng, 700, 0.6)
        if extrap_target is not None:          # hit a chosen total, keeping sample and extrapolation consistent
            avg = max(4.0, round(extrap_target * tested / (in_period * errors), 2))
        short_sample = round(errors * avg, 2)
        extrap = int(round(short_sample * in_period / tested, -2))
        return {
            "regime": reg, "finding": finding, "topic": PCA_FINDINGS[finding][1], "entries_tested": tested,
            "entries_in_error": errors, "entries_in_period": in_period, "period_months": period,
            "shortfall_in_sample": short_sample, "shortfall_extrapolated": extrap, "correction_limit": limit,
            "annual_duty_paid": int(round(max(extrap * 12 / period * rng.uniform(4, 40), 20000), -3)),
            "cause": cause, "intent_indicator": intent, "authority_contact": contact, "controls": controls,
            "corrected_going_forward": corrected, "other_product_lines": other_lines,
            "months_to_limitation": pick(rng, [2, 5, 9, 14, 26]), "prior_disclosures": pick(rng, [0, 0, 0, 1, 2]),
            "prior_penalty": chance(rng, 0.12), "discovered_by": pick(rng, ["internal audit", "the customs broker",
                                                                            "the new trade compliance manager"]),
            "importer": _names(rng, "importer")[0], "auditor": person(rng), "head_of_compliance": person(rng),
            "memo_date": rand_date(rng), "currency": pick_w(rng, CURRENCIES), "ref": ident(rng, "PCA", 4, "-"),
        }

    def _contact_text(self, f):
        return {"none": "no contact from the customs authority on this subject",
                "unrelated_audit": f"the authority has announced an audit of the importer's "
                                   f"{PCA_OTHER_TOPIC[f['topic']]} for the same period (it does not cover {f['topic']})",
                "covers_topic": f"the authority's audit notice received last month covers {f['topic']}",
                "query_on_entries": "a customs officer has asked questions about two of the affected entries"}[
            f["authority_contact"]]

    def render(self, f, rng, surface):
        what = PCA_FINDINGS[f["finding"]][0]
        other = {"clean": "spot checks of other product lines found nothing similar",
                 "similar_pattern": "a quick look at two other product lines suggests the same pattern", None: None}[
            f["other_product_lines"]]
        if surface == "json":
            return dumps(prune({
                "finding": {"ref": f["ref"], "summary": what, "topic": f["topic"], "found_by": f["discovered_by"],
                            "cause_so_far": PCA_CAUSES[f["cause"]], "other_evidence": PCA_INTENT.get(f["intent_indicator"])},
                "sample": {"entries_tested": f["entries_tested"], "entries_in_error": f["entries_in_error"],
                           "period_months": f["period_months"], "entries_in_period": f["entries_in_period"],
                           "shortfall_in_sample": f["shortfall_in_sample"],
                           "shortfall_extrapolated": f["shortfall_extrapolated"], "currency": f["currency"],
                           "other_product_lines": other},
                "importer": {"name": f["importer"], "annual_duty_paid": f["annual_duty_paid"],
                             "controls": PCA_CONTROLS[f["controls"]],
                             "corrected_for_new_entries": f["corrected_going_forward"],
                             "earlier_disclosures": f["prior_disclosures"],
                             "earlier_penalty": True if f["prior_penalty"] else None},
                "authority": {"contact": self._contact_text(f),
                              "routine_amendment_limit": f["correction_limit"],
                              "months_until_oldest_entries_time_barred": f["months_to_limitation"]},
            }))
        m, fd = _money_fmt(rng, f["currency"]), _date_fmt(rng)
        if surface == "audit_memo":
            para1 = [f"During a review of the last {f['period_months']} months of entries, {f['discovered_by']} found "
                     f"that {what}.",
                     f"{f['entries_in_error']} of {f['entries_tested']} sampled entries are affected, with "
                     f"{m(f['shortfall_in_sample'])} of duty underpaid in the sample.",
                     f"Straight-line extrapolation over the {f['entries_in_period']:,} entries in the period gives "
                     f"about {m(f['shortfall_extrapolated'])}.", (other[0].upper() + other[1:] + ".") if other else ""]
            para2 = [f"Cause so far: {PCA_CAUSES[f['cause']]}.",
                     (PCA_INTENT[f["intent_indicator"]][0].upper() + PCA_INTENT[f["intent_indicator"]][1:] + ".")
                     if f["intent_indicator"] else "",
                     f"Controls: {PCA_CONTROLS[f['controls']]}.",
                     {"yes": "New entries have been corrected.", "in_progress": "The fix for new entries is in progress.",
                      "no": "New entries are still being made the same way."}[f["corrected_going_forward"]]]
            para3 = [f"Authority: {self._contact_text(f)}.",
                     f"The authority accepts routine amendments up to {m(f['correction_limit'])} in total.",
                     f"The oldest affected entries become time-barred in {f['months_to_limitation']} months.",
                     f"Earlier voluntary disclosures: {f['prior_disclosures']}"
                     + ("; one earlier penalty." if f["prior_penalty"] else ".")]
            return "\n".join([f"MEMO {f['ref']}   {fd(f['memo_date'])}", f"To: {f['head_of_compliance']}, head of trade compliance",
                              f"From: {f['auditor']}", f"Re: post-clearance review, {f['importer']}", "",
                              noisy(join_sentences(para1), rng, 0.1), "", noisy(join_sentences(para2), rng, 0.1), "",
                              join_sentences(para3)])
        # workpaper: sampled error lines plus a summary block
        shown = min(f["entries_in_error"], pick(rng, [3, 4]))
        base = f["memo_date"] - timedelta(days=30 * f["period_months"])
        shares = [rng.uniform(0.5, 1.5) for _ in range(f["entries_in_error"])]
        tot = sum(shares)
        rows = []
        for i in range(shown):
            d = base + timedelta(days=rng.randint(0, 30 * f["period_months"] - 1))
            rows.append([ident(rng, "E", 7, ""), fmt_date(d, style="iso"),
                         m(round(f["shortfall_in_sample"] * shares[i] / tot, 2))])
        rows.sort(key=lambda r: r[1])
        if f["entries_in_error"] > shown:
            rows.append([f"(+{f['entries_in_error'] - shown} more)", "", ""])
        summary = kv_lines([
            ("Finding", what), ("Sample", f"{f['entries_in_error']} of {f['entries_tested']} entries in error"),
            ("Sample shortfall", m(f["shortfall_in_sample"])),
            ("Extrapolated shortfall", f"{m(f['shortfall_extrapolated'])} over {f['entries_in_period']:,} entries in "
                                       f"{f['period_months']} months"),
            ("Annual duty paid", m(f["annual_duty_paid"])), ("Cause", PCA_CAUSES[f["cause"]]),
            ("Other evidence", PCA_INTENT.get(f["intent_indicator"])), ("Other product lines", other),
            ("Controls", PCA_CONTROLS[f["controls"]]), ("Fixed for new entries", _h(f["corrected_going_forward"])),
            ("Authority contact", self._contact_text(f)), ("Routine amendment limit", m(f["correction_limit"])),
            ("Oldest entries time-barred in", f"{f['months_to_limitation']} months"),
            ("Earlier disclosures", f["prior_disclosures"]), ("Earlier penalty", _yn(f["prior_penalty"])),
        ], rng, skip_none=pick(rng, [True, False]))
        return (f"WORKPAPER {f['ref']} -- {f['importer']} -- prepared by {initials(f['auditor'])}\n\n"
                + table(["entry", "date", "duty short"], rows, rng) + "\n\n" + summary)

    def questions(self):
        return {
            "resolution_route": choice("What should the importer's trade-compliance team do about this finding?", {
                "record_only": "Record the finding as immaterial and take no further action.",
                "fix_forward": "Fix the process for future entries and document it, without filing anything on past "
                               "entries.",
                "amend_past_entries": "File routine amendments on the affected entries and pay the shortfall.",
                "voluntary_disclosure": "Make a formal voluntary disclosure to the customs authority covering the "
                                        "whole affected period.",
                "engage_counsel": "Bring in outside customs counsel before any contact with the authority, because "
                                  "the facts may point to intent.",
            }),
            "culpability_level": score("How culpable does the importer's conduct look on the facts gathered so far?", [
                "Blameless: an error the importer could not reasonably have caught.",
                "Clerical: an isolated slip in keying or data entry.",
                "Careless: the importer did not take reasonable care.",
                "Grossly careless: known obligations or warnings were plainly disregarded.",
                "Deliberate: there are signs the misstatement was intended.",
            ]),
            "control_weakness": score("How weak are the importer's controls over the data that caused this error?", [
                "Sound: the error slipped through controls that generally work.",
                "Patchy: controls exist but are out of date or not applied consistently.",
                "Weak: controls depend on individuals and nobody checks the entries.",
                "Absent: nothing stands between the source data and the entries lodged.",
            ]),
            "disclosure_still_voluntary": noul(
                "A disclosure filed now would still count as voluntary.",
                yes="The authority has not yet opened an audit or query that covers this issue, so the importer "
                    "would be coming forward first.",
                no="The authority has already opened an audit or query that reaches this issue."),
            "widen_review": noul(
                "The review should be extended beyond the period and product lines sampled so far.",
                yes="Other product lines show the same pattern, or they have not been checked and the cause (master "
                    "data, a reading of guidance, supplier paperwork, or still unknown) would reach beyond the sample.",
                no="The cause is one clerk's keying slip, or spot checks of other product lines came back clean."),
        }


def initials(name):
    return "".join(w[0] for w in name.split()).upper()


# =========================================================================== schema 5
ECOM_CATEGORIES = ["phone accessories", "cosmetics and skincare", "children's toys", "fast-fashion apparel",
                   "small electronics", "sports supplements", "vaping accessories", "car parts"]
ECOM_SAFETY = {"children's toys": "toys without the conformity marking",
               "small electronics": "mains chargers without the conformity marking",
               "phone accessories": "power banks without battery test reports",
               "cosmetics and skincare": "creams listing a banned colourant",
               "sports supplements": "supplements naming a prohibited stimulant",
               "vaping accessories": "e-liquid above the nicotine limit",
               "car parts": "brake pads without approval marks",
               "fast-fashion apparel": "children's hoodies with drawcords"}


class EcommerceLowValueBatch(Schema):
    name = "ecommerce_low_value_batch"
    description = ("A batch of low-value e-commerce parcels from one seller at an express hub: release, check, "
                   "hold or refer, and whether values and splitting look genuine.")
    surfaces = {"json": 0.45, "risk_engine_summary": 0.35, "hub_shift_log": 0.2}

    def sample_factors(self, rng):
        reg = regime(rng)
        parcels = pick(rng, [120, 240, 480, 900, 1600])
        under_share = near(rng, 0.25, 0.12, lo=0.02, hi=0.8)
        ratio = near(rng, 0.72, 0.14, lo=0.25, hi=1.05)
        vague = near(rng, 0.2, 0.12, lo=0.0, hi=0.8)
        repeat_max = pick(rng, [1, 2, 2, 3, 4, 6])
        ipr = pick(rng, [0, 0, 0, 0, 2, 5])
        safety = pick(rng, [0, 0, 0, 0, 1, 3])
        xray = pick_w(rng, {"consistent": 5, "mismatch": 2, None: 2})
        scheme = pick_w(rng, {"registered": 5, "not_registered": 3, None: 1})
        seizures = pick(rng, [0, 0, 0, 1, 2])
        warning = chance(rng, 0.2)
        if reg == "clear":
            if chance(rng, 0.5):
                under_share, ratio, vague, repeat_max = between(rng, 0.02, 0.07), between(rng, 0.9, 1.02), \
                    between(rng, 0.0, 0.06), pick(rng, [1, 2])
                ipr, safety, xray, scheme, seizures, warning = 0, 0, "consistent", "registered", 0, False
            else:
                under_share, ratio, vague, repeat_max = between(rng, 0.5, 0.75), between(rng, 0.28, 0.5), \
                    between(rng, 0.4, 0.7), pick(rng, [6, 8, 9])
                xray, scheme, seizures, warning = "mismatch", "not_registered", pick(rng, [3, 4, 5]), True
                ipr = pick(rng, [0, 6, 14])
        elif reg == "conflicting":
            variant = pick(rng, ["clean_seller_low_values", "bad_history_clean_batch", "brand_hits_fair_values"])
            if variant == "clean_seller_low_values":
                scheme, seizures, warning = "registered", 0, False
                ratio, under_share, repeat_max = between(rng, 0.38, 0.55), between(rng, 0.35, 0.55), pick(rng, [4, 6])
            elif variant == "bad_history_clean_batch":
                seizures, warning = pick(rng, [3, 4]), True
                ratio, under_share, vague, xray = between(rng, 0.85, 1.0), between(rng, 0.03, 0.1), \
                    between(rng, 0.02, 0.1), "consistent"
            else:
                ipr, ratio, under_share = pick(rng, [9, 14, 22]), between(rng, 0.85, 1.0), between(rng, 0.05, 0.15)
            if chance(rng, 0.5):
                xray = None
        cat = pick(rng, ECOM_CATEGORIES)
        thr = pick(rng, [100, 150, 200])
        n_under = int(round(parcels * under_share))
        n_vague = int(round(parcels * vague))
        multi = 0 if repeat_max < 3 else max(1, int(round(parcels * rng.uniform(0.004, 0.03))))
        sample_n = pick(rng, [10, 20, 30])
        xray_n = pick(rng, [20, 30, 40])
        median_decl = round(thr * rng.uniform(0.25, 0.7), 2)
        # share of the price sample below 70% of listing, if per-parcel ratios spread ~0.15 around the median
        below = 0.5 * (1 + math.erf((0.7 - ratio) / (0.15 * math.sqrt(2))))
        n_below = max(0, min(sample_n, int(round(sample_n * below + rng.uniform(-1.0, 1.0)))))
        orgs = _names(rng, "shop", "courier")
        return {
            "regime": reg, "category": cat, "parcels": parcels, "threshold": thr, "just_under_threshold": n_under,
            "declared_to_listing_ratio": round(ratio, 2), "price_sample_n": sample_n,
            "price_sample_below_70pct": n_below,
            "vague_descriptions": n_vague, "max_parcels_one_consignee": repeat_max,
            "consignees_with_3plus": multi, "ipr_hits": ipr, "safety_hits": safety if cat in ECOM_SAFETY else 0,
            "xray_result": xray, "xray_sample": xray_n,
            "xray_mismatches": rng.randint(2, 7) if xray == "mismatch" else 0,
            "vat_scheme": scheme, "seizures_12m": seizures, "warning_letter": warning,
            "median_declared": median_decl, "declared_total": round(median_decl * parcels * rng.uniform(1.0, 1.25), 2),
            "currency": pick_w(rng, CURRENCIES), "seller": orgs[0], "origin": _countries(rng, 1)[0],
            "courier": orgs[1], "batch_id": ident(rng, "MB", 8, ""),
            "hub": f"{pick(rng, ['North', 'East', 'Airport', 'Riverside', 'Central'])} express hub",
            "arrived": rand_ts(rng, hours=(0, 6)), "officer": person(rng),
        }

    def render(self, f, rng, surface):
        m = _money_fmt(rng, f["currency"])
        vat = {"registered": "registered", "not_registered": "not registered", None: None}[f["vat_scheme"]]
        price_line = (f"{f['price_sample_below_70pct']} of {f['price_sample_n']} sampled parcels declared below 70% "
                      f"of the platform listing; median declared/listing ratio {f['declared_to_listing_ratio']}")
        xray = (None if f["xray_result"] is None else
                f"{f['xray_mismatches']} of {f['xray_sample']} images inconsistent with declared contents"
                if f["xray_result"] == "mismatch" else f"{f['xray_sample']} images consistent with declared contents")
        history = ({0: "no seizures", 1: "1 seizure"}.get(f["seizures_12m"], f"{f['seizures_12m']} seizures")
                   + " in 12 months" + ("; warning letter issued" if f["warning_letter"] else ""))
        safety_txt = ECOM_SAFETY.get(f["category"], "unmarked goods")
        if surface == "json":
            return dumps(prune({
                "batch": {"id": f["batch_id"], "hub": f["hub"], "courier": f["courier"],
                          "arrived": fmt_ts(f["arrived"], style="iso"), "parcels": f["parcels"],
                          "goods": f["category"], "declared_total": f["declared_total"], "currency": f["currency"],
                          "median_declared_value": f["median_declared"], "duty_free_threshold": f["threshold"]},
                "seller": {"name": f["seller"], "ships_from": f["origin"], "import_vat_scheme": vat,
                           "history": history},
                "risk_hits": {"declared_85_to_100pct_of_threshold": f["just_under_threshold"],
                              "generic_descriptions": f["vague_descriptions"],
                              "max_parcels_to_one_consignee_same_day": f["max_parcels_one_consignee"],
                              "consignees_with_3_or_more_parcels": f["consignees_with_3plus"],
                              "brand_watchlist": f["ipr_hits"],
                              "product_safety": f"{f['safety_hits']} ({safety_txt})" if f["safety_hits"] else 0},
                "checks": {"price_sample": price_line, "xray": xray},
            }))
        if surface == "risk_engine_summary":
            rows = [["R-VAL-07 declared 85-100% of threshold", f"{f['just_under_threshold']} "
                                                             f"({round(100 * f['just_under_threshold'] / f['parcels'])}%)"],
                    ["R-VAL-12 below 70% of listing price", f"{f['price_sample_below_70pct']}/{f['price_sample_n']} sampled"],
                    ["R-DSC-03 generic description", str(f["vague_descriptions"])],
                    ["R-CNS-02 3+ parcels, one consignee, one day",
                     f"{f['consignees_with_3plus']} consignee(s), max {f['max_parcels_one_consignee']}"],
                    ["R-IPR-11 brand watchlist", str(f["ipr_hits"])],
                    ["R-SAF-04 " + safety_txt, str(f["safety_hits"])]]
            head = (f"RISK ENGINE  batch {f['batch_id']}  {f['courier']} / {f['hub']}  {fmt_ts(f['arrived'], rng)}\n"
                    f"Seller: {f['seller']} ({f['origin']})   import-VAT scheme: {show(vat, rng)}\n"
                    f"Parcels: {f['parcels']:,}   goods: {f['category']}   declared total: {m(f['declared_total'])}   "
                    f"threshold: {m(f['threshold'])}")
            tail = [f"Seller history: {history}", f"X-ray: {show(xray, rng)}",
                    f"Median declared value/listing ratio: {f['declared_to_listing_ratio']}"]
            return head + "\n\n" + table(["rule", "hits"], rows, rng, style="fixed") + "\n\n" + "\n".join(tail)
        # hub_shift_log
        t = f["arrived"]
        ev = [f"manifest {f['batch_id']} from {f['courier']}: {f['parcels']:,} parcels of {f['category']}, seller "
              f"{f['seller']} ({f['origin']}), declared {m(f['declared_total'])}",
              f"{f['just_under_threshold']} parcels declared between 85% and 100% of the {m(f['threshold'])} threshold",
              f"price check: {price_line}",
              f"{f['vague_descriptions']} lines say only 'gift', 'accessories' or 'parts'",
              (f"{f['consignees_with_3plus']} consignee(s) receiving 3+ parcels today, max "
               f"{f['max_parcels_one_consignee']} to one address") if f["consignees_with_3plus"]
              else f"no consignee receives more than {f['max_parcels_one_consignee']} parcels",
              f"brand watchlist hits: {f['ipr_hits']}; safety hits ({safety_txt}): {f['safety_hits']}",
              f"x-ray: {xray}" if xray else "x-ray not run yet (scanner queue)",
              f"seller: import-VAT scheme {show(vat, rng)}; {history}"]
        out = []
        for e in ev:
            t += timedelta(minutes=rng.randint(3, 19))
            out.append(f"{t.strftime('%H:%M')}  {noisy(e, rng, 0.08)}")
        return f"SHIFT LOG -- {f['hub']} -- {fmt_date(f['arrived'].date(), rng)} -- {f['officer']}\n" + "\n".join(out)

    def questions(self):
        return {
            "batch_handling": choice("How should the hub handle this batch?", {
                "release_all": "Release every parcel in the batch as declared.",
                "release_hold_flagged": "Release the batch but hold the individually flagged parcels for checks.",
                "hold_for_evidence": "Hold the batch until the courier supplies order confirmations or payment "
                                     "records for the parcels.",
                "examine_sample": "Open and examine a sample of parcels before deciding on the rest.",
                "detain_batch": "Detain the whole batch for action by a specialist enforcement unit.",
            }),
            "specialist_referral": choice("Which specialist team, if any, should this batch be passed to?", {
                "none_needed": "None; the hub team can deal with it using its own checks.",
                "valuation_team": "The valuation team, for prices that look under-declared.",
                "ip_rights_team": "The intellectual-property enforcement team, for possible counterfeit branded goods.",
                "product_safety_authority": "The product-safety market-surveillance authority, for goods that may be "
                                            "unsafe or unlawfully marked.",
                "revenue_investigations": "Revenue investigations, for a pattern that looks like organised duty "
                                          "evasion by the seller.",
            }),
            "undervaluation_likelihood": score("How likely is it that parcels in this batch are declared below the "
                                               "price the buyers actually paid?", [
                "Unlikely: declared values track the listing prices.",
                "Possible: a few parcels look low, but most values are credible.",
                "Likely: values are low across much of the batch.",
                "Near certain: declared values sit systematically far below the prices paid.",
            ]),
            "description_quality": score("How well do the declared contents let an officer tell what is in the "
                                         "parcels?", [
                "Useless: most descriptions say nothing about the goods, or the scans contradict them.",
                "Weak: many descriptions are too generic to classify or risk-assess.",
                "Serviceable: a minority are vague, but the batch can be assessed.",
                "Good: descriptions are specific and match what the scans show.",
            ]),
            "threshold_splitting": noul(
                "Orders in this batch appear to have been split into separate parcels to keep each under the "
                "duty-free threshold.",
                yes="Several consignees get multiple parcels at once and declared values bunch just below the threshold.",
                no="Parcels per consignee and the spread of declared values look like ordinary retail orders."),
        }


# =========================================================================== schema 6
# purpose -> what the visit is for, the general-list pool (description, qty, unit value, carries a serial number),
# goods that cannot travel on a carnet, what a sale plan looks like for this kind of visit, and the holder's trade
CARNET_PURPOSES = {
    "trade_fair": {
        "label": "an exhibition stand at the {event}", "holder": "maker",
        "items": [("demonstration packaging machine", 1, 42000, True), ("touchscreen display stands", 4, 1800, True),
                  ("sample product units", 30, 120, False), ("stand lighting rig", 1, 3500, True),
                  ("modular stand walls", 12, 260, False)],
        "consumables": ["brochures and branded giveaway pens", "food and drink samples for tasting at the stand",
                        "spare parts to be fitted to a local customer's machine"],
        "may_sell": "the holder's email says display units 'may be sold to visitors if the price is right'",
        "price_list": "a price list offering delivery from the stand was packed with the goods"},
    "film_shoot": {
        "label": "filming a documentary series", "holder": "production",
        "items": [("cinema camera bodies", 2, 28000, True), ("prime lens set", 1, 36000, True),
                  ("LED light panels", 8, 900, True), ("field audio recorder kit", 1, 4200, True),
                  ("camera drone", 1, 6500, True)],
        "consumables": ["batteries, tape and other consumables to be used up during the shoot",
                        "props to be left with the local production partner"],
        "may_sell": "the producer mentions selling the {item} to the local production partner after the shoot",
        "price_list": None},
    "sales_samples": {
        "label": "showing the new collection to retail buyers", "holder": "fashion",
        "items": [("sample garments", 140, 85, False), ("sample handbags", 35, 210, False),
                  ("display mannequins", 3, 400, False), ("sample footwear pairs", 48, 130, False)],
        "consumables": ["printed look-books and giveaway tote bags", "fabric swatch cards to be left with buyers"],
        "may_sell": "the holder's email says buyers 'can keep any samples they order in volume'",
        "price_list": "an order form offering the samples themselves at a discount was packed with the goods"},
    "concert_tour": {
        "label": "a concert tour", "holder": "touring",
        "items": [("backline amplifiers", 6, 2400, True), ("drum kit", 1, 5200, False),
                  ("mixing console", 1, 18000, True), ("stage lighting fixtures", 24, 1100, True),
                  ("wireless microphone set", 1, 7600, True)],
        "consumables": ["merchandise T-shirts to be sold at the venues",
                        "guitar strings and drum heads to be used up on tour"],
        "may_sell": "the tour manager mentions selling the {item} to a local venue at the end of the tour",
        "price_list": None},
}
CARNET_HISTORY = {"clean": "no problems on earlier carnets",
                  "late_once": "one earlier carnet discharged late, two years ago",
                  "claim_paid": "a claim was paid by the guarantor on an earlier carnet"}


class TemporaryAdmissionCarnet(Schema):
    name = "temporary_admission_carnet"
    description = ("Goods arriving for temporary admission under a carnet: whether to admit them, and how well the "
                   "carnet and the goods line up.")
    surfaces = {"json": 0.45, "carnet_check_sheet": 0.3, "agent_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        purpose = pick(rng, list(CARNET_PURPOSES))
        P = CARNET_PURPOSES[purpose]
        margin = pick(rng, [-12, -3, 2, 6, 15, 45, 120])
        match = pick_w(rng, {"all_match": 5, "detail_mismatch": 2, "extra_item": 2, "missing_item": 1})
        has_consumables = chance(rng, 0.2)
        sale = pick_w(rng, {"none": 6, "may_sell": 2, "price_list": 1})
        history = pick_w(rng, {"clean": 6, "late_once": 2, "claim_paid": 1})
        vouchers = pick(rng, [1, 2, 3, 4, 6])
        country_listed = True
        if reg == "clear":
            if chance(rng, 0.5):
                margin, match, has_consumables, sale, history = pick(rng, [60, 120, 200]), "all_match", False, "none", \
                    "clean"
            else:
                margin, match, sale, history = pick(rng, [-20, -8]), pick(rng, ["extra_item", "missing_item"]), \
                    pick(rng, ["may_sell", "price_list"]), "claim_paid"
                has_consumables, country_listed = True, chance(rng, 0.5)
        elif reg == "borderline":
            margin = near(rng, 3, 6, integer=True)
            if chance(rng, 0.5):
                match = "detail_mismatch"
            else:
                has_consumables = True
        else:
            variant = pick(rng, ["clean_holder_selling", "bad_history_everything_else_fine", "consumables_only"])
            if variant == "clean_holder_selling":
                history, match, margin = "clean", "all_match", pick(rng, [30, 60])
                sale = pick(rng, ["may_sell", "price_list"])
            elif variant == "bad_history_everything_else_fine":
                history, match, margin, sale = "claim_paid", pick(rng, ["all_match", "detail_mismatch"]), \
                    pick(rng, [20, 45]), "none"
            else:
                has_consumables, match, margin, history, sale = True, "all_match", pick(rng, [4, 30, 90]), "clean", "none"
        if sale == "price_list" and P["price_list"] is None:
            sale = "may_sell"
        pool = P["items"]
        items = [[d, q, v] for d, q, v, _ in rng.sample(pool, min(len(pool), pick(rng, [3, 4])))]
        chosen = {i[0] for i in items}
        serial_items = [d for d, _, _, ser in pool if ser and d in chosen]
        match_item = pick(rng, items)
        if match == "detail_mismatch":
            if serial_items:
                detail = f"serial number on the {pick(rng, serial_items)} differs from the general list by two digits"
            else:
                d, q, _ = pick(rng, [i for i in items if i[1] > 1] or items)
                detail = f"{q + rng.randint(2, 9)} {d} counted against {q} on the general list"
        else:
            detail = None
        arrival = rand_date(rng)
        if purpose == "trade_fair":
            ev_start = arrival + timedelta(days=rng.randint(2, 5))
            ev_end = ev_start + timedelta(days=rng.randint(3, 6))
            exit_d = ev_end + timedelta(days=rng.randint(1, 6))
        else:
            ev_start, ev_end = arrival + timedelta(days=1), None
            exit_d = arrival + timedelta(days=rng.randint(10, 45))
        valid_until = exit_d + timedelta(days=margin)
        if valid_until <= arrival:   # a carnet that lapsed before arrival would never be presented; let it lapse mid-stay
            valid_until = arrival + timedelta(days=rng.randint(1, max(1, (exit_d - arrival).days - 1)))
            margin = (valid_until - exit_d).days
        extra = pick(rng, [x for x in pool if x[0] not in chosen] or pool)
        sale_text = P.get(sale) if sale != "none" else None
        if sale_text:
            sale_text = sale_text.format(item=pick(rng, items)[0])
        return {
            "regime": reg, "purpose": purpose,
            "purpose_text": P["label"].format(event=pick(rng, ["Tidewater Packaging Expo", "Meridian Industry Week",
                                                               "Northgate Design Fair"])),
            "items": items, "carnet_value": sum(q * v for _, q, v in items),
            "arrival": arrival, "event_start": ev_start, "event_end": ev_end, "planned_exit": exit_d,
            "valid_until": valid_until, "margin_days": margin, "goods_match": match,
            "mismatch_detail": detail, "match_item": match_item[0], "extra_item": extra[0],
            "consumables": pick(rng, P["consumables"]) if has_consumables else None,
            "consumables_value": pick(rng, [180, 650, 1400, 3200]) if has_consumables else None,
            "sale_intent": sale, "sale_text": sale_text, "holder_history": history,
            "import_vouchers_left": vouchers, "country_listed": country_listed,
            "holder": _names(rng, P["holder"])[0], "agent": person(rng),
            "issuer": f"{_countries(rng, 1)[0]} chamber of commerce", "carnet_no": ident(rng, "CRN", 7, "-"),
            "currency": pick_w(rng, CURRENCIES), "office": f"{pick(rng, ['Harbour', 'Airport', 'Border'])} customs office",
        }

    MARGIN_RULE = "office practice: at least 7 days of carnet validity must remain after the planned re-export"

    def _gap(self, f):
        g = f["margin_days"]
        return (f"{g} days after the planned re-export" if g > 0 else
                "on the planned re-export day" if g == 0 else f"{-g} days before the planned re-export")

    def _match_text(self, f):
        return {"all_match": "all items, quantities and serial numbers agree with the general list",
                "detail_mismatch": f["mismatch_detail"],
                "extra_item": f"{f['extra_item']} found in the cases but not on the general list",
                "missing_item": f"{f['match_item']} on the general list but not presented"}[f["goods_match"]]

    def render(self, f, rng, surface):
        items = [f"{q} x {d}" for d, q, _ in f["items"]]
        if surface == "json":
            return dumps(prune({
                "carnet": {"number": f["carnet_no"], "holder": f["holder"], "issued_by": f["issuer"],
                           "valid_until": f["valid_until"].isoformat(), "expiry_vs_planned_exit": self._gap(f),
                           "total_value": f["carnet_value"],
                           "currency": f["currency"], "importation_vouchers_left": f["import_vouchers_left"],
                           "this_country_listed_for_use": f["country_listed"]},
                "visit": {"purpose": f["purpose_text"], "arrival": f["arrival"].isoformat(),
                          "event_ends": f["event_end"].isoformat() if f["event_end"] else None,
                          "planned_exit": f["planned_exit"].isoformat(), "office": f["office"]},
                "goods": {"general_list": items, "check_against_list": self._match_text(f),
                          "not_on_carnet": ({"description": f["consumables"], "value": f["consumables_value"]}
                                            if f["consumables"] else None),
                          "sale_note": f["sale_text"]},
                "holder_record": CARNET_HISTORY[f["holder_history"]],
                "validity_check": self.MARGIN_RULE,
            }))
        m, fd = _money_fmt(rng, f["currency"]), _date_fmt(rng)
        consumables = f"{f['consumables']} (value {m(f['consumables_value'])})" if f["consumables"] else None
        if surface == "carnet_check_sheet":
            head = kv_lines([
                ("Carnet", f"{f['carnet_no']} ({f['issuer']})"), ("Holder", f["holder"]),
                ("Valid until", f"{fd(f['valid_until'])} ({self._gap(f)}; {self.MARGIN_RULE})"),
                ("Carnet value", m(f["carnet_value"])),
                ("Importation vouchers left", f["import_vouchers_left"]),
                ("Country listed on carnet", _yn(f["country_listed"])),
                ("Purpose", f["purpose_text"]), ("Arrived", fd(f["arrival"])),
                ("Event ends", fd(f["event_end"]) if f["event_end"] else None),
                ("Planned re-export", fd(f["planned_exit"])),
                ("Holder record", CARNET_HISTORY[f["holder_history"]]),
            ], rng)
            rows = [[d, q, m(v)] for d, q, v in f["items"]]
            notes = [f"Check against list: {self._match_text(f)}."]
            if consumables:
                notes.append(f"Also in the cases, not on the carnet: {consumables}.")
            if f["sale_text"]:
                notes.append(f["sale_text"][0].upper() + f["sale_text"][1:] + ".")
            return (f"TEMPORARY ADMISSION -- CARNET CHECK ({f['office']})\n" + head + "\n\nGeneral list\n"
                    + table(["item", "qty", "unit value"], rows, rng) + "\n\n" + noisy(" ".join(notes), rng, 0.1))
        # agent_email: the holder's customs agent writes to the office ahead of arrival
        a = f["agent"]
        s = [f"We act for {f['holder']}, arriving {fd(f['arrival'])} under carnet {f['carnet_no']} "
             f"(issued by the {f['issuer']}, valid until {fd(f['valid_until'])}, {self._gap(f)}).",
             f"The goods are for {f['purpose_text']}"
             + (f", which ends {fd(f['event_end'])}" if f["event_end"] else "")
             + f", and are due to leave on {fd(f['planned_exit'])}.",
             f"General list: {', '.join(items)}; total {m(f['carnet_value'])}.",
             f"Our pre-check: {self._match_text(f)}."]
        if consumables:
            s.append(f"The shipment also carries {consumables}, which are not on the carnet.")
        if f["sale_text"]:
            s.append(f"Please also note: {f['sale_text']}.")
        if f["holder_history"] != "clean":
            s.append(f"For transparency: {CARNET_HISTORY[f['holder_history']]}.")
        if not f["country_listed"]:
            s.append("Please note this country was not ticked on the carnet cover.")
        s.append("One importation voucher remains." if f["import_vouchers_left"] == 1
                 else f"{f['import_vouchers_left']} importation vouchers remain.")
        return "\n".join([f"Subject: Carnet {f['carnet_no']} -- arrival {fd(f['arrival'])}", f"From: {a}",
                          f"To: {f['office']}", "", "Dear officer,", "", noisy(" ".join(s), rng, 0.12), "",
                          "Kind regards,", a, f"Customs agent for {f['holder']}", "",
                          f"[office note: {self.MARGIN_RULE}]"])

    def questions(self):
        return {
            "admission_decision": choice("How should the goods presented under this carnet be admitted?", {
                "admit_as_presented": "Admit every item temporarily under the carnet as presented.",
                "admit_eligible_part": "Admit the eligible items under the carnet and require a normal import entry, "
                                       "with duty, for the rest.",
                "admit_with_extra_security": "Admit under the carnet but take an extra deposit against goods staying "
                                             "behind.",
                "examine_first": "Examine the goods against the carnet's general list before deciding on admission.",
                "refuse_temporary_admission": "Refuse temporary admission; the goods must be entered for home use "
                                              "or taken back out.",
            }),
            "non_return_risk": score("How likely is it that some of these goods will not leave the country under the "
                                     "carnet in time?", [
                "Remote: a clean holder, professional goods and a comfortable schedule.",
                "Low: one small point to watch, but the goods are very likely to leave.",
                "Real: sale plans, a tight schedule or a poor record make non-return plausible.",
                "High: several signs that goods will be sold or kept, or the carnet cannot cover the stay.",
            ]),
            "list_correspondence": score("How closely do the goods presented match the carnet's general list?", [
                "Poorly: items are missing, extra or cannot be identified from the list.",
                "Partly: most items match, but at least one needs explaining.",
                "Closely: everything matches apart from a trivial clerical point.",
                "Exactly: every item, quantity and serial number agrees with the list.",
            ]),
            "validity_covers_stay": noul(
                "The carnet stays valid for the whole planned stay, including the time needed to take the goods back "
                "out.",
                yes="The carnet expires at least 7 days after the planned re-export, the margin office practice asks for.",
                no="The carnet expires before the planned re-export, or less than 7 days after it."),
            "all_items_eligible": noul("Every item presented can be admitted temporarily under this carnet."),
        }


WORKFLOW = Workflow(
    name="customs_declarations",
    title="Customs declarations review",
    description="How a customs broker's compliance desk, or a customs officer, should treat an import or export "
                "declaration: classification, value, origin, licensing, release and disclosure.",
    domain="international trade / customs compliance",
    schemas=[ImportEntryReview(), PreferentialOriginClaim(), DualUseExport(), PostClearanceAudit(),
             EcommerceLowValueBatch(), TemporaryAdmissionCarnet()],
    notes="Every question concerns customs treatment. A commercial invoice appears only as a customs document; "
          "no vendor bill is paid, held or rejected and nothing is matched against a purchase order or a delivery. "
          "Tariff codes, rates, agreements, control-list entries and most countries are invented.",
)
