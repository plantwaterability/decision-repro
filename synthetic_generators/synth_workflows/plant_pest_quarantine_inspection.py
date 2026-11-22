"""Plant pest quarantine inspection: what a border plant-health inspector does with an import consignment.

A (fictional) national plant protection authority, the Directorate of Plant Biosecurity, inspects consignments at
ports and airports against its written Import Conditions Schedule (ICS): fresh produce and cut flowers with an
organism found on them, wood packaging around non-plant goods, cold-treatment and fumigation records for fruit-fly
host fruit, nursery stock and other plants for planting under permit, and bulk grain vessels. The decisions are the
inspector's and the duty officer's: release, treat, re-export, destroy or hold; how much to trust a treatment mark
or a temperature log; whether a tolerance is met; whether to tell the exporting country.

Every state quotes the ICS clauses that govern the decision. Pest taxa are real organisms, but their regulatory
status here is the fictional schedule's, not any real country's. Origin countries, ports, importers and people are
invented. No customs tariff, valuation or origin question is asked, no bill or payment appears, and no
conversation with a member of the public is drafted.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

from .base import (
    ORG_A,
    REAL_NAME_CLASHES,
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
    table,
)

AUTHORITY = "Directorate of Plant Biosecurity"
COUNTRIES = ["Calvera", "Ostrand", "Marenzia", "Tobruna", "Selvana", "Kestria", "Durovia", "Pelagia", "Arnholt",
             "Vantoro", "Lisandra", "Morvane", "Quellan", "Rhodessa", "Sarvenia", "Talmor", "Uskara", "Veridia",
             "Zoravia", "Brenmark", "Castellon", "Ithrace"]
SEA_PORTS = ["Port Ardrey", "Halsey Container Terminal", "Saltmarsh Wharf", "Kerrow River Port", "Fairhaven Quay",
             "Dunlin Bay Terminal", "Old Pier 7", "Ravel Point Terminal", "Easterly Freight Gateway", "Corran Docks"]
AIR_PORTS = ["Brennock Air Cargo Hub", "Mallory Field Air Cargo", "Westley International cargo shed",
             "Tarrant Airport perishables centre"]
PORTS = SEA_PORTS + AIR_PORTS
BULK_BERTHS = ["Grenmouth Bulk Berth", "Port Ardrey grain terminal", "Saltmarsh No. 3 bulk berth", "Corran grain wharf",
               "Dunlin Bay bulk terminal", "Easterly silo berth"]


def _ds(rng):
    return pick(rng, ["iso", "iso", "us", "long", "eu"])


def _cap(s):
    return s[0].upper() + s[1:] if s else s


IMPORTER_SUFFIX = {
    "produce": ["Fresh", "Produce", "Fresh Imports", "Trading", "Fruit Co.", "Flowers", "Horticulture", "Foods"],
    "goods": ["Industries", "Trading", "Supply Co.", "Imports", "Group", "Holdings", "Distribution", "Building Supplies"],
    "plants": ["Nurseries", "Horticulture", "Plant Propagators", "Seeds", "Growers", "Bulbs"],
    "grain": ["Grain", "Commodities", "Milling", "Stockfeeds", "Grain Trading", "Oilseeds"],
}


def _importer(rng, kind="produce"):
    """ORG_A prefix + a trade-appropriate suffix, skipping combinations that name a real business."""
    a, sfx = rng.choice(ORG_A), IMPORTER_SUFFIX[kind]
    i = rng.randrange(len(sfx))
    for k in range(len(sfx)):
        name = f"{a} {sfx[(i + k) % len(sfx)]}"
        if name not in REAL_NAME_CLASHES:
            return name
    return name


def _entry(rng):
    return f"{pick(rng, ['IMP', 'BIO', 'ENT'])}-{rng.randint(20, 27)}{rng.randint(100000, 999999)}"


def _container(rng):
    letters = "".join(rng.choice("ABCDEFGHJKLMNPRSTUWXZ") for _ in range(3))
    return f"{letters}U{rng.randint(1000000, 9999999)}"


# =========================================================================== schema 1
# commodity -> (kind, shelf days range, fumigation tolerated)
PRODUCE = {
    "table grapes": ("fruit", (10, 25), True), "mandarins": ("fruit", (14, 30), True), "lemons": ("fruit", (20, 40), True),
    "avocados": ("fruit", (8, 18), False), "mangoes": ("fruit", (6, 14), False), "papayas": ("fruit", (4, 9), False),
    "fresh chillies": ("vegetable", (6, 12), True), "capsicums": ("vegetable", (8, 16), True),
    "tomatoes": ("vegetable", (5, 12), True), "apples": ("fruit", (30, 60), True), "pears": ("fruit", (25, 50), True),
    "cut roses": ("flower", (4, 9), True), "cut chrysanthemums": ("flower", (6, 12), True),
    "cut carnations": ("flower", (6, 14), True), "fresh basil": ("herb", (3, 6), False),
    "fresh ginger": ("vegetable", (30, 60), True), "sweet potatoes": ("vegetable", (25, 50), True),
    "lychees": ("fruit", (5, 12), False), "dragon fruit": ("fruit", (8, 16), True), "cherries": ("fruit", (6, 12), False),
    "blueberries": ("fruit", (5, 12), False), "asparagus": ("vegetable", (4, 9), True), "snow peas": ("vegetable", (5, 10), True),
    "orchid sprays": ("flower", (8, 16), True),
}
# taxon -> (common group, ICS status, genus contains a Schedule 2 species)
ORGANISMS = {
    "Bactrocera dorsalis": ("fruit fly", "schedule_2", True), "Ceratitis capitata": ("fruit fly", "schedule_2", True),
    "Zeugodacus cucurbitae": ("fruit fly", "schedule_2", True), "Thrips palmi": ("thrips", "schedule_2", True),
    "Scirtothrips dorsalis": ("thrips", "schedule_2", True), "Tuta absoluta": ("moth", "schedule_2", True),
    "Thaumatotibia leucotreta": ("moth", "schedule_2", True), "Spodoptera frugiperda": ("moth", "schedule_2", True),
    "Liriomyza huidobrensis": ("leafminer", "schedule_2", True), "Drosophila suzukii": ("vinegar fly", "schedule_2", True),
    "Frankliniella occidentalis": ("thrips", "schedule_4", True), "Tetranychus urticae": ("mite", "schedule_4", False),
    "Aphis gossypii": ("aphid", "schedule_4", False), "Planococcus citri": ("mealybug", "schedule_4", False),
    "Aonidiella aurantii": ("scale insect", "schedule_4", False), "Bemisia tabaci": ("whitefly", "schedule_4", False),
    "Helicoverpa armigera": ("moth", "schedule_4", False), "Epiphyas postvittana": ("moth", "schedule_4", False),
    "Myzus persicae": ("aphid", "schedule_4", False), "Pseudococcus viburni": ("mealybug", "schedule_4", True),
    "Brevipalpus californicus": ("flat mite", "unlisted", False), "Dysmicoccus neobrevipes": ("mealybug", "unlisted", True),
    "Maconellicoccus hirsutus": ("mealybug", "unlisted", False), "Chaetanaphothrips signipennis": ("thrips", "unlisted", False),
}
# group -> list of (stage, singular form)
STAGES = {"fruit fly": [("larvae in fruit", "larva in a fruit"), ("larvae in fruit", "larva in a fruit"),
                        ("puparia in packing material", "puparium in packing material")],
          "moth": [("larvae", "larva"), ("larvae in fruit", "larva in a fruit"), ("pupae", "pupa")],
          "thrips": [("adults", "adult"), ("adults and larvae", "adult")],
          "leafminer": [("larvae in leaf mines", "larva in a leaf mine"), ("puparia", "puparium")],
          "vinegar fly": [("larvae in fruit", "larva in a fruit"), ("adults", "adult")],
          "mite": [("adults and eggs", "adult"), ("adults", "adult")],
          "aphid": [("adults and nymphs", "adult"), ("winged adults", "winged adult")],
          "mealybug": [("adults and crawlers", "adult"), ("adult females", "adult female")],
          "scale insect": [("scale covers with live crawlers", "scale cover with live crawlers"), ("adult scales", "adult scale")],
          "whitefly": [("adults", "adult"), ("pupae on leaves", "pupa on a leaf")],
          "flat mite": [("adults", "adult"), ("adults and eggs", "adult")]}
HOST_GROUPS = {"fruit": {"fruit fly", "moth", "thrips", "mite", "mealybug", "scale insect", "vinegar fly", "flat mite"},
               "vegetable": {"fruit fly", "moth", "thrips", "whitefly", "aphid", "leafminer", "mite", "mealybug"},
               "flower": {"thrips", "mite", "aphid", "mealybug", "leafminer", "moth", "whitefly", "flat mite"},
               "herb": {"aphid", "thrips", "whitefly", "leafminer", "mite"}}
FLY_HOSTS = {"table grapes", "mandarins", "lemons", "avocados", "mangoes", "papayas", "fresh chillies", "capsicums",
             "tomatoes", "apples", "pears", "lychees", "dragon fruit", "cherries", "blueberries"}
TREATMENTS = {"fruit": "methyl bromide fumigation (T-MB-04) or cold treatment (CT-3)",
              "vegetable": "methyl bromide fumigation (T-MB-02)", "flower": "methyl bromide fumigation (T-MB-07)",
              "herb": "none approved for this commodity"}


class ConsignmentInterception(Schema):
    name = "consignment_interception"
    description = "A fresh produce or cut flower consignment with an organism found at inspection: release, treat, re-export, destroy or hold."
    surfaces = {"json": 0.5, "inspection_report": 0.25, "duty_officer_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        commodity = pick(rng, list(PRODUCE))
        kind, shelf_rng, fum_ok = PRODUCE[commodity]
        status_w = {"schedule_2": 4, "schedule_4": 4, "unlisted": 2}
        if reg == "clear":
            status_w = {"schedule_2": 1, "schedule_4": 1}
        status = pick_w(rng, status_w)

        def fits(t, c=None):
            c = c or commodity
            g = ORGANISMS[t][0]
            if g in ("fruit fly", "vinegar fly") and c not in FLY_HOSTS:
                return False
            return g in HOST_GROUPS[PRODUCE[c][0]]
        taxon = pick(rng, [t for t, v in ORGANISMS.items() if v[1] == status and fits(t)]
                     or [t for t in ORGANISMS if fits(t)])
        id_level = pick_w(rng, {"lab_species": 4, "inspector_species": 3, "genus_only": 2, "pending": 2})
        alive = chance(rng, 0.75)
        count = pick(rng, [1, 1, 2, 3, 5, 8, 14, 30])
        sample_units = pick(rng, [600, 600, 600, 450, 300])
        cert = pick_w(rng, {"in_order": 6, "ad_missing": 2, "mismatch": 1})
        shelf = rng.randint(*shelf_rng)
        pref = pick(rng, ["treat", "reexport", "reexport", "destroy", "release"])
        priors = pick(rng, [0, 0, 0, 1, 2, 4])
        if reg == "clear":
            id_level, alive = pick(rng, ["lab_species", "inspector_species"]), True
            cert = "in_order" if chance(rng, 0.7) else cert
        elif reg == "conflicting":
            twist = pick(rng, ["dead_fly", "clean_cert_bad_find", "genus", "perishable_pending", "no_treatment"])
            if twist == "no_treatment":
                commodity = pick(rng, ["avocados", "mangoes", "lychees", "cherries", "blueberries", "fresh basil", "papayas"])
                kind, shelf_rng, fum_ok = PRODUCE[commodity]
                shelf = rng.randint(shelf_rng[0], shelf_rng[1] + 4)
                taxon = pick(rng, [t for t, v in ORGANISMS.items() if v[1] != "schedule_4" and fits(t)])
                alive, id_level = True, pick(rng, ["lab_species", "inspector_species"])
                pref = pick(rng, ["reexport", "reexport", "treat", "destroy"])
            elif twist == "dead_fly":
                if commodity not in FLY_HOSTS:
                    commodity = pick(rng, sorted(FLY_HOSTS))
                    kind, shelf_rng, fum_ok = PRODUCE[commodity]
                    shelf = rng.randint(*shelf_rng)
                taxon = pick(rng, [t for t, v in ORGANISMS.items() if v[0] == "fruit fly"])
                alive, id_level = False, "lab_species"
            elif twist == "clean_cert_bad_find":
                cert, alive = "in_order", True
                taxon = pick(rng, [t for t, v in ORGANISMS.items() if v[1] == "schedule_2" and fits(t)])
                id_level = pick(rng, ["lab_species", "inspector_species"])
            elif twist == "genus":
                id_level = "genus_only"
                taxon = pick(rng, [t for t, v in ORGANISMS.items() if v[1] != "schedule_2" and fits(t)]
                             or [t for t in ORGANISMS if fits(t)])
            else:
                id_level, shelf = "pending", rng.randint(2, 4)
        group = ORGANISMS[taxon][0]
        stage = pick(rng, STAGES[group])
        return {"regime": reg, "commodity": commodity, "kind": kind, "fumigation_tolerated": fum_ok,
                "taxon": taxon, "group": group, "status": ORGANISMS[taxon][1], "genus_has_schedule_2": ORGANISMS[taxon][2],
                "id_level": id_level, "alive": alive, "life_stage": stage[1] if count == 1 else stage[0], "count": count,
                "sample_units": sample_units, "cartons": rng.randint(240, 4200), "certificate": cert,
                "shelf_days": shelf, "importer_preference": pref, "prior_interceptions_12m": priors,
                "origin": pick(rng, COUNTRIES), "importer": _importer(rng),
                "port": pick(rng, PORTS if PRODUCE[commodity][1][0] >= 10 else AIR_PORTS),
                "inspector": person(rng), "entry": _entry(rng), "date": rand_date(rng)}

    # ------------------------------------------------------------------ derived
    @staticmethod
    def _actionable(f):
        if f["id_level"] == "pending":
            return True                       # ICS 6.5: held as actionable until identified
        if f["id_level"] == "genus_only":
            return f["genus_has_schedule_2"] and (f["alive"] or f["group"] == "fruit fly")
        if not f["alive"] and f["group"] != "fruit fly":
            return False
        return f["status"] in ("schedule_2", "unlisted")

    def expected(self, f):
        """Intended answers from the latent factors (spread audit only; never rendered)."""
        act = self._actionable(f)
        treatable = f["kind"] != "herb" and f["fumigation_tolerated"]
        if not act:
            dec = "release"
        elif f["id_level"] in ("pending", "genus_only"):
            dec = "hold_for_identification" if f["shelf_days"] >= 5 else ("treat_then_release" if treatable else "destroy")
        elif f["group"] == "fruit fly" and f["alive"] or not treatable:
            dec = "reexport" if f["shelf_days"] >= 7 and f["importer_preference"] == "reexport" else "destroy"
        else:
            dec = "treat_then_release"
        # pest risk
        if f["status"] == "schedule_4" and f["id_level"] in ("lab_species", "inspector_species"):
            risk = 0
        elif f["status"] == "schedule_4" and not f["genus_has_schedule_2"]:
            risk = 1
        else:
            risk = 3 if f["group"] in ("fruit fly", "moth", "vinegar fly") else 2
            if not f["alive"]:
                risk -= 1
            if f["status"] == "unlisted" and f["id_level"] in ("lab_species", "inspector_species"):
                risk = min(risk, 2)
        if risk >= 1 and f["count"] >= 8 and f["alive"]:
            risk = min(3, risk + 1)
        ident_conf = {"lab_species": 3, "inspector_species": 2, "genus_only": 1, "pending": 0}[f["id_level"]]
        notice = ((f["status"] == "schedule_2" and f["id_level"] in ("lab_species", "inspector_species")
                   and (f["alive"] or f["group"] == "fruit fly")) or f["certificate"] != "in_order")
        return {"consignment_action": dec, "pest_risk_level": max(0, risk), "identification_confidence": ident_conf,
                "actionable_pest": "yes" if act else "no", "noncompliance_notice_due": "yes" if notice else "no"}

    # ------------------------------------------------------------------ rendering
    def _rules(self, rng, f):
        return [
            pick(rng, ["ICS 6.3 An interception is actionable when a live organism is a Schedule 2 pest or is not "
                       "listed in Schedule 2 or 4 (unassessed). Schedule 4 organisms (present and not under official "
                       "control) are not actionable. Dead specimens are actionable only for Schedule 2 fruit flies.",
                       "ICS 6.3 Actionable: live Schedule 2 pests and live organisms on neither Schedule 2 nor 4. Not "
                       "actionable: Schedule 4 organisms. Dead specimens: actionable only if Schedule 2 fruit flies."]),
            "ICS 6.5 An organism identified only to a genus that contains a Schedule 2 species, or not yet identified, "
            "is treated as actionable and the consignment held until identification, unless it is too perishable to hold.",
            pick(rng, ["ICS 6.7 Actionable consignments may be treated by an approved treatment and released; live "
                       "fruit fly larvae may not be treated on arrival and the consignment must be re-exported or "
                       "destroyed. Without an approved treatment: re-export or destroy. Re-export only at the importer's "
                       "request and with at least 7 days of shelf life left; otherwise destroy.",
                       "ICS 6.7 Treat with an approved treatment where one exists for the pest and commodity; live "
                       "fruit fly larvae are not treatable on arrival (re-export or destroy); otherwise re-export or destroy. "
                       "Re-export needs the importer's request and 7+ days of shelf life; failing that, destroy."]),
            "ICS 9.1 Notify the exporting country of significant non-compliance: a confirmed Schedule 2 interception, "
            "or a missing or incorrect certificate or additional declaration.",
        ]

    def _find(self, f):
        live = "live" if f["alive"] else "dead"
        name = {"lab_species": f["taxon"], "inspector_species": f["taxon"],
                "genus_only": f"{f['taxon'].split()[0]} sp.", "pending": f"unidentified {f['group']}"}[f["id_level"]]
        return f"{f['count']} {live} {f['life_stage']} ({name})"

    def _id_text(self, f):
        return {"lab_species": "species confirmed by the entomology lab",
                "inspector_species": "identified to species by the inspector at the bench (no lab confirmation)",
                "genus_only": "lab could only take it to genus (immature specimens)",
                "pending": "specimens sent to the lab; no identification yet"}[f["id_level"]]

    def _cert_text(self, f):
        return {"in_order": "phytosanitary certificate in order",
                "ad_missing": "phytosanitary certificate lacks the additional declaration the ICS requires for this pathway",
                "mismatch": "phytosanitary certificate lists a different packhouse and carton count"}[f["certificate"]]

    def render(self, f, rng, surface):
        ds = _ds(rng)
        treat = TREATMENTS[f["kind"]]
        if f["kind"] != "herb" and not f["fumigation_tolerated"]:
            treat = "fumigation approved but damages this commodity (importer will not accept it); no other approved option"
        pref = {"treat": "asks for treatment", "reexport": "wants to re-export to origin", "destroy": "would rather destroy than pay for treatment",
                "release": "says the find is trivial and asks for release"}[f["importer_preference"]]
        if surface == "json":
            return dumps(prune({
                "consignment": {"entry": f["entry"], "commodity": f["commodity"], "origin": f["origin"], "port": f["port"],
                                "cartons": f["cartons"], "importer": f["importer"], "certificate": self._cert_text(f),
                                "remaining_shelf_life_days": f["shelf_days"]},
                "inspection": {"inspector": f["inspector"], "date": fmt_date(f["date"], style="iso"),
                               "units_inspected": f["sample_units"], "finding": self._find(f),
                               "identification": self._id_text(f)},
                "options": {"approved_treatment": treat, "importer_position": pref},
                "history": {"interceptions_same_exporter_12m": f["prior_interceptions_12m"]},
                "rules": self._rules(rng, f),
            }))
        if surface == "inspection_report":
            head = kv_lines([("Entry", f["entry"]), ("Port", f["port"]), ("Date", fmt_date(f["date"], style=ds)),
                             ("Commodity", f"{f['commodity']}, {f['cartons']:,} cartons"), ("Origin", f["origin"]),
                             ("Importer", f["importer"]), ("Inspector", f["inspector"]),
                             ("Sample", f"{f['sample_units']} units"), ("Finding", self._find(f)),
                             ("Identification", self._id_text(f)), ("Certificate", self._cert_text(f)),
                             ("Shelf life remaining", f"{f['shelf_days']} days"), ("Approved treatment", treat),
                             ("Importer position", pref),
                             ("Prior interceptions, this exporter (12 mo)", f["prior_interceptions_12m"])], rng)
            return (f"{AUTHORITY.upper()} -- CONSIGNMENT INSPECTION REPORT\n{head}\n\nApplicable ICS clauses:\n"
                    + "\n".join(self._rules(rng, f)))
        lead = person(rng)
        para = [f"{f['entry']}: {f['cartons']:,} cartons of {f['commodity']} from {f['origin']} at {f['port']}, "
                f"importer {f['importer']}.",
                f"Out of {f['sample_units']} units inspected we found {self._find(f)}; {self._id_text(f)}.",
                _cap(self._cert_text(f)) + ".",
                f"Remaining shelf life about {f['shelf_days']} days. Treatment: {treat}.",
                f"The importer {pref}."]
        if f["prior_interceptions_12m"]:
            para.append(f"This exporter has had {f['prior_interceptions_12m']} interception(s) in the past 12 months.")
        return "\n".join([f"Subject: {f['entry']} {f['commodity']} -- direction needed", f"From: {f['inspector']}",
                          f"To: {lead} (duty officer)", "", f"{lead.split()[0]},", "",
                          noisy(join_sentences(para), rng, 0.12), "", "For reference:", bullets(self._rules(rng, f), rng),
                          "", f["inspector"].split()[0]])

    def questions(self):
        return {
            "consignment_action": choice("What should happen to this consignment?", {
                "release": "Release it to the importer without measures.",
                "treat_then_release": "Apply an approved treatment, then release it.",
                "reexport": "Order it re-exported at the importer's cost.",
                "destroy": "Order it destroyed under supervision.",
                "hold_for_identification": "Hold it under seal until the organism is identified.",
            }),
            "pest_risk_level": score("How much plant-health risk does the organism found pose to the importing country?", [
                "Negligible: a pest already established and not regulated.",
                "Low: some concern, but limited establishment potential or no live stages.",
                "Significant: a regulated or unassessed organism with live stages.",
                "High: a damaging regulated pest, alive or in numbers, on a host it can establish from.",
            ]),
            "identification_confidence": score("How confident is the identification of the organism?", [
                "None yet: the specimens have not been identified.",
                "Low: identified only to genus.",
                "Moderate: a species call made at the bench without lab confirmation.",
                "High: species confirmed by the laboratory.",
            ]),
            "actionable_pest": noul(
                "The organism found is actionable under the import conditions.",
                yes="Under the schedule's rules the find requires phytosanitary measures (or a hold) before release.",
                no="The find is a non-regulated organism, or a dead specimen the rules do not act on."),
            "noncompliance_notice_due": noul(
                "The exporting country's plant protection organisation must be notified of non-compliance."),
        }


# =========================================================================== schema 2
GOODS = ["ceramic floor tiles", "granite slabs", "machinery spare parts", "steel coils", "glassware", "brake pads",
         "solar panel frames", "marble countertops", "wire rope", "electric motors", "industrial valves", "water pumps",
         "bathroom fittings", "cast iron pipe fittings", "furniture hardware", "lead-acid batteries", "porcelain sinks",
         "transformer cores"]
BORERS = ["Cerambycidae (longhorn beetle) larvae", "Buprestidae (jewel beetle) larvae", "Scolytinae (bark beetle) adults",
          "Siricidae (woodwasp) larvae", "Bostrichidae (auger beetle) adults"]
MARK_TEXT = {
    "clear": "treatment mark present on two opposite sides: symbol, country code, producer no. and HT code all legible",
    "partial": "mark present but the producer number is missing on every piece",
    "illegible": "mark stamped but smudged to illegibility on most pieces",
    "suspicious": "freehand-stencilled mark with a country code that does not match the origin and a non-standard layout",
    "absent": "no treatment mark on any piece",
}
INSECT_TEXT = {
    "none": "no insect holes, frass or live stages",
    "old_holes": "a few round exit holes (2-4 mm) with no frass, apparently old",
    "fresh_frass": "fresh, pale frass at several holes; no live insects seen",
    "live": "live {borer} found in one pallet block",
}


class WoodPackagingCheck(Schema):
    name = "wood_packaging_check"
    description = "Wood packaging around non-plant goods checked for treatment marks, bark and pests: release, treat, strip, re-export or hold."
    surfaces = {"json": 0.5, "container_check_sheet": 0.25, "inspector_notes": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        mark = pick_w(rng, {"clear": 4, "partial": 2, "illegible": 1, "suspicious": 1, "absent": 1})
        insect = pick_w(rng, {"none": 5, "old_holes": 2, "fresh_frass": 1, "live": 2})
        n_bark = pick(rng, [0, 0, 1, 2, 3])
        bark_mode = pick_w(rng, {"within": 3, "near": 3, "over": 2})
        unpack = chance(rng, 0.55)
        fum_ok = chance(rng, 0.75)
        priors = pick(rng, [0, 0, 0, 1, 2, 3, 5])
        member = chance(rng, 0.5)
        borer_id = pick(rng, ["pending", "pending", "identified"])
        if reg == "clear":
            if chance(rng, 0.55):
                mark, insect, n_bark, priors = "clear", "none", pick(rng, [0, 1]), pick(rng, [0, 0, 1])
                bark_mode = "within"
            else:
                mark, insect = pick(rng, ["absent", "suspicious"]), pick(rng, ["none", "old_holes", "fresh_frass"])
        elif reg == "borderline":
            n_bark = max(1, n_bark)
            bark_mode = "near"
        else:
            twist = pick(rng, ["clean_mark_live", "bad_mark_clean_wood", "history"])
            if twist == "clean_mark_live":
                mark, insect = "clear", pick(rng, ["fresh_frass", "live"])
            elif twist == "bad_mark_clean_wood":
                mark, insect, n_bark = pick(rng, ["partial", "illegible"]), "none", 0
            else:
                priors, member = pick(rng, [3, 4, 6]), True
                mark = pick(rng, ["clear", "partial", "suspicious"])
        bark = []
        for _ in range(n_bark):
            if bark_mode == "within":
                w, l = round(between(rng, 0.8, 2.8), 1), round(between(rng, 5, 40), 0)
            elif bark_mode == "near":
                w = round(between(rng, 2.5, 4.5), 1)
                l = round(near(rng, 50 / w, 4, lo=3), 0)
            else:
                w, l = round(between(rng, 3.5, 7.0), 1), round(between(rng, 15, 40), 0)
            bark.append([w, l])
        return {"regime": reg, "goods": pick(rng, GOODS),
                "wpm_type": pick(rng, ["pallets", "crates", "pieces of dunnage", "skids", "timber bearers"]),
                "pieces": rng.randint(8, 64), "mark": mark, "insect": insect, "borer": pick(rng, BORERS),
                "borer_id": borer_id if insect == "live" else None, "bark_pieces": bark,
                "unpack_possible": unpack, "fumigation_feasible": fum_ok, "prior_noncompliances_12m": priors,
                "scheme_member": member, "origin": pick(rng, COUNTRIES), "port": pick(rng, SEA_PORTS),
                "importer": _importer(rng, "goods"), "inspector": person(rng), "container": _container(rng),
                "entry": _entry(rng), "date": rand_date(rng)}

    @staticmethod
    def _bark_ok(f):
        return all(w < 3.0 or w * l < 50.0 for w, l in f["bark_pieces"])

    def _compliant(self, f):
        return f["mark"] == "clear" and self._bark_ok(f) and f["insect"] in ("none", "old_holes")

    def expected(self, f):
        """Intended answers from the latent factors (spread audit only; never rendered)."""
        comp = self._compliant(f)
        if f["insect"] == "live" and f["borer_id"] == "pending":
            act = "hold_for_entomology"
        elif comp:
            act = "release"
        elif f["insect"] in ("fresh_frass", "live"):
            act = "treat_container" if f["fumigation_feasible"] else "reexport_container"
        elif f["unpack_possible"]:
            act = "remove_and_destroy_wood_only"
        else:
            act = "treat_container" if f["fumigation_feasible"] else "reexport_container"
        if comp:
            follow = "no_followup"
        elif f["prior_noncompliances_12m"] >= 3 or f["mark"] == "suspicious":
            follow = "suspend_scheme_membership" if f["scheme_member"] else "raised_inspection_rate"
        elif f["prior_noncompliances_12m"] >= 1:
            follow = "raised_inspection_rate"
        else:
            follow = "advisory_letter"
        cred = {"clear": 3, "partial": 2, "illegible": 1, "suspicious": 0, "absent": 0}[f["mark"]]
        if f["insect"] in ("fresh_frass", "live"):
            cred = max(0, cred - 1)
        inf = {"none": 0, "old_holes": 1, "fresh_frass": 2, "live": 3}[f["insect"]]
        return {"wpm_action": act, "importer_followup": follow, "mark_credibility": cred, "infestation_evidence": inf,
                "bark_within_tolerance": "yes" if self._bark_ok(f) else "no"}

    def _rules(self, rng):
        return [
            pick(rng, ["ICS 7.2 Wood packaging must bear the approved treatment mark with all four elements (symbol, "
                       "country code, producer number, treatment code), legible on two opposite sides.",
                       "ICS 7.2 Compliant marks show the symbol, country code, producer number and HT/MB code, legibly, "
                       "on at least two opposite sides."]),
            pick(rng, ["ICS 7.3 Bark is tolerated only in pieces under 3 cm wide (any length), or wider pieces whose "
                       "surface area is under 50 cm2.",
                       "ICS 7.3 Bark tolerance: any piece < 3 cm wide; a piece >= 3 cm wide only if its area is < 50 cm2."]),
            "ICS 7.5 Live wood-boring stages or fresh frass make the packaging non-compliant; live stages not yet "
            "identified hold the container for entomology.",
            pick(rng, ["ICS 7.6 Non-compliant packaging: remove and destroy it at an approved premises if the goods can "
                       "be unpacked there; otherwise, or where there is active infestation, fumigate the container if "
                       "feasible, else re-export.",
                       "ICS 7.6 Strip and destroy non-compliant wood where goods can be unpacked at approved premises; "
                       "fumigate the whole container where infestation is active or unpacking is impossible; re-export "
                       "if fumigation is not feasible."]),
            "ICS 7.9 Importer follow-up: first non-compliance in 12 months, advisory letter; repeat, raised inspection "
            "rate; three or more, or a suspect mark, suspension from the compliance scheme for members.",
        ]

    def _bark_text(self, f):
        if not f["bark_pieces"]:
            return "no bark"
        return "; ".join(f"{w:g} x {l:g} cm" for w, l in f["bark_pieces"])

    def render(self, f, rng, surface):
        ds = _ds(rng)
        insect = INSECT_TEXT[f["insect"]].format(borer=f["borer"])
        if f["insect"] == "live":
            insect += "; " + ("sent to entomology, no ID yet" if f["borer_id"] == "pending"
                              else "identified at the bench as a common, non-regulated species")
        unpack = ("approved unpacking premises available nearby" if f["unpack_possible"]
                  else "goods cannot be unpacked locally (heavy-lift, no approved premises)")
        fum = "fumigation feasible" if f["fumigation_feasible"] else "fumigation not feasible (goods or container type)"
        hist = (f"{f['prior_noncompliances_12m']} non-compliance(s) in 12 months; "
                + ("compliance-scheme member" if f["scheme_member"] else "not in the compliance scheme"))
        if surface == "json":
            return dumps(prune({
                "consignment": {"entry": f["entry"], "container": f["container"], "goods": f["goods"], "origin": f["origin"],
                                "port": f["port"], "importer": f["importer"]},
                "packaging": {"type": f["wpm_type"], "pieces": f["pieces"], "mark": MARK_TEXT[f["mark"]],
                              "bark_pieces_w_x_l": self._bark_text(f), "insect_evidence": insect},
                "logistics": {"unpacking": unpack, "fumigation": fum},
                "importer_record": hist,
                "rules": self._rules(rng),
            }))
        if surface == "container_check_sheet":
            rows = [["Mark", MARK_TEXT[f["mark"]]], ["Bark (w x l)", self._bark_text(f)], ["Insects", insect],
                    ["Unpacking", unpack], ["Fumigation", fum], ["Importer record", hist]]
            head = kv_lines([("Entry", f["entry"]), ("Container", f["container"]), ("Port", f["port"]),
                             ("Date", fmt_date(f["date"], style=ds)), ("Goods", f"{f['goods']} from {f['origin']}"),
                             ("Packaging", f"{f['pieces']} {f['wpm_type']}"), ("Importer", f["importer"]),
                             ("Inspector", f["inspector"])], rng)
            return (f"WOOD PACKAGING CHECK -- {AUTHORITY}\n{head}\n\n" + table(["item", "observed"], rows, rng)
                    + "\n\n" + "\n".join(self._rules(rng)))
        para = [f"{f['container']} ({f['goods']}, {f['origin']}) opened at {f['port']}.",
                f"{f['pieces']} {f['wpm_type']}: {MARK_TEXT[f['mark']]}.",
                f"Bark: {self._bark_text(f)}.", _cap(insect) + ".", _cap(unpack) + "; " + fum + ".",
                f"Importer {f['importer']}: {hist}."]
        if f["mark"] == "suspicious":
            para.append(pick(rng, ["Broker insists the pallets came from a certified supplier.",
                                   "Importer says the supplier changed and they were not told."]))
        return (f"{fmt_date(f['date'], style=ds)} -- {f['inspector']} -- wood packaging, entry {f['entry']}\n\n"
                + noisy(join_sentences(para), rng, 0.12) + "\n\nRules:\n" + bullets(self._rules(rng), rng))

    def questions(self):
        return {
            "wpm_action": choice("What should be done with this container and its wood packaging?", {
                "release": "Release the container; the packaging complies.",
                "treat_container": "Fumigate the whole container with the goods in it before release.",
                "remove_and_destroy_wood_only": "Unpack at approved premises, destroy the wood, release the goods.",
                "reexport_container": "Order the container re-exported with its packaging.",
                "hold_for_entomology": "Hold the container until the insects found are identified.",
            }),
            "importer_followup": choice("What follow-up should the importer receive?", {
                "no_followup": "None; nothing was wrong with this consignment.",
                "advisory_letter": "An advisory letter explaining the requirement.",
                "raised_inspection_rate": "Inspect a higher share of the importer's future containers for a period.",
                "suspend_scheme_membership": "Suspend the importer from the compliance scheme.",
            }),
            "mark_credibility": score("How far can the treatment mark be trusted as evidence that the wood was treated?", [
                "Not at all: missing, or looks fabricated.",
                "Little: present but unreadable.",
                "Somewhat: genuine-looking but incomplete or undercut by what was found in the wood.",
                "Fully: complete, legible and consistent with the wood's condition.",
            ]),
            "infestation_evidence": score("How strong is the evidence of pest activity in the wood?", [
                "None: no holes, frass or insects.",
                "Old signs only: exit holes without frass.",
                "Recent activity: fresh frass but no insects found.",
                "Active: live wood-boring insects present.",
            ]),
            "bark_within_tolerance": noul(
                "The bark on the packaging is within the tolerance in the import conditions.",
                yes="Every piece of bark is under 3 cm wide, or under 50 cm2 in area.",
                no="At least one piece is 3 cm or wider and 50 cm2 or more in area."),
        }


# =========================================================================== schema 3
# commodity -> (host quality, shelf life days after arrival, fumigation schedule approved)
CT_HOSTS = {"mandarins": ("good", 25, True), "oranges": ("good", 35, True), "grapefruit": ("moderate", 35, True),
            "table grapes": ("poor", 20, True), "apples": ("moderate", 50, True), "pears": ("moderate", 40, True),
            "lemons": ("poor", 40, True),
            "kiwifruit": ("poor", 40, False), "persimmons": ("good", 25, False), "cherries": ("good", 12, True),
            "plums": ("good", 18, True), "lychees": ("good", 10, False), "pomegranates": ("moderate", 30, False),
            "nectarines": ("good", 14, True), "blueberries": ("moderate", 12, True)}
SCHEDULES = [("CT-1", 0.0, 10), ("CT-2", 0.6, 12), ("CT-3", 1.1, 14), ("CT-4", 1.7, 16), ("CT-5", 2.2, 18),
             ("CT-6", 1.4, 15), ("CT-7", 0.6, 13)]
FLY_STATUS = {"widespread": "Bactrocera and Ceratitis species established throughout the exporting country",
              "limited": "Ceratitis capitata present in the coastal districts only",
              "freedom": "packhouse is in a district the ICS recognises as fruit-fly free, with trapping records attached"}


class TreatmentRecordVerification(Schema):
    name = "treatment_record_verification"
    description = "In-transit cold-treatment records for fruit-fly host fruit checked on arrival: accept, extend, fumigate, re-export or destroy."
    surfaces = {"json": 0.5, "probe_log_summary": 0.25, "treatment_review_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        commodity = pick(rng, list(CT_HOSTS))
        host, shelf, fum = CT_HOSTS[commodity]
        sched, thr, days_req = pick(rng, SCHEDULES)
        voyage = rng.randint(days_req + 2, days_req + 14)
        start_delay = pick(rng, [1, 1, 2, 3, 4])
        n_pulp = pick_w(rng, {3: 4, 4: 3, 2: 1})
        offsets = [round(rng.uniform(-0.25, 0.25), 2) for _ in range(n_pulp)]
        exc = None
        gap_h = None
        cold_store = chance(rng, 0.55)
        status = pick_w(rng, {"widespread": 4, "limited": 3, "freedom": 2})
        if reg == "clear":
            if chance(rng, 0.5):
                n_pulp, exc, gap_h = max(3, n_pulp), None, None
                offsets = [round(rng.uniform(-0.2, 0.2), 2) for _ in range(n_pulp)]
                voyage = days_req + rng.randint(5, 12)
            else:
                exc = {"day": rng.randint(4, days_req - 1), "probe": 1, "temp": round(thr + between(rng, 0.8, 2.5), 1),
                       "hours": round(between(rng, 5, 30), 1)}
                voyage = days_req + rng.randint(2, 5)
        elif reg == "borderline":
            voyage = days_req + start_delay + near(rng, 0, 2, integer=True)
            if chance(rng, 0.5):
                exc = {"day": rng.randint(3, max(4, voyage - 4)), "probe": rng.randint(1, n_pulp),
                       "temp": round(thr + between(rng, 0.1, 0.6), 1), "hours": round(near(rng, 2.0, 1.2, lo=0.3), 1)}
            if chance(rng, 0.6):
                k = rng.randrange(n_pulp)
                offsets[k] = round(pick(rng, [-1, 1]) * near(rng, 0.32, 0.12, lo=0.05), 2)
        else:
            twist = pick(rng, ["probe_bad", "gap", "late_reset", "messy"])
            if twist == "messy":
                n_pulp = pick(rng, [2, 3])
                offsets = [round(rng.uniform(-0.25, 0.25), 2) for _ in range(n_pulp)]
                offsets[-1] = round(pick(rng, [-1, 1]) * between(rng, 0.35, 0.8), 2)
                gap_h = round(between(rng, 2.5, 12.0), 1)
                exc = {"day": rng.randint(3, days_req), "probe": 1, "temp": round(thr + between(rng, 0.3, 1.2), 1),
                       "hours": round(between(rng, 2.5, 9), 1)}
                voyage = days_req + rng.randint(3, 9)
            elif twist == "probe_bad":
                offsets[0] = round(pick(rng, [-1, 1]) * between(rng, 0.4, 0.9), 2)
                voyage = days_req + rng.randint(4, 10)
            elif twist == "gap":
                gap_h = round(between(rng, 1.0, 9.0), 1)
                voyage = days_req + rng.randint(3, 10)
            else:
                exc = {"day": rng.randint(days_req - 4, days_req + 2), "probe": rng.randint(1, n_pulp),
                       "temp": round(thr + between(rng, 0.4, 1.5), 1), "hours": round(between(rng, 2.5, 7), 1)}
                voyage = days_req + rng.randint(2, 8)
        if exc and exc["day"] >= voyage - 1:
            exc["day"] = max(2, voyage - 2)
        load = rand_date(rng)
        return {"regime": reg, "commodity": commodity, "host_quality": host, "shelf_days": shelf + rng.randint(-4, 6),
                "fumigation_approved": fum and chance(rng, 0.7), "fumigation_listed": fum, "schedule": sched, "threshold_c": thr, "days_required": days_req,
                "voyage_days": voyage, "start_delay_days": start_delay, "pulp_offsets": offsets, "excursion": exc,
                "gap_hours": gap_h, "gap_day": rng.randint(3, 9), "cold_store_available": cold_store,
                "fly_status": status, "origin": pick(rng, COUNTRIES), "port": pick(rng, SEA_PORTS),
                "vessel": f"MV {pick(rng, ['Arden', 'Coral', 'Northern', 'Silver', 'Kestrel', 'Polar', 'Amber'])} "
                          f"{pick(rng, ['Star', 'Reefer', 'Bay', 'Dawn', 'Crest', 'Trader'])}",
                "importer": _importer(rng), "officer": person(rng), "entry": _entry(rng), "container": _container(rng),
                "loaded": load}

    # ------------------------------------------------------------------ derived
    @staticmethod
    def _valid_probes(f):
        return sum(1 for o in f["pulp_offsets"] if abs(o) <= 0.3)

    def _days_achieved(self, f):
        start = f["start_delay_days"]
        end = f["voyage_days"]
        reset_at = start
        exc = f["excursion"]
        if exc and exc["hours"] > 2.0 and exc["day"] > start:
            reset_at = exc["day"]
        if f["gap_hours"] and f["gap_hours"] > 2.0 and f["gap_day"] > reset_at:
            reset_at = max(reset_at, f["gap_day"])
        return end - reset_at

    def _record_valid(self, f):
        return self._valid_probes(f) >= 3

    def expected(self, f):
        """Intended answers from the latent factors (spread audit only; never rendered)."""
        valid = self._record_valid(f)
        achieved = self._days_achieved(f)
        met = valid and achieved >= f["days_required"]
        if met:
            verdict = "accept_treatment"
        elif valid and f["cold_store_available"]:
            verdict = "extend_treatment_on_arrival"
        elif f["fumigation_approved"]:
            verdict = "retreat_by_fumigation"
        else:
            verdict = "reexport" if f["shelf_days"] >= f["voyage_days"] + 5 else "destroy"
        integ = 3
        if self._valid_probes(f) < len(f["pulp_offsets"]):
            integ -= 1
        if len(f["pulp_offsets"]) < 3 or self._valid_probes(f) < 3:
            integ -= 1
        if f["gap_hours"]:
            integ -= 1
        if f["excursion"] and f["excursion"]["hours"] > 2.0:
            integ -= 1
        risk = {"good": (3, 2, 1), "moderate": (2, 1, 0), "poor": (1, 0, 0)}[f["host_quality"]][
            ("widespread", "limited", "freedom").index(f["fly_status"])]
        return {"treatment_verdict": verdict, "log_integrity": max(0, integ), "fly_risk_if_untreated": risk,
                "schedule_met": "yes" if met else "no",
                "probe_calibration_acceptable": "yes" if valid else "no"}

    def _rules(self, rng, f):
        return [
            pick(rng, [f"ICS 8.2 Schedule {f['schedule']}: pulp temperature at or below {f['threshold_c']} C for "
                       f"{f['days_required']} consecutive days, counted from when every pulp probe first reaches it.",
                       f"ICS 8.2 {f['schedule']} requires {f['days_required']} consecutive days with all pulp probes at "
                       f"or below {f['threshold_c']} C; day one is the first day all probes are down."]),
            pick(rng, ["ICS 8.3 Any pulp reading above the schedule temperature for more than 2 hours, or a recording "
                       "gap longer than 2 hours, restarts the day count.",
                       "ICS 8.3 The count restarts after any excursion above the schedule temperature lasting over 2 h, "
                       "and after any gap in recording over 2 h."]),
            "ICS 8.4 At least 3 pulp probes are required; a probe whose ice-point check at loading was off by more than "
            "0.3 C is invalid and its readings are disregarded.",
            pick(rng, ["ICS 8.6 A valid but incomplete treatment may be completed in an approved cold store at the port. "
                       "Otherwise fumigate where an approved schedule exists for the commodity; failing that, re-export "
                       "if the remaining shelf life covers a return voyage as long as the inbound one plus 5 days, else "
                       "destroy.",
                       "ICS 8.6 Incomplete but valid: finish in an approved port cold store. Invalid, or no cold store: "
                       "approved fumigation if one exists; else re-export when shelf life exceeds the voyage length by "
                       "at least 5 days, otherwise destroy."]),
        ]

    def _probe_text(self, f):
        return ", ".join(f"P{i + 1} {o:+.2f} C" for i, o in enumerate(f["pulp_offsets"]))

    def _events(self, f):
        ev = []
        exc = f["excursion"]
        if exc:
            ev.append(f"day {exc['day']}: pulp probe P{min(exc['probe'], len(f['pulp_offsets']))} read {exc['temp']} C "
                      f"for {exc['hours']} h")
        if f["gap_hours"]:
            ev.append(f"day {f['gap_day']}: logger gap of {f['gap_hours']} h (all channels)")
        return ev

    def render(self, f, rng, surface):
        ds = _ds(rng)
        arrival = f["loaded"] + timedelta(days=f["voyage_days"])
        ev = self._events(f) or ["no excursions or gaps recorded"]
        cs = "approved port cold store has space" if f["cold_store_available"] else "no approved cold-store space at the port this week"
        if f["fumigation_approved"]:
            fum = f"approved fumigation schedule exists for {f['commodity']} and the port chamber has a slot"
        elif f.get("fumigation_listed"):
            fum = (f"an approved fumigation schedule exists for {f['commodity']}, but the port's only approved chamber "
                   "is out of service for two weeks")
        else:
            fum = f"no approved fumigation schedule for {f['commodity']}"
        if surface == "json":
            return dumps(prune({
                "consignment": {"entry": f["entry"], "commodity": f["commodity"], "origin": f["origin"],
                                "container": f["container"], "vessel": f["vessel"], "port": f["port"],
                                "importer": f["importer"], "remaining_shelf_life_days": f["shelf_days"]},
                "treatment": {"schedule": f["schedule"], "loaded": fmt_date(f["loaded"], style="iso"),
                              "arrived": fmt_date(arrival, style="iso"),
                              "all_pulp_probes_at_temperature_from_day": f["start_delay_days"],
                              "days_logged": f["voyage_days"], "pulp_probe_ice_point_offsets": self._probe_text(f),
                              "events": ev},
                "origin_status": FLY_STATUS[f["fly_status"]],
                "options": {"cold_store": cs, "fumigation": fum},
                "rules": self._rules(rng, f),
            }))
        if surface == "probe_log_summary":
            rows = [[f"P{i + 1}", f"{o:+.2f}"] for i, o in enumerate(f["pulp_offsets"])]
            head = kv_lines([("Entry / container", f"{f['entry']} / {f['container']}"), ("Vessel", f["vessel"]),
                             ("Commodity", f"{f['commodity']} ({f['origin']})"), ("Schedule", f["schedule"]),
                             ("Loaded", fmt_date(f["loaded"], style=ds)), ("Arrived", fmt_date(arrival, style=ds)),
                             ("All pulp probes at temperature from", f"day {f['start_delay_days']}"),
                             ("Log ends", f"day {f['voyage_days']}")], rng)
            return (f"COLD TREATMENT LOG SUMMARY\n{head}\n\nIce-point check at loading (offset, C):\n"
                    + table(["probe", "offset"], rows, rng) + "\n\nEvents:\n" + bullets(ev, rng)
                    + f"\n\nOrigin: {FLY_STATUS[f['fly_status']]}.\nPort options: {cs}; {fum}. Shelf life ~{f['shelf_days']} days."
                    + "\n\n" + "\n".join(self._rules(rng, f)))
        para = [f"Reviewed the in-transit cold treatment for {f['entry']} ({f['commodity']} from {f['origin']}, "
                f"{f['vessel']}, importer {f['importer']}).",
                f"Schedule {f['schedule']}. Loaded {fmt_date(f['loaded'], style=ds)}, arrived {fmt_date(arrival, style=ds)}; "
                f"the log runs to day {f['voyage_days']} and every pulp probe was at temperature from day {f['start_delay_days']}.",
                f"Ice-point offsets at loading: {self._probe_text(f)}.",
                "Events: " + "; ".join(ev) + ".",
                f"Origin: {FLY_STATUS[f['fly_status']]}.",
                f"{_cap(cs)}; {fum}; about {f['shelf_days']} days of shelf life left."]
        if f["regime"] == "conflicting":
            para.append(pick(rng, ["The shipping line's reefer technician says the excursion was a probe being re-seated.",
                                   "The importer's agent says the gap was a logger download and the fruit stayed cold.",
                                   "Exporter's certificate states the treatment was completed in transit."]))
        return (f"MEMO -- treatment verification, {f['port']}\nFrom: {f['officer']}\n\n"
                + noisy(join_sentences(para), rng, 0.1) + "\n\nICS:\n" + bullets(self._rules(rng, f), rng))

    def questions(self):
        return {
            "treatment_verdict": choice("What should be done with this consignment given its treatment record?", {
                "accept_treatment": "Accept the in-transit treatment as complete and release the fruit.",
                "extend_treatment_on_arrival": "Complete the remaining treatment days in an approved cold store at the port.",
                "retreat_by_fumigation": "Disregard the cold treatment and fumigate under an approved schedule.",
                "reexport": "Order the consignment re-exported.",
                "destroy": "Order the consignment destroyed.",
            }),
            "log_integrity": score("How trustworthy is the temperature record as evidence of the treatment?", [
                "Untrustworthy: too few valid probes and breaks in the record.",
                "Weak: more than one defect in probes or recording.",
                "Mostly sound: one defect that the rules can absorb.",
                "Sound: valid probes and an unbroken record.",
            ]),
            "fly_risk_if_untreated": score("If the treatment proved ineffective, how likely is this fruit to carry live fruit flies?", [
                "Very unlikely: a poor host from a recognised fly-free district.",
                "Unlikely: limited host suitability or limited fly presence at origin.",
                "Possible: a suitable host from an origin with flies present.",
                "Likely: a preferred host from an origin where flies are widespread.",
            ]),
            "schedule_met": noul(
                "The record shows the cold-treatment schedule was fully met before arrival.",
                yes="With at least three valid pulp probes, the consecutive days at temperature after any restart reach "
                    "the schedule's requirement.",
                no="Too few valid probes, or the count after a restart falls short of the required days."),
            "probe_calibration_acceptable": noul(
                "Enough pulp probes passed their calibration check for the record to be used."),
        }


# =========================================================================== schema 4
# material -> (form, quarantine pathogen named on the permit, RNQP checked visually, tolerance %, genus needs PEQ)
PLANTING = {
    "rose budwood": ("budwood", "Rose rosette virus", "crown gall", 0.5, True),
    "apple rootstocks": ("rootstocks", "Erwinia amylovora", "woolly aphid colonies", 1.0, True),
    "strawberry runners": ("runners", "Strawberry crinkle virus", "Phytophthora crown rot", 0.5, True),
    "potato minitubers": ("minitubers", "Potato spindle tuber viroid", "common scab", 1.0, True),
    "tulip bulbs": ("bulbs", "Ditylenchus dipsaci", "Fusarium basal rot", 2.0, False),
    "lily bulbs": ("bulbs", "Lily mottle virus", "Botrytis lesions", 2.0, False),
    "citrus budwood": ("budwood", "Citrus tristeza virus (severe strains)", "citrus scale on wood", 0.5, True),
    "grapevine cuttings": ("cuttings", "Flavescence doree phytoplasma", "crown gall", 0.5, True),
    "orchid flasks": ("tissue culture flasks", "Odontoglossum ringspot virus", "flask contamination", 1.0, False),
    "tomato seed": ("seed lots", "Tomato brown rugose fruit virus", "seed discolouration", 1.0, False),
    "pepper seed": ("seed lots", "Pepper chat fruit viroid", "seed discolouration", 1.0, False),
    "blueberry plantlets": ("plantlets", "Blueberry scorch virus", "stem canker", 0.5, True),
    "banana plantlets": ("tissue culture plantlets", "Banana bunchy top virus", "off-type plantlets", 1.0, True),
    "olive cuttings": ("cuttings", "Xylella fastidiosa", "olive knot galls", 0.5, True),
    "stone fruit rootstocks": ("rootstocks", "Plum pox virus", "crown gall", 0.5, True),
    "hop rhizomes": ("rhizomes", "Hop stunt viroid", "Verticillium wilt symptoms", 1.0, True),
}
USES = {"nursery_sale": "propagation for sale to commercial nurseries", "own_orchard": "planting in the importer's own orchard",
        "research": "a contained research trial at a university station", "retail": "retail sale to home gardeners"}


class PlantsForPlantingInspection(Schema):
    name = "plants_for_planting_inspection"
    description = "Nursery stock, bulbs or seed imported under permit, checked against tolerances, testing and post-entry quarantine conditions."
    surfaces = {"json": 0.5, "permit_check_form": 0.25, "lab_and_inspection_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        material = pick(rng, list(PLANTING))
        form, qpest, rnqp, tol, peq = PLANTING[material]
        units = pick(rng, [500, 1200, 2500, 6000, 15000, 40000])
        inspected = min(units, pick(rng, [200, 300, 400, 600]))
        rate = near(rng, tol, tol * 0.9, lo=0.0) if chance(rng, 0.85) else tol * between(rng, 2.1, 4.0)
        result = pick_w(rng, {"negative": 6, "positive_confirmed": 1, "positive_screen": 1, "inconclusive": 1})
        pre_export = pick_w(rng, {"declared_correct": 6, "declared_wrong_method": 1, "missing": 1})
        peq_booked = chance(rng, 0.7)
        permit_valid = chance(rng, 0.92)
        n_lots = pick(rng, [1, 2, 3, 4, 6])
        separable = chance(rng, 0.6) and n_lots > 1
        use = pick(rng, list(USES))
        if reg == "clear":
            if chance(rng, 0.55):
                result, pre_export, permit_valid, peq_booked = "negative", "declared_correct", True, True
                rate = between(rng, 0.0, tol * 0.4)
            else:
                result = pick(rng, ["positive_confirmed", "negative"])
                if result == "negative":
                    rate = tol * between(rng, 2.2, 5.0)
        elif reg == "conflicting":
            twist = pick(rng, ["clean_look_positive", "bad_paper_clean_stock", "screen_only"])
            if twist == "clean_look_positive":
                rate, result = between(rng, 0.0, tol * 0.3), pick(rng, ["positive_confirmed", "positive_screen"])
            elif twist == "bad_paper_clean_stock":
                rate, result = between(rng, 0.0, tol * 0.5), "negative"
                pre_export = pick(rng, ["declared_wrong_method", "missing"])
            else:
                result = "positive_screen"
        symptomatic = int(round(inspected * rate / 100))
        affected_lots = 0
        if result == "positive_confirmed" or symptomatic > inspected * tol / 100:
            affected_lots = max(1, min(n_lots, rng.randint(1, max(1, n_lots // 2))))
        return {"regime": reg, "material": material, "form": form, "quarantine_pathogen": qpest, "rnqp": rnqp,
                "tolerance_pct": tol, "peq_required": peq, "units": units, "units_inspected": inspected,
                "symptomatic_units": symptomatic, "lab_result": result, "pre_export_test": pre_export,
                "peq_booked": peq_booked, "permit_valid": permit_valid, "lots": n_lots, "lots_separable": separable,
                "affected_lots": affected_lots, "intended_use": use, "origin": pick(rng, COUNTRIES),
                "port": pick(rng, PORTS), "importer": _importer(rng, "plants"), "inspector": person(rng),
                "permit": ident(rng, "PP", 6, "/"), "entry": _entry(rng), "date": rand_date(rng),
                "importer_note": pick(rng, [None, "urgent: planting window closes in two weeks",
                                            "same supplier as last year, no problems then",
                                            "customer orders already taken for this stock"])}

    def _rate(self, f):
        return 100.0 * f["symptomatic_units"] / f["units_inspected"]

    def _permit_ok(self, f):
        return (f["permit_valid"] and f["pre_export_test"] == "declared_correct"
                and (f["peq_booked"] or not f["peq_required"]))

    def expected(self, f):
        """Intended answers from the latent factors (spread audit only; never rendered)."""
        over = self._rate(f) > f["tolerance_pct"]
        res = f["lab_result"]
        if res == "positive_confirmed" or over:
            act = "destroy_affected_lots" if f["lots_separable"] else "reexport_or_destroy_all"
        elif res in ("positive_screen", "inconclusive") or f["pre_export_test"] != "declared_correct" or not f["permit_valid"]:
            act = "hold_for_testing"
        elif f["peq_required"]:
            act = "release_to_post_entry_quarantine"
        else:
            act = "release"
        if res in ("positive_screen", "inconclusive"):
            follow = "send_to_reference_lab"
        elif f["pre_export_test"] != "declared_correct" and res == "negative":
            follow = "test_retained_sample"
        elif f["peq_required"] and act != "reexport_or_destroy_all":
            follow = "grow_out_in_quarantine"
        else:
            follow = "no_further_testing"
        r = self._rate(f) / f["tolerance_pct"]
        margin = 0 if r <= 0.5 else 1 if r <= 1.0 else 2 if r <= 2.0 else 3
        spread = {"positive_confirmed": 3, "positive_screen": 2, "inconclusive": 1, "negative": 0}[res]
        if over:
            spread += 1
        spread += {"nursery_sale": 1, "retail": 1, "own_orchard": 0, "research": -1}[f["intended_use"]]
        return {"planting_material_action": act, "testing_followup": follow, "tolerance_margin": margin,
                "spread_potential": max(0, min(3, spread)), "permit_conditions_met": "yes" if self._permit_ok(f) else "no"}

    def _rules(self, rng, f):
        peq = (f"post-entry quarantine at a Level 2 facility for {f['form']} of this genus (booking required before arrival)"
               if f["peq_required"] else "no post-entry quarantine for this material")
        return [
            f"Permit condition 1: pre-export testing of every lot for {f['quarantine_pathogen']} by PCR, declared on the "
            "phytosanitary certificate with method and result.",
            f"Permit condition 2: {peq}.",
            pick(rng, [f"ICS 5.4 Regulated non-quarantine pest ({f['rnqp']}): no more than {f['tolerance_pct']}% of units "
                       "inspected may show symptoms.",
                       f"ICS 5.4 Tolerance for {f['rnqp']}: symptoms on at most {f['tolerance_pct']}% of the inspected units."]),
            pick(rng, ["ICS 5.6 A confirmed quarantine pest or a tolerance failure: destroy or re-export the affected lots "
                       "where lots are separately packed and certified, otherwise the whole consignment. Screening "
                       "positives and inconclusive results are held for reference-lab confirmation; missing or wrong "
                       "pre-export testing holds the consignment for testing of the retained sample.",
                       "ICS 5.6 Confirmed detection or tolerance exceeded: affected lots out (whole consignment if lots "
                       "are not separable). Screen-positive or inconclusive: hold for the reference lab. Pre-export "
                       "test absent or wrong method: hold and test the retained sample."]),
        ]

    def _result_text(self, f):
        return {"negative": f"border PCR for {f['quarantine_pathogen']}: negative",
                "positive_confirmed": f"border PCR for {f['quarantine_pathogen']}: POSITIVE (confirmed on a second assay)",
                "positive_screen": f"ELISA screen for {f['quarantine_pathogen']}: positive; PCR not yet run",
                "inconclusive": f"PCR for {f['quarantine_pathogen']}: inconclusive (late amplification, repeat advised)"}[f["lab_result"]]

    def _pre_text(self, f):
        return {"declared_correct": "certificate declares PCR testing of all lots, negative",
                "declared_wrong_method": "certificate declares testing by visual inspection and ELISA, not PCR",
                "missing": "certificate carries no testing declaration"}[f["pre_export_test"]]

    def render(self, f, rng, surface):
        ds = _ds(rng)
        lots = (f"{f['lots']} lots, " + ("separately packed and labelled" if f["lots_separable"] else "mixed in the same boxes")
                if f["lots"] > 1 else "single lot")
        aff = (f"symptoms / positives traced to {f['affected_lots']} of {f['lots']} lots" if f["affected_lots"] and f["lots"] > 1
               else None)
        peq = ("PEQ facility booked" if f["peq_booked"] else "no PEQ booking found") if f["peq_required"] else None
        permit = "permit current" if f["permit_valid"] else "permit expired last month"
        if surface == "json":
            return dumps(prune({
                "consignment": {"entry": f["entry"], "material": f["material"], "units": f["units"], "lots": lots,
                                "origin": f["origin"], "port": f["port"], "importer": f["importer"],
                                "intended_use": USES[f["intended_use"]], "importer_note": f["importer_note"]},
                "permit": {"number": f["permit"], "status": permit, "pre_export_testing": self._pre_text(f),
                           "post_entry_quarantine": peq},
                "inspection": {"inspector": f["inspector"], "units_inspected": f["units_inspected"],
                               "units_with_symptoms": f"{f['symptomatic_units']} ({f['rnqp']})",
                               "lab": self._result_text(f), "lots_affected": aff},
                "rules": self._rules(rng, f),
            }))
        if surface == "permit_check_form":
            head = kv_lines([("Entry", f["entry"]), ("Permit", f"{f['permit']} ({permit})"), ("Port", f["port"]),
                             ("Date", fmt_date(f["date"], style=ds)), ("Material", f"{f['material']}, {f['units']:,} units"),
                             ("Lots", lots), ("Origin", f["origin"]), ("Importer", f["importer"]),
                             ("Intended use", USES[f["intended_use"]]), ("Pre-export testing", self._pre_text(f)),
                             ("PEQ", peq or "not required"), ("Inspected", f"{f['units_inspected']} units"),
                             (f"Symptoms ({f['rnqp']})", f["symptomatic_units"]), ("Lab", self._result_text(f)),
                             ("Lots affected", aff)], rng)
            return f"PLANTS FOR PLANTING -- PERMIT AND INSPECTION CHECK\n{head}\n\n" + "\n".join(self._rules(rng, f))
        para = [f"{f['entry']}: {f['units']:,} {f['material']} ({lots}) from {f['origin']} for {f['importer']}, "
                f"intended for {USES[f['intended_use']]}.",
                f"Permit {f['permit']}: {permit}; {self._pre_text(f)}." + (f" {peq}." if peq else ""),
                f"Inspected {f['units_inspected']} units: {f['symptomatic_units']} with {f['rnqp']} symptoms.",
                _cap(self._result_text(f)) + "."]
        if aff:
            para.append(_cap(aff) + ".")
        if f["importer_note"]:
            para.append(f"Importer's note: {f['importer_note']}.")
        return "\n".join([f"Subject: {f['entry']} {f['material']} -- inspection and lab results", f"From: {f['inspector']}",
                          "", noisy(join_sentences(para), rng, 0.1), "", "Conditions:", bullets(self._rules(rng, f), rng)])

    def questions(self):
        return {
            "planting_material_action": choice("What should happen to this planting material?", {
                "release": "Release it to the importer.",
                "release_to_post_entry_quarantine": "Release it only into the booked post-entry quarantine facility.",
                "hold_for_testing": "Hold it under seal until outstanding testing is resolved.",
                "destroy_affected_lots": "Destroy or re-export the affected lots and let the rest proceed.",
                "reexport_or_destroy_all": "Re-export or destroy the whole consignment.",
            }),
            "testing_followup": choice("What further testing, if any, should be arranged?", {
                "no_further_testing": "None beyond what is already done.",
                "test_retained_sample": "Test the retained sample at the border laboratory by the permit's method.",
                "grow_out_in_quarantine": "Grow the material out in post-entry quarantine and test during that period.",
                "send_to_reference_lab": "Send material to the national reference laboratory for confirmation.",
            }),
            "tolerance_margin": score("How do the symptoms seen compare with the tolerance for the regulated non-quarantine pest?", [
                "Well within: under half the tolerance.",
                "Within, but close to the limit.",
                "Over the tolerance, by up to double.",
                "Far over: more than double the tolerance.",
            ]),
            "spread_potential": score("If released without further measures, how much potential does this consignment have to spread a regulated pest?", [
                "Minimal: clean results and a contained end use.",
                "Limited: an open question, but contained or low-volume use.",
                "Considerable: a likely pest presence or wide distribution of the material.",
                "Severe: a confirmed regulated pest in material headed for wide propagation.",
            ]),
            "permit_conditions_met": noul(
                "The consignment meets the conditions of its import permit.",
                yes="The permit is current, pre-export testing is declared by the required method, and any required "
                    "post-entry quarantine is booked.",
                no="The permit has lapsed, the testing declaration is missing or uses the wrong method, or required "
                   "quarantine space is not booked."),
        }


# =========================================================================== schema 5
# grain -> end uses it can plausibly be imported for
GRAINS = {"wheat": ["milling", "feed_flaked", "feed_whole", "sowing"], "barley": ["malting", "feed_flaked", "feed_whole"],
          "maize": ["milling", "feed_flaked", "feed_whole", "sowing"], "sorghum": ["feed_flaked", "feed_whole"],
          "soybeans": ["crushing", "crushing", "feed_whole", "sowing"], "canola": ["crushing", "crushing", "sowing"],
          "oats": ["milling", "feed_whole"], "field peas": ["splitting", "feed_whole", "sowing"],
          "lentils": ["splitting", "splitting", "sowing"], "rice (paddy)": ["milling"]}
WEEDS = ["Ambrosia trifida", "Sorghum halepense", "Striga asiatica", "Cuscuta spp.", "Solanum elaeagnifolium",
         "Ambrosia artemisiifolia", "Avena sterilis", "Cenchrus echinatus", "Xanthium spinosum", "Orobanche ramosa"]
STORAGE_INSECTS = ["Sitophilus oryzae", "Tribolium castaneum", "Rhyzopertha dominica", "Oryzaephilus surinamensis",
                   "Cryptolestes ferrugineus", "Sitophilus granarius"]
PREV_CARGO = {"clean": ["wheat from a regular trade partner", "fertiliser (urea)", "coal", "steel billets", "sugar"],
              "concern": ["groundnuts from a port with recorded khapra beetle", "rice from a region under khapra-beetle "
                          "measures", "used jute sacks", "sorghum rejected at another port for weed seeds"]}
END_USE = {"milling": ("milling (devitalises seed)", True), "feed_flaked": ("steam-flaking for stockfeed (devitalises)", True),
           "feed_whole": ("whole-grain stockfeed (no devitalising step)", False),
           "crushing": ("oilseed crushing (devitalises)", True), "sowing": ("re-sale as planting seed", False),
           "malting": ("malting (germinates the grain; does not devitalise weed seed)", False),
           "splitting": ("dehulling and splitting (devitalises)", True)}


class BulkGrainContaminant(Schema):
    name = "bulk_grain_contaminant"
    description = "A bulk grain vessel sampled on arrival for weed seeds, soil and storage insects: discharge, restrict, fumigate, resample or refuse."
    surfaces = {"json": 0.5, "sampling_report": 0.25, "vessel_notes": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        grain = pick(rng, list(GRAINS))
        tonnes = pick(rng, [8000, 12500, 18000, 24000, 31000, 42000, 55000])
        req = math.ceil(tonnes / 1000 / 2)            # 1 composite per 2,000 t or part
        taken = req + pick(rng, [0, 0, 0, 0, 1, 1, -1, -2])
        increments = pick_w(rng, {10: 4, 12: 4, 15: 2, 8: 1, 6: 0.5})
        tol = 2
        weed_level = near(rng, tol, 1.6, lo=0.0)
        insects = pick_w(rng, {"none": 6, "common_live": 2, "trogoderma": 1})
        soil = round(near(rng, 0.08, 0.06, lo=0.0), 2)
        end_use = pick(rng, GRAINS[grain])
        processor = chance(rng, 0.7)
        prev = pick_w(rng, {"clean": 4, "concern": 1})
        hold_cert = chance(rng, 0.8)
        residue = chance(rng, 0.15)
        if reg == "clear":
            if chance(rng, 0.5):
                taken, increments, weed_level, insects, soil = req + 1, 12, between(rng, 0.0, 0.6), "none", round(between(rng, 0.0, 0.04), 2)
                prev, hold_cert, residue = "clean", True, False
            else:
                insects = pick(rng, ["trogoderma", "common_live"])
        if reg == "borderline" and chance(rng, 0.25):
            taken = req - 1 if chance(rng, 0.6) else req
            increments = pick(rng, [8, 9, 10])
        if reg == "borderline" and chance(rng, 0.3):
            weed_level, processor = between(rng, 2.6, 6.0), True
            end_use = pick(rng, [u for u in GRAINS[grain] if u != "sowing"])
        elif reg == "conflicting":
            twist = pick(rng, ["clean_sample_bad_hold", "sowing_use", "thin_sampling", "insects"])
            if twist == "clean_sample_bad_hold":
                weed_level, insects, prev, residue = between(rng, 0.0, 1.0), "none", "concern", chance(rng, 0.6)
                hold_cert = chance(rng, 0.4)
            elif twist == "sowing_use":
                grain = pick(rng, [g for g, u in GRAINS.items() if "sowing" in u])
                end_use, weed_level = "sowing", between(rng, 2.5, 6.0)
            elif twist == "thin_sampling":
                taken, weed_level = max(1, pick(rng, [req // 3, req // 2, req - 1, req - 2])), between(rng, 0.0, 1.5)
                increments = pick(rng, [5, 6, 8, 10])
            else:
                insects, weed_level = "common_live", between(rng, 2.5, 5)
        taken = max(1, taken)
        comps = [max(0, int(round(weed_level * between(rng, 0.4, 1.6)))) for _ in range(taken)]
        insect_comps = rng.randint(1, taken)
        return {"regime": reg, "grain": grain, "tonnes": tonnes, "composites_required": req, "weed_counts_per_kg": comps,
                "increments_per_composite": increments, "weed_species": pick(rng, WEEDS), "tolerance_per_kg": tol,
                "insects": insects, "insect_species": pick(rng, STORAGE_INSECTS), "insect_composites": insect_comps, "soil_pct": soil, "end_use": end_use,
                "processor_available": processor, "previous_cargo_kind": prev,
                "previous_cargo": pick(rng, PREV_CARGO[prev]), "hold_cleanliness_certificate": hold_cert,
                "residue_seen": residue, "origin": pick(rng, COUNTRIES), "port": pick(rng, BULK_BERTHS),
                "vessel": f"MV {pick(rng, ['Grain', 'Ocean', 'Sea', 'Prairie', 'Harvest', 'Great'])} "
                          f"{pick(rng, ['Spirit', 'Venture', 'Pioneer', 'Glory', 'Hope', 'Majesty', 'Runner'])}",
                "holds": pick(rng, [3, 4, 5, 6, 7]), "officer": person(rng), "importer": _importer(rng, "grain"),
                "entry": _entry(rng), "date": rand_date(rng), "agent_view": pick(rng, [None, "push", "push", None])}

    def _sampling_ok(self, f):
        return len(f["weed_counts_per_kg"]) >= f["composites_required"] and f["increments_per_composite"] >= 10

    def expected(self, f):
        """Intended answers from the latent factors (spread audit only; never rendered)."""
        mx = max(f["weed_counts_per_kg"])
        tol = f["tolerance_per_kg"]
        weed_ok = mx <= tol
        soil_ok = f["soil_pct"] <= 0.1
        devit = END_USE[f["end_use"]][1]
        over = not weed_ok or not soil_ok
        refusable = over and (f["end_use"] == "sowing" or mx > 10 * tol or not f["processor_available"])
        if f["insects"] == "trogoderma" or refusable:
            act = "refuse_discharge"            # the samples already in hand settle it; more sampling cannot cure it
        elif f["insects"] == "common_live":
            act = "fumigate_in_hold"
        elif not self._sampling_ok(f):
            act = "resample_before_decision"
        elif over:
            act = "discharge_to_approved_processor"
        else:
            act = "discharge_unrestricted"
        short = f["composites_required"] - len(f["weed_counts_per_kg"])
        samp = 3
        if short >= 1:
            samp -= 1
        if short >= max(2, f["composites_required"] / 3):
            samp -= 1
        if f["increments_per_composite"] < 10:
            samp -= 1
        if f["increments_per_composite"] <= 6:
            samp -= 1
        sev = 0
        if mx > tol * 3 or f["insects"] == "trogoderma":
            sev = 3
        elif not weed_ok or not soil_ok or f["insects"] == "common_live":
            sev = 2
        elif mx >= tol / 2 or f["soil_pct"] >= 0.05:
            sev = 1
        concern = f["previous_cargo_kind"] == "concern" or f["residue_seen"] or not f["hold_cleanliness_certificate"]
        return {"bulk_cargo_action": act, "sampling_adequacy": max(0, samp), "contaminant_severity": sev,
                "within_weed_seed_tolerance": "yes" if weed_ok else "no",
                "hold_history_concern": "yes" if concern else "no"}

    def _rules(self, rng, f):
        return [
            pick(rng, ["ICS 11.2 Sampling: one composite per 2,000 t or part, each built from at least 10 increments "
                       "drawn across the holds during discharge or by probe.",
                       "ICS 11.2 At least one composite sample for every 2,000 t (or part); every composite needs 10 or "
                       "more increments taken across the holds."]),
            pick(rng, ["ICS 11.4 Tolerances per composite: Schedule 6 weed seeds no more than 2 per kg; soil no more "
                       "than 0.1% by weight.",
                       "ICS 11.4 No composite may exceed 2 Schedule 6 weed seeds per kg or 0.1% soil by weight."]),
            "ICS 11.5 Live common storage insects: fumigate in the hold before discharge. Trogoderma larvae, cast skins "
            "or adults not identified to a non-regulated species: discharge refused.",
            pick(rng, ["ICS 11.6 Grain over tolerance may be discharged only to an approved processor whose process "
                       "devitalises seed, and never if intended for sowing or if any composite exceeds ten times the "
                       "weed-seed tolerance; otherwise discharge is refused.",
                       "ICS 11.6 Over-tolerance grain: discharge only to an approved devitalising processor; refused if "
                       "for sowing, over 10x the weed-seed tolerance, or no approved processor is available."]),
        ]

    def _insect_text(self, f):
        return {"none": "no live insects in sieved samples",
                "common_live": f"live {f['insect_species']} in {f['insect_composites']} of "
                               f"{len(f['weed_counts_per_kg'])} composites",
                "trogoderma": "Trogoderma larval cast skins in hold 2 sweepings; species not determined"}[f["insects"]]

    def render(self, f, rng, surface):
        ds = _ds(rng)
        comps = f["weed_counts_per_kg"]
        use = END_USE[f["end_use"]][0]
        proc = "approved processor has capacity" if f["processor_available"] else "no approved processor slot this month"
        hold = ("hold-cleanliness certificate from load port attached" if f["hold_cleanliness_certificate"]
                else "no hold-cleanliness certificate supplied")
        res = "old residue caked on hold 3 frames" if f["residue_seen"] else None
        if surface == "json":
            return dumps(prune({
                "vessel": {"name": f["vessel"], "entry": f["entry"], "port": f["port"], "origin": f["origin"],
                           "grain": f["grain"], "tonnes": f["tonnes"], "holds": f["holds"], "importer": f["importer"],
                           "end_use": use, "previous_cargo": f["previous_cargo"], "hold_certificate": hold,
                           "hold_observation": res},
                "sampling": {"composites_taken": len(comps), "increments_per_composite": f["increments_per_composite"],
                             "weed_seeds_per_kg_by_composite": comps, "weed_species": f["weed_species"],
                             "soil_pct_by_weight": f["soil_pct"], "insects": self._insect_text(f)},
                "processing": proc,
                "rules": self._rules(rng, f),
            }))
        if surface == "sampling_report":
            rows = [[f"C{i + 1}", str(c)] for i, c in enumerate(comps)]
            head = kv_lines([("Vessel", f"{f['vessel']} ({f['holds']} holds)"), ("Entry", f["entry"]),
                             ("Port / date", f"{f['port']} / {fmt_date(f['date'], style=ds)}"),
                             ("Cargo", f"{f['tonnes']:,} t {f['grain']} from {f['origin']}"), ("End use", use),
                             ("Increments per composite", f["increments_per_composite"]),
                             ("Weed species", f["weed_species"]), ("Soil", f"{f['soil_pct']}%"),
                             ("Insects", self._insect_text(f)), ("Previous cargo", f["previous_cargo"]),
                             ("Hold", hold + (f"; {res}" if res else "")), ("Processor", proc)], rng)
            return (f"BULK GRAIN SAMPLING REPORT -- {AUTHORITY}\n{head}\n\nWeed seeds per kg:\n"
                    + table(["composite", "seeds/kg"], rows, rng) + "\n\n" + "\n".join(self._rules(rng, f)))
        para = [f"{f['vessel']} at {f['port']} with {f['tonnes']:,} t of {f['grain']} from {f['origin']} for {f['importer']}; "
                f"declared end use {use}.",
                f"Composites: {len(comps)}, {f['increments_per_composite']} increments each; Schedule 6 seeds "
                f"({f['weed_species']}) per kg: {', '.join(str(c) for c in comps)}.",
                f"Soil {f['soil_pct']}% by weight. {_cap(self._insect_text(f))}.",
                f"Previous cargo: {f['previous_cargo']}; {hold}." + (f" {_cap(res)}." if res else ""),
                _cap(proc) + "."]
        if f["agent_view"] == "push":
            para.append(pick(rng, ["Ship's agent says demurrage is running and wants discharge to start tonight.",
                                   "Agent argues the weed seeds are 'within normal trade standards'."]))
        return (f"Duty log {fmt_date(f['date'], style=ds)} -- {f['officer']}\n\n" + noisy(join_sentences(para), rng, 0.1)
                + "\n\nICS:\n" + bullets(self._rules(rng, f), rng))

    def questions(self):
        return {
            "bulk_cargo_action": choice("What should be done with this vessel's cargo?", {
                "discharge_unrestricted": "Allow discharge with no restriction on where the grain goes.",
                "discharge_to_approved_processor": "Allow discharge only to an approved processor that devitalises seed.",
                "fumigate_in_hold": "Fumigate in the holds before any discharge.",
                "resample_before_decision": "Take further samples before deciding.",
                "refuse_discharge": "Refuse discharge of the cargo.",
            }),
            "sampling_adequacy": score("How adequate is the sampling as a basis for a decision on this cargo?", [
                "Inadequate: far too few composites or increments.",
                "Weak: short on more than one requirement.",
                "Nearly adequate: one requirement narrowly missed.",
                "Adequate: enough composites, each properly built.",
            ]),
            "contaminant_severity": score("How serious is the contamination found in the cargo?", [
                "Negligible: little or nothing found.",
                "Minor: present but within tolerance.",
                "Significant: over a tolerance, or live common insects.",
                "Severe: far over tolerance, or a suspected regulated storage pest.",
            ]),
            "within_weed_seed_tolerance": noul(
                "The weed-seed counts are within the tolerance in the import conditions.",
                yes="No composite exceeds the permitted number of Schedule 6 seeds per kilogram.",
                no="At least one composite exceeds it."),
            "hold_history_concern": noul(
                "The vessel's previous cargo or the condition of its holds raises an additional concern."),
        }


WORKFLOW = Workflow(
    name="plant_pest_quarantine_inspection",
    title="Plant pest quarantine inspection",
    description="Border plant-health decisions on import consignments (release, treat, re-export, destroy, hold), judged "
                "against a fictional authority's written import conditions.",
    domain="agriculture / plant biosecurity",
    schemas=[ConsignmentInterception(), WoodPackagingCheck(), TreatmentRecordVerification(),
             PlantsForPlantingInspection(), BulkGrainContaminant()],
    notes="Fictional authority (Directorate of Plant Biosecurity) and Import Conditions Schedule; real pest taxa with "
          "invented regulatory status. Phytosanitary measures only: no tariff, valuation or origin questions, no bills.",
)
