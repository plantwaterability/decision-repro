"""Crop insurance claims: loss-adjustment decisions on a multi-peril crop policy, made against its written terms.

A (fictional) crop insurer's loss adjusters and claims supervisors rule on yield-loss claims: whether the damage
came from an insured cause, whether an appraisal of a damaged crop is good enough to release the acreage for
another use, whether a replanted field earns a replant payment, whether unplanted acres qualify as prevented
planting, and whether mycotoxin-damaged grain earns a quality adjustment. Every state quotes the provisions of the
insurer's policy that govern the decision (an invented "MPC-7" multi-peril crop policy with invented crop
provisions and thresholds), so the answer follows from the written terms plus the evidence in the file.

Each schema draws a `regime` first: most cases sit near a written threshold (a stand count, a sample count, a toxin
level, an acreage cap), carry an adjuster's finding that cuts against the insured's account, or leave a key record
pending. Insurers, farms, counties, stations and people are fictional. Nothing here bills, invoices or pays anyone,
and no conversation with a policyholder is drafted or triaged.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

from .base import (
    LAST,
    ORG_A,
    Schema,
    Workflow,
    between,
    bullets,
    chance,
    choice,
    dumps,
    fmt_date,
    ident,
    join_sentences,
    kv_lines,
    maybe,
    money,
    near,
    noisy,
    noul,
    org,
    person,
    pick,
    pick_w,
    prune,
    regime,
    score,
    table,
)

COUNTIES = ["Ashby", "Belmont", "Calder", "Dunmore", "Ellery", "Fenwick", "Garnet", "Halloway", "Ingram", "Jessup",
            "Kinross", "Lamont", "Merrow", "Norland", "Ostrander", "Pelham", "Quarles", "Renwick", "Stanhope",
            "Tolliver", "Upham", "Varden", "Wexley", "Yardley", "Arden", "Brule", "Coldwater", "Driscoll"]
STATION_KIND = ["airport ASOS", "ag research farm", "mesonet site", "co-op observer station", "extension weather station",
                "cooperative gauge", "research station"]
INSURER_SUFFIX = ["Crop Mutual", "Crop Insurance", "Farm Mutual", "Agricultural Assurance", "Mutual", "Crop Assurance",
                  "Rural Insurance", "Farmers Mutual"]

# crop -> unit, approved-yield range, projected price range ($/unit), final planting date (month, day)
CROPS = {
    "corn": dict(unit="bu", aph=(140, 215), price=(4.10, 4.85), fpd=(5, 31)),
    "soybeans": dict(unit="bu", aph=(42, 64), price=(10.10, 11.60), fpd=(6, 10)),
    "grain sorghum": dict(unit="bu", aph=(68, 105), price=(3.90, 4.60), fpd=(6, 20)),
    "spring wheat": dict(unit="bu", aph=(40, 68), price=(6.10, 7.30), fpd=(5, 31)),
    "winter wheat": dict(unit="bu", aph=(45, 78), price=(5.90, 7.10), fpd=(10, 15)),
    "barley": dict(unit="bu", aph=(58, 90), price=(4.90, 5.90), fpd=(5, 31)),
    "oats": dict(unit="bu", aph=(60, 95), price=(3.40, 4.10), fpd=(5, 31)),
    "upland cotton": dict(unit="lb", aph=(650, 1150), price=(0.68, 0.84), fpd=(6, 10)),
    "sunflowers": dict(unit="lb", aph=(1350, 1950), price=(0.21, 0.27), fpd=(6, 10)),
    "canola": dict(unit="lb", aph=(1550, 2250), price=(0.19, 0.25), fpd=(6, 5)),
    "dry beans": dict(unit="lb", aph=(1700, 2500), price=(0.30, 0.42), fpd=(6, 10)),
    "peanuts": dict(unit="lb", aph=(3600, 4700), price=(0.23, 0.27), fpd=(6, 5)),
    "rice": dict(unit="lb", aph=(6800, 8300), price=(0.14, 0.17), fpd=(5, 25)),
    "popcorn": dict(unit="lb", aph=(4200, 5600), price=(0.14, 0.18), fpd=(5, 31)),
    "field peas": dict(unit="bu", aph=(35, 55), price=(6.50, 8.40), fpd=(5, 25)),
}
SPRING_CROPS = [c for c in CROPS if c != "winter wheat"]


def _county(rng):
    return f"{pick(rng, COUNTIES)} County"


def _insured(rng):
    style = pick(rng, ["person", "farms", "llc", "partnership", "person"])
    if style == "person":
        return person(rng)
    ln = pick(rng, LAST)
    if style == "farms":
        return f"{ln} {pick(rng, ['Family Farms', 'Farms', 'Grain', 'Acres', 'Ag'])}"
    if style == "llc":
        return f"{ln} {pick(rng, ['Land & Cattle', 'Farms', 'Brothers Farming', 'Ridge Farms'])} LLC"
    return f"{ln}-{pick(rng, LAST)} {pick(rng, ['Partnership', 'Farm Partnership', 'JV'])}"


def _insurer(rng):
    if chance(rng, 0.3):
        return org(rng, "insurer")
    return f"{pick(rng, ORG_A)} {pick(rng, INSURER_SUFFIX)}"


def _unit_no(rng):
    return f"{rng.randint(1, 9):04d}-{rng.randint(1, 14):04d}-{pick(rng, ['BU', 'OU', 'OU', 'EU'])}"


def _qty(x, unit):
    return f"{x:,.0f} {unit}" if unit != "bu" or x >= 100 else f"{x:.1f} {unit}"


def _ds(rng):
    """One date style per document."""
    return pick(rng, ["iso", "iso", "us", "long"])


def _cap(s):
    return s[0].upper() + s[1:] if s else s


def _pct(x):
    return f"{x * 100:.0f}%"


CROP_ADJ = {"soybeans": "soybean", "dry beans": "dry bean", "sunflowers": "sunflower", "field peas": "field pea",
            "peanuts": "peanut"}


def _adj(crop):
    """The crop name as a modifier: 'soybean fields', not 'soybeans fields'."""
    return CROP_ADJ.get(crop, crop)


def _d(x):
    """A date from a factor value: a date, or its ISO string as stored in meta.factors."""
    return date.fromisoformat(x) if isinstance(x, str) else x


# =========================================================================== schema 1
CAUSE_LABEL = {"drought": "drought", "excess_moisture": "excess moisture", "hail": "hail", "freeze": "freeze",
               "wind": "wind / lodging", "plant_disease": "plant disease", "insects": "insect damage",
               "wildlife": "wildlife"}
DISEASES = ["tar spot", "gray leaf spot", "white mold", "Fusarium head blight", "stripe rust", "sudden death syndrome",
            "Sclerotinia stem rot", "bacterial leaf streak", "anthracnose stalk rot", "Verticillium wilt",
            "southern rust", "frogeye leaf spot"]
INSECTS = ["western bean cutworm", "soybean aphid", "grasshoppers", "wheat stem sawfly", "corn rootworm", "spider mites",
           "fall armyworm", "lygus bugs", "stink bugs", "cotton bollworm", "banded sunflower moth", "flea beetles"]
WILDLIFE = ["feral hogs", "elk", "white-tailed deer", "Canada geese", "blackbirds", "sandhill cranes", "raccoons",
            "pronghorn"]
# practice issue -> (kind, causes it fits, text variants with {share_hint})
ISSUES = {
    "fungicide_skipped": ("gfp", ("plant_disease", "wind"), [
        "no fungicide went on although the county advisory recommended a pass at the susceptible stage",
        "the insured skipped the fungicide pass that the extension alert called for three weeks before symptoms"]),
    "insecticide_late": ("gfp", ("insects",), [
        "scouting records show counts above the treatment threshold for about two weeks before anyone sprayed",
        "the insured's own scout flagged threshold counts, but the field was not treated until the damage was done"]),
    "nitrogen_short": ("gfp", ("drought", "excess_moisture", "hail", "wind", "freeze"), [
        "nitrogen was applied at roughly half the rate in the insured's own nutrient plan",
        "the side-dress nitrogen pass was never made; tissue tests show a deficiency across the field"]),
    "weeds_uncontrolled": ("gfp", ("drought", "excess_moisture", "hail"), [
        "heavy waterhemp pressure; the post-emergence herbicide pass was never made",
        "grass and broadleaf escapes across much of the field; no post-emergence application on record"]),
    "seeding_rate_low": ("gfp", ("drought", "hail", "wildlife", "freeze"), [
        "the planter was set well below the seeding rate recommended for this variety",
        "seeding rate was about two thirds of the agronomic recommendation for these soils"]),
    "pivot_breakdown": ("equipment", ("drought",), [
        "the centre pivot was out of service for about two weeks in July with a failed gearbox",
        "the irrigation well pump failed mid-season and was not repaired for 17 days"]),
    "tile_outlet": ("field", ("excess_moisture",), [
        "the worst damage sits in a low area where the tile outlet has been plugged since last season",
        "the drowned-out area matches a collapsed tile line the insured had known about since spring"]),
    "harvest_delay": ("delay", ("wind", "wildlife", "hail"), [
        "the crop was ready about five weeks before harvest began, with no weather reason recorded for the delay",
        "harvest started well after neighbours finished; the insured says the combine was tied up elsewhere"]),
}
DISPUTE_TEXT = {
    "gfp": ["The insured says the agronomist advised against it this year and wants that on the file.",
            "The insured disagrees and says the practice is normal for the area.",
            "Insured disputes the finding: 'every neighbour farms it the same way'."],
    "equipment": ["The insured says the repair was delayed by the dealer, not by him.",
                  "Insured argues the breakdown itself was caused by the heat."],
    "field": ["The insured says the tile problem only matters in a wet year like this one.",
              "Insured disputes that the low spot is any worse than in a normal year."],
    "delay": ["The insured says custom harvesters were booked solid.",
              "Insured says the delay made no difference to the loss."],
}
SUP = {"contradicts": 0, "weak": 1, "partial": 2, "strong": 3}


def _weather_text(cause, sup, w, crop):
    """Station / area record text for a cause at a support level (numbers pre-sampled in w)."""
    if cause == "drought":
        return (f"{w['station']}: Jun-Aug rainfall {w['val']}% of the 30-year normal; "
                f"{w['n']} days above 35 C")
    if cause == "excess_moisture":
        return (f"{w['station']}: rainfall {w['val']}% of normal for the 30 days after planting; "
                f"{w['n']} days with standing water reported in the township")
    if cause == "hail":
        if sup in ("strong", "partial"):
            return f"storm reports: hail of {w['size']} cm reported about {w['km']} km from the unit on the damage date"
        if sup == "weak":
            return f"storm reports: nearest hail report {w['km']} km away, two days before the damage date"
        return "storm reports: no hail or thunderstorm activity logged within 80 km in the ten days before the damage date"
    if cause == "freeze":
        return f"{w['station']}: overnight low of {w['val']:.1f} C on the night the damage was found"
    if cause == "wind":
        return f"{w['station']}: peak gust {w['val']} km/h during the reported storm"
    if cause == "plant_disease":
        d = w["agent"]
        return {"strong": f"extension bulletin confirms a widespread {d} outbreak in the county; lab confirmed {d} on samples from this unit",
                "partial": f"scattered {d} reports in the county; field symptoms consistent with {d}",
                "weak": f"no county reports of {d}; symptoms described only by the insured",
                "contradicts": f"lab result on samples from this unit: no {d} found; leaf pattern consistent with nutrient deficiency"}[sup]
    if cause == "insects":
        d = w["agent"]
        return {"strong": f"extension trap counts for {d} far above threshold county-wide; feeding confirmed on the unit",
                "partial": f"{d} reported in parts of the county; feeding signs on the unit",
                "weak": f"no county reports of {d}; only a few feeding signs seen",
                "contradicts": f"no {d} found in sweeps or plant checks; damage pattern does not match feeding"}[sup]
    d = w["agent"]
    return {"strong": f"wildlife agency depredation report: {d} damage verified on this and two adjoining farms",
            "partial": f"{d} sightings reported in the township; tracks and feeding at the field edge",
            "weak": f"no depredation reports; insured describes {d} at night",
            "contradicts": f"agency officer found no sign of {d}; losses spread evenly across the field interior"}[sup]


class CauseOfLossDetermination(Schema):
    name = "cause_of_loss_determination"
    description = "A yield-loss claim where the adjuster must rule whether the shortfall came from an insured cause."
    surfaces = {"json": 0.5, "adjuster_worksheet": 0.2, "claim_notes": 0.15, "supervisor_email": 0.15}

    def sample_factors(self, rng):
        reg = regime(rng)
        crop = pick(rng, list(CROPS))
        c = CROPS[crop]
        cause = pick(rng, list(CAUSE_LABEL))
        sup = pick_w(rng, {"strong": 3, "partial": 4, "weak": 2, "contradicts": 1})
        issue, share, disputed, pending, excuse = None, 0.0, False, None, None
        nb_shift = 0
        notice_days = pick(rng, [0, 1, 1, 2, 2, 3, 4, 5])
        scout_before = None
        if reg == "clear":
            arche = pick_w(rng, {"pay": 3, "deny": 3, "less": 2, "hold": 1})
            if arche == "pay":
                sup, notice_days = "strong", pick(rng, [0, 1, 2])
            elif arche == "deny":
                sup = "contradicts"
                fits = [k for k, v in ISSUES.items() if cause in v[1]]
                issue = pick(rng, fits) if fits else None
                share = between(rng, 0.55, 0.9) if issue else 0.0
            elif arche == "less":
                sup = pick(rng, ["strong", "partial"])
                fits = [k for k, v in ISSUES.items() if cause in v[1]]
                issue = pick(rng, fits) if fits else None
                share = between(rng, 0.22, 0.42) if issue else 0.0
            else:
                sup = pick(rng, ["weak", "partial"])
                pending = "lab_id" if cause in ("plant_disease", "insects") else "station_summary"
        else:
            if chance(rng, 0.6):
                fits = [k for k, v in ISSUES.items() if cause in v[1]]
                if fits:
                    issue = pick(rng, fits)
                    share = near(rng, pick(rng, [0.04, 0.2, 0.2, 0.5, 0.85]), 0.12, lo=0.01, hi=0.97)
                    disputed = chance(rng, 0.5)
            if chance(rng, 0.18):
                pending = "lab_id" if cause in ("plant_disease", "insects") and chance(rng, 0.7) else "station_summary"
            notice_days = near(rng, 3, 2, lo=0, hi=7, integer=True)
            if reg == "conflicting":
                twist = pick(rng, ["gfp_dispute", "excuse", "neighbours", "late_scout"])
                if twist == "gfp_dispute":
                    fits = [k for k, v in ISSUES.items() if cause in v[1] and v[0] == "gfp"]
                    if fits:
                        issue, share, disputed = pick(rng, fits), between(rng, 0.22, 0.6), True
                        sup = pick(rng, ["strong", "partial"])
                elif twist == "excuse":
                    cause, issue, disputed = "excess_moisture", "weeds_uncontrolled", True
                    share = between(rng, 0.15, 0.45)
                    excuse = "too_wet_to_spray"
                    sup = pick(rng, ["strong", "partial", "weak", "contradicts"])
                elif twist == "neighbours":
                    nb_shift = pick(rng, [-1, 1])
                    sup = "contradicts" if nb_shift < 0 else "strong"
                else:
                    scout_before = rng.randint(4, 9)
                    notice_days = pick(rng, [0, 1, 2])
        if issue is None:
            share, disputed = 0.0, False
        # weather numbers for the final cause / support level
        station = f"{pick(rng, COUNTIES)} {pick(rng, STATION_KIND)}"
        w = {"station": station}
        if cause == "drought":
            w["val"] = {"strong": rng.randint(25, 55), "partial": rng.randint(58, 80), "weak": rng.randint(82, 98),
                        "contradicts": rng.randint(104, 140)}[sup]
            w["n"] = {"strong": rng.randint(9, 20), "partial": rng.randint(4, 10), "weak": rng.randint(1, 5),
                      "contradicts": rng.randint(0, 2)}[sup]
        elif cause == "excess_moisture":
            w["val"] = {"strong": rng.randint(185, 270), "partial": rng.randint(135, 175), "weak": rng.randint(105, 125),
                        "contradicts": rng.randint(60, 95)}[sup]
            w["n"] = {"strong": rng.randint(10, 20), "partial": rng.randint(4, 9), "weak": rng.randint(1, 3),
                      "contradicts": 0}[sup]
        elif cause == "hail":
            w["km"] = {"strong": rng.randint(1, 6), "partial": rng.randint(10, 25), "weak": rng.randint(30, 60),
                       "contradicts": None}[sup]
            w["size"] = {"strong": round(rng.uniform(2.5, 5.0), 1), "partial": round(rng.uniform(1.0, 2.2), 1)}.get(sup)
        elif cause == "freeze":
            w["val"] = {"strong": rng.uniform(-4.5, -2.2), "partial": rng.uniform(-1.8, -0.8),
                        "weak": rng.uniform(-0.5, 1.0), "contradicts": rng.uniform(2.5, 6.0)}[sup]
        elif cause == "wind":
            w["val"] = {"strong": rng.randint(95, 130), "partial": rng.randint(70, 90), "weak": rng.randint(50, 65),
                        "contradicts": rng.randint(25, 45)}[sup]
        else:
            w["agent"] = pick(rng, {"plant_disease": DISEASES, "insects": INSECTS, "wildlife": WILDLIFE}[cause])
        nb = {"strong": rng.randint(45, 75), "partial": rng.randint(65, 85), "weak": rng.randint(85, 97),
              "contradicts": rng.randint(94, 108)}[sup]
        if nb_shift > 0:
            nb = rng.randint(93, 104)
        elif nb_shift < 0:
            nb = rng.randint(50, 64)
        aph = rng.randint(*c["aph"])
        cov = pick(rng, [0.65, 0.70, 0.75, 0.75, 0.80, 0.85])
        guar = aph * cov
        ratio = between(rng, 0.3, 0.9)
        yield_tc = round(guar * ratio, 1) if c["unit"] == "bu" else int(round(guar * ratio))
        month = pick(rng, {"freeze": [5, 9, 9], "hail": [6, 7, 8], "wind": [7, 8, 9], "drought": [7, 8, 9],
                           "excess_moisture": [5, 6, 7], "plant_disease": [7, 8, 9], "insects": [6, 7, 8, 9],
                           "wildlife": [6, 7, 8, 9]}[cause])
        found = date(pick(rng, [2025, 2026]), month, rng.randint(1, 28))
        notice = found + timedelta(days=notice_days)
        scout = found - timedelta(days=scout_before) if scout_before else None
        return {"regime": reg, "crop": crop, "unit": c["unit"], "cause": cause, "weather_support": sup,
                "weather": w, "neighbour_yield_pct": maybe(rng, nb, 0.15), "issue": issue,
                "issue_variant": rng.randrange(2), "uninsured_share": round(share, 3), "disputed": disputed,
                "excuse": excuse, "pending": pending, "aph": aph, "coverage": cov,
                "guarantee": round(guar, 1) if c["unit"] == "bu" else int(round(guar)), "yield_to_count": yield_tc, "acres": round(between(rng, 38, 640), 1),
                "found": found, "notice": notice, "scout_report": scout, "insured": _insured(rng),
                "insurer": _insurer(rng), "adjuster": person(rng), "county": _county(rng), "unit_no": _unit_no(rng),
                "claim_no": ident(rng, "CL", 7, "-"), "policy_no": ident(rng, "MPC", 6, "-"),
                "agent_view": pick(rng, ["supportive", "neutral", "pushing"])}

    # ------------------------------------------------------------------ derived
    @staticmethod
    def _eff_share(f):
        if f["issue"] is None:
            return 0.0
        if f["excuse"] == "too_wet_to_spray" and SUP[f["weather_support"]] >= 2:
            return 0.0           # the insured cause itself kept the sprayer out: not an uninsured cause
        return f["uninsured_share"]

    @staticmethod
    def _support(f):
        s = SUP[f["weather_support"]]
        nb = f["neighbour_yield_pct"]
        if nb is not None:
            if nb >= 93 and s >= 2:
                s -= 1
            elif nb <= 65 and s <= 1:
                s += 1
        if f["pending"] == "station_summary":
            s = min(s, 1)
        return s

    def expected(self, f):
        """Intended answers from the latent factors (spread audit only; never rendered)."""
        u = self._eff_share(f)
        s = self._support(f)
        kind = ISSUES[f["issue"]][0] if f["issue"] else None
        if kind == "gfp" and f["disputed"] and u >= 0.2:
            ruling = "refer_good_farming_review"
        elif f["pending"]:
            ruling = "hold_for_cause_evidence"
        elif s == 0 or u >= 0.85:
            ruling = "deny_uninsured_cause"
        elif u >= 0.04:
            ruling = "pay_less_uninsured_appraisal"
        elif s >= 2:
            ruling = "pay_as_insured_loss"
        else:
            ruling = "hold_for_cause_evidence"
        first = min(_d(d) for d in (f["found"], f["scout_report"]) if d is not None)
        on_time = (_d(f["notice"]) - first).days <= 3
        share_lvl = 0 if u < 0.04 else 1 if u < 0.2 else 2 if u < 0.5 else 3
        return {"loss_cause_ruling": ruling, "weather_evidence_support": s, "uninsured_cause_share": share_lvl,
                "notice_given_on_time": "yes" if on_time else "no",
                "good_farming_departure": "yes" if kind == "gfp" and u >= 0.04 else "no"}

    # ------------------------------------------------------------------ rendering
    def _rules(self, rng, short=False):
        r = [pick(rng, ["§12(a) Insured causes: adverse weather (drought, excess moisture, hail, freeze, wind), fire, "
                        "wildlife, and insects or plant disease unless due to insufficient or improper control measures.",
                        "§12(a) We insure against loss of production from adverse weather, fire, wildlife, and insect or "
                        "disease damage, except where control measures were inadequate or improperly applied."]),
             pick(rng, ["§12(c) Not insured: failure to follow good farming practices; breakdown of irrigation "
                        "equipment; drainage problems that are a condition of the field; harvest delayed without an "
                        "insured cause.",
                        "§12(c) Excluded: departures from good farming practices, failure or breakdown of irrigation "
                        "equipment or facilities, inadequate drainage of the field itself, and unexcused harvest delay."]),
             pick(rng, ["§14(b) Production lost to an uninsured cause is appraised and added to production to count. "
                        "Where we allege a good-farming-practice failure and you dispute it, the question goes to the "
                        "Practices Review Unit before the claim is adjusted.",
                        "§14(b) An appraisal for uninsured causes is added to production to count. A disputed "
                        "good-farming-practice allegation is referred to our Practices Review Unit for determination "
                        "before adjustment."])]
        if not short:
            r.append(pick(rng, ["§20(a) Give notice of damage within 72 hours of when you discovered, or should "
                                "reasonably have discovered, it.",
                                "§20(a) Notice is due within 72 hours of discovery of damage; discovery includes "
                                "damage noted by your own scout or agronomist."]))
        else:
            r.append("§20(a) Notice within 72 hours of discovery.")
        return r

    def _issue_text(self, f):
        if not f["issue"]:
            return None
        return ISSUES[f["issue"]][2][f["issue_variant"] % len(ISSUES[f["issue"]][2])]

    def _uca(self, f):
        """The adjuster's preliminary uninsured-cause appraisal per acre (None when not yet done)."""
        if not f["issue"] or f["uninsured_share"] <= 0:
            return None
        short = max(0.0, f["guarantee"] - f["yield_to_count"])
        v = short * f["uninsured_share"]
        return round(v, 1) if f["unit"] == "bu" else round(v)

    def _dispute_line(self, f, rng):
        if not f["issue"]:
            return None
        if f["excuse"] == "too_wet_to_spray":
            return pick(rng, ["The insured says the field was too wet to carry a sprayer for most of June.",
                              "Insured states the post-emergence pass was impossible: ground too soft to travel "
                              "for three weeks after planting."])
        if not f["disputed"]:
            return pick(rng, ["The insured does not dispute this.", "Insured acknowledged the point at the inspection.",
                              None])
        return pick(rng, DISPUTE_TEXT[ISSUES[f["issue"]][0]])

    OWN_RECORD = {"drought": "the insured's own rain-gauge log", "excess_moisture": "the insured's own rain-gauge log",
                  "freeze": "the insured's own yard thermometer reading", "wind": "the insured's own account of the storm",
                  "hail": "the insured's phone photos of hailstones", "plant_disease": "the insured's own scouting notes",
                  "insects": "the insured's own scouting notes", "wildlife": "the insured's own trail-camera photos"}

    def _pending_text(self, f):
        if f["pending"] == "lab_id":
            return "pathogen / pest identification sent to the diagnostic lab; result not back"
        if f["pending"] == "station_summary":
            return f"official weather / agency records requested; only {self.OWN_RECORD[f['cause']]} on file"
        return None

    def _weather_line(self, f):
        if f["pending"] == "station_summary":
            return f"{self.OWN_RECORD[f['cause']]} only (official records requested)"
        return _weather_text(f["cause"], f["weather_support"], f["weather"], f["crop"])

    def render(self, f, rng, surface):
        ds = _ds(rng)
        u = f["unit"]
        uca = self._uca(f)
        issue_text = self._issue_text(f)
        dispute = self._dispute_line(f, rng)
        nb = f["neighbour_yield_pct"]
        nb_text = (f"comparable {_adj(f['crop'])} fields within 5 km are yielding about {nb}% of their usual"
                   if nb is not None else None)
        statement = pick(rng, [f"Lost most of the crop to {CAUSE_LABEL[f['cause']]}, nothing we could do.",
                               f"The {CAUSE_LABEL[f['cause']]} took the top end off this field.",
                               f"{_cap(CAUSE_LABEL[f['cause']])} damage across the whole unit.",
                               f"Never seen {CAUSE_LABEL[f['cause']]} this bad on this ground."])
        if surface == "json":
            return dumps(prune({
                "claim": {"claim_no": f["claim_no"], "insured": f["insured"], "crop": f["crop"],
                          "county": f["county"], "unit": f["unit_no"], "acres": f["acres"],
                          "claimed_cause": CAUSE_LABEL[f["cause"]], "insured_statement": statement},
                "policy": {"insurer": f["insurer"], "policy_no": f["policy_no"], "approved_yield": f"{f['aph']} {u}/ac",
                           "coverage_level": _pct(f["coverage"]), "provisions": self._rules(rng)},
                "production": {"guarantee_per_acre": f"{f['guarantee']} {u}",
                               "production_to_count_per_acre": f"{f['yield_to_count']} {u}",
                               "uninsured_cause_appraisal_per_acre": f"{uca} {u}" if uca is not None and chance(rng, 0.7)
                               else ("not yet completed" if issue_text else None)},
                "inspection": {"adjuster": f["adjuster"], "practice_finding": issue_text, "insured_response": dispute},
                "evidence": {"weather_record": self._weather_line(f), "neighbouring_fields": nb_text,
                             "outstanding": self._pending_text(f)},
                "notice": {"damage_discovered_per_insured": fmt_date(f["found"], style="iso"),
                           "notice_received": fmt_date(f["notice"], style="iso"),
                           "agronomist_scouting_report": fmt_date(f["scout_report"], style="iso")
                           if f["scout_report"] else None},
            }))
        if surface == "adjuster_worksheet":
            rows = [("Claim", f["claim_no"]), ("Insured", f["insured"]), ("Crop / county", f"{f['crop']} / {f['county']}"),
                    ("Unit", f"{f['unit_no']} ({f['acres']} ac)"), ("Approved yield", f"{f['aph']} {u}/ac"),
                    ("Coverage level", _pct(f["coverage"])), ("Production to count", f"{f['yield_to_count']} {u}/ac"),
                    ("Cause claimed", CAUSE_LABEL[f["cause"]]),
                    ("Damage discovered (insured)", fmt_date(f["found"], style=ds)),
                    ("Notice received", fmt_date(f["notice"], style=ds))]
            if f["scout_report"]:
                rows.append(("Scouting report on file dated", fmt_date(f["scout_report"], style=ds)))
            if uca is not None:
                rows.append(("Uninsured-cause appraisal", f"{uca} {u}/ac (preliminary)"))
            body = kv_lines(rows, rng)
            remarks = [f"Weather: {self._weather_line(f)}."]
            if nb_text:
                remarks.append(_cap(nb_text) + ".")
            if issue_text:
                remarks.append(f"Field finding: {issue_text}.")
            if dispute:
                remarks.append(dispute)
            if f["pending"]:
                remarks.append(_cap(self._pending_text(f)) + ".")
            return (f"{f['insurer'].upper()} -- LOSS WORKSHEET (yield loss)\n{body}\n\nAdjuster remarks ({f['adjuster']}):\n"
                    + noisy(join_sentences(remarks), rng, 0.12) + "\n\nPolicy references:\n"
                    + bullets(self._rules(rng, short=True), rng))
        if surface == "claim_notes":
            t = f["found"]
            notes = [(f["notice"], f"FNOL taken by phone. Insured reports {CAUSE_LABEL[f['cause']]} on {f['crop']}, "
                                   f"unit {f['unit_no']}; says damage first noticed {fmt_date(f['found'], style=ds)}.")]
            if f["scout_report"]:
                notes.append((f["notice"] + timedelta(days=1),
                               f"Insured emailed agronomist scouting report dated {fmt_date(f['scout_report'], style=ds)} "
                               f"describing the same damage."))
            insp = f["notice"] + timedelta(days=rng.randint(2, 9))
            line = f"Field inspection ({f['adjuster']}). Production to count {f['yield_to_count']} {u}/ac against a guarantee of {f['guarantee']} {u}/ac."
            if issue_text:
                line += f" Noted: {issue_text}."
            notes.append((insp, line))
            if dispute:
                notes.append((insp, dispute))
            notes.append((insp + timedelta(days=rng.randint(1, 4)), f"Weather check: {self._weather_line(f)}."
                          + (f" {_cap(nb_text)}." if nb_text else "")))
            if uca is not None and chance(rng, 0.75):
                notes.append((insp + timedelta(days=5), f"Prelim uninsured-cause appraisal {uca} {u}/ac."))
            if f["pending"]:
                notes.append((insp + timedelta(days=6), _cap(self._pending_text(f)) + "."))
            if f["agent_view"] == "pushing":
                notes.append((insp + timedelta(days=7), "Agent called asking when this will be paid; insured is 'a big account'."))
            log = "\n".join(f"{fmt_date(d, style='iso')}  {noisy(x, rng, 0.08)}" for d, x in notes)
            return (f"Claim {f['claim_no']} -- {f['insured']} -- {f['crop']}, {f['county']}\n"
                    f"Policy {f['policy_no']} ({f['insurer']}), coverage {_pct(f['coverage'])}, APH {f['aph']} {u}/ac\n\n"
                    f"{log}\n\nRelevant terms: " + " ".join(self._rules(rng, short=True)))
        # supervisor_email
        sup_name = person(rng)
        para = [f"Need a ruling on {f['claim_no']} ({f['insured']}, {f['acres']} ac {f['crop']}, {f['county']}).",
                f"Claimed cause is {CAUSE_LABEL[f['cause']]}; production to count came in at {f['yield_to_count']} {u}/ac "
                f"on an approved yield of {f['aph']} at {_pct(f['coverage'])} coverage.",
                f"Weather check: {self._weather_line(f)}."]
        if nb_text:
            para.append(_cap(nb_text) + ".")
        if issue_text:
            para.append(f"At inspection I found the following: {issue_text}.")
            if uca is not None:
                para.append(f"My preliminary uninsured-cause appraisal is {uca} {u}/ac.")
        if dispute:
            para.append(dispute)
        if f["pending"] == "lab_id":
            para.append(_cap(self._pending_text(f)) + ".")
        para.append(f"Insured puts discovery at {fmt_date(f['found'], style=ds)}; notice came in "
                    f"{fmt_date(f['notice'], style=ds)}."
                    + (f" The agronomist's scouting report is dated {fmt_date(f['scout_report'], style=ds)}."
                       if f["scout_report"] else ""))
        if f["agent_view"] == "pushing":
            para.append("The agent is pressing for a quick payment.")
        return "\n".join([f"Subject: Cause-of-loss ruling needed -- {f['claim_no']}", f"From: {f['adjuster']}",
                          f"To: {sup_name}", "", f"{sup_name.split()[0]},", "", noisy(join_sentences(para), rng, 0.12), "",
                          "Policy language I'm working from:", bullets(self._rules(rng), rng), "", f["adjuster"].split()[0]])

    def questions(self):
        return {
            "loss_cause_ruling": choice("How should the adjuster rule on the cause of this loss?", {
                "pay_as_insured_loss": "Adjust the claim as a loss from the insured peril, with no deduction.",
                "pay_less_uninsured_appraisal": "Adjust it as an insured loss, but add an appraisal for the part caused "
                                                "by uninsured causes to production to count.",
                "refer_good_farming_review": "Send the disputed farming-practice question to the practices review unit "
                                             "before adjusting.",
                "deny_uninsured_cause": "Deny the claim: the shortfall was not caused by an insured peril.",
                "hold_for_cause_evidence": "Make no ruling yet; obtain the missing weather, lab or field evidence first.",
            }),
            "weather_evidence_support": score("How well do independent weather and area records support the claimed cause?", [
                "Contradicted: the records point away from the claimed cause.",
                "Weak: the records are silent, second-hand or only loosely consistent.",
                "Partial: consistent with the cause, but timing or extent is left open.",
                "Strong: official records and neighbouring fields clearly show the peril struck this unit.",
            ]),
            "uninsured_cause_share": score("How much of the yield shortfall is attributable to causes the policy does not insure?", [
                "None or negligible.",
                "Minor: well under a fifth of the shortfall.",
                "Substantial: roughly a fifth to a half of the shortfall.",
                "Dominant: more than half of the shortfall.",
            ]),
            "notice_given_on_time": noul(
                "The insured gave notice of the damage within the policy's notice period.",
                yes="Notice arrived within 72 hours of when the damage was, or reasonably should have been, discovered.",
                no="Counting from the earliest credible discovery date in the file, notice came later than 72 hours."),
            "good_farming_departure": noul(
                "A departure from good farming practices contributed to this loss."),
        }


# =========================================================================== schema 2
# crop group -> list of (stage label, maturity class); maturity classes: early < mid < late < mature
STAGES = {
    "corn": ([("V12", "early"), ("VT (tasselling)", "early"), ("R2 (blister)", "mid"), ("R3 (milk)", "mid"),
              ("R4 (dough)", "late"), ("R5 (dent)", "late"), ("R6 (black layer)", "mature")], "R4 (dough)"),
    "beans": ([("R1", "early"), ("R2 (full bloom)", "early"), ("R3 (beginning pod)", "mid"), ("R4 (full pod)", "mid"),
               ("R5 (beginning seed)", "late"), ("R6 (full seed)", "late"), ("R7", "mature")], "R5 (beginning seed)"),
    "small_grain": ([("boot", "early"), ("heading", "early"), ("flowering", "mid"), ("milk", "mid"),
                     ("soft dough", "late"), ("hard dough", "mature"), ("ripe", "mature")], "soft dough"),
    "sorghum": ([("boot", "early"), ("half bloom", "mid"), ("milk", "mid"), ("soft dough", "late"),
                 ("hard dough", "late"), ("physiological maturity", "mature")], "soft dough"),
    "sunflowers": ([("R3 (bud)", "early"), ("R5.5 (half bloom)", "mid"), ("R6", "mid"), ("R7 (back of head yellowing)", "late"),
                    ("R8", "late"), ("R9", "mature")], "R7"),
    "canola": ([("bolting", "early"), ("early flower", "early"), ("full flower", "mid"), ("pod fill", "late"),
                ("seed colour change", "late"), ("swathing stage", "mature")], "pod fill"),
    "cotton": ([("first square", "early"), ("first bloom", "mid"), ("peak bloom", "mid"), ("first open boll", "late"),
                ("30% open bolls", "late"), ("60% open bolls", "mature")], "first open boll"),
}
APPRAISAL_CROPS = {"corn": "corn", "popcorn": "corn", "soybeans": "beans", "dry beans": "beans", "field peas": "beans",
                   "spring wheat": "small_grain", "winter wheat": "small_grain", "barley": "small_grain",
                   "oats": "small_grain", "grain sorghum": "sorghum", "sunflowers": "sunflowers", "canola": "canola",
                   "upland cotton": "cotton"}
OTHER_USES = {
    "corn": ["chop it for silage", "chop it and haul it to a dairy", "graze it with cow-calf pairs",
             "let a neighbour graze it under a lease", "disk it under and plant a cover crop"],
    "beans": ["cut and bale it for hay", "disk it under and plant a cover crop", "graze it with stocker cattle",
              "abandon it and work the ground for next season"],
    "small_grain": ["chop it for silage", "bale it for hay", "graze it with cow-calf pairs",
                    "disk it under and plant forage sorghum", "let a neighbour graze it under a lease"],
    "sorghum": ["chop it for silage", "graze it with stocker cattle", "bale it for hay",
                "disk it under and plant a cover crop"],
    "sunflowers": ["chop it for silage", "disk it under and plant a cover crop", "graze it with cow-calf pairs",
                   "abandon it and work the ground for next season"],
    "canola": ["swath and bale it for feed", "graze it with sheep", "disk it under and plant a cover crop",
               "abandon it and work the ground for next season"],
    "cotton": ["shred the stalks and plant wheat", "disk it under and plant a cover crop",
               "abandon it and work the ground for next season", "shred it and plant grain sorghum"],
}
UNINSURED_DAMAGE = ["herbicide injury from the insured's own pre-emergence application on the north end",
                    "a strip where the insured's sprayer boom drifted dicamba onto the crop",
                    "weed escapes in the low half where the post pass was never made",
                    "wheel-track compaction from a wet-season grain cart path that predates the damage",
                    "a block planted with bin-run seed of an unadapted variety"]


def _samples_required(acres):
    if acres <= 40:
        return 3
    if acres <= 80:
        return 4
    if acres <= 160:
        return 6
    return 6 + math.ceil((acres - 160) / 80)


class PreharvestAppraisal(Schema):
    name = "preharvest_appraisal"
    description = "An appraisal of a damaged standing crop the insured wants to destroy or put to another use."
    surfaces = {"json": 0.5, "appraisal_worksheet": 0.25, "consent_request": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        crop = pick(rng, list(APPRAISAL_CROPS))
        c = CROPS[crop]
        stages, ready = STAGES[APPRAISAL_CROPS[crop]]
        acres = round(between(rng, 22, 420), 1)
        req = _samples_required(acres)
        taken = req + pick(rng, [0, 0, 1, 2, -1])
        pattern = pick_w(rng, {"grid": 4, "transects": 3, "field_edge": 1, "worst_area_only": 1})
        stage_i = rng.randrange(len(stages))
        cv = between(rng, 0.08, 0.45)
        aph = rng.randint(*c["aph"])
        cov = pick(rng, [0.65, 0.70, 0.75, 0.80, 0.85])
        guar = aph * cov
        mean_ratio = between(rng, 0.25, 1.15)
        destroyed = 0.0
        agent_ok = False
        uninsured = None
        days_to_use = pick(rng, [2, 3, 5, 7, 10, 14, 21, 30])
        if reg == "clear":
            arche = pick(rng, ["accept", "more", "immature", "uninsured", "destroyed"])
            if arche in ("accept", "uninsured"):
                taken, pattern, cv = req + rng.randint(0, 3), pick(rng, ["grid", "transects"]), between(rng, 0.08, 0.2)
                stage_i = rng.choice([i for i, s in enumerate(stages) if s[1] in ("late", "mature")])
                if arche == "uninsured":
                    uninsured = pick(rng, UNINSURED_DAMAGE)
            elif arche == "more":
                taken = max(1, req - rng.randint(2, 3))
            elif arche == "immature":
                stage_i = rng.choice([i for i, s in enumerate(stages) if s[1] == "early"])
            else:
                destroyed = round(acres * between(rng, 0.1, 0.4), 1)
                agent_ok = chance(rng, 0.5)
        elif reg == "borderline":
            taken = req + pick(rng, [-1, 0, 0, 1])
            cv = near(rng, 0.30, 0.08, lo=0.06)
            stage_i = rng.choice([i for i, s in enumerate(stages) if s[1] in ("mid", "late")])
            mean_ratio = near(rng, 1.0, 0.18, lo=0.3)
            if chance(rng, 0.3):
                uninsured = pick(rng, UNINSURED_DAMAGE)
        else:
            twist = pick(rng, ["destroyed", "destroyed", "pressure", "own_estimate"])
            if twist == "destroyed":
                destroyed = round(acres * between(rng, 0.04, 0.3), 1)
                agent_ok = chance(rng, 0.6)
            elif twist == "pressure":
                stage_i = rng.choice([i for i, s in enumerate(stages) if s[1] in ("early", "mid")])
                days_to_use = pick(rng, [2, 3, 4])
            if chance(rng, 0.35):
                uninsured = pick(rng, UNINSURED_DAMAGE)
        if reg == "borderline" and chance(rng, 0.25):
            destroyed = round(acres * between(rng, 0.03, 0.15), 1)
            agent_ok = chance(rng, 0.5)
        taken = max(1, taken)
        mean = guar * mean_ratio
        vals = [max(0.0, mean * (1 + rng.gauss(0, cv))) for _ in range(taken)]
        if c["unit"] == "bu":
            vals = [round(v, 1) for v in vals]
        else:
            vals = [float(round(v)) for v in vals]
        own_est = round(guar * between(rng, 0.15, 0.6)) if chance(rng, 0.4) else None
        return {"regime": reg, "crop": crop, "unit": c["unit"], "acres": acres, "samples_required": req,
                "samples": vals, "pattern": pattern, "stage": stages[stage_i][0], "maturity": stages[stage_i][1],
                "ready_stage": ready, "aph": aph, "coverage": cov, "guarantee": round(guar, 1),
                "destroyed_acres": destroyed, "agent_verbal_ok": agent_ok and destroyed > 0,
                "uninsured_damage": uninsured, "intended_use": pick(rng, OTHER_USES[APPRAISAL_CROPS[crop]]),
                "days_to_use": days_to_use,
                "insured_estimate": own_est, "insured": _insured(rng), "insurer": _insurer(rng), "adjuster": person(rng),
                "county": _county(rng), "unit_no": _unit_no(rng), "claim_no": ident(rng, "CL", 7, "-"),
                "date": date(pick(rng, [2025, 2026]),
                             {"early": 6, "mid": 7, "late": 8, "mature": 9}[stages[stage_i][1]]
                             - (1 if crop == "winter wheat" else 0), rng.randint(1, 28))}

    @staticmethod
    def _stats(f):
        v = f["samples"]
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
        return m, (sd / m if m > 0 else 1.0)

    def _reliability(self, f):
        m, cv = self._stats(f)
        s = 3
        short = f["samples_required"] - len(f["samples"])
        if short >= 1:
            s -= 1
        if short >= 2:
            s -= 1
        if f["pattern"] in ("field_edge", "worst_area_only"):
            s -= 1
        s -= {"early": 2, "mid": 1}.get(f["maturity"], 0)
        if cv > 0.30:
            s -= 1
        return max(0, min(3, s))

    def expected(self, f):
        """Intended answers from the latent factors (spread audit only; never rendered)."""
        m, cv = self._stats(f)
        rel = self._reliability(f)
        bad_samples = len(f["samples"]) < f["samples_required"] or f["pattern"] in ("field_edge", "worst_area_only")
        immature = f["maturity"] in ("early", "mid")
        if bad_samples:
            step = "take_more_samples"
        elif immature:
            step = "leave_check_strips" if f["days_to_use"] <= 7 else "defer_until_crop_matures"
        elif cv > 0.30:
            step = "leave_check_strips"
        else:
            step = "accept_and_release"
        if f["destroyed_acres"] > 0:
            basis = "guarantee_on_destroyed_acres"
        elif rel <= 1:
            basis = "no_count_until_reappraisal"
        elif f["uninsured_damage"]:
            basis = "appraisal_plus_uninsured_cause"
        else:
            basis = "appraised_yield_all_acres"
        r = m / f["guarantee"]
        short = 0 if r >= 1.0 else 1 if r >= 0.85 else 2 if r >= 0.5 else 3
        return {"appraisal_next_step": step, "production_to_count_basis": basis, "appraisal_reliability": rel,
                "shortfall_against_guarantee": short,
                "acreage_destroyed_before_consent": "yes" if f["destroyed_acres"] > 0 else "no"}

    def _rules(self, rng, f):
        return [
            pick(rng, ["§16(a) Acreage may not be destroyed or put to another use until we give written consent after "
                       "an appraisal. Consent from your agent is not consent from us.",
                       "§16(a) You must obtain our written consent before destroying the crop or using the acreage for "
                       "another purpose; agents cannot give consent."]),
            pick(rng, ["§16(b) Appraisal samples: up to 40 ac, 3; 40.1-80 ac, 4; 80.1-160 ac, 6; one more for each "
                       "additional 80 ac or part. Samples must represent the whole field.",
                       "§16(b) Minimum samples per field: 3 (to 40 ac), 4 (to 80 ac), 6 (to 160 ac), plus 1 per further "
                       "80 ac or fraction, spread so they represent the field."]),
            f"§16(c) For {f['crop']}, appraisals before {f['ready_stage']} are provisional; we may require "
            "representative unharvested check strips to be left for a later appraisal, or defer the appraisal.",
            pick(rng, ["§16(d) On acreage destroyed without our consent, production to count is the full per-acre "
                       "guarantee. Damage from uninsured causes is appraised and added.",
                       "§16(d) Acres destroyed before consent count at the production guarantee; appraised uninsured-"
                       "cause damage is added to production to count."]),
        ]

    def render(self, f, rng, surface):
        ds = _ds(rng)
        u = f["unit"]
        pat = {"grid": "grid across the field", "transects": "two diagonal transects",
               "field_edge": "along the field road within 60 m of the entrance",
               "worst_area_only": "in the most damaged part of the field only"}[f["pattern"]]
        own = (f"Insured's own estimate: {f['insured_estimate']} {u}/ac." if f["insured_estimate"] else None)
        own_letter = (pick(rng, [f"We figure it makes maybe {f['insured_estimate']} {u} an acre.",
                                 f"My yield check says about {f['insured_estimate']} {u}/ac at best."])
                      if f["insured_estimate"] else "")
        destroyed = None
        if f["destroyed_acres"] > 0:
            destroyed = (f"{f['destroyed_acres']} ac on the {pick(rng, ['east', 'west', 'north', 'south'])} side were "
                         f"already {pick(rng, ['chopped', 'disked', 'grazed', 'baled'])} before the adjuster's visit"
                         + ("; the insured says the agent told him by phone it was fine" if f["agent_verbal_ok"] else ""))
        use = f"wants to {f['intended_use']} in {f['days_to_use']} days"
        if surface == "json":
            return dumps(prune({
                "request": {"claim_no": f["claim_no"], "insured": f["insured"], "crop": f["crop"], "county": f["county"],
                            "unit": f["unit_no"], "field_acres": f["acres"], "intended_use": f["intended_use"],
                            "days_until_planned_use": f["days_to_use"], "insured_estimate_per_acre": f["insured_estimate"]},
                "policy": {"insurer": f["insurer"], "approved_yield": f["aph"], "coverage_level": _pct(f["coverage"]),
                           "yield_unit": f"{u}/ac", "provisions": self._rules(rng, f)},
                "appraisal": {"adjuster": f["adjuster"], "date": fmt_date(f["date"], style="iso"),
                              "growth_stage": f["stage"], "sample_placement": pat,
                              "sample_yields_per_acre": f["samples"]},
                "field_notes": {"acreage_already_used": destroyed, "other_damage": f["uninsured_damage"]},
            }))
        if surface == "appraisal_worksheet":
            rows = [[str(i + 1), f"{v:g}"] for i, v in enumerate(f["samples"])]
            head = kv_lines([("Claim", f["claim_no"]), ("Insured", f["insured"]), ("Crop", f["crop"]),
                             ("Unit / field", f"{f['unit_no']} / {f['acres']} ac"), ("County", f["county"]),
                             ("Appraised by", f"{f['adjuster']}, {fmt_date(f['date'], style=ds)}"),
                             ("Growth stage", f["stage"]), ("Sample placement", pat),
                             ("APH / coverage", f"{f['aph']} {u}/ac at {_pct(f['coverage'])}")], rng)
            notes = [f"Insured {use}."]
            if destroyed:
                notes.append(_cap(destroyed) + ".")
            if f["uninsured_damage"]:
                notes.append(f"Also noted: {f['uninsured_damage']}.")
            if own:
                notes.append(own)
            return (f"PRE-HARVEST APPRAISAL WORKSHEET -- {f['insurer']}\n{head}\n\n"
                    + table(["sample", f"{u}/ac"], rows, rng) + "\n\nNotes: " + noisy(join_sentences(notes), rng, 0.1)
                    + "\n\nPolicy: " + " ".join(self._rules(rng, f)))
        # consent_request: insured's letter + adjuster's reply draft facts
        m = sum(f["samples"]) / len(f["samples"])
        letter = [f"We'd like consent to {f['intended_use']} on unit {f['unit_no']} ({f['acres']} ac of {f['crop']}).",
                  pick(rng, ["The custom crew is booked, so we need an answer quickly.",
                             "Feed is short this year and the cattle need it.",
                             "There's nothing left worth harvesting out there.", ""]),
                  own_letter]
        if destroyed and chance(rng, 0.6):
            letter.append(pick(rng, ["We already started on the worst corner to see what it would make.",
                                     "Part of it is already done, the agent said that was fine."]))
        adj = [f"Appraised {fmt_date(f['date'], style=ds)} at {f['stage']}: {len(f['samples'])} samples placed {pat}, "
               f"values {', '.join(f'{v:g}' for v in f['samples'])} {u}/ac (mean {m:.1f}).",
               f"APH {f['aph']} {u}/ac at {_pct(f['coverage'])} coverage."]
        if destroyed:
            adj.append(_cap(destroyed) + ".")
        if f["uninsured_damage"]:
            adj.append(f"Also {f['uninsured_damage']}.")
        return "\n".join([
            f"From: {f['insured']}", f"To: claims@{f['insurer'].split()[0].lower()}-crop.example", f"Re: claim {f['claim_no']}",
            "", noisy(join_sentences(letter), rng, 0.2), "", "-----", f"Adjuster file note ({f['adjuster']}):",
            noisy(join_sentences(adj), rng, 0.08), "", "Applicable provisions:", bullets(self._rules(rng, f), rng)])

    def questions(self):
        return {
            "appraisal_next_step": choice("What should the adjuster do with this appraisal and the request to use the acreage?", {
                "accept_and_release": "Accept the appraisal and give written consent to use the acreage now.",
                "leave_check_strips": "Consent only if the insured leaves representative unharvested strips for a "
                                      "later appraisal.",
                "take_more_samples": "Go back and take further samples before any consent is given.",
                "defer_until_crop_matures": "Withhold consent and appraise again once the crop reaches the stage the "
                                            "policy requires.",
            }),
            "production_to_count_basis": choice("On what basis should production to count be set for this field?", {
                "appraised_yield_all_acres": "The appraised yield applies to every acre of the field.",
                "appraisal_plus_uninsured_cause": "The appraised yield plus an appraisal for damage from uninsured causes.",
                "guarantee_on_destroyed_acres": "The full guarantee on acres used before consent, and the appraisal on "
                                                "the rest.",
                "no_count_until_reappraisal": "Nothing yet: this appraisal cannot support a count and a new one is needed.",
            }),
            "appraisal_reliability": score("How far can the insurer rely on this appraisal?", [
                "Not at all: too few, badly placed or premature samples.",
                "Questionable: one serious weakness undermines it.",
                "Reasonable: minor weaknesses, but usable.",
                "Solid: enough representative samples at a stage the policy accepts.",
            ]),
            "shortfall_against_guarantee": score("How far does the appraised yield fall short of the per-acre production guarantee?", [
                "No shortfall: the appraisal meets or exceeds the guarantee.",
                "Slight: less than 15% below the guarantee.",
                "Material: roughly 15% to 50% below the guarantee.",
                "Severe: less than half the guarantee remains.",
            ]),
            "acreage_destroyed_before_consent": noul(
                "Part of this acreage was destroyed or put to another use before the insurer gave consent.",
                yes="Some acres were chopped, grazed, baled or worked before the insurer's written consent, whatever "
                    "anyone else said.",
                no="Nothing has been destroyed or used yet; the acreage is intact for appraisal."),
        }


# =========================================================================== schema 3
# crop -> (replant threshold, plants per acre; replant payment quantity per acre, text)
REPLANT = {"corn": (22000, "8 bu"), "popcorn": (20000, "260 lb"), "soybeans": (75000, "3 bu"),
           "grain sorghum": (38000, "4 bu"), "upland cotton": (26000, "55 lb"), "sunflowers": (13000, "175 lb"),
           "canola": (160000, "175 lb"), "dry beans": (52000, "150 lb"), "field peas": (220000, "2 bu")}
RP_INSURED = {
    "crusting": "soil crusted after a {mm} mm downpour two days after planting and seedlings could not break through",
    "hail": "hail at emergence cut seedlings off at the soil line",
    "frost": "a frost night ({t} C at the nearest station) killed emerged seedlings",
    "sandblasting": "high winds sandblasted seedlings on the lighter soils",
    "ponding": "water stood on the low ground for {d} days after heavy rain",
}
RP_UNINSURED = {
    "herbicide_carryover": "the injury pattern matches carryover from last season's herbicide program",
    "planter_failure": "gaps line up with two planter row units that were not dropping seed",
    "seed_quality": "a germination test on the leftover seed lot came back at {g}%",
    "planting_depth": "seed was placed about {cm} cm deep into cold soil",
    "fertilizer_burn": "seedlings show salt injury where starter fertilizer went in-furrow at a high rate",
}
AREA_NAMES = ["North field", "South 40", "Home quarter", "River bottom", "West 80", "Pivot 3", "East half", "Hill field",
              "Creek field", "Sandy knob", "Terrace field", "Section 12 NE", "Section 7 SW", "Old pasture", "Bridge field",
              "Field 4", "Field 9", "Tower field"]
RP_VERIFY = {
    "inspected_before": "adjuster inspected and counted stands before replanting",
    "strips_left": "insured replanted before inspection but left untouched strips of the original stand; strips not yet inspected",
    "insurer_release": "insurer released the acreage for replanting by phone after reviewing the insured's stand counts and photos",
    "photos_only": "insured replanted before any inspection; only phone photos of the original stand remain",
    "agent_ok": "insured replanted on the agent's say-so before the insurer was told; nothing of the original stand left",
}


class ReplantPaymentRequest(Schema):
    name = "replant_payment_request"
    description = "A request for a replant payment after an early-season stand loss, judged against the replant provisions."
    surfaces = {"json": 0.5, "replant_claim_form": 0.25, "adjuster_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        crop = pick(rng, list(REPLANT))
        thr = REPLANT[crop][0]
        unit_acres = round(between(rng, 55, 700), 1)
        min_size = min(20.0, 0.2 * unit_acres)
        n_areas = pick(rng, [1, 1, 2, 2, 3])
        claimed = pick(rng, list(RP_INSURED))
        signal = pick_w(rng, {"confirmed": 4, "partly_supported": 3, "unclear": 2, "points_elsewhere": 1})
        verify = pick_w(rng, {"inspected_before": 5, "strips_left": 2, "insurer_release": 2, "photos_only": 1,
                              "agent_ok": 1})
        size_target = near(rng, min_size * 1.05, max(4, min_size * 0.45), lo=3)
        stand_ratio = near(rng, 0.8, 0.25, lo=0.25)
        partial = chance(rng, 0.35)
        if reg == "clear":
            arche = pick(rng, ["pay", "pay", "partial", "deny_cause", "deny_verify"])
            verify = "inspected_before" if arche != "deny_verify" else pick(rng, ["photos_only", "agent_ok"])
            signal = "confirmed" if arche in ("pay", "partial") else ("points_elsewhere" if arche == "deny_cause" else signal)
            size_target = min_size * between(rng, 1.5, 3.0)
            stand_ratio = between(rng, 0.3, 0.7)
            partial = arche == "partial"
        elif reg == "conflicting":
            twist = pick(rng, ["cause", "verify", "partial"])
            if twist == "cause":
                signal = pick(rng, ["unclear", "points_elsewhere"])
                verify = "inspected_before"
            elif twist == "verify":
                verify = pick(rng, ["agent_ok", "insurer_release", "strips_left"])
                signal = pick(rng, ["confirmed", "partly_supported"])
            else:
                partial = True
                signal = "confirmed"
        # replanted areas: main area near stand_ratio, others vary; partial -> one area above threshold
        areas = []
        names = rng.sample(AREA_NAMES, 4)
        remaining = max(3.0, size_target)
        for i in range(n_areas):
            ac = round(remaining / (n_areas - i) * between(rng, 0.7, 1.3), 1) if i < n_areas - 1 else round(remaining, 1)
            ac = max(2.0, ac)
            remaining = max(2.0, remaining - ac)
            r = stand_ratio * between(rng, 0.85, 1.15)
            areas.append([names[i], ac, int(round(thr * r, -2))])
        if partial:
            areas.append([names[len(areas)], round(between(rng, 4, 25), 1), int(round(thr * between(rng, 1.03, 1.3), -2))])
        alt = None
        if signal in ("unclear", "points_elsewhere"):
            alt = pick(rng, list(RP_UNINSURED))
        return {"regime": reg, "crop": crop, "threshold": thr, "payment_qty": REPLANT[crop][1], "unit_acres": unit_acres,
                "areas": areas, "claimed_cause": claimed, "cause_signal": signal, "alt_cause": alt, "verification": verify,
                "counts_per_area": pick(rng, [3, 4, 5, 6, 8]), "ev_variant": rng.randrange(3),
                "nums": {"mm": rng.randint(45, 110), "t": round(rng.uniform(-3.5, -0.8), 1), "d": rng.randint(3, 8),
                         "g": rng.randint(52, 74), "cm": round(rng.uniform(6, 9), 1)},
                "insured": _insured(rng), "insurer": _insurer(rng), "adjuster": person(rng), "county": _county(rng),
                "unit_no": _unit_no(rng), "claim_no": ident(rng, "RP", 6, "-"),
                "replant_date": date(pick(rng, [2025, 2026]), rng.randint(5, 6), rng.randint(1, 28)),
                "price": round(between(rng, *CROPS[crop]["price"]), 2)}

    def _eligible(self, f):
        return round(sum(a[1] for a in f["areas"] if a[2] < f["threshold"]), 1)

    def expected(self, f):
        """Intended answers from the latent factors (spread audit only; never rendered)."""
        thr = f["threshold"]
        tot = sum(a[1] for a in f["areas"])
        wmean = sum(a[1] * a[2] for a in f["areas"]) / tot
        r = wmean / thr
        shortfall = 0 if r >= 1.0 else 1 if r >= 0.9 else 2 if r >= 0.5 else 3
        conf = {"confirmed": 3, "partly_supported": 2, "unclear": 1, "points_elsewhere": 0}[f["cause_signal"]]
        elig = self._eligible(f)
        size_ok = elig >= min(20.0, 0.2 * f["unit_acres"]) - 1e-9
        verifiable = f["verification"] in ("inspected_before", "strips_left", "insurer_release")
        if not verifiable:
            dec = "deny_replant_payment"
        elif f["verification"] == "strips_left" or conf == 1:
            dec = "verify_before_deciding"
        elif conf == 0 or elig == 0 or not size_ok:
            dec = "deny_replant_payment"
        elif elig < tot - 1e-9:
            dec = "pay_on_qualifying_acres_only"
        else:
            dec = "pay_on_all_replanted_acres"
        return {"replant_decision": dec, "stand_shortfall": shortfall, "insured_cause_confidence": conf,
                "size_rule_met": "yes" if size_ok else "no", "damage_verifiable": "yes" if verifiable else "no"}

    def _rules(self, rng, f):
        return [
            pick(rng, [f"§18(a) A replant payment is due only where an insured cause reduced the stand of {f['crop']} "
                       f"below {f['threshold']:,} plants per acre and the acreage is replanted.",
                       f"§18(a) Replant threshold for {f['crop']}: {f['threshold']:,} plants/ac. Stands below it "
                       "caused by an insured cause qualify when replanted."]),
            pick(rng, ["§18(b) Acres qualifying for payment must total at least the lesser of 20 acres or 20% of the "
                       "insured planted acreage of the unit.",
                       "§18(b) Minimum: qualifying replanted acres >= the smaller of 20 ac or 20% of the unit's insured "
                       "planted acres."]),
            pick(rng, ["§18(c) You must give notice and let us inspect before replanting, unless we authorise "
                       "replanting without inspection. An agent cannot authorise it.",
                       "§18(c) No payment unless we inspected, or released the acreage, before replanting; leaving "
                       "representative unreplanted strips preserves the right to inspect."]),
            f"§18(d) Payment: {f['payment_qty']}/ac x projected price x share, on qualifying acres only.",
        ]

    def _cause_text(self, f):
        n = f["nums"]
        base = RP_INSURED[f["claimed_cause"]].format(**n)
        alt = RP_UNINSURED[f["alt_cause"]].format(**n) if f["alt_cause"] else None
        sig = f["cause_signal"]
        if sig == "confirmed":
            ev = ["station records and neighbours' fields confirm it",
                  "photos and the station record line up with the insured's account",
                  "the township's other replant claims and the station log back it up"][f["ev_variant"]]
            return f"Insured reports {base}; {ev}.", None
        if sig == "partly_supported":
            return (f"Insured reports {base}; the field inspection found damage consistent with it, but neighbours on "
                    "similar ground report little trouble and nothing else independent ties it to this field."), None
        if sig == "unclear":
            return f"Insured reports {base}.", f"Inspection also notes that {alt}; the pattern does not settle which explains the gaps."
        return f"Insured reports {base}.", f"Inspection found that {alt}, and the damaged rows do not follow the low ground or the storm track."

    def render(self, f, rng, surface):
        ds = _ds(rng)
        cause, alt = self._cause_text(f)
        thr = f["threshold"]
        per_k = thr / 1000
        area_rows = [[a[0], f"{a[1]:g}", f"{a[2]:,}"] for a in f["areas"]]
        verify = RP_VERIFY[f["verification"]]
        if surface == "json":
            return dumps(prune({
                "request": {"claim_no": f["claim_no"], "insured": f["insured"], "crop": f["crop"], "county": f["county"],
                            "unit": f["unit_no"], "unit_insured_planted_acres": f["unit_acres"],
                            "replanted_on": fmt_date(f["replant_date"], style="iso")},
                "policy": {"insurer": f["insurer"], "projected_price": money(f["price"], style="sym"),
                           "provisions": self._rules(rng, f)},
                "stand_counts": {"method": f"{f['counts_per_area']} counts per area, plants per acre",
                                 "replanted_areas": [{"area": a[0], "acres": a[1], "original_stand_per_acre": a[2]}
                                                     for a in f["areas"]]},
                "cause": {"reported": cause, "adjuster_note": alt},
                "verification": verify,
            }))
        if surface == "replant_claim_form":
            head = kv_lines([("Claim", f["claim_no"]), ("Insured", f["insured"]), ("Insurer", f["insurer"]),
                             ("Crop", f["crop"]), ("County / unit", f"{f['county']} / {f['unit_no']}"),
                             ("Insured planted acres (unit)", f["unit_acres"]),
                             ("Replanted", fmt_date(f["replant_date"], style=ds)),
                             ("Inspection status", verify)], rng)
            counts = table(["area", "acres", "orig. stand (plants/ac)"], area_rows, rng)
            return (f"REPLANT PAYMENT CLAIM\n{head}\n\nStand counts ({f['counts_per_area']} per area):\n{counts}\n\n"
                    f"Cause: {noisy(cause, rng, 0.1)}" + (f"\n{alt}" if alt else "") + "\n\nProvisions:\n"
                    + "\n".join(self._rules(rng, f)))
        # adjuster_email: stand counts given per 1/1000 acre so the reader converts
        lines = []
        for a in f["areas"]:
            lines.append(f"{a[0]}: {a[1]:g} ac, average {a[2] / 1000:.1f} plants per 1/1000 ac row length")
        sup = person(rng)
        para = [f"Replant claim {f['claim_no']} for {f['insured']} ({f['crop']}, {f['county']}).",
                f"The unit has {f['unit_acres']} insured planted acres; replanting was done {fmt_date(f['replant_date'], style=ds)}.",
                cause]
        if alt:
            para.append(alt)
        para.append(_cap(verify) + ".")
        return "\n".join([f"Subject: Replant {f['claim_no']} -- your call", f"From: {f['adjuster']}", f"To: {sup}", "",
                          noisy(join_sentences(para), rng, 0.1), "",
                          f"Stand counts (threshold {f['threshold']:,} plants/ac = {per_k:g} per 1/1000 ac):",
                          bullets(lines, rng), "", "Provisions:", bullets(self._rules(rng, f), rng), "",
                          f["adjuster"].split()[0]])

    def questions(self):
        return {
            "replant_decision": choice("How should the replant payment request be decided?", {
                "pay_on_all_replanted_acres": "Pay the replant payment on every acre the insured replanted.",
                "pay_on_qualifying_acres_only": "Pay only on replanted acres whose original stand was below the threshold.",
                "deny_replant_payment": "Refuse the replant payment.",
                "verify_before_deciding": "Decide nothing yet: inspect the remaining strips or pin down the cause first.",
            }),
            "stand_shortfall": score("Across the replanted acres, how far below the replant threshold was the original stand?", [
                "Not below: the stand met or beat the threshold.",
                "Marginal: under a tenth below the threshold.",
                "Clear: between a tenth and a half below the threshold.",
                "Devastated: less than half the threshold survived.",
            ]),
            "insured_cause_confidence": score("How confident can the insurer be that an insured cause, not a planting or product problem, thinned the stand?", [
                "Not confident: the evidence points to an uninsured cause.",
                "Doubtful: an uninsured explanation is about as likely.",
                "Fairly confident: consistent damage, thin corroboration.",
                "Confident: the insured cause is corroborated by independent records.",
            ]),
            "size_rule_met": noul(
                "The qualifying replanted acreage meets the policy's minimum size for a replant payment.",
                yes="Acres replanted from a below-threshold stand total at least the lesser of 20 acres or 20% of the "
                    "unit's insured planted acres.",
                no="Counting only below-threshold acres, the total falls short of that minimum."),
            "damage_verifiable": noul(
                "The insurer had, or still has, a fair chance to verify the original stand loss."),
        }


# =========================================================================== schema 4
PP_FIELD = {
    "none": None,
    "low_spots": "a few low spots hold water in most springs but the rest of the field drains well",
    "tile_outlet": "the main tile outlet collapsed last autumn and was not repaired before spring",
    "pothole": "the field sits in a closed pothole basin that ponds in most springs",
    "creek_bottom": "the claimed acres are creek-bottom ground that floods whenever the creek is high",
}
PP_ACTION = {
    "left_idle": "acres left idle after the final planting date",
    "second_crop": "a second crop ({second}) planted on the acres on {d2}",
    "cover_hayed": "a cover crop seeded {d2} and cut for hay on {d3}",
    "cover_grazed_early": "a cover crop seeded {d2} and grazed from {d3}",
    "cover_unharvested": "a cover crop seeded {d2}, not hayed or grazed before {d3}",
}
SECOND_CROPS = ["soybeans", "forage sorghum", "millet", "sunflowers", "buckwheat", "sudangrass"]


class PreventedPlantingClaim(Schema):
    name = "prevented_planting_claim"
    description = "A prevented-planting claim for acres that went unplanted in a wet spring, judged against the policy terms."
    surfaces = {"json": 0.5, "pp_review_memo": 0.25, "claim_file_notes": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        crop = pick(rng, SPRING_CROPS)
        c = CROPS[crop]
        year = pick(rng, [2025, 2026])
        fpd = date(year, *c["fpd"])
        rain = near(rng, 150, 70, lo=70, integer=True)
        area = maybe(rng, near(rng, 65, 30, lo=8, hi=98, integer=True), 0.15)
        field = pick_w(rng, {"none": 6, "low_spots": 2, "tile_outlet": 1, "pothole": 1, "creek_bottom": 1})
        prior_pp = pick(rng, [0, 0, 0, 1, 1, 2, 3])
        action = pick_w(rng, {"left_idle": 4, "second_crop": 2, "cover_hayed": 1, "cover_grazed_early": 1,
                              "cover_unharvested": 2})
        before_scd = False
        imagery = False
        hist = [rng.randint(180, 900) for _ in range(4)]
        claim_ratio = near(rng, 1.0, 0.12)
        if reg == "borderline" and chance(rng, 0.12):
            imagery = True
        if reg == "clear":
            arche = pick(rng, ["pay", "pay", "deny", "reduced"])
            if arche in ("pay", "reduced"):
                rain, area, field, prior_pp = rng.randint(190, 280), rng.randint(10, 40), "none", 0
                action = "left_idle" if arche == "pay" else pick(rng, ["second_crop", "cover_hayed"])
                claim_ratio = between(rng, 0.5, 0.9)
            else:
                rain, area = rng.randint(85, 120), rng.randint(84, 97)
        elif reg == "conflicting":
            twist = pick(rng, ["imagery", "imagery", "chronic", "scd", "missing_area", "cover"])
            if twist == "imagery":
                imagery = True
            elif twist == "chronic":
                rain, area = rng.randint(180, 260), rng.randint(15, 45)
                field, prior_pp = pick(rng, ["pothole", "creek_bottom", "tile_outlet"]), pick(rng, [2, 3])
            elif twist == "scd":
                before_scd = True
                rain, area = rng.randint(170, 240), rng.randint(20, 50)
            elif twist == "missing_area":
                area = None
                rain = rng.randint(125, 165)
            else:
                action = pick(rng, ["cover_hayed", "cover_grazed_early", "cover_unharvested"])
        claimed = round(max(hist) * claim_ratio)
        d2 = fpd + timedelta(days=rng.randint(12, 30))
        late_cut = action == "cover_unharvested"
        d3 = date(year, 11, 1) if late_cut else d2 + timedelta(days=rng.randint(45, 90))
        if action == "cover_unharvested":
            d3 = date(year, 11, rng.randint(1, 20))
        aph = rng.randint(*c["aph"])
        return {"regime": reg, "crop": crop, "unit": c["unit"], "fpd": fpd, "rain_pct": rain, "area_planted_pct": area,
                "field_condition": field, "prior_pp_claims": prior_pp, "action": action, "second_crop": pick(rng, SECOND_CROPS),
                "d2": d2, "d3": d3, "cause_before_sales_closing": before_scd, "imagery_shows_planted": imagery,
                "history": hist, "claimed_acres": claimed, "aph": aph, "coverage": pick(rng, [0.70, 0.75, 0.80, 0.85]),
                "photos": pick_w(rng, {"dated_photos": 3, "undated_photos": 2, "none": 1}),
                "insured": _insured(rng), "insurer": _insurer(rng), "adjuster": person(rng), "county": _county(rng),
                "unit_no": _unit_no(rng), "claim_no": ident(rng, "PP", 6, "-"), "year": year,
                "notice_date": fpd + timedelta(days=pick(rng, [-3, 0, 1, 2, 3, 5]))}

    @staticmethod
    def _second(f):
        return f["action"] in ("second_crop", "cover_hayed", "cover_grazed_early")

    @staticmethod
    def _field_level(f):
        lvl = {"none": 0, "low_spots": 1, "tile_outlet": 2, "pothole": 3, "creek_bottom": 2}[f["field_condition"]]
        if f["prior_pp_claims"] >= 2:
            lvl = min(3, lvl + 1)
        return lvl

    def expected(self, f):
        """Intended answers from the latent factors (spread audit only; never rendered)."""
        area = f["area_planted_pct"]
        if area is None:
            ev = 1 if f["rain_pct"] >= 160 else 0
        else:
            ev = 3 if area <= 45 else 2 if area <= 65 else 1 if area < 80 else 0
            if f["rain_pct"] < 115:
                ev = max(0, ev - 1)
        fl = self._field_level(f)
        if f["imagery_shows_planted"]:
            out = "adjust_as_late_planted_acreage"
        elif area is None and f["rain_pct"] < 170:
            out = "request_area_evidence"
        elif f["cause_before_sales_closing"] or fl >= 2 or (area is not None and area >= 80) or ev == 0:
            out = "deny_prevented_planting"
        elif self._second(f):
            out = "pay_reduced_for_second_crop"
        else:
            out = "pay_full_prevented_planting"
        return {"pp_claim_outcome": out, "area_condition_evidence": ev, "field_condition_likelihood": fl,
                "acres_within_history": "yes" if f["claimed_acres"] <= max(f["history"]) else "no",
                "second_crop_rules_triggered": "yes" if self._second(f) else "no"}

    def _rules(self, rng, f):
        return [
            pick(rng, ["§22(a) Prevented planting is payable when an insured cause occurring after the sales closing date "
                       "(March 15) kept you from planting by the final planting date, the cause was general to the "
                       "surrounding area, and it also prevented other producers from planting similar acreage.",
                       "§22(a) We pay prevented planting only if an insured cause arising after sales closing (Mar 15) "
                       "stopped planting by the final planting date and was general in the area, preventing other "
                       "producers on similar ground from planting."]),
            pick(rng, ["§22(b) Not payable where the failure to plant results from a condition of the field, such as "
                       "inadequate drainage, rather than area-wide weather.",
                       "§22(b) Excluded: inability to plant caused by the field's own condition (e.g. poor drainage)."]),
            pick(rng, ["§22(e) Eligible acres may not exceed the most acres of the crop you planted in any one of the "
                       "four most recent crop years.",
                       "§22(e) PP acres are capped at the greatest acreage of the crop planted in any of the last four "
                       "crop years."]),
            pick(rng, ["§22(g) Payment is 55% of the guarantee; it drops to 35% if a second crop is planted on the "
                       "acreage. A cover crop hayed or grazed before November 1 counts as a second crop.",
                       "§22(g) 55% of the guarantee, reduced to 35% when a second crop follows; haying or grazing a "
                       "cover crop before Nov 1 makes it a second crop."]),
            "§22(h) Acreage actually planted during the late planting period is adjusted as late-planted acreage, "
            "not as prevented planting.",
        ]

    def _action_text(self, f, ds):
        return PP_ACTION[f["action"]].format(second=f["second_crop"], d2=fmt_date(f["d2"], style=ds),
                                             d3=fmt_date(f["d3"], style=ds))

    def render(self, f, rng, surface):
        ds = _ds(rng)
        hist_years = [f["year"] - 4 + i for i in range(4)]
        hist = ", ".join(f"{y}: {a} ac" for y, a in zip(hist_years, f["history"]))
        area = (f"{f['area_planted_pct']}% of similar ground intended for {f['crop']} in the township was planted by "
                f"the final planting date" if f["area_planted_pct"] is not None else None)
        field = PP_FIELD[f["field_condition"]]
        imagery = ("aerial imagery from three weeks after the final planting date shows emerged rows on part of the "
                   "claimed acres" if f["imagery_shows_planted"] else None)
        scd = ("the insured's photos show the claimed acres under standing water in February, before sales closing"
               if f["cause_before_sales_closing"] else None)
        photos = {"dated_photos": "dated photos of standing water on file", "undated_photos": "undated photos on file",
                  "none": "no photos"}[f["photos"]]
        prior = (f"{f['prior_pp_claims']} prior prevented-planting claim(s) on these fields in the last four years"
                 if f["prior_pp_claims"] else None)
        if surface == "json":
            return dumps(prune({
                "claim": {"claim_no": f["claim_no"], "insured": f["insured"], "crop": f["crop"], "county": f["county"],
                          "unit": f["unit_no"], "prevented_planting_acres_claimed": f["claimed_acres"],
                          "notice_received": fmt_date(f["notice_date"], style="iso"),
                          "use_of_claimed_acres_after_final_planting_date": self._action_text(f, "iso")},
                "policy": {"insurer": f["insurer"], "final_planting_date": fmt_date(f["fpd"], style="iso"),
                           "coverage_level": _pct(f["coverage"]), "provisions": self._rules(rng, f)},
                "area_evidence": {"rainfall_30d_before_fpd_pct_normal": f["rain_pct"],
                                  "township_planting_progress": area or "county progress report not yet obtained"},
                "field_evidence": {"adjuster": f["adjuster"], "field_condition": field, "history": prior,
                                   "photos": photos, "imagery": imagery, "pre_sales_closing": scd},
                "acreage_history": dict(zip([str(y) for y in hist_years], f["history"])),
            }))
        if surface == "pp_review_memo":
            sup = person(rng)
            para = [f"{f['insured']} claims prevented planting on {f['claimed_acres']} acres of {f['crop']} "
                    f"(unit {f['unit_no']}, {f['county']}); final planting date {fmt_date(f['fpd'], style=ds)}.",
                    f"Rainfall in the 30 days before that date was {f['rain_pct']}% of normal.",
                    (_cap(area) + "." if area else "We do not yet have the township planting progress figures.")]
            if field:
                para.append(f"On inspection, {field}.")
            if prior:
                para.append(_cap(prior) + ".")
            if imagery:
                para.append(_cap(imagery) + ".")
            if scd:
                para.append(_cap(scd) + ".")
            para.append(f"After the final planting date: {self._action_text(f, ds)}.")
            para.append(f"Acreage history: {hist}.")
            para.append(_cap(photos) + ".")
            return "\n".join([f"MEMO -- prevented planting review", f"To: {sup}", f"From: {f['adjuster']}",
                              f"Re: {f['claim_no']}", "", noisy(join_sentences(para), rng, 0.1), "",
                              "Policy:", bullets(self._rules(rng, f), rng)])
        # claim_file_notes
        notes = [(f["notice_date"], f"PP notice received for {f['claimed_acres']} ac {f['crop']}, unit {f['unit_no']}."),
                 (f["fpd"] + timedelta(days=rng.randint(4, 12)),
                  f"Field visit. {_cap(field) + '.' if field else 'No drainage problems seen.'} {_cap(photos)}.")]
        notes.append((f["fpd"] + timedelta(days=rng.randint(13, 20)),
                      f"Station data: {f['rain_pct']}% of normal rain in the 30 days before FPD. "
                      + (_cap(area) + "." if area else "Planting-progress report requested, not received.")))
        if prior:
            notes.append((f["fpd"] + timedelta(days=21), _cap(prior) + "."))
        if imagery:
            notes.append((f["fpd"] + timedelta(days=24), _cap(imagery) + "."))
        if scd:
            notes.append((f["fpd"] + timedelta(days=25), _cap(scd) + "."))
        notes.append((f["d3"] if f["d3"] > f["d2"] else f["d2"], f"Insured reports {self._action_text(f, ds)}."))
        notes.append((f["d3"] + timedelta(days=3), f"Acreage history pulled: {hist}."))
        notes.sort(key=lambda x: x[0])
        log = "\n".join(f"{fmt_date(d, style='iso')} | {noisy(x, rng, 0.08)}" for d, x in notes)
        return (f"CLAIM FILE {f['claim_no']} -- {f['insured']} -- {f['insurer']}\nFinal planting date "
                f"{fmt_date(f['fpd'], style=ds)}; coverage {_pct(f['coverage'])}\n\n{log}\n\nTerms: "
                + " ".join(self._rules(rng, f)))

    def questions(self):
        return {
            "pp_claim_outcome": choice("How should this prevented-planting claim be settled?", {
                "pay_full_prevented_planting": "Pay prevented planting at the full rate on the eligible acres.",
                "pay_reduced_for_second_crop": "Pay prevented planting at the reduced rate that applies after a second crop.",
                "adjust_as_late_planted_acreage": "Treat the acres as planted late and adjust them under the late "
                                                  "planting rules instead.",
                "deny_prevented_planting": "Refuse the claim: the policy conditions for prevented planting are not met.",
                "request_area_evidence": "Hold the claim until evidence of conditions across the area is obtained.",
            }),
            "area_condition_evidence": score("How well does the file show that planting conditions were bad across the area, not just on this farm?", [
                "None: most similar acreage nearby was planted on time, or nothing shows conditions beyond this farm.",
                "Thin: wet weather is documented, but most neighbours on similar ground still managed to plant.",
                "Fair: a large share of similar acreage nearby also went unplanted, though many still planted.",
                "Strong: area records show most similar acreage could not be planted.",
            ]),
            "field_condition_likelihood": score("How likely is it that the acres went unplanted because of the field's own condition rather than the season's weather?", [
                "Unlikely: nothing suggests a drainage or terrain problem.",
                "Possible: minor wet spots, otherwise a sound field.",
                "Likely: a known drainage or flooding problem on the claimed acres.",
                "Near certain: a chronic problem that has kept these acres out before.",
            ]),
            "acres_within_history": noul(
                "The prevented-planting acres claimed are within the policy's acreage history limit.",
                yes="The claimed acres do not exceed the most acres of the crop planted in any of the last four years.",
                no="The claimed acres exceed that four-year maximum."),
            "second_crop_rules_triggered": noul(
                "What happened on the acres afterwards brings in the reduced second-crop payment."),
        }


# =========================================================================== schema 5
# crop -> (toxin, unit, adjustment threshold, destruction level, weather that favours it)
TOXINS = {
    "corn": [("aflatoxin", "ppb", 20.0, 300.0, "hot_dry"), ("fumonisin", "ppm", 5.0, 60.0, "hot_dry"),
             ("deoxynivalenol (DON)", "ppm", 5.0, 20.0, "wet_silking")],
    "winter wheat": [("deoxynivalenol (DON)", "ppm", 2.0, 10.0, "wet_flowering")],
    "spring wheat": [("deoxynivalenol (DON)", "ppm", 2.0, 10.0, "wet_flowering")],
    "barley": [("deoxynivalenol (DON)", "ppm", 1.0, 8.0, "wet_flowering")],
    "grain sorghum": [("aflatoxin", "ppb", 20.0, 300.0, "hot_dry")],
    "popcorn": [("aflatoxin", "ppb", 20.0, 300.0, "hot_dry")],
}
WEATHER_FILL = {
    "hot_dry": ("{d} days above 35 C and {p}% of normal rain during grain fill",
                "a mild, moist grain fill ({d} days above 35 C, {p}% of normal rain)"),
    "wet_silking": ("rain on {d} of the 10 days around silking", "a dry silking period (rain on {d} of 10 days)"),
    "wet_flowering": ("rain on {d} of the 10 days around flowering", "a dry flowering period (rain on {d} of 10 days)"),
}
SAMPLERS = {"adjuster": "the adjuster", "inspector": "a licensed grain inspector approved by the insurer",
            "elevator": "the buying elevator at delivery", "insured": "the insured"}
LABS = {"approved_lab": "approved laboratory (quantitative)", "strip_test": "elevator quick strip test"}


class QualityAdjustmentClaim(Schema):
    name = "quality_adjustment_claim"
    description = "A mycotoxin quality-adjustment request on harvested grain, judged against the sampling and cause rules."
    surfaces = {"json": 0.5, "lot_table": 0.25, "adjuster_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        crop = pick(rng, list(TOXINS))
        tox, tunit, thr, destroy, wx = pick(rng, TOXINS[crop])
        n_lots = pick(rng, [2, 3, 3, 4])
        sampler = pick_w(rng, {"adjuster": 3, "inspector": 3, "elevator": 2, "insured": 1})
        lab = pick_w(rng, {"approved_lab": 5, "strip_test": 2})
        commingled = pick_w(rng, {"none": 6, "other_unit": 2, "purchased": 1})
        records = pick_w(rng, {"scale_tickets": 4, "bin_measurements": 2, "yield_monitor_only": 2, "none": 1})
        moisture = round(near(rng, 16.5, 3.5, lo=12.5, hi=23.0), 1)
        aeration = pick_w(rng, {"working": 6, "failed": 2, "none": 1})
        weeks = pick(rng, [0, 0, 1, 3, 6, 10, 16])
        weather_fav = chance(rng, 0.6)
        harvest_reading = None
        level = near(rng, 1.15, 0.6, lo=0.2)
        sold_before = 0
        if reg == "clear":
            arche = pick(rng, ["allow", "allow", "below", "invalid", "partial_sold"])
            if arche in ("allow", "partial_sold"):
                sampler, lab, commingled, weeks, weather_fav = pick(rng, ["adjuster", "inspector"]), "approved_lab", "none", 0, True
                records, moisture, aeration = "scale_tickets", round(between(rng, 13.0, 15.5), 1), "working"
                level = between(rng, 1.6, 6.0)
                sold_before = 1 if arche == "partial_sold" else 0
            elif arche == "below":
                sampler, lab = pick(rng, ["adjuster", "inspector"]), "approved_lab"
                level = between(rng, 0.2, 0.75)
            else:
                sampler, lab = "insured", "strip_test"
        elif reg == "borderline":
            level = near(rng, 1.0, 0.3, lo=0.4)
            if chance(rng, 0.4):
                sold_before = 1
            if chance(rng, 0.2):     # a harvest screen near the threshold, then weeks in a wet bin
                moisture, weeks = round(between(rng, 17.5, 20.0), 1), pick(rng, [6, 8, 10])
                harvest_reading = round(thr * near(rng, 1.0, 0.2, lo=0.6), 1)
                level = between(rng, 1.3, 2.5)
        else:
            twist = pick(rng, ["storage", "storage", "mixed", "mixed", "sold", "elevator"])
            if twist == "storage":
                moisture, aeration, weeks = round(between(rng, 17.5, 22.0), 1), pick(rng, ["failed", "none", "working"]), pick(rng, [6, 10, 16, 24])
                harvest_reading = maybe(rng, round(thr * between(rng, 0.1, 0.7), 1), 0.4)
                weather_fav = chance(rng, 0.4)
                level = between(rng, 1.3, 5.0)
            elif twist == "mixed":
                moisture, aeration, weeks = round(between(rng, 17.5, 21.0), 1), pick(rng, ["failed", "none"]), pick(rng, [8, 12, 20])
                harvest_reading = round(thr * between(rng, 1.05, 1.5), 1)
                weather_fav = True
                level = between(rng, 2.4, 6.0)
                sampler, lab = pick(rng, ["adjuster", "inspector"]), "approved_lab"
            elif twist == "sold":
                sold_before = pick(rng, [1, 2])
                level = between(rng, 1.3, 4.0)
            else:
                sampler = "elevator"
                lab = pick(rng, ["approved_lab", "strip_test"])
                level = between(rng, 1.2, 4.0)
        lots = []
        for i in range(n_lots):
            v = thr * max(0.05, level * between(rng, 0.7, 1.3))
            if v >= destroy and level < 9:
                v = destroy * between(rng, 0.6, 0.95)
            v = round(v, 1) if tunit == "ppm" else round(v)
            lots.append({"lot": pick(rng, ["Bin", "Bin", "Bag", "Pile"]) + f" {i + 1}",
                         "bushels": rng.randint(4, 40) * 500, "result": v, "sampled": True})
        if chance(rng, 0.06) and reg != "clear":
            lots[0]["result"] = round(destroy * between(rng, 1.02, 1.5), 1 if tunit == "ppm" else 0)
        sold_before = min(sold_before, n_lots - 1)
        for k in range(sold_before):
            lots[-1 - k]["sampled"] = False
            lots[-1 - k]["result"] = None
        if weeks == 0 and harvest_reading is None and chance(rng, 0.3):
            harvest_reading = lots[0]["result"]
        return {"regime": reg, "crop": crop, "toxin": tox, "toxin_unit": tunit, "threshold": thr, "destroy_level": destroy,
                "weather_kind": wx, "weather_favourable": weather_fav,
                "wx_d": (rng.randint(8, 20) if wx == "hot_dry" else rng.randint(6, 10)) if weather_fav else rng.randint(0, 3),
                "wx_p": rng.randint(30, 60) if weather_fav else rng.randint(95, 140), "lots": lots, "sampler": sampler,
                "lab": lab, "commingled": commingled, "records": records, "binning_moisture": moisture,
                "aeration": aeration, "weeks_stored_before_sampling": weeks, "harvest_reading": harvest_reading,
                "in_storage": chance(rng, 0.6), "insured": _insured(rng), "insurer": _insurer(rng),
                "adjuster": person(rng), "county": _county(rng), "unit_no": _unit_no(rng),
                "claim_no": ident(rng, "QA", 6, "-"),
                "harvest": date(pick(rng, [2025, 2026]),
                                pick(rng, [7, 8] if crop in ("winter wheat", "barley") else [8, 9] if crop == "spring wheat"
                                     else [9, 10, 11]), rng.randint(1, 28))}

    def _validity(self, f):
        s = 3
        if f["sampler"] == "insured":
            s -= 2
        elif f["sampler"] == "elevator":
            s -= 1
        if f["lab"] == "strip_test":
            s -= 1
        if f["commingled"] != "none":
            s -= 1
        return max(0, s)

    @staticmethod
    def _origin(f):
        hr, thr = f["harvest_reading"], f["threshold"]
        poor_storage = ((f["binning_moisture"] >= 17.5 or f["aeration"] != "working")
                        and f["weeks_stored_before_sampling"] >= 6)
        if poor_storage:
            if hr is not None and hr > thr:
                return "mixed_field_and_storage"     # already over at harvest, and grew in a badly kept bin
            if hr is not None:
                return "storage_after_harvest"       # clean at harvest, over after weeks in a wet or unaerated bin
            return "cannot_tell_yet" if f["weather_favourable"] else "storage_after_harvest"
        if f["weather_favourable"] or f["weeks_stored_before_sampling"] <= 1:
            return "field_weather_damage"
        return "cannot_tell_yet"

    def expected(self, f):
        """Intended answers from the latent factors (spread audit only; never rendered)."""
        val = self._validity(f)
        origin = self._origin(f)
        res = [l["result"] for l in f["lots"] if l["sampled"]]
        mx = max(res)
        thr, dl = f["threshold"], f["destroy_level"]
        conc = 0 if mx <= thr else 1 if mx <= thr * 1.25 else 2 if mx < dl else 3
        unsampled = any(not l["sampled"] for l in f["lots"])
        if val <= 1:
            ruling = "resample_before_ruling" if f["in_storage"] else "deny_quality_adjustment"
        elif conc == 0 or origin == "storage_after_harvest":
            ruling = "deny_quality_adjustment"
        elif origin == "mixed_field_and_storage":
            ruling = "allow_at_harvest_time_level"
        elif origin == "cannot_tell_yet":
            ruling = "resample_before_ruling" if f["in_storage"] else "deny_quality_adjustment"
        elif unsampled:
            ruling = "allow_on_sampled_lots_only"
        else:
            ruling = "allow_full_quality_adjustment"
        trace = f["commingled"] == "none" and f["records"] in ("scale_tickets", "bin_measurements")
        return {"quality_adjustment_ruling": ruling, "toxin_origin": origin, "sample_validity": val,
                "toxin_level_concern": conc, "production_traceable": "yes" if trace else "no"}

    def _rules(self, rng, f):
        u = f["toxin_unit"]
        return [
            pick(rng, [f"§15(a) Production is quality-adjusted for {f['toxin']} only where the deficiency results from "
                       f"an insured cause before harvest and the level exceeds {f['threshold']:g} {u}. Deterioration in "
                       "storage is not insured.",
                       f"§15(a) {_cap(f['toxin'])} above {f['threshold']:g} {u} qualifies for adjustment if caused by insured "
                       "weather before harvest; damage arising in storage does not."]),
            pick(rng, ["§15(c) Samples must be taken by us or by a disinterested third party we approve, before the "
                       "production is commingled, sold or fed, and tested quantitatively by an approved laboratory.",
                       "§15(c) Only samples drawn by us or an approved disinterested sampler, before commingling, sale or "
                       "feeding, and tested quantitatively at an approved lab, are accepted."]),
            "§15(d) Production sold or commingled before an acceptable sample was taken cannot be quality-adjusted.",
            f"§15(e) Above {f['destroy_level']:g} {u}, production may be destroyed with our consent and counted as zero.",
        ]

    def _wx(self, f):
        a, b = WEATHER_FILL[f["weather_kind"]]
        return (a if f["weather_favourable"] else b).format(d=f["wx_d"], p=f["wx_p"])

    def render(self, f, rng, surface):
        ds = _ds(rng)
        u = f["toxin_unit"]
        comm = {"none": None, "other_unit": "lots 1-2 also hold grain from the insured's other unit",
                "purchased": "one bin was topped up with grain bought from a neighbour"}[f["commingled"]]
        recs = {"scale_tickets": "load-by-load scale tickets", "bin_measurements": "bin measurements by the adjuster",
                "yield_monitor_only": "yield-monitor totals only", "none": "no production records"}[f["records"]]
        aer = {"working": "aeration fans running", "failed": "aeration fan failed for about three weeks",
               "none": "no aeration in the bins"}[f["aeration"]]
        lots = [{"lot": l["lot"], "bushels": l["bushels"],
                 "result": f"{l['result']:g} {u}" if l["sampled"] else "sold before sampling"} for l in f["lots"]]
        hr = (f"{f['harvest_reading']:g} {u}" if f["harvest_reading"] is not None else None)
        stored = (f"sampled {f['weeks_stored_before_sampling']} weeks after binning"
                  if f["weeks_stored_before_sampling"] else "sampled at harvest from trucks")
        status = "grain still in the insured's bins" if f["in_storage"] else "sampled grain since delivered to the elevator"
        if surface == "json":
            return dumps(prune({
                "claim": {"claim_no": f["claim_no"], "insured": f["insured"], "crop": f["crop"], "county": f["county"],
                          "unit": f["unit_no"], "harvest_completed": fmt_date(f["harvest"], style="iso"),
                          "status": status},
                "policy": {"insurer": f["insurer"], "provisions": self._rules(rng, f)},
                "sampling": {"sampled_by": SAMPLERS[f["sampler"]], "test": LABS[f["lab"]], "timing": stored,
                             "lots": lots, "harvest_time_reading": hr},
                "storage": {"moisture_at_binning_pct": f["binning_moisture"], "aeration": aer, "commingling": comm,
                            "production_records": recs},
                "season": {"grain_fill_weather": self._wx(f)},
            }))
        if surface == "lot_table":
            rows = [[l["lot"], f"{l['bushels']:,}", l["result"]] for l in lots]
            head = kv_lines([("Claim", f["claim_no"]), ("Insured", f["insured"]), ("Crop", f["crop"]),
                             ("County / unit", f"{f['county']} / {f['unit_no']}"), ("Toxin", f["toxin"]),
                             ("Sampled by", SAMPLERS[f["sampler"]]), ("Test", LABS[f["lab"]]), ("Timing", stored),
                             ("Harvest-time reading", hr), ("Moisture at binning", f"{f['binning_moisture']}%"),
                             ("Aeration", aer), ("Commingling", comm or "none reported"), ("Records", recs),
                             ("Status", status)], rng)
            return (f"QUALITY ADJUSTMENT -- lot results\n{head}\n\n" + table(["lot", "bushels", f["toxin"]], rows, rng)
                    + f"\n\nSeason: {self._wx(f)}.\n\nProvisions:\n" + "\n".join(self._rules(rng, f)))
        para = [f"{f['insured']} has asked for a quality adjustment on {f['crop']} from unit {f['unit_no']} "
                f"({f['county']}), harvest finished {fmt_date(f['harvest'], style=ds)}.",
                f"Samples were taken by {SAMPLERS[f['sampler']]} ({stored}) and run on an {LABS[f['lab']]}.",
                "Results: " + "; ".join(f"{l['lot']} ({l['bushels']:,} bu) {l['result']}" for l in lots) + ".",
                f"Grain went into the bins at {f['binning_moisture']}% moisture with {aer}."]
        if hr:
            para.append(f"A reading at harvest showed {hr}.")
        if comm:
            para.append(_cap(comm) + ".")
        para.append(f"Production records: {recs}. Grain fill saw {self._wx(f)}.")
        para.append(_cap(status) + ".")
        return "\n".join([f"MEMO -- quality adjustment request {f['claim_no']}", f"From: {f['adjuster']} ({f['insurer']})",
                          "", noisy(join_sentences(para), rng, 0.1), "", "Provisions:", bullets(self._rules(rng, f), rng)])

    def questions(self):
        return {
            "quality_adjustment_ruling": choice("How should the quality-adjustment request be decided?", {
                "allow_full_quality_adjustment": "Allow the quality adjustment on all of the production.",
                "allow_on_sampled_lots_only": "Allow it only on lots sampled properly before they left the insured's control.",
                "allow_at_harvest_time_level": "Allow it, but only at the toxin level documented at harvest, not the "
                                               "higher level found after storage.",
                "deny_quality_adjustment": "Refuse the quality adjustment.",
                "resample_before_ruling": "Have the stored grain sampled again properly before deciding.",
            }),
            "toxin_origin": choice("Where does the evidence place the origin of the toxin problem?", {
                "field_weather_damage": "In the field, from weather during the growing season.",
                "storage_after_harvest": "In storage, after the grain was harvested.",
                "mixed_field_and_storage": "Partly in the field and partly in storage.",
                "cannot_tell_yet": "The file cannot yet say whether it began in the field or in storage.",
            }),
            "sample_validity": score("How far do the samples meet the policy's sampling requirements?", [
                "Not at all: wrong sampler, wrong test and compromised grain.",
                "Poorly: more than one requirement is missed.",
                "Mostly: one requirement is doubtful.",
                "Fully: approved sampler, quantitative lab test, uncommingled grain.",
            ]),
            "toxin_level_concern": score("Where do the sampled toxin levels sit against the policy's levels?", [
                "Below the adjustment threshold on every sampled lot.",
                "At or just over the threshold (within about a quarter above it).",
                "Clearly over the threshold, but below the destruction level.",
                "At or above the level at which the grain may be destroyed.",
            ]),
            "production_traceable": noul(
                "The production being adjusted can be traced reliably to this unit.",
                yes="The grain was kept separate and load or bin records tie it to the unit.",
                no="It was mixed with other grain or the records cannot tie it to the unit."),
        }


WORKFLOW = Workflow(
    name="crop_insurance_claims",
    title="Crop insurance claims",
    description="Loss-adjustment decisions on a multi-peril crop policy: cause of loss, appraisals, replant payments, "
                "prevented planting and quality adjustment, judged against the written policy provisions.",
    domain="agriculture / crop insurance",
    schemas=[CauseOfLossDetermination(), PreharvestAppraisal(), ReplantPaymentRequest(), PreventedPlantingClaim(),
             QualityAdjustmentClaim()],
    notes="Fictional insurer and policy form (MPC-7) with invented thresholds; adjusters' rulings on the claim file only. "
          "No bills, invoices or payments are processed, and no policyholder conversation is answered.",
)
