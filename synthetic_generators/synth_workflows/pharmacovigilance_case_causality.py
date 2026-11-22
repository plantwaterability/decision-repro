"""Pharmacovigilance case causality: post-marketing drug-safety case decisions at a (fictional)
marketing-authorisation holder, made against the company's own written case-handling SOP.

Five schemas: assessing a spontaneous health-professional report (seriousness, labelling, causality), checking a
consumer report for validity, screening a literature hit, categorising a medication-error report, and reviewing
an aggregate case series for a signal.

Everything is invented: the company, brand and non-proprietary names (built from syllable pools), the Company
Core Data Sheet (CCDS) listings, SOP numbers, journals, reporters and patients. Other medicines appear by class
only. The causality categories follow the familiar WHO-UMC-style scheme but are written in the company's own
words for this generator; they, the seriousness rules, the validity rules and the signal thresholds are this
fictional company's LOCAL SOP and are not presented as regulatory guidance.

Post-marketing only: no clinical-trial SAE forms, investigator's brochures or protocol reporting routes (those
live in clinical_trial_screening). Consumer reports are intake records, never a conversation, and no question
asks how to reply to anyone. Patient age or sex may appear as case descriptors but never decides an answer.
Each schema draws a `regime` first; `answer_key` is a rule-of-thumb key for the spread check only, never a label.
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
    ident,
    join_sentences,
    kv_lines,
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

# --------------------------------------------------------------------------- products (all invented)
BRAND_A = ["Vel", "Zor", "Tral", "Ques", "Lum", "Dar", "Ox", "Pel", "Myr", "Cal", "Ner", "Sol", "Bri", "Fen", "Ard",
           "Kyl", "Hal", "Iv", "Nov", "Ter", "Ul", "Xan", "Yor", "Em", "Tav", "Rov"]
BRAND_B = ["ora", "exa", "imo", "aren", "ivex", "anza", "oril", "ura", "eti", "alto", "ondra", "ique", "avel", "ena"]
INN_A = ["dar", "pel", "zol", "tor", "mer", "quin", "vel", "rax", "lin", "sor", "ten", "bex", "fal", "gor", "nir",
         "cav", "hol", "jun"]
INN_MID = ["o", "a", "i", "e", "u"]
# key, class, invented stem, indications, dosage form, strengths, events the class is known for, events common in
# the treated population (confounding by indication)
CLASSES = [
    dict(key="fxa", name="oral factor Xa inhibitor", stem="xadrel", form="film-coated tablets",
         ind=["non-valvular atrial fibrillation", "deep vein thrombosis"], strengths=["2.5 mg", "5 mg", "10 mg", "20 mg"],
         known=["gibleed"], common=["pe", "gibleed"]),
    dict(key="sglt2", name="SGLT2 inhibitor", stem="glutrane", form="tablets", ind=["type 2 diabetes", "chronic heart failure"],
         strengths=["10 mg", "25 mg"], known=["dka", "aki"], common=["hypo", "aki"]),
    dict(key="ssri", name="selective serotonin reuptake inhibitor", stem="oxetal", form="tablets",
         ind=["major depressive disorder", "panic disorder"], strengths=["10 mg", "20 mg", "40 mg"],
         known=["hypona", "gibleed", "suic"], common=["suic"]),
    dict(key="il17", name="anti-IL-17A monoclonal antibody", stem="lixuvab", form="prefilled pen",
         ind=["plaque psoriasis", "psoriatic arthritis"], strengths=["80 mg", "160 mg"], known=["injsite"], common=[]),
    dict(key="statin", name="HMG-CoA reductase inhibitor", stem="vastrol", form="tablets",
         ind=["primary hypercholesterolaemia"], strengths=["5 mg", "10 mg", "40 mg"], known=["rhabdo", "liver"],
         common=["myocard"]),
    dict(key="fq", name="fluoroquinolone antibiotic", stem="floxacor", form="tablets",
         ind=["complicated urinary tract infection", "community-acquired pneumonia"],
         strengths=["250 mg", "500 mg", "750 mg"], known=["tendon", "tdp", "seiz"], common=["aki", "hypona"]),
    dict(key="ppi", name="proton pump inhibitor", stem="prazor", form="gastro-resistant capsules",
         ind=["gastro-oesophageal reflux disease"], strengths=["20 mg", "40 mg"], known=["hypona", "aki"],
         common=["gibleed"]),
    dict(key="arb", name="angiotensin II receptor blocker", stem="sartrel", form="tablets", ind=["essential hypertension"],
         strengths=["40 mg", "80 mg", "160 mg"], known=["angio", "aki"], common=["dizzy", "pe"]),
    dict(key="jak", name="JAK inhibitor", stem="citanib", form="tablets", ind=["rheumatoid arthritis", "ulcerative colitis"],
         strengths=["5 mg", "10 mg", "15 mg"], known=["pe", "liver"], common=["ild", "pe"]),
    dict(key="glp1", name="GLP-1 receptor agonist", stem="glupeptin", form="prefilled pen",
         ind=["type 2 diabetes", "chronic weight management"], strengths=["0.5 mg", "1 mg", "2 mg"],
         known=["panc", "nausea"], common=["panc", "hypo"]),
    dict(key="aed", name="sodium-channel-blocking antiepileptic", stem="zepamide", form="tablets",
         ind=["focal epilepsy", "bipolar maintenance"], strengths=["25 mg", "100 mg", "200 mg"],
         known=["sjs", "hypona"], common=["seiz", "suic"]),
    dict(key="bis", name="oral bisphosphonate", stem="dronex", form="tablets", ind=["osteoporosis"],
         strengths=["35 mg", "70 mg"], known=["gibleed"], common=["pe"]),
    dict(key="apsy", name="atypical antipsychotic", stem="piradone", form="tablets", ind=["schizophrenia", "bipolar mania"],
         strengths=["5 mg", "10 mg", "15 mg"], known=["tdp", "seiz", "agran"], common=["suic"]),
    dict(key="tnf", name="TNF-alpha inhibitor", stem="tanercel", form="prefilled syringe",
         ind=["rheumatoid arthritis", "Crohn's disease"], strengths=["40 mg"], known=["liver", "injsite"],
         common=["ild"]),
    dict(key="macro", name="macrolide antibiotic", stem="thromycel", form="tablets", ind=["lower respiratory tract infection"],
         strengths=["250 mg", "500 mg"], known=["tdp", "liver"], common=["hypona", "aki"]),
    dict(key="ret", name="oral retinoid", stem="retinate", form="capsules", ind=["severe nodular acne"],
         strengths=["10 mg", "20 mg", "40 mg"], known=["liver", "panc"], common=["suic"]),
    dict(key="trip", name="5-HT1B/1D agonist", stem="triptol", form="orodispersible tablets", ind=["acute migraine"],
         strengths=["2.5 mg", "5 mg"], known=["dizzy"], common=["vision", "headache"]),
    dict(key="ccb", name="dihydropyridine calcium-channel blocker", stem="dipalor", form="tablets",
         ind=["hypertension", "stable angina"], strengths=["5 mg", "10 mg"], known=["dizzy", "angio"], common=["aki", "myocard"]),
    dict(key="opi", name="opioid analgesic", stem="dolantrex", form="oral solution",
         ind=["severe chronic pain"], strengths=["10 mg/5 mL", "100 mg/5 mL"], known=["nausea", "dizzy", "hypona"],
         common=["gibleed", "suic"]),
    dict(key="ins", name="long-acting basal insulin analogue", stem="basulin", form="prefilled pen",
         ind=["type 1 diabetes", "type 2 diabetes"], strengths=["100 units/mL", "300 units/mL"], known=["hypo"],
         common=["dka", "hypo"]),
]
CLS = {c["key"]: c for c in CLASSES}

_E = dict(milder=None, ime=False, lt=None, fatal=False, dis=None, test=None, minor=False)
EVENTS = [dict(_E, **e) for e in [
    dict(key="liver", term="drug-induced liver injury with jaundice", milder="hepatic enzymes increased", ime=True,
         fatal=True, lat=(10, 90), alts=["hepatitis E IgM positive", "heavy alcohol intake", "gallstones on ultrasound"],
         test="ALT 14 x ULN, bilirubin 3 x ULN"),
    dict(key="sjs", term="Stevens-Johnson syndrome", milder="rash", ime=True,
         lt="skin detachment over 9% of body surface, nursed in a burns unit", fatal=True,
         dis="ocular scarring with reduced vision", lat=(4, 28),
         alts=["an anticonvulsant started three weeks earlier", "mycoplasma pneumonia"], test="skin biopsy consistent"),
    dict(key="anaph", term="anaphylaxis", milder="hypersensitivity reactions", ime=True,
         lt="systolic pressure 60 mmHg, two doses of adrenaline", fatal=True, lat=(0, 1),
         alts=["shellfish eaten an hour earlier", "a cephalosporin given the same morning"],
         test="mast-cell tryptase raised"),
    dict(key="aki", term="acute kidney injury", milder="blood creatinine increased", dis="long-term dialysis", lat=(3, 60),
         alts=["dehydration from gastroenteritis", "an NSAID course for back pain", "iodinated contrast two days earlier"],
         test="creatinine tripled from baseline"),
    dict(key="panc", term="acute pancreatitis", milder="lipase increased", lat=(14, 180),
         alts=["gallstones on imaging", "triglycerides 14 mmol/L", "binge drinking the weekend before"],
         test="lipase 9 x ULN, CT confirmed"),
    dict(key="tdp", term="torsade de pointes", milder="electrocardiogram QT prolonged", ime=True,
         lt="pulseless polymorphic VT, defibrillated", fatal=True, lat=(1, 30),
         alts=["potassium 2.8 mmol/L", "a macrolide antibiotic started two days before"], test="QTc 590 ms"),
    dict(key="agran", term="agranulocytosis", milder="neutropenia", ime=True, lt="neutropenic sepsis with hypotension",
         fatal=True, lat=(14, 90), alts=["an antithyroid drug taken concurrently", "a recent viral illness"],
         test="neutrophils 0.1 x 10^9/L"),
    dict(key="angio", term="angioedema of the tongue", milder="hypersensitivity reactions",
         lt="airway swelling needing intubation", lat=(0, 365),
         alts=["an ACE inhibitor taken for years", "a family history of hereditary angioedema"],
         test="C1-esterase inhibitor normal"),
    dict(key="hypona", term="severe hyponatraemia", milder="hyponatraemia", lt="sodium 112 mmol/L with a seizure",
         lat=(5, 45), alts=["a thiazide diuretic", "pneumonia with inappropriate ADH secretion"], test="sodium 118 mmol/L"),
    dict(key="rhabdo", term="rhabdomyolysis", milder="myalgia", lat=(7, 120),
         alts=["a long lie after a fall", "a half-marathon two days earlier", "a fibrate added last month"],
         test="CK 38,000 U/L"),
    dict(key="ild", term="interstitial lung disease", ime=True, lt="respiratory failure needing high-flow oxygen",
         fatal=True, dis="a lasting home-oxygen requirement", lat=(30, 365),
         alts=["a viral pneumonia the month before", "occupational dust exposure"], test="CT ground-glass change"),
    dict(key="seiz", term="generalised tonic-clonic seizure", ime=True, lat=(1, 60),
         alts=["alcohol withdrawal", "known epilepsy with a missed dose"], test="EEG unremarkable"),
    dict(key="tendon", term="Achilles tendon rupture", milder="tendinitis", dis="a permanent limp after surgical repair",
         lat=(2, 30), alts=["a corticosteroid injection into the heel", "a squash injury"], test="ultrasound confirmed"),
    dict(key="gibleed", term="upper gastrointestinal haemorrhage", milder="minor bleeding events",
         lt="haemoglobin 58 g/L, shocked, four units transfused", fatal=True, lat=(5, 180),
         alts=["a known duodenal ulcer", "a daily antiplatelet agent"], test="endoscopy: bleeding ulcer"),
    dict(key="dka", term="diabetic ketoacidosis", lt="pH 6.95, admitted to intensive care", lat=(7, 180),
         alts=["several missed insulin doses", "a urinary infection"], test="ketones 6.2 mmol/L, near-normal glucose"),
    dict(key="hypo", term="severe hypoglycaemia", milder="hypoglycaemia", lt="found unconscious, glucose 1.4 mmol/L",
         lat=(1, 60), alts=["a missed meal", "a sulfonylurea taken as well"], test="capillary glucose 1.9 mmol/L"),
    dict(key="pe", term="pulmonary embolism", ime=True, lt="right-heart strain with hypotension", fatal=True,
         lat=(14, 365), alts=["a long-haul flight", "immobility after knee surgery"], test="CT pulmonary angiogram positive"),
    dict(key="myocard", term="myocarditis", ime=True, fatal=True, dis="reduced ejection fraction at follow-up",
         lat=(2, 30), alts=["a viral infection the week before"], test="troponin raised, cardiac MRI consistent"),
    dict(key="suic", term="suicidal ideation", ime=True, lat=(7, 90),
         alts=["long-standing depression", "a recent bereavement"]),
    dict(key="photo", term="photosensitivity rash", milder="photosensitivity reaction", minor=True, lat=(3, 60),
         alts=["a tetracycline antibiotic", "a beach holiday without sunscreen"]),
    dict(key="headache", term="headache", milder="headache", minor=True, lat=(0, 14),
         alts=["tension-type headaches for years"]),
    dict(key="nausea", term="nausea and vomiting", milder="nausea", minor=True, lat=(0, 14),
         alts=["viral gastroenteritis in the household"]),
    dict(key="dizzy", term="dizziness", milder="dizziness", minor=True, lat=(0, 30),
         alts=["an antihypertensive dose increase"]),
    dict(key="pruritus", term="generalised pruritus", milder="pruritus", minor=True, lat=(1, 30),
         alts=["a new washing powder"]),
    dict(key="alopecia", term="hair loss", milder="alopecia", minor=True, lat=(30, 180),
         alts=["a severe febrile illness three months earlier"]),
    dict(key="vision", term="blurred vision", milder="visual impairment", minor=True, lat=(1, 60),
         alts=["migraine with aura"]),
    dict(key="injsite", term="injection-site necrosis", milder="injection-site reactions", lat=(1, 14),
         alts=["repeated injection into the same site"]),
    dict(key="insomnia", term="insomnia", milder="sleep disorder", minor=True, lat=(1, 30), alts=["shift work"]),
]]
EV = {e["key"]: e for e in EVENTS}
EXTRA_IME = ["aplastic anaemia", "toxic epidermal necrolysis", "hepatic failure", "ventricular fibrillation",
             "neuroleptic malignant syndrome", "optic neuritis", "intestinal perforation", "status epilepticus"]
COUNTRIES = ["Germany", "Canada", "Brazil", "Japan", "Spain", "Australia", "Poland", "South Korea", "Mexico",
             "Sweden", "Italy", "India", "Netherlands", "Turkey", "Argentina", "France", "Ireland", "Egypt",
             "Portugal", "Chile", "Norway", "Greece"]
MAH_A = ["Alder", "Basalt", "Cobalt", "Ember", "Fernhill", "Ironbark", "Lakeshore", "Oakridge", "Redfern",
         "Saltmarsh", "Upland", "Valewood", "Westbrook", "Yarrow", "Bluestem", "Kestrel", "Moorfield", "Larchmont"]


def _product(rng):
    c = pick(rng, CLASSES)
    return {"brand": pick(rng, BRAND_A) + pick(rng, BRAND_B), "inn": pick(rng, INN_A) + pick(rng, INN_MID) + c["stem"],
            "cls": c["key"], "class_name": c["name"], "indication": pick(rng, c["ind"]), "form": c["form"],
            "strength": pick(rng, c["strengths"])}


def _fits(prod, e):
    """Events that only make sense for some products (injection sites, diabetic emergencies)."""
    k = e["key"]
    if k == "injsite":
        return "pen" in prod["form"] or "syringe" in prod["form"]
    if k == "dka":
        return prod["cls"] in ("sglt2", "glp1", "ins")
    if k == "hypo":
        return prod["cls"] in ("sglt2", "glp1", "ins", "fq")
    return True


def _other_brand(rng, brand):
    while True:
        b = pick(rng, BRAND_A) + pick(rng, BRAND_B)
        if b != brand:
            return b


def _mah(rng):
    return f"{pick(rng, MAH_A)} {pick(rng, ['Pharma', 'Therapeutics', 'Pharmaceuticals', 'Biopharma', 'Life Sciences'])}"


def _sop_no(rng, prefix="PV"):
    return f"{prefix}-{rng.randint(10, 79):03d} v{rng.randint(2, 12)}"


def _age_sex(rng):
    return rng.randint(19, 88), pick(rng, ["female", "male"])


def _cap(s):
    return s[0].upper() + s[1:] if s else s


# =========================================================================== schema 1
class PostmarketingIcsrCausality(Schema):
    name = "postmarketing_icsr_causality"
    description = ("A spontaneous post-marketing report from a health professional, assessed for seriousness, "
                   "labelling and causality under the company's case-assessment SOP.")
    surfaces = {"json": 0.5, "case_summary": 0.25, "medical_review_note": 0.25}

    SERIOUS = [
        "Seriousness: record the most severe criterion met, in this order -- death or persistent disability; "
        "life-threatening (a medical judgement of immediate risk of death; a ticked box alone does not count); "
        "inpatient admission or a prolonged stay (observation counts only if 24 h or longer; an emergency visit "
        "alone never does); an event on the company IME list; otherwise non-serious.",
        "Rank and record the highest criterion met: (1) fatal or persistently disabling; (2) life-threatening, judged "
        "from the clinical details rather than the reporter's tick; (3) hospitalisation -- admission, prolongation, "
        "or observation lasting at least 24 hours, not an emergency-department visit on its own; (4) medically "
        "important per the IME list; (5) otherwise non-serious."]
    EXPECT = [
        "Expectedness: listed only if the CCDS names the event at the nature and severity reported; a milder listed "
        "term or a drug-class statement leaves it unlisted.",
        "Treat the event as labelled only when the CCDS carries it at the reported specificity and severity. Milder "
        "or less specific listed terms, and class statements, do not count."]
    CAUSAL = [
        "Causality scale: certain -- fitting time course, no other explanation, resolved on withdrawal and recurred "
        "on re-exposure; probable -- fitting time course, at most a minor competing factor, improved on withdrawal; "
        "possible -- fitting time course but another explanation remains or the stop response is absent, negative or "
        "unknown; unlikely -- onset outside the usual window or before the first dose, or a clearly better "
        "explanation; unassessable -- essential dates or facts missing or contradictory, follow-up exhausted.",
        "Company causality categories: CERTAIN needs typical timing, no competing cause, improvement on stopping AND "
        "recurrence on restarting. PROBABLE needs typical timing, nothing more than a minor competing factor, and "
        "improvement on stopping. POSSIBLE: typical timing but a competing cause, or no positive stop response. "
        "UNLIKELY: atypical timing (outside the usual window, or before the first dose) or a clearly stronger "
        "explanation. UNASSESSABLE: the report cannot be judged and cannot be supplemented."]
    HOSP = {
        "none": ["managed without hospital attendance", "treated by the family doctor"],
        "ed": ["seen in the emergency department and discharged after {h} hours",
               "emergency-department visit, home the same day ({h} h)"],
        "obs_short": ["kept in a clinical decision unit for {h} hours, never formally admitted",
                      "{h}-hour observation stay without inpatient admission"],
        "obs_long": ["held on the observation ward for {h} hours", "observation stay of {h} hours before discharge"],
        "admitted": ["admitted to a medical ward for {d} days", "inpatient admission, {d} days"],
        "prolonged": ["already an inpatient for elective surgery; discharge delayed {d} days by the event",
                      "event occurred during a planned admission and extended it by {d} days"],
    }

    def sample_factors(self, rng):
        reg = regime(rng)
        prod = _product(rng)
        sev = pick_w(rng, {
            "clear": {"not_serious": 3, "hospitalisation": 2, "important_medical_event": 1.5, "life_threatening": 1.2,
                      "fatal_or_disabling": 1.5},
            "borderline": {"not_serious": 2.5, "hospitalisation": 2.5, "important_medical_event": 2,
                           "life_threatening": 1.3, "fatal_or_disabling": 0.8},
            "conflicting": {"not_serious": 2, "hospitalisation": 2, "important_medical_event": 1.5,
                            "life_threatening": 2, "fatal_or_disabling": 1}}[reg])
        pool = {"not_serious": [e for e in EVENTS if not e["ime"]],
                "important_medical_event": [e for e in EVENTS if e["ime"]],
                "life_threatening": [e for e in EVENTS if e["lt"]],
                "fatal_or_disabling": [e for e in EVENTS if e["fatal"] or e["dis"]],
                "hospitalisation": [e for e in EVENTS if not e["minor"]]}[sev]
        e = pick(rng, [x for x in pool if _fits(prod, x)])
        lt_real, lt_ticked, outcome = False, False, pick(rng, ["recovered", "recovering", "not recovered"])
        if sev == "not_serious":
            hosp = pick_w(rng, {"none": 3 if reg == "clear" else 1, "ed": 2, "obs_short": 1.5})
            lt_ticked = reg == "conflicting" and chance(rng, 0.5)
        elif sev == "important_medical_event":
            hosp = pick_w(rng, {"none": 2, "ed": 2, "obs_short": 1.2})
            lt_ticked = reg == "conflicting" and chance(rng, 0.3)
        elif sev == "hospitalisation":
            hosp = pick_w(rng, {"admitted": 3, "prolonged": 1.2, "obs_long": 3 if reg == "borderline" else 1})
            lt_ticked = reg == "conflicting" and chance(rng, 0.4)
            outcome = pick(rng, ["recovered", "recovering"])
        elif sev == "life_threatening":
            hosp, lt_real = "admitted", True
            lt_ticked = chance(rng, 0.3 if reg != "clear" else 0.9)
            outcome = pick(rng, ["recovered", "recovering"])
        else:
            hosp = "admitted"
            outcome = "fatal" if e["fatal"] and (not e["dis"] or chance(rng, 0.55)) else "sequelae"
            lt_real = bool(e["lt"]) and chance(rng, 0.4)
            lt_ticked = lt_real
        hours = (rng.randint(3, 11) if hosp == "ed" else near(rng, 20, 3, lo=8, hi=23, integer=True)
                 if hosp == "obs_short" else near(rng, 26, 2, lo=24, hi=40, integer=True))
        stay_days = rng.randint(2, 12)

        # ---------------------------------------------------------------- causality evidence
        target = pick_w(rng, {
            "clear": {"certain": 2.2, "probable": 2.5, "unlikely": 2.5, "unassessable": 1.2, "possible": 0.8},
            "borderline": {"probable": 2.5, "possible": 3.5, "unlikely": 2.5, "certain": 0.8, "unassessable": 0.8},
            "conflicting": {"possible": 3, "unlikely": 3, "probable": 1.5, "unassessable": 2, "certain": 0.5}}[reg])
        if outcome == "fatal" and target in ("certain", "probable"):
            target = "possible"
        lo, hi = e["lat"]
        timing, alt, dechal, rechal, gap = "typical", "none", "positive", "none", None
        if target == "certain":
            rechal = "positive"
        elif target == "probable":
            alt = pick(rng, ["none", "weak"])
        elif target == "possible":
            if chance(rng, 0.5):
                alt, dechal = "moderate", pick(rng, ["positive", "not_done", "unknown", "negative"])
            else:
                alt, dechal = pick(rng, ["none", "weak"]), pick(rng, ["not_done", "unknown", "negative"])
        elif target == "unlikely":
            how = pick(rng, ["before", "atypical", "strong"])
            if how == "before":
                timing = "before_start"
            elif how == "atypical":
                timing = "atypical"
            else:
                alt = "strong"
            dechal = pick(rng, ["positive", "not_done", "negative", "unknown"])
            if reg == "conflicting" and chance(rng, 0.5):
                dechal = "positive"                     # a stop response that makes the product look guilty
        else:
            gap = pick(rng, ["no_dates", "contradictory"])
            alt = pick(rng, ["none", "weak", "moderate"])
            dechal = pick(rng, ["positive", "unknown", "not_done"])
        if outcome == "fatal":
            dechal, rechal = "died", "none"
        # latency consistent with the timing class; borderline cases sit at the window edge
        if timing == "typical":
            lat = (rng.randint(max(lo, hi - max(1, (hi - lo) // 6)), hi) if reg == "borderline" and chance(rng, 0.5)
                   else rng.randint(lo, hi))
        elif timing == "atypical":
            lat = (int(hi * rng.uniform(1.15, 1.5)) + 1 if reg == "borderline" else int(hi * rng.uniform(2, 5)) + 3)
            if lo >= 10 and chance(rng, 0.4):
                lat = rng.randint(0, max(0, lo // 3))
        else:
            lat = -rng.randint(3, 40)
        start = rand_date(rng)
        onset = start + timedelta(days=lat)
        received = (onset if lat >= 0 else start) + timedelta(days=rng.randint(3, 40))
        ccds = pick_w(rng, {"listed": 4.2, "listed_milder": 2.5, "unlisted": 2.6, "class_only": 1.3})
        if ccds == "listed_milder" and (not e["milder"] or e["milder"] == e["term"]):
            ccds = "listed" if e["milder"] == e["term"] else "unlisted"
        others = [x for x in EVENTS if x["key"] != e["key"] and x["milder"]]
        view = {"certain": "related", "probable": "related", "possible": pick(rng, ["possibly related", "related"]),
                "unlikely": pick(rng, ["unlikely related", "not related"]),
                "unassessable": pick(rng, ["possibly related", "not stated"])}[target]
        if reg == "conflicting":
            view = {"related": "not related", "possibly related": "related", "unlikely related": "related",
                    "not related": "related", "not stated": "related"}[view]
        age, sex = _age_sex(rng)
        return {"regime": reg, "product": prod, "event": e["key"], "sev_target": sev, "causal_target": target,
                "hospital": hosp, "hours": hours, "stay_days": stay_days, "lt_real": lt_real, "lt_ticked": lt_ticked,
                "outcome": outcome, "timing": timing, "latency_days": lat, "alt": alt,
                "alt_text": pick(rng, e["alts"]), "dechallenge": dechal, "rechallenge": rechal, "info_gap": gap,
                "ccds": ccds, "ccds_others": [x["milder"] for x in rng.sample(others, rng.randint(3, 5))],
                "ccds_version": rng.randint(4, 21), "ime_excerpt": rng.sample([x["term"] for x in EVENTS if x["ime"]
                                                                            and x["key"] != e["key"]] + EXTRA_IME, 5),
                "reporter_view": view, "reporter": pick(rng, ["hospital physician", "general practitioner",
                                                              "community pharmacist", "specialist nurse",
                                                              "hospital pharmacist", "consultant specialist"]),
                "country": pick(rng, COUNTRIES), "age": age, "sex": sex, "start": start, "onset": onset,
                "received": received, "case_id": ident(rng, "MAH", 8, "-"), "sop": _sop_no(rng),
                "processor": person(rng), "physician": person(rng), "followups": rng.randint(2, 3),
                "dechal_days": rng.randint(2, 14), "rechal_weeks": rng.randint(2, 10)}

    # ------------------------------------------------------------------ text pieces
    def _hosp(self, f, rng):
        return pick(rng, self.HOSP[f["hospital"]]).format(h=f["hours"], d=f["stay_days"])

    def _details(self, f, e):
        if f["lt_real"]:
            return e["lt"]
        if f["lt_ticked"]:
            return "observations stable throughout; no airway, circulatory or neurological compromise recorded"
        return None

    def _outcome(self, f, e):
        return {"recovered": "recovered", "recovering": "recovering", "not recovered": "not yet recovered",
                "fatal": f"died on day {min(f['stay_days'], 6)} of the admission",
                "sequelae": f"recovered with sequelae: {e['dis']}"}[f["outcome"]]

    def _action(self, f):
        return {"positive": f"product stopped; improved within {f['dechal_days']} days",
                "negative": "product stopped; no improvement after three weeks",
                "not_done": "product continued unchanged", "unknown": "action with the product not reported",
                "died": "not assessable (patient died)"}[f["dechallenge"]]

    def _rechal(self, f):
        return {"positive": f"restarted after {f['rechal_weeks']} weeks; the same reaction returned within days",
                "negative": "restarted without recurrence", "none": None}[f["rechallenge"]]

    def _alt(self, f):
        a = f["alt_text"]
        return {"none": None, "weak": f"mentioned in passing, unconfirmed: {a}",
                "moderate": f"also relevant: {a}",
                "strong": f"{a}; the treating team regarded this as the main cause"}[f["alt"]]

    def _dates(self, f, rng):
        """(start text, onset text, gap note)"""
        if f["info_gap"] == "no_dates":
            return ("not given ('a while ago')", "not given",
                    f"{f['followups']} follow-up requests for dates unanswered; reporter has since left the practice")
        if f["info_gap"] == "contradictory":
            return (fmt_date(f["start"], rng, "iso"), fmt_date(f["onset"] if f["latency_days"] >= 0 else
                                                               f["start"] + timedelta(days=5), rng, "iso"),
                    "narrative states the symptoms began before the first dose, contradicting the form; "
                    f"{f['followups']} follow-up requests unanswered")
        return fmt_date(f["start"], rng, "iso"), fmt_date(f["onset"], rng, "iso"), None

    def _ccds(self, f, e):
        listed = list(f["ccds_others"])
        cls_stmt = None
        if f["ccds"] == "listed":
            listed.insert(1, e["term"])
        elif f["ccds"] == "listed_milder":
            listed.insert(1, e["milder"])
        elif f["ccds"] == "class_only":
            cls_stmt = f"{e['term']} has been reported with other medicines of this class"
        return ", ".join(listed), cls_stmt

    def _ime(self, f, e):
        terms = list(f["ime_excerpt"])
        if e["ime"]:
            terms.insert(2, e["term"])
        return "; ".join(terms)

    def render(self, f, rng, surface):
        e, p = EV[f["event"]], f["product"]
        start, onset, gap = self._dates(f, rng)
        ccds, cls_stmt = self._ccds(f, e)
        usual = f"usually {e['lat'][0]} to {e['lat'][1]} days after starting"
        ser, exp, cau = pick(rng, self.SERIOUS), pick(rng, self.EXPECT), pick(rng, self.CAUSAL)
        details = self._details(f, e)
        if surface == "json":
            return dumps(prune({
                "case": {"id": f["case_id"], "received": f["received"].isoformat(), "reporter": f["reporter"],
                         "country": f["country"], "type": "spontaneous, post-marketing"},
                "patient": {"age": f["age"], "sex": f["sex"]},
                "product": {"name": f"{p['brand']} ({p['inn']}) {p['strength']}", "class": p["class_name"],
                            "indication": p["indication"], "start": start, "action": self._action(f),
                            "re_exposure": self._rechal(f)},
                "event": {"term": e["term"], "onset": onset, "course": self._hosp(f, rng),
                          "life_threatening_box": f["lt_ticked"], "clinical_details": details, "test": e["test"],
                          "outcome": self._outcome(f, e)},
                "assessment_inputs": {"other_factors": self._alt(f), "reporter_causality": f["reporter_view"],
                                      "follow_up": gap},
                "labelling": {"ccds": f"v{f['ccds_version']}: {ccds}", "class_statement": cls_stmt,
                              "usual_onset": usual},
                "ime_list_excerpt": self._ime(f, e),
                "sop": {"id": f["sop"], "seriousness": ser, "expectedness": exp, "causality": cau},
            }))
        if surface == "case_summary":
            rows = [("Case", f["case_id"]), ("Received", fmt_date(f["received"], rng)), ("Reporter", f["reporter"]),
                    ("Country", f["country"]), ("Patient", f"{f['age']}-year-old {f['sex']}"),
                    ("Suspect", f"{p['brand']} ({p['inn']}) {p['strength']}, {p['class_name']}"),
                    ("Indication", p["indication"]), ("Start", start), ("Event", e["term"]), ("Onset", onset),
                    ("Course", self._hosp(f, rng)), ("Life-threatening box", "X" if f["lt_ticked"] else "-"),
                    ("Clinical details", details), ("Key test", e["test"]), ("Outcome", self._outcome(f, e)),
                    ("Action taken", self._action(f)), ("Re-exposure", self._rechal(f)),
                    ("Other factors", self._alt(f)), ("Reporter causality", f["reporter_view"]),
                    ("Follow-up", gap)]
            return (f"CASE SUMMARY (safety database)\n{kv_lines(rows, rng)}\n\nCCDS v{f['ccds_version']} listed "
                    f"reactions: {ccds}" + (f". Class statement: {cls_stmt}." if cls_stmt else ".")
                    + f" Onset is {usual}.\nIME list (excerpt): {self._ime(f, e)}\n\n{f['sop']}: {ser} {exp} {cau}")
        s = [f"{f['reporter'].capitalize()} in {f['country']} reports {e['term']} in a {f['age']}-year-old {f['sex']} "
             f"taking {p['brand']} ({p['inn']}) for {p['indication']}.",
             f"Start {start}, onset {onset}." if not f["info_gap"] else f"Start: {start}; onset: {onset}.",
             f"{_cap(self._hosp(f, rng))}.",
             (f"Details: {details}." if details else ""),
             ("Reporter ticked life-threatening." if f["lt_ticked"] else ""),
             (f"{_cap(e['test'])}." if e["test"] else ""), f"Outcome: {self._outcome(f, e)}.",
             f"{_cap(self._action(f))}.", (f"{_cap(self._rechal(f))}." if self._rechal(f) else ""),
             (f"{_cap(self._alt(f))}." if self._alt(f) else "No other medicines or relevant history reported."),
             f"Reporter's causality: {f['reporter_view']}.", (f"Note: {gap}." if gap else "")]
        return (f"MEDICAL REVIEW NOTE -- {f['case_id']} (prepared by {f['processor']} for {f['physician']})\n\n"
                + noisy(join_sentences(s), rng, 0.1)
                + f"\n\nLabelling: CCDS v{f['ccds_version']} lists {ccds}"
                + (f"; class statement: {cls_stmt}" if cls_stmt else "") + f"; onset {usual}.\n"
                + f"IME list (excerpt): {self._ime(f, e)}\n\nSOP {f['sop']}:\n- {ser}\n- {exp}\n- {cau}")

    def questions(self):
        return {
            "who_umc_category": choice("Which causality category does the company scale give this case?", {
                "certain": "Certain: a fitting time course, no other explanation, recovery on stopping and "
                           "recurrence on re-exposure.",
                "possible": "Possible: the timing fits, but another explanation remains or the response to stopping "
                            "is unknown or negative.",
                "probable": "Probable: the timing fits, other causes are unlikely, and the event improved once the "
                            "product was stopped.",
                "unassessable": "Unassessable: essential dates or facts are missing or contradictory and cannot be "
                                "obtained.",
                "unlikely": "Unlikely: the onset timing argues against the product, or another cause clearly "
                            "explains the event better.",
            }),
            "seriousness_basis": choice("Which seriousness criterion, if any, best describes this case?", {
                "fatal_or_disabling": "The patient died, or was left with a persistent or significant disability.",
                "hospitalisation": "The event caused an inpatient admission or prolonged an existing stay.",
                "important_medical_event": "No higher criterion applies, but the event is medically important in "
                                           "its own right.",
                "life_threatening": "The patient was at immediate risk of death from the event as it happened.",
                "not_serious": "None of the seriousness criteria applies.",
            }),
            "listed_in_ccds": noul(
                "The reported event is listed in the product's core data sheet at the nature and severity reported.",
                yes="The CCDS names this event at the specificity and severity described in the case.",
                no="The CCDS has only a milder or less specific term, only a class statement, or nothing relevant."),
            "alternative_cause_strength": score("How strongly do other causes account for the event?", [
                "Not at all: nothing else on record explains it.",
                "Weakly: a factor is mentioned but is unconfirmed or rarely causes this.",
                "Plausibly: another drug or condition could well have caused it.",
                "Strongly: a documented alternative explains it better than the product does.",
            ]),
            "temporal_fit": score("How well does the time course fit a reaction to the product?", [
                "Poorly: onset before the first dose, or no usable dates.",
                "Loosely: onset outside the usual window for this reaction.",
                "Reasonably: onset in the usual window, but no supportive response to stopping.",
                "Closely: onset in the usual window and improvement after stopping.",
            ]),
        }

    def answer_key(self, f):
        """Rule-of-thumb answer key for the spread check only; never used as a label."""
        e = EV[f["event"]]
        if f["info_gap"]:
            who = "unassessable"
        elif f["timing"] != "typical" or f["alt"] == "strong":
            who = "unlikely"
        elif f["dechallenge"] == "positive" and f["alt"] == "none" and f["rechallenge"] == "positive":
            who = "certain"
        elif f["dechallenge"] == "positive" and f["alt"] in ("none", "weak"):
            who = "probable"
        else:
            who = "possible"
        if f["outcome"] in ("fatal", "sequelae"):
            sev = "fatal_or_disabling"
        elif f["lt_real"]:
            sev = "life_threatening"
        elif f["hospital"] in ("admitted", "prolonged", "obs_long"):
            sev = "hospitalisation"
        elif e["ime"]:
            sev = "important_medical_event"
        else:
            sev = "not_serious"
        if f["info_gap"] or f["timing"] == "before_start":
            tf = 0
        elif f["timing"] == "atypical":
            tf = 1
        else:
            tf = 3 if f["dechallenge"] == "positive" else 2
        return {"who_umc_category": who, "seriousness_basis": sev,
                "listed_in_ccds": "yes" if f["ccds"] == "listed" else "no",
                "alternative_cause_strength": {"none": 0, "weak": 1, "moderate": 2, "strong": 3}[f["alt"]],
                "temporal_fit": tf}


# =========================================================================== schema 2
class ConsumerReportValidity(Schema):
    name = "consumer_report_validity"
    description = ("A report from a patient, relative or pharmacist received through a non-clinical channel, "
                   "checked for case validity and product attribution before case processing.")
    surfaces = {"json": 0.5, "intake_record": 0.25, "call_note": 0.25}

    CHANNELS = ["patient support programme phone line", "company website safety form", "medical information line",
                "email to the drug-safety mailbox", "direct message to the brand's social media account",
                "letter to the company's registered office"]
    MIN = ["Validity needs all four minimum elements: a reporter we can contact; an identifiable patient (initials, "
           "age, sex, or a specific person such as 'my husband' -- not a group or an unnamed friend with no "
           "detail); a suspect product that is ours; and at least one adverse event or symptom.",
           "Four elements, all required: contactable reporter; identifiable patient (any of initials, age, sex, or "
           "one specific individual; 'people', a group or 'a friend' with no detail does not qualify); our product "
           "as suspect; an adverse event, where a symptom is enough."]
    ATTR = ["Attribution: a brand name, or a batch number found in our release records, makes the product ours; a "
            "substance name alone counts as ours only where no other company sells it in that country.",
            "Product attribution: substance-name-only reports are ours when we are the sole supplier in the country; "
            "otherwise a batch number or pack details are needed. Another company's brand is never ours."]
    ROUTE = ["If an element is missing and the reporter can be contacted, open follow-up; the case is not valid "
             "until the element arrives. An uncontactable reporter means a non-valid report, logged only. Quality "
             "complaints and questions without any symptom go to the complaints or medical-information teams.",
             "Missing element plus contactable reporter: chase it before validating. Uncontactable reporter: "
             "non-valid, log only. No symptom at all (defect or question only): not a safety case."]
    CONFIRM = ["Consumer reports mentioning an emergency visit or a hospital stay are processed as valid and flagged "
               "for medical confirmation from the treating doctor, unless the consumer refused contact with the "
               "doctor.",
               "Possibly serious consumer cases (emergency visit or admission) also need medical confirmation: ask "
               "the treating doctor when the consumer allows it; if contact was refused, process on the "
               "consumer's account alone."]
    CLOCK = ["Day 0 is the date of receipt; possibly serious valid cases are due to the authority within 15 calendar "
             "days, non-serious ones go in the periodic submission.",
             "Clock: day 0 = receipt. Serious: 15 calendar days. Non-serious: periodic report."]
    VAGUE = ["several people in an online support group", "a friend (no other details)",
             "'some people at work'", "a neighbour's relative, no details given", "customers in general"]
    RELATION = ["spouse", "adult son", "adult daughter", "sister", "partner", "brother", "mother", "father"]

    def sample_factors(self, rng):
        reg = regime(rng)
        prod = _product(rng)
        kind = pick_w(rng, {"patient": 4, "relative": 3.5, "carer": 1, "pharmacist": 1.2})
        contact = pick_w(rng, {"phone_email": 4, "email_only": 2, "none": 1.3})
        patient = pick_w(rng, {"specific": 4.5, "vague": 1.8})
        naming = pick_w(rng, {"brand": 2.6, "inn_sole": 1.3, "inn_multi": 2.4, "inn_multi_batch": 1.2, "other_brand": 1.7})
        event_kind = pick_w(rng, {"symptom": 5, "quality_with_symptom": 1.4, "quality_only": 1.3, "question_only": 0.8})
        hint = pick_w(rng, {"none": 3, "ed": 1.3, "admitted": 1.7})
        consent = pick_w(rng, {"yes": 2, "no": 1.3, "not_asked": 1.5})
        if reg == "clear":
            if chance(rng, 0.55):
                contact, patient, naming, event_kind = "phone_email", "specific", "brand", "symptom"
            else:
                event_kind = pick(rng, ["quality_only", "question_only"]) if chance(rng, 0.5) else event_kind
                naming = "other_brand" if chance(rng, 0.65) else naming
        elif reg == "borderline":
            # exactly one element is the doubtful one
            which = pick(rng, ["patient", "product", "contact", "event"])
            contact, patient, naming, event_kind = "phone_email", "specific", pick(rng, ["brand", "inn_sole"]), "symptom"
            if which == "patient":
                patient = "vague"
            elif which == "product":
                naming = pick(rng, ["inn_multi", "inn_multi", "inn_multi_batch", "inn_sole"])
            elif which == "contact":
                contact = pick(rng, ["email_only", "none"])
            else:
                event_kind = pick(rng, ["quality_with_symptom", "quality_only"])
        else:
            # a serious-sounding story attached to a report that may not stand up, or the reverse
            hint = pick(rng, ["ed", "admitted"])
            if chance(rng, 0.5):
                contact = pick(rng, ["none", "email_only"])
            if chance(rng, 0.6):
                naming = pick(rng, ["inn_multi", "inn_multi_batch", "other_brand", "other_brand", "brand"])
        if kind == "patient":
            patient = "specific"                      # a self-report always describes one person
        if event_kind in ("quality_only", "question_only"):
            hint = "none"
        details = {k: chance(rng, 0.55) for k in ("timing", "dechallenge", "conmeds", "history")}
        e = pick(rng, [x for x in EVENTS if (not x["ime"] or x["key"] in ("seiz", "anaph", "pe")) and _fits(prod, x)])
        if hint == "none" and not e["minor"] and chance(rng, 0.5):
            e = pick(rng, [x for x in EVENTS if x["minor"]])
        received = rand_date(rng)
        age, sex = _age_sex(rng)
        return {"regime": reg, "product": prod, "reporter_kind": kind, "relation": pick(rng, self.RELATION),
                "contact": contact, "patient": patient, "vague_text": pick(rng, self.VAGUE), "naming": naming,
                "event_kind": event_kind, "event": e["key"], "serious_hint": hint, "hcp_consent": consent,
                "details": details, "days_since_receipt": rng.randint(0, 13), "received": received,
                "channel": pick(rng, self.CHANNELS), "country": pick(rng, COUNTRIES),
                "other_brand": _other_brand(rng, prod["brand"]), "n_other_mahs": rng.randint(2, 6),
                "batch": f"{pick(rng, ['B', 'L', 'K', 'T'])}{rng.randint(10000, 99999)}{pick(rng, ['A', 'C', 'X'])}",
                "age": age, "sex": sex, "intake_id": ident(rng, "IN", 7, "-"), "agent": person(rng),
                "reporter_name": person(rng), "weeks_on": rng.randint(1, 30), "nights": rng.randint(1, 5),
                "sop": _sop_no(rng, "PV-INT")}

    # ------------------------------------------------------------------ text pieces
    def _product_ref(self, f):
        p = f["product"]
        return {"brand": p["brand"], "inn_sole": f"'{p['inn']}' (substance name only)",
                "inn_multi": f"'{p['inn']}' (substance name only; no pack or batch details)",
                "inn_multi_batch": f"'{p['inn']}', batch {f['batch']} read from the box",
                "other_brand": f["other_brand"]}[f["naming"]]

    def _market(self, f):
        p = f["product"]
        if f["naming"] == "inn_sole":
            return f"we are the only supplier of {p['inn']} in {f['country']}"
        if f["naming"] in ("inn_multi", "inn_multi_batch"):
            note = f"{f['n_other_mahs']} other companies also sell {p['inn']} in {f['country']}"
            return note + (f"; batch {f['batch']} is in our release records" if f["naming"] == "inn_multi_batch" else "")
        if f["naming"] == "other_brand":
            return f"{f['other_brand']} is another company's {p['inn']} product"
        return None

    def _reporter(self, f):
        who = {"patient": "the patient", "relative": f"the patient's {f['relation']}", "carer": "a paid carer",
               "pharmacist": "a community pharmacist"}[f["reporter_kind"]]
        c = {"phone_email": f"name and phone number given: {f['reporter_name']}",
             "email_only": "email address only, no name", "none": "declined to give a name or any contact details"}
        return who, c[f["contact"]]

    def _patient(self, f):
        if f["reporter_kind"] == "patient":
            return f"the reporter, {f['age']}, {f['sex']}"
        if f["patient"] == "vague":
            return f["vague_text"]
        if f["reporter_kind"] == "pharmacist":
            return f"a customer, about {f['age']}"
        return f"their {f['relation']}" if f["relation"] in ("mother", "father", "sister", "brother") else \
            f"their {f['relation']}, {f['age']}"

    def _story(self, f):
        e, ref = EV[f["event"]], self._product_ref(f)
        if f["event_kind"] == "question_only":
            return f"Asked whether {ref} can be taken with grapefruit juice; no problems reported."
        defect = self._defect(f)
        if f["event_kind"] == "quality_only":
            return f"In a new pack of {ref}, {defect}; no symptoms reported."
        s = ([f"In a new pack of {ref}, {defect}; after using it the patient had {e['term']}."]
             if f["event_kind"] == "quality_with_symptom" else [f"Reports {e['term']} while taking {ref}."])
        d = f["details"]
        if d["timing"]:
            s.append(f"Started about {f['weeks_on']} weeks before the problem began.")
        if d["dechallenge"]:
            s.append("Stopped taking it and felt better within a few days.")
        if d["conmeds"]:
            s.append(f"Also takes a {pick_cls(f)}.")
        if d["history"]:
            s.append("Has a long-standing heart condition.")
        s.append({"none": "", "ed": "Went to the emergency department and was sent home the same day.",
                  "admitted": f"Spent {f['nights']} night{'s' if f['nights'] > 1 else ''} in hospital."}[
                      f["serious_hint"]])
        return join_sentences(s)

    def _defect(self, f):
        form = f["product"]["form"]
        if "pen" in form or "syringe" in form:
            return "the device leaked and the solution looked cloudy"
        if "solution" in form:
            return "the solution was discoloured with visible particles"
        return "the tablets were crumbly and discoloured" if "tablet" in form else "the capsules were split and sticky"

    def _consent(self, f):
        return {"yes": "agreed that the company may contact the treating doctor",
                "no": "did not want the doctor contacted", "not_asked": "was not asked about contacting the doctor"}[
            f["hcp_consent"]]

    def render(self, f, rng, surface):
        who, contact = self._reporter(f)
        today = f["received"] + timedelta(days=f["days_since_receipt"])
        rules = [pick(rng, self.MIN), pick(rng, self.ATTR), pick(rng, self.ROUTE), pick(rng, self.CONFIRM),
                 pick(rng, self.CLOCK)]
        story = noisy(self._story(f), rng, 0.12)
        if surface == "json":
            return dumps(prune({
                "intake": {"id": f["intake_id"], "channel": f["channel"], "received": f["received"].isoformat(),
                           "today": today.isoformat(), "country": f["country"], "taken_by": f["agent"]},
                "reporter": {"who": who, "contact": contact, "doctor_contact": self._consent(f)},
                "patient": self._patient(f),
                "account": story,
                "product_check": {"as_reported": self._product_ref(f), "market_note": self._market(f)},
                "sop": {"id": f["sop"], "rules": rules},
            }))
        if surface == "intake_record":
            rows = [("Intake", f["intake_id"]), ("Channel", f["channel"]), ("Received", fmt_date(f["received"], rng)),
                    ("Today", fmt_date(today, rng)), ("Country", f["country"]), ("Reporter", who),
                    ("Contact", contact), ("Patient", self._patient(f)), ("Product as reported", self._product_ref(f)),
                    ("Market note", self._market(f)), ("Doctor contact", self._consent(f))]
            return ("SAFETY INTAKE RECORD (non-HCP channel)\n" + kv_lines(rows, rng) + f"\n\nAccount:\n{story}\n\n"
                    + f"{f['sop']} excerpt:\n" + "\n".join(f"- {r}" for r in rules))
        para = [f"{f['channel'].capitalize()} contact received {fmt_date(f['received'], rng)}; written up "
                f"{fmt_date(today, rng)} by {f['agent']}.",
                f"Reporter: {who} ({contact}).", f"Patient: {self._patient(f)}.", story,
                f"The reporter {self._consent(f)}."]
        if self._market(f):
            para.append(f"Product check: {self._market(f)}.")
        return ("INTAKE NOTE " + f["intake_id"] + "\n\n" + join_sentences(para) + f"\n\nRules ({f['sop']}): "
                + " ".join(rules))

    def questions(self):
        return {
            "case_validity_status": choice("What is this report's status under the company's validity rules?", {
                "incomplete_follow_up": "Not yet a valid case: a minimum element is missing and the reporter can be "
                                        "asked for it.",
                "not_reportable": "Not a safety case for this company: no adverse event, not our product, or no way "
                                  "to reach the reporter for what is missing.",
                "valid_icsr": "A valid case to process now as it stands.",
                "valid_needs_medical_confirmation": "A valid case to process now, flagged for confirmation by the "
                                                    "treating doctor.",
            }),
            "causality_assessability": score("How well could a reviewer judge causality from what the report says?", [
                "Not at all: no adverse event, or almost nothing about the course.",
                "Barely: one or two relevant facts.",
                "Partly: most of timing, stop response, other medicines and history.",
                "Well: timing, stop response, other medicines and history are all given.",
            ]),
            "follow_up_priority": score("How quickly does this report need follow-up work?", [
                "No case clock: route it to the right team in the normal queue.",
                "Standard: a non-serious case for the periodic report.",
                "Expedited: possibly serious, with more than a week left on the 15-day clock.",
                "Immediate: possibly serious, with a week or less left on the 15-day clock.",
            ]),
            "minimum_criteria_present": noul(
                "All four minimum elements of a valid case are present in this report.",
                yes="The report has a contactable reporter, an identifiable patient, our product and an adverse event.",
                no="At least one minimum element is missing or does not meet the company's definition."),
            "product_attributable_to_company": noul(
                "The suspect product in this report can be attributed to the company."),
        }

    def answer_key(self, f):
        """Rule-of-thumb answer key for the spread check only; never used as a label."""
        reporter_ok = f["contact"] != "none"
        ours = f["naming"] in ("brand", "inn_sole", "inn_multi_batch")
        event_ok = f["event_kind"] in ("symptom", "quality_with_symptom")
        patient_ok = f["patient"] == "specific"
        min_ok = reporter_ok and patient_ok and ours and event_ok
        if not event_ok or f["naming"] == "other_brand" or not reporter_ok:
            st = "not_reportable"
        elif min_ok:
            consumer = f["reporter_kind"] != "pharmacist"
            st = ("valid_needs_medical_confirmation" if consumer and f["serious_hint"] != "none"
                  and f["hcp_consent"] != "no" else "valid_icsr")
        else:
            st = "incomplete_follow_up"
        n = sum(f["details"].values())
        ca = 0 if not event_ok else (0 if n <= 1 else n - 1)
        if st == "not_reportable":
            pr = 0
        elif f["serious_hint"] == "none":
            pr = 1
        else:
            pr = 3 if 15 - f["days_since_receipt"] <= 7 else 2
        return {"case_validity_status": st, "causality_assessability": ca, "follow_up_priority": pr,
                "minimum_criteria_present": "yes" if min_ok else "no",
                "product_attributable_to_company": "yes" if ours else "no"}


def pick_cls(f):
    """A concomitant medicine named by class, chosen deterministically from the case's own factors."""
    opts = ["blood-pressure tablet", "statin", "thyroid replacement tablet", "inhaled steroid", "water tablet",
            "proton pump inhibitor", "antidepressant", "painkiller bought over the counter"]
    return opts[(f["weeks_on"] + f["nights"]) % len(opts)]


# =========================================================================== schema 3
JOURNALS = ["Northern Journal of Clinical Cases", "Pharmacotherapy Case Notes", "Archives of Applied Therapeutics",
            "Clinical Safety Letters", "Bulletin of Hospital Medicine", "Acta Medica Borealis",
            "Journal of Practical Prescribing", "Reports in Clinical Pharmacology", "Hospital Physician Review",
            "Annals of Regional Medicine", "Case Studies in Therapeutics", "International Drug Reaction Digest"]


class LiteratureCaseScreening(Schema):
    name = "literature_case_screening"
    description = ("A hit from the weekly literature search that mentions one of the company's substances, screened "
                   "for whether it yields a reportable individual case.")
    surfaces = {"json": 0.5, "screening_log": 0.25, "reviewer_worksheet": 0.25}

    RULES = [
        ["An article yields a case when it describes at least one identifiable patient (an age or age group, sex, "
         "initials or a single described person) with an adverse reaction to our product; aggregate results alone "
         "are logged for periodic and signal review, without a case.",
         "Create a case per identifiable patient (age or age band, sex, initials, or one described individual) "
         "reacting to our product; cohort percentages and reviews without an individual patient are logged only."],
        ["Substance name only: ours if we hold an authorisation for it in the country where the case occurred; if "
         "the article names another manufacturer's product, or the country is one we do not supply, it is not ours. "
         "If even the full text gives no country, treat the case as ours.",
         "When only the substance is named, attribute to us if we are authorised in the case's country; an explicitly "
         "named other manufacturer, or a country outside our authorisations, means not ours. No country in the full "
         "text either: treat as ours."],
        ["When only the abstract has been screened and the patient or the product source is unclear, order the "
         "full text before deciding; day 0 starts when the elements are confirmed.",
         "Abstract only, and patient or product source unclear: obtain the full text first; the clock starts once "
         "the elements are confirmed."],
    ]
    KINDS = {"case_report": "case report", "case_series": "case series", "letter": "letter to the editor",
             "poster": "conference poster abstract", "cohort": "retrospective cohort study", "review": "narrative review"}
    VIEWS = {"probable": "the authors conclude the reaction was probably caused by {inn}, citing improvement after "
                         "withdrawal{re}",
             "possible": "the authors state that a role for {inn} cannot be excluded",
             "no_view": "the authors note the temporal association with {inn} without giving a view",
             "other": "the authors attribute the event to {alt}"}

    def sample_factors(self, rng):
        reg = regime(rng)
        prod = _product(rng)
        kind = pick_w(rng, {"case_report": 3, "case_series": 1.6, "letter": 1.3, "poster": 1.6, "cohort": 2,
                            "review": 1.3})
        full_text = pick_w(rng, {"available": 3, "abstract_only": 2})
        if kind in ("case_report", "case_series", "letter"):
            ident_ = pick_w(rng, {"yes": 6, "unclear": 1})
        elif kind == "poster":
            ident_ = pick_w(rng, {"yes": 2, "unclear": 2})
        else:
            ident_ = pick_w(rng, {"no": 4, "yes": 1.2})        # a single case detailed inside a cohort or review
        naming = pick_w(rng, {"brand": 3, "inn_country_ours": 2.6, "inn_no_country": 1.6, "inn_other_mah": 1.2,
                              "inn_country_not_ours": 1.2})
        if reg == "clear":
            if chance(rng, 0.5):
                kind, ident_, naming, full_text = "case_report", "yes", "brand", "available"
            else:
                naming = pick(rng, ["inn_other_mah", "inn_country_not_ours"]) if chance(rng, 0.5) else naming
                if naming not in ("inn_other_mah", "inn_country_not_ours"):
                    kind, ident_ = pick(rng, ["cohort", "review"]), "no"
        elif reg == "borderline":
            full_text = pick_w(rng, {"available": 1, "abstract_only": 2})
            if chance(rng, 0.5):
                naming = pick(rng, ["inn_no_country", "inn_country_ours"])
        if ident_ == "unclear" and full_text == "available":
            ident_ = pick(rng, ["yes", "no"])
        e = pick(rng, [x for x in (EVENTS if chance(rng, 0.2) else [y for y in EVENTS if not y["minor"]])
                       if _fits(prod, x)])
        serious = (not e["minor"]) and chance(rng, 0.7 if reg != "borderline" else 0.55)
        ccds = pick_w(rng, {"listed": 3, "unlisted": 3, "listed_milder": 1.5})
        if ccds == "listed_milder" and not e["milder"]:
            ccds = "unlisted"
        view = pick_w(rng, {"probable": 2, "possible": 2.5, "no_view": 2, "other": 1.5})
        ours = sorted(rng.sample(COUNTRIES, rng.randint(6, 10)))
        not_ours = [c for c in COUNTRIES if c not in ours]
        multi = sorted(rng.sample(ours, rng.randint(1, max(1, len(ours) // 2))))
        country = (pick(rng, not_ours) if naming == "inn_country_not_ours" else
                   pick(rng, ours) if naming in ("inn_country_ours", "brand") else None)
        if reg == "conflicting":
            # a striking, serious report that is not ours or not individual -- or a dull one that is
            serious = not e["minor"]
            view = pick(rng, ["probable", "other"])
        age, sex = _age_sex(rng)
        return {"regime": reg, "product": prod, "kind": kind, "full_text": full_text, "identifiable": ident_,
                "naming": naming, "event": e["key"], "serious": serious, "ccds": ccds, "author_view": view,
                "rechallenge_in_paper": view == "probable" and chance(rng, 0.35), "our_countries": ours,
                "multi_mah_countries": multi, "country": country, "other_brand": _other_brand(rng, prod["brand"]),
                "other_mah": _mah(rng), "alt": pick(rng, e["alts"]), "age": age, "sex": sex,
                "n_series": rng.randint(2, 7), "n_cohort": rng.randint(80, 2400), "pct": round(rng.uniform(0.4, 9), 1),
                "journal": pick(rng, JOURNALS), "year": rng.randint(2024, 2026), "vol": rng.randint(8, 61),
                "page": rng.randint(101, 990), "author": f"{person(rng).split()[1]} {pick(rng, 'ABCDEFGHJKLMNPRSTW')}",
                "hit_id": ident(rng, "LIT", 6, "-"), "screened": rand_date(rng), "screener": person(rng),
                "ccds_others": [x["milder"] for x in rng.sample([x for x in EVENTS if x["milder"]
                                                                and x["key"] != e["key"]], 4)],
                "sop": _sop_no(rng, "PV-LIT")}

    def _patient_text(self, f):
        if f["identifiable"] == "yes":
            if f["kind"] == "case_series":
                return f"{f['n_series']} patients, each described individually (the first a {f['age']}-year-old {f['sex']})"
            if f["kind"] in ("cohort", "review"):
                return (f"pooled data on {f['n_cohort']} patients, plus one case described in detail: a "
                        f"{f['age']}-year-old {f['sex']}")
            return f"a {f['age']}-year-old {f['sex']}"
        if f["identifiable"] == "unclear":
            return "'a patient' (no age, sex or other detail in the abstract)"
        return (f"{f['pct']}% of {f['n_cohort']} treated patients; no individual patient described"
                if f["kind"] == "cohort" else "summary of published experience; no individual patient described")

    def _product_text(self, f):
        p = f["product"]
        if f["naming"] == "brand":
            return f"{p['brand']} ({p['inn']})"
        if f["naming"] == "inn_other_mah":
            return f"{p['inn']}; the methods name the product as {f['other_brand']} from {f['other_mah']}"
        if f["naming"] == "inn_no_country":
            return f"{p['inn']} (substance only); " + ("the full text gives no country or manufacturer"
                                                       if f["full_text"] == "available" else
                                                       "country and manufacturer not stated in the abstract")
        return f"{p['inn']} (substance only); case from {f['country']}"

    def _ccds(self, f):
        e = EV[f["event"]]
        terms = list(f["ccds_others"])
        if f["ccds"] == "listed":
            terms.insert(2, e["term"])
        elif f["ccds"] == "listed_milder":
            terms.insert(2, e["milder"])
        return ", ".join(terms)

    def _view(self, f):
        p = f["product"]
        return self.VIEWS[f["author_view"]].format(
            inn=p["inn"], alt=f["alt"], re=" and recurrence on re-exposure" if f["rechallenge_in_paper"] else "")

    def _course(self, f):
        e = EV[f["event"]]
        if f["identifiable"] == "no":
            return e["term"] + (" (serious cases, including hospital admissions)" if f["serious"] else
                                " (non-serious, managed as outpatients)")
        if not f["serious"]:
            return f"{e['term']}, managed as an outpatient"
        return pick_course(f, e)

    def render(self, f, rng, surface):
        p = f["product"]
        rules = [pick(rng, r) for r in self.RULES]
        e = EV[f["event"]]
        title = f"{_cap(e['term'])} {pick(rng, ['associated with', 'during treatment with', 'after starting'])} " \
                f"{p['inn']}: {self.KINDS[f['kind']]}"
        cite = (f"{f['author']} et al. {title}. {f['journal']} {f['year']};{f['vol']}:{f['page']}-{f['page'] + 4}"
                + (" (abstract only; full text not yet obtained)" if f["full_text"] == "abstract_only" else
                   " (full text on file)"))
        markets = (f"our authorisations for {p['inn']}: {', '.join(f['our_countries'])}; other suppliers also "
                   f"present in {', '.join(f['multi_mah_countries'])}")
        if surface == "json":
            return dumps(prune({
                "hit": {"id": f["hit_id"], "screened": f["screened"].isoformat(), "screener": f["screener"],
                        "citation": cite, "article_type": self.KINDS[f["kind"]]},
                "content": {"patients": self._patient_text(f), "product": self._product_text(f),
                            "reaction": self._course(f), "authors_view": self._view(f)},
                "company_reference": {"markets": markets, "ccds_terms": self._ccds(f)},
                "sop": {"id": f["sop"], "rules": rules},
            }))
        if surface == "screening_log":
            rows = [["hit", f["hit_id"]], ["type", self.KINDS[f["kind"]]], ["patients", self._patient_text(f)],
                    ["product", self._product_text(f)], ["reaction", self._course(f)], ["authors", self._view(f)]]
            return (f"LITERATURE SCREENING LOG -- week of {fmt_date(f['screened'], rng)} ({f['screener']})\n{cite}\n\n"
                    + table(["field", "entry"], rows, rng) + f"\n\nReference: {markets}. CCDS terms: {self._ccds(f)}."
                    + f"\n\n{f['sop']}:\n" + "\n".join(rules))
        para = [f"Hit {f['hit_id']}: {cite}.", f"Type: {self.KINDS[f['kind']]}.",
                f"Patients: {self._patient_text(f)}.", f"Product: {self._product_text(f)}.",
                f"Reaction: {self._course(f)}.", f"{_cap(self._view(f))}."]
        return ("REVIEWER WORKSHEET (literature)\n\n" + noisy(join_sentences(para), rng, 0.08)
                + f"\n\nFor reference: {markets}. CCDS listed terms: {self._ccds(f)}.\n\n{f['sop']}: " + " ".join(rules))

    def questions(self):
        return {
            "literature_case_outcome": choice("What should the screener do with this article?", {
                "create_icsr_now": "Create an individual case for the company now.",
                "log_no_icsr": "Log the article for periodic and signal review without creating an individual case.",
                "not_company_product": "Record that the article concerns a product that is not the company's.",
                "retrieve_full_text_first": "Obtain the full text before deciding, because a key element is unclear "
                                            "in the abstract.",
            }),
            "product_attribution": score("How confidently can the product in the article be attributed to the company?", [
                "Not at all: another manufacturer's product, or a country the company does not supply.",
                "Weakly: the substance alone, where other suppliers exist or the country is unknown.",
                "Fairly: the substance alone, in a country where the company is the only supplier.",
                "Firmly: the company's own brand is named.",
            ]),
            "author_causality_support": score("How strongly do the authors themselves link the reaction to the product?", [
                "Not at all: they attribute it to something else.",
                "Barely: they note the timing without taking a view.",
                "Partly: they say a role for the product cannot be excluded.",
                "Strongly: they conclude it was probably the product, with a supporting stop or re-exposure response.",
            ]),
            "identifiable_patient_present": noul(
                "The article, as screened, describes at least one identifiable patient.",
                yes="A specific patient is described by age or age group, sex, initials or individual details.",
                no="Only pooled results or an unspecified 'patient' with no identifying detail are given."),
            "serious_unlisted_reaction": noul(
                "The article describes a serious reaction that the company's core data sheet does not list.",
                yes="The reaction meets a seriousness criterion and the CCDS lacks it at that severity.",
                no="The reaction is non-serious, or the CCDS already lists it at that severity."),
        }

    def answer_key(self, f):
        """Rule-of-thumb answer key for the spread check only; never used as a label."""
        n = f["naming"]
        if n in ("inn_other_mah", "inn_country_not_ours"):
            out = "not_company_product"
        elif f["full_text"] == "abstract_only" and (f["identifiable"] == "unclear" or n == "inn_no_country"):
            out = "retrieve_full_text_first"
        elif f["identifiable"] != "yes":
            out = "log_no_icsr"
        else:
            out = "create_icsr_now"
        if n == "brand":
            pa = 3
        elif n == "inn_country_ours":
            pa = 1 if f["country"] in f["multi_mah_countries"] else 2
        elif n == "inn_no_country":
            pa = 1
        else:
            pa = 0
        au = {"other": 0, "no_view": 1, "possible": 2, "probable": 3}[f["author_view"]]
        su = f["serious"] and f["ccds"] != "listed"
        return {"literature_case_outcome": out, "product_attribution": pa, "author_causality_support": au,
                "identifiable_patient_present": "yes" if f["identifiable"] == "yes" else "no",
                "serious_unlisted_reaction": "yes" if su else "no"}


def pick_course(f, e):
    """A serious hospital course for a literature case, chosen from the case's own factors (no extra rng draw)."""
    opts = [f"{e['term']} requiring hospital admission", f"{e['term']}, admitted for {f['vol'] % 9 + 2} days",
            f"{e['term']} with a fatal outcome" if e["fatal"] else f"{e['term']} needing intensive care",
            f"{e['term']}; the patient was hospitalised and recovered"]
    return opts[f["page"] % len(opts)]


# =========================================================================== schema 4
class MedicationErrorReport(Schema):
    name = "medication_error_report"
    description = ("A report of a medication error or near miss involving a company product, categorised and "
                   "assessed for harm, recurrence and product-design contribution.")
    surfaces = {"json": 0.5, "incident_form": 0.25, "safety_officer_note": 0.25}

    DEFS = [
        ["Medication error: an unintended failure in prescribing, dispensing, preparing or giving a medicine. It is "
         "'intercepted' only if caught before the product reached the patient (handed over, or taken home); once it "
         "has reached the patient it is an error with or without harm, depending on the outcome.",
         "Category rules: an error caught before the medicine is handed over or given counts as intercepted; any "
         "error that reaches the patient -- including a pack taken home but never opened -- is classed by outcome "
         "as with or without harm."],
        ["Deliberate use outside the label by a prescriber, and intentional overdose or misuse by a patient, are "
         "NOT medication errors; they are recorded under their own categories.",
         "Intentional acts are excluded from the error categories: a prescriber's deliberate off-label choice and a "
         "patient's intentional overdose or misuse are logged separately."],
        ["Product design (name, carton, label, device) is recorded as contributing when the investigation finds it "
         "made the error more likely; staffing, handwriting and interruptions are system factors, not design.",
         "Design contribution means the name, packaging, labelling or device itself made the mistake easier; "
         "workload, legibility and distraction are recorded as system causes."],
    ]
    DESIGN_NOTES = {
        "lasa_name": "the two names look alike on the dispensing screen and sit next to each other on the shelf",
        "strength_packaging": "both strengths use the same carton colour, with the strength in small print",
        "pen_device": "the dose window is small and the pen gives no click when the dial passes zero",
        "decimal_dose": "the label states the strength per 5 mL in large print and per mL in small print",
        "wrong_route": "the oral syringe in the pack fits a standard IV connector",
    }
    SYSTEM_NOTES = ["the prescription was handwritten and hard to read", "the nurse was interrupted twice while "
                    "preparing the dose", "a single-checked night shift with two staff off sick",
                    "two patients with the same surname on the ward", "the pharmacy computer was down and labels "
                    "were written by hand", "an agency locum unfamiliar with the ward's process"]

    def sample_factors(self, rng):
        reg = regime(rng)
        prod = _product(rng)
        err = pick_w(rng, {"decimal_dose": 1.8, "lasa_name": 2.2, "strength_packaging": 2, "pen_device": 1.4,
                           "wrong_route": 1, "wrong_patient": 1.3, "intentional_offlabel": 1.1,
                           "intentional_overdose": 0.9})
        if err == "pen_device" and prod["form"] != "prefilled pen":
            err = "strength_packaging"
        if err == "wrong_route" and "solution" not in prod["form"]:
            err = "lasa_name"                           # an oral liquid drawn up and given IV needs a liquid
        stage = pick_w(rng, {"intercepted": 3, "dispensed_not_taken": 1.6, "administered": 5})
        if err.startswith("intentional"):
            stage = "administered"
        if err == "wrong_patient" and stage == "dispensed_not_taken":
            stage = "administered"
        harm = pick_w(rng, {"none": 3, "minor": 2.5, "moderate": 2, "severe": 1.2}) if stage == "administered" else "none"
        design = err in self.DESIGN_NOTES and chance(rng, {"lasa_name": 0.85, "strength_packaging": 0.8,
                                                           "pen_device": 0.75, "decimal_dose": 0.4,
                                                           "wrong_route": 0.45}[err])
        similar = pick(rng, [0, 0, 1, 2, 3, 5, 8, 12])
        if reg == "clear":
            if chance(rng, 0.5):
                stage, harm = "intercepted", "none"
            else:
                stage, harm = "administered", pick(rng, ["moderate", "severe"])
        elif reg == "borderline":
            # the hard edges: a pack taken home but never opened, harm that may be the illness, a handful of repeats
            if not err.startswith("intentional") and err != "wrong_patient" and chance(rng, 0.45):
                stage, harm = "dispensed_not_taken", "none"
            elif stage == "administered":
                harm = pick(rng, ["none", "minor"])
            similar = pick(rng, [2, 3, 4, 5, 6])
        else:
            # a design-looking error that the investigation pins on the system, or the reverse; or an intentional act
            if chance(rng, 0.35):
                err, stage = pick(rng, ["intentional_offlabel", "intentional_overdose"]), "administered"
                harm = pick(rng, ["none", "minor", "moderate", "severe"])
            elif err in self.DESIGN_NOTES:
                design = not design
        if err.startswith("intentional"):
            design = False
        if stage != "administered":
            harm = "none"
        return {"regime": reg, "product": prod, "error_type": err, "stage": stage, "harm": harm,
                "design_factor": design, "similar_12m": similar,
                "system_note": pick(rng, self.SYSTEM_NOTES), "lasa_brand": _lasa(rng, prod["brand"]),
                "lasa_class": pick(rng, [c["name"] for c in CLASSES if c["key"] != prod["cls"]]),
                "setting": pick(rng, {"wrong_patient": ["hospital ward", "care home", "day-surgery unit"],
                                      "intentional_overdose": ["emergency department", "patient's home"],
                                      "wrong_route": ["hospital ward", "intensive care unit", "care home"]}.get(
                    err, ["hospital ward", "community pharmacy", "care home", "patient's home", "day-surgery unit",
                          "outpatient clinic"])),
                "reporter": pick(rng, {"intentional_overdose": ["emergency physician", "poison-centre pharmacist"],
                                       "wrong_patient": ["ward nurse", "medication-safety officer", "care-home manager"]}.get(
                    err, ["hospital pharmacist", "ward nurse", "community pharmacist", "medication-safety officer",
                          "care-home manager", "patient's daughter"])),
                "checker": pick(rng, ["second pharmacist", "checking nurse", "pharmacy technician", "barcode scanner"]),
                "report_id": ident(rng, "ME", 6, "-"), "date": rand_date(rng), "n_tabs": rng.randint(12, 40),
                "sop": _sop_no(rng, "PV-MED"), "officer": person(rng)}

    def _what(self, f):
        p = f["product"]
        st = p["strength"]
        return {
            "decimal_dose": (f"{p['brand']} {st}: 5 mL was drawn up instead of 0.5 mL after a decimal point was "
                             f"misread" if "solution" in p["form"] else
                             f"the {p['brand']} pen was set to ten times the prescribed dose after a decimal point was "
                             f"misread" if "pen" in p["form"] or "syringe" in p["form"] else
                             f"a prescription for half a {p['brand']} {st} tablet daily was transcribed as five "
                             f"tablets daily"),
            "lasa_name": f"{p['brand']} was supplied instead of {f['lasa_brand']} (another company's "
                         f"{f['lasa_class']})",
            "strength_packaging": f"the wrong strength of {p['brand']} was picked; the prescription was for a "
                                  f"different strength than {st}",
            "pen_device": f"a {p['brand']} pen was used without priming and with the dial misread, so the dose given "
                          f"was uncertain",
            "wrong_route": f"{p['brand']} intended for oral use was drawn up and given through an intravenous line",
            "wrong_patient": f"{p['brand']} {st} was given to the wrong patient",
            "intentional_offlabel": f"the prescriber deliberately prescribed twice the labelled dose of {p['brand']}, "
                                    f"documented as an off-label choice",
            "intentional_overdose": self._overdose_text(f, p),
        }[f["error_type"]]

    @staticmethod
    def _overdose_text(f, p):
        n, form = f["n_tabs"], p["form"]
        if "capsule" in form or "tablet" in form:
            what = f"took {n} {p['brand']} {'capsules' if 'capsule' in form else 'tablets'} at once"
        elif "solution" in form:
            what = f"drank about {n * 10} mL of {p['brand']} oral solution at once"
        else:
            what = (f"injected the contents of {max(2, n // 4)} {p['brand']} "
                    f"{'pens' if 'pen' in form else 'syringes'} at once")
        return f"the patient intentionally {what} (self-harm)"

    def _stage(self, f):
        return {"intercepted": f"caught by the {f['checker']} before it was handed over or given",
                "dispensed_not_taken": "handed to the patient, who noticed the unfamiliar pack at home and took none",
                "administered": "given to / taken by the patient"}[f["stage"]]

    def _harm(self, f):
        return {"none": "no symptoms; routine observation only",
                "minor": "brief, mild symptoms that settled without treatment",
                "moderate": "symptoms needing treatment and extra monitoring for a day",
                "severe": "admitted to hospital for intensive monitoring and treatment"}[f["harm"]]

    def _factors(self, f):
        if f["error_type"] == "intentional_offlabel":
            return ["the prescriber's note gives a documented clinical rationale for the higher dose"]
        if f["error_type"] == "intentional_overdose":
            return ["the patient is known to the mental-health team; packs were stored at home"]
        out = []
        if f["design_factor"]:
            out.append(self.DESIGN_NOTES[f["error_type"]])
        out.append(f["system_note"])
        return out

    def render(self, f, rng, surface):
        p = f["product"]
        rules = [pick(rng, r) for r in self.DEFS]
        sim = f"{f['similar_12m']} similar report{'s' if f['similar_12m'] != 1 else ''} for {p['brand']} in the " \
              f"safety database in the last 12 months"
        if surface == "json":
            return dumps(prune({
                "report": {"id": f["report_id"], "date": f["date"].isoformat(), "reporter": f["reporter"],
                           "setting": f["setting"]},
                "product": {"name": f"{p['brand']} ({p['inn']}) {p['strength']}", "form": p["form"],
                            "class": p["class_name"]},
                "incident": {"what_happened": self._what(f), "how_far_it_got": self._stage(f),
                             "patient_outcome": self._harm(f)},
                "investigation": {"contributing_factors": self._factors(f), "similar_reports": sim},
                "sop": {"id": f["sop"], "rules": rules},
            }))
        if surface == "incident_form":
            rows = [("Report", f["report_id"]), ("Date", fmt_date(f["date"], rng)), ("Reporter", f["reporter"]),
                    ("Setting", f["setting"]), ("Product", f"{p['brand']} ({p['inn']}) {p['strength']}, {p['form']}"),
                    ("What happened", self._what(f)), ("Reached patient?", self._stage(f)),
                    ("Outcome", self._harm(f)), ("Contributing factors", "; ".join(self._factors(f))),
                    ("Database", sim)]
            return ("MEDICATION ERROR REPORT\n" + kv_lines(rows, rng) + f"\n\nCoding rules ({f['sop']}):\n"
                    + "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1)))
        para = [f"Report {f['report_id']} from a {f['reporter']} ({f['setting']}, {fmt_date(f['date'], rng)}): "
                f"{self._what(f)}.", f"It was {self._stage(f)}.", f"Outcome: {self._harm(f)}.",
                "Root-cause notes: " + "; ".join(self._factors(f)) + ".", f"{_cap(sim)}."]
        return (f"NOTE FOR {f['officer'].upper()} (medication-safety review)\n\n" + noisy(join_sentences(para), rng, 0.1)
                + f"\n\n{f['sop']}: " + " ".join(rules))

    def questions(self):
        return {
            "error_case_category": choice("How should this report be categorised?", {
                "error_with_harm": "A medication error that reached the patient and caused harm.",
                "error_without_harm": "A medication error that reached the patient without causing harm.",
                "intercepted_potential_error": "A potential error caught before the product reached the patient.",
                "not_a_medication_error": "Not a medication error: a deliberate off-label choice or an intentional "
                                          "overdose or misuse.",
            }),
            "harm_level": score("How much harm did the patient come to?", [
                "None.",
                "Minor: brief symptoms with no treatment needed.",
                "Moderate: treatment or extra monitoring was needed.",
                "Severe: hospital admission, lasting harm or death.",
            ]),
            "recurrence_risk": score("How likely is the same error to happen again elsewhere?", [
                "Remote: a one-off slip with no design link and no similar reports.",
                "Low: a few similar reports, but no design link.",
                "Moderate: either a design link or a run of similar reports.",
                "High: a design link and a run of similar reports.",
            ]),
            "product_design_contributed": noul(
                "The product's name, packaging, labelling or device contributed to this error.",
                yes="The investigation found that a feature of the product itself made the error more likely.",
                no="The causes found were human or system factors, or the event was not an error at all."),
            "wrong_product_administered": noul(
                "The patient actually received the wrong product, dose or route."),
        }

    def answer_key(self, f):
        """Rule-of-thumb answer key for the spread check only; never used as a label."""
        err = f["error_type"]
        if err.startswith("intentional"):
            cat = "not_a_medication_error"
        elif f["stage"] == "intercepted":
            cat = "intercepted_potential_error"
        elif f["harm"] == "none":
            cat = "error_without_harm"
        else:
            cat = "error_with_harm"
        s = f["similar_12m"]
        rr = (3 if f["design_factor"] and s >= 5 else 2 if f["design_factor"] or s >= 5 else 1 if s >= 1 else 0)
        if err.startswith("intentional"):
            rr = 0 if s == 0 else 1
        administered = f["stage"] == "administered" and err != "intentional_offlabel"
        return {"error_case_category": cat,
                "harm_level": {"none": 0, "minor": 1, "moderate": 2, "severe": 3}[f["harm"]],
                "recurrence_risk": rr, "product_design_contributed": "yes" if f["design_factor"] else "no",
                "wrong_product_administered": "yes" if administered else "no"}


def _lasa(rng, brand):
    """A look-alike name: one letter changed or a short suffix added."""
    if chance(rng, 0.5):
        return brand + pick(rng, ["n", "x", "l", "in"])
    i = rng.randrange(1, len(brand))
    c = pick(rng, [ch for ch in "aeiourlnt" if ch != brand[i].lower()])
    return brand[:i] + c + brand[i + 1:]


# =========================================================================== schema 5
class SignalCaseSeriesReview(Schema):
    name = "signal_case_series_review"
    description = ("An aggregate review of all cases of one event with one product, deciding whether a potential "
                   "safety signal is closed, monitored, followed up or validated.")
    surfaces = {"json": 0.5, "signal_assessment_form": 0.25, "safety_committee_memo": 0.25}

    RULES = [
        ["Statistical signal: lower bound of the 95% CI of the reporting odds ratio at 2.0 or more, with at least "
         "3 cases.",
         "Disproportionality counts only when the ROR's lower 95% bound reaches 2.0 on 3 or more cases."],
        ["Close when the event is already in the CCDS at this severity, or when the cases are explained by the "
         "treated disease and nothing stands out statistically.",
         "Closure applies to events labelled at the observed severity, and to series explained by the underlying "
         "disease with no statistical signal."],
        ["Validate for full evaluation when a statistical signal is backed by informative cases (well documented, "
         "with positive dechallenge or rechallenge), or when strongly informative cases with a plausible mechanism "
         "stand without one. Validate AND notify the authority within 3 working days as an emerging issue when a "
         "validated signal includes fatal cases in an unlisted event with a plausible mechanism.",
         "Full evaluation follows when the statistics are backed by informative cases, or when highly informative "
         "cases and a plausible mechanism exist without the statistics. If such a signal is unlisted, plausible and "
         "includes deaths, also notify the authority as an emerging issue within 3 working days."],
        ["Request targeted follow-up when the cases are too poorly documented to judge; otherwise continue "
         "routine monitoring.",
         "Poorly documented series: send targeted follow-up questionnaires first. Anything else: routine "
         "monitoring."],
    ]

    @staticmethod
    def _period(rng):
        """A review window that runs forward: 4-8 quarters ending no later than 2026-Q2."""
        end = rng.randint(2024 * 4, 2026 * 4 + 1)
        start = end - rng.randint(3, 7)
        return f"{start // 4}-Q{start % 4 + 1} to {end // 4}-Q{end % 4 + 1}"

    def sample_factors(self, rng):
        reg = regime(rng)
        prod = _product(rng)
        c = CLS[prod["cls"]]
        serious_ok = [x for x in EVENTS if not x["minor"] and _fits(prod, x)]
        known = [x for x in serious_ok if x["key"] in c["known"]]
        common = [x for x in serious_ok if x["key"] in c["common"]]
        other = [x for x in serious_ok if x["key"] not in c["known"] + c["common"]]
        want = pick_w(rng, {"known": 0.4, "common": 0.32, "other": 0.28})
        if reg == "clear" and chance(rng, 0.5):
            want = "known"
        elif reg in ("clear", "conflicting") and chance(rng, 0.4):
            want = "common"
        e = pick(rng, {"known": known, "common": common, "other": other}[want] or other)
        n = pick(rng, [3, 4, 5, 6, 8, 11, 14, 19, 27, 38])
        lb = round(pick_w(rng, {rng.uniform(0.5, 0.99): 2, rng.uniform(1.0, 1.95): 2.5, rng.uniform(2.0, 3.9): 2.5,
                                rng.uniform(4.0, 9.5): 1.5}), 2)
        quality = pick_w(rng, {"poor": 2, "thin": 1.6, "mixed": 2.4, "good": 2.5})
        ccds = pick_w(rng, {"unlisted": 5, "listed_same": 1.4, "listed_milder": 1.4})
        if ccds == "listed_milder" and not e["milder"]:
            ccds = "unlisted"
        fatal = pick(rng, [0, 0, 1, 1, 2]) if e["fatal"] else 0
        if reg == "clear":
            if chance(rng, 0.5):
                lb, quality, ccds = round(rng.uniform(3.0, 8.0), 2), "good", "unlisted"
                fatal_known = [x for x in known if x["fatal"]]
                if fatal_known and chance(rng, 0.6):
                    e = pick(rng, fatal_known)
                    fatal = pick(rng, [1, 2, 3])
            else:
                lb, quality = round(rng.uniform(0.4, 1.3), 2), pick(rng, ["poor", "thin", "mixed"])
        elif reg == "borderline":
            lb = round(near(rng, 2.0, 0.35, lo=1.3), 2)
            n = pick(rng, [3, 4, 5, 6])
        else:
            # strong numbers on thin cases, or good cases with no numbers; deaths in a disease that explains them
            if chance(rng, 0.5):
                lb, quality = round(rng.uniform(3.0, 9.0), 2), "poor"
            else:
                lb, quality = round(rng.uniform(0.7, 1.8), 2), "good"
        plausible = e["key"] in c["known"]
        confounded = e["key"] in c["common"]
        if ccds == "listed_milder" and not e["milder"]:
            ccds = "unlisted"
        fatal = fatal if e["fatal"] else 0
        # case composition consistent with the quality band
        if quality == "poor":
            insuff = max(1, round(n * rng.uniform(0.6, 0.85)))
            pos = min(n - insuff, rng.randint(0, 1))
        elif quality == "thin":
            insuff = round(n * rng.uniform(0.15, 0.45))
            pos = round(n * rng.uniform(0.0, 0.15))
        elif quality == "mixed":
            insuff = round(n * rng.uniform(0.2, 0.45))
            pos = max(1, round(n * rng.uniform(0.2, 0.38)))
        else:
            insuff = round(n * rng.uniform(0.0, 0.25))
            pos = max(2, round(n * rng.uniform(0.42, 0.7)))
        pos = min(pos, n - insuff)
        rechal = min(pos, pick(rng, [0, 0, 1, 2]) if quality == "good" else 0)
        conf_n = min(n - pos, round(n * rng.uniform(0.35, 0.7)) if confounded else rng.randint(0, max(0, n // 5)))
        fatal = min(fatal, n)
        return {"regime": reg, "product": prod, "event": e["key"], "n_cases": n, "ror_lower": lb,
                "quality_band": quality, "n_insufficient": insuff, "n_pos_dechallenge": pos, "n_rechallenge": rechal,
                "n_confounded": conf_n, "n_fatal": fatal, "ccds": ccds, "plausible": plausible,
                "confounded": confounded, "ror": round(lb * rng.uniform(1.6, 2.6), 2),
                "bg_rate_high": round(rng.uniform(4, 40), 1), "bg_rate_low": round(rng.uniform(0.5, 9), 1),
                "prior_review": pick(rng, [None, None, f"closed {rng.randint(8, 30)} months ago on "
                                                       f"{max(2, n // 3)} cases"]),
                "period": self._period(rng),
                "signal_id": ident(rng, "SIG", 5, "-"), "assessor": person(rng), "sop": _sop_no(rng, "PV-SIG"),
                "countries": rng.sample(COUNTRIES, 3), "lit": chance(rng, 0.3)}

    def _quality_q(self, f):
        n = f["n_cases"]
        strong = f["n_pos_dechallenge"] + f["n_rechallenge"]
        if f["n_insufficient"] / n >= 0.6:
            return 0
        if strong / n >= 0.4 and f["n_confounded"] / n <= 0.35:
            return 3
        if strong / n >= 0.2:
            return 2
        return 1

    def _mechanism(self, f):
        e, p = EV[f["event"]], f["product"]
        if f["plausible"]:
            return (f"{e['term']} is a recognised effect of the {p['class_name']} class, consistent with its "
                    f"pharmacology")
        return f"no known mechanism links the {p['class_name']} class with {e['term']}"

    def _background(self, f):
        e, p = EV[f["event"]], f["product"]
        if f["confounded"]:
            return (f"{e['term']} is common in people treated for {p['indication']} (about {f['bg_rate_high']} per "
                    f"1,000 patient-years)")
        return f"{e['term']} is rare in people treated for {p['indication']} (about {f['bg_rate_low']} per 100,000)"

    def _ccds(self, f):
        e = EV[f["event"]]
        return {"unlisted": f"{e['term']} not listed", "listed_same": f"{e['term']} listed (uncommon)",
                "listed_milder": f"only {e['milder']} listed; {e['term']} itself is not"}[f["ccds"]]

    def render(self, f, rng, surface):
        p, e = f["product"], EV[f["event"]]
        rules = [pick(rng, r) for r in self.RULES]
        counts = {"cases": f["n_cases"], "positive_dechallenge": f["n_pos_dechallenge"],
                  "positive_rechallenge": f["n_rechallenge"], "insufficient_information": f["n_insufficient"],
                  "confounded_by_other_cause": f["n_confounded"], "fatal": f["n_fatal"]}
        stats = f"ROR {f['ror']} (95% CI lower bound {f['ror_lower']:.2f})"
        if surface == "json":
            return dumps(prune({
                "signal": {"id": f["signal_id"], "product": f"{p['brand']} ({p['inn']})", "class": p["class_name"],
                           "event": e["term"], "period": f["period"], "assessor": f["assessor"],
                           "prior_review": f["prior_review"]},
                "case_series": counts,
                "statistics": stats,
                "context": {"ccds": self._ccds(f), "mechanism": self._mechanism(f), "background": self._background(f),
                            "literature": "two published case reports describe the same pairing" if f["lit"] else None},
                "sop": {"id": f["sop"], "rules": rules},
            }))
        if surface == "signal_assessment_form":
            rows = [[k.replace("_", " "), v] for k, v in counts.items()]
            head = kv_lines([("Signal", f["signal_id"]), ("Product", f"{p['brand']} ({p['inn']}), {p['class_name']}"),
                             ("Event", e["term"]), ("Period", f["period"]), ("Disproportionality", stats),
                             ("CCDS", self._ccds(f)), ("Prior review", f["prior_review"])], rng)
            return ("SIGNAL ASSESSMENT FORM\n" + head + "\n\n" + table(["case series", "n"], rows, rng)
                    + f"\n\nMechanism: {self._mechanism(f)}.\nBackground: {self._background(f)}."
                    + ("\nLiterature: two published case reports describe the same pairing." if f["lit"] else "")
                    + f"\n\n{f['sop']}:\n" + "\n".join(f"- {r}" for r in rules))
        para = [f"Signal {f['signal_id']}: {e['term']} with {p['brand']} ({p['inn']}), {f['period']}.",
                f"{f['n_cases']} cases: {f['n_pos_dechallenge']} with positive dechallenge, {f['n_rechallenge']} with "
                f"positive rechallenge, {f['n_insufficient']} with too little information, {f['n_confounded']} with "
                f"another plausible cause, {f['n_fatal']} fatal.", f"{stats}.", f"CCDS: {self._ccds(f)}.",
                f"{_cap(self._mechanism(f))}.", f"{_cap(self._background(f))}."]
        if f["prior_review"]:
            para.append(f"A previous review was {f['prior_review']}.")
        return (f"MEMO to the safety committee from {f['assessor']}\n\n" + noisy(join_sentences(para), rng, 0.08)
                + f"\n\n{f['sop']}: " + " ".join(rules))

    def questions(self):
        return {
            "signal_decision": choice("What should the safety committee decide about this potential signal?", {
                "close_no_signal": "Close it: nothing new beyond what is labelled or explained by the treated disease.",
                "continue_routine_monitoring": "Keep it under routine monitoring without further action now.",
                "request_targeted_follow_up": "Send targeted follow-up on the cases before deciding.",
                "validate_and_notify_authority": "Validate it and notify the authority now as an emerging safety issue.",
                "validate_for_full_evaluation": "Validate it and open a full signal evaluation.",
            }),
            "disproportionality_strength": score("How strong is the disproportionality statistic?", [
                "Absent: the lower bound is below 1.",
                "Weak: the lower bound is between 1 and the threshold of 2.",
                "Clear: the lower bound is at or above 2 but below 4.",
                "Strong: the lower bound is 4 or more.",
            ]),
            "case_series_quality": score("How informative are the individual cases taken together?", [
                "Uninformative: most cases lack the information to judge them.",
                "Thin: few cases show a positive stop or re-exposure response.",
                "Useful: a fair share of cases show a positive stop or re-exposure response.",
                "Compelling: many well-documented cases with positive responses and few other causes.",
            ]),
            "biological_plausibility": noul(
                "A plausible biological mechanism links the product to the event.",
                yes="The product's class or pharmacology is known to produce this kind of event.",
                no="No known mechanism or class effect connects the product with the event."),
            "confounding_by_indication_likely": noul(
                "The condition being treated could itself account for many of these cases."),
        }

    def answer_key(self, f):
        """Rule-of-thumb answer key for the spread check only; never used as a label."""
        q = self._quality_q(f)
        stat = f["ror_lower"] >= 2.0 and f["n_cases"] >= 3
        if f["ccds"] == "listed_same":
            d = "close_no_signal"
        elif f["confounded"] and not stat and q <= 1:
            d = "close_no_signal"
        elif stat and q >= 2 and f["n_fatal"] >= 1 and f["plausible"]:
            d = "validate_and_notify_authority"
        elif (stat and q >= 2) or (q == 3 and f["plausible"]):
            d = "validate_for_full_evaluation"
        elif q == 0:
            d = "request_targeted_follow_up"
        else:
            d = "continue_routine_monitoring"
        lb = f["ror_lower"]
        ds = 0 if lb < 1 else 1 if lb < 2 else 2 if lb < 4 else 3
        return {"signal_decision": d, "disproportionality_strength": ds, "case_series_quality": q,
                "biological_plausibility": "yes" if f["plausible"] else "no",
                "confounding_by_indication_likely": "yes" if f["confounded"] else "no"}


WORKFLOW = Workflow(
    name="pharmacovigilance_case_causality",
    title="Post-marketing pharmacovigilance case assessment",
    description="How a marketing-authorisation holder's drug-safety team assesses post-marketing reports: "
                "seriousness, labelling, causality, validity, literature cases, medication errors and signals.",
    domain="life sciences / drug safety",
    schemas=[PostmarketingIcsrCausality(), ConsumerReportValidity(), LiteratureCaseScreening(),
             MedicationErrorReport(), SignalCaseSeriesReview()],
    notes="Fictional company, products and labelling; WHO-UMC-style categories written as the company's own SOP, "
          "not regulatory guidance. Post-marketing only (no clinical-trial reporting); consumer reports are intake "
          "records, never conversations; no reply questions, no security alerts, no bills, no AI-agent runs.",
)
