"""Nursing fall-risk intervention: an inpatient's fall-risk band and prevention bundle from nursing assessment notes,
against a hospital's written fall-prevention protocol, plus reassessment, post-fall care, enhanced monitoring and
bundle audits.

Everything here is fictional: the hospitals, wards, staff and patients, and the "Ward Fall Risk Index" (WFRI) with its
items, weights, bands and bundles. The index is invented for this generator; it does not reproduce the Morse Fall
Scale, Hendrich II, STRATIFY or any other real instrument, and none of it is clinical guidance. Item weights and band
cut-offs differ between the fictional hospitals' protocol versions.

Age is a scored item of the index, as in real fall-risk tools. Social context enters the way an equity-aware protocol
writes it: fall-prevention teaching goes through a qualified interpreter (or to the carer of a confused patient),
video-monitoring redirection only works in the languages the monitoring staff speak, family presence counts when a
relative can stay, and the family is told about a fall in its own language. None of it lowers the standard of care.

Each schema draws a `regime` first ('clear' / 'borderline' / 'conflicting'): borderline cases land a point from a band
cut-off, an orthostatic drop at the 20 mmHg line, a fall six months ago against a six-month item; conflicting cases
carry a patient who insists they need no help, a stale score-sheet total, or a request that the protocol does not
support. Every schema defines `intended(f) -> {qid: answer}` (choice key, "yes"/"no", or score level index), the
protocol applied to the factors; it is used only to check that answers spread and is never rendered.
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
    join_sentences,
    kv_lines,
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
    table,
)

HOSP_STEMS = ["Ambergate", "Bellhaven", "Corrigan Bay", "Dovecote", "Elmshaw", "Foxwarren", "Glenbrook", "Hartwell",
              "Inchmore", "Jessamine", "Kingsmere", "Lowther Vale", "Millbeck", "Nettlefield", "Oakhurst Rise",
              "Penrose", "Quayside", "Ravenstone", "Sedgefield", "Tolland", "Upton Cross", "Verity Lake"]
HOSP_KINDS = ["Hospital", "General Hospital", "Medical Center", "Community Hospital", "Infirmary"]
WARDS = ["Ward 7B Acute Medicine", "Ward 4 Orthopaedics", "Ward 12 Care of Older People", "Ward 9 Respiratory",
         "Ward 3 General Surgery", "Ward 6A Cardiology", "Ward 10 Stroke Unit", "Ward 5 Renal", "Ward 11 Oncology",
         "Ward 8 Neurology", "Ward 2 Short Stay", "Ward 14 Gastroenterology"]
ADMISSIONS = ["community-acquired pneumonia", "cellulitis of the leg", "hip fracture, day 1 after fixation",
              "knee replacement, day 1", "heart failure exacerbation", "COPD exacerbation", "urinary infection with "
              "confusion", "stroke, now on the rehab pathway", "upper GI bleed", "syncope work-up",
              "diabetic foot infection", "acute kidney injury", "alcohol withdrawal", "acute pancreatitis",
              "step-down from ICU after sepsis", "neutropenic fever on chemotherapy", "Parkinson's medication review",
              "rib fractures after a fall at home", "small bowel obstruction", "day 2 after bowel resection",
              "low sodium", "dehydration with diarrhoea", "asthma attack", "pulmonary embolism", "lumbar spine surgery, "
              "day 1", "acute cholecystitis", "anaemia needing transfusion", "decompensated cirrhosis",
              "first seizure", "atrial fibrillation with fast rate", "recurrent hypoglycaemia"]
RISK_MEDS = {"opioid": ["oxycodone", "morphine", "hydromorphone PCA", "tramadol"],
             "sedative": ["zopiclone", "lorazepam", "temazepam"],
             "antipsychotic": ["haloperidol", "quetiapine"],
             "bp_change": ["amlodipine (started this admission)", "lisinopril (dose increased)"],
             "diuretic": ["furosemide", "bumetanide"],
             "anticonvulsant": ["levetiracetam", "gabapentin"]}
SAFE_MEDS = ["paracetamol", "omeprazole", "atorvastatin", "levothyroxine", "insulin glargine", "metformin",
             "salbutamol inhaler", "vitamin D", "senna", "co-amoxiclav", "cefalexin", "enoxaparin prophylaxis"]
ANTICOAG = ["apixaban", "warfarin", "rivaroxaban", "treatment-dose enoxaparin"]
TETHERS = ["IV pump", "urinary catheter", "oxygen tubing", "telemetry pack", "wound drain", "PCA pump"]
LANGUAGES = ["Spanish", "Vietnamese", "Arabic", "Polish", "Haitian Creole", "Somali", "Mandarin", "Tagalog",
             "Portuguese", "Russian", "Korean", "Punjabi", "Ukrainian", "Cantonese", "Bengali", "Farsi"]
RACE_ETH = ["Asian", "Black or African American", "Hispanic or Latino", "White", "American Indian or Alaska Native",
            "Native Hawaiian or Pacific Islander", "Middle Eastern or North African", "Multiracial",
            "declined to state"]


def _hospital(rng):
    return f"{pick(rng, HOSP_STEMS)} {pick(rng, HOSP_KINDS)}"


def _hm(t):
    return t.strftime("%H:%M")


def _cap(text):
    return text[:1].upper() + text[1:] if text else text


def _nurse(rng):
    return person(rng) + pick(rng, [", RN", " RN", ""])


# =========================================================================== the fictional index
def _tool(rng):
    return {"ver": f"WFRI-{rng.randint(2, 6)}.{rng.randint(0, 9)}", "w_fall": pick(rng, [4, 5]),
            "age_lo": pick(rng, [65, 70]), "age_hi": pick(rng, [80, 85]), "mod": pick(rng, [5, 6]),
            "high": pick(rng, [11, 12]), "ortho": 20}


def _items(rng, reg, tool, overrides=None):
    """Assessment items for one patient. Returns a dict the index scores."""
    it = {
        "age": pick_w(rng, {rng.randint(22, 59): 2, rng.randint(60, 79): 3, rng.randint(80, 97): 2.5}),
        "fall_months_ago": pick_w(rng, {None: 5, rng.randint(1, 5): 2.5, rng.randint(7, 24): 1.5}),
        "mobility": pick_w(rng, {"independent": 4, "walking_aid": 3, "one_assist": 2, "bed_bound": 0.8}),
        "cognition": pick_w(rng, {"oriented": 6, "forgetful_impulsive": 2.2, "delirium": 1.4}),
        "toileting": chance(rng, 0.4),
        "meds": [],
        "tethers": rng.sample(TETHERS, pick_w(rng, {0: 3, 1: 3, 2: 2, 3: 1})),
        "ortho_drop": pick_w(rng, {rng.randint(0, 12): 5, rng.randint(13, 19): 1.5, rng.randint(20, 38): 2,
                                   None: 1.2}),
        "sensory": pick_w(rng, {None: 6, "poor vision (macular degeneration)": 1, "hearing aids, not wearing them": 1,
                                "cataracts, glasses at home": 0.8}),
        "injury_flag": pick_w(rng, {None: 6, "on " + pick(rng, ANTICOAG): 2, "known osteoporosis": 1,
                                    "recent hip surgery": 0.8}),
    }
    for cls in rng.sample(list(RISK_MEDS), pick_w(rng, {0: 3, 1: 3, 2: 2, 3: 1, 4: 0.4})):
        it["meds"].append((cls, pick(rng, RISK_MEDS[cls])))
    if reg == "borderline":
        if chance(rng, 0.35):
            it["fall_months_ago"] = pick(rng, [5, 6, 7])
        if chance(rng, 0.35):
            it["ortho_drop"] = pick(rng, [18, 19, 20, 21])
        if chance(rng, 0.3):
            it["age"] = pick(rng, [tool["age_lo"] - 1, tool["age_lo"], tool["age_hi"] - 1, tool["age_hi"]])
    if overrides:
        it.update(overrides)
    if it["mobility"] == "bed_bound":
        it["ortho_drop"] = None              # no standing pressure in a patient who cannot stand
    if "urinary catheter" in it["tethers"]:
        it["toileting"] = False              # catheterised: no urgency or nocturia to score
    return it


def _score(tool, it):
    pts = tool["w_fall"] if (it["fall_months_ago"] is not None and it["fall_months_ago"] <= 6) else 0
    pts += 2 if it["age"] >= tool["age_hi"] else 1 if it["age"] >= tool["age_lo"] else 0
    pts += {"independent": 0, "walking_aid": 2, "one_assist": 3, "bed_bound": 1}[it["mobility"]]
    pts += {"oriented": 0, "forgetful_impulsive": 2, "delirium": 4}[it["cognition"]]
    pts += 2 if it["toileting"] else 0
    pts += min(3, len(it["meds"]))
    pts += min(2, len(it["tethers"]))
    pts += 2 if (it["ortho_drop"] is None or it["ortho_drop"] >= tool["ortho"]) else 0
    pts += 1 if it["sensory"] else 0
    return pts


def _band(tool, total, it=None):
    """0 low, 1 moderate, 2 high, 3 high with an injury-prone flag."""
    b = 2 if total >= tool["high"] else 1 if total >= tool["mod"] else 0
    if b == 2 and it is not None and it["injury_flag"]:
        return 3
    return b


def _steer(rng, reg, tool, overrides=None, target=None):
    """Draw items until the total lands where the regime wants it (near a cut-off when borderline)."""
    for _ in range(60):
        it = _items(rng, reg, tool, overrides)
        tot = _score(tool, it)
        if target is None:
            if reg == "borderline" and min(abs(tot - tool["mod"]), abs(tot - tool["high"])) > 1:
                continue
            return it
        if target(tot):
            return it
    return it


def _tool_text(tool, voice):
    """The index as each schema's documents quote it (wording differs by schema)."""
    t = tool
    if voice == "admission":
        return (f"Ward Fall Risk Index {t['ver']}: fall in the last 6 months {t['w_fall']}; age {t['age_lo']}-"
                f"{t['age_hi'] - 1} 1, {t['age_hi']}+ 2; mobility: walking aid 2, needs one person to stand or walk 3, "
                f"bed-bound 1; forgetful or impulsive 2, delirium screen positive 4; toileting urgency, frequency or "
                f"nocturia 2; each high-risk medicine class (opioid, sedative/hypnotic, antipsychotic, new or changed "
                f"BP medicine, diuretic, anticonvulsant) 1, max 3; each line or tube 1, max 2; orthostatic systolic "
                f"drop of {t['ortho']}+ mmHg (or not measured) 2; impaired vision or hearing 1. Bands: 0-{t['mod'] - 1} "
                f"low, {t['mod']}-{t['high'] - 1} moderate, {t['high']}+ high; high plus an anticoagulant, "
                f"osteoporosis or recent hip surgery = high, injury-prone.")
    return (f"{t['ver']} items | prior fall <=6 mo: {t['w_fall']} | age >= {t['age_lo']}: 1, >= {t['age_hi']}: 2 | "
            f"walker/cane 2, assist-of-one 3, bed-bound 1 | impulsive/forgetful 2, delirium 4 | toilet urgency 2 | "
            f"risk-med classes 1 each (max 3) | tubes/lines 1 each (max 2) | postural SBP fall >= {t['ortho']} or "
            f"unmeasured 2 | sight/hearing deficit 1 || low < {t['mod']}, moderate {t['mod']}-{t['high'] - 1}, high >= "
            f"{t['high']}")


def _mobility_txt(m, rng):
    return pick(rng, {"independent": ["walks independently", "mobile without aids"],
                      "walking_aid": ["walks with a frame", "uses a stick, steady with it", "mobilises with a walker"],
                      "one_assist": ["needs one person to stand and walk", "unsteady, needs a hand to transfer"],
                      "bed_bound": ["bed-bound, hoist for transfers", "nursed in bed, not weight-bearing"]}[m])


def _cog_txt(c, rng):
    return pick(rng, {"oriented": ["alert and oriented", "orientated, follows instructions"],
                      "forgetful_impulsive": ["forgetful, gets up without calling", "impulsive, forgets to use the call "
                                                                                  "bell"],
                      "delirium": ["delirium screen positive", "acutely confused, delirium screen positive"]}[c])


def _items_narrative(it, rng):
    s = [f"Age {it['age']}."]
    fm = it["fall_months_ago"]
    s.append(pick(rng, ["No falls reported in the past year.", "Denies previous falls."]) if fm is None else
             f"Fell at home about {fm} month{'s' if fm != 1 else ''} ago.")
    s.append(_cap(_mobility_txt(it["mobility"], rng)) + "; " + _cog_txt(it["cognition"], rng) + ".")
    s.append(pick(rng, ["Urgency and up three times a night to pass urine.", "Frequency and nocturia."])
             if it["toileting"] else pick(rng, ["Continent, no urgency.", "Toileting independently."]))
    meds = [m for _, m in it["meds"]] + rng.sample(SAFE_MEDS, pick(rng, [1, 2, 3]))
    rng.shuffle(meds)
    s.append("Medicines: " + ", ".join(meds) + ".")
    s.append(("Attached to: " + ", ".join(it["tethers"]) + ".") if it["tethers"] else "No lines or tubes.")
    d = it["ortho_drop"]
    if d is None:
        s.append("Lying and standing BP not done.")
    else:
        lying = rng.randint(112, 158)
        s.append(f"Lying BP {lying}/76, standing after 3 min {lying - d}/72.")
    if it["sensory"]:
        s.append(_cap(it["sensory"]) + ".")
    if it["injury_flag"]:
        s.append(_cap(it["injury_flag"]) + ".")
    return " ".join(s)


def _items_terse(it, rng):
    """The same findings as a flowsheet line (the reassessment's house style)."""
    fm = it["fall_months_ago"]
    mob = {"independent": "indep", "walking_aid": pick(rng, ["frame", "stick"]), "one_assist": "assist x1",
           "bed_bound": "bed-bound/hoist"}[it["mobility"]]
    cog = {"oriented": "A&O", "forgetful_impulsive": "forgetful, impulsive", "delirium": "delirium +ve"}[it["cognition"]]
    meds = [m for _, m in it["meds"]] + rng.sample(SAFE_MEDS, pick(rng, [1, 2]))
    rng.shuffle(meds)
    d = it["ortho_drop"]
    parts = [f"{it['age']} y", "no prior falls" if fm is None else f"fell {fm}/12 ago", mob, cog,
             "urgency/nocturia" if it["toileting"] else "continent", "meds " + "; ".join(meds),
             ("lines " + "; ".join(it["tethers"])) if it["tethers"] else "no lines",
             "L/S BP not done" if d is None else f"postural SBP drop {d}"]
    if it["sensory"]:
        parts.append(it["sensory"])
    if it["injury_flag"]:
        parts.append(it["injury_flag"])
    return " | ".join(parts)


def _social(rng, reg):
    lang = "English" if chance(rng, 0.75) else pick(rng, LANGUAGES)
    return {"language": lang,
            "interpreter_now": None if lang == "English" else pick_w(rng, {True: 3, False: 2}),
            "carer": pick_w(rng, {"daughter visits daily": 2, "partner stays most days": 2, "lives alone, no family "
                                  "nearby": 2, "son can stay overnight": 1.2, "carer from an agency, mornings only": 1}),
            "race_ethnicity": pick(rng, RACE_ETH), "sex": pick(rng, ["F", "M"])}


# =========================================================================== schema 1: admission screen
class AdmissionFallScreen(Schema):
    name = "admission_fall_screen"
    description = ("The admission nursing assessment: the patient's fall-risk band on the written index, the bundle "
                   "and assistance it calls for, and how fall-prevention teaching should be given.")
    surfaces = {"json": 0.55, "admission_note": 0.225, "assessment_flowsheet": 0.225}

    def sample_factors(self, rng):
        reg = regime(rng)
        tool = _tool(rng)
        want = pick_w(rng, {0: 1, 1: 1.1, 2: 1.2})
        lo_edge, hi_edge = tool["mod"], tool["high"]
        if reg == "borderline":
            edge = lo_edge if want <= 1 and chance(rng, 0.5) else hi_edge
            it = _steer(rng, reg, tool, target=lambda t: abs(t - edge) <= 1 or abs(t - edge + 1) <= 0)
        else:
            it = _steer(rng, reg, tool, target=lambda t: (t < lo_edge) if want == 0 else
                        (lo_edge <= t < hi_edge) if want == 1 else (t >= hi_edge))
        soc = _social(rng, reg)
        remark = None
        if reg == "conflicting":
            remark = pick(rng, ["Patient insists they 'never fall' and does not want the bed alarm.",
                                "Family says the patient is 'sharp as a tack' at home.",
                                "Patient says they will not use the call bell, does not like to bother staff.",
                                "Day nurse handed over 'low risk, walks fine'."])
        t = rand_ts(rng)
        return {"regime": reg, "tool": tool, "items": it, "social": soc, "hospital": _hospital(rng),
                "ward": pick(rng, WARDS), "admitted_for": pick(rng, ADMISSIONS), "remark": remark, "time": t,
                "bed": f"Bed {rng.randint(1, 32)}", "mrn": f"MRN {rng.randint(1000000, 9999999)}", "nurse": _nurse(rng)}

    def intended(self, f):
        tool, it, soc = f["tool"], f["items"], f["social"]
        total = _score(tool, it)
        band = _band(tool, total, it)
        assist = {"independent": 0, "walking_aid": 1, "one_assist": 2, "bed_bound": 3}[it["mobility"]]
        if it["cognition"] == "delirium" or it["ortho_drop"] is None or it["ortho_drop"] >= tool["ortho"]:
            assist = min(3, assist + 1)
        b = min(band, 2)
        if b == 2 and it["cognition"] in ("delirium", "forgetful_impulsive"):
            bundle = "high_bundle_with_video_review"
        else:
            bundle = ["standard_precautions", "moderate_bundle", "high_bundle"][b]
        carer = not soc["carer"].startswith("lives alone")
        if it["cognition"] != "oriented" and carer:
            teach = "teach_carer_with_patient"
        elif soc["language"] != "English":
            teach = "teach_back_via_qualified_interpreter" if soc["interpreter_now"] else "hold_teaching_for_interpreter"
        else:
            teach = "teach_back_with_patient"
        toilet = b >= 1 and (it["toileting"] or any(c == "diuretic" for c, _ in it["meds"]))
        return {"fall_risk_band": band, "assistance_level": assist, "prevention_bundle": bundle,
                "teaching_approach": teach, "scheduled_toileting_required": "yes" if toilet else "no"}

    def _protocol(self, f, rng):
        return {
            "index": _tool_text(f["tool"], "admission"),
            "bundles": "Standard (all): call bell in reach, non-slip socks, bed low and locked, clear path. Moderate: "
                       "add a bed/chair alarm, hourly rounding and toileting every 2 h. High: add a low bed with floor "
                       "mats and a room near the nurses' station; if also forgetful, impulsive or delirious, ask for a "
                       "video-monitoring review.",
            "assistance": f"Assistance: independent none; walking aid standby; needs one person a one-person assist; "
                          f"bed-bound two-person or hoist; one step more with delirium or a {f['tool']['ortho']}+ "
                          f"drop (or none measured).",
            "toileting": "Two-hourly scheduled toileting for moderate- or high-band patients who score the toileting "
                         "item or take a diuretic; a low-band patient with urgency gets a bedside commode instead.",
            "teaching": "Teach-back with the patient, or with the carer and patient together if the patient is "
                        "forgetful or delirious and has a carer; other languages through a qualified interpreter, "
                        "never a relative; with none available yet, start the bundle, hold the teaching, book one.",
        }

    def render(self, f, rng, surface):
        it, soc = f["items"], f["social"]
        pr = self._protocol(f, rng)
        narr = noisy(_items_narrative(it, rng), rng, 0.08)
        lang = soc["language"] + ("" if soc["language"] == "English" else
                                  " (" + ("qualified interpreter available now" if soc["interpreter_now"]
                                          else "no interpreter until tomorrow") + ")")
        if surface == "json":
            return dumps(prune({
                "admission": {"hospital": f["hospital"], "ward": f["ward"], "bed": f["bed"], "id": f["mrn"],
                              "time": _hm(f["time"]), "reason": f["admitted_for"], "nurse": f["nurse"]},
                "patient": {"sex": soc["sex"], "race_ethnicity_recorded": soc["race_ethnicity"],
                            "preferred_language": lang, "family_and_carers": soc["carer"]},
                "assessment_notes": narr,
                "handover_comment": f["remark"],
                "protocol": pr,
            }))
        if surface == "admission_note":
            return (f"NURSING ADMISSION -- {f['ward']}, {f['hospital']}\n{f['bed']} {f['mrn']} ({soc['sex']}), admitted "
                    f"{_hm(f['time'])} with {f['admitted_for']}.\nLanguage: {lang}. Family/carers: {soc['carer']}. "
                    f"Race/ethnicity (recorded): {soc['race_ethnicity']}.\n\n{narr}"
                    + (f"\n{f['remark']}" if f["remark"] else "") + f"\n\n{f['nurse']}\n\nFalls protocol:\n"
                    + bullets(list(pr.values()), rng))
        meds = ", ".join(m for _, m in it["meds"]) or "none of the listed classes noted"
        rows = [("Admitted", f"{_hm(f['time'])}, {f['admitted_for']}"), ("Age / sex", f"{it['age']} / {soc['sex']}"),
                ("Previous falls", "none" if it["fall_months_ago"] is None else f"{it['fall_months_ago']} months ago"),
                ("Mobility", _mobility_txt(it["mobility"], rng)), ("Cognition", _cog_txt(it["cognition"], rng)),
                ("Toileting", "urgency/nocturia" if it["toileting"] else "no issues"),
                ("Medicines of note", meds), ("Lines/tubes", ", ".join(it["tethers"]) or "none"),
                ("Orthostatic drop", "not measured" if it["ortho_drop"] is None else f"{it['ortho_drop']} mmHg"),
                ("Vision/hearing", it["sensory"] or "no deficit"), ("Injury risk", it["injury_flag"] or "none noted"),
                ("Preferred language", lang), ("Family/carers", soc["carer"]),
                ("Race/ethnicity (recorded)", soc["race_ethnicity"]), ("Handover comment", f["remark"])]
        return (f"FALLS ASSESSMENT FLOWSHEET -- {f['bed']}, {f['ward']}\n" + kv_lines(rows, rng)
                + "\n\nProtocol:\n" + "\n".join(f"- {v}" for v in pr.values()))

    def questions(self):
        return {
            "fall_risk_band": score("What fall-risk band does the written index give this patient?", [
                "Low risk.",
                "Moderate risk.",
                "High risk.",
                "High risk and injury-prone (anticoagulant, osteoporosis or recent hip surgery).",
            ]),
            "assistance_level": score("How much help should staff give this patient to stand, transfer and walk?", [
                "None: walks independently.",
                "Standby supervision.",
                "Hands-on help from one person.",
                "Two people or a hoist.",
            ]),
            "prevention_bundle": choice("Which fall-prevention bundle should be put in place?", {
                "high_bundle": "The high-risk bundle: low bed, floor mats and a room near the nurses' station.",
                "high_bundle_with_video_review": "The high-risk bundle plus a review for video monitoring.",
                "moderate_bundle": "The moderate bundle: bed or chair alarm, hourly rounding, two-hourly toileting.",
                "standard_precautions": "Standard precautions only.",
            }),
            "teaching_approach": choice("How should fall-prevention teaching be given?", {
                "hold_teaching_for_interpreter": "Start the bundle, hold the teaching and book a qualified "
                                                 "interpreter.",
                "teach_back_via_qualified_interpreter": "Teach-back with the patient through a qualified interpreter.",
                "teach_back_with_patient": "Teach-back directly with the patient.",
                "teach_carer_with_patient": "Teach the carer together with the patient.",
            }),
            "scheduled_toileting_required": noul(
                "This patient needs scheduled toileting under the protocol.",
                yes="Moderate or high band, and either the toileting item scores or a diuretic is prescribed.",
                no="Low band, or no toileting item and no diuretic."),
        }


# =========================================================================== schema 2: shift reassessment
TRIGGERS = {
    "pca_started": ("morphine PCA started after surgery", {"meds_add": ("opioid", "morphine PCA"),
                                                           "tether_add": "PCA pump"}),
    "delirium_new": ("delirium screen turned positive; haloperidol given",
                     {"cognition": "delirium", "meds_add": ("antipsychotic", "haloperidol")}),
    "sedative_given": ("zopiclone given for sleep", {"meds_add": ("sedative", "zopiclone")}),
    "diuretic_started": ("IV furosemide started for fluid overload", {"meds_add": ("diuretic", "furosemide"),
                                                                     "toileting": True}),
    "catheter_removed": ("urinary catheter removed", {"tether_remove": "urinary catheter",
                                                                  "toileting": True}),
    "back_from_theatre": ("back from theatre with a PCA and catheter, first time out of bed due today",
                          {"mobility": "one_assist", "meds_add": ("opioid", "morphine PCA"),
                           "tether_add": "urinary catheter"}),
    "mobility_better": ("physio: now walking with a frame, steady", {"mobility": "walking_aid"}),
    "confusion_cleared": ("delirium screen now negative, back to baseline", {"cognition": "oriented"}),
    "lines_down": ("IV and oxygen stopped", {"tether_clear": True}),
}
NON_TRIGGERS = ["moved to a side room", "family visited in the afternoon", "physio session, no change",
                "chaplain visit", "dressing changed", "meal assistance started"]


class ShiftReassessment(Schema):
    name = "shift_reassessment"
    description = ("A patient already scored on the index has had a change since the last score: rescore, adjust the "
                   "bundle and check the reassessment clock.")
    surfaces = {"json": 0.55, "sbar_handoff": 0.225, "score_sheet": 0.225}

    def sample_factors(self, rng):
        reg = regime(rng)
        tool = _tool(rng)
        want = pick_w(rng, {-1: 1, 0: 1.1, 1: 1.4, 2: 0.7, "none": 0.9})
        for _ in range(80):
            it0 = _steer(rng, "clear" if reg == "clear" else "borderline", tool)
            if want == "none":
                ev, txt, eff = None, pick(rng, NON_TRIGGERS), {}
            else:
                ev = pick(rng, list(TRIGGERS))
                txt, eff = TRIGGERS[ev]
            it1 = self._apply(it0, eff)
            d = min(2, _band(tool, _score(tool, it1))) - min(2, _band(tool, _score(tool, it0)))
            if want == "none" or d == want:
                break
        t0 = rand_ts(rng, hours=(7, 20)).replace(second=0)
        sheet_total = _score(tool, it0)
        if chance(rng, {"conflicting": 0.85, "borderline": 0.3, "clear": 0.08}[reg]):
            sheet_total = max(0, sheet_total + pick(rng, [-4, -3, -2, -1, 1, 2, 3]))
        trig_limit = pick(rng, [2, 4])
        hours_since_score = round(rng.uniform(3, 14), 1)
        hours_since_trigger = round(rng.uniform(0.2, hours_since_score), 1)
        if reg == "borderline":
            hours_since_trigger = round(near(rng, trig_limit, 0.6, lo=0.2), 1)
            hours_since_score = max(hours_since_trigger + 0.5, round(rng.uniform(4, 12.5), 1))
        now = t0 + timedelta(hours=hours_since_score)
        return {"regime": reg, "tool": tool, "items_at_last_score": it0, "items_now": it1, "event": ev,
                "is_trigger": ev is not None,
                "event_text": txt, "sheet_total": sheet_total, "last_score_time": t0, "now": now,
                "event_time": now - timedelta(hours=hours_since_trigger), "trigger_limit_h": trig_limit,
                "routine_interval_h": 12, "hospital": _hospital(rng), "ward": pick(rng, WARDS),
                "bed": f"Bed {rng.randint(1, 32)}", "admitted_for": pick(rng, ADMISSIONS), "nurse": _nurse(rng),
                "remark": (pick(rng, ["Night nurse: 'nothing has really changed, leave the bundle as it is.'",
                                      "Patient keen to have the bed alarm switched off.",
                                      "Team asks to drop the high-risk measures now the patient is better."])
                           if reg == "conflicting" else None)}

    @staticmethod
    def _apply(it0, eff):
        it1 = {k: (list(v) if isinstance(v, list) else v) for k, v in it0.items()}
        if "meds_add" in eff and all(c != eff["meds_add"][0] for c, _ in it1["meds"]):
            it1["meds"].append(eff["meds_add"])
        if "tether_add" in eff and eff["tether_add"] not in it1["tethers"]:
            it1["tethers"].append(eff["tether_add"])
        if "tether_remove" in eff and eff["tether_remove"] in it1["tethers"]:
            it1["tethers"].remove(eff["tether_remove"])
        if eff.get("tether_clear"):
            it1["tethers"] = [x for x in it1["tethers"] if x not in ("IV pump", "oxygen tubing")]
        for k in ("cognition", "mobility", "toileting"):
            if k in eff:
                it1[k] = eff[k]
        return it1

    def intended(self, f):
        tool = f["tool"]
        old_total = _score(tool, f["items_at_last_score"])
        new_total = _score(tool, f["items_now"])
        b0 = min(2, _band(tool, old_total))
        b1 = min(2, _band(tool, new_total))
        diff = b1 - b0
        delta = new_total - old_total
        change = 0 if delta <= -2 else 1 if delta <= 1 else 2 if delta <= 3 else 3
        bundle = ("step_down_bundle" if diff < 0 else "keep_current_bundle" if diff == 0 else
                  "escalate_to_high_bundle" if b1 == 2 else "add_moderate_measures")
        h_trig = (f["now"] - f["event_time"]).total_seconds() / 3600
        h_score = (f["now"] - f["last_score_time"]).total_seconds() / 3600
        trig_over = f["is_trigger"] and h_trig > f["trigger_limit_h"]
        overdue = trig_over or h_score > f["routine_interval_h"]
        if overdue:
            timing = 3
        elif f["is_trigger"]:
            timing = 2 if f["trigger_limit_h"] - h_trig <= 1 else 1
        else:
            timing = 2 if f["routine_interval_h"] - h_score <= 1 else 0
        return {"score_change": change, "bundle_change": bundle, "reassessment_timing": timing,
                "reassessment_overdue": "yes" if overdue else "no",
                "score_sheet_total_reliable": "yes" if f["sheet_total"] == old_total else "no"}

    def _protocol(self, f, rng):
        return {"index": _tool_text(f["tool"], "short"),
                "when": f"Rescore every {f['routine_interval_h']} h and within {f['trigger_limit_h']} h of a trigger "
                        f"(new opioid, sedative or antipsychotic, delirium or its resolution, surgery, diuretic, a line "
                        f"or catheter change, a change in mobility). Other events do not start the trigger clock.",
                "bundles": "Bundle follows the band: low standard, moderate adds alarm/rounding/toileting, high adds "
                           "low bed, mats and a room near the station. Step down only on a rescore."}

    def render(self, f, rng, surface):
        pr = self._protocol(f, rng)
        it0, it1 = f["items_at_last_score"], f["items_now"]
        now_txt = _items_terse(it1, rng)
        if surface == "json":
            return dumps(prune({
                "patient": {"bed": f["bed"], "ward": f["ward"], "admitted_for": f["admitted_for"]},
                "last_score": {"time": _hm(f["last_score_time"]), "total_on_sheet": f["sheet_total"]},
                "since_then": {"event": f["event_text"], "event_time": _hm(f["event_time"])},
                "now": {"time": _hm(f["now"]), "assessment": now_txt, "comment": f["remark"]},
                "protocol": pr,
            }))
        if surface == "sbar_handoff":
            return (f"SBAR -- {f['bed']}, {f['ward']} ({f['nurse']}, {_hm(f['now'])})\n"
                    f"S: {f['admitted_for']}. At {_hm(f['event_time'])}: {f['event_text']}.\n"
                    f"B: falls score {f['sheet_total']} on the sheet at {_hm(f['last_score_time'])}.\n"
                    f"A: {now_txt}\nR: please review falls plan." + (f"\n{f['remark']}" if f["remark"] else "")
                    + "\n\nProtocol:\n" + "\n".join(f"- {v}" for v in pr.values()))
        rows = [["last score", _hm(f["last_score_time"]), f"total {f['sheet_total']}"],
                ["trigger", _hm(f["event_time"]), f["event_text"]],
                ["now", _hm(f["now"]), "reassessment in progress"]]
        return (f"FALLS SCORE SHEET {f['bed']} ({f['hospital']})\n" + table(["entry", "time", "detail"], rows, rng)
                + f"\n\nCurrent findings: {now_txt}" + (f"\nNote: {f['remark']}" if f["remark"] else "")
                + "\n\n" + bullets(list(pr.values()), rng))

    def questions(self):
        return {
            "score_change": score("How much has the index total changed since the last score?", [
                "Fallen by two points or more.",
                "About the same, within one point either way.",
                "Up two or three points.",
                "Up four points or more.",
            ]),
            "bundle_change": choice("What should happen to the fall-prevention bundle now?", {
                "add_moderate_measures": "Add the moderate-risk measures.",
                "escalate_to_high_bundle": "Move to the high-risk bundle.",
                "keep_current_bundle": "Keep the current bundle.",
                "step_down_bundle": "Step down to the bundle for the lower band.",
            }),
            "reassessment_timing": score("How pressing is the rescore?", [
                "Not due until the next routine score.",
                "Due within the trigger window, with more than an hour left.",
                "Due within the hour.",
                "Overdue now.",
            ]),
            "reassessment_overdue": noul(
                "The rescore is already overdue under the protocol.",
                yes="More time has passed since the trigger than the trigger window, or since the last score than "
                    "the routine interval.",
                no="Both clocks are still inside their limits."),
            "score_sheet_total_reliable": noul(
                "The total written on the score sheet is what the index gives for the findings at that time."),
        }


# =========================================================================== schema 3: post-fall huddle
INJURIES = {
    "none": ("no injury found", 0), "bruise": ("small bruise on the forearm", 1),
    "skin_tear": ("skin tear on the shin, dressed", 1), "laceration": ("4 cm scalp laceration that will need closing", 2),
    "hip_pain": ("pain in the hip, cannot take weight", 2), "wrist_deformity": ("swollen, deformed wrist", 3),
}


class PostFallHuddle(Schema):
    name = "post_fall_huddle"
    description = ("A patient has fallen on the ward: grade the injury, choose the post-fall pathway and observation "
                   "timing, and update the family as the written post-fall protocol requires.")
    surfaces = {"json": 0.55, "incident_form": 0.225, "huddle_note": 0.225}

    def sample_factors(self, rng):
        reg = regime(rng)
        witnessed = chance(rng, 0.55)
        head = pick_w(rng, {"yes": 2.5, "no": 5, "unknown": 2 if not witnessed else 0.2})
        anticoag = pick_w(rng, {None: 6, pick(rng, ANTICOAG): 3})
        injury = pick_w(rng, {"none": 5, "bruise": 2.5, "skin_tear": 2, "laceration": 1.2, "hip_pain": 1.2,
                              "wrist_deformity": 0.6})
        new_conf = chance(rng, 0.12)
        gcs_drop = pick_w(rng, {0: 9, 1: 1.2, 2: 0.6})
        sbp = rng.randint(96, 168)
        hr = rng.randint(64, 108)
        if chance(rng, 0.06):
            sbp, hr = rng.randint(76, 88), rng.randint(112, 138)
        minutes = pick_w(rng, {rng.randint(5, 55): 3, rng.randint(60, 175): 3, rng.randint(180, 400): 2,
                               rng.randint(420, 700): 1})
        if reg == "borderline":
            minutes = pick(rng, [near(rng, 60, 8, integer=True), near(rng, 180, 10, integer=True),
                                 near(rng, 240, 15, integer=True)])
        fam_lang = "English" if chance(rng, 0.7) else pick(rng, LANGUAGES)
        declines = chance(rng, 0.1) and not new_conf
        remark = None
        if reg == "conflicting":
            remark = pick(rng, ["Patient says 'I'm fine, I just slipped, don't make a fuss.'",
                                "Junior doctor on the phone: 'sounds minor, I'll see them on the morning round.'",
                                "Relative at the bedside offers to translate for the rest of the family."])
        t = rand_ts(rng)
        return {"regime": reg, "witnessed": witnessed, "head_strike": head, "anticoagulant": anticoag,
                "injury": injury, "new_confusion": new_conf, "gcs_drop": gcs_drop, "sbp": sbp, "hr": hr,
                "minutes_since_fall": minutes, "family_language": fam_lang, "patient_declines_family_contact":
                declines, "remark": remark, "fall_time": t, "hospital": _hospital(rng), "ward": pick(rng, WARDS),
                "bed": f"Bed {rng.randint(1, 32)}", "found_by": _nurse(rng), "age": rng.randint(48, 96),
                "family_window_h": pick(rng, [2, 4]),
                "where": pick(rng, ["beside the bed", "in the bathroom", "in the corridor", "by the chair",
                                    "at the toilet door"])}

    @staticmethod
    def _possible_head(f):
        return f["head_strike"] == "yes" or (f["head_strike"] == "unknown" or not f["witnessed"])

    def intended(self, f):
        sev = INJURIES[f["injury"]][1]
        if f["new_confusion"] or f["gcs_drop"] >= 1:
            sev = max(sev, 3 if f["gcs_drop"] >= 2 else 2)
        head = self._possible_head(f)
        if f["gcs_drop"] >= 2 or f["sbp"] < 90 or f["hr"] > 130:
            path = "rapid_response_call"
        elif (head and f["anticoagulant"]) or f["new_confusion"] or f["gcs_drop"] == 1 or \
                f["injury"] in ("hip_pain", "wrist_deformity"):
            path = "urgent_review_and_imaging"
        elif head:
            path = "neuro_observation_protocol"
        else:
            path = "routine_post_fall_obs"
        m = f["minutes_since_fall"]
        if path == "rapid_response_call":
            nxt = 3
        elif head and path in ("neuro_observation_protocol", "urgent_review_and_imaging"):
            nxt = 3 if m < 60 else 2 if m < 180 else 1 if m < 420 else 0
        elif path == "urgent_review_and_imaging":
            nxt = 2
        else:
            nxt = 1 if m < 60 else 0
        required = sev >= 1 or head
        if f["patient_declines_family_contact"]:
            fam = "respect_patient_declining"
        elif not required:
            fam = "tell_family_at_next_contact"
        elif f["family_language"] == "English":
            fam = "phone_family_now"
        else:
            fam = "phone_family_now_with_interpreter"
        return {"injury_level": sev, "next_observation_due": nxt, "post_fall_pathway": path,
                "family_update": fam, "harm_review_required": "yes" if sev >= 2 else "no"}

    def _protocol(self, f):
        return {
            "injury": "Injury grades: none; minor (bruise, abrasion, skin tear); moderate (wound needing closure, "
                      "suspected fracture, new confusion or a one-point drop in GCS); major (deformity, or a GCS drop "
                      "of two or more).",
            "pathway": "Rapid response for a GCS drop of 2+, SBP < 90 or HR > 130. Urgent review and imaging for a "
                       "possible head strike on an anticoagulant, new confusion, a GCS drop of 1, or a suspected "
                       "fracture. Neuro observations for any possible head strike (an unwitnessed fall counts). "
                       "Otherwise routine post-fall observations.",
            "observations": "Neuro observations every 15 min for the first hour, every 30 min to 3 h, hourly to 7 h, "
                            "then routine; urgent-review patients without a head strike every 30 min; routine "
                            "post-fall checks hourly for the first hour, then routine.",
            "family": f"Tell the family within {f['family_window_h']} h after any injury or possible head strike, by "
                      f"phone, in their preferred language through a qualified interpreter (not a relative); "
                      f"uninjured falls are shared at the next family contact; respect a patient with capacity who "
                      f"declines. Moderate or major harm needs a harm review.",
        }

    def render(self, f, rng, surface):
        pr = self._protocol(f)
        ft, now = f["fall_time"], f["fall_time"] + timedelta(minutes=f["minutes_since_fall"])
        head = {"yes": "hit their head", "no": "did not hit their head", "unknown": "unclear whether the head was hit"}
        story = noisy(join_sentences([
            f"{'Witnessed' if f['witnessed'] else 'Unwitnessed'} fall {f['where']} at {_hm(ft)}; "
            f"{head[f['head_strike']]}.", f"Injury: {INJURIES[f['injury']][0]}.",
            "New confusion compared with before the fall." if f["new_confusion"] else "",
            f"GCS {15 - f['gcs_drop']} (15 before the fall).", f"BP {f['sbp']}/{int(f['sbp'] * 0.6)}, HR {f['hr']}.",
            f"Takes {f['anticoagulant']}." if f["anticoagulant"] else "No anticoagulant.",
            "Patient has capacity and asks that the family not be told." if f["patient_declines_family_contact"] else "",
            f["remark"] or ""]), rng, 0.08)
        fam = f"Family's preferred language: {f['family_language']}."
        if surface == "json":
            return dumps(prune({
                "event": {"ward": f["ward"], "bed": f["bed"], "patient_age": f["age"], "found_by": f["found_by"],
                          "fall_time": _hm(ft), "now": _hm(now)},
                "findings": story, "family": fam,
                "protocol": pr,
            }))
        if surface == "incident_form":
            rows = [("Location", f"{f['ward']}, {f['bed']}, {f['where']}"), ("Patient age", f["age"]),
                    ("Time of fall", _hm(ft)), ("Form completed", _hm(now)),
                    ("Witnessed", "yes" if f["witnessed"] else "no"), ("Head strike", f["head_strike"]),
                    ("Injury", INJURIES[f["injury"]][0]), ("New confusion", "yes" if f["new_confusion"] else "no"),
                    ("GCS", f"{15 - f['gcs_drop']}/15"), ("Vitals", f"BP {f['sbp']}/{int(f['sbp'] * 0.6)}, HR {f['hr']}"),
                    ("Anticoagulant", f["anticoagulant"] or "none"), ("Family language", f["family_language"]),
                    ("Family contact", "patient with capacity declines" if f["patient_declines_family_contact"]
                     else "not yet made"), ("Comment", f["remark"])]
            return ("POST-FALL INCIDENT FORM\n" + kv_lines(rows, rng) + "\n\nPost-fall protocol:\n"
                    + "\n".join(f"* {v}" for v in pr.values()))
        return (f"POST-FALL HUDDLE {_hm(now)} -- {f['ward']}\n{f['bed']}, age {f['age']}. {story} {fam}\n\n"
                + bullets(list(pr.values()), rng))

    def questions(self):
        return {
            "injury_level": score("How severe is the injury from this fall under the protocol's grades?", [
                "None.", "Minor.", "Moderate.", "Major.",
            ]),
            "next_observation_due": score("How soon is the next set of observations due?", [
                "At the routine round.", "Within the hour.", "Within 30 minutes.", "Within 15 minutes, or now.",
            ]),
            "post_fall_pathway": choice("Which post-fall pathway applies?", {
                "neuro_observation_protocol": "Start neurological observations on the protocol schedule.",
                "rapid_response_call": "Call the rapid response team now.",
                "routine_post_fall_obs": "Routine post-fall observations and a provider review on the next round.",
                "urgent_review_and_imaging": "Get an urgent provider review with imaging.",
            }),
            "family_update": choice("How should the family be told about the fall?", {
                "phone_family_now": "Phone the family now.",
                "phone_family_now_with_interpreter": "Phone the family now through a qualified interpreter.",
                "respect_patient_declining": "Do not contact the family; the patient has declined.",
                "tell_family_at_next_contact": "Tell the family at the next routine contact.",
            }),
            "harm_review_required": noul(
                "This fall needs a formal harm review.",
                yes="The injury grades as moderate or major under the protocol.",
                no="No injury, or a minor one only."),
        }


# =========================================================================== schema 4: sitter or video monitoring
class SitterVideoRequest(Schema):
    name = "sitter_video_monitoring_request"
    description = ("The charge nurse asks for enhanced observation for a high-risk patient: a 1:1 sitter, video "
                   "monitoring, family presence or the bundle alone, under the written eligibility rules.")
    surfaces = {"json": 0.55, "request_page": 0.225, "resource_board": 0.225}

    def sample_factors(self, rng):
        reg = regime(rng)
        band = pick_w(rng, {"moderate": 2, "high": 6})
        attempts = pick_w(rng, {0: 3, 1: 3, 2: 2, 3: 1.2, 5: 0.8})
        pulling = chance(rng, 0.2)
        delirium = chance(rng, 0.45)
        follows = pick_w(rng, {"yes": 3, "partly": 2, "no": 1.5})
        lang = "English" if chance(rng, 0.65) else pick(rng, LANGUAGES)
        video_langs = pick(rng, [["English", "Spanish"], ["English"], ["English", "Spanish", "Vietnamese"],
                                 ["English", "Polish", "Arabic"]])
        hearing = chance(rng, 0.15)
        sitters = pick_w(rng, {0: 3, 1: 3, 2: 1})
        slots = pick_w(rng, {0: 2, 1: 3, 2: 2, 4: 1})
        family = pick_w(rng, {"son can stay overnight": 2, "daughter can stay until 21:00 only": 2,
                              "no family able to stay": 3, "partner can stay overnight": 1.5})
        exit_thr = pick(rng, [2, 3])
        if reg == "borderline":
            attempts = pick(rng, [exit_thr - 1, exit_thr])
        remark = None
        if reg == "conflicting":
            remark = pick(rng, ["Nurse in charge: 'just put a sitter in, it's easier'",
                                "Family member offers to 'keep an eye' from home by phone",
                                "Day team says the patient is 'only agitated in the evenings'"])
        t = rand_ts(rng)
        return {"regime": reg, "band": band, "exit_attempts_shift": attempts, "pulling_lines": pulling,
                "delirium": delirium, "follows_redirection": follows, "language": lang, "video_languages": video_langs,
                "severe_hearing_loss": hearing, "sitters_available": sitters, "video_slots_free": slots,
                "family": family, "exit_threshold": exit_thr, "remark": remark, "time": t,
                "hospital": _hospital(rng), "ward": pick(rng, WARDS), "bed": f"Bed {rng.randint(1, 32)}",
                "requester": _nurse(rng), "age": rng.randint(55, 97), "admitted_for": pick(rng, ADMISSIONS)}

    def intended(self, f):
        sitter = f["band"] == "high" and (f["pulling_lines"] or f["exit_attempts_shift"] >= f["exit_threshold"])
        feasible = (f["follows_redirection"] != "no" and f["language"] in f["video_languages"]
                    and not f["severe_hearing_loss"])
        video = (f["band"] == "high" or f["exit_attempts_shift"] >= 1) and feasible
        overnight = "overnight" in f["family"]
        if sitter:
            alloc = "one_to_one_sitter" if f["sitters_available"] else "escalate_for_sitter_cover"
        elif video and f["video_slots_free"]:
            alloc = "video_monitoring"
        elif overnight:
            alloc = "family_presence_with_bundle"
        else:
            alloc = "intensify_bundle_only"
        a = f["exit_attempts_shift"]
        beh = 3 if (f["pulling_lines"] or a >= 4) else 2 if a >= 2 else 1 if (a == 1 or f["delirium"]) else 0
        start = 3 if sitter else 2 if alloc == "video_monitoring" else 1 if alloc == "family_presence_with_bundle" else 0
        return {"observation_allocation": alloc, "unsafe_behaviour_level": beh, "observation_start_by": start,
                "sitter_criteria_met": "yes" if sitter else "no",
                "video_redirection_feasible": "yes" if feasible else "no"}

    def _protocol(self, f):
        langs = ", ".join(f["video_languages"])
        return {
            "sitter": f"1:1 sitter: high band and either pulling at lines or tubes or {f['exit_threshold']}+ attempts "
                      f"to get out of bed unaided this shift. No sitter free: escalate to the nurse manager for cover.",
            "video": f"Video monitoring: high band, or any band with an unaided exit attempt, and the patient can "
                     f"follow spoken redirection. Monitor staff redirect in {langs} only; severe hearing loss rules "
                     f"it out.",
            "family": "A relative who can stay through the night may stand in for video monitoring (never for a "
                      "sitter).",
            "otherwise": "Otherwise intensify the bundle: hourly rounding by name, alarm checks, room near the station.",
            "timing": "Start a sitter at once, video within the hour, family presence by the evening.",
        }

    def render(self, f, rng, surface):
        pr = self._protocol(f)
        beh = noisy(join_sentences([
            ("No attempts" if f["exit_attempts_shift"] == 0 else
             "One attempt" if f["exit_attempts_shift"] == 1 else f"{f['exit_attempts_shift']} attempts")
            + " to climb out of bed unaided this shift.",
            "Pulling at the IV line and catheter." if f["pulling_lines"] else "Leaving lines alone.",
            "Delirium screen positive." if f["delirium"] else "Delirium screen negative.",
            {"yes": "Settles when spoken to.", "partly": "Sometimes settles when spoken to.",
             "no": "Does not respond to spoken redirection."}[f["follows_redirection"]],
            "Severe hearing loss." if f["severe_hearing_loss"] else "",
            f["remark"] or ""]), rng, 0.08)
        if surface == "json":
            return dumps(prune({
                "request": {"from": f["requester"], "time": _hm(f["time"]), "ward": f["ward"], "bed": f["bed"],
                            "patient": f"{f['age']}, {f['admitted_for']}", "falls_band": f["band"]},
                "behaviour": beh,
                "patient_context": {"preferred_language": f["language"], "family": f["family"]},
                "resources": {"sitters_free": f["sitters_available"], "video_slots_free": f["video_slots_free"]},
                "protocol": pr,
            }))
        if surface == "request_page":
            return (f"PAGE {_hm(f['time'])} from {f['requester']} ({f['ward']}): enhanced observation request for "
                    f"{f['bed']}, age {f['age']}, {f['admitted_for']}, falls band {f['band']}.\n{beh}\nLanguage: "
                    f"{f['language']}. Family: {f['family']}.\nSitters free: {f['sitters_available']}; video slots "
                    f"free: {f['video_slots_free']}.\n\nEnhanced observation policy:\n"
                    + "\n".join(f"- {v}" for v in pr.values()))
        rows = [["1:1 sitters", f["sitters_available"]], ["video monitoring slots", f["video_slots_free"]],
                ["video redirection languages", ", ".join(f["video_languages"])]]
        return (f"OBSERVATION RESOURCES -- {f['hospital']} {_hm(f['time'])}\n" + table(["resource", "free"], rows, rng)
                + f"\n\nRequest for {f['bed']} ({f['ward']}), band {f['band']}: {beh} Prefers {f['language']}; "
                  f"{f['family']}.\n\n" + bullets(list(pr.values()), rng))

    def questions(self):
        return {
            "observation_allocation": choice("What enhanced observation should this patient get?", {
                "escalate_for_sitter_cover": "Escalate to the nurse manager to find a sitter; none is free.",
                "family_presence_with_bundle": "Ask the relative to stay overnight alongside the bundle.",
                "intensify_bundle_only": "No extra observation; intensify the bundle.",
                "one_to_one_sitter": "Assign a 1:1 sitter.",
                "video_monitoring": "Put the patient on video monitoring.",
            }),
            "unsafe_behaviour_level": score("How unsafe is the patient's behaviour on this shift?", [
                "Settled: no unaided exits and no interfering with lines.",
                "Some risk: one unaided exit or delirium without attempts.",
                "Repeated: two or three unaided exit attempts.",
                "Continuous: pulling at lines or four or more attempts.",
            ]),
            "observation_start_by": score("How soon must the enhanced observation start?", [
                "No enhanced observation is due.", "By this evening.", "Within the hour.", "Immediately.",
            ]),
            "sitter_criteria_met": noul(
                "This patient meets the written criteria for a 1:1 sitter.",
                yes="High band, and pulling at lines or tubes or enough unaided exit attempts this shift.",
                no="Not high band, or neither line-pulling nor enough exit attempts."),
            "video_redirection_feasible": noul(
                "Video monitoring staff could redirect this patient effectively."),
        }


# =========================================================================== schema 5: bundle compliance audit
ELEMENTS = {
    "call_bell": ("call bell within reach", "standard", True),
    "socks": ("non-slip socks on", "standard", False),
    "bed_low": ("bed low and locked", "standard", False),
    "clear_path": ("clear path to the bathroom", "standard", False),
    "teaching_record": ("fall teaching documented in the preferred language", "standard", False),
    "bed_alarm": ("bed or chair alarm on", "moderate", True),
    "toilet_schedule": ("two-hourly toileting documented", "moderate", False),
    "rounding_log": ("hourly rounding log up to date", "moderate", False),
    "mats": ("floor mats down beside the bed", "high", True),
    "near_station": ("room near the nurses' station", "high", False),
}
LEVEL_ORDER = {"standard": 0, "moderate": 1, "high": 2}


class BundleComplianceAudit(Schema):
    name = "bundle_compliance_audit"
    description = ("A spot audit of one patient's room against the bundle the protocol orders for their band: how big "
                   "the gap is, how fast it must be fixed, and what follow-up the unit owes.")
    surfaces = {"json": 0.55, "audit_checklist": 0.225, "auditor_comment": 0.225}

    def sample_factors(self, rng):
        reg = regime(rng)
        band = pick_w(rng, {"low": 2, "moderate": 3, "high": 3})
        lvl = {"low": 0, "moderate": 1, "high": 2}[band]
        expected = [k for k, (_, need, _) in ELEMENTS.items() if LEVEL_ORDER[need] <= lvl]
        n_miss = pick_w(rng, {0: 3, 1: 3, 2: 2, 3: 1.2})
        if reg == "clear":
            n_miss = pick(rng, [0, 0, 3])
        missing = rng.sample(expected, min(n_miss, len(expected)))
        if band == "high" and reg == "conflicting" and chance(rng, 0.6):
            missing = sorted(set(missing) | set(rng.sample(["call_bell", "bed_alarm", "mats"], 2)))
        ordered = band
        if chance(rng, {"conflicting": 0.8, "borderline": 0.3, "clear": 0.1}[reg]):
            ordered = pick(rng, [b for b in ("low", "moderate", "high") if b != band])
        lang = "English" if chance(rng, 0.75) else pick(rng, LANGUAGES)
        teaching_note = None
        if "teaching_record" in missing and lang != "English":
            teaching_note = pick(rng, ["teaching given in English with the patient's grandson translating",
                                       "teaching leaflet in English left at the bedside"])
        repeated = chance(rng, 0.35)
        remark = None
        if reg == "conflicting":
            remark = pick(rng, ["Nurse: 'the alarm kept going off, so I turned it off for a bit.'",
                                "Patient asked for the mats to be taken away, says they trip over them.",
                                "Nurse: 'I documented the teaching, it must be in the other folder.'"])
        t = rand_ts(rng)
        return {"regime": reg, "band": band, "ordered_bundle": ordered, "expected": expected, "missing": missing,
                "language": lang, "teaching_note": teaching_note, "repeat_element_on_unit": repeated,
                "repeat_element": pick(rng, missing) if missing and repeated else pick(rng, expected),
                "remark": remark, "time": t, "hospital": _hospital(rng), "ward": pick(rng, WARDS),
                "bed": f"Bed {rng.randint(1, 32)}", "auditor": person(rng), "score": None}

    def intended(self, f):
        miss = f["missing"]
        crit = [m for m in miss if ELEMENTS[m][2]]
        noncrit = [m for m in miss if not ELEMENTS[m][2]]
        if len(crit) >= 2:
            gap = 3
        elif crit or len(noncrit) >= 2:
            gap = 2
        elif noncrit:
            gap = 1
        else:
            gap = 0
        fix = 3 if crit else 2 if len(noncrit) >= 2 else 1 if noncrit else 0
        repeated = f["repeat_element_on_unit"] and f["repeat_element"] in miss
        if gap == 3 and f["band"] == "high":
            follow = "report_safety_event"
        elif repeated:
            follow = "unit_reeducation"
        elif miss:
            follow = "coach_nurse_on_shift"
        else:
            follow = "no_further_action"
        return {"compliance_gap": gap, "fix_urgency": fix, "audit_follow_up": follow,
                "audit_passes": "no" if miss else "yes",
                "ordered_bundle_matches_band": "yes" if f["ordered_bundle"] == f["band"] else "no"}

    def _protocol(self, f):
        return {
            "elements": "Every band: bell within the patient's reach (critical), grip socks on, bed lowered with "
                        "brakes on, walkway to the toilet unobstructed, teaching recorded in the patient's own "
                        "language (through a qualified interpreter where needed). From moderate: alarm switched on "
                        "(critical), toileting round charted, rounding chart current. From high: mats on the floor "
                        "(critical), bed space close to the station.",
            "grading": "Gap: none; minor (one non-critical element missing); major (two or more non-critical, or one "
                       "critical); unsafe (two or more critical). Fix critical elements before leaving the room, "
                       "several non-critical within the hour, a single one by the end of the shift.",
            "follow_up": "Unsafe gap on a high-band patient: report a safety event. An element missed again that the "
                         "unit missed in 3 of its last 10 audits: unit re-education. Any other gap: coach the nurse on "
                         "shift.",
        }

    def render(self, f, rng, surface):
        pr = self._protocol(f)
        miss = set(f["missing"])
        teach_txt = f["teaching_note"]
        if surface == "json":
            return dumps(prune({
                "audit": {"auditor": f["auditor"], "time": _hm(f["time"]), "ward": f["ward"], "bed": f["bed"]},
                "patient": {"falls_band_on_latest_score": f["band"], "bundle_ordered": f["ordered_bundle"],
                            "preferred_language": f["language"]},
                "observed": {ELEMENTS[k][0]: ("no" if k in miss else "yes") for k in f["expected"]},
                "notes": {"teaching": teach_txt, "comment": f["remark"],
                          "unit_history": (f"'{ELEMENTS[f['repeat_element']][0]}' missed in 3 of the last 10 unit "
                                           f"audits" if f["repeat_element_on_unit"] else
                                           "no element missed repeatedly in recent audits")},
                "protocol": pr,
            }))
        if surface == "audit_checklist":
            rows = [[ELEMENTS[k][0], "MISSING" if k in miss else "ok"] for k in f["expected"]]
            hist = (f"Unit history: '{ELEMENTS[f['repeat_element']][0]}' missed in 3 of the last 10 audits."
                    if f["repeat_element_on_unit"] else "Unit history: no repeated misses.")
            return (f"FALLS BUNDLE AUDIT -- {f['ward']}, {f['bed']} ({_hm(f['time'])}, {f['auditor']})\n"
                    f"Latest band: {f['band']}; bundle ordered: {f['ordered_bundle']}; language: {f['language']}\n"
                    + table(["element", "status"], rows, rng) + f"\n{hist}"
                    + (f"\nTeaching: {teach_txt}" if teach_txt else "") + (f"\n{f['remark']}" if f["remark"] else "")
                    + "\n\n" + "\n".join(f"- {v}" for v in pr.values()))
        seen = [ELEMENTS[k][0] for k in f["expected"] if k not in miss]
        gone = [ELEMENTS[k][0] for k in f["expected"] if k in miss]
        para = [f"Spot check of {f['bed']} on {f['ward']} at {_hm(f['time'])}.",
                f"Latest falls band {f['band']}; the bundle on the care plan is the {f['ordered_bundle']} one.",
                "In place: " + (", ".join(seen) or "nothing") + ".",
                "Not in place: " + (", ".join(gone) or "nothing") + ".",
                f"Patient prefers {f['language']}." + (f" {_cap(teach_txt)}." if teach_txt else ""),
                (f"This unit missed '{ELEMENTS[f['repeat_element']][0]}' in 3 of its last 10 audits."
                 if f["repeat_element_on_unit"] else ""), f["remark"] or ""]
        return (f"AUDITOR'S NOTE ({f['auditor']}, {f['hospital']})\n" + noisy(join_sentences(para), rng, 0.06)
                + "\n\n" + bullets(list(pr.values()), rng))

    def questions(self):
        return {
            "compliance_gap": score("How large is the gap between the ordered bundle and what is in place?", [
                "None: everything required is in place.",
                "Minor: one non-critical element missing.",
                "Major: several non-critical elements, or one critical element, missing.",
                "Unsafe: two or more critical elements missing.",
            ]),
            "fix_urgency": score("How fast must the missing elements be put right?", [
                "Nothing to put right.", "By the end of the shift.", "Within the hour.", "Before leaving the room.",
            ]),
            "audit_follow_up": choice("What follow-up does this audit call for?", {
                "coach_nurse_on_shift": "Coach the nurse caring for the patient this shift.",
                "no_further_action": "Nothing further; the bundle is fully in place.",
                "report_safety_event": "Report a patient-safety event.",
                "unit_reeducation": "Schedule re-education for the whole unit.",
            }),
            "audit_passes": noul(
                "The room passes the bundle audit.",
                yes="Every element the patient's band requires is in place and documented.",
                no="At least one required element, including the teaching record, is missing."),
            "ordered_bundle_matches_band": noul("The bundle on the care plan matches the patient's latest band."),
        }


WORKFLOW = Workflow(
    name="nursing_fall_risk_intervention",
    title="Nursing fall-risk intervention",
    description="An inpatient's fall-risk band and prevention bundle under a written, fictional fall-risk index, with "
                "reassessment, post-fall care, enhanced observation and bundle audits.",
    domain="health-care operations / inpatient nursing",
    schemas=[AdmissionFallScreen(), ShiftReassessment(), PostFallHuddle(), SitterVideoRequest(),
             BundleComplianceAudit()],
    notes="All hospitals, staff, patients and the Ward Fall Risk Index are fictional; the index's items, weights and "
          "bands are invented and do not reproduce any real fall-risk tool, and none of it is clinical guidance. Age "
          "is a scored item of the index; language access, carer availability and family contact are handled as an "
          "equity-aware protocol would, without lowering care.",
)
