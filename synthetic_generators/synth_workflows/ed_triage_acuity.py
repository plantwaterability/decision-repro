"""ED triage acuity: which acuity level, stream and standing order a facility's written triage protocol gives a
patient arriving at an emergency department, and how a waiting, audited or ambulance-delivered patient is handled.

Everything here is fictional: the hospitals, the triage protocols and every threshold in them, the staff and the
encounters. Each state quotes the excerpt of its facility's written protocol that the case turns on: a five-level
acuity scale in the ESI style (immediate life-saving need, then high-risk presentation, then the count of resource
types), with danger-zone vital-sign limits, standing orders, reassessment intervals, time-to-clinician targets and
offload rules invented for this generator. The limits differ between the fictional facilities. None of it is
clinical guidance, and none of it reproduces the text of a real triage instrument.

Each schema draws a `regime` first ('clear' / 'borderline' / 'conflicting'). Borderline cases put a vital sign, a
pain score, a last-known-well time or a clock right at a written limit; conflicting cases carry a companion who
downplays a red flag, a pain score the observed behaviour does not match, a recheck that undoes the first reading,
or a hindsight outcome that tempts an auditor away from the protocol.

Age appears only where the written protocol uses it clinically: the pediatric schema's age-banded vital-sign
limits, its young-infant fever rule and its weight plausibility check. The adult schemas render no age, sex or
other protected attribute.

Every schema also defines `intended(f) -> {qid: answer}` (choice key, "yes"/"no", or score level index): the
written protocol applied to the factors. It is used only to check that answers spread across options; it is never
rendered and never stored in the factors.
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
    initials_name,
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

# =========================================================================== facilities and people
HOSP_STEMS = ["Kestrel Valley", "Ashbourne Vale", "Larkspur", "Wrenmoor", "Hollin Ridge", "Corbel Bay", "Sablecrest",
              "Tamarind Point", "Fallowmere", "Quarrystone", "Brackenford", "Eskerdale", "Rookhaven", "Mistlewood",
              "Calder Fen", "Pellmoor", "Orrin Vale", "Gannet Head", "Velloran", "Marsh Linden", "Halberd Hill",
              "Cinderford Lake", "Stillwater Bend", "Beacon Moor", "Lantern Cove", "Thistle Bank", "Oxbow Crossing",
              "Fernhallow", "Greywater", "Duncairn"]
HOSP_KINDS = ["Regional Medical Center", "General Hospital", "Community Hospital", "Memorial Hospital",
              "Medical Center", "University Hospital", "District Hospital"]


def _facility(rng):
    return f"{pick(rng, HOSP_STEMS)} {pick(rng, HOSP_KINDS)}"


def _nurse(rng):
    name = person(rng) if chance(rng, 0.6) else initials_name(rng)
    return name + pick(rng, [", RN", " RN", ""])


def _hm(t):
    return t.strftime("%H:%M")


def _cap(text):
    return text[:1].upper() + text[1:] if text else text


def _enc(rng, ts):
    return f"ED-{ts.strftime('%y')}{rng.randrange(100000, 999999)}"


AGO = ["this morning", "about two hours ago", "an hour ago", "last night", "after lunch", "around midnight",
       "45 minutes ago", "this afternoon", "early this evening", "on waking", "three hours ago", "at about 6 pm"]
SINCE = ["this morning", "last night", "lunchtime", "about 6 pm", "midnight", "waking", "yesterday evening",
         "early afternoon", "the weekend", "breakfast", "about an hour ago", "the start of the shift"]
DUR = ["two days", "three days", "a week", "36 hours", "four days", "five days", "ten days", "24 hours"]
EXERT = ["carrying shopping up the stairs", "shovelling snow", "walking the dog uphill", "mowing the lawn",
         "climbing a ladder", "rushing for a bus", "moving furniture", "cycling to work"]
HOME_MEDS = ["amlodipine", "metformin", "atorvastatin", "levothyroxine", "salbutamol inhaler", "sertraline",
             "omeprazole", "lisinopril", "bisoprolol", "montelukast", "insulin glargine", "gabapentin",
             "tamsulosin", "simvastatin", "beclometasone inhaler", "citalopram"]
ALLERGIES = ["NKDA", "NKDA", "none known", "penicillin", "sulfa drugs", "latex", "codeine", "ibuprofen", "shellfish"]


def _fill(text, rng):
    if "{ago}" in text:
        text = text.replace("{ago}", pick(rng, AGO))
    if "{since}" in text:
        text = text.replace("{since}", pick(rng, SINCE))
    if "{dur}" in text:
        text = text.replace("{dur}", pick(rng, DUR))
    if "{exert}" in text:
        text = text.replace("{exert}", pick(rng, EXERT))
    return text


def _meds(rng):
    k = pick(rng, [0, 0, 1, 1, 2, 3])
    return ", ".join(rng.sample(HOME_MEDS, k)) if k else "none"


# =========================================================================== vital signs
PROFILES = {
    "normal": {"hr": (62, 94), "rr": (12, 18), "sbp": (108, 150), "spo2": (96, 100), "t": (36.3, 37.3)},
    "pain": {"hr": (76, 104), "rr": (14, 20), "sbp": (118, 166), "spo2": (96, 100), "t": (36.4, 37.4)},
    "cardiac": {"hr": (58, 108), "rr": (14, 20), "sbp": (112, 182), "spo2": (94, 99), "t": (36.2, 37.2)},
    "infection": {"hr": (86, 124), "rr": (16, 26), "sbp": (94, 140), "spo2": (93, 99), "t": (37.7, 39.6)},
    "resp": {"hr": (86, 122), "rr": (18, 28), "sbp": (112, 160), "spo2": (90, 96), "t": (36.5, 37.8)},
    "bleed": {"hr": (90, 126), "rr": (16, 24), "sbp": (92, 128), "spo2": (95, 100), "t": (36.0, 37.0)},
    "neuro": {"hr": (58, 98), "rr": (12, 20), "sbp": (142, 196), "spo2": (94, 99), "t": (36.3, 37.2)},
}
DZ_PREF = {"normal": ["hr", "hr", "rr"], "pain": ["hr", "hr", "rr"], "cardiac": ["hr", "sbp", "spo2"],
           "infection": ["hr", "rr", "sbp", "spo2"], "resp": ["rr", "spo2", "spo2", "hr"], "bleed": ["hr", "sbp"],
           "neuro": ["hr", "spo2"]}


def _fix_dbp(v, rng):
    if v.get("sbp") is None:
        v["dbp"] = None
        return
    v["dbp"] = max(40, min(v["sbp"] - 22, int(round(v["sbp"] * rng.uniform(0.55, 0.68)))))


def _draw_vitals(rng, prof):
    r = PROFILES[prof]
    v = {"hr": rng.randint(*r["hr"]), "rr": rng.randint(*r["rr"]), "sbp": rng.randint(*r["sbp"]),
         "spo2": rng.randint(*r["spo2"]), "temp_c": round(rng.uniform(*r["t"]), 1)}
    if v["rr"] % 2 and chance(rng, 0.6):
        v["rr"] -= 1
    _fix_dbp(v, rng)
    return v


def _dz_hits(p, v):
    """Danger-zone vital signs under protocol p (missing values never count)."""
    out = []
    if v.get("hr") is not None and v["hr"] > p["hr_lim"]:
        out.append("hr")
    if v.get("rr") is not None and v["rr"] > p["rr_lim"]:
        out.append("rr")
    if v.get("spo2") is not None and v["spo2"] < p["spo2_lim"]:
        out.append("spo2")
    if v.get("sbp") is not None and v["sbp"] < p["sbp_lim"]:
        out.append("sbp")
    return out


def _settle(v, p, rng):
    """Pull every danger-zone vital comfortably inside the written limits."""
    v["hr"] = min(v["hr"], p["hr_lim"] - rng.randint(3, 18))
    v["rr"] = min(v["rr"], p["rr_lim"] - pick(rng, [2, 2, 4, 6]))
    v["spo2"] = min(100, max(v["spo2"], p["spo2_lim"] + rng.randint(1, 4)))
    if v["sbp"] < 100:
        v["sbp"] = rng.randint(100, 124)
        _fix_dbp(v, rng)


def _cross(v, p, rng, which, mode):
    """Put one danger-zone vital beyond its limit ('beyond') or right at it ('near', either side)."""
    if which == "hr":
        v["hr"] = p["hr_lim"] + rng.randint(8, 28) if mode == "beyond" else near(rng, p["hr_lim"], 5, integer=True)
    elif which == "rr":
        v["rr"] = (p["rr_lim"] + pick(rng, [2, 4, 6, 8]) if mode == "beyond"
                   else p["rr_lim"] + pick(rng, [-2, -1, 0, 0, 1, 2]))
    elif which == "spo2":
        v["spo2"] = (max(p["l1_spo2"] + 1, p["spo2_lim"] - rng.randint(2, 5)) if mode == "beyond"
                     else p["spo2_lim"] + pick(rng, [-2, -1, -1, 0, 0, 1]))
    else:
        v["sbp"] = rng.randint(81, 87) if mode == "beyond" else 90 + pick(rng, [-3, -2, -1, 0, 0, 1, 2, 4])
        _fix_dbp(v, rng)


def _just_inside(v, p, rng, which):
    """Set one vital just inside its danger-zone limit (the other half of a reading that crosses on recheck)."""
    if which == "hr":
        v["hr"] = p["hr_lim"] - rng.randint(1, 10)
    elif which == "rr":
        v["rr"] = p["rr_lim"] - pick(rng, [0, 1, 2, 2])
    elif which == "spo2":
        v["spo2"] = p["spo2_lim"] + pick(rng, [0, 1, 2])
    else:
        v["sbp"] = rng.randint(91, 104)
        _fix_dbp(v, rng)


def _cross_mild(v, p, rng, which):
    """Cross one danger-zone limit by a modest margin."""
    if which == "hr":
        v["hr"] = p["hr_lim"] + rng.randint(3, 14)
    elif which == "rr":
        v["rr"] = p["rr_lim"] + pick(rng, [2, 2, 4])
    elif which == "spo2":
        v["spo2"] = max(p["l1_spo2"] + 1, p["spo2_lim"] - rng.randint(1, 3))
    else:
        v["sbp"] = rng.randint(82, 88)
        _fix_dbp(v, rng)


def _jitter(v, p, rng):
    """Small measurement-to-measurement drift that never moves a vital across its danger-zone limit."""
    before = set(_dz_hits(p, v))
    old = dict(v)
    for k, d in (("hr", 4), ("rr", 2), ("spo2", 1), ("sbp", 6)):
        if v.get(k) is not None:
            v[k] = v[k] + rng.randint(-d, d)
    v["spo2"] = min(100, v["spo2"]) if v.get("spo2") is not None else None
    if v.get("temp_c") is not None:
        v["temp_c"] = round(v["temp_c"] + rng.uniform(-0.2, 0.2), 1)
    after = set(_dz_hits(p, v))
    for k in before ^ after:
        v[k] = old[k]
    if v.get("sbp") is not None:
        _fix_dbp(v, rng)


def _c_to_f(c):
    return round(c * 9 / 5 + 32, 1)


def _temp_txt(c, unit):
    if c is None:
        return None
    return f"{_c_to_f(c):.1f} °F" if unit == "F" else f"{c:.1f} °C"


def _vs_dict(v, unit):
    d = {"hr": v.get("hr"), "rr": v.get("rr"), "spo2_pct": v.get("spo2"),
         "bp": f"{v['sbp']}/{v['dbp']}" if v.get("sbp") is not None else None}
    if v.get("temp_c") is not None:
        if unit == "F":
            d["temp_f"] = _c_to_f(v["temp_c"])
        else:
            d["temp_c"] = v["temp_c"]
    return d


def _vs_line(v, unit, rng, style=None):
    style = style or pick(rng, ["compact", "comma", "pipe"])
    miss = pick(rng, ["--", "not obtained", "n/r"])
    bp = f"{v['sbp']}/{v['dbp']}" if v.get("sbp") is not None else miss
    hr = v["hr"] if v.get("hr") is not None else miss
    rr = v["rr"] if v.get("rr") is not None else miss
    sp = f"{v['spo2']}%" if v.get("spo2") is not None else miss
    t = _temp_txt(v.get("temp_c"), unit) or miss
    if style == "compact":
        return f"BP {bp}  HR {hr}  RR {rr}  SpO2 {sp} RA  T {t}"
    if style == "comma":
        return f"HR {hr}, BP {bp}, RR {rr}, SpO2 {sp} on room air, temp {t}"
    return f"P {hr} | BP {bp} | R {rr} | Sat {sp} | T {t}"


# =========================================================================== the adult protocol (fictional)
def _adult_protocol(rng):
    return {"rev": f"{rng.randint(2, 9)}.{rng.randint(0, 4)}", "hr_lim": pick(rng, [100, 105, 110]),
            "rr_lim": pick(rng, [20, 22, 24]), "spo2_lim": pick(rng, [92, 93, 94]), "sbp_lim": 90,
            "l1_spo2": pick(rng, [85, 86, 88]), "l2_pain": pick(rng, [7, 8]), "sep_hr": pick(rng, [90, 100]),
            "sep_rr": pick(rng, [20, 22]), "sep_t_hi": pick(rng, [38.0, 38.3]), "sep_t_lo": 36.0,
            "stroke_h": pick(rng, [4.5, 6, 8]), "ft_open": pick(rng, [8, 9, 10, 11]),
            "ft_close": pick(rng, [20, 21, 22, 23]), "unit": pick_w(rng, {"C": 3, "F": 1}),
            "older_age": pick(rng, [65, 70, 75])}


def _protocol_title(facility, p, rng, voice="walkin"):
    return pick(rng, {
        "walkin": [f"{facility} ED Triage Acuity Standard, rev {p['rev']} (excerpt)",
                   f"{facility}: emergency department triage protocol v{p['rev']}, relevant sections"],
        "retriage": [f"Waiting-room rules -- {facility} triage std {p['rev']}",
                     f"Reassessment card ({facility}, protocol {p['rev']})"],
        "audit": [f"Reference: triage standard {p['rev']} in force at {facility} on the date of attendance",
                  f"Audit criteria taken from {facility} protocol revision {p['rev']}"],
        "offload": [f"Ambulance arrivals key, {facility} ({p['rev']})",
                    f"{facility} offload and triage card {p['rev']}"],
    }[voice])


def _adult_rules(p, rng, keys, voice="walkin"):
    """The protocol lines a case needs (rule name -> text), in the house style of the schema (`voice`). Voices never
    share long runs of words, so a copied state of one schema cannot look like a copy of another."""
    t = lambda c: _temp_txt(c, p["unit"]).replace(" ", "")  # noqa: E731
    l1, l2, hr, rr, sp, G = p["l1_spo2"], p["l2_pain"], p["hr_lim"], p["rr_lim"], p["spo2_lim"], p["older_age"]
    texts = {
        "walkin": {
            "level_1": [f"Level 1: unresponsive or pain-only; airway/breathing support now; SpO2 < {l1}%; SBP < 80.",
                        f"Level 1 = immediate life-saving need: no response or pain-only, airway or ventilation "
                        f"support, SpO2 below {l1}%, systolic below 80."],
            "level_2": [f"Level 2: high-risk presentation; new confusion or lethargy; pain >= {l2}/10 with distress "
                        f"seen at triage.",
                        f"Level 2 = high-risk situation, new confusion/lethargy, or pain {l2}/10 or more plus observed "
                        f"distress."],
            "resources": ["Else count resource types (labs, ECG, X-ray, CT/ultrasound/MRI, IV or nebulised drugs, IV "
                          "fluids, consult, procedure; not exam, tablets, splints, dressings): 2+ = Level 3, 1 = 4, "
                          "0 = 5."],
            "danger_zone": [f"Danger zone (latest set): HR > {hr}, RR > {rr}, SpO2 < {sp}% or SBP < 90 turns a would-be "
                            f"Level 3 into Level 2."],
            "age_pregnancy": [f"Age {G}+ with abdominal pain, a fall, head injury, back pain, dizziness or a faint: "
                              f"Level 3 at least (before the danger zone). Known/possible pregnancy with "
                              f"abdominal/flank pain or after a fall or blow: Level 2."],
            "equity": ["Payer, immigration status, race, religion and housing never change the level. Triage "
                       "questions go through a qualified interpreter, never relatives."],
        },
        "retriage": {
            "level_1": [f"L1 = pain-only or no response, airway/breathing help, sat under {l1}%, systolic under 80."],
            "level_2": [f"L2 = high-risk complaint, new confusion or drowsiness, or pain {l2}+/10 with visible "
                        f"distress."],
            "resources": ["L3 / L4 / L5 = two-plus / one / zero resource types (bloods, ECG, plain film, scans, "
                          "parenteral drugs or fluids, consult, procedure)."],
            "danger_zone": [f"Would-be L3 with pulse over {hr}, resp over {rr}, sat under {sp}% or systolic under 90 "
                            f"moves up to L2."],
            "age_pregnancy": [f"{G} and over with abdominal pain, fall, head injury, back pain, dizziness or faint: "
                              f"never below L3 (before the DZ step). Pregnant or maybe pregnant plus belly/flank pain or "
                              f"a fall: L2."],
            "equity": ["Re-triage on clinical findings only; cost worries, coverage or housing never lower the "
                       "level. Reassessment questions use a qualified interpreter."],
        },
        "audit": {
            "level_1": [f"Level 1 applied where a life-saving step was needed at once (pain-only or no response, "
                        f"airway support, saturation under {l1}%, systolic under 80)."],
            "level_2": [f"Level 2 applied to high-risk presentations, new confusion or lethargy, and pain of {l2}+ "
                        f"with recorded distress."],
            "resources": ["Below that, the count of resource types decided it (bloods, ECG, radiographs, scans, "
                          "drips or injections, specialist input, procedures): two or more gave 3, one gave 4, none 5."],
            "danger_zone": [f"Heart rate above {hr}, breathing rate above {rr}, saturation below {sp}% or systolic "
                            f"below 90 lifted a would-be 3 to 2."],
            "age_pregnancy": [f"From age {G}, a fall, head or back injury, abdominal pain, dizziness or fainting "
                              f"meant Level 3 at least (decided before the danger-zone step); a known or possible "
                              f"pregnancy with abdominal pain or trauma meant Level 2."],
            "equity": ["Use of a qualified interpreter must be recorded whenever English was not the preferred "
                       "language (a missing element otherwise). Insurance, ethnicity, faith and housing are never "
                       "grounds for a level."],
        },
        "offload": {
            "level_1": [f"[1] resus: unresponsive or pain-only, needs airway or ventilation help, SpO2 under {l1}%, "
                        f"SBP under 80"],
            "level_2": [f"[2] emergent: high-risk story, new confusion or drowsiness, pain {l2}/10 or worse plus "
                        f"distress"],
            "resources": ["[3-5] resource types expected: 2 or more -> 3, exactly 1 -> 4, none -> 5 (bloods, ECG, "
                          "imaging, IV/IM/neb meds, IV fluid, consult, procedure)"],
            "danger_zone": [f"[DZ] HR above {hr}, RR above {rr}, SpO2 below {sp}%, SBP below 90 on ED arrival vitals: "
                            f"a would-be 3 becomes 2"],
            "age_pregnancy": [f"[AGE] {G}+ and a fall, a blow to the head, abdo or back pain, a dizzy spell or a "
                              f"faint: 3 or higher, before DZ. [PREG] known/possible pregnancy plus abdo/flank pain or "
                              f"trauma: 2"],
            "equity": ["[ACCESS] interpreter for handover questions when needed; coverage, ethnicity and housing "
                       "never move the level"],
        },
    }[voice]
    texts["standing_orders"] = [
        f"Standing orders (first that applies): stroke page if focal deficit and LKW <= {p['stroke_h']:g} h or "
        f"unknown; sepsis labs if suspected infection plus 2+ of HR > {p['sep_hr']}, RR > {p['sep_rr']}, T >= "
        f"{t(p['sep_t_hi'])} or < {t(p['sep_t_lo'])}, new confusion (1 from age {G}); ECG in 10 min for chest pain not "
        f"from chest-wall injury, palpitations, syncope, or breathlessness with heart disease; X-ray for limb injury "
        f"with bony tenderness or loss of use."]
    texts["fast_track"] = [f"Fast track {p['ft_open']:02d}:00-{p['ft_close']:02d}:00, ambulatory Level 4-5 only."]
    return {k: pick(rng, texts[k]) for k in keys}


# the registration block in each schema's house style: values are phrased per voice so no two schemas share runs
_PAYER_TXT = {
    "retriage": {"commercial plan": "private cover", "public plan": "public cover", "older-adult public plan":
                 "older-adult public cover", "uninsured": "no cover", "emergency coverage only": "emergency-only cover"},
    "audit": {"commercial plan": "commercially insured", "public plan": "publicly insured", "older-adult public plan":
              "insured via the older-adult programme", "uninsured": "uninsured at the time",
              "emergency coverage only": "on the emergency coverage pathway"},
    "offload": {"commercial plan": "private insurer", "public plan": "state plan", "older-adult public plan":
                "seniors' plan", "uninsured": "no insurance", "emergency coverage only": "emergency-only coverage"},
}
_HOUSING_TXT = {
    "retriage": {"stable housing": "housed", "unstably housed (staying with friends)": "sofa-surfing",
                 "experiencing homelessness": "no fixed abode"},
    "audit": {"stable housing": "stable address", "unstably housed (staying with friends)": "insecure housing",
              "experiencing homelessness": "homeless at the time of the visit"},
    "offload": {"stable housing": "lives at home", "unstably housed (staying with friends)": "between addresses",
                "experiencing homelessness": "sleeping rough"},
}
_INTERP_TXT = {
    "retriage": {"phone_interpreter": "phone interp", "video_interpreter": "video interp",
                 "family_member": "relative interpreting", "none": "no interp"},
    "audit": {"phone_interpreter": "telephone interpreter used", "video_interpreter": "video interpreter used",
              "family_member": "a relative interpreted", "none": "no interpreter recorded"},
    "offload": {"phone_interpreter": "via the phone line", "video_interpreter": "via video interpreter",
                "family_member": "family translating", "none": "no interpreter"},
}
_RACE_LABEL = {"walkin": "race/ethnicity (recorded)", "retriage": "R/E", "audit": "documented race/ethnicity",
               "offload": "ethnicity per registration"}


# per-schema shorthand for the shared presentation narratives (nurse / auditor / paramedic house styles)
_VOICE_SUBS = {
    "retriage": {"the": "", "a": "", "and": "+", "with": "w/", "left": "L", "right": "R", "about": "~",
                 "minutes": "min", "for": "x", "hours": "h", "patient": "pt", "since": "since", "now": "now",
                 "while": "whilst", "no": "nil", "not": "not"},
    "audit": {"the": "", "a": "", "since": "from", "for": "over", "after": "following",
              "is": "was", "has": "had", "now": "by then", "feels": "felt", "says": "said", "cannot": "could not",
              "to": "to", "of": "of"},
    "offload": {"the": "", "a": "", "and": "&", "with": "c/", "left": "lt", "right": "rt",
                "about": "approx.", "minutes": "mins", "at": "@", "now": "currently", "after": "post", "no": "nil",
                "while": "during", "has": "has", "into": "into", "very": "v.", "not": "not",
                "but": "but", "then": "then", "back": "back"},
}


def _voice_text(text, voice):
    subs = _VOICE_SUBS.get(voice)
    if not subs:
        return text
    out = []
    for w in text.split(" "):
        core = w.strip(",;.:'()")
        if core.lower() in subs:
            rep = subs[core.lower()]
            if not rep:
                continue
            w = w.replace(core, rep)
        out.append(w)
    return " ".join(out)


def _l1(p, v, mental, airway):
    return bool(mental == "pain_only" or airway
                or (v.get("spo2") is not None and v["spo2"] < p["l1_spo2"])
                or (v.get("sbp") is not None and v["sbp"] < 80))


def _l2_core(p, mental, high_risk, pain, distress):
    return bool(high_risk or mental in ("new_confusion", "lethargic")
                or (pain is not None and pain >= p["l2_pain"] and distress))


def _adult_level(p, v, mental, airway, high_risk, pain, distress, n_res, pt=None):
    """The written adult protocol: Level 1-5 (1 = most acute). `pt` carries the clinical demographics: a known or
    possible pregnancy with an abdominal or trauma presentation is Level 2; an older patient with a fall, abdominal,
    head, back or dizziness/faint presentation is never below Level 3 (applied before the danger-zone check)."""
    if _l1(p, v, mental, airway):
        return 1
    if _l2_core(p, mental, high_risk, pain, distress):
        return 2
    if pt and pt["preg_cat"] and pt["pregnancy"] in ("possible", "confirmed"):
        return 2
    base = 3 if n_res >= 2 else (4 if n_res == 1 else 5)
    if pt and pt["geri_cat"] and pt["age"] >= p["older_age"] and base > 3:
        base = 3
    if base == 3 and _dz_hits(p, v):
        return 2
    return base


def _sepsis_count(p, v, mental):
    n = 0
    n += v.get("hr") is not None and v["hr"] > p["sep_hr"]
    n += v.get("rr") is not None and v["rr"] > p["sep_rr"]
    tc = v.get("temp_c")
    n += tc is not None and (tc >= p["sep_t_hi"] or tc < p["sep_t_lo"])
    n += mental in ("new_confusion", "lethargic")
    return int(n)


def _ft_open(p, ts):
    m = ts.hour * 60 + ts.minute
    return p["ft_open"] * 60 <= m < p["ft_close"] * 60


# =========================================================================== adult presentations
def V(text, res=(), high_risk=None, pain=(0, 3), prof="normal", ecg=False, deficit=None, crit=None, mental=None,
      ambulatory=True):
    return {"text": text, "res": list(res), "high_risk": high_risk, "pain": list(pain), "prof": prof, "ecg": ecg,
            "deficit": deficit, "crit": crit, "mental": mental, "ambulatory": ambulatory}


ADULT_CX = {
    "cardiac": [
        (["chest pressure", "central chest tightness", "chest heaviness"], [
            V("came on {ago} while {exert}; spreads to the jaw and left arm, clammy at the desk",
              ("labs", "ecg", "xray", "iv_meds"), "exertional chest pressure with radiation and sweating", (4, 8),
              "cardiac", ecg=True),
            V("intermittent pressure in the chest since {since}, worse walking uphill, eases with rest",
              ("labs", "ecg", "xray"), None, (2, 5), "cardiac", ecg=True),
            V("dull ache behind the breastbone since {since}, came on at rest, no radiation",
              ("labs", "ecg"), None, (3, 6), "cardiac", ecg=True)]),
        (["sharp chest pain", "chest pain on breathing in"], [
            V("a week of cough, now sharp right-sided pain on deep breaths since {since}", ("labs", "ecg", "xray"),
              None, (3, 6), "normal", ecg=True),
            V("stabbing pain at the side of the chest with each breath since {since}, no pain on exertion",
              ("ecg", "xray"), None, (3, 6), "normal", ecg=True)]),
        (["palpitations", "heart racing", "fluttering in the chest"], [
            V("fluttering on and off since {since}, no fainting, no chest pain", ("labs", "ecg"), None, (0, 2),
              "cardiac", ecg=True),
            V("heart pounding at rest {ago}, lasted about 20 minutes each time, feels normal now", ("ecg",), None,
              (0, 1), "normal", ecg=True)]),
        (["fainted", "collapse with brief loss of consciousness", "syncope"], [
            V("passed out while climbing stairs at work; a coworker says out for under a minute",
              ("labs", "ecg"), "fainting during exertion", (0, 2), "cardiac", ecg=True),
            V("collapsed while jogging {ago} with no warning, fully awake within a minute",
              ("labs", "ecg", "advanced_imaging"), "fainting during exertion", (0, 2), "cardiac", ecg=True),
            V("felt lightheaded and slid to the floor while having blood taken at a clinic, back to normal within "
              "seconds", ("ecg",), None, (0, 1), "normal", ecg=True),
            V("fainted standing in a hot queue after skipping meals all day, feels fine now", ("ecg",), None, (0, 1),
              "normal", ecg=True)]),
        (["upper abdominal discomfort", "indigestion-type pain"], [
            V("burning under the breastbone since {since} not settled by antacids; diabetic with high cholesterol",
              ("labs", "ecg"), None, (3, 6), "cardiac", ecg=True),
            V("pressure in the upper abdomen since {since} with breathlessness on walking; has heart stents",
              ("labs", "ecg", "xray"), "possible cardiac pain with known coronary disease", (3, 6), "cardiac",
              ecg=True)]),
        (["short of breath", "breathless with swollen ankles"], [
            V("known heart failure; sleeping on three pillows for {dur}, ankles more swollen than usual",
              ("labs", "ecg", "xray", "iv_meds"), None, (0, 2), "resp", ecg=True),
            V("breathless on exertion for {dur}, weight up 3 kg, ran out of water tablets", ("labs", "ecg", "xray"),
              None, (0, 2), "resp", ecg=True)]),
        (["chest wall pain", "rib pain"], [
            V("sore over the left ribs since lifting a sofa yesterday; pain reproduced by pressing on the spot", (),
              None, (2, 5)),
            V("fell against a table edge {ago}; pain only on twisting, tender over one rib, breathing comfortably",
              ("xray",), None, (3, 6))]),
    ],
    "infection": [
        (["fever and flank pain", "burning urination with fever"], [
            V("chills and fever since {since}, burning on passing urine, pain in the right flank", ("labs", "iv_meds"),
              None, (3, 6), "infection"),
            V("urinary symptoms for {dur}, now shivering with back pain and vomited once",
              ("labs", "iv_meds", "advanced_imaging"), None, (3, 7), "infection")]),
        (["cough and fever", "productive cough"], [
            V("green sputum and fevers for {dur}, breathless climbing stairs", ("labs", "xray"), None, (1, 4),
              "infection"),
            V("cough for {dur}, fevers overnight, aching all over, eating little", ("labs", "xray", "iv_meds"), None,
              (1, 4), "infection"),
            V("dry cough and a temperature for {dur}, managing at home, wants a check", (), None, (0, 3),
              "infection")]),
        (["leg redness", "red patch on the leg"], [
            V("red, hot, swollen lower leg; redness has spread past the pen line drawn at the clinic this morning",
              ("labs", "iv_meds"), None, (3, 6), "infection"),
            V("small red patch on the shin {dur} ago, now palm-sized and warm, walking fine", (), None, (2, 4),
              "infection")]),
        (["fever during chemotherapy", "fever, on cancer treatment"], [
            V("last chemotherapy 9 days ago, fever at home this evening, shivering", ("labs", "iv_meds", "xray"),
              "fever during chemotherapy", (0, 3), "infection"),
            V("on chemotherapy, fever since {since}; the oncology helpline told them to come straight in",
              ("labs", "iv_meds"), "fever during chemotherapy", (0, 3), "infection")]),
        (["sore throat", "throat pain and fever"], [
            V("sore throat for {dur}, swallowing fluids fine, voice normal", (), None, (2, 5), "infection"),
            V("fever and sore throat for {dur}, one-sided swelling at the back of the throat, voice muffled",
              ("labs", "iv_meds", "consult"), None, (5, 8), "infection")]),
        (["painful urination", "urinary frequency"], [
            V("burning and frequency for {dur}, no back pain", ("labs",), None, (1, 3), "infection")]),
        (["painful lump", "abscess"], [
            V("tender fluctuant swelling on the inner thigh for {dur}, getting bigger", ("procedure",), None, (3, 6),
              "infection"),
            V("painful swollen lump in the armpit with a head on it, feverish last night", ("procedure", "labs"),
              None, (4, 7), "infection")]),
        (["vomiting / diarrhoea", "gastro symptoms"], [
            V("vomiting and loose stools since {since}, cannot keep fluids down, dizzy on standing",
              ("labs", "iv_meds"), None, (2, 5), "infection"),
            V("fever and vomiting five times since {since}, passed very little urine today", ("labs", "iv_meds"),
              None, (2, 5), "infection"),
            V("loose stools since {since} after a takeaway, keeping fluids down, no blood", (), None, (1, 3),
              "infection")]),
        (["infected bite", "redness around a bite"], [
            V("dog bite to the hand two days ago, now swollen and red with streaks up the forearm",
              ("labs", "iv_meds", "xray"), None, (3, 6), "infection")]),
        (["facial swelling", "toothache with swelling"], [
            V("swelling of the jaw from a bad tooth for {dur}, fever last night, can open the mouth normally",
              ("labs", "iv_meds"), None, (4, 7), "infection")]),
    ],
    "neuro": [
        (["arm weakness", "weak arm and slurred speech"], [
            V("sudden weakness of the right arm and slurred speech, noticed by a coworker",
              ("labs", "advanced_imaging", "consult"), "new one-sided weakness", (0, 2), "neuro", deficit=True),
            V("dropped a mug, cannot lift the left arm properly, speech thick", ("labs", "advanced_imaging", "consult"),
              "new one-sided weakness", (0, 2), "neuro", deficit=True)]),
        (["facial droop", "face drooping on one side"], [
            V("the left side of the face is drooping and drooling, words muddled",
              ("labs", "advanced_imaging", "consult"), "new facial droop with speech change", (0, 2), "neuro",
              deficit=True)]),
        (["sudden vision loss", "sudden visual disturbance"], [
            V("sudden loss of vision in the right eye, 'like a curtain coming down'", ("advanced_imaging", "consult"),
              "sudden loss of vision", (0, 2), "neuro", deficit=True),
            V("abrupt double vision with veering to one side when walking", ("labs", "advanced_imaging", "consult"),
              "sudden double vision with unsteadiness", (0, 2), "neuro", deficit=True)]),
        (["transient neurological symptoms", "episode of speech or hand trouble, resolved"], [
            V("could not find words for about 15 minutes, completely back to normal before arrival",
              ("labs", "advanced_imaging", "ecg"), "transient neurological symptoms", (0, 1), "neuro", deficit=False),
            V("right hand went numb and clumsy for 20 minutes, resolved; neuro check at triage normal",
              ("labs", "advanced_imaging", "ecg"), "transient neurological symptoms", (0, 1), "neuro",
              deficit=False)]),
        (["leg weakness", "unsteady walking"], [
            V("sudden clumsiness of the left leg, dragging the foot", ("labs", "advanced_imaging", "consult"),
              "new one-sided weakness", (0, 2), "neuro", deficit=True, ambulatory=False)]),
    ],
    "other": [
        (["hand laceration", "cut to the palm", "finger laceration"], [
            V("gaping 3 cm cut to the palm from a kitchen knife, bleeding controlled, fingers move and feel normal",
              ("procedure",), None, (3, 6)),
            V("cut the base of the thumb on a broken glass, edges gape, worried glass is still in it",
              ("procedure", "xray"), None, (3, 6)),
            V("small superficial cut to a fingertip, edges together, bleeding stopped", (), None, (1, 3))]),
        (["head injury", "scalp laceration"], [
            V("tripped on the stairs and hit the back of the head, 2 cm scalp cut; takes a blood thinner for an "
              "irregular heartbeat", ("advanced_imaging", "procedure"), "head strike while on a blood thinner", (2, 5)),
            V("hit the head on a cupboard door, small scalp cut, no blackout, not on blood thinners", ("procedure",),
              None, (2, 4))]),
        (["abdominal pain", "tummy pain"], [
            V("pain started around the navel last night and moved to the lower right abdomen, off food, hurts to "
              "walk", ("labs", "advanced_imaging", "iv_meds"), None, (5, 8), "pain"),
            V("crampy abdominal pain with loose stools for two days, drinking normally, no blood", (), None, (2, 5)),
            V("lower abdominal cramps since {since}, vomited twice, tolerating sips", ("labs",), None, (3, 5))]),
        (["headache"], [
            V("headache like the usual migraines, own medication has not helped, light hurts the eyes",
              ("iv_meds",), None, (5, 8), "pain"),
            V("mild headache for {dur} after long shifts, no other symptoms", (), None, (2, 4)),
            V("sudden severe headache that peaked within a minute while lifting weights, 'worst ever'",
              ("labs", "advanced_imaging", "iv_meds"), "sudden worst-ever headache", (8, 10), "pain")]),
        (["back pain", "low back pain"], [
            V("lower back pain after lifting boxes yesterday, no numbness, walking normally", (), None, (4, 7)),
            V("back pain with new numbness around the groin and difficulty passing urine since this morning",
              ("advanced_imaging", "consult", "procedure"), "back pain with new saddle numbness and urinary retention",
              (5, 8), "pain")]),
        (["medication refill", "ran out of medication"], [
            V("ran out of blood-pressure tablets three days ago, no symptoms, wants a prescription", (), None, (0, 0)),
            V("out of the preventer inhaler, mild cough, not breathless", (), None, (0, 0))]),
        (["rash", "itchy rash"], [
            V("itchy rash on both forearms after a new detergent, lips and tongue normal", (), None, (0, 2)),
            V("hives across the trunk after starting a new antibiotic, no swelling, breathing fine", (), None,
              (0, 2))]),
        (["allergic reaction", "lip swelling"], [
            V("hives after eating at a restaurant, lips swelling, voice slightly hoarse", ("iv_meds",),
              "lip swelling and voice change after an exposure", (0, 2), "resp")]),
        (["eye complaint", "eye problem"], [
            V("something flew into the eye while grinding metal, watering and gritty", ("procedure",), None, (3, 6)),
            V("red itchy eye for {dur}, vision normal", (), None, (0, 2))]),
        (["toothache", "dental pain"], [
            V("toothache for {dur}, no facial swelling, eating normally", (), None, (4, 7))]),
        (["ear pain", "earache"], [
            V("ear pain for {dur}, hearing a little muffled, no discharge", (), None, (2, 5))]),
        (["nosebleed"], [
            V("nosebleed for 40 minutes not stopping with pressure; takes a blood thinner", ("labs", "procedure"),
              None, (0, 2), "bleed"),
            V("nosebleed earlier, stopped with pressure, wants it checked", (), None, (0, 1))]),
        (["vomiting blood", "black stools"], [
            V("vomited dark material and has had black stools since {since}, lightheaded on standing",
              ("labs", "iv_meds", "consult"), "gastrointestinal bleeding with lightheadedness", (1, 4), "bleed")]),
        (["asthma flare", "wheezing"], [
            V("used the reliever inhaler four times since {since}, wheezy, speaking in full sentences", ("iv_meds",),
              None, (0, 2), "resp"),
            V("asthma flare, speaking in short phrases, reliever not helping", ("iv_meds", "xray", "labs"), None,
              (0, 2), "resp")]),
        (["sudden breathlessness"], [
            V("sudden breathlessness and sharp chest pain two days after a long-haul flight, one calf swollen",
              ("labs", "ecg", "advanced_imaging"), "sudden breathlessness after long travel with a swollen calf",
              (3, 6), "resp", ecg=True)]),
        (["anxiety", "panic symptoms"], [
            V("feels panicky with tingling fingers, similar episodes before, no chest pain", (), None, (0, 1))]),
        (["thoughts of self-harm"], [
            V("tells the triage nurse about thoughts of ending their life and a specific plan", ("consult",),
              "thoughts of self-harm with a plan", (0, 0))]),
        (["flank pain", "kidney stone pain"], [
            V("severe colicky left flank pain coming in waves, history of kidney stones, vomited once",
              ("labs", "iv_meds", "advanced_imaging"), None, (7, 10), "pain")]),
        (["burn"], [
            V("splashed hot cooking oil on the forearm, blistering patch about palm-sized", (), None, (4, 7)),
            V("kettle scald to the lower leg, blisters bigger than two palms", ("iv_meds", "consult"), None, (6, 9),
              "pain")]),
        (["insect sting"], [
            V("bee sting on the hand, local swelling only, no breathing problems", (), None, (1, 3))]),
        (["dizziness", "vertigo"], [
            V("room-spinning dizziness on turning in bed since this morning, walking steadily between episodes",
              ("iv_meds",), None, (0, 1))]),
        (["high blood sugar"], [
            V("home meter reads 'HI', very thirsty, passing a lot of urine for {dur}", ("labs", "iv_meds"), None,
              (0, 2))]),
        (["something stuck in the throat"], [
            V("fish bone feels stuck in the throat since dinner, swallowing saliva fine", ("xray", "consult"), None,
              (2, 4))]),
    ],
    "critical": [
        (["unresponsive after tablets", "possible overdose"], [
            V("brought in by a friend after taking an unknown number of tablets, now very hard to rouse",
              ("labs", "ecg", "iv_meds"), "possible overdose", (0, 0), "normal", crit="mental", ambulatory=False)]),
        (["severe breathlessness"], [
            V("too breathless to finish a word, lips dusky, reliever inhaler not helping", ("labs", "iv_meds", "xray"),
              "severe breathlessness, speaking in single words", (0, 2), "resp", crit="spo2", ambulatory=False)]),
        (["throat swelling after an allergen"], [
            V("throat feels like it is closing after eating peanuts, noisy breathing in, drooling", ("iv_meds",),
              "airway swelling after an allergen", (0, 2), "resp", crit="airway")]),
        (["collapse", "vomiting blood with collapse"], [
            V("vomited a large amount of blood in the car park and collapsed, pale and sweaty",
              ("labs", "iv_meds", "consult"), "major bleeding with collapse", (0, 3), "bleed", crit="sbp",
              ambulatory=False)]),
    ],
}

LIMBS = [
    ("ankle", ["twisted the ankle stepping off a kerb", "rolled the ankle playing basketball",
               "went over on the ankle on a hiking trail", "slipped on wet stairs and turned the ankle"],
     "the outer ankle bone", True),
    ("foot", ["dropped a box of tiles on the foot", "kicked a door frame barefoot",
              "landed awkwardly on the foot jumping off a wall"], "the base of the little-toe bone", True),
    ("wrist", ["fell onto an outstretched hand", "slipped on ice and landed on the wrist",
               "came off a scooter and put a hand out"], "the bone at the base of the thumb", False),
    ("knee", ["twisted the knee during a football match", "knelt hard on a concrete step",
              "knee buckled stepping off a ladder"], "the kneecap", True),
    ("finger", ["jammed a finger catching a ball", "caught a finger in a car door",
                "bent a finger back playing volleyball"], "the middle bone of the finger", False),
    ("shoulder", ["fell from a stepladder onto the shoulder", "fell off a bicycle onto the shoulder",
                  "was tackled onto the shoulder at rugby"], "the collarbone", False),
    ("elbow", ["fell on the elbow on a wet floor", "fell off a skateboard onto the elbow"], "the point of the elbow",
     False),
    ("hand", ["punched a wall", "had the hand caught between two pallets at work", "fell onto a closed fist"],
     "the knuckle of the little finger", False),
    ("toe", ["stubbed the big toe on a bed frame", "dropped a dumbbell on the toes"], "the big toe", True),
    ("lower leg", ["was kicked in the shin during a match", "struck the shin on a trailer hitch"], "the shin bone",
     True),
    ("thumb", ["bent the thumb back skiing", "caught the thumb in a dog lead"], "the base of the thumb", False),
]


def _draw_limb(rng, reg):
    site, mechs, bone, lower = pick(rng, LIMBS)
    if reg == "clear":
        tender = chance(rng, 0.5)
        can_use = (not tender) or chance(rng, 0.45)
    else:
        tender = chance(rng, 0.45)
        can_use = chance(rng, 0.62)
    deformity = tender and not can_use and chance(rng, 0.3)
    parts = [f"{pick(rng, mechs)} {pick(rng, AGO)}."]
    if reg == "borderline":
        parts.append(pick(rng, [f"Mildly tender right on {bone}.", f"Tender when {bone} itself is pressed, "
                                f"though less than the soft tissue."]) if tender else
                     pick(rng, [f"Tender in the soft tissue just next to {bone}, not on the bone itself.",
                                f"Sore around the joint; pressing on {bone} is not painful."]))
    else:
        parts.append(pick(rng, [f"Point tenderness directly over {bone}.", f"Very sore when {bone} is pressed."])
                     if tender else
                     pick(rng, [f"No tenderness over {bone}; soreness is in the soft tissue.",
                                f"Pressing on {bone} is not painful; diffuse mild swelling."]))
    if lower:
        if reg == "borderline":
            parts.append("Managed four limping steps at the desk." if can_use else
                         "Took two steps, then would not put the foot down again.")
        else:
            parts.append(pick(rng, ["Walking with a limp.", "Walked in unaided."]) if can_use else
                         pick(rng, ["Unable to put any weight on it.", "Cannot take a single step on it."]))
    else:
        parts.append(pick(rng, ["Full grip, moves all fingers.", "Can lift and use the arm."]) if can_use else
                     pick(rng, ["Cannot grip or use the hand.", "Holding the arm still, will not move it."]))
    if deformity:
        parts.append(pick(rng, ["Visibly deformed.", "Obvious angulation."]))
    res = (["xray"] if (tender or not can_use) else []) + (["procedure", "iv_meds"] if deformity else [])
    pain = [7, 10] if deformity else ([4, 8] if tender else [2, 6])
    return {"family": "limb", "cc": pick(rng, [f"{site} injury", f"{site} pain", f"injured {site}"]),
            "detail": " ".join(parts), "resources": res, "high_risk": None, "pain_range": pain, "prof": "pain",
            "ecg_rule": False, "deficit_present": None, "crit": None, "mental0": None,
            "ambulatory0": can_use or not lower, "limb_site": site, "bony_tenderness": tender, "can_use_limb": can_use,
            "deformity": deformity}


GERI_CC = {"abdominal pain", "tummy pain", "head injury", "scalp laceration", "back pain", "low back pain",
           "dizziness", "vertigo", "fainted", "collapse with brief loss of consciousness", "syncope"}
PREG_CC = {"abdominal pain", "tummy pain", "flank pain", "kidney stone pain", "head injury", "scalp laceration"}
RACE_ETH = ["Asian", "Black or African American", "Hispanic or Latino", "White", "American Indian or Alaska Native",
            "Native Hawaiian or Pacific Islander", "Middle Eastern or North African", "Multiracial",
            "declined to state"]
LANGUAGES = ["Spanish", "Vietnamese", "Arabic", "Polish", "Haitian Creole", "Somali", "Mandarin", "Tagalog",
             "Portuguese", "Russian", "Korean", "Amharic", "Punjabi", "Ukrainian", "Dari", "French", "Cantonese"]
PAYERS = {"commercial plan": 3.5, "public plan": 2.5, "older-adult public plan": 2, "uninsured": 1.4,
          "emergency coverage only": 0.6}
HOUSING = {"stable housing": 8, "unstably housed (staying with friends)": 1.2, "experiencing homelessness": 1}
INTERP_TXT = {"phone_interpreter": "qualified phone interpreter", "video_interpreter": "qualified video interpreter",
              "family_member": "family member interpreting", "none": "no interpreter used"}
EQUITY_NOTES = ["Registration note: no insurance on file; suggested the community clinic might be cheaper.",
                "Registration note: patient asked what the visit costs and whether immigration will be told.",
                "Security note: 'regular from the shelter, probably just wants somewhere warm'.",
                "Registration note: card declined; told the patient to settle the bill before being seen."]


def _patient(rng, cx, reg, p):
    """Clinical demographics plus the social context a registration record carries."""
    fall = cx["family"] == "limb" and any(w in cx["detail"] for w in ("fell", "slipped", "fall", "landed", "went over"))
    geri_cat = cx["cc"] in GERI_CC or fall
    preg_cat = cx["cc"] in PREG_CC or fall
    if geri_cat and reg == "borderline" and chance(rng, 0.5):
        age = near(rng, p["older_age"], 3, integer=True)
    else:
        age = pick_w(rng, {rng.randint(18, 39): 3.5, rng.randint(40, 64): 3.5, rng.randint(65, 94): 3})
    sex = pick(rng, ["F", "M"])
    pregnancy = None
    if sex == "F" and 15 <= age <= 50:
        pregnancy = pick_w(rng, {"not_pregnant": 6, "possible": 2.2 if preg_cat else 0.5,
                                 "confirmed": 1.6 if preg_cat else 0.4})
    lang = "English" if chance(rng, 0.8) else pick(rng, LANGUAGES)
    interp = "not_needed" if lang == "English" else pick_w(rng, {"phone_interpreter": 4, "video_interpreter": 2.5,
                                                                 "family_member": 2, "none": 1})
    return {"age": age, "sex": sex, "pregnancy": pregnancy, "race_ethnicity": pick(rng, RACE_ETH),
            "language": lang, "interpreter": interp, "payer": pick_w(rng, PAYERS), "housing": pick_w(rng, HOUSING),
            "geri_cat": geri_cat, "preg_cat": preg_cat,
            "weeks": rng.randint(6, 30) if pregnancy == "confirmed" else None,
            "lmp_weeks": rng.randint(5, 10) if pregnancy == "possible" else None}


def _preg_text(pt, voice="walkin"):
    w, lmp = pt["weeks"], pt["lmp_weeks"]
    return {"confirmed": {"walkin": f"pregnant ~{w} wk", "retriage": f"pregnant {w}/40",
                          "audit": f"known pregnancy, {w} weeks", "offload": f"pregnant ({w} wks)"},
            "possible": {"walkin": f"possible (LMP {lmp} wk ago, no test)", "retriage": f"? pregnant, LMP {lmp}/52",
                         "audit": f"possible pregnancy recorded (LMP {lmp} weeks)",
                         "offload": f"may be pregnant (period {lmp} wks ago)"},
            "not_pregnant": {"walkin": "not pregnant", "retriage": "not pregnant", "audit": "pregnancy excluded",
                             "offload": "not pregnant"}}.get(pt["pregnancy"], {}).get(voice)


def _pt_vals(pt, voice):
    payer = pt["payer"] if voice == "walkin" else _PAYER_TXT[voice][pt["payer"]]
    housing = pt["housing"] if voice == "walkin" else _HOUSING_TXT[voice][pt["housing"]]
    interp = (INTERP_TXT if voice == "walkin" else _INTERP_TXT[voice]).get(pt["interpreter"])
    return payer, housing, interp


def _pt_fields(pt, voice="walkin"):
    """The registration block, for JSON surfaces."""
    payer, housing, interp = _pt_vals(pt, voice)
    return {"age": pt["age"], "sex": pt["sex"], "pregnancy": _preg_text(pt, voice),
            "race_ethnicity": pt["race_ethnicity"], "language": pt["language"],
            "interpreter": interp, "payer": payer, "housing": housing}


def _pt_line(pt, voice="walkin"):
    """A one-line demographic header for text surfaces, ordered differently in each house style."""
    payer, housing, interp = _pt_vals(pt, voice)
    head = {"walkin": f"{pt['age']}{pt['sex']}", "retriage": f"{pt['age']} {pt['sex']}",
            "audit": f"{'female' if pt['sex'] == 'F' else 'male'}, {pt['age']}",
            "offload": f"{pt['age']}-year-old {pt['sex']}"}[voice]
    lang = (None if pt["language"] == "English" else
            {"walkin": f"prefers {pt['language']} ({interp})", "retriage": f"{pt['language']}-speaking, {interp}",
             "audit": f"preferred language {pt['language']}; {interp}",
             "offload": f"speaks {pt['language']} {interp}"}[voice])
    race = f"{_RACE_LABEL[voice]}: {pt['race_ethnicity']}"
    preg = _preg_text(pt, voice)
    order = {"walkin": [head, preg, lang, payer, housing, race], "retriage": [head, lang, preg, housing, payer, race],
             "audit": [race, head, preg, lang, housing, payer], "offload": [head, preg, race, lang, payer, housing]}
    return ("; " if voice != "retriage" else " | ").join(x for x in order[voice] if x)


def _draw_adult(rng, fam, reg="clear", exclude_high_risk=False):
    if fam == "limb":
        return _draw_limb(rng, reg)
    groups = ADULT_CX[fam]
    for _ in range(30):
        ccs, variants = pick(rng, groups)
        var = pick(rng, variants)
        if not (exclude_high_risk and var["high_risk"]):
            break
    return {"family": fam, "cc": pick(rng, ccs), "detail": _fill(var["text"], rng), "resources": list(var["res"]),
            "high_risk": var["high_risk"], "pain_range": list(var["pain"]), "prof": var["prof"],
            "ecg_rule": var["ecg"], "deficit_present": var["deficit"], "crit": var["crit"], "mental0": var["mental"],
            "ambulatory0": var["ambulatory"], "limb_site": None, "bony_tenderness": None, "can_use_limb": None,
            "deformity": None}


def _adult_order(p, cx, v, mental, lkw_h, pt=None):
    """The single standing order the written protocol starts (priority stroke > sepsis > ECG > X-ray)."""
    if cx["family"] == "neuro" and cx["deficit_present"] and (lkw_h is None or lkw_h <= p["stroke_h"]):
        return "stroke_team_page"
    need = 1 if (pt and pt["age"] >= p["older_age"]) else 2
    if cx["family"] == "infection" and _sepsis_count(p, v, mental) >= need:
        return "sepsis_screen_labs"
    if cx["ecg_rule"]:
        return "ecg_within_10_minutes"
    if cx["family"] == "limb" and (cx["bony_tenderness"] or not cx["can_use_limb"]):
        return "limb_xray_order"
    return "no_standing_order"


MENTAL_TXT = {
    "alert": ["alert and oriented x4", "A&O x4, speech clear", "alert, answers appropriately"],
    "anxious": ["alert, anxious", "alert and oriented, visibly worried"],
    "new_confusion": ["disoriented to date and place; {who} says this is new", "muddled, keeps asking where they are "
                      "(new per {who})"],
    "lethargic": ["drowsy, opens eyes to voice then drifts off", "lethargic, slow to answer questions"],
    "pain_only": ["responds only to a painful stimulus", "not rousable to voice, groans to sternal rub"],
}
OBS_DISTRESS = ["grimacing, guarding", "pale and sweaty, rocking with pain", "tearful, cannot sit still",
                "moaning, holding the painful area", "restless, wincing with every movement"]
OBS_CRIT = {"spo2": ["sitting forward, tripoding, working hard to breathe", "exhausted, accessory muscles in use"],
            "airway": ["anxious, drooling, hand at the throat", "frightened, sitting bolt upright, cannot swallow"],
            "sbp": ["pale, clammy, slumped in the wheelchair", "grey and sweaty, barely able to sit up"],
            "mental": ["slumped in the wheelchair, snoring respirations", "eyes closed, limp in the chair"]}
OBS_CALM = ["resting comfortably, chatting", "sitting quietly in no obvious distress", "scrolling on a phone, relaxed",
            "calm, answering questions easily", "laughing with their companion", "eating a snack, settled"]
DOWNPLAY = ["{who_cap} thinks it is just indigestion and nearly did not come in.",
            "Apologises for wasting time and says it is probably nothing.",
            "{who_cap} says this has happened before and settled on its own.",
            "Says they only came because {who} insisted.", "Would like to go home after a quick check."]
ALARM = ["Very anxious; asks for a scan straight away.", "{who_cap} says the patient 'looks awful' and wants a doctor now.",
         "Says the pain is the worst ever and something must be badly wrong.",
         "Has read about the symptoms online and is convinced it is serious.",
         "{who_cap} is upset about the wait and asks for the patient to be seen first."]


def _remark(rng, kind, who):
    w = who or "a friend"
    t = pick(rng, DOWNPLAY if kind == "downplay" else ALARM)
    if "{who_cap}" in t and not who:
        t = pick(rng, ["Says it is probably nothing.", "Apologises for coming in."]) if kind == "downplay" else \
            pick(rng, ["Very anxious; asks for a scan straight away.", "Insists on seeing a doctor now."])
    return t.replace("{who_cap}", "The " + w if who else "").replace("{who}", w)


def _mental_text(rng, mental, who):
    return pick(rng, MENTAL_TXT[mental]).replace("{who}", who or "a friend")


def _shape_adult_vitals(rng, p, cx, reg):
    """Vitals for an adult presentation, steered by regime. Returns (first set, recheck set or None)."""
    v = _draw_vitals(rng, cx["prof"])
    _settle(v, p, rng)
    prefs = DZ_PREF[cx["prof"]]
    recheck = None
    if reg == "clear":
        if chance(rng, 0.35):
            _cross(v, p, rng, pick(rng, prefs), "beyond")
    elif reg == "borderline":
        _cross(v, p, rng, pick(rng, prefs), "near")
    else:
        which = pick(rng, prefs)
        recheck = dict(v)
        _jitter(recheck, p, rng)
        if chance(rng, 0.5):    # the first reading crosses, the recheck does not
            _cross_mild(v, p, rng, which)
            _just_inside(recheck, p, rng, which)
        else:                   # a settled first reading, a recheck that crosses
            _just_inside(v, p, rng, which)
            _cross_mild(recheck, p, rng, which)
    if cx["family"] == "infection" and v["hr"] <= p["hr_lim"] and chance(rng, 0.6):
        v["hr"] = max(v["hr"], rng.randint(86, p["hr_lim"]))
        if recheck is not None and recheck["hr"] <= p["hr_lim"]:
            recheck["hr"] = min(p["hr_lim"], max(recheck["hr"], v["hr"] + rng.randint(-4, 4)))
    if cx["family"] == "infection" and v["rr"] <= p["rr_lim"] and chance(rng, 0.5):
        v["rr"] = max(v["rr"], rng.randint(p["sep_rr"] - 2, p["rr_lim"]))
    if cx["family"] == "infection" and reg == "borderline" and chance(rng, 0.5):
        v["temp_c"] = round(p["sep_t_hi"] + pick(rng, [-0.2, -0.1, 0.0, 0.1]), 1)
    if cx["family"] == "infection" and reg == "clear" and chance(rng, 0.3):
        v["temp_c"] = round(rng.uniform(36.8, 37.6), 1)
    return v, recheck


def _apply_crit(rng, p, cx, reg, v):
    """Level 1-type presentations; borderline ones sit right at the Level 1 limit. Returns (mental, airway)."""
    crit, mental, airway = cx["crit"], None, False
    if crit == "spo2":
        v["spo2"] = (p["l1_spo2"] + pick(rng, [-2, -1, 0, 1, 2]) if reg == "borderline"
                     else p["l1_spo2"] - rng.randint(2, 7))
        v["rr"] = max(v["rr"], p["rr_lim"] + rng.randint(4, 10))
    elif crit == "sbp":
        v["sbp"] = 80 + pick(rng, [-3, -1, 0, 2, 4]) if reg == "borderline" else rng.randint(64, 78)
        v["hr"] = max(v["hr"], rng.randint(112, 136))
        _fix_dbp(v, rng)
    elif crit == "airway":
        airway = reg != "borderline"
        v["hr"] = max(v["hr"], rng.randint(104, 128) if airway else rng.randint(92, 108))
        if airway:
            v["rr"] = max(v["rr"], p["rr_lim"] + rng.randint(2, 6))
            v["spo2"] = min(v["spo2"], rng.randint(p["l1_spo2"] + 2, 95))
    elif crit == "mental":
        mental = "lethargic" if reg == "borderline" else "pain_only"
        v["rr"] = rng.randint(7, 11) if chance(rng, 0.6) else v["rr"]
        v["spo2"] = min(v["spo2"], rng.randint(p["l1_spo2"] + 1, 95))
    return mental, airway


def _lkw_hours(rng, p, reg):
    w = p["stroke_h"]
    if reg == "clear":
        return round(rng.uniform(0.5, 2.5), 2) if chance(rng, 0.6) else round(rng.uniform(w + 3, 30), 2)
    if reg == "borderline":
        return max(0.25, round(near(rng, w, 1.2) * 4) / 4)
    return None if chance(rng, 0.5) else round(rng.uniform(0.5, w + 10), 2)


def _lkw_text(lkw_h, tri, rng):
    if lkw_h is None:
        return pick(rng, ["unknown: found by family, unclear when it began", "not established; lives alone, "
                          "neighbour unsure when last seen normal", "unknown (woke from a nap with it; nap length unclear)"])
    t = tri - timedelta(hours=lkw_h)
    day = "" if t.date() == tri.date() else " the previous day"
    return f"{_hm(t)}{day}"


# =========================================================================== schema 1
class AdultWalkinTriage(Schema):
    name = "adult_walkin_triage"
    description = ("An adult who walked in to the emergency department, seen at the triage desk: the acuity level, "
                   "stream and standing order the written protocol gives.")
    surfaces = {"json": 0.55, "triage_note": 0.225, "ehr_triage_form": 0.225}

    def sample_factors(self, rng):
        reg = regime(rng)
        p = _adult_protocol(rng)
        fam = pick_w(rng, {"cardiac": 19, "infection": 21, "neuro": 9, "limb": 18, "other": 28, "critical": 6.5})
        cx = _draw_adult(rng, fam, reg)
        v, recheck = _shape_adult_vitals(rng, p, cx, reg)
        mental, airway = (_apply_crit(rng, p, cx, reg, v) if fam == "critical" else (None, False))
        if fam == "critical":
            recheck = None
        if mental is None:
            mental = "alert"
            if fam == "infection" and chance(rng, 0.12):
                mental = "new_confusion"
            elif fam in ("cardiac", "other") and chance(rng, 0.03):
                mental = "lethargic"
            elif chance(rng, 0.2):
                mental = "anxious"
        lo, hi = cx["pain_range"]
        pain = rng.randint(lo, hi)
        distress = chance(rng, 0.7 if reg == "clear" else 0.55) if pain >= 7 else chance(rng, 0.06)
        if reg == "borderline" and hi >= 6 and chance(rng, 0.35):
            pain, distress = p["l2_pain"] + pick(rng, [-1, -1, 0]), True
        if reg == "conflicting" and hi >= 5 and chance(rng, 0.45):
            pain, distress = rng.randint(max(p["l2_pain"], 8), 10), False
        if chance(rng, 0.05) and fam != "critical":
            pain = None
        lkw_h = _lkw_hours(rng, p, reg) if fam == "neuro" else None
        who = pick_w(rng, {None: 3, "friend": 2, "family member": 2, "partner": 2, "coworker": 1, "neighbour": 1})
        pt = _patient(rng, cx, reg, p)
        v_now = recheck or v
        lvl = _adult_level(p, v_now, mental, airway, cx["high_risk"], pain, distress, len(cx["resources"]), pt)
        equity_note = pick(rng, EQUITY_NOTES) if chance(rng, 0.3 if reg == "conflicting" else 0.08) else None
        remark = None
        if reg == "conflicting" and chance(rng, 0.65):
            remark = _remark(rng, "downplay" if lvl <= 2 else "alarm", who)
        elif chance(rng, 0.12) and fam != "critical":
            remark = _remark(rng, pick(rng, ["downplay", "alarm"]), who)
        arr = rand_ts(rng)
        if reg == "borderline" and chance(rng, 0.35):
            edge = pick(rng, [p["ft_open"], p["ft_close"]])
            arr = arr.replace(hour=edge, minute=rng.randint(0, 25)) - timedelta(minutes=rng.randint(0, 45))
        tri = arr + timedelta(minutes=rng.randint(2, 24))
        ambulatory = cx["ambulatory0"] and (fam != "critical") and chance(rng, 0.93)
        mode = (pick(rng, ["walked in", "walked in from the car park", "dropped off by a friend, walked in",
                           "came by taxi", "walked in with a family member", "came by bus"]) if ambulatory else
                pick(rng, ["wheeled in from the car park by family", "brought to the door by a friend in a wheelchair",
                           "helped in by two friends", "brought in a borrowed wheelchair"]))
        return {"regime": reg, "facility": _facility(rng), "protocol": p, "family": fam, **cx,
                "vitals": v, "recheck": recheck, "recheck_min": rng.randint(8, 25) if recheck else None,
                "mental": mental, "airway_support_needed": airway, "pain": pain, "distress": distress,
                "lkw_h": lkw_h, "companion": who, "remark": remark, "arrival": arr, "triage_time": tri,
                "ambulatory": ambulatory, "arrival_mode": mode, "nurse": _nurse(rng), "encounter": _enc(rng, arr),
                "home_meds": _meds(rng), "allergies": pick(rng, ALLERGIES), "patient": pt, "equity_note": equity_note}

    def _story(self, f, rng):
        s = [f["detail"][0].upper() + f["detail"][1:] + ("" if f["detail"].endswith(".") else ".")]
        if f["airway_support_needed"]:
            s.append("Stridor at rest, struggling to speak.")
        elif f["crit"] == "airway":
            s.append("Noisy breathing settling after own adrenaline auto-injector in the car; speaking in sentences.")
        if f["remark"]:
            s.append(f["remark"])
        return noisy(join_sentences(s), rng, 0.15)

    def _obs(self, f, rng):
        if f["family"] == "critical" and (f["regime"] != "borderline" or f["crit"] in ("spo2", "sbp")):
            return pick(rng, OBS_CRIT[f["crit"]])
        return pick(rng, OBS_DISTRESS if f["distress"] else OBS_CALM)

    def render(self, f, rng, surface):
        p, unit = f["protocol"], f["protocol"]["unit"]
        tri = f["triage_time"]
        story, obs = self._story(f, rng), self._obs(f, rng)
        mental = _mental_text(rng, f["mental"], f["companion"])
        pain = f"{f['pain']}/10" if f["pain"] is not None else pick(rng, ["declined to rate", "unable to rate"])
        rules = _adult_rules(p, rng, ["level_1", "level_2", "resources", "danger_zone", "age_pregnancy",
                                      "standing_orders", "fast_track", "equity"])
        title = _protocol_title(f["facility"], p, rng)
        lkw = _lkw_text(f["lkw_h"], tri, rng) if f["family"] == "neuro" else None
        pt = f["patient"]
        re_t = tri + timedelta(minutes=f["recheck_min"]) if f["recheck"] else None
        if surface == "json":
            vit = {"triage_set": dict(_vs_dict(f["vitals"], unit), taken=_hm(tri))}
            if f["recheck"]:
                vit["repeat_set"] = dict(_vs_dict(f["recheck"], unit), taken=_hm(re_t))
            return dumps(prune({
                "arrival": {"encounter": f["encounter"], "arrived": f["arrival"].strftime("%Y-%m-%dT%H:%M"),
                            "mode": f["arrival_mode"], "accompanied_by": f["companion"], "ambulatory": f["ambulatory"]},
                "triage": {"nurse": f["nurse"], "time": _hm(tri), "chief_complaint": f["cc"], "history": story,
                           "appearance": obs, "mental_status": mental, "pain_score": pain, "last_known_well": lkw},
                "vital_signs": vit,
                "registration": dict(_pt_fields(pt), note=f["equity_note"]),
                "protocol": dict(rules, title=title),
            }))
        if surface == "triage_note":
            lines = [f"ED TRIAGE NOTE -- {f['facility']}", f"Enc {f['encounter']}. Arrived {_hm(f['arrival'])} "
                     f"({f['arrival_mode']}). Triaged {_hm(tri)} by {f['nurse']}.", f"Pt: {_pt_line(pt)}", "",
                     f"CC: {f['cc']}", f"HPI: {story}", f"Appearance: {obs}. {mental[0].upper() + mental[1:]}. "
                     f"Pain {pain}.", f"VS {_hm(tri)}: {_vs_line(f['vitals'], unit, rng)}"]
            if f["recheck"]:
                lines.append(f"Repeat VS {_hm(re_t)}: {_vs_line(f['recheck'], unit, rng)}")
            if lkw:
                lines.append(f"Last known well: {lkw}")
            if f["equity_note"]:
                lines.append(f["equity_note"])
            lines += ["", title + ":", bullets(list(rules.values()), rng)]
            return "\n".join(lines)
        rows = [("Encounter", f["encounter"]), ("Arrival", f"{fmt_date(f['arrival'].date(), rng)} {_hm(f['arrival'])}"),
                ("Arrival mode", f["arrival_mode"]), ("Ambulatory", "yes" if f["ambulatory"] else "no"),
                ("Triage nurse", f["nurse"]), ("Triage time", _hm(tri)), ("Chief complaint", f["cc"]),
                ("History", story), ("General appearance", obs), ("Neuro / LOC", mental), ("Pain", pain),
                ("Vitals (triage)", _vs_line(f["vitals"], unit, rng)),
                ("Vitals (repeat)", f"{_vs_line(f['recheck'], unit, rng)} at {_hm(re_t)}" if f["recheck"] else None),
                ("Last known well", lkw), ("Accompanied by", f["companion"]),
                ("Age / sex", f"{pt['age']} / {pt['sex']}"), ("Pregnancy", _preg_text(pt)),
                ("Preferred language", pt["language"] + ("" if pt["language"] == "English"
                                                         else f" ({INTERP_TXT[pt['interpreter']]})")),
                ("Payer", pt["payer"]), ("Housing", pt["housing"]), ("Race/ethnicity (recorded)", pt["race_ethnicity"]),
                ("Registration note", f["equity_note"])]
        return ("TRIAGE ASSESSMENT\n" + kv_lines(rows, rng) + f"\n\nProtocol reference: {title}\n"
                + "\n".join(f"[{k.replace('_', ' ')}] {t}" for k, t in rules.items()))

    def intended(self, f):
        p = f["protocol"]
        v = f["recheck"] or f["vitals"]
        n = len(f["resources"])
        lvl = _adult_level(p, v, f["mental"], f["airway_support_needed"], f["high_risk"], f["pain"], f["distress"], n,
                           f["patient"])
        if lvl == 1:
            stream = "resuscitation_bay"
        elif lvl == 2:
            stream = "monitored_acute_bed"
        elif lvl == 3:
            stream = "main_department_queue"
        else:
            stream = "fast_track" if (f["ambulatory"] and _ft_open(p, f["triage_time"])) else "main_department_queue"
        return {"triage_acuity": 5 - lvl, "resource_types_expected": min(n, 3), "triage_stream": stream,
                "standing_order": _adult_order(p, f, v, f["mental"], f["lkw_h"], f["patient"]),
                "danger_zone_vitals": "yes" if _dz_hits(p, v) else "no"}

    def questions(self):
        return {
            "triage_acuity": score("How acute is this patient under the facility's written triage protocol?", [
                "Level 5 (non-urgent): stable, with no resource type expected beyond the examination.",
                "Level 4 (less urgent): stable, with one resource type expected.",
                "Level 3 (urgent): two or more resource types expected and vital signs outside the danger zone.",
                "Level 2 (emergent): a high-risk presentation, new confusion, severe pain with distress, or "
                "danger-zone vital signs in a patient needing several resources.",
                "Level 1 (resuscitation): needs an immediate life-saving intervention.",
            ]),
            "resource_types_expected": score(
                "How many distinct resource types is this visit likely to use, counted the way the protocol "
                "counts them?", [
                    "None: examination, oral medication, a splint or a simple dressing at most.",
                    "One type, for example labs alone or a single X-ray.",
                    "Two types, for example labs plus an ECG or imaging.",
                    "Three or more types, for example labs, advanced imaging and IV medication.",
                ]),
            "triage_stream": choice("Where should the patient go from the triage desk?", {
                "resuscitation_bay": "Straight to a resuscitation bay, with the resuscitation team called.",
                "monitored_acute_bed": "Into the next monitored acute-care bed, ahead of everyone waiting.",
                "main_department_queue": "Into the main-department queue for their level, reassessed while waiting.",
                "fast_track": "To the fast-track stream for ambulatory patients with simple, low-resource problems.",
            }),
            "standing_order": choice("Which nurse-initiated standing order, if any, should start at triage?", {
                "ecg_within_10_minutes": "Record an ECG within ten minutes and show it to a physician.",
                "limb_xray_order": "Order an X-ray of the injured limb before the clinician sees the patient.",
                "no_standing_order": "None of the protocol's standing orders applies; the patient waits for a "
                                     "clinician.",
                "sepsis_screen_labs": "Draw the sepsis screening bloods and alert the nurse in charge.",
                "stroke_team_page": "Page the stroke team and prepare for urgent brain imaging.",
            }),
            "danger_zone_vitals": noul(
                "At least one of the patient's most recent vital signs lies in the protocol's danger zone.",
                yes="The latest heart rate, respiratory rate, oxygen saturation or systolic pressure crosses a "
                    "danger-zone limit written in the protocol excerpt.",
                no="Every latest vital sign is inside the written limits, even if one sits right at a limit or an "
                   "earlier reading crossed it."),
        }


# =========================================================================== pediatric protocol (fictional)
# label, upper age bound in months (exclusive), base HR limit, base RR limit, typical HR range, typical RR range
PED_BANDS = [
    ("<3 mo", 3, 180, 55, (122, 160), (32, 48)),
    ("3-11 mo", 12, 165, 48, (110, 148), (26, 40)),
    ("1-3 y", 48, 150, 38, (94, 134), (20, 30)),
    ("4-7 y", 96, 135, 30, (80, 118), (18, 24)),
    ("8-12 y", 156, 120, 25, (70, 106), (15, 21)),
    ("13-17 y", 216, 110, 22, (60, 98), (12, 19)),
]


def _ped_band(age_m):
    for i, b in enumerate(PED_BANDS):
        if age_m < b[1]:
            return i
    return len(PED_BANDS) - 1


def _ped_protocol(rng):
    hr_off, rr_off = pick(rng, [-5, 0, 5]), pick(rng, [-2, 0, 2])
    return {"rev": f"P{rng.randint(1, 6)}.{rng.randint(0, 9)}",
            "bands": [[b[0], b[2] + hr_off, b[3] + rr_off] for b in PED_BANDS],
            "spo2_lim": pick(rng, [92, 93, 94]), "l1_spo2": pick(rng, [85, 88]),
            "infant_fever_days": pick(rng, [28, 60, 90]), "wet_lim": pick(rng, [3, 4]),
            "antipyretic_h": pick(rng, [4, 6]), "unit": pick_w(rng, {"C": 3, "F": 1})}


def _exp_weight(age_m):
    if age_m < 12:
        return 3.4 + 0.55 * age_m
    y = age_m / 12
    return 2 * y + 8 if y < 6 else 3 * y + 3


def _age_text(age_m, age_d):
    if age_m < 3:
        return f"{age_d} days"
    if age_m < 24:
        return f"{age_m} months"
    y, m = divmod(age_m, 12)
    if m == 0 or y >= 6:
        return f"{y} years"
    return f"{y} years {m} month" + ("" if m == 1 else "s")


def P(text, res=(), high_risk=None, fam="other", fever=0.1, pain=(0, 3), wheeze=False, ages=(6, 200)):
    return {"text": text, "res": list(res), "high_risk": high_risk, "fam": fam, "fever": fever, "pain": list(pain),
            "wheeze": wheeze, "ages": list(ages)}


PED_CX = [
    (["fever", "hot to touch, feeding less"], [
        P("temperature at home this evening, feeding a little less than usual, otherwise settled", ("labs",), None,
          "fever", 0.9, (0, 0), ages=(0, 5)),
        P("fever since this afternoon, sleepier than usual between feeds", ("labs", "procedure"), None, "fever", 0.9,
          (0, 0), ages=(0, 5))]),
    (["fever and cough", "cough and temperature"], [
        P("fever and cough for {dur}, eating less, drinking fine", (), None, "fever", 0.8, (0, 2), ages=(6, 120)),
        P("cough for {dur}, fevers at night, breathing a little faster than usual", ("xray",), None, "fever", 0.8,
          (0, 2), ages=(6, 120)),
        P("fever for {dur} with cough, off food, crackles heard by the GP this morning", ("labs", "xray"), None,
          "fever", 0.85, (0, 2), ages=(12, 160))]),
    (["wheeze", "breathing fast and wheezy"], [
        P("known asthma, cough and wheeze since {since}, reliever inhaler helped only briefly", ("iv_meds",), None,
          "resp", 0.15, (0, 2), wheeze=True, ages=(24, 200)),
        P("wheezy with a cold since {since}, had wheezing episodes twice last winter", ("iv_meds",), None, "resp", 0.3,
          (0, 2), wheeze=True, ages=(12, 72))]),
    (["barking cough", "noisy breathing"], [
        P("barking cough and noisy breathing at night, settles when calm", (), None, "resp", 0.2, (0, 2),
          ages=(6, 72)),
        P("barking cough with noisy breathing even when sitting quietly", ("iv_meds",), None, "resp", 0.2, (0, 2),
          ages=(6, 72))]),
    (["vomiting and diarrhoea", "vomiting"], [
        P("vomiting and loose stools since {since}, keeping down small sips", (), None, "gi", 0.4, (1, 4),
          ages=(3, 150)),
        P("vomiting everything since {since}, loose stools, drinking very little", (), None, "gi", 0.5, (1, 4),
          ages=(3, 150))]),
    (["abdominal pain", "tummy ache"], [
        P("pain moved to the lower right side, walking bent over, off food", ("labs", "advanced_imaging", "iv_meds"),
          None, "other", 0.35, (5, 8), ages=(36, 215)),
        P("tummy ache on and off for {dur}, eating normally, last stool three days ago", (), None, "other", 0.05,
          (2, 4), ages=(24, 160))]),
    (["ear pain"], [P("pulling at the ear, unsettled at night for {dur}", (), None, "fever", 0.5, (2, 5),
                      ages=(6, 144))]),
    (["sore throat"], [P("sore throat and not wanting to eat for {dur}, drinking fine", (), None, "fever", 0.6,
                         (2, 5), ages=(36, 215))]),
    (["rash with fever", "spots and fever"], [
        P("fever with a rash of small purple spots that do not fade when a glass is pressed on them",
          ("labs", "iv_meds"), "non-blanching rash with fever", "fever", 0.95, (0, 3), ages=(3, 120))]),
    (["rash", "itchy skin"], [P("itchy dry patches flaring on the elbows and knees for {dur}", (), None, "other", 0.0,
                                (0, 2), ages=(3, 200))]),
    (["arm injury", "fall onto the arm"], [
        P("fell from the monkey bars onto the arm, swelling above the wrist, tender on the bone", ("xray",), None,
          "injury", 0.0, (4, 7), ages=(36, 200)),
        P("fell off a trampoline, forearm visibly bent", ("xray", "procedure", "iv_meds"), None, "injury", 0.0,
          (7, 10), ages=(36, 200))]),
    (["head injury", "bump to the head"], [
        P("rolled off the sofa onto carpet, cried straight away, now playing normally", (), None, "injury", 0.0,
          (0, 2), ages=(6, 48)),
        P("fell from a bunk bed and hit the head, vomited twice since, sleepier than usual", ("advanced_imaging",),
          "head injury with repeated vomiting and drowsiness", "injury", 0.0, (2, 5), ages=(24, 160))]),
    (["chin laceration", "cut chin"], [
        P("split the chin on the edge of the pool, gaping 1.5 cm cut", ("procedure",), None, "injury", 0.0, (2, 5),
          ages=(18, 144))]),
    (["swallowed object", "possible ingestion"], [
        P("may have swallowed a button battery from a remote control in the last hour", ("xray", "consult"),
          "possible button-battery ingestion", "other", 0.0, (0, 2), ages=(9, 72)),
        P("found with an open bottle of a relative's blood-pressure tablets, unsure how many taken",
          ("labs", "ecg", "consult"), "possible ingestion of another person's medication", "other", 0.0, (0, 1),
          ages=(12, 60)),
        P("swallowed a small coin an hour ago, eating and drinking normally since", ("xray",), None, "other", 0.0,
          (0, 1), ages=(12, 96))]),
    (["seizure with fever", "fit"], [
        P("shaking of all limbs for about two minutes with a fever, now awake and back to normal", ("labs",), None,
          "fever", 0.9, (0, 1), ages=(6, 60))]),
    (["poor feeding", "not feeding well"], [
        P("feeding less for {dur}, fewer wet nappies, unsettled", ("labs",), None, "gi", 0.3, (0, 1), ages=(0, 8))]),
    (["won't use arm"], [
        P("will not use the arm after being swung by the hands, holding it bent at the elbow", ("procedure",), None,
          "injury", 0.0, (2, 5), ages=(12, 60))]),
    (["object in nose"], [P("pushed a bead up the nostril, breathing fine", ("procedure",), None, "other", 0.0,
                            (0, 2), ages=(18, 84))]),
    (["scald", "burn"], [
        P("pulled a hot drink onto the chest, blistered patch smaller than the child's palm", (), None, "injury", 0.0,
          (3, 6), ages=(9, 96)),
        P("kettle scald over the chest and arm, blisters bigger than two of the child's palms", ("iv_meds", "consult"),
          None, "injury", 0.0, (6, 9), ages=(9, 96))]),
    (["painful urination", "urine infection symptoms"], [
        P("stinging when passing urine and going more often for {dur}", ("labs",), None, "fever", 0.4, (2, 4),
          ages=(36, 200)),
        P("fever with back pain and vomiting once, stinging when passing urine", ("labs", "iv_meds"), None, "fever",
          0.85, (3, 6), ages=(48, 200))]),
    (["abdominal pain and vomiting"], [
        P("central tummy pain since {since} with vomiting, does not want to walk", ("labs", "advanced_imaging"), None,
          "other", 0.3, (4, 7), ages=(48, 200))]),
    (["foot wound"], [
        P("stepped on broken glass at the beach, deep cut on the sole, may still have glass in it",
          ("procedure", "xray"), None, "injury", 0.0, (3, 6), ages=(48, 215))]),
    (["sticky eye"], [P("sticky discharge and redness of one eye for {dur}", (), None, "other", 0.05, (0, 1),
                        ages=(3, 120))]),
    (["limp"], [P("limping since waking, no injury anyone remembers, low temperature", ("labs", "xray"), None, "fever",
                  0.6, (2, 4), ages=(24, 120))]),
    (["bee sting"], [P("bee sting on the foot, local swelling only, breathing normal", (), None, "other", 0.0, (1, 3),
                       ages=(12, 200))]),
    (["ankle injury"], [P("twisted the ankle at netball, tender over the bone, limping", ("xray",), None, "injury",
                          0.0, (3, 6), ages=(120, 215))]),
]

PED_MENTAL_TXT = {
    "alert": ["alert, playful", "alert, interacting with caregiver", "awake, watching the room"],
    "irritable": ["irritable but settles in caregiver's arms", "crying, consolable"],
    "lethargic": ["floppy and sleepy, hard to engage", "lethargic, weak cry"],
    "inconsolable": ["inconsolable, high-pitched cry", "screaming, cannot be settled by caregiver"],
    "pain_only": ["responds only to painful stimulus", "unrousable to voice"],
}


def _ped_rules(p, rng):
    t38 = _temp_txt(38.0, p["unit"]).replace(" ", "")
    bands = "; ".join(f"{b[0]} {b[1]}/{b[2]}" for b in p["bands"])
    return {
        "age_band_limits": f"Danger zone by age band (HR/RR above): {bands}. SpO2 < {p['spo2_lim']}% at any age.",
        "level_1": f"Level 1: unresponsive or responds only to pain, apnoea, or SpO2 < {p['l1_spo2']}%.",
        "level_2": pick(rng, [
            f"Level 2: high-risk presentation; lethargic, floppy or inconsolable; severe dehydration; temperature "
            f">= {t38} under {p['infant_fever_days']} days old; a danger-zone vital when otherwise Level 3-4.",
            f"Level 2 if any of: high-risk presentation, lethargy or inconsolability, severe dehydration, fever "
            f">= {t38} under {p['infant_fever_days']} days of age, danger-zone vital (would-be Level 3-4 only)."]),
        "resources": "Otherwise by resource types expected: two or more gives Level 3, one gives 4, none gives 5 "
                     "(drinks, medicine by mouth, the exam and bedside tests are not counted).",
        "hydration": f"Dehydration signs: cap refill > 2 s; sticky/dry mouth; no tears; < {p['wet_lim']} wet nappies or "
                     f"voids in 24 h; lethargy. 0 none, 1-2 mild, 3-4 moderate, all 5 (or cap refill > 3 s with "
                     f"lethargy) severe.",
        "standing_orders": f"Levels 1-2: clinician now, nothing nurse-started first. Levels 3-5, first that applies: "
                           f"nebuliser for wheeze with asthma or prior wheeze; oral rehydration trial for "
                           f"vomiting/diarrhoea with mild or moderate dehydration; antipyretic for T >= {t38} from 3 "
                           f"months if none in the last {p['antipyretic_h']} h.",
        "equity": "Never let insurance, immigration questions, ethnicity, faith or address alter the level. History "
                  "through a professional interpreter if the caregiver prefers another language, never a child.",
        "weight": "Dosing weight: measured on the department scale this visit and plausible for age; not caregiver "
                  "estimates or old records.",
    }


def _ped_dz(p, f, v):
    b = p["bands"][_ped_band(f["age_months"])]
    hits = []
    if v.get("hr") is not None and v["hr"] > b[1]:
        hits.append("hr")
    if v.get("rr") is not None and v["rr"] > b[2]:
        hits.append("rr")
    if v.get("spo2") is not None and v["spo2"] < p["spo2_lim"]:
        hits.append("spo2")
    return hits


def _ped_dehydration(p, f):
    n = 0
    n += f["cap_refill_s"] is not None and f["cap_refill_s"] > 2.0
    n += f["membranes"] in ("sticky", "dry")
    n += f["tears"] == "absent"
    n += f["wet_24h"] is not None and f["wet_24h"] < p["wet_lim"]
    n += f["mental"] in ("lethargic", "pain_only")
    if n >= 5 or (f["cap_refill_s"] is not None and f["cap_refill_s"] > 3.0 and f["mental"] == "lethargic"):
        return 3
    return 2 if n >= 3 else (1 if n >= 1 else 0)


# =========================================================================== schema 2
class PediatricTriage(Schema):
    name = "pediatric_triage"
    description = ("A child brought to the emergency department, triaged against the written pediatric protocol "
                   "with age-banded vital-sign limits.")
    surfaces = {"json": 0.55, "peds_triage_note": 0.225, "triage_flowsheet": 0.225}

    def sample_factors(self, rng):
        reg = regime(rng)
        p = _ped_protocol(rng)
        fam_w = {"fever": 27, "gi": 22, "resp": 13, "injury": 22, "other": 16}
        fam = pick_w(rng, fam_w)
        opts = [(ccs, var) for ccs, vs in PED_CX for var in vs if var["fam"] == fam]
        ccs, var = pick(rng, opts)
        lo, hi = var["ages"]
        if var["text"].startswith(("temperature at home", "fever since this afternoon")) and reg == "borderline":
            age_d = max(5, near(rng, p["infant_fever_days"], 12, integer=True))
            age_m = int(age_d / 30.4)
        else:
            age_m = rng.randint(lo, hi)
            age_d = int(age_m * 30.4) + rng.randint(0, 29)
            age_m = min(int(age_d / 30.4), 215)
        bi = _ped_band(age_m)
        band = PED_BANDS[bi]
        hr = rng.randint(*band[4])
        rr = rng.randint(*band[5])
        spo2 = rng.randint(96, 100)
        febrile = chance(rng, var["fever"])
        temp = round(rng.uniform(38.1, 40.0), 1) if febrile else round(rng.uniform(36.4, 37.6), 1)
        if febrile and reg == "borderline" and chance(rng, 0.5):
            temp = round(38.0 + pick(rng, [-0.2, -0.1, 0.0, 0.1, 0.2]), 1)
        hr += int(max(0.0, temp - 37.5) * 10)
        if fam == "resp":
            rr += rng.randint(4, 12)
            spo2 = rng.randint(91, 98)
        v = {"hr": hr, "rr": rr, "spo2": spo2, "temp_c": temp}
        lim = p["bands"][bi]
        # steer one vital relative to the age-band limit
        which = pick(rng, ["hr", "rr", "spo2"] if fam == "resp" else ["hr", "hr", "rr"])
        lims = {"hr": lim[1], "rr": lim[2]}
        if reg == "clear":
            if chance(rng, 0.3):
                if which == "spo2":
                    v["spo2"] = p["spo2_lim"] - rng.randint(2, 4)
                else:
                    v[which] = lims[which] + rng.randint(6, 20)
            else:
                v["hr"] = min(v["hr"], lim[1] - rng.randint(8, 25))
                v["rr"] = min(v["rr"], lim[2] - rng.randint(3, 8))
                v["spo2"] = max(v["spo2"], p["spo2_lim"] + 2)
        elif reg == "borderline":
            v["hr"] = min(v["hr"], lim[1] - rng.randint(5, 20))
            v["rr"] = min(v["rr"], lim[2] - rng.randint(2, 6))
            v["spo2"] = max(v["spo2"], p["spo2_lim"] + 1)
            if which == "spo2":
                v["spo2"] = p["spo2_lim"] + pick(rng, [-2, -1, 0, 1])
            elif which == "hr":
                v["hr"] = near(rng, lim[1], 6, integer=True)
            else:
                v["rr"] = lim[2] + pick(rng, [-2, -1, 0, 1, 2])
        else:
            v["hr"] = min(v["hr"], lim[1] - rng.randint(4, 16))
            v["rr"] = min(v["rr"], lim[2] - rng.randint(2, 6))
            v["spo2"] = max(v["spo2"], p["spo2_lim"] + 1)
            if chance(rng, 0.35):
                v[which if which != "spo2" else "hr"] = lims[which if which != "spo2" else "hr"] + rng.randint(4, 14)
        v["spo2"] = min(100, v["spo2"])
        # level-1 presentations are rare in a walk-in pediatric stream
        mental = pick_w(rng, {"alert": 6, "irritable": 3, "lethargic": 0.6, "inconsolable": 0.5})
        if chance(rng, 0.03):
            mental = "pain_only"
        if fam == "resp" and chance(rng, 0.05):
            v["spo2"] = p["l1_spo2"] + pick(rng, [-3, -1, 0, 1])
        # hydration signs, aimed at a target degree
        if fam == "gi":
            target = pick_w(rng, {0: 1.5, 1: 3, 2: 3, 3: 1.3})
        elif fam == "fever":
            target = pick_w(rng, {0: 4, 1: 3, 2: 1.2})
        else:
            target = pick_w(rng, {0: 8, 1: 1})
        if target == 3:
            mental = "lethargic"
        k = {0: 0, 1: rng.randint(1, 2), 2: rng.randint(3, 4), 3: 5}[target]
        slots = ["cap", "membranes", "tears", "wet"]
        present = rng.sample(slots, min(k, 4))
        edge = reg == "borderline"
        cap = (pick(rng, [2.5, 3.0, 3.5]) if "cap" in present else (2.0 if edge and chance(rng, 0.5) else 1.5))
        if target == 3 and chance(rng, 0.5):
            cap = 3.5
        membranes = pick(rng, ["sticky", "dry"]) if "membranes" in present else "moist"
        tears = "absent" if "tears" in present else ("reduced" if edge and chance(rng, 0.5) else "present")
        wet = (rng.randint(0, p["wet_lim"] - 1) if "wet" in present
               else (p["wet_lim"] if edge and chance(rng, 0.5) else rng.randint(p["wet_lim"] + 1, p["wet_lim"] + 5)))
        if chance(rng, 0.1):
            wet = None
        if age_m < 3 and chance(rng, 0.5):
            tears = None
        # resources for vomiting / diarrhoea follow the hydration picture
        res = list(var["res"])
        if fam == "gi" and var["text"].startswith(("vomiting", "feeding")):
            res = {0: [], 1: [], 2: ["labs"], 3: ["labs", "iv_meds"]}[target] if var["text"].startswith("vomiting") \
                else res
        # antipyretic history
        last_dose = None
        if febrile and chance(rng, 0.6):
            hrs = (round(near(rng, p["antipyretic_h"], 1.0) * 2) / 2 if reg != "clear"
                   else pick(rng, [1, 1.5, 2, 8, 10]))
            last_dose = {"drug": pick(rng, ["acetaminophen (paracetamol)", "ibuprofen"]), "hours_ago": max(0.5, hrs)}
            if reg == "conflicting" and chance(rng, 0.4):
                last_dose["hours_ago"] = None
        # weight and where it came from
        w_true = round(_exp_weight(age_m) * rng.uniform(0.84, 1.18), 1)
        src = pick_w(rng, {"scale_this_visit": 5.5, "caregiver_estimate": 2.5, "previous_record": 2})
        lb_error = src == "scale_this_visit" and chance(rng, 0.18 if reg == "conflicting" else 0.08)
        weight = round(w_true * 2.2, 1) if lb_error else w_true
        caregiver = pick(rng, ["mother", "father", "grandparent", "aunt", "uncle", "foster carer", "parent",
                               "both parents", "childminder"] + (["older sibling (adult)"] if age_m >= 96 else []))
        concern = None
        if reg == "conflicting" and chance(rng, 0.6):
            concern = pick(rng, ["Caregiver feels the child is 'much worse than they look'.",
                                 "Caregiver thinks it is just teething and wanted reassurance.",
                                 "Caregiver says the child 'is not themselves today'.",
                                 "Caregiver apologises for coming, says the child perked up in the car."])
        arr = rand_ts(rng)
        return {"regime": reg, "facility": _facility(rng), "protocol": p, "family": fam,
                "cc": pick(rng, ccs), "detail": _fill(var["text"], rng), "resources": res,
                "high_risk": var["high_risk"], "wheeze_rule": var["wheeze"],
                "vomiting_diarrhoea": var["text"].startswith("vomiting"), "age_months": age_m, "age_days": age_d,
                "vitals": v, "mental": mental, "cap_refill_s": cap, "membranes": membranes, "tears": tears,
                "wet_24h": wet, "hydration_target": target, "last_antipyretic": last_dose,
                "weight_kg": weight, "weight_source": src, "weight_unit_error": lb_error,
                "pain": rng.randint(*var["pain"]), "caregiver": caregiver, "caregiver_remark": concern,
                "sex": pick(rng, ["F", "M"]), "race_ethnicity": pick(rng, RACE_ETH),
                "caregiver_language": "English" if chance(rng, 0.78) else pick(rng, LANGUAGES),
                "payer": pick_w(rng, {"commercial plan": 3, "public children's plan": 3.5, "uninsured": 0.8,
                                      "emergency coverage only": 0.5}),
                "arrival": arr, "triage_time": arr + timedelta(minutes=rng.randint(3, 30)),
                "nurse": _nurse(rng), "encounter": _enc(rng, arr)}

    def _level(self, f):
        p, v = f["protocol"], f["vitals"]
        if f["mental"] == "pain_only" or (v["spo2"] < p["l1_spo2"]):
            return 1
        infant_fever = f["age_days"] < p["infant_fever_days"] and v["temp_c"] >= 38.0
        if (f["high_risk"] or f["mental"] in ("lethargic", "inconsolable") or infant_fever
                or _ped_dehydration(p, f) == 3):
            return 2
        n = len(f["resources"])
        base = 3 if n >= 2 else (4 if n == 1 else 5)
        return 2 if (base in (3, 4) and _ped_dz(p, f, v)) else base

    def intended(self, f):
        p, v = f["protocol"], f["vitals"]
        lvl = self._level(f)
        deh = _ped_dehydration(p, f)
        if lvl <= 2:
            first = "immediate_clinician_review"
        elif f["wheeze_rule"]:
            first = "nebulised_bronchodilator"
        elif f["vomiting_diarrhoea"] and deh in (1, 2):
            first = "oral_rehydration_trial"
        elif v["temp_c"] >= 38.0 and f["age_months"] >= 3 and (f["last_antipyretic"] is None or (
                f["last_antipyretic"]["hours_ago"] is not None
                and f["last_antipyretic"]["hours_ago"] >= p["antipyretic_h"])):
            first = "antipyretic_dose"
        else:
            first = "no_nurse_initiated_treatment"
        plausible = 0.6 <= f["weight_kg"] / _exp_weight(f["age_months"]) <= 1.6
        return {"peds_acuity": 5 - lvl, "dehydration_degree": deh, "first_triage_treatment": first,
                "vitals_outside_age_band": "yes" if _ped_dz(p, f, v) else "no",
                "weight_usable_for_dosing": "yes" if (f["weight_source"] == "scale_this_visit" and plausible) else "no"}

    def _texts(self, f, rng):
        p = f["protocol"]
        unit = p["unit"]
        dose = f["last_antipyretic"]
        if dose is None:
            dose_t = pick(rng, ["no antipyretic given today", "nothing given for fever before arrival", "none given"])
        elif dose["hours_ago"] is None:
            dose_t = f"{dose['drug']} given earlier today, caregiver unsure what time"
        else:
            dose_t = f"{dose['drug']} given {dose['hours_ago']:g} h before triage"
        src = {"scale_this_visit": "department scale, this visit", "caregiver_estimate": "caregiver estimate",
               "previous_record": f"record from a clinic visit {pick(rng, [2, 3, 5, 7])} months ago"}[f["weight_source"]]
        hyd = {"capillary_refill": f"{f['cap_refill_s']:g} s", "mucous_membranes": f["membranes"],
               "tears": f["tears"] or "not assessed",
               "wet_nappies_or_voids_24h": f["wet_24h"] if f["wet_24h"] is not None else "caregiver unsure"}
        story = f["detail"][0].upper() + f["detail"][1:] + "."
        if f["caregiver_remark"]:
            story += " " + f["caregiver_remark"]
        scale = "FLACC" if f["age_months"] < 48 else ("faces scale" if f["age_months"] < 96 else "numeric")
        pain = f"{f['pain']}/10 ({scale})"
        return dose_t, src, hyd, noisy(story, rng, 0.12), _temp_txt(f["vitals"]["temp_c"], unit), pain

    def render(self, f, rng, surface):
        p, v = f["protocol"], f["vitals"]
        dose_t, src, hyd, story, temp, pain = self._texts(f, rng)
        age = _age_text(f["age_months"], f["age_days"])
        mental = pick(rng, PED_MENTAL_TXT[f["mental"]])
        rules = _ped_rules(p, rng)
        title = pick(rng, [f"{f['facility']} pediatric triage standard {p['rev']} (excerpt)",
                           f"Pediatric ED triage protocol {p['rev']}, {f['facility']}"])
        tri = f["triage_time"]
        if surface == "json":
            return dumps(prune({
                "child": {"encounter": f["encounter"], "age": age, "sex": f["sex"], "brought_by": f["caregiver"],
                          "caregiver_language": f["caregiver_language"], "payer": f["payer"],
                          "race_ethnicity_recorded": f["race_ethnicity"],
                          "weight": {"kg": f["weight_kg"], "source": src}},
                "presentation": {"chief_complaint": f["cc"], "history": story, "behaviour": mental,
                                 "pain_score": pain, "antipyretic_before_arrival": dose_t},
                "triage": {"time": _hm(tri), "nurse": f["nurse"],
                           "vitals": {"hr": v["hr"], "rr": v["rr"], "spo2_pct": v["spo2"], "temperature": temp},
                           "hydration": hyd},
                "protocol": dict(rules, title=title),
            }))
        if surface == "peds_triage_note":
            lines = [f"PEDS TRIAGE -- {f['facility']}  enc {f['encounter']}",
                     f"{age} old {'girl' if f['sex'] == 'F' else 'boy'}, with {f['caregiver']}"
                     + ("" if f["caregiver_language"] == "English" else f" (prefers {f['caregiver_language']})")
                     + f". Triage {_hm(tri)}, {f['nurse']}. {f['payer']}.",
                     f"CC: {f['cc']}. {story}",
                     f"Behaviour: {mental}. Pain {pain}.",
                     f"VS: HR {v['hr']}  RR {v['rr']}  SpO2 {v['spo2']}%  T {temp}",
                     f"Hydration: cap refill {hyd['capillary_refill']}, membranes {hyd['mucous_membranes']}, tears "
                     f"{hyd['tears']}, wet nappies/voids 24 h: {hyd['wet_nappies_or_voids_24h']}",
                     f"Antipyretic: {dose_t}.", f"Weight: {f['weight_kg']} kg ({src}).", "", title + ":",
                     bullets(list(rules.values()), rng)]
            return "\n".join(lines)
        rows = [["Heart rate", v["hr"]], ["Resp rate", v["rr"]], ["SpO2", f"{v['spo2']}%"], ["Temperature", temp],
                ["Cap refill", hyd["capillary_refill"]], ["Mucous membranes", hyd["mucous_membranes"]],
                ["Tears", hyd["tears"]], ["Wet nappies/voids (24 h)", hyd["wet_nappies_or_voids_24h"]],
                ["Weight", f"{f['weight_kg']} kg ({src})"]]
        head = kv_lines([("Encounter", f["encounter"]), ("Age", age), ("Sex", f["sex"]), ("With", f["caregiver"]),
                         ("Caregiver language", f["caregiver_language"]), ("Payer", f["payer"]),
                         ("Triage", f"{_hm(tri)} {f['nurse']}"), ("Complaint", f["cc"]), ("History", story),
                         ("Behaviour", mental), ("Pain", pain), ("Antipyretic", dose_t)], rng)
        return (f"PEDIATRIC TRIAGE FLOWSHEET\n{head}\n\n{table(['item', 'value'], rows, rng)}\n\n{title}\n"
                + "\n".join(f"- {t}" for t in rules.values()))

    def questions(self):
        return {
            "peds_acuity": score("How acute is this child under the written pediatric triage protocol?", [
                "Level 5: stable, with no resource type expected.",
                "Level 4: stable, with one resource type expected.",
                "Level 3: two or more resource types expected and vital signs inside the age-band limits.",
                "Level 2: a high-risk presentation, lethargy, severe dehydration, a febrile young infant, or a "
                "danger-zone vital sign.",
                "Level 1: needs an immediate life-saving intervention.",
            ]),
            "dehydration_degree": score("How dehydrated is the child by the protocol's count of signs?", [
                "None: no dehydration sign present.",
                "Mild: one or two signs.",
                "Moderate: three or four signs.",
                "Severe: every sign present, or a very slow capillary refill in a lethargic child.",
            ]),
            "first_triage_treatment": choice("What should the triage nurse start first for this child?", {
                "antipyretic_dose": "Give a weight-based fever medicine under the standing order.",
                "immediate_clinician_review": "Get a clinician to the child now; no nurse-started treatment should "
                                              "come first.",
                "nebulised_bronchodilator": "Start the bronchodilator nebuliser under the standing order.",
                "no_nurse_initiated_treatment": "Start nothing under standing orders; the child waits for the "
                                                "clinician.",
                "oral_rehydration_trial": "Begin a supervised oral rehydration trial with small, frequent drinks.",
            }),
            "vitals_outside_age_band": noul(
                "At least one of the child's vital signs is outside the limits for the child's age band.",
                yes="The heart rate or breathing rate is above the limit in the row for the child's age, or the "
                    "oxygen saturation is below the protocol's floor.",
                no="Every vital sign is within the limits of the child's own age-band row, even if it would be out "
                   "of range for an older child."),
            "weight_usable_for_dosing": noul(
                "The weight on the triage record can be used for weight-based medication doses."),
        }


# =========================================================================== schema 3
WAIT_EVENTS = {
    "no_change": ["no new complaints, waiting quietly", "asks how much longer the wait will be; symptoms unchanged",
                  "says nothing has changed"],
    "pain_worse": ["says the pain is now {p}/10 and getting worse", "pain has crept up to {p}/10 while waiting"],
    "pain_eased": ["pain has eased to {p}/10 since the oral analgesia given at triage",
                   "feels a lot better, pain down to {p}/10"],
    "vomiting": ["vomited twice in the waiting-room toilet", "has started vomiting, cannot keep water down"],
    "fainted": ["fainted while queuing at the vending machine, caught by another visitor",
                "slumped in the chair for a few seconds, then came round"],
    "new_confusion": ["companion says the patient is now muddled and keeps asking where they are",
                      "now disoriented to place; was oriented at triage"],
    "chest_tightness": ["now describes tightness across the chest that began 10 minutes ago",
                        "reports new central chest pressure while sitting"],
    "more_breathless": ["more short of breath, has to pause mid-sentence", "breathing harder than at triage"],
    "fever_spike": ["feels hot and shivery, teeth chattering", "says they feel feverish and cold at the same time"],
}


class WaitingRoomRetriage(Schema):
    name = "waiting_room_retriage"
    description = ("A patient re-checked in the waiting room some time after triage: re-triage on current findings "
                   "under the written reassessment rules.")
    surfaces = {"json": 0.55, "reassessment_flowsheet": 0.225, "waiting_room_note": 0.225}

    def sample_factors(self, rng):
        reg = regime(rng)
        p = _adult_protocol(rng)
        iv = {"2": 15, "3": pick(rng, [30, 45, 60]), "4": pick(rng, [60, 90, 120]), "5": pick(rng, [120, 180])}
        fam = pick_w(rng, {"cardiac": 2, "infection": 3, "limb": 3, "other": 5})
        cx = _draw_adult(rng, fam, "clear", exclude_high_risk=True)
        if cx["high_risk"]:
            cx = _draw_adult(rng, "limb", "clear")
        v0 = _draw_vitals(rng, cx["prof"])
        _settle(v0, p, rng)
        lo, hi = cx["pain_range"]
        pain0 = rng.randint(lo, max(lo, min(hi, p["l2_pain"] - 1)))
        res0 = list(cx["resources"])
        pt = _patient(rng, cx, "clear", p)
        lvl0 = _adult_level(p, v0, "alert", False, None, pain0, False, len(res0), pt)
        if lvl0 <= 2:               # a patient the protocol puts at Level 2 is not left in the waiting room
            pt["pregnancy"], pt["weeks"], pt["lmp_weeks"] = "not_pregnant", None, None
            lvl0 = _adult_level(p, v0, "alert", False, None, pain0, False, len(res0), pt)
        assigned = lvl0 - 1 if (lvl0 >= 4 and chance(rng, 0.35)) else lvl0
        if reg == "clear":
            ev = pick_w(rng, {"no_change": 3, "pain_eased": 2, "fainted": 1, "new_confusion": 1,
                              "chest_tightness": 1})
        elif reg == "borderline":
            ev = pick_w(rng, {"pain_worse": 3, "vomiting": 2, "more_breathless": 2, "fever_spike": 2,
                              "no_change": 2, "pain_eased": 1.5})
        else:
            ev = pick_w(rng, {"no_change": 2, "pain_worse": 2, "pain_eased": 2, "more_breathless": 1,
                              "fever_spike": 1, "vomiting": 1})
        v1 = dict(v0)
        _jitter(v1, p, rng)
        pain1, distress1, mental1, hr1, res1 = pain0, False, "alert", None, list(res0)
        prefs = DZ_PREF[cx["prof"]]
        if ev == "pain_worse":
            pain1 = min(10, pain0 + rng.randint(2, 4))
            distress1 = chance(rng, 0.6)
            v1["hr"] += rng.randint(3, 10)
            if reg == "borderline":
                pain1, distress1 = p["l2_pain"] + pick(rng, [-1, 0]), True
            if reg == "conflicting":
                pain1, distress1 = max(pain1, p["l2_pain"]), False
        elif ev == "pain_eased":
            pain1 = max(0, pain0 - rng.randint(2, 4))
            v1["hr"] -= rng.randint(4, 12)
            if "iv_meds" in res1:
                res1.remove("iv_meds")
            if reg == "conflicting":
                _cross(v1, p, rng, pick(rng, prefs), "beyond")
        elif ev == "vomiting":
            if "iv_meds" not in res1:
                res1.append("iv_meds")
            if "labs" not in res1 and chance(rng, 0.5):
                res1.append("labs")
            v1["hr"] += rng.randint(3, 10)
        elif ev == "fainted":
            hr1 = "fainted in the waiting room"
        elif ev == "new_confusion":
            mental1 = "new_confusion"
        elif ev == "chest_tightness":
            hr1 = "new chest tightness while waiting"
        elif ev == "more_breathless":
            v1["rr"] += rng.randint(2, 6)
            v1["spo2"] -= rng.randint(1, 3)
            _cross(v1, p, rng, pick(rng, ["rr", "spo2"]), "near" if reg == "borderline" else "beyond")
            if "xray" not in res1:
                res1.append("xray")
        elif ev == "fever_spike":
            v1["temp_c"] = round(rng.uniform(38.4, 39.6), 1)
            v1["hr"] += rng.randint(8, 18)
            if reg == "borderline":
                _cross(v1, p, rng, "hr", "near")
            if "labs" not in res1:
                res1.append("labs")
        elif ev == "no_change" and reg == "conflicting":
            _cross(v1, p, rng, pick(rng, prefs), "beyond")
        v1["spo2"] = min(100, v1["spo2"])
        _fix_dbp(v1, rng)
        missing = None
        if chance(rng, 0.42):
            missing = pick(rng, ["rr", "sbp", "spo2", "temp_c"] if v0["temp_c"] >= 38.0 else ["rr", "sbp", "spo2"])
            v1[missing] = None
            if missing == "sbp":
                v1["dbp"] = None
        interval = iv[str(assigned)]
        if reg == "clear":
            gap = int(interval * pick(rng, [0.5, 0.7, 1.6, 2.0]))
        elif reg == "borderline":
            gap = max(5, near(rng, interval, 10, integer=True))
        else:
            gap = int(interval * rng.uniform(0.4, 2.2))
        tri = rand_ts(rng, hours=(8, 23))
        last_check = None
        if chance(rng, 0.45):
            last_check = tri + timedelta(minutes=rng.randint(15, 70))
        now = (last_check or tri) + timedelta(minutes=gap)
        remark = None
        if reg == "conflicting" and ev == "no_change":
            remark = pick(rng, ["Patient insists they feel fine and just want to be seen.",
                                "Says they are 'totally fine, just bored of waiting'."])
        return {"regime": reg, "facility": _facility(rng), "protocol": p, "intervals_min": iv, "family": fam, **cx,
                "resources": res0, "resources_now": res1, "vitals_triage": v0, "pain_triage": pain0,
                "protocol_level_at_triage": lvl0, "assigned_level": assigned, "event": ev,
                "event_text": (pick(rng, ["symptoms settling, feels better than at triage",
                                          "says they feel much better now"]) if ev == "pain_eased" and pain0 <= 2
                               else pick(rng, WAIT_EVENTS[ev]).replace("{p}", str(pain1))), "remark": remark,
                "vitals_now": v1, "missing_now": missing, "pain_now": pain1, "distress_now": distress1,
                "mental_now": mental1, "high_risk_now": hr1, "triage_time": tri, "last_check": last_check,
                "check_time": now, "minutes_since_last": gap, "nurse": _nurse(rng), "checker": _nurse(rng),
                "encounter": _enc(rng, tri), "patient": pt}

    def _new_level(self, f):
        return _adult_level(f["protocol"], f["vitals_now"], f["mental_now"], False, f["high_risk_now"],
                            f["pain_now"], f["distress_now"], len(f["resources_now"]), f["patient"])

    def intended(self, f):
        p = f["protocol"]
        lvl1, assigned = self._new_level(f), f["assigned_level"]
        v0, v1 = f["vitals_triage"], f["vitals_now"]
        dz_new = set(_dz_hits(p, v1)) - set(_dz_hits(p, v0))
        if lvl1 == 1 or f["high_risk_now"] or f["mental_now"] == "new_confusion" or dz_new:
            trend = 3
        elif f["event"] in ("pain_worse", "vomiting", "more_breathless", "fever_spike") or (
                v1.get("hr") is not None and v1["hr"] >= v0["hr"] + 10):
            trend = 2
        elif f["event"] == "pain_eased":
            trend = 0
        else:
            trend = 1
        if lvl1 < assigned:
            call = "upgrade_room_now" if lvl1 <= 2 else "upgrade_in_queue"
        elif lvl1 == assigned:
            call = "keep_current_level"
        else:
            call = "downgrade_or_fast_track"
        if lvl1 <= 2:
            nxt = 3
        else:
            m = f["intervals_min"][str(lvl1)]
            nxt = 0 if m >= 120 else (1 if m >= 60 else 2)
        complete = all(v1.get(k) is not None for k in ("hr", "rr", "sbp", "spo2")) and (
            v1.get("temp_c") is not None or v0["temp_c"] < 38.0)
        overdue = f["minutes_since_last"] > f["intervals_min"][str(assigned)]
        return {"retriage_call": call, "condition_trend": trend, "next_reassessment_due": nxt,
                "reassessment_overdue": "yes" if overdue else "no",
                "repeat_set_complete": "yes" if complete else "no"}

    def render(self, f, rng, surface):
        p = f["protocol"]
        unit, iv = p["unit"], f["intervals_min"]
        rules = _adult_rules(p, rng, ["level_1", "level_2", "resources", "danger_zone", "age_pregnancy", "equity"],
                             "retriage")
        rules["reassessment"] = (f"Reassess every {iv['2']} min (L2), {iv['3']} (L3), {iv['4']} (L4), {iv['5']} (L5) "
                                 f"from triage or the last documented check; re-triage on current findings; L1-2 go to "
                                 f"a treatment space at once.")
        rules["repeat_set"] = "Repeat set = HR, RR, BP, SpO2 (+ temperature if febrile at triage)."
        title = _protocol_title(f["facility"], p, rng, "retriage")
        tri, now, last = f["triage_time"], f["check_time"], f["last_check"]
        report = f["event_text"] + "." + (" " + f["remark"] if f["remark"] else "")
        report = noisy(report[0].upper() + report[1:], rng, 0.12)
        obs = pick(rng, OBS_DISTRESS if f["distress_now"] else OBS_CALM)
        pain1 = f"{f['pain_now']}/10"
        if surface == "json":
            return dumps(prune({
                "patient": dict(_pt_fields(f["patient"], "retriage"), encounter=f["encounter"], chief_complaint=f["cc"],
                                triage_history=_voice_text(f["detail"], "retriage")),
                "triage": {"time": _hm(tri), "nurse": f["nurse"], "assigned_level": f["assigned_level"],
                           "vitals": _vs_dict(f["vitals_triage"], unit), "pain": f"{f['pain_triage']}/10"},
                "waiting_room_check": {"time": _hm(now), "by": f["checker"],
                                       "previous_reassessment": _hm(last) if last else "none since triage",
                                       "patient_report": report, "appearance": obs, "pain": pain1,
                                       "mental_status": _mental_text(rng, f["mental_now"], "the companion"),
                                       "vitals": _vs_dict(f["vitals_now"], unit)},
                "protocol": dict(rules, title=title),
            }))
        if surface == "reassessment_flowsheet":
            def row(label, t, v, pain):
                return [label, _hm(t), v["hr"] if v.get("hr") is not None else "--",
                        f"{v['sbp']}/{v['dbp']}" if v.get("sbp") is not None else "--",
                        v["rr"] if v.get("rr") is not None else "--",
                        f"{v['spo2']}%" if v.get("spo2") is not None else "--",
                        _temp_txt(v.get("temp_c"), unit) or "--", pain]
            rows = [row("Triage", tri, f["vitals_triage"], f"{f['pain_triage']}/10")]
            if last:
                rows.append(["Reassess", _hm(last), "", "", "", "", "", "(note: 'no change')"])
            rows.append(row("Now", now, f["vitals_now"], pain1))
            head = kv_lines([("Encounter", f["encounter"]), ("Patient", _pt_line(f["patient"], "retriage")),
                             ("Complaint", f["cc"]),
                             ("Assigned level at triage", f["assigned_level"]), ("Triage nurse", f["nurse"]),
                             ("Checked by", f["checker"])], rng)
            return (f"WAITING ROOM REASSESSMENT\n{head}\n\n"
                    + table(["", "time", "HR", "BP", "RR", "SpO2", "Temp", "pain"], rows, rng)
                    + f"\n\nPatient report: {report}\nAppearance: {obs}\n\n{title}\n"
                    + "\n".join(f"- {t}" for t in rules.values()))
        lines = [f"Waiting-room check {_hm(now)} -- {f['checker']}", f"Pt {_pt_line(f['patient'], 'retriage')}",
                 f"{f['encounter']}, triaged {_hm(tri)} as Level {f['assigned_level']} ({f['cc']}; "
                 f"{_voice_text(f['detail'], 'retriage')}).",
                 "Last reassessment: " + (_hm(last) if last else "none documented since triage") + ".",
                 f"Now: {report} {obs[0].upper() + obs[1:]}. Pain {pain1}. "
                 f"{_cap(_mental_text(rng, f['mental_now'], 'the companion'))}.",
                 f"Repeat VS: {_vs_line(f['vitals_now'], unit, rng)}",
                 f"(Triage VS: {_vs_line(f['vitals_triage'], unit, rng)})", "", title + ":",
                 bullets(list(rules.values()), rng)]
        return "\n".join(lines)

    def questions(self):
        return {
            "retriage_call": choice("How should the patient be re-triaged at this check?", {
                "downgrade_or_fast_track": "Lower the level, or move the patient to fast track, because the current "
                                           "findings place them lower than assigned.",
                "keep_current_level": "Keep the assigned level and continue reassessing at its interval.",
                "upgrade_in_queue": "Raise the level and move the patient up the queue, still in the waiting area.",
                "upgrade_room_now": "Raise the level and bring the patient into a treatment space immediately.",
            }),
            "condition_trend": score("How has the patient's condition changed since triage?", [
                "Improved: symptoms or vital signs are better than at triage.",
                "Unchanged: nothing material has moved.",
                "Worse but contained: a symptom or vital sign has worsened without reaching a written trigger.",
                "Deteriorating: a new danger-zone vital sign or a red-flag symptom has appeared.",
            ]),
            "next_reassessment_due": score(
                "How soon must this patient next be reassessed, given the level that applies now?", [
                    "Two hours or more from now.",
                    "Within about an hour to ninety minutes.",
                    "Within thirty to forty-five minutes.",
                    "Within fifteen minutes, or the patient should not stay in the waiting room at all.",
                ]),
            "reassessment_overdue": noul(
                "This check came later than the protocol interval for the level assigned at triage.",
                yes="More time had passed since triage or the last documented reassessment than the written "
                    "interval for the assigned level allows.",
                no="The check happened within the written interval for the assigned level."),
            "repeat_set_complete": noul(
                "The vital signs recorded at this check make up a complete repeat set as the protocol defines it."),
        }


# =========================================================================== schema 4
AUDIT_OUTCOMES = ["discharged home after review", "admitted to a medical ward", "admitted to intensive care",
                  "admitted to the short-stay unit", "transferred for specialist care", "discharged with follow-up"]


class TriageLevelAudit(Schema):
    name = "triage_level_audit"
    description = ("A completed triage record pulled for quality review: was the level right under the written "
                   "protocol, judged on what was known at triage?")
    surfaces = {"json": 0.55, "audit_worksheet": 0.225, "educator_email": 0.225}

    DOC_ITEMS = ["hr", "rr", "bp", "spo2", "temp", "pain", "mental_status", "history_detail"]

    def sample_factors(self, rng):
        reg = regime(rng)
        p = _adult_protocol(rng)
        targets = {"1": 5, "2": pick(rng, [10, 15]), "3": pick(rng, [30, 60]), "4": pick(rng, [60, 90, 120]),
                   "5": pick(rng, [120, 180, 240])}
        if reg == "clear":
            scen = pick_w(rng, {"match": 1.3, "under2": 1})
        elif reg == "borderline":
            scen = pick_w(rng, {"match": 1.5, "under1": 2.2, "over1": 1.3, "under2": 0.4})
        else:
            scen = pick_w(rng, {"match": 1.5, "under1": 1.5, "over1": 1, "under2": 0.9})
        # an under-triage scenario needs a presentation the protocol puts at Level 4 or more acute
        need = {"under1": 4, "under2": 3}.get(scen, 5)
        for _ in range(8):
            fam = pick_w(rng, {"cardiac": 3, "infection": 3, "neuro": 1, "limb": 2, "other": 4, "critical": 0.4})
            cx = _draw_adult(rng, fam, reg)
            v, _ = _shape_adult_vitals(rng, p, cx, "clear" if reg == "clear" else "borderline")
            mental, airway = (_apply_crit(rng, p, cx, reg, v) if fam == "critical" else (None, False))
            mental = mental or ("new_confusion" if fam == "infection" and chance(rng, 0.1) else "alert")
            lo, hi = cx["pain_range"]
            pain = rng.randint(lo, hi)
            distress = chance(rng, 0.6) if pain >= 7 else chance(rng, 0.05)
            if reg == "borderline" and hi >= 6 and chance(rng, 0.4):
                pain, distress = p["l2_pain"] + pick(rng, [-1, 0]), True
            lkw_h = _lkw_hours(rng, p, reg) if fam == "neuro" else None
            # documentation gaps (vitals that are missing are missing for the auditor too)
            k = pick_w(rng, {0: 4, 1: 3, 2: 1.6, 3: 1, 4: 0.55, 5: 0.25})
            missing = rng.sample(self.DOC_ITEMS, k)
            if k >= 4 and sum(1 for m in missing if m in ("hr", "rr", "bp", "spo2")) < 2:
                missing = ["hr", "spo2"] + [m for m in missing if m not in ("hr", "spo2")][:k - 2]
            vd = dict(v)
            for m in missing:
                if m == "bp":
                    vd["sbp"], vd["dbp"] = None, None
                elif m == "temp":
                    vd["temp_c"] = None
                elif m in ("hr", "rr", "spo2"):
                    vd[m] = None
            pain_d = None if "pain" in missing else pain
            pt = _patient(rng, cx, reg, p)
            correct = _adult_level(p, vd, mental, airway, cx["high_risk"], pain_d, distress, len(cx["resources"]), pt)
            if correct <= need:
                break
        n_vit_missing = sum(1 for m in missing if m in ("hr", "rr", "bp", "spo2"))
        if n_vit_missing >= 2:
            scen = "match"
        if pt["language"] != "English" and pt["interpreter"] in ("family_member", "none"):
            missing = missing + ["qualified_interpreter"]   # an equity element of the record
        shift = {"match": 0, "under1": 1, "under2": 2, "over1": -1}[scen]
        assigned = correct + shift
        if assigned > 5:
            assigned = 5 if correct < 5 else correct - 1
        if assigned < 1:
            assigned = correct + 1
        tgt = targets[str(correct)]
        if reg == "clear":
            wait = (rng.randint(1, max(2, tgt)) if assigned <= correct
                    else int(tgt * rng.uniform(2.5, 5)) + rng.randint(5, 20))
        elif reg == "borderline":
            wait = max(1, near(rng, tgt, max(6, tgt * 0.3), integer=True))
        else:
            wait = max(1, int(tgt * rng.uniform(0.3, 3.0)))
        if correct == 1:
            wait = min(wait, rng.randint(2, 25))
        outcome = pick(rng, AUDIT_OUTCOMES)
        comment = None
        if reg == "conflicting":
            if assigned == correct:
                outcome = pick(rng, ["admitted to intensive care", "transferred for specialist care"])
                comment = pick(rng, ["Physician comment: 'should have been triaged higher given where they ended up.'",
                                     "Ward feedback: 'this patient was much sicker than the triage level suggested.'"])
            elif assigned > correct:
                outcome = pick(rng, ["discharged home after review", "discharged with follow-up"])
                comment = pick(rng, ["Charge nurse: 'went home the same night, no harm done.'",
                                     "Triage nurse note: 'busy shift; level felt right at the time.'",
                                     "Charge nurse: 'frequent attender, uninsured, usually nothing serious.'",
                                     "Triage nurse note: 'hard to get a history, language barrier, seemed fine.'"])
        arr = rand_ts(rng)
        return {"regime": reg, "facility": _facility(rng), "protocol": p, "targets_min": targets, **cx,
                "vitals_documented": vd, "pain_documented": pain_d, "distress": distress, "mental": mental,
                "airway_support_needed": airway, "lkw_h": lkw_h, "missing_items": missing,
                "protocol_level": correct, "assigned_level": assigned, "wait_to_clinician_min": wait,
                "hindsight_outcome": outcome, "reviewer_comment": comment, "arrival": arr,
                "triage_time": arr + timedelta(minutes=rng.randint(1, max(1, min(18, wait - 1)))),
                "triage_nurse": _nurse(rng),
                "auditor": person(rng), "record_id": f"TQA-{rng.randint(1000, 9999)}", "encounter": _enc(rng, arr),
                "patient": pt}

    def intended(self, f):
        miss = f["missing_items"]
        m = len(miss)
        doc = 3 if m == 0 else (2 if m == 1 else (1 if m <= 3 else 0))
        nv = sum(1 for x in miss if x in ("hr", "rr", "bp", "spo2"))
        a, c = f["assigned_level"], f["protocol_level"]
        if nv >= 2:
            finding = "cannot_determine_from_record"
        elif a == c:
            finding = "correct_level"
        elif a > c:
            finding = "under_triaged"
        else:
            finding = "over_triaged"
        under = a - c
        beyond = f["wait_to_clinician_min"] > f["targets_min"][str(c)]
        if under <= 0:
            impact = 0
        elif not beyond:
            impact = 1
        else:
            impact = 2 if under == 1 else 3
        if impact == 3:
            follow = "safety_event_report"
        elif impact == 2:
            follow = "qi_case_review"
        elif finding != "correct_level" or doc <= 1:
            follow = "individual_feedback"
        else:
            follow = "no_further_action"
        return {"audit_finding": finding, "review_follow_up": follow, "care_delay_impact": impact,
                "documentation_quality": doc, "waited_beyond_target": "yes" if beyond else "no"}

    def _record(self, f, rng):
        unit = f["protocol"]["unit"]
        miss = set(f["missing_items"])
        hist = (_voice_text(f["detail"], "audit") if "history_detail" not in miss
                else pick(rng, ["no history recorded beyond the complaint", "history field left blank"]))
        if f["lkw_h"] is not None or f["family"] == "neuro":
            hist += f"; last known well {_lkw_text(f['lkw_h'], f['triage_time'], rng)}"
        mental = None if "mental_status" in miss else _mental_text(rng, f["mental"], "family")
        return {"chief_complaint": f["cc"], "history": hist,
                "appearance": pick(rng, OBS_DISTRESS if f["distress"] else OBS_CALM),
                "mental_status": mental,
                "pain": f"{f['pain_documented']}/10" if f["pain_documented"] is not None else None,
                "vitals": _vs_dict(f["vitals_documented"], unit)}

    def render(self, f, rng, surface):
        p, tg = f["protocol"], f["targets_min"]
        rules = _adult_rules(p, rng, ["level_1", "level_2", "resources", "danger_zone", "age_pregnancy", "equity"],
                             "audit")
        rules["clinician_targets"] = (f"Time from arrival to clinician: Level 1 within {tg['1']} min, Level 2 "
                                      f"{tg['2']} min, Level 3 {tg['3']} min, Level 4 {tg['4']} min, Level 5 "
                                      f"{tg['5']} min.")
        rules["audit_rule"] = pick(rng, ["Audit against what was documented at triage, not against later results.",
                                         "Judge the level on the information available at triage; hindsight does "
                                         "not change the correct level."])
        title = _protocol_title(f["facility"], p, rng, "audit")
        rec = self._record(f, rng)
        seen = f["arrival"] + timedelta(minutes=f["wait_to_clinician_min"])
        if surface == "json":
            return dumps(prune({
                "audit": {"id": f["record_id"], "auditor": f["auditor"], "encounter": f["encounter"]},
                "triage_record": dict(rec, nurse=f["triage_nurse"], assigned_level=f["assigned_level"],
                                      triage_time=_hm(f["triage_time"]), patient=_pt_fields(f["patient"], "audit")),
                "timeline": {"arrived": _hm(f["arrival"]), "seen_by_clinician": _hm(seen),
                             "minutes_to_clinician": f["wait_to_clinician_min"]},
                "later_course": {"outcome": f["hindsight_outcome"], "comment": f["reviewer_comment"]},
                "protocol": dict(rules, title=title),
            }))
        vit = rec["vitals"]
        lab = {"hr": "HR {}", "bp": "BP {}", "rr": "RR {}", "spo2_pct": "SpO2 {}%", "temp_c": "T {} °C",
               "temp_f": "T {} °F"}
        vit_line = ", ".join(lab[k].format(vit[k]) for k in lab if vit.get(k) is not None)
        not_doc = [{"hr": "heart rate", "rr": "resp rate", "bp": "blood pressure", "spo2": "SpO2",
                    "temp": "temperature", "pain": "pain score", "mental_status": "mental status",
                    "history_detail": "history detail", "qualified_interpreter": "qualified interpreter use"}[m]
                   for m in f["missing_items"]]
        if surface == "audit_worksheet":
            rows = [("Audit", f["record_id"]), ("Encounter", f["encounter"]),
                    ("Patient", _pt_line(f["patient"], "audit")),
                    ("Triage nurse", f["triage_nurse"]),
                    ("Assigned level", f["assigned_level"]), ("Chief complaint", rec["chief_complaint"]),
                    ("History", rec["history"]), ("Appearance", rec["appearance"]),
                    ("Mental status", rec["mental_status"] or "not documented"),
                    ("Pain", rec["pain"] or "not documented"), ("Vitals", vit_line or "none documented"),
                    ("Not documented", ", ".join(not_doc) if not_doc else "nothing"),
                    ("Timeline", f"arrived {_hm(f['arrival'])}, triaged {_hm(f['triage_time'])}, seen by "
                                 f"clinician {_hm(seen)}"),
                    ("Later course", f["hindsight_outcome"]), ("Comment on file", f["reviewer_comment"])]
            return (f"TRIAGE QUALITY AUDIT WORKSHEET -- {f['facility']}\n" + kv_lines(rows, rng)
                    + f"\n\n{title}:\n" + "\n".join(f"* {t}" for t in rules.values()))
        body = [f"Subject: Triage audit {f['record_id']} ({f['encounter']})", f"From: {f['auditor']}",
                "To: ED triage quality group", "",
                noisy(join_sentences([
                    f"Pulled this one from the random sample. Triaged at {_hm(f['triage_time'])} by "
                    f"{f['triage_nurse']} as Level {f['assigned_level']}.",
                    f"Patient: {_pt_line(f['patient'], 'audit')}.",
                    f"Complaint: {rec['chief_complaint']} -- {rec['history']}.",
                    f"Documented: {vit_line or 'no vital signs'}; pain {rec['pain'] or 'not recorded'}; "
                    f"{rec['mental_status'] or 'mental status not recorded'}; {rec['appearance']}.",
                    f"Arrived {_hm(f['arrival'])}, seen by a clinician at {_hm(seen)}.",
                    f"Later course: {f['hindsight_outcome']}.",
                    f["reviewer_comment"] or ""]), rng, 0.1), "",
                f"Protocol for reference ({title}):", bullets(list(rules.values()), rng), "", f["auditor"].split()[0]]
        return "\n".join(body)

    def questions(self):
        return {
            "audit_finding": choice("What should the audit conclude about the assigned triage level?", {
                "cannot_determine_from_record": "The record lacks the vital signs needed to say what level the "
                                                "protocol required.",
                "correct_level": "The assigned level matches what the protocol gives for the documented findings.",
                "over_triaged": "The patient was placed at a more acute level than the documented findings support.",
                "under_triaged": "The patient was placed at a less acute level than the documented findings require.",
            }),
            "review_follow_up": choice("What follow-up should the audit trigger?", {
                "individual_feedback": "Private feedback to the triage nurse on the level or the documentation.",
                "no_further_action": "File the audit; nothing needs to follow.",
                "qi_case_review": "Present the case at the department's quality-improvement review.",
                "safety_event_report": "Log a patient-safety event and start the formal review process.",
            }),
            "care_delay_impact": score("How much could the triage decision have delayed this patient's care?", [
                "Not at all: the level matched or erred toward faster care.",
                "Slightly: placed too low, but still seen within the correct level's target time.",
                "Materially: one level too low, and the wait ran past the correct level's target.",
                "Seriously: two or more levels too low, and the wait ran past the correct level's target.",
            ]),
            "documentation_quality": score("How complete is the triage documentation?", [
                "Unusable: four or more required elements are missing.",
                "Major gaps: two or three required elements are missing.",
                "Minor gap: one required element is missing.",
                "Complete: every required element is recorded.",
            ]),
            "waited_beyond_target": noul(
                "The patient waited longer than the protocol's target time for the level they should have received.",
                yes="The time from arrival to clinician exceeds the written target for the level the documented "
                    "findings call for.",
                no="The patient was seen within the written target for the level the documented findings call for, "
                   "whatever level was actually assigned."),
        }


# =========================================================================== schema 5
EMS_TREATMENTS = {
    "cardiac": ["aspirin 300 mg chewed", "IV access left forearm", "12-lead ECG transmitted", "GTN spray x1"],
    "infection": ["IV access", "500 mL saline bolus given", "paracetamol IV", "blood glucose checked"],
    "neuro": ["IV access", "blood glucose checked (normal)", "pre-alert called"],
    "limb": ["limb splinted", "inhaled analgesia", "ice pack"],
    "other": ["IV access", "antiemetic given", "wound dressed", "blood glucose checked"],
    "critical": ["high-flow oxygen", "IV access x2", "airway adjunct considered", "fluids running"],
}


def _ems_treatments(rng, cx):
    """What a crew plausibly did en route for this presentation."""
    fam, res = cx["family"], cx["resources"]
    if fam == "critical":
        pool = EMS_TREATMENTS["critical"]
    elif fam == "cardiac":
        chest = any(w in cx["cc"] for w in ("chest", "abdominal", "indigestion"))
        pool = (["aspirin 300 mg chewed", "GTN spray x1"] if chest else []) + ["IV access left forearm",
                                                                             "12-lead ECG transmitted"]
    elif fam in ("infection", "neuro", "limb"):
        pool = EMS_TREATMENTS[fam]
    elif not res:
        return [pick(rng, ["observations only", "no treatment required", "reassurance and observations"])]
    else:
        pool = (["wound dressed", "pressure dressing applied"] if "procedure" in res else []) + \
               (["IV access", "antiemetic given"] if ("iv_meds" in res or "labs" in res) else []) + \
               ["blood glucose checked", "observations only"]
    return rng.sample(pool, rng.randint(1, min(3, len(pool))))


class AmbulanceOffloadTriage(Schema):
    name = "ambulance_offload_triage"
    description = ("A patient arriving by ambulance at the emergency department: acuity, where the patient goes and "
                   "whether the crew can be released under the written offload rules.")
    surfaces = {"json": 0.55, "handover_note": 0.225, "offload_log": 0.225}

    ELEMENTS = ["complaint_or_mechanism", "findings", "vital_signs", "treatment_given", "times"]

    def sample_factors(self, rng):
        reg = regime(rng)
        p = _adult_protocol(rng)
        target = pick(rng, [20, 30, 45])
        fam = pick_w(rng, {"cardiac": 2.6, "infection": 2.3, "neuro": 1.0, "limb": 3.2, "other": 4.0, "critical": 1.1})
        cx = _draw_adult(rng, fam, reg)
        v_ed, recheck = _shape_adult_vitals(rng, p, cx, reg)
        v_ed = recheck or v_ed
        mental, airway = (_apply_crit(rng, p, cx, reg, v_ed) if fam == "critical" else (None, False))
        # the crew's last set, taken en route: close to the ED set; in conflicting cases it looked better
        v_pre = dict(v_ed)
        for k, d in (("hr", 8), ("rr", 2), ("spo2", 1), ("sbp", 12)):
            v_pre[k] = v_pre[k] + rng.randint(-d, d)
        v_pre["spo2"] = min(100, v_pre["spo2"])
        v_pre["temp_c"] = round(v_pre["temp_c"] + rng.uniform(-0.3, 0.2), 1)
        if reg == "conflicting" and fam != "critical":
            for k in _dz_hits(p, v_ed):
                _just_inside(v_pre, p, rng, k)
        _fix_dbp(v_pre, rng)
        mental = mental or pick_w(rng, {"alert": 8, "anxious": 2, "new_confusion": 0.5 if fam == "infection" else 0.1})
        lo, hi = cx["pain_range"]
        pain = rng.randint(lo, hi)
        distress = chance(rng, 0.6) if pain >= 7 else chance(rng, 0.05)
        lkw_h = _lkw_hours(rng, p, reg) if fam == "neuro" else None
        treatments = _ems_treatments(rng, cx)
        walks = cx["ambulatory0"] and chance(rng, 0.85)
        infusion = any(("bolus" in t or "running" in t) for t in treatments) and chance(rng, 0.45)
        oxygen = fam == "critical" or (cx["prof"] == "resp" and chance(rng, 0.5)) or chance(rng, 0.05)
        spinal = ((fam == "limb" and cx["limb_site"] in ("shoulder", "lower leg") and chance(rng, 0.25))
                  or (cx["cc"] in ("head injury", "scalp laceration") and chance(rng, 0.35)))
        if spinal:
            walks = False
        if chance(rng, 0.6 if reg != "conflicting" else 0.4) and fam in ("limb", "other", "cardiac", "infection"):
            walks, infusion, oxygen, spinal = cx["ambulatory0"], False, False, False
        crew_view = None
        if reg == "conflicting":
            crew_view = pick(rng, ["Crew: 'walked to the ambulance, should be fine for the waiting room.'",
                                   "Crew lead says the patient is 'stable, fit to sit'.",
                                   "Crew wants to hand over quickly; patient 'just needs a chair'."])
        if reg == "borderline" and chance(rng, 0.5):
            infusion = False
            treatments = [t for t in treatments if "running" not in t] + ["saline bolus completed, cannula capped"]
        arrival = rand_ts(rng)
        if reg == "clear":
            mins = pick(rng, [rng.randint(3, max(4, target - 14)), int(target * rng.uniform(2.2, 3.5))])
        elif reg == "borderline":
            mins = max(2, near(rng, target, 9, integer=True))
        else:
            mins = rng.randint(4, int(target * 2.6))
        n_miss = pick_w(rng, {0: 5, 1: 3.5, 2: 1})
        if reg == "clear":
            n_miss = pick(rng, [0, 0, 2])
        missing = rng.sample(self.ELEMENTS, n_miss)
        return {"regime": reg, "facility": _facility(rng), "protocol": p, "offload_target_min": target, **cx,
                "vitals_prehospital": v_pre, "vitals_ed": v_ed, "mental": mental, "airway_support_needed": airway,
                "pain": pain, "distress": distress, "lkw_h": lkw_h, "treatments": treatments,
                "walks_unaided": walks, "infusion_running": infusion, "oxygen_required": oxygen,
                "spinal_precautions": spinal, "crew_view": crew_view, "arrival": arrival,
                "handover_time": arrival + timedelta(minutes=mins), "minutes_at_door": mins,
                "handover_missing": missing, "call_to_door_min": rng.randint(32, 75),
                "patient": _patient(rng, cx, reg, p),
                "response_min": rng.randint(6, 18), "unit_id": f"{pick(rng, ['Medic', 'Ambulance', 'Rescue', 'A'])} "
                                                       f"{rng.randint(2, 89)}",
                "paramedic": person(rng), "receiving_nurse": _nurse(rng), "encounter": _enc(rng, arrival),
                "monitored_beds_free": pick(rng, [0, 0, 1, 1, 2, 3])}

    def _level(self, f):
        return _adult_level(f["protocol"], f["vitals_ed"], f["mental"], f["airway_support_needed"], f["high_risk"],
                            f["pain"], f["distress"], len(f["resources"]), f["patient"])

    def intended(self, f):
        p = f["protocol"]
        lvl = self._level(f)
        fit = (lvl >= 3 and f["mental"] in ("alert", "anxious") and f["walks_unaided"] and not f["infusion_running"]
               and not f["oxygen_required"] and not f["spinal_precautions"] and not _dz_hits(p, f["vitals_ed"]))
        if lvl == 1:
            dest = "resuscitation_bay"
        elif lvl == 2:
            dest = "monitored_bed"
        else:
            dest = "waiting_room_fit_to_sit" if fit else "ambulance_offload_area"
        m, t = f["minutes_at_door"], f["offload_target_min"]
        clock = 3 if m > 2 * t else (2 if m > t else (1 if m >= t - 10 else 0))
        return {"arrival_acuity": 5 - lvl, "offload_clock_pressure": clock, "offload_destination": dest,
                "fit_to_sit": "yes" if fit else "no",
                "handover_complete": "no" if f["handover_missing"] else "yes"}

    def _handover(self, f, rng):
        unit = f["protocol"]["unit"]
        miss = set(f["handover_missing"])
        h = {}
        if "complaint_or_mechanism" not in miss:
            h["complaint_or_mechanism"] = f"{f['cc']}: {_voice_text(f['detail'], 'offload')}"
        if "findings" not in miss:
            bits = [_mental_text(rng, f["mental"], "family")]
            bits.append("walks unaided" if f["walks_unaided"] else "not walking, stretcher throughout")
            if f["spinal_precautions"]:
                bits.append("spinal precautions maintained")
            if f["airway_support_needed"]:
                bits.append("stridor, struggling to speak")
            h["findings"] = "; ".join(bits)
        if "vital_signs" not in miss:
            h["vital_signs"] = _vs_line(f["vitals_prehospital"], unit, rng)
        if "treatment_given" not in miss:
            h["treatment_given"] = ", ".join(f["treatments"])
        if "times" not in miss:
            call = f["arrival"] - timedelta(minutes=f["call_to_door_min"])
            t = f"call {_hm(call)}, on scene {_hm(call + timedelta(minutes=f['response_min']))}"
            if f["family"] == "neuro":
                t += f", last known well {_lkw_text(f['lkw_h'], f['arrival'], rng)}"
            h["times"] = t
        return h

    def render(self, f, rng, surface):
        p, unit, t = f["protocol"], f["protocol"]["unit"], f["offload_target_min"]
        rules = _adult_rules(p, rng, ["level_1", "level_2", "resources", "danger_zone", "age_pregnancy", "equity"],
                             "offload")
        rules["fit_to_sit"] = ("[SIT] Level 3-5, alert, walks unaided or with own aid, no infusion, no oxygen, no "
                               "spinal precautions, no DZ vital")
        rules["offload"] = (f"[OFFLOAD] crew released within {t} min of arrival at the door; L1 resus, L2 monitored bed, "
                            f"others not fit to sit wait in the offload area on an ED stretcher")
        rules["handover"] = "[MIST] complaint or mechanism, findings, vital signs, treatment, times (onset or LKW)"
        title = _protocol_title(f["facility"], p, rng, "offload")
        h = self._handover(f, rng)
        ed_vs = _vs_dict(f["vitals_ed"], unit)
        status = {"infusion_running": f["infusion_running"], "on_oxygen": f["oxygen_required"],
                  "spinal_precautions": f["spinal_precautions"], "walks_unaided": f["walks_unaided"]}
        if surface == "json":
            return dumps(prune({
                "arrival": {"unit": f["unit_id"], "paramedic": f["paramedic"], "at_door": _hm(f["arrival"]),
                            "handover_now": _hm(f["handover_time"]), "encounter": f["encounter"]},
                "handover": dict(h, crew_comment=f["crew_view"]),
                "patient": _pt_fields(f["patient"], "offload"),
                "ed_triage": {"nurse": f["receiving_nurse"], "vitals": ed_vs, "pain": f"{f['pain']}/10",
                              "appearance": pick(rng, OBS_DISTRESS if f["distress"] else OBS_CALM),
                              "patient_status": status},
                "department": {"monitored_beds_free": f["monitored_beds_free"]},
                "protocol": dict(rules, title=title),
            }))
        if surface == "handover_note":
            lines = [f"AMBULANCE HANDOVER -- {f['unit_id']} to {f['facility']} ED",
                     f"At door {_hm(f['arrival'])}; handover {_hm(f['handover_time'])}. Paramedic {f['paramedic']}, "
                     f"received by {f['receiving_nurse']}.", f"Pt: {_pt_line(f['patient'], 'offload')}"]
            labels = {"complaint_or_mechanism": "M (complaint/mechanism)", "findings": "I (findings)",
                      "vital_signs": "S (signs)", "treatment_given": "T (treatment)", "times": "Times"}
            for k in self.ELEMENTS:
                if k in h:
                    lines.append(f"{labels[k]}: {h[k]}")
            if f["crew_view"]:
                lines.append(f["crew_view"])
            lines += [f"ED triage VS: {_vs_line(f['vitals_ed'], unit, rng)}; pain {f['pain']}/10",
                      "Status: " + ", ".join([("infusion running" if f["infusion_running"] else "no infusion"),
                                              ("on oxygen" if f["oxygen_required"] else "room air"),
                                              ("spinal precautions" if f["spinal_precautions"] else "no spinal "
                                                                                                    "precautions"),
                                              ("walks unaided" if f["walks_unaided"] else "stretcher-bound")]),
                      f"Monitored beds free: {f['monitored_beds_free']}", "", title + ":",
                      bullets(list(rules.values()), rng)]
            return "\n".join(lines)
        m = f["minutes_at_door"]
        at = lambda k: f["arrival"] + timedelta(minutes=min(k, max(0, m - 1)))  # noqa: E731
        ev = [(f["arrival"], f"{f['unit_id']} at ambulance bay door")]
        if "vital_signs" in h:
            ev.append((at(1), f"crew VS (en route): {h['vital_signs']}"))
        ev += [(at(3), "triage nurse at stretcher"), (at(6), f"ED VS: {_vs_line(f['vitals_ed'], unit, rng)}")]
        if m > 16:
            ev.append((at(m // 2), pick(rng, ["no offload space, crew still with patient",
                                               "crew waiting on stretcher in corridor", "offload area full"])))
        ev.append((f["handover_time"], "handover in progress"))
        log = "\n".join(f"{_hm(t_)}  {e}" for t_, e in ev)
        rest = kv_lines([("Patient", _pt_line(f["patient"], "offload"))]
                        + [(k.replace("_", " ").capitalize(), v_) for k, v_ in h.items() if k != "vital_signs"]
                        + [("Crew comment", f["crew_view"]), ("Pain", f"{f['pain']}/10"),
                           ("Infusion running", "yes" if f["infusion_running"] else "no"),
                           ("Oxygen", "yes" if f["oxygen_required"] else "no"),
                           ("Spinal precautions", "yes" if f["spinal_precautions"] else "no"),
                           ("Walks unaided", "yes" if f["walks_unaided"] else "no"),
                           ("Monitored beds free", f["monitored_beds_free"])], rng)
        return (f"OFFLOAD LOG {f['encounter']} ({f['facility']})\n{log}\n\n{rest}\n\n{title}\n"
                + "\n".join(f"- {r}" for r in rules.values()))

    def questions(self):
        return {
            "arrival_acuity": score("How acute is this ambulance patient under the written triage protocol?", [
                "Level 5: stable, no resource type expected.",
                "Level 4: stable, one resource type expected.",
                "Level 3: two or more resource types and no danger-zone vital sign.",
                "Level 2: high-risk, confused or lethargic, severe pain with distress, or danger-zone vitals with "
                "several resources expected.",
                "Level 1: needs an immediate life-saving intervention.",
            ]),
            "offload_clock_pressure": score("How pressing is the offload clock for this ambulance crew?", [
                "Comfortable: well inside the offload target.",
                "Tight: within ten minutes of the offload target.",
                "Overdue: past the target; the charge nurse must step in.",
                "Badly overdue: more than double the target; escalate beyond the department.",
            ]),
            "offload_destination": choice("Where should the patient go from the ambulance stretcher?", {
                "ambulance_offload_area": "To the offload area on an ED stretcher, under an offload nurse, so the "
                                          "crew can leave.",
                "monitored_bed": "Into a monitored acute-care bed straight away.",
                "resuscitation_bay": "Straight into a resuscitation bay.",
                "waiting_room_fit_to_sit": "To the waiting room in a chair, as fit to sit.",
            }),
            "fit_to_sit": noul(
                "The patient meets every written criterion to wait in the waiting room in a chair.",
                yes="Level 3-5, alert, walking unaided, no infusion, oxygen or spinal precautions, and no danger-zone "
                    "vital sign.",
                no="At least one written fit-to-sit criterion fails, whatever the crew's view."),
            "handover_complete": noul("The paramedic handover covers every element the protocol requires."),
        }


WORKFLOW = Workflow(
    name="ed_triage_acuity",
    title="ED triage acuity",
    description="Which acuity level, stream and standing order a written emergency-department triage protocol gives "
                "a patient, and how waiting, audited and ambulance-delivered patients are handled under it.",
    domain="health-care operations / emergency department",
    schemas=[AdultWalkinTriage(), PediatricTriage(), WaitingRoomRetriage(), TriageLevelAudit(),
             AmbulanceOffloadTriage()],
    notes="All hospitals, protocols, thresholds, staff and encounters are fictional; the five-level scale is "
          "ESI-style in structure only, with invented limits that vary by facility, and is not clinical guidance. "
          "Adult states render no age or sex; pediatric age is used only by the protocol's age bands, infant-fever "
          "rule and weight check. No EMS call-taking, no hospital capacity decisions, no lab or drug-safety review.",
)
