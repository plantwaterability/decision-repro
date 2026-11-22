"""EMS dispatch priority: what call priority and response a written dispatch guide gives a call, from the
call-taker's notes, and how dispatchers handle updates, transfers and a queue of pending calls.

Everything here is fictional: the counties and communications centers, the dispatch guides and every threshold in
them, the units, streets, staff and callers. Each guide uses an invented five-level priority scale (P1 most
urgent ... P5 referral or no ambulance) with complaint cards, response targets, transfer crew matrices and queue
rules written for this generator. The limits vary between the fictional agencies. None of it is real dispatch or
clinical guidance, and it does not reproduce any real dispatch protocol's determinant codes or wording.

Each schema draws a `regime` first ('clear' / 'borderline' / 'conflicting'). Borderline cases sit on a written
limit (a seizure at the card's minute mark, a fall height at the threshold, a crew arrival two minutes either side
of the target); conflicting cases carry a caller who downplays what the answers show, a downgrade offered by a
source the guide does not accept, or a requested crew level that does not cover the patient's infusions.

No patient age, sex or other protected attribute is a decision factor; relationship words in callers' speech are
colour only. Every schema defines `intended(f) -> {qid: answer}` (choice key, "yes"/"no", or score level index),
the written guide applied to the factors. It is used only to check answer spread; it is never rendered.
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

# =========================================================================== shared pools
COUNTY_STEMS = ["Wexmoor", "Tallowin", "Quenby Flats", "Orchard Vale", "Saltreach", "Brindle Creek", "Coldspring",
                "Marrowby", "Ketterly", "Pinnock", "Drayfield", "Ulverbrook", "Ashcombe Fen", "Lowmarsh",
                "Heronsgate", "Vantry", "Esk Hollow", "Fenwick Downs", "Galloway Rise", "Ironsedge", "Morrow Bay",
                "Thornapple", "Cobblestone", "Larchmont Vale", "Redwater", "Siskin Falls", "Hartsell", "Upper Brannock"]
STREET_A = ["Tallow", "Brindle", "Quarry", "Heron", "Saltreach", "Orchard", "Larch", "Mill Pond", "Kettle", "Foundry",
            "Granary", "Wicket", "Spindle", "Copper Beech", "Haverly", "Juniper Hill", "Sexton", "Weir", "Bramble",
            "Lantern", "Cobbler", "Fairground", "Old Turnpike", "Ridgeback", "Pewter", "Cinder", "Marlow", "Sandpiper",
            "Thresher", "Vicarage", "Osprey", "Gristmill", "Hawthorn", "Ferrier", "Lockkeeper", "Tannery"]
STREET_B = ["St", "Ave", "Rd", "Ln", "Dr", "Ct", "Way", "Blvd", "Pl", "Terrace"]
PLACES = ["kitchen", "bathroom", "front garden", "living room", "bedroom", "hallway", "driveway", "back yard",
          "office", "warehouse floor", "shop", "car park", "bus shelter", "gym changing room", "stairwell"]


LANGUAGES = ["Spanish", "Vietnamese", "Arabic", "Polish", "Haitian Creole", "Somali", "Mandarin", "Tagalog",
             "Portuguese", "Russian", "Korean", "Amharic", "Punjabi", "Ukrainian", "Dari", "French"]


def _county(rng):
    return pick(rng, COUNTY_STEMS)


def _street(rng):
    return f"{pick(rng, STREET_A)} {pick(rng, STREET_B)}"


def _address(rng):
    return f"{rng.randint(2, 9899)} {_street(rng)}"


def _hms(t):
    return t.strftime("%H:%M:%S")


def _hm(t):
    return t.strftime("%H:%M")


def _cad(rng, t):
    return f"{t.strftime('%y')}-{rng.randint(1000, 99999):05d}"


def _unit_names(rng, kinds):
    """Distinct unit names for a list of kinds ('als', 'bls', 'engine')."""
    nums = rng.sample(range(2, 79), len(kinds))
    label = {"als": "Medic", "bls": "Amb", "engine": "Engine"}
    return [f"{label[k]} {n}" for k, n in zip(kinds, nums)]


def _cap(text):
    return text[:1].upper() + text[1:] if text else text


# =========================================================================== schema 1: medical call priority
def C(says, bases, flags=(), unc=False, num=None, card="", q="", referral=False, unsafe=0.03):
    """A complaint card. bases: [(narrative fragment, base priority)]; flags: [(key, caller words, priority)];
    num: (factor key, guide parameter, comparison 'ge'/'le', priority when met)."""
    return {"says": says, "bases": bases, "flags": list(flags), "unc": unc, "num": num, "card": card, "q": q,
            "referral": referral, "unsafe": unsafe}


MED_CARDS = {
    "chest_pain": C(["chest pain", "pain in the chest", "tight chest", "crushing feeling in the chest"], [("", 3)],
                    [("sweating", "sweating and clammy", 2), ("breathless", "short of breath with it", 2),
                     ("heart_history", "has had a heart attack before", 2)], unc=True,
                    card="Chest pain: P2 if sweating or clammy, short of breath, or known heart disease; otherwise P3.",
                    q="sweating / short of breath / heart history?"),
    "breathing": C(["trouble breathing", "can't catch their breath", "breathing problems", "asthma attack"], [("", 3)],
                   [("no_sentences", "cannot finish a sentence", 2), ("blue_lips", "lips going blue-grey", 1),
                    ("inhaler_failed", "inhaler is not helping", 2)], unc=True,
                   card="Breathing problems: P1 if lips or face blue/grey; P2 if unable to speak full sentences or the "
                        "reliever is not working; otherwise P3.", q="full sentences? colour? inhaler working?"),
    "stroke": C(["face drooping", "slurred speech and a weak arm", "can't move one side", "speech has gone strange"],
                [("", 3)], unc=True, num=("onset_h", "stroke_h", "le", 2),
                card="Stroke signs (face, arm, speech): P2 if they began within {stroke_h:g} h or the start time is "
                     "unknown; P3 if longer ago.", q="when did it start?"),
    "fall": C(["fell down", "had a fall", "fell off a ladder", "slipped and fell"], [("", 4)],
              [("hurt_stuck", "hurt and cannot get up", 3), ("head_thinner", "hit their head and takes blood thinners",
                                                                2), ("deformity", "leg looks bent the wrong way", 3)],
              unc=True, num=("height_m", "fall_m", "ge", 2),
              card="Falls: P2 from {fall_m:g} m or higher, or a head strike on blood thinners; P3 if hurt and unable "
                   "to get up, or an obvious deformity; uninjured lift assist P4.",
              q="how far? hurt? head? blood thinners?"),
    "abdominal": C(["stomach pain", "belly pain", "bad pain in the stomach"],
                   [("", 3), ("mild, on and off for a week", 4)],
                   [("near_faint", "nearly passed out", 2), ("vomit_blood", "vomiting blood", 2)],
                   card="Abdominal pain: P2 with fainting or near-fainting, or vomiting blood; P3 otherwise; mild "
                        "pain for days P4.", q="fainting? vomiting blood?"),
    "bleeding": C(["bleeding badly", "cut and bleeding", "won't stop bleeding"], [("", 3)],
                  [("spurting", "blood is spurting out", 1), ("not_stopping", "not stopping with firm pressure", 2),
                   ("thinner", "on blood thinners", 2)],
                  card="Bleeding: P1 if spurting; P2 if not controlled by firm pressure or on blood thinners; "
                       "otherwise P3.", q="spurting? stopping with pressure? blood thinners?"),
    "seizure": C(["having a fit", "seizure", "shaking all over"], [("known epilepsy", 3), ("", 3)],
                 [("first_seizure", "never had one before", 2)], unc=True, num=("seizing_min", "seizure_min", "ge", 1),
                 card="Seizures: P1 if still fitting at {seizure_min} min or longer; P2 for a first-ever seizure; P3 "
                      "for a fit that has stopped.", q="still fitting? how long? first ever?"),
    "diabetic": C(["diabetic and not right", "sugar is low", "diabetic, acting strange"],
                  [("", 3), ("alert now and eating a sandwich", 4)], [("cannot_swallow", "too drowsy to eat or drink",
                                                                       2)], unc=True,
                  card="Diabetic problems: P2 if too drowsy to swallow; P3 otherwise; P4 if alert and already eating.",
                  q="able to eat or drink?"),
    "allergy": C(["allergic reaction", "came out in hives", "reaction to a bee sting"], [("", 3)],
                 [("throat", "says their throat is closing", 1), ("lips", "lips and tongue swelling", 2)],
                 card="Allergic reactions: P1 if the throat is closing or breathing is difficult; P2 with lip or "
                      "tongue swelling; hives alone P3.", q="breathing? lips or tongue?"),
    "overdose": C(["took too many tablets", "overdose", "swallowed a lot of pills"], [("", 3)],
                  [("drowsy", "getting drowsy", 2), ("intentional", "took them on purpose", 2)], unc=True,
                  card="Overdose or poisoning: P2 if drowsy or intentional; accidental and alert P3.",
                  q="drowsy? on purpose?", unsafe=0.08),
    "back_pain": C(["back pain", "can't move with back pain"], [("", 4)],
                   [("leg_numb", "legs going numb", 3), ("fainting", "fainted with the pain", 2)],
                   card="Back pain: P2 with fainting; P3 with new leg numbness or weakness; otherwise P4.",
                   q="numbness? fainting?"),
    "sick": C(["not well", "generally unwell", "feeling ill", "weak and unwell"], [("", 4)],
              [("hot_confused", "hot and confused", 2), ("cannot_stand", "can't stand up", 3)], unc=True,
              card="Sick person: P2 if feverish and confused; P3 if unable to stand; otherwise P4.",
              q="confused? able to stand?"),
    "headache": C(["bad headache", "head pain"], [("", 4)],
                  [("sudden", "came on suddenly, worst ever", 2), ("weak_arm", "an arm has gone weak with it", 2)],
                  card="Headache: P2 if sudden and severe or with weakness; otherwise P4.", q="sudden? weakness?"),
    "heat": C(["overheated", "collapsed in the heat"], [("", 3)], [("confused", "confused, not making sense", 2)],
              unc=True, card="Heat illness: P2 if confused; otherwise P3.", q="confused?"),
    "psych": C(["mental health crisis", "very distressed", "talking about ending it"], [("", 3)],
               [("means", "has tablets in hand and says they will take them", 2)],
               card="Mental health: P2 if threatening self-harm with the means at hand; otherwise P3.",
               q="means at hand?", unsafe=0.35),
    "assault": C(["assaulted", "beaten up", "attacked in the street"], [("", 3)],
                 [("stab_shot", "stab wound to the side", 1), ("head_drowsy", "head injury and getting drowsy", 2)],
                 unc=True, card="Assault: P1 for stab or gunshot wounds; P2 for a head injury with drowsiness; "
                                "otherwise P3.", q="weapon wounds? head injury?", unsafe=0.4),
    "burn": C(["burned", "scalded"], [("", 3)],
              [("airway_burn", "burns around the mouth, voice gone hoarse", 1),
               ("large", "the burn covers most of an arm or more", 2)],
              card="Burns: P1 with burns to the mouth or airway; P2 if larger than a whole arm; otherwise P3.",
              q="face or mouth? how big?"),
    "eye": C(["eye injury", "something in the eye"], [("", 4)],
             [("chemical", "chemical splashed into it", 3), ("penetrating", "something stuck in the eye", 3)],
             card="Eye problems: P3 for a chemical splash or a penetrating object; otherwise P4.",
             q="chemical? object in the eye?"),
    "cold_flu": C(["cough and fever", "flu symptoms", "bad cold"], [("", 5)], [("breathless", "getting breathless", 3)],
                  card="Cold or flu symptoms: P3 if breathless; otherwise referral list.", q="breathless?",
                  referral=True),
    "medication": C(["question about medication", "missed a dose of tablets", "not sure about their tablets"],
                    [("", 5)], [("double_drowsy", "took a double dose and feels drowsy", 2)],
                    card="Medication questions: P2 if drowsy after extra doses; otherwise referral list.",
                    q="extra dose? drowsy?", referral=True),
    "minor_cut": C(["small cut", "cut finger"], [("", 5)], [("still_bleeding", "still bleeding through a towel", 3)],
                   card="Minor wounds: P3 if still bleeding; otherwise referral list.", q="bleeding stopped?",
                   referral=True),
    "urinary": C(["burning when passing urine", "water infection"], [("", 5)],
                 [("rigors", "shaking with a high temperature", 3)],
                 card="Urinary symptoms: P3 with fever and shaking; otherwise referral list.", q="fever or shaking?",
                 referral=True),
    "toothache": C(["toothache", "dental pain"], [("", 5)], [("swelling", "face swelling up toward the eye", 3)],
                   card="Dental pain: P3 if facial swelling is spreading; otherwise referral list.", q="swelling?",
                   referral=True),
    "dizziness": C(["dizzy", "light-headed"], [("", 4)],
                   [("fainted", "fainted briefly", 3), ("chest", "chest pain with it", 2)],
                   card="Dizziness: P2 with chest pain; P3 after a faint; otherwise P4.", q="fainted? chest pain?"),
    "choking": C(["choking", "food stuck in the throat"], [("", 2)],
                 [("no_air", "can't breathe, speak or cough at all", 1)], unc=True,
                 card="Choking: P1 if unable to breathe, speak or cough; P2 otherwise.", q="speaking or coughing?"),
    "animal_bite": C(["dog bite", "bitten by a dog"], [("", 4)],
                     [("face_neck", "the bite is on the face", 3), ("heavy_bleed", "bleeding heavily", 2)],
                     card="Animal bites: P2 with heavy bleeding; P3 to the face or neck; otherwise P4.",
                     q="where? bleeding?", unsafe=0.08),
    "person_down": C(["someone collapsed", "man down on the pavement", "person lying in the road"], [("", 3)], [],
                     unc=True, card="Person down, problem unknown: P3 if alert; otherwise the universal rules apply.",
                     q=""),
    "electrocution": C(["electric shock", "got a shock off a cable"], [("", 2)],
                       [("contact", "still touching the cable", 1)], unc=True,
                       card="Electrocution: P1 if still in contact with the source; otherwise P2.",
                       q="still in contact?"),
}
REL_BY_AGE = {"child": ["parent", "teacher", "sports coach", "grandparent", "childminder"],
              "adult": ["partner", "flatmate", "coworker", "friend", "parent", "neighbour", "sibling"],
              "older": ["son", "daughter", "partner", "neighbour", "grandchild", "home carer"]}
THIRD_PARTY = ["passer-by", "shop worker", "bus driver", "relative calling from another town",
               "security guard who saw it on camera"]
FACILITY = {"child": ["school nurse", "nursery manager"], "adult": ["pharmacist", "gym first-aider", "workplace "
                                                                    "first-aider"],
            "older": ["care-home nurse", "day-centre manager"]}
BREATH_TXT = {"normal": ["breathing normally", "breathing fine", "breathing okay, regular"],
              "abnormal": ["gasping", "making snoring noises, irregular", "barely breathing",
                           "not breathing that I can see", "breathing every few seconds, gurgling"],
              "unknown": ["can't tell if they're breathing", "not sure, can't see properly"]}
CONSC_TXT = {"alert": ["awake and talking", "alert, answering", "awake"],
             "not_alert": ["not answering me", "won't wake up", "eyes closed, not responding", "out cold"],
             "unknown": ["don't know, I can't get to them", "not sure, they're across the road"]}
INSTRUCT_FLAGS = {"spurting", "not_stopping", "stab_shot", "no_air", "heavy_bleed", "throat", "lips"}


def _age_band(age):
    return "child" if age < 16 else ("older" if age >= 65 else "adult")


class MedicalCallPriority(Schema):
    name = "medical_call_priority"
    description = ("A medical 911-style call in progress: the priority and response the written dispatch guide gives "
                   "from the call-taker's key questions, the patient's age and pregnancy status, and the complaint "
                   "card.")
    surfaces = {"json": 0.5, "cad_log": 0.25, "call_summary": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        g = {"rev": f"{rng.randint(3, 12)}.{rng.randint(0, 9)}", "stroke_h": pick(rng, [3, 4.5, 6]),
             "fall_m": pick(rng, [1.5, 2, 3]), "seizure_min": pick(rng, [3, 5, 10]), "county": _county(rng),
             "cp_age": pick(rng, [30, 35, 40]), "older_age": pick(rng, [65, 70, 75])}
        consc = pick_w(rng, {"alert": 66, "not_alert": 27, "unknown": 7})
        if consc == "alert":
            weights = {k: (2.6 if c["referral"] else 2.4 if c["bases"][0][1] == 4 else 1.0)
                       for k, c in MED_CARDS.items() if k != "person_down"}
            weights["chest_pain"] = 1.8
            weights["abdominal"] = 1.8
            for k in ("bleeding", "choking", "burn", "seizure", "diabetic", "allergy"):
                weights[k] = 3.0
        else:
            weights = {k: 1.0 for k, c in MED_CARDS.items() if c["unc"]}
        key = pick_w(rng, weights)
        card = MED_CARDS[key]
        base_frag, base = pick(rng, card["bases"])
        # the patient: age and sex are clinical inputs to the guide (age thresholds, pregnancy question)
        if key == "chest_pain" and reg == "borderline":
            age = near(rng, g["cp_age"], 4, integer=True)
        elif reg == "borderline" and chance(rng, 0.4):
            age = near(rng, g["older_age"], 3, integer=True)
        else:
            low = base >= 4
            age = pick_w(rng, {"infant": 0.15 if low else 0.3, "child": 1, "adult": 6.5, "older": 1.4 if low else 2.5})
            age = {"infant": rng.randint(0, 0), "child": rng.randint(1, 15), "adult": rng.randint(16, 64),
                   "older": rng.randint(65, 96)}[age]
        if key in ("chest_pain", "stroke", "psych", "assault", "medication") and age < 12:
            age = rng.randint(18, 80)
        sex = pick(rng, ["F", "M"])
        pregnancy = None
        if key == "abdominal" and sex == "F" and 12 <= age <= 55:
            pregnancy = pick_w(rng, {"no": 3, "unsure": 1.4, "yes": 1.2})
        band = _age_band(age)
        # caller
        if consc == "unknown":
            ctype = "third_party"
        elif consc == "not_alert":
            ctype = pick_w(rng, {"second_party": 6, "third_party": 2, "facility": 2})
        else:
            ctype = pick_w(rng, {"first_party": 4 if age >= 12 else 0, "second_party": 4,
                                 "third_party": 0 if base >= 5 else 1, "facility": 1.5})
        rel = ("the patient" if ctype == "first_party" else pick(rng, THIRD_PARTY) if ctype == "third_party"
               else pick(rng, FACILITY[band]) if ctype == "facility" else pick(rng, REL_BY_AGE[band]))
        age_reported = age if ctype != "third_party" else max(1, int(round(age / 5.0)) * 5)
        breathing = None
        if consc == "not_alert":
            breathing = pick_w(rng, {"normal": 4, "abnormal": 6, "unknown": 1 if ctype == "third_party" else 0.3})
        elif consc == "unknown":
            breathing = "unknown"
        flags_present = []
        if card["flags"]:
            n = (pick(rng, [0, 0, 1]) if reg == "clear" else
                 pick_w(rng, {0: 4, 1: 1, 2: 0.3} if base >= 4 else {0: 3, 1: 1.5, 2: 0.5}))
            flags_present = [fl[0] for fl in rng.sample(card["flags"], min(n, len(card["flags"])))]
        card_unanswered = bool(card["q"]) and chance(rng, (0.1 if base >= 4 else 0.18) if reg != "clear" else 0.05)
        if card_unanswered:
            flags_present = []
        num_val = None
        if card["num"]:
            _, param, _, _ = card["num"]
            thr = g[param]
            if key == "fall" and chance(rng, 0.5):
                num_val = 0.0
            elif reg == "borderline":
                num_val = round(near(rng, thr, max(0.5, thr * 0.35)) * 2) / 2
            elif reg == "conflicting" and chance(rng, 0.35):
                num_val = None
            else:
                num_val = round(rng.uniform(thr * 0.3, thr * 2.2) * 2) / 2
            if key == "seizure" and num_val is not None and chance(rng, 0.35):
                num_val = 0.0
            if num_val is not None and num_val < 0:
                num_val = 0.0
        missing = []
        if reg != "clear" and chance(rng, 0.45):
            for _ in range(pick(rng, [1, 1, 2])):
                m = pick_w(rng, {"address": 3, "callback": 3, "consciousness": 0.8, "breathing": 0.8})
                if m not in missing:
                    missing.append(m)
        if card_unanswered:
            missing.append("card")
        if consc == "alert" and "breathing" in missing:
            missing.remove("breathing")
        referral_ok = card["referral"] and chance(rng, 0.75)
        # social context: the caller's language and any cost or immigration worry
        language = "English" if chance(rng, 0.78) else pick(rng, LANGUAGES)
        access = "not_needed"
        if language != "English":
            access = pick_w(rng, {"language_line": 5.5, "family_interpreting": 3, "none": 1.5})
            if access == "family_interpreting" and ctype == "first_party":
                access = "none"
        worry = None
        if chance(rng, 0.14 if reg == "conflicting" else 0.06):
            worry = pick(rng, ["asks whether the ambulance will cost money; says they have no insurance",
                               "worried that calling will cause immigration problems",
                               "says they can't afford an ambulance and might take a taxi instead"])
        unsafe = chance(rng, card["unsafe"] if reg != "conflicting" else card["unsafe"] * 1.5)
        demeanour = pick_w(rng, {"calm": 5, "upset": 3, "hysterical": 1.2})
        contradiction = reg == "conflicting" and chance(rng, 0.45)
        remark = None
        if reg == "conflicting":
            if consc != "alert" or flags_present:
                remark = pick(rng, ["caller says it's probably nothing and they will likely be fine",
                                    "caller keeps saying 'no need to rush, this happens sometimes'",
                                    "caller says the patient 'just needs checking over'"])
            else:
                remark = pick(rng, ["caller insists it is a life-or-death emergency and wants everyone sent",
                                    "caller shouting that they need an ambulance with lights right now",
                                    "caller says they will complain if a crew doesn't come immediately"])
        t0 = rand_ts(rng)
        return {"regime": reg, "guide": g, "card": key, "said": pick(rng, card["says"]), "base_note": base_frag,
                "base_priority": base, "age": age_reported, "sex": sex, "pregnancy": pregnancy,
                "consciousness": consc, "breathing": breathing, "flags_present": flags_present,
                "card_unanswered": card_unanswered, "measure": num_val, "missing": missing,
                "referral_accepted": referral_ok, "scene_unsafe": unsafe, "caller_type": ctype, "caller_rel": rel,
                "demeanour": demeanour, "contradiction": contradiction, "remark": remark, "received": t0,
                "caller_language": language, "language_access": access, "cost_or_status_worry": worry,
                "cad": _cad(rng, t0), "address": _address(rng), "cross": _street(rng), "place": pick(rng, PLACES),
                "call_taker": person(rng)}

    @staticmethod
    def _priority(f):
        card, g = MED_CARDS[f["card"]], f["guide"]
        p = f["base_priority"]
        for k, _, pri in card["flags"]:
            if k in f["flags_present"]:
                p = min(p, pri)
        if f["card_unanswered"] and card["flags"]:
            p = min(p, min(pri for _, _, pri in card["flags"]))
        if f["card"] == "chest_pain" and f["age"] >= g["cp_age"]:
            p = min(p, 2)
        if f["card"] == "abdominal" and f["pregnancy"] in ("yes", "unsure"):
            p = min(p, 2)
        if card["num"]:
            _, param, cmp, pri = card["num"]
            v = f["measure"]
            met = v is None or (v >= g[param] if cmp == "ge" else v <= g[param])
            if f["card"] in ("fall", "seizure") and v == 0.0:
                met = False
            if met:
                p = min(p, pri)
        consc = "unknown" if "consciousness" in f["missing"] else f["consciousness"]
        breathing = "unknown" if "breathing" in f["missing"] else f["breathing"]
        if consc == "not_alert" and breathing in ("abnormal", "unknown"):
            p = 1
        elif consc in ("not_alert", "unknown"):
            p = min(p, 2)
        special_age = f["age"] < 1 or f["age"] >= g["older_age"]
        if p == 5 and (not f["referral_accepted"] or special_age):
            p = 4
        if p == 4 and special_age:
            p = 3
        return p

    def intended(self, f):
        p = self._priority(f)
        if p == 5:
            resp = "nurse_line_referral"
        elif f["scene_unsafe"]:
            resp = "stage_for_police"
        elif p == 1:
            resp = "als_lights_with_first_responder"
        elif p == 2:
            resp = "als_lights_sirens"
        else:
            resp = "ambulance_no_lights"
        rel = {"first_party": 3, "facility": 3, "second_party": 2, "third_party": 0}[f["caller_type"]]
        if f["caller_type"] in ("first_party", "second_party") and f["demeanour"] != "calm":
            rel -= 1
        if f["contradiction"]:
            rel -= 1
        if f["language_access"] in ("family_interpreting", "none"):
            rel = min(rel, 1)
        consc = "unknown" if "consciousness" in f["missing"] else f["consciousness"]
        breathing = "unknown" if "breathing" in f["missing"] else f["breathing"]
        with_pt = f["caller_type"] in ("first_party", "second_party", "facility")
        seizing = f["card"] == "seizure" and (f["measure"] is None or f["measure"] > 0)
        hands_on = with_pt and not f["scene_unsafe"] and (
            consc == "not_alert" or f["card"] in ("bleeding", "choking", "burn") or seizing
            or (f["card"] == "diabetic" and consc == "alert" and "cannot_swallow" not in f["flags_present"])
            or bool(set(f["flags_present"]) & INSTRUCT_FLAGS))
        return {"dispatch_priority": 5 - p, "caller_information_quality": max(0, rel),
                "response_configuration": resp,
                "key_questions_complete": ("no" if f["missing"] or f["language_access"] in ("family_interpreting", "none")
                                           else "yes"),
                "hands_on_instructions_now": "yes" if hands_on else "no"}

    def _answers(self, f, rng):
        card = MED_CARDS[f["card"]]
        a = {"consciousness": None if "consciousness" in f["missing"] else pick(rng, CONSC_TXT[f["consciousness"]])}
        if f["consciousness"] != "alert":
            a["breathing"] = None if "breathing" in f["missing"] else pick(rng, BREATH_TXT[f["breathing"]])
        else:
            airway = {"no_air", "throat", "blue_lips", "no_sentences", "breathless", "inhaler_failed"}
            if f["card"] == "breathing" or set(f["flags_present"]) & airway:
                ok = pick(rng, ["struggling to breathe", "breathing fast and noisy", "working hard to breathe"])
            else:
                ok = pick(rng, ["breathing normally", "talking fine"])
            a["breathing"] = None if "breathing" in f["missing"] else ok
        bits = [f["base_note"]] if f["base_note"] else []
        bits += [words for k, words, _ in card["flags"] if k in f["flags_present"]]
        if card["num"]:
            v = f["measure"]
            if f["card"] == "stroke":
                bits.append("doesn't know when it started" if v is None else f"started about {v:g} h ago")
            elif f["card"] == "fall":
                bits.append("not sure how far they fell" if v is None else
                            ("fell from standing" if v == 0 else f"fell about {v:g} m"))
            elif f["card"] == "seizure":
                bits.append("can't say how long it's been" if v is None else
                            ("the fit has stopped, lasted a couple of minutes" if v == 0
                             else f"still fitting, about {v:g} min so far"))
        if f["card_unanswered"]:
            card_a = ("(not asked)" if card["referral"] else
                      pick(rng, ["(no answer: line dropped)", "(caller did not answer, went to help)", "(not asked)"]))
        elif bits:
            card_a = "; ".join(bits)
        else:
            card_a = pick(rng, ["no to all", "none of those", "no"])
        a["card"] = card_a
        if f["pregnancy"] is not None:
            a["pregnant"] = {"yes": pick(rng, ["yes, about 9 weeks", "yes, early pregnancy"]),
                             "unsure": pick(rng, ["could be, period is late", "not sure"]),
                             "no": "no"}[f["pregnancy"]]
        return a

    def _guide(self, f, rng):
        g = f["guide"]
        card = MED_CARDS[f["card"]]
        line = card["card"].format(**g)
        if f["card"] == "chest_pain":
            line = (f"Chest pain: P2 if aged {g['cp_age']} or over, sweating or clammy, short of breath, or known "
                    f"heart disease; otherwise P3.")
        if f["card"] == "abdominal":
            line = ("Abdominal pain: P2 with fainting or near-fainting, vomiting blood, or a pregnancy that is "
                    "confirmed or possible; P3 otherwise; mild pain for days P4. Ask about pregnancy for any female "
                    "patient aged 12-55.")
        return {
            "title": pick(rng, [f"{g['county']} County dispatch guide {g['rev']}: medical calls",
                                f"Medical call coding, {g['county']} Regional 911 (guide rev {g['rev']})"]),
            "universal": "Not alert with absent, gasping, snoring or irregular breathing (or breathing unknown): P1. "
                         "Not alert but breathing normally, or alertness unknown: at least P2.",
            "card": line,
            "unanswered": "A card question left unanswered is coded as if the answer were yes.",
            "age": f"Patients under 1 year or aged {g['older_age']}+: a P4 code becomes P3 and the nurse line is not "
                   f"offered.",
            "referral": "Referral-list complaints go to the nurse advice line (P5) if the caller accepts; if not, P4.",
            "staging": "Any weapon, violence or threat at the scene: crews stage until police clear it.",
            "access": "Callers who prefer another language: connect the language line before the key questions; "
                      "answers relayed by a child or bystander do not count as obtained. Never ask about cost, "
                      "insurance or immigration status; they never change the code or the response.",
            "instructions": "If someone is with the patient and the scene is safe, coach hands-on first aid for: "
                            "a patient who is not alert (CPR or recovery position), any bleeding call or heavy "
                            "bleeding (pressure), choking (back blows), burns (cool water), a seizure still going "
                            "(protect from injury), lip or throat swelling (help with their auto-injector), a "
                            "diabetic who can still swallow (sugary drink).",
            "responses": "P1 ALS lights/sirens plus nearest first responder; P2 ALS lights/sirens; P3-P4 ambulance "
                         "without lights; P5 nurse advice line.",
        }

    def render(self, f, rng, surface):
        a = self._answers(f, rng)
        gd = self._guide(f, rng)
        t0 = f["received"]
        loc_ok = "address" not in f["missing"]
        cb = "callback" not in f["missing"]
        caller = f"{f['caller_rel']} ({f['caller_type'].replace('_', ' ')})"
        age_txt = ("under 1 year" if f["age"] < 1 else
                   (f"about {f['age']}" if f["caller_type"] == "third_party" else str(f["age"])))
        notes = [f"caller is {f['demeanour']}"]
        if f["scene_unsafe"]:
            threats = ["says the person who did it is still outside", "someone at the address is drunk and "
                       "threatening", "dog loose and aggressive at the door"]
            if f["caller_type"] != "first_party":
                threats.append("patient has a knife and is shouting")
            notes.append(pick(rng, threats))
        if f["remark"]:
            notes.append(f["remark"])
        if f["caller_language"] != "English":
            notes.append(f"caller prefers {f['caller_language']}; " + {
                "language_line": "language line connected before questions",
                "family_interpreting": pick(rng, ["caller's 12-year-old interpreting", "neighbour interpreting"]),
                "none": "no interpreter, questions asked in English"}[f["language_access"]])
        if f["cost_or_status_worry"]:
            notes.append(f["cost_or_status_worry"])
        if MED_CARDS[f["card"]]["referral"]:
            notes.append("offered nurse advice line: " + ("caller accepts" if f["referral_accepted"]
                                                          else "caller declines, wants an ambulance"))
        if f["contradiction"]:
            notes.append(pick(rng, ["caller first said awake, later 'not really responding'",
                                    "answers keep changing between questions"]) if f["consciousness"] != "alert"
                         else pick(rng, ["answers keep changing between questions",
                                         "caller contradicts their own account of what happened"]))
        cq = MED_CARDS[f["card"]]["q"]
        if surface == "json":
            return dumps(prune({
                "incident": {"cad": f["cad"], "received": _hms(t0), "chief_complaint": f["said"],
                             "location": {"address": f["address"], "cross_street": f["cross"],
                                          "verified": loc_ok, "where": f["place"]},
                             "callback_number": "obtained" if cb else "not obtained"},
                "caller": {"who": caller, "with_patient": f["caller_type"] != "third_party"},
                "patient": {"age": age_txt, "sex": f["sex"], "pregnant": a.get("pregnant")},
                "key_questions": {"alert": a["consciousness"] or "not asked",
                                  "breathing": a["breathing"] or "not asked",
                                  "card_question": f"{cq} {a['card']}" if cq else None},
                "remarks": noisy("; ".join(notes), rng, 0.1),
                "guide": gd,
            }))
        if surface == "cad_log":
            t = t0
            lines = [f"INC {f['cad']}  MEDICAL  {gd['title']}", f"{_hms(t)}  CALL RCVD  "
                     f"{pick(rng, ['wireless', 'landline', 'relay', 'VoIP'])}"]

            def step(text):
                nonlocal t
                t += timedelta(seconds=rng.randint(6, 22))
                lines.append(f"{_hms(t)}  {text}")
            step(f"LOC  {f['address']} x {f['cross']}" + ("  [verified]" if loc_ok else "  [unverified]"))
            step("CB#  " + ("ok" if cb else "--"))
            step(f"CALLER  {caller}")
            step(f"CC  {f['said']}  PT {age_txt} {f['sex']}")
            step(f"Q alert?  A: {a['consciousness'] or 'not asked'}")
            step(f"Q breathing?  A: {a['breathing'] or 'not asked'}")
            if "pregnant" in a:
                step(f"Q pregnant?  A: {a['pregnant']}")
            if cq:
                step(f"Q {cq}  A: {a['card']}")
            for n in notes:
                step(f"NOTE  {n}")
            lines += ["", "Guide extract:"] + [f"  {k}: {v}" for k, v in gd.items() if k != "title"]
            return "\n".join(lines)
        para = [f"Call {f['cad']} at {_hm(t0)} from {f['caller_rel']} reporting {f['said']} at {f['address']} "
                f"({'location verified' if loc_ok else 'location not yet verified'}; callback "
                f"{'obtained' if cb else 'not obtained'}).",
                f"Patient {age_txt}, {f['sex']}." + (f" Pregnant: {a['pregnant']}." if "pregnant" in a else ""),
                f"Alertness: {a['consciousness'] or 'not asked'}. Breathing: {a['breathing'] or 'not asked'}."]
        if cq:
            para.append(f"Card question ({cq}): {a['card']}.")
        para.append(_cap("; ".join(notes)) + ".")
        return (f"CALL-TAKER SUMMARY -- {f['call_taker']}\n" + noisy(join_sentences(para), rng, 0.1) + "\n\n"
                + gd["title"] + "\n" + bullets([v for k, v in gd.items() if k != "title"], rng))

    def questions(self):
        return {
            "dispatch_priority": score("What priority does the written dispatch guide give this call?", [
                "P5: referral to the nurse advice line, no ambulance.",
                "P4: non-urgent ambulance.",
                "P3: urgent ambulance without lights and sirens.",
                "P2: emergency response with lights and sirens.",
                "P1: immediate life threat, the fastest and fullest response.",
            ]),
            "caller_information_quality": score("How far can the dispatcher rely on what the caller reports?", [
                "Poor: the caller is not with the patient, or the answers contradict each other.",
                "Limited: the caller is with the patient but distressed or guessing.",
                "Good: someone with the patient answering clearly.",
                "Direct: the patient or a trained person on the line, answering clearly.",
            ]),
            "response_configuration": choice("How should this call be resourced?", {
                "als_lights_sirens": "Send an advanced life support ambulance with lights and sirens.",
                "als_lights_with_first_responder": "Send an advanced life support ambulance with lights and sirens "
                                                   "and the nearest first-responder unit too.",
                "ambulance_no_lights": "Send an ambulance without lights and sirens.",
                "nurse_line_referral": "Transfer the caller to the nurse advice line instead of sending an ambulance.",
                "stage_for_police": "Send the ambulance to wait at a safe distance until police have secured the "
                                    "scene.",
            }),
            "key_questions_complete": noul(
                "The call-taker obtained every key answer the guide requires before coding the call.",
                yes="Address verified, callback number taken, alertness and breathing asked, and the complaint "
                    "card's question answered.",
                no="At least one of those was skipped, lost or left unanswered."),
            "hands_on_instructions_now": noul(
                "The call-taker should talk the caller through hands-on first aid before the crew arrives."),
        }


# =========================================================================== schema 2: traffic collision
MECH = {
    "rear_end": (["rear-end shunt", "car ran into the back of another", "low-speed bump at the lights"], (8, 45)),
    "side_impact": (["side impact at a junction", "T-boned at the crossroads"], (25, 75)),
    "head_on": (["head-on collision", "two cars hit head on"], (45, 100)),
    "rollover": (["car rolled over", "vehicle flipped onto its roof"], (50, 110)),
    "into_object": (["car went off the road into a tree", "car hit a lamp post"], (35, 100)),
    "pedestrian": (["pedestrian hit by a car", "someone knocked down crossing"], (10, 70)),
    "cyclist": (["cyclist hit by a car", "bike and car collided"], (10, 60)),
    "motorcycle": (["motorbike down", "motorcyclist came off"], (25, 110)),
    "pileup": (["several cars piled up", "multi-vehicle crash in fog"], (35, 100)),
}
VULNERABLE = ("pedestrian", "cyclist", "motorcycle")
FIRE_HAZARDS = {"fuel_leak", "vehicle_fire", "downed_power_line", "hazmat_placard"}
HIGH_MECH = ("head_on", "rollover")
CALLER_POS = {"involved_driver": "driver of one of the cars", "passenger": "passenger", "witness": "witness standing "
              "at the scene", "passing_motorist": "driver who saw it while passing", "remote_relative":
              "relative phoning after a text from someone involved", "police_on_scene": "police officer on scene"}


class TrafficCollisionCall(Schema):
    name = "traffic_collision_call"
    description = ("A road traffic collision reported to the communications center: priority, resources, scene "
                   "safety and how firm the patient count is, under the written collision guide.")
    surfaces = {"json": 0.5, "collision_cad": 0.25, "call_transcript": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        g = {"rev": f"{rng.randint(2, 9)}.{rng.randint(0, 9)}", "unit": pick_w(rng, {"km/h": 3, "mph": 2}),
             "mci": pick(rng, [5, 6, 8]), "county": _county(rng), "older_age": pick(rng, [65, 70])}
        g["high_speed"] = pick(rng, [60, 70, 80]) if g["unit"] == "km/h" else pick(rng, [40, 45, 50])
        mech = pick_w(rng, {"rear_end": 5, "side_impact": 3.5, "head_on": 1.2, "rollover": 1.2, "into_object": 2,
                            "pedestrian": 1.1, "cyclist": 1, "motorcycle": 1, "pileup": 1})
        lo, hi = MECH[mech][1]
        conv = 1.0 if g["unit"] == "km/h" else 0.62
        speed = int(round(rng.uniform(lo, hi) * conv / 5) * 5)
        if reg == "borderline":
            speed = int(near(rng, g["high_speed"], 10, integer=True))
        if chance(rng, 0.2 if reg == "conflicting" else 0.08):
            speed = None
        road = pick_w(rng, {"highway": 3, "urban": 4, "rural": 2})
        vehicles = (rng.randint(3, 9) if mech == "pileup" else (1 if mech in ("into_object", "rollover") or
                                                                 mech in VULNERABLE and chance(rng, 0.2)
                                                                 else 2))
        if mech == "pileup":
            people = rng.randint(3, 14)
        else:
            people = vehicles + rng.randint(0, 3)
        injured = rng.randint(0, people)
        if chance(rng, 0.4 if reg == "clear" else 0.22):
            injured = 0
        if mech in VULNERABLE:
            injured = max(injured, 1)
        uncon = pick_w(rng, {"no": 9, "yes": 1, "unknown": 1}) if injured else "no"
        entrap = pick_w(rng, {"no": 9, "yes": 1.1, "unknown": 1}) if mech not in VULNERABLE else "no"
        # age and pregnancy of the injured are clinical inputs to the guide
        oldest = None
        if injured:
            if reg == "borderline" and chance(rng, 0.5):
                oldest = near(rng, g["older_age"], 4, integer=True)
            else:
                oldest = pick_w(rng, {rng.randint(17, 40): 4, rng.randint(41, 64): 3, rng.randint(65, 92): 2.2})
            if chance(rng, 0.12):
                oldest = None
        pregnant = injured > 0 and mech not in ("motorcycle", "cyclist") and chance(rng, 0.1)
        ejected = mech in ("rollover", "head_on") and chance(rng, 0.12)
        wants_check = injured == 0 and chance(rng, 0.5)
        hazards = []
        for h, p in (("fuel_leak", 0.08), ("vehicle_fire", 0.11), ("downed_power_line", 0.11),
                     ("hazmat_placard", 0.06), ("road_rage_weapon", 0.07)):
            if chance(rng, p * (2 if reg == "conflicting" else 1)):
                hazards.append(h)
        caller = pick_w(rng, {"involved_driver": 3, "passenger": 1.5, "witness": 3, "passing_motorist": 2,
                              "remote_relative": 0.8, "police_on_scene": 1})
        count_style = "exact"
        if caller in ("passing_motorist", "remote_relative") or reg == "borderline":
            count_style = pick(rng, ["exact", "range", "unknown"])
        if reg == "conflicting" and chance(rng, 0.4):
            count_style = "changed"
        lane_blocked = chance(rng, 0.6 if road == "highway" else 0.4)
        remark = None
        if reg == "conflicting":
            remark = pick(rng, ["caller says 'everyone is fine, just shaken' but then mentions someone not moving",
                                "caller thinks nobody is hurt but can hear someone screaming from one car",
                                "caller says it looks bad but is driving past and did not stop"])
            if "not moving" in remark and uncon == "no" and injured:
                uncon = "unknown"
        t0 = rand_ts(rng)
        return {"regime": reg, "guide": g, "mechanism": mech, "mech_text": pick(rng, MECH[mech][0]),
                "speed": speed, "road": road, "vehicles": vehicles, "people": people, "injured": injured,
                "unconscious": uncon, "entrapment": entrap, "ejected": ejected, "wants_check": wants_check,
                "oldest_injured_age": oldest, "pregnant_injured": pregnant,
                "hazards": hazards, "caller_pos": caller, "count_style": count_style,
                "count_low": max(1, people - rng.randint(1, 2)), "lane_blocked": lane_blocked, "remark": remark,
                "received": t0, "cad": _cad(rng, t0),
                "location": (f"{pick(rng, ['Highway', 'Route', 'State Road'])} {rng.randint(2, 99)} "
                             f"{pick(rng, ['northbound', 'southbound', 'eastbound', 'westbound'])} near mile "
                             f"{rng.randint(1, 140)}" if road == "highway" else
                             f"{_street(rng)} at {_street(rng)}" if road == "urban" else
                             f"{_street(rng)}, {rng.randint(1, 9)} km past the {pick(rng, ['farm shop', 'reservoir', 'old mill', 'church'])}"),
                "dispatcher": person(rng)}

    def _high(self, f):
        g = f["guide"]
        if f["mechanism"] in HIGH_MECH:
            return True
        if f["speed"] is None:
            return f["road"] == "highway"
        return f["speed"] >= g["high_speed"]

    def _count(self, f):
        return None if f["count_style"] == "unknown" else f["people"]

    def intended(self, f):
        g, high = f["guide"], self._high(f)
        inj = f["injured"]
        entrap = f["entrapment"] == "yes" or (f["entrapment"] == "unknown" and high)
        if (f["unconscious"] == "yes" or entrap or f["ejected"]
                or (f["mechanism"] in ("pedestrian", "cyclist") and high)):
            p = 1
        elif inj and (high or f["mechanism"] in VULNERABLE or f["unconscious"] == "unknown" or f["pregnant_injured"]
                      or (f["oldest_injured_age"] is not None and f["oldest_injured_age"] >= g["older_age"])):
            p = 2
        elif inj:
            p = 3
        elif f["wants_check"] or set(f["hazards"]) & FIRE_HAZARDS:
            p = 4
        else:
            p = 5
        people = f["people"]
        if p == 5:
            pkg = "law_enforcement_only"
        elif people >= g["mci"] and f["count_style"] != "unknown":
            pkg = "mass_casualty_activation"
        elif entrap or set(f["hazards"]) & FIRE_HAZARDS:
            pkg = "ambulance_and_fire_rescue"
        elif inj >= 2:
            pkg = "multiple_ambulances"
        else:
            pkg = "single_ambulance"
        if f["count_style"] == "unknown":
            cert = 0
        elif f["count_style"] in ("range", "changed") or f["caller_pos"] in ("passing_motorist", "remote_relative"):
            cert = 1
        elif f["caller_pos"] == "witness":
            cert = 2
        else:
            cert = 3
        stage = bool(set(f["hazards"]) & {"downed_power_line", "vehicle_fire", "hazmat_placard", "road_rage_weapon"})
        return {"collision_priority": 5 - p, "resource_package": pkg, "patient_count_certainty": cert,
                "stage_before_entering": "yes" if stage else "no",
                "traffic_control_needed": "yes" if f["lane_blocked"] else "no"}

    def _facts(self, f, rng):
        g = f["guide"]
        if f["count_style"] == "unknown":
            cnt = pick(rng, ["no idea how many people", "can't see how many are in the cars"])
        elif f["count_style"] == "range":
            cnt = f"maybe {f['count_low']} or {f['people']} people involved"
        elif f["count_style"] == "changed":
            cnt = f"first said {f['count_low']} people, now says {f['people']}"
        else:
            cnt = f"{f['people']} people involved"
        hurt = ("nobody seems hurt" if f["injured"] == 0 else
                f"{f['injured']} hurt" if f["injured"] > 1 else "one person hurt")
        who_old = "oldest injured" if f["injured"] > 1 else "the injured person is"
        if f["injured"] == 0 and f["wants_check"]:
            hurt += pick(rng, [", one driver wants their neck checked", ", someone asking to be looked at"])
        if f["injured"]:
            if f["oldest_injured_age"] is None:
                hurt += pick(rng, ["; ages not known", "; can't say how old"])
            else:
                hurt += f"; {who_old} about {f['oldest_injured_age']}"
            if f["pregnant_injured"]:
                hurt += pick(rng, ["; one injured occupant is pregnant", "; an injured driver says they are pregnant"])
        sp = (pick(rng, ["speed unknown", "caller can't say how fast"]) if f["speed"] is None
              else f"about {f['speed']} {g['unit']}")
        haz = {"fuel_leak": "fuel leaking onto the road", "vehicle_fire": "smoke and flames from one engine",
               "downed_power_line": "power line down across a car", "hazmat_placard": "lorry with a hazard "
               "placard leaking", "road_rage_weapon": "one driver waving a bat at the other"}
        nv = f["vehicles"]
        vtxt = {1: "one vehicle", 2: "two vehicles", 3: "three vehicles"}.get(nv, f"{nv} vehicles")
        return {"what": f["mech_text"], "speed": sp, "vehicles": vtxt, "count": cnt,
                "injuries": hurt,
                "unconscious": {"yes": "one person not responding", "no": "everyone conscious as far as known",
                                "unknown": "caller can't tell if anyone is unconscious"}[f["unconscious"]],
                "trapped": {"yes": "someone trapped, door won't open", "no": "nobody trapped",
                            "unknown": "not sure if anyone is trapped"}[f["entrapment"]],
                "ejected": "one occupant thrown from the vehicle" if f["ejected"] else None,
                "hazards": ", ".join(haz[h] for h in f["hazards"]) or "none reported",
                "lanes": ("blocking a live lane" if f["lane_blocked"] else
                          pick(rng, ["cars moved onto the verge", "off the carriageway, traffic flowing"]))}

    def _guide(self, f, rng):
        g = f["guide"]
        return {
            "title": pick(rng, [f"{g['county']} collision card {g['rev']}",
                                f"Road traffic collision coding -- {g['county']} Public Safety Communications, "
                                f"rev {g['rev']}"]),
            "p1": f"P1: anyone unconscious, trapped or thrown from a vehicle; pedestrian or cyclist struck at high "
                  f"speed. Trapped status unknown at a high-speed crash counts as trapped.",
            "p2_p5": f"P2: injuries with a high-speed, head-on or rollover mechanism, any motorcyclist, pedestrian or "
                     f"cyclist, consciousness unknown, or an injured person aged {g['older_age']}+ or pregnant. P3: "
                     f"injuries, lower speed. P4: uninjured but asking to be "
                     "checked. P5: damage only, police response only.",
            "speed": f"High speed means {g['high_speed']} {g['unit']} or more; if speed is unknown, highways count as "
                     f"high speed and other roads as low.",
            "resources": f"{g['mci']}+ people involved (use the highest count given): mass-casualty activation. "
                         f"Trapped occupants, fire, fuel leak, downed lines or a hazardous load: add fire/rescue (with "
                         f"no one injured, one ambulance stands by at P4). Two or more injured: one ambulance each.",
            "scene": "Downed power lines, vehicle fire, hazardous load or violence: crews stage until made safe. "
                     "Request police traffic control for any blocked live lane.",
        }

    def render(self, f, rng, surface):
        fa = self._facts(f, rng)
        gd = self._guide(f, rng)
        t0 = f["received"]
        who = CALLER_POS[f["caller_pos"]]
        if surface == "json":
            return dumps(prune({
                "incident": {"cad": f["cad"], "received": _hms(t0), "location": f["location"],
                             "road_type": f["road"], "type": "road traffic collision"},
                "caller": {"position": who, "remark": f["remark"]},
                "reported": fa,
                "guide": gd,
            }))
        if surface == "collision_cad":
            t = t0
            out = [f"{_hms(t)}  RTC  {f['cad']}  {f['location']}", f"          caller: {who}"]
            for k in ("what", "speed", "vehicles", "count", "injuries", "unconscious", "trapped", "ejected",
                      "hazards", "lanes"):
                if fa.get(k):
                    t += timedelta(seconds=rng.randint(5, 25))
                    out.append(f"{_hms(t)}  {k.upper():<12} {fa[k]}")
            if f["remark"]:
                out.append(f"          NB {f['remark']}")
            return "\n".join(out) + f"\n\n{gd['title']}\n" + "\n".join(f"- {v}" for k, v in gd.items()
                                                                       if k != "title")
        lines = [f"[{_hms(t0)}] CT: 911, what is the address of your emergency?",
                 f"Caller: {f['location']}. There's been a {fa['what']}.",
                 "CT: Are you involved or a witness?", f"Caller: I'm the {who}.",
                 "CT: How many people and vehicles?", f"Caller: {fa['vehicles']}, {fa['count']}.",
                 "CT: Is anyone hurt, trapped or not awake?",
                 f"Caller: {fa['injuries']}; {fa['trapped']}; {fa['unconscious']}."]
        if fa.get("ejected"):
            lines.append(f"Caller: and {fa['ejected']}.")
        lines += ["CT: Any fire, fuel, wires or danger?", f"Caller: {fa['hazards']}. They're {fa['lanes']}.",
                  "CT: Roughly how fast were they going?", f"Caller: {fa['speed']}."]
        if f["remark"]:
            lines.append(f"(CT note: {f['remark']})")
        return ("CALL TRANSCRIPT (abridged)\n" + "\n".join(noisy(x, rng, 0.06) for x in lines) + "\n\n"
                + gd["title"] + "\n" + "\n".join(f"* {v}" for k, v in gd.items() if k != "title"))

    def questions(self):
        return {
            "collision_priority": score("What priority does the collision guide give this call?", [
                "P5: damage only; police attend, no medical response.",
                "P4: nobody injured, but someone asks to be checked.",
                "P3: injuries from a lower-speed collision.",
                "P2: injuries with a dangerous mechanism or a vulnerable road user.",
                "P1: someone unconscious, trapped, thrown out, or a road user struck at speed.",
            ]),
            "resource_package": choice("What should be sent to this collision?", {
                "ambulance_and_fire_rescue": "Ambulance plus fire and rescue for extrication or fire risk.",
                "law_enforcement_only": "Police only; no ambulance is needed.",
                "mass_casualty_activation": "Activate the mass-casualty plan with several ambulances and a "
                                            "supervisor.",
                "multiple_ambulances": "One ambulance for each injured person, without the mass-casualty plan.",
                "single_ambulance": "A single ambulance.",
            }),
            "patient_count_certainty": score("How firmly is the number of people involved established?", [
                "Unknown: no count could be given.",
                "Rough: a guess, a range or a count that changed, or from someone not at the scene.",
                "Fair: a definite count from a bystander at the scene.",
                "Firm: a definite count from someone involved or from police on scene.",
            ]),
            "stage_before_entering": noul(
                "Responding crews must hold back until the scene has been made safe.",
                yes="A downed power line, a vehicle fire, a leaking hazardous load or violence is reported at the "
                    "scene.",
                no="None of the hazards that require staging is reported; a fuel leak alone brings fire/rescue "
                   "but does not stop crews."),
            "traffic_control_needed": noul("Police traffic control should be requested for this collision."),
        }


# =========================================================================== schema 3: interfacility transfer
IFT_CATS = {
    "stemi": ("heart attack (STEMI) for emergency angioplasty", 3, True),
    "stroke_lvo": ("large-vessel stroke for clot retrieval", 3, True),
    "trauma": ("trauma needing a surgeon not available on site", 3, True),
    "aortic": ("suspected aortic dissection for surgery", 3, True),
    "sepsis_icu": ("septic shock for intensive care", 2, False),
    "gi_bleed": ("gastrointestinal bleed for endoscopy and ICU", 2, False),
    "resp_failure": ("respiratory failure for intensive care", 2, False),
    "dka": ("diabetic ketoacidosis for high-dependency care", 2, False),
    "cardiac_ward": ("chest pain for a cardiology ward bed", 1, False),
    "psych_admit": ("mental health admission", 1, False),
    "ortho_ward": ("hip fracture for the orthopaedic ward", 1, False),
    "pneumonia_ward": ("pneumonia for a respiratory ward bed", 1, False),
    "care_home": ("discharge back to a care home", 0, False),
    "dialysis": ("dialysis appointment", 0, False),
    "imaging": ("outpatient MRI appointment", 0, False),
    "rehab": ("move to a rehabilitation unit", 0, False),
}
# intervention -> tier needed (0 van, 1 BLS, 2 ALS, 3 critical care)
INTERVENTIONS = {
    "saline_lock": ("saline lock, nothing running", 1), "o2_low": ("oxygen 2 L/min nasal", 0),
    "o2_mid": ("oxygen 4 L/min nasal", 1), "o2_mask": ("oxygen 10 L/min by mask", 2),
    "monitor": ("continuous cardiac monitoring", 2), "maint_fluids": ("maintenance IV fluids", 2),
    "heparin": ("heparin infusion (fixed rate)", 2), "ppi": ("proton-pump inhibitor infusion (fixed rate)", 2),
    "nitro": ("nitroglycerin infusion, titrated to pain", 3), "insulin": ("insulin infusion, titrated hourly", 3),
    "labetalol": ("labetalol infusion, titrated to blood pressure", 3),
    "norepi": ("norepinephrine infusion", 3), "propofol": ("propofol sedation infusion", 3),
    "blood": ("blood transfusion running", 3), "vent": ("mechanical ventilation", 3), "bipap": ("BiPAP", 3),
}
CAT_INTERVENTIONS = {
    "stemi": [["monitor", "heparin"], ["monitor", "heparin", "nitro"], ["monitor", "o2_mid"]],
    "stroke_lvo": [["monitor", "saline_lock"], ["monitor", "labetalol"], ["monitor", "maint_fluids"]],
    "trauma": [["monitor", "maint_fluids"], ["monitor", "blood"], ["monitor", "vent", "propofol"]],
    "aortic": [["monitor", "labetalol"], ["monitor", "maint_fluids"]],
    "sepsis_icu": [["monitor", "maint_fluids", "norepi"], ["monitor", "maint_fluids"], ["monitor", "maint_fluids",
                                                                                         "o2_mask"]],
    "gi_bleed": [["monitor", "blood", "ppi"], ["monitor", "ppi", "maint_fluids"]],
    "resp_failure": [["monitor", "vent", "propofol"], ["monitor", "bipap"], ["monitor", "o2_mask"]],
    "dka": [["monitor", "insulin", "maint_fluids"], ["monitor", "maint_fluids"]],
    "cardiac_ward": [["monitor", "heparin"], ["monitor", "saline_lock"], ["saline_lock"]],
    "psych_admit": [[], ["saline_lock"]],
    "ortho_ward": [["maint_fluids", "o2_low"], ["saline_lock", "o2_mid"], ["saline_lock"]],
    "pneumonia_ward": [["o2_mid", "saline_lock"], ["o2_mask", "maint_fluids"], ["o2_low"]],
    "care_home": [[], ["o2_low"]], "dialysis": [[], ["o2_low"]], "imaging": [[]], "rehab": [[], ["saline_lock"]],
}
CREW_KEYS = ["non_ambulance_transport", "bls_ambulance", "als_ambulance", "critical_care_ground", "air_medical"]
TIER_NAME = ["stretcher/wheelchair van", "BLS ambulance", "ALS (paramedic) ambulance", "critical care team",
             "helicopter (critical care)"]
RECEIVING_KIND = {"care_home": ["Care Home", "Nursing Home"], "dialysis": ["Dialysis Unit", "Kidney Center"],
                  "imaging": ["Imaging Center", "Diagnostic Center"], "rehab": ["Rehabilitation Centre",
                                                                                "Rehab Hospital"],
                  "psych_admit": ["Behavioral Health Center", "Mental Health Unit"]}
SITE_KIND = ["Community Hospital", "Medical Center", "Rural Health Clinic", "Memorial Hospital", "Regional Hospital",
             "Urgent Care Center"]


class InterfacilityTransferRequest(Schema):
    name = "interfacility_transfer_request"
    description = ("A hospital asks the communications center for a patient transfer: crew level, priority and "
                   "whether it can be dispatched now, under the written interfacility transport matrix.")
    surfaces = {"json": 0.5, "transfer_form": 0.25, "transfer_line_log": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        g = {"rev": f"IFT-{rng.randint(1, 9)}{pick(rng, ['a', 'b', 'c', ''])}", "air_min": pick(rng, [45, 60, 75]),
             "bls_o2": pick(rng, [4, 6]), "county": _county(rng)}
        cat = pick_w(rng, {"stemi": 1.3, "stroke_lvo": 1.2, "trauma": 1.1, "aortic": 0.6, "sepsis_icu": 1.2,
                           "gi_bleed": 1, "resp_failure": 1, "dka": 0.8, "cardiac_ward": 1.2, "psych_admit": 1,
                           "ortho_ward": 1, "pneumonia_ward": 1, "care_home": 1.4, "dialysis": 1.1, "imaging": 0.8,
                           "rehab": 0.9})
        label, klass, critical = IFT_CATS[cat]
        items = list(pick(rng, CAT_INTERVENTIONS[cat]))
        if g["bls_o2"] == 4 and "o2_mid" in items and chance(rng, 0.3):
            items[items.index("o2_mid")] = "o2_low"
        dist = int(rng.uniform(15, 90) if chance(rng, 0.6) else rng.uniform(90, 260))
        if reg == "borderline" and critical:
            dist = int(near(rng, g["air_min"] / 1.0, 15) * 1.0)
        ground_eta = max(12, int(dist * rng.uniform(0.85, 1.2)))
        if reg == "borderline" and critical:
            ground_eta = max(12, near(rng, g["air_min"], 10, integer=True))
        aircraft = chance(rng, 0.75)
        weather = pick_w(rng, {"flyable": 7, "marginal, pilot's call": 1.5, "grounded": 1.5})
        unstable = klass >= 1 and chance(rng, 0.25 if reg == "clear" else 0.35)
        sbp = rng.randint(78, 89) if unstable and chance(rng, 0.6) else rng.randint(96, 162)
        spo2 = rng.randint(86, 89) if unstable and sbp >= 90 else rng.randint(92, 99)
        hr = rng.randint(118, 142) if unstable else rng.randint(62, 112)
        req = self._required(items, g)
        if reg == "conflicting":
            requested = max(0, req - pick(rng, [1, 1, 2])) if req >= 1 else 1
        else:
            requested = pick_w(rng, {req: 6, max(0, req - 1): 2, min(3, req + 1): 1.2})
        accept_md = chance(rng, 0.68)
        payer = pick_w(rng, {"commercial plan": 3, "public plan": 3, "older-adult public plan": 2, "uninsured": 1.5,
                             "emergency coverage only (no permanent coverage)": 0.8})
        auth_hold = reg == "conflicting" and chance(rng, 0.5)
        declines_blood = cat in ("gi_bleed", "trauma") and chance(rng, 0.15)
        if declines_blood:
            items = [i for i in items if i != "blood"] or ["monitor", "maint_fluids"]
        age = (rng.randint(18, 94) if cat not in ("pneumonia_ward", "psych_admit", "trauma") or chance(rng, 0.8)
               else rng.randint(2, 15))
        bed = chance(rng, 0.55 if reg != "clear" else 0.7)
        nurse_view = None
        if reg == "conflicting":
            nurse_view = pick(rng, ["sending nurse says this is 'very urgent, please hurry'",
                                    "sending physician says 'any crew will do, they're stable'",
                                    "sending unit wants the patient gone before shift change"])
        t0 = rand_ts(rng, hours=(6, 22))
        county = g["county"]
        return {"regime": reg, "guide": g, "category": cat, "category_text": label, "class": klass,
                "time_critical": critical, "interventions": items, "distance_km": dist, "ground_eta_min": ground_eta,
                "aircraft_available": aircraft, "weather": weather, "vitals": {"hr": hr, "sbp": sbp, "spo2": spo2},
                "unstable": unstable, "requested_tier": requested, "age": age, "sex": pick(rng, ["F", "M"]),
                "payer": payer, "authorization_hold_requested": auth_hold, "declines_blood_products": declines_blood, "accepting_physician": accept_md,
                "bed_confirmed": bed, "sender_remark": nurse_view, "received": t0,
                "sending": f"{pick(rng, COUNTY_STEMS)} {pick(rng, SITE_KIND)}",
                "receiving": f"{pick(rng, COUNTY_STEMS)} " + pick(rng, RECEIVING_KIND.get(
                    cat, ["University Hospital", "Regional Medical Center", "General Hospital"]
                    + (["Heart Institute"] if cat in ("stemi", "cardiac_ward", "aortic") else []))),
                "request_id": f"T{rng.randint(10000, 99999)}", "clerk": person(rng), "sending_md": "Dr. " + person(rng),
                "accepting_md": ("Dr. " + person(rng)) if accept_md else None, "county": county}

    @staticmethod
    def _required(items, g):
        tier = 0
        titrated = 0
        for it in items:
            t = INTERVENTIONS[it][1]
            if it == "o2_mid" and g["bls_o2"] < 4:
                t = 2
            tier = max(tier, t)
            titrated += t == 3
        infusions = sum(1 for it in items if it in ("heparin", "ppi", "nitro", "insulin", "labetalol", "norepi",
                                                    "propofol", "maint_fluids"))
        if infusions >= 3:
            tier = 3
        return tier

    def intended(self, f):
        g = f["guide"]
        req = self._required(f["interventions"], g)
        if f["time_critical"]:
            req = max(req, 2)
        air = (f["time_critical"] and f["ground_eta_min"] > g["air_min"] and f["aircraft_available"]
               and f["weather"] == "flyable")
        crew = "air_medical" if air else CREW_KEYS[req]
        prio = f["class"]
        if f["unstable"] and prio < 2:
            prio = 2
        risk = {0: 0, 1: 0, 2: 1, 3: 2}[req]
        if f["unstable"] or "vent" in f["interventions"] or "norepi" in f["interventions"]:
            risk = min(3, risk + 1)
        if prio >= 2:
            ok_now = f["accepting_physician"]
        else:
            ok_now = f["bed_confirmed"]
        return {"crew_level": crew, "transfer_priority": prio, "en_route_intervention_likelihood": risk,
                "requested_level_sufficient": "yes" if f["requested_tier"] >= req else "no",
                "dispatch_now_permitted": "yes" if ok_now else "no"}

    @staticmethod
    def _slot_word(f):
        return {"dialysis": "appointment slot", "imaging": "appointment slot", "care_home": "place",
                "rehab": "place"}.get(f["category"], "bed")

    def _guide(self, f, rng):
        g = f["guide"]
        return {
            "title": pick(rng, [f"{g['county']} interfacility transport matrix {g['rev']}",
                                f"Transfer desk matrix ({g['county']} Joint Dispatch Authority, {g['rev']})"]),
            "van": "Van: stable, nothing running, oxygen no more than 2 L/min from own supply, no monitoring.",
            "bls": f"BLS: saline lock only, oxygen up to {g['bls_o2']} L/min, no cardiac monitoring.",
            "als": "ALS: cardiac monitoring, IV fluids, oxygen by mask, up to two fixed-rate infusions.",
            "critical_care": "Critical care team: any titrated infusion, vasopressor, blood product, ventilator or "
                             "BiPAP, or three or more infusions.",
            "air": f"Helicopter: time-critical diagnosis, ground time over {g['air_min']} min, aircraft available and "
                   f"weather flyable (marginal counts as not flyable).",
            "priority": "Immediate: time-critical diagnoses. Emergent: ICU-level transfers or SBP < 90, SpO2 < 90% "
                        "or HR > 130 on the latest vitals. Urgent: ward admissions. Scheduled: appointments and "
                        "discharges.",
            "release": "Immediate and emergent transfers go once an accepting physician is named; urgent and "
                       "scheduled transfers wait for a confirmed receiving bed, place or appointment slot. Payer status or a pending insurance "
                       "authorization never holds a transfer or lowers the crew level; the county transport assistance "
                       "fund covers patients without coverage.",
        }

    def render(self, f, rng, surface):
        gd = self._guide(f, rng)
        items = [INTERVENTIONS[i][0] for i in f["interventions"]] or ["nothing running, no oxygen"]
        v = f["vitals"]
        vit = f"HR {v['hr']}, BP {v['sbp']}/{max(40, int(v['sbp'] * 0.6))}, SpO2 {v['spo2']}%"
        req = TIER_NAME[f["requested_tier"]]
        t0 = f["received"]
        if surface == "json":
            return dumps(prune({
                "request": {"id": f["request_id"], "received": _hm(t0), "from": f["sending"], "to": f["receiving"],
                            "requested_by": f["sending_md"], "crew_requested": req, "sender_comment":
                            f["sender_remark"]},
                "patient": {"age": f["age"], "sex": f["sex"], "reason": f["category_text"], "running": items,
                            "latest_vitals": vit, "payer": f["payer"], "care_preferences":
                            "declines blood products (religious belief)" if f["declines_blood_products"] else None},
                "sending_unit_note": ("please hold until insurance authorization comes through"
                                      if f["authorization_hold_requested"] else None),
                "logistics": {"road_distance_km": f["distance_km"], "ground_time_min": f["ground_eta_min"],
                              "helicopter": "available" if f["aircraft_available"] else "unavailable",
                              "flying_weather": f["weather"]},
                "acceptance": {"accepting_physician": f["accepting_md"] or "not yet named",
                               "receiving_" + self._slot_word(f).replace(" ", "_"):
                               "confirmed" if f["bed_confirmed"] else "pending"},
                "matrix": gd,
            }))
        if surface == "transfer_form":
            rows = [("Request", f["request_id"]), ("Date/time", f"{fmt_date(t0.date(), rng)} {_hm(t0)}"),
                    ("Sending", f"{f['sending']} ({f['sending_md']})"), ("Receiving", f["receiving"]),
                    ("Patient", f"{f['age']} {f['sex']}"), ("Reason", f["category_text"]),
                    ("Running / support", "; ".join(items)), ("Latest vitals", vit), ("Payer", f["payer"]),
                    ("Care preferences", "declines blood products (religious belief)" if f["declines_blood_products"]
                     else None),
                    ("Sending unit note", "hold until insurance authorization is received"
                     if f["authorization_hold_requested"] else None),
                    ("Crew level requested", req), ("Distance / ground time",
                                                    f"{f['distance_km']} km / ~{f['ground_eta_min']} min"),
                    ("Helicopter", f"{'available' if f['aircraft_available'] else 'not available'}; weather "
                                   f"{f['weather']}"),
                    ("Accepting physician", f["accepting_md"] or "not yet named"),
                    (_cap(self._slot_word(f)), "confirmed" if f["bed_confirmed"] else "pending"),
                    ("Comment", f["sender_remark"])]
            return ("INTERFACILITY TRANSFER REQUEST\n" + kv_lines(rows, rng) + f"\n\n{gd['title']}\n"
                    + "\n".join(f"- {k}: {val}" for k, val in gd.items() if k != "title"))
        t = t0
        log = []
        for text in [f"call from {f['sending']}, {f['sending_md']}: {f['age']}{f['sex']}, {f['category_text']}",
                     f"patient has: {'; '.join(items)}", f"vitals {vit}",
                     f"they ask for {'an' if req[0] in 'AEIOU' else 'a'} {req}",
                     f"receiving {f['receiving']}: " + (f"accepted by {f['accepting_md']}" if f["accepting_md"]
                                                        else "accepting physician not yet named")
                     + (f"; {self._slot_word(f)} confirmed" if f["bed_confirmed"] else f"; {self._slot_word(f)} pending"),
                     f"road {f['distance_km']} km, ~{f['ground_eta_min']} min by road; helicopter "
                     f"{'available' if f['aircraft_available'] else 'unavailable'}, weather {f['weather']}"]:
            t += timedelta(minutes=rng.randint(1, 4))
            log.append(f"{_hm(t)} {text}")
        log.append(f"{_hm(t)} payer: {f['payer']}" + ("; patient declines blood products (religious belief)"
                                                    if f["declines_blood_products"] else ""))
        if f["authorization_hold_requested"]:
            log.append(f"{_hm(t)} admin asks us to wait for the insurance authorization before sending")
        if f["sender_remark"]:
            log.append(f"{_hm(t)} {f['sender_remark']}")
        return (f"TRANSFER DESK LOG {f['request_id']} ({f['clerk']})\n" + "\n".join(log) + f"\n\n{gd['title']}\n"
                + bullets([val for k, val in gd.items() if k != "title"], rng))

    def questions(self):
        return {
            "crew_level": choice("What level of crew and vehicle should take this transfer?", {
                "air_medical": "A helicopter with a critical care crew.",
                "als_ambulance": "A paramedic (advanced life support) ambulance.",
                "bls_ambulance": "A basic life support ambulance.",
                "critical_care_ground": "A ground critical care team.",
                "non_ambulance_transport": "A stretcher or wheelchair van, not an ambulance.",
            }),
            "transfer_priority": score("How soon must this transfer move under the matrix?", [
                "Scheduled: at the booked time.",
                "Urgent: within the next few hours, as a ward admission.",
                "Emergent: as soon as a suitable crew is free.",
                "Immediate: dispatch now; the diagnosis is time-critical.",
            ]),
            "en_route_intervention_likelihood": score(
                "How likely is the crew to have to intervene during the journey?", [
                    "Unlikely: a stable patient with nothing that needs adjusting.",
                    "Possible: monitoring or fluids that may need attention.",
                    "Likely: infusions to titrate or borderline vital signs.",
                    "Very likely: ventilated, on vasopressors, or unstable with critical-care support.",
                ]),
            "requested_level_sufficient": noul(
                "The crew level the sending hospital asked for covers everything this patient needs en route.",
                yes="The requested level is at or above the level the matrix requires for what is running.",
                no="Something running or required by the matrix is beyond the scope of the requested crew."),
            "dispatch_now_permitted": noul(
                "The matrix allows this transfer to be dispatched now.",
                yes="The acceptance condition for its priority band is already met.",
                no="It still waits on a named accepting physician or a confirmed bed, as its band requires."),
        }


# =========================================================================== schema 4: call update
UPDATE_EVENTS = {
    "stopped_breathing": (1, "upgrade", True, ["now not breathing", "has stopped breathing, caller panicking",
                                               "gone blue and isn't breathing"]),
    "unresponsive": (2, "upgrade", True, ["won't wake up now but still breathing", "has passed out, breathing"]),
    "bleeding_worse": (2, "upgrade", True, ["bleeding much heavier, soaking through towels",
                                            "bleeding won't stop even with pressure now"]),
    "seizure": (2, "upgrade", True, ["has started fitting", "now having a seizure on the floor"]),
    "chest_sweating": (2, "upgrade", False, ["now has chest pain and is sweating", "complaining of crushing chest "
                                                                                    "pain"]),
    "better_caller": (4, "downgrade", False, ["says the patient is up, walking and talking now",
                                              "patient feels much better, says not to bother"]),
    "better_responder": (4, "downgrade", False, ["first-responder crew on scene: minor injuries only, patient "
                                                 "walking", "engine crew on scene: patient alert, vitals normal"]),
    "minor_police": (4, "downgrade", False, ["police on scene: 'looks minor, they're sitting up'",
                                             "officer reports patient talking and only grazed"]),
    "eta_query": (None, "none", False, ["caller asking how long the ambulance will be", "caller asks for an ETA"]),
    "cost_cancel": (None, "none", False, ["caller asks to cancel: worried about the bill, no insurance",
                                          "caller says they can't afford an ambulance and will get a taxi"]),
    "extra_detail": (None, "none", False, ["caller adds the patient has taken paracetamol",
                                           "caller says the front door is unlocked"]),
}
ORIG_COMPLAINTS = ["fall in the kitchen", "abdominal pain", "breathing difficulty", "chest pain", "cut hand",
                   "unwell, weak", "allergic reaction", "dizziness", "back pain", "assault, facial injury",
                   "diabetic, low sugar", "headache", "fall down stairs", "faint at work", "leg injury at the park",
                   "vomiting", "sick person at the library", "pedal cyclist fell off"]


class CallUpdateReprioritize(Schema):
    name = "call_update_reprioritize"
    description = ("New information on a call that already has a unit assigned: revise the priority and the unit "
                   "plan under the written update and response-target rules.")
    surfaces = {"json": 0.5, "cad_event_log": 0.25, "unit_board": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        g = {"rev": f"{rng.randint(2, 8)}.{rng.randint(0, 9)}", "county": _county(rng),
             "targets": {"1": pick(rng, [8, 10, 12]), "2": pick(rng, [12, 15]), "3": pick(rng, [20, 25, 30]),
                         "4": pick(rng, [30, 45, 60])}}
        orig = pick_w(rng, {1: 1, 2: 3, 3: 4, 4: 2.5})
        if reg == "clear":
            ev = pick_w(rng, {"stopped_breathing": 1, "bleeding_worse": 1, "better_responder": 1.2, "eta_query": 1.2})
        elif reg == "borderline":
            ev = pick_w(rng, {"unresponsive": 1, "seizure": 1, "chest_sweating": 1, "better_responder": 1,
                              "extra_detail": 1, "eta_query": 0.6, "stopped_breathing": 0.5})
        else:
            ev = pick_w(rng, {"better_caller": 1.6, "minor_police": 1.4, "stopped_breathing": 0.6,
                              "chest_sweating": 0.6, "better_responder": 0.6, "cost_cancel": 1.0})
        if UPDATE_EVENTS[ev][1] == "downgrade" and orig >= 4:
            orig = pick(rng, [2, 3])
        assigned_kind = pick_w(rng, {"als": 5, "bls": 3})
        names = _unit_names(rng, [assigned_kind, "als", "als", "bls", "engine"])
        assigned = {"unit": names[0], "kind": assigned_kind, "eta": rng.randint(2, 16)}
        elapsed = rng.randint(2, 14)
        rev = self._rev(orig, ev)
        tgt = g["targets"][str(rev)]
        if reg == "borderline":
            assigned["eta"] = max(1, near(rng, tgt - elapsed, 3, integer=True))
        others = []
        for j, (nm, kind) in enumerate(zip(names[1:], ["als", "als", "bls", "engine"])):
            status = pick_w(rng, {"available": 4, "clearing hospital": 1.5, "committed": 2})
            if j == 0 or kind == "engine":
                status = "available"            # one ALS unit and the local engine are always free somewhere
            eta = rng.randint(2, 22) if status != "committed" else None
            if kind == "engine":
                eta = rng.randint(2, 8)
            if status == "clearing hospital" and eta is not None:
                eta += rng.randint(5, 12)
            others.append({"unit": nm, "kind": kind, "status": status, "eta": eta})
        rng.shuffle(others)
        source = {"better_responder": "first-responder crew", "minor_police": "police officer",
                  "better_caller": "caller"}.get(ev, pick(rng, ["caller (callback)", "caller (called back in)"]))
        t0 = rand_ts(rng)
        return {"regime": reg, "guide": g, "original_priority": orig, "event": ev,
                "event_text": pick(rng, UPDATE_EVENTS[ev][3]), "source": source, "assigned": assigned,
                "others": others, "elapsed_min": elapsed, "created": t0, "cad": _cad(rng, t0),
                "complaint": pick(rng, ORIG_COMPLAINTS), "address": _address(rng), "dispatcher": person(rng)}

    @staticmethod
    def _rev(orig, ev):
        new, kind, _, _ = UPDATE_EVENTS[ev]
        if kind == "upgrade":
            return min(orig, new)
        if kind == "downgrade" and ev == "better_responder":
            return max(orig, new)
        return orig

    def intended(self, f):
        g = f["guide"]
        orig, ev = f["original_priority"], f["event"]
        rev = self._rev(orig, ev)
        kind = UPDATE_EVENTS[ev][1]
        if rev < orig:
            handling = "upgrade_priority"
        elif rev > orig:
            handling = "downgrade_priority"
        elif kind == "downgrade":
            handling = "hold_pending_confirmation"
        else:
            handling = "keep_priority"
        tgt = g["targets"][str(rev)]
        need_als = rev <= 2
        a = f["assigned"]
        total = f["elapsed_min"] + a["eta"]
        wrong_type = need_als and a["kind"] != "als"
        if handling == "downgrade_priority" and orig <= 2 and rev >= 3:
            plan = "continue_without_lights"
        elif not wrong_type and total <= tgt:
            plan = "keep_assigned_unit"
        elif wrong_type:
            plan = "send_als_as_well"
        else:
            cands = [u for u in f["others"] if u["status"] != "committed" and u["eta"] is not None
                     and (u["kind"] == "als" or (not need_als and u["kind"] == "bls"))
                     and f["elapsed_min"] + u["eta"] <= tgt and u["eta"] < a["eta"] - 2]
            plan = "swap_to_closer_unit" if cands else "add_first_responder"
        diff = total - tgt
        if wrong_type or diff > 7:
            gap = 3
        elif diff > 2:
            gap = 2
        elif diff >= -2:
            gap = 1
        else:
            gap = 0
        return {"revised_priority": 4 - rev, "assigned_arrival_vs_target": gap, "update_handling": handling,
                "unit_plan": plan, "new_caller_instructions": "yes" if UPDATE_EVENTS[ev][2] else "no"}

    def _guide(self, f, rng):
        g, t = f["guide"], f["guide"]["targets"]
        return {
            "title": pick(rng, [f"Call update procedure -- {g['county']} dispatch guide {g['rev']}",
                                f"{g['county']} County: changing a call after dispatch (rev {g['rev']})"]),
            "targets": f"Response targets from call creation: P1 {t['1']} min, P2 {t['2']} min, P3 {t['3']} min, "
                       f"P4 {t['4']} min. P1-P2 need an ALS unit; a BLS unit does not meet them.",
            "upgrade_downgrade": "Raise the priority on new information from anyone. Lower it only on the word of a "
                                 "crew or clinician on scene; police or caller reports alone do not lower it. A "
                                 "request to cancel over cost or immigration worries changes nothing: the response "
                                 "keeps coming.",
            "new_codes": "Not breathing: P1. Unresponsive but breathing, heavy uncontrolled bleeding, an active "
                         "seizure, or chest pain with sweating: P2. Responder confirms minor: P4.",
            "units": "A BLS unit holding a P1-P2 call keeps going and the closest ALS unit is sent as well. If the "
                     "assigned unit is the right type but cannot meet the target, swap to an available unit of that "
                     "type that can; if none can, add the nearest engine as first responder. A call lowered from "
                     "P1-P2 to P3-P4 continues without lights.",
        }

    def render(self, f, rng, surface):
        gd = self._guide(f, rng)
        a = f["assigned"]
        t0 = f["created"]
        now = t0 + timedelta(minutes=f["elapsed_min"])
        def ustat(u):  # noqa: E306
            if u["status"] == "committed":
                return "committed on another call"
            return f"{u['status']}, ETA {u['eta']} min" if u["status"] == "available" else \
                f"{u['status']}, could arrive in ~{u['eta']} min"
        if surface == "json":
            return dumps(prune({
                "call": {"cad": f["cad"], "created": _hms(t0), "now": _hms(now), "complaint": f["complaint"],
                         "address": f["address"], "current_priority": f"P{f['original_priority']}"},
                "update": {"from": f["source"], "text": f["event_text"]},
                "assigned_unit": {"unit": a["unit"], "type": a["kind"].upper(), "eta_min": a["eta"]},
                "other_units": [{"unit": u["unit"], "type": "ENGINE" if u["kind"] == "engine" else u["kind"].upper(),
                                 "status": ustat(u)} for u in f["others"]],
                "guide": gd,
            }))
        if surface == "cad_event_log":
            ev = [(t0, f"CREATED P{f['original_priority']}  {f['complaint']}  {f['address']}"),
                  (t0 + timedelta(seconds=rng.randint(40, 110)), f"DISP {a['unit']} ({a['kind'].upper()})"),
                  (now - timedelta(seconds=rng.randint(20, 60)), f"UPDATE from {f['source']}: {f['event_text']}"),
                  (now, f"{a['unit']} enroute, ETA {a['eta']} min")]
            board = "\n".join(f"   {u['unit']:<10} {('ENGINE' if u['kind'] == 'engine' else u['kind'].upper()):<6} "
                              f"{ustat(u)}" for u in f["others"])
            return (f"CAD {f['cad']} ({f['dispatcher']})\n" + "\n".join(f"{_hms(t)}  {e}" for t, e in ev)
                    + f"\n\nOther units:\n{board}\n\n{gd['title']}\n"
                    + "\n".join(f"- {v}" for k, v in gd.items() if k != "title"))
        rows = [[a["unit"], a["kind"].upper(), "assigned to this call", f"ETA {a['eta']} min"]]
        rows += [[u["unit"], "ENGINE" if u["kind"] == "engine" else u["kind"].upper(), u["status"],
                  "-" if u["eta"] is None else f"~{u['eta']} min"] for u in f["others"]]
        head = (f"Call {f['cad']}: P{f['original_priority']} {f['complaint']}, {f['address']}. Created {_hm(t0)}, "
                f"now {_hm(now)}.\nNew from {f['source']}: \"{noisy(f['event_text'], rng, 0.08)}\"")
        return (f"UNIT STATUS BOARD -- {f['dispatcher']}\n{head}\n\n" + table(["unit", "type", "status", "to scene"],
                                                                            rows, rng)
                + f"\n\n{gd['title']}\n" + bullets([v for k, v in gd.items() if k != "title"], rng))

    def questions(self):
        return {
            "revised_priority": score("What priority should the call carry after this update?", [
                "P4: non-urgent.",
                "P3: urgent, no lights.",
                "P2: emergency, lights and sirens.",
                "P1: immediate life threat.",
            ]),
            "assigned_arrival_vs_target": score(
                "How does the assigned unit's arrival compare with the target for the priority that now applies?", [
                    "Comfortably inside the target, with three minutes or more to spare.",
                    "Close to the target, within two minutes either side.",
                    "Late by three to seven minutes.",
                    "Late by more than seven minutes, or the unit is the wrong type for the priority.",
                ]),
            "update_handling": choice("How should the dispatcher treat this update?", {
                "downgrade_priority": "Lower the call's priority on the strength of the update.",
                "hold_pending_confirmation": "Leave the priority as it is until a crew or clinician on scene "
                                             "confirms the improvement.",
                "keep_priority": "Leave the priority as it is; the update does not change it.",
                "upgrade_priority": "Raise the call's priority.",
            }),
            "unit_plan": choice("What should happen with the units on this call?", {
                "add_first_responder": "Keep the assigned unit and add the nearest engine as first responder.",
                "send_als_as_well": "Keep the assigned crew coming and send the closest ALS unit as well.",
                "continue_without_lights": "Keep the assigned unit but switch it to normal road speed.",
                "keep_assigned_unit": "Keep the assigned unit coming as it is.",
                "swap_to_closer_unit": "Reassign the call to a closer available unit of the right type.",
            }),
            "new_caller_instructions": noul(
                "The call-taker should give the caller new instructions to carry out before help arrives."),
        }


# =========================================================================== schema 5: pending queue
QUEUE_COMPLAINTS = {
    1: ["not breathing", "unconscious, gasping", "cardiac arrest reported by a passer-by"],
    2: ["chest pain with sweating", "stroke signs within the hour", "heavy bleeding", "seizure, first ever",
        "breathing difficulty, can't speak"],
    3: ["fall with a leg deformity", "abdominal pain", "allergic reaction, hives", "diabetic, drowsy but eating",
        "assault, facial cuts", "back pain with leg numbness"],
    4: ["fall, uninjured, needs help up", "back pain", "minor head bump, alert", "generally unwell",
        "cough and fever, declined nurse line", "dog bite to the hand"],
}
REFERRAL_ELIGIBLE = {"cough and fever, declined nurse line": False, "generally unwell": True, "back pain": True,
                     "minor head bump, alert": False}


class PendingQueueAssignment(Schema):
    name = "pending_queue_assignment"
    description = ("Several calls waiting and one unit coming free: which call gets it, what happens to the rest, "
                   "and whether to ask a neighbouring agency for help, under the written queue rules.")
    surfaces = {"json": 0.5, "status_board": 0.25, "dispatch_chat": 0.25}

    LABELS = ["A", "B", "C", "D"]

    def sample_factors(self, rng):
        reg = regime(rng)
        g = {"rev": f"Q{rng.randint(1, 9)}.{rng.randint(0, 9)}", "county": _county(rng),
             "hold": {"1": 2, "2": pick(rng, [5, 8, 10]), "3": pick(rng, [20, 30]), "4": pick(rng, [45, 60])},
             "mutual_aid_min": pick(rng, [10, 15, 20]), "bls_first_min": pick(rng, [8, 10])}
        unit_kind = pick_w(rng, {"als": 6, "bls": 4})
        prios = [pick_w(rng, {1: 0.4, 2: 2.6, 3: 3.2, 4: 2.8}) for _ in range(4)]
        if reg == "clear" and chance(rng, 0.5):
            prios[0] = 2
        pressure = pick_w(rng, {"light": 3, "building": 4, "heavy": 2.2})
        near_i = rng.randrange(4)
        calls = []
        for i, p in enumerate(prios):
            lim = g["hold"][str(p)]
            if p == 1:
                wait = rng.randint(0, 3)
            elif pressure == "light":
                wait = max(1, int(lim * rng.uniform(0.1, 0.6)))
            elif pressure == "building":
                wait = (max(1, near(rng, lim, max(3, lim * 0.25), integer=True)) if i == near_i
                        else max(1, int(lim * rng.uniform(0.1, 0.75))))
            else:
                wait = rng.randint(1, int(lim * 1.8) + 3)
            cmp = pick(rng, QUEUE_COMPLAINTS[p])
            calls.append({"priority": p, "wait": wait, "complaint": cmp, "eta": rng.randint(3, 22),
                          "address": _address(rng), "note": None,
                          "context": pick_w(rng, {None: 6, "caller at the night shelter": 1,
                                                  "frequent caller (9 calls this month)": 1,
                                                  "caller prefers Spanish, language line used": 0.8,
                                                  "no fixed address, calling from a payphone": 0.6})})
        rng.shuffle(calls)
        for lab, c in zip(self.LABELS, calls):
            c["label"] = lab
        # a caller update on one of the lower calls
        low = [c for c in calls if c["priority"] >= 3]
        if low and chance(rng, 0.4):
            c = pick(rng, low)
            c["note"] = pick(rng, ["caller rang back: still waiting, patient cold and in pain",
                                   "caller rang back asking for an ETA", "caller says they could get a lift instead"])
        # other units: committed, some clearing
        n_units = rng.randint(2, 5)
        kinds = [pick_w(rng, {"als": 5, "bls": 3}) for _ in range(n_units)]
        names = _unit_names(rng, [unit_kind] + kinds)
        board = []
        for nm, k in zip(names[1:], kinds):
            st = pick_w(rng, {"on scene": 2, "transporting": 2, "at hospital": 2, "out of service": 0.6})
            clear = None if st == "out of service" else (rng.randint(3, 15) if st == "at hospital" else
                                                         rng.randint(10, 50))
            if reg == "conflicting" and k == "als" and clear is not None and chance(rng, 0.5):
                clear = rng.randint(g["mutual_aid_min"] + 3, 55)
            if clear is not None and pressure == "heavy":
                clear += rng.randint(8, 25)       # a busy evening: units are slow to come back as well
            elif clear is not None and pressure == "building" and k == "als":
                clear += rng.randint(0, 10)
            board.append({"unit": nm, "kind": k, "status": st, "clears_in": clear})
        now = rand_ts(rng)
        supervisor = None
        if reg == "conflicting":
            supervisor = pick(rng, ["supervisor: 'take the closest call first, we need units back fast'",
                                    "supervisor: 'don't call for mutual aid unless we're at zero'",
                                    "supervisor suggests sending the free unit to the oldest call",
                                    "supervisor: 'the shelter one calls every night, put them last'"])
        return {"regime": reg, "guide": g, "queue_pressure": pressure, "free_unit": names[0], "free_kind": unit_kind,
                "calls": calls,
                "board": board, "now": now, "supervisor_remark": supervisor, "dispatcher": person(rng)}

    def _eff(self, f, c):
        lim = f["guide"]["hold"][str(c["priority"])]
        if c["priority"] in (3, 4) and c["wait"] > lim:
            return c["priority"] - 1
        return c["priority"]

    def _choose(self, f):
        g = f["guide"]
        als_soon = any(u["kind"] == "als" and u["clears_in"] is not None and u["clears_in"] <= g["bls_first_min"]
                       for u in f["board"])
        elig = []
        for c in f["calls"]:
            e = self._eff(f, c)
            if f["free_kind"] == "bls" and e <= 2 and als_soon:
                continue
            elig.append((e, -c["wait"], c["eta"], c["label"], c))
        if not elig:
            elig = [(self._eff(f, c), -c["wait"], c["eta"], c["label"], c) for c in f["calls"]]
        return min(elig)[4]

    def intended(self, f):
        g = f["guide"]
        chosen = self._choose(f)
        rest = [c for c in f["calls"] if c is not chosen]
        over = [(c, c["wait"] - g["hold"][str(c["priority"])]) for c in f["calls"]
                if c["wait"] > g["hold"][str(c["priority"])]]
        if any(c["priority"] <= 2 for c, _ in over):
            breach = 3
        elif any(d > 15 for _, d in over) or len(over) >= 2:
            breach = 2
        elif over:
            breach = 1
        else:
            breach = 0
        if any(c["wait"] > g["hold"][str(c["priority"])] for c in rest):
            follow = "welfare_callback"
        elif any(REFERRAL_ELIGIBLE.get(c["complaint"], False) for c in rest):
            follow = "offer_nurse_line"
        elif any(c["wait"] > g["hold"][str(c["priority"])] / 2 for c in rest):
            follow = "give_eta_update"
        else:
            follow = "no_action_needed"
        soon = [u for u in f["board"] if u["clears_in"] is not None and u["clears_in"] <= 10]
        later = [u for u in f["board"] if u["clears_in"] is not None and u["clears_in"] <= 30]
        status = 0 if len(soon) >= 2 else (1 if soon else (2 if later else 3))
        als_ma = any(u["kind"] == "als" and u["clears_in"] is not None and u["clears_in"] <= g["mutual_aid_min"]
                     for u in f["board"])
        urgent_left = any(self._eff(f, c) <= 2 for c in rest)
        aid = (urgent_left and not als_ma) or (len(rest) >= 2 and not later)
        return {"next_unit_assignment": "call_" + chosen["label"].lower(), "hold_limit_breach": breach,
                "system_depletion": status, "remaining_calls_followup": follow,
                "mutual_aid_request": "yes" if aid else "no"}

    def _guide(self, f, rng):
        g = f["guide"]
        h = g["hold"]
        return {
            "title": pick(rng, [f"Queue management rules {g['rev']} -- {g['county']}",
                                f"{g['county']} pending-call rules (supervisor card {g['rev']})"]),
            "hold_limits": f"Hold limits: P1 {h['1']} min, P2 {h['2']} min, P3 {h['3']} min, P4 {h['4']} min.",
            "order": "A free unit goes to the most urgent pending call; ties go to the longest wait, then the "
                     "shortest drive. A P3 or P4 call past its hold limit is treated one level more urgent. Who the "
                     "caller is, how often they call or where they live never changes the order.",
            "bls": f"A BLS unit takes P3-P4 calls; it may take a P1-P2 call only when no ALS unit clears within "
                   f"{g['bls_first_min']} min.",
            "callers": "Calls left past their hold limit get a welfare callback; P4 calls that suit the nurse line "
                       "are offered it; calls past half their limit get an ETA update.",
            "mutual_aid": f"Request mutual aid when a P1-P2 call (after the age rule) is left waiting with no ALS unit "
                          f"clearing within {g['mutual_aid_min']} min, or when two or more calls are left and no unit "
                          f"clears within 30 min.",
        }

    def render(self, f, rng, surface):
        gd = self._guide(f, rng)
        now = f["now"]

        def clr(u):
            return "out of service" if u["clears_in"] is None else f"clears in ~{u['clears_in']} min"
        if surface == "json":
            return dumps(prune({
                "status": {"time": _hm(now), "free_unit": f"{f['free_unit']} ({f['free_kind'].upper()})",
                           "supervisor": f["supervisor_remark"]},
                "pending_calls": [{"call": c["label"], "priority": f"P{c['priority']}", "complaint": c["complaint"],
                                   "caller_context": c["context"],
                                   "received": _hm(now - timedelta(minutes=c["wait"])),
                                   "drive_from_free_unit_min": c["eta"], "update": c["note"]} for c in f["calls"]],
                "other_units": [{"unit": u["unit"], "type": u["kind"].upper(), "status": u["status"],
                                 "availability": clr(u)} for u in f["board"]],
                "rules": gd,
            }))
        if surface == "status_board":
            q = table(["call", "pri", "complaint", "rcvd", "drive", "update"],
                      [[c["label"], f"P{c['priority']}", c["complaint"], _hm(now - timedelta(minutes=c["wait"])),
                        f"{c['eta']} min", "; ".join(x for x in (c["note"], c["context"]) if x)] for c in f["calls"]],
                      rng, "fixed")
            u = table(["unit", "type", "status", "free"],
                      [[x["unit"], x["kind"].upper(), x["status"], clr(x)] for x in f["board"]], rng, "fixed")
            return (f"SYSTEM STATUS {_hm(now)} -- {f['free_unit']} ({f['free_kind'].upper()}) now available\n\n"
                    f"PENDING\n{q}\n\nUNITS\n{u}\n"
                    + (f"\n{f['supervisor_remark']}\n" if f["supervisor_remark"] else "")
                    + f"\n{gd['title']}\n" + "\n".join(f"- {v}" for k, v in gd.items() if k != "title"))
        t = now - timedelta(minutes=2)
        lines = []

        def say(who, text):
            nonlocal t
            t += timedelta(seconds=rng.randint(10, 50))
            lines.append(f"[{_hm(t)}] {who}: {text}")
        me = f["dispatcher"].split()[0].lower()
        say(me, f"{f['free_unit']} ({f['free_kind'].upper()}) just went available")
        for c in f["calls"]:
            say(me, f"call {c['label']}: P{c['priority']} {c['complaint']}, in since "
                    f"{_hm(now - timedelta(minutes=c['wait']))}, {c['eta']} min drive" +
                (f" ({c['note']})" if c["note"] else "") + (f" [{c['context']}]" if c["context"] else ""))
        say("board", "; ".join(f"{u['unit']} {u['kind'].upper()} {clr(u)}" for u in f["board"]))
        if f["supervisor_remark"]:
            say("sup", f["supervisor_remark"].split(": ", 1)[-1].strip("'"))
        return ("#dispatch-floor\n" + "\n".join(noisy(x, rng, 0.05) for x in lines) + f"\n\n{gd['title']}\n"
                + bullets([v for k, v in gd.items() if k != "title"], rng))

    def questions(self):
        return {
            "next_unit_assignment": choice("Which pending call should the free unit be sent to?", {
                "call_a": "The pending call labelled A.",
                "call_b": "The pending call labelled B.",
                "call_c": "The pending call labelled C.",
                "call_d": "The pending call labelled D.",
            }),
            "hold_limit_breach": score("How far past their hold limits are the pending calls right now?", [
                "No pending call is past its hold limit.",
                "One lower-priority call is past its limit, by fifteen minutes or less.",
                "A lower-priority call is more than fifteen minutes past its limit, or several calls are past "
                "theirs.",
                "A P1 or P2 call is already past its hold limit.",
            ]),
            "system_depletion": score("How depleted is the system once the free unit is committed?", [
                "Healthy: two or more units clear within ten minutes.",
                "Thin: exactly one unit clears within ten minutes.",
                "Strained: nothing clears within ten minutes, but something does within thirty.",
                "Exhausted: no unit is expected to clear within thirty minutes.",
            ]),
            "remaining_calls_followup": choice("What should be done for the callers still waiting?", {
                "give_eta_update": "Call them back with an updated estimate of when a unit will come.",
                "no_action_needed": "Nothing yet; they are well within their hold limits.",
                "offer_nurse_line": "Offer a suitable low-priority caller the nurse advice line instead.",
                "welfare_callback": "Make a welfare callback to a caller who has waited past the hold limit.",
            }),
            "mutual_aid_request": noul(
                "The center should ask a neighbouring agency for mutual aid now.",
                yes="An urgent call is left waiting with no ALS unit clearing in time, or several calls are left "
                    "with nothing clearing within thirty minutes, as the rules define.",
                no="The units clearing soon can cover what is left under the written rules."),
        }


WORKFLOW = Workflow(
    name="ems_dispatch_priority",
    title="EMS dispatch priority",
    description="What priority and response a written dispatch guide gives an emergency call, and how dispatchers "
                "handle collisions, interfacility transfers, call updates and a queue of pending calls under it.",
    domain="emergency services / EMS dispatch",
    schemas=[MedicalCallPriority(), TrafficCollisionCall(), InterfacilityTransferRequest(), CallUpdateReprioritize(),
             PendingQueueAssignment()],
    notes="All agencies, guides, thresholds, units, streets and people are fictional; the P1-P5 scale, complaint "
          "cards, matrices and queue rules are invented and are not real dispatch protocols or clinical guidance. "
          "No patient age, sex or other protected attribute is used. No ED triage acuity, no hospital capacity or "
          "diversion decisions.",
)
