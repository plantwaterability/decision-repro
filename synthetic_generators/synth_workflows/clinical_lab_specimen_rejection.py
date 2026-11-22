"""Clinical laboratory specimen rejection: accept, reject or conditionally accept specimens at a (fictional)
hospital laboratory against its written pre-analytical rules.

Five schemas: a chemistry specimen with haemolysis, delayed separation or suspected EDTA carry-over; a blood-bank
sample under a zero-tolerance labelling rule; a microbiology specimen against transport and quality rules; a
citrate coagulation sample (fill, haematocrit, clots, line contamination, stability); and a specimen shipped in
from an outside clinic with a temperature-logger excursion or paperwork gaps.

Everything is invented: patients (base.person), record numbers, wards, clinics, collectors and rule numbers.
Every limit in a state (per-analyte haemolysis-index cut-offs, stability times, fill tolerances, labelling and
transport rules, the referred-assay stability table) is this fictional laboratory's LOCAL rule, written for the
generator and rendered into the state; none of it is presented as real clinical guidance. Drugs appear by class
only. The lab-to-ward phone log is an internal record; no question asks how to reply to anyone. Date of birth
appears only as an identifier to match; no protected attribute decides anything.

Each schema draws a `regime` first: borderline cases sit next to a written limit (a haemolysis index of 44
against a limit of 40, a tube filled to 88% against 90%, a urine 2 h 10 min old against a 2-hour rule);
conflicting cases pair a clean-looking specimen with one disqualifying fact, or a clinician's pressure with a
rule that does not bend. `answer_key` on each schema is a rule-of-thumb key used only for the spread check; it is
never a label.
"""
from __future__ import annotations

from datetime import date, timedelta

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
    near,
    noisy,
    noul,
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

WARDS = ["Emergency Department", "Intensive Care Unit", "Acute Medical Unit", "Ward 7B (cardiology)",
         "Ward 4A (renal)", "Ward 12 (general surgery)", "Ward 9 (respiratory)", "Ward 3C (haematology)",
         "Ward 6 (orthopaedics)", "Ward 10 (gastroenterology)", "Ward 2 (care of older people)",
         "Ward 15 (oncology)", "Delivery Suite", "Ward 5 (stroke)", "High Dependency Unit", "Day Surgery Unit"]
CLINICS = ["Fernhill Medical Practice", "Oakridge Family Practice", "Lakeshore Health Centre",
           "Redfern Surgery", "Valewood Medical Group", "Upland Rheumatology Clinic", "Westbrook Renal Clinic",
           "Saltmarsh Endocrine Clinic", "Driftwood Neurology Clinic", "Basalt Haematology Clinic",
           "Copperline Gastro Clinic", "Ashgrove Diabetes Centre", "Pinecrest Paediatric Clinic",
           "Yarrow Fertility Clinic", "Elmstead Hepatology Clinic", "Kestrel Metabolic Clinic"]


def _mrn(rng):
    return f"{pick(rng, ['H', 'M', 'R', 'K'])}{rng.randrange(1000000, 9999999)}"


def _dob(rng):
    return rand_date(rng, date(1938, 1, 1), date(2004, 12, 31))


def _hm(minutes):
    h, m = divmod(int(minutes), 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"


# =========================================================================== schema 1
# analyte -> (HI limit, HI comment band upper bound or None, unspun stability hours, EDTA-sensitive)
ANALYTES = {
    "potassium": (40, 100, 4, True), "LDH": (25, None, 8, False), "AST": (70, 150, 8, False),
    "ALT": (250, None, 8, False), "magnesium": (150, 300, 8, True), "phosphate": (250, None, 4, False),
    "creatinine": (600, None, 12, False), "sodium": (600, None, 12, False), "urea": (600, None, 12, False),
    "glucose (no fluoride)": (600, None, 2, False), "total bilirubin": (120, 250, 8, False),
    "hs-troponin": (90, None, 8, False), "CK": (180, 350, 8, False), "iron": (100, None, 8, False),
    "serum folate": (40, None, 8, False), "calcium": (600, None, 8, True), "ALP": (500, None, 8, True),
    "lipase": (300, None, 8, False),
}
PANELS = {
    "renal profile": ["sodium", "potassium", "urea", "creatinine"],
    "bone profile": ["calcium", "phosphate", "ALP"],
    "liver profile": ["ALT", "ALP", "total bilirubin"],
    "magnesium": ["magnesium"], "LDH": ["LDH"], "AST": ["AST"], "hs-troponin": ["hs-troponin"], "CK": ["CK"],
    "iron": ["iron"], "serum folate": ["serum folate"], "glucose": ["glucose (no fluoride)"],
    "lipase": ["lipase"], "phosphate": ["phosphate"], "potassium": ["potassium"],
}
# clinical question -> (key analyte, panels that come with it, urgency 0-3 by default)
QUESTIONS = {
    "possible hyperkalaemia on the ECG": ("potassium", ["renal profile"], 3),
    "chest pain, repeat troponin at 3 hours": ("hs-troponin", ["hs-troponin", "renal profile"], 3),
    "acute kidney injury monitoring": ("creatinine", ["renal profile", "bone profile"], 2),
    "refeeding risk, daily electrolytes": ("phosphate", ["bone profile", "magnesium", "renal profile"], 2),
    "jaundice under investigation": ("total bilirubin", ["liver profile", "AST"], 2),
    "rhabdomyolysis follow-up": ("CK", ["CK", "renal profile"], 2),
    "anaemia work-up": ("iron", ["iron", "serum folate", "LDH"], 1),
    "suspected pancreatitis": ("lipase", ["lipase", "liver profile"], 3),
    "pre-operative bloods for next week": ("creatinine", ["renal profile", "liver profile"], 1),
    "routine annual review": ("creatinine", ["renal profile", "liver profile", "bone profile"], 0),
    "diabetes review": ("glucose (no fluoride)", ["glucose", "renal profile"], 0),
    "magnesium replacement monitoring": ("magnesium", ["magnesium", "potassium"], 2),
}
IN_VIVO = ["a mechanical aortic valve with known low-grade haemolysis", "hereditary spherocytosis",
           "a positive haemolysis screen earlier this week", "support on ECMO",
           "a suspected delayed haemolytic transfusion reaction"]


class ChemistrySpecimenAcceptance(Schema):
    name = "chemistry_specimen_acceptance"
    description = ("A serum/plasma chemistry specimen flagged at receipt for haemolysis, delayed separation or "
                   "suspected EDTA carry-over, checked against the lab's analyte limits.")
    surfaces = {"json": 0.5, "lis_receipt": 0.25, "bench_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        question = pick(rng, list(QUESTIONS))
        key, panels, urg = QUESTIONS[question]
        panels = list(panels)
        if chance(rng, 0.3):
            extra = pick(rng, [p for p in PANELS if p not in panels])
            panels.append(extra)
        ordered = []
        for p in panels:
            for a in PANELS[p]:
                if a not in ordered:
                    ordered.append(a)
        if urg >= 2:
            loc = pick(rng, ["Emergency Department", "Intensive Care Unit", "Acute Medical Unit",
                             "High Dependency Unit"] if urg == 3 else WARDS[2:])
        else:
            loc = pick(rng, CLINICS + ["Outpatient phlebotomy"])
        outpatient = loc in CLINICS or loc == "Outpatient phlebotomy"
        # the binding haemolysis limit: the lowest one among the ordered analytes
        lim_an = min(ordered, key=lambda a: ANALYTES[a][0])
        lim = ANALYTES[lim_an][0]
        key_lim, key_band = ANALYTES[key][0], ANALYTES[key][1]
        strict_h = min(ANALYTES[a][2] for a in ordered)
        problem = pick_w(rng, {"haemolysis": 5, "delay": 4, "edta": 1.2, "none": 1.5})
        hi = rng.randint(2, max(3, int(lim * 0.6)))
        mins = rng.randint(25, int(strict_h * 60 * 0.7))
        edta = False
        in_vivo = None
        if outpatient and chance(rng, 0.6):
            mins = rng.randint(int(strict_h * 45), int(strict_h * 60 * 2.0))     # courier from a clinic
        banded = [a for a in ordered if ANALYTES[a][1]]
        if problem == "haemolysis" and banded and chance(rng, 0.35):
            # inside a comment band: reportable with a comment unless a stricter analyte is also ordered
            a = pick(rng, banded)
            hi = rng.randint(ANALYTES[a][0] + 1, ANALYTES[a][1])
        elif problem == "haemolysis":
            target = pick(rng, [key_lim, key_band or key_lim * 2, lim, lim * 3])
            if reg == "borderline":
                hi = near(rng, target, max(4, target * 0.12), lo=1, integer=True)
            elif reg == "clear":
                hi = pick(rng, [rng.randint(1, max(2, int(lim * 0.5))), int(target * rng.uniform(1.6, 3.5))])
            else:
                hi = int(target * rng.uniform(0.9, 2.6))
        elif problem == "delay":
            if reg == "borderline":
                mins = near(rng, strict_h * 60, 20, lo=20, integer=True)
            else:
                mins = int(strict_h * 60 * rng.uniform(1.15, 2.8)) if reg != "clear" or chance(rng, 0.6) else mins
        elif problem == "edta":
            edta = "potassium" in ordered or "calcium" in ordered
        if reg == "conflicting":
            # haemolysis that may be real (in vivo), or a clinician who insists a striking result is genuine
            if chance(rng, 0.6) and hi > lim:
                in_vivo = pick(rng, IN_VIVO)
            elif chance(rng, 0.5) and "potassium" in ordered:
                edta = True
        if edta and not ("potassium" in ordered or "calcium" in ordered):
            edta = False
        # first-pass values held for review
        k_val = None
        if "potassium" in ordered:
            k_val = round(rng.uniform(3.6, 4.9) + (hi / 100) * 0.4 + (mins > 240) * rng.uniform(0.4, 1.2), 1)
            if edta:
                k_val = round(rng.uniform(7.2, 11.5), 1)
        ca_val = None
        if "calcium" in ordered or edta or (k_val is not None and k_val > 6.0 and chance(rng, 0.7)):
            # the lab adds a calcium as an EDTA check whenever a potassium looks implausible
            ca_val = round(rng.uniform(0.6, 1.5), 2) if edta else round(rng.uniform(2.1, 2.55), 2)
        prev_k = round(rng.uniform(3.7, 4.8), 1) if k_val is not None and chance(rng, 0.55) else None
        return {"regime": reg, "question": question, "key_analyte": key, "panels": panels, "ordered": ordered,
                "location": loc, "outpatient": outpatient, "urgency_base": urg, "hi": hi,
                "minutes_to_spin": mins, "edta_signs": edta, "in_vivo": in_vivo, "k_first_pass": k_val,
                "ca_first_pass": ca_val, "prev_k": prev_k,
                "lipaemic_index": rng.randint(5, 60), "icteric_index": rng.randint(1, 8),
                "difficult_draw": chance(rng, 0.3), "clinician_pressure": reg == "conflicting" and chance(rng, 0.5),
                "patient": person(rng), "mrn": _mrn(rng), "accession": ident(rng, "C", 8, ""),
                "collected": rand_ts(rng, hours=(8, 16) if outpatient else (0, 23)), "scientist": person(rng),
                "requester": "Dr " + person(rng),
                "rule_id": f"PA-{rng.randint(3, 9)}.{rng.randint(1, 6)}",
                "tube": pick(rng, ["lithium heparin (green)", "gel serum (gold)", "gel serum (gold)"])}

    # ------------------------------------------------------------------ the rule as it applies
    def _status(self, f):
        """(beyond-limit analytes, comment-band analytes, stability failures)."""
        beyond, band, late = [], [], []
        for a in f["ordered"]:
            lim, up, h, _ = ANALYTES[a]
            if f["hi"] > lim:
                if up is not None and f["hi"] <= up:
                    band.append(a)
                else:
                    beyond.append(a)
            if f["minutes_to_spin"] > h * 60:
                late.append(a)
        if f["in_vivo"]:
            band, beyond = band + beyond, []        # real haemolysis is reported with a comment
        return beyond, band, late

    def _rules(self, f, rng):
        rows = []
        for a in f["ordered"]:
            lim, up, h, _ = ANALYTES[a]
            hi_txt = f"HI <= {lim} report" + (f"; {lim + 1}-{up} report with comment; > {up} withhold"
                                                if up else f"; > {lim} withhold")
            rows.append((a, hi_txt, f"{h} h"))
        extra = [f"Suspected EDTA carry-over (potassium above 7.0 with calcium below 1.8 mmol/L): reject the whole "
                 f"specimen.",
                 "Haemolysis documented as occurring in the patient (in vivo) is reported with a comment, not "
                 "withheld.",
                 "Withheld analytes are cancelled; request a new specimen if the withheld analyte is the one the "
                 "clinical question depends on, or if nothing ordered can be reported."]
        return rows, extra

    def _note(self, f, rng):
        s = [f"Request from {f['location']}: {f['question']}."]
        if f["difficult_draw"]:
            s.append(pick(rng, ["Phlebotomy note: difficult venous access, third attempt.",
                                "Collected by butterfly after two failed attempts."]))
        if f["in_vivo"]:
            s.append(f"Clinical details mention {f['in_vivo']}.")
        if f["clinician_pressure"]:
            s.append(pick(rng, ["Ward phoned asking for the potassium to be released as is; they say it is real.",
                                "Registrar asked us to report everything today without delay."]))
        if f["outpatient"] and f["minutes_to_spin"] > 120:
            s.append("Arrived by courier from the community; collection time from the request form.")
        return noisy(join_sentences(s), rng, 0.12)

    def render(self, f, rng, surface):
        rows, extra = self._rules(f, rng)
        fp = []
        if f["k_first_pass"] is not None:
            fp.append(f"potassium {f['k_first_pass']} mmol/L" + (f" (previous {f['prev_k']})" if f["prev_k"] else ""))
        if f["ca_first_pass"] is not None:
            fp.append(f"adjusted calcium {f['ca_first_pass']} mmol/L"
                      + ("" if "calcium" in f["ordered"] else " (added by the lab as an EDTA check)"))
        note = self._note(f, rng)
        received = f["collected"] + timedelta(minutes=f["minutes_to_spin"] - rng.randint(5, 15))
        spun = f["collected"] + timedelta(minutes=f["minutes_to_spin"])
        if surface == "json":
            return dumps(prune({
                "specimen": {"accession": f["accession"], "tube": f["tube"], "collected": fmt_ts(f["collected"], style="iso"),
                             "centrifuged": fmt_ts(spun, style="iso"), "location": f["location"],
                             "tests": ", ".join(f["panels"])},
                "indices": {"haemolysis_index": f["hi"], "lipaemic_index": f["lipaemic_index"],
                            "icteric_index": f["icteric_index"]},
                "first_pass_results_held": "; ".join(fp) or None,
                "patient": {"name": f["patient"], "mrn": f["mrn"]},
                "notes": note,
                "rule": {"id": f"{f['rule_id']} (local)",
                         "limits": [f"{a}: {t}; unspun stability {h}" for a, t, h in rows], "also": extra},
            }))
        if surface == "lis_receipt":
            head = kv_lines([("Accession", f["accession"]), ("Patient", f"{f['patient']} ({f['mrn']})"),
                             ("Source", f["location"]), ("Tests", ", ".join(f["panels"])), ("Tube", f["tube"]),
                             ("Collected", fmt_ts(f["collected"], rng)), ("Received", fmt_ts(received, rng)),
                             ("Centrifuged", fmt_ts(spun, rng)),
                             ("Indices H/L/I", f"{f['hi']} / {f['lipaemic_index']} / {f['icteric_index']}"),
                             ("Held results", "; ".join(fp) or None)], rng)
            tab = table(["analyte", "haemolysis rule", "unspun"], [list(r) for r in rows], rng)
            return (f"LIS SPECIMEN RECEIPT -- HOLD FOR REVIEW\n{head}\n\nComments: {note}\n\n"
                    f"Rule {f['rule_id']} (local)\n{tab}\n" + "\n".join(f"* {x}" for x in extra))
        lines = [f"Bench note {fmt_ts(spun, rng)} -- {f['accession']} ({f['patient']}, {f['mrn']})",
                 f"{note.rstrip('.')}. Tube {f['tube']}, collected {fmt_ts(f['collected'], rng)}, spun after "
                 f"{_hm(f['minutes_to_spin'])}. HI {f['hi']}, LI {f['lipaemic_index']}, II {f['icteric_index']}."
                 + (f" On hold: {'; '.join(fp)}." if fp else ""), "",
                 f"{f['rule_id']} (local) for what was ordered:"]
        lines += [f"  {a}: {t} (unspun {h})" for a, t, h in rows]
        lines += [f"  {x}" for x in extra]
        lines += ["", f"-- {f['scientist']}"]
        return "\n".join(lines)

    def questions(self):
        return {
            "specimen_decision": choice("What should the laboratory do with this specimen?", {
                "accept_all": "Run every ordered test and report the results without qualification.",
                "accept_with_comment": "Run every ordered test and attach an interference comment to the "
                                       "affected results.",
                "reject_recollect": "Reject the specimen and ask the requester for a new one.",
                "run_unaffected_only": "Report only the tests the rule still allows, cancel the others, and do "
                                       "not ask for a new specimen.",
            }),
            "interference_source": choice("What mainly limits the usefulness of this specimen?", {
                "delayed_separation": "The blood sat too long before it was centrifuged.",
                "edta_contamination": "Anticoagulant from another tube appears to have got into it.",
                "in_vitro_haemolysis": "Red cells broke during or after collection.",
                "in_vivo_haemolysis": "The haemolysis reflects the patient's own condition.",
                "no_significant_interference": "Nothing about it falls outside the lab's limits.",
            }),
            "interference_impact": score("How badly could the problems found distort the results that were ordered?", [
                "Not at all: everything ordered is within its limits.",
                "Slightly: some results need an interference comment.",
                "Substantially: some ordered results cannot be reported.",
                "Severely: the result the request hinges on cannot be trusted.",
            ]),
            "result_need_timing": score("How soon does the requester need these results to act on them?", [
                "Routine: the next scheduled review.",
                "Within days, before a planned appointment or procedure.",
                "Today, for an inpatient management decision.",
                "Within the hour, for an emergency decision.",
            ]),
            "stability_window_met": noul(
                "The specimen was centrifuged within the unspun stability time of every analyte ordered.",
                yes="The time from collection to centrifugation is within the shortest stability time listed for "
                    "the ordered analytes.",
                no="At least one ordered analyte's unspun stability time was exceeded."),
        }

    def answer_key(self, f):
        """Rule-of-thumb answer key for the spread check only; never used as a label."""
        beyond, band, late = self._status(f)
        hard = sorted(set(beyond) | set(late))
        key = f["key_analyte"]
        if f["edta_signs"]:
            dec = "reject_recollect"
        elif not hard and not band:
            dec = "accept_all"
        elif not hard:
            dec = "accept_with_comment"
        elif key in hard or len(hard) == len(f["ordered"]):
            dec = "reject_recollect"
        else:
            dec = "run_unaffected_only"
        if f["edta_signs"]:
            src = "edta_contamination"
        elif beyond:
            src = "in_vitro_haemolysis"
        elif late:
            src = "delayed_separation"
        elif band:
            src = "in_vivo_haemolysis" if f["in_vivo"] else "in_vitro_haemolysis"
        else:
            src = "no_significant_interference"
        if f["edta_signs"] or key in hard:
            imp = 3
        elif hard:
            imp = 2
        elif band:
            imp = 1
        else:
            imp = 0
        urg = f["urgency_base"]
        if f["outpatient"]:
            urg = min(urg, 1)
        return {"specimen_decision": dec, "interference_source": src, "interference_impact": imp,
                "result_need_timing": urg, "stability_window_met": "no" if late else "yes"}


# =========================================================================== schema 2
class TransfusionSampleLabelling(Schema):
    name = "transfusion_sample_labelling"
    description = ("A group-and-screen or crossmatch sample received by the blood bank, checked against the "
                   "zero-tolerance labelling policy and the patient's transfusion history.")
    surfaces = {"json": 0.5, "bank_receipt_log": 0.25, "bench_phone_log": 0.25}

    # discrepancy -> (identity-level?, mandatory field?)
    DISCREPANCY = {"none": (False, False), "middle_name_missing": (False, False), "time_missing": (False, False),
                   "name_spelling": (False, True), "dob_digit": (False, True), "signature_missing": (False, True),
                   "date_missing": (False, True), "addressograph": (False, True), "mrn_other": (True, True),
                   "wbit_history": (True, False)}
    CONTEXT = {
        "mhp": "major haemorrhage protocol activated; patient actively bleeding",
        "theatre_soon": "emergency laparotomy listed for the next available theatre",
        "surgery_tomorrow": "elective hip replacement booked for tomorrow morning",
        "transfuse_today": "symptomatic anaemia; two units planned for this afternoon",
        "antenatal": "routine antenatal booking bloods",
        "pre_assessment": "pre-assessment clinic for surgery in two weeks",
        "oncology_day": "day-unit transfusion booked for tomorrow",
    }

    CONTEXT_WARDS = {
        "mhp": ["Emergency Department", "Main Theatres", "Delivery Suite", "Intensive Care Unit"],
        "theatre_soon": ["Emergency Department", "Ward 12 (general surgery)", "Surgical Assessment Unit"],
        "surgery_tomorrow": ["Ward 6 (orthopaedics)", "Elective Orthopaedic Unit"],
        "transfuse_today": ["Ward 3C (haematology)", "Ward 10 (gastroenterology)", "Ward 2 (care of older people)",
                            "Acute Medical Unit"],
        "antenatal": ["Antenatal Clinic", "Community Midwifery Hub"],
        "pre_assessment": ["Pre-assessment Clinic"],
        "oncology_day": ["Oncology Day Unit", "Haematology Day Unit"],
    }
    CALLS_LABEL = ["the patient is definitely who we say they are", "can you not just correct the label here",
                   "we need blood now, please do not make us redo it", "it is a typo, the wristband is right"]
    CALLS_CLEAN = ["please can this be prioritised", "theatre are asking when the crossmatch will be ready",
                   "just checking the sample arrived"]

    def sample_factors(self, rng):
        reg = regime(rng)
        first, last = person(rng).split(" ", 1)
        middle = pick(rng, [None, None, person(rng).split()[0]])
        dob = _dob(rng)
        mrn = _mrn(rng)
        if reg == "clear":
            disc = pick(rng, ["none", "none", "mrn_other", "mrn_other", "addressograph"])
        elif reg == "borderline":
            disc = pick_w(rng, {"middle_name_missing": 2, "time_missing": 2, "name_spelling": 3, "dob_digit": 2.5,
                                "signature_missing": 1.5, "date_missing": 1, "none": 1.5})
        else:
            disc = pick_w(rng, {"wbit_history": 3, "name_spelling": 1.5, "dob_digit": 1, "none": 2,
                                "addressograph": 1, "signature_missing": 1})
        if disc == "middle_name_missing" and not middle:
            middle = person(rng).split()[0]
        history = pick_w(rng, {"none": 4, "on_file": 5})
        if disc == "wbit_history":
            history = "on_file"
        context = pick_w(rng, {"mhp": 1.4, "theatre_soon": 1.2, "surgery_tomorrow": 2, "transfuse_today": 2,
                               "antenatal": 1.5, "pre_assessment": 1.5, "oncology_day": 1.2})
        if reg == "conflicting" and chance(rng, 0.4):
            context = pick(rng, ["mhp", "theatre_soon"])
        label = "addressograph" if disc == "addressograph" else pick(rng, ["handwritten", "bedside_printed"])
        groups = ["O RhD positive", "A RhD positive", "B RhD positive", "O RhD negative", "A RhD negative",
                  "AB RhD positive", "B RhD negative"]
        hist_group = pick(rng, groups) if history == "on_file" else None
        today = hist_group if disc != "wbit_history" else pick(rng, [g for g in groups if g != hist_group])
        if history == "none":
            today = pick(rng, groups)
        similar = chance(rng, 0.25)
        # tube fields as written
        t_first, t_last, t_dob, t_mrn = first, last, dob, mrn
        if disc == "name_spelling":
            t_last = self._misspell(rng, last)
        elif disc == "dob_digit":
            t_dob = dob.replace(day=min(28, dob.day % 28 + 1)) if chance(rng, 0.5) else dob.replace(
                year=dob.year + pick(rng, [-10, 10, 1, -1]))
        elif disc == "mrn_other":
            t_mrn = _mrn(rng)
        collected = rand_ts(rng, hours=(8, 17) if context in ("antenatal", "pre_assessment", "oncology_day")
                            else (0, 23))
        return {"regime": reg, "discrepancy": disc, "history": history, "hist_group": hist_group,
                "today_group": today, "context": context, "label_type": label, "similar_name_on_ward": similar,
                "first": first, "last": last, "middle": middle, "dob": dob, "mrn": mrn,
                "tube_first": t_first, "tube_last": t_last, "tube_dob": t_dob, "tube_mrn": t_mrn,
                "tube_middle": None if disc == "middle_name_missing" else middle,
                "collector": person(rng), "ward": pick(rng, self.CONTEXT_WARDS[context]), "collected": collected,
                "received": collected + timedelta(minutes=rng.randint(12, 140)),
                "hb": round(rng.uniform(58, 84) if context in ("transfuse_today", "mhp", "oncology_day")
                            else rng.uniform(95, 140)),
                "sample_id": ident(rng, "BB", 7, ""), "scientist": person(rng),
                "caller": "Dr " + person(rng), "policy_id": f"BT-{rng.randint(2, 14)}",
                "caller_says": pick(rng, [None, None] + (self.CALLS_LABEL if self.DISCREPANCY[disc][1] or
                                                         self.DISCREPANCY[disc][0] else self.CALLS_CLEAN))}

    @staticmethod
    def _misspell(rng, s):
        if len(s) < 4:
            return s + "e"
        i = rng.randint(1, len(s) - 3)
        return s[:i] + s[i + 1] + s[i] + s[i + 2:] if s[i] != s[i + 1] else s[:i] + s[i + 1:]

    def _tube(self, f, rng):
        name = " ".join(x for x in [f["tube_first"], f["tube_middle"], f["tube_last"]] if x)
        return {"name": name, "dob": f["tube_dob"].isoformat(), "mrn": f["tube_mrn"],
                "collected_date": None if f["discrepancy"] == "date_missing" else f["collected"].date().isoformat(),
                "collected_time": None if f["discrepancy"] in ("time_missing", "date_missing")
                else f["collected"].strftime("%H:%M"),
                "collector_signature": None if f["discrepancy"] == "signature_missing" else f["collector"],
                "label": f["label_type"].replace("_", "-")}

    POLICY = ["Mandatory on the tube, matching the request exactly: first name, surname, date of birth, record "
              "number, date of collection and the collector's signature. Middle names and the time of collection "
              "are recommended, not mandatory.",
              "Labels must be handwritten or printed at the bedside; pre-printed addressograph labels are not "
              "accepted. Tubes are never amended or relabelled in the laboratory.",
              "A patient with no historical group needs a second, independently collected sample before "
              "group-specific red cells are issued.",
              "A group that differs from the historical record is treated as a wrong-blood-in-tube event.",
              "During a major haemorrhage protocol, emergency group O units are issued while any rejected sample is "
              "recollected."]

    def render(self, f, rng, surface):
        tube = self._tube(f, rng)
        req_name = " ".join(x for x in [f["first"], f["middle"], f["last"]] if x)
        hist = (f"historical group {f['hist_group']} on file" if f["history"] == "on_file"
                else "no historical group on file")
        ctx = self.CONTEXT[f["context"]]
        sim = "another patient with a similar name is on the same ward" if f["similar_name_on_ward"] else None
        if surface == "json":
            return dumps(prune({
                "sample": {"id": f["sample_id"], "received": fmt_ts(f["received"], style="iso"), "ward": f["ward"],
                           "tube_label": tube, "todays_group": f["today_group"]},
                "request_form": {"name": req_name, "dob": f["dob"].isoformat(), "mrn": f["mrn"],
                                 "clinical_details": ctx, "hb_g_per_l": f["hb"]},
                "history": {"blood_bank_record": hist, "ward_alert": sim},
                "call_from_ward": f["caller_says"],
                "policy": {"id": f"{f['policy_id']} (local)", "rules": self.POLICY},
            }))
        if surface == "bank_receipt_log":
            rows = [["first name", f["first"], f["tube_first"]], ["middle name", show(f["middle"], rng),
                                                                     show(tube["name"].split()[1] if f["tube_middle"] else None, rng)],
                    ["surname", f["last"], f["tube_last"]],
                    ["date of birth", fmt_date(f["dob"], style="eu"), fmt_date(f["tube_dob"], style="eu")],
                    ["record no.", f["mrn"], f["tube_mrn"]],
                    ["collected", "-", f"{show(tube['collected_date'], rng)} {tube['collected_time'] or ''}".strip()],
                    ["signature", "-", show(tube["collector_signature"], rng)], ["label type", "-", tube["label"]]]
            return (f"BLOOD BANK RECEIPT {f['sample_id']} ({f['ward']}, received {fmt_ts(f['received'], rng)})\n"
                    + table(["field", "request", "tube"], rows, rng)
                    + f"\n\nToday's group: {f['today_group']}; {hist}.\nClinical: {ctx}; Hb {f['hb']} g/L."
                    + (f"\nNote: {sim}." if sim else "")
                    + (f"\nWard call: \"{f['caller_says']}\"" if f["caller_says"] else "")
                    + f"\n\nPolicy {f['policy_id']} (local):\n" + "\n".join(f"- {x}" for x in self.POLICY))
        issues = []
        if tube["name"] != req_name:
            issues.append(f"tube name reads '{tube['name']}', request says '{req_name}'")
        if f["tube_dob"] != f["dob"]:
            issues.append(f"DOB on tube {fmt_date(f['tube_dob'], style='eu')} vs request {fmt_date(f['dob'], style='eu')}")
        if f["tube_mrn"] != f["mrn"]:
            issues.append(f"record number on tube {f['tube_mrn']} vs request {f['mrn']}")
        if tube["collector_signature"] is None:
            issues.append("no collector signature")
        if tube["collected_date"] is None:
            issues.append("no collection date")
        elif tube["collected_time"] is None:
            issues.append("no collection time")
        if f["label_type"] == "addressograph":
            issues.append("addressograph label")
        lines = [f"{fmt_ts(f['received'], rng)}  {f['sample_id']} received from {f['ward']} "
                 f"(collector {f['collector']}). {ctx[0].upper() + ctx[1:]}; Hb {f['hb']}.",
                 f"Label check: " + ("; ".join(issues) if issues else "all fields agree with the request") + ".",
                 f"Grouped {f['today_group']}; {hist}."]
        if sim:
            lines.append(f"Ward alert: {sim}.")
        if f["caller_says"]:
            lines.append(f"{fmt_ts(f['received'] + timedelta(minutes=rng.randint(4, 30)), rng)}  {f['caller']} "
                         f"phoned: {noisy(f['caller_says'], rng, 0.1).rstrip('.?')}.")
        lines.append(f"-- {f['scientist']}")
        return "\n".join(lines) + f"\n\n{f['policy_id']} (local): " + " ".join(self.POLICY)

    def questions(self):
        return {
            "bank_sample_decision": choice("What should the blood bank do with this sample?", {
                "accept_for_testing": "Accept it and proceed to group, screen and issue as normal.",
                "accept_require_confirmatory_sample": "Accept it for testing, but require a second independent "
                                                      "sample before group-specific red cells are issued.",
                "issue_emergency_units_and_recollect": "Issue emergency group O units now and have a new sample "
                                                       "taken.",
                "reject_and_recollect": "Reject it and ask the ward to take a new, correctly labelled sample.",
            }),
            "labelling_error_gravity": score("How serious is the labelling problem with this sample?", [
                "None: the tube and the request agree in every field.",
                "Minor: only a recommended, non-mandatory field is affected.",
                "Major: a mandatory field is wrong, missing or on a non-permitted label.",
                "Critical: the tube may belong to a different patient.",
            ]),
            "wrong_blood_in_tube_risk": score("How likely is it that the blood in this tube is not the named patient's?", [
                "Remote.",
                "Low, but a slip in the process cannot be ruled out.",
                "Real: the way the tube was labelled could hide a mix-up.",
                "High: the evidence points to another patient's blood.",
            ]),
            "identifiers_fully_match": noul(
                "The patient identifiers on the tube agree exactly with those on the request.",
                yes="First name, surname, date of birth and record number are identical on tube and request.",
                no="At least one of those identifiers differs between tube and request."),
            "clinical_delay_harmful": noul(
                "Waiting for a new sample before compatible blood is available would put this patient at risk."),
        }

    def answer_key(self, f):
        """Rule-of-thumb answer key for the spread check only; never used as a label."""
        identity, mandatory = self.DISCREPANCY[f["discrepancy"]]
        bad = identity or mandatory
        if bad and f["context"] == "mhp":
            dec = "issue_emergency_units_and_recollect"
        elif bad:
            dec = "reject_and_recollect"
        elif f["history"] == "none":
            dec = "accept_require_confirmatory_sample"
        else:
            dec = "accept_for_testing"
        grav = 3 if identity else 2 if mandatory else 1 if f["discrepancy"] != "none" else 0
        base = {"none": 0, "middle_name_missing": 0, "time_missing": 0, "name_spelling": 1, "dob_digit": 1,
                "signature_missing": 1, "date_missing": 1, "addressograph": 2, "mrn_other": 3, "wbit_history": 3}
        risk = min(3, base[f["discrepancy"]] + (1 if f["similar_name_on_ward"] and base[f["discrepancy"]] < 3 else 0))
        match = f["discrepancy"] not in ("name_spelling", "dob_digit", "mrn_other")
        harmful = f["context"] in ("mhp", "theatre_soon") or (f["context"] == "transfuse_today" and f["hb"] < 70)
        return {"bank_sample_decision": dec, "labelling_error_gravity": grav, "wrong_blood_in_tube_risk": risk,
                "identifiers_fully_match": "yes" if match else "no",
                "clinical_delay_harmful": "yes" if harmful else "no"}


# =========================================================================== schema 3
# specimen -> (time limit in hours at room temperature, at 2-8 C (None: must not be refrigerated),
#              replaceability 0-3, the tests that do not depend on viable organisms)
MICRO = {
    "urine, plain container": (2, 24, 0, ["urinary antigen tests"]),
    "urine, boric acid container": (48, 48, 0, []),
    "sputum": (4, 24, 1, ["mycobacterial culture", "respiratory PCR panel"]),
    "wound swab": (48, 48, 0, []),
    "faeces": (4, 72, 0, ["ova, cysts and parasites"]),
    "CSF": (2, None, 3, ["bacterial PCR"]),
    "tissue biopsy": (4, 24, 3, []),
    "blood culture set": (4, None, 2, []),
    "joint aspirate": (6, 24, 2, ["crystal microscopy"]),
    "bronchoalveolar lavage": (4, 24, 2, ["fungal PCR", "respiratory PCR panel"]),
}
# specimen -> (quality defect, how it reads, how badly it hurts the culture 1-3)
DEFECTS = {
    "urine, plain container": [],
    "urine, boric acid container": [("underfilled_boric", "filled to well below the line, so the preservative is "
                                                          "over-concentrated", 3)],
    "sputum": [("squamous", "microscopy: >25 squamous cells and <10 pus cells per low-power field", 2)],
    "wound swab": [("dry_swab", "dry swab with no transport medium", 3)],
    "faeces": [("formed", "formed stool (Bristol type 3) sent for C. difficile testing", 2)],
    "CSF": [],
    "tissue biopsy": [("formalin", "placed in formalin rather than sterile saline", 3)],
    "blood culture set": [("underfilled_bc", "aerobic bottle about 3 mL, anaerobic about 4 mL (target 8-10 mL)", 1)],
    "joint aspirate": [("clotted", "partly clotted, no anticoagulant tube", 1)],
    "bronchoalveolar lavage": [],
}


class MicrobiologySpecimenAcceptance(Schema):
    CLINICAL = {
        "urine, plain container": ["suspected urinary infection", "dysuria and frequency", "fever, catheter in situ",
                                   "recurrent urinary infections"],
        "urine, boric acid container": ["suspected urinary infection", "antenatal screening", "loin pain and fever"],
        "sputum": ["productive cough, consolidation on X-ray", "bronchiectasis exacerbation",
                   "weight loss and night sweats", "ventilator-associated pneumonia query"],
        "wound swab": ["post-operative wound discharge", "infected leg ulcer", "cellulitis around a surgical drain"],
        "faeces": ["diarrhoea after antibiotics", "bloody diarrhoea after travel", "outbreak screening on the ward"],
        "CSF": ["headache, neck stiffness, photophobia", "fever and confusion", "shunt malfunction query"],
        "tissue biopsy": ["possible prosthetic joint infection", "diabetic foot osteomyelitis query",
                          "non-healing sternal wound"],
        "blood culture set": ["fever and rigors", "suspected line infection", "neutropenic fever"],
        "joint aspirate": ["suspected septic arthritis", "hot swollen knee", "painful prosthetic hip"],
        "bronchoalveolar lavage": ["non-resolving pneumonia", "immunocompromised with new infiltrates",
                                   "ventilator-associated pneumonia query"],
    }
    name = "microbiology_specimen_acceptance"
    description = ("A microbiology specimen checked at reception against the lab's transport-time, container and "
                   "quality rules: process, qualify, part-process, hold or reject.")
    surfaces = {"json": 0.5, "reception_log": 0.25, "ward_call_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        spec = pick_w(rng, {"urine, plain container": 3, "urine, boric acid container": 1.5, "sputum": 2,
                            "wound swab": 2, "faeces": 2, "CSF": 1, "tissue biopsy": 1, "blood culture set": 1.2,
                            "joint aspirate": 1, "bronchoalveolar lavage": 1})
        rt_h, cold_h, repl, other = MICRO[spec]
        stored = pick_w(rng, {"room temperature": 5, "refrigerated": 3 if cold_h else 0})
        limit_h = rt_h if stored == "room temperature" else cold_h
        problem = pick_w(rng, {"delay": 4, "defect": 3 if DEFECTS[spec] else 0, "label": 1.3, "leak": 0.8,
                               "none": 2})
        hours = round(rng.uniform(0.3, limit_h * 0.7), 1)
        defect = None
        label_issue = None
        leak = False
        if problem == "delay":
            hours = (near(rng, limit_h, max(0.2, limit_h * 0.12), lo=0.2) if reg == "borderline"
                     else round(limit_h * rng.uniform(1.2, 3.0), 1))
        elif problem == "defect":
            defect = pick(rng, DEFECTS[spec])[0]
        elif problem == "label":
            label_issue = pick(rng, ["unlabelled container (request form only)",
                                     "container labelled with a different patient's record number",
                                     "surname on container differs from the request"])
        elif problem == "leak":
            leak = True
        if reg == "clear" and problem == "none":
            hours = round(rng.uniform(0.3, limit_h * 0.5), 1)
        # antibiotics relative to collection: None (not started), negative = started before collection (hours)
        abx = pick_w(rng, {None: 5, "after": 2.5, "before": 1.5})
        if reg == "conflicting" and chance(rng, 0.5):
            abx = "after"
        abx_hours = rng.randint(1, 20) if abx else None
        extra_tests = [t for t in other if chance(rng, 0.45)]
        if reg == "conflicting" and other and not extra_tests and chance(rng, 0.6):
            extra_tests = [pick(rng, other)]
        clinical = pick(rng, self.CLINICAL[spec])
        hospital_only = spec in ("CSF", "tissue biopsy", "blood culture set", "joint aspirate", "bronchoalveolar lavage")
        location = pick(rng, WARDS if hospital_only or chance(rng, 0.5) else CLINICS[:8])
        collected = rand_ts(rng, hours=(8, 17) if location in CLINICS else (0, 23))
        discharged = chance(rng, 0.12)
        return {"regime": reg, "specimen": spec, "stored": stored, "limit_hours": limit_h, "hours_old": hours,
                "defect": defect, "label_issue": label_issue, "leaking": leak, "antibiotics": abx,
                "antibiotic_hours": abx_hours, "extra_tests": extra_tests, "patient_discharged": discharged,
                "replaceability_base": repl, "collected": collected,
                "received": collected + timedelta(minutes=int(hours * 60)),
                "location": location, "patient": person(rng), "mrn": _mrn(rng),
                "lab_no": ident(rng, "MB", 7, ""), "clinical": clinical,
                "rule_id": f"MB-{rng.randint(1, 12)}", "receptionist": person(rng),
                "refrigerate_csf": spec == "CSF" and stored == "refrigerated"}

    def _rule_text(self, f):
        rt, cold, _, other = MICRO[f["specimen"]]
        r = [f"{f['specimen']}: process within {rt} h at room temperature"
             + (f" or {cold} h at 2-8 C" if cold else "; do not refrigerate") + "."]
        d = DEFECTS[f["specimen"]]
        if d:
            r.append(f"Not suitable for routine culture if {d[0][1].split(' (')[0]}.")
        if other:
            o = ", ".join(other)
            r.append(f"{o[0].upper() + o[1:]} can still be done when the culture rule is not met.")
        r += ["Irreplaceable specimens (CSF, tissue) are never discarded: process with a disclaimer, or hold and "
              "call the ward if identity is in doubt or the container makes culture impossible.",
              "Unlabelled, mislabelled or leaking replaceable specimens are rejected and a new specimen requested."]
        return r

    def _abx_text(self, f):
        if not f["antibiotics"]:
            return "no antibiotics started"
        if f["antibiotics"] == "after":
            return f"antibiotics started {f['antibiotic_hours']} h after this specimen was taken"
        return f"specimen taken {f['antibiotic_hours']} h after antibiotics were started"

    def render(self, f, rng, surface):
        rules = self._rule_text(f)
        defect = next((d[1] for d in DEFECTS[f["specimen"]] if d[0] == f["defect"]), None)
        tests = ["culture"] + f["extra_tests"]
        if surface == "json":
            return dumps(prune({
                "specimen": {"lab_number": f["lab_no"], "type": f["specimen"], "tests_requested": tests,
                             "collected": fmt_ts(f["collected"], style="iso"),
                             "received": fmt_ts(f["received"], style="iso"), "stored_in_transit": f["stored"],
                             "condition": defect, "label_check": f["label_issue"] or "matches request",
                             "leaking": f["leaking"] or None},
                "patient": {"name": f["patient"], "mrn": f["mrn"], "location": f["location"],
                            "clinical_details": f["clinical"], "antibiotics": self._abx_text(f),
                            "status": "discharged since collection" if f["patient_discharged"] else None},
                "rules": {"id": f"{f['rule_id']} (local)", "text": rules},
            }))
        if surface == "reception_log":
            head = kv_lines([("Lab no.", f["lab_no"]), ("Specimen", f["specimen"]), ("Tests", ", ".join(tests)),
                             ("From", f["location"]), ("Collected", fmt_ts(f["collected"], rng)),
                             ("Received", fmt_ts(f["received"], rng)), ("Transit", f"{f['stored']}"),
                             ("Age at receipt", _hm(f["hours_old"] * 60)), ("Condition", defect),
                             ("Label", f["label_issue"] or "OK"), ("Leaking", "YES" if f["leaking"] else None),
                             ("Antibiotics", self._abx_text(f)), ("Clinical", f["clinical"])], rng)
            return (f"MICROBIOLOGY RECEPTION LOG\n{head}\n\nRule {f['rule_id']} (local):\n"
                    + "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1)))
        s = [f"{f['lab_no']}: {f['specimen']} for {', '.join(tests)} from {f['location']} ({f['clinical']}).",
             f"Received {_hm(f['hours_old'] * 60)} after collection, {f['stored']} in transit."]
        if defect:
            s.append(f"Specimen {defect}.")
        if f["label_issue"]:
            s.append(f"Label problem: {f['label_issue']}.")
        if f["leaking"]:
            s.append("Container leaking into the bag.")
        s.append(self._abx_text(f).capitalize() + ".")
        if f["patient_discharged"]:
            s.append("Patient has been discharged since.")
        return (f"Reception note for the duty microbiologist ({f['receptionist']}, {fmt_ts(f['received'], rng)})\n"
                + noisy(join_sentences(s), rng, 0.1) + f"\n\n{f['rule_id']} (local): " + " ".join(rules))

    def questions(self):
        return {
            "micro_specimen_decision": choice("How should microbiology handle this specimen?", {
                "hold_and_call_ward": "Keep it safely and phone the ward before doing anything else.",
                "process_priority_tests_only": "Do only the requested tests that the problem does not affect.",
                "process_routinely": "Process it in full as normal.",
                "process_with_disclaimer": "Process it, but qualify the report because the specimen breaks a rule.",
                "reject_request_recollection": "Reject it and ask for a new specimen.",
            }),
            "specimen_replaceability": score("How hard would it be to obtain an equally informative specimen now?", [
                "Easy: a new one can be taken at any time with no loss.",
                "Some effort: another collection is possible but inconvenient.",
                "Difficult: it needs an invasive procedure, or treatment has since changed what it would show.",
                "Impossible: this specimen cannot be repeated.",
            ]),
            "culture_validity_risk": score("How likely is a culture from this specimen to mislead?", [
                "Unlikely: it meets the transport and quality rules.",
                "Somewhat: it is slightly outside a rule.",
                "Likely: overgrowth, contamination or die-off could distort what grows.",
                "Very likely: the container or condition makes a meaningful culture almost impossible.",
            ]),
            "transport_rules_met": noul(
                "The specimen meets the lab's transport-time, temperature and container rules.",
                yes="It arrived within the time limit for how it was stored, in the right container, and in a "
                    "condition the rule accepts.",
                no="It was too old for how it was stored, stored at the wrong temperature, or sent in an "
                   "unacceptable container or condition."),
            "recollection_equivalent": noul(
                "A new specimen taken now would be as informative as this one.",
                yes="The site can be sampled again easily and nothing since collection, such as antibiotics, would "
                    "change the result.",
                no="Repeating it needs an invasive procedure, cannot be done, or antibiotics have started since."),
        }

    def _issue(self, f):
        late = f["hours_old"] > f["limit_hours"] or f["refrigerate_csf"]
        return late, f["defect"] is not None

    def answer_key(self, f):
        """Rule-of-thumb answer key for the spread check only; never used as a label."""
        late, defect = self._issue(f)
        irreplaceable = f["replaceability_base"] == 3
        equivalent = (f["replaceability_base"] <= 1 and f["antibiotics"] != "after" and not f["patient_discharged"])
        if f["label_issue"] or f["leaking"]:
            dec = "hold_and_call_ward" if f["replaceability_base"] >= 2 else "reject_request_recollection"
        elif not late and not defect:
            dec = "process_routinely"
        elif f["defect"] == "formalin":
            dec = "hold_and_call_ward"
        elif irreplaceable or f["defect"] == "underfilled_bc":
            dec = "process_with_disclaimer"
        elif f["extra_tests"]:
            dec = "process_priority_tests_only"
        elif not equivalent:
            dec = "process_with_disclaimer"
        else:
            dec = "reject_request_recollection"
        repl = f["replaceability_base"]
        if f["antibiotics"] == "after" or f["patient_discharged"]:
            repl = min(3, max(repl + 1, 2))
        sev = next((d[2] for d in DEFECTS[f["specimen"]] if d[0] == f["defect"]), 0)
        if f["leaking"]:
            sev = max(sev, 2)
        if late:
            sev = max(sev, 1 if f["hours_old"] <= f["limit_hours"] * 1.5 and not f["refrigerate_csf"] else 2)
        return {"micro_specimen_decision": dec, "specimen_replaceability": repl, "culture_validity_risk": sev,
                "transport_rules_met": "no" if (late or defect) else "yes",
                "recollection_equivalent": "yes" if equivalent else "no"}


# =========================================================================== schema 4
COAG_TESTS = {"PT/INR": 24, "aPTT": 4, "fibrinogen": 8, "D-dimer": 8, "anti-Xa (heparin)": 1}
ANTICOAG = {"none": "no anticoagulant", "vka": "a vitamin K antagonist", "ufh": "an unfractionated heparin infusion",
            "lmwh": "prophylactic low-molecular-weight heparin", "doac": "a direct oral anticoagulant"}


class CoagulationSampleReview(Schema):
    name = "coagulation_sample_review"
    description = ("A citrate coagulation sample checked against the lab's fill, haematocrit, clot, line-draw and "
                   "stability rules before results are released.")
    surfaces = {"json": 0.5, "analyser_hold_queue": 0.25, "haematology_bench_note": 0.25}

    DECISIONS = {
        "pre_op_hours": "surgery this afternoon; anaesthetist wants a clotting screen",
        "vka_dosing": "warfarin-type dosing decision at today's anticoagulation clinic",
        "ufh_titration": "heparin infusion rate due to be adjusted on the next result",
        "bleeding": "active bleeding; deciding on blood product support",
        "dic_query": "possible disseminated intravascular coagulation in sepsis",
        "routine": "routine monitoring, results for next week's review",
        "outpatient_review": "outpatient clinic review in three days",
        "pe_query": "suspected pulmonary embolism, D-dimer to guide imaging",
    }

    DECISION_WARDS = {
        "pre_op_hours": ["Ward 12 (general surgery)", "Ward 6 (orthopaedics)", "Surgical Assessment Unit"],
        "vka_dosing": ["Anticoagulation Clinic"], "ufh_titration": ["Intensive Care Unit", "Ward 7B (cardiology)",
                                                                    "High Dependency Unit", "Ward 4A (renal)"],
        "bleeding": ["Emergency Department", "Intensive Care Unit", "Ward 10 (gastroenterology)"],
        "dic_query": ["Intensive Care Unit", "Acute Medical Unit", "High Dependency Unit"],
        "routine": ["Outpatient phlebotomy", "Ward 2 (care of older people)", "Ward 15 (oncology)"],
        "outpatient_review": ["Outpatient phlebotomy", "Anticoagulation Clinic"],
        "pe_query": ["Emergency Department", "Acute Medical Unit"],
    }
    INPATIENT = ("pre_op_hours", "ufh_titration", "bleeding", "dic_query", "pe_query")

    def sample_factors(self, rng):
        reg = regime(rng)
        decision = pick_w(rng, {"pre_op_hours": 1.5, "ufh_titration": 1.6, "bleeding": 1.3, "dic_query": 1.3,
                                "pe_query": 1.0, "vka_dosing": 1.0, "routine": 0.8, "outpatient_review": 0.8})
        anticoag = {"vka_dosing": "vka", "ufh_titration": "ufh"}.get(decision) or pick_w(
            rng, {"none": 4.5, "vka": 1, "lmwh": 1.5, "doac": 2.2, "ufh": 0.8 if decision in self.INPATIENT else 0})
        tests = {"pre_op_hours": ["PT/INR", "aPTT", "fibrinogen"], "vka_dosing": ["PT/INR"],
                 "ufh_titration": pick(rng, [["aPTT"], ["anti-Xa (heparin)"]]),
                 "bleeding": ["PT/INR", "aPTT", "fibrinogen"], "dic_query": ["PT/INR", "aPTT", "fibrinogen", "D-dimer"],
                 "routine": pick(rng, [["PT/INR"], ["PT/INR", "aPTT"]]), "outpatient_review": ["PT/INR", "aPTT"],
                 "pe_query": ["D-dimer"]}[decision]
        tests = list(tests)
        fill = round(rng.uniform(92, 104), 0)
        hct = round(rng.uniform(0.30, 0.48), 2)
        clot = "none"
        line = (pick_w(rng, {"venepuncture": 6, "arterial line": 1.5, "central line": 1.2, "PICC": 0.8})
                if decision in self.INPATIENT else "venepuncture")
        hours = round(rng.uniform(0.3, 2.5), 1)
        contaminated = False
        problem = pick_w(rng, {"fill": 3.5, "time": 2.5, "line": 8.5 if decision in self.INPATIENT else 0, "hct": 0.6,
                               "clot": 0.6, "none": 2.5})
        if problem == "fill":
            fill = (near(rng, 88, 3, lo=70, hi=100, digits=0) if reg == "borderline"
                    else pick(rng, [round(rng.uniform(72, 84)), round(rng.uniform(85, 89))]))
        elif problem == "time":
            lim = min(COAG_TESTS[t] for t in tests)
            hours = (near(rng, lim, max(0.15, lim * 0.15), lo=0.2) if reg == "borderline"
                     else round(rng.uniform(lim * 1.2, max(lim * 1.3, min(30, lim * 6))), 1))
        elif problem == "line":
            line = pick(rng, ["arterial line", "central line", "PICC"])
            contaminated = chance(rng, 0.85)
        elif problem == "hct":
            hct = near(rng, 0.55, 0.03, digits=2) if reg == "borderline" else round(rng.uniform(0.57, 0.66), 2)
        elif problem == "clot":
            clot = pick(rng, ["small clot", "fibrin strands"])
        if reg != "clear" and problem != "fill" and chance(rng, 0.3):
            fill = near(rng, 89, 4, lo=78, hi=100, digits=0)      # under-filling rides along with other problems
        if reg == "conflicting":
            # a line draw from a patient genuinely on heparin, or a clean-looking tube with one disqualifier
            if chance(rng, 0.4) and decision in self.INPATIENT:
                line, anticoag = pick(rng, ["central line", "arterial line"]), "ufh"
                contaminated = chance(rng, 0.6)
                if "aPTT" not in tests and "anti-Xa (heparin)" not in tests:
                    tests.append("aPTT")
        if contaminated and anticoag == "ufh" and chance(rng, 0.3):
            contaminated = False
        # first-pass results (held)
        res = {}
        for t in tests:
            if t == "PT/INR":
                res[t] = round(rng.uniform(2.0, 3.6), 1) if anticoag == "vka" else round(rng.uniform(0.9, 1.3), 1)
            elif t == "aPTT":
                if contaminated:
                    res[t] = pick(rng, [">180 s (no clot detected)", f"{rng.randint(150, 178)} s"])
                elif anticoag == "ufh":
                    res[t] = f"{rng.randint(55, 105)} s"
                else:
                    res[t] = f"{rng.randint(26, 40)} s"
            elif t == "fibrinogen":
                res[t] = f"{round(rng.uniform(1.0, 4.5), 1)} g/L"
            elif t == "D-dimer":
                res[t] = f"{rng.randint(150, 3200)} ng/mL FEU"
            else:
                res[t] = f"{round(rng.uniform(0.05, 1.6) if not contaminated else rng.uniform(1.8, 3.5), 2)} IU/mL"
        tt = f"{rng.randint(110, 240)} s" if contaminated else (
            f"{rng.randint(15, 21)} s" if chance(rng, 0.5) else None)
        collected = rand_ts(rng, hours=(0, 23) if decision in self.INPATIENT else (8, 16))
        return {"regime": reg, "decision": decision, "anticoagulant": anticoag, "tests": tests, "fill_pct": fill,
                "haematocrit": hct, "clot": clot, "draw_site": line, "hours_old": hours,
                "heparin_contaminated": contaminated, "first_pass": res, "thrombin_time": tt,
                "line_flushed_note": line != "venepuncture" and chance(rng, 0.5),
                "haemolysed_slightly": chance(rng, 0.12), "collected": collected,
                "patient": person(rng), "mrn": _mrn(rng), "ward": pick(rng, self.DECISION_WARDS[decision]),
                "sample_no": ident(rng, "CG", 7, ""),
                "scientist": person(rng), "rule_id": f"HC-{rng.randint(2, 11)}",
                "requester_note": pick(rng, [None, "please run whatever you can", "urgent", "results needed before "
                                             "the list starts", "repeat if abnormal"])}

    RULES = ["Fill: 90-110% of the tube's nominal volume is acceptable; 85-89% may be reported for PT/INR only, with "
             "a comment; below 85% reject.",
             "Haematocrit above 0.55: reject and recollect into a citrate-adjusted tube.",
             "Any clot or fibrin strands: reject.",
             "Stability from collection: PT/INR 24 h; aPTT 4 h; fibrinogen and D-dimer 8 h; anti-Xa for heparin 1 h. "
             "Report only the tests still within their limit and cancel the rest.",
             "Suspected heparin contamination (line draw, aPTT unmeasurable or far above the expected range, thrombin "
             "time very prolonged): reject and request a venepuncture sample.",
             "A patient on a direct oral anticoagulant: report PT and aPTT with the standard drug-interference comment."]

    def render(self, f, rng, surface):
        res = "; ".join(f"{k} {v}" for k, v in f["first_pass"].items())
        ctx = self.DECISIONS[f["decision"]]
        site = f["draw_site"] + (" (line flushed before the draw, per note)" if f["line_flushed_note"] else "")
        if surface == "json":
            return dumps(prune({
                "sample": {"number": f["sample_no"], "tube": "3.2% sodium citrate", "fill_percent": f["fill_pct"],
                           "clot_check": f["clot"], "drawn_from": site,
                           "collected": fmt_ts(f["collected"], style="iso"),
                           "hours_since_collection": f["hours_old"],
                           "slight_haemolysis": f["haemolysed_slightly"] or None},
                "patient": {"name": f["patient"], "mrn": f["mrn"], "ward": f["ward"],
                            "latest_haematocrit": f["haematocrit"], "anticoagulation": ANTICOAG[f["anticoagulant"]],
                            "clinical_context": ctx},
                "held_results": {"first_pass": res, "thrombin_time": f["thrombin_time"]},
                "requester_note": f["requester_note"],
                "rules": {"id": f"{f['rule_id']} (local)", "items": self.RULES},
            }))
        if surface == "analyser_hold_queue":
            rows = [[k, v, f"{COAG_TESTS[k]} h"] for k, v in f["first_pass"].items()]
            if f["thrombin_time"]:
                rows.append(["thrombin time", f["thrombin_time"], "-"])
            head = kv_lines([("Sample", f["sample_no"]), ("Patient", f"{f['patient']} ({f['mrn']}), {f['ward']}"),
                             ("Fill", f"{f['fill_pct']:.0f}%"), ("Clot check", f["clot"]), ("Drawn from", site),
                             ("Age", f"{f['hours_old']} h"), ("Last Hct", f["haematocrit"]),
                             ("Anticoagulant", ANTICOAG[f["anticoagulant"]]), ("Context", ctx),
                             ("Note", f["requester_note"])], rng)
            return (f"ANALYSER HOLD QUEUE\n{head}\n\n" + table(["test", "first pass", "stability"], rows, rng)
                    + f"\n\n{f['rule_id']} (local):\n" + "\n".join(f"- {r}" for r in self.RULES))
        s = [f"{f['sample_no']} from {f['ward']}: {', '.join(f['tests'])}. {ctx[0].upper() + ctx[1:]}.",
             f"Citrate tube about {f['fill_pct']:.0f}% full, {'no clot' if f['clot'] == 'none' else f['clot']} "
             f"on check, drawn from {site}, "
             f"{f['hours_old']} h old at analysis.",
             f"Patient on {ANTICOAG[f['anticoagulant']]}; last haematocrit {f['haematocrit']}.",
             f"First pass: {res}" + (f"; TT {f['thrombin_time']}" if f["thrombin_time"] else "") + "."]
        if f["haemolysed_slightly"]:
            s.append("Plasma slightly pink.")
        if f["requester_note"]:
            s.append(f"Request says: {f['requester_note']}.")
        return (f"Haematology bench note -- {f['scientist']}\n" + noisy(join_sentences(s), rng, 0.1)
                + f"\n\nRules {f['rule_id']} (local): " + " ".join(self.RULES))

    def questions(self):
        return {
            "coag_sample_decision": choice("What should the laboratory do with this coagulation sample?", {
                "reject_and_recollect": "Reject it and ask for a new sample.",
                "run_all_tests": "Release every requested result as measured.",
                "run_stable_tests_only": "Release only the tests the sample is still valid for and cancel the others.",
                "run_with_comment": "Release the results with a comment about a pre-analytical limitation.",
            }),
            "result_distortion_likelihood": score("How likely is it that the held results misrepresent the patient's clotting?", [
                "Very unlikely.",
                "Unlikely: a minor limitation that should not change the numbers much.",
                "Possible: some results may be shifted.",
                "Likely: at least one result is probably wrong.",
                "Almost certain: the sample cannot reflect the patient.",
            ]),
            "decision_timing": score("How quickly does a clinical decision depend on these results?", [
                "Not soon: routine monitoring.",
                "Within days.",
                "Later today.",
                "Within the next hour or so.",
            ]),
            "line_contamination_suspected": noul(
                "The sample is likely contaminated with heparin from the line it was drawn through.",
                yes="It came from a line and the clotting times are far beyond what the patient's treatment explains.",
                no="It was a venepuncture, or the results fit the patient's own anticoagulation."),
            "within_fill_tolerance": noul("The citrate tube is filled within the lab's acceptable range for all the tests requested."),
        }

    def answer_key(self, f):
        """Rule-of-thumb answer key for the spread check only; never used as a label."""
        fill, tests = f["fill_pct"], f["tests"]
        late = [t for t in tests if f["hours_old"] > COAG_TESTS[t]]
        valid = [t for t in tests if t not in late]
        fill_ok = fill >= 90
        fill_pt_only = 85 <= fill < 90
        if (f["clot"] != "none" or f["haematocrit"] > 0.55 or fill < 85 or f["heparin_contaminated"]
                or not valid):
            dec = "reject_and_recollect"
        elif fill_pt_only:
            dec = ("run_with_comment" if tests == ["PT/INR"] else
                   "run_stable_tests_only" if "PT/INR" in valid else "reject_and_recollect")
        elif late:
            dec = "run_stable_tests_only"
        elif f["anticoagulant"] == "doac" and any(t in tests for t in ("PT/INR", "aPTT")):
            dec = "run_with_comment"
        elif f["haemolysed_slightly"]:
            dec = "run_with_comment"
        else:
            dec = "run_all_tests"
        if f["heparin_contaminated"] or f["clot"] == "small clot" or f["haematocrit"] > 0.58 or fill < 80:
            dist = 4
        elif f["clot"] != "none" or f["haematocrit"] > 0.55 or fill < 85 or (late and len(late) == len(tests)):
            dist = 3
        elif late or fill < 90:
            dist = 2
        elif f["anticoagulant"] == "doac" or f["haemolysed_slightly"] or f["draw_site"] != "venepuncture":
            dist = 1
        else:
            dist = 0
        timing = {"pre_op_hours": 3, "bleeding": 3, "ufh_titration": 2, "dic_query": 3, "vka_dosing": 2,
                  "pe_query": 2, "outpatient_review": 1, "routine": 0}[f["decision"]]
        return {"coag_sample_decision": dec, "result_distortion_likelihood": dist, "decision_timing": timing,
                "line_contamination_suspected": "yes" if f["heparin_contaminated"] else "no",
                "within_fill_tolerance": "yes" if fill_ok else "no"}


# =========================================================================== schema 5
# assay -> (required condition, allowance text, max transit days, the allowance in (deg C, minutes) or 'no_thaw')
REFERRED = {
    "ACTH (frozen EDTA plasma)": ("frozen", "no_thaw", 5),
    "renin and aldosterone (frozen plasma)": ("frozen", "no_thaw", 5),
    "vitamin B1 (whole blood, protected from light)": ("refrigerated", (25, 240), 3),
    "factor VIII activity (frozen citrate plasma)": ("frozen", "no_thaw", 4),
    "tacrolimus level (whole blood)": ("refrigerated", (30, 480), 5),
    "anti-NMDA receptor antibodies (CSF)": ("refrigerated", (25, 360), 4),
    "very long chain fatty acids (plasma)": ("frozen", (-10, 120), 7),
    "porphyrin screen (urine, protected from light)": ("refrigerated", (25, 240), 2),
    "homocysteine (separated plasma)": ("refrigerated", (25, 120), 3),
    "insulin and C-peptide (frozen serum)": ("frozen", "no_thaw", 5),
    "parathyroid hormone (EDTA plasma)": ("refrigerated", (25, 180), 2),
    "neuronal antibody panel (serum)": ("refrigerated", (30, 720), 7),
    "ammonia-sensitive amino acid profile (frozen plasma)": ("frozen", (-10, 60), 5),
    "complement C3/C4 (frozen serum)": ("frozen", (-10, 240), 6),
}
IRREPLACEABLE = ["collected before plasma exchange was started", "taken during the acute attack, now resolved",
                 "a post-mortem sample", "a baseline taken before the first dose of treatment",
                 "taken at the time of the event; the patient has since been treated",
                 "drawn before the first transfusion"]


class ReferredSpecimenReceipt(Schema):
    name = "referred_specimen_receipt"
    description = ("A specimen shipped in from an outside clinic for a special assay, with a temperature-logger "
                   "trace and paperwork to check against the referral lab's acceptance table.")
    surfaces = {"json": 0.5, "goods_in_record": 0.25, "referral_desk_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        assay = pick(rng, list(REFERRED))
        cond, allow, max_days = REFERRED[assay]
        days = rng.randint(1, max_days - 1)
        dry_ice_left = cond == "frozen"
        thawed = False
        if cond == "frozen":
            peak = round(rng.uniform(-75, -25), 0)
            above_min = 0
        else:
            peak = round(rng.uniform(4, 8.5), 1)
            above_min = 0
        exc = pick_w(rng, {"none": 3, "within": 2.5, "beyond": 3})
        if exc != "none":
            if cond == "frozen":
                if allow == "no_thaw":
                    if exc == "within":
                        peak, above_min = round(rng.uniform(-30, -15), 0), rng.randint(20, 300)   # still frozen
                    else:
                        peak, above_min = round(rng.uniform(-8, 6), 0), rng.randint(60, 900)
                        thawed = peak > -2 and chance(rng, 0.6)
                else:
                    t, m = allow
                    peak = round(rng.uniform(t - 3, t + 6) if reg == "borderline" else
                                 (rng.uniform(t - 12, t - 1) if exc == "within" else rng.uniform(t + 1, 5)), 0)
                    above_min = (near(rng, m, m * 0.3, lo=5, integer=True) if reg == "borderline"
                                 else rng.randint(int(m * 0.2), int(m * 0.8)) if exc == "within"
                                 else rng.randint(int(m * 1.3), m * 6))
                    if peak <= t:
                        above_min = 0 if exc == "within" and chance(rng, 0.3) else above_min
                dry_ice_left = exc == "within" and chance(rng, 0.5)
            else:
                t, m = allow
                if reg == "borderline":
                    peak = round(near(rng, t, 3), 1)
                    above_min = near(rng, m, m * 0.35, lo=10, integer=True)
                elif exc == "within":
                    peak, above_min = round(rng.uniform(t - 6, t), 1), rng.randint(10, m)
                else:
                    peak, above_min = round(rng.uniform(t + 1, t + 12), 1), rng.randint(int(m * 1.2), m * 5)
        if reg == "borderline" and chance(rng, 0.4):
            days = near(rng, max_days, 1, lo=1, integer=True)
        paperwork = pick_w(rng, {"complete": 4, "clinical_missing": 2, "unsigned": 1.5, "no_collection_date": 1,
                                 "second_id_missing": 2, "id_conflict": 1})
        irreplaceable = pick(rng, IRREPLACEABLE) if chance(rng, 0.35) else None
        if reg == "conflicting":
            if chance(rng, 0.5):
                irreplaceable = irreplaceable or pick(rng, IRREPLACEABLE)
                exc = "beyond" if exc == "none" else exc
                if cond != "frozen" and peak <= REFERRED[assay][1][0]:
                    peak, above_min = round(REFERRED[assay][1][0] + rng.uniform(1, 8), 1), REFERRED[assay][1][1] * 2
                elif cond == "frozen" and above_min == 0:
                    peak, above_min = (round(rng.uniform(-8, 4), 0), rng.randint(90, 600))
                    dry_ice_left = False
            else:
                paperwork = pick(rng, ["second_id_missing", "id_conflict"])
        thr = -10 if allow == "no_thaw" else allow[0]
        if peak <= thr:
            above_min = 0
        sender_issues = pick(rng, [0, 0, 0, 1, 2, 4])
        logger_ok = not (reg == "conflicting" and chance(rng, 0.15))
        collected = rand_ts(rng)
        return {"regime": reg, "assay": assay, "condition": cond, "transit_days": days, "max_days": max_days,
                "peak_c": peak, "minutes_above": above_min, "dry_ice_left": dry_ice_left, "thawed": thawed,
                "paperwork": paperwork, "irreplaceable": irreplaceable, "sender_issues_90d": sender_issues,
                "logger_downloaded": logger_ok, "collected": collected,
                "received": (collected + timedelta(days=days)).replace(hour=rng.randint(8, 17),
                                                                       minute=rng.randrange(60)),
                "clinic": pick(rng, CLINICS), "patient": person(rng), "mrn": _mrn(rng), "dob": _dob(rng),
                "consignment": ident(rng, "RX", 7, ""), "courier_ref": f"{pick(rng, ['TRK', 'CN', 'AWB'])}{rng.randrange(10**8, 10**9)}",
                "desk": person(rng), "table_id": f"RF-{rng.randint(1, 9)}"}

    def _stable(self, f):
        cond, allow, max_days = REFERRED[f["assay"]]
        if f["transit_days"] > max_days:
            return False
        if cond == "frozen" and allow == "no_thaw":
            return f["peak_c"] <= -10 and not f["thawed"]
        t, m = allow
        if f["peak_c"] <= t:
            return True
        return f["minutes_above"] <= m

    def _allow_text(self, f):
        cond, allow, max_days = REFERRED[f["assay"]]
        if allow == "no_thaw":
            a = "must stay frozen (logger at or below -10 C throughout)"
        else:
            a = (f"{cond}; excursions above {allow[0]} C acceptable for up to {allow[1]} min in total")
        return f"{f['assay']}: {a}; maximum {max_days} days from collection to receipt."

    def _paper_text(self, f):
        return {"complete": "request form complete", "clinical_missing": "no clinical details on the request",
                "unsigned": "request form not signed by the requesting clinician",
                "no_collection_date": "collection date missing from tube and form",
                "second_id_missing": "tube carries the name only; no date of birth or record number",
                "id_conflict": "date of birth on the tube differs from the request form"}[f["paperwork"]]

    RULES = ["Identity: tube and form must share the name plus a second identifier. A gap or conflict means "
             "quarantine (up to 5 working days) while the sender confirms in writing; irreplaceable specimens are "
             "always quarantined, never discarded.",
             "Stability: outside the table the specimen is unsuitable. Replaceable specimens are rejected and the "
             "sender asked for a new one; irreplaceable ones may be tested with a qualified report.",
             "Missing clinical details or signature alone do not stop testing; ask the sender to complete them.",
             "Repeated shipping failures from one sender (three or more in 90 days) trigger a packaging review "
             "with that sender."]

    def render(self, f, rng, surface):
        allow = REFERRED[f["assay"]][1]
        thr = -10 if allow == "no_thaw" else allow[0]
        if not f["logger_downloaded"]:
            logger = "logger present but would not download"
        elif f["minutes_above"]:
            logger = f"peak {f['peak_c']} C; {f['minutes_above']} min above {thr} C"
        else:
            logger = f"peak {f['peak_c']} C; never above {thr} C"
        ice = ("dry ice still present" if f["dry_ice_left"] else "no dry ice left") if f["condition"] == "frozen" else \
            "cool packs"
        look = "contents visibly thawed" if f["thawed"] else None
        if surface == "json":
            return dumps(prune({
                "consignment": {"id": f["consignment"], "from": f["clinic"], "courier_ref": f["courier_ref"],
                                "collected": f["collected"].date().isoformat(), "received": fmt_ts(f["received"], style="iso"),
                                "days_in_transit": f["transit_days"], "packaging": ice, "logger": logger,
                                "on_opening": look},
                "specimen": {"assay": f["assay"], "patient": f["patient"], "record": f["mrn"],
                             "paperwork": self._paper_text(f), "note_from_sender": f["irreplaceable"]},
                "sender_history": {"shipping_problems_last_90_days": f["sender_issues_90d"]},
                "acceptance": {"table": f"{f['table_id']} (local)", "assay_rule": self._allow_text(f),
                               "general": self.RULES},
            }))
        if surface == "goods_in_record":
            head = kv_lines([("Consignment", f["consignment"]), ("Sender", f["clinic"]), ("Courier", f["courier_ref"]),
                             ("Received", fmt_ts(f["received"], rng)), ("Collected", fmt_date(f["collected"].date(), rng)),
                             ("Assay", f["assay"]), ("Packaging", ice), ("Logger", logger), ("On opening", look),
                             ("Paperwork", self._paper_text(f)), ("Sender note", f["irreplaceable"]),
                             ("Sender problems (90 d)", f["sender_issues_90d"])], rng)
            return (f"REFERRALS GOODS-IN\n{head}\n\nAcceptance table {f['table_id']} (local)\n"
                    + self._allow_text(f) + "\n" + "\n".join(f"- {r}" for r in self.RULES))
        body = [f"Subject: {f['consignment']} from {f['clinic']} -- {f['assay']}", f"From: {f['desk']} (referrals desk)",
                "To: duty clinical scientist", "",
                noisy(join_sentences([
                    f"Box arrived {fmt_ts(f['received'], rng)}, {f['transit_days']} day(s) after collection, {ice}.",
                    f"Logger: {logger}." if f["logger_downloaded"] else "The logger would not download.",
                    (look.capitalize() + ".") if look else "",
                    f"Paperwork: {self._paper_text(f)}.",
                    (f"The sender says the specimen was {f['irreplaceable']}." if f["irreplaceable"] else ""),
                    (f"This sender has had {f['sender_issues_90d']} shipping problem(s) in the last 90 days."
                     if f["sender_issues_90d"] else "")]), rng), "",
                f"Table {f['table_id']} (local): {self._allow_text(f)} " + " ".join(self.RULES), "", f["desk"].split()[0]]
        return "\n".join(body)

    def questions(self):
        return {
            "receipt_decision": choice("What should the referral laboratory do with this specimen?", {
                "accept_and_test": "Accept it and run the assay as normal.",
                "accept_with_qualified_report": "Run the assay but qualify the report because the specimen was "
                                                "outside its acceptance conditions.",
                "quarantine_and_query_sender": "Hold it untested until the sender resolves a query.",
                "reject_and_notify_sender": "Reject it as unsuitable and tell the sender.",
            }),
            "sender_follow_up": choice("What should the lab ask of the sending clinic?", {
                "confirm_identity_in_writing": "Confirm the patient's identity details in writing.",
                "no_action_needed": "Nothing beyond the normal report.",
                "review_packaging_and_shipping": "Review how the clinic packs and ships specimens.",
                "send_replacement_specimen": "Collect and send a new specimen.",
            }),
            "excursion_severity": score("How serious was the temperature or transit problem in shipping?", [
                "None: conditions held throughout.",
                "Within allowance: a brief excursion the table accepts.",
                "Beyond allowance: outside the table, but not grossly.",
                "Gross: thawed, or far outside the table for a long time, or the record is missing.",
            ]),
            "documentation_gap": score("How serious are the gaps in the paperwork?", [
                "None.",
                "Minor: clinical details missing.",
                "Significant: an unsigned request or a missing collection date.",
                "Identity-level: an identifier missing or in conflict.",
            ]),
            "analyte_stability_preserved": noul(
                "Under the acceptance table, the specimen is still suitable for this assay.",
                yes="The logger trace and transit time stay within the table's limits for the assay.",
                no="The trace or transit time breaks the table's limits, or the specimen thawed when it must not."),
        }

    def answer_key(self, f):
        """Rule-of-thumb answer key for the spread check only; never used as a label."""
        stable = self._stable(f) and f["logger_downloaded"]
        identity = f["paperwork"] in ("second_id_missing", "id_conflict")
        irr = f["irreplaceable"] is not None
        if identity:
            dec = "quarantine_and_query_sender"
        elif not stable:
            dec = "accept_with_qualified_report" if irr else "reject_and_notify_sender"
        else:
            dec = "accept_and_test"
        if identity:
            fu = "confirm_identity_in_writing"
        elif not stable and not irr:
            fu = "send_replacement_specimen"
        elif (not stable) or f["sender_issues_90d"] >= 2 or (f["condition"] == "frozen" and not f["dry_ice_left"]):
            fu = "review_packaging_and_shipping"
        else:
            fu = "no_action_needed"
        cond, allow, max_days = REFERRED[f["assay"]]
        if not f["logger_downloaded"] or f["thawed"] or (not stable and f["minutes_above"] > 600):
            sev = 3
        elif not stable:
            sev = 2
        elif f["minutes_above"] > 0 or (cond == "frozen" and f["peak_c"] > -20):
            sev = 1
        else:
            sev = 0
        doc = {"complete": 0, "clinical_missing": 1, "unsigned": 2, "no_collection_date": 2,
               "second_id_missing": 3, "id_conflict": 3}[f["paperwork"]]
        return {"receipt_decision": dec, "sender_follow_up": fu, "excursion_severity": sev, "documentation_gap": doc,
                "analyte_stability_preserved": "yes" if stable else "no"}


WORKFLOW = Workflow(
    name="clinical_lab_specimen_rejection",
    title="Clinical laboratory specimen acceptance",
    description="Whether a hospital laboratory accepts, rejects or conditionally accepts a specimen under its "
                "written pre-analytical rules.",
    domain="life sciences / clinical laboratory medicine",
    schemas=[ChemistrySpecimenAcceptance(), TransfusionSampleLabelling(), MicrobiologySpecimenAcceptance(),
             CoagulationSampleReview(), ReferredSpecimenReceipt()],
    notes="Fictional laboratory, patients and local rules; limits are rendered into each state and are not real "
          "clinical guidance. No patient-facing conversation, no security alerts, no bills, no AI-agent runs.",
)
