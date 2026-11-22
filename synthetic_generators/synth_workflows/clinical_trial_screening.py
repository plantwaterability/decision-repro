"""Clinical trial screening: operational decisions at a (fictional) trial site or sponsor.

Six schemas: pre-screening a candidate against a protocol's eligibility criteria, assessing an adverse event
for expedited reporting, classifying a protocol deviation, selecting a site, applying a toxicity table's
dose-modification rules, and grading a source-data-verification finding from a monitoring visit.

Everything is invented: investigational products ("ZX-4417"), protocol codes, sponsors, sites and subject ids.
Prior and concomitant treatments are named by drug CLASS only, never by a real product. Each schema draws a
`regime` first and puts the protocol's own limits in the state (a protocol excerpt, a toxicity table, a
deviation plan), so every case is decidable in principle but most sit on a cut-off, carry sources that
disagree, or miss a result. Age appears only in the pre-screen schema, as a protocol range the candidate is
always comfortably inside; no protected attribute decides anything.
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
    show,
    table,
)

# invented investigational-product code prefixes
DRUG_PREFIX = ["ZX", "QRV", "VLK", "HXN", "PLD", "WMT", "YBR", "KSO", "TQE", "DUX"]
ONC_INDICATIONS = [
    "advanced solid tumours after standard therapy", "relapsed or refractory follicular lymphoma",
    "metastatic colorectal cancer, third line", "unresectable hepatocellular carcinoma",
    "metastatic non-small cell lung cancer", "relapsed multiple myeloma", "advanced soft-tissue sarcoma",
    "metastatic pancreatic adenocarcinoma",
]
OTHER_INDICATIONS = [
    "moderate-to-severe plaque psoriasis", "heart failure with preserved ejection fraction",
    "type 2 diabetes with chronic kidney disease", "rheumatoid arthritis after biologic failure",
    "idiopathic pulmonary fibrosis", "ulcerative colitis, moderate to severe",
]
PRIOR_THERAPY = ["cytotoxic chemotherapy", "an anti-PD-1 antibody", "an oral tyrosine kinase inhibitor",
                 "an investigational agent in another study", "an anti-CD20 antibody", "a PARP inhibitor"]
PENDING_ITEMS = ["hepatitis B core antibody", "hepatitis C RNA", "coagulation panel (INR)",
                 "central over-read of the screening ECG", "confirmation that the archival tissue block arrived",
                 "urinalysis for protein"]


def _drug(rng):
    return f"{pick(rng, DRUG_PREFIX)}-{rng.randrange(1000, 10000)}"


def _protocol(rng, drug, phase=None):
    phase = phase or pick(rng, [1, 2, 2, 3])
    return f"{drug}-{phase}{rng.randrange(1, 40):02d}"


def _site_no(rng):
    return rng.randrange(101, 480)


def _subject(rng, site):
    return f"{site}-{rng.randrange(1, 90):03d}"


def _d(n):
    return timedelta(days=n)


def _anc_text(v, unit):
    return f"{int(round(v * 1000)):,}/µL" if unit == "per_ul" else f"{v:.2f} x10^9/L"


def _cr_text(egfr, unit):
    mg = 72.0 / max(egfr, 5)
    return f"{mg * 88.4:.0f} µmol/L" if unit == "umol" else f"{mg:.2f} mg/dL"


def _prev_version(v, rng):
    major, minor = (int(x) for x in v.split("."))
    return f"{major}.{minor - 1}" if minor else f"{major - 1}.{rng.randint(0, 2)}"


def _kv(pairs, rng, skip_none=True):
    """kv_lines in a label-safe house style ("equals" would turn "Root cause (site)" into a mangled key)."""
    return kv_lines(pairs, rng, style=pick(rng, ["colon", "colon", "aligned", "dash"]), skip_none=skip_none)


def _institution(rng):
    """A fictional hospital-type research site (a family practice rarely runs these studies)."""
    for _ in range(6):
        name = org(rng, "clinic")
        if "Family Practice" not in name:
            return name
    return name.replace("Family Practice", "Hospital")


def _ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _fd(d, rng):
    """fmt_date with ONE style per case (a real document does not mix date formats). The style is drawn from the
    case's own rng on first use and kept on it, so output stays deterministic."""
    st = getattr(rng, "_cts_date_style", None)
    if st is None:
        st = pick(rng, ["iso", "iso", "us", "long", "eu"])
        rng._cts_date_style = st
    return fmt_date(d, style=st)


# =========================================================================== schema 1
class EligibilityPrescreen(Schema):
    name = "eligibility_prescreen"
    description = "A referred candidate checked against a protocol's eligibility criteria before the screening visit."
    surfaces = {"json": 0.45, "screening_checklist": 0.3, "coordinator_note": 0.25}

    CRITS = ["ecog", "egfr", "anc", "alt", "washout", "lab_window", "qtcf"]
    CAPACITY = {
        "alert": "alert and oriented at clinic; asked sensible questions about the study",
        "fluctuating": "drowsy after an inpatient procedure this week; nursing notes intermittent confusion, improving",
        "impaired": "investigator noted the candidate could not retain the study information at the last visit",
    }

    def sample_factors(self, rng):
        reg = regime(rng)
        status = {c: "ok" for c in self.CRITS}
        conflict = None
        if reg == "clear":
            if chance(rng, 0.5):
                status[pick(rng, self.CRITS)] = "fail"
        elif reg == "borderline":
            for c in rng.sample(self.CRITS, pick(rng, [1, 1, 2])):
                status[c] = "tight"
        else:
            conflict = pick(rng, ["ecog_disagree", "egfr_two_values", "washout_recall", "pi_optimistic"])
            key = {"ecog_disagree": "ecog", "egfr_two_values": "egfr",
                   "washout_recall": "washout"}.get(conflict) or pick(rng, self.CRITS)
            status[key] = "tight"
            if conflict == "pi_optimistic" and chance(rng, 0.4):
                status[pick(rng, [c for c in self.CRITS if c != key])] = "fail"

        age_lo, age_hi = pick(rng, [(18, 75), (18, 80), (21, 70), (18, 85)])
        ecog_max = pick(rng, [1, 1, 2])
        s = status["ecog"]
        ecog = ((rng.randrange(ecog_max) if chance(rng, 0.8) else ecog_max) if s == "ok"
                else ecog_max if s == "tight" else ecog_max + 1)
        ecog_other = ecog_max + 1 if conflict == "ecog_disagree" else None

        egfr_min = pick(rng, [30, 45, 50, 60])
        s = status["egfr"]
        if s == "ok":
            egfr = int(round(egfr_min * rng.uniform(1.18, 1.9)))
        elif s == "tight":
            egfr = near(rng, egfr_min, egfr_min * 0.05, integer=True)
        else:
            egfr = int(round(egfr_min * rng.uniform(0.62, 0.86)))
        egfr = min(egfr, 115)
        egfr_prev = None
        if conflict == "egfr_two_values":
            egfr_prev = egfr_min - rng.randint(2, 5) if egfr >= egfr_min else egfr_min + rng.randint(1, 4)

        anc_min = pick(rng, [1.0, 1.5, 1.5])
        s = status["anc"]
        if s == "ok":
            anc = round(anc_min * rng.uniform(1.4, 3.4), 2)
        elif s == "tight":
            anc = near(rng, anc_min, anc_min * 0.05, digits=2)
        else:
            anc = round(anc_min * rng.uniform(0.5, 0.8), 2)

        indication = pick(rng, ONC_INDICATIONS)
        liver_mets = chance(rng, 0.2) and not any(w in indication for w in ("lymphoma", "myeloma"))
        alt_x = pick(rng, [2.5, 3.0])
        alt_uln = pick(rng, [33, 40, 45, 50])
        alt_lim = alt_uln * (5.0 if liver_mets else alt_x)
        s = status["alt"]
        if s == "ok":
            alt = int(alt_uln * rng.uniform(0.35, 1.6))
        elif s == "tight":
            alt = near(rng, alt_lim, alt_lim * 0.05, integer=True)
        else:
            alt = int(alt_lim * rng.uniform(1.25, 2.0))

        washout = pick(rng, [14, 21, 28, 28])
        s = status["washout"]
        gap = (washout + rng.randint(8, 60) if s == "ok" else washout + pick(rng, [-1, -1, 0, 1]) if s == "tight"
               else washout - rng.randint(4, 10))
        recall_shift = pick(rng, [-3, -2, 2, 3]) if conflict == "washout_recall" else None

        lab_window = pick(rng, [7, 14, 14, 21])
        s = status["lab_window"]
        lab_age = (rng.randint(1, lab_window - 3) if s == "ok" else lab_window + pick(rng, [0, 1]) if s == "tight"
                   else lab_window + rng.randint(4, 12))

        qtc_max = pick(rng, [450, 470, 480])
        s = status["qtcf"]
        qtc = (rng.randint(392, qtc_max - 18) if s == "ok" else near(rng, qtc_max, 6, integer=True) if s == "tight"
               else qtc_max + rng.randint(10, 35))
        trip = [qtc + rng.randint(-7, 7) for _ in range(2)]
        trip.append(3 * qtc - sum(trip))          # three readings whose mean is exactly qtc
        ecg_done = not (reg != "clear" and s == "ok" and chance(rng, 0.15))

        pending = pick(rng, PENDING_ITEMS) if chance(rng, {"clear": 0.0, "borderline": 0.3,
                                                             "conflicting": 0.45}[reg]) else None
        capacity = (pick_w(rng, {"alert": 9, "fluctuating": 1}) if reg == "clear"
                    else pick_w(rng, {"alert": 6, "fluctuating": 3, "impaired": 1}))
        icf_current = pick(rng, ["2.1", "3.0", "4.0", "4.1", "5.0"])
        icf_outdated = chance(rng, 0.2)
        rand_day = rand_date(rng)
        lab_date = rand_day - _d(lab_age)
        review_day = min(lab_date + _d(rng.randint(1, 3)), rand_day - _d(1))
        pi_comment = ("Looks eligible to me, please book randomisation." if conflict == "pi_optimistic" else
                      pick(rng, [None, None, "Keen to enrol; clinically a good candidate.",
                                 "Please confirm the numbers against the protocol before booking.",
                                 "Happy to proceed if everything is within limits."]))
        site = _site_no(rng)
        drug = _drug(rng)
        return {
            "regime": reg, "criterion_status": status, "conflict": conflict,
            "drug": drug, "protocol": _protocol(rng, drug, pick(rng, [2, 2, 3])), "indication": indication,
            "site": site, "prescreen_id": f"PS-{site}-{rng.randrange(1, 200):03d}",
            "age_lo": age_lo, "age_hi": age_hi, "age": rng.randint(age_lo + 5, age_hi - 5),
            "ecog_max": ecog_max, "ecog": ecog, "ecog_other": ecog_other,
            "egfr_min": egfr_min, "egfr": egfr, "egfr_prev": egfr_prev,
            "cr_unit": pick(rng, ["mgdl", "mgdl", "umol"]),
            "anc_min": anc_min, "anc": anc, "anc_unit": pick(rng, ["si", "si", "per_ul"]),
            "liver_mets": liver_mets, "alt_x": alt_x, "alt_uln": alt_uln, "alt": alt,
            "washout_days": washout, "gap_days": gap, "recall_shift": recall_shift,
            "prior_therapy": pick(rng, PRIOR_THERAPY),
            "lab_window_days": lab_window, "lab_age_days": lab_age,
            "qtc_max": qtc_max, "qtcf": qtc, "qtcf_triplicate": trip, "ecg_done": ecg_done,
            "show_triplicate": chance(rng, 0.5),
            "pending": pending, "capacity": capacity, "lar_permitted": chance(rng, 0.35),
            "icf_current": icf_current, "icf_signed": _prev_version(icf_current, rng) if icf_outdated else icf_current,
            "randomisation_date": rand_day, "lab_date": lab_date, "review_date": review_day,
            "last_dose_date": rand_day - _d(gap),
            "coordinator": person(rng), "investigator": person(rng), "pi_comment": pi_comment,
            "referral": pick(rng, ["outside oncologist", "own clinic", "tumour board", "self-referral"]),
        }

    # ------------------------------------------------------------------ text pieces
    def _criteria(self, f):
        alt = f"ALT <= {f['alt_x']:g} x ULN (<= 5 x ULN with liver involvement by tumour)"
        return {
            "inclusion": [f"Age {f['age_lo']}-{f['age_hi']} years at consent", f"ECOG performance status 0-{f['ecog_max']}",
                          f"eGFR >= {f['egfr_min']} mL/min/1.73m2", f"ANC >= {f['anc_min']:.1f} x10^9/L", alt,
                          f"At least {f['washout_days']} days from last systemic anticancer or investigational "
                          f"therapy to randomisation"],
            "exclusion": [f"QTcF > {f['qtc_max']} ms (mean of triplicate ECG)"],
            "timing": f"Screening labs within {f['lab_window_days']} days before randomisation",
        }

    def _last_dose_text(self, f, rng):
        if f["recall_shift"] is None:
            return _fd(f["last_dose_date"], rng)
        alt_day = f["last_dose_date"] + _d(f["recall_shift"])
        return (f"{_fd(f['last_dose_date'], rng)} per referral letter; candidate recalls "
                f"{_fd(alt_day, rng)}; infusion records requested")

    def _ecg_text(self, f):
        if not f["ecg_done"]:
            return None
        if f["show_triplicate"]:
            return "QTcF " + "/".join(str(x) for x in f["qtcf_triplicate"]) + " ms"
        return f"QTcF {f['qtcf']} ms (mean of 3)"

    def _coord_remarks(self, f, rng):
        s = []
        if f["ecog_other"] is not None:
            s.append(f"Research nurse scored ECOG {f['ecog_other']} at triage; investigator recorded {f['ecog']}.")
        elif f["criterion_status"]["ecog"] == "tight":
            s.append(pick(rng, {1: ["Rests for a couple of hours most afternoons.",
                                    "Has stopped working but manages all self-care."],
                                2: ["Spends a good part of the day resting in a chair.",
                                    "Needs help with shopping; up and about for most of the morning."]}[f["ecog_max"]]))
        if f["egfr_prev"] is not None:
            s.append(f"Referring lab had eGFR {f['egfr_prev']} a week earlier.")
        if not f["ecg_done"]:
            s.append("Screening ECG not done yet.")
        if f["liver_mets"]:
            s.append("Liver lesions on the last CT.")
        return noisy(join_sentences(s), rng, 0.15) if s else None

    def render(self, f, rng, surface):
        crit = self._criteria(f)
        anc = _anc_text(f["anc"], f["anc_unit"])
        cr = _cr_text(f["egfr"], f["cr_unit"])
        remarks = self._coord_remarks(f, rng)
        ecg = self._ecg_text(f)
        consent_line = (f"Pre-screening consent signed on v{f['icf_signed']}; current approved version v{f['icf_current']}. "
                        f"Capacity: {self.CAPACITY[f['capacity']]}. Representative consent "
                        + ("allowed by protocol." if f["lar_permitted"] else "not allowed by protocol."))
        if surface == "json":
            return dumps(prune({
                "candidate": {"prescreen_id": f["prescreen_id"], "age": f["age"], "diagnosis": f["indication"],
                              "liver_involvement": f["liver_mets"], "referral_source": f["referral"],
                              "ecog": prune({"investigator": f["ecog"], "research_nurse": f["ecog_other"]}),
                              "last_systemic_therapy": {"type": f["prior_therapy"],
                                                        "last_dose": (f["last_dose_date"].isoformat()
                                                                      if f["recall_shift"] is None
                                                                      else self._last_dose_text(f, rng))}},
                "protocol_excerpt": dict(crit, code=f["protocol"]),
                "labs": {"collected": f["lab_date"].isoformat(), "egfr_ml_min_1_73m2": f["egfr"], "creatinine": cr,
                         "anc": anc, "alt_u_per_l": f["alt"], "alt_uln_u_per_l": f["alt_uln"],
                         "ecg": ecg or "not done", "pending": f["pending"]},
                "schedule": {"prescreen_review": f["review_date"].isoformat(),
                             "planned_randomisation": f["randomisation_date"].isoformat()},
                "consent": {"prescreen_icf_version_signed": f["icf_signed"], "current_icf_version": f["icf_current"],
                            "capacity_note": self.CAPACITY[f["capacity"]],
                            "representative_consent_allowed": f["lar_permitted"]},
                "notes": prune({"coordinator": remarks, "investigator": f["pi_comment"]}),
            }))
        if surface == "screening_checklist":
            ecog = str(f["ecog"]) + (f" (RN: {f['ecog_other']})" if f["ecog_other"] is not None else "")
            egfr = f"{f['egfr']} ({cr})" + (f"; prior {f['egfr_prev']}" if f["egfr_prev"] is not None else "")
            rows = [["Age", f"{f['age_lo']}-{f['age_hi']}", f["age"]],
                    ["ECOG", f"0-{f['ecog_max']}", ecog],
                    ["eGFR", f">= {f['egfr_min']}", egfr],
                    ["ANC", f">= {f['anc_min']:.1f} x10^9/L", anc],
                    ["ALT", f"<= {f['alt_x']:g} x ULN (5x if liver involved)",
                     f"{f['alt']} U/L, ULN {f['alt_uln']}" + ("; liver involved" if f["liver_mets"] else "")],
                    ["Washout", f">= {f['washout_days']} d", self._last_dose_text(f, rng)],
                    ["Labs date", f"<= {f['lab_window_days']} d pre-rand", _fd(f["lab_date"], rng)],
                    ["QTcF", f"exclude > {f['qtc_max']}", ecg or "ECG not done"]]
            head = _kv([("Protocol", f"{f['protocol']} ({f['indication']})"), ("Candidate", f["prescreen_id"]),
                             ("Planned randomisation", _fd(f["randomisation_date"], rng)),
                             ("Last systemic therapy", f["prior_therapy"]),
                             ("Checked by", f["coordinator"])], rng)
            tail = [f"Pending: {f['pending']}" if f["pending"] else None, f"Consent: {consent_line}",
                    f"Coordinator: {remarks}" if remarks else None,
                    f"PI: {f['pi_comment']}" if f["pi_comment"] else None]
            return ("PRE-SCREENING ELIGIBILITY CHECKLIST\n" + head + "\n\n"
                    + table(["Criterion", "Protocol", "Candidate"], rows, rng) + "\n\n"
                    + "\n".join(x for x in tail if x))
        # coordinator_note
        body = [f"Pre-screened {f['prescreen_id']} for {f['protocol']} ({f['indication']}; referral: "
                f"{f['referral']}). Planned randomisation {_fd(f['randomisation_date'], rng)}.",
                f"Labs from {_fd(f['lab_date'], rng)}: eGFR {f['egfr']} (creatinine {cr}), ANC {anc}, "
                f"ALT {f['alt']} U/L (ULN {f['alt_uln']}).",
                f"ECOG {f['ecog']} per {f['investigator'].split()[-1]}'s last note.",
                f"{ecg}." if ecg else "",
                f"Last dose of {f['prior_therapy']}: {self._last_dose_text(f, rng)}.", remarks or "",
                f"Still waiting on {f['pending']}." if f["pending"] else "",
                f"Consent: {consent_line}"]
        limits = "; ".join(crit["inclusion"][1:] + crit["exclusion"] + [crit["timing"]])
        return "\n".join([
            f"Subject: Pre-screen {f['prescreen_id']} / {f['protocol']} -- eligibility check",
            f"From: {f['coordinator']}", f"To: {f['investigator']}", f"Date: {_fd(f['review_date'], rng)}", "",
            f"Hi Dr {f['investigator'].split()[-1]},", "", noisy(join_sentences(body), rng, 0.1), "",
            f"Limits I checked against (protocol section 5): {limits}.", "",
            "Can you let me know how you want to proceed?", f["coordinator"].split()[0]])

    def questions(self):
        return {
            "prescreen_decision": choice("What should the site do next with this candidate?", {
                "proceed_to_screening": "Book the full screening visit; nothing seen so far rules the candidate out.",
                "repeat_or_complete_labs": "Repeat or complete the laboratory work inside the protocol window "
                                           "before deciding anything.",
                "query_medical_monitor": "Put the eligibility question to the sponsor's medical monitor before "
                                         "going further.",
                "move_randomisation_date": "Keep the candidate but shift the planned randomisation date so the "
                                           "timing rules are met.",
                "record_screen_failure": "Log the candidate as a screen failure because a criterion is plainly missed.",
            }),
            "consent_route": choice("How should informed consent be handled for this candidate?", {
                "standard_consent_visit": "Take consent from the candidate at the screening visit on the current form.",
                "capacity_check_first": "Have the investigator assess and document decision-making capacity before "
                                        "any consent discussion.",
                "representative_consent": "Seek consent from a legally authorised representative, as the protocol "
                                          "allows, with the candidate's assent.",
                "reconsent_current_version": "Have the candidate sign the newest approved consent version before any "
                                             "further study procedure.",
                "no_permitted_route": "Do not pursue consent, because the protocol offers no permitted route for "
                                      "this candidate.",
            }),
            "cutoff_proximity": score("How close does the candidate come to an eligibility cut-off?", [
                "Comfortable: every checked value is well inside its limit.",
                "Close: a value is near a limit but clearly on the right side of it.",
                "On the line: a value or date sits at the cut-off or turns on rounding, units or which source counts.",
                "Outside: at least one value or date misses its limit.",
            ]),
            "signoff_readiness": score("How ready is the record for the investigator to sign off eligibility?", [
                "Not ready: key results are missing, out of window or contradict each other.",
                "Gaps: one required item is pending or has to be repeated.",
                "Nearly: everything is on file but one entry needs clarifying.",
                "Ready: all required results are present, current and consistent.",
            ]),
            "assessable_criteria_met": noul(
                "On the record as it stands, the candidate satisfies every eligibility criterion that can be assessed now.",
                yes="Each criterion with a result on file is met, counting the washout and lab-window rules.",
                no="At least one assessable criterion is missed, or a result on file points to ineligibility."),
        }


# =========================================================================== schema 2
class AdverseEventAssessment(Schema):
    name = "adverse_event_assessment"
    description = ("An adverse event reported from a site, assessed for seriousness, causality, expectedness and "
                   "the reporting route it triggers.")
    surfaces = {"json": 0.45, "sae_form": 0.3, "safety_narrative": 0.25}

    # verbatim term, reference-safety-information term it maps to (None: nothing close), competing explanation,
    # treatment given, test that would settle it
    EVENTS = [
        ("hepatitis", "transaminases increased", "new liver lesions on the restaging scan", "drug held, liver tests "
         "twice weekly", "liver ultrasound"),
        ("ALT increased", "transaminases increased", "gallstone seen on ultrasound", "drug held, liver tests twice "
         "weekly", "hepatitis serology"),
        ("pneumonitis", "pneumonitis", "tumour spreading along the lung lymphatics", "oral corticosteroid course",
         "bronchoscopy"),
        ("interstitial lung disease", "pneumonitis", "a recent chest infection", "oral corticosteroid course",
         "CT chest"),
        ("febrile neutropenia", "neutropenia", "an infected central line", "IV antibiotics", "blood cultures"),
        ("colitis", "diarrhoea", "a C. difficile infection", "IV fluids and corticosteroids", "stool toxin test"),
        ("atrial fibrillation", None, "long-standing hypertensive heart disease", "rate control", "echocardiogram"),
        ("pulmonary embolism", None, "cancer-associated thrombosis", "anticoagulation", "CT pulmonary angiogram"),
        ("hyponatraemia", "hyponatraemia", "a diuretic started three weeks earlier", "fluid restriction",
         "urine osmolality"),
        ("rash with blistering", "rash maculopapular", "a new antibiotic course", "topical steroid and antihistamine",
         "skin biopsy"),
        ("fall with wrist fracture", "dizziness", "slipping on an icy step", "cast and analgesia", "orthostatic "
         "blood pressure"),
    ]
    HOSP = {
        "none": "managed as an outpatient",
        "ed_discharged": "seen in the emergency department and sent home the same night",
        "observation": "kept on the observation unit for about 20 hours without formal admission",
        "admitted": "admitted as an inpatient",
        "prolonged": "already an inpatient; discharge delayed by the event",
        "elective": "in hospital for a procedure booked before enrolment; the event was found during that stay",
    }
    CAUSALITY = ["related", "possibly related", "unlikely related", "not related", "not yet assessed"]
    OTHER_RSI = ["fatigue", "nausea", "decreased appetite", "anaemia", "hypothyroidism", "infusion-related reaction",
                 "arthralgia", "pruritus"]

    def sample_factors(self, rng):
        reg = regime(rng)
        verbatim, rsi_term, alt_text, treatment, test = pick(rng, self.EVENTS)
        if reg == "clear":
            serious_clear = chance(rng, 0.55)
            grade = pick(rng, [3, 4]) if serious_clear else pick(rng, [1, 2])
            hosp = pick(rng, ["admitted", "prolonged"]) if serious_clear else "none"
            evidence = pick(rng, ["strong", "weak"])
        elif reg == "borderline":
            grade = pick(rng, [2, 3, 3])
            hosp = pick_w(rng, {"ed_discharged": 3, "observation": 3, "elective": 2, "admitted": 3, "none": 1})
            evidence = pick(rng, ["mixed", "mixed", "strong", "weak"])
        else:
            grade = pick(rng, [3, 3, 4])
            hosp = pick_w(rng, {"admitted": 3, "observation": 2, "ed_discharged": 1})
            evidence = pick(rng, ["strong", "weak"])
        if reg != "borderline" and chance(rng, 0.04):
            grade, hosp = 5, "admitted"
        life_threat = grade == 4 and chance(rng, 0.6) or (reg == "conflicting" and grade == 3 and chance(rng, 0.4))

        # expectedness against the reference safety information
        if rsi_term is None:
            rsi_status = pick(rng, ["not_listed", "class_only"])
            listed_max = None
        elif grade >= 2 and chance(rng, {"clear": 0.3, "borderline": 0.5, "conflicting": 0.4}[reg]):
            rsi_status, listed_max = "listed_lower", grade - 1
        else:
            rsi_status = "listed_same"
            listed_max = pick(rng, [None, max(grade, 3), 4]) if grade < 5 else None
        if grade == 5 and rsi_status == "listed_same":
            rsi_status, listed_max = "listed_lower", 4

        # causality evidence
        latency = {"strong": rng.randint(10, 60), "mixed": rng.randint(5, 120), "weak": rng.randint(150, 400)}[evidence]
        action = ("interrupted" if evidence == "strong" else
                  pick(rng, ["interrupted", "none", "reduced", "withdrawn"]))
        if action == "none":
            dechallenge = "not_applicable"
        elif evidence == "strong":
            dechallenge = pick(rng, ["positive", "positive", "too_early"])
        elif evidence == "weak":
            dechallenge = pick(rng, ["negative", "negative", "too_early"])
        else:
            dechallenge = pick(rng, ["positive", "negative", "too_early", "unknown"])
        alt_present = evidence != "strong" or (reg == "conflicting" and chance(rng, 0.3))
        if reg == "clear":
            inv = "related" if evidence == "strong" else pick(rng, ["not related", "unlikely related"])
        elif reg == "conflicting":
            inv = pick(rng, ["not related", "unlikely related"]) if evidence == "strong" else "related"
            if chance(rng, 0.25):
                inv = "not yet assessed"
        else:
            inv = pick(rng, ["possibly related", "possibly related", "unlikely related", "related"])

        n_missing = {"clear": pick(rng, [0, 0, 1]), "borderline": pick(rng, [0, 1, 2]),
                     "conflicting": pick(rng, [1, 2, 3])}[reg]
        gaps = rng.sample(["onset", "outcome", "test", "conmeds", "last_dose"], n_missing)
        if dechallenge == "unknown" and "outcome" not in gaps and chance(rng, 0.5):
            gaps.append("outcome")
        first_dose = rand_date(rng)
        onset = first_dose + _d(latency)
        schedule = pick(rng, ["oral, once daily", "IV infusion every 21 days", "IV infusion every 14 days"])
        since_last = (min(latency, pick(rng, [0, 1])) if schedule.startswith("oral")
                      else latency % (21 if "21" in schedule else 14))
        aware = onset + _d(pick(rng, [0, 0, 1, 2, 3]))
        site = _site_no(rng)
        drug = _drug(rng)
        return {
            "regime": reg, "evidence": evidence, "verbatim": verbatim, "rsi_term": rsi_term, "alt_cause": alt_text,
            "alt_present": alt_present, "treatment": treatment, "test": test, "grade": grade, "hospital": hosp,
            "life_threatening_ticked": life_threat, "rsi_status": rsi_status, "rsi_listed_max_grade": listed_max,
            "latency_days": latency, "action_taken": action, "dechallenge": dechallenge,
            "investigator_causality": inv, "gaps": sorted(gaps),
            "outcome": ("fatal" if grade == 5 else
                        pick(rng, ["recovering", "recovered", "not recovered"]) if dechallenge != "unknown"
                        else "not recovered"),
            "drug": drug, "drug_class": pick(rng, ["kinase inhibitor", "bispecific antibody", "antibody-drug conjugate",
                                                   "checkpoint inhibitor"]),
            "protocol": _protocol(rng, drug), "subject": _subject(rng, site), "site": site,
            "blinding": pick(rng, ["open-label", "open-label", "double-blind"]),
            "first_dose": first_dose, "onset": onset, "last_dose": onset - _d(since_last), "aware": aware,
            "schedule": schedule, "sent": aware + _d(pick(rng, [0, 0, 1, 1, 2, 4])),
            "case_id": ident(rng, "SR", 6, "-"),
            "ib_edition": rng.randint(3, 11), "rsi_others": rng.sample(self.OTHER_RSI, 3),
            "investigator": person(rng), "indication": pick(rng, ONC_INDICATIONS),
        }

    # ------------------------------------------------------------------ text pieces
    def _rsi(self, f):
        others = list(f["rsi_others"])
        if f["rsi_status"] in ("listed_same", "listed_lower"):
            g = f["rsi_listed_max_grade"]
            others.insert(1, f"{f['rsi_term']} ({'all grades' if g is None else f'grades 1-{g}'})")
            return {"listed_terms": others}
        if f["rsi_status"] == "class_only":
            return {"listed_terms": others,
                    "class_statement": f"{f['verbatim'].split(' with ')[0]} has been reported with other "
                                       f"{f['drug_class']}s; no cases with {f['drug']} to date"}
        return {"listed_terms": others}

    ROUTES = ("Reporting rules (safety plan 9.2): an event is serious only if it is fatal, life-threatening (box ticked) "
              "or needs or prolongs an inpatient admission; ED visits, observation stays and pre-planned admissions do "
              "not count. Serious events go to the sponsor within 24 h. A serious event that is also unexpected (not "
              "listed in the RSI at the grade seen) and that the investigator calls related, possibly related or not "
              "yet assessed is expedited to regulators: 7 days if fatal or life-threatening, otherwise 15 days.")

    def _dechallenge_text(self, f):
        return {"positive": "improved after the drug was stopped", "negative": "no improvement after stopping",
                "too_early": "stopped too recently to judge", "unknown": "not documented",
                "not_applicable": "drug continued"}[f["dechallenge"]]

    def _narrative(self, f, rng):
        g = f["gaps"]
        s = [f"Subject {f['subject']} ({f['indication']}) started {f['drug']} ({f['schedule']}) on "
             f"{_fd(f['first_dose'], rng)}.",
             ("Presented with " if "onset" in g else f"On {_fd(f['onset'], rng)} presented with ")
             + f"{f['verbatim']}, CTCAE grade {f['grade']}; {self.HOSP[f['hospital']]}."]
        if "test" not in g:
            t = f["test"][0].upper() + f["test"][1:]
            s.append(pick(rng, [f"{t} requested.", f"{t} done, report awaited."]))
        if f["alt_present"]:
            s.append(pick(rng, [f"Of note: {f['alt_cause']}.", f"Team also considering {f['alt_cause']}."]))
        s.append(f"Treated with {f['treatment']}.")
        s.append({"interrupted": "Study drug interrupted", "none": "Study drug continued", "reduced": "Dose reduced",
                  "withdrawn": "Study drug permanently stopped"}[f["action_taken"]]
                 + ("" if "last_dose" in g or f["action_taken"] == "none" else f" (last dose {_fd(f['last_dose'], rng)})") + ".")
        s.append({"positive": "The event improved after the drug was stopped.",
                  "negative": "No improvement after the drug was stopped.",
                  "too_early": "Too soon after stopping to judge the response."}.get(f["dechallenge"], ""))
        if f["outcome"] == "fatal":
            s.append("The subject died in hospital.")
        return noisy(join_sentences(s), rng, 0.12)

    def render(self, f, rng, surface):
        g = f["gaps"]
        outcome = None if "outcome" in g else f["outcome"]
        rsi = self._rsi(f)
        if surface == "json":
            return dumps(prune({
                "report": {"case_id": f["case_id"], "protocol": f["protocol"], "subject": f["subject"],
                           "blinding": f["blinding"], "site_aware": f["aware"].isoformat(),
                           "sent_to_sponsor": f["sent"].isoformat(), "reporter": f["investigator"]},
                "event": {"verbatim_term": f["verbatim"], "onset": None if "onset" in g else f["onset"].isoformat(),
                          "ctcae_grade": f["grade"], "hospital_course": self.HOSP[f["hospital"]],
                          "life_threatening_box_ticked": f["life_threatening_ticked"], "outcome": outcome or "unknown"},
                "study_drug": {"name": f["drug"], "first_dose": f["first_dose"].isoformat(),
                               "last_dose_before_onset": None if "last_dose" in g else f["last_dose"].isoformat(),
                               "action_taken": f["action_taken"], "dechallenge": self._dechallenge_text(f)},
                "assessment": {"investigator_causality": f["investigator_causality"],
                               "other_explanation": f["alt_cause"] if f["alt_present"] else None,
                               "concomitant_medications": "not provided" if "conmeds" in g else "listed on page 3"},
                "reference_safety_information": dict(rsi, source=f"Investigator's Brochure ed. {f['ib_edition']}"),
                "narrative": self._narrative(f, rng),
                "reporting_rules": self.ROUTES,
            }))
        if surface == "sae_form":
            rows = [("Case", f["case_id"]), ("Protocol / subject", f"{f['protocol']} / {f['subject']}"),
                    ("Study drug", f"{f['drug']} ({f['blinding']})"), ("Event (verbatim)", f["verbatim"]),
                    ("Onset", None if "onset" in g else _fd(f["onset"], rng)), ("Grade", f["grade"]),
                    ("Hospital course", self.HOSP[f["hospital"]]),
                    ("Life-threatening", "X" if f["life_threatening_ticked"] else "-"),
                    ("Outcome", outcome), ("Action with study drug", f["action_taken"]),
                    ("Dechallenge", self._dechallenge_text(f)),
                    ("Causality (investigator)", f["investigator_causality"]),
                    ("Site awareness", _fd(f["aware"], rng)), ("Sent", _fd(f["sent"], rng))]
            rsi_line = "; ".join(rsi["listed_terms"]) + (f". {rsi['class_statement']}" if "class_statement" in rsi else "")
            return ("SAFETY REPORT FORM (SAE / event of special interest) -- initial\n" + _kv(rows, rng, skip_none=False)
                    + f"\n\nIB ed. {f['ib_edition']} reference safety information: {rsi_line}"
                    + "\n\nNarrative:\n" + self._narrative(f, rng) + "\n\n" + self.ROUTES)
        # safety_narrative: the sponsor safety physician's intake note
        lines = [f"Intake -- {f['case_id']} ({f['protocol']}, subject {f['subject']}, {f['blinding']})", "",
                 self._narrative(f, rng), "",
                 bullets([f"Investigator causality: {f['investigator_causality']}",
                          f"Life-threatening box: {'ticked' if f['life_threatening_ticked'] else 'not ticked'}",
                          f"Outcome: {show(outcome, rng)}",
                          f"Site aware {_fd(f['aware'], rng)}, report received {_fd(f['sent'], rng)}",
                          "Concomitant medications: " + ("missing" if "conmeds" in g else "on file")], rng),
                 "", f"RSI (IB ed. {f['ib_edition']}): " + ", ".join(rsi["listed_terms"])
                 + (f". {rsi['class_statement']}." if "class_statement" in rsi else "."), "", self.ROUTES]
        return "\n".join(lines)

    def questions(self):
        return {
            "seriousness_met": noul(
                "The event meets at least one regulatory seriousness criterion.",
                yes="It caused death, was life-threatening, required or prolonged inpatient admission, caused lasting "
                    "disability, or is an important medical event in its own right.",
                no="None of the seriousness criteria applies to the event as documented."),
            "drug_causality": score("How strongly does the evidence implicate the study drug in this event?", [
                "Unrelated: another cause explains it fully, or the timing rules the drug out.",
                "Unlikely: the timing fits poorly or a better explanation is on the table.",
                "Possible: the timing fits, but other causes are about as plausible.",
                "Probable: the timing fits, it eased when the drug stopped, and nothing explains it better.",
                "Near certain: a recognised reaction to the drug with a clear stop-and-recover pattern.",
            ]),
            "rsi_expectedness": choice("How does the event compare with the study drug's reference safety information?", {
                "listed_at_observed_severity": "The reference safety information lists this event at the severity "
                                               "that was seen.",
                "listed_only_milder": "The event is listed, but only at a lower severity than this occurrence, so "
                                      "this one counts as unexpected.",
                "class_warning_only": "Only a statement about the drug class mentions it; the product's own listing "
                                      "does not.",
                "not_covered": "Nothing in the reference safety information covers this kind of event.",
            }),
            "reporting_route": choice("Which reporting path does this event call for?", {
                "crf_only": "Record it on the case report form only; no serious adverse event report is due.",
                "sae_report_no_expedite": "Send a serious adverse event report to the sponsor within 24 hours, with "
                                          "no expedited report to regulators.",
                "expedited_15_day": "Report it to regulators as a suspected unexpected serious reaction within 15 "
                                    "calendar days.",
                "expedited_7_day": "Report it to regulators within 7 calendar days as a fatal or life-threatening "
                                   "suspected unexpected reaction.",
            }),
            "report_quality": score("How well does the report equip a safety physician to assess the case?", [
                "Poor: onset, outcome or dosing details are missing.",
                "Thin: the basics are there, but key tests or the drug-stop response are not.",
                "Workable: enough to assess, with a few follow-up questions.",
                "Complete: timeline, tests, treatment, outcome and dosing are all given.",
            ]),
        }


# =========================================================================== schema 3
class ProtocolDeviation(Schema):
    name = "protocol_deviation"
    description = "A protocol deviation logged at a site: classify it, decide on IRB reporting and set the corrective action."
    surfaces = {"json": 0.45, "deviation_log": 0.3, "cra_email": 0.25}

    LABEL = {"visit_window": "visit outside window", "missed_safety_lab": "missed safety assessment",
             "outdated_consent": "informed consent version", "procedure_before_consent": "procedure before consent",
             "dosing_error": "study drug dosing", "temperature_excursion": "IP storage temperature"}

    def _details(self, t, level, rng):
        """Type-specific facts: (description, reason given by the site, consequence so far, endpoint touched)."""
        if t == "visit_window":
            w, target = pick(rng, [2, 3, 7]), pick(rng, [29, 57, 85, 113, 169])
            extra = {"mild": rng.randint(1, 3), "edge": pick(rng, [0, 1]), "severe": rng.randint(6, 18)}[level]
            off = (w + extra) * pick(rng, [1, 1, -1])
            endpoint = level == "severe" or (level == "edge" and chance(rng, 0.5))
            return (f"Day {target} visit took place on study day {target + off}; protocol window is +/-{w} days. "
                    + ("Tumour imaging for the primary endpoint was due at this visit."
                       if endpoint else "Safety labs and vital signs were due."),
                    pick(rng, ["subject travelling", "clinic closed for a public holiday",
                               "scheduling system booked the wrong week", "subject unwell with a cold"]),
                    "assessments done at the late visit" if off > 0 else "assessments done early", endpoint)
        if t == "missed_safety_lab":
            lab = pick(rng, ["liver panel", "potassium and magnesium", "pre-dose ECG", "full blood count"])
            if level == "mild":
                desc = f"Pre-dose {lab} not drawn at the cycle visit; dosing delayed one day until it was done and resulted."
                cons = "result normal; dose given a day late within the cycle window"
            elif level == "edge":
                desc = (f"Pre-dose {lab} not drawn within the required 72 h; dose given on a result from "
                        f"{rng.randint(5, 9)} days earlier.")
                cons = pick(rng, ["next result normal", "next result still pending"])
            else:
                desc = (f"Pre-dose {lab} not drawn; dose given on a result from {rng.randint(10, 21)} days earlier.")
                cons = "next result showed a grade 3 abnormality and the subject was admitted"
            return desc, pick(rng, ["phlebotomy slot missed", "new nurse unaware of the requirement",
                                    "sample haemolysed and not repeated"]), cons, False
        if t == "outdated_consent":
            new = pick(rng, ["3.0", "4.0", "5.1"])
            old = _prev_version(new, rng)
            adds = {"mild": "administrative corrections (contact numbers, typos)",
                    "edge": "an updated visit schedule and one extra optional blood sample",
                    "severe": "new safety information about liver injury"}[level]
            n = {"mild": 1, "edge": rng.randint(1, 2), "severe": rng.randint(2, 4)}[level]
            return (f"Subject attended {n} visit(s) still on ICF v{old} after v{new} was approved and activated at "
                    f"the site. v{new} adds {adds}.", pick(rng, ["re-consent flag not set in the site tracker",
                                                               "coordinator on leave", "subject declined to wait"]),
                    "re-consented at the following visit" if level != "severe" else "dosed on each of those visits",
                    False)
        if t == "procedure_before_consent":
            if level == "mild":
                return ("Screening CT taken 12 days before consent as routine care was used for the baseline "
                        "assessment. Protocol section 7.1 lets standard-of-care imaging from the prior 28 days count.",
                        "routine care scan", "no study-specific procedure was done early", True)
            if level == "edge":
                return ("Consent form is dated the same day as the screening blood draw, but no signature time was "
                        "written; the lab timestamp is 08:50 and the consent visit was booked for 09:30.",
                        "clinic running early", "cannot tell which came first", False)
            return (f"Research tumour biopsy performed {rng.randint(1, 3)} days before the date on the consent "
                    "signature page.", "biopsy slot would otherwise be lost", "biopsy tissue already sent to the "
                    "central lab", True)
        if t == "dosing_error":
            if level == "mild":
                return ("Subject missed one daily dose and reported it at the next visit; diary updated.",
                        "forgot while travelling", "no symptoms", False)
            if level == "edge":
                return (f"Pharmacy dispensed the bottle for one dose level above the assigned level; taken for "
                        f"{rng.randint(2, 4)} days before the error was noticed.", "look-alike labels",
                        pick(rng, ["no new adverse events", "mild nausea, resolved"]), False)
            return (f"Subject took twice the assigned dose for {rng.randint(5, 8)} days after misreading the dosing "
                    "card.", "dosing card unclear", "grade 3 diarrhoea on the fifth day", False)
        # temperature_excursion
        if level == "mild":
            return ("Drug fridge reached 9.1 °C for 35 minutes (range 2-8 °C). Kits quarantined; none dispensed "
                    "before the sponsor confirmed they were within stability data.", "door left ajar",
                    "no affected kit given", False)
        if level == "edge":
            return (f"Drug fridge at {round(rng.uniform(10.0, 11.5), 1)} °C for about {rng.randint(4, 7)} h overnight "
                    "(range 2-8 °C). One kit from it was given to this subject before the sponsor's stability "
                    "assessment came back.", "alarm not escalated overnight", "stability assessment still pending",
                    False)
        return (f"Kits stored at 14 °C for about {rng.randint(24, 40)} h (range 2-8 °C); two were given to this "
                "subject. Sponsor later assessed the conditions as outside the stability data.",
                "power cut, backup fridge not used", "subject has had no new adverse events", False)

    def sample_factors(self, rng):
        reg = regime(rng)
        t = pick(rng, list(self.LABEL))
        level = {"clear": pick(rng, ["mild", "severe"]), "borderline": "edge",
                 "conflicting": pick(rng, ["mild", "severe"])}[reg]
        desc, reason, cons, endpoint = self._details(t, level, rng)
        if reg == "clear":
            repeat = 0 if level == "mild" else pick(rng, [0, 1])
            site_view = "minor" if level == "mild" else "important"
        elif reg == "borderline":
            repeat = pick(rng, [0, 0, 1, 2])
            site_view = pick(rng, ["minor", "important", None])
        else:
            # the site's own view and the facts pull apart: a small slip that keeps recurring, or a serious one
            # the site files as minor
            repeat = rng.randint(3, 5) if level == "mild" else 0
            site_view = "minor"
        prior_capa = repeat >= 2 and chance(rng, 0.6)
        found = rand_date(rng)
        site = _site_no(rng)
        drug = _drug(rng)
        return {
            "regime": reg, "dev_type": t, "level": level, "description": desc, "reason": reason,
            "consequence": cons, "endpoint_touched": endpoint, "repeat_6mo": repeat, "prior_capa": prior_capa,
            "site_view": site_view, "detected_by": pick(rng, ["site self-identified", "CRA at monitoring visit",
                                                             "central data review"]),
            "capa_status": pick_w(rng, {"none yet": 3, "note to file written": 3, "staff retrained": 2}),
            "irb_days": pick(rng, [5, 10]), "found": found, "occurred": found - _d(rng.randint(1, 40)),
            "protocol": _protocol(rng, drug), "drug": drug, "site": site, "subject": _subject(rng, site),
            "dev_id": f"PD-{site}-{rng.randrange(1, 60):04d}", "cra": person(rng), "coordinator": person(rng),
            "blinded": chance(rng, 0.4), "policy_variant": rng.randrange(4),
        }

    PLAN = [
        "Important = may significantly affect a participant's safety, rights or wellbeing, or the reliability of key "
        "data (eligibility, consent, dosing, primary endpoint). Everything else is minor.",
        "A deviation is important when it could materially change a participant's safety, rights or wellbeing, or the "
        "completeness or accuracy of key data such as eligibility, consent, dosing or the primary endpoint; the rest "
        "are minor.",
    ]
    IRB = [
        "Report within {n} working days any deviation that increased risk to participants, affected their rights or "
        "welfare, or recurred after a corrective action; others go in the continuing-review summary.",
        "Prompt report (within {n} working days) required for deviations that put participants at greater risk, "
        "touched their rights or welfare, or repeated after corrective action. All others: summarise at continuing "
        "review.",
    ]

    def _policies(self, f):
        return {"deviation_plan": self.PLAN[f["policy_variant"] % 2],
                "irb_policy": self.IRB[f["policy_variant"] // 2].format(n=f["irb_days"])}

    def render(self, f, rng, surface):
        pol = self._policies(f)
        if surface == "json":
            return dumps(prune({
                "deviation": {"id": f["dev_id"], "protocol": f["protocol"], "subject": f["subject"],
                              "category": self.LABEL[f["dev_type"]], "description": f["description"],
                              "reason_given": f["reason"], "consequence": f["consequence"],
                              "occurred": f["occurred"].isoformat(), "identified": f["found"].isoformat(),
                              "identified_by": f["detected_by"], "site_proposed_class": f["site_view"]},
                "site_history": {"same_category_last_6_months": f["repeat_6mo"],
                                 "earlier_capa_for_category": f["prior_capa"] if f["repeat_6mo"] else None},
                "actions": {"corrective_action": f["capa_status"], "irb_notified": False},
                "policy_excerpts": pol,
            }))
        if surface == "deviation_log":
            rows = [("Deviation ID", f["dev_id"]), ("Protocol", f["protocol"]), ("Subject", f["subject"]),
                    ("Category", self.LABEL[f["dev_type"]]), ("Date occurred", _fd(f["occurred"], rng)),
                    ("Date identified", _fd(f["found"], rng)), ("Identified by", f["detected_by"]),
                    ("Description", f["description"]), ("Root cause (site)", f["reason"]),
                    ("Impact noted", f["consequence"]), ("Site classification", show(f["site_view"], rng)),
                    ("Similar deviations, last 6 months", f["repeat_6mo"]),
                    ("Corrective action", f["capa_status"])]
            if f["prior_capa"]:
                rows.append(("Note", "a CAPA for this category was closed earlier this year"))
            return ("SITE PROTOCOL DEVIATION LOG -- entry\n" + _kv(rows, rng, skip_none=False)
                    + "\n\nPlan: " + pol["deviation_plan"] + "\nIRB: " + pol["irb_policy"])
        body = [f"During review for {f['protocol']} I logged {f['dev_id']} for subject {f['subject']}.",
                f["description"], f"Site's explanation: {f['reason']}. So far: {f['consequence']}."]
        if f["site_view"]:
            body.append(f"The site has marked it {f['site_view']}.")
        if f["repeat_6mo"]:
            body.append(f"This is the {_ordinal(f['repeat_6mo'] + 1)} {self.LABEL[f['dev_type']]} deviation at the "
                        "site in six months" + (", after an earlier CAPA." if f["prior_capa"] else "."))
        body.append(f"Corrective action so far: {f['capa_status']}.")
        return "\n".join([
            f"Subject: {f['dev_id']} -- {self.LABEL[f['dev_type']]} (site {f['site']})",
            f"From: {f['cra']} (CRA)", f"To: {f['coordinator']}; study management", "",
            noisy(join_sentences(body), rng, 0.12), "",
            f"For reference -- deviation plan: {pol['deviation_plan']} IRB SOP: {pol['irb_policy']}", "",
            f["cra"].split()[0]])

    def questions(self):
        return {
            "deviation_class": choice("How should this event be classified under the study's deviation plan?", {
                "no_deviation": "It stays within what the protocol permits, so nothing needs to be logged.",
                "minor_deviation": "A departure with no meaningful effect on participant safety, rights or key data.",
                "important_deviation": "A departure that could significantly affect a participant's safety or rights "
                                       "or the reliability of key data.",
                "suspected_serious_breach": "A breach likely to affect participant safety or the trial's scientific "
                                            "value to a significant degree, needing escalation to the sponsor.",
            }),
            "irb_prompt_report": noul(
                "The site must report this deviation to its IRB within the prompt-reporting deadline instead of "
                "holding it for the continuing-review summary.",
                yes="It raised risk to the participant, touched their rights or welfare, or recurred after a "
                    "corrective action.",
                no="It meets none of the prompt-reporting conditions in the IRB policy."),
            "participant_exposure": score("How far did the deviation expose the participant to risk or loss of a protection?", [
                "None: nothing about the participant's care or rights changed.",
                "Theoretical: a safeguard lapsed, but no harm followed and none was likely.",
                "Real: the participant faced avoidable risk or lost a protection they were owed.",
                "Harm: the participant was hurt, or came close to it, because of the deviation.",
            ]),
            "key_data_impact": score("How much does the deviation weaken this participant's key trial data?", [
                "None: key data are untouched.",
                "Slight: a secondary value is late or missing but recoverable.",
                "Material: a key efficacy or safety measurement is compromised.",
                "Severe: the participant's primary endpoint or eligibility data may be unusable.",
            ]),
            "corrective_response": choice("What corrective response fits this deviation?", {
                "note_to_file": "Document it in a note to file; nothing further is needed.",
                "targeted_retraining": "Retrain the staff involved on the step that was missed.",
                "root_cause_capa": "Open a formal corrective and preventive action with a root-cause analysis.",
                "sponsor_quality_escalation": "Escalate to the sponsor's quality team, which may send a for-cause audit.",
            }),
        }


# =========================================================================== schema 4
class SiteFeasibility(Schema):
    name = "site_feasibility"
    description = "A candidate site's feasibility package for a new study: select it, keep it in reserve, select on conditions or decline."
    surfaces = {"json": 0.45, "feasibility_questionnaire": 0.3, "selection_memo": 0.25}

    ONC_REQ = ["-80 °C freezer for PK samples", "triplicate ECG with central upload",
               "contrast CT within 72 h of dosing", "2-8 °C drug storage with continuous logging",
               "infusion chair for a 4-hour infusion", "overnight stay for 24 h PK sampling"]
    OTHER_REQ = ["-80 °C freezer for biomarker samples", "triplicate ECG with central upload",
                 "2-8 °C drug storage with continuous logging", "spirometry with central over-read",
                 "a trained assessor for the disease-activity score", "overnight stay for 24 h PK sampling"]
    CAP = {"in_place": "in place", "shared": "shared with another department", "on_order": "on order",
           "partner": "through a partner hospital nearby", "missing": "not available"}
    DIMS = ["enrol", "quality", "capability", "speed"]

    def sample_factors(self, rng):
        reg = regime(rng)
        if reg == "clear":
            lv = pick(rng, ["good", "bad"])
            q = {d: lv for d in self.DIMS}
            if chance(rng, 0.3):
                q[pick(rng, self.DIMS)] = "mid"
        elif reg == "borderline":
            q = {d: pick(rng, ["mid", "mid", "good"]) for d in self.DIMS}
        else:
            bad = rng.sample(self.DIMS, 2)
            q = {d: ("bad" if d in bad else "good") for d in self.DIMS}

        months = pick(rng, [12, 15, 18])
        competing = {"good": pick(rng, [0, 1]), "mid": pick(rng, [1, 2]), "bad": pick(rng, [2, 3, 4])}[q["enrol"]]
        pool = int(pick(rng, [40, 70, 110, 160, 240, 380]) * rng.uniform(0.85, 1.15))
        realistic = pool * 0.06 * (1 - 0.15 * competing) * months / 12
        fac = {"good": rng.uniform(0.75, 1.05), "mid": rng.uniform(1.3, 1.8), "bad": rng.uniform(2.3, 3.5)}[q["enrol"]]
        projected = max(3, int(round(realistic * fac)))
        prior_promised = rng.randint(6, 20)
        ratio = {"good": rng.uniform(0.85, 1.25), "mid": rng.uniform(0.55, 0.8), "bad": rng.uniform(0.2, 0.45)}[q["enrol"]]
        new_to_sponsor = chance(rng, 0.15)

        audit = {"good": pick(rng, ["no findings", "two minor findings, both closed"]),
                 "mid": pick(rng, ["three minor findings, one still open", "one major finding on delegation log "
                                   "gaps, CAPA closed"]),
                 "bad": pick(rng, ["major finding on consent documentation, CAPA still open",
                                   "major finding on drug accountability, repeat of an earlier finding"])}[q["quality"]]
        query_days = {"good": rng.randint(3, 8), "mid": rng.randint(9, 16), "bad": rng.randint(18, 35)}[q["quality"]]
        pi_trials = {"good": rng.randint(6, 20), "mid": rng.randint(2, 5), "bad": rng.randint(0, 1)}[q["quality"]]
        fte = {"good": pick(rng, [1.0, 1.2, 1.5]), "mid": pick(rng, [0.5, 0.6, 0.8]),
               "bad": pick(rng, [0.2, 0.3, 0.4])}[q["quality"]]

        indication = pick(rng, ONC_INDICATIONS + OTHER_INDICATIONS)
        reqs = rng.sample(self.ONC_REQ if indication in ONC_INDICATIONS else self.OTHER_REQ, 4)
        caps = {r: pick(rng, ["in_place", "in_place", "shared"]) for r in reqs}
        if q["capability"] == "mid":
            caps[reqs[0]] = pick(rng, ["on_order", "partner"])
        elif q["capability"] == "bad":
            caps[reqs[0]] = "missing"
            caps[reqs[1]] = pick(rng, ["missing", "on_order"])
        on_order_weeks = rng.randint(3, 10)

        fpi_target = pick(rng, [10, 12, 16, 20])
        est = {"good": fpi_target - rng.randint(2, 5), "mid": fpi_target + rng.randint(0, 6),
               "bad": fpi_target + rng.randint(8, 16)}[q["speed"]]
        irb = pick(rng, ["central IRB", "central IRB", "local IRB, meets monthly"])
        contract = max(3, est - (3 if irb.startswith("central") else 6) - rng.randint(0, 2))
        drug = _drug(rng)
        return {
            "regime": reg, "dims": q, "months": months, "target_total": pick(rng, [6, 8, 10, 12]),
            "competing": competing, "pool": maybe(rng, pool, 0.15 if reg != "clear" else 0.0),
            "pool_pi_estimate": int(pool * rng.uniform(1.3, 2.2)), "projected_total": projected,
            "prior_promised": None if new_to_sponsor else prior_promised,
            "prior_enrolled": None if new_to_sponsor else int(round(prior_promised * ratio)),
            "audit": audit, "query_days": None if new_to_sponsor else query_days, "pi_trials": pi_trials,
            "coordinator_fte": fte, "capabilities": caps, "on_order_weeks": on_order_weeks,
            "fpi_target_weeks": fpi_target, "est_activation_weeks": max(4, est), "irb": irb,
            "contract_weeks": contract, "budget_over_fmv_pct": pick(rng, [0, 0, 5, 12, 25]),
            "site_no": _site_no(rng), "institution": _institution(rng), "pi": person(rng), "lead": person(rng),
            "protocol": _protocol(rng, drug, pick(rng, [2, 3])), "indication": indication,
        }

    def _model(self, f):
        """The sponsor's enrolment model from the rendered pool (the PI's estimate when no EHR count exists)."""
        base = f["pool"] if f["pool"] is not None else f["pool_pi_estimate"]
        return round(base * 0.06 * (1 - 0.15 * f["competing"]) * f["months"] / 12, 1)

    def _model_text(self, f):
        src = "EHR count" if f["pool"] is not None else "PI's estimate, no EHR count"
        return (f"about {self._model(f)} participants over {f['months']} months (sponsor model: 6% of matching patients "
                f"from the {src}, less 15% per competing study)")

    def _cap_text(self, f, r):
        s = f["capabilities"][r]
        return self.CAP[s] + (f", delivery in about {f['on_order_weeks']} weeks" if s == "on_order" else "")

    def render(self, f, rng, surface):
        pool_txt = (f"{f['pool']} (EHR query, last 12 months)" if f["pool"] is not None
                    else f"about {f['pool_pi_estimate']} (PI estimate; EHR query not run)")
        prior = (None if f["prior_promised"] is None else
                 f"committed {f['prior_promised']}, enrolled {f['prior_enrolled']}")
        if surface == "json":
            return dumps(prune({
                "site": {"number": f["site_no"], "institution": f["institution"], "principal_investigator": f["pi"],
                         "pi_studies_led": f["pi_trials"], "coordinator_fte": f["coordinator_fte"]},
                "study": {"protocol": f["protocol"], "indication": f["indication"],
                          "target_per_site": f["target_total"], "enrolment_months": f["months"],
                          "first_participant_target_weeks": f["fpi_target_weeks"]},
                "enrolment": {"site_projection_total": f["projected_total"],
                              "sponsor_model_from_pool": self._model_text(f),
                              "matching_patients_12m": f["pool"],
                              "pi_estimate_matching_patients": None if f["pool"] is not None else f["pool_pi_estimate"],
                              "competing_studies_same_population": f["competing"]},
                "track_record": {"last_similar_study": prior or "no earlier study with this sponsor",
                                 "median_query_resolution_days": f["query_days"], "last_audit": f["audit"]},
                "capabilities": {r: self._cap_text(f, r) for r in f["capabilities"]},
                "start_up": {"contract_turnaround_weeks": f["contract_weeks"], "ethics_review": f["irb"],
                             "estimated_weeks_to_activation": f["est_activation_weeks"],
                             "budget_vs_fair_market_value": (f"+{f['budget_over_fmv_pct']}%"
                                                             if f["budget_over_fmv_pct"] else "within range")},
            }))
        if surface == "feasibility_questionnaire":
            qa = [("Patients with this condition seen in the past 12 months?", pool_txt),
                  ("Participants you expect to enrol over the enrolment period?",
                   f"{f['projected_total']} over {f['months']} months [sponsor check: {self._model_text(f)}]"),
                  ("Competing studies in the same population?", f["competing"]),
                  ("Studies led by the proposed PI?", f["pi_trials"]),
                  ("Dedicated coordinator time (FTE)?", f["coordinator_fte"]),
                  ("Enrolment on your last comparable study?", show(prior, rng)),
                  ("Most recent audit or inspection outcome?", f["audit"])]
            qa += [(f"Can you provide: {r}?", self._cap_text(f, r)) for r in f["capabilities"]]
            qa += [("Contract and ethics timeline?", f"contract ~{f['contract_weeks']} wk; {f['irb']}; activation "
                                                     f"expected ~{f['est_activation_weeks']} wk")]
            head = (f"SITE FEASIBILITY QUESTIONNAIRE -- {f['protocol']} ({f['indication']})\n"
                    f"Site {f['site_no']}, {f['institution']}; completed by {f['pi']}\n"
                    f"Sponsor asks: {f['target_total']} participants per site, first participant within "
                    f"{f['fpi_target_weeks']} weeks of selection.\n")
            return head + "\n".join(f"{i}. {qq}\n   {a}" for i, (qq, a) in enumerate(qa, 1))
        # selection_memo
        rows = [["Projection", f"{f['projected_total']} in {f['months']} mo (target {f['target_total']})"],
                ["Sponsor model", self._model_text(f)],
                ["Patient pool", pool_txt], ["Competing studies", f["competing"]],
                ["Last comparable study", show(prior, rng)], ["Query resolution", show(f["query_days"], rng,
                                                                                      lambda v: f"{v} days median")],
                ["Audit history", f["audit"]],
                ["Activation", f"~{f['est_activation_weeks']} wk vs {f['fpi_target_weeks']} wk target"]]
        gaps = [f"{r}: {self._cap_text(f, r)}" for r in f["capabilities"] if f["capabilities"][r] != "in_place"]
        para = [f"Site {f['site_no']} ({f['institution']}, PI {f['pi']}) returned its package for {f['protocol']}.",
                f"The PI has led {f['pi_trials']} studies; coordinator time offered is {f['coordinator_fte']} FTE.",
                f"Budget request is {f['budget_over_fmv_pct']}% above fair market value." if f["budget_over_fmv_pct"]
                else ""]
        return "\n".join([f"Feasibility summary -- {f['protocol']} -- prepared by {f['lead']}", "",
                          noisy(join_sentences(para), rng, 0.1), "", table(["Item", "Site response"], rows, rng),
                          "", "Equipment and facilities:",
                          bullets(gaps or ["all listed requirements in place"], rng)])

    def questions(self):
        return {
            "site_selection": choice("What should the feasibility team decide about this site?", {
                "select_first_wave": "Select the site and activate it with the first group of sites.",
                "select_on_conditions": "Select the site only once named gaps, such as equipment or staffing, are closed.",
                "keep_in_reserve": "Hold the site in reserve and activate it only if other sites fall short.",
                "decline_site": "Do not use this site for the study.",
            }),
            "projection_credibility": score("How believable is the site's enrolment projection?", [
                "Unrealistic: more than double the sponsor's pool model, or well above it from a site that enrolled "
                "under half of its last commitment.",
                "Optimistic: between a quarter above and double the sponsor's pool model.",
                "Plausible: in line with the pool model, but the pool is only the PI's estimate or there is no "
                "earlier study to judge the site by.",
                "Well supported: in line with a model built on a counted pool, from a site that delivered at least 80% "
                "of its last commitment.",
            ]),
            "activation_timing": score("How does the site's likely first enrolment compare with the study's target?", [
                "Far behind: expected activation is eight or more weeks after the first-participant target.",
                "Late: three to seven weeks after the target.",
                "On target: within two weeks either side of it.",
                "Ahead: three or more weeks before the target.",
            ]),
            "facilities_complete": noul(
                "The site can run every protocol-required procedure with facilities it already has access to.",
                yes="Each listed requirement is in place, shared on site, or covered by an existing arrangement.",
                no="At least one requirement is missing or depends on equipment that has not arrived."),
            "quality_plan_needed": noul(
                "The site's record on earlier studies calls for a quality plan before it is activated."),
        }


# =========================================================================== schema 5
class DoseModification(Schema):
    name = "dose_modification"
    description = "A dosing-day decision for a participant with a toxicity, made under the protocol's dose-modification table."
    surfaces = {"json": 0.45, "lab_flowsheet": 0.3, "investigator_note": 0.25}

    # value bands per grade 0..4 (mid-band draws for clear cases); ALT is in multiples of the ULN
    BANDS = {
        "neutropenia": [(2.1, 4.5), (1.55, 1.95), (1.05, 1.45), (0.55, 0.95), (0.1, 0.45)],
        "thrombocytopenia": [(160, 320), (85, 140), (52, 72), (27, 47), (8, 22)],
        "alt_increase": [(0.4, 0.95), (1.2, 2.8), (3.2, 4.8), (5.5, 12.0), (21.0, 30.0)],
    }
    # the value at the boundary between grade g-1 and g, for g = 1..4
    EDGES = {"neutropenia": [2.0, 1.5, 1.0, 0.5], "thrombocytopenia": [150, 75, 50, 25],
             "alt_increase": [1.0, 3.0, 5.0, 20.0]}
    GRADING = {
        "neutropenia": "ANC grade 1 <2.0-1.5; grade 2 <1.5-1.0; grade 3 <1.0-0.5; grade 4 <0.5 (x10^9/L)",
        "thrombocytopenia": "Platelets grade 1 <150-75; grade 2 <75-50; grade 3 <50-25; grade 4 <25 (x10^9/L)",
        "alt_increase": "ALT grade 1 >ULN-3.0 x ULN; grade 2 >3.0-5.0 x ULN; grade 3 >5.0-20.0 x ULN; grade 4 >20 x ULN",
    }
    DOSES = [(200, 150, 100), (400, 300, 200), (60, 40, 20), (120, 80, 60)]

    @staticmethod
    def _grade(tox, v):
        if tox == "alt_increase":
            return 0 if v <= 1.0 else 1 if v <= 3.0 else 2 if v <= 5.0 else 3 if v <= 20.0 else 4
        for g, e in enumerate(DoseModification.EDGES[tox]):
            if v >= e:
                return g
        return 4

    def _value(self, tox, rng, grade=None, edge_between=None):
        """A lab value (x10^9/L, or x ULN for ALT) mid-band for `grade`, or within ~5% of the edge below grade
        `edge_between` (the boundary between edge_between-1 and edge_between)."""
        if edge_between is not None:
            e = self.EDGES[tox][edge_between - 1]
            v = near(rng, e, e * 0.05, digits=3)
        else:
            v = rng.uniform(*self.BANDS[tox][grade])
        return round(v, 0) if tox == "thrombocytopenia" else round(v, 2) if tox == "neutropenia" else v

    def sample_factors(self, rng):
        reg = regime(rng)
        tox = pick(rng, list(self.BANDS))
        situation = pick(rng, ["scheduled", "recheck"])
        current = pick_w(rng, {0: 4, -1: 3, -2: 2})
        max_delay = pick(rng, [14, 21, 28])
        recurrence = chance(rng, 0.35)
        cause_grade, days_held = None, None
        if situation == "scheduled":
            if reg == "clear":
                v = self._value(tox, rng, grade=pick(rng, [0, 1, 3]))
            elif reg == "borderline":
                v = self._value(tox, rng, edge_between=pick(rng, [2, 3]))
            else:
                v = self._value(tox, rng, grade=pick(rng, [2, 3, 4]))
        else:
            cause_grade = pick(rng, [2, 3])
            if reg == "clear":
                days_held = rng.randint(4, max_delay - 5)
                v = self._value(tox, rng, grade=pick(rng, [0, 1, 3]))
            elif reg == "borderline":
                days_held = pick(rng, [max_delay - 1, max_delay, max_delay + 1, rng.randint(5, max_delay - 3)])
                v = self._value(tox, rng, edge_between=2)
            else:
                days_held = rng.randint(6, max_delay + 4)
                v = self._value(tox, rng, grade=pick(rng, [1, 2]))
                current = pick(rng, [-2, -2, -1])
        uln = pick(rng, [33, 40, 45, 50])
        if tox == "alt_increase":
            alt_u = max(8, int(round(v * uln)))
            v = alt_u / uln
        grade = self._grade(tox, v)
        prev_grade = pick(rng, [0, 0, 1]) if situation == "scheduled" else cause_grade
        prev_v = self._value(tox, rng, grade=prev_grade)
        if tox == "alt_increase":
            prev_v = max(8, int(round(prev_v * uln))) / uln
        response = pick_w(rng, {"partial response": 3, "stable disease": 4, "stable disease, markers rising": 2,
                                "complete response": 1})
        if reg == "conflicting":
            plan = pick(rng, ["Give full dose today; tolerating everything else well and keen to continue.",
                              "Would stop the drug altogether; this has gone on long enough."])
        else:
            plan = pick(rng, [None, None, "Proposing to follow the table.", "Happy to dose if the table allows.",
                              "Would like to keep dose intensity up if at all possible."])
        today = rand_date(rng)
        drug = _drug(rng)
        site = _site_no(rng)
        return {
            "regime": reg, "toxicity": tox, "situation": situation, "value": v, "grade": grade, "uln": uln,
            "prev_value": prev_v, "cause_grade": cause_grade, "days_held": days_held, "max_delay": max_delay,
            "recurrence": recurrence, "current_level": current, "doses": pick(rng, self.DOSES),
            "response": response, "plan": plan, "cycle": rng.randint(3, 9), "today": today,
            "prev_date": today - _d(days_held if days_held is not None else pick(rng, [21, 28])),
            "unit": pick(rng, ["si", "si", "per_ul"]), "drug": drug, "protocol": _protocol(rng, drug),
            "subject": _subject(rng, site), "investigator": person(rng), "pharmacist": person(rng),
        }

    # ------------------------------------------------------------------ text pieces
    def _lab(self, f, v):
        t = f["toxicity"]
        if t == "alt_increase":
            return f"ALT {int(round(v * f['uln']))} U/L (ULN {f['uln']}; {v:.1f} x ULN)"
        name = "ANC" if t == "neutropenia" else "Platelets"
        if f["unit"] == "per_ul":
            return (f"{name} {int(round(v * 1000)):,}/µL (= {v:.2f} x10^9/L)" if t == "neutropenia"
                    else f"{name} {int(round(v * 1000)):,}/µL (= {int(v)} x10^9/L)")
        return f"{name} {v:.2f} x10^9/L" if t == "neutropenia" else f"{name} {int(v)} x10^9/L"

    def _levels(self, f):
        return {str(lv): f"{mg} mg" for lv, mg in zip((0, -1, -2), f["doses"])}

    def _rules(self, f):
        return [
            "Grade 1: continue at the current level.",
            "Grade 2: hold until grade 1 or better, then resume at the same level; for a recurrence resume one "
            "level lower.",
            "Grade 3: hold until grade 1 or better, then resume one level lower.",
            "Grade 4: discontinue. Haematological grade 4 that recovers within 7 days is managed as grade 3.",
            f"Held more than {f['max_delay']} consecutive days without recovery to grade 1 or better: discontinue.",
            "No re-escalation. A reduction below level -2 means discontinuation.",
        ]

    def _episode(self, f):
        if f["recurrence"]:
            return "an earlier episode of this toxicity was managed per the table"
        n = -f["current_level"]
        return "first episode" + (f"; {n} earlier reduction{'s' if n > 1 else ''} for a different toxicity" if n else "")

    def _context(self, f, rng):
        cur = f"{f['current_level']} ({self._levels(f)[str(f['current_level'])]})"
        if f["situation"] == "scheduled":
            return (f"Cycle {f['cycle']} day 1 due today; current dose level {cur}.",
                    self._episode(f))
        return (f"Recheck during a hold: drug held since {_fd(f['prev_date'], rng)} ({f['days_held']} days as of "
                f"today) for grade {f['cause_grade']} toxicity; current dose level {cur}.",
                self._episode(f))

    def render(self, f, rng, surface):
        ctx, episode = self._context(f, rng)
        today, prev = self._lab(f, f["value"]), self._lab(f, f["prev_value"])
        if surface == "json":
            return dumps(prune({
                "participant": {"subject": f["subject"], "protocol": f["protocol"], "best_response": f["response"],
                                "episode_history": episode},
                "visit": {"date": f["today"].isoformat(), "context": ctx, "labs_today": today,
                          "previous_result": {"date": f["prev_date"].isoformat(), "value": prev}},
                "dosing": {"current_level": f["current_level"], "levels": self._levels(f),
                           "hold_started": f["prev_date"].isoformat() if f["situation"] == "recheck" else None,
                           "days_held": f["days_held"]},
                "protocol_table": {"grading": self.GRADING[f["toxicity"]], "rules": self._rules(f)},
                "investigator_plan": f["plan"],
            }))
        if surface == "lab_flowsheet":
            rows = [[fmt_date(f["prev_date"], rng, "iso"), prev,
                     "hold started" if f["situation"] == "recheck" else f"C{f['cycle'] - 1}D1 dosed"],
                    [fmt_date(f["today"], rng, "iso"), today, "today"]]
            return "\n".join([
                f"FLOWSHEET -- {f['subject']} ({f['protocol']}, {f['drug']})", "",
                table(["Date", "Result", "Note"], rows, rng), "", ctx, f"History: {episode}.",
                f"Best response so far: {f['response']}.",
                "Dose levels: " + ", ".join(f"{k} = {v}" for k, v in self._levels(f).items()) + ".", "",
                "Protocol 8.2 grading: " + self.GRADING[f["toxicity"]], "Protocol 8.3 rules:",
                bullets(self._rules(f), rng)]
                + (["", f"Investigator: {f['plan']}"] if f["plan"] else []))
        s = [ctx, f"Today's result: {today}; previous {prev} on {_fd(f['prev_date'], rng)}.",
             f"History: {episode}.", f"Best response {f['response']}.", f["plan"] or ""]
        return "\n".join([
            f"To: {f['pharmacist']} (research pharmacy)", f"From: {f['investigator']}",
            f"Re: {f['subject']} -- dosing today?", "", noisy(join_sentences(s), rng, 0.12), "",
            "Levels: " + "; ".join(f"{k}: {v}" for k, v in self._levels(f).items()),
            "Table (protocol 8.2-8.3): " + self.GRADING[f["toxicity"]] + ". " + " ".join(self._rules(f))])

    def questions(self):
        return {
            "dosing_action": choice("What does the protocol's toxicity table call for today?", {
                "give_current_dose": "Give today's dose at the current level with no change.",
                "hold_and_recheck": "Withhold today's dose and repeat the assessment before trying again.",
                "resume_one_level_lower": "Give today's dose at one level below the current one.",
                "discontinue_drug": "Stop the study drug permanently under the protocol's discontinuation rules.",
                "consult_medical_monitor": "Call the sponsor's medical monitor, because the table does not settle "
                                           "this situation.",
            }),
            "toxicity_grade_today": score("What grade does today's toxicity reach under the protocol's grading?", [
                "Grade 0-1: normal or only mildly outside the normal range.",
                "Grade 2: moderately abnormal.",
                "Grade 3: severely abnormal.",
                "Grade 4: life-threatening.",
            ]),
            "continuation_case": score("How strong is the case for keeping this participant on the study drug?", [
                "Weak: the toxicity keeps returning and benefit is doubtful.",
                "Shaky: some benefit, but the toxicity or the reduction limit is closing in.",
                "Reasonable: benefit is visible and the toxicity looks manageable.",
                "Strong: clear benefit and plenty of room to manage the toxicity.",
            ]),
            "reduction_still_available": noul(
                "The protocol still allows this participant to be moved to a lower dose level."),
            "dosing_permitted_today": noul(
                "The participant may receive study drug today under the protocol's rules.",
                yes="Today's value is grade 1 or better on the protocol's grading, any hold has lasted no longer than the "
                    "delay limit, and if the rules call for a reduction a permitted lower level still exists.",
                no="Today's value is grade 2 or worse, the hold has run past the delay limit, or the rules call for "
                   "discontinuation or a reduction below level -2."),
        }


# =========================================================================== schema 6
class SdvFinding(Schema):
    name = "sdv_finding"
    description = "A source-data-verification discrepancy found at a monitoring visit, to be graded and followed up."
    surfaces = {"json": 0.45, "monitoring_report": 0.3, "edc_query_thread": 0.25}

    POINTS = ["target_lesions", "ae_onset", "conmed", "vitals", "eligibility_lab", "dose_given"]
    SOURCE_Q = {
        "clean": "signed and dated at the time by the person who made the entry",
        "late_explained": "late entry, initialled and dated {n} days after the visit, reason given: {why}",
        "overwritten": "original figure overwritten; the correction has no initials, date or reason",
        "edited_later": "audit trail shows the source entry was changed {n} days after the visit by a user not on "
                        "the delegation log",
        "missing": "source document could not be located during the visit",
    }
    RESP = {
        "agrees": "agrees the EDC entry is wrong and will correct it",
        "disputes": "says the EDC value is right and the source is in error",
        "no_response": "has not answered the query after {n} days",
        "undelegated_fix": "corrected the EDC entry, but the correction came from a user not on the delegation log",
    }

    REPLY = {
        "agrees": "Agreed, the EDC entry is wrong. Will correct.",
        "disputes": "We believe the EDC value is correct and the source is in error.",
        "undelegated_fix": "Corrected in EDC. [system: change made under an account not on the site delegation log]",
    }

    def _point(self, kind, flip, rng, visit):
        """(data point label, EDC value, source value, source document, protocol reference)."""
        if kind == "target_lesions":
            b = rng.randint(40, 140)
            if flip:
                lo = int(b * 0.7)
                pr, sd = lo - pick(rng, [0, 1]), lo + pick(rng, [2, 3])
                edc, src = (pr, sd) if chance(rng, 0.6) else (sd, pr)
            else:
                edc = int(b * rng.uniform(0.45, 0.6))
                src = edc + pick(rng, [-3, -2, 2, 3])
            return (f"Sum of target-lesion diameters, week {pick(rng, [8, 16, 24])}", f"{edc} mm", f"{src} mm",
                    "radiology tumour-measurement worksheet",
                    f"Baseline sum {b} mm; protocol: a fall of 30% or more from baseline is a partial response")
        if kind == "ae_onset":
            first = visit - _d(rng.randint(60, 240))
            if flip:
                before, after = first - _d(pick(rng, [1, 2, 3])), first + _d(pick(rng, [1, 2, 4]))
                edc, src = (before, after) if chance(rng, 0.5) else (after, before)
            else:
                edc = first + _d(rng.randint(10, 40))
                src = edc + _d(pick(rng, [-5, -3, 3, 6]))
            term = pick(rng, ["peripheral oedema", "headache", "elevated blood pressure", "nausea"])
            return (f"Onset date of adverse event '{term}'", edc.isoformat(), src.isoformat(), "clinic visit note",
                    f"First dose {first.isoformat()}; events starting on or after the first dose count as "
                    "treatment-emergent")
        if kind == "conmed":
            med = ("an azole antifungal (a strong CYP3A inhibitor)" if flip
                   else pick(rng, ["a topical emollient", "an over-the-counter antacid"]))
            return ("Concomitant medications", "not recorded", f"{med}, started during cycle {rng.randint(2, 6)}",
                    "clinic visit note", "Protocol 6.4: strong CYP3A inhibitors are prohibited while on study drug")
        if kind == "vitals":
            if flip:
                src = pick(rng, [161, 162, 163, 164, 165])
                edc = int(str(src)[0] + str(src)[2] + str(src)[1])      # last two digits transposed
            else:
                src = rng.randint(118, 146)
                edc = src + pick(rng, [-6, -4, 4, 6])
            return ("Pre-dose systolic blood pressure", f"{edc} mmHg", f"{src} mmHg", "nursing vital-signs sheet",
                    "Protocol 8.5: withhold the dose if systolic pressure is 160 mmHg or higher")
        if kind == "eligibility_lab":
            edc = round(rng.uniform(1.0, 1.3), 2)
            src = (int(round(rng.uniform(1.49, 1.66) * 88.4)) if flip
                   else int(round(edc * 88.4)) + pick(rng, [-3, 3]))
            return ("Screening serum creatinine", f"{edc:.2f} mg/dL", f"{src} µmol/L", "local laboratory report",
                    "Inclusion 4.1(f): serum creatinine no higher than 1.5 mg/dL")
        levels = pick(rng, [(200, 150, 100), (400, 300, 200), (60, 40, 20)])
        i = pick(rng, [0, 1])
        if flip:
            edc, src = f"{levels[i]} mg daily, cycle 4", f"{levels[i + 1]} mg daily, cycle 4"
        else:
            edc, src = f"{levels[i]} mg daily x 14 days", f"{levels[i]} mg daily x 13 days (one missed dose in diary)"
        return ("Study drug dose taken", edc, src, "pharmacy dispensing log and subject diary",
                "Dose levels " + " / ".join(f"{x} mg" for x in levels) + "; every reduction must be recorded in EDC")

    def sample_factors(self, rng):
        reg = regime(rng)
        kind = pick(rng, self.POINTS)
        if reg == "clear":
            good = chance(rng, 0.5)
            flip = not good
            src_q = "clean" if good else pick(rng, ["overwritten", "edited_later"])
            rate = rng.uniform(0.5, 1.5) if good else rng.uniform(9, 16)
            prior = 0 if good else rng.randint(2, 4)
            resp = "agrees" if good else pick(rng, ["disputes", "no_response"])
        elif reg == "borderline":
            flip = chance(rng, 0.5)
            src_q = pick_w(rng, {"clean": 3, "late_explained": 3, "missing": 1})
            rate, prior = rng.uniform(3, 7), pick(rng, [0, 1, 1, 2])
            resp = pick(rng, list(self.RESP))
        else:
            if chance(rng, 0.5):     # tidy site, but the one bad point is a key value with a doctored source
                flip, src_q, rate, prior = True, pick(rng, ["overwritten", "edited_later"]), rng.uniform(0.5, 2), 0
                resp = pick(rng, ["agrees", "disputes"])
            else:                    # sloppy site, trivial point, clean source
                flip, src_q, rate, prior = False, "clean", rng.uniform(8, 15), rng.randint(3, 5)
                resp = pick(rng, ["agrees", "undelegated_fix"])
        visit = rand_date(rng)
        label, edc, src, doc, ref = self._point(kind, flip, rng, visit)
        fields = rng.randint(180, 620)
        drug = _drug(rng)
        site = _site_no(rng)
        return {
            "regime": reg, "point": kind, "category_flip": flip, "source_quality": src_q, "error_rate_pct": rate,
            "prior_same_type": prior, "site_response": resp, "label": label, "edc_value": edc,
            "source_value": None if src_q == "missing" else src, "source_doc": doc, "protocol_ref": ref,
            "fields_verified": fields, "discrepancies": max(1, int(round(fields * rate / 100))),
            "late_days": rng.randint(9, 30),
            "late_reason": pick(rng, ["copied from an outside report", "first written on a worksheet that was later "
                                      "transcribed", "entered after an EHR outage"]), "query_days_open": rng.randint(6, 40), "visit_date": visit,
            "visit_no": rng.randint(3, 14), "cra": person(rng), "coordinator": person(rng),
            "protocol": _protocol(rng, drug), "site": site, "subject": _subject(rng, site),
            "query_id": ident(rng, "Q", 6, "-"),
            # the same source page carries a later, signed entry that matches the EDC value (a corrected reading)
            "source_has_matching_entry": src_q in ("clean", "late_explained") and chance(rng, 0.35),
        }

    def _match_note(self, f):
        if not f["source_has_matching_entry"]:
            return None
        return (f"the same {f['source_doc']} also has a later entry, signed and dated the same day, recording "
                f"{f['edc_value']} as a corrected reading")

    def _src_cond(self, f):
        return self.SOURCE_Q[f["source_quality"]].format(n=f["late_days"], why=f["late_reason"])

    def _resp(self, f):
        return self.RESP[f["site_response"]].format(n=f["query_days_open"])

    def render(self, f, rng, surface):
        if surface == "json":
            return dumps(prune({
                "visit": {"type": "interim monitoring", "number": f["visit_no"], "date": f["visit_date"].isoformat(),
                          "monitor": f["cra"], "site": f["site"], "protocol": f["protocol"]},
                "finding": {"subject": f["subject"], "data_point": f["label"], "edc_value": f["edc_value"],
                            "source_value": f["source_value"], "source_document": f["source_doc"],
                            "source_condition": self._src_cond(f), "monitor_note": self._match_note(f),
                            "protocol_reference": f["protocol_ref"]},
                "site_metrics": {"fields_verified_this_visit": f["fields_verified"],
                                 "discrepancies_this_visit": f["discrepancies"],
                                 "same_type_findings_earlier_visits": f["prior_same_type"]},
                "follow_up": {"query": f["query_id"], "site_response": self._resp(f)},
            }))
        if surface == "monitoring_report":
            rows = [("Subject", f["subject"]), ("Data point", f["label"]), ("EDC", f["edc_value"]),
                    ("Source", f["source_value"] or "not located"), ("Source document", f["source_doc"]),
                    ("Condition of source", self._src_cond(f)), ("Monitor note", self._match_note(f)),
                    ("Protocol reference", f["protocol_ref"])]
            return "\n".join([
                f"INTERIM MONITORING VISIT {f['visit_no']} -- {f['protocol']} site {f['site']} -- "
                f"{_fd(f['visit_date'], rng)}", f"Monitor: {f['cra']}", "",
                "4. Source data verification",
                f"{f['fields_verified']} fields verified; {f['discrepancies']} discrepancies. Earlier-visit findings of "
                f"this type: {f['prior_same_type']}.", "", "Finding 4.3:", _kv(rows, rng), "",
                f"Site response ({f['coordinator']}): {self._resp(f)}."])
        opened = f["visit_date"]
        msgs = [(opened, f["cra"], noisy(f"{f['label']} for {f['subject']}: EDC shows {f['edc_value']}, "
                                         f"{f['source_doc']} shows {f['source_value'] or 'nothing (not located)'}. "
                                         f"Source: {self._src_cond(f)}."
                                         + (f" Note: {self._match_note(f)}." if self._match_note(f) else "")
                                         + " Please review.", rng, 0.1))]
        if f["site_response"] != "no_response":
            msgs.append((opened + _d(rng.randint(2, 9)), f["coordinator"], self.REPLY[f["site_response"]]))
        lines = [f"EDC query {f['query_id']} -- {f['protocol']} / site {f['site']} -- status: "
                 + ("open" if f["site_response"] in ("disputes", "no_response") else "answered"),
                 f"Reference: {f['protocol_ref']}", ""]
        lines += [f"[{fmt_date(d, rng, 'iso')}] {who}: {text}" for d, who, text in msgs]
        if f["site_response"] == "no_response":
            lines.append(f"(no site reply; open {f['query_days_open']} days)")
        lines += ["", f"Visit {f['visit_no']} SDV totals: {f['discrepancies']} of {f['fields_verified']} fields "
                      f"discrepant; earlier-visit findings of this type: {f['prior_same_type']}."]
        return "\n".join(lines)

    def questions(self):
        return {
            "finding_grade": choice("How should the monitor grade this finding in the visit report?", {
                "observation_only": "An observation worth noting, with no action required from the site.",
                "minor_finding": "A minor finding: an isolated error with no effect on safety or key data.",
                "major_finding": "A major finding: an error that could affect key data or participant safety, or a "
                                 "repeating pattern.",
                "critical_finding": "A critical finding: data integrity or participant safety is seriously "
                                    "compromised, possibly on purpose.",
            }),
            "data_point_next_step": choice("What should happen next with the data point in question?", {
                "mark_verified": "Mark the value verified, since the source supports it once the context is read.",
                "query_for_correction": "Raise or keep a data query open until the site corrects the entry against "
                                        "the source.",
                "investigator_clarification": "Ask the investigator for a signed, dated note saying what the source "
                                              "should read.",
                "integrity_referral": "Refer the data point to the sponsor's quality team for a data-integrity review.",
            }),
            "error_pattern": score("How systematic do the site's data errors look?", [
                "Isolated: a one-off slip.",
                "Occasional: a few scattered errors with no common cause.",
                "Recurring: the same kind of error keeps turning up.",
                "Pervasive: errors run across subjects and data types.",
            ]),
            "analysis_impact": score("How much could this discrepancy change a safety or efficacy conclusion for the participant?", [
                "None: the value feeds no key analysis.",
                "Minor: it shifts a secondary value without changing any category.",
                "Moderate: it touches a key value but probably not its category.",
                "Major: it could flip a response category, a dosing or safety call, or eligibility.",
            ]),
            "source_record_reliable": noul(
                "The source record for this data point can be trusted as an original, attributable entry made at "
                "the time.",
                yes="The entry is signed or traceable to its author, was made at the time, and any change is dated "
                    "and explained.",
                no="The source is missing, was changed without a trail, or cannot be tied to who wrote it or when."),
        }


WORKFLOW = Workflow(
    name="clinical_trial_screening",
    title="Clinical trial screening and conduct",
    description="Operational decisions at a trial site or sponsor: eligibility, safety reporting, deviations, site "
                "selection, dose modification and monitoring findings.",
    domain="clinical research operations",
    schemas=[EligibilityPrescreen(), AdverseEventAssessment(), ProtocolDeviation(), SiteFeasibility(),
             DoseModification(), SdvFinding()],
    notes="Fictional investigational products, protocols, sponsors and sites; prior and concomitant treatments by "
          "drug class only. No patient-facing conversation, no security alerts, no vendor bills, no AI-agent runs. "
          "Age appears only as a protocol range the candidate sits well inside.",
)
