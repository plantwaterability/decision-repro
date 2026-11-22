"""Prior authorization review: does a health plan cover a requested service or drug under its written
medical-necessity policy, and how should the utilisation-management (UM) team respond?

Everything here is fictional: the plans, the policy numbers, the drug names, the clinicians and the members.
The policy criteria rendered into each state are invented plan rules written for this generator (thresholds,
step-therapy lists, access standards). They are not clinical guidance and do not describe any real payer.

Each schema draws a `regime` first ('clear' / 'borderline' / 'conflicting'). Borderline cases sit near a
written threshold (5.5 weeks of therapy against a 6-week rule, an X-ray 94 days old against a 90-day rule);
conflicting cases carry an order that claims a warning sign the exam does not show, a chart that disagrees
with the pharmacy claims, or an attending's opinion that the objective criteria do not support.
"""
from __future__ import annotations

from datetime import timedelta

from .base import (
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
    money,
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

PLAN_STEMS = ["Corvale", "Brightwater", "Tamsford", "Oakhollow", "Linden Bay", "Marrowfield", "Quillon",
              "Ravensgate", "Sablewood", "Thistledown", "Wrenfield", "Ashcombe"]
PLAN_KINDS = ["Health Plan", "Care Plan", "Mutual Health", "Health Cooperative"]
PRODUCTS = ["commercial PPO", "commercial HMO", "individual exchange plan", "managed Medicaid",
            "employer self-funded plan"]


def _plan(rng):
    return f"{pick(rng, PLAN_STEMS)} {pick(rng, PLAN_KINDS)}"


def _doctor(rng):
    return "Dr. " + person(rng)


def _clinician(rng, specialty):
    """A name with the right credential for the ordering specialty."""
    if "nurse practitioner" in specialty:
        return person(rng) + ", NP"
    if "physician assistant" in specialty:
        return person(rng) + ", PA-C"
    return "Dr. " + person(rng)


def _mo(n):
    return "1 month" if n == 1 else f"{n} months"


def _wk(x):
    """5.5 -> '5.5 weeks', 1 -> '1 week'."""
    return "1 week" if x == 1 else f"{x:g} weeks"


def _half(x):
    return max(0.0, round(x * 2) / 2)


# =========================================================================== schema 1: outpatient imaging
STUDIES = {
    "lumbar_mri": dict(
        study="MRI lumbar spine without contrast", policy="RAD-112", weeks=6, region="lumbar spine",
        complaint=["low back pain radiating into the left leg", "low back pain with right-sided sciatica",
                   "low back pain since lifting a heavy box at work"],
        care="physical therapy, a supervised home exercise program or scheduled anti-inflammatory medication",
        flags=["new bowel or bladder dysfunction", "progressive leg weakness", "known cancer with new back pain",
               "fever with suspected spinal infection"],
        soft_flags=["intermittent numbness in the foot", "a remote history of skin cancer, excised",
                    "pain that wakes the member at night"],
        exam_ok="Strength 5/5 in both legs, reflexes symmetric, no saddle numbness.",
        exam_flag="Left ankle dorsiflexion 3/5, down from 4/5 two weeks ago; diminished left ankle reflex.",
        change="new numbness along the outer edge of the left foot on exam",
        specialties=["family medicine", "internal medicine", "orthopaedic spine", "physical medicine and rehabilitation",
                     "urgent care nurse practitioner"],
        purpose=["to evaluate for disc herniation before an epidural injection", "to plan a surgical referral",
                 "to rule out nerve compression"], xray=False,
        kinds={"physical therapy": 4, "home exercise program": 2, "anti-inflammatory medication": 3, "chiropractic care": 1}),
    "cervical_mri": dict(
        study="MRI cervical spine without contrast", policy="RAD-114", weeks=6, region="cervical spine",
        complaint=["neck pain radiating into the right arm", "neck pain with tingling in both hands"],
        care="physical therapy, a supervised home exercise program or scheduled anti-inflammatory medication",
        flags=["progressive arm weakness", "unsteady gait suggesting cord compression", "known cancer with new neck pain"],
        soft_flags=["hand clumsiness the member puts down to the cold", "occasional dizziness when looking up"],
        exam_ok="Grip and arm strength normal, reflexes 2+ and symmetric, gait steady.",
        exam_flag="Right triceps 4-/5, new since the last visit; Hoffmann sign positive on the right.",
        change="reduced sensation over the right thumb, not present at the time of that study",
        specialties=["family medicine", "internal medicine", "neurology", "orthopaedic spine"],
        purpose=["to look for a compressed nerve root", "before referral to spine surgery"], xray=False,
        kinds={"physical therapy": 4, "home exercise program": 2, "anti-inflammatory medication": 3, "chiropractic care": 1}),
    "knee_mri": dict(
        study="MRI left knee without contrast", policy="RAD-140", weeks=4, region="knee",
        complaint=["left knee pain with catching after a twisting injury", "left knee swelling and giving way on stairs"],
        care="physical therapy or activity modification with a brace",
        flags=["a locked knee that cannot be straightened", "gross instability after an acute injury",
               "suspected joint infection"],
        soft_flags=["a sensation of the knee catching now and then", "swelling after long walks"],
        exam_ok="Full range of motion, no effusion, ligaments stable to stress testing.",
        exam_flag="Lacks the last 20 degrees of extension and cannot be passively straightened; moderate effusion.",
        change="a new twisting injury three weeks ago with joint-line tenderness",
        specialties=["family medicine", "sports medicine", "orthopaedic surgery", "urgent care physician assistant"],
        purpose=["to assess for a meniscal tear", "to plan possible arthroscopy"], xray=True,
        kinds={"physical therapy": 4, "a brace with activity modification": 3, "anti-inflammatory medication": 2,
               "home exercise program": 1}),
    "shoulder_mri": dict(
        study="MRI right shoulder without contrast", policy="RAD-146", weeks=6, region="shoulder",
        complaint=["right shoulder pain and weakness lifting overhead", "right shoulder pain after a fall on an outstretched arm"],
        care="physical therapy or a home exercise program, with or without a steroid injection",
        flags=["inability to raise the arm after an acute injury", "suspected infection or tumour"],
        soft_flags=["weakness that comes and goes", "pain lying on the right side at night"],
        exam_ok="Rotator cuff strength 5/5; pain at the end of range only.",
        exam_flag="Cannot actively raise the arm above 60 degrees; positive drop-arm test since the fall.",
        change="a fall onto the shoulder since then, with new weakness in external rotation",
        specialties=["family medicine", "sports medicine", "orthopaedic surgery"],
        purpose=["to assess the rotator cuff", "for surgical planning"], xray=True,
        kinds={"physical therapy": 4, "home exercise program": 2, "a steroid injection, then home exercises": 2, "anti-inflammatory medication": 1}),
    "sinus_ct": dict(
        study="CT sinuses without contrast", policy="RAD-171", weeks=4, region="sinuses",
        complaint=["nasal congestion and facial pressure", "repeated sinus infections with facial pain"],
        care="a daily nasal steroid spray with saline rinses, or a completed antibiotic course",
        flags=["swelling around the eye or a change in vision", "a suspected tumour",
               "signs that infection has spread beyond the sinuses"],
        soft_flags=["occasional blurred vision the member links to screen use", "one nosebleed last month"],
        exam_ok="Mild turbinate swelling; eye movements full, no facial swelling.",
        exam_flag="Swelling of the right upper eyelid with pain on eye movement.",
        change="two further infections treated with antibiotics",
        specialties=["family medicine", "otolaryngology", "urgent care nurse practitioner"],
        purpose=["to evaluate for chronic sinus disease before referral", "to plan sinus surgery"], xray=False,
        kinds={"nasal steroid spray with saline rinses": 4, "an antibiotic course": 3, "saline rinses alone": 2,
               "antihistamine tablets": 1}),
}


def _imaging_policy(s):
    t = (f"{s['policy']}: covered after at least {s['weeks']} weeks of documented conservative care "
         f"({s['care']}), OR sooner when the examination documents a warning sign: {'; '.join(s['flags'])}. "
         f"Repeat imaging of the {s['region']} within 12 months requires a documented change in condition.")
    if s["xray"]:
        t += " Plain X-rays of the area taken within the previous 90 days are also required."
    return t


class AdvancedImagingRequest(Schema):
    name = "advanced_imaging_request"
    description = "An outpatient MRI or CT prior-authorization request checked against the plan's written imaging policy."
    surfaces = {"json": 0.5, "intake_form": 0.25, "fax_packet": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        key = pick(rng, list(STUDIES))
        s = STUDIES[key]
        thr = s["weeks"]
        cons_weeks = _half(between(rng, 0, thr + 6))
        cons_kind = pick_w(rng, s["kinds"])
        first_kind = list(s["kinds"])[0]
        cons_doc = pick_w(rng, {"treatment_records": 3, "office_note": 4, "member_report": 2})
        pt_visits = None
        flag, flag_support = None, None
        prior_months = pick(rng, [None, None, None, None, 5, 9, 14, 30])
        change = None
        xray_days = pick(rng, [20, 45, 80, 130, 200, None]) if s["xray"] else None
        expedite = chance(rng, 0.2)
        exam_documented = chance(rng, 0.88)
        variant = None
        if reg == "clear":
            if chance(rng, 0.5):                      # meets the policy cleanly
                if chance(rng, 0.65):
                    cons_weeks, cons_kind, cons_doc = float(thr + rng.randint(2, 8)), first_kind, "treatment_records"
                    pt_visits = (rng.randint(8, 12), 12) if cons_kind == "physical therapy" else None
                else:
                    flag, flag_support, expedite = pick(rng, s["flags"]), "exam", True
                    cons_weeks = _half(between(rng, 0, max(1, thr - 3)))
                prior_months, exam_documented = None, True
                xray_days = rng.randint(5, 40) if s["xray"] else None
            else:                                     # plainly premature
                cons_weeks = pick(rng, [0.0, 1.0, 1.5, 2.0])
                cons_doc = pick(rng, ["member_report", "office_note"])
                prior_months = pick(rng, [None, 4, 6])
                xray_days = pick(rng, [None, 240]) if s["xray"] else None
                expedite = False
        elif reg == "borderline":
            cons_weeks = _half(near(rng, thr, 1.5, lo=0.5))
            cons_doc = pick_w(rng, {"office_note": 4, "member_report": 3, "treatment_records": 2})
            if s["xray"]:
                xray_days = near(rng, 90, 18, lo=10, integer=True)
            if chance(rng, 0.35):
                flag, flag_support = pick(rng, s["soft_flags"]), pick(rng, ["order_only", "exam"])
            if prior_months is not None:
                prior_months = near(rng, 12, 2, lo=6, integer=True)
        else:
            variant = pick(rng, ["flag_unsupported", "therapy_attendance", "repeat_claimed_change", "urgent_no_basis"])
            if variant == "flag_unsupported":
                flag, flag_support, expedite = pick(rng, s["flags"]), "order_only", True
                cons_weeks = _half(between(rng, 0, thr - 1))
            elif variant == "therapy_attendance":
                cons_kind, cons_doc = first_kind, "treatment_records"
                cons_weeks = float(thr + rng.randint(1, 5))
                if cons_kind == "physical therapy":
                    pt_visits = (rng.randint(2, 4), pick(rng, [8, 10, 12]))
                else:
                    pt_visits = None   # the record shows gaps in use instead (see _note)
            elif variant == "repeat_claimed_change":
                prior_months = rng.randint(4, 10)
                cons_weeks = float(thr + rng.randint(0, 6))
            else:
                expedite = True
                cons_weeks = _half(near(rng, thr, 2, lo=0.5))
            if chance(rng, 0.4):
                exam_documented = variant == "flag_unsupported" or chance(rng, 0.5)
        if cons_kind == "physical therapy" and pt_visits is None and cons_weeks >= 2 and chance(rng, 0.5):
            planned = pick(rng, [8, 10, 12])
            pt_visits = (max(1, min(planned, int(cons_weeks * 1.5 * rng.uniform(0.5, 1.0)))), planned)
        if cons_weeks == 0:
            cons_kind, cons_doc, pt_visits = None, None, None
        if prior_months is not None:
            if variant == "repeat_claimed_change":
                change = "asserted"
            else:
                change = pick_w(rng, {"documented": 2, "asserted": 2, "none": 2})
        if flag and flag_support is None:
            flag_support = "order_only"
        sym_weeks = max(cons_weeks + rng.randint(0, 6), 1.0)
        spec = pick(rng, s["specialties"])
        received = rand_date(rng)
        return {
            "regime": reg, "variant": variant, "study_key": key, "study": s["study"], "policy_id": s["policy"],
            "policy_weeks": thr, "complaint": pick(rng, s["complaint"]), "symptom_weeks": sym_weeks,
            "conservative_weeks": cons_weeks, "conservative_kind": cons_kind, "conservative_documentation": cons_doc,
            "pt_visits": pt_visits, "warning_sign_claimed": flag, "warning_sign_support": flag_support,
            "exam_documented": exam_documented, "prior_imaging_months_ago": prior_months,
            "change_since_prior": change, "xray_days_ago": xray_days, "expedite_requested": expedite,
            "appointment_in_days": pick(rng, [2, 4, 7, 10, 16, 25]), "received": received,
            "ordering_specialty": spec, "ordering_provider": _clinician(rng, spec),
            "member_age": rng.randint(24, 78), "plan": _plan(rng), "product": pick(rng, PRODUCTS),
            "member_id": ident(rng, "M", 9, ""), "request_id": ident(rng, "PA", 7, "-"),
            "purpose": pick(rng, s["purpose"]),
            "expedite_reason": (pick(rng, ["severe pain, unable to work", "worsening symptoms", "patient anxious",
                                           "provider requests urgent review"]) if expedite else None),
        }

    def _note(self, f, rng):
        s = STUDIES[f["study_key"]]
        p = [pick(rng, [f"{f['member_age']}-year-old, {_wk(f['symptom_weeks'])} of {f['complaint']}.",
                        f"{f['member_age']} y/o with {f['complaint']}, onset about {_wk(f['symptom_weeks'])} ago."])]
        w, kind = f["conservative_weeks"], f["conservative_kind"]
        if not kind:
            p.append(pick(rng, ["No formal treatment tried yet; wants imaging first.",
                                "Has not started therapy.", "Conservative care not yet started."]))
        elif f["pt_visits"]:
            a, n = f["pt_visits"]
            p.append(f"Attended {a} of {n} scheduled physical therapy visits over {_wk(w)}"
                     + (", discharged to a home program." if a >= n * 0.7
                        else pick(rng, ["; stopped going, says it was not helping.",
                                        "; missed sessions because of work.", "."])))
        else:
            src = {"treatment_records": "", "office_note": "", "member_report": pick(rng, [" (per patient)",
                                                                                         " (self-reported)"])}[f["conservative_documentation"]]
            p.append(f"{kind[0].upper() + kind[1:]} for roughly {_wk(w)}{src}.")
            if f["variant"] == "therapy_attendance":
                p.append(pick(rng, ["Records show long gaps; used it on and off.",
                                    "Fill history shows only two refills over that period.",
                                    "Admits to using it only when the pain flares."]))
        if f["warning_sign_claimed"] in s["flags"]:
            p.append(pick(rng, [f"Concern for {f['warning_sign_claimed']}.",
                                f"Order flags a warning sign: {f['warning_sign_claimed']}."]))
        elif f["warning_sign_claimed"]:
            p.append(f"Also reports {f['warning_sign_claimed']}.")
        if f["exam_documented"]:
            p.append("Exam: " + (s["exam_flag"] if f["warning_sign_support"] == "exam" and
                                 f["warning_sign_claimed"] in s["flags"] else s["exam_ok"]))
        if f["prior_imaging_months_ago"] is not None:
            p.append(f"Had imaging of the {s['region']} {f['prior_imaging_months_ago']} months ago.")
            p.append({"documented": f"Since then: {s['change']}.",
                      "asserted": pick(rng, ["Symptoms are worse than at the time of that study.",
                                             "Patient feels things have progressed."]),
                      "none": "Symptoms about the same as then."}[f["change_since_prior"]])
        p.append(f"Ordering {f['study']} {f['purpose']}.")
        return noisy(join_sentences(p), rng, 0.15)

    def render(self, f, rng, surface):
        s = STUDIES[f["study_key"]]
        note = self._note(f, rng)
        policy = _imaging_policy(s)
        appt = f["received"] + timedelta(days=f["appointment_in_days"])
        xray = (None if f["xray_days_ago"] is None else f"{f['xray_days_ago']} days before the request")
        if surface == "json":
            return dumps(prune({
                "request": {"id": f["request_id"], "received": f["received"].isoformat(), "service": f["study"],
                            "priority_requested": "expedited" if f["expedite_requested"] else "standard",
                            "expedite_reason": f["expedite_reason"], "scheduled_date": appt.isoformat(),
                            "ordering_provider": f["ordering_provider"], "ordering_specialty": f["ordering_specialty"]},
                "member": {"id": f["member_id"], "plan": f["plan"], "product": f["product"]},
                "clinical": {"office_note": note, "warning_sign_on_order": f["warning_sign_claimed"],
                             "plain_xray": xray if s["xray"] else None,
                             "prior_imaging_report_attached": (False if f["prior_imaging_months_ago"] is not None
                                                               and chance(rng, 0.5) else None)},
                "policy": {"id": s["policy"], "criteria": policy},
            }))
        if surface == "intake_form":
            rows = [("Request #", f["request_id"]), ("Received", fmt_date(f["received"], rng)),
                    ("Member", f"{f['member_id']} / {f['plan']} ({f['product']})"),
                    ("Service", f["study"]), ("Ordering provider", f"{f['ordering_provider']}, {f['ordering_specialty']}"),
                    ("Date of service", fmt_date(appt, rng)),
                    ("Priority", "EXPEDITED" if f["expedite_requested"] else "standard"),
                    ("Reason for expedite", f["expedite_reason"]),
                    ("Conservative therapy", f"{f['conservative_kind']}, {_wk(f['conservative_weeks'])}"
                     if f["conservative_kind"] else "none"),
                    ("X-ray within 90 days", show(xray, rng) if s["xray"] else None),
                    ("Warning signs noted on order", f["warning_sign_claimed"] or "none")]
            return ("PRIOR AUTHORIZATION REQUEST -- OUTPATIENT IMAGING\n" + kv_lines(rows, rng)
                    + f"\n\nClinical notes (as submitted):\n{note}\n\nApplicable policy {policy}")
        # fax_packet
        pages = rng.randint(3, 9)
        attach = ["office visit note"]
        if f["conservative_documentation"] == "treatment_records":
            attach.append({"physical therapy": "physical therapy notes", "chiropractic care": "chiropractic notes",
                           "home exercise program": "home exercise log"}.get(f["conservative_kind"], "pharmacy fill history"))
        if s["xray"] and f["xray_days_ago"] is not None:
            attach.append(f"X-ray report ({f['xray_days_ago']} days old)")
        lines = [f"FAX  {fmt_ts(rand_ts(rng, f['received'], (8, 17)), rng)}   Pages: {pages}",
                 f"To: {f['plan']} Utilization Management   From: {f['ordering_provider']} ({f['ordering_specialty']})",
                 f"RE: prior auth {f['study']}, member {f['member_id']}"
                 + (" -- URGENT" if f["expedite_requested"] else ""), "",
                 "Enclosed: " + ", ".join(attach) + ".", "",
                 "OFFICE NOTE (excerpt)", note, "",
                 f"-- UM intake stamp: {f['request_id']}, scheduled {fmt_date(appt, rng)}; policy on file {policy}"]
        return "\n".join(lines)

    def questions(self):
        return {
            "imaging_determination": choice("What should the utilization-management team decide on this imaging request?", {
                "approve": "Approve: a route through the policy is documented (the required weeks of conservative "
                           "care, followed consistently and not just self-reported, or a listed warning sign backed "
                           "by exam findings) and the X-ray and repeat-imaging requirements are met.",
                "deny_not_necessary": "Deny: conservative care is more than 2 weeks short of the minimum with no "
                                      "warning sign named, or it repeats imaging from the last 12 months with no "
                                      "change since.",
                "offer_peer_to_peer": "Offer a call with a plan physician: conservative care is within 2 weeks of the "
                                      "minimum, a warning sign is named without supporting exam findings, or therapy "
                                      "was attended under 70% of scheduled visits or with long gaps.",
                "pend_for_records": "Pend for specific records: a route is met on what is described, but treatment "
                                    "records are missing for self-reported care, a plain X-ray within 90 days is "
                                    "missing, or a change since imaging in the last 12 months is only asserted.",
            }),
            "criteria_alignment": score("How closely does the documented clinical picture match the plan's imaging policy?", [
                "Clearly outside: no route through the policy is documented.",
                "Short: one required element is missing or falls below the stated minimum.",
                "Arguable: the elements are nearly all there, or met only on a generous reading.",
                "Met: every required element, or a qualifying warning sign, is documented.",
            ]),
            "decisive_gap": choice("Which gap in the submitted record matters most for this decision?", {
                "conservative_care_proof": "Proof of how long, and how consistently, conservative treatment was actually followed.",
                "exam_findings": "Examination findings that would confirm or rule out the warning sign the order relies on.",
                "nothing_material": "Nothing material; the record is complete enough to decide either way.",
                "prior_study_details": "The report of the earlier imaging, or evidence of what has changed since it.",
                "xray_currency": "A plain X-ray report recent enough to satisfy the policy.",
            }),
            "decision_clock": score("How quickly does the plan need to issue its decision?", [
                "Standard: the plan's ordinary 15-calendar-day window is acceptable.",
                "Prompt: the scan is booked within days, so an early standard decision avoids a reschedule.",
                "Expedited: waiting could seriously jeopardise the member's function, so the 72-hour track applies.",
                "Same day: the record describes a deficit that may be progressing now.",
            ]),
            "warning_sign_documented": noul(
                "The record documents a warning sign that the policy accepts in place of conservative care.",
                yes="A sign listed in the policy is supported by examination findings in the notes.",
                no="No listed sign is present, or it is named on the order without supporting findings."),
        }


# =========================================================================== schema 2: specialty drug / step therapy
DRUG_AREAS = {
    "plaque_psoriasis": dict(condition="moderate-to-severe plaque psoriasis", drug="velmatrizumab", policy="RX-4471",
                             steps=["corvanex", "tildaprine"], step_weeks=12, measure="body surface area affected",
                             unit="%", thr=10, specialty="dermatology",
                             label="two loading doses, then every 8 weeks", high="every 4 weeks"),
    "rheumatoid_arthritis": dict(condition="rheumatoid arthritis", drug="ostralimab", policy="RX-4520",
                                 steps=["nemtrexate", "halcovine"], step_weeks=12, measure="swollen joint count",
                                 unit=" joints", thr=6, specialty="rheumatology",
                                 label="every 2 weeks", high="weekly"),
    "chronic_migraine": dict(condition="chronic migraine (prevention)", drug="pendrigant", policy="RX-4602",
                             steps=["bravolol", "tomirazine"], step_weeks=8, measure="headache days per month",
                             unit=" days", thr=8, specialty="neurology or headache medicine",
                             label="once monthly", high="every 2 weeks"),
    "severe_asthma": dict(condition="severe persistent asthma", drug="exarolimab", policy="RX-4655",
                          steps=["high-dose fluvanide inhaler", "lomiterol add-on"], step_weeks=12,
                          measure="steroid-requiring flare-ups in the past 12 months", unit="", thr=2,
                          specialty="pulmonology or allergy", label="every 4 weeks", high="every 2 weeks"),
    "atopic_dermatitis": dict(condition="moderate-to-severe atopic dermatitis", drug="daprolimab", policy="RX-4710",
                              steps=["cresolone ointment", "tacravex cream"], step_weeks=8,
                              measure="body surface area affected", unit="%", thr=10,
                              specialty="dermatology or allergy", label="every 2 weeks after a loading dose",
                              high="weekly"),
}
TRIAL_OUTCOMES = {"inadequate_response": "inadequate response", "intolerance": "stopped for side effects",
                  "stopped_other": "stopped (cost / pharmacy access)", "not_tried": "not tried",
                  "contraindicated": "contraindicated per prescriber"}


def _drug_policy(a):
    return (f"{a['policy']} ({a['drug']}): covered for {a['condition']} when (1) {a['measure']} is at least "
            f"{a['thr']}{a['unit']} (most recent value); (2) the member had an inadequate response or intolerance to BOTH {a['steps'][0]} "
            f"and {a['steps'][1]}, each taken for at least {a['step_weeks']} weeks unless stopped sooner for side "
            f"effects, or has a documented contraindication to them; (3) prescribed by or with {a['specialty']}. "
            f"Plan dosing table: {a['label']}. Transition rule: a member new to the plan who was stable on the drug "
            f"under prior coverage receives one 30-day supply within the first 90 days of enrolment.")


class SpecialtyDrugStepTherapy(Schema):
    name = "specialty_drug_step_therapy"
    description = "A pharmacy prior-authorization request for a specialty drug governed by a step-therapy policy."
    surfaces = {"json": 0.5, "pharmacy_pa_form": 0.25, "prescriber_letter": 0.25}

    def _trial(self, rng, a, quality):
        sw = a["step_weeks"]
        if quality == "good":
            outcome = pick_w(rng, {"inadequate_response": 4, "intolerance": 1})
            weeks = sw + rng.randint(2, 20) if outcome == "inadequate_response" else rng.randint(3, sw)
            evidence = pick_w(rng, {"claims": 3, "chart": 2})
        elif quality == "short":
            outcome = pick_w(rng, {"inadequate_response": 4, "stopped_other": 1})
            weeks = near(rng, sw, 3, lo=2, integer=True)
            evidence = pick_w(rng, {"chart": 3, "attestation": 2, "claims": 1})
        elif quality == "missing":
            outcome = pick_w(rng, {"not_tried": 3, "contraindicated": 2})
            weeks = 0
            evidence = "attestation" if outcome == "contraindicated" else None
        else:
            outcome = pick(rng, list(TRIAL_OUTCOMES))
            weeks = 0 if outcome in ("not_tried", "contraindicated") else rng.randint(2, sw + 16)
            evidence = None if outcome == "not_tried" else pick(rng, ["claims", "chart", "attestation"])
        fills = None
        if evidence == "claims" or (weeks and chance(rng, 0.5)):
            fills = max(0, round(weeks / 4.3))
        return {"weeks": weeks, "outcome": outcome, "evidence": evidence, "claims_fills_30d": fills}

    def sample_factors(self, rng):
        reg = regime(rng)
        area = pick(rng, list(DRUG_AREAS))
        a = DRUG_AREAS[area]
        thr = a["thr"]
        variant = None
        new_member = chance(rng, 0.3)
        prior_cov_on_drug = new_member and chance(rng, 0.6)
        enrolled_days = rng.randint(10, 150) if new_member else rng.randint(200, 1400)
        stable_months = rng.randint(1, 18) if prior_cov_on_drug else None
        specialist = chance(rng, 0.7)
        freq = pick_w(rng, {"label": 4, "high": 1})
        freq_rationale = None
        if reg == "clear":
            good = chance(rng, 0.5)
            q = ["good", "good"] if good else [pick(rng, ["missing", "short"]), pick(rng, ["missing", "good"])]
            measure = thr + (rng.randint(3, 12) if good else -rng.randint(max(1, thr // 3), max(2, thr // 2)))
            specialist, freq = (True, "label") if good else (chance(rng, 0.4), freq)
            if good:
                new_member, prior_cov_on_drug, enrolled_days, stable_months = False, False, rng.randint(200, 900), None
        elif reg == "borderline":
            q = [pick(rng, ["good", "short"]), "short"]
            rng.shuffle(q)
            measure = near(rng, thr, max(1, thr * 0.2), lo=0, integer=True)
            if new_member:
                enrolled_days = near(rng, 90, 8, lo=30, integer=True)
        else:
            variant = pick(rng, ["chart_vs_claims", "contraindication_unsupported", "transition_member", "high_dose_argued"])
            q = ["good", pick(rng, ["good", "short"])]
            measure = thr + rng.randint(-2, 6)
            if variant == "transition_member":
                new_member, prior_cov_on_drug = True, True
                enrolled_days = rng.randint(20, 120)
                stable_months = rng.randint(1, 14)
                q = [pick(rng, ["missing", "short"]), pick(rng, ["missing", "short"])]
            elif variant == "contraindication_unsupported":
                q = ["good", "missing"]
            elif variant == "high_dose_argued":
                freq = "high"
                freq_rationale = pick(rng, ["partial response at the standard interval, with flare scores attached",
                                            "symptoms return in the last two weeks of each interval (diary attached)",
                                            "prescriber prefers tighter control"])
        if variant == "chart_vs_claims":
            new_member, prior_cov_on_drug, stable_months = False, False, None
            enrolled_days = rng.randint(300, 1400)
        trials = [self._trial(rng, a, qq) for qq in q]
        if enrolled_days < 120:
            # a new member's earlier fills sit with the previous insurer, not in this plan's claims
            for t in trials:
                t["claims_fills_30d"] = None
                if t["evidence"] == "claims":
                    t["evidence"] = "prior_insurer_records"
        if variant == "chart_vs_claims":
            t = trials[1]
            t["outcome"], t["evidence"] = "inadequate_response", "chart"
            t["weeks"] = a["step_weeks"] + rng.randint(2, 8)
            t["claims_fills_30d"] = rng.randint(1, 2)
        if variant == "contraindication_unsupported":
            trials[1].update(outcome="contraindicated", weeks=0, evidence="attestation", claims_fills_30d=None)
        if freq == "high" and freq_rationale is None:
            freq_rationale = maybe(rng, pick(rng, ["disease not controlled at label interval",
                                                   "same schedule the member used before"]), 0.4)
        f = {
            "regime": reg, "variant": variant, "area": area, "drug": a["drug"], "policy_id": a["policy"],
            "measure_value": max(0, measure), "measure_threshold": thr, "trials": trials,
            "prescriber_is_specialist": specialist,
            "prescriber_specialty": (a["specialty"].split(" or ")[0] if specialist else
                                     pick(rng, ["family medicine", "internal medicine", "primary care nurse practitioner"])),
            "requested_frequency": freq, "frequency_rationale": freq_rationale,
            "new_member": new_member, "enrolled_days": enrolled_days, "prior_coverage_on_drug": prior_cov_on_drug,
            "months_stable_on_drug": stable_months, "request_kind": "continuation" if prior_cov_on_drug else "new start",
            "plan": _plan(rng), "member_id": ident(rng, "M", 9, ""), "request_id": ident(rng, "RXPA", 6, "-"), "received": rand_date(rng), "pharmacist": person(rng),
            "measure_recorded_weeks_ago": pick(rng, [1, 3, 6, 14, None]),
        }
        f["prescriber"] = _clinician(rng, f["prescriber_specialty"])
        return f

    def _trial_line(self, a, i, t, rng):
        agent = a["steps"][i]
        if t["outcome"] == "not_tried":
            return f"{agent}: not tried"
        if t["outcome"] == "contraindicated":
            return f"{agent}: contraindicated" + (" (prescriber attestation, no supporting labs)"
                                                   if t["evidence"] == "attestation" else "")
        ev = {"claims": "plan pharmacy claims", "chart": "chart note", "attestation": "prescriber attestation",
              "prior_insurer_records": "previous insurer's pharmacy printout", None: "no source given"}[t["evidence"]]
        n = t["claims_fills_30d"]
        fills = "" if n is None else f"; claims show {n} x 30-day fill{'' if n == 1 else 's'}"
        return f"{agent}: {t['weeks']} weeks, {TRIAL_OUTCOMES[t['outcome']]} (source: {ev}{fills})"

    def render(self, f, rng, surface):
        a = DRUG_AREAS[f["area"]]
        policy = _drug_policy(a)
        spec = f["prescriber_specialty"]
        dose = a["label"] if f["requested_frequency"] == "label" else a["high"]
        measure = f"{f['measure_value']}{a['unit']}".strip()
        m_age = f["measure_recorded_weeks_ago"]
        if surface == "json":
            return dumps(prune({
                "request": {"id": f["request_id"], "received": f["received"].isoformat(), "drug": a["drug"],
                            "diagnosis": a["condition"], "type": f["request_kind"], "dosing_requested": dose,
                            "dosing_rationale": f["frequency_rationale"], "prescriber": f["prescriber"],
                            "prescriber_specialty": spec},
                "clinical": {"measure": a["measure"], "value": f["measure_value"],
                             "recorded_weeks_ago": m_age,
                             "prior_therapies": [self._trial_line(a, i, t, rng) for i, t in enumerate(f["trials"])]},
                "member": {"id": f["member_id"], "plan": f["plan"], "days_since_enrolment": f["enrolled_days"],
                           "on_drug_under_prior_coverage": (f"yes, about {f['months_stable_on_drug']} month"
                                                            + ("" if f["months_stable_on_drug"] == 1 else "s")
                                                            if f["prior_coverage_on_drug"] else None)},
                "policy": {"id": a["policy"], "text": policy},
            }))
        if surface == "pharmacy_pa_form":
            box = lambda b: "[x]" if b else "[ ]"  # noqa: E731
            rows = [("Request", f["request_id"]), ("Drug / dosing", f"{a['drug']} -- {dose}"),
                    ("Diagnosis", a["condition"]), ("New start / continuation", f["request_kind"]),
                    ("Prescriber", f"{f['prescriber']} ({spec})"),
                    (a["measure"].capitalize(), measure + ("" if m_age is None else f" (recorded {m_age} wk ago)")),
                    ("Member enrolled", f"{f['enrolled_days']} days ago")]
            trials = "\n".join(f"  {box(t['outcome'] != 'not_tried')} {self._trial_line(a, i, t, rng)}"
                               for i, t in enumerate(f["trials"]))
            extra = []
            if f["prior_coverage_on_drug"]:
                extra.append(f"Prescriber states member has taken {a['drug']} for {_mo(f['months_stable_on_drug'])} "
                             f"under previous insurance.")
            if f["frequency_rationale"]:
                extra.append(f"Dosing rationale: {f['frequency_rationale']}.")
            return (f"{f['plan'].upper()} -- PHARMACY PRIOR AUTHORIZATION\n{kv_lines(rows, rng)}\n\n"
                    f"Preferred agents tried:\n{trials}\n"
                    + (("\n" + noisy(" ".join(extra), rng, 0.1) + "\n") if extra else "")
                    + f"\nCriteria on file: {policy}")
        # prescriber_letter
        t0, t1 = f["trials"]
        para = [f"I am writing to request coverage of {a['drug']} ({dose}) for my patient, member {f['member_id']}, "
                f"who has {a['condition']}."]
        para.append(f"Current {a['measure']}: {measure}" + ("." if m_age is None else
                                                              f", measured {_wk(m_age)} ago."))
        for i, t in enumerate((t0, t1)):
            agent = a["steps"][i]
            line = {"not_tried": f"We have not used {agent}.",
                    "contraindicated": f"{agent} is not appropriate for this patient.",
                    "inadequate_response": f"{agent} was taken for {t['weeks']} weeks without adequate benefit.",
                    "intolerance": f"{agent} was stopped after {t['weeks']} weeks because of side effects.",
                    "stopped_other": f"{agent} was stopped after {t['weeks']} weeks when the pharmacy could not supply it."
                    }[t["outcome"]]
            para.append(line[0].upper() + line[1:])
        if f["prior_coverage_on_drug"]:
            para.append(f"The patient has been on {a['drug']} for {_mo(f['months_stable_on_drug'])} and joined your "
                        f"plan recently; interrupting therapy risks a flare.")
        if f["frequency_rationale"]:
            para.append(f"The requested interval is needed because of {f['frequency_rationale']}.")
        letter = noisy(join_sentences(para), rng, 0.12)
        fills = [f"{a['steps'][i]}: {t['claims_fills_30d']} fill{'' if t['claims_fills_30d'] == 1 else 's'}"
                 for i, t in enumerate((t0, t1))
                 if t["claims_fills_30d"] is not None]
        intake = kv_lines([("Intake pharmacist", f["pharmacist"]), ("Request", f["request_id"]),
                           ("Prescriber specialty", spec), ("Enrolment", f"{f['enrolled_days']} days"),
                           ("Claims history (this plan)", "; ".join(fills) if fills else "no fills on record")], rng)
        return f"LETTER OF MEDICAL NECESSITY\n\n{letter}\n\n{f['prescriber']}\n\n---\n{intake}\nPolicy: {policy}"

    def questions(self):
        return {
            "drug_determination": choice("What should the pharmacy review team decide on this request?", {
                "approve_as_requested": "Approve the drug at the requested dose and interval.",
                "approve_limited_quantity": "Approve the drug but only at the interval in the plan's dosing table.",
                "deny_criteria_not_met": "Deny it because the step-therapy or clinical criteria are not met.",
                "offer_peer_to_peer": "Offer the prescriber a discussion with a plan physician before deciding.",
                "pend_for_records": "Pend the request and ask the prescriber for specific records that would settle it.",
            }),
            "steps_satisfied": noul(
                "The member has satisfied the policy's step-therapy requirement.",
                yes="Both preferred agents were tried long enough, stopped for side effects, or credibly contraindicated.",
                no="A preferred agent was skipped, cut short without a qualifying reason, or is poorly documented."),
            "exception_case_strength": score("How strong is the case for waiving any unmet part of the step-therapy rule?", [
                "None: no reason is offered, or nothing is unmet.",
                "Weak: a preference or an unsupported assertion.",
                "Credible: a plausible clinical reason with some documentation.",
                "Compelling: documented harm or contraindication makes the skipped step unreasonable.",
            ]),
            "dosing_departure": score("How far does the requested dosing depart from the plan's dosing table?", [
                "Not at all: it matches the table.",
                "Slightly: a timing difference a pharmacist would usually accept.",
                "Moderately: a shorter interval or an extra dose, with a stated reason.",
                "Substantially: roughly double the maintenance frequency.",
                "Far outside the table, with no documented reason.",
            ]),
            "transition_supply_due": noul(
                "The member qualifies for the plan's one-time transition supply of this drug."),
        }


# =========================================================================== schema 3: inpatient concurrent review
DIAGNOSES = {
    "pneumonia": dict(label="community-acquired pneumonia", iv="IV antibiotics", oral="oral antibiotics", o2=True,
                      tests=["repeat chest X-ray", "sputum culture result"]),
    "cellulitis": dict(label="cellulitis of the right lower leg", iv="IV antibiotics", oral="oral antibiotics", o2=False,
                       tests=["ultrasound to rule out an abscess", "repeat blood cultures"]),
    "heart_failure": dict(label="acute on chronic heart failure", iv="IV diuretic", oral="oral diuretic", o2=True,
                          tests=["echocardiogram", "cardiology consult"]),
    "copd": dict(label="COPD exacerbation", iv="IV steroids", oral="oral steroid taper", o2=True,
                 tests=["home oxygen qualification walk test", "repeat blood gas"]),
    "pyelonephritis": dict(label="pyelonephritis", iv="IV antibiotics", oral="oral antibiotics", o2=False,
                           tests=["CT to rule out a kidney abscess", "repeat blood cultures"]),
    "foot_infection": dict(label="diabetic foot infection", iv="IV antibiotics", oral="oral antibiotics", o2=False,
                           tests=["MRI to rule out bone infection", "podiatry debridement"]),
}
ACUTE_POLICY = ("IP-300 continued-stay criteria: at least ONE of (a) an IV medication that cannot be given at a lower "
                "level of care; (b) oxygen above the member's baseline that is still being adjusted; (c) unstable "
                "vital signs within the last 24 hours; (d) a test or procedure that requires inpatient monitoring. "
                "Days spent only awaiting placement or equipment are reviewed as alternate-level-of-care days.")


class InpatientContinuedStay(Schema):
    name = "inpatient_continued_stay"
    description = "A hospital's request for more inpatient days, reviewed against the plan's continued-stay criteria."
    surfaces = {"json": 0.5, "case_manager_update": 0.25, "review_worksheet": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        dx = pick(rng, list(DIAGNOSES))
        d = DIAGNOSES[dx]
        day = rng.randint(3, 11)
        iv = pick_w(rng, {"iv_ongoing": 3, "switched_oral": 3, "iv_could_be_oral": 2})
        o2 = (pick_w(rng, {"room_air": 3, "at_baseline": 2, "above_baseline_weaning": 3, "above_baseline_rising": 1})
              if d["o2"] else "room_air")
        vitals = pick_w(rng, {"stable": 5, "single_low_fever": 2, "tachycardia": 1, "low_bp_episode": 1})
        labs = pick_w(rng, {"improving": 5, "flat": 3, "worsening": 1})
        pending = pick(rng, [None, None, None] + d["tests"])
        pt_rec = pick_w(rng, {"home": 3, "home_with_services": 3, "skilled_nursing": 2, None: 1})
        barrier = pick_w(rng, {None: 4, "snf_bed_search": 2, "home_oxygen_delivery": 1, "no_caregiver": 1,
                               "snf_auth_pending": 1})
        attending = pick_w(rng, {"not_safe": 2, "discharge_24_48h": 3, "family_requests_day": 1, None: 2})
        variant = None
        if reg == "clear":
            if chance(rng, 0.5):
                iv, vitals, labs = "iv_ongoing", pick(rng, ["tachycardia", "low_bp_episode"]), pick(rng, ["worsening", "flat"])
                o2 = "above_baseline_rising" if d["o2"] else "room_air"
                barrier, attending = None, "not_safe"
            else:
                iv, o2, vitals, labs, pending = "switched_oral", ("at_baseline" if d["o2"] else "room_air"), "stable", "improving", None
                barrier = pick(rng, ["snf_bed_search", "no_caregiver", None])
                attending = pick(rng, ["discharge_24_48h", "family_requests_day"])
        elif reg == "borderline":
            # exactly one weak criterion, if any
            weak = pick(rng, ["fever", "o2_weaning", "iv_today", "pending_test"])
            iv, vitals, pending = "switched_oral", "stable", None
            o2 = "at_baseline" if d["o2"] else "room_air"
            if weak == "fever":
                vitals = "single_low_fever"
            elif weak == "o2_weaning" and d["o2"]:
                o2 = "above_baseline_weaning"
            elif weak == "iv_today":
                iv = "iv_could_be_oral"
            else:
                pending = pick(rng, d["tests"])
        else:
            variant = pick(rng, ["attending_vs_criteria", "acute_but_ready", "mixed_barrier"])
            if variant == "attending_vs_criteria":
                iv, vitals, labs, attending = "iv_could_be_oral", "stable", "flat", "not_safe"
                o2 = pick(rng, ["at_baseline", "above_baseline_weaning"]) if d["o2"] else "room_air"
            elif variant == "acute_but_ready":
                iv, pt_rec, attending = "iv_ongoing", "home", "discharge_24_48h"
                vitals = "stable"
            else:
                barrier = pick(rng, ["snf_bed_search", "snf_auth_pending", "home_oxygen_delivery"])
                vitals = pick(rng, ["single_low_fever", "stable"])
                o2 = "above_baseline_weaning" if d["o2"] else "room_air"
                iv = pick(rng, ["switched_oral", "iv_could_be_oral"])
        if barrier == "home_oxygen_delivery" and not d["o2"]:
            barrier = "no_caregiver"
        lives_alone = chance(rng, 0.4)
        # keep the discharge story coherent: a bed search implies a facility recommendation, and so on
        if barrier in ("snf_bed_search", "snf_auth_pending"):
            pt_rec = "skilled_nursing"
        elif barrier in ("home_oxygen_delivery", "no_caregiver"):
            pt_rec = pick(rng, ["home", "home_with_services"])
            lives_alone = lives_alone or barrier == "no_caregiver"
        return {
            "regime": reg, "variant": variant, "dx": dx, "diagnosis": d["label"], "hospital_day": day,
            "authorized_through_day": day, "additional_days_requested": pick(rng, [1, 2, 2, 3, 4]),
            "iv_status": iv, "oxygen": o2, "vitals_24h": vitals, "lab_trend": labs, "pending_test": pending,
            "therapy_recommendation": pt_rec, "discharge_barrier": barrier, "attending_view": attending,
            "admissions_6mo": pick(rng, [0, 0, 1, 1, 2, 3, 4]), "lives_alone": lives_alone,
            "temp_max_c": round(rng.uniform(38.0, 38.4), 1) if vitals == "single_low_fever" else round(rng.uniform(36.5, 37.6), 1),
            "hr_max": rng.randint(104, 122) if vitals == "tachycardia" else rng.randint(68, 98),
            "sbp_min": rng.randint(82, 89) if vitals == "low_bp_episode" else rng.randint(102, 138),
            "o2_lpm": {"room_air": 0, "at_baseline": pick(rng, [0, 2]), "above_baseline_weaning": pick(rng, [1, 2, 3]),
                       "above_baseline_rising": pick(rng, [3, 4, 5])}[o2],
            "facility": f"{pick(rng, ['St. Brennan', 'Valewood', 'Northgate', 'Lakeshore', 'Ironbark'])} "
                        f"{pick(rng, ['Regional Hospital', 'Medical Center', 'Community Hospital'])}",
            "case_manager": person(rng), "plan": _plan(rng), "auth_id": ident(rng, "IP", 7, "-"),
            "review_ts": rand_ts(rng, hours=(7, 16)),
        }

    def _iv_text(self, f):
        d = DIAGNOSES[f["dx"]]
        return {"iv_ongoing": f"on {d['iv']}", "switched_oral": f"changed to {d['oral']} this morning",
                "iv_could_be_oral": f"remains on {d['iv']}; eating and drinking, tolerating oral medication"}[f["iv_status"]]

    def _o2_text(self, f):
        return {"room_air": "room air", "at_baseline": f"{f['o2_lpm']} L/min, the same as at home" if f["o2_lpm"] else "room air (baseline)",
                "above_baseline_weaning": f"{f['o2_lpm']} L/min above home baseline, weaned down from {f['o2_lpm'] + 1} yesterday",
                "above_baseline_rising": f"{f['o2_lpm']} L/min, increased overnight"}[f["oxygen"]]

    def _barrier_text(self, f):
        return {None: None, "snf_bed_search": "awaiting a skilled-nursing bed; two facilities declined, three pending",
                "home_oxygen_delivery": "home oxygen equipment delivery not until tomorrow afternoon",
                "no_caregiver": "lives alone; daughter can stay with the member from the weekend",
                "snf_auth_pending": "skilled-nursing admission waiting on a separate authorization"}[f["discharge_barrier"]]

    def _attending_text(self, f, rng):
        return {None: None, "not_safe": pick(rng, ["Attending: not safe for discharge.", "MD feels patient is not ready to go home."]),
                "discharge_24_48h": "Attending anticipates discharge in 24-48 hours.",
                "family_requests_day": "Family has asked for one more day before discharge."}[f["attending_view"]]

    def render(self, f, rng, surface):
        pt = {None: "not yet evaluated", "home": "home, no services", "home_with_services": "home with home health",
              "skilled_nursing": "skilled-nursing facility"}[f["therapy_recommendation"]]
        vit = f"Tmax {f['temp_max_c']} C, HR max {f['hr_max']}, SBP min {f['sbp_min']}"
        if surface == "json":
            return dumps(prune({
                "stay": {"auth_id": f["auth_id"], "facility": f["facility"], "diagnosis": f["diagnosis"],
                         "hospital_day": f["hospital_day"], "authorized_through_day": f["authorized_through_day"],
                         "additional_days_requested": f["additional_days_requested"]},
                "clinical_24h": {"medications": self._iv_text(f), "oxygen": self._o2_text(f), "vitals": vit,
                                 "labs": f"{f['lab_trend']} trend", "pending": f["pending_test"],
                                 "therapy_recommendation": pt},
                "discharge": {"barrier": self._barrier_text(f), "attending_note": self._attending_text(f, rng),
                              "lives_alone": f["lives_alone"], "admissions_last_6_months": f["admissions_6mo"]},
                "criteria": ACUTE_POLICY,
            }))
        if surface == "case_manager_update":
            parts = [f"Clinical update for {f['auth_id']}, hospital day {f['hospital_day']}, {f['diagnosis']}.",
                     f"Patient {self._iv_text(f)}; oxygen {self._o2_text(f)}.", f"Last 24h: {vit}.",
                     f"Labs {f['lab_trend']}.",
                     f"Pending: {f['pending_test']}." if f["pending_test"] else "",
                     f"PT/OT recommend {pt}.", self._attending_text(f, rng) or "",
                     (f"Barrier: {self._barrier_text(f)}." if f["discharge_barrier"] else ""),
                     (f"{f['admissions_6mo']} admission{'' if f['admissions_6mo'] == 1 else 's'} in the last 6 months."
                      if f["admissions_6mo"] else ""),
                     f"Requesting {f['additional_days_requested']} more day(s)."]
            return (f"From: {f['case_manager']}, RN CM, {f['facility']}\nTo: {f['plan']} concurrent review\n"
                    f"Sent: {fmt_ts(f['review_ts'], rng)}\n\n" + noisy(join_sentences(parts), rng, 0.12)
                    + f"\n\n[Reviewer reference] {ACUTE_POLICY}")
        rows = [["IV medication", self._iv_text(f)], ["Oxygen", self._o2_text(f)], ["Vitals (24h)", vit],
                ["Lab trend", f["lab_trend"]], ["Pending tests", f["pending_test"] or "none"],
                ["Therapy rec.", pt], ["Discharge barrier", self._barrier_text(f) or "none documented"]]
        head = kv_lines([("Auth", f["auth_id"]), ("Facility", f["facility"]), ("Dx", f["diagnosis"]),
                         ("Day of stay", f"{f['hospital_day']} (authorized through day {f['authorized_through_day']})"),
                         ("Requested", f"+{f['additional_days_requested']} day(s)"),
                         ("Readmissions (6 mo)", f["admissions_6mo"]),
                         ("Lives alone", "yes" if f["lives_alone"] else "no")], rng)
        att = self._attending_text(f, rng)
        return ("CONCURRENT REVIEW WORKSHEET\n" + head + "\n\n" + table(["item", "finding"], rows, rng)
                + (f"\n\nNotes: {att}" if att else "") + f"\n\n{ACUTE_POLICY}")

    def questions(self):
        return {
            "stay_determination": choice("How should the plan respond to the request for more inpatient days?", {
                "approve_one_day_reassess": "Authorize a single day and review again tomorrow.",
                "approve_requested_days": "Authorize all of the additional days requested.",
                "authorize_alternate_level": "Cover the extra days at the alternate-level-of-care rate while "
                                             "placement or equipment is arranged.",
                "deny_continued_stay": "Decline further inpatient days; the member no longer needs hospital care.",
                "refer_physician_advisor": "Refer the case to a plan physician advisor to speak with the attending.",
            }),
            "acute_need_today": noul(
                "The member still meets the plan's inpatient continued-stay criteria today.",
                yes="IV medication with no switch to oral and no note that oral medication is tolerated; oxygen "
                    "above home baseline that is still being changed (weaned or increased); or a heart rate over "
                    "100 or a systolic pressure under 90 in the last 24 hours.",
                no="Medication is oral or could be (oral medication tolerated), oxygen is at baseline, vitals are "
                   "stable or show only a low-grade fever, and anything pending is a test or consult that does not "
                   "itself need inpatient monitoring."),
            "barrier_is_logistical": noul(
                "What keeps the member in hospital today is placement or logistics rather than medical need.",
                yes="The member could safely leave if a bed, equipment or caregiver were in place.",
                no="The member's clinical condition itself still requires hospital care."),
            "clinical_stability": score("How stable is the member's condition on the latest review?", [
                "Unstable: vital signs or oxygen needs are moving the wrong way.",
                "Improving but not settled: one parameter is still being actively managed.",
                "Stable, but still on treatment that is being given in hospital.",
                "Stable and on treatment that could be given at home or in a facility.",
            ]),
            "bounce_back_risk": score("How likely is the member to be readmitted within 30 days if discharged on the current plan?", [
                "Low: first admission, good support and a clear follow-up plan.",
                "Moderate: one risk factor such as living alone or a recent admission.",
                "High: several risk factors, or a discharge plan that is not yet in place.",
                "Very high: frequent recent admissions and an unresolved clinical or support gap.",
            ]),
        }


# =========================================================================== schema 4: out-of-network exception
# (specialty, care needed, active course with this clinician, range for {n})
SPECIALTIES = [
    ("pediatric neurosurgery", "a tethered-cord release", "post-operative follow-up {n} weeks after a first surgery by this surgeon", (2, 10)),
    ("gynecologic oncology", "surgery and chemotherapy planning", "chemotherapy with this oncologist, cycle {n} of 6", (2, 5)),
    ("hand surgery", "a flexor tendon repair", "a staged repair, {n} weeks after the first stage by this surgeon", (2, 8)),
    ("adult congenital heart disease", "follow-up of a repaired heart defect", "medication titration with this cardiologist over the past {n} months", (2, 9)),
    ("movement disorders neurology", "adjustment of a deep brain stimulator", "stimulator programming visits with this neurologist for {n} months", (2, 12)),
    ("pediatric rheumatology", "management of juvenile arthritis", "infusions every 4 weeks with this clinic, dose {n} of 12", (2, 10)),
    ("maternal-fetal medicine", "care for a high-risk pregnancy", "pregnancy at {n} weeks, followed by this specialist since week 12", (22, 35)),
    ("eating disorder partial hospitalization", "a structured day-treatment program", "week {n} of a 12-week day program", (2, 9)),
]


def _access_policy(miles):
    return (f"Access standard: an in-network clinician of the needed specialty within {miles} miles, and a routine "
            f"specialist appointment within 20 business days. Transition of care: members in an active course of "
            f"treatment when they join the plan may keep an out-of-network clinician for up to 90 days.")


class OutOfNetworkException(Schema):
    name = "out_of_network_exception"
    description = "A request to see an out-of-network specialist at in-network cost sharing (a network gap exception)."
    surfaces = {"json": 0.5, "exception_request": 0.25, "member_call_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        spec, need, course_tpl, n_range = pick(rng, SPECIALTIES)
        rural = chance(rng, 0.4)
        std_miles = 75 if rural else 45
        nearest = rng.randint(8, std_miles + 60)
        wait = rng.randint(4, 45)
        accepting = pick_w(rng, {True: 5, False: 2, None: 2})
        in_count = pick(rng, [0, 1, 1, 2, 3, 5])
        unique = pick_w(rng, {None: 3, "asserted": 3, "verified": 2})
        course_kind = pick(rng, [None, "active", "active"])
        joined_days = rng.randint(10, 400)
        requester = pick(rng, ["member", "out_of_network_provider", "primary_care"])
        preference = pick_w(rng, {None: 3, "reputation": 2, "closer_to_work": 1, "prior_relationship": 2})
        variant = None
        if reg == "clear":
            if chance(rng, 0.5):
                in_count, nearest, wait, accepting, unique = 0, std_miles + rng.randint(40, 150), 60, None, "verified"
            else:
                in_count, nearest, wait, accepting, unique = rng.randint(3, 6), rng.randint(5, 20), rng.randint(3, 10), True, None
                course_kind, preference = None, pick(rng, ["reputation", "closer_to_work"])
        elif reg == "borderline":
            nearest = near(rng, std_miles, 10, lo=5, integer=True)
            wait = near(rng, 20, 5, lo=1, integer=True)
            in_count = pick(rng, [1, 1, 2])
            if course_kind:
                joined_days = near(rng, 90, 12, lo=5, integer=True)
        else:
            variant = pick(rng, ["mid_course_no_gap", "gap_but_preference", "unique_claim_unverified"])
            if variant == "mid_course_no_gap":
                course_kind = "active"
                in_count, nearest, wait, accepting = rng.randint(2, 5), rng.randint(8, 30), rng.randint(5, 14), True
                joined_days = rng.randint(100, 160)
            elif variant == "gap_but_preference":
                in_count, wait, accepting = 1, rng.randint(24, 40), pick(rng, [False, None])
                preference = pick(rng, ["reputation", "closer_to_work"])
                unique = None
            else:
                unique, in_count, nearest = "asserted", rng.randint(1, 3), rng.randint(15, std_miles)
        n = rng.randint(*n_range) if course_kind else None
        d = {
            "regime": reg, "variant": variant, "specialty": spec, "care_needed": need, "rural": rural,
            "standard_miles": std_miles, "nearest_in_network_miles": nearest,
            "in_network_wait_business_days": wait, "in_network_accepting": accepting,
            "in_network_count_within_standard": in_count if nearest <= std_miles else 0,
            "uniqueness_claim": unique, "active_course": course_kind,
            "course_text": course_tpl.format(n=n) if course_kind else None,
            "days_since_joining_plan": joined_days, "requester": requester, "member_preference": preference,
            "oon_provider": _doctor(rng), "oon_miles": rng.randint(3, 140),
            "single_case_agreement_offered": pick_w(rng, {True: 2, False: 1, None: 2}),
            "est_cost_oon_usd": round(rng.uniform(4000, 90000), -2), "cost_ratio_vs_network": round(rng.uniform(1.1, 3.2), 1),
            "service_date_days": pick(rng, [5, 12, 20, 35, 60]), "plan": _plan(rng), "member_id": ident(rng, "M", 9, ""),
            "ticket": ident(rng, "NGX", 6, "-"), "received": rand_date(rng), "navigator": person(rng),
        }
        # members mid-treatment are often recent joiners (drawn last, so every other factor is unchanged)
        if course_kind and reg != "borderline" and variant != "mid_course_no_gap" and chance(rng, 0.6):
            d["days_since_joining_plan"] = rng.randint(10, 85)
        return d

    def _course(self, f):
        return f["course_text"]

    def _uniq(self, f):
        return {None: None, "asserted": "the treating clinician's office states no in-network clinician performs this",
                "verified": "plan medical director confirmed no in-network clinician performs this"}[f["uniqueness_claim"]]

    def _pref(self, f):
        return {None: None, "reputation": "member prefers this clinician because of their reputation",
                "closer_to_work": "the office is closer to the member's workplace",
                "prior_relationship": "member saw this clinician before changing plans"}[f["member_preference"]]

    def _pref_said(self, f):
        return {"reputation": "they want this clinician because of the clinician's reputation",
                "closer_to_work": "the office is near their workplace",
                "prior_relationship": "they saw this clinician before changing plans"}[f["member_preference"]]

    def render(self, f, rng, surface):
        pol = _access_policy(f["standard_miles"])
        acc = {True: "accepting new patients", False: "not accepting new patients", None: "not confirmed"}[f["in_network_accepting"]]
        if surface == "json":
            return dumps(prune({
                "request": {"ticket": f["ticket"], "received": f["received"].isoformat(), "requested_by": f["requester"],
                            "specialty": f["specialty"], "care_needed": f["care_needed"],
                            "out_of_network_clinician": f["oon_provider"], "distance_miles": f["oon_miles"],
                            "service_in_days": f["service_date_days"], "stated_reasons": [x for x in
                                                                                         (self._uniq(f), self._pref(f), self._course(f)) if x],
                            "active_treatment_with_this_clinician": self._course(f) or "none reported"},
                "network_search": {"nearest_in_network_miles": f["nearest_in_network_miles"],
                                   "in_network_within_standard": f["in_network_count_within_standard"],
                                   "earliest_in_network_appointment_business_days": f["in_network_wait_business_days"],
                                   "nearest_clinician_status": acc},
                "member": {"id": f["member_id"], "plan": f["plan"], "days_since_joining": f["days_since_joining_plan"],
                           "area": "rural" if f["rural"] else "urban"},
                "cost": {"estimated_out_of_network_cost_usd": f["est_cost_oon_usd"],
                         "multiple_of_network_rate": f["cost_ratio_vs_network"],
                         "single_case_agreement_offered": f["single_case_agreement_offered"]},
                "policy": pol,
            }))
        if surface == "exception_request":
            reasons = [x for x in (self._uniq(f), self._course(f), self._pref(f)) if x]
            rows = [("Ticket", f["ticket"]), ("Received", fmt_date(f["received"], rng)),
                    ("Requested by", f["requester"].replace("_", " ")), ("Specialty", f["specialty"]),
                    ("Needed", f["care_needed"]), ("Out-of-network clinician", f"{f['oon_provider']} ({f['oon_miles']} mi)"),
                    ("Planned date", f"in {f['service_date_days']} days"),
                    ("Nearest in-network", f"{f['nearest_in_network_miles']} mi, {acc}"),
                    ("In-network options inside standard", f["in_network_count_within_standard"]),
                    ("Earliest in-network appt", f"{f['in_network_wait_business_days']} business days"),
                    ("Member joined plan", f"{f['days_since_joining_plan']} days ago"),
                    ("Active treatment with this clinician", self._course(f) or "none reported"),
                    ("Est. cost", money(f["est_cost_oon_usd"], rng) + f" (~{f['cost_ratio_vs_network']}x network)"),
                    ("Single-case agreement", show(f["single_case_agreement_offered"], rng,
                                                   lambda v: "provider open to one" if v else "declined"))]
            return ("NETWORK GAP EXCEPTION REQUEST\n" + kv_lines(rows, rng) + "\n\nReasons given:\n"
                    + (bullets(reasons, rng) if reasons else "(none stated)") + f"\n\n{pol}")
        # member_call_note
        said = []
        if f["active_course"]:
            said.append(f"Member is mid-treatment: {self._course(f)}.")
        else:
            said.append("No treatment is under way with this clinician yet.")
        if f["member_preference"]:
            said.append(pick(rng, ["Member says ", "Caller explains "]) + self._pref_said(f) + ".")
        if f["uniqueness_claim"]:
            said.append(self._uniq(f)[0].upper() + self._uniq(f)[1:] + ".")
        said.append(f"Navigator search: nearest in-network {f['specialty']} {f['nearest_in_network_miles']} miles away ({acc}); "
                    f"earliest appointment {f['in_network_wait_business_days']} business days; "
                    f"{f['in_network_count_within_standard']} option(s) inside the {f['standard_miles']}-mile standard.")
        said.append(f"Care needed: {f['care_needed']}, planned in {f['service_date_days']} days.")
        said.append(f"Joined plan {f['days_since_joining_plan']} days ago.")
        return (f"CALL NOTE {fmt_ts(rand_ts(rng, f['received'], (8, 18)), rng)} -- {f['navigator']}, member navigation\n"
                f"Member {f['member_id']} ({f['plan']}), ticket {f['ticket']}\n\n"
                + noisy(join_sentences(said), rng, 0.15) + f"\n\nRef: {pol}")

    def questions(self):
        return {
            "network_exception_decision": choice("How should the plan answer this out-of-network request?", {
                "approve_exception": "Grant in-network cost sharing with the requested clinician for this episode of care.",
                "approve_transition_period": "Grant it only for a limited transition period, then move care in network.",
                "deny_exception": "Refuse the exception; the member may still go out of network at out-of-network rates.",
                "pend_network_search": "Hold the decision while the plan confirms whether an in-network clinician is available.",
                "redirect_in_network": "Refer the member to a named in-network clinician and help book the appointment.",
            }),
            "access_standard_breached": noul(
                "The plan's network fails its own distance or appointment-wait standard for this member.",
                yes="No in-network option lies inside the distance standard, the earliest in-network appointment is "
                    "more than 20 business days away, or the only option inside the standard is not accepting new "
                    "patients.",
                no="At least one in-network option inside the distance standard can see the member within 20 "
                   "business days (a 'not confirmed' status does not count against it)."),
            "provider_uniqueness": score("How convincingly does the file show that no in-network clinician can provide this care?", [
                "Not at all: in-network clinicians routinely provide it.",
                "Weakly: the claim is a preference or an unverified statement.",
                "Plausibly: the in-network options are thin, distant or unconfirmed.",
                "Convincingly: the plan itself confirmed the care is not available in network.",
            ]),
            "disruption_harm": score("How much would making the member switch clinicians now set back their care?", [
                "Negligible: no treatment is under way with this clinician.",
                "Minor: ongoing adjustment visits (medication titration or device programming) that another "
                "clinician could take over with some repeated visits.",
                "Significant: a structured course part-way through (an infusion series, a day program, a staged "
                "repair or post-operative follow-up), or a pregnancy before 28 weeks.",
                "Serious: chemotherapy part-way through its cycles, or a high-risk pregnancy at 28 weeks or later.",
            ]),
            "transition_protection_applies": noul(
                "The plan's transition-of-care rule covers this member's current treatment.",
                yes="The member is part-way through a course of treatment with this clinician and joined the plan 90 "
                    "days ago or less.",
                no="No course of treatment is under way with this clinician, or the member joined more than 90 days "
                   "ago."),
        }


# =========================================================================== schema 5: internal appeal
APPEAL_SERVICES = [
    dict(service="MRI lumbar spine (RAD-112)", reason="fewer than 6 weeks of documented conservative care",
         good="physical therapy discharge notes covering 9 weeks of treatment",
         partial="a home-exercise log kept by the member for 5 weeks",
         weak="a letter restating that the member is in severe pain", specialty="spine",
         expedite=["leg weakness getting worse week by week", "member cannot work and is losing income"]),
    dict(service="ostralimab (RX-4520)", reason="no documented trial of halcovine",
         good="pharmacy records from the previous insurer showing 14 weeks of halcovine",
         partial="a chart note saying halcovine was tried 'for a few months' two years ago",
         weak="a prescriber attestation that halcovine 'did not work'", specialty="rheumatology",
         expedite=["joint flares worsening while off treatment", "member has run out of the previous medication"]),
    dict(service="inpatient days 5-7, heart failure (IP-300)", reason="continued-stay criteria not met after day 4",
         good="nursing flowsheets showing oxygen increased on day 5 and IV diuretic restarted",
         partial="a physician note describing fatigue and poor appetite on day 5",
         weak="a letter explaining the family could not care for the member at home", specialty="cardiology",
         expedite=[], post_only=True),
    dict(service="20 additional outpatient speech therapy visits", reason="no measurable progress in the last 10 visits",
         good="a standardised re-test showing gains since the last authorization",
         partial="therapist notes describing slow but steady improvement without scores",
         weak="a parent's letter describing how much the therapy matters", specialty="speech-language pathology",
         expedite=["the child is losing skills during the gap in therapy", "a school placement meeting is next week"]),
    dict(service="in-lab sleep study (SLP-020)", reason="a home sleep test must be tried first",
         good="two home sleep tests that failed for technical reasons",
         partial="a note that the member has a condition that may make home testing unreliable",
         weak="the member's preference for testing in a lab", specialty="sleep medicine",
         expedite=["member drives for work and cannot return until tested", "daytime sleepiness getting worse"]),
]


class DenialAppeal(Schema):
    name = "denial_appeal"
    description = "A first-level internal appeal of a prior-authorization denial, with whatever new information came in."
    surfaces = {"json": 0.5, "appeal_letter": 0.25, "appeal_summary": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        svc_i = rng.randrange(len(APPEAL_SERVICES))
        info = pick_w(rng, {"good": 2, "partial": 3, "weak": 2, "none": 1})
        cited = chance(rng, 0.75)
        same_spec = chance(rng, 0.55)
        overlooked = chance(rng, 0.15)
        svc = APPEAL_SERVICES[svc_i]
        kind = pick_w(rng, {"pre_service_standard": 5, "pre_service_expedited": 2, "post_service": 2})
        if svc.get("post_only"):
            kind = "post_service"
        window = {"pre_service_standard": 30, "pre_service_expedited": 3, "post_service": 60}[kind]
        elapsed = rng.randint(0, window)
        aor = pick_w(rng, {"member_filed": 3, "provider_with_form": 3, "provider_no_form": 1})

        variant = None
        if reg == "clear":
            if chance(rng, 0.5):
                info, cited, same_spec = "good", True, True
            else:
                info, cited, overlooked = pick(rng, ["weak", "none"]), True, False
            elapsed = rng.randint(1, max(1, window // 3))
        elif reg == "borderline":
            info = "partial"
            elapsed = near(rng, window * 0.85, window * 0.15, lo=0, hi=window, integer=True)
        else:
            variant = pick(rng, ["overlooked_records", "good_info_procedural_gap", "expedite_disputed"])
            if variant == "overlooked_records":
                overlooked, info = True, pick(rng, ["none", "weak"])
            elif variant == "good_info_procedural_gap":
                info, aor, cited = "good", "provider_no_form", False
            elif not svc.get("post_only"):
                kind, window, info = "pre_service_expedited", 3, pick(rng, ["partial", "weak"])
                elapsed = rng.randint(1, 3)
            else:
                info, elapsed = "partial", rng.randint(40, 60)
        return {
            "regime": reg, "variant": variant, "service_index": svc_i, "new_information": info,
            "denial_cited_criterion": cited, "original_reviewer_same_specialty": same_spec,
            "records_overlooked_at_denial": overlooked, "appeal_kind": kind, "decision_window_days": window,
            "days_elapsed": elapsed, "representation": aor, "service_pending": kind != "post_service",
            "member_statement": pick_w(rng, {None: 2, "calm": 2, "distressed": 2, "threatens_complaint": 1}),
            "expedite_basis": (maybe(rng, pick(rng, svc["expedite"]), 0.25) if kind == "pre_service_expedited" else None),
            "plan": _plan(rng), "appeal_id": ident(rng, "APL", 6, "-"), "member_id": ident(rng, "M", 9, ""),
            "denial_date": rand_date(rng), "coordinator": person(rng), "reviewer": _doctor(rng),
        }

    def render(self, f, rng, surface):
        s = APPEAL_SERVICES[f["service_index"]]
        new = {"good": s["good"], "partial": s["partial"], "weak": s["weak"], "none": None}[f["new_information"]]
        kind = {"pre_service_standard": "pre-service, standard", "pre_service_expedited": "pre-service, expedited",
                "post_service": "post-service"}[f["appeal_kind"]]
        received = f["denial_date"] + timedelta(days=rng.randint(3, 40))
        denial = (f"Denied: {s['reason']}." + ("" if f["denial_cited_criterion"]
                                                 else " (Notice cites the policy number but not which criterion failed.)"))
        rep = {"member_filed": "filed by the member",
               "provider_with_form": "filed by the treating clinician with a signed representative form",
               "provider_no_form": "filed by the treating clinician; no representative form on file"}[f["representation"]]
        orig = ("original reviewer's specialty matches the service" if f["original_reviewer_same_specialty"]
                else "original reviewer was a general internist")
        overlooked = ("Audit note: records submitted with the first request were not opened before the denial."
                      if f["records_overlooked_at_denial"] else None)
        stmt = {None: None, "calm": "Member asks the plan to reconsider in light of the enclosed.",
                "distressed": "Member writes that the delay is making daily life very hard.",
                "threatens_complaint": "Member says they will complain to the state regulator if this is not fixed."}[f["member_statement"]]
        if surface == "json":
            return dumps(prune({
                "appeal": {"id": f["appeal_id"], "type": kind, "received": received.isoformat(), "representation": rep,
                           "days_elapsed": f["days_elapsed"], "decision_window_days": f["decision_window_days"],
                           "expedite_basis": f["expedite_basis"], "member_statement": stmt},
                "original_denial": {"service": s["service"], "date": f["denial_date"].isoformat(), "notice": denial,
                                    "reviewer": orig, "audit_note": overlooked},
                "new_information": {"summary": new or "no new documents enclosed"},
                "context": {"member": f["member_id"], "plan": f["plan"], "service_still_pending": f["service_pending"]},
            }))
        if surface == "appeal_letter":
            who = "I" if f["representation"] == "member_filed" else "We"
            if stmt and who == "We":
                stmt = {"calm": "", "distressed": "Our patient tells us the delay is making daily life very hard.",
                        "threatens_complaint": "Our patient has said they will complain to the state regulator."
                        }[f["member_statement"]]
            elif stmt:
                stmt = {"calm": "Please reconsider in light of what I have enclosed.",
                        "distressed": "The delay is making my daily life very hard.",
                        "threatens_complaint": "If this is not fixed I will complain to the state regulator."
                        }[f["member_statement"]]
            body = [f"{who} are appealing the denial of {s['service']} dated {fmt_date(f['denial_date'], rng)}."
                    if who == "We" else f"I am appealing the denial of {s['service']} dated {fmt_date(f['denial_date'], rng)}.",
                    f"The denial said: {s['reason']}.",
                    (f"Enclosed: {new}." if new else "We believe the original decision was simply wrong."),
                    (f"Please treat this as urgent: {f['expedite_basis']}." if f["expedite_basis"] else ""),
                    stmt or ""]
            tail = kv_lines([("Appeal no.", f["appeal_id"]), ("Logged as", kind), ("Received", fmt_date(received, rng)),
                             ("Clock", f"day {f['days_elapsed']} of {f['decision_window_days']}"),
                             ("Filed", rep), ("Denial notice", "cites criterion" if f["denial_cited_criterion"]
                                              else "policy number only"), ("First review", orig)], rng)
            return ("Re: Appeal of prior authorization denial\n\n" + noisy(join_sentences(body), rng, 0.15)
                    + "\n\n--- appeals unit intake ---\n" + tail + (f"\n{overlooked}" if overlooked else ""))
        rows = [["Appeal", f["appeal_id"]], ["Service", s["service"]], ["Type", kind],
                ["Clock", f"day {f['days_elapsed']} of {f['decision_window_days']}"],
                ["Denial basis", s["reason"]], ["Criterion cited", "yes" if f["denial_cited_criterion"] else "no"],
                ["First reviewer", orig], ["Filed", rep], ["New information", new or "none"]]
        extra = [x for x in (overlooked, stmt, f"Expedite basis: {f['expedite_basis']}" if f["expedite_basis"] else None) if x]
        return (f"APPEAL CASE SUMMARY -- {f['plan']}\nPrepared by {f['coordinator']} for {f['reviewer']}\n\n"
                + table(["field", "detail"], rows, rng) + ("\n\n" + bullets(extra, rng) if extra else ""))

    def questions(self):
        return {
            "appeal_resolution": choice("How should the plan resolve this appeal?", {
                "overturn_denial": "Reverse the denial and authorize the service in full.",
                "overturn_in_part": "Reverse the denial for part of what was requested, such as fewer visits or days.",
                "route_to_matched_specialist": "Send the file to a reviewer in the same specialty before ruling.",
                "uphold_denial": "Keep the denial in place; the appeal does not change the outcome.",
            }),
            "new_information_material": noul(
                "The appeal adds clinical information that bears directly on the reason for denial.",
                yes="The new material addresses the specific criterion the denial relied on.",
                no="Nothing new was sent, or what was sent does not speak to that criterion."),
            "original_denial_defensible": score("How defensible was the original denial on the record the first reviewer had?", [
                "Indefensible: the record already supported approval, or was not properly read.",
                "Weak: a thin or procedurally flawed decision.",
                "Reasonable: a fair reading of an incomplete record.",
                "Solid: the record plainly did not meet the criterion.",
            ]),
            "appeal_time_pressure": score("How much time pressure is the appeals unit under on this file?", [
                "Little: most of the decision window remains and nothing is scheduled.",
                "Some: the window is past its midpoint or a service date is approaching.",
                "High: the window closes within days, or the member's condition makes the expedited track apply.",
                "Critical: the deadline is today or the member is waiting on care now.",
            ]),
            "procedural_flaw": noul("The first denial had a procedural flaw serious enough to matter on appeal."),
        }


WORKFLOW = Workflow(
    name="prior_authorization_review",
    title="Health-plan prior authorization review",
    description="Whether a health plan should authorize a requested service or drug under its written, fictional "
                "medical-necessity policy, and how the utilization-management team should respond.",
    domain="healthcare payer / utilization management",
    schemas=[AdvancedImagingRequest(), SpecialtyDrugStepTherapy(), InpatientContinuedStay(), OutOfNetworkException(),
             DenialAppeal()],
    notes="All plans, policies, drugs and people are fictional; the criteria are invented plan rules, not clinical "
          "guidance. Reviews are payer-side decisions on requests, not member conversations or claim payments.",
)
