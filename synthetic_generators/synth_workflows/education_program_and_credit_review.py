"""Education program and credit review: institutional academic decisions under written (fictional) policies.

Five schemas: a registrar's transfer-credit equivalency evaluation; an accreditor's finding on one standard; a state
system's academic program review (continue, revise, consolidate or sunset on enrolment and outcome data); a student's
course-substitution petition; and a prior-learning assessment for credit. Each state quotes only the one to three
clauses that govern the case, with thresholds sampled per case: a 5-quarter-credit course against a 4-credit home
course, a pass rate one point under the bar, a petition one credit over the substitution cap.

Students are identified only by a name, an id and their academic record. No protected attribute appears as a factor
and nothing turns on who a student is. `intended(f)` records the answer logic a careful registrar, reviewer or
committee would apply to the rendered facts; the builder never uses it. Institutions, accreditors, policies and
people are invented.
"""
from __future__ import annotations

from .base import (
    ORG_A,
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
    table,
)

SCHOOL_B = ["University", "State University", "College", "Community College", "Technical College", "Institute of Technology"]
GRADES = ["A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "F"]
GPTS = {"A": 4.0, "A-": 3.7, "B+": 3.3, "B": 3.0, "B-": 2.7, "C+": 2.3, "C": 2.0, "C-": 1.7, "D+": 1.3, "D": 1.0, "F": 0.0}


def _school(rng, kinds=SCHOOL_B):
    return f"{pick(rng, ORG_A)} {pick(rng, kinds)}"


def _clamp(x, lo=0, hi=3):
    return max(lo, min(hi, x))


def _pol(rng, prefix):
    return f"{prefix} {rng.randint(2, 9)}.{rng.randint(1, 19)}"


# =========================================================================== schema 1: transfer credit
# home course: (code, title, credits, lab, science-type) -> incoming variants (title, same discipline, match range)
HOME_COURSES = {
    ("CHEM 1210", "General Chemistry I", 4, True, True): [
        ("General Chemistry I with Laboratory", True, (75, 100)), ("Chemistry for Health Sciences", True, (45, 75)),
        ("Physical Science Survey", False, (20, 45))],
    ("BIOL 2420", "Microbiology", 4, True, True): [
        ("Microbiology for Allied Health", True, (70, 98)), ("Introduction to Microbes and Disease", True, (45, 72)),
        ("Environmental Science", False, (15, 40))],
    ("PHYS 2210", "Physics for Scientists I", 4, True, True): [
        ("Calculus-Based Physics I", True, (78, 100)), ("Algebra-Based Physics I", True, (50, 75)),
        ("Astronomy", False, (15, 40))],
    ("MATH 2250", "Calculus II", 4, False, True): [
        ("Integral Calculus", True, (75, 100)), ("Applied Calculus for Business", True, (35, 65)),
        ("Finite Mathematics", False, (10, 35))],
    ("CS 1410", "Programming I", 4, False, True): [
        ("Introduction to Programming in Python", True, (70, 98)), ("Computer Applications", True, (25, 50)),
        ("Web Design Fundamentals", False, (20, 45))],
    ("STAT 2040", "Statistical Methods", 3, False, True): [
        ("Elementary Statistics", True, (70, 98)), ("Statistics for the Social Sciences", True, (55, 85)),
        ("Quantitative Reasoning", False, (25, 50))],
    ("PSYC 1010", "Introduction to Psychology", 3, False, False): [
        ("General Psychology", True, (80, 100)), ("Psychology of Personal Adjustment", True, (40, 70)),
        ("Introduction to Sociology", False, (15, 40))],
    ("ACCT 2010", "Financial Accounting", 3, False, False): [
        ("Principles of Accounting I", True, (78, 100)), ("Bookkeeping Fundamentals", True, (40, 70)),
        ("Personal Finance", False, (15, 40))],
    ("ECON 2010", "Principles of Microeconomics", 3, False, False): [
        ("Microeconomics", True, (80, 100)), ("Survey of Economics", True, (45, 72)),
        ("Business Environment", False, (20, 45))],
    ("ENGL 1020", "Composition II", 3, False, False): [
        ("Argument and Research Writing", True, (72, 98)), ("Technical Writing", True, (45, 72)),
        ("Introduction to Literature", False, (20, 45))],
    ("HIST 1700", "World History to 1500", 3, False, False): [
        ("World Civilizations I", True, (75, 100)), ("Western Civilization I", True, (45, 70)),
        ("Introduction to Anthropology", False, (15, 35))],
    ("BIOL 2320", "Human Anatomy", 4, True, True): [
        ("Human Anatomy and Physiology I", True, (60, 90)), ("Anatomy with Cadaver Laboratory", True, (80, 100)),
        ("Biology of Human Health", False, (20, 45))],
    ("NURS 2100", "Pathophysiology", 3, False, True): [
        ("Pathophysiology for Health Professions", True, (75, 100)), ("Disease Processes", True, (50, 78)),
        ("Medical Terminology", False, (10, 30))],
    ("MUS 1100", "Music Theory I", 3, False, False): [
        ("Theory and Aural Skills I", True, (72, 98)), ("Fundamentals of Music", True, (35, 60)),
        ("Music Appreciation", False, (10, 30))],
}
# incoming course prefix: by keyword in the incoming title, else the home course's own subject
PREFIX_KEYS = [("Astronomy", "AST"), ("Physical Science", "PHS"), ("Environmental", "ENV"), ("Sociology", "SOC"),
               ("Anthropology", "ANT"), ("Literature", "LIT"), ("Technical Writing", "ENG"), ("Personal Finance", "FIN"),
               ("Business", "BUS"), ("Finite", "MAT"), ("Quantitative", "MAT"), ("Web Design", "CIS"),
               ("Computer Applications", "CIS"), ("Medical Terminology", "HLT"), ("Biology of Human", "BIO"),
               ("Music Appreciation", "MUS"), ("Bookkeeping", "ACC")]
HOME_TO_SOURCE = {"CHEM": "CHM", "BIOL": "BIO", "PHYS": "PHY", "MATH": "MAT", "CS": "CSC", "STAT": "STA", "PSYC": "PSY",
                  "ACCT": "ACC", "ECON": "ECO", "ENGL": "ENG", "HIST": "HIS", "NURS": "NUR", "MUS": "MUS"}


def _prefix(in_title, home_code):
    for k, v in PREFIX_KEYS:
        if k in in_title:
            return v
    return HOME_TO_SOURCE.get(home_code.split()[0], "GEN")


ACCRED = {
    "regional": ("accredited by a recognised regional body", True),
    "national": ("accredited by a recognised national body", True),
    "foreign_eval": ("institution abroad; credential evaluation on file rates it equivalent to an accredited institution", True),
    "foreign_none": ("institution abroad; no credential evaluation submitted", False),
    "unrecognised": ("accredited only by a body the policy does not recognise", False),
}


class TransferCreditEquivalency(Schema):
    name = "transfer_credit_equivalency"
    description = "A registrar's evaluation of one incoming transfer course against a home-institution course under the written transfer-credit policy."
    surfaces = {"json": 0.5, "evaluation_form": 0.25, "evaluator_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng, clear=0.25, borderline=0.45, conflicting=0.30)
        home = pick(rng, list(HOME_COURSES))
        code, title, hcr, lab, sci = home
        variants = HOME_COURSES[home]
        v = pick_w(rng, {0: 3.5, 1: 3, 2: 2.5})
        in_title, same, (lo, hi) = variants[v]
        m = pick(rng, [70, 75, 80])
        match = rng.randint(lo, hi)
        if reg == "borderline" and v < 2:
            match = near(rng, m, 9, lo=5, hi=100, integer=True)
        system = pick_w(rng, {"semester": 6, "quarter": 3})
        if system == "quarter":
            in_cr = pick(rng, [4, 5]) if hcr == 3 else pick(rng, [5, 6, 4])
        else:
            in_cr = pick_w(rng, {hcr: 7, hcr - 1: 2, hcr + 1: 1})
        acc = pick_w(rng, {"regional": 6, "national": 2, "foreign_eval": 1.5, "foreign_none": 0.5, "unrecognised": 0.5})
        min_grade = pick(rng, ["C", "C", "C-"])
        grade = pick_w(rng, {"A": 2, "A-": 2, "B+": 2, "B": 3, "B-": 2, "C+": 2, "C": 2, "C-": 0.8, "D+": 0.5, "D": 0.3})
        rec = pick(rng, [7, 10])
        age = pick_w(rng, {rng.randint(0, 4): 5, rng.randint(5, 9): 3, rng.randint(10, 18): 3})
        lab_in = (chance(rng, 0.75) if lab else None)
        level = "college" if chance(rng, 0.96) else "developmental"
        docs = pick_w(rng, {"syllabus": 6, "catalogue": 4})
        official = chance(rng, 0.75 if docs == "syllabus" else 0.5)
        signoff = chance(rng, 0.7 if docs == "syllabus" else 0.45)
        if reg == "clear":
            if chance(rng, 0.5):
                v, (in_title, same, _) = 0, variants[0]
                match, acc, grade, age, lab_in, level = rng.randint(m + 8, 100), "regional", pick(rng, ["A", "B+", "B"]), rng.randint(0, 4), (True if lab else None), "college"
                in_cr = hcr if system == "semester" else (5 if hcr == 3 else 6)
            else:
                twist = pick(rng, ["grade", "acc", "level"])
                grade = pick(rng, ["D", "D+", "F"]) if twist == "grade" else grade
                acc = "unrecognised" if twist == "acc" else acc
                level = "developmental" if twist == "level" else level
        elif reg == "conflicting":
            twist = pick(rng, ["old_science", "strong_but_short", "grade_edge", "foreign"])
            if twist == "old_science" and sci:
                age, match = rng.randint(rec + 1, rec + 8), rng.randint(m + 5, 100)
            elif twist == "strong_but_short":
                match = rng.randint(m + 5, 100)
                system, in_cr = ("quarter", 4) if hcr == 3 else ("semester", hcr - 1)
            elif twist == "grade_edge":
                grade = pick(rng, ["C-", "D+"]) if min_grade == "C" else pick(rng, ["C-", "D+"])
                match = rng.randint(m + 5, 100)
            else:
                acc = pick(rng, ["foreign_eval", "foreign_none"])
        return {
            "regime": reg, "home_code": code, "home_title": title, "home_credits": hcr, "home_lab": lab, "science": sci,
            "in_title": in_title, "same_discipline": same, "match_pct": match, "match_min": m, "system": system,
            "in_credits": in_cr, "accreditation": acc, "min_grade": min_grade, "grade": grade, "recency_years": rec,
            "age_years": age, "lab_in": lab_in, "level": level, "docs": docs, "official": official, "signoff": signoff,
            "source": _school(rng), "home_school": _school(rng, ["University", "State University", "College"]),
            "in_code": f"{_prefix(in_title, code)} {rng.randint(101, 299)}",
            "student": person(rng), "sid": ident(rng, "S", 7, ""), "evaluator": person(rng),
            "policy": _pol(rng, "TCP"), "date": rand_date(rng),
        }

    @staticmethod
    def _converted(f):
        return round(f["in_credits"] * 2 / 3, 2) if f["system"] == "quarter" else float(f["in_credits"])

    def intended(self, f):
        conv = self._converted(f)
        cr_ok = conv >= f["home_credits"] - 0.01
        lab_ok = (not f["home_lab"]) or bool(f["lab_in"])
        grade_ok = GPTS[f["grade"]] >= GPTS[f["min_grade"]]
        recent = (not f["science"]) or f["age_years"] <= f["recency_years"]
        acc_ok = ACCRED[f["accreditation"]][1]
        if f["level"] != "college" or not acc_ok or not grade_ok:
            ruling = "no_credit"
        elif not recent:
            ruling = "general_elective"
        elif f["match_pct"] >= f["match_min"] and f["same_discipline"]:
            ruling = "direct_equivalent" if (cr_ok and lab_ok) else "equivalent_with_shortfall"
        elif f["same_discipline"]:
            ruling = "departmental_elective"
        else:
            ruling = "general_elective"
        mp = f["match_pct"]
        align = 0 if mp < 40 else 1 if mp < 65 else 2 if mp < 88 else 3
        evq = (f["docs"] == "syllabus") + f["official"] + f["signoff"]
        return {"equivalency_ruling": ruling, "outcome_alignment": align, "evaluation_record_quality": evq,
                "grade_and_recency_ok": "yes" if grade_ok and recent else "no",
                "credit_after_conversion_sufficient": "yes" if cr_ok else "no"}

    def _rules(self, f, rng):
        lab = "; a lab course needs a lab" if f["home_lab"] else ""
        r = [f"{f['policy']}(a): credit only for college-level work from institutions accredited by a recognised body "
             f"(abroad: with a credential evaluation), passed with {f['min_grade']} or better.",
             f"{f['policy']}(b): direct equivalent = >= {f['match_min']}% of the home course's outcomes and at least its "
             f"credit after conversion (quarter x 2/3){lab}. Short only on credit or lab -> equivalent, shortfall made up; "
             f"same discipline under {f['match_min']}% -> departmental elective; otherwise general elective."]
        if f["science"] or chance(rng, 0.3):
            r.append(f"{f['policy']}(c): science, mathematics and computing courses older than {f['recency_years']} years "
                     "transfer as general elective only.")
        return r

    def _lab_txt(self, f):
        return None if f["lab_in"] is None else ("with lab" if f["lab_in"] else "no lab component")

    def render(self, f, rng, surface):
        cr = f"{f['in_credits']} {f['system']} credits"
        docs = "syllabus with outcomes" if f["docs"] == "syllabus" else "catalogue description only"
        lvl = None if f["level"] == "college" else "developmental (pre-college) level per source catalogue"
        if surface == "json":
            return dumps(prune({
                "student": {"name": f["student"], "id": f["sid"]},
                "incoming": {"institution": f["source"], "accreditation": ACCRED[f["accreditation"]][0],
                             "course": f"{f['in_code']} {f['in_title']}", "credits": cr, "lab": self._lab_txt(f),
                             "grade": f["grade"], "completed_years_ago": f["age_years"], "level": lvl},
                "home_course": f"{f['home_code']} {f['home_title']} ({f['home_credits']} cr{', with lab' if f['home_lab'] else ''})",
                "evaluation": {"outcomes_matched_pct": f["match_pct"], "basis": docs,
                               "transcript": "official" if f["official"] else "unofficial copy",
                               "faculty_sign_off": "yes" if f["signoff"] else "pending"},
                "policy": self._rules(f, rng),
            }))
        if surface == "evaluation_form":
            rows = [("Student", f"{f['student']} ({f['sid']})"), ("From", f"{f['source']} - {ACCRED[f['accreditation']][0]}"),
                    ("Incoming", f"{f['in_code']} {f['in_title']}, {cr}" + (f", {self._lab_txt(f)}" if f["lab_in"] is not None else "")),
                    ("Level", lvl), ("Grade / completed", f"{f['grade']} / {f['age_years']} yr ago"),
                    ("Requested as", f"{f['home_code']} {f['home_title']} ({f['home_credits']} cr)"),
                    ("Outcomes matched", f"{f['match_pct']}% (needs {f['match_min']}%)"),
                    ("Basis", docs), ("Transcript", "official" if f["official"] else "unofficial"),
                    ("Faculty sign-off", "yes" if f["signoff"] else "pending"),
                    ("Min. grade", f"{f['min_grade']} ({f['policy']})")]
            if f["science"]:
                rows.append(("Recency limit", f"{f['recency_years']} yrs (science/math/computing)"))
            return "TRANSFER CREDIT EVALUATION\n" + kv_lines(rows, rng) + "\n\n" + self._rules(f, rng)[1]
        para = [f"{f['student']} ({f['sid']}) wants {f['in_code']} {f['in_title']} from {f['source']} counted as "
                f"{f['home_code']} {f['home_title']} ({f['home_credits']} cr).",
                f"It carried {cr}" + (f", {self._lab_txt(f)}" if f["lab_in"] is not None else "") + f"; grade {f['grade']}, "
                f"taken {f['age_years']} years ago.",
                f"Source is {ACCRED[f['accreditation']][0]}.", ("Catalogue lists it as developmental." if lvl else ""),
                f"Comparing the {docs}, I count {f['match_pct']}% of our outcomes covered.",
                ("Transcript is an unofficial copy." if not f["official"] else ""),
                ("Department sign-off still pending." if not f["signoff"] else "")]
        return (f"Evaluator note - {f['evaluator']}, {fmt_date(f['date'], rng)}\n\n" + noisy(join_sentences(para), rng, 0.1)
                + "\n\n" + bullets(self._rules(f, rng), rng))

    def questions(self):
        return {
            "equivalency_ruling": choice("How should the incoming course be credited?", {
                "direct_equivalent": "As a direct equivalent of the home course.",
                "equivalent_with_shortfall": "As the home course, with the missing credit or lab made up separately.",
                "departmental_elective": "As elective credit in the same department, not as the home course.",
                "general_elective": "As general elective credit only.",
                "no_credit": "No credit at all.",
            }),
            "outcome_alignment": score("How closely does the incoming course cover the home course's learning outcomes?", [
                "Little overlap: different material.",
                "Partial: shared topics but core outcomes missing.",
                "Substantial: most outcomes covered, with gaps.",
                "Near-complete: essentially the same course.",
            ]),
            "evaluation_record_quality": score("How solid is the documentation behind this evaluation?", [
                "Thin: catalogue text, an unofficial transcript and no faculty review.",
                "Weak: only one of syllabus, official transcript and faculty sign-off is in place.",
                "Adequate: two of the three are in place.",
                "Complete: syllabus, official transcript and faculty sign-off.",
            ]),
            "grade_and_recency_ok": noul(
                "The grade earned and the age of the course both satisfy the policy's limits.",
                yes="The grade meets the minimum and, for a science, mathematics or computing course, it is within the recency limit.",
                no="The grade is below the minimum, or a science, mathematics or computing course is older than the limit."),
            "credit_after_conversion_sufficient": noul(
                "After conversion to semester credits the incoming course carries at least as much credit as the home course."),
        }


# =========================================================================== schema 2: accreditation finding
# standard -> (citation, statement template, direction, unit, threshold choices, typical spread)
STANDARDS = {
    "faculty_qualifications": ("Std 3.2", "at least {t}% of credit hours taught by faculty with a terminal degree or documented equivalent experience", ">=", "%", [70, 75, 80], 8),
    "assessment_cycle": ("Std 4.1", "at least {t}% of degree programs have completed a full assessment cycle (outcomes, measures, results, use of results) within three years", ">=", "%", [80, 85, 90], 9),
    "financial_reserves": ("Std 5.3", "unrestricted reserves of at least {t} months of operating expenses, or a Commission-approved recovery plan", ">=", " months", [3, 4, 6], 1.5),
    "licensure_pass": ("Std 7.2", "a three-year average first-time licensure pass rate of at least {t}%", ">=", "%", [75, 80, 85], 6),
    "clinical_ratio": ("Std 6.4", "no more than {t} students per clinical instructor in any placement", "<=", " students", [8, 10, 12], 2),
    "graduation_rate": ("Std 2.4", "a six-year graduation rate of at least {t}%", ">=", "%", [45, 50, 55], 6),
}
ACCREDITORS = ["Midland Commission on Higher Learning", "Northern Plains Council of Colleges", "Coastal Accrediting Commission",
               "Council for Allied Health Program Accreditation", "Board for Nursing Education Accreditation",
               "Western Association for Institutional Quality", "Commission on Technical and Career Programs"]
EVIDENCE = {"audited": "audited institutional data with source files", "ir_report": "institutional research report with source tables",
            "self_reported": "figure stated in the self-study; source data not provided"}


class AccreditationStandardFinding(Schema):
    name = "accreditation_standard_finding"
    description = "An accreditation review team's finding on one standard, given the institution's data, trend, evidence and history."
    surfaces = {"json": 0.5, "team_worksheet": 0.25, "chair_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng, clear=0.25, borderline=0.45, conflicting=0.30)
        std = pick(rng, list(STANDARDS))
        cite, tmpl, direc, unit, tchoices, spr = STANDARDS[std]
        t = pick(rng, tchoices)
        sign = 1 if direc == ">=" else -1
        if reg == "clear":
            cur = t + sign * pick(rng, [rng.uniform(1.2, 2.5), -rng.uniform(1.2, 2.5)]) * spr
        elif reg == "borderline":
            cur = near(rng, t, spr * 0.45)
        else:
            cur = t + sign * rng.uniform(-1.2, 1.2) * spr
        slope = pick_w(rng, {-1: 2, 0: 2.2, 0.5: 2.2, 1: 2, 2: 1}) * spr * rng.uniform(0.15, 0.4) * sign
        series = [cur - 2 * slope + rng.uniform(-0.15, 0.15) * spr, cur - slope + rng.uniform(-0.15, 0.15) * spr, cur]
        dig = 1 if unit in ("%", " months") else 0
        series = [round(max(0.0, x), dig) for x in series]
        if unit == " students":
            series = [max(4, int(round(x))) for x in series]
        evidence = pick_w(rng, {"audited": 4, "ir_report": 4, "self_reported": 1.6})
        prior = pick_w(rng, {"none": 4, "recommendation": 2, "non_compliance": 3.5})
        years_since = pick(rng, [1, 2, 2, 3, 3, 4, 5])
        corr = pick(rng, [2, 2, 3])
        plan = pick_w(rng, {"approved recovery plan": 1, "plan in draft": 1, None: 3}) if std == "financial_reserves" else None
        if reg == "borderline" and chance(rng, 0.3):
            prior, years_since = "non_compliance", corr + pick(rng, [-1, 0, 0, 1])
        if reg == "conflicting":
            twist = pick_w(rng, {"self_report_meets": 1, "repeat": 2, "plan": 1})
            if twist == "self_report_meets":
                evidence = "self_reported"
                series[-1] = round(t + sign * spr * rng.uniform(0.3, 1.2), dig) if unit != " students" else max(4, t - 1)
            elif twist == "repeat":
                prior, years_since = "non_compliance", corr + pick(rng, [0, 1])
                series[-1] = round(t - sign * spr * rng.uniform(0.1, 0.9), dig) if unit != " students" else t + pick(rng, [1, 2])
            elif std == "financial_reserves":
                plan = "approved recovery plan"
        return {
            "regime": reg, "standard": std, "cite": cite, "statement": tmpl.format(t=t), "direction": direc, "unit": unit,
            "threshold": t, "series": series, "evidence": evidence, "prior": prior, "years_since_prior": years_since,
            "correction_years": corr, "recovery_plan": plan, "accreditor": pick(rng, ACCREDITORS),
            "institution": _school(rng, ["University", "College", "State University", "Community College"]),
            "chair": person(rng), "year": pick(rng, [2025, 2026]), "policy": f"Policy {rng.randint(10, 49)}",
        }

    def _meets_value(self, f, v):
        return v >= f["threshold"] if f["direction"] == ">=" else v <= f["threshold"]

    def intended(self, f):
        cur = f["series"][-1]
        verifiable = f["evidence"] != "self_reported"
        meets_val = self._meets_value(f, cur) or f["recovery_plan"] == "approved recovery plan"
        meets = meets_val and verifiable
        spr = STANDARDS[f["standard"]][5]
        sign = 1 if f["direction"] == ">=" else -1
        margin = sign * (cur - f["threshold"]) / spr
        trend = sign * (f["series"][-1] - f["series"][0]) / spr
        persistent = (f["prior"] == "non_compliance" and f["years_since_prior"] >= f["correction_years"] and not meets)
        if meets:
            finding = ("complies_with_recommendation" if (margin < 0.35 or trend < -0.15 or f["recovery_plan"]
                                                           or f["prior"] == "non_compliance") else "complies")
        else:
            finding = "non_compliant_warning" if persistent else "non_compliant_monitoring"
        if not verifiable or finding == "non_compliant_warning":
            follow = "focused_site_visit"
        elif finding == "non_compliant_monitoring":
            follow = "monitoring_report"
        elif finding == "complies_with_recommendation":
            follow = "annual_report_update"
        else:
            follow = "no_follow_up"
        gap = 0 if margin >= 0.6 else 1 if margin >= 0 else 2 if margin >= -0.4 else 3 if margin >= -1.0 else 4
        traj = 0 if trend < -0.1 else 1 if trend < 0.12 else 2 if trend < 0.45 else 3
        return {"standard_finding": finding, "follow_up_required": follow, "distance_from_standard": gap,
                "improvement_trajectory": traj, "deficiency_persists": "yes" if persistent else "no"}

    def _v(self, f, v):
        return f"{v}{f['unit']}"

    def _rules(self, f):
        return [f"{f['cite']}: the institution demonstrates {f['statement']}.",
                f"{f['policy']}: a non-compliance finding must be corrected within {f['correction_years']} years; if it "
                "persists past that period the institution is placed on Warning. Figures without source data do not demonstrate compliance."]

    def _history(self, f):
        if f["prior"] == "none":
            return "no prior finding on this standard"
        what = "a recommendation" if f["prior"] == "recommendation" else "a non-compliance finding"
        y = f["years_since_prior"]
        return f"{what} on this standard {y} year{'s' if y > 1 else ''} ago"

    def render(self, f, rng, surface):
        s = f["series"]
        yrs = [f["year"] - 2, f["year"] - 1, f["year"]]
        if surface == "json":
            return dumps(prune({
                "institution": f["institution"], "accreditor": f["accreditor"],
                "standard": self._rules(f)[0], "data": {str(y): self._v(f, v) for y, v in zip(yrs, s)},
                "evidence": EVIDENCE[f["evidence"]], "history": self._history(f), "recovery_plan": f["recovery_plan"],
                "policy": self._rules(f)[1],
            }))
        if surface == "team_worksheet":
            rows = [[str(y), self._v(f, v)] for y, v in zip(yrs, s)]
            head = kv_lines([("Institution", f["institution"]), ("Standard", f"{f['cite']} - required: {f['direction']} {f['threshold']}{f['unit']}"),
                             ("Evidence", EVIDENCE[f["evidence"]]), ("History", self._history(f)),
                             ("Recovery plan", f["recovery_plan"]),
                             ("Correction period", f"{f['correction_years']} yrs, then Warning ({f['policy']})")], rng)
            return f"{f['accreditor'].upper()} - TEAM WORKSHEET\n{head}\n\n" + table(["year", "value"], rows, rng)
        para = [f"For {f['cite']} the team looked at {f['institution']}'s figures: "
                + ", ".join(f"{self._v(f, v)} in {y}" for y, v in zip(yrs, s)) + ".",
                f"Evidence offered: {EVIDENCE[f['evidence']]}.", f"History: {self._history(f)}.",
                (f"Recovery plan status: {f['recovery_plan']}." if f["recovery_plan"] else ""),
                pick(rng, ["The institution's response stresses recent investment.", "Staff were candid about the gap.",
                           "The self-study describes the situation as stable.", ""])]
        return (f"Memo from {f['chair']}, team chair, {f['accreditor']}\n\n" + noisy(join_sentences(para), rng, 0.1)
                + "\n\n" + "\n".join(self._rules(f)))

    def questions(self):
        return {
            "standard_finding": choice("What finding should the team record on this standard?", {
                "complies": "Complies with the standard.",
                "complies_with_recommendation": "Complies, with a recommendation for improvement.",
                "non_compliant_monitoring": "Does not comply; correction to be monitored.",
                "non_compliant_warning": "Does not comply after the correction period; place the institution on Warning.",
            }),
            "follow_up_required": choice("What follow-up should the accreditor require?", {
                "no_follow_up": "None until the next scheduled review.",
                "annual_report_update": "An update in the institution's next annual report.",
                "monitoring_report": "A monitoring report showing correction within a set period.",
                "focused_site_visit": "A focused site visit to verify the evidence.",
            }),
            "distance_from_standard": score("How far is the institution from meeting the standard?", [
                "Comfortably meets it.",
                "Just meets it.",
                "Narrowly short.",
                "Clearly short.",
                "Far short.",
            ]),
            "improvement_trajectory": score("How is the institution's performance on this standard moving?", [
                "Getting worse.",
                "Flat.",
                "Improving slowly.",
                "Improving clearly.",
            ]),
            "deficiency_persists": noul(
                "A deficiency on this standard has persisted beyond the correction period the accreditor allows.",
                yes="It was found non-compliant before, the correction period has run out, and it still does not demonstrate compliance.",
                no="There was no earlier non-compliance, the period has not run out, or it now demonstrates compliance."),
        }


# =========================================================================== schema 3: program review
# (program, degree level)
PROGRAMS = [("BA French", "bachelor"), ("BS Physics", "bachelor"), ("AAS Welding Technology", "associate"),
            ("MS Applied Statistics", "master"), ("BFA Ceramics", "bachelor"), ("BS Medical Laboratory Science", "bachelor"),
            ("BS Agricultural Economics", "bachelor"), ("AAS Paralegal Studies", "associate"), ("MA Public History", "master"),
            ("BS Geology", "bachelor"), ("AS Fire Science", "associate"), ("BA Theatre Arts", "bachelor"),
            ("MEd Educational Leadership", "master"), ("BS Actuarial Science", "bachelor"), ("AAS Diesel Technology", "associate"),
            ("BA Classics", "bachelor"), ("BS Nuclear Medicine Technology", "bachelor"), ("MS Forestry", "master"),
            ("BA German", "bachelor"), ("AAS Surgical Technology", "associate"), ("BS Dairy Science", "bachelor"),
            ("MA Philosophy", "master")]
THRESH = {"associate": [8, 10], "bachelor": [10, 12, 15], "master": [5, 6, 8]}


class AcademicProgramReview(Schema):
    name = "academic_program_review"
    description = "A state system's periodic review of one degree program against its productivity rule: continue, revise, consolidate, sunset or wait for data."
    surfaces = {"json": 0.5, "review_summary": 0.25, "dashboard": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng, clear=0.25, borderline=0.45, conflicting=0.30)
        prog, lvl = pick(rng, PROGRAMS)
        thr = pick(rng, THRESH[lvl])
        if reg == "clear":
            avg = thr * pick(rng, [rng.uniform(1.4, 2.5), rng.uniform(0.3, 0.6)])
        elif reg == "borderline":
            avg = near(rng, thr, thr * 0.18, lo=1)
        else:
            avg = thr * rng.uniform(0.5, 1.3)
        grads = [max(0, round(avg + rng.uniform(-0.25, 0.25) * thr)) for _ in range(2)]
        grads.append(max(0, round(3 * avg - sum(grads))))
        majors_now = max(4, round(avg * rng.uniform(3.0, 5.5)))
        trend = pick_w(rng, {rng.uniform(0.12, 0.45): 2, rng.uniform(-0.08, 0.08): 3, rng.uniform(-0.45, -0.2): 2.5})
        majors_then = max(3, round(majors_now / (1 + trend)))
        outlook = pick_w(rng, {"growing": 2.5, "stable": 3, "declining": 2})
        place = pick_w(rng, {round(rng.uniform(86, 97)): 3, round(rng.uniform(72, 85)): 3, round(rng.uniform(60, 71)): 2,
                             round(rng.uniform(42, 59)): 1.5, None: 1})
        resp = rng.randint(18, 35) if place is None else rng.randint(45, 88)
        only = chance(rng, 0.15)
        accred_dep = chance(rng, 0.1)
        related = pick(rng, [None, "a related program in the same college"]) if not only else None
        missing_year = chance(rng, 0.14)
        if reg == "conflicting":
            twist = pick(rng, ["growing_but_low", "exempt", "strong_outcomes_low", "declining_ok"])
            if twist == "growing_but_low":
                avg_ = thr * rng.uniform(0.55, 0.9)
                grads = [max(0, round(avg_ * x)) for x in (0.8, 1.0, 1.2)]
                majors_then, outlook = max(3, round(majors_now / 1.35)), "growing"
            elif twist == "exempt":
                only = True
                related = None
            elif twist == "strong_outcomes_low":
                place, resp = round(rng.uniform(88, 98)), rng.randint(55, 85)
            else:
                majors_then, outlook = round(majors_now * 1.5), "declining"
        return {
            "regime": reg, "program": prog, "level": lvl, "threshold": thr, "grads": grads, "missing_year": missing_year,
            "majors_then": majors_then, "majors_now": majors_now, "outlook": outlook, "placement": place,
            "survey_response": resp, "cost_ratio": round(rng.uniform(0.7, 1.9), 2), "service_share": rng.randint(5, 70),
            "only_in_system": only, "accreditation_dependency": accred_dep, "related_program": related,
            "institution": _school(rng, ["Community College", "Technical College", "State College"] if lvl == "associate"
                                   else ["University", "State University", "College"]),
            "board_policy": f"Board Policy {rng.randint(3, 6)}.{rng.randint(10, 40)}", "chair": person(rng),
            "year": pick(rng, [2025, 2026]),
        }

    def _avg(self, f):
        g = f["grads"][1:] if f["missing_year"] else f["grads"]
        return sum(g) / len(g)

    def intended(self, f):
        below = self._avg(f) < f["threshold"]
        exempt = f["only_in_system"] or f["accreditation_dependency"]
        trend = (f["majors_now"] - f["majors_then"]) / f["majors_then"]
        pts = (trend >= 0.1) - (trend <= -0.2) + (f["outlook"] == "growing") - (f["outlook"] == "declining")
        pts += f["majors_now"] >= 4 * f["threshold"]
        demand = 0 if pts <= -1 else 1 if pts == 0 else 2 if pts == 1 else 3
        p = f["placement"]
        outc = 1 if p is None else 3 if p >= 85 else 2 if p >= 72 else 1 if p >= 60 else 0
        if f["missing_year"] and abs(self._avg(f) - f["threshold"]) <= 0.15 * f["threshold"]:
            rec = "defer_pending_data"
        elif not below:
            rec = "continue" if (outc >= 1 and demand >= 1) else "continue_with_action_plan"
        elif exempt or demand >= 2:
            rec = "continue_with_action_plan"
        elif f["related_program"]:
            rec = "consolidate_with_related_program"
        else:
            rec = "sunset_with_teach_out"
        return {"review_recommendation": rec, "demand_strength": demand, "graduate_outcomes": outc,
                "below_productivity_threshold": "yes" if below else "no", "exemption_applies": "yes" if exempt else "no"}

    def _rules(self, f, rng):
        lvl = f["level"]
        r = [f"{f['board_policy']} §2: a {lvl}'s program whose three-year average of graduates is below {f['threshold']} must "
             "be put on an action plan, consolidated or closed.",
             "§3: exempt from closure under §2 (an action plan is still required): the only program of its kind in the "
             "system, or one another program needs for its accreditation."]
        if chance(rng, 0.5):
            r.append("§4: a closure needs a teach-out plan for enrolled students.")
        return r

    def _g(self, f, i, rng):
        if i == 0 and f["missing_year"]:
            return pick(rng, ["not reported", "n/a", "data pending"])
        return f["grads"][i]

    def render(self, f, rng, surface):
        y = f["year"]
        g = {str(y - 3 + i): self._g(f, i, rng) for i in range(3)}
        place = (f"{f['placement']}% employed or in further study (survey response {f['survey_response']}%)" if f["placement"]
                 else f"not reliable (survey response {f['survey_response']}%)")
        flags = [x for x in [("only program of its kind in the system" if f["only_in_system"] else None),
                             ("required for the accreditation of another program" if f["accreditation_dependency"] else None),
                             f["related_program"]] if x]
        if surface == "json":
            return dumps(prune({
                "program": f["program"], "institution": f["institution"], "graduates_by_year": g,
                "majors": {str(y - 5): f["majors_then"], str(y): f["majors_now"]}, "labour_market_outlook": f["outlook"],
                "graduate_placement": place, "cost_per_credit_hour_vs_median": f"{f['cost_ratio']}x",
                "credit_hours_taught_to_non_majors_pct": f["service_share"], "notes": flags or None,
                "policy": self._rules(f, rng),
            }))
        if surface == "dashboard":
            rows = [[k, str(v)] for k, v in g.items()]
            head = (f"PROGRAM DASHBOARD - {f['program']}, {f['institution']}\nProductivity floor: {f['threshold']} "
                    f"graduates/yr, 3-yr avg ({f['board_policy']} §2; exempt if only program in system or needed for "
                    "another program's accreditation, §3)")
            tail = kv_lines([("Majors", f"{f['majors_then']} ({y - 5}) -> {f['majors_now']} ({y})"), ("Outlook", f["outlook"]),
                             ("Placement", place), ("Cost/credit hr", f"{f['cost_ratio']}x median"),
                             ("Service teaching", f"{f['service_share']}% of credit hours"),
                             ("Notes", "; ".join(flags) if flags else None)], rng)
            return head + "\n\n" + table(["year", "graduates"], rows, rng) + "\n\n" + tail
        para = [f"{f['program']} at {f['institution']} graduated " + ", ".join(f"{v} in {k}" for k, v in g.items()) + ".",
                f"Majors moved from {f['majors_then']} to {f['majors_now']} over five years; the regional labour-market outlook is {f['outlook']}.",
                f"Placement: {place}.", f"Cost per credit hour is {f['cost_ratio']} times the institutional median, and "
                f"{f['service_share']}% of the department's credit hours go to non-majors.",
                ("Note: " + "; ".join(flags) + "." if flags else "")]
        return (f"Program review summary ({f['year']}), prepared by {f['chair']}\n\n" + noisy(join_sentences(para), rng, 0.1)
                + "\n\n" + bullets(self._rules(f, rng), rng))

    def questions(self):
        return {
            "review_recommendation": choice("What should the review recommend for this program?", {
                "continue": "Continue the program as it is.",
                "continue_with_action_plan": "Continue it under an action plan with targets and a review date.",
                "consolidate_with_related_program": "Fold it into a related program.",
                "sunset_with_teach_out": "Close it to new students and teach out those enrolled.",
                "defer_pending_data": "Defer the recommendation until the missing data are supplied.",
            }),
            "demand_strength": score("How strong is student and labour-market demand for the program?", [
                "Weak and falling.",
                "Soft.",
                "Healthy.",
                "Strong and growing.",
            ]),
            "graduate_outcomes": score("How good are the program's graduate outcomes on the data available?", [
                "Poor.",
                "Mediocre, or impossible to judge from the data.",
                "Good.",
                "Excellent.",
            ]),
            "below_productivity_threshold": noul(
                "The program falls below the system's minimum degree-productivity threshold.",
                yes="Its three-year average of graduates is under the floor for its degree level.",
                no="Its three-year average meets or exceeds the floor."),
            "exemption_applies": noul(
                "The program qualifies for one of the policy's exemptions from closure under the productivity rule."),
        }


# =========================================================================== schema 4: course substitution
MAJORS = {
    "BS Mechanical Engineering": [("ME 3410 Heat Transfer", 3, True), ("ME 3200 Machine Design", 3, True), ("ME 4950 Senior Design I", 3, True)],
    "BA Psychology": [("PSYC 3010 Research Methods", 4, False), ("PSYC 3420 Cognitive Psychology", 3, False), ("PSYC 4900 Capstone Seminar", 3, False)],
    "BS Nursing": [("NURS 3300 Pharmacology", 3, True), ("NURS 3600 Community Health", 3, True), ("NURS 3150 Health Assessment", 3, True)],
    "BA Elementary Education": [("EDUC 3250 Teaching Reading", 3, True), ("EDUC 3400 Classroom Management", 3, True), ("EDUC 3100 Child Development", 3, False)],
    "BS Accounting": [("ACCT 3100 Intermediate Accounting I", 3, False), ("ACCT 4300 Auditing", 3, False), ("ACCT 3500 Cost Accounting", 3, False)],
    "BS Biology": [("BIOL 3320 Genetics", 4, False), ("BIOL 3500 Ecology", 4, False), ("BIOL 4100 Cell Biology", 3, False)],
    "BA Economics": [("ECON 3010 Intermediate Microeconomics", 3, False), ("ECON 3200 Econometrics", 3, False), ("ECON 4400 Public Finance", 3, False)],
    "BS Computer Science": [("CS 3500 Operating Systems", 3, False), ("CS 3300 Algorithms", 3, False), ("CS 4400 Database Systems", 3, False)],
}
SUBS = {  # substitute descriptions by reason and department
    "abroad": "{d} {lvl}{n}0 course taken on study abroad",
    "transfer": "{d} {lvl}{n}0 course completed before transferring",
    "same": ["{d} {lvl}{n}0 special topics course", "{d} {lvl}{n}0 graduate cross-listed course", "{d} {lvl}{n}0 seminar"],
    "other": ["{d} {lvl}{n}0 course from a neighbouring department", "{d} {lvl}{n}0 interdisciplinary course"],
}
NEIGHBOURS = {"ME": ["PHYS", "MATH", "MSE"], "PSYC": ["SOC", "BIOL", "STAT"], "NURS": ["BIOL", "HLTH", "PHAR"],
              "EDUC": ["PSYC", "SOC", "ENGL"], "ACCT": ["FIN", "BUS", "ECON"], "BIOL": ["CHEM", "ENVS", "BIOC"],
              "ECON": ["STAT", "FIN", "POLS"], "CS": ["MATH", "ECE", "STAT"]}
REASONS = {"not_offered": "the required course is not offered before the student's planned graduation",
           "conflict": "the required course clashes with another required course every term it runs",
           "abroad": "the student took a similar course while studying abroad",
           "transfer": "a similar course was completed before transferring in"}


class CourseSubstitutionPetition(Schema):
    name = "course_substitution_petition"
    description = "A student's petition to substitute a completed course for a required major course, under the written substitution regulation."
    surfaces = {"json": 0.5, "petition_form": 0.25, "advisor_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng, clear=0.25, borderline=0.45, conflicting=0.30)
        major = pick(rng, list(MAJORS))
        req, req_cr, lic = pick(rng, MAJORS[major])
        dept = req.split()[0]
        req_level = int(req.split()[1][0])
        same = chance(rng, 0.65)
        sub_level = pick_w(rng, {req_level: 5, req_level + 1: 1.5, req_level - 1: 1.5}) if req_level < 4 else pick_w(rng, {4: 6, 3: 1.5})
        sub_cr = pick_w(rng, {req_cr: 6, req_cr - 1: 1.5, req_cr + 1: 1})
        m = pick(rng, [60, 70, 75])
        cap = pick(rng, [6, 9])
        overlap = rng.randint(30, 98)
        prior = pick_w(rng, {0: 4, 3: 3, cap - 3: 2, cap - 1: 1, cap: 1})
        if reg == "borderline":
            if chance(rng, 0.5):
                overlap = near(rng, m - 4, 10, lo=10, hi=100, integer=True)
            else:
                prior = max(0, cap - sub_cr + pick(rng, [0, 0, 1, 1, 2]))
        advisor_signed = chance(rng, 0.88)
        syllabus = chance(rng, 0.9)
        terms = pick_w(rng, {0: 2.5, 1: 3, 2: 2, 3: 1.2, 4: 0.8})
        late = terms == 0 and chance(rng, 0.35)
        audit_days = rng.randint(4, 40) if terms == 0 else None
        offered_next = chance(rng, 0.5)
        reason = pick_w(rng, {"not_offered": 2, "conflict": 1, "abroad": 1, "transfer": 1})
        if reg == "clear":
            if chance(rng, 0.5):
                overlap, prior, advisor_signed, syllabus, sub_level, lic = rng.randint(m + 10, 98), 0, True, True, max(sub_level, req_level), False
            else:
                overlap, sub_level = rng.randint(20, m - 25), min(sub_level, req_level - 1)
        elif reg == "conflicting":
            twist = pick(rng, ["licensure", "late_strong", "over_cap", "unsigned"])
            if twist == "licensure":
                lic, overlap = True, rng.randint(m + 5, 98)
            elif twist == "late_strong":
                terms, late, audit_days, overlap = 0, True, rng.randint(3, 12), rng.randint(m + 5, 98)
            elif twist == "over_cap":
                prior, overlap = cap - sub_cr + pick(rng, [1, 2, 3]), rng.randint(m + 5, 98)
            else:
                advisor_signed, overlap = False, rng.randint(m, 98)
        return {
            "regime": reg, "major": major, "required": req, "req_credits": req_cr, "req_level": req_level,
            "licensure_required": lic, "same_department": same, "sub_level": sub_level, "sub_credits": sub_cr,
            "overlap_pct": overlap, "overlap_min": m, "cap": cap, "prior_subs": max(0, prior), "advisor_signed": advisor_signed,
            "syllabus": syllabus, "terms_to_grad": terms, "filed_after_census": late, "audit_days": audit_days,
            "offered_next_term": offered_next, "reason": reason,
            "sub_title": (SUBS[reason] if reason in ("abroad", "transfer") else pick(rng, SUBS["same" if same else "other"])).format(
                d=dept if same else pick(rng, NEIGHBOURS[dept]),
                lvl=sub_level, n=rng.randint(10, 99)),
            "student": person(rng), "sid": ident(rng, "S", 7, ""), "advisor": person(rng),
            "reg": f"AR-{rng.randint(10, 39)}", "date": rand_date(rng),
        }

    def intended(self, f):
        within = f["prior_subs"] + f["sub_credits"] <= f["cap"]
        if not f["advisor_signed"] or not f["syllabus"]:
            ruling = "return_incomplete"
        elif not within or f["sub_level"] < f["req_level"]:
            ruling = "deny_offer_directed_study" if f["reason"] == "not_offered" and f["terms_to_grad"] <= 1 else "deny"
        elif f["overlap_pct"] >= f["overlap_min"]:
            ruling = "approve"
        elif f["overlap_pct"] >= f["overlap_min"] - 15:
            ruling = "approve_with_supplementary_work"
        elif f["reason"] == "not_offered" and f["terms_to_grad"] <= 1:
            ruling = "deny_offer_directed_study"
        else:
            ruling = "deny"
        if f["licensure_required"]:
            route = "accreditation_coordinator"
        elif f["filed_after_census"]:
            route = "academic_standards_committee"
        elif not f["same_department"]:
            route = "college_dean"
        else:
            route = "department_chair"
        o = f["overlap_pct"]
        ov = 0 if o < 40 else 1 if o < 60 else 2 if o < 80 else 3
        t = f["terms_to_grad"]
        if t == 0:
            tp = 3 if (f["audit_days"] or 99) <= 21 else 2
        elif t == 1:
            tp = 1 if f["offered_next_term"] else 2
        else:
            tp = 1 if t == 2 else 0
        return {"petition_ruling": ruling, "approval_route": route, "content_overlap": ov, "graduation_timing_pressure": tp,
                "within_substitution_cap": "yes" if within else "no"}

    def _rules(self, f):
        return [f"{f['reg']} §2: substitutions may total no more than {f['cap']} credits of major requirements; the substitute "
                f"must be at the same or a higher level and cover at least {f['overlap_min']}% of the required course's outcomes.",
                f"{f['reg']} §3-5: licensure- or accreditation-required courses need the program accreditation coordinator's "
                "approval; final-term petitions filed after census go to the Academic Standards Committee; substitutes from "
                "another department need the dean's signature."]

    def _timing(self, f):
        if f["terms_to_grad"] == 0:
            return f"final term; degree audit closes in {f['audit_days']} days"
        nxt = "offered next term" if f["offered_next_term"] else "not offered next term"
        when = "after next term" if f["terms_to_grad"] == 1 else f"in {f['terms_to_grad']} terms"
        return f"graduates {when}; required course {nxt}"

    def render(self, f, rng, surface):
        sub = f"{f['sub_title']} ({f['sub_credits']} cr)"
        req = f"{f['required']} ({f['req_credits']} cr" + (", licensure/accreditation-required)" if f["licensure_required"] else ")")
        if surface == "json":
            return dumps(prune({
                "student": {"name": f["student"], "id": f["sid"], "major": f["major"]},
                "petition": {"required_course": req, "substitute": sub, "reason": REASONS[f["reason"]],
                             "outcome_overlap_pct": f["overlap_pct"], "syllabus_attached": f["syllabus"],
                             "filed": "after census" if f["filed_after_census"] else "before census"},
                "record": {"major_credits_already_substituted": f["prior_subs"], "timing": self._timing(f)},
                "signatures": {"advisor": "signed" if f["advisor_signed"] else "missing"},
                "regulation": self._rules(f),
            }))
        if surface == "petition_form":
            rows = [("Student", f"{f['student']} ({f['sid']}), {f['major']}"), ("Required course", req), ("Substitute", sub),
                    ("Reason", REASONS[f["reason"]]), ("Outcome overlap", f"{f['overlap_pct']}% (min {f['overlap_min']}%)"),
                    ("Credits already substituted", f"{f['prior_subs']} of {f['cap']} allowed ({f['reg']} §2)"),
                    ("Syllabus", "attached" if f["syllabus"] else "not attached"), ("Timing", self._timing(f)),
                    ("Filed", "after census" if f["filed_after_census"] else "before census"),
                    ("Advisor signature", "yes" if f["advisor_signed"] else "missing")]
            return "COURSE SUBSTITUTION PETITION\n" + kv_lines(rows, rng) + "\n\n" + self._rules(f)[1]
        body = [f"Subject: Substitution petition - {f['student']} ({f['sid']})", f"From: {f['advisor']}", "", "Hi,", "",
                noisy(join_sentences([
                    f"{f['student']} ({f['major']}) asks to use {sub} in place of {req}.",
                    f"Reason: {REASONS[f['reason']]}.", f"Comparing syllabi I'd put the outcome overlap at about {f['overlap_pct']}%."
                    if f["syllabus"] else "The substitute's syllabus is not attached yet.",
                    f"The student has already substituted {f['prior_subs']} major credits.", f"Timing: {self._timing(f)}.",
                    ("The petition came in after census." if f["filed_after_census"] else ""),
                    ("I have signed the form." if f["advisor_signed"] else "I have not signed the form yet.")]), rng, 0.1),
                "", bullets(self._rules(f), rng)]
        return "\n".join(body)

    def questions(self):
        return {
            "petition_ruling": choice("How should the substitution petition be decided?", {
                "approve": "Approve the substitution.",
                "approve_with_supplementary_work": "Approve it once the student completes set work covering the missing outcomes.",
                "deny": "Deny it; the student must take the required course.",
                "deny_offer_directed_study": "Deny it, but arrange a directed study of the required course so graduation is not delayed.",
                "return_incomplete": "Return it unprocessed until the missing signature or syllabus is supplied.",
            }),
            "approval_route": choice("Who must give the final approval on this petition?", {
                "department_chair": "The department chair.",
                "college_dean": "The college dean.",
                "accreditation_coordinator": "The program's accreditation coordinator.",
                "academic_standards_committee": "The Academic Standards Committee.",
            }),
            "content_overlap": score("How much of the required course's content does the substitute cover?", [
                "Little.",
                "Some, with major gaps.",
                "Most of it.",
                "Nearly all of it.",
            ]),
            "graduation_timing_pressure": score("How much time pressure does the student's graduation timeline put on this petition?", [
                "None: graduation is a year or more away.",
                "Some: it must be settled within the next two terms.",
                "High: graduation is close and the required course is not available in time.",
                "Acute: final term, with the degree audit closing within three weeks.",
            ]),
            "within_substitution_cap": noul(
                "Approving this petition would keep the student within the regulation's cap on substituted credits.",
                yes="Credits already substituted plus this course's credits do not exceed the cap.",
                no="Together they would exceed the cap."),
        }


# =========================================================================== schema 5: prior learning assessment
# (course, credits, field, plausible experience)
PLA_COURSES = [("BUS 2300 Project Management", 3, "IT/business", ["nine years as an operations supervisor", "six years coordinating construction projects"]),
               ("IT 1500 Networking Fundamentals", 3, "IT/business", ["four years as a network technician", "military service as a communications specialist"]),
               ("HSA 2200 Medical Coding", 3, "health", ["six years in a hospital billing office", "five years as a clinic coding clerk"]),
               ("ACCT 1100 Bookkeeping", 3, "IT/business", ["a decade keeping the books for a family business", "seven years as an accounts clerk"]),
               ("CAD 1200 Technical Drawing", 3, "technical", ["eight years as a drafting technician", "six years in a fabrication shop's design office"]),
               ("MKT 2100 Principles of Marketing", 3, "IT/business", ["five years running social media for a retailer", "seven years as a sales coordinator"]),
               ("WLD 1400 Structural Welding", 4, "technical", ["five years as a lead welder", "military service as a metalworker"]),
               ("ECE 1300 Early Childhood Curriculum", 3, "education", ["eight years in childcare leadership", "six years as a preschool assistant teacher"]),
               ("HOSP 2100 Food and Beverage Operations", 3, "technical", ["seven years managing a restaurant kitchen", "five years as a banquet captain"]),
               ("LOG 2200 Supply Chain Operations", 3, "IT/business", ["military service as a logistics specialist", "six years as a warehouse shift lead"]),
               ("EMS 1500 Emergency Care Fundamentals", 4, "health", ["six years as a volunteer firefighter-EMT", "four years as a ski patroller"]),
               ("IT 2400 Cloud Administration", 3, "IT/business", ["four years as a systems administrator", "three years on a hosting company's support desk"])]
CERTS = {'BUS 2300 Project Management': 'Certified Project Practitioner (CPP)',
         'IT 1500 Networking Fundamentals': 'Certified Network Associate (CNA-2)',
         'HSA 2200 Medical Coding': 'Professional Coder Credential (PCC)',
         'ACCT 1100 Bookkeeping': 'Certified Bookkeeping Technician (CBT)',
         'CAD 1200 Technical Drawing': 'Associate Drafter Credential (ADC)',
         'MKT 2100 Principles of Marketing': 'Digital Marketing Associate (DMA)',
         'WLD 1400 Structural Welding': 'Structural Welder Level II',
         'ECE 1300 Early Childhood Curriculum': 'Child Development Associate Credential (CDAC)',
         'HOSP 2100 Food and Beverage Operations': 'Food Safety Manager Credential',
         'LOG 2200 Supply Chain Operations': 'Certified Logistics Associate (CLA)',
         'EMS 1500 Emergency Care Fundamentals': 'Emergency Medical Responder Certificate',
         'IT 2400 Cloud Administration': 'Cloud Operations Associate'}


class PriorLearningAssessment(Schema):
    name = "prior_learning_assessment"
    description = "A request for academic credit for prior learning (portfolio, challenge exam or certification) under the written PLA policy."
    surfaces = {"json": 0.5, "assessment_record": 0.25, "coordinator_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng, clear=0.25, borderline=0.45, conflicting=0.30)
        course, cr, field, exps = pick(rng, PLA_COURSES)
        n_out = pick(rng, [4, 5, 6])
        cut = pick(rng, [70, 75, 80])
        cur = 5 if field in ("IT/business", "health") else pick(rng, [7, 10])
        cap = pick(rng, [15, 24, 30])
        # the route the facts are built around; borderline values near a line may still tip the case
        scen = pick_w(rng, {"course": 2.2, "elective": 1.5, "evidence": 1.4, "exam_offer": 1.3, "over_cap": 3.4, "short": 1.5})
        method = pick_w(rng, {"portfolio": 5, "exam": 2.3, "certification": 3})
        if scen == "elective" and method == "exam":
            method = pick(rng, ["portfolio", "certification"])
        if scen == "short" and method == "certification":
            method = pick(rng, ["portfolio", "exam"])
        room = cap - cr
        if scen == "over_cap":
            awarded = room + (pick(rng, [1, 2]) if reg == "borderline" else rng.randint(1, 6))
        elif reg == "borderline" and chance(rng, 0.4):
            awarded = room + pick(rng, [-2, -1, 0, 0, 1])
        else:
            awarded = pick_w(rng, {0: 4, rng.randint(3, 12): 3.5, rng.randint(max(0, room - 6), room): 1.5})
        since = rng.randint(0, cur)
        if (scen == "short" and method == "portfolio") or (method != "portfolio" and chance(rng, 0.55)) or chance(rng, 0.15):
            since = cur + (pick(rng, [1, 2]) if reg == "borderline" else rng.randint(1, 8))
        elif reg == "borderline" and method == "portfolio":
            since = cur - pick(rng, [0, 0, 1])
        verif = pick_w(rng, {"signed employer letter on letterhead": 2, "employer letter plus licensing-body record": 1})
        a = b = n_out
        score_ = exam_mode = cert = cert_status = cert_verified = None
        if method == "portfolio":
            if scen == "evidence":
                verif = pick(rng, ["unsigned statement", "none", "none"])
                a = b = pick(rng, [n_out, n_out, n_out - 1])
            elif scen == "elective":
                a = pick(rng, [n_out - 1, n_out - 2]) if n_out > 4 else n_out - 1
                b = a if chance(rng, 0.7) else max(n_out // 2, a - 1)
            elif scen == "exam_offer":
                a, b = n_out, n_out - pick(rng, [1, 2])
            elif scen == "short" and chance(rng, 0.6):
                since = rng.randint(0, cur)
                a = b = max(1, n_out // 2 - 1)
            elif scen in ("over_cap",):
                a = b = pick(rng, [n_out, n_out, n_out - 1])
                verif = pick_w(rng, {"signed employer letter on letterhead": 3, "employer letter plus licensing-body record": 1,
                                     "unsigned statement": 1, "none": 1})
        elif method == "exam":
            exam_mode = ("online, unproctored" if scen == "exam_offer"
                         else pick(rng, ["proctored at the testing centre", "remotely proctored (webcam)"]))
            if scen == "short":
                score_ = (cut - pick(rng, [1, 2, 3])) if reg == "borderline" else rng.randint(40, cut - 9)
            else:
                score_ = (cut + pick(rng, [0, 1, 2, 4])) if reg == "borderline" else rng.randint(cut + 4, 98)
        else:
            cert = CERTS[course]
            cert_status = {"course": "current, mapped to this course", "elective": "current, mapped to elective credit only",
                           "exam_offer": "expired"}.get(scen) or pick(rng, ["current, mapped to this course",
                                                                            "current, mapped to elective credit only",
                                                                            "not on the crosswalk"])
            cert_verified = scen != "evidence" or (cert_status == "not on the crosswalk" and chance(rng, 0.5))
            if scen == "evidence" and cert_status != "not on the crosswalk":
                cert_verified = False
        if reg == "conflicting" and scen == "course" and method == "portfolio":
            b = n_out - 1 if chance(rng, 0.5) else b
        return {
            "regime": reg, "scenario": scen, "course": course, "credits": cr, "field": field, "method": method,
            "n_outcomes": n_out, "met_a": a, "met_b": b, "exam_score": score_, "exam_mode": exam_mode, "cut_score": cut,
            "cert": cert, "cert_status": cert_status, "cert_verified": cert_verified, "currency_years": cur,
            "years_since_practice": since, "verification": verif, "cap": cap, "already_awarded": max(0, awarded),
            "student": person(rng), "sid": ident(rng, "S", 7, ""), "coordinator": person(rng),
            "policy": f"PLA-{rng.randint(2, 9)}", "date": rand_date(rng), "experience": pick(rng, exps),
        }

    def intended(self, f):
        within = f["already_awarded"] + f["credits"] <= f["cap"]
        current = f["years_since_practice"] <= f["currency_years"]
        verified = f["verification"] in ("signed employer letter on letterhead", "employer letter plus licensing-body record")
        m, n = f["method"], f["n_outcomes"]
        lo = min(f["met_a"], f["met_b"])
        if not within:
            award = "deny_credit"
        elif m == "exam":
            if f["exam_mode"] == "online, unproctored":
                award = "offer_challenge_exam"
            else:
                award = "award_course_credit" if f["exam_score"] >= f["cut_score"] else "deny_credit"
        elif m == "certification":
            st = f["cert_status"]
            if st == "expired":
                award = "offer_challenge_exam"
            elif st == "not on the crosswalk" or not f["cert_verified"]:
                award = "request_further_evidence"
            else:
                award = "award_course_credit" if st == "current, mapped to this course" else "award_elective_credit"
        elif not current:
            award = "deny_credit"
        elif not verified:
            award = "request_further_evidence"
        elif lo == n:
            award = "award_course_credit"
        elif max(f["met_a"], f["met_b"]) == n:
            award = "offer_challenge_exam"
        elif lo * 2 >= n:
            award = "award_elective_credit"
        else:
            award = "deny_credit"
        if m == "portfolio":
            share = lo / n
            comp = 3 if share == 1 else 2 if share >= 0.75 else 1 if share >= 0.5 else 0
            ver = {"employer letter plus licensing-body record": 3, "signed employer letter on letterhead": 2,
                   "unsigned statement": 1, "none": 0}[f["verification"]]
        elif m == "exam":
            d = f["exam_score"] - f["cut_score"]
            comp = 3 if d >= 8 else 2 if d >= 0 else 1 if d >= -8 else 0
            ver = {"proctored at the testing centre": 3, "remotely proctored (webcam)": 2}.get(f["exam_mode"], 1)
        else:
            comp = {"current, mapped to this course": 3, "current, mapped to elective credit only": 2, "expired": 1,
                    "not on the crosswalk": 1}[f["cert_status"]]
            if not f["cert_verified"]:
                ver = 1 if f["cert_status"] != "not on the crosswalk" else 0
            else:
                ver = 3 if f["cert_status"].startswith("current") else 2
        return {"pla_award": award, "competency_demonstrated": comp, "evidence_verification": ver,
                "within_pla_cap": "yes" if within else "no", "experience_current": "yes" if current else "no"}

    def _rules(self, f):
        return [f"{f['policy']} §2: course credit = both assessors rate every outcome 'meets' (portfolio), a proctored exam "
                f"score >= {f['cut_score']}, or a current, issuer-verified certification mapped to the course. Half or more of "
                "the outcomes met -> elective credit. Assessors split on full credit, an unproctored attempt or an expired "
                "certification -> offer the proctored exam.",
                f"§3: portfolio practice must fall within the last {f['currency_years']} years and be verified by an employer "
                f"or licensing body. §5: prior-learning credit is capped at {f['cap']} credits per degree."]

    def _evidence(self, f):
        if f["method"] == "portfolio":
            return f"portfolio: assessor A {f['met_a']}/{f['n_outcomes']} outcomes 'meets', assessor B {f['met_b']}/{f['n_outcomes']}"
        if f["method"] == "exam":
            return f"challenge exam: {f['exam_score']}/100, {f['exam_mode']}"
        return (f"certification: {f['cert']} ({f['cert_status']}; "
                + ("verified with the issuing body)" if f["cert_verified"] else "copy of certificate only)"))

    def render(self, f, rng, surface):
        if surface == "json":
            return dumps(prune({
                "student": {"name": f["student"], "id": f["sid"], "pla_credits_already_awarded": f["already_awarded"]},
                "request": {"course": f"{f['course']} ({f['credits']} cr)", "basis": f["experience"],
                            "evidence": self._evidence(f), "last_relevant_practice_years_ago": f["years_since_practice"],
                            "verification": f["verification"] if f["method"] == "portfolio" else None},
                "policy": self._rules(f),
            }))
        if surface == "assessment_record":
            rows = [("Student", f"{f['student']} ({f['sid']})"), ("Course", f"{f['course']} ({f['credits']} cr)"),
                    ("Experience", f["experience"]), ("Evidence", self._evidence(f)),
                    ("Verification", f["verification"] if f["method"] == "portfolio" else None),
                    ("Last practice", f"{f['years_since_practice']} yrs ago (window {f['currency_years']} yrs)"),
                    ("PLA credits to date", f"{f['already_awarded']} (cap {f['cap']})")]
            return "PRIOR LEARNING ASSESSMENT RECORD\n" + kv_lines(rows, rng) + "\n\n" + self._rules(f)[0]
        para = [f"{f['student']} ({f['sid']}) is seeking credit for {f['course']} ({f['credits']} cr) on the strength of {f['experience']}.",
                f"Evidence: {self._evidence(f)}.",
                (f"Verification: {f['verification']}." if f["method"] == "portfolio" else ""),
                f"Most recent relevant practice: {f['years_since_practice']} years ago.",
                f"The student already holds {f['already_awarded']} prior-learning credits."]
        return (f"Coordinator note - {f['coordinator']} ({fmt_date(f['date'], rng)})\n\n" + noisy(join_sentences(para), rng, 0.1)
                + "\n\n" + bullets(self._rules(f), rng))

    def questions(self):
        return {
            "pla_award": choice("What should the prior-learning committee award?", {
                "award_course_credit": "Credit for the course itself.",
                "award_elective_credit": "Elective credit in the discipline, not the course.",
                "request_further_evidence": "Nothing yet; ask for verification or further evidence first.",
                "offer_challenge_exam": "Nothing yet; offer the challenge exam to settle it.",
                "deny_credit": "No credit.",
            }),
            "competency_demonstrated": score("How fully has the student demonstrated the course's competencies?", [
                "Barely: well short of the standard.",
                "Partly: some competencies shown, important ones missing.",
                "Mostly: close to the standard.",
                "Fully: every competency shown at the required level.",
            ]),
            "evidence_verification": score("How well verified is the evidence offered?", [
                "Unverified.",
                "Weakly verified.",
                "Reasonably verified.",
                "Fully verified by a proctored exam, a current credential or a signed employer letter with consistent assessments.",
            ]),
            "within_pla_cap": noul(
                "Awarding the credit requested would keep the student within the policy's cap on prior-learning credit.",
                yes="Credits already awarded plus this course's credits do not exceed the cap.",
                no="Together they would exceed the cap."),
            "experience_current": noul(
                "The practice offered as evidence falls within the policy's currency window."),
        }


WORKFLOW = Workflow(
    name="education_program_and_credit_review",
    title="Education program and credit review",
    description="Registrar, accreditor, system-office and committee decisions on transfer credit, accreditation standards, program viability, course substitutions and prior-learning credit, judged against written (fictional) academic policies.",
    domain="higher education administration",
    schemas=[TransferCreditEquivalency(), AccreditationStandardFinding(), AcademicProgramReview(),
             CourseSubstitutionPetition(), PriorLearningAssessment()],
    notes="Decisions rest on academic records and written rules only; students appear with a name, id and record and no "
          "protected attribute. No admissions of individuals, no financial aid, no academic misconduct (covered elsewhere).",
)
