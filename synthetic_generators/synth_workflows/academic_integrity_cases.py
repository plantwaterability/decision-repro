"""Academic integrity cases: suspected plagiarism, collusion, exam misconduct and outsourced work at fictional
universities, from the first similarity report to the sanction panel.

Five schemas: reviewing a text-matching report, a collusion allegation between students on the same task, an
invigilator's exam-hall report, an authorship concern where the evidence is circumstantial, and a panel setting
a sanction once a breach is found or admitted. Each samples a `regime` first ('clear' / 'borderline' /
'conflicting') and lets it steer the factors: similarity near the level a department usually acts on, overlap
that permitted collaboration might explain, indicators that point different ways, and facts no one has checked.

Institutions, courses, staff and students are fictional. Mitigating circumstances are limited to events such as
bereavement, housing loss or heavy paid work; no student's protected characteristics are stated or used.
"""
from __future__ import annotations

from datetime import timedelta

from .base import (
    FIRST,
    LAST,
    Schema,
    Workflow,
    chance,
    choice,
    dumps,
    fmt_date,
    fmt_ts,
    join_sentences,
    kv_lines,
    maybe,
    near,
    noisy,
    noul,
    org,
    pick,
    pick_w,
    prune,
    rand_date,
    rand_ts,
    regime,
    score,
    table,
)

SUBJECTS = [("HIST", "History"), ("COMP", "Computing"), ("BIOL", "Biology"), ("ECON", "Economics"),
            ("PSYC", "Psychology"), ("MECH", "Mechanical Engineering"), ("CHEM", "Chemistry"), ("LAWS", "Law"),
            ("BUSI", "Business"), ("STAT", "Statistics"), ("ENGL", "English"), ("GEOG", "Geography")]
LEVELS = {"first_year": "first-year undergraduate", "second_year": "second-year undergraduate",
          "final_year": "final-year undergraduate", "masters": "taught master's", "doctoral": "doctoral"}


def _names(rng, n):
    """n fictional people with distinct first AND last names."""
    return [f"{a} {b}" for a, b in zip(rng.sample(FIRST, n), rng.sample(LAST, n))]


def _course(rng, level=None, codes=None):
    """A course code; `codes` restricts the subject (a programming task sits in COMP, a lab report in a lab subject)."""
    pool = [x for x in SUBJECTS if codes is None or x[0] in codes] or SUBJECTS
    code, subj = pick(rng, pool)
    band = {"first_year": 1, "second_year": 2, "final_year": 3, "masters": 7, "doctoral": 8}.get(level, pick(rng, [1, 2, 3]))
    return f"{code} {band}{rng.randrange(0, 100):02d}", subj


def _sid(rng):
    return f"S{rng.randrange(1000000, 9999999)}"


def _level(rng, weights=(4, 3, 3, 2, 0)):
    return pick(rng, list(LEVELS), list(weights))


# =========================================================================== schema 1
class SimilarityReportReview(Schema):
    name = "similarity_report_review"
    description = "A text-matching report on a submitted assignment, awaiting a decision on whether it becomes an integrity case."
    surfaces = {"json": 0.5, "similarity_summary": 0.25, "marker_referral": 0.25}

    SOURCES = {"web_page": "web page", "journal_article": "published article", "student_paper_other_institution":
               "student paper (another institution)", "student_paper_same_course": "earlier submission on this course",
               "own_earlier_work": "the student's own earlier submission", "set_text_or_template": "assignment brief / template text",
               "reference_list": "reference-list entries"}
    SOURCES_A = {"web_page": "a web page", "journal_article": "a published article",
                 "student_paper_other_institution": "a student paper from another institution",
                 "student_paper_same_course": "an earlier submission on this course",
                 "own_earlier_work": "the student's own earlier submission", "set_text_or_template": "the assignment brief or template",
                 "reference_list": "reference-list entries"}
    ASSESS = {"essay": "Essay", "lab_report": "Lab report", "literature_review": "Literature review",
              "reflective_log": "Reflective log", "dissertation_chapter": "Dissertation chapter"}
    SWAPS = ["Stilted substitutions ('commence' for 'start', 'aforementioned populace' for 'these people') suggest words were swapped to beat the matcher.",
             "Some matched sentences have odd synonyms dropped in, as if reworded by a paraphrasing tool.",
             "Word order is kept but key terms are replaced with near-synonyms that make little sense in context."]

    def sample_factors(self, rng):
        reg = regime(rng)
        level = _level(rng)
        overall = near(rng, 24, 14, lo=4, hi=80, integer=True)
        biblio_excluded = chance(rng, 0.7)
        quoted_share = pick(rng, [0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
        patchwriting = chance(rng, 0.35)
        synonyms = chance(rng, 0.12)
        prior = pick(rng, [0, 0, 0, 1, 2])
        training = chance(rng, 0.7)
        main_src = pick_w(rng, {"web_page": 3, "journal_article": 3, "student_paper_other_institution": 1,
                                "student_paper_same_course": 1, "own_earlier_work": 1, "set_text_or_template": 1})
        main_share = pick(rng, [0.25, 0.4, 0.6, 0.8])
        if reg == "clear":
            if chance(rng, 0.5):
                overall, quoted_share, patchwriting, synonyms, prior = rng.randint(5, 12), 0.9, False, False, 0
                main_src, biblio_excluded = pick(rng, ["set_text_or_template", "journal_article"]), True
            else:
                overall, quoted_share, synonyms, prior = rng.randint(42, 70), 0.1, True, pick(rng, [0, 1])
                main_src, main_share, patchwriting = pick(rng, ["student_paper_same_course", "web_page"]), 0.8, True
        elif reg == "conflicting":
            twist = pick(rng, ["high_but_references", "low_but_student_paper", "self_plagiarism", "final_year_excuse"])
            if twist == "high_but_references":
                overall, biblio_excluded, main_src, quoted_share = rng.randint(35, 55), False, "set_text_or_template", 0.7
                synonyms = False
            elif twist == "low_but_student_paper":
                overall, main_src, main_share, quoted_share = rng.randint(9, 15), "student_paper_same_course", 0.8, 0.1
            elif twist == "self_plagiarism":
                main_src, main_share, overall = "own_earlier_work", 0.8, rng.randint(25, 45)
            else:
                level, patchwriting, quoted_share, prior = "final_year", True, 0.3, 0
        srcs = [(main_src, max(1, round(overall * main_share)))]
        rest = overall - srcs[0][1]
        others = [k for k in self.SOURCES if k != main_src and k != "reference_list"]
        while rest > 0 and len(srcs) < 5:
            v = min(rest, rng.randint(1, max(1, rest)))
            srcs.append((pick(rng, others), v))
            rest -= v
        if not biblio_excluded:
            srcs.append(("reference_list", rng.randint(3, 9)))
        merged = {}
        for k, v in srcs:
            merged[k] = merged.get(k, 0) + v
        srcs = sorted(merged.items(), key=lambda kv: -kv[1])
        student, marker, officer = _names(rng, 3)
        course, subj = _course(rng, level)
        kinds = [k for k in self.ASSESS if k != "dissertation_chapter" or level in ("final_year", "masters")]
        f = {"regime": reg, "level": level, "assessment": pick(rng, kinds),
             "weight_pct": pick(rng, [10, 20, 25, 40, 60]), "overall_similarity": sum(v for _, v in srcs),
             "sources": srcs, "bibliography_excluded": biblio_excluded, "quoted_share_of_matches": quoted_share,
             "patchwriting": patchwriting, "synonym_substitution": synonyms, "prior_cases": prior,
             "integrity_module_completed": training, "student": student, "student_id": _sid(rng), "marker": marker,
             "officer": officer, "course": course, "subject": subj, "university": org(rng, "school"),
             "submitted": rand_date(rng), "marker_view": pick(rng, ["lenient", "neutral", "suspicious"]),
             "swap_idx": rng.randrange(3)}
        # share of the matched text carrying a citation or quotation marks (quoted text counts; close paraphrase
        # with a citation at the end of the paragraph counts; synonym-swapped text is never cited)
        cited = quoted_share + (1 - quoted_share) * pick(rng, [0.0, 0.05, 0.15, 0.3, 0.5])
        if patchwriting:
            cited += pick(rng, [0.3, 0.45])
        if synonyms:
            cited = quoted_share
        f["cited_or_quoted_share"] = min(0.95, round(cited * 20) / 20)
        # what has been checked so far: the automated report only, a manual side-by-side check, drafts requested,
        # an interview with the student. A side-by-side check that found nothing beyond quotations, references or
        # set text ends the trail there.
        f["matches_explained"] = (f["cited_or_quoted_share"] >= 0.7 or srcs[0][0] in ("set_text_or_template", "reference_list")
                                  or f["overall_similarity"] <= 12)
        checks = pick_w(rng, {"none": 3, "side_by_side": 3, "drafts": 2, "meeting": 1})
        if f["matches_explained"] and checks in ("drafts", "meeting"):
            checks = "side_by_side"
        f["checks_done"] = checks
        f["drafts_result"] = pick(rng, ["notes_dont_cover", "no_drafts", "pasted_late"])
        f["interview_result"] = pick(rng, ["admitted", "misunderstood_citing", "could_not_explain"])
        f["uncited_passages"] = max(2, round(f["overall_similarity"] * (1 - f["cited_or_quoted_share"]) / 4))
        f["checked_on"] = f["submitted"] + timedelta(days=rng.randint(3, 12))
        return f

    DRAFTS_RESULT = {"notes_dont_cover": "the student sent an outline and some notes, none of which cover the matched passages",
                     "no_drafts": "the student says they wrote straight into the final file and has no drafts",
                     "pasted_late": "the version history shows the matched paragraphs pasted in on the last evening"}
    INTERVIEW_RESULT = {"admitted": "the student admitted copying the passages under deadline pressure",
                        "misunderstood_citing": "the student says they thought a citation at the end of a paragraph was enough",
                        "could_not_explain": "the student could not say where the matched passages came from"}

    def _checks(self, f):
        """The checks done so far, as short clauses (always at least the manual-comparison line)."""
        top = self.SOURCES_A[f["sources"][0][0]]
        if f["checks_done"] == "none":
            return ["Manual comparison: not yet done; only the automated report so far"]
        if f["matches_explained"]:
            out = ["Manual comparison done by the marker: the matched text is mostly quotations, reference entries "
                   "or the assignment brief"]
        else:
            out = [f"Manual comparison done by the marker: {f['uncited_passages']} passages match {top} with no "
                   "quotation marks or citation"]
        if f["checks_done"] in ("drafts", "meeting"):
            out.append("Drafts requested: " + self.DRAFTS_RESULT[f["drafts_result"]])
        if f["checks_done"] == "meeting":
            out.append("Student interviewed: " + self.INTERVIEW_RESULT[f["interview_result"]])
        return out

    def _marker_note(self, f, rng):
        s = []
        if f["patchwriting"]:
            s.append(pick(rng, ["Several paragraphs follow their sources almost line by line with a citation at the end.",
                                "Close paraphrase throughout section 2; citations are there but the wording is barely changed."]))
        if f["synonym_substitution"]:
            s.append(self.SWAPS[f["swap_idx"]])
        if f["quoted_share_of_matches"] >= 0.7:
            s.append("Most highlighted text is in quotation marks with references.")
        elif f["cited_or_quoted_share"] <= 0.35:
            s.append("Few of the highlighted passages are quoted or referenced.")
        elif f["quoted_share_of_matches"] <= 0.3 and f["cited_or_quoted_share"] >= 0.6:
            s.append("Most matched passages have a citation, though few are in quotation marks.")
        s.append({"lenient": pick(rng, ["I think this is poor referencing rather than cheating.", "Probably inexperience."]),
                  "neutral": "Flagging for a second opinion.",
                  "suspicious": pick(rng, ["This doesn't read like their earlier work.", "I'm fairly sure this is copied."])}[f["marker_view"]])
        return noisy(join_sentences(s), rng, 0.15)

    def render(self, f, rng, surface):
        src_rows = [{"source_type": self.SOURCES[k], "percent": v} for k, v in f["sources"]]
        if surface == "json":
            return dumps(prune({
                "submission": {"student_id": f["student_id"], "course": f["course"], "assessment": self.ASSESS[f["assessment"]],
                               "weight_percent": f["weight_pct"], "level": LEVELS[f["level"]], "submitted": f["submitted"].isoformat()},
                "report": {"overall_similarity_percent": f["overall_similarity"], "bibliography_excluded": f["bibliography_excluded"],
                           "sources": src_rows,
                           "share_of_matches_in_quotation_marks": f"{int(f['quoted_share_of_matches'] * 100)}%",
                           "share_of_matches_cited_or_quoted": f"{round(f['cited_or_quoted_share'] * 100)}%"},
                "marker": {"name": f["marker"], "note": self._marker_note(f, rng)},
                "checks_so_far": self._checks(f),
                "student_record": {"previous_integrity_cases": f["prior_cases"],
                                   "integrity_module_completed": f["integrity_module_completed"]},
            }))
        if surface == "similarity_summary":
            rows = [[str(i + 1), self.SOURCES[k], f"{v}%"] for i, (k, v) in enumerate(f["sources"])]
            return "\n".join([
                f"SIMILARITY REPORT -- {f['university']}", f"Submission: {f['student_id']} / {f['course']} / {self.ASSESS[f['assessment']]} "
                f"({f['weight_pct']}% of module)", f"Overall similarity: {f['overall_similarity']}%   "
                f"Bibliography excluded: {'yes' if f['bibliography_excluded'] else 'no'}   "
                f"Matches in quotation marks: {int(f['quoted_share_of_matches'] * 100)}%   "
                f"Matches cited or quoted: {round(f['cited_or_quoted_share'] * 100)}%", "",
                table(["#", "source", "match"], rows, rng), "",
                f"Marker ({f['marker']}): {self._marker_note(f, rng)}",
                "Checks so far (" + fmt_date(f["checked_on"], rng) + "):", "\n".join("  - " + c for c in self._checks(f)),
                f"Student record: {f['prior_cases']} previous case(s); integrity module "
                f"{'completed' if f['integrity_module_completed'] else 'not completed'}."])
        top = f["sources"][0]
        return "\n".join([
            f"Subject: Possible integrity issue -- {f['course']} {self.ASSESS[f['assessment']].lower()}",
            f"From: {f['marker']}", f"To: {f['officer']} (Academic Integrity Officer)", "",
            f"Hi {f['officer'].split()[0]},", "",
            noisy(join_sentences([
                f"The {self.ASSESS[f['assessment']].lower()} from {f['student_id']} ({LEVELS[f['level']]}, worth {f['weight_pct']}%) "
                f"came back at {f['overall_similarity']}% similarity"
                + ("" if f["bibliography_excluded"] else " with the bibliography left in") + ".",
                f"The biggest match is {top[1]}% from {self.SOURCES_A[top[0]]}.",
                f"About {round(f['cited_or_quoted_share'] * 100)}% of the matched text has a citation or quotation marks.",
                self._marker_note(f, rng),
                " ".join(c + "." for c in self._checks(f)),
                f"The student has {f['prior_cases']} previous case(s) on file." if f["prior_cases"] else "No previous cases on file."]), rng, 0.1),
            "", "Thanks,", f["marker"].split()[0]])

    def questions(self):
        return {
            "referral_decision": choice("How should the flagged submission be taken forward?", {
                "department_case": "Open a formal case to be handled by the department's academic integrity officer.",
                "no_case": "Treat the matches as properly attributed or trivial and mark the work normally.",
                "refer_to_panel": "Send it straight to the university panel because it is serious or a repeat.",
                "skills_support": "Mark it, with any referencing deduction the scheme allows, and refer the student for "
                                  "study-skills support rather than a case.",
            }),
            "copying_intent": score("How strongly does the evidence point to deliberate copying rather than poor referencing?", [
                "Not at all: the matches are quotations, references or set text.",
                "Weakly: patchy referencing typical of inexperience.",
                "Moderately: close paraphrase or uncited passages beyond careless slips.",
                "Strongly: disguised or wholesale copying, such as swapped synonyms or another student's text.",
            ]),
            "unoriginal_extent": score("How much of the assessed work is not the student's own?", [
                "Negligible: a sentence or two.",
                "Limited: a few passages in otherwise original work.",
                "Substantial: a significant section or line of argument.",
                "Most of it: the work largely reproduces other sources.",
            ]),
            "attribution_present": noul(
                "The matched material is attributed to its sources, even if imperfectly.",
                yes="Most matched passages carry a citation or quotation marks, even with formatting errors.",
                no="Much of the matched material appears with no citation or quotation marks at all."),
            "next_evidence": choice("What should be looked at before a decision is made?", {
                "drafts_and_notes": "The student's drafts, notes and version history.",
                "no_further_evidence": "Nothing more; the checks already done are enough to decide.",
                "side_by_side_check": "A manual side-by-side check of the flagged passages against the sources, "
                                      "excluding references and set text.",
                "student_meeting": "A meeting with the student about how the work was written.",
            }),
        }


# =========================================================================== schema 2
class CollusionAllegation(Schema):
    name = "collusion_allegation"
    description = "Submissions from two or more students that look alike, on a task with stated collaboration rules."
    surfaces = {"json": 0.5, "comparison_report": 0.25, "instructor_email": 0.25}

    TASKS = {"programming_assignment": "programming assignment", "lab_report": "lab report", "problem_set": "problem set",
             "project_writeup": "individual write-up of a group project"}
    TASK_SUBJECTS = {"programming_assignment": {"COMP", "STAT"}, "lab_report": {"BIOL", "CHEM", "MECH", "PSYC"},
                     "problem_set": {"ECON", "STAT", "MECH", "COMP", "CHEM"}, "project_writeup": None}
    TASKS_PL = {"programming_assignment": "programming assignments", "lab_report": "lab reports", "problem_set": "problem sets",
                "project_writeup": "individual project write-ups"}
    POLICY = {"individual_only": "Work must be entirely individual; do not discuss solutions.",
              "discuss_not_share": "You may discuss ideas but must {produce} yourself.",
              "pairs_allowed": "You may work in registered pairs and submit identical work as a pair.",
              "brief_unclear": "Collaboration is encouraged (brief does not say what may be shared)."}

    def _policy(self, f):
        produce = "write your own code" if f["task"] == "programming_assignment" else "write up your answers"
        return self.POLICY[f["policy"]].format(produce=produce)
    ACCOUNTS = {"both_deny": "Both say they worked alone and cannot explain the overlap.",
                "one_admits_sharing": "{a} admits sending their work 'to check the format'; {b} says they only looked at it.",
                "one_says_copied": "{a} says {b} must have taken their work from a shared lab machine; {b} denies it.",
                "studied_together": "Both say they worked through it together in the library, as they thought was allowed."}

    def sample_factors(self, rng):
        reg = regime(rng)
        task = pick(rng, list(self.TASKS))
        policy = pick_w(rng, {"individual_only": 2, "discuss_not_share": 4, "pairs_allowed": 1, "brief_unclear": 2})
        n = pick(rng, [2, 2, 2, 3])
        overlap = near(rng, 55, 25, lo=15, hi=98, integer=True)
        errors = pick(rng, [0, 0, 1, 2, 4])
        identical = chance(rng, 0.35)
        gap = pick(rng, [3, 20, 90, 600, 2900])
        account = pick_w(rng, {"both_deny": 3, "one_admits_sharing": 2, "one_says_copied": 2, "studied_together": 3})
        history = pick_w(rng, {"independent_histories": 2, "single_upload": 3, "not_available": 3})
        common = pick_w(rng, {"none": 3, "tutorial_solution": 2, "starter_code": 1})
        forum = chance(rng, 0.2)
        prior = pick(rng, [0, 0, 0, 1])
        if reg == "clear":
            if chance(rng, 0.5):
                overlap, errors, identical, account, history, policy = rng.randint(15, 30), 0, False, "studied_together", "independent_histories", pick(rng, ["discuss_not_share", "pairs_allowed"])
                common = pick(rng, ["tutorial_solution", "starter_code"])
            else:
                overlap, errors, identical, gap, policy, common = rng.randint(85, 98), 4, True, 3, "individual_only", "none"
                account = pick(rng, ["one_admits_sharing", "both_deny"])
        elif reg == "conflicting":
            twist = pick(rng, ["high_overlap_explained", "unclear_brief_identical", "one_victim"])
            if twist == "high_overlap_explained":
                overlap, common, errors, history = rng.randint(70, 85), pick(rng, ["tutorial_solution", "starter_code"]), 0, "independent_histories"
            elif twist == "unclear_brief_identical":
                policy, identical, errors, forum = "brief_unclear", True, pick(rng, [2, 4]), True
            else:
                account, overlap, history, errors = "one_says_copied", rng.randint(75, 95), "single_upload", pick(rng, [2, 4])
        if task != "programming_assignment" and common == "starter_code":
            common = "tutorial_solution"
        if policy == "pairs_allowed" and n > 2:
            policy = "discuss_not_share"
        names = _names(rng, n + 1)
        f = {"regime": reg, "task": task, "policy": policy, "n_students": n, "overlap_pct": overlap,
             "shared_unusual_errors": errors, "identical_comments_or_wording": identical, "submission_gap_min": gap,
             "account": account, "version_history": history, "common_source": common, "ta_forum_post": forum,
             "prior_case_one_student": prior, "students": names[:n], "instructor": names[n],
             "ids": [_sid(rng) for _ in range(n)], "course": _course(rng, None, self.TASK_SUBJECTS[task])[0],
             "university": org(rng, "school"),
             "weight_pct": pick(rng, [5, 10, 15, 20, 30]), "due": rand_ts(rng, hours=(12, 23))}
        # what the overlap consists of: a shared approach (what permitted discussion produces), some word-for-word
        # passages, or near-identical work line by line
        kind = pick_w(rng, {"approach_only": 2, "partly_verbatim": 2, "verbatim": 1})
        if overlap >= 85:
            kind = "verbatim"
        elif overlap <= 35:
            kind = "partly_verbatim" if identical else "approach_only"
        elif identical and kind == "approach_only":
            kind = "partly_verbatim"
        f["overlap_kind"] = kind
        return f

    def _kind(self, f):
        code = f["task"] == "programming_assignment"
        return {"approach_only": "same overall approach and structure, but each wrote their own " + ("code" if code else "wording"),
                "partly_verbatim": "several " + ("functions" if code else "paragraphs") + " are identical line for line; the rest differs",
                "verbatim": "near-identical line by line throughout, layout included"}[f["overlap_kind"]]

    def _common(self, f):
        return {"none": None, "tutorial_solution": "the week 6 tutorial worked solution covers much of the same ground",
                "starter_code": "about a third of the matching code is the provided starter code"}[f["common_source"]]

    def _history(self, f):
        noun = "commit history" if f["task"] == "programming_assignment" else "document version history"
        return {"independent_histories": f"{noun} shows each student building the work up separately over days",
                "single_upload": f"{noun} shows {f['students'][-1].split()[0]}'s work arriving in a single upload an hour before the deadline",
                "not_available": f"no {noun} available"}[f["version_history"]]

    def _gap(self, f):
        g = f["submission_gap_min"]
        return f"{g} minutes apart" if g < 120 else (f"{g // 60} hours apart" if g < 1440 else f"{g // 1440} days apart")

    def render(self, f, rng, surface):
        a, b = f["students"][0].split()[0], f["students"][-1].split()[0]
        account = self.ACCOUNTS[f["account"]].format(a=a, b=b)
        if surface == "json":
            return dumps(prune({
                "assessment": {"course": f["course"], "task": self.TASKS[f["task"]], "weight_percent": f["weight_pct"],
                               "collaboration_rule": self._policy(f),
                               "ta_forum_post": "A TA replied on the forum that students could 'compare approaches'." if f["ta_forum_post"] else None},
                "students": [{"name": s, "id": i} for s, i in zip(f["students"], f["ids"])],
                "comparison": {"overlap_percent": f["overlap_pct"], "nature_of_overlap": self._kind(f),
                               "shared_unusual_errors": f["shared_unusual_errors"],
                               "identical_comments_or_wording": f["identical_comments_or_wording"],
                               "submitted": self._gap(f), "common_material": self._common(f), "history": self._history(f)},
                "statements": {"summary": account},
                "records": {"previous_case_for_one_student": bool(f["prior_case_one_student"])},
            }))
        if surface == "comparison_report":
            lines = [f"PAIRWISE COMPARISON -- {f['course']} {self.TASKS[f['task']]} ({f['weight_pct']}%)",
                     "Students: " + ", ".join(f"{s} ({i})" for s, i in zip(f["students"], f["ids"])),
                     f"Rule in brief: \"{self._policy(f)}\"", "",
                     kv_lines([("Structural / textual overlap", f"{f['overlap_pct']}%"), ("Nature of overlap", self._kind(f)),
                               ("Shared unusual errors", f["shared_unusual_errors"]),
                               ("Identical comments or phrasing", "yes" if f["identical_comments_or_wording"] else "no"),
                               ("Submission times", self._gap(f)), ("Shared course material", self._common(f)),
                               ("History", self._history(f))], rng),
                     "", "Statements: " + account]
            if f["ta_forum_post"]:
                lines.append("Note: a TA told students on the forum they could 'compare approaches'.")
            if f["prior_case_one_student"]:
                lines.append(f"Note: {a} has one previous integrity case.")
            return "\n".join(lines)
        s = [f"I've got {f['n_students']} {self.TASKS_PL[f['task']]} in {f['course']} that look very alike ({f['overlap_pct']}% overlap).",
             f"On a close read: {self._kind(f)}.",
             (f"They share {f['shared_unusual_errors']} unusual error(s)" + (" and some identical comments." if f["identical_comments_or_wording"] else ".")
              if f["shared_unusual_errors"] else ("No shared mistakes, but some identical comments." if f["identical_comments_or_wording"] else "No shared mistakes to speak of.")),
             f"They were submitted {self._gap(f)}.",
             (f"To be fair, {self._common(f)}." if self._common(f) else ""),
             f"The {self._history(f)}.",
             f"The brief says: '{self._policy(f)}'",
             ("A TA did tell them on the forum they could compare approaches, which muddies it." if f["ta_forum_post"] else ""),
             f"When I asked: {account}"]
        return "\n".join([f"Subject: Suspected collusion -- {f['course']}", f"From: {f['instructor']}", "To: academic integrity office", "",
                          noisy(join_sentences(s), rng, 0.15), "", f["instructor"].split()[0]])

    def questions(self):
        return {
            "collusion_outcome": choice("How should the suspected collusion be handled?", {
                "formal_case_all": "Open formal cases against every student involved.",
                "formal_case_one": "Open a formal case against one student only, where the evidence shows the other did not knowingly share.",
                "informal_caution": "Give the students an informal caution about the collaboration rules, with no case.",
                "interview_first": "Interview the students separately before deciding whether there is a case.",
                "no_case": "Close it: permitted collaboration or shared course material explains the similarity.",
            }),
            "overlap_significance": score("How far does the overlap go beyond what shared material and permitted discussion would explain?", [
                "Not at all: shared material and allowed discussion account for it.",
                "A little: some unexplained overlap, mostly in routine parts.",
                "Considerably: substantial unexplained overlap, including in the substance.",
                "Entirely: the submissions are essentially one piece of work.",
            ]),
            "knowing_breach": score("How strong are the signs that the students knew they were breaking the collaboration rules?", [
                "None: the brief, or a member of staff, allowed working together in a way that could produce this overlap.",
                "Weak: sharing was not allowed, but beyond a similar approach nothing suggests they knew they were crossing the line.",
                "Moderate: sharing was not allowed and the work shares unusual errors or copied passages, but no one admits passing work on.",
                "Strong: sharing was not allowed and a student admits passing the work on, or one copy arrived in a single late upload.",
            ]),
            "permitted_collaboration_explains": noul(
                "The similarity could have arisen from collaboration the assignment allowed.",
                yes="The rules, or what staff told students, allowed the kind of working together that would produce this overlap.",
                no="The overlap goes beyond anything the rules or staff guidance allowed."),
            "same_treatment_for_all": noul(
                "All the students involved should be treated the same way.",
                yes="Nothing in the evidence separates the students' roles; each is implicated in the same way.",
                no="The evidence separates their roles: for example one built the work up over time while another's copy "
                   "arrived in a single late upload, or one admits passing it on."),
        }


# =========================================================================== schema 3
class ExamMisconductReport(Schema):
    name = "exam_misconduct_report"
    description = "An invigilator's report of a suspected breach of the rules during an in-person examination."
    surfaces = {"json": 0.5, "invigilator_report": 0.25, "incident_log": 0.25}

    INCIDENTS = {"phone_on_person": "mobile phone on the candidate's person during the exam",
                 "notes_in_calculator_cover": "handwritten notes inside the calculator cover",
                 "talking": "candidate whispering to the candidate at the next desk",
                 "smartwatch": "candidate wearing a smartwatch",
                 "unauthorised_formula_sheet": "a printed formula sheet of the candidate's own on the desk",
                 "looking_at_neighbour": "candidate repeatedly looking at the next candidate's script"}
    DEVICE = {"switched_off": "switched off, in a pocket", "on_silent": "switched on and silent, in a pocket",
              "screen_on_in_hand": "in the candidate's hand with the screen on"}
    WATCH = {"switched_off": "switched off, on the wrist", "on_silent": "switched on, on the wrist",
             "screen_on_in_hand": "screen lit; candidate seen looking at it"}

    def _device(self, f):
        d = f["device_state"]
        if d is None:
            return None
        return (self.WATCH if f["incident"] == "smartwatch" else self.DEVICE)[d]
    RESPONSE = {"handed_over": "handed it over at once", "said_forgot": "said they forgot it was there",
                "argued_then_complied": "argued, then complied", "denies": "denies doing anything wrong"}
    # the same four response types, for incidents where nothing is handed over
    RESPONSE_BEHAVIOUR = {"handed_over": "stopped at once when warned", "said_forgot": "said they were only asking to borrow a pen",
                          "argued_then_complied": "argued, then moved to another desk as asked",
                          "denies": "denies doing anything wrong"}
    QUANT = {"STAT", "ECON", "MECH", "CHEM", "BIOL", "COMP"}

    def _response(self, f):
        table_ = self.RESPONSE_BEHAVIOUR if f["incident"] in ("talking", "looking_at_neighbour") else self.RESPONSE
        return table_[f["student_response"]]

    def sample_factors(self, rng):
        reg = regime(rng)
        inc = pick(rng, list(self.INCIDENTS))
        device = pick(rng, list(self.DEVICE)) if inc in ("phone_on_person", "smartwatch") else None
        relevance = (pick_w(rng, {"relevant": 3, "unrelated": 2, "not_checked": 2})
                     if inc in ("notes_in_calculator_cover", "unauthorised_formula_sheet", "phone_on_person", "smartwatch") else None)
        response = pick_w(rng, {"handed_over": 4, "said_forgot": 3, "argued_then_complied": 2, "denies": 3})
        second = chance(rng, 0.5)
        announced = chance(rng, 0.85)
        exam_minutes = pick(rng, [90, 120, 180])
        minute = min(pick(rng, [4, 20, 45, 80, 110, 150]), exam_minutes - 10)
        neighbour = pick_w(rng, {"similar_errors": 2, "no_similarity": 2, "not_compared": 3}) if inc in ("talking", "looking_at_neighbour") else None
        prior = pick(rng, [0, 0, 0, 1])
        level = _level(rng)
        if reg == "clear":
            if chance(rng, 0.5):
                inc, device, relevance, response, second, announced, neighbour = "phone_on_person", "switched_off", "unrelated", "handed_over", True, False, None
            else:
                inc, relevance, response, second, prior = "notes_in_calculator_cover", "relevant", pick(rng, ["denies", "argued_then_complied"]), True, pick(rng, [0, 1])
                device, neighbour = None, None
        elif reg == "conflicting":
            twist = pick(rng, ["device_on_but_unrelated", "relevant_notes_no_witness", "looking_with_errors", "watch_denies"])
            if twist == "device_on_but_unrelated":
                inc, device, relevance, response = "phone_on_person", "screen_on_in_hand", "unrelated", "said_forgot"
                neighbour = None
            elif twist == "relevant_notes_no_witness":
                inc, relevance, second, response, device, neighbour = "unauthorised_formula_sheet", "relevant", False, "said_forgot", None, None
            elif twist == "looking_with_errors":
                inc, neighbour, second, device, relevance = "looking_at_neighbour", pick(rng, ["similar_errors", "not_compared"]), False, None, None
            else:
                inc, device, relevance, response, neighbour = "smartwatch", "on_silent", "not_checked", "denies", None
        student, inv1, inv2 = _names(rng, 3)
        quant = inc in ("notes_in_calculator_cover", "unauthorised_formula_sheet")
        course, subj = _course(rng, level, self.QUANT if quant else None)
        f = {"regime": reg, "incident": inc, "device_state": device, "content_relevance": relevance,
             "student_response": response, "second_invigilator_witnessed": second, "device_announcement_made": announced,
             "minutes_into_exam": minute, "neighbour_scripts": neighbour, "prior_cases": prior, "level": level,
             "exam_weight_pct": pick(rng, [30, 50, 60, 70, 100]), "exam_minutes": exam_minutes,
             "candidate": student, "candidate_no": f"C{rng.randrange(10000, 99999)}", "invigilator": inv1,
             "second_invigilator": inv2, "course": course, "venue": pick(rng, ["Sports Hall A", "Great Hall", "Room L2.14", "Exam Centre 3"]),
             "start": rand_ts(rng, hours=(9, 14)), "university": org(rng, "school"),
             "content_detail": pick(rng, ["two worked examples from the lecture notes", "formulas for the topics on the paper",
                                          "a shopping list and a bus timetable", "messages from the morning, nothing course-related"])}
        # the rule printed on this paper's cover sheet: whether what was seen breaks it depends on the variant
        if inc in ("phone_on_person", "smartwatch"):
            rule = pick_w(rng, {"devices_banned": 1, "devices_off_allowed": 2})
        elif inc == "unauthorised_formula_sheet":
            rule = pick_w(rng, {"supplied_sheet_only": 2, "own_sheet_allowed": 3})
        elif inc == "notes_in_calculator_cover":
            rule = "no_materials"
        else:
            rule = "no_communication"
        f["paper_rule"] = rule
        return f

    RULES = {"devices_banned": "No phones, smartwatches or other devices on the person; they stay in bags at the front of the hall.",
             "devices_off_allowed": "Phones and smartwatches must be switched off and put away; a switched-off device in a pocket is not a breach.",
             "supplied_sheet_only": "Only the formula sheet supplied with the paper may be used.",
             "own_sheet_allowed": "Candidates may bring one A4 sheet of their own formulas or notes.",
             "no_materials": "No notes or written material other than the question paper and the supplied sheets.",
             "no_communication": "Candidates must not communicate with other candidates or look at their work."}
    PROCEDURE = ("The department deals with a first breach the candidate accepts (a warning, or zero for the affected "
                 "questions or paper); repeat cases and cases the candidate disputes go to the university panel.")

    def _content(self, f):
        r = f["content_relevance"]
        if r is None:
            return None
        if r == "not_checked":
            return "contents not checked at the time"
        detail = f["content_detail"]
        if r == "relevant" and "shopping" in detail or r == "relevant" and "messages" in detail:
            detail = "formulas for the topics on the paper"
        if r == "unrelated" and ("formulas" in detail or "worked" in detail):
            detail = "a shopping list and a bus timetable" if f["incident"] != "phone_on_person" else "messages from the morning, nothing course-related"
        return ("relevant to the paper: " if r == "relevant" else "not related to the paper: ") + detail

    def _neighbour(self, f):
        return {None: None, "similar_errors": "the two scripts share the same unusual error in question 3",
                "no_similarity": "the two scripts show no unusual similarity", "not_compared": "scripts not yet compared"}[f["neighbour_scripts"]]

    def render(self, f, rng, surface):
        t = f["start"] + timedelta(minutes=f["minutes_into_exam"])
        if surface == "json":
            return dumps(prune({
                "exam": {"course": f["course"], "venue": f["venue"], "start": fmt_ts(f["start"], style="iso"),
                         "duration_minutes": f["exam_minutes"], "weight_percent": f["exam_weight_pct"],
                         "device_announcement_made": f["device_announcement_made"], "rule_on_cover_sheet": self.RULES[f["paper_rule"]]},
                "procedure": self.PROCEDURE,
                "candidate": {"number": f["candidate_no"], "level": LEVELS[f["level"]], "previous_integrity_cases": f["prior_cases"]},
                "incident": {"observed": self.INCIDENTS[f["incident"]], "minutes_into_exam": f["minutes_into_exam"],
                             "device_state": self._device(f),
                             "content": self._content(f), "neighbour_comparison": self._neighbour(f),
                             "candidate_response": self._response(f)},
                "witnesses": {"reporting_invigilator": f["invigilator"],
                              "second_invigilator": f["second_invigilator"] if f["second_invigilator_witnessed"] else None,
                              "corroborated": f["second_invigilator_witnessed"]},
            }))
        if surface == "invigilator_report":
            rows = [("Examination", f"{f['course']}, {f['venue']}"), ("Date", fmt_date(f["start"].date(), rng)),
                    ("Rule on cover sheet", self.RULES[f["paper_rule"]]),
                    ("Candidate no.", f["candidate_no"]), ("Time observed", f"{t.strftime('%H:%M')} ({f['minutes_into_exam']} min in)"),
                    ("Nature of incident", self.INCIDENTS[f["incident"]]),
                    ("Device state", self._device(f)),
                    ("Content", self._content(f)), ("Scripts compared", self._neighbour(f)),
                    ("Candidate's response", self._response(f)),
                    ("Witnessed by second invigilator", f["second_invigilator"] if f["second_invigilator_witnessed"] else "no"),
                    ("Device announcement read at start", "yes" if f["device_announcement_made"] else "not recorded"),
                    ("Paper weighting", f"{f['exam_weight_pct']}% of module")]
            return (f"EXAMINATION IRREGULARITY REPORT -- {f['university']}\n" + kv_lines(rows, rng)
                    + f"\n\nReporting invigilator: {f['invigilator']}\nProcedure: {self.PROCEDURE}")
        ev = [(f["start"], f"exam started, {f['exam_minutes']} min" + ("; device announcement read" if f["device_announcement_made"] else "")),
              (t, f"{f['invigilator'].split()[0]} observed {self.INCIDENTS[f['incident']]} (cand. {f['candidate_no']})")]
        if f["device_state"]:
            ev.append((t + timedelta(minutes=1), f"device {self._device(f)}"))
        if self._content(f):
            ev.append((t + timedelta(minutes=2), self._content(f)))
        ev.append((t + timedelta(minutes=2), f"candidate {self._response(f)}"))
        if f["second_invigilator_witnessed"]:
            ev.append((t + timedelta(minutes=3), f"{f['second_invigilator'].split()[0]} attended and confirmed"))
        if self._neighbour(f):
            ev.append((t + timedelta(minutes=f["exam_minutes"]), self._neighbour(f)))
        return (f"{f['venue']} incident log -- {f['course']} ({f['exam_weight_pct']}%)\n"
                + f"Cover-sheet rule: {self.RULES[f['paper_rule']]}\n"
                + "\n".join(f"{x.strftime('%H:%M')}  {noisy(e, rng, 0.05)}" for x, e in ev)
                + f"\nCandidate record: {LEVELS[f['level']]}, {f['prior_cases']} previous case(s).\n"
                + f"Procedure: {self.PROCEDURE}")

    def questions(self):
        return {
            "exam_case_outcome": choice("What should happen to the reported exam incident?", {
                "department_penalty": "Apply a penalty within the department's powers, such as a zero on the affected questions or paper.",
                "no_breach": "Record that no breach of the exam rules took place.",
                "refer_to_panel": "Refer it to the university panel because it may warrant more than the department can impose.",
                "warning_no_penalty": "Record a technical breach with a formal warning but no mark penalty.",
            }),
            "advantage_potential": score("How much could the candidate have gained from what was found or seen?", [
                "Nothing: the item or behaviour had no bearing on the paper.",
                "Little: a marginal help at most.",
                "Meaningful: it could have helped with part of the paper.",
                "Large: it could have supplied answers to a significant part of the paper.",
            ]),
            "deliberateness": score("How likely is it that the breach was deliberate?", [
                "Unlikely: it looks like an oversight.",
                "Possible: the account of an oversight is plausible but not certain.",
                "Likely: the circumstances suggest the item or behaviour was intended to help.",
                "Almost certain: preparation, concealment or use is evident.",
            ]),
            "breach_established": noul(
                "A breach of the examination rules is established on the invigilators' evidence.",
                yes="What was found or seen breaks a stated exam rule, and the record of it is reliable.",
                no="What was seen does not clearly break a rule, or the observation is uncorroborated and disputed."),
            "script_handling": choice("How should the candidate's script be handled while the case is decided?", {
                "mark_excluding_affected": "Mark it, leaving out the questions the material could have helped with.",
                "mark_normally": "Mark it in the normal way; the incident does not affect the marking.",
                "mark_withhold_result": "Mark it but withhold the result until the case is decided.",
                "void_attempt": "Void the attempt so the candidate must resit the paper.",
            }),
        }


# =========================================================================== schema 4
class AuthorshipConcern(Schema):
    name = "authorship_concern"
    description = "A marker suspects a submission was written by someone or something else; the evidence is circumstantial."
    surfaces = {"json": 0.45, "concern_form": 0.3, "email_chain": 0.25}

    INDICATORS = {"style_shift": "writing style and vocabulary far from the student's earlier work",
                  "metadata_other_author": "document properties list another person as author",
                  "grade_jump": "a mark far above the student's record",
                  "fabricated_references": "references that do not exist",
                  "detector_flag": "a generated-text detector flagged the work",
                  "listing_found": "a request matching the brief was found posted on an essay-bidding site",
                  "unfamiliar_terminology": "uses methods never taught on the course"}
    ASSESS = {"essay": "essay", "take_home_exam": "take-home exam", "coding_project": "coding project",
              "dissertation_chapter": "dissertation chapter"}
    VIVA = {"not_held": "no authorship interview held yet", "explained_well": "at interview the student explained the work fluently",
            "struggled": "at interview the student could not explain key sections or define terms used",
            "declined": "the student declined the authorship interview"}

    def sample_factors(self, rng):
        reg = regime(rng)
        level = _level(rng, (3, 3, 4, 3, 1))
        pool = list(self.INDICATORS)
        k = pick(rng, [1, 2, 2, 3])
        inds = rng.sample(pool, k)
        prior_avg = pick(rng, [52, 58, 61, 64, 70])
        this_mark = min(95, prior_avg + pick(rng, [2, 8, 15, 22, 30]))
        drafts = pick_w(rng, {"full_history": 2, "some_notes": 3, "none": 3})
        viva = pick_w(rng, {"not_held": 4, "explained_well": 2, "struggled": 2, "declined": 1})
        detector = maybe(rng, pick(rng, [18, 45, 71, 88, 97]), 0.3)
        fabricated = pick(rng, [0, 0, 1, 3, 6]) if "fabricated_references" in inds else 0
        if reg == "clear":
            if chance(rng, 0.5):
                inds, drafts, viva, this_mark, detector = ["detector_flag"], "full_history", "explained_well", prior_avg + 4, 62
            else:
                inds = ["listing_found", "metadata_other_author", "style_shift"]
                drafts, viva, this_mark = "none", pick(rng, ["struggled", "declined"]), min(95, prior_avg + 28)
        elif reg == "conflicting":
            twist = pick(rng, ["strong_signals_good_viva", "weak_signals_bad_viva", "detector_only_high", "refs_but_drafts"])
            if twist == "strong_signals_good_viva":
                inds, viva = ["metadata_other_author", "style_shift"], "explained_well"
            elif twist == "weak_signals_bad_viva":
                inds, viva, drafts = ["grade_jump"], "struggled", "some_notes"
            elif twist == "detector_only_high":
                inds, detector, drafts, viva = ["detector_flag"], pick(rng, [94, 98]), "none", "not_held"
            else:
                inds, drafts, viva = ["fabricated_references", "style_shift"], "full_history", "not_held"
                fabricated = pick(rng, [3, 6])
        if "fabricated_references" in inds and not fabricated:
            fabricated = pick(rng, [1, 3])
        if "detector_flag" in inds and detector is None:
            detector = pick(rng, [71, 88, 97])
        if "grade_jump" in inds:
            this_mark = max(this_mark, min(95, prior_avg + 20))
        student, marker, officer = _names(rng, 3)
        course, subj = _course(rng, level)
        kinds = [k for k in self.ASSESS if k != "dissertation_chapter" or level in ("final_year", "masters")]
        return {"regime": reg, "level": level, "assessment": pick(rng, kinds), "indicators": inds,
                "prior_average_mark": prior_avg, "this_mark": this_mark, "drafts": drafts, "viva": viva,
                "detector_score": detector, "fabricated_references": fabricated, "weight_pct": pick(rng, [20, 30, 40, 50, 100]),
                "prior_cases": pick(rng, [0, 0, 0, 1]),
                "days_to_exam_board": pick(rng, [2, 4, 6, 10, 18, 25, 40, 60]), "award_decision_this_board": level in ("final_year", "masters") and chance(rng, 0.6),
                "student": student, "student_id": _sid(rng), "marker": marker, "officer": officer, "course": course,
                "university": org(rng, "school"), "raised": rand_date(rng)}

    DRAFTS = {"full_history": "full version history showing the work built up over three weeks",
              "some_notes": "some handwritten notes and one early outline", "none": "no drafts or notes offered"}

    def _indicator_lines(self, f):
        out = []
        for k in f["indicators"]:
            t = self.INDICATORS[k]
            if k == "detector_flag":
                t += f" ({f['detector_score']}% likely generated; university guidance: detector output is not evidence on its own)"
            if k == "fabricated_references":
                t += f" ({f['fabricated_references']} of the reference list)"
            if k == "grade_jump":
                t += f" ({f['this_mark']}% against an average of {f['prior_average_mark']}%)"
            out.append(t)
        return out

    def render(self, f, rng, surface):
        inds = self._indicator_lines(f)
        board = (f"exam board in {f['days_to_exam_board']} days" +
                 ("; the student's final award is decided at that board" if f["award_decision_this_board"] else ""))
        if surface == "json":
            return dumps(prune({
                "submission": {"student_id": f["student_id"], "course": f["course"], "assessment": self.ASSESS[f["assessment"]],
                               "weight_percent": f["weight_pct"], "level": LEVELS[f["level"]], "provisional_mark": f["this_mark"]},
                "concern": {"raised_by": f["marker"], "raised": f["raised"].isoformat(), "indicators": inds},
                "evidence_so_far": {"student_average_before": f["prior_average_mark"], "process_evidence": self.DRAFTS[f["drafts"]],
                                    "authorship_interview": self.VIVA[f["viva"]]},
                "context": {"previous_integrity_cases": f["prior_cases"], "timetable": board},
            }))
        if surface == "concern_form":
            rows = [("Student", f"{f['student_id']} ({LEVELS[f['level']]})"), ("Course", f["course"]),
                    ("Assessment", f"{self.ASSESS[f['assessment']]}, {f['weight_pct']}% of module"),
                    ("Provisional mark", f"{f['this_mark']}% (previous average {f['prior_average_mark']}%)"),
                    ("Drafts / process evidence", self.DRAFTS[f["drafts"]]), ("Interview", self.VIVA[f["viva"]]),
                    ("Previous cases", f["prior_cases"]), ("Timetable", board)]
            return "\n".join([f"AUTHORSHIP CONCERN FORM -- {f['university']}", kv_lines(rows, rng), "", "Grounds for concern:",
                              "\n".join(f"  {i + 1}. {x}" for i, x in enumerate(inds)), "", f"Raised by {f['marker']}, {fmt_date(f['raised'], rng)}"])
        first = [f"I'm not comfortable with the {self.ASSESS[f['assessment']]} from {f['student_id']} in {f['course']}.",
                 "What worries me: " + "; ".join(inds) + ".",
                 pick(rng, ["I could be wrong.", "It just doesn't sound like them.", ""])]
        second = [f"Thanks. {self.DRAFTS[f['drafts']][0].upper() + self.DRAFTS[f['drafts']][1:]}; {self.VIVA[f['viva']]}.",
                  f"Timing: {board}.", (f"They have {f['prior_cases']} earlier case on file." if f["prior_cases"] else "")]
        return "\n".join([f"From: {f['marker']}", f"To: {f['officer']}", f"Subject: Authorship concern -- {f['course']}", "",
                          noisy(join_sentences(first), rng, 0.15), "", "----", f"From: {f['officer']}", f"To: {f['marker']}", "",
                          noisy(join_sentences(second), rng, 0.1)])

    def questions(self):
        return {
            "authorship_next_step": choice("What should the integrity officer do next with this concern?", {
                "authorship_interview": "Hold an authorship interview where the student explains and defends the work.",
                "no_further_action": "Take no further action; the concern is not strong enough to pursue.",
                "refer_to_panel": "Refer it to the panel as a formal allegation now.",
                "request_process_evidence": "Ask the student for drafts, notes, sources and version history first.",
            }),
            "authorship_doubt": score("How strong is the evidence that the submitted work is not the student's own?", [
                "Negligible: nothing beyond a hunch or a single unreliable signal.",
                "Slight: one or two circumstantial signs with innocent explanations.",
                "Moderate: several signs that point the same way but nothing conclusive.",
                "Strong: independent signs corroborate each other and the student's account is weak.",
                "Overwhelming: direct evidence such as a traced commission or another author's file.",
            ]),
            "resolution_pressure": score("How pressing is it to reach a decision on this concern?", [
                "Not pressing: the exam board that confirms the mark is more than four weeks away.",
                "Some urgency: the board is one to four weeks away, so it should be settled before marks are confirmed.",
                "Pressing: the board that confirms this mark meets within a week.",
                "Immediate: the board meets within a week and decides the student's final award on it.",
            ]),
            "decidable_on_record": noul(
                "The concern can fairly be decided now on the record as it stands, without asking anything more of the student.",
                yes="The student has already been interviewed or has declined an interview, or the only signs are a detector "
                    "score or a mark rise that a full version history answers, so it can be closed without involving them.",
                no="The student has not yet had the chance to answer, and the concern rests on more than a detector score or "
                   "a mark rise, or no full version history answers it."),
            "anomalies_point_elsewhere": noul(
                "The anomalies are better explained by someone or something else producing the work than by the student's own uneven effort."),
        }


# =========================================================================== schema 5
class SanctionPanel(Schema):
    name = "sanction_panel"
    description = "A university panel setting the outcome after an integrity breach has been found or admitted."
    surfaces = {"json": 0.5, "panel_briefing": 0.25, "case_table": 0.25}

    BREACH = {"plagiarism": "copied and uncited material", "collusion": "unauthorised collaboration on individual work",
              "exam_device": "unauthorised notes or device in an examination",
              "contract_cheating": "submitting work produced by a third party",
              "data_fabrication": "fabricated data in a lab or project report"}
    MITIGATION = {"none": None, "bereavement": "a close family bereavement in the weeks before the deadline",
                  "heavy_paid_work": "working 30+ hours a week in paid employment during term",
                  "housing_loss": "losing their accommodation mid-term"}
    PROGRAMMES = ["Nursing", "Teacher Education", "Social Work", "Pharmacy"]
    PROG_CODES = {"Nursing": "NURS", "Teacher Education": "EDUC", "Social Work": "SOCW", "Pharmacy": "PHAR"}

    def sample_factors(self, rng):
        reg = regime(rng)
        breach = pick(rng, list(self.BREACH))
        level = _level(rng, (3, 3, 3, 2, 1))
        extent = near(rng, 35, 30, lo=5, hi=100, integer=True)
        prior = pick(rng, [0, 0, 0, 1, 2])
        admission = pick_w(rng, {"early_admission": 3, "admitted_at_hearing": 2, "denied_throughout": 3})
        mit = pick_w(rng, {"none": 4, "bereavement": 2, "heavy_paid_work": 2, "housing_loss": 1})
        mit_documented = mit != "none" and chance(rng, 0.6)
        training = chance(rng, 0.7)
        professional = chance(rng, 0.25)
        disputed = admission == "denied_throughout" and chance(rng, 0.6)
        attending = pick_w(rng, {"will_attend": 3, "written_only": 2, "no_response": 1})
        if reg == "clear":
            if chance(rng, 0.5):
                breach, level, extent, prior, admission, mit, mit_documented = "plagiarism", "first_year", rng.randint(5, 15), 0, "early_admission", "none", False
                professional, disputed = False, False
            else:
                breach, extent, prior, admission = pick(rng, ["contract_cheating", "data_fabrication"]), rng.randint(80, 100), pick(rng, [1, 2]), "denied_throughout"
                level = pick(rng, ["final_year", "masters", "doctoral"])
        elif reg == "conflicting":
            twist = pick(rng, ["serious_but_mitigated", "minor_but_repeat", "professional_minor", "admitted_contract"])
            if twist == "serious_but_mitigated":
                breach, extent, mit, mit_documented, admission = "contract_cheating", rng.randint(60, 100), pick(rng, ["bereavement", "housing_loss"]), True, "early_admission"
            elif twist == "minor_but_repeat":
                breach, extent, prior = "plagiarism", rng.randint(8, 20), 2
            elif twist == "professional_minor":
                professional, breach, extent = True, pick(rng, ["collusion", "data_fabrication"]), rng.randint(10, 25)
            else:
                breach, admission, prior, level = "contract_cheating", "early_admission", 0, "first_year"
        if level == "doctoral" and breach == "exam_device":
            breach = "data_fabrication"
        student, chair, secretary = _names(rng, 3)
        prog = pick(rng, self.PROGRAMMES) if professional else None
        course, subj = _course(rng, level)
        if prog:
            band = {"first_year": 1, "second_year": 2, "final_year": 3, "masters": 7, "doctoral": 8}[level]
            course = f"{self.PROG_CODES[prog]} {band}{rng.randrange(0, 100):02d}"
        return {"regime": reg, "breach": breach, "level": level, "extent_pct": extent,
                "assessment_weight_pct": pick(rng, [15, 25, 40, 50, 100]), "prior_findings": prior, "admission": admission,
                "mitigation": mit, "mitigation_documented": mit_documented, "integrity_training_completed": training,
                "professional_programme": prog, "facts_disputed": disputed,
                "student_attendance": attending, "student": student, "student_id": _sid(rng), "chair": chair,
                "secretary": secretary, "course": course, "university": org(rng, "school"), "hearing": rand_date(rng),
                "doctoral_research_data": level == "doctoral" and breach == "data_fabrication"}

    ADMIT = {"early_admission": "admitted the breach when first asked", "admitted_at_hearing": "admitted it at the hearing",
             "denied_throughout": "has denied it throughout"}
    ATTEND = {"will_attend": "will attend in person", "written_only": "has sent a written statement and will not attend",
              "no_response": "has not responded to the invitation"}

    def render(self, f, rng, surface):
        mit = self.MITIGATION[f["mitigation"]]
        mit_txt = None if mit is None else mit + (" (documented)" if f["mitigation_documented"] else " (no evidence supplied)")
        if surface == "json":
            return dumps(prune({
                "case": {"student_id": f["student_id"], "level": LEVELS[f["level"]], "course": f["course"],
                         "programme_with_professional_registration": f["professional_programme"],
                         "hearing_date": f["hearing"].isoformat()},
                "breach": {"type": self.BREACH[f["breach"]], "share_of_assessment_affected_percent": f["extent_pct"],
                           "assessment_weight_percent": f["assessment_weight_pct"], "student_position": self.ADMIT[f["admission"]],
                           "facts_disputed": f["facts_disputed"]},
                "history": {"previous_findings": f["prior_findings"], "integrity_training_completed": f["integrity_training_completed"]},
                "mitigation": {"circumstances": mit_txt},
                "attendance": self.ATTEND[f["student_attendance"]],
            }))
        if surface == "panel_briefing":
            s = [f"The panel is asked to determine the outcome for {f['student_id']}, a {LEVELS[f['level']]} student on {f['course']}"
                 + (f" (a {f['professional_programme']} programme with professional registration)" if f["professional_programme"] else "") + ".",
                 f"The breach concerns {self.BREACH[f['breach']]}, affecting about {f['extent_pct']}% of an assessment worth "
                 f"{f['assessment_weight_pct']}% of the module.",
                 f"The student {self.ADMIT[f['admission']]}" + ("; the facts remain in dispute." if f["facts_disputed"] else "."),
                 (f"There {'is' if f['prior_findings'] == 1 else 'are'} {f['prior_findings']} previous finding(s) on record."
                  if f["prior_findings"] else "There are no previous findings."),
                 ("The integrity training module was completed." if f["integrity_training_completed"] else "The integrity training module was not completed."),
                 (f"Mitigation offered: {mit_txt}." if mit_txt else "No mitigation offered."),
                 f"The student {self.ATTEND[f['student_attendance']]}."]
            return f"PANEL BRIEFING -- {f['university']}\nSecretary: {f['secretary']}   Chair: {f['chair']}\n\n" + noisy(join_sentences(s), rng, 0.05)
        rows = [["Breach", self.BREACH[f["breach"]]], ["Extent", f"~{f['extent_pct']}% of a {f['assessment_weight_pct']}% assessment"],
                ["Level", LEVELS[f["level"]]], ["Professional programme", f["professional_programme"] or "no"],
                ["Position", self.ADMIT[f["admission"]]], ["Facts disputed", "yes" if f["facts_disputed"] else "no"],
                ["Previous findings", f["prior_findings"]], ["Training completed", "yes" if f["integrity_training_completed"] else "no"],
                ["Mitigation", mit_txt or "none"], ["Attendance", self.ATTEND[f["student_attendance"]]]]
        return f"Case {f['student_id']} -- {f['course']} -- hearing {fmt_date(f['hearing'], rng)}\n\n" + table(["item", "detail"], rows, rng)

    def questions(self):
        return {
            "sanction_band": score("What level of outcome does the breach warrant?", [
                "A formal reprimand; the mark stands.",
                "Resubmission with the mark capped at a bare pass.",
                "Zero for the assessment, with no resubmission.",
                "Failure of the whole module.",
                "Suspension or expulsion from the programme.",
            ]),
            "culpability": score("How culpable is the student, taking intent, experience and candour into account?", [
                "Low: an inexperienced student, a naive lapse, frank about it.",
                "Moderate: should have known better, but no planning or concealment.",
                "High: a knowing breach by someone who knew the rules.",
                "Very high: planned deception, concealment or a repeat after warnings.",
            ]),
            "further_referral": choice("What referral, if any, should accompany the outcome?", {
                "fitness_to_practise": "Refer to the fitness-to-practise process, because the programme leads to professional registration.",
                "none_needed": "No referral beyond the outcome itself.",
                "research_integrity": "Refer to the research integrity office, because research data or outputs are involved.",
                "study_skills": "Refer the student to study-skills support on academic writing and referencing.",
            }),
            "panel_format": choice("How should the panel proceed with the case?", {
                "adjourn_for_information": "Adjourn until missing information, such as evidence for the mitigation, is supplied.",
                "decide_on_papers": "Decide on the written papers without a hearing.",
                "hearing_with_student": "Hold a hearing at which the student can speak.",
                "hearing_with_witnesses": "Hold a full hearing with the student and the staff who raised the case.",
            }),
            "mitigation_warrants_reduction": noul(
                "The student's circumstances justify an outcome below what the breach would otherwise warrant.",
                yes="Documented circumstances plausibly contributed to the breach and are serious enough to temper the outcome.",
                no="There is no mitigation, it is unevidenced, or it does not bear on why the breach happened."),
        }


WORKFLOW = Workflow(
    name="academic_integrity_cases",
    title="Academic integrity cases",
    description="How suspected plagiarism, collusion, exam misconduct and outsourced work should be handled at a "
                "university, from first report to sanction.",
    domain="higher education / academic integrity",
    schemas=[SimilarityReportReview(), CollusionAllegation(), ExamMisconductReport(), AuthorshipConcern(), SanctionPanel()],
    notes="Student academic conduct at fictional institutions. No student support threads, no security alerts, no "
          "AI-agent run review (a generated-text detector appears only as one weak indicator about a student's "
          "submission).",
)
