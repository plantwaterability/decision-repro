"""Manuscript triage: an academic journal's editorial decisions on a submitted paper, from the desk screen to an
integrity query, a revision round, a transfer offer after rejection, and an appeal.

Every journal, portfolio, manuscript, screening tool and person is fictional. Journal policies quoted in the
states (word limits, registration and ethics-statement rules, the one-appeal rule) are invented for this
generator.

Each schema draws a `regime` first. Borderline cases sit near a line (a manuscript 7% over the word limit, a
similarity score dominated by the methods section, a trial registered a few weeks after first enrolment);
conflicting cases pair a strong signal with a contrary one: an exciting result on a thin design, a combative
response letter that nevertheless fixes the main objection, a factual reviewer error on a peripheral point.
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

# Topics: (title, ethics kind "human" / "children" / "animal" / None, unit, design, finding, overclaim,
#          short running title used after the desk screen)
JOURNALS = {
    "brackwater": dict(
        name="Brackwater Ecology", scope="field and modelling studies of coastal, estuarine and wetland ecosystems",
        limit=7000, prefix="BWE",
        core=[("Seagrass recovery after dredging in a temperate estuary", None, "survey sites", "repeated field surveys",
               "how seagrass cover changed after dredging", "a dredging ban would restore seagrass meadows within two years", "Seagrass recovery after dredging"),
              ("Salt-marsh bird nesting success under rising spring tides", "animal", "nests", "repeated field surveys",
               "how tide height relates to nest losses", "tidal flooding is now the main threat to marsh birds everywhere", "Marsh nesting and spring tides")],
        adjacent=[("Microplastic loads in a freshwater lake food web", None, "water and fish samples", "a sampling survey",
                   "microplastic concentrations at each level of the food web", "microplastics are driving the lake's fish decline", "Microplastics in a lake food web"),
                  ("Laboratory growth of oyster larvae under warming", "animal", "rearing tanks", "a laboratory experiment",
                   "larval growth at three temperatures", "warming will collapse wild oyster populations this decade", "Oyster larvae under warming")],
        outside=[("Consumer attitudes to sustainably labelled seafood", "human", "survey respondents", "an online survey",
                  "stated willingness to pay for labelled seafood", "labelling schemes will end overfishing", "Attitudes to labelled seafood"),
                 ("A deep-learning model for masking clouds in satellite images", None, "satellite scenes",
                  "a benchmark comparison", "masking accuracy against existing tools",
                  "the model makes manual cloud screening obsolete", "Deep-learning cloud masking")],
        objections=["dredged and undredged sites differ in depth and sediment, so the comparison is confounded",
                    "surveys in different years were done in different seasons"],
        sisters={"broad": "Open Earth & Life Reports", "specialty": "Wetland Restoration Notes",
                 "short": "Brackwater Ecology Letters"}),
    "rural_care": dict(
        name="Rural Care Delivery Quarterly", scope="organisation, workforce and outcomes of health services in rural "
                                                  "and remote areas", limit=5000, prefix="RCDQ",
        core=[("Telehealth follow-up and readmissions in rural critical-access hospitals", "human", "patients",
               "a retrospective cohort", "readmission rates with and without telehealth follow-up",
               "telehealth follow-up prevents most readmissions and should be mandated", "Telehealth and rural readmissions"),
              ("Nurse practitioner staffing and wait times in remote clinics", "human", "clinics", "a cross-sectional survey",
               "the association between staffing and appointment waits", "adding one nurse practitioner halves wait times", "NP staffing and clinic waits")],
        adjacent=[("Emergency department crowding in a mid-sized city hospital", "human", "emergency visits",
                   "a retrospective audit", "crowding patterns by hour and day", "the new triage rule eliminated crowding", "Crowding in a city ED"),
                  ("Medical students' interest in rural careers", "human", "students", "a questionnaire study",
                   "stated interest in rural practice", "a rural placement guarantees rural recruitment", "Students and rural careers")],
        outside=[("A new immunoassay for serum ferritin", "human", "blood samples", "a method-comparison study",
                  "agreement with the reference assay", "the assay should replace current laboratory methods", "A ferritin immunoassay"),
                 ("Car-park pricing and staff retention in a metropolitan hospital", "human", "staff", "a staff survey",
                  "views on parking charges", "free parking would solve the hospital's retention problem", "Parking charges and retention")],
        objections=["there is no adjustment for case mix between hospitals",
                    "the outcome misses readmissions to other hospitals"],
        sisters={"broad": "Open Health Research Reports", "specialty": "Community Health Workforce",
                 "short": "Rural Care Brief Reports"}),
    "lattice": dict(
        name="Lattice: Computational Materials", scope="computational and data-driven studies of materials properties",
        limit=8000, prefix="LCM",
        core=[("Machine-learned potentials for lithium diffusion in garnet electrolytes", None, "training structures",
               "a simulation study", "diffusion barriers compared with first-principles values",
               "the potential makes first-principles calculations unnecessary for all garnets", "ML potentials for garnets"),
              ("High-throughput screening of halide perovskite stability", None, "candidate compositions",
               "a high-throughput screen", "predicted decomposition energies", "the top candidates will be stable in real devices", "Perovskite stability screen")],
        adjacent=[("Synthesis of a doped zinc oxide film, with supporting simulations", None, "films",
                   "a synthesis and characterisation study", "measured conductivity against simulated values",
                   "the film is ready for commercial displays", "Doped zinc oxide films"),
                  ("Open-source software for converting crystal-structure files", None, "test files", "a benchmark comparison",
                   "conversion accuracy across formats", "the tool should become the community standard", "Crystal-file conversion tool")],
        outside=[("Engineering students' attitudes to simulation software", "human", "students", "a classroom survey",
                  "attitudes before and after a course", "simulation courses transform engineering education", "Students and simulation software"),
                 ("Scheduling corrosion inspections for road bridges", None, "bridges", "an optimisation study",
                  "inspection schedules under budget limits", "the schedule would prevent all corrosion failures", "Bridge inspection scheduling")],
        objections=["the model is fitted and tested on the same structures",
                    "the simulation parameters are not justified against experiments"],
        sisters={"broad": "Open Physical Sciences Reports", "specialty": "Materials Data & Software",
                 "short": "Lattice Letters"}),
    "labour_policy": dict(
        name="Labour Policy Review Quarterly", scope="empirical evaluation of labour-market policies and institutions",
        limit=9000, prefix="LPRQ",
        core=[("Minimum-wage increases and hours worked in retail", None, "stores", "a difference-in-differences design",
               "changes in hours after the increase", "minimum-wage increases never reduce employment", "Minimum wage and retail hours"),
              ("A wage-subsidy program and youth employment: evidence from a phased rollout", "human", "participants",
               "a phased-rollout trial", "employment 12 months after enrolment",
               "the subsidy should be expanded nationally at once", "Wage subsidy and youth jobs")],
        adjacent=[("A theoretical model of union bargaining over pensions", None, "scenarios", "a calibrated model",
                   "how bargaining shifts pension outcomes", "unions always secure better pensions", "Union bargaining over pensions"),
                  ("Household saving after a pension reform", None, "households", "a household panel",
                   "saving rates before and after the reform", "the reform caused the fall in saving", "Saving after pension reform")],
        outside=[("Consumer demand for electric bicycles", None, "survey respondents", "a choice experiment",
                  "price sensitivity of demand", "subsidies would double e-bike sales", "Demand for e-bikes"),
                 ("A textile workers' guild, 1850-1890: an archival history", None, "archival records", "an archival study",
                  "the guild's membership and disputes", "the guild explains modern labour law", "A textile guild, 1850-1890")],
        objections=["treated and comparison groups were on different trends before the policy",
                    "the standard errors ignore clustering by region"],
        sisters={"broad": "Open Social Science Reports", "specialty": "Journal of Workplace Programs",
                 "short": "Labour Policy Letters"}),
    "early_literacy": dict(
        name="Early Literacy Inquiry", scope="research on how children aged 3 to 9 learn to read and write",
        limit=8000, prefix="ELI",
        core=[("Phonics instruction intensity and reading gains in first grade", "children", "children",
               "a cluster-randomised trial", "reading gains under two instruction schedules",
               "daily phonics guarantees every child reads by age seven", "Phonics intensity and reading"),
              ("Shared book reading at home and vocabulary at age four", "children", "families", "a longitudinal cohort",
               "the association between reading at home and vocabulary", "reading at home causes large vocabulary gains", "Home reading and vocabulary")],
        adjacent=[("Reading-comprehension strategies for adolescents", "children", "students", "a classroom trial",
                   "comprehension scores after strategy teaching", "strategy teaching closes the attainment gap", "Comprehension strategies"),
                  ("Teacher workload in primary schools", "human", "teachers", "a staff survey",
                   "hours worked and sources of workload", "cutting marking would end teacher attrition", "Primary teacher workload")],
        outside=[("Second-language learning in adult migrants", "human", "learners", "a cohort study",
                  "language gains over a course", "the course method works for every adult learner", "Adult second-language learning"),
                 ("Screen time and sleep in toddlers", "children", "families", "a parent questionnaire",
                  "the association between screen time and sleep", "screens are the main cause of poor toddler sleep", "Toddler screen time and sleep")],
        objections=["the reading test has not been validated for this age group",
                    "there is no control for baseline differences between classes"],
        sisters={"broad": "Open Social Science Reports", "specialty": "Classroom Practice Research",
                 "short": "Early Literacy Brief Reports"}),
}
ETHICS_RULE = ("Research with human participants or animals must state the approving ethics committee and reference "
               "number, and how consent was obtained; trials must be registered before the first participant enrols.")


def _topics(j):
    return j["core"] + j["adjacent"] + j["outside"]


def _journal(rng):
    k = pick(rng, list(JOURNALS))
    return k, JOURNALS[k]


def _ms_id(rng, j, d):
    return f"{j['prefix']}-{d.year}-{rng.randint(100, 2999):04d}"


def _author(rng):
    return person(rng)


# =========================================================================== schema 1: desk screen
class DeskScreen(Schema):
    name = "desk_screen"
    description = "A new submission at the editor's desk: send it for review, turn it away, return it or offer a transfer."
    surfaces = {"json": 0.5, "submission_record": 0.25, "editor_notes": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        jk, j = _journal(rng)
        fit = pick_w(rng, {"core": 4, "adjacent": 3, "outside": 2})
        words_ratio = round(between(rng, 0.6, 1.25), 2)
        n = pick(rng, [8, 14, 24, 60, 180, 900, 4200])
        claim = pick_w(rng, {"matched": 4, "overreach": 2, "modest": 2})
        sim_total = rng.randint(4, 38)
        sim_max_source = rng.randint(1, max(2, sim_total // 2))
        sim_where = pick_w(rng, {"methods": 3, "introduction": 2, "results_discussion": 1})
        sim_own_preprint = chance(rng, 0.25)
        ethics_state = pick_w(rng, {"full": 4, "partial": 2, "absent": 1, "exemption_claimed": 1})
        data_stmt = pick_w(rng, {"repository": 3, "on_request": 3, None: 1})
        variant = None
        if reg == "clear":
            if chance(rng, 0.5):
                fit, words_ratio, claim, sim_total, ethics_state = "core", round(between(rng, 0.7, 0.95), 2), "matched", rng.randint(3, 9), "full"
                sim_max_source = rng.randint(1, 3)
                n = pick(rng, [180, 900, 4200])
            else:
                fit, claim = "outside", pick(rng, ["overreach", "matched"])
        elif reg == "borderline":
            b = pick(rng, ["length", "adjacent_scope", "similarity_methods", "small_n"])
            if b == "length":
                words_ratio = near(rng, 1.0, 0.1, lo=0.8)
            elif b == "adjacent_scope":
                fit = "adjacent"
            elif b == "similarity_methods":
                sim_total, sim_where = near(rng, 22, 6, lo=8, integer=True), "methods"
                sim_max_source = near(rng, 9, 4, lo=2, integer=True)
            else:
                n, claim = pick(rng, [8, 14, 24]), "matched"
        else:
            ethics_topics = [t for t in j["core"] if t[1]]
            variant = pick(rng, ["exciting_thin_design", "clean_but_recycled_text"]
                           + (["perfect_fit_missing_ethics"] * 2 if ethics_topics else []))
            if variant == "exciting_thin_design":
                fit, claim, n = "core", "overreach", pick(rng, [8, 14, 24])
            elif variant == "perfect_fit_missing_ethics":
                fit, claim, ethics_state = "core", "matched", pick(rng, ["absent", "exemption_claimed", "partial"])
            else:
                fit, sim_total, sim_where, sim_own_preprint = "core", rng.randint(28, 45), "results_discussion", False
                sim_max_source = rng.randint(14, 25)
        topic = pick(rng, [t for t in j["core"] if t[1]]) if variant == "perfect_fit_missing_ethics" else pick(rng, j[fit])
        title, ethics_kind, unit, design = topic[:4]
        if ethics_kind is None:
            ethics_state = "not_applicable" if chance(rng, 0.8) else "absent"
        if sim_max_source > sim_total:
            sim_max_source = sim_total
        received = rand_date(rng)
        return {
            "regime": reg, "variant": variant, "journal_key": jk, "journal": j["name"], "scope_fit": fit,
            "title": title, "ethics_kind": ethics_kind, "ethics_statement": ethics_state, "unit": unit,
            "finding": topic[4], "overclaim": topic[5],
            "word_count": int(j["limit"] * words_ratio), "word_limit": j["limit"], "design": design, "sample_n": n,
            "claim_strength": claim, "similarity_total_pct": sim_total, "similarity_top_source_pct": sim_max_source,
            "similarity_location": sim_where, "similarity_top_source_is_own_preprint": sim_own_preprint,
            "data_statement": data_stmt, "corresponding_author": _author(rng), "received": received,
            "manuscript_id": _ms_id(rng, j, received), "assistant": person(rng),
            "cover_letter_tone": pick_w(rng, {"plain": 3, "hyped": 2, "pleading": 1}),
            "article_type": pick_w(rng, {"research article": 5, "short communication": 1, "methods paper": 1}),
        }

    def _claim_text(self, f):
        d, n, u = f["design"], f"{f['sample_n']:,}", f["unit"]
        return {"matched": f"Using {d} of {n} {u}, we examine {f['finding']} and discuss the limits of the design.",
                "overreach": f"Using {d} of {n} {u}, we show that {f['overclaim']}.",
                "modest": f"This descriptive report uses {d} of {n} {u} to document {f['finding']}."}[f["claim_strength"]]

    def _ethics_text(self, f):
        ref = f"{f['sample_n'] % 97 + 100}/{f['received'].year % 100}"
        if f["ethics_kind"] == "animal":
            return {"full": f"approved by the institutional animal ethics committee (ref. AEC-{ref}); field permits held",
                    "partial": "animal ethics approval was obtained (committee and reference not given)",
                    "absent": None, "exemption_claimed": "no animal ethics approval was needed because the study was observational",
                    "not_applicable": "not applicable (no human participants or animals)"}[f["ethics_statement"]]
        consent = ("parental consent and child assent obtained" if f["ethics_kind"] == "children"
                   else "written consent obtained")
        return {"full": f"approved by the university research ethics committee (ref. REC-{ref}); {consent}",
                "partial": "ethics approval was obtained (committee and reference not given)",
                "absent": None, "exemption_claimed": "formal ethics review was not required because data were anonymised",
                "not_applicable": "not applicable (no human participants or animals)"}[f["ethics_statement"]]

    def _sim_text(self, f):
        where = {"methods": "mostly in the methods section", "introduction": "mostly in the introduction",
                 "results_discussion": "including results and discussion paragraphs"}[f["similarity_location"]]
        own = " (the top source is the authors' own preprint)" if f["similarity_top_source_is_own_preprint"] else ""
        return f"{f['similarity_total_pct']}% overall, largest single source {f['similarity_top_source_pct']}%, {where}{own}"

    @staticmethod
    def _transfer(j):
        return (f"work in a neighbouring field may be offered to {j['sisters']['specialty']} or "
                f"{j['sisters']['broad']}")

    def _cover(self, f):
        return {"plain": "We believe the work fits the journal's readership.",
                "hyped": "This is the first study to settle this long-standing question and will be widely cited.",
                "pleading": "We would be grateful for a quick decision, as the first author's contract ends soon."}[f["cover_letter_tone"]]

    def render(self, f, rng, surface):
        j = JOURNALS[f["journal_key"]]
        data = {"repository": "data and code deposited in a public repository", "on_request": "data available on request",
                None: None}[f["data_statement"]]
        if surface == "json":
            return dumps(prune({
                "submission": {"id": f["manuscript_id"], "journal": f["journal"], "received": f["received"].isoformat(),
                               "title": f["title"], "type": f["article_type"], "words": f["word_count"],
                               "corresponding_author": f["corresponding_author"], "cover_letter_excerpt": self._cover(f)},
                "abstract_excerpt": self._claim_text(f),
                "checks": {"similarity_report": self._sim_text(f), "ethics_statement": self._ethics_text(f) or "missing",
                           "data_availability": data or "missing"},
                "journal_policy": {"scope": j["scope"], "word_limit": j["limit"], "ethics": ETHICS_RULE,
                                   "transfers": self._transfer(j)},
            }))
        if surface == "submission_record":
            rows = [("Manuscript", f["manuscript_id"]), ("Journal", f["journal"]), ("Received", fmt_date(f["received"], rng)),
                    ("Type", f["article_type"]), ("Title", f["title"]),
                    ("Word count", f"{f['word_count']:,} (limit {f['word_limit']:,})"),
                    ("Similarity check", self._sim_text(f)), ("Ethics statement", show(self._ethics_text(f), rng)),
                    ("Data availability", show(data, rng))]
            return (f"EDITORIAL SYSTEM -- initial QC\n{kv_lines(rows, rng, skip_none=False)}\n\nAbstract (excerpt): "
                    f"{self._claim_text(f)}\nCover letter: \"{self._cover(f)}\"\n\nAims and scope: {j['scope']}. {ETHICS_RULE} "
                    f"Transfers: {self._transfer(j)}.")
        ed = person(rng)
        notes = [f"Read the abstract of {f['manuscript_id']}, \"{f['title']}\".", self._claim_text(f),
                 f"EA check ({f['assistant']}): {f['word_count']:,} words against our {f['word_limit']:,};"
                 f" similarity {self._sim_text(f)}.",
                 f"Ethics statement: {self._ethics_text(f) or 'none found'}.",
                 f"Data: {data or 'no statement'}.",
                 pick(rng, ["Not sure this is really us.", "Topic is squarely ours.", "Could go either way on fit.", ""])
                 if f["scope_fit"] != "core" else pick(rng, ["Topic is squarely ours.", ""])]
        return (f"HANDLING EDITOR NOTES -- {ed}, {f['journal']}\n\n" + noisy(join_sentences(notes), rng, 0.12)
                + f"\n\nScope: {j['scope']}.\nPolicy: {ETHICS_RULE}\nTransfers: {self._transfer(j)}.")

    def questions(self):
        return {
            "desk_decision": choice("What should the editor do with this new submission?", {
                "desk_reject_contribution": "Decline without review: the topic is in scope, but the abstract claims a "
                                            "sweeping result from 24 or fewer units.",
                "desk_reject_scope": "Decline without review: the topic is outside the journal's scope and not in a "
                                     "neighbouring field either.",
                "offer_transfer": "Decline here and offer a sister journal: the topic is in a neighbouring field, "
                                  "related to but not within the journal's scope.",
                "return_for_fixes": "Return to the authors: in scope, but over the word limit, without a data "
                                    "statement, missing required ethics details, or with text overlap of 25%+ overall "
                                    "or 10%+ from one source other than the authors' own preprint.",
                "send_for_review": "Invite external reviewers: in scope, claims proportionate to the design, and no "
                                   "technical problem.",
            }),
            "scope_fit": score("How well does the manuscript fit the journal's stated aims and scope?", [
                "Outside: a reader of this journal would not expect to see it.",
                "Marginal: related field, but not the journal's subject.",
                "Reasonable: within scope, if not central.",
                "Central: exactly the kind of work the journal exists to publish.",
            ]),
            "review_prospects": score("How likely is the manuscript to come through peer review at this journal?", [
                "Very unlikely: a sweeping claim from 24 or fewer units, or 25%+ text overlap that reaches the "
                "results and discussion.",
                "Doubtful: 24 or fewer units with proportionate claims, or a sweeping claim from 60 to 180 units.",
                "Uncertain: a descriptive report on 60 or more units, or a sweeping claim from 900 or more.",
                "Fair: proportionate claims, with limits discussed, from 60 to 180 units.",
                "Good: proportionate claims, with limits discussed, from 900 or more units.",
            ]),
            "main_desk_concern": choice("What is the most serious problem an editor would raise at this stage?", {
                "claims_outrun_design": "The conclusions go beyond what the study design and sample can support.",
                "ethics_or_integrity": "Missing ethics documentation or reused text that must be cleared up first.",
                "fit_with_journal": "The topic does not belong in this journal.",
                "no_serious_problem": "Nothing serious; any issues can be handled in review.",
                "presentation_and_format": "Length, missing statements or formatting that break the submission rules.",
            }),
            "ethics_documentation_adequate": noul(
                "The manuscript documents the ethics approval and consent its research requires.",
                yes="Either no approval is needed, or the committee, reference and consent process are stated.",
                no="Approval is needed but missing, incomplete, or replaced by an unsupported exemption claim."),
        }


# =========================================================================== schema 2: research-ethics / integrity query
ISSUES = {
    "late_registration": dict(what="the trial was registered after enrolment began",
                              detail="registry record dated {n} weeks after the first participant enrolled"),
    "image_duplication": dict(what="an image-screening tool flagged possible duplicated panels",
                              detail="two panels in Figure {n} match after rotation, reported as different conditions"),
    "overlapping_paper": dict(what="the results overlap an earlier paper by the same authors",
                              detail="about {n}% of the tables repeat a 2023 article that is not cited"),
    "patient_photo": dict(what="a figure shows an identifiable patient",
                          detail="the clinic photograph in Figure {n} shows a patient's face; no consent form on file"),
    "authorship_dispute": dict(what="a former team member says they were left off the author list",
                               detail="they claim to have collected {n}% of the data"),
    "ethics_approval_doubt": dict(what="the stated ethics approval may not cover this study",
                                  detail="the cited approval number belongs to a different project from {n} years earlier"),
}


class ResearchEthicsQuery(Schema):
    name = "research_ethics_query"
    description = "A publication-ethics or research-integrity concern raised about a manuscript the journal is handling."
    surfaces = {"json": 0.5, "concern_email": 0.25, "case_log": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        jk, j = _journal(rng)
        topic = pick(rng, j["core"] + j["adjacent"])
        issue = pick(rng, self._issues_for(jk, topic))
        n = {"late_registration": rng.randint(1, 40), "image_duplication": rng.randint(2, 6),
             "overlapping_paper": rng.randint(15, 80), "patient_photo": rng.randint(1, 4),
             "authorship_dispute": rng.randint(10, 60), "ethics_approval_doubt": rng.randint(2, 7)}[issue]
        evidence = pick_w(rng, {"strong": 2, "moderate": 3, "weak": 2})
        explanation = pick_w(rng, {None: 3, "plausible": 2, "partial": 2, "evasive": 1})
        stage = pick_w(rng, {"under_review": 4, "revision": 2, "accepted": 2, "in_production": 1})
        source = pick_w(rng, {"screening_tool": 2, "reviewer": 3, "anonymous_reader": 2, "named_colleague": 1})
        prior = pick_w(rng, {0: 6, 1: 2, 2: 1})
        variant = None
        if reg == "clear":
            if chance(rng, 0.5):
                evidence, explanation, prior = "weak", "plausible", 0
                if issue == "late_registration":
                    n = rng.randint(1, 3)
            else:
                evidence, explanation, prior = "strong", pick(rng, ["evasive", None]), pick(rng, [1, 2])
        elif reg == "borderline":
            evidence = "moderate"
            explanation = pick(rng, ["partial", None])
            if issue == "late_registration":
                n = near(rng, 6, 4, lo=1, integer=True)
            if issue == "overlapping_paper":
                n = near(rng, 30, 10, lo=10, integer=True)
        else:
            variant = pick(rng, ["strong_evidence_good_explanation", "weak_evidence_bad_history", "late_stage_moderate"])
            if variant == "strong_evidence_good_explanation":
                evidence, explanation = "strong", "plausible"
            elif variant == "weak_evidence_bad_history":
                evidence, prior, source = "weak", 2, "anonymous_reader"
            else:
                evidence, stage = "moderate", pick(rng, ["accepted", "in_production"])
        received = rand_date(rng)
        return {
            "regime": reg, "variant": variant, "journal_key": jk, "journal": j["name"], "issue": issue, "issue_n": n,
            "evidence_strength": evidence, "author_explanation": explanation, "stage": stage, "raised_by": source,
            "prior_concerns_same_group": prior, "institution_known": chance(rng, 0.85),
            "corresponding_author": _author(rng), "editor": person(rng), "ethics_officer": person(rng),
            "raised_on": received, "manuscript_id": _ms_id(rng, j, received),
            "title": topic[6], "design": topic[3],
        }

    @staticmethod
    def _issues_for(jk, topic):
        """Only the concerns that could arise for this kind of study."""
        out = ["overlapping_paper", "authorship_dispute"]
        design = topic[3]
        if topic[1]:
            out.append("ethics_approval_doubt")
        if "trial" in design:
            out.append("late_registration")
        if jk == "rural_care":
            out.append("patient_photo")
        if any(w in design for w in ("laboratory", "synthesis", "method-comparison", "experiment")):
            out.append("image_duplication")
        return out

    def _evidence_text(self, f):
        return {"strong": ISSUES[f["issue"]]["detail"].format(n=f["issue_n"]) + "; checked independently by the ethics team",
                "moderate": ISSUES[f["issue"]]["detail"].format(n=f["issue_n"]) + "; not yet verified",
                "weak": "an allegation only; no supporting material was provided"}[f["evidence_strength"]]

    def _explanation_text(self, f):
        return {None: None,
                "plausible": {"late_registration": "the registry submission was delayed by a portal outage; the protocol was approved beforehand",
                              "image_duplication": "the authors supply original uncropped images showing the panels are different blots",
                              "overlapping_paper": "the earlier paper used a pilot subset; the authors offer to cite it and explain the overlap",
                              "patient_photo": "a signed consent form covering publication is attached",
                              "authorship_dispute": "the authors provide lab records showing the person's role was technical and acknowledged",
                              "ethics_approval_doubt": "the committee confirms an extension covers the new study"}[f["issue"]],
                "partial": "the authors acknowledge the problem and say documents will follow",
                "evasive": "the authors reply that the concern is malicious and decline to provide records"}[f["author_explanation"]]

    def render(self, f, rng, surface):
        who = {"screening_tool": "the publisher's automated screening", "reviewer": "a peer reviewer",
               "anonymous_reader": "an anonymous email", "named_colleague": "a named researcher at another university"}[f["raised_by"]]
        stage = f["stage"].replace("_", " ")
        expl = self._explanation_text(f)
        if surface == "json":
            return dumps(prune({
                "manuscript": {"id": f["manuscript_id"], "journal": f["journal"], "title": f["title"], "stage": stage,
                               "corresponding_author": f["corresponding_author"]},
                "concern": {"summary": ISSUES[f["issue"]]["what"], "raised_by": who, "raised_on": f["raised_on"].isoformat(),
                            "evidence": self._evidence_text(f)},
                "response": {"authors": expl or "no reply yet"},
                "history": {"previous_concerns_about_this_group": f["prior_concerns_same_group"],
                            "institution_contact_on_file": f["institution_known"]},
            }))
        if surface == "concern_email":
            body = [f"Dear {f['editor'].split()[0]},", "",
                    noisy(join_sentences([
                        f"I am writing about {f['manuscript_id']}, currently {stage} at {f['journal']}.",
                        f"The concern is that {ISSUES[f['issue']]['what']}.",
                        f"Details: {self._evidence_text(f)}.",
                        (f"When asked, {expl}." if expl else "The authors have not been contacted yet."),
                        (f"Our records show {f['prior_concerns_same_group']} earlier concern(s) about this group."
                         if f["prior_concerns_same_group"] else "")]), rng, 0.1), "",
                    f"Raised via {who}.", "", "Best wishes,", f"{f['ethics_officer']}, Publishing Ethics"]
            return f"Subject: Integrity concern -- {f['manuscript_id']}\n\n" + "\n".join(body)
        rows = [["Manuscript", f["manuscript_id"]], ["Stage", stage], ["Concern", ISSUES[f["issue"]]["what"]],
                ["Source", who], ["Evidence", self._evidence_text(f)], ["Author response", expl or "none yet"],
                ["Prior concerns (group)", f["prior_concerns_same_group"]],
                ["Institution contact", "on file" if f["institution_known"] else "not identified"]]
        return (f"INTEGRITY CASE LOG -- {f['journal']}\nOpened {fmt_date(f['raised_on'], rng)} by {f['ethics_officer']}\n\n"
                + table(["field", "entry"], rows, rng))

    def questions(self):
        return {
            "integrity_next_step": choice("What should the journal do next about this concern?", {
                "continue_with_note": "Carry on handling the manuscript and note the concern on file; it has been answered.",
                "pause_and_contact_institution": "Stop handling the manuscript and ask the authors' institution to look into it.",
                "reject_on_integrity_grounds": "Reject the manuscript because the problem cannot be put right.",
                "request_documents": "Ask the authors for specific records or an explanation before going further.",
            }),
            "misconduct_indication": score("How strongly does the evidence point to deliberate misconduct rather than honest error?", [
                "Not at all: an innocent explanation is documented.",
                "Weakly: an honest mistake is the likelier reading.",
                "Moderately: the pattern is hard to explain innocently.",
                "Strongly: deliberate manipulation or concealment is the likely reading.",
            ]),
            "harm_if_published": score("If the concern proved well founded, how damaging would publishing the paper be?", [
                "Negligible: a housekeeping correction at most.",
                "Limited: a correction would be needed after publication.",
                "Serious: an expression of concern or retraction would follow.",
                "Severe: readers or participants would be harmed or misled on a central finding.",
            ]),
            "handling_can_continue": noul(
                "Editorial handling can continue while the concern is being resolved.",
                yes="The issue can be fixed in revision or is minor enough not to taint the review.",
                no="The concern goes to whether the work can be trusted at all, so handling should stop."),
            "institution_notification": noul("The authors' institution should be informed at this stage."),
        }


# =========================================================================== schema 3: revision round
class RevisionRoundDecision(Schema):
    name = "revision_round_decision"
    description = "A revised manuscript returned with a response to reviewers, awaiting the editor's next decision."
    surfaces = {"json": 0.5, "response_table": 0.25, "editor_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        jk, j = _journal(rng)
        points = rng.randint(6, 22)
        full = rng.randint(0, points)
        rebutted = rng.randint(0, points - full)
        ignored = rng.randint(0, max(0, (points - full - rebutted) // 2))
        partial = points - full - rebutted - ignored
        key = pick_w(rng, {"addressed": 3, "partially": 3, "rebutted_with_argument": 2, "ignored": 1})
        round_no = pick_w(rng, {2: 5, 3: 2, 4: 1})
        tone = pick_w(rng, {"courteous": 5, "defensive": 2, "combative": 1})
        new_analyses = chance(rng, 0.5)
        over_limit = chance(rng, 0.15)
        reviewer_available = pick_w(rng, {"both": 4, "one": 3, "none": 1})
        special_issue_days = pick(rng, [None, None, None, 5, 14, 40])
        variant = None
        if reg == "clear":
            if chance(rng, 0.5):
                full, rebutted, ignored = points - 1, 1, 0
                partial, key, tone, over_limit = 0, "addressed", "courteous", False
            else:
                full, ignored = rng.randint(0, 2), rng.randint(3, 6)
                rebutted, key, tone = rng.randint(2, 4), "ignored", pick(rng, ["defensive", "combative"])
                partial = max(0, points - full - rebutted - ignored)
                points = full + rebutted + ignored + partial
        elif reg == "borderline":
            key = pick(rng, ["partially", "rebutted_with_argument"])
            full = int(points * rng.uniform(0.5, 0.75))
            rebutted = rng.randint(1, 3)
            ignored = pick(rng, [0, 0, 1])
            partial = max(0, points - full - rebutted - ignored)
            points = full + rebutted + ignored + partial
        else:
            variant = pick(rng, ["combative_but_fixed", "polite_but_dodged", "late_round_new_problem"])
            if variant == "combative_but_fixed":
                tone, key, new_analyses = "combative", "addressed", True
            elif variant == "polite_but_dodged":
                tone, key = "courteous", pick(rng, ["rebutted_with_argument", "ignored"])
                full = int(points * 0.8)
                rebutted, ignored = 1, 0
                partial = points - full - rebutted
            else:
                round_no, key, new_analyses = pick(rng, [3, 4]), "addressed", True
        months = rng.randint(2 + 3 * (round_no - 1), 6 + 5 * (round_no - 1))
        received = rand_date(rng)
        return {
            "regime": reg, "variant": variant, "journal_key": jk, "journal": j["name"], "round": round_no,
            "points_total": points, "points_addressed": full, "points_partial": partial, "points_rebutted": rebutted,
            "points_unanswered": ignored, "key_objection_status": key, "response_tone": tone,
            "new_analyses_added": new_analyses, "new_analysis_raises_issue": variant == "late_round_new_problem" or chance(rng, 0.1),
            "now_over_word_limit": over_limit, "reviewers_available": reviewer_available,
            "months_in_review": months, "special_issue_closes_in_days": special_issue_days,
            "key_objection": pick(rng, j["objections"]),
            "manuscript_id": _ms_id(rng, j, received), "received": received, "editor": person(rng),
            "title": pick(rng, j["core"] + j["adjacent"])[6], "author": _author(rng),
        }

    def _key_text(self, f):
        return {"addressed": "a new analysis addresses it directly and the conclusions were adjusted",
                "partially": "the authors added a sensitivity check but kept the original claims",
                "rebutted_with_argument": "the authors argue at length that the concern does not apply, without new analysis",
                "ignored": "the response letter does not mention it"}[f["key_objection_status"]]

    def render(self, f, rng, surface):
        rev = {"both": "both original reviewers agreed to look again", "one": "one original reviewer declined to re-review",
               "none": "neither original reviewer is available"}[f["reviewers_available"]]
        si = (f"special issue closes in {f['special_issue_closes_in_days']} days" if f["special_issue_closes_in_days"] else None)
        tone = {"courteous": "courteous", "defensive": "defensive in places",
                "combative": "combative; accuses reviewer 2 of bias"}[f["response_tone"]]
        if surface == "json":
            return dumps(prune({
                "manuscript": {"id": f["manuscript_id"], "journal": f["journal"], "title": f["title"],
                               "revision_round": f["round"], "months_in_review": f["months_in_review"],
                               "special_issue": si},
                "response_to_reviewers": {"points_raised": f["points_total"], "fully_addressed": f["points_addressed"],
                                          "partly_addressed": f["points_partial"], "rebutted": f["points_rebutted"],
                                          "not_answered": f["points_unanswered"], "tone": tone,
                                          "main_objection": f["key_objection"], "main_objection_response": self._key_text(f)},
                "changes": {"new_analyses": f["new_analyses_added"],
                            "new_analysis_note": ("the new analysis uses a different outcome definition from the original"
                                                  if f["new_analyses_added"] and f["new_analysis_raises_issue"] else None),
                            "over_word_limit": f["now_over_word_limit"]},
                "reviewers": rev,
            }))
        if surface == "response_table":
            rows = [["Fully addressed", f["points_addressed"]], ["Partly addressed", f["points_partial"]],
                    ["Rebutted", f["points_rebutted"]], ["Not answered", f["points_unanswered"]]]
            extra = [f"Main objection (reviewer 2): {f['key_objection']} -- {self._key_text(f)}.",
                     f"Tone of response letter: {tone}.",
                     ("New analyses added; note they use a different outcome definition." if f["new_analyses_added"] and
                      f["new_analysis_raises_issue"] else "New analyses added." if f["new_analyses_added"] else "No new analyses."),
                     ("Revised text now exceeds the word limit." if f["now_over_word_limit"] else ""),
                     f"Reviewers: {rev}.", (si.capitalize() + "." if si else "")]
            return (f"RESPONSE-TO-REVIEWERS TALLY -- {f['manuscript_id']} (round {f['round']}, {f['months_in_review']} "
                    f"months in review)\n\n" + table(["reviewer points", "count"], rows, rng) + "\n\n" + bullets(extra, rng))
        memo = [f"{f['manuscript_id']} is back for round {f['round']} after {f['months_in_review']} months in the system.",
                f"Of {f['points_total']} reviewer points, {f['points_addressed']} are fully dealt with, {f['points_partial']} "
                f"partly, {f['points_rebutted']} rebutted and {f['points_unanswered']} left unanswered.",
                f"On the main objection ({f['key_objection']}), {self._key_text(f)}.",
                f"The letter is {tone}.",
                ("They added new analyses, but these switch to a different outcome definition." if f["new_analyses_added"]
                 and f["new_analysis_raises_issue"] else ""),
                ("It is now over the word limit." if f["now_over_word_limit"] else ""),
                rev[0].upper() + rev[1:] + ".", (si.capitalize() + "." if si else "")]
        return f"EDITOR'S MEMO -- {f['editor']}, {f['journal']}\n\n" + noisy(join_sentences(memo), rng, 0.12)

    def questions(self):
        return {
            "revision_decision": choice("What decision should the editor make on this revision?", {
                "accept": "Accept the manuscript for publication.",
                "accept_after_minor_edits": "Accept once small editorial or presentation changes are made, without further review.",
                "further_major_revision": "Ask for another substantial revision.",
                "reject": "Reject the manuscript; the revision has not answered the fundamental problems.",
                "return_to_reviewers": "Send the revision back to reviewers before deciding.",
            }),
            "critique_resolution": score("How fully have the authors dealt with the reviewers' substantive points?", [
                "Barely: most substantive points are dodged or dismissed.",
                "Partly: the easy points are fixed, the hard ones are not.",
                "Mostly: remaining disagreements are reasoned and minor.",
                "Fully: every substantive point is answered with changes or convincing argument.",
            ]),
            "key_objection_resolved": noul(
                "The main methodological objection from the last round has been resolved.",
                yes="New evidence or analysis removes the objection, or the claims were narrowed to what the data support.",
                no="The objection stands: it was argued away without evidence, patched partially, or ignored."),
            "rereview_required": noul("The revision needs to go back to at least one reviewer before a decision."),
            "timeline_pressure": score("How much pressure is there to reach a decision on this round quickly?", [
                "None beyond the journal's normal handling times.",
                "Some: the manuscript has been in the system for many months.",
                "Considerable: a special issue closes soon, or this is a late round.",
                "Acute: the issue closes within days, or the manuscript has been in review well over a year.",
            ]),
        }


# =========================================================================== schema 4: transfer after rejection
REJECTION_REASONS = {
    "scope": ["both reviewers note the work is sound but addresses a question outside this journal's readership",
              "the reviewers liked the data but saw it as belonging to a neighbouring field"],
    "advance": ["the reviewers find the methods careful but the findings largely confirm earlier work",
                "reports describe a competent study with a modest, incremental advance"],
    "methods": ["reviewer 1 identifies a confound that the design cannot rule out",
                "both reports question whether the sample supports the main conclusion"],
    "reporting": ["the reports say key methods details and data are missing, making the work impossible to assess",
                  "reviewers could not evaluate the analysis because the code and protocol were not provided"],
}


class TransferAfterRejection(Schema):
    name = "transfer_after_rejection"
    description = "A manuscript declined after review, considered for a transfer offer within the publisher's portfolio."
    surfaces = {"json": 0.5, "decision_draft": 0.25, "portfolio_form": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        jk, j = _journal(rng)
        reason = pick(rng, list(REJECTION_REASONS))
        secondary = pick(rng, [None, None] + [r for r in REJECTION_REASONS if r != reason])
        soundness = pick_w(rng, {"sound": 3, "minor_issues": 3, "major_issues": 2, "unsound": 1})
        length = pick_w(rng, {"short": 2, "standard": 5, "long": 2})
        reports = pick_w(rng, {2: 5, 3: 2, 1: 1})
        report_quality = pick_w(rng, {"detailed": 4, "brief": 3, "one_unusable": 1 if reports > 1 else 0})
        author_open = pick_w(rng, {True: 3, False: 1, None: 3})
        apc_concern = chance(rng, 0.25)
        variant = None
        if reg == "clear":
            if chance(rng, 0.5):
                reason, soundness, report_quality, secondary = pick(rng, ["scope", "advance"]), "sound", "detailed", None
            else:
                reason, soundness, secondary = "methods", "unsound", "reporting"
        elif reg == "borderline":
            soundness = pick(rng, ["minor_issues", "major_issues"])
            secondary = pick(rng, [r for r in REJECTION_REASONS if r != reason])
        else:
            variant = pick(rng, ["sound_but_author_reluctant", "scope_but_flawed", "short_data_note"])
            if variant == "sound_but_author_reluctant":
                reason, soundness, author_open, apc_concern = "advance", "sound", False, True
            elif variant == "scope_but_flawed":
                reason, soundness, secondary = "scope", "major_issues", "methods"
            else:
                reason, soundness, length = "advance", "minor_issues", "short"
        received = rand_date(rng)
        return {
            "regime": reg, "variant": variant, "journal_key": jk, "journal": j["name"], "main_reason": reason,
            "secondary_reason": secondary, "soundness": soundness, "length": length, "n_reports": reports,
            "report_quality": report_quality, "author_open_to_transfer": author_open, "fee_concern": apc_concern,
            "manuscript_id": _ms_id(rng, j, received), "decided_on": received, "editor": person(rng),
            "title": pick(rng, j["core"] + j["adjacent"])[6], "reason_text": pick(rng, REJECTION_REASONS[reason]),
            "secondary_text": pick(rng, REJECTION_REASONS[secondary]) if secondary else None,
        }

    def _sound_text(self, f):
        return {"sound": "the reviewers found no errors in the analysis",
                "minor_issues": "reviewers list a few fixable analysis issues",
                "major_issues": "reviewers list analysis problems that would need new work",
                "unsound": "reviewers consider the central analysis invalid"}[f["soundness"]]

    def render(self, f, rng, surface):
        j = JOURNALS[f["journal_key"]]
        port = {"broad": f"{j['sisters']['broad']} (broad scope; judges soundness, not significance)",
                "specialty": f"{j['sisters']['specialty']} (narrower specialty title)",
                "short": f"{j['sisters']['short']} (short communications, up to 3,000 words)"}
        length = {"short": "about 3,000 words", "standard": "about 6,500 words", "long": "about 11,000 words"}[f["length"]]
        quality = {"detailed": "detailed and constructive", "brief": "brief (under a page each)",
                   "one_unusable": "detailed apart from one two-line dismissal"}[f["report_quality"]]
        author = {True: "authors said at submission they would consider a transfer",
                  False: "authors opted out of transfers at submission", None: None}[f["author_open_to_transfer"]]
        fee = "authors' funder does not cover open-access fees" if f["fee_concern"] else None
        if surface == "json":
            return dumps(prune({
                "manuscript": {"id": f["manuscript_id"], "journal": f["journal"], "title": f["title"], "length": length,
                               "decision": "reject after review", "decided_on": f["decided_on"].isoformat()},
                "reviews": {"number": f["n_reports"], "quality": quality, "summary": [x for x in (f["reason_text"],
                                                                                              f["secondary_text"]) if x],
                            "soundness": self._sound_text(f)},
                "authors": {"transfer_preference": author, "fees": fee},
                "portfolio_options": list(port.values()),
            }))
        if surface == "decision_draft":
            body = [f"Dear {person(rng).split()[0]},", "",
                    noisy(join_sentences([
                        f"Thank you for submitting {f['manuscript_id']}, \"{f['title']}\", to {f['journal']}.",
                        f"We received {f['n_reports']} report(s), which are {quality}.",
                        f"In short, {f['reason_text']}.", (f"In addition, {f['secondary_text']}." if f["secondary_text"] else ""),
                        f"On the analysis itself, {self._sound_text(f)}.",
                        "I am therefore unable to accept the manuscript."]), rng, 0.08), "",
                    "[TRANSFER PARAGRAPH -- editor to complete]", "",
                    f"{f['editor']}, Handling Editor"]
            notes = [x for x in (author, fee, f"length {length}") if x]
            return "DRAFT DECISION LETTER\n\n" + "\n".join(body) + "\n\nNotes: " + "; ".join(notes) + "\nPortfolio: " + "; ".join(port.values())
        rows = [("Manuscript", f["manuscript_id"]), ("Title", f["title"]), ("Length", length),
                ("Reports", f"{f['n_reports']}, {quality}"), ("Main reason for rejection", f["reason_text"]),
                ("Other concerns", f["secondary_text"]), ("Soundness", self._sound_text(f)),
                ("Author transfer preference", show(author, rng)), ("Fees", fee)]
        return ("TRANSFER CONSIDERATION FORM\n" + kv_lines(rows, rng) + "\n\nAvailable titles:\n"
                + bullets(list(port.values()), rng))

    def questions(self):
        return {
            "transfer_offer": choice("Which transfer, if any, should the editor offer the authors?", {
                "broad_sound_science_title": "Offer the broad-scope title that publishes any technically sound work.",
                "no_transfer": "Make no transfer offer.",
                "short_communications_title": "Offer the short-communications title, as a condensed report.",
                "specialty_title": "Offer the narrower specialty title whose readers the work suits better.",
            }),
            "rejection_basis": choice("What was the main reason the manuscript was turned down?", {
                "incomplete_reporting": "The work could not be assessed because methods, data or code were missing.",
                "limited_advance": "The work is competent but adds too little to what is known.",
                "methodological_flaws": "The design or analysis cannot support the conclusions.",
                "outside_scope": "The work is sound but belongs to a different readership.",
            }),
            "soundness_from_reports": score("How technically sound is the work, going by the reports?", [
                "Unsound: the central analysis does not hold.",
                "Doubtful: problems that need new work.",
                "Largely sound: fixable issues only.",
                "Sound: no technical errors found.",
            ]),
            "rework_before_transfer": score("How much revision would the manuscript need before another title could consider it?", [
                "None: it could be considered as it stands.",
                "Light: presentation or length changes.",
                "Substantial: added analyses or reporting.",
                "Fundamental: a redesigned study or a new dataset.",
            ]),
            "reviews_can_travel": noul(
                "The existing review reports would be useful to a receiving journal.",
                yes="The reports are substantive and assess the work's soundness, so a receiving editor could rely on them.",
                no="The reports are too thin, too tied to this journal's scope, or unusable."),
        }


# =========================================================================== schema 5: appeal of a rejection
APPEAL_GROUNDS = {
    "reviewer_factual_error": "a reviewer misread a result the rejection relied on",
    "reviewer_conflict": "a reviewer works on a competing project and should not have reviewed",
    "editor_misread": "the editor summarised the reports wrongly in the decision letter",
    "new_data": "new data collected since submission answer the main criticism",
    "disagreement": "the authors disagree with the reviewers' view of the work's importance",
}
APPEAL_GROUNDS_LETTER = {
    "reviewer_factual_error": "reviewer 2 misread a result that the rejection relied on",
    "reviewer_conflict": "one reviewer works on a competing project and should not have been asked",
    "editor_misread": "the decision letter misstates what the reviewers actually said",
    "new_data": "we have collected new data that answer the main criticism",
    "disagreement": "we believe the reviewers underrated the importance of the work",
}
APPEAL_POLICY = ("Appeals policy: one appeal per manuscript, lodged within 30 days of the decision, stating specific "
                 "grounds such as a factual error or a conflict of interest; disagreement with the editor's judgement "
                 "of importance is not a ground. Appeals are heard by an editor not involved in the original decision.")


class RejectionAppeal(Schema):
    name = "rejection_appeal"
    description = "Authors appeal a rejection; an uninvolved editor must rule on it under the journal's appeals policy."
    surfaces = {"json": 0.5, "appeal_letter": 0.25, "appeal_case_sheet": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        jk, j = _journal(rng)
        grounds = pick(rng, list(APPEAL_GROUNDS))
        extra_grounds = pick(rng, [None, None, "disagreement"]) if grounds != "disagreement" else None
        verified = pick_w(rng, {"confirmed": 2, "plausible": 3, "unfounded": 2})
        centrality = pick_w(rng, {"central": 3, "contributing": 3, "peripheral": 2})
        decision_kind = pick_w(rng, {"post_review": 4, "desk": 2})
        days_since = rng.randint(3, 45)
        prior_appeal = chance(rng, 0.08)
        tone = pick_w(rng, {"measured": 4, "aggrieved": 2, "threatening": 1})
        variant = None
        if reg == "clear":
            if chance(rng, 0.5):
                grounds, verified, centrality, days_since, prior_appeal = "reviewer_factual_error", "confirmed", "central", rng.randint(3, 20), False
            else:
                grounds, extra_grounds, verified, days_since = "disagreement", None, "unfounded", rng.randint(3, 25)
        elif reg == "borderline":
            days_since = near(rng, 30, 4, lo=10, integer=True)
            verified = pick(rng, ["plausible", "confirmed"])
            centrality = pick(rng, ["contributing", "peripheral"])
        else:
            variant = pick(rng, ["real_error_peripheral", "strong_case_out_of_time", "conflict_unproven_threatening"])
            if variant == "real_error_peripheral":
                grounds, verified, centrality = "reviewer_factual_error", "confirmed", "peripheral"
            elif variant == "strong_case_out_of_time":
                grounds, verified, centrality, days_since = pick(rng, ["reviewer_factual_error", "editor_misread"]), "confirmed", "central", rng.randint(34, 60)
            else:
                grounds, verified, tone = "reviewer_conflict", "plausible", "threatening"
        if grounds == "new_data":
            decision_kind = pick(rng, ["post_review", "desk"])
        received = rand_date(rng)
        return {
            "regime": reg, "variant": variant, "journal_key": jk, "journal": j["name"], "grounds": grounds,
            "extra_grounds": extra_grounds, "claim_verified": verified, "centrality": centrality,
            "decision_kind": decision_kind, "days_since_decision": days_since, "previous_appeal": prior_appeal,
            "tone": tone, "manuscript_id": _ms_id(rng, j, received), "appeal_received": received,
            "author": _author(rng), "appeals_editor": person(rng), "title": pick(rng, j["core"] + j["adjacent"])[6],
        }

    def _check_text(self, f):
        g = f["grounds"]
        return {"confirmed": {"reviewer_factual_error": "the appeals editor confirmed the reviewer misread Table 2",
                              "reviewer_conflict": "the reviewer is a co-investigator on a competing grant",
                              "editor_misread": "the letter attributes to reviewer 1 a criticism neither reviewer made",
                              "new_data": "the new dataset is described in detail and addresses the sample-size criticism",
                              "disagreement": "the reviews are internally consistent"}[g],
                "plausible": {"reviewer_factual_error": "the reviewer's reading is questionable but not clearly wrong",
                              "reviewer_conflict": "the reviewer and the authors' rivals have co-authored before, but not recently",
                              "editor_misread": "the letter compresses the reports in a way that could mislead",
                              "new_data": "new data are promised but only summarised",
                              "disagreement": "the reviews are brief on the importance question"}[g],
                "unfounded": {"reviewer_factual_error": "the reviewer's reading of the result is correct",
                              "reviewer_conflict": "no link between the reviewer and a competing group was found",
                              "editor_misread": "the letter accurately summarises the reports",
                              "new_data": "the 'new data' repeat the submitted analysis",
                              "disagreement": "the reviews explain their view of the work's importance in detail"}[g]}[f["claim_verified"]]

    def _weight_text(self, f):
        if f["grounds"] == "disagreement":
            return {"central": "the judgement of importance was the main reason for rejection",
                    "contributing": "importance was one of several reasons given",
                    "peripheral": "importance was barely mentioned; the rejection rested on other points"}[f["centrality"]]
        return {"central": "the point was the main reason given for rejection",
                "contributing": "the point was one of several reasons given",
                "peripheral": "the point was a side remark in one report"}[f["centrality"]]

    def render(self, f, rng, surface):
        decided = f["appeal_received"] - timedelta(days=f["days_since_decision"])
        grounds = APPEAL_GROUNDS[f["grounds"]] + (f"; also {APPEAL_GROUNDS[f['extra_grounds']]}" if f["extra_grounds"] else "")
        kind = {"post_review": "rejected after peer review", "desk": "rejected without review"}[f["decision_kind"]]
        if surface == "json":
            return dumps(prune({
                "appeal": {"manuscript": f["manuscript_id"], "journal": f["journal"], "received": f["appeal_received"].isoformat(),
                           "original_decision": kind, "decision_date": decided.isoformat(), "grounds": grounds,
                           "earlier_appeal_on_record": f["previous_appeal"] or None},
                "assessment": {"check_of_grounds": self._check_text(f), "role_in_decision": self._weight_text(f)},
                "correspondence": {"tone": f["tone"]},
                "policy": APPEAL_POLICY,
            }))
        if surface == "appeal_letter":
            opener = {"measured": "We respectfully ask the journal to reconsider its decision.",
                      "aggrieved": "We were dismayed by the handling of our manuscript.",
                      "threatening": "Unless this decision is reversed we will raise the matter with the publisher and on social media."}[f["tone"]]
            mine = APPEAL_GROUNDS_LETTER[f["grounds"]] + (f"; and {APPEAL_GROUNDS_LETTER[f['extra_grounds']]}"
                                                          if f["extra_grounds"] else "")
            body = [opener, f"Our manuscript {f['manuscript_id']} (\"{f['title']}\") was {kind} on {fmt_date(decided, rng)}.",
                    f"Our grounds: {mine}."]
            if f["previous_appeal"]:
                body.append("We appealed once before on different grounds.")
            sheet = kv_lines([("Received", fmt_date(f["appeal_received"], rng)),
                              ("Days since decision", f["days_since_decision"]),
                              ("Appeals editor check", self._check_text(f)), ("Weight in decision", self._weight_text(f))], rng)
            return (f"Subject: Appeal -- {f['manuscript_id']}\nFrom: {f['author']}\n\n" + noisy(join_sentences(body), rng, 0.12)
                    + f"\n\n--- appeals desk ---\n{sheet}\n{APPEAL_POLICY}")
        rows = [["Manuscript", f["manuscript_id"]], ["Original decision", kind], ["Decision date", fmt_date(decided, rng)],
                ["Appeal received", fmt_date(f["appeal_received"], rng)], ["Grounds", grounds],
                ["Check of grounds", self._check_text(f)], ["Role in decision", self._weight_text(f)],
                ["Earlier appeal", "yes" if f["previous_appeal"] else "no"], ["Tone", f["tone"]]]
        return (f"APPEAL CASE SHEET -- {f['journal']} (appeals editor {f['appeals_editor']})\n\n"
                + table(["item", "detail"], rows, rng) + f"\n\n{APPEAL_POLICY}")

    def questions(self):
        return {
            "appeal_ruling": choice("How should the appeals editor rule?", {
                "fresh_review": "Send the manuscript to new reviewers and decide again on their reports.",
                "invite_new_submission": "Uphold the rejection but invite a substantially new submission later.",
                "reinstate_for_revision": "Reverse the rejection and invite a revision on the existing reports.",
                "uphold_rejection": "Keep the rejection; the appeal does not change it.",
            }),
            "grounds_substantiated": noul(
                "The appeal identifies a real error or unfairness in how the manuscript was handled.",
                yes="The error or conflict it describes checks out on the record.",
                no="The claim is unfounded, or it is a disagreement about judgement dressed as an error."),
            "error_weight": score("How much did the alleged error or unfairness drive the original decision?", [
                "Not at all: it played no part, or did not happen.",
                "A little: a side point among many.",
                "Considerably: one of the main reasons given.",
                "Decisively: the rejection rested on it.",
            ]),
            "case_strength": score("Taken as a whole, how strong is the authors' case for a different outcome?", [
                "Weak: nothing that would change the decision.",
                "Modest: a fair point that probably does not change the outcome.",
                "Good: a real chance the outcome should change.",
                "Compelling: the original decision cannot stand.",
            ]),
            "appeal_admissible": noul("The appeal is admissible under the journal's appeals policy."),
        }


WORKFLOW = Workflow(
    name="manuscript_triage",
    title="Journal manuscript triage",
    description="An academic journal's editorial decisions on a manuscript: desk screening, integrity concerns, "
                "revision rounds, transfer offers after rejection and appeals.",
    domain="scholarly publishing / editorial management",
    schemas=[DeskScreen(), ResearchEthicsQuery(), RevisionRoundDecision(), TransferAfterRejection(), RejectionAppeal()],
    notes="Journals, portfolios, policies and people are fictional. No reviewer-score reconciliation panels "
          "(grant_review covers those) and no student-assignment integrity cases (academic_integrity_cases).",
)
