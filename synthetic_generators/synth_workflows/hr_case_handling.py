"""HR case handling: employee-relations complaints at fictional employers, from intake to appeal.

Five schemas: routing a new complaint, interim measures while an investigation runs, reaching findings on an
investigator's evidence, deciding a disciplinary appeal, and assessing a report of retaliation after someone
raised a concern. Each samples a `regime` first ('clear' / 'borderline' / 'conflicting') and lets it steer the
factors, so most cases have evidence that is thin or one-sided, wishes that pull against policy, or a key fact
that nobody has checked yet.

Every case is about CONDUCT and PROCEDURE: bullying, abusive language, threats, exclusion from work, out-of-hours
contact, confidentiality breaches, credit and rota disputes, and treatment after a concern was raised. No party's
race, religion, disability, nationality, age or sex is stated, and none shapes any answer. Organisations and
people are fictional.
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
    email_addr,
    fmt_date,
    ident,
    join_sentences,
    kv_lines,
    maybe,
    noisy,
    noul,
    org,
    pick,
    pick_w,
    prune,
    rand_date,
    regime,
    score,
    table,
)

# (department, site)
DEPTS = [
    ("Fulfilment", "Distribution Centre 2"), ("Finance Operations", "head office"),
    ("Field Engineering", "North depot"), ("Facilities", "East campus"), ("Payroll", "head office"),
    ("Product Design", "the design studio"), ("Laboratory Services", "Site Lab 3"), ("Store 14", "Store 14"),
    ("Logistics Planning", "head office"), ("Plant Maintenance", "Plant B"), ("Claims Processing", "Riverside office"),
    ("Kitchen and Catering", "the staff restaurant"),
]

# allegation key -> (gravity 1 low .. 4 very high, the procedure it really belongs to (None = conduct),
#                    [(first-person detail, third-person detail), ...]); {r} respondent, {c} affected employee
ALLEGATIONS = {
    "repeated_belittling": (3, None, [
        ("{r} keeps making remarks about my work in front of the team, calling it 'amateur' and 'a waste of everyone's time'",
         "{r} keeps making remarks about {c}'s work in front of the team, calling it 'amateur' and 'a waste of everyone's time'"),
        ("{r} mocks what I say in stand-ups and has told other people I 'won't last here'",
         "{r} mocks what {c} says in stand-ups and has told other people {c} 'won't last here'")]),
    "abusive_messages": (2, None, [
        ("{r} swore at me in the team chat over a late delivery and called me useless",
         "{r} swore at {c} in the team chat over a late delivery and called {c} useless"),
        ("{r} sent me a string of messages late one evening calling me incompetent and worse",
         "{r} sent {c} a string of messages late one evening calling {c} incompetent and worse")]),
    "single_heated_exchange": (1, None, [
        ("{r} shouted at me on the floor over a missed handover and walked off",
         "{r} shouted at {c} on the floor over a missed handover and walked off"),
        ("{r} and I had a heated argument in the break room and {r} swore at me",
         "{r} and {c} had a heated argument in the break room and {r} swore at {c}")]),
    "exclusion_from_work": (2, None, [
        ("I have been left off meeting invites, the shared rota and project emails that the rest of the team gets",
         "{c} has been left off meeting invites, the shared rota and project emails that the rest of the team gets"),
        ("{r} reassigns my tasks without telling me and leaves me out of planning meetings",
         "{r} reassigns {c}'s tasks without telling them and leaves {c} out of planning meetings")]),
    "out_of_hours_contact": (1, None, [
        ("{r} messages me late at night and at weekends and gets annoyed if I don't answer straight away",
         "{r} messages {c} late at night and at weekends and gets annoyed if {c} doesn't answer straight away"),
        ("{r} phones me on my days off about routine things that could wait",
         "{r} phones {c} on days off about routine things that could wait")]),
    "confidentiality_breach": (2, None, [
        ("{r} told other people on the team about my pay and what was said in my review",
         "{r} told other people on the team about {c}'s pay and what was said in {c}'s review"),
        ("details from my grievance last year have been discussed openly by {r}",
         "details from {c}'s grievance last year have been discussed openly by {r}")]),
    "threatening_remark": (4, None, [
        ("{r} stood in the doorway so I couldn't leave and said I would 'regret' going to HR",
         "{r} stood in a doorway so {c} couldn't leave and said {c} would 'regret' going to HR"),
        ("{r} told me in the car park that I should 'watch myself' after the stock audit",
         "{r} told {c} in the car park to 'watch yourself' after the stock audit")]),
    "credit_for_work": (1, None, [
        ("{r} presented my analysis at the quarterly review as their own work",
         "{r} presented {c}'s analysis at the quarterly review as their own work"),
        ("{r} put their name on the report I wrote and sent it to the director",
         "{r} put their name on a report {c} wrote and sent it to the director")]),
    "overtime_allocation": (1, "rota", [
        ("overtime on my shift keeps going to the same three people and I never get offered any",
         "overtime on {c}'s shift keeps going to the same three people and {c} is never offered any"),
        ("the weekend overtime list is never shared and I only find out afterwards who got it",
         "the weekend overtime list is never shared and {c} only finds out afterwards who got it")]),
    "rating_dispute": (1, "performance", [
        ("my annual rating was dropped to 'partially meets' and I don't think {r} applied the criteria fairly",
         "{c}'s annual rating was dropped to 'partially meets' and {c} doesn't think {r} applied the criteria fairly"),
        ("{r} marked me down on my review against targets that were changed halfway through the year",
         "{r} marked {c} down at review against targets that were changed halfway through the year")]),
}
RELATION_TEXT = {
    "line_manager": "my manager, {r}", "peer": "a colleague on my team, {r}", "direct_report": "{r}, who reports to me",
    "other_team": "{r} from {other}", "senior_leader": "{r}, the head of our area",
}
RELATION_LABEL = {
    "line_manager": "complainant's line manager", "peer": "peer, same team", "direct_report": "reports to complainant",
    "other_team": "works in another team", "senior_leader": "senior leader above complainant's manager",
}
WISH_TEXT = {
    "formal": ["I want this formally investigated.", "I would like a formal investigation please."],
    "informal": ["I just want it to stop. I don't want anyone to get into trouble.",
                 "Could someone have a quiet word? I don't want a big process."],
    "record_only": ["I only want this on record in case it happens again.",
                    "Please just note it for now, I'm not asking you to do anything yet."],
    "move_teams": ["Honestly I'd rather move to another team than go through a process.",
                   "Is there any way I can be moved to a different shift?"],
    "unclear": ["I'm not sure what I want to happen, I just needed to tell someone.", ""],
}
EVIDENCE_TEXT = {
    "screenshots": "I have screenshots of the messages.", "emails": "I've kept the emails.",
    "diary": "I've kept a diary of dates since it started.", "none": "There isn't anything in writing.",
    "cctv_possible": "It would have been on the camera by the loading bay.",
}
CHANNELS = {"complainant": ["HR portal", "email to HR", "in person with HR"], "colleague": ["email to HR"],
            "anonymous": ["Speak-Up line"]}


# who can plausibly be the subject of each kind of concern
ALLOWED_RELATIONS = {
    "repeated_belittling": ["line_manager", "peer", "senior_leader", "other_team"],
    "abusive_messages": ["line_manager", "peer", "direct_report", "other_team"],
    "single_heated_exchange": ["peer", "other_team", "direct_report", "line_manager"],
    "exclusion_from_work": ["line_manager", "peer", "senior_leader"],
    "out_of_hours_contact": ["line_manager", "senior_leader"],
    "confidentiality_breach": ["line_manager", "peer", "other_team"],
    "threatening_remark": ["peer", "line_manager", "direct_report", "other_team"],
    "credit_for_work": ["peer", "line_manager", "other_team", "senior_leader"],
    "overtime_allocation": ["line_manager", "other_team"],
    "rating_dispute": ["line_manager", "senior_leader"],
}
# the kinds of evidence that fit each kind of concern
EVIDENCE_FOR = {
    "repeated_belittling": ["diary", "emails", "none"], "abusive_messages": ["screenshots", "screenshots", "none"],
    "single_heated_exchange": ["cctv_possible", "none", "none"], "exclusion_from_work": ["emails", "diary", "none"],
    "out_of_hours_contact": ["screenshots", "diary", "none"], "confidentiality_breach": ["emails", "screenshots", "none"],
    "threatening_remark": ["cctv_possible", "diary", "none"], "credit_for_work": ["emails", "none"],
    "overtime_allocation": ["emails", "diary", "none"], "rating_dispute": ["emails", "none"],
}
REPEATED = {"repeated_belittling", "out_of_hours_contact", "exclusion_from_work", "overtime_allocation"}
EYEWITNESSED = {"repeated_belittling", "single_heated_exchange", "threatening_remark"}


def _dept(rng):
    return pick(rng, DEPTS)


def _names(rng, n):
    """n fictional people with distinct first AND last names (no two parties in a case share a name)."""
    return [f"{a} {b}" for a, b in zip(rng.sample(FIRST, n), rng.sample(LAST, n))]


def _org_domain(name):
    return name.split()[0].lower() + "-corp.example"


# =========================================================================== schema 1
class ComplaintIntake(Schema):
    name = "complaint_intake"
    description = "A newly raised employee-relations concern at intake: which route it takes, who handles it, how fast."
    surfaces = {"json": 0.5, "intake_form": 0.25, "message": 0.25}

    MINOR = ["single_heated_exchange", "credit_for_work", "overtime_allocation", "out_of_hours_contact"]
    MIDDLE = ["abusive_messages", "exclusion_from_work", "out_of_hours_contact", "confidentiality_breach",
              "repeated_belittling", "rating_dispute"]
    GRAVE = ["threatening_remark", "repeated_belittling"]

    def sample_factors(self, rng):
        reg = regime(rng)
        relation = pick_w(rng, {"line_manager": 4, "peer": 4, "direct_report": 1, "other_team": 2, "senior_leader": 1})
        source = pick_w(rng, {"complainant": 6, "colleague": 2, "anonymous": 2})
        prior = pick(rng, [0, 0, 0, 1, 1, 2])
        share = chance(rng, 0.6)
        if reg == "clear":
            if chance(rng, 0.5):
                alleg = pick(rng, self.MINOR)
                incidents, months, witnesses, evidence = pick(rng, [1, 1, 2]), 1, pick(rng, [0, 1]), "none"
                wish, prior, source = pick(rng, ["informal", "record_only"]), 0, "complainant"
                relation = pick(rng, ["peer", "other_team", "line_manager"])
            else:
                alleg = pick(rng, self.GRAVE)
                incidents, months = pick(rng, [1, 4, 6, 9]), pick(rng, [1, 3, 6])
                witnesses, evidence = pick(rng, [2, 3]), pick(rng, ["screenshots", "cctv_possible", "diary"])
                wish, prior, source, share = "formal", pick(rng, [1, 2]), "complainant", True
        else:
            alleg = pick(rng, self.MIDDLE)
            incidents, months = pick(rng, [1, 2, 3, 4, 6]), pick(rng, [1, 2, 4, 8])
            witnesses = pick(rng, [0, 1, 1, 2])
            evidence = pick_w(rng, {"screenshots": 2, "emails": 2, "diary": 2, "none": 3, "cctv_possible": 1})
            wish = pick_w(rng, {"formal": 3, "informal": 3, "record_only": 1, "move_teams": 2, "unclear": 2})
            if reg == "conflicting":
                twist = pick(rng, ["grave_but_informal", "second_hand", "minor_but_history", "process_as_conduct"])
                if twist == "grave_but_informal":
                    alleg, wish = pick(rng, self.GRAVE), pick(rng, ["informal", "record_only", "move_teams"])
                elif twist == "second_hand":
                    source = pick(rng, ["colleague", "anonymous"])
                    witnesses = None
                elif twist == "minor_but_history":
                    alleg, prior = pick(rng, self.MINOR), 3
                else:
                    alleg, relation, prior = pick(rng, ["overtime_allocation", "rating_dispute"]), "line_manager", 2
                    wish = "formal"
        if source != "complainant" and wish in ("move_teams",):
            wish = "unclear"
        if relation not in ALLOWED_RELATIONS[alleg]:
            relation = pick(rng, ALLOWED_RELATIONS[alleg])
        if alleg in REPEATED:
            incidents = max(incidents, pick(rng, [2, 3, 5]))
            months = max(months, 2)
        elif alleg == "rating_dispute":
            incidents, months = 1, 1
        elif alleg in ("threatening_remark", "single_heated_exchange"):
            incidents = min(incidents, pick(rng, [1, 1, 2]))
        if evidence not in EVIDENCE_FOR[alleg]:
            evidence = pick(rng, EVIDENCE_FOR[alleg])
        days_since = maybe(rng, pick(rng, [0, 1, 2, 5, 12, 30, 75]), 0.35 if reg == "conflicting" else 0.2)
        received = rand_date(rng)
        next_shared = pick(rng, [0, 1, 2, 3, 7]) if share else None
        dept, site = _dept(rng)
        other_dept = pick(rng, [d for d, _ in DEPTS if d != dept])
        people = _names(rng, 6)
        return {
            "regime": reg, "allegation": alleg, "gravity": ALLEGATIONS[alleg][0], "relation": relation,
            "source": source, "incidents": incidents, "period_months": months, "days_since_last": days_since,
            "witnesses_named": witnesses, "evidence": evidence, "wish": wish,
            "respondent_prior_cases_24m": prior, "work_together": share, "next_shared_shift_days": next_shared,
            "department": dept, "site": site, "other_department": other_dept, "received": received,
            "case_ref": f"ER-{received.year}-{rng.randrange(100, 9999):04d}", "complainant": people[0],
            "respondent": people[1], "witness_names": people[2:2 + (witnesses or 0)], "colleague": people[5],
            "employer": org(rng, "company"),
            "channel": pick(rng, CHANNELS[source]), "tone": pick(rng, ["measured", "upset", "angry", "terse"]),
            "detail_idx": rng.randrange(2), "wish_idx": rng.randrange(2),
        }

    def _statement(self, f, rng):
        """The concern in the reporter's voice: the employee (first person), a colleague, or a hotline operator."""
        r, c = f["respondent"].split()[0], f["complainant"].split()[0]
        first, third = ALLEGATIONS[f["allegation"]][2][f["detail_idx"]]
        voice = f["source"]
        s = []
        if voice == "complainant":
            who = RELATION_TEXT[f["relation"]].format(r=f["respondent"], other=f["other_department"])
            s.append(pick(rng, [f"I want to raise a concern about {who}.", f"I'm writing about {who}.",
                                f"I need to report how {who} has been treating me."]))
            d = first.format(r=r)
        else:
            rel = RELATION_LABEL[f["relation"]].replace("complainant's", f"{c}'s").replace("complainant", c)
            s.append(f"I'm raising this for {c}, who doesn't feel able to. It's about {f['respondent']} ({rel})."
                     if voice == "colleague" else
                     f"Caller reports concerns about {f['respondent']} ({rel}) in {f['department']}.")
            d = third.format(r=r, c=c)
        s.append(d[0].upper() + d[1:] + ".")
        a, d_last = f["allegation"], f["days_since_last"]
        if a == "rating_dispute":
            if d_last is not None:
                s.append("The rating came through today." if d_last == 0 else f"The rating was confirmed {d_last} day(s) ago.")
            else:
                s.append("No date is given for when the rating was confirmed.")
        else:
            if a == "overtime_allocation":
                s.append(f"It has happened on about {f['incidents']} overtime rounds over {f['period_months']} months.")
            elif f["incidents"] == 1:
                s.append("It has happened once." if voice == "complainant" else "Described as a single incident.")
            elif voice == "complainant":
                s.append(pick(rng, [f"This has happened about {f['incidents']} times over the last {f['period_months']} months.",
                                    f"It's been going on for {f['period_months']} months, {f['incidents']} times that I can count."]))
            else:
                s.append(f"Said to have happened around {f['incidents']} times over {f['period_months']} months.")
            if d_last is not None:
                if f["incidents"] == 1 and a != "overtime_allocation":
                    s.append("It happened today." if d_last == 0 else f"It happened {d_last} day(s) ago.")
                else:
                    s.append("The latest was today." if d_last == 0 else f"The most recent time was {d_last} day(s) ago.")
            else:
                s.append("I can't say exactly when the last time was." if voice == "complainant"
                         else "No date is given for the latest incident.")
        w, names = f["witnesses_named"], f["witness_names"]
        seen = a in EYEWITNESSED
        if w is None:
            s.append("Not clear who else saw it." if seen else "Not clear whether anyone else can confirm it.")
        elif w == 0:
            s.append(pick(rng, ["Nobody else was around.", "No one else saw it as far as I know."]) if seen
                     else "Nobody else can really confirm it.")
        else:
            verb = ("was there" if w == 1 else "were there") if seen else "can confirm this"
            s.append(f"{', '.join(names)} {verb}.")
        ev = EVIDENCE_TEXT[f["evidence"]]
        if f["evidence"] == "diary" and f["incidents"] == 1:
            ev = "I wrote down what was said straight afterwards."
        if voice != "complainant":
            ev = {"screenshots": f"{c} has screenshots of the messages.", "emails": f"{c} has kept the emails.",
                  "diary": f"{c} has kept a diary of dates.", "none": "Nothing in writing that the reporter knows of.",
                  "cctv_possible": "It may have been caught on the loading-bay camera."}[f["evidence"]]
        s.append(ev)
        if voice == "complainant":
            if f["tone"] == "upset":
                s.append(pick(rng, ["I dread coming in now.", "It's really getting to me."]))
            elif f["tone"] == "angry":
                s.append("I'm not putting up with this any more.")
            s.append(WISH_TEXT[f["wish"]][f["wish_idx"]])
            if f["work_together"] and f["next_shared_shift_days"] is not None:
                s.append("We're on the same shift again today." if f["next_shared_shift_days"] == 0 else
                         f"We're next on shift together in {f['next_shared_shift_days']} day(s).")
        elif voice == "colleague":
            s.append(pick(rng, [f"I don't know if {c} wants a formal complaint.", f"{c} asked me not to make a fuss but I think someone should know."]))
        else:
            s.append("Reporter declined to say whether the affected employee knows about this call.")
        return noisy(join_sentences(s), rng, 0.15)

    def render(self, f, rng, surface):
        first = f["source"] == "complainant"
        stmt = self._statement(f, rng)
        wish = {"formal": "formal investigation", "informal": "informal resolution", "record_only": "note on file only",
                "move_teams": "move to another team", "unclear": None}[f["wish"]] if first else None
        if surface == "json":
            return dumps(prune({
                "case": {"reference": f["case_ref"], "received": f["received"].isoformat(), "channel": f["channel"],
                         "raised_by": {"complainant": "the affected employee", "colleague": "a colleague on their behalf",
                                       "anonymous": "anonymous reporter"}[f["source"]]},
                "parties": {"complainant": ({"name": f["complainant"], "department": f["department"]} if f["source"] != "anonymous"
                                            else {"department": f["department"]}),
                            "respondent": {"name": f["respondent"], "relationship": RELATION_LABEL[f["relation"]],
                                           "er_cases_last_24_months": f["respondent_prior_cases_24m"]}},
                "concern": {"account": stmt, "outcome_sought": wish,
                            "last_incident": ("not stated" if f["days_since_last"] is None
                                              else f"{f['days_since_last']} day(s) before the report")},
                "working_arrangements": {"work_together_day_to_day": f["work_together"],
                                         "next_shared_shift_in_days": f["next_shared_shift_days"],
                                         "site": f["site"]},
            }))
        if surface == "intake_form":
            rows = [("Case ref", f["case_ref"]), ("Date received", fmt_date(f["received"], rng)), ("Channel", f["channel"]),
                    ("Raised by", {"complainant": f["complainant"], "colleague": f"colleague of {f['complainant']}",
                                   "anonymous": "withheld"}[f["source"]]),
                    ("Area", f"{f['department']}, {f['site']}"), ("Subject of concern", f["respondent"]),
                    ("Relationship", RELATION_LABEL[f["relation"]]),
                    ("Prior ER cases (subject, 24m)", f["respondent_prior_cases_24m"]),
                    ("Work together", "yes" if f["work_together"] else "no"),
                    ("Next shared shift", "n/a" if f["next_shared_shift_days"] is None else
                     fmt_date(f["received"] + timedelta(days=f["next_shared_shift_days"]), rng)),
                    ("Outcome sought", wish)]
            return (f"EMPLOYEE RELATIONS -- INTAKE ({f['employer']})\n" + kv_lines(rows, rng, skip_none=False)
                    + "\n\nDescription of concern:\n" + stmt)
        # message: an email from the employee or a colleague, or a hotline operator's note
        if f["source"] == "anonymous":
            return "\n".join([
                f"SPEAK-UP LINE -- report {f['case_ref']}", f"Received: {fmt_date(f['received'], rng)}",
                f"Organisation: {f['employer']} / {f['department']}", "Reporter: anonymous, no call-back number left",
                f"Prior reports naming this person (24 months): {f['respondent_prior_cases_24m']}", "",
                "Operator's summary:", stmt])
        sender = f["complainant"] if first else f["colleague"]
        dom = _org_domain(f["employer"])
        subj = pick(rng, ["Complaint", "Concern about treatment at work", "Need to raise something", "Confidential"])
        return "\n".join([
            f"Subject: {subj}{'' if first else ' (on behalf of a colleague)'}",
            f"From: {sender} <{email_addr(sender, dom)}>",
            f"To: hr-casework@{dom}", f"Date: {fmt_date(f['received'], rng)}", "", "Hi,", "", stmt, "",
            pick(rng, ["Thanks,", "Regards,", "Thank you,"]), sender.split()[0]])

    def questions(self):
        return {
            "intake_route": choice("Which route should this concern take?", {
                "early_resolution": "Resolve it informally: HR or a manager speaks to those involved and agrees how "
                                    "things will change, with no formal process.",
                "formal_investigation": "Open a formal investigation under the conduct policy, with an appointed "
                                        "investigator and interviews.",
                "mediation": "Offer both people mediation with a trained, neutral mediator to repair the working relationship.",
                "record_only": "Log the concern on file and take no further step unless more is raised.",
                "redirect_procedure": "Send it to a different process, such as the rota or performance-review appeal "
                                      "route, because it is not really a conduct complaint.",
            }),
            "allegation_gravity": score("How serious would the conduct described be if it were shown to be true?", [
                "Low: friction or a one-off lapse in courtesy that people can move past.",
                "Moderate: a breach of the conduct policy worth a conversation or a first warning.",
                "High: targeted or repeated conduct that could justify a final warning.",
                "Very high: threats or sustained intimidation that could end someone's employment.",
            ]),
            "case_handler": choice("Who should take charge of this concern?", {
                "hr_partner": "The HR business partner for the area, handling it directly.",
                "independent_investigator": "An investigator from outside the reporting lines of everyone involved.",
                "line_manager": "The line manager of the people involved, with advice from HR.",
                "senior_leader": "A senior leader above the person complained about, because of that person's position.",
            }),
            "first_step_timing": score("How soon does HR need to take a first step on this concern?", [
                "Within the usual two-week acknowledgement period.",
                "Within a few working days, before memories fade.",
                "Before the people involved are next rostered or scheduled together.",
                "Today, because someone may face harm or further pressure in the meantime.",
            ]),
            "actionable_as_reported": noul(
                "The concern as reported gives HR enough detail to act without first going back to whoever raised it.",
                yes="The affected employee reported it first-hand, naming the person, what happened and when it last "
                    "happened.",
                no="It comes second-hand or anonymously, or it gives no date for when it last happened, so HR must "
                   "first go back to whoever raised it."),
        }


# =========================================================================== schema 2
class InterimMeasures(Schema):
    name = "interim_measures"
    description = "An open conduct investigation where HR must decide what, if anything, changes at work until it ends."
    surfaces = {"json": 0.5, "case_note": 0.25, "manager_email": 0.25}

    ALLEG = {"repeated_belittling": "repeated belittling of the complainant in team meetings",
             "threatening_remark": "a threatening remark made to the complainant in the car park",
             "abusive_messages": "abusive messages sent to the complainant in the team chat",
             "confidentiality_breach": "sharing the complainant's pay and review details with the team",
             "altering_records": "changing the complainant's timesheet entries after they had been approved",
             "exclusion_from_work": "cutting the complainant out of rotas, meetings and project emails"}
    SEVERE = {"threatening_remark", "altering_records", "repeated_belittling"}

    def sample_factors(self, rng):
        reg = regime(rng)
        alleg = pick(rng, list(self.ALLEG))
        role = pick_w(rng, {"line_manager": 4, "peer": 3, "shift_supervisor": 2, "manager_other_team": 1})
        setup = pick_w(rng, {"same_team_same_shift": 4, "same_site_other_shift": 3, "different_sites": 1, "mostly_remote": 2})
        contact = pick_w(rng, {"none": 3, "routine_work_only": 4, "tense_encounter": 2, "contacted_witnesses": 1})
        access = chance(rng, 0.3 if alleg == "altering_records" else 0.15)
        request = pick_w(rng, {"nothing_specific": 3, "different_manager": 2, "work_from_other_site": 2,
                               "respondent_removed": 2, "move_me": 1})
        constraint = pick_w(rng, {"none": 4, "only_qualified_supervisor": 2, "small_team_of_four": 2})
        resp_reaction = pick_w(rng, {"denies_calmly": 4, "angry_denial": 2, "partial_admission": 2, "not_yet_informed": 1})
        witnesses_under_resp = pick(rng, [0, 0, 1, 2, 3])
        weeks = pick(rng, [2, 3, 4, 6, 8])
        if reg == "clear":
            if chance(rng, 0.5):
                alleg, setup, contact, access, witnesses_under_resp = "confidentiality_breach", "different_sites", "none", False, 0
                role, request = pick(rng, ["peer", "manager_other_team"]), "nothing_specific"
            else:
                alleg, contact, setup = "threatening_remark", pick(rng, ["tense_encounter", "contacted_witnesses"]), "same_team_same_shift"
                role, resp_reaction = pick(rng, ["line_manager", "shift_supervisor"]), "angry_denial"
        elif reg == "conflicting":
            twist = pick(rng, ["severe_but_irreplaceable", "complainant_wants_move", "access_but_calm", "unknown_contact"])
            if twist == "severe_but_irreplaceable":
                alleg, constraint, setup = pick(rng, sorted(self.SEVERE)), "only_qualified_supervisor", "same_team_same_shift"
            elif twist == "complainant_wants_move":
                request, contact = "move_me", pick(rng, ["routine_work_only", "none"])
            elif twist == "access_but_calm":
                access, contact, resp_reaction, alleg = True, "none", "partial_admission", pick(rng, ["altering_records", "confidentiality_breach"])
            else:
                contact = None
        opened = rand_date(rng)
        return {"regime": reg, "allegation": alleg, "respondent_role": role, "work_setup": setup,
                "contact_since_complaint": contact, "respondent_has_record_access": access,
                "complainant_request": request, "operational_constraint": constraint,
                "respondent_reaction": resp_reaction, "witnesses_reporting_to_respondent": witnesses_under_resp,
                "expected_weeks": weeks, "days_open": pick(rng, [1, 3, 6, 10, 15]), "opened": opened,
                "case_ref": ident(rng, "ER", 5), **dict(zip(["complainant", "respondent", "investigator", "manager"], _names(rng, 4))),
                "dept": _dept(rng)[0],
                "union_rep_involved": chance(rng, 0.3)}

    def _contact_text(self, f, rng):
        c = f["contact_since_complaint"]
        return {None: "Not yet established whether they have had any contact since the complaint.",
                "none": "No contact between them since the complaint was made.",
                "routine_work_only": "They still deal with each other on routine work; nothing untoward reported.",
                "tense_encounter": pick(rng, ["There was a tense exchange at handover two days ago; the complainant says they "
                                              "were told to 'think carefully' about what they say.",
                                              "Colleagues describe a confrontation in the corridor this week, though accounts differ."]),
                "contacted_witnesses": "Two witnesses say the respondent asked them what they had told HR."}[c]

    def render(self, f, rng, surface):
        role = f["respondent_role"].replace("_", " ")
        setup = {"same_team_same_shift": "same team, same shift", "same_site_other_shift": "same site, different shifts",
                 "different_sites": "different sites", "mostly_remote": "both mostly remote"}[f["work_setup"]]
        req = {"nothing_specific": None, "different_manager": "asked to report to a different manager for now",
               "work_from_other_site": "asked to work from another site until it's over",
               "respondent_removed": "asked for the respondent to be sent home",
               "move_me": "asked to be moved to another team"}[f["complainant_request"]]
        cons = {"none": None, "only_qualified_supervisor": "respondent is the only qualified supervisor on that shift",
                "small_team_of_four": "team of four; no other shift to move anyone to"}[f["operational_constraint"]]
        reaction = {"denies_calmly": "denies the allegation, cooperative", "angry_denial": "denies it angrily; called it a stitch-up",
                    "partial_admission": "accepts some of what happened but disputes the context",
                    "not_yet_informed": "not yet told about the complaint"}[f["respondent_reaction"]]
        if surface == "json":
            return dumps(prune({
                "case": {"reference": f["case_ref"], "opened": f["opened"].isoformat(), "days_open": f["days_open"],
                         "allegation": self.ALLEG[f["allegation"]], "investigator": f["investigator"],
                         "expected_duration_weeks": f["expected_weeks"]},
                "parties": {"respondent_role": role, "respondent_response": reaction, "complainant_request": req,
                            "workplace_representative_involved": f["union_rep_involved"]},
                "workplace": {"department": f["dept"], "arrangement": setup, "constraint": cons,
                              "contact_since_complaint": self._contact_text(f, rng)},
                "evidence": {"respondent_can_edit_relevant_records": f["respondent_has_record_access"],
                             "witnesses_managed_by_respondent": f["witnesses_reporting_to_respondent"]},
            }))
        if surface == "case_note":
            rows = [("Case", f["case_ref"]), ("Opened", fmt_date(f["opened"], rng)),
                    ("Allegation", self.ALLEG[f["allegation"]]), ("Respondent", f"{f['respondent']} ({role})"),
                    ("Complainant", f["complainant"]), ("Working arrangement", setup),
                    ("Respondent response", reaction), ("Complainant request", req or "none recorded"),
                    ("Operational constraint", cons), ("Witnesses line-managed by respondent", f["witnesses_reporting_to_respondent"]),
                    ("Respondent access to records in scope", "yes" if f["respondent_has_record_access"] else "no"),
                    ("Investigation ETA", f"{f['expected_weeks']} weeks")]
            return ("CASE NOTE -- interim arrangements review\n" + kv_lines(rows, rng) + "\n\n"
                    + noisy(self._contact_text(f, rng), rng, 0.15))
        together = {"same_team_same_shift": "They're on the same team and the same shift.",
                    "same_site_other_shift": "They're on the same site but on different shifts.",
                    "different_sites": "They work at different sites.",
                    "mostly_remote": "They both work mostly from home."}[f["work_setup"]]
        s = [f"{f['investigator']} is looking into {f['complainant'].split()[0]}'s complaint about "
             f"{f['respondent'].split()[0]} ({f['case_ref']}, {self.ALLEG[f['allegation']]}), and both are still working in {f['dept']}.",
             together, self._contact_text(f, rng)]
        if cons:
            s.append({"only_qualified_supervisor": f"{f['respondent'].split()[0]} is the only one qualified to supervise that shift, "
                                                   "so I'd struggle to cover if they're out.",
                      "small_team_of_four": "There are only four of us, so moving anyone is hard."}[f["operational_constraint"]])
        if req:
            s.append(f"{f['complainant'].split()[0]} has {req}.")
        if f["respondent_has_record_access"]:
            s.append(f"{f['respondent'].split()[0]} still has edit rights on the timesheet system.")
        if f["witnesses_reporting_to_respondent"]:
            s.append(f"{f['witnesses_reporting_to_respondent']} of the witnesses report to {f['respondent'].split()[0]}.")
        s.append(pick(rng, ["My instinct is to keep things as they are, but I want to do this properly.",
                            "What do you want me to do in the meantime?", "Can we talk today?"]))
        return "\n".join([f"Subject: RE: {f['case_ref']} -- what do we do in the meantime?",
                          f"From: {f['manager']}", "To: HR business partner", "",
                          noisy(join_sentences(s), rng, 0.2), "", f["manager"].split()[0]])

    def questions(self):
        return {
            "interim_step": choice("What should change at work while the investigation runs?", {
                "adjust_arrangements": "Change rotas, reporting lines or locations so the two rarely meet, keeping both at work.",
                "move_complainant": "Move the complainant to another team or shift, as they have asked.",
                "no_change": "Leave arrangements as they are and remind both people of the expected conduct.",
                "paid_precautionary_leave": "Send the respondent home on paid precautionary leave until the investigation ends.",
                "restrict_access": "Keep the respondent working but remove their access to specific records, systems or people.",
            }),
            "retaliation_exposure": score("How exposed is the complainant to retaliation or further harm if nothing changes?", [
                "Minimal: the two rarely cross paths and nothing has happened since.",
                "Some: they still meet at work, but contact has been uneventful.",
                "Real: the respondent controls parts of the complainant's work or there has been friction since.",
                "High: there has been a hostile encounter, pressure on witnesses or a threat.",
            ]),
            "evidence_interference_risk": score("How likely is the investigation's evidence to be affected if nothing changes?", [
                "Unlikely: evidence is secured and the witnesses are independent of the respondent.",
                "Slight: some witnesses work under the respondent, but there is no sign of interference.",
                "Moderate: the respondent can reach relevant records or sway several witnesses.",
                "Serious: there are signs of contact with witnesses or of records being changed.",
            ]),
            "respondent_presence_untenable": noul(
                "Keeping the respondent in their current role while the case is open poses a risk that lesser steps cannot manage.",
                yes="The risk to people or to evidence cannot be handled by adjustments short of removing the respondent.",
                no="Any risk can be handled by adjustments that leave the respondent in their role."),
            "burden_on_respondent": noul(
                "Any disruption from interim arrangements in this case should fall on the respondent rather than the complainant.",
                yes="Nothing stands in the way: the complainant has not asked to move themselves, and the respondent "
                    "can be moved without leaving a shift without its only qualified supervisor.",
                no="The complainant has asked to move or work elsewhere themselves, or the respondent is the only "
                   "qualified supervisor on the shift."),
        }


# =========================================================================== schema 3
class InvestigationFindings(Schema):
    name = "investigation_findings"
    description = "An investigator's evidence summary on a conduct allegation, awaiting findings and a recommendation."
    surfaces = {"json": 0.5, "report_memo": 0.25, "evidence_grid": 0.25}

    ALLEG = {"abusive_messages": ("sent abusive messages to a colleague in the team chat", 2),
             "repeated_belittling": ("repeatedly belittled a colleague in team meetings", 3),
             "threatening_remark": ("made a threatening remark to a colleague after an audit", 4),
             "confidentiality_breach": ("disclosed a colleague's pay and review details to others", 2),
             "altering_records": ("edited a report's timesheet entries after approval", 3),
             "exclusion_from_work": ("deliberately excluded a report from rotas and meetings", 2),
             "misuse_of_authority": ("pressured staff to work unpaid overtime", 3)}

    def sample_factors(self, rng):
        reg = regime(rng)
        alleg = pick(rng, list(self.ALLEG))
        support = pick(rng, [0, 1, 1, 2, 3])
        contra = pick(rng, [0, 0, 1, 1, 2])
        ties = pick(rng, [0, 0, 1, 2])
        docs = pick_w(rng, {"confirms": 2, "partly_supports": 3, "ambiguous": 3, "none": 3, "contradicts": 1})
        account = pick_w(rng, {"denies": 4, "partial_admission": 3, "full_admission": 1, "declined_interview": 1})
        consistency = pick_w(rng, {"consistent": 5, "minor_discrepancies": 3, "changed_materially": 1})
        procedure = pick_w(rng, {"fair": 6, "evidence_not_put_to_respondent": 1, "key_witness_not_heard": 2,
                                 "investigator_same_team": 1})
        prior = pick(rng, [0, 0, 0, 1, 1, 2])
        if reg == "clear":
            if chance(rng, 0.5):
                support, contra, ties, docs, account, consistency, procedure = 3, 0, 0, "confirms", pick(rng, ["partial_admission", "full_admission"]), "consistent", "fair"
            else:
                support, contra, docs, account, consistency = 0, 2, pick(rng, ["none", "contradicts"]), "denies", pick(rng, ["minor_discrepancies", "changed_materially"])
        elif reg == "borderline":
            support, contra = pick(rng, [1, 1, 2]), pick(rng, [1, 1, 2])
            docs = pick(rng, ["ambiguous", "partly_supports", "none"])
        else:
            twist = pick(rng, ["strong_but_unfair", "admission_but_weak", "witnesses_tied", "docs_vs_witnesses"])
            if twist == "strong_but_unfair":
                support, docs, procedure = pick(rng, [2, 3]), "confirms", pick(rng, ["evidence_not_put_to_respondent", "key_witness_not_heard", "investigator_same_team"])
            elif twist == "admission_but_weak":
                account, support, docs = "partial_admission", 0, "none"
            elif twist == "witnesses_tied":
                support, ties, contra = 2, 2, 1
            else:
                docs = pick(rng, ["confirms", "contradicts"])
                support, contra = (0, 2) if docs == "confirms" else (2, 0)
        ties = min(ties, support + contra)
        return {"regime": reg, "allegation": alleg, "gravity": self.ALLEG[alleg][1], "witnesses_support": support,
                "witnesses_contradict": contra, "witnesses_with_ties": ties, "documents": docs,
                "respondent_account": account, "complainant_consistency": consistency, "procedure_issue": procedure,
                "live_warnings": prior, "weeks_taken": pick(rng, [2, 4, 5, 7, 11]),
                "case_ref": ident(rng, "INV", 4), "people": _names(rng, 3 + support + contra),
                "employer": org(rng, "company"), "dated": rand_date(rng)}

    RECORD_OF = {"abusive_messages": "The team-chat export", "repeated_belittling": "Meeting notes and one meeting recording",
                 "threatening_remark": "Car-park CCTV", "confidentiality_breach": "Email and chat logs",
                 "altering_records": "The timesheet audit trail", "exclusion_from_work": "Rota and calendar records",
                 "misuse_of_authority": "Overtime and door-access records"}
    DOC_TEXT = {"confirms": "{rec} shows the conduct much as described.",
                "partly_supports": "{rec} confirms the date and that something happened, but not what was said or done.",
                "ambiguous": "{rec} exists but can be read either way.",
                "none": "Nothing usable: {rec_l} was not retained or does not cover the time.",
                "contradicts": "{rec} does not fit the complainant's account of the timing."}

    def _doc(self, f):
        rec = self.RECORD_OF[f["allegation"]]
        return self.DOC_TEXT[f["documents"]].format(rec=rec, rec_l=rec[0].lower() + rec[1:])

    ACCOUNT_TEXT = {"denies": "Denies the allegation in full.",
                    "partial_admission": "Accepts part of what is alleged but says it was provoked and has been taken out of context.",
                    "full_admission": "Accepts the conduct and has apologised in writing.",
                    "declined_interview": "Declined to attend two interview dates; written response not received."}
    CONS_TEXT = {"consistent": "Complainant's account consistent across intake and interview.",
                 "minor_discrepancies": "Complainant's dates shift by a few days between intake and interview.",
                 "changed_materially": "Complainant's account at interview differs from intake on who was present and what was said."}
    PROC_TEXT = {"fair": None,
                 "evidence_not_put_to_respondent": "The chat export was obtained after the respondent's interview and has not been put to them.",
                 "key_witness_not_heard": "One person the complainant named as present was on leave and has not been interviewed.",
                 "investigator_same_team": "The investigator works in the same team as the complainant."}

    def _witness_rows(self, f, rng):
        """Witness rows; a tie to one side is noted on a witness who backs that side (credibility questions)."""
        names = f["people"][3:]
        rows = [[n, "supports complainant's account", ""] for n in names[:f["witnesses_support"]]]
        rows += [[n, pick(rng, ["did not see anything of the kind", "recalls it differently"]), ""]
                 for n in names[f["witnesses_support"]:]]
        for i in range(f["witnesses_with_ties"]):
            sup = [r for r in rows if r[1].startswith("supports") and not r[2]]
            con = [r for r in rows if not r[1].startswith("supports") and not r[2]]
            if sup and (not con or chance(rng, 0.5)):
                pick(rng, sup)[2] = pick(rng, ["close friend of complainant", "shares a car with complainant"])
            elif con:
                pick(rng, con)[2] = pick(rng, ["reports to respondent", "respondent's former team-mate"])
        rng.shuffle(rows)
        return rows

    def render(self, f, rng, surface):
        text = self.ALLEG[f["allegation"]][0]
        rows = self._witness_rows(f, rng)
        f = dict(f, investigator=f["people"][0], respondent=f["people"][1])
        if surface == "json":
            return dumps(prune({
                "investigation": {"reference": f["case_ref"], "investigator": f["investigator"], "report_date": f["dated"].isoformat(),
                                  "weeks_taken": f["weeks_taken"], "standard_of_proof": "balance of probabilities",
                                  "procedural_note": self.PROC_TEXT[f["procedure_issue"]]},
                "allegation": {"respondent": f["respondent"], "summary": f"It is alleged that the respondent {text}."},
                "evidence": {"witnesses": [{"name": r[0], "account": r[1], **({"note": r[2]} if r[2] else {})} for r in rows],
                             "documents": self._doc(f),
                             "complainant_consistency": self.CONS_TEXT[f["complainant_consistency"]],
                             "respondent_account": self.ACCOUNT_TEXT[f["respondent_account"]]},
                "record": {"live_warnings_on_file": f["live_warnings"]},
            }))
        if surface == "report_memo":
            paras = [f"INVESTIGATION REPORT {f['case_ref']} -- {f['employer']}",
                     f"Investigator: {f['investigator']}    Date: {fmt_date(f['dated'], rng)}", "",
                     f"1. Allegation. That {f['respondent']} {text}.",
                     ("2. Witnesses. " + (f"{len(rows)} interviewed. " + " ".join(
                         f"{r[0]} {r[1]}{' (' + r[2] + ')' if r[2] else ''}." for r in rows) if rows else "No witnesses identified.")),
                     f"3. Documents. {self._doc(f)}",
                     f"4. Complainant. {self.CONS_TEXT[f['complainant_consistency']]}",
                     f"5. Respondent. {self.ACCOUNT_TEXT[f['respondent_account']]} "
                     f"Live warnings on file: {f['live_warnings']}."]
            if self.PROC_TEXT[f["procedure_issue"]]:
                paras.append(f"6. Limitations. {self.PROC_TEXT[f['procedure_issue']]}")
            return "\n".join(paras)
        grid = table(["witness", "account", "note"], rows, rng) if rows else "(no witnesses)"
        return "\n".join([
            f"Evidence summary {f['case_ref']} | respondent: {f['respondent']} | allegation: {text}", "", grid, "",
            kv_lines([("Documents", self._doc(f)), ("Complainant", self.CONS_TEXT[f["complainant_consistency"]]),
                      ("Respondent", self.ACCOUNT_TEXT[f["respondent_account"]]), ("Live warnings", f["live_warnings"]),
                      ("Process note", self.PROC_TEXT[f["procedure_issue"]])], rng)])

    def questions(self):
        return {
            "allegation_finding": choice("What finding should the investigator reach on the allegation?", {
                "insufficient_evidence": "No finding either way; the evidence is too thin or too evenly balanced.",
                "not_upheld": "The evidence shows, on balance, that the conduct did not happen as alleged.",
                "partly_upheld": "Some of the alleged conduct is shown, but not all of it or not as described.",
                "upheld": "The evidence shows, on balance, that the conduct happened as alleged.",
            }),
            "evidence_weight": score("How strong is the evidence that the alleged conduct took place?", [
                "Weak: it rests on one account that other evidence does not bear out.",
                "Mixed: accounts conflict and nothing independent settles them.",
                "Fairly strong: independent witnesses or records back most of the account.",
                "Compelling: documents, independent witnesses or an admission leave little doubt.",
            ]),
            "recommended_next_step": choice("What should the investigator recommend next?", {
                "close_no_action": "Close the case with no action against the respondent.",
                "disciplinary_hearing": "Refer the respondent to a formal disciplinary hearing.",
                "further_enquiries": "Reopen enquiries first, such as hearing a missed witness or putting new evidence to a party.",
                "management_action": "Deal with it outside the disciplinary process through coaching, training or a clear expectations letter.",
            }),
            "sanction_indicated": score("If the conduct were proven at a hearing, what outcome would it point to?", [
                "Informal guidance or training only.",
                "A first written warning.",
                "A final written warning.",
                "Dismissal.",
            ]),
            "process_was_fair": noul(
                "The investigation gave both parties a fair hearing.",
                yes="Both parties heard the evidence against them and could respond, relevant witnesses were heard and "
                    "the investigator had no conflict.",
                no="A party did not see material evidence, a relevant witness was skipped, or the investigator was not independent."),
        }


# =========================================================================== schema 4
class DisciplinaryAppeal(Schema):
    name = "disciplinary_appeal"
    description = "An employee's appeal against a disciplinary sanction, to be decided by an appeal manager."
    surfaces = {"json": 0.5, "appeal_letter": 0.25, "appeal_pack": 0.25}

    MISCONDUCT = {"timesheet_irregularities": ("claimed 11 hours not worked over three weeks", 3),
                  "abusive_language": ("swore at a colleague in front of a visitor", 2),
                  "safety_rule_breach": ("drove a forklift without the required pre-use check", 2),
                  "unauthorised_absence": ("failed to attend two shifts without notice", 2),
                  "vehicle_misuse": ("used a company van for a personal house move", 2),
                  "confidentiality_breach": ("forwarded a staff salary spreadsheet to a personal address", 3)}
    SANCTIONS = ["first_written_warning", "final_written_warning", "demotion", "dismissal"]

    def sample_factors(self, rng):
        reg = regime(rng)
        mis = pick(rng, list(self.MISCONDUCT))
        grav = self.MISCONDUCT[mis][1]
        sanction = pick_w(rng, {"first_written_warning": 2, "final_written_warning": 3, "demotion": 1, "dismissal": 3})
        years = pick(rng, [0.8, 2, 4, 7, 12, 19])
        live = pick(rng, [0, 0, 0, 1, 2])
        defect = pick_w(rng, {"none": 4, "outcome_letter_late": 2, "no_companion_offered": 1, "evidence_not_shared": 2,
                              "same_person_investigated_and_decided": 1})
        new_ev = pick_w(rng, {"none": 4, "new_witness": 2, "system_logs": 1, "character_references": 2})
        comparator = pick_w(rng, {"none_cited": 4, "lesser_sanction_similar": 2, "same_sanction_similar": 1, "claimed_unverified": 2})
        grounds = []
        days_lodged = pick(rng, [2, 4, 5, 6, 9])
        if reg == "clear":
            if chance(rng, 0.5):
                defect, new_ev, comparator, live = "none", "character_references", "none_cited", pick(rng, [1, 2])
                sanction = "final_written_warning" if grav < 3 else "dismissal"
                days_lodged = 3
            else:
                defect = pick(rng, ["evidence_not_shared", "same_person_investigated_and_decided"])
                new_ev, sanction, live = "system_logs", "dismissal", 0
        elif reg == "borderline":
            sanction = "dismissal" if live >= 1 else pick(rng, ["final_written_warning", "dismissal"])
        else:
            twist = pick(rng, ["long_service_serious", "flaw_but_admitted", "late_but_strong", "comparator"])
            if twist == "long_service_serious":
                years, live, mis, sanction = pick(rng, [12, 19]), 0, pick(rng, ["timesheet_irregularities", "confidentiality_breach"]), "dismissal"
            elif twist == "flaw_but_admitted":
                defect, new_ev = pick(rng, ["outcome_letter_late", "no_companion_offered"]), "none"
            elif twist == "late_but_strong":
                days_lodged, new_ev = pick(rng, [8, 11]), "system_logs"
            else:
                comparator = pick(rng, ["lesser_sanction_similar", "claimed_unverified"])
        grav = self.MISCONDUCT[mis][1]
        if defect != "none":
            grounds.append("procedure")
        if new_ev in ("new_witness", "system_logs"):
            grounds.append("new_evidence")
        if comparator in ("lesser_sanction_similar", "claimed_unverified"):
            grounds.append("inconsistent_treatment")
        if not grounds or chance(rng, 0.5):
            grounds.append("too_harsh")
        return {"regime": reg, "misconduct": mis, "gravity": grav, "sanction": sanction, "service_years": years,
                "live_warnings_before": live, "procedural_defect": defect, "new_evidence": new_ev,
                "comparator": comparator, "grounds": grounds, "days_to_lodge": days_lodged, "deadline_days": 5,
                "admitted_at_hearing": pick_w(rng, {"yes": 2, "partly": 2, "no": 3}),
                **dict(zip(["employee", "decision_maker", "appeal_manager", "new_witness"], _names(rng, 4))),
                "case_ref": ident(rng, "DA", 4), "hearing_date": rand_date(rng), "dept": _dept(rng)[0]}

    RECORDS = {"timesheet_irregularities": "badge and system logs showing presence on site for most of the disputed hours",
               "abusive_language": "the visitor's written note saying they did not hear anything offensive",
               "safety_rule_breach": "the checklist app export showing a pre-use check logged that morning",
               "unauthorised_absence": "phone records showing two calls to the shift line before each missed shift",
               "vehicle_misuse": "a fleet booking showing the van was signed out for a site delivery that day",
               "confidentiality_breach": "mail logs showing the address was a shared work mailbox, not a personal one"}

    def _new_material(self, f, first_person=False):
        ev = f["new_evidence"]
        if ev == "none":
            return None
        if ev == "character_references":
            return "three character references from colleagues"
        if ev == "new_witness":
            return (f"a statement from {f['new_witness']}, who was not interviewed, saying "
                    + ("I was" if first_person else "the employee was") + " told the practice was allowed")
        return self.RECORDS[f["misconduct"]]

    DEFECT = {"none": None, "outcome_letter_late": "outcome letter sent nine working days after the hearing (policy: five)",
              "no_companion_offered": "invitation letter did not mention the right to be accompanied",
              "evidence_not_shared": "the witness statements were not given to the employee before the hearing",
              "same_person_investigated_and_decided": "the manager who investigated also chaired the hearing"}
    COMP = {"none_cited": None, "lesser_sanction_similar": "HR records confirm a colleague received a first warning last year for similar conduct",
            "same_sanction_similar": "a colleague received the same sanction for similar conduct last year",
            "claimed_unverified": "employee says others have done the same without action; not checked"}

    def render(self, f, rng, surface):
        what = self.MISCONDUCT[f["misconduct"]][0]
        sanc = f["sanction"].replace("_", " ")
        late = f["days_to_lodge"] > f["deadline_days"]
        if surface == "json":
            return dumps(prune({
                "original_decision": {"reference": f["case_ref"], "finding": f"The employee {what}.", "sanction": sanc,
                                      "decided_by": f["decision_maker"], "hearing_date": f["hearing_date"].isoformat(),
                                      "employee_admitted": f["admitted_at_hearing"]},
                "employee": {"name": f["employee"], "department": f["dept"], "service_years": f["service_years"],
                             "live_warnings_before_this_case": f["live_warnings_before"]},
                "appeal": {"grounds": f["grounds"], "lodged_working_days_after_outcome": f["days_to_lodge"],
                           "policy_deadline_working_days": f["deadline_days"], "new_material": self._new_material(f),
                           "comparison_case": self.COMP[f["comparator"]]},
                "hr_file_check": {"procedural_issue_found": self.DEFECT[f["procedural_defect"]]},
            }))
        if surface == "appeal_letter":
            s = [f"Dear {f['appeal_manager']},", "",
                 f"I am appealing against the {sanc} I was given after the hearing on {fmt_date(f['hearing_date'], rng)}."]
            if "procedure" in f["grounds"] and f["procedural_defect"] != "none":
                s.append({"outcome_letter_late": "The outcome letter came much later than the policy says.",
                          "no_companion_offered": "Nobody told me I could bring someone with me.",
                          "evidence_not_shared": "I never saw the witness statements before the hearing.",
                          "same_person_investigated_and_decided": f"{f['decision_maker']} investigated me and then decided the case."}[f["procedural_defect"]])
            if f["new_evidence"] != "none":
                s.append(f"I am attaching {self._new_material(f, first_person=True)}.")
            if "inconsistent_treatment" in f["grounds"]:
                s.append("Other people have done the same and got a lot less, or nothing.")
            if "too_harsh" in f["grounds"]:
                yrs = "less than a year" if f["service_years"] < 1 else f"{f['service_years']} years"
                s.append(f"I have worked here {yrs} and I think this is far too harsh.")
            if late:
                s.append("Sorry this is later than the deadline, I was waiting for the documents.")
            body = "\n\n".join([s[0], noisy(join_sentences(s[2:]), rng, 0.2)])
            head = f"Subject: Appeal -- {f['case_ref']}\nReceived: {f['days_to_lodge']} working days after outcome (deadline {f['deadline_days']})\n\n"
            return head + body + f"\n\nYours sincerely,\n{f['employee']}"
        rows = [("Case", f["case_ref"]), ("Employee", f"{f['employee']}, {f['dept']}"),
                ("Service", f"{f['service_years']} years"), ("Finding", what), ("Sanction", sanc),
                ("Admitted at hearing", f["admitted_at_hearing"]), ("Prior live warnings", f["live_warnings_before"]),
                ("Grounds", ", ".join(g.replace("_", " ") for g in f["grounds"])),
                ("Lodged", f"day {f['days_to_lodge']} (deadline day {f['deadline_days']})"),
                ("New material", self._new_material(f)), ("HR check of procedure", self.DEFECT[f["procedural_defect"]] or "no issues noted"),
                ("Comparator", self.COMP[f["comparator"]])]
        return "APPEAL PACK -- prepared for the appeal manager\n" + kv_lines(rows, rng, skip_none=True)

    def questions(self):
        return {
            "appeal_result": choice("How should the appeal manager decide the appeal?", {
                "overturn": "Overturn the sanction and remove it from the employee's record.",
                "refer_back": "Send the case back for the new material to be investigated before any decision.",
                "rehear": "Order a fresh disciplinary hearing before a different manager.",
                "substitute_lesser": "Keep the finding but replace the sanction with a lower one.",
                "uphold": "Confirm the original finding and sanction.",
            }),
            "sanction_harshness": score("How harsh was the original sanction for the misconduct that was found?", [
                "Lenient for what was found.",
                "Proportionate.",
                "On the harsh side; a lower level was open to a reasonable manager.",
                "Plainly excessive for the conduct and the employee's record.",
            ]),
            "procedural_flaw_impact": score("How much did any flaw in the original process undermine the decision?", [
                "No flaw of any consequence.",
                "A technical slip that changed nothing.",
                "A flaw that may have affected the outcome.",
                "A flaw serious enough to make the decision unsafe.",
            ]),
            "grounds_could_change_outcome": noul(
                "At least one ground of appeal could realistically change the outcome.",
                yes="A ground goes to the heart of the case: records or a new witness contradicting the finding, "
                    "evidence withheld from the employee, the investigator also deciding the case, or an HR-confirmed "
                    "lighter sanction for similar conduct.",
                no="The grounds are only a late outcome letter, a missing note about a companion, character "
                   "references, an unchecked claim about others, or a plea that the sanction is harsh."),
            "finding_is_safe": noul(
                "The finding of misconduct itself is safe, whatever happens to the sanction.",
                yes="The employee admitted the conduct at the hearing, or nothing new and no serious procedural flaw "
                    "undercuts the evidence the finding rests on.",
                no="Records or a new witness contradict the facts found, or the employee never saw the evidence or "
                   "the investigator also decided the case, and the employee did not admit the conduct."),
        }


# =========================================================================== schema 5
class RetaliationReport(Schema):
    name = "retaliation_report"
    description = "An employee who raised a concern reports worse treatment since; HR must judge the link and respond."
    surfaces = {"json": 0.5, "employee_email": 0.25, "triage_note": 0.25}

    CONCERNS = {"safety_report": "reported a blocked fire exit and missing guard rails to the site safety lead",
                "expense_concern": "raised concerns with internal audit about how a manager was coding client entertainment expenses",
                "grievance_about_manager": "lodged a grievance about {their} manager's handling of leave requests",
                "witness_statement": "gave a witness statement in a colleague's bullying investigation",
                "data_handling": "reported to the data protection officer that client files were being shared on personal drives"}
    TOLD_BY = {"safety_report": "the site safety lead", "expense_concern": "internal audit",
               "grievance_about_manager": "HR", "witness_statement": "the investigator",
               "data_handling": "the data protection officer"}
    ACTIONS = {"shift_pattern_changed": "moved from day shifts to a rotating night pattern",
               "overtime_withdrawn": "taken off the overtime list",
               "rating_lowered": "given a lower end-of-year rating than the previous two years",
               "excluded_from_project": "removed from the migration project team",
               "disciplinary_started": "invited to a disciplinary investigation over punctuality",
               "probation_extended": "had their probation extended by three months"}

    def sample_factors(self, rng):
        reg = regime(rng)
        concern = pick(rng, list(self.CONCERNS))
        action = pick(rng, list(self.ACTIONS))
        gap = pick(rng, [6, 14, 30, 45, 90, 160])
        knew = pick_w(rng, {"yes": 4, "no": 2, "unclear": 3})
        reason = pick_w(rng, {"predates_concern": 3, "written_after_concern": 3, "none_documented": 2})
        team_wide = chance(rng, 0.25)
        perf = pick_w(rng, {"strong": 3, "mixed": 3, "documented_issues": 2})
        effect_days = pick(rng, [0, 2, 5, 14, 30])
        if reg == "clear":
            if chance(rng, 0.5):
                gap, knew, reason, team_wide, perf = pick(rng, [6, 14]), "yes", "none_documented", False, "strong"
            else:
                gap, knew, reason, team_wide = pick(rng, [90, 160]), "no", "predates_concern", True
        elif reg == "conflicting":
            twist = pick(rng, ["close_timing_but_reason", "team_wide_but_singled", "weak_perf_close_timing"])
            if twist == "close_timing_but_reason":
                gap, knew, reason = pick(rng, [6, 14]), "yes", "predates_concern"
            elif twist == "team_wide_but_singled":
                team_wide, knew, reason = True, "yes", "written_after_concern"
            else:
                perf, gap, reason = "documented_issues", pick(rng, [14, 30]), "written_after_concern"
        raised = rand_date(rng)
        return {"regime": reg, "concern": concern, "adverse_action": action, "days_between": gap,
                "decision_maker_knew": knew, "business_reason": reason, "applied_team_wide": team_wide,
                "prior_performance": perf, "takes_effect_in_days": effect_days, "concern_raised": raised,
                **dict(zip(["employee", "decision_maker", "hr_triager"], _names(rng, 3))),
                "case_ref": ident(rng, "RT", 4), "dept": _dept(rng)[0],
                "concern_upheld": pick_w(rng, {"upheld": 2, "not_upheld": 1, "still_open": 3}), "knew_idx": rng.randrange(3)}

    KNEW = {"yes": ["{dm} was copied on {by}'s acknowledgement of the concern",
                    "{dm} attended the meeting where {by} discussed the concern",
                    "{dm} was interviewed by {by} about the concern two weeks before the decision"],
            "no": ["{dm} took over the team after the concern was raised; {by} handled it confidentially",
                   "{dm} says they were not aware of it; the concern was logged confidentially with {by}",
                   "the decision was signed off by {dm} in another region, who has no contact with {by}"],
            "unclear": ["{dm} was copied on a thread that mentions 'the issue raised' but may not have read it",
                        "the employee says they mentioned it to {dm} in passing; {dm} does not recall that",
                        "{dm} sits in the same weekly leadership meeting as {by}; minutes do not mention the concern"]}
    # (reason recorded BEFORE the concern, what a note written AFTER it cites)
    REASON = {"shift_pattern_changed": ("a rota review dated before the concern moves this role to rotating nights", "rota coverage"),
              "overtime_withdrawn": ("an overtime budget cut approved before the concern halves the overtime list", "the overtime budget"),
              "rating_lowered": ("calibration notes dated before the concern already flag two missed objectives", "missed objectives"),
              "excluded_from_project": ("a project re-plan dated before the concern cut the team from six to four", "a smaller project team"),
              "disciplinary_started": ("punctuality records from before the concern show repeated late starts", "late starts"),
              "probation_extended": ("a probation review dated before the concern notes gaps in required training", "training gaps")}

    def _reason(self, f):
        before, cited = self.REASON[f["adverse_action"]]
        return {"predates_concern": before,
                "written_after_concern": f"a note justifying the change, dated after the concern was raised, cites {cited}",
                "none_documented": "no written business reason found"}[f["business_reason"]]

    PERF = {"strong": "met or exceeded objectives in each of the last two reviews",
            "mixed": "mixed reviews; one missed objective last year",
            "documented_issues": "two documented conversations about punctuality this year"}

    def render(self, f, rng, surface):
        action_date = f["concern_raised"] + timedelta(days=f["days_between"])
        knew = self.KNEW[f["decision_maker_knew"]][f["knew_idx"]].format(dm=f["decision_maker"], by=self.TOLD_BY[f["concern"]])
        concern3 = self.CONCERNS[f["concern"]].format(their="their")
        if surface == "json":
            return dumps(prune({
                "report": {"reference": f["case_ref"], "employee": f["employee"], "department": f["dept"],
                           "complaint": f"Says they were {self.ACTIONS[f['adverse_action']]} because they "
                                        f"{concern3}."},
                "timeline": {"concern_raised": f["concern_raised"].isoformat(), "adverse_decision": action_date.isoformat(),
                             "change_takes_effect_in_days": f["takes_effect_in_days"],
                             "original_concern_status": f["concern_upheld"].replace("_", " ")},
                "decision": {"made_by": f["decision_maker"], "awareness_of_concern": knew,
                             "stated_business_reason": self._reason(f),
                             "applied_to_others": "same change applied across the team" if f["applied_team_wide"] else "only this employee"},
                "employee_record": {"performance": self.PERF[f["prior_performance"]]},
            }))
        if surface == "employee_email":
            s = [f"Back on {fmt_date(f['concern_raised'], rng)} I {self.CONCERNS[f['concern']].format(their='my')}.",
                 f"On {fmt_date(action_date, rng)} I was {self.ACTIONS[f['adverse_action']]}.",
                 pick(rng, ["I don't think that's a coincidence.", "Nobody has given me a proper reason.",
                            "I was told it was 'a business decision' and nothing more."]),
                 {"strong": "My reviews have always been good.", "mixed": "My reviews have been fine.",
                  "documented_issues": "I know I've had a couple of conversations about timekeeping but that was sorted."}[f["prior_performance"]]]
            if f["applied_team_wide"]:
                s.append("I know a couple of others got the same change, but they didn't get singled out like I did.")
            if f["takes_effect_in_days"] == 0:
                s.append("It starts today.")
            elif f["takes_effect_in_days"] <= 5:
                s.append(f"It kicks in within {f['takes_effect_in_days']} days.")
            else:
                s.append(f"It takes effect in {f['takes_effect_in_days']} days.")
            return "\n".join([f"Subject: Being treated differently since I raised a concern", f"From: {f['employee']}",
                              "To: HR", "", noisy(join_sentences(s), rng, 0.2), "", f["employee"].split()[0]])
        rows = [("Ref", f["case_ref"]), ("Employee", f"{f['employee']} ({f['dept']})"),
                ("Original concern", concern3), ("Raised", fmt_date(f["concern_raised"], rng)),
                ("Status of concern", f["concern_upheld"].replace("_", " ")),
                ("Adverse change", self.ACTIONS[f["adverse_action"]]), ("Decided", fmt_date(action_date, rng)),
                ("Effective", "today" if f["takes_effect_in_days"] == 0 else f"in {f['takes_effect_in_days']} days"),
                ("What the decision-maker knew", knew), ("Business reason on file", self._reason(f)),
                ("Applied to others", "yes" if f["applied_team_wide"] else "no"), ("Performance record", self.PERF[f["prior_performance"]])]
        return f"RETALIATION REPORT -- triage by {f['hr_triager']}\n" + kv_lines(rows, rng)

    def questions(self):
        return {
            "link_plausibility": score("How plausible is it that the worse treatment is connected to the concern the employee raised?", [
                "Implausible: the change was fixed beforehand or has nothing to do with the concern.",
                "Unlikely: the timing or awareness does not really fit.",
                "Possible: timing and awareness fit, but there are other credible explanations.",
                "Likely: the timing, awareness and lack of a prior reason all point the same way.",
                "Strongly indicated: the change singles the employee out soon after the concern, with no genuine reason.",
            ]),
            "retaliation_response": choice("How should HR respond to this report?", {
                "close_with_explanation": "Explain the documented business reasons and close: a reason dated before "
                                          "the concern, or a team-wide change, and the decision-maker did not "
                                          "clearly know of the concern.",
                "independent_review": "Have a manager outside the chain review the adverse decision on its merits, "
                                      "because the signs point both ways.",
                "pause_and_investigate": "Put the change on hold and investigate: the decision-maker knew, the change "
                                         "came within a month of the concern, and no reason predates the concern.",
                "reverse_now": "Reverse the change at once and then establish the facts: all of those signs are present "
                               "and the change takes effect within two days.",
            }),
            "hold_window": score("How pressing is it to act before the change takes effect?", [
                "Not pressing: the change takes effect a month or more from now.",
                "Fairly soon: it takes effect in one to three weeks.",
                "Pressing: it takes effect within the next few days.",
                "Immediate: it takes effect today.",
            ]),
            "decision_maker_aware": noul(
                "The person who made the adverse decision knew about the employee's concern when they made it.",
                yes="The record or timing shows the decision-maker had been told of the concern, or was named in it.",
                no="Nothing shows the decision-maker knew, or the decision was fixed before the concern was raised."),
            "would_have_happened_anyway": noul(
                "The same change would have happened anyway for documented business reasons.",
                yes="A business reason dated before the concern already called for the change, or the same change was "
                    "applied across the whole team.",
                no="The only reason on file was written after the concern, or none is documented, and the change fell "
                   "on this employee alone."),
        }


WORKFLOW = Workflow(
    name="hr_case_handling",
    title="HR case handling",
    description="How employee-relations complaints, interim measures, investigation findings, disciplinary appeals "
                "and retaliation reports should be handled at an employer.",
    domain="human resources / employee relations",
    schemas=[ComplaintIntake(), InterimMeasures(), InvestigationFindings(), DisciplinaryAppeal(), RetaliationReport()],
    notes="Conduct and procedure only; no case turns on a protected attribute and none is stated. No customer "
          "threads, no security alerts, no expense claims, no AI-agent runs.",
)
