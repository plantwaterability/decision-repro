"""Grant review: how a funding body screens, scores and manages research and community grants.

Six schemas follow an award from application to delivery: the first-round eligibility screen, a panel
reconciling reviewers who disagree, the annual progress report that gates the next tranche, a request for more
time or a budget transfer, a small city arts grant, and a panel member's conflict-of-interest check. Every
funder, institution and person is fictional. The call's or fund's own rules are always IN the state, so
eligibility questions are judged against stated terms rather than outside knowledge. Each schema draws a
`regime` first ('clear' / 'borderline' / 'conflicting') and lets it steer the factors, as the reference does.
Topics carry tags (lab equipment, fieldwork, human participants, archives) so that milestones, delay causes and
departments always fit the project they are attached to.
"""
from __future__ import annotations

from datetime import date, timedelta

from .base import (
    ORG_A,
    Schema,
    Workflow,
    between,
    bullets,
    chance,
    choice,
    dumps,
    dumps_pretty,
    email_addr,
    fmt_date,
    ident,
    join_sentences,
    kv_lines,
    maybe,
    money,
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
    table,
)

# =========================================================================== shared content
CURRENCIES = {"USD": 5, "GBP": 3, "EUR": 2, "CAD": 1}
FUNDER_SUFFIX = ["Research Council", "Foundation", "Science Trust", "Charitable Trust", "Research Fund"]
SALARY = {"USD": 72000, "GBP": 52000, "EUR": 60000, "CAD": 74000}   # rough full cost of one postdoc-year

# research calls. on-theme aims: (title, verb phrase, theme index, tags); off-theme aims: (title, verb phrase,
# a side aim an applicant might bolt on to reach a call theme, that theme's index)
CALLS = [
    {"code": "RC", "title": "Resilient Coasts",
     "themes": ["flood risk and defences", "coastal habitat change", "community adaptation planning"],
     "on": [("Saltmarsh loss and household flood exposure",
             "measure how saltmarsh loss changes flood exposure for low-lying homes", 1, ("field",)),
            ("Planning managed retreat with fishing towns",
             "co-design managed-retreat plans with three fishing towns", 2, ("people",)),
            ("Storm-surge warnings for small harbours",
             "build and test a storm-surge early-warning model for small harbours", 0, ("field", "equip"))],
     "off": [("Deep-ocean carbon over the last glacial cycle",
              "reconstruct deep-ocean carbon uptake over the last glacial cycle",
              "use the same sediment cores to date past storm surges on the coast", 0),
             ("Microplastics in upland lakes", "catalogue microplastic types in upland lakes",
              "track how river floods carry those particles into estuary saltmarshes", 1),
             ("Fuel use in offshore supply vessels", "cut the fuel use of offshore supply vessels",
              "ask harbour communities how changes to vessel routes would affect them", 2)]},
    {"code": "HA", "title": "Healthy Ageing at Home",
     "themes": ["falls and mobility", "loneliness and social connection", "safe use of medicines"],
     "on": [("Home exercise to prevent falls after 75",
             "test a home exercise programme meant to cut falls among people over 75", 0, ("people",)),
            ("Befriending schemes for people living alone",
             "evaluate volunteer befriending schemes for older people who live alone", 1, ("people",)),
            ("Safer prescribing at hospital discharge",
             "reduce medication errors when older patients move from hospital to home", 2, ("people",))],
     "off": [("Sleep and night-shift nursing", "improve sleep quality among night-shift nurses",
              "check whether poor sleep among carers predicts falls among the older people they visit", 0),
             ("Sports injury in adolescent athletes", "follow sports injuries among adolescent athletes",
              "adapt the balance-training protocol for adults over 70", 0),
             ("Genetic markers of early hearing loss", "map genetic markers of early-onset hearing loss",
              "ask whether hearing loss in later life deepens social isolation", 1)]},
    {"code": "CM", "title": "Circular Materials",
     "themes": ["recyclable plastics", "low-carbon construction materials", "recovering metals from used batteries"],
     "on": [("Solvent sorting of mixed packaging",
             "develop a solvent process that separates mixed plastic packaging for recycling", 0, ("equip",)),
            ("A lower-emission cement blend", "trial a cement blend that halves process emissions", 1, ("equip",)),
            ("Pilot-scale cobalt recovery",
             "recover cobalt and nickel from spent laptop batteries at pilot scale", 2, ("equip",))],
     "off": [("Faster catalysts for drug synthesis", "design faster catalysts for pharmaceutical synthesis",
              "test whether the same catalysts can break down mixed plastic waste", 0),
             ("Glass from a sixteenth-century wreck",
              "characterise glass recovered from a sixteenth-century shipwreck",
              "compare the old glass with recycled glass now used in building materials", 1),
             ("Ventilation control for office comfort",
              "improve thermal comfort in offices through ventilation control",
              "estimate the embodied carbon of the duct materials involved", 1)]},
    {"code": "VA", "title": "Voices from the Archive",
     "themes": ["digitising at-risk collections", "computational analysis of historical texts",
                "public history and participation"],
     "on": [("Rescuing the dockworkers' minute books",
             "digitise and catalogue a decaying run of dockworkers' union minute books", 0, ("archive",)),
            ("Two centuries of parish registers", "apply text mining to two centuries of parish registers", 1,
             ("archive",)),
            ("Residents annotate the photograph archive",
             "run workshops in which residents annotate a digitised photograph archive", 2, ("people", "archive"))],
     "off": [("A life of a forgotten novelist", "write a critical biography of a nineteenth-century novelist",
              "digitise the novelist's surviving letters, now kept in a damp basement", 0),
             ("Attitudes to public libraries today", "survey present-day attitudes to public libraries",
              "invite respondents to add their memories to a local history collection", 2),
             ("Staging a restoration comedy", "stage and document a new production of a restoration comedy",
              "use text analysis to compare the play with other comedies of its decade", 1)]},
    {"code": "WS", "title": "Water Security for Growing Towns",
     "themes": ["groundwater monitoring", "drought forecasting", "safe reuse of treated wastewater"],
     "on": [("Low-cost groundwater sensing",
             "install low-cost sensors to track groundwater levels across a farming district", 0, ("field", "equip")),
            ("Seasonal drought outlooks for small utilities",
             "improve seasonal drought forecasts for small water utilities", 1, ("field",)),
            ("Irrigating vegetables with treated wastewater",
             "test whether treated wastewater can safely irrigate vegetable crops", 2, ("field", "equip"))],
     "off": [("River navigation in the early modern period", "trace the history of river navigation in the region",
              "use historical records of low river levels to test drought forecasts", 1),
             ("Hail damage to greenhouses", "model hail damage to commercial greenhouses",
              "look at whether greenhouse growers could reuse treated wastewater", 2),
             ("Noise near treatment works", "measure noise exposure for homes near water treatment works",
              "ask residents how they feel about reusing treated wastewater", 2)]},
]
# (panel area, departments that work in it) per call code
AREAS = {"RC": ("Environmental Sciences", ["Geography", "Earth Sciences", "Marine Science", "Planning"]),
         "HA": ("Health Research", ["Public Health", "Nursing", "Psychology", "Pharmacy"]),
         "CM": ("Engineering and Materials", ["Chemistry", "Materials Science", "Civil Engineering"]),
         "VA": ("Arts and Humanities", ["History", "Digital Humanities", "Information Studies", "English"]),
         "WS": ("Environmental Sciences", ["Civil Engineering", "Earth Sciences", "Geography", "Agriculture"])}
TOPICS = [(on[0], c["code"], on[3]) for c in CALLS for on in c["on"]]
METHOD_LINES = [
    "The work runs in three packages, each with a named lead.",
    "A postdoctoral researcher will carry out most of the day-to-day work.",
    "A six-month pilot tests the methods before the main phase begins.",
    "Results will go into open-access papers and a short briefing for practitioners.",
    "An advisory group of end users meets twice a year to steer the work.",
]


def _topic(rng, need=None):
    """(title, call code, tags) of a funded project; `need` restricts it to topics carrying that tag."""
    return pick(rng, [t for t in TOPICS if need is None or need in t[2]])


def _funder(rng):
    return f"{rng.choice(ORG_A)} {rng.choice(FUNDER_SUFFIX)}"


def _domain(name):
    return "".join(ch for ch in name.lower().split()[0] if "a" <= ch <= "z") + ".example.org"


def _r(x, step=100):
    return int(round(x / step)) * step


def _add_months(d, n):
    y, m = divmod(d.month - 1 + n, 12)
    return date(d.year + y, m + 1, min(d.day, 28))


def _ym(d):
    return f"{d.year}-{d.month:02d}"


def _date(d, rng):
    """A date in a house style that always shows the year (eligibility arithmetic depends on it)."""
    return fmt_date(d, rng, style=pick(rng, ["iso", "iso", "long", "us", "eu"]))


def _mstyle(rng):
    """One money style per rendered state, so figures in one artefact look alike."""
    return pick(rng, ["sym", "sym", "code"])


def _num(x):
    return f"{x:,}"


def _pl(n, word, plural=None):
    return f"{n} {word if n == 1 else (plural or word + 's')}"


# =========================================================================== schema 1
CAP_REASONS = {
    "equipment": "The quote for the main equipment rose after the call opened; we ask the panel to accept the difference.",
    "salary": "The excess reflects the institution's new pay scale, confirmed after our budget was drafted.",
    "scope": "We have gone slightly over the limit so the final work package can run in full, as the pilot showed it must.",
}
CAP_WEAK = "We believe the scale of the work justifies the higher amount."
EXT_NOTES = [
    "The head of department has confirmed by email that the contract will be renewed.",
    "A contract renewal is under discussion with the department.",
    "The PI expects to be made permanent in the next promotion round.",
]


class ProposalFirstScreen(Schema):
    name = "proposal_first_screen"
    description = ("A research proposal at first-round screening, checked against the call's written rules, its "
                   "themes and its budget before any expert sees it.")
    surfaces = {"json": 0.45, "screening_sheet": 0.3, "intake_email": 0.25}

    HOST_LETTER = {"signed": "signed by the head of department", "unsigned": "unsigned draft",
                   "wrong_signatory": "signed by a senior colleague, not the head of department",
                   "missing": "not received"}

    def _plant(self, f, rng, issue, severe):
        if issue == "over_cap":
            if severe:
                f["over_pct"] = between(rng, 0.18, 0.45, digits=3)
                f["cap_note_kind"] = pick(rng, ["weak", None])
            else:
                f["over_pct"] = near(rng, 0.03, 0.06, digits=3)
                f["cap_note_kind"] = "reason" if chance(rng, 0.7) else None
        elif issue == "overhead":
            f["indirect_pct"] = f["indirect_cap_pct"] + (pick(rng, [12, 15, 20]) if severe else pick(rng, [1, 2, 3, 5]))
        elif issue == "pi_status":
            if f["pi_rule"] == "contract":
                f["contract_margin_months"] = -pick(rng, [12, 18, 24]) if severe else -pick(rng, [1, 2, 3, 4, 6])
                f["extension_note"] = None if severe else maybe(rng, pick(rng, EXT_NOTES), 0.3)
            elif f["pi_rule"] == "early_career":
                limit = f["career_limit_years"] * 12
                if severe:
                    f["phd_months_before_deadline"] = limit + pick(rng, [30, 42, 60])
                    f["career_break_months"] = 0
                else:
                    f["phd_months_before_deadline"] = limit + pick(rng, [2, 4, 7, 10, 14])
                    f["career_break_months"] = pick(rng, [0, 6, 9, 12, 18])
            else:
                f["other_role"] = "lead" if severe else pick(rng, ["co_lead", "co_lead", "co_investigator"])
        elif issue == "fit":
            f["fit"] = "outside" if severe else pick(rng, ["partial", "partial", "tangential"])
        elif issue == "documents":
            if severe:
                f["host_letter"] = "missing"
                if f["dmp_required"]:
                    f["dmp"] = None
            else:
                o = pick(rng, ["unsigned", "wrong_signatory"] + (["dmp", "dmp"] if f["dmp_required"] else []))
                if o == "dmp":
                    f["dmp"] = None
                else:
                    f["host_letter"] = o
        elif issue == "letter_missing":
            f["host_letter"] = "missing"
        elif issue == "justification":
            f["justification"] = "thin" if severe else "partial"
        elif issue == "track_record":
            f["track"] = "thin"
        elif issue == "duration":
            f["duration"] = f["max_months"] + (pick(rng, [12, 18]) if severe else pick(rng, [2, 3, 6]))
            f["duration_note"] = None if severe else "The final months cover write-up and dissemination only."

    def sample_factors(self, rng):
        reg = regime(rng)
        call = pick(rng, CALLS)
        cur = pick_w(rng, CURRENCIES)
        max_months = pick(rng, [24, 36, 36, 48])
        oh_cap = pick(rng, [10, 15, 15, 20, 25])
        pi_rule = pick(rng, ["contract", "early_career", "one_lead"])
        limit = pick(rng, [6, 8, 10])
        dmp_required = chance(rng, 0.6)
        f = {
            "regime": reg, "call_code": call["code"], "currency": cur, "funder": _funder(rng),
            "cap": pick(rng, [150000, 200000, 250000, 300000, 400000, 500000]), "max_months": max_months,
            "indirect_cap_pct": oh_cap, "pi_rule": pi_rule, "career_limit_years": limit, "dmp_required": dmp_required,
            # a clean application first; the regime then plants problems in it
            "fit": "central", "over_pct": between(rng, -0.3, -0.03, digits=3),
            "indirect_pct": pick(rng, [oh_cap, oh_cap, max(5, oh_cap - 5)]),
            "duration": pick(rng, [m for m in (18, 24, 30, 36, 42, 48) if max_months - 12 <= m <= max_months]),
            "duration_note": None,
            "contract_margin_months": None if chance(rng, 0.45) else pick(rng, [6, 12, 24]),
            "extension_note": None,
            "phd_months_before_deadline": rng.randint(24, limit * 12 - 4) if pi_rule == "early_career" else None,
            "career_break_months": 0, "other_role": None,
            "host_letter": "signed", "dmp": "attached" if dmp_required else pick(rng, [None, "attached"]),
            "justification": "itemised", "track": pick_w(rng, {"strong": 2, "moderate": 3}),
            "cap_note_kind": None, "pattern": None,
        }
        issues = []
        if reg == "clear":
            if chance(rng, 0.5):
                issues.append((pick(rng, ["over_cap", "overhead", "pi_status", "fit", "documents", "duration"]), True))
                if chance(rng, 0.4):
                    issues.append((pick(rng, ["justification", "track_record"]), False))
        elif reg == "borderline":
            pool = {"over_cap": 3, "overhead": 2, "pi_status": 3, "fit": 3, "documents": 2, "justification": 2,
                    "duration": 1, "track_record": 1}
            first = pick_w(rng, pool)
            issues.append((first, False))
            if chance(rng, 0.45):
                issues.append((pick_w(rng, {k: v for k, v in pool.items() if k != first}), False))
        else:
            pattern = pick(rng, ["thin_team", "off_theme_star", "over_cap_reasoned", "thin_budget", "edge_eligibility"])
            f["pattern"] = pattern
            if pattern == "thin_team":          # strong idea, junior team, host letter missing
                issues += [("track_record", False), ("letter_missing", True)]
            elif pattern == "off_theme_star":   # excellent group, paperwork perfect, drifting from the call
                f["track"] = "strong"
                issues.append(("fit", False))
            elif pattern == "over_cap_reasoned":
                f["track"] = "strong"
                f["over_pct"] = between(rng, 0.02, 0.09, digits=3)
                f["cap_note_kind"] = "reason"
            elif pattern == "thin_budget":
                f["track"] = "strong"
                issues.append(("justification", True))
            else:
                f["track"] = "strong"
                issues.append(("pi_status", False))
        for issue, severe in issues:
            self._plant(f, rng, issue, severe)
        f["issues"] = [f"{i}:{'severe' if s else 'mild'}" for i, s in issues]
        if f["phd_months_before_deadline"] is None:
            f["phd_months_before_deadline"] = {"strong": rng.randint(150, 300), "moderate": rng.randint(50, 160),
                                               "thin": rng.randint(26, 90)}[f["track"]]

        # ---- the proposal itself: title and summary follow the planted fit
        fit = f["fit"]
        if fit == "central":
            title, phrase, th, _ = pick(rng, call["on"])
            summary = f"This project will {phrase}."
        elif fit == "partial":
            title, phrase, side, th = pick(rng, call["off"])
            summary = f"The main aim is to {phrase}. A smaller work package will also {side}."
        else:
            title, phrase, _, _ = pick(rng, call["off"])
            th = rng.randrange(3)
            summary = f"This project will {phrase}." + (
                f" The findings could later inform work on {call['themes'][th]}." if fit == "tangential" else "")
        f.update(title=title, summary=summary + " " + pick(rng, METHOD_LINES), theme_selected=call["themes"][th])

        # ---- budget: built from the target total, then re-added so every figure is consistent
        tgt_total = f["cap"] * (1 + f["over_pct"])
        direct_t = tgt_total / (1 + f["indirect_pct"] / 100)
        eq = 0.0 if chance(rng, 0.35) else rng.uniform(0.05, 0.2)
        lines = {"equipment": _r(direct_t * eq, 500), "consumables": _r(direct_t * rng.uniform(0.06, 0.14)),
                 "travel": _r(direct_t * rng.uniform(0.03, 0.07)), "other": _r(direct_t * rng.uniform(0.01, 0.04))}
        lines = {"staff": _r(direct_t) - sum(lines.values()), **lines}
        direct = sum(lines.values())
        indirect = int(round(direct * f["indirect_pct"] / 100))
        total = direct + indirect
        if total <= f["cap"]:
            f["cap_note_kind"] = None
        cap_note = None
        if f["cap_note_kind"] == "weak":
            cap_note = CAP_WEAK
        elif f["cap_note_kind"] == "reason":
            cap_note = CAP_REASONS[pick(rng, (["equipment"] if lines["equipment"] else []) + ["salary", "scope"])]
        staff_months = max(12, f["duration"] - pick(rng, [0, 3, 6]))
        fte = lines["staff"] / (staff_months / 12 * SALARY[cur] * rng.uniform(0.9, 1.12))
        if fte <= 1.05:
            staff_txt = (f"Postdoctoral researcher at {max(0.2, round(fte, 1)):.1f} FTE for {staff_months} months on "
                         "the institutional pay scale.")
        else:
            staff_txt = (f"Postdoctoral researcher at 1.0 FTE for {staff_months} months, plus research assistant time "
                         f"totalling {round(fte - 1.0, 1):.1f} FTE, on the institutional pay scale.")
        just = f["justification"]
        if just == "thin":
            jtext = pick(rng, ["All costs follow standard institutional rates.",
                               "Budget prepared by the research office; a breakdown is available on request."])
        else:
            parts = [staff_txt]
            if lines["equipment"]:
                parts.append("Equipment as per the attached supplier quote." if just == "itemised"
                             else "Equipment figure is an estimate; a quote will follow.")
            if just == "itemised":
                parts.append(f"Consumables and running costs itemised by work package; travel covers "
                             f"{rng.randint(3, 8)} site visits and one conference.")
            else:
                parts.append("Consumables and travel based on previous projects.")
            jtext = " ".join(parts)
        deadline = rand_date(rng)
        start = _add_months(deadline, pick(rng, [4, 6, 9])).replace(day=1)
        end = _add_months(start, f["duration"]) - timedelta(days=1)
        track = f["track"]
        f.update(
            lines=lines, direct=direct, indirect=indirect, total=total, cap_note=cap_note, justification_text=jtext,
            deadline=deadline, start=start, end=end,
            call_ref=f"{call['code']}-{deadline.year}-{rng.randint(1, 4)}", app_ref=ident(rng, call["code"], 5, "-"),
            other_ref=ident(rng, call["code"], 5, "-"),
            pi=person(rng), institution=org(rng, "school"), co_investigators=pick(rng, [0, 1, 2, 2, 3, 4]),
            outputs_5y={"strong": rng.randint(18, 40), "moderate": rng.randint(6, 15), "thin": rng.randint(0, 3)}[track],
            awards_as_pi={"strong": rng.randint(3, 7), "moderate": rng.randint(1, 2), "thin": 0}[track],
            intake_officer=person(rng), screener=person(rng),
        )
        if f["pi_rule"] == "early_career":
            f["awards_as_pi"] = min(f["awards_as_pi"], 2)
        return f

    # ------------------------------------------------------------------ rendering
    def _rule_text(self, f):
        return {"contract": "The PI's employment contract with the host institution must cover the whole project period.",
                "early_career": f"The PI's PhD must have been awarded no more than {f['career_limit_years']} years "
                                "before the call deadline; documented career breaks extend this limit.",
                "one_lead": "Nobody may lead more than one application to this call."}[f["pi_rule"]]

    def _docs_required(self, f):
        return ["letter of support from the host institution, signed by the head of department"] + (
            ["data management plan"] if f["dmp_required"] else [])

    def _pi_display(self, f):
        return ("Prof. " if f["track"] == "strong" and f["pi_rule"] != "early_career" else "Dr ") + f["pi"]

    def _contract(self, f, rng=None):
        if f["contract_margin_months"] is None:
            return "open-ended"
        d = _add_months(f["end"], f["contract_margin_months"])
        return "fixed-term, ends " + (_date(d, rng) if rng else d.isoformat())

    def _phd(self, f):
        return _add_months(f["deadline"], -f["phd_months_before_deadline"])

    def _other_app(self, f):
        if f["other_role"] is None:
            return None
        return {"lead": "lead applicant", "co_lead": "co-lead", "co_investigator": "co-investigator"}[
            f["other_role"]] + f" on {f['other_ref']}"

    def render(self, f, rng, surface):
        call = next(c for c in CALLS if c["code"] == f["call_code"])
        cur, L = f["currency"], f["lines"]
        rule = f["pi_rule"]
        summary = noisy(f["summary"], rng, 0.1)
        dmp_txt = f["dmp"] if f["dmp"] else ("not received" if f["dmp_required"] else None)
        if surface == "json":
            show_contract = rule == "contract" or chance(rng, 0.35)
            show_phd = rule == "early_career" or chance(rng, 0.35)
            return dumps(prune({
                "call": {"ref": f["call_ref"], "funder": f["funder"], "title": call["title"], "themes": call["themes"],
                         "deadline": f["deadline"].isoformat(),
                         "rules": {"max_request": f["cap"], "currency": cur, "max_duration_months": f["max_months"],
                                   "indirect_cap_pct_of_direct": f["indirect_cap_pct"],
                                   "applicant_rule": self._rule_text(f),
                                   "required_documents": self._docs_required(f)}},
                "application": {"ref": f["app_ref"], "title": f["title"], "theme_selected": f["theme_selected"],
                                "summary": summary, "start": f["start"].isoformat(),
                                "duration_months": f["duration"], "duration_note": f["duration_note"]},
                "applicant": {"pi": self._pi_display(f), "institution": f["institution"],
                              "contract": self._contract(f) if show_contract else None,
                              "contract_comment": f["extension_note"] if show_contract else None,
                              "phd_awarded": _ym(self._phd(f)) if show_phd else None,
                              "career_break_months": f["career_break_months"] or None,
                              "other_applications_to_call": (self._other_app(f) or "none") if rule == "one_lead" else None,
                              "outputs_last_5_years": f["outputs_5y"], "awards_held_as_pi": f["awards_as_pi"],
                              "co_investigators": f["co_investigators"]},
                "budget": {**{k: (v or None) for k, v in L.items()}, "direct_total": f["direct"],
                           "indirect": f["indirect"], "total_requested": f["total"],
                           "justification": f["justification_text"], "applicant_note": f["cap_note"]},
                "documents": {"host_letter": self.HOST_LETTER[f["host_letter"]], "data_management_plan": dmp_txt},
            }))
        st = _mstyle(rng)
        rules = [f"Maximum request {money(f['cap'], None, cur, st)} in total, including indirect costs",
                 f"Maximum duration {f['max_months']} months",
                 f"Indirect costs at most {f['indirect_cap_pct']}% of direct costs",
                 self._rule_text(f).rstrip("."),
                 "Required: " + "; ".join(self._docs_required(f))]
        pi_row = {"contract": ("Contract", self._contract(f, rng)
                               + (f" ({f['extension_note'].rstrip('.')})" if f["extension_note"] else "")),
                  "early_career": ("PhD awarded", self._phd(f).strftime("%B %Y") + (
                      f"; documented career break of {f['career_break_months']} months" if f["career_break_months"]
                      else "")),
                  "one_lead": ("Other applications to this call", self._other_app(f) or "none")}[rule]
        rate = round(100 * f["indirect"] / f["direct"], 1)
        if surface == "screening_sheet":
            rows = [[k.capitalize(), _num(v)] for k, v in L.items() if v]
            rows += [["Direct total", _num(f["direct"])], [f"Indirect ({rate}%)", _num(f["indirect"])],
                     ["TOTAL REQUESTED", _num(f["total"])]]
            head = kv_lines([
                ("Application", f["app_ref"]), ("Title", f["title"]), ("PI", f"{self._pi_display(f)}, {f['institution']}"),
                ("Theme selected", f["theme_selected"]),
                ("Duration", f"{f['duration']} months from {_date(f['start'], rng)}"
                 + (f" ({f['duration_note'].rstrip('.').lower()})" if f["duration_note"] else "")),
                ("Track record", f"{f['outputs_5y']} outputs in last 5 years; {_pl(f['awards_as_pi'], 'award')} held "
                                 f"as PI; {_pl(f['co_investigators'], 'co-investigator')}"),
                pi_row,
                ("Host letter", self.HOST_LETTER[f["host_letter"]]),
                ("Data management plan", dmp_txt),
            ], rng)
            parts = [f"SCREENING WORKSHEET | {f['funder']} | {call['title']} ({f['call_ref']})",
                     f"Call deadline: {_date(f['deadline'], rng)}. Themes: {'; '.join(call['themes'])}.",
                     "", "Call rules:", bullets(rules, rng), "", head, "", f"Summary: {summary}", "",
                     f"Budget ({cur})", table(["Line", "Amount"], rows, rng), "",
                     f"Justification: {f['justification_text']}"]
            if f["cap_note"]:
                parts.append(f"Applicant's note on the budget: {f['cap_note']}")
            return "\n".join(parts)
        # intake_email
        dom = _domain(f["funder"])
        docs = [f"host letter: {self.HOST_LETTER[f['host_letter']]}"] + (
            [f"data management plan: {dmp_txt}"] if dmp_txt else [])
        pi_sentence = {"contract": f"Contract status: {pi_row[1]}.",
                       "early_career": f"PhD awarded {pi_row[1]}.",
                       "one_lead": f"Other applications to this call: {pi_row[1]}."}[rule]
        body = [
            f"From: {email_addr(f['intake_officer'], dom)}", f"To: screening@{dom}",
            f"Subject: For screening: {f['app_ref']} ({f['call_ref']}) - {f['title']}", "",
            f"Hi {f['screener'].split()[0]},", "",
            f"Next one in the queue for the {call['title']} call (deadline {_date(f['deadline'], rng)}). Rules as a "
            "reminder:", bullets(rules, rng), "",
            noisy(join_sentences([
                f"{self._pi_display(f)} at {f['institution']} asks for {money(f['total'], None, cur, st)} over "
                f"{f['duration']} months ({money(f['direct'], None, cur, st)} direct plus "
                f"{money(f['indirect'], None, cur, st)} indirect, {rate}% of direct).",
                f"Theme ticked on the form: {f['theme_selected']}.",
                f"{f['outputs_5y']} outputs in the last five years and {_pl(f['awards_as_pi'], 'award')} held as PI.",
                pi_sentence]), rng, 0.1),
            "", f"Their summary: \"{summary}\"",
            "", "Budget lines: " + ", ".join(f"{k} {_num(v)}" for k, v in L.items() if v) + f" ({cur}).",
            f"Justification given: {f['justification_text']}",
        ]
        if f["duration_note"]:
            body.append(f"On the duration they say: {f['duration_note']}")
        if f["cap_note"]:
            body.append(f"Cover note on the budget: \"{f['cap_note']}\"")
        body += ["", "Documents in: " + "; ".join(docs) + ".", "", "Thanks,", f["intake_officer"].split()[0]]
        return "\n".join(body)

    def questions(self):
        return {
            "screen_outcome": choice("What should the screening officer do with this application?", {
                "advance_to_panel": "Pass it to the expert panel as submitted.",
                "advance_with_queries": "Pass it to the panel and ask the applicant to clarify named points before "
                                        "the meeting.",
                "return_for_correction": "Hold it back and give the applicant a short deadline to fix an "
                                         "administrative problem.",
                "reject_ineligible": "Reject it at screening because it breaks one of the call's eligibility rules.",
                "reject_out_of_scope": "Reject it at screening because the research is not what this call funds.",
            }),
            "primary_concern": choice("Which issue most needs a screener's attention in this application?", {
                "none_material": "Nothing stands out; the application is in order for expert review.",
                "eligibility_or_documents": "Whether the applicant and the paperwork meet the call's formal "
                                            "requirements.",
                "fit_to_call": "Whether the research addresses the themes this call was set up to fund.",
                "budget": "Whether the amount asked for, or how it is split and explained, is acceptable.",
                "applicant_capacity": "Whether the applicant's experience and record are enough to deliver the work.",
            }),
            "call_fit": score("How closely does the proposed research match the call's stated themes?", [
                "Outside: it addresses none of the call's themes.",
                "Tangential: a theme is mentioned only as a possible later use of the results.",
                "Partial: part of the work serves a call theme, but the main aim lies elsewhere.",
                "Central: the main aims address a call theme directly.",
            ]),
            "budget_justification": score("How well does the application justify the money it requests?", [
                "Unjustified: major costs are unexplained or look inflated for the work described.",
                "Weak: several costs lack a basis such as a quote, a rate or a time allocation.",
                "Adequate: most costs are explained, with a gap or two a panel may query.",
                "Well justified: every major cost is itemised and tied to the work plan.",
            ]),
            "eligibility_met": noul(
                "The application satisfies all of the eligibility rules set out in the call.",
                yes="Amount, duration, cost rates, applicant status and required documents all fit the written rules.",
                no="At least one written rule is broken, or compliance cannot be confirmed from what was submitted."),
        }


# =========================================================================== schema 2
FLAWS = [
    {"review": "The sample of {n} sites cannot support the comparisons proposed, and there is no power calculation.",
     "addresses": "A power calculation, now in annex D, shows that {n} sites give adequate power for the primary comparison.",
     "partial": "We accept the point: the primary comparison stays, and the secondary comparisons become exploratory.",
     "sidesteps": "Our earlier studies used similar numbers and were published in leading journals."},
    {"review": "Everything depends on access to a partner's data, yet no data-sharing agreement is in place.",
     "addresses": "The data-sharing agreement was signed last month; a copy is attached.",
     "partial": "The agreement is with the partner's legal team and we expect it before the start date.",
     "sidesteps": "We have strong informal links with the partner."},
    {"review": "Work package 2 needs a custom tool the team has not yet built or tested.",
     "addresses": "The tool has since been built and tested; the results are in the response annex.",
     "partial": "The build is under way; if it slips, work package 2 can use a commercial alternative at extra cost.",
     "sidesteps": "Our engineers are confident the tool will be ready in time."},
    {"review": "Aim 1 rests on an effect that a large 2024 study failed to find, and the proposal does not discuss it.",
     "addresses": "That study used a different population; our pilot reproduced the effect in our setting (n = 180).",
     "partial": "We will add a pre-registered test of the effect in year 1 before committing to aim 2.",
     "sidesteps": "We are aware of the wider literature on this question."},
]
MISREADS = [
    {"review": "There is no plan for sharing the data.",
     "response": "The data management plan is in annex C and covers deposit in a public repository."},
    {"review": "The proposal never says how ethics approval will be obtained.",
     "response": "Section 6 sets out the ethics route; the pilot phase already has approval."},
    {"review": "Nobody on the team has statistical training.",
     "response": "Our co-investigator {coi} is a statistician and leads work package 3."},
]
POS_SPECIFIC = [
    "The pilot data make the targets believable, and the analysis plan is detailed.",
    "The team has run this method before; the work packages are well sequenced with clear decision points.",
    "Access to the key sites and records is secured through named partners, which removes the usual risk here.",
    "A clearly argued case with realistic milestones and a budget that matches the work.",
]
POS_GENERIC = [
    "An exciting and timely proposal from a strong group.",
    "I enjoyed reading this. Important topic; I would like to see it funded.",
    "Excellent team, excellent idea. Strongly supported.",
    "A very good proposal that deserves support.",
]
MID = [
    "Sound design overall, though the impact section is thin.",
    "Good science, but the timetable for the second work package looks tight.",
    "Worth doing. Competent and clearly written, although the novelty is moderate.",
    "A reasonable plan; I would like more detail on how the partners will be involved.",
]
LOW_GENERAL = [
    "The aims are too broad for the time requested, and the methods are underspecified.",
    "It is hard to see what will be delivered by the end; the milestones are vague.",
    "The case for why this matters now is not made, and the design has gaps.",
    "The literature review is dated and the research questions are not clearly framed.",
    "I could not find a convincing account of how the data will be analysed.",
]
FEAS_DOUBT = [
    "I am not convinced the team can deliver the main study in the time available.",
    "Ambitious. I doubt the targets are realistic without more groundwork.",
]
HARSH = [
    "A careful, well-argued proposal. It is incremental rather than transformative, and I keep high marks for the latter.",
    "Nothing wrong with the plan as such; I simply see it as routine work for a group of this standing.",
]
LOW_EXP = ["This is outside my own field, so I have concentrated on the overall design. ",
           "I am not a specialist in these methods. "]
EXPERT_ENDORSE = [
    "I use this approach myself. The feasibility worries are overstated: the pilot results are exactly what the field "
    "has been missing.",
    "Having run similar studies, I think the design is stronger than it first looks; the sample is adequate for the "
    "primary question.",
]


def _draw(rng, bank, used):
    """A text from `bank` not yet used in this case (so two reviewers never write the same sentence)."""
    x = pick(rng, [t for t in bank if t not in used] or bank)
    used.append(x)
    return x


class PanelScoreReconciliation(Schema):
    name = "panel_score_reconciliation"
    description = ("A proposal whose external reviews disagree sharply, brought to the funding panel for one "
                   "reconciled recommendation.")
    surfaces = {"json": 0.45, "panel_sheet": 0.3, "review_excerpts": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        scale = pick_w(rng, {6: 3, 5: 1, 7: 1})
        n_rev = pick(rng, [3, 3, 4])
        line_frac = rng.uniform(0.7, 0.8)
        funding_line = round(scale * line_frac, 1)
        feas = pick_w(rng, {"sound": 4, "stretched": 3, "doubtful": 2})
        resp_kind = pick_w(rng, {"addresses": 2, "partial": 3, "sidesteps": 2})
        if reg == "clear":
            good = chance(rng, 0.5)
            centre = scale * (rng.uniform(0.84, 0.95) if good else rng.uniform(0.3, 0.5))
            gap, basis = pick(rng, [0, 1]), "none"
            feas = "sound" if good else pick(rng, ["stretched", "doubtful"])
        elif reg == "borderline":
            centre = funding_line + rng.uniform(-0.4, 0.6)
            gap = 2
            basis = pick_w(rng, {"concrete_flaw": 3, "misread": 2, "harsh_marker": 2, "low_expertise": 2,
                                 "generic_praise": 2})
        else:
            centre = funding_line + rng.uniform(0.0, 0.9)
            gap = pick(rng, [3, 3, 4])
            basis = pick_w(rng, {"concrete_flaw": 4, "misread": 2, "expert_endorse": 2, "generic_praise": 1})
        if basis in ("generic_praise", "expert_endorse"):
            direction = "high"
        elif basis in ("concrete_flaw", "misread", "harsh_marker"):
            direction = "low"
        else:
            direction = pick(rng, ["low", "high"])
        if basis == "expert_endorse":
            feas = pick(rng, ["stretched", "doubtful"])

        def clamp(x):
            return max(1, min(scale, int(round(x))))

        level = centre - gap * 0.4 if direction == "high" else centre + gap * 0.3
        others = [clamp(level + rng.uniform(-0.6, 0.6)) for _ in range(n_rev - 1)]
        mean_o = sum(others) / len(others)
        out_score = clamp(mean_o + gap if direction == "high" else mean_o - gap)
        out_pos = rng.randrange(n_rev)
        overall = others[:out_pos] + [out_score] + others[out_pos:]

        # feasibility sub-scores follow each reviewer's overall view and the proposal's real feasibility
        feas_base = {"sound": 0.85, "stretched": 0.64, "doubtful": 0.45}[feas] * scale
        feasibility = [clamp(0.5 * overall[i] + 0.5 * feas_base + rng.uniform(-0.5, 0.5)) for i in range(n_rev)]
        if basis == "concrete_flaw":
            feasibility[out_pos] = clamp(min(out_score, feas_base - 1.5))
        elif basis == "expert_endorse":
            feasibility[out_pos] = clamp(scale - rng.randint(0, 1))

        flaw_i, mis_i = rng.randrange(len(FLAWS)), rng.randrange(len(MISREADS))
        n_sites = pick(rng, [6, 8, 10, 12])
        coi = person(rng)
        trim = pick(rng, [None, None, None, 10, 15, 20])
        expertise, comments, used = [], [], []
        for i in range(n_rev):
            s = overall[i]
            if i == out_pos and basis != "none":
                exp = {"concrete_flaw": pick_w(rng, {"high": 3, "medium": 1}), "misread": pick(rng, ["medium", "high"]),
                       "harsh_marker": pick(rng, ["high", "high", "medium"]), "low_expertise": "low",
                       "generic_praise": pick(rng, ["low", "medium"]), "expert_endorse": "high"}[basis]
                txt = {"concrete_flaw": lambda: FLAWS[flaw_i]["review"].format(n=n_sites),
                       "misread": lambda: MISREADS[mis_i]["review"] + " " + _draw(rng, LOW_GENERAL, used),
                       "harsh_marker": lambda: pick(rng, HARSH),
                       "low_expertise": lambda: pick(rng, LOW_EXP) + _draw(
                           rng, LOW_GENERAL if direction == "low" else POS_GENERIC, used),
                       "generic_praise": lambda: _draw(rng, POS_GENERIC, used),
                       "expert_endorse": lambda: pick(rng, EXPERT_ENDORSE)}[basis]()
            else:
                exp = pick(rng, ["medium", "low", "medium"]) if basis == "expert_endorse" else \
                    pick_w(rng, {"high": 2, "medium": 2, "low": 1})
                frac = s / scale
                if basis == "expert_endorse":
                    txt = _draw(rng, FEAS_DOUBT + MID, used)
                elif frac >= 0.75:
                    generic = reg == "conflicting" and basis == "concrete_flaw" and chance(rng, 0.6)
                    txt = _draw(rng, POS_GENERIC if generic else POS_SPECIFIC, used)
                elif frac >= 0.55:
                    txt = _draw(rng, MID, used)
                else:
                    txt = _draw(rng, LOW_GENERAL, used)
            expertise.append(exp)
            comments.append(txt)
        if trim:
            j = next(i for i in range(n_rev) if i != out_pos)
            comments[j] += f" The equipment line looks high; a cut of about {trim}% seems possible."

        if basis == "concrete_flaw":
            response = FLAWS[flaw_i][resp_kind].format(n=n_sites)
        elif basis == "misread":
            response = MISREADS[mis_i]["response"].format(coi=coi) if chance(rng, 0.75) else None
        elif basis == "expert_endorse":
            response = "The pilot study (annex B) reached 92% of its recruitment target in four months."
        else:
            response = pick(rng, [None, "We thank the reviewers and have no factual corrections to offer."])
        if basis != "concrete_flaw":
            resp_kind = None
        title, code, _ = _topic(rng)
        apps = rng.randint(25, 70)
        return {
            "regime": reg, "scale": scale, "funding_line": funding_line, "n_reviewers": n_rev, "feasibility_true": feas,
            "basis": basis, "direction": direction, "gap_target": gap, "outlier_index": out_pos,
            "overall": overall, "feasibility": feasibility, "expertise": expertise, "comments": comments,
            "response": response, "response_kind": resp_kind, "budget_trim_pct": trim,
            "resubmission": chance(rng, 0.3), "introducer_lean": pick(rng, [None, None, "positive", "critical"]),
            "applications": apps, "awards_expected": max(3, round(apps * rng.uniform(0.12, 0.25))),
            "title": title, "call_code": code, "pi": person(rng), "institution": org(rng, "school"),
            "funder": _funder(rng), "currency": pick_w(rng, CURRENCIES),
            "requested": _r(rng.uniform(120, 600) * 1000, 1000), "months": pick(rng, [24, 36, 36, 48]),
            "ref": ident(rng, "PRJ", 6, "-"), "introducer": person(rng), "panel": AREAS[code][0] + " Panel",
        }

    def _spread(self, f):
        """Secretariat note: how far the furthest review sits from the mean of the others."""
        sc = f["overall"]
        best_i = max(range(len(sc)), key=lambda i: abs(sc[i] - sum(sc[:i] + sc[i + 1:]) / (len(sc) - 1)))
        others = sc[:best_i] + sc[best_i + 1:]
        gap = round(abs(sc[best_i] - sum(others) / len(others)), 1)
        mean = round(sum(sc) / len(sc), 2)
        high = sum(1 for e in f["expertise"] if e == "high")
        return (f"mean {mean}; R{best_i + 1} sits {gap} points from the mean of the other reviews; "
                f"{high} of {len(sc)} reviewers rate their own expertise high")

    def _intro_note(self, f):
        if f["introducer_lean"] is None:
            return None
        return {"positive": "Leaning towards the more favourable reviews; happy to be persuaded otherwise.",
                "critical": "Leaning towards the more critical reviews until the panel has discussed them."}[
            f["introducer_lean"]]

    def render(self, f, rng, surface):
        sc, cur = f["scale"], f["currency"]
        sub = "resubmission (previous outcome: invited to resubmit)" if f["resubmission"] else "first submission"
        comments = [noisy(c, rng, 0.08) for c in f["comments"]]
        if surface == "json":
            return dumps(prune({
                "application": {"ref": f["ref"], "title": f["title"], "pi": f["pi"], "institution": f["institution"],
                                "requested": f["requested"], "currency": cur, "duration_months": f["months"],
                                "submission": sub},
                "reviews": [{"reviewer": f"R{i + 1}", "expertise": f["expertise"][i], "overall": f["overall"][i],
                             "feasibility": f["feasibility"][i], "comment": comments[i]}
                            for i in range(f["n_reviewers"])],
                "round": {"panel": f["panel"], "score_scale": f"1-{sc}, {sc} = outstanding",
                          "lowest_mean_funded_last_round": f["funding_line"], "applications": f["applications"],
                          "awards_expected": f["awards_expected"]},
                "applicant_response": f["response"],
                "introducer_note": self._intro_note(f),
                "secretariat_note": self._spread(f),
            }))
        st = _mstyle(rng)
        req = money(f["requested"], None, cur, st)
        mean = round(sum(f["overall"]) / len(f["overall"]), 2)
        if surface == "panel_sheet":
            rows = [[f"R{i + 1}", f["expertise"][i], f["overall"][i], f["feasibility"][i]] for i in range(f["n_reviewers"])]
            parts = [f"{f['panel'].upper()} | {f['funder']} | score sheet",
                     kv_lines([("Application", f"{f['ref']}: {f['title']}"),
                               ("PI", f"{f['pi']}, {f['institution']}"),
                               ("Request", f"{req} over {f['months']} months"),
                               ("Submission", sub),
                               ("Scale", f"1-{sc} ({sc} = outstanding)"),
                               ("Last round", f"lowest funded mean {f['funding_line']}; "
                                              f"{f['awards_expected']} awards expected from {f['applications']} applications")],
                              rng),
                     "", table(["Reviewer", "Expertise", "Overall", "Feasibility"], rows, rng),
                     f"Mean overall: {mean} (range {min(f['overall'])}-{max(f['overall'])})",
                     f"Secretariat note: {self._spread(f)}", "",
                     "Comment extracts:"]
            parts += [f"R{i + 1}: {c}" for i, c in enumerate(comments)]
            if f["response"]:
                parts += ["", f"Applicant response: {f['response']}"]
            if f["introducer_lean"]:
                parts += [f"Introducer ({f['introducer']}): {self._intro_note(f)}"]
            return "\n".join(parts)
        # review_excerpts
        parts = [f"Reviews received for {f['ref']}, \"{f['title']}\" (PI {f['pi']}, {f['institution']}; {sub})", ""]
        for i, c in enumerate(comments):
            parts += [f"Reviewer {i + 1} | expertise: {f['expertise'][i]} | overall {f['overall'][i]}/{sc}, "
                      f"feasibility {f['feasibility'][i]}/{sc}", f"> {c}", ""]
        if f["response"]:
            parts += ["Applicant's response to reviewers:", f"> {noisy(f['response'], rng, 0.1)}", ""]
        parts.append(f"For reference: last round nothing below a mean of {f['funding_line']} was funded; "
                     f"about {f['awards_expected']} awards are expected from {f['applications']} applications. "
                     f"Request: {req} over {f['months']} months. Secretariat note: {self._spread(f)}.")
        if f["introducer_lean"]:
            parts.append(f"Introducer's pre-meeting note: {self._intro_note(f)}")
        return "\n".join(parts)

    def questions(self):
        return {
            "panel_recommendation": choice("What should the panel recommend for this proposal?", {
                "fund": "Recommend an award at the amount requested.",
                "fund_with_revisions": "Recommend an award once the applicant makes specified changes, such as a "
                                       "smaller budget or a narrower work package.",
                "reserve_list": "Place it just below the funding line, to be funded only if money becomes available.",
                "invite_resubmission": "Decline it this round but encourage a revised application to a later round.",
                "decline": "Decline it without encouraging this project to come back.",
            }),
            "merit_consensus": score("Weighing all the reviews together, how strong is the proposal's research case?", [
                "Poor: basic weaknesses that more work would not fix.",
                "Fair: a reasonable idea with serious gaps in design or rationale.",
                "Good: solid, with weaknesses a panel would expect to see addressed.",
                "Very good: strong throughout, with only minor reservations.",
                "Outstanding: among the best in the round on every criterion.",
            ]),
            "feasibility_rating": score("How achievable is the proposed work with the time, team and money requested?", [
                "Unrealistic: key parts depend on things that are not in place.",
                "Doubtful: important risks are recognised but not dealt with.",
                "Achievable: some risk, with sensible fallbacks.",
                "Very achievable: the groundwork is done and the plan is well within the team's reach.",
            ]),
            "outlier_review_discount": noul(
                "The review furthest from the others should count for less in the panel's judgement.",
                yes="That review rests on a misreading, thin expertise, praise or criticism without specifics, or a "
                    "personal marking habit rather than on the proposal itself.",
                no="That review raises a specific, well-founded point that the applicant's response does not fully "
                   "answer, or it sits no more than one point from the mean of the other reviews."),
            "extra_review_needed": noul(
                "The panel needs one more expert review before it can reach a sound decision.",
                yes="The mean is within half a point of last round's funding line and at most one reviewer rates their "
                    "own expertise high, so the decision rests on thin expert input.",
                no="The mean is clearly above or below the funding line, or two or more high-expertise reviews are "
                   "already in."),
        }


# =========================================================================== schema 3
# (milestone, tag it needs or None), in the order a project would meet them
MILESTONES = [
    ("Ethics approval in place", "people"), ("Postdoctoral researcher appointed", None),
    ("Partner agreements signed", None), ("Advisory board convened", None), ("Pilot study complete", None),
    ("Main instrument commissioned", "equip"), ("First field season complete", "field"),
    ("Survey instrument validated", "people"), ("Collection catalogue complete", "archive"),
    ("Stakeholder workshop held", None), ("Interim dataset deposited", None), ("Interim analysis complete", None),
    ("First paper submitted", None),
]
MILESTONE_NAMES = [m for m, _ in MILESTONES]
# cause: (kind, explanation, milestone it delays, supporting document or None, topic tag it needs or None)
SLIP_CAUSES = {
    "ethics": ("external", "The ethics committee took five months to approve a protocol amendment.",
               "Ethics approval in place", "the committee's correspondence", "people"),
    "supplier": ("external", "The supplier delivered the main instrument seven months late after a fire at its factory.",
                 "Main instrument commissioned", "the supplier's delivery notice", "equip"),
    "partner": ("external", "A partner organisation withdrew after a change of management, and a replacement had to "
                            "be found.", "Partner agreements signed", "the replacement partner's signed agreement", None),
    "weather": ("external", "Flooding closed the field sites for most of the planned season.",
                "First field season complete", "the landowner's closure notices", "field"),
    "archive_closure": ("external", "The record office holding the collection closed for five months for roof repairs.",
                        "Collection catalogue complete", "the record office's closure notice", "archive"),
    "recruitment": ("mixed", "The postdoctoral post had to be advertised twice and was filled only in month nine.",
                    "Postdoctoral researcher appointed", None, None),
    "teaching": ("internal", "The PI's teaching load was heavier than planned, and project work was deprioritised.",
                 "Pilot study complete", None, None),
    "pipeline": ("internal", "Building the analysis pipeline took much longer than we estimated.",
                 "Interim analysis complete", None, None),
    "unexplained": ("unexplained", "Some activities have moved into next year.", "Interim dataset deposited", None, None),
}
EXTERNAL_CAUSES = ["ethics", "supplier", "partner", "weather", "archive_closure"]
STATUS_TEXT = {"met": "met", "met_late": "met late", "partly_met": "partly met", "missed": "missed",
               "deferred_by_agreement": "deferred (agreed with funder)"}


class AwardProgressContinuation(Schema):
    name = "award_progress_continuation"
    description = ("An annual progress report on a multi-year award, read to decide whether next year's tranche "
                   "of funding is released.")
    surfaces = {"json": 0.4, "progress_report": 0.25, "officer_memo": 0.2, "portal_export": 0.15}

    def sample_factors(self, rng):
        reg = regime(rng)
        delivery = pick_w(rng, {"on_track": 2, "slipping": 3, "behind": 3, "failing": 1})
        cause = pick_w(rng, {"ethics": 2, "supplier": 2, "partner": 1, "weather": 1, "archive_closure": 1,
                             "recruitment": 2, "teaching": 1, "pipeline": 1, "unexplained": 1})
        evidence = chance(rng, 0.5)
        spend_ratio = rng.uniform(0.7, 1.1)
        statement = pick_w(rng, {"certified": 5, "draft": 2, "not_received": 1})
        late_days = pick(rng, [0, 0, 0, 0, 6, 14, 30])
        prior_concern = chance(rng, 0.2)
        recovery = pick_w(rng, {"credible": 3, "vague": 3, None: 2})
        tone = pick(rng, ["candid", "candid", "upbeat"])
        outputs_boost = False
        pattern = None
        if reg == "clear":
            if chance(rng, 0.5):
                delivery, cause, evidence = "on_track", pick(rng, EXTERNAL_CAUSES), True
                spend_ratio, statement, late_days, prior_concern = rng.uniform(0.93, 1.03), "certified", 0, False
                recovery = None
            else:
                delivery, cause = "failing", pick(rng, ["teaching", "pipeline", "unexplained"])
                spend_ratio = pick(rng, [rng.uniform(1.17, 1.35), rng.uniform(0.3, 0.45)])
                statement, late_days = pick(rng, ["draft", "not_received"]), pick(rng, [30, 45, 60])
                prior_concern, recovery = True, pick(rng, [None, "vague"])
        elif reg == "borderline":
            delivery = pick(rng, ["slipping", "behind"])
            cause = pick_w(rng, {"ethics": 1, "supplier": 1, "partner": 1, "weather": 1, "archive_closure": 1,
                                 "recruitment": 3, "pipeline": 1, "teaching": 1})
            evidence = chance(rng, 0.4)
            spend_ratio = pick(rng, [rng.uniform(0.66, 0.88), rng.uniform(0.95, 1.1)])
            recovery = pick(rng, ["credible", "vague", "vague"])
        else:
            pattern = pick(rng, ["rosy_report", "external_but_failing", "delivering_but_overspent", "strong_outputs"])
            if pattern == "rosy_report":
                delivery, tone, cause = "behind", "upbeat", pick(rng, ["unexplained", "pipeline", "teaching"])
                spend_ratio, recovery = rng.uniform(0.93, 1.05), pick(rng, [None, "vague"])
            elif pattern == "external_but_failing":
                delivery, cause, evidence = "failing", pick(rng, ["supplier", "ethics", "partner", "archive_closure"]), True
                spend_ratio, recovery = rng.uniform(0.45, 0.62), "credible"
            elif pattern == "delivering_but_overspent":
                delivery, spend_ratio, statement = pick(rng, ["on_track", "slipping"]), rng.uniform(1.18, 1.32), "draft"
            else:
                delivery, outputs_boost = "behind", True
                cause = pick(rng, ["recruitment", "weather", "pipeline"])
        kind, explanation, affected, evidence_doc, need = SLIP_CAUSES[cause]
        if evidence_doc is None:
            evidence = False
        title, code, tags = _topic(rng, need)

        years = pick(rng, [3, 3, 4, 5])
        year_done = rng.randint(1, years - 1)
        annual = _r(rng.uniform(60000, 180000), 1000)
        start = rand_date(rng, date(2023, 1, 1), date(2025, 6, 1)).replace(day=1)
        report_date = _add_months(start, 12 * year_done)
        n_due = pick(rng, [4, 5, 5, 6])
        pool = [m for m, t in MILESTONES if m != affected and (t is None or t in tags)]
        chosen = rng.sample(pool, n_due - 1) + [affected]
        chosen.sort(key=MILESTONE_NAMES.index)
        span = 12 * year_done
        dues = [_add_months(start, max(1, round((i + 1) * span / (n_due + 0.5)))) for i in range(n_due)]
        status = ["met"] * n_due
        a = chosen.index(affected)
        rest = [i for i in range(n_due) if i != a]
        rng.shuffle(rest)
        if delivery == "on_track":
            status[a] = "met_late"
        elif delivery == "slipping":
            status[a] = pick(rng, ["partly_met", "missed"])
            status[rest[0]] = "met_late"
        elif delivery == "behind":
            status[a] = "missed"
            status[rest[0]] = pick(rng, ["missed", "partly_met"])
            status[rest[1]] = "met_late"
        else:
            status[a] = "missed"
            for i in rest[: max(2, n_due // 2)]:
                status[i] = "missed"
            status[rest[-1]] = "partly_met"
        if delivery in ("behind", "failing") and chance(rng, 0.2):
            missed_rest = [i for i in rest if status[i] == "missed"]
            if missed_rest:
                status[missed_rest[0]] = "deferred_by_agreement"
        milestones = [{"ref": f"M{i + 1}", "item": chosen[i], "due": _ym(dues[i]), "status": status[i]}
                      for i in range(n_due)]
        budget_to_date = annual * year_done
        spent = _r(budget_to_date * spend_ratio, 100)
        good = {"on_track": 2, "slipping": 1, "behind": 1, "failing": 0}[delivery] + (2 if outputs_boost else 0)
        staff_opts = {None: 5, "co_i_left": 1} if cause == "recruitment" else {None: 5, "postdoc_replaced": 2,
                                                                                 "co_i_left": 1}
        return {
            "regime": reg, "pattern": pattern, "delivery": delivery, "cause": cause, "cause_kind": kind,
            "explanation": explanation, "affected_milestone": affected,
            "evidence": evidence_doc if evidence else None, "spend_ratio": round(spend_ratio, 3),
            "statement": statement, "late_days": late_days, "prior_concern": prior_concern, "recovery": recovery,
            "tone": tone, "milestones": milestones, "years": years, "year_done": year_done, "annual": annual,
            "total": annual * years, "start": start, "report_date": report_date, "budget_to_date": budget_to_date,
            "spent": spent, "papers_published": rng.randint(0, good + year_done - 1) if good else 0,
            "papers_submitted": rng.randint(0, 1 + good), "datasets": rng.randint(0, 1) if good else 0,
            "events": rng.randint(0, 2 + good), "staff_change": pick_w(rng, staff_opts),
            "title": title, "call_code": code, "holder": person(rng), "institution": org(rng, "school"),
            "funder": _funder(rng), "currency": pick_w(rng, CURRENCIES), "award_ref": ident(rng, "AWD", 6, "-"),
            "officer": person(rng),
        }

    def _texts(self, f):
        if f["tone"] == "upbeat":
            summary = "The project has made excellent progress and remains on course to deliver all its objectives."
        else:
            summary = {"on_track": "Progress this year was broadly to plan; one milestone ran late, as explained below.",
                       "slipping": "Most of this year's work went to plan, but some milestones slipped, as set out below.",
                       "behind": "Progress this year was slower than planned in several areas, as set out below.",
                       "failing": "This has been a difficult year, and much of the planned work is not yet done."}[
                f["delivery"]]
        recovery = {"credible": "A revised timetable is attached: the delayed work moves to the first half of next year, "
                                "and the final-year schedule has been compressed with the partners' agreement.",
                    "vague": "We expect to catch up during the coming year.", None: None}[f["recovery"]]
        staff = {"postdoc_replaced": "The first postdoctoral researcher left in the spring; a replacement started "
                                     "two months later.",
                 "co_i_left": "One co-investigator has moved to another institution and now contributes informally.",
                 None: None}[f["staff_change"]]
        concern = ("Last year's review asked for closer tracking of the milestones." if f["prior_concern"] else None)
        return summary, recovery, staff, concern

    def _statement(self, f):
        return {"certified": "certified by the institution's finance office", "draft": "draft, not yet certified",
                "not_received": "not received"}[f["statement"]]

    def _obj(self, f, rng):
        summary, recovery, staff, concern = self._texts(f)
        return prune({
            "award": {"ref": f["award_ref"], "title": f["title"], "holder": f["holder"], "institution": f["institution"],
                      "funder": f["funder"], "currency": f["currency"], "total_award": f["total"],
                      "duration_years": f["years"], "year_reported": f["year_done"], "start": _ym(f["start"]),
                      "next_tranche": f["annual"]},
            "milestones": f["milestones"],
            "finance": {"budget_to_date": f["budget_to_date"], "spent_to_date": f["spent"],
                        "variance_vs_profile_pct": round(100 * (f["spent"] / f["budget_to_date"] - 1), 1),
                        "statement": self._statement(f)},
            "narrative": {"summary": noisy(summary, rng, 0.1), "delays": noisy(f["explanation"], rng, 0.1),
                          "evidence_attached": f["evidence"], "recovery_plan": recovery},
            "outputs": {"papers_published": f["papers_published"], "papers_submitted": f["papers_submitted"],
                        "datasets_released": f["datasets"], "public_events": f["events"]},
            "record": {"report_received": "on time" if not f["late_days"] else f"{f['late_days']} days late",
                       "previous_review": concern, "staff_changes": staff},
        })

    def render(self, f, rng, surface):
        if surface == "json":
            return dumps(self._obj(f, rng))
        if surface == "portal_export":
            return dumps_pretty(self._obj(f, rng), rng)
        cur, st = f["currency"], _mstyle(rng)
        summary, recovery, staff, concern = self._texts(f)
        var = round(100 * (f["spent"] / f["budget_to_date"] - 1), 1)
        if surface == "progress_report":
            rows = [[m["ref"], m["item"], m["due"], STATUS_TEXT[m["status"]]] for m in f["milestones"]]
            narrative = join_sentences([summary, f["explanation"],
                                        f"Supporting document: {f['evidence']}." if f["evidence"] else "",
                                        recovery or "", staff or ""])
            parts = [f"ANNUAL PROGRESS REPORT: YEAR {f['year_done']} OF {f['years']}",
                     kv_lines([("Award", f"{f['award_ref']} ({f['funder']})"), ("Title", f["title"]),
                               ("Grant holder", f"{f['holder']}, {f['institution']}"),
                               ("Total award", money(f["total"], None, cur, st)),
                               ("Period", f"{_date(_add_months(f['report_date'], -12), rng)} to "
                                          f"{_date(f['report_date'] - timedelta(days=1), rng)}"),
                               ("Report received", "on time" if not f["late_days"]
                                else f"{f['late_days']} days after the due date")], rng),
                     "", "Milestones due to date", table(["Ref", "Milestone", "Due", "Status"], rows, rng), "",
                     f"Budget to date: {money(f['budget_to_date'], None, cur, st)}. Spent to date: "
                     f"{money(f['spent'], None, cur, st)} ({'+' if var >= 0 else ''}{var}% against profile). "
                     f"Financial statement: {self._statement(f)}.",
                     f"Outputs: {f['papers_published']} paper(s) published, {f['papers_submitted']} submitted, "
                     f"{f['datasets']} dataset(s) released, {f['events']} public event(s).",
                     "", "Grant holder's narrative:", noisy(narrative, rng, 0.12)]
            if concern:
                parts += ["", f"Note from previous review: {concern}"]
            return "\n".join(parts)
        # officer_memo
        cnt = {k: sum(1 for m in f["milestones"] if m["status"] == k) for k in STATUS_TEXT}
        tally = ", ".join(f"{v} {STATUS_TEXT[k]}" for k, v in cnt.items() if v)
        aff = next(m for m in f["milestones"] if m["item"] == f["affected_milestone"])
        stmt = {"certified": "has been certified by the institution's finance office",
                "draft": "is still a draft and has not been certified",
                "not_received": "has not been received"}[f["statement"]]
        paras = [
            f"{f['holder']} ({f['institution']}) has reported on year {f['year_done']} of {f['years']} of "
            f"{f['award_ref']}, \"{f['title']}\".",
            f"Of the {len(f['milestones'])} milestones due so far: {tally}.",
            f"The report says of '{aff['item']}' ({STATUS_TEXT[aff['status']]}): \"{f['explanation']}\"",
            (f"They have attached {f['evidence']}." if f["evidence"] else "No supporting document was attached."),
            f"Spend to date is {money(f['spent'], None, cur, st)} against a profile of "
            f"{money(f['budget_to_date'], None, cur, st)} ({'+' if var >= 0 else ''}{var}%); the financial "
            f"statement {stmt}.",
            ("The report arrived on time." if not f["late_days"] else f"The report arrived {f['late_days']} days late."),
            f"Outputs so far: {f['papers_published']} published and {f['papers_submitted']} submitted papers, "
            f"{f['events']} public event(s).",
            f"Their summary: \"{summary}\"",
            (f"On catching up: \"{recovery}\"" if recovery else "The report does not include a recovery plan."),
            staff or "", concern or "",
        ]
        return "\n".join([
            "To: Continuation Committee", f"From: {f['officer']}, Programme Officer",
            f"Re: {f['award_ref']}, year {f['year_done']} progress report", "",
            noisy(join_sentences(paras), rng, 0.08), "",
            f"The next tranche ({money(f['annual'], None, cur, st)}) is due on "
            f"{_date(f['report_date'] + timedelta(days=30), rng)}.",
        ])

    def questions(self):
        return {
            "continuation_release": choice("What should happen to next year's funding for this award?", {
                "release_in_full": "Release the next tranche on schedule with no conditions: no milestone missed or "
                                   "partly met, spend within 10% of profile, and a certified statement.",
                "release_with_conditions": "Release it with conditions such as a revised milestone plan: one milestone "
                                           "missed or partly met, spend off profile, or a draft statement.",
                "release_reduced": "Release part now and hold the rest: two or more milestones missed or partly met, "
                                   "with only a vague catch-up plan or spending well off profile.",
                "withhold_pending_plan": "Hold the whole tranche until an acceptable recovery plan arrives: milestones "
                                         "have slipped and there is no recovery plan at all, or no financial statement.",
                "suspend_for_review": "Suspend the award and refer it to the committee: most milestones are missed for "
                                      "reasons within the team's control and no credible plan is offered.",
            }),
            "follow_up_contact": choice("What follow-up should the programme officer arrange before the next report "
                                        "is due?", {
                "routine_only": "Nothing beyond the normal reporting cycle: milestones on track, spend on profile, "
                                "report complete.",
                "written_query": "A written request for a missing item, such as evidence for a claimed outside delay "
                                 "or an explanation for a late report.",
                "progress_call": "A call with the grant holder to go through two or more slipped milestones or a vague "
                                 "catch-up plan.",
                "finance_check": "A check of the institution's expenditure records: the financial statement is "
                                 "uncertified or missing, or spending runs more than 15% over profile.",
                "site_visit": "An in-person visit: most milestones due have been missed, or last year's concern has "
                              "come back with more slippage.",
            }),
            "milestone_delivery": score("How well has the project delivered the milestones due so far?", [
                "Poor: most milestones due have been missed, with no credible route to recovery.",
                "Behind: several have slipped, though catching up still looks possible.",
                "Mostly on track: one late or partial milestone, or slippage already agreed with the funder.",
                "On track: everything due has been delivered on time.",
            ]),
            "spend_alignment": score("How well does spending to date line up with the progress the project has made?", [
                "Out of line: spending is more than 10% over the profile while milestones have been missed.",
                "Questionable: spending is on profile but two or more milestones are missed or partly met, or it is "
                "more than 10% off profile while everything is delivered.",
                "Broadly in line: under-spent by more than 10% in step with missed milestones, or on profile with one "
                "milestone missed or partly met.",
                "In line: spending within 10% of the profile and no milestone missed or partly met.",
            ]),
            "delay_outside_control": noul(
                "The slippage reported this year was caused mainly by events outside the grant holder's control.",
                yes="The delay traces to a regulator, supplier, partner or event the team could not reasonably have "
                    "prevented or planned around.",
                no="The delay comes from the team's own planning, staffing or management, or no convincing cause "
                   "is given."),
        }


# =========================================================================== schema 4
# reason: (kind, text, supporting document or None, topic tag it needs or None)
EXT_REASONS = {
    "supplier": ("external", "Delivery of the main equipment slipped by seven months because of the supplier's backlog.",
                 "the supplier's delivery notice", "equip"),
    "site_access": ("external", "The partner organisation closed its sites to researchers for five months during a "
                                "reorganisation.", "the partner's letter", None),
    "ethics": ("external", "The ethics committee asked for two rounds of amendments, which held up recruitment of "
                           "participants by four months.", "the committee's correspondence", "people"),
    "archive_closure": ("external", "The record office holding the collection closed for five months for roof repairs.",
                        "the record office's closure notice", "archive"),
    "staff_departure": ("mixed", "The postdoctoral researcher left in month {m} and the post took five months to "
                                 "refill.", None, None),
    "underestimate": ("internal", "The analysis has turned out to be more involved than we planned for.", None, None),
    "new_opportunity": ("scope", "A large new dataset has just become available and we would like time to add it "
                                 "to the study.", None, None),
    "transfer_only": ("transfer", "", None, None),
}
EXTERNAL_REASONS = ["supplier", "site_access", "ethics", "archive_closure"]
TRANSFERS = [
    {"key": "travel_to_consumables", "from": "travel", "to": "consumables", "scope": 0,
     "why": "Two conferences moved online; the saving would cover extra materials and analysis costs."},
    {"key": "equipment_to_staff", "from": "equipment", "to": "staff", "scope": 0,
     "why": "The equipment came in under quote; the saving would keep the research assistant on for longer."},
    {"key": "consumables_to_staff", "from": "consumables", "to": "staff", "scope": 1,
     "why": "Less material is needed than planned, so we would keep the technician on for longer instead."},
    {"key": "staff_to_equipment", "from": "staff", "to": "equipment", "scope": 2,
     "why": "Salary savings from a four-month vacancy would buy additional equipment so the work can run in parallel."},
    {"key": "staff_to_other", "from": "staff", "to": "other_direct", "scope": 3,
     "why": "We would like to use salary savings to fund a public exhibition, which was not part of the original plan."},
    {"key": "consumables_to_travel", "from": "consumables", "to": "travel", "scope": 2,
     "why": "We propose adding fieldwork at a new partner site that was not in the original plan."},
]


class ExtensionReallocationRequest(Schema):
    name = "extension_reallocation_request"
    description = ("A grant holder asks for a no-cost extension, a transfer between budget lines, or both, under "
                   "the funder's written variation rules.")
    surfaces = {"json": 0.45, "pi_email": 0.3, "variation_form": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        req_type = pick_w(rng, {"extension": 3, "transfer": 2, "both": 3})
        max_ext = pick(rng, [6, 12, 12])
        notice = pick(rng, [60, 90])
        officer_pct = pick(rng, [10, 15, 20])
        equip_committee = chance(rng, 0.5)
        # clean defaults
        ext = pick(rng, [m for m in (3, 4, 6, 9) if m <= max_ext])
        prior_ext = 0
        days_notice = notice + rng.randint(15, 100)
        reason = pick(rng, EXTERNAL_REASONS)
        evidence = True
        tr = pick(rng, [t for t in TRANSFERS if t["scope"] <= 1])
        tr_target = rng.uniform(1.0, officer_pct * 0.45)
        prior_tr = pick(rng, [0, 0, 2, 3])
        endorsed = True
        progress = "on_track_except_delay"
        pattern = None
        if reg == "clear":
            if chance(rng, 0.5):
                ext = max_ext + pick(rng, [6, 12])
                prior_ext = 1
                days_notice = rng.randint(5, 25)
                reason, evidence = pick(rng, ["underestimate", "new_opportunity"]), False
                tr = pick(rng, [t for t in TRANSFERS if t["scope"] >= 2])
                tr_target = officer_pct * rng.uniform(1.5, 2.5)
                endorsed, progress = chance(rng, 0.3), "behind"
        elif reg == "borderline":
            # only the knobs this kind of request actually has
            knob_pool = {"extension": ["notice", "length", "reason"], "transfer": ["notice", "transfer_size"],
                         "both": ["notice", "length", "transfer_size", "reason"]}[req_type]
            knobs = rng.sample(knob_pool, min(len(knob_pool), pick(rng, [1, 2, 2])))
            if "notice" in knobs:
                days_notice = near(rng, notice, 12, lo=10, integer=True)
            if "length" in knobs:
                ext = max_ext + pick(rng, [0, 0, 1, 2, 3])
            if "transfer_size" in knobs:
                tr = pick(rng, [t for t in TRANSFERS if t["from"] in ("staff", "equipment", "consumables")])
                tr_target = near(rng, officer_pct - prior_tr, 3, lo=1.0)
            if "reason" in knobs:
                reason = pick(rng, ["staff_departure", "staff_departure", "supplier", "site_access"])
                evidence = reason != "staff_departure" and chance(rng, 0.3)
            progress = pick(rng, ["on_track_except_delay", "on_track_except_delay", "behind"])
            endorsed = chance(rng, 0.75)
        else:
            pattern = pick(rng, ["good_reason_rule_break", "in_policy_scope_drift", "evidence_vs_progress"])
            if pattern == "good_reason_rule_break":
                prior_ext = pick(rng, [1, 1, 0])
                req_type = pick(rng, ["extension", "both"])
                if prior_ext == 0:
                    days_notice = rng.randint(15, notice - 15)
            elif pattern == "in_policy_scope_drift":
                tr = pick(rng, [t for t in TRANSFERS if t["scope"] >= 2])
                tr_target = rng.uniform(officer_pct * 0.4, officer_pct * 0.9)
                req_type = pick(rng, ["transfer", "both"])
            else:
                reason, evidence, progress = pick(rng, EXTERNAL_REASONS), True, "behind"
                ext, req_type = max_ext, pick(rng, ["extension", "both"])
        if req_type == "extension":
            tr = None
        if req_type == "transfer":
            reason, evidence = "transfer_only", False
        kind, reason_text, evidence_doc, need = EXT_REASONS[reason]
        if evidence_doc is None:
            evidence = False
        title, code, _ = _topic(rng, need)

        months_total = pick(rng, [24, 36, 36, 48])
        start = rand_date(rng, date(2023, 1, 1), date(2025, 3, 1)).replace(day=1)
        end = _add_months(start, months_total) - timedelta(days=1)
        received = end - timedelta(days=days_notice)
        elapsed = min(0.97, max(0.4, (received - start).days / (end - start).days))
        direct = _r(rng.uniform(150000, 480000), 1000)
        shares = {"staff": rng.uniform(0.55, 0.68), "equipment": rng.uniform(0.06, 0.18),
                  "travel": rng.uniform(0.03, 0.07), "other_direct": rng.uniform(0.01, 0.04)}
        lines = {k: _r(direct * v, 100) for k, v in shares.items()}
        lines["consumables"] = direct - sum(lines.values())
        lines = {k: lines[k] for k in ("staff", "equipment", "consumables", "travel", "other_direct")}
        indirect = _r(direct * pick(rng, [0.1, 0.15, 0.2]), 100)
        total = direct + indirect
        pace = {"staff": elapsed * rng.uniform(0.7, 0.9), "equipment": rng.uniform(0.75, 0.95),
                "consumables": elapsed * rng.uniform(0.6, 1.0), "travel": elapsed * rng.uniform(0.4, 0.9),
                "other_direct": elapsed * rng.uniform(0.5, 1.0)}
        spent = {k: _r(lines[k] * min(1.0, pace[k]), 100) for k in lines}
        transfer = None
        if tr is not None:
            remaining_from = lines[tr["from"]] - spent[tr["from"]]
            amount = _r(min(total * tr_target / 100, remaining_from * 0.9), 100)
            if amount < 500:
                amount = _r(max(500, remaining_from * 0.5), 100)
            transfer = {"from": tr["from"], "to": tr["to"], "amount": amount,
                        "pct_of_award": round(100 * amount / total, 1), "why": tr["why"], "scope": tr["scope"],
                        "key": tr["key"]}
        return {
            "regime": reg, "pattern": pattern, "request_type": req_type,
            "extension_months": ext if req_type != "transfer" else None,
            "prior_extensions": prior_ext, "days_notice": days_notice, "reason": reason, "reason_kind": kind,
            "reason_text": reason_text.format(m=rng.randint(10, 20)) or None,
            "evidence": evidence_doc if evidence else None,
            "transfer": transfer, "prior_transfers_pct": prior_tr, "endorsed": endorsed, "progress": progress,
            "max_extension_months": max_ext, "min_notice_days": notice, "officer_transfer_limit_pct": officer_pct,
            "equipment_to_committee": equip_committee, "lines": lines, "spent": spent, "indirect": indirect,
            "total": total, "start": start, "end": end, "received": received,
            "title": title, "call_code": code, "holder": person(rng), "institution": org(rng, "school"),
            "funder": _funder(rng), "currency": pick_w(rng, CURRENCIES), "award_ref": ident(rng, "GR", 6, "-"),
            "officer": person(rng),
        }

    def _rules(self, f):
        extra = ("Any transfer into equipment goes to the grants committee. " if f["equipment_to_committee"] else "")
        return {"max_extension_months": f["max_extension_months"], "extensions_per_award": 1,
                "min_notice_days_before_end": f["min_notice_days"],
                "officer_may_approve_transfers_up_to_pct_of_award": f["officer_transfer_limit_pct"],
                "notes": f"Extensions carry no extra money. The transfer limit counts all transfers made over the "
                         f"life of the award. {extra}Nothing may move into or out of indirect costs; larger or "
                         f"out-of-rule requests go to the grants committee."}

    def _rules_text(self, f):
        return [f"no-cost extensions of up to {f['max_extension_months']} months, one per award",
                f"requests at least {f['min_notice_days']} days before the end date",
                f"the programme officer may approve transfers totalling up to {f['officer_transfer_limit_pct']}% of "
                "the award over its life; anything larger goes to the grants committee",
                "no transfers into or out of indirect costs"] + (
            ["any transfer into equipment goes to the committee"] if f["equipment_to_committee"] else [])

    def _progress(self, f):
        if f["progress"] == "behind":
            return "Two of the four work packages are currently behind schedule."
        if f["request_type"] == "transfer":
            return "The project is otherwise on schedule."
        return "Apart from the delayed work package, the project is on schedule."

    def render(self, f, rng, surface):
        cur, t = f["currency"], f["transfer"]
        new_end = _add_months(f["end"], f["extension_months"]) if f["extension_months"] else None
        rows = [[k.replace("_", " "), f["lines"][k], f["spent"][k], f["lines"][k] - f["spent"][k]] for k in f["lines"]]
        if surface == "json":
            return dumps(prune({
                "award": {"ref": f["award_ref"], "title": f["title"], "holder": f["holder"],
                          "institution": f["institution"], "funder": f["funder"], "currency": cur,
                          "total": f["total"], "of_which_indirect": f["indirect"], "start": f["start"].isoformat(),
                          "end": f["end"].isoformat(), "extensions_granted_so_far": f["prior_extensions"],
                          "transfers_approved_so_far_pct": f["prior_transfers_pct"]},
                "request": {"received": f["received"].isoformat(),
                            "extension_months": f["extension_months"],
                            "proposed_end": new_end.isoformat() if new_end else None,
                            "transfer": ({"from": t["from"], "to": t["to"], "amount": t["amount"],
                                          "purpose": t["why"]} if t else None),
                            "reason": noisy(f["reason_text"], rng, 0.1) if f["reason_text"] else None,
                            "evidence_attached": f["evidence"],
                            "progress": self._progress(f), "research_office_endorsed": f["endorsed"]},
                "budget_position": [{"line": r[0], "awarded": r[1], "spent": r[2], "remaining": r[3]} for r in rows],
                "rules": self._rules(f),
            }))
        st = _mstyle(rng)
        trows = [[r[0]] + [_num(x) for x in r[1:]] for r in rows]
        if surface == "pi_email":
            asks = []
            if f["extension_months"]:
                asks.append(f"a no-cost extension of {f['extension_months']} months, moving the end date from "
                            f"{_date(f['end'], rng)} to {_date(new_end, rng)}")
            if t:
                asks.append(f"permission to move {money(t['amount'], None, cur, st)} from {t['from'].replace('_', ' ')} "
                            f"to {t['to'].replace('_', ' ')}")
            body = noisy(join_sentences([
                f"I am writing to request {' and '.join(asks)} on {f['award_ref']}.",
                f["reason_text"] or "",
                (f"I attach {f['evidence']}." if f["evidence"] else ""),
                (t["why"] if t else ""),
                self._progress(f),
                ("Our research office has reviewed this request and supports it." if f["endorsed"] else ""),
                ("I should mention that the award was already extended once." if f["prior_extensions"] else ""),
            ]), rng, 0.12)
            return "\n".join([
                f"Subject: {f['award_ref']}: variation request",
                f"From: {email_addr(f['holder'], _domain(f['institution']))}",
                f"To: {email_addr(f['officer'], _domain(f['funder']))}",
                f"Date: {_date(f['received'], rng)}", "",
                f"Dear {f['officer'].split()[0]},", "", body, "", "Best wishes,", f["holder"], "",
                "---- added by the grants system ----",
                kv_lines([("Award", f"{f['title']} ({f['institution']})"),
                          ("Award total", f"{money(f['total'], None, cur, st)}, of which indirect "
                                          f"{money(f['indirect'], None, cur, st)}"),
                          ("Current end date", _date(f["end"], rng)),
                          ("Extensions so far", f["prior_extensions"]),
                          ("Transfers approved so far", f"{f['prior_transfers_pct']}% of award")], rng),
                "Variation rules: " + "; ".join(self._rules_text(f)) + ".",
                table(["Line", "Awarded", "Spent", "Remaining"], trows, rng),
            ])
        # variation_form
        req_rows = [("Award", f["award_ref"]), ("Grant holder", f"{f['holder']}, {f['institution']}"),
                    ("Award period", f"{_date(f['start'], rng)} to {_date(f['end'], rng)}"),
                    ("Request received", _date(f["received"], rng)),
                    ("Extension requested", f"{f['extension_months']} months" if f["extension_months"] else "none"),
                    ("Budget transfer", f"{_num(t['amount'])} {cur} from {t['from'].replace('_', ' ')} to "
                                        f"{t['to'].replace('_', ' ')} ({t['pct_of_award']}% of award)" if t else "none"),
                    ("Previous extensions", f["prior_extensions"]),
                    ("Previous transfers", f"{f['prior_transfers_pct']}% of award"),
                    ("Evidence attached", f["evidence"] or "none"),
                    ("Research office endorsement", "yes" if f["endorsed"] else "no")]
        return "\n".join([
            f"GRANT VARIATION REQUEST | {f['funder']}", kv_lines(req_rows, rng), "",
            "Reason given:", noisy(join_sentences([f["reason_text"] or "", t["why"] if t else "", self._progress(f)]),
                                   rng, 0.1),
            "", f"Budget position ({cur})", table(["Line", "Awarded", "Spent", "Remaining"], trows, rng),
            f"Indirect costs (not transferable): {_num(f['indirect'])}", "",
            "Rules that apply:", bullets(self._rules_text(f), rng),
        ])

    def questions(self):
        return {
            "variation_decision": choice("How should the funder respond to this variation request?", {
                "approve_as_requested": "Approve the extension and any budget transfer exactly as asked.",
                "approve_modified": "Approve a reduced version, such as a shorter extension or a smaller transfer.",
                "request_more_information": "Decide nothing yet and ask the grant holder for specific missing "
                                            "evidence or detail.",
                "decline": "Refuse the request; the award continues on its current terms and end date.",
            }),
            "justification_strength": score("How convincing is the grant holder's case for the change?", [
                "Unconvincing: the reason is the team's own under-estimate, or a wish to add new work.",
                "Thin: a plausible cause with no supporting document (such as a staff departure), or a transfer that "
                "moves money into kinds of activity the award did not plan.",
                "Reasonable: a documented outside cause while other work is also behind, or a transfer of genuine "
                "savings towards the original aims.",
                "Compelling: a documented outside cause, with the rest of the project on schedule.",
            ]),
            "scope_shift": score("How far would the change move the project away from the work originally funded?", [
                "Not at all: the same aims and outputs, with only the timing or the cost split changing.",
                "Slightly: a small change of method or resources in service of the original aims.",
                "Moderately: an activity would be added, dropped or substantially redesigned.",
                "Substantially: the money would in effect fund different work.",
            ]),
            "within_policy": noul(
                "The request falls within the funder's written rules on extensions and budget transfers.",
                yes="Length, notice, number of extensions, and the size and direction of any transfer all fit the "
                    "stated rules.",
                no="At least one element breaks a stated rule or limit."),
            "committee_referral": noul("The request goes beyond what a programme officer may approve alone and needs "
                                       "the grants committee."),
        }


# =========================================================================== schema 5
# kind: summary template, permit or None, lasting output, (participants lo, hi), (audience lo, hi) or None,
# the label of the space / kit line in the budget
ARTS_PROJECTS = [
    {"kind": "mural", "text": "paint a mural on the side wall of {venue}, designed with residents at four drop-in "
                              "sessions", "permit": "wall-use permit", "lasting": True, "part": (30, 120),
     "aud": None, "space": "Scaffolding and access equipment"},
    {"kind": "street_festival", "text": "run a one-day festival of music and dance on {street}",
     "permit": "road-closure permit", "lasting": False, "part": (20, 60), "aud": (500, 2500),
     "space": "Stage and sound hire"},
    {"kind": "print_workshops", "text": "hold a ten-week printmaking course for teenagers at {venue}, ending with a "
                                        "show of their work", "permit": None, "lasting": False, "part": (10, 24),
     "aud": (40, 150), "space": "Venue hire"},
    {"kind": "oral_history", "text": "record oral histories from long-standing residents of {area} and hold a "
                                     "listening evening", "permit": None, "lasting": True, "part": (15, 40),
     "aud": (40, 120), "space": "Venue hire"},
    {"kind": "park_concerts", "text": "stage two evening concerts by a community choir in {park}",
     "permit": "park events permit", "lasting": False, "part": (25, 60), "aud": (150, 800),
     "space": "Stage and sound hire"},
    {"kind": "popup_gallery", "text": "run a three-week pop-up exhibition of local photographers in an empty shop on "
                                      "{street}", "permit": "temporary-use licence", "lasting": False, "part": (8, 20),
     "aud": (300, 1500), "space": "Shop unit rent"},
    {"kind": "youth_theatre", "text": "devise and perform a new play with a youth theatre group at {venue}",
     "permit": None, "lasting": False, "part": (12, 25), "aud": (80, 300), "space": "Venue hire"},
]
VENUES = ["the Northgate library", "the Elm Street community centre", "the old tram depot", "the Riverside youth club",
          "the Market Hall"]
STREETS = ["Market Row", "Canal Street", "Station Parade", "Mill Lane"]
PARKS = ["Hollins Park", "the Riverside green", "Quarry Gardens"]
NEIGHBOURHOODS = ["the Eastfield estate", "Harbourside ward", "the Old Town", "Brookside"]
GROUP_SUFFIX = ["Community Arts Collective", "Residents' Association", "Arts Network", "Print Club",
                "Heritage Group", "Youth Arts Project"]
SEASONS = ["Spring", "Summer", "Autumn", "Winter"]


class CommunityArtsApplication(Schema):
    name = "community_arts_application"
    description = "A local group's application to a city's small community-arts grant fund, judged against its guidelines."
    surfaces = {"json": 0.45, "application_form": 0.3, "assessor_email": 0.25}

    def _plant(self, f, rng, issue, severe):
        if issue == "over_cap":
            f["req_frac"] = rng.uniform(1.25, 1.6) if severe else pick(rng, [1.0, rng.uniform(1.01, 1.06)])
        elif issue == "artist_pay":
            f["rate"] = int(f["fee_floor"] * 0.5) if severe else f["fee_floor"] - pick(rng, [10, 25, 50])
            f["pay_note"] = None if severe else "Our artists have agreed to donate part of their time."
        elif issue == "hospitality":
            f["hosp_pct"] = f["hosp_cap"] + (rng.uniform(8, 12) if severe else rng.uniform(0.6, 3.5))
        elif issue == "match":
            if f["match_min"]:
                f["match_pct"] = 0.0 if severe else max(0.0, f["match_min"] - rng.uniform(1, 4))
            else:
                f["hosp_pct"] = f["hosp_cap"] + rng.uniform(0.6, 3.5)
        elif issue == "structure":
            f["structure"] = "unincorporated"
            f["sponsor"] = None if severe else "agreed in principle, letter to follow"
        elif issue == "location":
            f["based_in_city"] = False
        elif issue == "readiness":
            if severe:
                f["venue_status"], f["weeks_to_start"] = "not yet found", rng.randint(2, 4)
                f["permit_status"] = "not applied for" if f["permit"] else None
            else:
                f["weeks_to_start"] = rng.randint(4, 7)
                if f["permit"] and chance(rng, 0.7):
                    f["permit_status"] = "applied for, decision pending"
                else:
                    f["venue_status"] = "provisional booking"
        elif issue == "prior_report":
            f["prior_grant"] = "report_overdue" if severe else "report_filed_late"
        elif issue == "reach":
            f["reach"] = "weak" if severe else "thin"

    def sample_factors(self, rng):
        reg = regime(rng)
        proj = pick(rng, ARTS_PROJECTS)
        cap = pick(rng, [3000, 5000, 5000, 7500, 10000])
        floor = pick(rng, [200, 250, 300])
        hosp_cap = pick(rng, [5, 10, 10])
        match_min = pick(rng, [None, None, 10, 20])
        f = {
            "regime": reg, "project_kind": proj["kind"], "currency": pick_w(rng, {"USD": 4, "GBP": 3, "CAD": 2, "EUR": 1}),
            "cap": cap, "fee_floor": floor, "hosp_cap": hosp_cap, "match_min": match_min, "permit": proj["permit"],
            "lasting": proj["lasting"], "req_frac": rng.uniform(0.6, 0.95),
            "rate": floor + pick(rng, [0, 25, 50, 100]), "pay_note": None,
            "hosp_pct": pick(rng, [0.0, rng.uniform(1, max(1.5, hosp_cap - 2))]),
            "match_pct": (match_min or 0) + rng.uniform(3, 20) if (match_min or chance(rng, 0.6)) else 0.0,
            "structure": pick(rng, ["registered_charity", "unincorporated"]), "sponsor": None, "based_in_city": True,
            "venue_status": "confirmed", "permit_status": "granted" if proj["permit"] else None,
            "weeks_to_start": rng.randint(8, 20), "reach": "normal",
            "access": pick(rng, ["free", "free", "free", "pay_what_you_can"]),
            "prior_grant": pick_w(rng, {"none": 3, "report_filed": 2}),
            "projects_run": pick(rng, [0, 1, 3, 6]), "insurance": pick(rng, ["own policy", "covered by the venue"]),
            "partner_letter": chance(rng, 0.6), "pattern": None,
        }
        if f["structure"] == "unincorporated":
            f["sponsor"] = f"{rng.choice(ORG_A)} Community Trust"
        pool = {"over_cap": 3, "artist_pay": 3, "hospitality": 2, "match": 2, "structure": 2, "location": 1,
                "readiness": 3, "prior_report": 1, "reach": 2}
        issues = []
        if reg == "clear":
            if chance(rng, 0.5):
                issues.append((pick(rng, ["over_cap", "artist_pay", "structure", "prior_report", "readiness"]), True))
                if chance(rng, 0.5):
                    issues.append((pick(rng, ["reach", "hospitality", "readiness"]), True))
        elif reg == "borderline":
            first = pick_w(rng, pool)
            issues.append((first, False))
            if chance(rng, 0.45):
                issues.append((pick_w(rng, {k: v for k, v in pool.items() if k != first}), False))
        else:
            pattern = pick(rng, ["great_reach_rule_break", "rules_fine_weak_reach", "ready_but_underpaid"])
            f["pattern"] = pattern
            if pattern == "great_reach_rule_break":
                f["reach"], f["access"], f["projects_run"] = "wide", "free", pick(rng, [3, 6, 10])
                issues.append((pick(rng, ["structure", "prior_report", "over_cap", "location"]),
                               pick(rng, [True, False])))
            elif pattern == "rules_fine_weak_reach":
                f["projects_run"] = pick(rng, [6, 10])
                issues.append(("reach", pick(rng, [True, False])))
            else:
                issues.append(("artist_pay", False))
                f["partner_letter"] = True
        seen = []
        for issue, severe in issues:
            if issue not in seen:
                seen.append(issue)
                self._plant(f, rng, issue, severe)
        f["issues"] = [f"{i}:{'severe' if s else 'mild'}" for i, s in issues]
        if f["prior_grant"] != "none":
            f["projects_run"] = max(1, f["projects_run"])

        # ---- reach follows the kind of project, then the planted issue
        plo, phi = proj["part"]
        participants = rng.randint(plo, phi)
        audience = rng.randint(*proj["aud"]) if proj["aud"] else None
        if f["reach"] == "wide":
            participants = phi + rng.randint(0, phi // 2)
            audience = int(proj["aud"][1] * rng.uniform(1.0, 1.4)) if proj["aud"] else None
        elif f["reach"] == "thin":
            participants = plo
            audience = max(15, int(proj["aud"][0] * rng.uniform(0.4, 0.7))) if proj["aud"] else None
            f["access"] = pick(rng, ["ticketed", "free"])
        elif f["reach"] == "weak":
            participants = max(6, int(plo * 0.6))
            audience = rng.randint(15, 35) if proj["aud"] else None
            f["access"] = "members_only"

        # ---- budget: request and match first, then lines that add up exactly
        cap = f["cap"]
        R = _r(cap * f["req_frac"], 10)
        m = f["match_pct"] / 100
        T = _r(R / (1 - m), 10) if m > 0 else R
        artists = pick(rng, [1, 2, 2, 3, 4])
        rate = f["rate"]
        days = max(1, int(T * rng.uniform(0.3, 0.5) / (rate * artists)))
        while rate * artists * days > 0.6 * T and (days > 1 or artists > 1):
            if days > 1:
                days -= 1
            else:
                artists -= 1
        fees = rate * artists * days
        refresh = _r(R * f["hosp_pct"] / 100, 10)
        rest = T - fees - refresh
        space_free = proj["space"] == "Venue hire" and chance(rng, 0.25)
        space_cost = 0 if space_free else _r(rest * rng.uniform(0.1, 0.35), 10)
        marketing = _r(rest * rng.uniform(0.08, 0.18), 10)
        contingency = _r(rest * rng.uniform(0.0, 0.08), 10)
        materials = rest - space_cost - marketing - contingency
        other = T - R
        place = {"venue": pick(rng, VENUES), "street": pick(rng, STREETS), "park": pick(rng, PARKS),
                 "area": pick(rng, NEIGHBOURHOODS)}
        city = rng.choice(ORG_A)
        opened = rand_date(rng)
        f.update(
            participants=participants, audience=audience,
            request=R, total_cost=T, other_funding=other, artists=artists, days=days, artist_fees=fees,
            refreshments=refresh, space_label=proj["space"], space_cost=space_cost, space_free=space_free,
            marketing=marketing, contingency=contingency, materials=materials,
            hosp_actual_pct=round(100 * refresh / R, 1) if R else 0.0,
            match_actual_pct=round(100 * other / T, 1) if T else 0.0,
            other_source=pick(rng, ["ward councillors' fund", "local business sponsorship", "in-kind volunteer time",
                                    "a community foundation grant"]) if other else None,
            summary=proj["text"].format(**place), venue_name=place["venue"], city=city,
            neighbour_town=pick(rng, [a for a in ORG_A if a != city]),
            group=f"{rng.choice(ORG_A)} {pick(rng, GROUP_SUFFIX)}",
            contact=person(rng), officer=person(rng), assessor=person(rng), ref=ident(rng, "CAF", 4, "-"),
            round=f"{pick(rng, SEASONS)} {opened.year}", start=opened + timedelta(weeks=f["weeks_to_start"]),
            ticket_price=pick(rng, [8, 12, 15]) if f["access"] == "ticketed" else None,
        )
        return f

    def _rules_list(self, f, st):
        cur = f["currency"]
        r = [f"requests up to {money(f['cap'], None, cur, st)}",
             f"artists paid at least {money(f['fee_floor'], None, cur, st)} per day",
             f"refreshments and hospitality no more than {f['hosp_cap']}% of the amount requested"]
        if f["match_min"]:
            r.append(f"at least {f['match_min']}% of the total cost from other sources (cash or in kind)")
        r += [f"groups must be based in {f['city']}; unincorporated groups apply through a fiscal sponsor",
              "previous grantees must have filed their final report", "activities must be open to the public"]
        return r

    def _structure(self, f):
        if f["structure"] == "registered_charity":
            return "registered charity"
        return "unincorporated group, " + (f"fiscal sponsor {f['sponsor']}" if f["sponsor"] else "no fiscal sponsor named")

    def _prior(self, f):
        return {"none": "none", "report_filed": "one grant in the last round; final report filed",
                "report_filed_late": "one grant in the last round; final report filed seven weeks late",
                "report_overdue": "one grant in the last round; final report overdue"}[f["prior_grant"]]

    def _access(self, f, st):
        return {"free": "free and open to all", "pay_what_you_can": "open to all, pay what you can",
                "ticketed": "open to all, tickets " + money(f["ticket_price"], None, f["currency"], st)
                if f["ticket_price"] else "ticketed",
                "members_only": "for group members and their guests"}[f["access"]]

    def _based(self, f):
        return f["city"] if f["based_in_city"] else f"{f['neighbour_town']} (the project itself takes place in {f['city']})"

    def _reach(self, f):
        s = f"{f['participants']} people taking part"
        return s + (f", audience of about {f['audience']}" if f["audience"] else ", work stays on public view")

    def render(self, f, rng, surface):
        cur = f["currency"]
        fund = f"{f['city']} Community Arts Fund"
        permit = (f"{f['permit']}: {f['permit_status']}" if f["permit"] else None)
        summary = noisy(f"We will {f['summary']}.", rng, 0.12)
        if f["pay_note"]:
            summary += " " + f["pay_note"]
        st = _mstyle(rng)
        if surface == "json":
            return dumps(prune({
                "fund": {"name": fund, "round": f["round"],
                         "rules": {"max_request": f["cap"], "currency": cur, "min_artist_day_rate": f["fee_floor"],
                                   "hospitality_max_pct_of_request": f["hosp_cap"],
                                   "min_other_funding_pct_of_cost": f["match_min"],
                                   "eligibility": f"Groups based in {f['city']}; unincorporated groups apply through a "
                                                  "fiscal sponsor; previous grantees must have filed their final report.",
                                   "access": "Activities must be open to the public."}},
                "applicant": {"ref": f["ref"], "group": f["group"], "contact": f["contact"],
                              "structure": self._structure(f), "based_in": self._based(f),
                              "previous_grants": self._prior(f), "projects_run_before": f["projects_run"],
                              "partner_letter": "attached" if f["partner_letter"] else None},
                "project": {"summary": summary, "start": f["start"].isoformat(),
                            "venue_status": f["venue_status"], "permit": permit,
                            "expected_participants": f["participants"], "expected_audience": f["audience"],
                            "access": self._access(f, st), "insurance": f["insurance"],
                            "lasting_output": f["lasting"] or None},
                "budget": {"artist_fees": {"artists": f["artists"], "days_each": f["days"], "day_rate": f["rate"],
                                           "subtotal": f["artist_fees"]},
                           "materials": f["materials"],
                           f["space_label"].lower().replace(" ", "_"): (
                               f"0 (provided free by {f['venue_name']})" if f["space_free"] else f["space_cost"]),
                           "marketing": f["marketing"],
                           "refreshments": f["refreshments"] or None, "contingency": f["contingency"] or None,
                           "total_cost": f["total_cost"],
                           "other_funding": ({"amount": f["other_funding"], "source": f["other_source"]}
                                             if f["other_funding"] else None),
                           "requested": f["request"]},
            }))
        rows = [["Artist fees", f"{f['artists']} artist(s) x {f['days']} day(s) at {_num(f['rate'])}/day",
                 _num(f["artist_fees"])],
                ["Materials", "", _num(f["materials"])],
                [f["space_label"], f"provided free by {f['venue_name']}" if f["space_free"] else "",
                 _num(f["space_cost"])],
                ["Marketing and print", "", _num(f["marketing"])]]
        if f["refreshments"]:
            rows.append(["Refreshments", "", _num(f["refreshments"])])
        if f["contingency"]:
            rows.append(["Contingency", "", _num(f["contingency"])])
        rows.append(["TOTAL COST", "", _num(f["total_cost"])])
        if f["other_funding"]:
            rows.append(["Other funding", f["other_source"], "-" + _num(f["other_funding"])])
        rows.append(["REQUESTED", "", _num(f["request"])])
        btable = table(["Item", "Detail", f"Cost ({cur})"], rows, rng)
        if surface == "application_form":
            return "\n".join([
                f"{fund.upper()} | SMALL GRANTS APPLICATION | {f['round']}",
                kv_lines([("Reference", f["ref"]), ("Group", f["group"]), ("Contact", f["contact"]),
                          ("Legal structure", self._structure(f)), ("Based in", self._based(f)),
                          ("Projects run before", f["projects_run"]),
                          ("Previous grants from the fund", self._prior(f)),
                          ("Start date", _date(f["start"], rng)), ("Venue", f["venue_status"]),
                          ("Permit", permit), ("Expected reach", self._reach(f)),
                          ("Access", self._access(f, st)), ("Insurance", f["insurance"]),
                          ("Partner letter", "attached" if f["partner_letter"] else "none")], rng),
                "", "What will you do?", summary, "", btable, "",
                "Guidelines printed on the form: " + "; ".join(self._rules_list(f, st)) + ".",
            ])
        # assessor_email
        facts = join_sentences([
            f"{f['group']} ({self._structure(f)}; based in {self._based(f)}) want to {f['summary']}, starting "
            f"{_date(f['start'], rng)}.",
            f"They have run {_pl(f['projects_run'], 'project')} before; previous grants: {self._prior(f)}.",
            f"Venue: {f['venue_status']}." + (f" {permit[0].upper() + permit[1:]}." if permit else ""),
            f"Expected reach: {self._reach(f)} ({self._access(f, st)}).",
            f["pay_note"] or "",
        ])
        return "\n".join([
            f"Subject: {fund} {f['round']}: {f['ref']} for assessment",
            f"From: {f['officer']}", f"To: {f['assessor']}", "",
            f"Hi {f['assessor'].split()[0]},", "",
            "Next one for you. The guidelines, as a reminder:", bullets(self._rules_list(f, st), rng), "",
            noisy(facts, rng, 0.1), "", "Budget as submitted:", btable, "",
            f"Thanks, {f['officer'].split()[0]}",
        ])

    def questions(self):
        return {
            "arts_award_decision": choice("What should the assessor recommend for this application?", {
                "award_in_full": "Award the full amount requested.",
                "award_reduced": "Award a smaller sum that covers the core activity but not every cost requested.",
                "award_conditional": "Offer the grant, with payment held until named conditions such as a permit or a "
                                     "confirmed venue are met.",
                "invite_reapplication": "Decline this round and invite the group to revise and reapply in the next one.",
                "decline": "Decline the application outright.",
            }),
            "community_benefit": score("How much public benefit would the project deliver for the money requested?", [
                "Little: open only to members and their guests, or only a handful of people would take part or attend.",
                "Some: a small turnout for this kind of project, or entry by paid ticket.",
                "Good: a typical local turnout, free or pay-what-you-can.",
                "High: an unusually wide turnout, or a lasting public work, free to all.",
            ]),
            "delivery_readiness": score("How ready is the group to deliver the project as described, on the dates "
                                        "given?", [
                "Not ready: the venue, permission or people are missing, and time is short.",
                "Shaky: important arrangements are still provisional.",
                "Mostly ready: minor arrangements remain, with enough time to finish them.",
                "Ready: venue, people and permissions are all in place.",
            ]),
            "budget_rules_met": noul(
                "The budget stays within every spending rule in the fund's guidelines.",
                yes="The request, artist pay, hospitality share and any match-funding requirement all sit within the "
                    "stated limits.",
                no="At least one budget figure breaks a stated limit or falls short of a stated minimum."),
            "permissions_in_place": noul("The venue and any permits the activity needs are secured, or will clearly be "
                                         "secured before it starts."),
        }


# =========================================================================== schema 6
STATEMENTS = ["I am confident I can assess this application fairly.", "Happy to step out if the chair prefers.",
              "I have had no contact with the PI for some time."]


class PanelConflictCheck(Schema):
    name = "panel_conflict_check"
    description = ("A panel member's conflict-of-interest position on one application, checked against the "
                   "funder's written policy before the meeting.")
    surfaces = {"json": 0.4, "declaration_form": 0.25, "secretariat_email": 0.2, "coi_register_export": 0.15}

    def sample_factors(self, rng):
        reg = regime(rng)
        window = pick(rng, [36, 48, 60])
        cthresh = pick(rng, [10, 12, 15])
        rel = pick(rng, ["coauthor", "joint_grant", "consortium", "same_institution", "competitor", "advisory_board",
                         "former_supervisor", "same_department", "pending_application", "refereed_paper"])
        months_since = None
        consortium_size = None
        shared_centre = False
        declared = pick_w(rng, {"prompt": 3, "late": 3, "partial": 2, "undeclared": 1})
        role = pick_w(rng, {"lead reviewer": 3, "second reader": 3, "panel member": 2, "chair": 1})
        scored = role in ("lead reviewer", "second reader") and chance(rng, 0.55)
        score_vs = pick(rng, ["in_line", "in_line", "higher", "lower"])
        only_expert = chance(rng, 0.25)
        statement = maybe(rng, pick(rng, STATEMENTS), 0.5)
        pattern = None
        if reg == "clear":
            if chance(rng, 0.5):
                rel = pick(rng, ["refereed_paper", "advisory_board", "refereed_paper"])
                declared, score_vs = "prompt", "in_line"
            else:
                rel = pick(rng, ["same_department", "former_supervisor", "pending_application", "personal_friend",
                                 "coauthor"])
                declared, scored = "prompt", False
                if rel == "coauthor":
                    months_since = rng.randint(3, window // 3)
        elif reg == "borderline":
            rel = pick(rng, ["coauthor", "joint_grant", "consortium", "same_institution", "same_institution",
                             "competitor"])
            if rel in ("coauthor", "joint_grant"):
                months_since = max(1, window + pick(rng, [-6, -4, -2, -1, 1, 2, 3, 5]))
            elif rel == "consortium":
                consortium_size = max(3, cthresh + pick(rng, [-3, -2, -1, 0, 1, 2]))
            elif rel == "same_institution":
                shared_centre = chance(rng, 0.6)
            declared = pick_w(rng, {"prompt": 3, "late": 2, "partial": 1})
        else:
            pattern = pick(rng, ["hidden_major", "late_minor_rival", "scarce_expert"])
            if pattern == "hidden_major":
                rel = pick(rng, ["coauthor", "former_supervisor", "pending_application", "joint_grant"])
                if rel in ("coauthor", "joint_grant"):
                    months_since = rng.randint(4, window - 6)
                declared = pick(rng, ["undeclared", "partial"])
                role, scored, score_vs = pick(rng, ["lead reviewer", "second reader"]), True, "higher"
                statement = STATEMENTS[0]
            elif pattern == "late_minor_rival":
                rel = pick(rng, ["competitor", "same_institution"])
                declared = "late"
                role, scored, score_vs = pick(rng, ["lead reviewer", "second reader"]), True, pick(rng, ["lower", "higher"])
            else:
                rel = pick(rng, ["coauthor", "same_institution", "consortium", "joint_grant"])
                if rel in ("coauthor", "joint_grant"):
                    months_since = max(1, window + pick(rng, [-8, -3, 2, 6]))
                if rel == "consortium":
                    consortium_size = cthresh + pick(rng, [-2, -1, 1, 3])
                only_expert, declared = True, "prompt"
                statement = STATEMENTS[0]
        if rel in ("coauthor", "joint_grant") and months_since is None:
            months_since = rng.randint(6, window * 2)
        if rel == "consortium" and consortium_size is None:
            consortium_size = rng.randint(4, 40)
        if rel in ("advisory_board", "refereed_paper") and declared in ("partial", "undeclared"):
            declared = "prompt"
        if not scored:
            score_vs = None
        title, code, _ = _topic(rng)
        area, depts = AREAS[code]
        pi_inst = org(rng, "school")
        mem_inst = pi_inst
        if rel not in ("same_department", "same_institution"):
            while mem_inst == pi_inst:
                mem_inst = org(rng, "school")
        pi_dept = pick(rng, depts)
        mem_dept = pi_dept if rel == "same_department" else pick(rng, [d for d in depts if d != pi_dept])
        meeting = rand_date(rng)
        sc = pick(rng, [6, 6, 7, 5])
        base = rng.randint(max(2, sc - 3), sc - 1)
        member_score = {"in_line": base, "higher": min(sc, base + 2), "lower": max(1, base - 2), None: None}[score_vs]
        return {
            "regime": reg, "pattern": pattern, "relationship": rel, "months_since": months_since,
            "papers_together": rng.randint(1, 4), "consortium_size": consortium_size, "shared_centre": shared_centre,
            "supervision_years_ago": rng.randint(4, 18), "window_months": window, "consortium_threshold": cthresh,
            "declared": declared, "role": role, "scored": scored, "score_vs_others": score_vs,
            "member_score": member_score, "other_scores": [base + d for d in pick(rng, [(0, 0), (0, -1), (1, 0)])],
            "score_scale": sc, "only_expert": only_expert, "statement": statement,
            "other_conflicts_this_round": pick(rng, [0, 0, 1, 1, 2, 4]),
            "apps_from_member_institution": pick(rng, [0, 1, 2, 4]),
            "member": person(rng), "member_inst": mem_inst, "member_dept": mem_dept, "pi": person(rng),
            "pi_inst": pi_inst, "pi_dept": pi_dept, "meeting": meeting,
            "declared_on": meeting - timedelta(days=rng.randint(35, 70)),
            "late_declared_on": meeting - timedelta(days=rng.randint(3, 12)),
            "app_ref": ident(rng, "APP", 5, "-"), "title": title, "call_code": code,
            "panel": f"{area} Panel {pick(rng, ['A', 'B', 'C'])}", "funder": _funder(rng),
            "secretary": person(rng), "chair": person(rng),
            "consortium": f"{rng.choice(ORG_A)} {pick(rng, ['Network', 'Consortium', 'Alliance'])}",
        }

    # ------------------------------------------------------------------ text pieces
    def _fact(self, f):
        """What links the member to the PI, phrased to follow 'the member'."""
        rel, m = f["relationship"], f["months_since"]
        k = f["papers_together"]
        return {
            "coauthor": f"co-authored {_pl(k, 'paper')} with the PI; the most recent appeared {m} months before the "
                        "meeting",
            "joint_grant": f"was a co-investigator with the PI on a grant that ended {m} months before the meeting",
            "pending_application": "is named with the PI on an application now under review at another funder",
            "same_department": f"works in the same department as the PI ({f['pi_dept']}, {f['pi_inst']})",
            "same_institution": f"works at the PI's university in a different department ({f['member_dept']}; the PI "
                                f"is in {f['pi_dept']})" + ("; both belong to the university's cross-department "
                                                            "research centre" if f["shared_centre"] else ""),
            "former_supervisor": f"supervised the PI's doctorate, completed {f['supervision_years_ago']} years ago",
            "consortium": f"belongs, like the PI's group, to the {f['consortium']}, which has "
                          f"{f['consortium_size']} partner institutions",
            "advisory_board": "sits with the PI on the advisory board of a national charity",
            "competitor": "leads a group working on the same question; the two groups have published rival results",
            "refereed_paper": "refereed one of the PI's papers for a journal last year (anonymous review)",
            "personal_friend": "is a close personal friend of the PI",
        }[rel]

    def _understated(self, f):
        return {"coauthor": "occasional contact with the PI at conferences",
                "joint_grant": "worked with the PI some years ago",
                "pending_application": "some informal discussions with the PI about future work",
                "former_supervisor": "knows the PI from earlier in the PI's career",
                "same_institution": "same university", "competitor": "works in a related area",
                "consortium": "loose network membership", "same_department": "same university",
                "personal_friend": "knows the PI socially"}.get(f["relationship"], "no relevant links")

    def _declaration(self, f):
        d = f["declared"]
        if d == "prompt":
            return {"status": "declared when applications were assigned", "date": f["declared_on"],
                    "text": f"The member {self._fact(f)}."}
        if d == "late":
            return {"status": "declared after the member's assessment was submitted" if f["scored"]
                    else "declared a few days before the meeting", "date": f["late_declared_on"],
                    "text": f"The member {self._fact(f)}."}
        if d == "partial":
            return {"status": "declared when applications were assigned", "date": f["declared_on"],
                    "text": f"Declared: {self._understated(f)}."}
        return {"status": "nothing declared", "date": None, "text": None}

    def _check(self, f):
        if f["declared"] in ("prompt", "late"):
            return "Nothing found beyond the declaration."
        src = "The secretariat's publication and grants search" if f["relationship"] in (
            "coauthor", "joint_grant", "pending_application") else "Another panel member"
        return f"{src} shows that the member {self._fact(f)}."

    def _policy(self, f):
        w, c = f["window_months"], f["consortium_threshold"]
        return {
            "withdraw_if": ["same department or research unit as the PI",
                            f"a joint publication or joint grant with the PI in the last {w} months",
                            "a joint application with the PI under review anywhere",
                            "doctoral supervisor or student of the PI, at any time",
                            "a close personal relationship or a financial interest",
                            f"membership of the same consortium where it has fewer than {c} partners"],
            "declare_and_chair_decides": ["same institution, different department",
                                          f"collaboration that ended more than {w} months ago",
                                          "shared advisory, editorial or committee roles",
                                          "direct competition in the same research area",
                                          f"the same consortium where it has {c} or more partners"],
        }

    def _scores(self, f):
        if not f["scored"]:
            return None
        return (f"{f['member_score']}/{f['score_scale']} (other assessors: "
                f"{', '.join(str(s) for s in f['other_scores'])})")

    def _obj(self, f):
        dec = self._declaration(f)
        return prune({
            "panel": {"name": f["panel"], "funder": f["funder"], "meeting": f["meeting"].isoformat(),
                      "chair": f["chair"] if f["role"] != "chair" else None},
            "member": {"name": f["member"], "institution": f["member_inst"], "department": f["member_dept"],
                       "role_on_this_application": f["role"], "assessment_submitted": f["scored"],
                       "score_given": self._scores(f), "statement": f["statement"],
                       "only_panel_member_with_this_expertise": f["only_expert"] or None,
                       "other_conflicts_declared_this_round": f["other_conflicts_this_round"],
                       "applications_from_own_institution_this_round": f["apps_from_member_institution"]},
            "application": {"ref": f["app_ref"], "title": f["title"], "pi": f["pi"], "institution": f["pi_inst"],
                            "department": f["pi_dept"]},
            "declaration": {"status": dec["status"], "date": dec["date"].isoformat() if dec["date"] else None,
                            "text": dec["text"]},
            "secretariat_check": self._check(f),
            "policy": self._policy(f),
        })

    def render(self, f, rng, surface):
        if surface == "json":
            return dumps(self._obj(f))
        if surface == "coi_register_export":
            return dumps_pretty(self._obj(f), rng)
        dec = self._declaration(f)
        pol = self._policy(f)
        if surface == "declaration_form":
            parts = [
                f"CONFLICT OF INTEREST DECLARATION | {f['funder']} | {f['panel']}",
                kv_lines([("Panel member", f"{f['member']} ({f['member_dept']}, {f['member_inst']})"),
                          ("Application", f"{f['app_ref']}: {f['title']}"),
                          ("PI", f"{f['pi']} ({f['pi_dept']}, {f['pi_inst']})"),
                          ("Role on application", f["role"]),
                          ("Assessment submitted", "yes, score " + self._scores(f) if f["scored"] else "no"),
                          ("Declaration status", dec["status"]),
                          ("Date declared", _date(dec["date"], rng) if dec["date"] else None),
                          ("Member's entry", dec["text"] or "(blank)"),
                          ("Member's comment", f["statement"]),
                          ("Meeting date", _date(f["meeting"], rng))], rng),
                "", "Secretariat check: " + self._check(f)]
            if f["only_expert"]:
                parts.append("Note: the member is the only panellist with expertise in this area.")
            parts += [f"Other conflicts the member has declared this round: {f['other_conflicts_this_round']}; "
                      f"applications from the member's own institution this round: {f['apps_from_member_institution']}.",
                      "", "Policy: the member must withdraw where there is", bullets(pol["withdraw_if"], rng),
                      "The member declares and the chair decides where there is",
                      bullets(pol["declare_and_chair_decides"], rng)]
            return "\n".join(parts)
        # secretariat_email
        to = "Deputy chair" if f["role"] == "chair" else f["chair"]
        greet = "Hi," if f["role"] == "chair" else f"Hi {f['chair'].split()[0]},"
        role_txt = {"lead reviewer": "is lead reviewer for this application",
                    "second reader": "is second reader for this application",
                    "panel member": "sits on the panel but has no assignment for this application",
                    "chair": "chairs the panel, so this comes to you as deputy chair"}[f["role"]]
        n_other = f["other_conflicts_this_round"]
        paras = [
            f"A conflict question for {f['app_ref']} ({f['pi']}, {f['pi_dept']}, {f['pi_inst']}) before the "
            f"{f['panel']} meeting on {_date(f['meeting'], rng)}.",
            f"{f['member']} ({f['member_dept']}, {f['member_inst']}) {role_txt}.",
            {"prompt": f"Their declaration, dated {_date(f['declared_on'], rng)}, says the member {self._fact(f)}.",
             "late": f"On {_date(f['late_declared_on'], rng)} they told us that the member {self._fact(f)}.",
             "partial": f"Their declaration mentions only \"{self._understated(f)}\". {self._check(f)}",
             "undeclared": f"They declared nothing. {self._check(f)}"}[f["declared"]],
            (f"Their assessment is already in: {self._scores(f)}." if f["scored"] else "They have not scored it."),
            (f"They add: \"{f['statement']}\"" if f["statement"] else ""),
            ("They are the only panellist with expertise in this area." if f["only_expert"] else ""),
            (f"They have {_pl(n_other, 'other declared conflict')} this round." if n_other else ""),
        ]
        return "\n".join([
            f"Subject: COI query, {f['app_ref']} / {f['member']}", f"From: {f['secretary']} (panel secretariat)",
            f"To: {to}", "", greet, "", noisy(join_sentences(paras), rng, 0.08), "",
            f"For reference, our policy: withdrawal is required for {'; '.join(pol['withdraw_if'])}. A declaration "
            f"with a chair's decision covers {'; '.join(pol['declare_and_chair_decides'])}.", "",
            "Thanks,", f["secretary"].split()[0],
        ])

    def questions(self):
        return {
            "recusal_required": noul(
                "The panel member must be kept out of the assessment and discussion of this application.",
                yes="The link is one the policy lists as requiring withdrawal, or a fair observer would see it as "
                    "disqualifying.",
                no="The link is absent, remote, or of a kind the policy handles with a declaration alone."),
            "conflict_severity": score("How serious is the member's link to this application under the funder's "
                                       "policy?", [
                "None: no relationship the policy recognises.",
                "Minor: a remote or indirect link that only needs recording.",
                "Significant: a close link that calls for the chair's ruling.",
                "Disqualifying: a relationship the policy says requires withdrawal.",
            ]),
            "declaration_quality": score("How complete and timely was the member's own declaration of the link?", [
                "Undisclosed: the member declared nothing; the secretariat or another panellist found the link.",
                "Inadequate: declared only after the member's own assessment was in, or the declaration understates "
                "the link the secretariat found.",
                "Adequate: the full link was declared late, but before the member had submitted any assessment.",
                "Full: the full link was declared when applications were assigned.",
            ]),
            "conflict_handling": choice("How should the secretariat handle this member for this application?", {
                "take_part_normally": "Let the member assess and discuss the application as usual.",
                "take_part_with_record": "Let the member take part, with the link written into the meeting record.",
                "leave_for_item": "Have the member leave the room and give no score for this application only.",
                "withdraw_from_institution": "Keep the member away from every application from that institution "
                                             "this round.",
                "stand_down_from_round": "Stand the member down from the whole funding round.",
            }),
            "submitted_scores_treatment": choice("What should happen to any assessment this member has already "
                                                 "given?", {
                "no_action": "Nothing needs to change: no assessment has been given, or none is affected.",
                "keep_with_note": "Keep the assessment, with a note of the relationship placed beside it.",
                "replace_assessment": "Strike the member's assessment of this application and get one from another "
                                      "reviewer.",
                "recheck_ranking": "Strike it and re-check the ranking of every application the member's input may "
                                   "have shaped.",
            }),
        }


WORKFLOW = Workflow(
    name="grant_review",
    title="Grant review",
    description="How a funding body screens, scores and manages research and community grants: eligibility "
                "screens, panel reconciliation, continuation reviews, award variations, small arts grants and "
                "panel conflicts of interest.",
    domain="Research funding / grants administration",
    schemas=[ProposalFirstScreen(), PanelScoreReconciliation(), AwardProgressContinuation(),
             ExtensionReallocationRequest(), CommunityArtsApplication(), PanelConflictCheck()],
    notes="Fictional funders, institutions and applicants. The call's or fund's own rules are always in the state. "
          "Decisions turn on eligibility rules, evidence, budgets and relationships, never on a protected attribute. "
          "No vendor bills are reconciled here: budgets are applicants' plans and award spend, not invoices.",
)
