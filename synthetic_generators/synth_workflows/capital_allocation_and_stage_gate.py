"""Capital allocation and stage-gate decisions inside a (fictional) diversified manufacturer.

Five schemas: a development project at a stage gate judged against the gate's written must-meet criteria; a capital
expenditure request judged against the hurdle-rate, payback, contingency and delegation rules; one project's fate in a
portfolio reprioritisation after a budget cut; a make-versus-buy sourcing request; and a post-investment review of a
project that missed its approved case. Every state quotes only the one to three clauses of the (invented) governance
manual that decide the case, with thresholds sampled per case, so each answer comes from comparing the facts with a
rule: an IRR half a point under the hurdle, a scorecard one point short, a saving that vanishes once the overhead that
stays behind is taken out.

Cash flows are generated first and NPV, IRR and payback are computed from them, so the figures in a state are
consistent with each other. `intended(f)` records the answer logic a careful reviewer would apply to the rendered
facts; the builder never uses it (it exists for the spread audit). Companies, documents and people are invented.
"""
from __future__ import annotations

import math

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


# =========================================================================== finance helpers
def _npv(rate, flows):
    return sum(cf / (1.0 + rate) ** t for t, cf in enumerate(flows))


def _irr(flows):
    lo, hi = -0.9, 2.0
    f_lo = _npv(lo, flows)
    if f_lo * _npv(hi, flows) > 0:
        return None
    for _ in range(80):
        mid = (lo + hi) / 2
        fm = _npv(mid, flows)
        if f_lo * fm <= 0:
            hi = mid
        else:
            lo, f_lo = mid, fm
    return (lo + hi) / 2


def _payback(flows):
    cum = flows[0]
    for t in range(1, len(flows)):
        if flows[t] > 0 and cum + flows[t] >= 0:
            return round(t - 1 + (-cum) / flows[t], 1)
        cum += flows[t]
    return None


def _annuity(rate, profile):
    return sum(p / (1.0 + rate) ** (t + 1) for t, p in enumerate(profile))


def _usd(x, style="m"):
    """Dollar amounts in the house styles finance decks use."""
    if x is None:
        return "n/a"
    neg = "-" if x < 0 else ""
    a = abs(x)
    if style == "full":
        return f"{neg}${a:,.0f}"
    if style == "code":
        return f"{neg}USD {a / 1e6:.2f}m" if a >= 1e6 else f"{neg}USD {a / 1e3:,.0f}k"
    if a >= 1e6:
        return f"{neg}${a / 1e6:.2f}M" if a < 1e7 else f"{neg}${a / 1e6:.1f}M"
    return f"{neg}${a / 1e3:,.0f}k"


def _pc(x, d=1):
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


def _doc(rng, prefix):
    return f"{prefix}-{rng.randint(2, 9)}"


def _clamp(x, lo=0, hi=3):
    return max(lo, min(hi, x))


BUS = ["Specialty Coatings", "Industrial Pumps", "Consumer Appliances", "Packaging Films", "Medical Consumables",
       "Agricultural Equipment", "Building Products", "Water Treatment", "Electrical Components", "Food Ingredients",
       "Precision Bearings", "Fluid Connectors"]
SITES = ["Dayton plant", "Monterrey plant", "Leeds works", "Ostrava site", "Greenville plant", "Pune plant",
         "Ballarat works", "Rotterdam site", "Joliet plant", "Sherbrooke plant", "Tilburg site", "Saltillo plant",
         "Wichita plant", "Kraków site", "Chattanooga plant"]


# =========================================================================== schema 1: stage gate
# (business unit, project, compliance item the gate requires)
PROJECTS = [
    ("Specialty Coatings", "low-VOC marine primer", "regulatory registration plan"),
    ("Specialty Coatings", "graphene-filled anti-corrosion topcoat", "freedom-to-operate opinion"),
    ("Consumer Appliances", "compact heat-pump tumble dryer", "safety certification plan"),
    ("Consumer Appliances", "induction hob with app control", "radio/EMC certification plan"),
    ("Packaging Films", "recyclable mono-material stand-up pouch", "food-contact compliance file"),
    ("Packaging Films", "compostable coffee-capsule lidding", "food-contact compliance file"),
    ("Medical Consumables", "single-use catheter tray kit", "regulatory pathway memo"),
    ("Medical Consumables", "antimicrobial foam wound dressing", "regulatory pathway memo"),
    ("Industrial Pumps", "variable-speed booster pump skid", "pressure-equipment conformity plan"),
    ("Industrial Pumps", "magnetic-drive chemical transfer pump", "freedom-to-operate opinion"),
    ("Agricultural Equipment", "precision seed-metering retrofit kit", "freedom-to-operate opinion"),
    ("Agricultural Equipment", "autonomous orchard sprayer", "machinery safety assessment"),
    ("Building Products", "fire-rated mass-timber connector", "fire-test certification plan"),
    ("Building Products", "recycled-content roofing membrane", "product certification plan"),
    ("Water Treatment", "modular UV disinfection unit", "drinking-water approval plan"),
    ("Water Treatment", "smart leak-detection valve", "radio/EMC certification plan"),
    ("Electrical Components", "solid-state circuit breaker", "safety certification plan"),
    ("Electrical Components", "11 kW wall-charger power module", "radio/EMC certification plan"),
    ("Food Ingredients", "plant-based egg replacer", "novel-food dossier plan"),
    ("Food Ingredients", "reduced-sugar fruit filling", "labelling compliance review"),
    ("Precision Bearings", "ceramic hybrid spindle bearing", "freedom-to-operate opinion"),
    ("Fluid Connectors", "push-to-connect hydrogen fitting", "pressure-equipment conformity plan"),
]
# gate -> (label, TRL range for the must-meet, customer-evidence label, its count range)
GATES = {
    2: ("Gate 2 (build business case)", (3, 4), "customer interviews confirming the need", (8, 15)),
    3: ("Gate 3 (go to development)", (5, 6), "signed LOIs or pilot agreements", (3, 6)),
    4: ("Gate 4 (go to launch)", (7, 7), "field trials with customer sign-off", (2, 4)),
}
GATE_ITEMS = ("technical", "customer", "financial", "cost", "compliance")
RAMP = [0.25, 0.6, 0.9, 1.0, 1.0, 1.0, 0.85]
FIX_TEXT = {
    "technical": "next TRL demo booked in {w} weeks",
    "customer": "further commitments in negotiation, expected within {w} weeks",
    "financial": "value-engineering review due in {w} weeks",
    "cost": "supplier re-quote due in {w} weeks",
    "compliance": "due in {w} weeks",
}


class StageGateReview(Schema):
    name = "stage_gate_review"
    description = "A development project at a stage gate, judged against the gate's written must-meet criteria: go, conditional go, recycle, hold or kill."
    surfaces = {"json": 0.5, "gate_minutes": 0.25, "gate_scorecard": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng, clear=0.25, borderline=0.45, conflicting=0.30)
        bu, proj, comp_item = pick(rng, PROJECTS)
        gate = pick(rng, [2, 3, 3, 4])
        glabel, trl_rng, voc_label, voc_rng = GATES[gate]
        trl_min, voc_min = rng.randint(*trl_rng), rng.randint(*voc_rng)
        hurdle, tol = pick(rng, [9, 10, 11, 12, 13, 14]), pick(rng, [5, 8, 10, 12, 15])
        smin, fix_limit = pick(rng, [55, 60, 65, 70]), pick(rng, [4, 6, 8])
        st = {k: "ok" for k in GATE_ITEMS}
        score_st, funded, recycles = "ok", chance(rng, 0.85), pick(rng, [0, 0, 0, 0, 1])
        if reg == "clear":
            if chance(rng, 0.55):
                score_st = pick(rng, ["ok", "ok", "ok", "fail"])
            else:
                for k in rng.sample(GATE_ITEMS, pick(rng, [2, 3])):
                    st[k] = "fail"
                score_st = pick(rng, ["ok", "fail"])
        elif reg == "borderline":
            allpass = chance(rng, 0.3)
            for k in rng.sample(GATE_ITEMS, pick(rng, [1, 1, 1, 2, 2, 3])):
                st[k] = "near_ok" if allpass else pick(rng, ["near_ok", "near_fail", "near_fail"])
            score_st = pick(rng, ["ok", "ok", "near_ok", "near_fail"])
            if chance(rng, 0.1):
                st[pick(rng, ["customer", "cost", "compliance"])] = "missing"
        else:
            twist = pick_w(rng, {"far_fail": 5, "scorecard": 2, "unfunded": 1.5, "recycled": 1.5})
            if twist == "far_fail":
                k = pick(rng, GATE_ITEMS)
                st[k] = pick(rng, ["fail", "fail", "near_fail"] + (["missing"] if k in ("customer", "cost", "compliance") else []))
                for j in rng.sample(GATE_ITEMS, 2):
                    if j != k:
                        st[j] = "near_ok"
                score_st = pick(rng, ["ok", "ok", "near_ok", "fail"])
            elif twist == "scorecard":
                for j in rng.sample(GATE_ITEMS, 2):
                    st[j] = "near_ok"
                score_st = "fail"
            elif twist == "unfunded":
                funded, score_st = False, "ok"
            else:
                st[pick(rng, GATE_ITEMS)] = "near_fail"
                recycles = 1
        # values
        trl = {"ok": min(9, trl_min + pick(rng, [1, 1, 2])), "near_ok": trl_min, "near_fail": trl_min - 1,
               "fail": max(1, trl_min - 2)}[st["technical"]]
        voc = {"ok": math.ceil(voc_min * rng.uniform(1.3, 2.0)), "near_ok": voc_min + pick(rng, [0, 0, 1]),
               "near_fail": voc_min - 1, "fail": max(0, round(voc_min * rng.uniform(0.2, 0.5))), "missing": None}[st["customer"]]
        pi = {"ok": rng.uniform(0.25, 1.3), "near_ok": rng.uniform(0.01, 0.08), "near_fail": rng.uniform(-0.07, -0.01),
              "fail": rng.uniform(-0.45, -0.13)}[st["financial"]]
        gap = {"ok": tol - rng.uniform(3, 12), "near_ok": tol - rng.uniform(0, 1.5), "near_fail": tol + rng.uniform(0.5, 3),
               "fail": tol + rng.uniform(6, 20), "missing": None}[st["cost"]]
        comp = {"ok": "complete", "near_ok": "complete", "near_fail": "in progress", "fail": "not started",
                "missing": None}[st["compliance"]]
        points = {"ok": smin + rng.randint(4, 24), "near_ok": smin + rng.randint(0, 2), "near_fail": smin - rng.randint(1, 4),
                  "fail": smin - rng.randint(8, 20)}[score_st]
        invest = round(math.exp(rng.gauss(math.log(4.5e6), 0.55)), -5)
        peak = invest * (1 + pi) / _annuity(hurdle / 100, RAMP)
        flows = [-invest] + [round(peak * p, -3) for p in RAMP]
        fix_weeks = (near(rng, fix_limit, 3, lo=1, integer=True) if reg != "clear" else rng.randint(2, 12))
        return {
            "regime": reg, "bu": bu, "project": proj, "compliance_item": comp_item, "gate": gate, "gate_label": glabel,
            "trl_min": trl_min, "voc_min": voc_min, "voc_label": voc_label, "hurdle": hurdle, "cost_tol": tol,
            "score_min": smin, "fix_limit": fix_limit, "status": st, "score_status": score_st,
            "trl": trl, "voc": voc, "investment": invest, "flows": flows,
            "npv": round(_npv(hurdle / 100, flows), -3), "irr": _irr(flows), "cost_gap_pct": None if gap is None else round(gap, 1),
            "compliance_status": comp, "points": points, "fix_weeks": fix_weeks, "funded": funded,
            "prior_recycles": recycles, "champion": person(rng), "chair": person(rng),
            "tone": "upbeat" if reg == "conflicting" or chance(rng, 0.3) else pick(rng, ["candid", "neutral"]),
            "doc": _doc(rng, "SGC"), "date": rand_date(rng), "company": org(rng),
        }

    # ------------------------------------------------------------------ derived
    @staticmethod
    def _fails(f):
        return [k for k in GATE_ITEMS if f["status"][k] in ("near_fail", "fail", "missing")]

    def intended(self, f):
        st, fails = f["status"], self._fails(f)
        kill_npv = f["npv"] < -0.10 * f["investment"]
        score_ok = f["points"] >= f["score_min"]
        if kill_npv or (fails and f["prior_recycles"] >= 1):
            dec = "kill"
        elif not fails:
            dec = "go" if score_ok else "recycle"
        elif len(fails) == 1 and st[fails[0]] == "near_fail" and f["fix_weeks"] <= f["fix_limit"] and score_ok:
            dec = "go_with_conditions"
        else:
            dec = "recycle"
        if dec in ("go", "go_with_conditions") and not f["funded"]:
            dec = "hold"
        rank = {"fail": 0, "missing": 1, "near_fail": 2}
        if fails:
            worst = sorted(fails, key=lambda k: (rank[st[k]], GATE_ITEMS.index(k)))[0]
            gap = {"technical": "technical_readiness", "customer": "customer_evidence", "financial": "financial_case",
                   "cost": "cost_or_compliance", "compliance": "cost_or_compliance"}[worst]
        else:
            gap = "no_must_meet_gap"
        n_fail, n_near_ok = len(fails), sum(st[k] == "near_ok" for k in GATE_ITEMS)
        far = sum(st[k] in ("fail", "missing") for k in fails)
        if n_fail >= 3 or (n_fail == 2 and far == 2):
            mat = 0
        elif n_fail == 2 or far == 1:
            mat = 1
        elif n_fail == 1 or n_near_ok >= 2:
            mat = 2
        else:
            mat = 3
        pi = f["npv"] / f["investment"]
        att = 0 if pi < 0 else 1 if pi < 0.2 else 2 if pi < 0.5 else 3 if pi < 1.0 else 4
        if f["cost_gap_pct"] is not None and f["cost_gap_pct"] > f["cost_tol"] + 5:
            att -= 1
        return {"gate_decision": dec, "binding_gap": gap, "evidence_maturity": mat, "commercial_attractiveness": _clamp(att, 0, 4),
                "must_meets_satisfied": "no" if fails else "yes"}

    # ------------------------------------------------------------------ rendering
    def _rules(self, f, rng, short=False):
        a = (f"{f['doc']} §4.2, {f['gate_label']} must-meets: TRL >= {f['trl_min']}; >= {f['voc_min']} {f['voc_label']}; "
             f"NPV > 0 at {f['hurdle']}%; unit cost <= target +{f['cost_tol']}%; {f['compliance_item']} complete.")
        b = (f"§4.4: Go = all must-meets + scorecard >= {f['score_min']}/100; one near-miss closable within "
             f"{f['fix_limit']} weeks = conditional Go; else Recycle.")
        c = "§4.6: Kill if NPV < -10% of investment or a must-meet fails again after a Recycle; passing but unfunded = Hold."
        if short:
            short_voc = {2: "interviews", 3: "LOIs/pilots", 4: "accepted field trials"}[f["gate"]]
            return (f"Must-meets ({f['doc']} §4.2): TRL>={f['trl_min']} | >={f['voc_min']} {short_voc} "
                    f"| NPV>0 @{f['hurdle']}% | cost <= target+{f['cost_tol']}% | "
                    f"{f['compliance_item']}. Go: all + score >= {f['score_min']}; one near-miss closable in <= "
                    f"{f['fix_limit']} wks -> conditional (§4.4). Kill: NPV < -10% of spend, or a repeat fail after Recycle; "
                    f"funded team needed or Hold (§4.6).")
        return [a, b, c]

    def _evidence(self, f, rng):
        st = f["status"]
        fw = f["fix_weeks"]
        ev = {"trl": f"TRL {f['trl']}" + (f" ({FIX_TEXT['technical'].format(w=fw)})" if st["technical"] == "near_fail" else ""),
              "customer": (f"{f['voc']} {f['voc_label'].split(' confirming')[0].split(' with')[0]}" if f["voc"] is not None
                           else pick(rng, ["summaries not attached", "count not reported"]))
              + (f"; {FIX_TEXT['customer'].format(w=fw)}" if st["customer"] == "near_fail" else ""),
              "cost": (f"{f['cost_gap_pct']:+.1f}% vs target" if f["cost_gap_pct"] is not None
                       else pick(rng, ["pending supplier quotes", "TBD"]))
              + (f" ({FIX_TEXT['cost'].format(w=fw)})" if st["cost"] == "near_fail" else ""),
              "compliance": (f["compliance_status"] if f["compliance_status"] else pick(rng, ["not reported", "-"]))
              + (f", {FIX_TEXT['compliance'].format(w=fw)}" if st["compliance"] == "near_fail" else "")}
        fin = f"invest {_usd(f['investment'])}; NPV @ {f['hurdle']}% {_usd(f['npv'])}; IRR {_pc(f['irr'])}"
        if st["financial"] == "near_fail":
            fin += f" ({FIX_TEXT['financial'].format(w=fw)})"
        ev["financial"] = fin
        return ev

    def _champion(self, f, rng):
        if f["tone"] == "upbeat":
            s = pick(rng, ["Customers love it and the window is now; the open points are paperwork.",
                           "The team is confident this is ready to move forward.",
                           "Every month we wait hands the slot to a competitor; we see no real blockers."])
        elif f["tone"] == "candid":
            s = pick(rng, ["We know some items are still soft and would accept conditions.",
                           "Not everything is closed; we have plans for the gaps.",
                           "We'd rather be told now if this isn't good enough."])
        else:
            s = pick(rng, ["Submitted for the scheduled gate.", "Evidence pack as attached.", ""])
        return noisy(s, rng, 0.15) if s else ""

    def render(self, f, rng, surface):
        ev = self._evidence(f, rng)
        team = "funded team in place" if f["funded"] else pick(rng, ["no development team free until next quarter",
                                                                     "team not yet funded in this year's plan"])
        if surface == "json":
            return dumps(prune({
                "project": {"name": f["project"], "business_unit": f["bu"], "gate": f["gate_label"], "champion": f["champion"],
                            "recycles_at_this_gate": f["prior_recycles"], "resourcing": team},
                "evidence": {"technology_readiness": ev["trl"], "customer_evidence": ev["customer"],
                             "financials": ev["financial"], "unit_cost": ev["cost"], f["compliance_item"].replace(" ", "_"): ev["compliance"],
                             "scorecard": f"{f['points']}/100"},
                "champion_note": self._champion(f, rng) or None,
                "criteria": self._rules(f, rng)[: 3 if (f["prior_recycles"] or not f["funded"] or f["npv"] < 0 or chance(rng, 0.4)) else 2],
            }))
        if surface == "gate_minutes":
            lines = [f"{f['gate_label'].split(' (')[0].upper()} REVIEW - {f['project']} ({f['bu']})",
                     f"Date: {fmt_date(f['date'], rng)}   Chair: {f['chair']}   Champion: {f['champion']}",
                     f"Recycles at this gate so far: {f['prior_recycles']}   Resourcing: {team}", "", "Evidence presented:",
                     bullets([f"Technology: {ev['trl']}", f"Customer: {ev['customer']}", f"Financials: {ev['financial']}",
                              f"Unit cost: {ev['cost']}", f"{f['compliance_item'].capitalize()}: {ev['compliance']}",
                              f"Scorecard: {f['points']}/100"], rng)]
            ch = self._champion(f, rng)
            if ch:
                lines += ["", f"Champion: \"{ch}\""]
            return "\n".join(lines + ["", self._rules(f, rng, short=True)])
        rows = [["TRL", f">= {f['trl_min']}", ev["trl"]],
                ["Customer evidence", f">= {f['voc_min']}", ev["customer"]],
                [f"NPV @ {f['hurdle']}%", "> 0", ev["financial"]],
                ["Unit cost", f"<= target +{f['cost_tol']}%", ev["cost"]],
                [f["compliance_item"].capitalize(), "complete", ev["compliance"]],
                ["Scorecard", f">= {f['score_min']}", f"{f['points']}/100"]]
        head = (f"{f['company']} - {f['gate_label']} scorecard ({f['doc']})\nProject: {f['project']}, {f['bu']}; "
                f"champion {f['champion']}; recycles at this gate: {f['prior_recycles']}; {team}")
        tail = (f"Conditional Go: one near-miss closable in <= {f['fix_limit']} wks. Kill: NPV < -10% of investment or "
                "repeat failure after Recycle. Passing but unfunded: Hold.")
        return f"{head}\n\n{table(['must-meet', 'required', 'evidence'], rows, rng)}\n\n{tail}"

    def questions(self):
        return {
            "gate_decision": choice("What should the gate keepers decide at this review?", {
                "go": "Go: pass the gate and release the next stage's budget.",
                "go_with_conditions": "Conditional go: pass, with the open item to be closed by a set date.",
                "recycle": "Recycle: send the project back to rework its evidence and return to this gate.",
                "hold": "Hold: the project passes on merit but is parked until a funded team is available.",
                "kill": "Kill: stop the project and release its people and money.",
            }),
            "binding_gap": choice("Which gap most limits the project at this gate?", {
                "no_must_meet_gap": "None: every must-meet criterion is met.",
                "technical_readiness": "Technical readiness is short of what the gate requires.",
                "customer_evidence": "Customer evidence is thin or missing.",
                "financial_case": "The financial case does not clear the hurdle.",
                "cost_or_compliance": "Unit cost or the required compliance work is not where it needs to be.",
            }),
            "evidence_maturity": score("How mature is the project's evidence relative to what this gate requires?", [
                "Immature: three or more must-meet items unproven.",
                "Partial: two items short, or one well short.",
                "Nearly there: one item narrowly short, or several only just met.",
                "Mature: every must-meet demonstrated with room to spare.",
            ]),
            "commercial_attractiveness": score("How attractive is the project commercially on the figures presented?", [
                "Unattractive: it does not earn the hurdle rate.",
                "Marginal: it barely clears the hurdle.",
                "Fair: a modest margin over the hurdle.",
                "Solid: comfortably above the hurdle.",
                "Strong: value well in excess of the money it needs.",
            ]),
            "must_meets_satisfied": noul(
                "Every must-meet criterion for this gate is satisfied on the evidence presented.",
                yes="Each must-meet item is shown at or beyond its stated threshold.",
                no="At least one must-meet item falls short, is incomplete or is not evidenced."),
        }


# =========================================================================== schema 2: capex request
CAPEX = {
    "capacity_expansion": ["second film extrusion line", "additional five-axis CNC machining cell",
                           "third-shift packaging line", "warehouse mezzanine and racking", "new curing oven bank"],
    "cost_reduction": ["LED lighting retrofit", "VSD compressor and air-leak programme", "robotic palletiser on line 3",
                       "heat recovery on the curing ovens", "automated vision inspection replacing manual checks",
                       "solar carport and battery for peak shaving"],
    "replacement": ["replacement of the 1994 steam boiler", "forklift fleet replacement with electric trucks",
                    "rebuild of the 1,200-tonne hydraulic press", "replacement of the obsolete PLC control system"],
    "compliance": ["wastewater pretreatment upgrade for the new discharge permit",
                   "dust-collection upgrade for the combustible-dust standard",
                   "machine-guarding retrofit for the corrective-action order", "stack scrubber for the revised air permit"],
}
# benefit basis by category, strongest first
BASIS = {"capacity_expansion": ["contracted volumes", "market forecast", "aspirational volumes"],
         "cost_reduction": ["metered baseline", "engineering estimate", "vendor's savings claim"],
         "replacement": ["avoided maintenance history", "engineering estimate", "vendor's savings claim"],
         "compliance": ["permit requirement"]}
BASIS_PTS = {"contracted volumes": 1, "metered baseline": 1, "avoided maintenance history": 1, "permit requirement": 1,
             "engineering estimate": 0, "market forecast": 0, "vendor's savings claim": -1, "aspirational volumes": -1}
CAT_LABEL = {"capacity_expansion": "capacity expansion", "cost_reduction": "cost reduction",
             "replacement": "end-of-life replacement", "compliance": "regulatory compliance"}


class CapexRequestApproval(Schema):
    name = "capex_request_approval"
    description = "A capital expenditure request checked against the hurdle-rate, payback, contingency and approval-authority rules."
    surfaces = {"json": 0.5, "afe_form": 0.25, "fpa_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng, clear=0.25, borderline=0.45, conflicting=0.30)
        cat = pick_w(rng, {"capacity_expansion": 3, "cost_reduction": 4, "replacement": 2, "compliance": 1.3})
        hurdle = pick(rng, [9, 10, 11, 12, 13, 14])
        life = pick(rng, [5, 6, 7, 8, 10])
        ramp = [0.5, 0.9] + [1.0] * (life - 2) if cat == "capacity_expansion" else [1.0] * life
        # the payback limit is a round number close to the payback a project earning the hurdle would have
        pay_lim = max(2, round(_annuity((hurdle + pick(rng, [-3, -2, -1, 1])) / 100, ramp)))
        cont_min = pick(rng, [10, 10, 15])
        a1 = pick(rng, [250e3, 500e3, 750e3])
        a2 = pick(rng, [2e6, 3e6, 5e6])
        base_cost = round(math.exp(rng.gauss(math.log(1.6e6), 0.75)), -4)
        good = reg == "clear" and chance(rng, 0.55)
        if reg == "clear":
            tgt = rng.uniform(7, 15) if good else rng.uniform(-9, -4)
        elif reg == "borderline":
            tgt = near(rng, 1.0, 2.0)
        else:
            tgt = rng.uniform(-3, 9)
        cont = cont_min if good else pick_w(rng, {cont_min: 8, cont_min + 5: 2, 5: 1, 0: 0.6})
        amount = round(base_cost * (1 + cont / 100), -3)
        irr_target = max(0.005, (hurdle + tgt) / 100)
        benefit = amount / _annuity(irr_target, ramp)
        flows = [-amount] + [round(benefit * p, -2) for p in ramp]
        irr, payback = _irr(flows), _payback(flows)
        sponsor_rate = hurdle if reg != "conflicting" or chance(rng, 0.5) else hurdle - pick(rng, [2, 3, 4])
        linked = None
        if not good and ((reg == "conflicting" and chance(rng, 0.25)) or chance(rng, 0.04)):
            linked = round(amount * rng.uniform(0.4, 1.2), -4)
        envelope = round(amount * (rng.uniform(1.5, 6) if good or chance(rng, 0.6) else rng.uniform(0.35, 0.95)), -4)
        least_cost = (True if good else chance(rng, 0.55)) if cat == "compliance" else None
        quote = ("three firm bids" if good else
                 pick_w(rng, {"three firm bids": 4, "single firm quote": 2, "budgetary estimate (+/-25%)": 2, "internal estimate": 1}))
        track = (round(rng.uniform(0.95, 1.1), 2) if good else
                 pick_w(rng, {None: 2, round(rng.uniform(0.95, 1.1), 2): 3, round(rng.uniform(0.76, 0.94), 2): 3,
                              round(rng.uniform(0.5, 0.74), 2): 2}))
        pool = BASIS[cat]
        basis = pool[0] if good else pick_w(rng, dict(zip(pool, [3, 3, 1.3][: len(pool)])))
        down = [flows[0]] + [round(x * 0.85, -2) for x in flows[1:]]
        return {
            "regime": reg, "category": cat, "project": pick(rng, CAPEX[cat]), "site": pick(rng, SITES), "bu": pick(rng, BUS),
            "hurdle": hurdle, "payback_limit": pay_lim, "contingency_min": cont_min, "auth_site": a1, "auth_division": a2,
            "base_cost": base_cost, "contingency_pct": cont, "amount": amount, "life": life, "flows": flows,
            "annual_benefit": flows[-1], "irr": irr, "payback": payback, "downside_irr": _irr(down),
            "sponsor_rate": sponsor_rate, "sponsor_npv": round(_npv(sponsor_rate / 100, flows), -3), "linked_phase": linked,
            "envelope_left": envelope, "least_cost_shown": least_cost, "quote": quote, "track_record": track,
            "benefit_basis": basis, "sponsor": person(rng), "analyst": person(rng), "afe": ident(rng, "AFE", 5, "-"),
            "doc": _doc(rng, "CGM"), "date": rand_date(rng),
        }

    def _total(self, f):
        base = f["base_cost"] * (1 + max(f["contingency_pct"], f["contingency_min"]) / 100)
        return base + (f["linked_phase"] or 0)

    def intended(self, f):
        disc = f["category"] != "compliance"
        irr = f["irr"] if f["irr"] is not None else -1
        if disc:
            clears = irr * 100 >= f["hurdle"] and f["payback"] is not None and f["payback"] <= f["payback_limit"]
        else:
            clears = bool(f["least_cost_shown"])
        case_error = f["contingency_pct"] < f["contingency_min"] or f["linked_phase"] is not None
        weak_basis = f["quote"].startswith(("budgetary", "internal"))
        if not clears:
            far = disc and (irr * 100 < f["hurdle"] - 2.5 or f["payback"] is None or f["payback"] > f["payback_limit"] + 1.5)
            dec = "reject" if far else "return_for_rework"
        elif case_error:
            dec = "return_for_rework"
        elif disc and f["amount"] > f["envelope_left"]:
            dec = "defer_to_next_cycle"
        elif weak_basis or (disc and (f["downside_irr"] or -1) * 100 < f["hurdle"]) or (f["track_record"] or 1) < 0.75:
            dec = "approve_with_conditions"
        else:
            dec = "approve"
        if disc:
            if not clears:
                rob = 0
            elif (f["downside_irr"] or -1) * 100 < f["hurdle"]:
                rob = 1
            elif irr * 100 - f["hurdle"] < 7:
                rob = 2
            else:
                rob = 3
        else:
            rob = (2 if not weak_basis else 1) if f["least_cost_shown"] else 0
        pts = {"three firm bids": 1, "single firm quote": 0, "budgetary estimate (+/-25%)": -1, "internal estimate": -1}[f["quote"]]
        tr = f["track_record"]
        pts += 0 if tr is None else (1 if tr >= 0.95 else 0 if tr >= 0.75 else -1)
        pts += BASIS_PTS[f["benefit_basis"]]
        cred = 0 if pts <= -1 else 1 if pts == 0 else 2 if pts == 1 else 3
        return {"capex_decision": dec, "case_robustness": rob, "assumption_credibility": cred,
                "meets_return_tests": "yes" if clears else "no",
                "capital_committee_required": "yes" if self._total(f) > f["auth_division"] else "no"}

    def _rules(self, f, short=False):
        if short:
            return (f"Hurdle {f['hurdle']}% IRR / payback <= {f['payback_limit']} yrs (compliance exempt if least-cost shown); "
                    f"contingency >= {f['contingency_min']}%, linked phases aggregated; authority: site <= {_usd(f['auth_site'])}, "
                    f"division <= {_usd(f['auth_division'])}, above -> Capital Committee.")
        return [f"{f['doc']} §3.1: discretionary projects need IRR >= {f['hurdle']}% (hurdle) and payback <= {f['payback_limit']} "
                "years; compliance projects are exempt if the least-cost compliant option is shown.",
                f"§3.4: contingency of >= {f['contingency_min']}% must be included; linked phases count as one request.",
                f"§6.2: approval up to {_usd(f['auth_site'])} site director, up to {_usd(f['auth_division'])} division "
                "president, above that Capital Committee."]

    def _fin(self, f, rng):
        if f["category"] == "compliance":
            return {"amount_requested": _usd(f["amount"], pick(rng, ["m", "full"])),
                    "contingency_included": f"{f['contingency_pct']}%", "life_years": f["life"],
                    "added_operating_cost_per_year": _usd(abs(f["annual_benefit"]) * 0.08)}
        return {"amount_requested": _usd(f["amount"], pick(rng, ["m", "full"])),
                "contingency_included": f"{f['contingency_pct']}%", "annual_benefit": _usd(f["annual_benefit"]),
                "life_years": f["life"], "irr": _pc(f["irr"]), "simple_payback_years": f["payback"],
                "sponsor_npv": f"{_usd(f['sponsor_npv'])} at {f['sponsor_rate']}%",
                "irr_if_benefits_15pct_lower": _pc(f["downside_irr"])}

    def _notes(self, f, rng):
        s = []
        if f["linked_phase"]:
            s.append(f"Phase 2 ({_usd(f['linked_phase'])}) will follow next year as a separate request.")
        if f["track_record"] is not None:
            s.append(f"Post-audits of this site's last three projects realised {round(f['track_record'] * 100)}% of forecast benefits on average.")
        if f["least_cost_shown"] is True:
            s.append("Two cheaper designs were costed; neither meets the permit limits, so this is the lowest-cost compliant option.")
        elif f["least_cost_shown"] is False:
            s.append("No alternative designs were costed.")
        if f["category"] != "compliance" and f["sponsor_rate"] != f["hurdle"]:
            s.append(f"Sponsor used the group WACC of {f['sponsor_rate']}% for NPV.")
        return noisy(join_sentences(s), rng, 0.12)

    def render(self, f, rng, surface):
        fin = self._fin(f, rng)
        notes = self._notes(f, rng)
        what = f"{f['project']} at the {f['site']}"
        if surface == "json":
            return dumps(prune({
                "request": {"afe": f["afe"], "project": what, "category": CAT_LABEL[f["category"]], "sponsor": f["sponsor"],
                            "cost_basis": f["quote"], "benefit_basis": f["benefit_basis"],
                            "remaining_capital_envelope": _usd(f["envelope_left"]) if f["category"] != "compliance" else None},
                "financials": fin, "notes": notes or None, "policy": self._rules(f),
            }))
        comp = f["category"] == "compliance"
        if surface == "afe_form":
            rows = [("AFE", f["afe"]), ("Project", what), ("Category", CAT_LABEL[f["category"]]), ("Sponsor", f["sponsor"]),
                    ("Amount", fin["amount_requested"] + f" (incl. {fin['contingency_included']} contingency)"),
                    ("Cost basis", f["quote"]),
                    ("Annual benefit", None if comp else f"{fin['annual_benefit']} ({f['benefit_basis']})"),
                    ("Added running cost", fin.get("added_operating_cost_per_year")),
                    ("Life", f"{f['life']} yrs"), ("IRR / payback", None if comp else f"{fin['irr']} / {f['payback']} yrs"),
                    ("Downside IRR (-15% benefits)", fin.get("irr_if_benefits_15pct_lower")),
                    ("Sponsor NPV", fin.get("sponsor_npv")),
                    ("Capital envelope left", _usd(f["envelope_left"]) if f["category"] != "compliance" else None),
                    ("Policy limits", self._rules(f, short=True))]
            return "AUTHORISATION FOR EXPENDITURE\n" + kv_lines(rows, rng) + (f"\n\nComments: {notes}" if notes else "")
        body = [f"Subject: {f['afe']} - {f['project']} ({_usd(f['amount'])})", f"From: {f['analyst']} (FP&A)",
                f"To: capital-review@{f['bu'].split()[0].lower()}.example.com", "",
                noisy(join_sentences([
                    f"{f['sponsor']} has submitted the {what}, a {CAT_LABEL[f['category']]} project, for {fin['amount_requested']} "
                    f"including {fin['contingency_included']} contingency ({f['quote']}).",
                    (f"It is required by the {f['project'].split(' for the ')[-1]} and adds about "
                     f"{fin['added_operating_cost_per_year']} a year in running costs." if comp else
                     f"Benefits of {fin['annual_benefit']} a year over {f['life']} years ({f['benefit_basis']}) give an IRR of "
                     f"{fin['irr']} and payback of {f['payback']} years; with benefits 15% lower the IRR is "
                     f"{fin['irr_if_benefits_15pct_lower']}. The sponsor shows NPV of {fin['sponsor_npv']}."),
                    (f"Remaining envelope this year: {_usd(f['envelope_left'])}." if f["category"] != "compliance" else "")]), rng, 0.1),
                notes, "", "Relevant policy:", bullets(self._rules(f), rng)]
        return "\n".join(x for x in body if x is not None)

    def questions(self):
        return {
            "capex_decision": choice("What should the approver do with this capital request?", {
                "approve": "Approve the request as submitted.",
                "approve_with_conditions": "Approve, but release funds only once named conditions are met (firm pricing, a staged release, a benefits check).",
                "return_for_rework": "Send it back to correct or complete the case before any decision.",
                "defer_to_next_cycle": "Accept the case but defer it to the next capital cycle because this year's envelope cannot fund it.",
                "reject": "Reject the request; the project does not meet the investment tests.",
            }),
            "case_robustness": score("How well does the justification hold up against the test that applies to it and a weaker-benefits case?", [
                "It fails the applicable test.",
                "It passes, but a modest shortfall in benefits would sink it.",
                "It passes and survives the downside, with a thin margin.",
                "It passes comfortably and survives the downside with room to spare.",
            ]),
            "assumption_credibility": score("How credible are the cost and benefit assumptions behind the request?", [
                "Weak: rough estimates, optimistic volumes or a poor delivery record.",
                "Mixed: some solid inputs, some soft ones.",
                "Good: mostly firm pricing and measured baselines.",
                "Strong: firm bids, measured or contracted benefits and a reliable track record.",
            ]),
            "meets_return_tests": noul(
                "The project passes the return tests the capital manual sets for its category.",
                yes="A discretionary project meets both the hurdle and the payback limit, or a compliance project shows the least-cost option.",
                no="It misses the hurdle or the payback limit, or a compliance project shows no least-cost comparison."),
            "capital_committee_required": noul(
                "Under the delegation rules this request must go to the Capital Committee."),
        }


# =========================================================================== schema 3: portfolio reprioritisation
# (project template, obligation or None)
PORTFOLIO_PROJECTS = [
    ("ERP upgrade for the {bu} division", None), ("new grinding line at the {site}", None),
    ("digital twin for the {site}", None), ("expansion of the {site} warehouse", None),
    ("next-generation {bu} product platform", None), ("automation of the {site} assembly hall", None),
    ("solvent-recovery unit at the {site}", "air-permit condition"),
    ("customer-funded tooling for a supply contract", "signed customer supply contract"),
    ("fire-protection upgrade at the {site}", "fire authority's compliance order"),
    ("emissions-control retrofit at the {site}", "regulator's compliance order"),
    ("e-commerce channel build for {bu}", None), ("predictive-maintenance rollout across {bu} plants", None),
]
SPONSOR_ARG = {
    "sunk": ["We have already put {spent} into this; stopping now throws that away.",
             "With {spent} already spent, cancelling would waste the investment to date."],
    "mixed": ["We are {spent} in, and the remaining benefits are still real.",
              "Stopping would waste the {spent} spent so far, and the market still needs it."],
    "forward": ["Every remaining dollar still returns more than it costs on our numbers.",
                "The remaining spend buys the bulk of the benefit; the case stands on what is left to do."],
    "strategic": ["This is core to the division's three-year plan.", "Customers have been told this capability is coming."],
}


class PortfolioReprioritisation(Schema):
    name = "portfolio_reprioritisation"
    description = "One project's fate when the capital portfolio is cut: keep, rephase, pause or cancel under the written ranking rules."
    surfaces = {"json": 0.5, "portfolio_memo": 0.25, "ranking_sheet": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng, clear=0.25, borderline=0.45, conflicting=0.30)
        tmpl, oblig = pick(rng, PORTFOLIO_PROJECTS)
        if oblig and chance(rng, 0.45):
            tmpl, oblig = pick(rng, [p for p in PORTFOLIO_PROJECTS if p[1] is None])
        kind = "mandatory" if oblig else "discretionary"
        bu, site = pick(rng, BUS), pick(rng, SITES)
        cut_line = pick(rng, [0.3, 0.4, 0.5, 0.6, 0.8])
        rephase_max = pick(rng, [5, 8, 10])
        restart_max = pick(rng, [6, 9, 12])
        rem_total = round(math.exp(rng.gauss(math.log(6e6), 0.6)), -5)
        rem_year = round(rem_total * rng.uniform(0.35, 0.8), -5)
        spent = round(rem_total * rng.uniform(0.2, 2.5), -5)
        if reg == "clear":
            ratio = pick(rng, [rng.uniform(1.7, 3.0), rng.uniform(0.1, 0.5)])
        elif reg == "borderline":
            ratio = near(rng, 1.0, 0.25, lo=0.05)
        else:
            ratio = pick(rng, [rng.uniform(0.3, 0.85), rng.uniform(1.05, 2.0)])
        fnpv = round(cut_line * ratio * rem_total, -4)
        save_min = pick(rng, [30, 40, 50])
        if reg == "conflicting" and ratio < 1 and chance(rng, 0.45):
            exit_share = rng.uniform(1 - save_min / 100 + 0.03, 1.1)
        elif reg == "borderline" and chance(rng, 0.3):
            exit_share = near(rng, 1 - save_min / 100, 0.08, lo=0.0)
        else:
            exit_share = pick_w(rng, {rng.uniform(0.0, 0.08): 4, rng.uniform(0.12, 0.3): 3, rng.uniform(0.35, 0.55): 2,
                                      rng.uniform(0.75, 1.1): 1})
        exit_cost = round(rem_year * exit_share, -4)
        restart = pick_w(rng, {None: 2, near(rng, restart_max, 3, lo=2, integer=True): 3, rng.randint(2, 5): 2, rng.randint(14, 24): 2})
        rephase_ok = chance(rng, 0.75)
        rephase_loss = ((near(rng, rephase_max, 3, lo=1, digits=1) if reg == "borderline" else
                         round(pick_w(rng, {rng.uniform(1, rephase_max): 4, rng.uniform(rephase_max + 1, 20): 2}), 1))
                        if rephase_ok else None)
        status = pick_w(rng, {"green": 4, "amber": 3, "red": 2})
        if reg == "conflicting" and ratio >= 1 and chance(rng, 0.5):
            status = "red"
        slip = {"green": pick(rng, [0, 0, 1]), "amber": pick(rng, [2, 3, 4]), "red": pick(rng, [5, 6, 8, 10])}[status]
        arg = pick_w(rng, {"sunk": 3, "mixed": 2, "forward": 3, "strategic": 2})
        if reg == "conflicting" and ratio < 1:
            arg = pick(rng, ["sunk", "sunk", "mixed", "strategic"])
        deadline = pick(rng, [6, 9, 12, 18, 24, 30]) if kind == "mandatory" else None
        return {
            "regime": reg, "project": tmpl.format(bu=bu, site=site), "kind": kind, "bu": bu,
            "obligation": oblig, "deadline_months": deadline,
            "cut_line": cut_line, "rephase_max_loss": rephase_max, "restart_max": restart_max, "save_min": save_min,
            "remaining_total": rem_total, "remaining_this_year": rem_year, "spent": spent, "forward_npv": fnpv,
            "exit_cost": exit_cost, "restart_months": restart, "rephase_loss_pct": rephase_loss,
            "deferrable": round(rem_year * rng.uniform(0.3, 0.6), -5) if rephase_ok else None,
            "status": status, "slip_months": slip, "argument": arg, "sponsor": person(rng), "pmo": person(rng),
            "budget_from": round(max(30, rem_year / 1e6 * rng.uniform(6, 14)), 0), "cut_pct": pick(rng, [10, 12, 15, 18, 20, 25]),
            "doc": _doc(rng, "PP"), "date": rand_date(rng), "rank_of": rng.randint(18, 40),
        }

    def _pi(self, f):
        return f["forward_npv"] / f["remaining_total"]

    def intended(self, f):
        pi = self._pi(f)
        above = pi >= f["cut_line"]
        rephase = f["rephase_loss_pct"] is not None and f["rephase_loss_pct"] <= f["rephase_max_loss"]
        saving = f["remaining_this_year"] - f["exit_cost"]
        restartable = f["restart_months"] is not None and f["restart_months"] <= f["restart_max"]
        if f["kind"] == "mandatory":
            act = "rephase_spend" if (rephase and f["deadline_months"] >= 18) else "keep_fully_funded"
            basis = "obligation"
        elif above:
            act = "rephase_spend" if (rephase and pi < 1.6 * f["cut_line"]) else "keep_fully_funded"
            basis = "delivery_risk" if (f["status"] == "red" and f["slip_months"] >= 6) else "forward_value"
        else:
            if saving < f["save_min"] / 100 * f["remaining_this_year"]:
                act, basis = ("rephase_spend" if rephase else "keep_fully_funded"), "exit_costs"
            else:
                act = "pause_and_preserve" if restartable else "cancel"
                basis = "delivery_risk" if (f["status"] == "red" and f["slip_months"] >= 6) else "forward_value"
        r = pi / f["cut_line"]
        fv = 0 if r < 0.6 else 1 if r < 1.0 else 2 if r < 1.6 else 3
        e = f["exit_cost"] / f["remaining_this_year"]
        ec = 0 if e < 0.1 else 1 if e < 0.3 else 2 if e < 0.6 else 3
        return {"portfolio_action": act, "deciding_basis": basis, "forward_value": fv, "exit_cost_burden": ec,
                "sunk_cost_reasoning": "yes" if f["argument"] in ("sunk", "mixed") else "no"}

    def _rules(self, f, rng):
        r = [f"{f['doc']} §2: discretionary projects rank on forward value = NPV of remaining cash flows / remaining spend "
             f"(sunk spend excluded); below {f['cut_line']:.1f} is under the cut line.",
             f"§4: under the line, pause if restartable within {f['restart_max']} months, else cancel; rephasing allowed if "
             f"it costs <= {f['rephase_max_loss']}% of forward NPV; stop only if that saves >= {f['save_min']}% of this "
             "year's remaining spend after exit costs."]
        if f["kind"] == "mandatory" or chance(rng, 0.35):
            r.insert(1, "§3: projects with a regulatory or contractual obligation are funded ahead of the ranking.")
        return r

    def _arg(self, f, rng):
        return pick(rng, SPONSOR_ARG[f["argument"]]).format(spent=_usd(f["spent"]))

    def render(self, f, rng, surface):
        rest = (f"{f['restart_months']} months" if f["restart_months"] is not None else None)
        reph = (f"defer {_usd(f['deferrable'])} to next year, forward NPV -{f['rephase_loss_pct']}%"
                if f["rephase_loss_pct"] is not None else "not possible (single contract)")
        if surface == "json":
            return dumps(prune({
                "context": f"Portfolio cut {f['cut_pct']}% from ${f['budget_from']:.0f}M",
                "project": {"name": f["project"], "obligation": f["obligation"],
                            "obligation_deadline_months": f["deadline_months"], "sponsor": f["sponsor"],
                            "status": f"{f['status']}, {f['slip_months']} months behind" if f["slip_months"] else f["status"]},
                "money": {"spent_to_date": _usd(f["spent"]), "remaining_spend_total": _usd(f["remaining_total"]),
                          "remaining_this_year": _usd(f["remaining_this_year"]),
                          "forward_npv": _usd(f["forward_npv"]), "cost_to_exit_now": _usd(f["exit_cost"])},
                "options": {"rephase": reph, "restart_after_pause": rest or "not assessed"},
                "sponsor_view": noisy(self._arg(f, rng), rng, 0.1),
                "rules": self._rules(f, rng),
            }))
        if surface == "portfolio_memo":
            para = [f"With the portfolio cut by {f['cut_pct']}%, {f['project']} is under review.",
                    (f"It is tied to a {f['obligation']} due in {f['deadline_months']} months." if f["obligation"] else ""),
                    f"Spent so far {_usd(f['spent'])}; {_usd(f['remaining_total'])} remains, {_usd(f['remaining_this_year'])} of it this year.",
                    f"Forward NPV on the remaining spend is {_usd(f['forward_npv'])}.",
                    f"Stopping now would cost {_usd(f['exit_cost'])} in termination and demobilisation.",
                    f"Rephasing: {reph}.", (f"A pause could be restarted in about {rest}." if rest else "Restart time after a pause has not been assessed."),
                    f"Delivery status {f['status']}" + (f", {f['slip_months']} months late." if f["slip_months"] else "."),
                    f"Sponsor ({f['sponsor']}): \"{self._arg(f, rng)}\""]
            return (f"MEMO - portfolio review ({fmt_date(f['date'], rng)})\nFrom: {f['pmo']}, PMO\n\n"
                    + noisy(join_sentences(para), rng, 0.08) + "\n\n" + "\n".join(self._rules(f, rng)))
        pi = self._pi(f)
        rows = [[f["project"], _usd(f["spent"]), _usd(f["remaining_total"]), _usd(f["forward_npv"]), f"{pi:.2f}",
                 _usd(f["exit_cost"]), f["status"]]]
        head = (f"Reprioritisation sheet - cut {f['cut_pct']}% | cut line {f['cut_line']:.1f} forward NPV per remaining $ "
                f"({f['doc']} §2, sunk spend excluded) | pause only if restart <= {f['restart_max']} mo, else cancel | "
                f"rephase if NPV loss <= {f['rephase_max_loss']}% | stop only if it saves >= {f['save_min']}% of this year's spend net of exit costs (§4)")
        extra = kv_lines([("Obligation", f"{f['obligation']}, due in {f['deadline_months']} months" if f["obligation"] else None),
                          ("Rephase option", reph), ("Restart after pause", rest or "not assessed"),
                          ("Slip", f"{f['slip_months']} months" if f["slip_months"] else None),
                          ("Sponsor comment", self._arg(f, rng))], rng)
        return head + "\n\n" + table(["project", "spent", "remaining", "fwd NPV", "fwd ratio", "exit cost", "RAG"], rows, rng) + "\n\n" + extra

    def questions(self):
        return {
            "portfolio_action": choice("What should the portfolio committee do with this project?", {
                "keep_fully_funded": "Keep it funded on its current plan.",
                "rephase_spend": "Keep it, but push part of this year's spend into next year.",
                "pause_and_preserve": "Pause it now and preserve the work so it can restart later.",
                "cancel": "Cancel it and release the remaining budget.",
            }),
            "deciding_basis": choice("What should the committee's decision on this project mainly rest on?", {
                "obligation": "A regulatory or contractual obligation that takes it out of the ranking.",
                "forward_value": "How its forward value per remaining dollar compares with the cut line.",
                "exit_costs": "The cost of stopping now compared with what stopping would save this year.",
                "delivery_risk": "Its delivery troubles, which undermine the forecast it is ranked on.",
            }),
            "forward_value": score("How much value does finishing the project create for the money still to be spent?", [
                "Well below the cut line.",
                "Somewhat below the cut line.",
                "At or somewhat above the cut line.",
                "Far above the cut line.",
            ]),
            "exit_cost_burden": score("How heavy is the cost of stopping the project now, relative to this year's remaining spend?", [
                "Trivial: stopping saves almost all of this year's spend.",
                "Modest: a noticeable but small share is lost.",
                "Heavy: a large share of the saving is eaten by exit costs.",
                "Prohibitive: stopping saves little or nothing this year.",
            ]),
            "sunk_cost_reasoning": noul(
                "The sponsor's case for keeping the project leans on money already spent.",
                yes="The argument cites the spend to date as a reason to continue.",
                no="The argument rests on remaining value, strategy or obligations, not on past spend."),
        }


# =========================================================================== schema 4: make vs buy
# (item, typical annual volume, typical in-house unit cost, supplier trade)
ITEMS_MVB = [
    ("machined pump housing", 18000, 140, "Precision Machining"), ("injection-moulded control-panel bezel", 220000, 3.2, "Moulding"),
    ("PCB assembly for the motor controller", 60000, 38, "Electronics"), ("powder coating of steel frames", 90000, 6.5, "Coatings"),
    ("die-cast gearbox cover", 45000, 22, "Castings"), ("wire harness for the dryer range", 150000, 9.8, "Harness Systems"),
    ("stamped mounting brackets", 400000, 0.9, "Stampings"), ("turned stainless shafts", 75000, 14, "Precision Machining"),
    ("CNC-cut gasket kits", 120000, 2.4, "Sealing Products"), ("blow-moulded detergent bottles", 900000, 0.35, "Plastics"),
    ("welded skid bases", 2500, 820, "Fabrication"), ("heat-exchanger brazing", 12000, 95, "Thermal Products"),
]
SUPPLIER_LOC = ["same region, 2-day lead time", "neighbouring country, 1-week lead time", "overseas, 6-week lead time by sea"]


class MakeVsBuy(Schema):
    name = "make_vs_buy"
    description = "A request to move a component or process from in-house production to an outside supplier, checked against the sourcing standard."
    surfaces = {"json": 0.5, "sourcing_brief": 0.25, "cost_table": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng, clear=0.25, borderline=0.45, conflicting=0.30)
        item, vol0, cost0, trade = pick(rng, ITEMS_MVB)
        vol = int(round(vol0 * rng.uniform(0.6, 1.6), -2 if vol0 < 10000 else -3))
        unit = cost0 * rng.uniform(0.8, 1.25)
        dm = unit * rng.uniform(0.35, 0.55)
        dl = unit * rng.uniform(0.12, 0.25)
        voh = unit * rng.uniform(0.05, 0.12)
        foh = unit - dm - dl - voh
        avoid = pick(rng, [0.0, 0.1, 0.2, 0.3, 0.5])
        avoidable = dm + dl + voh + foh * avoid
        # which sourcing-standard gate binds; the regime decides how close to its line the case sits
        scen = pick_w(rng, {"buy": 3, "split": 1.6, "pilot": 1.4, "quality": 1.1, "core": 1.0, "uneconomic": 2.2,
                            "budgetary": 1.1})
        k = pick(rng, [12, 18, 24])
        if scen == "uneconomic":
            sav = (rng.uniform(-0.15, -0.04) if reg == "clear" else near(rng, 0.0, 0.03) if reg == "borderline"
                   else rng.uniform(-0.1, 0.06))
        elif scen == "budgetary":
            sav = rng.uniform(0.02, 0.13)
        elif reg == "clear":
            sav = rng.uniform(0.16, 0.3)
        elif reg == "borderline":
            sav = near(rng, 0.06, 0.04, lo=0.01)
        else:
            sav = rng.uniform(0.05, 0.2)
        landed = avoidable * (1 - sav)
        freight = landed * rng.uniform(0.03, 0.09)
        price = landed - freight
        annual_saving = (avoidable - landed) * vol
        if annual_saving > 0:
            if scen == "uneconomic" and sav > 0:
                mult = rng.uniform(1.1, 1.8)
            elif reg == "borderline":
                mult = near(rng, 0.92, 0.12, lo=0.3)
            else:
                mult = rng.uniform(0.3, 0.85)
            transition = round(annual_saving * (k / 12) * mult, -3)
        else:
            transition = round(avoidable * vol * rng.uniform(0.03, 0.08), -3)
        ppm_lim = pick(rng, [250, 500, 1000])
        ppm_kind = {"pilot": "short", "quality": "high"}.get(scen, "ok")
        if scen in ("uneconomic", "core", "budgetary"):
            ppm_kind = pick_w(rng, {"ok": 3, "high": 1.2, "short": 1})
        if reg == "borderline" and ppm_kind == "ok":
            ppm = round(ppm_lim * rng.uniform(0.75, 1.0))
        else:
            ppm = {"ok": round(ppm_lim * rng.uniform(0.2, 0.9)), "high": round(ppm_lim * rng.uniform(1.03, 1.6 if reg == "borderline" else 2.2)),
                   "short": round(ppm_lim * rng.uniform(0.3, 0.9))}[ppm_kind]
        cap = round(rng.uniform(0.5, 0.85), 2) if scen == "split" else 1.0
        core = scen == "core" or chance(rng, 0.06)
        waiver = (pick(rng, ["requested, pending", "not requested"]) if scen == "core"
                  else "granted" if core else None)
        basis = pick_w(rng, {"full_cost": 5, "avoidable": 4})
        if reg == "conflicting" and avoid < 0.5:
            basis = "full_cost"
        quote_kind = "budgetary" if scen == "budgetary" else "firm"
        return {
            "regime": reg, "item": item, "volume": vol, "dm": round(dm, 2), "dl": round(dl, 2), "voh": round(voh, 2),
            "foh": round(foh, 2), "avoid_share": avoid, "supplier_price": round(price, 2), "freight_duty": round(freight, 2),
            "transition_cost": transition, "payback_limit_months": k, "ppm_limit": ppm_lim, "ppm": ppm, "ppm_kind": ppm_kind,
            "capacity_share": cap, "core_item": core, "cto_waiver": waiver, "request_basis": basis, "quote": quote_kind,
            "scenario": scen,
            "supplier": f"{pick(rng, ORG_A)} {trade}", "location": pick(rng, SUPPLIER_LOC), "requester": person(rng),
            "site": pick(rng, SITES), "doc": _doc(rng, "SS"), "ref": ident(rng, "MVB", 4, "-"), "date": rand_date(rng),
        }

    def _costs(self, f):
        full = f["dm"] + f["dl"] + f["voh"] + f["foh"]
        avoidable = f["dm"] + f["dl"] + f["voh"] + f["foh"] * f["avoid_share"]
        landed = f["supplier_price"] + f["freight_duty"]
        return full, avoidable, landed

    def intended(self, f):
        full, avoidable, landed = self._costs(f)
        sav_pct = (avoidable - landed) / avoidable
        annual = (avoidable - landed) * f["volume"]
        pay = f["transition_cost"] / (annual / 12) if annual > 0 else None
        qualified = f["ppm_kind"] != "short" and f["ppm"] <= f["ppm_limit"]
        if f["quote"] == "budgetary" and 0 < sav_pct < 0.15:
            dec = "defer_pending_data"
        elif f["core_item"] and f["cto_waiver"] != "granted":
            dec = "keep_making"
        elif pay is None or pay > f["payback_limit_months"]:
            dec = "keep_making"
        elif f["ppm_kind"] == "short":
            dec = "qualify_with_pilot_lot"
        elif not qualified:
            dec = "keep_making"
        elif f["capacity_share"] < 1.0:
            dec = "split_volume"
        else:
            dec = "buy_from_supplier"
        adv = 0 if sav_pct <= 0 else 1 if sav_pct < 0.05 else 2 if sav_pct < 0.15 else 3
        pts = (0 if f["ppm"] <= 0.5 * f["ppm_limit"] and f["ppm_kind"] == "ok" else 1 if qualified else 2)
        pts += (f["capacity_share"] < 1.0) + ("overseas" in f["location"])
        return {"sourcing_decision": dec, "true_cost_advantage": adv, "supply_risk": _clamp(pts),
                "overhead_overstated": "yes" if f["request_basis"] == "full_cost" and f["avoid_share"] < 1 else "no",
                "supplier_quality_qualified": "yes" if qualified else "no"}

    def _rules(self, f):
        return [f"{f['doc']} §2: compare supplier landed cost with in-house AVOIDABLE cost (allocated overhead that stays is excluded).",
                f"§3: savings must repay the one-time transition cost within {f['payback_limit_months']} months.",
                f"§5: supplier needs a 12-month record <= {f['ppm_limit']} PPM; core-technology items need a CTO waiver."]

    def _fmt(self, x):
        return f"${x:,.2f}" if x < 100 else f"${x:,.0f}"

    def _claim(self, f):
        full, avoidable, landed = self._costs(f)
        base = full if f["request_basis"] == "full_cost" else avoidable
        return base, (base - landed) * f["volume"]

    def render(self, f, rng, surface):
        full, avoidable, landed = self._costs(f)
        base, claim = self._claim(f)
        ppm_txt = (f"{f['ppm']} PPM, trailing 12 months" if f["ppm_kind"] != "short"
                   else f"{f['ppm']} PPM over {pick(rng, ['5', '6', '7'])} months (new supplier)")
        cap_txt = "full volume" if f["capacity_share"] >= 1 else f"{round(f['capacity_share'] * 100)}% of volume"
        core = (f"yes - CTO waiver {f['cto_waiver']}" if f["core_item"] else "no")
        claim_txt = f"{_usd(claim)}/yr vs {'full' if f['request_basis'] == 'full_cost' else 'avoidable'} cost"
        if surface == "json":
            return dumps(prune({
                "request": {"ref": f["ref"], "item": f["item"], "site": f["site"], "annual_volume": f["volume"],
                            "requested_by": f["requester"], "claimed_saving": claim_txt},
                "in_house_cost_per_unit": {"material": self._fmt(f["dm"]), "direct_labour": self._fmt(f["dl"]),
                                           "variable_overhead": self._fmt(f["voh"]), "allocated_fixed_overhead": self._fmt(f["foh"]),
                                           "fixed_overhead_avoidable_if_bought": f"{round(f['avoid_share'] * 100)}%"},
                "supplier": {"name": f["supplier"], "unit_price": self._fmt(f["supplier_price"]), "quote": f["quote"],
                             "freight_and_duty_per_unit": self._fmt(f["freight_duty"]), "quality": ppm_txt,
                             "capacity": cap_txt, "location": f["location"]},
                "transition_cost_one_time": _usd(f["transition_cost"]), "core_technology_item": core,
                "policy": self._rules(f)[:2] + ([self._rules(f)[2]] if f["core_item"] or f["ppm_kind"] != "ok" or chance(rng, 0.6) else []),
            }))
        if surface == "sourcing_brief":
            para = [f"{f['requester']} proposes buying the {f['item']} ({f['volume']:,} a year) from {f['supplier']} "
                    f"({f['location']}) instead of making it at the {f['site']}.",
                    f"In-house cost is {self._fmt(full)} a unit: material {self._fmt(f['dm'])}, labour {self._fmt(f['dl'])}, "
                    f"variable overhead {self._fmt(f['voh'])} and allocated fixed overhead {self._fmt(f['foh'])}, of which "
                    f"finance estimates {round(f['avoid_share'] * 100)}% would go away.",
                    f"The {f['quote']} quote is {self._fmt(f['supplier_price'])} plus {self._fmt(f['freight_duty'])} freight and duty.",
                    f"One-time transition (tooling transfer, qualification): {_usd(f['transition_cost'])}.",
                    f"Supplier quality: {ppm_txt}; capacity: {cap_txt}.",
                    (f"The item is on the core-technology list; CTO waiver {f['cto_waiver']}." if f["core_item"] else ""),
                    f"The request claims a saving of {claim_txt}."]
            return (f"SOURCING BRIEF {f['ref']} ({fmt_date(f['date'], rng)})\n\n" + noisy(join_sentences(para), rng, 0.08)
                    + "\n\n" + bullets(self._rules(f), rng))
        rows = [["material", self._fmt(f["dm"])], ["direct labour", self._fmt(f["dl"])],
                ["variable overhead", self._fmt(f["voh"])],
                ["allocated fixed overhead", f"{self._fmt(f['foh'])} ({round(f['avoid_share'] * 100)}% avoidable)"],
                ["supplier price", self._fmt(f["supplier_price"]) + (" (budgetary)" if f["quote"] == "budgetary" else "")],
                ["freight + duty", self._fmt(f["freight_duty"])]]
        head = kv_lines([("Item", f"{f['item']} - {f['volume']:,}/yr"), ("Supplier", f"{f['supplier']}, {f['location']}"),
                         ("Quality", ppm_txt + f" (limit {f['ppm_limit']} PPM, 12-month record)"), ("Capacity", cap_txt),
                         ("Transition cost", f"{_usd(f['transition_cost'])} (must repay within {f['payback_limit_months']} months)"),
                         ("Core-technology list", core), ("Requester's saving", claim_txt)], rng)
        return (f"MAKE/BUY COST COMPARISON {f['ref']}\n{head}\n\n" + table(["per unit", "cost"], rows, rng)
                + f"\n\nBasis ({f['doc']} §2): landed cost vs avoidable in-house cost.")

    def questions(self):
        return {
            "sourcing_decision": choice("What should the sourcing council decide on this make-or-buy request?", {
                "keep_making": "Keep making it in-house.",
                "buy_from_supplier": "Move the full volume to the supplier.",
                "split_volume": "Move part of the volume and keep the rest in-house.",
                "qualify_with_pilot_lot": "Run a pilot lot to build the supplier's quality record before deciding.",
                "defer_pending_data": "Defer until firm pricing is available to confirm the saving.",
            }),
            "true_cost_advantage": score("Compared on the basis the standard requires, how large is the supplier's cost advantage?", [
                "None: making it in-house is as cheap or cheaper.",
                "Slight: under 5% cheaper.",
                "Moderate: 5-15% cheaper.",
                "Large: more than 15% cheaper.",
            ]),
            "supply_risk": score("How much supply risk would relying on this supplier add?", [
                "Little: strong quality record, ample capacity, short lead time.",
                "Some: one weak point to manage.",
                "Considerable: several weak points.",
                "High: quality, capacity and distance all work against it.",
            ]),
            "overhead_overstated": noul(
                "The request's saving counts overhead that would not go away if the item were bought.",
                yes="It compares the supplier with full in-house cost although part of the allocated overhead would remain.",
                no="It compares the supplier with the avoidable in-house cost."),
            "supplier_quality_qualified": noul(
                "The supplier meets the standard's quality qualification requirement."),
        }


# =========================================================================== schema 5: post-investment review
PIR_PROJECTS = ["third extrusion line", "automated warehouse", "new paint shop", "bottling line upgrade",
                "plant-wide energy management system", "robotic welding cell", "co-generation unit",
                "regional distribution centre", "blow-moulding expansion", "precision grinding cell"]
CAUSE_EVIDENCE = {
    "external_market": "industry volume index {m:+.0f}% since approval",
    "project_execution": "commissioning finished {d} months late; contractor rework on {what}",
    "estimating_optimism": "case assumed {a}% line efficiency; best month achieved {b}%",
    "approved_scope_change": "{n} approved change orders added {c} (scope added: {what2})",
}


class PostInvestmentReview(Schema):
    name = "post_investment_review"
    description = "A post-investment review of a completed capital project that missed its approved case: close it, record lessons, demand recovery or review exit."
    surfaces = {"json": 0.5, "pir_report": 0.25, "steering_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng, clear=0.25, borderline=0.45, conflicting=0.30)
        x_lim, y_lim = pick(rng, [10, 15, 20]), pick(rng, [15, 20, 25])
        hurdle = pick(rng, [9, 10, 11, 12])
        capex = round(math.exp(rng.gauss(math.log(8e6), 0.6)), -5)
        life = pick(rng, [8, 10, 12])
        irr_a = (hurdle + rng.uniform(3, 12)) / 100
        ben = capex / _annuity(irr_a, [1.0] * life)
        cause = pick(rng, list(CAUSE_EVIDENCE))
        x, y = x_lim / 100, y_lim / 100
        capex_channel = cause in ("approved_scope_change", "project_execution") and not (cause == "project_execution" and chance(rng, 0.4))
        sig = chance(rng, 0.5) if reg == "clear" else None if reg == "borderline" else chance(rng, 0.7)
        if sig is None:
            over = near(rng, x, 0.035) if capex_channel else rng.uniform(-0.04, x - 0.03)
            real = rng.uniform(1 - y + 0.03, 1.0) if capex_channel else near(rng, 1 - y, 0.04)
        elif sig:
            over = rng.uniform(x + 0.02, x + 0.3) if capex_channel else rng.uniform(-0.04, x * 0.7)
            real = rng.uniform(0.85, 1.0) if capex_channel else rng.uniform(0.5, 1 - y - 0.02)
        else:
            over = rng.uniform(-0.05, x - 0.02)
            real = rng.uniform(1 - y + 0.02, 1.03)
        actual = round(capex * (1 + over), -4)
        realised = round(ben * real, -3)
        fwd = ben * min(1.05, real * rng.uniform(0.95, 1.15))
        opcf = round(fwd * rng.uniform(0.6, 1.0) if real > 0.68 or chance(rng, 0.3) else -ben * rng.uniform(0.04, 0.2), -3)
        flows_a = [-capex] + [round(ben, -3)] * life
        flows_r = [-actual] + [realised, realised] + [round(fwd, -3)] * (life - 2)
        if opcf < 0:
            flows_r = [-actual] + [realised, realised] + [opcf] * (life - 2)
        sponsor_cause = cause if (reg != "conflicting" or chance(rng, 0.3)) else "external_market" if cause != "external_market" else "project_execution"
        short = max(0.0, 1 - real)
        a_eff = rng.randint(88, 95)
        ev = {"m": max(-35.0, -(short * 100 * rng.uniform(0.55, 0.85)) - 2) if cause == "external_market" else rng.uniform(-3, 6),
              "d": max(3, round(max(over, 0) * 25 + short * 25)) if cause == "project_execution" else rng.randint(0, 2),
              "a": a_eff, "b": a_eff - max(3, round(short * 75)) if cause == "estimating_optimism" else a_eff - rng.randint(-2, 3),
              "n": rng.randint(3, 7) if cause == "approved_scope_change" else rng.randint(0, 1),
              "c": _usd(capex * max(over, 0.03) * rng.uniform(0.85, 1.0)) if cause == "approved_scope_change" else _usd(capex * rng.uniform(0.0, 0.02)),
              "what": pick(rng, ["the conveyor interfaces", "the control panels", "the foundations", "the steam tie-in"]),
              "what2": pick(rng, ["a second packing station", "larger silos", "an extra robot cell", "a mezzanine"])}
        return {
            "regime": reg, "project": pick(rng, PIR_PROJECTS), "site": pick(rng, SITES), "bu": pick(rng, BUS),
            "x_lim": x_lim, "y_lim": y_lim, "hurdle": hurdle, "recovery_days": pick(rng, [60, 90]),
            "capex": capex, "actual_capex": actual, "life": life, "case_benefit": round(ben, -3), "realised_benefit": realised,
            "approved_npv": round(_npv(hurdle / 100, flows_a), -4), "approved_irr": _irr(flows_a),
            "reforecast_npv": round(_npv(hurdle / 100, flows_r), -4), "operating_cf": opcf,
            "cause": cause, "sponsor_cause": sponsor_cause, "evidence": ev,
            "prior_flags": pick_w(rng, {0: 6, 1: 3, 2: 1.5}), "sponsor": person(rng), "reviewer": person(rng),
            "approved_year": pick(rng, [2021, 2022, 2023]), "doc": _doc(rng, "PIR"), "date": rand_date(rng),
        }

    def _variance(self, f):
        over = f["actual_capex"] / f["capex"] - 1
        short = 1 - f["realised_benefit"] / f["case_benefit"]
        return over, short

    def intended(self, f):
        over, short = self._variance(f)
        sig = over * 100 > f["x_lim"] or short * 100 > f["y_lim"]
        if not sig:
            out = "close_as_delivered"
        elif f["reforecast_npv"] >= 0:
            out = "close_with_lessons"
        elif f["operating_cf"] >= 0:
            out = "recovery_plan_required"
        else:
            out = "exit_options_review"
        q = f["reforecast_npv"] / f["approved_npv"]
        vs = 0 if q >= 0.85 else 1 if q >= 0.3 else 2 if q >= -0.3 else 3
        sc = {"estimating_optimism": 2, "project_execution": 1, "external_market": 0, "approved_scope_change": 0}[f["cause"]]
        sc += (f["prior_flags"] >= 1) + (f["sponsor_cause"] != f["cause"]) - (not sig)
        return {"pir_outcome": out, "variance_cause": f["cause"], "value_shortfall": vs, "future_case_scrutiny": _clamp(sc),
                "significant_variance": "yes" if sig else "no"}

    def _evidence_lines(self, f):
        ev = f["evidence"]
        out = [CAUSE_EVIDENCE["external_market"].format(**ev)]
        if ev["d"] >= 1:
            out.append(CAUSE_EVIDENCE["project_execution"].format(**ev) if ev["d"] >= 3
                       else f"commissioning {ev['d']} month{'s' if ev['d'] > 1 else ''} late")
        else:
            out.append("commissioned on schedule")
        out.append(CAUSE_EVIDENCE["estimating_optimism"].format(**ev))
        out.append(CAUSE_EVIDENCE["approved_scope_change"].format(**ev) if ev["n"] >= 2
                   else f"one minor change order ({ev['c']})" if ev["n"] == 1 else "no change orders")
        return out

    SPONSOR_SAYS = {"external_market": "volumes fell with the market", "project_execution": "the contractor delivered late",
                    "estimating_optimism": "the original assumptions were too ambitious",
                    "approved_scope_change": "the scope grew with approved changes"}

    def _rules(self, f):
        return [f"{f['doc']} §2: significant variance = capex over approval by > {f['x_lim']}% or benefits > {f['y_lim']}% "
                "short of the case.",
                f"§4: re-forecast NPV < 0 at the original {f['hurdle']}% hurdle -> recovery plan within {f['recovery_days']} "
                "days; if operating cash flow is also negative -> exit-options review.",
                "§5: the review team, not the sponsor, classifies the cause."]

    def render(self, f, rng, surface):
        over, short = self._variance(f)
        evid = self._evidence_lines(f)
        rng.shuffle(evid)
        opcf = f"{_usd(f['operating_cf'])}/yr"
        if surface == "json":
            return dumps(prune({
                "project": {"name": f"{f['project']}, {f['site']}", "approved": f["approved_year"], "sponsor": f["sponsor"]},
                "approved_case": {"capex": _usd(f["capex"]), "annual_benefit": _usd(f["case_benefit"]),
                                  "npv": _usd(f["approved_npv"]), "irr": _pc(f["approved_irr"])},
                "actuals": {"capex": _usd(f["actual_capex"]), "benefit_realised_avg_first_2_years": _usd(f["realised_benefit"]),
                            "operating_cash_flow_last_12_months": opcf, "reforecast_npv": _usd(f["reforecast_npv"])},
                "findings": evid, "sponsor_explanation": self.SPONSOR_SAYS[f["sponsor_cause"]],
                "earlier_reviews_flagged_for_this_sponsor": f["prior_flags"], "procedure": self._rules(f)[:2],
            }))
        if surface == "pir_report":
            rows = [["capex", _usd(f["capex"]), _usd(f["actual_capex"]), f"{over * 100:+.0f}%"],
                    ["annual benefit", _usd(f["case_benefit"]), _usd(f["realised_benefit"]), f"{-short * 100:+.0f}%"],
                    [f"NPV @ {f['hurdle']}%", _usd(f["approved_npv"]), _usd(f["reforecast_npv"]) + " (re-forecast)", ""]]
            return (f"POST-INVESTMENT REVIEW - {f['project']}, {f['site']} (approved {f['approved_year']})\n"
                    f"Reviewer: {f['reviewer']}   Sponsor: {f['sponsor']}\n"
                    f"Significance test ({f['doc']} §2): capex > +{f['x_lim']}% or benefits > {f['y_lim']}% short\n\n"
                    + table(["", "case", "actual", "var"], rows, rng)
                    + f"\n\nOperating cash flow, last 12 months: {opcf}\nFindings:\n" + bullets(evid, rng)
                    + f"\nSponsor's explanation: {self.SPONSOR_SAYS[f['sponsor_cause']]}."
                    + f"\nPrior reviews flagged for this sponsor: {f['prior_flags']}")
        body = [f"Subject: PIR - {f['project']} ({f['site']})", f"From: {f['reviewer']}", "To: capital steering group", "",
                noisy(join_sentences([
                    f"The {f['project']} was approved in {f['approved_year']} at {_usd(f['capex'])} for {_usd(f['case_benefit'])} a year "
                    f"(NPV {_usd(f['approved_npv'])}).",
                    f"It cost {_usd(f['actual_capex'])} and delivered {_usd(f['realised_benefit'])} a year over its first two years; "
                    f"operating cash flow over the last 12 months is {opcf}.",
                    f"Re-forecast NPV at {f['hurdle']}%: {_usd(f['reforecast_npv'])}.",
                    f"{f['sponsor']} attributes the gap to the view that {self.SPONSOR_SAYS[f['sponsor_cause']]}.",
                    "What we found: " + "; ".join(evid) + ".",
                    (f"This sponsor has {f['prior_flags']} earlier flagged review(s)." if f["prior_flags"] else "")]), rng, 0.08),
                "", "Procedure:", bullets(self._rules(f), rng)]
        return "\n".join(body)

    def questions(self):
        return {
            "pir_outcome": choice("How should the review be closed out?", {
                "close_as_delivered": "Close it: the project delivered within the tolerances.",
                "close_with_lessons": "Close it with a significant-variance finding and lessons recorded; the project still creates value.",
                "recovery_plan_required": "Require the sponsor to file a recovery plan.",
                "exit_options_review": "Open a review of exit options for the asset.",
            }),
            "variance_cause": choice("What mainly explains the gap between the approved case and the outcome?", {
                "external_market": "Market conditions outside the project's control.",
                "project_execution": "Delays or rework in delivering the project.",
                "estimating_optimism": "Assumptions in the original case that were never realistic.",
                "approved_scope_change": "Scope added through approved changes after the original approval.",
            }),
            "value_shortfall": score("How much of the approved value has been lost on the re-forecast?", [
                "Little: the re-forecast is close to the approved value.",
                "Some: clearly lower, but the project still creates good value.",
                "Most: the re-forecast is near zero.",
                "All and more: the project now destroys substantial value.",
            ]),
            "future_case_scrutiny": score("How much extra scrutiny should this sponsor's future capital cases receive?", [
                "None beyond the normal process.",
                "A little: a finance check on the key assumptions.",
                "Considerable: independent review of volumes and costs.",
                "Maximum: independent review plus staged funding releases.",
            ]),
            "significant_variance": noul(
                "The project's variance counts as significant under the review procedure.",
                yes="The capex overrun or the benefit shortfall exceeds its stated tolerance.",
                no="Both the capex overrun and the benefit shortfall are within tolerance."),
        }


WORKFLOW = Workflow(
    name="capital_allocation_and_stage_gate",
    title="Capital allocation and stage-gate decisions",
    description="Stage-gate, capital-request, portfolio, make-or-buy and post-investment decisions in a manufacturer, judged against its written (fictional) capital governance rules.",
    domain="management / corporate finance",
    schemas=[StageGateReview(), CapexRequestApproval(), PortfolioReprioritisation(), MakeVsBuy(), PostInvestmentReview()],
    notes="Internal management decisions on projects and capital before or after the spend; no supplier bills, no purchase-order "
          "matching, no payment decisions. Figures are computed from generated cash flows so NPV, IRR and payback agree.",
)
