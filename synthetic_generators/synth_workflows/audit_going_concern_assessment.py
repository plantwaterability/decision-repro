"""Audit going-concern assessment: an external auditor decides whether a client can keep operating through the
look-forward period, and what that means for the accounts and the audit report.

Every state quotes the (fictional) audit firm's own written criteria -- the look-forward period, which funding
may be counted, how covenant breaches are classified, how management's plans are weighed, when the report is
modified and who must be consulted -- so each case is decidable in principle from the state alone. Companies,
lenders, people and numbers are invented. Amounts are internally consistent: ratios are computed from the
figures shown, and covenant tests from the definitions quoted.

Not accounts payable or procurement: no vendor bills, purchase orders or payment runs appear anywhere; the
decisions are audit judgements about the client's solvency and reporting.

Each schema also carries `intended(f)`: the answer the written criteria point to for each question (choice key,
'yes'/'no', or a score level index). It is never rendered and never used as a label; it exists so the answer
spread can be checked for collapsed questions before any teacher labels the data.
"""
from __future__ import annotations

import calendar
from datetime import date, timedelta

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
    lognormal_amount,
    near,
    noisy,
    noul,
    org,
    person,
    pick,
    pick_w,
    prune,
    regime,
    score,
    table,
)

CURRENCIES = ["USD", "USD", "EUR", "GBP", "CAD", "AUD"]
SYM = {"USD": "$", "EUR": "€", "GBP": "£", "CAD": "CA$", "AUD": "A$"}

INDUSTRIES = [
    "regional bus and coach operator", "specialty coffee roaster with a café estate", "contract electronics manufacturer",
    "pre-revenue medical-device developer", "boutique hotel group", "commercial printing business",
    "agricultural machinery dealer", "dental-practice software vendor", "independent cinema chain",
    "craft brewery with a taproom estate", "airport ground-handling contractor", "plastic packaging converter",
    "fitness-club operator", "furniture retailer with eleven showrooms", "offshore survey vessel operator",
    "residential care-home operator", "specialty chemicals blender", "event staging and lighting company",
    "frozen-food processor", "e-bike assembler and distributor", "regional newspaper publisher",
    "precision machining subcontractor", "garden-centre group", "ski-resort operator", "freight brokerage",
    "biotech platform company", "rooftop solar installer", "vocational training provider",
    "kitchen-appliance importer", "structural steel fabricator", "children's soft-play centre operator",
    "independent veterinary group", "luxury knitwear brand", "vertical-farming start-up",
]
PRESSURES = [
    "the loss of its largest customer", "input-cost inflation it could not pass on", "a delayed product approval",
    "a failed product launch", "higher interest costs on floating-rate debt", "a weak trading season",
    "a fall in volumes after a competitor's price cuts", "a costly site relocation", "a warranty-claims spike",
    "the end of a government support scheme", "a long-running dispute with its landlord",
    "slower-than-planned customer adoption", "a systems migration that overran badly",
    "two consecutive years of losses", "an expensive expansion that has not yet paid back",
]
NAME_A = ["Ashcombe", "Bramblewick", "Calderbrook", "Dunmarsh", "Eldersley", "Fallowmere", "Gorsefield", "Hollinwood",
          "Kestlemoor", "Larkhollow", "Millbeck", "Netherby", "Oxcombe", "Quarrendon", "Rookwood", "Saltcote",
          "Underhill", "Vexmoor", "Wrenfield", "Yewdale", "Amberlow", "Birchmere", "Coldharbour", "Dovecote",
          "Easterbrook", "Foxhollow", "Greywell", "Hartsway", "Kingsmere", "Linthorpe", "Marrowby", "Northcote",
          "Orchardleigh", "Pebblecombe", "Ravensby", "Stourleigh", "Tollgate", "Brightwater", "Copperfield",
          "Fennick", "Halloway", "Merriott", "Penhallow", "Sedgemoor", "Talbrook", "Westerleigh", "Alcester",
          "Branscombe", "Carrow", "Druridge"]
SUFFIX = ["", "", " Ltd", " plc", " LLC", " Inc", " Holdings", " Group", " & Co", " Enterprises"]
LENDERS = ["Northbank", "Coastline Commercial Bank", "Fairhaven Capital", "Granary Trust Bank", "Stonebridge Lending",
           "Larchmont Bank", "Bellweather Credit", "Union Quay Bank", "Pemberton Finance", "Cardale Bank"]


def _company(rng):
    return pick(rng, NAME_A) + pick(rng, SUFFIX)


def _fy_end(rng):
    y = pick(rng, [2025, 2025, 2026])
    m = pick(rng, [3, 6, 9, 12, 12, 12])
    return date(y, m, calendar.monthrange(y, m)[1])


def _amt_style(rng):
    return pick(rng, ["m", "k", "full", "k"])


def kamt(k, cur, style):
    """An amount held in THOUSANDS, written in one house style: $4.21m, $4,210k or $4,210,000."""
    neg = "-" if k < 0 else ""
    k = abs(k)
    s = SYM.get(cur, cur + " ")
    if style == "m":
        return f"{neg}{s}{k / 1000:,.2f}m"
    if style == "k":
        return f"{neg}{s}{k:,.0f}k"
    return f"{neg}{s}{k * 1000:,.0f}"


def _ds(rng):
    return pick(rng, ["iso", "long", "us", "eu", "iso"])


def _team(rng):
    return {"senior": person(rng), "manager": person(rng), "partner": person(rng)}


def _first(name):
    return name.split()[0]


# =========================================================================== schema 1
class LiquidityRunwayEvaluation(Schema):
    name = "liquidity_runway_evaluation"
    description = ("The auditor's look-forward liquidity test: does cash plus the funding the firm's policy allows "
                   "carry the client through the going-concern period?")
    surfaces = {"json": 0.5, "workpaper": 0.3, "senior_email": 0.2}

    def sample_factors(self, rng):
        reg = regime(rng)
        fy = _fy_end(rng)
        gap = rng.randint(50, 125)
        basis = pick(rng, ["report_date", "balance_sheet_date"])
        window = 12.0 + (round(gap / 30.4, 1) if basis == "report_date" else 0.0)
        burn = max(40, int(round(lognormal_amount(rng, 380, 0.7), -1)))
        downside_pct = pick(rng, [10, 15, 20, 25, 30])
        has_debt = chance(rng, 0.55)
        debt_due = int(round(burn * rng.uniform(2, 9), -1)) if has_debt else 0
        debt_month = rng.randint(2, int(window)) if has_debt else None
        refi = pick_w(rng, {"signed": 2, "term_sheet": 3, "in_discussion": 3, "not_started": 2}) if has_debt else None
        refi_copy_on_file = (chance(rng, 0.6) if refi == "signed" else None)
        has_fac = chance(rng, 0.6)
        fac = int(round(burn * rng.uniform(1, 6), -1)) if has_fac else 0
        fac_status = (pick_w(rng, {"committed": 4, "committed_conditional": 3, "uncommitted": 2, "unknown": 1})
                      if has_fac else None)
        bank_conf = chance(rng, 0.55) if has_fac else None
        equity = int(round(burn * rng.uniform(2, 6), -1)) if (not has_debt and chance(rng, 0.3)) else 0
        fstatus = pick_w(rng, {"board_approved": 8, "draft_unapproved": 2, "short_horizon": 1, "not_provided": 1})
        prior_miss = pick_w(rng, {"low": 3, "mid": 3, "high": 3, "none": 1})
        growth_last = round(rng.uniform(-14, 12), 1)
        growth_gap = pick_w(rng, {"aligned": 4, "stretched": 3, "aggressive": 3})
        if reg == "clear":
            good = chance(rng, 0.55)
            target = rng.uniform(1.45, 2.4) if good else rng.uniform(0.3, 0.75)
            fstatus = "board_approved"
            if fac_status == "unknown":
                fac_status = "committed"
            prior_miss = "low" if good else "high"
            growth_gap = "aligned" if good else pick(rng, ["stretched", "aggressive"])
        elif reg == "borderline":
            target = near(rng, 1.02, 0.2)
        else:
            target = rng.uniform(0.8, 1.4)
            # the forecasting record and the numbers disagree, or a key document is missing
            if chance(rng, 0.5):
                prior_miss, growth_gap = pick(rng, [("low", "aggressive"), ("high", "aligned")])
            if chance(rng, 0.3):
                fstatus = pick(rng, ["not_provided", "short_horizon", "draft_unapproved"])
        if reg != "clear" and chance(rng, 0.22):
            # rescued after the year end: a signed refinancing or a completed placing is what makes the numbers work
            if has_debt:
                refi, refi_copy_on_file = "signed", chance(rng, 0.7)
                debt_due = int(round(burn * rng.uniform(4, 9), -1))
            else:
                equity = int(round(burn * rng.uniform(3, 7), -1))
            target = rng.uniform(1.05, 1.4)
        cond_limit = cond_value = None
        if fac_status == "committed_conditional":
            cond_limit = pick(rng, [2.5, 3.0, 3.5, 4.0])
            cond_value = round(near(rng, cond_limit, 0.6, lo=0.8), 2)
            if cond_value == cond_limit:
                cond_value = round(cond_value + pick(rng, [-0.05, 0.05]), 2)
        miss_pct = {"low": rng.randint(1, 8), "mid": rng.randint(11, 22), "high": rng.randint(26, 48),
                    "none": None}[prior_miss]
        growth_assumed = round(growth_last + {"aligned": rng.uniform(-2, 4), "stretched": rng.uniform(7, 14),
                                              "aggressive": rng.uniform(18, 35)}[growth_gap], 1)
        counted_fac = fac if (fac_status == "committed" or
                              (fac_status == "committed_conditional" and cond_value < cond_limit)) else 0
        need = burn * window + (debt_due if (has_debt and refi != "signed") else 0)
        cash = target * need - counted_fac - equity
        cash = int(round(max(cash, burn * 0.6), -1))
        fh = rng.randint(6, int(window) - 2) if fstatus == "short_horizon" else (
            None if fstatus == "not_provided" else int(window + rng.randint(0, 6)))
        return {
            "regime": reg, "company": _company(rng), "industry": pick(rng, INDUSTRIES), "pressure": pick(rng, PRESSURES),
            "currency": pick(rng, CURRENCIES), "year_end": fy, "report_date": fy + timedelta(days=gap),
            "look_forward_basis": basis, "window_months": window, "cash_k": cash, "monthly_outflow_k": burn,
            "downside_pct": downside_pct, "debt_due_k": debt_due, "debt_due_month": debt_month,
            "refinancing": refi, "refinancing_copy_on_file": refi_copy_on_file,
            "facility_k": fac, "facility_status": fac_status, "facility_condition_limit": cond_limit,
            "facility_condition_value": cond_value, "bank_confirmation": bank_conf,
            "lender": pick(rng, LENDERS), "equity_after_year_end_k": equity,
            "forecast_status": fstatus, "forecast_horizon_months": fh, "prior_forecast_miss": prior_miss,
            "prior_forecast_miss_pct": miss_pct, "revenue_growth_last_year_pct": growth_last,
            "revenue_growth_assumed_pct": growth_assumed, "growth_gap": growth_gap,
            "recurring_losses": chance(rng, 0.6), "cfo_tone": pick(rng, ["confident", "confident", "measured", "worried"]),
            "team": _team(rng), "wp_ref": f"GC-{rng.randint(1, 4)}.{rng.randint(1, 9)}",
        }

    # ------------------------------------------------------------------ the arithmetic the criteria imply
    @staticmethod
    def calc(f):
        burn, w = f["monthly_outflow_k"], f["window_months"]
        st = f["facility_status"]
        counted = f["facility_k"] if (st == "committed" or (st == "committed_conditional" and
                                                            f["facility_condition_value"] < f["facility_condition_limit"])) else 0
        debt_counts = f["debt_due_k"] > 0 and f["refinancing"] != "signed"
        avail = f["cash_k"] + counted + f["equity_after_year_end_k"]
        need = burn * w + (f["debt_due_k"] if debt_counts else 0)
        down_burn = burn * (1 + f["downside_pct"] / 100)
        need_down = down_burn * w + (f["debt_due_k"] if debt_counts else 0)
        # runway: months until counted funds are exhausted, the maturity paid when it falls due if it can be
        if debt_counts:
            m = f["debt_due_month"]
            if avail - burn * m >= f["debt_due_k"]:
                runway = (avail - f["debt_due_k"]) / burn
            else:
                runway = min(avail / burn, m)
        else:
            runway = avail / burn
        # what the position would look like without the post-year-end funding (signed refinancing / placing)
        need_pre = burn * w + (f["debt_due_k"] if f["debt_due_k"] > 0 else 0)
        avail_pre = f["cash_k"] + counted
        return {"counted_facility": counted, "available": avail, "need": need, "ratio": avail / need,
                "ratio_down": avail / need_down, "runway": runway, "runway_ratio": runway / w,
                "ratio_without_post_ye": avail_pre / need_pre,
                "ratio_without_facility": (f["cash_k"] + f["equity_after_year_end_k"]) / need}

    @staticmethod
    def reliability(f):
        if f["forecast_status"] == "not_provided":
            return 0
        pts = {"low": 2, "mid": 1, "high": 0, "none": 0}[f["prior_forecast_miss"]]
        pts += {"aligned": 1, "stretched": 0, "aggressive": -1}[f["growth_gap"]]
        pts += {"board_approved": 1, "draft_unapproved": 0, "short_horizon": -1}[f["forecast_status"]]
        return 0 if pts <= 0 else 1 if pts == 1 else 2 if pts <= 3 else 3

    def intended(self, f):
        c = self.calc(f)
        rel = self.reliability(f)
        missing = f["forecast_status"] in ("not_provided", "short_horizon") or (
            f["facility_status"] == "unknown" and c["ratio"] < 1 <= (c["available"] + f["facility_k"]) / c["need"])
        if missing:
            concl = "cannot_conclude"
        elif c["ratio"] < 1:
            concl = "substantial_doubt"
        elif c["ratio_without_post_ye"] < 1 and (f["refinancing"] == "signed" or f["equity_after_year_end_k"]):
            concl = "doubt_alleviated"
        elif c["ratio_down"] < 1 and (rel <= 1 or c["ratio"] < 1.1):
            concl = "substantial_doubt"
        else:
            concl = "no_substantial_doubt"
        rr = c["runway_ratio"]
        runway = 0 if rr < 0.6 else 1 if rr < 1.0 else 2 if rr < 1.25 else 3
        if f["forecast_status"] in ("not_provided", "short_horizon", "draft_unapproved"):
            step = "obtain_approved_forecast"
        elif f["debt_due_k"] and (f["refinancing"] in ("term_sheet", "in_discussion") or
                                  (f["refinancing"] == "signed" and not f["refinancing_copy_on_file"])):
            step = "inspect_refinancing_terms"
        elif (f["facility_status"] in ("committed_conditional", "unknown") or
              (f["facility_k"] and not f["bank_confirmation"] and c["counted_facility"])) and \
                c["ratio_without_facility"] < 1.1 and f["facility_status"] != "uncommitted":
            step = "confirm_facility_with_lender"
        elif c["ratio"] < 1.3 or rel <= 1:
            step = "stress_test_assumptions"
        else:
            step = "no_further_work"
        return {"doubt_conclusion": concl, "runway_adequacy": runway, "forecast_reliability": rel,
                "downside_coverage": "yes" if c["ratio_down"] >= 1 else "no", "next_audit_step": step}

    # ------------------------------------------------------------------ rendering
    def _policy(self, f, rng):
        lf = ("twelve months from the date the auditor's report is expected to be signed"
              if f["look_forward_basis"] == "report_date" else "twelve months from the balance-sheet date")
        return {
            "GC-2": pick(rng, [f"The assessment period runs {lf}.", f"Management's assessment must cover {lf}."]),
            "GC-4": pick(rng, ["Count cash and committed facilities whose drawdown conditions are met today; "
                               "uncommitted lines do not count.",
                               "Only cash and committed borrowing facilities that could be drawn today may be treated "
                               "as available; uncommitted or on-demand lines are excluded."]),
            "GC-5": pick(rng, ["Debt falling due inside the period is an outflow unless a signed refinancing "
                               "agreement is on file.",
                               "Treat maturities within the period as repaid from available funds unless the "
                               "replacement facility has been signed."]),
            "GC-6": pick(rng, ["Consider whether the downside case in the file would exhaust funding within the period.",
                               "A plausible downside that exhausts funding inside the period is itself an indicator "
                               "of substantial doubt when the forecasting record is weak."]),
        }

    def _facility_text(self, f, cur, st):
        if not f["facility_k"]:
            return None
        base = f"{kamt(f['facility_k'], cur, st)} undrawn revolving facility with {f['lender']}"
        s = f["facility_status"]
        if s == "committed":
            return base + ", committed to maturity"
        if s == "committed_conditional":
            return (base + f", committed but drawable only while net leverage is below {f['facility_condition_limit']:.2f}x"
                    f" (latest certificate: {f['facility_condition_value']:.2f}x)")
        if s == "uncommitted":
            return base + ", uncommitted and repayable on demand"
        return base + "; facility letter not yet provided to the audit team"

    def _debt_text(self, f, cur, st, ds, rng):
        if not f["debt_due_k"]:
            return None
        due = f["year_end"] + timedelta(days=int(f["debt_due_month"] * 30.4))
        refi = {"signed": "replacement facility signed after the year end"
                          + ("" if f["refinancing_copy_on_file"] else " (copy of the signed agreement not yet on file)"),
                "term_sheet": "indicative term sheet received, not signed",
                "in_discussion": "refinancing discussions under way with two lenders",
                "not_started": "no refinancing discussions started yet"}[f["refinancing"]]
        return f"{kamt(f['debt_due_k'], cur, st)} term loan due {fmt_date(due, rng, ds)}; {refi}"

    def _forecast_text(self, f):
        s = f["forecast_status"]
        if s == "not_provided":
            return "Management has not yet provided a cash-flow forecast."
        h = f["forecast_horizon_months"]
        lead = {"board_approved": f"Board-approved monthly forecast covering {h} months from the year end.",
                "draft_unapproved": f"Draft forecast covering {h} months; not yet reviewed by the board.",
                "short_horizon": f"Management's forecast runs for only {h} months from the year end."}[s]
        return lead

    def _cfo_line(self, f, cur, st):
        mgmt_runway = (f["cash_k"] + f["facility_k"] + f["equity_after_year_end_k"]) / f["monthly_outflow_k"]
        t = f["cfo_tone"]
        if t == "confident" and mgmt_runway >= 12:
            return (f"The CFO describes the position as comfortable: \"we have roughly {mgmt_runway:.0f} months of "
                    f"runway on our numbers\".")
        if t == "worried":
            return "The CFO admitted cash will be tight around the loan maturity and asked whether a note will be needed."
        return "The CFO expects trading to stabilise and regards the forecast as prudent."

    def render(self, f, rng, surface):
        cur, st, ds = f["currency"], _amt_style(rng), _ds(rng)
        burn = f["monthly_outflow_k"]
        down = int(round(burn * (1 + f["downside_pct"] / 100), -1))
        fac = self._facility_text(f, cur, st)
        debt = self._debt_text(f, cur, st, ds, rng)
        fc = self._forecast_text(f)
        pol = self._policy(f, rng)
        miss = (None if f["prior_forecast_miss_pct"] is None else
                f"last year's actual net outflow came in {f['prior_forecast_miss_pct']}% above the forecast")
        eq = (f"Equity placing of {kamt(f['equity_after_year_end_k'], cur, st)} completed and received after the year end."
              if f["equity_after_year_end_k"] else None)
        cfo = self._cfo_line(f, cur, st)
        team = f["team"]
        if surface == "json":
            return dumps(prune({
                "engagement": {"entity": f["company"], "industry": f["industry"], "workpaper": f["wp_ref"],
                               "year_end": f["year_end"].isoformat(), "planned_report_date": f["report_date"].isoformat(),
                               "prepared_by": team["senior"]},
                "liquidity": {"amounts_in": f"{cur} thousands", "cash_at_year_end": f["cash_k"],
                              "forecast_monthly_net_outflow": burn,
                              "downside_monthly_net_outflow": down,
                              "downside_case": f"revenue {f['downside_pct']}% below plan",
                              "recurring_losses": f["recurring_losses"],
                              "equity_received_after_year_end": f["equity_after_year_end_k"] or None},
                "financing": {"revolving_facility": fac, "bank_confirmation_received": f["bank_confirmation"],
                              "term_debt": debt},
                "forecast": {"status": f["forecast_status"], "horizon_months": f["forecast_horizon_months"],
                             "revenue_growth_assumed_pct": f["revenue_growth_assumed_pct"],
                             "revenue_growth_last_year_pct": f["revenue_growth_last_year_pct"],
                             "prior_year_outflow_vs_forecast_pct": f["prior_forecast_miss_pct"]},
                "firm_policy": pol,
                "notes": noisy(join_sentences([f"Trading has been hit by {f['pressure']}.", cfo]), rng, 0.15),
            }))
        if surface == "workpaper":
            head = [f"{f['wp_ref']}  Going concern -- liquidity and look-forward funding",
                    f"Client: {f['company']} ({f['industry']})",
                    f"Year end: {fmt_date(f['year_end'], rng, ds)}    Planned sign-off: {fmt_date(f['report_date'], rng, ds)}",
                    f"Prepared: {team['senior']}    Reviewed: {pick(rng, [team['manager'], 'not yet reviewed'])}", "", ""]
            rows = [["Cash at year end", kamt(f["cash_k"], cur, st)],
                    ["Forecast net outflow / month", kamt(burn, cur, st)],
                    [f"Downside outflow / month (revenue -{f['downside_pct']}%)", kamt(down, cur, st)]]
            if f["equity_after_year_end_k"]:
                rows.append(["Equity received after year end", kamt(f["equity_after_year_end_k"], cur, st)])
            body = table(["Item", "Amount"], rows, rng)
            extra = bullets([x for x in [fac and "Facility: " + fac,
                                         None if f["bank_confirmation"] is None else
                                         ("Bank confirmation received." if f["bank_confirmation"] else
                                          "Bank confirmation requested, not yet returned."),
                                         debt and "Debt: " + debt, fc,
                                         miss and f"Forecasting record: {miss}.",
                                         f"Revenue growth assumed {f['revenue_growth_assumed_pct']:+.1f}% vs "
                                         f"{f['revenue_growth_last_year_pct']:+.1f}% achieved last year.",
                                         "Recurring losses in each of the last two years." if f["recurring_losses"] else None]
                             if x], rng)
            policy = "\n".join(f"  {k}: {v}" for k, v in pol.items())
            notes = noisy(join_sentences([f"Trading hit by {f['pressure']}.", cfo]), rng, 0.2)
            return "\n".join(head) + body + "\n\n" + extra + "\n\nFirm policy extract:\n" + policy + \
                "\n\nPreparer notes: " + notes
        # senior_email
        para1 = join_sentences([
            f"I've drafted {f['wp_ref']} for {f['company']}.",
            f"Year-end cash was {kamt(f['cash_k'], cur, st)} and the forecast shows net outflows of about "
            f"{kamt(burn, cur, st)} a month ({kamt(down, cur, st)} in the downside case, revenue {f['downside_pct']}% "
            f"below plan).",
            eq,
        ])
        para2 = join_sentences([fac and f"There is a {fac}.",
                                None if f["bank_confirmation"] is None else
                                ("The bank has confirmed it." if f["bank_confirmation"] else
                                 "We're still waiting on the bank confirmation."),
                                debt and f"On debt: {debt}."])
        para3 = join_sentences([fc, miss and f"For context, {miss}.",
                                f"Plan assumes revenue growth of {f['revenue_growth_assumed_pct']:+.1f}% against "
                                f"{f['revenue_growth_last_year_pct']:+.1f}% last year.", cfo])
        lf = pol["GC-2"]
        return "\n".join([
            f"Subject: {f['company']} -- going concern runway ({fmt_date(f['year_end'], rng, ds)} year end)",
            f"From: {team['senior']}", f"To: {team['manager']}", "",
            f"Hi {_first(team['manager'])},", "",
            noisy(para1, rng, 0.1), "", noisy(para2, rng, 0.1) if para2 else "", "", noisy(para3, rng, 0.15), "",
            f"Reminder of the firm rules: {lf} {pol['GC-4']} {pol['GC-5']}",
            f"Planned sign-off is {fmt_date(f['report_date'], rng, ds)}.", "",
            "Thanks,", _first(team["senior"]),
        ])

    def questions(self):
        return {
            "doubt_conclusion": choice("What should the engagement team conclude about the client's ability to "
                                       "continue as a going concern?", {
                "cannot_conclude": "A document the assessment depends on is missing or too short, so no conclusion "
                                   "can be reached yet.",
                "doubt_alleviated": "The position only holds because of funding secured after the year end; record "
                                    "the doubt as alleviated and make sure the conditions are disclosed.",
                "no_substantial_doubt": "Allowable funding covers the whole assessment period, and the downside does "
                                        "not change that; no substantial doubt arises.",
                "substantial_doubt": "Counting only what the policy allows, the client is unlikely to meet its "
                                     "obligations through the period; substantial doubt exists.",
            }),
            "runway_adequacy": score("How well does the base-case runway cover the assessment period?", [
                "Falls far short: countable funds run out well before the period ends.",
                "Falls short: countable funds run out before the period ends.",
                "Just covers it, with little to spare.",
                "Covers it comfortably, with a clear margin.",
            ]),
            "forecast_reliability": score("How far can the auditor rely on management's cash-flow forecast?", [
                "Not at all: it is missing, or contradicted by the company's own record.",
                "Little: past forecasts missed badly or the assumptions are stretched.",
                "Reasonably: some optimism, but broadly in line with recent performance.",
                "Substantially: approved, realistic and backed by an accurate forecasting record.",
            ]),
            "downside_coverage": noul(
                "Under the downside case in the file, countable funding still lasts to the end of the assessment period.",
                yes="With downside outflows, cash plus the funding the policy lets you count meets every outflow and "
                    "maturity to the end of the period.",
                no="With downside outflows, countable funding runs out, or a maturity cannot be met, before the period ends."),
            "next_audit_step": choice("What should the audit team do next on going concern?", {
                "confirm_facility_with_lender": "Confirm the undrawn facility's terms and drawdown conditions directly "
                                                "with the lender.",
                "inspect_refinancing_terms": "Obtain and read the refinancing documents for the debt that falls due "
                                             "within the period.",
                "no_further_work": "No further going-concern procedures; document the conclusion and move on.",
                "obtain_approved_forecast": "Ask for a board-approved forecast that runs to the end of the assessment period.",
                "stress_test_assumptions": "Re-run the forecast with harsher assumptions and test them against recent trading.",
            }),
        }


# =========================================================================== schema 2
ADDBACK_KINDS = ["restructuring costs", "one-off legal fees", "pro forma synergies from an acquisition",
                 "site closure costs", "costs of a failed product launch", "management-reorganisation costs",
                 "an inventory write-down", "consultancy fees on a systems project"]


class CovenantBreachAssessment(Schema):
    name = "covenant_breach_assessment"
    description = ("A financial-covenant problem at or after the year end: is there a breach, how is the loan "
                   "classified, and how much pressure is the client under?")
    surfaces = {"json": 0.5, "covenant_calc": 0.25, "review_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        fy = _fy_end(rng)
        scenario = pick_w(rng, {"ye_breach": 35, "addback_dispute": 20, "projected_breach": 25, "compliant": 20})
        binding = pick_w(rng, {"leverage": 3, "interest_cover": 1})
        lev_max = pick(rng, [2.5, 2.75, 3.0, 3.25, 3.5, 4.0])
        ic_min = pick(rng, [2.0, 2.5, 3.0, 3.5])
        if reg == "clear":
            t = {"ye_breach": rng.uniform(1.15, 1.35), "addback_dispute": rng.uniform(1.12, 1.3),
                 "projected_breach": rng.uniform(0.75, 0.88), "compliant": rng.uniform(0.5, 0.72)}[scenario]
        elif reg == "borderline":
            t = {"ye_breach": rng.uniform(1.01, 1.07), "addback_dispute": rng.uniform(1.01, 1.08),
                 "projected_breach": rng.uniform(0.9, 0.98), "compliant": rng.uniform(0.86, 0.96)}[scenario]
        else:
            t = {"ye_breach": rng.uniform(1.02, 1.2), "addback_dispute": rng.uniform(1.02, 1.15),
                 "projected_breach": rng.uniform(0.85, 0.97), "compliant": rng.uniform(0.7, 0.93)}[scenario]
        e_agree = int(round(lognormal_amount(rng, 6500, 0.7), -1))
        if binding == "leverage":
            # interest follows from a realistic borrowing rate; cover stays comfortably above its floor
            lev = lev_max * t
            rate = rng.uniform(0.05, 0.09)
            ic = 1 / (lev * rate)
            if ic < ic_min * 1.25:
                ic = ic_min * rng.uniform(1.25, 1.8)
        else:
            # interest cover only binds for a highly geared borrower paying a high rate
            ic_min = pick(rng, [2.0, 2.5])
            t = min(t, 1.15)
            ic = ic_min / t
            lev = 1 / (ic * rng.uniform(0.09, 0.12))
            lev_max = int((lev / rng.uniform(0.78, 0.93)) * 4 + 1) / 4      # leverage limit set wide of the actual
        net_debt = int(round(lev * e_agree, -1))
        interest = int(round(e_agree / ic, -1))
        # add-backs: claimed by management; the agreement may allow none, some or all of them
        addback_kind = pick(rng, ADDBACK_KINDS)
        clause = pick_w(rng, {"none_permitted": 2, "capped": 2, "permitted": 1})
        cap_pct = pick(rng, [5, 10]) if clause == "capped" else None
        if scenario == "addback_dispute":
            t_m = rng.uniform(0.85, 0.97)
            e_mgmt = (net_debt / (lev_max * t_m)) if binding == "leverage" else (ic_min / t_m) * interest
            extra = max(10, int(round(e_mgmt - e_agree, -1)))
            if clause == "permitted":
                clause = pick(rng, ["none_permitted", "capped"])
                cap_pct = pick(rng, [5, 10]) if clause == "capped" else None
            if clause == "none_permitted":
                allowed = 0
                e_rep = e_agree
            else:
                e_rep = int(round(e_agree / (1 + cap_pct / 100), -1))
                allowed = e_agree - e_rep
            claimed = allowed + extra
        else:
            e_rep = e_agree
            claimed = allowed = 0
            if chance(rng, 0.3):
                clause = "permitted"
                cap_pct = None
                claimed = allowed = int(round(e_agree * rng.uniform(0.03, 0.12), -1))
                e_rep = e_agree - claimed
        # next quarterly / half-year test: projection on the agreement's definition
        months_next = pick(rng, [3, 3, 6])
        if scenario == "projected_breach":
            t_next = rng.uniform(1.03, 1.2) if reg == "clear" else rng.uniform(0.96, 1.08)
        elif scenario == "compliant":
            t_next = t * rng.uniform(0.92, 1.06)
        else:
            t_next = t * rng.uniform(0.72, 1.02)
        t_next = round(t_next, 3)
        if abs(t_next - 1.0) < 0.004:
            t_next = 1.01
        breach_ye = t > 1
        if breach_ye:
            waiver = pick_w(rng, {"none": 2, "requested": 2, "declined": 2, "waiver_of_breach": 3,
                                  "covenant_holiday": 2, "verbal_only": 2})
        else:
            waiver = pick_w(rng, {"none": 6, "covenant_holiday": 1})
        if scenario == "addback_dispute" and chance(rng, 0.5):
            waiver = "none"            # management believes it complied, so nobody asked
        waiver_day = None
        if waiver in ("waiver_of_breach", "covenant_holiday"):
            waiver_day = rng.randint(-25, -1) if chance(rng, 0.45) else rng.randint(5, 70)
        rule = pick(rng, ["by_issue", "by_reporting_date"])
        other_debt = int(round(net_debt * rng.uniform(0.15, 0.6), -1)) if chance(rng, 0.75) else 0
        cross_default = bool(other_debt) and chance(rng, 0.75)
        return {"regime": reg, "company": _company(rng), "industry": pick(rng, INDUSTRIES), "currency": pick(rng, CURRENCIES),
                "lender": pick(rng, LENDERS), "year_end": fy, "report_date": fy + timedelta(days=rng.randint(55, 110)),
                "scenario": scenario, "binding_covenant": binding, "leverage_max": lev_max, "interest_cover_min": ic_min,
                "ebitda_reported_k": e_rep, "addback_kind": addback_kind, "addback_claimed_k": claimed,
                "addback_allowed_k": allowed, "addback_clause": clause, "addback_cap_pct": cap_pct,
                "net_debt_k": net_debt, "net_interest_k": interest, "tightness_ye": round(t, 4),
                "months_to_next_test": months_next, "tightness_next": t_next, "waiver": waiver,
                "waiver_days_after_year_end": waiver_day, "classification_rule": rule,
                "other_borrowings_k": other_debt, "cross_default_clause": cross_default,
                "lender_stance": pick(rng, ["supportive", "supportive", "neutral", "reserving rights"]),
                "team": _team(rng), "loan_name": pick(rng, ["Term Loan A", "senior term facility", "acquisition facility",
                                                            "amortising term loan", "Facility B"])}

    @staticmethod
    def calc(f):
        e_agree = f["ebitda_reported_k"] + f["addback_allowed_k"]
        e_mgmt = f["ebitda_reported_k"] + f["addback_claimed_k"]
        lev = f["net_debt_k"] / e_agree
        ic = e_agree / f["net_interest_k"]
        lev_m = f["net_debt_k"] / e_mgmt
        ic_m = e_mgmt / f["net_interest_k"]
        breach = lev > f["leverage_max"] or ic < f["interest_cover_min"]
        mgmt_breach = lev_m > f["leverage_max"] or ic_m < f["interest_cover_min"]
        tn = f["tightness_next"]
        head_next = 1 - tn         # headroom on the binding covenant at the next test (share of the limit)
        wd = f["waiver_days_after_year_end"]
        waiver_valid = f["waiver"] in ("waiver_of_breach", "covenant_holiday") and (
            f["classification_rule"] == "by_issue" or (wd is not None and wd < 0))
        return {"lev": lev, "ic": ic, "lev_mgmt": lev_m, "ic_mgmt": ic_m, "breach": breach,
                "mgmt_breach": mgmt_breach, "next_breach": tn > 1, "head_next": head_next, "waiver_valid": waiver_valid}

    def intended(self, f):
        c = self.calc(f)
        w = f["waiver"]
        if c["breach"]:
            if w == "requested":
                cls = "await_lender_response"
            elif not c["waiver_valid"]:
                cls = "current"
            elif w == "waiver_of_breach" and c["next_breach"]:
                cls = "current"
            else:
                cls = "noncurrent_disclose_risk"
        else:
            cls = "noncurrent_disclose_risk" if (c["next_breach"] or c["head_next"] < 0.1) else "noncurrent"
        h = c["head_next"]
        if w == "covenant_holiday" and c["waiver_valid"]:
            h = max(h, 0.3)          # tests suspended for twelve months
        head = 0 if h < 0 else 1 if h < 0.10 else 2 if h < 0.25 else 3
        if c["breach"] and not c["waiver_valid"]:
            pressure = 3
        elif (c["next_breach"] and not (w == "covenant_holiday" and c["waiver_valid"])) or (
                c["breach"] and w == "requested"):
            pressure = 2 if f["months_to_next_test"] <= 3 else 1
        elif h < 0.1:
            pressure = 1
        else:
            pressure = 0
        live = c["breach"] and not c["waiver_valid"]
        expected = c["next_breach"] and not (w == "covenant_holiday" and c["waiver_valid"])
        cross = "yes" if (f["cross_default_clause"] and (live or expected)) else "no"
        return {"loan_classification": cls, "covenant_headroom": head, "waiver_deadline_pressure": pressure,
                "breach_at_reporting_date": "yes" if c["breach"] else "no", "cross_default_exposure": cross}

    def _rule_text(self, f):
        when = ("before the financial statements are authorised for issue" if f["classification_rule"] == "by_issue"
                else "on or before the reporting date")
        return {
            "DC-1": "Covenants are measured exactly as the loan agreement defines them, not on management's "
                    "adjusted figures.",
            "DC-2": f"A loan in breach at the reporting date is a current liability unless the lender, {when}, has "
                    f"given up in writing its right to demand repayment arising from that breach.",
            "DC-3": "If a breach is waived but a further breach at a test date within twelve months is probable and "
                    "the lender's test would apply, the loan is still current. Covenant holidays suspend those tests.",
            "DC-4": "A loan that complies at the reporting date stays non-current; disclose the risk where headroom "
                    "at a coming test is thin or a breach is forecast.",
        }

    def _clause_text(self, f):
        if f["addback_clause"] == "none_permitted":
            return "Consolidated EBITDA: as reported, with no adjustment for exceptional or non-recurring items."
        if f["addback_clause"] == "capped":
            return (f"Consolidated EBITDA: as reported, plus non-recurring items not exceeding {f['addback_cap_pct']}% "
                    f"of reported EBITDA in any test period.")
        return "Consolidated EBITDA: as reported, plus non-recurring items evidenced to the lender's satisfaction."

    def _waiver_text(self, f, rng, ds):
        w = f["waiver"]
        if w == "none":
            return None
        d = (f["year_end"] + timedelta(days=f["waiver_days_after_year_end"])
             if f["waiver_days_after_year_end"] is not None else None)
        return {"requested": f"Waiver requested from {f['lender']}; credit committee decision awaited.",
                "declined": f"{f['lender']} declined the waiver request and has reserved its rights.",
                "verbal_only": f"Relationship manager at {f['lender']} said on a call the bank 'will not act on it'; "
                               f"nothing in writing.",
                "waiver_of_breach": f"Signed letter from {f['lender']} dated {fmt_date(d, rng, ds)} waiving the "
                                    f"breach at the year-end test; later tests apply as normal.",
                "covenant_holiday": f"Signed amendment from {f['lender']} dated {fmt_date(d, rng, ds)} waiving the "
                                    f"year-end test and suspending covenant tests for the following twelve months."}[w]

    def render(self, f, rng, surface):
        cur, st, ds = f["currency"], _amt_style(rng), _ds(rng)
        c = self.calc(f)
        rules = self._rule_text(f)
        nt = f["year_end"] + timedelta(days=int(f["months_to_next_test"] * 30.4))
        if f["binding_covenant"] == "leverage":
            proj = f"projected net leverage at the {fmt_date(nt, rng, ds)} test: {f['leverage_max'] * f['tightness_next']:.2f}x"
        else:
            proj = (f"projected interest cover at the {fmt_date(nt, rng, ds)} test: "
                    f"{f['interest_cover_min'] / f['tightness_next']:.2f}x")
        cert = (f"Management's compliance certificate: leverage {c['lev_mgmt']:.2f}x, interest cover {c['ic_mgmt']:.2f}x"
                + (f" (after adding back {kamt(f['addback_claimed_k'], cur, st)} of {f['addback_kind']})"
                   if f["addback_claimed_k"] else "")
                + (" -- shown as compliant" if not c["mgmt_breach"] else " -- shown as a breach"))
        waiver = self._waiver_text(f, rng, ds)
        other = (f"Other borrowings of {kamt(f['other_borrowings_k'], cur, st)}"
                 + (" contain a cross-default clause referencing this loan." if f["cross_default_clause"]
                    else " have no cross-default clause.")) if f["other_borrowings_k"] else "No other borrowings."
        stance = {"supportive": f"{f['lender']} has been described by management as supportive.",
                  "neutral": f"No recent correspondence with {f['lender']} beyond routine reporting.",
                  "reserving rights": f"{f['lender']} sent a letter reserving its rights after the year end."}[f["lender_stance"]]
        team = f["team"]
        if surface == "json":
            return dumps(prune({
                "client": {"entity": f["company"], "industry": f["industry"], "year_end": f["year_end"].isoformat(),
                           "planned_report_date": f["report_date"].isoformat()},
                "loan": {"facility": f["loan_name"], "lender": f["lender"],
                         "covenants": {"max_net_debt_to_ebitda": f["leverage_max"],
                                       "min_ebitda_to_net_interest": f["interest_cover_min"]},
                         "ebitda_definition": self._clause_text(f)},
                "year_end_figures": {"amounts_in": f"{cur} thousands", "reported_ebitda": f["ebitda_reported_k"],
                                     "management_addbacks": f["addback_claimed_k"] or None,
                                     "addback_description": f["addback_kind"] if f["addback_claimed_k"] else None,
                                     "net_debt": f["net_debt_k"], "net_interest": f["net_interest_k"]},
                "outlook": {"next_test_in_months": f["months_to_next_test"], "projection": proj,
                            "management_certificate": cert},
                "lender_position": {"waiver": waiver, "stance": stance},
                "other_debt": other,
                "classification_policy": rules,
            }))
        if surface == "covenant_calc":
            rows = [["Reported EBITDA", kamt(f["ebitda_reported_k"], cur, st)],
                    ["Add-backs claimed", kamt(f["addback_claimed_k"], cur, st) if f["addback_claimed_k"] else "none"],
                    ["Net debt", kamt(f["net_debt_k"], cur, st)], ["Net interest", kamt(f["net_interest_k"], cur, st)],
                    ["Max leverage (agreement)", f"{f['leverage_max']:.2f}x"],
                    ["Min interest cover (agreement)", f"{f['interest_cover_min']:.2f}x"]]
            return "\n".join([
                f"COVENANT TEST -- {f['company']} -- year end {fmt_date(f['year_end'], rng, ds)}",
                f"Facility: {f['loan_name']} ({f['lender']})", "",
                table(["Line", "Value"], rows, rng), "",
                "Agreement definition: " + self._clause_text(f),
                cert + ".", "Outlook: " + proj + ".",
                "Lender: " + (waiver or "no waiver requested or received.") + " " + stance,
                other, "",
                "Classification rules (firm manual):",
                "\n".join(f"  {k} {v}" for k, v in rules.items()),
                f"Planned report date: {fmt_date(f['report_date'], rng, ds)}",
            ])
        # review_note
        notes = [
            f"RN-{rng.randint(1, 30)} on {f['company']} debt classification (preparer {team['senior']}).",
            f"The {f['loan_name']} with {f['lender']} carries a maximum net debt / EBITDA of {f['leverage_max']:.2f}x "
            f"and a minimum interest cover of {f['interest_cover_min']:.2f}x.",
            f"Year-end reported EBITDA {kamt(f['ebitda_reported_k'], cur, st)}, net debt {kamt(f['net_debt_k'], cur, st)}, "
            f"net interest {kamt(f['net_interest_k'], cur, st)}.",
            cert + ".", self._clause_text(f), "Looking ahead, " + proj + ".",
            waiver or "There is no waiver on file.", stance, other,
        ]
        return "\n".join([
            f"REVIEW NOTE -- from {team['manager']} to {team['senior']}", "",
            noisy(join_sentences(notes), rng, 0.1), "",
            "Firm rules to apply:", "\n".join(f"- {k}: {v}" for k, v in rules.items()), "",
            f"Please clear before sign-off on {fmt_date(f['report_date'], rng, ds)}.",
        ])

    def questions(self):
        return {
            "loan_classification": choice("How should the loan be presented in the year-end balance sheet?", {
                "await_lender_response": "Hold the classification open until the lender decides on the pending "
                                         "waiver request.",
                "current": "Present the whole loan as a current liability.",
                "noncurrent": "Keep it in non-current liabilities with only the usual covenant disclosure.",
                "noncurrent_disclose_risk": "Keep it non-current, but disclose the breach, waiver or thin headroom "
                                            "and the risk at coming tests.",
            }),
            "covenant_headroom": score("How much room is there on the tightest covenant at the next test date?", [
                "None: a breach is projected at the next test.",
                "Thin: within about a tenth of the limit.",
                "Moderate: roughly a tenth to a quarter of the limit to spare.",
                "Ample: more than a quarter of the limit to spare, or tests suspended.",
            ]),
            "waiver_deadline_pressure": score("How soon must the client secure lender relief to avoid the debt "
                                              "becoming repayable?", [
                "No deadline within the coming year.",
                "Before a test date later in the year.",
                "Before the next quarterly test, a few months away.",
                "Already overdue: the lender can demand repayment now.",
            ]),
            "breach_at_reporting_date": noul(
                "Measured the way the loan agreement defines it, a covenant was breached at the reporting date.",
                yes="On the agreement's own EBITDA definition, a ratio at the year end was outside its limit.",
                no="On the agreement's definition, both ratios were within their limits at the year end."),
            "cross_default_exposure": noul(
                "Through cross-default, the client's other borrowings are exposed to early repayment, either now or "
                "at the next covenant test."),
        }


# =========================================================================== schema 3
PLAN_TYPES = {
    # type: (statuses strong -> weak, evidence key)
    "asset_sale": (["signed_sale_agreement", "heads_of_terms", "marketing"], "signed_sale_agreement"),
    "sale_leaseback": (["signed_sale_agreement", "offer_received", "exploring"], "signed_sale_agreement"),
    "equity_raise": (["underwritten", "soft_commitments", "planned"], "investor_commitments"),
    "cost_programme": (["implemented", "announced", "proposed"], "savings_delivery_tracking"),
    "parent_support": (["binding_with_capacity", "binding_capacity_unclear", "comfort_letter"], "parent_capacity_evidence"),
    "lender_deferral": (["signed", "credit_committee", "requested"], "lender_written_agreement"),
}
STATUS_WEIGHT = [0.9, 0.55, 0.2]
ASSETS = ["the freehold warehouse", "a surplus office building", "a non-core subsidiary", "surplus plant and equipment",
          "a minority stake in a joint venture", "the head-office site", "the former production site",
          "an unused development plot", "the regional depot", "the investment property", "a second-hand vehicle fleet",
          "the company's share of a property partnership"]


class ManagementPlansEvaluation(Schema):
    name = "management_plans_evaluation"
    description = ("Management's plans to close a forecast funding gap (disposals, equity, cost cuts, parent support, "
                   "lender deferrals), evaluated against the firm's going-concern criteria.")
    surfaces = {"json": 0.5, "plans_memo": 0.25, "evaluation_workpaper": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        shortfall = int(round(lognormal_amount(rng, 3500, 0.7), -1))
        window = 12
        n = pick_w(rng, {1: 3, 2: 4, 3: 3})
        types = rng.sample(list(PLAN_TYPES), n)
        if reg == "clear":
            good = chance(rng, 0.55)
            cover = rng.uniform(1.25, 1.8) if good else rng.uniform(0.45, 0.75)
            lvl_w = [6, 2, 1] if good else [1, 2, 6]
        elif reg == "borderline":
            cover = near(rng, 1.03, 0.15)
            lvl_w = [3, 4, 3]
        else:
            cover = rng.uniform(0.85, 1.5)
            lvl_w = [3, 2, 4]
        shares = [rng.uniform(0.5, 1.5) for _ in types]
        tot = sum(shares)
        plans = []
        for tp, sh in zip(types, shares):
            lvl = pick(rng, [0, 1, 2], lvl_w)
            amt = int(round(shortfall * cover * sh / tot, -1))
            month = rng.randint(2, 11)
            late = chance(rng, 0.12)
            if late:
                month = rng.randint(13, 16)      # lands after the assessment period ends
            docs = chance(rng, 0.8 if lvl == 0 else 0.45 if lvl == 1 else 0.15)
            plans.append({"type": tp, "status_level": lvl, "status": PLAN_TYPES[tp][0][lvl], "amount_k": amt,
                          "expected_month": month, "documents_on_file": docs,
                          "asset": pick(rng, ASSETS) if tp in ("asset_sale", "sale_leaseback") else None})
        outstanding = None
        if reg == "conflicting" or chance(rng, 0.15):
            big = max(plans, key=lambda p: p["amount_k"])
            if not big["documents_on_file"] and big["status_level"] <= 1:
                outstanding = big["type"]
        return {"regime": reg, "company": _company(rng), "industry": pick(rng, INDUSTRIES), "pressure": pick(rng, PRESSURES),
                "currency": pick(rng, CURRENCIES), "year_end": _fy_end(rng), "window_months": window,
                "shortfall_k": shortfall, "plans": plans, "board_approved_plans": chance(rng, 0.75),
                "track_record_pct": pick(rng, [35, 50, 60, 70, 80, 90, 100]),
                "outstanding_promised": outstanding, "parent": pick(rng, NAME_A) + " Holdings",
                "team": _team(rng), "mgmt_tone": pick(rng, ["upbeat", "upbeat", "matter-of-fact", "defensive"])}

    @staticmethod
    def calc(f):
        inside = [p for p in f["plans"] if p["expected_month"] <= f["window_months"]]
        total = sum(p["amount_k"] for p in inside)
        allamt = sum(p["amount_k"] for p in f["plans"]) or 1
        wavg = sum(p["amount_k"] * STATUS_WEIGHT[p["status_level"]] for p in inside) / max(total, 1)
        adj = 0.1 if f["track_record_pct"] >= 80 else -0.1 if f["track_record_pct"] < 55 else 0.0
        adj += 0 if f["board_approved_plans"] else -0.1
        prob = wavg + adj
        evidenced = sum(p["amount_k"] for p in f["plans"] if p["documents_on_file"]) / allamt
        unproven = {}
        for p in f["plans"]:
            key = PLAN_TYPES[p["type"]][1]
            gap = p["amount_k"] * (1 - STATUS_WEIGHT[p["status_level"]]) * (0.6 if p["documents_on_file"] else 1.0)
            unproven[key] = unproven.get(key, 0) + gap
        return {"inside_total": total, "closes": total >= f["shortfall_k"], "prob": prob, "evidenced": evidenced,
                "priority": max(sorted(unproven), key=lambda k: unproven[k])}

    def intended(self, f):
        c = self.calc(f)
        p = c["prob"]
        prob = 0 if p < 0.35 else 1 if p < 0.6 else 2 if p < 0.8 else 3
        e = c["evidenced"]
        ev = 0 if e < 0.25 else 1 if e < 0.5 else 2 if e < 0.8 else 3
        if not c["closes"]:
            concl = "plans_insufficient"
        elif f["outstanding_promised"] or not f["board_approved_plans"]:
            concl = "evaluation_incomplete"
        elif prob >= 2:
            concl = "doubt_alleviated"
        else:
            concl = "material_uncertainty_remains"
        return {"plans_conclusion": concl, "implementation_probability": prob, "plan_evidence_quality": ev,
                "plans_close_gap": "yes" if c["closes"] else "no", "priority_evidence": c["priority"]}

    STATUS_TEXT = {
        "signed_sale_agreement": "sale agreement signed, completion conditional only on routine filings",
        "heads_of_terms": "non-binding heads of terms agreed with a buyer",
        "marketing": "being marketed through an agent; no offers yet",
        "offer_received": "written offer received from an investor, subject to due diligence",
        "exploring": "management exploring interest from investors",
        "underwritten": "placing fully underwritten by the company's broker",
        "soft_commitments": "soft commitments from existing shareholders, nothing signed",
        "planned": "raise planned for later in the year; no investors approached yet",
        "implemented": "programme implemented; first months' savings visible in management accounts",
        "announced": "programme announced to staff; consultations under way",
        "proposed": "proposal prepared for the board; not yet approved",
        "binding_with_capacity": "binding support letter from the parent; parent's accounts show ample liquidity",
        "binding_capacity_unclear": "binding support letter from the parent; the parent's own finances have not been seen",
        "comfort_letter": "letter of comfort from the parent stating it 'intends' to support; not legally binding",
        "signed": "deferral of scheduled repayments signed by the lender",
        "credit_committee": "deferral recommended by the relationship team; awaiting the lender's credit committee",
        "requested": "deferral requested by management; no response from the lender",
    }
    TYPE_TEXT = {"asset_sale": "Disposal of {asset}", "sale_leaseback": "Sale and leaseback of {asset}",
                 "equity_raise": "Equity raise", "cost_programme": "Cost-reduction programme",
                 "parent_support": "Financial support from {parent}", "lender_deferral": "Deferral of loan repayments"}
    OUTSTANDING = {"asset_sale": "the signed sale agreement", "sale_leaseback": "the investor's signed offer",
                   "equity_raise": "the broker's underwriting letter", "cost_programme": "the savings tracker",
                   "parent_support": "the parent's latest accounts", "lender_deferral": "the lender's written agreement"}

    def _plan_line(self, p, f, cur, st, rng, ds):
        title = self.TYPE_TEXT[p["type"]].format(asset=p["asset"] or "", parent=f["parent"])
        when = f["year_end"] + timedelta(days=int(p["expected_month"] * 30.4))
        return (title, kamt(p["amount_k"], cur, st), fmt_date(when, rng, ds), self.STATUS_TEXT[p["status"]],
                "on file" if p["documents_on_file"] else "not seen")

    def _policy(self, rng):
        return {
            "GC-8": "Management's plans may be taken into account only if it is probable they will be implemented "
                    "within the assessment period and probable that, once implemented, they will mitigate the conditions.",
            "GC-9": pick(rng, ["Amounts expected after the end of the assessment period are ignored.",
                               "Only cash expected to arrive within the twelve-month assessment period counts."]),
            "GC-10": pick(rng, ["Non-binding letters, unapproved proposals and unsolicited interest carry little weight.",
                                "Weight plans by the evidence: signed agreements high, intentions and proposals low."]),
        }

    def render(self, f, rng, surface):
        cur, st, ds = f["currency"], _amt_style(rng), _ds(rng)
        lines = [self._plan_line(p, f, cur, st, rng, ds) for p in f["plans"]]
        pol = self._policy(rng)
        out = (f"Management says {self.OUTSTANDING[f['outstanding_promised']]} will follow 'in the next few days'."
               if f["outstanding_promised"] else None)
        team = f["team"]
        if surface == "json":
            return dumps(prune({
                "client": {"entity": f["company"], "industry": f["industry"], "year_end": f["year_end"].isoformat(),
                           "assessment_period_months": f["window_months"]},
                "forecast_gap": {"amounts_in": f"{cur} thousands", "base_case_shortfall": f["shortfall_k"],
                                 "cause": f["pressure"]},
                "management_plans": [{"plan": t, "amount": p["amount_k"], "expected_cash_date": d, "status": s,
                                      "supporting_documents": e} for (t, _, d, s, e), p in zip(lines, f["plans"])],
                "governance": {"plans_approved_by_board": f["board_approved_plans"],
                               "share_of_past_plans_delivered_pct": f["track_record_pct"],
                               "open_request": out},
                "firm_policy": pol,
            }))
        if surface == "plans_memo":
            tone = {"upbeat": "We are confident these actions remove any doubt about the company's future.",
                    "matter-of-fact": "The actions below are expected to close the gap shown in the forecast.",
                    "defensive": "We do not accept that the forecast gap is a going-concern issue, but set out our "
                                 "actions for completeness."}[f["mgmt_tone"]]
            body = [f"MEMO -- {f['company']} -- management's plans", f"To: {team['partner']} (audit partner)",
                    f"From: Chief Financial Officer", "",
                    noisy(join_sentences([f"Following {f['pressure']}, our base-case forecast shows a funding "
                                          f"shortfall of {kamt(f['shortfall_k'], cur, st)} over the next "
                                          f"{f['window_months']} months.", tone]), rng, 0.1), ""]
            for i, (t, a, d, s, e) in enumerate(lines, 1):
                body.append(f"{i}. {t}: {a}, expected {d}. Status: {s}.")
            body += ["", ("The board approved these plans at its last meeting." if f["board_approved_plans"]
                          else "The plans will go to the board next month."),
                     out or ""]
            body += ["", "[Auditor annotation] Firm policy: " + " ".join(pol.values()),
                     f"[Auditor annotation] Documents on file: " +
                     ", ".join(f"plan {i} {e}" for i, (_, _, _, _, e) in enumerate(lines, 1)) +
                     f". Past plans delivered: {f['track_record_pct']}%."]
            return "\n".join(x for x in body if x is not None)
        rows = [[t, a, d, e] for (t, a, d, s, e) in lines]
        detail = bullets([f"{t}: {s}." for (t, a, d, s, e) in lines], rng)
        return "\n".join([
            f"GC-3.1 Evaluation of management's plans -- {f['company']}",
            f"Year end {fmt_date(f['year_end'], rng, ds)}; assessment period {f['window_months']} months; "
            f"preparer {team['senior']}", "",
            f"Base-case shortfall to be covered: {kamt(f['shortfall_k'], cur, st)}", "",
            table(["Plan", "Amount", "Expected", "Evidence"], rows, rng), "", detail, "",
            f"Board approval of plans: {'yes' if f['board_approved_plans'] else 'not yet'}. "
            f"Share of previous plans delivered on time: {f['track_record_pct']}%.",
            out or "", "", "Criteria:", "\n".join(f"  {k}: {v}" for k, v in pol.items()),
        ])

    def questions(self):
        return {
            "plans_conclusion": choice("What should the auditor conclude after evaluating management's plans?", {
                "doubt_alleviated": "The plans are probable of implementation and close the gap; the doubt is "
                                    "alleviated, with the conditions and plans disclosed.",
                "evaluation_incomplete": "The evaluation cannot be finished: approval or a document the main plan "
                                         "rests on is still outstanding.",
                "material_uncertainty_remains": "The plans would close the gap but are not probable enough; a "
                                                "material uncertainty remains.",
                "plans_insufficient": "Even if every plan landed on time, the gap within the period would not close.",
            }),
            "implementation_probability": score("How probable is it that the plans will be carried out as described "
                                                "within the period?", [
                "Speculative: mostly intentions and proposals.",
                "Possible: under way, but key steps are outside management's control.",
                "Probable: mainly agreed or signed, with a credible delivery record.",
                "Near-certain: signed, approved and largely already happening.",
            ]),
            "plan_evidence_quality": score("How well is the funding the plans promise backed by documents the "
                                           "auditor has actually seen?", [
                "Barely: almost none of it is documented.",
                "Partly: under half of the value is documented.",
                "Mostly: most of the value is documented.",
                "Fully: nearly all of the value is documented by third parties.",
            ]),
            "plans_close_gap": noul(
                "If carried out as described, the plans would bring in enough cash within the period to close the "
                "forecast shortfall.",
                yes="The plan amounts expected inside the assessment period add up to at least the shortfall.",
                no="Counting only cash expected inside the period, the plans fall short of the shortfall."),
            "priority_evidence": choice("Which evidence should the auditor chase first?", {
                "investor_commitments": "Signed subscription or underwriting commitments for the equity raise.",
                "lender_written_agreement": "The lender's written, signed agreement to defer repayments.",
                "parent_capacity_evidence": "The parent's latest accounts and its legal commitment to fund the client.",
                "savings_delivery_tracking": "Tracking that shows the cost savings are actually being delivered.",
                "signed_sale_agreement": "A signed agreement for the asset sale or sale and leaseback.",
            }),
        }


# =========================================================================== schema 4
# event: (direction, adjusting by default?, procedure days, text template)
EVENTS = {
    "customer_insolvency": ("worse", None, (6, 15), "{cp}, a customer owing {amt} at the year end, entered insolvency on {d}"),
    "lender_demand": ("worse", True, (10, 25), "{cp} issued a reservation-of-rights letter on {d} citing the "
                                               "year-end covenant position and requested repayment of {amt}"),
    "site_fire": ("worse", False, (8, 20), "a fire on {d} destroyed stock and equipment at the main site; "
                                           "uninsured losses estimated at {amt}"),
    "litigation_judgment": ("worse", True, (5, 12), "judgment was handed down on {d} in a claim brought before the "
                                                     "year end, awarding {cp} {amt}"),
    "contract_termination": ("worse", False, (6, 18), "{cp} gave notice on {d} terminating its supply contract, worth "
                                                      "about {amt} of annual gross margin"),
    "inventory_sold_below_cost": ("worse", True, (4, 10), "year-end stock was sold on {d} for {amt} less than its "
                                                          "carrying amount"),
    "tax_assessment": ("worse", True, (6, 15), "the tax authority issued an assessment on {d} for {amt} relating "
                                               "to the two prior years"),
    "key_supplier_failure": ("worse", False, (7, 18), "{cp}, the sole supplier of a key component, ceased trading on "
                                                      "{d}; replacement sourcing is expected to cost {amt}"),
    "refinancing_completed": ("better", False, (3, 8), "a new {amt} facility with {cp} was signed on {d}, replacing "
                                                       "the loan that was due within the year"),
    "equity_placing": ("better", False, (3, 8), "an equity placing raising {amt} completed on {d}"),
    "asset_sale_completed": ("better", False, (4, 10), "the sale of a surplus property completed on {d}, bringing "
                                                       "in {amt}"),
}


class SubsequentEventsReview(Schema):
    name = "subsequent_events_review"
    description = ("An event between the balance-sheet date and the planned signing date: does it adjust the "
                   "accounts, change the going-concern view, or move the signing date?")
    surfaces = {"json": 0.5, "events_checklist": 0.25, "partner_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        ev = pick_w(rng, {k: (2 if v[0] == "worse" else 1.6) for k, v in EVENTS.items()})
        direction, adj_default, (plo, phi), _ = EVENTS[ev]
        materiality = int(round(lognormal_amount(rng, 450, 0.6), -1))
        headroom = int(round(materiality * rng.uniform(3, 18), -1))
        prior = pick_w(rng, {"no_doubt": 3, "doubt_disclosed": 2})
        if direction == "better":
            amt = int(round(headroom * rng.uniform(0.3, 1.5), -1))
            if chance(rng, 0.5):
                prior = "doubt_disclosed"
        elif reg == "clear":
            amt = pick(rng, [int(round(materiality * rng.uniform(0.1, 0.45), -1)),
                             int(round(materiality * rng.uniform(1.4, 2.5), -1)),
                             int(round(headroom * rng.uniform(1.1, 2.0), -1))])
        elif reg == "borderline":
            amt = pick(rng, [int(round(materiality * near(rng, 1.0, 0.3), -1)),
                             int(round(materiality * near(rng, 0.5, 0.15), -1)),
                             int(round(headroom * near(rng, 0.25, 0.1), -1)),
                             int(round(headroom * near(rng, 1.0, 0.2), -1))])
        else:
            amt = pick(rng, [int(round(materiality * rng.uniform(0.2, 1.6), -1)),
                             int(round(headroom * rng.uniform(0.15, 1.3), -1))])
        amt = max(amt, 10)
        if adj_default is None:
            existed = chance(rng, 0.5)
        else:
            existed = adj_default if chance(rng, 0.85) else (not adj_default)
        days_after = rng.randint(10, 80)
        days_to_sign = rng.randint(3, 40)
        proc_days = rng.randint(plo, phi)
        if reg == "clear":
            days_to_sign = pick(rng, [rng.randint(phi + 12, phi + 35), rng.randint(1, max(2, plo - 2))])
        elif reg == "borderline":
            days_to_sign = max(1, proc_days + rng.randint(-4, 8))
        return {"regime": reg, "company": _company(rng), "industry": pick(rng, INDUSTRIES),
                "currency": pick(rng, CURRENCIES), "year_end": _fy_end(rng), "event": ev, "direction": direction,
                "amount_k": amt, "materiality_k": materiality, "liquidity_headroom_k": headroom,
                "prior_position": prior, "condition_existed": existed, "event_days_after_ye": days_after,
                "days_to_signing": days_to_sign, "procedure_days": proc_days,
                "counterparty": org(rng), "team": _team(rng),
                "mgmt_view": pick(rng, ["no_impact", "minor", "will_disclose", "already_adjusted", None])}

    def intended(self, f):
        better = f["direction"] == "better"
        amt, mat, head = f["amount_k"], f["materiality_k"], f["liquidity_headroom_k"]
        if better:
            shift = 0 if f["prior_position"] == "doubt_disclosed" else 1
        elif amt < 0.5 * mat:
            shift = 1
        elif amt < 0.25 * head:
            shift = 2
        elif amt < head and f["event"] != "lender_demand":
            shift = 3
        else:
            shift = 4
        slack = f["days_to_signing"] - f["procedure_days"]
        press = 0 if slack >= 15 else 1 if slack >= 5 else 2 if slack >= 0 else 3
        adjust_figures = (not better) and f["condition_existed"] and amt >= mat
        if shift >= 3 or (better and f["prior_position"] == "doubt_disclosed"):
            resp = "move_signing_date" if slack < 0 and shift >= 3 else "reassess_going_concern"
        elif amt < mat:
            resp = "no_change"
        elif f["condition_existed"]:
            resp = "adjust_figures"
        else:
            resp = "disclose_nonadjusting"
        return {"going_concern_shift": shift, "signing_date_pressure": press,
                "adjusting_event": "yes" if f["condition_existed"] else "no",
                "figures_must_be_adjusted": "yes" if adjust_figures else "no", "event_response": resp}

    def _facts(self, f, rng, cur, st, ds):
        d = f["year_end"] + timedelta(days=f["event_days_after_ye"])
        txt = EVENTS[f["event"]][3].format(cp=f["counterparty"], amt=kamt(f["amount_k"], cur, st), d=fmt_date(d, rng, ds))
        ctx = {
            "customer_insolvency": ("The balance was already more than 120 days overdue at the year end and the customer "
                                    "had missed two payment plans." if f["condition_existed"] else
                                    "The customer had paid on time until shortly before it failed; its collapse followed "
                                    "the loss of its own main contract after the year end."),
            "lender_demand": ("The letter refers to the covenant test at the year end." if f["condition_existed"] else
                              "The letter refers only to a missed reporting deadline after the year end."),
            "site_fire": ("Investigators link it to wiring faults reported, but not fixed, before the year end."
                          if f["condition_existed"] else "The cause was a lightning strike."),
            "litigation_judgment": ("The claim concerns products shipped in the prior year." if f["condition_existed"]
                                    else "The claim concerns a delivery made after the year end."),
            "contract_termination": ("The customer had warned in writing before the year end that it was re-tendering."
                                     if f["condition_existed"] else "The notice came without warning."),
            "inventory_sold_below_cost": ("The stock had been slow-moving for over a year." if f["condition_existed"]
                                          else "Prices collapsed after a competitor's launch in the new year."),
            "tax_assessment": ("It covers transactions in the years already closed." if f["condition_existed"] else
                               "It follows a change in law enacted after the year end."),
            "key_supplier_failure": ("The supplier's distress was known before the year end." if f["condition_existed"]
                                     else "The supplier's collapse was sudden."),
        }.get(f["event"], "")
        return txt[0].upper() + txt[1:] + ".", ctx

    MGMT = {"no_impact": "Management's view: no impact on the accounts.",
            "minor": "Management's view: minor; mention in the directors' report only.",
            "will_disclose": "Management's view: a note will cover it.",
            "already_adjusted": "Management says the draft figures already reflect it."}

    def render(self, f, rng, surface):
        cur, st, ds = f["currency"], _amt_style(rng), _ds(rng)
        fact, ctx = self._facts(f, rng, cur, st, ds)
        sign = f["year_end"] + timedelta(days=f["event_days_after_ye"] + rng.randint(5, 20))
        sign_date = sign + timedelta(days=f["days_to_signing"])
        prior = {"no_doubt": "Going-concern assessment before this event: no material uncertainty; headroom of "
                             f"{kamt(f['liquidity_headroom_k'], cur, st)} over the assessment period.",
                 "doubt_disclosed": "Going-concern assessment before this event: material uncertainty, disclosed in "
                                    f"note 2; forecast headroom {kamt(f['liquidity_headroom_k'], cur, st)} only if "
                                    "planned funding arrives."}[f["prior_position"]]
        mgmt = self.MGMT.get(f["mgmt_view"]) if f["mgmt_view"] else None
        proc = f"Estimated time to complete procedures on the event: about {f['procedure_days']} working days."
        team = f["team"]
        policy = {"SE-1": "Events that give evidence of conditions existing at the balance-sheet date are adjusted in "
                          "the figures; other material events are disclosed but not adjusted.",
                  "SE-2": "An event that changes the going-concern assessment requires the assessment to be "
                          "re-performed up to the signing date.",
                  "SE-3": "The report may not be signed until procedures on known events are complete."}
        if surface == "json":
            return dumps(prune({
                "engagement": {"entity": f["company"], "industry": f["industry"],
                               "year_end": f["year_end"].isoformat(), "today": sign.isoformat(),
                               "planned_signing_date": sign_date.isoformat()},
                "event": {"description": fact, "background": ctx or None, "management_view": mgmt},
                "thresholds": {"amounts_in": f"{cur} thousands", "overall_materiality": f["materiality_k"],
                               "event_amount": f["amount_k"]},
                "going_concern_before_event": prior, "remaining_work": proc, "firm_policy": policy,
            }))
        if surface == "events_checklist":
            return "\n".join([
                f"SUBSEQUENT EVENTS REVIEW -- {f['company']} -- year end {fmt_date(f['year_end'], rng, ds)}",
                f"Updated {fmt_date(sign, rng, ds)} by {team['senior']}; signing planned {fmt_date(sign_date, rng, ds)}",
                "", "Event identified:", "  " + noisy(fact + (" " + ctx if ctx else ""), rng, 0.1),
                "", kv_lines([("Overall materiality", kamt(f["materiality_k"], cur, st)),
                              ("Amount involved", kamt(f["amount_k"], cur, st)),
                              ("Management position", mgmt)], rng),
                "", prior, proc, "", "Firm rules:", "\n".join(f"  {k}: {v}" for k, v in policy.items()),
            ])
        return "\n".join([
            f"Subject: {f['company']} -- something after the year end",
            f"From: {team['manager']}", f"To: {team['partner']}", "",
            f"{_first(team['partner'])},", "",
            noisy(join_sentences([f"Picked up in today's subsequent-events inquiries: {fact}", ctx,
                                  f"For scale, overall materiality is {kamt(f['materiality_k'], cur, st)}.",
                                  mgmt]), rng, 0.15), "",
            prior, "", noisy(join_sentences([proc, f"We are booked to sign on {fmt_date(sign_date, rng, ds)}, "
                                                   f"{f['days_to_signing']} working days from now."]), rng, 0.1), "",
            "Firm guidance, for reference: " + " ".join(policy.values()), "",
            _first(team["manager"]),
        ])

    def questions(self):
        return {
            "going_concern_shift": score("How does this event change the going-concern picture?", [
                "Improves it: it eases or removes a doubt recorded earlier.",
                "Leaves it unchanged.",
                "Weakens it slightly, well within the existing headroom.",
                "Weakens it materially, eating into most of the headroom.",
                "Undermines it: the headroom is gone or the lender can call the debt.",
            ]),
            "signing_date_pressure": score("How much pressure does the planned signing date put on dealing with this "
                                           "event?", [
                "Little: there is ample time before signing.",
                "Some: the work fits, with a few days to spare.",
                "Heavy: the work will run right up to the signing date.",
                "Unworkable: the work cannot be finished before the planned date.",
            ]),
            "adjusting_event": noul(
                "The event gives evidence about a condition that already existed at the balance-sheet date.",
                yes="The underlying cause or condition was present at the year end; the event only revealed it.",
                no="The condition arose after the year end."),
            "figures_must_be_adjusted": noul(
                "The year-end figures themselves must be adjusted for this event before the report is signed."),
            "event_response": choice("What should the engagement team do about this event?", {
                "adjust_figures": "Have management adjust the year-end figures to reflect it.",
                "disclose_nonadjusting": "Leave the figures, but have management add a note describing the event "
                                         "and its effect.",
                "move_signing_date": "Re-run the going-concern assessment and move the signing date to allow for it.",
                "no_change": "Record it on file; it needs no change to the statements.",
                "reassess_going_concern": "Re-perform the going-concern assessment before the planned signing.",
            }),
        }


# =========================================================================== schema 5
class AuditReportModification(Schema):
    name = "audit_report_modification"
    description = ("At completion: given the going-concern findings and the draft note, what form does the audit "
                   "report take, and who must be consulted before it is released?")
    surfaces = {"json": 0.5, "completion_memo": 0.25, "draft_note_review": 0.25}

    FINDINGS = {
        "no_mu": [
            "forecast shows headroom throughout the period; the renewed facility was signed before the year end",
            "losses have narrowed and the parent's binding support is backed by its audited accounts",
            "the refinancing completed after the year end leaves headroom in the downside case too",
            "cash reserves cover forecast outflows for the whole period even if the expected new contracts do not start",
            "the covenant reset agreed during the year leaves room at every test date in the forecast",
            "the loss-making division was sold before the year end and the remaining business generates cash",
        ],
        "mu": [
            "the forecast depends on a refinancing that is still at term-sheet stage",
            "headroom disappears in the downside case and the main lender's support is not committed",
            "the plan relies on an equity raise that is only supported by soft commitments",
            "a covenant breach is forecast at the next test and no waiver has been agreed",
            "the business depends on renewing its largest contract, which the customer is re-tendering",
            "funds run out in month ten unless the shareholder loan is extended, and the shareholder has not committed",
            "the main facility expires within the period and the lender will decide on renewal only after these results",
        ],
        "basis_inappropriate": [
            "the lender appointed receivers over the main operating subsidiary after the year end",
            "the board resolved after the year end to cease trading and wind the business down",
            "the company lost its operating licence and has no realistic alternative but to stop trading",
            "the directors applied to the court for an insolvency process after the year end",
            "the only customer terminated its contract and the board has begun selling the remaining assets",
        ],
        "multiple": [
            "the refinancing, a major lawsuit and the renewal of the core customer contract are all unresolved, and "
            "each on its own could exhaust the headroom",
            "management will not extend the forecast beyond nine months, and the two remaining funding routes "
            "both depend on a lawsuit that has not been decided",
            "the forecast relies on a regulatory approval, a refinancing and a large customer settlement, none of "
            "which is assured and all of which are needed",
            "management has declined to produce a forecast beyond six months, and the lender's support is also undecided",
        ],
    }

    def sample_factors(self, rng):
        reg = regime(rng)
        finding = pick_w(rng, {"no_mu": 23, "mu": 34, "basis_inappropriate": 28, "multiple": 15})
        els = {"conditions": chance(rng, 0.75), "plans": chance(rng, 0.65), "explicit_mu": chance(rng, 0.55),
               "judgements": chance(rng, 0.6)}
        if reg == "clear":
            full = chance(rng, 0.6)
            els = {k: full for k in els}
            if finding == "no_mu":
                els["explicit_mu"] = False
        elif reg == "conflicting":
            # thorough wording on everything except the one element that matters
            els = {k: True for k in els}
            els[pick(rng, ["explicit_mu", "plans", "conditions"])] = False
        if finding == "no_mu" and els["explicit_mu"] and chance(rng, 0.7):
            els["explicit_mu"] = False
        listed = chance(rng, 0.3)
        dispute = chance(rng, 0.15)
        headroom_thin = chance(rng, 0.5) if finding == "no_mu" else None
        return {"regime": reg, "company": _company(rng), "industry": pick(rng, INDUSTRIES),
                "year_end": _fy_end(rng), "finding": finding, "finding_text": pick(rng, self.FINDINGS[finding]),
                "headroom_thin": headroom_thin, "note_elements": els, "listed": listed,
                "board_disputes_auditor": dispute, "management_basis": "going_concern",
                "team": _team(rng), "board_view": pick(rng, ["confident", "confident", "concerned", "insistent"]),
                "pages": rng.randint(28, 96)}

    @staticmethod
    def report_form(f):
        fd, e = f["finding"], f["note_elements"]
        if fd == "basis_inappropriate":
            return "adverse_opinion"
        if fd == "multiple":
            return "disclaimer_of_opinion"
        if fd == "mu":
            return "unmodified_with_mu_section" if (e["explicit_mu"] and e["conditions"] and e["plans"]) \
                else "qualified_opinion"
        return "unmodified"

    def intended(self, f):
        fd, e = f["finding"], f["note_elements"]
        form = self.report_form(f)
        if fd == "no_mu":
            req = ["conditions", "plans", "judgements"]
            frac = sum(e[k] for k in req) / 3 - (0.34 if e["explicit_mu"] else 0)
        elif fd == "basis_inappropriate":
            frac = 0.4 if (e["conditions"] and e["judgements"]) else 0.2 if e["conditions"] else 0.0
        else:
            frac = sum(e[k] for k in ("conditions", "plans", "explicit_mu", "judgements")) / 4
        adequacy = 0 if frac < 0.34 else 1 if frac < 0.6 else 2 if frac < 0.9 else 3
        sig = {"no_mu": 1 if f["headroom_thin"] else 0, "mu": 2, "multiple": 3, "basis_inappropriate": 3}[fd]
        if f["board_disputes_auditor"] and form != "unmodified":
            route = "risk_committee"
        elif form in ("adverse_opinion", "disclaimer_of_opinion") or (f["listed"] and form != "unmodified"):
            route = "technical_panel"
        elif form in ("unmodified_with_mu_section", "qualified_opinion"):
            route = "second_partner_review"
        else:
            route = "engagement_partner_only"
        return {"report_form": form, "note_adequacy": adequacy, "uncertainty_significance": sig,
                "basis_appropriate": "no" if fd == "basis_inappropriate" else "yes", "consultation_route": route}

    NOTE_TEXT = {
        "conditions": "describes the trading losses and the funding shortfall that give rise to the assessment",
        "plans": "sets out management's plans for dealing with them and how far they have progressed",
        "explicit_mu": "states in terms that a material uncertainty exists that may cast significant doubt on the "
                       "company's ability to continue as a going concern",
        "judgements": "explains the significant judgements the directors made in reaching their conclusion",
    }
    POLICY = {
        "AR-1": "A material uncertainty that is adequately disclosed: unmodified opinion with a separate "
                "material-uncertainty section. Inadequately disclosed: qualified opinion.",
        "AR-2": "Going-concern basis used when it is inappropriate: adverse opinion.",
        "AR-3": "Several interacting uncertainties, or a forecast management refuses to extend, that prevent a "
                "conclusion: disclaimer of opinion.",
        "Q-7": "Consult the technical panel before any adverse opinion or disclaimer, and before any going-concern "
               "modification for a listed entity. Unlisted entities: second-partner review for a "
               "material-uncertainty section or a qualified opinion. Any disagreement with the board over the "
               "report goes to the risk committee first.",
    }

    NOTE_SENTENCES = {
        "conditions": ["The company made a loss for the year and its headroom against its banking facilities narrowed "
                       "during the second half.",
                       "Trading in the year was below expectations and the company's cash-flow forecast shows that "
                       "additional funding will be needed within the next twelve months.",
                       "The company has net current liabilities at the year end and its main facility falls due for "
                       "renewal during the coming year."],
        "plans": ["The directors are negotiating a replacement facility and have identified cost savings and the sale "
                  "of surplus property.",
                  "Management's plans include an equity raise supported by the principal shareholder and a programme "
                  "of cost reductions, part of which is already in place.",
                  "The directors expect to agree a deferral of loan repayments with the lender and to sell a non-core "
                  "business."],
        "explicit_mu": ["These conditions indicate that a material uncertainty exists that may cast significant doubt "
                        "on the company's ability to continue as a going concern.",
                        "The directors acknowledge that these circumstances represent a material uncertainty that may "
                        "cast significant doubt on the company's ability to continue as a going concern."],
        "judgements": ["In reaching their conclusion the directors judged that the lender will continue to support "
                       "the business and that the forecast sales recovery is achievable.",
                       "The key judgements are the timing of the refinancing and the level of sales assumed for the "
                       "second half of the coming year."],
    }

    def render(self, f, rng, surface):
        ds = _ds(rng)
        e = f["note_elements"]
        req = ["conditions", "plans", "judgements"] + (["explicit_mu"] if f["finding"] != "no_mu" else [])
        order = ("conditions", "plans", "explicit_mu", "judgements")
        present = [self.NOTE_TEXT[k] for k in order if e[k]]
        absent = [self.NOTE_TEXT[k] for k in order if not e[k] and k in req]
        team = f["team"]
        thin = ({True: "Headroom in the downside case is small.", False: "Headroom is comfortable in all scenarios.",
                 None: None}[f["headroom_thin"]])
        board = {"confident": "The board considers the going-concern basis clearly appropriate.",
                 "concerned": "The board accepts the position is finely balanced.",
                 "insistent": "The board insists the note is sufficient and does not want the report to draw "
                              "attention to it."}[f["board_view"]]
        dispute = ("The audit committee chair has said the board will not accept any modification of the report."
                   if f["board_disputes_auditor"] else None)
        if surface == "json":
            return dumps(prune({
                "engagement": {"entity": f["company"], "industry": f["industry"], "year_end": f["year_end"].isoformat(),
                               "listed": f["listed"], "engagement_partner": team["partner"]},
                "going_concern_findings": {"summary": f["finding_text"], "downside_headroom": thin,
                                           "basis_used_by_management": "going concern"},
                "draft_note": {"covers": present or None, "does_not_cover": absent or None},
                "board": {"view": board, "dispute": dispute},
                "firm_policy": self.POLICY,
            }))
        if surface == "completion_memo":
            return "\n".join([
                f"COMPLETION MEMO -- {f['company']} -- {fmt_date(f['year_end'], rng, ds)}",
                f"Entity type: {'listed' if f['listed'] else 'unlisted'}; partner {team['partner']}; "
                f"manager {team['manager']}", "",
                "Going concern: " + noisy(f["finding_text"][0].upper() + f["finding_text"][1:] + ".", rng, 0.1)
                + (" " + thin if thin else ""),
                "Basis adopted by the directors: going concern.", "",
                "Draft accounts note 1.2 (going concern):",
                bullets(["covers: " + x for x in present] + ["missing: " + x for x in absent], rng), "",
                board + (" " + dispute if dispute else ""), "",
                "Firm policy:", "\n".join(f"  {k} {v}" for k, v in self.POLICY.items()),
            ])
        sents = ["The financial statements have been prepared on a going-concern basis."]
        sents += [(self.NOTE_SENTENCES[k][0] if (k == "conditions" and f["finding"] == "no_mu")
                   else pick(rng, self.NOTE_SENTENCES[k])) for k in order if e[k]]
        if not e["explicit_mu"]:
            sents.append(pick(rng, ["Accordingly the directors consider it appropriate to adopt the going-concern basis.",
                                    "The directors are therefore confident the company will continue in operation."]))
        return "\n".join([
            f"Draft note 1.2 -- Going concern ({f['company']}, {f['pages']}-page draft accounts)", "",
            "\"" + noisy(" ".join(sents), rng, 0.05) + "\"", "",
            f"Reviewer comments ({team['manager']}):",
            bullets([f"Audit finding: {f['finding_text']}." + (" " + thin if thin else "")] +
                    [board] + ([dispute] if dispute else []) +
                    [f"Entity is {'listed' if f['listed'] else 'not listed'}."], rng), "",
            "Policy: " + " ".join(f"[{k}] {v}" for k, v in self.POLICY.items()),
        ])

    def questions(self):
        return {
            "report_form": choice("What form should the auditor's report take?", {
                "adverse_opinion": "An adverse opinion, because the going-concern basis itself is inappropriate.",
                "disclaimer_of_opinion": "A disclaimer: the uncertainties are too many, or evidence too limited, to "
                                         "form an opinion.",
                "qualified_opinion": "A qualified opinion, because the material uncertainty is not adequately disclosed.",
                "unmodified": "A clean opinion with no going-concern section.",
                "unmodified_with_mu_section": "A clean opinion with a separate section drawing attention to the "
                                              "adequately disclosed material uncertainty.",
            }),
            "note_adequacy": score("How adequately does the draft going-concern note reflect what the audit found?", [
                "Inadequate: it misses or misstates the essential points.",
                "Partial: some required elements are missing.",
                "Largely adequate: one element is thin or missing.",
                "Fully adequate: everything the position requires is there.",
            ]),
            "uncertainty_significance": score("How significant is the uncertainty over the company's survival "
                                              "through the assessment period?", [
                "Remote: no real doubt.",
                "Present but contained: a close call that still clears the bar.",
                "Significant: a material uncertainty.",
                "Fundamental: survival through the period is unlikely or cannot be judged.",
            ]),
            "basis_appropriate": noul(
                "Preparing these financial statements on a going-concern basis is appropriate.",
                yes="The company has a realistic alternative to liquidation or ceasing trade, even if doubt remains.",
                no="Liquidation or ceasing to trade is intended, or there is no realistic alternative to it."),
            "consultation_route": choice("Who must review or be consulted before the report is released?", {
                "engagement_partner_only": "The engagement partner may sign without further consultation.",
                "risk_committee": "The firm's risk committee, because the board disputes the report.",
                "second_partner_review": "A second partner must review the going-concern conclusion and report wording.",
                "technical_panel": "The firm's technical panel must be consulted before release.",
            }),
        }


WORKFLOW = Workflow(
    name="audit_going_concern_assessment",
    title="Audit going-concern assessment",
    description="An external auditor's going-concern evaluation: liquidity runway, covenants, management's plans, "
                "subsequent events and the resulting audit report, against the firm's written criteria.",
    domain="accounting / external audit",
    schemas=[LiquidityRunwayEvaluation(), CovenantBreachAssessment(), ManagementPlansEvaluation(),
             SubsequentEventsReview(), AuditReportModification()],
    notes="Fictional companies, lenders and audit-firm rules. Decisions are about the client's solvency and "
          "reporting; no vendor bills, payment approvals or procurement.",
)
