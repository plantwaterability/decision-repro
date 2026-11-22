"""GMP batch release: a pharmaceutical quality unit decides whether a batch may be certified, held or rejected.

A (fictional) contract manufacturer's QA department and qualified person review batches against written SOPs and
specifications: the executed batch record and certificate of analysis before certification, an out-of-specification
laboratory investigation, an environmental-monitoring excursion during aseptic filling, a temperature excursion on a
batch arriving at a distribution site, and the visual-inspection and AQL results of a parenteral batch. Every state
quotes the SOP clauses or specification limits that govern the decision, so the answer follows from the written
criteria plus the record, which is often near a limit, internally inconsistent, or missing a closure.

Products, companies, sites and people are invented; product names are made up and are not real brands or medicines.
This is pre-release (or receipt) batch disposition only: no material-review-board decisions on parts, no post-market
recall, no bill, invoice or payment, and no conversation with a customer.
"""
from __future__ import annotations

import math
from datetime import timedelta

from .base import (
    ORG_A,
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
    table,
)

# product -> form category
PRODUCTS = {
    "Varostatin 20 mg film-coated tablets": "oral_solid", "Tobravir 300 mg capsules": "oral_solid",
    "Dorafenide 5 mg tablets": "oral_solid", "Lumiprazole 40 mg gastro-resistant capsules": "oral_solid",
    "Ostrafen 400 mg tablets": "oral_solid", "Sertalix 50 mg tablets": "oral_solid",
    "Rivotamide 150 mg prolonged-release tablets": "oral_solid", "Fenzolast 4 mg chewable tablets": "oral_solid",
    "Glucaprin 500 mg tablets": "oral_solid", "Somnovar 7.5 mg tablets": "oral_solid",
    "Quinavast 10 mg/mL oral solution": "oral_liquid", "Pellucin 250 mg/5 mL powder for oral suspension": "oral_liquid",
    "Zelmicort 0.1% cream": "semi_solid", "Dermatrex 2% gel": "semi_solid",
    "Kelocaine 2% solution for injection": "sterile", "Merovance 1 g powder for solution for infusion": "sterile",
    "Belantrel 25 mg/mL concentrate for infusion": "sterile", "Calvestrin 0.5 mg/mL eye drops": "sterile",
    "Hepralux 5,000 IU/mL solution for injection": "sterile", "Prenovax suspension for injection": "biologic",
    "Nexudrin 60 mg/0.6 mL solution in pre-filled syringe": "biologic", "Ardelimab 100 mg/mL concentrate": "biologic",
    "Tessavir 1 mg/mL solution for injection": "sterile", "Orvalin 40 mg/mL suspension for injection": "biologic",
}
STERILE = [p for p, k in PRODUCTS.items() if k in ("sterile", "biologic")]
BATCH_SIZES = {"oral_solid": ["150,000 tablets", "480,000 tablets", "1.2 million tablets", "300,000 tablets"],
               "oral_liquid": ["1,200 L", "2,500 L", "800 L"], "semi_solid": ["450 kg", "900 kg", "1,500 kg"],
               "sterile": ["62,000 vials", "24,000 ampoules", "40,000 vials", "15,000 bottles"],
               "biologic": ["18,500 syringes", "9,200 vials", "30,000 syringes"]}
SITES = ["Building 3, Line 2", "Building 1, Line 4", "Suite B filling line", "Oral solids hall 2", "Line FL-07",
         "Packaging hall C", "Aseptic suite A", "Building 5, Line 1"]


def _company(rng):
    return f"{pick(rng, ORG_A)} {pick(rng, ['Pharma', 'Pharmaceuticals', 'Biologics', 'Laboratories', 'Therapeutics', 'Contract Manufacturing'])}"


def _batch(rng):
    return f"{pick(rng, ['B', 'L', 'VB', 'KX', 'P'])}{rng.randint(20, 27)}{rng.randint(1000, 9999)}"


def _ds(rng):
    return pick(rng, ["iso", "iso", "us", "long", "eu"])


def _cap(s):
    return s[0].upper() + s[1:] if s else s


# =========================================================================== schema 1
# deviation -> (text, true class, investigation scope it points to, rework-able failure?)
DEVIATIONS = {
    "late_signature": ("operator signed step {n} about forty minutes after performing it", "minor", "none"),
    "line_clearance_sig": ("line clearance record for the packaging line missing the second signature", "minor", "none"),
    "metal_detector": ("metal detector rejected 3 tablets; challenge test before and after passed", "minor", "batch"),
    "dp_alarm": ("processing room differential-pressure alarm for 12 minutes during manufacture", "minor", "campaign"),
    "torque_out": ("granulation end-point torque outside the validated range for about 2 minutes", "major", "batch"),
    "hold_time": ("bulk hold time exceeded the validated 24 h by {h} h", "major", "batch"),
    "coating_temp": ("coating pan inlet temperature {c} C above the upper limit for 15 minutes; controller drift found", "major", "campaign"),
    "mixer_speed": ("homogeniser speed below the validated range for about 6 minutes; speed sensor recalibrated afterwards", "major", "campaign"),
    "wrong_excipient_lot": ("excipient lot dispensed that had not been released by QC; found after mixing", "major", "material"),
    "api_supplier_change": ("API lot from the supplier's new synthesis route used; change notification still under review", "major", "material"),
    "filter_integrity": ("post-use filter integrity test failed once; repeat test passed", "critical", "batch"),
    "filter_time": ("sterilising filter used beyond its validated contact time by {h} h", "critical", "campaign"),
}
DEV_FORMS = {"metal_detector": ("oral_solid",), "torque_out": ("oral_solid",), "coating_temp": ("oral_solid",),
             "mixer_speed": ("oral_liquid", "semi_solid"), "filter_integrity": ("sterile", "biologic"),
             "filter_time": ("sterile", "biologic")}
CLASSES = ["minor", "major", "critical"]


class BatchRecordReleaseReview(Schema):
    name = "batch_record_release_review"
    description = "An executed batch record and certificate of analysis reviewed before QP certification: certify, hold, rework or reject."
    surfaces = {"json": 0.5, "disposition_checklist": 0.25, "qa_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        product = pick(rng, list(PRODUCTS))
        form = PRODUCTS[product]
        assay = round(near(rng, 100.0, 4.5), 1)
        imp = round(near(rng, 0.6, 0.45, lo=0.05), 2)
        diss_min = round(near(rng, 88, 6, lo=60, hi=100), 0) if form == "oral_solid" else None
        yld = round(near(rng, 99.0, 2.2), 1)
        app = round(near(rng, 0.25, 0.3, lo=0.0), 2) if form == "oral_solid" else None
        label = pick_w(rng, {0: 6, 1: 1, 2: 1, 12: 1})
        label_explained = chance(rng, 0.5)
        pool = [k for k in DEVIATIONS if form in DEV_FORMS.get(k, (form,))]
        n_dev = pick(rng, [0, 1, 1, 2, 2, 3])
        devs = []
        for k in rng.sample(pool, n_dev):
            true = DEVIATIONS[k][1]
            claimed = true if chance(rng, 0.7) else CLASSES[max(0, CLASSES.index(true) - 1)]
            devs.append({"key": k, "claimed": claimed, "closed": chance(rng, 0.6), "impact_confirmed": False})
        rework = chance(rng, 0.3)
        if reg == "clear":
            if chance(rng, 0.55):
                assay, imp, yld, label = round(between(rng, 98.5, 101.5), 1), round(between(rng, 0.1, 0.5), 2), round(between(rng, 98.0, 100.5), 1), 0
                diss_min = rng.randint(90, 99) if form == "oral_solid" else None
                devs = [{"key": "late_signature", "claimed": "minor", "closed": True, "impact_confirmed": False}][:pick(rng, [0, 1])]
            else:
                if form == "oral_solid" and chance(rng, 0.5):
                    app, rework = round(between(rng, 0.8, 2.5), 2), chance(rng, 0.6)
                else:
                    assay = round(pick(rng, [between(rng, 90.0, 93.5), between(rng, 106.5, 109.0)]), 1)
        elif reg == "borderline":
            assay = round(near(rng, pick(rng, [95.0, 105.0]), 0.8), 1)
            yld = round(near(rng, pick(rng, [97.0, 101.5]), 0.6), 1)
            if form == "oral_solid" and chance(rng, 0.3):
                app, rework = round(near(rng, 0.5, 0.2, lo=0.1), 2), chance(rng, 0.7)
                devs = [d for d in devs if d["closed"]]
            elif form == "oral_solid" and chance(rng, 0.5):
                assay, diss_min = round(between(rng, 97.0, 103.0), 1), rng.randint(76, 88)
                devs = [d for d in devs if DEVIATIONS[d["key"]][1] == "minor"]
                for d in devs:
                    d["closed"], d["claimed"] = True, "minor"
                yld = round(between(rng, 97.5, 101.0), 1)
        else:
            twist = pick(rng, ["downgraded", "supply_pressure", "material", "clean_cr_bad_dev", "messy", "appearance"])
            if twist == "downgraded":
                k = pick(rng, [d for d in pool if DEVIATIONS[d][1] != "minor"])
                devs = [{"key": k, "claimed": "minor", "closed": True, "impact_confirmed": False}]
            elif twist == "material":
                devs = [{"key": pick(rng, ["wrong_excipient_lot", "api_supplier_change"]), "claimed": "major",
                         "closed": chance(rng, 0.4), "impact_confirmed": False}]
                imp = round(near(rng, 1.0, 0.15), 2)
            elif twist == "appearance":
                product = pick(rng, [p for p, k in PRODUCTS.items() if k == "oral_solid"])
                form, diss_min = "oral_solid", rng.randint(87, 97)
                pool = [k for k in DEVIATIONS if form in DEV_FORMS.get(k, (form,))]
                assay, imp = round(between(rng, 97.5, 102.5), 1), round(between(rng, 0.2, 0.7), 2)
                app, rework = round(between(rng, 0.6, 2.0), 2), chance(rng, 0.75)
                devs = [d for d in devs if d["closed"] and DEVIATIONS[d["key"]][1] == "minor"]
            elif twist == "messy":
                k = pick(rng, [d for d in pool if DEVIATIONS[d][1] == "major"])
                devs = [{"key": k, "claimed": "minor", "closed": False, "impact_confirmed": False}]
                yld = round(pick(rng, [between(rng, 94.0, 96.8), between(rng, 101.7, 103.0)]), 1)
                label, label_explained = pick(rng, [2, 12, 40]), False
            elif twist == "clean_cr_bad_dev":
                assay, imp = round(between(rng, 98.5, 101.0), 1), round(between(rng, 0.2, 0.5), 2)
                k = pick(rng, [d for d in pool if DEVIATIONS[d][1] != "minor"])
                devs = [{"key": k, "claimed": DEVIATIONS[k][1], "closed": chance(rng, 0.5),
                         "impact_confirmed": DEVIATIONS[k][1] == "critical" and chance(rng, 0.5)}]
        for d in devs:
            if DEVIATIONS[d["key"]][1] == "critical" and d["closed"] and chance(rng, 0.4):
                d["impact_confirmed"] = True
        return {"regime": reg, "product": product, "form": form, "assay": assay, "impurities_total": imp,
                "dissolution_min": diss_min, "yield_pct": yld, "label_discrepancy": label,
                "label_explained": label_explained if label else True, "deviations": devs,
                "rework_procedure_approved": rework, "appearance_defect_pct": app,
                "batch": _batch(rng),
                "batch_size": pick(rng, BATCH_SIZES[form]).replace("tablets", "capsules" if "capsule" in product else "tablets"),
                "site": pick(rng, SITES), "company": _company(rng), "reviewer": person(rng), "qp": person(rng),
                "stock_days": pick(rng, [None, None, 4, 9, 21, 60]),
                "n": rng.randint(8, 46), "h": rng.randint(2, 14), "c": rng.randint(2, 6), "date": rand_date(rng)}

    # ------------------------------------------------------------------ derived
    @staticmethod
    def _spec_status(f):
        """('pass' | 'fail' | 'stage2'), failing test name."""
        if not 95.0 <= f["assay"] <= 105.0:
            return "fail", "assay"
        if f["impurities_total"] > 1.0:
            return "fail", "impurities"
        if f["dissolution_min"] is not None and f["dissolution_min"] < 85:
            return ("stage2", "dissolution") if f["dissolution_min"] >= 70 else ("fail", "dissolution")
        if f.get("appearance_defect_pct") is not None and f["appearance_defect_pct"] > 0.5:
            return "fail", "appearance"
        return "pass", None

    def expected(self, f):
        """Intended answers from the latent factors (spread audit only; never rendered)."""
        spec, failing = self._spec_status(f)
        devs = f["deviations"]
        true = [DEVIATIONS[d["key"]][1] for d in devs]
        open_nonminor = any(DEVIATIONS[d["key"]][1] != "minor" and not d["closed"] for d in devs)
        misclassified = any(d["claimed"] != DEVIATIONS[d["key"]][1] for d in devs)
        critical_confirmed = any(d["impact_confirmed"] for d in devs)
        yield_ok = 97.0 <= f["yield_pct"] <= 101.5
        label_ok = f["label_discrepancy"] == 0 or f["label_explained"]
        if critical_confirmed:
            dec = "reject_batch"
        elif spec == "fail":
            dec = ("rework_under_approved_procedure" if f["rework_procedure_approved"] and failing == "appearance"
                   else "reject_batch")
        elif open_nonminor or misclassified or not yield_ok or not label_ok:
            dec = "hold_pending_deviation_closure"
        elif spec == "stage2":
            dec = "hold_pending_additional_testing"
        else:
            dec = "certify_release"
        scopes = [DEVIATIONS[d["key"]][2] for d in devs if DEVIATIONS[d["key"]][1] != "minor" or DEVIATIONS[d["key"]][2] != "none"]
        if "material" in scopes or (spec == "fail" and failing == "impurities"):
            scope = "all_batches_using_material_lot"
        elif "campaign" in scopes:
            scope = "same_campaign_batches"
        elif "batch" in scopes or spec != "pass" or not yield_ok:
            scope = "this_batch_only"
        else:
            scope = "no_investigation_needed"
        if "critical" in true:
            imp = 3
        elif any(DEVIATIONS[d["key"]][1] == "major" and (not d["closed"] or d["claimed"] == "minor") for d in devs):
            imp = 2
        elif "major" in true or any(DEVIATIONS[d["key"]][2] != "none" for d in devs):
            imp = 1
        else:
            imp = 0
        gaps = sum([any(not d["closed"] for d in devs), not yield_ok, not label_ok, misclassified])
        comp = max(0, 3 - gaps)
        return {"release_decision": dec, "investigation_scope": scope, "deviation_impact": imp,
                "record_review_completeness": comp, "all_release_specs_met": "yes" if spec == "pass" else "no"}

    def _rules(self, rng, f):
        diss = (" Dissolution: S1, every unit >= Q+5% (Q = 80% at 30 min); if S1 fails with all units >= 70%, test 12 "
                "more units at S2 before concluding; any unit below 70% fails. Coating/appearance defects <= 0.5% of "
                "units." if f["form"] == "oral_solid" else "")
        return [
            f"SPEC: assay 95.0-105.0% of label claim; total impurities <= 1.0%.{diss}",
            pick(rng, ["QA-SOP-014 §5: certify only when every specification is met, every major or critical deviation "
                       "is closed with a documented impact assessment, yield is within 97.0-101.5% (or explained by a "
                       "closed deviation) and label reconciliation is exact or explained.",
                       "QA-SOP-014 §5: no certification while a major or critical deviation is open, a result is out of "
                       "specification, yield falls outside 97.0-101.5% without a closed deviation, or labels do not "
                       "reconcile."]),
            pick(rng, ["QA-SOP-014 §6: QA must confirm each deviation's class against SOP definitions (critical: could "
                       "affect sterility or patient safety; major: outside a validated parameter or unreleased material; "
                       "minor: documentation only). A misclassified deviation is re-opened.",
                       "QA-SOP-014 §6: classes are QA's to confirm (critical = sterility/patient-safety risk, major = "
                       "validated range exceeded or unreleased material used, minor = documentation); a wrongly "
                       "classed deviation goes back open."]),
            "QA-SOP-014 §8: only coating/appearance failures may be reworked, and only where the approved sorting-and-"
            "recoating procedure is on file for the product; assay, impurity and dissolution failures are rejected.",
        ]

    def _dev_lines(self, f):
        out = []
        for i, d in enumerate(f["deviations"]):
            txt = DEVIATIONS[d["key"]][0].format(n=f["n"], h=f["h"], c=f["c"])
            st = "closed" if d["closed"] else "open"
            if d["impact_confirmed"]:
                st = "closed: product impact confirmed"
            out.append(f"DEV-{f['batch'][-4:]}-{i + 1} ({d['claimed']} as raised, {st}): {txt}")
        return out

    def render(self, f, rng, surface):
        ds = _ds(rng)
        devs = self._dev_lines(f) or ["none raised"]
        diss = f"min {f['dissolution_min']:.0f}% (S1, n=6)" if f["dissolution_min"] is not None else None
        label = ("reconciled exactly" if f["label_discrepancy"] == 0 else
                 f"{f['label_discrepancy']} label(s) unaccounted for" + ("; explained in the packaging log" if f["label_explained"] else ""))
        rework = ("approved sorting-and-recoating procedure on file for this product" if f["rework_procedure_approved"]
                  else "no approved rework procedure")
        app = f"{f['appearance_defect_pct']}% of units" if f["appearance_defect_pct"] is not None else None
        stock = f"market stock-out in {f['stock_days']} days" if f["stock_days"] else None
        if surface == "json":
            return dumps(prune({
                "batch": {"product": f["product"], "batch": f["batch"], "size": f["batch_size"], "site": f["site"],
                          "manufacturer": f["company"]},
                "coa": {"assay_pct_label": f["assay"], "total_impurities_pct": f["impurities_total"], "dissolution": diss,
                        "coating_appearance_defects": app},
                "batch_record": {"yield_pct": f["yield_pct"], "label_reconciliation": label, "deviations": devs,
                                 "rework": rework},
                "context": {"supply": stock, "reviewer": f["reviewer"]},
                "criteria": self._rules(rng, f),
            }))
        if surface == "disposition_checklist":
            rows = [["Assay (% LC)", f"{f['assay']}"], ["Total impurities (%)", f"{f['impurities_total']}"]]
            if diss:
                rows.append(["Dissolution", diss])
            if app:
                rows.append(["Appearance defects", app])
            rows += [["Yield (%)", f"{f['yield_pct']}"], ["Label reconciliation", label], ["Rework", rework]]
            head = kv_lines([("Product", f["product"]), ("Batch", f"{f['batch']} ({f['batch_size']})"),
                             ("Site", f"{f['site']}, {f['company']}"), ("Reviewed by", f["reviewer"]),
                             ("Date", fmt_date(f["date"], style=ds)), ("Supply", stock)], rng)
            return (f"BATCH DISPOSITION CHECKLIST\n{head}\n\n" + table(["item", "record"], rows, rng)
                    + "\n\nDeviations:\n" + bullets(devs, rng) + "\n\n" + "\n".join(self._rules(rng, f)))
        para = [f"{f['batch']} ({f['product']}, {f['batch_size']}) is ready for your review.",
                f"CoA: assay {f['assay']}%, total impurities {f['impurities_total']}%" + (f", dissolution {diss}" if diss else "")
                + (f", appearance defects {app}" if app else "") + ".",
                f"Yield {f['yield_pct']}%; labels {label}.", "Deviations: " + "; ".join(devs) + ".", _cap(rework) + "."]
        if stock:
            para.append(pick(rng, [f"Supply chain flags a {stock}.", f"Commercial is asking for it; {stock}."]))
        if f["regime"] == "conflicting":
            para.append(pick(rng, ["Production considers everything minor.", "Ops lead says the record is clean apart from paperwork."]))
        return "\n".join([f"Subject: {f['batch']} disposition -- {f['product']}", f"From: {f['reviewer']} (QA)",
                          f"To: {f['qp']} (QP)", "", noisy(join_sentences(para), rng, 0.1), "", "Criteria:",
                          bullets(self._rules(rng, f), rng)])

    def questions(self):
        return {
            "release_decision": choice("What disposition should QA give this batch now?", {
                "certify_release": "Certify the batch for release.",
                "hold_pending_deviation_closure": "Keep it in quarantine until the open record issues and deviations are closed.",
                "hold_pending_additional_testing": "Keep it in quarantine until the further testing the specification calls for is done.",
                "rework_under_approved_procedure": "Rework it under the approved rework procedure, then retest.",
                "reject_batch": "Reject the batch.",
            }),
            "investigation_scope": choice("How far should the investigation reach beyond this batch?", {
                "no_investigation_needed": "No investigation is needed; nothing points to a product problem.",
                "this_batch_only": "This batch only; the cause is specific to it.",
                "same_campaign_batches": "Other batches made on the same equipment in the same campaign.",
                "all_batches_using_material_lot": "Every batch that used the same raw material or API lot.",
            }),
            "deviation_impact": score("How much could the recorded deviations affect product quality?", [
                "None: documentation slips only.",
                "Limited: a process event with a documented no-impact assessment.",
                "Material: a validated limit was exceeded or unreleased material used, and not yet resolved.",
                "Serious: sterility assurance or patient safety could be affected.",
            ]),
            "record_review_completeness": score("How complete is the batch documentation for a release decision?", [
                "Far from complete: several open items.",
                "Incomplete: two open items.",
                "Nearly complete: one open item.",
                "Complete: every item closed or reconciled.",
            ]),
            "all_release_specs_met": noul(
                "Every release test on the certificate of analysis meets its specification.",
                yes="Assay, impurities and any dissolution result are within the stated limits and no further stage is needed.",
                no="A result is outside its limit, or a staged test has not yet been completed."),
        }


# =========================================================================== schema 2
# test -> (spec text, limit kind, lower, upper, unit)
OOS_TESTS = {
    "assay": ("95.0-105.0% of label claim", 95.0, 105.0, "%"),
    "impurity B": ("not more than 0.15%", None, 0.15, "%"),
    "water content (KF)": ("not more than 2.5%", None, 2.5, "%"),
    "dissolution (mean, n=6)": ("not less than 80% (Q) at 30 min", 80.0, None, "%"),
    "content uniformity (AV)": ("acceptance value not more than 15.0", None, 15.0, ""),
}
CAUSES = {
    "integration": ("manual re-integration of the {peak} peak; audit trail shows the baseline was moved after the run", "documented"),
    "standard_prep": ("reference standard prepared in the wrong volumetric flask (logbook entry and balance printout attached)", "documented"),
    "instrument": ("detector lamp energy below limit during the run; service ticket confirms the lamp was replaced next day", "documented"),
    "bath_temp": ("dissolution bath logged at 35.1 C for the run (spec 37.0 +/- 0.5 C); temperature log printout attached", "documented"),
    "vessel_leak": ("dissolution vessel 4 found cracked after the run; noted by the analyst, not photographed", "partial"),
    "undissolved": ("undissolved residue seen in one sample flask, noted on the worksheet; no repeat of that unit", "partial"),
    "kf_titer": ("KF titrant titer check was 9 days overdue; logbook shows the last check outside the window", "documented"),
    "kf_drift": ("analyst noticed high drift on the KF cell during the run; no drift value recorded", "partial"),
    "analyst_error": ("analyst believes a dilution step was mis-pipetted; nothing recorded at the time", "asserted"),
    "none_found": ("no laboratory error identified; system suitability passed", "none"),
}
TEST_CAUSES = {"assay": ["integration", "standard_prep", "instrument", "analyst_error", "none_found"],
               "impurity B": ["integration", "standard_prep", "instrument", "analyst_error", "none_found"],
               "water content (KF)": ["kf_titer", "kf_drift", "analyst_error", "none_found"],
               "dissolution (mean, n=6)": ["bath_temp", "vessel_leak", "analyst_error", "none_found"],
               "content uniformity (AV)": ["integration", "undissolved", "standard_prep", "analyst_error", "none_found"]}
TEST_FORMS = {"assay": None, "impurity B": None, "water content (KF)": ("oral_solid", "sterile"),
              "dissolution (mean, n=6)": ("oral_solid",), "content uniformity (AV)": ("oral_solid",)}


class OosInvestigationReview(Schema):
    name = "oos_investigation_review"
    description = "An out-of-specification QC result and its Phase I investigation, reviewed by QA for the batch decision."
    surfaces = {"json": 0.5, "investigation_worksheet": 0.25, "qa_review_thread": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        test = pick(rng, list(OOS_TESTS))
        forms = TEST_FORMS[test]
        product = pick(rng, [p for p, k in PRODUCTS.items() if forms is None or k in forms])
        spec, lo, hi, unit = OOS_TESTS[test]
        causes = TEST_CAUSES[test]
        cause = pick_w(rng, {c: (2 if CAUSES[c][1] in ("documented", "partial") else 1.5) for c in causes})
        timing = pick_w(rng, {"after_phase1": 5, "before_phase1": 2, "not_done": 3})
        analyst = pick_w(rng, {"second": 4, "same": 1})
        n_retest = pick(rng, [3, 3, 3, 6])
        averaged = chance(rng, 0.12)
        phase2 = pick_w(rng, {"not_started": 6, "complete_no_cause": 2, "complete_mfg_cause": 1})
        trend = pick_w(rng, {"stable": 3, "drifting": 1})
        shared_run = chance(rng, 0.3)
        retest_pass = chance(rng, 0.75)
        if reg == "clear":
            if chance(rng, 0.5):
                cause = pick(rng, [c for c in causes if CAUSES[c][1] == "documented"])
                timing, analyst, averaged, retest_pass = "after_phase1", "second", False, True
            else:
                cause, timing, phase2 = "none_found", "not_done", "complete_no_cause"
        elif reg == "conflicting":
            twist = pick(rng, ["early_retest", "averaging", "weak_cause_pass", "trend"])
            if twist == "early_retest":
                timing, retest_pass = "before_phase1", True
            elif twist == "averaging":
                averaged, timing, retest_pass = True, "after_phase1", True
            elif twist == "weak_cause_pass":
                cause, timing, retest_pass = pick(rng, [c for c in causes if CAUSES[c][1] in ("asserted", "partial")]), "after_phase1", True
            else:
                trend, cause = "drifting", pick(rng, ["none_found", "analyst_error"])
        if timing == "not_done":
            averaged = False
        # the original OOS value: just over the limit
        if lo is not None:
            orig = round(lo - abs(near(rng, 0, 1.5 if lo > 50 else 0.03)) - 0.1, 1 if lo > 50 else 2)
        else:
            orig = round(hi + abs(near(rng, 0, hi * 0.15)) + hi * 0.01, 2 if hi < 5 else 1)
        if lo is not None:
            pass_vals = [round(between(rng, lo + 1.0, lo + 4.5), 1) for _ in range(n_retest)]
            fail_vals = [round(between(rng, lo - 1.5, lo + 0.8), 1) for _ in range(n_retest)]
        else:
            pass_vals = [round(hi * between(rng, 0.6, 0.93), 2) for _ in range(n_retest)]
            fail_vals = [round(hi * between(rng, 0.95, 1.12), 2) for _ in range(n_retest)]
        retests = (pass_vals if retest_pass else fail_vals) if timing != "not_done" else []
        prev = []
        for _ in range(5):
            if lo is not None:
                prev.append(round(lo + (between(rng, 0.3, 1.2) if trend == "drifting" else between(rng, 2.0, 6.0)), 1))
            else:
                prev.append(round(hi * (between(rng, 0.85, 0.98) if trend == "drifting" else between(rng, 0.4, 0.75)), 2))
        return {"regime": reg, "product": product, "test": test, "spec": spec, "lower": lo, "upper": hi, "unit": unit,
                "original": orig, "cause": cause, "cause_evidence": CAUSES[cause][1], "retest_timing": timing,
                "retest_analyst": analyst, "retests": retests, "averaged": averaged, "phase2": phase2,
                "trend": trend, "previous_batches": prev, "instrument_shared_run": shared_run,
                "batch": _batch(rng), "analyst": person(rng), "qa": person(rng), "lab_id": ident(rng, "OOS", 4, "-"),
                "date": rand_date(rng), "company": _company(rng)}

    def _in_spec(self, f, v):
        return (f["lower"] is None or v >= f["lower"]) and (f["upper"] is None or v <= f["upper"])

    def expected(self, f):
        """Intended answers from the latent factors (spread audit only; never rendered)."""
        strength = {"documented": 3, "partial": 2, "asserted": 1, "none": 0}[f["cause_evidence"]]
        done = f["retest_timing"] != "not_done"
        breach = done and (f["retest_timing"] == "before_phase1" or f["retest_analyst"] == "same" or f["averaged"])
        all_pass = done and all(self._in_spec(f, v) for v in f["retests"])
        if breach:
            concl = "repeat_phase_one_under_qa"
        elif strength >= 2:
            if not done:
                concl = "invalidate_and_order_retest"
            elif all_pass:
                concl = "invalidate_and_report_retest"
            else:
                concl = "open_phase_two_investigation" if f["phase2"] == "not_started" else "confirm_oos_and_reject"
        elif f["phase2"] != "not_started":
            concl = "confirm_oos_and_reject"
        else:
            concl = "open_phase_two_investigation"
        tic = 0
        if f["retest_timing"] == "before_phase1":
            tic += 2
        if done and f["retest_analyst"] == "same":
            tic += 1
        if f["averaged"]:
            tic += 2
        if done and all_pass and strength <= 1:
            tic += 1
        permitted = strength >= 2 and f["retest_timing"] != "before_phase1"
        other = f["trend"] == "drifting" or (f["cause"] == "instrument" and f["instrument_shared_run"]) or f["phase2"] == "complete_mfg_cause"
        return {"oos_conclusion": concl, "assignable_cause_strength": strength, "testing_into_compliance_concern": min(3, tic),
                "retest_permitted": "yes" if permitted else "no", "other_batches_implicated": "yes" if other else "no"}

    def _rules(self, rng):
        return [
            pick(rng, ["QC-SOP-031 §4: Phase I looks for an assignable laboratory cause. The original result may be "
                       "invalidated only on objective evidence (audit trail, logbook, instrument or equipment records); "
                       "an analyst's recollection alone is not evidence.",
                       "QC-SOP-031 §4: invalidating an OOS needs a documented lab cause backed by records such as the "
                       "audit trail, logbooks or service reports; unrecorded analyst error does not qualify."]),
            pick(rng, ["QC-SOP-031 §5: retest only after Phase I has identified a cause, by a second analyst, with the "
                       "pre-set number of replicates (3); report all results. Never average an OOS result with passing results.",
                       "QC-SOP-031 §5: retests require a completed Phase I with an assignable cause, a different analyst "
                       "and n = 3 as pre-defined; all results reported; no averaging with the OOS value."]),
            "QC-SOP-031 §6: with no assignable lab cause, open a Phase II investigation covering manufacturing. A result "
            "not invalidated by Phase II is a confirmed OOS and the batch is rejected. Any breach of §5 sends the "
            "investigation back to Phase I under QA oversight.",
        ]

    def _retest_text(self, f):
        if f["retest_timing"] == "not_done":
            return "no retest performed yet"
        when = "before Phase I was concluded" if f["retest_timing"] == "before_phase1" else "after Phase I was concluded"
        who = "by the original analyst" if f["retest_analyst"] == "same" else "by a second analyst"
        vals = ", ".join(f"{v:g}" for v in f["retests"])
        s = f"retest {when}, {who}: {vals}"
        if f["averaged"]:
            allv = [f["original"]] + f["retests"]
            s += f"; reported value = mean of original and retests, {sum(allv) / len(allv):.2f}"
        return s

    def render(self, f, rng, surface):
        ds = _ds(rng)
        u = f["unit"]
        cause_txt = CAUSES[f["cause"]][0].format(peak="main" if f["test"] != "impurity B" else "impurity B")
        phase2 = {"not_started": "Phase II not started",
                  "complete_no_cause": "Phase II complete: no manufacturing or laboratory cause found",
                  "complete_mfg_cause": "Phase II complete: blend uniformity issue traced to a mixer speed setting"}[f["phase2"]]
        prev = ", ".join(f"{v:g}" for v in f["previous_batches"])
        shared = ("the same HPLC run also carried release samples for two other batches" if f["instrument_shared_run"]
                  and f["cause"] == "instrument" else None)
        if surface == "json":
            return dumps(prune({
                "result": {"lab_id": f["lab_id"], "product": f["product"], "batch": f["batch"], "test": f["test"],
                           "specification": f["spec"], "original_result": f"{f['original']:g}{u}", "analyst": f["analyst"]},
                "phase_1": {"finding": cause_txt, "retest": self._retest_text(f)},
                "phase_2": phase2,
                "history": {"previous_5_batches": prev, "note": shared},
                "procedure": self._rules(rng),
            }))
        if surface == "investigation_worksheet":
            head = kv_lines([("OOS no.", f["lab_id"]), ("Product / batch", f"{f['product']} / {f['batch']}"),
                             ("Test", f"{f['test']} (spec {f['spec']})"), ("Original result", f"{f['original']:g}{u}"),
                             ("Analyst", f["analyst"]), ("Opened", fmt_date(f["date"], style=ds)),
                             ("Phase I finding", cause_txt), ("Retest", self._retest_text(f)), ("Phase II", phase2),
                             ("Previous 5 batches", prev), ("Note", shared)], rng)
            return f"OOS INVESTIGATION WORKSHEET -- {f['company']}\n{head}\n\n" + "\n".join(self._rules(rng))
        t = rand_ts(rng, hours=(8, 16))
        msgs = [(f["analyst"], f"{f['lab_id']}: {f['test']} on {f['batch']} came in at {f['original']:g}{u} against {f['spec']}"),
                (f["analyst"], f"phase I: {cause_txt}"), (f["analyst"], self._retest_text(f)),
                (f["qa"], f"{phase2}. last five batches: {prev}")]
        if shared:
            msgs.append((f["analyst"], shared))
        if f["regime"] == "conflicting":
            msgs.append((f["analyst"], pick(rng, ["retests are clearly fine, can we close this?",
                                                  "I'm confident it was my pipetting", "production says the batch is good"])))
        out = []
        for who, text in msgs:
            t += timedelta(minutes=rng.randint(3, 50))
            out.append(f"{fmt_ts(t, style='log')}  {who}: {noisy(text, rng, 0.08)}")
        return f"QA review thread -- {f['lab_id']}\n" + "\n".join(out) + "\n\nSOP:\n" + bullets(self._rules(rng), rng)

    def questions(self):
        return {
            "oos_conclusion": choice("How should QA conclude the out-of-specification investigation at this point?", {
                "invalidate_and_report_retest": "Invalidate the original result and report the valid retest results.",
                "invalidate_and_order_retest": "Invalidate the original result and order a retest under the procedure.",
                "open_phase_two_investigation": "Escalate to a full Phase II investigation covering manufacturing.",
                "repeat_phase_one_under_qa": "Send the investigation back to Phase I under QA oversight; the process was not followed.",
                "confirm_oos_and_reject": "Treat the result as a confirmed failure and reject the batch.",
            }),
            "assignable_cause_strength": score("How strong is the evidence for an assignable laboratory cause?", [
                "None: no laboratory error was found.",
                "Weak: asserted but not recorded anywhere.",
                "Moderate: a plausible cause with some supporting observation.",
                "Strong: objective records demonstrate the error.",
            ]),
            "testing_into_compliance_concern": score("How much does the handling of retests suggest testing into compliance?", [
                "None: the procedure was followed.",
                "Some: one irregularity in how passing results were obtained.",
                "Considerable: retests run out of sequence or by the same analyst.",
                "Serious: several irregularities, such as early retests plus averaging.",
            ]),
            "retest_permitted": noul(
                "The procedure permits a retest to be used in this investigation.",
                yes="Phase I has a documented assignable cause and any retest followed it.",
                no="There is no documented cause, or the retest was run before Phase I concluded."),
            "other_batches_implicated": noul(
                "The investigation should extend to other batches."),
        }


# =========================================================================== schema 3
ORGANISMS = {
    "Staphylococcus epidermidis": "skin", "Micrococcus luteus": "skin", "Cutibacterium acnes": "skin",
    "Staphylococcus hominis": "skin", "Corynebacterium sp.": "skin", "Bacillus cereus": "spore",
    "Bacillus subtilis": "spore", "Paenibacillus sp.": "spore", "Aspergillus brasiliensis": "mould",
    "Penicillium sp.": "mould", "Cladosporium sp.": "mould", "Pseudomonas fluorescens": "gram_negative",
    "Ralstonia pickettii": "gram_negative", "Burkholderia cepacia complex": "gram_negative",
}
LOCATIONS = {
    "filling_needles": ("Grade A", "settle plate beside the filling needles", True),
    "stopper_bowl": ("Grade A", "contact plate on the stopper bowl rim", True),
    "glove": ("Grade A", "glove print, operator's right hand, after an intervention", True),
    "a_background": ("Grade A", "active air sample at the RABS rear panel, away from open vials", False),
    "b_floor": ("Grade B", "contact plate on the Grade B floor by the airlock", False),
    "b_air": ("Grade B", "active air sample in the Grade B corridor", False),
}


class EmExcursionImpact(Schema):
    name = "em_excursion_impact"
    description = "An environmental-monitoring excursion during an aseptic fill: batch impact and corrective scope."
    surfaces = {"json": 0.5, "em_deviation_report": 0.25, "qa_impact_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        loc = pick_w(rng, {"filling_needles": 2, "stopper_bowl": 1, "glove": 2, "a_background": 2, "b_floor": 2, "b_air": 1})
        grade = LOCATIONS[loc][0]
        cfu = 1 if grade == "Grade A" else int(near(rng, 5, 4, lo=1, integer=True))
        org = pick(rng, list(ORGANISMS))
        org_id = pick_w(rng, {"identified": 5, "pending": 1})
        timing = pick_w(rng, {"during_fill": 5, "setup": 2, "after_fill": 1})
        intervention = pick_w(rng, {"recorded": 4, "unrecorded": 1, "none": 2})
        trend = pick(rng, [0, 0, 1, 1, 2, 3, 5])
        media = pick_w(rng, {"passed_recent": 6, "overdue": 1, "failed_last": 1})
        sterility = pick_w(rng, {"pending": 4, "passed": 3})
        if reg == "clear":
            if chance(rng, 0.5):
                loc, org, timing, intervention, trend, media, org_id = "b_floor", pick(rng, ["Micrococcus luteus", "Staphylococcus epidermidis"]), "during_fill", "none", 0, "passed_recent", "identified"
                grade, cfu, sterility = "Grade B", rng.randint(5, 8), "passed"
            else:
                loc, org, timing, org_id = "filling_needles", pick(rng, ["Bacillus cereus", "Aspergillus brasiliensis", "Burkholderia cepacia complex"]), "during_fill", "identified"
                grade, cfu = "Grade A", rng.randint(1, 3)
        elif reg == "borderline":
            if grade == "Grade B":
                cfu = int(near(rng, 5, 1, lo=2, integer=True))
        else:
            twist = pick(rng, ["commensal_at_needles", "background_mould", "trend", "media_fail"])
            if twist == "commensal_at_needles":
                loc, grade, org, timing, intervention = "filling_needles", "Grade A", "Staphylococcus epidermidis", "during_fill", "recorded"
                cfu = 1
            elif twist == "background_mould":
                loc, grade, org = "a_background", "Grade A", pick(rng, ["Penicillium sp.", "Cladosporium sp."])
                cfu = 1
            elif twist == "trend":
                trend = pick(rng, [3, 4, 6])
            else:
                media = "failed_last"
        if grade == "Grade A":
            cfu = max(1, min(cfu, 4))
        if timing != "during_fill":
            intervention = "none"            # interventions on open product only happen during filling
        return {"regime": reg, "location": loc, "grade": grade, "cfu": cfu, "organism": org,
                "organism_class": ORGANISMS[org], "organism_id": org_id, "timing": timing, "intervention": intervention,
                "excursions_30d": trend, "media_fill": media, "sterility_test": sterility,
                "product": pick(rng, STERILE), "batch": _batch(rng), "line": pick(rng, ["FL-02", "FL-07", "RABS 3", "RABS line A4", "FL-11", "Suite B filler"]),
                "operator": person(rng), "micro": person(rng), "qa": person(rng), "company": _company(rng),
                "date": rand_ts(rng, hours=(6, 22))}

    @staticmethod
    def _action(f):
        if f["grade"] == "Grade A":
            return f["cfu"] >= 1
        lim = 5
        return f["cfu"] >= lim

    def expected(self, f):
        """Intended answers from the latent factors (spread audit only; never rendered)."""
        product_zone = LOCATIONS[f["location"]][2] and f["timing"] == "during_fill"
        objectionable = f["organism_class"] in ("spore", "mould", "gram_negative")
        if f["organism_id"] == "pending" or f["excursions_30d"] >= 3:
            imp = "extend_investigation_before_decision"
        elif f["media_fill"] == "failed_last" or (product_zone and (objectionable or f["intervention"] == "unrecorded")):
            imp = "reject_batch"
        elif not self._action(f):
            imp = "release_with_documented_assessment"
        elif f["sterility_test"] == "pending" and (product_zone or f["grade"] == "Grade A"):
            imp = "release_after_sterility_test_and_assessment"
        else:
            imp = "release_with_documented_assessment"
        if f["media_fill"] == "failed_last" or (f["excursions_30d"] >= 4 and f["grade"] == "Grade A"):
            corr = "suspend_line"
        elif f["media_fill"] == "overdue" or (product_zone and f["intervention"] == "unrecorded"):
            corr = "media_fill_requalification"
        elif objectionable or f["excursions_30d"] >= 3:
            corr = "cleaning_disinfection_review"
        else:
            corr = "operator_retraining"
        c = 0
        if f["grade"] == "Grade A":
            c += 1
        if product_zone:
            c += 1
        if objectionable:
            c += 1
        if f["media_fill"] != "passed_recent" or f["intervention"] == "unrecorded":
            c += 1
        if not self._action(f):
            c = min(c, 1)
        t = f["excursions_30d"]
        pat = 0 if t == 0 else 1 if t <= 1 else 2 if t <= 3 else 3
        return {"em_batch_impact": imp, "corrective_scope": corr, "sterility_assurance_concern": min(3, c),
                "excursion_pattern": pat, "product_zone_affected": "yes" if product_zone else "no"}

    def _rules(self, rng):
        return [
            pick(rng, ["EM-SOP-007 §3: Grade A action limit: any growth (1 CFU or more) on any sample. Grade B action "
                       "limit: 5 CFU (settle or contact plate) or 10 CFU/m3 (active air); alert at 3 CFU / 5 CFU/m3.",
                       "EM-SOP-007 §3: limits: Grade A, action at >= 1 CFU for every sample type; Grade B, alert 3 / action "
                       "5 CFU per plate and alert 5 / action 10 CFU per m3 of air."]),
            "EM-SOP-007 §5: every action-level result needs identification to species and a batch impact assessment. "
            "Below action level, document and trend; no batch hold.",
            pick(rng, ["EM-SOP-007 §6: reject the batch when a recovery in the critical zone during open filling is a "
                       "spore-former, mould or Gram-negative, or follows an unrecorded intervention, or when the last media "
                       "fill failed. A Grade A recovery otherwise allows release only with a passed sterility test and a "
                       "documented assessment. Pending identification, or three or more excursions on the line in 30 "
                       "days, extend the investigation before any decision.",
                       "EM-SOP-007 §6: critical-zone recovery during filling of an objectionable organism (spore, mould, "
                       "Gram-negative) or after an unlogged intervention, or a failed last media fill = reject. Other "
                       "Grade A recoveries: release only after a passing sterility test plus assessment. ID pending or a "
                       "trend of 3+ excursions in 30 days: investigation extended first."]),
        ]

    def render(self, f, rng, surface):
        ds = _ds(rng)
        grade, where, _ = LOCATIONS[f["location"]]
        org = f["organism"] if f["organism_id"] == "identified" else "Gram-positive rod, identification pending"
        timing = {"during_fill": "while vials were open on the line", "setup": "during line set-up, before any vials were open",
                  "after_fill": "after the last vial was stoppered"}[f["timing"]]
        interv = {"recorded": "an intervention to clear a fallen stopper was logged two minutes before",
                  "unrecorded": "camera shows an operator reaching into the RABS; no intervention was logged",
                  "none": "no interventions in the hour before"}[f["intervention"]]
        media = {"passed_recent": "last media fill passed four months ago", "overdue": "media fill for this line is two months overdue",
                 "failed_last": "the last media fill on this line had one positive unit and is under investigation"}[f["media_fill"]]
        ster = "sterility test pending (day 9 of 14)" if f["sterility_test"] == "pending" else "sterility test passed"
        trend = f"{f['excursions_30d']} other action-level excursion(s) on this line in the last 30 days"
        if surface == "json":
            return dumps(prune({
                "batch": {"product": f["product"], "batch": f["batch"], "line": f["line"], "filled": fmt_date(f["date"].date(), style="iso")},
                "excursion": {"grade": grade, "sample": where, "cfu": f["cfu"], "organism": org, "timing": timing,
                              "interventions": interv},
                "line_history": {"trend": trend, "media_fill": media},
                "testing": ster,
                "sop": self._rules(rng),
            }))
        if surface == "em_deviation_report":
            head = kv_lines([("Deviation", ident(rng, "DEV", 5, "-")), ("Product / batch", f"{f['product']} / {f['batch']}"),
                             ("Line", f["line"]), ("Date", fmt_date(f["date"].date(), style=ds)), ("Grade", grade),
                             ("Sample", where), ("Count", f"{f['cfu']} CFU"), ("Organism", org), ("Timing", timing),
                             ("Interventions", interv), ("Trend", trend), ("Media fill", media), ("Sterility", ster)], rng)
            return f"ENVIRONMENTAL MONITORING DEVIATION -- {f['company']}\n{head}\n\n" + "\n".join(self._rules(rng))
        para = [f"During the fill of {f['batch']} ({f['product']}) on {f['line']}, the {where} ({grade}) grew "
                f"{f['cfu']} CFU, {org}.", f"It was exposed {timing}; {interv}.", _cap(trend) + "; " + media + ".",
                _cap(ster) + "."]
        if f["regime"] == "conflicting":
            para.append(pick(rng, ["Production's view is that a single skin organism is routine and the batch should go.",
                                   "The micro lab notes the plate may have been knocked during retrieval.",
                                   "Operations points out the line passed its last three batches."]))
        return (f"MEMO -- EM impact assessment\nFrom: {f['micro']} (Microbiology)\nTo: {f['qa']} (QA)\n\n"
                + noisy(join_sentences(para), rng, 0.1) + "\n\nSOP:\n" + bullets(self._rules(rng), rng))

    def questions(self):
        return {
            "em_batch_impact": choice("What should QA decide about the batch filled during this excursion?", {
                "release_with_documented_assessment": "Release it with the impact assessment documented.",
                "release_after_sterility_test_and_assessment": "Release it only after the sterility test passes and the assessment is approved.",
                "extend_investigation_before_decision": "Keep it on hold and extend the investigation before deciding.",
                "reject_batch": "Reject the batch.",
            }),
            "corrective_scope": choice("What corrective action does the event call for on the line?", {
                "operator_retraining": "Retrain and requalify the operators involved in aseptic technique.",
                "cleaning_disinfection_review": "Review cleaning and disinfection, including sporicide use and rotation.",
                "media_fill_requalification": "Requalify the line with a media fill before further batches.",
                "suspend_line": "Stop aseptic filling on the line until the investigation closes.",
            }),
            "sterility_assurance_concern": score("How much does the excursion undermine sterility assurance for this batch?", [
                "Little: a below-action result away from the product.",
                "Some: an action-level result with no route to the product.",
                "Considerable: near the product or an objectionable organism.",
                "Severe: an objectionable organism at the product during filling, or a failed control.",
            ]),
            "excursion_pattern": score("Is the excursion isolated or part of a pattern on this line?", [
                "Isolated: no other excursions in the last month.",
                "Occasional: one other excursion recently.",
                "Recurring: two or three other excursions recently.",
                "Persistent: an adverse trend of four or more excursions.",
            ]),
            "product_zone_affected": noul(
                "The contamination was recovered where it could reach exposed product.",
                yes="The sample was in the critical zone (needles, stopper bowl, gloves at the open line) while containers were open.",
                no="The sample was away from exposed product, or taken when no containers were open."),
        }


# =========================================================================== schema 4
# product type -> (storage text, freeze sensitive, budget hours at <=25 C, cold chain?)
CC_PRODUCTS = {
    "monoclonal antibody concentrate": ("2-8 C", True, 72, True), "vaccine suspension": ("2-8 C", True, 48, True),
    "insulin cartridges": ("2-8 C", True, 120, True), "peptide pre-filled syringes": ("2-8 C", True, 96, True),
    "antibiotic eye drops": ("2-8 C", False, 168, True), "enzyme powder for infusion": ("2-8 C", False, 120, True),
    "heparin solution": ("2-8 C", False, 96, True),
}
CC_NAMES = {
    "monoclonal antibody concentrate": ["Ardelimab 100 mg/mL concentrate", "Velutumab 20 mg/mL concentrate",
                                        "Corizumab 150 mg/mL concentrate"],
    "vaccine suspension": ["Prenovax suspension for injection", "Imuvelle suspension for injection", "Tetrasure suspension"],
    "insulin cartridges": ["Glyvana 100 units/mL cartridges", "Insulex 100 units/mL cartridges"],
    "peptide pre-filled syringes": ["Nexudrin 60 mg/0.6 mL solution in pre-filled syringe", "Peptarel 10 mg/0.5 mL pre-filled syringe"],
    "antibiotic eye drops": ["Calvestrin 0.5 mg/mL eye drops", "Oftacyn 0.3% eye drops"],
    "enzyme powder for infusion": ["Enzavel 35 mg powder for infusion", "Lysenza 50 mg powder for infusion"],
    "heparin solution": ["Hepralux 5,000 IU/mL solution for injection", "Coagrin 1,000 IU/mL solution for injection"],
}
LANES = ["Port Ardrey hub to Northcote DC by road", "airport cargo terminal to the Westley DC", "cross-border truck via Kessler",
         "rail-and-road via Grenmouth", "ocean reefer then road from the Corran docks", "regional courier van, three drops"]


class ColdChainExcursion(Schema):
    name = "cold_chain_excursion"
    description = "A refrigerated batch received at a distribution site after a temperature excursion: release to stock, alert, quarantine or reject."
    surfaces = {"json": 0.5, "logger_summary": 0.25, "receiving_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        ptype = pick(rng, list(CC_PRODUCTS))
        storage, freeze, budget, _ = CC_PRODUCTS[ptype]
        tmax = round(near(rng, 14, 7, lo=8.2, hi=30), 1)
        tmin = round(near(rng, 2.0, 2.2, lo=-4, hi=6), 1)
        hours = round(near(rng, budget * 0.25, budget * 0.2, lo=0.5), 1)
        prior = round(budget * near(rng, 0.35, 0.3, lo=0.0, hi=1.2), 1)
        calib = pick_w(rng, {"valid": 9, "expired": 1})
        winter_dwell = chance(rng, 0.3)
        gap = maybe(rng, round(between(rng, 0.5, 6), 1), 0.55 if winter_dwell else 0.8)
        second = pick_w(rng, {None: 7, "agrees": 3, "disagrees": 1})
        if reg == "clear":
            if chance(rng, 0.5):
                tmax, tmin, hours, prior, calib, gap, second = round(between(rng, 9, 14), 1), round(between(rng, 3, 5), 1), round(between(rng, 1, 6), 1), round(budget * between(rng, 0, 0.2), 1), "valid", None, None
                winter_dwell = False
            else:
                tmin, winter_dwell = round(between(rng, -3.5, -0.5), 1), True
                freeze = True
                ptype = pick(rng, [p for p, v in CC_PRODUCTS.items() if v[1]])
                storage, _, budget, _ = CC_PRODUCTS[ptype]
        elif reg == "borderline":
            prior = round(budget * near(rng, 0.6, 0.2, lo=0.0), 1)
            hours = round(near(rng, (budget - prior) * 0.95, budget * 0.15, lo=0.5), 1)
        else:
            twist = pick(rng, ["expired_logger", "hot_spike", "second_disagrees", "gap_in_cold"])
            if twist == "expired_logger":
                calib = "expired"
            elif twist == "hot_spike":
                tmax, hours = round(between(rng, 25.5, 31), 1), round(between(rng, 0.5, 3), 1)
            elif twist == "second_disagrees":
                second = "disagrees"
            else:
                gap, winter_dwell, tmin = round(between(rng, 1.5, 6), 1), True, round(between(rng, 0.8, 2.5), 1)
        return {"regime": reg, "product_type": ptype, "storage": storage, "freeze_sensitive": freeze,
                "budget_h": budget, "max_c": tmax, "min_c": tmin, "excursion_h": hours, "prior_h": prior,
                "logger_calibration": calib, "gap_h": gap, "second_logger": second, "winter_dwell": winter_dwell,
                "product": pick(rng, CC_NAMES[ptype]),
                "batch": _batch(rng), "lane": pick(rng, LANES), "shipment": ident(rng, "SHP", 6, "-"),
                "receiver": person(rng), "qa": person(rng), "company": _company(rng),
                "received": rand_ts(rng, hours=(5, 19)), "stock_days": pick(rng, [None, None, 3, 7, 30])}

    def _cumulative(self, f):
        return f["prior_h"] + f["excursion_h"] + (f["gap_h"] or 0.0)

    def expected(self, f):
        """Intended answers from the latent factors (spread audit only; never rendered)."""
        cum = self._cumulative(f)
        used = cum / f["budget_h"]
        trusted = f["logger_calibration"] == "valid" and f["second_logger"] != "disagrees"
        frozen = f["min_c"] <= 0.0
        if f["freeze_sensitive"] and frozen and trusted:
            disp = "reject_and_destroy"
        elif not trusted or f["max_c"] > 25.0 or used > 1.0 or (f["freeze_sensitive"] and frozen):
            disp = "quarantine_for_stability_assessment"
        elif used > 0.75:
            disp = "release_with_budget_recorded_and_alert"
        else:
            disp = "release_to_stock"
        cons = 0 if used <= 0.25 else 1 if used <= 0.75 else 2 if used <= 1.0 else 3
        trust = 3
        if f["logger_calibration"] != "valid":
            trust -= 2
        if f["gap_h"]:
            trust -= 1
        if f["second_logger"] == "disagrees":
            trust -= 1
        within = f["max_c"] <= 25.0 and used <= 1.0 and not (f["freeze_sensitive"] and frozen)
        freeze_possible = f["min_c"] <= 0.5 or (f["winter_dwell"] and f["gap_h"] is not None)
        return {"excursion_disposition": disp, "stability_budget_consumed": cons, "logger_data_trust": max(0, trust),
                "within_allowable_excursion": "yes" if within else "no",
                "freezing_exposure_possible": "yes" if freeze_possible else "no"}

    def _rules(self, rng, f):
        fr = (" Product is freeze-sensitive: any reading at or below 0 C on a trusted logger means rejection."
              if f["freeze_sensitive"] else " Product is not freeze-sensitive.")
        return [
            pick(rng, [f"GDP-SOP-022 §4: label storage {f['storage']}. Stability data allow a cumulative {f['budget_h']} h "
                       f"between 8 and 25 C over the batch's life; nothing above 25 C.{fr}",
                       f"GDP-SOP-022 §4: storage {f['storage']}; allowable excursion budget {f['budget_h']} h total at 8-25 C, "
                       f"counted across every excursion the batch has had; any reading over 25 C is outside the data.{fr}"]),
            "GDP-SOP-022 §5: a recording gap counts against the budget for its full length. Data from a logger with an "
            "expired calibration, or contradicted by a second logger, cannot support release.",
            pick(rng, ["GDP-SOP-022 §6: within budget, release to stock and record the hours used; when more than 75% of "
                       "the budget has been used, also alert QA and flag the batch. Outside the data or untrusted data: "
                       "quarantine for a stability assessment.",
                       "GDP-SOP-022 §6: release within budget (hours logged); over 75% used means release plus a QA alert "
                       "and batch flag; anything outside the stability data, or unreliable data, goes to quarantine."]),
        ]

    def render(self, f, rng, surface):
        ds = _ds(rng)
        gap = f"logger gap of {f['gap_h']} h during transit" if f["gap_h"] else None
        second = {None: None, "agrees": "second logger in the same shipper agrees within 0.5 C",
                  "disagrees": "second logger in the same shipper shows a much smaller excursion (max 9.1 C)"}[f["second_logger"]]
        calib = ("calibration certificate valid" if f["logger_calibration"] == "valid"
                 else "calibration certificate expired two months before shipment")
        dwell = "shipment sat on an outdoor apron for about 3 h at -6 C ambient" if f["winter_dwell"] else None
        stock = f"site runs out of this product in {f['stock_days']} days" if f["stock_days"] else None
        if surface == "json":
            return dumps(prune({
                "shipment": {"id": f["shipment"], "product": f["product"], "product_type": f["product_type"],
                             "batch": f["batch"], "lane": f["lane"], "received": fmt_ts(f["received"], style="iso")},
                "logger": {"max_c": f["max_c"], "min_c": f["min_c"], "hours_between_8_and_25_c": f["excursion_h"],
                           "calibration": calib, "gap": gap, "second_logger": second},
                "batch_history": {"excursion_hours_already_used": f["prior_h"]},
                "context": {"handling_note": dwell, "supply": stock},
                "sop": self._rules(rng, f),
            }))
        if surface == "logger_summary":
            rows = [["Max", f"{f['max_c']} C"], ["Min", f"{f['min_c']} C"], ["Time 8-25 C", f"{f['excursion_h']} h"],
                    ["Calibration", calib], ["Gap", gap or "none"], ["Second logger", second or "none in shipper"]]
            head = kv_lines([("Shipment", f["shipment"]), ("Product", f"{f['product']} ({f['product_type']})"),
                             ("Batch", f["batch"]), ("Lane", f["lane"]), ("Received", fmt_ts(f["received"], rng)),
                             ("Excursion hours already on batch record", f["prior_h"]), ("Handling", dwell), ("Supply", stock)], rng)
            return (f"TEMPERATURE LOGGER SUMMARY -- {f['company']}\n{head}\n\n" + table(["reading", "value"], rows, rng)
                    + "\n\n" + "\n".join(self._rules(rng, f)))
        para = [f"Shipment {f['shipment']} ({f['product']}, batch {f['batch']}) arrived {fmt_ts(f['received'], rng)} via {f['lane']}.",
                f"Logger: max {f['max_c']} C, min {f['min_c']} C, {f['excursion_h']} h between 8 and 25 C; {calib}."]
        if gap:
            para.append(_cap(gap) + ".")
        if second:
            para.append(_cap(second) + ".")
        if dwell:
            para.append(_cap(dwell) + ".")
        para.append(f"The batch record already shows {f['prior_h']} h of excursion time from earlier legs.")
        if stock:
            para.append(_cap(stock) + ".")
        return "\n".join([f"Subject: Temperature excursion on receipt -- {f['batch']}", f"From: {f['receiver']} (receiving)",
                          f"To: {f['qa']} (QA)", "", noisy(join_sentences(para), rng, 0.1), "", "Ref:",
                          bullets(self._rules(rng, f), rng)])

    def questions(self):
        return {
            "excursion_disposition": choice("What should be done with this batch on receipt?", {
                "release_to_stock": "Release it to saleable stock and record the excursion hours.",
                "release_with_budget_recorded_and_alert": "Release it, record the hours, and alert QA that the budget is nearly used.",
                "quarantine_for_stability_assessment": "Quarantine it pending a stability assessment.",
                "reject_and_destroy": "Reject and destroy the batch.",
            }),
            "stability_budget_consumed": score("How much of the batch's allowable excursion budget has now been used?", [
                "A quarter or less.",
                "More than a quarter, up to three quarters.",
                "More than three quarters, but within the budget.",
                "More than the whole budget.",
            ]),
            "logger_data_trust": score("How far can the temperature record be trusted?", [
                "Not at all: an uncalibrated logger with further problems.",
                "Little: an uncalibrated logger, or several problems.",
                "Mostly: one problem, such as a gap or a disagreeing second logger.",
                "Fully: calibrated, complete and corroborated or uncontradicted.",
            ]),
            "within_allowable_excursion": noul(
                "The batch's total excursion stays within what the stability data allow.",
                yes="No reading above 25 C, no freezing of a freeze-sensitive product, and cumulative hours (gaps included) within the budget.",
                no="A reading is outside the range, or cumulative hours exceed the budget."),
            "freezing_exposure_possible": noul(
                "The product may have been exposed to freezing temperatures."),
        }


# =========================================================================== schema 5
# defect -> (class, source lead)
DEFECTS = {
    "glass lamellae": ("critical", "primary_container_supplier"), "cracked vial neck": ("critical", "primary_container_supplier"),
    "stopper fragment": ("critical", "primary_container_supplier"), "visible fibre": ("major", "filling_process"),
    "low fill volume": ("major", "filling_process"), "misplaced stopper": ("major", "filling_process"),
    "protein-like particles": ("critical", "formulation_or_product"), "haze in solution": ("major", "formulation_or_product"),
    "cosmetic scratch": ("minor", "primary_container_supplier"), "crimp cosmetic defect": ("minor", "filling_process"),
}
# sample-size plan: (lot size range text, n, Ac major, Ac minor)
AQL_PLAN = [("10,001-35,000", 315, 5, 21), ("35,001-150,000", 500, 7, 21), ("3,201-10,000", 200, 3, 14)]


class VisualInspectionAql(Schema):
    name = "visual_inspection_aql"
    description = "A parenteral batch after 100% visual inspection and AQL sampling: release, reinspect, tighten, investigate or reject."
    surfaces = {"json": 0.5, "inspection_record": 0.25, "qa_discussion": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        lots, n, ac_maj, ac_min = pick(rng, AQL_PLAN)
        lo_hi = {315: (12000, 34000), 500: (40000, 140000), 200: (3500, 9800)}[n]
        size = rng.randint(*lo_hi)
        crit = pick_w(rng, {0: 8, 1: 2, 2: 1})
        maj = max(0, int(near(rng, ac_maj, 3, integer=True)))
        minr = max(0, int(near(rng, ac_min * 0.7, 8, integer=True)))
        rates = {"critical": round(near(rng, 0.22, 0.22, lo=0.0), 2), "major": round(near(rng, 0.8, 0.6, lo=0.05), 2),
                 "minor": round(near(rng, 1.6, 1.2, lo=0.1), 2)}
        reinspected = chance(rng, 0.2)
        product = pick(rng, STERILE)
        defect = pick_w(rng, {d: (1 if DEFECTS[d][0] == "critical" else 2) for d in DEFECTS
                              if not ("suspension" in product and d == "haze in solution")})
        new_comp = chance(rng, 0.3)
        if reg == "borderline" and chance(rng, 0.3):          # a cosmetic-defect batch near its minor limits
            product = pick(rng, STERILE)
            defect = pick(rng, ["cosmetic scratch", "crimp cosmetic defect"])
            crit, maj = 0, rng.randint(0, ac_maj // 2)
            minr = int(near(rng, ac_min, 5, lo=ac_min // 2, integer=True))
            rates = {"critical": round(between(rng, 0.02, 0.25), 2), "major": round(between(rng, 0.1, 0.9), 2),
                     "minor": round(near(rng, 2.0, 0.8, lo=0.5), 2)}
        elif reg == "borderline" and chance(rng, 0.4):
            c = pick(rng, ["critical", "major", "minor"])
            act = {"critical": 0.5, "major": 1.5, "minor": 3.0}[c]
            rates[c] = round(near(rng, act * 1.2, act * 0.45, lo=0.01), 2)
        if reg == "clear":
            if chance(rng, 0.5):
                crit, maj, minr = 0, rng.randint(0, max(0, ac_maj - 3)), rng.randint(0, ac_min // 2)
                rates = {"critical": round(between(rng, 0.02, 0.15), 2), "major": round(between(rng, 0.1, 0.6), 2),
                         "minor": round(between(rng, 0.3, 1.2), 2)}
                defect = pick(rng, ["cosmetic scratch", "crimp cosmetic defect"])
            else:
                crit, reinspected = rng.randint(1, 3), True
                defect = pick(rng, ["glass lamellae", "cracked vial neck", "protein-like particles"])
        elif reg == "conflicting":
            twist = pick(rng, ["escape", "high_reject_clean_aql", "new_glass"])
            if twist == "escape":
                rates = {"critical": round(between(rng, 0.0, 0.1), 2), "major": round(between(rng, 0.1, 0.4), 2),
                         "minor": round(between(rng, 0.2, 0.9), 2)}
                maj = ac_maj + rng.randint(1, 4)
                defect = pick(rng, ["visible fibre", "low fill volume", "misplaced stopper"])
            elif twist == "high_reject_clean_aql":
                crit, maj = 0, rng.randint(0, ac_maj)
                rates["critical"] = round(between(rng, 0.6, 1.8), 2)
                defect = pick(rng, ["glass lamellae", "protein-like particles", "stopper fragment"])
            else:
                new_comp, defect = True, pick(rng, ["glass lamellae", "cracked vial neck", "cosmetic scratch"])
        return {"regime": reg, "lot_size": size, "lot_range": lots, "sample_n": n, "ac_major": ac_maj, "ac_minor": ac_min,
                "aql_critical": crit, "aql_major": maj, "aql_minor": minr, "reject_rates": rates,
                "reinspected": reinspected, "main_defect": defect, "new_component_lot": new_comp,
                "product": product, "batch": _batch(rng), "inspection": pick(rng, ["automated (AVI)", "manual booths", "semi-automated"]),
                "inspector_lead": person(rng), "qa": person(rng), "company": _company(rng), "date": rand_date(rng)}

    LIMITS = {"critical": (0.3, 0.5), "major": (1.0, 1.5), "minor": (2.0, 3.0)}

    def _aql_ok(self, f):
        return f["aql_critical"] == 0 and f["aql_major"] <= f["ac_major"] and f["aql_minor"] <= f["ac_minor"]

    def expected(self, f):
        """Intended answers from the latent factors (spread audit only; never rendered)."""
        ok = self._aql_ok(f)
        r = f["reject_rates"]
        over_action = any(r[c] > self.LIMITS[c][1] for c in r)
        over_alert = any(r[c] > self.LIMITS[c][0] for c in r)
        if not ok:
            out = "reject_batch" if f["reinspected"] else "reinspect_100_percent"
        elif over_action:
            out = "hold_for_investigation"
        elif over_alert:
            out = "extended_aql_sample"
        else:
            out = "release"
        cls, src = DEFECTS[f["main_defect"]]
        low_rates = all(r[c] <= self.LIMITS[c][0] for c in r)
        if not ok and low_rates:
            lead = "inspection_performance"
        elif src == "primary_container_supplier" and not f["new_component_lot"] and cls == "minor":
            lead = "filling_process"
        else:
            lead = src
        if f["aql_critical"] > 0 or (cls == "critical" and r["critical"] > self.LIMITS["critical"][0]):
            crit = 3                 # critical defects escaped into accepted units, or critical rejects above alert
        elif cls in ("critical", "major") or f["aql_major"] > f["ac_major"] / 2 or r["major"] > self.LIMITS["major"][0]:
            crit = 2
        elif f["aql_minor"] > f["ac_minor"] / 2 or r["minor"] > self.LIMITS["minor"][0]:
            crit = 1
        else:
            crit = 0
        worst = max(r[c] / self.LIMITS[c][1] for c in r)
        sig = 0 if not over_alert else 1 if not over_action else 2 if worst <= 2.0 else 3
        return {"inspection_outcome": out, "defect_source_lead": lead, "defect_criticality": crit,
                "reject_rate_signal": sig, "aql_acceptance_met": "yes" if ok else "no"}

    def _rules(self, rng, f):
        return [
            pick(rng, ["VI-SOP-009 §4: 100% inspection reject-rate limits, alert / action: critical 0.3% / 0.5%; major "
                       "1.0% / 1.5%; minor 2.0% / 3.0%. Above an alert limit, the AQL sample is doubled before release; "
                       "above an action limit, hold the batch for investigation.",
                       "VI-SOP-009 §4: reject-rate limits (alert / action): critical 0.3/0.5%, major 1.0/1.5%, minor "
                       "2.0/3.0%. Alert exceeded: double the AQL sample. Action exceeded: hold and investigate."]),
            f"VI-SOP-009 §5: AQL sample of accepted units for lots of {f['lot_range']}: n = {f['sample_n']}. Accept on "
            f"critical 0, major <= {f['ac_major']}, minor <= {f['ac_minor']}.",
            "VI-SOP-009 §6: an AQL failure sends the batch for one 100% reinspection; failure after reinspection rejects the batch.",
        ]

    def render(self, f, rng, surface):
        ds = _ds(rng)
        r = f["reject_rates"]
        comp = ("vials from a new glass supplier lot were used for this batch" if f["new_component_lot"]
                else "same vial and stopper lots as the previous three batches")
        rein = "this is the AQL after a 100% reinspection" if f["reinspected"] else "first AQL after 100% inspection"
        if surface == "json":
            return dumps(prune({
                "batch": {"product": f["product"], "batch": f["batch"], "lot_size": f["lot_size"], "inspection": f["inspection"],
                          "components": comp, "status": rein},
                "inspection_100pct": {"reject_rate_pct": r, "most_frequent_defect": f["main_defect"]},
                "aql": {"sample_size": f["sample_n"], "critical": f["aql_critical"], "major": f["aql_major"], "minor": f["aql_minor"]},
                "sop": self._rules(rng, f),
            }))
        if surface == "inspection_record":
            rows = [[c, f"{r[c]}%", str({'critical': f['aql_critical'], 'major': f['aql_major'], 'minor': f['aql_minor']}[c])]
                    for c in ("critical", "major", "minor")]
            head = kv_lines([("Product", f["product"]), ("Batch", f"{f['batch']} ({f['lot_size']:,} units)"),
                             ("Inspection", f"{f['inspection']}; {rein}"), ("Date", fmt_date(f["date"], style=ds)),
                             ("Most frequent defect", f["main_defect"]), ("Components", comp)], rng)
            return (f"VISUAL INSPECTION AND AQL RECORD -- {f['company']}\n{head}\n\n"
                    + table(["class", "100% reject rate", f"AQL defects (n={f['sample_n']})"], rows, rng)
                    + "\n\n" + "\n".join(self._rules(rng, f)))
        para = [f"{f['batch']} ({f['product']}, {f['lot_size']:,} units) finished {f['inspection']} inspection; {rein}.",
                f"Reject rates: critical {r['critical']}%, major {r['major']}%, minor {r['minor']}%; most frequent defect: {f['main_defect']}.",
                f"AQL (n={f['sample_n']}): {f['aql_critical']} critical, {f['aql_major']} major, {f['aql_minor']} minor.",
                _cap(comp) + "."]
        if f["regime"] == "conflicting":
            para.append(pick(rng, ["The inspection lead thinks the AQL inspectors are being stricter than the line.",
                                   "Production notes the batch is needed for a tender delivery next week.",
                                   "Engineering says the camera settings were changed on the second shift."]))
        return "\n".join([f"Subject: {f['batch']} visual inspection / AQL -- decision needed", f"From: {f['inspector_lead']}",
                          f"To: {f['qa']} (QA)", "", noisy(join_sentences(para), rng, 0.1), "", "SOP:", bullets(self._rules(rng, f), rng)])

    def questions(self):
        return {
            "inspection_outcome": choice("What should happen to this batch after inspection?", {
                "release": "Release the batch.",
                "reinspect_100_percent": "Send the whole batch for a second 100% inspection.",
                "extended_aql_sample": "Take a doubled AQL sample before deciding on release.",
                "reject_batch": "Reject the batch.",
                "hold_for_investigation": "Hold the batch while the high reject rate is investigated.",
            }),
            "defect_source_lead": choice("Where should the investigation look first for the source of the defects?", {
                "primary_container_supplier": "The vials, stoppers or other primary packaging components.",
                "filling_process": "The filling and stoppering process.",
                "inspection_performance": "The 100% inspection itself, which let defects through.",
                "formulation_or_product": "The product or formulation itself.",
            }),
            "defect_criticality": score("How serious are the defects found in this batch?", [
                "Negligible: cosmetic only, at low levels.",
                "Minor: cosmetic defects at a noticeable level.",
                "Major: defects that affect function or dose but not sterility.",
                "Critical: particles or container damage that could harm a patient.",
            ]),
            "reject_rate_signal": score("What do the 100% inspection reject rates signal?", [
                "Normal: every category within its alert limit.",
                "Elevated: an alert limit exceeded, action limits met.",
                "Out of control: an action limit exceeded, up to twice the limit.",
                "Severe: an action limit exceeded more than twice over.",
            ]),
            "aql_acceptance_met": noul(
                "The AQL sample meets the acceptance criteria.",
                yes="No critical defects, and major and minor counts at or below their acceptance numbers.",
                no="A critical defect was found, or a major or minor count exceeds its acceptance number."),
        }


WORKFLOW = Workflow(
    name="gmp_batch_release",
    title="GMP batch release",
    description="Pharmaceutical QA / qualified-person decisions to certify, hold, rework, quarantine or reject a batch, "
                "judged against written specifications and SOP criteria.",
    domain="pharmaceutical manufacturing / quality assurance",
    schemas=[BatchRecordReleaseReview(), OosInvestigationReview(), EmExcursionImpact(), ColdChainExcursion(),
             VisualInspectionAql()],
    notes="Fictional products, companies and SOPs. Pre-release (or receipt) batch disposition only: no part MRB decisions, "
          "no post-market recalls, no bills or payments, no customer threads.",
)
