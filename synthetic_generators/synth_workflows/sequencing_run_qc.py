"""Sequencing run QC: release, repeat or hold decisions in a (fictional) clinical sequencing laboratory, made
against the laboratory's own written QC SOP.

Five schemas: releasing a whole sequencing run from its lane metrics, passing or repeating one clinical sample
from its coverage and identity metrics, resolving a suspected sample mix-up, deciding whether a library-prep
batch goes on to pooling, and routing a tumour sample whose purity, DNA integrity or depth may limit what can be
reported.

Everything is invented: the instruments ("SQ-04"), flow-cell classes, reagent lots, run and sample ids, assay
names and every metric. Gene symbols in coverage gaps and referral hotspots are real HGNC symbols, used only as
labels. Every limit rendered into a state (Q30 share, clusters passing filter, contamination, depth for a given
allele fraction, tumour-content minimum, no-template-control reads) is this fictional laboratory's LOCAL SOP,
written for the generator; none is a vendor specification or real guidance. A sex check appears only as a
sample-identity metric (sex on the request against sex inferred from the data); it never decides anything
about a person.

Each schema draws a `regime` first. Borderline cases put a metric just either side of its limit (a Q30 share of
79.6% against 80%, contamination of 2.1% against 2.0%); conflicting cases pair clean metrics with a fault in the
run log, a pinned-looking swap that nobody has cross-checked, or a colleague's note urging release under
turnaround pressure. `answer_key` on each schema is a rule-of-thumb key used only for the spread check; it is
never a label.
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
    fmt_ts,
    hexid,
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
    rand_ts,
    regime,
    score,
    show,
    table,
)

# --------------------------------------------------------------------------- shared pools
# (flow-cell class, lanes, expected yield per lane in Gb)
FLOWCELLS = [("high-output patterned flow cell", 4, (280, 420)), ("8-lane production flow cell", 8, (110, 190)),
             ("mid-output flow cell", 2, (70, 140)), ("high-output flow cell, 2 lanes", 2, (300, 520)),
             ("dual-lane patterned flow cell", 2, (420, 640)), ("4-lane rapid flow cell", 4, (90, 160))]
READS = ["2x150", "2x150", "2x100", "2x125", "1x75 + 2x8 index"]
PANEL_ASSAYS = {
    "CardioPanel v3": ["MYH7", "MYBPC3", "TNNT2", "TNNI3", "LMNA", "KCNQ1", "KCNH2", "SCN5A", "RYR2", "PKP2", "DSP",
                       "FLNC", "BAG3"],
    "Hereditary Cancer 84-gene panel": ["BRCA1", "BRCA2", "PALB2", "MLH1", "MSH2", "MSH6", "PMS2", "APC", "TP53",
                                        "CHEK2", "ATM", "CDH1", "PTEN", "RAD51C", "STK11"],
    "Neuro-epilepsy panel v4": ["SCN1A", "SCN2A", "SCN8A", "KCNQ2", "STXBP1", "CDKL5", "PCDH19", "SLC2A1", "GRIN2A",
                                "DEPDC5", "SYNGAP1"],
    "Renal cystic disease panel v2": ["PKD1", "PKD2", "HNF1B", "PKHD1", "UMOD", "NPHP1", "DNAJB11", "GANAB"],
    "Inherited retinal panel v2": ["ABCA4", "USH2A", "RPGR", "RHO", "PRPF31", "EYS", "CRB1", "CEP290", "RPE65"],
    "Hearing loss panel v3": ["GJB2", "SLC26A4", "OTOF", "MYO7A", "TECTA", "CDH23", "STRC", "MYO15A"],
    "Connective tissue panel v2": ["FBN1", "TGFBR1", "TGFBR2", "SMAD3", "COL3A1", "ACTA2", "MYH11", "COL5A1"],
    "Immunodeficiency panel v2": ["BTK", "IL2RG", "RAG1", "RAG2", "ADA", "CYBB", "WAS", "STAT3", "CTLA4"],
    "Skeletal dysplasia panel v1": ["FGFR3", "COL2A1", "COL1A1", "COL1A2", "SOX9", "COMP", "ACAN", "NPR2"],
    "Metabolic disorders panel v3": ["PAH", "GAA", "GALC", "HEXA", "ACADM", "MMUT", "OTC", "CPS1", "PCCB"],
    "Lipid disorders panel v1": ["LDLR", "APOB", "PCSK9", "LDLRAP1", "APOE", "LPL", "ABCG5", "ABCG8"],
    "Neuromuscular panel v5": ["DMD", "CAPN3", "DYSF", "SGCA", "LMNA", "RYR1", "TTN", "COL6A1", "SMN1"],
}
EXOME_INDICATIONS = {
    "developmental delay with seizures": ["SCN2A", "SYNGAP1", "STXBP1", "KCNQ2", "CDKL5", "GRIN2B", "DYRK1A"],
    "multiple congenital anomalies": ["CHD7", "KMT2D", "NIPBL", "ANKRD11", "SOX9", "TBX5", "ZEB2"],
    "intellectual disability": ["ARID1B", "ANKRD11", "KMT2A", "DDX3X", "SETD5", "MED13L", "ADNP"],
    "suspected mitochondrial disease": ["POLG", "SURF1", "NDUFS4", "TWNK", "SCO2", "ETFDH", "PDHA1"],
    "progressive ataxia": ["SPTBN2", "CACNA1A", "SETX", "APTX", "ATM", "SYNE1", "KCNC3"],
    "short stature with skeletal findings": ["FGFR3", "NPR2", "SHOX", "ACAN", "COL2A1", "IGF1R", "PTPN11"],
    "neonatal hypotonia": ["SMN1", "DMPK", "MTM1", "RYR1", "NEB", "ACTA1", "CHRNE"],
}
# approximate coding-exon counts, so a low-coverage exon number always exists in its gene
EXONS = {"MYH7": 40, "MYBPC3": 34, "TNNT2": 16, "TNNI3": 8, "LMNA": 12, "KCNQ1": 16, "KCNH2": 15, "SCN5A": 28,
         "RYR2": 105, "PKP2": 14, "DSP": 24, "FLNC": 48, "BAG3": 4, "BRCA1": 23, "BRCA2": 27, "PALB2": 13, "MLH1": 19,
         "MSH2": 16, "MSH6": 10, "PMS2": 15, "APC": 16, "TP53": 11, "CHEK2": 15, "ATM": 63, "CDH1": 16, "PTEN": 9,
         "RAD51C": 9, "STK11": 10, "SCN1A": 26, "SCN2A": 27, "SCN8A": 27, "KCNQ2": 17, "STXBP1": 20, "CDKL5": 21,
         "PCDH19": 6, "SLC2A1": 10, "GRIN2A": 14, "DEPDC5": 43, "SYNGAP1": 19, "PKD1": 46, "PKD2": 15, "HNF1B": 9,
         "PKHD1": 67, "UMOD": 11, "NPHP1": 20, "DNAJB11": 10, "GANAB": 25, "ABCA4": 50, "USH2A": 72, "RPGR": 19,
         "RHO": 5, "PRPF31": 14, "EYS": 44, "CRB1": 12, "CEP290": 54, "RPE65": 14, "GJB2": 2, "SLC26A4": 21,
         "OTOF": 47, "MYO7A": 49, "TECTA": 24, "CDH23": 70, "STRC": 29, "MYO15A": 66, "FBN1": 66, "TGFBR1": 9,
         "TGFBR2": 8, "SMAD3": 9, "COL3A1": 51, "ACTA2": 9, "MYH11": 43, "COL5A1": 66, "BTK": 19, "IL2RG": 8,
         "RAG1": 2, "RAG2": 2, "ADA": 12, "CYBB": 13, "WAS": 12, "STAT3": 24, "CTLA4": 4, "FGFR3": 18, "COL2A1": 54,
         "COL1A1": 51, "COL1A2": 52, "SOX9": 3, "COMP": 19, "ACAN": 19, "NPR2": 22, "PAH": 13, "GAA": 20, "GALC": 17,
         "HEXA": 14, "ACADM": 12, "MMUT": 13, "OTC": 10, "CPS1": 38, "PCCB": 15, "LDLR": 18, "APOB": 29, "PCSK9": 12,
         "LDLRAP1": 9, "APOE": 4, "LPL": 10, "ABCG5": 13, "ABCG8": 13, "DMD": 79, "CAPN3": 24, "DYSF": 55, "SGCA": 10,
         "RYR1": 106, "TTN": 363, "COL6A1": 35, "SMN1": 9, "GRIN2B": 14, "DYRK1A": 11, "CHD7": 38, "KMT2D": 55,
         "NIPBL": 47, "ANKRD11": 13, "TBX5": 9, "ZEB2": 10, "ARID1B": 20, "KMT2A": 36, "DDX3X": 17, "SETD5": 23,
         "MED13L": 31, "ADNP": 5, "POLG": 23, "SURF1": 9, "NDUFS4": 5, "TWNK": 5, "SCO2": 2, "ETFDH": 13, "PDHA1": 11,
         "SPTBN2": 38, "CACNA1A": 47, "SETX": 26, "APTX": 9, "SYNE1": 146, "KCNC3": 5, "SHOX": 7, "IGF1R": 21,
         "PTPN11": 16, "DMPK": 15, "MTM1": 15, "NEB": 183, "ACTA1": 7, "CHRNE": 12}
EXOME_ASSAYS = ["clinical exome v2", "rapid trio exome (NICU)", "clinical exome v3 (beta)", "exome reanalysis capture"]
DUE_TEST = {"routine": "routine diagnostic", "urgent": "urgent oncology germline", "nicu": "rapid neonatal",
            "prenatal": "prenatal (ongoing pregnancy)"}


def _instrument(rng):
    return f"SQ-{rng.randint(1, 16):02d}"


def _run_id(rng, day, instr):
    fc = "".join(rng.choice("ABCDEFGHJKLMNPRSTVWXYZ0123456789") for _ in range(7))
    return f"{day:%y%m%d}_{instr.replace('-', '')}_{rng.randint(80, 999):04d}_{fc}"


def _sample_id(rng):
    return f"{pick(rng, ['GS', 'SX', 'DN', 'LB'])}{rng.randint(10, 26)}-{rng.randint(1000, 99999):05d}"


def _well(rng):
    return f"{'ABCDEFGH'[rng.randrange(8)]}{rng.randint(1, 12)}"


def _adjacent(well, rng):
    r, c = "ABCDEFGH".index(well[0]), int(well[1:])
    opts = []
    if c < 12:
        opts.append(f"{well[0]}{c + 1}")
    if c > 1:
        opts.append(f"{well[0]}{c - 1}")
    if r < 7:
        opts.append(f"{'ABCDEFGH'[r + 1]}{c}")
    if r > 0:
        opts.append(f"{'ABCDEFGH'[r - 1]}{c}")
    return pick(rng, opts)


def _far_well(well, rng):
    while True:
        w = _well(rng)
        if abs("ABCDEFGH".index(w[0]) - "ABCDEFGH".index(well[0])) + abs(int(w[1:]) - int(well[1:])) > 2:
            return w


def _band(rel):
    """Shortfall bands used across the module: 0 none, 1 marginal (<=5%), 2 clear (<=15%), 3 gross."""
    if rel <= 0:
        return 0
    return 1 if rel <= 0.05 else 2 if rel <= 0.15 else 3


# =========================================================================== schema 1
FAULT_LOGS = ["W-2214 flow-cell valve pressure deviation during cycles {a}-{b}",
              "E-0817 chiller temperature excursion (+3.1 C for 22 min) at cycle {a}",
              "W-1130 air detected in lane {lane} reagent line at cycle {a}",
              "E-3302 focus drift on lane {lane}, tiles 1201-1312",
              "W-0450 excitation power below spec during read 1, cycles {a}-{b}",
              "E-1906 low flow rate on lane {lane} from cycle {a}",
              "W-2740 reagent cartridge temperature out of range at load",
              "E-5121 stage movement retried 14 times on lane {lane}",
              "E-0611 pump stall alarm at cycle {a}, auto-recovered"]
BENIGN_LOGS = ["W-0090 network share unreachable for 4 min (upload retried)",
               "W-0312 door opened during setup (before priming)", "I-0420 reagent lot registered",
               "W-0077 instrument PC disk at 81%", "I-2001 run completed, data transfer verified",
               "W-0145 operator login timed out during run (no effect on chemistry)",
               "I-0733 wash completed after run"]
OPERATOR_NOTES = ["Loaded at the usual concentration.", "Cluster preview looked dense on lane 1.",
                  "First run on the new reagent lot.", "Run finished overnight; nothing odd at unload.",
                  "Spike-in control added at 1%.", "Pool quantified twice; values agreed.",
                  "Flow cell had sat in the fridge 3 days before loading.", "Loaded by a trainee under supervision."]


class RunReleaseReview(Schema):
    name = "run_release_review"
    description = ("A completed sequencing run's lane metrics, log and sample list, reviewed against the lab's run "
                   "release SOP: release, release some lanes, requeue or hold.")
    surfaces = {"json": 0.5, "run_summary": 0.25, "qc_review_email": 0.25}

    MIN_METRICS = ("q30", "pf", "yield")
    LANE_COMBOS = [("pf", "q30", "yield"), ("undet",), ("q30", "err"), ("yield",), ("pf",), ("q30",)]
    WIDE_CAUSES = {"overclustering": ("pf", "q30"), "underloading": ("yield",), "chemistry": ("q30", "err")}

    def sample_factors(self, rng):
        reg = regime(rng)
        fc, n_lanes, (ylo, yhi) = pick(rng, FLOWCELLS)
        exp_yield = rng.randint(ylo, yhi)
        lim = {"q30": pick(rng, [75.0, 80.0, 85.0]), "pf": pick(rng, [60.0, 65.0, 70.0]),
               "yield": pick(rng, [80, 85, 90]), "err": pick(rng, [1.0, 1.5, 2.0]), "undet": pick(rng, [8.0, 10.0, 12.0])}
        twist = None
        if reg == "clear":
            mode = pick_w(rng, {"none": 3, "lane_local": 3.5, "run_wide": 2.5})
            sev = None if mode == "none" else pick(rng, ["clear", "gross", "gross"])
            fault = (mode == "run_wide" and chance(rng, 0.5)) or (mode == "lane_local" and chance(rng, 0.3))
        elif reg == "borderline":
            mode = pick_w(rng, {"none": 2.5, "lane_local": 4, "run_wide": 3})
            sev = None if mode == "none" else pick(rng, ["marginal", "marginal", "clear"])
            fault = mode != "none" and chance(rng, 0.4)
        else:
            twist = pick(rng, ["clean_but_log", "lane_with_log", "wide_pressure", "prev_failed_lane"])
            mode = {"clean_but_log": "none", "lane_with_log": "lane_local", "wide_pressure": "run_wide",
                    "prev_failed_lane": "lane_local"}[twist]
            sev = None if mode == "none" else pick(rng, ["marginal", "clear", "clear", "gross"])
            fault = twist != "wide_pressure"
        # which lanes and metrics fail
        bad_lanes = []
        bad_metrics = ()
        cause = None
        if mode == "lane_local":
            k = 2 if n_lanes >= 4 and chance(rng, 0.3) else 1
            bad_lanes = sorted(rng.sample(range(1, n_lanes + 1), k))
            bad_metrics = pick(rng, self.LANE_COMBOS)
            cause = "lane"
        elif mode == "run_wide":
            cause = pick(rng, ["chemistry", "chemistry", "overclustering"] if fault
                         else ["overclustering", "underloading"])
            bad_lanes = list(range(1, n_lanes + 1))
            bad_metrics = self.WIDE_CAUSES[cause]
        lanes = []
        for ln in range(1, n_lanes + 1):
            m = {"lane": ln,
                 "q30": round(rng.uniform(max(lim["q30"] + 3.5, 86.0), 94.6), 1),
                 "pf": round(rng.uniform(lim["pf"] + 5, min(lim["pf"] + 20, 88)), 1),
                 "yield": int(round(exp_yield * rng.uniform(lim["yield"] / 100 + 0.05, 1.14))),
                 "err": round(rng.uniform(0.22, lim["err"] * 0.65), 2),
                 "undet": round(rng.uniform(1.2, lim["undet"] * 0.6), 1)}
            if ln in bad_lanes:
                for met in bad_metrics:
                    rel = {"marginal": rng.uniform(0.006, 0.048), "clear": rng.uniform(0.065, 0.14),
                           "gross": rng.uniform(0.18, 0.4)}[sev]
                    if met in ("err", "undet"):
                        rel = rel * (2.5 if sev == "gross" else 1.0)
                        m[met] = round(lim[met] * (1 + rel), 2 if met == "err" else 1)
                    elif met == "yield":
                        m["yield"] = int(exp_yield * lim["yield"] / 100 * (1 - rel))
                    else:
                        m[met] = round(lim[met] * (1 - rel), 1)
            lanes.append(m)
        if reg == "borderline":
            # a metric sitting just on the passing side somewhere, so the reviewer has to check each limit
            ln = pick(rng, lanes)
            met = pick(rng, ["q30", "pf", "err", "undet"])
            if met not in bad_metrics or ln["lane"] not in bad_lanes:
                if met in ("err", "undet"):
                    ln[met] = round(lim[met] * rng.uniform(0.93, 0.995), 2 if met == "err" else 1)
                else:
                    ln[met] = round(lim[met] * rng.uniform(1.003, 1.03), 1)
        # fault signals
        fault_kind = None
        log = rng.sample(BENIGN_LOGS, rng.randint(1, 3))
        prev = pick(rng, ["passed", "passed", "passed with one lane re-queued (pool issue)"])
        if fault:
            fault_kind = "prev_run" if twist == "prev_failed_lane" else pick(rng, ["log", "log", "prev_run"])
            if fault_kind == "log":
                tmpl = pick(rng, [t for t in FAULT_LOGS if ("{lane}" in t) == (mode == "lane_local")]
                            if mode != "none" else FAULT_LOGS)
                lane_ref = bad_lanes[0] if bad_lanes else rng.randint(1, n_lanes)
                a = rng.randint(12, 230)
                log.append(tmpl.format(a=a, b=a + rng.randint(4, 40), lane=lane_ref))
            else:
                words = {"q30": "low Q30", "pf": "low clusters passing filter", "yield": "low yield",
                         "err": "high spike-in error", "undet": "high undetermined reads"}
                what = " and ".join(words[m] for m in bad_metrics) if bad_metrics else "low Q30"
                prev = (f"failed: {what} on all lanes (service visit booked)" if mode != "lane_local"
                        else f"failed: {what} on lane {bad_lanes[0]} (fluidics ticket open)")
        rng.shuffle(log)
        n_clin = pick(rng, [0, 2, 6, 6, 11, 18, 24])
        due = pick(rng, [2, 3, 4, 5, 6, 7, 8, 10, 13, 17])
        requeue = pick(rng, [2, 3, 3, 4, 5])
        if twist == "wide_pressure":
            n_clin, due = max(n_clin, 6), rng.randint(1, 4)
        note = pick(rng, OPERATOR_NOTES)
        if twist == "wide_pressure":
            note = pick(rng, ["Clinical team need these by Friday and the numbers are only just under. Release?",
                              "Only a hair below the limits; the oncology samples are already late. Can we release?"])
        day = rand_date(rng)
        instr = _instrument(rng)
        return {"regime": reg, "twist": twist, "mode": mode, "severity": sev, "cause": cause, "fault": fault,
                "fault_kind": fault_kind, "flowcell": fc, "n_lanes": n_lanes, "expected_yield_gb": exp_yield,
                "limits": lim, "lanes": lanes, "bad_lanes": bad_lanes, "log": log, "previous_run": prev,
                "clinical_samples": n_clin, "research_samples": pick(rng, [0, 4, 12, 30, 60]),
                "earliest_due_days": due if n_clin else None, "requeue_days": requeue, "operator_note": note,
                "operator": person(rng), "reviewer": person(rng), "instrument": instr, "read_length": pick(rng, READS),
                "run_id": _run_id(rng, day, instr), "completed": day,
                "assays": rng.sample(list(PANEL_ASSAYS) + EXOME_ASSAYS, rng.randint(1, 3)),
                "reagent_lot": f"RL-{rng.randint(20000, 99999)}", "sop_id": f"SQ-QC-{rng.randint(2, 19):02d}",
                "sop_variant": rng.randrange(2)}

    # ------------------------------------------------------------------ helpers
    def _fails(self, f):
        lim, out = f["limits"], []
        for m in f["lanes"]:
            for met in ("q30", "pf", "err", "undet"):
                v = m[met]
                rel = (lim[met] - v) / lim[met] if met in ("q30", "pf") else (v - lim[met]) / lim[met]
                if rel > 0:
                    out.append((m["lane"], met, rel))
            need = f["expected_yield_gb"] * lim["yield"] / 100
            if m["yield"] < need:
                out.append((m["lane"], "yield", (need - m["yield"]) / need))
        return out

    def _sop(self, f):
        lim = f["limits"]
        if f["sop_variant"] == 0:
            return [f"Lane limits: share of bases >= Q30 at least {lim['q30']:g}%; clusters passing filter at least "
                    f"{lim['pf']:g}%; yield at least {lim['yield']}% of expected; spike-in error at most "
                    f"{lim['err']:g}%; undetermined reads at most {lim['undet']:g}%.",
                    "Release the run when every lane passes. If only some lanes fail and the rest pass, release the "
                    "passing lanes and requeue the failed lanes' libraries. If every lane fails, requeue the run.",
                    "Any missed limit together with a fault signal (an error in the run log, or the same failure on "
                    "the instrument's previous run) means hold: release nothing, requeue nothing, open a ticket."]
        return [f"A lane passes only if Q30 share >= {lim['q30']:g}%, PF >= {lim['pf']:g}%, yield >= {lim['yield']}% "
                f"of the expected figure, spike-in error <= {lim['err']:g}% and undetermined <= {lim['undet']:g}%.",
                "Hold everything while a fault signal (a logged instrument error, or the same failure on this "
                "instrument's last run) accompanies any missed limit.",
                "Without a fault signal: all lanes pass -> release; some lanes fail -> keep the passing lanes and "
                "requeue the rest; all lanes fail -> requeue the whole run."]

    def _samples_text(self, f):
        if not f["clinical_samples"]:
            return f"research only ({f['research_samples']} samples)"
        return (f"{f['clinical_samples']} clinical (earliest report due in {f['earliest_due_days']} days) and "
                f"{f['research_samples']} research")

    def render(self, f, rng, surface):
        lim = f["limits"]
        sop = self._sop(f)
        exp = f["expected_yield_gb"]
        if surface == "json":
            return dumps(prune({
                "run": {"id": f["run_id"], "instrument": f["instrument"], "flow_cell": f["flowcell"],
                        "read_length": f["read_length"], "expected_yield_per_lane_gb": exp,
                        "completed": f["completed"].isoformat(), "reagent_lot": f["reagent_lot"],
                        "assays": ", ".join(f["assays"])},
                "lanes": [{"lane": m["lane"], "q30_pct": m["q30"], "pf_pct": m["pf"], "yield_gb": m["yield"],
                           "spike_in_error_pct": m["err"], "undetermined_pct": m["undet"]} for m in f["lanes"]],
                "run_log": f["log"], "instrument_history": {"previous_run": f["previous_run"]},
                "samples": {"on_run": self._samples_text(f), "resequencing_turnaround_days": f["requeue_days"]},
                "operator_note": f["operator_note"],
                "sop": {"id": f"{f['sop_id']} (local)", "rules": sop},
            }))
        rows = [[m["lane"], f"{m['q30']:.1f}", f"{m['pf']:.1f}", m["yield"], f"{m['err']:.2f}", f"{m['undet']:.1f}"]
                for m in f["lanes"]]
        tab = table(["lane", "%>=Q30", "%PF", "Gb", "err %", "undet %"], rows, rng)
        if surface == "run_summary":
            head = kv_lines([("Run", f["run_id"]), ("Instrument", f"{f['instrument']} ({f['flowcell']})"),
                             ("Reads", f["read_length"]), ("Completed", fmt_date(f["completed"], rng)),
                             ("Expected yield", f"{exp} Gb per lane"), ("Reagent lot", f["reagent_lot"]),
                             ("Samples", self._samples_text(f)),
                             ("Resequencing turnaround", f"about {f['requeue_days']} days")], rng)
            log = "\n".join(f"  {x}" for x in f["log"])
            return (f"RUN SUMMARY\n{head}\n\n{tab}\n\nRun log (warnings and errors):\n{log}\n"
                    f"Previous run on {f['instrument']}: {f['previous_run']}\nOperator ({f['operator']}): "
                    f"{f['operator_note']}\n\n{f['sop_id']} (local):\n" + "\n".join(sop))
        body = [f"Subject: run review {f['run_id']}", f"From: {f['operator']}", f"To: {f['reviewer']}", "",
                f"Hi {f['reviewer'].split()[0]},", "",
                noisy(join_sentences([f"{f['instrument']} finished the {f['flowcell']} run "
                                      f"({', '.join(f['assays'])}) on {fmt_date(f['completed'], rng)}.",
                                      f"On it: {self._samples_text(f)}.", f["operator_note"],
                                      f"A requeue would take about {f['requeue_days']} days."]), rng, 0.12), "",
                tab, "", "Log warnings/errors: " + "; ".join(f["log"]) + ".",
                f"Last run on this instrument: {f['previous_run']}.", "", f["operator"].split()[0], "",
                f"[{f['sop_id']} (local)] " + " ".join(sop)]
        return "\n".join(body)

    def questions(self):
        return {
            "run_disposition": choice("What should happen to this run's data?", {
                "hold_for_root_cause": "Hold all of it, release and requeue nothing, until the suspected instrument "
                                       "or reagent fault is understood.",
                "release_full_run": "Release every lane for analysis.",
                "release_passing_lanes": "Release the lanes that meet every limit and requeue the libraries from "
                                         "the failed lanes.",
                "requeue_whole_run": "Release nothing and requeue every library on a new run.",
            }),
            "threshold_shortfall": score("How far does the worst lane metric fall short of its SOP limit?", [
                "Not at all: every lane meets every limit.",
                "Marginally: the worst miss is within about 5% of the limit.",
                "Clearly: the worst miss is between about 5% and 15% of the limit.",
                "Grossly: the worst miss is more than about 15% of the limit.",
            ]),
            "turnaround_pressure": score("How tight would the clinical turnaround be if this run's data had to be "
                                         "regenerated?", [
                "No pressure: research samples only, or due dates far beyond a resequencing run.",
                "Some: a resequencing run fits, with a few days to spare.",
                "Tight: a resequencing run would only just make the earliest clinical due date.",
                "Already lost: a resequencing run would miss the earliest clinical due date.",
            ]),
            "failure_confined_to_lanes": noul(
                "The run's metric failures are confined to particular lanes while the other lanes pass.",
                yes="Some lanes miss a limit and every remaining lane meets all of them.",
                no="Every lane misses a limit, or no lane misses any."),
            "instrument_fault_suspected": noul("The record shows signs of an instrument or chemistry fault on this "
                                               "run."),
        }

    def answer_key(self, f):
        """Rule-of-thumb answer key for the spread check only; never used as a label."""
        fails = self._fails(f)
        lanes_failed = sorted({ln for ln, _, _ in fails})
        if not fails:
            disp = "release_full_run"
        elif f["fault"]:
            disp = "hold_for_root_cause"
        elif len(lanes_failed) < f["n_lanes"]:
            disp = "release_passing_lanes"
        else:
            disp = "requeue_whole_run"
        worst = max((r for _, _, r in fails), default=0)
        if not f["clinical_samples"]:
            tp = 0
        else:
            slack = f["earliest_due_days"] - f["requeue_days"]
            tp = 0 if slack >= 6 else 1 if slack >= 3 else 2 if slack >= 0 else 3
        confined = bool(fails) and len(lanes_failed) < f["n_lanes"]
        return {"run_disposition": disp, "threshold_shortfall": _band(worst), "turnaround_pressure": tp,
                "failure_confined_to_lanes": "yes" if confined else "no",
                "instrument_fault_suspected": "yes" if f["fault"] else "no"}


# =========================================================================== schema 2
class SampleCoverageQc(Schema):
    name = "sample_coverage_qc"
    description = ("One clinical sample's coverage, duplication, contamination and identity metrics against the "
                   "lab's per-sample QC SOP: release, fill gaps, top up, re-prep or ask for a new specimen.")
    surfaces = {"json": 0.5, "lims_qc_export": 0.25, "bench_chat": 0.25}

    SCEN = {"clear": {"clean": 2.5, "sex_mismatch": 1.3, "low_depth_complex": 1.5, "low_depth_dup": 1, "gaps_few": 1.2,
                      "contam_sample": 2.6},
            "borderline": {"clean": 1.6, "gaps_few": 2, "low_depth_complex": 2, "contam_near": 5, "low_depth_dup": 1,
                           "gaps_many": 1, "sex_inconclusive": 1},
            "conflicting": {"sex_mismatch": 1.4, "contam_prep": 3.6, "low_depth_dup": 1.5, "gaps_many": 1.5,
                            "contam_sample": 2.5, "sex_inconclusive": 1}}

    def sample_factors(self, rng):
        reg = regime(rng)
        exome = chance(rng, 0.4)
        if exome:
            assay = pick(rng, EXOME_ASSAYS)
            indication = pick(rng, list(EXOME_INDICATIONS))
            genes = EXOME_INDICATIONS[indication]
            lim = {"mean": pick(rng, [80, 100, 120]), "pct20": pick(rng, [95.0, 96.0, 97.0]),
                   "dup": pick(rng, [15.0, 20.0, 25.0]), "contam": pick(rng, [2.0, 3.0]), "dna": pick(rng, [100, 150, 200])}
        else:
            assay = pick(rng, list(PANEL_ASSAYS))
            indication = None
            genes = PANEL_ASSAYS[assay]
            lim = {"mean": pick(rng, [200, 300, 400]), "pct20": pick(rng, [98.0, 99.0, 99.5]),
                   "dup": pick(rng, [30.0, 40.0, 50.0]), "contam": pick(rng, [1.5, 2.0, 3.0]), "dna": pick(rng, [50, 80, 100])}
        max_fill = pick(rng, [3, 5, 6])
        scen = pick_w(rng, self.SCEN[reg])
        mean = int(lim["mean"] * rng.uniform(1.15, 2.2))
        pct20 = round(min(99.9, lim["pct20"] + rng.uniform(0.4, 99.9 - lim["pct20"])), 1)
        dup = round(lim["dup"] * rng.uniform(0.3, 0.8), 1)
        contam = round(lim["contam"] * rng.uniform(0.02, 0.4), 2)
        sex = "match"
        n_gaps = 0
        lib_left = chance(rng, 0.75)
        dna_left = int(lim["dna"] * rng.uniform(1.3, 6))
        neighbours_contam = False
        if scen == "sex_mismatch":
            sex = "mismatch"
        elif scen == "sex_inconclusive":
            sex = "inconclusive"
            if reg == "conflicting":
                contam = round(lim["contam"] * rng.uniform(0.7, 1.4), 2)
        elif scen == "low_depth_complex":
            mean = int(lim["mean"] * (rng.uniform(0.55, 0.9) if reg == "clear" else rng.uniform(0.86, 0.995)))
            pct20 = round(lim["pct20"] - rng.uniform(0.3, 4.0), 1)
            lib_left = True if reg == "clear" else chance(rng, 0.7)
        elif scen == "low_depth_dup":
            mean = int(lim["mean"] * rng.uniform(0.5, 0.95))
            pct20 = round(lim["pct20"] - rng.uniform(0.5, 6.0), 1)
            dup = round(lim["dup"] * rng.uniform(1.15, 1.9), 1)
            dna_left = int(lim["dna"] * pick(rng, [rng.uniform(0.2, 0.9), rng.uniform(1.1, 4)]))
        elif scen == "gaps_few":
            n_gaps = rng.randint(1, max_fill)
        elif scen == "gaps_many":
            n_gaps = rng.randint(max_fill + 1, max_fill + 6)
            if reg == "borderline":
                n_gaps = max_fill + pick(rng, [0, 1, 1])
        elif scen in ("contam_prep", "contam_sample", "contam_near"):
            if scen == "contam_near":
                contam = round(near(rng, lim["contam"] * 1.04, lim["contam"] * 0.15, lo=0.1), 2)
            else:
                contam = round(lim["contam"] * rng.uniform(1.15, 3.5), 2)
            neighbours_contam = scen == "contam_prep" or (scen == "contam_near" and chance(rng, 0.4))
            dna_left = int(lim["dna"] * pick(rng, [rng.uniform(0.3, 0.9), rng.uniform(1.2, 5), rng.uniform(1.2, 5)]))
        if scen.startswith("low_depth") and chance(rng, 0.65):
            n_gaps = rng.randint(1, max_fill + 3)          # thin coverage usually leaves holes in the genes too
        elif scen.startswith("contam") or scen.startswith("sex"):
            n_gaps = rng.randint(1, 3) if chance(rng, 0.3) else 0
        if reg == "borderline" and scen == "clean":
            which = pick(rng, ["mean", "pct20", "dup"])
            if which == "mean":
                mean = int(lim["mean"] * rng.uniform(1.0, 1.06)) + 1
            elif which == "pct20":
                pct20 = round(lim["pct20"] + rng.uniform(0.0, 0.3), 1)
            else:
                dup = round(lim["dup"] * rng.uniform(0.95, 1.0), 1)
        gaps, seen = [], set()
        while len(gaps) < n_gaps:
            g = pick(rng, genes)
            e = rng.randint(1, EXONS.get(g, 20))
            if (g, e) not in seen:
                seen.add((g, e))
                gaps.append([g, e, rng.randint(6, 19)])
        gaps.sort(key=lambda x: (x[0], x[1]))
        tech_note = None
        if scen == "low_depth_dup" and reg == "conflicting":
            tech_note = "Just top it up, the library is still in the freezer."
        elif scen == "sex_mismatch" and reg == "conflicting":
            tech_note = "Everything else is excellent; probably a typo on the request form?"
        elif scen == "contam_prep":
            tech_note = pick(rng, ["Plate was prepped on the older handler deck.", None])
        elif scen == "contam_sample":
            tech_note = pick(rng, ["Collection container arrived with a cracked cap.", None])
        elif scen == "gaps_many" and reg == "conflicting":
            tech_note = "Overall coverage is well over the limit, so I think we can release."
        else:
            tech_note = pick(rng, [None, None, "No issues at prep.", "Low input at extraction, as noted on the request.",
                                   "Sample arrived 2 days late but within stability."])
        plate = None
        if contam >= 0.5 * lim["contam"]:
            plate = ("neighbouring wells on the prep plate also raised (1-2%)" if neighbours_contam
                     else "rest of the prep plate clean (all below 0.5%)")
        return {"regime": reg, "scenario": scen, "exome": exome, "assay": assay, "indication": indication,
                "prep_plate": plate,
                "limits": lim, "max_fill": max_fill, "mean": mean, "pct20": pct20, "dup": dup, "contam": contam,
                "sex_check": sex, "gaps": gaps, "library_left": lib_left, "dna_left_ng": dna_left,
                "neighbours_contaminated": neighbours_contam, "tech_note": tech_note,
                "sample": _sample_id(rng), "run_id": _run_id(rng, rand_date(rng), _instrument(rng)),
                "scientist": person(rng), "tech": person(rng), "insert": rng.randint(160, 340),
                "sop_id": f"SQ-SMP-{rng.randint(2, 30):02d}", "sop_variant": rng.randrange(2),
                "specimen": pick(rng, ["EDTA blood", "EDTA blood", "saliva", "extracted DNA (referral)", "buccal swab"])}

    def _sop(self, f):
        lim, mf = f["limits"], f["max_fill"]
        if f["sop_variant"] == 0:
            return [f"Sample limits: mean depth >= {lim['mean']}x; >= {lim['pct20']:g}% of targets at 20x or more; "
                    f"duplicates <= {lim['dup']:g}%; contamination estimate <= {lim['contam']:g}%; inferred sex must "
                    "agree with the request.",
                    f"Indication genes: every coding exon at 20x or more; up to {mf} exons below that may be filled "
                    "by an orthogonal method, more than that needs more sequencing.",
                    "Short on depth: top up the same library if it remains and duplicates are within limit; "
                    f"otherwise prepare a new library if at least {lim['dna']} ng DNA remains; otherwise request a "
                    "new specimen.",
                    "Contamination over the limit: new library if the prep plate shows the same problem and DNA "
                    "allows, else new specimen. An unexplained sex discrepancy always needs a new specimen."]
        return [f"Per-sample acceptance ({'exome' if f['exome'] else 'panel'}): mean coverage {lim['mean']}x minimum, "
                f"20x breadth {lim['pct20']:g}% minimum, duplication {lim['dup']:g}% maximum, contamination "
                f"{lim['contam']:g}% maximum, sex concordant with the request.",
                f"Gaps in indication genes: fill up to {mf} exons orthogonally; beyond {mf}, sequence more.",
                f"Depth shortfall -> top-up if the library is still available and duplication is in range; else "
                f"re-prep from stored DNA (needs {lim['dna']} ng); else new specimen.",
                "A sex discrepancy without an explanation on file -> new specimen. Contamination above the limit -> "
                "re-prep only when it traces to the prep plate and DNA suffices; otherwise new specimen."]

    def _gaps_text(self, f):
        if not f["gaps"]:
            return "none"
        return "; ".join(f"{g} exon {e} ({d}x)" for g, e, d in f["gaps"])

    def _sex_text(self, f):
        return {"match": "agrees with request", "mismatch": "DISAGREES with request (no explanation on file)",
                "inconclusive": "inconclusive (too few reads on chromosome Y targets)"}[f["sex_check"]]

    def render(self, f, rng, surface):
        sop = self._sop(f)
        test = f["assay"] + (f" ({f['indication']})" if f["indication"] else "")
        mats = (f"library {'remaining' if f['library_left'] else 'used up'}; {f['dna_left_ng']} ng DNA in store")
        if surface == "json":
            return dumps(prune({
                "sample": {"id": f["sample"], "test": test, "specimen": f["specimen"], "run": f["run_id"]},
                "metrics": {"mean_depth": f["mean"], "pct_targets_20x": f["pct20"], "duplication_pct": f["dup"],
                            "contamination_pct": f["contam"], "sex_check": self._sex_text(f),
                            "median_insert_bp": f["insert"], "prep_plate_contamination": f["prep_plate"]},
                "indication_genes_below_20x": [f"{g} exon {e} ({d}x)" for g, e, d in f["gaps"]] or None,
                "materials": mats, "bench_note": f["tech_note"],
                "sop": {"id": f"{f['sop_id']} (local)", "rules": sop},
            }))
        if surface == "lims_qc_export":
            rows = [("Sample", f["sample"]), ("Test", test), ("Specimen", f["specimen"]), ("Run", f["run_id"]),
                    ("Mean depth", f"{f['mean']}x"), ("Targets >= 20x", f"{f['pct20']:.1f}%"),
                    ("Duplicates", f"{f['dup']:.1f}%"), ("Contamination", f"{f['contam']:.2f}%"),
                    ("Sex check", self._sex_text(f)), ("Insert (median)", f"{f['insert']} bp"),
                    ("Prep plate", f["prep_plate"]),
                    ("Indication exons < 20x", self._gaps_text(f)), ("Materials", mats),
                    ("Note", f["tech_note"])]
            return ("LIMS SAMPLE QC EXPORT\n" + kv_lines(rows, rng) + f"\n\n{f['sop_id']} (local):\n"
                    + "\n".join(f"- {x}" for x in sop))
        msgs = [(f["tech"], f"{f['sample']} ({test}) is back: {f['mean']}x mean, {f['pct20']:.1f}% at 20x, dups "
                            f"{f['dup']:.1f}%, contamination {f['contam']:.2f}%"),
                (f["tech"], f"sex check {self._sex_text(f)}"
                 + (f"; {f['prep_plate']}" if f["prep_plate"] else "")),
                (f["scientist"], "any gaps in the indication genes?"),
                (f["tech"], "none" if not f["gaps"] else self._gaps_text(f)),
                (f["scientist"], "what material is left?"), (f["tech"], mats)]
        if f["tech_note"]:
            msgs.append((f["tech"], f["tech_note"]))
        hh, mm = rng.randint(8, 17), rng.randint(0, 30)
        out = ["#ngs-sample-qc"]
        for who, text in msgs:
            mm += rng.randint(1, 5)
            out.append(f"[{hh:02d}:{mm % 60:02d}] {who.split()[0].lower()}: {noisy(text, rng, 0.08)}")
        return "\n".join(out) + f"\n\n(pinned) {f['sop_id']} (local): " + " ".join(sop)

    def questions(self):
        return {
            "sample_qc_outcome": choice("What should happen to this sample's sequencing result?", {
                "new_library_prep": "Prepare a new library from the stored DNA and sequence it again.",
                "pass_release": "Release the data for analysis as it stands.",
                "pass_with_gap_fill": "Release the data and fill the named low-coverage exons with an orthogonal "
                                      "method.",
                "request_new_specimen": "Reject this result and ask the referrer for a new specimen.",
                "top_up_sequencing": "Sequence the existing library deeper before release.",
            }),
            "coverage_adequacy": score("How adequate is the coverage for the test that was requested?", [
                "Inadequate: well short of the depth or breadth limits.",
                "Short: just under a depth or breadth limit.",
                "Adequate with holes: limits met, but exons in the indication genes fall below 20x.",
                "Complete: limits met and every indication exon at 20x or more.",
            ]),
            "identity_concern": score("How much doubt is there that the data come from this patient alone?", [
                "None: contamination is low and the sex check agrees.",
                "Slight: contamination is raised but within the limit, or the sex check is inconclusive.",
                "Material: contamination is over the limit.",
                "Serious: the sex check disagrees with the request, or contamination is far over the limit.",
            ]),
            "contamination_within_limit": noul(
                "The sample's contamination estimate is within the SOP limit.",
                yes="The estimate is at or below the limit for this assay.",
                no="The estimate is above the limit for this assay."),
            "indication_genes_complete": noul(
                "Every coding exon of the indication genes reached 20x in this run."),
        }

    def answer_key(self, f):
        """Rule-of-thumb answer key for the spread check only; never used as a label."""
        lim = f["limits"]
        short = f["mean"] < lim["mean"] or f["pct20"] < lim["pct20"]
        contam_over = f["contam"] > lim["contam"]
        dna_ok = f["dna_left_ng"] >= lim["dna"]
        topup_ok = f["library_left"] and f["dup"] <= lim["dup"]
        n = len(f["gaps"])
        if f["sex_check"] == "mismatch":
            out = "request_new_specimen"
        elif contam_over:
            out = "new_library_prep" if f["neighbours_contaminated"] and dna_ok else "request_new_specimen"
        elif short:
            out = "top_up_sequencing" if topup_ok else "new_library_prep" if dna_ok else "request_new_specimen"
        elif n > f["max_fill"]:
            out = "top_up_sequencing" if topup_ok else "new_library_prep" if dna_ok else "pass_with_gap_fill"
        elif n:
            out = "pass_with_gap_fill"
        else:
            out = "pass_release"
        if f["mean"] < 0.75 * lim["mean"] or f["pct20"] < lim["pct20"] - 3:
            cov = 0
        elif short:
            cov = 1
        elif n:
            cov = 2
        else:
            cov = 3
        if f["sex_check"] == "mismatch" or f["contam"] > 2 * lim["contam"]:
            ident_ = 3
        elif contam_over:
            ident_ = 2
        elif f["contam"] >= 0.5 * lim["contam"] or f["sex_check"] == "inconclusive":
            ident_ = 1
        else:
            ident_ = 0
        return {"sample_qc_outcome": out, "coverage_adequacy": cov, "identity_concern": ident_,
                "contamination_within_limit": "no" if contam_over else "yes",
                "indication_genes_complete": "no" if n else "yes"}


# =========================================================================== schema 3
TRIGGERS = {"sex": "sex inferred from the data disagrees with the request",
            "fingerprint": "sequencing fingerprint does not match the pre-extraction genotyping panel",
            "trio": "trio relationship check failed (Mendelian errors far above the limit)",
            "dup_genotype": "genotype is identical to another sample in the batch",
            "history": "genotype differs from the patient's earlier sample held by the lab"}


class SampleIdentityInvestigation(Schema):
    name = "sample_identity_investigation"
    description = ("A suspected sample mix-up flagged by an identity check, with the evidence gathered so far: "
                   "release, correct a documented swap, re-extract or recollect.")
    surfaces = {"json": 0.5, "investigation_log": 0.25, "qc_memo": 0.25}

    ORIGINS = ["plate_handling_error", "sample_sheet_index_error", "collection_labelling_error",
               "biological_explanation", "no_true_discrepancy"]

    def sample_factors(self, rng):
        reg = regime(rng)
        origin = pick_w(rng, {"plate_handling_error": 2.3, "sample_sheet_index_error": 2.0,
                              "collection_labelling_error": 2.0, "biological_explanation": 1.8,
                              "no_true_discrepancy": 2.0})
        bio = pick(rng, ["transplant", "twin"]) if origin == "biological_explanation" else None
        well = _well(rng)
        partner_sample = _sample_id(rng)
        test = pick_w(rng, {"routine": 3, "urgent": 2, "nicu": 1.2, "prenatal": 1})
        # evidence by origin
        own = "mismatch"
        cross = "not_tested"
        recip = "not_tested"
        partner_well = None
        sheet_edit = False
        history = pick(rng, ["none_on_file", "none_on_file", "match"])
        doc = None
        trigger = pick(rng, list(TRIGGERS))
        rest_checked = chance(rng, 0.3)
        same_session = False
        if origin == "plate_handling_error":
            partner_well = _adjacent(well, rng)
            cross = "match"
            recip = "match" if chance(rng, 0.55) else "not_tested"
            trigger = pick(rng, ["fingerprint", "sex", "trio"])
        elif origin == "sample_sheet_index_error":
            partner_well = _far_well(well, rng)
            cross = "match"
            recip = "match" if chance(rng, 0.6) else "not_tested"
            sheet_edit = True
            trigger = pick(rng, ["fingerprint", "sex", "trio"])
        elif origin == "collection_labelling_error":
            own = "match"
            history = "mismatch" if chance(rng, 0.7) else "none_on_file"
            trigger = "sex" if history == "none_on_file" else pick(rng, ["sex", "trio", "history", "history"])
            same_session = chance(rng, 0.45)
        elif origin == "biological_explanation":
            own = "match"
            if bio == "transplant":
                trigger = "sex"
                doc = pick(rng, ["requisition notes an allogeneic stem-cell transplant (donor of the other sex)",
                                 "clinical letter on file: bone-marrow transplant in 2019, donor of the other sex"])
            else:
                trigger = "dup_genotype"
                doc = "requisitions for both samples state they are monozygotic twins referred together"
            if reg == "conflicting" or (reg == "borderline" and chance(rng, 0.35)):
                doc = None if chance(rng, 0.6) else doc
        else:
            own = "match"
            trigger = pick(rng, ["sex", "fingerprint"])
        if trigger == "fingerprint" and origin == "no_true_discrepancy":
            first_pass = "too few panel SNPs covered on first pass (low depth at 9 of 24 sites)"
        elif trigger == "sex" and origin == "no_true_discrepancy":
            first_pass = "chromosome Y target reads near the call threshold on this capture"
        else:
            first_pass = None
        # conflicting: a pinned-looking swap that has not been cross-checked, or a reciprocal match that was
        # never actually run; borderline: partial evidence
        if reg == "conflicting" and origin in ("plate_handling_error", "sample_sheet_index_error"):
            recip = "not_tested"
        if reg == "clear" and origin in ("plate_handling_error", "sample_sheet_index_error"):
            recip = "match"
        if reg == "clear" and origin == "biological_explanation" and doc is None:
            doc = "requisition notes an allogeneic stem-cell transplant (donor of the other sex)"
        due = {"routine": rng.randint(10, 30), "urgent": rng.randint(4, 12), "nicu": rng.randint(2, 6),
               "prenatal": rng.randint(3, 9)}[test]
        recollect = {"routine": rng.randint(5, 14), "urgent": rng.randint(4, 10), "nicu": rng.randint(2, 5),
                     "prenatal": rng.randint(7, 21)}[test]
        multichannel = origin == "plate_handling_error" and chance(rng, 0.6)
        return {"regime": reg, "origin": origin, "bio": bio, "trigger": trigger, "own_concordance": own,
                "cross_match": cross, "reciprocal_match": recip, "sample_sheet_edited": sheet_edit,
                "history_genotype": history, "documentation": doc, "first_pass_issue": first_pass,
                "rest_of_plate_checked": rest_checked, "same_collection_session_flag": same_session,
                "multichannel_step": multichannel, "test": test, "due_days": due, "recollect_days": recollect,
                "sample": _sample_id(rng), "partner": partner_sample, "well": well, "partner_well": partner_well,
                "plate": f"P{rng.randint(100, 999)}", "investigator": person(rng), "director": person(rng),
                "assay": pick(rng, list(PANEL_ASSAYS) + EXOME_ASSAYS), "ticket": ident(rng, "NC", 5, "-"),
                "opened": rand_ts(rng), "sop_id": f"SQ-ID-{rng.randint(2, 15):02d}", "sop_variant": rng.randrange(3),
                "concordance_pct": {"match": round(rng.uniform(99.2, 100.0), 1),
                                    "mismatch": round(rng.uniform(41.0, 63.0), 1)}[own]}

    def _sop(self, f):
        v = f["sop_variant"]
        if v == 0:
            return ["Correct a swap only when each sample's sequencing fingerprint matches the OTHER sample's "
                    "pre-extraction genotype (a reciprocal cross-match); without it, re-extract both from the "
                    "primary tubes.",
                    "If the primary tube itself disagrees with the patient's earlier genotype or the request, reject "
                    "and recollect.",
                    "Release when re-analysis shows no real discrepancy, or when documented biology (transplant, "
                    "twin) explains it and the fingerprint matches the pre-extraction genotype."]
        if v == 1:
            return ["Swap correction (relabel or re-demultiplex) needs a two-way cross-match of fingerprints "
                    "against pre-extraction genotypes; a one-way match means re-extraction of both samples.",
                    "Tube-level error (primary tube inconsistent with earlier results or request): new specimen.",
                    "Discrepancy explained by documented transplant or twinning, or not reproduced on re-analysis: "
                    "release."]
        return ["Identity rules: (a) reciprocal cross-match proven -> correct the swap; (b) suspected lab mix-up "
                "not proven both ways -> re-extract both from primary tubes; (c) primary tube wrong -> recollect; "
                "(d) documented biological cause or no reproducible discrepancy -> release."]

    def _evidence(self, f):
        ev = [f"trigger: {TRIGGERS[f['trigger']]}"]
        if f["first_pass_issue"]:
            settled = ("inferred sex now agrees with the request" if f["trigger"] == "sex"
                       else "all 24 panel SNPs covered")
            ev.append(f"first pass: {f['first_pass_issue']}; repeat analysis at full depth: {settled}, fingerprint "
                      f"concordance {f['concordance_pct']}% with its own pre-extraction panel")
        else:
            ev.append(f"fingerprint vs own pre-extraction genotype: {f['concordance_pct']}% concordant")
        if f["cross_match"] == "match":
            ev.append(f"its sequencing fingerprint matches the pre-extraction genotype of {f['partner']} "
                      f"(well {f['partner_well']}; this sample in {f['well']}, plate {f['plate']})")
            ev.append("reverse check (" + f["partner"] + " sequencing vs this sample's pre-extraction genotype): "
                      + ("match" if f["reciprocal_match"] == "match" else "not yet run"))
        if f["sample_sheet_edited"]:
            ev.append("sample sheet was edited after the run was loaded; the index pairs of these two samples differ "
                      "from the plate map")
        if f["history_genotype"] == "mismatch" and f["trigger"] != "history":
            ev.append("genotype differs from the patient's earlier sample tested by the lab")
        elif f["history_genotype"] == "match":
            ev.append("genotype matches the patient's earlier sample tested by the lab")
        if f["documentation"]:
            ev.append(f["documentation"])
        elif f["bio"]:
            ev.append("clinician mentions a possible " + ("transplant" if f["bio"] == "transplant" else "twin")
                      + " by phone; nothing in the request or records")
        if f["same_collection_session_flag"]:
            ev.append("two other tubes from the same clinic session arrived with handwritten, unscanned labels")
        if f["multichannel_step"]:
            ev.append("the extraction plate was handled with a multichannel pipette by column")
        ev.append("other samples on the plate: " + ("fingerprints checked, all concordant" if f["rest_of_plate_checked"]
                                                   else "not yet checked"))
        return ev

    def render(self, f, rng, surface):
        ev = self._evidence(f)
        sop = self._sop(f)
        ctx = (f"{DUE_TEST[f['test']]} test; report due in {f['due_days']} days; a new specimen would take about "
               f"{f['recollect_days']} days to arrive")
        if surface == "json":
            return dumps(prune({
                "case": {"ticket": f["ticket"], "sample": f["sample"], "assay": f["assay"],
                         "opened": fmt_ts(f["opened"], style="iso"), "investigator": f["investigator"]},
                "evidence": ev, "clinical_context": ctx,
                "sop": {"id": f"{f['sop_id']} (local)", "rules": sop},
            }))
        if surface == "investigation_log":
            t = f["opened"]
            lines = [f"NON-CONFORMANCE {f['ticket']} -- sample identity ({f['sample']}, {f['assay']})"]
            for e in ev:
                t += timedelta(minutes=rng.randint(12, 240))
                lines.append(f"{fmt_ts(t, style='log')}  {f['investigator'].split()[0]}: {noisy(e, rng, 0.06)}")
            return "\n".join(lines) + f"\n\nContext: {ctx}.\n\n{f['sop_id']} (local):\n" + "\n".join(sop)
        return ("MEMO -- identity investigation " + f["ticket"] + "\n\nTo: " + f["director"] + "\nFrom: "
                + f["investigator"] + "\n\n" + noisy(join_sentences([f"Re {f['sample']} ({f['assay']}).",
                                                           ctx[0].upper() + ctx[1:] + "."]),
                                                     rng, 0.1)
                + "\n\nFindings so far:\n" + bullets(ev, rng) + f"\n\nPer {f['sop_id']} (local): " + " ".join(sop))

    def questions(self):
        return {
            "identity_resolution": choice("How should the lab resolve this identity flag?", {
                "correct_documented_swap": "Correct the swap between the two named samples and release both.",
                "re_extract_from_primary": "Re-extract the samples involved from their primary tubes and test again.",
                "reject_and_recollect": "Reject the specimen and ask for a fresh collection from the patient.",
                "release_identity_confirmed": "Release the result; the sample's identity is confirmed.",
            }),
            "likely_origin": choice("Where did the discrepancy most likely arise?", {
                "biological_explanation": "In the patient's own biology, such as a transplant or a twin.",
                "collection_labelling_error": "At collection, before the tube reached the lab.",
                "no_true_discrepancy": "Nowhere; the first check was misleading and identity holds.",
                "plate_handling_error": "During extraction or library prep, between neighbouring wells.",
                "sample_sheet_index_error": "In the sample sheet, where index pairs were assigned wrongly.",
            }),
            "mixup_likelihood": score("How likely is it that this result belongs to a different person?", [
                "Remote: the flag has an innocent explanation.",
                "Possible: a mix-up cannot be excluded yet.",
                "Probable: most of the evidence points to a mix-up.",
                "Near certain: the evidence pins the mix-up down.",
            ]),
            "result_delay_impact": score("How much harm would a delay to this result cause?", [
                "Little: a routine report with plenty of time before it is due.",
                "Some: the report would be late, but care would not change.",
                "Considerable: a treatment decision is waiting on it.",
                "Severe: a time-limited decision (pregnancy, newborn care) depends on it and a new specimen is "
                "slow or hard to get.",
            ]),
            "other_samples_at_risk": noul(
                "Other samples handled alongside this one may also be affected.",
                yes="The likely cause could have touched other samples, and they have not all been checked.",
                no="The cause is confined to this case, or the other samples have been checked and are fine."),
        }

    def answer_key(self, f):
        """Rule-of-thumb answer key for the spread check only; never used as a label."""
        o = f["origin"]
        if o in ("plate_handling_error", "sample_sheet_index_error"):
            res = "correct_documented_swap" if f["reciprocal_match"] == "match" else "re_extract_from_primary"
            ml = 3 if f["reciprocal_match"] == "match" else 2
        elif o == "collection_labelling_error":
            res = "reject_and_recollect"
            ml = 3 if f["history_genotype"] == "mismatch" else 2
        elif o == "biological_explanation":
            # undocumented biology leaves a tube that disagrees with the request: the SOP's recollect branch
            res = "release_identity_confirmed" if f["documentation"] else "reject_and_recollect"
            ml = 0 if f["documentation"] else 1
        else:
            res, ml = "release_identity_confirmed", 0
        slack = f["due_days"] - f["recollect_days"]
        if f["test"] in ("prenatal", "nicu"):
            imp = 3
        elif f["test"] == "urgent":
            imp = 2
        else:
            imp = 0 if slack >= 7 else 1
        if o in ("plate_handling_error", "sample_sheet_index_error"):
            risk = not f["rest_of_plate_checked"]
        elif o == "collection_labelling_error":
            risk = f["same_collection_session_flag"]
        else:
            risk = False
        return {"identity_resolution": res, "likely_origin": o, "mixup_likelihood": ml, "result_delay_impact": imp,
                "other_samples_at_risk": "yes" if risk else "no"}


# =========================================================================== schema 4
MANUAL_ROUTES = ["by hand with a multichannel pipette", "by hand, single-channel pipetting"]
KITS = ["enzymatic fragmentation kit", "hybrid-capture prep kit", "mechanical-shearing prep kit",
        "low-input hybrid-capture kit", "UMI capture prep kit"]


def _plate_wells(n):
    """The first n wells of a 96-well plate in column order (A1..H1, A2..)."""
    return [f"{'ABCDEFGH'[i % 8]}{i // 8 + 1}" for i in range(n)]


class LibraryBatchQc(Schema):
    name = "library_batch_qc"
    description = ("A library-prep batch's per-library QC and its controls, checked against the lab's pooling "
                   "rules before the libraries go on to sequencing.")
    surfaces = {"json": 0.5, "batch_qc_sheet": 0.25, "prep_handover_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        n = pick(rng, [16, 24, 32, 48, 48, 96])
        lim = {"conc": pick(rng, [1.5, 2.0, 3.0]), "size_lo": pick(rng, [250, 280, 300]),
               "size_hi": pick(rng, [550, 600, 650]), "dimer": pick(rng, [2.0, 3.0, 5.0]),
               "ntc": pick(rng, [5000, 10000, 20000]), "recovery": pick(rng, [95.0, 97.0, 98.0]),
               "drop_pct": pick(rng, [10, 15, 15, 20])}
        if reg == "clear":
            scen = pick_w(rng, {"clean": 2.5, "scattered_few": 2, "ntc_high": 3.6, "column": 2.8, "pos_fail": 1,
                                "row": 1.2})
        elif reg == "borderline":
            scen = pick_w(rng, {"clean": 1.8, "scattered_few": 2.0, "scattered_edge": 2.2, "ntc_near": 3.5,
                                "pos_near": 1.5, "row": 3})
        else:
            scen = pick_w(rng, {"column_looks_random": 2.4, "lot_wide": 2.2, "ntc_high_good_libs": 4.5,
                                "pos_fail_clean_libs": 1.4, "scattered_many": 1.4})
        max_drop = int(n * lim["drop_pct"] / 100)
        fails, pattern, ntc, recovery, dimer_batch = 0, "none", int(lim["ntc"] * rng.uniform(0.02, 0.35)), \
            round(rng.uniform(lim["recovery"] + 0.6, 100.0), 1), round(lim["dimer"] * rng.uniform(0.1, 0.5), 1)
        if scen == "scattered_few":
            fails = rng.randint(1, max(1, max_drop - 1)) if max_drop > 1 else 1
            pattern = "scattered"
        elif scen == "scattered_edge":
            fails = max_drop + pick(rng, [0, 0, 1, 1])
            pattern = "scattered"
        elif scen == "scattered_many":
            fails = rng.randint(max_drop + 2, max_drop + 2 + max(2, n // 6))
            pattern = "scattered"
        elif scen == "row":
            fails = pick(rng, [3, 4, 5])
            pattern = "row"
        elif scen in ("column", "column_looks_random"):
            fails = min(8, n // 4) if n >= 32 else max(2, n // 8)
            pattern = "column" if scen == "column" else "column_unflagged"
        elif scen == "lot_wide":
            fails = rng.randint(max(2, n // 5), max(3, n // 2))
            pattern = "lot"
            dimer_batch = round(lim["dimer"] * rng.uniform(1.3, 3.0), 1)
        elif scen in ("ntc_high", "ntc_high_good_libs"):
            ntc = int(lim["ntc"] * rng.uniform(1.4, 8))
            if scen == "ntc_high":
                fails = pick(rng, [0, 1, 2])
                pattern = "scattered" if fails else "none"
        elif scen == "ntc_near":
            ntc = int(near(rng, lim["ntc"] * 1.03, lim["ntc"] * 0.12, lo=100))
            fails = pick(rng, [0, 1, 2])
            pattern = "scattered" if fails else "none"
        elif scen in ("pos_fail", "pos_fail_clean_libs"):
            recovery = round(lim["recovery"] - rng.uniform(1.5, 12), 1)
        elif scen == "pos_near":
            recovery = round(near(rng, lim["recovery"], 1.2, hi=100.0), 1)
        if scen in ("scattered_few", "scattered_edge", "pos_near", "pos_fail", "row") and \
                chance(rng, {"clear": 0.08, "borderline": 0.22, "conflicting": 0.25}[reg]):
            ntc = int(lim["ntc"] * rng.uniform(1.05, 3.0))       # the blank is over as well
        fails = min(fails, n - 1)
        lot_status = pick(rng, ["in date", "in date", "expires in 9 days", "first batch on this lot"])
        if scen == "lot_wide":
            lot_status = pick(rng, ["first batch on this lot", "expired 3 days ago (used in error)"])
        # positions of the failed libraries on the part of the plate in use
        layout = _plate_wells(n)
        used_cols = n // 8
        if pattern == "row" and used_cols < 3:
            pattern = "column"
        if pattern in ("column", "column_unflagged"):
            col = rng.randint(1, used_cols)
            fail_wells = [f"{r}{col}" for r in "ABCDEFGH"][:max(2, min(8, fails))]
        elif pattern == "row":
            row = pick(rng, "ABCDEFGH")
            fail_wells = [f"{row}{c}" for c in sorted(rng.sample(range(1, used_cols + 1), min(fails, used_cols)))]
        else:
            fail_wells = sorted(rng.sample(layout, fails), key=layout.index)
        fails = len(fail_wells)
        reasons = [pick(rng, ["low yield", "low yield", "fragment size too large", "fragment size too small",
                              "adapter dimer high"]) for _ in fail_wells]
        if pattern == "lot":
            reasons = [pick(rng, ["low yield", "adapter dimer high"]) for _ in fail_wells]
        note = None
        if scen == "column_looks_random":
            note = "A handful of low yields, probably just the low-input samples."
        elif scen == "ntc_high_good_libs":
            note = "Libraries all look great, NTC is a bit high but it's only the blank."
        elif scen == "pos_fail_clean_libs":
            note = "Every patient library passed; the reference sample just came out a bit low."
        elif scen == "scattered_many":
            note = "Several failures, but scattered across the plate; no pattern I can see."
        elif scen == "lot_wide":
            note = "Yields lower across the board today."
        else:
            bad_rows = {w[0] for w in fail_wells}
            free_rows = [r for r in "ABCDEFGH" if r not in bad_rows] or ["A"]
            note = pick(rng, [None, "Prep ran without incident.", "Two samples were low input at receipt.",
                              "Normalisation done on the handler.",
                              f"Bead clean-up repeated once for row {pick(rng, free_rows)}."])
        return {"regime": reg, "scenario": scen, "n_libraries": n, "limits": lim, "failed": fails,
                "fail_wells": fail_wells, "fail_reasons": reasons, "pattern": pattern, "ntc_reads": ntc,
                "pos_recovery": recovery, "pos_expected": pick(rng, [48, 64, 96, 120]),
                "dimer_batch_median": dimer_batch, "lot_status": lot_status, "note": note,
                "route": "on liquid handler LH-" + str(rng.randint(1, 9)) if chance(rng, 0.7) else
                pick(rng, MANUAL_ROUTES), "kit": pick(rng, KITS), "lot": f"KL-{rng.randint(1000, 9999)}{hexid(rng, 2).upper()}",
                "batch": ident(rng, "PB", 5, "-"), "operator": person(rng), "lead": person(rng),
                "assay": pick(rng, list(PANEL_ASSAYS) + EXOME_ASSAYS), "date": rand_date(rng),
                "sop_id": f"SQ-LIB-{rng.randint(2, 22):02d}", "sop_variant": rng.randrange(2),
                "median_conc": round(rng.uniform(lim["conc"] * 2, lim["conc"] * 9), 1),
                "median_size": rng.randint(lim["size_lo"] + 40, lim["size_hi"] - 60)}

    def _sop(self, f):
        lim = f["limits"]
        if f["sop_variant"] == 0:
            return [f"Each library: concentration >= {lim['conc']:g} nM, fragment peak {lim['size_lo']}-"
                    f"{lim['size_hi']} bp, adapter dimer <= {lim['dimer']:g}%.",
                    f"Controls: no-template control <= {lim['ntc']:,} reads; positive reference sample recovers >= "
                    f"{lim['recovery']:g}% of its known variants.",
                    f"Pool all when every library and both controls pass. Up to {lim['drop_pct']}% of libraries may be "
                    "dropped for repeat if the failures are scattered and both controls pass. Repeat the whole "
                    f"batch, with fresh reagent aliquots, if more than {lim['drop_pct']}% fail, the positive control "
                    "fails, or the no-template control is over its limit.",
                    "Quarantine and investigate, before any repeat, if failures follow a pattern (a column or row, "
                    "one reagent lot, one handler channel)."]
        return [f"Library acceptance: >= {lim['conc']:g} nM; peak between {lim['size_lo']} and {lim['size_hi']} bp; "
                f"adapter dimer no more than {lim['dimer']:g}%.",
                f"Blank (NTC) must stay at or under {lim['ntc']:,} reads; the reference sample must recover "
                f"{lim['recovery']:g}% or more of expected variants.",
                "Failures clustered by column, row, lot or channel -> stop, quarantine, root cause first.",
                f"Otherwise: clean batch -> pool everything; scattered failures up to {lim['drop_pct']}% of the "
                f"batch with good controls -> pool the rest and repeat the failures; more than that, a failed "
                "reference sample or an over-limit blank -> repeat the batch with fresh aliquots."]

    def render(self, f, rng, surface):
        lim = f["limits"]
        sop = self._sop(f)
        fails = [f"{w} ({r})" for w, r in zip(f["fail_wells"], f["fail_reasons"])]
        ctrl = (f"NTC {f['ntc_reads']:,} reads; reference sample {f['pos_recovery']:g}% of "
                f"{f['pos_expected']} known variants recovered")
        if surface == "json":
            return dumps(prune({
                "batch": {"id": f["batch"], "assay": f["assay"], "libraries": f["n_libraries"], "kit": f["kit"],
                          "kit_lot": f"{f['lot']} ({f['lot_status']})", "prepared": f["route"],
                          "operator": f["operator"], "date": f["date"].isoformat()},
                "library_qc": {"failed": f["failed"], "failed_wells": fails or None,
                               "median_concentration_nm": f["median_conc"], "median_peak_bp": f["median_size"],
                               "median_adapter_dimer_pct": f["dimer_batch_median"]},
                "controls": {"ntc_reads": f["ntc_reads"], "reference_recovery_pct": f["pos_recovery"],
                             "reference_known_variants": f["pos_expected"]},
                "operator_note": f["note"],
                "sop": {"id": f"{f['sop_id']} (local)", "rules": sop},
            }))
        if surface == "batch_qc_sheet":
            rows = [["libraries", f["n_libraries"]], ["failed", f"{f['failed']}" + (": " + ", ".join(fails) if fails else "")],
                    ["median conc (nM)", f["median_conc"]], ["median peak (bp)", f["median_size"]],
                    ["median adapter dimer (%)", f["dimer_batch_median"]], ["NTC reads", f"{f['ntc_reads']:,}"],
                    ["reference recovery", f"{f['pos_recovery']:g}% of {f['pos_expected']}"],
                    ["kit lot", f"{f['lot']} ({f['lot_status']})"], ["prepared", f["route"]]]
            return (f"LIBRARY BATCH QC {f['batch']} -- {f['assay']}, {f['kit']}, {fmt_date(f['date'], rng)}\n"
                    + table(["item", "value"], rows, rng) + (f"\n\nOperator: {f['note']}" if f["note"] else "")
                    + f"\n\n{f['sop_id']} (local):\n" + "\n".join(sop))
        body = [f"Handover -- prep batch {f['batch']} ({f['assay']})", f"From {f['operator']} to {f['lead']}", "",
                noisy(join_sentences([f"{f['n_libraries']} libraries prepared {f['route']} with the {f['kit']} "
                                      f"(lot {f['lot']}, {f['lot_status']}).",
                                      (f"{f['failed']} failed QC: {', '.join(fails)}." if fails else
                                       "No library failed QC."),
                                      f"Controls: {ctrl}.",
                                      f"Median adapter dimer {f['dimer_batch_median']:g}%.", f["note"] or ""]),
                      rng, 0.1), "", f"{f['sop_id']} (local): " + " ".join(sop)]
        return "\n".join(body)

    def questions(self):
        return {
            "batch_decision": choice("What should happen to this library batch?", {
                "pool_all": "Pool every library for sequencing.",
                "pool_excluding_failed": "Pool the passing libraries and repeat only the failed ones.",
                "quarantine_and_investigate": "Stop the batch and find the cause before anything is pooled or "
                                              "repeated.",
                "repeat_whole_batch": "Discard the batch and prepare every library again.",
            }),
            "control_performance": score("How well did the batch's controls perform?", [
                "Failed: a control is well outside its limit.",
                "Weak: a control just misses its limit.",
                "Acceptable: both controls pass, one of them close to its limit.",
                "Clean: both controls pass comfortably.",
            ]),
            "failure_burden": score("How heavy is the library failure burden in this batch?", [
                "None: every library passed.",
                "Light: a few percent of the batch.",
                "Moderate: up to the share the SOP lets the lab drop.",
                "Heavy: more than the SOP lets the lab drop.",
            ]),
            "ntc_contamination_significant": noul(
                "The no-template control shows contamination that matters under the SOP.",
                yes="The control's read count is above the SOP limit.",
                no="The control's read count is at or below the SOP limit."),
            "systematic_cause_likely": noul(
                "The batch's problems point to a systematic cause rather than chance."),
        }

    def answer_key(self, f):
        """Rule-of-thumb answer key for the spread check only; never used as a label."""
        lim = f["limits"]
        ntc_over = f["ntc_reads"] > lim["ntc"]
        pos_fail = f["pos_recovery"] < lim["recovery"]
        patterned = f["pattern"] in ("column", "column_unflagged", "lot", "row")
        share = f["failed"] / f["n_libraries"]
        if patterned:
            dec = "quarantine_and_investigate"
        elif ntc_over or pos_fail or share * 100 > lim["drop_pct"]:
            dec = "repeat_whole_batch"
        elif f["failed"]:
            dec = "pool_excluding_failed"
        else:
            dec = "pool_all"
        ntc_r = f["ntc_reads"] / lim["ntc"]
        pos_gap = lim["recovery"] - f["pos_recovery"]
        if ntc_r > 1.5 or pos_gap > 2:
            ctrl = 0
        elif ntc_over or pos_fail:
            ctrl = 1
        elif ntc_r > 0.8 or pos_gap > -1.0:
            ctrl = 2
        else:
            ctrl = 3
        burden = 0 if f["failed"] == 0 else 1 if share <= 0.05 else 2 if share * 100 <= lim["drop_pct"] else 3
        systematic = patterned or f["dimer_batch_median"] > lim["dimer"]
        return {"batch_decision": dec, "control_performance": ctrl, "failure_burden": burden,
                "ntc_contamination_significant": "yes" if ntc_over else "no",
                "systematic_cause_likely": "yes" if systematic else "no"}


# =========================================================================== schema 5
TUMOURS = {
    "lung adenocarcinoma": ["EGFR exons 18-21", "KRAS codon 12", "BRAF V600", "MET exon 14 splice sites", "ERBB2 exon 20"],
    "colorectal adenocarcinoma": ["KRAS codons 12/13/61", "NRAS codons 12/13/61", "BRAF V600", "PIK3CA exon 20"],
    "melanoma": ["BRAF V600", "NRAS codon 61", "KIT exons 11/13"],
    "gastrointestinal stromal tumour": ["KIT exons 9/11/13/17", "PDGFRA exons 12/18"],
    "glioma": ["IDH1 R132", "IDH2 R172", "TERT promoter", "H3F3A K27"],
    "cholangiocarcinoma": ["IDH1 R132", "FGFR2 hotspots", "BRAF V600"],
    "thyroid carcinoma": ["BRAF V600", "NRAS codon 61", "TERT promoter", "HRAS codon 61"],
    "endometrial carcinoma": ["POLE exonuclease domain", "PIK3CA exons 9/20", "PTEN hotspots"],
    "breast carcinoma": ["PIK3CA exons 9/20", "ESR1 ligand-binding domain", "AKT1 E17"],
    "urothelial carcinoma": ["FGFR3 hotspots", "PIK3CA exons 9/20", "TERT promoter"],
    "ovarian carcinoma": ["BRCA1 coding exons", "BRCA2 coding exons", "TP53 exons 4-10"],
    "pancreatic adenocarcinoma": ["KRAS codons 12/13/61", "BRAF V600", "TP53 exons 4-10"],
    "prostate adenocarcinoma": ["BRCA2 coding exons", "ATM coding exons", "AR ligand-binding domain"],
    "soft-tissue sarcoma": ["KIT exons 9/11", "CTNNB1 exon 3", "IDH1 R132"],
}
SPECIMENS = ["resection block", "core biopsy block", "lymph-node excision block"]
SITE_SPECIMENS = {"lung adenocarcinoma": ["cytology cell block (EBUS)", "bronchial biopsy block"],
                  "pancreatic adenocarcinoma": ["cytology cell block (EUS-FNA)"],
                  "thyroid carcinoma": ["cytology cell block (FNA)"],
                  "colorectal adenocarcinoma": ["endoscopic biopsy block"],
                  "gastrointestinal stromal tumour": ["endoscopic biopsy block"],
                  "glioma": ["stereotactic biopsy block"], "endometrial carcinoma": ["endometrial biopsy block"],
                  "urothelial carcinoma": ["TURBT chips block"]}
GENERAL_ASSAYS = ["Solid tumour hotspot panel v5", "Pan-solid 52-gene panel"]
SITE_ASSAYS = {"lung adenocarcinoma": ["Lung cancer DNA panel v2"],
               "colorectal adenocarcinoma": ["GI tumour panel v3"], "gastrointestinal stromal tumour": ["GI tumour panel v3"],
               "cholangiocarcinoma": ["GI tumour panel v3"], "pancreatic adenocarcinoma": ["GI tumour panel v3"],
               "glioma": ["Neuro-oncology panel v2"], "endometrial carcinoma": ["Gynae-oncology panel v1"],
               "ovarian carcinoma": ["Gynae-oncology panel v1"]}


class TumourSampleQc(Schema):
    name = "tumour_sample_qc"
    description = ("A tumour sample for a somatic panel whose purity, DNA integrity or depth may limit what can be "
                   "reported, routed under the lab's tumour-sample SOP.")
    surfaces = {"json": 0.5, "molecular_worksheet": 0.25, "signout_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        tumour = pick(rng, list(TUMOURS))
        hotspots = rng.sample(TUMOURS[tumour], min(len(TUMOURS[tumour]), rng.randint(2, 3)))
        lim = {"content": pick(rng, [20, 20, 25, 30]), "floor": pick(rng, [10, 10, 15]),
               "depth": pick(rng, [500, 500, 600]), "caveat_depth": pick(rng, [250, 300]),
               "hotspot": pick(rng, [250, 300]), "din": pick(rng, [2.5, 3.0]), "deam": pick(rng, [2.0, 2.5]),
               "gap": pick(rng, [10, 15])}
        if lim["floor"] >= lim["content"]:
            lim["floor"] = lim["content"] - 10
        scen = {"clear": pick_w(rng, {"good": 2.5, "content_low_macro": 1.3, "din_fail": 1.6, "nothing_left": 1.3}),
                "borderline": pick_w(rng, {"content_edge": 3, "depth_edge": 2.5, "good": 1.2, "purity_gap": 2}),
                "conflicting": pick_w(rng, {"purity_gap": 2.5, "content_low_macro": 1.5, "deam_high_good": 2,
                                            "din_fail": 1, "depth_edge": 1.2})}[reg]
        path = rng.choice([20, 25, 30, 30, 35, 40, 50, 60, 70])
        comp = max(5, path + rng.randint(-14, 8))
        din = round(rng.uniform(lim["din"] + 0.6, 7.5), 1)
        depth = int(lim["depth"] * rng.uniform(1.0, 2.2))
        hot_min = int(min(depth * rng.uniform(0.5, 0.95), depth))
        deam = round(rng.uniform(0.8, lim["deam"] * 0.85), 1)
        if chance(rng, 0.25):
            deam = round(rng.uniform(lim["deam"] * 1.05, lim["deam"] * 2.6), 1)
        macro = None
        other_blocks = pick(rng, [0, 0, 1, 1, 2, 3])
        if scen == "content_low_macro":
            path = pick(rng, [5, 8, 10, 12, 15])
            comp = max(3, path + rng.randint(-4, 4))
            macro = rng.choice([35, 40, 50, 60])
            if reg == "conflicting":
                other_blocks = 0
        elif scen == "content_edge":
            path = lim["content"] + pick(rng, [-5, -3, -2, 0, 0, 2, 5])
            comp = max(3, path + rng.randint(-6, 6))
            macro = pick(rng, [None, None, path + 20])
        elif scen == "purity_gap":
            path = pick(rng, [30, 40, 45, 50, 60])
            gap = lim["gap"] + pick(rng, [-4, -1, 2, 6, 12])
            comp = max(4, path - gap)
            macro = pick(rng, [None, None, comp + 25])
        elif scen == "depth_edge":
            depth = int(near(rng, lim["depth"], lim["depth"] * 0.25, lo=150))
            hot_min = int(depth * rng.uniform(0.5, 0.9))
        elif scen == "din_fail":
            din = round(rng.uniform(1.1, lim["din"] - 0.2), 1)
            depth = int(lim["depth"] * rng.uniform(0.4, 0.95))
            hot_min = int(depth * rng.uniform(0.4, 0.9))
        elif scen == "nothing_left":
            path, comp, other_blocks, macro = pick(rng, [3, 5, 6]), pick(rng, [3, 4, 6]), 0, None
        elif scen == "deam_high_good":
            deam = round(rng.uniform(lim["deam"] * 1.1, lim["deam"] * 3), 1)
        if reg == "borderline" and scen != "depth_edge" and chance(rng, 0.3):
            hot_min = int(near(rng, lim["hotspot"], lim["hotspot"] * 0.15, lo=80))
        if scen != "deam_high_good" and din < 3.5 and chance(rng, 0.5):
            deam = round(rng.uniform(lim["deam"] * 0.9, lim["deam"] * 2.2), 1)
        matched_normal = chance(rng, 0.35)
        block_age = rng.randint(0, 9)
        return {"regime": reg, "scenario": scen, "tumour": tumour, "hotspots": hotspots, "limits": lim,
                "path_pct": path, "comp_pct": comp, "din": din, "unique_depth": depth, "hotspot_min_depth": hot_min,
                "deamination_index": deam, "macro_area_pct": macro, "other_blocks": other_blocks,
                "matched_normal": matched_normal, "block_age_years": block_age,
                "specimen": pick(rng, SPECIMENS + SITE_SPECIMENS.get(tumour, []) * 2),
                "case": ident(rng, "MP", 6, "-"), "block": f"{pick(rng, ['A', 'B', 'C', 'D'])}{rng.randint(1, 9)}",
                "scientist": person(rng), "pathologist": "Dr " + person(rng), "date": rand_date(rng),
                "assay": pick(rng, GENERAL_ASSAYS + SITE_ASSAYS.get(tumour, []) * 2),
                "sop_id": f"SQ-SOM-{rng.randint(2, 18):02d}", "sop_variant": rng.randrange(2),
                "integrity_metric": pick(rng, ["DIN", "DIN", "DV200"])}

    def _effective(self, f):
        """The tumour content the SOP says to use."""
        p, c = f["path_pct"], f["comp_pct"]
        return c if p - c > f["limits"]["gap"] else p

    def _integrity(self, f):
        if f["integrity_metric"] == "DV200":
            return f"DV200 {int(f['din'] * 10 + 5)}%", f"DV200 {int(f['limits']['din'] * 10 + 5)}%"
        return f"DIN {f['din']:.1f}", f"DIN {f['limits']['din']:g}"

    def _sop(self, f):
        lim = f["limits"]
        _, imin = self._integrity(f)
        if f["sop_variant"] == 0:
            return [f"Tumour content: use the pathologist's estimate unless the computational purity is more than "
                    f"{lim['gap']} points lower, then use the computational figure. Minimum {lim['content']}%; "
                    f"reports with a sensitivity caveat are allowed down to {lim['floor']}%.",
                    f"DNA integrity at least {imin}. Unique depth at least {lim['depth']}x median and {lim['hotspot']}x "
                    f"at each referral hotspot for full reporting; down to {lim['caveat_depth']}x with a caveat.",
                    "Below the minimum content, macrodissect and repeat when the pathologist has marked a richer "
                    "area; otherwise caveat if above the floor, or ask for another block, or cancel if none is left.",
                    f"Integrity failure: another block if one exists, else cancel. Deamination index above "
                    f"{lim['deam']:g} flags likely FFPE artefact among low-level C>T calls."]
        return [f"Content rule: pathologist estimate stands unless computed purity is lower by more than "
                f"{lim['gap']} points. Full report from {lim['content']}%; caveated report from {lim['floor']}%.",
                f"Need {imin} or better; median unique coverage {lim['depth']}x and per-hotspot {lim['hotspot']}x for "
                f"a full report, {lim['caveat_depth']}x for a caveated one.",
                "Content under the minimum: marked enrichable area -> macrodissect and rerun; no such area -> caveat "
                "(above the floor) or new block; nothing more available -> cancel. Integrity below the limit -> new "
                "block or cancel.",
                f"C>T deamination index over {lim['deam']:g} = probable fixation artefact at low allele fractions."]

    def render(self, f, rng, surface):
        sop = self._sop(f)
        integ, _ = self._integrity(f)
        macro = (f"pathologist has marked an area of about {f['macro_area_pct']}% tumour for macrodissection"
                 if f["macro_area_pct"] else "no richer area marked")
        blocks = (f"{f['other_blocks']} other block{'s' if f['other_blocks'] != 1 else ''} on file"
                  if f["other_blocks"] else "no other block available")
        if surface == "json":
            return dumps(prune({
                "specimen": {"case": f["case"], "tumour": f["tumour"], "type": f["specimen"], "block": f["block"],
                             "block_age_years": f["block_age_years"], "assay": f["assay"],
                             "referral_hotspots": f["hotspots"]},
                "pathology": {"tumour_content_pct": f["path_pct"], "macrodissection": macro, "other_material": blocks},
                "sequencing": {"computed_purity_pct": f["comp_pct"], "dna_integrity": integ,
                               "median_unique_depth": f["unique_depth"],
                               "lowest_hotspot_depth": f["hotspot_min_depth"],
                               "deamination_index": f["deamination_index"],
                               "matched_normal": "sequenced" if f["matched_normal"] else "not available"},
                "sop": {"id": f"{f['sop_id']} (local)", "rules": sop},
            }))
        if surface == "molecular_worksheet":
            age = f"{f['block_age_years']} y old" if f["block_age_years"] else "under 1 y old"
            rows = [("Case", f["case"]), ("Tumour", f["tumour"]), ("Specimen", f"{f['specimen']} {f['block']} ({age})"),
                    ("Assay", f["assay"]), ("Referral hotspots", ", ".join(f["hotspots"])),
                    ("Pathologist content", f"{f['path_pct']}% ({f['pathologist']})"),
                    ("Computed purity", f"{f['comp_pct']}%"), ("DNA integrity", integ),
                    ("Median unique depth", f"{f['unique_depth']}x"),
                    ("Lowest hotspot depth", f"{f['hotspot_min_depth']}x"),
                    ("Deamination index", f["deamination_index"]),
                    ("Matched normal", "yes" if f["matched_normal"] else "no"),
                    ("Macrodissection", macro), ("Further material", blocks)]
            return ("MOLECULAR PATHOLOGY WORKSHEET\n" + kv_lines(rows, rng) + f"\n\n{f['sop_id']} (local):\n"
                    + "\n".join(f"{i}. {x}" for i, x in enumerate(sop, 1)))
        para = [f"{f['case']}: {f['tumour']}, {f['specimen']} {f['block']}, for {f['assay']} "
                f"(hotspots: {', '.join(f['hotspots'])}).",
                f"{f['pathologist']} estimates {f['path_pct']}% tumour; the pipeline puts purity at {f['comp_pct']}%.",
                f"{integ}; median unique depth {f['unique_depth']}x, lowest referral hotspot {f['hotspot_min_depth']}x.",
                f"Deamination index {f['deamination_index']}; matched normal "
                + ("sequenced." if f["matched_normal"] else "not available."),
                macro[0].upper() + macro[1:] + "; " + blocks + "."]
        return (f"SIGN-OUT NOTE ({f['scientist']}, {fmt_date(f['date'], rng)})\n\n" + noisy(join_sentences(para), rng, 0.08)
                + f"\n\n{f['sop_id']} (local): " + " ".join(sop))

    def questions(self):
        return {
            "tumour_sample_route": choice("How should this tumour sample be taken forward?", {
                "cancel_insufficient_material": "Cancel the test; no usable material is left.",
                "macrodissect_and_repeat": "Macrodissect the marked tumour-rich area and repeat the test.",
                "report_full": "Report the results in full.",
                "report_with_sensitivity_caveat": "Report the results with a statement that sensitivity is limited.",
                "request_new_block": "Ask pathology for another block and test that instead.",
            }),
            "detection_sensitivity": score("How well could this run detect a heterozygous variant carried by half "
                                           "of the tumour cells in the referral hotspots?", [
                "Poorly: such a variant would likely fall below what the run can detect.",
                "Marginally: detection is close to a coin toss at the expected allele fraction.",
                "Reasonably: detection is likely, with a small margin.",
                "Reliably: the expected allele fraction sits well above the detection limit.",
            ]),
            "purity_estimate_agreement": score("How well do the pathologist's and the computed tumour content agree?", [
                "Contradictory: more than 20 points apart.",
                "Divergent: 13 to 20 points apart.",
                "Roughly consistent: 6 to 12 points apart.",
                "Close: within 5 points.",
            ]),
            "meets_tumour_content_minimum": noul(
                "The tumour content the SOP says to use meets the minimum for a full report.",
                yes="The figure chosen under the SOP's rule is at or above the full-report minimum.",
                no="The figure chosen under the SOP's rule is below the full-report minimum."),
            "ffpe_artefact_concern": noul(
                "Fixation artefact is a real concern for low-level C>T calls in this sample."),
        }

    def answer_key(self, f):
        """Rule-of-thumb answer key for the spread check only; never used as a label."""
        lim = f["limits"]
        eff = self._effective(f)
        integ_ok = f["din"] >= lim["din"]
        full_depth = f["unique_depth"] >= lim["depth"] and f["hotspot_min_depth"] >= lim["hotspot"]
        cav_depth = f["unique_depth"] >= lim["caveat_depth"] and f["hotspot_min_depth"] >= lim["caveat_depth"] * 0.8
        if not integ_ok:
            route = "request_new_block" if f["other_blocks"] else "cancel_insufficient_material"
        elif eff < lim["content"]:
            if f["macro_area_pct"]:
                route = "macrodissect_and_repeat"
            elif eff >= lim["floor"] and cav_depth:
                route = "report_with_sensitivity_caveat"
            elif f["other_blocks"]:
                route = "request_new_block"
            else:
                route = "cancel_insufficient_material"
        elif full_depth:
            route = "report_full"
        elif cav_depth:
            route = "report_with_sensitivity_caveat"
        else:
            route = "request_new_block" if f["other_blocks"] else "report_with_sensitivity_caveat"
        depth = min(f["unique_depth"], f["hotspot_min_depth"] * 1.5)
        lod = 2500.0 / max(depth, 1)          # % allele fraction detectable at this depth (lab rule of thumb)
        ratio = (eff / 4) / lod
        sens = 0 if ratio < 0.8 else 1 if ratio < 1.5 else 2 if ratio < 2.5 else 3
        d = abs(f["path_pct"] - f["comp_pct"])
        agree = 3 if d <= 5 else 2 if d <= 12 else 1 if d <= 20 else 0
        return {"tumour_sample_route": route, "detection_sensitivity": sens, "purity_estimate_agreement": agree,
                "meets_tumour_content_minimum": "yes" if eff >= lim["content"] else "no",
                "ffpe_artefact_concern": "yes" if f["deamination_index"] > lim["deam"] else "no"}


WORKFLOW = Workflow(
    name="sequencing_run_qc",
    title="Sequencing run and sample QC",
    description="Whether a clinical sequencing lab releases, repeats or holds a run, a sample or a batch under its "
                "written QC SOP.",
    domain="life sciences / genomics laboratory operations",
    schemas=[RunReleaseReview(), SampleCoverageQc(), SampleIdentityInvestigation(), LibraryBatchQc(),
             TumourSampleQc()],
    notes="Invented instruments, lots, runs, samples and limits; gene symbols only as labels. The QC limits are the "
          "fictional lab's own SOP, not vendor specifications or guidance. No patient-facing conversation, no "
          "security alerts, no bills, no AI-agent runs.",
)
