"""Variant curation review: germline variant and copy-number classification decisions at a (fictional)
diagnostic genomics laboratory, made against the laboratory's own written curation SOP.

Five schemas: classifying a sequence variant from its raw evidence and a curator's draft, re-reviewing a
reported variant of uncertain significance when new evidence arrives, deciding whether a secondary finding from
an exome is reported, classifying a copy-number variant on a points sheet, and resolving a classification that
disagrees with another laboratory's.

Gene symbols and gene-disease pairs are real (HGNC symbols, well-known associations); every variant, allele
count, assay result, family, patient, laboratory and SOP number is invented. The curation rules rendered into
each state (points per evidence code, class thresholds, gene-specific frequency cut-offs, the secondary-findings
list, the CNV points sheet) are this fictional laboratory's LOCAL SOP, written for the generator. They borrow the
familiar evidence-code vocabulary but are not, and must not be read as, the published professional guidelines
or any real laboratory's policy.

Each schema draws a `regime` first. Borderline cases put one item next to a cut-off (4 informative meioses
against a 5-meiosis rule, an allele frequency just over the gene's maximum credible frequency, a points total one
short of a class boundary); conflicting cases pair strong evidence in both directions, or carry a draft that
over-weights an assay the SOP says to discount. `answer_key` on each schema is a rule-of-thumb key used only for
the spread check; it is never a label. No protected attribute decides anything.
"""
from __future__ import annotations

from .base import (
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
    show,
    table,
)

# --------------------------------------------------------------------------- gene pool
# (symbol, disorder, inheritance, mechanism, maximum credible AF, critical region or None, on the lab's
#  secondary-findings list, category, protein length). Mechanism "lof": loss of function is the established
#  disease mechanism (null variants count); "missense": disease is driven by specific missense changes.
_G = [
    ("MYH7", "hypertrophic cardiomyopathy", "AD", "missense", 4e-5, "motor domain", True, "cardiac", 1935),
    ("MYBPC3", "hypertrophic cardiomyopathy", "AD", "lof", 4e-5, None, True, "cardiac", 1274),
    ("TNNT2", "dilated cardiomyopathy", "AD", "missense", 2e-5, "tropomyosin-binding region", True, "cardiac", 298),
    ("KCNQ1", "long QT syndrome type 1", "AD", "lof", 3e-5, "pore and S5-S6 region", True, "cardiac", 676),
    ("KCNH2", "long QT syndrome type 2", "AD", "lof", 3e-5, "pore helix", True, "cardiac", 1159),
    ("SCN5A", "Brugada syndrome", "AD", "lof", 5e-5, "voltage-sensor segments", True, "cardiac", 2016),
    ("LMNA", "dilated cardiomyopathy with conduction disease", "AD", "lof", 2e-5, "rod domain", True, "cardiac", 664),
    ("PKP2", "arrhythmogenic cardiomyopathy", "AD", "lof", 1e-4, None, True, "cardiac", 881),
    ("DSP", "arrhythmogenic cardiomyopathy", "AD", "lof", 5e-5, None, True, "cardiac", 2871),
    ("RYR2", "catecholaminergic polymorphic ventricular tachycardia", "AD", "missense", 1e-5,
     "N-terminal hotspot", True, "cardiac", 4967),
    ("FBN1", "Marfan syndrome", "AD", "lof", 2e-5, "calcium-binding EGF-like domains", True, "aortopathy", 2871),
    ("COL3A1", "vascular Ehlers-Danlos syndrome", "AD", "missense", 1e-5, "triple-helical Gly-X-Y region", True,
     "aortopathy", 1466),
    ("TGFBR2", "Loeys-Dietz syndrome", "AD", "missense", 1e-5, "kinase domain", True, "aortopathy", 567),
    ("BRCA1", "hereditary breast and ovarian cancer", "AD", "lof", 5e-5, "RING and BRCT domains", True, "cancer",
     1863),
    ("BRCA2", "hereditary breast and ovarian cancer", "AD", "lof", 5e-5, "DNA-binding domain", True, "cancer", 3418),
    ("PALB2", "hereditary breast cancer", "AD", "lof", 5e-5, None, True, "cancer", 1186),
    ("MLH1", "Lynch syndrome", "AD", "lof", 3e-5, "ATPase domain", True, "cancer", 756),
    ("MSH2", "Lynch syndrome", "AD", "lof", 3e-5, None, True, "cancer", 934),
    ("MSH6", "Lynch syndrome", "AD", "lof", 1e-4, None, True, "cancer", 1360),
    ("APC", "familial adenomatous polyposis", "AD", "lof", 2e-5, None, True, "cancer", 2843),
    ("TP53", "Li-Fraumeni syndrome", "AD", "missense", 1e-5, "DNA-binding domain", True, "cancer", 393),
    ("PTEN", "PTEN hamartoma tumour syndrome", "AD", "lof", 1e-5, "phosphatase domain", True, "cancer", 403),
    ("CDH1", "hereditary diffuse gastric cancer", "AD", "lof", 2e-5, None, True, "cancer", 882),
    ("RAD51C", "hereditary ovarian cancer", "AD", "lof", 5e-5, None, True, "cancer", 376),
    ("STK11", "Peutz-Jeghers syndrome", "AD", "lof", 1e-5, "kinase domain", True, "cancer", 433),
    ("VHL", "von Hippel-Lindau disease", "AD", "lof", 1e-5, "beta domain", True, "cancer", 213),
    ("RET", "multiple endocrine neoplasia type 2", "AD", "missense", 1e-5, "cysteine-rich domain", True, "cancer",
     1114),
    ("LDLR", "familial hypercholesterolaemia", "AD", "lof", 2e-4, "ligand-binding repeats", True, "lipid", 860),
    ("APOB", "familial hypercholesterolaemia", "AD", "missense", 2e-4, "receptor-binding region", True, "lipid", 4563),
    ("RYR1", "malignant hyperthermia susceptibility", "AD", "missense", 5e-5, "MH hotspot region", True,
     "anaesthesia", 5038),
    ("CACNA1S", "malignant hyperthermia susceptibility", "AD", "missense", 5e-5, "voltage-sensor region", True,
     "anaesthesia", 1873),
    ("SCN1A", "Dravet syndrome", "AD", "lof", 5e-6, "pore-forming segments", False, "neurodevelopmental", 2009),
    ("KCNQ2", "developmental and epileptic encephalopathy", "AD", "missense", 5e-6, "pore and voltage sensor",
     False, "neurodevelopmental", 872),
    ("STXBP1", "developmental and epileptic encephalopathy", "AD", "lof", 5e-6, None, False, "neurodevelopmental",
     603),
    ("NF1", "neurofibromatosis type 1", "AD", "lof", 1e-5, "GAP-related domain", False, "neurodevelopmental", 2839),
    ("TSC2", "tuberous sclerosis complex", "AD", "lof", 1e-5, "GAP domain", False, "neurodevelopmental", 1807),
    ("HNF1A", "maturity-onset diabetes of the young", "AD", "lof", 3e-5, "DNA-binding domain", False, "endocrine",
     631),
    ("GCK", "maturity-onset diabetes of the young", "AD", "lof", 3e-5, "glucose-binding site", False, "endocrine",
     465),
    ("PKD1", "autosomal dominant polycystic kidney disease", "AD", "lof", 5e-5, None, False, "renal", 4303),
    ("COL1A1", "osteogenesis imperfecta", "AD", "lof", 1e-5, "triple-helical domain", False, "skeletal", 1464),
    ("CFTR", "cystic fibrosis", "AR", "lof", 1e-2, "nucleotide-binding domain", False, "recessive", 1480),
    ("PAH", "phenylketonuria", "AR", "lof", 8e-3, "catalytic domain", False, "recessive", 452),
    ("GAA", "Pompe disease", "AR", "lof", 4e-3, "catalytic domain", False, "recessive", 952),
    ("GJB2", "nonsyndromic hearing loss", "AR", "lof", 1e-2, None, False, "recessive", 226),
    ("ABCA4", "Stargardt disease", "AR", "lof", 5e-3, None, False, "recessive", 2273),
    ("USH2A", "Usher syndrome type 2", "AR", "lof", 5e-3, None, False, "recessive", 5202),
]
GENES = [dict(sym=a, disorder=b, inh=c, mech=d, max_af=e, rare=e / 5, domain=f, sf=g, cat=h, length=i)
         for a, b, c, d, e, f, g, h, i in _G]
SF_GENES = [g for g in GENES if g["sf"]]
OFF_SF_GENES = [g for g in GENES if not g["sf"]]

AA = ["Ala", "Arg", "Asn", "Asp", "Cys", "Gln", "Glu", "Gly", "His", "Ile", "Leu", "Lys", "Met", "Phe", "Pro",
      "Ser", "Thr", "Trp", "Tyr", "Val"]
# (ref aa, alt aa, ref base, alt base, codon position) -- single-base changes that really give that substitution
MISSENSE = [("Arg", "Trp", "C", "T", 1), ("Arg", "Gln", "G", "A", 2), ("Arg", "Cys", "C", "T", 1),
            ("Arg", "His", "G", "A", 2), ("Gly", "Ser", "G", "A", 1), ("Gly", "Arg", "G", "A", 1),
            ("Gly", "Asp", "G", "A", 2), ("Ala", "Val", "C", "T", 2), ("Ala", "Thr", "G", "A", 1),
            ("Pro", "Leu", "C", "T", 2), ("Pro", "Ser", "C", "T", 1), ("Thr", "Met", "C", "T", 2),
            ("Thr", "Ile", "C", "T", 2), ("Asp", "Asn", "G", "A", 1), ("Asp", "Gly", "A", "G", 2),
            ("Glu", "Lys", "G", "A", 1), ("Glu", "Gly", "A", "G", 2), ("Leu", "Pro", "T", "C", 2),
            ("Leu", "Arg", "T", "G", 2), ("Cys", "Tyr", "G", "A", 2), ("Cys", "Arg", "T", "C", 1),
            ("Tyr", "Cys", "A", "G", 2), ("Tyr", "His", "T", "C", 1), ("Ser", "Phe", "C", "T", 2),
            ("Ser", "Leu", "C", "T", 2), ("Val", "Met", "G", "A", 1), ("Val", "Ala", "T", "C", 2),
            ("Ile", "Thr", "T", "C", 2), ("Ile", "Val", "A", "G", 1), ("Asn", "Ser", "A", "G", 2),
            ("His", "Arg", "A", "G", 2), ("His", "Tyr", "C", "T", 1), ("Lys", "Glu", "A", "G", 1),
            ("Lys", "Arg", "A", "G", 2), ("Met", "Thr", "T", "C", 2), ("Met", "Val", "A", "G", 1),
            ("Phe", "Leu", "T", "C", 1), ("Phe", "Ser", "T", "C", 2), ("Trp", "Arg", "T", "C", 1),
            ("Gln", "Arg", "A", "G", 2), ("Gln", "His", "G", "C", 3), ("Asn", "Lys", "C", "G", 3)]
NONSENSE = [("Arg", "C", "T", 1), ("Gln", "C", "T", 1), ("Trp", "G", "A", 2), ("Tyr", "C", "A", 3),
            ("Glu", "G", "T", 1), ("Lys", "A", "T", 1), ("Ser", "C", "G", 2), ("Cys", "C", "A", 3),
            ("Leu", "T", "A", 2)]
NULL_TYPES = ("nonsense", "frameshift", "canonical_splice", "start_loss")
SPLICE_TYPES = ("canonical_splice", "intronic_splice_region", "synonymous")
CONSEQUENCE = {"missense": "missense", "nonsense": "stop gained", "frameshift": "frameshift",
               "canonical_splice": "canonical splice site", "start_loss": "initiation codon loss",
               "inframe_del": "in-frame deletion", "synonymous": "synonymous",
               "intronic_splice_region": "intronic, splice region"}

VAL_ASSAYS = ["a calibrated cell-based activity assay", "a multiplexed functional assay calibrated on known variants",
              "a patch-clamp current assay run with pathogenic and benign controls",
              "a homology-directed repair reporter assay", "a mismatch-repair activity assay in cell extracts",
              "a protein stability and localisation assay with a control panel",
              "an RNA splicing study in patient cells alongside control samples",
              "a minigene splicing assay run with reference variants",
              "a transcriptional activation assay calibrated against known variants",
              "a deep mutational scan with published classification thresholds",
              "a receptor-uptake assay in transfected cells with control alleles",
              "an enzyme activity assay with benchmarked cut-offs"]
UNVAL_ASSAYS = ["a single-replicate reporter assay in a preprint",
                "a western blot on patient fibroblasts without control lines",
                "a yeast complementation assay not calibrated against known variants",
                "an in-house overexpression assay with no benign controls",
                "a conference abstract describing reduced activity (no methods given)",
                "an RNA study on blood with no control samples",
                "a structural modelling paper with one in-vitro binding test",
                "a thesis chapter reporting reduced protein levels in two cell lines",
                "a zebrafish knock-down with no rescue experiment"]
# the re-review schema words its assays differently (no long word runs shared with the classification schema)
VAL_STUDIES = ["calibrated cell-based activity study", "multiplex functional screen, calibrated thresholds",
               "patch-clamp study including control alleles", "HDR reporter study (calibrated)",
               "MMR activity study in cell extracts (calibrated)", "protein stability study, control panel",
               "patient-cell RNA study with matched controls", "minigene study benchmarked on reference alleles",
               "transactivation study calibrated on known alleles", "saturation mutagenesis study, published cut-offs",
               "receptor uptake study with control alleles", "enzyme kinetics study, benchmarked thresholds"]
UNVAL_STUDIES = ["preprint luciferase reporter (one replicate)", "fibroblast immunoblot, no control lines",
                 "uncalibrated yeast growth rescue", "overexpression study lacking benign controls",
                 "meeting abstract citing lower activity", "blood RNA study, no controls",
                 "modelling paper plus one binding test", "doctoral thesis: less protein in two cell lines",
                 "morpholino knock-down, no rescue"]
LAB_SUFFIX = ["Genomics", "Molecular Diagnostics", "Genetics Laboratory", "Clinical Genomics",
              "Diagnostic Genetics", "Genome Diagnostics"]
LAB_STEMS = ["Alder", "Basalt", "Cobalt", "Driftwood", "Ember", "Fernhill", "Ironbark", "Juniper", "Lakeshore",
             "Oakridge", "Pinecrest", "Redfern", "Saltmarsh", "Upland", "Valewood", "Westbrook", "Yarrow",
             "Ashgrove", "Bluestem", "Copperline", "Dunmore", "Elmstead", "Kestrel", "Larchmont", "Moorfield"]
CLASS_TEXT = {"pathogenic": "pathogenic", "likely_pathogenic": "likely pathogenic",
              "uncertain_significance": "uncertain significance (VUS)", "likely_benign": "likely benign",
              "benign": "benign"}
CLASS_SHORT = {"pathogenic": "P", "likely_pathogenic": "LP", "uncertain_significance": "VUS",
               "likely_benign": "LB", "benign": "B"}
CLASS_ORDER = ["benign", "likely_benign", "uncertain_significance", "likely_pathogenic", "pathogenic"]

# Points and class wording. Each schema words the same local rules its own way, with several variants, so
# states from different schemas never share long word runs (the n-gram audit treats those as copies).
SOP_PC = {
    "germline": (["Points: very strong 8, strong 4, moderate 2, supporting 1; benign strong -4, benign supporting -1.",
                  "Weights -- pathogenic VS=8, S=4, M=2, P=1 | benign S=-4, P=-1.",
                  "Each code scores by its strength: 8 if very strong, 4 if strong, 2 if moderate, 1 if supporting; "
                  "benign codes score -4 (strong) or -1 (supporting)."],
                 ["Class from the total: 10 or more pathogenic; 6 to 9 likely pathogenic; 0 to 5 uncertain "
                  "significance; -1 to -6 likely benign; -7 or less benign. Allele frequency above 5% is benign on "
                  "its own.",
                  "Total >= 10 P | 6-9 LP | 0-5 VUS | -1 to -6 LB | <= -7 B. BA1 (AF > 5%) = B regardless of points.",
                  "Classify on the summed points: ten or more is pathogenic, six to nine likely pathogenic, zero to "
                  "five a VUS, minus one to minus six likely benign, minus seven or lower benign. BA1 alone makes "
                  "a variant benign."]),
    "vus": (["Code weights: 8 / 4 / 2 / 1 for very strong / strong / moderate / supporting; benign -4 / -1.",
             "Scoring: VS 8, S 4, M 2, Sup 1 (benign S -4, Sup -1)."],
            ["Bands: P at 10+, LP 6 to 9, VUS 0 to 5, LB -1 to -6, B -7 and below.",
             "A total of 6-9 is LP and 10+ is P; 0-5 stays VUS; -1 to -6 is LB; -7 or less is B."]),
    "discordance": (["Points ascend by strength: supporting 1, moderate 2, strong 4, very strong 8 (benign: "
                     "supporting -1, strong -4).",
                     "Scale used: sup=1 mod=2 str=4 vstr=8; benign sup=-1 str=-4."],
                    ["Class cut-offs on the total: pathogenic from 10; likely pathogenic from 6; uncertain from 0; "
                     "likely benign down to -6; benign at -7 or lower.",
                     "VUS spans 0..5; LP 6..9; P 10 and up; LB -6..-1; B up to -7."]),
}


def _pc(rng, which):
    pts, cls = SOP_PC[which]
    return pick(rng, pts), pick(rng, cls)


def _lab(rng):
    return f"{pick(rng, LAB_STEMS)} {pick(rng, LAB_SUFFIX)}"


def _sop_id(rng):
    return f"GV-{rng.randint(10, 38)} rev {rng.randint(3, 14)}"


def _cls(total, ba1=False):
    if ba1:
        return "benign"
    if total >= 10:
        return "pathogenic"
    if total >= 6:
        return "likely_pathogenic"
    if total >= 0:
        return "uncertain_significance"
    if total >= -6:
        return "likely_benign"
    return "benign"


def _near_boundary(total):
    return total in (-7, -6, -1, 0, 5, 6, 9, 10)


def _fmt_af(af):
    return "0" if af == 0 else f"{af:.7f}".rstrip("0")


def _variant(rng, g, vtype, last_exon=False):
    """HGVS-style coding and protein descriptions that are internally consistent."""
    hi = max(30, g["length"] - 4)
    codon = rng.randint(max(12, hi - 40), hi) if last_exon else rng.randint(12, hi)
    if vtype == "missense":
        ref, alt, rb, ab, pos = pick(rng, MISSENSE)
        return f"c.{3 * (codon - 1) + pos}{rb}>{ab}", f"p.({ref}{codon}{alt})"
    if vtype == "nonsense":
        ref, rb, ab, pos = pick(rng, NONSENSE)
        return f"c.{3 * (codon - 1) + pos}{rb}>{ab}", f"p.({ref}{codon}*)"
    if vtype == "frameshift":
        n = 3 * (codon - 1) + rng.randint(1, 3)
        kind = pick(rng, ["del", "dup", "del"])
        c = f"c.{n}{kind}" if chance(rng, 0.7) else f"c.{n}_{n + 1}{kind}"
        return c, f"p.({pick(rng, AA)}{codon}{pick(rng, AA)}fs*{rng.randint(3, 48)})"
    if vtype == "canonical_splice":
        n = 3 * (codon - 1) + rng.randint(1, 3)
        return pick(rng, [f"c.{n}+1G>A", f"c.{n}+2T>C", f"c.{n}+1G>T", f"c.{n + 1}-1G>A",
                          f"c.{n + 1}-2A>G", f"c.{n + 1}-1G>C"]), "p.?"
    if vtype == "intronic_splice_region":
        n = 3 * (codon - 1) + rng.randint(1, 3)
        return pick(rng, [f"c.{n}+5G>A", f"c.{n}+3A>G", f"c.{n}+4A>G", f"c.{n + 1}-6T>G", f"c.{n + 1}-9C>A",
                          f"c.{n + 1}-12T>A"]), "p.?"
    if vtype == "synonymous":
        aa = pick(rng, ["Gly", "Ala", "Leu", "Val", "Pro", "Thr", "Ser", "Arg"])
        return f"c.{3 * codon}{pick(rng, ['C>T', 'G>A', 'T>C', 'A>G'])}", f"p.({aa}{codon}=)"
    if vtype == "inframe_del":
        n = 3 * (codon - 1) + 1
        return f"c.{n}_{n + 2}del", f"p.({pick(rng, AA)}{codon}del)"
    return pick(rng, ["c.1A>G", "c.2T>C", "c.3G>A"]), "p.(Met1?)"


def _pop(rng, g, cat):
    """Population allele count / number / frequency for a target band of the gene's cut-offs."""
    an = rng.randrange(118000, 282000, 2)
    rare, mx = g["rare"], g["max_af"]
    if cat == "absent":
        t = 0.0
    elif cat == "rare":
        t = rng.uniform(rare * 0.2, rare)
    elif cat == "mid":
        t = rng.uniform(rare * 1.3, mx * 0.95)
    elif cat == "above":
        t = rng.uniform(mx * 1.2, min(0.045, mx * 15))
    elif cat == "ba1":
        t = rng.uniform(0.055, 0.21)
    elif cat == "near_rare":
        t = near(rng, rare, rare * 0.45, lo=rare * 0.3, digits=10)
    else:  # near_max
        t = near(rng, mx, mx * 0.3, lo=mx * 0.55, digits=10)
    ac = 0 if t == 0 else max(1, round(t * an))
    af = ac / an
    hom = int(round(af * af * an / 2)) if af > 0.002 else 0
    return {"ac": ac, "an": an, "af": af, "hom": hom}


# =========================================================================== evidence -> points (the SOP)
def _codes(f, g):
    """(code, points) the local SOP assigns to the RAW evidence in f. BA1 carries points None."""
    out = []
    null = f["vtype"] in NULL_TYPES
    pvs1 = null and g["mech"] == "lof"
    if pvs1:
        out.append(("PVS1_Moderate", 2) if f["last_exon"] else ("PVS1", 8))
    af = f["pop"]["af"]
    if af > 0.05:
        out.append(("BA1", None))
    elif af > g["max_af"]:
        out.append(("BS1", -4))
    elif af <= g["rare"]:
        out.append(("PM2_Supporting", 1))
    fn = f["functional"]
    if fn == "val_dmg":
        out.append(("PS3", 4))
    elif fn == "unval_dmg":
        out.append(("PS3_Supporting", 1))
    elif fn in ("val_norm", "conflict"):
        out.append(("BS3", -4))
    elif fn == "unval_norm":
        out.append(("BS3_Supporting", -1))
    if f["de_novo"] == "confirmed":
        out.append(("PS2", 4))
    elif f["de_novo"] == "unconfirmed":
        out.append(("PM6", 2))
    if f["nonseg"]:
        out.append(("BS4", -4))
    else:
        m = f["meioses"]
        if m >= 7:
            out.append(("PP1_Strong", 4))
        elif m >= 5:
            out.append(("PP1_Moderate", 2))
        elif m >= 3:
            out.append(("PP1", 1))
    if g["inh"] == "AD" and af <= g["rare"]:
        p = f["probands"]
        if p >= 6:
            out.append(("PS4", 4))
        elif p >= 3:
            out.append(("PS4_Moderate", 2))
        elif p >= 1:
            out.append(("PS4_Supporting", 1))
    if g["inh"] == "AR":
        if f["in_trans"] >= 2:
            out.append(("PM3_Strong", 4))
        elif f["in_trans"] == 1:
            out.append(("PM3", 2))
    if f["domain_hit"]:
        out.append(("PM1", 2))
    if f["predictor"] is not None:
        if f["predictor"] >= 0.75:
            out.append(("PP3", 1))
        elif f["predictor"] <= 0.15:
            out.append(("BP4", -1))
    if f["splice_delta"] is not None and not pvs1:
        if f["splice_delta"] >= 0.5:
            out.append(("PP3", 1))
        elif f["splice_delta"] < 0.1:
            out.append(("BP4", -1))
            if f["vtype"] == "synonymous":
                out.append(("BP7", -1))
    if f["healthy_obs"]:
        out.append(("BS2", -4))
    return out


def _tally(codes):
    ba1 = any(c == "BA1" for c, _ in codes)
    pos = sum(p for _, p in codes if p and p > 0)
    neg = -sum(p for _, p in codes if p and p < 0)
    return pos, neg, pos - neg, ba1


# =========================================================================== schema 1
class GermlineVariantClassification(Schema):
    name = "germline_variant_classification"
    description = ("A sequence variant's raw evidence and a curator's draft codes, reviewed against the lab's "
                   "points-based classification SOP before sign-off.")
    surfaces = {"json": 0.5, "curation_worksheet": 0.25, "signout_email": 0.25}

    DRAFT_ERRORS = ["pvs1_last_exon", "pvs1_wrong_mechanism", "ps3_unvalidated", "pm2_not_rare",
                    "ps4_not_rare", "pp1_overcount", "bs1_missed", "bs4_missed", "ps2_unconfirmed",
                    "ps3_underweighted", "pp3_with_pvs1"]

    def sample_factors(self, rng):
        reg = regime(rng)
        g = pick(rng, GENES)
        ad = g["inh"] == "AD"
        if reg == "clear":
            direction = pick(rng, ["path", "benign"])
        elif reg == "borderline":
            direction = pick_w(rng, {"path": 4, "benign": 3, "mixed": 3})
        else:
            direction = pick_w(rng, {"mixed": 6, "path": 3, "benign": 1})
        P = direction == "path" or (direction == "mixed" and chance(rng, 0.5))

        def side():
            # in a mixed case each item independently leans one way or the other
            return P if direction != "mixed" else chance(rng, 0.5)

        # ---------------------------------------------------------------- variant type
        if P and g["mech"] == "lof":
            vtype = pick_w(rng, {"nonsense": 2, "frameshift": 2, "canonical_splice": 1.5, "start_loss": 0.3,
                                 "missense": 3.5, "inframe_del": 0.8})
        elif P:
            vtype = pick_w(rng, {"missense": 8, "inframe_del": 1.5})
        else:
            vtype = pick_w(rng, {"missense": 5, "synonymous": 2, "intronic_splice_region": 2, "inframe_del": 0.8})
        if reg == "conflicting" and g["mech"] == "missense" and chance(rng, 0.3):
            vtype = pick(rng, ["nonsense", "frameshift"])          # a null variant where nulls are not the mechanism
        last_exon = vtype in NULL_TYPES and vtype != "start_loss" and chance(rng, 0.22 if reg != "clear" else 0.05)

        # ---------------------------------------------------------------- population frequency
        if reg == "borderline" and chance(rng, 0.55):
            af_cat = pick(rng, ["near_rare", "near_max", "near_max"])
        elif side():
            af_cat = pick_w(rng, {"absent": 5, "rare": 3, "mid": 1.5, "above": 1.3})
        else:
            af_cat = pick_w(rng, {"above": 4.5, "mid": 2.5, "rare": 1, "absent": 0.5,
                                  "ba1": 1.2 if reg == "clear" else 0.25})
        pop = _pop(rng, g, af_cat)

        # ---------------------------------------------------------------- computational evidence
        predictor = splice = None
        if vtype in ("missense", "inframe_del"):
            if side():
                predictor = round(pick_w(rng, {rng.uniform(0.76, 0.98): 6, rng.uniform(0.3, 0.7): 3,
                                               rng.uniform(0.05, 0.15): 1}), 2)
            else:
                predictor = round(pick_w(rng, {rng.uniform(0.02, 0.15): 6, rng.uniform(0.3, 0.7): 3,
                                               rng.uniform(0.76, 0.95): 1}), 2)
            if reg == "borderline" and chance(rng, 0.35):
                predictor = round(pick(rng, [near(rng, 0.75, 0.04), near(rng, 0.15, 0.03)]), 2)
        elif vtype in SPLICE_TYPES:
            if vtype == "canonical_splice":
                splice = round(rng.uniform(0.55, 0.99), 2)
            elif side():
                splice = round(pick_w(rng, {rng.uniform(0.5, 0.9): 6, rng.uniform(0.1, 0.45): 3}), 2)
            else:
                splice = round(pick_w(rng, {rng.uniform(0.0, 0.09): 6, rng.uniform(0.1, 0.45): 3}), 2)
            if reg == "borderline" and vtype != "canonical_splice" and chance(rng, 0.35):
                splice = round(pick(rng, [near(rng, 0.5, 0.05), near(rng, 0.1, 0.02, lo=0.0)]), 2)

        # ---------------------------------------------------------------- functional, family and case data
        if side():
            functional = pick_w(rng, {"val_dmg": 3, "unval_dmg": 3, "none": 4})
        else:
            functional = pick_w(rng, {"val_norm": 3, "unval_norm": 1.5, "none": 5})
        if reg == "conflicting" and chance(rng, 0.3):
            functional = pick(rng, ["conflict", "unval_dmg"])
        de_novo = "none"
        if ad and g["cat"] in ("neurodevelopmental", "skeletal", "aortopathy", "cardiac") and side() and chance(rng, 0.45):
            de_novo = pick(rng, ["confirmed", "unconfirmed"])
        if side():
            meioses = pick(rng, [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            nonseg = chance(rng, 0.04)
        else:
            meioses = pick(rng, [0, 0, 0, 1, 2, 3])
            nonseg = chance(rng, 0.35)
        if reg == "borderline" and chance(rng, 0.4) and not nonseg:
            meioses = pick(rng, [2, 3, 4, 5, 6, 7])
        probands = 0
        in_trans = 0
        if ad:
            probands = pick(rng, [0, 1, 2, 3, 4, 5, 6, 8]) if side() else pick(rng, [0, 0, 1, 2])
            if reg == "borderline" and chance(rng, 0.3):
                probands = pick(rng, [2, 3, 5, 6])
        else:
            in_trans = pick(rng, [0, 1, 1, 2, 3]) if side() else pick(rng, [0, 0, 1])
        domain_hit = bool(g["domain"]) and vtype in ("missense", "inframe_del") and chance(rng, 0.55 if side() else 0.15)
        healthy_obs = (not side()) and chance(rng, 0.3)
        # one contrary item against the main direction: weak in borderline cases, strong in conflicting ones
        p_contra = {"clear": 0.0, "borderline": 0.5, "conflicting": 0.65}[reg] if direction != "mixed" else 0.0
        if chance(rng, p_contra):
            strong = reg == "conflicting" and chance(rng, 0.6)
            if direction == "path":
                opts = (["nonseg", "val_norm", "healthy"] if strong else ["unval_norm", "pred_low", "af_mid"])
            else:
                opts = (["val_dmg", "de_novo", "meioses"] if strong else ["unval_dmg", "pred_high", "probands"])
            c = pick(rng, opts)
            if c == "nonseg":
                nonseg = True
            elif c == "val_norm":
                functional = "conflict" if functional in ("val_dmg", "unval_dmg") else "val_norm"
            elif c == "healthy":
                healthy_obs = True
            elif c == "unval_norm" and functional == "none":
                functional = "unval_norm"
            elif c == "unval_dmg" and functional == "none":
                functional = "unval_dmg"
            elif c == "val_dmg":
                functional = "val_dmg"
            elif c == "de_novo" and ad:
                de_novo = pick(rng, ["confirmed", "unconfirmed"])
            elif c == "meioses":
                meioses, nonseg = pick(rng, [5, 6, 7, 8]), False
            elif c == "probands" and ad:
                probands = pick(rng, [1, 2, 3])
            elif c == "pred_low" and predictor is not None:
                predictor = round(rng.uniform(0.03, 0.15), 2)
            elif c == "pred_high" and predictor is not None:
                predictor = round(rng.uniform(0.76, 0.93), 2)
            elif c == "af_mid" and pop["ac"] == 0:
                pop = _pop(rng, g, "mid")
        f = {"regime": reg, "direction": direction, "gene": g["sym"], "vtype": vtype, "last_exon": last_exon,
             "af_band": af_cat, "pop": pop, "predictor": predictor, "splice_delta": splice,
             "functional": functional, "de_novo": de_novo, "meioses": meioses, "nonseg": nonseg,
             "probands": probands, "in_trans": in_trans, "domain_hit": domain_hit, "healthy_obs": healthy_obs}
        c, p = _variant(rng, g, vtype, last_exon)
        f.update({"hgvs_c": c, "hgvs_p": p})

        # ---------------------------------------------------------------- the curator's draft
        codes = _codes(f, g)
        p_err = {"clear": 0.05, "borderline": 0.3, "conflicting": 0.7}[reg]
        err = None
        if chance(rng, p_err):
            cand = self._applicable_errors(f, g, codes)
            err = pick(rng, cand) if cand else None
        draft = self._draft(codes, err, f, g)
        pos, neg, total, ba1 = _tally(codes)
        dpos, dneg, dtotal, dba1 = _tally(draft)
        # public archive submissions loosely track the true class, sometimes span both sides
        true_cls = _cls(total, ba1)
        n_sub = pick(rng, [0, 1, 2, 2, 3, 4])
        subs = []
        for _ in range(n_sub):
            i = CLASS_ORDER.index(true_cls) + pick(rng, [0, 0, 0, 1, -1, 1, -1, 2, -2])
            subs.append(CLASS_ORDER[max(0, min(4, i))])
        f.update({
            "draft_error": err, "draft_codes": [c for c, _ in draft], "draft_total": dtotal,
            "draft_class": _cls(dtotal, dba1), "submissions": subs,
            "curator": person(rng), "director": person(rng), "sop_id": _sop_id(rng),
            "case_id": ident(rng, "VC", 6, "-"), "proband_id": ident(rng, "GX", 5, ""),
            "indication": g["disorder"], "zygosity": "heterozygous" if ad or chance(rng, 0.6) else "homozygous",
            "assay": pick(rng, VAL_ASSAYS) if functional in ("val_dmg", "val_norm", "conflict") else
            pick(rng, UNVAL_ASSAYS) if functional in ("unval_dmg", "unval_norm") else None,
            "assay2": pick(rng, UNVAL_ASSAYS) if functional == "conflict" else None,
            "controls": rng.randint(8, 40), "families": max(1, (meioses + 2) // 3) if meioses else 0,
            "date": rand_date(rng),
            "curator_tone": pick(rng, ["confident", "hedged", "neutral", "neutral"]),
        })
        return f

    def _applicable_errors(self, f, g, codes):
        names = [c for c, _ in codes]
        out = []
        if "PVS1_Moderate" in names:
            out.append("pvs1_last_exon")
        if f["vtype"] in NULL_TYPES and g["mech"] == "missense":
            out.append("pvs1_wrong_mechanism")
        if f["functional"] in ("unval_dmg", "conflict"):
            out.append("ps3_unvalidated")
        if "PM2_Supporting" not in names and "BA1" not in names:
            out.append("pm2_not_rare")
        if g["inh"] == "AD" and f["probands"] >= 3 and f["pop"]["af"] > g["rare"]:
            out.append("ps4_not_rare")
        if "PP1_Moderate" in names or "PP1" in names:
            out.append("pp1_overcount")
        if "BS1" in names:
            out.append("bs1_missed")
        if "BS4" in names:
            out.append("bs4_missed")
        if f["de_novo"] == "unconfirmed":
            out.append("ps2_unconfirmed")
        if "PS3" in names:
            out.append("ps3_underweighted")
        if "PVS1" in names and f["vtype"] == "canonical_splice":
            out.append("pp3_with_pvs1")
        return out

    def _draft(self, codes, err, f, g):
        d = list(codes)
        rep = {"pvs1_last_exon": ("PVS1_Moderate", ("PVS1", 8)), "ps2_unconfirmed": ("PM6", ("PS2", 4)),
               "ps3_underweighted": ("PS3", ("PS3_Supporting", 1)), "pp1_overcount": (None, None)}
        if err in ("pvs1_last_exon", "ps2_unconfirmed", "ps3_underweighted"):
            old, new = rep[err]
            d = [new if c == old else (c, p) for c, p in d]
        elif err == "pvs1_wrong_mechanism":
            d.insert(0, ("PVS1", 8))
        elif err == "ps3_unvalidated":
            d = [(c, p) for c, p in d if c not in ("PS3_Supporting", "BS3")] + [("PS3", 4)]
        elif err == "pm2_not_rare":
            d = [(c, p) for c, p in d if c != "BS1"] + [("PM2_Supporting", 1)]
        elif err == "ps4_not_rare":
            p = f["probands"]
            d.append(("PS4", 4) if p >= 6 else ("PS4_Moderate", 2))
        elif err == "pp1_overcount":
            d = [("PP1_Strong", 4) if c in ("PP1", "PP1_Moderate") else (c, p) for c, p in d]
        elif err == "bs1_missed":
            d = [(c, p) for c, p in d if c != "BS1"]
        elif err == "bs4_missed":
            d = [(c, p) for c, p in d if c != "BS4"] + ([("PP1", 1)] if f["meioses"] >= 3 else [])
        elif err == "pp3_with_pvs1":
            d.append(("PP3", 1))
        return d

    # ------------------------------------------------------------------ text pieces
    def _evidence(self, f, g):
        """Raw evidence, one entry per evidence line (dict: section -> text)."""
        pop = f["pop"]
        ev = {}
        if pop["ac"] == 0:
            ev["population"] = f"absent from popDB v4.1 ({pop['an']:,} alleles, region well covered)"
        else:
            ev["population"] = (f"popDB v4.1: {pop['ac']} of {pop['an']:,} alleles (AF {_fmt_af(pop['af'])})"
                                + (f", {pop['hom']} homozygotes" if pop["hom"] else ", no homozygotes"))
        if f["predictor"] is not None:
            ev["computational"] = f"missense meta-predictor score {f['predictor']:.2f}"
        if f["splice_delta"] is not None:
            ev["computational"] = (f"splice predictor delta {f['splice_delta']:.2f} "
                                   f"({'donor' if '+' in f['hgvs_c'] else 'acceptor'} site)")
        if f["vtype"] in NULL_TYPES and f["vtype"] != "start_loss":
            ev["transcript_effect"] = ("in the last coding exon; predicted to escape nonsense-mediated decay"
                                       if f["last_exon"] else "predicted to trigger nonsense-mediated decay")
        fn = f["functional"]
        if fn in ("val_dmg", "val_norm"):
            ev["functional"] = (f"{f['assay']} (validated against {f['controls']} control variants): "
                                + ("loss of function" if fn == "val_dmg" else "activity in the normal range"))
        elif fn in ("unval_dmg", "unval_norm"):
            ev["functional"] = f"{f['assay']}: " + ("reduced function reported" if fn == "unval_dmg"
                                                    else "no effect reported")
        elif fn == "conflict":
            ev["functional"] = (f"{f['assay']} (validated against {f['controls']} control variants): normal activity; "
                                f"separately, {f['assay2']} reports reduced function")
        if f["nonseg"]:
            ev["segregation"] = "an affected first-degree relative tested NEGATIVE for the variant"
        elif f["meioses"]:
            ev["segregation"] = (f"co-segregates with disease in {f['families']} famil"
                                 f"{'y' if f['families'] == 1 else 'ies'}; informative meioses: {f['meioses']}")
        if f["de_novo"] == "confirmed":
            ev["de_novo"] = "absent in both unaffected parents; parentage confirmed by SNP fingerprinting"
        elif f["de_novo"] == "unconfirmed":
            ev["de_novo"] = "absent in both unaffected parents; parentage not confirmed (samples sent separately)"
        if f["probands"]:
            ev["case_data"] = (f"seen in {f['probands']} unrelated proband{'s' if f['probands'] > 1 else ''} "
                               f"with {g['disorder']} (lab cohort and literature)")
        if f["in_trans"]:
            ev["case_data"] = (f"found in trans with a pathogenic variant in {f['in_trans']} affected "
                               f"individual{'s' if f['in_trans'] > 1 else ''} (phase confirmed by parental testing)")
        if f["domain_hit"]:
            ev["location"] = f"residue lies in the {g['domain']}, a lab-defined critical region"
        elif g["domain"] and f["vtype"] == "missense":
            ev["location"] = f"outside the {g['domain']}"
        if f["healthy_obs"]:
            ev["unaffected_carriers"] = ("homozygous in a healthy adult in the lab's control series" if g["inh"] == "AR"
                                         else "carried by three unaffected relatives past the usual onset of disease")
        return ev

    def _sop(self, f, g):
        """Only the SOP lines that bear on the evidence present (plus points and classes)."""
        r = [f"Frequency cut-offs for {g['sym']} ({g['disorder']}): rarity at or below AF {_fmt_af(g['rare'])}; "
             f"maximum credible AF {_fmt_af(g['max_af'])} (above it: BS1)."]
        if f["vtype"] in NULL_TYPES:
            r.append("PVS1 only where loss of function is the established mechanism for the gene "
                     + f"({g['sym']}: {'yes' if g['mech'] == 'lof' else 'no, disease is missense-driven'}); "
                     "moderate if the variant escapes nonsense-mediated decay; do not add PP3 to PVS1.")
        if f["functional"] != "none":
            r.append("Functional data: validated assay (control variants run) = PS3/BS3 strong; unvalidated = "
                     "supporting; if a validated and an unvalidated assay disagree, use the validated one only.")
        if f["meioses"] or f["nonseg"]:
            r.append("Segregation: 7+ informative meioses PP1_Strong, 5-6 moderate, 3-4 supporting; an affected "
                     "relative without the variant = BS4.")
        if f["de_novo"] != "none":
            r.append("De novo: parentage confirmed PS2 (strong), unconfirmed PM6 (moderate).")
        if f["probands"]:
            r.append("Unrelated probands (only if the variant is rare): 6+ strong, 3-5 moderate, 1-2 supporting.")
        if f["in_trans"]:
            r.append("In trans with a pathogenic variant: one observation PM3 (moderate), two or more strong.")
        if f["predictor"] is not None or f["splice_delta"] is not None:
            r.append("Predictors: missense score 0.75 or more PP3, 0.15 or less BP4; splice delta 0.50 or more "
                     "PP3, below 0.10 BP4 (plus BP7 if synonymous).")
        if f["domain_hit"]:
            r.append("Lab-defined critical region (missense): PM1 moderate.")
        if f["healthy_obs"]:
            r.append("Unaffected carriers incompatible with the disorder's penetrance: BS2 strong.")
        return r

    SECOND = ("Second-curator review before sign-off if the total is within one point of a class boundary, if the "
              "reviewed codes change the draft's class, or if public submissions include both a pathogenic-side "
              "and a benign-side call.")

    def _note(self, f, rng):
        dc = CLASS_TEXT[f["draft_class"]]
        s = {"confident": [f"Fairly clear-cut, I'd sign this out as {dc}.", f"Happy this is {dc}."],
             "hedged": [f"Leaning {dc} but would like a second look at the weighting.",
                        f"Drafted as {dc}; not fully sure about the functional data."],
             "neutral": [f"Draft class {dc}.", f"Codes as listed; draft {dc}."]}[f["curator_tone"]]
        extra = []
        if f["draft_error"] == "ps3_unvalidated":
            extra.append("The assay result looks convincing to me.")
        if f["draft_error"] == "pvs1_wrong_mechanism":
            extra.append("Truncating, so PVS1.")
        if f["draft_error"] == "bs1_missed":
            extra.append("Frequency seems low enough for an adult-onset condition.")
        if f["submissions"]:
            extra.append("Public submissions noted but not used as evidence.")
        return noisy(join_sentences([pick(rng, s)] + extra), rng, 0.12)

    def _subs_text(self, f):
        if not f["submissions"]:
            return "no public submissions"
        counts = {}
        for s in f["submissions"]:
            counts[s] = counts.get(s, 0) + 1
        return ", ".join(f"{CLASS_SHORT[k]} x{v}" for k, v in sorted(counts.items(), key=lambda kv: CLASS_ORDER.index(kv[0])))

    def render(self, f, rng, surface):
        g = next(x for x in GENES if x["sym"] == f["gene"])
        ev = self._evidence(f, g)
        sop = self._sop(f, g)
        note = self._note(f, rng)
        sp, sc = _pc(rng, "germline")
        codes = ", ".join(f["draft_codes"]) or "none"
        variant = f"{g['sym']} {f['hgvs_c']} {f['hgvs_p']}"
        if surface == "json":
            return dumps(prune({
                "variant": {"case": f["case_id"], "gene": g["sym"], "hgvs": f"{f['hgvs_c']} {f['hgvs_p']}",
                            "consequence": CONSEQUENCE[f["vtype"]], "zygosity": f["zygosity"],
                            "disorder": g["disorder"], "inheritance": g["inh"]},
                "evidence": ev,
                "draft": {"curator": f["curator"], "codes": codes, "points_claimed": f["draft_total"],
                          "proposed_class": CLASS_TEXT[f["draft_class"]], "note": note},
                "public_submissions": self._subs_text(f),
                "sop": {"id": f"{f['sop_id']} (local)", "points": sp, "classes": sc,
                        "rules": sop, "second_review": self.SECOND},
            }))
        if surface == "curation_worksheet":
            head = kv_lines([("Case", f["case_id"]), ("Proband", f["proband_id"]), ("Variant", variant),
                             ("Consequence", CONSEQUENCE[f["vtype"]]), ("Zygosity", f["zygosity"]),
                             ("Disorder / inheritance", f"{g['disorder']} / {g['inh']}"),
                             ("Curator", f["curator"]), ("Drafted", fmt_date(f["date"], rng))], rng)
            evl = bullets([f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in ev.items()], rng)
            return (f"VARIANT CURATION WORKSHEET\n{head}\n\nEvidence\n{evl}\n\n"
                    f"Draft codes: {codes} = {f['draft_total']} pts -> {CLASS_TEXT[f['draft_class']]}\n"
                    f"Curator note: {note}\nPublic archive: {self._subs_text(f)}\n\n"
                    f"SOP {f['sop_id']} (local) excerpt\n" + "\n".join([sp, sc] + sop + [self.SECOND]))
        # signout_email
        lines = [f"Subject: sign-out {f['case_id']} -- {variant}", f"From: {f['curator']}",
                 f"To: {f['director']}", "",
                 f"Hi {f['director'].split()[0]},", "",
                 f"Draft for {f['proband_id']} ({g['disorder']}, {f['zygosity']}). {note}", "",
                 "What we have:", bullets([v[0].upper() + v[1:] for v in ev.values()], rng), "",
                 f"Codes applied: {codes} ({f['draft_total']} points). Public archive: {self._subs_text(f)}.", "",
                 f"For reference, {f['sop_id']}: " + " ".join([sp, sc] + sop + [self.SECOND]), "",
                 f["curator"].split()[0]]
        return "\n".join(lines)

    def questions(self):
        return {
            "classification_call": choice("Under the lab's SOP, how should this variant be classified?", {
                "benign": "Benign: the evidence against a disease role reaches the benign threshold or the "
                          "frequency alone rules it out.",
                "likely_benign": "Likely benign: the evidence leans against a disease role but stops short of benign.",
                "likely_pathogenic": "Likely pathogenic: the correctly weighted evidence reaches the likely "
                                     "pathogenic band but not the pathogenic one.",
                "pathogenic": "Pathogenic: the correctly weighted evidence reaches the pathogenic threshold.",
                "uncertain_significance": "Uncertain significance: the evidence does not settle the variant "
                                          "either way.",
            }),
            "pathogenic_evidence_weight": score(
                "How much pathogenic evidence does the SOP allow this variant once each item is weighted "
                "correctly?", [
                    "Next to none: at most a supporting point.",
                    "Some: a few points, short of the likely pathogenic band on their own.",
                    "Substantial: enough on their own for likely pathogenic.",
                    "Overwhelming: enough on their own for pathogenic.",
                ]),
            "frequency_compatible": noul(
                "The variant's population frequency is compatible with it causing the disorder.",
                yes="The frequency is at or below the gene's maximum credible allele frequency in the SOP.",
                no="The frequency is above the gene's maximum credible frequency, or above the stand-alone "
                   "benign level."),
            "evidence_conflict": score("How strongly does the evidence for this variant pull in opposite directions?", [
                "Not at all: every item points the same way.",
                "Slightly: one weak item points the other way.",
                "Materially: contrary evidence of real weight sits against the main direction.",
                "Sharply: strong evidence on both sides.",
            ]),
            "second_curator_needed": noul(
                "The SOP requires a second curator to review this call before it is signed out."),
        }

    def answer_key(self, f):
        """Rule-of-thumb answer key for the spread check only; never used as a label."""
        g = next(x for x in GENES if x["sym"] == f["gene"])
        codes = _codes(f, g)
        pos, neg, total, ba1 = _tally(codes)
        cls = _cls(total, ba1)
        conflict = min(pos, 8 if ba1 else neg)
        sides = {("path" if s in ("pathogenic", "likely_pathogenic") else "ben" if s in ("benign", "likely_benign")
                  else "vus") for s in f["submissions"]}
        second = (_near_boundary(total) and not ba1) or f["draft_class"] != cls or {"path", "ben"} <= sides
        return {
            "classification_call": cls,
            "pathogenic_evidence_weight": 0 if pos <= 1 else 1 if pos <= 5 else 2 if pos <= 9 else 3,
            "frequency_compatible": "yes" if f["pop"]["af"] <= g["max_af"] else "no",
            "evidence_conflict": 0 if conflict == 0 else 1 if conflict <= 1 else 2 if conflict <= 3 else 3,
            "second_curator_needed": "yes" if second else "no",
        }


# =========================================================================== schema 2
class VusReclassificationRequest(Schema):
    name = "vus_reclassification_request"
    description = ("A previously reported variant of uncertain significance re-reviewed because new evidence "
                   "arrived or a clinician asked: reclassify, keep, or ask for more data.")
    surfaces = {"json": 0.5, "clinician_request": 0.25, "reanalysis_note": 0.25}

    REASONS = {
        "routine": ("scheduled re-review of reported VUSs", None),
        "cascade": ("relatives are asking whether predictive testing is worthwhile", None),
        "surgery": ("risk-reducing surgery is being planned", "surgery date"),
        "device": ("a decision on a preventive implantable defibrillator is pending", "device clinic"),
        "prenatal": ("the patient's partner is pregnant and prenatal testing is being discussed", "pregnancy"),
        "screening": ("an intensive screening programme is being considered", None),
    }
    ORIG_ITEMS = [("PM2_Supporting", 1), ("PP3", 1), ("PM1", 2), ("PS4_Supporting", 1), ("PP1", 1),
                  ("PS3_Supporting", 1), ("BP4", -1)]

    def sample_factors(self, rng):
        reg = regime(rng)
        g = pick(rng, [x for x in GENES if x["inh"] == "AD"]) if chance(rng, 0.85) else pick(rng, GENES)
        # an "overlapping" new item re-reports something the original review counted, so that code must be there
        overlap = chance(rng, {"clear": 0.12, "borderline": 0.4, "conflicting": 0.8}[reg])
        forced_up = (pick(rng, ["segregation", "func_val", "func_unval"] + (["probands"] if g["inh"] == "AD" else []))
                     if overlap else None)
        need = {"segregation": {"PP1"}, "probands": {"PS4_Supporting", "PM2_Supporting"},
                "func_val": {"PS3_Supporting"}, "func_unval": {"PS3_Supporting"}, None: set()}[forced_up]
        vtype = pick(rng, ["missense", "missense", "missense", "inframe_del", "intronic_splice_region"])
        # original classification: a VUS with a total of 0-5
        while True:
            k = rng.randint(1, 4)
            orig = rng.sample(self.ORIG_ITEMS, k)
            if not g["domain"] or vtype == "intronic_splice_region":
                orig = [x for x in orig if x[0] != "PM1"]
            orig += [x for x in self.ORIG_ITEMS if x[0] in need and x not in orig]
            tot = sum(p for _, p in orig)
            if orig and 0 <= tot <= 5 and not ({"PP3", "BP4"} <= {c for c, _ in orig}):
                break
        orig.sort(key=lambda x: -x[1])
        orig_names = [c for c, _ in orig]
        orig_probands = rng.randint(1, 2) if "PS4_Supporting" in orig_names else 0
        orig_meioses = rng.randint(3, 4) if "PP1" in orig_names else 0

        # new evidence items: (kind, points delta, independent?)
        kinds = ["func_val", "func_unval", "segregation", "probands", "population", "de_novo"]
        if g["inh"] == "AR":
            kinds = ["func_val", "func_unval", "segregation", "population", "in_trans"]
        n_items = 1 if reg == "clear" else pick(rng, [1, 2, 2, 3])
        if reg == "clear":
            # one decisive item: a validated assay, a frequency update, strong segregation, a de novo case
            chosen = [pick(rng, [k for k in kinds if k in ("func_val", "population", "segregation", "de_novo",
                                                           "in_trans")])]
        else:
            chosen = rng.sample(kinds, n_items)
        if overlap and forced_up not in chosen:
            chosen[0] = forced_up
        items = []
        for kind in chosen:
            it = {"kind": kind, "independent": True}
            up = chance(rng, {"clear": 0.6, "borderline": 0.65, "conflicting": 0.55}[reg]) or kind == forced_up
            if kind == "func_val":
                it["result"] = "damaging" if up else "normal"
                it["pts"] = 4 if up else -4
                if "PS3_Supporting" in orig_names:
                    it["pts"] = 3 if up else -5          # replaces the supporting functional point
            elif kind == "func_unval":
                it["result"] = "damaging" if up else "normal"
                it["pts"] = 1 if up else -1
                if "PS3_Supporting" in orig_names:
                    it["pts"] = 0 if up else -2
            elif kind == "segregation":
                if up:
                    m = orig_meioses + rng.randint(1, 6)
                    if reg == "clear":
                        m = rng.randint(7, 10)
                    if reg == "borderline":
                        m = pick(rng, [4, 5, 6, 7])
                    it["meioses_total"] = m
                    new_pp1 = 4 if m >= 7 else 2 if m >= 5 else 1 if m >= 3 else 0
                    it["pts"] = new_pp1 - (1 if "PP1" in orig_names else 0)
                    it["result"] = "co-segregates"
                else:
                    it["result"] = "non-segregation"
                    it["pts"] = -4 - (1 if "PP1" in orig_names else 0)
            elif kind == "probands":
                k_new = rng.randint(1, 5)
                if reg == "borderline":
                    k_new = pick(rng, [1, 2, 3, 4])
                it["new_probands"] = k_new
                tot_p = orig_probands + k_new
                tier = 4 if tot_p >= 6 else 2 if tot_p >= 3 else 1
                it["pts"] = tier - (1 if "PS4_Supporting" in orig_names else 0)
                it["result"] = "case series"
                if "PM2_Supporting" not in orig_names:
                    it["pts"] = 0                      # probands only count for rare variants
                    it["not_rare"] = True
            elif kind == "population":
                if up and reg != "clear":
                    it["result"] = "still rare"
                    it["pts"] = 0
                    it["ac"] = rng.randint(0, 1)
                else:
                    it["result"] = "now common"
                    # BS1 now applies; rarity (PM2) and the rarity-dependent proband code both fall away
                    it["pts"] = -4 - (1 if "PM2_Supporting" in orig_names else 0) \
                        - (1 if "PS4_Supporting" in orig_names else 0)
                    it["ac"] = 0
                it["an"] = rng.randrange(600000, 810000, 2)
                if it["result"] == "now common":
                    it["ac"] = max(2, round(g["max_af"] * rng.uniform(1.4, 6) * it["an"]))
                elif it["ac"] and it["ac"] / it["an"] > g["rare"]:
                    it["ac"] = 0
            elif kind == "de_novo":
                it["result"] = "confirmed" if reg == "clear" else pick(rng, ["confirmed", "unconfirmed"])
                it["pts"] = 4 if it["result"] == "confirmed" else 2
            else:  # in_trans
                it["result"] = "in trans"
                it["pts"] = 2
            items.append(it)
        # conflicting / some borderline: one impressive-looking item that is not independent of what was counted
        if overlap:
            cand = [it for it in items if it["kind"] == forced_up]
            if cand:
                it = cand[0]
                it["independent"] = False
                it["overlap"] = {"segregation": "the newly reported family is the original proband's family, "
                                                "re-ascertained through a cousin at another centre",
                                 "probands": "one of the published probands is our own patient, reported by the "
                                             "referring hospital",
                                 "func_val": "this is the peer-reviewed version of the preprint already counted; "
                                             "same experiments, and still no control variants were run",
                                 "func_unval": "the result is the same preprint already counted in the original "
                                               "review"}[it["kind"]]
        delta = 0
        for it in items:
            if it["independent"]:
                delta += it["pts"]
            elif it["kind"] == "probands":
                tot_p = orig_probands + it["new_probands"] - 1
                tier = 4 if tot_p >= 6 else 2 if tot_p >= 3 else 1 if tot_p >= 1 else 0
                it["pts_if_overlap"] = (tier - (1 if "PS4_Supporting" in orig_names else 0)
                                        if "PM2_Supporting" in orig_names else 0)
                delta += it["pts_if_overlap"]
        new_total = sum(p for _, p in orig) + delta

        # pending evidence that could still move the call
        pending = pick_w(rng, {None: 4, "relatives": 3, "parents": 1.5, "assay": 2})
        if pending == "parents" and g["inh"] != "AD":
            pending = "relatives"
        reason = pick_w(rng, {"routine": 2, "cascade": 2.5, "surgery": 2, "device": 1, "prenatal": 1,
                              "screening": 1.5})
        if reason == "surgery" and g["cat"] not in ("cancer",):
            reason = pick(rng, ["cascade", "screening", "routine"])
        if reason == "device" and g["cat"] != "cardiac":
            reason = pick(rng, ["cascade", "routine"])
        days = {"routine": None, "cascade": None, "screening": None,
                "surgery": pick(rng, [12, 18, 25, 40, 60, 90]), "device": pick(rng, [10, 21, 35, 70]),
                "prenatal": pick(rng, [5, 9, 14, 20])}[reason]
        requester_view = pick_w(rng, {"wants_upgrade": 3, "neutral": 4, "wants_closure": 1.5})
        orig_date = rand_date(rng)
        return {
            "regime": reg, "gene": g["sym"], "orig_codes": [c for c, _ in orig], "orig_points": [p for _, p in orig],
            "orig_total": sum(p for _, p in orig), "orig_probands": orig_probands, "orig_meioses": orig_meioses,
            "items": items, "new_total": new_total, "pending": pending, "reason": reason,
            "days_to_decision": days, "requester_view": requester_view,
            "vus_used_for_care": chance(rng, 0.6),
            "hgvs": " ".join(_variant(rng, g, vtype)), "vtype": vtype,
            "orig_report_date": orig_date, "orig_report_year": orig_date.year - rng.randint(1, 6),
            "requester": person(rng), "requester_role": pick(rng, ["genetic counsellor", "clinical geneticist",
                                                                   "cardiologist" if g["cat"] == "cardiac" else
                                                                   "oncologist" if g["cat"] == "cancer" else
                                                                   "specialist physician"]),
            "scientist": person(rng), "case_id": ident(rng, "RR", 5, "-"), "report_id": ident(rng, "GR", 7, ""),
            "lab_name": _lab(rng), "pub_year": rng.randint(2023, 2026),
            "assay": pick(rng, VAL_STUDIES), "unval_assay": pick(rng, UNVAL_STUDIES), "controls": rng.randint(9, 45),
            "centre": f"{pick(rng, LAB_STEMS)} {pick(rng, ['University Hospital', 'Children’s Hospital', 'Medical Centre', 'Clinical Genetics Service'])}",
        }

    def _item_text(self, f, g, it):
        k = it["kind"]
        if k == "func_val" and not it["independent"]:
            s = f"journal article ({f['pub_year']}) on the variant's protein function: loss of function"
        elif k == "func_val":
            s = (f"{f['assay']} ({f['pub_year']}; {f['controls']} controls run): "
                 + ("function lost" if it["result"] == "damaging" else "function normal"))
        elif k == "func_unval":
            s = f"{f['unval_assay']}: " + ("lower function" if it["result"] == "damaging" else "no effect seen")
        elif k == "segregation":
            if it["result"] == "non-segregation":
                s = "an affected relative tested at another centre does NOT carry the variant"
            else:
                s = (f"further relatives tested; the variant tracks with disease, informative meioses now "
                     f"{it['meioses_total']} in total (originally {f['orig_meioses'] or 'none'})")
        elif k == "probands":
            s = (f"case series ({f['pub_year']}) reports {it['new_probands']} more unrelated proband"
                 f"{'s' if it['new_probands'] > 1 else ''} with {g['disorder']} carrying the variant")
        elif k == "population":
            s = (f"popDB v5 release: {it['ac']} of {it['an']:,} alleles"
                 + (f" (AF {_fmt_af(it['ac'] / it['an'])})" if it["ac"] else " (absent)"))
        elif k == "de_novo":
            s = (f"seen de novo in an unrelated affected child at {f['centre']}; parentage "
                 + ("confirmed" if it["result"] == "confirmed" else "not confirmed"))
        else:
            s = "identified in trans with a known pathogenic variant in an affected person (phased)"
        if not it["independent"]:
            s += f". Note: {it['overlap']}"
        return s

    def _pending_text(self, f):
        return {None: None,
                "relatives": "two more affected relatives have agreed to testing (up to 2 further informative meioses)",
                "parents": "parental samples for the de novo observation have been requested",
                "assay": "a validated functional study at a collaborating lab is expected in about 8 weeks"}[f["pending"]]

    def _sop(self, g, rng):
        return [*_pc(rng, "vus"),
                f"{g['sym']}: maximum credible AF {_fmt_af(g['max_af'])}; rarity at or below {_fmt_af(g['rare'])}.",
                "Re-review: new items are added only if independent of evidence already counted; a stronger "
                "version of an existing code replaces it (never adds to it).",
                "Validated assay = strong; unvalidated = supporting. Segregation 7+ meioses strong, 5-6 moderate, "
                "3-4 supporting; non-segregation BS4. Probands count only for rare variants: strong at six or "
                "more, moderate at three to five, supporting at one or two. De novo confirmed strong, unconfirmed "
                "moderate.",
                "Amended report and recontact of the ordering clinician whenever the class moves to likely "
                "pathogenic or pathogenic; for a move to likely benign or benign, recontact only if the VUS was "
                "being used to guide care, otherwise the update goes out with the next routine report."]

    def render(self, f, rng, surface):
        g = next(x for x in GENES if x["sym"] == f["gene"])
        orig = ", ".join(f"{c} ({p:+d})" for c, p in zip(f["orig_codes"], f["orig_points"]))
        pop0 = ("population data at the original review: absent from popDB v4 (rare)" if "PM2_Supporting" in f["orig_codes"]
                else "population data at the original review: popDB v4 frequency above the rarity cut-off, below the "
                     "maximum credible AF")
        new = [self._item_text(f, g, it) for it in f["items"]]
        why, clock = self.REASONS[f["reason"]]
        deadline = (None if f["days_to_decision"] is None else
                    f"{clock} in {f['days_to_decision']} days" if clock != "pregnancy" else
                    f"prenatal decision needed within {f['days_to_decision']} days")
        view = {"wants_upgrade": "The family has been told this may be upgraded and we would welcome that.",
                "neutral": "Grateful for an updated view either way.",
                "wants_closure": "If it cannot be resolved we would rather close the question for now."}[f["requester_view"]]
        care = ("The VUS was used to justify annual surveillance for the patient." if f["vus_used_for_care"]
                else "The VUS has not been used for any management decision.")
        pend = self._pending_text(f)
        if surface == "json":
            return dumps(prune({
                "original_report": {"report": f["report_id"], "issued": f["orig_report_date"].isoformat(),
                                    "gene": g["sym"], "variant": f["hgvs"], "disorder": g["disorder"],
                                    "codes": orig, "total_points": f["orig_total"], "population": pop0,
                                    "class": "uncertain significance", "care_use": care},
                "request": {"id": f["case_id"], "from": f"{f['requester']} ({f['requester_role']})",
                            "reason": why, "deadline": deadline, "comment": view},
                "new_evidence": new,
                "pending": pend,
                "sop": {"id": f"{_sop_id(rng)} (local)", "rules": self._sop(g, rng)},
            }))
        if surface == "clinician_request":
            body = [f"Subject: re-review request -- {g['sym']} {f['hgvs']} (report {f['report_id']})",
                    f"From: {f['requester']}, {f['requester_role']}", f"To: variant-review@{f['lab_name'].split()[0].lower()}.example", "",
                    "Dear colleagues,", "",
                    noisy(join_sentences([f"Your {fmt_date(f['orig_report_date'], rng)} report classed this variant as a VUS.",
                                          f"I am asking for a re-review because {why}.",
                                          (f"Timing: {deadline}." if deadline else ""), care, view]), rng), "",
                    "New information since the report:", bullets(new, rng),
                    "" if not pend else f"\nAlso: {pend}.", "", "Kind regards,", f["requester"]]
            sop = "\n".join(self._sop(g, rng))
            return ("\n".join(body) + f"\n\n[attached at intake] Original codes: {orig} = {f['orig_total']} pts; "
                    f"{pop0}\n" + sop)
        rows = [["original", c, f"{p:+d}"] for c, p in zip(f["orig_codes"], f["orig_points"])]
        lines = [f"REANALYSIS NOTE {f['case_id']} -- {g['sym']} {f['hgvs']} ({g['disorder']})",
                 f"Requested by {f['requester']} ({f['requester_role']}): {why}"
                 + (f"; {deadline}" if deadline else "") + ".",
                 care, "", table(["evidence", "code", "points"], rows, rng),
                 f"Original total: {f['orig_total']} (VUS); {pop0}", "", "New since report:",
                 "\n".join(f"{i}. {t}" for i, t in enumerate(new, 1))]
        if pend:
            lines += ["", f"Pending: {pend}"]
        lines += ["", f"Requester comment: {view}", "", "SOP (local):", *self._sop(g, rng)]
        return "\n".join(lines)

    def questions(self):
        return {
            "reclassification_outcome": choice("What should come of this re-review?", {
                "downgrade_to_benign_side": "Reclassify the variant to likely benign or benign.",
                "keep_vus_no_change": "Keep it as uncertain significance; nothing obtainable soon would settle it.",
                "keep_vus_request_data": "Keep it as uncertain significance for now and ask for the specific "
                                         "pending data that could move it across a boundary.",
                "upgrade_to_pathogenic_side": "Reclassify the variant to likely pathogenic or pathogenic.",
            }),
            "new_evidence_quality": score("How much does the new evidence add to what was already known?", [
                "Little: it repeats data already counted or comes from an unvalidated source.",
                "Modest: one new item of supporting weight.",
                "Solid: at least one robust, independent item.",
                "Strong: several robust items that are independent of each other and of the original review.",
            ]),
            "decision_timeliness": score("How soon does the lab need to conclude this re-review?", [
                "At the next routine review cycle.",
                "Within a few weeks, so relatives can decide about testing.",
                "Before a planned clinical decision some weeks away.",
                "Within days, because a time-limited decision depends on it.",
            ]),
            "evidence_is_independent": noul(
                "All of the new evidence is independent of the observations counted in the original review.",
                yes="Every new item comes from different people, families or experiments than those already scored.",
                no="At least one new item re-reports a family, patient or experiment that was already counted."),
            "patient_recontact_warranted": noul(
                "The ordering clinician should be recontacted with an amended report now.",
                yes="The class changes in a way the SOP says must be communicated immediately.",
                no="The class stays the same, or the change can wait for the next routine report."),
        }

    def answer_key(self, f):
        """Rule-of-thumb answer key for the spread check only; never used as a label."""
        t = f["new_total"]
        if t >= 6:
            out = "upgrade_to_pathogenic_side"
        elif t <= -1:
            out = "downgrade_to_benign_side"
        else:
            reach = {None: 0, "relatives": 1, "parents": 2, "assay": 4}[f["pending"]]
            if f["pending"] == "assay":
                close = t + 4 >= 6 or t - 4 <= -1
            else:
                close = t + reach >= 6
            out = "keep_vus_request_data" if reach and close else "keep_vus_no_change"
        robust = [it for it in f["items"] if it["independent"] and abs(it["pts"]) >= 2]
        weak = [it for it in f["items"] if it["independent"] and 0 < abs(it["pts"]) < 2]
        if len(robust) >= 2:
            q = 3
        elif robust:
            q = 2
        elif weak:
            q = 1
        else:
            q = 0
        d = f["days_to_decision"]
        if d is None:
            timely = 1 if f["reason"] in ("cascade", "screening") else 0
        else:
            timely = 3 if d <= 14 else 2
        indep = all(it["independent"] for it in f["items"])
        recontact = out == "upgrade_to_pathogenic_side" or (out == "downgrade_to_benign_side" and f["vus_used_for_care"])
        return {"reclassification_outcome": out, "new_evidence_quality": q, "decision_timeliness": timely,
                "evidence_is_independent": "yes" if indep else "no",
                "patient_recontact_warranted": "yes" if recontact else "no"}


# =========================================================================== schema 3
INDICATIONS = ["trio exome for early-onset epilepsy", "exome for global developmental delay",
               "exome for short stature with skeletal dysplasia", "exome for congenital hearing loss",
               "genome for suspected primary immunodeficiency", "exome for progressive ataxia",
               "exome for congenital kidney anomalies", "trio genome for multiple congenital anomalies",
               "exome for unexplained intellectual disability", "exome for inherited retinal dystrophy",
               "exome for suspected mitochondrial disease", "genome for neuromuscular weakness",
               "exome for recurrent pancreatitis", "exome for early-onset cataract",
               "trio exome for autism with seizures", "exome for familial hearing loss"]
SF_CATEGORIES = {"cancer": "cancer predisposition", "cardiac": "inherited heart and vascular conditions",
                 "aortopathy": "inherited heart and vascular conditions", "lipid": "inherited lipid disorders",
                 "anaesthesia": "anaesthetic reaction risk"}


class SecondaryFindingReview(Schema):
    name = "secondary_finding_review"
    description = ("A variant found outside the reason for an exome or genome test, checked against the lab's "
                   "secondary-findings list, the patient's recorded choices and the call's quality.")
    surfaces = {"json": 0.5, "review_sheet": 0.25, "results_meeting_chat": 0.25}

    REGION = {"none": None, "pseudogene": "segmental duplication with a close pseudogene",
              "homopolymer": "next to a 9-base homopolymer run", "low_map": "low-mappability region"}

    def sample_factors(self, rng):
        reg = regime(rng)
        on_list = chance(rng, 0.8)
        g = pick(rng, SF_GENES if on_list else OFF_SF_GENES)
        cls = pick_w(rng, {"pathogenic": 4, "likely_pathogenic": 4, "uncertain_significance": 1.3})
        if not on_list:
            cls = pick(rng, ["pathogenic", "likely_pathogenic"])
        reduced_pen = g["cat"] in ("cancer", "cardiac", "lipid") and chance(rng, 0.18)
        consent = pick_w(rng, {"all": 3.0, "selected": 3, "opted_out": 1.6, "ambiguous": 1.5})
        selected = None
        if consent == "selected":
            cats = sorted(set(SF_CATEGORIES.values()))
            selected = sorted(rng.sample(cats, rng.randint(1, 3)))
        # call quality
        depth = rng.randint(55, 210)
        vaf = round(rng.uniform(0.4, 0.58), 2)
        region = "none"
        strand_bias = False
        conf = pick_w(rng, {"confirmed": 1.5, "pending": 4, "not_done": 2.5})
        if reg == "clear":
            if chance(rng, 0.5):
                cls, consent, selected, conf = pick(rng, ["pathogenic", "likely_pathogenic"]), "all", None, "confirmed"
            else:
                depth, vaf = rng.randint(9, 24), round(rng.uniform(0.09, 0.2), 2)
                region = pick(rng, ["pseudogene", "low_map"])
                conf = pick(rng, ["pending", "not_done"])
        else:
            if chance(rng, 0.6):
                which = pick(rng, ["depth", "vaf", "region", "strand"])
                if which == "depth":
                    depth = near(rng, 30, 8, lo=8, integer=True)
                elif which == "vaf":
                    vaf = round(pick(rng, [near(rng, 0.35, 0.06), near(rng, 0.65, 0.05), rng.uniform(0.12, 0.3)]), 2)
                elif which == "region":
                    region = pick(rng, ["pseudogene", "homopolymer", "low_map"])
                else:
                    strand_bias = True
            if reg == "conflicting":
                # a high-value finding with a shaky call, or a clean call the patient's choices may not cover
                if chance(rng, 0.5):
                    cls, conf = "pathogenic", pick(rng, ["pending", "not_done"])
                    vaf = round(rng.uniform(0.14, 0.3), 2)
                    region = pick(rng, ["pseudogene", "homopolymer", "none"])
                else:
                    consent = pick(rng, ["ambiguous", "selected", "opted_out"])
                    if consent == "selected":
                        cats = sorted(set(SF_CATEGORIES.values()))
                        selected = sorted(rng.sample(cats, rng.randint(1, 2)))
        family_history = pick_w(rng, {"none_known": 4, "suggestive": 2, "not_asked": 1})
        return {"regime": reg, "gene": g["sym"], "on_list": on_list, "class": cls, "reduced_penetrance": reduced_pen,
                "consent": consent, "selected_categories": selected, "depth": depth, "vaf": vaf,
                "region": region, "strand_bias": strand_bias, "confirmation": conf,
                "family_history": family_history, "indication": pick(rng, INDICATIONS),
                "hgvs": " ".join(_variant(rng, g, pick(rng, ["nonsense", "frameshift", "missense"])
                                          if g["mech"] == "lof" else "missense")),
                "case_id": ident(rng, "SF", 5, "-"), "patient_id": ident(rng, "GX", 6, ""),
                "analyst": person(rng), "director": person(rng), "counsellor": person(rng),
                "consent_date": rand_date(rng), "sop_id": _sop_id(rng),
                "ambiguity": pick(rng, ["the secondary-findings box is unticked but the free-text line reads "
                                        "'yes to anything useful'",
                                        "two consent forms on file disagree: the newer one opts out, the older "
                                        "one opts in",
                                        "the form was signed by a relative and the patient's own choice is not "
                                        "recorded",
                                        "the opt-in box is ticked but 'not cancer' is handwritten beside it",
                                        "the interpreter's note says the choice was not discussed"])}

    def _category(self, g):
        return SF_CATEGORIES.get(g["cat"])

    def _covers(self, f, g):
        if f["consent"] == "all":
            return True
        if f["consent"] == "selected":
            return self._category(g) in (f["selected_categories"] or [])
        return False

    def _needs_confirmation(self, f):
        return (f["depth"] < 30 or not 0.35 <= f["vaf"] <= 0.65 or f["region"] != "none" or f["strand_bias"])

    def _consent_text(self, f):
        return {"all": "opted in to all secondary findings",
                "selected": "opted in to selected categories only: " + ", ".join(f["selected_categories"] or []),
                "opted_out": "opted out of secondary findings",
                "ambiguous": f"unclear: {f['ambiguity']}"}[f["consent"]]

    SOP = ["Secondary findings: report only pathogenic or likely pathogenic variants in genes on the lab's "
           "secondary-findings list, and only in categories the patient opted into; never report a VUS as a "
           "secondary finding.",
           "Before reporting, confirm by a second method if depth is below 30x, allele fraction is outside "
           "0.35-0.65, the region is flagged, or strand bias is present.",
           "If the consent record does not show a clear choice, refer to the results review committee; do not "
           "infer a choice."]

    def render(self, f, rng, surface):
        g = next(x for x in GENES if x["sym"] == f["gene"])
        listed = (f"on the SF list ({self._category(g)})" if f["on_list"] else "not on the SF list")
        reg_txt = self.REGION[f["region"]]
        conf = {"confirmed": "confirmed by Sanger", "pending": "second-method confirmation requested",
                "not_done": "no confirmation yet"}[f["confirmation"]]
        pen = ("this allele is a recognised reduced-penetrance allele" if f["reduced_penetrance"] else None)
        fam = {"none_known": "no relevant family history recorded",
               "suggestive": f"family history mentions relatives with {g['disorder']}",
               "not_asked": None}[f["family_history"]]
        if surface == "json":
            return dumps(prune({
                "finding": {"case": f["case_id"], "gene": g["sym"], "variant": f["hgvs"], "zygosity": "heterozygous",
                            "condition": g["disorder"], "classification": CLASS_TEXT[f["class"]],
                            "list_status": listed, "penetrance_note": pen},
                "call_quality": {"depth": f["depth"], "allele_fraction": f["vaf"], "region_flag": reg_txt,
                                 "strand_bias": f["strand_bias"], "confirmation": conf},
                "patient": {"id": f["patient_id"], "test": f["indication"], "secondary_findings_choice":
                            self._consent_text(f), "choice_recorded": f["consent_date"].isoformat(),
                            "family_history": fam},
                "sop": {"id": f"{f['sop_id']} (local)", "rules": self.SOP},
            }))
        if surface == "review_sheet":
            rows = [("Case", f["case_id"]), ("Patient", f["patient_id"]), ("Primary test", f["indication"]),
                    ("Finding", f"{g['sym']} {f['hgvs']} (het)"), ("Condition", g["disorder"]),
                    ("Classification", CLASS_TEXT[f["class"]]), ("SF list", listed),
                    ("Depth / AF", f"{f['depth']}x / {f['vaf']:.2f}"),
                    ("Region", reg_txt or "unflagged"), ("Strand bias", "yes" if f["strand_bias"] else "no"),
                    ("Confirmation", conf), ("Patient choice", self._consent_text(f)),
                    ("Choice recorded", fmt_date(f["consent_date"], rng)), ("Penetrance", pen),
                    ("Family history", show(fam, rng))]
            return ("SECONDARY FINDING REVIEW SHEET\n" + kv_lines(rows, rng) + f"\n\nSOP {f['sop_id']} (local):\n"
                    + "\n".join(self.SOP))
        msgs = [(f["analyst"], f"{g['sym']} {f['hgvs']} came up in {f['patient_id']} ({f['indication']}), "
                               f"classified {CLASS_TEXT[f['class']]}, {listed}"),
                (f["analyst"], f"{f['depth']}x, AF {f['vaf']:.2f}" + (f", {reg_txt}" if reg_txt else "")
                 + (", strand bias flagged" if f["strand_bias"] else "") + f"; {conf}"),
                (f["counsellor"], f"consent form: {self._consent_text(f)}")]
        if pen:
            msgs.append((f["director"], f"note {pen}"))
        if fam:
            msgs.append((f["counsellor"], fam))
        msgs.append((f["director"], pick(rng, ["what does the SOP say we do here?", "thoughts before I sign?",
                                               "let's settle this one today", "can we close this out?"])))
        out = [f"#results-review ({f['case_id']})"]
        hh = rng.randint(8, 16)
        mm = rng.randint(0, 40)
        for who, text in msgs:
            mm += rng.randint(1, 6)
            out.append(f"[{hh:02d}:{mm % 60:02d}] {who.split()[0].lower()}: {noisy(text, rng, 0.08)}")
        return "\n".join(out) + f"\n\n(pinned) SOP {f['sop_id']} (local): " + " ".join(self.SOP)

    def questions(self):
        return {
            "secondary_finding_handling": choice("How should the lab handle this finding?", {
                "confirm_then_report": "Report it as a secondary finding only after a second method confirms the call.",
                "do_not_report": "Leave it off the report; it is not a reportable secondary finding for this patient.",
                "refer_to_review_committee": "Send it to the results review committee because the patient's "
                                             "recorded choice is unclear.",
                "report_secondary_finding": "Report it now as a secondary finding.",
            }),
            "call_confidence": score("How confident can the lab be that the variant call is real and germline?", [
                "Low: the call looks like an artefact or a low-level mosaic.",
                "Guarded: one or more quality flags need resolving.",
                "Reasonable: a minor flag, otherwise clean.",
                "High: clean metrics or already confirmed by a second method.",
            ]),
            "actionability": score("How much could knowing this result change the patient's care?", [
                "Not at all: an uncertain result or a gene without established management.",
                "A little: some management exists, but it is limited or unproven.",
                "Moderately: surveillance or precautions with real but partial benefit.",
                "Greatly: proven prevention or early-treatment steps for a high-penetrance condition.",
            ]),
            "consent_covers_finding": noul(
                "The patient's recorded choice covers receiving this kind of finding.",
                yes="The patient opted in to all secondary findings, or to a category that includes this gene.",
                no="The patient opted out, chose categories that exclude it, or the record shows no clear choice."),
            "artefact_or_mosaic_suspected": noul(
                "The call shows signs of being a sequencing artefact or a mosaic rather than a germline "
                "heterozygous variant."),
        }

    def answer_key(self, f):
        """Rule-of-thumb answer key for the spread check only; never used as a label."""
        g = next(x for x in GENES if x["sym"] == f["gene"])
        reportable = f["on_list"] and f["class"] != "uncertain_significance"
        need_conf = self._needs_confirmation(f)
        if not reportable:
            h = "do_not_report"
        elif f["consent"] == "ambiguous":
            h = "refer_to_review_committee"
        elif not self._covers(f, g):
            h = "do_not_report"
        elif need_conf and f["confirmation"] != "confirmed":
            h = "confirm_then_report"
        else:
            h = "report_secondary_finding"
        if f["confirmation"] == "confirmed":
            cc = 3
        else:
            pen = (f["depth"] < 30) + (2 if f["vaf"] < 0.2 else 1 if not 0.35 <= f["vaf"] <= 0.65 else 0) \
                + (f["region"] != "none") + f["strand_bias"]
            cc = max(0, 3 - pen)
        if f["class"] == "uncertain_significance":
            act = 0
        elif not f["on_list"]:
            act = 0 if g["cat"] == "recessive" else 1
        elif f["reduced_penetrance"] or g["cat"] in ("lipid", "anaesthesia"):
            act = 2
        else:
            act = 3
        art = f["vaf"] < 0.3 or (f["region"] != "none" and f["confirmation"] != "confirmed") or f["strand_bias"]
        return {"secondary_finding_handling": h, "call_confidence": cc, "actionability": act,
                "consent_covers_finding": "yes" if self._covers(f, g) else "no",
                "artefact_or_mosaic_suspected": "yes" if art else "no"}


# =========================================================================== schema 4
# (gene, band, chromosome) for established haploinsufficient genes on the lab's dosage list
HI_GENES = [("NSD1", "5q35.3", 5), ("EHMT1", "9q34.3", 9), ("SHANK3", "22q13.33", 22), ("MBD5", "2q23.1", 2),
            ("SATB2", "2q33.1", 2), ("ZEB2", "2q22.3", 2), ("TCF4", "18q21.2", 18), ("SYNGAP1", "6p21.32", 6),
            ("ANKRD11", "16q24.3", 16), ("CHD7", "8q12.2", 8), ("RAI1", "17p11.2", 17), ("NRXN1", "2p16.3", 2),
            ("FOXP1", "3p13", 3), ("ARID1B", "6q25.3", 6), ("DYRK1A", "21q22.13", 21), ("MEF2C", "5q14.3", 5),
            ("PAFAH1B1", "17p13.3", 17), ("KMT2D", "12q13.12", 12), ("SOX5", "12p12.1", 12),
            ("CTNNB1", "3p22.1", 3), ("SETBP1", "18q12.3", 18), ("NFIX", "19p13.13", 19)]
# (recurrent region, chromosome, typical size kb)
RECURRENT = [("1q21.1 distal", 1, 1350), ("15q13.3 (BP4-BP5)", 15, 1600), ("16p11.2 proximal", 16, 600),
             ("16p13.11", 16, 1500), ("17q12", 17, 1400), ("22q11.2 proximal (A-D)", 22, 2600),
             ("7q11.23", 7, 1550), ("3q29", 3, 1600), ("15q11.2 (BP1-BP2)", 15, 500), ("2q13", 2, 1700),
             ("17q21.31", 17, 550), ("1q21.1 proximal (TAR)", 1, 450), ("17p11.2", 17, 3700),
             ("16p11.2 distal", 16, 220), ("8p23.1", 8, 3800), ("10q11.22-q11.23", 10, 7000)]
# real cytobands for CNVs away from the listed genes and regions; approximate chromosome lengths (Mb)
BANDS = ["1p36.22", "1q41", "2p25.3", "2q37.3", "3p26.3", "3q13.31", "4p16.3", "4q28.3", "5p15.33", "5q31.3",
         "6p22.3", "6q16.1", "7p21.1", "7q31.1", "8p23.3", "8q24.21", "9p24.3", "9q22.33", "10p15.3", "10q26.3",
         "11p15.5", "11q23.3", "12p13.31", "12q21.31", "13q31.3", "14q32.33", "15q26.3", "16p13.3", "16q23.1",
         "17p13.1", "17q25.3", "18p11.32", "18q22.3", "19p13.3", "19q13.42", "20p12.1", "20q13.33", "21q22.3",
         "22q12.3", "22q13.31"]
CHROM_MB = {1: 248, 2: 242, 3: 198, 4: 190, 5: 181, 6: 171, 7: 159, 8: 145, 9: 138, 10: 134, 11: 135, 12: 133,
            13: 114, 14: 107, 15: 102, 16: 90, 17: 83, 18: 80, 19: 59, 20: 64, 21: 47, 22: 51}
PHENOTYPES = ["global developmental delay", "seizures and hypotonia", "autism and speech delay",
              "congenital heart defect with dysmorphic features", "short stature and microcephaly",
              "intellectual disability", "multiple congenital anomalies", "growth restriction and feeding difficulty",
              "macrocephaly and overgrowth", "behavioural difficulties and learning disability"]


def _cnv_class(pts):
    """pts are hundredths of a point."""
    if pts >= 99:
        return "pathogenic"
    if pts >= 90:
        return "likely_pathogenic"
    if pts > -90:
        return "uncertain_significance"
    if pts > -99:
        return "likely_benign"
    return "benign"


class CnvClassification(Schema):
    name = "cnv_classification"
    description = ("A copy-number variant from a chromosomal microarray or genome, scored on the lab's CNV points "
                   "sheet before it goes on a report.")
    surfaces = {"json": 0.5, "cnv_worksheet": 0.25, "draft_report": 0.25}

    SHEET = ["Gene content (losses): complete deletion of a dosage-list haploinsufficient gene +1.00; partial "
             "deletion including its 5' coding start +0.90; partial deletion of its 3' end only +0.45; covers a "
             "recurrent syndrome region in full +1.00 (use the single highest of these). Otherwise 25-34 "
             "protein-coding genes +0.45, 35 or more +0.90, 0-24 genes 0.",
             "Gene content (gains): covers a recurrent duplication region in full +1.00; both breakpoints inside "
             "one dosage-list gene +0.45; 50 or more protein-coding genes +0.45; otherwise 0.",
             "No protein-coding genes at all: -0.60.",
             "Population controls: 90% or more of the CNV lies in CNVs seen in over 1% of controls -1.00; 50-89% "
             "-0.30.",
             "Inheritance: de novo +0.45; inherited from an affected parent +0.30; inherited from an unaffected "
             "parent -0.45 (losses) or -0.30 (gains); not tested 0.",
             "Phenotype highly specific for the gene or region: +0.15.",
             "Classes: 0.99 or more pathogenic; 0.90 to 0.98 likely pathogenic; -0.89 to 0.89 uncertain; -0.98 to "
             "-0.90 likely benign; -0.99 or less benign.",
             "Second method (qPCR or MLPA) before a reportable call (P, LP or uncertain) if fewer than 20 probes "
             "or bins support it, the log2 ratio suggests mosaicism, or the CNV is intragenic."]

    # kb per probe (arrays) or per informative bin/exon target (sequencing callers)
    DENSITY = {"CGH array (180k)": 14.0, "SNP array (750k)": 4.5, "SNP array (2.6M)": 1.4,
               "exome read-depth caller": 22.0, "genome read-depth caller (1 kb bins)": 1.0}
    # borderline set-ups that land in the narrow likely bands: (kind, content, population, inheritance)
    BANDS = [("loss", "hi_5prime", "none", "not_tested"), ("loss", "hi_3prime", "none", "not_tested"),
             ("loss", "hi_3prime", "none", "de_novo"), ("loss", "genes25", "none", "not_tested"),
             ("loss", "genes35", "none", "not_tested"), ("gain", "intragenic", "none", "de_novo"),
             ("loss", "no_genes", "partial", "not_tested"), ("gain", "no_genes", "none", "inherited_unaffected"),
             ("loss", "genes25", "none", "de_novo"), ("loss", "recurrent", "none", "not_tested")]

    def _place(self, rng, content, kind, hi, rec, reg):
        """chromosome, band, size (kb) and protein-coding gene count consistent with the gene content."""
        if content in ("hi_complete", "hi_5prime", "hi_3prime"):
            size = pick(rng, [rng.randint(18, 59), rng.randint(60, 300), rng.randint(60, 900)])
            return hi[2], hi[1], size, (rng.randint(1, 6) if size > 120 else 1)
        if content == "intragenic":
            return hi[2], hi[1], rng.randint(12, 140), 1
        if content == "recurrent":
            return rec[1], rec[0], int(rec[2] * rng.uniform(0.95, 1.1)), rng.randint(8, 40)
        band = pick(rng, BANDS)
        chrom = int(band.split("p")[0].split("q")[0])
        if content == "no_genes":
            return chrom, band, rng.randint(30, 600), 0
        n = pick(rng, [rng.randint(2, 20), rng.randint(21, 29), rng.randint(30, 40), rng.randint(41, 70)])
        if reg == "borderline":
            n = pick(rng, [near(rng, 25, 3, lo=18, integer=True), near(rng, 35, 3, integer=True),
                           near(rng, 50, 4, integer=True)])
        return chrom, band, max(60, int(n * rng.uniform(25, 70))), n

    def sample_factors(self, rng):
        reg = regime(rng)
        kind = pick_w(rng, {"loss": 6.5, "gain": 3.5})
        if kind == "loss":
            content = pick_w(rng, {"hi_complete": 1.6, "hi_5prime": 1.4, "hi_3prime": 1.4, "recurrent": 1.4,
                                   "genes": 2.4, "no_genes": 1.6})
        else:
            content = pick_w(rng, {"recurrent": 2, "intragenic": 2, "genes": 3, "no_genes": 1.7})
        hi = pick(rng, HI_GENES)
        rec = pick(rng, RECURRENT)
        pop = pick_w(rng, {"none": 5, "partial": 2.6 if content in ("genes", "no_genes") else 1.6,
                           "full": 1.6 if content in ("genes", "no_genes") else 0.4})
        inh = pick_w(rng, {"de_novo": 2.0, "inherited_affected": 1, "inherited_unaffected": 1.6, "not_tested": 7})
        n_genes = None
        if reg == "clear":
            if chance(rng, 0.5):
                kind, content, pop = "loss", pick(rng, ["hi_complete", "recurrent"]), "none"
                inh = pick(rng, ["de_novo", "de_novo", "not_tested"])
            else:
                content, pop = "no_genes", "full"
                inh = pick(rng, ["inherited_unaffected", "not_tested"])
        elif reg == "conflicting":
            # strong gene content against an unaffected carrier parent, or heavy overlap with common control CNVs
            if chance(rng, 0.5):
                inh = "inherited_unaffected"
                if content in ("genes", "no_genes"):
                    content = "hi_5prime" if kind == "loss" else "recurrent"
            else:
                pop = pick(rng, ["partial", "full"])
            if chance(rng, 0.4):
                inh = "not_tested"
        elif chance(rng, 0.45):
            # borderline set-ups that land in or next to the narrow likely bands
            kind, content, pop, inh = pick(rng, self.BANDS)
            if content in ("genes35", "genes25"):
                n_genes = rng.randint(35, 44) if content == "genes35" else rng.randint(25, 34)
                content = "genes"
            if content == "recurrent":
                pop = "partial"
        chrom, band, size, n = self._place(rng, content, kind, hi, rec, reg)
        if n_genes is not None:
            size = max(size, int(n_genes * rng.uniform(25, 70)))
        n_genes = n if n_genes is None else n_genes
        specific = content in ("hi_complete", "hi_5prime", "recurrent") and reg != "borderline" and chance(rng, 0.35)
        mosaic = chance(rng, 0.1 if reg == "clear" else 0.22)
        gap = pick_w(rng, {"nothing": 3.2, "genes": 3.2,
                           "hi_5prime": 4 if content in ("genes", "hi_3prime", "no_genes") and kind == "loss" else 0})
        if content == "intragenic":
            gap = "nothing"                       # both breakpoints sit inside one gene
        extra_genes = rng.randint(2, 12) if gap == "genes" else 0
        if gap == "genes" and reg == "borderline" and content == "genes":
            n_genes = pick(rng, [rng.randint(20, 24), rng.randint(30, 34), rng.randint(44, 49)])
        gap_kb = pick(rng, [5, 12, 30, 80, 150, 260])
        if gap != "nothing" and gap_kb < 30:
            gap_kb = pick(rng, [60, 120, 210, 340])
        platform = pick_w(rng, dict(zip(self.DENSITY, [3, 2, 1, 2.5, 1.5])))
        probes = max(3, int(round(size / self.DENSITY[platform] * rng.uniform(0.8, 1.2))))
        log2 = (round(rng.uniform(-0.45, -0.25), 2) if kind == "loss" else round(rng.uniform(0.18, 0.3), 2)) if mosaic \
            else (round(rng.uniform(-1.1, -0.75), 2) if kind == "loss" else round(rng.uniform(0.42, 0.62), 2))
        lo = {13: 19, 14: 18, 15: 20, 21: 13, 22: 17}.get(chrom, 1)
        start = rng.randint(lo * 1_000_000, (CHROM_MB[chrom] - 6) * 1_000_000)
        view = pick(rng, [None, None, "looks pathogenic to me", "I'd hold this as uncertain", "parents would settle it"]
                    + (["probably a benign polymorphism"] if inh != "de_novo" else []))
        return {"regime": reg, "kind": kind, "content": content, "hi_gene": hi[0], "region": rec[0],
                "chrom": chrom, "band": band, "size_kb": size, "start": start, "n_genes": n_genes,
                "pop_overlap": pop, "inheritance": inh, "phenotype_specific": specific, "probes": probes,
                "mosaic": mosaic, "log2": log2, "gap_kb": gap_kb, "gap_content": gap, "gap_extra_genes": extra_genes,
                "phenotype": pick(rng, PHENOTYPES), "case_id": ident(rng, "CN", 6, "-"),
                "patient_id": ident(rng, "GX", 6, ""), "scientist": person(rng), "platform": platform,
                "scientist_view": view,
                "sop_id": f"CNV-{rng.randint(2, 19)} rev {rng.randint(2, 9)}"}

    # ------------------------------------------------------------------ the points sheet
    def _points(self, f, alt=False):
        """Hundredths. alt=True scores the CNV with its uncertain flank included."""
        content, n = f["content"], f["n_genes"]
        if alt and f["gap_content"] == "hi_5prime" and f["kind"] == "loss":
            content = "hi_5prime" if content in ("genes", "hi_3prime", "no_genes") else content
        if alt and f["gap_content"] == "genes":
            n = n + f["gap_extra_genes"]
            if content == "no_genes":
                content = "genes"
        if f["kind"] == "loss":
            gc = {"hi_complete": 100, "hi_5prime": 90, "hi_3prime": 45, "recurrent": 100}.get(content)
            if gc is None:
                gc = -60 if content == "no_genes" else (90 if n >= 35 else 45 if n >= 25 else 0)
                if content == "hi_3prime" and n >= 35:
                    gc = 90
        else:
            gc = {"recurrent": 100, "intragenic": 45}.get(content)
            if gc is None:
                gc = -60 if content == "no_genes" else (45 if n >= 50 else 0)
        pop = {"none": 0, "partial": -30, "full": -100}[f["pop_overlap"]]
        inh = {"de_novo": 45, "inherited_affected": 30, "not_tested": 0,
               "inherited_unaffected": -45 if f["kind"] == "loss" else -30}[f["inheritance"]]
        spec = 15 if f["phenotype_specific"] else 0
        return gc + pop + inh + spec, gc

    def _content_text(self, f):
        c = f["content"]
        if c == "hi_complete":
            return f"whole of {f['hi_gene']} (dosage list) plus {f['n_genes'] - 1} other genes" if f["n_genes"] > 1 \
                else f"whole of {f['hi_gene']} (dosage list)"
        if c == "hi_5prime":
            return f"5' part of {f['hi_gene']} (dosage list), including the coding start; {f['n_genes']} genes touched"
        if c == "hi_3prime":
            return f"3' exons of {f['hi_gene']} (dosage list) only; coding start intact; {f['n_genes']} genes touched"
        if c == "intragenic":
            return f"both breakpoints inside {f['hi_gene']} (dosage list), exons partly duplicated"
        if c == "recurrent":
            return f"spans the {f['region']} recurrent region in full ({f['n_genes']} genes)"
        if c == "genes":
            return f"{f['n_genes']} protein-coding genes, none on the dosage list"
        return "no protein-coding genes"

    def _gap_text(self, f):
        if f["gap_content"] == "nothing":
            return f"flanking probe gaps {f['gap_kb']} kb, gene-free"
        if f["gap_content"] == "genes":
            return f"distal breakpoint in a {f['gap_kb']} kb probe gap containing {f['gap_extra_genes']} more genes"
        return f"distal breakpoint in a {f['gap_kb']} kb probe gap containing the 5' end of {f['hi_gene']}"

    def _inh_text(self, f):
        return {"de_novo": "de novo (both parents tested)", "inherited_affected": "inherited from a parent with a "
                "similar phenotype", "inherited_unaffected": "inherited from a clinically unaffected parent",
                "not_tested": "parents not tested"}[f["inheritance"]]

    def _pop_text(self, f):
        return {"none": "no overlap with common control CNVs", "partial": "about 60% of the interval overlaps CNVs "
                "seen in more than 1% of controls", "full": "lies almost entirely within CNVs seen in more than 1% "
                "of controls"}[f["pop_overlap"]]

    def render(self, f, rng, surface):
        end = f["start"] + f["size_kb"] * 1000
        coord = f"chr{f['chrom']}:{f['start']:,}-{end:,}"
        word = "deletion" if f["kind"] == "loss" else "duplication"
        mos = (f"log2 ratio {f['log2']:+.2f} (" + ("intermediate; possible mosaicism" if f["mosaic"] else "as expected")
               + ")")
        if surface == "json":
            return dumps(prune({
                "cnv": {"case": f["case_id"], "type": word, "location": f"{f['band']} {coord}",
                        "size_kb": f["size_kb"], "platform": f["platform"], "probes": f["probes"], "signal": mos,
                        "breakpoints": self._gap_text(f)},
                "content": {"genes": self._content_text(f), "population": self._pop_text(f)},
                "patient": {"id": f["patient_id"], "phenotype": f["phenotype"],
                            "phenotype_specific_for_region": f["phenotype_specific"] or None,
                            "inheritance": self._inh_text(f)},
                "scientist_note": f["scientist_view"],
                "sop": {"id": f"{f['sop_id']} (local)", "points_sheet": self.SHEET},
            }))
        if surface == "cnv_worksheet":
            rows = [["type", word], ["location", f"{f['band']} {coord}"], ["size", f"{f['size_kb']:,} kb"],
                    ["probes", f["probes"]], ["signal", mos], ["breakpoints", self._gap_text(f)],
                    ["genes", self._content_text(f)], ["controls", self._pop_text(f)],
                    ["inheritance", self._inh_text(f)], ["phenotype", f["phenotype"]
                                                         + (" (specific for this region)" if f["phenotype_specific"] else "")]]
            return (f"CNV WORKSHEET {f['case_id']} ({f['platform']})\n" + table(["field", "value"], rows, rng)
                    + (f"\n\nScientist: {f['scientist_view']}" if f["scientist_view"] else "")
                    + f"\n\nPoints sheet {f['sop_id']} (local):\n" + "\n".join(f"- {s}" for s in self.SHEET))
        para = [f"{f['platform']} analysis of {f['patient_id']} (referred for {f['phenotype']}) shows a "
                f"{f['size_kb']:,} kb {word} at {f['band']} ({coord}), supported by {f['probes']} probes, {mos}.",
                f"Content: {self._content_text(f)}.", f"{self._gap_text(f).capitalize()}.",
                f"Controls: {self._pop_text(f)}.", f"Inheritance: {self._inh_text(f)}."]
        if f["phenotype_specific"]:
            para.append("The referral features are characteristic of this gene or region.")
        if f["scientist_view"]:
            para.append(f"Analyst comment: {f['scientist_view']}.")
        return (f"DRAFT CNV REPORT -- {f['case_id']} (not for release)\n\n" + noisy(join_sentences(para), rng, 0.08)
                + f"\n\nPoints sheet {f['sop_id']} (local): " + " ".join(self.SHEET))

    def questions(self):
        return {
            "cnv_call": choice("How should this copy-number variant be classified on the points sheet?", {
                "benign": "Benign: the score reaches the benign threshold.",
                "likely_benign": "Likely benign: the score falls in the narrow likely benign band.",
                "likely_pathogenic": "Likely pathogenic: the score falls in the narrow likely pathogenic band.",
                "pathogenic": "Pathogenic: the score reaches the pathogenic threshold.",
                "uncertain_significance": "Uncertain significance: the score sits between the benign-side and "
                                          "pathogenic-side bands.",
            }),
            "dosage_sensitivity_evidence": score("How strong is the dosage-sensitivity case from the genes or region involved?", [
                "None: no dosage-sensitive gene or region, or no genes at all.",
                "Weak: only the number of genes counts for anything.",
                "Partial: part of a dosage-sensitive gene is involved.",
                "Strong: a whole dosage-sensitive gene or recurrent region, or its coding start, is involved.",
            ]),
            "breakpoint_sensitivity": score("How much could the uncertain breakpoints change the classification?", [
                "Not at all: the uncertain stretches hold no genes.",
                "Slightly: more genes could be involved, but the class would stay the same.",
                "Moderately: the class could shift by one step.",
                "Decisively: the class could shift by two steps or more.",
            ]),
            "parental_testing_could_change_call": noul(
                "Testing the parents could change this CNV's classification.",
                yes="Inheritance is untested, and a de novo or an unaffected-carrier result would move the score "
                    "into another class.",
                no="Inheritance is already known, or neither possible result would change the class."),
            "orthogonal_confirmation_needed": noul(
                "This CNV needs confirmation by a second method before it is reported."),
        }

    def answer_key(self, f):
        """Rule-of-thumb answer key for the spread check only; never used as a label."""
        pts, gc = self._points(f)
        cls = _cnv_class(pts)
        if f["content"] in ("hi_complete", "hi_5prime", "recurrent") or (f["content"] == "genes" and gc >= 90):
            dose = 3 if f["content"] != "genes" else 1
        elif f["content"] in ("hi_3prime", "intragenic"):
            dose = 2
        elif f["content"] == "genes" and gc > 0:
            dose = 1
        else:
            dose = 0
        alt_pts, _ = self._points(f, alt=True)
        alt = _cnv_class(alt_pts)
        steps = abs(CLASS_ORDER.index(alt) - CLASS_ORDER.index(cls))
        bp = (1 if f["gap_content"] != "nothing" else 0) if steps == 0 else 2 if steps == 1 else 3
        if f["inheritance"] == "not_tested":
            up = _cnv_class(pts + 45)
            down = _cnv_class(pts + (-45 if f["kind"] == "loss" else -30))
            parental = up != cls or down != cls
        else:
            parental = False
        reportable = cls in ("pathogenic", "likely_pathogenic", "uncertain_significance")
        ortho = reportable and (f["probes"] < 20 or f["mosaic"] or f["content"] == "intragenic")
        return {"cnv_call": cls, "dosage_sensitivity_evidence": dose, "breakpoint_sensitivity": bp,
                "parental_testing_could_change_call": "yes" if parental else "no",
                "orthogonal_confirmation_needed": "yes" if ortho else "no"}


# =========================================================================== schema 5
class LabDiscordanceResolution(Schema):
    name = "lab_discordance_resolution"
    description = ("Our classification of a variant disagrees with another laboratory's public submission; the "
                   "discordance review decides what to do about it.")
    surfaces = {"json": 0.5, "discordance_log": 0.25, "lab_to_lab_email": 0.25}

    # code family: one strength per family, and the two sides of a family never together
    FAM = {"PS3": "fn", "BS3": "fn", "PP3": "pred", "BP4": "pred", "PM2": "pop", "BS1": "pop", "PS2": "dn",
           "PM6": "dn", "PP1": "seg", "PS4": "case", "PM1": "dom"}
    POOL = [("PM2_Supporting", 1), ("PP3", 1), ("PM1", 2), ("PS4_Moderate", 2), ("PS4_Supporting", 1), ("PP1", 1),
            ("PP1_Moderate", 2), ("PS3_Supporting", 1), ("PS3", 4), ("PS2", 4), ("PM6", 2), ("BS3_Supporting", -1),
            ("BP4", -1), ("BS3", -4), ("BS1", -4)]
    TARGETS = {"clear": [1, 2, 3, 7, 8, 11, 12, -3, -4, -5], "borderline": [5, 6, 9, 10, 0, -1, -6, -7],
               "conflicting": [1, 2, 3, 4, 6, 7, 8, 11, -2, -4]}
    # (code, points, note) for each root cause's key item
    OUR_DATA = [("PP1_Moderate", 2, "internal families, 5 informative meioses (unpublished)"),
                ("PS2", 4, "confirmed de novo case in our cohort (unpublished)"),
                ("PS4_Moderate", 2, "three probands from our own cohort (unpublished)"),
                ("BS3", -4, "our validated assay, normal activity (unpublished)"),
                ("BS1", -4, "our in-house control series puts the frequency above the maximum credible AF")]
    NEWER = [("BS1", -4, "popDB v4 frequency above the maximum credible AF"),
             ("PS3", 4, "validated functional assay published after 2020"),
             ("PS4_Moderate", 2, "case series published after 2020"),
             ("BS3", -4, "validated assay published after 2020, normal activity")]
    WRONG = [("PS3", 4, 1, "we scored an assay without control variants as strong"),
             ("PP1_Strong", 4, 2, "we scored five informative meioses as strong"),
             ("PS4", 4, 2, "we scored four probands as strong"),
             ("PM6", 2, 0, "we scored an assumed de novo where the parents were never tested")]
    THEIR_DATA = {1: "a further unrelated proband in their cohort", 2: "two internal families, 5 informative meioses",
                  4: "a validated assay run in their lab showing loss of function",
                  -1: "an unvalidated assay showing normal activity", -4: "healthy adult carriers in their family series"}

    def _fam(self, code):
        return self.FAM[code.split("_")[0]]

    def _codeset(self, rng, target, must):
        used = [self._fam(c) for c, _, _ in must]
        for _ in range(500):
            sel = [x for x in rng.sample(self.POOL, rng.randint(0, 4)) if self._fam(x[0]) not in used]
            fams = [self._fam(c) for c, _ in sel]
            if len(set(fams)) != len(fams):
                continue
            codes = list(must) + [(c, p, None) for c, p in sel]
            if sum(p for _, p, _ in codes) == target and len(codes) >= 2:
                return codes
        return None

    def sample_factors(self, rng):
        reg = regime(rng)
        g = pick(rng, [x for x in GENES if x["inh"] == "AD"])
        root = pick_w(rng, {"external_data": 4.3, "rule_weighting": 3.5, "outdated": 1.0, "our_data": 1.2})
        sub = pick_w(rng, {"ours_right": 1.0, "ours_wrong": 1.3, "sop_silent": 1.3}) if root == "rule_weighting" else None
        outcome = pick_w(rng, {"revise": 4, "consensus": 3.2, "keep": 2.3}) if root == "external_data" else None
        for _ in range(400):
            T = pick(rng, self.TARGETS[reg])
            ours_cls = _cls(T)
            o = CLASS_ORDER.index(ours_cls)
            must, their_cls, extra, note = [], None, None, None
            if root == "external_data":
                p = pick(rng, [1, 2, 4, -1, -4])
                comb = CLASS_ORDER.index(_cls(T + p))
                step = 1 if p > 0 else -1
                if outcome == "revise" and comb == o + step:
                    their_cls = CLASS_ORDER[comb]
                elif outcome == "consensus" and comb == o + step and 0 <= o + 2 * step <= 4:
                    their_cls = CLASS_ORDER[o + 2 * step]
                elif outcome == "keep" and 0 <= o + step <= 4:
                    their_cls, p = CLASS_ORDER[o + step], 0
                else:
                    continue
                extra = p
                note = (self.THEIR_DATA[p] if p else
                        pick(rng, ["one small family with 2 informative meioses",
                                   "a proband who is also in our own cohort (same patient)",
                                   "an unpublished assay with no control variants and no stated result threshold"]))
            elif root == "our_data":
                c, pts, nt = pick(rng, self.OUR_DATA)
                must = [(c, pts, nt)]
                their_cls = _cls(T - pts)
            elif root == "outdated":
                c, pts, nt = pick(rng, self.NEWER)
                must = [(c, pts, nt)]
                their_cls = _cls(T - pts)
            elif sub == "ours_wrong":
                c, pts, right, nt = pick(rng, self.WRONG)
                must = [(c, pts, None)]
                their_cls = _cls(T - pts + right)
                note = nt
            elif sub == "ours_right":
                if chance(rng, 0.6):
                    must, d = [("PS3_Supporting", 1, "assay without control variants")], 3
                    note = "the uncontrolled assay is scored as strong"
                else:
                    must, d = [("PM2_Supporting", 1, None)], 1
                    note = "rarity is scored at moderate rather than supporting"
                their_cls = _cls(T + d)
            else:  # sop_silent
                if g["domain"] and chance(rng, 0.6):
                    must, d = [("PM1", 2, None)], 2
                    note = f"PM1 is applied at strong for the {g['domain']} under a gene-specific specification"
                else:
                    must, d = [("PM2_Supporting", 1, None)], -5
                    note = (f"a lower, gene-specific frequency cut-off for {g['sym']} is applied, giving BS1 "
                            f"instead of PM2")
                their_cls = _cls(T + d)
            if their_cls == ours_cls:
                continue
            codes = self._codeset(rng, T, must)
            if codes is None:
                continue
            break
        else:  # pragma: no cover - the loop above always finds a case in practice
            raise RuntimeError("lab_discordance_resolution: could not build a case")
        reports = pick(rng, [0, 1, 2, 3, 5, 8, 13])
        return {"regime": reg, "gene": g["sym"], "root": root, "sub": sub, "ext_outcome": outcome,
                "our_codes": [list(x) for x in codes], "our_total": T, "our_class": ours_cls,
                "their_class": their_cls, "their_extra_pts": extra, "their_note": note,
                "their_eval_year": rng.randint(2013, 2017) if root == "outdated" else rng.randint(2022, 2026),
                "our_eval_year": rng.randint(2024, 2026), "reports_issued": reports,
                "management_on_record": reports > 0 and chance(rng, 0.45), "ext_lab": _lab(rng),
                "hgvs": " ".join(_variant(rng, g, pick(rng, ["missense", "missense", "inframe_del"]))),
                "curator": person(rng), "ext_contact": person(rng), "case_id": ident(rng, "DR", 5, "-"),
                "accession": f"VA{rng.randint(100000, 999999)}", "sop_id": _sop_id(rng),
                "panel_exists": chance(rng, 0.5)}

    def _ours(self, f):
        return ", ".join(f"{c} ({p:+d}" + (f"; {n})" if n else ")") for c, p, n in f["our_codes"])

    def _their_summary(self, f):
        root, sub = f["root"], f["sub"]
        their = CLASS_TEXT[f["their_class"]]
        if root == "external_data":
            pts = f"worth {f['their_extra_pts']:+d} under our SOP" if f["their_extra_pts"] else \
                "scores nothing under our SOP"
            return f"{their}; cites data not in our records: {f['their_note']} ({pts})"
        if root == "our_data":
            return f"{their}; the evidence listed is the published data only, with none of our internal data"
        if root == "outdated":
            return f"{their}; last evaluated {f['their_eval_year']}, with none of the evidence published since"
        if sub == "ours_wrong":
            return f"{their}; same data as ours, but one item weighted lower (our internal check: {f['their_note']})"
        if sub == "sop_silent":
            return f"{their}; same data as ours, but {f['their_note']}; our SOP has not adopted any {f['gene']}-specific rules"
        return f"{their}; same data as ours, but {f['their_note']}, which our SOP does not allow"

    POLICY = ["Discordance policy: share unshared case data with the other lab first; keep our call if theirs rests "
             "on outdated data or on weighting our SOP does not allow; adopt a revised call when the combined "
             "evidence, scored under our SOP, supports it (which may sit between the two calls); escalate when the "
             "difference rests on a rule our SOP does not address."]

    def render(self, f, rng, surface):
        g = next(x for x in GENES if x["sym"] == f["gene"])
        rules = [*_pc(rng, "discordance"), *self.POLICY]
        ours = self._ours(f)
        theirs = self._their_summary(f)
        impact = (f"{f['reports_issued']} of our reports carry this variant" if f["reports_issued"] else
                  "no patient reports issued with this variant yet")
        if f["management_on_record"]:
            impact += "; at least one patient's preventive care followed our result"
        panel = ("an expert panel for this gene accepts referrals" if f["panel_exists"]
                 else "no expert panel covers this gene; the lab's discordance committee is the escalation route")
        if surface == "json":
            return dumps(prune({
                "variant": {"gene": g["sym"], "hgvs": f["hgvs"], "disorder": g["disorder"], "archive": f["accession"]},
                "our_classification": {"class": CLASS_TEXT[f["our_class"]], "codes": ours, "total": f["our_total"],
                                       "evaluated": f["our_eval_year"]},
                "other_lab": {"lab": f["ext_lab"], "summary": theirs, "evaluated": f["their_eval_year"]},
                "context": {"impact": impact, "escalation": panel, "case": f["case_id"]},
                "sop": {"id": f"{f['sop_id']} (local)", "rules": rules},
            }))
        if surface == "discordance_log":
            rows = [["our lab", CLASS_SHORT[f["our_class"]], f["our_eval_year"], f"{ours} = {f['our_total']}"],
                    [f["ext_lab"], CLASS_SHORT[f["their_class"]], f["their_eval_year"], theirs]]
            return (f"DISCORDANCE LOG {f['case_id']} -- {g['sym']} {f['hgvs']} ({g['disorder']}), archive "
                    f"{f['accession']}\n\n" + table(["lab", "class", "evaluated", "basis"], rows, rng)
                    + f"\n\nImpact: {impact}.\nEscalation: {panel}.\nCurator: {f['curator']}\n\nSOP (local):\n"
                    + "\n".join(rules))
        body = [f"Subject: {g['sym']} {f['hgvs']} -- our {CLASS_SHORT[f['our_class']]} vs your "
                f"{CLASS_SHORT[f['their_class']]} ({f['accession']})", f"From: {f['curator']}",
                f"To: {f['ext_contact']} ({f['ext_lab']})", "",
                f"Dear {f['ext_contact'].split()[0]},", "",
                noisy(join_sentences([f"We have this variant as {CLASS_TEXT[f['our_class']]} ({ours}; total "
                                      f"{f['our_total']}, evaluated {f['our_eval_year']}).",
                                      f"Your submission, as we read it: {theirs}.", "Could we compare notes?"]), rng),
                "", "Best wishes,", f["curator"], "",
                f"[internal note for the discordance review] {impact}. Escalation route: {panel}. SOP (local): "
                + " ".join(rules)]
        return "\n".join(body)

    def questions(self):
        return {
            "discordance_resolution": choice("How should our lab resolve the disagreement?", {
                "escalate_to_expert_panel": "Escalate it; the difference rests on a rule our SOP does not settle.",
                "keep_our_call": "Keep our classification, document why, and tell the other lab.",
                "move_to_consensus_call": "Move to a class between the two calls that the combined evidence supports.",
                "revise_to_external_call": "Revise our classification to match the other lab's.",
            }),
            "discordance_root_cause": choice("What mainly explains the disagreement?", {
                "external_data_unseen_by_us": "The other lab holds case or functional data we have not scored.",
                "our_data_unseen_by_them": "We hold unpublished data the other lab has not seen.",
                "outdated_classification": "The other lab's call predates evidence that is now available.",
                "rule_weighting_difference": "Both labs have the same data but weight it differently.",
            }),
            "management_impact": score("How much does this disagreement matter for patient care?", [
                "Hardly: both calls lead to the same management.",
                "A little: report wording differs, but care would rarely change.",
                "Considerably: one call supports predictive testing or surveillance the other does not.",
                "Greatly: patients' preventive care already depends on which call is right.",
            ]),
            "call_margin": score("How many points separate our total from the class boundary on the other lab's side?", [
                "Comfortable: four points or more.",
                "Moderate: three points.",
                "Narrow: two points.",
                "Knife-edge: a single point would move our call across.",
            ]),
            "data_exchange_likely_resolves": noul(
                "Exchanging case data between the two labs would likely bring the classifications together.",
                yes="The disagreement comes from data one lab holds or has considered and the other has not.",
                no="Both labs have weighed the same data; the difference is in how the rules are applied."),
        }

    def answer_key(self, f):
        """Rule-of-thumb answer key for the spread check only; never used as a label."""
        root, sub = f["root"], f["sub"]
        if root == "external_data":
            res = {"revise": "revise_to_external_call", "consensus": "move_to_consensus_call",
                   "keep": "keep_our_call"}[f["ext_outcome"]]
            rc = "external_data_unseen_by_us"
        elif root == "our_data":
            res, rc = "keep_our_call", "our_data_unseen_by_them"
        elif root == "outdated":
            res, rc = "keep_our_call", "outdated_classification"
        else:
            rc = "rule_weighting_difference"
            res = {"ours_wrong": "revise_to_external_call", "sop_silent": "escalate_to_expert_panel",
                   "ours_right": "keep_our_call"}[sub]
        o, t = CLASS_ORDER.index(f["our_class"]), CLASS_ORDER.index(f["their_class"])
        crosses = (o >= 3) != (t >= 3)
        g = next(x for x in GENES if x["sym"] == f["gene"])
        if not crosses:
            imp = 0 if (o >= 3 and t >= 3) or (o <= 1 and t <= 1) else 1
        else:
            imp = 3 if (g["sf"] and f["management_on_record"]) else 2
        tot = f["our_total"]
        if t > o:
            dist = {0: -6, 1: 0, 2: 6, 3: 10}[o] - tot
        else:
            dist = tot - {4: 9, 3: 5, 2: -1, 1: -7}[o]
        margin = 3 if dist <= 1 else 2 if dist == 2 else 1 if dist == 3 else 0
        return {"discordance_resolution": res, "discordance_root_cause": rc, "management_impact": imp,
                "call_margin": margin, "data_exchange_likely_resolves": "no" if root == "rule_weighting" else "yes"}


WORKFLOW = Workflow(
    name="variant_curation_review",
    title="Germline variant curation review",
    description="How a diagnostic genomics lab classifies, re-reviews and reports germline variants under its "
                "written curation SOP.",
    domain="life sciences / clinical genomics",
    schemas=[GermlineVariantClassification(), VusReclassificationRequest(), SecondaryFindingReview(),
             CnvClassification(), LabDiscordanceResolution()],
    notes="Real gene symbols with invented variants, evidence, labs and patients; the SOP rules are the fictional "
          "lab's own and are not presented as professional guidelines. No patient-facing conversation, no security "
          "alerts, no bills, no AI-agent runs.",
)
