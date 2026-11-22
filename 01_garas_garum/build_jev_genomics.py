#!/usr/bin/env python3
"""
build_jev_genomics.py -- real-world genomics / variant-interpretation records in the jev_data record schema
(build_jev_data.record / build_jev_ext2.make_record; read back through trainers/cache_features.iter_units).
Methods follow research/realworld_data.md §3.1 (vote-share soft labels), §3.2 (expert labels with a calibrated base
rate) and §3.8 (integrity rules). Write-up, verified licences, statistics: research/genomics_pipeline.md.

Sources
  clinvar   NCBI ClinVar tab-delimited release (submission_summary + variant_summary) and the GRCh38 VCF.
            NCBI "places no restrictions on the use or distribution of the data"; ClinVar asks for attribution.
            Unit = one variant-condition pair with >= 2 INDEPENDENT submitting organisations. Each organisation's
            current germline classification (P / LP / VUS / LB / B) is one vote; the labels are the smoothed vote
            shares. train / val are stratified samples (keep rates: 0.5 of pairs whose votes span P/LP vs VUS vs
            B/LB, 0.15 of within-bin disagreements, 0.04 of unanimous pairs; <= ~1,200 per gene);
            meta.sample_weight = 1 / keep rate restores the population. test / test_temporal are uniform samples.
  civic     CIViC nightly ClinicalEvidenceSummaries (CC0 1.0). Unit = one accepted evidence item (identical states
            merged). Labels: the curator's evidence level and significance, smoothed toward a leave-one-out
            hierarchical base rate (§3.2 with lambda = 1 / (1 + a), a = 1).
  enrichment: Orphanet product 9 (CC BY 4.0): inheritance mode of the condition, joined through the Orphanet ids
            ClinVar lists for each MedGen condition.
  PharmGKB / ClinPGx clinical annotations: NOT used (CC BY-SA 4.0 plus "Under no circumstances can PharmGKB data be
            sold for other's private or commercial use"); see the doc.

Questions (none is one of the 15 typed-decisions benchmark qids; checked against synth_workflows.exclusion)
  clinvar   germline_classification   score 5   benign < likely benign < uncertain < likely pathogenic < pathogenic,
                                                each level with a written ACMG/AMP-2015-style definition
            clinically_actionable     noul      P or LP (a positive, reportable finding) -- derived: g_P + g_LP
  civic     evidence_level            score 5   E inferential < D preclinical < C case study < B clinical < A validated
            evidence_significance     choice    per evidence type (predictive / prognostic / diagnostic / predisposing
                                                / oncogenic / functional), direction folded in ("not_<x>" = the item
                                                does not support x)

Labels
  clinvar   g_k = (c_k + a m_k) / (n + a): the Dirichlet-multinomial posterior mean of the organisations' votes
            (realworld_data §3.1), m = the train base rate, a = argmax of the DM marginal likelihood over train pairs.
            meta.counts keeps the raw votes; meta.n_raters = independent organisations; per-vote provenance
            (organisation, level, date, SCV) is in <out>/clinvar_votes.tsv.gz, keyed by record id.
            Independence: one vote per organisation FAMILY (corporate siblings such as the two Labcorp labs, Quest +
            Athena, GeneDx + Sema4, and labs of one institution are collapsed, the most recent vote kept).
            Votes: ReviewStatus "criteria provided, single submitter" only (labs that state ACMG-style criteria);
            expert-panel / practice-guideline SCVs are NOT votes (they re-curate the others) and go to
            meta.expert_panel only; flagged, no-assertion-criteria and somatic-only SCVs are dropped.
  civic     t = (onehot + a p_hat) / (1 + a), a = 1 (= lambda 0.5 blend); p_hat = leave-one-out hierarchical rate
            (evidence type > type x gene > type x molecular profile > type x profile x disease) from TRAIN items only,
            per-level prior strength chosen by LOO log score.

State (what a curator sees BEFORE classifying; no classification text)
  clinvar   gene, cytoband, transcript HGVS + protein change, variant type, molecular consequence (VCF MC),
            population allele frequencies (ExAC / GO-ESP / 1000 Genomes from the ClinVar VCF, only those reported;
            gnomAD is not in the file), reported allele origin, the condition (MedGen name; "none named" when the
            submitters gave "not provided" / "not specified", which are pooled) and its Orphanet inheritance. NEVER: ClinicalSignificance, review status, number of submitters, submitter names,
            LastEvaluated, OtherIDs (OMIM allelic-variant / UniProt VAR links mark known disease alleles), rs / RCV /
            SCV / Variation ids, submitter free text. Every rendered state passes OUTCOME_WORDS (pathogen*, benign,
            uncertain, VUS, likely, conflicting, risk allele, ...); a hit drops the record (counted).
  civic     molecular profile (genes, variant types, HGVS), disease, phenotypes, therapies + interaction, variant
            origin, evidence type, source kind / year / trial ids. NEVER: evidence statement, rating, direction,
            significance, level, molecular-profile summary, author names. Per-type outcome-word scan, hits dropped.
  Surfaces: compact sorted-key JSON (", " / ": " separators) for ~60% of records, a text worksheet for the rest.

Splits (identical gene hash for both sources: no gene is in both train and an eval split)
  gene      sha1("jev-genomics-gene:" + SYMBOL) % 100: < 10 test genes, 10-14 val genes, else train genes.
  temporal  cutoff 2025-07-01 (DateLastEvaluated). train / val labels use only votes evaluated before the cutoff
            (undated votes count as before); a train-gene pair that had < 2 such votes is not written.
            test-gene pairs with >= 2 pre-cutoff votes -> split `test`; test-gene pairs that only reached two
            independent votes after the cutoff -> split `test_temporal`. Eval labels use all current votes.
            meta.strictly_prospective marks pairs whose every vote is a version-1 SCV evaluated after the cutoff.
  civic     same gene hash (a profile whose genes straddle splits is dropped); test_temporal = test-gene items
            created after the cutoff (evidence id above the largest id last reviewed before it).
Output: <out>/{train,val,test,test_temporal}/{clinvar,civic}.jsonl (~157 MB at the defaults),
<out>/clinvar_votes.tsv.gz, <out>/manifest.json, <out>/check_report.json, <out>/_cache/ (compact streamed extracts,
~120 MB; the raw dumps are streamed and never stored). Whole directory ~285 MB.

Usage (laptop is fine: ~1.05 GB is streamed, peak RAM ~2 GB; fetch ~1.5 min, build ~1.5 min, check ~10 s):
  python3 build_jev_genomics.py fetch --out /tmp/genomics_jev
  mkdir -p /tmp/jevstub && echo 'AutoTokenizer = None' > /tmp/jevstub/transformers.py \\
      && echo 'class Tensor: pass' > /tmp/jevstub/torch.py
  PYTHONPATH=/tmp/jevstub .venv/bin/python build_jev_genomics.py build --out /tmp/genomics_jev
  PYTHONPATH=/tmp/jevstub .venv/bin/python build_jev_genomics.py check --out /tmp/genomics_jev
  PYTHONPATH=/tmp/jevstub .venv/bin/python build_jev_genomics.py self-test
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import math
import random
import re
import statistics
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_jev_data import choice_q, noul_q, option_keys, record, score_q, validate  # noqa: E402

csv.field_size_limit(1 << 30)

UA = "jev-genomics-build/0.1 (research dataset build; streamed bulk downloads)"
CLINVAR_TAB = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/"
URLS = {
    "submission_summary": CLINVAR_TAB + "submission_summary.txt.gz",
    "variant_summary": CLINVAR_TAB + "variant_summary.txt.gz",
    "vcf": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz",
    "orphanet_p9": "https://www.orphadata.com/data/xml/en_product9_ages.xml",
    "civic_evidence": "https://civicdb.org/downloads/nightly/nightly-ClinicalEvidenceSummaries.tsv",
    "civic_profiles": "https://civicdb.org/downloads/nightly/nightly-MolecularProfileSummaries.tsv",
    "civic_variants": "https://civicdb.org/downloads/nightly/nightly-VariantSummaries.tsv",
}
LICENSE_CLINVAR = "ClinVar/NCBI: no use restrictions; cite PMID 29165669"
LICENSE_ORPHA = "; inheritance: Orphanet CC BY 4.0"
LICENSE_CIVIC = ("CIViC (civicdb.org): CC0 1.0 Universal (Creative Commons Public Domain Dedication); cite Griffith "
                 "et al. 2017, PMID 28138153.")

CUTOFF = date(2025, 7, 1)
TEST_PCT, VAL_PCT = 10, 5
SPLITS = ("train", "val", "test", "test_temporal")
JSON_SURFACE_PCT = 60
SPLIT_REASON = {"train": "gene;asof", "val": "gene;asof", "test": "gene_heldout", "test_temporal": "gene_heldout;new"}
QIDS_BENCH_CHECK = ("germline_classification", "clinically_actionable", "evidence_level", "evidence_significance")

# --------------------------------------------------------------------------- ClinVar vocabulary
LEVEL_KEYS = ["B", "LB", "VUS", "LP", "P"]
CLASS_MAP = {
    "Benign": "B", "Likely benign": "LB", "Uncertain significance": "VUS", "Likely pathogenic": "LP",
    "Pathogenic": "P", "Pathogenic, low penetrance": "P", "Likely pathogenic, low penetrance": "LP",
    "VUS-high": "VUS", "VUS-mid": "VUS", "VUS-low": "VUS",
}
VOTE_STATUS = "criteria provided, single submitter"
EXPERT_STATUS = {"reviewed by expert panel", "practice guideline"}
NO_CONDITION_CUIS = {"C3661900", "CN169374", "na", "", "-"}       # MedGen "not provided", "not specified"
NO_CONDITION = "__none__"
SMALL_TYPES = {"single nucleotide variant", "Deletion", "Duplication", "Insertion", "Indel", "Microsatellite"}

GERMLINE_LEVELS = [
    "Benign: strong evidence it does not cause the condition",
    "Likely benign: over 90% certain it does not cause the condition",
    "Uncertain significance: evidence insufficient or conflicting",
    "Likely pathogenic: over 90% certain it causes the condition",
    "Pathogenic: strong evidence it causes the condition",
]
GERMLINE_INSTR = ("How would a clinical genetics lab classify this germline variant for the condition (ACMG/AMP 2015 "
                  "five-tier scheme)?")
ACTIONABLE_INSTR = ("A clinical lab would report this variant as a positive finding for the condition.\n"
                    "Yes means: pathogenic or likely pathogenic.\n"
                    "No means: uncertain significance, likely benign or benign.")

OUTCOME_WORDS = re.compile(
    r"(pathogen\w*|benign\w*|\blikely\b|uncertain|\bVUS\b|unknown significance|conflict\w*|deleterious|"
    r"disease[- ]causing|causative|risk allele|risk factor|protective|drug response|actionable|reportable|"
    r"classif\w*|clinvar|expert panel|review status|\bstars?\b|criteria provided|submitter\w*|\bACMG\b)", re.I)

# Organisation families: votes from one family are not independent (one corporate lab under two submitter names,
# a lab and its acquirer, or two labs of one institution). Checked in order; the fallback is the institution part.
FAMILY_RULES = [
    (r"labcorp|laboratory corporation of america|invitae|integrated genetics", "labcorp"),
    (r"quest diagnostics|athena diagnostics", "quest"),
    (r"genedx|sema4", "genedx"),
    (r"\barup\b", "arup"),
    (r"illumina", "illumina"),
    (r"ambry", "ambry"),
    (r"myriad", "myriad"),
    (r"fulgent", "fulgent"),
    (r"eurofins", "eurofins"),
    (r"revvity|perkinelmer", "revvity"),
    (r"natera", "natera"),
    (r"preventiongenetics|exact sciences", "preventiongenetics"),
    (r"color diagnostics|color health|color genomics", "color"),
    (r"mayo clinic", "mayo"),
    (r"baylor genetics|baylor college of medicine", "baylor"),
    (r"blueprint genetics", "blueprint"),
    (r"centogene", "centogene"),
]
CORP_SUFFIX = re.compile(r"^(inc|inc\.|llc|ltd|ltd\.|gmbh|corp|corporation|co|s\.a\.|ag|plc|s\.l\.|pvt\.? ltd\.?|"
                         r"limited|llp|b\.v\.|bv|sa|srl|s\.r\.l\.)$", re.I)

# --------------------------------------------------------------------------- CIViC vocabulary
CIVIC_LEVELS = [
    "E - Inferential: the association is inferred indirectly (related variant, pathway or mechanism).",
    "D - Preclinical: in vitro or in vivo models (cell lines, animals, assays) support it.",
    "C - Case study: individual case reports or small case series.",
    "B - Clinical evidence: clinical trials or other primary patient cohorts support it.",
    "A - Validated: proven or consensus association (guideline, approved therapy or test).",
]
CIVIC_LEVEL_KEYS = ["E", "D", "C", "B", "A"]
CIVIC_LEVEL_INSTR = ("How strong is this evidence item under CIViC's evidence levels, as a CIViC curator and editor "
                     "would grade it from the cited source?")
CIVIC_SIG_INSTR = {
    "Predictive": "What does this evidence item report about the molecular profile's effect on response to the "
                  "therapy in this disease?",
    "Prognostic": "What does this evidence item report about the molecular profile and patient outcome in this "
                  "disease?",
    "Diagnostic": "What does this evidence item report about the molecular profile as a marker for diagnosing this "
                  "disease?",
    "Predisposing": "What does this evidence item report about the germline variant's role in predisposition to this "
                    "disease?",
    "Oncogenic": "What does this evidence item report about whether the somatic variant drives this cancer?",
    "Functional": "What does this evidence item report about the variant's effect on the gene product's function?",
}
CIVIC_SIG = {   # (evidence type, significance slug, direction) -> option key, description
    "Predictive": {
        "sensitivity_response": "Sensitivity or response: the profile is associated with response to the therapy.",
        "resistance": "Resistance: the profile is associated with resistance to the therapy.",
        "reduced_sensitivity": "Reduced sensitivity: response to the therapy is diminished but not abolished.",
        "adverse_response": "Adverse response: the profile is associated with toxicity or adverse events on the "
                            "therapy.",
        "not_sensitivity_response": "Does not support sensitivity or response: the source finds no response benefit "
                                    "linked to the profile.",
        "not_resistance": "Does not support resistance: the source finds the profile does not confer resistance.",
    },
    "Prognostic": {
        "poor_outcome": "Poor outcome: the profile is associated with worse survival or disease course.",
        "better_outcome": "Better outcome: the profile is associated with better survival or disease course.",
        "not_prognostic": "No prognostic association: the source finds no link (or argues against a link) between "
                          "the profile and outcome.",
    },
    "Diagnostic": {
        "positive": "Positive marker: the profile supports the diagnosis of this disease.",
        "negative": "Negative marker: the profile argues against, or helps exclude, this diagnosis.",
        "not_positive": "Does not support a positive association: the source finds the profile is not a marker of "
                        "this disease.",
    },
    "Predisposing": {
        "predisposition": "Predisposition: the germline variant raises the risk of developing the disease.",
        "uncertain_significance": "Uncertain: the source does not establish whether the variant raises risk.",
    },
    "Oncogenic": {
        "oncogenicity": "Oncogenic: the variant contributes to tumour development or progression.",
        "not_oncogenicity": "Does not support oncogenicity: the source argues the variant does not drive the "
                            "cancer.",
    },
    "Functional": {
        "gain_of_function": "Gain of function: the variant increases or constitutively activates the gene product's "
                            "activity.",
        "loss_of_function": "Loss of function: the variant reduces or abolishes the gene product's activity.",
        "dominant_negative": "Dominant negative: the variant product interferes with the normal product.",
        "not_dominant_negative": "Does not support a dominant-negative effect.",
        "unaltered_function": "Unaltered function: the variant leaves the gene product's activity unchanged.",
        "neomorphic": "Neomorphic: the variant gives the gene product a new function.",
        "not_neomorphic": "Does not support a neomorphic effect.",
    },
}
CIVIC_SIG_SLUG = {"Sensitivity/Response": "sensitivity_response", "Resistance": "resistance",
                  "Reduced Sensitivity": "reduced_sensitivity", "Adverse Response": "adverse_response",
                  "Poor Outcome": "poor_outcome", "Better Outcome": "better_outcome", "Positive": "positive",
                  "Negative": "negative", "Predisposition": "predisposition",
                  "Uncertain Significance": "uncertain_significance", "Oncogenicity": "oncogenicity",
                  "Gain of Function": "gain_of_function", "Loss of Function": "loss_of_function",
                  "Dominant Negative": "dominant_negative", "Unaltered Function": "unaltered_function",
                  "Neomorphic": "neomorphic", "Protectiveness": "protectiveness", "N/A": "na"}
CIVIC_GLOBAL_WORDS = r"pathogen\w*|benign|\blevel [a-e]\b|evidence level|\brating\b|\bstars?\b|supports?\b|does not support"
CIVIC_TYPE_WORDS = {
    "Predictive": r"sensitiv\w*|resist\w*|respon\w*|adverse|toxic\w*|refractor\w*|benefit\w*",
    "Prognostic": r"poor|better|favou?rabl\w*|unfavou?rabl\w*|worse|surviv\w*|outcome",
    "Diagnostic": r"positive|negative|diagnos(is|ed|tic marker)",
    "Predisposing": r"predispos\w*|protect\w*|uncertain|susceptib\w*|risk",
    "Oncogenic": r"oncogenicity|driver|protect\w*",
    "Functional": r"gain[- ]of[- ]function|loss[- ]of[- ]function|dominant[- ]negative|neomorph\w*|unaltered|"
                  r"activating|inactivating",
}


# --------------------------------------------------------------------------- small helpers
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(s, n=16):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def frac(key):
    """Deterministic uniform [0, 1) from a string."""
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:12], 16) / float(1 << 48)


def gene_split(gene):
    b = int(hashlib.sha1(("jev-genomics-gene:" + gene.strip().upper()).encode()).hexdigest(), 16) % 100
    return "test" if b < TEST_PCT else "val" if b < TEST_PCT + VAL_PCT else "train"


MONTHS = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}


def clinvar_date(s):
    m = re.match(r"(\w{3}) (\d{1,2}), (\d{4})$", (s or "").strip())
    if not m or m.group(1) not in MONTHS:
        return None
    try:
        return date(int(m.group(3)), MONTHS[m.group(1)], int(m.group(2)))
    except ValueError:
        return None


def jdump(obj):
    """The benchmark's state surface: compact JSON, sorted keys, ', ' / ': ' separators."""
    return json.dumps(obj, sort_keys=True, separators=(", ", ": "), ensure_ascii=False)


def entropy(p):
    return -sum(x * math.log(x) for x in p if x > 0)


def label_stats(dists, weights=None):
    """mean max-p / entropy / one-hot share / exact-zero share; `weights` (meta.sample_weight) reweights a
    stratified sample back to its population."""
    if not dists:
        return {}
    w = weights or [1.0] * len(dists)
    W = sum(w)

    def wm(vals):
        return sum(x * y for x, y in zip(vals, w)) / W
    return {"n": len(dists), "mean_max_p": round(wm([max(d) for d in dists]), 4),
            "mean_entropy_nats": round(wm([entropy(d) for d in dists]), 4),
            "share_one_hot": round(wm([float(max(d) > 1 - 1e-9) for d in dists]), 4),
            "share_exact_zeros": round(wm([sum(1 for x in d if x == 0) / len(d) for d in dists]), 4)}


# --------------------------------------------------------------------------- streaming downloads
class HashingReader(io.RawIOBase):
    """Wraps an HTTP response: counts and md5-hashes the compressed bytes as they stream past."""

    def __init__(self, resp):
        self.resp, self.md5, self.n = resp, hashlib.md5(), 0

    def readable(self):
        return True

    def readinto(self, b):
        data = self.resp.read(len(b))
        n = len(data)
        b[:n] = data
        self.md5.update(data)
        self.n += n
        return n


def open_url(url, timeout=600):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=timeout)


def stream_lines(url):
    """(line iterator over the gunzipped stream, HashingReader). Nothing is written to disk."""
    raw = HashingReader(open_url(url))
    gz = gzip.GzipFile(fileobj=io.BufferedReader(raw, 1 << 20))
    return io.TextIOWrapper(gz, encoding="utf-8", errors="replace", newline="\n"), raw


def with_retries(fn, what, tries=3):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            log(f"{what}: {type(e).__name__}: {e}; retry {i + 1} in {10 * (i + 1)}s")
            time.sleep(10 * (i + 1))


def remote_md5(url):
    try:
        with open_url(url + ".md5", timeout=60) as r:
            return r.read().decode().split()[0]
    except Exception:  # noqa: BLE001
        return None


def download_small(url, path):
    def get():
        with open_url(url) as r:
            data = r.read()
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(data)
        tmp.rename(path)
        return len(data)
    return with_retries(get, url)


# --------------------------------------------------------------------------- fetch
SCV_COLS = ["VariationID", "ClinicalSignificance", "DateLastEvaluated", "ReportedPhenotypeInfo", "ReviewStatus",
            "CollectionMethod", "OriginCounts", "Submitter", "SCV", "ContributesToAggregateClassification"]
VAR_COLS = ["VariationID", "AlleleID", "Type", "Name", "GeneID", "GeneSymbol", "Origin", "Assembly", "Chromosome",
            "Start", "Cytogenetic", "Guidelines", "RS# (dbSNP)"]
VCF_KEYS = ["CLNHGVS", "CLNVC", "MC", "AF_EXAC", "AF_ESP", "AF_TGP"]


def fetch_submissions(cache):
    """submission_summary -> _cache/scv.tsv.gz: the SCV columns we use (never the free-text Description or the
    submitted phenotype text), for variants with >= 2 distinct submitters holding a 5-tier germline classification.
    Returns the set of those VariationIDs."""
    out_path = cache / "scv.tsv.gz"
    needed = set()

    def run():
        needed.clear()
        lines, raw = stream_lines(URLS["submission_summary"])
        tmp = out_path.with_name(out_path.name + ".part")
        n_rows = n_kept = 0
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as out:
            out.write("\t".join(SCV_COLS) + "\n")
            hdr = idx = None
            cur, buf, prev = None, [], -1

            def flush():
                nonlocal n_kept
                subs = {r[7] for r in buf if r[1] in CLASS_MAP}
                if len(subs) >= 2:
                    needed.add(int(cur))
                    for r in buf:
                        out.write("\t".join(r) + "\n")
                    n_kept += len(buf)

            for line in lines:
                if line.startswith("#VariationID\t"):
                    hdr = line[1:].rstrip("\n").split("\t")
                    idx = [hdr.index(c) for c in SCV_COLS]
                    continue
                if line.startswith("#"):
                    continue
                p = line.rstrip("\n").split("\t")
                if hdr is None or len(p) < len(hdr):
                    continue
                n_rows += 1
                vid = p[0]
                if vid != cur:
                    if buf:
                        flush()
                    if int(vid) < prev:
                        raise RuntimeError("submission_summary is not sorted by VariationID; the grouped extract "
                                           "assumes it is")
                    prev, cur, buf = int(vid), vid, []
                buf.append([p[i].replace("\t", " ") for i in idx])
            if buf:
                flush()
        tmp.rename(out_path)
        return {"rows": n_rows, "rows_kept": n_kept, "variants_kept": len(needed), "gz_bytes": raw.n,
                "md5": raw.md5.hexdigest()}
    info = with_retries(run, "submission_summary")
    info["md5_expected"] = remote_md5(URLS["submission_summary"])
    return info, needed


def fetch_variants(cache, needed):
    """variant_summary -> _cache/variants.tsv.gz (needed VariationIDs; GRCh38 row, else GRCh37) and
    _cache/cui_orphanet.tsv.gz (MedGen CUI -> Orphanet codes, from every row's PhenotypeIDS)."""
    out_path = cache / "variants.tsv.gz"

    def run():
        lines, raw = stream_lines(URLS["variant_summary"])
        best, cui_orpha = {}, defaultdict(set)
        hdr = None
        n = 0
        for line in lines:
            if line.startswith("#AlleleID"):
                hdr = line[1:].rstrip("\n").split("\t")
                H = {k: i for i, k in enumerate(hdr)}
                continue
            p = line.rstrip("\n").split("\t")
            if hdr is None or len(p) < len(hdr):
                continue
            n += 1
            for cond in re.split(r"[|;]", p[H["PhenotypeIDS"]]):
                ids = cond.split(",")
                cuis = [x.split(":", 1)[1] for x in ids if x.startswith("MedGen:")]
                orphas = [x.split(":", 1)[1] for x in ids if x.startswith("Orphanet:")]
                for c in cuis:
                    cui_orpha[c].update(orphas)
            try:
                vid = int(p[H["VariationID"]])
            except ValueError:
                continue
            if vid not in needed:
                continue
            asm = p[H["Assembly"]]
            if asm not in ("GRCh38", "GRCh37"):
                continue
            if vid in best and not (asm == "GRCh38" and best[vid][7] != "GRCh38"):
                continue
            best[vid] = [p[H[c]].replace("\t", " ") for c in VAR_COLS]
        tmp = out_path.with_name(out_path.name + ".part")
        with gzip.open(tmp, "wt", encoding="utf-8") as out:
            out.write("\t".join(VAR_COLS) + "\n")
            for vid in sorted(best):
                out.write("\t".join(best[vid]) + "\n")
        tmp.rename(out_path)
        with gzip.open(cache / "cui_orphanet.tsv.gz", "wt", encoding="utf-8") as out:
            for c in sorted(cui_orpha):
                if cui_orpha[c]:
                    out.write(f"{c}\t{','.join(sorted(cui_orpha[c]))}\n")
        return {"rows": n, "variants_kept": len(best), "gz_bytes": raw.n, "md5": raw.md5.hexdigest(),
                "cuis_with_orphanet": sum(1 for v in cui_orpha.values() if v)}
    info = with_retries(run, "variant_summary")
    info["md5_expected"] = remote_md5(URLS["variant_summary"])
    return info


def fetch_vcf(cache, needed):
    """GRCh38 VCF -> _cache/vcf.tsv.gz: genomic HGVS, variant class, molecular consequence and the three allele
    frequencies, for needed VariationIDs. CLNSIG / CLNREVSTAT / CLNSIGCONF etc. are never read."""
    out_path = cache / "vcf.tsv.gz"

    def run():
        lines, raw = stream_lines(URLS["vcf"])
        tmp = out_path.with_name(out_path.name + ".part")
        n = kept = 0
        with gzip.open(tmp, "wt", encoding="utf-8") as out:
            out.write("\t".join(["VariationID", "CHROM", "POS", "REF", "ALT"] + VCF_KEYS) + "\n")
            for line in lines:
                if line.startswith("#"):
                    continue
                p = line.rstrip("\n").split("\t", 8)
                n += 1
                try:
                    vid = int(p[2])
                except ValueError:
                    continue
                if vid not in needed:
                    continue
                info = dict(kv.split("=", 1) for kv in p[7].split(";") if "=" in kv)
                out.write("\t".join([p[2], p[0], p[1], p[3], p[4]] + [info.get(k, "") for k in VCF_KEYS]) + "\n")
                kept += 1
        tmp.rename(out_path)
        return {"rows": n, "kept": kept, "gz_bytes": raw.n, "md5": raw.md5.hexdigest()}
    info = with_retries(run, "vcf")
    info["md5_expected"] = remote_md5(URLS["vcf"])
    return info


def parse_orphanet(xml_path):
    """Orphanet product 9 -> {orphacode: {name, inheritance[], onset[]}}; uninformative values dropped."""
    skip = {"Not applicable", "Unknown", "No data available", "Not yet documented", "All ages"}
    out, licence = {}, None
    root = ET.parse(xml_path).getroot()
    lic = root.find("./Availability/Licence/ShortIdentifier")
    licence = lic.text if lic is not None else None
    for d in root.iter("Disorder"):
        code = d.findtext("OrphaCode")
        if not code:
            continue
        inh = [e.findtext("Name") for e in d.findall("./TypeOfInheritanceList/TypeOfInheritance")]
        ons = [e.findtext("Name") for e in d.findall("./AverageAgeOfOnsetList/AverageAgeOfOnset")]
        out[code] = {"name": d.findtext("Name"), "inheritance": [x for x in inh if x and x not in skip],
                     "onset": [x for x in ons if x and x not in skip]}
    return out, licence, root.get("date")


def fetch(a):
    out = Path(a.out)
    cache = out / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    man = {"fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "urls": URLS, "files": {}}
    if not a.skip_clinvar:
        log("streaming submission_summary.txt.gz (~0.4 GB) ...")
        info, needed = fetch_submissions(cache)
        man["files"]["submission_summary"] = info
        log(f"  {info}")
        log("streaming variant_summary.txt.gz (~0.45 GB) ...")
        man["files"]["variant_summary"] = fetch_variants(cache, needed)
        log(f"  {man['files']['variant_summary']}")
        log("streaming clinvar.vcf.gz GRCh38 (~0.19 GB) ...")
        man["files"]["vcf"] = fetch_vcf(cache, needed)
        log(f"  {man['files']['vcf']}")
    for key, fn in (("orphanet_p9", "orphanet_p9.xml"), ("civic_evidence", "civic_evidence.tsv"),
                    ("civic_profiles", "civic_profiles.tsv"), ("civic_variants", "civic_variants.tsv")):
        n = download_small(URLS[key], cache / fn)
        man["files"][key] = {"bytes": n, "md5": hashlib.md5((cache / fn).read_bytes()).hexdigest()}
        log(f"{key}: {n} bytes")
    orpha, lic, odate = parse_orphanet(cache / "orphanet_p9.xml")
    (cache / "orphanet_p9.json").write_text(json.dumps({"licence": lic, "date": odate, "disorders": orpha}))
    (cache / "orphanet_p9.xml").unlink()
    man["files"]["orphanet_p9"].update({"licence_in_file": lic, "file_date": odate, "disorders": len(orpha)})
    old = {}
    if (cache / "fetch_manifest.json").exists():
        old = json.loads((cache / "fetch_manifest.json").read_text())
    old.setdefault("files", {}).update(man["files"])
    old.update({k: v for k, v in man.items() if k != "files"})
    (cache / "fetch_manifest.json").write_text(json.dumps(old, indent=1))
    for k, v in man["files"].items():
        if v.get("md5_expected") and v["md5_expected"] != v.get("md5"):
            log(f"WARNING {k}: md5 {v.get('md5')} != published {v['md5_expected']} (the file may have been replaced "
                f"mid-stream; re-run fetch)")
    log("fetch done")


# --------------------------------------------------------------------------- ClinVar: votes -> pairs
def family_of(org):
    s = org.lower()
    for rx, fam in FAMILY_RULES:
        if re.search(rx, s):
            return fam
    parts = [x.strip() for x in org.split(",") if x.strip()]
    while len(parts) > 1 and CORP_SUFFIX.match(parts[-1]):
        parts.pop()
    inst = parts[-1] if parts else org
    return re.sub(r"[^a-z0-9]+", " ", inst.lower()).strip()


def condition_key(reported):
    """ReportedPhenotypeInfo 'C123:Name;C456:Name' -> (key, {cui: name}). 'not provided' / 'not specified' / 'na'
    pool into NO_CONDITION: both mean the submitter named no condition."""
    names = {}
    for part in (reported or "").split(";"):
        cui, _, name = part.partition(":")
        cui = cui.strip()
        if cui in NO_CONDITION_CUIS:
            continue
        names[cui] = name.strip()
    return ("|".join(sorted(names)) if names else NO_CONDITION), names


def origins_of(origin_counts):
    return {o.split(":")[0].strip() for o in (origin_counts or "").split(";") if o.strip()}


def collapse(votes):
    """{family: latest vote}; undated votes sort oldest."""
    by = {}
    for v in votes:
        k = v["family"]
        if k not in by or (v["date"] or date.min) >= (by[k]["date"] or date.min):
            by[k] = v
    return list(by.values())


def counts_of(votes):
    c = Counter(v["level"] for v in votes)
    return [c.get(k, 0) for k in LEVEL_KEYS]


def bin3(level):
    return {"B": 0, "LB": 0, "VUS": 1, "LP": 2, "P": 2}[level]


def iter_variant_groups(cache):
    with gzip.open(cache / "scv.tsv.gz", "rt", encoding="utf-8") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        H = {k: i for i, k in enumerate(hdr)}
        cur, buf = None, []
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[0] != cur:
                if buf:
                    yield int(cur), buf
                cur, buf = p[0], []
            buf.append({c: p[H[c]] for c in hdr})
        if buf:
            yield int(cur), buf


def pairs_of_variant(rows, drops):
    """SCV rows of one variant -> {condition key: {"votes": [...], "names": {...}, "expert": [...]}}."""
    pairs = defaultdict(lambda: {"votes": [], "names": {}, "expert": []})
    for r in rows:
        lvl = CLASS_MAP.get(r["ClinicalSignificance"])
        if lvl is None:
            drops["scv_not_5_tier"] += 1
            continue
        status = r["ReviewStatus"]
        ck, names = condition_key(r["ReportedPhenotypeInfo"])
        d = clinvar_date(r["DateLastEvaluated"])
        if status in EXPERT_STATUS:
            pairs[ck]["expert"].append({"level": lvl, "status": status, "date": d.isoformat() if d else None})
            pairs[ck]["names"].update(names)
            continue
        if status != VOTE_STATUS:
            drops[f"scv_status:{status}"] += 1
            continue
        og = origins_of(r["OriginCounts"])
        if og and og <= {"somatic"}:
            drops["scv_somatic_only"] += 1
            continue
        scv = r["SCV"]
        pairs[ck]["votes"].append({"level": lvl, "org": r["Submitter"], "family": family_of(r["Submitter"]),
                                   "date": d, "scv": scv, "v1": scv.endswith(".1"),
                                   "method": r["CollectionMethod"]})
        pairs[ck]["names"].update(names)
    return pairs


# --------------------------------------------------------------------------- ClinVar: variant info + state
def load_variants(cache):
    out = {}
    with gzip.open(cache / "variants.tsv.gz", "rt", encoding="utf-8") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for line in f:
            p = line.rstrip("\n").split("\t")
            out[int(p[0])] = tuple(p[1:])
    return {"cols": hdr[1:], "rows": out}


def load_vcf(cache):
    out = {}
    with gzip.open(cache / "vcf.tsv.gz", "rt", encoding="utf-8") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for line in f:
            p = line.rstrip("\n").split("\t")
            out[int(p[0])] = tuple(p[1:])
    return {"cols": hdr[1:], "rows": out}


def load_orpha(cache):
    cui = {}
    with gzip.open(cache / "cui_orphanet.tsv.gz", "rt", encoding="utf-8") as f:
        for line in f:
            c, codes = line.rstrip("\n").split("\t")
            cui[c] = codes.split(",")
    p9 = json.loads((cache / "orphanet_p9.json").read_text())
    return cui, p9["disorders"]


def as_dict(table, vid):
    row = table["rows"].get(vid)
    return dict(zip(table["cols"], row)) if row else None


def parse_af(s):
    try:
        x = float(s)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and 0 <= x <= 1 else None


def human(term):
    return term.replace("_", " ").strip()


def split_name(name):
    """'NM_000059.4(BRCA2):c.68-7T>A (p.Arg23Ter)' -> ('NM_000059.4(BRCA2):c.68-7T>A', 'p.Arg23Ter')."""
    m = re.match(r"^(.*?)\s*\((p\.[^()]*)\)\s*$", name)
    return (m.group(1), m.group(2)) if m else (name, None)


def variant_facts(vid, variants, vcf):
    """Pre-classification facts about one variant, or (None, reason)."""
    v = as_dict(variants, vid)
    if v is None:
        return None, "no_variant_summary_row"
    try:
        gene_id = int(v["GeneID"])
    except ValueError:
        gene_id = -1
    if gene_id <= 0 or not v["GeneSymbol"] or "," in v["GeneSymbol"] or v["GeneSymbol"] == "-":
        return None, "not_single_gene"
    if v["Type"] not in SMALL_TYPES:
        return None, f"type:{v['Type']}"
    hgvs, prot = split_name(v["Name"])
    if not re.search(r":[cnmg]\.", hgvs):
        return None, "name_not_hgvs"
    x = as_dict(vcf, vid) or {}
    mc = []
    for item in (x.get("MC") or "").split(","):
        if "|" in item:
            mc.append(human(item.split("|", 1)[1]))
    origin = [o.strip() for o in (v["Origin"] or "").split(";") if o.strip() and o.strip() not in ("not provided", "-")]
    facts = {
        "gene": v["GeneSymbol"], "chromosome": v["Chromosome"] if v["Chromosome"] not in ("", "na", "-") else None,
        "cytoband": v["Cytogenetic"] if v["Cytogenetic"] not in ("", "-", "na") else None,
        "hgvs": hgvs, "protein_change": prot, "type": v["Type"],
        "genomic_hgvs": x.get("CLNHGVS") or None,
        "molecular_consequence": sorted(set(mc)),
        "af": {"exac": parse_af(x.get("AF_EXAC")), "go_esp": parse_af(x.get("AF_ESP")),
               "1000_genomes": parse_af(x.get("AF_TGP"))},
        "allele_origin": sorted(set(origin)),
        "acmg_sf_gene": bool(re.search(r"ACMG", v["Guidelines"] or "")),
        "allele_id": v["AlleleID"], "rs": v["RS# (dbSNP)"], "assembly": v["Assembly"],
    }
    return facts, None


def condition_facts(ck, names, cui_orpha, orpha):
    if ck == NO_CONDITION:
        return {"name": None, "inheritance": [], "onset": [], "orphanet": []}
    cuis = ck.split("|")
    inh, ons, codes = set(), set(), set()
    for c in cuis:
        for code in cui_orpha.get(c, []):
            d = orpha.get(code)
            if d:
                codes.add(code)
                inh.update(d["inheritance"])
                ons.update(d["onset"])
    nm = "; ".join(names.get(c) or c for c in cuis)
    return {"name": nm, "inheritance": sorted(inh), "onset": sorted(ons), "orphanet": sorted(codes)}


def fmt_af(x):
    return None if x is None else float(f"{x:.6g}")


AF_NAMES = (("exac", "ExAC"), ("1000_genomes", "1000 Genomes"), ("go_esp", "GO-ESP"))


def clinvar_state(vf, cf, surface):
    """Decision-relevant facts only: gene + cytoband, transcript HGVS + protein change, type, consequence, allele
    frequencies (only those reported), allele origin, condition + inheritance."""
    af = {k: fmt_af(vf["af"][k]) for k, _ in AF_NAMES if vf["af"][k] is not None}
    cond_name = cf["name"] or "none named (assess against the gene's associated disorders)"
    if surface == "json":
        obj = {"condition": {"name": cond_name, **({"inheritance": cf["inheritance"]} if cf["inheritance"] else {})},
               "gene": vf["gene"],
               "variant": {"hgvs": vf["hgvs"], "type": vf["type"]},
               "allele_frequency": af or "not reported in ExAC, GO-ESP or 1000 Genomes"}
        if vf["cytoband"]:
            obj["location"] = vf["cytoband"]
        if vf["protein_change"]:
            obj["variant"]["protein"] = vf["protein_change"]
        if vf["molecular_consequence"]:
            obj["variant"]["consequence"] = vf["molecular_consequence"]
        if vf["allele_origin"]:
            obj["variant"]["origin"] = vf["allele_origin"]
        return jdump(obj)
    lines = [f"Gene: {vf['gene']}" + (f" ({vf['cytoband']})" if vf["cytoband"] else ""),
             f"Variant: {vf['hgvs']}" + (f" ({vf['protein_change']})" if vf["protein_change"] else "")
             + f", {vf['type']}"]
    if vf["molecular_consequence"]:
        lines.append(f"Consequence: {', '.join(vf['molecular_consequence'])}")
    lines.append("Allele frequency: " + ("; ".join(f"{nm} {af[k]}" for k, nm in AF_NAMES if k in af)
                                         if af else "not reported in ExAC, GO-ESP or 1000 Genomes"))
    if vf["allele_origin"]:
        lines.append(f"Allele origin: {', '.join(vf['allele_origin'])}")
    lines.append(f"Condition: {cond_name}" + (f" (inheritance: {', '.join(cf['inheritance']).lower()})"
                                              if cf["inheritance"] else ""))
    return "\n".join(lines)


def round_dist(vals, nd=6):
    """Round to nd decimals; the rounding residual goes to the largest entry, so the sum stays exactly 1."""
    r = [round(x, nd) for x in vals]
    i = max(range(len(r)), key=lambda k: r[k])
    r[i] = round(r[i] + 1.0 - sum(r), nd + 3)
    return r


# --------------------------------------------------------------------------- Dirichlet-multinomial fit
def dm_fit(count_vectors):
    """Empirical-Bayes Dirichlet-multinomial: m = pooled base rate (+0.5 per level), a = argmax marginal likelihood
    (log grid then golden-section). Identical count vectors are pooled, so this is fast for 10^5-10^6 items."""
    from scipy.special import gammaln
    import numpy as np
    pooled = Counter(tuple(c) for c in count_vectors)
    C = np.array(list(pooled.keys()), dtype=float)
    w = np.array(list(pooled.values()), dtype=float)
    K = C.shape[1]
    tot = (C * w[:, None]).sum(0)
    m = (tot + 0.5) / (tot.sum() + 0.5 * K)
    n = C.sum(1)

    def ll(a):
        al = a * m
        return float((w * (gammaln(a) - gammaln(n + a) + (gammaln(C + al) - gammaln(al)).sum(1))).sum())
    grid = [10 ** (x / 20) for x in range(-40, 81)]
    best = max(grid, key=ll)
    lo, hi = math.log(best / 10 ** 0.05), math.log(best * 10 ** 0.05)
    g = (math.sqrt(5) - 1) / 2
    for _ in range(50):
        c1, d1 = hi - g * (hi - lo), lo + g * (hi - lo)
        if ll(math.exp(c1)) > ll(math.exp(d1)):
            hi = d1
        else:
            lo = c1
    a = math.exp((lo + hi) / 2)
    return a, [float(x) for x in m], ll(a)


def smooth(c, a, m):
    n = sum(c)
    return [(c[k] + a * m[k]) / (n + a) for k in range(len(c))]


class Agreement:
    """Inter-rater ceiling for V labels (realworld_data §3.7): each organisation's vote predicted from the OTHER
    organisations' smoothed shares (leave-one-out) vs the base rate, plus weighted pairwise exact agreement / kappa
    on the 5 levels and the 3 bins. Weights = meta.sample_weight, so the numbers describe the population."""

    def __init__(self, a, m):
        self.a, self.m = a, m
        self.s = Counter()

    def add(self, cnt, w):
        a, m, K = self.a, self.m, len(cnt)
        n = sum(cnt)
        for b in range(K):
            if not cnt[b]:
                continue
            rest = [cnt[k] - (k == b) for k in range(K)]
            q = [(rest[k] + a * m[k]) / (n - 1 + a) for k in range(K)]
            ww = w * cnt[b]
            self.s["votes"] += ww
            self.s["nll_peer"] -= ww * math.log(q[b])
            self.s["nll_base"] -= ww * math.log(m[b])
            self.s["brier_peer"] += ww * sum((q[k] - (k == b)) ** 2 for k in range(K))
            self.s["brier_base"] += ww * sum((m[k] - (k == b)) ** 2 for k in range(K))
            self.s["top1_peer"] += ww * (max(range(K), key=lambda k: (q[k], -abs(k - 2))) == b)
            self.s["top1_base"] += ww * (max(range(K), key=lambda k: m[k]) == b)
        self.s["po_num"] += w * sum(c * (c - 1) for c in cnt)
        self.s["po_den"] += w * n * (n - 1)
        c3 = [cnt[0] + cnt[1], cnt[2], cnt[3] + cnt[4]]
        self.s["po3_num"] += w * sum(c * (c - 1) for c in c3)

    def result(self):
        s, m = self.s, self.m
        if not s["votes"]:
            return {}
        po, po3 = s["po_num"] / s["po_den"], s["po3_num"] / s["po_den"]
        pe = sum(x * x for x in m)
        m3 = [m[0] + m[1], m[2], m[3] + m[4]]
        pe3 = sum(x * x for x in m3)
        v = s["votes"]
        return {"loo_from_peers": {"nll": round(s["nll_peer"] / v, 4), "brier": round(s["brier_peer"] / v, 4),
                                   "top1": round(s["top1_peer"] / v, 4)},
                "loo_base_rate": {"nll": round(s["nll_base"] / v, 4), "brier": round(s["brier_base"] / v, 4),
                                  "top1": round(s["top1_base"] / v, 4)},
                "pairwise_agreement_5level": round(po, 4), "kappa_5level": round((po - pe) / (1 - pe), 4),
                "pairwise_agreement_3bin": round(po3, 4), "kappa_3bin": round((po3 - pe3) / (1 - pe3), 4)}


# --------------------------------------------------------------------------- ClinVar: build
def clinvar_candidates(cache, variants, vcf, cui_orpha, orpha, drops, diag):
    """Yield one dict per variant-condition pair with >= 2 independent organisations (current view)."""
    for vid, rows in iter_variant_groups(cache):
        pairs = pairs_of_variant(rows, drops)
        live = {ck: p for ck, p in pairs.items() if len({v["org"] for v in p["votes"]}) >= 2}
        if not live:
            continue
        vf, why = variant_facts(vid, variants, vcf)
        if vf is None:
            drops[f"variant:{why}"] += len(live)
            continue
        split = gene_split(vf["gene"])
        for ck, p in live.items():
            raw_orgs = len({v["org"] for v in p["votes"]})
            cur = collapse(p["votes"])
            if len(cur) < 2:
                drops["fewer_than_2_independent_families"] += 1
                continue
            if len(cur) < raw_orgs:
                diag["family_collapsed_pairs"] += 1
                if len({v["level"] for v in p["votes"]}) > len({v["level"] for v in cur}):
                    diag["family_collapse_removed_a_disagreement"] += 1
            pre = collapse([v for v in p["votes"] if v["date"] is None or v["date"] < CUTOFF])
            if split in ("train", "val"):
                if len(pre) < 2:
                    drops[f"{split}_gene_pair_only_after_cutoff"] += 1
                    continue
                use, out_split = pre, split
            else:
                use = cur
                out_split = "test" if len(pre) >= 2 else "test_temporal"
            yield {"vid": vid, "ck": ck, "names": p["names"], "split": out_split, "votes": use, "vf": vf,
                   "n_pre": len(pre), "n_cur": len(cur), "expert": p["expert"],
                   "strict_prosp": all(v["v1"] and v["date"] and v["date"] >= CUTOFF for v in cur)}


def clinvar_record(c, cf, a, m, keep_w):
    vf, votes = c["vf"], c["votes"]
    cnt = counts_of(votes)
    g = round_dist(smooth(cnt, a, m))
    n = sum(cnt)
    rid = f"clinvar-{sha(str(c['vid']) + '|' + c['ck'])}"
    surface = "json" if frac("surface:" + rid) * 100 < JSON_SURFACE_PCT else "text"
    state = clinvar_state(vf, cf, surface)
    qs = {"germline_classification": score_q(GERMLINE_INSTR, GERMLINE_LEVELS),
          "clinically_actionable": noul_q(ACTIONABLE_INSTR)}
    p_yes = round(min(max(g[3] + g[4], 0.0), 1.0), 6)
    labs = {"germline_classification": {GERMLINE_LEVELS[k]: g[k] for k in range(5)},
            "clinically_actionable": {"yes": p_yes, "no": round(1.0 - p_yes, 6)}}
    rec = record("clinvar", state, qs, labs, {q: "soft" for q in qs}, None)
    rec["id"] = rec["base_id"] = rid
    rec["license"] = LICENSE_CLINVAR + (LICENSE_ORPHA if cf["inheritance"] else "")
    levels = [v["level"] for v in votes]
    exp = sorted(c["expert"], key=lambda e: e["date"] or "")
    rec["meta"] = {
        "dataset": "clinvar 2026-09-28", "variation_id": c["vid"], "gene": vf["gene"], "group": vf["gene"],
        "condition_ids": [] if c["ck"] == NO_CONDITION else c["ck"].split("|"),
        "n_raters": n, "counts": dict(zip(LEVEL_KEYS, cnt)),
        "conflict_5level": len(set(levels)) > 1, "conflict_3bin": len({bin3(x) for x in levels}) > 1,
        "label_source": {"germline_classification": f"votes:{n}", "clinically_actionable": f"votes:{n}"},
        "label_method": "dm_posterior_mean;actionable=P+LP",
        "split_reason": SPLIT_REASON[c["split"]], "n_votes_pre_cutoff": c["n_pre"],
        "strictly_prospective": c["strict_prosp"], "expert_panel": exp[-1]["level"] if exp else None,
        "surface": surface, "sample_weight": round(keep_w, 3),
    }
    validate(rec)
    vote_rows = [(rid, c["split"], v["org"], v["level"], v["date"].isoformat() if v["date"] else "", v["scv"],
                  v["method"]) for v in votes]
    return rec, cnt, vote_rows


def stratum(levels):
    if len({bin3(x) for x in levels}) > 1:
        return "conflict_3bin"
    return "conflict_5level_only" if len(set(levels)) > 1 else "concordant"


def build_clinvar(a, out, writer, man):
    cache = out / "_cache"
    log("loading extracts ...")
    variants, vcf = load_variants(cache), load_vcf(cache)
    cui_orpha, orpha = load_orpha(cache)
    log(f"  variants {len(variants['rows'])}, vcf {len(vcf['rows'])}, cui->orphanet {len(cui_orpha)}, "
        f"orphanet disorders {len(orpha)}")
    keep = {"conflict_3bin": a.keep_conflict3, "conflict_5level_only": a.keep_conflict5,
            "concordant": a.keep_concordant}

    # pass 1: train counts for the DM fit (all train pairs, before any subsampling) + population statistics
    drops, diag = Counter(), Counter()
    train_counts, pop, gene_expected = [], Counter(), Counter()
    for c in clinvar_candidates(cache, variants, vcf, cui_orpha, orpha, drops, diag):
        st = stratum([v["level"] for v in c["votes"]])
        pop[(c["split"], "pairs")] += 1
        pop[(c["split"], "stratum:" + st)] += 1
        pop[(c["split"], "conflict_5level")] += st != "concordant"
        pop[(c["split"], "conflict_3bin")] += st == "conflict_3bin"
        if c["split"] == "train":
            train_counts.append(counts_of(c["votes"]))
            gene_expected[c["vf"]["gene"]] += keep[st]
    # per-gene cap as a uniform within-gene sampling rate (not "first N by id", which would favour old variants)
    gene_rate = {g: min(1.0, a.max_per_gene / e) for g, e in gene_expected.items()}
    alpha, m, ll = dm_fit(train_counts)
    log(f"DM fit on {len(train_counts)} train pairs: a = {alpha:.4f}, m = {[round(x, 4) for x in m]}")
    eval_rate = {"test": min(1.0, a.max_test / max(1, pop[("test", "pairs")])),
                 "test_temporal": min(1.0, a.max_test_temporal / max(1, pop[("test_temporal", "pairs")]))}

    # pass 2: records (+ vote provenance in a side file outside the split directories)
    rng_drops, scan_hits = Counter(), Counter()
    written, sl, gene_n = Counter(), defaultdict(list), Counter()
    agree = {s: Agreement(alpha, m) for s in SPLITS}
    seen_state = set()
    votes_f = gzip.open(out / "clinvar_votes.tsv.gz", "wt", encoding="utf-8")
    votes_f.write("record_id\tsplit\torganisation\tlevel\tdate_last_evaluated\tscv\tcollection_method\n")
    for c in clinvar_candidates(cache, variants, vcf, cui_orpha, orpha, Counter(), Counter()):
        st = stratum([v["level"] for v in c["votes"]])
        sp = c["split"]
        q = keep[st] if sp in ("train", "val") else eval_rate[sp]
        u = frac(f"keep:{c['vid']}|{c['ck']}")
        if u >= q:
            rng_drops[f"{sp}_subsampled"] += 1
            continue
        if sp == "train":
            if u >= q * gene_rate[c["vf"]["gene"]]:
                rng_drops["train_gene_cap"] += 1
                continue
            q *= gene_rate[c["vf"]["gene"]]
        cf = condition_facts(c["ck"], c["names"], cui_orpha, orpha)
        rec, cnt, vote_rows = clinvar_record(c, cf, alpha, m, 1.0 / q)
        hit = OUTCOME_WORDS.search(rec["state"])
        if hit:
            scan_hits[hit.group(0).lower()] += 1
            continue
        if rec["state"] in seen_state:
            rng_drops["duplicate_state"] += 1
            continue
        seen_state.add(rec["state"])
        if sp == "train":
            gene_n[c["vf"]["gene"]] += 1
        writer.write(sp, "clinvar", rec)
        for row in vote_rows:
            votes_f.write("\t".join(str(x).replace("\t", " ") for x in row) + "\n")
        written[sp] += 1
        w = 1.0 / q
        agree[sp].add(cnt, w)
        n = sum(cnt)
        g = [rec["labels"]["germline_classification"][x] for x in GERMLINE_LEVELS]
        y = rec["labels"]["clinically_actionable"]["yes"]
        ry = (cnt[3] + cnt[4]) / n
        sl[(sp, "germline_classification", "smoothed")].append(g)
        sl[(sp, "germline_classification", "raw")].append([x / n for x in cnt])
        sl[(sp, "clinically_actionable", "smoothed")].append([y, 1 - y])
        sl[(sp, "clinically_actionable", "raw")].append([ry, 1 - ry])
        sl[(sp, "w")].append(w)
        sl[(sp, "stratum")].append(st)
        sl[(sp, "n")].append(n)
        sl[(sp, "surface")].append(rec["meta"]["surface"])
        sl[(sp, "cond_none")].append(c["ck"] == NO_CONDITION)
        sl[(sp, "inheritance")].append(bool(cf["inheritance"]))
        sl[(sp, "prosp")].append(c["strict_prosp"])
    votes_f.close()
    log(f"clinvar written: {dict(written)}; outcome-word drops {dict(scan_hits)}")

    qk = [(q, k) for q in ("germline_classification", "clinically_actionable") for k in ("smoothed", "raw")]
    splits = {}
    for s in SPLITS:
        if not pop[(s, "pairs")]:
            continue
        sts, ws = sl[(s, "stratum")], sl[(s, "w")]
        nrec = max(1, len(sts))
        splits[s] = {
            "population_pairs": pop[(s, "pairs")],
            "population_strata": {k: pop[(s, "stratum:" + k)] for k in keep},
            "population_conflict_rate_5level": round(pop[(s, "conflict_5level")] / pop[(s, "pairs")], 4),
            "population_conflict_rate_3bin": round(pop[(s, "conflict_3bin")] / pop[(s, "pairs")], 4),
            "records": written[s],
            "records_strata": dict(Counter(sts)),
            "records_conflict_rate_5level": round(sum(1 for x in sts if x != "concordant") / nrec, 4),
            "records_conflict_rate_3bin": round(sum(1 for x in sts if x == "conflict_3bin") / nrec, 4),
            "raters_per_record": dict(sorted(Counter(min(x, 10) for x in sl[(s, "n")]).items())),
            "mean_raters": round(statistics.fmean(sl[(s, "n")]), 3) if sts else None,
            "share_no_specific_condition": round(statistics.fmean(sl[(s, "cond_none")]), 4) if sts else None,
            "share_with_orphanet_inheritance": round(statistics.fmean(sl[(s, "inheritance")]), 4) if sts else None,
            "share_json_surface": round(sum(1 for x in sl[(s, "surface")] if x == "json") / nrec, 4),
            "strictly_prospective_records": sum(sl[(s, "prosp")]),
            "rater_agreement(population-weighted)": agree[s].result(),
            "label_stats_records": {f"{q}/{k}": label_stats(sl[(s, q, k)]) for q, k in qk},
            "label_stats_population_weighted": {f"{q}/{k}": label_stats(sl[(s, q, k)], ws) for q, k in qk},
        }
    man["clinvar"] = {
        "license": LICENSE_CLINVAR, "cutoff": CUTOFF.isoformat(),
        "dirichlet": {"a": round(alpha, 5), "m": dict(zip(LEVEL_KEYS, [round(x, 5) for x in m])),
                      "train_pairs": len(train_counts), "loglik": round(ll, 2)},
        "keep_rates_train_val": keep, "eval_rates": {k: round(v, 4) for k, v in eval_rate.items()},
        "max_per_gene": a.max_per_gene,
        "drops_candidate": dict(sorted(drops.items(), key=lambda kv: -kv[1])),
        "drops_record": dict(rng_drops), "outcome_word_drops": dict(scan_hits),
        "family_collapse": dict(diag), "splits": splits,
        "train_genes_written": len(gene_n), "top_train_genes": gene_n.most_common(15),
        "votes_side_file": "clinvar_votes.tsv.gz (record_id, split, organisation, level, date, SCV, method)",
    }


# --------------------------------------------------------------------------- CIViC
def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def civic_profiles(cache):
    """molecular_profile_id -> {genes, variant_types, hgvs}"""
    var = {r["variant_id"]: r for r in read_tsv(cache / "civic_variants.tsv")}
    out = {}
    for r in read_tsv(cache / "civic_profiles.tsv"):
        genes, vtypes, hgvs = set(), set(), []
        for vid in [x.strip() for x in (r.get("variant_ids") or "").split(",") if x.strip()]:
            v = var.get(vid)
            if not v:
                continue
            for g in (v.get("gene"), v.get("5_prime_partner"), v.get("3_prime_partner")):
                if g and g.strip() and g.strip().lower() not in ("unknown", "multiple"):
                    genes.add(g.strip())
            if v.get("feature_type") == "Gene" and v.get("feature_name"):
                genes.add(v["feature_name"].strip())
            vtypes.update(human(t) for t in (v.get("variant_types") or "").split(",") if t.strip())
            hgvs += [h.strip() for h in (v.get("hgvs_descriptions") or "").split(",")
                     if h.strip() and not h.strip().startswith("ENST")][:2]
        out[r["molecular_profile_id"]] = {"genes": sorted(genes), "variant_types": sorted(vtypes), "hgvs": hgvs[:4]}
    return out


def civic_year(citation):
    m = re.search(r"(\d{4})\s*$", citation or "")
    return int(m.group(1)) if m else None


def civic_state(it, prof, surface):
    therapies = [t.strip() for t in (it["therapies"] or "").split(",") if t.strip()]
    phen = [t.strip() for t in (it["phenotypes"] or "").split(",") if t.strip()]
    nct = [t.strip() for t in (it["nct_ids"] or "").split(",") if t.strip()]
    src_kind = {"PubMed": "journal article (PubMed)", "ASCO": "ASCO meeting abstract",
                "ASH": "ASH meeting abstract"}.get(it["source_type"], it["source_type"])
    year = civic_year(it["citation"])
    if surface == "json":
        obj = {"clinical_context": {"disease": it["disease"] or None, "phenotypes": phen or None,
                                    "therapies": therapies or None,
                                    "therapy_interaction": it["therapy_interaction_type"] or None,
                                    "variant_origin": it["variant_origin"] or None},
               "evidence_item": {"evidence_type": it["evidence_type"],
                                 "source": {k: v for k, v in (("kind", src_kind), ("year", year),
                                                              ("clinical_trials", nct)) if v}},
               "molecular_profile": {"name": it["molecular_profile"], "genes": prof["genes"] or None,
                                     "variant_types": prof["variant_types"] or None, "hgvs": prof["hgvs"] or None}}
        for k in obj:
            obj[k] = {kk: vv for kk, vv in obj[k].items() if vv is not None}
        return jdump(obj)
    lines = ["Cancer variant knowledgebase: evidence item to curate",
             f"Evidence type: {it['evidence_type']}",
             f"Molecular profile: {it['molecular_profile']}"]
    if prof["genes"]:
        lines.append(f"Gene(s): {', '.join(prof['genes'])}")
    if prof["variant_types"]:
        lines.append(f"Variant type: {', '.join(prof['variant_types'])}")
    if prof["hgvs"]:
        lines.append(f"HGVS: {'; '.join(prof['hgvs'])}")
    lines.append(f"Variant origin: {it['variant_origin'] or 'not given'}")
    lines.append(f"Disease: {it['disease'] or 'not given'}")
    if phen:
        lines.append(f"Phenotypes: {', '.join(phen)}")
    if therapies:
        lines.append(f"Therapies: {', '.join(therapies)}" + (f" ({it['therapy_interaction_type'].lower()})"
                                                              if it["therapy_interaction_type"] else ""))
    lines.append(f"Source: {src_kind}" + (f", {year}" if year else ""))
    if nct:
        lines.append(f"Registered clinical trials: {', '.join(nct)}")
    return "\n".join(lines)


def civic_sig_key(it):
    slug = CIVIC_SIG_SLUG.get(it["significance"])
    t = it["evidence_type"]
    if slug is None or t not in CIVIC_SIG:
        return None
    if t == "Prognostic" and (slug == "na" or it["evidence_direction"] == "Does Not Support"):
        return "not_prognostic"
    key = slug if it["evidence_direction"] == "Supports" else (
        f"not_{slug}" if it["evidence_direction"] == "Does Not Support" else None)
    return key if key in CIVIC_SIG[t] else None


def hier_rate(levels_counts, keys, ks):
    """Hierarchical Dirichlet-smoothed rate: levels_counts = [Counter at root, ..., Counter at leaf];
    each level shrinks toward its parent with prior strength ks[i]."""
    K = len(keys)
    p = [1.0 / K] * K
    for i, c in enumerate(levels_counts):
        n = sum(c.get(k, 0) for k in keys)
        p = [(c.get(k, 0) + ks[i] * p[j]) / (n + ks[i]) for j, k in enumerate(keys)]
    return p


def build_civic(a, out, writer, man):
    cache = out / "_cache"
    profs = civic_profiles(cache)
    items = read_tsv(cache / "civic_evidence.tsv")
    drops = Counter()
    # temporal: evidence ids are allocated in creation order; an item created after the cutoff has an id above every
    # item last reviewed before it.
    pre_ids = [int(r["evidence_id"]) for r in items if (r["last_review_date"] or "9999")[:10] < CUTOFF.isoformat()]
    id_cut = max(pre_ids) if pre_ids else 0
    groups = {}
    for it in items:
        if it["evidence_status"] != "accepted":
            drops["not_accepted"] += 1
            continue
        if it.get("is_flagged") == "true":
            drops["flagged"] += 1
            continue
        if it["evidence_level"] not in CIVIC_LEVEL_KEYS:
            drops["bad_level"] += 1
            continue
        prof = profs.get(it["molecular_profile_id"], {"genes": [], "variant_types": [], "hgvs": []})
        genes = prof["genes"] or [it["molecular_profile"].split()[0]]
        sp = {gene_split(g) for g in genes}
        if len(sp) > 1:
            drops["genes_straddle_splits"] += 1
            continue
        split = sp.pop()
        if split == "test" and int(it["evidence_id"]) > id_cut:
            split = "test_temporal"
        rid = f"civic-{sha(it['evidence_id'])}"
        surface = "json" if frac("surface:" + rid) * 100 < JSON_SURFACE_PCT else "text"
        state = civic_state(it, prof, surface)
        words = re.compile(f"({CIVIC_GLOBAL_WORDS}|{CIVIC_TYPE_WORDS.get(it['evidence_type'], 'a^')})", re.I)
        hit = words.search(state.replace(f'"evidence_type": "{it["evidence_type"]}"', "")
                           .replace(f"Evidence type: {it['evidence_type']}", ""))
        if hit:
            drops[f"outcome_word:{hit.group(0).lower()}"] += 1
            continue
        g = groups.setdefault(state, {"items": [], "split": split, "genes": genes, "rid": rid, "surface": surface,
                                      "type": it["evidence_type"], "profile": it["molecular_profile_id"],
                                      "disease": it["disease"]})
        g["items"].append(it)
        if split == "test_temporal" and g["split"] == "test":
            g["split"] = "test"   # merged with an older identical state: not prospective
    # hierarchical LOO base rates from TRAIN items only
    def path(g):
        return [("type", g["type"]), ("gene", g["type"], g["genes"][0]), ("prof", g["type"], g["profile"]),
                ("pd", g["type"], g["profile"], g["disease"])]
    tables = {"level": defaultdict(Counter), "sig": defaultdict(Counter)}
    for g in groups.values():
        if g["split"] != "train":
            continue
        for it in g["items"]:
            for node in path(g):
                tables["level"][node][it["evidence_level"]] += 1
                sk = civic_sig_key(it)
                if sk:
                    tables["sig"][node][sk] += 1

    def loo_rate(g, which, keys, own, ks):
        cs = []
        for node in path(g):
            c = Counter(tables[which].get(node, Counter()))
            if g["split"] == "train":
                for k, v in own.items():
                    c[k] -= v
            cs.append(c)
        return hier_rate(cs, keys, ks)

    def choose_ks(which):
        grid = [0.5, 1, 2, 4, 8, 16]
        best, best_ll = None, -1e18
        for k1 in grid:
            for k2 in grid:
                ks = [1.0, k1, k2, k2]
                ll = 0.0
                for g in groups.values():
                    if g["split"] != "train":
                        continue
                    for it in g["items"]:
                        y = it["evidence_level"] if which == "level" else civic_sig_key(it)
                        if y is None:
                            continue
                        keys = CIVIC_LEVEL_KEYS if which == "level" else list(CIVIC_SIG[g["type"]])
                        p = loo_rate(g, which, keys, Counter({y: 1}), ks)
                        ll += math.log(max(p[keys.index(y)], 1e-12))
                if ll > best_ll:
                    best, best_ll = ks, ll
        return best, best_ll
    ks_level, ll_level = choose_ks("level")
    ks_sig, ll_sig = choose_ks("sig")
    A = 1.0
    written, labst = Counter(), defaultdict(list)
    for state, g in groups.items():
        lv = Counter(it["evidence_level"] for it in g["items"])
        sg = Counter(k for k in (civic_sig_key(it) for it in g["items"]) if k)
        n = len(g["items"])
        qs, labs = {}, {}
        p = loo_rate(g, "level", CIVIC_LEVEL_KEYS, lv, ks_level)
        t = [(lv.get(k, 0) + A * p[j]) / (n + A) for j, k in enumerate(CIVIC_LEVEL_KEYS)]
        qs["evidence_level"] = score_q(CIVIC_LEVEL_INSTR, CIVIC_LEVELS)
        labs["evidence_level"] = {CIVIC_LEVELS[j]: t[j] for j in range(5)}
        keys = list(CIVIC_SIG[g["type"]])
        if sum(sg.values()):
            ns = sum(sg.values())
            p2 = loo_rate(g, "sig", keys, sg, ks_sig)
            t2 = [(sg.get(k, 0) + A * p2[j]) / (ns + A) for j, k in enumerate(keys)]
            order = sorted(keys)                       # alphabetical choice keys, as the benchmark
            qs["evidence_significance"] = choice_q(CIVIC_SIG_INSTR[g["type"]], {k: CIVIC_SIG[g["type"]][k]
                                                                                for k in order})
            labs["evidence_significance"] = {k: t2[keys.index(k)] for k in order}
        else:
            drops["significance_unmapped(level only)"] += 1
        rec = record("civic", state, qs, labs, {q: "soft" for q in qs}, None)
        rec["id"] = rec["base_id"] = g["rid"]
        rec["license"] = LICENSE_CIVIC
        rec["meta"] = {
            "dataset": "civic nightly 2026-10-01",
            "evidence_ids": [int(it["evidence_id"]) for it in g["items"]],
            "citation_ids": sorted({it["citation_id"] for it in g["items"]}),
            "genes": g["genes"], "group": g["genes"][0], "evidence_type": g["type"], "n_raters": n,
            "counts": {"evidence_level": dict(lv), "evidence_significance": dict(sg)},
            "label_source": {q: f"expert:{n}" for q in qs},
            "label_method": {q: "loo_hier_rate_blend(a=1)" for q in qs},
            "split_reason": {"train": "train_gene", "val": "val_gene", "test": "test_gene",
                             "test_temporal": f"test_gene;created_after_cutoff(evidence_id>{id_cut})"}[g["split"]],
            "surface": g["surface"], "sample_weight": 1.0,
        }
        validate(rec)
        writer.write(g["split"], "civic", rec)
        written[g["split"]] += 1
        labst[(g["split"], "evidence_level")].append(t)
        if "evidence_significance" in labs:
            labst[(g["split"], "evidence_significance")].append(list(labs["evidence_significance"].values()))
    man["civic"] = {
        "license": LICENSE_CIVIC, "items": len(items), "records": dict(written), "drops": dict(drops),
        "temporal_evidence_id_cut": id_cut, "lambda_a": A,
        "prior_strengths": {"evidence_level": ks_level, "evidence_significance": ks_sig},
        "loo_loglik": {"evidence_level": round(ll_level, 2), "evidence_significance": round(ll_sig, 2)},
        "label_stats": {f"{s}/{q}": label_stats(v) for (s, q), v in sorted(labst.items())},
    }
    log(f"civic written: {dict(written)}; drops {dict(drops)}")


# --------------------------------------------------------------------------- build / check
class CompactWriter:
    def __init__(self, out):
        self.out, self.files = Path(out), {}

    def write(self, split, source, rec):
        key = (split, source)
        if key not in self.files:
            path = self.out / split / f"{source}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.files[key] = path.open("w", encoding="utf-8")
        self.files[key].write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")

    def close(self):
        for f in self.files.values():
            f.close()


def exclusion_check():
    spec = importlib.util.spec_from_file_location("exclusion", HERE / "synth_workflows" / "exclusion.py")
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    bad = [q for q in QIDS_BENCH_CHECK if q in ex.EXCLUDED_QIDS]
    assert not bad, f"benchmark qids used: {bad}"
    return {"checked_qids": list(QIDS_BENCH_CHECK), "benchmark_qids_hit": bad}


def build(a):
    out = Path(a.out)
    for s in SPLITS:
        for f in (out / s).glob("*.jsonl") if (out / s).exists() else []:
            f.unlink()
    man = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "exclusion_check": exclusion_check(), "gene_split": {"test_pct": TEST_PCT, "val_pct": VAL_PCT,
                                                                "hash": "sha1('jev-genomics-gene:' + SYMBOL) % 100"}}
    fm = out / "_cache" / "fetch_manifest.json"
    if fm.exists():
        man["fetch"] = json.loads(fm.read_text())
    w = CompactWriter(out)
    try:
        if "clinvar" in a.sources:
            build_clinvar(a, out, w, man)
        if "civic" in a.sources:
            build_civic(a, out, w, man)
    finally:
        w.close()
    (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    log(f"manifest -> {out / 'manifest.json'}")


def load_cache_features():
    spec = importlib.util.spec_from_file_location("cache_features", HERE / "trainers" / "cache_features.py")
    cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf)
    return cf


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def tokens(s):
    return re.findall(r"[a-z]+|\d+", s.lower())


def leak_probe(recs, folds=5, max_n=60000, seed=0):
    """Cross-validated multinomial naive Bayes, state tokens -> 3-bin majority (B/LB, VUS, P/LP), against a
    structured-field baseline (molecular consequence x frequency bucket x no-condition flag, cross-fitted cell rate).
    A BoW model far above the baseline would point at leaked phrases; its top tokens are listed for reading."""
    rng = random.Random(seed)
    recs = [r for r in recs if r["source"] == "clinvar"]
    rng.shuffle(recs)
    recs = recs[:max_n]

    def y_of(r):
        c = r["meta"]["counts"]
        b = [c["B"] + c["LB"], c["VUS"], c["LP"] + c["P"]]
        return max(range(3), key=lambda k: (b[k], -abs(k - 1)))

    def cell(r):
        st = r["state"]
        mc = re.search(r"(missense|nonsense|frameshift|splice[ _](donor|acceptor)|synonymous|intron|stop[ _]gained|"
                       r"inframe|5 prime UTR|3 prime UTR|start[ _]lost)", st)
        af = re.findall(r"(?:exac|ExAC)\"?:? ([0-9][0-9.e+-]*[0-9]|not reported|null)", st)
        afb = "none" if not af or af[0] in ("null", "not reported") else (
            "common" if float(af[0]) >= 0.01 else "rare" if float(af[0]) >= 0.0001 else "ultrarare")
        nocond = "no specific condition" in st or "none named" in st
        return (mc.group(1) if mc else "other", afb, nocond)
    ys = [y_of(r) for r in recs]
    toks = [set(tokens(r["state"])) for r in recs]
    cells = [cell(r) for r in recs]
    fold = [i % folds for i in range(len(recs))]
    acc_nb = acc_base = acc_major = 0
    for f in range(folds):
        tr = [i for i in range(len(recs)) if fold[i] != f]
        te = [i for i in range(len(recs)) if fold[i] == f]
        prior = Counter(ys[i] for i in tr)
        tc = [Counter() for _ in range(3)]
        for i in tr:
            tc[ys[i]].update(toks[i])
        vocab = set().union(*tc)
        V = len(vocab)
        totals = [sum(c.values()) for c in tc]
        cc = defaultdict(Counter)
        for i in tr:
            cc[cells[i]][ys[i]] += 1
        maj = prior.most_common(1)[0][0]
        for i in te:
            sc = []
            for k in range(3):
                s = math.log(prior[k] + 1)
                for t in toks[i]:
                    if t in vocab:
                        s += math.log((tc[k][t] + 1) / (totals[k] + V))
                sc.append(s)
            acc_nb += max(range(3), key=lambda k: sc[k]) == ys[i]
            cel = cc.get(cells[i])
            acc_base += (cel.most_common(1)[0][0] if cel else maj) == ys[i]
            acc_major += maj == ys[i]
    n = len(recs)
    # top discriminating tokens, P/LP vs B/LB (log-odds with +1 smoothing, min count 50)
    c0, c2 = Counter(), Counter()
    for t, y in zip(toks, ys):
        if y == 0:
            c0.update(t)
        elif y == 2:
            c2.update(t)
    n0, n2 = max(1, ys.count(0)), max(1, ys.count(2))
    lo = {t: math.log((c2[t] + 1) / (n2 + 2)) - math.log((c0[t] + 1) / (n0 + 2))
          for t in set(c0) | set(c2) if c0[t] + c2[t] >= 50}
    top_p = sorted(lo, key=lambda t: -lo[t])[:25]
    top_b = sorted(lo, key=lambda t: lo[t])[:25]
    return {"n": n, "class_share(B/LB, VUS, P/LP)": [round(ys.count(k) / n, 4) for k in range(3)],
            "acc_majority": round(acc_major / n, 4), "acc_structured_cells": round(acc_base / n, 4),
            "acc_bag_of_words_nb": round(acc_nb / n, 4),
            "top_tokens_toward_P_LP": [(t, round(lo[t], 2)) for t in top_p],
            "top_tokens_toward_B_LB": [(t, round(lo[t], 2)) for t in top_b]}


def check(a):
    out = Path(a.out)
    cf = load_cache_features()
    recs_by = defaultdict(list)
    n_rec = Counter()
    states = defaultdict(set)
    genes = defaultdict(set)
    for s in SPLITS:
        for p in sorted((out / s).glob("*.jsonl")) if (out / s).exists() else []:
            for r in read_jsonl(p):
                validate(r)
                for qid, q in r["questions"].items():
                    keys, texts, target = cf.question_options(q, r["labels"][qid])
                    assert keys == option_keys(q), (r["id"], qid)
                    assert [r["labels"][qid][k] for k in keys] == target, (r["id"], qid, "round trip")
                    assert abs(sum(target) - 1) < 1e-6 and 2 <= len(target) <= 52
                hit = OUTCOME_WORDS.search(r["state"]) if r["source"] == "clinvar" else None
                assert not hit, (r["id"], hit.group(0))
                n_rec[(r["source"], s)] += 1
                states[s].add(r["state"])
                g = r["meta"].get("genes") or [r["meta"]["gene"]]
                genes[s].update(x.upper() for x in g)
                recs_by[(r["source"], s)].append(r)
    units = Counter()
    for u in cf.iter_units(str(out), list(SPLITS), set(), None):
        assert abs(sum(u["target"]) - 1.0) < 1e-6
        units[(u["source"], u["split"], u["qtype"])] += 1
    overlap_states = {f"{x}&{y}": len(states[x] & states[y]) for i, x in enumerate(SPLITS) for y in SPLITS[i + 1:]}
    overlap_genes = {f"{x}&{y}": len(genes[x] & genes[y]) for i, x in enumerate(SPLITS) for y in SPLITS[i + 1:]}
    assert not any(overlap_states.values()), overlap_states
    # test and test_temporal both draw on the held-out genes; every other pair must be gene-disjoint
    assert not any(v for k, v in overlap_genes.items() if k != "test&test_temporal"), overlap_genes
    report = {"records": {f"{s}/{sp}": n for (s, sp), n in sorted(n_rec.items())},
              "units_iter_units": {f"{s}/{sp}/{q}": n for (s, sp, q), n in sorted(units.items())},
              "state_overlap_between_splits": overlap_states, "gene_overlap_between_splits": overlap_genes,
              "disk_bytes": sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
              "corpus_bytes": sum(p.stat().st_size for s in SPLITS for p in (out / s).glob("*.jsonl")
                                  if (out / s).exists())}
    # label statistics pooled over every written unit, vs the typed-decisions gold (0.659 / 0.767)
    pooled = defaultdict(list)
    for (src, sp), rs in recs_by.items():
        for r in rs:
            for qid in r["questions"]:
                keys = option_keys(r["questions"][qid])
                d = [r["labels"][qid][k] for k in keys]
                pooled[(src, qid)].append(d)
                pooled[(src, "all")].append(d)
                pooled[("all", "all")].append(d)
    report["label_stats_all_splits"] = {f"{s}/{q}": label_stats(v) for (s, q), v in sorted(pooled.items())}
    report["typed_decisions_gold"] = {"mean_max_p": 0.659, "mean_entropy_nats": 0.767,
                                      "entropy_by_type": {"choice": 0.864, "noul": 0.454, "score": 0.930}}
    tr = recs_by[("clinvar", "train")] + recs_by[("clinvar", "val")]
    if tr:
        report["leak_probe_clinvar_train_val"] = leak_probe(tr)
    samples = []
    rng = random.Random(7)
    pool = [r for r in recs_by[("clinvar", "train")] if r["meta"]["conflict_3bin"]]
    pool2 = [r for r in recs_by[("clinvar", "train")] if not r["meta"]["conflict_5level"]]
    for r in rng.sample(pool, min(2, len(pool))) + rng.sample(pool2, min(1, len(pool2))) + \
            rng.sample(recs_by[("clinvar", "test_temporal")], min(1, len(recs_by[("clinvar", "test_temporal")]))) + \
            rng.sample(recs_by[("civic", "train")], min(1, len(recs_by[("civic", "train")]))):
        samples.append({"id": r["id"], "source": r["source"], "state": r["state"],
                        "labels": {q: {k[:24]: round(v, 3) for k, v in lab.items()} for q, lab in r["labels"].items()},
                        "meta": {k: r["meta"].get(k) for k in ("n_raters", "counts", "conflict_5level",
                                                               "conflict_3bin", "split_reason", "label_source")}})
    report["samples"] = samples
    (out / "check_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=1, default=str))
    for s in samples:
        print("\n----", s["id"], s["meta"]["split_reason"])
        print(s["state"])
        print(json.dumps(s["labels"], indent=1))
        print(json.dumps(s["meta"], default=str))
    log(f"check passed; report -> {out / 'check_report.json'}")


# --------------------------------------------------------------------------- self-test
def self_test():
    drops = Counter()
    rows = [
        {"ClinicalSignificance": "Pathogenic", "ReviewStatus": VOTE_STATUS, "ReportedPhenotypeInfo": "C3661900:not provided",
         "DateLastEvaluated": "Jan 02, 2024", "OriginCounts": "germline:1", "Submitter": "Labcorp Genetics (formerly Invitae), Labcorp",
         "SCV": "SCV1.2", "CollectionMethod": "clinical testing"},
        {"ClinicalSignificance": "Likely pathogenic", "ReviewStatus": VOTE_STATUS, "ReportedPhenotypeInfo": "CN169374:not specified",
         "DateLastEvaluated": "Feb 02, 2025", "OriginCounts": "germline:1",
         "Submitter": "Women's Health and Genetics/Laboratory Corporation of America, LabCorp", "SCV": "SCV2.1",
         "CollectionMethod": "clinical testing"},
        {"ClinicalSignificance": "Uncertain significance", "ReviewStatus": VOTE_STATUS, "ReportedPhenotypeInfo": "na",
         "DateLastEvaluated": "-", "OriginCounts": "unknown:na", "Submitter": "GeneDx", "SCV": "SCV3.1",
         "CollectionMethod": "clinical testing"},
        {"ClinicalSignificance": "Pathogenic", "ReviewStatus": VOTE_STATUS, "ReportedPhenotypeInfo": "C3661900:not provided",
         "DateLastEvaluated": "Aug 01, 2025", "OriginCounts": "germline:1", "Submitter": "Sema4, Sema4", "SCV": "SCV4.1",
         "CollectionMethod": "clinical testing"},
        {"ClinicalSignificance": "Benign", "ReviewStatus": "reviewed by expert panel",
         "ReportedPhenotypeInfo": "C3661900:not provided", "DateLastEvaluated": "Aug 01, 2025", "OriginCounts": "germline:na",
         "Submitter": "ClinGen VCEP", "SCV": "SCV5.1", "CollectionMethod": "curation"},
        {"ClinicalSignificance": "Pathogenic", "ReviewStatus": "no assertion criteria provided",
         "ReportedPhenotypeInfo": "C123:Some syndrome", "DateLastEvaluated": "-", "OriginCounts": "germline:na",
         "Submitter": "OMIM", "SCV": "SCV6.1", "CollectionMethod": "literature only"},
        {"ClinicalSignificance": "Pathogenic", "ReviewStatus": VOTE_STATUS, "ReportedPhenotypeInfo": "C123:Some syndrome",
         "DateLastEvaluated": "Mar 03, 2023", "OriginCounts": "somatic:1", "Submitter": "Lab X", "SCV": "SCV7.1",
         "CollectionMethod": "clinical testing"},
    ]
    pairs = pairs_of_variant(rows, drops)
    assert set(pairs) == {NO_CONDITION}, pairs.keys()     # C123 had no usable vote (no criteria / somatic)
    p = pairs[NO_CONDITION]
    assert len(p["votes"]) == 4 and len(p["expert"]) == 1
    cur = collapse(p["votes"])
    fams = sorted(v["family"] for v in cur)
    assert fams == ["genedx", "labcorp"], fams           # Labcorp x2 -> 1, GeneDx + Sema4 -> 1
    lab = {v["family"]: v["level"] for v in cur}
    assert lab == {"labcorp": "LP", "genedx": "P"}, lab  # the latest vote per family wins (Sema4 2025-08 > GeneDx undated)
    pre = collapse([v for v in p["votes"] if v["date"] is None or v["date"] < CUTOFF])
    assert {v["family"]: v["level"] for v in pre} == {"labcorp": "LP", "genedx": "VUS"}
    assert drops["scv_status:no assertion criteria provided"] == 1 and drops["scv_somatic_only"] == 1
    assert family_of("Myriad Genetics, Inc.") == "myriad" and family_of("Genomenon, Inc, Genomenon, Inc") == "genomenon"
    assert family_of("Institute of Human Genetics, University of Leipzig Medical Center") == \
        "university of leipzig medical center"
    # DM fit recovers a strongly concordant population (small a) and smoothing keeps mass everywhere
    rng = random.Random(0)
    cv = []
    for _ in range(3000):
        k = rng.choice([0, 1, 2, 2, 3, 4])
        n = rng.choice([2, 2, 3, 4])
        c = [0] * 5
        for _ in range(n):
            c[k if rng.random() < 0.85 else rng.randrange(5)] += 1
        cv.append(c)
    a_hat, m_hat, _ = dm_fit(cv)
    assert 0.05 < a_hat < 20, a_hat
    g = smooth([0, 0, 0, 2, 0], a_hat, m_hat)
    assert abs(sum(g) - 1) < 1e-9 and min(g) > 0 and max(g) == g[3]
    # state + record + outcome-word scan
    vf = {"gene": "BRCA2", "chromosome": "13", "cytoband": "13q13.1", "hgvs": "NM_000059.4(BRCA2):c.68-7T>A",
          "protein_change": None, "type": "single nucleotide variant", "genomic_hgvs": "NC_000013.11:g.32316455T>A",
          "molecular_consequence": ["intron variant"], "af": {"exac": 0.00012, "go_esp": None, "1000_genomes": None},
          "allele_origin": ["germline"], "acmg_sf_gene": True, "allele_id": "1", "rs": "-1", "assembly": "GRCh38"}
    for surf in ("json", "text"):
        cfacts = condition_facts(NO_CONDITION, {}, {}, {})
        st = clinvar_state(vf, cfacts, surf)
        assert not OUTCOME_WORDS.search(st), (surf, st)
        if surf == "json":
            assert json.loads(st)["gene"] == "BRCA2" and "genomic" not in st
    assert OUTCOME_WORDS.search("Condition: Benign familial neonatal seizures")
    assert not OUTCOME_WORDS.search("Condition: Stargardt disease 1")
    c = {"vid": 1, "ck": NO_CONDITION, "names": {}, "split": "train", "votes": cur, "vf": vf, "n_pre": 2, "n_cur": 2,
         "expert": p["expert"], "strict_prosp": False}
    rec, cnt, vrows = clinvar_record(c, condition_facts(NO_CONDITION, {}, {}, {}), a_hat, m_hat, 1.0)
    assert cnt == [0, 0, 0, 1, 1] and rec["meta"]["conflict_5level"] and not rec["meta"]["conflict_3bin"]
    assert len(vrows) == 2 and rec["meta"]["expert_panel"] == "B" and stratum(["LP", "P"]) == "conflict_5level_only"
    assert sum(round_dist([0.1234567, 0.2345678, 0.6419755])) == 1.0
    y = rec["labels"]["clinically_actionable"]["yes"]
    assert abs(y - sum(rec["labels"]["germline_classification"][GERMLINE_LEVELS[k]] for k in (3, 4))) < 1e-9
    assert split_name("NM_1.1(X):c.1A>T (p.Met1Leu)") == ("NM_1.1(X):c.1A>T", "p.Met1Leu")
    assert split_name("NM_1.1(X):c.-5A>T") == ("NM_1.1(X):c.-5A>T", None)
    # CIViC significance keys and hierarchical rate
    it = {"significance": "Resistance", "evidence_type": "Predictive", "evidence_direction": "Does Not Support"}
    assert civic_sig_key(it) == "not_resistance"
    it2 = {"significance": "N/A", "evidence_type": "Prognostic", "evidence_direction": "Does Not Support"}
    assert civic_sig_key(it2) == "not_prognostic"
    pr = hier_rate([Counter({"A": 3, "B": 1}), Counter({"A": 0, "B": 2})], ["A", "B"], [1.0, 2.0])
    assert abs(sum(pr) - 1) < 1e-9 and pr[1] > pr[0]
    assert gene_split("BRCA2") == gene_split("brca2")
    exclusion_check()
    print("self-test OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--out", default="/tmp/genomics_jev")
    f.add_argument("--skip-clinvar", action="store_true", help="only refresh the small CIViC / Orphanet files")
    b = sub.add_parser("build")
    b.add_argument("--out", default="/tmp/genomics_jev")
    b.add_argument("--sources", default="clinvar,civic")
    b.add_argument("--keep-conflict3", type=float, default=0.5,
                   help="train/val keep rate: pairs whose votes span P/LP vs VUS vs B/LB")
    b.add_argument("--keep-conflict5", type=float, default=0.15,
                   help="train/val keep rate: pairs that disagree only within a bin (P vs LP, B vs LB)")
    b.add_argument("--keep-concordant", type=float, default=0.04, help="train/val keep rate: unanimous pairs")
    b.add_argument("--max-test", type=int, default=8000, help="uniform cap on the gene-held-out `test` split")
    b.add_argument("--max-test-temporal", type=int, default=5000, help="uniform cap on `test_temporal`")
    b.add_argument("--max-per-gene", type=int, default=1200,
                   help="expected train records per gene (TTN, BRCA1/2, ...), applied as a within-gene rate")
    c = sub.add_parser("check")
    c.add_argument("--out", default="/tmp/genomics_jev")
    sub.add_parser("self-test")
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a)
    elif a.cmd == "build":
        a.sources = [s.strip() for s in a.sources.split(",") if s.strip()]
        build(a)
    elif a.cmd == "check":
        check(a)
    else:
        self_test()


if __name__ == "__main__":
    main()
