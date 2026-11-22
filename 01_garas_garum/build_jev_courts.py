#!/usr/bin/env python3
"""
build_jev_courts.py -- real-world "adjudication outcomes" records (design: research/realworld_data.md, P6 and
§3.1-3.2), built from public bulk data and written in the jev_data record schema (build_jev_data.record /
build_jev_ext2.make_record; read back through trainers/cache_features.iter_units).

Two record families, one `source` per court ("courts/<CourtListener court id>"):

  courts/ca1 .. ca11, cadc   U.S. courts of appeals, civil + agency + bankruptcy appeals decided on the merits by a
                             three-judge panel, FY2008-present.
        labels    FJC Integrated Database, appeals file (ap08on, OUTCOME + per-judge involvement codes JDG1-3INV)
        context   FJC civil IDB (cv88on; the district case as it stood when the appeal was docketed: how it ended,
                  for whom, stage reached, cause statute, pro se / IFP) joined on district + docket number
        links     CourtListener bulk 2026-09-30 (dockets, originating-court info, opinion clusters, panels):
                  CourtListener ids for provenance and knowledge probes, and an independent check of the FJC
                  disposition against CourtListener's own `disposition` text
  courts/<state court id>    state courts of last resort, civil cases decided 1995-1998
        labels    State Supreme Court Data Project (Harvard Dataverse doi:10.7910/DVN/Z80F7P, CC0): disposition,
                  every justice's vote
        links     CourtListener citations + clusters (reporter / volume / page) for provenance

Questions (qids are not benchmark qids; checked with synth_workflows.exclusion.exclusion_errors):
  appellate_disposition   choice 5   affirm / reverse_or_vacate / affirm_in_part / remand_only / dismiss  (federal)
  panel_dissent           noul       a judge on the three-judge panel dissents in whole or in part        (federal)
  degree_of_change        score 3    none < partial < full change to the judgment below                   (both)
  state_supreme_disposition choice 4 affirm / reverse / reverse_and_remand / affirm_in_part               (state)
  justice_votes_for_petitioner noul  a justice of the court votes for the party that sought review        (state)
  court_divided           noul       at least one justice votes against the majority                      (state)

Labels (meta.label_source[qid] carries the provenance tag; meta.label_method[qid] the exact construction):
  outcome (O)   t = lam * onehot(realised) + (1 - lam) * p_hat, lam = 0.5 by default. p_hat is a hierarchical
                Dirichlet-smoothed base rate over global > NOS family > NOS > circuit x NOS > circuit x NOS x year
                (state: global > issue area > state x area > state x area x year), computed LEAVE-ONE-OUT so a
                case's own outcome never enters its own base rate; each level's prior strength is chosen by LOO log
                score. Decider-marginalised: the state never names the panel, and federal panels are drawn from
                the circuit's judges, so a circuit x NOS x year rate is sum_j w_j P(y | cell, panel j) with w_j the
                empirical panel-assignment share in that cell (P6).
  votes (V)     federal dissent: the three judges' involvement codes (JDG1-3INV) give the vote split; the realised
                "at least one dissenter" is smoothed exactly like an outcome (lam blend with a LOO hierarchical rate).
                A per-judge beta-binomial "fresh panel" label was fitted and rejected: a three-judge panel has at
                most one dissenter, so overdispersion is unidentified (a -> grid bound) and the label collapses to
                the base rate whatever the panel did (numbers in the manifest).
                state votes: g = (c + a m) / (n + a), the Dirichlet(-Beta) posterior mean of the justices' vote
                share (§3.1), m the LOO cell rate, a by beta-binomial marginal likelihood (5-9 justices).
  EVAL records carry the ONE-HOT realised outcome (label_kind hard); the smoothed target and the train-only base
  rate sit in meta for comparison.

As-of rule. The state holds only what existed before the appellate decision: the appeal's docketing month, type,
whether it was argued or submitted on the briefs, related appeals, the lower court and how the case ended there
(only if that district termination is dated on or before the appeal was docketed), nature of suit, cause statute,
parties' kinds and roles (never names), fee / pro se status, the agency for agency appeals. Never: the appellate
judgment date, publication status, opinion text, syllabus, disposition, judges' votes or identities.
Every rendered state passes an outcome-word scan (OUTCOME_WORDS); hits are masked and counted, and a record with a
hit in free text is dropped. `check` adds a cross-validated bag-of-words leak probe against the base-rate model.

Splits (group = circuit + lead docket, so cross- and consolidated appeals stay together):
  train  decided <= 2023 (minus val);  val  10% hash sample of the same years;  eval  decided 2025-2026.
  2024 decisions are not written (buffer year). State-court cases (1995-98) are train / val only.
  Groups never straddle splits; eval states identical to a train state are dropped.

Usage (laptop is fine; ~8 GB is streamed, nothing large is stored; outputs < 500 MB):
  python3 build_jev_courts.py fetch  --cache /tmp/courts_jev/cache       # stream bulk sources -> small extracts
  python3 build_jev_courts.py build  --cache /tmp/courts_jev/cache --out /tmp/courts_jev
  python3 build_jev_courts.py check  --out /tmp/courts_jev                # iter_units read-back, scans, probe
  python3 build_jev_courts.py auth-check                                  # CourtListener API quota (no data use)
  python3 build_jev_courts.py --self-test                                 # no network

Credentials. ~/.courtlistener (mode 600): line 1 username or email, line 2 password, optional line 3 API token.
It is read only inside this process (cl_auth_header), only ever sent to www.courtlistener.com, never printed,
logged or written. HTTP Basic works for the v4 API (verified 2026-10-01); a token, if present, is preferred.
The bulk files and FJC / Dataverse downloads need no credentials.
"""
from __future__ import annotations

import argparse
import base64
import bz2
import csv
import gzip
import hashlib
import io
import json
import math
import os
import random
import re
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_jev_data import choice_q, noul_q, option_keys, record, score_q, sha, validate  # noqa: E402

csv.field_size_limit(1 << 30)

UA = "jev-courts-research/0.1 (research dataset build; bulk data per CourtListener terms)"
CL_BULK = "https://com-courtlistener-storage.s3-us-west-2.amazonaws.com/bulk-data/"
CL_API = "https://www.courtlistener.com/api/rest/v4/"
CL_SNAPSHOT = "2026-09-30"
FJC_APPEALS_URL = "https://www.fjc.gov/sites/default/files/idb/textfiles/ap08on_0.zip"
FJC_CIVIL_URL = "https://www.fjc.gov/sites/default/files/idb/textfiles/cv88on.zip"
SSCDP_URL = "https://dataverse.harvard.edu/api/access/datafile/656541"   # "NSF 95-98 Beta Version_July 2005.tab"
CRED_PATH = Path.home() / ".courtlistener"

LICENSE_FEDERAL = ("Public domain: FJC IDB (US federal work, 17 USC 105) + CourtListener bulk (Public Domain Mark 1.0); "
                   "CourtListener ToS: no FCRA uses")
LICENSE_STATE = ("CC0 1.0: State Supreme Court Data Project (doi:10.7910/DVN/Z80F7P); CourtListener links (PDM 1.0); "
                 "CourtListener ToS: no FCRA uses")
# meta.label_method holds these short codes; the manifest carries the definitions
LABEL_METHODS = {
    "O:lam*onehot+(1-lam)*loo_hier_rate": "lam * onehot(realised outcome) + (1 - lam) * leave-one-out hierarchical "
                                          "Dirichlet-smoothed base rate fitted on the train split (federal: global > NOS "
                                          "family > NOS or agency > circuit x NOS > circuit x NOS x decision year; state: "
                                          "global > issue area > state x area > state x area x year)",
    "O:onehot": "one-hot realised outcome (eval split)",
    "V3:lam*onehot(split)+(1-lam)*loo_hier_rate": "the three judges' involvement codes (FJC JDG1-3INV; C/D/F/G = "
                                                  "dissenting) decide the realised 'at least one dissenter'; smoothed like an "
                                                  "outcome toward a LOO hierarchical case-level dissent rate",
    "V3:onehot(split)": "one-hot 'at least one dissenter' from the vote split (eval split)",
    "V:dirichlet_posterior_mean": "(c + a m) / (n + a): c of n justices vote for the petitioner; m = LOO hierarchical "
                                  "per-justice pro-petitioner share; a fitted by beta-binomial marginal likelihood",
    "derived:sum_disposition_mass_by_change_level": "degree_of_change = disposition label mass summed by change level "
                                                    "(none: affirm, dismiss; partial: mixed, sent back only; full: "
                                                    "reversed / vacated, with or without remand)",
}

LAMBDA = 0.5
VAL_PCT = 10
TRAIN_MAX_YEAR, EVAL_YEARS = 2023, (2025, 2026)
LOW_RES = 1e-9

# --------------------------------------------------------------------------- outcome-word scan
OUTCOME_WORDS = re.compile(
    r"\b(affirm\w*|revers\w*|remand\w*|vacat\w*|dismiss\w*|we hold|holding|held that|dissent\w*|concur\w*|"
    r"overturn\w*|overrul\w*|upheld|uphold\w*|set aside|per curiam|en banc|mandate|sustain\w*|"
    r"granted|denied|denies|grants|judgment of the court of appeals|court of appeals held|prevail\w*|"
    r"won|lost|winning|losing|unanimous\w*|split decision|majority opinion)\b", re.I)


def scan(text):
    return [m.group(0).lower() for m in OUTCOME_WORDS.finditer(text)]


def mask(text):
    return OUTCOME_WORDS.sub("[masked]", text)


# --------------------------------------------------------------------------- code tables (FJC codebooks; federal works)
CIRCUIT_ID = {"0": "cadc", "1": "ca1", "2": "ca2", "3": "ca3", "4": "ca4", "5": "ca5", "6": "ca6", "7": "ca7",
              "8": "ca8", "9": "ca9", "10": "ca10", "11": "ca11"}
CIRCUIT_NAME = {"cadc": "U.S. Court of Appeals for the D.C. Circuit", "ca1": "U.S. Court of Appeals for the First Circuit",
                "ca2": "U.S. Court of Appeals for the Second Circuit", "ca3": "U.S. Court of Appeals for the Third Circuit",
                "ca4": "U.S. Court of Appeals for the Fourth Circuit", "ca5": "U.S. Court of Appeals for the Fifth Circuit",
                "ca6": "U.S. Court of Appeals for the Sixth Circuit", "ca7": "U.S. Court of Appeals for the Seventh Circuit",
                "ca8": "U.S. Court of Appeals for the Eighth Circuit", "ca9": "U.S. Court of Appeals for the Ninth Circuit",
                "ca10": "U.S. Court of Appeals for the Tenth Circuit",
                "ca11": "U.S. Court of Appeals for the Eleventh Circuit"}

# FJC appeals codebook Appendix C (district code -> name), plus the CourtListener court id.
_DISTRICTS = """
90 District of Columbia|dcd;00 Maine|med;01 Massachusetts|mad;02 New Hampshire|nhd;03 Rhode Island|rid;
04 Puerto Rico|prd;05 Connecticut|ctd;06 Northern District of New York|nynd;07 Eastern District of New York|nyed;
08 Southern District of New York|nysd;09 Western District of New York|nywd;10 Vermont|vtd;11 Delaware|ded;
12 New Jersey|njd;13 Eastern District of Pennsylvania|paed;14 Middle District of Pennsylvania|pamd;
15 Western District of Pennsylvania|pawd;91 Virgin Islands|vid;16 Maryland|mdd;17 Eastern District of North Carolina|nced;
18 Middle District of North Carolina|ncmd;19 Western District of North Carolina|ncwd;20 South Carolina|scd;
22 Eastern District of Virginia|vaed;23 Western District of Virginia|vawd;24 Northern District of West Virginia|wvnd;
25 Southern District of West Virginia|wvsd;3L Eastern District of Louisiana|laed;3N Middle District of Louisiana|lamd;
36 Western District of Louisiana|lawd;37 Northern District of Mississippi|msnd;38 Southern District of Mississippi|mssd;
39 Northern District of Texas|txnd;40 Eastern District of Texas|txed;41 Southern District of Texas|txsd;
42 Western District of Texas|txwd;43 Eastern District of Kentucky|kyed;44 Western District of Kentucky|kywd;
45 Eastern District of Michigan|mied;46 Western District of Michigan|miwd;47 Northern District of Ohio|ohnd;
48 Southern District of Ohio|ohsd;49 Eastern District of Tennessee|tned;50 Middle District of Tennessee|tnmd;
51 Western District of Tennessee|tnwd;52 Northern District of Illinois|ilnd;53 Central District of Illinois|ilcd;
54 Southern District of Illinois|ilsd;55 Northern District of Indiana|innd;56 Southern District of Indiana|insd;
57 Eastern District of Wisconsin|wied;58 Western District of Wisconsin|wiwd;60 Eastern District of Arkansas|ared;
61 Western District of Arkansas|arwd;62 Northern District of Iowa|iand;63 Southern District of Iowa|iasd;
64 Minnesota|mnd;65 Eastern District of Missouri|moed;66 Western District of Missouri|mowd;67 Nebraska|ned;
68 North Dakota|ndd;69 South Dakota|sdd;7- Alaska|akd;70 Arizona|azd;71 Northern District of California|cand;
72 Eastern District of California|caed;73 Central District of California|cacd;74 Southern District of California|casd;
75 Hawaii|hid;76 Idaho|idd;77 Montana|mtd;78 Nevada|nvd;79 Oregon|ord;80 Eastern District of Washington|waed;
81 Western District of Washington|wawd;93 Guam|gud;94 Northern Mariana Islands|nmid;82 Colorado|cod;83 Kansas|ksd;
84 New Mexico|nmd;85 Northern District of Oklahoma|oknd;86 Eastern District of Oklahoma|oked;
87 Western District of Oklahoma|okwd;88 Utah|utd;89 Wyoming|wyd;26 Northern District of Alabama|alnd;
27 Middle District of Alabama|almd;28 Southern District of Alabama|alsd;29 Northern District of Florida|flnd;
3A Middle District of Florida|flmd;3C Southern District of Florida|flsd;3E Northern District of Georgia|gand;
3G Middle District of Georgia|gamd;3J Southern District of Georgia|gasd"""
DISTRICTS = {}
for _item in _DISTRICTS.replace("\n", "").split(";"):
    _code, _rest = _item.strip().split(" ", 1)
    _name, _cl = _rest.split("|")
    DISTRICTS[_code] = (("U.S. District Court for the " + _name) if "District" in _name
                        else ("U.S. District Court for the District of " + _name), _cl)

APPTYPE = {"1": "petition for review of a federal agency decision", "2": "application to enforce a federal agency order",
           "3": "civil appeal, United States a party", "4": "civil appeal between private parties",
           "10": "bankruptcy appeal, direct from the bankruptcy court",
           "11": "bankruptcy appeal from a bankruptcy appellate panel",
           "12": "bankruptcy appeal from a district court's bankruptcy ruling"}
JURIS = {"1": "United States as plaintiff", "2": "United States as defendant", "3": "federal question",
         "4": "diversity of citizenship", "5": "local question (territorial court)"}
OUTCOME_KEY = {"1": "affirm", "2": "reverse_or_vacate", "3": "affirm_in_part", "6": "remand_only", "5": "dismiss"}
DISP_KEYS = ["affirm", "reverse_or_vacate", "affirm_in_part", "remand_only", "dismiss"]
DISP_CRITERIA = {
    "affirm": "Affirmed: the court approves the whole judgment or order under review and lets it stand as entered, "
              "including an affirmance that also sends the case back for routine further steps.",
    "reverse_or_vacate": "Reversed or vacated: the court overturns or sets aside the whole judgment or order under "
                         "review, whether or not it also sends the case back for further proceedings.",
    "affirm_in_part": "Mixed: part of the judgment is approved and the rest is overturned or set aside, with or "
                      "without further proceedings below.",
    "remand_only": "Sent back only: the court returns the entire case to the lower court or agency for further "
                   "action without approving or overturning the judgment.",
    "dismiss": "Appeal dismissed after consideration (for example moot, without merit, or frivolous), leaving the "
               "judgment below in place without a ruling on it.",
}
DISP_INSTR = ("How will the court of appeals panel dispose of this appeal? Choose the disposition entered in its "
              "judgment, using the definitions in the options.")
DISSENT_INSTR = ("Will a judge on the three-judge panel dissent from the panel's decision?\n"
                 "Yes means: at least one panel judge writes or joins an opinion dissenting in whole or in part.\n"
                 "No means: all three judges join the decision (separate concurring opinions do not count as a "
                 "dissent).")
CHANGE_LEVELS = ["none: the judgment below stands (affirmed, or the appeal dismissed)",
                 "partial: part of the judgment is changed, or the case goes back without a ruling on it",
                 "full: the judgment below is overturned or set aside in full"]
CHANGE_INSTR = "How much will the appellate decision change the judgment under review?"
CHANGE_OF = {"affirm": 0, "dismiss": 0, "affirm_in_part": 1, "remand_only": 1, "reverse_or_vacate": 2,
             "reverse": 2, "reverse_and_remand": 2}

# JS-44 / FJC Appendix A nature-of-suit codes (civil cover sheet categories)
NOS = {
    "110": "Contract: insurance", "120": "Contract: marine", "130": "Contract: Miller Act", "140": "Contract: negotiable instrument",
    "150": "Contract: recovery of overpayment / enforcement of judgment", "151": "Contract: Medicare Act recovery",
    "152": "Contract: recovery of defaulted student loans", "153": "Contract: recovery of veterans' benefits overpayment",
    "160": "Contract: stockholders' suits", "190": "Contract: other contract", "195": "Contract: contract product liability",
    "196": "Contract: franchise", "210": "Real property: land condemnation", "220": "Real property: foreclosure",
    "230": "Real property: rent, lease and ejectment", "240": "Real property: torts to land",
    "245": "Real property: tort product liability", "290": "Real property: all other real property",
    "310": "Torts: airplane personal injury", "315": "Torts: airplane product liability",
    "320": "Torts: assault, libel and slander", "330": "Torts: Federal Employers' Liability Act",
    "340": "Torts: marine personal injury", "345": "Torts: marine product liability", "350": "Torts: motor vehicle",
    "355": "Torts: motor vehicle product liability", "360": "Torts: other personal injury",
    "362": "Torts: medical malpractice", "365": "Torts: personal injury product liability",
    "367": "Torts: health care / pharmaceutical product liability", "368": "Torts: asbestos product liability",
    "370": "Torts: other fraud", "371": "Torts: Truth in Lending", "375": "Other statutes: False Claims Act",
    "376": "Other statutes: qui tam (31 U.S.C. 3729(a))", "380": "Torts: other personal property damage",
    "385": "Torts: property damage product liability", "400": "Other statutes: state reapportionment",
    "410": "Antitrust", "422": "Bankruptcy: appeal (28 U.S.C. 158)", "423": "Bankruptcy: withdrawal of reference (28 U.S.C. 157)",
    "430": "Banks and banking", "440": "Civil rights: other civil rights", "441": "Civil rights: voting",
    "442": "Civil rights: employment", "443": "Civil rights: housing / accommodations", "444": "Civil rights: welfare",
    "445": "Civil rights: Americans with Disabilities Act, employment", "446": "Civil rights: Americans with Disabilities Act, other",
    "448": "Civil rights: education", "450": "Commerce", "470": "Racketeer Influenced and Corrupt Organizations Act",
    "480": "Consumer credit", "485": "Telephone Consumer Protection Act", "490": "Cable / satellite TV",
    "610": "Forfeiture / penalty: agriculture", "620": "Forfeiture / penalty: other food and drug",
    "630": "Forfeiture / penalty: liquor laws", "640": "Forfeiture / penalty: railroad and truck",
    "650": "Forfeiture / penalty: airline regulations", "660": "Forfeiture / penalty: occupational safety and health",
    "690": "Forfeiture / penalty: other", "710": "Labor: Fair Labor Standards Act",
    "720": "Labor: labor / management relations", "730": "Labor: labor / management reporting and disclosure",
    "740": "Labor: Railway Labor Act", "751": "Labor: Family and Medical Leave Act", "790": "Labor: other labor litigation",
    "791": "Labor: Employee Retirement Income Security Act", "810": "Selective Service", "820": "Intellectual property: copyright",
    "830": "Intellectual property: patent", "835": "Intellectual property: patent (abbreviated new drug application)",
    "840": "Intellectual property: trademark", "850": "Securities / commodities / exchange", "860": "Social Security",
    "861": "Social Security: Medicare (HIA)", "862": "Social Security: black lung", "863": "Social Security: disability (DIWC / DIWW)",
    "864": "Social Security: supplemental security income (Title XVI)", "865": "Social Security: retirement / survivors (RSI)",
    "870": "Federal tax suits", "871": "Tax: IRS third-party suits (26 U.S.C. 7609)", "875": "Customer challenge (12 U.S.C. 3410)",
    "880": "Intellectual property: Defend Trade Secrets Act", "890": "Other statutory actions", "891": "Agricultural acts",
    "892": "Economic Stabilization Act", "893": "Environmental matters", "894": "Energy Allocation Act",
    "895": "Freedom of Information Act", "896": "Arbitration",
    "899": "Administrative Procedure Act / review of an agency decision",
    "900": "Equal Access to Justice fee determination", "910": "Domestic relations", "930": "Probate",
    "940": "Substitute trustee", "950": "Constitutionality of state statutes", "990": "Other", "992": "Local jurisdictional appeal",
    "999": "Miscellaneous",
}
EXCLUDED_NOS = {"460", "462", "463", "465",                 # immigration (§3.8.6)
                "510", "530", "535", "540", "550", "555", "560",  # prisoner / habeas / detainee
                "625", "920"}                               # drug seizure (quasi-criminal), insanity commitment
EXCLUDED_AGENCIES = {"6", "INS", "BIA", "EOIR", "DHS", "DOJ", "BCA", "CAHO", "CSRT", "BOP", "USCIS", "ICE"}
AGENCY = {"1": "National Labor Relations Board", "3": "Tax Court / Internal Revenue Service", "4": "Federal Trade Commission",
          "7": "Environmental Protection Agency", "8": "Federal Energy Regulatory Commission", "0": "Interstate Commerce Commission",
          "AGRI": "Department of Agriculture", "BRB": "Benefits Review Board (Labor)", "FCC": "Federal Communications Commission",
          "SEC": "Securities and Exchange Commission", "FAA": "Federal Aviation Administration", "LABR": "Department of Labor",
          "OSHC": "Occupational Safety and Health Review Commission", "OSHA": "Occupational Safety and Health Administration",
          "TRAN": "Department of Transportation", "STB": "Surface Transportation Board", "NRC": "Nuclear Regulatory Commission",
          "HHS": "Department of Health and Human Services", "DOE": "Department of Energy", "FLRA": "Federal Labor Relations Authority",
          "MSHR": "Federal Mine Safety and Health Review Commission", "MSHA": "Mine Safety and Health Administration",
          "BPA": "Bonneville Power Administration", "DEA": "Drug Enforcement Administration", "MSPB": "Merit Systems Protection Board",
          "DOI": "Department of the Interior", "FDA": "Food and Drug Administration", "TSA": "Transportation Security Administration",
          "OWCP": "Office of Workers' Compensation Programs", "RRRB": "Railroad Retirement Board", "PRC": "Postal Regulatory Commission",
          "LOC": "Library of Congress / Copyright Royalty Board", "HUD": "Department of Housing and Urban Development",
          "FMCS": "Federal Motor Carrier Safety Administration", "NTSB": "National Transportation Safety Board",
          "FMC": "Federal Maritime Commission", "FRA": "Federal Railroad Administration", "CFTR": "Commodity Futures Trading Commission",
          "EEOC": "Equal Employment Opportunity Commission", "FDIC": "Federal Deposit Insurance Corporation",
          "CFPB": "Consumer Financial Protection Bureau", "OCC": "Office of the Comptroller of the Currency",
          "FRS": "Federal Reserve System", "NCUA": "National Credit Union Administration", "FEC": "Federal Election Commission",
          "COMM": "Department of Commerce", "EDUC": "Department of Education", "DVA": "Department of Veterans Affairs",
          "VET": "Veterans Administration", "ITC": "International Trade Commission", "PATO": "Patent and Trademark Office",
          "SSA": "Social Security Administration", "TREA": "Department of the Treasury", "USPS": "U.S. Postal Service",
          "OPM": "Office of Personnel Management", "DOD": "Department of Defense", "ARMY": "Department of the Army",
          "NAVY": "Department of the Navy", "CPSC": "Consumer Product Safety Commission", "SBA": "Small Business Administration",
          "TTB": "Alcohol and Tobacco Tax and Trade Bureau", "BATF": "Bureau of Alcohol, Tobacco, Firearms and Explosives",
          "NSF": "National Science Foundation", "NASA": "National Aeronautics and Space Administration",
          "GSA": "General Services Administration", "OTS": "Office of Thrift Supervision"}

# FJC civil IDB codes (civil codebook, 1988 forward scheme)
CIV_DISP = {"0": "transferred to another district", "1": "sent back to state court", "2": "closed for want of prosecution",
            "3": "closed for lack of jurisdiction", "4": "default judgment", "5": "consent judgment",
            "6": "judgment on a motion before trial (e.g. summary judgment or judgment on the pleadings)",
            "7": "judgment on a jury verdict", "8": "judgment on a directed verdict", "9": "judgment after a bench trial",
            "10": "transferred to multidistrict litigation", "11": "returned to a federal agency",
            "12": "closed voluntarily by the plaintiff", "13": "closed by settlement", "14": "closed on other grounds",
            "15": "judgment on an arbitrator's award", "16": "judgment after trial de novo following arbitration",
            "17": "judgment on other grounds", "18": "statistical closing", "19": "magistrate judge's decision adopted",
            "20": "appeal from magistrate judge rejected"}
CIV_JUDGMENT = {"1": "plaintiff", "2": "defendant", "3": "both parties in part", "4": "unknown"}
CIV_NOJ = {"0": "no monetary award", "1": "monetary award only", "2": "monetary award and other relief",
           "3": "injunction", "4": "forfeiture, foreclosure or condemnation", "5": "costs only", "6": "costs and attorney fees"}
CIV_PROCPROG = {"1": "before issue joined, no court action", "2": "before issue joined, order entered",
                "3": "after issue joined, no court action", "4": "after issue joined, judgment on a motion",
                "5": "after issue joined, pretrial conference held", "6": "during a bench trial", "7": "during a jury trial",
                "8": "after a bench trial", "9": "after a jury trial", "10": "after issue joined, other",
                "11": "before issue joined, hearing held", "12": "before issue joined, motion decided", "13": "other"}
CIV_JURY = {"B": "both sides", "D": "defendant", "P": "plaintiff", "N": "none"}


def nos_family(code):
    if not code or not code.isdigit():
        return "other"
    c = int(code)
    for lo, hi, fam in [(100, 199, "contract"), (200, 299, "real_property"), (300, 369, "torts"), (370, 371, "fraud_tila"),
                        (375, 376, "false_claims"), (380, 399, "torts"), (400, 400, "constitutional"), (410, 410, "antitrust"),
                        (420, 429, "bankruptcy"), (430, 430, "banking"), (440, 449, "civil_rights"),
                        (450, 499, "commerce_consumer"), (600, 699, "forfeiture_regulatory"), (700, 799, "labor"),
                        (800, 819, "other_statutes"), (820, 849, "intellectual_property"), (850, 859, "securities"),
                        (860, 869, "social_security"), (870, 879, "tax"), (880, 880, "intellectual_property"),
                        (890, 899, "other_statutes"), (900, 999, "other")]:
        if lo <= c <= hi:
            return fam
    return "other"


# --------------------------------------------------------------------------- small utils
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fdate(s):
    """FJC MM/DD/YYYY -> date, None when missing (01/01/1900) or malformed."""
    try:
        d = datetime.strptime(s.strip(), "%m/%d/%Y").date()
    except (ValueError, AttributeError):
        return None
    return None if d.year <= 1900 else d


def idate(s):
    try:
        return date.fromisoformat(s[:10]) if s else None
    except ValueError:
        return None


def ym(d):
    return d.strftime("%Y-%m") if d else None


def hkey(*parts):
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def jdump(obj):
    return json.dumps(obj, sort_keys=True, separators=(", ", ": "), ensure_ascii=False)


def write_gz(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    n = 0
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    tmp.rename(path)
    return n


def read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def http_get(url, headers=None, timeout=300):
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == 4:
                raise
            log(f"  retry {attempt + 1} for {url}: {type(e).__name__}")
            time.sleep(5 * (attempt + 1))


class HttpRangeFile(io.RawIOBase):
    """A seekable read-only view of a remote file through HTTP Range requests (for zip central directories), so a
    330 MB zip is read without being stored."""

    def __init__(self, url, block=8 << 20):
        self.url, self.block, self.pos, self.nreq = url, block, 0, 0
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.cache_start, self.cache = -1, b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        if not (self.cache_start <= self.pos < self.cache_start + len(self.cache)):
            end = min(self.size, self.pos + self.block) - 1
            self.cache = http_get(self.url, {"Range": f"bytes={self.pos}-{end}"})
            self.cache_start = self.pos
            self.nreq += 1
        off = self.pos - self.cache_start
        n = min(len(b), len(self.cache) - off)
        b[:n] = self.cache[off:off + n]
        self.pos += n
        return n


class Bz2HttpStream(io.RawIOBase):
    """Stream-decompress a (possibly multi-stream) .bz2 over HTTP; nothing touches disk. A dropped connection is
    resumed with a Range request at the byte offset already consumed (the decompressor state carries over)."""

    def __init__(self, url):
        self.url = url
        self.d, self.buf, self.nin, self.nout = bz2.BZ2Decompressor(), b"", 0, 0
        self.resp = self._open()

    def _open(self):
        h = {"User-Agent": UA}
        if self.nin:
            h["Range"] = f"bytes={self.nin}-"
        return urllib.request.urlopen(urllib.request.Request(self.url, headers=h), timeout=600)

    def _read(self):
        for attempt in range(8):
            try:
                return self.resp.read(1 << 20)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                log(f"  stream error at {self.nin:,} bytes ({type(e).__name__}); resuming, attempt {attempt + 1}")
                time.sleep(5 * (attempt + 1))
                self.resp = self._open()
        raise RuntimeError(f"could not resume {self.url}")

    def readable(self):
        return True

    def readinto(self, b):
        while not self.buf:
            if self.d.eof:
                rest, self.d = self.d.unused_data, bz2.BZ2Decompressor()
                if rest:
                    self.buf = self.d.decompress(rest)
                    continue
            chunk = self._read()
            if not chunk:
                return 0
            self.nin += len(chunk)
            self.buf = self.d.decompress(chunk)
        n = min(len(b), len(self.buf))
        b[:n] = self.buf[:n]
        self.buf = self.buf[n:]
        self.nout += n
        return n


def cl_csv(name, every=2_000_000):
    """Rows (as dicts) of a CourtListener bulk CSV, streamed. Logs progress."""
    raw = Bz2HttpStream(CL_BULK + name)
    f = io.TextIOWrapper(io.BufferedReader(raw, 1 << 20), encoding="utf-8", newline="")
    r = csv.reader(f, escapechar="\\")
    hdr = next(r)
    n, t0 = 0, time.time()
    for row in r:
        n += 1
        if len(row) != len(hdr):
            continue
        yield dict(zip(hdr, row))
        if n % every == 0:
            log(f"  {name}: {n:,} rows, {raw.nin / 1e9:.2f} GB read, {raw.nin / 1e6 / (time.time() - t0):.1f} MB/s")
    log(f"  {name}: done, {n:,} rows, {raw.nin / 1e9:.2f} GB compressed")


def cl_docket_key(docket_number):
    """'19-55555' / 'No. 19-1234-cv' -> '1955555' / '1901234' (the FJC 7-character YYNNNNN form)."""
    m = re.search(r"(?<!\d)(\d{2})-(\d{1,5})(?!\d)", docket_number or "")
    return (m.group(1) + m.group(2).zfill(5)) if m else None


# --------------------------------------------------------------------------- credentials (CourtListener API only)
def cl_auth_header():
    """Authorization header for www.courtlistener.com, built in memory from ~/.courtlistener. Never printed."""
    st = os.stat(CRED_PATH)
    if stat.S_IMODE(st.st_mode) & 0o077:
        raise SystemExit(f"{CRED_PATH} must be mode 600")
    with open(CRED_PATH, encoding="utf-8") as f:
        lines = [ln.rstrip("\r\n") for ln in f.read().split("\n")]
    if len(lines) > 2 and lines[2].strip():
        return "Token " + lines[2].strip()
    return "Basic " + base64.b64encode(f"{lines[0].strip()}:{lines[1]}".encode()).decode()


def cl_api(path, auth, params=None):
    url = CL_API + path + (("?" + urllib.parse.urlencode(params)) if params else "")
    assert url.startswith("https://www.courtlistener.com/"), "credentials go to CourtListener only"
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Authorization": auth, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def auth_check(_a):
    auth = cl_auth_header()
    try:
        body = cl_api("api-usage/", auth)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: authentication failed or blocked")
        return
    kind = auth.split(" ", 1)[0]
    print(f"authenticated to the CourtListener v4 API with {kind} auth; membership={body.get('membership')}")
    for row in body.get("current_usage", []):
        print(f"  {row['scope']:10s} {row['rate']:>9s}  used {row['used']:4d}  remaining {row['remaining']:4d}")


# --------------------------------------------------------------------------- party kinds (names are never kept)
_BUSINESS = re.compile(r"\b(INC|LLC|L\.L\.C|CORP|CORPORATION|CO|COMPANY|BANK|N\.A|NA|LTD|LP|L\.P|LLP|PLC|INS|INSURANCE|"
                       r"HOLDINGS|ENTERPRISES|GROUP|PARTNERS|TRUST CO|FINANCIAL|MORTGAGE|SERVICES|INDUSTRIES|SYSTEMS|"
                       r"TECHNOLOGIES|HOSPITAL|MEDICAL CTR|ENERGY|RAILWAY|RAILROAD|AIRLINES|MOTORS|STORES|ASSUR|MUTUAL|"
                       r"PHARMA\w*|LABS?|INTL|INTERNATIONAL|CAPITAL|SECURITIES|PROPERTIES|REALTY|DEVELOPMENT)\b")
_GOV = re.compile(r"\b(CITY|COUNTY|STATE OF|COMMONWEALTH|TOWN|TOWNSHIP|VILLAGE|BOROUGH|PARISH|DEPT|DEPARTMENT|BOARD|"
                  r"COMMISSIONER|COMMISSION|SECRETARY|SECY|SEC'Y|ATTORNEY GENERAL|AUTHORITY|SCHOOL|SCH DIST|"
                  r"POLICE|SHERIFF|WARDEN|GOVERNOR|UNIV|UNIVERSITY|AGENCY|BUREAU|ADMINISTRATOR|DIRECTOR|"
                  r"MUNICIPALITY|PUBLIC|TRANSIT|HOUSING AUTH|STATE)\b")
_ORG = re.compile(r"\b(UNION|LOCAL|ASSN|ASSOCIATION|FOUNDATION|CHURCH|COUNCIL|COALITION|SOCIETY|CLUB|FEDERATION|"
                  r"INSTITUTE|PLAN|FUND|TRUSTEES|TRUST|COOP|COOPERATIVE|CENTER|LEAGUE|PARTY)\b")


def party_kind(name, is_us):
    if is_us:
        return "United States (federal government)"
    s = re.sub(r"[^A-Z0-9.' ]", " ", (name or "").upper())
    s = re.sub(r"\bET AL\b", " ", s).strip()
    if not s or s in ("-8", "UNKNOWN"):
        return "not recorded"
    if re.search(r"\b(USA|UNITED STATES|U S A|US)\b", s):
        return "United States (federal government)"
    if _GOV.search(s):
        return "government body or official"
    if _BUSINESS.search(s):
        return "business"
    if _ORG.search(s):
        return "organisation, union or benefit plan"
    return "individual"


def name_sig(name):
    """A one-way signature of a party name (hash of its first word and of the first 3 letters of its second), so the
    appellant can be matched to the district plaintiff / defendant without any name being stored."""
    w = [x for x in re.sub(r"[^A-Z0-9 ]", " ", (name or "").upper()).split()
         if x not in ("ET", "AL", "THE", "OF", "INC", "LLC")]
    if not w:
        return None
    return [hkey("n1", w[0]), hkey("n2", w[1][:3]) if len(w) > 1 else None]


def same_party(a, b):
    return bool(a and b and a[0] == b[0] and (a[1] is None or b[1] is None or a[1] == b[1]))


# --------------------------------------------------------------------------- fetch: FJC appeals
APPEAL_FIELDS = ["CIRCUIT", "DOCKET", "REOPEN", "DKTDATE", "USAPT", "USAPE", "APPTYPE", "AGENCY", "JURIS", "NOS", "DCIRC",
                 "DDIST", "DOFFICE", "DDOCKET", "DDKTDATE", "APPDATE", "FILEFEE", "DISP", "OUTCOME", "PUBSTAT", "JOINAPP",
                 "CONSDKT", "CRECDATE", "BRFILED", "SUBDATE", "HEARDATE", "JUDGDATE", "OPINION", "JDG1INV", "JDG2INV",
                 "JDG3INV", "TAPEYEAR", "PROSEFLE", "ENBANC", "TRMTYPE"]


def appeal_in_scope(r, why):
    if r["STATUSCD"] != "L":
        return why("pending")
    if r["DISP"] not in ("1", "2", "3"):
        return why("not_a_merits_termination")
    if r["OUTCOME"] not in OUTCOME_KEY:
        return why(f"outcome_{r['OUTCOME']}")
    if r["APPTYPE"] not in APPTYPE:
        return why("criminal_or_original_proceeding" if r["APPTYPE"].isdigit() and int(r["APPTYPE"]) >= 5 else "apptype_other")
    if r["NOS"] in EXCLUDED_NOS:
        return why("excluded_nos_immigration_prisoner")
    if r["APPTYPE"] in ("1", "2") and r["AGENCY"].strip().upper() in EXCLUDED_AGENCIES:
        return why("excluded_agency_immigration_prison")
    if r["ENBANC"] == "Y":
        return why("en_banc")
    if CIRCUIT_ID.get(r["CIRCUIT"]) is None:
        return why("circuit_unknown")
    if fdate(r["JUDGDATE"]) is None:
        return why("no_judgment_date")
    return True


def fetch_fjc_appeals(cache):
    out = cache / "fjc_appeals.jsonl.gz"
    if out.exists():
        log(f"have {out}")
        return
    log(f"FJC appeals: {FJC_APPEALS_URL}")
    data = http_get(FJC_APPEALS_URL)
    z = zipfile.ZipFile(io.BytesIO(data))
    member = [i for i in z.infolist() if i.filename.lower().endswith(".txt")][0]
    f = io.TextIOWrapper(z.open(member), encoding="latin-1", newline="")
    reasons = Counter()

    def why(k):
        reasons[k] += 1
        return False

    def rows():
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if appeal_in_scope(r, why) is not True:
                continue
            reasons["kept"] += 1
            keep = {k: r.get(k, "").strip() for k in APPEAL_FIELDS}
            # party kinds are derived here; the names themselves never leave this function
            keep["appellant_kind"] = party_kind(r["APPELLAN"], r["USAPT"] == "1")
            keep["appellee_kind"] = party_kind(r["APPELLEE"], r["USAPE"] == "1")
            keep["appellant_sig"] = name_sig(r["APPELLAN"])   # one-way; for the plaintiff/defendant role join
            yield keep

    n = write_gz(out, rows())
    (cache / "fjc_appeals.filter.json").write_text(json.dumps({"kept": n, "reasons": reasons}, indent=1))
    log(f"FJC appeals: kept {n:,}; {dict(reasons.most_common())}")


CIVIL_FIELDS = ["DISTRICT", "OFFICE", "DOCKET", "ORIGIN", "FILEDATE", "JURIS", "NOS", "TITL", "SECTION", "SUBSECT", "JURY",
                "CLASSACT", "DEMANDED", "MDLDOCK", "TERMDATE", "PROCPROG", "DISP", "NOJ", "AMTREC", "JUDGMENT", "PROSE",
                "IFP", "TAPEYEAR"]


def fetch_fjc_civil(cache):
    out = cache / "fjc_civil.jsonl.gz"
    if out.exists():
        log(f"have {out}")
        return
    keys = set()
    for r in read_gz(cache / "fjc_appeals.jsonl.gz"):
        if r["DDIST"] not in ("", "-8") and r["DDOCKET"] not in ("", "-8"):
            keys.add((r["DDIST"], r["DDOCKET"].zfill(7)))
    log(f"FJC civil: {len(keys):,} district dockets wanted; streaming {FJC_CIVIL_URL} by HTTP range")
    rf = HttpRangeFile(FJC_CIVIL_URL)
    z = zipfile.ZipFile(io.BufferedReader(rf, 1 << 20))
    member = [i for i in z.infolist() if i.filename.lower().endswith(".txt")][0]
    f = io.TextIOWrapper(z.open(member), encoding="latin-1", newline="")
    seen = Counter()

    def rows():
        rd = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        hdr = next(rd)
        i_d, i_k = hdr.index("DISTRICT"), hdr.index("DOCKET")
        for i, row in enumerate(rd):
            if i and i % 2_000_000 == 0:
                log(f"  civil rows {i:,}; matched {seen['matched']:,}; {rf.nreq} range requests")
            if len(row) <= max(i_d, i_k) or (row[i_d].strip(), row[i_k].strip().zfill(7)) not in keys:
                continue
            r = dict(zip(hdr, row))
            seen["matched"] += 1
            keep = {c: (r.get(c) or "").strip() for c in CIVIL_FIELDS}
            keep["plt_sig"], keep["def_sig"] = name_sig(r.get("PLT")), name_sig(r.get("DEF"))   # one-way
            yield keep

    n = write_gz(out, rows())
    log(f"FJC civil: kept {n:,} district records for {len(keys):,} wanted dockets ({rf.nreq} range requests)")


# --------------------------------------------------------------------------- fetch: CourtListener bulk
DOCKET_FIELDS = ["id", "court_id", "docket_number", "date_filed", "date_terminated", "date_argued", "appeal_from_id",
                 "appeal_from_str", "nature_of_suit", "cause", "appellate_fee_status", "appellate_case_type_information",
                 "originating_court_information_id", "idb_data_id", "parent_docket_id", "source"]


def fetch_cl(cache, snapshot):
    fed_keys = defaultdict(set)
    for r in read_gz(cache / "fjc_appeals.jsonl.gz"):
        fed_keys[CIRCUIT_ID[r["CIRCUIT"]]].add(r["DOCKET"].zfill(7))
    out_d = cache / "cl_dockets.jsonl.gz"
    if not out_d.exists():
        log(f"CourtListener dockets-{snapshot}: keeping the federal appellate dockets that match FJC appeals")

        def drows():
            for r in cl_csv(f"dockets-{snapshot}.csv.bz2"):
                c = r["court_id"]
                if c in fed_keys and cl_docket_key(r["docket_number"]) in fed_keys[c]:
                    yield {k: r.get(k, "") for k in DOCKET_FIELDS}
        log(f"  kept {write_gz(out_d, drows()):,} dockets")
    dockets = list(read_gz(out_d))
    oci_ids = {d["originating_court_information_id"] for d in dockets if d["originating_court_information_id"]}
    out_o = cache / "cl_oci.jsonl.gz"
    if not out_o.exists():
        keep = ["id", "docket_number", "date_filed", "date_disposed", "date_judgment", "date_filed_noa", "date_received_coa"]
        log(f"  kept {write_gz(out_o, ({k: r.get(k, '') for k in keep} for r in cl_csv(f'originating-court-information-{snapshot}.csv.bz2') if r['id'] in oci_ids)):,} originating-court rows")
    # state supreme court cases: reporter citations -> CourtListener clusters
    out_c = cache / "cl_citations.jsonl.gz"
    if not out_c.exists():
        want = set()
        for r in read_gz(cache / "sscdp.jsonl.gz"):
            if r.get("reporter_str") and r.get("rep_vol") and r.get("rep_page"):
                want.add((r["reporter_str"], r["rep_vol"], r["rep_page"]))
        log(f"  citations: {len(want):,} state-court reporter cites wanted")
        log(f"  kept {write_gz(out_c, ({'volume': r['volume'], 'reporter': r['reporter'], 'page': r['page'], 'cluster_id': r['cluster_id']} for r in cl_csv(f'citations-{snapshot}.csv.bz2') if (r['reporter'], r['volume'], r['page']) in want)):,} citations")
    docket_ids = {d["id"] for d in dockets}
    state_clusters = {r["cluster_id"] for r in read_gz(out_c)}
    out_k = cache / "cl_clusters.jsonl.gz"
    if not out_k.exists():
        keep = ["id", "docket_id", "date_filed", "disposition", "precedential_status", "source"]
        log(f"  clusters: keeping those of {len(docket_ids):,} appellate dockets and {len(state_clusters):,} state cites")
        log(f"  kept {write_gz(out_k, ({k: r.get(k, '') for k in keep} for r in cl_csv(f'opinion-clusters-{snapshot}.csv.bz2') if r['docket_id'] in docket_ids or r['id'] in state_clusters)):,} clusters")
    cluster_ids = {c["id"] for c in read_gz(out_k)}
    out_p = cache / "cl_panel.jsonl.gz"
    if not out_p.exists():
        def prows():
            for r in cl_csv(f"search_opinioncluster_panel-{snapshot}.csv.bz2"):
                ck = next((k for k in r if k.endswith("cluster_id")), None)
                pk = next((k for k in r if k.endswith("person_id")), None)
                if ck and pk and r[ck] in cluster_ids:
                    yield {"cluster_id": r[ck], "person_id": r[pk]}
        log(f"  kept {write_gz(out_p, prows()):,} panel rows")


# --------------------------------------------------------------------------- fetch: State Supreme Court Data Project
SSCDP_REPORTER = {"1": "A.2d", "2": "N.E.2d", "3": "N.W.2d", "4": "P.2d", "5": "S.E.2d", "6": "So. 2d", "7": "S.W.2d"}
STATES = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida",
          "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
          "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
          "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma",
          "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",
          "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]
STATE_COURT = {   # CourtListener court id and the court's name (court of last resort, civil side)
    "Alabama": ("ala", "Supreme Court of Alabama"), "Alaska": ("alaska", "Alaska Supreme Court"),
    "Arizona": ("ariz", "Arizona Supreme Court"), "Arkansas": ("ark", "Supreme Court of Arkansas"),
    "California": ("cal", "California Supreme Court"), "Colorado": ("colo", "Supreme Court of Colorado"),
    "Connecticut": ("conn", "Supreme Court of Connecticut"), "Delaware": ("del", "Supreme Court of Delaware"),
    "Florida": ("fla", "Supreme Court of Florida"), "Georgia": ("ga", "Supreme Court of Georgia"),
    "Hawaii": ("haw", "Hawaii Supreme Court"), "Idaho": ("idaho", "Idaho Supreme Court"),
    "Illinois": ("ill", "Illinois Supreme Court"), "Indiana": ("ind", "Indiana Supreme Court"),
    "Iowa": ("iowa", "Supreme Court of Iowa"), "Kansas": ("kan", "Supreme Court of Kansas"),
    "Kentucky": ("ky", "Kentucky Supreme Court"), "Louisiana": ("la", "Supreme Court of Louisiana"),
    "Maine": ("me", "Supreme Judicial Court of Maine"), "Maryland": ("md", "Court of Appeals of Maryland"),
    "Massachusetts": ("mass", "Massachusetts Supreme Judicial Court"), "Michigan": ("mich", "Michigan Supreme Court"),
    "Minnesota": ("minn", "Supreme Court of Minnesota"), "Mississippi": ("miss", "Mississippi Supreme Court"),
    "Missouri": ("mo", "Supreme Court of Missouri"), "Montana": ("mont", "Montana Supreme Court"),
    "Nebraska": ("neb", "Nebraska Supreme Court"), "Nevada": ("nev", "Nevada Supreme Court"),
    "New Hampshire": ("nh", "Supreme Court of New Hampshire"), "New Jersey": ("nj", "Supreme Court of New Jersey"),
    "New Mexico": ("nm", "New Mexico Supreme Court"), "New York": ("ny", "New York Court of Appeals"),
    "North Carolina": ("nc", "Supreme Court of North Carolina"), "North Dakota": ("nd", "North Dakota Supreme Court"),
    "Ohio": ("ohio", "Ohio Supreme Court"), "Oklahoma": ("okla", "Supreme Court of Oklahoma"),
    "Oregon": ("or", "Oregon Supreme Court"), "Pennsylvania": ("pa", "Supreme Court of Pennsylvania"),
    "Rhode Island": ("ri", "Supreme Court of Rhode Island"), "South Carolina": ("sc", "Supreme Court of South Carolina"),
    "South Dakota": ("sd", "South Dakota Supreme Court"), "Tennessee": ("tenn", "Tennessee Supreme Court"),
    "Texas": ("tex", "Texas Supreme Court"), "Utah": ("utah", "Utah Supreme Court"),
    "Vermont": ("vt", "Supreme Court of Vermont"), "Virginia": ("va", "Supreme Court of Virginia"),
    "Washington": ("wash", "Washington Supreme Court"), "West Virginia": ("wva", "West Virginia Supreme Court of Appeals"),
    "Wisconsin": ("wis", "Wisconsin Supreme Court"), "Wyoming": ("wyo", "Wyoming Supreme Court")}
SS_JURIS = {"2": "appeal or petition for review", "7": "appeal heard de novo", "8": "interlocutory appeal",
            "3": "rehearing or case restored to the calendar", "4": "returned from the U.S. Supreme Court",
            "13": "application for leave to appeal / certiorari"}
SS_COURT = {"1": "state trial court of limited jurisdiction", "2": "state trial court of general jurisdiction",
            "3": "state intermediate appellate court", "4": "the state supreme court itself", "5": "U.S. district court",
            "6": "U.S. court of appeals", "7": "U.S. Supreme Court"}
SS_DISP = {"1": "affirm", "2": "reverse", "3": "reverse_and_remand", "4": "affirm_in_part", "5": "affirm_in_part"}
SS_KEYS = ["affirm", "reverse", "reverse_and_remand", "affirm_in_part"]
SS_CRITERIA = {
    "affirm": "Affirmed: the court approves the decision under review in full.",
    "reverse": "Reversed or vacated outright: the decision under review is overturned or set aside in full and the "
               "case is not sent back for further proceedings.",
    "reverse_and_remand": "Reversed or vacated and sent back: the decision under review is overturned or set aside in "
                          "full and the case returns to a lower court for further proceedings.",
    "affirm_in_part": "Mixed: the decision is approved in part and overturned or modified in part, whether or not the "
                      "case is sent back.",
}
SS_DISP_INSTR = ("How will the state's highest court dispose of the decision under review? Choose the disposition in "
                 "its judgment, using the definitions in the options.")
SS_VOTE_INSTR = ("Does a justice of this court vote in favour of the petitioner / appellant, the party that asked the "
                 "court to review the decision below?\n"
                 "Yes means: the justice votes to give that party the relief it seeks, in whole or in part.\n"
                 "No means: the justice votes for the respondent / appellee.")
SS_DIV_INSTR = ("Will the court be divided in this case?\n"
                "Yes means: at least one participating justice votes against the side the majority favours.\n"
                "No means: every participating justice votes for the same side (separate concurrences allowed).")
SS_PARTY = [("persn", "individual"), ("busi", "business"), ("org", "organisation or association"),
            ("lgov", "local or county government"), ("sgov", "state government"), ("edu", "educational institution"),
            ("other", "other party")]
SS_ISSUES = {  # SSCDP civil issue flags -> wording (codebook labels, cleaned)
    "adopt": "adoption", "divorce": "separation and divorce", "childsup": "child support and custody",
    "domvioln": "domestic violence / restraining order", "paternty": "paternity", "dom_oth": "other domestic relations",
    "will": "probate, wills and intestate estates", "trustee": "guardianship, conservatorship or trusteeship",
    "est_oth": "other estate issue", "cp_enfor": "contract enforcement (breach, specific performance)",
    "relation": "debtor-creditor relations", "insure": "insurance", "tenant": "landlord-tenant",
    "ownershp": "ownership of real property", "cont_oth": "other contract issue", "malprc_m": "medical malpractice",
    "malprc_p": "professional malpractice", "automobl": "automobile accident", "products": "product liability",
    "toxic": "toxic substances", "cp_prem": "premises liability", "cp_libel": "libel, slander or defamation",
    "cp_workr": "employee injury / workers' compensation", "cp_disc": "employment discrimination",
    "cp_oth_l": "other labour dispute", "ptor_oth": "other tort", "redist": "apportionment and redistricting",
    "elect": "contested election", "ballot": "ballot access", "campaign": "campaign spending",
    "elec_oth": "other election issue", "aid2paro": "aid to parochial schools", "religion": "free exercise of religion",
    "cg_libel": "libel, slander or defamation (government party)", "oath": "loyalty oath",
    "protest": "protests, marches or picketing", "obscene": "obscenity", "comspch": "commercial speech",
    "frst_oth": "other First Amendment issue", "taxes": "taxation", "zoning": "zoning and planning",
    "utility": "utility regulation", "emdomain": "eminent domain", "license": "licensing and permits",
    "welfare": "government benefits / welfare", "transpt": "transportation regulation", "consumer": "consumer protection",
    "environm": "environmental protection", "greg_oth": "other government regulation",
    "affirmac": "race- or sex-conscious preference programmes / set-asides", "cg_enfor": "public contract enforcement",
    "cg_disc": "discrimination by a public employer", "pubc_oth": "other public contract issue",
    "foia": "access to government information", "abortion": "abortion", "dieright": "right to die",
    "drugtest": "mandatory drug testing", "gayright": "sexual-orientation rights", "priv_oth": "other privacy issue",
    "cg_prem": "premises liability (government party)", "cg_workr": "public employee injury / workers' compensation",
    "gtor_oth": "other tort against government",
    "cv_abuse": "abuse of discretion by the trial judge", "cv_arbit": "arbitrary-or-capricious standard",
    "cv_erro": "clearly-erroneous standard", "cv_atty": "counsel / attorney's fees", "cv_disc": "discovery",
    "cv_dscm": "discrimination", "cv_ev_m": "admission of evidence", "cv_ev_w": "expert or witness testimony",
    "cv_claim": "failure to state a claim", "cv_immun": "government immunity", "cv_jr_in": "jury instructions",
    "cv_jr_sl": "jury selection", "cv_moot": "mootness", "cv_ripe": "ripeness / exhaustion of remedies",
    "cv_recus": "recusal", "cv_stand": "standing to sue", "cv_suff": "sufficiency of the evidence",
    "cv_sumju": "summary judgment", "cv_injun": "validity of an injunction", "cv_venue": "venue",
    "cv_othis": "other legal issue"}
SS_AREA = {"gen_dom": "domestic relations", "gen_est": "estates", "gen_cont": "contracts", "gen_ptor": "torts",
           "gen_elec": "elections", "gen_frst": "First Amendment", "gen_greg": "government regulation",
           "gen_pubc": "public contracts", "gen_priv": "privacy", "gen_gtor": "torts against government"}
SS_EXCLUDE_FLAGS = ("gen_law", "invlster", "steriliz")   # practice of law (discipline, bar admission); sterilisation


def fetch_sscdp(cache):
    out = cache / "sscdp.jsonl.gz"
    if out.exists():
        log(f"have {out}")
        return
    log(f"SSCDP: {SSCDP_URL}")
    text = http_get(SSCDP_URL).decode("latin-1")
    cols = None
    reasons = Counter()

    def rows():
        nonlocal cols
        for r in csv.DictReader(io.StringIO(text), delimiter="\t"):
            if r.get("genissue") not in ("2", "3"):
                reasons["not_civil(criminal/juvenile/non-adversarial)"] += 1
                continue
            if any(r.get(f) == "1" for f in SS_EXCLUDE_FLAGS):
                reasons["excluded_issue(practice of law / sterilisation)"] += 1
                continue
            g = "cp" if r["genissue"] == "3" else "cg"
            if r.get(f"{g}_disp") not in SS_DISP:
                reasons[f"disp_{r.get(g + '_disp')}"] += 1
                continue
            if r.get("juris") not in SS_JURIS:
                reasons["juris_original_or_other"] += 1
                continue
            reasons["kept"] += 1
            keep = {k: r.get(k, "") for k in ("masterid", "id_1", "casen_1", "docket", "juris", "first_ct", "agency", "arbitrat", "dec1_yr",
                                              "rev_ct", "crossapp", "o_plain", "o_defend", "state_2", "reporter",
                                              "rep_vol", "rep_page", "decs_yr", "decs_mo", "decs_day", "classact",
                                              "amicus", "multi_p", "multi_r", "genissue", "enbanc", "docket_n")}
            keep["disp"] = r[f"{g}_disp"]
            for side in ("p1", "p2", "r1", "r2"):
                keep[side] = next((lbl for code, lbl in SS_PARTY if r.get(f"{side}_{code}") not in ("", "0", "88", "NA", None)), None)
            keep["issues"] = [k for k in SS_ISSUES if r.get(k) == "1"]
            keep["areas"] = [k for k in SS_AREA if r.get(k) == "1"]
            keep["usc"] = r.get(f"{g}_usc") == "1"
            keep["stc"] = r.get(f"{g}_stc") == "1"
            votes = [r.get(f"j{i}_vote") for i in range(1, 10)] + [r.get(f"s{i}_vote") for i in range(1, 4)]
            keep["votes_pet"] = sum(1 for v in votes if v == "1")
            keep["votes_res"] = sum(1 for v in votes if v == "2")
            keep["reporter_str"] = SSCDP_REPORTER.get(r.get("reporter"))
            yield keep

    n = write_gz(out, rows())
    (cache / "sscdp.filter.json").write_text(json.dumps({"kept": n, "reasons": reasons}, indent=1))
    log(f"SSCDP: kept {n:,}; {dict(reasons.most_common())}")


def fetch(a):
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    fetch_fjc_appeals(cache)
    fetch_sscdp(cache)
    fetch_fjc_civil(cache)
    if not a.skip_cl:
        fetch_cl(cache, a.cl_snapshot)
    sizes = {p.name: p.stat().st_size for p in cache.iterdir() if p.is_file()}
    log(f"cache: {sum(sizes.values()) / 1e6:.1f} MB  {sizes}")


# --------------------------------------------------------------------------- labels: hierarchical LOO base rates
def encode_levels(keys_by_level):
    import numpy as np
    out = []
    for keys in keys_by_level:
        d = {}
        out.append(np.array([d.setdefault(k, len(d)) for k in keys], dtype=np.int64))
    return out


def loo_rates(level_ids, y, K, fit_mask, kappas):
    """Hierarchical Dirichlet-smoothed class rates per item.

    level_ids: list of L int arrays (encode_levels; level 0 = coarsest). y: int class per item.
    fit_mask: the items whose outcomes are counted. A counted item's own outcome is subtracted from every level
    (leave-one-out), so it never enters its own rate; uncounted items (val, eval) get rates from the fit set only.
    Returns an [N, K] array: p_L = (n_L + kappa_L p_{L-1}) / (N_L + kappa_L)."""
    import numpy as np
    N = len(y)
    own = np.zeros((N, K))
    idx = np.where(fit_mask & (y >= 0))[0]
    own[idx, y[idx]] = 1.0
    p = np.full((N, K), 1.0 / K)
    for L, ids in enumerate(level_ids):
        cnt = np.zeros((int(ids.max()) + 1, K))
        np.add.at(cnt, ids[idx], own[idx])
        n = cnt[ids] - own
        p = (n + kappas[L] * p) / (n.sum(1, keepdims=True) + kappas[L])
    return p


def fit_kappas(level_ids, y, K, fit_mask, grid=(0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 2000)):
    """Coordinate search of each level's prior strength by mean leave-one-out log score on the fit set."""
    import numpy as np
    kap = [20.0] * len(level_ids)
    idx = np.where(fit_mask & (y >= 0))[0]

    def score(k):
        p = loo_rates(level_ids, y, K, fit_mask, k)
        return float(np.mean(np.log(np.clip(p[idx, y[idx]], 1e-12, 1))))
    best = score(kap)
    for _ in range(2):
        for L in range(len(kap)):
            for g in grid:
                trial = kap[:L] + [float(g)] + kap[L + 1:]
                s = score(trial)
                if s > best + 1e-9:
                    best, kap = s, trial
    return kap, best


def loo_binom_rate(level_ids, c, n, fit_mask, kappas):
    """Per-rater rate m (c successes of n raters per item), hierarchical and leave-one-out like loo_rates."""
    import numpy as np
    cf, nf = np.where(fit_mask, c, 0.0), np.where(fit_mask, n, 0.0)
    m = np.full(len(c), 0.5)          # uninformative start; level 0 (global, LOO) dominates it
    for L, ids in enumerate(level_ids):
        C = np.bincount(ids, weights=cf, minlength=int(ids.max()) + 1)
        T = np.bincount(ids, weights=nf, minlength=int(ids.max()) + 1)
        m = (C[ids] - cf + kappas[L] * m) / (T[ids] - nf + kappas[L])
    return np.clip(m, 1e-6, 1 - 1e-6)


def fit_binom_kappas(level_ids, c, n, fit_mask, grid=(5, 20, 50, 100, 200, 500, 1000, 3000)):
    import numpy as np
    kap = [200.0] * len(level_ids)

    def score(k):
        m = loo_binom_rate(level_ids, c, n, fit_mask, k)
        s = fit_mask & (n > 0)
        return float(np.sum(c[s] * np.log(m[s]) + (n[s] - c[s]) * np.log(1 - m[s])) / max(1.0, n[s].sum()))
    best = score(kap)
    for _ in range(2):
        for L in range(len(kap)):
            for g in grid:
                trial = kap[:L] + [float(g)] + kap[L + 1:]
                s = score(trial)
                if s > best + 1e-12:
                    best, kap = s, trial
    return kap, best


def fit_betabinom_a(c, n, m, grid=None):
    """Concentration a of theta ~ Beta(a m, a (1-m)) maximising the beta-binomial marginal likelihood of c of n."""
    import numpy as np
    from scipy.special import betaln
    grid = grid or list(np.exp(np.linspace(np.log(0.05), np.log(2000), 60)))
    m = np.clip(m, 1e-6, 1 - 1e-6)
    best, best_a = -np.inf, None
    for a in grid:
        al, be = a * m, a * (1 - m)
        ll = float(np.sum(betaln(c + al, n - c + be) - betaln(al, be)))
        if ll > best:
            best, best_a = ll, float(a)
    return best_a, best


def calib_report(p, y):
    """Brier (multiclass), log loss, accuracy and 15-bin top-label ECE of rates p against outcomes y."""
    import numpy as np
    K = p.shape[1]
    oh = np.eye(K)[y]
    conf, pred = p.max(1), p.argmax(1)
    bins = np.minimum((conf * 15).astype(int), 14)
    ece = sum(abs(np.mean(conf[bins == b]) - np.mean(pred[bins == b] == y[bins == b])) * np.mean(bins == b)
              for b in range(15) if np.any(bins == b))
    return {"n": int(len(y)), "brier": float(np.mean(np.sum((p - oh) ** 2, 1))),
            "log_loss": float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)))),
            "accuracy": float(np.mean(pred == y)), "ece15_top_label": float(ece)}


# --------------------------------------------------------------------------- build: federal appeals
def split_of_year(year, group, val_pct=VAL_PCT):
    if year in EVAL_YEARS:
        return "eval", f"eval: decided {year} (in {EVAL_YEARS[0]}-{EVAL_YEARS[-1]})"
    if year <= TRAIN_MAX_YEAR:
        if int(group[:8], 16) % 100 < val_pct:
            return "val", f"val: {val_pct}% hash sample (by group) of decisions <= {TRAIN_MAX_YEAR}"
        return "train", f"train: decided <= {TRAIN_MAX_YEAR}"
    return None, f"buffer year {year} (not written)"


def lower_court(appeal, civil_by_key):
    """The district case as it stood when the appeal was docketed (as-of rule), or None."""
    k = (appeal["DDIST"], appeal["DDOCKET"].zfill(7))
    cands = civil_by_key.get(k) or []
    dkt = fdate(appeal["DKTDATE"])
    ddkt = fdate(appeal["DDKTDATE"])
    best = None
    for c in cands:
        if appeal["DOFFICE"] and c["OFFICE"] and c["OFFICE"] != appeal["DOFFICE"]:
            continue
        fd = fdate(c["FILEDATE"])
        if ddkt and fd and abs((fd - ddkt).days) > 31:
            continue
        td = fdate(c["TERMDATE"])
        if td is None or dkt is None or td > dkt:      # terminated after the appeal was docketed: not as-of
            continue
        if best is None or td > fdate(best["TERMDATE"]):
            best = c
    pending = [c for c in cands if (fdate(c["TERMDATE"]) is None or (dkt and fdate(c["TERMDATE"]) > dkt))
               and not (appeal["DOFFICE"] and c["OFFICE"] and c["OFFICE"] != appeal["DOFFICE"])]
    return best, bool(cands), bool(pending)


def federal_state(a, civ, matched_any, rng):
    """Structured as-of state (dict of 2-4 sections) for one appeal."""
    cid = CIRCUIT_ID[a["CIRCUIT"]]
    dkt = fdate(a["DKTDATE"])
    appeal = {"court": CIRCUIT_NAME[cid], "docketed": ym(dkt), "appeal_type": APPTYPE[a["APPTYPE"]]}
    heard = fdate(a["HEARDATE"])
    appeal["argument"] = ("oral argument held" if (a["DISP"] == "1" or heard)
                          else "submitted on the briefs without oral argument")
    if heard:
        appeal["argued"] = ym(heard)
    sub = fdate(a["SUBDATE"])
    if sub and not heard:
        appeal["submitted"] = ym(sub)
    appeal["related_appeals"] = {"0": "none", "1": "cross-appeal from the same judgment",
                                 "2": "consolidated with other appeals", "3": "lead case of consolidated appeals"
                                 }.get(a["JOINAPP"], "not recorded")
    if a["REOPEN"] not in ("", "-8"):
        appeal["history"] = ("returned to this court by the U.S. Supreme Court" if a["REOPEN"] in ("M", "S")
                             else "appeal reopened or reinstated after an earlier closing")
    parties = {"appellant": a["appellant_kind"], "appellee": a["appellee_kind"]}
    if a["PROSEFLE"] in ("1", "3"):
        parties["appellant_representation"] = "self-represented (pro se)"
    elif a["PROSEFLE"] in ("0", "2"):
        parties["appellant_representation"] = "represented by counsel"
    if a["PROSEFLE"] in ("2", "3"):
        parties["appellee_representation"] = "self-represented (pro se)"
    if a["FILEFEE"] == "FP":
        parties["appellant_fee_status"] = "in forma pauperis (filing fee waived)"
    out = {"appeal": appeal, "parties": parties}
    if a["APPTYPE"] in ("1", "2"):
        ag = a["AGENCY"].strip().upper()
        out["agency_below"] = {"agency": AGENCY.get(ag, f"federal agency (code {ag})"),
                               "proceeding": "order under review" if a["APPTYPE"] == "1" else "order the agency seeks to enforce"}
        return out
    below = {}
    dist = DISTRICTS.get(a["DDIST"])
    if a["APPTYPE"] == "11":
        below["court"] = "bankruptcy appellate panel"
    elif a["APPTYPE"] == "10":
        below["court"] = "U.S. bankruptcy court" + (f" ({dist[0].split(' for the ', 1)[1]})" if dist else "")
    elif dist:
        below["court"] = dist[0]
    if a["NOS"] in NOS:
        below["nature_of_suit"] = f"{NOS[a['NOS']]} ({a['NOS']})"
    if a["JURIS"] in JURIS:
        below["jurisdiction_basis"] = JURIS[a["JURIS"]]
    ddkt = fdate(a["DDKTDATE"])
    if ddkt:
        below["case_filed"] = ym(ddkt)
    if civ:
        t, s = civ["TITL"], civ["SECTION"]
        if t and t not in ("-8", "0") and s and s not in ("-8", "0"):
            below["cause_statute"] = f"{int(t) if t.isdigit() else t} U.S.C. § {s.lstrip('0') or s}"
        below["how_the_case_ended"] = CIV_DISP.get(civ["DISP"], "not recorded")
        if civ["JUDGMENT"] in ("1", "2", "3"):
            below["judgment_for"] = CIV_JUDGMENT[civ["JUDGMENT"]]
        if civ["PROCPROG"] in CIV_PROCPROG:
            below["stage_reached"] = CIV_PROCPROG[civ["PROCPROG"]]
        if civ["NOJ"] in CIV_NOJ and civ["NOJ"] != "0":
            below["relief_entered"] = CIV_NOJ[civ["NOJ"]]
        if civ["CLASSACT"] == "1":
            below["class_action"] = "class action alleged"
        if civ["JURY"] in CIV_JURY and civ["JURY"] != "N":
            below["jury_demand"] = CIV_JURY[civ["JURY"]]
        if civ["DEMANDED"].isdigit() and 0 < int(civ["DEMANDED"]) < 9999:
            below["amount_demanded"] = f"about ${int(civ['DEMANDED']):,} thousand"
        td, fd = fdate(civ["TERMDATE"]), fdate(civ["FILEDATE"])
        if td:
            below["final_judgment_entered"] = ym(td)
        if td and fd and td >= fd:
            below["years_in_district_court"] = round((td - fd).days / 365.25, 1)
        role = ("plaintiff" if same_party(a.get("appellant_sig"), civ.get("plt_sig"))
                else "defendant" if same_party(a.get("appellant_sig"), civ.get("def_sig")) else None)
        if role:
            parties["appellant_role_below"] = role
    elif a["APPTYPE"] in ("3", "4"):
        # One phrase whether or not the district case appears LATER in the civil file: distinguishing the two would
        # read post-decision records (a remand re-terminates the district case after the appeal).
        below["how_the_case_ended"] = "no district termination on record before the appeal was docketed"
    if below:
        out["case_below"] = below
    return out


def render_memo(state):
    """The text surface: the same facts as labelled lines."""
    lines = []
    for sec in ("appeal", "agency_below", "case_below", "parties"):
        if sec not in state:
            continue
        lines.append(sec.replace("_", " ").upper() + ":")
        for k, v in state[sec].items():
            lines.append(f"- {k.replace('_', ' ')}: {v}")
    return "\n".join(lines)


def render(state, surface):
    return jdump(state) if surface == "json" else render_memo(state)


def blend(onehot_idx, p, K, lam):
    import numpy as np
    t = (1 - lam) * np.asarray(p, dtype=float)
    t[onehot_idx] += lam
    return t / t.sum()


def lab_dict(keys, vec):
    vals = [max(0.0, float(v)) for v in vec]
    s = sum(vals)
    vals = [round(v / s, 6) for v in vals]
    vals[-1] = round(1.0 - sum(vals[:-1]), 6)
    if vals[-1] < 0:   # rounding guard
        vals = [v / sum(max(0, x) for x in vals) for v in (max(0, x) for x in vals)]
    return dict(zip(keys, vals))


def make_questions(rng, keys, criteria):
    order = list(keys)
    rng.shuffle(order)
    return {k: criteria[k] for k in order}


def build_federal(a, cache, rng_seed):
    import numpy as np
    from scipy.special import betaln
    raw_appeals = list(read_gz(cache / "fjc_appeals.jsonl.gz"))
    dedup = {}
    for r in raw_appeals:   # FJC repeats some (circuit, docket, judgment date) rows across tape years: keep the latest
        k = (r["CIRCUIT"], r["DOCKET"], r["JUDGDATE"])
        if k not in dedup or r["TAPEYEAR"] >= dedup[k]["TAPEYEAR"]:
            dedup[k] = r
    appeals = list(dedup.values())
    log(f"federal: {len(raw_appeals):,} in-scope FJC merits terminations, {len(appeals):,} after removing repeated rows")
    civil_by_key = defaultdict(list)
    if (cache / "fjc_civil.jsonl.gz").exists():
        for c in read_gz(cache / "fjc_civil.jsonl.gz"):
            civil_by_key[(c["DISTRICT"], c["DOCKET"].zfill(7))].append(c)
    # CourtListener links
    cl_dockets, cl_clusters, panels = {}, defaultdict(list), defaultdict(list)
    if (cache / "cl_dockets.jsonl.gz").exists():
        for d in read_gz(cache / "cl_dockets.jsonl.gz"):
            k = (d["court_id"], cl_docket_key(d["docket_number"]))
            if k not in cl_dockets or d["source"] > cl_dockets[k]["source"]:
                cl_dockets[k] = d
        for c in read_gz(cache / "cl_clusters.jsonl.gz"):
            cl_clusters[c["docket_id"]].append(c)
        for p in read_gz(cache / "cl_panel.jsonl.gz"):
            panels[p["cluster_id"]].append(p["person_id"])
    # groups: union-find over the appellate lead docket and the district case, so related appeals (cross- and
    # consolidated appeals, successive appeals from one district case) share a group and never straddle splits
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    def nodes_of(r):
        cid = CIRCUIT_ID[r["CIRCUIT"]]
        lead = r["CONSDKT"] if r["JOINAPP"] in ("1", "2") and r["CONSDKT"] not in ("", "-8") else r["DOCKET"]
        ns = [f"dkt:{cid}:{lead.zfill(7)}", f"dkt:{cid}:{r['DOCKET'].zfill(7)}"]
        if r["APPTYPE"] not in ("1", "2") and r["DDIST"] not in ("", "-8") and r["DDOCKET"].strip("0-8"):
            ns.append(f"dist:{r['DDIST']}:{r['DDOCKET'].zfill(7)}")
        return ns
    for r in appeals:
        ns = nodes_of(r)
        for nd in ns[1:]:
            union(ns[0], nd)
    # batches: appeals decided together (same circuit, judgment date, submission / argument date, appeal type, NOS /
    # agency, district and party kinds) are one decision in practice (e.g. hundreds of bankruptcy appeals disposed of by one order). Keep one,
    # chosen at random -- never by outcome -- and merge their groups.
    batches = defaultdict(list)
    for r in appeals:
        batches[(r["CIRCUIT"], r["JUDGDATE"], r["SUBDATE"], r["HEARDATE"], r["APPTYPE"], r["NOS"], r["AGENCY"],
                 r["DDIST"], r["appellant_kind"], r["appellee_kind"])].append(r)
    brng = random.Random(rng_seed + 11)
    kept_appeals, n_collapsed = [], 0
    for members in batches.values():
        for m_ in members[1:]:
            union(nodes_of(members[0])[0], nodes_of(m_)[0])
        kept_appeals.append(members[0] if len(members) == 1 else brng.choice(members))
        n_collapsed += len(members) - 1
    appeals = kept_appeals
    log(f"federal: {len(appeals):,} after collapsing {n_collapsed:,} same-day batch duplicates")
    # items
    items = []
    for r in appeals:
        jd = fdate(r["JUDGDATE"])
        cid = CIRCUIT_ID[r["CIRCUIT"]]
        group = hkey("fed", find(nodes_of(r)[0]))
        if r["APPTYPE"] in ("1", "2"):
            nk = "agency:" + r["AGENCY"].strip().upper()
            fam = "agency"
        elif r["NOS"] not in ("", "-8", "-9"):
            nk, fam = "nos:" + r["NOS"], nos_family(r["NOS"])
        elif r["APPTYPE"] in ("10", "11", "12"):
            nk, fam = "bankruptcy", "bankruptcy"
        else:
            nk, fam = "other", "other"
        inv = [r["JDG1INV"], r["JDG2INV"], r["JDG3INV"]]
        votes_known = all(len(x) == 1 and x.isalpha() for x in inv)
        n_dis = sum(1 for x in inv if x in ("C", "D", "F", "G"))
        if not votes_known and n_dis == 0 and r["OPINION"] in ("6", "8", "9"):
            n_dis = 1
        items.append({"r": r, "year": jd.year, "cid": cid, "group": group, "nk": nk, "fam": fam,
                      "y": DISP_KEYS.index(OUTCOME_KEY[r["OUTCOME"]]), "votes_known": votes_known, "n_dis": n_dis,
                      "jd": jd})
    split_info = [split_of_year(it["year"], it["group"]) for it in items]
    for it, (sp, why) in zip(items, split_info):
        it["split"], it["split_reason"] = sp, why
    N = len(items)
    y = np.array([it["y"] for it in items])
    # base rates are fitted on the TRAIN split only (LOO within it); val and eval items get train-only rates, so no
    # val / eval outcome ever touches a training label
    fit = np.array([it["split"] == "train" for it in items])
    levels = encode_levels([["all"] * N, [it["fam"] for it in items], [it["nk"] for it in items],
                            [it["cid"] + "|" + it["nk"] for it in items],
                            [it["cid"] + "|" + it["nk"] + "|" + str(it["year"]) for it in items]])
    log("federal: fitting hierarchical LOO base rates (global > NOS family > NOS > circuit x NOS > circuit x NOS x year)")
    kap, ll = fit_kappas(levels, y, 5, fit)
    p_hat = loo_rates(levels, y, 5, fit, kap)
    prior = np.bincount(y[fit], minlength=5) / fit.sum()
    rep_disp = {"kappas": kap, "loo_log_score_train": ll, "class_rates_train": dict(zip(DISP_KEYS, prior.round(4).tolist())),
                "loo_on_train": calib_report(p_hat[fit], y[fit]),
                "global_rate_on_train": calib_report(np.tile(prior, (int(fit.sum()), 1)), y[fit])}
    for sp in ("val", "eval"):
        s = np.array([it["split"] == sp for it in items])
        if s.any():
            rep_disp[f"train_only_rates_on_{sp}"] = calib_report(p_hat[s], y[s])
            rep_disp[f"global_rate_on_{sp}"] = calib_report(np.tile(prior, (int(s.sum()), 1)), y[s])
    ev = np.array([it["split"] == "eval" for it in items])
    log(f"  kappas {kap}; LOO log score {ll:.4f}; {rep_disp['loo_on_train']}")
    # dissent: per-judge rate m (hierarchical LOO over cases with known votes), beta-binomial a, posterior predictive
    vk = np.array([it["votes_known"] for it in items])
    c = np.array([it["n_dis"] for it in items], dtype=float)
    nn = np.where(vk, 3.0, 0.0)
    sel = vk & fit
    ydis = (c > 0).astype(int)
    # Diagnostic only: a per-judge beta-binomial (theta ~ Beta(a m, a(1-m)), c of 3 dissent). A three-judge panel has
    # at most one dissenter in practice, so overdispersion is not identified and a runs to the grid bound -- the
    # "fresh panel" posterior predictive then equals the base rate whatever the panel did. Rejected as a label.
    kap_m, _ = fit_binom_kappas(levels, c, nn, sel)
    m = loo_binom_rate(levels, c, nn, sel, kap_m)
    a_hat, a_ll = fit_betabinom_a(c[sel], nn[sel], m[sel])
    al, be = c + a_hat * m, (nn - c) + a_hat * (1 - m)
    p_fresh = 1 - np.exp(betaln(al, be + 3) - betaln(al, be))
    # The label: the vote split decides the realised outcome (>= 1 dissenter of 3), smoothed toward a LOO
    # hierarchical case-level rate fitted on train cases with known votes (index 0 = yes, 1 = no).
    yd = 1 - ydis
    kap_d, ll_d = fit_kappas(levels, yd, 2, sel)
    p_d = loo_rates(levels, yd, 2, sel, kap_d)
    p_dis_prior = p_d[:, 0]
    rep_dis = {"label": "lam*onehot(vote split) + (1-lam)*LOO hierarchical dissent rate", "kappas": kap_d,
               "loo_log_score_train": ll_d, "cases_with_known_votes_train": int(sel.sum()),
               "per_judge_rate_train": float(c[sel].sum() / nn[sel].sum()), "dissent_rate_train": float(ydis[sel].mean()),
               "loo_rate_on_train": calib_report(p_d[sel], yd[sel]),
               "rejected_betabinomial": {"a": a_hat, "grid_max": 2000, "loglik": a_ll, "kappas_per_judge_rate": kap_m,
                                         "fresh_panel_mean_if_dissent": float(p_fresh[sel & (ydis == 1)].mean()) if (sel & (ydis == 1)).any() else None,
                                         "fresh_panel_mean_if_unanimous": float(p_fresh[sel & (ydis == 0)].mean()) if (sel & (ydis == 0)).any() else None}}
    for sp in ("val", "eval"):
        s = vk & np.array([it["split"] == sp for it in items])
        if s.any():
            rep_dis[f"train_only_rate_on_{sp}"] = calib_report(p_d[s], yd[s])
            rep_dis[f"dissent_rate_{sp}"] = float(ydis[s].mean())
    log(f"  dissent: case rate {rep_dis['dissent_rate_train']:.4f}; kappas {kap_d}; (rejected beta-binomial a={a_hat:.3g})")
    # CourtListener disposition cross-check (rough text classifier; order matters)
    agree = Counter()
    cl_word = [("affirm_in_part", re.compile(r"affirm\w*\s+in\s+part|in\s+part.{0,40}revers|revers\w*\s+in\s+part", re.I)),
               ("reverse_or_vacate", re.compile(r"\b(revers|vacat)|petition\w*\s+(for\s+review\s+)?(is\s+)?grant", re.I)),
               ("dismiss", re.compile(r"\bdismiss", re.I)),
               ("affirm", re.compile(r"\baffirm|\benforc|petition\w*\s+(for\s+review\s+)?(is\s+)?den", re.I)),
               ("remand_only", re.compile(r"\bremand", re.I))]
    records = []
    stats = Counter()
    for i, it in enumerate(items):
        if it["split"] is None:
            stats["buffer_2024"] += 1
            continue
        r = it["r"]
        civ, matched_any, _pending = lower_court(r, civil_by_key) if r["APPTYPE"] in ("3", "4", "10", "11", "12") else (None, False, False)
        stats["lower_court_as_of_match" if civ else ("lower_court_no_final_before_appeal" if matched_any else "lower_court_unmatched")] += 1
        cl = cl_dockets.get((it["cid"], r["DOCKET"].zfill(7)))
        cl_meta = None
        if cl:
            cls = [k for k in cl_clusters.get(cl["id"], []) if idate(k["date_filed"]) and abs((idate(k["date_filed"]) - it["jd"]).days) <= 45]
            disp_txt = " | ".join(k["disposition"] for k in cls if k["disposition"]).strip()
            cl_meta = {"docket_id": cl["id"], "cluster_ids": [k["id"] for k in cls],
                       "panel_size": max([len(panels.get(k["id"], [])) for k in cls] or [0]) or None}
            if disp_txt:
                guess = next((k for k, rx in cl_word if rx.search(disp_txt)), None)
                if guess:
                    agree["agree" if guess == DISP_KEYS[it["y"]] else f"fjc={DISP_KEYS[it['y']]}|cl={guess}"] += 1
                cl_meta["disposition_text"] = disp_txt[:200]
            stats["courtlistener_docket_linked"] += 1
            if cls:
                stats["courtlistener_cluster_linked"] += 1
        rng = random.Random(f"{rng_seed}|{it['group']}|{r['DOCKET']}")
        state = federal_state(r, civ, matched_any, rng)
        surface = "json" if int(hkey("surface", it["group"], r["DOCKET"])[:4], 16) % 100 < 60 else "memo"
        text = render(state, surface)
        hits = scan(text)
        if hits:
            stats["scan_hit_masked"] += 1
            text = mask(text)
        records.append((i, it, text, surface, hits, cl_meta, civ is not None))
    # caps (uniform random within split), then labels
    rng = random.Random(rng_seed)
    by_split = defaultdict(list)
    for rec in records:
        by_split[rec[1]["split"]].append(rec)
    caps = {"train": a.max_train, "val": a.max_val, "eval": a.max_eval}
    out = []
    train_states = set()
    # a group (circuit + lead docket) decided in a train-or-val year must not reappear in eval (reopened dockets,
    # Supreme Court remands): checked against ALL items of those years, written or not
    groups_of = defaultdict(set)
    for it in items:
        if it["split"]:
            groups_of[it["split"]].add(it["group"])
    for sp in ("train", "val", "eval"):
        pool = by_split.get(sp, [])
        rng.shuffle(pool)
        if caps[sp] and len(pool) > caps[sp]:
            pool = pool[:caps[sp]]
        for (i, it, text, surface, hits, cl_meta, has_civ) in pool:
            if sp == "eval" and (it["group"] in groups_of["train"] or it["group"] in groups_of["val"]):
                stats["eval_dropped_group_seen_in_train_or_val"] += 1
                continue
            if sp == "train":
                train_states.add(text)
            elif text in train_states:
                stats[f"{sp}_dropped_state_identical_to_train"] += 1
                continue
            r = it["r"]
            qrng = random.Random(f"{rng_seed}|q|{it['group']}|{r['DOCKET']}")
            yk = DISP_KEYS[it["y"]]
            p = p_hat[i]
            t_disp = blend(it["y"], p, 5, a.lam)
            qs = {"appellate_disposition": choice_q(DISP_INSTR, make_questions(qrng, DISP_KEYS, DISP_CRITERIA)),
                  "degree_of_change": score_q(CHANGE_INSTR, CHANGE_LEVELS)}
            if sp == "eval":
                ld = {k: (1.0 if k == yk else 0.0) for k in DISP_KEYS}
            else:
                ld = lab_dict(DISP_KEYS, t_disp)
            labs = {"appellate_disposition": {k: ld[k] for k in qs["appellate_disposition"]["criteria"]}}
            ch = [0.0, 0.0, 0.0]
            for k, v in ld.items():
                ch[CHANGE_OF[k]] += v
            labs["degree_of_change"] = lab_dict(CHANGE_LEVELS, ch)
            label_source = {"appellate_disposition": "outcome", "degree_of_change": "outcome"}
            label_method = {"appellate_disposition": "O:onehot" if sp == "eval" else "O:lam*onehot+(1-lam)*loo_hier_rate",
                            "degree_of_change": "derived:sum_disposition_mass_by_change_level"}
            raw = {"appellate_disposition": yk, "fjc_outcome_code": r["OUTCOME"], "fjc_disp_code": r["DISP"],
                   "fjc_judge_involvement": "".join(x if x else "-" for x in (r["JDG1INV"], r["JDG2INV"], r["JDG3INV"]))}
            t_dis = None
            if it["votes_known"]:
                qs["panel_dissent"] = noul_q(DISSENT_INSTR)
                t_dis = blend(int(yd[i]), p_d[i], 2, a.lam)
                if sp == "eval":
                    labs["panel_dissent"] = {"yes": float(it["n_dis"] > 0), "no": float(it["n_dis"] == 0)}
                    label_method["panel_dissent"] = "V3:onehot(split)"
                else:
                    labs["panel_dissent"] = lab_dict(["yes", "no"], t_dis)
                    label_method["panel_dissent"] = "V3:lam*onehot(split)+(1-lam)*loo_hier_rate"
                label_source["panel_dissent"] = "votes:3"
                raw["panel_dissent"] = "yes" if it["n_dis"] > 0 else "no"
                raw["vote_split"] = f"{3 - int(it['n_dis'])}-{int(it['n_dis'])}"
            kinds = {q: ("hard" if sum(1 for v in labs[q].values() if v > 0) == 1 else "soft") for q in qs}
            source = f"courts/{it['cid']}"
            rec = record(source, text, qs, labs, kinds, None)
            sig = json.dumps({k: option_keys(v) for k, v in qs.items()}, sort_keys=True)
            case_key = hkey("fjc", r["CIRCUIT"], r["DOCKET"], r["JUDGDATE"])
            rec["id"] = f"{source}-{sha(case_key + text + sig)}"
            rec["base_id"] = rec["id"]
            rec["license"] = LICENSE_FEDERAL
            as_of = max([d for d in (fdate(r["DKTDATE"]), fdate(r["BRFILED"]), fdate(r["SUBDATE"]), fdate(r["HEARDATE"])) if d]
                        or [fdate(r["DKTDATE"])])
            meta = {"dataset": f"fjc_idb:ap08on(2026-09-01)+cv88on(2026-08-26); courtlistener_bulk:{CL_SNAPSHOT}",
                    "group": it["group"], "case_key": case_key, "as_of": as_of.isoformat() if as_of else None,
                    "decision_date": it["jd"].isoformat(), "split_reason": it["split_reason"], "surface": surface,
                    "label_source": label_source, "label_method": label_method, "n_raters": 3 if it["votes_known"] else None,
                    "raw_outcome": raw, "lambda": a.lam if sp != "eval" else None,
                    "base_rate": {"appellate_disposition": lab_dict(DISP_KEYS, p),
                                  "panel_dissent": (lab_dict(["yes", "no"], p_d[i]) if it["votes_known"] else None)},
                    "smoothed_target": ({"appellate_disposition": lab_dict(DISP_KEYS, t_disp),
                                         "panel_dissent": lab_dict(["yes", "no"], t_dis) if t_dis is not None else None}
                                        if sp == "eval" else None),
                    "base_rate_cell": f"{it['cid']}|{it['nk']}|{it['year']}",
                    "court_docket": f"{it['cid']} {r['DOCKET'][:2]}-{r['DOCKET'][2:].lstrip('0') or '0'}",
                    "lower_court_record": "fjc_civil_as_of_appeal" if has_civ else None,
                    "fjc_pubstat": r["PUBSTAT"], "courtlistener": cl_meta, "outcome_scan_hits": hits or None}
            rec["meta"] = {k: v for k, v in meta.items() if v is not None}
            validate(rec)
            out.append(rec)
            stats[f"written_{sp}"] += 1
    rep = {"disposition_label_model": rep_disp, "dissent_label_model": rep_dis,
           "courtlistener_disposition_check": dict(agree.most_common()), "stats": dict(stats)}
    return out, rep


# --------------------------------------------------------------------------- build: state courts of last resort
def build_state(a, cache, rng_seed):
    import numpy as np
    rows = list(read_gz(cache / "sscdp.jsonl.gz"))
    cites = defaultdict(list)
    if (cache / "cl_citations.jsonl.gz").exists():
        for c in read_gz(cache / "cl_citations.jsonl.gz"):
            cites[(c["reporter"], c["volume"], c["page"])].append(c["cluster_id"])
    clusters = {}
    if (cache / "cl_clusters.jsonl.gz").exists():
        for c in read_gz(cache / "cl_clusters.jsonl.gz"):
            clusters[c["id"]] = c
    items = []
    stats = Counter()
    for r in rows:
        try:
            st = STATES[int(r["state_2"]) - 1]
            yr = int(r["decs_yr"])
        except (ValueError, IndexError):
            stats["bad_state_or_year"] += 1
            continue
        n = r["votes_pet"] + r["votes_res"]
        if n < 3:
            stats["fewer_than_3_votes"] += 1
            continue
        area = SS_AREA[r["areas"][0]] if r["areas"] else ("civil, government party" if r["genissue"] == "2" else "civil, private parties")
        # one opinion can carry several docket numbers (rows sharing a reporter citation): the citation is the case
        group = (hkey("sscdp-cite", r["state_2"], r["reporter_str"], r["rep_vol"], r["rep_page"])
                 if r.get("reporter_str") and r.get("rep_vol", "").isdigit() and r.get("rep_page", "").isdigit()
                 else hkey("sscdp-row", r["masterid"], r.get("id_1"), r.get("casen_1"), r.get("docket")))
        items.append({"r": r, "state": st, "year": yr, "area": area, "n": n, "c_pet": r["votes_pet"],
                      "y": SS_KEYS.index(SS_DISP[r["disp"]]),
                      "div": int(min(r["votes_pet"], r["votes_res"]) > 0), "group": group})
    rng0 = random.Random(rng_seed + 3)
    rng0.shuffle(items)
    seen = set()
    uniq = []
    for it in items:          # one record per case (citation); the pick is random, not outcome-dependent
        if it["group"] in seen:
            stats["collapsed_same_citation"] += 1
            continue
        seen.add(it["group"])
        uniq.append(it)
    items = uniq
    N = len(items)
    for it in items:
        it["split"] = "val" if int(it["group"][:8], 16) % 100 < VAL_PCT else "train"
    y = np.array([it["y"] for it in items])
    fit = np.array([it["split"] == "train" for it in items])
    val = ~fit
    levels = encode_levels([["all"] * N, [it["area"] for it in items], [it["state"] + "|" + it["area"] for it in items],
                            [it["state"] + "|" + it["area"] + "|" + str(it["year"]) for it in items]])
    kap, ll = fit_kappas(levels, y, 4, fit)
    p_hat = loo_rates(levels, y, 4, fit, kap)
    prior = np.bincount(y[fit], minlength=4) / fit.sum()
    rep = {"disposition_label_model": {"kappas": kap, "loo_log_score_train": ll,
                                       "class_rates_train": dict(zip(SS_KEYS, prior.round(4).tolist())),
                                       "loo_on_train": calib_report(p_hat[fit], y[fit]),
                                       "train_only_rates_on_val": calib_report(p_hat[val], y[val]),
                                       "global_rate_on_val": calib_report(np.tile(prior, (int(val.sum()), 1)), y[val])}}
    ydiv = np.array([it["div"] for it in items])
    kd, lld = fit_kappas(levels, ydiv, 2, fit)
    p_div = loo_rates(levels, ydiv, 2, fit, kd)
    pdv = np.bincount(ydiv[fit], minlength=2) / fit.sum()
    rep["divided_label_model"] = {"kappas": kd, "divided_rate_train": float(pdv[1]),
                                  "loo_on_train": calib_report(p_div[fit], ydiv[fit]),
                                  "train_only_rates_on_val": calib_report(p_div[val], ydiv[val]),
                                  "global_rate_on_val": calib_report(np.tile(pdv, (int(val.sum()), 1)), ydiv[val])}
    # justices' votes: per-justice pro-petitioner share m (LOO hierarchical), beta-binomial a, posterior mean
    c = np.array([it["c_pet"] for it in items], dtype=float)
    nn = np.array([it["n"] for it in items], dtype=float)
    kap_m, _ = fit_binom_kappas(levels, c, nn, fit)
    m = loo_binom_rate(levels, c, nn, fit, kap_m)
    a_hat, a_ll = fit_betabinom_a(c[fit], nn[fit], m[fit])
    g = (c + a_hat * m) / (nn + a_hat)
    rep["vote_label_model"] = {"a": a_hat, "kappas_vote_share": kap_m, "beta_binomial_loglik": a_ll,
                               "mean_pro_petitioner_share_train": float(c[fit].sum() / nn[fit].sum())}
    log(f"state: {N:,} cases; disposition kappas {kap}; vote a={a_hat:.3g}")
    out = []
    rng = random.Random(rng_seed + 7)
    rng.shuffle(items_idx := list(range(N)))
    if a.max_state and len(items_idx) > a.max_state:
        items_idx = items_idx[:a.max_state]
    for i in items_idx:
        it = items[i]
        r = it["r"]
        cl_id, court_name = STATE_COURT[it["state"]]
        sp = it["split"]
        roles = {}
        if r["o_plain"] in ("1", "2"):
            roles["original_plaintiff"] = "petitioner / appellant" if r["o_plain"] == "1" else "respondent / appellee"
        if r["o_defend"] in ("1", "2"):
            roles["original_defendant"] = "petitioner / appellant" if r["o_defend"] == "1" else "respondent / appellee"
        state = {"court": {"court": court_name, "state": it["state"], "decided_in": it["year"],
                           "review_sought_by": SS_JURIS[r["juris"]], "justices_participating": it["n"]},
                 "case_below": {"court_under_review": SS_COURT.get(r["rev_ct"], "not recorded"),
                                "first_heard_in": SS_COURT.get(r["first_ct"], "not recorded")},
                 "parties": {"petitioner": r["p1"] or "not recorded", "respondent": r["r1"] or "not recorded"},
                 "issues": {"case_type": "civil, government party" if r["genissue"] == "2" else "civil, private parties"}}
        if r["dec1_yr"].isdigit() and 1983 <= int(r["dec1_yr"]) <= 1998:
            state["case_below"]["first_decision_year"] = int(r["dec1_yr"])
        if r["agency"] == "1":
            state["case_below"]["agency_action_before_litigation"] = "yes"
        if r["arbitrat"] == "1":
            state["case_below"]["arbitration_or_mediation"] = "yes"
        if r["p2"]:
            state["parties"]["second_petitioner"] = r["p2"]
        if r["r2"]:
            state["parties"]["second_respondent"] = r["r2"]
        state["parties"].update(roles)
        for flag, txt in (("crossapp", "both sides appealed"), ("classact", "class action"), ("amicus", "amicus briefs filed"),
                          ("multi_p", "several petitioners"), ("multi_r", "several respondents")):
            if r[flag] == "1":
                state["parties"].setdefault("notes", []).append(txt)
        if r["areas"]:
            state["issues"]["subject_areas"] = [SS_AREA[k] for k in r["areas"]]
        subj = [SS_ISSUES[k] for k in r["issues"] if not k.startswith("cv_")]
        proc = [SS_ISSUES[k] for k in r["issues"] if k.startswith("cv_")]
        if subj:
            state["issues"]["subject_matter"] = subj
        if proc:
            state["issues"]["legal_questions_raised"] = proc
        if r["usc"]:
            state["issues"]["federal_constitutional_challenge_to_a_law"] = "yes"
        if r["stc"]:
            state["issues"]["state_constitutional_challenge_to_a_law"] = "yes"
        surface = "json" if int(hkey("surface", it["group"])[:4], 16) % 100 < 60 else "memo"
        text = jdump(state) if surface == "json" else "\n".join(
            [sec.replace("_", " ").upper() + ":\n" + "\n".join(f"- {k.replace('_', ' ')}: {', '.join(v) if isinstance(v, list) else v}"
                                                               for k, v in state[sec].items()) for sec in state])
        hits = scan(text)
        if hits:
            stats["scan_hit_masked"] += 1
            text = mask(text)
        qrng = random.Random(f"{rng_seed}|sq|{it['group']}")
        yk = SS_KEYS[it["y"]]
        t_disp = blend(it["y"], p_hat[i], 4, a.lam)
        ld = lab_dict(SS_KEYS, t_disp)
        qs = {"state_supreme_disposition": choice_q(SS_DISP_INSTR, make_questions(qrng, SS_KEYS, SS_CRITERIA)),
              "justice_votes_for_petitioner": noul_q(SS_VOTE_INSTR),
              "court_divided": noul_q(SS_DIV_INSTR),
              "degree_of_change": score_q(CHANGE_INSTR, CHANGE_LEVELS)}
        ch = [0.0, 0.0, 0.0]
        for k, v in ld.items():
            ch[CHANGE_OF[k]] += v
        t_div = blend(1 - it["div"], [p_div[i][1], p_div[i][0]], 2, a.lam)   # order: yes, no
        labs = {"state_supreme_disposition": {k: ld[k] for k in qs["state_supreme_disposition"]["criteria"]},
                "justice_votes_for_petitioner": lab_dict(["yes", "no"], [g[i], 1 - g[i]]),
                "court_divided": lab_dict(["yes", "no"], t_div),
                "degree_of_change": lab_dict(CHANGE_LEVELS, ch)}
        kinds = {q: ("hard" if sum(1 for v in labs[q].values() if v > 0) == 1 else "soft") for q in qs}
        source = f"courts/{cl_id}"
        rec = record(source, text, qs, labs, kinds, None)
        sig = json.dumps({k: option_keys(v) for k, v in qs.items()}, sort_keys=True)
        rec["id"] = f"{source}-{sha(it['group'] + text + sig)}"
        rec["base_id"] = rec["id"]
        rec["license"] = LICENSE_STATE
        cl_cl = cites.get((r["reporter_str"], r["rep_vol"], r["rep_page"]), []) if r["reporter_str"] else []
        cl_ok = [cid for cid in cl_cl if cid in clusters and str(clusters[cid].get("date_filed", ""))[:4] == str(it["year"])]
        stats["courtlistener_cluster_linked"] += bool(cl_ok)
        rec["meta"] = {
            "dataset": "sscdp:NSF 95-98 Beta Version_July 2005 (doi:10.7910/DVN/Z80F7P)",
            "group": it["group"], "case_key": it["group"], "as_of": f"{it['year']}", "decision_date": f"{it['year']}-{r['decs_mo'].zfill(2) if r['decs_mo'].isdigit() and int(r['decs_mo']) <= 12 else '??'}",
            "split_reason": f"{sp}: state-court cases 1995-98 are train/val only ({VAL_PCT}% hash sample to val)",
            "surface": surface,
            "label_source": {"state_supreme_disposition": "outcome", "degree_of_change": "outcome",
                             "justice_votes_for_petitioner": f"votes:{it['n']}", "court_divided": "outcome"},
            "label_method": {"state_supreme_disposition": "O:lam*onehot+(1-lam)*loo_hier_rate",
                             "justice_votes_for_petitioner": "V:dirichlet_posterior_mean",
                             "court_divided": "O:lam*onehot+(1-lam)*loo_hier_rate",
                             "degree_of_change": "derived:sum_disposition_mass_by_change_level"},
            "n_raters": it["n"],
            "raw_outcome": {"state_supreme_disposition": yk, "votes_for_petitioner": it["c_pet"],
                            "votes_for_respondent": it["n"] - it["c_pet"], "court_divided": "yes" if it["div"] else "no"},
            "lambda": a.lam, "base_rate": {"state_supreme_disposition": lab_dict(SS_KEYS, p_hat[i]),
                                           "pro_petitioner_vote_share": round(float(m[i]), 6)},
            "courtlistener": {"cluster_ids": cl_ok} if cl_ok else None,
            "outcome_scan_hits": hits or None}
        rec["meta"] = {k: v for k, v in rec["meta"].items() if v is not None}
        validate(rec)
        out.append((sp, rec))
    train_states = {rec["state"] for sp, rec in out if sp == "train"}
    kept = []
    for sp, rec in out:
        if sp == "val" and rec["state"] in train_states:
            stats["val_dropped_state_identical_to_train"] += 1
            continue
        kept.append((sp, rec))
        stats[f"written_{sp}"] += 1
    rep["stats"] = dict(stats)
    return kept, rep


def build(a):
    cache, outdir = Path(a.cache), Path(a.out)
    for sp in ("train", "val", "eval"):
        d = outdir / sp
        if d.exists():
            for p in d.glob("courts__*.jsonl"):
                p.unlink()
    files = {}

    def write(sp, rec):
        key = (sp, rec["source"])
        if key not in files:
            p = outdir / sp / (rec["source"].replace("/", "__") + ".jsonl")
            p.parent.mkdir(parents=True, exist_ok=True)
            files[key] = p.open("w", encoding="utf-8")
        files[key].write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {"built": datetime.now().isoformat(timespec="seconds"), "lambda": a.lam, "seed": a.seed,
                "label_methods": LABEL_METHODS,
                "licenses": {"federal": LICENSE_FEDERAL, "state": LICENSE_STATE},
                "splits": {"train": f"decided <= {TRAIN_MAX_YEAR} minus val", "val": f"{VAL_PCT}% hash sample by group of the train years",
                           "eval": f"decided {EVAL_YEARS[0]}-{EVAL_YEARS[-1]} (one-hot outcome labels)"},
                "caps": {"train": a.max_train, "val": a.max_val, "eval": a.max_eval, "state": a.max_state}}
    if not a.skip_federal:
        fed, rep = build_federal(a, cache, a.seed)
        for rec in fed:
            sp = rec["meta"]["split_reason"].split(":")[0]
            write(sp, rec)
        manifest["federal"] = rep
        f = cache / "fjc_appeals.filter.json"
        if f.exists():
            manifest["federal"]["fjc_filter"] = json.loads(f.read_text())
    if not a.skip_state and (cache / "sscdp.jsonl.gz").exists():
        st, rep = build_state(a, cache, a.seed)
        for sp, rec in st:
            write(sp, rec)
        manifest["state"] = rep
        f = cache / "sscdp.filter.json"
        if f.exists():
            manifest["state"]["sscdp_filter"] = json.loads(f.read_text())
    for fh in files.values():
        fh.close()
    counts = Counter()
    for (sp, src) in files:
        p = outdir / sp / (src.replace("/", "__") + ".jsonl")
        counts[sp] += sum(1 for _ in p.open())
    manifest["records"] = dict(counts)
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    log(f"wrote {dict(counts)} records to {outdir}")


# --------------------------------------------------------------------------- check
def load_cf():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cache_features", HERE / "trainers" / "cache_features.py")
    cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf)
    return cf


def entropy(v):
    return -sum(x * math.log(x) for x in v if x > 0)


def leak_probe(recs, qid, keys, folds=5, dim=1 << 15, seed=0):
    """Cross-validated hashed bag-of-words multinomial logistic (state -> realised label) vs the LOO base rate."""
    import numpy as np
    from scipy.optimize import minimize
    from scipy.sparse import csr_matrix
    rows, cols, ys, base = [], [], [], []
    for r in recs:
        y = keys.index(r["meta"]["raw_outcome"][qid])
        toks = re.findall(r"[a-z0-9]+", r["state"].lower())
        feats = {int(hashlib.md5(t.encode()).hexdigest()[:8], 16) % dim for t in toks}
        feats |= {int(hashlib.md5((a + "_" + b).encode()).hexdigest()[:8], 16) % dim for a, b in zip(toks, toks[1:])}
        rows.append(sorted(feats))
        ys.append(y)
        br = r["meta"]["base_rate"].get(qid)
        base.append([br[k] for k in keys] if br else None)
    n, K = len(ys), len(keys)
    X = csr_matrix((np.ones(sum(len(f) for f in rows)), np.concatenate([np.array(f, dtype=np.int64) for f in rows]),
                    np.cumsum([0] + [len(f) for f in rows])), shape=(n, dim))
    y = np.array(ys)
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, n)

    def fit(Xt, yt, lam):
        Y = np.eye(K)[yt]

        def f(w):
            W = w[:-K].reshape(dim, K)
            Z = Xt @ W + w[-K:]
            Z -= Z.max(1, keepdims=True)
            lse = np.log(np.exp(Z).sum(1))
            Pp = np.exp(Z - lse[:, None])
            obj = -((Z * Y).sum(1) - lse).sum() + 0.5 * lam * (W ** 2).sum()
            return obj / len(yt), np.concatenate([(Xt.T @ (Pp - Y) + lam * W).ravel(), (Pp - Y).sum(0)]) / len(yt)
        return minimize(f, np.zeros(dim * K + K), jac=True, method="L-BFGS-B", options={"maxiter": 300}).x

    def predict(w, Xs):
        Z = Xs @ w[:-K].reshape(dim, K) + w[-K:]
        Z -= Z.max(1, keepdims=True)
        return np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
    cv = {}
    for lam in (3.0, 30.0, 300.0):
        P = np.zeros((n, K))
        for k in range(folds):
            P[fold == k] = predict(fit(X[fold != k], y[fold != k], lam), X[fold == k])
        cv[lam] = calib_report(P, y)
    best_lam = min(cv, key=lambda l: cv[l]["log_loss"])
    have = np.array([b is not None for b in base])
    B = np.array([b if b is not None else [1.0 / K] * K for b in base])
    prior = np.bincount(y, minlength=K) / n
    out = {"n": n, "bow_cv_by_l2": {str(k): v for k, v in cv.items()}, "bow_cv_best": cv[best_lam],
           "base_rate": calib_report(B[have], y[have]) if have.any() else None,
           "global_prior": calib_report(np.tile(prior, (n, 1)), y)}
    out["bow_minus_base_rate_log_loss"] = (out["bow_cv_best"]["log_loss"] - out["base_rate"]["log_loss"]
                                           if out["base_rate"] else None)
    res_x = fit(X, y, best_lam)
    # top features of a full fit, mapped back to tokens
    vocab = {}
    for r in recs:
        toks = re.findall(r"[a-z0-9]+", r["state"].lower())
        for t in toks:
            vocab.setdefault(int(hashlib.md5(t.encode()).hexdigest()[:8], 16) % dim, t)
        for a_, b_ in zip(toks, toks[1:]):
            vocab.setdefault(int(hashlib.md5((a_ + "_" + b_).encode()).hexdigest()[:8], 16) % dim, a_ + " " + b_)
    W = res_x[:-K].reshape(dim, K)
    out["top_features"] = {keys[j]: [vocab.get(int(ix), str(ix)) for ix in np.argsort(-(W[:, j] - W.mean(1)))[:12]]
                           for j in range(K)}
    return out


def field_audit(recs, qid, keys, min_n=200):
    """For JSON-surface states: per (section.field = value), the realised-outcome mix; flags values whose top class
    share exceeds the global top share by > 0.25 (a field that nearly determines the outcome deserves a look)."""
    glob = Counter(r["meta"]["raw_outcome"][qid] for r in recs)
    top_glob = max(glob.values()) / sum(glob.values())
    by = defaultdict(Counter)
    for r in recs:
        if r["meta"].get("surface") != "json":
            continue
        st = json.loads(r["state"])
        y = r["meta"]["raw_outcome"][qid]
        for sec, d in st.items():
            for k, v in d.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    continue
                if isinstance(v, list):
                    v = "|".join(map(str, v))
                if k in ("docketed", "argued", "submitted", "case_filed", "final_judgment_entered"):
                    continue
                by[f"{sec}.{k}={v}"][y] += 1
    flags = []
    for fv, c in by.items():
        n = sum(c.values())
        if n >= min_n:
            k, top = c.most_common(1)[0]
            if top / n - top_glob > 0.25 or (k != glob.most_common(1)[0][0] and top / n > 0.5):
                flags.append({"field": fv, "n": n, "mix": {kk: round(vv / n, 3) for kk, vv in c.most_common()}})
    return {"global_mix": {k: round(v / sum(glob.values()), 3) for k, v in glob.most_common()},
            "values_checked": sum(1 for c in by.values() if sum(c.values()) >= min_n), "flags": flags}


def check(a):
    import numpy as np
    outdir = Path(a.out)
    cf = load_cf()
    report = {}
    recs = defaultdict(list)
    for sp in ("train", "val", "eval"):
        for p in sorted((outdir / sp).glob("courts__*.jsonl")):
            for line in p.open(encoding="utf-8"):
                r = json.loads(line)
                validate(r)
                for qid, q in r["questions"].items():
                    keys, texts, target = cf.question_options(q, r["labels"][qid])
                    assert keys == option_keys(q), (r["id"], qid)
                    assert abs(sum(target) - 1) < 1e-6 and all(t >= 0 for t in target), (r["id"], qid)
                    assert [r["labels"][qid][k] for k in keys] == target, (r["id"], qid)
                recs[sp].append(r)
    units, kinds, per_src = Counter(), Counter(), Counter()
    for u in cf.iter_units(str(outdir), ["train", "val", "eval"], set(), None):
        assert abs(sum(u["target"]) - 1) < 1e-6
        units[(u["split"], u["qtype"])] += 1
        kinds[(u["split"], u["label_kind"])] += 1
        per_src[(u["split"], u["source"])] += 1
    report["iter_units"] = {"units_by_split_type": {f"{s}/{t}": n for (s, t), n in sorted(units.items())},
                            "label_kind": {f"{s}/{k}": n for (s, k), n in sorted(kinds.items())},
                            "records": {sp: len(v) for sp, v in recs.items()},
                            "sources": len({s for _, s in per_src})}
    # ids / groups / overlap
    ids = [r["id"] for v in recs.values() for r in v]
    groups = {sp: {r["meta"]["group"] for r in v} for sp, v in recs.items()}
    states = {sp: {r["state"] for r in v} for sp, v in recs.items()}
    report["dedup"] = {"duplicate_ids": len(ids) - len(set(ids)),
                       "groups_train_and_eval": len(groups.get("train", set()) & groups.get("eval", set())),
                       "groups_train_and_val": len(groups.get("train", set()) & groups.get("val", set())),
                       "groups_val_and_eval": len(groups.get("val", set()) & groups.get("eval", set())),
                       "identical_states_train_and_eval": len(states.get("train", set()) & states.get("eval", set())),
                       "identical_states_train_and_val": len(states.get("train", set()) & states.get("val", set())),
                       "duplicate_states_within_train": len(recs["train"]) - len(states.get("train", set()))}
    # outcome-word scan on every state
    scan_hits = Counter()
    for v in recs.values():
        for r in v:
            for h in scan(r["state"]):
                scan_hits[h] += 1
    report["outcome_word_scan_on_written_states"] = dict(scan_hits) or "0 hits"
    report["masked_at_build"] = sum(1 for v in recs.values() for r in v if r["meta"].get("outcome_scan_hits"))
    # label distribution
    dist = {}
    for sp, v in recs.items():
        for fam in ("federal", "state"):
            mx, en, n = 0.0, 0.0, 0
            perq = defaultdict(lambda: [0.0, 0.0, 0])
            for r in v:
                if (r["source"] in {f"courts/{c}" for c in CIRCUIT_ID.values()}) != (fam == "federal"):
                    continue
                for qid, lab in r["labels"].items():
                    vals = list(lab.values())
                    mx += max(vals)
                    en += entropy(vals)
                    n += 1
                    perq[qid][0] += max(vals)
                    perq[qid][1] += entropy(vals)
                    perq[qid][2] += 1
            if n:
                dist[f"{sp}/{fam}"] = {"units": n, "mean_max_p": round(mx / n, 4), "mean_entropy_nats": round(en / n, 4),
                                       "per_question": {q: {"n": c, "mean_max_p": round(s / c, 4), "mean_entropy": round(e / c, 4)}
                                                        for q, (s, e, c) in perq.items()}}
    report["label_distribution"] = dist
    report["typed_decisions_gold_reference"] = {"mean_max_p": 0.659, "mean_entropy_nats": 0.767}
    # realised outcome distributions
    outc = defaultdict(Counter)
    for sp, v in recs.items():
        for r in v:
            ro = r["meta"]["raw_outcome"]
            for q in ("appellate_disposition", "panel_dissent", "state_supreme_disposition", "court_divided"):
                if q in ro:
                    outc[f"{sp}/{q}"][ro[q]] += 1
    report["realised_outcomes"] = {k: dict(c) for k, c in sorted(outc.items())}
    # exclusion list
    sys.path.insert(0, str(HERE))
    try:
        from synth_workflows.exclusion import exclusion_errors

        class _S:
            def __init__(self, name, qs):
                self.name, self._qs = name, qs

            def questions(self):
                return self._qs

        class _W:
            def __init__(self, name, schemas):
                self.name, self.schemas = name, schemas
        allq = sorted({q for v in recs.values() for r in v for q in r["questions"]})
        wf = _W("court_adjudication_outcomes", [_S("federal_appeals", {q: None for q in allq if q in ("appellate_disposition", "panel_dissent", "degree_of_change")}),
                                               _S("state_supreme", {q: None for q in allq if q in ("state_supreme_disposition", "justice_votes_for_petitioner", "court_divided", "degree_of_change")})])
        report["exclusion_errors"] = exclusion_errors(wf) or "none"
        report["qids"] = allq
    except Exception as e:  # noqa: BLE001
        report["exclusion_errors"] = f"could not run: {type(e).__name__}: {e}"
    # leak probe (cross-validated bag-of-words vs base rate), on train
    fed_train = [r for r in recs["train"] if "appellate_disposition" in r["questions"]]
    rng = random.Random(0)
    if fed_train:
        sample = rng.sample(fed_train, min(a.probe_n, len(fed_train)))
        report["leak_probe_federal_disposition"] = leak_probe(sample, "appellate_disposition", DISP_KEYS)
        report["field_audit_federal_disposition"] = field_audit(fed_train, "appellate_disposition", DISP_KEYS)
        dis = [r for r in fed_train if "panel_dissent" in r["meta"]["raw_outcome"]]
        report["field_audit_federal_dissent"] = field_audit(dis, "panel_dissent", ["yes", "no"])
    fed_eval = [r for r in recs["eval"] if "appellate_disposition" in r["questions"]]
    if fed_eval:
        report["field_audit_federal_disposition_eval"] = field_audit(fed_eval, "appellate_disposition", DISP_KEYS, 100)
    st_train = [r for r in recs["train"] if "state_supreme_disposition" in r["questions"]]
    if st_train:
        sample = rng.sample(st_train, min(a.probe_n, len(st_train)))
        report["leak_probe_state_disposition"] = leak_probe(sample, "state_supreme_disposition", SS_KEYS)
        report["field_audit_state_disposition"] = field_audit(st_train, "state_supreme_disposition", SS_KEYS, 100)
    # sizes
    report["bytes_on_disk"] = sum(p.stat().st_size for p in outdir.rglob("*.jsonl"))
    (outdir / "check_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))
    # samples
    samples = []
    for sp, src_pref in (("train", "courts/ca"), ("train", "courts/cadc"), ("eval", "courts/ca"), ("val", "courts/ca"), ("train", "courts/")):
        pool = [r for r in recs[sp] if r["source"].startswith(src_pref) and (src_pref != "courts/" or "state_supreme_disposition" in r["questions"])]
        if pool:
            samples.append(random.Random(len(samples)).choice(pool))
    (outdir / "samples.json").write_text(json.dumps(samples, indent=1, ensure_ascii=False))


# --------------------------------------------------------------------------- self-test (no network)
def self_test():
    import numpy as np
    assert cl_docket_key("19-55555") == "1955555" and cl_docket_key("No. 19-1234-cv") == "1901234"
    assert cl_docket_key("2019-1234") is None or cl_docket_key("2019-1234") == "1901234"
    assert scan("The district court granted summary judgment") == ["granted"]
    assert scan("judgment on a motion before trial") == []
    assert mask("We affirm.") == "We [masked]."
    assert party_kind("ACME CORP", False) == "business" and party_kind("CITY OF X", False) == "government body or official"
    assert party_kind("SMITH", False) == "individual" and party_kind("ANY", True).startswith("United States")
    assert same_party(name_sig("SMITH, JOHN"), name_sig("SMITH JOHN ET AL")) and not same_party(name_sig("SMITH"), name_sig("JONES"))
    # LOO rates: a case's own outcome never enters its own rate
    keys = encode_levels([["all"] * 4, ["a", "a", "b", "b"]])
    y = np.array([0, 1, 0, 0])
    p = loo_rates(keys, y, 2, np.ones(4, dtype=bool), [1.0, 1.0])
    p2 = loo_rates(keys, np.array([1, 1, 0, 0]), 2, np.ones(4, dtype=bool), [1.0, 1.0])
    assert abs(p[0] - p2[0]).max() < 1e-12, "item 0's own label changed its rate"
    # uncounted items (val / eval) never move anyone's rate
    p3 = loo_rates(keys, np.array([0, 1, 0, 1]), 2, np.array([True, True, True, False]), [1.0, 1.0])
    p4 = loo_rates(keys, np.array([0, 1, 0, 0]), 2, np.array([True, True, True, False]), [1.0, 1.0])
    assert abs(p3 - p4).max() < 1e-12
    m = loo_binom_rate(keys, np.array([1.0, 0, 0, 0]), np.array([3.0, 3, 3, 3]), np.ones(4, dtype=bool), [10.0, 10.0])
    m2 = loo_binom_rate(keys, np.array([0.0, 0, 0, 0]), np.array([3.0, 3, 3, 3]), np.ones(4, dtype=bool), [10.0, 10.0])
    assert abs(m[0] - m2[0]) < 1e-12
    t = blend(1, [0.7, 0.2, 0.1], 3, 0.5)
    assert abs(t.sum() - 1) < 1e-12 and abs(t[1] - 0.6) < 1e-12
    d = lab_dict(["x", "y", "z"], [0.2, 0.3, 0.5])
    assert abs(sum(d.values()) - 1) < 1e-9
    # every structured vocabulary entry passes the outcome-word scan
    for table in (APPTYPE, JURIS, CIV_DISP, CIV_JUDGMENT, CIV_NOJ, CIV_PROCPROG, NOS, AGENCY, SS_JURIS, SS_COURT, SS_ISSUES, SS_AREA):
        for v in table.values():
            assert not scan(v), (v, scan(v))
    for v in DISTRICTS.values():
        assert not scan(v[0]), v
    print("self-test ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=["fetch", "build", "check", "auth-check", "all"])
    ap.add_argument("--cache", default="/tmp/courts_jev/cache")
    ap.add_argument("--out", default="/tmp/courts_jev")
    ap.add_argument("--cl-snapshot", default=CL_SNAPSHOT)
    ap.add_argument("--skip-cl", action="store_true", help="fetch without the CourtListener bulk streams")
    ap.add_argument("--skip-federal", action="store_true")
    ap.add_argument("--skip-state", action="store_true")
    ap.add_argument("--lam", type=float, default=LAMBDA, help="weight on the realised outcome in O labels")
    ap.add_argument("--max-train", type=int, default=40000)
    ap.add_argument("--max-val", type=int, default=4000)
    ap.add_argument("--max-eval", type=int, default=0, help="0 = every 2025-26 decision in scope")
    ap.add_argument("--max-state", type=int, default=0, help="0 = every in-scope state case")
    ap.add_argument("--probe-n", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if a.stage in ("fetch", "all"):
        fetch(a)
    if a.stage in ("build", "all"):
        build(a)
    if a.stage in ("check", "all"):
        check(a)
    if a.stage == "auth-check":
        auth_check(a)


if __name__ == "__main__":
    main()
