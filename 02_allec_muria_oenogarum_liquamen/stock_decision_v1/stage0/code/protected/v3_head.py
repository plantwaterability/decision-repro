#!/usr/bin/env python3
"""stock_decision_v1 Stage 0: protected identity / overlap registry v3 over ALL model-visible fields (stage -> build -> scan ->
adjudicate; regress). CPU only; no model; labels never printed or stored. Supersedes garas_v2 evalindex_v2 (state-only text keys,
population-level attribution) -- see research/dataset_curation/garas_gemma_vbll_2026-10-12_v2_ERRATA.md E1 / E2.

  stage       fetch every registry_v3 population (registry_v2's 59 + the newer panels), checking every pin; STAGE/stage_manifest.json
  build       index every model-visible FIELD of every protected row into one immutable .npz; every key carries the protected
              (population, FILE) source id and the protected FIELD; coverage table by field x population in the manifest
  scan        every train/val unit of a candidate pool over all its visible fields -> hits_<pool>.jsonl.gz + summary_<pool>.json
              (uid = "<pool>|<member>|<line>|<qid>", identical to unittable_v2); a `confirmation/` split is never opened
  adjudicate  the prespecified template rules (below) -> hits_<pool>.adj.jsonl.gz + adjudication_<pool>.json
  regress     ERRATA E1 / E2 regression cases + synthetic fixtures

Model-visible FIELDS (jev records; the production renderer shows state, the question's instructions and the option texts):
  state               the record state (string or JSON; v2 flatten leaves)
  instructions        a question's instructions
  candidate_document  a question whose qid names a candidate (|doc_N, |passage_N, |cand_N, |tool_N: BRIGHT / ToolRet / BFCL
                      layout, where the document text sits in the instructions)
  option_text         choice option texts (criteria values; the key when the value is empty), score level texts
  noul_defs           a noul question's yes / no definitions (criteria values; shown under R2)
  question_other      any other string leaf of a question dict (not type / criteria / levels / instructions)
  record_other        string leaves of record-level keys that are not ids / labels / meta (context, documents, passages, ...)
  attachment_text     OCR / caption / alt text of attachments; attachments without text are counted as unindexable
  prompt              a fully rendered prompt (fullft devset rows: the whole visible input)
  record_id           record ids, lineage roots, sealed ids / clusters / entity keys
Identity families: v2's (state_exact, state_nfkc, sha1, state_flat, joint_*, question/options/template, rid, sealed, lineage) plus
  text   cnorm text units of EVERY field: the whole field text and JSON leaves (>= 5 words); for texts > 60 words also every
         paragraph / line of >= 12 words. text_long (>= 12 words, STRONG), text_short (5-11, weak), text_boiler (unit in > 100
         protected records, weak)
  doc    8-word-shingle documents of every CONTENT field text of >= 8 words -- state, candidate_document, record_other,
         attachment_text, prompt -- deduplicated by normalised text; every occurrence keeps its source file + field: near_dup
         (Jaccard >= 0.8), eval_in_cand / cand_in_eval (containment >= 0.9, >= 20 shingles), STRONG; shingle3 (>= 3 shared non-stop
         shingles) weak. Shingles in > 100 protected documents are stop shingles.
AMENDMENT A1 (v3.1, 2026-10-14 00:10Z; decided after seeing the RAW v3.0 counts of emo_pool_v1 and jdref_gsm8k only, before any
  adjudicated v3.0 count was read): v3.0 also shingled instructions / option texts / noul definitions. It flagged 1,042,916 of
  1,181,557 curated units STRONG, dominated by near_dup instructions->instructions (4.9M unit x file incidences), option_text and
  noul_defs near-dups and candidate instructions contained in rendered panel prompts: items that share a builder's question
  template but differ in state (the evaluation item). A template with a changed slot is not the same item and the exact-text DF rule
  (T2) cannot see slot-varied templates. v3.1 therefore shingles CONTENT fields only (above) and drops the v3.0 rule that tagged
  choice options of >= 20 words as candidate documents; question-level template fields are matched by exact text units (text_long,
  T1-adjudicated) and v1 joint / question identities. The raw v3.0 summaries are kept as evidence (v3.0_raw/).
Prespecified template rules (stated before any v3 scan result was seen; DF over the candidate pool's own TRAIN records):
  T1 (= v2 adjudicate_text_v2) a text_long hit whose every matched text unit occurs in >= 25 candidate train records -> weak text_template
  T2 a near_dup / eval_in_cand / cand_in_eval hit whose candidate document is an instructions / option_text / noul_defs /
     question_other text occurring (exact normalised text) in >= 25 candidate train records -> weak doc_template
  A unit stays STRONG if any other strong kind fired.
Exact / hash / lexical checks are necessary mechanical exclusions; they do not prove semantic non-overlap.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import io
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

SCHEMA = "stock-decision-protected-index-v3.1"
BUCKET = "YOUR_BUCKET"
S3 = f"s3://{BUCKET}/jev-race/"
SHINGLE_WORDS, DF_MAX, NEAR_J, CONTAIN, CONTAIN_MIN, SH3 = 8, 100, 0.8, 0.9, 20, 3
TEXT_MIN, TEXT_LONG, PARA_SPLIT_WORDS, DF_TPL = 5, 12, 60, 25
WORD = re.compile(r"\w+")
EVAL_PART = re.compile(r"^(val|valid|validation|dev|test|eval|heldout|held_out|holdout|novel|confirm|sealed|bench|probe|extended|"
                       r"overlap|ood|unlabel|teacher_fill|gde1|meragpt)", re.I)          # = fleet/ss_lib.py EVAL_PART
FIELDS = ("state", "instructions", "candidate_document", "option_text", "noul_defs", "question_other", "record_other",
          "attachment_text", "prompt", "record_id")
FID = {f: i for i, f in enumerate(FIELDS)}
CAND_QID = re.compile(r"\|(doc|passage|cand|candidate|tool|psg)_", re.I)
NONVISIBLE_TOP = {"id", "base_id", "source", "license", "meta", "labels", "label_kind", "questions", "state", "split", "targets",
                  "target", "gold", "answer", "answers", "label", "_evaluation", "weights", "weight", "expected", "metadata", "scoring",
                  "benchmark", "family", "record_id", "group_id", "uid", "decision_uid", "qid"}
ATTACH_KEYS = {"images", "image", "attachments", "attachment", "files", "media", "image_paths", "image_path", "image_url"}
OCR_KEYS = {"ocr", "ocr_text", "image_text", "caption", "captions", "alt_text", "transcript", "image_description"}
TEMPLATE_DOC_FIELDS = {FID["instructions"], FID["option_text"], FID["noul_defs"], FID["question_other"]}
CONTENT_FIDS = {FID["state"], FID["candidate_document"], FID["record_other"], FID["attachment_text"], FID["prompt"]}   # A1: shingled fields
STRONG = ("state_exact", "state_nfkc", "sha1", "state_flat", "joint_exact", "joint_norm", "text_long", "rid_nostate", "sealed_id",
          "sealed_cluster", "sealed_entity", "lineage", "near_dup", "eval_in_cand", "cand_in_eval")
WEAK = ("text_short", "text_boiler", "text_template", "doc_template", "question_options", "question_norm", "options_norm", "template",
        "rid_any", "lineage_weak", "shingle3")
ROOT_NUMERIC = re.compile(r"^(train|val|valid|dev|test|eval)?[-_]?\d{0,6}$", re.I)
D = None            # v1 decontam module (identity(), norm()) -- loaded by load_v1()
ENTITY = None
