#!/usr/bin/env python3
"""
build_jev_ext.py -- convert five leakage-checked Hugging Face decision datasets into records in
the jev_data schema (see build_jev_data.py: shared state + typed questions + label distributions),
deduplicated against the jev_real evaluation states.

Sources (output source name -> dataset, license, what is read):
  mghafiri             mghafiri/decision-model-scenarios      MIT           data/{train,validation,test}.jsonl
  vagmi                vagmi/jevlite_dataset                  CC-BY-SA-4.0  synth.train.jsonl + synth.eval.jsonl
  procedural/<config>  tasksource/procedural-typed-decisions  Apache-2.0    <config>/train-*.parquet, 12 configs
  jebadiah             frontier-infra/jebadiah-synth-v2       Apache-2.0    data/synth-<family>.jsonl, 24 files
  kaivoss              kaivoss/system-one-270m-data           Apache-2.0    train.jsonl (pre-rendered prompts)

Every file is fetched with huggingface_hub and parsed directly (no `load_dataset` schema inference,
which trips on jebadiah's mixed nested states). Mapping notes, checked against real rows:
  * mghafiri: choice criteria can be null (-> humanized key); score criteria are either strings or
    dicts with varying fields ({level|label|situation, definition, examples}); some noul questions
    carry {true, false} descriptions. `targets[q].note` and `.difficulty` are never read.
  * noul descriptions (mghafiri, vagmi, jebadiah, kaivoss) are appended to the instructions as
    "Yes means: ... / No means: ...": the teachers labelled with them in view.
  * jebadiah: gold is `target[q]` ({option: prob}); `provenance.questions[q].ordered` choice
    questions keep their stored order, the rest are shuffled. Its states are nested objects,
    serialized as compact JSON like the typed-decisions eval states. Every record carries
    meta.teacher_includes_base_model = true (one teacher is Qwen3.8-27B, our base model).
  * kaivoss: state, question and options are parsed back out of `prompt`; `target` is aligned with
    `letters`; noul options are "yes"/"no" in either order.
  * procedural: questions or answers that come back null are skipped; at most
    --procedural-per-config records per config, sampled from its train split.

Label rules: a question is dropped if its target is missing, all-zero, not the option count, off
by more than --renorm-tol from summing to 1, or has K > 52 options (claude_lmtune's letters);
tiny rounding errors are renormalized. label_kind is "hard" if exactly one option has mass, else
"soft". Choice `criteria` are shuffled with a per-source seeded rng (dict order = display order).

Leakage control (always on; the --eval file is required):
  exact    the state, normalized as analysis/claude_overlap.py `norm` (whitespace collapsed,
           lowercased, surrounding double quotes stripped), equals a fingerprint state -> dropped
  shingle  the state shares >= 3 distinct 8-word shingles with any ONE fingerprint state -> dropped
Fingerprint sets: --eval (tag jev_real, the jev_real test states) and any --fingerprint TAG=PATH
(jsonl rows with a `state` field, e.g. the typed-decisions train rows, which share the
benchmark's generator). Counts are reported per source and per tag.

--strict also drops the domains that overlap typed-decisions' four workflows (customer support /
helpdesk, security incidents / SIEM, invoice processing / procurement, agent traces / tool
selection); see STRICT below, and the run prints every value it excluded.

Output: <out>/{train,val}/<source>.jsonl ('/' -> '__' in file names) and <out>/manifest.json.
Split = hash of the normalized state (build_jev_data.split_of with no test share): ~95/5.
The run ends by reading the corpus back through trainers/cache_features.iter_units.

Usage (on a node with network; HF access is anonymous):
  export HF_HOME=/data/local/hf_jev_ext
  python3 build_jev_ext.py --out /data/shared/jev_ext \
      --eval /data/shared/jev_real_copy/real.jsonl \
      --fingerprint typed_train=/data/shared/jev/sofa_rows/td_train.jsonl
  python3 build_jev_ext.py --strict --out /data/shared/jev_ext_strict --eval ... --fingerprint ...
  python3 build_jev_ext.py --self-test          # no network
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import random
import re
import sys
import tempfile
import time
import types
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "analysis"))

from claude_overlap import norm  # noqa: E402  the eval-overlap normalization, reused verbatim

from build_jev_data import (  # noqa: E402
    Writer,
    choice_q,
    humanize,
    normalize,
    noul_label,
    noul_q,
    option_keys,
    record,
    score_q,
    shuffled,
    split_of,
    validate,
)

MAX_K = 52            # claude_lmtune letters: A-Z a-z
VAL_PCT = 5
SHINGLE_WORDS = 8
SHINGLE_MIN_SHARED = 3
EVAL_TAG = "jev_real"

DATASETS = {
    "mghafiri": ("mghafiri/decision-model-scenarios", "MIT"),
    "vagmi": ("vagmi/jevlite_dataset", "CC-BY-SA-4.0 (share-alike: derived records inherit it)"),
    "procedural": ("tasksource/procedural-typed-decisions", "Apache-2.0"),
    "jebadiah": ("frontier-infra/jebadiah-synth-v2", "Apache-2.0"),
    "kaivoss": ("kaivoss/system-one-270m-data", "Apache-2.0"),
}
PROCEDURAL_CONFIGS = [
    "arithmetic", "entity_belief_tracking", "event_state_reconstruction", "evidence_sufficiency",
    "multi_view_adjudication", "needle_retrieval", "partial_observation_calibration",
    "policy_applicability", "policy_under_uncertainty", "record_aggregation",
    "state_perturbation", "table_lookup",
]
JEBADIAH_FAMILIES = [
    "agent_tool_selection", "claim_evidence", "code_review_comment", "content_policy", "cs_chat",
    "email_routing", "form_validation", "helpdesk_ticket", "incident_alert", "invoice_receipt",
    "jde_added_requirements", "jde_coverage", "jde_echo", "jde_parrot", "jde_parts",
    "log_severity", "meeting_actions", "pr_triage", "product_review", "sales_lead",
    "scheduling_dispatch", "summary_faithfulness", "translation_adequacy", "voice_intent",
]

# --strict: meta field checked per dataset, the explicit values excluded, and a pattern that also
# catches support / security / invoice / agent-trace values the explicit lists did not anticipate.
# Values were read off the actual rows. Beyond the survey's lists: mghafiri it_helpdesk,
# telecom_support and agent_tool_routing; jebadiah's five jde_* families (every state is a
# helpdesk ticket with a technician reply); procedural multi_view_adjudication (support-ticket
# intent/urgency adjudication).
STRICT = {
    "mghafiri": ("domain", {
        "customer_support", "it_helpdesk", "telecom_support", "cybersecurity_soc",
        "procurement_vendor_risk", "agent_tool_routing"}),
    "vagmi": ("task", {
        "synth:a customer support ticket", "synth:a security alert from a SIEM",
        "synth:a vendor invoice with line items",
        "synth:a transcript of an AI agent completing a task"}),
    "jebadiah": ("family", {
        "helpdesk_ticket", "cs_chat", "invoice_receipt", "incident_alert", "agent_tool_selection",
        "jde_added_requirements", "jde_coverage", "jde_echo", "jde_parrot", "jde_parts"}),
    "kaivoss": ("domain", {"security alert", "customer support email", "IT helpdesk ticket"}),
    "procedural": ("config", {"multi_view_adjudication"}),
}
STRICT_PATTERN = re.compile(r"support|help ?desk|siem|security|cyber|invoice|procure|vendor|agent|tool", re.IGNORECASE)


def family_of(source):
    return source.split("/")[0]


def strict_hit(source, meta):
    """(value, how) if --strict drops this record, else (value, None)."""
    field, values = STRICT[family_of(source)]
    v = str(meta.get(field, ""))
    if v in values:
        return v, "explicit"
    if STRICT_PATTERN.search(v):
        return v, "pattern"
    return v, None


# --------------------------------------------------------------------------- small helpers
def as_obj(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except ValueError:
            return x
    return x


def text_of(v):
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    return json.dumps(v, ensure_ascii=False)


def state_text(x):
    """Strings verbatim; nested objects as compact JSON (the typed-decisions eval states' form)."""
    if x is None:
        return ""
    return x.strip() if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def instructions_text(instr):
    """mghafiri sometimes ships {question, <rule excerpt / definitions ...>}: question first."""
    if isinstance(instr, dict):
        head = text_of(instr.get("question"))
        rest = [f"{humanize(k)}: {text_of(v)}" for k, v in instr.items()
                if k != "question" and text_of(v)]
        return "\n".join(([head] if head else []) + rest).strip()
    return text_of(instr)


def noul_instructions(instr, criteria):
    if not isinstance(criteria, dict):
        return instr
    t = text_of(criteria.get("true", criteria.get("yes")))
    f = text_of(criteria.get("false", criteria.get("no")))
    extra = ([f"Yes means: {t}"] if t else []) + ([f"No means: {f}"] if f else [])
    return "\n".join([instr] + extra)


LEVEL_NAME_KEYS = ("level", "label", "name", "situation", "title", "value")


def level_text(c):
    """A score level as display text: 'Name: definition (e.g. ...)' for dict levels."""
    if not isinstance(c, dict):
        return text_of(c)
    name_key = next((k for k in LEVEL_NAME_KEYS if text_of(c.get(k))), None)
    name = text_of(c[name_key]) if name_key else ""
    body, examples = [], []
    for k, v in c.items():
        if k == name_key:
            continue
        t = "; ".join(text_of(x) for x in v) if isinstance(v, list) else text_of(v)
        if not t:
            continue
        if k == "examples":
            examples.append(t)
        elif k in ("definition", "description", "meaning"):
            body.append(t)
        else:
            body.append(f"{humanize(k)}: {t}")
    out = name + (": " if name and body else "") + " ".join(body)
    if examples:
        out += f" (e.g. {'; '.join(examples)})"
    return out.strip()


def unique_levels(levels):
    if all(levels) and len(set(levels)) == len(levels):
        return levels
    return [f"{i} - {lv}" if lv else str(i) for i, lv in enumerate(levels)]


def kind_of(lab):
    return "hard" if sum(1 for v in lab.values() if v > 0) == 1 else "soft"


class Labeler:
    """Target cleaning shared by every converter; counts why questions are dropped."""

    def __init__(self, tol):
        self.tol, self.drops = tol, Counter()

    def dist(self, keys, vals):
        if len(keys) < 2:
            self.drops["k<2"] += 1
            return None
        if len(keys) > MAX_K:
            self.drops["k>52"] += 1
            return None
        if len(set(keys)) != len(keys):
            self.drops["duplicate_option_keys"] += 1
            return None
        if vals is None or len(vals) != len(keys):
            self.drops["target_count_mismatch"] += 1
            return None
        if any(v is None for v in vals):
            self.drops["target_missing"] += 1
            return None
        try:
            vals = [float(v) for v in vals]
        except (TypeError, ValueError):
            self.drops["target_not_numeric"] += 1
            return None
        if any(not math.isfinite(v) or v < 0 for v in vals):
            self.drops["target_out_of_range"] += 1
            return None
        s = sum(vals)
        if s <= 0:
            self.drops["target_all_zero"] += 1
            return None
        if abs(s - 1.0) > self.tol:
            self.drops["target_bad_sum"] += 1
            return None
        if abs(s - 1.0) > 1e-9:
            self.drops["renormalized(kept)"] += 1
        return normalize(dict(zip(keys, vals, strict=True)))

    def noul(self, p):
        if p is None:
            self.drops["target_missing"] += 1
            return None
        try:
            p = float(p)
        except (TypeError, ValueError):
            self.drops["target_not_numeric"] += 1
            return None
        if not math.isfinite(p) or p < -self.tol or p > 1 + self.tol:
            self.drops["target_out_of_range"] += 1
            return None
        return noul_label(min(max(p, 0.0), 1.0))

    def noul_pair(self, p_true, p_false):
        d = self.dist(["yes", "no"], [p_true, p_false])
        return None if d is None else noul_label(d["yes"])


def add_choice(qs, labs, qid, instr, criteria, lab, rng, keep_order=False):
    crit = criteria if keep_order else shuffled(criteria, rng)
    qs[qid] = choice_q(instr, crit)
    labs[qid] = {k: lab[k] for k in crit}


def make_record(source, state, qs, labs, meta):
    kinds = {q: kind_of(labs[q]) for q in qs}
    rec = record(source, state, qs, labs, kinds, meta)
    repo, lic = DATASETS[family_of(source)]
    rec["license"] = lic
    rec["meta"] = {"dataset": repo, **meta}
    return rec


# --------------------------------------------------------------------------- converters
def convert_mghafiri(rows, rng, lab):
    """One row per scenario: state (JSON string), questions and targets (JSON strings)."""
    for row in rows:
        state = state_text(row.get("state"))
        Q, T = as_obj(row.get("questions")), as_obj(row.get("targets"))
        if not state or not isinstance(Q, dict) or not isinstance(T, dict):
            lab.drops["bad_row"] += 1
            continue
        qs, labs = {}, {}
        for qid, q in Q.items():
            t = T.get(qid)
            if not isinstance(q, dict) or not isinstance(t, dict):
                lab.drops["target_missing"] += 1
                continue
            instr = instructions_text(q.get("instructions"))
            if not instr:
                lab.drops["empty_instructions"] += 1
                continue
            typ, crit = q.get("type"), q.get("criteria")
            if typ == "choice" and isinstance(crit, dict):
                keys = [str(k) for k in crit]
                probs = {str(k): v for k, v in (t.get("probabilities") or {}).items()}
                if set(probs) != set(keys):
                    lab.drops["target_count_mismatch"] += 1
                    continue
                d = lab.dist(keys, [probs[k] for k in keys])
                if d:
                    criteria = {str(k): text_of(v) or humanize(k) for k, v in crit.items()}
                    add_choice(qs, labs, qid, instr, criteria, d, rng)
            elif typ == "score" and isinstance(crit, list):
                levels = unique_levels([level_text(c) for c in crit])
                probs = {str(k): v for k, v in (t.get("probabilities") or {}).items()}
                if set(probs) != {str(i) for i in range(len(levels))}:
                    lab.drops["target_count_mismatch"] += 1
                    continue
                d = lab.dist(levels, [probs[str(i)] for i in range(len(levels))])
                if d:
                    qs[qid], labs[qid] = score_q(instr, levels), d
            elif typ == "noul":
                d = lab.noul(t.get("noul"))
                if d:
                    qs[qid], labs[qid] = noul_q(noul_instructions(instr, crit)), d
            else:
                lab.drops["unsupported_question"] += 1
        if qs:
            yield make_record("mghafiri", state, qs, labs, {
                "orig_id": row.get("id"), "orig_split": row.get("_split"),
                "domain": row.get("domain"), "pattern": row.get("pattern")})


def convert_vagmi(rows, rng, lab):
    """One row per question; rows sharing a state become one record."""
    groups = defaultdict(list)
    for row in rows:
        s = state_text(row.get("state"))
        if s:
            groups[s].append(row)
        else:
            lab.drops["bad_row"] += 1
    for state, grp in groups.items():
        qs, labs, keys_seen = {}, {}, []
        for i, row in enumerate(grp):
            k = str(row.get("_key") or "")
            qid = f"q{k.rsplit('-', 1)[-1]}" if "-" in k else f"q{i}"
            if qid in qs:
                qid = f"{qid}_{i}"
            keys_seen.append(k)
            instr = text_of(row.get("question"))
            opts = [str(o) for o in (row.get("options") or [])]
            label = row.get("label")
            typ, crit = row.get("type"), row.get("criteria")
            if not instr:
                lab.drops["empty_instructions"] += 1
                continue
            if not isinstance(label, list) or len(label) != len(opts):
                lab.drops["target_count_mismatch"] += 1
                continue
            if typ == "choice":
                d = lab.dist(opts, label)
                if d:
                    crit = crit if isinstance(crit, dict) else {}
                    criteria = {o: text_of(crit.get(o)) or humanize(o) for o in opts}
                    add_choice(qs, labs, qid, instr, criteria, d, rng)
            elif typ == "score":
                levels = unique_levels(opts)
                d = lab.dist(levels, label)
                if d:
                    qs[qid], labs[qid] = score_q(instr, levels), d
            elif typ == "noul":
                low = [o.lower() for o in opts]
                if sorted(low) != ["false", "true"]:
                    lab.drops["noul_option_keys"] += 1
                    continue
                d = lab.noul_pair(label[low.index("true")], label[low.index("false")])
                if d:
                    qs[qid], labs[qid] = noul_q(noul_instructions(instr, crit)), d
            else:
                lab.drops["unsupported_question"] += 1
        if qs:
            yield make_record("vagmi", state, qs, labs, {
                "task": grp[0].get("task"), "orig_source": grp[0].get("source"),
                "orig_split": grp[0].get("_split"), "orig_keys": keys_seen})


def convert_procedural(rows, config, rng, lab):
    """questions / answers are JSON strings keyed by question id; unasked questions are null."""
    source = f"procedural/{config}"
    for row in rows:
        state = state_text(row.get("state"))
        Q, A = as_obj(row.get("questions")), as_obj(row.get("answers"))
        if not state or not isinstance(Q, dict) or not isinstance(A, dict):
            lab.drops["bad_row"] += 1
            continue
        qs, labs = {}, {}
        for qid, q in Q.items():
            a = A.get(qid)
            if not isinstance(q, dict) or not isinstance(a, dict):
                lab.drops["null_question(skipped)"] += 1
                continue
            instr = text_of(q.get("instructions"))
            typ, crit = q.get("type"), q.get("criteria")
            if not instr:
                lab.drops["empty_instructions"] += 1
                continue
            if typ == "choice" and isinstance(crit, dict):
                probs = a.get("probabilities")
                if probs is None:
                    lab.drops["null_question(skipped)"] += 1
                    continue
                keys = [str(k) for k in crit]
                probs = {str(k): v for k, v in probs.items()}
                if set(probs) != set(keys):
                    lab.drops["target_count_mismatch"] += 1
                    continue
                d = lab.dist(keys, [probs[k] for k in keys])
                if d:
                    criteria = {str(k): text_of(v) or humanize(k) for k, v in crit.items()}
                    add_choice(qs, labs, qid, instr, criteria, d, rng)
            elif typ == "score" and isinstance(crit, list):
                probs, legend = a.get("probabilities"), a.get("legend") or {}
                if probs is None:
                    lab.drops["null_question(skipped)"] += 1
                    continue
                probs = {str(k): v for k, v in probs.items()}
                n = len(crit)
                if set(probs) != {str(i) for i in range(n)}:
                    lab.drops["target_count_mismatch"] += 1
                    continue
                levels = unique_levels([text_of(legend.get(str(i))) or level_text(c)
                                        for i, c in enumerate(crit)])
                d = lab.dist(levels, [probs[str(i)] for i in range(n)])
                if d:
                    qs[qid], labs[qid] = score_q(instr, levels), d
            elif typ == "noul":
                if a.get("noul") is None:
                    lab.drops["null_question(skipped)"] += 1
                    continue
                d = lab.noul(a.get("noul"))
                if d:
                    qs[qid], labs[qid] = noul_q(noul_instructions(instr, crit)), d
            else:
                lab.drops["unsupported_question"] += 1
        if qs:
            yield make_record(source, state, qs, labs, {
                "config": config, "orig_id": row.get("id"), "level": row.get("level")})


def convert_jebadiah(rows, rng, lab):
    """Nested state object; gold in top-level target[q] as {option: prob}."""
    for row in rows:
        state = state_text(row.get("state"))
        Q, T = as_obj(row.get("questions")), as_obj(row.get("target"))
        prov = as_obj(row.get("provenance")) or {}
        pq = (prov.get("questions") or {}) if isinstance(prov, dict) else {}
        if not state or not isinstance(Q, dict) or not isinstance(T, dict):
            lab.drops["bad_row"] += 1
            continue
        qs, labs = {}, {}
        for qid, q in Q.items():
            t = T.get(qid)
            if not isinstance(q, dict) or not isinstance(t, dict):
                lab.drops["target_missing"] += 1
                continue
            instr = instructions_text(q.get("instructions"))
            typ, crit = q.get("type"), q.get("criteria")
            if not instr:
                lab.drops["empty_instructions"] += 1
                continue
            t = {str(k): v for k, v in t.items()}
            if typ == "choice" and isinstance(crit, dict):
                keys = [str(k) for k in crit]
                if set(t) != set(keys):
                    lab.drops["target_count_mismatch"] += 1
                    continue
                d = lab.dist(keys, [t[k] for k in keys])
                if d:
                    criteria = {str(k): text_of(v) or humanize(k) for k, v in crit.items()}
                    ordered = bool((pq.get(qid) or {}).get("ordered"))
                    add_choice(qs, labs, qid, instr, criteria, d, rng, keep_order=ordered)
            elif typ == "noul":
                if set(t) != {"true", "false"}:
                    lab.drops["noul_option_keys"] += 1
                    continue
                d = lab.noul_pair(t["true"], t["false"])
                if d:
                    qs[qid], labs[qid] = noul_q(noul_instructions(instr, crit)), d
            elif typ == "score" and isinstance(crit, list):
                raw = [level_text(c) for c in crit]
                levels = unique_levels(raw)
                vals = [t.get(str(i), t.get(r)) for i, r in enumerate(raw)]
                if len(t) != len(levels):
                    lab.drops["target_count_mismatch"] += 1
                    continue
                d = lab.dist(levels, vals)
                if d:
                    qs[qid], labs[qid] = score_q(instr, levels), d
            else:
                lab.drops["unsupported_question"] += 1
        if qs:
            judge = prov.get("judge", {}) if isinstance(prov, dict) else {}
            family = row.get("subset") or str(row.get("family", "")).split(":")[0] or row.get("_file_family")
            yield make_record("jebadiah", state, qs, labs, {
                "family": family, "orig_id": row.get("id"),
                "teacher_includes_base_model": True,
                "teachers": judge.get("answered"), "gold": judge.get("gold")})


KAIVOSS_STATE = re.compile(r"<state>\n?(.*?)\n?</state>", re.DOTALL)
KAIVOSS_QUESTION = re.compile(r"Question:[ \t]*(.*?)\n[ \t]*Options:[ \t]*\n(.*?)(?:\n[ \t]*\n[ \t]*Answer|\n[ \t]*Answer:|\Z)", re.DOTALL)
KAIVOSS_OPTION = re.compile(r"^([A-Z])\. (.+?)(?: - (.*))?$")


def parse_kaivoss(prompt):
    """(state, question, [(letter, key, desc)]) from a pre-rendered prompt, or None."""
    m = KAIVOSS_STATE.search(prompt or "")
    if not m:
        return None
    qm = KAIVOSS_QUESTION.search(prompt[m.end():])
    if not qm:
        return None
    opts = []
    for line in qm.group(2).split("\n"):
        mm = KAIVOSS_OPTION.match(line.strip())
        if mm:
            opts.append([mm.group(1), mm.group(2).strip(), (mm.group(3) or "").strip()])
        elif opts and line.strip():
            opts[-1][2] = (opts[-1][2] + " " + line.strip()).strip()  # wrapped description
    return m.group(1).strip(), qm.group(1).strip(), [tuple(o) for o in opts]


def convert_kaivoss(rows, rng, lab):
    """One row per question with a rendered prompt; rows sharing state_id become one record."""
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("state_id") or row.get("prompt", "")[:200]].append(row)
    for sid, grp in groups.items():
        qs, labs, state = {}, {}, None
        for i, row in enumerate(grp):
            parsed = parse_kaivoss(row.get("prompt"))
            if not parsed:
                lab.drops["unparseable_prompt"] += 1
                continue
            st, instr, opts = parsed
            if state is None:
                state = st
            elif st != state:
                lab.drops["state_differs_within_state_id"] += 1
                continue
            letters = [o[0] for o in opts]
            target = row.get("target")
            if letters != list(row.get("letters") or []) or not isinstance(target, list):
                lab.drops["letter_mismatch"] += 1
                continue
            if len(target) != len(opts):
                lab.drops["target_count_mismatch"] += 1
                continue
            qid, typ = f"q{i}", row.get("qtype")
            keys = [o[1] for o in opts]
            if typ == "choice":
                d = lab.dist(keys, target)
                if d:
                    criteria = {k: desc or humanize(k) for _, k, desc in opts}
                    add_choice(qs, labs, qid, instr, criteria, d, rng)
            elif typ == "score":
                levels = unique_levels([f"{k} - {desc}" if desc else k for _, k, desc in opts])
                d = lab.dist(levels, target)
                if d:
                    qs[qid], labs[qid] = score_q(instr, levels), d
            elif typ == "noul":
                low = [k.lower() for k in keys]
                alias = {"true": "yes", "false": "no"}
                low = [alias.get(k, k) for k in low]
                if sorted(low) != ["no", "yes"]:
                    lab.drops["noul_option_keys"] += 1
                    continue
                iy, in_ = low.index("yes"), low.index("no")
                d = lab.noul_pair(target[iy], target[in_])
                if d:
                    desc = {"true": opts[iy][2], "false": opts[in_][2]}
                    qs[qid], labs[qid] = noul_q(noul_instructions(instr, desc)), d
            else:
                lab.drops["unsupported_question"] += 1
        if qs and state:
            yield make_record("kaivoss", state, qs, labs, {
                "domain": grp[0].get("domain"), "state_id": sid,
                "ambiguity": [r.get("ambiguity") for r in grp]})


# --------------------------------------------------------------------------- loading
def with_retry(fn, what, tries=7):
    """Anonymous HF access: back off on 429 / transient errors instead of hammering."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            fatal = type(e).__name__ in ("OfflineModeIsEnabled", "RepositoryNotFoundError",
                                         "EntryNotFoundError", "GatedRepoError")
            if fatal or status in (401, 403, 404) or attempt == tries - 1:
                raise
            wait = 30 * 2 ** attempt if status == 429 else 5 * (attempt + 1)
            print(f"  [hf] {what}: {type(e).__name__} (HTTP {status}); retry in {wait}s", flush=True)
            time.sleep(wait)


def hf_file(repo, filename, token=None):
    from huggingface_hub import hf_hub_download
    return with_retry(lambda: hf_hub_download(repo, filename, repo_type="dataset", token=token),
                      f"{repo}/{filename}")


def hf_list(repo, token=None):
    from huggingface_hub import HfApi
    return with_retry(lambda: HfApi().list_repo_files(repo, repo_type="dataset", token=token),
                      f"list {repo}")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as e:
                    raise ValueError(f"{path}:{n}: {e}") from e


def load_mghafiri(a, rng):
    repo = DATASETS["mghafiri"][0]
    for split in ("train", "validation", "test"):
        for row in read_jsonl(hf_file(repo, f"data/{split}.jsonl", a.hf_token)):
            row["_split"] = split
            yield row


def load_vagmi(a, rng):
    repo = DATASETS["vagmi"][0]
    for fn, split in (("synth.train.jsonl", "train"), ("synth.eval.jsonl", "validation")):
        for row in read_jsonl(hf_file(repo, fn, a.hf_token)):
            row["_split"] = split
            yield row


def load_jebadiah(a, rng):
    repo = DATASETS["jebadiah"][0]
    try:
        files = sorted(f for f in hf_list(repo, a.hf_token) if re.fullmatch(r"data/synth-[\w-]+\.jsonl", f))
    except Exception as e:  # noqa: BLE001
        print(f"  [jebadiah] listing failed ({type(e).__name__}); using the known 24 families")
        files = []
    files = files or [f"data/synth-{f}.jsonl" for f in JEBADIAH_FAMILIES]
    for fn in files:
        fam = fn[len("data/synth-"):-len(".jsonl")]
        for row in read_jsonl(hf_file(repo, fn, a.hf_token)):
            row["_file_family"] = fam
            yield row


def load_kaivoss(a, rng):
    return read_jsonl(hf_file(DATASETS["kaivoss"][0], "train.jsonl", a.hf_token))


def load_procedural(a, rng, config):
    import pyarrow.parquet as pq
    repo = DATASETS["procedural"][0]
    try:
        files = sorted(f for f in hf_list(repo, a.hf_token)
                       if f.startswith(f"{config}/train-") and f.endswith(".parquet"))
    except Exception:  # noqa: BLE001
        files = []
    files = files or [f"{config}/train-00000-of-00001.parquet"]
    tables = [pq.read_table(hf_file(repo, f, a.hf_token)) for f in files]
    rows = [r for t in tables for r in t.to_pylist()]
    k = min(len(rows), max(1, int(a.procedural_per_config * a.oversample)))
    return [rows[i] for i in sorted(rng.sample(range(len(rows)), k))]


# --------------------------------------------------------------------------- fingerprints
WORD = re.compile(r"\w+")


def shingles(s, n=SHINGLE_WORDS):
    w = WORD.findall(norm(s))
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


class Fingerprints:
    """Exact normalized-state matches and 8-word shingle overlap against fingerprint states."""

    def __init__(self):
        self.exact = defaultdict(set)     # norm(state) -> tags
        self.index = defaultdict(list)    # hash(shingle) -> [state index]
        self.tags, self.texts, self.sizes, self.paths = [], [], Counter(), defaultdict(list)

    def add_file(self, tag, path):
        seen = set()
        for row in read_jsonl(path):
            s = row.get("state")
            if s is None:
                continue
            s = state_text(s)
            ns = norm(s)
            if not ns or ns in seen:
                continue
            seen.add(ns)
            self.add(tag, s)
        self.paths[tag].append(str(path))
        print(f"fingerprints: {len(seen):6d} states from {path} (tag {tag})")

    def add(self, tag, s):
        ns = norm(s)
        if tag in self.exact[ns]:
            return
        self.exact[ns].add(tag)
        sid = len(self.texts)
        self.tags.append(tag)
        self.texts.append(s)
        self.sizes[tag] += 1
        for sh in shingles(s):
            self.index[hash(sh)].append(sid)

    def check(self, s):
        """None, or (kind, tag, n_shared, fingerprint_index). The eval tag wins ties."""
        tags = self.exact.get(norm(s))
        if tags:
            return ("exact", EVAL_TAG if EVAL_TAG in tags else min(tags), None, None)
        counts = Counter()
        for sh in shingles(s):
            for sid in self.index.get(hash(sh), ()):
                counts[sid] += 1
        hits = [(sid, c) for sid, c in counts.items() if c >= SHINGLE_MIN_SHARED]
        if not hits:
            return None
        hits.sort(key=lambda x: (self.tags[x[0]] != EVAL_TAG, -x[1]))
        sid, c = hits[0]
        return ("shingle", self.tags[sid], c, sid)

    def shared(self, s, sid, k=2):
        return sorted(shingles(s) & shingles(self.texts[sid]))[:k]


# --------------------------------------------------------------------------- pipeline
def new_stats(source, lab):
    """Per-source manifest entry. `values` counts the --strict field (domain / task / family /
    config) over candidates that survived dedup, before any strict exclusion; `written`,
    `units`, `label_kinds` and `qtypes` describe what was written."""
    repo, lic = DATASETS[family_of(source)]
    return {"dataset": repo, "license": lic, "candidates": 0, "written": Counter(),
            "units": Counter(), "label_kinds": Counter(), "qtypes": Counter(),
            "drops": Counter(), "dedup_exact": Counter(), "dedup_shingle": Counter(),
            "strict_excluded": Counter(), "values": Counter(), "question_drops": lab.drops,
            "shingle_examples": [], "errors": []}


def process(source, cands, cap, fps, strict, writer, rng, st):
    """validate -> within-source dedup -> fingerprint dedup -> strict -> cap -> write.

    Dedup runs before strict, so its counts are the same in the full and the strict build; every
    candidate is checked even after the cap is reached, so the counts cover all candidates."""
    st["candidates"] = len(cands)
    rng.shuffle(cands)
    seen_ids, seen_states, kept = set(), set(), 0
    field = STRICT[family_of(source)][0]
    for rec in cands:
        try:
            validate(rec)
        except AssertionError as e:
            st["drops"]["invalid"] += 1
            if len(st["errors"]) < 5:
                st["errors"].append(f"invalid: {e}")
            continue
        ns = norm(rec["state"])
        if rec["id"] in seen_ids or ns in seen_states:
            st["drops"]["duplicate_state"] += 1
            continue
        seen_ids.add(rec["id"])
        seen_states.add(ns)
        hit = fps.check(rec["state"]) if fps else None
        if hit:
            kind, tag, n_shared, sid = hit
            st["dedup_" + kind][tag] += 1
            if kind == "shingle" and len(st["shingle_examples"]) < 3:
                st["shingle_examples"].append({"record": rec["id"], "tag": tag, "shared": n_shared,
                                               "sample": fps.shared(rec["state"], sid)})
            continue
        value, how = strict_hit(source, rec["meta"])
        st["values"][f"{field}={value}"] += 1
        if strict and how:
            st["strict_excluded"][f"{field}={value}" + (" (pattern)" if how == "pattern" else "")] += 1
            continue
        if kept >= cap:
            st["drops"]["over_cap"] += 1
            continue
        split = split_of(rec["state"], val_pct=VAL_PCT, test_pct=0)
        writer.write(split, source, rec)
        kept += 1
        st["written"][split] += 1
        st["units"][split] += len(rec["questions"])
        st["label_kinds"].update(rec["label_kind"].values())
        st["qtypes"].update(q["type"] for q in rec["questions"].values())
    return st


def plain(st):
    return {k: (dict(v) if isinstance(v, Counter) else v) for k, v in st.items()}


def report_source(source, st, strict):
    w = st["written"]
    print(f"[{source}] candidates {st['candidates']}, written train {w.get('train', 0)} / val {w.get('val', 0)}; "
          f"dedup exact {dict(st['dedup_exact'])} shingle {dict(st['dedup_shingle'])}; "
          f"drops {dict(st['drops'])}", flush=True)
    if st["question_drops"]:
        print(f"[{source}]   question-level: {dict(st['question_drops'])}")
    if strict and st["strict_excluded"]:
        print(f"[{source}]   strict excluded: {dict(st['strict_excluded'])}")
    for ex in st["shingle_examples"]:
        print(f"[{source}]   shingle hit {ex}")


def build(a):
    out = Path(a.out)
    stale = [p for sp in ("train", "val", "test") for p in (out / sp).glob("*.jsonl")]
    if stale or (out / "manifest.json").exists():
        sys.exit(f"{out} already holds a corpus ({len(stale)} jsonl files); write to a new --out "
                 "(stale files from another run would be read by iter_units as part of this one)")
    fps = Fingerprints()
    for p in a.eval:
        fps.add_file(EVAL_TAG, p)
    for spec in a.fingerprint:
        tag, _, p = spec.partition("=")
        if not p:
            sys.exit(f"--fingerprint wants TAG=PATH, got {spec!r}")
        if not Path(p).exists():
            print(f"fingerprints: {p} not found, skipping tag {tag}")
            continue
        fps.add_file(tag, p)
    if not fps.sizes.get(EVAL_TAG):
        sys.exit("no jev_real eval states loaded: refusing to build without the eval dedup")

    writer = Writer(out)
    manifest = {
        "builder": "build_jev_ext.py", "seed": a.seed, "strict": a.strict,
        "splits": {"train": 100 - VAL_PCT, "val": VAL_PCT, "by": "sha1 of the normalized state"},
        "max_per_source": a.max_per_source, "procedural_per_config": a.procedural_per_config,
        "max_options": MAX_K, "renorm_tol": a.renorm_tol,
        "dedup": {"normalization": "analysis/claude_overlap.py norm", "shingle_words": SHINGLE_WORDS,
                  "shingle_min_shared_with_one_state": SHINGLE_MIN_SHARED,
                  "fingerprints": {t: {"states": n, "paths": fps.paths[t]} for t, n in fps.sizes.items()}},
        "strict_rules": {k: {"field": f, "values": sorted(v), "pattern": STRICT_PATTERN.pattern}
                         for k, (f, v) in STRICT.items()} if a.strict else None,
        "sources": {},
    }
    jobs = []
    for name in a.sources:
        if name == "procedural":
            jobs += [(f"procedural/{c}", c) for c in a.procedural_configs]
        else:
            jobs.append((name, None))
    loaders = {"mghafiri": (load_mghafiri, convert_mghafiri), "vagmi": (load_vagmi, convert_vagmi),
               "jebadiah": (load_jebadiah, convert_jebadiah), "kaivoss": (load_kaivoss, convert_kaivoss)}
    for source, config in jobs:
        rng, lab = random.Random(f"{a.seed}:{source}"), Labeler(a.renorm_tol)
        st = new_stats(source, lab)
        print(f"[{source}] loading...", flush=True)
        try:
            if config:
                rows = load_procedural(a, rng, config)
                cands = list(convert_procedural(rows, config, rng, lab))
                cap = a.procedural_per_config
            else:
                load, conv = loaders[source]
                cands = list(conv(load(a, rng), rng, lab))
                cap = a.max_per_source
            process(source, cands, min(cap, a.max_per_source), fps, a.strict, writer, rng, st)
        except Exception as e:  # noqa: BLE001
            st["errors"].append(f"FAILED {type(e).__name__}: {e}")
            print(f"[{source}] FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
        report_source(source, st, a.strict)
        manifest["sources"][source] = plain(st)
    writer.close()
    out.mkdir(parents=True, exist_ok=True)
    manifest["readback"] = final_check(out)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print_summary(manifest)
    print(f"manifest written to {out / 'manifest.json'}")


def print_summary(m):
    tot = Counter()
    print("\ndedup (dropped records): exact / shingle, per fingerprint tag")
    for s, st in m["sources"].items():
        print(f"  {s:40s} exact {st['dedup_exact']}  shingle {st['dedup_shingle']}")
        tot.update({f"exact:{k}": v for k, v in st["dedup_exact"].items()})
        tot.update({f"shingle:{k}": v for k, v in st["dedup_shingle"].items()})
    print(f"  total: {dict(tot)}")
    if m["strict"]:
        print("\nstrict: excluded values (records dropped)")
        for s, st in m["sources"].items():
            for v, n in sorted(st["strict_excluded"].items()):
                print(f"  {s:40s} {v:60s} {n}")


# --------------------------------------------------------------------------- final check
def load_cache_features():
    """trainers/cache_features.py, as every downstream script imports it. It imports torch and
    transformers at module level but iter_units/question_options use neither; stub them only if
    they are not installed, so the check also runs on a bare CPU box."""
    for mod in ("torch", "transformers"):
        try:
            importlib.import_module(mod)
        except ImportError:
            stub = types.ModuleType(mod)
            stub.AutoTokenizer = None
            sys.modules[mod] = stub
            print(f"(final check: {mod} not installed, stubbed for the import)")
    spec = importlib.util.spec_from_file_location("cache_features", HERE / "trainers" / "cache_features.py")
    cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf)
    return cf


def final_check(out):
    """Re-validate every written record and read the corpus back through iter_units."""
    cf = load_cache_features()
    recs = Counter()
    for split in ("train", "val"):
        for p in sorted(Path(out, split).glob("*.jsonl")):
            for rec in read_jsonl(p):
                validate(rec)
                for qid, q in rec["questions"].items():
                    keys, texts, target = cf.question_options(q, rec["labels"][qid])
                    assert keys == option_keys(q) and len(texts) == len(target) <= MAX_K, (rec["id"], qid)
                recs[(rec["source"], split)] += 1
    units, kinds, qtypes = Counter(), Counter(), Counter()
    for u in cf.iter_units(str(out), ["train", "val"], set(), None):
        assert abs(sum(u["target"]) - 1.0) < 1e-6 and 2 <= len(u["target"]) <= MAX_K, u["record_id"]
        units[(u["source"], u["split"])] += 1
        kinds[(u["source"], u["label_kind"])] += 1
        qtypes[(u["source"], u["qtype"])] += 1
    sources = sorted({s for s, _ in recs} | {s for s, _ in units})
    table = {}
    hdr = (f"{'source':34s} {'rec_train':>9s} {'rec_val':>7s} {'units_tr':>8s} {'units_val':>9s} "
           f"{'soft':>6s} {'hard':>6s} {'choice':>6s} {'score':>6s} {'noul':>6s}")
    print("\nread back through cache_features.iter_units (units = decision units):")
    print(hdr)
    for s in sources:
        row = {"records": {"train": recs[(s, "train")], "val": recs[(s, "val")]},
               "units": {"train": units[(s, "train")], "val": units[(s, "val")]},
               "label_kind": {k: kinds[(s, k)] for k in ("soft", "hard") if kinds[(s, k)]},
               "qtype": {k: qtypes[(s, k)] for k in ("choice", "score", "noul") if qtypes[(s, k)]}}
        table[s] = row
        print(f"{s:34s} {row['records']['train']:9d} {row['records']['val']:7d} {row['units']['train']:8d} "
              f"{row['units']['val']:9d} {kinds[(s, 'soft')]:6d} {kinds[(s, 'hard')]:6d} "
              f"{qtypes[(s, 'choice')]:6d} {qtypes[(s, 'score')]:6d} {qtypes[(s, 'noul')]:6d}")
    def tot(c, keys):
        return {k: sum(v for (_, kk), v in c.items() if kk == k) for k in keys}

    total, tu = tot(recs, ("train", "val")), tot(units, ("train", "val"))
    tk, tq = tot(kinds, ("soft", "hard")), tot(qtypes, ("choice", "score", "noul"))
    print(f"{'TOTAL':34s} {total['train']:9d} {total['val']:7d} {tu['train']:8d} {tu['val']:9d} "
          f"{tk['soft']:6d} {tk['hard']:6d} {tq['choice']:6d} {tq['score']:6d} {tq['noul']:6d}")
    return {"per_source": table, "total": {"records": total, "units": tu, "label_kind": tk, "qtype": tq}}


# --------------------------------------------------------------------------- self-test
def self_test():
    """Every converter on hand-made rows shaped like the real ones, then the whole pipeline
    (fingerprint dedup, strict, write, iter_units read-back) on a temp dir. No network."""
    rng = random.Random(0)
    lab = Labeler(0.01)
    mg_state = json.dumps({"application": {"permit_type_selected": "MA", "description": "Replace the rotted deck; ledger bolted to the house"}})
    mg = [{
        "id": "b13-0004", "domain": "government_services", "pattern": "verification", "_split": "train",
        "state": mg_state,
        "questions": json.dumps({
            "correct_type": {"type": "choice", "instructions": {"question": "Which permit is required?", "rule_excerpt": "RBP: attached decks at any height."},
                             "criteria": {"rbp": "Residential Building Permit", "ma": "Minor Alteration", "other": None}},
            "severity": {"type": "score", "instructions": "How serious is it?",
                         "criteria": [{"level": "None"}, {"level": "Administrative", "definition": "Paperwork only"},
                                      {"situation": "Small risk", "examples": ["304 indoors"]}]},
            "tone": {"type": "score", "instructions": "How formal?", "criteria": ["Formal", "Neutral", "Casual"]},
            "type_match": {"type": "noul", "instructions": "Does the selected permit match?",
                           "criteria": {"true": "It matches", "false": "It does not"}},
            "orphan": {"type": "noul", "instructions": "Has no target"}}),
        "targets": json.dumps({
            "correct_type": {"difficulty": "moderate", "note": "SECRET-NOTE ledger is attached", "probabilities": {"rbp": 0.9, "ma": 0.05, "other": 0.05}},
            "severity": {"difficulty": "moderate", "note": "SECRET-NOTE two", "probabilities": {"0": 0.0, "1": 0.8, "2": 0.2}},
            "tone": {"difficulty": "easy", "note": "SECRET-NOTE three", "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0}},
            "type_match": {"difficulty": "insufficient", "note": "SECRET-NOTE four", "noul": 0.032}}),
    }, {"id": "cs-1", "domain": "customer_support", "pattern": "routing", "_split": "test",
        "state": json.dumps({"ticket": "My parcel never arrived and I want a refund today"}),
        "questions": json.dumps({"refund": {"type": "noul", "instructions": "Refund?"}}),
        "targets": json.dumps({"refund": {"difficulty": "easy", "note": "x", "noul": 0.9}})}]
    vg_state = "Agent ALEX-77B halted the reconciliation pipeline after a 503 from the legacy SQL cluster and dumped state to disk."
    vg = [
        {"type": "score", "state": vg_state, "question": "Assess data integrity.", "options": ["Loss", "Partial", "Full"],
         "task": "synth:a production incident log", "source": "synth", "ordered": True, "_key": "synth-288-2", "label": [0.002, 0.001, 0.997]},
        {"type": "choice", "state": vg_state, "question": "Why did it halt?", "options": ["DB outage", "Bug"],
         "criteria": {"DB outage": "The SQL cluster was unavailable", "Bug": "A code defect"}, "task": "synth:a production incident log",
         "source": "synth", "_key": "synth-288-3", "label": [1.0, 0.0]},
        {"type": "noul", "state": vg_state, "question": "Was data corrupted?", "options": ["false", "true"],
         "criteria": {"true": "Corruption found", "false": "No corruption"}, "task": "synth:a production incident log",
         "source": "synth", "_key": "synth-288-4", "label": [0.9, 0.1]},
        {"type": "choice", "state": vg_state, "question": "Broken row", "options": ["a", "b", "c"], "criteria": {},
         "task": "synth:a production incident log", "source": "synth", "_key": "synth-288-5", "label": [0.5, 0.5]},
    ]
    pr = [{"id": "arithmetic:train:1", "level": 2,
           "state": "Starting at 12:35: standup (35 min), then code review (35 min). Tasks run back to back in this order.",
           "questions": json.dumps({
               "longest": {"type": "choice", "criteria": {"standup": "standup", "code review": "code review"}, "instructions": "Which task is longest?"},
               "before_noon": {"type": "score", "criteria": ["0", "1", "2"], "instructions": "How many tasks start before 12:00?"},
               "done": {"type": "noul", "instructions": "Is everything done by 13:30?"},
               "unasked": None}),
           "answers": json.dumps({
               "longest": {"type": "choice", "probabilities": {"standup": 0.5, "code review": 0.5}},
               "before_noon": {"type": "score", "legend": {"0": "0", "1": "1", "2": "2"}, "probabilities": {"0": 1.0, "1": 0.0, "2": 0.0}},
               "done": {"type": "noul", "noul": 0.0}, "unasked": None})}]
    jb_state = {"request": "Cancel order CP-2204, I am in a hurry", "steps_taken": [{"tool_used": "search_orders", "output": "status: Shipped"}]}
    jb = [{"id": "synth-v2:helpdesk_ticket:1", "subset": "helpdesk_ticket", "family": "helpdesk_ticket:1", "state": jb_state,
           "questions": {"next": {"type": "choice", "instructions": "Which tool next?",
                                  "criteria": {"P1": "urgent", "P2": "high", "P3": "normal"}},
                         "cancellable": {"type": "noul", "instructions": "Can it be cancelled?",
                                         "criteria": {"true": "Not shipped", "false": "Already shipped"}}},
           "target": {"next": {"P1": 0.2, "P2": 0.3, "P3": 0.499999}, "cancellable": {"true": 0.000321, "false": 0.99968}},
           "provenance": {"judge": {"answered": ["deepseek", "qwen"], "gold": "mean"},
                          "questions": {"next": {"ordered": True}}}},
          {"id": "synth-v2:claim_evidence:2", "subset": "claim_evidence", "state": {"claim": "The lab spent $15,200 on microscopes in April",
                                                                                     "evidence": "The finance report lists $12,000 for microscopes in May"},
           "questions": {"supported": {"type": "noul", "instructions": "Is the claim supported?"}},
           "target": {"supported": {"true": 0.1, "false": 0.9}}, "provenance": {}}]
    kv_prompt = ("<state>\n2026-09-21 ERROR AuthService: Unexpected exception in MFA module\n</state>\n\nQuestion: {q}\n"
                 "Options:\n{opts}\n\nAnswer with one letter.\nAnswer:")
    many = "\n".join(f"{chr(65 + i % 26)}. o{i} - option {i}" for i in range(53))
    kv = [{"prompt": kv_prompt.format(q="Primary category?", opts="A. authentication - Failed logins.\nB. mfa_error - MFA problem.\nC. system_error - General error."),
           "letters": ["A", "B", "C"], "target": [0.0001, 0.9998, 0.0001], "qtype": "choice", "domain": "application error log", "state_id": "s1", "ambiguity": "clear"},
          {"prompt": kv_prompt.format(q="Escalate?", opts="A. no - Not warranted.\nB. yes - Warranted."),
           "letters": ["A", "B"], "target": [0.2, 0.8], "qtype": "noul", "domain": "application error log", "state_id": "s1", "ambiguity": "borderline"},
          {"prompt": kv_prompt.format(q="Urgency?", opts="A. 1 - Normal.\nB. 2 - Soon.\nC. 3 - Now."),
           "letters": ["A", "B", "C"], "target": [0.1, 0.3, 0.6000005], "qtype": "score", "domain": "application error log", "state_id": "s1", "ambiguity": "ambiguous"},
          {"prompt": kv_prompt.format(q="Too many options?", opts=many), "letters": [chr(65 + i % 26) for i in range(53)],
           "target": [1 / 53] * 53, "qtype": "choice", "domain": "application error log", "state_id": "s1", "ambiguity": "clear"}]

    fx = {"mghafiri": list(convert_mghafiri(mg, rng, lab)), "vagmi": list(convert_vagmi(vg, rng, lab)),
          "procedural/arithmetic": list(convert_procedural(pr, "arithmetic", rng, lab)),
          "jebadiah": list(convert_jebadiah(jb, rng, lab)), "kaivoss": list(convert_kaivoss(kv, rng, lab))}
    for name, recs in fx.items():
        assert recs, f"{name}: no records"
        for r in recs:
            validate(r)
        print(f"ok  {name:24s} {len(recs)} records, {sum(len(r['questions']) for r in recs)} questions")

    m = fx["mghafiri"][0]
    blob = json.dumps(m)
    assert "SECRET-NOTE" not in blob and "difficulty" not in blob and "moderate" not in blob, "note/difficulty leaked"
    assert "Rule excerpt: RBP" in m["questions"]["correct_type"]["instructions"]
    assert m["questions"]["correct_type"]["criteria"]["other"] == "Other"
    assert m["questions"]["severity"]["levels"] == ["None", "Administrative: Paperwork only", "Small risk (e.g. 304 indoors)"]
    assert m["labels"]["severity"]["Administrative: Paperwork only"] == 0.8 and m["label_kind"]["tone"] == "hard"
    assert "Yes means: It matches" in m["questions"]["type_match"]["instructions"]
    assert abs(m["labels"]["type_match"]["yes"] - 0.032) < 1e-12 and "orphan" not in m["questions"]
    v = fx["vagmi"][0]
    assert set(v["questions"]) == {"q2", "q3", "q4"} and abs(v["labels"]["q4"]["yes"] - 0.1) < 1e-12
    assert v["label_kind"]["q3"] == "hard" and v["label_kind"]["q2"] == "soft"
    p = fx["procedural/arithmetic"][0]
    assert "unasked" not in p["questions"] and lab.drops["null_question(skipped)"] == 1
    assert p["questions"]["before_noon"]["levels"] == ["0", "1", "2"] and p["labels"]["done"]["yes"] == 0.0
    j = fx["jebadiah"][0]
    assert j["meta"]["teacher_includes_base_model"] is True and j["meta"]["family"] == "helpdesk_ticket"
    assert list(j["questions"]["next"]["criteria"]) == ["P1", "P2", "P3"], "ordered choice must keep its order"
    assert abs(sum(j["labels"]["next"].values()) - 1) < 1e-12 and j["state"].startswith('{"request"')
    k = fx["kaivoss"][0]
    assert set(k["questions"]) == {"q0", "q1", "q2"} and lab.drops["k>52"] == 1
    assert abs(k["labels"]["q1"]["yes"] - 0.8) < 1e-12
    assert "Yes means: Warranted." in k["questions"]["q1"]["instructions"]
    assert k["questions"]["q2"]["levels"] == ["1 - Normal.", "2 - Soon.", "3 - Now."]
    assert k["state"] == "2026-09-21 ERROR AuthService: Unexpected exception in MFA module"
    print("ok  mapping checks (note/difficulty never emitted, noul yes/no, ordered choices, K>52 skipped)")

    # whole pipeline: dedup + strict + write + iter_units read-back
    fps = Fingerprints()
    fps.add(EVAL_TAG, f'"{vg_state.upper()}  "')                           # exact after norm
    fps.add(EVAL_TAG, "Earlier: The lab spent $15,200 on microscopes in April, the finance report lists $12,000 "
                      "for microscopes in May according to a memo.")   # >= 3 shared 8-word shingles
    fps.add("typed_train", "an unrelated typed-decisions train state about invoices and purchase orders")
    with tempfile.TemporaryDirectory() as tmp:
        for strict in (False, True):
            out = Path(tmp) / ("strict" if strict else "full")
            writer, sts = Writer(out), {}
            for name, recs in fx.items():
                st = new_stats(name, Labeler(0.01))
                sts[name] = process(name, list(recs), 5000, fps, strict, writer, random.Random(1), st)
            writer.close()
            assert sts["vagmi"]["dedup_exact"] == {EVAL_TAG: 1}, sts["vagmi"]["dedup_exact"]
            assert sts["jebadiah"]["dedup_shingle"] == {EVAL_TAG: 1}, sts["jebadiah"]["dedup_shingle"]
            if strict:
                assert sts["mghafiri"]["strict_excluded"] == {"domain=customer_support": 1}
                assert sts["jebadiah"]["strict_excluded"] == {"family=helpdesk_ticket": 1}
            else:
                assert not any(s["strict_excluded"] for s in sts.values())
            rb = final_check(out)
            assert "vagmi" not in rb["per_source"]
            assert rb["per_source"]["mghafiri"]["records"]["train"] + rb["per_source"]["mghafiri"]["records"]["val"] == (1 if strict else 2)
            assert not (out / "test").exists()
            print(f"ok  pipeline ({'strict' if strict else 'full'}): exact + shingle dedup, strict, iter_units read-back")
    assert split_of("x" * 40, val_pct=VAL_PCT, test_pct=0) in ("train", "val")
    assert Counter(split_of(f"state {i}", val_pct=VAL_PCT, test_pct=0) for i in range(4000))["val"] in range(120, 290)
    print("\nsample (mghafiri):")
    print(json.dumps(m, indent=1, ensure_ascii=False)[:1500])
    print("\nself-test passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="mghafiri,vagmi,procedural,jebadiah,kaivoss",
                    help="comma-separated subset of: mghafiri,vagmi,procedural,jebadiah,kaivoss")
    ap.add_argument("--procedural-configs", default=",".join(PROCEDURAL_CONFIGS))
    ap.add_argument("--max-per-source", type=int, default=5000)
    ap.add_argument("--procedural-per-config", type=int, default=500)
    ap.add_argument("--oversample", type=float, default=2.0,
                    help="procedural rows sampled per config = this x --procedural-per-config")
    ap.add_argument("--renorm-tol", type=float, default=0.01,
                    help="renormalize targets whose sum is within this of 1; drop the rest")
    ap.add_argument("--eval", action="append", default=[],
                    help="jev_real test jsonl (repeatable); required for a build")
    ap.add_argument("--fingerprint", action="append", default=[],
                    help="TAG=PATH jsonl with a `state` field, also deduplicated against (repeatable)")
    ap.add_argument("--strict", action="store_true",
                    help="also drop domains that overlap typed-decisions' four workflows")
    ap.add_argument("--out", default="jev_ext")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--self-test", action="store_true", help="run converter + pipeline tests offline and exit")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    a.sources = [s.strip() for s in a.sources.split(",") if s.strip()]
    unknown = [s for s in a.sources if s not in DATASETS]
    if unknown:
        ap.error(f"unknown sources: {unknown}")
    a.procedural_configs = [c.strip() for c in a.procedural_configs.split(",") if c.strip()]
    if not a.eval:
        ap.error("--eval <jev_real test jsonl> is required (the eval dedup is not optional)")
    build(a)


if __name__ == "__main__":
    main()
