#!/usr/bin/env python3
"""ZERO-SHOT BY ASKING: score each option with Qwen3.8-27B's own next-token distribution.

Everything so far reads hidden states into a head trained on jev_data, and that head is what fails
to transfer (acc 0.470 on jev_real). This asks the untouched instruct model instead: the options are
lettered, the model is asked for the letter, and p(option) is the softmax over the letter tokens at
the first answer position. The letter/slot assignment is a nuisance (the positional bias R-25
diagnosed), so for `choice` questions p is averaged over `--n-perm` option orders, as in
zeroshot_perm_marg.py. No training anywhere, so this is zero-shot by construction.

The npz is what analysis/per_config_external.py reads (`test_p`, `test_t`, `test_ks`, `test_soft`,
`test_qid`, `test_pred_ent`, `test_tot`), so the per-config table and the JEV comparison come out of
the same script as every other number. Keep "zeroshot" in the output file name: that script derives
the protocol label from it.

(53) --whole-case lists every question of the unit's case (all its criteria, no letters) before asking this one, as
the typed-decisions teacher saw it; --noul-defs shows a noul's yes/no definitions. Both default OFF (prompts unchanged).

    python3 trainers/claude_lmscore.py --data /data/local/jev_real \
        --out analysis/claude_lmscore_zeroshot_units.npz --n-perm 4

(aux-vbll, 2026-10-04) --aux-vbll: a --lora-ckpt trained with claude_lmtune.py --aux-vbll carries a variational Bayesian last layer
on the decision hidden state (claude_lmtune.AuxVBLL). The scorer then also reads, for every prompt, the hidden state the letter logits
come from (the same forward: nothing is run twice), applies the head (posterior draws: a fixed seeded set, so the file is
deterministic), maps each prompt's distribution back to the unit's option order and averages over the unit's prompts as test_p is.
test_p and every other existing array are UNCHANGED. Added arrays (n units; kmax as test_p; per-prompt quantities averaged over the
unit's prompts):
    test_aux_p      [n, kmax]  the head's predictive E_q[softmax(z)] (the selected head, --aux-vbll-head)
    test_aux_epi    [n]        EPISTEMIC: mutual information I(y; beta | x) = H(E_q p) - E_q H(p) per prompt, nats
    test_aux_alea   [n]        ALEATORIC: expected entropy E_q H(p) per prompt, nats
    test_aux_pvar   [n]        sum over options of Var_q[p_j] (a predictive-variance summary of the same draws)
    test_aux_lvar   [n]        closed-form logit variance phi_j^T S phi_j, mean over the shown options (no sampling)
    test_aux_tot    [n]        H(test_aux_p);   test_aux_mx [n]: max of test_aux_p
    aux_vbll_heads  (json)     the checkpoint's head grid; aux_vbll_head: the selected index; aux_vbll_samples: draws used
    test_aux_grid_{p,epi,alea,pvar,lvar}  [G, n(, kmax)]  every head of a grid (only when the checkpoint has G > 1 heads)
--aux-combine mean|geo (default off) adds test_combo_p [n, kmax] = (1 - w) p_main + w p_aux, or p_main^(1-w) p_aux^w renormalised
(w = --aux-combine-weight), per unit. Two drop-in VIEWS are written beside --out, so every script that reads test_p works unchanged:
<out>_auxvbll.npz (test_p = test_aux_p, test_epi = test_aux_epi, test_pred_ent = test_aux_alea, test_tot / test_mx of it; the
claude_vbll.py / claude_uncertainty.py keys) and, with --aux-combine, <out>_auxcombo.npz (test_p = test_combo_p). The head reads the
raw letter logits: with --ordinal-readout, test_p's score rows go through the readout, the head's do not.

(p95, 2026-10-04) --num-head: a --lora-ckpt trained with claude_lmtune.py --num-head carries a numeric mixture head (claude_lmtune.NumHead)
on the decision hidden state. For every NUMERIC unit of the scored data (a record with meta.bin_edges / meta.axis, build_numeric_jev.py)
the scorer reads that state from the same forward, applies the head and writes test_p's row as the head's K-bin masses (level order), so
analysis/eval_numeric.py reads the file unchanged; test_tot / test_pred_ent / test_mx / test_perm_mi of those rows follow test_p. Every
other row, and every other existing array, is what the scorer writes without the flag. Added (n units; NaN / -1 on non-numeric rows):
    test_num            [n] bool       the row is a numeric unit read by the head
    test_letter_p       [n, kmax]      the letter readout's distribution (test_p without --num-head), every row
    test_num_logpi / test_num_mu / test_num_scale  [n, C]   the mixture: log weights, locations, scales in the axis coordinate u
    test_num_transform  [n] int8       index into num_head.transforms (std / logit / log / neglog); test_num_lo / test_num_hi [n] the axis
    test_num_mean / test_num_sd [n]    the mixture's mean and sd in DISPLAY units (the question's own; quadrature)
    test_num_q05 / test_num_q50 / test_num_q95 [n]   its 5 / 50 / 95% quantiles, display units (exact inversion of the CDF)
    num_head            (json)         the head's spec (family, components, coordinates)
and, when any numeric unit was scored, the view <out>_letters.npz (test_p = test_letter_p), so eval_numeric.py can score both readouts.
One prompt per numeric unit (score questions keep their level order); --symmetrize is refused with --num-head.

(p101, 2026-10-04) --prior-offset meta[:KEY] (a --lora-ckpt trained with claude_lmtune.py --prior-offset) and/or --prior-offset-file FILE:
for every unit with a log prior (meta.log_prior_bins / meta.KEY of its record; or FILE's entry for its qid, which takes precedence:
JSONL lines {"qid": .., "log_prior_bins": [..]} or one JSON {qid: [..]}; a list in the question's stored option order or a dict by
option key), each prompt's letter distribution is softmax(letter logits + log pi) in float64, the offsets mapped through the prompt's
option order; test_p averages it over the unit's prompts as before. Every other unit, and every existing array of theirs, is what the
scorer writes without the flag. Added (n units; NaN / False where a unit has no offsets):
    test_evidence_logits [n, kmax]  the EVIDENCE: the raw letter logits mapped back to the stored option (bin) order, mean over the
                                    unit's prompts (a per-prompt constant is irrelevant: q = softmax(log pi + l)); one prompt per unit
                                    (score questions without --symmetrize): softmax(test_log_prior + this) == test_p exactly
    test_log_prior       [n, kmax]  the log prior applied (stored option order)
    test_prior_offset    [n] bool;  prior_offset (json): the source and counts
Any other prior is then applied offline, without the backbone: analysis/apply_prior.py --npz OUT --prior SPEC.

(rowid, 2026-10-04) IMMUTABLE ROW IDENTITY, always written (research/eval_row_identity.md). test_qid is a question KEY shared by many
records ('evidence_level'), and the scorer drops units (K > 52, too long), so a row's position or (qid, target) does not say which
case it is: an evaluator that matched rows in order on (qid, target) could hand a prediction to the wrong case after a skip. Every
row now carries (n units; every existing array and the log are unchanged):
    test_rid        [n] str   the unit's record id (as str)
    test_uid        [n] str   json [record_id, qid] -- plus a third element m >= 1 for the m-th REPEAT of the same (record_id, qid)
                              in cf.iter_units order (a duplicated record, the same id in two splits). Computed over the FULL
                              enumeration of --data / --splits, before --configs / --limit / --exclude-states / --sample, so a
                              unit's id never depends on which other units were scored (unit_uids; analysis/rowid.py recomputes it)
    test_input_sha  [n] str   sha1 (hex) of the exact prompt text(s) scored for the unit, utf-8, the unit's prompts joined by
                              "\\x1e" in scoring order (one prompt: sha1 of that prompt)
    test_opt_sha    [n] str   sha1 (hex) of json [option keys, canonical (stored) order] (option_sha)
    skipped_units   ()  str   json [[uid, reason], ...]: every unit of the scored list left with NO prompt ("k>52" / "too_long")
    rowid_scheme    ()  str   ROWID_SCHEME
The views (<out>_auxvbll / _auxcombo / _letters.npz) carry the four per-row arrays too. Evaluators join on test_uid
(analysis/rowid.py); files without it go through rowid.py's legacy path, which replays plan_prompts (this script's own unit loop and
skip logic) with the scoring run's tokenizer and --max-len.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import itertools
import json
import os
import random
import string
import time
from collections import Counter, defaultdict

import numpy as np
import torch

MODEL = "Qwen/Qwen3.8-27B"
HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("cf", os.path.join(HERE, "cache_features.py"))
cf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cf)

# 52 single-character labels. Each must be ONE token as the first assistant token, checked at load.
LABELS = string.ascii_uppercase + string.ascii_lowercase

# (rowid) the identity arrays every npz carries (module docstring); analysis/rowid.py holds the same definitions for evaluators
ROWID_SCHEME = ("rowid-1: uid = json [record_id, qid(, m)] over the full iter_units enumeration; input_sha = sha1 of the unit's "
                "prompts joined by \\x1e; opt_sha = sha1 of json option keys (stored order)")
ROWID_ROW_KEYS = ("test_rid", "test_uid", "test_input_sha", "test_opt_sha")
ROWID_KEYS = ROWID_ROW_KEYS + ("skipped_units", "rowid_scheme")
UID_FIELD = "_rowid_uid"


def unit_uids(units):
    """(rowid) The immutable id of every unit of cf.iter_units, in its order: json [record_id, qid] (as str), plus a third element
    m >= 1 for the m-th repeat of the same (record_id, qid). Pass the FULL enumeration (before any filter or sample)."""
    seen, out = Counter(), []
    for u in units:
        key = [str(u["record_id"]), str(u["qid"])]
        m = seen[tuple(key)]
        seen[tuple(key)] += 1
        out.append(json.dumps(key + [m] if m else key, ensure_ascii=False))
    return out


def option_sha(keys):
    """(rowid) sha1 hex of json [option keys] in canonical (stored) order."""
    return hashlib.sha1(json.dumps([str(k) for k in keys], ensure_ascii=False).encode("utf-8")).hexdigest()


def prompt_sha(texts):
    """(rowid) sha1 hex of the unit's prompt texts (scoring order) joined by \\x1e."""
    return hashlib.sha1("\x1e".join(texts).encode("utf-8")).hexdigest()

_ORDINAL = None


def ordinal():
    """(49) trainers/claude_ordinal.py, loaded on first use: with every ordinal option off nothing new is imported, so a node
    whose checkout predates (49) runs the default path unchanged."""
    global _ORDINAL
    if _ORDINAL is None:
        s = importlib.util.spec_from_file_location("claude_ordinal", os.path.join(HERE, "claude_ordinal.py"))
        _ORDINAL = importlib.util.module_from_spec(s)
        s.loader.exec_module(_ORDINAL)
    return _ORDINAL


def apply_ordinal_readout(rows, logp, units, mode):
    """(49) --ordinal-readout: replace each `score` prompt's letter log-probs (rows[r] = (unit index, order, text); letter j
    shows level order[j]) by the readout's, computed in level order; returns how many prompts changed. 'none': nothing."""
    if mode == "none":
        return 0
    od, n = ordinal(), 0
    for r, (ui, order, _) in enumerate(rows):
        if units[ui]["qtype"] == "score" and len(order) >= 2:
            logp[r] = od.readout_letters_np(logp[r], order, mode)
            n += 1
    return n

PROMPT = """{state}

Question: {instructions}

Options:
{options}

Answer with the letter of the single best option and nothing else."""

# (53) WHOLE-CASE RENDERING (--whole-case). The typed-decisions teacher labelled whole cases: the state plus every question
# of the case, with every criterion, in one request. So the prompt for one decision lists the WHOLE case first (each
# question with its instructions and option criteria, no letters, so no letter position can be copied), then asks for one
# question exactly as PROMPT does, and the letter distribution is read at the end as before: still one forward pass per
# decision. A case with one question renders exactly as PROMPT.
PROMPT_CASE = """{state}

This case has {n} questions:

{listing}

Answer question {k}: {instructions}

Options:
{options}

Answer with the letter of the single best option and nothing else."""

# Overflow (--max-len): the listing's option criteria are cut to this many words, level by level, before any other question
# is dropped (0 = instructions only); then other questions are dropped, last listed first; the target never is. Only when
# no other question fits is the state cut, as in the single-question prompt.
CASE_CRIT_WORDS = (None, 24, 8, 0)
# A group of single-question records sharing (source, state) with more distinct questions than this is a shared template
# rather than a case: its units stay single-question.
MAX_STATE_CASE = 12


def noul_definitions(crit):
    """{"yes": .., "no": ..} from a noul's `criteria` keyed true/false or yes/no (any case); "" for a side not given;
    None when there is no definition at all (as trainers/claude_verbal_label.py reads them)."""
    if not isinstance(crit, dict) or not crit:
        return None
    low = {str(k).strip().lower(): v for k, v in crit.items()}

    def side(names):
        return next((str(low[k]).strip() for k in names if low.get(k) is not None and str(low[k]).strip()), "")

    d = {"yes": side(("yes", "true")), "no": side(("no", "false"))}
    return d if (d["yes"] or d["no"]) else None


def option_texts(q, noul_defs=False):
    """The option texts shown for a unit (or a case_questions view). `noul_defs`: a noul with definitions shows
    "Yes: <definition>" / "No: <definition>" instead of the bare Yes / No (in the letters and in the case listing)."""
    texts = q["option_texts"]
    if noul_defs and q["qtype"] == "noul":
        d = noul_definitions(q.get("noul_criteria"))
        if d:
            texts = [f"{t}: {d[k]}" if d.get(k) else t for t, k in zip(texts, q["option_keys"])]
    return texts


def _case_view(u):
    return {"qid": str(u["qid"]), "qtype": u["qtype"], "instructions": u["instructions"],
            "option_keys": list(u["option_keys"]), "option_texts": list(u["option_texts"]),
            "noul_criteria": u.get("noul_criteria")}


def _question_signature(u):
    """Two units with this equal ask the same question (option order aside): one entry in the case listing."""
    return (u["qtype"], " ".join(str(u["instructions"]).split()),
            tuple(sorted(" ".join(str(t).split()) for t in u["option_texts"])))


def attach_case_questions(units, max_state_case=MAX_STATE_CASE):
    """Group cf.iter_units units into CASES and attach to every unit:
        case_questions  the case's questions (compact views, stored order; the SAME list object for the whole case)
        case_index      this unit's position in case_questions
        case_id         a stable id (sha1) of the case key, which seeds the scoring-time listing order
    Grouping, without rewriting any data:
      * a record with several questions is one case (typed-style synthetic sets, jev_ext_strict, jev_data multi-question
        sources): key (split, source, record id);
      * single-question records group by (split, source, state): jev_real stores typed-decisions as ONE RECORD PER QUESTION
        with a shared state (convert_jevbench.emit). An empty state, or a group with more than `max_state_case` distinct
        questions (a shared template, not a case), stays single.
    Questions repeated within a case (option-reshuffled variants, duplicate rows) are listed once. Returns a summary."""
    per_rec = Counter((u["split"], u["source"], u["record_id"]) for u in units)
    groups = defaultdict(list)
    for i, u in enumerate(units):
        rk = (u["split"], u["source"], u["record_id"])
        if per_rec[rk] > 1:
            key = ("record",) + rk
        elif str(u["state"]).strip():
            key = ("state", u["split"], u["source"], u["state"])
        else:
            key = ("single", i)
        groups[key].append(i)
    sizes, multi_by_src, units_by_src, too_big = Counter(), Counter(), Counter(), 0
    for key, idx in groups.items():
        views, pos, where = [], {}, {}
        for i in idx:
            s = _question_signature(units[i])
            if s not in pos:
                pos[s] = len(views)
                views.append(_case_view(units[i]))
            where[i] = pos[s]
        if key[0] == "state" and len(views) > max_state_case:
            too_big += 1
            for i in idx:
                cid = hashlib.sha1(json.dumps(["single", units[i]["split"], units[i]["source"], units[i]["record_id"],
                                                str(units[i]["qid"])]).encode()).hexdigest()[:16]
                units[i].update(case_questions=[_case_view(units[i])], case_index=0, case_id=cid)
                sizes[1] += 1
            continue
        cid = hashlib.sha1(json.dumps(list(key), ensure_ascii=False).encode()).hexdigest()[:16]
        for i in idx:
            units[i].update(case_questions=views, case_index=where[i], case_id=cid)
        sizes[len(views)] += len(idx)
    for u in units:
        units_by_src[u["source"]] += 1
        multi_by_src[u["source"]] += len(u["case_questions"]) > 1
    return {"units": len(units), "cases": len({u["case_id"] for u in units}),
            "units_by_case_size": dict(sorted(sizes.items())), "state_groups_too_big": too_big,
            "units_in_multi_question_cases": int(sum(multi_by_src.values())),
            "multi_by_source": {s: f"{multi_by_src[s]}/{units_by_src[s]}" for s in sorted(units_by_src)
                                if multi_by_src[s]}}


def fixed_case_order(unit, seed=0):
    """The scoring-time order of the case listing: a permutation of case_questions seeded by the case id, so it is the
    same for every question of the case, for every run, and whichever units are sampled."""
    n = len(unit.get("case_questions") or ())
    order = list(range(n))
    h = int(hashlib.sha1(f"{seed}|{unit.get('case_id', '')}".encode()).hexdigest()[:16], 16)
    random.Random(h).shuffle(order)
    return order


def _clip_words(text, cap):
    w = str(text).split()
    if cap is None or len(w) <= cap:
        return " ".join(w)
    return " ".join(w[:cap]) + " ..."


def case_listing(views, seq, tgt, order, tgt_texts, cap, noul_defs=False):
    """(listing text, k): the questions views[seq] numbered 1..n, each with its instructions and (cap != 0) its option
    criteria as unlettered bullets, cut to `cap` words (None = whole). The target views[tgt] shows the unit's OWN option
    texts in the presented `order`, so the canonical order is never shown; k = its number in the listing."""
    lines = []
    for n, vi in enumerate(seq, 1):
        v = views[vi]
        lines.append(f"{n}. {_clip_words(v['instructions'], None)}")
        if cap != 0:
            texts, idx = (tgt_texts, order) if vi == tgt else (option_texts(v, noul_defs), range(len(v["option_texts"])))
            lines += [f"   - {_clip_words(texts[o], cap)}" for o in idx]
    return "\n".join(lines), seq.index(tgt) + 1


def bos_prefix(tok):
    """tok.bos_token if the tokenizer's default encoding starts with it, else "" (Qwen: none)."""
    b = getattr(tok, "bos_token", None)
    if not b or tok.bos_token_id is None:
        return ""
    ids = tok("x", add_special_tokens=True)["input_ids"]
    return b if ids and ids[0] == tok.bos_token_id else ""


# (66) PER-FAMILY CHAT PRESETS, keyed by config.model_type. The scorer, the tuner and the backbone probe all look the model type
# up here (chat_preset) and pass the entry to build_prompt(chat=); a model type that is not listed renders exactly as before.
#   muse_glimmer (meta-models/Muse-Glimmer-30B, ATEM template): the generation prompt stops at "<|start|>assistant", and the model
#   then names a recipient: " to=self<|message|>" opens the reasoning channel, " to=user<|message|>" the answer (the tokenizer's
#   own response_template says so). The template has no switch that turns reasoning off (it ignores enable_thinking), so the
#   answer header is PRE-FILLED: exactly what the template renders for an assistant reply to the user (the backbone probe's pre
#   stage checks this against the template), as enable_thinking=False pre-fills Qwen's empty think block. Reasoning strength is
#   the lowest level the card offers. The date is pinned because by default the template stamps TODAY's date (strftime_now)
#   into its system block, so the same unit would render differently on different days.
CHAT_PRESETS = {
    "muse_glimmer": {"chat_kwargs": {"reasoning_strength": "low", "current_date": "2026-10-01"},
                     "answer_prefix": " to=user<|message|>"},
}


def chat_preset(model_type, plain=False):
    """(66) a copy of CHAT_PRESETS[model_type], or None (an unlisted type, or --plain-prompt, which uses no chat template)."""
    p = None if plain else CHAT_PRESETS.get(model_type)
    return json.loads(json.dumps(p)) if p else None


def model_loader(config, choice="auto"):
    """(66) ("causal-lm" | "image-text", transformers auto class) that loads `config` for TEXT-ONLY letter scoring.
    'auto' keeps AutoModelForCausalLM -- the old load, exactly -- for every model type this transformers maps to a causal LM. That
    includes Gemma-4, whose causal-LM mapping IS its image-text class (Gemma4ForConditionalGeneration), and Qwen3.5 (text-only
    Qwen3_5ForCausalLM), so every backbone used so far loads as before. A type mapped ONLY as image-text-to-text (Muse-Glimmer,
    Mistral3) gets AutoModelForImageTextToText: the class the checkpoint is published as, called with input_ids alone (no
    pixel_values), so the vision tower never runs, the language model reads the prompt, and the model's own head (softcap, logit
    multiplier) gives the letter logits. 'causal-lm' / 'image-text' force one."""
    import transformers
    from transformers.models.auto import modeling_auto as ma
    mt = getattr(config, "model_type", None)
    if choice == "image-text" or (choice == "auto" and mt not in ma.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
                                  and mt in ma.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES):
        return "image-text", transformers.AutoModelForImageTextToText
    return "causal-lm", transformers.AutoModelForCausalLM


def lora_scale_kw(te, cfg):
    """(ss, 2026-10-08) apply_lora's `scale` keyword for a checkpoint config: {} for a default checkpoint (no lora_scale_rule: alpha / r,
    exactly as before), {"scale": s} for one trained under an explicit scale rule (train_e2e.lora_scale_from_config). A train_e2e.py
    too old to know scale rules refuses such a checkpoint instead of scoring it with the wrong adapter multiplier."""
    if cfg.get("lora_scale_rule") is None and "lora_alpha" not in cfg:
        return {}
    fn = getattr(te, "lora_scale_from_config", None)
    if fn is None:
        raise SystemExit(f"the checkpoint uses LoRA scale rule {cfg.get('lora_scale_rule')!r}; this train_e2e.py predates scale rules")
    s = fn(cfg)
    return {} if s is None else {"scale": s}


def lora_scope(model, targets):
    """(66) The module path of a multimodal model's LANGUAGE MODEL (e.g. "model.language_model") when train_e2e.apply_lora's rule
    for `targets` would also adapt a Linear OUTSIDE it (Muse-Glimmer's vision tower has q/k/v_proj), else None. Passed as
    apply_lora(scope=), LoRA stays on the language model's attention / MLP. None -- Qwen, LFM2, and Gemma-4, whose vision Linears
    match no target -- means apply_lora runs exactly as before: same adapters, same init draws, same checkpoint keys."""
    lm = None
    for path in ("model.language_model", "language_model", "model.text_model"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if isinstance(obj, torch.nn.Module):
            lm = path
            break
    if lm is None:
        return None
    for name, mod in model.named_modules():
        for child_name, child in mod.named_children():
            path = f"{name}.{child_name}" if name else child_name
            hit = any(path.endswith(t) if "." in t else t in child_name for t in targets)   # apply_lora's rule
            if hit and isinstance(child, torch.nn.Linear) and not path.startswith(lm + "."):
                return lm
    return None


def build_prompt(tok, unit, order, max_len, margin=16, plain=False, whole_case=False, noul_defs=False, case_order=None,
                 stats=None, chat=None):
    """Chat-formatted prompt with options in `order`, the STATE cut by tokens to fit `max_len`. `plain`: no chat template
    (for pre-trained BASE models): the same text followed by "\n\nAnswer:", scored over " A", " B", ... (leading space).
    `whole_case` (53): when unit["case_questions"] (attach_case_questions) holds more than one question, PROMPT_CASE lists
    them all in `case_order` (indices into case_questions; default fixed_case_order(unit)) and asks this one, lettered as
    here; overflow is handled as CASE_CRIT_WORDS says. `noul_defs`: show a noul's definitions (option_texts).
    `stats` (a dict): counts which rendering each prompt got. `chat` (66): a chat_preset() entry -- extra chat-template kwargs
    and an answer header appended after the generation prompt. All defaults: the prompt is exactly the pre-(53) one."""
    texts = option_texts(unit, noul_defs)
    opts = "\n".join(f"{LABELS[j]}) {texts[o]}" for j, o in enumerate(order))
    chat = chat or {}

    def render(state, case=None):
        if case is None:
            msg = PROMPT.format(state=state, instructions=unit["instructions"], options=opts)
        else:
            listing, k, n = case
            msg = PROMPT_CASE.format(state=state, n=n, listing=listing, k=k, instructions=unit["instructions"], options=opts)
        if plain:
            # a base model expects the BOS its tokenizer would add (Gemma is badly conditioned without it); prompts are
            # tokenized with add_special_tokens=False everywhere, so it goes in the text. Qwen has no BOS: unchanged.
            return bos_prefix(tok) + msg + "\n\nAnswer:"
        return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False,
                                       **chat.get("chat_kwargs", {})) + chat.get("answer_prefix", "")

    def count(key):
        if stats is not None:
            stats[key] = stats.get(key, 0) + 1

    views = unit.get("case_questions") if whole_case else None
    if whole_case and views is None:
        raise ValueError("whole_case needs the units grouped first: attach_case_questions(units)")
    if whole_case and len(views) < 2:
        count("single_question")
    if views and len(views) > 1:
        tgt = unit["case_index"]
        seq = list(case_order) if case_order is not None else fixed_case_order(unit)
        if sorted(seq) != list(range(len(views))):
            raise ValueError(f"case_order {seq} is not a permutation of the {len(views)} case questions")
        n_state = len(tok(unit["state"], add_special_tokens=False)["input_ids"])

        def fits(keep, cap):
            listing, k = case_listing(views, keep, tgt, order, texts, cap, noul_defs)
            over = len(tok(render("", (listing, k, len(keep))), add_special_tokens=False)["input_ids"])
            return (listing, k, len(keep)) if n_state <= max_len - over - margin else None

        for cap in CASE_CRIT_WORDS:                      # 1. cut the listing's criteria, level by level
            case = fits(seq, cap)
            if case:
                count("case_full" if cap is None else "case_criteria_cut")
                return render(unit["state"], case)
        keep = list(seq)                                 # 2. drop other questions, last listed first; never the target
        while len(keep) > 2:
            del keep[max(i for i, vi in enumerate(keep) if vi != tgt)]
            case = fits(keep, CASE_CRIT_WORDS[-1])
            if case:
                count("case_questions_dropped")
                return render(unit["state"], case)
        count("case_all_dropped")                        # 3. nothing else fits: the single-question prompt, state cut

    overhead = len(tok(render(""), add_special_tokens=False)["input_ids"])
    budget = max_len - overhead - margin
    if budget <= 0:
        return None
    state = unit["state"]
    enc = tok(state, add_special_tokens=False, return_offsets_mapping=True)
    if len(enc["input_ids"]) > budget:
        state = state[: int(enc["offset_mapping"][budget - 1][1])]
    return render(state)


def plan_prompts(tok, units, rng, n_perm=4, max_len=4096, all_perms_max_k=0, exclude_identity=False, symmetrize=False,
                 plain=False, whole_case=False, noul_defs=False, seed=0, chat=None, fwd=None, stats=None):
    """The scorer's unit loop (main() calls exactly this): every unit's option orders, drawn from `rng` in unit order, and its prompt
    texts. -> (rows [(unit index, order, text)], skipped {"k>52": n, "too_long": n} (the log's counts, per prompt), skip_reason
    {unit index: "k>52" | "too_long"} for every unit left with NO prompt). `fwd` (--forward-sample): units outside it get their
    orders drawn but no prompt. (rowid) analysis/rowid.py replays this with a scoring run's tokenizer and arguments to re-derive
    which units an old npz (no test_uid) holds, in order."""
    rows, skipped, reason = [], {"k>52": 0, "too_long": 0}, {}
    for ui, u in enumerate(units):
        k = len(u["option_keys"])
        if k > len(LABELS):
            skipped["k>52"] += 1
            reason[ui] = "k>52"
            continue
        orders = [list(range(k))]
        if u["qtype"] == "choice" and k >= 2:
            if k <= all_perms_max_k:
                orders = [list(o) for o in itertools.permutations(range(k))]
            elif exclude_identity:
                orders = []
                while len(orders) < n_perm:
                    o = list(rng.permutation(k))
                    if o != list(range(k)):
                        orders.append(o)
            else:
                for _ in range(n_perm - 1):
                    orders.append(list(rng.permutation(k)))
        elif symmetrize and u["qtype"] == "noul":
            orders = [[0, 1], [1, 0]]
        elif symmetrize and u["qtype"] == "score" and k >= 2:
            orders = [list(range(k)), list(range(k))[::-1]]
        corder = fixed_case_order(u, seed) if whole_case else None    # the case listing: one fixed order per case
        if fwd is not None and ui not in fwd:
            continue
        n0 = len(rows)
        for order in orders:
            text = build_prompt(tok, u, order, max_len, plain=plain, whole_case=whole_case,
                                noul_defs=noul_defs, case_order=corder, stats=stats, chat=chat)
            if text is None:
                skipped["too_long"] += 1
                if len(rows) == n0:
                    reason[ui] = "too_long"
                break
            rows.append((ui, order, text))
    return rows, skipped, reason


def dup_layers(stack, spec, base=None):
    """rebuild the decoder-layer list with a block repeated (weights shared) and keep config.layer_types / num_hidden_layers
    consistent (Qwen3.5-style hybrids pick each layer's mask by POSITION from config.layer_types). `base` = (layers, types) of the
    undup'd model, so a sweep can re-apply from scratch; returns it."""
    cfg = stack.config
    if base is None:
        base = (list(stack.layers), list(getattr(cfg, "layer_types", []) or []))
    layers, types = base
    order = list(range(len(layers)))
    if spec:
        st, ln, *rp = [int(x) for x in spec.split(":")]; rp = rp[0] if rp else 1
        order = list(range(st + ln)) + list(range(st, st + ln)) * rp + list(range(st + ln, len(layers)))
    stack.layers = torch.nn.ModuleList([layers[i] for i in order])
    if types:
        cfg.layer_types = [types[i] for i in order]
    cfg.num_hidden_layers = len(order)
    return base


class FrozenLabelHead(torch.nn.Module):
    """(speed, 2026-10-06) --letter-head selected: ONLY the option-letter rows of a frozen plain Linear output head, so the letter
    logits are a [B, H] x [H, 52] product instead of the full-vocabulary head followed by a selection. The same map as
    cf.head_logits(model, h)[:, label_ids] -- the head, then output_multiplier, then final_logit_softcapping, all pointwise, so they
    commute with the selection -- in the head's own dtype. A GEMM of another shape can round differently (bf16): not a claim of bit
    identity. Port of research/claude_bayes_handoff/speed_components.FrozenLabelHead (CPU-verified there). For the letter readout
    only, which normalises over the shown letters: never for full-vocabulary normalisation or generation. h keeps its autograd path
    (the trainer's LoRA gradients); the cached rows do not. Rebuild after any output-head change."""

    def __init__(self, head, label_ids, multiplier=None, softcap=None):
        super().__init__()
        if not isinstance(head, torch.nn.Linear):
            raise TypeError(f"the output head is a {type(head).__name__}, not a plain Linear: inspect its semantics before narrowing it")
        if head.weight.requires_grad or (head.bias is not None and head.bias.requires_grad):
            raise ValueError("the output head must be frozen")
        ids = torch.as_tensor(label_ids, dtype=torch.long, device=head.weight.device)
        if ids.ndim != 1 or not len(ids) or len(ids.unique()) != len(ids):
            raise ValueError("need a nonempty vector of distinct label ids")
        if bool((ids < 0).any()) or bool((ids >= head.out_features).any()):
            raise ValueError("label id out of range")
        self.register_buffer("weight", head.weight.detach().index_select(0, ids).clone())
        self.register_buffer("bias", None if head.bias is None else head.bias.detach().index_select(0, ids).clone())
        self.multiplier, self.softcap = multiplier, softcap

    def forward(self, h):
        z = torch.nn.functional.linear(h, self.weight, self.bias).float()
        if self.multiplier:
            z = z * self.multiplier
        if self.softcap:
            z = self.softcap * torch.tanh(z / self.softcap)
        return z


def label_head(model, label_ids):
    """(speed) FrozenLabelHead over the model's own output head with the multiplier / softcap cf.head_logits reads from the config:
    [B, H] -> the letter logits [B, len(label_ids)]. For no-grad inference callers (scorer, extractors): the head's requires_grad flag
    is cleared first, as nothing trains it there."""
    head = model.get_output_embeddings()
    if isinstance(head, torch.nn.Linear):
        head.requires_grad_(False)
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    return FrozenLabelHead(head, label_ids, getattr(cfg, "output_multiplier", None), getattr(cfg, "final_logit_softcapping", None))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/local/jev_real")
    ap.add_argument("--splits", default="test")
    ap.add_argument("--limit", type=int, default=None, help="cap units (smoke test)")
    ap.add_argument("--sample", type=int, default=0, help="a seeded random sample of this many units (0 = all)")
    ap.add_argument("--forward-sample", type=int, default=0,
                    help="(tfparity) SCORE only a seeded sample of this many units, but draw every unit's option orders exactly as "
                         "the full run does, so each scored prompt is the full run's prompt for that unit (unlike --sample, which "
                         "changes every draw). The npz then carries test_full_row = the unit's row in the full run's file. 0 = off")
    ap.add_argument("--forward-filter", choices=("all", "typed"), default="all",
                    help="with --forward-sample: sample among all units, or only typed-decisions units (qid without '/')")
    ap.add_argument("--exclude-states", default="", help="as claude_think.py (and drops K > 52 first, so a sample replays it)")
    ap.add_argument("--n-perm", type=int, default=4, help="option orders per choice unit, incl. identity")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--tokens-per-batch", type=int, default=24576)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs", default="", help="comma list of qid prefixes (config names); empty = all")
    ap.add_argument("--exclude-identity", action="store_true",
                    help="random option orders ONLY. R-43..45 (TODO.md): a benchmark's canonical order can "
                         "carry the signal, so any average that includes it is that artifact diluted")
    ap.add_argument("--all-perms-max-k", type=int, default=0,
                    help="choice units with K <= this score ALL K! orders, identity included: the exact "
                         "marginal over the order nuisance under a uniform prior. The full set is the same "
                         "whatever the benchmark's canonical order, so that order cannot leak, and unlike "
                         "--exclude-identity it is unbiased at small K (K = 2 would keep only the swap)")
    ap.add_argument("--symmetrize", action="store_true",
                    help="also marginalise presentation where meaning allows: yes/no in both orders, an "
                         "ordinal scale as written and reversed (the transformations that keep its order)")
    ap.add_argument("--ordinal-readout", choices=("none", "cumlink", "smooth"), default="none",
                    help="(49) read each `score` prompt's letter distribution through an ordinal map, in level order: "
                         "'smooth' = nearest unimodal distribution in KL (post hoc, any model), 'cumlink' = location-scale "
                         "cumulative logit derived from the letter logits (what claude_lmtune.py --ordinal-readout cumlink "
                         "trains); trainers/claude_ordinal.py. An adapter trained with a readout must be scored with it")
    ap.add_argument("--save-hidden", default="", help="also save each prompt's final hidden state at the answer position (fp16 .npy, "
                    "rows in the --save-rows order) -- the vector the letter logits are read from; for claude_gapa.py")
    ap.add_argument("--save-rows", action="store_true",
                    help="also store every prompt's letter log-probs and order (row_*), for claude_temper.py")
    ap.add_argument("--plain-prompt", action="store_true",
                    help="pre-trained BASE model: no chat template; prompt ends 'Answer:' and letters are scored with a leading space")
    ap.add_argument("--exact-path", action="store_true",
                    help="the FULL model forward (logits_to_keep=1) on EQUAL-LENGTH batches, no padding and no bare text stack: "
                         "for architectures whose stack or padding breaks the fast path (Gemma-4-12B-it, see find_text_stack)")
    ap.add_argument("--exact-batch", choices=("off", "pad", "pack"), default="off",
                    help="(gemmafast) with --exact-path: batch prompts of DIFFERENT lengths through the full model forward, exactly as "
                         "claude_lmtune.py --exact-batch does (its code is loaded): 'pad' = the length-sorted prompts right-padded "
                         "with an attention mask up to --tokens-per-batch padded tokens, each row read at its own last token (little "
                         "padding, since the scorer sorts by length); 'pack' = one row, per-prompt position_ids, up to --exact-tokens "
                         "tokens. Equal to off in exact arithmetic; in bf16 within the equal-length batching's own noise on Gemma-4 "
                         "(research/gemma_fast.md). off = equal-length batches only (unchanged)")
    ap.add_argument("--exact-tokens", type=int, default=8192,
                    help="--exact-batch pack: tokens per packed forward (pack attends over the whole row, so its attention grows "
                         "with the square of this)")
    ap.add_argument("--exact-head", choices=("model", "fp32"), default="model",
                    help="(gemmafast) with --exact-path: 'fp32' = the model forward up to the final hidden state, then the model's "
                         "head formula on the option letters in fp32 (claude_lmtune.py --exact-head fp32); 'model' = the model's "
                         "own bf16 head (unchanged)")
    ap.add_argument("--letter-head", choices=("full", "selected"), default="full",
                    help="(speed, 2026-10-06) stack path: 'full' (the default, unchanged) = the full-vocabulary head (cf.head_logits), "
                         "then the option letters; 'selected' = the same map from the 52 letter rows alone (FrozenLabelHead, cached "
                         "once). Equal in exact arithmetic; bf16 GEMM rounding may differ, so check per-item parity before relying on it")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dup-layers", default="", help="(43) RYS-style layer duplication at inference: START:LEN[:REPEATS] runs layers "
                    "START..START+LEN-1 REPEATS more times (default 1), same weights; the per-position layer_types follow")
    ap.add_argument("--lora-ckpt", default="", help="claude_lmtune.py checkpoint to apply before scoring")
    ap.add_argument("--whole-case", action="store_true",
                    help="(53) list every question of the unit's case (state-sharing single-question records, or a "
                         "multi-question record) before asking this one; unlettered, in a fixed order seeded by the case "
                         "and --seed; still one forward pass and the same letter readout. Single-question cases unchanged")
    ap.add_argument("--noul-defs", action="store_true",
                    help="(53) show a noul's yes/no definitions (its `criteria`, when the data has them) as 'Yes: ...' / "
                         "'No: ...'; with or without --whole-case")
    ap.add_argument("--loader", choices=("auto", "causal-lm", "image-text"), default="auto",
                    help="(66) model class: 'auto' = AutoModelForCausalLM whenever transformers maps the model type to one (every "
                         "backbone so far, Gemma-4 included: unchanged), else AutoModelForImageTextToText for an image-text-to-text "
                         "architecture (Muse-Glimmer, Mistral3), run on text-only prompts (model_loader)")
    ap.add_argument("--chat-preset", choices=("auto", "none"), default="auto",
                    help="(66) 'auto': the model type's CHAT_PRESETS entry if it has one (Muse-Glimmer: answer header pre-filled, "
                         "lowest reasoning strength, date pinned); no other model type has one, so their prompts are unchanged")
    ap.add_argument("--aux-vbll", action="store_true",
                    help="(aux-vbll) also apply the --lora-ckpt's auxiliary Bayesian last layer (claude_lmtune.py --aux-vbll) to every "
                         "prompt's decision hidden state: adds test_aux_* arrays (predictive, epistemic, aleatoric; see the module "
                         "docstring) and the view <out>_auxvbll.npz; test_p is unchanged")
    ap.add_argument("--aux-vbll-head", type=int, default=0, help="(aux-vbll) the head of the checkpoint's grid that test_aux_* holds")
    ap.add_argument("--aux-vbll-samples", type=int, default=0,
                    help="(aux-vbll) posterior draws (a fixed seeded set); 0 = the value the head was validated with in training")
    ap.add_argument("--aux-combine", choices=("off", "mean", "geo"), default="off",
                    help="(aux-vbll) post hoc, per unit: 'mean' = (1 - w) p_main + w p_aux, 'geo' = p_main^(1-w) p_aux^w "
                         "renormalised (product of experts, exponents summing to 1) -> test_combo_p and <out>_auxcombo.npz; "
                         "off (the default) = nothing")
    ap.add_argument("--aux-combine-weight", type=float, default=0.5, help="(aux-vbll) w of --aux-combine, in [0, 1]")
    ap.add_argument("--num-head", action="store_true",
                    help="(p95) apply the --lora-ckpt's numeric mixture head (claude_lmtune.py --num-head) to every NUMERIC unit: its "
                         "test_p row becomes the head's bin masses (eval_numeric.py reads it unchanged), plus the mixture parameters, "
                         "continuous summaries and test_letter_p (see the module docstring); other rows unchanged")
    ap.add_argument("--prior-offset", default="off",
                    help="(p101) 'meta' or 'meta:KEY': units whose record has meta.log_prior_bins (meta.KEY) are read as softmax(letter "
                         "logits + log pi), as claude_lmtune.py --prior-offset trained them; adds test_evidence_logits (any prior can then "
                         "be applied offline: analysis/apply_prior.py), test_log_prior, test_prior_offset. 'off' (the default) = unchanged")
    ap.add_argument("--prior-offset-file", default="",
                    help="(p101) an ALTERNATIVE prior at score time: JSONL {\"qid\", \"log_prior_bins\"} lines (or one JSON {qid: [...]}); "
                         "its units are read as with --prior-offset (the file's prior wins over meta)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    po_on = a.prior_offset != "off" or bool(a.prior_offset_file)     # (p101)
    if po_on:
        for bad, why in ((a.num_head, "--num-head"), (a.aux_vbll, "--aux-vbll"), (a.ordinal_readout != "none", "--ordinal-readout")):
            if bad:
                raise SystemExit(f"--prior-offset / --prior-offset-file with {why}: not supported")
    if a.aux_combine != "off" and not a.aux_vbll:
        raise SystemExit("--aux-combine needs --aux-vbll")
    if a.aux_vbll and not a.lora_ckpt:
        raise SystemExit("--aux-vbll needs a --lora-ckpt trained with claude_lmtune.py --aux-vbll")
    if not 0.0 <= a.aux_combine_weight <= 1.0 or a.aux_vbll_samples < 0:
        raise SystemExit("--aux-combine-weight in [0, 1], --aux-vbll-samples >= 0")
    if a.num_head and not a.lora_ckpt:
        raise SystemExit("--num-head needs a --lora-ckpt trained with claude_lmtune.py --num-head")
    if a.num_head and a.symmetrize:
        raise SystemExit("--num-head with --symmetrize: not supported (one prompt, in level order, per numeric unit)")
    if (a.exact_batch != "off" or a.exact_head != "model") and not a.exact_path:
        raise SystemExit("--exact-batch / --exact-head apply to the --exact-path only")
    if a.letter_head != "full" and a.exact_path:
        raise SystemExit("--letter-head selected applies to the stack path only (the exact path reads the model's own head)")
    lt = None
    if a.exact_batch != "off" or a.exact_head != "model":
        # (gemmafast) the TRAINER's implementation, so scoring and training read the letters the same way
        _lspec = importlib.util.spec_from_file_location("lt_for_scoring", os.path.join(HERE, "claude_lmtune.py"))
        lt = importlib.util.module_from_spec(_lspec)
        _lspec.loader.exec_module(lt)
        if not hasattr(lt, "letter_head_fp32"):
            raise SystemExit("--exact-batch / --exact-head need the (gemmafast) claude_lmtune.py beside claude_lmscore.py")
    if "jev_real" in a.data and not any(t in os.path.basename(a.out) for t in ("zeroshot", "_fitted_")):
        # "_fitted_" marks a model trained on the typed-decisions TRAIN split ((61), user OK 2026-09-30): never a zero-shot file
        raise SystemExit("--out must contain 'zeroshot' (or '_fitted_' for a train-split-fitted model) so the protocol is labelled")

    from transformers import AutoConfig, AutoTokenizer
    mcfg = AutoConfig.from_pretrained(a.model)
    loader, model_cls = model_loader(mcfg, a.loader)
    chat = chat_preset(mcfg.model_type, a.plain_prompt) if a.chat_preset == "auto" else None
    if chat:
        print(f"chat preset for {mcfg.model_type!r}: {json.dumps(chat)}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    if getattr(tok, "padding_side", "right") != "right":
        # the last real token is read from attention_mask.sum() - 1, which assumes RIGHT padding; set it rather than refuse
        # (Gemma's tokenizer defaults to left padding)
        print(f"tokenizer padding_side was {tok.padding_side!r}; setting 'right'", flush=True)
        tok.padding_side = "right"
    label_ids = [tok((" " + c) if a.plain_prompt else c, add_special_tokens=False)["input_ids"] for c in LABELS]
    if any(len(x) != 1 for x in label_ids):
        raise SystemExit(f"a label is not a single token: {[c for c, x in zip(LABELS, label_ids) if len(x) != 1]}")
    label_ids = torch.tensor([x[0] for x in label_ids])

    t0 = time.time()
    dev = a.device
    model = model_cls.from_pretrained(
        a.model, dtype=torch.bfloat16 if dev == "cuda" else torch.float32, device_map=dev)
    if loader != "causal-lm":                         # (66)
        print(f"loader {loader}: {type(model).__name__} ({model_cls.__name__}) for model_type {mcfg.model_type!r}, text-only "
              f"prompts", flush=True)
    model.eval()
    stack = None if a.exact_path else cf.find_text_stack(model, tok, dev)
    if a.lora_ckpt:
        # Same adapters claude_lmtune.py trained; refuse a partial load, which would score a model
        # that is neither the base nor the tuned one while looking fine.
        spec = importlib.util.spec_from_file_location("te", os.path.join(HERE, "train_e2e.py"))
        te = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(te)
        ck = torch.load(a.lora_ckpt, map_location="cpu")
        if bool(ck["config"].get("plain_prompt", False)) != bool(a.plain_prompt):
            # an adapter trained on one prompt format scored on the other is a different (untested) model
            raise SystemExit(f"adapter was trained with plain_prompt={ck['config'].get('plain_prompt', False)}; "
                             f"pass the same --plain-prompt setting")
        scope = ck["config"].get("lora_scope")             # (66) only in checkpoints whose LoRA was kept to the language model
        te.apply_lora(model, int(ck["config"]["lora_r"]), targets=tuple(ck["config"]["lora_targets"]),
                      **({"scope": scope} if scope else {}), **lora_scale_kw(te, ck["config"]))
        want = {n for n, _ in model.named_parameters() if n.endswith(".A") or n.endswith(".B")}
        if set(ck["lora"]) != want:
            raise SystemExit(f"LoRA key mismatch: {len(set(ck['lora']) ^ want)} keys differ")
        model.load_state_dict(ck["lora"], strict=False)
        print(f"LoRA applied from {a.lora_ckpt}: {len(want)} adapter tensors, "
              f"step {ck['config'].get('steps')}", flush=True)
        for f in ("whole_case", "noul_defs"):          # a legitimate ablation either way, so a note, not a refusal
            if bool(ck["config"].get(f, False)) != bool(getattr(a, f)):
                print(f"NOTE: adapter trained with {f}={ck['config'].get(f, False)}, scored with {f}={getattr(a, f)}",
                      flush=True)
        if ck["config"].get("chat_preset") != chat:    # (66) likewise
            print(f"NOTE: adapter trained with chat preset {ck['config'].get('chat_preset')}, scored with {chat}", flush=True)
        if ck["config"].get("exact_head", "model") != a.exact_head:   # (gemmafast) a precision choice: a note, not a refusal
            print(f"NOTE: adapter trained with --exact-head {ck['config'].get('exact_head', 'model')}, scored with {a.exact_head}",
                  flush=True)
        if ck["config"].get("letter_head", "full") != a.letter_head:   # (speed) likewise a rounding choice
            print(f"NOTE: adapter trained with --letter-head {ck['config'].get('letter_head', 'full')}, scored with {a.letter_head}",
                  flush=True)
        tr = ck["config"].get("ordinal_readout", "none")   # (49) an adapter trained through a readout IS that readout
        if tr != "none" and tr != a.ordinal_readout:
            raise SystemExit(f"adapter was trained with --ordinal-readout {tr}; pass the same --ordinal-readout")
    if a.dup_layers:
        if stack is None:
            raise SystemExit("--dup-layers needs the stack path")
        dup_layers(stack, a.dup_layers)
    lm_head = model.get_output_embeddings()
    lsel = None                                      # (speed) --letter-head selected: the letter rows of the head, cached once
    if a.letter_head == "selected":
        try:
            lsel = label_head(model, label_ids.to(lm_head.weight.device))
        except (TypeError, ValueError) as e:
            raise SystemExit(f"--letter-head selected: {e}") from None
        print(f"letter head: selected ({lsel.weight.shape[0]} rows of {lm_head.weight.shape[0]}, {lsel.weight.dtype}; multiplier "
              f"{lsel.multiplier}, softcap {lsel.softcap})", flush=True)
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)
    aux, lt_aux, aux_S = None, lt, 0                 # (aux-vbll) off: none of it runs
    if a.aux_vbll:
        if lt_aux is None:                           # the head's code is the trainer's (as the gemmafast readout above)
            _aspec = importlib.util.spec_from_file_location("lt_for_aux", os.path.join(HERE, "claude_lmtune.py"))
            lt_aux = importlib.util.module_from_spec(_aspec)
            _aspec.loader.exec_module(lt_aux)
        if not hasattr(lt_aux, "AuxVBLL"):
            raise SystemExit("--aux-vbll needs the (aux-vbll) claude_lmtune.py beside claude_lmscore.py")
        if "aux_vbll" not in ck:
            raise SystemExit(f"--aux-vbll: {a.lora_ckpt} has no aux head (it was trained without claude_lmtune.py --aux-vbll)")
        ent = ck["aux_vbll"]
        if list(ent.get("label_ids") or []) != label_ids.tolist():
            raise SystemExit("--aux-vbll: the checkpoint's letter tokens differ from this tokenizer's (another --model / "
                             "--plain-prompt?)")
        aux = lt_aux.AuxVBLL.from_checkpoint(ent, lm_head.weight.detach()[label_ids.to(lm_head.weight.device)]).to(dev).eval()
        if not 0 <= a.aux_vbll_head < len(aux.heads):
            raise SystemExit(f"--aux-vbll-head {a.aux_vbll_head}: the checkpoint has {len(aux.heads)} head(s)")
        aux_S = a.aux_vbll_samples or int(ent.get("samples", 64))
        print(f"aux-vbll: {len(aux.heads)} head(s) " + "; ".join(f"{g}: {lt_aux.aux_head_name(h)}" for g, h in enumerate(aux.heads))
              + f"; test_aux_* = head {a.aux_vbll_head}; {aux_S} posterior draws", flush=True)
    nh = None                                        # (p95) off: none of it runs
    if a.num_head:
        if lt_aux is None:                           # the head's code is the trainer's
            _nspec = importlib.util.spec_from_file_location("lt_for_num", os.path.join(HERE, "claude_lmtune.py"))
            lt_aux = importlib.util.module_from_spec(_nspec)
            _nspec.loader.exec_module(lt_aux)
        if not hasattr(lt_aux, "NumHead"):
            raise SystemExit("--num-head needs the (p95) claude_lmtune.py beside claude_lmscore.py")
        if "num_head" not in ck:
            raise SystemExit(f"--num-head: {a.lora_ckpt} has no numeric head (it was trained without claude_lmtune.py --num-head)")
        nh = lt_aux.NumHead.from_checkpoint(ck["num_head"]).to(dev).eval()
        if nh.H != lm_head.weight.shape[1]:
            raise SystemExit(f"--num-head: the head reads {nh.H}-dim states, this model's are {lm_head.weight.shape[1]}")
        print(f"num-head: {nh.C}-component {nh.family} mixture, dim {nh.D} (trained with letters {ck['num_head'].get('letters')}, "
              f"CRPS weight {ck['num_head'].get('crps')}); numeric units' test_p = the head's bin masses", flush=True)
    elif a.lora_ckpt and "num_head" in ck:
        print("NOTE: the --lora-ckpt carries a numeric head (claude_lmtune.py --num-head); without --num-head, numeric units are read "
              "through the letters", flush=True)
    if a.lora_ckpt and ck["config"].get("prior_offset") and not po_on:     # (p101)
        print(f"NOTE: the --lora-ckpt was trained with claude_lmtune.py --prior-offset {ck['config']['prior_offset']} (its letters are the "
              f"EVIDENCE); without --prior-offset they are read as the posterior", flush=True)
    lt_po = None                                     # (p101) the offsets' code is the trainer's
    if po_on:
        lt_po = lt or lt_aux
        if lt_po is None:
            _pspec = importlib.util.spec_from_file_location("lt_for_prior_offset", os.path.join(HERE, "claude_lmtune.py"))
            lt_po = importlib.util.module_from_spec(_pspec)
            _pspec.loader.exec_module(lt_po)
        if not hasattr(lt_po, "prior_offset_index"):
            raise SystemExit("--prior-offset needs the (p101) claude_lmtune.py beside claude_lmscore.py")

    units = list(cf.iter_units(a.data, [s for s in a.splits.split(",") if s], set(), None))
    for u, uid in zip(units, unit_uids(units)):   # (rowid) over the FULL enumeration, before any filter or sample
        u[UID_FIELD] = uid
    if a.whole_case:                             # group BEFORE any filter or sample: the listing shows the whole case
        print(f"whole-case grouping: {attach_case_questions(units)}", flush=True)
    if a.noul_defs:
        nn = [u for u in units if u["qtype"] == "noul"]
        print(f"noul definitions: {sum(1 for u in nn if noul_definitions(u.get('noul_criteria')))} of {len(nn)} noul "
              f"units have them (the rest show bare Yes / No)", flush=True)
    if a.configs:
        want = set(a.configs.split(","))
        units = [u for u in units if str(u["qid"]).split("/")[0] in want]
        missing = sorted(want - {str(u["qid"]).split("/")[0] for u in units})
        if missing:
            raise SystemExit(f"--configs names with no units: {missing}")
    if a.limit:
        units = units[: a.limit]
    if a.exclude_states:                         # as claude_think.py: drop units whose state occurs in the eval set
        import json as _json
        spec = importlib.util.spec_from_file_location("ov", os.path.join(os.path.dirname(HERE), "analysis", "claude_overlap.py"))
        ov = importlib.util.module_from_spec(spec); spec.loader.exec_module(ov)
        bad = {ov.norm(_json.loads(l)["state"]) for l in open(a.exclude_states)}
        units = [u for u in units if len(u["option_keys"]) <= len(LABELS) and ov.norm(u["state"]) not in bad]
    if a.sample and len(units) > a.sample:
        units = [units[i] for i in sorted(np.random.default_rng(a.seed + 7).choice(len(units), a.sample, replace=False))]
    rng = np.random.default_rng(a.seed)
    fwd = None
    if a.forward_sample:                         # (tfparity) which units get a forward pass; the order draws below are unchanged
        elig = [ui for ui, u in enumerate(units) if len(u["option_keys"]) <= len(LABELS)
                and (a.forward_filter == "all" or "/" not in str(u["qid"]))]
        pick = np.random.default_rng(a.seed + 11).choice(len(elig), min(a.forward_sample, len(elig)), replace=False)
        fwd = {elig[i] for i in pick}
        print(f"forward sample: {len(fwd)} of {len(elig)} {a.forward_filter} units scored; option orders drawn for all "
              f"{len(units)} units, as in the full run", flush=True)
    wc_stats = {}
    rows, skipped, skip_reason = plan_prompts(tok, units, rng, n_perm=a.n_perm, max_len=a.max_len, all_perms_max_k=a.all_perms_max_k,
                                              exclude_identity=a.exclude_identity, symmetrize=a.symmetrize, plain=a.plain_prompt,
                                              whole_case=a.whole_case, noul_defs=a.noul_defs, seed=a.seed, chat=chat, fwd=fwd,
                                              stats=wc_stats)
    print(f"units {len(units)}, prompts {len(rows)}, skipped {skipped}", flush=True)
    row_num = {}                                           # (p95) prompt row -> its unit's numeric axis spec (numeric units only)
    if nh is not None:
        nidx = lt_aux.numeric_index(a.data, [s for s in a.splits.split(",") if s])
        for r, (ui, order, _) in enumerate(rows):
            u = units[ui]
            spec = nidx.get((u["record_id"], u["qid"]))
            if spec is not None and u["qtype"] == "score" and len(u["option_keys"]) == spec["K"]:
                row_num[r] = spec
        print(f"num-head: {len({rows[r][0] for r in row_num})} numeric units ({len(row_num)} prompts) of {len(units)}", flush=True)
    unit_po, row_po = {}, {}                               # (p101) unit -> log prior (stored order); prompt row -> letter order
    if po_on:
        try:
            po_key = lt_po.prior_offset_key(a.prior_offset)
        except ValueError as e:
            raise SystemExit(str(e)) from None
        pidx = lt_po.prior_offset_index(a.data, [s for s in a.splits.split(",") if s], po_key) if po_key else {}
        pfile = {}
        if a.prior_offset_file:
            with open(a.prior_offset_file, encoding="utf-8") as f:
                txt = f.read()
            try:
                obj = json.loads(txt)
                pfile = {str(k): v for k, v in obj.items()} if isinstance(obj, dict) and "qid" not in obj else \
                    {str(obj["qid"]): obj["log_prior_bins"]}
            except ValueError:
                for ln in txt.splitlines():
                    if ln.strip():
                        o = json.loads(ln)
                        pfile[str(o["qid"])] = o["log_prior_bins"]
        n_file = 0
        for ui, u in enumerate(units):
            val = pfile.get(str(u["qid"]))
            try:
                if val is not None:
                    unit_po[ui] = lt_po.offsets_for_keys(val, list(u["option_keys"]))
                    n_file += 1
                elif (u["record_id"], u["qid"]) in pidx:
                    unit_po[ui] = pidx[(u["record_id"], u["qid"])]
            except ValueError as e:
                raise SystemExit(f"--prior-offset: {u['qid']}: {e}") from None
            if ui in unit_po and len(unit_po[ui]) != len(u["option_keys"]):
                raise SystemExit(f"--prior-offset: {u['qid']}: {len(unit_po[ui])} offsets for {len(u['option_keys'])} options")
        for r, (ui, order, _) in enumerate(rows):
            if ui in unit_po:
                row_po[r] = np.asarray(unit_po[ui], dtype=np.float64)[np.asarray(order, dtype=np.int64)]
        print(f"prior-offset: {len({rows[r][0] for r in row_po})} of {len({ui for ui, _, _ in rows})} scored units ({len(row_po)} prompts) "
              f"read as softmax(log pi + letters): {n_file} from {a.prior_offset_file or 'no file'}, the rest from "
              f"{'meta.' + po_key if po_key else 'nothing else'}", flush=True)
    if a.whole_case:
        # case_criteria_cut / case_questions_dropped / case_all_dropped = the --max-len overflow paths, per prompt
        print(f"whole-case prompts: {dict(sorted(wc_stats.items()))}", flush=True)
        ex = next((r for r in rows if len(units[r[0]]["case_questions"]) > 1), None)
        if ex:
            print("example whole-case prompt:", repr(ex[2][-3000:]), flush=True)
    print("example prompt tail:", repr(rows[0][2][-300:]), flush=True)

    # Length-sorted batches under a token budget. Only the LAST position goes through the LM head.
    lens = [len(tok(r[2], add_special_tokens=False)["input_ids"]) for r in rows]
    order_idx = np.argsort(lens)
    logp = {}                                              # row index -> log p over its K options
    if a.save_hidden and a.exact_path:
        raise SystemExit("--save-hidden needs the stack path (it reads the hidden state the head is applied to)")
    HID = None
    aux_rows = {}                                          # (aux-vbll) row index -> (p [G, k], epi [G], alea [G], pvar [G], lvar [G])
    num_rows = {}                                          # (p95) row index -> (bin masses [k] level order, logpi, mu, s [C])
    ev_rows = {}                                           # (p101) row index -> raw letter logits [k] (letter order), float64
    i, done, t1 = 0, 0, time.time()
    with torch.no_grad():
        while i < len(order_idx):
            L = lens[order_idx[min(i, len(order_idx) - 1)]]
            j = i
            if a.exact_batch == "pack":          # (gemmafast) consecutive length-sorted prompts up to --exact-tokens
                tot = 0
                while j < len(order_idx) and (j == i or tot + lens[order_idx[j]] <= a.exact_tokens):
                    tot += lens[order_idx[j]]
                    j += 1
            while a.exact_batch != "pack" and j < len(order_idx) and (j - i + 1) * lens[order_idx[j]] <= a.tokens_per_batch \
                    and (not a.exact_path or a.exact_batch == "pad" or lens[order_idx[j]] == L):
                j += 1
            j = max(j, i + 1)
            idx = order_idx[i:j]
            enc = tok([rows[r][2] for r in idx], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(dev)
            caps = [] if (aux is not None or nh is not None) else None   # (aux-vbll / p95) each prompt's decision state, same forward
            if lt is not None:                 # (gemmafast) letter logits [B, 52] from the trainer's code
                lab_all = label_ids.to(dev)
                if a.exact_batch != "off":
                    z = lt.exact_letter_logits(model, tok, [rows[r][2] for r in idx], lab_all, dev, a.exact_batch, a.exact_head,
                                               **({"keep_h": caps} if caps is not None else {}))
                else:                          # equal lengths, no padding, fp32 head
                    hl = lt.inner_model(model)(**enc, use_cache=False).last_hidden_state[:, -1]
                    if caps is not None:
                        caps.append(hl)
                    z = lt.letter_head_fp32(model, hl, lab_all)
            elif a.exact_path:                 # equal lengths: no padding; the model's own forward, last position only
                with (lt_aux.head_input(model, caps, lambda x: x[:, -1]) if caps is not None else contextlib.nullcontext()):
                    z = model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1].float()
            else:
                h = stack(**enc, use_cache=False).last_hidden_state
                last = enc["attention_mask"].sum(1) - 1
                hl = h[torch.arange(len(idx), device=dev), last]
                if caps is not None:
                    caps.append(hl)
                z = cf.head_logits(model, hl) if lsel is None else lsel(hl)               # softcap-aware (Gemma); (speed) letters
                if a.save_hidden:
                    if HID is None:
                        HID = np.zeros((len(rows), hl.shape[1]), dtype=np.float16)
                    HID[np.asarray(idx)] = hl.float().cpu().numpy().astype(np.float16)
            if aux is not None:                # (aux-vbll) the head on the same decision states and the same letter logits
                kb = [len(rows[r][1]) for r in idx]
                ell = z[:, : max(kb)] if lt is not None or lsel is not None else z[:, label_ids[: max(kb)].to(dev)]
                hb = torch.cat(caps, 0)
                if hb.shape[0] != len(idx):
                    raise SystemExit(f"--aux-vbll: {hb.shape[0]} decision states for {len(idx)} prompts")
                pa, ea, aa, pv, lv = (x.float().cpu().numpy() for x in aux.predict(hb, ell, torch.tensor(kb, device=dev), aux_S))
                for b, r in enumerate(idx):
                    aux_rows[int(r)] = (pa[:, b, : kb[b]], ea[:, b], aa[:, b], pv[:, b], lv[:, b])
            if nh is not None:                 # (p95) the numeric head on the same decision states, numeric prompts only
                hb = torch.cat(caps, 0)
                if hb.shape[0] != len(idx):
                    raise SystemExit(f"--num-head: {hb.shape[0]} decision states for {len(idx)} prompts")
                sel = [b for b, r in enumerate(idx) if int(r) in row_num]
                if sel:
                    ntf, nu, nkb, _ = lt_aux.num_batch([row_num[int(idx[b])] for b in sel], dev)
                    nlp, nmu, ns = nh(hb[torch.tensor(sel, device=hb.device)], ntf)
                    nlm, nvalid = lt_aux.num_log_masses(nlp, nmu, ns, nu, nkb, nh.family)
                    npm = nlm.exp().masked_fill(~nvalid, 0.0).cpu().numpy()
                    nlp, nmu, ns = (x.cpu().numpy() for x in (nlp, nmu, ns))
                    for q_, b in enumerate(sel):         # (not j: the batch loop's end index)
                        num_rows[int(idx[b])] = (npm[q_, : int(nkb[q_])], nlp[q_], nmu[q_], ns[q_])
            for b, r in enumerate(idx):
                k = len(rows[r][1])
                zb = z[b, :k] if lt is not None or lsel is not None else z[b, label_ids[:k].to(dev)]   # (gemmafast / speed) letters
                if int(r) in row_po:           # (p101) the evidence + the log prior (letter order), in float64
                    ev_rows[int(r)] = zb.double().cpu().numpy()
                    logp[int(r)] = torch.log_softmax(torch.from_numpy(ev_rows[int(r)] + row_po[int(r)]), -1).numpy()
                    continue
                logp[int(r)] = torch.log_softmax(zb, -1).cpu().numpy()
            done += len(idx)
            i = j
            if done % 512 < len(idx):
                el = time.time() - t1
                print(f"  {done}/{len(rows)} prompts  {done / el:.1f}/s  eta {(len(rows) - done) / max(done / el, 1e-9) / 60:.1f} min  L={L}", flush=True)

    if a.ordinal_readout != "none":
        print(f"ordinal readout {a.ordinal_readout}: {apply_ordinal_readout(rows, logp, units, a.ordinal_readout)} "
              f"score prompts", flush=True)
    # Map every permuted distribution back to the original option order and average.
    by_unit, rows_of = {}, {}
    for r, (ui, order, _) in enumerate(rows):
        p = np.zeros(len(order))
        p[np.asarray(order)] = np.exp(logp[r])
        by_unit.setdefault(ui, []).append(p)
        rows_of.setdefault(ui, []).append(r)
    letter_by_unit = None
    if nh is not None:                                       # (p95) numeric units: the head's per-prompt bin masses (level order)
        letter_by_unit = {ui: list(ps) for ui, ps in by_unit.items()}
        for ui, rs in rows_of.items():
            if all(r in num_rows for r in rs):
                by_unit[ui] = [np.asarray(num_rows[r][0], dtype=np.float64) for r in rs]
    keep = sorted(by_unit)
    kmax = max(len(units[ui]["option_keys"]) for ui in keep)
    n = len(keep)
    P = np.zeros((n, kmax)); T = np.zeros((n, kmax))
    ks = np.zeros(n, dtype=np.int64); soft = np.zeros(n, dtype=bool)
    tot = np.zeros(n); mean_ent = np.zeros(n); qid, src = [], []
    for row, ui in enumerate(keep):
        u, ps = units[ui], np.stack(by_unit[ui])
        k = ps.shape[1]
        pm = ps.mean(0)
        v = np.asarray(u["target"], dtype=np.float64)
        P[row, :k], T[row, :k] = pm, v / max(v.sum(), 1e-12)
        ks[row], soft[row] = k, u["label_kind"] == "soft"
        tot[row] = -(pm * np.log(np.clip(pm, 1e-12, 1))).sum()
        mean_ent[row] = -(ps * np.log(np.clip(ps, 1e-12, 1))).sum(1).mean()
        qid.append(str(u["qid"])); src.append(str(u["source"]))
    extra = {}
    if a.save_rows:
        pos = {ui: row for row, ui in enumerate(keep)}
        ro = np.full((len(rows), kmax), -1, dtype=np.int64); rl = np.full((len(rows), kmax), -1e9)
        for r, (ui, order, _) in enumerate(rows):
            ro[r, : len(order)] = order; rl[r, : len(order)] = logp[r]
        extra = {"row_unit": np.array([pos[ui] for ui, _, _ in rows]), "row_order": ro, "row_logp": rl}
    if a.save_hidden:
        pos = {ui: row for row, ui in enumerate(keep)}
        np.save(a.save_hidden, HID)
        np.savez_compressed(a.save_hidden.replace(".npy", "_rows.npz"), row_unit=np.array([pos[ui] for ui, _, _ in rows]),
                            row_order=np.array([np.pad(np.asarray(o), (0, kmax - len(o)), constant_values=-1) for _, o, _ in rows]),
                            row_logp=np.array([np.pad(logp[r], (0, kmax - len(logp[r])), constant_values=-1e9) for r in range(len(rows))]),
                            row_label_ids=label_ids.numpy())
        print(f"saved {HID.shape} hidden states to {a.save_hidden}", flush=True)
    if a.ordinal_readout != "none":                          # (49) only when on: a default file's keys are unchanged
        extra["ordinal_readout"] = np.array(a.ordinal_readout)
    if a.exact_batch != "off":                               # (gemmafast) likewise
        extra["exact_batch"] = np.array(a.exact_batch)
    if a.exact_head != "model":
        extra["exact_head"] = np.array(a.exact_head)
    if a.letter_head != "full":                              # (speed) likewise
        extra["letter_head"] = np.array(a.letter_head)
    if chat:                                                 # (66) likewise
        extra["chat_preset"] = np.array(json.dumps(chat))
    if loader != "causal-lm":
        extra["loader"] = np.array(loader)
    if fwd is not None:                                      # the row each scored unit has in the full run's file
        full_row = {ui: r for r, ui in enumerate(i for i, u in enumerate(units) if len(u["option_keys"]) <= len(LABELS))}
        extra["test_full_row"] = np.array([full_row[ui] for ui in keep], dtype=np.int64)
    views = {}
    if nh is not None:                                       # (p95) only when on: a default file's keys are unchanged
        C = nh.C
        letP, isnum, tfa = np.zeros((n, kmax)), np.zeros(n, dtype=bool), np.full(n, -1, dtype=np.int8)
        NLP, NMU, NS = (np.full((n, C), np.nan) for _ in range(3))
        NLO, NHI = np.full(n, np.nan), np.full(n, np.nan)
        for row, ui in enumerate(keep):
            lp = np.stack(letter_by_unit[ui]).mean(0)
            letP[row, : len(lp)] = lp
            r0 = rows_of[ui][0]
            if all(r in num_rows for r in rows_of[ui]):
                spec = row_num[r0]
                isnum[row], tfa[row], NLO[row], NHI[row] = True, spec["tfi"], spec["lo"], spec["hi"]
                NLP[row], NMU[row], NS[row] = num_rows[r0][1], num_rows[r0][2], num_rows[r0][3]
        stats = {k: np.full(n, np.nan) for k in ("mean", "sd", "q05", "q50", "q95")}
        if isnum.any():
            sm = lt_aux.num_summaries(NLP[isnum], NMU[isnum], NS[isnum], nh.family,
                                      [lt_aux.NUM_TRANSFORMS[i] for i in tfa[isnum]], NLO[isnum], NHI[isnum])
            stats["mean"][isnum], stats["sd"][isnum] = sm["mean"], sm["sd"]
            for j, k in enumerate(("q05", "q50", "q95")):
                stats[k][isnum] = sm["q"][:, j]
        extra.update(test_num=isnum, test_letter_p=letP, test_num_logpi=NLP, test_num_mu=NMU, test_num_scale=NS,
                     test_num_transform=tfa, test_num_lo=NLO, test_num_hi=NHI, **{f"test_num_{k}": v for k, v in stats.items()},
                     num_head=np.array(json.dumps({k: v for k, v in ck["num_head"].items() if k != "state"})))
        if isnum.any():
            views["letters"] = dict(test_p=letP, test_t=T, test_ks=ks, test_soft=soft, test_qid=np.array(qid),
                                    test_source=np.array(src))
        print(f"num-head: {int(isnum.sum())} numeric units read by the head; pooled acc there head "
              f"{float((P[isnum].argmax(1) == T[isnum].argmax(1)).mean()) if isnum.any() else float('nan'):.4f} / letters "
              f"{float((letP[isnum].argmax(1) == T[isnum].argmax(1)).mean()) if isnum.any() else float('nan'):.4f}", flush=True)
    if aux is not None:                                      # (aux-vbll) only when on: a default file's keys are unchanged
        G, g = len(aux.heads), a.aux_vbll_head
        AP, AE, AA, AV, AL = np.zeros((G, n, kmax)), np.zeros((G, n)), np.zeros((G, n)), np.zeros((G, n)), np.zeros((G, n))
        for row, ui in enumerate(keep):                      # each prompt back to the unit's option order, then the mean
            k = len(units[ui]["option_keys"])
            for r in rows_of[ui]:
                order = np.asarray(rows[r][1])
                AP[:, row, order] += aux_rows[r][0]
                for arr, j in ((AE, 1), (AA, 2), (AV, 3), (AL, 4)):
                    arr[:, row] += aux_rows[r][j]
            m = len(rows_of[ui])
            AP[:, row, :k] /= m
            for arr in (AE, AA, AV, AL):
                arr[:, row] /= m
        ap_ = AP[g]
        a_tot = -(ap_ * np.log(np.clip(ap_, 1e-12, 1))).sum(1)
        extra.update(test_aux_p=ap_, test_aux_epi=AE[g], test_aux_alea=AA[g], test_aux_pvar=AV[g], test_aux_lvar=AL[g],
                     test_aux_tot=a_tot, test_aux_mx=ap_.max(1), aux_vbll_heads=np.array(json.dumps(aux.heads)),
                     aux_vbll_head=np.array(g), aux_vbll_samples=np.array(aux_S))
        if G > 1:
            extra.update(test_aux_grid_p=AP, test_aux_grid_epi=AE, test_aux_grid_alea=AA, test_aux_grid_pvar=AV,
                         test_aux_grid_lvar=AL)
        base = dict(test_t=T, test_ks=ks, test_soft=soft, test_qid=np.array(qid), test_source=np.array(src))
        views["auxvbll"] = dict(test_p=ap_, test_tot=a_tot, test_pred_ent=AA[g], test_epi=AE[g], test_mx=ap_.max(1),
                                test_aux_pvar=AV[g], test_aux_lvar=AL[g], aux_vbll_heads=np.array(json.dumps(aux.heads)),
                                aux_vbll_head=np.array(g), **base)
        acc_aux = float((ap_.argmax(1) == T.argmax(1)).mean())
        print(f"aux-vbll head {g}: pooled acc {acc_aux:.4f}; mean epistemic {AE[g].mean():.4e}, aleatoric {AA[g].mean():.4f}, "
              f"logit-var {AL[g].mean():.4e}", flush=True)
        if a.aux_combine != "off":
            cp = lt_aux.combine_readouts(P, ap_, a.aux_combine, a.aux_combine_weight)
            c_tot = -(cp * np.log(np.clip(cp, 1e-12, 1))).sum(1)
            extra.update(test_combo_p=cp, aux_combine=np.array(a.aux_combine), aux_combine_weight=np.array(a.aux_combine_weight))
            views["auxcombo"] = dict(test_p=cp, test_tot=c_tot, test_mx=cp.max(1), aux_combine=np.array(a.aux_combine),
                                     aux_combine_weight=np.array(a.aux_combine_weight), **base)
            print(f"aux-combine {a.aux_combine} w={a.aux_combine_weight:g}: pooled acc "
                  f"{float((cp.argmax(1) == T.argmax(1)).mean()):.4f}", flush=True)
    if po_on:                                                # (p101) only when on: a default file's keys are unchanged
        EV, LP, has = np.full((n, kmax), np.nan), np.full((n, kmax), np.nan), np.zeros(n, dtype=bool)
        for row, ui in enumerate(keep):
            if ui not in unit_po or not all(r in ev_rows for r in rows_of[ui]):
                continue
            k = len(units[ui]["option_keys"])
            acc = np.zeros(k)
            for r in rows_of[ui]:                            # each prompt's letters back to the stored option (bin) order
                acc[np.asarray(rows[r][1], dtype=np.int64)] += ev_rows[r]
            EV[row, :k] = acc / len(rows_of[ui])
            LP[row, :k] = unit_po[ui]
            has[row] = True
        extra.update(test_evidence_logits=EV, test_log_prior=LP, test_prior_offset=has,
                     prior_offset=np.array(json.dumps({"spec": a.prior_offset, "file": a.prior_offset_file, "units": int(has.sum()),
                                                       "prompts": len(row_po)})))
        print(f"prior-offset: {int(has.sum())} units with test_evidence_logits; pooled acc there "
              f"{float((P[has].argmax(1) == T[has].argmax(1)).mean()) if has.any() else float('nan'):.4f}", flush=True)
    ident = {                                                # (rowid) immutable row identity (module docstring)
        "test_rid": np.array([str(units[ui]["record_id"]) for ui in keep], dtype=str),
        "test_uid": np.array([units[ui][UID_FIELD] for ui in keep], dtype=str),
        "test_input_sha": np.array([prompt_sha([rows[r][2] for r in rows_of[ui]]) for ui in keep], dtype=str),
        "test_opt_sha": np.array([option_sha(units[ui]["option_keys"]) for ui in keep], dtype=str),
    }
    ident_top = {"skipped_units": np.array(json.dumps([[units[ui][UID_FIELD], why] for ui, why in sorted(skip_reason.items())
                                                       if ui not in by_unit], ensure_ascii=False)),
                 "rowid_scheme": np.array(ROWID_SCHEME)}
    np.savez_compressed(a.out, test_p=P, test_t=T, test_ks=ks, test_soft=soft,
                        test_qid=np.array(qid), test_source=np.array(src),
                        test_tot=tot, test_pred_ent=mean_ent, test_mx=P.max(1),
                        test_perm_mi=tot - mean_ent, **extra, **ident, **ident_top)
    for tag, arrs in views.items():                          # (aux-vbll) drop-in files: test_p = the head's / the combination
        vout = (a.out[:-4] if a.out.endswith(".npz") else a.out) + f"_{tag}.npz"
        np.savez_compressed(vout, **arrs, **ident)
        print(f"wrote view {vout}", flush=True)
    acc = float((P.argmax(1) == T.argmax(1)).mean())
    print(f"wrote {a.out}: {n} units, pooled acc {acc:.4f} (argmax; tie-naive), "
          f"{time.time() - t0:.0f}s total", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
