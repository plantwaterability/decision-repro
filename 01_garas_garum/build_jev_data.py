#!/usr/bin/env python3
"""
build_jev_data.py -- pull public Hugging Face datasets and convert them into
decision-model training records (shared state + typed questions + label distributions).

Record schema (one JSON object per line):
{
  "id":        "<source>-<hash>",
  "source":    "banking77",
  "license":   "CC-BY-4.0 (verify)",
  "state":     "text every question reads",
  "questions": {qid: {"type": "choice", "instructions": str, "criteria": {key: description}}
                qid: {"type": "noul",   "instructions": str}
                qid: {"type": "score",  "instructions": str, "levels": [str, ...]}},
  "labels":    {qid: {option_key: probability}},   # one-hot or soft; always sums to 1
  "label_kind":{qid: "hard" | "soft" | "noisy_hard"},
  "meta":      {...}
}
Option keys for noul are "yes"/"no"; for score they are the level strings.
Dict order of "criteria" is the option order shown to the model (it is shuffled).

Usage:
  pip install datasets tasksource
  python build_jev_data.py --out jev_data --max-per-source 5000
  python build_jev_data.py --sources xlam --hf-token hf_...        # gated: accept terms first
  python build_jev_data.py --sources tasksource --tasksource-tasks glue/rte,glue/sst2
  python build_jev_data.py --self-test                              # no network needed

Output: <out>/{train,val,test}/<source>.jsonl and <out>/manifest.json.
Splits are assigned by a hash of the state text, so reshuffled variants of the same
example never leak across splits.
Add --holdout-source to force whole sources into an extra <out>/ood/ split: the hash
split only controls leakage, so without it every source appears in every split and
domain-level OOD evaluation of the confidence head is impossible.
"""
import argparse
import copy
import hashlib
import itertools
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

MAX_OPTIONS = 255

# Best-known licenses. VERIFY each against the current dataset card before use.
LICENSES = {
    "banking77": "CC-BY-4.0 (verify)",
    "clinc": "CC-BY-3.0 (verify)",
    "go_emotions": "Apache-2.0 (verify)",
    "mhs": "CC-BY-4.0 (verify)",
    "helpsteer2": "CC-BY-4.0",
    "xlam": "CC-BY-4.0, gated research release (verify terms)",
    "gsm8k": "MIT (verify)",
    "ms_marco": "Non-commercial research terms (verify)",
    "tasksource": "Varies per task (check each source dataset)",
}


# --------------------------------------------------------------------------- helpers
def sha(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def norm_text(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def humanize(name):
    s = str(name).replace("_", " ").replace("-", " ").strip()
    return s[:1].upper() + s[1:]


def one_hot(key, keys):
    return {k: (1.0 if k == key else 0.0) for k in keys}


def normalize(d):
    total = sum(d.values())
    if total <= 0:
        raise ValueError("empty distribution")
    return {k: v / total for k, v in d.items()}


def noul_label(p):
    p = float(p)
    return {"yes": p, "no": 1.0 - p}


def shuffled(criteria, rng):
    items = list(criteria.items())
    rng.shuffle(items)
    return dict(items)


def choice_q(instructions, criteria):
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def noul_q(instructions):
    return {"type": "noul", "instructions": instructions}


def score_q(instructions, levels):
    return {"type": "score", "instructions": instructions, "levels": list(levels)}


def option_keys(q):
    if q["type"] == "choice":
        return list(q["criteria"])
    if q["type"] == "noul":
        return ["yes", "no"]
    return list(q["levels"])


def license_for(source):
    return LICENSES.get(source.split("/")[0], "unknown (verify)")


def record(source, state, questions, labels, kinds, meta=None):
    sig = json.dumps({k: option_keys(v) for k, v in questions.items()}, sort_keys=True)
    return {
        "id": f"{source}-{sha(state + sig)}",
        "source": source,
        "license": license_for(source),
        "state": state,
        "questions": questions,
        "labels": labels,
        "label_kind": kinds,
        "meta": meta or {},
    }


def validate(r):
    assert isinstance(r["state"], str) and r["state"].strip(), "empty state"
    assert r["questions"], "no questions"
    for qid, q in r["questions"].items():
        keys = option_keys(q)
        assert 2 <= len(keys) <= MAX_OPTIONS, f"{qid}: {len(keys)} options"
        assert len(set(keys)) == len(keys), f"{qid}: duplicate option keys"
        assert q["instructions"].strip(), f"{qid}: empty instructions"
        if q["type"] == "choice":
            assert all(str(v).strip() for v in q["criteria"].values()), f"{qid}: empty criterion"
        lab = r["labels"][qid]
        assert set(lab) == set(keys), f"{qid}: label keys do not match options"
        assert all(0.0 <= v <= 1.0 for v in lab.values()), f"{qid}: probability out of range"
        assert abs(sum(lab.values()) - 1.0) < 1e-6, f"{qid}: labels sum to {sum(lab.values())}"
        assert r["label_kind"].get(qid) in ("hard", "soft", "noisy_hard"), f"{qid}: bad label_kind"


def reshuffle(r, rng):
    """A variant with every choice question's option order reshuffled."""
    v = copy.deepcopy(r)
    for q in v["questions"].values():
        if q["type"] == "choice":
            q["criteria"] = shuffled(q["criteria"], rng)
    sig = json.dumps({k: option_keys(q) for k, q in v["questions"].items()}, sort_keys=True)
    v["id"] = f"{v['source']}-{sha(v['state'] + sig)}"
    v["meta"] = {**v["meta"], "variant_of": r["id"]}
    return v


# --------------------------------------------------------------------------- converters
INTENT_INSTR = [
    "Which intent best matches this customer message?",
    "What is the customer asking for?",
    "Classify the request into one of the listed intents.",
    "Which category does this message belong to?",
]
MC_INSTR = [
    "Which option is correct?",
    "Select the best answer.",
    "Which of the options is right?",
]
TOOL_INSTR = [
    "Which tool should be called first to handle this request?",
    "Which of the available tools best fits this request?",
    "Route this request to the most appropriate tool.",
]
EMO_INSTR = [
    "Does this comment express {e}?",
    "Is {e} one of the emotions conveyed in this comment?",
]


def convert_intents(rows, source, text_key, label_key, names, rng,
                    oos_name=None, k_range=(4, 16), p_drop_true=0.15):
    """Intent datasets -> choice over a random subset of intents plus 'none of these'.
    The true intent is sometimes withheld (and out-of-scope rows never have it), so the
    model learns that 'none of these' is a real answer, not a formality."""
    inscope = [n for n in names if n != oos_name]
    for row in rows:
        text = str(row[text_key]).strip()
        if not text:
            continue
        true = names[row[label_key]]
        is_oos = true == oos_name
        pool = [n for n in inscope if n != true]
        distractors = rng.sample(pool, min(rng.randint(*k_range), len(pool)))
        drop = is_oos or rng.random() < p_drop_true
        opts = distractors if drop else distractors + [true]
        criteria = {n: humanize(n) for n in opts}
        criteria["none_of_these"] = "None of the listed intents fits this message"
        criteria = shuffled(criteria, rng)
        gold = "none_of_these" if drop else true
        yield record(
            source, text,
            {"intent": choice_q(rng.choice(INTENT_INSTR), criteria)},
            {"intent": one_hot(gold, list(criteria))},
            {"intent": "hard"},
            {"true_intent": true, "true_withheld": drop and not is_oos, "out_of_scope": is_oos},
        )


def convert_ts_classification(rows, task, names, rng):
    """tasksource Classification template: sentence1, sentence2, labels (ClassLabel)."""
    if not (2 <= len(names) <= MAX_OPTIONS):
        return
    src = f"tasksource/{task}"
    for row in rows:
        s1 = str(row.get("sentence1") or "").strip()
        s2 = str(row.get("sentence2") or "").strip()
        lab = row.get("labels")
        if not s1 or lab is None or not (0 <= int(lab) < len(names)):
            continue
        state = s1 if not s2 else f"Text A:\n{s1}\n\nText B:\n{s2}"
        criteria = shuffled({f"l{i}": humanize(n) for i, n in enumerate(names)}, rng)
        instr = f"Task: {task}. Which label applies" + (" to Text B given Text A?" if s2 else " to this text?")
        yield record(src, state, {"label": choice_q(instr, criteria)},
                     {"label": one_hot(f"l{int(lab)}", list(criteria))}, {"label": "hard"}, {"task": task})


def convert_ts_multiple_choice(rows, task, rng):
    """tasksource MultipleChoice template: inputs, choice0..choiceN, labels (int)."""
    src = f"tasksource/{task}"
    for row in rows:
        ctx = str(row.get("inputs") or "").strip()
        lab = row.get("labels")
        keys = sorted((k for k in row if re.fullmatch(r"choice\d+", k)), key=lambda k: int(k[6:]))
        choices = {k: str(row[k]).strip() for k in keys if row[k] is not None and str(row[k]).strip()}
        gold = f"choice{lab}" if lab is not None else None
        if not ctx or len(choices) < 2 or gold not in choices or len(choices) > MAX_OPTIONS:
            continue
        criteria = shuffled(choices, rng)
        yield record(src, ctx, {"answer": choice_q(rng.choice(MC_INSTR), criteria)},
                     {"answer": one_hot(gold, list(criteria))}, {"answer": "hard"}, {"task": task})


GO_EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
]


def convert_go_emotions(rows, rng, n_questions=(3, 6), min_raters=2):
    """go_emotions 'raw' config (one row per rater) -> several noul questions per comment,
    each labelled with the fraction of raters who ticked that emotion (soft label).
    Mixes emotions some rater chose with emotions nobody chose."""
    by_id, text = defaultdict(list), {}
    for r in rows:
        if r.get("example_very_unclear"):
            continue
        by_id[r["id"]].append(r)
        text[r["id"]] = r["text"]
    for cid, rs in by_id.items():
        if len(rs) < min_raters or not str(text[cid]).strip():
            continue
        p = {e: sum(int(r[e]) for r in rs) / len(rs) for e in GO_EMOTIONS}
        present = [e for e in GO_EMOTIONS if p[e] > 0]
        absent = [e for e in GO_EMOTIONS if p[e] == 0]
        n = rng.randint(*n_questions)
        chosen = rng.sample(present, min(len(present), max(1, n // 2)))
        chosen += rng.sample(absent, min(len(absent), n - len(chosen)))
        qs, labs, kinds = {}, {}, {}
        for e in chosen:
            qid = f"emotion_{e}"
            instr = ("Is this comment emotionally neutral?" if e == "neutral"
                     else rng.choice(EMO_INSTR).format(e=e))
            qs[qid], labs[qid], kinds[qid] = noul_q(instr), noul_label(p[e]), "soft"
        yield record("go_emotions", str(text[cid]).strip(), qs, labs, kinds, {"n_raters": len(rs)})


# VERIFY against the Measuring Hate Speech codebook on the dataset card before training:
# the value coding and level wording below are assumptions.
MHS_HATESPEECH_LEVELS = ["no", "unclear", "yes"]  # assumed coding of `hatespeech` 0/1/2
MHS_LIKERT_LEVELS = ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"]  # assumed 0..4
MHS_LIKERT_ITEMS = {
    "insult": "Is this comment insulting toward its target?",
    "humiliate": "Is this comment humiliating toward its target?",
    "violence": "Does this comment call for or endorse violence?",
}


def convert_mhs(rows, rng, min_raters=2):
    """measuring-hate-speech (one row per annotator) -> soft score questions per comment."""
    by_id, text = defaultdict(list), {}
    for r in rows:
        by_id[r["comment_id"]].append(r)
        text[r["comment_id"]] = r["text"]
    for cid, rs in by_id.items():
        if len(rs) < min_raters or not str(text[cid]).strip():
            continue
        qs, labs, kinds = {}, {}, {}
        hs = Counter(int(r["hatespeech"]) for r in rs if r.get("hatespeech") is not None)
        if hs and set(hs) <= set(range(len(MHS_HATESPEECH_LEVELS))):
            qs["hatespeech"] = score_q("Does this comment contain hate speech?", MHS_HATESPEECH_LEVELS)
            labs["hatespeech"] = normalize({lv: hs.get(i, 0) for i, lv in enumerate(MHS_HATESPEECH_LEVELS)})
            kinds["hatespeech"] = "soft"
        for item, instr in MHS_LIKERT_ITEMS.items():
            c = Counter(int(r[item]) for r in rs if r.get(item) is not None)
            if not c or not set(c) <= set(range(len(MHS_LIKERT_LEVELS))):
                continue
            qid = f"{item}_agreement"
            qs[qid] = score_q(f"{instr} Rate your agreement.", MHS_LIKERT_LEVELS)
            labs[qid] = normalize({lv: c.get(i, 0) for i, lv in enumerate(MHS_LIKERT_LEVELS)})
            kinds[qid] = "soft"
        if qs:
            yield record("mhs", str(text[cid]).strip(), qs, labs, kinds, {"n_raters": len(rs)})


HS2_ATTRS = {
    "helpfulness": "How helpful is the response to the prompt overall?",
    "correctness": "Does the response include all pertinent facts without errors?",
    "coherence": "How consistent and clear is the response?",
    "complexity": "How much intellectual depth or domain expertise does writing this response require?",
    "verbosity": "How much detail does the response include relative to what the prompt asks for?",
}
HS2_LEVELS = ["0 - very low", "1 - low", "2 - moderate", "3 - high", "4 - very high"]


def convert_helpsteer2(rows, rng, n_attrs=(2, 5)):
    """HelpSteer2 -> score questions (0-4) on a random subset of the five attributes."""
    for r in rows:
        prompt, resp = str(r["prompt"]).strip(), str(r["response"]).strip()
        if not prompt or not resp:
            continue
        state = f"User prompt:\n{prompt}\n\nAssistant response:\n{resp}"
        attrs = rng.sample(list(HS2_ATTRS), rng.randint(*n_attrs))
        qs, labs, kinds = {}, {}, {}
        for a in attrs:
            v = r.get(a)
            if v is None or not (0 <= int(v) < len(HS2_LEVELS)):
                continue
            qs[a] = score_q(HS2_ATTRS[a], HS2_LEVELS)
            labs[a] = one_hot(HS2_LEVELS[int(v)], HS2_LEVELS)
            kinds[a] = "hard"
        if qs:
            yield record("helpsteer2", state, qs, labs, kinds)


def convert_xlam(rows, rng, p_drop_gold=0.15):
    """xlam-function-calling-60k -> 'which tool first?' choice + 'multiple calls?' noul.
    Sometimes withholds the correct tool so 'none fits' has positive examples."""
    for r in rows:
        tools, answers = r["tools"], r["answers"]
        tools = json.loads(tools) if isinstance(tools, str) else tools
        answers = json.loads(answers) if isinstance(answers, str) else answers
        names = [t["name"] for t in tools]
        if not names or len(set(names)) != len(names) or len(names) + 1 > MAX_OPTIONS:
            continue
        gold_name = answers[0]["name"] if answers else None
        if gold_name is not None and gold_name not in names:
            continue
        drop = gold_name is not None and len(names) > 1 and rng.random() < p_drop_gold
        criteria = {
            f"t{i}": f"{t['name']}: {str(t.get('description', '')).strip() or 'no description'}"
            for i, t in enumerate(tools) if not (drop and t["name"] == gold_name)
        }
        criteria["none"] = "None of the listed tools fits; answer directly or ask for a different tool"
        criteria = shuffled(criteria, rng)
        gold = "none" if (gold_name is None or drop) else f"t{names.index(gold_name)}"
        qs = {
            "first_tool": choice_q(rng.choice(TOOL_INSTR), criteria),
            "multi_call": noul_q("Will fulfilling this request take more than one tool call?"),
        }
        labs = {
            "first_tool": one_hot(gold, list(criteria)),
            "multi_call": noul_label(1.0 if len(answers) > 1 else 0.0),
        }
        yield record("xlam", str(r["query"]).strip(), qs, labs,
                     {"first_tool": "hard", "multi_call": "hard"},
                     {"n_calls": len(answers), "gold_withheld": drop})


def gsm8k_distractors(gold, rng, n=3):
    cands = {gold + d for d in (1, -1, 2, -2, 10, -10)} | {gold * 2, gold * 10}
    if gold % 2 == 0:
        cands.add(gold // 2)
    if gold % 10 == 0 and gold:
        cands.add(gold // 10)
    cands.discard(gold)
    if gold >= 0:
        cands = {c for c in cands if c >= 0}
    return rng.sample(sorted(cands), n)


def convert_gsm8k(rows, rng):
    """GSM8K -> 4-option choice; distractors mimic off-by-one, x10 and halving mistakes."""
    for r in rows:
        tail = str(r["answer"]).split("####")[-1].strip().replace(",", "")
        try:
            gold = int(tail)
        except ValueError:
            continue
        opts = [gold] + gsm8k_distractors(gold, rng)
        criteria = shuffled({k: str(v) for k, v in zip("abcd", opts)}, rng)
        yield record("gsm8k", str(r["question"]).strip(),
                     {"answer": choice_q("What is the final numerical answer?", criteria)},
                     {"answer": one_hot("a", list(criteria))}, {"answer": "hard"})


def convert_ms_marco(rows, rng):
    """MS MARCO v2.1 -> noul relevance. Unselected passages are unjudged rather than
    confirmed irrelevant, so negatives are marked noisy_hard (down-weight them)."""
    for r in rows:
        q, p = str(r["query"]).strip(), r["passages"]
        pairs = list(zip(p["passage_text"], p["is_selected"]))
        sel = [t for t, s in pairs if s]
        uns = [t for t, s in pairs if not s]
        picks = ([(rng.choice(sel), 1.0)] if sel else []) + ([(rng.choice(uns), 0.0)] if uns else [])
        for text, y in picks:
            state = f"Query: {q}\n\nPassage: {str(text).strip()}"
            yield record("ms_marco", state,
                         {"answers_query": noul_q("Does the passage contain the answer to the query?")},
                         {"answers_query": noul_label(y)},
                         {"answers_query": "hard" if y == 1.0 else "noisy_hard"})


# --------------------------------------------------------------------------- loading
def hf_load(candidates, config=None, split="train", token=None):
    """Try each repo id, then its auto-converted Parquet branch (needed for datasets
    that still ship loading scripts, e.g. banking77, on recent `datasets` versions)."""
    from datasets import load_dataset
    errors = []
    for name in candidates:
        for kw in ({}, {"revision": "refs/convert/parquet"}):
            try:
                return load_dataset(name, config, split=split, token=token, **kw)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name} {kw or ''}: {type(e).__name__}: {str(e)[:160]}")
    raise RuntimeError("could not load any of:\n  " + "\n  ".join(errors))


def subsample(ds, n, rng):
    """Row-level subsample (oversampled 1.5x to survive filtering)."""
    k = min(len(ds), int(n * 1.5) + 1)
    return ds.shuffle(seed=rng.randint(0, 2**31 - 1)).select(range(k))


def class_names(ds, col):
    feat = ds.features.get(col)
    if feat is None or not hasattr(feat, "names"):
        raise RuntimeError(f"column {col!r} is not a ClassLabel; cannot recover label names")
    return feat.names


def label_names(ds, col, text_col):
    """ClassLabel names, else rebuilt from a per-row label-text column (Parquet copies drop the
    ClassLabel feature but often keep the text). Refuses a mapping that is not 0..n-1."""
    try:
        return class_names(ds, col)
    except RuntimeError:
        if text_col not in ds.column_names:
            raise
    names = {}
    for i, t in zip(ds[col], ds[text_col], strict=True):
        if names.setdefault(int(i), str(t)) != str(t):
            raise RuntimeError(f"label {i} maps to both {names[int(i)]!r} and {t!r}")
    if sorted(names) != list(range(len(names))):
        raise RuntimeError(f"labels in {col!r} are not 0..{len(names) - 1}")
    return [names[i] for i in range(len(names))]


def src_banking77(a, rng):
    # PolyAI/banking77 is still a loading script, which datasets >= 4 refuses, and it has no
    # auto-converted Parquet branch. mteb/banking77 is a Parquet copy with the same integer
    # labels (card_arrival = 11 in both) and a label_text column instead of a ClassLabel.
    ds = hf_load(["PolyAI/banking77", "mteb/banking77"], token=a.hf_token)
    yield from convert_intents(subsample(ds, a.max_per_source, rng), "banking77",
                               "text", "label", label_names(ds, "label", "label_text"), rng)


def src_clinc(a, rng):
    ds = hf_load(["clinc/clinc_oos", "clinc_oos"], config="plus", token=a.hf_token)
    yield from convert_intents(subsample(ds, a.max_per_source, rng), "clinc",
                               "text", "intent", class_names(ds, "intent"), rng, oos_name="oos")


def src_go_emotions(a, rng):
    ds = hf_load(["google-research-datasets/go_emotions"], config="raw", token=a.hf_token)
    yield from convert_go_emotions(ds, rng)  # needs all rater rows to group


def src_mhs(a, rng):
    ds = hf_load(["ucberkeley-dlab/measuring-hate-speech"], token=a.hf_token)
    yield from convert_mhs(ds, rng)


def src_helpsteer2(a, rng):
    ds = hf_load(["nvidia/HelpSteer2"], token=a.hf_token)
    yield from convert_helpsteer2(subsample(ds, a.max_per_source, rng), rng)


def src_xlam(a, rng):
    ds = hf_load(["Salesforce/xlam-function-calling-60k"], token=a.hf_token)
    yield from convert_xlam(subsample(ds, a.max_per_source, rng), rng)


def src_gsm8k(a, rng):
    ds = hf_load(["openai/gsm8k", "gsm8k"], config="main", token=a.hf_token)
    yield from convert_gsm8k(subsample(ds, a.max_per_source, rng), rng)


def src_ms_marco(a, rng):
    ds = hf_load(["microsoft/ms_marco", "ms_marco"], config="v2.1", token=a.hf_token)
    yield from convert_ms_marco(subsample(ds, a.max_per_source, rng), rng)


def src_tasksource(a, rng):
    from tasksource import load_task
    per_task = max(1, a.max_per_source // max(1, len(a.tasksource_tasks)))
    for task in a.tasksource_tasks:
        try:
            dd = load_task(task)
            ds = dd["train"] if "train" in dd else dd[list(dd)[0]]
            rows = subsample(ds, per_task, rng)
            if "choice0" in ds.column_names:
                gen = convert_ts_multiple_choice(rows, task, rng)
            elif "sentence1" in ds.column_names:
                gen = convert_ts_classification(rows, task, class_names(ds, "labels"), rng)
            else:
                print(f"  [tasksource] skip {task}: unsupported template {ds.column_names}")
                continue
            yield from itertools.islice(gen, per_task)
        except Exception as e:  # noqa: BLE001
            print(f"  [tasksource] {task} failed: {type(e).__name__}: {str(e)[:160]}")


SOURCES = {
    "banking77": src_banking77,
    "clinc": src_clinc,
    "go_emotions": src_go_emotions,
    "mhs": src_mhs,
    "helpsteer2": src_helpsteer2,
    "xlam": src_xlam,
    "gsm8k": src_gsm8k,
    "ms_marco": src_ms_marco,
    "tasksource": src_tasksource,
}
# ms_marco REMOVED from the default 2026-10-01: its terms are non-commercial research only, and the user ruled that nothing NC
# is trained on (the trainer also drops NC units). xlam gated; tasksource opt-in.
DEFAULT_SOURCES = "banking77,clinc,go_emotions,mhs,helpsteer2,gsm8k"


# --------------------------------------------------------------------------- writing
def split_of(state, val_pct=5, test_pct=5):
    b = int(sha(norm_text(state)), 16) % 100
    return "test" if b < test_pct else "val" if b < test_pct + val_pct else "train"


def split_for(source, state, holdout=()):
    """Split assignment, kept separate from `build` so it is testable.

    Two different jobs, deliberately layered:
      * `split_of` hashes the STATE, which controls leakage -- a reshuffled variant or a
        near-duplicate can never straddle the split line.
      * a source in `holdout` is forced to "ood", the only thing that creates DOMAIN SHIFT.
        Without it every source appears in every split, and the state-level novelty head
        has nothing to be wrong about.
    """
    return "ood" if source in holdout else split_of(state)


class Writer:
    def __init__(self, out):
        self.out, self.files = Path(out), {}

    def write(self, split, source, rec):
        key = (split, source)
        if key not in self.files:
            path = self.out / split / f"{source.replace('/', '__')}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.files[key] = path.open("w", encoding="utf-8")
        self.files[key].write(json.dumps(rec, ensure_ascii=False) + "\n")

    def close(self):
        for f in self.files.values():
            f.close()


def build(a):
    rng = random.Random(a.seed)
    exclude = set()
    holdout = {s.strip() for s in (a.holdout_source or "").split(",") if s.strip()}
    if holdout:
        print(f"holdout sources -> split 'ood' (never train/val/test): {sorted(holdout)}")
    if a.exclude_file:
        with open(a.exclude_file, encoding="utf-8") as f:
            exclude = {sha(norm_text(line)) for line in f if line.strip()}
        print(f"loaded {len(exclude)} exclusion hashes")

    writer, seen = Writer(a.out), set()
    manifest = {"seed": a.seed, "max_per_source": a.max_per_source, "variants": a.variants,
                "holdout_sources": sorted(holdout), "sources": {}}
    for name in a.sources:
        print(f"[{name}] loading...")
        stats = {"license": license_for(name), "written": Counter(), "invalid": 0,
                 "duplicates": 0, "excluded": 0, "label_kinds": Counter(), "errors": []}
        kept = 0
        try:
            for rec in SOURCES[name](a, rng):
                if kept >= a.max_per_source:
                    break
                if sha(norm_text(rec["state"])) in exclude:
                    stats["excluded"] += 1
                    continue
                try:
                    validate(rec)
                except AssertionError as e:
                    stats["invalid"] += 1
                    if len(stats["errors"]) < 5:
                        stats["errors"].append(f"invalid: {e}")
                    continue
                if rec["id"] in seen:
                    stats["duplicates"] += 1
                    continue
                # `split_of` is a hash of the state, which controls LEAKAGE (reshuffled
                # variants and near-duplicates cannot straddle the split), not DOMAIN SHIFT.
                # Holding out a whole source is what gives the state-level novelty head
                # something to be wrong about.
                split = "ood" if name in holdout else split_of(rec["state"])
                for r in [rec] + [reshuffle(rec, rng) for _ in range(a.variants)]:
                    if r["id"] in seen:
                        continue
                    seen.add(r["id"])
                    writer.write(split, r["source"], r)
                    stats["written"][split] += 1
                    stats["label_kinds"].update(r["label_kind"].values())
                kept += 1
        except Exception as e:  # noqa: BLE001
            stats["errors"].append(f"{type(e).__name__}: {e}")
            print(f"[{name}] FAILED: {type(e).__name__}: {str(e)[:300]}")
        stats["written"], stats["label_kinds"] = dict(stats["written"]), dict(stats["label_kinds"])
        manifest["sources"][name] = stats
        print(f"[{name}] base records kept: {kept}, written by split: {stats['written']}, "
              f"invalid: {stats['invalid']}, duplicates: {stats['duplicates']}")
    writer.close()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest written to {Path(a.out) / 'manifest.json'}")


# --------------------------------------------------------------------------- self-test
def self_test():
    """Runs every converter on synthetic rows shaped like the real datasets. No network."""
    rng = random.Random(0)
    intents = [f"intent_{i}" for i in range(30)] + ["oos"]
    fx = {
        "intents": list(convert_intents(
            [{"text": f"message {i}", "intent": i % len(intents)} for i in range(62)],
            "clinc", "text", "intent", intents, rng, oos_name="oos")),
        "ts_cls": list(convert_ts_classification(
            [{"sentence1": "A man sleeps.", "sentence2": "A person rests.", "labels": 0},
             {"sentence1": "Great movie!", "sentence2": "", "labels": 1}],
            "glue/rte", ["entailment", "not_entailment"], rng)),
        "ts_mc": list(convert_ts_multiple_choice(
            [{"inputs": "The sky is", "choice0": "blue", "choice1": "loud", "choice2": "", "labels": 0}],
            "sciq", rng)),
        "go_emotions": list(convert_go_emotions(
            [{"id": "c1", "text": "Thanks so much!", "example_very_unclear": False,
              **{e: int(e in ("gratitude", "joy") and k < 2) for e in GO_EMOTIONS}} for k in range(3)], rng)),
        "mhs": list(convert_mhs(
            [{"comment_id": 7, "text": "some comment", "hatespeech": h, "insult": s, "humiliate": 1, "violence": 0}
             for h, s in [(0, 1), (1, 3), (2, 4)]], rng)),
        "helpsteer2": list(convert_helpsteer2(
            [{"prompt": "Explain tides.", "response": "The moon...", "helpfulness": 3, "correctness": 4,
              "coherence": 4, "complexity": 1, "verbosity": 2}], rng)),
        "xlam": list(convert_xlam(
            [{"query": "Weather in Paris and Rome?",
              "tools": json.dumps([{"name": "get_weather", "description": "Current weather"},
                                   {"name": "get_time", "description": "Local time"}]),
              "answers": json.dumps([{"name": "get_weather", "arguments": {"city": "Paris"}},
                                     {"name": "get_weather", "arguments": {"city": "Rome"}}])}] * 20, rng)),
        "gsm8k": list(convert_gsm8k(
            [{"question": "Tom has 3 apples and buys 4 more.", "answer": "3+4=7\n#### 7"},
             {"question": "Big numbers", "answer": "#### 1,200"}], rng)),
        "ms_marco": list(convert_ms_marco(
            [{"query": "capital of france",
              "passages": {"passage_text": ["Paris is the capital.", "Lyon is a city."], "is_selected": [1, 0]}}], rng)),
    }
    failures = 0
    for name, recs in fx.items():
        assert recs, f"{name}: converter produced no records"
        for r in recs:
            try:
                validate(r)
                validate(reshuffle(r, rng))
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
        print(f"ok  {name:12s} {len(recs):3d} records")
    oos = [r for r in fx["intents"] if r["meta"]["out_of_scope"]]
    assert oos and all(r["labels"]["intent"]["none_of_these"] == 1.0 for r in oos), "oos rows must map to none"
    assert any(r["meta"]["gold_withheld"] for r in fx["xlam"]), "xlam should sometimes withhold the gold tool"
    assert fx["gsm8k"][1]["meta"] == {} and "1200" in fx["gsm8k"][1]["questions"]["answer"]["criteria"].values()
    assert all(split_of(r["state"]) == split_of(reshuffle(r, rng)["state"]) for r in fx["xlam"])
    # `split_for` layers domain shift on top of leakage control. A held-out source goes to
    # "ood" whatever its state hash says; a non-held-out source is untouched by the holdout
    # set. Without this the state-level novelty head has nothing to be wrong about.
    assert all(split_for("clinc", r["state"], {"clinc"}) == "ood" for r in fx["intents"][:25])
    assert split_for("mhs", "some state", {"clinc"}) == split_of("some state")
    assert split_for("clinc", "some state", ()) == split_of("some state")
    print("ok  split_for: holdout -> ood, others unchanged")
    print("\nsample (go_emotions):")
    print(json.dumps(fx["go_emotions"][0], indent=2)[:1200])
    print("\nsample (xlam):")
    print(json.dumps(fx["xlam"][0], indent=2)[:1200])
    if failures:
        sys.exit(f"{failures} validation failures")
    print("\nself-test passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default=DEFAULT_SOURCES,
                    help=f"comma-separated subset of: {','.join(SOURCES)}")
    ap.add_argument("--tasksource-tasks", default="glue/rte,glue/sst2",
                    help="comma-separated tasksource task ids (see tasksource.list_tasks())")
    ap.add_argument("--max-per-source", type=int, default=5000, help="base records per source")
    ap.add_argument("--variants", type=int, default=1, help="extra option-reshuffled copies per record")
    ap.add_argument("--exclude-file", help="text file, one benchmark item per line, to drop exact matches")
    ap.add_argument("--holdout-source", default="",
                    help="comma-separated sources forced into an 'ood' split, never train/val/test. "
                         "Use for domain-level OOD evaluation of the novelty/confidence head: the "
                         "hash split only controls leakage and leaves every source in every split.")
    ap.add_argument("--out", default="jev_data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hf-token", default=None, help="needed for gated datasets such as xlam")
    ap.add_argument("--self-test", action="store_true", help="run converter tests offline and exit")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    a.sources = [s.strip() for s in a.sources.split(",") if s.strip()]
    unknown = [s for s in a.sources if s not in SOURCES]
    if unknown:
        ap.error(f"unknown sources: {unknown}")
    a.tasksource_tasks = [t.strip() for t in a.tasksource_tasks.split(",") if t.strip()]
    build(a)


if __name__ == "__main__":
    main()
