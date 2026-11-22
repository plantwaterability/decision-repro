#!/usr/bin/env python3
"""
build_jev_ext2.py -- a second extension corpus: twelve license-checked public decision datasets from domains and
reasoning types that jev_data and jev_ext_strict do not cover, converted to the jev_data record schema and
deduplicated against every evaluation set the project reports on and against the existing training corpora.

Sources (output source -> origin; licence; labels; question types). Soft = a real annotator / vote distribution.
  circa                 google-research-datasets/circa                CC-BY-4.0 *    soft (5 raters)   choice + noul
                        indirect answers to yes/no questions: how X reads Y's reply (dialogue pragmatics)
  argq                  ibm-research/argument_quality_ranking_30k      CC-BY-3.0      soft (10 raters)  noul + choice
                        argument quality (would you use it in a speech?) and stance on a debate motion
  scruples/anecdotes    tasksource/scruples (= allenai/scruples v1.0)  Apache-2.0     soft (votes)      choice
                        r/AmItheAsshole posts: who is in the wrong (community vote distribution over 5 verdicts)
  scruples/dilemmas     allenai/scruples v1.0 dilemmas (GCS release)   Apache-2.0     soft (5-10 raters) choice
                        which of two real actions is less ethical
  mtbench_human         lmsys/mt_bench_human_judgments                 CC-BY-4.0      soft (1-7 judges) choice
                        pairwise expert preference between two assistants' answers (A / B / tie)
  summeval              SummEval expert annotations (Yale-LILY, GCS) + mteb/summeval articles   MIT **   soft (3 experts) score x4
                        summary coherence / consistency / fluency / relevance, 1-5
  politeness/{wikipedia,stackexchange}  ConvoKit Stanford Politeness Corpus  CC-BY-4.0  soft (5 raters)  score
                        how polite a request is (5 levels, binned from the 1-25 ratings)
  casino/{outcomes,strategies}  kchawla123/casino (CaSiNo)             CC-BY-4.0      hard              choice + score + noul
                        campsite negotiation dialogues: priorities, self-reported satisfaction / liking, strategies
  code_contests         deepmind/code_contests                         CC-BY-4.0      hard (verdicts)   noul
                        would this submitted solution pass all tests (accepted vs rejected submissions)
  tabfact               table-benchmark/tabfact (= TabFact)            CC-BY-4.0      hard              noul (several per table)
                        is a statement supported by a Wikipedia table
  medical/{medmcqa,pubmedqa}  openlifescienceai/medmcqa; qiaojin/PubMedQA   Apache-2.0; MIT   hard / noisy_hard  choice
                        medical-exam MCQ; biomedical abstract -> yes / no / maybe
  commonsense/{csqa,siqa,cosmosqa,winogrande,ecare}   tau/commonsense_qa (MIT); baber/social_i_qa (Social IQa,
                        CC-BY-4.0); Samsoup/cosmos_qa (CosmosQA, CC-BY-4.0); allenai/winogrande (CC-BY);
                        12ml/e-CARE (MIT)                                            hard              choice
  bbq                   walledai/BBQ (= nyu-mll/BBQ)                   CC-BY-4.0      hard              choice
                        ambiguous vs disambiguated social contexts ("can't be determined" is often right)
  *  Circa's README says "Creative Commons Attribution 4.0" but links the BY-SA 4.0 text: treat as share-alike.
  ** SummEval annotations are MIT; the article text is CNN / Daily Mail news (cnn_dailymail is Apache-2.0 on the Hub,
     copyright stays with the publishers). Drop it with --sources if that matters for a release.
Every record's `license` field carries the per-source string in LICENSES below.

Leakage control (always on). A candidate is dropped if, against ANY fingerprint state:
  exact    its state -- raw, or flattened (JSON states -> their string leaves joined) -- normalized as
           analysis/claude_overlap.py `norm` equals a fingerprint state's normalized raw or flattened form
  leaf     one of its raw source fields (question, context, argument, ... ; >= 5 words) equals, after `norm`, one
           string leaf of a fingerprint state (e.g. an MMLU question inside a jev-bench JSON state), or (>= 3 words)
           a whole fingerprint state
  shingle  it shares >= 3 distinct 8-word shingles with any ONE fingerprint state (as build_jev_ext.py)
Fingerprint states come from
  * fetched eval STATES (default; --no-fetch-evals to skip): jev-bench v0.1.1 test, all 22 configs; typed-decisions
    all splits (only the `state` parquet column is read); Typed Decision Bench v0.3 suite items + the full TrialGPT and
    Abt-Buy source tables (a superset of its 359 withheld items); MMLU test (all subjects) + ARC-Challenge/Easy test;
    LegalBench test, all 162 tasks (text columns only); PhishNChips core emails; the community tool_risk and
    ticket_routing items; DICES-350/990 conversations; ChaosNLI (SNLI, MNLI, alpha-NLI). Label fields are never
    kept: each fetcher reads the state / text fields only.
  * --eval TAG=PATH (file or directory of jsonl with a `state` field): e.g. jev_real test, the converted extevals
    (b, community, tdb incl. its rebuilt withheld items), dices, chaos_extra. jev_predictions* files are never read.
  * --fingerprint TAG=PATH: training corpora -- jev_data, jev_ext_strict (required unless --allow-missing-corpora).
The build refuses to run if an eval family could not be fetched (unless --allow-missing-evals), and prints per
source the drops by kind and tag. Candidates are also deduplicated across all new sources (exact normalized state).

Splits: train / val ~95/5 by a hash of a per-source GROUP key (article, problem, dialogue, debate topic, table, ...), so
near-duplicate states (16 summaries of one article, prefixes of one negotiation) never straddle train and val.
Output: <out>/{train,val}/<source>.jsonl ('/' -> '__') and <out>/manifest.json; the run ends by reading the corpus back
through trainers/cache_features.iter_units (build_jev_ext.final_check).

Usage (on a node; pandas + pyarrow + huggingface_hub; HF access is anonymous):
  export HF_HOME=/data/local/hf_jev_ext2
  python3 build_jev_ext2.py --out /data/local/jev_ext2 \\
      --eval jev_real=/data/local/jev_real/test/real.jsonl \\
      --eval extevals=/data/local/extevals --eval dices=/data/shared/dices \\
      --fingerprint jev_data=/data/local/jev_data --fingerprint jev_ext_strict=/data/local/jev_ext_strict \\
      --fingerprint typed_train=/data/shared/jev/sofa_rows/td_train.jsonl   # optional: typed-decisions train is also fetched
  python3 build_jev_ext2.py --self-test                                   # no network
  python3 build_jev_ext2.py --small --limit-rows 400 --out /tmp/x ...     # smallest files only (a laptop smoke test)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import pickle
import random
import re
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "analysis"))

from claude_overlap import norm  # noqa: E402  the eval-overlap normalization, reused verbatim

from build_jev_data import (  # noqa: E402
    Writer,
    choice_q,
    noul_q,
    record,
    score_q,
    shuffled,
    split_of,
    validate,
)
from build_jev_ext import Labeler, final_check, kind_of, plain, read_jsonl, with_retry  # noqa: E402

_spec = importlib.util.spec_from_file_location("convert_extevals", HERE / "trainers" / "convert_extevals.py")
cx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cx)   # stdlib-only module: its pins (jev-bench, LegalBench, TDB, community) and `literals`

MAX_K = 52
VAL_PCT = 5
SHINGLE_WORDS = 8
SHINGLE_MIN_SHARED = 3
LEAF_MIN_WORDS = 5
PROBE_MIN_WORDS = 3
WORD = re.compile(r"\w+")

LICENSES = {
    "circa": "CC-BY-4.0 (README; its link points at CC-BY-SA-4.0, so treat derived records as share-alike)",
    "argq": "CC-BY-3.0 (IBM Debater IBM-ArgQ-Rank-30kArgs)",
    "scruples": "Apache-2.0 (allenai/scruples; anecdote text from Reddit r/AmItheAsshole)",
    "mtbench_human": "CC-BY-4.0 (lmsys/mt_bench_human_judgments; responses by GPT-4, GPT-3.5, Claude-v1, Vicuna, Alpaca, LLaMA)",
    "summeval": "MIT (SummEval annotations; mteb/summeval) -- article text is CNN/Daily Mail news (publisher copyright)",
    "politeness": "CC-BY-4.0 (ConvoKit Stanford Politeness Corpus)",
    "casino": "CC-BY-4.0 (CaSiNo, kushalchawla/CaSiNo)",
    "code_contests": "CC-BY-4.0 (deepmind/code_contests)",
    "tabfact": "CC-BY-4.0 (TabFact; via the table-benchmark/tabfact Parquet copy)",
    "medical/medmcqa": "Apache-2.0 (MedMCQA; GitHub repo MIT)",
    "medical/pubmedqa": "MIT (PubMedQA)",
    "commonsense/csqa": "MIT (CommonsenseQA)",
    "commonsense/siqa": "CC-BY-4.0 (Social IQa)",
    "commonsense/cosmosqa": "CC-BY-4.0 (CosmosQA)",
    "commonsense/winogrande": "CC-BY (WinoGrande dataset; code Apache-2.0)",
    "commonsense/ecare": "MIT (e-CARE, waste-wood/e-CARE)",
    "bbq": "CC-BY-4.0 (BBQ, nyu-mll/BBQ)",
}
ORIGINS = {
    "circa": "google-research-datasets/circa",
    "argq": "ibm-research/argument_quality_ranking_30k",
    "scruples/anecdotes": "tasksource/scruples",
    "scruples/dilemmas": "https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/data/dilemmas.tar.gz",
    "mtbench_human": "lmsys/mt_bench_human_judgments",
    "summeval": "https://storage.googleapis.com/sfr-summarization-repo-research/model_annotations.aligned.jsonl + mteb/summeval",
    "politeness/wikipedia": "https://zissou.infosci.cornell.edu/convokit/datasets/wikipedia-politeness-corpus/wikipedia-politeness-corpus.zip",
    "politeness/stackexchange": "https://zissou.infosci.cornell.edu/convokit/datasets/stack-exchange-politeness-corpus/stack-exchange-politeness-corpus.zip",
    "casino/outcomes": "kchawla123/casino",
    "casino/strategies": "kchawla123/casino",
    "code_contests": "deepmind/code_contests",
    "tabfact": "table-benchmark/tabfact",
    "medical/medmcqa": "openlifescienceai/medmcqa",
    "medical/pubmedqa": "qiaojin/PubMedQA",
    "commonsense/csqa": "tau/commonsense_qa",
    "commonsense/siqa": "baber/social_i_qa",
    "commonsense/cosmosqa": "Samsoup/cosmos_qa",
    "commonsense/winogrande": "allenai/winogrande",
    "commonsense/ecare": "12ml/e-CARE",
    "bbq": "walledai/BBQ",
}
# default record caps per output source (records, before the train/val split); --scale multiplies them
CAPS = {
    "circa": 2500, "argq": 2500, "scruples/anecdotes": 2000, "scruples/dilemmas": 1500, "mtbench_human": 2000,
    "summeval": 1600, "politeness/wikipedia": 1200, "politeness/stackexchange": 1200, "casino/outcomes": 1030,
    "casino/strategies": 1200, "code_contests": 2000, "tabfact": 1500, "medical/medmcqa": 1200, "medical/pubmedqa": 1200,
    "commonsense/csqa": 500, "commonsense/siqa": 500, "commonsense/cosmosqa": 500, "commonsense/winogrande": 500,
    "commonsense/ecare": 500, "bbq": 1200,
}
FAMILIES = ["circa", "argq", "scruples", "mtbench_human", "summeval", "politeness", "casino", "code_contests", "tabfact",
            "medical", "commonsense", "bbq"]


def family_of(source):
    return source.split("/")[0]


def license_of(source):
    return LICENSES.get(source) or LICENSES[family_of(source)]


# --------------------------------------------------------------------------- download helpers
def hf_file(a, repo, filename, revision=None):
    from huggingface_hub import hf_hub_download
    return with_retry(lambda: hf_hub_download(repo, filename, repo_type="dataset", revision=revision, token=a.hf_token),
                      f"{repo}/{filename}")


def hf_list(a, repo, revision=None):
    from huggingface_hub import HfApi
    return with_retry(lambda: HfApi().list_repo_files(repo, repo_type="dataset", revision=revision, token=a.hf_token),
                      f"list {repo}")


def hf_sha(a, repo):
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(repo, token=a.hf_token).sha
    except Exception as e:  # noqa: BLE001
        return f"unresolved ({type(e).__name__})"


def url_file(a, url):
    """A non-Hub download (GitHub raw, GCS, ConvoKit), cached under --cache, with retries."""
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / (hashlib.sha256(url.encode()).hexdigest()[:20] + "_" + re.sub(r"[^\w.-]", "_", url.rsplit("/", 1)[-1])[:60])
    if local.exists():
        return local

    def get():
        req = urllib.request.Request(url, headers={"User-Agent": "jev-build-ext2"})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = r.read()
        tmp = local.with_name(local.name + ".part")
        tmp.write_bytes(data)
        tmp.rename(local)
        return local
    return with_retry(get, url)


def read_parquet(path, columns=None):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=columns).to_pylist()


def read_csv(path, **kw):
    csv.field_size_limit(1 << 30)
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, **kw))


def sample(rows, n, rng):
    rows = list(rows)
    if n is None or len(rows) <= n:
        return rows
    return [rows[i] for i in sorted(rng.sample(range(len(rows)), n))]


def clean(s):
    return re.sub(r"[ \t]+\n", "\n", str(s or "")).strip()


def one_hot(gold, keys):
    return {k: (1.0 if k == gold else 0.0) for k in keys}


def vote_dist(lab, keys, counts):
    """Vote counts -> a distribution through the shared Labeler checks (None, counted in lab.drops, if unusable)."""
    tot = sum(counts)
    if tot <= 0:
        lab.drops["target_all_zero"] += 1
        return None
    return lab.dist(keys, [c / tot for c in counts])


def make_record(source, state, qs, labs, meta, group, probes=(), kinds=None):
    kinds = {q: (kinds or {}).get(q) or kind_of(labs[q]) for q in qs}
    rec = record(source, state, qs, labs, kinds, None)
    rec["base_id"] = rec["id"]
    rec["license"] = license_of(source)
    rec["meta"] = {"dataset": ORIGINS[source], "group": str(group), **meta}
    rec["_probes"] = [p for p in probes if p]
    return rec


# --------------------------------------------------------------------------- converters (pure: rows -> records)
CIRCA_KEYS = {
    "Yes": ("yes", "Yes"),
    "No": ("no", "No"),
    "Probably yes / sometimes yes": ("probably_yes", "Probably yes / sometimes yes"),
    "Probably no": ("probably_no", "Probably no"),
    "Yes, subject to some conditions": ("yes_conditions", "Yes, subject to some conditions"),
    "In the middle, neither yes nor no": ("in_the_middle", "In the middle, neither yes nor no"),
    "Other": ("other", "Other: the reply does not answer the yes/no question either way"),
}
CIRCA_YES = {"yes", "probably_yes", "yes_conditions"}
CIRCA_INSTR = [
    "X asked Y a yes/no question and Y answered indirectly. How would X most likely interpret Y's reply?",
    "How should Y's reply to X's yes/no question be understood?",
]


def conv_circa(rows, rng, lab):
    """One record per (context, question, answer); soft labels from the 5 strict judgements. 'I am not sure how X will
    interpret Y's answer' (the rater's uncertainty, not a reading of the answer) and NA are dropped before normalizing."""
    keys = [k for k, _ in CIRCA_KEYS.values()]
    for r in rows:
        ctx, q, ans = clean(r.get("context")), clean(r.get("question-X")), clean(r.get("answer-Y"))
        votes = [CIRCA_KEYS[j][0] for j in str(r.get("judgements") or "").split("#") if j in CIRCA_KEYS]
        if not (ctx and q and ans):
            lab.drops["bad_row"] += 1
            continue
        if len(votes) < 3:
            lab.drops["too_few_votes"] += 1
            continue
        c = Counter(votes)
        d = vote_dist(lab, keys, [c.get(k, 0) for k in keys])
        p_yes = lab.noul(sum(c[k] for k in CIRCA_YES) / len(votes))
        if not d or not p_yes:
            continue
        state = f"Context: {ctx}\nX asks: {q}\nY replies: {ans}"
        qs, labs = {}, {}
        crit = shuffled({k: desc for k, desc in CIRCA_KEYS.values()}, rng)
        qs["interpretation"] = choice_q(rng.choice(CIRCA_INSTR), crit)
        labs["interpretation"] = {k: d[k] for k in crit}
        qs["means_yes"] = noul_q("Would X take Y's reply as a yes (a plain yes, a probable yes, or a yes under some conditions)?")
        labs["means_yes"] = p_yes
        yield make_record("circa", state, qs, labs, {"n_raters": len(votes), "context": ctx}, group=norm(q), probes=[q, ans])


ARGQ_QUALITY = ("Disregarding your own opinion on the motion, would you recommend that a friend preparing a speech on this "
                "motion use this argument as is?")


def conv_argq(rows, rng, lab):
    """IBM-ArgQ-Rank-30kArgs: WA = the reliability-weighted share of 'yes' quality votes (10 raters); stance_WA in [-1, 1]
    is the weighted stance vote, mapped to P(support) = (1 + stance_WA) / 2."""
    for r in rows:
        arg, topic = clean(r.get("argument")), clean(r.get("topic"))
        try:
            wa = float(r.get("WA"))
        except (TypeError, ValueError):
            lab.drops["bad_row"] += 1
            continue
        if not arg or not topic:
            lab.drops["bad_row"] += 1
            continue
        qs, labs = {}, {}
        p = lab.noul(min(max(wa, 0.0), 1.0))
        if p:
            qs["recommend"], labs["recommend"] = noul_q(ARGQ_QUALITY), p
        try:
            st = float(r.get("stance_WA"))
        except (TypeError, ValueError):
            st = None
        if st is not None:
            ps = min(max((1.0 + st) / 2.0, 0.0), 1.0)
            d = lab.dist(["support", "contest"], [ps, 1.0 - ps])
            if d:
                crit = shuffled({"support": "It supports the motion (argues for it)",
                                 "contest": "It contests the motion (argues against it)"}, rng)
                qs["stance"] = choice_q("Does this argument support or contest the motion?", crit)
                labs["stance"] = {k: d[k] for k in crit}
        if qs:
            yield make_record("argq", f"Debate motion: {topic}\nArgument: {arg}", qs, labs,
                              {"n_raters": 10, "topic": topic, "orig_set": r.get("set")}, group=norm(topic), probes=[arg])


SCRUPLES_VERDICTS = {
    "AUTHOR": ("author", "The author (the person who wrote the post)"),
    "OTHER": ("other", "The other person or people in the story, not the author"),
    "EVERYBODY": ("everybody", "Everyone involved"),
    "NOBODY": ("nobody", "No one"),
    "INFO": ("info", "It cannot be judged without more information"),
}


def conv_scruples_anecdotes(rows, rng, lab, max_chars):
    keys = [k for k, _ in SCRUPLES_VERDICTS.values()]
    for r in rows:
        title, text = clean(r.get("title")), clean(r.get("text"))
        scores = r.get("label_scores") or {}
        state = f"{title}\n\n{text}" if title else text
        if not text:
            lab.drops["bad_row"] += 1
            continue
        if len(state) > max_chars:
            lab.drops["too_long"] += 1
            continue
        votes = {SCRUPLES_VERDICTS[k][0]: int(v or 0) for k, v in scores.items() if k in SCRUPLES_VERDICTS}
        n = sum(votes.values())
        if n < 3:
            lab.drops["too_few_votes"] += 1
            continue
        d = vote_dist(lab, keys, [votes.get(k, 0) for k in keys])
        if not d:
            continue
        crit = shuffled({k: desc for k, desc in SCRUPLES_VERDICTS.values()}, rng)
        qs = {"verdict": choice_q("Who is in the wrong in the situation this post describes?", crit)}
        labs = {"verdict": {k: d[k] for k in crit}}
        yield make_record("scruples/anecdotes", state, qs, labs,
                          {"n_raters": n, "orig_id": r.get("id"), "post_type": r.get("post_type"), "orig_split": r.get("_split")},
                          group=r.get("post_id") or r.get("id"), probes=[title])


def conv_scruples_dilemmas(rows, rng, lab):
    """Two actions; soft label = annotators choosing each as LESS ethical (gold + human-performance annotations)."""
    for r in rows:
        acts = r.get("actions") or []
        if len(acts) != 2:
            lab.drops["bad_row"] += 1
            continue
        a1, a2 = clean(acts[0].get("description")), clean(acts[1].get("description"))
        g = list(r.get("gold_annotations") or [0, 0])
        h = list(r.get("human_perf_annotations") or [0, 0])
        votes = [int(g[0]) + int(h[0]), int(g[1]) + int(h[1])]
        if not a1 or not a2 or norm(a1) == norm(a2):
            lab.drops["bad_row"] += 1
            continue
        d = vote_dist(lab, ["first", "second"], votes)
        if not d:
            continue
        state = f"Two people each describe something they did.\nAction 1: {a1}\nAction 2: {a2}"
        crit = shuffled({"first": f"Action 1 ({a1})", "second": f"Action 2 ({a2})"}, rng)
        qs = {"less_ethical": choice_q("Which of the two actions is less ethical?", crit)}
        yield make_record("scruples/dilemmas", state, qs, {"less_ethical": {k: d[k] for k in crit}},
                          {"n_raters": sum(votes), "orig_id": r.get("id"), "orig_split": r.get("_split")},
                          group=r.get("id"), probes=[a1, a2])


def mt_turns(conv):
    """[(user, assistant), ...] from an MT-bench conversation (alternating user / assistant messages)."""
    msgs = [(m.get("role"), clean(m.get("content"))) for m in conv or []]
    out = []
    for i in range(0, len(msgs) - 1, 2):
        (r1, u), (r2, a) = msgs[i], msgs[i + 1]
        if r1 != "user" or r2 != "assistant":
            return None
        out.append((u, a))
    return out


def conv_mtbench(rows, rng, lab, max_chars):
    """Group the human judgments by (question, unordered model pair, turn); the soft label is the vote share for each
    model or a tie. Which model is shown as Assistant A is a seeded coin flip; model names are never shown."""
    groups = defaultdict(list)
    for r in rows:
        pair = tuple(sorted([r["model_a"], r["model_b"]]))
        groups[(int(r["question_id"]), pair, int(r["turn"]))].append(r)
    for (qid, (m1, m2), turn), grp in sorted(groups.items()):
        votes, convs = Counter(), {}
        for r in grp:
            convs[r["model_a"]], convs[r["model_b"]] = r["conversation_a"], r["conversation_b"]
            w = r["winner"]
            votes["tie" if w.startswith("tie") else (r["model_a"] if w == "model_a" else r["model_b"])] += 1
        first, second = (m1, m2) if rng.random() < 0.5 else (m2, m1)
        ta, tb = mt_turns(convs.get(first)), mt_turns(convs.get(second))
        if not ta or not tb or len(ta) < turn or len(tb) < turn:
            lab.drops["bad_row"] += 1
            continue
        if turn == 1:
            state = f"User:\n{ta[0][0]}\n\n[Assistant A]\n{ta[0][1]}\n\n[Assistant B]\n{tb[0][1]}"
            instr = "Which assistant answered the user better?"
        else:
            def show(t, name):
                return "\n\n".join(f"User: {u}\n{name}: {x}" for u, x in t[:2])
            state = (f"Two assistants held the same two-turn conversation.\n\n=== Conversation with Assistant A ===\n"
                     f"{show(ta, 'Assistant A')}\n\n=== Conversation with Assistant B ===\n{show(tb, 'Assistant B')}")
            instr = "Judging the replies to the user's SECOND message (in the context of the whole conversation), which assistant did better?"
        if len(state) > max_chars:
            lab.drops["too_long"] += 1
            continue
        d = vote_dist(lab, ["a", "b", "tie"], [votes[first], votes[second], votes["tie"]])
        if not d:
            continue
        crit = shuffled({"a": "Assistant A", "b": "Assistant B", "tie": "Tie: the two are about equally good"}, rng)
        yield make_record("mtbench_human", state, {"preference": choice_q(instr, crit)}, {"preference": {k: d[k] for k in crit}},
                          {"n_raters": sum(votes.values()), "question_id": qid, "turn": turn,
                           "model_shown_as_a": first, "model_shown_as_b": second},
                          group=f"q{qid}", probes=[ta[0][0]])


SUMMEVAL_Q = {
    "coherence": "Rate the summary's COHERENCE: the collective quality of its sentences -- well-structured and "
                 "well-organized, building from sentence to sentence into a coherent body of information.",
    "consistency": "Rate the summary's factual CONSISTENCY with the article: it should state only facts supported by the "
                   "article, with no hallucinated content.",
    "fluency": "Rate the summary's FLUENCY: the quality of its individual sentences (grammar, formatting, readability).",
    "relevance": "Rate the summary's RELEVANCE: does it select the important content of the article and leave out the rest?",
}
SUMMEVAL_LEVELS = ["1 - very poor", "2 - poor", "3 - acceptable", "4 - good", "5 - excellent"]


def conv_summeval(rows, articles, rng, lab, max_chars):
    """The 3 EXPERT annotations per (article, system summary) give each 1-5 score question a soft label."""
    for r in rows:
        art, summ = clean(articles.get(r.get("id"))), clean(r.get("decoded"))
        ex = r.get("expert_annotations") or []
        if not art or not summ or len(ex) < 2:
            lab.drops["bad_row"] += 1
            continue
        state = f"Article:\n{art}\n\nSummary:\n{summ}"
        if len(state) > max_chars:
            lab.drops["too_long"] += 1
            continue
        qs, labs = {}, {}
        for dim, instr in SUMMEVAL_Q.items():
            c = Counter(int(e[dim]) for e in ex if e.get(dim) in (1, 2, 3, 4, 5))
            d = vote_dist(lab, SUMMEVAL_LEVELS, [c.get(i + 1, 0) for i in range(5)])
            if d:
                qs[dim], labs[dim] = score_q(instr, SUMMEVAL_LEVELS), d
        if qs:
            yield make_record("summeval", state, qs, labs, {"n_raters": len(ex), "orig_id": r.get("id"), "system": r.get("model_id")},
                              group=r.get("id"))


POLITE_LEVELS = ["very impolite", "somewhat impolite", "neutral", "somewhat polite", "very polite"]
POLITE_WHERE = {"wikipedia": "a Wikipedia talk page (a request to another editor)",
                "stackexchange": "a Stack Exchange site (a request to another user)"}


def polite_bin(score):
    """1-25 slider rating -> one of five levels (1-5, 6-10, 11-15, 16-20, 21-25)."""
    s = int(score)
    return None if not 1 <= s <= 25 else (s - 1) // 5


def conv_politeness(rows, community, rng, lab):
    """rows: (utterance, {worker: 1-25 rating}); soft label over the 5 binned levels."""
    for utt, ann in rows:
        text = clean(utt.get("text"))
        bins = [polite_bin(v) for v in (ann or {}).values()]
        bins = [b for b in bins if b is not None]
        if not text:
            lab.drops["bad_row"] += 1
            continue
        if len(bins) < 3:
            lab.drops["too_few_votes"] += 1
            continue
        c = Counter(bins)
        d = vote_dist(lab, POLITE_LEVELS, [c.get(i, 0) for i in range(5)])
        if not d:
            continue
        state = f"A request posted on {POLITE_WHERE[community]}:\n{text}"
        yield make_record(f"politeness/{community}", state, {"politeness": score_q("How polite is this request?", POLITE_LEVELS)},
                          {"politeness": d}, {"n_raters": len(bins), "orig_id": utt.get("id")},
                          group=f"{community}:{utt.get('id')}", probes=[text])


CASINO_INTRO = ("Two campsite neighbours, Agent 1 and Agent 2, negotiate how to split 3 packages each of food, water and "
                "firewood. Each has private priorities (high / medium / low) and reasons for them.")
CASINO_ITEMS = {"food": "Food", "water": "Water", "firewood": "Firewood"}
CASINO_SAT = ["Extremely dissatisfied", "Slightly dissatisfied", "Undecided", "Slightly satisfied", "Extremely satisfied"]
CASINO_LIKE = ["Extremely dislike", "Slightly dislike", "Undecided", "Slightly like", "Extremely like"]
CASINO_STRATEGIES = {
    "small-talk": "engage in small talk or rapport building unrelated to the negotiation itself (greetings, chat about the trip)",
    "self-need": "argue for the speaker's OWN need for an item (e.g. 'we need extra water because it is hot')",
    "other-need": "argue for an item on behalf of someone else, such as family, friends or a pet",
    "no-need": "point out that the speaker does not need an item (e.g. 'we have plenty of water')",
    "elicit-pref": "try to find out the other agent's preferences (e.g. 'what do you need most?')",
    "uv-part": "undermine the other agent's needs, suggesting they do not really need an item",
    "vouch-fair": "appeal to fairness, calling for a fair deal or calling a deal fair or unfair",
    "showing-empathy": "show empathy toward the other agent's situation",
    "promote-coordination": "promote coordination toward a deal that works for both",
}
CASINO_EVENTS = {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}
CASINO_AGENT = {"mturk_agent_1": "Agent 1", "mturk_agent_2": "Agent 2"}


def casino_deal(td):
    you, they = td.get("issue2youget") or {}, td.get("issue2theyget") or {}
    fmt = lambda d: ", ".join(f"{d.get(k) or 0} {k.lower()}" for k in ("Food", "Water", "Firewood"))  # noqa: E731
    return fmt(you), fmt(they)


def casino_lines(chat, upto=None, events=True):
    out = []
    for i, c in enumerate(chat[: upto if upto is not None else len(chat)]):
        who, other = CASINO_AGENT.get(c.get("id"), c.get("id")), None
        other = "Agent 2" if who == "Agent 1" else "Agent 1"
        t = c.get("text")
        if t in CASINO_EVENTS:
            if not events:
                continue
            if t == "Submit-Deal":
                you, they = casino_deal(c.get("task_data") or {})
                out.append(f"[{who} proposes a final deal: {who} gets {you}; {other} gets {they}]")
            elif t == "Accept-Deal":
                out.append(f"[{who} accepts the deal]")
            elif t == "Reject-Deal":
                out.append(f"[{who} rejects the deal]")
            else:
                out.append(f"[{who} walks away without a deal]")
        else:
            out.append(f"{who}: {clean(t)}")
    return out


def casino_gid(r):
    return "casino:" + hashlib.sha1(json.dumps([c.get("text") for c in r["chat_logs"]]).encode()).hexdigest()[:16]


def conv_casino_outcomes(rows, rng, lab):
    """Whole dialogue incl. the deal events -> item priorities (hard, from the private value tables) and each agent's
    post-negotiation self-reports (hard, a single rater: the participant)."""
    for r in rows:
        pi, chat = r.get("participant_info") or {}, r.get("chat_logs") or []
        if set(pi) != set(CASINO_AGENT) or not chat:
            lab.drops["bad_row"] += 1
            continue
        state = CASINO_INTRO + "\n\n" + "\n".join(casino_lines(chat))
        qs, labs = {}, {}
        focus = rng.choice(sorted(CASINO_AGENT))
        for agent, name in sorted(CASINO_AGENT.items()):
            v2i = {k: str(v).lower() for k, v in (pi[agent].get("value2issue") or {}).items()}
            tag = name.split()[-1]
            for level, word in (("High", "most"), ("Low", "least")):
                if agent != focus and level == "Low":
                    continue
                gold = v2i.get(level)
                if gold not in CASINO_ITEMS:
                    continue
                crit = shuffled(dict(CASINO_ITEMS), rng)
                qid = f"{level.lower()}_priority_{tag}"
                qs[qid] = choice_q(f"Which item does {name} need {word} (their {level.lower()}est priority)?", crit)
                labs[qid] = one_hot(gold, list(crit))
            if agent == focus:
                out = pi[agent].get("outcomes") or {}
                for key, levels, instr in (("satisfaction", CASINO_SAT, f"After the negotiation, how satisfied did {name} report being with the outcome?"),
                                           ("opponent_likeness", CASINO_LIKE, f"After the negotiation, how much did {name} report liking the other agent?")):
                    if out.get(key) in levels:
                        qs[f"{key}_{tag}"] = score_q(instr, levels)
                        labs[f"{key}_{tag}"] = one_hot(out[key], levels)
        if qs:
            yield make_record("casino/outcomes", state, qs, labs, {"focus_agent": CASINO_AGENT[focus]}, group=casino_gid(r))


def conv_casino_strategies(rows, rng, lab, per_dialogue=3, n_questions=3):
    """Utterance-level strategy annotations (396 dialogues): the dialogue up to and including one message, then noul
    questions 'does the last message ...?' over a mix of strategies it uses and does not use."""
    for r in rows:
        ann, chat = r.get("annotations") or [], r.get("chat_logs") or []
        msgs = [(i, c) for i, c in enumerate(chat) if c.get("text") not in CASINO_EVENTS]
        if not ann:
            continue
        if [clean(t) for t, _ in ann] != [clean(c["text"]) for _, c in msgs[: len(ann)]]:
            lab.drops["annotation_misaligned"] += 1
            continue
        picks = sorted(rng.sample(range(len(ann)), min(per_dialogue, len(ann))))
        for j in picks:
            used = {s.strip() for s in str(ann[j][1]).split(",") if s.strip() in CASINO_STRATEGIES}
            unused = [s for s in CASINO_STRATEGIES if s not in used]
            chosen = rng.sample(sorted(used), min(len(used), max(1, n_questions // 2)))
            chosen += rng.sample(unused, min(len(unused), n_questions - len(chosen)))
            idx, c = msgs[j]
            who = CASINO_AGENT.get(c.get("id"), c.get("id"))
            state = (CASINO_INTRO + "\n\n" + "\n".join(casino_lines(chat, upto=idx + 1, events=False)) +
                     f"\n\n(The last message, from {who}, is the one being asked about.)")
            qs, labs = {}, {}
            for s in chosen:
                qid = "uses_" + s.replace("-", "_")
                qs[qid] = noul_q(f"Does the last message {CASINO_STRATEGIES[s]}?")
                labs[qid] = {"yes": 1.0, "no": 0.0} if s in used else {"yes": 0.0, "no": 1.0}
            if qs:
                yield make_record("casino/strategies", state, qs, labs, {"utterance_index": j}, group=casino_gid(r),
                                  probes=[clean(c.get("text"))])


CC_SOURCE = {0: "unknown source", 1: "CodeChef", 2: "Codeforces", 3: "HackerEarth", 4: "Code Jam", 5: "AtCoder", 6: "Aizu"}
CC_LANG = {2: "C++", 3: "Python 3", 4: "Java"}
CC_Q = ("Is this submission correct, i.e. would it produce the right output on all of the problem's tests within the "
        "time and memory limits?")


def conv_code_contests(problems, rng, lab, max_chars, per_side=2, max_code=3000):
    """Accepted (`solutions`) vs rejected (`incorrect_solutions`) submissions; up to `per_side` of each per problem, as
    many rejected as accepted so each problem is balanced. Python 2 / unknown-language submissions are skipped."""
    for p in problems:
        desc = clean(p.get("description"))
        if not desc:
            lab.drops["bad_row"] += 1
            continue
        tl = p.get("time_limit") or {}
        secs = (tl.get("seconds") or 0) + (tl.get("nanos") or 0) / 1e9
        mem = (p.get("memory_limit_bytes") or 0) / 1e6
        limits = ", ".join(x for x in (f"time limit {secs:g} s" if secs else "", f"memory limit {mem:g} MB" if mem else "") if x)
        head = f"Programming problem ({CC_SOURCE.get(p.get('source'), 'unknown source')}{'; ' + limits if limits else ''}):\n{desc}"

        def pool(field):
            s = p.get(field) or {}
            out = [(lang, clean(code)) for lang, code in zip(s.get("language") or [], s.get("solution") or [])
                   if lang in CC_LANG and code and 3 <= len(clean(code).splitlines()) and len(code) <= max_code]
            rng.shuffle(out)
            return out
        good, bad = pool("solutions"), pool("incorrect_solutions")
        n = min(per_side, len(good), len(bad))
        if n == 0:
            lab.drops["no_balanced_pair"] += 1
            continue
        for ok, (lang, code) in [(True, x) for x in good[:n]] + [(False, x) for x in bad[:n]]:
            state = f"{head}\n\nSubmitted solution ({CC_LANG[lang]}):\n{code}"
            if len(state) > max_chars:
                lab.drops["too_long"] += 1
                continue
            yield make_record("code_contests", state, {"accepted": noul_q(CC_Q)},
                              {"accepted": {"yes": 1.0, "no": 0.0} if ok else {"yes": 0.0, "no": 1.0}},
                              {"problem": p.get("name"), "source_site": CC_SOURCE.get(p.get("source")), "language": CC_LANG[lang],
                               "difficulty": p.get("difficulty"), "cf_rating": p.get("cf_rating")},
                              group=p.get("name") or desc[:200])


def conv_tabfact(rows, rng, lab, max_chars, per_table=5):
    """One record per table (its caption + rows as the state) with up to `per_table` statements as noul questions."""
    tables = defaultdict(list)
    for r in rows:
        tables[(clean(r.get("table_title")), str(r.get("table") or "").strip())].append(r)
    instr = ["Is the following statement supported by the table? Statement: {s}",
             "According to the table, is this statement true? Statement: {s}"]
    for (title, table), grp in tables.items():
        lines = [" | ".join(c.strip() for c in ln.split("#")) for ln in table.splitlines() if ln.strip()]
        if len(lines) < 2:
            lab.drops["bad_row"] += 1
            continue
        state = (f"Table: {title}\n" if title else "Table:\n") + "\n".join(lines)
        if len(state) > max_chars:
            lab.drops["too_long"] += 1
            continue
        qs, labs, probes = {}, {}, []
        for i, r in enumerate(rng.sample(grp, min(per_table, len(grp)))):
            s, ans = clean(r.get("question")), str(r.get("answer") or "")
            if not s or ans not in ("Entailed", "Refuted"):
                lab.drops["bad_row"] += 1
                continue
            qs[f"s{i}"] = noul_q(rng.choice(instr).format(s=s))
            labs[f"s{i}"] = {"yes": 1.0, "no": 0.0} if ans == "Entailed" else {"yes": 0.0, "no": 1.0}
            probes.append(s)
        if qs:
            yield make_record("tabfact", state, qs, labs, {"n_statements": len(grp)},
                              group=hashlib.sha1(state.encode()).hexdigest()[:16], probes=probes)


MC_INSTR = ["Which option is correct?", "Select the best answer.", "Which of the options is right?"]


def conv_medmcqa(rows, rng, lab):
    for r in rows:
        if str(r.get("choice_type")) != "single":
            lab.drops["multi_answer(skipped)"] += 1
            continue
        q = clean(r.get("question"))
        opts = [clean(r.get(k)) for k in ("opa", "opb", "opc", "opd")]
        cop = r.get("cop")
        if not q or not all(opts) or cop not in (0, 1, 2, 3) or len({norm(o) for o in opts}) < 4:
            lab.drops["bad_row"] += 1
            continue
        subj = clean(r.get("subject_name"))
        state = (f"Medical exam question ({subj}):\n{q}" if subj else f"Medical exam question:\n{q}")
        keys = ["a", "b", "c", "d"]
        crit = shuffled(dict(zip(keys, opts)), rng)
        yield make_record("medical/medmcqa", state, {"answer": choice_q(rng.choice(MC_INSTR), crit)},
                          {"answer": one_hot(keys[cop], list(crit))}, {"subject": subj or None, "orig_id": r.get("id")},
                          group=r.get("id"), probes=[q])


PUBMED_CRIT = {"yes": "Yes", "no": "No", "maybe": "Maybe: the evidence is mixed or inconclusive"}


def conv_pubmedqa(rows, rng, lab, max_chars):
    """pqa_labeled (expert yes/no/maybe; hard) and pqa_artificial (heuristic yes/no; noisy_hard). The abstract's
    conclusion (long_answer) is never shown."""
    for r in rows:
        q = clean(r.get("question"))
        ctx = r.get("context") or {}
        parts = [f"{str(lbl).upper()}: {clean(t)}" if lbl else clean(t)
                 for t, lbl in zip(ctx.get("contexts") or [], list(ctx.get("labels") or []) + [None] * 50)]
        gold = str(r.get("final_decision") or "").lower()
        if not q or not parts or gold not in PUBMED_CRIT:
            lab.drops["bad_row"] += 1
            continue
        state = f"Research question: {q}\n\nAbstract (conclusion withheld):\n" + "\n".join(parts)
        if len(state) > max_chars:
            lab.drops["too_long"] += 1
            continue
        crit = shuffled(dict(PUBMED_CRIT), rng)
        kind = "hard" if r.get("_config") == "pqa_labeled" else "noisy_hard"
        yield make_record("medical/pubmedqa", state,
                          {"answer": choice_q("According to this study, what is the answer to the research question?", crit)},
                          {"answer": one_hot(gold, list(crit))}, {"pubid": r.get("pubid"), "config": r.get("_config")},
                          group=r.get("pubid"), probes=[q], kinds={"answer": kind})


def mc_record(source, state, items, rng, meta, group, probes):
    """items: [(instructions, [option texts], gold index)] -> one record, one choice question per item."""
    qs, labs = {}, {}
    for i, (instr, opts, gold) in enumerate(items):
        keys = [chr(97 + j) for j in range(len(opts))]
        crit = shuffled(dict(zip(keys, opts)), rng)
        qid = "answer" if len(items) == 1 else f"q{i}"
        qs[qid], labs[qid] = choice_q(instr, crit), one_hot(keys[gold], list(crit))
    return make_record(source, state, qs, labs, meta, group=group, probes=probes)


def valid_opts(opts):
    return len(opts) >= 2 and all(opts) and len({norm(o) for o in opts}) == len(opts)


def conv_csqa(rows, rng, lab):
    for r in rows:
        q = clean(r.get("question"))
        ch = r.get("choices") or {}
        labels, texts = list(ch.get("label") or []), [clean(t) for t in ch.get("text") or []]
        gold = r.get("answerKey")
        if not q or gold not in labels or not valid_opts(texts):
            lab.drops["bad_row"] += 1
            continue
        yield mc_record("commonsense/csqa", q, [("Which answer is the most sensible?", texts, labels.index(gold))], rng,
                        {"concept": r.get("question_concept")}, group=r.get("id"), probes=[q])


def grouped_mc(rows, source, ctx_key, rng, lab, extract, max_q=3):
    """Rows sharing a context become one record with one question per row (SIQA, CosmosQA, BBQ)."""
    by = defaultdict(list)
    for r in rows:
        ctx = clean(r.get(ctx_key))
        if ctx:
            by[ctx].append(r)
        else:
            lab.drops["bad_row"] += 1
    for ctx, grp in by.items():
        items, seen = [], set()
        for r in grp:
            it = extract(r)
            if not it or not valid_opts(it[1]) or norm(it[0]) in seen:
                lab.drops["bad_row"] += 1
                continue
            seen.add(norm(it[0]))
            items.append(it)
        if items:
            items = items[:max_q]
            yield mc_record(source, ctx, items, rng, {"n_questions": len(items), **(grp[0].get("_meta") or {})},
                            group=norm(ctx)[:200], probes=[ctx] + [i[0] for i in items])


def siqa_item(r):
    try:
        gold = int(r.get("label")) - 1
    except (TypeError, ValueError):
        return None
    opts = [clean(r.get(k)) for k in ("answerA", "answerB", "answerC")]
    q = clean(r.get("question"))
    return (q, opts, gold) if q and 0 <= gold < 3 else None


def cosmos_item(r):
    try:
        gold = int(r.get("label"))
    except (TypeError, ValueError):
        return None
    opts = [clean(r.get(f"answer{i}")) for i in range(4)]
    q = clean(r.get("question"))
    return (q, opts, gold) if q and 0 <= gold < 4 else None


def bbq_item(r):
    opts = [clean(x) for x in r.get("choices") or []]
    q, gold = clean(r.get("question")), r.get("answer")
    return (q, opts, int(gold)) if q and isinstance(gold, int) and 0 <= gold < len(opts) else None


def conv_winogrande(rows, rng, lab):
    for r in rows:
        s, o1, o2, ans = clean(r.get("sentence")), clean(r.get("option1")), clean(r.get("option2")), str(r.get("answer"))
        if "_" not in s or ans not in ("1", "2") or not valid_opts([o1, o2]):
            lab.drops["bad_row"] += 1
            continue
        yield mc_record("commonsense/winogrande", s, [("Which option correctly fills the blank (_) in the sentence?", [o1, o2], int(ans) - 1)],
                        rng, {}, group=norm(s)[:200], probes=[s])


def conv_ecare(rows, rng, lab):
    for r in rows:
        prem, kind = clean(r.get("premise")), str(r.get("question") or "").strip().lower()
        c1, c2 = clean(r.get("choice1")), clean(r.get("choice2"))
        try:
            gold = int(r.get("label"))
        except (TypeError, ValueError):
            gold = -1
        if not prem or kind not in ("cause", "effect") or gold not in (0, 1) or not valid_opts([c1, c2]):
            lab.drops["bad_row"] += 1
            continue
        instr = "Which is the more plausible CAUSE of this event?" if kind == "cause" else "Which is the more plausible EFFECT of this event?"
        yield mc_record("commonsense/ecare", prem, [(instr, [c1, c2], gold)], rng, {"relation": kind},
                        group=r.get("idx") or norm(prem), probes=[prem])


# --------------------------------------------------------------------------- loaders (network) -> candidate records
def n_rows(a, source, factor=1.0):
    cap = cap_of(a, source)
    n = int(cap * a.oversample * factor) + 1
    return min(n, a.limit_rows) if a.limit_rows else n


def cap_of(a, source):
    return max(1, min(a.max_per_source, int(round(CAPS[source] * a.scale))))


def load_circa(a, rng, lab):
    rows = read_parquet(hf_file(a, ORIGINS["circa"], "data/train-00000-of-00001.parquet"))
    return conv_circa(sample(rows, n_rows(a, "circa"), rng), rng, lab)


def load_argq(a, rng, lab):
    rows = []
    for fn in ("train.csv", "dev.csv", "test.csv"):
        rows += read_csv(hf_file(a, ORIGINS["argq"], fn))
    return conv_argq(sample(rows, n_rows(a, "argq"), rng), rng, lab)


def load_scruples_anecdotes(a, rng, lab):
    rows = []
    for split in (("dev",) if a.small else ("train", "dev", "test")):
        for r in read_jsonl(hf_file(a, ORIGINS["scruples/anecdotes"], f"{split}.scruples-anecdotes.jsonl")):
            r["_split"] = split
            rows.append(r)
    return conv_scruples_anecdotes(sample(rows, n_rows(a, "scruples/anecdotes", 1.3), rng), rng, lab, a.max_state_chars)


def load_scruples_dilemmas(a, rng, lab):
    rows = []
    with tarfile.open(url_file(a, ORIGINS["scruples/dilemmas"]), "r:gz") as tf:
        for split in ("train", "dev", "test"):
            f = tf.extractfile(f"dilemmas/{split}.scruples-dilemmas.jsonl")
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    r["_split"] = split
                    rows.append(r)
    return conv_scruples_dilemmas(sample(rows, n_rows(a, "scruples/dilemmas"), rng), rng, lab)


def load_mtbench(a, rng, lab):
    files = [f for f in hf_list(a, ORIGINS["mtbench_human"]) if re.fullmatch(r"data/human-.*\.parquet", f)]
    if not files:
        raise RuntimeError("no data/human-*.parquet in lmsys/mt_bench_human_judgments")
    rows = [r for f in files for r in read_parquet(hf_file(a, ORIGINS["mtbench_human"], f))]
    return conv_mtbench(rows, rng, lab, a.max_state_chars)


def load_summeval(a, rng, lab):
    files = [f for f in hf_list(a, "mteb/summeval") if f.endswith(".parquet")]
    articles = {r["id"]: r["text"] for f in files for r in read_parquet(hf_file(a, "mteb/summeval", f), ["id", "text"])}
    rows = list(read_jsonl(url_file(a, ORIGINS["summeval"].split(" + ")[0])))
    return conv_summeval(sample(rows, n_rows(a, "summeval"), rng), articles, rng, lab, a.max_state_chars)


NUMPY_SCALAR_GLOBALS = {("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"), ("numpy", "dtype")}


class ScalarsOnly(pickle.Unpickler):
    """ConvoKit stores `Annotations` ({worker: rating}) as a pickle of dicts of numpy int64. Refuse every global except
    the numpy scalar / dtype constructors, so a tampered file cannot execute code."""

    def find_class(self, module, name):
        if (module, name) in NUMPY_SCALAR_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing global {module}.{name}")


def load_politeness(a, rng, lab, community):
    source = f"politeness/{community}"
    corpus = ORIGINS[source].rsplit("/", 1)[-1][: -len(".zip")]
    with zipfile.ZipFile(url_file(a, ORIGINS[source])) as z:
        utts = [json.loads(line) for line in z.read(f"{corpus}/utterances.jsonl").decode("utf-8").splitlines() if line.strip()]
        ann = ScalarsOnly(io.BytesIO(z.read(f"{corpus}/Annotations-bin.p"))).load()
    rows = []
    for u in utts:
        m = re.search(r"bin\{(\d+)\}", str((u.get("meta") or {}).get("Annotations")))
        if m and int(m.group(1)) < len(ann):
            rows.append((u, {str(k): int(v) for k, v in ann[int(m.group(1))].items()}))
        else:
            lab.drops["annotations_missing"] += 1
    return conv_politeness(sample(rows, n_rows(a, source), rng), community, rng, lab)


def casino_rows(a):
    return read_parquet(hf_file(a, ORIGINS["casino/outcomes"], "data/train-00000-of-00001.parquet"))


def load_casino_outcomes(a, rng, lab):
    return conv_casino_outcomes(sample(casino_rows(a), n_rows(a, "casino/outcomes"), rng), rng, lab)


def load_casino_strategies(a, rng, lab):
    return conv_casino_strategies(casino_rows(a), rng, lab)


CC_COLUMNS = ["name", "description", "source", "difficulty", "solutions", "incorrect_solutions", "cf_rating",
              "time_limit", "memory_limit_bytes"]


def load_code_contests(a, rng, lab):
    files = sorted(hf_list(a, ORIGINS["code_contests"]))
    if a.small:
        shards = [f for f in files if f.startswith("data/valid-")]
    else:
        train = [f for f in files if f.startswith("data/train-")]
        k = max(1, min(a.code_shards, len(train)))
        shards = [train[round(i * (len(train) - 1) / max(1, k - 1))] for i in range(k)] if k > 1 else train[:1]
        shards = sorted(set(shards))
    problems = []
    for f in shards:
        problems += read_parquet(hf_file(a, ORIGINS["code_contests"], f), CC_COLUMNS)
    per_problem = 4
    problems = sample(problems, max(1, n_rows(a, "code_contests") // per_problem + 1), rng)
    return conv_code_contests(problems, rng, lab, a.max_state_chars, per_side=per_problem // 2)


def load_tabfact(a, rng, lab):
    fn = "data/validation-00000-of-00001.parquet" if a.small else "data/train-00000-of-00001.parquet"
    rows = read_parquet(hf_file(a, ORIGINS["tabfact"], fn), ["table_title", "table", "question", "answer"])
    by = defaultdict(list)
    for r in rows:
        by[(r["table_title"], r["table"])].append(r)
    keys = sample(sorted(by, key=lambda k: hashlib.sha1(repr(k).encode()).hexdigest()), n_rows(a, "tabfact"), rng)
    return conv_tabfact([r for k in keys for r in by[k]], rng, lab, a.max_state_chars)


def load_medmcqa(a, rng, lab):
    fn = "data/validation-00000-of-00001.parquet" if a.small else "data/train-00000-of-00001.parquet"
    cols = ["id", "question", "opa", "opb", "opc", "opd", "cop", "choice_type", "subject_name"]
    rows = [r for r in read_parquet(hf_file(a, ORIGINS["medical/medmcqa"], fn), cols) if r.get("choice_type") == "single"]
    return conv_medmcqa(sample(rows, n_rows(a, "medical/medmcqa"), rng), rng, lab)


def load_pubmedqa(a, rng, lab):
    import pyarrow.parquet as pq
    cols = ["pubid", "question", "context", "final_decision"]
    n = n_rows(a, "medical/pubmedqa")
    labeled = sample(read_parquet(hf_file(a, ORIGINS["medical/pubmedqa"], "pqa_labeled/train-00000-of-00001.parquet"), cols), n, rng)
    for r in labeled:
        r["_config"] = "pqa_labeled"
    art, k = [], max(0, n - len(labeled))
    if k and not a.small:   # 211k rows: sample row indices before materializing Python objects
        t = pq.read_table(hf_file(a, ORIGINS["medical/pubmedqa"], "pqa_artificial/train-00000-of-00001.parquet"), columns=cols)
        art = t.take(sorted(rng.sample(range(t.num_rows), min(k, t.num_rows)))).to_pylist()
        for r in art:
            r["_config"] = "pqa_artificial"
    return conv_pubmedqa(labeled + art, rng, lab, a.max_state_chars)


def load_csqa(a, rng, lab):
    rows = read_parquet(hf_file(a, ORIGINS["commonsense/csqa"], "data/train-00000-of-00001.parquet"))
    return conv_csqa(sample(rows, n_rows(a, "commonsense/csqa"), rng), rng, lab)


def load_siqa(a, rng, lab):
    fn = "social_iqa_validation.parquet" if a.small else "social_iqa_train.parquet"
    rows = read_parquet(hf_file(a, ORIGINS["commonsense/siqa"], fn))
    return grouped_mc(sample(rows, n_rows(a, "commonsense/siqa"), rng), "commonsense/siqa", "context", rng, lab, siqa_item)


def load_cosmosqa(a, rng, lab):
    fn = "data/validation-00000-of-00001.parquet" if a.small else "data/train-00000-of-00001.parquet"
    rows = read_parquet(hf_file(a, ORIGINS["commonsense/cosmosqa"], fn))
    return grouped_mc(sample(rows, n_rows(a, "commonsense/cosmosqa", 1.5), rng), "commonsense/cosmosqa", "context", rng, lab, cosmos_item)


def load_winogrande(a, rng, lab):
    fn = "winogrande_xl/validation-00000-of-00001.parquet" if a.small else "winogrande_xl/train-00000-of-00001.parquet"
    rows = read_parquet(hf_file(a, ORIGINS["commonsense/winogrande"], fn))
    return conv_winogrande(sample(rows, n_rows(a, "commonsense/winogrande"), rng), rng, lab)


def load_ecare(a, rng, lab):
    rows = read_csv(hf_file(a, ORIGINS["commonsense/ecare"], "validation.csv" if a.small else "train.csv"))
    return conv_ecare(sample(rows, n_rows(a, "commonsense/ecare"), rng), rng, lab)


def load_bbq(a, rng, lab):
    rows = []
    for f in sorted(f for f in hf_list(a, ORIGINS["bbq"]) if f.startswith("data/") and f.endswith(".parquet")):
        got = read_parquet(hf_file(a, ORIGINS["bbq"], f))
        for r in got:
            r["_meta"] = {"category": r.get("category")}
        rows += sample(got, n_rows(a, "bbq", 2.0 / 11), rng)   # ~even across the 11 categories, 2 questions per context
    return grouped_mc(rows, "bbq", "context", rng, lab, bbq_item)


LOADERS = {
    "circa": load_circa, "argq": load_argq,
    "scruples/anecdotes": load_scruples_anecdotes, "scruples/dilemmas": load_scruples_dilemmas,
    "mtbench_human": load_mtbench, "summeval": load_summeval,
    "politeness/wikipedia": lambda a, rng, lab: load_politeness(a, rng, lab, "wikipedia"),
    "politeness/stackexchange": lambda a, rng, lab: load_politeness(a, rng, lab, "stackexchange"),
    "casino/outcomes": load_casino_outcomes, "casino/strategies": load_casino_strategies,
    "code_contests": load_code_contests, "tabfact": load_tabfact,
    "medical/medmcqa": load_medmcqa, "medical/pubmedqa": load_pubmedqa,
    "commonsense/csqa": load_csqa, "commonsense/siqa": load_siqa, "commonsense/cosmosqa": load_cosmosqa,
    "commonsense/winogrande": load_winogrande, "commonsense/ecare": load_ecare,
    "bbq": load_bbq,
}
assert set(LOADERS) == set(CAPS) == set(ORIGINS)


# --------------------------------------------------------------------------- fingerprints
def flatten(state):
    """(flat text, [string leaves]) for a state: a JSON object / list / string becomes its string leaves joined by
    newlines (keys dropped), so an eval state stored as json.dumps({...}) is compared on its text, not its syntax."""
    raw = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    obj = None
    t = raw.strip()
    if t[:1] in ("{", "[", '"'):
        try:
            obj = json.loads(t)
        except ValueError:
            obj = None
    if obj is None and not isinstance(state, str):
        obj = state
    if obj is None:
        return raw, [raw]
    leaves = []

    def walk(o):
        if isinstance(o, str):
            if o.strip():
                leaves.append(o)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif o is not None and not isinstance(o, bool):
            leaves.append(str(o))
    walk(obj)
    return ("\n".join(leaves) if leaves else raw), leaves


def shingle_hashes(normed, n=SHINGLE_WORDS):
    """{hash: shingle} for the 8-word shingles of an already normalized text (build_jev_ext.shingles' definition)."""
    w = WORD.findall(normed)
    return {hash(s): s for s in (" ".join(w[i:i + n]) for i in range(len(w) - n + 1))}


class Index:
    """Exact (whole state), leaf (one string field) and 8-word-shingle fingerprints of many states. Shingles live in
    sorted numpy arrays (hash -> state id), so hundreds of thousands of eval / corpus states fit in a few hundred MB.
    Tags have a class, 'eval' or 'corpus'; when a candidate matches both, the eval tag is reported."""

    def __init__(self):
        self.names, self.klass, self.sizes, self.paths = [], [], Counter(), defaultdict(list)
        self.whole, self.leaf, self.seen = {}, {}, set()
        self.state_tag, self._h, self._s = [], [], []
        self.H = self.S = self.ST = None

    def tag(self, name, klass):
        if name not in self.names:
            self.names.append(name)
            self.klass.append(klass)
        return self.names.index(name)

    def _put(self, d, key, tid):
        old = d.get(key)
        if old is None or (self.klass[old] != "eval" and self.klass[tid] == "eval"):
            d[key] = tid

    def add(self, tid, state):
        import numpy as np
        if state is None:
            return False
        raw = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        flat, leaves = flatten(state)
        nf = norm(flat)
        if not nf or (tid, hash(nf)) in self.seen:
            return False
        self.seen.add((tid, hash(nf)))
        self._put(self.whole, hash(nf), tid)
        self._put(self.whole, hash(norm(raw)), tid)
        for lf in leaves:
            nl = norm(lf)
            if len(nl.split()) >= LEAF_MIN_WORDS:
                self._put(self.leaf, hash(nl), tid)
        hs = shingle_hashes(nf)
        if hs:
            sid = len(self.state_tag)
            self.state_tag.append(tid)
            self._h.append(np.fromiter(hs, dtype=np.int64, count=len(hs)))
            self._s.append(np.full(len(hs), sid, dtype=np.int32))
        self.sizes[self.names[tid]] += 1
        return True

    def add_rows(self, name, klass, rows, path=None):
        tid, n = self.tag(name, klass), 0
        for st in rows:
            n += bool(self.add(tid, st))
        if path:
            self.paths[name].append(str(path))
        return n

    def freeze(self):
        import numpy as np
        if self._h:
            H, S = np.concatenate(self._h), np.concatenate(self._s)
            order = np.argsort(H, kind="stable")
            self.H, self.S = H[order], S[order]
        else:
            self.H, self.S = np.zeros(0, np.int64), np.zeros(0, np.int32)
        self.ST = np.asarray(self.state_tag, dtype=np.int32)
        self._h, self._s = [], []
        print(f"fingerprints: {sum(self.sizes.values())} states under {len(self.names)} tags, "
              f"{len(self.H):,} shingle entries, {len(self.leaf):,} leaf keys", flush=True)

    def _shingle(self, nf):
        """('shingle', tag id, n_shared, 2 shared shingles) for the best fingerprint state sharing >= 3 shingles."""
        import numpy as np
        hs = shingle_hashes(nf)
        if len(hs) < SHINGLE_MIN_SHARED or not len(self.H):
            return None
        keys = np.fromiter(hs, dtype=np.int64, count=len(hs))
        lo, hi = np.searchsorted(self.H, keys, "left"), np.searchsorted(self.H, keys, "right")
        nz = np.nonzero(hi > lo)[0]
        if len(nz) < SHINGLE_MIN_SHARED:
            return None
        ids = np.concatenate([self.S[lo[i]:hi[i]] for i in nz])
        u, c = np.unique(ids, return_counts=True)
        sel = c >= SHINGLE_MIN_SHARED
        if not sel.any():
            return None
        sid, n = min(zip(u[sel].tolist(), c[sel].tolist()), key=lambda x: (self.klass[self.ST[x[0]]] != "eval", -x[1]))
        shared = [hs[int(keys[i])] for i in nz if (self.S[lo[i]:hi[i]] == sid).any()][:2]
        return ("shingle", int(self.ST[sid]), n, shared)

    def check(self, state, probes=()):
        """None, or (kind, tag name, n_shared, sample). Every kind is tried until an EVAL tag matches, so a candidate
        that overlaps both an eval set and a training corpus is reported against the eval set."""
        raw = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        flat, _ = flatten(state)
        nf = norm(flat)
        found = [("exact", self.whole[k], None, None) for k in {hash(nf), hash(norm(raw))} if k in self.whole]
        for p in probes:
            np_ = norm(p)
            k = hash(np_)
            if len(np_.split()) < PROBE_MIN_WORDS:
                continue   # a stock reply ("ok", "sure") equal to a one-word eval item is not a shared item
            if k in self.whole:
                found.append(("exact", self.whole[k], None, [np_[:120]]))
            elif len(np_.split()) >= LEAF_MIN_WORDS and k in self.leaf:
                found.append(("leaf", self.leaf[k], None, [np_[:120]]))
        if not any(self.klass[f[1]] == "eval" for f in found):
            sh = self._shingle(nf)
            if sh:
                found.append(sh)
        if not found:
            return None
        order = {"exact": 0, "leaf": 1, "shingle": 2}
        kind, tid, n, smp = min(found, key=lambda f: (self.klass[f[1]] != "eval", order[f[0]]))
        return (kind, self.names[tid], n, smp)


def jsonl_states(path):
    """States from one jsonl file or every *.jsonl under a directory. Prediction files (Jev / TypeSafe outputs, e.g.
    jev_predictions.jsonl or a cached results/*/test_predictions.jsonl) are never opened."""
    p = Path(path)
    files = [p] if p.is_file() else sorted(q for q in p.rglob("*.jsonl"))
    for f in files:
        if "prediction" in f.name.lower() or "results" in f.parts:
            continue
        for row in read_jsonl(f):
            if isinstance(row, dict) and row.get("state") is not None:
                yield row["state"]


# ---- eval STATE fetchers: each yields (tag, state); label fields are never kept
JEVBENCH_CONFIGS = ["arc_challenge", "banking77", "boolq", "chaosnli", "civil_comments", "clinc150", "fever_evidence",
                    "go_emotions", "helpsteer2_helpfulness", "helpsteer2_verbosity", "ledgar", "massive",
                    "measuring_hate_speech", "mmlu", "mnli", "paws", "sms_spam", "sst5", "strategyqa_closed",
                    "strategyqa_grounded", "stsb", "yelp5"]
LABELISH = {"answer", "label", "labels", "gold", "index", "idx", "id", "target", "y"}


def ev_jevbench(a):
    for cfg in JEVBENCH_CONFIGS:
        for row in read_jsonl(hf_file(a, cx.JEVBENCH, f"data/{cfg}/test.jsonl", cx.JEVBENCH_V011)):
            yield f"jevbench/{cfg}", row.get("state")


def ev_typed(a):
    for split in ("train", "test"):
        for r in read_parquet(hf_file(a, "LocalLLaMA/typed-decisions", f"all/{split}-00000-of-00001.parquet"), ["state"]):
            yield f"typed_decisions/{split}", r["state"]


def ev_tdb(a):
    repo, rev = cx.TDB
    for suite in cx.TDB_SUITES:
        for row in read_jsonl(hf_file(a, repo, f"bench/suites/{suite}/items.jsonl", rev)):
            if row.get("state") is not None:
                yield f"tdb/{suite}", row["state"]
    # the 359 withheld items are rebuilt from these two tables: fingerprint the tables' text in full (a superset)
    repo, rev, path = cx.TRIALGPT_PARQUET
    for r in read_parquet(hf_file(a, repo, path, rev), ["note", "criterion_text"]):
        yield "tdb/withheld_trialgpt_notes", r["note"]
        yield "tdb/withheld_trialgpt_criteria", r["criterion_text"]
    repo, rev = cx.ABTBUY_PARQUET
    for path in ("source/source/0000.parquet", "target/target/0000.parquet"):
        for r in read_parquet(hf_file(a, repo, path, rev), ["name", "description"]):
            yield "tdb/withheld_abtbuy", {"name": r.get("name"), "description": r.get("description")}


def ev_mmlu_arc(a):
    for r in read_parquet(hf_file(a, "cais/mmlu", "all/test-00000-of-00001.parquet"), ["question", "choices"]):
        yield "mmlu/test", {"question": r["question"], "choices": list(r["choices"] or [])}
    for cfg in ("ARC-Challenge", "ARC-Easy"):
        for r in read_parquet(hf_file(a, "allenai/ai2_arc", f"{cfg}/test-00000-of-00001.parquet"), ["question", "choices"]):
            yield f"arc/{cfg}", {"question": r["question"], "choices": list((r["choices"] or {}).get("text") or [])}


def ev_legalbench(a):
    repo, rev = cx.LEGALBENCH_HF
    files = sorted(f for f in hf_list(a, repo, rev) if re.fullmatch(r"data/[^/]+/test\.tsv", f))
    if len(files) < 150:
        raise RuntimeError(f"expected ~162 LegalBench test files, found {len(files)}")
    for f in files:
        for row in read_csv(hf_file(a, repo, f, rev), delimiter="\t"):
            text = {k: v for k, v in row.items() if k and k.strip().lower() not in LABELISH and v and not re.fullmatch(r"[\d.\-]+", v.strip())}
            if text:
                yield "legalbench/test", text


def ev_community(a):
    repo, rev, fn = cx.PHISHNCHIPS
    for row in read_csv(hf_file(a, repo, fn, rev)):
        try:
            email = json.loads(row["email_content"])
        except (KeyError, ValueError):
            continue
        yield "community/phishnchips", {k: email.get(k) for k in cx.EMAIL_FIELDS}
    gh = lambda key, path: url_file(a, cx.GH.format(repo=cx.PINS[key][0], rev=cx.PINS[key][1], path=path))  # noqa: E731
    for t in read_jsonl(gh("jev-benchmark", "tasks.jsonl")):
        yield "community/tool_risk", t.get("state")
    c2 = cx.literals(Path(gh("jev-measured", "bench/labelled.py")).read_text(encoding="utf-8"), "UNAMBIGUOUS", "AMBIGUOUS")
    for t, _gold in c2["UNAMBIGUOUS"]:
        yield "community/ticket_routing", t
    for t in c2["AMBIGUOUS"]:
        yield "community/ticket_routing", t if isinstance(t, str) else t[0]


DICES_BASE = "https://raw.githubusercontent.com/google-research-datasets/dices-dataset/main/"
DICES_FILES = ["350/diverse_safety_adversarial_dialog_350.csv", "990/diverse_safety_adversarial_dialog_990.csv"]


def ev_dices(a):
    csv.field_size_limit(1 << 30)
    for path in DICES_FILES:
        seen = set()
        with open(url_file(a, DICES_BASE + path), encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("item_id") in seen:
                    continue
                seen.add(row.get("item_id"))
                yield f"dices/{path.split('/')[0]}", {"conversation": row.get("context"), "chatbot_response": row.get("response")}


def ev_chaosnli(a):
    for fn in ("raw/chaosNLI_snli.jsonl", "raw/chaosNLI_mnli_m.jsonl", "raw/chaosNLI_alphanli.jsonl"):
        for r in read_jsonl(hf_file(a, "earino/chaosnli", fn)):
            ex = r.get("example") or {}
            keep = {k: ex.get(k) for k in ("premise", "hypothesis", "obs1", "obs2", "hyp1", "hyp2") if ex.get(k)}
            if keep:
                yield f"chaosnli/{fn.split('_', 1)[1].split('.')[0]}", keep


EVAL_FETCHERS = {"jevbench": ev_jevbench, "typed_decisions": ev_typed, "tdb": ev_tdb, "mmlu_arc": ev_mmlu_arc,
                 "legalbench": ev_legalbench, "community": ev_community, "dices": ev_dices, "chaosnli": ev_chaosnli}


def load_fingerprints(a):
    idx, coverage = Index(), {}
    for spec, klass in [(s, "eval") for s in a.eval] + [(s, "corpus") for s in a.fingerprint]:
        tag, _, p = spec.partition("=")
        if not p:
            tag, p = Path(spec).stem, spec
        if not Path(p).exists():
            print(f"fingerprints: {p} not found (tag {tag}), skipping (jev_data / jev_ext_strict are checked below)")
            continue
        n = idx.add_rows(f"{klass}:{tag}" if klass == "corpus" else f"given:{tag}", klass, jsonl_states(p), p)
        print(f"fingerprints: {n:7d} states from {p} ({klass} {tag})", flush=True)
    missing_corpora = [c for c in ("jev_data", "jev_ext_strict") if f"corpus:{c}" not in idx.sizes]
    if missing_corpora and not a.allow_missing_corpora:
        sys.exit(f"no --fingerprint for {missing_corpora}: the build must be deduplicated against the existing corpora "
                 "(or pass --allow-missing-corpora)")
    if a.fetch_evals:
        for fam, fn in EVAL_FETCHERS.items():
            if fam in a.skip_evals:
                coverage[fam] = "skipped by --skip-evals"
                continue
            t0, per = time.time(), Counter()
            try:
                for tag, st in fn(a):
                    per[tag] += bool(idx.add(idx.tag(f"eval:{tag}", "eval"), st))
                coverage[fam] = dict(per)
                print(f"fingerprints: eval {fam:16s} {sum(per.values()):7d} states in {time.time() - t0:5.1f}s {dict(per)}", flush=True)
            except Exception as e:  # noqa: BLE001
                coverage[fam] = f"FAILED {type(e).__name__}: {str(e)[:300]}"
                print(f"fingerprints: eval {fam} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
    else:
        coverage = {fam: "not fetched (--no-fetch-evals)" for fam in EVAL_FETCHERS}
    missing = [f for f, v in coverage.items() if not isinstance(v, dict)]
    if missing and not a.allow_missing_evals:
        sys.exit(f"eval families not covered: {missing}; fix the network / pass --eval files, or --allow-missing-evals")
    if not any(k == "eval" for k in idx.klass):
        sys.exit("no eval fingerprints at all: refusing to build")
    idx.freeze()
    return idx, coverage


# --------------------------------------------------------------------------- pipeline
def new_stats(source, lab):
    return {"dataset": ORIGINS[source], "license": license_of(source), "cap": None, "candidates": 0,
            "written": Counter(), "units": Counter(), "label_kinds": Counter(), "qtypes": Counter(), "drops": Counter(),
            "dedup_exact": Counter(), "dedup_leaf": Counter(), "dedup_shingle": Counter(), "question_drops": lab.drops,
            "overlap_examples": [], "errors": [], "hf_sha": None}


def process(source, cands, cap, idx, writer, rng, st, seen_states, max_chars):
    """validate -> length -> cross-source exact dedup -> fingerprint dedup -> cap -> write. Every candidate is checked,
    also after the cap is reached, so the overlap counts describe all candidates."""
    st["candidates"], st["cap"] = len(cands), cap
    rng.shuffle(cands)
    kept = 0
    for rec in cands:
        probes = rec.pop("_probes", [])
        try:
            validate(rec)
            for qid, q in rec["questions"].items():
                assert 2 <= len(rec["labels"][qid]) <= MAX_K, f"{qid}: K={len(rec['labels'][qid])}"
        except AssertionError as e:
            st["drops"]["invalid"] += 1
            if len(st["errors"]) < 5:
                st["errors"].append(f"invalid: {e}")
            continue
        if len(rec["state"]) > max_chars:
            st["drops"]["too_long"] += 1
            continue
        ns = hash(norm(rec["state"]))
        if ns in seen_states:
            st["drops"]["duplicate_state"] += 1
            continue
        seen_states.add(ns)
        hit = idx.check(rec["state"], probes) if idx else None
        if hit:
            kind, tag, n_shared, sample_ = hit
            st["dedup_" + kind][tag] += 1
            if len(st["overlap_examples"]) < 4:
                st["overlap_examples"].append({"record": rec["id"], "kind": kind, "tag": tag, "shared": n_shared, "sample": sample_})
            continue
        if kept >= cap:
            st["drops"]["over_cap"] += 1
            continue
        split = split_of(str(rec["meta"].get("group") or rec["state"]), val_pct=VAL_PCT, test_pct=0)
        rec["split"] = split
        writer.write(split, source, rec)
        kept += 1
        st["written"][split] += 1
        st["units"][split] += len(rec["questions"])
        st["label_kinds"].update(rec["label_kind"].values())
        st["qtypes"].update(q["type"] for q in rec["questions"].values())
    return st


def report_source(source, st):
    w = st["written"]
    dd = {k: dict(st[f"dedup_{k}"]) for k in ("exact", "leaf", "shingle") if st[f"dedup_{k}"]}
    print(f"[{source}] candidates {st['candidates']}, written train {w.get('train', 0)} / val {w.get('val', 0)} "
          f"(cap {st['cap']}); overlap drops {dd or 0}; other drops {dict(st['drops'])}", flush=True)
    if st["question_drops"]:
        print(f"[{source}]   row/question-level: {dict(st['question_drops'])}")
    for ex in st["overlap_examples"][:2]:
        print(f"[{source}]   overlap example {ex}")
    for e in st["errors"]:
        print(f"[{source}]   {e}")


def build(a):
    out = Path(a.out)
    stale = [p for sp in ("train", "val", "test") for p in (out / sp).glob("*.jsonl")]
    if stale or (out / "manifest.json").exists():
        sys.exit(f"{out} already holds a corpus ({len(stale)} jsonl files); write to a new --out")
    t0 = time.time()
    idx, coverage = load_fingerprints(a)
    writer, seen = Writer(out), set()
    manifest = {
        "builder": "build_jev_ext2.py", "seed": a.seed, "small": a.small, "limit_rows": a.limit_rows,
        "splits": {"train": 100 - VAL_PCT, "val": VAL_PCT, "by": "sha1 of the normalized per-source group key (meta.group)"},
        "caps": {s: cap_of(a, s) for s in a.sources}, "max_state_chars": a.max_state_chars, "oversample": a.oversample,
        "dedup": {"normalization": "analysis/claude_overlap.py norm", "shingle_words": SHINGLE_WORDS,
                  "shingle_min_shared_with_one_state": SHINGLE_MIN_SHARED, "leaf_min_words": LEAF_MIN_WORDS,
                  "fingerprint_states": dict(idx.sizes), "fingerprint_paths": dict(idx.paths),
                  "eval_coverage": coverage,
                  "not_fetched_note": "tool_risk / ticket_routing / PhishNChips come from their pinned sources; LegalBench is "
                                      "every task's test.tsv (text columns); TDB's withheld items are covered by their full "
                                      "source tables (TrialGPT notes + criteria, Abt-Buy records)."},
        "licenses": {s: license_of(s) for s in a.sources}, "sources": {},
    }
    for source in a.sources:
        rng, lab = random.Random(f"{a.seed}:{source}"), Labeler(a.renorm_tol)
        st = new_stats(source, lab)
        print(f"[{source}] loading...", flush=True)
        try:
            cands = list(LOADERS[source](a, rng, lab))
            process(source, cands, cap_of(a, source), idx, writer, rng, st, seen, a.max_state_chars)
        except Exception as e:  # noqa: BLE001
            st["errors"].append(f"FAILED {type(e).__name__}: {e}")
            print(f"[{source}] FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
        if not a.small:
            st["hf_sha"] = hf_sha(a, ORIGINS[source]) if "/" in ORIGINS[source] and "://" not in ORIGINS[source] else None
        report_source(source, st)
        manifest["sources"][source] = plain(st)
    writer.close()
    out.mkdir(parents=True, exist_ok=True)
    manifest["readback"] = final_check(out)
    manifest["seconds"] = round(time.time() - t0, 1)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    print_summary(manifest)
    failed = [s for s, v in manifest["sources"].items() if any(e.startswith("FAILED") for e in v["errors"])]
    print(f"manifest written to {out / 'manifest.json'}" + (f"; FAILED sources: {failed}" if failed else ""))
    return 1 if failed else 0


def print_summary(m):
    print("\noverlap drops (records) by kind and fingerprint tag:")
    tot = Counter()
    for s, st in m["sources"].items():
        row = {k: st[f"dedup_{k}"] for k in ("exact", "leaf", "shingle") if st[f"dedup_{k}"]}
        print(f"  {s:28s} {row if row else 'none'}")
        for k in ("exact", "leaf", "shingle"):
            for tag, n in st[f"dedup_{k}"].items():
                tot[f"{k}:{tag}"] += n
    print(f"  total: {dict(sorted(tot.items()))}")
    print("\nwritten records per source (train / val):")
    for s, st in m["sources"].items():
        w = st["written"]
        print(f"  {s:28s} {w.get('train', 0):6d} / {w.get('val', 0):5d}   {st['license']}")
    tt = sum(st["written"].get("train", 0) for st in m["sources"].values())
    tv = sum(st["written"].get("val", 0) for st in m["sources"].values())
    print(f"  {'TOTAL':28s} {tt:6d} / {tv:5d}")


# --------------------------------------------------------------------------- self-test (offline)
def self_test():
    """Every converter on hand-made rows shaped like the real ones, the fingerprint index (exact / leaf / shingle,
    eval priority), then the whole pipeline into a temp dir and the iter_units read-back. No network."""
    rng, lab = random.Random(0), Labeler(0.01)
    long_arg = "school uniforms remove visible differences in wealth so that children are judged by what they do and say"
    chat = [{"text": "Hello! What do you need most for the trip?", "id": "mturk_agent_1", "task_data": {}},
            {"text": "We really need water, it is very hot where we camp.", "id": "mturk_agent_2", "task_data": {}},
            {"text": "Submit-Deal", "id": "mturk_agent_1", "task_data": {"issue2youget": {"Food": "2", "Water": "1", "Firewood": "3"},
                                                                         "issue2theyget": {"Food": "1", "Water": "2", "Firewood": "0"}}},
            {"text": "Accept-Deal", "id": "mturk_agent_2", "task_data": {}}]
    pinfo = {a_: {"value2issue": {"High": h, "Medium": "Food", "Low": lo},
                  "outcomes": {"satisfaction": "Slightly satisfied", "opponent_likeness": "Extremely like"}}
             for a_, h, lo in (("mturk_agent_1", "Firewood", "Water"), ("mturk_agent_2", "Water", "Firewood"))}
    fx = {
        "circa": list(conv_circa([
            {"context": "X wants to know about Y's food preferences.", "question-X": "Do you like spicy food?",
             "answer-Y": "I put hot sauce on everything.", "judgements": "Yes#Yes#Probably yes / sometimes yes#Yes#I am not sure how X will interpret Y’s answer"},
            {"context": "c", "question-X": "q?", "answer-Y": "a", "judgements": "NA#I am not sure how X will interpret Y’s answer#Yes"}], rng, lab)),
        "argq": list(conv_argq([{"argument": long_arg, "topic": "We should adopt school uniforms", "set": "train",
                                 "WA": "0.8", "MACE-P": "0.9", "stance_WA": "-1", "stance_WA_conf": "1.0"}], rng, lab)),
        "scruples/anecdotes": list(conv_scruples_anecdotes([{
            "id": "x1", "post_id": "p1", "title": "AITA for eating my roommate's leftovers?", "text": "They were in the fridge for a week.",
            "label_scores": {"AUTHOR": 5, "OTHER": 1, "EVERYBODY": 0, "NOBODY": 1, "INFO": 0}, "post_type": "HISTORICAL"}], rng, lab, 6500)),
        "scruples/dilemmas": list(conv_scruples_dilemmas([{
            "id": "d1", "actions": [{"description": "borrowing a hook"}, {"description": "yelling at a waiter"}],
            "gold_annotations": [1, 4], "human_perf_annotations": [0, 5]}], rng, lab)),
        "mtbench_human": list(conv_mtbench([
            {"question_id": 81, "model_a": "m1", "model_b": "m2", "winner": w, "judge": j, "turn": 1,
             "conversation_a": [{"role": "user", "content": "Write a haiku about rain."}, {"role": "assistant", "content": "Rain taps the window"}],
             "conversation_b": [{"role": "user", "content": "Write a haiku about rain."}, {"role": "assistant", "content": "Wet."}]}
            for w, j in (("model_a", "e1"), ("model_a", "e2"), ("tie", "e3"))], rng, lab, 6500)),
        "summeval": list(conv_summeval([{"id": "cnn-1", "decoded": "a man won the lottery .", "model_id": "M8",
                                         "expert_annotations": [{"coherence": 2, "consistency": 5, "fluency": 4, "relevance": 3}] * 2 +
                                                               [{"coherence": 3, "consistency": 5, "fluency": 5, "relevance": 3}]}],
                                       {"cnn-1": "(CNN) A man from Ohio won the lottery on Tuesday."}, rng, lab, 6500)),
        "politeness/wikipedia": list(conv_politeness([({"id": "7", "text": "Could you please take a look at my edit?"},
                                                       {"w1": 18, "w2": 22, "w3": 13, "w4": 20, "w5": 25})], "wikipedia", rng, lab)),
        "casino/outcomes": list(conv_casino_outcomes([{"chat_logs": chat, "participant_info": pinfo, "annotations": []}], rng, lab)),
        "casino/strategies": list(conv_casino_strategies([{"chat_logs": chat, "participant_info": pinfo,
                                                           "annotations": [["Hello! What do you need most for the trip?", "small-talk,elicit-pref"],
                                                                           ["We really need water, it is very hot where we camp.", "self-need"]]}], rng, lab)),
        "code_contests": list(conv_code_contests([{"name": "1A. Theatre Square", "description": "Count flagstones needed.",
                                                   "source": 2, "difficulty": 7, "cf_rating": 1000,
                                                   "time_limit": {"seconds": 1, "nanos": 0}, "memory_limit_bytes": 256000000,
                                                   "solutions": {"language": [3, 1], "solution": ["n,m,a=map(int,input().split())\nx=-(-n//a)\nprint(x*-(-m//a))", "print 1\nprint 2\nprint 3"]},
                                                   "incorrect_solutions": {"language": [2], "solution": ["#include <cstdio>\nint main(){\nprintf(\"1\");}"]}}],
                                                 rng, lab, 6500)),
        "tabfact": list(conv_tabfact([{"table_title": "turkish cup", "table": "round#clubs\nfirst round#156\nfinal#2\n",
                                       "question": q, "answer": ans} for q, ans in (("the final has 2 clubs", "Entailed"),
                                                                                     ("the first round has 5 clubs", "Refuted"))], rng, lab, 6500)),
        "medical/medmcqa": list(conv_medmcqa([{"id": "m1", "question": "Which vitamin deficiency causes scurvy?", "opa": "Vitamin A",
                                               "opb": "Vitamin B12", "opc": "Vitamin C", "opd": "Vitamin D", "cop": 2,
                                               "choice_type": "single", "subject_name": "Biochemistry"},
                                              {"id": "m2", "question": "x", "opa": "a", "opb": "b", "opc": "c", "opd": "d", "cop": 0, "choice_type": "multi"}], rng, lab)),
        "medical/pubmedqa": list(conv_pubmedqa([{"pubid": 1, "question": "Does coffee improve memory?", "_config": "pqa_labeled",
                                                 "context": {"contexts": ["We tested 40 adults.", "Recall improved slightly."], "labels": ["METHODS", "RESULTS"]},
                                                 "final_decision": "maybe"}], rng, lab, 6500)),
        "commonsense/csqa": list(conv_csqa([{"id": "c1", "question": "Where do you keep milk cold?", "question_concept": "milk",
                                             "choices": {"label": list("ABCDE"), "text": ["fridge", "oven", "desk", "car", "sock"]}, "answerKey": "A"}], rng, lab)),
        "commonsense/siqa": list(grouped_mc([{"context": "Tracy stayed out late.", "question": "What will Tracy want next?",
                                              "answerA": "sleep", "answerB": "run a marathon", "answerC": "eat sand", "label": "1"},
                                             {"context": "Tracy stayed out late.", "question": "How would Riley feel?",
                                              "answerA": "worried", "answerB": "bored", "answerC": "thrilled", "label": "1"}],
                                            "commonsense/siqa", "context", rng, lab, siqa_item)),
        "commonsense/cosmosqa": list(grouped_mc([{"context": "I missed the bus again today.", "question": "Why was I late?",
                                                  "answer0": "I missed the bus", "answer1": "I was early", "answer2": "It rained",
                                                  "answer3": "None of the above choices .", "label": 0}],
                                                "commonsense/cosmosqa", "context", rng, lab, cosmos_item)),
        "commonsense/winogrande": list(conv_winogrande([{"sentence": "Sarah was a better surgeon than Maria so _ got the easier cases.",
                                                         "option1": "Sarah", "option2": "Maria", "answer": "2"}], rng, lab)),
        "commonsense/ecare": list(conv_ecare([{"idx": "e1", "premise": "The ice melted.", "question": "cause",
                                               "choice1": "The sun came out.", "choice2": "It snowed.", "label": "0"}], rng, lab)),
        "bbq": list(grouped_mc([{"context": "A grandson and his grandfather were booking a cab.", "question": "Who was not comfortable with the phone?",
                                 "choices": ["The grandfather", "The grandson", "Can't be determined"], "answer": 2, "category": "Age",
                                 "_meta": {"category": "Age"}}], "bbq", "context", rng, lab, bbq_item)),
    }
    for name, recs in fx.items():
        assert recs, f"{name}: no records"
        for r in recs:
            validate(r)
        print(f"ok  {name:26s} {len(recs)} records, {sum(len(r['questions']) for r in recs)} questions")
    c = fx["circa"][0]
    assert len(fx["circa"]) == 1 and lab.drops["too_few_votes"] == 1, "circa: 'not sure' / NA votes must be dropped"
    assert abs(c["labels"]["interpretation"]["yes"] - 0.75) < 1e-9 and abs(c["labels"]["means_yes"]["yes"] - 1.0) < 1e-9
    assert c["meta"]["n_raters"] == 4 and c["label_kind"]["interpretation"] == "soft"
    g = fx["argq"][0]
    assert abs(g["labels"]["recommend"]["yes"] - 0.8) < 1e-9 and g["labels"]["stance"]["contest"] == 1.0
    assert fx["scruples/anecdotes"][0]["labels"]["verdict"]["author"] == 5 / 7
    assert fx["scruples/dilemmas"][0]["labels"]["less_ethical"]["second"] == 0.9
    mt = fx["mtbench_human"][0]
    p = mt["labels"]["preference"]
    assert abs(p["tie"] - 1 / 3) < 1e-9 and abs(max(p["a"], p["b"]) - 2 / 3) < 1e-9 and "m1" not in mt["state"]
    sm = fx["summeval"][0]
    assert sm["labels"]["coherence"]["2 - poor"] == 2 / 3 and sm["labels"]["consistency"]["5 - excellent"] == 1.0
    assert fx["politeness/wikipedia"][0]["labels"]["politeness"]["somewhat polite"] == 0.4
    co = fx["casino/outcomes"][0]
    assert "[Agent 1 proposes a final deal" in co["state"] and any(q.startswith("satisfaction_") for q in co["questions"])
    assert co["labels"]["high_priority_1"]["firewood"] == 1.0 and co["labels"]["high_priority_2"]["water"] == 1.0
    cs = fx["casino/strategies"]
    assert all("Submit" not in r["state"] and "proposes" not in r["state"] for r in cs)
    cc = fx["code_contests"]
    assert len(cc) == 2 and sorted(r["labels"]["accepted"]["yes"] for r in cc) == [0.0, 1.0], "one accepted + one rejected (py2 skipped)"
    assert "Codeforces; time limit 1 s, memory limit 256 MB" in cc[0]["state"]
    tf = fx["tabfact"][0]
    assert len(tf["questions"]) == 2 and "first round | 156" in tf["state"] and "Statement:" not in tf["state"]
    assert len(fx["medical/medmcqa"]) == 1 and fx["medical/pubmedqa"][0]["labels"]["answer"]["maybe"] == 1.0
    assert "RESULTS: Recall improved slightly." in fx["medical/pubmedqa"][0]["state"]
    assert len(fx["commonsense/siqa"]) == 1 and set(fx["commonsense/siqa"][0]["questions"]) == {"q0", "q1"}
    assert fx["commonsense/winogrande"][0]["labels"]["answer"]["b"] == 1.0
    print("ok  mapping checks (vote shares, dropped rater-uncertainty votes, balanced verdicts, grouped contexts)")

    # fingerprint index
    idx = Index()
    ev, cp = idx.tag("eval:jevbench/mmlu", "eval"), idx.tag("corpus:jev_data", "corpus")
    idx.add(ev, json.dumps({"subject": "Biochemistry", "question": "Which vitamin deficiency causes scurvy?",
                            "choices": ["A", "B", "C", "D"]}))
    idx.add(ev, "Could you please take a look at my edit?")                       # exact (after the politeness prefix? no: leaf)
    idx.add(cp, "Earlier, a man from Ohio won the lottery on Tuesday and then he bought a boat and a new house with it all.")
    idx.add(ev, json.dumps({"passage": "school uniforms remove visible differences in wealth so that children are judged by what they do",
                            "q": "?"}))
    idx.add(cp, "school uniforms remove visible differences in wealth so that children are judged by what they do and say")
    idx.freeze()
    med = fx["medical/medmcqa"][0]
    assert idx.check(med["state"], ["Which vitamin deficiency causes scurvy?"])[:2] == ("leaf", "eval:jevbench/mmlu")
    assert idx.check("Medical exam question:\nWhich vitamin deficiency causes scurvy?", []) is None, "short text: no shingles"
    hit = idx.check(g["state"], [])
    assert hit[0] == "shingle" and hit[1] == "eval:jevbench/mmlu", f"eval tag must win over corpus: {hit}"
    assert idx.check('"COULD you please take a look   at my edit?"', [])[:2] == ("exact", "eval:jevbench/mmlu")
    print("ok  fingerprint index: exact / leaf / shingle, eval tags take priority")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "ext2"
        writer, seen, sts = Writer(out), set(), {}
        for name, recs in fx.items():
            recs = [dict(r, _probes=list(r.get("_probes", []))) for r in recs]
            sts[name] = process(name, recs, 5000, idx, writer, random.Random(1), new_stats(name, Labeler(0.01)), seen, 6500)
        writer.close()
        assert sts["medical/medmcqa"]["dedup_leaf"] == {"eval:jevbench/mmlu": 1}, sts["medical/medmcqa"]["dedup_leaf"]
        assert sts["argq"]["dedup_shingle"] == {"eval:jevbench/mmlu": 1}, sts["argq"]["dedup_shingle"]
        assert sts["politeness/wikipedia"]["dedup_exact"] == {"eval:jevbench/mmlu": 1}, sts["politeness/wikipedia"]
        rb = final_check(out)
        assert "argq" not in rb["per_source"] and "medical/medmcqa" not in rb["per_source"]
        assert all("_probes" not in json.loads(line) for p in out.rglob("*.jsonl") for line in open(p))
        print("ok  pipeline: dedup, write, iter_units read-back")
    print("\nsample (circa):")
    print(json.dumps(c, indent=1, ensure_ascii=False)[:1400])
    print("\nself-test passed")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="all", help=f"comma-separated output sources or families (default all): {', '.join(CAPS)}")
    ap.add_argument("--out", help="output corpus directory (must not already hold a corpus)")
    ap.add_argument("--eval", action="append", default=[], help="TAG=PATH eval jsonl file or directory (repeatable)")
    ap.add_argument("--fingerprint", action="append", default=[],
                    help="TAG=PATH training-corpus jsonl file or directory, e.g. jev_data=..., jev_ext_strict=... (repeatable)")
    ap.add_argument("--no-fetch-evals", dest="fetch_evals", action="store_false", help="do not fetch eval states from the Hub")
    ap.add_argument("--skip-evals", default="", help=f"comma-separated eval families not to fetch: {', '.join(EVAL_FETCHERS)}")
    ap.add_argument("--allow-missing-evals", action="store_true", help="build even if an eval family could not be fetched")
    ap.add_argument("--allow-missing-corpora", action="store_true", help="build without jev_data / jev_ext_strict fingerprints")
    ap.add_argument("--max-per-source", type=int, default=3000, help="ceiling on every per-source cap")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every default per-source cap")
    ap.add_argument("--oversample", type=float, default=1.5, help="candidate rows read per source = this x its cap")
    ap.add_argument("--max-state-chars", type=int, default=6500, help="drop states longer than this (~1.7k tokens)")
    ap.add_argument("--code-shards", type=int, default=3, help="CodeContests train shards to read (~200 MB each)")
    ap.add_argument("--small", action="store_true", help="read each source's smallest split (laptop smoke test only)")
    ap.add_argument("--limit-rows", type=int, default=0, help="cap candidate rows per source (smoke test)")
    ap.add_argument("--cache", default=str(Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "jev_ext2_urls"),
                    help="cache for non-Hub downloads")
    ap.add_argument("--renorm-tol", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--self-test", action="store_true", help="offline converter + dedup + pipeline tests, then exit")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.out:
        ap.error("--out is required")
    want = [s.strip() for s in a.sources.split(",") if s.strip()]
    a.sources = list(CAPS) if want == ["all"] else [s for s in CAPS if s in want or family_of(s) in want]
    unknown = [w for w in want if w != "all" and w not in CAPS and w not in FAMILIES]
    if unknown or not a.sources:
        ap.error(f"unknown sources {unknown}; choose from {list(CAPS)} or families {FAMILIES}")
    a.skip_evals = {s.strip() for s in a.skip_evals.split(",") if s.strip()}
    bad = a.skip_evals - set(EVAL_FETCHERS)
    if bad:
        ap.error(f"unknown eval families {sorted(bad)}")
    return build(a)


if __name__ == "__main__":
    sys.exit(main())
