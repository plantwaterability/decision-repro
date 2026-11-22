#!/usr/bin/env python3
"""Decision Index 0.2.1 (github.com/apolinario/decision-index, MIT kit) -> our record format, one data dir per benchmark, so
trainers/claude_lmscore.py can score our models on exactly the board's requests and analysis/score_decision_index.py can map
every prediction back to the kit's request id and score it with the kit's own scorer.

EVALUATION ONLY. Never write these records into a training data directory. Several sources forbid it or more: ANLI is CC BY-NC;
RAGTruth contexts include MS MARCO (non-commercial) and Yelp (no redistribution); GPQA asks that its questions never be posted in
plain text (private S3 copy only; never in the repo, logs, or a public place); BBH carries the BIG-bench canary (kept on every
BBH record as meta.canary); HLE is gated. The licence strings below are written so cache_features.is_noncommercial refuses the
NC ones even if a record ever strayed into a training dir.

Input: the kit's frozen suite as staged by `python -m decision_index suite import` (suite-0.2/: selected-rows.jsonl.gz,
added-rows.jsonl.gz, excluded-questions.json, manifest.json), read through the kit's own Suite class under edition 0.2.1 with
the exclusions applied: exactly the 119,898 + 30,419 scoreable requests (ToolRet/BRIGHT answerable queries, the ACOS review
subset and the Home appliances cut applied at read time, as the kit does).

Layout written under --out (one data dir per benchmark; split name `eval`):
  <slug>/eval/<slug>.jsonl     records: one per kit request; record id = the kit's run_id (`_evaluation.run_id`)
  manifest.json                counts per benchmark (requests, questions, K histogram, K > 52 questions) vs the kit's
                               expected counts, licences, the BBH canary, sha256 of the source suite files
  README.txt                   eval-only terms

Record conventions (as trainers/convert_extevals.py, the closest precedent):
  * state: an object state is json.dumps(state, ensure_ascii=False) (as convert_extevals.py); a TEXT state is kept verbatim
    (--state-format text, the default: what the kit's own reference engine renders, `engines.base.text`, and how most of our
    training corpora store text; --state-format json quotes it as convert_extevals.py / jev-bench do, escaped newlines and
    all). An object `instructions` (ToolRet/BRIGHT: {task, candidate}) or option description (ChessBench {uci, san}, POP909
    {names, pitch_classes}, cfcolor palettes) is json.dumps'ed; a null description becomes "" (the renderer shows the key).
  * question id (the unit qid in the npz) = "<run_id>|<question key>"; the kit's option keys are kept verbatim as our option
    keys, so a probability vector maps back key by key. noul questions (RAGTruth, PhishNChips booleans) keep type noul.
  * labels: the kit's `expected`, one-hot ("hard"); a question whose gold is null (unjudged ToolRet/BRIGHT candidates,
    PhishNChips' unscored signal questions) gets an all-zero label, label_kind "none" (the kit's scorer, not our label,
    decides what counts).

K > 52. claude_lmscore.py reads one of 52 single-token letters (A-Z a-z) and SKIPS a unit with more options (unanswered =
wrong in the index). A choice question with K > 52 options (BANKING77 K=77, CLINC150 K=151, POP909 (its full chord
vocabulary has K=129), API-Bank K=53 (full catalog), 40 ChessBench positions with 53-81 legal moves) is split, without dropping
any option, into a two-round TOURNAMENT that is still one forward pass per prompt and no generation:
  round 1  C = ceil(K / 52) chunk questions "<run_id>|<key>|c<j>of<C>", the options dealt into C balanced chunks by a
           seeded shuffle (seed = the question id, so the canonical order cannot leak), same state and instructions;
  round 2  `round2` (run on the node from the round-1 npz) asks one question "<run_id>|<key>|final" over the top
           m = 52 // C options of every chunk (at most 52, so one letter each).
analysis/score_decision_index.py combines them: finalists share 1 - 1e-3 by their round-2 probabilities, the others share
1e-3 by round 1; the choice is the round-2 argmax. With only round 1 it falls back to the chunk probabilities / C.

  python3 trainers/convert_decision_index.py build --kit tmp/decision_index --suite-dir tmp/decision_index_work/suite-0.2 \
      --out tmp/decision_index_work/di_v021 [--bbh-dir WORK/artifacts/benchmark-suite/raw/repos/cand-BIG-Bench-Hard/bbh]
  python3 trainers/convert_decision_index.py round2 --data DIR/04_banking77 --npz r1.npz --out DIR/04_banking77_r2
  python3 trainers/convert_decision_index.py k52 --data DIR       # list the benchmarks that need round 2 (one slug per line)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

MAX_K = 52                     # trainers/claude_lmscore.py LABELS (A-Z a-z)
K52_EPS = 1e-3                 # probability mass left to round-1 losers (analysis/score_decision_index.py)

# catalog id -> (slug, kit dataset name, area or None if shown-not-counted in 0.2.1, licence string)
BENCH = {
    1: ("01_bfcl", "BFCL", "tools", "Apache-2.0 (gorilla repo)"),
    2: ("02_toolret", "ToolRet", "tools", "see ToolRet repo LICENSE; queries/tools aggregate upstream tool datasets with their own terms"),
    3: ("03_apibank", "API-Bank", "tools", "see DAMO-ConvAI api-bank/LICENSE"),
    4: ("04_banking77", "BANKING77", "retrieval", "CC BY 4.0"),
    5: ("05_clinc150", "CLINC150+OOS", "retrieval", "CC BY 3.0"),
    6: ("06_routerbench", "RouterBench", None, "see withmartian/routerbench LICENSE"),
    9: ("09_home_appliances", "Home appliance simulator", "tools", "MIT (decision-index kit)"),
    10: ("10_sgd", "SGD/SGD-X", None, "CC BY-SA 4.0"),
    11: ("11_contractnli", "ContractNLI", "language", "see stanfordnlp/contract-nli LICENSE and dataset terms"),
    12: ("12_anli", "ANLI", "language", "CC BY-NC 4.0 (non-commercial)"),
    20: ("20_bpomp", "BPoMP", "arts", "per Zenodo record 7299879"),
    21: ("21_humicroedit", "Humicroedit", "arts", "SemEval-2020 Task 7 terms"),
    22: ("22_pop909", "POP909-CL", "arts", "see POP909-CL repo LICENSE; POP909 underlying data research use only"),
    23: ("23_cfcolor", "cfcolor", "arts", "per the cfcolor project page"),
    24: ("24_mmlu", "MMLU", None, "MIT"),
    25: ("25_gpqa_diamond", "GPQA Diamond", "knowledge", "CC BY 4.0; authors ask that questions are never posted in plain text: private copies only"),
    26: ("26_arc_easy", "ARC-Easy", None, "CC BY-SA 4.0"),
    27: ("27_arc_challenge", "ARC-Challenge", None, "CC BY-SA 4.0"),
    28: ("28_winogrande", "WinoGrande", "language", "see allenai/winogrande card"),
    29: ("29_hellaswag", "HellaSwag", "language", "MIT"),
    30: ("30_gsm8k", "GSM8K", "knowledge", "MIT"),
    31: ("31_chessbench", "ChessBench", "knowledge", "Apache-2.0 (searchless_chess code); see repo LICENSE"),
    32: ("32_musr", "MuSR", "knowledge", "MIT"),
    33: ("33_sata_bench", "SATA-Bench", "knowledge", "see sata-bench repo LICENSE"),
    34: ("34_simplebench", "SimpleBench", None, "MIT"),
    36: ("36_bright", "BRIGHT", "retrieval", "CC BY 4.0"),
    37: ("37_esci", "Amazon ESCI", "retrieval", "Apache-2.0"),
    38: ("38_acos", "ACOS", "language", "not stated in the NUSTM/ACOS repository"),
    39: ("39_finentity", "FinEntity", "language", "not stated in the FinEntity repository"),
    40: ("40_isarcasmeval", "iSarcasmEval", "language", "see iSarcasmEval repo LICENSE"),
    41: ("41_vast", "VAST", "language", "not stated in the zero-shot-stance repository"),
    42: ("42_nli4ct", "NLI4CT", "language", "SemEval-2024 Task 2 terms"),
    43: ("43_cruxeval", "CRUXEval", "knowledge", "MIT"),
    44: ("44_cladder", "CLadder", "knowledge", "see causalNLP/cladder LICENSE"),
    45: ("45_hle", "HLE", "knowledge", "MIT per dataset card; gated access (terms accepted on the Hub); never in training corpora"),
    48: ("48_forecastbench", "ForecastBench", "arts", "see forecastbench-datasets LICENSE"),
    50: ("50_habermas", "Habermas Machine", "arts", "see google-deepmind/habermas_machine LICENSE"),
    56: ("56_phishnchips", "PhishNChips", "retrieval", "per PhishNChips SOURCE_LICENSES.md (project MIT; URL sources academic use only, with attribution)"),
    57: ("57_mmlu_pro", "MMLU-Pro", "knowledge", "MIT"),
    58: ("58_bbh", "BBH", "knowledge", "MIT; carries the BIG-bench canary: never in training corpora"),
    59: ("59_ragtruth", "RAGTruth", "language", "MIT annotations; contexts include MS MARCO (non-commercial) and the Yelp Open Dataset (no redistribution): evaluation only"),
    61: ("61_hover", "HoVer", "retrieval", "CC BY-SA 4.0"),
    62: ("62_when2call", "When2Call", "tools", "CC BY 4.0"),
    64: ("64_newyorker", "New Yorker", "arts", "CC BY 4.0 annotations"),
}
# Scoreable requests per benchmark under 0.2.1 (README table, "after the 442 exclusions", with the 0.2.1 cuts: ToolRet 685,
# BRIGHT 220, Home appliances 88). Sum = 119,898 + 30,419 = 150,317.
EXPECTED = {1: 1694, 2: 685, 3: 508, 4: 3080, 5: 5500, 6: 10000, 9: 88, 10: 2500, 11: 123, 12: 3200, 20: 5000, 21: 2628, 22: 2000,
            23: 5000, 24: 14033, 25: 196, 26: 2376, 27: 1172, 28: 1267, 29: 10042, 30: 2638, 31: 5000, 32: 752, 33: 1650, 34: 10,
            36: 220, 37: 5000, 38: 1565, 39: 979, 40: 4600, 41: 3006, 42: 5500, 43: 570, 44: 5000, 45: 501, 48: 10139, 50: 1676,
            56: 2000, 57: 12032, 58: 5507, 59: 2700, 61: 4000, 62: 3652, 64: 528}
INDEX_IDS = sorted(n for n, b in BENCH.items() if b[2])          # the 38 counted in 0.2.1
NC_IDS = {12, 22, 56, 59}                                        # licence strings that cache_features.NC_LICENCE matches


def dumps(x) -> str:
    return json.dumps(x, ensure_ascii=False)


def as_text(x) -> str:
    """A kit text field as our renderer will show it: strings verbatim, None -> "", anything else as JSON."""
    if x is None:
        return ""
    return x if isinstance(x, str) else dumps(x)


def option_keys(q: dict) -> list[str]:
    """Our option keys for a kit question, in the order cache_features.question_options stores them."""
    if q["type"] == "choice":
        return [str(k) for k in q["criteria"]]
    if q["type"] == "noul":
        return ["yes", "no"]
    raise ValueError(f"unsupported question type {q['type']!r}")


def our_question(q: dict, keys: list[str] | None = None) -> dict:
    """The kit question in our schema; `keys` restricts a choice to those option keys (K > 52 chunks), in canonical order."""
    t = q["type"]
    out = {"type": t, "instructions": as_text(q.get("instructions"))}
    if t == "choice":
        crit = {str(k): as_text(v) for k, v in q["criteria"].items()}
        out["criteria"] = crit if keys is None else {k: crit[k] for k in keys}
    elif t == "noul":
        if q.get("criteria") is not None:
            out["criteria"] = q["criteria"]
    else:
        raise ValueError(f"unsupported question type {t!r}")
    return out


def label(q: dict, gold, keys: list[str] | None = None) -> tuple[dict, str]:
    """One-hot over our option keys from the kit's `expected` (noul gold: bool); all-zero ("none") when the gold is null or,
    for a K > 52 chunk, not among the chunk's options."""
    ks = keys if keys is not None else option_keys(q)
    if gold is None:
        return {k: 0.0 for k in ks}, "none"
    if q["type"] == "noul":
        g = "yes" if gold is True or str(gold).lower() in ("1", "true", "yes") else "no"
    else:
        g = str(gold)
    if keys is None and g not in ks:
        raise ValueError(f"gold {gold!r} is not an option")
    return {k: (1.0 if k == g else 0.0) for k in ks}, ("hard" if g in ks else "none")


def qid_of(run_id: str, qkey: str, part: str | None = None) -> str:
    return f"{run_id}|{qkey}" + (f"|{part}" if part else "")


def split_qid(qid: str) -> tuple[str, str, str | None]:
    """(run_id, question key, part) from a qid. A run id never contains '|' (checked at build); a question key may (ACOS keys
    hold '#', not '|', but be safe): the part is only ever 'c<j>of<C>' or 'final', so peel it off the right."""
    run_id, rest = qid.split("|", 1)
    head, sep, tail = rest.rpartition("|")
    if sep and (tail == "final" or (tail.startswith("c") and "of" in tail and tail[1:].replace("of", "").isdigit())):
        return run_id, head, tail
    return run_id, rest, None


def chunks_for(run_id: str, qkey: str, keys: list[str], max_k: int = MAX_K) -> list[list[str]]:
    """The K > 52 round-1 chunks: a shuffle seeded by the question id, dealt round-robin into C = ceil(K / max_k) balanced
    chunks, each kept in canonical (kit) relative order. Deterministic, so `round2` and the scorer recompute it exactly."""
    c = math.ceil(len(keys) / max_k)
    order = list(keys)
    random.Random(int(hashlib.sha256(f"di-k52:{run_id}|{qkey}".encode()).hexdigest()[:16], 16)).shuffle(order)
    pos = {k: i for i, k in enumerate(keys)}
    return [sorted(order[j::c], key=pos.__getitem__) for j in range(c)]


def finalists_per_chunk(n_chunks: int, max_k: int = MAX_K) -> int:
    return max_k // n_chunks


# ----------------------------------------------------------------------------------------------------------- build
def kit_suite(kit: Path, suite_dir: Path, edition: str):
    sys.path.insert(0, str(kit))
    from decision_index.suite.io import Suite  # the kit's own reader: exclusions + read-time subsets
    s = Suite(suite_dir, edition)
    rep = s.verify(strict=False)
    return s, rep


def bbh_canary(bbh_dir: Path | None) -> str | None:
    if not bbh_dir or not bbh_dir.exists():
        return None
    found = {json.loads(p.read_text()).get("canary") for p in sorted(bbh_dir.glob("*.json"))}
    found.discard(None)
    if len(found) != 1:
        raise SystemExit(f"expected one BBH canary string in {bbh_dir}, found {len(found)}")
    return found.pop()


def render_state(state, fmt: str = "text") -> str:
    """Our record's state string: objects as JSON; a text state verbatim ("text") or JSON-quoted ("json")."""
    if fmt == "text" and isinstance(state, str):
        return state
    return dumps(state)


def convert_row(row: dict, canary: str | None, stats: dict, state_format: str = "text") -> dict:
    e = row["_evaluation"]
    n, rid = int(e["catalog_id"]), str(e["run_id"])
    if "|" in rid:
        raise ValueError(f"run id with '|': {rid}")
    slug, name, area, lic = BENCH[n]
    qs, labels, kinds, k52 = {}, {}, {}, {}
    for qkey, q in row["questions"].items():
        if "|" in qkey:
            raise ValueError(f"question key with '|' in {rid}: {qkey!r}")   # keeps split_qid unambiguous
        keys = option_keys(q)
        gold = (row.get("expected") or {}).get(qkey)
        stats["k_hist"][len(keys)] += 1
        if q["type"] == "choice" and len(keys) > MAX_K:
            parts = chunks_for(rid, qkey, keys)
            stats["k52_questions"] += 1
            for j, ck in enumerate(parts):
                qid = qid_of(rid, qkey, f"c{j}of{len(parts)}")
                qs[qid] = our_question(q, ck)
                labels[qid], kinds[qid] = label(q, gold, ck)
            k52[qkey] = {"K": len(keys), "chunks": len(parts), "finalists_per_chunk": finalists_per_chunk(len(parts)), "keys": keys}
            stats["units"] += len(parts)
        else:
            qid = qid_of(rid, qkey)
            qs[qid] = our_question(q)
            labels[qid], kinds[qid] = label(q, gold)
            stats["units"] += 1
    meta = {"catalog_id": n, "benchmark": name, "run_id": rid, "group_id": e.get("group_id"), "track": e.get("track"),
            "payload_sha256": e.get("payload_sha256"), "area": area, "in_index_021": area is not None, "eval_only": True,
            "builder": "decision-index kit 0.2.1 (suite-0.2 rows) via trainers/convert_decision_index.py"}
    if k52:
        meta["k52"] = k52
    if n == 58 and canary:
        meta["canary"] = canary
    return {"id": rid, "base_id": f"di/{n}/{e.get('group_id')}", "source": f"decision_index/{slug}", "license": f"EVALUATION ONLY. {lic}",
            "state": render_state(row["state"], state_format), "questions": qs, "labels": labels, "label_kind": kinds, "meta": meta, "split": "eval"}


def cmd_build(a) -> int:
    out = Path(a.out)
    suite, ver = kit_suite(Path(a.kit), Path(a.suite_dir), a.edition)
    if not ver.get("match") and not a.allow_mismatch:
        raise SystemExit(f"suite does not verify against the kit's pinned hashes: {json.dumps(ver)[:600]}\n(--allow-mismatch to "
                         "convert anyway, e.g. a build without HLE)")
    canary = bbh_canary(Path(a.bbh_dir) if a.bbh_dir else None)
    want = set(BENCH) if a.include_display else set(INDEX_IDS)
    if a.only:
        want &= {int(x) for x in a.only.split(",")}
    files, stats = {}, defaultdict(lambda: {"requests": 0, "units": 0, "questions": 0, "k52_questions": 0, "k_hist": Counter(),
                                            "noul": 0, "null_gold": 0})
    seen = set()
    try:
        for row in suite.rows(apply_exclusions=True):
            n = int(row["_evaluation"]["catalog_id"])
            if n not in want:
                continue
            rid = row["_evaluation"]["run_id"]
            if rid in seen:
                raise SystemExit(f"duplicate run id {rid}")
            seen.add(rid)
            st = stats[n]
            rec = convert_row(row, canary, st, a.state_format)
            st["requests"] += 1
            st["questions"] += len(row["questions"])
            st["noul"] += sum(q["type"] == "noul" for q in row["questions"].values())
            st["null_gold"] += sum(v is None for v in (row.get("expected") or {}).values())
            if n not in files:
                slug = BENCH[n][0]
                (out / slug / "eval").mkdir(parents=True, exist_ok=True)
                files[n] = open(out / slug / "eval" / f"{slug}.jsonl", "w", encoding="utf-8")
            files[n].write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        for f in files.values():
            f.close()
    per = {}
    for n in sorted(stats):
        st = stats[n]
        per[BENCH[n][0]] = {"catalog_id": n, "benchmark": BENCH[n][1], "area_021": BENCH[n][2], "requests": st["requests"],
                            "expected_requests": EXPECTED[n], "matches_kit": st["requests"] == EXPECTED[n], "questions": st["questions"],
                            "units_round1": st["units"], "noul_questions": st["noul"], "null_gold_questions": st["null_gold"],
                            "k_max": max(st["k_hist"]), "k_hist": dict(sorted(st["k_hist"].items())),
                            "k_gt_52_questions": st["k52_questions"], "needs_round2": st["k52_questions"] > 0,
                            "licence": BENCH[n][3], "non_commercial": n in NC_IDS}
    missing = sorted(n for n in want if n not in stats)
    manifest = {"name": "Decision Index 0.2.1, converted for trainers/claude_lmscore.py", "edition": a.edition, "eval_only": True,
                "state_format": a.state_format,
                "kit": "https://github.com/apolinario/decision-index (MIT)", "suite_verify": ver, "benchmarks": per,
                "missing_benchmarks": [BENCH[n][1] for n in missing],
                "totals": {"requests": sum(v["requests"] for v in per.values()), "expected": sum(EXPECTED[n] for n in want),
                           "questions": sum(v["questions"] for v in per.values()), "units_round1": sum(v["units_round1"] for v in per.values())},
                "k52": {"max_k": MAX_K, "rule": "ceil(K/52) chunks by a seeded shuffle, top 52//C per chunk to one final question",
                        "benchmarks": [s for s, v in per.items() if v["needs_round2"]]},
                "qid": "<run_id>|<question key>[|c<j>of<C> | |final]",
                "bbh_canary": canary,
                "terms": "Evaluation only. Do not train on any of this. GPQA: never post in plain text (private S3 only). RAGTruth: "
                         "MS MARCO (NC) and Yelp (no redistribution) contexts. ANLI CC BY-NC. HLE gated. BBH canary kept."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    (out / "README.txt").write_text(
        "Decision Index 0.2.1 suite (rebuilt from public sources with the MIT kit github.com/apolinario/decision-index),\n"
        "converted for trainers/claude_lmscore.py by trainers/convert_decision_index.py.\n\n"
        "EVALUATION ONLY: never train on these records, never copy them into a training data directory, never publish them.\n"
        "GPQA Diamond (25_gpqa_diamond): the authors ask that the questions never be posted in plain text; keep this private.\n"
        "RAGTruth (59_ragtruth): MS MARCO (non-commercial) and Yelp (no redistribution) contexts.\n"
        "ANLI (12_anli): CC BY-NC 4.0. HLE (45_hle): gated on the Hub.\n"
        f"BBH (58_bbh) carries the BIG-bench canary; it is kept on every BBH record (meta.canary): {canary}\n")
    for s, v in per.items():
        flag = "" if v["matches_kit"] else "   <-- differs from the kit's expected count"
        print(f"{s:22s} requests {v['requests']:6d} / kit {v['expected_requests']:6d}  questions {v['questions']:7d}  "
              f"units {v['units_round1']:7d}  K<= {v['k_max']:3d}  K>52 q {v['k_gt_52_questions']:5d}{flag}")
    print(json.dumps(manifest["totals"]), "missing:", manifest["missing_benchmarks"])
    return 0 if not missing and all(v["matches_kit"] for v in per.values()) else 1


# ----------------------------------------------------------------------------------------------------------- round 2
def read_records(data_dir: Path, split: str = "eval"):
    for p in sorted((data_dir / split).glob("*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def load_npz_probs(path: Path) -> dict[str, list[float]]:
    import numpy as np

    z = np.load(path, allow_pickle=False)
    qids, P, ks = z["test_qid"], z["test_p"], z["test_ks"]
    return {str(q): [float(x) for x in P[i, : int(ks[i])]] for i, q in enumerate(qids)}


def chunk_qids(rec: dict, qkey: str) -> list[str]:
    """The round-1 chunk question ids of a K > 52 question, in chunk order (c0of C, c1of C, ...)."""
    info = rec["meta"]["k52"][qkey]
    out = [qid_of(rec["id"], qkey, f"c{j}of{info['chunks']}") for j in range(info["chunks"])]
    missing = [q for q in out if q not in rec["questions"]]
    if missing:
        raise ValueError(f"{rec['id']} {qkey}: chunk questions {missing} not in the record")
    return out


def finalists(rec: dict, qkey: str, probs: dict) -> list[str] | None:
    """Round-2 option keys of a K > 52 question, in canonical (kit) order: the top 52 // C of every chunk by round-1 probability
    (ties: canonical order inside the chunk). None when a chunk has no round-1 prediction."""
    info, chosen = rec["meta"]["k52"][qkey], set()
    for qid in chunk_qids(rec, qkey):
        keys = list(rec["questions"][qid]["criteria"])
        p = probs.get(qid)
        if p is None or len(p) != len(keys):
            return None
        chosen.update(keys[i] for i in sorted(range(len(keys)), key=lambda i: (-p[i], i))[: info["finalists_per_chunk"]])
    return [k for k in info["keys"] if k in chosen]


def final_question(rec: dict, qkey: str, keys: list[str]) -> tuple[dict, dict, str]:
    """(question, label, label_kind) of the round-2 question over `keys`: same instructions, criteria from the chunks."""
    parts = chunk_qids(rec, qkey)
    crit = {}
    for qid in parts:
        crit.update(rec["questions"][qid]["criteria"])
    gold = next((k for qid in parts for k, v in rec["labels"][qid].items() if v == 1.0), None)
    q = {"type": "choice", "instructions": rec["questions"][parts[0]]["instructions"], "criteria": {k: crit[k] for k in keys}}
    return q, {k: (1.0 if k == gold else 0.0) for k in keys}, ("hard" if gold in keys else "none")


def cmd_round2(a) -> int:
    """Round-2 records for every K > 52 question of a round-1 data dir, from the round-1 npz: one 'final' question over the
    top 52 // C options of every chunk, same state and instructions. Written to OUT/eval/<slug>_r2.jsonl."""
    data, out = Path(a.data), Path(a.out)
    probs = load_npz_probs(Path(a.npz))
    recs, missing, n_final = [], 0, 0
    for rec in read_records(data):
        k52 = rec.get("meta", {}).get("k52")
        if not k52:
            continue
        qs, labels, kinds = {}, {}, {}
        for qkey in k52:
            keys = finalists(rec, qkey, probs)
            if keys is None:
                missing += 1
                continue
            fq = qid_of(rec["id"], qkey, "final")
            qs[fq], labels[fq], kinds[fq] = final_question(rec, qkey, keys)
            n_final += 1
        if qs:
            recs.append({**{k: rec[k] for k in ("id", "base_id", "source", "license", "state", "meta", "split")},
                         "questions": qs, "labels": labels, "label_kind": kinds})
    slug = data.name
    (out / "eval").mkdir(parents=True, exist_ok=True)
    with open(out / "eval" / f"{slug}_r2.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    print(f"round2 {slug}: {n_final} final questions in {len(recs)} records; {missing} K>52 questions lacked round-1 "
          f"predictions (left to the round-1 fallback)")
    return 0 if n_final else 1


def cmd_k52(a) -> int:
    m = json.loads((Path(a.data) / "manifest.json").read_text())
    for s in m["k52"]["benchmarks"]:
        print(s)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="kit suite -> one data dir per benchmark")
    b.add_argument("--kit", required=True, help="the decision-index kit checkout (its decision_index package is imported)")
    b.add_argument("--suite-dir", required=True, help="the staged suite (suite-0.2/ from `suite import`)")
    b.add_argument("--edition", default="0.2.1")
    b.add_argument("--out", required=True)
    b.add_argument("--bbh-dir", default=None, help="BIG-Bench-Hard bbh/ dir of the build, to copy the canary onto BBH records")
    b.add_argument("--include-display", action="store_true", help="also the 6 shown-not-counted sets (MMLU, ARC x2, SimpleBench, "
                   "RouterBench, SGD)")
    b.add_argument("--only", default="", help="comma list of catalog ids")
    b.add_argument("--state-format", choices=("text", "json"), default="text",
                   help="a text state verbatim (default) or JSON-quoted as convert_extevals.py; object states are JSON either way")
    b.add_argument("--allow-mismatch", action="store_true")
    b.set_defaults(fn=cmd_build)
    r = sub.add_parser("round2", help="K > 52 final questions from a round-1 npz")
    r.add_argument("--data", required=True)
    r.add_argument("--npz", required=True)
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_round2)
    k = sub.add_parser("k52", help="list the benchmark dirs that need round 2")
    k.add_argument("--data", required=True)
    k.set_defaults(fn=cmd_k52)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
