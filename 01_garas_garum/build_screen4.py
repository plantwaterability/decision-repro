#!/usr/bin/env python3
"""(SCREEN4) Data builder for the evidence-grounded, capability-preserving Garas / (99) continuation screen.

Design tmp/screen4/design.md section 5; pre-registration draft tmp/screen4/prereg_draft.md. Nothing here trains or scores; every output
is deterministic given its inputs and --seed. Sources: tasksource/tasksource-jev-typed-decisions at the PINNED revision TS_REVISION
(configs filtered-full and full; shard sha256s checked), PC's published replay of (99)'s training set (results_pc/data/pc_replay_v1.tgz
67359006: R, R2, mix2 val), the Bayesian-evidence families (tmp/bayes_families/INTERFACE.md v0) and teacher score files.

  eg        the EVIDENCE-GROUNDED training pool (EG; the board's losses: claim verification against evidence, support / sufficiency,
            relevance with hard negatives, answerability). Allow-list EG_SOURCES only; each row passes, in order:
              licence   clean line: license_use == commercial, no NC / research-only string, no custom / unspecified licence string;
              holdout   never a DI benchmark source, a DI blocked root, or a source of one of our eval families (HOLDOUT_SOURCES);
              length    state <= MAX_STATE_CHARS (so the 2,048-token training prompt never cuts evidence away);
              overlap   no exact / leaf / 8-word-shingle (>= 3 shared) hit of the eval fingerprints OR the training corpora (X99) in the
                        build_jev_ext2 Index, no sealed final_v1 / FRESH CONFIRMATION index hit (ids + state sha1);
              convert   build_jev_ext3.ts_question (K <= 52, normalised targets, distinct options).
            DERIVED VARIANTS (exact labels by construction; base + variant kept together as one pair / cluster):
              unanswerable  a single-gold choice row of a relevance / answerability / multi-hop source with the gold option REMOVED and
                            NONE_OPTION appended -> one-hot on it (missing evidence);
              distractor    the same kind of row with DISTRACTORS (2) appended: other rows' options of the same source with the highest
                            word-Jaccard to the gold option, never equal to an existing option; target unchanged.
            Train / val split by tasksource group (sha1 % 100 < 15 -> val); per-source cap; exact unit counts.
  didev     the DI-FAMILY DEV panel: tasksource `full` rows of the DI benchmarks' own (non-test) material -- hover, esci, anli, hellaswag,
            winogrande, ARC, commonsense_qa, banking77, clinc_oos, nli4ct, humicroedit, circa, openbookqa, isarcasm, tabfact -- with
            every row whose state hits the DI public panel, our evals, the training corpora or a sealed / confirmation index REMOVED.
            Selection only (dev); `check` refuses any overlap with a training arm. A PROXY of the skill in tasksource's recast format,
            not the board's protocol.
  caldev    the calibration dev: a seeded sample of (99)'s own mix2 val (PC's replay val/), NC records dropped (OPTISO's split rule).
  arms      a = R + R2 (PC's arm r, 12,000 units); b = R + R2[2,000] + EG[4,000]; c = b with PRESERVATION targets; d = R + R2[500] + EG +
            Bayes[1,500] with the same preservation (every arm 12,000 units = 1,500 steps x 8). AMEND A1: Bayes = bayes_families_v1 train
            (INTERFACE v1), BAYES_BLOCKS = 1,000 retrieval-style (the five ret_* families) + 500 numeric (the five A1 families), whole groups
            round-robin per block, exact fill; a group with ANY member hitting the sealed / CONFIRM-2610 index (ids, state sha1, clusters),
            over-long, K > 52, NC or non-distribution labels is dropped whole. R2[n] = a seeded exact-fill subset of R2
            records (R2[500] a subset of R2[2,000]). Preservation (trainers/screen4_loss.py): replay units -> (99)'s order-marginal
            prediction, EG units -> stock Gemma's tempered prediction where it agrees with gold, Bayes units unchanged. `--phase plain`
            writes a, b and the two teacher dirs; `--phase mixed` (after the teacher passes) writes c and d.
  check     re-verifies an arms dir (counts, subset / prefix rules, roles, pairs, licences, train-vs-dev disjointness, file pins).

  python3 build_screen4.py --self-test
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE / "analysis", HERE / "fleet", HERE / "trainers"):
    sys.path.insert(0, str(_p))

VERSION = "build_screen4.py v4 (2026-10-12; Garas line; AMEND A3)"
TS_REPO = "tasksource/tasksource-jev-typed-decisions"
TS_REVISION = "f8ff7e81ed820eeac1ad7afd63449fe99c8a76c3"
# shard pins: --shard-pins JSON {relative path: sha256} (tmp/screen4/audit/ts_pins_f8ff7e81.json, measured read-only on 2026-10-12: the
# 13 filtered-full shards -- NONE equals the dataset's own filtering-report.json, which describes an earlier export of 1,305,478 rows;
# 1,304,043 here -- and the `full` config's shards)
COLS = ["state", "kind", "id", "options", "target", "question", "source", "variant", "split", "group_id", "license", "license_use"]
EG_VAL_PCT = 15
MAX_STATE_CHARS = 6000
NONE_OPTION = "None of the listed options is supported by the given information."
N_DISTRACTORS = 2
NC_LICENCE = re.compile(r"\bNC\b|non[- ]?commercial|research[- ]only|research use only|academic use only|cc[- ]by[- ]nc|\bby-nc\b", re.I)   # = cache_features / ss_lib (tested)
CUSTOM_LICENCE = re.compile(r"custom|unspecified|unknown|^\s*$|^\s*other\s*$|^\s*none\s*$|^\s*n/?a\s*$", re.I)
# the EVIDENCE-GROUNDED allow-list: source (exact tasksource name) -> (family, variants allowed)
EG_SOURCES = {
    "vitaminc": ("claim_verification", False), "wice": ("claim_verification", False), "fool-me-twice": ("claim_verification", False),
    "health_fact": ("claim_verification", False), "doc-nli": ("support", False), "WANLI": ("adversarial_nli", False),
    "procedural-typed-decisions/evidence_sufficiency": ("support", False),
    "procedural-typed-decisions/needle_retrieval": ("relevance", True),
    "wands": ("relevance", False), "nevir": ("relevance", True), "hotpotqa/supporting-fact": ("multi_hop", True),
    "wiki_hop/original": ("multi_hop", True), "squad-v2/answerability": ("answerability", False),
    "quac/answerability": ("answerability", False),
}
# DI-family dev blocks: block -> tasksource source prefixes (the `full` config). TARGET blocks are the board's evidence-grounded losses.
DIDEV_BLOCKS = {
    "hover": ("hover",), "esci": ("esci",), "anli": ("anli/",), "hellaswag": ("hellaswag",), "winogrande": ("winogrande/",),
    "arc": ("ai2_arc/",), "commonsense_qa": ("commonsense_qa",), "banking77": ("banking77",), "clinc": ("clinc_oos/",),
    "nli4ct": ("nli4ct_semeval2024",), "humicroedit": ("humicroedit/",), "circa": ("circa",), "openbookqa": ("openbookqa",),
    "isarcasm": ("isarcasm/",), "tabfact": ("tabfact",),
}
DIDEV_TARGET = ("hover", "esci", "anli")
DIDEV_MIN_NONTARGET = 150         # AMEND A2: a non-target block may hold fewer than 300 units, down to this floor
# never in any TRAINING pool: DI benchmarks + their blocked roots, and the sources of our eval families (alnum path components;
# prefix match for aliases of >= 5 characters)
HOLDOUT_SOURCES = {
    "bench4:civil_comments": "civilcomments", "bench4:go_emotions": "goemotions", "bench4:chaosnli": "chaosnli",
    "bench4:measuring_hate_speech": "measuringhatespeech", "jevbench:sms_spam": "smsspam", "jevbench:massive": "massive",
    "jevbench:ledgar": "ledgar", "jevbench:helpsteer2": "helpsteer2", "jevbench:fever_evidence": "feverevidence",
    "jevbench:boolq": "boolq", "jevbench:paws": "paws", "jevbench:strategyqa": "strategyqa", "jevbench:stsb": "stsb",
    "jevbench:sst5": "sst5", "jevbench:yelp5": "yelp", "typed-decisions": "typeddecisions", "dices": "dices", "legalbench": "legalbench",
    **{f"DI:{r}": re.sub(r"[^0-9a-z]", "", r) for r in (
        "acos", "ai2_arc", "anli", "api-bank", "arc-challenge", "arc-easy", "banking77", "bbh", "belebele", "bfcl", "bigbench", "blimp",
        "bright", "chessbench", "circa", "cladder", "clinc150", "clinc_oos", "commonsense_qa", "contract-nli", "cos_e", "cruxeval",
        "esci", "financial_phrasebank", "finentity", "forecastbench", "gpqa", "gsm8k", "hellaswag", "hle", "hover", "humicroedit",
        "isarcasm", "it-support-tickets", "jevbench", "judgebench", "mmlu", "musr", "nli4ct", "openbookqa", "phishnchips",
        "ragtruth", "routerbench", "sata-bench", "schema_guided_dstc8", "sgd", "simplebench", "tabfact", "toolret", "truthfulqa",
        "vast", "when2call", "winogrande", "habermas", "bpomp", "pop909", "cfcolor", "newyorker")},
}
KEEP_SOURCES = ("procedural-typed-decisions/",)    # user 2026-10-12: TD-format data allowed (typed results are FITTED)
ROLES = ("replay", "eg", "bayes")
ARM_SPEC = {   # arm -> (R units, R2 subset units, EG units, Bayes units, preservation)
    "a": (6000, 6000, 0, 0, False), "b": (6000, 2000, 4000, 0, False), "c": (6000, 2000, 4000, 0, True),
    "d": (6000, 500, 4000, 1500, True),
}
N_TOTAL = 12000


# ============================================================================ helpers
def sha1(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def alnum(s):
    return re.sub(r"[^0-9a-z]", "", str(s).casefold())


def norm(s):
    from claude_overlap import norm as _n
    return _n(s if isinstance(s, str) else json.dumps(s, ensure_ascii=False))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, recs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def canon(r):
    """a record exactly as write_jsonl stores it. AMEND A3: sort_keys also SORTS a choice question's criteria keys, i.e. its stored option
    order, so every file this builder writes (arms, teacher dirs, dev panels) holds options in sorted-key order and the teachers scored
    that order; anything read from OUTSIDE (PC's replay) must pass through this before its units are enumerated against a teacher npz."""
    return json.loads(json.dumps(r, ensure_ascii=False, sort_keys=True))


def file_table(root):
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(Path(root).rglob("*.jsonl"))}


def units_of(rec):
    return len(rec["questions"])


def holdout_of(source):
    """the eval / DI family a tasksource source belongs to (HOLDOUT_SOURCES), or None; KEEP_SOURCES win"""
    if any(source.startswith(k) for k in KEEP_SOURCES):
        return None
    parts = [alnum(p) for p in source.split("/") if alnum(p)]
    for fam, a in HOLDOUT_SOURCES.items():
        if any(p == a or (len(a) >= 5 and p.startswith(a)) for p in parts):
            return fam
    return None


def licence_status(row, line="clean"):
    """None = keep; else the drop reason"""
    if line == "eval":
        return None
    lic, use = str(row.get("license") or ""), str(row.get("license_use") or "")
    if use != "commercial":
        return f"license_use_{use or 'missing'}"
    if NC_LICENCE.search(lic):
        return "nc_licence_string"
    if any(CUSTOM_LICENCE.search(x.strip()) for x in lic.split(",")):
        return "custom_or_unspecified_licence"
    return None


def option_keys(q):
    """cache_features.question_options' stored key order (tested equal): choice = criteria order, noul = yes / no, score = levels"""
    if q["type"] == "choice":
        return list(q["criteria"])
    if q["type"] == "noul":
        return ["yes", "no"]
    if q["type"] == "score":
        return list(q["levels"])
    raise ValueError(f"question type {q['type']!r}")


# ============================================================================ exclusions
class Exclusions:
    """build_jev_ext2.Index over eval files (class 'eval') and training corpora (class 'corpus') + sealed / confirmation indexes."""

    def __init__(self, idx=None, coverage=None, indexes=()):
        self.idx, self.coverage = idx, coverage or {}
        self.sha1s, self.ids, self.clusters, self.info = collections.defaultdict(set), collections.defaultdict(set), collections.defaultdict(set), {}
        for name, path in indexes:
            ix = json.load(gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8"))
            for v in ix.get("state_sha1") or []:
                self.sha1s[str(v)].add(name)
            for v in list(ix.get("ids") or []) + list(ix.get("unassigned_eval_only_ids") or []):
                self.ids[str(v)].add(name)
            for v in ix.get("clusters") or []:
                self.clusters[str(v)].add(name)
            if not (ix.get("state_sha1") or ix.get("ids")):
                raise SystemExit(f"exclusion index {name} ({path}) has no ids / state sha1s")
            self.info[name] = {"path": str(path), "sha256": sha256_file(path), "state_sha1": len(ix.get("state_sha1") or []),
                               "ids": len(ix.get("ids") or []), "clusters": len(ix.get("clusters") or [])}

    def hit(self, rid, state, cluster=None):
        """None, or a tag. cluster: '<source>:<group>' (the sealed indexes' cluster key; ss_lib.hits' rule)"""
        if str(rid) in self.ids:
            return "index_id:" + sorted(self.ids[str(rid)])[0]
        if cluster is not None and str(cluster) in self.clusters:
            return "index_cluster:" + sorted(self.clusters[str(cluster)])[0]
        h = sha1(norm(state))
        if h in self.sha1s:
            return "index_sha1:" + sorted(self.sha1s[h])[0]
        if self.idx is not None:
            r = self.idx.check(state)
            if r:
                return f"{r[1]}|{r[0]}"
        return None


def load_exclusions(a):
    """the confirmation-index refusal FIRST (cheap), then the fingerprints (eval files / fetched eval families / training corpora)"""
    if not a.confirm_index and not a.no_confirm_index:
        raise SystemExit("no --confirm-index: the fresh confirmation set must be sealed and excluded BEFORE the screen's data exists "
                         "(--no-confirm-index is for tests only)")
    import build_jev_ext2 as x2
    idx, cov = None, {}
    if a.eval or a.fetch_evals or a.corpus:
        ns = argparse.Namespace(eval=a.eval, fingerprint=a.corpus, allow_missing_corpora=True, fetch_evals=a.fetch_evals,
                                skip_evals=[], allow_missing_evals=a.allow_missing_evals, hf_token=None, cache=str(Path(a.out) / ".urlcache"))
        idx, cov = x2.load_fingerprints(ns)
    ix = [(f"sealed{i}", p) for i, p in enumerate(a.sealed)] + ([("confirm", a.confirm_index)] if a.confirm_index else [])
    return Exclusions(idx, cov, ix)


# ============================================================================ tasksource rows
def iter_ts_rows(shard_dir, pins, check_sha=True):
    import pyarrow.parquet as pq
    for rel, want in sorted(pins.items()):
        p = Path(shard_dir) / rel
        if not p.exists():
            raise SystemExit(f"missing shard {p}")
        if check_sha and sha256_file(p) != want:
            raise SystemExit(f"shard {rel}: sha256 {sha256_file(p)[:12]} is not the pinned {want[:12]} (revision {TS_REVISION})")
        for b in pq.ParquetFile(p).iter_batches(batch_size=20000, columns=COLS):
            for r in b.to_pylist():
                r["_config"] = rel.split("/")[0]
                yield r


def ts_record(row, lab, role, extra_meta=None):
    """one tasksource row -> one jev record (one question; build_jev_ext3.ts_question), or None"""
    import build_jev_ext3 as x3
    got = x3.ts_question(row, random.Random(f"screen4/v2|q|{row['id']}"), lab)
    if not got:
        return None
    q, d = got
    qid = "q_" + sha1(str(row["id"]))[:12]
    soft = sum(1 for v in d.values() if v > 0) > 1
    gid = str(row.get("group_id") or row["id"])
    meta = {"ts": {"id": str(row["id"]), "source": row["source"], "group_id": gid, "variant": row.get("variant"),
                   "license_use": row.get("license_use"), "kind": row.get("kind"), "config": row.get("_config"), "revision": TS_REVISION},
            "screen4": {"role": role, **(extra_meta or {})}}
    return {"id": f"s4ts-{sha1(str(row['id']))[:20]}", "source": f"ts4/{row['source']}", "license": str(row.get("license") or ""),
            "base_id": f"ts4/{sha1(gid)[:20]}", "state": row["state"], "questions": {qid: q}, "labels": {qid: d},
            "label_kind": {qid: "soft" if soft else "hard"}, "meta": meta}


def single_gold(row):
    t = [float(x) for x in (row.get("target") or [])]
    return row.get("kind") == "choice" and len(t) >= 3 and sum(1 for x in t if x > 0) == 1 and abs(sum(t) - 1) < 1e-6


def words(s):
    return set(re.findall(r"\w+", str(s).lower()))


def derive_unanswerable(row):
    t = [float(x) for x in row["target"]]
    g = t.index(max(t))
    opts = [o for j, o in enumerate(row["options"]) if j != g] + [NONE_OPTION]
    return dict(row, id=f"{row['id']}::unans", options=opts, target=[0.0] * (len(opts) - 1) + [1.0], variant="s4_unanswerable")


def derive_distractor(row, pool):
    """append N_DISTRACTORS options from other rows of the same source, highest word-Jaccard to the gold option first"""
    t = [float(x) for x in row["target"]]
    gold = row["options"][t.index(max(t))]
    have = {norm(o) for o in row["options"]}
    gw = words(gold)
    cands = []
    for rid, o in pool:
        if rid == row["id"] or norm(o) in have:
            continue
        ow = words(o)
        j = len(gw & ow) / max(1, len(gw | ow))
        cands.append((-j, sha1(f"{row['id']}|{o}"), o))
    cands.sort()
    add = []
    for _, _, o in cands:
        if norm(o) not in have and o not in add:
            add.append(o)
        if len(add) == N_DISTRACTORS:
            break
    if len(add) < N_DISTRACTORS:
        return None
    return dict(row, id=f"{row['id']}::distr", options=list(row["options"]) + add, target=t + [0.0] * len(add), variant="s4_distractor")


def build_eg(rows, excl, n_train=4000, n_val=600, seed=0, cap_frac=0.10, variant_frac=0.25):
    """-> (train records, val records, manifest). Base + variant are emitted together (one pair, one split)."""
    from build_jev_ext import Labeler
    lab = Labeler(0.01)
    drops, cand, opt_pool = collections.Counter(), [], collections.defaultdict(list)
    seen_ids = set()
    for r in rows:
        src = str(r.get("source") or "")
        if src not in EG_SOURCES:
            continue
        if str(r["id"]) in seen_ids:
            drops["duplicate_id_across_configs"] += 1
            continue
        seen_ids.add(str(r["id"]))
        why = licence_status(r)
        if why:
            drops[f"licence:{why}"] += 1
            continue
        if holdout_of(src):
            drops["holdout"] += 1
            continue
        st = str(r.get("state") or "")
        if not st.strip():
            drops["empty_state"] += 1
            continue
        if len(st) + sum(len(str(o)) for o in (r.get("options") or [])) > MAX_STATE_CHARS:
            drops["too_long"] += 1
            continue
        if r.get("kind") == "choice":
            for o in r.get("options") or []:
                opt_pool[src].append((str(r["id"]), str(o)))
        cand.append((sha1(f"screen4/v2|eg|{seed}|{r['id']}"), r))
    cand.sort(key=lambda x: x[0])
    cap_tr, cap_va = max(1, math.ceil(cap_frac * n_train)), max(1, math.ceil(cap_frac * n_val))
    cnt = {"train": collections.Counter(), "val": collections.Counter()}
    out = {"train": [], "val": []}
    n = {"train": 0, "val": 0}
    want = {"train": n_train, "val": n_val}
    seen_sig, overlap = set(), collections.Counter()
    pairs = collections.Counter()
    for _, r in cand:
        if n["train"] >= n_train and n["val"] >= n_val:
            break
        src = str(r["source"])
        fam, variants_ok = EG_SOURCES[src]
        gid = str(r.get("group_id") or r["id"])
        split = "val" if int(sha1(f"screen4/v2|egsplit|{gid}"), 16) % 100 < EG_VAL_PCT else "train"
        if n[split] >= want[split]:
            continue
        if cnt[split][src] >= (cap_tr if split == "train" else cap_va):
            drops["cap"] += 1
            continue
        hit = excl.hit(r["id"], r["state"], f"{src}:{gid}") if excl is not None else None
        if hit:
            drops["overlap"] += 1
            overlap[hit] += 1
            continue
        sig = (norm(r["state"]), norm(r.get("question") or ""))
        if sig in seen_sig:
            drops["duplicate_decision"] += 1
            continue
        group = [("base", r)]
        if variants_ok and single_gold(r):
            u = int(sha1(f"screen4/v2|var|{seed}|{r['id']}"), 16) % 10000 / 10000.0
            if u < variant_frac / 2:
                group.append(("unanswerable", derive_unanswerable(r)))
            elif u < variant_frac:
                dv = derive_distractor(r, opt_pool[src])
                if dv is not None:
                    group.append(("distractor", dv))
        recs = []
        for member, row in group:
            rec = ts_record(row, lab, "eg", {"family": fam, "member": member, "pair_id": str(r["id"])})
            if rec is None:
                recs = []
                break
            recs.append(rec)
        if not recs:
            drops["convert"] += 1
            continue
        u = sum(units_of(x) for x in recs)
        if n[split] + u > want[split]:
            if len(recs) > 1:          # keep the base alone rather than overshoot
                recs = recs[:1]
                u = 1
            if n[split] + u > want[split]:
                continue
        seen_sig.add(sig)
        for x in recs:
            pairs[x["meta"]["screen4"]["member"]] += 1
        out[split].extend(recs)
        n[split] += u
        cnt[split][src] += u
    if n["train"] < n_train or n["val"] < n_val:
        raise SystemExit(f"EG pool too small: train {n['train']}/{n_train}, val {n['val']}/{n_val} ({dict(drops)})")
    man = {"version": VERSION, "revision": TS_REVISION, "n_train": n["train"], "n_val": n["val"], "seed": seed, "drops": dict(drops),
           "convert_drops": dict(lab.drops), "overlap_tags": dict(overlap.most_common()), "members": dict(pairs),
           "train_by_source": dict(cnt["train"].most_common()), "val_by_source": dict(cnt["val"].most_common()),
           "families": {s: f for s, (f, _) in EG_SOURCES.items()}, "cap": [cap_tr, cap_va], "max_state_chars": MAX_STATE_CHARS}
    return out["train"], out["val"], man


def didev_block(source):
    for b, prefixes in DIDEV_BLOCKS.items():
        if any(source == p or source.startswith(p) for p in prefixes):
            return b
    return None


def build_didev(rows, excl, per_block=300, per_target=500, seed=0, min_nontarget=DIDEV_MIN_NONTARGET, count_only=False):
    """-> ({block: records}, manifest). Dev only: licence ignored (eval use), every overlap (eval or corpus) dropped.
    Each block takes up to its quota (target per_target, non-target per_block) of filtered records in the seeded order. AMEND A2: a
    TARGET block must reach per_target exactly; a NON-TARGET block may stop short down to min_nontarget (its actual n is recorded in
    `blocks` / `short_blocks`; the reader reports per-block n and CIs). Every failing block is named in ONE error (no first-failure stop).
    count_only: no quotas, no failure: the manifest's `eligible` holds every block's full post-filter count (the A2 diagnostic)."""
    from build_jev_ext import Labeler
    lab = Labeler(0.01)
    drops, cand = collections.Counter(), collections.defaultdict(list)
    for r in rows:
        if r.get("_config") != "full":
            continue
        b = didev_block(str(r.get("source") or ""))
        if b is None:
            continue
        cand[b].append((sha1(f"screen4/v2|didev|{seed}|{r['id']}"), r))
    out, overlap, convert = {}, collections.Counter(), collections.Counter()
    for b in DIDEV_BLOCKS:
        want = None if count_only else (per_target if b in DIDEV_TARGET else per_block)
        recs, seen = [], set()
        for _, r in sorted(cand.get(b, []), key=lambda x: x[0]):
            if want is not None and len(recs) >= want:
                break
            if len(str(r.get("state") or "")) + sum(len(str(o)) for o in (r.get("options") or [])) > MAX_STATE_CHARS:
                drops[f"{b}:too_long"] += 1
                continue
            hit = excl.hit(r["id"], r["state"], f"{r.get('source')}:{r.get('group_id') or r['id']}") if excl is not None else None
            if hit:
                drops[f"{b}:overlap"] += 1
                overlap[hit] += 1
                continue
            sig = (norm(r["state"]), norm(r.get("question") or ""))
            if sig in seen:
                drops[f"{b}:duplicate_decision"] += 1
                continue
            before = dict(lab.drops)
            rec = ts_record(r, lab, "dev", {"block": b, "target_block": b in DIDEV_TARGET})
            if rec is None:
                drops[f"{b}:convert"] += 1
                for k, v in lab.drops.items():
                    if v != before.get(k, 0):
                        convert[f"{b}:{k}"] += v - before.get(k, 0)
                continue
            seen.add(sig)
            recs.append(rec)
        out[b] = recs
    man = {"version": VERSION, "revision": TS_REVISION, "target_blocks": list(DIDEV_TARGET), "drops": dict(drops),
           "convert_drops": dict(convert), "overlap_tags": dict(overlap.most_common()), "seed": seed, "candidates": {b: len(cand.get(b, []))
                                                                                                              for b in DIDEV_BLOCKS}}
    if count_only:
        man["eligible"] = {b: len(v) for b, v in out.items()}
        return out, man
    fail = {}
    for b, recs in out.items():
        need = per_target if b in DIDEV_TARGET else min(min_nontarget, per_block)
        if len(recs) < need:
            fail[b] = (len(recs), need)
    if fail:
        raise SystemExit("DI-family dev blocks short after the filters: " + "; ".join(f"{b}: {n} < {k}" for b, (n, k) in fail.items()))
    man["blocks"] = {b: len(v) for b, v in out.items()}
    man["short_blocks"] = {b: len(v) for b, v in out.items() if b not in DIDEV_TARGET and len(v) < per_block}
    man["rule"] = f"target blocks exactly {per_target}; non-target blocks up to {per_block}, at least {min(min_nontarget, per_block)} (AMEND A2)"
    return out, man


def build_caldev(val_dir, n=3000, seed=77):
    """a seeded sample of mix2 val UNITS (records kept whole; greedy exact fill), NC records dropped"""
    recs = [r for f in sorted(Path(val_dir).glob("*.jsonl")) for r in read_jsonl(f)]
    recs = [r for r in recs if not NC_LICENCE.search(str(r.get("license") or "")) and "ms_marco" not in str(r.get("source") or "")]
    recs.sort(key=lambda r: sha1(f"screen4/v2|caldev|{seed}|{r['id']}"))
    out, tot = [], 0
    for r in recs:
        u = units_of(r)
        if tot + u <= n:
            out.append(r)
            tot += u
        if tot == n:
            break
    if tot != n:
        raise SystemExit(f"caldev: {tot} != {n} units")
    return out


# ============================================================================ replay subsets and the bayes pool
def replay_records(d):
    return [r for f in sorted(Path(d).glob("*.jsonl")) for r in read_jsonl(f)]


def replay_subset(recs, n, seed=0):
    """records of R2 in a seeded order, greedy exact fill to n units (skipping a record that would overshoot)"""
    order = sorted(recs, key=lambda r: sha1(f"screen4/v2|r2|{seed}|{r['id']}"))
    out, tot = [], 0
    for r in order:
        u = units_of(r)
        if tot + u <= n:
            out.append(r)
            tot += u
        if tot == n:
            break
    if tot != n:
        raise SystemExit(f"replay subset: {tot} != {n} units")
    return out


# AMEND A1: arm d's 1,500 Bayes units = 1,000 retrieval-style (A2: HoVer / BRIGHT competencies, the board's largest loss) + 500 numeric
# (A1: prior use; PCD found its bench cost grows with the AMOUNT, ATTR_NUMERIC, so a small share), every family of a block round-robin
BAYES_BLOCKS = {
    "ret": (("ret_hard_negatives", "ret_answerability", "ret_missing_evidence", "ret_distractors", "ret_evidence_strength"), 1000),
    "num": (("prior_change", "evidence_strength", "evidence_removal", "irrelevant_evidence", "loss_change"), 500),
}


def reachable(n, sizes):
    """can a sub-multiset of `sizes` sum to exactly n (n small)?"""
    can = 1
    for z in sorted(sizes):
        if z <= n:
            can |= can << z
            can &= (1 << (n + 1)) - 1
        if can >> n & 1:
            return True
    return bool(can >> n & 1)


def bayes_pool(fam_dir, blocks=None, seed=0, excl=None, max_chars=MAX_STATE_CHARS):
    """-> (records, manifest). The Bayesian-evidence families' train split (INTERFACE.md v1): WHOLE groups (meta.bf.group) only; a group
    is dropped if ANY member is outside train / carries a holdout, hits an exclusion (sealed / confirmation ids, state sha1s, clusters),
    exceeds max_chars (state + options + instructions: the 2,048-token prompt must never cut evidence), has K > 52 or a target that is
    not a distribution, or carries an NC licence; the families of each block take groups round-robin (seeded group order) and every
    block is filled EXACTLY (a group that would overshoot is skipped)."""
    blocks = blocks or BAYES_BLOCKS
    root = Path(fam_dir)
    groups, drops, hits = collections.defaultdict(list), collections.Counter(), collections.Counter()
    for f in sorted((root / "train").glob("bf_*.jsonl")):
        for r in read_jsonl(f):
            bf = (r.get("meta") or {}).get("bf") or {}
            if bf.get("split") != "train" or bf.get("holdout") not in (None, "none"):
                raise SystemExit(f"bayes record {r.get('id')}: split {bf.get('split')} holdout {bf.get('holdout')} in train/")
            groups[(bf.get("family"), bf.get("group"))].append(r)
    bad = set()
    for key, rs in groups.items():
        for r in rs:
            why = None
            q = next(iter(r["questions"].values()))
            lab = next(iter(r["labels"].values()))
            vals = [float(v) for v in lab.values()]
            size = len(json.dumps(r["state"], ensure_ascii=False) if not isinstance(r["state"], str) else r["state"]) + \
                sum(len(str(x)) for x in (q.get("criteria") or {}).values()) + len(str(q.get("instructions") or ""))
            if NC_LICENCE.search(str(r.get("license") or "")):
                why = "nc_licence"
            elif len(r["questions"]) != 1 or len(option_keys(q)) > 52:
                why = "shape"
            elif any(not math.isfinite(v) or v < 0 for v in vals) or abs(sum(vals) - 1) > 1e-3:
                why = "labels"
            elif size > max_chars:
                why = "too_long"
            elif excl is not None:
                h = excl.hit(r["id"], r["state"], f"{r.get('source')}:{(r.get('meta') or {}).get('group')}")
                if h:
                    why = "exclusion"
                    hits[h] += 1
            if why:
                drops[why] += 1
                bad.add(key)
    by_fam = collections.defaultdict(list)
    for (fam, g), rs in groups.items():
        if (fam, g) not in bad:
            by_fam[fam].append((sha1(f"screen4/v2|bf|{seed}|{g}"), g, rs))
    for fam in by_fam:
        by_fam[fam].sort()
    out, per_fam = [], collections.Counter()
    for block, (fams, n_units) in blocks.items():
        missing = [f for f in fams if not by_fam.get(f)]
        if missing:
            raise SystemExit(f"bayes pool: no usable groups for {missing}")
        tot, it = 0, {f: 0 for f in fams}
        while tot < n_units and any(it[f] < len(by_fam[f]) for f in fams):
            for f in fams:
                if tot == n_units or it[f] >= len(by_fam[f]):
                    continue
                _, g, rs = by_fam[f][it[f]]
                it[f] += 1
                u = sum(units_of(r) for r in rs)
                rest = n_units - tot - u
                if rest < 0 or (rest < 64 and rest and not reachable(rest, [sum(units_of(r) for r in x[2]) for f2 in fams
                                                                          for x in by_fam[f2][it[f2]:]])):
                    continue                  # overshoot, or the remainder could not be filled exactly by the unvisited groups
                for r in sorted(rs, key=lambda x: x["id"]):
                    r = json.loads(json.dumps(r))
                    r["meta"]["screen4"] = {"role": "bayes", "family": f, "block": block, "pair_id": str(g),
                                            "member": r["meta"]["bf"].get("member")}
                    out.append(r)
                tot += u
                per_fam[f] += u
        if tot != n_units:
            raise SystemExit(f"bayes pool: block {block} filled {tot} != {n_units} units")
    man = {"blocks": {b: {"families": list(f), "units": n} for b, (f, n) in blocks.items()}, "units_by_family": dict(per_fam),
           "groups_dropped": len(bad), "drops_records": dict(drops), "exclusion_hits": dict(hits.most_common()), "max_chars": max_chars,
           "seed": seed}
    return out, man


# ============================================================================ arms
def tag_role(recs, role, prefix=None):
    out = []
    for r in recs:
        r = json.loads(json.dumps(r))
        r.setdefault("meta", {}).setdefault("screen4", {})["role"] = role
        out.append(r)
    return out


def write_arm(out, name, parts, val_dir):
    """parts: [(file stem, records)]; val = PC's mix2 val files verbatim (the trainer's identity-gate sample)"""
    d = Path(out) / f"arm_{name}"
    for stem, recs in parts:
        if recs:
            write_jsonl(d / "train" / f"{stem}.jsonl", recs)
    (d / "val").mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(val_dir).glob("*.jsonl")):
        (d / "val" / f.name).write_bytes(f.read_bytes())
    return d


def unit_list(recs):
    """[(uid json, record, qid)] in cf.iter_units order for these records (one file, record order)"""
    seen, out = collections.Counter(), []
    for r in recs:
        for qid in r["questions"]:
            key = [str(r["id"]), str(qid)]
            m = seen[tuple(key)]
            seen[tuple(key)] += 1
            out.append((json.dumps(key + [m] if m else key, ensure_ascii=False), r, qid))
    return out


def gold_vector(r, qid):
    keys = option_keys(r["questions"][qid])
    lab = r["labels"][qid]
    v = []
    for i, k in enumerate(keys):
        v.append(float(lab[k]) if k in lab else float(lab[str(i)]))
    return v


def apply_mixed(recs, new_targets):
    """records with labels replaced by the mixed targets (stored key order); label_kind soft where changed"""
    out = []
    by_id = collections.defaultdict(dict)
    for uid, r, qid in unit_list(recs):
        if uid in new_targets:
            by_id[r["id"]][qid] = new_targets[uid]
    for r in recs:
        r = json.loads(json.dumps(r))
        for qid, vec in by_id.get(r["id"], {}).items():
            keys = option_keys(r["questions"][qid])
            if len(keys) != len(vec):
                raise SystemExit(f"{r['id']} {qid}: mixed target K {len(vec)} != {len(keys)}")
            old = gold_vector(r, qid)
            if max(abs(a - b) for a, b in zip(old, vec)) > 0:
                r["labels"][qid] = {k: round(float(p), 10) for k, p in zip(keys, vec)}
                r["label_kind"][qid] = "soft"
                r["meta"].setdefault("screen4", {})["preserved"] = True
        out.append(r)
    return out


def build_plain_arms(replay_dir, eg_dir, bayes_recs, out, seed=0, bayes_man=None):
    """a, b (+ d's composition without targets) and the teacher dirs. -> manifest"""
    R = tag_role(replay_records(Path(replay_dir) / "R" / "train"), "replay")
    R2 = tag_role(replay_records(Path(replay_dir) / "R2" / "train"), "replay")
    eg = read_jsonl(Path(eg_dir) / "train" / "eg.jsonl")
    egv = read_jsonl(Path(eg_dir) / "val" / "eg.jsonl")
    if sum(map(units_of, R)) != 6000 or sum(map(units_of, R2)) != 6000:
        raise SystemExit("PC's replay must hold 6,000 + 6,000 units")
    if sum(map(units_of, eg)) < 4000:
        raise SystemExit("EG pool below 4,000 units")
    R2b, R2d = replay_subset(R2, 2000, seed), replay_subset(R2, 500, seed)
    if not {r["id"] for r in R2d} <= {r["id"] for r in R2b}:
        raise SystemExit("R2[500] is not a subset of R2[2,000]")
    val = Path(replay_dir) / "val"
    write_arm(out, "a", [("replay_R", R), ("replay_R2", R2)], val)
    write_arm(out, "b", [("replay_R", R), ("replay_R2", R2b), ("eg", eg)], val)
    write_jsonl(Path(out) / "teach_replay" / "test" / "replay.jsonl", R + R2b)          # (99)'s teacher pass (covers c and d)
    write_jsonl(Path(out) / "teach_eg" / "test" / "eg.jsonl", eg)                     # stock Gemma's teacher pass
    write_jsonl(Path(out) / "teach_eg" / "dev" / "eg_val.jsonl", egv)                 # its temperature fit (EG dev)
    comp = {"a": (R, R2, [], []), "b": (R, R2b, eg, []), "c": (R, R2b, eg, []), "d": (R, R2d, eg, bayes_recs)}
    man = {"version": VERSION, "seed": seed, "spec": ARM_SPEC,
           "units": {k: {"R": sum(map(units_of, v[0])), "R2": sum(map(units_of, v[1])), "EG": sum(map(units_of, v[2])),
                         "bayes": sum(map(units_of, v[3]))} for k, v in comp.items()},
           "r2_subset_ids": {"2000": [r["id"] for r in R2b], "500": [r["id"] for r in R2d]}}
    for k, v in man["units"].items():
        if sum(v.values()) != N_TOTAL:
            raise SystemExit(f"arm {k}: {sum(v.values())} units != {N_TOTAL}")
    if bayes_recs:
        write_jsonl(Path(out) / "bayes_pool" / "bayes.jsonl", bayes_recs)
        man["bayes_pool"] = bayes_man
    man["files_plain"] = {k: file_table(Path(out) / k) for k in ("arm_a", "arm_b", "teach_replay", "teach_eg")}
    (Path(out) / "arms_plain_manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    return man


def build_mixed_arms(replay_dir, eg_dir, out, parent_npz, teacher_npz, teacher_dev_npz, seed=0, lam_parent=None, lam_teacher=None):
    """c and d with preservation targets (screen4_loss.plan_targets). -> manifest"""
    import screen4_loss as sl
    lam_parent = sl.LAM_PARENT if lam_parent is None else lam_parent
    lam_teacher = sl.LAM_TEACHER if lam_teacher is None else lam_teacher
    pman = json.load(open(Path(out) / "arms_plain_manifest.json"))
    # AMEND A3: PC's replay files keep their source option order; the plain phase wrote them (arm b, the teacher dir) in sorted-key order
    # (write_jsonl), and the teacher scored THAT. Canonicalise first, then require the teacher dirs to be byte-identical to what is mixed.
    R = [canon(r) for r in tag_role(replay_records(Path(replay_dir) / "R" / "train"), "replay")]
    R2l = [canon(r) for r in tag_role(replay_records(Path(replay_dir) / "R2" / "train"), "replay")]
    if len({r["id"] for r in R + R2l}) != len(R) + len(R2l):
        raise SystemExit("PC's replay repeats a record id: the id-keyed R2 subsets and mixed targets would be ambiguous")
    R2 = {r["id"]: r for r in R2l}
    R2b = [R2[i] for i in pman["r2_subset_ids"]["2000"]]
    R2d = [R2[i] for i in pman["r2_subset_ids"]["500"]]
    eg = [canon(r) for r in read_jsonl(Path(eg_dir) / "train" / "eg.jsonl")]
    egv = [canon(r) for r in read_jsonl(Path(eg_dir) / "val" / "eg.jsonl")]
    for rel, recs in (("teach_replay/test/replay.jsonl", R + R2b), ("teach_eg/test/eg.jsonl", eg), ("teach_eg/dev/eg_val.jsonl", egv)):
        f = Path(out) / rel
        if not f.exists() or f.read_bytes() != "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in recs).encode("utf-8"):
            raise SystemExit(f"{rel} (what the teacher scored) is not byte-identical to the records being mixed (AMEND A3)")
    bpath = Path(out) / "bayes_pool" / "bayes.jsonl"
    bayes = read_jsonl(bpath) if bpath.exists() else []
    # teacher rows, exact-joined on the teacher dirs' own unit enumeration
    tr_units = unit_list(R + R2b)
    parent = sl.teacher_rows(parent_npz, [u for u, _, _ in tr_units], [gold_vector(r, q) for _, r, q in tr_units],
                             [len(gold_vector(r, q)) for _, r, q in tr_units])
    eg_units = unit_list(eg)
    teacher = sl.teacher_rows(teacher_npz, [u for u, _, _ in eg_units], [gold_vector(r, q) for _, r, q in eg_units],
                              [len(gold_vector(r, q)) for _, r, q in eg_units])
    ev_units = unit_list(egv)
    tdev = sl.teacher_rows(teacher_dev_npz, [u for u, _, _ in ev_units], [gold_vector(r, q) for _, r, q in ev_units],
                           [len(gold_vector(r, q)) for _, r, q in ev_units])
    T = sl.fit_temperature([tdev[u] for u, _, _ in ev_units], [gold_vector(r, q) for _, r, q in ev_units])
    units = [{"_uid": u, "target": gold_vector(r, q)} for u, r, q in tr_units + eg_units]
    roles = {u: "replay" for u, _, _ in tr_units}
    roles.update({u: "eg" for u, _, _ in eg_units})
    new, st = sl.plan_targets(units, roles, parent, teacher, T, lam_parent, lam_teacher)
    # the uids are those of the teacher dirs; arm files hold the same records, so uids coincide (no repeated (id, qid) pairs: checked)
    if len({u for u, _, _ in tr_units + eg_units}) != len(tr_units) + len(eg_units):
        raise SystemExit("repeated (record id, qid) across replay + EG: uids would be ambiguous")
    Rm, R2bm, egm = apply_mixed(R, new), apply_mixed(R2b, new), apply_mixed(eg, new)
    R2dm = [r for r in R2bm if r["id"] in {x["id"] for x in R2d}]
    val = Path(replay_dir) / "val"
    write_arm(out, "c", [("replay_R", Rm), ("replay_R2", R2bm), ("eg", egm)], val)
    write_arm(out, "d", [("replay_R", Rm), ("replay_R2", R2dm), ("eg", egm), ("bayes", bayes)], val)
    man = {"version": VERSION, "lam_parent": lam_parent, "lam_teacher": lam_teacher, "teacher_T": T, "stats": st,
           "parent_npz_sha256": sha256_file(parent_npz), "teacher_npz_sha256": sha256_file(teacher_npz),
           "teacher_dev_npz_sha256": sha256_file(teacher_dev_npz), "files_mixed": {k: file_table(Path(out) / k) for k in ("arm_c", "arm_d")},
           "bayes_units": sum(map(units_of, bayes))}
    if pman["units"]["d"]["bayes"] and not bayes:
        raise SystemExit("arm d needs the bayes pool")
    (Path(out) / "arms_mixed_manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    return man


def _without_targets(r):
    r = json.loads(json.dumps(r))
    r.pop("labels", None)
    r.pop("label_kind", None)
    ((r.get("meta") or {}).get("screen4") or {}).pop("preserved", None)
    return json.dumps(r, ensure_ascii=False)          # key ORDER kept: a reordered option list is a difference


def arm_records(out, arm):
    return [r for f in sorted((Path(out) / f"arm_{arm}" / "train").glob("*.jsonl")) for r in read_jsonl(f)]


def check_arms(out, dev_dirs=(), expect=None):
    """-> problems (empty = OK)"""
    out, probs = Path(out), []
    pman = json.load(open(out / "arms_plain_manifest.json"))
    arms = [a for a in "abcd" if (out / f"arm_{a}").exists()]
    recs = {a: arm_records(out, a) for a in arms}
    for a in arms:
        n = sum(map(units_of, recs[a]))
        if n != N_TOTAL:
            probs.append(f"arm {a}: {n} units != {N_TOTAL}")
        roles = collections.Counter(r["meta"]["screen4"]["role"] for r in recs[a] for _ in r["questions"])
        want = pman["units"][a]
        if roles.get("replay", 0) != want["R"] + want["R2"] or roles.get("eg", 0) != want["EG"] or roles.get("bayes", 0) != want["bayes"]:
            probs.append(f"arm {a}: roles {dict(roles)} != spec {want}")
        nc = [r["id"] for r in recs[a] if NC_LICENCE.search(str(r.get("license") or ""))]
        if nc:
            probs.append(f"arm {a}: {len(nc)} NC records")
    ids = {a: {r["id"] for r in recs[a]} for a in arms}
    if "a" in ids and "b" in ids:
        rb = {r["id"] for r in recs["b"] if r["meta"]["screen4"]["role"] == "replay"}
        if not rb <= ids["a"]:
            probs.append("b's replay is not a subset of a's")
    for x, y in (("c", "b"), ("d", "c")):
        if x in ids and y in ids:
            rx = {r["id"] for r in recs[x] if r["meta"]["screen4"]["role"] != "bayes"}
            if not rx <= ids[y]:
                probs.append(f"{x}'s non-bayes records are not a subset of {y}'s")
    if "c" in ids and "b" in ids and ids["c"] != ids["b"]:
        probs.append("c and b differ in records (c must be b with mixed targets)")
    if "c" in recs and "b" in recs:   # AMEND A3: record for record, same options in the same order; targets within the mixing bound
        mp = out / "arms_mixed_manifest.json"
        lam = max(json.load(open(mp)).get(k, 0.3) for k in ("lam_parent", "lam_teacher")) if mp.exists() else 0.3
        if len(recs["c"]) != len(recs["b"]) or any(_without_targets(x) != _without_targets(y) for x, y in zip(recs["c"], recs["b"])):
            probs.append("c is not b record for record (same records, order and options; only targets may differ) (AMEND A3)")
        else:
            far = sum(1 for x, y in zip(recs["c"], recs["b"]) for q in y["questions"]
                      if max(abs(u - v) for u, v in zip(gold_vector(x, q), gold_vector(y, q))) > lam / (1 + lam) + 2e-3)
            if far:
                probs.append(f"c: {far} units' targets move more than lam / (1 + lam) from b's gold (AMEND A3)")
    if "d" in recs and "c" in recs:
        cj = {r["id"]: json.dumps(r, sort_keys=True) for r in recs["c"]}
        bad = sum(1 for r in recs["d"] if r["meta"]["screen4"]["role"] != "bayes" and cj.get(r["id"]) != json.dumps(r, sort_keys=True))
        if bad:
            probs.append(f"d: {bad} non-bayes records differ from c's (AMEND A3)")
    for a in [x for x in ("a", "b") if x in recs]:
        if any(r["meta"]["screen4"].get("preserved") for r in recs[a]):
            probs.append(f"arm {a} carries preservation targets")
    pairs = collections.defaultdict(list)
    for a in arms:
        for r in recs[a]:
            s4 = r["meta"]["screen4"]
            if s4["role"] == "eg":
                pairs[(a, s4.get("pair_id"))].append(s4.get("member"))
    bad = [k for k, m in pairs.items() if m.count("base") != 1]
    if bad:
        probs.append(f"{len(bad)} EG pairs without exactly one base member")
    if "d" in recs:
        bm = (pman.get("bayes_pool") or {}).get("units_by_family") or {}
        got = collections.Counter(r["meta"]["screen4"]["family"] for r in recs["d"] if r["meta"]["screen4"]["role"] == "bayes"
                                  for _ in r["questions"])
        if bm and dict(got) != bm:
            probs.append(f"arm d: bayes units by family {dict(got)} != the pool manifest {bm}")
        ids_d = [r["id"] for r in recs["d"] if r["meta"]["screen4"]["role"] == "bayes"]
        if len(ids_d) != len(set(ids_d)):
            probs.append("arm d: a bayes record appears twice")
    vals = {a: file_table(out / f"arm_{a}" / "val") for a in arms}
    if len({json.dumps(v, sort_keys=True) for v in vals.values()}) > 1:
        probs.append("val files differ between arms")
    train_states = {norm(r["state"]) for a in arms for r in recs[a]}
    train_ids = {r["id"] for a in arms for r in recs[a]}
    for d in dev_dirs:
        for f in sorted(Path(d).rglob("*.jsonl")):
            for r in read_jsonl(f):
                if r["id"] in train_ids or norm(r["state"]) in train_states:
                    probs.append(f"dev {f.name}: record {r['id']} is in a training arm")
                    break
    if expect:
        for k, v in expect.items():
            if file_table(out / k) != v:
                probs.append(f"{k}: files differ from the registered pins")
    return probs


# ============================================================================ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("eg", "didev", "caldev", "arms", "check"))
    ap.add_argument("--ts", help="dir holding the pinned tasksource shards (filtered-full/ and full/)")
    ap.add_argument("--shard-pins", help="JSON {relative path: sha256}")
    ap.add_argument("--no-shard-check", action="store_true", help="tests only")
    ap.add_argument("--eval", action="append", default=[], help="TAG=PATH eval jsonl dir / file")
    ap.add_argument("--corpus", action="append", default=[], help="TAG=PATH training corpus (X99 parts)")
    ap.add_argument("--fetch-evals", action="store_true")
    ap.add_argument("--allow-missing-evals", action="store_true", help="tests only (no network): build without the fetched eval families")
    ap.add_argument("--sealed", action="append", default=[])
    ap.add_argument("--confirm-index")
    ap.add_argument("--no-confirm-index", action="store_true")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--cap-frac", type=float, default=0.10, help="EG: at most this share of n-train per tasksource source")
    ap.add_argument("--didev-per-block", type=int, default=300)
    ap.add_argument("--didev-per-target", type=int, default=500)
    ap.add_argument("--didev-min-nontarget", type=int, default=DIDEV_MIN_NONTARGET)
    ap.add_argument("--count-only", action="store_true", help="didev: report every block's post-filter count; write nothing")
    ap.add_argument("--replay", help="pc_replay_v1 dir (R/, R2/, val/)")
    ap.add_argument("--eg", help="EG pool dir")
    ap.add_argument("--bayes", help="bayes_families dir (arm d)")
    ap.add_argument("--phase", choices=("plain", "mixed"))
    ap.add_argument("--parent-npz")
    ap.add_argument("--teacher-npz")
    ap.add_argument("--teacher-dev-npz")
    ap.add_argument("--dev", action="append", default=[], help="dev dirs that must not overlap any arm (check)")
    ap.add_argument("--expect")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.cmd or not a.out:
        ap.error("a command and --out are required")
    out, t0 = Path(a.out), time.time()
    pins = json.load(open(a.shard_pins)) if a.shard_pins else {}
    if a.cmd in ("eg", "didev"):
        if (out / "manifest.json").exists():
            raise SystemExit(f"{out} already holds a build")
        excl = load_exclusions(a)
        rows = iter_ts_rows(a.ts, pins, check_sha=not a.no_shard_check)
        if a.cmd == "eg":
            tr, va, man = build_eg(rows, excl, a.n_train, a.n_val, a.seed, a.cap_frac)
            write_jsonl(out / "train" / "eg.jsonl", tr)
            write_jsonl(out / "val" / "eg.jsonl", va)
        else:
            blocks, man = build_didev(rows, excl, a.didev_per_block, a.didev_per_target, seed=a.seed, min_nontarget=a.didev_min_nontarget,
                                      count_only=a.count_only)
            if a.count_only:
                print("DIDEV_COUNTS " + json.dumps({k: man[k] for k in ("eligible", "candidates", "drops", "convert_drops")}))
                return 0
            for b, recs in blocks.items():
                write_jsonl(out / "test" / f"didev_{b}.jsonl", recs)
        man["exclusions"] = {"eval_coverage": excl.coverage, "sizes": dict(excl.idx.sizes) if excl.idx else {}, "indexes": excl.info}
        man["shard_pins"] = pins
        man["files"] = file_table(out)
        (out / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
        print(f"{a.cmd.upper()}_OK {json.dumps({k: man[k] for k in man if k in ('n_train', 'n_val', 'blocks', 'drops')})} "
              f"({time.time() - t0:.0f}s)")
    elif a.cmd == "caldev":
        recs = build_caldev(Path(a.replay) / "val", seed=77)
        write_jsonl(out / "test" / "caldev.jsonl", recs)
        print(f"CALDEV_OK {sum(map(units_of, recs))} units")
    elif a.cmd == "arms":
        if a.phase == "plain":
            bayes, bman = [], None
            if a.bayes:
                excl = load_exclusions(a)
                bayes, bman = bayes_pool(a.bayes, BAYES_BLOCKS, a.seed, excl)
                if sum(map(units_of, bayes)) != ARM_SPEC["d"][3]:
                    raise SystemExit("the bayes blocks do not add up to arm d's Bayes units")
            man = build_plain_arms(a.replay, a.eg, bayes, out, a.seed, bman)
            print(f"ARMS_PLAIN_OK {json.dumps(man['units'])}")
        else:
            man = build_mixed_arms(a.replay, a.eg, out, a.parent_npz, a.teacher_npz, a.teacher_dev_npz, a.seed)
            print(f"ARMS_MIXED_OK T {man['teacher_T']:.4f} stats {man['stats']}")
        probs = check_arms(out, a.dev)
        if probs:
            raise SystemExit(f"ARMS_CHECK_FAILED {probs}")
    elif a.cmd == "check":
        probs = check_arms(out, a.dev, json.load(open(a.expect)) if a.expect else None)
        print("CHECK_OK" if not probs else f"CHECK_FAILED {probs}")
        return 1 if probs else 0
    return 0


def self_test():
    assert holdout_of("civil_comments/toxicity_share") == "bench4:civil_comments"
    assert holdout_of("hover") == "DI:hover" and holdout_of("esci") == "DI:esci" and holdout_of("anli/a1") == "DI:anli"
    assert holdout_of("WANLI") is None and holdout_of("vitaminc") is None and holdout_of("wands") is None
    assert holdout_of("procedural-typed-decisions/evidence_sufficiency") is None
    assert holdout_of("fever-evidence-related") == "jevbench:fever_evidence" and holdout_of("fool-me-twice") is None
    assert all(holdout_of(s) is None for s in EG_SOURCES), [s for s in EG_SOURCES if holdout_of(s)]
    assert all(holdout_of(p.rstrip("/")) for ps in DIDEV_BLOCKS.values() for p in ps), [p for ps in DIDEV_BLOCKS.values() for p in ps
                                                                                       if not holdout_of(p.rstrip("/"))]
    assert licence_status({"license": "cc-by-sa-4.0", "license_use": "commercial"}) is None
    assert licence_status({"license": "cc-by-nc-4.0", "license_use": "commercial"}) == "nc_licence_string"
    assert licence_status({"license": "apache-2.0, Custom (DPI)", "license_use": "commercial"}) == "custom_or_unspecified_licence"
    assert licence_status({"license": "mit", "license_use": "unspecified"}) == "license_use_unspecified"
    for s in ARM_SPEC.values():
        assert sum(s[:4]) == N_TOTAL
    assert sum(n for _, n in BAYES_BLOCKS.values()) == ARM_SPEC["d"][3]
    assert reachable(13, [4, 5, 8, 12]) and not reachable(11, [4, 5, 8, 12]) and reachable(0, []) and not reachable(3, [4, 4])
    print("SELF_TEST_OK build_screen4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
