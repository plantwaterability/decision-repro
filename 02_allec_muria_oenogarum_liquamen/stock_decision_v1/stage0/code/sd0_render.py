#!/usr/bin/env python3
"""Stage 0 render + decision rows (stock_decision_v1). For a selection (pilot exposures or a panel) render every row with the R2
renderer exactly as the trainer does (trainers/fullft_render.py: renderer_spec / pool_units / was_cut with the pinned production
scorer claude_lmscore e0672bd0 build_prompt, noul_defs=True, max_len 8,192, pinned Gemma tokenizer) under the row's FROZEN option
order, and write
  decisions_<tag>.jsonl.gz : one row per unique decision with the roadmap's required fields (state, question, kind, ordered options,
                             gold by semantic option key, identities, hashes, rendered token count, K, weight; teacher fields empty)
  exposures_<tag>.jsonl.gz : one row per exposure (pos, decision_uid, exposure_k, option_order, prompt_sha256, n_tokens)
  render_<tag>.json        : header (renderer spec + sha256, counts, strata, token totals, file sha256s)
Option order: choice questions get a permutation drawn from numpy default_rng(sha256(seed|mixture|uid|k)); noul / score keep the stored
(semantic) order. A state that would have to be cut is refused. --sealed: print counts only (confirmation panel).

  python3 sd0_render.py --sel SEL.parquet --tag M0 --mixture M0 --pools NAME=DIR ... --code PRODDIR --tokenizer TOKDIR \
        --fullft fullft_render.py --meta pool_meta.json --out DIR [--sealed]
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

TOK_REV = "google/gemma-4-31B-it@842da3794eaa0b77d5f08bae87a17459d91ff475"


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canon(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


SEM = {}


def sv(x):
    """a string field value or None (pandas gives NaN for missing values, and NaN is truthy)"""
    return x if isinstance(x, str) and x else None


def semantics(r, u, pm):
    """label semantics under the shared schema (stock_decision_v1/schema/label_semantics_v1.json, renderer_contract):
    target_basis = WHAT the gold means, label_verification = HOW its correctness was established, label_provenance = the Stage 0
    source-map label_basis verbatim. An unmapped label_basis refuses the row. LLM-reviewed upstream labels map to
    'upstream_llm_reviewed' (unverified tier), never to an independent class."""
    pmap = SEM["schema"]["stage0_provenance_map"]
    lb = r.label_basis
    if lb not in pmap:
        raise SystemExit(f"{r.uid}: label_basis {lb!r} has no stage0_provenance_map entry (refused, no default)")
    ent = dict(pmap[lb])
    tbq = sv(getattr(r, "target_basis_q", None))
    tsrc = "stage0_provenance_map[label_basis]"
    if isinstance(tbq, str) and tbq:             # the generator's per-question declaration wins over the source-level default
        ent["target_basis"] = tbq
        tsrc = str(getattr(r, "target_semantics_declared", "") or "declared")
    lvq = sv(getattr(r, "label_verification_q", None))
    if lvq:                                      # renderer_contract: a per-row verified value, when one exists, wins
        ent["label_verification"] = lvq
    if ent["target_basis"] not in SEM["schema"]["target_basis"] or ent["label_verification"] not in SEM["schema"]["label_verification"]:
        raise SystemExit(f"{r.uid}: target basis {ent['target_basis']!r} / verification {ent['label_verification']!r} not in the schema")
    key_ = f"{lb} -> {ent['target_basis']} / {ent['label_verification']}"
    SEM["audit"][key_] = SEM["audit"].get(key_, 0) + 1
    ev = [f"source_record:{u['record_id']}", f"source:{pm['source_repo'].split(' ')[0]}@{pm['source_revision'].split(' ')[0]}"]
    le = getattr(r, "label_evidence", None)
    if isinstance(le, str) and le:
        ev.append(f"label_evidence:{le}")
    if sv(r.generator_mechanism):
        ev.append(f"generator:{r.generator_mechanism}@{pm.get('generator_revision') or pm['source_revision']}")
    detail = [x for x in (sv(getattr(r, "m1_category", None)) and f"m1_category={r.m1_category}",
                          sv(getattr(r, "variant", None)) and f"tasksource_variant={r.variant}",
                          f"upstream_source={r.upstream_source}") if x]
    lo = sv(getattr(r, "loss_by_option", None))
    out_loss = json.loads(lo) if isinstance(lo, str) and lo else None
    return {"target_basis": ent["target_basis"], "target_basis_source": tsrc, "loss_by_option": out_loss,
            "label_verification": ent["label_verification"], "label_provenance": lb,
            "label_provenance_detail": "; ".join(detail), "evidence_refs": ev,
            "label_semantics_schema_sha256": SEM["sha256"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mixture", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--pools", action="append", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--fullft", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sealed", action="store_true")
    ap.add_argument("--schema", required=True, help="stock_decision_v1/schema/label_semantics_v1.json")
    ap.add_argument("--require-distinct-prompts", action="store_true", help="refuse two exposures of one decision with the same prompt")
    a = ap.parse_args()
    spec_ = importlib.util.spec_from_file_location("fullft_render", a.fullft)
    FR = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(FR)
    ls, md5s = FR.load_scorer(a.code)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rspec, ren = FR.renderer_spec(md5s, a.tokenizer, a.max_len, True, TOK_REV)
    pools = FR.kv_list(a.pools)
    meta = json.load(open(a.meta))
    SEM["schema"] = json.load(open(a.schema))
    if SEM["schema"].get("schema") != "sdv1-label-semantics-v1":
        raise SystemExit("--schema is not sdv1-label-semantics-v1")
    SEM["sha256"] = FR.sha256_file(a.schema)
    SEM["audit"] = {}
    sel = pd.read_parquet(a.sel)
    if "pos" not in sel:
        sel = sel.sort_values("uid").reset_index(drop=True)
        sel["pos"] = np.arange(len(sel))
    sel = sel.sort_values("pos").reset_index(drop=True)
    if "exposure_k" not in sel:
        sel["exposure_k"] = 0
    units = FR.pool_units(ls, pools, set(sel.uid))
    os.makedirs(a.out, exist_ok=True)
    dpath = os.path.join(a.out, f"decisions_{a.tag}.jsonl.gz")
    epath = os.path.join(a.out, f"exposures_{a.tag}.jsonl.gz")
    for p in (dpath, epath):
        if os.path.exists(p):
            raise SystemExit(f"{p} exists (immutable)")
    stats, seen = Counter(), set()
    prompts_by_uid = {}
    ntok_by = Counter()
    with gzip.open(dpath + ".partial", "wt", encoding="utf-8") as fd, gzip.open(epath + ".partial", "wt", encoding="utf-8") as fe:
        for r in sel.itertuples():
            u = units[r.uid]
            keys = [str(x) for x in u["option_keys"]]
            k = len(keys)
            given = getattr(r, "option_order_keys", None)
            if given is not None and not (isinstance(given, float) and np.isnan(given)) and len(given):
                given = [str(x) for x in given]               # an explicit per-exposure order (semantic keys) from the selection
                if sorted(given) != sorted(keys):
                    raise SystemExit(f"{a.tag} pos {r.pos}: given option order is not a permutation of {r.uid}'s keys")
                order = [keys.index(x) for x in given]
                if u["qtype"] != "choice" and order != list(range(k)):
                    raise SystemExit(f"{a.tag} pos {r.pos}: a reordered {u['qtype']} question")
            elif u["qtype"] == "choice" and k >= 2:
                s = int(sha(f"{a.seed}|{a.mixture}|{r.uid}|{int(r.exposure_k)}")[:16], 16)
                order = [int(x) for x in np.random.default_rng(s).permutation(k)]
                if int(r.exposure_k) > 0:                  # later exposures of a decision get a permutation different from exposure 0
                    s0 = int(sha(f"{a.seed}|{a.mixture}|{r.uid}|0")[:16], 16)
                    base = [int(x) for x in np.random.default_rng(s0).permutation(k)]
                    alt = 0
                    while order == base:
                        alt += 1
                        s = int(sha(f"{a.seed}|{a.mixture}|{r.uid}|{int(r.exposure_k)}|alt{alt}")[:16], 16)
                        order = [int(x) for x in np.random.default_rng(s).permutation(k)]
            else:
                order = list(range(k))
            text = ls.build_prompt(tok, u, order, a.max_len, noul_defs=True)
            if text is None or FR.was_cut(ls, tok, u, order, text, a.max_len, True):
                raise SystemExit(f"{a.tag} pos {r.pos}: does not fit {a.max_len} without cutting the state (refused)")
            ntok = len(tok(text, add_special_tokens=False)["input_ids"])
            ph = sha(text)
            if ph in prompts_by_uid.setdefault(r.uid, set()):
                stats["duplicate_prompt_exposures"] += 1
                if a.require_distinct_prompts:
                    raise SystemExit(f"{a.tag} pos {r.pos}: exposure {int(r.exposure_k)} of {r.uid} repeats an earlier exposure's prompt")
            prompts_by_uid[r.uid].add(ph)
            tv = np.asarray(u["target"], dtype=np.float64)
            if not np.isfinite(tv).all() or (tv < 0).any() or tv.sum() <= 0 or len(set(keys)) != k:
                raise SystemExit(f"{a.tag} pos {r.pos}: invalid target or duplicate keys")
            tv = tv / tv.sum()
            gold = {keys[i]: float(tv[i]) for i in range(k)}
            texts = [str(t) for t in ls.option_texts(u, True)]
            opts = [{"key": keys[i], "text": texts[i]} for i in range(k)]
            fe.write(json.dumps({"pos": int(r.pos), "exposure_id": f"{a.tag}:{int(r.pos)}", "decision_uid": r.uid,
                                 "exposure_k": int(r.exposure_k), "option_order": [keys[i] for i in order],
                                 "slot_map": {chr(65 + j) if j < 26 else chr(97 + j - 26): keys[o] for j, o in enumerate(order)},
                                 "prompt_sha256": sha(text), "n_tokens": ntok, "renderer_sha256": ren}, ensure_ascii=False) + "\n")
            stats["exposures"] += 1
            stats["tokens"] += ntok
            ntok_by[getattr(r, "len_bucket", "?")] += ntok
            if r.uid in seen:
                continue
            seen.add(r.uid)
            pm = meta["pools"][r.uid.split("|")[0]]
            rec_uid = r.uid.rsplit("|", 1)[0]
            row = {"decision_uid": r.uid, "record_uid": rec_uid, "parent_uid": f"{r.uid.split('|')[0]}|{r.parent_key}",
                   "ancestry_component_id": int(r.comp), "source_repo": pm["source_repo"], "source_revision": pm["source_revision"],
                   "source_record_id": str(u["record_id"]), "source_family": r.canonical_family, "upstream_source": r.upstream_source,
                   "skill": r.skill, "domain": r.domain, "area": r.area, "generator_family": sv(r.generator_mechanism),
                   "generator_revision": pm.get("generator_revision") if sv(r.generator_mechanism) else None, "split": a.split,
                   "state": u.get("state_text", u.get("state")), "question": u["instructions"], "kind": u["qtype"],
                   "ordered_options": opts, "semantic_order": u["qtype"] in ("score",), "gold_by_option_key": gold,
                   **semantics(r, u, pm),
                   "label_uncertainty": {"label_kind": u["label_kind"], "tmax": float(tv.max()), "hard": bool(tv.max() >= 0.999999)},
                   "license_use": {"licence": u["license"], "line": r.licence_line}, "teacher_runs": [],
                   "counterfactual_parent_uid": None, "renderer_sha256": ren, "prompt_sha256_first_exposure": sha(text),
                   "option_sha256": sha(canon(opts)), "gold_sha256": sha(canon(gold)), "tokenizer_revision": TOK_REV,
                   "rendered_token_count": ntok, "K": k, "k_bucket": r.k_bucket, "len_bucket": r.len_bucket, "weight": 1.0,
                   "di_ancestry": bool(r.di_ancestry), "m1_category": sv(getattr(r, "m1_category", None)),
                   "td_workflow_imitation": sv(getattr(r, "td_workflow_imitation", None)),
                   "parent_route": sv(getattr(r, "parent_route", None)), "m2_bucket": sv(getattr(r, "m2_bucket", None)),
                   "m2_sub_allocation": sv(getattr(r, "m2_sub_allocation", None)),
                   "di_target_benchmarks": sv(getattr(r, "di_target_benchmarks", None))}
            fd.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            stats["decisions"] += 1
    os.replace(dpath + ".partial", dpath)
    os.replace(epath + ".partial", epath)
    hdr = {"tag": a.tag, "mixture": a.mixture, "split": a.split, "renderer_sha256": ren, "renderer_spec": rspec,
           "renderer": "R2 = production build_prompt (claude_lmscore e0672bd0) with noul_defs=True, max_len 8192, margin 16, no truncation",
           "option_order_rule": "a selection column option_order_keys (explicit semantic-key order per exposure) wins; otherwise choice: numpy default_rng(sha256(seed|mixture|uid|exposure_k)[:16]) permutation (exposure_k > 0: redrawn with |altN until it differs from exposure 0); noul / score stored order",
           "seed": a.seed, "counts": dict(stats), "tokens_by_len_bucket": dict(ntok_by),
           "label_semantics": {"schema_sha256": SEM["sha256"], "conversion_counts": SEM["audit"]},
           "td_workflow_imitation_rows": int(sel["td_workflow_imitation"].notna().sum()) if "td_workflow_imitation" in sel else 0,
           "decisions_sha256": FR.sha256_file(dpath), "exposures_sha256": FR.sha256_file(epath),
           "decisions_md5": FR.md5_file(dpath), "exposures_md5": FR.md5_file(epath), "selection_sha256": FR.sha256_file(a.sel)}
    json.dump(hdr, open(os.path.join(a.out, f"render_{a.tag}.json"), "w"), indent=1, sort_keys=True, default=str)
    print(json.dumps({"tag": a.tag, "counts": dict(stats), "renderer_sha256": ren[:16]}))


if __name__ == "__main__":
    main()
