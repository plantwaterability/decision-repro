#!/usr/bin/env python3
"""(FULLFT) The RENDER MANIFEST: every training exposure's exact prompt, frozen BEFORE parent scoring and training, so the parent caches
(fullft_parent.py), preservation.py's join and the trainer (fullft_train.py) all see byte-identical prompts.

  render    --data DIR --splits train [--exposures EXP.jsonl] --tokenizer DIR --parents PARENTS.json --out ROWS.jsonl
            -> ROWS.jsonl (one row per exposure, exposure order) + ROWS.jsonl.header.json
  eligible  --rows ROWS.jsonl --out UNITS.jsonl [--route stock|garas]    the preservation-eligible rows (preservation.py --units)
  assemble  --rows ROWS.jsonl --joined JOINED.jsonl --out TRAIN.jsonl    eligible rows replaced by preservation.py's joined rows (1:1 by
                                                                          exposure_id, same order), ineligible rows unchanged
  render-freeze --pools NAME=DIR ... --exposures exposures_<R>.jsonl.gz --tokenizer DIR --parents PARENTS.json --out ROWS.jsonl
            (v2) a PILOT-DATA v2 freeze_r2 recipe: uid "<pool>|<member>|<line>|<qid>" (tokcount / unittable_v2), the FROZEN per-exposure
            option order, parent_route stock_gemma / garas_d99 / none, tokens_r1 re-checked. (v3) exposure_k = the freeze's replay_ix,
            its IDENTITY label (freeze_v2.py: decision uid#replay_ix; the option order is seeded by uid + replay_ix), NOT the
            occurrence count in pos order: B / C / D keep A's surviving replay_ix 1 exposure where B replaced the replay_ix 0 one, and
            positions interleave. (uid, replay_ix) must be unique.
  devset    --pools NAME=DIR ... --tables NAME=UNITS_TOKENS.jsonl.gz ... --hits NAME=HITS.jsonl.gz ... --mix exposures_<R>.jsonl.gz
  render-vision POOLDIR --tokenizer DIR --processor DIR [--pixels-from MAP.json] --pixel-dir DIR --out ROWS.jsonl
            (v5, vision round 1) a v2 IMAGE-conditioned render manifest (research/vision_trainer_plan_2026-11-06.md): every
            unit has exactly one image; pixels either come from the PINNED processor now (mode A: PIL decode +
            processor pixel_values -> PIXEL-DIR/<image sha256>.pt) or from an ALREADY-STAGED --pixels-from MAP sha256->.pt
            (mode B: shape checked only). The prompt is built exactly as `render` (the pinned scorer's build_prompt; the
            state carries the processor's image-marker token per image); the marker run is EXPANDED to [boi, soft*K, eoi],
            n_tokens is the EXPANDED count, each row records images: [{sha256, s3, width, height, n_expanded, span [start, K]
            in EXPANDED ids}] (boi / eoi stay ordinary tokens), and the header pins image_layout {boi_id, soft_id, eoi_id,
            marker_id} + processor/tokenizer file sha256s (verified against the tokenizer's own ids).
            --panel NAME=FILE.jsonl ... --n N --out DEVDIR    (v2) the LR-probe DEV data dir: N pool VAL units (B's pool mix; clean
            licence, K <= 52, tokens <= --max-tokens, no STRONG eval-index hit, QASPER NC papers out, (v3) not a record of any panel)
            + whole SCREEN4 panels. (v4) --panel NAME=DIR: every *.jsonl in DIR is a block (panel "panel:NAME:<file stem>");
            --panel-sample NAME=N: N units of that panel, allocated over its files in proportion (largest remainder), drawn per file
            with numpy default_rng([seed, sha256(name|file)]) from the units in file order; records keep only the drawn questions

RENDERER = the pinned production scorer's build_prompt (claude_lmscore.py e0672bd0, loaded by path, md5-checked) with the pinned
Gemma-4 tokenizer's chat template, max_len (default 32768: the pilot's longest stratum), no whole-case listing, plain chat prompt,
--noul-defs as given. renderer_sha256 = sha256 of the canonical JSON of {scorer md5, cache_features md5, tokenizer.json /
tokenizer_config.json / chat_template sha256s, max_len, margin, noul_defs, whole_case, plain, labels}. A state that would have to be
CUT to fit max_len is refused (no silent truncation; --allow-truncation records it instead).

OPTION ORDER: the k-th exposure of a decision shows its options in the order drawn by numpy default_rng(sha256(render_seed | uid | k));
choice questions with K >= 2 are permuted (identity allowed, as claude_lmtune's make_example), noul / score keep the stored order. The
same decision in two recipe manifests therefore gets the SAME prompt for its k-th exposure, so a parent cache is shared.

ROUTING (mixture_spec_v1.json parent_preservation; never by hard / soft label_kind): an explicit row `parent_route` (stock | garas |
none) wins; else the target basis (exact numeric / posterior / action -> none, observed / realised outcome -> garas); else the mixture
family: retrieval_ranking_classification, grounding_answerability_language, knowledge_preservation, comparative_outcome_cost -> stock
Gemma; arts_human_disagreement, tools_policy -> Garas d99; exact_bayesian_numeric -> none. Unknown family without a route: refused.
route none = numeric_family true: such rows never get a parent (preservation.py refuses them) and keep their oracle gold.

PARENT REVISION: sha256 of the canonical JSON of the parent's full spec (base repo + commit, adapter md5 if any, scorer md5, readout,
attention kernel, dtype) from PARENTS.json, so it is immutable, known before scoring, and re-verified by fullft_parent.py at load.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import Counter

import numpy as np

VERSION = "fullft_render v5 (2026-11-06)"
SCHEMA = "fullft-render-manifest-v1"
SCHEMA_V2 = "fullft-render-manifest-v2"          # (vision v6) + images / image_layout / EXPANDED n_tokens (render-vision)
PINNED_SCORER_MD5 = "e0672bd0f515ccb17d97035c84c3d339"
PINNED_CF_MD5 = "49613a40faed8cd612175e81c3f3925f"
FAMILY_ROUTE = {"retrieval_ranking_classification": "stock", "grounding_answerability_language": "stock",
                "knowledge_preservation": "stock", "comparative_outcome_cost": "stock", "arts_human_disagreement": "garas",
                "tools_policy": "garas", "exact_bayesian_numeric": "none"}
BASIS_NONE = ("exact_numeric", "exact_posterior", "exact_bayes", "loss_table_action", "exact_action")
BASIS_GARAS = ("observed_outcome", "realised_outcome", "realized_outcome")
ROUTES = ("stock", "garas", "none")


def md5_file(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canon(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_scorer(code_dir):
    """the pinned claude_lmscore.py (+ its cache_features.py) from code_dir (or code_dir/trainers), md5-checked."""
    d = code_dir if os.path.exists(os.path.join(code_dir, "claude_lmscore.py")) else os.path.join(code_dir, "trainers")
    p, c = os.path.join(d, "claude_lmscore.py"), os.path.join(d, "cache_features.py")
    for f, want in ((p, PINNED_SCORER_MD5), (c, PINNED_CF_MD5)):
        got = md5_file(f)
        if got != want and not os.environ.get("JEV_ALLOW_UNPINNED_SCORER"):
            raise SystemExit(f"fullft_render: {f} md5 {got} != pin {want}")
    spec = importlib.util.spec_from_file_location("ls_pinned_fullft", p)
    ls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ls)
    return ls, {"scorer_md5": md5_file(p), "cache_features_md5": md5_file(c)}


def tokenizer_files(tok_dir):
    out = {}
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"):
        p = os.path.join(tok_dir, f)
        if os.path.exists(p):
            out[f] = sha256_file(p)
    if "tokenizer.json" not in out:
        raise SystemExit(f"{tok_dir}: no tokenizer.json")
    return out


def renderer_spec(code_md5s, tok_dir, max_len, noul_defs, tok_revision=""):
    spec = {"schema": SCHEMA, "builder": "claude_lmscore.build_prompt", **code_md5s, "tokenizer_files": tokenizer_files(tok_dir),
            "tokenizer_revision": tok_revision, "max_len": int(max_len), "margin": 16, "noul_defs": bool(noul_defs),
            "whole_case": False, "plain": False, "chat_preset": None, "labels": "A-Z a-z", "tokenize": "add_special_tokens=False"}
    return spec, hashlib.sha256(canon(spec).encode()).hexdigest()


def parent_specs(path):
    """PARENTS.json: {"stock": {...}, "garas": {...}, "scorer": {...}} -> {route: (parent_id, revision sha256, full spec)}"""
    js = json.load(open(path)) if isinstance(path, str) else path
    sc = js.get("scorer") or {}
    out = {}
    for r in ("stock", "garas"):
        if r not in js:
            continue
        s = {**js[r], "scorer": sc}
        if not s.get("parent_id") or not s.get("base_revision"):
            raise SystemExit(f"parents: {r} needs parent_id and base_revision")
        if len(s["base_revision"]) != 40:
            raise SystemExit(f"parents: {r} base_revision must be a 40-hex commit")
        out[r] = (s["parent_id"], hashlib.sha256(canon(s).encode()).hexdigest(), s)
    return out


def route_of(row):
    r = row.get("parent_route")
    if r is not None:
        if r not in ROUTES:
            raise ValueError(f"parent_route {r!r} not in {ROUTES}")
        return r
    kind = str(row.get("target_basis_kind") or "")
    if any(kind.startswith(b) for b in BASIS_NONE):
        return "none"
    if any(kind.startswith(b) for b in BASIS_GARAS):
        return "garas"
    fam = row.get("mixture_family")
    if fam not in FAMILY_ROUTE:
        raise ValueError(f"no parent route for family {fam!r} (basis {kind!r}); set parent_route explicitly")
    return FAMILY_ROUTE[fam]


def option_order(render_seed, uid, k_exp, qtype, k):
    order = list(range(k))
    if qtype == "choice" and k >= 2:
        s = int(hashlib.sha256(f"{render_seed}|{uid}|{k_exp}".encode()).hexdigest()[:16], 16)
        order = [int(x) for x in np.random.default_rng(s).permutation(k)]
    return order


def render(a):
    ls, md5s = load_scorer(a.code)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    spec, ren = renderer_spec(md5s, a.tokenizer, a.max_len, a.noul_defs, a.tokenizer_revision)
    parents = parent_specs(a.parents)
    splits = [s for s in a.splits.split(",") if s]
    units = list(ls.cf.iter_units(a.data, splits, set(), None))
    uids = ls.unit_uids(units)
    by_uid = dict(zip(uids, units))
    if len(by_uid) != len(units):
        raise SystemExit("duplicate unit uids")
    files = sorted(os.path.join(a.data, s, f) for s in splits for f in os.listdir(os.path.join(a.data, s)) if f.endswith(".jsonl"))
    data_fp = {os.path.relpath(f, a.data): sha256_file(f) for f in files}
    if a.exposures:
        exps = [json.loads(l) for l in open(a.exposures) if l.strip()]
    else:
        exps = [{"decision_uid": u, "parent_route": a.route_all} if a.route_all else {"decision_uid": u} for u in uids]
    seen, weights, rows = Counter(), {}, []
    stats = Counter()
    tmp = a.out + ".partial"
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists: immutable output, choose a new path")
    with open(tmp, "w", encoding="utf-8") as f:
        for n, e in enumerate(exps):
            uid = e["decision_uid"]
            u = by_uid.get(uid)
            if u is None:
                raise SystemExit(f"exposure {n}: decision {uid} not in {a.data} / {splits}")
            k = len(u["option_keys"])
            if k > 52:
                raise SystemExit(f"exposure {n}: {uid} has K = {k} > 52 (the shared reader caps at 52)")
            kx = seen[uid]
            seen[uid] += 1
            order = option_order(a.render_seed, uid, kx, u["qtype"], k)
            text = ls.build_prompt(tok, u, order, a.max_len, noul_defs=a.noul_defs)
            if text is None:
                raise SystemExit(f"exposure {n}: {uid} does not fit max_len {a.max_len}")
            truncated = was_cut(ls, tok, u, order, text, a.max_len, a.noul_defs)
            if truncated and not a.allow_truncation:
                raise SystemExit(f"exposure {n}: {uid}: the state would be CUT to fit {a.max_len} tokens (refused; no silent truncation)")
            ntok = len(tok(text, add_special_tokens=False)["input_ids"])
            tv = np.asarray(u["target"], dtype=np.float64)
            if not np.isfinite(tv).all() or (tv < 0).any() or tv.sum() <= 0:
                raise SystemExit(f"exposure {n}: {uid}: invalid target")
            tv = tv / tv.sum()
            keys = [str(x) for x in u["option_keys"]]
            if len(set(keys)) != len(keys):
                raise SystemExit(f"exposure {n}: {uid}: duplicate option keys")
            route = route_of(e)
            w = float(e.get("weight", 1.0))
            if weights.setdefault(uid, w) != w:
                raise SystemExit(f"decision {uid}: two weights")
            row = {"exposure_id": f"{a.tag}:{n}", "exposure_k": kx, "decision_uid": uid, "renderer_sha256": ren,
                   "prompt_sha256": sha256_text(text), "option_order": [keys[i] for i in order], "option_keys": keys,
                   "gold": {keys[i]: float(tv[i]) for i in range(k)}, "numeric_family": route == "none", "parent_route": route,
                   "weight": w, "n_tokens": ntok, "qtype": u["qtype"], "label_kind": u["label_kind"], "source": u["source"],
                   "license": u["license"], "base_id": str(u["base_id"]), "record_id": str(u["record_id"]), "qid": str(u["qid"]),
                   "mixture_family": e.get("mixture_family"), "target_basis_kind": e.get("target_basis_kind"),
                   "group_id": e.get("group_id"), "truncated": truncated, "prompt": text}
            if route != "none":
                if route not in parents:
                    raise SystemExit(f"exposure {n}: route {route} but PARENTS.json has no {route} parent")
                row["parent_id"], row["parent_revision"] = parents[route][0], parents[route][1]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats[f"route_{route}"] += 1
            stats[f"family_{e.get('mixture_family')}"] += 1
            stats["tokens"] += ntok
            stats["truncated"] += int(truncated)
            for lo, hi in ((1, 512), (513, 2048), (2049, 8192), (8193, 32768)):
                if lo <= ntok <= hi:
                    stats[f"len_{lo}_{hi}"] += 1
    os.replace(tmp, a.out)
    header = {"manifest_header": True, "schema": SCHEMA, "version": VERSION, "tag": a.tag, "renderer_sha256": ren, "renderer_spec": spec,
              "render_seed": a.render_seed, "data_dir_fingerprint": data_fp, "splits": splits, "exposures_file_sha256":
              sha256_file(a.exposures) if a.exposures else None, "n_exposures": len(exps), "n_unique_decisions": len(weights),
              "total_decision_weight": float(sum(weights.values())), "parents": {r: {"parent_id": p[0], "parent_revision": p[1],
              "spec": p[2]} for r, p in parents.items()}, "stats": dict(stats), "rows_sha256": sha256_file(a.out)}
    with open(a.out + ".header.json", "w") as f:
        json.dump(header, f, indent=1, sort_keys=True)
    return header


def eligible(a):
    n = 0
    with open(a.rows, encoding="utf-8") as f, open(a.out, "x", encoding="utf-8") as g:
        for line in f:
            r = json.loads(line)
            if r["numeric_family"] or (a.route and r["parent_route"] != a.route):
                continue
            g.write(line if line.endswith("\n") else line + "\n")
            n += 1
    return {"eligible": n}


def assemble(a):
    """TRAIN = ROWS with every eligible row replaced by its joined counterpart (exposure order, 1:1)."""
    hdr = json.load(open(a.rows + ".header.json"))
    if hdr["rows_sha256"] != sha256_file(a.rows):
        raise SystemExit("rows file differs from its header's sha256")
    jf = open(a.joined, encoding="utf-8")
    n = nj = 0
    with open(a.rows, encoding="utf-8") as f, open(a.out + ".partial", "x", encoding="utf-8") as g:
        for line in f:
            r = json.loads(line)
            if not r["numeric_family"]:
                jl = jf.readline()
                if not jl:
                    raise SystemExit(f"joined file ended before eligible row {r['exposure_id']}")
                j = json.loads(jl)
                if j["exposure_id"] != r["exposure_id"] or j["prompt_sha256"] != r["prompt_sha256"] or j["gold"] != r["gold"]:
                    raise SystemExit(f"joined row {j.get('exposure_id')} does not match {r['exposure_id']} (order / prompt / gold)")
                pp = j.get("parent_preservation") or {}
                if pp.get("gold_overwritten") is not False or pp.get("parent_revision") != r["parent_revision"]:
                    raise SystemExit(f"{r['exposure_id']}: joined parent entry invalid")
                r = j
                nj += 1
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    if jf.readline():
        raise SystemExit("joined file has rows beyond the eligible rows")
    os.replace(a.out + ".partial", a.out)
    hdr2 = {**hdr, "assembled_from": {"rows_sha256": hdr["rows_sha256"], "joined_sha256": sha256_file(a.joined)},
            "rows_sha256": sha256_file(a.out), "n_with_parent": nj}
    with open(a.out + ".header.json", "w") as f:
        json.dump(hdr2, f, indent=1, sort_keys=True)
    return {"rows": n, "with_parent": nj}


FREEZE_ROUTE = {"stock_gemma": "stock", "garas_d99": "garas", "none": "none"}


def was_cut(ls, tok, u, order, text, max_len, noul_defs):
    """(v3) did build_prompt CUT the state to fit max_len? v1 / v2 tested `state in text`, which is wrong for a state with leading
    whitespace: PROMPT starts with {state} and the chat template trims the message, so such a state is never found verbatim (two
    real cases: contractnli nda-18, an EG val record). Exact rule: the prompt differs from the same build_prompt with no length
    limit; the cheap test first (a state found verbatim was not cut)."""
    st = u["state"]
    if isinstance(st, str) and st in text:
        return False
    return text != ls.build_prompt(tok, u, order, 1 << 40, noul_defs=noul_defs)


def open_jsonl(path):
    import gzip
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8")


def kv_list(items):
    out = {}
    for x in items or []:
        k, v = x.split("=", 1)
        if k in out:
            raise SystemExit(f"{k} given twice")
        out[k] = v
    return out


def pool_units(ls, pools, wanted):
    """pools {name: root dir with <split>/*.jsonl}; wanted: freeze uids "<pool>|<split>/<file>|<line 1-based>|<qid>" -> {uid: unit}.
    The units are the PINNED loader's own (cf.iter_units, one file at a time); the uid labels come from the same file / line /
    question enumeration in lockstep (record id and qid re-checked on every unit)."""
    from pathlib import Path
    need = {}
    for u in wanted:
        p, mem, _, _ = u.split("|")
        need.setdefault(p, set()).add(mem)
    out = {}
    for p, mems in sorted(need.items()):
        if p not in pools:
            raise SystemExit(f"pool {p} is needed but not given (--pools {p}=DIR)")
        for mem in sorted(mems):
            split, name = mem.split("/", 1)
            path = Path(pools[p], split, name)
            if "/" in name or not path.exists():
                raise SystemExit(f"{p}: member {mem} not found under {pools[p]}")
            labels = []
            with open(path, encoding="utf-8") as f:
                for ln, line in enumerate(f, 1):
                    rec = json.loads(line)
                    for qid in rec["questions"]:
                        labels.append((ln, str(qid), rec["id"]))
            units = list(ls.cf.iter_units(pools[p], [split], {path.stem}, None))
            if len(units) != len(labels):
                raise SystemExit(f"{p}/{mem}: the loader yields {len(units)} units, the file has {len(labels)} questions")
            for (ln, qid, rid), u in zip(labels, units):
                if str(u["qid"]) != qid or u["record_id"] != rid:
                    raise SystemExit(f"{p}/{mem}:{ln}: loader / enumeration out of step")
                key = f"{p}|{mem}|{ln}|{qid}"
                if key in wanted:
                    out[key] = u
    missing = set(wanted) - set(out)
    if missing:
        raise SystemExit(f"{len(missing)} wanted units not found, e.g. {sorted(missing)[:3]}")
    return out


def render_freeze(a):
    """(v2) a freeze_r2 recipe -> the render manifest (same row schema as render; exposure order = the frozen `pos` order)."""
    ls, md5s = load_scorer(a.code)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    spec, ren = renderer_spec(md5s, a.tokenizer, a.max_len, a.noul_defs, a.tokenizer_revision)
    parents = parent_specs(a.parents)
    pools = kv_list(a.pools)
    with open_jsonl(a.exposures) as f:
        exps = [json.loads(l) for l in f if l.strip()]
    exps = [e for e in exps if not e.get("manifest_header")]
    if [e["pos"] for e in exps] != list(range(len(exps))):
        raise SystemExit("exposures are not in pos order 0..n-1")
    units = pool_units(ls, pools, {e["uid"] for e in exps})
    seen, weights, stats, dtok = Counter(), {}, Counter(), []
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists: immutable output, choose a new path")
    tmp = a.out + ".partial"
    with open(tmp, "w", encoding="utf-8") as f:
        for n, e in enumerate(exps):
            uid, u = e["uid"], units[e["uid"]]
            k = len(u["option_keys"])
            order = [int(x) for x in e["option_order"]]
            if sorted(order) != list(range(k)) or int(e.get("K", k)) != k:
                raise SystemExit(f"exposure {n}: {uid}: frozen option order {order} is not a permutation of its {k} options")
            if k > 52:
                raise SystemExit(f"exposure {n}: {uid} has K = {k} > 52")
            if "replay_ix" not in e:
                raise SystemExit(f"exposure {n}: {uid}: no replay_ix")
            kx = int(e["replay_ix"])
            if kx < 0 or (uid, kx) in seen:
                raise SystemExit(f"exposure {n}: {uid}#{kx}: (uid, replay_ix) twice or negative")
            seen[(uid, kx)] += 1
            stats[f"replay_ix_{kx}"] += 1
            text = ls.build_prompt(tok, u, order, a.max_len, noul_defs=a.noul_defs)
            if text is None:
                raise SystemExit(f"exposure {n}: {uid} does not fit max_len {a.max_len}")
            truncated = was_cut(ls, tok, u, order, text, a.max_len, a.noul_defs)
            if truncated and not a.allow_truncation:
                raise SystemExit(f"exposure {n}: {uid}: the state would be CUT to fit {a.max_len} tokens (refused)")
            ntok = len(tok(text, add_special_tokens=False)["input_ids"])
            if "tokens_r1" in e and not a.noul_defs:
                dtok.append(ntok - int(e["tokens_r1"]))
            tv = np.asarray(u["target"], dtype=np.float64)
            if not np.isfinite(tv).all() or (tv < 0).any() or tv.sum() <= 0:
                raise SystemExit(f"exposure {n}: {uid}: invalid target")
            tv = tv / tv.sum()
            keys = [str(x) for x in u["option_keys"]]
            if len(set(keys)) != len(keys):
                raise SystemExit(f"exposure {n}: {uid}: duplicate option keys")
            route = FREEZE_ROUTE.get(e.get("parent_route"))
            if route is None:
                raise SystemExit(f"exposure {n}: parent_route {e.get('parent_route')!r} not in {sorted(FREEZE_ROUTE)}")
            w = float(e.get("weight", 1.0))
            if weights.setdefault(uid, w) != w:
                raise SystemExit(f"decision {uid}: two weights")
            row = {"exposure_id": f"{a.tag}:{n}", "exposure_k": kx, "decision_uid": uid, "renderer_sha256": ren,
                   "prompt_sha256": sha256_text(text), "option_order": [keys[i] for i in order], "option_keys": keys,
                   "gold": {keys[i]: float(tv[i]) for i in range(k)}, "numeric_family": route == "none", "parent_route": route,
                   "weight": w, "n_tokens": ntok, "qtype": u["qtype"], "label_kind": u["label_kind"], "source": u["source"],
                   "license": u["license"], "base_id": str(u["base_id"]), "record_id": str(u["record_id"]), "qid": str(u["qid"]),
                   "mixture_family": e.get("area"), "block": e.get("block"), "root": e.get("root"), "group_id": e.get("case"),
                   "pool": e.get("pool"), "pos": e.get("pos"), "di_ancestry": e.get("di_ancestry"), "truncated": truncated, "prompt": text}
            if route != "none":
                if route not in parents:
                    raise SystemExit(f"exposure {n}: route {route} but PARENTS.json has no {route} parent")
                row["parent_id"], row["parent_revision"] = parents[route][0], parents[route][1]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats[f"route_{route}"] += 1
            stats[f"block_{e.get('block')}"] += 1
            stats["tokens"] += ntok
            stats["truncated"] += int(truncated)
            for lo, hi in ((1, 512), (513, 2048), (2049, 8192), (8193, 32768)):
                if lo <= ntok <= hi:
                    stats[f"len_{lo}_{hi}"] += 1
    d = np.asarray(dtok) if dtok else np.zeros(0)
    tokchk = {"compared": int(d.size), "differ": int((d != 0).sum()), "max_abs": int(np.abs(d).max()) if d.size else 0}
    if d.size and (tokchk["max_abs"] > 16 or tokchk["differ"] > 0.005 * d.size):
        os.remove(tmp)
        raise SystemExit(f"tokens differ from the freeze's tokens_r1 beyond tolerance: {tokchk}")
    os.replace(tmp, a.out)
    uids_seen = {uid for uid, _ in seen}
    only_replay = sum(1 for uid in uids_seen if (uid, 0) not in seen)
    header = {"manifest_header": True, "schema": SCHEMA, "version": VERSION, "tag": a.tag, "renderer_sha256": ren, "renderer_spec": spec,
              "freeze": {"exposures_file_sha256": sha256_file(a.exposures), "exposures_file_md5": md5_file(a.exposures),
                         "pools": sorted(pools), "option_order": "frozen per exposure", "tokens_r1_check": tokchk,
                         "exposure_k": "the freeze's replay_ix (identity uid#replay_ix, not the pos-order occurrence count)",
                         "decisions_with_replay_ix_1_only": only_replay},
              "n_exposures": len(exps), "n_unique_decisions": len(weights), "total_decision_weight": float(sum(weights.values())),
              "parents": {r: {"parent_id": p[0], "parent_revision": p[1], "spec": p[2]} for r, p in parents.items()},
              "stats": dict(stats), "rows_sha256": sha256_file(a.out)}
    with open(a.out + ".header.json", "w") as f:
        json.dump(header, f, indent=1, sort_keys=True)
    return header


def devset(a):
    """(v2) DEV data dir for the LR probe: DEVDIR/test/<pool>__<stem>.jsonl holding the selected VAL units (each record reduced to
    its selected question) + DEVDIR/test/panel_<name>.jsonl (whole panels) + DEVDIR/panels.json (record id, qid -> panel). Pool
    units: split val, nc false, 2 <= K <= 52, tok_r1 <= max_tokens, no STRONG hit (raw scan; stricter than the adjudicated train
    rule), base id not in the QASPER NC exclusions; n units allocated to pools in proportion to the --mix recipe's exposures
    (largest remainder), drawn with numpy default_rng(seed) per pool from the sorted eligible uids."""
    from pathlib import Path
    pools, tables, hits, panels = kv_list(a.pools), kv_list(a.tables), kv_list(a.hits), kv_list(a.panel)
    excl = {k: set(json.load(open(v))["excluded_base_ids"]) for k, v in kv_list(a.exclude_base_ids).items()}
    with open_jsonl(a.mix) as f:
        mix = Counter(json.loads(l)["pool"] for l in f if l.strip())
    tot = sum(mix.values())
    quota = {p: a.n * c / tot for p, c in mix.items()}
    alloc = {p: int(math.floor(q)) for p, q in quota.items()}
    for p in sorted(quota, key=lambda p: (-(quota[p] - alloc[p]), p))[: a.n - sum(alloc.values())]:
        alloc[p] += 1
    out_dir = Path(a.out)
    if out_dir.exists():
        raise SystemExit(f"{a.out} exists")
    (out_dir / "test").mkdir(parents=True)
    report, pmap = {"alloc": alloc, "eligible": {}, "panels": {}, "excluded_as_panel_records": {}}, {}
    panel_ids = set()                                  # (v3) a pool val record that is also a panel record is the panel's
    for name, path in sorted(panels.items()):
        for fp in (sorted(Path(path).glob("*.jsonl")) if Path(path).is_dir() else [Path(path)]):
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        panel_ids.add(str(json.loads(line)["id"]))
    for p, want in sorted(alloc.items()):
        if want == 0:
            continue
        strong = set()
        with open_jsonl(hits[p]) as f:
            for l in f:
                h = json.loads(l)
                if h.get("s"):
                    strong.add(h["uid"])
        elig = []
        with open_jsonl(tables[p]) as f:
            for l in f:
                t = json.loads(l)
                if t["split"] != "val" or t.get("nc") or not (2 <= int(t["K"]) <= 52) or int(t["tok_r1"]) > a.max_tokens \
                        or t["uid"] in strong:
                    continue
                elig.append(t["uid"])
        if p in excl:                                  # QASPER NC papers: by the loader's base id rule, read from the val records
            bad = set()
            for mem in sorted({u.split("|")[1] for u in elig}):
                with open(Path(pools[p], mem), encoding="utf-8") as f:
                    for ln, line in enumerate(f, 1):
                        rec = json.loads(line)
                        if str(rec.get("base_id") or rec.get("meta", {}).get("variant_of") or rec["id"]) in excl[p]:
                            bad.update(f"{p}|{mem}|{ln}|{q}" for q in rec["questions"])
            elig = [u for u in elig if u not in bad]
        if panel_ids:
            bad, n0 = set(), len(elig)
            for mem in sorted({u.split("|")[1] for u in elig}):
                with open(Path(pools[p], mem), encoding="utf-8") as f:
                    for ln, line in enumerate(f, 1):
                        rec = json.loads(line)
                        if str(rec["id"]) in panel_ids:
                            bad.update(f"{p}|{mem}|{ln}|{q}" for q in rec["questions"])
            elig = [u for u in elig if u not in bad]
            report["excluded_as_panel_records"][p] = n0 - len(elig)
        elig.sort()
        report["eligible"][p] = len(elig)
        if len(elig) < want:
            raise SystemExit(f"{p}: {len(elig)} eligible val units < {want} wanted")
        rng = np.random.default_rng([a.seed, int(hashlib.sha256(p.encode()).hexdigest()[:8], 16)])
        by_mem = {}
        for i in sorted(rng.permutation(len(elig))[:want].tolist()):
            uid = elig[i]
            _, mem, ln, qid = uid.split("|")
            by_mem.setdefault(mem, []).append((int(ln), qid, uid))
        n_written = 0
        for mem, items in sorted(by_mem.items()):
            lines = {}
            for ln, qid, uid in items:
                lines.setdefault(ln, []).append(qid)
            src = Path(pools[p], mem)
            dst = out_dir / "test" / f"{p}__{Path(mem).stem}.jsonl"
            kept = []
            with open(src, encoding="utf-8") as f:
                for ln, line in enumerate(f, 1):
                    if ln not in lines:
                        continue
                    rec = json.loads(line)
                    qs = [q for q in rec["questions"] if str(q) in lines[ln]]
                    rec["questions"] = {q: rec["questions"][q] for q in qs}
                    rec["labels"] = {q: rec["labels"][q] for q in qs}
                    kept.append(rec)
                    for q in qs:
                        key = json.dumps([str(rec["id"]), str(q)], ensure_ascii=False)
                        if key in pmap:
                            raise SystemExit(f"devset: unit {key} twice (record ids must be unique across the DEV dir)")
                        pmap[key] = f"pilotval:{p}"
                        n_written += 1
            with open(dst, "w", encoding="utf-8") as g:
                for rec in kept:
                    g.write(json.dumps(rec, ensure_ascii=False) + "\n")
        report.setdefault("written", {})[p] = n_written
    samples = {k: int(v) for k, v in kv_list(a.panel_sample).items()}
    for name, path in sorted(panels.items()):
        files = sorted(Path(path).glob("*.jsonl")) if Path(path).is_dir() else [Path(path)]
        if not files:
            raise SystemExit(f"panel {name}: no .jsonl under {path}")
        units = {}
        for fp in files:
            lst = []
            with open(fp, encoding="utf-8") as f:
                for ln, line in enumerate(f, 1):
                    if line.strip():
                        rec = json.loads(line)
                        lst.extend((ln, str(q)) for q in rec["questions"])
            units[fp] = lst
        keep = {fp: {u for u in lst} for fp, lst in units.items()}
        if name in samples:
            tot_u = sum(len(v) for v in units.values())
            want = samples[name]
            if want > tot_u:
                raise SystemExit(f"panel {name}: {want} wanted > {tot_u} units")
            quota = {fp: want * len(v) / tot_u for fp, v in units.items()}
            alloc_p = {fp: int(math.floor(q)) for fp, q in quota.items()}
            for fp in sorted(quota, key=lambda fp: (-(quota[fp] - alloc_p[fp]), str(fp.name)))[: want - sum(alloc_p.values())]:
                alloc_p[fp] += 1
            for fp, lst in units.items():
                rng = np.random.default_rng([a.seed, int(hashlib.sha256(f"{name}|{fp.name}".encode()).hexdigest()[:8], 16)])
                keep[fp] = {lst[i] for i in rng.permutation(len(lst))[: alloc_p[fp]].tolist()}
        n, blocks = 0, {}
        for fp in files:
            block = fp.stem if len(files) > 1 else None
            dst = out_dir / "test" / (f"panel_{name}__{fp.stem}.jsonl" if block else f"panel_{name}.jsonl")
            with open(fp, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
                for ln, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    qs = [q for q in rec["questions"] if (ln, str(q)) in keep[fp]]
                    if not qs:
                        continue
                    rec["questions"] = {q: rec["questions"][q] for q in qs}
                    rec["labels"] = {q: rec["labels"][q] for q in qs}
                    g.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    for q in qs:
                        key = json.dumps([str(rec["id"]), str(q)], ensure_ascii=False)
                        if key in pmap:
                            raise SystemExit(f"devset: unit {key} twice (record ids must be unique across the DEV dir)")
                        pmap[key] = f"panel:{name}:{block}" if block else f"panel:{name}"
                        n += 1
                        if block:
                            blocks[block] = blocks.get(block, 0) + 1
        report["panels"][name] = {"units": n, "sha256": sha256_file(files[0]) if len(files) == 1 else
                                  hashlib.sha256("".join(sha256_file(fp) for fp in files).encode()).hexdigest(),
                                  **({"blocks": blocks} if blocks else {}), **({"sampled": samples[name]} if name in samples else {})}
    json.dump({"panel_of": pmap, "report": report, "seed": a.seed, "n": a.n, "max_tokens": a.max_tokens,
               "mix_md5": md5_file(a.mix)}, open(out_dir / "panels.json", "w"), indent=1, sort_keys=True)
    return report


def image_layout(tok, processor):
    """the Gemma-4 image token ids, verified against BOTH the pinned processor's token strings and the tokenizer itself:
    marker (processor.image_token, e.g. 258880), boi (`<|image>`), eoi (`<image|>`). All must be single tokens; the trainer
    (fullft_train Tokens._expand) trusts only these pinned ids from the manifest header."""
    out = {}
    for k, s in (("marker_id", processor.image_token), ("boi_id", processor.boi_token), ("eoi_id", processor.eoi_token)):
        ids = tok(s, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise SystemExit(f"image layout: {s!r} is not a single token ({ids})")
        out[k] = ids[0]
    out["soft_id"] = out["marker_id"]                          # the SAME special id expands into the [boi, soft*K, eoi] run
    return out


def processor_files(pdir):
    out = {f: sha256_file(os.path.join(pdir, f)) for f in sorted(os.listdir(pdir))
           if f.endswith(".json") and (f.startswith("processor") or f.startswith("preprocessor"))}
    if not out:
        raise SystemExit(f"{pdir}: no processor / preprocessor json files to pin")
    return out


def expand_count(proc_ids, mi, layout):
    """K = the processor's soft-token run of one image: the ids strictly between boi and the following eoi (the processor
    replaced the marker at text-token position `mi` with [boi, soft*K, eoi]; boi stands at `mi` in the processed ids)."""
    if mi >= len(proc_ids) or proc_ids[mi] != layout["boi_id"]:
        mi = proc_ids.index(layout["boi_id"], max(0, mi - 2))
    e = proc_ids.index(layout["eoi_id"], mi + 1)
    return e - mi - 1


def render_vision(a):
    """(v5, vision round 1) the v2 IMAGE-conditioned render manifest (module docstring). Pool records carry a top-level
    `image` object ({path (mode A) or pixels+n_expanded (mode B: --pixels-from MAP), s3, width, height, sha256}) that
    iter_units does not forward, so the rows are read here directly; the question / label handling is the pinned loader's
    own question_options. State = plain text with exactly ONE image-marker token (round 1 = single image)."""
    ls, md5s = load_scorer(a.code)
    from transformers import AutoTokenizer, AutoProcessor
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    proc = AutoProcessor.from_pretrained(a.processor)
    layout = image_layout(tok, proc)
    staged = json.load(open(a.pixels_from)) if a.pixels_from else {}
    if a.pixel_dir:
        os.makedirs(a.pixel_dir, exist_ok=True)
    exps = []
    with open_jsonl(a.pool) as f:
        for l in f:
            if not l.strip():
                continue
            r = json.loads(l)
            if not r.get("manifest_header"):
                exps.append(r)
    if not exps:
        raise SystemExit(f"{a.pool}: no records")
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists: immutable output, choose a new path")
    tmp, stats, weights = a.out + ".partial", {"rows": 0, "images": 0, "tokens": 0, "expansions": 0}, {}
    with open(tmp, "w", encoding="utf-8") as f:
        for n, rec in enumerate(exps):
            img = rec.get("image")            # (C7) a record with NO image = a text-only replay row: plain v1-style row, no images field
            qids = list(rec["questions"])
            if len(qids) != 1:
                raise SystemExit(f"record {n}: vision round 1 pools record ONE question per image row (got {len(qids)})")
            qid = qids[0]
            q = rec["questions"][qid]
            keys, texts, target = ls.cf.question_options(q, rec["labels"][qid])
            if not (2 <= len(keys) <= 52):
                raise SystemExit(f"record {n}: K outside 2..52")
            u = {"split": "train", "source": rec.get("source") or "vision", "license": str(rec.get("license") or ""),
                 "record_id": str(rec["id"]), "base_id": str(rec.get("base_id") or rec["id"]), "qid": str(qid),
                 "qtype": q.get("type") or "choice", "state": rec["state"], "instructions": q["instructions"],
                 "option_keys": keys, "option_texts": texts,
                 "noul_criteria": q.get("criteria") if q.get("type") == "noul" else None,
                 "target": target, "label_kind": (rec.get("label_kind") or {}).get(qid) or "hard"}
            uid = f"{a.tag}|{rec['id']}|{qid}"
            order = option_order(a.render_seed, uid, 0, u["qtype"], len(keys))
            text_ids0 = tok(u["state"], add_special_tokens=False)["input_ids"]
            if img is not None and text_ids0.count(layout["marker_id"]) != 1:
                raise SystemExit(f"record {n} ({rec['id']}): the state must carry exactly one image-marker token")
            sha, wpx, hpx, K, px_src, span, ntok = None, None, None, None, None, None, 0
            sha = img.get("sha256") if img else None
            if img:
                if not sha:
                    from PIL import Image
                    with Image.open(img["path"]) as im:
                        sha, wpx, hpx = sha256_file(img["path"]), im.size[0], im.size[1]
                else:
                    wpx, hpx = img.get("width"), img.get("height")
            px_src = (staged.get(sha) or img.get("pixels")) if img else None
            K = None
            if img and px_src is None and img.get("path"):
                import torch as _t
                from PIL import Image as _Im
                with _Im.open(img["path"]) as im:
                    pout = proc(text=u["state"], images=[im], return_tensors="pt", add_special_tokens=False)
                    K = expand_count(pout["input_ids"][0].tolist(), text_ids0.index(layout["marker_id"]), layout)
                    px = {"pixel_values": pout["pixel_values"], "image_position_ids": pout["image_position_ids"]}
                if a.pixel_dir:
                    dst = os.path.join(a.pixel_dir, sha + ".pt")
                    _t.save(px, dst)
                    px_src = dst
                else:
                    px_src = ""
            elif img and px_src:
                if K is None:
                    K = int(img["n_expanded"])
                import torch as _t2
                pxchk = _t2.load(px_src, weights_only=False)
                if isinstance(pxchk, dict):
                    if pxchk["pixel_values"].dim() != 3:
                        raise SystemExit(f"record {n}: staged pixels {px_src} is not [1, R, D]")
                elif pxchk.dim() != 3:
                    raise SystemExit(f"record {n}: staged pixels {px_src} is not [1, R, D]")
            elif img:
                raise SystemExit(f"record {n}: no pixels (mode A needs image.path, mode B needs a staged sha)")
            text = ls.build_prompt(tok, u, order, a.max_len, noul_defs=a.noul_defs)
            if text is None:
                raise SystemExit(f"record {n}: {uid} does not fit max_len {a.max_len}")
            ptids = tok(text, add_special_tokens=False)["input_ids"]
            if img:
                mi = ptids.index(layout["marker_id"])               # EXPANDED coordinates; the soft run after boi
                span = (mi + 1, int(K))
                ntok = len(ptids) - 1 + int(K) + 2                  # minus the marker, plus [boi, K soft, eoi]
            else:
                ntok = len(ptids)                                   # text-only row: the plain v1 count
            tv = np.asarray(target, dtype=np.float64)
            if not np.isfinite(tv).all() or (tv < 0).any() or tv.sum() <= 0:
                raise SystemExit(f"record {n}: invalid target")
            tv = tv / tv.sum()
            w = float(rec.get("weight", 1.0))
            if weights.setdefault(uid, w) != w:
                raise SystemExit(f"decision {uid}: two weights")
            row = {"exposure_id": f"{a.tag}:{n}", "exposure_k": 0, "decision_uid": uid, "renderer_sha256": None,
                   "prompt_sha256": sha256_text(text), "option_order": [keys[i] for i in order], "option_keys": keys,
                   "gold": {keys[i]: float(tv[i]) for i in range(len(keys))}, "numeric_family": True, "parent_route": "none",
                   "weight": w, "n_tokens": ntok, "qtype": u["qtype"], "label_kind": u["label_kind"], "source": u["source"],
                   "license": u["license"], "base_id": u["base_id"], "record_id": u["record_id"], "qid": str(qid),
                   **({"group_id": rec["group_id"]} if rec.get("group_id") else {}),
                   "truncated": False, "prompt": text,
                   **({"images": [{"sha256": sha, "s3": img.get("s3"), "width": wpx, "height": hpx, "n_expanded": int(K),
                                   "span": [span[0], span[1]], "pixels": px_src, "photo_credit": img.get("photo_credit")}]}
                      if img else {})}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["rows"] += 1
            stats["images"] += 1 if img else 0
            stats["text_rows"] = stats.get("text_rows", 0) + (0 if img else 1)
            stats["tokens"] += ntok
            stats["expansions"] += int(K or 0)
    spec = {"schema": SCHEMA_V2, "builder": "claude_lmscore.build_prompt+Gemma4Processor", **md5s,
            "tokenizer_files": tokenizer_files(a.tokenizer), "processor_files": processor_files(a.processor),
            "image_layout": layout, "max_len": int(a.max_len), "noul_defs": bool(a.noul_defs),
            "state_contract": "one image-marker token per image; n_tokens = EXPANDED count"}
    ren = hashlib.sha256(canon(spec).encode()).hexdigest()
    with open(tmp) as f, open(a.out, "w", encoding="utf-8") as g:
        for line in f:
            r = json.loads(line)
            r["renderer_sha256"] = ren
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.remove(tmp)
    header = {"manifest_header": True, "schema": SCHEMA_V2, "version": VERSION, "tag": a.tag, "renderer_sha256": ren,
              "renderer_spec": spec, "image_layout": layout, "render_seed": a.render_seed,
              "n_exposures": stats["rows"], "n_unique_decisions": len(weights),
              "total_decision_weight": float(sum(weights.values())), "parents": {}, "stats": stats,
              "rows_sha256": sha256_file(a.out)}
    with open(a.out + ".header.json", "w") as f:
        json.dump(header, f, indent=1, sort_keys=True)
    return header


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("render", "eligible", "assemble", "render-freeze", "devset", "render-vision"))
    ap.add_argument("--code", default=os.path.dirname(os.path.abspath(__file__)), help="dir with the pinned claude_lmscore.py")
    ap.add_argument("--data")
    ap.add_argument("--splits", default="train")
    ap.add_argument("--exposures", default="")
    ap.add_argument("--route-all", choices=ROUTES, default=None, help="no --exposures: route every unit this way (tests / probes)")
    ap.add_argument("--tokenizer")
    ap.add_argument("--tokenizer-revision", default="842da3794eaa0b77d5f08bae87a17459d91ff475")
    ap.add_argument("--parents")
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--noul-defs", type=int, default=0)
    ap.add_argument("--render-seed", type=int, default=0)
    ap.add_argument("--allow-truncation", action="store_true")
    ap.add_argument("--tag", default="m")
    ap.add_argument("--rows")
    ap.add_argument("--joined")
    ap.add_argument("--route", choices=("stock", "garas"), default=None)
    ap.add_argument("--pools", nargs="*", help="render-freeze / devset: NAME=DIR (pool root holding <split>/*.jsonl)")
    ap.add_argument("--tables", nargs="*", help="devset: NAME=units_tokens_<pool>.jsonl.gz")
    ap.add_argument("--hits", nargs="*", help="devset: NAME=hits_<pool>.jsonl.gz")
    ap.add_argument("--panel", nargs="*", help="devset: NAME=FILE.jsonl or NAME=DIR (every *.jsonl a block) (a dev panel)")
    ap.add_argument("--panel-sample", nargs="*", help="devset: NAME=N, a seeded per-file-proportional sample of that panel")
    ap.add_argument("--mix", default="", help="devset: the recipe's exposures file (pool shares)")
    ap.add_argument("--exclude-base-ids", nargs="*", help="devset: POOL=exclusions.json ({excluded_base_ids: [...]})")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--pool", default="", help="render-vision: the single pool file of image-conditioned records")
    ap.add_argument("--processor", default="", help="render-vision: dir with the pinned Gemma-4 processor / preprocessor json")
    ap.add_argument("--pixels-from", default="", help="render-vision: MODE B, sha256 -> staged <pixel .pt> json (else processor mode A)")
    ap.add_argument("--pixel-dir", default="", help="render-vision: mode A writes each image's processor output here as <sha>.pt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = {"render": render, "eligible": eligible, "assemble": assemble, "render-freeze": render_freeze, "devset": devset,
           "render-vision": render_vision}[a.cmd](a)
    print("FULLFT_RENDER " + json.dumps({k: v for k, v in out.items() if k not in ("renderer_spec", "data_dir_fingerprint", "parents")}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
