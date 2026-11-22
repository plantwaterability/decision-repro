#!/usr/bin/env python3
"""ss_lib.py: helpers for the GPU SCREENSAVER (fleet/task_ss.sh, started by gq_runner.sh v3 on idle slots; user 2026-10-07: "whenever
GPUs are idle, can you do just be doing an extreme overfit on all our data. make something different per gpu ...").

Subcommands (each prints what task_ss.sh reads; errors -> exit 1 with a message on stderr):
  check-table TABLE                     validate fleetq/screensaver/variants.tsv (ids, slots, numbers, the FITTED rule)
  choose --table T --slot S --gpu-mib M --done F --failed F --claims F
                                        this slot's variants in preference order, "ID<TAB>free|claimed" (listing files: `aws s3api
                                        list-objects-v2 --query 'Contents[].[Key,LastModified]' --output text` of fleetq/screensaver/
                                        done/ failed/ claims/)
  row --table T --id V                  one row as JSON
  pack --out DIR --group G=TRAIN[,TRAIN..][:VAL[,VAL..]] ... --exclude NAME=DIR[,DIR..] ... [--input NAME=MD5 ...]
                                        build a pack: train/<G>__<stem>.jsonl + val/<G>__<stem>.jsonl from training splits, every unit
                                        whose record id or normalised state is in an --exclude set DROPPED (and counted), then a
                                        re-check that 0 remain; manifest.json (per-source unit counts, clean / NC, exclusion results)
  verify --pack DIR --exclude NAME=SPEC ...  re-check a built pack's train split against exclusion sets (exit 1 on any overlap)
  tar --dir DIR --out F.tgz             deterministic tgz (sorted, mtime 0, uid 0, no xattrs) -> prints its md5
  select --pack DIR --groups G,G|* --out DIR      a variant's data dir: symlinks to the pack's files of those groups
  count --data DIR [--allow-nc] [--code DIR] --out F   train / val units the trainer will use (its NC and K <= 52 filters), JSON
  plan --table T --id V --pack-md5 M --pack-manifest F --count F --code-md5 M --s3 ROOT
                                        the pinned plan (trainer arguments, step budget) as JSON
  exec --plan P --code DIR --data D --out O [--py PY] [--dyn-log L] [--extra ARGS]
                                        exec the trainer with the plan's arguments (this process becomes the trainer)
  summary --plan P --out DIR LOG...     end-of-variant artefacts: capacity_curve.tsv (train loss by step, val by eval step, epochs),
                                        summary.json (val-minimum step = "best epoch count" evidence) [+ --resume FILE: full loss trace]
  dyncheck PART... [--per-step N]      a --dyn-log part is sane (keys incl. the soft-target fit fields, steps, draws, kl = ce - H >= 0)
  subset --pack DIR --out DIR --units N  a seeded, source-stratified subset of a built pack (the memorisation pack mem20k)
  addval --pack DIR --out DIR --group G=VAL --parent-md5 M [--exclude NAME=SPEC ...] [--input NAME=MD5 ...]
                                        a DERIVED pack: the built parent copied byte for byte + a val split for group G (pack p1x2 =
                                        p1x + bp_ssval_v1: p1x's bp group had no val); train re-checked against the added val
  guard PATH...                         exit 1 if a file outside the screensaver's own code points at a screensaver artefact (the
                                        "never a candidate, never a zero-shot / leaderboard row" rule, checked mechanically)
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import io
import json
import math
import os
import re
import sys
import tarfile
import time
from datetime import datetime

COLS = ["id", "slot", "base", "data", "nc", "lora_r", "targets", "lr", "batch", "epochs", "max_steps", "min_gpu_gib", "dyn", "flags",
        "tests", "downstream"]
MODELS = {   # short base names in the table -> HF ids (a full id or a local path is used as given)
    "q27": "Qwen/Qwen3.8-27B", "g31": "google/gemma-4-31B-it", "g12": "google/gemma-4-12B-it", "e4b": "google/gemma-4-E4B-it"}
MAX_LETTERS = 52                      # claude_lmtune drops units with more options than its letters (ls.LABELS)
NC_LICENCE = re.compile(r"\bNC\b|non[- ]?commercial|research[- ]only|research use only|academic use only|cc[- ]by[- ]nc|\bby-nc\b", re.I)
NC_SOURCES = {"ms_marco"}             # = trainers/cache_features.py (count prefers the pinned code's own copy)
SS_CKPT = re.compile(r"screensaver/[^\s'\"]*\.(pt|resume)\b|\.screensaver\.pt\b|\bss_[a-z0-9_-]+\.pt\b")


def die(msg):
    print(f"ss_lib: {msg}", file=sys.stderr)
    raise SystemExit(1)


# --------------------------------------------------------------------------- the variants table
def read_table(path):
    rows, seen = [], set()
    lines = [l.rstrip("\n") for l in open(path, encoding="utf-8")]
    head = None
    for n, l in enumerate(lines, 1):
        if not l.strip() or l.startswith("#"):
            continue
        f = l.split("\t")
        if head is None:
            head = f
            if head[:len(COLS)] != COLS:
                die(f"{path}: header must start with {COLS}, got {head}")
            continue
        if len(f) < len(COLS):
            die(f"{path}:{n}: {len(f)} columns, want {len(COLS)}")
        r = dict(zip(head, f))
        r["_line"] = n
        rows.append(r)
    if head is None:
        die(f"{path}: no header")
    return rows


def check_table(rows):
    """The table's own rules; returns a list of problems (empty = OK)."""
    bad, ids, slots, fitted = [], set(), {}, []
    for r in rows:
        i = r["id"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", i):
            bad.append(f"{i}: id must be [a-z0-9_-] (no dots: failed/<id>.<slot>.<ts> is split on them)")
        if i in ids:
            bad.append(f"{i}: duplicate id")
        ids.add(i)
        if r["slot"] != "-":
            if r["slot"] in slots:
                bad.append(f"{i}: slot {r['slot']} already maps to {slots[r['slot']]}")
            slots[r["slot"]] = i
        if r["nc"] not in ("clean", "nc"):
            bad.append(f"{i}: nc must be clean or nc")
        if r["targets"] not in ("attn", "attn+mlp"):
            bad.append(f"{i}: targets must be attn or attn+mlp")
        if r["dyn"] not in ("yes", "no"):
            bad.append(f"{i}: dyn must be yes or no")
        if not re.fullmatch(r"\d+x\d+", r["batch"]):
            bad.append(f"{i}: batch must be BxA")
        try:
            assert int(r["lora_r"]) > 0 and float(r["lr"]) > 0 and float(r["epochs"]) > 0 and int(r["max_steps"]) > 0
            assert float(r["min_gpu_gib"]) > 0
        except (AssertionError, ValueError):
            bad.append(f"{i}: lora_r / lr / epochs / max_steps / min_gpu_gib must be positive numbers")
        if ":" not in r["data"]:
            bad.append(f"{i}: data must be PACK:GROUPS")
        if "fitted" in i or "td" in r["data"].split(":", 1)[-1].split(","):
            fitted.append(i)
        if not r["tests"].strip() or not r["downstream"].strip():
            bad.append(f"{i}: every row states what it tests and its downstream use")
    if len(fitted) > 1:
        bad.append(f"at most ONE variant may train on typed-decisions data (labelled FITTED): {fitted}")
    for i in fitted:
        r = next(x for x in rows if x["id"] == i)
        if "fitted" not in i or "FITTED" not in r["tests"] + r["downstream"]:
            bad.append(f"{i}: a typed-decisions variant needs 'fitted' in its id and 'FITTED' in its description")
    return bad


def listing(path, sub):
    """`list-objects-v2 ... --output text` lines -> {name under .../SUB/: LastModified}; the line 'None' = an empty listing."""
    out = {}
    if not path or not os.path.exists(path):
        return out
    for l in open(path, encoding="utf-8"):
        f = l.rstrip("\n").split("\t")
        k = f[0]
        m = k.find(f"/{sub}/")
        if m >= 0 and k != "None":
            out[k[m + len(sub) + 2:]] = f[1] if len(f) > 1 else ""
    return out


def choose(rows, slot, gpu_mib, done, failed, claims, max_fail=3, skip=(), tag=""):
    """This slot's variants in preference order: its mapped row(s) first, then the unassigned rows (slot '-') in table order.
    Skipped: done, failed >= max_fail times, skipped on this node (skip/<id>.<TAG>), needing more GPU memory than this GPU has,
    mapped to another slot."""
    fails = collections.Counter(n.split(".")[0] for n in failed)
    out = []
    for mine in (True, False):
        for r in rows:
            if (r["slot"] == slot) != mine or (not mine and r["slot"] != "-"):
                continue
            if r["id"] in done or fails[r["id"]] >= max_fail or f"{r['id']}.{tag}" in skip:
                continue
            if float(r["min_gpu_gib"]) * 1024 > gpu_mib:
                continue
            out.append((r["id"], "claimed" if r["id"] in claims else "free", r["data"].split(":")[0]))
    return out


# --------------------------------------------------------------------------- packs
def norm(s):
    """= analysis/claude_overlap.norm (the fleet's state-overlap key; tested equal)."""
    s = re.sub(r"\s+", " ", str(s)).strip()
    if len(s) > 1 and s[0] == s[-1] == '"':
        s = s[1:-1]
    return s.strip().lower()


def h(s):
    return hashlib.sha1(s.encode("utf-8", "surrogatepass")).hexdigest()[:20]


EVAL_PART = re.compile(r"^(val|valid|validation|dev|test|eval|heldout|held_out|holdout|novel|confirm|sealed|bench|probe|extended|"
                       r"overlap|ood|unlabel|teacher_fill|gde1|meragpt)", re.I)


def jsonl_pairs(spec):
    """'DIR' or 'DIR,DIR' or file paths -> sorted [(path, name relative to its DIR)] of *.jsonl (recursive; AppleDouble ._* skipped).
    'evalsplits:DIR' = only the files under DIR inside a DIRECTORY named like an eval split (val / dev / test / eval / heldout /
    novel / confirm / sealed / bench...) and under no train* directory: the eval splits of a corpus whose layout is not known in
    advance (file names are not used: a train file may well be called test_of_time.jsonl)."""
    out = {}
    for d in spec.split(","):
        if d.startswith("evalsplits:"):
            d = d[len("evalsplits:"):]
            for p in glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True):
                rel = os.path.relpath(p, d)
                dirs = rel.split(os.sep)[:-1]
                if not os.path.basename(p).startswith("._") and any(EVAL_PART.match(x) for x in dirs) \
                        and not any(x.lower().startswith("train") for x in dirs):
                    out[p] = rel
        elif os.path.isfile(d):
            out[d] = os.path.basename(d)
        else:
            for p in glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True):
                if not os.path.basename(p).startswith("._"):
                    out[p] = os.path.relpath(p, d)
    return sorted(out.items())


def jsonl_files(spec):
    return [p for p, _ in jsonl_pairs(spec)]


def is_nc(rec, stem):
    src = str(rec.get("source") or stem)
    fam = src.split("/")[0].split("__")[0]
    lic = str(rec.get("license") or (rec.get("meta") or {}).get("license") or "")
    return bool(NC_LICENCE.search(lic)) or fam in NC_SOURCES


def file_md5(p):
    m = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            m.update(b)
    return m.hexdigest()


RESERVED = re.compile(r"dirichlet_share", re.I)   # research/final_suite.md 3.4: reserved until the final suite retires


def shash(state):
    """= build_jev_rd3.shash as hex: sha1 of claude_overlap.norm of the state (a non-string state as JSON) -- the sealed indexes' key."""
    s = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    return hashlib.sha1(norm(s).encode("utf-8")).hexdigest()


def _entity_fn():
    """build_jev_rd1.entity_keys (the sealed rd index's entity keys) from the repo ($JEV_J or the cwd), or None if not importable."""
    for d in (os.environ.get("JEV_J", ""), os.getcwd(), "/data/shared/jev"):
        if d and os.path.exists(os.path.join(d, "build_jev_rd1.py")):
            try:
                import importlib.util
                sys.path[:0] = [d, os.path.join(d, "analysis")]
                spec = importlib.util.spec_from_file_location("ss_rd1", os.path.join(d, "build_jev_rd1.py"))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod.entity_keys
            except Exception as e:  # noqa: BLE001
                print(f"ss_lib: build_jev_rd1.entity_keys not importable from {d}: {e!r}", file=sys.stderr)
    return None


def load_exclusions(excludes):
    """[(NAME, spec)] -> X: the exclusion keys. spec = jsonl dirs / files / 'evalsplits:DIR' (records: normalised states + ids), or
    'index:PATH' (a sealed exclusion index, research/final_suite.md 3.4: ids + unassigned eval-only ids, state sha1s, entity keys,
    cluster keys). X: ids {id: names}, states {h: names}, nostate (ids of state-less records), idx_ids / idx_sha1 / idx_ekeys /
    idx_clusters {key: names}, info {name: what was read}, entity (the entity-key function or None)."""
    X = {k: collections.defaultdict(set) for k in ("ids", "states", "idx_ids", "idx_sha1", "idx_ekeys", "idx_clusters")}
    X["nostate"], X["info"], X["entity"] = set(), {}, None
    for name, spec in excludes:
        if spec.startswith("index:"):
            path = spec[len("index:"):]
            ix = json.load(gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8"))
            for k, src in (("idx_ids", "ids"), ("idx_ids", "unassigned_eval_only_ids"), ("idx_sha1", "state_sha1"),
                           ("idx_ekeys", "entity_keys"), ("idx_clusters", "clusters")):
                for v in ix.get(src) or []:
                    X[k][str(v)].add(name)
            X["info"][name] = {"spec": spec, "kind": "sealed index", "md5": file_md5(path), "suite": ix.get("suite"),
                               **{k: len(ix.get(k) or []) for k in ("ids", "unassigned_eval_only_ids", "state_sha1", "entity_keys",
                                                                    "clusters")}}
            if not (ix.get("ids") or ix.get("state_sha1")):
                die(f"exclusion index {name} ({path}) has no ids / state sha1s")
            continue
        fs, n, nst = jsonl_files(spec), 0, 0
        for p in fs:
            for l in open(p, encoding="utf-8"):
                if not l.strip():
                    continue
                r = json.loads(l)
                n += 1
                if r.get("id") is not None:
                    X["ids"][str(r["id"])].add(name)
                    if r.get("state") is None:
                        X["nostate"].add(str(r["id"]))
                if r.get("state") is not None:
                    X["states"][h(norm(r["state"]))].add(name); nst += 1
        X["info"][name] = {"spec": spec, "files": len(fs), "records": n, "with_state": nst, "md5": {p: file_md5(p) for p in fs}}
        if n == 0 and not all(x.startswith("evalsplits:") for x in spec.split(",")):
            die(f"exclusion set {name} ({spec}) has no records: refusing to build a pack without it")
    if X["idx_ekeys"]:
        X["entity"] = _entity_fn()
    return X


def hits(r, X, stem=""):
    """The reasons a TRAINING record must be dropped (empty set: keep), and the exclusion sets sharing only its id (kept, counted)."""
    why, idk = set(), set()
    st = r.get("state")
    if st is not None:
        why |= {f"state:{x}" for x in X["states"].get(h(norm(st)), ())}
        if X["idx_sha1"]:
            why |= {f"state:{x}" for x in X["idx_sha1"].get(shash(st), ())}
    rid = str(r.get("id"))
    if rid in X["nostate"]:
        why |= {f"id:{x}" for x in X["ids"][rid]}
    elif rid in X["ids"] and not why:
        idk |= X["ids"][rid]
    why |= {f"id:{x}" for x in X["idx_ids"].get(rid, ())}
    if X["idx_clusters"]:
        m = r.get("meta") if isinstance(r.get("meta"), dict) else {}
        src = str(r.get("source") or stem)
        for s_ in {src, src.split("/")[0], src.replace("/", "__"), stem}:
            why |= {f"cluster:{x}" for x in X["idx_clusters"].get(f"{s_}:{m.get('group') or r.get('id')}", ())}
    if X["entity"] is not None and X["idx_ekeys"]:
        try:
            for k in X["entity"](r):
                why |= {f"entity:{x}" for x in X["idx_ekeys"].get(k, ())}
        except Exception:  # noqa: BLE001 -- a record the rd entity rules cannot read has no entity keys
            pass
    m = r.get("meta") if isinstance(r.get("meta"), dict) else {}
    if RESERVED.search(str(r.get("source") or stem)) or RESERVED.search(str(m.get("family") or "")):
        why.add("reserved:dirichlet_share")
    return why, idk


LOADER = {"qo": None}                   # cache_features.question_options when `pack --loader-check DIR` is given


def schema_ok(r):
    """claude_lmtune's loader needs questions {qid: {...}} and labels with every qid (cache_features.iter_units); with --loader-check,
    every question must also pass the loader's own question_options (what iter_units calls)."""
    q, lab = r.get("questions"), r.get("labels")
    if not (isinstance(q, dict) and bool(q) and isinstance(lab, dict) and all(k in lab for k in q) and r.get("id") is not None):
        return False
    if LOADER["qo"] is not None:
        try:
            for k, x in q.items():
                LOADER["qo"](x, lab[k])
        except Exception:  # noqa: BLE001 -- any loader error: the trainer would die on this record
            return False
    return True


def set_loader(d):
    import importlib.util
    sys.path[:0] = [d]
    spec = importlib.util.spec_from_file_location("ss_cf_loader", os.path.join(d, "cache_features.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    LOADER["qo"] = mod.question_options


def recheck(out, X):
    """Every TRAIN record of the pack at OUT, read back from the written files: no exclusion hit, loader schema OK."""
    bad = collections.Counter()
    for p in sorted(glob.glob(os.path.join(out, "train", "*.jsonl"))):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            why, _ = hits(r, X, os.path.splitext(os.path.basename(p))[0])
            for w in why:
                bad[w.split(":")[0]] += 1
            if not schema_ok(r):
                bad["schema"] += 1
    return {"train": {"state_overlaps": bad["state"], "stateless_id_overlaps": bad["id"], "index_cluster": bad["cluster"],
                      "index_entity": bad["entity"], "reserved": bad["reserved"], "schema": bad["schema"]}, "ok": not bad,
            "entity_check": X["entity"] is not None or not X["idx_ekeys"]}


def build_pack(out, groups, excludes, inputs):
    """groups: [(G[@nc], train_spec, val_spec)], excludes: [(NAME, spec)]. Writes out/train, out/val, out/manifest.json.
    G@nc tags every record of that group NC (licence string), so the trainer drops it unless --allow-nc (clean / NC lines)."""
    X = load_exclusions(excludes)
    man = {"built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "inputs": dict(inputs), "groups": {}, "exclusions": X["info"],
           "dropped": {}, "id_overlap_kept": {}, "units": {}, "records": {}, "files": {}, "recheck": {},
           "entity_check": X["entity"] is not None if X["idx_ekeys"] else None,
           "rule": "a TRAIN record is DROPPED when (1) its normalised state (analysis/claude_overlap.norm) is the state of any record "
                   "of an exclusion set, or its id is the id of an exclusion record that has no state; (2) it matches a sealed index "
                   "(research/final_suite.md 3.4) by id / unassigned eval-only id / normalised-state sha1 / cluster <source>:<group> / "
                   "entity key (build_jev_rd1.entity_keys, when importable: entity_check); (3) it is of the reserved family "
                   "dirichlet_share; (4) it lacks the loader's schema (questions + labels). An id shared with an exclusion record "
                   "whose state differs is kept and counted (id_overlap_kept): ids like 'train0' recur. val = monitoring, kept."}
    for split in ("train", "val"):
        os.makedirs(os.path.join(out, split), exist_ok=True)
    for g, tspec, vspec in groups:
        g, _, flag = g.partition("@")         # NAME@nc: a source DECLARED non-commercial: every record tagged NC (the trainer's filter)
        force_nc = flag == "nc"
        if force_nc:
            man.setdefault("declared_nc", []).append(g)
        for split, spec in (("train", tspec), ("val", vspec)):
            if not spec:
                continue
            fs = jsonl_pairs(spec)
            if split == "train" and not fs:
                die(f"group {g}: no train files in {spec}")
            for p, rel in fs:
                stem = os.path.splitext(os.path.basename(p))[0]
                oname = f"{g}__{os.path.splitext(rel)[0].replace(os.sep, '__')}.jsonl"
                if os.path.exists(os.path.join(out, split, oname)):
                    die(f"two inputs map to {split}/{oname}")
                key = f"{split}/{oname}"
                kept, drop, idk, ndrop = 0, collections.Counter(), collections.Counter(), 0
                with open(os.path.join(out, split, oname), "w", encoding="utf-8") as w:
                    for l in open(p, encoding="utf-8"):
                        if not l.strip():
                            continue
                        r = json.loads(l)
                        if not schema_ok(r):
                            drop["schema"] += 1; ndrop += 1; continue
                        if split == "train":           # val = monitoring only: the exclusions guard what is TRAINED on
                            why, idk_ = hits(r, X, stem)
                            if why:
                                drop.update(why); ndrop += 1; continue
                            idk.update(idk_)
                        if not r.get("source"):        # the pack renames files: keep the effective source (the NC filter reads it)
                            r["source"] = stem
                        if force_nc and not is_nc(r, stem):
                            old = str(r.get("license") or (r.get("meta") or {}).get("license") or "")
                            r["license"] = (old + " | NC: source declared non-commercial (screensaver pack)").strip(" |")
                        w.write(json.dumps(r, ensure_ascii=False) + "\n"); kept += 1
                        u = len(r.get("questions") or {})
                        nc = "nc" if is_nc(r, stem) else "clean"
                        src = str(r.get("source") or stem).split("/")[0]
                        c = man["units"].setdefault(split, {}).setdefault(g, {}).setdefault(src, {"clean": 0, "nc": 0})
                        c[nc] += u
                man["records"][key] = kept
                if drop:
                    man["dropped"][key] = dict(drop)
                    man.setdefault("dropped_records", {})[key] = ndrop
                if idk:
                    man["id_overlap_kept"][key] = dict(idk)
                man["groups"].setdefault(g, {"train": [], "val": []})[split].append(oname)
                man["files"][key] = {"from": p, "md5": file_md5(os.path.join(out, split, oname))}
    man["recheck"] = recheck(out, X)          # on the WRITTEN files
    if not man["recheck"]["ok"]:
        die(f"RE-CHECK FAILED: {man['recheck']}")
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1, sort_keys=True)
    return man


def det_tar(d, out):
    """A deterministic tgz of directory d (the pack's top dir is its basename): sorted, mtime 0, uid/gid 0, no xattrs."""
    base = os.path.basename(os.path.normpath(d))
    with open(out, "wb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as z, \
            tarfile.open(fileobj=z, mode="w", format=tarfile.PAX_FORMAT) as t:   # streamed: a pack of any size, never in memory
        for root, dirs, files in os.walk(d):
            dirs.sort()
            for n in sorted(dirs) + sorted(files):
                p = os.path.join(root, n)
                if n.startswith("._"):
                    continue
                ti = t.gettarinfo(p, arcname=os.path.join(base, os.path.relpath(p, d)))
                ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname, ti.pax_headers = 0, 0, 0, "", "", {}
                if ti.isfile():
                    with open(p, "rb") as f:
                        t.addfile(ti, f)
                elif ti.isdir():
                    t.addfile(ti)
    return file_md5(out)


def select(pack, groups, out):
    man = json.load(open(os.path.join(pack, "manifest.json")))
    want = sorted(man["groups"]) if groups == "*" else groups.split(",")
    miss = [g for g in want if g not in man["groups"]]
    if miss:
        die(f"groups {miss} not in the pack (has {sorted(man['groups'])})")
    for split in ("train", "val"):
        os.makedirs(os.path.join(out, split), exist_ok=True)
        for p in glob.glob(os.path.join(out, split, "*.jsonl")):
            os.remove(p)
        for g in want:
            for n in man["groups"][g][split]:
                os.symlink(os.path.abspath(os.path.join(pack, split, n)), os.path.join(out, split, n))
    return want


def count(data, allow_nc, code=None):
    """Train / val units the trainer keeps: cache_features.iter_units (the pinned code's when given), minus NC unless allowed,
    minus more than MAX_LETTERS options; per source."""
    cf = None
    if code and os.path.exists(os.path.join(code, "trainers", "cache_features.py")):
        import importlib.util
        spec = importlib.util.spec_from_file_location("ss_cf", os.path.join(code, "trainers", "cache_features.py"))
        cf = importlib.util.module_from_spec(spec)
        sys.path[:0] = [os.path.join(code, "trainers"), code]
        spec.loader.exec_module(cf)
    out = {"code_cache_features": bool(cf)}
    for split in ("train", "val"):
        n, per = 0, collections.Counter()
        if cf is not None:
            units = cf.iter_units(data, [split], set(), None)
            nc_f = getattr(cf, "is_noncommercial", None)
            for u in units:
                if not allow_nc and nc_f is not None and nc_f(u):
                    continue
                if len(u["option_keys"]) > MAX_LETTERS:
                    continue
                n += 1; per[str(u["source"]).split("/")[0]] += 1
        else:                                   # no pinned code (tests): records x questions, NC by the same rule
            for p in sorted(glob.glob(os.path.join(data, split, "*.jsonl"))):
                stem = os.path.splitext(os.path.basename(p))[0]
                for l in open(p, encoding="utf-8"):
                    r = json.loads(l)
                    if not allow_nc and is_nc(r, stem):
                        continue
                    k = len(r["questions"]); n += k; per[str(r.get("source") or stem).split("/")[0]] += k
        out[split] = n
        out[f"{split}_by_source"] = dict(per)
    return out


# --------------------------------------------------------------------------- the plan and the trainer's command line
def lora_info(row):
    """the run's LoRA convention from its row (rank, placement, scale rule): the adapter multiplier the trainer will apply
    (train_e2e.lora_scaling: alpha_over_r = 32 / r; rs = anchor / sqrt(r))."""
    fl = row.get("flags", "").split()
    rule = fl[fl.index("--lora-scale-rule") + 1] if "--lora-scale-rule" in fl else "alpha_over_r"
    anchor = float(fl[fl.index("--lora-scale-anchor") + 1]) if "--lora-scale-anchor" in fl else 8.0
    r = int(row["lora_r"])
    return {"r": r, "targets": row["targets"], "scale_rule": rule, "alpha": 32, "anchor": anchor if rule == "rs" else None,
            "scale": (32 / r) if rule == "alpha_over_r" else anchor / math.sqrt(r)}


def make_plan(row, pack_md5, pack_man, cnt, code_md5, s3root):
    b, acc = (int(x) for x in row["batch"].split("x"))
    per_step = b * acc
    units = int(cnt["train"])
    if units <= 0:
        die(f"{row['id']}: no train units")
    budget = min(int(row["max_steps"]), max(1, math.ceil(float(row["epochs"]) * units / per_step)))
    horizon = 2 * budget                       # the cosine of a 2B-step run, stopped at B: lr falls only to half its peak
    ev = max(int(os.environ.get("SS_EVAL_MIN", "50")), min(1000, int(round(budget / 50.0, -1))))   # ~50 evals per variant
    model = MODELS.get(row["base"], row["base"])
    v = row["id"]
    args = ["--model", model, "--steps", str(horizon), "--stop-after-step", str(budget), "--batch-size", str(b), "--grad-accum",
            str(acc), "--lr", row["lr"], "--warmup", "40", "--lora-r", row["lora_r"], "--max-len", "2048", "--eval-every", str(ev),
            "--save-every", str(min(ev, 200)), "--eval-units", "400", "--seed", "0", "--resume", "auto",
            # the resume file (one per variant, replaced in place) carries the LoRA; --ckpt-s3 "" = no per-save LoRA copies on S3
            # (claude_ckpt keeps a .stepNNNNN copy of every save: ~300 x 0.4 GB per variant); the final LoRA goes up at the end
            "--resume-s3", f"{s3root}/screensaver/{v}/resume", "--ckpt-s3", ""]
    if row["targets"] == "attn+mlp":
        args.append("--lora-mlp")
    if row["nc"] == "nc":
        args.append("--allow-nc")
    if row["flags"].strip() not in ("", "-"):
        args += row["flags"].split()
    fitted = "fitted" in v
    return {"variant": v, "row": {k: row[k] for k in COLS}, "lora": lora_info(row), "model": model, "pack": row["data"].split(":")[0],
            "groups": row["data"].split(":", 1)[1], "pack_md5": pack_md5, "code_md5": code_md5, "units_train": units,
            "units_val": int(cnt["val"]), "units_by_source": cnt.get("train_by_source", {}), "examples_per_step": per_step,
            "steps_budget": budget, "steps_horizon": horizon, "epochs_target": float(row["epochs"]),
            "epochs_planned": round(budget * per_step / units, 3), "eval_every": ev, "args": args,
            "role": "screensaver", "candidate": False, "zero_shot_eligible": False, "teacher_eligible": False,
            "fitted": fitted, "nc_line": row["nc"] == "nc",
            "pack_inputs": pack_man.get("inputs", {}), "exclusion_sets": sorted(pack_man.get("exclusions", {})),
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


PACK_KEYS = ("dropped", "inputs", "units", "groups", "files", "exclusions", "recheck")


def count_units(pack_dir, groups):
    """{split: {group: {source: {clean, nc}}}} recounted from a pack's own files (the build_pack rule), with each file's md5."""
    units, md5s = {}, {}
    for g, sp in groups.items():
        for split in ("train", "val"):
            for fn in sp.get(split, []):
                p = os.path.join(pack_dir, split, fn)
                if not os.path.exists(p):
                    die(f"{pack_dir}: {split}/{fn} listed by the manifest is missing")
                md5s[f"{split}/{fn}"] = file_md5(p)
                stem = os.path.splitext(fn)[0]
                for l in open(p, encoding="utf-8"):
                    r = json.loads(l)
                    src = str(r.get("source") or stem).split("/")[0]
                    c = units.setdefault(split, {}).setdefault(g, {}).setdefault(src, {"clean": 0, "nc": 0})
                    c["nc" if is_nc(r, stem) else "clean"] += len(r.get("questions") or {})
    return units, md5s


def normalize_pack(pm, pack_dir=None):
    """A pack manifest in the REGULAR schema whatever built it (review AM P1). Regular packs: checked for PACK_KEYS. A SUBSET pack
    (kind 'subset', ss_lib subset): its own selection provenance under 'subset' (rule, seed, quotas, available / selected units by
    source, selected-line hashes), its parent's under 'parent_pack' (name, md5, inputs, built, exclusion history, the parent's own
    drops), 'dropped' = {} (a subset drops nothing by exclusion: it draws from verified training files), 'inputs' = the parent's inputs
    + the parent pack's md5, and group-indexed 'units' RECOUNTED from the pack's files (pack_dir; never defaulted to empty). The 10-08
    mem20k manifest (written before this schema) is converted the same way; files are md5-checked against the manifest."""
    if pm.get("kind") != "subset":
        miss = [k for k in PACK_KEYS if k not in pm]
        if miss:
            die(f"pack manifest lacks {miss}")
        return pm
    if all(k in pm for k in PACK_KEYS) and "subset" in pm and "parent_pack" in pm:
        return pm                                        # already the compatible schema (subset() since 10-08)
    if not pack_dir or not os.path.isdir(pack_dir):
        die("a subset pack manifest in the 10-07 schema needs the pack's files (pack_dir) to recount its per-group units")
    units, md5s = count_units(pack_dir, pm["groups"])
    bad = [k for k, m in md5s.items() if k in pm.get("files", {}) and pm["files"][k].get("md5") != m]
    if bad:
        die(f"{pack_dir}: {len(bad)} file(s) differ from the subset manifest's md5s, e.g. {bad[:3]}")
    seed = re.search(r"seeded \(([^)]*)\)", pm.get("rule", ""))
    out = dict(pm)
    out.update({
        "inputs": {**(pm.get("parent_inputs") or {}), f"parent_pack:{pm.get('parent')}": pm.get("parent_md5")},
        "dropped": {}, "id_overlap_kept": {}, "units": units,
        "subset": {"rule": pm.get("rule"), "seed": seed.group(1) if seed else None, "built": pm.get("built"),
                   "units": pm.get("units"), "selected": pm.get("selected"), "schema": "converted from the 10-07 subset manifest"},
        "parent_pack": {"name": pm.get("parent"), "md5": pm.get("parent_md5"), "inputs": pm.get("parent_inputs"),
                        "built": pm.get("parent_built"), "dropped": pm.get("dropped_in_parent"),
                        "exclusion_sets": sorted(pm.get("exclusions") or {}), "recheck": pm.get("recheck")}})
    return out


MANIFEST_REQUIRED = ("variant", "pack", "pack_md5", "groups", "code_md5", "units_selected", "units_trained", "files", "exclusions",
                     "exclusion_recheck", "args", "candidate", "zero_shot_eligible", "teacher_eligible")


def validate_manifest(m):
    """A per-run manifest that can be published: every required key, real md5s, non-empty provenance, the role flags off; subsets carry
    their selection rule and parent md5. -> list of problems (empty = valid)."""
    bad = [f"missing {k}" for k in MANIFEST_REQUIRED if k not in m]
    if bad:
        return bad
    for k in ("pack_md5", "code_md5"):
        if not re.fullmatch(r"[0-9a-f]{32}", str(m[k])):
            bad.append(f"{k} {m[k]!r} is not an md5")
    if not m["groups"] or not m["files"] or not m["exclusions"]:
        bad.append("empty groups / files / exclusions")
    tr = m["units_selected"].get("train") or {}
    if not tr or not any(c.get("clean", 0) + c.get("nc", 0) for g in tr.values() for c in g.values()):
        bad.append("no selected train units by group / source")
    if int(m["units_trained"]) <= 0:
        bad.append("units_trained <= 0")
    if not (m["exclusion_recheck"] or {}).get("ok"):
        bad.append("the pack's exclusion re-check is not ok")
    if m["candidate"] is not False or m["zero_shot_eligible"] is not False or m["teacher_eligible"] is not False:
        bad.append("role flags must be False")
    if m.get("pack_kind") == "subset":
        sub, par = m.get("subset") or {}, m.get("parent_pack") or {}
        if not sub.get("rule") or not re.fullmatch(r"[0-9a-f]{32}", str(par.get("md5"))):
            bad.append("a subset needs its selection rule and its parent pack's md5")
    return bad


def manifest(plan, pack_man, identity="", dynamics=None, pack_dir=None):
    """Per-variant PROVENANCE: corpus md5, per-source unit counts of the selected groups, the exclusion checks, role flags, the LoRA
    scale convention; a subset pack's selection rule and parent provenance."""
    pm = normalize_pack(pack_man, pack_dir)
    groups = sorted(pm["groups"]) if plan["groups"] == "*" else plan["groups"].split(",")
    miss = [g for g in groups if g not in pm["groups"]]
    if miss:
        die(f"plan groups {miss} not in the pack manifest")
    keys = [f"{s}/{n}" for g in groups for s in ("train", "val") for n in pm["groups"][g][s]]
    out = {"variant": plan["variant"], "role": "screensaver: user-directed idle filler (2026-10-07); NOT a candidate; never in a "
           "zero-shot / leaderboard row, stack or teacher" + ("; FITTED on typed-decisions data" if plan["fitted"] else ""),
           "candidate": False, "zero_shot_eligible": False, "teacher_eligible": False, "fitted": plan["fitted"],
           "licence_line": "NC (research use; --allow-nc)" if plan["nc_line"] else "clean (the trainer's NC filter on)",
           "model": plan["model"], "pack": plan["pack"], "pack_md5": plan["pack_md5"], "pack_kind": pm.get("kind", "pack"),
           "groups": groups, "code_md5": plan["code_md5"], "trainer_identity": identity,
           "units_selected": {s: {g: pm["units"].get(s, {}).get(g, {}) for g in groups} for s in ("train", "val")},
           "units_trained": plan["units_train"], "units_by_source_trained": plan["units_by_source"],
           "files": {k: pm["files"][k] for k in keys if k in pm["files"]},
           "exclusions": pm["exclusions"], "dropped": {k: v for k, v in pm["dropped"].items() if k in keys},
           "id_overlap_kept": {k: v for k, v in pm.get("id_overlap_kept", {}).items() if k in keys},
           "exclusion_recheck": pm["recheck"], "pack_inputs": pm.get("inputs", {}),
           "steps_budget": plan["steps_budget"], "epochs_planned": plan["epochs_planned"], "args": plan["args"],
           "lora": plan.get("lora") or lora_info(plan["row"]), "exposure_planned": exposure(plan),
           **({"dynamics": dynamics} if dynamics is not None else {})}
    if pm.get("kind") == "subset":
        par = dict(pm["parent_pack"])
        par["dropped"] = {k: v for k, v in (par.get("dropped") or {}).items() if k in keys}   # the parent's drops in THESE files
        out.update({"subset": pm["subset"], "parent_pack": par})
    return out


def write_json_atomic(obj, path, check=None):
    """tmp file, read back + parse, an optional validator, then os.replace: a published file is never empty or invalid."""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
    back = json.load(open(tmp))
    probs = check(back) if check else []
    if probs:
        os.remove(tmp)
        die(f"refusing to write {path}: " + "; ".join(probs))
    os.replace(tmp, path)


# --------------------------------------------------------------------------- end-of-variant artefacts
STEP_MEAN = re.compile(r"^\s*\[step-mean\] step (\d+) loss ([0-9.eE+-]+|nan) over (\d+) examples")
VAL = re.compile(r"^step (\d+)\s+val soft-CE ([0-9.eE+-]+|nan)\s+acc ([0-9.eE+-]+|nan)")


def exposure(plan, steps_done=None):
    """What a run's step count means for its units (pooled sampler: the unit list is cycled in a fixed seeded order, so pass p sees
    every unit once): passes, observations per unit, unique coverage; < 1 pass = a PARTIAL training run (coverage), not an overfit."""
    U, per = plan["units_train"], plan["examples_per_step"]
    st = plan["steps_budget"] if steps_done is None else steps_done
    passes = st * per / U if U else 0.0
    return {"steps": st, "units_train": U, "examples_per_step": per, "passes": round(passes, 4),
            "unique_coverage": round(min(1.0, passes), 4), "observations_per_unit": round(passes, 4),
            "kind": ("PARTIAL run: fewer than one pass, coverage only" if passes < 1 else
                     "few passes: trajectories too short for fit judgements" if passes < 3 else "repeated exposure")}


def dyn_runs(paths):
    """per trainer start, the dynamics status JSON task_ss.sh wrote (requested / enabled / reason / steps covered)."""
    out = []
    for p in sorted(paths):
        try:
            out.append(json.load(open(p)))
        except (OSError, ValueError):
            pass
    req = [r for r in out if r.get("requested") == "yes"]
    cov = sum(max(0, int(r.get("to_step") or 0) - int(r.get("from_step") or 0)) for r in out if r.get("enabled") == "yes")
    tot = sum(max(0, int(r.get("to_step") or 0) - int(r.get("from_step") or 0)) for r in out)
    return {"runs": out, "requested": bool(req), "steps_with_dynamics": cov, "steps_total": tot,
            "gaps": [{"from_step": r.get("from_step"), "to_step": r.get("to_step"), "reason": r.get("reason")}
                     for r in out if r.get("requested") == "yes" and r.get("enabled") != "yes"]}


def update_norms(lora_ckpt, scale):
    """||scale * B A||_F per adapted module of a LoRA checkpoint (the function update the adapter adds to W), via the r x r Gram
    matrices (no out x in product); total = sqrt(sum of squares). None when the file is not a torch LoRA checkpoint."""
    try:
        import torch
        ck = torch.load(lora_ckpt, map_location="cpu", weights_only=True)
        lo = ck["lora"]
    except Exception:  # noqa: BLE001
        return None
    norms = []
    for n in sorted({k[:-2] for k in lo if k.endswith(".A")}):
        A, B = lo[n + ".A"].double(), lo[n + ".B"].double()
        g = float(torch.trace((A @ A.T) @ (B.T @ B)).clamp_min(0)) ** 0.5
        norms.append(scale * g)
    if not norms:
        return None
    return {"modules": len(norms), "total_fro": math.sqrt(sum(x * x for x in norms)), "mean_fro": sum(norms) / len(norms),
            "max_fro": max(norms), "scale": scale, "note": "||scale * B A||_F of the final adapter, per module"}


def summary(plan, logs, out, resume=None, dyn_status=(), lora_ckpt=None):
    tr, va = {}, {}
    for p in logs:                              # later logs win (a resumed run redoes, bit for bit, the steps after its save)
        for l in open(p, encoding="utf-8", errors="replace"):
            m = STEP_MEAN.match(l)
            if m:
                tr[int(m.group(1))] = float(m.group(2))
            m = VAL.match(l)
            if m:
                va[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    full = None
    if resume and os.path.exists(resume):       # the resume file's per-step loss trace: every step, not every 10th
        try:
            import torch
            full = [float(x) for x in torch.load(resume, map_location="cpu", weights_only=True)["train_state"]["losses"]]
        except Exception:  # noqa: BLE001 -- not a torch resume file (tests) or unreadable: the logs' every-10th-step curve
            try:
                full = [float(x) for x in json.load(open(resume))["losses"]]
            except Exception:  # noqa: BLE001
                full = None
    os.makedirs(out, exist_ok=True)
    U, per = plan["units_train"], plan["examples_per_step"]
    no_val = plan.get("units_val") == 0         # (10-10, ss09) an EMPTY val split: the trainer logs "val soft-CE 0.0000 acc 0.0000",
    if no_val:                                 # zeros that are not measurements: never a val minimum / epoch evidence
        va = {}
    steps = sorted(set(tr) | set(va) | (set(range(1, len(full) + 1)) if full else set()))
    with open(os.path.join(out, "capacity_curve.tsv"), "w") as f:
        f.write("step\tepoch\ttrain_loss_logged\ttrain_loss_step\tval_soft_ce\tval_acc\n")
        for s in steps:
            v = va.get(s, ("", ""))
            f.write(f"{s}\t{s * per / U:.4f}\t{tr.get(s, '')}\t{full[s - 1] if full and s <= len(full) else ''}\t{v[0]}\t{v[1]}\n")
    vs = [(s, x[0]) for s, x in va.items() if not math.isnan(x[0])]
    best = min(vs, key=lambda t: (t[1], t[0])) if vs else (None, float("nan"))
    last = max(va) if va else None
    res = {"variant": plan["variant"], "steps_budget": plan["steps_budget"], "units_train": U, "examples_per_step": per,
           "last_step_logged": max(steps) if steps else 0, "epochs_done": round((max(steps) if steps else 0) * per / U, 3),
           "val_step0": va.get(0, (None,))[0], "val_min_step": best[0], "val_min_soft_ce": best[1],
           "val_min_epoch": round(best[0] * per / U, 3) if best[0] is not None else None,
           "val_end_step": last, "val_end_soft_ce": va[last][0] if last is not None else None,
           "overfit_gap": (va[last][0] - best[1]) if last is not None and best[0] is not None else None,
           "train_loss_end": (full[-1] if full else tr[max(tr)] if tr else None), "train_loss_min": min(full or list(tr.values()) or [float("nan")]),
           "evals": len(va), "logs": [os.path.basename(p) for p in logs], "full_trace": bool(full),
           "exposure": exposure(plan, max(steps) if steps else 0), "dynamics": dyn_runs(dyn_status),
           "lora": plan.get("lora") or (lora_info(plan["row"]) if "row" in plan else None),
           "update_norms": (update_norms(lora_ckpt, (plan.get("lora") or lora_info(plan["row"]))["scale"])
                            if lora_ckpt and (plan.get("lora") or "row" in plan) else None),
           "note": "val = the TRAINING distribution's val split; the val-minimum step is evidence for a real run's epoch count, "
                   "not a model selection. Never a candidate.",
           **({"val": "NONE: the pack has no val split for this variant's groups (plan units_val 0); the logged 0.0000 val lines "
                      "are the trainer's empty-val output, not measurements, and are left out"} if no_val else {})}
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1)
    return res


DYN_KEYS = ("step", "epoch", "i", "uid", "src", "k", "gold", "p_gold", "p_tgt", "loss", "correct")
FIT_KEYS = ("h_tgt", "ce_tgt", "kl_tgt", "l2_tgt")


def _dup_keys(pairs):
    seen = {}
    for k, v in pairs:
        if k in seen:
            raise ValueError(f"duplicate key {k!r}")
        seen[k] = v
    return seen


def _num(x, ints=False):
    """a finite number of the right kind (bool is not a number here)"""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    if ints:
        return isinstance(x, int)
    return math.isfinite(x)


def dyncheck(path, per_step=0, need_fit=True, expect=None):
    """A --dyn-log part is VALID (review AM P2: enforced, fails closed): every record parses with no duplicate JSON key; every key
    (the soft-target fit fields too, unless need_fit is off); integer step / i / k / gold with 0 <= gold < k, epoch an int or null,
    correct a bool, uid present; finite p_gold / p_tgt / loss (and h / ce / kl / l2) in range (p in [0, 1], 0 <= H <= log k,
    kl = ce - H >= 0, 0 <= l2 <= 2); no duplicate (step, i); steps never go back and leave no gap; each step's draws are 0..n-1
    (n = per_step when given). Only the LAST line may be torn (a run killed mid-write: counted as torn_tail). expect = (first, last)
    steps the run reports: missing steps there are COVERAGE gaps (reported, not invalidity). -> (ok, stats)."""
    st, bad, steps, last, seen = collections.Counter(), collections.Counter(), collections.defaultdict(set), -1, set()
    op = gzip.open if path.endswith(".gz") else open
    lines = op(path, "rt", encoding="utf-8").read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for n, line in enumerate(lines):
        st["lines"] += 1
        try:
            r = json.loads(line, object_pairs_hook=_dup_keys)
            if not isinstance(r, dict):
                raise ValueError("not an object")
        except ValueError as e:
            if n == len(lines) - 1 and "duplicate key" not in str(e):
                st["torn_tail"] += 1
            else:
                bad["duplicate_json_key" if "duplicate key" in str(e) else "unparsable"] += 1
            continue
        miss = [k for k in DYN_KEYS + (FIT_KEYS if need_fit else ()) if k not in r]
        if miss:
            bad["missing:" + ",".join(miss)] += 1
            continue
        if not all(_num(r[k], ints=True) for k in ("step", "i", "k", "gold")) or not (r["epoch"] is None or _num(r["epoch"], True)) \
                or not isinstance(r["correct"], bool) or r["uid"] in (None, "", []):
            bad["type"] += 1
            continue
        fl = ("p_gold", "p_tgt", "loss") + (FIT_KEYS if need_fit else ())
        if not all(_num(r[k]) for k in fl):
            bad["nonfinite"] += 1
            continue
        s_, i_, k_ = r["step"], r["i"], r["k"]
        if k_ < 1 or not 0 <= r["gold"] < k_ or s_ < 0 or i_ < 0:
            bad["range:index"] += 1
        if not (0 <= r["p_gold"] <= 1 and 0 <= r["p_tgt"] <= 1 + 1e-6) or r["loss"] < -1e-6:
            bad["range:prob"] += 1
        if need_fit:
            h, ce, kl, l2 = (r[x] for x in FIT_KEYS)
            if not (-1e-6 <= h <= math.log(k_) + 1e-4) or ce < -1e-6 or kl < -1e-4 or abs(ce - h - kl) > 1e-3 or not 0 <= l2 <= 2 + 1e-6:
                bad["range:fit"] += 1
        if (s_, i_) in seen:
            bad["duplicate_step_i"] += 1
            continue
        seen.add((s_, i_))
        if s_ < last:
            bad["step_back"] += 1
        last = max(last, s_)
        steps[s_].add(i_)
        st["records"] += 1
    for s_, ii in steps.items():
        if ii != set(range(len(ii))) or (per_step and len(ii) != per_step):
            bad["step_draws"] += 1
    if steps:
        gaps = (max(steps) - min(steps) + 1) - len(steps)
        if gaps:
            bad["step_gaps"] += gaps
    out = {**dict(st), "steps": len(steps), "first_step": min(steps) if steps else None, "last_step": max(steps) if steps else None}
    if expect and expect[0] is not None and expect[1] is not None and expect[1] >= expect[0]:
        want = set(range(int(expect[0]), int(expect[1]) + 1))
        out["coverage"] = {"expected_first": int(expect[0]), "expected_last": int(expect[1]), "missing": len(want - set(steps)),
                           "extra": len(set(steps) - want)}
    if not st["records"]:
        bad["no_records"] += 1
    return not bad, {**out, "bad": dict(bad)}


def subset(pack, out, units, seed, floor=100, clean_only=True, val_units=2000, parent_md5=None):
    """A fixed SOURCE-STRATIFIED subset of a built pack (the memorisation pack): whole records (all their questions) drawn in a seeded
    order per source until the source's quota of UNITS is reached; quota = max(min(available, floor), proportional share), scaled so
    the total is <= units; clean_only: NC units never drawn (the clean line). val: the same rule over the pack's val split.
    The parent's exclusion record is inherited (the subset is a subset of verified training files); provenance in manifest.json."""
    import random as _r
    pm = json.load(open(os.path.join(pack, "manifest.json")))
    os.makedirs(out, exist_ok=True)
    parent = os.path.basename(pack.rstrip("/"))
    rule = (f"seeded ({seed}) draw of whole records, target {units} train units, PROPORTIONAL to each source's size with a floor: "
            f"per-source quota = max(min(available, {floor}), proportional share) (not even across sources), clean_only={clean_only}; "
            f"val: the same rule, {val_units} units")
    man = {"built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kind": "subset", "parent": parent, "parent_md5": parent_md5,
           "parent_inputs": pm.get("inputs"), "parent_built": pm.get("built"), "rule": rule,
           "exclusions": pm.get("exclusions"), "recheck": pm.get("recheck"), "dropped_in_parent": pm.get("dropped"),
           "declared_nc": pm.get("declared_nc", []), "groups": {}, "units": {}, "records": {}, "files": {}, "selected": {}}
    for split, want in (("train", units), ("val", val_units)):
        os.makedirs(os.path.join(out, split), exist_ok=True)
        pool = collections.defaultdict(list)         # source -> [(file, line no, units)]
        for p in sorted(glob.glob(os.path.join(pack, split, "*.jsonl"))):
            stem = os.path.splitext(os.path.basename(p))[0]
            for n, l in enumerate(open(p, encoding="utf-8")):
                r = json.loads(l)
                if clean_only and is_nc(r, stem):
                    continue
                pool[str(r.get("source") or stem).split("/")[0]].append((os.path.basename(p), n, len(r["questions"])))
        avail = {s_: sum(x[2] for x in v) for s_, v in pool.items()}
        tot = sum(avail.values())
        if not tot:
            continue
        quota = {s_: max(min(a_, floor), int(want * a_ / tot)) for s_, a_ in avail.items()}
        scale = min(1.0, want / max(1, sum(quota.values())))
        quota = {s_: max(1, int(q * scale)) for s_, q in quota.items()}
        keep = collections.defaultdict(set)
        got = collections.Counter()
        for s_ in sorted(pool):
            order = list(pool[s_])
            _r.Random(f"{seed}|{split}|{s_}").shuffle(order)
            for fn, n, u in order:
                if got[s_] + u > quota[s_]:
                    continue
                keep[fn].add(n); got[s_] += u
                if got[s_] >= quota[s_]:
                    break
        for fn in sorted(keep):
            src_p = os.path.join(pack, split, fn)
            dst_p = os.path.join(out, split, fn)
            with open(dst_p, "w", encoding="utf-8") as w:
                for n, l in enumerate(open(src_p, encoding="utf-8")):
                    if n in keep[fn]:
                        w.write(l)
            man["files"][f"{split}/{fn}"] = {"from": f"{man['parent']}/{split}/{fn}", "md5": file_md5(dst_p), "lines": len(keep[fn])}
            man["records"][f"{split}/{fn}"] = len(keep[fn])
            man["selected"][f"{split}/{fn}"] = hashlib.md5(",".join(map(str, sorted(keep[fn]))).encode()).hexdigest()
            g = fn.split("__")[0]
            man["groups"].setdefault(g, {"train": [], "val": []})[split].append(fn)
        man["units"][split] = {"total": sum(got.values()), "by_source": dict(sorted(got.items())), "available": dict(sorted(avail.items())),
                               "quota": dict(sorted(quota.items()))}
    # (review AM P1) the regular-pack schema on top: a per-run manifest reads a subset like any pack (normalize_pack)
    man = normalize_pack(man, out)
    man["subset"]["schema"] = "native (ss_lib subset, 10-08)"
    write_json_atomic(man, os.path.join(out, "manifest.json"), lambda m: [k for k in PACK_KEYS if k not in m])
    return man


def add_val(pack, out, adds, parent_md5, excludes=(), inputs=()):
    """A DERIVED pack (2026-10-10, ss09's empty val): the built parent pack at PACK copied file for file (every train / val file
    byte-identical, md5-checked against the parent manifest) + a VAL split for group(s) that had none. adds: [(G, val_spec)]. The new
    val records pass the loader schema; any that hit EXCLUDES (eval sets / sealed indexes, the train rule) are dropped and counted.
    Then the copied TRAIN split is re-checked against the added val (0 shared normalised states / state-less ids, else die): the val
    is never trained on. Manifest = the parent's in the regular schema + derived provenance; the parent is never modified."""
    import shutil
    pm = normalize_pack(json.load(open(os.path.join(pack, "manifest.json"))), pack)
    if os.path.exists(out):
        die(f"{out} exists")
    man = json.loads(json.dumps(pm))
    for split in ("train", "val"):
        os.makedirs(os.path.join(out, split), exist_ok=True)
        for g in sorted(pm["groups"]):
            for n in pm["groups"][g][split]:
                src_p, dst_p = os.path.join(pack, split, n), os.path.join(out, split, n)
                shutil.copyfile(src_p, dst_p)
                want = (pm["files"].get(f"{split}/{n}") or {}).get("md5")
                if want != file_md5(dst_p):
                    die(f"{split}/{n}: md5 {file_md5(dst_p)} is not the parent manifest's {want}")
    X = load_exclusions(excludes) if excludes else None
    added, drops = {}, {}
    for g, spec in adds:
        if g not in pm["groups"]:
            die(f"group {g} is not in the parent pack (has {sorted(pm['groups'])})")
        fs = jsonl_pairs(spec)
        if not fs:
            die(f"group {g}: no val files in {spec}")
        force_nc = g in (pm.get("declared_nc") or [])
        for p, rel in fs:
            stem = os.path.splitext(os.path.basename(p))[0]
            oname = f"{g}__{os.path.splitext(rel)[0].replace(os.sep, '__')}.jsonl"
            if os.path.exists(os.path.join(out, "val", oname)):
                die(f"val/{oname} exists in the parent pack")
            key, kept, drop = f"val/{oname}", 0, collections.Counter()
            with open(os.path.join(out, "val", oname), "w", encoding="utf-8") as w:
                for l in open(p, encoding="utf-8"):
                    if not l.strip():
                        continue
                    r = json.loads(l)
                    if not schema_ok(r):
                        drop["schema"] += 1; continue
                    why = hits(r, X, stem)[0] if X is not None else set()
                    if why:
                        drop.update(why); continue
                    if not r.get("source"):
                        r["source"] = stem
                    if force_nc and not is_nc(r, stem):
                        old = str(r.get("license") or (r.get("meta") or {}).get("license") or "")
                        r["license"] = (old + " | NC: source declared non-commercial (screensaver pack)").strip(" |")
                    w.write(json.dumps(r, ensure_ascii=False) + "\n"); kept += 1
                    src = str(r.get("source") or stem).split("/")[0]
                    c = man["units"].setdefault("val", {}).setdefault(g, {}).setdefault(src, {"clean": 0, "nc": 0})
                    c["nc" if is_nc(r, stem) else "clean"] += len(r.get("questions") or {})
            if not kept:
                die(f"{key}: every record dropped ({dict(drop)})")
            man["groups"][g]["val"].append(oname)
            man["records"][key] = kept
            man["files"][key] = {"from": p, "md5": file_md5(os.path.join(out, "val", oname))}
            added[key] = {"group": g, "from": p, "records": kept, "md5": man["files"][key]["md5"]}
            if drop:
                drops[key] = dict(drop)
                man["dropped"][key] = dict(drop)
    XV = load_exclusions([(f"added_val_{g}", spec) for g, spec in adds])
    rv = recheck(out, XV)                      # the copied TRAIN split vs the added val: never trained on
    if not rv["ok"]:
        die(f"RE-CHECK FAILED: the train split shares records with the added val: {rv}")
    man["exclusions"].update({k: {**v, "role": "added val (train re-checked against it)"} for k, v in XV["info"].items()})
    if X is not None:
        man["exclusions"].update({f"added_val_screen:{k}": {**v, "role": "screen of the added val records only"}
                                  for k, v in X["info"].items()})
    man["inputs"] = {**pm.get("inputs", {}), f"parent_pack:{os.path.basename(os.path.normpath(pack))}": parent_md5, **dict(inputs)}
    man["recheck"] = {**pm["recheck"], "ok": bool(pm["recheck"].get("ok")) and rv["ok"],
                      "added_val": {"train_vs_added_val": rv["train"], "ok": rv["ok"], "val_drops": drops}}
    man["derived"] = {"kind": "parent pack + added val", "parent": os.path.basename(os.path.normpath(pack)), "parent_md5": parent_md5,
                      "parent_built": pm.get("built"), "train_files_identical_to_parent": True, "added": added,
                      "rule": "every parent file copied and md5-checked against the parent manifest; the added val records pass the "
                              "loader schema and the given exclusion sets (dropped + counted); the train split re-checked against the "
                              "added val (0 shared states / state-less ids)"}
    man["built"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json_atomic(man, os.path.join(out, "manifest.json"), lambda m: [k for k in PACK_KEYS if k not in m])
    return man


def guard(paths, own=("task_ss.sh", "ss_lib.py", "gq_runner.sh", "ss_corpus.sh", "ss_variants.tsv", "test_screensaver.py",
                      "task_ss_valscore.sh")):
    """Every place in the code under PATHS (*.sh *.py *.tsv *.json *.yaml) that names a screensaver CHECKPOINT (a .pt / .resume
    under screensaver/, lora_ss_*, ss_<id>.pt, *.screensaver.pt), except the screensaver's own files: a scorer / stack / teacher /
    leaderboard job pointing at one fails this."""
    hits = []
    for root in paths:
        fs = [root] if os.path.isfile(root) else [p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                                                  if os.path.isfile(p) and p.endswith((".sh", ".py", ".tsv", ".json", ".yaml"))
                                                  and os.path.relpath(p, root).split(os.sep)[0] not in ("tmp", ".venv", "__pycache__")]
        for p in fs:
            if os.path.basename(p) in own:
                continue
            try:
                txt = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            for m in SS_CKPT.finditer(txt):
                hits.append((p, m.group(0)))
    return hits


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("check-table"); c.add_argument("table")
    c = sp.add_parser("choose")
    for k in ("--table", "--slot", "--done", "--failed", "--claims", "--skip", "--tag"):
        c.add_argument(k, required=k in ("--table", "--slot"), default="")
    c.add_argument("--gpu-mib", type=float, required=True)
    c = sp.add_parser("row"); c.add_argument("--table", required=True); c.add_argument("--id", required=True)
    c = sp.add_parser("pack"); c.add_argument("--out", required=True); c.add_argument("--group", action="append", default=[])
    c.add_argument("--exclude", action="append", default=[]); c.add_argument("--input", action="append", default=[])
    c.add_argument("--loader-check", default="", help="an trainers/ dir: validate every record with its cache_features.question_options")
    c = sp.add_parser("verify"); c.add_argument("--pack", required=True); c.add_argument("--exclude", action="append", default=[])
    c = sp.add_parser("tar"); c.add_argument("--dir", required=True); c.add_argument("--out", required=True)
    c = sp.add_parser("select"); c.add_argument("--pack", required=True); c.add_argument("--groups", required=True)
    c.add_argument("--out", required=True)
    c = sp.add_parser("count"); c.add_argument("--data", required=True); c.add_argument("--allow-nc", action="store_true")
    c.add_argument("--code", default=""); c.add_argument("--out", required=True)
    c = sp.add_parser("plan")
    for k in ("--table", "--id", "--pack-md5", "--pack-manifest", "--count", "--code-md5", "--s3"):
        c.add_argument(k, required=True)
    c = sp.add_parser("manifest"); c.add_argument("--plan", required=True); c.add_argument("--pack-manifest", required=True)
    c.add_argument("--dynamics", default="", help="a dynamics status JSON file (this start's), recorded in the manifest")
    c.add_argument("--pack-dir", default="", help="the pack's extracted dir (a subset's per-group units are recounted from it)")
    c.add_argument("--out", default="", help="write here (validated, tmp + rename; exit 1 and nothing written if invalid)")
    c.add_argument("--note", default="", help="a provenance note recorded in the manifest (e.g. a backfill)")
    c.add_argument("--identity", default="")
    c = sp.add_parser("exec")
    for k in ("--plan", "--code", "--data", "--out"):
        c.add_argument(k, required=True)
    c.add_argument("--py", default=sys.executable); c.add_argument("--dyn-log", default=""); c.add_argument("--extra", default="")
    c = sp.add_parser("summary"); c.add_argument("--plan", required=True); c.add_argument("--out", required=True)
    c.add_argument("--dyn-status", action="append", default=[], help="dynamics status JSON files (one per trainer start)")
    c.add_argument("--lora-ckpt", default="", help="the final adapter (--out): function-update norms ||scale B A||")
    c.add_argument("--resume", default=""); c.add_argument("logs", nargs="*")
    c = sp.add_parser("guard"); c.add_argument("paths", nargs="+")
    c = sp.add_parser("dyncheck"); c.add_argument("parts", nargs="+"); c.add_argument("--per-step", type=int, default=0)
    c.add_argument("--expect-from", type=int, default=None); c.add_argument("--expect-to", type=int, default=None)
    c.add_argument("--json", default="", help="write {ok, stats} per part here (one part) / a list (several)")
    c.add_argument("--no-fit", action="store_true", help="a trainer before the soft-target fit fields (c8f75631)")
    c = sp.add_parser("subset"); c.add_argument("--pack", required=True); c.add_argument("--out", required=True)
    c.add_argument("--units", type=int, required=True); c.add_argument("--seed", default="20261008")
    c.add_argument("--floor", type=int, default=100); c.add_argument("--val-units", type=int, default=2000)
    c.add_argument("--with-nc", action="store_true"); c.add_argument("--parent-md5", default=None)
    c = sp.add_parser("addval"); c.add_argument("--pack", required=True); c.add_argument("--out", required=True)
    c.add_argument("--group", action="append", required=True, help="G=VAL_SPEC: the val split to add to group G")
    c.add_argument("--parent-md5", required=True); c.add_argument("--exclude", action="append", default=[])
    c.add_argument("--input", action="append", default=[])
    a = ap.parse_args()

    if a.cmd == "check-table":
        bad = check_table(read_table(a.table))
        for b in bad:
            print("TABLE:", b)
        print(f"TABLE_{'OK' if not bad else 'BAD'} {len(read_table(a.table))} rows")
        return 1 if bad else 0
    if a.cmd == "choose":
        rows = read_table(a.table)
        bad = check_table(rows)
        if bad:
            die("variants table invalid: " + "; ".join(bad))
        for v, st, pk in choose(rows, a.slot, a.gpu_mib, set(listing(a.done, "done")), list(listing(a.failed, "failed")),
                                set(listing(a.claims, "claims")), skip=set(listing(a.skip, "skip")), tag=a.tag):
            print(f"{v}\t{st}\t{pk}")
        return 0
    if a.cmd == "row":
        r = [x for x in read_table(a.table) if x["id"] == a.id]
        if not r:
            die(f"no variant {a.id}")
        print(json.dumps(r[0]))
        return 0
    if a.cmd == "pack":
        groups = []
        for g in a.group:
            name, _, rest = g.partition("=")
            tr, _, va = rest.partition(":")
            groups.append((name, tr, va))
        excl = [tuple(x.split("=", 1)) for x in a.exclude]
        inputs = [tuple(x.split("=", 1)) for x in a.input]
        if a.loader_check:
            set_loader(a.loader_check)
        man = build_pack(a.out, groups, excl, inputs)
        tot = {s: sum(c["clean"] + c["nc"] for g in man["units"].get(s, {}).values() for c in g.values()) for s in ("train", "val")}
        print(f"PACK_OK {a.out} train {tot['train']} units, val {tot['val']}; dropped "
              f"{sum(sum(v.values()) for v in man['dropped'].values())} records by the exclusions; re-check {man['recheck']}")
        return 0
    if a.cmd == "verify":                     # an independent re-check of a built pack against (possibly more) exclusion sets
        X = load_exclusions([tuple(x.split("=", 1)) for x in a.exclude])
        r = recheck(a.pack, X)
        print(json.dumps({"verify": r, "exclusions": {k: {x: v.get(x) for x in ("files", "records", "with_state", "ids", "state_sha1")}
                                                      for k, v in X["info"].items()}}))
        return 0 if r["ok"] else 1
    if a.cmd == "tar":
        print(det_tar(a.dir, a.out))
        return 0
    if a.cmd == "select":
        print(",".join(select(a.pack, a.groups, a.out)))
        return 0
    if a.cmd == "count":
        r = count(a.data, a.allow_nc, a.code or None)          # to a file: cache_features may print to stdout
        json.dump(r, open(a.out, "w"), sort_keys=True)
        print(f"COUNT train {r['train']} val {r['val']}")
        return 0
    if a.cmd == "plan":
        row = [x for x in read_table(a.table) if x["id"] == a.id][0]
        plan = make_plan(row, a.pack_md5, json.load(open(a.pack_manifest)), json.load(open(a.count)), a.code_md5, a.s3)
        print(json.dumps(plan, indent=1, sort_keys=True))
        return 0
    if a.cmd == "manifest":
        dy = json.load(open(a.dynamics)) if a.dynamics and os.path.exists(a.dynamics) else None
        m = manifest(json.load(open(a.plan)), json.load(open(a.pack_manifest)), a.identity, dy, a.pack_dir or None)
        if a.note:
            m["note"] = a.note
        probs = validate_manifest(m)
        if a.out:
            write_json_atomic(m, a.out, validate_manifest)
            print(f"MANIFEST_OK {a.out}")
        else:
            if probs:
                die("invalid manifest: " + "; ".join(probs))
            print(json.dumps(m, indent=1, sort_keys=True, default=str))
        return 0
    if a.cmd == "exec":
        plan = json.load(open(a.plan))
        argv = [a.py, os.path.join(a.code, "trainers", "claude_lmtune.py"), "--data", a.data, "--out", a.out, *plan["args"]]
        if a.dyn_log:
            argv += ["--dyn-log", a.dyn_log]
        argv += a.extra.split()
        print("SS_EXEC " + " ".join(argv), flush=True)
        env = dict(os.environ, PYTHONPATH=a.code + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
        os.execvpe(argv[0], argv, env)
    if a.cmd == "summary":
        r = summary(json.load(open(a.plan)), a.logs, a.out, a.resume or None, a.dyn_status, a.lora_ckpt or None)
        print("SS_SUMMARY " + json.dumps({k: r[k] for k in ("last_step_logged", "epochs_done", "val_min_step", "val_min_soft_ce",
                                                          "val_end_soft_ce", "train_loss_end")}
                                         | {"passes": r["exposure"]["passes"], "kind": r["exposure"]["kind"],
                                            "dyn_steps": r["dynamics"]["steps_with_dynamics"]}))
        return 0
    if a.cmd == "dyncheck":
        allok, res = True, []
        for p in a.parts:
            try:
                ok, stt = dyncheck(p, a.per_step, not a.no_fit, (a.expect_from, a.expect_to))
            except (OSError, EOFError, UnicodeDecodeError) as e:   # an unreadable / truncated .gz is an invalid part, not a crash
                ok, stt = False, {"bad": {"unreadable": str(e)[:200]}}
            allok &= ok
            res.append({"part": os.path.basename(p), "check_ok": ok, **stt})
            print(f"SS_DYN_CHECK {'OK' if ok else 'BAD'} {os.path.basename(p)} " + json.dumps(stt, sort_keys=True))
        if a.json:
            write_json_atomic(res[0] if len(res) == 1 else res, a.json)
        return 0 if allok else 1
    if a.cmd == "subset":
        m = subset(a.pack, a.out, a.units, a.seed, a.floor, not a.with_nc, a.val_units, a.parent_md5)
        su = m["subset"]["units"]
        print(f"SUBSET_OK {a.out}: train {su.get('train', {}).get('total')} units from {len(su.get('train', {}).get('by_source', {}))} "
              f"sources (proportional with a floor), val {su.get('val', {}).get('total')}")
        return 0
    if a.cmd == "addval":
        adds = [tuple(g.split("=", 1)) for g in a.group]
        if any(len(x) != 2 or not all(x) for x in adds):
            die("--group must be G=VAL_SPEC")
        m = add_val(a.pack, a.out, adds, a.parent_md5, [tuple(x.split("=", 1)) for x in a.exclude],
                    [tuple(x.split("=", 1)) for x in a.input])
        uv = m["units"].get("val", {})
        print(f"ADDVAL_OK {a.out}: " + ", ".join(f"{g} val {sum(c['clean'] + c['nc'] for c in uv.get(g, {}).values())} units"
                                                  for g, _ in adds)
              + f"; train re-check vs the added val {m['recheck']['added_val']['train_vs_added_val']}")
        return 0
    if a.cmd == "guard":
        hits = guard(a.paths)
        for p, mk in hits:
            print(f"GUARD: {p} names a screensaver artefact ({mk})")
        print(f"GUARD_{'OK' if not hits else 'FAIL'} {len(hits)} hit(s)")
        return 1 if hits else 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
