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


def load_v1(code_dir):
    global D
    import importlib.util
    spec = importlib.util.spec_from_file_location("decontam_v1", Path(code_dir) / "decontam.py")
    D = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(D)
    return D


def load_entity(repo):
    """build_jev_rd1.entity_keys (sealed rd index entity keys), as fleet/ss_lib._entity_fn does."""
    global ENTITY
    import importlib.util
    sys.path[:0] = [repo, os.path.join(repo, "analysis")]
    spec = importlib.util.spec_from_file_location("ev3_rd1", os.path.join(repo, "build_jev_rd1.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ENTITY = mod.entity_keys


def file_md5(p, kind="md5"):
    h = hashlib.new(kind)
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "little")


def hexkey(hx):
    return int(hx[:16], 16)


def cnorm(s):                                   # = analysis/claude_overlap.norm (fleet state key)
    s = re.sub(r"\s+", " ", str(s)).strip()
    if len(s) > 1 and s[0] == s[-1] == '"':
        s = s[1:-1]
    return s.strip().lower()


def flatten(state):                             # = build_jev_ext2.flatten
    raw = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    obj, t = None, raw.strip()
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


def shingles(normed):
    w = WORD.findall(normed)
    return np.unique(np.fromiter((hash(" ".join(w[i:i + SHINGLE_WORDS])) for i in range(len(w) - SHINGLE_WORDS + 1)),
                                 dtype=np.int64)) if len(w) >= SHINGLE_WORDS else np.zeros(0, np.int64)


def fam_of(rec, stem):
    return str(rec.get("source") or stem).split("/")[0].split("__")[0].lower()


def lineage_roots(rec, stem):
    """(strong, weak) namespaced roots. The bare record id and short numeric-like roots are weak (ids like 'train0' recur)."""
    m = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
    fam, rid = fam_of(rec, stem), str(rec.get("id"))
    cands = [rec.get("base_id"), m.get("variant_of"), m.get("group_id"), (m.get("bf") or {}).get("group") if isinstance(m.get("bf"), dict) else None,
             (m.get("lineage") or {}).get("root") if isinstance(m.get("lineage"), dict) else None]
    if isinstance(m.get("lineage"), dict):
        cands.extend(m["lineage"].get("parents") or [])
    if m.get("source_case_id") is not None:
        cands.append(f"sourcecase:{m['source_case_id']}")
    strong, weak = set(), set()
    for c in cands:
        if c is None or not str(c).strip():
            continue
        c = str(c).strip()
        (weak if (c == rid or len(c) < 6 or ROOT_NUMERIC.match(c)) else strong).add(f"lin:{fam}:{c}")
    return strong, weak


def text_units(raw):
    """cnorm text units of one field text -> {key: words}: the whole text (>= TEXT_MIN words) and, for texts of more than
    PARA_SPLIT_WORDS words, every blank-line paragraph and every line of >= TEXT_LONG words."""
    raw = str(raw)
    out = {}
    t = cnorm(raw)
    n = len(t.split())
    if n >= TEXT_MIN:
        out[h64("t:" + t)] = n
    if n > PARA_SPLIT_WORDS:
        for p in re.split(r"\n\s*\n", raw) + raw.split("\n"):
            pt = cnorm(p)
            pn = len(pt.split())
            if pn >= TEXT_LONG:
                out[h64("t:" + pt)] = pn
    return out


def state_keys(state):
    """record-level keys of one state -> (dict family -> set of ints, the flat normalised state text, text units)."""
    raw = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    k = {"state_exact": {h64("x:" + raw)}, "state_nfkc": {h64("n:" + D.norm(raw))}}
    sh = {hexkey(hashlib.sha1(cnorm(raw).encode("utf-8")).hexdigest())}
    if not isinstance(state, str):
        sh.add(hexkey(hashlib.sha1(cnorm(str(state)).encode("utf-8")).hexdigest()))
    k["sha1"] = sh
    flat, leaves = flatten(state)
    nf = cnorm(flat)
    k["state_flat"] = {h64("f:" + nf)}
    texts = {}
    for t in [nf] + [cnorm(x) for x in leaves]:
        n = len(t.split())
        if n >= TEXT_MIN:
            texts[h64("t:" + t)] = n
    for x in [flat] + leaves:
        texts.update(text_units(x))
    return k, nf, texts


def str_leaves(o, out):
    if isinstance(o, str):
        if o.strip():
            out.append(o)
    elif isinstance(o, dict):
        for v in o.values():
            str_leaves(v, out)
    elif isinstance(o, (list, tuple)):
        for v in o:
            str_leaves(v, out)


def question_fields(qid, q):
    """[(field id, text)] of every model-visible text of one question."""
    out = []
    if not isinstance(q, dict):
        lv = []
        str_leaves(q, lv)
        return [(FID["question_other"], x) for x in lv]
    cand = bool(CAND_QID.search(str(qid)))
    ins = q.get("instructions")
    if isinstance(ins, str):
        if ins.strip():
            out.append((FID["candidate_document"] if cand else FID["instructions"], ins))
    elif ins is not None:
        lv = []
        str_leaves(ins, lv)
        out += [(FID["question_other"], x) for x in lv]
    t = q.get("type")
    crit = q.get("criteria")
    if isinstance(crit, dict):
        for k, v in crit.items():
            txt = str(v) if v not in (None, "") else str(k)
            out.append((FID["noul_defs"] if t == "noul" else FID["option_text"], txt))
    elif crit is not None:
        lv = []
        str_leaves(crit, lv)
        out += [(FID["option_text"], x) for x in lv]
    if q.get("levels") is not None:
        lv = []
        str_leaves(q["levels"], lv)
        out += [(FID["option_text"], x) for x in lv]
    for k, v in q.items():
        if k in ("type", "instructions", "criteria", "levels") or k in NONVISIBLE_TOP:
            continue
        lv = []
        str_leaves(v, lv)
        out += [(FID["question_other"], x) for x in lv]
    return out


def looks_textual(s):
    s = str(s).strip()
    if s.startswith(("data:", "http://", "https://", "/", "s3://")) or re.fullmatch(r"[A-Za-z0-9+/=\s]{200,}", s):
        return False
    return len(s.split()) >= TEXT_MIN


def record_fields(rec, kind):
    """([(field id, text)] record-level visible texts other than the state, number of unindexable attachments)."""
    if kind == "rendered":
        return [(FID["prompt"], str(rec.get("prompt") or ""))], 0
    out, unidx = [], 0
    for k, v in rec.items():
        if k in NONVISIBLE_TOP or k.startswith("_"):
            continue
        lv = []
        str_leaves(v, lv)
        if k in ATTACH_KEYS:
            for x in lv:
                if looks_textual(x):
                    out.append((FID["attachment_text"], x))
                else:
                    unidx += 1
            if not lv and v:
                unidx += 1
        elif k in OCR_KEYS:
            out += [(FID["attachment_text"], x) for x in lv]
        else:
            out += [(FID["record_other"], x) for x in lv]
    return out, unidx


def unit_keys(rec, q):
    try:
        i = D.identity(rec, q)
    except Exception:                                   # noqa: BLE001 -- unsupported question type: state-level keys only
        return None
    return {"joint_exact": hexkey(i["joint_exact"]), "joint_norm": hexkey(i["joint_normalized"]),
            "question_options": hexkey(i["question_options"]), "question_norm": hexkey(i["question_normalized"]),
            "options_norm": hexkey(i["options_normalized"]), "template": hexkey(i["template"])}


def doc_of(text, fid=None):
    """(doc key, normalised text) of a CONTENT field text (A1) with >= SHINGLE_WORDS words, else None."""
    if fid is not None and fid not in CONTENT_FIDS:
        return None
    t = cnorm(text)
    if len(WORD.findall(t)) < SHINGLE_WORDS:
        return None
    return h64("d:" + t), t


def keyfn_sha256():
    """sha256 over the source of every function / constant that defines a key (scan refuses an index built otherwise)."""
    import inspect
    src = "".join(inspect.getsource(f) for f in (h64, hexkey, cnorm, flatten, shingles, fam_of, lineage_roots, text_units, state_keys,
                                                   str_leaves, question_fields, looks_textual, record_fields, unit_keys, doc_of))
    src += repr((SHINGLE_WORDS, DF_MAX, TEXT_MIN, TEXT_LONG, PARA_SPLIT_WORDS, WORD.pattern, ROOT_NUMERIC.pattern, FIELDS, CAND_QID.pattern,
                 sorted(NONVISIBLE_TOP), sorted(ATTACH_KEYS), sorted(OCR_KEYS), sorted(CONTENT_FIDS))) + file_md5(D.__file__, "sha256")
    return hashlib.sha256(src.encode()).hexdigest()


def bounded_imap(pool, fn, jobs, inflight):
    """ordered imap that never holds more than `inflight` pending tasks (the job generator streams a tarball)."""
    pend = collections.deque()
    for j in jobs:
        pend.append(pool.apply_async(fn, (j,)))
        if len(pend) >= inflight:
            yield pend.popleft().get()
    while pend:
        yield pend.popleft().get()


# =============================================================================================== stage
class HashReader:
    def __init__(self, f):
        self.f, self.md5, self.n = f, hashlib.md5(), 0

    def read(self, n=-1):
        b = self.f.read(n)
        self.md5.update(b)
        self.n += len(b)
        return b

    def drain(self):
        while self.read(1 << 22):
            pass


def s3_has(key):
    return subprocess.run(["aws", "s3api", "head-object", "--bucket", BUCKET, "--key", "jev-race/" + key],
                          capture_output=True).returncode == 0


def clean(name):
    return "/".join(x for x in name.split("/") if x not in (".", ""))


def evalsplit_member(name):
    parts = name.split("/")[:-1]
    return any(EVAL_PART.match(x) for x in parts) and not any(x.lower().startswith("train") for x in parts)


def stream_tar(key, want, outdir, side=None):
    """stream s3 KEY (tar.gz); extract members for which want(name) into outdir; side(name) -> extra dir for side members."""
    p = subprocess.Popen(["aws", "s3", "cp", "--only-show-errors", S3 + key, "-"], stdout=subprocess.PIPE)
    hr, got = HashReader(p.stdout), {}
    with tarfile.open(fileobj=hr, mode="r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = clean(m.name)
            if os.path.basename(name).startswith("._"):
                continue
            dst = None
            if want(name):
                dst = Path(outdir) / name
            elif side is not None and side(name):
                dst = Path(side.dir) / name
            if dst is None:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            f, h = tf.extractfile(m), hashlib.md5()
            with open(dst, "wb") as o:
                for b in iter(lambda: f.read(1 << 22), b""):
                    h.update(b)
                    o.write(b)
            got[name] = h.hexdigest()
    hr.drain()
    p.wait()
    if p.returncode:
        raise RuntimeError(f"aws s3 cp {key} exit {p.returncode}")
    return hr.md5.hexdigest(), got


def stage(a):
    reg = json.loads(Path(a.registry).read_text())
    st = Path(a.stage)
    st.mkdir(parents=True, exist_ok=True)
    manp = st / "stage_manifest.json"
    man = json.loads(manp.read_text()) if manp.exists() else {"populations": {}}
    man.update(registry_sha256=file_md5(a.registry, "sha256"), code_sha256=file_md5(__file__, "sha256"))
    only = set(a.only.split(",")) if a.only else None
    env = dict(os.environ, HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HOME=str(st / ".hf"), TOKENIZERS_PARALLELISM="false")
    env.pop("HF_TOKEN", None)
    env.pop("HUGGING_FACE_HUB_TOKEN", None)
    side_dir = st / "_side"

    def save():
        tmp = manp.with_suffix(".partial")
        tmp.write_text(json.dumps(man, indent=1, sort_keys=True))
        os.replace(tmp, manp)

    for P in reg["populations"]:
        name = P["population"]
        if only and name not in only:
            continue
        if man["populations"].get(name, {}).get("status") == "ok" and not a.redo:
            continue
        t0, rec = time.time(), {"kind": P["kind"], "required": P["required"], "historical": P.get("historical", False)}
        out = st / name
        try:
            if P["kind"] in ("tarball",):
                key = next((k for k in P["s3_candidates"] if s3_has(k)), None)
                if key is None:
                    raise RuntimeError(f"no S3 key among {P['s3_candidates']}")
                rec["s3_key"] = key
                if "files" in P:
                    wantset = {f["member"]: f["expected_md5"] for f in P["files"]}
                    want = wantset.__contains__
                elif P.get("member_regex") == "EVALSPLITS":
                    wantset, want = None, (lambda n: n.endswith(".jsonl") and evalsplit_member(n))
                else:
                    rx = re.compile(P["member_regex"])
                    wantset, want = None, (lambda n: bool(rx.match(n)))
                side = None
                if name == "decision_index_v021":
                    side = lambda n: n.startswith("suite-0.2/") or n.startswith("kit/")   # noqa: E731
                    side.dir = str(side_dir / "di_v021")
                tmd5, got = stream_tar(key, want, out, side)
                rec["tarball_md5"] = tmd5
                if P.get("tarball_md5") and tmd5 != P["tarball_md5"]:
                    raise RuntimeError(f"TARBALL_MD5 {tmd5} != pinned {P['tarball_md5']}")
                if wantset is not None:
                    missing = sorted(set(wantset) - set(got))
                    bad = sorted(k for k in wantset if k in got and got[k] != wantset[k])
                    if missing or bad:
                        raise RuntimeError(f"members missing {len(missing)} {missing[:3]} md5-mismatch {len(bad)} {bad[:3]}")
                    got = {k: got[k] for k in wantset}
                rec["files"] = [{"path": str(out / k), "member": k, "md5": v, "kind": "jev"} for k, v in sorted(got.items())]
                if not got and P.get("files"):
                    raise RuntimeError("no files")
            elif P["kind"] == "opaque_index":
                key = None
                for k in P["s3_candidates"]:
                    if s3_has(k):
                        key = k
                        break
                if key is None:          # the sealed final_v1 indexes live under a dated subdir: search like task_screen4.sh
                    base = os.path.basename(P["s3_candidates"][0])
                    ls = subprocess.run(["aws", "s3", "ls", "--recursive", S3 + "sealed/"], capture_output=True, text=True).stdout
                    hits = [l.split()[-1] for l in ls.splitlines() if l.split() and l.split()[-1].endswith("/" + base)]
                    key = hits[0].split("jev-race/", 1)[1] if hits else None
                if key is None:
                    raise RuntimeError("index key not found")
                out.mkdir(parents=True, exist_ok=True)
                dst = out / os.path.basename(key)
                subprocess.run(["aws", "s3", "cp", "--only-show-errors", S3 + key, str(dst)], check=True)
                m5 = file_md5(dst)
                if m5 != P["expected_md5"]:
                    raise RuntimeError(f"INDEX_MD5 {m5} != {P['expected_md5']}")
                rec.update(s3_key=key, files=[{"path": str(dst), "md5": m5, "kind": "opaque_index"}])
            elif P["kind"] == "build_jevbench":
                jr = out / "jev_real"
                if not (jr / "test/real.jsonl").exists():
                    r = subprocess.run([sys.executable, "trainers/convert_jevbench.py", "--out", str(jr)], cwd=a.repo, env=env,
                                       capture_output=True, text=True)
                    (out / "convert_jevbench.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-20000:])
                    if r.returncode:
                        raise RuntimeError(f"convert_jevbench exit {r.returncode}")
                files, subs = [], []
                for f in P["files"]:
                    p = jr / f["member"]
                    m5 = file_md5(p)
                    files.append({"path": str(p), "member": f["member"], "md5": m5, "kind": "jev", "pinned_md5": f["expected_md5"]})
                    if m5 != f["expected_md5"]:
                        subs.append(f["member"])
                rec.update(files=files, code_md5=file_md5(Path(a.repo) / "trainers/convert_jevbench.py"))
                if subs:
                    rec["substitute"] = {"members": subs, "reason": "rebuilt from the Hub today; bytes differ from the 10-10 pin "
                                         "(Hub dataset not revision-pinned by convert_jevbench.py); the current version is indexed"}
            elif P["kind"] == "derive_td":
                jr = st / "jev_real_valtest/jev_real"
                rows = {s: [l for l in open(jr / s / "real.jsonl", encoding="utf-8") if json.loads(l).get("source") == "typed-decisions"]
                        for s in ("train", "val")}
                hold = not rows["val"]
                (out / "train").mkdir(parents=True, exist_ok=True)
                (out / "val").mkdir(parents=True, exist_ok=True)
                with open(out / "train/td.jsonl", "w") as tr, open(out / "val/td.jsonl", "w") as va:   # = tmp/emo_svi/emo_pool.sh TD split
                    for l in rows["train"]:
                        r = json.loads(l)
                        if hold and int(hashlib.sha1(str(r.get("base_id") or r["id"]).encode()).hexdigest(), 16) % 20 == 0:
                            va.write(l)
                        else:
                            tr.write(l)
                    for l in rows["val"]:
                        va.write(l)
                files, subs = [], []
                for f in P["files"]:
                    p = out / f["member"]
                    m5 = file_md5(p)
                    files.append({"path": str(p), "member": f["member"], "md5": m5, "kind": "jev", "pinned_md5": f["expected_md5"]})
                    if m5 != f["expected_md5"]:
                        subs.append(f["member"])
                rec["files"] = files
                if subs:
                    rec["substitute"] = {"members": subs, "reason": "derived from the rebuilt jev_real; differs from the 10-10 pin"}
            elif P["kind"] == "build_di03_gsm8k":
                I = P["inputs"]
                kit_tgz = side_dir / "kit03.tgz"
                subprocess.run(["aws", "s3", "cp", "--only-show-errors", S3 + I["kit"], str(kit_tgz)], check=True)
                if file_md5(kit_tgz) != I["kit_md5"]:
                    raise RuntimeError("KIT_MD5")
                kit = side_dir / "kit03"
                kit.mkdir(parents=True, exist_ok=True)
                subprocess.run(["tar", "xzf", str(kit_tgz), "-C", str(kit)], check=True)
                code = side_dir / "code_di03"            # the pinned fleet copies (fleetq/code/di03/), as fleet/task_di03.sh fetches them
                (code / "trainers").mkdir(parents=True, exist_ok=True)
                subprocess.run(["aws", "s3", "cp", "--only-show-errors", S3 + "fleetq/code/di03/convert_di03.py", str(code / "trainers/convert_di03.py")], check=True)
                (code / "trainers/convert_decision_index.py").write_bytes((Path(a.repo) / "trainers/convert_decision_index.py").read_bytes())
                for rel, want5 in (("trainers/convert_di03.py", I["convert_di03_md5"]), ("trainers/convert_decision_index.py", I["convert_decision_index_md5"])):
                    if file_md5(code / rel) != want5:
                        raise RuntimeError(f"CODE_MD5 {rel}")
                s02 = side_dir / "di_v021/suite-0.2"
                if not (s02 / "selected-rows.jsonl.gz").exists():
                    raise RuntimeError("stage decision_index_v021 first (suite-0.2 side files)")
                pq = subprocess.run([sys.executable, "-c", "from huggingface_hub import hf_hub_download as h; print(h('openai/gsm8k', "
                                     "'main/test-00000-of-00001.parquet', repo_type='dataset', revision='740312add88f781978c0658806c59bc2815b9866', token=False))"],
                                    env=env, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
                kd = next(p.parent.parent for p in kit.rglob("editions.py") if p.parent.name == "decision_index")
                s03, work = side_dir / "suite-0.3", side_dir / "build03"
                r = subprocess.run([sys.executable, "trainers/convert_di03.py", "suite03", "--kit", str(kd), "--suite02", str(s02), "--parquet", pq,
                                    "--work", str(work), "--out", str(s03), "--report", str(out.parent / "_side/suite03_verify.json")],
                                   cwd=str(code), env=env, capture_output=True, text=True)
                if r.returncode:
                    raise RuntimeError(f"SUITE03_FAILED {r.stdout[-300:]} {r.stderr[-300:]}")
                r = subprocess.run([sys.executable, "trainers/convert_di03.py", "gsm8k", "--kit", str(kd), "--suite-dir", str(s03), "--out", str(out)],
                                   cwd=str(code), env=env, capture_output=True, text=True)
                if r.returncode:
                    raise RuntimeError(f"GSM8K03_FAILED {r.stdout[-300:]} {r.stderr[-300:]}")
                p = out / "30_gsm8k_v3/eval/30_gsm8k_v3.jsonl"
                rec.update(files=[{"path": str(p), "member": "30_gsm8k_v3/eval/30_gsm8k_v3.jsonl", "md5": file_md5(p), "kind": "jev"}],
                           gsm8k_parquet_sha256=file_md5(pq, "sha256"), suite03=json.loads((side_dir / "suite03_verify.json").read_text()).get("ok"))
            elif P["kind"] == "di_raw_rows":
                sys.path.insert(0, str(Path(a.repo) / "trainers"))
                import importlib.util
                spec = importlib.util.spec_from_file_location("cdi_v2", Path(a.repo) / "trainers/convert_decision_index.py")
                cdi = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(cdi)
                out.mkdir(parents=True, exist_ok=True)
                files, counts = [], collections.Counter()
                for mem in P["inputs"]["members"]:
                    src = side_dir / "di_v021" / mem
                    dst = out / (Path(mem).name.replace(".jsonl.gz", "") + ".jsonl")
                    stats = {"k_hist": collections.Counter(), "k52_questions": 0, "units": 0}
                    with gzip.open(src, "rt", encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as o:
                        for line in f:
                            row = json.loads(line)
                            try:
                                r2 = cdi.convert_row(row, None, stats, "text")
                                counts["converted"] += 1
                            except Exception:                          # noqa: BLE001 -- unknown catalog id etc.: state fingerprint only
                                e = row.get("_evaluation") or {}
                                r2 = {"id": str(e.get("run_id") or row.get("id")), "state": row.get("state"),
                                      "questions": row.get("questions") if isinstance(row.get("questions"), dict) else {},
                                      "source": f"decision_index/raw_{e.get('catalog_id')}", "meta": {"eval_only": True}}
                                counts["raw_fields_only"] += 1
                            o.write(json.dumps(r2, ensure_ascii=False) + "\n")
                    files.append({"path": str(dst), "member": mem, "md5": file_md5(dst), "kind": "jev", "source_md5": file_md5(src)})
                rec.update(files=files, counts=dict(counts))
            elif P["kind"] == "ext2_fetch":
                for rel, want5 in P["code_md5"].items():
                    if file_md5(Path(a.repo) / rel) != want5:
                        raise RuntimeError(f"CODE_MD5 {rel}")
                out.mkdir(parents=True, exist_ok=True)
                code = ("import argparse, json, sys, hashlib\nsys.path[:0]=[%r, %r, %r]\nimport build_jev_ext2 as x2\n"
                        "a=argparse.Namespace(cache=%r, hf_token=None)\nfam=sys.argv[1]\nn=0\n"
                        "with open(sys.argv[2],'w',encoding='utf-8') as o:\n"
                        "  for tag, st in x2.EVAL_FETCHERS[fam](a):\n"
                        "    if st is None: continue\n"
                        "    n+=1; o.write(json.dumps({'id': f'{tag}#{n}', 'source': 'ext2eval/'+tag, 'state': st}, ensure_ascii=False)+'\\n')\n"
                        "print('N', n)\n") % (a.repo, a.repo + "/analysis", a.repo + "/trainers", str(out / ".urlcache"))
                (out / "fetch.py").write_text(code)
                files, fails = [], {}
                for fam in P["families"]:
                    dst = out / f"{fam}.jsonl"
                    r = subprocess.run([sys.executable, str(out / "fetch.py"), fam, str(dst)], cwd=a.repo, env=env, capture_output=True, text=True)
                    if r.returncode or not dst.exists():
                        fails[fam] = (r.stderr or r.stdout)[-400:]
                        continue
                    files.append({"path": str(dst), "member": f"{fam}.jsonl", "md5": file_md5(dst), "kind": "fingerprint",
                                  "rows": int(r.stdout.split()[-1])})
                rec["files"] = files
                if fails:
                    rec["family_failures"] = fails
                    raise RuntimeError(f"fetch failed for {sorted(fails)}")
            elif P["kind"] == "jsonl_gz":                      # (v3) a gzip JSONL object (fullft devset rows: rendered prompts)
                key = next((k for k in P["s3_candidates"] if s3_has(k)), None)
                if key is None:
                    raise RuntimeError(f"no S3 key among {P['s3_candidates']}")
                out.mkdir(parents=True, exist_ok=True)
                gz = out / os.path.basename(key)
                subprocess.run(["aws", "s3", "cp", "--only-show-errors", S3 + key, str(gz)], check=True)
                m5 = file_md5(gz)
                if P.get("expected_md5") and m5 != P["expected_md5"]:
                    raise RuntimeError(f"OBJECT_MD5 {m5} != {P['expected_md5']}")
                dst = out / (os.path.basename(key).replace(".gz", ""))
                with gzip.open(gz, "rt", encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as o:
                    for line in f:
                        r = json.loads(line)
                        if r.get("manifest_header"):
                            continue
                        o.write(json.dumps(r, ensure_ascii=False) + "\n")
                gz.unlink()
                rec.update(s3_key=key, object_md5=m5, files=[{"path": str(dst), "member": os.path.basename(key), "md5": file_md5(dst),
                                                              "kind": P.get("file_kind", "jev")}])
            elif P["kind"] == "side_raw_rows":                 # (v3) raw DI suite rows produced by an earlier population's staging
                sys.path.insert(0, str(Path(a.repo) / "trainers"))
                import importlib.util
                spec = importlib.util.spec_from_file_location("cdi_v3", Path(a.repo) / "trainers/convert_decision_index.py")
                cdi = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(cdi)
                if file_md5(Path(a.repo) / "trainers/convert_decision_index.py") != P["convert_decision_index_md5"]:
                    raise RuntimeError("CODE_MD5 convert_decision_index.py")
                out.mkdir(parents=True, exist_ok=True)
                srcs = sorted(p for g in P["globs"] for p in (st / "_side").glob(g))
                if not srcs:
                    raise RuntimeError(f"no side files for {P['globs']} (stage {P.get('after')} first)")
                files, counts = [], collections.Counter()
                for src in srcs:
                    dst = out / (src.name.replace(".jsonl.gz", "").replace(".jsonl", "") + ".jsonl")
                    stats = {"k_hist": collections.Counter(), "k52_questions": 0, "units": 0}
                    op = gzip.open if str(src).endswith(".gz") else open
                    with op(src, "rt", encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as o:
                        for line in f:
                            if not line.strip():
                                continue
                            row = json.loads(line)
                            if not isinstance(row, dict) or ("state" not in row and "questions" not in row):
                                counts["not_a_row"] += 1
                                continue
                            try:
                                r2 = cdi.convert_row(row, None, stats, "text")
                                counts["converted"] += 1
                            except Exception:                          # noqa: BLE001 -- keep every visible field of the raw row
                                e = row.get("_evaluation") or {}
                                r2 = {"id": str(e.get("run_id") or row.get("id")), "state": row.get("state"),
                                      "questions": row.get("questions") if isinstance(row.get("questions"), dict) else {},
                                      "source": f"decision_index/raw_{e.get('catalog_id')}", "meta": {"eval_only": True}}
                                counts["raw_fields_only"] += 1
                            o.write(json.dumps(r2, ensure_ascii=False) + "\n")
                    files.append({"path": str(dst), "member": str(src.relative_to(st / "_side")), "md5": file_md5(dst), "kind": "jev",
                                  "source_md5": file_md5(src)})
                rec.update(files=files, counts=dict(counts))
            else:
                raise RuntimeError(f"unknown kind {P['kind']}")
            rec["status"] = "ok"
        except Exception as e:                                        # noqa: BLE001
            rec.update(status="failed", error=f"{type(e).__name__}: {str(e)[:600]}")
        rec["seconds"] = round(time.time() - t0, 1)
        man["populations"][name] = rec
        save()
        print(json.dumps({"population": name, "status": rec["status"], "files": len(rec.get("files", [])),
                          "error": rec.get("error", "")[:200], "s": rec["seconds"]}), flush=True)
    req_bad = [n for n, r in man["populations"].items() if r["required"] and r["status"] != "ok"]
    print(json.dumps({"stage_done": True, "required_not_ok": req_bad}), flush=True)




# =============================================================================================== build
def iter_records(path, kind):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _index_file(job):
    src, path, kind = job
    fam = collections.defaultdict(lambda: ([], []))          # family -> (keys, field ids)
    tdf = collections.Counter()                                # text-unit key -> protected records holding it
    docs, occ = {}, set()                                      # doc key -> normalised text ; {(doc key, field)}
    cov = collections.Counter()
    stem = Path(path).stem
    nrec = nunit = 0

    def add(f_, key, fid):
        fam[f_][0].append(key)
        fam[f_][1].append(fid)

    def add_text(units, fid, seen):
        for key, n in units.items():
            add("text", key, fid)
            seen.add(key)
        cov[f"{FIELDS[fid]}:text_units"] += len(units)

    def add_doc(text, fid):
        d = doc_of(text, fid)
        if d is None:
            return
        if d[0] not in docs:
            docs[d[0]] = d[1]
        occ.add((d[0], fid))
        cov[f"{FIELDS[fid]}:docs"] += 1

    for rec in iter_records(path, kind):
        nrec += 1
        seen = set()
        rid = rec.get("id") if kind != "rendered" else (rec.get("record_id") or rec.get("decision_uid"))
        if rid is not None and kind in ("jev", "rendered"):
            add("rid_any", h64("r:" + str(rid)), FID["record_id"])
            if rec.get("state") is None and kind == "jev":
                add("rid_nostate", h64("r:" + str(rid)), FID["record_id"])
        if kind == "jev":
            s, w = lineage_roots(rec, stem)
            for x in s:
                add("lineage", h64(x), FID["record_id"])
            for x in w:
                add("lineage_weak", h64(x), FID["record_id"])
        st = rec.get("state")
        if st is not None:
            k, nf, tx = state_keys(st)
            for f_, vals in k.items():
                for v in vals:
                    add(f_, v, FID["state"])
            add_text(tx, FID["state"], seen)
            cov["state:present_rows"] += 1
            cov["state:chars"] += len(nf)
            add_doc(nf, FID["state"])
        rf, unidx = record_fields(rec, kind)
        cov["attachments_unindexable"] += unidx
        present = set()
        for fid, text in rf:
            present.add(fid)
            cov[f"{FIELDS[fid]}:instances"] += 1
            cov[f"{FIELDS[fid]}:chars"] += len(text)
            add_text(text_units(text), fid, seen)
            add_doc(text, fid)
        qs = rec.get("questions") if isinstance(rec.get("questions"), dict) else {}
        for qid, q in qs.items():
            nunit += 1
            for fid, text in question_fields(qid, q):
                present.add(fid)
                cov[f"{FIELDS[fid]}:instances"] += 1
                cov[f"{FIELDS[fid]}:chars"] += len(text)
                add_text(text_units(text), fid, seen)
                add_doc(text, fid)
            if kind == "jev" and isinstance(q, dict) and st is not None:
                u = unit_keys(rec, q)
                if u:
                    for f_, v in u.items():
                        add(f_, v, FID["instructions"] if f_ in ("question_norm",) else FID["option_text"] if f_ == "options_norm" else FID["state"])
        for fid in present:
            cov[f"{FIELDS[fid]}:present_rows"] += 1
        tdf.update(seen)
    out = {}
    for f_, (ks, fs) in fam.items():
        a = np.zeros(len(ks), dtype=[("k", "<u8"), ("f", "i1")])
        a["k"] = np.array(ks, dtype=np.uint64)
        a["f"] = np.array(fs, dtype=np.int8)
        out[f_] = np.unique(a)
    dl = [(dk, shingles(t)) for dk, t in docs.items()]
    occ_a = np.array(sorted(occ), dtype=[("d", "<u8"), ("f", "i1")]) if occ else np.zeros(0, dtype=[("d", "<u8"), ("f", "i1")])
    tk = np.array(list(tdf.keys()), dtype=np.uint64)
    tc = np.array(list(tdf.values()), dtype=np.int64)
    cov["records"], cov["units"] = nrec, nunit
    return src, out, dl, occ_a, (tk, tc), cov


def build(a):
    assert os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0 (shingle hashes must be reproducible)"
    load_v1(a.v1_code)
    st = json.loads((Path(a.stage) / "stage_manifest.json").read_text())
    reg = json.loads(Path(a.registry).read_text())
    pops = [P["population"] for P in reg["populations"]]
    req = {P["population"]: P["required"] for P in reg["populations"]}
    srcs, jobs, opaque, coverage = [], [], [], {}
    for name in pops:
        r = st["populations"].get(name)
        if not r or r["status"] != "ok":
            coverage[name] = {"indexed": False, "required": req[name], "reason": (r or {}).get("error", "not staged")}
            continue
        coverage[name] = {"indexed": True, "required": req[name], "files": len(r.get("files", [])),
                          "substitute": r.get("substitute"), "s3_key": r.get("s3_key"), "tarball_md5": r.get("tarball_md5")}
        for f in r.get("files", []):
            sid = len(srcs)
            srcs.append({"population": name, "file": f.get("member") or os.path.basename(f["path"]), "md5": f["md5"], "kind": f["kind"]})
            if f["kind"] == "opaque_index":
                opaque.append((sid, f["path"]))
            else:
                if file_md5(f["path"]) != f["md5"]:
                    raise SystemExit(f"staged file changed: {f['path']}")
                jobs.append((sid, f["path"], "jev" if f["kind"] == "jev" else f["kind"]))
    fk, ff, fs = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    doc_id, sh_list, did_list = {}, [], []
    occ_d, occ_s, occ_f = [], [], []
    tdf_k, tdf_c = [], []
    covf = collections.defaultdict(collections.Counter)
    t0 = time.time()
    jobs.sort(key=lambda j: -os.path.getsize(j[1]))
    with mp.get_context("fork").Pool(a.procs) as pool:
        for n, (sid, out, dl, occ_a, tdf, cov) in enumerate(pool.imap_unordered(_index_file, jobs, chunksize=1)):
            pop = srcs[sid]["population"]
            covf[pop].update(cov)
            tdf_k.append(tdf[0])
            tdf_c.append(tdf[1])
            for f_, arr in out.items():
                fk[f_].append(arr["k"])
                ff[f_].append(arr["f"])
                fs[f_].append(np.full(len(arr), sid, np.int32))
            for dk, sh in dl:
                if dk in doc_id:
                    continue
                did = len(doc_id)
                doc_id[dk] = did
                if len(sh):
                    sh_list.append(sh)
                    did_list.append(np.full(len(sh), did, np.int32))
            if len(occ_a):
                occ_d.append(np.fromiter((doc_id[int(x)] for x in occ_a["d"]), dtype=np.int32, count=len(occ_a)))
                occ_f.append(occ_a["f"])
                occ_s.append(np.full(len(occ_a), sid, np.int32))
            if n % 50 == 0:
                print(f"build: {n + 1}/{len(jobs)} files, {len(doc_id)} docs, {time.time() - t0:.0f}s", flush=True)
    for sid, path in opaque:                             # sealed exclusion indexes (ss_lib.load_exclusions semantics; hashes / ids only)
        ix = json.load(gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8"))
        add = {"sealed_id": ([h64("r:" + str(v)) for v in list(ix.get("ids") or []) + list(ix.get("unassigned_eval_only_ids") or [])], FID["record_id"]),
               "sha1": ([hexkey(str(v)) for v in ix.get("state_sha1") or []], FID["state"]),
               "sealed_cluster": ([h64("c:" + str(v)) for v in ix.get("clusters") or []], FID["record_id"]),
               "sealed_entity": ([h64("e:" + str(v)) for v in ix.get("entity_keys") or []], FID["record_id"])}
        if not (add["sealed_id"][0] or add["sha1"][0]):
            raise SystemExit(f"opaque index {path} has no ids / state sha1s")
        pop = srcs[sid]["population"]
        coverage[pop]["opaque_counts"] = {k: len(v[0]) for k, v in add.items()}
        for f_, (v, fid) in add.items():
            if v:
                arr = np.unique(np.array(v, dtype=np.uint64))
                fk[f_].append(arr)
                ff[f_].append(np.full(len(arr), fid, np.int8))
                fs[f_].append(np.full(len(arr), sid, np.int32))
    arrays = {}
    for f_ in fk:
        K, F, Sx = np.concatenate(fk[f_]), np.concatenate(ff[f_]), np.concatenate(fs[f_])
        o = np.lexsort((F, Sx, K))
        K, F, Sx = K[o], F[o], Sx[o]
        keep = np.ones(len(K), bool)
        keep[1:] = (K[1:] != K[:-1]) | (Sx[1:] != Sx[:-1]) | (F[1:] != F[:-1])
        arrays[f"k_{f_}"], arrays[f"s_{f_}"], arrays[f"f_{f_}"] = K[keep], Sx[keep], F[keep]
    del fk, ff, fs
    if "k_text" in arrays:                               # text units in > DF_MAX protected records are boilerplate: weak only
        Kt, Ct = np.concatenate(tdf_k), np.concatenate(tdf_c)
        o = np.argsort(Kt, kind="stable")
        Kt, Ct = Kt[o], Ct[o]
        ut, it = np.unique(Kt, return_index=True)
        dft = np.add.reduceat(Ct, it)
        boiler = ut[dft > DF_MAX]
        isb = np.isin(arrays["k_text"], boiler)
        for p_ in ("k", "s", "f"):
            arrays[f"{p_}_text_boiler"] = arrays[f"{p_}_text"][isb]
            arrays[f"{p_}_text"] = arrays[f"{p_}_text"][~isb]
        del Kt, Ct, ut, it, dft
    ndoc = len(doc_id)
    del doc_id
    H = np.concatenate(sh_list) if sh_list else np.zeros(0, np.int64)
    Dd = np.concatenate(did_list) if did_list else np.zeros(0, np.int32)
    del sh_list, did_list
    o = np.argsort(H, kind="stable")
    H, Dd = H[o], Dd[o]
    del o
    uniq, cnt = np.unique(H, return_counts=True)
    stop_keys = uniq[cnt > DF_MAX]
    nonstop = np.repeat(cnt <= DF_MAX, cnt)
    del uniq, cnt
    doc_total = np.bincount(Dd, minlength=ndoc).astype(np.int32)
    doc_nonstop = np.bincount(Dd[nonstop], minlength=ndoc).astype(np.int32)
    H, Dd = H[nonstop], Dd[nonstop]
    del nonstop
    OD = np.concatenate(occ_d) if occ_d else np.zeros(0, np.int32)
    OS = np.concatenate(occ_s) if occ_s else np.zeros(0, np.int32)
    OF = np.concatenate(occ_f) if occ_f else np.zeros(0, np.int8)
    o = np.lexsort((OF, OS, OD))
    OD, OS, OF = OD[o], OS[o], OF[o]
    occ_ptr = np.zeros(ndoc + 1, np.int64)
    np.cumsum(np.bincount(OD, minlength=ndoc), out=occ_ptr[1:])
    arrays.update(sh_H=H, sh_D=Dd, sh_stop=stop_keys, doc_total=doc_total, doc_nonstop=doc_nonstop, occ_ptr=occ_ptr, occ_src=OS, occ_fid=OF)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    npz = out / "protected_index_v3.npz"
    if npz.exists():
        raise SystemExit(f"{npz} exists: indexes are immutable; use a new --out")
    np.savez(out / "protected_index_v3.partial.npz", **arrays)
    os.replace(out / "protected_index_v3.partial.npz", npz)
    by_field = {}
    for pop, c in covf.items():
        by_field[pop] = {"records": c["records"], "units": c["units"], "attachments_unindexable": c["attachments_unindexable"],
                         "fields": {f: {m: c[f"{f}:{m}"] for m in ("present_rows", "instances", "chars", "text_units", "docs") if c[f"{f}:{m}"]}
                                    for f in FIELDS if any(c[f"{f}:{m}"] for m in ("present_rows", "instances", "text_units", "docs"))}}
        coverage[pop]["records"] = c["records"]
        coverage[pop]["units"] = c["units"]
    for pop, c in coverage.items():
        if c.get("opaque_counts"):
            by_field[pop] = {"opaque_index": c["opaque_counts"], "fields": {"state": {"sha1_keys": c["opaque_counts"].get("sha1", 0)},
                                                                            "record_id": {"keys": sum(v for k, v in c["opaque_counts"].items() if k != "sha1")}}}
    digests = {}
    for pop in pops:
        sids = {i for i, s in enumerate(srcs) if s["population"] == pop}
        if not sids:
            continue
        h = hashlib.sha256()
        for f_ in sorted(k[2:] for k in arrays if k.startswith("k_")):
            m = np.isin(arrays[f"s_{f_}"], np.array(sorted(sids), np.int32))
            h.update(f_.encode())
            h.update(np.unique(arrays[f"k_{f_}"][m]).tobytes())
        dm = np.isin(OS, np.array(sorted(sids), np.int32))
        h.update(b"docs")
        h.update(np.unique(OD[dm]).astype(np.int64).tobytes())
        digests[pop] = h.hexdigest()
    req_missing = [n for n, c in coverage.items() if c["required"] and not c["indexed"]]
    man = {"schema": SCHEMA, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": sys.version.split()[0],
           "pythonhashseed": os.environ.get("PYTHONHASHSEED"), "code_sha256": file_md5(__file__, "sha256"), "keyfn_sha256": keyfn_sha256(),
           "v1_decontam_sha256": file_md5(Path(a.v1_code) / "decontam.py", "sha256"), "registry_sha256": file_md5(a.registry, "sha256"),
           "stage_manifest_sha256": file_md5(Path(a.stage) / "stage_manifest.json", "sha256"), "populations": pops, "fields": FIELDS,
           "sources": srcs,
           "params": {"shingle_words": SHINGLE_WORDS, "df_max": DF_MAX, "near_jaccard": NEAR_J, "containment": CONTAIN,
                      "containment_min_shingles": CONTAIN_MIN, "shingle3": SH3, "text_min_words": TEXT_MIN, "text_long_words": TEXT_LONG,
                      "paragraph_split_above_words": PARA_SPLIT_WORDS, "template_df": DF_TPL, "candidate_qid_regex": CAND_QID.pattern,
                      "shingled_content_fields": sorted(FIELDS[i] for i in CONTENT_FIDS)},
           "strong_kinds": STRONG, "weak_kinds": WEAK, "unique_docs": ndoc, "doc_occurrences": int(len(OD)),
           "shingle_entries_nonstop": int(len(H)), "stop_shingles": int(len(stop_keys)),
           "family_sizes": {k[2:]: int(len(v)) for k, v in arrays.items() if k.startswith("k_")}, "coverage": coverage,
           "coverage_by_field": by_field, "protected_panel_digest_sha256": digests, "required_not_indexed": req_missing,
           "sealed_not_indexable": reg.get("sealed_not_indexable"),
           "coverage_statement": "per population and per field (coverage_by_field); there is deliberately no single coverage_complete flag",
           "npz_sha256": file_md5(npz, "sha256"), "npz_md5": file_md5(npz)}
    (out / "protected_index_v3_manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: man[k] for k in ("unique_docs", "doc_occurrences", "shingle_entries_nonstop", "stop_shingles",
                                           "required_not_indexed", "npz_md5")}), flush=True)


# =============================================================================================== scan
IDX = {}


def _lookup(fam, keys):
    """-> [(key, [(src, field), ...])] for keys present in family fam."""
    K = IDX.get("k_" + fam)
    if K is None or not len(keys):
        return []
    S_, F_ = IDX["s_" + fam], IDX["f_" + fam]
    q = np.array(sorted(set(keys)), dtype=np.uint64)
    lo, hi = np.searchsorted(K, q, "left"), np.searchsorted(K, q, "right")
    return [(int(q[i]), list(zip(S_[lo[i]:hi[i]].tolist(), F_[lo[i]:hi[i]].tolist()))) for i in np.nonzero(hi > lo)[0]]


def _occ(did, cap=6):
    p0, p1 = IDX["occ_ptr"][did], IDX["occ_ptr"][did + 1]
    return list(zip(IDX["occ_src"][p0:p0 + cap].tolist(), IDX["occ_fid"][p0:p0 + cap].tolist())), int(p1 - p0)


def _near(sh):
    if not len(sh) or not len(IDX["sh_H"]):
        return None
    stop = IDX["sh_stop"]
    if len(stop):
        pos = np.searchsorted(stop, sh)
        pos[pos >= len(stop)] = len(stop) - 1
        sh = sh[stop[pos] != sh]
    n = len(sh)
    if n == 0:
        return None
    H, Dd = IDX["sh_H"], IDX["sh_D"]
    lo, hi = np.searchsorted(H, sh, "left"), np.searchsorted(H, sh, "right")
    nz = np.nonzero(hi > lo)[0]
    if len(nz) == 0:
        return {"n": n, "shared": 0}
    ids = np.concatenate([Dd[lo[i]:hi[i]] for i in nz])
    u, c = np.unique(ids, return_counts=True)
    ne = IDX["doc_nonstop"][u]
    jac = c / np.maximum(1, n + ne - c)
    ce = c / np.maximum(1, ne)
    cc = c / n
    b = int(np.argmax(jac))
    res = {"n": n, "shared": int(c.max()), "j": round(float(jac.max()), 4), "j_doc": int(u[b]), "ce": 0.0, "cc": 0.0}
    m1 = ne >= CONTAIN_MIN
    if m1.any():
        bi = int(np.argmax(np.where(m1, ce, -1)))
        res["ce"], res["ce_doc"] = round(float(ce[bi]), 4), int(u[bi])
    if n >= CONTAIN_MIN:
        bc = int(np.argmax(cc))
        res["cc"], res["cc_doc"] = round(float(cc[bc]), 4), int(u[bc])
    res["sh3_doc"] = int(u[int(np.argmax(c))])
    return res


def _record_matches(rec, stem, kind="jev"):
    """-> (record-level match list, {qid: unit-level match list}, text_long keys by scope, near docs by scope, near detail).
    A match is (kind, candidate field id, protected src id, protected field id)."""
    R = []
    tk = {"_rec": set()}
    nk = {"_rec": []}
    nd = {}

    def text_hits(units, cfid, M, tks):
        for key, ps in _lookup("text", list(units)):
            kind_ = "text_long" if units[key] >= TEXT_LONG else "text_short"
            M.extend((kind_, cfid, s, f) for s, f in ps)
            if kind_ == "text_long":
                tks.add(key)
        for key, ps in _lookup("text_boiler", list(units)):
            M.extend(("text_boiler", cfid, s, f) for s, f in ps)

    def doc_hits(text, cfid, M, nks, scope):
        d = doc_of(text, cfid)
        if d is None:
            return
        near = _near(shingles(d[1]))
        if not near or not near.get("shared", 0):
            return
        for kind_, ok, dkey in (("near_dup", near["j"] >= NEAR_J, "j_doc"), ("eval_in_cand", near.get("ce", 0) >= CONTAIN, "ce_doc"),
                                ("cand_in_eval", near.get("cc", 0) >= CONTAIN, "cc_doc"), ("shingle3", near["shared"] >= SH3, "sh3_doc")):
            if not ok:
                continue
            occ, _ = _occ(near[dkey])
            M.extend((kind_, cfid, s, f) for s, f in occ)
            if kind_ != "shingle3":
                nks.append((format(d[0], "016x"), cfid, kind_))
        if near["j"] >= NEAR_J or near.get("ce", 0) >= CONTAIN or near.get("cc", 0) >= CONTAIN:
            nd.setdefault(scope, []).append({"f": FIELDS[cfid], **{k_: near[k_] for k_ in ("n", "shared", "j", "ce", "cc") if k_ in near}})

    st = rec.get("state")
    if st is not None:
        k, nf, tx = state_keys(st)
        for f_ in ("state_exact", "state_nfkc", "sha1", "state_flat"):
            for _, ps in _lookup(f_, list(k[f_])):
                R.extend((f_, FID["state"], s, f) for s, f in ps)
        text_hits(tx, FID["state"], R, tk["_rec"])
        doc_hits(nf, FID["state"], R, nk["_rec"], "_rec")
    rf, _ = record_fields(rec, kind)
    for fid, text in rf:
        text_hits(text_units(text), fid, R, tk["_rec"])
        doc_hits(text, fid, R, nk["_rec"], "_rec")
    rid = str(rec.get("id"))
    for fam_, kind_ in (("rid_nostate", "rid_nostate"), ("rid_any", "rid_any"), ("sealed_id", "sealed_id")):
        for _, ps in _lookup(fam_, [h64("r:" + rid)]):
            R.extend((kind_, FID["record_id"], s, f) for s, f in ps)
    m = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
    src = str(rec.get("source") or stem)
    ck = [h64("c:" + f"{s_}:{m.get('group') or rec.get('id')}") for s_ in {src, src.split('/')[0], src.replace('/', '__'), stem}]
    for _, ps in _lookup("sealed_cluster", ck):
        R.extend(("sealed_cluster", FID["record_id"], s, f) for s, f in ps)
    if ENTITY is not None and "k_sealed_entity" in IDX:
        try:
            ek = [h64("e:" + str(x)) for x in ENTITY(rec)]
        except Exception:                                 # noqa: BLE001 -- as ss_lib.hits
            ek = []
        for _, ps in _lookup("sealed_entity", ek):
            R.extend(("sealed_entity", FID["record_id"], s, f) for s, f in ps)
    s_, w_ = lineage_roots(rec, stem)
    for _, ps in _lookup("lineage", [h64(x) for x in s_]):
        R.extend(("lineage", FID["record_id"], s, f) for s, f in ps)
    for _, ps in _lookup("lineage", [h64(x) for x in w_]) + _lookup("lineage_weak", [h64(x) for x in s_ | w_]):
        R.extend(("lineage_weak", FID["record_id"], s, f) for s, f in ps)
    U = {}
    qs = rec.get("questions") if isinstance(rec.get("questions"), dict) else {}
    for qid, q in qs.items():
        M, tks, nks = [], set(), []
        for fid, text in question_fields(qid, q):
            text_hits(text_units(text), fid, M, tks)
            doc_hits(text, fid, M, nks, qid)
        u = unit_keys(rec, q) if isinstance(q, dict) and st is not None else None
        if u:
            for f_ in ("joint_exact", "joint_norm"):
                for _, ps in _lookup(f_, [u[f_]]):
                    M.extend((f_, FID["state"], s, f) for s, f in ps)
            for f_ in ("question_options", "question_norm", "options_norm", "template"):
                if f_ == "options_norm" and q.get("type") in ("noul", "score"):
                    continue                      # Yes/No and level menus are shared by construction
                cf = FID["instructions"] if f_ == "question_norm" else FID["option_text"] if f_ == "options_norm" else FID["state"]
                for _, ps in _lookup(f_, [u[f_]]):
                    M.extend((f_, cf, s, f) for s, f in ps)
        U[qid] = (M, tks, nks)
    return R, U, tk["_rec"], nk["_rec"], nd


def _scan_chunk(job):
    pool_tag, member, lines = job
    stem = Path(member).stem
    rows, stats = [], collections.Counter()
    for line_no, line in lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        R, U, rtk, rnk, nd = _record_matches(rec, stem)
        if not U:
            stats["records_without_questions"] += 1
        for qid, (M, tks, nks) in U.items():
            stats["units"] += 1
            allm = set(R) | set(M)
            us = {m_[0] for m_ in allm if m_[0] in STRONG}
            uw = {m_[0] for m_ in allm if m_[0] in WEAK}
            if not (us or uw):
                continue
            ms = sorted(allm, key=lambda x: (x[0] not in STRONG, x))
            row = {"uid": f"{pool_tag}|{member}|{line_no}|{qid}", "s": sorted(us), "w": sorted(uw), "m": ms[:200]}
            if len(ms) > 200:
                row["m_trunc"] = len(ms)
            t_ = sorted(rtk | tks)
            if t_:
                row["tk"] = [format(x, "016x") for x in t_]
            n_ = rnk + nks
            if n_:
                row["nk"] = [list(x) for x in n_]
            nd_ = nd.get("_rec", []) + nd.get(qid, [])
            if nd_:
                row["nd"] = nd_[:6]
            rows.append(row)
            stats["strong_units" if us else "weak_only_units"] += 1
            for k_ in us:
                stats["strong:" + k_] += 1
            for k_ in uw:
                stats["weak:" + k_] += 1
    return member, rows, stats


def member_of(name):
    parts = clean(name).split("/")
    for i, x in enumerate(parts[:-1]):
        if x in ("train", "val", "validation", "dev", "test", "confirmation", "eval"):
            return "/".join(parts[i:])
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def iter_members(spec, splits, chunk=2000):
    """(member, [(line_no, line)] chunks) of every *.jsonl under SPEC whose split dir is in splits; confirmation never opened."""
    def ok(member):
        sp = member.split("/")[0]
        return member.endswith(".jsonl") and sp in splits and sp != "confirmation" and not os.path.basename(member).startswith("._")

    def chunks(member, f):
        buf = []
        for i, line in enumerate(f, 1):
            buf.append((i, line))
            if len(buf) >= chunk:
                yield member, buf
                buf = []
        if buf:
            yield member, buf
    if spec.startswith("s3://") or spec.endswith(".tgz") or spec.endswith(".tar.gz"):
        cmd = ["aws", "s3", "cp", "--only-show-errors", spec, "-"] if spec.startswith("s3://") else ["cat", spec]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        hr = HashReader(p.stdout)
        with tarfile.open(fileobj=hr, mode="r|gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                mem = member_of(m.name)
                if not ok(mem):
                    continue
                f = io.StringIO(tf.extractfile(m).read().decode("utf-8"))      # stream members are not seekable (py3.12)
                yield from chunks(mem, f)
        hr.drain()
        p.wait()
        iter_members.last_md5 = hr.md5.hexdigest()
    else:
        root = Path(spec)
        for p in sorted(root.rglob("*.jsonl")):
            mem = str(p.relative_to(root))
            if not ok(mem):
                continue
            with open(p, encoding="utf-8") as f:
                yield from chunks(mem, f)
        iter_members.last_md5 = None


def load_index(a):
    global IDX
    load_v1(a.v1_code)
    if getattr(a, "repo", ""):
        load_entity(a.repo)
    man = json.loads((Path(a.index) / "protected_index_v3_manifest.json").read_text())
    npz = Path(a.index) / "protected_index_v3.npz"
    if file_md5(npz) != man["npz_md5"] or man["python"] != sys.version.split()[0]:
        raise SystemExit("index integrity / python version mismatch")
    if man["keyfn_sha256"] != keyfn_sha256():
        raise SystemExit("scanner key functions differ from the index builder's (rebuild the index)")
    z = np.load(npz)
    IDX = {k: z[k] for k in z.files}
    return man


def scan(a):
    assert os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0"
    man = load_index(a)
    srcs = man["sources"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hits_p = out / f"hits_{a.pool}.jsonl.gz"
    if hits_p.exists():
        raise SystemExit(f"{hits_p} exists (immutable)")
    splits = set(a.splits.split(","))
    tot = collections.Counter()
    by_member = collections.defaultdict(collections.Counter)
    by_pop, by_file, by_field = collections.Counter(), collections.Counter(), collections.Counter()
    t0 = time.time()
    with gzip.open(out / f"hits_{a.pool}.partial.jsonl.gz", "wt", encoding="utf-8") as h, mp.get_context("fork").Pool(a.procs) as pool:
        jobs = ((a.pool, mem, lines) for mem, lines in iter_members(a.input, splits, a.chunk))
        for n, (member, rows, stats) in enumerate(bounded_imap(pool, _scan_chunk, jobs, 3 * a.procs)):
            tot.update(stats)
            by_member[member].update(stats)
            for r in rows:
                sp, wp = set(), set()
                ms = []
                seen_f = set()
                files_u, kf_u = set(), set()
                for kind_, cf, s, pf in r["m"]:
                    pop, fil = srcs[s]["population"], srcs[s]["file"]
                    strong = kind_ in STRONG
                    (sp if strong else wp).add(pop)
                    ms.append([kind_, FIELDS[cf], pop, fil, FIELDS[pf]])
                    if strong:
                        files_u.add(f"{pop}|{fil}")
                        kf_u.add(f"{kind_}|{FIELDS[cf]}->{FIELDS[pf]}")
                by_file.update(files_u)
                by_field.update(kf_u)
                for p in sp:
                    by_pop["strong:" + p] += 1
                for p in wp - sp:
                    by_pop["weak_only:" + p] += 1
                r["m"], r["sp"], r["wp"] = ms, sorted(sp), sorted(wp)
                h.write(json.dumps(r, separators=(",", ":")) + "\n")
            if n % 50 == 0:
                print(f"scan {a.pool}: chunk {n}, units {tot['units']}, strong {tot['strong_units']}, {time.time() - t0:.0f}s", flush=True)
    os.replace(out / f"hits_{a.pool}.partial.jsonl.gz", hits_p)
    summ = {"schema": SCHEMA, "pool": a.pool, "input": a.input, "input_md5": iter_members.last_md5, "splits": sorted(splits),
            "index_npz_md5": man["npz_md5"], "totals": dict(tot), "by_population": dict(by_pop),
            "strong_units_by_protected_file_raw": dict(by_file), "strong_units_by_kind_and_field_raw": dict(by_field),
            "by_member": {k: dict(v) for k, v in sorted(by_member.items())},
            "hits_file": hits_p.name, "hits_md5": file_md5(hits_p), "seconds": round(time.time() - t0, 1),
            "policy": "units with any STRONG kind are withheld (reversible ledger); WEAK-only units stay eligible and are listed for "
                      "adjudication; absence from the hits file = no indexed identity found. Run `adjudicate` (rules T1 / T2) next."}
    (out / f"summary_{a.pool}.json").write_text(json.dumps(summ, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: summ[k] for k in ("pool", "totals", "hits_md5", "seconds")}), flush=True)


# =============================================================================================== adjudicate
NEAR_KINDS = ("near_dup", "eval_in_cand", "cand_in_eval")
NEED = {}


def _df_chunk(job):
    _, member, lines = job
    tk, dk = NEED["tk"], NEED["dk"]
    c = collections.Counter()
    for _, line in lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        keys = set()
        st = rec.get("state")
        texts = []
        if st is not None:
            _, nf, tx = state_keys(st)
            keys.update(k for k in tx if k in tk)
            texts.append(nf)
        rf, _ = record_fields(rec, "jev")
        texts += [t for _, t in rf]
        qs = rec.get("questions") if isinstance(rec.get("questions"), dict) else {}
        for qid, q in qs.items():
            texts += [t for _, t in question_fields(qid, q)]
        for t in texts:
            keys.update(k for k in text_units(t) if k in tk)
            d = doc_of(t)
            if d is not None and d[0] in dk:
                keys.add(d[0])
        c.update(keys)
    return c


def adjudicate(a):
    assert os.environ.get("PYTHONHASHSEED") == "0"
    load_v1(a.v1_code)
    hp = Path(a.scan_dir) / f"hits_{a.pool}.jsonl.gz"
    dst = Path(a.scan_dir) / f"hits_{a.pool}.adj.jsonl.gz"
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    tk, dk = set(), set()
    for r in (json.loads(l) for l in gzip.open(hp, "rt")):
        if "text_long" in r["s"]:
            tk.update(int(x, 16) for x in r.get("tk", []))
        if any(k in r["s"] for k in NEAR_KINDS):
            dk.update(int(x[0], 16) for x in r.get("nk", []) if x[1] in TEMPLATE_DOC_FIELDS)
    NEED["tk"], NEED["dk"] = tk, dk
    df = collections.Counter()
    with mp.get_context("fork").Pool(a.procs) as pool:
        jobs = ((a.pool, mem, lines) for mem, lines in iter_members(a.input, {"train"}, a.chunk))
        for c in bounded_imap(pool, _df_chunk, jobs, 3 * a.procs):
            df.update(c)
    in_md5 = iter_members.last_md5
    moved = collections.Counter()
    tot = collections.Counter()
    with gzip.open(str(dst) + ".partial", "wt") as g:
        for r in (json.loads(l) for l in gzip.open(hp, "rt")):
            s, w = set(r["s"]), set(r["w"])
            drop = set()
            if "text_long" in s:
                ks = [int(x, 16) for x in r.get("tk", [])]
                dmin = min((df[k] for k in ks), default=0)
                r.setdefault("adj", {})["text_long_min_candidate_df"] = dmin
                if ks and dmin >= DF_TPL:
                    drop.add("text_long")
                    w.add("text_template")
                    moved["T1_text_long_to_text_template"] += 1
            nks = [x for x in r.get("nk", []) if x[2] in NEAR_KINDS]
            for kind_ in NEAR_KINDS:
                if kind_ not in s:
                    continue
                docs = [x for x in nks if x[2] == kind_]
                tpl = [x for x in docs if x[1] in TEMPLATE_DOC_FIELDS and df[int(x[0], 16)] >= DF_TPL]
                if docs and len(tpl) == len(docs):
                    drop.add(kind_)
                    w.add("doc_template")
                    moved[f"T2_{kind_}_to_doc_template"] += 1
            if drop:
                s -= drop
                r["m"] = [m_ for m_ in r["m"] if m_[0] not in drop] + [["text_template" if k_ == "text_long" else "doc_template", "-", "-", "-", k_]
                                                                       for k_ in sorted(drop)]
                r["sp"] = sorted({m_[2] for m_ in r["m"] if m_[0] in STRONG}) if not r.get("m_trunc") else r["sp"]
                r["adj"] = dict(r.get("adj", {}), demoted=sorted(drop))
            r["s"], r["w"] = sorted(s), sorted(w)
            tot["strong_units" if s else "weak_only_units"] += 1
            for k_ in s:
                tot["strong:" + k_] += 1
            g.write(json.dumps(r, separators=(",", ":")) + "\n")
    os.replace(str(dst) + ".partial", dst)
    summ = {"pool": a.pool, "rules": {"T1": "text_long with every matched unit in >= 25 candidate train records -> weak text_template",
                                      "T2": "near_dup / eval_in_cand / cand_in_eval whose every candidate doc is an instructions / option / "
                                            "noul-def / question_other text in >= 25 candidate train records -> weak doc_template"},
            "df_tpl": DF_TPL, "input_md5": in_md5, "keys_checked": {"text": len(tk), "doc": len(dk)}, "moved": dict(moved),
            "after": dict(tot), "file": dst.name, "md5": file_md5(dst)}
    (Path(a.scan_dir) / f"adjudication_{a.pool}.json").write_text(json.dumps(summ, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summ), flush=True)


# =============================================================================================== regress
def _units_of_file(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if line.strip():
                out.append((ln, json.loads(line)))
    return out


def regress_errata(a):
    """E1 / E2 from finished scans. --e1-hits: adjudicated hits of emo_pool_v1; --e2-hits + --e2-cand: hits of the jev_data_ref TRAIN
    gsm8k pool and the candidate file itself; --stage: the staged protected files (E2 reverse row count on the BRIGHT file)."""
    res = {}
    # E1: which protected FILES the jev_data gsm8k train units hit
    files = collections.Counter()
    kinds_by_file = collections.defaultdict(collections.Counter)
    n_units = 0
    for r in (json.loads(l) for l in gzip.open(a.e1_hits, "rt")):
        if "/jev_data__gsm8k.jsonl" not in r["uid"] or not r["s"]:
            continue
        n_units += 1
        fs = set()
        for kind_, cf, pop, fil, pf in r["m"]:
            if kind_ in STRONG:
                fs.add(f"{pop}|{fil}")
                kinds_by_file[f"{pop}|{fil}"][kind_] += 1
        files.update(fs)
    rb = [k for k in files if "06_routerbench" in k]
    g30 = {k: v for k, v in files.items() if "30_gsm8k" in k}
    res["E1"] = {"jev_data_gsm8k_strong_units": n_units, "units_by_protected_file": dict(files.most_common(30)),
                 "routerbench_units": sum(files[k] for k in rb), "gsm8k30_units": g30,
                 "gsm8k30_kinds": {k: dict(kinds_by_file[k]) for k in g30},
                 "pass": bool(rb) and sum(files[k] for k in rb) >= 1140 and not any(
                     kk in ("state_exact", "state_nfkc", "sha1", "state_flat", "joint_exact", "joint_norm", "text_long", "near_dup",
                            "eval_in_cand", "cand_in_eval") for k in g30 for kk in kinds_by_file[k])}
    # E2 forward: jev_data_ref TRAIN gsm8k units with a strong hit on the BRIGHT file's candidate-document field
    cand = {ln: rec for ln, rec in _units_of_file(a.e2_cand)}
    qtext = {ln: cnorm(rec["state"]) for ln, rec in cand.items()}
    fwd_units, fwd_q = 0, set()
    for r in (json.loads(l) for l in gzip.open(a.e2_hits, "rt")):
        ok = [m for m in r["m"] if m[0] in STRONG and "36_bright" in m[3] and m[4] == "candidate_document"]
        if ok:
            fwd_units += 1
            fwd_q.add(qtext[int(r["uid"].split("|")[2])])
    # E2 reverse: rows of the staged 0.2.1 BRIGHT file whose candidate documents contain a jev_data train gsm8k question
    qs = {}
    for ln, t in qtext.items():
        sh = set(shingles(t).tolist())
        if len(sh) >= CONTAIN_MIN:
            qs.setdefault(t, sh)
    inv = collections.defaultdict(set)
    for t, sh in qs.items():
        for x in sh:
            inv[x].add(t)
    bright = next(Path(a.stage).glob("decision_index_v021/di_v021/36_bright/eval/36_bright.jsonl"))
    rows_hit, q_hit = set(), set()
    for ln, rec in _units_of_file(bright):
        for qid, q in (rec.get("questions") or {}).items():
            for fid, text in question_fields(qid, q):
                if fid != FID["candidate_document"]:
                    continue
                sh = set(shingles(cnorm(text)).tolist())
                cnt = collections.Counter(t for x in sh for t in inv.get(x, ()))
                for t, c in cnt.items():
                    if c / len(qs[t]) >= CONTAIN:
                        rows_hit.add(ln)
                        q_hit.add(t)
    res["E2"] = {"forward_units": fwd_units, "forward_distinct_questions": len(fwd_q), "reverse_bright_rows": len(rows_hit),
                 "reverse_distinct_questions": len(q_hit), "expected": {"questions": 11, "rows": 8},
                 "pass": len(q_hit) == 11 and len(rows_hit) == 8 and fwd_q == q_hit}
    return res


def regress_fixtures(a):
    """build + scan + adjudicate a synthetic protected registry against a synthetic candidate pool; every expected identity must fire,
    the unrelated record must not, the confirmation split must not be opened, and T1 must demote the template text."""
    import shutil
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="v3fx_"))
    try:
        words = lambda tag, n: " ".join(f"{tag}{i}" for i in range(n))                     # noqa: E731
        S1, DOC, Q, OPT, LONG, SEAL, TPL = (words("alpha", 30), words("filler", 40) + " " + words("gsmq", 30) + " " + words("tail", 20),
                                            words("gsmq", 30), words("option", 14), words("lorem", 200), words("sealed", 25), words("tplins", 15))
        SLOT = words("slotq", 25) + " {} " + words("slotr", 25)
        ev = d / "stage/fx_eval"
        ev.mkdir(parents=True)
        recs = [{"id": "e1", "state": S1, "questions": {"q": {"type": "noul", "instructions": TPL, "criteria": {"yes": "y", "no": "n"}}}},
                {"id": "e2", "state": "query text here", "questions": {"x|doc_0": {"type": "noul", "instructions": DOC, "criteria": {"yes": "y", "no": "n"}}}},
                {"id": "e3", "state": "short", "questions": {"q": {"type": "choice", "instructions": "pick", "criteria": {"A": OPT, "B": "b"}}}},
                {"id": "e4", "state": LONG, "questions": {"q": {"type": "noul", "instructions": "i", "criteria": {"yes": "y", "no": "n"}}}},
                {"id": "e5", "state": words("evst", 20), "questions": {"q": {"type": "noul", "instructions": SLOT.format("acmecorp"),
                                                                         "criteria": {"yes": "acmecorp " + words("ydef", 14), "no": words("ndef", 9) + " acmecorp"}}}}]
        (ev / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        sx = d / "stage/fx_sealed"
        sx.mkdir(parents=True)
        with gzip.open(sx / "ix.json.gz", "wt") as f:
            json.dump({"ids": ["sealed-id-1"], "state_sha1": [hashlib.sha1(cnorm(SEAL).encode()).hexdigest()]}, f)
        man = {"populations": {"fx_eval": {"status": "ok", "required": True, "kind": "tarball", "files": [
            {"path": str(ev / "test.jsonl"), "member": "fx/test.jsonl", "md5": file_md5(ev / "test.jsonl"), "kind": "jev"}]},
            "fx_sealed": {"status": "ok", "required": True, "kind": "opaque_index", "files": [
                {"path": str(sx / "ix.json.gz"), "md5": file_md5(sx / "ix.json.gz"), "kind": "opaque_index"}]}}}
        (d / "stage/stage_manifest.json").write_text(json.dumps(man))
        (d / "reg.json").write_text(json.dumps({"populations": [{"population": "fx_eval", "required": True, "kind": "tarball"},
                                                               {"population": "fx_sealed", "required": True, "kind": "opaque_index"}]}))
        cp = d / "cand"
        (cp / "train").mkdir(parents=True)
        (cp / "confirmation").mkdir(parents=True)
        LONG2 = LONG.replace("lorem7 ", "ipsum7 ")
        c = [{"id": "c1", "state": S1, "questions": {"q": {"type": "noul", "instructions": "x", "criteria": {"yes": "y", "no": "n"}}}},
             {"id": "c2", "state": Q, "questions": {"q": {"type": "noul", "instructions": "x", "criteria": {"yes": "y", "no": "n"}}}},
             {"id": "c3", "state": "other state words", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"A": OPT, "B": "zz"}}}},
             {"id": "c4", "state": LONG2, "questions": {"q": {"type": "noul", "instructions": "x", "criteria": {"yes": "y", "no": "n"}}}},
             {"id": "c5", "state": SEAL, "questions": {"q": {"type": "noul", "instructions": "x", "criteria": {"yes": "y", "no": "n"}}}},
             {"id": "c6", "state": words("unrelated", 40), "questions": {"q": {"type": "noul", "instructions": "x", "criteria": {"yes": "y", "no": "n"}}}}]
        c += [{"id": f"t{i}", "state": words(f"st{i}x", 12), "questions": {"q": {"type": "noul", "instructions": TPL, "criteria": {"yes": "y", "no": "n"}}}}
              for i in range(30)]
        c += [{"id": "c37", "state": words("cdst", 20), "questions": {"q": {"type": "noul", "instructions": SLOT.format("globexinc"),
                                                                       "criteria": {"yes": "globexinc " + words("ydef", 14), "no": words("ndef", 9) + " globexinc"}}}}]
        (cp / "train/a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in c))
        (cp / "confirmation/z.jsonl").write_text(json.dumps({"id": "z", "state": S1, "questions": {"q": {"type": "noul", "instructions": "x", "criteria": {}}}}) + "\n")
        A = argparse.Namespace(registry=str(d / "reg.json"), stage=str(d / "stage"), v1_code=a.v1_code, out=str(d / "ix"), procs=2)
        build(A)
        B = argparse.Namespace(index=str(d / "ix"), v1_code=a.v1_code, repo="", input=str(cp), pool="fx", splits="train,val,confirmation",
                               out=str(d / "scan"), procs=2, chunk=2000)
        scan(B)
        C = argparse.Namespace(v1_code=a.v1_code, scan_dir=str(d / "scan"), pool="fx", input=str(cp), procs=2, chunk=2000)
        adjudicate(C)
        H = {r["uid"].split("|")[1] + ":" + r["uid"].split("|")[2]: r for r in (json.loads(l) for l in gzip.open(d / "scan/hits_fx.adj.jsonl.gz", "rt"))}
        get = lambda ln: H.get(f"train/a.jsonl:{ln}", {"s": [], "w": [], "m": []})       # noqa: E731
        checks = {
            "state_exact_c1": "state_exact" in get(1)["s"],
            "candidate_document_containment_c2": any(m[0] in ("cand_in_eval", "text_long") and m[4] == "candidate_document" for m in get(2)["m"]),
            "option_text_c3": any(m[0] == "text_long" and m[1] == "option_text" and m[4] == "option_text" for m in get(3)["m"]),
            "near_dup_c4": "near_dup" in get(4)["s"],
            "sealed_sha1_c5": "sha1" in get(5)["s"] and "fx_sealed" in get(5)["sp"],
            "unrelated_c6_clean": not get(6)["s"],
            "file_attribution": any(m[3] == "fx/test.jsonl" for m in get(1)["m"]),
            "confirmation_not_opened": not any("confirmation/" in k for k in H),
            "T1_template_demoted": all(not get(ln)["s"] and "text_template" in get(ln)["w"] for ln in range(7, 37)),
            "A1_slot_template_not_strong_c37": not get(37)["s"],
        }
        return {"checks": checks, "pass": all(checks.values())}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def regress(a):
    assert os.environ.get("PYTHONHASHSEED") == "0"
    out = {"fixtures": regress_fixtures(a)}
    if a.e1_hits:
        load_index(a)
        out.update(regress_errata(a))
    out["pass"] = all(v.get("pass") for v in out.values() if isinstance(v, dict))
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--registry", required=True)
    s.add_argument("--stage", required=True)
    s.add_argument("--repo", required=True)
    s.add_argument("--only", default="")
    s.add_argument("--redo", action="store_true")
    b = sub.add_parser("build")
    b.add_argument("--registry", required=True)
    b.add_argument("--stage", required=True)
    b.add_argument("--v1-code", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--procs", type=int, default=6)
    c = sub.add_parser("scan")
    c.add_argument("--index", required=True)
    c.add_argument("--v1-code", required=True)
    c.add_argument("--repo", default="")
    c.add_argument("--input", required=True)
    c.add_argument("--pool", required=True)
    c.add_argument("--splits", default="train,val")
    c.add_argument("--out", required=True)
    c.add_argument("--procs", type=int, default=6)
    c.add_argument("--chunk", type=int, default=1000)
    j = sub.add_parser("adjudicate")
    j.add_argument("--v1-code", required=True)
    j.add_argument("--scan-dir", required=True)
    j.add_argument("--pool", required=True)
    j.add_argument("--input", required=True)
    j.add_argument("--procs", type=int, default=6)
    j.add_argument("--chunk", type=int, default=1000)
    r = sub.add_parser("regress")
    r.add_argument("--v1-code", required=True)
    r.add_argument("--index", default="")
    r.add_argument("--repo", default="")
    r.add_argument("--stage", default="")
    r.add_argument("--e1-hits", default="")
    r.add_argument("--e2-hits", default="")
    r.add_argument("--e2-cand", default="")
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    {"stage": stage, "build": build, "scan": scan, "adjudicate": adjudicate, "regress": regress}[a.cmd](a)


if __name__ == "__main__":
    main()
