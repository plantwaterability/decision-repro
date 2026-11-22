#!/usr/bin/env python3
"""Garas pilot v2: pinned known-evaluation index (stage -> build -> scan). CPU only; no model; labels never printed or stored.

  stage  fetch every registry_v2 population onto local disk, checking every pin (tarball md5, member md5, kit sha256 gates);
         writes STAGE/stage_manifest.json (per population: status, files, md5s, substitutes, failures).
  build  index every staged population into one immutable .npz (sorted 64-bit key families + 8-word-shingle arrays) and a
         manifest (input md5s, code sha256, parameters, coverage_complete). Opaque sealed indexes (final_v1 hl/rd, CONFIRM-2610)
         are adapted: ids / unassigned eval-only ids / state sha1 / clusters / entity keys, exactly ss_lib.hits' semantics.
  scan   check every train/val unit of a candidate pool (S3 tgz or directory; a `confirmation/` split is never opened) and write
         hits_<pool>.jsonl.gz (uid = "<pool>|<member>|<line>|<qid>") + summary_<pool>.json.

Identity families (STRONG = the unit is withheld from the pilot; WEAK = counted for adjudication only):
  STRONG state_exact, state_nfkc (v1 decontam norm), sha1 (claude_overlap.norm sha1 = ss_lib shash; also every sealed index
         state_sha1), state_flat (ext2 flatten: JSON-vs-text), joint_exact / joint_norm (v1 decontam state+question+options),
         text_long (a >= 12-word normalised text unit -- whole flat state or one JSON leaf -- equal on both sides),
         rid_nostate (record id of a state-less eval record), sealed_id, sealed_cluster, sealed_entity (build_jev_rd1.entity_keys),
         lineage (source-family-namespaced base / variant / group / case roots that are not the bare record id),
         near_dup (8-word-shingle Jaccard >= 0.8), eval_in_cand / cand_in_eval (containment >= 0.9, >= 20 shingles).
  WEAK   text_short (5-11 words), question_options / question_norm / options_norm / template (v1 decontam), rid_any,
         lineage_weak (roots equal to a record id or short numeric-like), shingle3 (>= 3 shared non-stop shingles).
Shingles with eval document frequency > DF_MAX are template boilerplate: excluded from near-dup on both sides (reported).
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

SCHEMA = "garas-evalindex-v2"
BUCKET = "YOUR_BUCKET"
S3 = f"s3://{BUCKET}/jev-race/"
SHINGLE_WORDS, DF_MAX, NEAR_J, CONTAIN, CONTAIN_MIN, SH3 = 8, 100, 0.8, 0.9, 20, 3
TEXT_MIN, TEXT_LONG = 5, 12
WORD = re.compile(r"\w+")
EVAL_PART = re.compile(r"^(val|valid|validation|dev|test|eval|heldout|held_out|holdout|novel|confirm|sealed|bench|probe|extended|"
                       r"overlap|ood|unlabel|teacher_fill|gde1|meragpt)", re.I)          # = fleet/ss_lib.py EVAL_PART
STRONG = ("state_exact", "state_nfkc", "sha1", "state_flat", "joint_exact", "joint_norm", "text_long", "rid_nostate", "sealed_id",
          "sealed_cluster", "sealed_entity", "lineage", "near_dup", "eval_in_cand", "cand_in_eval")
WEAK = ("text_short", "text_boiler", "question_options", "question_norm", "options_norm", "template", "rid_any", "lineage_weak", "shingle3")
FAMILIES = ("state_exact", "state_nfkc", "sha1", "state_flat", "joint_exact", "joint_norm", "text", "rid_nostate", "rid_any",
            "sealed_id", "sealed_cluster", "sealed_entity", "lineage", "lineage_weak", "question_options", "question_norm",
            "options_norm", "template")
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
    spec = importlib.util.spec_from_file_location("ev2_rd1", os.path.join(repo, "build_jev_rd1.py"))
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


def state_keys(state):
    """record-level keys of one state -> dict family -> set of ints, plus the shingle array and text units."""
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
    return k, shingles(nf), texts


def unit_keys(rec, q):
    try:
        i = D.identity(rec, q)
    except Exception:                                   # noqa: BLE001 -- unsupported question type: state-level keys only
        return None
    return {"joint_exact": hexkey(i["joint_exact"]), "joint_norm": hexkey(i["joint_normalized"]),
            "question_options": hexkey(i["question_options"]), "question_norm": hexkey(i["question_normalized"]),
            "options_norm": hexkey(i["options_normalized"]), "template": hexkey(i["template"])}


def keyfn_sha256():
    """sha256 over the source of every function / constant that defines an index key (scan refuses an index built otherwise)."""
    import inspect
    src = "".join(inspect.getsource(f) for f in (h64, hexkey, cnorm, flatten, shingles, fam_of, lineage_roots, state_keys, unit_keys))
    src += repr((SHINGLE_WORDS, DF_MAX, TEXT_MIN, TEXT_LONG, WORD.pattern, ROOT_NUMERIC.pattern)) + file_md5(D.__file__, "sha256")
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
                                r2 = {"id": str(e.get("run_id") or row.get("id")), "state": row.get("state"), "source": f"decision_index/raw_{e.get('catalog_id')}",
                                      "meta": {"eval_only": True}}
                                counts["fingerprint_only"] += 1
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
def _index_file(job):
    pop_i, path, kind = job
    fam = collections.defaultdict(list)
    states = []                                     # (sha1key, shingles) per record with a state
    texts = collections.Counter()                   # text-unit key -> number of records holding it (eval document frequency)
    nrec = nunit = 0
    stem = Path(path).stem
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            nrec += 1
            st = rec.get("state")
            rid = rec.get("id")
            if rid is not None and kind == "jev":
                fam["rid_any"].append(h64("r:" + str(rid)))
                if st is None:
                    fam["rid_nostate"].append(h64("r:" + str(rid)))
            if kind == "jev":
                s, w = lineage_roots(rec, stem)
                fam["lineage"].extend(h64(x) for x in s)
                fam["lineage_weak"].extend(h64(x) for x in w)
            if st is None:
                continue
            k, sh, tx = state_keys(st)
            for f_, vals in k.items():
                fam[f_].extend(vals)
            texts.update(set(tx))
            states.append((min(k["sha1"]), sh))
            if kind == "jev" and isinstance(rec.get("questions"), dict):
                for q in rec["questions"].values():
                    if not isinstance(q, dict):
                        continue
                    u = unit_keys(rec, q)
                    nunit += 1
                    if u:
                        for f_, v in u.items():
                            fam[f_].append(v)
    fam["text"] = list(texts)
    out = {f_: np.unique(np.array(v, dtype=np.uint64)) for f_, v in fam.items() if v}
    tdf = (np.array(list(texts.keys()), dtype=np.uint64), np.array(list(texts.values()), dtype=np.int64))
    return pop_i, path, out, states, nrec, nunit, tdf


def build(a):
    assert os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0 (shingle hashes must be reproducible)"
    load_v1(a.v1_code)
    st = json.loads((Path(a.stage) / "stage_manifest.json").read_text())
    reg = json.loads(Path(a.registry).read_text())
    pops = [P["population"] for P in reg["populations"]]
    req = {P["population"]: P["required"] for P in reg["populations"]}
    jobs, opaque, coverage = [], [], {}
    for i, name in enumerate(pops):
        r = st["populations"].get(name)
        if not r or r["status"] != "ok":
            coverage[name] = {"indexed": False, "required": req[name], "reason": (r or {}).get("error", "not staged")}
            continue
        coverage[name] = {"indexed": True, "required": req[name], "files": len(r.get("files", [])),
                          "substitute": r.get("substitute"), "s3_key": r.get("s3_key"), "tarball_md5": r.get("tarball_md5")}
        for f in r.get("files", []):
            if f["kind"] == "opaque_index":
                opaque.append((i, f["path"]))
            else:
                if file_md5(f["path"]) != f["md5"]:
                    raise SystemExit(f"staged file changed: {f['path']}")
                jobs.append((i, f["path"], f["kind"]))
    fam_keys, fam_pops = collections.defaultdict(list), collections.defaultdict(list)
    sid_of, sh_list, sid_list, state_pop = {}, [], [], []
    per_pop = collections.Counter()
    per_pop_units = collections.Counter()
    tdf_k, tdf_c = [], []
    t0 = time.time()
    jobs.sort(key=lambda j: -os.path.getsize(j[1]))
    with mp.get_context("fork").Pool(a.procs) as pool:
        for n, (pi, path, out, states, nrec, nunit, tdf) in enumerate(pool.imap_unordered(_index_file, jobs, chunksize=1)):
            tdf_k.append(tdf[0])
            tdf_c.append(tdf[1])
            per_pop[pops[pi]] += nrec
            per_pop_units[pops[pi]] += nunit
            for f_, arr in out.items():
                fam_keys[f_].append(arr)
                fam_pops[f_].append(np.full(len(arr), pi, np.int16))
            for skey, sh in states:
                if skey in sid_of:
                    continue
                sid = len(state_pop)
                sid_of[skey] = sid
                state_pop.append(pi)
                if len(sh):
                    sh_list.append(sh)
                    sid_list.append(np.full(len(sh), sid, np.int32))
            if n % 50 == 0:
                print(f"build: {n + 1}/{len(jobs)} files, {len(state_pop)} states, {time.time() - t0:.0f}s", flush=True)
    for pi, path in opaque:                              # sealed exclusion indexes (ss_lib.load_exclusions semantics)
        ix = json.load(gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8"))
        add = {"sealed_id": [h64("r:" + str(v)) for v in list(ix.get("ids") or []) + list(ix.get("unassigned_eval_only_ids") or [])],
               "sha1": [hexkey(str(v)) for v in ix.get("state_sha1") or []],
               "sealed_cluster": [h64("c:" + str(v)) for v in ix.get("clusters") or []],
               "sealed_entity": [h64("e:" + str(v)) for v in ix.get("entity_keys") or []]}
        if not (add["sealed_id"] or add["sha1"]):
            raise SystemExit(f"opaque index {path} has no ids / state sha1s")
        coverage[pops[pi]]["opaque_counts"] = {k: len(v) for k, v in add.items()}
        for f_, v in add.items():
            if v:
                arr = np.unique(np.array(v, dtype=np.uint64))
                fam_keys[f_].append(arr)
                fam_pops[f_].append(np.full(len(arr), pi, np.int16))
    arrays = {}
    for f_ in fam_keys:
        K = np.concatenate(fam_keys[f_])
        Pp = np.concatenate(fam_pops[f_])
        o = np.lexsort((Pp, K))
        K, Pp = K[o], Pp[o]
        keep = np.ones(len(K), bool)
        keep[1:] = (K[1:] != K[:-1]) | (Pp[1:] != Pp[:-1])
        arrays[f"k_{f_}"], arrays[f"p_{f_}"] = K[keep], Pp[keep]
    if "k_text" in arrays:                               # text units in > DF_MAX eval records are boilerplate: weak only
        Kt, Ct = np.concatenate(tdf_k), np.concatenate(tdf_c)
        o = np.argsort(Kt, kind="stable")
        Kt, Ct = Kt[o], Ct[o]
        ut, it = np.unique(Kt, return_index=True)
        dft = np.add.reduceat(Ct, it)
        boiler = ut[dft > DF_MAX]
        isb = np.isin(arrays["k_text"], boiler)
        arrays["k_text_boiler"], arrays["p_text_boiler"] = arrays["k_text"][isb], arrays["p_text"][isb]
        arrays["k_text"], arrays["p_text"] = arrays["k_text"][~isb], arrays["p_text"][~isb]
        del Kt, Ct, ut, it, dft
    H = np.concatenate(sh_list) if sh_list else np.zeros(0, np.int64)
    S = np.concatenate(sid_list) if sid_list else np.zeros(0, np.int32)
    del sh_list, sid_list
    o = np.argsort(H, kind="stable")
    H, S = H[o], S[o]
    del o
    uniq, start, cnt = np.unique(H, return_index=True, return_counts=True)
    stop_keys = uniq[cnt > DF_MAX]
    df_entry = np.repeat(cnt.astype(np.int32), cnt)
    nonstop = df_entry <= DF_MAX
    nstate = len(state_pop)
    state_total = np.bincount(S, minlength=nstate).astype(np.int32)
    state_nonstop = np.bincount(S[nonstop], minlength=nstate).astype(np.int32)
    H, S = H[nonstop], S[nonstop]
    arrays.update(sh_H=H, sh_S=S, sh_stop=stop_keys, state_pop=np.array(state_pop, np.int16), state_total=state_total,
                  state_nonstop=state_nonstop)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    npz = out / "evalindex_v2.npz"
    if npz.exists():
        raise SystemExit(f"{npz} exists: indexes are immutable; use a new --out")
    np.savez(out / "evalindex_v2.partial.npz", **arrays)
    os.replace(out / "evalindex_v2.partial.npz", npz)
    req_missing = [n for n, c in coverage.items() if c["required"] and not c["indexed"]]
    subs = {n: c["substitute"] for n, c in coverage.items() if c.get("substitute")}
    hist = [P for P in reg["populations"] if P.get("historical")]
    man = {"schema": SCHEMA, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": sys.version.split()[0],
           "pythonhashseed": os.environ.get("PYTHONHASHSEED"), "code_sha256": file_md5(__file__, "sha256"), "keyfn_sha256": keyfn_sha256(),
           "v1_decontam_sha256": file_md5(Path(a.v1_code) / "decontam.py", "sha256"), "registry_sha256": file_md5(a.registry, "sha256"),
           "stage_manifest_sha256": file_md5(Path(a.stage) / "stage_manifest.json", "sha256"), "populations": pops,
           "params": {"shingle_words": SHINGLE_WORDS, "df_max": DF_MAX, "near_jaccard": NEAR_J, "containment": CONTAIN,
                      "containment_min_shingles": CONTAIN_MIN, "shingle3": SH3, "text_min_words": TEXT_MIN, "text_long_words": TEXT_LONG},
           "strong_kinds": STRONG, "weak_kinds": WEAK, "records_per_population": dict(per_pop), "units_per_population": dict(per_pop_units),
           "unique_states": nstate, "shingle_entries_nonstop": int(len(H)), "stop_shingles": int(len(stop_keys)),
           "family_sizes": {k[2:]: int(len(v)) for k, v in arrays.items() if k.startswith("k_")}, "coverage": coverage,
           "historical_tags_indexed": sum(coverage[P["population"]]["indexed"] for P in hist), "historical_tags": len(hist),
           "historical_files_indexed": sum(coverage[P["population"]].get("files", 0) for P in hist if coverage[P["population"]]["indexed"]),
           "required_not_indexed": req_missing, "substitutes": subs,
           "coverage_complete": not req_missing,
           "coverage_scope": "the registry_v2 populations (47 historical tags / 976 files + the named additions). Not the Decision Index "
                             "private / new-domain panels, not unregistered public copies of the same benchmarks, not semantic paraphrase.",
           "npz_sha256": file_md5(npz, "sha256"), "npz_md5": file_md5(npz)}
    (out / "evalindex_v2_manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: man[k] for k in ("unique_states", "shingle_entries_nonstop", "stop_shingles", "coverage_complete",
                                           "required_not_indexed", "npz_md5")}), flush=True)


# =============================================================================================== scan
IDX = {}


def _lookup(fam, keys):
    """-> list of (key, [pop ids]) for keys present in family fam."""
    K, P = IDX.get("k_" + fam), IDX.get("p_" + fam)
    if K is None or not len(keys):
        return []
    q = np.array(sorted(set(keys)), dtype=np.uint64)
    lo, hi = np.searchsorted(K, q, "left"), np.searchsorted(K, q, "right")
    return [(int(q[i]), P[lo[i]:hi[i]].tolist()) for i in np.nonzero(hi > lo)[0]]


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
    H, S = IDX["sh_H"], IDX["sh_S"]
    lo, hi = np.searchsorted(H, sh, "left"), np.searchsorted(H, sh, "right")
    nz = np.nonzero(hi > lo)[0]
    if len(nz) == 0:
        return {"n": n, "shared": 0}
    ids = np.concatenate([S[lo[i]:hi[i]] for i in nz])
    u, c = np.unique(ids, return_counts=True)
    ne = IDX["state_nonstop"][u]
    jac = c / np.maximum(1, n + ne - c)
    ce = c / np.maximum(1, ne)
    cc = c / n
    b = int(np.argmax(jac))
    res = {"n": n, "shared": int(c.max()), "j": round(float(jac.max()), 4), "j_pop": int(IDX["state_pop"][u[b]]),
           "ce": 0.0, "cc": 0.0}
    m1 = ne >= CONTAIN_MIN
    if m1.any():
        res["ce"] = round(float(ce[m1].max()), 4)
        res["ce_pop"] = int(IDX["state_pop"][u[m1][int(np.argmax(ce[m1]))]])
    if n >= CONTAIN_MIN:
        bc = int(np.argmax(cc))
        res["cc"], res["cc_pop"] = round(float(cc[bc]), 4), int(IDX["state_pop"][u[bc]])
    return res


def _scan_chunk(job):
    pool_tag, member, lines = job
    stem = Path(member).stem
    rows, stats = [], collections.Counter()
    for line_no, line in lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        st, qs = rec.get("state"), rec.get("questions") or {}
        rec_strong, rec_weak, pops = collections.defaultdict(set), collections.defaultdict(set), set()
        near = None
        if st is not None:
            k, sh, tx = state_keys(st)
            for f_ in ("state_exact", "state_nfkc", "sha1", "state_flat"):
                for _, ps in _lookup(f_, list(k[f_])):
                    rec_strong[f_].update(ps)
            for key, ps in _lookup("text", list(tx)):
                (rec_strong["text_long"] if tx[key] >= TEXT_LONG else rec_weak["text_short"]).update(ps)
            for key, ps in _lookup("text_boiler", list(tx)):
                rec_weak["text_boiler"].update(ps)
            if len(cnorm(st if isinstance(st, str) else json.dumps(st, ensure_ascii=False))) < 20:
                stats["short_state_records"] += 1
            near = _near(sh)
            if near and near.get("shared", 0):
                if near["j"] >= NEAR_J:
                    rec_strong["near_dup"].add(near["j_pop"])
                if near.get("ce", 0) >= CONTAIN:
                    rec_strong["eval_in_cand"].add(near["ce_pop"])
                if near.get("cc", 0) >= CONTAIN:
                    rec_strong["cand_in_eval"].add(near["cc_pop"])
                if near["shared"] >= SH3:
                    rec_weak["shingle3"].add(near["j_pop"])
        rid = str(rec.get("id"))
        for _, ps in _lookup("rid_nostate", [h64("r:" + rid)]):
            rec_strong["rid_nostate"].update(ps)
        for _, ps in _lookup("rid_any", [h64("r:" + rid)]):
            rec_weak["rid_any"].update(ps)
        for _, ps in _lookup("sealed_id", [h64("r:" + rid)]):
            rec_strong["sealed_id"].update(ps)
        m = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
        src = str(rec.get("source") or stem)
        ck = [h64("c:" + f"{s_}:{m.get('group') or rec.get('id')}") for s_ in {src, src.split('/')[0], src.replace('/', '__'), stem}]
        for _, ps in _lookup("sealed_cluster", ck):
            rec_strong["sealed_cluster"].update(ps)
        if ENTITY is not None and "k_sealed_entity" in IDX:
            try:
                ek = [h64("e:" + str(x)) for x in ENTITY(rec)]
            except Exception:                                 # noqa: BLE001 -- as ss_lib.hits
                ek = []
            for _, ps in _lookup("sealed_entity", ek):
                rec_strong["sealed_entity"].update(ps)
        s_, w_ = lineage_roots(rec, stem)
        for _, ps in _lookup("lineage", [h64(x) for x in s_]):
            rec_strong["lineage"].update(ps)
        for _, ps in _lookup("lineage", [h64(x) for x in w_]) + _lookup("lineage_weak", [h64(x) for x in s_ | w_]):
            rec_weak["lineage_weak"].update(ps)
        for qid, q in qs.items():
            stats["units"] += 1
            us, uw = {k_: set(v) for k_, v in rec_strong.items()}, {k_: set(v) for k_, v in rec_weak.items()}
            u = unit_keys(rec, q) if isinstance(q, dict) else None
            if u:
                for f_ in ("joint_exact", "joint_norm"):
                    for _, ps in _lookup(f_, [u[f_]]):
                        us.setdefault(f_, set()).update(ps)
                for f_ in ("question_options", "question_norm", "options_norm", "template"):
                    if f_ == "options_norm" and q.get("type") in ("noul", "score"):
                        continue                      # Yes/No and level menus are shared by construction
                    for _, ps in _lookup(f_, [u[f_]]):
                        uw.setdefault(f_, set()).update(ps)
            else:
                stats["unit_keys_unavailable"] += 1
            if us or uw:
                row = {"uid": f"{pool_tag}|{member}|{line_no}|{qid}", "s": sorted(us), "w": sorted(uw),
                       "sp": sorted({p for v in us.values() for p in v})[:8], "wp": sorted({p for v in uw.values() for p in v})[:8]}
                if near and near.get("shared", 0):
                    row["nd"] = {k_: near[k_] for k_ in ("n", "shared", "j", "ce", "cc") if k_ in near}
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


def iter_members(spec, splits):
    """(member, [(line_no, line)] chunks) of every *.jsonl under SPEC whose split dir is in splits; confirmation never opened."""
    def ok(member):
        sp = member.split("/")[0]
        return member.endswith(".jsonl") and sp in splits and sp != "confirmation" and not os.path.basename(member).startswith("._")

    def chunks(member, f):
        buf = []
        for i, line in enumerate(f, 1):
            buf.append((i, line))
            if len(buf) >= 2000:
                yield member, buf
                buf = []
        if buf:
            yield member, buf
    if spec.startswith("s3://") or spec.endswith(".tgz"):
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


def scan(a):
    global IDX
    assert os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0"
    load_v1(a.v1_code)
    if a.repo:
        load_entity(a.repo)
    man = json.loads((Path(a.index) / "evalindex_v2_manifest.json").read_text())
    npz = Path(a.index) / "evalindex_v2.npz"
    if file_md5(npz) != man["npz_md5"] or man["python"] != sys.version.split()[0]:
        raise SystemExit("index integrity / python version mismatch")
    if man["keyfn_sha256"] != keyfn_sha256():
        raise SystemExit("scanner key functions differ from the index builder's (rebuild the index)")
    z = np.load(npz)
    IDX = {k: z[k] for k in z.files}
    pops = man["populations"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hits_p = out / f"hits_{a.pool}.jsonl.gz"
    if hits_p.exists():
        raise SystemExit(f"{hits_p} exists (immutable)")
    splits = set(a.splits.split(","))
    tot = collections.Counter()
    by_member = collections.defaultdict(collections.Counter)
    by_pop = collections.Counter()
    t0 = time.time()
    with gzip.open(out / f"hits_{a.pool}.partial.jsonl.gz", "wt", encoding="utf-8") as h, mp.get_context("fork").Pool(a.procs) as pool:
        jobs = ((a.pool, mem, lines) for mem, lines in iter_members(a.input, splits))
        for n, (member, rows, stats) in enumerate(bounded_imap(pool, _scan_chunk, jobs, 3 * a.procs)):
            tot.update(stats)
            by_member[member].update(stats)
            for r in rows:
                for p in r["sp"]:
                    by_pop["strong:" + pops[p]] += 1
                for p in r["wp"]:
                    by_pop["weak:" + pops[p]] += 1
                r["sp"] = [pops[p] for p in r["sp"]]
                r["wp"] = [pops[p] for p in r["wp"]]
                h.write(json.dumps(r, separators=(",", ":")) + "\n")
            if n % 50 == 0:
                print(f"scan {a.pool}: chunk {n}, units {tot['units']}, strong {tot['strong_units']}, {time.time() - t0:.0f}s", flush=True)
    os.replace(out / f"hits_{a.pool}.partial.jsonl.gz", hits_p)
    summ = {"schema": SCHEMA, "pool": a.pool, "input": a.input, "input_md5": iter_members.last_md5, "splits": sorted(splits),
            "index_npz_md5": man["npz_md5"], "index_coverage_complete": man["coverage_complete"], "totals": dict(tot),
            "by_population": dict(by_pop), "by_member": {k: dict(v) for k, v in sorted(by_member.items())},
            "hits_file": hits_p.name, "hits_md5": file_md5(hits_p), "seconds": round(time.time() - t0, 1),
            "policy": "units with any STRONG kind are withheld from the pilot (reversible ledger); WEAK-only units stay eligible and are "
                      "listed for adjudication; absence from the hits file = no indexed identity found"}
    (out / f"summary_{a.pool}.json").write_text(json.dumps(summ, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: summ[k] for k in ("pool", "totals", "hits_md5", "seconds")}), flush=True)


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
    a = ap.parse_args()
    {"stage": stage, "build": build, "scan": scan}[a.cmd](a)


if __name__ == "__main__":
    main()
