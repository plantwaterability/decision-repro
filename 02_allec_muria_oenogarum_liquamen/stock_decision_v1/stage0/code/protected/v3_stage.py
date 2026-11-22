

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


