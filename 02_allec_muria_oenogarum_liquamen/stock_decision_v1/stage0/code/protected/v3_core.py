

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
