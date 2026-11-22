

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
