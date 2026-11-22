

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
