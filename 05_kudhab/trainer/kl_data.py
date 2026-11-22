#!/usr/bin/env python3
"""kl_data.py (KL agent, 2026-11-15): training / dev rows of our manifests -> token ids in karpo's layout (karpo-blocks-v1, one question per row,
karpo_decision's own Encoder at a6fa38bf) + gold / teacher distributions in the SHOWN option order.
  kl_data.py MANIFEST.jsonl OUT_PREFIX TOKENIZER_DIR PINNED_CODE_DIR [--requests fs5.jsonl.gz] [--teacher teacher.jsonl.gz] [--max N] [--workers 64]
A row's request = the MX fork's parsed request (fs5.jsonl.gz: exactly what the teacher was labelled on) when it has one; else parsed here with
our own label sequence (jev_labels: A-Z, a-z, AA..; mx_tl_convert stops at Z) and, without a "Question:" block, no instructions (karpo's default).
Teacher q = the teacher file's probabilities (keys must equal the row's option keys). Rows that do not parse / exceed 255 options are dropped, counted.
OUT: .ids.npy (int32, all rows' tokens), .meta.npz (off, K, w, ooff, gold, q, hasq), .rowids.txt, .info.json (counts, sha256, length stats)."""
import argparse, collections, gzip, hashlib, json, multiprocessing as mp, sys, time
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("manifest"); ap.add_argument("out"); ap.add_argument("tokdir"); ap.add_argument("pinned")
ap.add_argument("--requests", default=""); ap.add_argument("--teacher", default=""); ap.add_argument("--exclude-ids", default="", help="file of exposure ids left out entirely (counted as excluded_by_list); empty = none")
ap.add_argument("--max", type=int, default=0); ap.add_argument("--workers", type=int, default=64)
a = ap.parse_args()
sys.path.insert(0, a.pinned)
import jev_labels
LAB = jev_labels.labels()
HEAD = "<bos><|turn>user\n"; TAIL = "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
def parse(r):
    p = r["prompt"]; keys = r.get("option_order") or list(r["gold"].keys())
    if not p.startswith(HEAD) or not p.endswith(TAIL): return None, "render_head_tail"
    body = p[len(HEAD):-len(TAIL)]
    i = body.rfind("\n\nOptions:\n")
    if i < 0: return None, "no_options_block"
    head, rest = body[:i], body[i + len("\n\nOptions:\n"):]
    j = rest.rfind("\n\nAnswer with the letter")
    if j < 0: return None, "no_answer_line"
    opts = rest[:j]
    q = head.find("\n\nQuestion: ")
    state_text, instr = (head, None) if q < 0 else (head[:q], head[q + len("\n\nQuestion: "):])
    try: state = json.loads(state_text)
    except Exception: state = state_text
    if len(keys) > len(LAB) or len(keys) > 255: return None, "too_many_options"
    cs = LAB[:len(keys)]; texts = []; pos = 0
    if not opts.startswith(cs[0] + ") "): return None, "first_option"
    for n, c in enumerate(cs):
        start = pos + len(c) + 2
        if n + 1 < len(cs):
            nxt = opts.find("\n" + cs[n + 1] + ") ", start)
            if nxt < 0: return None, "option_sequence"
            texts.append(opts[start:nxt]); pos = nxt + 1
        else: texts.append(opts[start:])
    if len(set(keys)) != len(keys): return None, "duplicate_keys"
    qn = {"type": "choice", "criteria": dict(zip(keys, texts))}
    if instr is not None and instr.strip(): qn["instructions"] = instr.strip()
    return {"state": state, "questions": {"q": qn}}, ("own_noq" if q < 0 else "own")
ENC = None
def init():
    global ENC
    from karpo_decision.codes import answer_codes
    from karpo_decision.encoding import Encoder
    from karpo_decision.families import gemma
    from karpo_decision.templates import template
    proc = gemma.load_processor(a.tokdir, 0); codes, _ = answer_codes(proc.tokenizer)
    ENC = Encoder(proc, codes, template("karpo-blocks-v1"), gemma.MARKER)
def tok(req):
    try:
        e = ENC.plan(req, None)
        assert len(e) == 1 and int(e[0].markers[0]) == e[0].length - 1
        return np.asarray(e[0].input_ids, dtype=np.int32)
    except Exception as ex: return f"{type(ex).__name__}: {str(ex)[:100]}"
if __name__ == "__main__":
    t0 = time.time(); st = collections.Counter(); reqs = {}; teach = {}
    for f in [x for x in a.requests.split(",") if x]:      # one or several files (comma-separated); the first file that names an id wins
        for l in gzip.open(f, "rt"):
            r = json.loads(l)
            if r["id"] in reqs: st["request_id_in_two_files"] += 1; continue
            reqs[r["id"]] = r["request"] if "request" in r else {"state": r["state"], "questions": r["questions"]}
    for f in [x for x in a.teacher.split(",") if x]:
        for l in gzip.open(f, "rt"):
            r = json.loads(l)
            if "p" not in r: continue
            if r["id"] in teach: st["teacher_id_in_two_files"] += 1; continue
            teach[r["id"]] = r["p"]
    print(f"KLDATA requests {len(reqs)} teacher {len(teach)} {time.time() - t0:.0f} s", flush=True)
    rows = []; shown = set()
    excl = set(l.strip() for l in open(a.exclude_ids) if l.strip()) if a.exclude_ids else set()
    for l in open(a.manifest):
        r = json.loads(l)
        if r.get("manifest_header"): continue
        if a.max and len(rows) >= a.max: break
        eid = r["exposure_id"]
        if eid in excl: st["excluded_by_list"] += 1; continue
        if eid in reqs: req, how = reqs[eid], "mx"
        else: req, how = parse(r)
        if req is None: st["drop_" + how] += 1; continue
        if how not in shown and how != "mx":
            shown.add(how); print("EXAMPLE", how, r.get("kind"), json.dumps(req)[-700:], flush=True)
        if len(req["questions"]) != 1: st["drop_multi_question_request"] += 1; continue
        qid = next(iter(req["questions"])); keys = list(req["questions"][qid]["criteria"]) if isinstance(req["questions"][qid].get("criteria"), dict) else None
        if keys is None: st["drop_non_choice_request"] += 1; continue
        if set(keys) != set(r["gold"]): st["drop_gold_keys"] += 1; continue
        gold = [float(r["gold"][k]) for k in keys]
        if not abs(sum(gold) - 1.0) < 1e-4: st["drop_gold_sum"] += 1; continue
        q = teach.get(eid)
        if q is not None and set(q) != set(keys): st["teacher_key_mismatch"] += 1; q = None
        st["req_" + how] += 1; st["kind_" + str(r.get("kind"))] += 1
        rows.append((eid, req, gold, None if q is None else [float(q[k]) for k in keys], float(r.get("weight", 1.0))))
    print(f"KLDATA parsed {len(rows)} rows {dict(st)} {time.time() - t0:.0f} s", flush=True)
    with mp.Pool(a.workers, initializer=init) as pool:
        toks = pool.map(tok, [r[1] for r in rows], chunksize=512)
    keep = [i for i, t in enumerate(toks) if not isinstance(t, str)]
    for t in toks:
        if isinstance(t, str): st["drop_encode:" + t.split(":")[0]] += 1
    lens = np.array([len(toks[i]) for i in keep], dtype=np.int64); off = np.concatenate([[0], np.cumsum(lens)])
    ids = np.concatenate([toks[i] for i in keep]).astype(np.int32)
    K = np.array([len(rows[i][2]) for i in keep], dtype=np.int16); ooff = np.concatenate([[0], np.cumsum(K.astype(np.int64))])
    gold = np.concatenate([np.asarray(rows[i][2], dtype=np.float32) for i in keep])
    hasq = np.array([rows[i][3] is not None for i in keep])
    q = np.concatenate([np.asarray(rows[i][3] if rows[i][3] is not None else [0.0] * len(rows[i][2]), dtype=np.float32) for i in keep])
    w = np.array([rows[i][4] for i in keep], dtype=np.float32)
    h = hashlib.sha256()
    for x in (ids, off, K, ooff, gold, q, hasq, w): h.update(np.ascontiguousarray(x).tobytes())
    np.save(a.out + ".ids.npy", ids); np.savez(a.out + ".meta.npz", off=off, K=K, w=w, ooff=ooff, gold=gold, q=q, hasq=hasq)
    open(a.out + ".rowids.txt", "w").write("\n".join(rows[i][0] for i in keep) + "\n")
    qs = [int(x) for x in np.quantile(lens, [0.5, 0.9, 0.99, 0.999])]
    info = {"data_sha256": h.hexdigest(), "rows": len(keep), "tokens": int(lens.sum()), "len_median_p90_p99_p999": qs, "len_max": int(lens.max()),
            "over_8192": int((lens > 8192).sum()), "over_4096": int((lens > 4096).sum()), "teacher_rows": int(hasq.sum()), "weight": float(w.sum()),
            "K_max": int(K.max()), "stats": dict(st), "manifest": a.manifest, "template": "karpo-blocks-v1", "karpo_code": "a6fa38bf",
            "teacher_argmax_is_gold": int(sum(int(np.argmax(rows[i][3]) == np.argmax(rows[i][2])) for i in keep if rows[i][3] is not None))}
    json.dump(info, open(a.out + ".info.json", "w"), indent=1); print("KLDATA_DONE", json.dumps(info), f"{time.time() - t0:.0f} s", flush=True)
