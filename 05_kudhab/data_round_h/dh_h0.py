#!/usr/bin/env python3
"""dh_h0.py (DH fork, round H, 2026-11-16): requests for the FS5 rows that have NO karpo teacher target (MX's converter stops at 26 options and needs a
"Question:" block). Parser = kl_data.py's own parse (so the layout equals what the K trainer builds for these rows) with ONE change: a row whose body
starts with "Question: " and has no state (the GPQA shape: empty state, question as instructions, as the kit's normalize_text.gpqa asks it) becomes
state "" + instructions = the question (kl_data's fallback puts the whole text in state with no instructions).
  dh_h0.py FS5.jsonl TEACHER.jsonl.gz OUT.jsonl.gz    OUT rows: {"id": exposure_id, "n_tokens", "request": {"state", "questions": {"q": {...}}}}"""
import collections, gzip, hashlib, json, string, sys
src, teacher, out = sys.argv[1:4]
LAB = list(string.ascii_uppercase) + list(string.ascii_lowercase) + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
try:
    sys.path.insert(0, "/data/shared/fs5/code/esbar_code_v2e/trainers"); import jev_labels
    L2 = jev_labels.labels(); assert L2[:len(LAB)][:300] == LAB[:min(len(L2), 300)][:300] or L2[:130] == LAB[:130], "label sequence differs"; LAB = L2
    print("labels from jev_labels", len(LAB), LAB[24:30], LAB[50:55])
except ImportError:
    print("jev_labels not importable: built-in sequence", len(LAB))
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
    how = "own"
    if q >= 0: state_text, instr = head[:q], head[q + len("\n\nQuestion: "):]
    elif head.startswith("Question: "): state_text, instr, how = "", head[len("Question: "):], "gpqa_shape"
    else: state_text, instr, how = head, None, "own_noq"
    try: state = json.loads(state_text) if state_text else ""
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
    return {"state": state, "questions": {"q": qn}}, how
have = set()
for l in gzip.open(teacher, "rt"):
    have.add(json.loads(l)["id"])
st = collections.Counter(); bysrc = collections.Counter(); ex = {}
h = hashlib.sha256(); n = 0
with gzip.open(out, "wt", compresslevel=6) as f:
    for l in open(src):
        r = json.loads(l)
        if r["exposure_id"] in have: continue
        st["no_teacher"] += 1
        req, how = parse(r)
        if req is None: st["drop_" + how] += 1; bysrc[("DROP", r["source_family"], how)] += 1; continue
        if set(req["questions"]["q"]["criteria"]) != set(r["gold"]): st["drop_gold_keys"] += 1; continue
        st["req_" + how] += 1; st["kind_" + str(r.get("kind"))] += 1; bysrc[r["source_family"]] += 1
        line = json.dumps({"id": r["exposure_id"], "n_tokens": int(r["n_tokens"]), "request": req}, ensure_ascii=False)
        f.write(line + "\n"); h.update((line + "\n").encode()); n += 1
        if how not in ex: ex[how] = json.dumps(req, ensure_ascii=False)[:500]
print("H0", n, dict(st)); print("BYSRC", sorted(((v, str(k)) for k, v in bysrc.items()), reverse=True)[:40])
for k, v in ex.items(): print("EXAMPLE", k, v)
json.dump({"rows": n, "stats": dict(st), "by_source": {str(k): v for k, v in bysrc.items()}, "content_sha256": h.hexdigest()}, open(out.replace(".jsonl.gz", ".stats.json"), "w"))
