#!/usr/bin/env python3
"""fs5h_build.py (DH fork, round H, 2026-11-16): targeted single-question training pools for the DI gap benchmarks, in FS5's row format.
Same conventions as fs5_build.py (its helpers, Gate and renderer are reused unchanged): DI-format rows (kit normalizer twins), rendered with
build_manifest_v3.make_render (R3), gated with build_manifest_v3.Gate over the DI 0.3 public suite + FS1's / v8's held-out lists + cov2 + MMLU-CF
val, deduplicated against every FS5 row, quotas in sha1 order. Rows keep FS5's 23 keys; exposure ids fs5h:<uid>. Nothing is appended to FS5.
Pools: cfcolor (TRAIN ratings only), hellaswag (official train), banking77 (official train, 77-way), gsm8k_ag (official train, the kit's
answer-groups-v1 distractors computed over TRAIN golds), sata_mrc (MultiRC train in SATA-Bench's wording), sata_goemo (GoEmotions train, same wording).
Sealed sets, the guard set, the ESCI test split and benchmark test splits are never read as rows (cfcolor's test ratings matrix is read for
(user, palette) id bans only: its ratings are never used).
  fs5h_build.py [--caps ...]   layout: /data/shared/fs5 (code, gate, tok), /data/shared/dh (src, fs5.jsonl, out)"""
import argparse, collections, csv, glob, gzip, hashlib, json, math, os, random, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("--w", default="/data/shared/fs5"); ap.add_argument("--d", default="/data/shared/dh")
ap.add_argument("--suite", default="/data/shared/munku/kit/suite-0.3"); ap.add_argument("--seed", type=int, default=20261116)
ap.add_argument("--caps", default="cfcolor=9000,hellaswag=12000,banking77=9000,gsm8k_ag=10000,sata_mrc=12000,sata_goemo=5000")
a = ap.parse_args()
W, D = a.w, a.d; S = f"{D}/src"
sys.path[:0] = [f"{W}/code/esbar_code_v2e/trainers", f"{W}/scripts", f"{D}/scripts"]
import build_manifest_v3 as BM
import claude_lmscore as clm
import gsm8k_ag_kit as AG
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(f"{W}/tok")
CAPS = {k: int(v) for k, v in (x.split("=") for x in a.caps.split(","))}
REP = {"caps": CAPS, "pools": {}}
USER, QUE, OPT = "<|turn>user\n", "\n\nQuestion: ", "\n\nOptions:\n"
EXCLUDED_QIDS = {"action", "needs_review", "outcome", "risk", "urgency", "category", "churn_risk", "needs_human", "discrepancy_severity",
                 "disposition", "duplicate", "matches_order", "credential_compromise", "severity", "true_positive"}


def jl(p):
    with (gzip.open if p.endswith(".gz") else open)(p, "rt", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def sha1(s):
    return hashlib.sha1(s.encode()).hexdigest()


def md5d(s):
    return hashlib.md5(s.encode()).digest()


def parse_prompt(p):                       # = fs1_build.parse_prompt
    io_ = p.rfind(OPT); iq = p.rfind(QUE, 0, io_); iu = p.find(USER)
    if io_ < 0 or iu < 0:
        return None
    if iq < 0:
        body = p[iu + len(USER):io_]
        return ("", body[len("Question: "):]) if body.startswith("Question: ") else None
    return p[iu + len(USER):iq], p[iq + len(QUE):io_]


def as_text(x):
    return "" if x is None else x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def key_of(st, ins, opts=None):
    k = BM.norm_text(st) + "|" + BM.norm_text(ins)
    if opts is not None:
        k += "|" + "|".join(sorted(BM.norm_text(o) for o in opts))
    return md5d(k)


def prompt_options(p, k):
    io_ = p.rfind(OPT); ia = p.rfind("\n\nAnswer with")
    if io_ < 0 or ia < io_:
        return None
    parts = re.split(r"\n(?=[A-Za-z]\) )", p[io_ + len(OPT):ia])
    if len(parts) != k:
        return None
    return [re.sub(r"^[A-Za-z]\) ", "", x, count=1) for x in parts]


# ------------------------------------------------------------------ base = the FS5 manifest (read only: keys for dedupe)
FS5 = f"{D}/fs5.jsonl"
seen, uids, seen_opt = set(), set(), set()
KEYS = None; base_c = collections.Counter(); h5 = hashlib.sha256()
for l in open(FS5, encoding="utf-8"):
    h5.update(l.encode())
    r = json.loads(l)
    if KEYS is None:
        KEYS = list(r.keys())
    pp = parse_prompt(r["prompt"]) if r.get("kind") != "gen" else None
    if pp:
        seen.add(key_of(*pp))
        po = prompt_options(r["prompt"], len(r["option_order"]))
        if po is not None:
            seen_opt.add(key_of(pp[0], pp[1], po))
        else:
            base_c["options_unparsed"] += 1
    else:
        base_c["unparsed"] += 1
    uids.add(r["decision_uid"]); base_c["rows"] += 1
assert h5.hexdigest() == "9f6f7a48b9873ebd25961219b225d75f276d07f8da89bd8c7ba52e2ecf39112f", "FS5 input sha mismatch " + h5.hexdigest()
REP["base"] = {**base_c, "fs5_rows_sha256": h5.hexdigest()}
print("BASE", json.dumps(REP["base"]), flush=True)

# ------------------------------------------------------------------ gate (= fs5_build.py's)
gate = BM.Gate(); g = collections.Counter()
for f in ("selected-rows", "added-rows", "gsm8k-rows"):
    for r in jl(f"{a.suite}/{f}.jsonl.gz"):
        gate.add(r); g["suite"] += 1
HO = [f"{W}/gate/{x}" for x in ("dev.jsonl", "eval_fresh.jsonl", "cal.jsonl", "headfit.jsonl", "when2call_4opt_clean_dev.jsonl",
                                "official_dev_splits_eval.jsonl", "official_dev_splits_eval_v2.jsonl", "libeval_clean.jsonl")]
HO += sorted(glob.glob(f"{W}/gate/HO_*_v9.jsonl.gz")) + sorted(glob.glob(f"{W}/gate/v7/HO_*.jsonl"))
for path in HO:
    for r in jl(path):
        if r.get("manifest_header"):
            continue
        if "prompt" in r:
            pp = parse_prompt(r["prompt"])
            if not pp:
                g["heldout_unparsed"] += 1; continue
            r = {"state": pp[0], "questions": {"q": {"instructions": pp[1]}}}
        gate.add(r); g[path.rsplit("/", 1)[-1]] += 1
for r in jl(f"{W}/gate/cov2_heldout_excl.jsonl.gz"):
    for t in [r.get("q")] + list(r.get("parts") or []):
        if t:
            gate.exact.add(BM.norm_text(t)); gate.grams |= BM.grams13(t); g["cov2_texts"] += 1
import pyarrow.parquet as pq
for p in sorted(glob.glob(f"{W}/src/mmlucf/**/*.parquet", recursive=True)):
    for r in pq.read_table(p).to_pylist():
        q = r.get("Question") or r.get("question") or ""
        opts = [str(r.get(k) or "") for k in ("A", "B", "C", "D")]
        gate.add({"state": "", "questions": {"q": {"instructions": q + "\n" + "\n".join(opts)}}})
        gate.exact.add(BM.norm_text(q)); g["mmlu_cf_val"] += 1
REP["gate"] = {**g, "exact": len(gate.exact), "grams13": len(gate.grams)}
print("GATE", json.dumps(REP["gate"]), flush=True)
assert g["suite"] > 150000, g

rng = random.Random(a.seed)
render = BM.make_render(tok, clm, rng, split="train", tag="fs5h")
POOL_ROWS = {}
os.makedirs(f"{D}/out/pools", exist_ok=True)


def emit(pool, cands, line, licence, content_of, key_opts=False, balance=None, group_of=None):
    """fs5_build.emit, plus group_of: candidates of one group (a select-all item) are kept or dropped together and the cap falls on a group
    boundary. cands: {id, state, qk, q, gold, meta}; gate -> dedup -> quota (sha1 order, optional label balance) -> render."""
    c = collections.Counter(); ok = []
    cap = CAPS[pool]
    golds = collections.defaultdict(set); local = set()
    for x in cands:
        st, ins = as_text(x["state"]), as_text(x["q"]["instructions"])
        opts = [as_text(v) for v in (x["q"].get("criteria") or {}).values()] if key_opts else None
        x["_key"] = key_of(st, ins, opts)
        golds[x["_key"]].add(json.dumps(as_text(x["q"]["criteria"].get(x["gold"]))))
    order = (lambda x: (sha1(group_of(x)), sha1(x["id"]))) if group_of else (lambda x: sha1(x["id"]))
    bad_groups = set()
    for x in sorted(cands, key=order):
        assert x["qk"] not in EXCLUDED_QIDS, x["qk"]
        k = x["_key"]; why = None
        if len(golds[k]) > 1:
            why = "dropped_conflicting_label_in_source"
        elif k in local:
            why = "dropped_dup_within_source"
        elif k in seen or (key_opts and k in seen_opt):
            why = "dropped_dup_of_base_or_pool"
        else:
            local.add(k)
            content, ekeys = content_of(x)
            if not gate.clean(content, exact_keys=ekeys):
                why = "dropped_gate"
        if why:
            c[why] += 1
            if group_of:
                bad_groups.add(group_of(x))
            continue
        ok.append((x, k))
    if group_of:
        n0 = len(ok); ok = [(x, k) for x, k in ok if group_of(x) not in bad_groups]
        c["dropped_group_incomplete"] = n0 - len(ok); c["groups_dropped"] = len(bad_groups)
    c["eligible"] = len(ok)
    take = []
    if group_of:
        cur = None
        for x, k in ok:
            gk = group_of(x)
            if gk != cur and len(take) >= cap:
                break
            cur = gk; take.append((x, k))
    elif balance:
        lab = collections.Counter(balance(x) for x, _ in ok)
        quota = math.ceil(cap / max(1, len(lab)))
        lc = collections.Counter(); rest = []
        for x, k in ok:
            l = balance(x)
            if lc[l] < quota and len(take) < cap:
                lc[l] += 1; take.append((x, k))
            else:
                rest.append((x, k))
        take += rest[:max(0, cap - len(take))]
    else:
        take = ok[:cap]
    rows = []; over = set()
    for x, k in take:
        uid = f"fs5h|{pool}|{x['id']}"
        row = render(uid, f"fs5h_{pool}", x["state"], x["q"]["instructions"], x["q"]["criteria"], x["gold"], f"fs5h_{pool}")
        if row is None:
            c["dropped_over_4096_tokens"] += 1
            if group_of:
                over.add(group_of(x))
            continue
        assert row["decision_uid"] not in uids
        seen.add(k); uids.add(row["decision_uid"])
        row = {kk: row.get(kk) for kk in KEYS}
        rows.append((row, {"uid": uid, "exposure_id": row["exposure_id"], "pool": pool, "line": line, "licence": licence, "qk": x["qk"],
                           "group": group_of(x) if group_of else None, **(x.get("meta") or {})}))
    if over:
        n0 = len(rows); rows = [(r, m) for r, m in rows if m["group"] not in over]; c["dropped_group_over_tokens"] = n0 - len(rows)
    c["kept"] = len(rows)
    gold_top = collections.Counter(max(r["gold"], key=r["gold"].get) for r, _ in rows)
    REP["pools"][pool] = {"candidates": len(cands), **c, "line": line, "licence": licence, "gold_top": dict(gold_top.most_common(12)),
                          "groups": len({m["group"] for _, m in rows}) if group_of else None,
                          "k_hist": dict(collections.Counter(len(r["option_order"]) for r, _ in rows)),
                          "tokens_mean": round(sum(r["n_tokens"] for r, _ in rows) / max(1, len(rows)), 1),
                          "tokens": sum(r["n_tokens"] for r, _ in rows)}
    POOL_ROWS[pool] = rows
    with open(f"{D}/out/pools/fs5h_{pool}.rows.jsonl", "w", encoding="utf-8") as f:
        for r, _ in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    with open(f"{D}/out/pools/fs5h_{pool}.meta.jsonl", "w", encoding="utf-8") as f:
        for _, m in rows:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print("POOL", pool, json.dumps(REP["pools"][pool]), flush=True)
    if rows:
        print("EXAMPLE", pool, rows[0][0]["prompt"][:700].replace("\n", "\\n"), "...", rows[0][0]["prompt"][-260:].replace("\n", "\\n"), rows[0][0]["gold"] if len(rows[0][0]["gold"]) < 12 else "", flush=True)


# ------------------------------------------------------------------ cfcolor: TRAIN ratings only (kit adapters_creative.cfcolor twin)
def _hex(rgb):
    return ["#" + "".join(f"{int(round(float(x) * 255)):02x}" for x in rgb[i:i + 3]) for i in range(0, 15, 3)]


import numpy as np
from scipy.io import loadmat
dm = loadmat(f"{S}/cfcolor/release/allMTurkRatings.mat", squeeze_me=True)
tm = loadmat(f"{S}/cfcolor/release/themeData.mat", squeeze_me=True, struct_as_record=False)["datapoints"]
rgb = np.asarray(tm.rgb); train = np.asarray(dm["train_vec"]); test = np.asarray(dm["test_vec"])
ckeys = [tuple(_hex(x)) for x in rgb]
test_ids = collections.defaultdict(set)                       # (user) -> palette ids that are test material for that user: id ban only
for u, p in zip(test[:, 0].tolist(), test[:, 1].tolist()):
    test_ids[int(u)].add(int(p))
cands = []; cc = collections.Counter(); PER_USER = 6
by_user = collections.defaultdict(dict)
for u, p, r in train.tolist():
    by_user[int(u)].setdefault(int(p), set()).add(int(r))
for user in sorted(by_user):
    banned_keys = {ckeys[p - 1] for p in test_ids.get(user, ())}
    uniq = {p: next(iter(rs)) for p, rs in by_user[user].items() if len(rs) == 1 and p not in test_ids.get(user, ()) and ckeys[p - 1] not in banned_keys}
    cc["users"] += 1
    order = sorted(uniq.items(), key=lambda z: hashlib.sha256(f"H:{user}:{z[0]}".encode()).hexdigest())
    if len(order) < 6:
        cc["user_too_few"] += 1; continue
    targets = order[:]; random.Random(f"{a.seed}:cf:{user}").shuffle(targets)
    n = 0; used = set()
    for (pa, ra), (pb, rb) in zip(targets[::2], targets[1::2]):
        if n >= PER_USER:
            break
        if abs(ra - rb) < 2 or ckeys[pa - 1] == ckeys[pb - 1] or pa in used or pb in used:
            continue
        h = []
        for hp, hr in order:
            if hp not in (pa, pb) and hp not in used and ckeys[hp - 1] not in (ckeys[pa - 1], ckeys[pb - 1]):
                h.append({"colors": _hex(rgb[hp - 1]), "rating": hr})
                if len(h) == 8:
                    break
        if len(h) < 4:
            continue
        flip = int(hashlib.sha256(f"pair:{user}:{pa}:{pb}".encode()).hexdigest()[-1], 16) % 2
        A, B = ((pb, rb), (pa, ra)) if flip else ((pa, ra), (pb, rb))
        used |= {pa, pb}; n += 1
        cands.append({"id": f"cfcolor_train:{user}:{pa}:{pb}", "state": {"history": h, "rating_scale": "1 = lowest preference; 5 = highest preference"},
                      "qk": "preference", "q": {"type": "choice", "instructions": "Which five-color palette would this user rate more highly, given their earlier ratings?",
                                                 "criteria": {"A": _hex(rgb[A[0] - 1]), "B": _hex(rgb[B[0] - 1])}},
                      "gold": "A" if A[1] > B[1] else "B", "meta": {"user": user, "rating_gap": abs(ra - rb), "split": "train_vec"}})
REP["cfcolor_build"] = {**cc, "train_rows": int(train.shape[0]), "test_rows_read_for_id_ban_only": int(test.shape[0]), "candidates": len(cands)}
print("CFCOLOR", json.dumps(REP["cfcolor_build"]), flush=True)
emit("cfcolor", cands, "NC-SA", "CC BY-NC-SA 2.5 CA (project page dgp.toronto.edu/~donovan/color: 'Data and code are released under the Creative Commons BY-NC-SA license'); TRAIN ratings only",
     lambda x: (json.dumps(x["q"]["criteria"]), ()), key_opts=True, balance=lambda x: x["gold"])

# ------------------------------------------------------------------ HellaSwag official train (kit normalize_direct.knowledge twin)
clean = lambda s: " ".join(str(s).split())
cands = []
for i, r in enumerate(pq.read_table(f"{S}/hellaswag_train.parquet").to_pylist()):
    ends = [clean(x) for x in r["endings"]]
    if len(ends) != 4 or len(set(ends)) != 4 or str(r["label"]) not in "0123" or str(r["label"]) == "":
        continue
    cands.append({"id": f"HellaSwag:train:{i}:{r['ind']}", "state": {}, "qk": "q1",
                  "q": {"type": "choice", "instructions": "Which continuation is most plausible?\n" + clean(r["ctx"]), "criteria": dict(zip("ABCD", ends))},
                  "gold": "ABCD"[int(r["label"])], "meta": {"ind": r["ind"], "source_id": r.get("source_id"), "split_type": r.get("split_type")}})
emit("hellaswag", cands, "clean", "MIT (Rowan/hellaswag @218ec52 card: 'MIT https://github.com/rowanz/hellaswag/blob/master/LICENSE'); official train split",
     lambda x: (x["q"]["instructions"].split("\n", 1)[1] + "\n" + "\n".join(x["q"]["criteria"].values()), (x["q"]["instructions"],)), key_opts=True)

# ------------------------------------------------------------------ BANKING77 official train (kit normalize_direct.banking77 twin, 77-way)
labels = json.load(open(f"{S}/banking77_categories.json")); assert len(labels) == 77
cands = []
with open(f"{S}/banking77_train.csv", newline="", encoding="utf-8") as f:
    for i, r in enumerate(csv.DictReader(f)):
        assert r["category"] in labels
        cands.append({"id": f"BANKING77:train:{i}", "state": {}, "qk": "q1",
                      "q": {"type": "choice", "instructions": "Classify the banking intent of this user request:\n" + r["text"], "criteria": {f"option_{j}": v for j, v in enumerate(labels)}},
                      "gold": f"option_{labels.index(r['category'])}", "meta": {"category": r["category"]}})
emit("banking77", cands, "clean", "CC BY 4.0 (PolyAI-LDN/task-specific-datasets LICENSE at the pinned commit); official train.csv",
     lambda x: (x["q"]["instructions"].split("\n", 1)[1], (x["q"]["instructions"], x["q"]["instructions"].split("\n", 1)[1])), balance=lambda x: x["meta"]["category"])

# ------------------------------------------------------------------ GSM8K official train, answer-groups-v1 distractors over TRAIN golds (kit gsm8k_v3 rule)
gs = pq.read_table(f"{S}/gsm8k_train.parquet").to_pylist(); golds = {}
for i, r in enumerate(gs):
    try:
        golds[i] = AG.gold_of(r["answer"])
    except Exception:
        pass
sets = AG.option_sets(golds); cands = []
for i in sorted(golds):
    for count in (10, 4):
        ordered, correct = AG.options(golds[i], sets[i][count], count, i)
        cands.append({"id": f"GSM8K:train:{i}:{count}", "state": {"question": gs[i]["question"], "task": f"GSM8K deterministic {count}-choice numeric selection"}, "qk": "answer",
                      "q": {"type": "choice", "instructions": "Choose the numeric answer to the problem in state. Do not provide reasoning. This is a named multiple-choice adaptation of GSM8K.",
                            "criteria": {f"option_{j}": v for j, v in enumerate(ordered)}}, "gold": f"option_{correct}", "meta": {"track": count, "group": sets[i]["group"]}})
REP["gsm8k_build"] = {"train_rows": len(gs), "golds": len(golds), "groups": len({s["group"] for s in sets.values()})}
emit("gsm8k_ag", cands, "clean", "MIT (openai/gsm8k @740312a card); official train split; distractors = golds of other TRAIN problems (kit answer-groups-v1 rule)",
     lambda x: (x["state"]["question"], (x["state"]["question"],)), key_opts=True, balance=lambda x: x["meta"]["track"])

# ------------------------------------------------------------------ SATA wording (kit normalize_text.sata twin): MultiRC train
def sata_cands(item_id, paragraph, question, choices, meta):
    ch = list(choices); random.Random(f"{a.seed}:SATA:{item_id}").shuffle(ch)
    out = []
    for j, (text, label) in enumerate(ch):
        out.append({"id": f"{item_id}:option_{j}", "state": {"paragraph": paragraph, "question": question, "options": [t for t, _ in ch]}, "qk": f"option_{j}",
                    "q": {"type": "choice", "instructions": f"Question: {question}\nDoes this candidate correctly answer the question?\nCandidate: {text}", "criteria": {"no": "No", "yes": "Yes"}},
                    "gold": "yes" if label else "no", "item": item_id, "meta": {**meta, "n_options": len(ch), "n_true": sum(1 for _, l in ch if l)}})
    return out


cands = []; mc = collections.Counter()
for r in jl(f"{S}/multirc/MultiRC/train.jsonl"):
    p = r["passage"]; text = re.sub(r"<b>Sent \d+: </b>", "", p["text"]); text = clean(re.sub(r"<br\s*/?>", " ", text))
    for q in p["questions"]:
        ans = [(clean(x["text"]), bool(x["label"])) for x in q["answers"]]
        if len({BM.norm_text(t) for t, _ in ans}) != len(ans) or not (2 <= len(ans) <= 15) or not any(l for _, l in ans) or all(l for _, l in ans):
            mc["skip_item"] += 1; continue
        mc["items"] += 1
        cands += sata_cands(f"MultiRC:train:{r['idx']}:{q['idx']}", text, clean(q["question"]), ans, {"source": "MultiRC train"})
REP["sata_mrc_build"] = dict(mc)
emit("sata_mrc", cands, "states no licence", "MultiRC (Khashabi et al., 2018) train via the SuperGLUE v2 distribution; dataset page states no licence (as FS1's train_sata_multirc)",
     lambda x: (x["state"]["paragraph"] + "\n" + x["state"]["question"], (x["state"]["paragraph"],)), group_of=lambda x: x["item"])

# ------------------------------------------------------------------ SATA wording: GoEmotions train (multi-label analogue)
EMO = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
       "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]
card = open(f"{D}/lic/goemotions_README.md", encoding="utf-8").read()
m = re.findall(r"'(\d+)': (\w+)", card)
names = {}
for k, v in m:
    names.setdefault(int(k), v)
assert [names.get(i) for i in range(28)] == EMO, [names.get(i) for i in range(28)]
cands = []; gc = collections.Counter()
for r in pq.read_table(f"{S}/goemotions_train.parquet").to_pylist():
    labs = sorted(set(int(x) for x in r["labels"]))
    if len(labs) < 2 or 27 in labs or len(r["text"].split()) < 6 or "[NAME]" in r["text"] or "[RELIGION]" in r["text"]:
        gc["skip"] += 1; continue
    rr = random.Random(f"{a.seed}:goemo:{r['id']}")
    dis = rr.sample([i for i in range(27) if i not in labs], rr.choice([3, 4, 5]))
    ch = [(EMO[i], True) for i in labs] + [(EMO[i], False) for i in dis]
    gc["items"] += 1
    cands += sata_cands(f"GoEmotions:train:{r['id']}", r["text"], "Which of these emotions does the writer of this comment express?", ch, {"source": "GoEmotions train"})
REP["sata_goemo_build"] = dict(gc)
emit("sata_goemo", cands, "clean", "Apache-2.0 (google-research-datasets/go_emotions card at the pinned revision); official train split, comments with >= 2 emotion labels",
     lambda x: (x["state"]["paragraph"], (x["state"]["paragraph"],)), group_of=lambda x: x["item"])

# ------------------------------------------------------------------ outputs: rows (FS5's 23 keys), meta sidecar, report
tot = 0; hh = hashlib.sha256()
with open(f"{D}/out/fs5h_rows.jsonl", "w", encoding="utf-8") as f, open(f"{D}/out/fs5h_rows.meta.jsonl", "w", encoding="utf-8") as fm:
    for pool, rows in POOL_ROWS.items():
        for r, m in rows:
            l = json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"; f.write(l); hh.update(l.encode()); fm.write(json.dumps(m, ensure_ascii=False) + "\n"); tot += 1
REP["rows"] = tot; REP["rows_sha256"] = hh.hexdigest(); REP["by_pool"] = {k: len(v) for k, v in POOL_ROWS.items()}
REP["tokens"] = sum(r["n_tokens"] for rows in POOL_ROWS.values() for r, _ in rows)
json.dump(REP, open(f"{D}/out/fs5h_report.json", "w"), indent=1)
print("FS5H_DONE", tot, REP["by_pool"], "rows_sha256", hh.hexdigest(), "tokens", REP["tokens"], flush=True)
