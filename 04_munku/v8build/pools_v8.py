#!/usr/bin/env python3
"""v8 new pools: official TRAIN splits in the DI board's own format (train twins of the kit normalizers: same row builders,
instructions, criteria and label conventions; only the source split differs), rendered with build_manifest_v3.make_render
(renderer R3 ddfaf6c0, as every v7.x pool) and decontaminated with build_manifest_v3.Gate (exact state >= 25 chars + 13-grams
+ short items) against the DI 0.3 public suite AND our internal held-out sets.

  python3 pools_v8.py --kit KIT --code CODE --scripts SCRIPTS --tok TOKDIR --suite SUITE --heldout F1,F2,... --src SRC --out OUTDIR
Writes OUTDIR/<pool>.rows.jsonl (rendered manifest rows, all candidates) + OUTDIR/pools_report.json.
"""
import argparse, collections, csv, gzip, hashlib, json, random, re, sqlite3, sys, unicodedata

ap = argparse.ArgumentParser()
for k in ("kit", "code", "scripts", "tok", "suite", "heldout", "src", "out"):
    ap.add_argument(f"--{k}", required=True)
ap.add_argument("--esci-rows", type=int, default=20000)
ap.add_argument("--seed", type=int, default=20261109)
ap.add_argument("--only", default="", help="comma list of pools to (re)build; default all")
a = ap.parse_args()
sys.path[:0] = [a.kit, a.code, a.scripts]
from decision_index.suite.build.layout import choice_row
from decision_index.suite.build import adapters_added as AA
import build_manifest_v3 as BM
import claude_lmscore as clm
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(a.tok)
REPORT = collections.defaultdict(dict)


def jl(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def parse_prompt(p):
    """manifest prompt -> (state_text, instructions, [option texts])"""
    io_ = p.rfind("\n\nOptions:\n"); iq = p.rfind("\n\nQuestion: ", 0, io_)
    st = p[p.index("<|turn>user\n") + len("<|turn>user\n"):iq]
    ins = p[iq + len("\n\nQuestion: "):io_]
    blk = p[io_ + len("\n\nOptions:\n"):p.rfind("\n\nAnswer with")]
    return st, ins, [m.group(1) for m in re.finditer(r"^[A-Za-z]\) (.*)$", blk, re.M)]


# ---- gate: DI suite + internal held-out sets (DI-format rows, or manifest rows parsed back to state / instructions)
gate = BM.Gate()
n_suite = 0
for f in ("selected-rows", "added-rows", "gsm8k-rows"):
    for r in jl(f"{a.suite}/{f}.jsonl.gz"):
        gate.add(r); n_suite += 1
n_ho = collections.Counter()
for path in [p for p in a.heldout.split(",") if p]:
    for r in jl(path):
        if r.get("manifest_header"):
            continue
        if "prompt" in r:
            st, ins, _ = parse_prompt(r["prompt"])
            r = {"state": st, "questions": {"q": {"instructions": ins}}}
        gate.add(r); n_ho[path.rsplit("/", 1)[-1]] += 1
REPORT["gate"] = {"suite_rows": n_suite, "heldout_rows": dict(n_ho), "exact_keys": len(gate.exact), "grams13": len(gate.grams)}
print("gate", json.dumps(REPORT["gate"]), flush=True)

rng = random.Random(a.seed)
render = BM.make_render(tok, clm, rng, split="train", tag="esbar_v8")


ONLY = set(x for x in a.only.split(",") if x)


def emit(pool, rows, gate_on="state"):
    """DI rows -> gated, rendered manifest rows (one question per row)."""
    if ONLY and pool not in ONLY:
        return
    out, c = [], collections.Counter()
    for r in rows:
        (qk, q), = r["questions"].items()
        st = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False)
        key = st if gate_on == "state" else q["instructions"]
        content = r.get("v8_content") or key            # (as build_manifest_v3's bigk) template-free content for 13-grams
        if not gate.clean(content, exact_keys=(key,)):
            c["dropped_gate"] += 1
            continue
        row = render(f"v8|{pool}|{r['id']}", pool, r["state"], q["instructions"], q["criteria"], r["expected"][qk], pool)
        if row is None:
            c["dropped_over_4096_tokens"] += 1
            continue
        row["v8_meta"] = r.get("v8_meta") or {}
        out.append(row); c["kept"] += 1
    with open(f"{a.out}/{pool}.rows.jsonl", "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    REPORT[pool].update({"candidates": len(rows), **c, "gold": dict(collections.Counter(max(x["gold"], key=x["gold"].get) for x in out))})
    print(pool, json.dumps(REPORT[pool]), flush=True)


# ---- CLINC150+OOS: kit normalize_direct.clinc150 on train / oos_train / val / oos_val
obj = json.load(open(f"{a.src}/clinc_data_full.json"))
labels = sorted({p[1] for p in obj["train"]} | {"oos"}); assert len(labels) == 151
desc = ["out of scope: none of the listed intents" if l == "oos" else l.replace("_", " ") for l in labels]
rows = []
for split in ("train", "oos_train", "val", "oos_val"):
    for i, (text, label) in enumerate(obj.get(split, [])):
        rows.append({**choice_row("CLINC150+OOS", split, i, "Classify the intent of this user request, or choose out of scope if none applies:\n" + text,
                               desc, labels.index(label)), "v8_content": text})
emit("train_clinc150_v8", rows, gate_on="instructions")

# ---- iSarcasmEval task A English: kit normalize_text.isarcasm on train.En.csv (headline track; 'tweet' column = test 'text')
rows = []
with open(f"{a.src}/isarcasm_train.En.csv", encoding="utf-8-sig", newline="") as f:
    for i, row in enumerate(csv.DictReader(f)):
        text = (row.get("tweet") or "").strip()
        if not text or row.get("sarcastic") not in {"0", "1"}:
            continue
        rows.append({"id": f"iSarcasmEval-A-En:train:{i}", "state": text,
                     "questions": {"sarcastic": dict(type="choice", instructions="Is this text intended to be sarcastic?", criteria={"no": "No", "yes": "Yes"})},
                     "expected": {"sarcastic": "yes" if row["sarcastic"] == "1" else "no"}})
emit("train_isarcasm_v8", rows)

# ---- New Yorker caption matching: kit adapters_added.newyorker on config `matching` (split 0) TRAIN only
import pyarrow.parquet as pq
rows = []
for p in sorted(__import__("glob").glob(f"{a.src}/newyorker_train-*.parquet")):
    t = pq.read_table(p, columns=["instance_id", "image_location", "image_description", "image_uncanny_description", "entities", "caption_choices", "label", "contest_number"])
    for r in t.to_pylist():
        keys = AA.letters(len(r["caption_choices"]))
        rows.append({"id": f"64:{r['instance_id']}", "state": {"scene": r["image_location"], "description": r["image_description"], "uncanny_description": r["image_uncanny_description"], "entities": r["entities"]},
                     "questions": {"q": dict(type="choice", instructions=AA.NEWYORKER_INSTRUCTIONS, criteria=dict(zip(keys, r["caption_choices"])))},
                     "expected": {"q": r["label"]}, "v8_meta": {"contest_number": r["contest_number"]}})
emit("train_newyorker_v8", rows)

# ---- WinoGrande XL train: kit normalize_direct.knowledge (WinoGrande) on the xl TRAIN parquet
rows = []
for p in sorted(__import__("glob").glob(f"{a.src}/winogrande_train-*.parquet")):
    for i, r in enumerate(pq.read_table(p).to_pylist()):
        rows.append({**choice_row("WinoGrande", "train", i, "Which option correctly fills the blank?\n" + r["sentence"], [r["option1"], r["option2"]], int(r["answer"]) - 1), "v8_content": r["sentence"]})
emit("train_winogrande_v8", rows, gate_on="instructions")

# ---- HoVer train: kit adapters_added.hover on hover_train_release_v1.1.json (+ wiki DB), V5 exclusion: no claim whose
# supporting documents or hpqa_id appear in the dev set (= the DI test)
dev = json.load(open(f"{a.src}/hover_dev_release_v1.1.json"))
dev_titles = {t for r in dev for t, _ in r["supporting_facts"]}; dev_hpqa = {r["hpqa_id"] for r in dev}
db = sqlite3.connect(f"file:{a.src}/hover_wiki_wo_links.db?mode=ro", uri=True)
rows, hc = [], collections.Counter()
for r in json.load(open(f"{a.src}/hover_train_release_v1.1.json")):
    titles = list(dict.fromkeys(t for t, _ in r["supporting_facts"]))
    if set(titles) & dev_titles:
        hc["excluded_dev_doc_title"] += 1; continue
    if r.get("hpqa_id") in dev_hpqa:
        hc["excluded_dev_hpqa_id"] += 1; continue
    ev = []
    for t in titles:
        found = db.execute("SELECT id, text FROM documents WHERE id=?", (unicodedata.normalize("NFD", t),)).fetchall()
        if len(found) != 1:
            ev = None; break
        ev.append({"title": t, "text": found[0][1]})
    if ev is None:
        hc["missing_doc"] += 1; continue
    rows.append({"id": f"61:{r['uid']}", "state": {"claim": r["claim"], "evidence": ev},
                 "questions": {"q": dict(type="choice", instructions=AA.HOVER_INSTRUCTIONS, criteria=AA.HOVER_CRITERIA)}, "expected": {"q": r["label"]},
                 "v8_meta": {"num_hops": r["num_hops"], "hpqa_id": r.get("hpqa_id")}})
REPORT["train_hover_v8"].update(dict(hc))
emit("train_hover_v8", rows)

# ---- Amazon ESCI TRAIN (user lifted the never-read rule for the official train split, 11-09): kit normalize_text.esci on
# split == "train"; sampled BY QUERY; zero query overlap with the ESCI test split (query_id AND normalized query text);
# locale mix = the DI suite's ESCI locale mix; label targets E 35 / S 30 / C 15 / I 20 %, <= 4 pairs per query.
import pyarrow.compute as pc
ex = pq.read_table(f"{a.src}/esci_examples.parquet")
test = ex.filter(pc.equal(ex["split"], "test")); train = ex.filter(pc.equal(ex["split"], "train"))
test_qid = set(test["query_id"].to_pylist()); test_qtext = {BM.norm_text(q) for q in test["query"].to_pylist()}
loc_mix = collections.Counter()
for f in ("selected-rows", "added-rows"):
    for r in jl(f"{a.suite}/{f}.jsonl.gz"):
        if r.get("family") == "Amazon-ESCI" or str(r.get("id", "")).startswith("Amazon-ESCI:"):
            loc_mix[(r.get("metadata") or r.get("provenance") or {}).get("locale")] += 1
tot = sum(loc_mix.values())
TARGET = {"E": 0.35, "S": 0.30, "C": 0.15, "I": 0.20}
quota = {(lo, lb): int(round(a.esci_rows * n / tot * TARGET[lb])) for lo, n in loc_mix.items() for lb in TARGET}
byq = collections.defaultdict(list)
for r in train.select(["example_id", "query", "query_id", "product_id", "product_locale", "esci_label"]).to_pylist():
    byq[r["query_id"]].append(r)
qids = sorted(byq); rng_e = random.Random(a.seed + 37); rng_e.shuffle(qids)
ec = collections.Counter(); picked = []; filled = collections.Counter()
for qid in qids:
    if all(filled[k] >= v for k, v in quota.items()):
        break
    prs = byq[qid]
    if qid in test_qid or BM.norm_text(prs[0]["query"]) in test_qtext:
        ec["query_overlap_with_test_skipped"] += 1; continue
    rng_e.shuffle(prs); took = 0
    for r in sorted(prs, key=lambda x: {"C": 0, "S": 1, "I": 2, "E": 3}[x["esci_label"]]):   # rare labels first within a query
        k = (r["product_locale"], r["esci_label"])
        if took < 4 and filled[k] < quota.get(k, 0):
            picked.append(r); filled[k] += 1; took += 1
    ec["queries_used"] += took > 0
pids = {(r["product_locale"], r["product_id"]) for r in picked}
pt = pq.read_table(f"{a.src}/esci_products.parquet", filters=[("product_id", "in", sorted({p for _, p in pids}))])
prod = {}
for r in pt.to_pylist():
    if (r["product_locale"], r["product_id"]) in pids:
        prod[(r["product_locale"], r["product_id"])] = r
crit = {"E": "Exact: the product satisfies the search query.", "S": "Substitute: a product that could substitute for the requested product.",
        "C": "Complement: a product that complements the requested product.", "I": "Irrelevant: the product does not address the requested product need."}
rows = []
for r in sorted(picked, key=lambda x: x["example_id"]):
    pr = prod.get((r["product_locale"], r["product_id"]))
    if pr is None or pr.get("product_title") is None:
        ec["missing_product_text"] += 1; continue
    product = {k.removeprefix("product_"): pr[k] for k in ["product_title", "product_description", "product_bullet_point", "product_brand", "product_color"] if pr.get(k) is not None}
    rows.append({"id": f"Amazon-ESCI:train:{r['example_id']}", "state": {"search_query": r["query"], "product": product},
                 "questions": {"answer": dict(type="choice", instructions="Classify the relevance of this product to the search query using the ESCI categories.", criteria=crit)},
                 "expected": {"answer": r["esci_label"]}, "v8_meta": {"query_id": r["query_id"], "locale": r["product_locale"], "product_id": r["product_id"]}})
picked_q = {r["query_id"] for r in picked}
REPORT["train_esci_v8"].update({**ec, "suite_locale_mix": dict(loc_mix), "quota": {f"{k[0]}/{k[1]}": v for k, v in quota.items()},
                                "filled": {f"{k[0]}/{k[1]}": v for k, v in filled.items()}, "train_split_pairs": train.num_rows,
                                "train_queries": len(byq), "queries_sampled": len(picked_q), "test_query_ids_in_sample": len(picked_q & test_qid),
                                "test_query_texts_in_sample": sum(BM.norm_text(byq[q][0]["query"]) in test_qtext for q in picked_q)})
emit("train_esci_v8", rows)
json.dump(REPORT, open(f"{a.out}/pools_report{'_' + '_'.join(sorted(ONLY)) if ONLY else ''}.json", "w"), indent=1)
print("POOLS_V8_DONE", flush=True)
