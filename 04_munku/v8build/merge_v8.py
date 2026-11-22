#!/usr/bin/env python3
"""v8 pre-join assembly.
1) EXACT-MATCH DEDUPE of every row (v7.5's rows + the new pool rows) against the full DI 0.3 public suite, by three keys:
   item  = normalized state + question instructions + sorted option texts (the whole decision);
   leaf  = any >= 25-char normalized content string (string state, the state's string leaves outside tools / evidence / product /
           entities, or the instructions) that occurs in <= 2 suite rows (so templates never match);
   esci  = normalized search query + product title.
   Every flagged row is removed and listed (uid, family, key, suite id).
2) caps: each new family <= --cap (5%) of v8 rows, ESCI <= --esci-cap (8%); seeded sampling by uid hash.
3) writes v8_prejoin.jsonl (v7.5 lines byte-identical, minus removals, then the new rows) and v8_new_units.jsonl (+ headers) for the
   E1 forward (fullft_train --extract-features) and preservation.py join.
"""
import argparse, collections, gzip, hashlib, json, math, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--pools", required=True); ap.add_argument("--suite", required=True)
ap.add_argument("--scripts", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--cap", type=float, default=0.05); ap.add_argument("--esci-cap", type=float, default=0.08)
a = ap.parse_args()
sys.path.insert(0, a.scripts)
from build_manifest_v3 import R3
import unicodedata


def N(s):
    """Unicode-aware normalization (letters / digits in any script kept; case, punctuation and whitespace folded)."""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    return " ".join(re.sub(r"[\W_]+", " ", s, flags=re.UNICODE).split())

SKIP = {"tools", "functions", "function", "evidence", "product", "entities"}


def jl(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                yield l, json.loads(l)


def as_text(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def leaves(x, out):
    if isinstance(x, str):
        out.append(x)
    elif isinstance(x, dict):
        for k, v in x.items():
            if k not in SKIP:
                leaves(v, out)
    elif isinstance(x, list):
        for v in x:
            leaves(v, out)
    return out


def item_key(state_text, instr, opts):
    return hashlib.sha1((N(state_text) + "||" + N(instr) + "||" + "|".join(sorted(N(o) for o in opts))).encode()).hexdigest()


def esci_key(q, t):
    return N(q) + "||" + N(t) if q and t else None


# ---- suite keys
items, leafc, leaf_id, esci = {}, collections.Counter(), {}, {}
for f in ("selected-rows", "added-rows", "gsm8k-rows"):
    for _, r in jl(f"{a.suite}/{f}.jsonl.gz"):
        sid = r.get("id"); st = r.get("state")
        st_text = as_text(st) if st is not None else ""
        cont = set()
        if isinstance(st, str):
            cont.add(N(st))
        elif isinstance(st, (dict, list)):
            cont |= {N(s) for s in leaves(st, [])}
            if isinstance(st, dict) and isinstance(st.get("product"), dict):
                k = esci_key(st.get("search_query"), st["product"].get("title"))
                if k: esci[k] = sid
        bare = len(N(st_text)) < 25                      # instructions carry the content only when the state is empty / short
        for q in (r.get("questions") or {}).values():
            ins = as_text(q.get("instructions") or "")
            items[item_key(st_text, ins, [as_text(v) for v in (q.get("criteria") or {}).values()])] = sid
            if bare:
                cont.add(N(ins))
        for c in cont:
            if len(c) >= 25:
                leafc[c] += 1; leaf_id.setdefault(c, sid)
leafset = {c for c, n in leafc.items() if n <= 2}
print("suite keys: items", len(items), "content leaves (<=2 rows)", len(leafset), "esci pairs", len(esci), flush=True)

AMZ = re.compile(r"Shopping search query \([^)]*\): (.*)\n\nCandidate product:\nTitle: (.*)\n")


def check(row):
    p = row["prompt"]; io_ = p.rfind("\n\nOptions:\n"); iq = p.rfind("\n\nQuestion: ", 0, io_)
    st = p[p.index("<|turn>user\n") + 12:iq]; ins = p[iq + 12:io_]
    opts = [m.group(1) for m in re.finditer(r"^[A-Za-z]\) (.*)$", p[io_ + 11:p.rfind("\n\nAnswer with")], re.M)]
    k = item_key(st, ins, opts)
    if k in items:
        return "item", items[k]
    try:
        sj = json.loads(st)
    except Exception:
        sj = None
    cand = [N(s) for s in leaves(sj, [])] if isinstance(sj, (dict, list)) else [N(st)]
    if len(N(st)) < 25:
        cand.append(N(ins))
    for c in cand:
        if len(c) >= 25 and c in leafset:
            return "leaf", leaf_id[c]
    ek = None
    if isinstance(sj, dict) and isinstance(sj.get("product"), dict):
        ek = esci_key(sj.get("search_query"), sj["product"].get("title"))
    else:
        m = AMZ.search(st)
        if m: ek = esci_key(m.group(1), m.group(2))
    if ek and ek in esci:
        return "esci", esci[ek]
    return None


removed = []
base_keep = []
for l, r in jl(a.base):
    hit = check(r)
    if hit:
        removed.append({"decision_uid": r["decision_uid"], "family": r["source_family"], "origin": "v7.5", "key": hit[0], "suite_id": hit[1]})
    else:
        base_keep.append(l)
print("base rows kept", len(base_keep), "removed", len(removed), flush=True)

POOLS = ["train_clinc150_v8", "train_isarcasm_v8", "train_newyorker_v8", "train_winogrande_v8", "train_hover_v8", "train_esci_v8"]
# existing-row dedupe for the new pools: same decision content already in v7.5 (e.g. the 752 train_clinc150 / 739 train_newyorker rows)
def sec_keys(p):
    io_ = p.rfind("\n\nOptions:\n"); iq = p.rfind("\n\nQuestion: ", 0, io_)
    st = p[p.index("<|turn>user\n") + 12:iq]; ins = p[iq + 12:io_]
    ks = {"si:" + N(st) + "||" + N(ins)}
    try:
        sj = json.loads(st)
    except Exception:
        sj = None
    if isinstance(sj, dict):
        if isinstance(sj.get("claim"), str):
            ks.add("claim:" + N(sj["claim"]))
        if isinstance(sj.get("product"), dict):
            k = esci_key(sj.get("search_query"), sj["product"].get("title"))
            if k: ks.add("esci:" + k)
    else:
        m = AMZ.search(st)
        if m:
            k = esci_key(m.group(1), m.group(2))
            if k: ks.add("esci:" + k)
    return ks
base_items = set()
for l in base_keep:
    base_items |= sec_keys(json.loads(l)["prompt"])
new = {}
dup_base = collections.Counter()
for pool in POOLS:
    keep = []
    for l, r in jl(f"{a.pools}/{pool}.rows.jsonl"):
        hit = check(r)
        if hit:
            removed.append({"decision_uid": r["decision_uid"], "family": pool, "origin": "new", "key": hit[0], "suite_id": hit[1]}); continue
        if sec_keys(r["prompt"]) & base_items:
            dup_base[pool] += 1; continue
        r.pop("v8_meta", None)
        keep.append(r)
    keep.sort(key=lambda r: hashlib.sha256(("v8cap|" + r["decision_uid"]).encode()).hexdigest())   # seeded order for caps
    new[pool] = keep
avail = {k: len(v) for k, v in new.items()}
# ---- caps (fixed point: total depends on the capped counts)
take = dict(avail)
for _ in range(50):
    total = len(base_keep) + sum(take.values())
    nt = {k: min(avail[k], int(math.floor((a.esci_cap if k == "train_esci_v8" else a.cap) * total))) for k in avail}
    if nt == take:
        break
    take = nt
total = len(base_keep) + sum(take.values())
rows_new = [r for k in POOLS for r in new[k][:take[k]]]
# ---- write pre-join manifest + new-rows unit manifest
import hashlib as H
def write(path, lines):
    h = H.sha256(); n = 0; w = {}
    with open(path, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l); h.update(l.encode()); n += 1
            r = json.loads(l); w.setdefault(r["decision_uid"], float(r.get("weight", 1.0)))
    hdr = {"manifest_header": True, "schema_version": "sdv1-stage2-v1", "renderer_sha256": R3, "rows": n, "rows_sha256": h.hexdigest(),
           "total_decision_weight": float(sum(w.values()))}
    json.dump(hdr, open(path + ".header.json", "w"), indent=2)
    return hdr
new_lines = [json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows_new]
h_units = write(f"{a.out}/v8_new_units.jsonl", new_lines)
h_pre = write(f"{a.out}/v8_prejoin.jsonl", base_keep + new_lines)
with open(f"{a.out}/v8_removed.jsonl", "w") as f:
    for x in removed:
        f.write(json.dumps(x) + "\n")
fam_new = collections.Counter(r["source_family"] for r in rows_new)
gold = {k: dict(collections.Counter(max(r["gold"], key=r["gold"].get) for r in new[k][:take[k]])) for k in POOLS}
rep = {"base_rows_v75": len(base_keep) + sum(1 for x in removed if x["origin"] == "v7.5"), "base_removed": sum(1 for x in removed if x["origin"] == "v7.5"),
       "new_removed_suite": sum(1 for x in removed if x["origin"] == "new"), "new_dup_of_existing_v75": dict(dup_base), "new_available": avail, "new_taken": take,
       "rows_v8": total, "steps_v8": math.ceil(total / 64), "share": {k: round(v / total, 4) for k, v in take.items()}, "gold_taken": gold,
       "removed_by_family_key": dict(collections.Counter(f"{x['origin']}:{x['family']}:{x['key']}" for x in removed)),
       "units_header": h_units, "prejoin_header": h_pre}
json.dump(rep, open(f"{a.out}/v8_merge_report.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
