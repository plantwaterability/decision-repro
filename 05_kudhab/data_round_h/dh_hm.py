#!/usr/bin/env python3
"""dh_hm.py (DH fork, round H, 2026-11-16): GROUPED multi-question requests (one state + 2-15 questions), the shape of SATA-Bench, ACOS,
ContractNLI, BFCL, ToolRet chunk sets and PhishNChips in the board kit. karpo's block layout lists every question's instructions in one
shared block, then one branch per question; question KEYS are not rendered, so only the question order and instructions matter.
Sources: (a) round-H SATA-wording items (pools sata_mrc / sata_goemo: all candidates of one select-all item, in the item's shuffled option
order, keys option_j as the kit's SATA rows); (b) FS5 / H0 rows that share one identical state (canonical JSON) inside one source family,
regrouped in decision_uid order (numeric-aware), chunks of <= 15. The single-question rows stay in their own files (nothing is removed).
Rows of the round-H drop list are left out.
  dh_hm.py FS5.jsonl REQ_FS5.jsonl.gz REQ_H0.jsonl.gz H_ROWS.jsonl H_META.jsonl REQ_H.jsonl.gz DROPLIST OUT.jsonl.gz
OUT lines: {"id": "fs5hm:<16 hex>", "n_tokens": est, "source_family": str, "request": {"state", "questions": {qk: {...}}},
            "members": {qk: exposure_id}, "gold": {qk: {option_key: prob}}}"""
import collections, gzip, hashlib, json, re, sys
fs5, req_fs5, req_h0, h_rows, h_meta, req_h, droplist, out = sys.argv[1:9]
MAXQ = 15
drop = {l.strip() for l in open(droplist) if l.strip() and not l.startswith("#")}
REQ = {}
for p in (req_fs5, req_h0, req_h):
    for l in gzip.open(p, "rt"):
        r = json.loads(l); REQ[r["id"]] = r["request"]
print("requests", len(REQ), flush=True)


def natkey(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def canon(state):
    return hashlib.sha1(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


G = collections.defaultdict(list); st = collections.Counter()
# (b) FS5 rows (and H0 rows: same manifest) grouped by (source family, identical state)
for l in open(fs5):
    r = json.loads(l); eid = r["exposure_id"]
    if eid in drop:
        st["dropped_droplist"] += 1; continue
    q = REQ.get(eid)
    if q is None:
        st["no_request"] += 1; continue
    if q["state"] in ("", {}, None):
        st["empty_state"] += 1; continue
    G[(r["source_family"], canon(q["state"]))].append((r["decision_uid"], eid, q, r["gold"]))
# (a) round-H SATA-wording items: one group per item, option order as built
meta = [json.loads(l) for l in open(h_meta)]
hgold = {}
for l in open(h_rows):
    r = json.loads(l); hgold[r["exposure_id"]] = r["gold"]
HG = collections.defaultdict(list)
for m in meta:
    if m["pool"] in ("sata_mrc", "sata_goemo") and m.get("group"):
        HG[("fs5h_" + m["pool"], m["group"])].append((m["qk"], m["exposure_id"]))
n = 0; by = collections.Counter(); byq = collections.Counter(); hs = hashlib.sha256()
with gzip.open(out, "wt") as f:
    def write(src, qs, members, gold, state):
        global n
        rid = "fs5hm:" + hashlib.sha1((src + "|" + "|".join(sorted(members.values()))).encode()).hexdigest()[:16]
        req = {"state": state, "questions": qs}
        line = json.dumps({"id": rid, "n_tokens": len(json.dumps(req, ensure_ascii=False)) // 3 + 40 * len(qs), "source_family": src,
                           "request": req, "members": members, "gold": gold}, ensure_ascii=False)
        f.write(line + "\n"); hs.update((line + "\n").encode()); n += 1; by[src] += 1; byq[src] += len(qs)
    for (src, item), mem in sorted(HG.items()):
        mem = sorted(mem, key=lambda x: natkey(x[0]))
        if not 2 <= len(mem) <= MAXQ:
            st["h_item_size_out_of_range"] += 1; continue
        qs, members, gold = {}, {}, {}
        state = REQ[mem[0][1]]["state"]
        ok = True
        for qk, eid in mem:
            rq = REQ.get(eid)
            if rq is None or canon(rq["state"]) != canon(state):
                ok = False; break
            qs[qk] = rq["questions"]["q"]; members[qk] = eid; gold[qk] = hgold[eid]
        if not ok:
            st["h_item_state_mismatch"] += 1; continue
        write(src, qs, members, gold, state)
    for (src, _), mem in sorted(G.items()):
        if len(mem) < 2:
            continue
        mem = sorted(mem, key=lambda x: natkey(x[0]))
        seenq = set(); uniq = []
        for uid, eid, q, gd in mem:
            qq = q["questions"]["q"]
            k = json.dumps([qq.get("instructions"), qq["criteria"]], sort_keys=True, ensure_ascii=False)
            if k in seenq:
                st["dup_question_in_group"] += 1; continue
            seenq.add(k); uniq.append((uid, eid, q, gd))
        if len(uniq) < 2:
            st["group_one_distinct_question"] += 1; continue
        chunks = [uniq[i:i + MAXQ] for i in range(0, len(uniq), MAXQ)]
        if len(chunks) > 1 and len(chunks[-1]) < 2:          # a tail of one question borrows the previous chunk's last one
            chunks[-1] = [chunks[-2][-1]] + chunks[-1]; chunks[-2] = chunks[-2][:-1]
        for ch in chunks:
            if len(ch) < 2:
                continue
            qs = {f"q{j}": x[2]["questions"]["q"] for j, x in enumerate(ch)}
            write(src, qs, {f"q{j}": x[1] for j, x in enumerate(ch)}, {f"q{j}": x[3] for j, x in enumerate(ch)}, ch[0][2]["state"])
print("HM", n, dict(st), flush=True)
for s, v in by.most_common(60):
    print("  %-44s requests %6d questions %7d" % (s[:44], v, byq[s]))
json.dump({"requests": n, "questions": sum(byq.values()), "stats": dict(st), "by_source": {s: {"requests": v, "questions": byq[s]} for s, v in by.items()},
           "content_sha256": hs.hexdigest()}, open(out.replace(".jsonl.gz", ".stats.json"), "w"), indent=1)
