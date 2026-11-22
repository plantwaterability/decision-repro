#!/usr/bin/env python3
"""DocNLI -> ANLI verbatim scan (stock_decision_v1). Downloads facebook/anli (all rounds; train / dev / test) at a pinned revision,
normalises premise and hypothesis (NFKC, whitespace collapsed, case kept), and for every DocNLI row in the given pools extracts the
(text_A, text_B) pairs of the visible state (single pair or packed 'Item X:' blocks) and reports:
  pair_match     text_A == an ANLI premise AND text_B == the hypothesis of that same ANLI pair (verbatim after normalisation)
  premise_match  text_A == some ANLI premise (any hypothesis)
  contained      a normalised ANLI premise of >= 20 words is contained in text_A and that pair's hypothesis equals text_B
Writes anli_scan.json (counts per pool, matched uids, ANLI pin) and prints the summary.
  python3 sd0_anli_scan.py --units units_pools.parquet --pool-dir pools/tasksource_ff_v1 --sets NAME=uids.txt ... --out DIR"""
import argparse
import collections
import json
import os
import re
import unicodedata

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download


def norm(s):
    return " ".join(unicodedata.normalize("NFKC", str(s)).split())


PAIR = re.compile(r"text_A: (.*?)\ntext_B: (.*?)(?=\n\nItem [A-Z]+:|\Z)", re.S)
PASSAGE = re.compile(r"\APassage A:\n(.*?)\n\nPassage B:\n(.*)\Z", re.S)
AB = re.compile(r"\AA: (.*?)\nB: (.*)\Z", re.S)


def pairs_of(st):
    out = PAIR.findall(st)
    for rx in (PASSAGE, AB):
        m = rx.match(st)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--sets", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    api = HfApi()
    info = api.dataset_info("facebook/anli")
    rev = info.sha
    files = [f for f in api.list_repo_files("facebook/anli", repo_type="dataset", revision=rev) if f.endswith(".parquet")]
    pairs, prem = {}, {}
    nrows = collections.Counter()
    for f in files:
        p = hf_hub_download("facebook/anli", f, repo_type="dataset", revision=rev, cache_dir=os.path.join(a.out, "hf"))
        df = pd.read_parquet(p)
        nrows[f] = len(df)
        for pr, hy, uid in zip(df.premise, df.hypothesis, df.uid):
            P, H = norm(pr), norm(hy)
            pairs.setdefault((P, H), f"{f}:{uid}")
            prem.setdefault(P, set()).add(H)
    long_prem = [P for P in prem if len(P.split()) >= 20]
    u = pd.read_parquet(a.units, columns=["uid", "canonical_family", "rk"])
    dn = set(u.loc[u.canonical_family == "doc_nli", "uid"])
    recs = {}
    with open(os.path.join(a.pool_dir, "train", "doc-nli.jsonl")) as f:
        for i, line in enumerate(f, 1):
            recs[i] = json.loads(line)
    res = {"anli": {"repo": "facebook/anli", "revision": rev, "files": dict(nrows), "unique_pairs": len(pairs), "unique_premises": len(prem),
                    "licence": "CC BY-NC 4.0"}, "pools": {}}
    for spec in a.sets:
        name, path = spec.split("=", 1)
        uids = [x.strip() for x in open(path) if x.strip()]
        dnu = [x for x in uids if x in dn]
        c = collections.Counter()
        hits = []
        for uid in dnu:
            line = int(uid.split("|")[2])
            st = recs[line]["state"]
            st = st if isinstance(st, str) else json.dumps(st)
            ps = [(norm(x), norm(y)) for x, y in pairs_of(st)]
            kind = None
            if any(p in pairs for p in ps):
                kind = "pair_match"
            elif any(P in prem for P, _ in ps):
                kind = "premise_match"
            else:
                for P, H in ps:
                    if any(H in prem[q] and q in P for q in long_prem if len(q) <= len(P) and q[:40] in P):
                        kind = "contained"
                        break
            c["doc_nli_rows"] += 1
            c["rows_with_pairs_parsed"] += bool(ps)
            if not ps:
                c["unparsed"] += 1
            if kind:
                c[kind] += 1
                c["any_anli_match"] += 1
                hits.append({"uid": uid, "kind": kind})
        res["pools"][name] = {"decisions": len(uids), **dict(c), "matched_uids": hits}
    json.dump(res, open(os.path.join(a.out, "anli_scan.json"), "w"), indent=1)
    print(json.dumps({"anli": {k: v for k, v in res["anli"].items() if k != "files"},
                      "pools": {k: {kk: vv for kk, vv in v.items() if kk != "matched_uids"} for k, v in res["pools"].items()}}))


if __name__ == "__main__":
    main()
