#!/usr/bin/env python3
"""NC re-host sweep (stock_decision_v1): which candidate families carry verbatim text of known NC-licensed NLI corpora?
Reference corpora (HF, pinned revision): facebook/anli (CC BY-NC 4.0, premises + hypotheses) and SICK (CC BY-NC-SA 3.0, sentence
pairs). Every reference text with >= 12 words (and, for long texts, every >= 12-word sentence) is hashed with EXACTLY the Stage 0
text-unit normalisation (sd0_textkeys: NFKC, casefold, non-word runs -> one space); the hashes are intersected with the text-unit
keys of every candidate pool (stage0 work/keys/keys_<pool>.npz). Short reference texts (< 12 words: most ANLI / SICK hypotheses)
are not indexed, so a family is counted by its shared premises / long sentences.
  python3 sd0_nc_rehost_sweep.py --keys keys --pools p1,p2 --units units_pools.parquet --textkeys sd0_textkeys.py --out sweep.json"""
import argparse
import collections
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import xxhash
from huggingface_hub import HfApi, hf_hub_download

REFS = [("facebook/anli", ("premise", "hypothesis"), "CC BY-NC 4.0"), ("RobZamp/sick", ("sentence_A", "sentence_B"), "CC BY-NC-SA 3.0")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", required=True)
    ap.add_argument("--pools", required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--textkeys", required=True)
    ap.add_argument("--cache", default="hfcache")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("tk", a.textkeys)
    TK = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(TK)
    api = HfApi()
    ref_hash, ref_meta = {}, {}
    for repo, cols, lic in REFS:
        try:
            info = api.dataset_info(repo)
        except Exception as e:                         # noqa: BLE001
            ref_meta[repo] = {"error": f"{type(e).__name__}"}
            continue
        files = [f for f in api.list_repo_files(repo, repo_type="dataset", revision=info.sha) if f.endswith((".parquet", ".csv", ".tsv", ".jsonl", ".txt"))]
        n = 0
        for f in files:
            p = hf_hub_download(repo, f, repo_type="dataset", revision=info.sha, cache_dir=a.cache)
            if f.endswith(".parquet"):
                df = pd.read_parquet(p)
            elif f.endswith((".tsv", ".txt")):
                df = pd.read_csv(p, sep="\t", on_bad_lines="skip")
            elif f.endswith(".csv"):
                df = pd.read_csv(p, on_bad_lines="skip")
            else:
                df = pd.read_json(p, lines=True)
            for c in cols:
                if c not in df:
                    continue
                for t in df[c].dropna().astype(str):
                    for u in TK.units_of(t):
                        ref_hash.setdefault(xxhash.xxh64_intdigest(u.encode()), repo)
                    n += 1
        ref_meta[repo] = {"revision": info.sha, "files": files, "texts": n, "licence": lic}
    rh = np.array(sorted(ref_hash), dtype=np.uint64)
    u = pd.read_parquet(a.units, columns=["rk", "canonical_family", "upstream_source", "pool_assign"]).drop_duplicates("rk")
    fam_of = dict(zip(u.rk, u.canonical_family))
    res = collections.defaultdict(lambda: collections.Counter())
    for t in a.pools.split(","):
        d = np.load(os.path.join(a.keys, f"keys_{t}.npz"))
        import gzip
        with gzip.open(os.path.join(a.keys, f"recs_{t}.jsonl.gz"), "rt") as g:
            rks = [json.loads(l)[0] for l in g]
        th, tr = d["tu_hash"], d["tu_rec"]
        m = np.isin(th, rh)
        hit_recs = np.unique(tr[m])
        for ri in hit_recs:
            rk = rks[ri]
            res[fam_of.get(rk, f"{t}:(not in units)")][t] += 1
        res["__total__"][t] = int(len(hit_recs))
    out = {"refs": ref_meta, "families_with_verbatim_reference_text": {k: dict(v) for k, v in sorted(res.items(), key=lambda x: -sum(x[1].values()))}}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out)[:3000])


if __name__ == "__main__":
    main()
