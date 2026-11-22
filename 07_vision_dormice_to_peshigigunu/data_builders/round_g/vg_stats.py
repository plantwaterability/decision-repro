"""delivery-block numbers from the built manifests (no image opened)"""
import json
from collections import Counter
G = "/data/shared/visdata_g/manifests_g/"
LET = "ABCDEFGHIJ"
for m in ("ug1_nc", "ug1c_nc"):
    hdr = json.load(open(G + m + ".header.json")); new = set(hdr["new_sources"])
    bad, cnt, kinds, cons, sub, gold, nopt, yn, imgs = Counter(), {}, Counter(), Counter(), Counter(), {}, {}, Counter(), {}
    n = 0
    for l in open(G + m + ".jsonl"):
        r = json.loads(l); n += 1; s = r["source"]
        ok = (r["gold"] in LET[:len(r["options"])] and r["images"] and len(r["images"]) == len(r["img_sha256"]) == len(r["img_phash"]) and len(set(r["options"])) == len(r["options"]))
        if not ok:
            bad[(s, s in new and r["id"] in ())] += 1
        if s in new and (m == "ug1c_nc") == (s == "chartqa_h") or (m == "ug1_nc" and s in new):
            x = r.get("extra") or {}
            sub[(s, r["subtask"])] += 1
            gold.setdefault(s, Counter())[r["gold"]] += 1
            nopt.setdefault(s, Counter())[len(r["options"])] += 1
            imgs.setdefault(s, set()).update(r["images"])
            if "count" in x: cnt.setdefault(s, Counter())[x["count"]] += 1
            if "kind" in x: kinds[(s, x["kind"])] += 1
            if "construct" in x: cons[x["construct"]] += 1
            if x.get("group"): kinds[(s, x["group"])] += 1
    print(json.dumps({"manifest": m, "rows": n, "invalid_by_source": {k[0]: v for k, v in bad.items()}, "subtask": {f"{k[0]}|{k[1]}": v for k, v in sub.items()},
                      "gold": {k: dict(sorted(v.items())) for k, v in gold.items()}, "n_options": {k: dict(sorted(v.items())) for k, v in nopt.items()},
                      "images": {k: len(v) for k, v in imgs.items()}, "count_hist": {k: dict(sorted(v.items())) for k, v in cnt.items()},
                      "kinds": {f"{k[0]}|{k[1]}": v for k, v in kinds.items()}, "fhm_constructs": dict(cons), "ade_cnt_rows_total": sum(1 for _ in [0]) and None}))
