#!/usr/bin/env python3
"""v8 final assembly: v8_prejoin.jsonl's new rows get their E1 parent_preservation from preservation.py's joined units (1:1 by
_row_index = position among the new rows, same order); v7.5's rows keep theirs byte for byte. Header re-stamped, trainer reader check,
q sanity on the new rows.  python3 assemble_v8.py PREJOIN JOINED BASE_ROWS OUT V75_HEADER"""
import hashlib, json, math, sys
pre, joined, nbase, out, v75h = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
E1ID = "c92e70ea0dcbfb4d62e04164e7fe48cff4b292b6b47175ae94b4f5987f63eeb3"
pp = {}
for l in open(joined):
    j = json.loads(l); pp[j["_row_index"]] = (j["decision_uid"], j["prompt_sha256"], j["parent_preservation"])
h = hashlib.sha256(); n = with_parent = new_joined = 0; uniq = {}
with open(out, "w", encoding="utf-8") as g:
    for i, l in enumerate(open(pre, encoding="utf-8")):
        if i >= nbase:
            r = json.loads(l); k = i - nbase
            uid, ps, p = pp[k]
            assert uid == r["decision_uid"] and ps == r["prompt_sha256"], f"new row {k} identity"
            r["parent_preservation"] = p; new_joined += 1
            l = json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
        else:
            r = json.loads(l)
        g.write(l); h.update(l.encode()); n += 1
        with_parent += r.get("parent_preservation") is not None
        uniq.setdefault(r["decision_uid"], float(r.get("weight", 1.0)))
assert new_joined == len(pp), (new_joined, len(pp))
base_h = json.load(open(v75h))
hdr = {"manifest_header": True, "schema_version": base_h["schema_version"], "renderer_sha256": base_h["renderer_sha256"], "rows": n,
       "rows_sha256": h.hexdigest(), "total_decision_weight": float(sum(uniq.values())),
       "parent_preservation": {"parent": "E1 (s3 esbar_v5/export)", "parent_revision": E1ID, "rows_with_parent": with_parent,
                               "source_manifest_rows_sha256": json.load(open(pre + ".header.json"))["rows_sha256"],
                               "q": "softmax of E1's letter logits over the shown options (fullft_train v8 --extract-features, policy B, pack 8192); "
                                    "v7.5 rows keep v7.4's join, new rows joined in the v8 prep"},
       "v8_derivation": {"from": "esbar_stage1_v75.jsonl (rows_sha256 7146db56...)", "steps": math.ceil(n / 64)}}
json.dump(hdr, open(out + ".header.json", "w"), indent=2)
sys.path[:0] = ["/tmp/c2e/esbar_code_v2e/trainers", "/tmp/c2e/esbar_code_v2e/engine", "/tmp/c2e/esbar_code_v2e"]
import fullft_train as ft
hh, rows = ft.read_manifest(out)
acc = ent = m = 0
for r in rows[nbase:]:
    q, gd = r["parent_preservation"]["probabilities"], r["gold"]
    m += 1; acc += max(q, key=q.get) == max(gd, key=gd.get); ent += -sum(v * math.log(v) for v in q.values() if v > 0)
print(json.dumps({"rows": n, "rows_sha256": hdr["rows_sha256"], "total_decision_weight": hdr["total_decision_weight"], "with_parent": with_parent,
                  "new_joined": new_joined, "steps": math.ceil(n / 64), "READ_MANIFEST_OK": len(rows),
                  "new_rows_E1_argmax_eq_gold": round(acc / m, 4), "new_rows_mean_entropy": round(ent / m, 4)}))
