"""row-level checks of the UG manifests with the trainer's own code (qv_train.py f555b7e9: its rows_sha256 expression and its Rows class, exec'd
unchanged) and the helper fork's never-train name rule (vh_facts.never_source). usage: vg_chk.py <qv_train.py> <vh_facts.py or -> <manifest> [...]"""
import hashlib, json, random, re, sys
from collections import Counter
from pathlib import Path
G = Path("/data/shared/visdata_g")
src = Path(sys.argv[1]).read_text()
assert hashlib.md5(src.encode()).hexdigest().startswith("f555b7e9"), "not the stock trainer"
a, b = src.index("class Rows:"), src.index("\n\n\ndef encode(")
ns = {"json": json, "random": random}
exec(src[a:b], ns)  # the trainer's Rows class, unchanged
never = None
if sys.argv[2] != "-":
    hs = Path(sys.argv[2]).read_text()
    i, j = hs.index("NEVER = re.compile("), hs.index("\n\n\ndef never_source(r):")
    k = hs.index("\n\n\n", j + 10)
    hn = {"re": re, "Path": Path}
    exec(hs[i:k], hn)
    never = hn["never_source"]
LET = "ABCDEFGHIJ"
for m in sys.argv[3:]:
    p = G / "manifests_g" / f"{m}.jsonl"
    sha = hashlib.sha256(Path(p).read_bytes()).hexdigest()  # the trainer's own expression (qv_train.py line 193)
    hdr = json.loads((G / "manifests_g" / f"{m}.header.json").read_text())
    R = ns["Rows"](str(p), 20261114, 64, 1, 0)
    n = len(R.rows)
    keys, bad, held = Counter(), 0, Counter()
    for r in R.rows:
        keys[tuple(sorted(k for k in r if k not in ("extra", "soft")))] += 1
        bad += not (r["gold"] in LET[:len(r["options"])] and r["images"] and len(r["images"]) == len(r["img_sha256"]) == len(r["img_phash"])
                    and len(set(r["options"])) == len(r["options"]) and 2 <= len(r["options"]) <= 10)
        if never:
            w = never(r)
            if w:
                held[(r["source"], w)] += 1
    new = set(hdr["new_sources"])
    smoke = Counter(r["source"] for s in range(6) for r in R.step_rows(s) if r["source"] in new)
    steps = -(-n // 64)
    print(json.dumps({"manifest": m, "rows": n, "steps": steps, "rows_div_64": n % 64 == 0, "trainer_rows_sha256": sha, "header_rows_sha256": hdr["rows_sha256"],
                      "sha_equal": sha == hdr["rows_sha256"], "key_sets": len(keys), "invalid_rows": bad,
                      "held_out_by_helper_name_rule": {f"{k[0]}: {k[1]}": v for k, v in held.items()},
                      "round_g_rows_in_the_first_6_steps": dict(smoke), "first_step_with_each_new_source": {s: next(i for i in range(steps) if any(r["source"] == s for r in R.step_rows(i))) for s in sorted(new)}}))
