"""numbers for the delivery block: per-source pool table, per-subtask rows, list / tar md5, manifest headers. Prints one JSON line."""
import glob, hashlib, json, os, subprocess
from collections import Counter
F = "/data/shared/visdata_g"
SF = "s3://YOUR_BUCKET/jev-race/engineering/vision/recipe_g"
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()  # noqa: E731
agg = {}
for l in open(F + "/pkg/MD5SUMS"):
    m, n = l.split()
    agg[n] = m
ps = json.load(open(F + "/manifests_g/pool_stats.json"))
src = {}
for s, st in ps["round_g_pool"].items():
    p = f"{F}/lists/{s}.jsonl"
    rows = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
    src[s] = {"stats": st, "line": ps["licences"][s][0], "licence": ps["licences"][s][1], "family": rows[0]["family"] if rows else None,
              "list_md5": md5(p) if rows else None, "tar_md5": agg.get(f"img_{s}.tar"), "subtask": dict(Counter(r["subtask"] for r in rows)),
              "n_options": dict(sorted(Counter(len(r["options"]) for r in rows).items())), "gold": dict(sorted(Counter(r["gold"] for r in rows).items())),
              "images": len({q for r in rows for q in r["images"]}),
              "kinds": dict(Counter((r.get("extra") or {}).get("kind") for r in rows if (r.get("extra") or {}).get("kind"))),
              "count_hist": dict(sorted(Counter((r.get("extra") or {}).get("count") for r in rows if "count" in (r.get("extra") or {})).items())),
              "groups": dict(Counter((r.get("extra") or {}).get("group") for r in rows if (r.get("extra") or {}).get("group"))),
              "constructs": dict(Counter((r.get("extra") or {}).get("construct") for r in rows if (r.get("extra") or {}).get("construct")))}
ls = subprocess.run(["aws", "s3", "ls", SF + "/img/"], capture_output=True, text=True).stdout
for l in ls.splitlines():
    p = l.split()
    if len(p) == 4 and p[3].startswith("img_") and p[3][4:-4] in src:
        src[p[3][4:-4]]["tar_bytes"] = int(p[2])
man = {}
for h in sorted(glob.glob(F + "/manifests_g/*.header.json")):
    d = json.load(open(h))
    n = d["manifest"]
    man[n] = {k: d[k] for k in ("rows", "steps_at_64", "rows_sha256", "added_rows", "added_per_source", "added_per_family", "added_per_subtask", "per_family",
                                "per_line", "hm_yes_share", "hm_per_source_yes", "repeat_factor", "exact_hash_hits_in_manifest_vs_proxy_rbh_charxiv_nevertrain",
                                "added_images_missing", "base", "new_sources", "notes")}
    man[n]["file_md5"] = md5(f"{F}/manifests_g/{n}.jsonl")
    man[n]["md5_in_aggregate"] = agg.get(f"{n}.jsonl")
out = {"sources": src, "manifests": man, "dedupe_against": ps["dedupe_against"], "aggregate_md5sums": {"md5": md5(F + "/pkg/MD5SUMS"), "lines": len(agg)},
       "s3_objects": subprocess.run(["aws", "s3", "ls", "--recursive", "--summarize", SF + "/"], capture_output=True, text=True).stdout.strip().splitlines()[-2:],
       "bad": subprocess.run("grep -c S3_BAD /data/shared/visdata_g/logs/g_package.log", shell=True, capture_output=True, text=True).stdout.strip()}
print("REPORT_JSON " + json.dumps(out, ensure_ascii=False))
