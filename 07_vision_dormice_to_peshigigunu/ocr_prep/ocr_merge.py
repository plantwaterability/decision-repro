"""Merge OCR shards into one cache (sorted by image path), check it covers the image list exactly, print stats.
usage: python ocr_merge.py <image_list.txt> <shard_dir> <out.jsonl>"""
import glob
import json
import statistics
import sys

list_f, sdir, out_f = sys.argv[1:4]
want = [l.rstrip("\n") for l in open(list_f) if l.strip()]
got = {}
for fn in sorted(glob.glob(f"{sdir}/shard_*.jsonl")):
    for l in open(fn):
        r = json.loads(l)
        got[r["image"]] = r
missing = [p for p in want if p not in got]
extra = set(got) - set(want)
if missing or extra:
    print(json.dumps({"error": "coverage", "missing": len(missing), "extra": len(extra), "eg": missing[:3]}))
    sys.exit(1)
with open(out_f, "w") as f:
    for p in sorted(want):
        f.write(json.dumps(got[p]) + "\n")
ok = [[x for x in got[p]["lines"] if x["score"] >= 0.5] for p in want]
chars = [sum(len(x["text"]) for x in ls) for ls in ok]
print(json.dumps({"images": len(want), "errors": sum("err" in got[p] for p in want), "with_text": sum(c > 0 for c in chars),
                  "with_text_frac": round(sum(c > 0 for c in chars) / len(want), 4),
                  "median_chars_when_text": statistics.median([c for c in chars if c] or [0]),
                  "over_1200_chars": sum(c > 1200 for c in chars), "over_60_lines": sum(len(ls) > 60 for ls in ok),
                  "median_ms": round(statistics.median(got[p]["ms"] for p in want), 1)}))
