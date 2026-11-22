"""Rebuild the restricted Winoground part of the vision suite on this node from the raw files (eval-only, research licence; never
leaves eval_restricted or the restricted S3 prefix). Uses the kit's own full_winoground and row writer; the raw files are read from
disk instead of the gated Hub repo, and their sha256 must equal the kit's pins.
usage: wg_rebuild.py <raw_dir holding data/examples.jsonl and data/images.zip> <out_root> <expected wg_rows sha256>"""
import hashlib, json, sys
from pathlib import Path
from decision_index.vision import build, rows as R

raw, root, want = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]

class LocalCtx(build.Ctx):
    def fetch(self, file, repo=None, revision=None):
        p = raw / file
        self.check(file, build.sha256_file(p))
        return str(p)

c = LocalCtx("full_winoground", root, root.parent / "wg_downloads_unused")
rep = c.write(build.full_winoground(c))
print("BUILD", json.dumps(rep))
assert all(v["match"] for v in rep.values()), rep
rows = [R.to_row(root, it) for it in R.select("full_winoground", R.load_items(root, "full_winoground"))]
out = root / "wg_rows.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(R.dumps(r) + "\n")
got = hashlib.sha256(out.read_bytes()).hexdigest()
s = R.summarize(rows)["Winoground"]; pin = build.spec.pins()["benchmarks"]["Winoground"]
print("ROWS", len(rows), got, "pins", {k: s.get(k) == v for k, v in pin.items()})
assert len(rows) == 800 and got == want and all(s.get(k) == v for k, v in pin.items()), (got, want)
print("WG_SUITE_OK")
