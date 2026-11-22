#!/usr/bin/env python3
"""jev_real test units whose STATE also occurs in jev_data (any split): the contaminated units.

jev-bench's go_emotions and measuring_hate_speech configs come from the same HF datasets jev_data's
go_emotions and mhs sources are built from, and neither the benchmark's test split nor jev_data's
hash split respects the other, so some test comments sit in jev_data train. jev-bench wraps each
comment in literal double quotes and jev_data does not; states are compared after stripping them and
normalising whitespace and case. Every question of a matching record is listed.

    python3 analysis/claude_overlap.py /data/local/jev_data /data/local/jev_real \
        analysis/jevreal_overlap_qids.json
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
import sys


def norm(s):
    s = re.sub(r"\s+", " ", str(s)).strip()
    if len(s) > 1 and s[0] == s[-1] == '"':
        s = s[1:-1]
    return s.strip().lower()


def main() -> int:
    data, real, out = sys.argv[1:4]
    seen = set()
    for p in glob.glob(os.path.join(data, "*", "*.jsonl")):
        with open(p, encoding="utf-8") as f:
            seen.update(norm(json.loads(line)["state"]) for line in f)
    qids, per = [], collections.Counter()
    for p in glob.glob(os.path.join(real, "test", "*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if norm(r["state"]) in seen:
                    qids.extend(str(q) for q in r["questions"])
                    per[r.get("source", "?")] += 1
    with open(out, "w") as f:
        json.dump(sorted(set(qids)), f)
    print(f"wrote {out}: {len(set(qids))} contaminated units from records {dict(per)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
