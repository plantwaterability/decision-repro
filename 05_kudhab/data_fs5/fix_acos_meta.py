#!/usr/bin/env python3
"""one-off: fs5_build.py (md5 1e5c0690) let ACOS's source line number overwrite the licence-line field of the ACOS sidecar meta
(meta only; manifest rows unaffected). Rename it to src_line and restore line = no-licence-stated (= the builder after the fix)."""
import json, sys
p = sys.argv[1]; out = []
for l in open(p):
    m = json.loads(l)
    if isinstance(m.get("line"), int):
        m["src_line"] = m.pop("line"); m = {"uid": m["uid"], "pool": m["pool"], "line": "no-licence-stated", "licence": m["licence"], **{k: v for k, v in m.items() if k not in ("uid", "pool", "licence")}}
    out.append(json.dumps(m, ensure_ascii=False) + "\n")
open(p, "w").writelines(out); print("fixed", len(out))
