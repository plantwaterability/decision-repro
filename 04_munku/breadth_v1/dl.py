#!/usr/bin/env python3
# breadth_v1: pinned, tokenless download of tasksource-jev-typed-decisions train shards (filtered-full + full)
import os, json, time, hashlib, sys
from huggingface_hub import hf_hub_download
D = "/data/local/breadth_v1"; REPO = "tasksource/tasksource-jev-typed-decisions"; SHA = "8df2e5c9a33f19a2c2b8febf291c3acb0a91a8a8"
out = D + "/ts_raw"; os.makedirs(out, exist_ok=True)
files = [f"filtered-full/train-{i:05d}-of-00013.parquet" for i in range(13)] + [f"full/train-{i:05d}-of-00015.parquet" for i in range(15)]
log = {}
for f in files:
    for att in range(6):
        try:
            p = hf_hub_download(REPO, f, repo_type="dataset", revision=SHA, local_dir=out, token=False); break
        except Exception as e:
            print("retry", f, repr(e)[:200], flush=True); time.sleep(20 * (att + 1))
    else:
        print("FAILED", f, flush=True); sys.exit(1)
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""): h.update(b)
    log[f] = {"bytes": os.path.getsize(p), "md5": h.hexdigest()}; print("ok", f, log[f], flush=True)
json.dump({"repo": REPO, "revision": SHA, "files": log}, open(D + "/ts_raw/fetch_log.json", "w"), indent=1)
print("DL_DONE", flush=True)
