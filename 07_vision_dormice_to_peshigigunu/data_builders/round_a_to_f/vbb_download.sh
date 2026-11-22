#!/bin/bash
# Round-B gap data: raw downloads (ctl2). Public sources, no tokens.
set -u
export AWS_DEFAULT_REGION=us-east-2
DL=/data/shared/visdata_dl_b; D=/data/local/visdata; mkdir -p $DL $D/logs; L=$D/logs/download_b.log
log(){ echo "[dlb $(date -u +%H:%M:%SZ)] $*" >> $L; }
export HF_HUB_DISABLE_XET=1; unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
log START
get(){ python3 -c "
import httpx, sys
with httpx.stream('GET', sys.argv[1], timeout=900, follow_redirects=True) as r:
    r.raise_for_status()
    with open(sys.argv[2], 'wb') as f:
        for c in r.iter_bytes(1 << 20): f.write(c)" "$1" "$2"; }
( get https://ai2-public-datasets.s3.amazonaws.com/diagrams/ai2d-all.zip $DL/ai2d-all.zip && log AI2D_OK || log AI2D_FAIL ) &
( get https://ai2-public-datasets.s3.amazonaws.com/diagrams/ai2d_test_ids.csv $DL/ai2d_test_ids.csv && log AI2D_TESTIDS_OK || log AI2D_TESTIDS_FAIL ) &
wait
python3 - <<'PY' >> $D/logs/download_b_py.log 2>&1
from huggingface_hub import snapshot_download, list_repo_files
import re
fs = list_repo_files("derek-thomas/ScienceQA", repo_type="dataset")
print("SQA files", fs[:20], flush=True)
snapshot_download("derek-thomas/ScienceQA", repo_type="dataset", local_dir="/data/shared/visdata_dl_b/scienceqa",
                  allow_patterns=[f for f in fs if re.search(r"train", f) and f.endswith(".parquet")] + ["README.md"], token=False, max_workers=4)
print("DL_OK scienceqa", flush=True)
snapshot_download("ibm-granite/ChartNet", repo_type="dataset", local_dir="/data/shared/visdata_dl_b/chartnet",
                  allow_patterns=[f"grounded_qa/grounded_qa_{i:02d}.parquet" for i in (2, 3, 4, 5)], token=False, max_workers=4)
print("DL_OK chartnet_b", flush=True)
snapshot_download("nvidia/Nemotron-Image-Training-v3.5", repo_type="dataset", local_dir="/data/shared/visdata_dl_b/nemotron",
                  allow_patterns=["coco_count_spatial_qa/*"], token=False, max_workers=4)
print("DL_OK nemotron_coco", flush=True)
PY
log "HF_RC=$?"
# 2.5VRD annotations (CC BY 4.0, google-research-datasets/2.5vrd)
python3 - <<'PY' >> $D/logs/download_b_py.log 2>&1
import httpx, re
t = httpx.get("https://raw.githubusercontent.com/google-research-datasets/2.5vrd/master/README.md", timeout=60).text
urls = sorted(set(re.findall(r"https://storage\.googleapis\.com/[^\s)\"']+", t)))
print("VRD urls", urls, flush=True)
for u in urls:
    if any(u.endswith(x) for x in (".zip", ".csv", ".json", ".tar.gz", ".tgz")):
        fn = "/data/shared/visdata_dl_b/vrd_" + u.rsplit("/", 1)[-1]
        with httpx.stream("GET", u, timeout=600, follow_redirects=True) as r:
            r.raise_for_status()
            with open(fn, "wb") as f:
                for c in r.iter_bytes(1 << 20): f.write(c)
        print("DL_OK", fn, flush=True)
PY
log "VRD_RC=$?"
du -sh $DL/* >> $L 2>&1
log DONE
