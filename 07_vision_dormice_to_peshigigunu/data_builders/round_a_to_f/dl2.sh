#!/bin/bash
set -u
DL=/data/shared/visdata_dl; L=/data/local/visdata/logs/download.log
log(){ echo "[dl2 $(date -u +%H:%M:%SZ)] $*" >> $L; }
export HF_HUB_DISABLE_XET=1; unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
while ! grep -q "\] DONE" $L; do sleep 15; done
log START
python3 - <<'PY' >> /data/local/visdata/logs/download_py.log 2>&1
from huggingface_hub import snapshot_download
snapshot_download("ibm-granite/ChartNet", repo_type="dataset", local_dir="/data/shared/visdata_dl/chartnet",
                  allow_patterns=["grounded_qa/grounded_qa_00.parquet", "grounded_qa/grounded_qa_01.parquet"], token=False, max_workers=4)
print("DL_OK chartnet2", flush=True)
snapshot_download("neuralcatcher/hateful_memes", repo_type="dataset", local_dir="/data/shared/visdata_dl/hm",
                  allow_patterns=["img/*"], token=False, max_workers=16)
print("DL_OK hm_img", flush=True)
PY
log "DL2_RC=$?"
du -sh $DL/chartnet $DL/hm >> $L 2>&1
log DL2_DONE
