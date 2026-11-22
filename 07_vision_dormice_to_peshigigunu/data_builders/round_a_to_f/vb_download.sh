#!/bin/bash
# Vision recipe data: raw downloads (ctl2). Public sources only, no tokens. Logs to $L.
set -u
S3=s3://YOUR_BUCKET/jev-race
V=$S3/engineering/vision
DL=/data/shared/visdata_dl; D=/data/local/visdata
mkdir -p $DL $D/logs; L=$D/logs/download.log
log(){ echo "[dl $(date -u +%H:%M:%SZ)] $*" >> $L; }
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
log START
hfget(){ # repo pattern... (dataset)
  python3 - "$@" <<'PY' >> /data/local/visdata/logs/download_py.log 2>&1
import sys
from huggingface_hub import snapshot_download
repo, local, *pats = sys.argv[1:]
p = snapshot_download(repo, repo_type="dataset", local_dir=local, allow_patterns=pats, token=False, max_workers=8)
print("DL_OK", repo, pats, p, flush=True)
PY
}
# S3-staged inputs (V1 images, the public suite for dedupe, DocVQA raw)
( aws s3 cp --quiet $V/v1/staged/v1_imgs.tar $DL/v1_imgs.tar && log V1_IMGS_OK || log V1_IMGS_FAIL ) &
( aws s3 cp --quiet $V/suite/suite_noWG.tar $DL/suite_noWG.tar && aws s3 cp --quiet $V/suite/suite_noWG.tar.md5 $DL/suite_noWG.tar.md5 && log SUITE_OK || log SUITE_FAIL ) &
( aws s3 sync --quiet $V/data/raw/docvqa/ $DL/docvqa/ && log DOCVQA_OK || log DOCVQA_FAIL ) &
wait
# HF sources
hfget osunlp/Multimodal-Mind2Web $DL/m2w 'data/train-*' && log M2W_OK || log M2W_FAIL
hfget lmms-lab-encoder/DocVQA $DL/infovqa 'InfographicVQA/train-*' && log INFOVQA_OK || log INFOVQA_FAIL
hfget naver-clova-ix/cord-v2 $DL/cord 'data/train-*' && log CORD_OK || log CORD_FAIL
hfget neuralcatcher/hateful_memes $DL/hm 'train.jsonl' 'dev_seen.jsonl' 'test_seen.jsonl' 'dev_unseen.jsonl' 'test_unseen.jsonl' && log HM_JSONL_OK || log HM_JSONL_FAIL
hfget MLL-Lab/MindCube $DL/mindcube 'data.zip' 'README.md' && log MINDCUBE_OK || log MINDCUBE_FAIL
hfget ibm-granite/ChartNet $DL/chartnet 'grounded_qa/*00000-of-*' 'grounded_qa/*00001-of-*' 'README.md' && log CHARTNET_OK || log CHARTNET_FAIL
hfget ServiceNow/GroundCUA $DL/groundcua 'instruction_tuning.tar.gz' 'README.md' && log GROUNDCUA_IT_OK || log GROUNDCUA_IT_FAIL
# FUNSD (the kit's own URL; training_data is disjoint from the board's testing_data)
( cd $DL && python3 -c "
import httpx; open('funsd_dataset.zip','wb').write(httpx.get('https://guillaumejaume.github.io/FUNSD/dataset.zip', timeout=300, follow_redirects=True).content)" && log FUNSD_OK || log FUNSD_FAIL )
# COCO captions (annotations only; images come from the V1 COCO subsets already on disk)
( cd $DL && python3 -c "
import httpx; open('annotations_trainval2017.zip','wb').write(httpx.get('http://images.cocodataset.org/annotations/annotations_trainval2017.zip', timeout=600, follow_redirects=True).content)" && log COCOCAP_OK || log COCOCAP_FAIL )
du -sh $DL/* >> $L 2>&1
log DONE
