#!/bin/bash
set -u
D=/data/local/visdata; DL=/data/shared/visdata_dl; S=$D/scripts; L=$D/logs/build.log
log(){ echo "[fix $(date -u +%H:%M:%SZ)] $*" >> $L; }
export OMP_NUM_THREADS=1
cd $D
log START
run(){ python3 $S/vb_build.py $1 > $D/logs/b_$1.log 2>&1 && log "FAM_OK $1 $(tail -1 $D/logs/b_$1.log | cut -c1-220)" || log "FAM_FAIL $1 $(tail -2 $D/logs/b_$1.log | tr '\n' ' ' | cut -c1-300)"; }
( aws s3 cp --quiet s3://YOUR_BUCKET/jev-race/engineering/vision/data/meme/meme_v1/img.tar $DL/meme_img.tar && log MEME_TAR_OK || log MEME_TAR_FAIL
  python3 - <<'PY'
import hashlib, json, tarfile
items = [json.loads(l) for l in open('/data/local/visdata/v1/pools/meme_items_train.jsonl')]
items.sort(key=lambda it: hashlib.sha256(it['id'].encode()).hexdigest())
want = {it['images'][0].split('/')[-1] for it in items[:3500]}
n = 0
with tarfile.open('/data/shared/visdata_dl/meme_img.tar', 'r|') as tf:
    for m in tf:
        if m.isfile() and m.name.split('/')[-1] in want:
            m.name = 'meme/' + m.name.split('/')[-1]
            tf.extract(m, '/data/local/visdata/memeimg')
            n += 1
print('MEME_EXTRACTED', n, flush=True)
PY
  log "MEME_EXTRACT $(ls $D/memeimg/meme 2>/dev/null | wc -l)"; run meme_v1 ) &
run sat & run coco_wino & wait
run docvqa & run infovqa_train & wait
log FIX_FAMILIES_DONE
python3 $S/vb_manifest.py > $D/logs/manifest.log 2>&1 && log "MANIFEST_OK $(grep -c rows_sha256 $D/logs/manifest.log) manifests" || log "MANIFEST_FAIL $(tail -3 $D/logs/manifest.log | tr '\n' ' ' | cut -c1-300)"
bash $S/vb_package.sh && log PACKAGE_DONE || log PACKAGE_FAIL
bash $S/vb_smoke.sh && log SMOKE_PKG_DONE || log SMOKE_PKG_FAIL
log ALL_DONE
