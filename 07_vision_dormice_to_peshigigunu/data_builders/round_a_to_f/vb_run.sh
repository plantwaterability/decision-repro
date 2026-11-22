#!/bin/bash
# Vision recipe round-A data build driver (ctl2). Families in parallel groups of 3 (4 CPUs).
set -u
D=/data/local/visdata; DL=/data/shared/visdata_dl; S=$D/scripts; L=$D/logs/build.log
log(){ echo "[build $(date -u +%H:%M:%SZ)] $*" >> $L; }
export OMP_NUM_THREADS=1
cd $D
log START
[ -d $D/v1img ] || { mkdir -p $D/v1img && tar xf $DL/v1_imgs.tar -C $D/v1img && log V1IMG_EXTRACT_OK || log V1IMG_EXTRACT_FAIL; }
if [ ! -d $D/public/suite ]; then
  EXP=$(awk '{print $1}' $DL/suite_noWG.tar.md5); GOT=$(md5sum $DL/suite_noWG.tar | awk '{print $1}')
  [ "$EXP" = "$GOT" ] && { mkdir -p $D/public && tar xf $DL/suite_noWG.tar -C $D/public && log "SUITE_EXTRACT_OK md5=$GOT"; } || log "SUITE_MD5_BAD exp=$EXP got=$GOT"
fi
log "v1img files=$(find $D/v1img -type f | wc -l) public files=$(find $D/public -type f | wc -l)"
run(){ python3 $S/vb_build.py $1 > $D/logs/b_$1.log 2>&1 && log "FAM_OK $1 $(tail -1 $D/logs/b_$1.log | cut -c1-200)" || log "FAM_FAIL $1 $(tail -2 $D/logs/b_$1.log | tr '\n' ' ' | cut -c1-300)"; }
run m2w & run infovqa_train & run docvqa & wait
while ! grep -q DL2_DONE $D/logs/download.log; do sleep 20; done; log DL2_SEEN
run cord_train & run funsd_train & run hm_train & wait
run coco_count & run vsr & run vrd25 & wait
run sat & run meme_v1 & run mindcube & wait
run coco_wino
log FAMILIES_DONE
