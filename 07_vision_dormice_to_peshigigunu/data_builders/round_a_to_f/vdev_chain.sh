#!/bin/bash
set -u
export AWS_DEFAULT_REGION=us-east-2 OMP_NUM_THREADS=1 HF_HUB_DISABLE_XET=1
D=/data/local/visdata; DV=/data/shared/visdata_dev; L=$D/logs/vdev_chain.log
S=s3://YOUR_BUCKET/jev-race/engineering/vision/recipe_dev
log(){ echo "[dev $(date -u +%H:%M:%SZ)] $*" >> $L; }
log START
python3 $D/scripts/vdev_build.py build > $D/logs/vdev_build.log 2>&1 && log "BUILD $(grep -c '"family"' $D/logs/vdev_build.log) families: $(grep '"family"' $D/logs/vdev_build.log | tr '\n' ' ' | cut -c1-800)" || { log "BUILD_FAIL $(tail -4 $D/logs/vdev_build.log | tr '\n' ' ' | cut -c1-500)"; exit 1; }
python3 $D/scripts/vdev_build.py rows > $D/logs/vdev_rows.log 2>&1 && log "ROWS $(tail -1 $D/logs/vdev_rows.log | cut -c1-900)" || { log "ROWS_FAIL $(tail -3 $D/logs/vdev_rows.log | tr '\n' ' ')"; exit 1; }
mkdir -p $DV/pkg; : > $DV/pkg/MD5SUMS
put(){ aws s3 cp --quiet "$1" "$2" && a=$(md5sum < "$1" | cut -d" " -f1) && b=$(aws s3 cp --quiet "$2" - | md5sum | cut -d" " -f1) && [ "$a" = "$b" ] && { log "S3_OK $(basename $1) $a"; echo "$a  $(basename $1)" >> $DV/pkg/MD5SUMS; } || log "S3_BAD $1"; }
put $DV/dev_rows.jsonl $S/dev_rows.jsonl; put $DV/dev_rows.header.json $S/dev_rows.header.json
tar cf $DV/pkg/img_dev.tar -C $DV img_dev && put $DV/pkg/img_dev.tar $S/img_dev.tar && rm -f $DV/pkg/img_dev.tar
for f in $DV/items_dev/*.jsonl; do put $f $S/items/$(basename $f); done
put $D/scripts/vdev_build.py $S/code/vdev_build.py; put $D/scripts/vdev_chain.sh $S/code/vdev_chain.sh
aws s3 cp --quiet $DV/pkg/MD5SUMS $S/MD5SUMS && log MD5SUMS_UP
log DEV_DONE
