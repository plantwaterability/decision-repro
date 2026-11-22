#!/bin/bash
set -u
export AWS_DEFAULT_REGION=us-east-2
D=/data/local/visdata; DD=/data/shared/visdata_d; L=$D/logs/a4_data.log
SD=s3://YOUR_BUCKET/jev-race/engineering/vision/recipe_d; SA=s3://YOUR_BUCKET/jev-race/engineering/vision/recipe_a
log(){ echo "[a4d $(date -u +%H:%M:%SZ)] $*" >> $L; }
mkdir -p $DD/pkg; : > $DD/pkg/MD5SUMS
put(){ aws s3 cp --quiet "$1" "$2" && a=$(md5sum < "$1" | cut -d" " -f1) && b=$(aws s3 cp --quiet "$2" - | md5sum | cut -d" " -f1) && [ "$a" = "$b" ] && { log "S3_OK $(basename $1) $a"; echo "$a  $(basename $1)" >> $3; } || log "S3_BAD $1"; }
python3 $D/scripts/vba4_mix.py > $D/logs/a4_mix.log 2>&1 && log "A4_MIX $(tail -1 $D/logs/a4_mix.log | cut -c1-900)" || { log "A4_MIX_FAIL $(tail -3 $D/logs/a4_mix.log | tr '\n' ' ')"; exit 1; }
for f in $DD/items_d/*.jsonl; do put $f $SD/items/$(basename $f) $DD/pkg/MD5SUMS; done
for fam in $(ls $DD/img_d); do tar cf $DD/pkg/img_$fam.tar -C $DD/img_d $fam && put $DD/pkg/img_$fam.tar $SD/img/img_$fam.tar $DD/pkg/MD5SUMS && rm -f $DD/pkg/img_$fam.tar; done
for f in $D/scripts/vbd_build.py $D/scripts/vba4_mix.py $D/scripts/vbd_package.sh; do put $f $SD/code/$(basename $f) $DD/pkg/MD5SUMS; done
aws s3 cp --quiet $DD/pkg/MD5SUMS $SD/MD5SUMS && log D_MD5SUMS_UP
: > $DD/pkg/MD5SUMS_a4
for f in a4_mix.jsonl a4_mix.header.json; do put $D/manifests/$f $SA/manifests/$f $DD/pkg/MD5SUMS_a4; done
log A4_DATA_DONE
