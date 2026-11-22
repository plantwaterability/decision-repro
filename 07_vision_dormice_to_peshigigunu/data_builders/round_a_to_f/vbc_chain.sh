#!/bin/bash
set -u
export AWS_DEFAULT_REGION=us-east-2 OMP_NUM_THREADS=1
D=/data/local/visdata; C=/data/shared/visdata_c; L=$D/logs/vbc_chain.log
S3=s3://YOUR_BUCKET/jev-race/engineering/vision/recipe_c
log(){ echo "[vbc $(date -u +%H:%M:%SZ)] $*" >> $L; }
while ! grep -q "^RC=" $D/logs/vbc_download.log; do sleep 20; done
log "DOWNLOAD $(grep -c '"files"' $D/logs/vbc_download.log) scenes ok, $(grep -c error $D/logs/vbc_download.log) errors"
python3 $D/scripts/vbc_hypersim.py build > $D/logs/vbc_build.log 2>&1 && log "BUILD $(tail -2 $D/logs/vbc_build.log | tr '\n' ' ' | cut -c1-300)" || { log "BUILD_FAIL $(tail -3 $D/logs/vbc_build.log | tr '\n' ' ' | cut -c1-300)"; exit 1; }
python3 $D/scripts/vbc_finish.py > $D/logs/vbc_finish.log 2>&1 && log "MANIFEST $(tail -1 $D/logs/vbc_finish.log | cut -c1-400)" || { log MANIFEST_FAIL; exit 1; }
mkdir -p $C/pkg; : > $C/pkg/MD5SUMS
put(){ aws s3 cp --quiet "$1" "$2" && a=$(md5sum < "$1" | cut -d" " -f1) && b=$(aws s3 cp --quiet "$2" - | md5sum | cut -d" " -f1) && [ "$a" = "$b" ] && { log "S3_OK $(basename $1) $a"; echo "$a  $(basename $1)" >> $C/pkg/MD5SUMS; } || log "S3_BAD $1"; }
for f in $C/items_c/*.jsonl; do put $f $S3/items/$(basename $f); done
for fam in $(ls $C/img_c); do tar cf $C/pkg/img_$fam.tar -C $C/img_c $fam && put $C/pkg/img_$fam.tar $S3/img/img_$fam.tar; done
for f in $C/manifests_c/*; do put $f $S3/manifests/$(basename $f); done
for f in $D/scripts/vbc_hypersim.py $D/scripts/vbc_finish.py $D/scripts/vbc_chain.sh $D/scripts/hypersim_urls.txt; do put $f $S3/code/$(basename $f); done
put $C/pkg/MD5SUMS $S3/MD5SUMS
log CHAIN_DONE
