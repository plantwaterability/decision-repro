#!/bin/bash
# private-like proxy: rows (dedupe + quotas) -> exclusion list + rows + images + code to S3, each object md5-read-back
set -u
export AWS_DEFAULT_REGION=us-east-2 OMP_NUM_THREADS=1
PV=/data/shared/visdata_priv; SC=/data/local/visdata/scripts; L=$PV/logs/chain.log
S=s3://YOUR_BUCKET/jev-race/engineering/vision/recipe_privlike
log(){ echo "[vpl $(date -u +%H:%M:%SZ)] $*" >> $L; }
put(){ aws s3 cp --quiet "$1" "$S/$2" || { log "PUT_FAIL $2"; return 1; }
  a=$(md5sum < "$1" | cut -d" " -f1); b=$(aws s3 cp --quiet "$S/$2" - | md5sum | cut -d" " -f1)
  if [ "$a" = "$b" ]; then echo "$a  $2" >> $PV/MD5SUMS; log "PUT_OK $2 $a"; else log "MD5_BAD $2"; return 1; fi; }
log START
cd $SC && /workspace/anaconda3/bin/python3 vpl_build.py rows > $PV/logs/rows.log 2>&1 || { log "ROWS_FAIL $(tail -2 $PV/logs/rows.log | tr '\n' ' ')"; exit 2; }
log "ROWS $(grep -c . $PV/priv_rows.jsonl) excl $(grep -c . $PV/privlike_image_exclusion.jsonl)"
: > $PV/MD5SUMS
put $PV/privlike_image_exclusion.jsonl privlike_image_exclusion.jsonl || exit 3
put $PV/priv_rows.jsonl priv_rows.jsonl || exit 3
put $PV/priv_rows.header.json priv_rows.header.json || exit 3
sed 's#^#img_priv/#' $PV/priv_images.txt > $PV/priv_images_tar.txt
tar cf $PV/img_priv.tar -C $PV -T $PV/priv_images_tar.txt || { log TAR_FAIL; exit 4; }
log "TAR $(du -h $PV/img_priv.tar | cut -f1) files $(grep -c . $PV/priv_images_tar.txt)"
put $PV/img_priv.tar img_priv.tar || exit 3
put $SC/vpl_build.py code/vpl_build.py; put $SC/vpl_chain.sh code/vpl_chain.sh; put $SC/vpl_dl.py code/vpl_dl.py; put $PV/dl/pins.json code/pins.json
for f in $PV/items_priv/*.jsonl; do put $f items_priv/$(basename $f); done
aws s3 cp --quiet $PV/MD5SUMS $S/MD5SUMS && log MD5SUMS_UP
log CHAIN_DONE
