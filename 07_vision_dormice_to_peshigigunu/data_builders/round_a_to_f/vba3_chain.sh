#!/bin/bash
set -u
export AWS_DEFAULT_REGION=us-east-2 OMP_NUM_THREADS=1
D=/data/local/visdata; L=$D/logs/a3_chain.log
S3B3=s3://YOUR_BUCKET/jev-race/engineering/vision/recipe_b3; S3A=s3://YOUR_BUCKET/jev-race/engineering/vision/recipe_a
log(){ echo "[a3c $(date -u +%H:%M:%SZ)] $*" >> $L; }
while [ $(cat $D/logs/bb3_ai2d.log $D/logs/bb3_sqa.log $D/logs/bb3_chartnet_b.log 2>/dev/null | grep -c "^RC=") -lt 3 ]; do sleep 30; done
log "RB2 $(grep -h '^RC=' $D/logs/bb3_*.log | tr '\n' ' ') $(for f in $D/items_b3/*.jsonl; do echo -n "$(basename $f .jsonl)=$(wc -l < $f) "; done)"
mkdir -p $D/pkg_b3; : > $D/pkg_b3/MD5SUMS
put(){ aws s3 cp --quiet "$1" "$2" && a=$(md5sum < "$1" | cut -d" " -f1) && b=$(aws s3 cp --quiet "$2" - | md5sum | cut -d" " -f1) && [ "$a" = "$b" ] && { log "S3_OK $(basename $1) $a"; echo "$a  $(basename $1)" >> $3; } || log "S3_BAD $1"; }
for f in $D/items_b3/*.jsonl; do put $f $S3B3/items/$(basename $f) $D/pkg_b3/MD5SUMS; done
for fam in $(ls $D/img_b3); do tar cf $D/pkg_b3/img_$fam.tar -C $D/img_b3 $fam && put $D/pkg_b3/img_$fam.tar $S3B3/img/img_$fam.tar $D/pkg_b3/MD5SUMS && rm -f $D/pkg_b3/img_$fam.tar; done
put $D/scripts/vbb_build.py $S3B3/code/vbb_build.py $D/pkg_b3/MD5SUMS
aws s3 cp --quiet $D/pkg_b3/MD5SUMS $S3B3/MD5SUMS && log "B3_MD5SUMS_UP"
while ! grep -q CHAIN_DONE $D/logs/vbc_chain.log 2>/dev/null; do sleep 30; done
log "ROUND_C $(grep MANIFEST $D/logs/vbc_chain.log | tail -1 | cut -c1-300)"
python3 $D/scripts/vba3_mix.py > $D/logs/a3_mix.log 2>&1 && log "A3_MIX $(tail -1 $D/logs/a3_mix.log | cut -c1-700)" || { log "A3_MIX_FAIL $(tail -3 $D/logs/a3_mix.log | tr '\n' ' ')"; exit 1; }
: > $D/pkg_b3/MD5SUMS_a3
for f in a3_mix.jsonl a3_mix.header.json; do put $D/manifests/$f $S3A/manifests/$f $D/pkg_b3/MD5SUMS_a3; done
log A3_CHAIN_DONE
