#!/bin/bash
# usage: vg_pkgimg.sh <tag> <source> [...] : items + per-source image tar -> s3 recipe_g (md5 read-back). Round F's vf_pkgimg.sh for round G.
set -u
export AWS_DEFAULT_REGION=us-east-2
G=/data/shared/visdata_g; L=$G/logs/g_package.log
SG=s3://YOUR_BUCKET/jev-race/engineering/vision/recipe_g
tag=$1; shift
log(){ echo "[pkg $tag $(date -u +%H:%M:%SZ)] $*" >> $L; }
M=$G/pkg/MD5SUMS_$tag; mkdir -p $G/pkg; : > $M
put(){ aws s3 cp --quiet "$1" "$2" && a=$(md5sum < "$1" | cut -d" " -f1) && b=$(aws s3 cp --quiet "$2" - | md5sum | cut -d" " -f1) && [ "$a" = "$b" ] && { log "S3_OK $(basename $1) $a"; echo "$a  $(basename $1)" >> $M; } || log "S3_BAD $1"; }
for fam in "$@"; do
  [ -s $G/items_g/$fam.jsonl ] || { log "NO_ITEMS $fam"; continue; }
  put $G/items_g/$fam.jsonl $SG/items/$fam.jsonl
  COPYFILE_DISABLE=1 tar cf $G/pkg/img_$fam.tar -C $G/img_g $fam && put $G/pkg/img_$fam.tar $SG/img/img_$fam.tar && rm -f $G/pkg/img_$fam.tar
done
aws s3 cp --quiet $M $SG/MD5SUMS_$tag && log "MD5SUMS_UP $(wc -l < $M) files, bad=$(grep -c "pkg $tag .*S3_BAD" $L)"
log "G_PACKAGE_DONE $tag"
