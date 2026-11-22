#!/bin/bash
# Round-B backup: items_b, per-family image tars, manifests_b -> S3 recipe_b/ with md5 read-back.
set -u
export AWS_DEFAULT_REGION=us-east-2
D=/data/local/visdata; L=$D/logs/package_b.log
S3=s3://YOUR_BUCKET/jev-race/engineering/vision/recipe_b
log(){ echo "[pkgb $(date -u +%H:%M:%SZ)] $*" >> $L; }
put(){
  aws s3 cp --quiet "$1" "$2" || { log "UPLOAD_FAIL $1"; return 1; }
  a=$(md5sum < "$1" | cut -d" " -f1); b=$(aws s3 cp --quiet "$2" - | md5sum | cut -d" " -f1)
  [ "$a" = "$b" ] && { log "S3_OK $(basename $1) md5=$a"; echo "$a  $(basename $1)" >> $D/pkg_b/MD5SUMS; } || log "S3_MD5_BAD $1 local=$a s3=$b"
}
mkdir -p $D/pkg_b; : > $D/pkg_b/MD5SUMS
log START
cd $D
for f in items_b/*.jsonl; do put $f $S3/items/$(basename $f); done
for fam in $(ls img_b); do tar cf pkg_b/img_$fam.tar -C img_b $fam && put pkg_b/img_$fam.tar $S3/img/img_$fam.tar; done
for f in manifests_b/*; do put $f $S3/manifests/$(basename $f); done
for f in scripts/vbb_build.py scripts/vbb_manifest.py scripts/vbb_download.sh scripts/vbb_package.sh; do put $f $S3/code/$(basename $f); done
put pkg_b/MD5SUMS $S3/MD5SUMS
log DONE
