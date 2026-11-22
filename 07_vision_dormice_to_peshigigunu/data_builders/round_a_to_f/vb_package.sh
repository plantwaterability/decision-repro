#!/bin/bash
# Back up every built pool + both manifests + the dedupe report + the gate package + the kit to S3, md5 read-back.
set -u
D=/data/local/visdata; L=$D/logs/package.log
S3=s3://YOUR_BUCKET/jev-race/engineering/vision/recipe_a
log(){ echo "[pkg $(date -u +%H:%M:%SZ)] $*" >> $L; }
put(){ # local remote
  aws s3 cp --quiet "$1" "$2" || { log "UPLOAD_FAIL $1"; return 1; }
  a=$(md5sum < "$1" | cut -d" " -f1); b=$(aws s3 cp --quiet "$2" - | md5sum | cut -d" " -f1)
  [ "$a" = "$b" ] && { log "S3_OK $(basename $1) md5=$a"; echo "$a  $(basename $1)" >> $D/pkg/MD5SUMS; } || log "S3_MD5_BAD $1 local=$a s3=$b"
}
mkdir -p $D/pkg; : > $D/pkg/MD5SUMS
log START
cd $D
# 1. built pools: items + per-family image tars
for f in items/*.jsonl; do put $f $S3/items/$(basename $f); done
for fam in $(ls img); do
  tar cf pkg/img_$fam.tar -C img $fam && put pkg/img_$fam.tar $S3/img/img_$fam.tar
done
# 2. manifests + headers + dedupe report and drops
for f in manifests/*; do put $f $S3/manifests/$(basename $f); done
put public_hashes.jsonl $S3/manifests/public_hashes.jsonl
# 3. gate package: the 1,500-row sample (sha-gated) + its images from the public suite on disk
aws s3 cp --quiet s3://YOUR_BUCKET/jev-race/engineering/vision/work/vision-sample-150-noWG.jsonl pkg/gate_rows.jsonl
GS=$(sha256sum pkg/gate_rows.jsonl | cut -d" " -f1)
[ "$GS" = "dbb0ce544057f10fa9582aa487356fb8baef52bf5faf2d1c656adc80e022738f" ] && log "GATE_SHA_OK $GS" || log "GATE_SHA_BAD $GS"
python3 - <<'PY' > pkg/gate_images.txt
import json
for l in open('/data/local/visdata/pkg/gate_rows.jsonl'):
    for p in json.loads(l)['images']:
        print(p)
PY
tar cf pkg/gate_images.tar -C public/suite -T pkg/gate_images.txt && put pkg/gate_images.tar $S3/gate/gate_images.tar
put pkg/gate_rows.jsonl $S3/gate/gate_rows.jsonl
# 4. kit (vision overlay merged) for in-trainer reads and the engine
tar czf pkg/kit_vis.tgz -C kit kit_vis && put pkg/kit_vis.tgz $S3/code/kit_vis.tgz
put pkg/MD5SUMS $S3/MD5SUMS
log DONE
