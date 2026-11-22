#!/bin/bash
# usage: vg_pkgmeta.sh <tag> : code + summaries + manifests + lists + provenance -> s3 recipe_g (md5 read-back), then the aggregate
# recipe_g/MD5SUMS (one line per file, last upload wins; the arm scripts look up " img_<source>.tar" in <recipe>/MD5SUMS). Round F's vf_pkgmeta.sh.
set -u
export AWS_DEFAULT_REGION=us-east-2
G=/data/shared/visdata_g; S=$G/scripts; L=$G/logs/g_package.log
SG=s3://YOUR_BUCKET/jev-race/engineering/vision/recipe_g
tag=$1
log(){ echo "[pkg $tag $(date -u +%H:%M:%SZ)] $*" >> $L; }
M=$G/pkg/MD5SUMS_$tag; mkdir -p $G/pkg; : > $M
put(){ aws s3 cp --quiet "$1" "$2" && a=$(md5sum < "$1" | cut -d" " -f1) && b=$(aws s3 cp --quiet "$2" - | md5sum | cut -d" " -f1) && [ "$a" = "$b" ] && { log "S3_OK $(basename $1) $a"; echo "$a  $(basename $1)" >> $M; } || log "S3_BAD $1"; }
for f in $G/summ/*.json; do put $f $SG/summ/$(basename $f); done
for f in $S/vg_*.py $S/vg_*.sh $S/cfg_*.json $G/launcher/qv_arm_ug.sh; do [ -f $f ] && put $f $SG/code/$(basename $f); done
for f in $G/manifests_g/*; do put $f $SG/manifests/$(basename $f); done
put $G/nevertrain_hashes.jsonl $SG/manifests/nevertrain_hashes.jsonl
for f in $G/lists/*.jsonl; do [ -f $f ] && put $f $SG/lists/$(basename $f); done
for f in $G/dl/SHA256SUMS_meta1 $G/dl/gqa/SHA256SUMS_gqa $G/dl/pixmo/plan.jsonl $G/dl/pixmo/plan_summary.json $G/dl/pixmo/fetch_status.jsonl $G/dl/pixmo/photo_screen.jsonl $G/dl/gqa/plan.jsonl $G/dl/gqa/plan_summary.json; do [ -f $f ] && put $f $SG/prov/$(basename $(dirname $f))_$(basename $f); done
for f in $G/logs/g_build.log $G/logs/g_download.log $G/logs/g_verify.log; do [ -f $f ] && put $f $SG/prov/$(basename $f); done
aws s3 cp --quiet $M $SG/MD5SUMS_$tag
python3 -I - <<'PY'
import glob, os
last = {}
for f in sorted(glob.glob("/data/shared/visdata_g/pkg/MD5SUMS_*"), key=os.path.getmtime):
    for l in open(f):
        m, n = l.split()
        last[n] = m
open("/data/shared/visdata_g/pkg/MD5SUMS", "w").write("".join(f"{m}  {n}\n" for n, m in sorted(last.items())))
PY
a=$(md5sum < $G/pkg/MD5SUMS | cut -d" " -f1); aws s3 cp --quiet $G/pkg/MD5SUMS $SG/MD5SUMS; b=$(aws s3 cp --quiet $SG/MD5SUMS - | md5sum | cut -d" " -f1)
log "MD5SUMS_UP $(wc -l < $M) files, bad=$(grep -c "pkg $tag .*S3_BAD" $L); aggregate MD5SUMS $(wc -l < $G/pkg/MD5SUMS) lines $a readback=$([ "$a" = "$b" ] && echo ok || echo BAD)"
log "G_PACKAGE_DONE $tag"
