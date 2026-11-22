#!/bin/bash
# FS5 upload: stage1_fs5.sh from stage1_fs1.sh (manifest path / S3 run prefix, expected sha, steps), then manifests + pools + code +
# licence evidence + logs to s3 .../esbar/runs/esbar_fs5/, md5 read-back of every object. Never writes under esbar_fs1/.
set -u; unset ASCALE_RUN_ID; export AWS_DEFAULT_REGION=us-east-2
W=/data/shared/fs5; S3F=s3://YOUR_BUCKET/jev-race/engineering/esbar/runs/esbar_fs5
LOG=$W/logs/upload.log; log() { echo "[upload $(date -u +%FT%TZ)] $*" >> $LOG; }
M=$W/out/esbar_stage1_fs5.jsonl
SHA=$(python3 -c "import json; print(json.load(open('$M.header.json'))['rows_sha256'])"); ROWS=$(python3 -c "import json; print(json.load(open('$M.header.json'))['rows'])")
STEPS=$(( (ROWS + 63) / 64 ))
python3 -c "import json,sys; v=json.load(open('$M.validate.json')); sys.exit(0 if v['READ_MANIFEST_OK']==$ROWS and v['sha256_file']=='$SHA' else 1)" || { log "VALIDATE_MISMATCH"; exit 1; }
if aws s3 ls $S3F/manifests/esbar_stage1_fs5.jsonl > /dev/null 2>&1; then log "REFUSE: manifest already on S3"; exit 1; fi
mkdir -p $W/stage1; sed -e "s|esbar_stage1_fs1\.jsonl|esbar_stage1_fs5.jsonl|g" \
  -e "s|PMSHA_EXPECT=8461365da84aaf5e0f3528900b314755b080212d6ea2cda9ceb0f41111cad644|PMSHA_EXPECT=$SHA|" \
  -e "s|STEPS_EXPECT=6011;|STEPS_EXPECT=$STEPS;|" -e "s|/engineering/esbar/runs/esbar_fs1$|/engineering/esbar/runs/esbar_fs5|" $W/ref/stage1_fs1.sh > $W/stage1/stage1_fs5.sh
diff $W/ref/stage1_fs1.sh $W/stage1/stage1_fs5.sh > $W/stage1/stage1_fs1_to_fs5.diff; log "stage1 diff lines: $(grep -c '^>' $W/stage1/stage1_fs1_to_fs5.diff)"
bash -n $W/stage1/stage1_fs5.sh || { log "STAGE1_SYNTAX"; exit 1; }
put() { aws s3 cp --only-show-errors "$1" "$S3F/$2" && echo "$(md5sum < "$1" | cut -c1-32)  $2" >> $W/stage1/MD5SUMS.new; }
: > $W/stage1/MD5SUMS.new
for f in esbar_stage1_fs5.jsonl esbar_stage1_fs5.jsonl.header.json esbar_stage1_fs5.jsonl.report.json esbar_stage1_fs5.jsonl.validate.json; do put $W/out/$f manifests/$f; done
for f in $W/out/pools/*.jsonl $W/out/fs5_pools_report.json; do put $f pools/$(basename $f); done
for f in fs5_build.py fs5_validate.py run_build.sh stage.sh build_manifest_v3.py upload.sh fix_acos_meta.py fix_and_upload.sh fs5_build_as_run_1e5c0690.py; do put $W/scripts/$f code/$f; done
for f in $W/lic/*; do put $f lic/$(basename $f); done; put $W/src/dl_sha256.txt lic/dl_sha256.txt
for f in stage1_fs5.sh stage1_fs1_to_fs5.diff; do put $W/stage1/$f scripts/$f; done
for f in build.log stage.log; do put $W/logs/$f logs/$f; done
ok=0; bad=0; : > $W/stage1/READBACK.txt
while read m k; do r=$(aws s3 cp --only-show-errors "$S3F/$k" - | md5sum | cut -c1-32)
  if [ "$r" = "$m" ]; then ok=$((ok+1)); echo "OK $k $m" >> $W/stage1/READBACK.txt; else bad=$((bad+1)); echo "BAD $k $m $r" >> $W/stage1/READBACK.txt; fi
done < $W/stage1/MD5SUMS.new
mv $W/stage1/MD5SUMS.new $W/stage1/MD5SUMS
aws s3 cp --only-show-errors $W/stage1/MD5SUMS $S3F/MD5SUMS; aws s3 cp --only-show-errors $W/stage1/READBACK.txt $S3F/READBACK_MD5.txt
log "UPLOAD_DONE ok=$ok bad=$bad sha=$SHA rows=$ROWS steps=$STEPS"
