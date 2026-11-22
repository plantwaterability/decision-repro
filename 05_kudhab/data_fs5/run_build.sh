#!/bin/bash
# FS5 build chain on ctl2 (CPU): builder -> trainer-code validation. Logs in $W/logs.
set -u; unset ASCALE_RUN_ID; export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false AWS_DEFAULT_REGION=us-east-2
W=/data/shared/fs5; cd $W; T=""; [ -x /usr/bin/time ] && T="/usr/bin/time -v"
cp -n /data/local/v8build/scripts/build_manifest_v3.py $W/scripts/build_manifest_v3.py; cp -n $W/ref/esbar_stage1_fs1.jsonl.header.json $W/in/
echo "[run_build $(date -u +%FT%TZ)] start bm=$(md5sum < $W/scripts/build_manifest_v3.py | cut -c1-8) builder=$(md5sum < $W/scripts/fs5_build.py | cut -c1-8)" > $W/logs/build.log
$T python3 -I $W/scripts/fs5_build.py --w $W >> $W/logs/build.log 2>&1; rc=$?
echo "[run_build $(date -u +%FT%TZ)] BUILD_RC $rc" >> $W/logs/build.log
[ $rc = 0 ] || exit 1
$T python3 -I $W/scripts/fs5_validate.py $W >> $W/logs/build.log 2>&1; echo "[run_build $(date -u +%FT%TZ)] VALIDATE_RC $?" >> $W/logs/build.log
