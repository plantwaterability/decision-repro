#!/bin/bash
# Restore TB1's s3 .../esbar_fs5/scripts/stage1_fs5.sh (md5 0a0978d8), which the data fork's upload.sh overwrote at 04:37Z with its
# strict-copy stage1 (d4e78e70). Regenerated with the stage1 part of TB1's ops/fsgen/gen_fs.sh (N=5, SEED=5) from FS4's stage1 (060d615b).
set -u; export AWS_DEFAULT_REGION=us-east-2
E=s3://YOUR_BUCKET/jev-race/engineering/esbar; O5=$E/runs/esbar_fs5
D=/data/shared/fs5/restore_tb1; mkdir -p $D; cd $D
N=5; SEED=5
aws s3 cp --only-show-errors $E/runs/esbar_fs4/scripts/stage1_fs4.sh . || exit 1
[ "$(md5sum < stage1_fs4.sh | cut -c1-8)" = 060d615b ] || { echo "SRC_MD5 $(md5sum < stage1_fs4.sh)"; exit 1; }
source /data/shared/fs5/scripts/gen_stage1_part.sh
got=$(md5sum < stage1_fs5.sh | cut -c1-8); echo "regenerated $got"
[ "$got" = 0a0978d8 ] || { echo "MISMATCH, not uploading"; exit 1; }
aws s3 cp --only-show-errors stage1_fs5.sh $O5/scripts/stage1_fs5.sh
rb=$(aws s3 cp --only-show-errors $O5/scripts/stage1_fs5.sh - | md5sum | cut -c1-8); echo "readback $rb"
# the data fork's strict copy + diff move to manifests/ (TB1: the data agent owns manifests/ only)
W=/data/shared/fs5/stage1
aws s3 cp --only-show-errors $W/stage1_fs5.sh $O5/manifests/stage1_fs5_strict_from_fs1.sh
aws s3 cp --only-show-errors $W/stage1_fs1_to_fs5.diff $O5/manifests/stage1_fs1_to_fs5.diff
for k in stage1_fs5_strict_from_fs1.sh stage1_fs1_to_fs5.diff; do echo "$k $(aws s3 cp --only-show-errors $O5/manifests/$k - | md5sum | cut -c1-8)"; done
md5sum $W/stage1_fs5.sh $W/stage1_fs1_to_fs5.diff | cut -c1-8
aws s3 rm --only-show-errors $O5/scripts/stage1_fs1_to_fs5.diff
aws s3 ls $O5/train/ 2>/dev/null | head; aws s3 ls $O5/scripts/
