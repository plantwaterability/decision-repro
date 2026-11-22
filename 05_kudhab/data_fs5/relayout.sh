#!/bin/bash
# Move the data fork's own objects under esbar_fs5/manifests/ (TB1 hand-off: the data agent owns manifests/ only); md5 read-back all.
set -u; export AWS_DEFAULT_REGION=us-east-2
O5=s3://YOUR_BUCKET/jev-race/engineering/esbar/runs/esbar_fs5; W=/data/shared/fs5/stage1
LOG=/data/shared/fs5/logs/upload.log; log() { echo "[relayout $(date -u +%FT%TZ)] $*" >> $LOG; }
: > $W/MD5SUMS.v2
while read m k; do
  case $k in
    manifests/*) nk=$k ;;
    scripts/stage1_fs5.sh) nk=manifests/stage1_fs5_strict_from_fs1.sh ;;   # already copied; TB1's file restored at scripts/
    scripts/stage1_fs1_to_fs5.diff) nk=manifests/stage1_fs1_to_fs5.diff ;;
    pools/*|code/*|lic/*|logs/*) nk=manifests/supplement/$k; aws s3 mv --only-show-errors $O5/$k $O5/$nk || log "MV_FAIL $k" ;;
    *) nk=$k ;;
  esac
  echo "$m  $nk" >> $W/MD5SUMS.v2
done < $W/MD5SUMS
for f in /data/shared/fs5/scripts/restore_tb1_stage1.sh /data/shared/fs5/scripts/gen_stage1_part.sh /data/shared/fs5/scripts/relayout.sh; do
  aws s3 cp --only-show-errors $f $O5/manifests/supplement/code/$(basename $f); echo "$(md5sum < $f | cut -c1-32)  manifests/supplement/code/$(basename $f)" >> $W/MD5SUMS.v2; done
aws s3 rm --only-show-errors $O5/MD5SUMS; aws s3 rm --only-show-errors $O5/READBACK_MD5.txt
ok=0; bad=0; : > $W/READBACK.v2
while read m k; do r=$(aws s3 cp --only-show-errors "$O5/$k" - | md5sum | cut -c1-32)
  if [ "$r" = "$m" ]; then ok=$((ok+1)); echo "OK $k $m" >> $W/READBACK.v2; else bad=$((bad+1)); echo "BAD $k $m $r" >> $W/READBACK.v2; fi
done < $W/MD5SUMS.v2
aws s3 cp --only-show-errors $W/MD5SUMS.v2 $O5/manifests/MD5SUMS; aws s3 cp --only-show-errors $W/READBACK.v2 $O5/manifests/READBACK_MD5.txt
log "RELAYOUT_DONE ok=$ok bad=$bad"; echo "RELAYOUT_DONE ok=$ok bad=$bad"
aws s3 ls --recursive $O5/ | awk '{print $4}' | sed "s|jev-race/engineering/esbar/runs/esbar_fs5/||" | grep -v "^manifests/supplement/" ; aws s3 ls --recursive $O5/manifests/supplement/ | wc -l
