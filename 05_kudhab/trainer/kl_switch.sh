#!/bin/bash
# kl_switch.sh ARM START "NEXT ORDER" (KL agent; coordinator order 22:05Z 11-15, item 4): wait for ARM's mid screen (step 3,287) on this node; unless it is
# >= START + 0.3 (its own untrained start screen), stop the arm (kl_stop.sh: markers + exact PIDs; the S3 resume state stays) and run the next order here.
set -u; source /data/local/kl/scripts/kl_env.sh; ARM=${1:?}; START=${2:?}; ORDER=${3:?}; O=$R/esbar_kl_$ARM; L=$K/logs/switch_$ARM.log
log() { echo "[kl_switch $ARM $(date -u +%FT%TZ)] $*" | tee -a $L; aws s3 cp --only-show-errors $L $O/logs/switch_$ARM.log 2>/dev/null; }
log "waiting for the mid screen (start $START; continue only at >= start + 0.3)"
while true; do X=$(grep -E '^INDEX KOWN' $M/work/kl_${ARM}_s3287/kown_s4400/score.txt 2>/dev/null | tail -1 | awk '{print $3}')
  [ -n "$X" ] && aws s3 ls $O/markers/READ_s4400_s3287 > /dev/null 2>&1 && break; sleep 20; done
if python3 -c "import sys; sys.exit(0 if float('$X') >= $START + 0.3 else 1)"; then
  echo "$(date -u +%FT%TZ) mid screen $X >= start $START + 0.3: the arm continues to its final" | aws s3 cp --only-show-errors - $O/markers/KEEP; log "KEEP mid screen $X >= start $START + 0.3"; exit 0; fi
log "mid screen $X < start $START + 0.3: stopping $ARM, then order: $ORDER"
bash $SC/kl_stop.sh $ARM "mid screen $X, not >= its start $START + 0.3" >> $L 2>&1 || { log "STOP FAILED"; exit 1; }
log "starting the keeper for: $ORDER"; export KL_ORDER="$ORDER"; exec bash $SC/kl_keeper.sh > $K/logs/kl_keeper.out 2>&1
