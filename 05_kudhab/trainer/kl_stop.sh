#!/bin/bash
# kl_stop.sh ARM "WHY" (KL agent): stop one arm on this node on the coordinator's order. Markers first (STOPPED_BY_COORDINATOR + DONE, so no guard or
# keeper starts it again), then exact PIDs only: keeper, chain, the trainer's launcher + its 8 ranks + their upload children, kl_run. No final save:
# the last periodic S3 resume dir (<= 100 steps old) and every export / read stay on S3.
set -u; source /data/local/kl/scripts/kl_env.sh; ARM=${1:?}; WHY=${2:-}; O=$R/esbar_kl_$ARM; L=$K/logs/stop_$ARM.log
log() { echo "[kl_stop $ARM $(date -u +%FT%TZ)] $*" | tee -a $L; }
last=$(grep -h '^STEP' $K/run/$ARM/logs/train.log 2>/dev/null | tail -1 | cut -c1-40); sv=$(grep -h 'SAVE_DONE_LOCAL' $K/run/$ARM/logs/train.log 2>/dev/null | tail -1 | cut -c1-60)
msg="$(date -u +%FT%TZ) $(hostname): $ARM STOPPED by the coordinator at its mid read ($WHY); last step: $last; last save: $sv; S3 outputs kept"
echo "$msg" | aws s3 cp --only-show-errors - $O/markers/STOPPED_BY_COORDINATOR && echo "STOPPED by the coordinator: $msg" | aws s3 cp --only-show-errors - $O/markers/DONE || { log "MARKERS FAILED"; exit 1; }
kp() { local p; for p in "$@"; do kill -$SIG $p 2>/dev/null && log "kill -$SIG $p ($WHAT)"; done; }
SIG=TERM; WHAT=keeper; kp $(pgrep -f '[k]l_keeper.sh'); WHAT=chain; kp $(pgrep -f '[k]l_chain.sh')
LA=$(pgrep -f '[t]orch.distributed.run'); RK=""; UP=""; for l in $LA; do RK="$RK $(pgrep -P $l)"; done; for r in $RK; do UP="$UP $(pgrep -P $r)"; done
SIG=KILL; WHAT=launcher; kp $LA; WHAT=rank; kp $RK; WHAT=upload-child; kp $UP
sleep 3; SIG=KILL; WHAT=trainer-left; kp $(pgrep -f '[k]l_train.py'); SIG=TERM; WHAT=kl_run; kp $(pgrep -f '[k]l_run.sh'); sleep 5; SIG=KILL; WHAT=kl_run-left; kp $(pgrep -f '[k]l_run.sh')
sleep 10; log "left: keeper $(pgrep -fc '[k]l_keeper.sh') chain $(pgrep -fc '[k]l_chain.sh') run $(pgrep -fc '[k]l_run.sh') train $(pgrep -fc '[k]l_train.py') gpu MiB $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
log "KL_STOP_DONE $msg"; aws s3 cp --only-show-errors $L $O/logs/stop_$ARM.log
