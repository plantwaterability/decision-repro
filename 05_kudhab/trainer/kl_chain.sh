#!/bin/bash
# kl_chain.sh (KL agent): the whole programme on one node, in order, every part idempotent (S3 markers): K1's first 200 steps -> lr probes -> lr choice
# -> K1 legs + reads -> K2 legs + reads. Relaunch-safe on a fresh worker (stages first). One chain per node (kl_run.sh takes the node lock).
set -u; export AWS_DEFAULT_REGION=us-east-2; unset ASCALE_RUN_ID
D=/data/local; K=$D/kl; R=s3://YOUR_BUCKET/jev-race/engineering/esbar/runs; SC=$K/scripts; mkdir -p $SC $K/logs; L=$K/logs/kl_chain.log
log() { echo "[kl_chain $(date -u +%FT%TZ)] $*" | tee -a $L; aws s3 cp --only-show-errors $L $R/esbar_kl_code/logs/kl_chain_$(hostname).log 2>/dev/null; }
exec 7> $K/chain.lock; flock -n 7 || { echo "another kl_chain runs"; exit 3; }
aws s3 sync --only-show-errors $R/esbar_kl_code/scripts/ $SC/ || { log "CHAIN_STOP scripts"; exit 1; }
log "start host $(hostname) order: ${KL_ORDER:-k1 k2}"
case " ${KL_ORDER:-k1 k2} " in *" k2 "*|*" k5 "*|*" k6 "*) KK=1;; *) KK=0;; esac      # karpo's weights are staged only on a node that runs K2
[ -f $K/.staged ] || KL_KARPO=$KK bash $SC/kl_stage.sh > $K/stage.out 2>&1 || { log "CHAIN_STOP stage: $(tail -1 $K/logs/stage.log | cut -c1-200)"; exit 1; }
has() { aws s3 ls $R/esbar_kl_$1/markers/$2 > /dev/null 2>&1; }
for arm in ${KL_ORDER:-k1 k2}; do
  aws s3 sync --only-show-errors $R/esbar_kl_code/scripts/ $SC/ 2>/dev/null      # scripts may have gained arms since this chain started (files are replaced by rename)
  has $arm DONE && { log "$arm DONE already"; continue; }
  case $arm in post*) bash $SC/kl_post.sh $arm >> $K/logs/run_$arm.out 2>&1; rc=$?; log "$arm rc=$rc: $(tail -1 $K/post/$arm/post.log 2>/dev/null | cut -c1-300)"; [ $rc = 0 ] || { log "CHAIN_STOP $arm rc=$rc"; exit 1; }; continue;; esac
  if [ $arm = k1 ] && ! has k1 LR_CHOICE; then
    bash $SC/kl_run.sh k1 >> $K/logs/run_k1.out 2>&1; rc=$?; [ $rc = 10 ] || { log "CHAIN_STOP k1 probe leg rc=$rc"; exit 1; }
    for p in p1e6 p5e6; do bash $SC/kl_run.sh $p >> $K/logs/run_$p.out 2>&1 || { log "CHAIN_STOP probe $p"; exit 1; }; done
    for x in k1 p1e6 p5e6; do aws s3 cp --only-show-errors $R/esbar_kl_k1/evals/evals_$x.jsonl $K/logs/evals_$x.jsonl || { log "CHAIN_STOP evals $x"; exit 1; }; done
    CH=$(python3 $SC/kl_lr_choice.py $K/logs/evals_k1.jsonl $K/logs/evals_p1e6.jsonl $K/logs/evals_p5e6.jsonl) || { log "CHAIN_STOP lr choice"; exit 1; }
    echo "$CH" | aws s3 cp --only-show-errors - $R/esbar_kl_k1/markers/LR_CHOICE; log "LR_CHOICE $CH"
  fi
  bash $SC/kl_run.sh $arm >> $K/logs/run_$arm.out 2>&1; rc=$?
  log "$arm rc=$rc: $(tail -1 $K/run/$arm/logs/chain.log | cut -c1-300)"
  [ $rc = 0 ] || { log "CHAIN_STOP $arm rc=$rc"; exit 1; }
done
echo "FREE: KL chain (${KL_ORDER:-k1 k2}) finished $(date -u +%FT%TZ); node idle" > /tmp/claim.txt
log "KL_CHAIN_DONE ${KL_ORDER:-k1 k2}"
