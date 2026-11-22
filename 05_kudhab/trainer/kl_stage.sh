#!/bin/bash
# kl_stage.sh (KL agent, 2026-11-15): stage a p8-class worker for the karpo-layout trainer. Idempotent; every part has its own .ok.
# venv_mxk + karpo's code (a6fa38bf) + kit + suite + refs (mx_kown.sh __setup__), pinned code v2e, munku (weights only), dev.jsonl,
# the FS5 requests / teacher / student manifest, the two reference screens. karpo's weights only with KL_KARPO=1 (K2 / parity; K1 never loads them).
set -u; unset ASCALE_RUN_ID; export AWS_DEFAULT_REGION=us-east-2
D=/data/local; K=$D/kl; M=$D/mx; S3=s3://YOUR_BUCKET/jev-race; E=$S3/engineering/esbar; R=$E/runs
mkdir -p $K/logs $K/data $K/ref $M/scripts $M/logs $M/manifests; L=$K/logs/stage.log
log() { echo "[kl_stage $(date -u +%FT%TZ)] $*" | tee -a $L; aws s3 cp --only-show-errors $L $R/esbar_kl_code/logs/stage_$(hostname).log 2>/dev/null; }
die() { log "KL_STAGE_FAILED $*"; exit 1; }
log "start host $(hostname) gpus $(nvidia-smi -L | wc -l) disk $(df -h $D | tail -1 | awk '{print $4}') mem $(free -g | awk 'NR==2{print $2}')G cpus $(nproc)"
aws s3 sync --only-show-errors $R/esbar_mx/scripts/ $M/scripts/ || die MX_SCRIPTS
# a node whose karpo environment is in use (venv there and either complete or its GPUs busy, e.g. the MX agent's labelling on p10): do not run the setup probe beside it
SKIPK=0; [ -x $D/venv_mxk/bin/python ] && { [ -f $M/kroot/.ok2 ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)" != 0 ]; } && SKIPK=1
if [ $SKIPK = 1 ]; then log "karpo environment already on this node: mx_kown setup skipped (venv_mxk + kroot left as they are)"; ( true ) & else
( bash $M/scripts/mx_kown.sh __setup__ x > $K/logs/kown_setup.out 2>&1; echo "rc=$?" >> $K/logs/kown_setup.out ) & fi
P1=$!
C=$D/esbar_code_v2e
if [ ! -f $C/.ok ] || [ ! -s $C/trainers/fullft_train.py ]; then
  mkdir -p $K/tmp; aws s3 cp --only-show-errors $E/code/esbar_code_v2e_full.tgz $K/tmp/code2e.tgz && [ "$(md5sum < $K/tmp/code2e.tgz | cut -c1-32)" = e6bd8a5aa97693be477ff50b5234f7ae ] || die CODE_FETCH
  aws s3 cp --only-show-errors $E/code/esbar_code_v2e_full.md5.txt $K/tmp/code2e.md5.txt; (cd $D && tar xzf $K/tmp/code2e.tgz && md5sum -c --quiet $K/tmp/code2e.md5.txt) > $K/logs/code2e_md5.txt 2>&1 || die CODE_MD5SUMS; touch $C/.ok
fi
[ "$(md5sum < $C/trainers/fullft_train.py | cut -c1-8)" = 4048f97c ] || die CODE_MD5
log "code v2e ok"
INIT=$D/soup75_in/munku; MUNKU_ID=6a0b6e48b55a950ee1f739af7c938cea5b16ec01ef2b233a033640f9a15f8db9; mkdir -p $INIT
( if [ ! -f $INIT/.ok ]; then
    aws s3 sync --only-show-errors --exclude 'vbll_head*' --exclude '*.pt' --exclude 'READBACK_MD5.txt' $R/esbar_wi_fs1f50/export/ $INIT/ || exit 1
    ID=$( (cd $INIT && ls model*.safetensors config.json | sort | xargs -P 13 -n 1 sha256sum | sort -k2) | sha256sum | cut -c1-64)
    [ "$ID" = "$MUNKU_ID" ] || { echo "INIT_IDENTITY $ID" >> $L; exit 1; }; echo $ID > $INIT/.identity; touch $INIT/.ok
  fi ) &
P2=$!
P3=""
if [ "${KL_KARPO:-0}" = 1 ]; then ( bash $M/scripts/mx_get.sh karpo > $K/logs/get_karpo.out 2>&1 ) & P3=$!; fi
MAN=$D/esbar_v5/manifests; mkdir -p $MAN
[ -s $MAN/dev.jsonl ] || aws s3 cp --only-show-errors $S3/engineering/bayesF/stage/manifests/dev.jsonl $MAN/dev.jsonl || die NO_DEV
for x in fs5.jsonl.gz teacher_karpo_fs5.jsonl.gz mxa.jsonl mxa.jsonl.header.json mxa.rows_sha256.txt mxa.SHA256SUMS SHA256SUMS; do
  [ -s $K/data/$x ] || aws s3 cp --only-show-errors $R/esbar_mx/tl/$x $K/data/$x || die "FETCH $x"; done
[ "$(sha256sum < $K/data/mxa.jsonl | cut -c1-64)" = "$(cut -c1-64 $K/data/mxa.rows_sha256.txt)" ] || die MXA_SHA
log "data ok: mxa $(wc -l < $K/data/mxa.jsonl) rows, requests $(zcat $K/data/fs5.jsonl.gz | wc -l), teacher $(zcat $K/data/teacher_karpo_fs5.jsonl.gz | wc -l); sums: $(cat $K/data/mxa.SHA256SUMS | cut -c1-12,65-120 | tr '\n' ' ')"
for t in munku karpo; do mkdir -p $K/ref/$t; for x in prefix_results.jsonl.gz prefix_compare.json decision_config.used.json; do
  [ -s $K/ref/$t/$x ] || aws s3 cp --only-show-errors $R/esbar_mx_s_$t/kown_s4400/$x $K/ref/$t/$x || log "REF_MISSING $t $x"; done; done
wait $P1; [ $SKIPK = 1 ] || log "kown setup: $(tail -2 $K/logs/kown_setup.out | tr '\n' ' ' | cut -c1-300)"
wait $P2 || die MUNKU; log "munku ok $(cut -c1-16 $INIT/.identity) $(du -sh $INIT | cut -f1)"
if [ -n "$P3" ]; then wait $P3; log "karpo: $(tail -1 $K/logs/get_karpo.out | cut -c1-200)"; [ -f $M/src/karpo/.ok ] || die KARPO; fi
[ $SKIPK = 1 ] && [ -d $M/kroot/karpo/src/karpo_decision ] || grep -q SETUP_OK $M/work/__setup__/kown/driver.log 2>/dev/null || die KOWN_SETUP
touch $K/.staged; log "KL_STAGE_DONE"
