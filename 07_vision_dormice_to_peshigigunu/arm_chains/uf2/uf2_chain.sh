#!/bin/bash
# data fork (round F): plain arm `uf2` = arm uf1 without its 10,000 chart add-on rows (ArxivQA, ChartQA): UE2's recipe on manifest uf2_nc_ade
# (289,200 rows; 4,519 steps = one pass, the last step wraps 16 rows). Same chain as uf1 (9b0f44d3) with every name changed.
# Node-side chain, re-runnable, modelled on fork A's twin_chain.sh (4f566e60) with every name, path and marker renamed (nothing here matches
# fork A's chains): restage from S3 (base model checked shard by shard) -> smoke once (S3 marker) -> train with S3 checkpoints every 100 steps
# (resumes) -> dev gates 1,000 / 2,000 / 3,000 / 4,000 / 4,519 (stop only on collapse, dev < 85) -> final dev >= 89.93 or no public read ->
# RBH two numbers (reported, not gating) -> proxy HELD read with the UE2 reference -> ONE public no-WG read. No soup sweep. No Winoground on
# this node (WG=0: the launcher never touches eval_restricted). Prereg: AGENT_NOTES 2026-11-12 (PREREG Arm UF2, 2026-11-13).
# usage: uf2_chain.sh <node tag: p12> <run name: uf2>
set -u
export AWS_DEFAULT_REGION=us-east-2
TAG=$1; NAME=$2
Q=/data/local/qv; UW=/data/local/uf2; L=$UW/logs/uf2_chain_$NAME.log
B=s3://YOUR_BUCKET/jev-race; V=$B/engineering/vision; A=$V/recipe_a; U=$V/recipe_f/uf2; C=$B/fleetq/claims
MANIFEST=uf2_nc_ade; MSHA=df059f4d8aa72f000eabe808025ef89bfb9f3c94fc6a0e93aada174578df15d2; STEPS=4519; GSTEPS=1000,2000,3000,4000,4519
mkdir -p $UW/logs $UW/code $Q/logs $Q/code $Q/kit $Q/dev $Q/public $Q/data/img
log(){ echo "[uf2 $(date -u +%H:%M:%SZ)] $*" >> $L; }
exec 9> $UW/chain.lock; flock -n 9 || { log "another uf2 chain holds the lock on this node: exit"; exit 0; }
log "START $TAG $NAME manifest=$MANIFEST steps=$STEPS seed=20261114 r16/a32 host=$(hostname)"
mirror(){ aws s3 cp --quiet $L $U/logs/uf2_chain_$NAME.$TAG.log; }
aws s3 sync --quiet $U/code/ $UW/code/ --exclude uf2_chain.sh && (cd $UW/code && md5sum -c MD5SUMS > $UW/logs/chain_code_md5.log 2>&1) && log "CODE launcher $(md5sum < $UW/code/qv_arm_uf.sh | cut -c1-8) chain $(md5sum < $UW/code/uf2_chain.sh | cut -c1-8)" || { log CODE_FETCH_FAIL; mirror; exit 3; }
AR=$UW/code/qv_arm_uf.sh
aws s3 ls $U/state/${NAME}_chain_done > /dev/null 2>&1 && { log "CHAIN_DONE already on S3"; exit 0; }
aws s3 ls $U/state/${NAME}_fail > /dev/null 2>&1 && { log "FAIL marker on S3: not relaunching; remove $U/state/${NAME}_fail to retry"; exit 5; }
[ "$(nvidia-smi -L 2>/dev/null | wc -l)" = "8" ] || { log "NOT_8_GPUS"; mirror; exit 6; }
echo started | aws s3 cp --quiet - $U/state/${NAME}_started
NL=$(aws s3 cp --quiet $U/state/${NAME}_launches - 2>/dev/null | tr -dc 0-9 | cut -c1-3); NL=$(( ${NL:-0} + 1 )); echo $NL | aws s3 cp --quiet - $U/state/${NAME}_launches; log "chain launch number $NL"
[ $NL -le 15 ] || { log "TOO_MANY_LAUNCHES ($NL): something restarts in a loop; holding"; echo "more than 15 chain launches $(date -u +%FT%TZ) on $TAG" | aws s3 cp --quiet - $U/state/${NAME}_fail; mirror; exit 5; }
aws s3 ls $C/ | grep -q "${TAG}__dataF-${NAME}__" || echo "data fork (round F): arm $NAME = UE2's recipe on uf2_nc_ade (289,200 rows, 4,519 steps), all 8 GPUs ~12 h, node-side chain uf2_chain.sh, laptop guard ops/recipe_f/uf2/uf2_guard.sh; written $(date -u +%Y-%m-%dT%H:%M:%SZ)" | aws s3 cp --quiet - $C/${TAG}__dataF-${NAME}__$(( $(date -u +%s) + 64800 ))
if [ ! -f $Q/UF2_RESTAGE_OK ]; then
  aws s3 sync --quiet $A/code/qv/ $Q/code/ && (cd $Q/code && md5sum -c MD5SUMS > $Q/logs/code_md5.log 2>&1) && log CODE_OK || { log CODE_MD5_BAD; mirror; exit 3; }
  ( bash $Q/code/qv_env.sh; log "ENV $(tail -2 $Q/logs/env.log | tr '\n' ' ')" ) &
  ( [ -d $Q/kit/kit_vis ] || { aws s3 cp --quiet $A/code/kit_vis.tgz $Q/kit/ && tar xzf $Q/kit/kit_vis.tgz -C $Q/kit; }
    D=$V/recipe_dev; [ -f $Q/dev/dev_rows.jsonl ] || { aws s3 cp --quiet $D/dev_rows.jsonl $Q/dev/ && aws s3 cp --quiet $D/dev_rows.header.json $Q/dev/ && aws s3 cp --quiet $D/img_dev.tar $Q/dev/ && tar xf $Q/dev/img_dev.tar -C $Q/dev && rm -f $Q/dev/img_dev.tar; }
    [ "$(sha256sum $Q/dev/dev_rows.jsonl | cut -d' ' -f1)" = "082d1eceec1f34ce122871f133f07e1200807c3ad19d5a8fe9acd3f30a75a72d" ] && log DEV_OK || log DEV_SHA_BAD
    if [ ! -f $Q/public/suite/vision-rows-noWG.jsonl ]; then  # the no-Winoground suite only; nothing from eval_restricted is ever fetched here
      aws s3 cp --quiet $V/suite/suite_noWG.tar $Q/public/ && aws s3 cp --quiet $V/suite/suite_noWG.tar.md5 $Q/public/
      [ "$(awk '{print $1}' $Q/public/suite_noWG.tar.md5)" = "$(md5sum < $Q/public/suite_noWG.tar | cut -d' ' -f1)" ] && tar xf $Q/public/suite_noWG.tar -C $Q/public && rm -f $Q/public/suite_noWG.tar || log SUITE_MD5_BAD
      [ -f $Q/public/suite/vision-rows-noWG.jsonl ] || aws s3 cp --quiet $V/work/vision-rows-noWG.jsonl $Q/public/suite/vision-rows-noWG.jsonl
    fi
    [ "$(sha256sum $Q/public/suite/vision-rows-noWG.jsonl | cut -d' ' -f1)" = "0d44a006e575363aee592dd7a03a5ae45cf095c45eef6ce74079e8a040ef5d3c" ] && log PUBLIC_SUITE_OK || log PUBLIC_ROWS_SHA_BAD ) &
  wait
  # base model: every shard the index names must be there, complete, and hold the tensors the index assigns to it (and match the Hub's sha256)
  MC=1; for try in 1 2 3; do
    $Q/venv/bin/python $UW/code/uf2_modelcheck.py $Q/models/qwen38_27b 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 > $UW/logs/modelcheck_$try.log 2>&1; MC=$?
    log "MODELCHECK try $try rc=$MC $(tail -1 $UW/logs/modelcheck_$try.log | cut -c1-300)"
    [ $MC -eq 0 ] && break
    HF_HUB_DISABLE_XET=1 $Q/venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.8-27B', revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0', local_dir='/data/local/qv/models/qwen38_27b', token=False, max_workers=16)" >> $Q/logs/env.log 2>&1
  done
  bad=$(cat $L $Q/logs/env.log | grep -c -E "_BAD|ENV_FUNCTIONAL_FAIL|MODEL_FAIL")
  if [ -f $Q/venv.functional.ok ] && [ $MC -eq 0 ] && [ "$bad" = "0" ]; then touch $Q/UF2_RESTAGE_OK; log RESTAGE_OK; else
    log "RESTAGE_FAIL bad=$bad modelcheck_rc=$MC"; n=$(aws s3 cp --quiet $U/state/${NAME}_restage_fails - 2>/dev/null || echo 0); echo $(( ${n:-0} + 1 )) | aws s3 cp --quiet - $U/state/${NAME}_restage_fails; mirror; exit 4; fi
fi
UA="MAN=$MANIFEST SHA=$MSHA LORA_R=16 LORA_A=32 SEED=20261114 RBH_DIAG=1 HELD=1 REF=ue2_noade/ckpt_03700 WG=0"; cd $Q
if ! aws s3 ls $U/state/${NAME}_${TAG}_smoke_ok > /dev/null 2>&1 && ! aws s3 ls $A/runs/$NAME/ckpt/ 2>/dev/null | grep -q ckpt_; then
  awk 'NR%46==1' $Q/dev/dev_rows.jsonl > $Q/logs/uf_dev65.jsonl; N=_${NAME}_smoke_$(date -u +%H%M%S)
  ( exec 9>&-; while sleep 5; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1; done > $Q/logs/uf2_smoke_mem.txt ) & MP=$!
  env $UA NPROC=8 PORT=29641 SAVE=2 GATE_FILE=$Q/logs/uf_dev65.jsonl PUBLIC=0 STEPS=4 GSTEPS=2,4 bash $AR $N 0 0
  PEAK=$(sort -n $Q/logs/uf2_smoke_mem.txt | tail -1); kill $MP
  sleep 20; mv $Q/runs/$N $Q/runs/${N}_local1
  env $UA NPROC=8 PORT=29641 SAVE=2 GATE_FILE=$Q/logs/uf_dev65.jsonl PUBLIC=0 RBH_SMOKE=1 STEPS=6 GSTEPS=6 bash $AR $N 0 0
  b1=$(grep -c "TRAIN_RC=0" $Q/logs/arm_$N.log); b2=$(grep -c "PULLED_S3_CKPT ckpt_00004" $Q/logs/arm_$N.log); b3=$(grep -c '"event": "gate"' $Q/runs/$N/train_log.jsonl); b4=$(grep "RBH_GATE pass" $Q/logs/arm_$N.log | grep -c '"read": 1530')
  b5=$(python3 -c "import json; d=json.load(open('$Q/runs/$N/held.json')); print(int(d['arm']['read'] > 1000 and d.get('ref', {}).get('read', 0) > 1000))" 2>/dev/null)
  # the 6 smoke steps are the arm's own first 6 steps (same manifest, same seed): rows used / skipped per step and the ADE rows among them
  b6=$(python3 $UW/code/uf2_rows_check.py $Q/data/$MANIFEST.jsonl 20261114 $Q/runs/${N}_local1/train_log.jsonl $Q/runs/$N/train_log.jsonl 2> $UW/logs/rows_check_smoke.err | tail -1)
  b7=$(echo "$b6" | grep -c '"ok": true')
  b8=$(ls -d $Q/eval_restricted/wg_suite 2>/dev/null | wc -l)
  log "smoke $N: train_ok=$b1 s3_resume=$b2 dev_reads_in_log=$b3 rbh_full_read=$b4 held_ok=${b5:-0} rows_ok=$b7 winoground_dirs=$b8 peak_train_mib=$PEAK $(grep -o '"lora_params": [0-9]*' $Q/runs/$N/train_log.jsonl | tail -1) $(grep '"event": "step"' $Q/runs/$N/train_log.jsonl | tail -1 | grep -o '"s": [0-9.]*') :: $(tail -1 $Q/logs/train_$N.out | cut -c1-200)"
  log "smoke rows: $(echo "$b6" | cut -c1-900)"
  if [ "$b1" = "2" ] && [ "$b2" = "1" ] && [ "$b3" -ge 1 ] && [ "$b4" -ge 1 ] && [ "${b5:-0}" = "1" ] && [ "$b7" = "1" ] && [ "$b8" = "0" ]; then log "SMOKE_OK $NAME"; echo ok | aws s3 cp --quiet - $U/state/${NAME}_${TAG}_smoke_ok; else log "SMOKE_FAIL $NAME: NOT launching"; echo "smoke failed $(date -u +%FT%TZ) on $TAG: train_ok=$b1 s3_resume=$b2 dev=$b3 rbh=$b4 held=${b5:-0} rows=$b7 wg=$b8" | aws s3 cp --quiet - $U/state/${NAME}_fail; mirror; exit 3; fi
  mirror
fi
log "$NAME launch"
( exec 9>&-; while sleep 600; do mirror; done ) & MIR=$!  # helper loops do not hold the chain lock
env $UA STEPS=$STEPS GSTEPS=$GSTEPS PUBLIC=1 bash $AR $NAME 85 89.93
log "$NAME arm rc=$? :: $(grep -E 'FINAL_DEV|RBH_GATE pass|HELD \{|SCORE RC' $Q/logs/arm_$NAME.log | cut -c1-700 | tr '\n' ' ')"
if grep -qE "\] DONE|PUBLIC_READ_EXISTS" $Q/logs/arm_$NAME.log; then
  python3 $UW/code/uf2_result.py $NAME $MANIFEST > $UW/logs/uf2_result.log 2>&1; log "result rc=$? $(tail -1 $UW/logs/uf2_result.log | cut -c1-900)"
  aws s3 sync --quiet $Q/runs/$NAME $A/runs/$NAME/ --exclude "ckpt_*" --exclude "public_read/rows/*"
  mirror; echo done | aws s3 cp --quiet - $U/state/${NAME}_chain_done; log CHAIN_DONE
elif grep -q "FINAL_DEV index=-1 " $Q/logs/arm_$NAME.log; then
  log "FINAL_DEV_READ_MISSING: the final checkpoint exists but its dev read was lost (node died during the read). No public read. Needs a hand: read dev on ckpt_$(printf %05d $STEPS), then decide"
  echo "final dev read missing $(date -u +%FT%TZ) on $TAG" | aws s3 cp --quiet - $U/state/${NAME}_fail; mirror
elif grep -q "FINAL_DEV_BELOW_BAR" $Q/logs/arm_$NAME.log; then
  python3 $UW/code/uf2_result.py $NAME $MANIFEST > $UW/logs/uf2_result.log 2>&1; aws s3 sync --quiet $Q/runs/$NAME $A/runs/$NAME/ --exclude "ckpt_*"
  mirror; echo below_bar | aws s3 cp --quiet - $U/state/${NAME}_chain_done; log "CHAIN_DONE (arm below its dev bar: no public read)"
elif grep -q '"why": "gate_stop"' $Q/runs/$NAME/train_log.jsonl 2>/dev/null; then
  mirror; echo dev_collapse | aws s3 cp --quiet - $U/state/${NAME}_chain_done; log "CHAIN_DONE (dev collapse stop: no public read)"
else log "ARM_NOT_FINISHED (a re-run of this chain resumes it)"; fi
kill $MIR 2>/dev/null; mirror
