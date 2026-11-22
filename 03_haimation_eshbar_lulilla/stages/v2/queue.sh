#!/bin/bash
# p8 queue (coordinator 11-02, unattended through the ~07:40Z outage window):
#   1. control: wait for ctrl_post_a (export + S3), run ctrl_post_b (features -> STAGE1_DONE); stage2_ctrl.sh (already waiting, pid 30240)
#      then runs head / reads / engine evals; wait for it to exit.
#   2. libeval_clean engine eval for final v1, the control and liquamen; libeval three-way (REF final: final / ctrl / liquamen).
#   3. Stage-1 v2: checks (manifest 1862c712, 84,455 rows, 4,921 cc_logit_offset rows; init = final v1 export 57524f2c), manifests,
#      held-out leakage check vs the v2 manifest (CPU, background), stage1_v2.sh (train + export + features).
#   4. stage2_v2.sh (head, reads, engine evals incl. pools + libeval, analysis, all-K Bayesian A-E).
# Every step is skipped when its marker exists, so a rerun after an interruption continues.
unset ASCALE_RUN_ID
export OMP_NUM_THREADS=1 PATH=/data/local/venv_t2141/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
CT=/data/local/esbar_ctrl_base_v1
F=/data/local/esbar_final_v2
V=/data/local/esbar_final_v1b
C=/data/local/esbar_code_v2
ENG=$C/engine
KIT=/data/local/heldout_loom/kit
LIQ=/data/local/heldout_loom/model
LIB=/data/local/esbar_data/heldout_library/libeval_clean.jsonl
PM=/data/local/esbar_data/out/esbar_stage1_final_v2.jsonl
PMSHA=1862c712afd92da324b884403a039c4a86f0e9e6f9f2d5d116d9c9f0b366f388
S3R=s3://YOUR_BUCKET/jev-race/engineering/esbar/runs/esbar_final_v2
mkdir -p $F/logs $F/manifests
log() { echo "[queue $(date -u +%FT%TZ)] $*" | tee -a $F/logs/queue.log; }
engine_eval() {  # engine_eval OUTDIR MODEL_DIR SCRIPT [SET=PATH ...]
  od=$1; md=$2; sc=$3; shift 3; mkdir -p $od
  t0=$(date +%s); pids=""
  for g in 0 1 2 3 4 5 6 7; do
    (cd $ENG && CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=4 PYTHONPATH=$KIT:$ENG python $F/scripts/$sc $md $od/shard$g.jsonl $g 8 "$@" \
       > $od/shard$g.log 2>&1) &
    pids="$pids $!"
  done
  rc=0; for p in $pids; do wait $p || rc=1; done
  log "ENGINE_EVAL $(basename $od) rc=$rc rows=$(cat $od/shard*.jsonl | wc -l) errs=$(cat $od/shard*.jsonl | grep -c '"err": "') wall_s=$(( $(date +%s) - t0 ))"
  return $rc
}
log "start"

# ---- 1. control
until [ -f $CT/POST_A_DONE ] || [ -f /workspace/.ascale/jobs/esbar_ctrl_post_a.rc ]; do sleep 30; done
if [ -f $CT/POST_A_DONE ]; then
  if [ ! -f $CT/STAGE1_DONE ]; then log "ctrl_post_b start"; bash $CT/scripts/ctrl_post_b.sh; fi
  [ -f $CT/STAGE1_DONE ] && log "ctrl STAGE1_DONE" || log "CTRL_POST_B_FAILED (control evals skipped)"
  if [ -f $CT/STAGE1_DONE ]; then
    while ps -p 30240 > /dev/null; do sleep 30; done
    log "stage2_ctrl exited: $([ -f $CT/STAGE2_DONE ] && echo STAGE2_DONE || echo 'NOT done')"
  fi
else
  log "CTRL_POST_A_FAILED: $(tail -2 $CT/logs/stage1.log | tr '\n' ' ')"
fi

# ---- 2. libeval for final v1 / control / liquamen
if [ ! -f $F/LIBEVAL_OTHERS_DONE ]; then
  engine_eval $F/eval_libeval_final $V/run_final/export eval_engine2.py libeval=$LIB
  [ -s $CT/run_ctrl/export/vbll_head.json ] && engine_eval $F/eval_libeval_ctrl $CT/run_ctrl/export eval_engine2.py libeval=$LIB
  engine_eval $F/eval_libeval_liquamen $LIQ eval_engine2.py libeval=$LIB
  if [ -d $F/eval_libeval_ctrl ]; then
    python $F/scripts/analyze3.py $F/libeval_refFinal.json final final=$F/eval_libeval_final ctrl=$F/eval_libeval_ctrl liquamen=$F/eval_libeval_liquamen \
      > $F/logs/analyze_libeval_refFinal.log 2>&1 && log "ANALYZE libeval refFinal ok" || log "ANALYZE_FAILED libeval refFinal"
  fi
  for d in eval_libeval_final eval_libeval_ctrl eval_libeval_liquamen; do [ -d $F/$d ] && aws s3 sync --only-show-errors $F/$d/ $S3R/$d/; done
  [ -f $F/libeval_refFinal.json ] && aws s3 cp --only-show-errors $F/libeval_refFinal.json $S3R/report/libeval_refFinal.json
  touch $F/LIBEVAL_OTHERS_DONE
fi

# ---- 3. v2 checks + training + features
if [ ! -f $F/STAGE1_DONE ]; then
  got=$(sha256sum < $PM | cut -c1-64)
  hdr=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['rows_sha256'])" $PM.header.json)
  n=$(wc -l < $PM)
  cc=$(python3 -c "
import json, sys
n = 0
for l in open(sys.argv[1]):
    if 'cc_logit_offset' in l and json.loads(l).get('cc_logit_offset') is not None:
        n += 1
print(n)" $PM)
  log "v2 manifest sha256 $got header $hdr rows $n cc_offset_rows $cc md5 $(md5sum < $PM | cut -c1-32)"
  [ "$got" = "$PMSHA" ] && [ "$hdr" = "$PMSHA" ] && [ "$n" = 84455 ] && [ "$cc" = 4921 ] || { log "V2_MANIFEST_CHECK_FAILED: stop"; exit 1; }
  ( cd $V/run_final/export && ls model*.safetensors config.json | sort | xargs -P 13 -n 1 sha256sum | sort -k2 ) > $F/init_sha256.txt
  ID=$(sha256sum < $F/init_sha256.txt | cut -c1-64)
  log "v2 INIT_IDENTITY (final v1 export) $ID"
  [ "$ID" = 57524f2ca80196adce57f3e16418312c31f4d1d3406a31f6ed47b2927cfdcb4c ] || { log "V2_INIT_IDENTITY_MISMATCH: stop"; exit 1; }
  for m in headfit cal dev when2call_4opt_clean_dev eval_fresh; do
    cp $CT/manifests/$m.jsonl $CT/manifests/$m.jsonl.header.json $F/manifests/
    g=$(sha256sum < $F/manifests/$m.jsonl | cut -c1-64); w=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['rows_sha256'])" $F/manifests/$m.jsonl.header.json)
    [ "$g" = "$w" ] || { log "MANIFEST_MISMATCH $m: stop"; exit 1; }
  done
  log "v2 manifests copied (headfit / cal / dev / w2c / eval_fresh sha256 = headers)"
  ( cd /tmp && nice -n 10 python $F/scripts/decontam_v2.py > $F/logs/decontam_v2.log 2>&1 ) &
  log "v2 stage1 start"
  bash $F/scripts/stage1_v2.sh
  [ -f $F/STAGE1_DONE ] || { log "V2_STAGE1_FAILED: stop ($(tail -2 $F/logs/stage1.log | tr '\n' ' '))"; exit 1; }
fi

# ---- 4. v2 head, reads, evals, Bayesian A-E
if [ ! -f $F/STAGE2_DONE ]; then
  log "v2 stage2 start"
  bash $F/scripts/stage2_v2.sh
fi
aws s3 cp --only-show-errors $F/logs/queue.log $S3R/report/queue.log
[ -f $F/logs/decontam_v2.log ] && aws s3 cp --only-show-errors $F/logs/decontam_v2.log $S3R/report/decontam_v2.log
log "QUEUE_DONE $([ -f $F/STAGE2_DONE ] && echo ok || echo 'v2 stage2 incomplete')"
