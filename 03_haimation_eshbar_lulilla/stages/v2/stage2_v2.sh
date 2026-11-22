#!/bin/bash
# ESBAR-V2 stage 2 on p8 worker (from queue.sh): v2's served head (sdv1_head refit -> fit-t --rule l01post_di -> score, CPU) into its
# export; served-head reads REF v2 (v2 / final v1 / liquamen) on dev, w2c, eval_fresh; v2's engine eval on the held-out files + held-out
# pools + libeval_clean (8 GPUs); analysis v2 - final v1 and v2 - liquamen (final v1 / liquamen held-out outputs reused, their libeval
# from queue.sh step 2); the all-K Bayesian A-E (label_rows.py, then bayes_ae_e2.py --ref 64,0 [pre-registered] and --ref 1024,0
# [exploratory]); S3 read-back verify; uploads.
unset ASCALE_RUN_ID
export OMP_NUM_THREADS=1 PATH=/data/local/venv_t2141/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
C=/data/local/esbar_code_v2
F=/data/local/esbar_final_v2
V=/data/local/esbar_final_v1b
P=/data/local/esbar_pilot_v1
RUN=$F/run_v2
MAN=$F/manifests
FV=$F/features_v2
HD=$F/head_v2
ENG=$C/engine
KIT=/data/local/heldout_loom/kit
LIQ=/data/local/heldout_loom/model
POOLS=/data/local/esbar_data/heldout_pools
LIB=/data/local/esbar_data/heldout_library/libeval_clean.jsonl
AK=/data/local/esbar_data/evalr/allk
S3R=s3://YOUR_BUCKET/jev-race/engineering/esbar/runs/esbar_final_v2
log() { echo "[stage2 $(date -u +%FT%TZ)] $*" | tee -a $F/logs/stage2.log; }
[ -f $F/STAGE1_DONE ] || { log "STAGE1 not done"; exit 1; }
log "start"

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

if [ ! -s $RUN/export/vbll_head.json ]; then
  mkdir -p $HD
  SDH="python $C/trainers/sdv1_head.py"
  t0=$(date +%s)
  { OMP_NUM_THREADS=48 $SDH refit --features $FV/headfit.pt --manifest $MAN/headfit.jsonl --expect "$(cat $FV/headfit.expect.json)" --pinned-code $C/trainers --out $HD \
    && OMP_NUM_THREADS=48 $SDH fit-t --features $FV/cal.pt --manifest $MAN/cal.jsonl --expect "$(cat $FV/cal.expect.json)" --pinned-code $C/trainers --heads $HD --rule l01post_di \
    && OMP_NUM_THREADS=48 $SDH score --features $FV/dev.pt --manifest $MAN/dev.jsonl --expect "$(cat $FV/dev.expect.json)" --pinned-code $C/trainers --heads $HD \
       --out $HD/dev_predictions.pt; } > $F/logs/head.log 2>&1
  rc=$?
  log "HEAD rc=$rc wall_s=$(( $(date +%s) - t0 ))"
  [ $rc -eq 0 ] || { log "HEAD_FAILED: $(tail -3 $F/logs/head.log)"; exit 1; }
  cp $HD/serving/vbll_head.safetensors $HD/serving/vbll_head.json $RUN/export/ || { log "HEAD_COPY_FAILED"; exit 1; }
fi
log "HEAD in export: $(sha256sum $RUN/export/vbll_head.safetensors | cut -c1-64) T=$(python -c "import json; print(json.load(open('$RUN/export/vbll_head.json'))['temperature'])")"
( aws s3 sync --only-show-errors $HD/ $S3R/head/ && aws s3 cp --only-show-errors $F/logs/head.log $S3R/head/head.log \
  && aws s3 cp --only-show-errors $RUN/export/vbll_head.safetensors $S3R/export/vbll_head.safetensors \
  && aws s3 cp --only-show-errors $RUN/export/vbll_head.json $S3R/export/vbll_head.json && log "HEAD_UPLOADED" ) || log "HEAD_UPLOAD_FAILED"

OMP_NUM_THREADS=32 python $F/scripts/head_read2.py $F/head_read_refV2.json $MAN v2 dev,when2call_4opt_clean_dev,eval_fresh \
  v2=$RUN/export:$FV final=$V/run_final/export:$V/features_final liquamen=$LIQ:$V/features_liquamen > $F/logs/head_read_refV2.log 2>&1 \
  && log "HEAD_READ refV2 ok" || log "HEAD_READ_FAILED refV2 $(tail -3 $F/logs/head_read_refV2.log)"

( bash $F/scripts/verify_s3.sh > /dev/null 2>&1; log "VERIFY_S3 $(tail -1 $F/logs/verify_s3.txt)" ) &
VS=$!

# all-K Bayesian A-E (CPU) in the background while the GPUs run the engine evals
( cd /tmp && OMP_NUM_THREADS=16 python3 $AK/label_rows.py --model $RUN/export --features $FV/cal.pt --out $FV/label_rows.pt > $F/logs/label_rows.log 2>&1 \
  && log "LABEL_ROWS ok (label_rows.py md5 $(md5sum < $AK/label_rows.py | cut -c1-32))" || { log "LABEL_ROWS_FAILED $(tail -3 $F/logs/label_rows.log)"; exit 1; }
  for ref in 64,0 1024,0; do
    tag=${ref%,*}
    ( t0=$(date +%s); OMP_NUM_THREADS=24 python3 $AK/bayes_ae_e2.py --code $C/trainers --heads $HD --cal $FV/cal.pt --label-rows $FV/label_rows.pt \
        --train-e $FV/headfit.pt --ref $ref --eval fresh=$FV/eval_fresh.pt dev=$FV/dev.pt w2c=$FV/when2call_4opt_clean_dev.pt \
        --out $F/bayes_ae_e2_v2_ref$tag.json > $F/logs/bayes_ae_e2_v2_ref$tag.log 2>&1
      log "BAYES_AE_E2 ref $ref rc=$? wall_s=$(( $(date +%s) - t0 )) (bayes_ae_e2.py md5 $(md5sum < $AK/bayes_ae_e2.py | cut -c1-32))" ) &
  done
  wait ) &
BA=$!

engine_eval $F/eval_v2 $RUN/export eval_engine.py || log "v2 eval had failures"
PSPEC=""
for f in $POOLS/*.jsonl; do [ -f $f ] || continue; b=$(basename $f .jsonl); PSPEC="$PSPEC pool:$b=$f"; done
ls -la $POOLS > $F/logs/pools_listing.txt; md5sum $POOLS/*.jsonl $LIB >> $F/logs/pools_listing.txt
engine_eval $F/eval_pools_v2 $RUN/export eval_engine2.py $PSPEC || log "pools v2 had failures"
engine_eval $F/eval_libeval_v2 $RUN/export eval_engine2.py libeval=$LIB || log "libeval v2 had failures"
A_2="v2=$F/eval_v2,$F/eval_pools_v2,$F/eval_libeval_v2"
A_F="final=$V/eval_final,$V/eval_pools_final,$F/eval_libeval_final"
A_L="liquamen=$P/eval_liquamen,$V/eval_pools_liquamen,$F/eval_libeval_liquamen"
python $F/scripts/analyze3.py $F/heldout_results_refV2.json v2 $A_2 $A_F $A_L > $F/logs/analyze_refV2.log 2>&1 && log "ANALYZE refV2 ok" || log "ANALYZE_FAILED refV2"
wait $VS
wait $BA
for d in eval_v2 eval_pools_v2 eval_libeval_v2; do aws s3 sync --only-show-errors $F/$d/ $S3R/$d/; done
aws s3 cp --only-show-errors $FV/label_rows.pt $S3R/features_v2/label_rows.pt
for f in $F/head_read_ref*.json $F/heldout_results_ref*.json $F/bayes_ae_e2_v2_ref*.json $F/logs/*.log $F/logs/*.txt; do [ -f $f ] && aws s3 cp --only-show-errors $f $S3R/report/$(basename $f); done
aws s3 sync --only-show-errors $F/scripts/ $S3R/scripts/
log "STAGE2_DONE"
touch $F/STAGE2_DONE
