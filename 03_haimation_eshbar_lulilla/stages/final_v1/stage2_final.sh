#!/bin/bash
# ESBAR-FINAL stage 2 on p8 worker (detached): waits for STAGE1_DONE; the served head (sdv1_head refit -> fit-t --rule l01post_di -> score,
# CPU) into the final export; the general regression read (dev 7,620 + when2call, each model's own served head: final / pilot / liquamen);
# the final engine eval on the pilot's held-out sets (8 GPUs); held-out pool files (if present) for final / pilot / liquamen; analysis
# (final vs liquamen, final vs pilot, reusing the pilot-run liquamen / pilot outputs); S3 read-back verify; uploads.
unset ASCALE_RUN_ID
export OMP_NUM_THREADS=1 PATH=/data/local/venv_t2141/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
C=/data/local/esbar_code_v2
F=/data/local/esbar_final_v1b
P=/data/local/esbar_pilot_v1
RUN=$F/run_final
MAN=$F/manifests
FF=$F/features_final
HD=$F/head_final
ENG=$C/engine
KIT=/data/local/heldout_loom/kit
LIQ=/data/local/heldout_loom/model
POOLS=/data/local/esbar_data/heldout_pools
S3R=s3://YOUR_BUCKET/jev-race/engineering/esbar/runs/stage1_final_v1
log() { echo "[stage2 $(date -u +%FT%TZ)] $*" | tee -a $F/logs/stage2.log; }
until [ -f $F/STAGE1_DONE ]; do sleep 30; done
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
  { OMP_NUM_THREADS=48 $SDH refit --features $FF/headfit.pt --manifest $MAN/headfit.jsonl --expect "$(cat $FF/headfit.expect.json)" --pinned-code $C/trainers --out $HD \
    && OMP_NUM_THREADS=48 $SDH fit-t --features $FF/cal.pt --manifest $MAN/cal.jsonl --expect "$(cat $FF/cal.expect.json)" --pinned-code $C/trainers --heads $HD --rule l01post_di \
    && OMP_NUM_THREADS=48 $SDH score --features $FF/dev.pt --manifest $MAN/dev.jsonl --expect "$(cat $FF/dev.expect.json)" --pinned-code $C/trainers --heads $HD \
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

# general regression read: dev (all 7,620) + when2call; all three models extracted here with esbar_code_v2
OMP_NUM_THREADS=32 python $F/scripts/head_read2.py $F/head_read.json $MAN final dev,when2call_4opt_clean_dev \
  final=$RUN/export:$FF pilot=$P/run_pilot/export:$F/features_pilot liquamen=$LIQ:$F/features_liquamen > $F/logs/head_read.log 2>&1 \
  && log "HEAD_READ ok" || log "HEAD_READ_FAILED $(tail -3 $F/logs/head_read.log)"

( bash $F/scripts/verify_s3.sh > /dev/null 2>&1; log "VERIFY_S3 $(tail -1 $F/logs/verify_s3.txt)" ) &
VS=$!

engine_eval $F/eval_final $RUN/export eval_engine.py || log "final eval had failures"
PSPEC=""
if [ -d $POOLS ]; then
  for f in $POOLS/*.jsonl $POOLS/*.jsonl.gz; do [ -f $f ] || continue; b=$(basename $f); b=${b%.gz}; b=${b%.jsonl}; PSPEC="$PSPEC pool:$b=$f"; done
fi
if [ -n "$PSPEC" ]; then
  log "POOLS found:$PSPEC"
  ls -la $POOLS > $F/logs/pools_listing.txt; md5sum $POOLS/* >> $F/logs/pools_listing.txt 2>/dev/null
  engine_eval $F/eval_pools_final $RUN/export eval_engine2.py $PSPEC || log "pools final had failures"
  engine_eval $F/eval_pools_pilot $P/run_pilot/export eval_engine2.py $PSPEC || log "pools pilot had failures"
  engine_eval $F/eval_pools_liquamen $LIQ eval_engine2.py $PSPEC || log "pools liquamen had failures"
  A_F="final=$F/eval_final,$F/eval_pools_final"; A_P="pilot=$P/eval_pilot,$F/eval_pools_pilot"; A_L="liquamen=$P/eval_liquamen,$F/eval_pools_liquamen"
else
  log "POOLS none at $POOLS"
  A_F="final=$F/eval_final"; A_P="pilot=$P/eval_pilot"; A_L="liquamen=$P/eval_liquamen"
fi
python $F/scripts/analyze2.py $F/heldout_results.json final $A_F $A_L $A_P > $F/logs/analyze.log 2>&1 && log "ANALYZE ok" || log "ANALYZE_FAILED"
wait $VS
for d in eval_final eval_pools_final eval_pools_pilot eval_pools_liquamen; do [ -d $F/$d ] && aws s3 sync --only-show-errors $F/$d/ $S3R/$d/; done
for f in $F/head_read.json $F/heldout_results.json $F/logs/*.log $F/logs/*.txt; do [ -f $f ] && aws s3 cp --only-show-errors $f $S3R/report/$(basename $f); done
aws s3 sync --only-show-errors $F/scripts/ $S3R/scripts/
log "STAGE2_DONE"
touch $F/STAGE2_DONE
