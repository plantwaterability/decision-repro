#!/bin/bash
# ESBAR-V2 stage 1 on p8 worker (detached, from queue.sh): train Stage-1 v2 from final v1's export (57524f2c) on esbar_stage1_final_v2
# (1862c712, 84,455 rows; one epoch = 1,320 steps, cosine over the epoch), the same Wave-6 flags; mirror each COMPLETE resume dir to S3
# (pruned by hand); upload the export; check cc_rows total = 4,921; features (v2: headfit / cal / dev / w2c / eval_fresh), stamped.
unset ASCALE_RUN_ID
export OMP_NUM_THREADS=1 PATH=/data/local/venv_t2141/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
C=/data/local/esbar_code_v2
export PYTHONPATH=$C/trainers:$C
F=/data/local/esbar_final_v2
RUN=$F/run_v2
INIT=/data/local/esbar_final_v1b/run_final/export   # final v1 (identity 57524f2c), checked by queue.sh
PM=/data/local/esbar_data/out/esbar_stage1_final_v2.jsonl
PMSHA=1862c712afd92da324b884403a039c4a86f0e9e6f9f2d5d116d9c9f0b366f388
STEPS=1320                                   # ceil(84,455 / 64): the epoch length at --epochs 1.0
S3R=s3://YOUR_BUCKET/jev-race/engineering/esbar/runs/esbar_final_v2
MAN=$F/manifests
mkdir -p $RUN $F/features_v2 $F/logs
log() { echo "[stage1 $(date -u +%FT%TZ)] $*" | tee -a $F/logs/stage1.log; }
TR="python -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --rdzv-backend c10d --local-addr 127.0.0.1"

publish_resume() {  # each COMPLETE local resume dir -> S3 (rank files, then COMPLETE). Older S3 resume dirs are pruned by hand
  for d in $RUN/resume_step*; do      # (per-object deletes run as direct, visible commands; this script deletes nothing on S3)
    [ -f $d/COMPLETE ] || continue
    st=$(basename $d)
    aws s3 ls $S3R/$st/COMPLETE > /dev/null 2>&1 && continue
    [ "$(ls $d/rank*.pt 2>/dev/null | wc -l)" -eq 8 ] || continue
    ok=1; for f in $d/rank*.pt; do aws s3 cp --only-show-errors $f $S3R/$st/$(basename $f) || ok=0; done
    [ $ok = 1 ] && [ -f $d/COMPLETE ] && aws s3 cp --only-show-errors $d/COMPLETE $S3R/$st/COMPLETE && log "RESUME_PUBLISHED $st"
    for f in $RUN/train_record.json $RUN/eval_step*.jsonl; do [ -f $f ] && aws s3 cp --only-show-errors $f $S3R/train/$(basename $f); done
  done
}

if [ ! -s $RUN/export/model.safetensors.index.json ]; then
  log "TRAIN start"
  t0=$(date +%s)
  ( while sleep 300; do publish_resume; done ) & PUB=$!
  $TR --rdzv-endpoint 127.0.0.1:29961 $C/trainers/fullft_train.py \
    --init-from $INIT --attention-policy B \
    --manifest $PM --expect-rows-sha256 $PMSHA \
    --val $MAN/dev.jsonl --val-max 2000 --eval-every 250 --eval-at-start --save-every 250 --keep-resume 1 \
    --decisions-per-step 64 --epochs 1.0 --max-steps $STEPS \
    --lr 7.5e-7 --lr-floor 0.1 --warmup 30 --weight-decay 0.0 --clip 1.0 \
    --alpha 0.35 --ineligible-gold-weight one --head on --head-prior 1 --head-kl 1 --head-cov full \
    --optim adamw-fused --param-dtype bf16 --reduce-dtype fp32 \
    --pack-tokens 8192 --ac 1 --offload-ac 0 --offload-min 8192 --evict-states-min 6145 \
    --mlp-chunk 4096 --mlp-chunk-min 16384 --seed 0 \
    --pinned-code $C/trainers --out $RUN >> $F/logs/train.log 2>&1
  rc=$?
  kill $PUB 2>/dev/null
  log "TRAIN rc=$rc wall_s=$(( $(date +%s) - t0 ))"
  CC=$(grep "^STEP" $F/logs/train.log | python3 -c "import sys, json; print(sum(json.loads(l[5:]).get('cc_rows', 0) for l in sys.stdin))")
  log "CC_ROWS_TOTAL $CC (expected 4921)"
  [ "$CC" = 4921 ] || { log "CC_ROWS_MISMATCH: stop and report"; exit 1; }
  [ $rc -eq 0 ] && [ -s $RUN/export/model.safetensors.index.json ] || { publish_resume; log "TRAIN_FAILED"; exit 1; }
fi
ES=$( (cd $RUN/export && ls model*.safetensors config.json | sort | xargs -P 13 -n 1 sha256sum | sort -k2) | tee $F/export_sha256.txt | sha256sum | cut -c1-64)
log "EXPORT_IDENTITY $ES"
( aws s3 sync --only-show-errors $RUN/export/ $S3R/export/ && log "EXPORT_UPLOADED" || log "EXPORT_UPLOAD_FAILED"
  for f in $RUN/train_record.json $RUN/eval_step*.jsonl $F/logs/train.log $F/export_sha256.txt $F/init_sha256.txt; do
    [ -f $f ] && aws s3 cp --only-show-errors $f $S3R/train/$(basename $f); done ) &
UP=$!

EXTRACTOR="$(md5sum $C/trainers/fullft_train.py | cut -c1-32)+$(md5sum $C/trainers/fullft_mem.py | cut -c1-32)"
port=29970
extract() {  # extract WHO EXPORT_DIR EXPORT_SHA ROLE
  who=$1; ed=$2; esha=$3; role=$4
  man=$MAN/$role.jsonl
  dig=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['rows_sha256'])" $man.header.json)
  Fe=$F/features_$who/$role.pt; FM=$F/features_$who/$role.stamp.json
  EXP="{\"identity_sha256\": \"esbar_final_v2:$who\", \"export_sha256\": \"$esha\", \"policy\": \"B\", \"role\": \"$role\", \"manifest_rows_sha256\": \"$dig\", \"extractor_md5\": \"$EXTRACTOR\"}"
  echo "$EXP" > $F/features_$who/$role.expect.json
  if python3 $C/trainers/sdv1_head.py check-features --features $Fe --meta $FM --manifest $man --expect "$EXP" > /dev/null 2>&1; then
    log "FEATURES_VALID $who $role (cached)"; return 0; fi
  port=$((port + 1))
  t0=$(date +%s)
  $TR --rdzv-endpoint 127.0.0.1:$port $C/trainers/fullft_train.py --init-from $ed --attention-policy B --seed 0 \
    --param-dtype bf16 --reduce-dtype fp32 --pack-tokens 8192 --ac 0 --offload-ac 0 --mlp-chunk 4096 --mlp-chunk-min 16384 \
    --pinned-code $C/trainers --manifest $man --expect-rows-sha256 $dig --extract-features $Fe \
    --out $F/features_$who/tmp_$role > $F/logs/features_${who}_$role.log 2>&1
  rc=$?
  log "FEATURES $who $role rc=$rc wall_s=$(( $(date +%s) - t0 ))"
  [ $rc -eq 0 ] || { log "FEATURES_FAILED $who $role"; return 1; }
  python3 $C/trainers/sdv1_head.py stamp-features --features $Fe --meta $FM --manifest $man --expect "$EXP" >> $F/logs/stage1.log 2>&1 \
    || { log "STAMP_FAILED $who $role"; return 1; }
}
LS=69aaa53af0a3a44e9ef9aab8860c3d2e523ca93e7ae893ac80d5463f809d5ef2
ok=1
for role in headfit cal dev when2call_4opt_clean_dev eval_fresh; do extract v2 $RUN/export $ES $role || ok=0; done
wait $UP
for who in v2; do aws s3 sync --only-show-errors --exclude 'tmp_*' $F/features_$who/ $S3R/features_$who/ || log "FEATURES_UPLOAD_FAILED $who"; done
[ $ok = 1 ] && { log "STAGE1_DONE"; touch $F/STAGE1_DONE; } || log "STAGE1_INCOMPLETE"
