#!/bin/bash
# ESBAR-FINAL stage 1 on p8 worker (detached): train Stage-1 final v1 from liquamen (one full epoch, cosine over the epoch), mirror each
# COMPLETE resume dir to S3, upload the export, then extract features (final: headfit / cal / dev / w2c; liquamen: dev) with the same
# trainer code, stamped for sdv1_head. The guard manifest is not used anywhere.
unset ASCALE_RUN_ID
export OMP_NUM_THREADS=1 PATH=/data/local/venv_t2141/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
C=/data/local/esbar_code_v2
export PYTHONPATH=$C/trainers:$C
F=/data/local/esbar_final_v1b
RUN=$F/run_final
INIT=$F/init
PM=/data/local/esbar_data/out/esbar_stage1_final_v1.jsonl
PMSHA=20a8337ef9064640cdd4d3786a6fec02281cb3c909b31b96c339bd7a7f386708
STEPS=1369                                   # ceil(87,601 / 64): the trainer's epoch length at --epochs 1.0
S3R=s3://YOUR_BUCKET/jev-race/engineering/esbar/runs/stage1_final_v1
MAN=$F/manifests
mkdir -p $RUN $F/features_final $F/features_liquamen $F/features_pilot $F/logs
log() { echo "[stage1 $(date -u +%FT%TZ)] $*" | tee -a $F/logs/stage1.log; }
TR="python -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --rdzv-backend c10d --local-addr 127.0.0.1"

publish_resume() {  # each COMPLETE local resume dir -> S3 (rank files, then COMPLETE); then older S3 resume dirs removed
  for d in $RUN/resume_step*; do
    [ -f $d/COMPLETE ] || continue
    st=$(basename $d)
    aws s3 ls $S3R/$st/COMPLETE > /dev/null 2>&1 && continue
    [ "$(ls $d/rank*.pt 2>/dev/null | wc -l)" -eq 8 ] || continue
    ok=1; for f in $d/rank*.pt; do aws s3 cp --only-show-errors $f $S3R/$st/$(basename $f) || ok=0; done
    [ $ok = 1 ] && [ -f $d/COMPLETE ] && aws s3 cp --only-show-errors $d/COMPLETE $S3R/$st/COMPLETE && log "RESUME_PUBLISHED $st"
    if [ $ok = 1 ]; then
      for old in $(aws s3 ls $S3R/ | awk '$1 == "PRE" {print $2}' | grep '^resume_step' | sed 's#/$##'); do
        [ "$old" = "$st" ] && continue
        [ "${old#resume_step}" -lt "${st#resume_step}" ] && aws s3 rm --only-show-errors --recursive $S3R/$old/ && log "RESUME_S3_PRUNED $old"
      done
    fi
    for f in $RUN/train_record.json $RUN/eval_step*.jsonl; do [ -f $f ] && aws s3 cp --only-show-errors $f $S3R/train/$(basename $f); done
  done
}

if [ ! -s $RUN/export/model.safetensors.index.json ]; then
  log "TRAIN start"
  t0=$(date +%s)
  ( while sleep 300; do publish_resume; done ) & PUB=$!
  $TR --rdzv-endpoint 127.0.0.1:29891 $C/trainers/fullft_train.py \
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
  [ $rc -eq 0 ] && [ -s $RUN/export/model.safetensors.index.json ] || { publish_resume; log "TRAIN_FAILED"; exit 1; }
fi
ES=$( (cd $RUN/export && ls model*.safetensors config.json | sort | xargs -P 13 -n 1 sha256sum | sort -k2) | tee $F/export_sha256.txt | sha256sum | cut -c1-64)
log "EXPORT_IDENTITY $ES"
( aws s3 sync --only-show-errors $RUN/export/ $S3R/export/ && log "EXPORT_UPLOADED" || log "EXPORT_UPLOAD_FAILED"
  for f in $RUN/train_record.json $RUN/eval_step*.jsonl $F/logs/train.log $F/export_sha256.txt $F/init_sha256.txt; do
    [ -f $f ] && aws s3 cp --only-show-errors $f $S3R/train/$(basename $f); done ) &
UP=$!

EXTRACTOR="$(md5sum $C/trainers/fullft_train.py | cut -c1-32)+$(md5sum $C/trainers/fullft_mem.py | cut -c1-32)"
port=29900
extract() {  # extract WHO EXPORT_DIR EXPORT_SHA ROLE
  who=$1; ed=$2; esha=$3; role=$4
  man=$MAN/$role.jsonl
  dig=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['rows_sha256'])" $man.header.json)
  Fe=$F/features_$who/$role.pt; FM=$F/features_$who/$role.stamp.json
  EXP="{\"identity_sha256\": \"esbar_final_v1b:$who\", \"export_sha256\": \"$esha\", \"policy\": \"B\", \"role\": \"$role\", \"manifest_rows_sha256\": \"$dig\", \"extractor_md5\": \"$EXTRACTOR\"}"
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
for role in headfit cal dev when2call_4opt_clean_dev; do extract final $RUN/export $ES $role || ok=0; done
for role in dev when2call_4opt_clean_dev; do extract liquamen $INIT $LS $role || ok=0; done
PES=4ba4b28161fa685583a17e85bb1bd3ee0f8b56eec925a2a04799c56611363b6e   # the pilot export identity (stage1_pilot export_sha256.txt)
for role in dev when2call_4opt_clean_dev; do extract pilot /data/local/esbar_pilot_v1/run_pilot/export $PES $role || ok=0; done
wait $UP
for who in final liquamen pilot; do aws s3 sync --only-show-errors --exclude 'tmp_*' $F/features_$who/ $S3R/features_$who/ || log "FEATURES_UPLOAD_FAILED $who"; done
[ $ok = 1 ] && { log "STAGE1_DONE"; touch $F/STAGE1_DONE; } || log "STAGE1_INCOMPLETE"
