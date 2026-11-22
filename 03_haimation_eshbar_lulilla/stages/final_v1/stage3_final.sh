#!/bin/bash
# ESBAR-FINAL stage 3 (coordinator add-on, after stage 2; does not touch the stage 1 / 2 pipeline): features on the fresh-family eval
# manifest (role eval_fresh) for final v1 and liquamen with the same extract() path and stamps; then bayes_ae.py (readouts A-D) on the
# final run's sdv1_head refit dir. liquamen is skipped for bayes_ae: it has no sdv1_head refit dir / cal features (its served head is an
# l01post_di export), and refitting its head is out of scope.
unset ASCALE_RUN_ID
export OMP_NUM_THREADS=1 PATH=/data/local/venv_t2141/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
C=/data/local/esbar_code_v2
export PYTHONPATH=$C/trainers:$C
F=/data/local/esbar_final_v1b
RUN=$F/run_final
INIT=$F/init
MAN=$F/manifests
S3R=s3://YOUR_BUCKET/jev-race/engineering/esbar/runs/stage1_final_v1
EM=/data/local/esbar_data/out/esbar_eval_fresh_v1.jsonl
EMSHA=ab48cba0baca10c4f861066e37c6f081eaac0a8a92ab1535f3a94f32f5281cc6
log() { echo "[stage3 $(date -u +%FT%TZ)] $*" | tee -a $F/logs/stage3.log; }
TR="python -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --rdzv-backend c10d --local-addr 127.0.0.1"
until [ -f $F/STAGE2_DONE ]; do sleep 30; done
log "start"
cp $EM $MAN/eval_fresh.jsonl && cp $EM.header.json $MAN/eval_fresh.jsonl.header.json || { log "COPY_FAILED"; exit 1; }
got=$(sha256sum < $MAN/eval_fresh.jsonl | cut -c1-64)
hdr=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['rows_sha256'])" $MAN/eval_fresh.jsonl.header.json)
log "eval_fresh rows=$(wc -l < $MAN/eval_fresh.jsonl) sha256=$got header=$hdr expected=$EMSHA md5=$(md5sum < $MAN/eval_fresh.jsonl | cut -c1-32)"
[ "$got" = "$EMSHA" ] && [ "$hdr" = "$EMSHA" ] || { log "EVAL_FRESH_SHA_MISMATCH"; exit 1; }

EXTRACTOR="$(md5sum $C/trainers/fullft_train.py | cut -c1-32)+$(md5sum $C/trainers/fullft_mem.py | cut -c1-32)"
port=29950
extract() {  # extract WHO EXPORT_DIR EXPORT_SHA ROLE  (= stage1_final.sh's extract)
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
  python3 $C/trainers/sdv1_head.py stamp-features --features $Fe --meta $FM --manifest $man --expect "$EXP" >> $F/logs/stage3.log 2>&1 \
    || { log "STAMP_FAILED $who $role"; return 1; }
}
ES=$(sha256sum < $F/export_sha256.txt | cut -c1-64)
LS=69aaa53af0a3a44e9ef9aab8860c3d2e523ca93e7ae893ac80d5463f809d5ef2
ok=1
extract final $RUN/export $ES eval_fresh || ok=0
extract liquamen $INIT $LS eval_fresh || ok=0
for who in final liquamen; do
  for x in eval_fresh.pt eval_fresh.stamp.json eval_fresh.expect.json; do
    [ -f $F/features_$who/$x ] && aws s3 cp --only-show-errors $F/features_$who/$x $S3R/features_$who/$x; done; done
aws s3 cp --only-show-errors $F/logs/features_final_eval_fresh.log $S3R/report/features_final_eval_fresh.log
aws s3 cp --only-show-errors $F/logs/features_liquamen_eval_fresh.log $S3R/report/features_liquamen_eval_fresh.log

FF=$F/features_final
if [ -s $FF/eval_fresh.pt ]; then
  t0=$(date +%s)
  (cd /tmp && OMP_NUM_THREADS=48 python3 /data/local/esbar_data/evalr/bayes_ae.py --code $C/trainers --heads $F/head_final --cal $FF/cal.pt \
     --eval fresh=$FF/eval_fresh.pt dev=$FF/dev.pt w2c=$FF/when2call_4opt_clean_dev.pt --out $F/bayes_ae_final.json) > $F/logs/bayes_ae_final.log 2>&1
  rc=$?
  log "BAYES_AE final rc=$rc wall_s=$(( $(date +%s) - t0 )) (bayes_ae.py md5 $(md5sum < /data/local/esbar_data/evalr/bayes_ae.py | cut -c1-32))"
  [ $rc -eq 0 ] || ok=0
else
  log "BAYES_AE final skipped: no eval_fresh features"; ok=0
fi
log "BAYES_AE liquamen skipped: no sdv1_head refit dir (vbll / det heads) or cal features for liquamen; its served head is an l01post_di export, so a refit would be needed"
for f in $F/bayes_ae_final.json $F/logs/bayes_ae_final.log $F/logs/stage3.log; do [ -f $f ] && aws s3 cp --only-show-errors $f $S3R/report/$(basename $f); done
[ $ok = 1 ] && { log "STAGE3_DONE"; touch $F/STAGE3_DONE; } || { log "STAGE3_INCOMPLETE"; touch $F/STAGE3_DONE; }
