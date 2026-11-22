#!/bin/bash
# FS5 (TB1 fork, 2026-11-12, coordinator order 15:3xZ) = stage1_fs1.sh (029210fe) with --seed 5 (the trainer's seed also draws the data order:
# rng.permutation over the manifest rows) and its own run dir / S3 prefix esbar_fs5. Same stock init, same manifest 8461365d (read from
# esbar_fs1/manifests), same recipe, legs and save rules. Nothing else changed.
# FS1 = retrain-from-stock pilot (fork E, PREREG FS1 in AGENT_NOTES 2026-11-10/11). init = STOCK gemma-4-31B-it @842da379 (NOT E1);
# data = v8 manifest + breadth_v1 clean/clean_src + capped pools_v4; ONE cosine schedule of 6011 steps in two legs:
# leg 1 --max-steps 2400 -> export_s2400, leg 2 resumes -> 6011 -> export_full. stage1_v8.sh with the init, manifest and lr
# changed; idempotent: on a fresh worker resume_fs1.sh restages, then this script resumes from the newest S3 mirror (--resume-s3).
unset ASCALE_RUN_ID
export OMP_NUM_THREADS=1 PATH=/data/local/venv_t2141/bin:$PATH AWS_DEFAULT_REGION=us-east-2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false TORCH_NCCL_AVOID_RECORD_STREAMS=1 NCCL_RUNTIME_CONNECT=0
C=/data/local/esbar_code_v2e; export PYTHONPATH=$C/trainers:$C
D=/data/local; F=$D/fs5; RUN=$F/run_fs5; INIT=$F/stock; MAN=$D/esbar_v5/manifests
LEG1=2400; STOCK_ID=4d59c3ca50908a6702b6db2b4af6fac9b5d5a37348435b26968085b4e0b2f04e
S3=s3://YOUR_BUCKET/jev-race; S3R=$S3/engineering/esbar/runs/esbar_fs5; S3M=$S3/engineering/esbar/runs/esbar_fs5/manifests
mkdir -p $RUN $F/logs $F/manifests $INIT
log() { echo "[stage1_fs5 $(date -u +%FT%TZ)] $*" | tee -a $F/logs/stage1.log; }
die() { log "STAGE1_FS5_FAILED $*"; exit 1; }
python3 -c "import torch, accelerate, sys; sys.exit(0 if torch.__version__ == '2.14.1+cu130' and torch.cuda.is_available() and torch.cuda.device_count() == 8 else 1)" || die ENV
[ "$(md5sum < $C/trainers/fullft_train.py | cut -c1-8)" = 4048f97c ] || die CODE_MD5
# ---- init identity (stock)
if [ ! -f $INIT/.ok ]; then
  aws s3 sync --only-show-errors $S3/models/gemma-4-31B-it_842da379/ $INIT/ || die SYNC_STOCK
  ID=$( (cd $INIT && ls model*.safetensors config.json | sort | xargs -P 3 -n 1 sha256sum | sort -k2) | sha256sum | cut -c1-64)
  [ "$ID" = "$STOCK_ID" ] || die "INIT_IDENTITY $ID"
  touch $INIT/.ok
fi
ls $INIT | grep -q "vbll_head\|adapter_" && die INIT_HAS_HEAD
# ---- manifest (from S3, sha-checked three ways)
# FS5: the data agent's manifest, discovered: exactly one *.jsonl.header.json in esbar_fs5/manifests/ (its .jsonl must hash to rows_sha256)
HN=$(aws s3 ls $S3M/ | awk '{print $4}' | grep '[.]jsonl[.]header[.]json$'); [ "$(echo "$HN" | grep -c .)" = 1 ] || die "MANIFEST_DISCOVERY: $(echo $HN)"
MN=${HN%.header.json}; PM=$F/manifests/$MN
for x in $MN $HN; do
  [ -s $F/manifests/$x ] || aws s3 cp --only-show-errors $S3M/$x $F/manifests/$x || die "FETCH $x"
done
HS=$(python3 -c "import json; print(json.load(open('$PM.header.json'))['rows_sha256'])")
FS=$(sha256sum < $PM | cut -c1-64)
[ -n "$HS" ] && [ "$HS" = "$FS" ] || { mv $PM $PM.bad_$(date +%s); mv $PM.header.json $PM.header.json.bad_$(date +%s); die "MANIFEST_SHA header=$HS file=$FS (incomplete or changed upload; local copies set aside)"; }; PMSHA_EXPECT=$HS
STEPS=$(python3 -c "import json, math; print(int(math.ceil(json.load(open('$PM.header.json'))['rows'] / 64)))")
[ "$STEPS" -gt "$LEG1" ] || die "STEPS $STEPS"
[ -s $MAN/dev.jsonl ] || die NO_DEV
log "ok: env, init stock $STOCK_ID, manifest $FS, steps $STEPS, leg1 $LEG1"
TR="python -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --rdzv-backend c10d --local-addr 127.0.0.1"
train() {
  $TR --rdzv-endpoint 127.0.0.1:29791 $C/trainers/fullft_train.py \
    --init-from $INIT --attention-policy B \
    --manifest $PM --expect-rows-sha256 $PMSHA_EXPECT \
    --val $MAN/dev.jsonl --val-max 2000 --eval-every 250 --eval-at-start --save-every 100 --keep-resume 1 \
    --resume-s3 $S3R/resume --keep-resume-s3 2 --resume-kind full --spot-watch on --force-save-utc 07:35 \
    --decisions-per-step 64 --epochs 1.0 --max-steps $1 \
    --lr 2.5e-6 --lr-floor 0.1 --warmup 100 --weight-decay 0.0 --clip 1.0 \
    --alpha 0.15 --ineligible-gold-weight one --head on --head-prior 1 --head-kl 1 --head-cov full \
    --optim adamw-fused --param-dtype bf16 --reduce-dtype fp32 \
    --pack-tokens 8192 --ac 1 --offload-ac 0 --offload-min 8192 --evict-states-min 6145 \
    --mlp-chunk 4096 --mlp-chunk-min 16384 --seed 5 \
    --pinned-code $C/trainers --out $RUN >> $F/logs/train.log 2>&1
}
upload_export() {   # $1 local dir, $2 s3 key name; sync then md5 read-back of every file
  local src=$1 dst=$S3R/$2 n bad
  aws s3 sync --only-show-errors $src/ $dst/ || return 1
  (cd $src && ls) | xargs -P 6 -I{} bash -c 'a=$(md5sum < "$0/{}" | cut -c1-32); b=$(aws s3 cp --only-show-errors "$1/{}" - | md5sum | cut -c1-32); [ "$a" = "$b" ] && echo "OK {} $a" || echo "BAD {} $a $b"' $src $dst > $F/logs/readback_$2.txt
  n=$(grep -c '^OK ' $F/logs/readback_$2.txt); bad=$(grep -c '^BAD ' $F/logs/readback_$2.txt)
  [ "$bad" = 0 ] && [ "$n" = "$(ls $src | wc -l)" ] || return 1
  aws s3 cp --only-show-errors $F/logs/readback_$2.txt $dst/READBACK_MD5.txt
  log "EXPORT_UPLOADED $2 files=$n (md5 read-back ok)"
}
push_logs() { aws s3 cp --only-show-errors $F/logs/train.log $S3R/train/train.log 2>/dev/null; aws s3 cp --only-show-errors $F/logs/stage1.log $S3R/train/stage1.log 2>/dev/null; for e in $RUN/eval_step*.jsonl; do [ -s $e ] && aws s3 cp --only-show-errors $e $S3R/train/evals/$(basename $e) 2>/dev/null; done; true; }
( while sleep 900; do push_logs; done ) & PUSHER=$!
trap 'kill $PUSHER 2>/dev/null' EXIT
# ---- leg 1 -> export_s$LEG1
H=export_s$LEG1
if ! aws s3 ls $S3R/$H/READBACK_MD5.txt > /dev/null 2>&1; then
  if [ ! -s $RUN/$H/model.safetensors.index.json ]; then
    if [ ! -s $RUN/export/model.safetensors.index.json ]; then
      log "LEG1 start: max-steps $LEG1 of schedule $STEPS"; t0=$(date +%s); train $LEG1; rc=$?
      log "LEG1 rc=$rc elapsed=$(( ($(date +%s) - t0) / 60 ))min"
      [ $rc = 0 ] && [ -s $RUN/export/model.safetensors.index.json ] || die "LEG1 rc=$rc"
    fi
    mv $RUN/export $RUN/$H || die MV_LEG1
  fi
  upload_export $RUN/$H $H || die UPLOAD_LEG1
  push_logs
fi
# ---- leg 2 -> export_full
if ! aws s3 ls $S3R/export_full/READBACK_MD5.txt > /dev/null 2>&1; then
  if [ ! -s $RUN/export/model.safetensors.index.json ]; then
    log "LEG2 start: resume -> $STEPS"; t0=$(date +%s); train $STEPS; rc=$?
    log "LEG2 rc=$rc elapsed=$(( ($(date +%s) - t0) / 60 ))min"
    [ $rc = 0 ] && [ -s $RUN/export/model.safetensors.index.json ] || die "LEG2 rc=$rc"
  fi
  upload_export $RUN/export export_full || die UPLOAD_FULL
fi
log "STAGE1_FS5_DONE"
push_logs
