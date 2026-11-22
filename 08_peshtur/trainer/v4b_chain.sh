#!/bin/bash
# v4b_chain.sh <ws_id> <tag> <arm_name>
# Isolated Qwen/Qwen3.5-4B 3-teacher distillation chain (ops/v4b).
# Zero writes outside s3://.../engineering/vision/v4b/ and local /data/local/{qv,v4b}/.
# Never touches Winoground (Public no-WG only). Never uses bare `wait`.
set -u
export AWS_DEFAULT_REGION=us-east-2 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

WS=$1; TAG=$2; NAME=$3
B=s3://YOUR_BUCKET/jev-race
V=$B/engineering/vision
A=$V/recipe_a
FG=$V/recipe_g
FF=$V/recipe_f
E=$V/recipe_e
WXS=$V/wx
V4B=$V/v4b
U=$V4B/arms/$NAME
C=$B/fleetq/claims

Q=/data/local/qv
UW=/data/local/v4b
L=$UW/logs/chain_$NAME.log
mkdir -p $UW/logs $UW/code $Q/logs $Q/code $Q/code_vl $Q/code_v4b/qv $Q/kit $Q/dev $Q/public $Q/data/img $Q/ocr $Q/models $Q/runs/$NAME $Q/rbh_gate $Q/priv

log() { echo "[$NAME $(date -u +%H:%M:%SZ)] $*" >> $L; }
mirror() { aws s3 cp --quiet $L $U/logs/chain_$NAME.$TAG.log 2>/dev/null || true; }

exec 9> $UW/chain.lock
flock -n 9 || { log "another chain holds the lock"; exit 0; }
log "START ws=$WS tag=$TAG host=$(hostname) gpus=$(nvidia-smi -L 2>/dev/null | wc -l)"

# 0. Check markers & GPUs
aws s3 ls $U/state/${NAME}_chain_done > /dev/null 2>&1 && { log "CHAIN_DONE already on S3"; exit 0; }
aws s3 ls $U/state/${NAME}_fail > /dev/null 2>&1 && { log "FAIL marker on S3: holding"; exit 5; }
[ "$(nvidia-smi -L 2>/dev/null | wc -l)" = "8" ] || { log "NOT_8_GPUS"; mirror; exit 6; }

NL=$(aws s3 cp --quiet $U/state/${NAME}_launches - 2>/dev/null | tr -dc 0-9 | cut -c1-3)
NL=$(( ${NL:-0} + 1 ))
echo "$NL" | aws s3 cp --quiet - $U/state/${NAME}_launches
log "launch_number=$NL"
[ "$NL" -le 12 ] || { echo "too many launches $(date -u +%FT%TZ)" | aws s3 cp --quiet - $U/state/${NAME}_fail; mirror; exit 5; }

# 1. Fetch our v4b code bundle from S3 and verify MD5SUMS
aws s3 cp --quiet --recursive $V4B/code/ $UW/code/ && (cd $UW/code && md5sum -c --quiet MD5SUMS) || { log "V4B_CODE_BAD"; mirror; exit 3; }
cp $UW/code/vx_train_vl.py $UW/code/wx_engine.py $UW/code/v4b_engine.py $UW/code/v4b_eval.py $UW/code/v4b_result.py $UW/code/qv_common.py $UW/code/qv_layout.py $UW/code/qv_ocr.py $Q/code_v4b/
cp $UW/code/qv_common.py $UW/code/qv_layout.py $UW/code/qv_ocr.py $Q/code_v4b/qv/

MKEY=qwen35_4b
MREPO=Qwen/Qwen3.5-4B
MREV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
MAN=wxs1_nc
MSHA=62ff4f7db91bfb72731927a4659a98d48b05b2f7bb10d1603f23e581d5bf5b7c
if [ "$NAME" = "v4b2" ]; then
  LORA_R=${LORA_R:-16}
  LORA_A=${LORA_A:-32}
  LR=${LR:-5e-5}
  HEAD_LR=${HEAD_LR:-3e-4}
  MERGER=${MERGER:-0}
  WD=${WD:-0.02}
fi
STEPS=${STEPS:-5051}
GSTEPS=${GSTEPS:-300,1000,2000,3000,4000,5051}
SAVE=${SAVE:-200}
SEED=${SEED:-20261119}
LORA_R=${LORA_R:-64}
LORA_A=${LORA_A:-128}
LR=${LR:-2e-4}
HEAD_LR=${HEAD_LR:-1e-3}
MERGER=${MERGER:-0.5}
WD=${WD:-0.01}
WARMUP=${WARMUP:-100}
FLOOR=${FLOOR:-0.05}
STOPB=${STOPB:-60}
FBAR=${FBAR:-75}
PUBSTEP=${PUBSTEP:-2500}

# 2. Restage env, kit, dev, public suite, and Qwen3.5-4B in parallel (explicit PIDs, no bare wait)
if [ ! -f $Q/V4B_RESTAGE_OK ]; then
  aws s3 sync --quiet $A/code/qv/ $Q/code/ && (cd $Q/code && md5sum -c --quiet MD5SUMS) || { log "CODE_QV_BAD"; mirror; exit 3; }
  aws s3 cp --quiet --recursive $V/recipe_vl/code2/ $Q/code_vl/ && (cd $Q/code_vl && md5sum -c --quiet MD5SUMS) || { log "VL_CODE_BAD"; mirror; exit 3; }
  # Stub qwen38_27b so qv_env.sh does NOT download 54 GB Qwen3.8-27B
  if [ ! -f $Q/models/qwen38_27b/model.safetensors.index.json ]; then
    mkdir -p $Q/models/qwen38_27b
    echo '{"weight_map": {}, "note": "v4b stub: Qwen3.8-27B not needed"}' > $Q/models/qwen38_27b/model.safetensors.index.json
  fi

  pids=()
  ( bash $Q/code/qv_env.sh; log "ENV $(tail -2 $Q/logs/env.log 2>/dev/null | tr '\n' ' ' | cut -c1-250)" ) & pids+=($!)
  (
    [ -d $Q/kit/kit_vis ] || { aws s3 cp --quiet $A/code/kit_vis.tgz $Q/kit/ && tar xzf $Q/kit/kit_vis.tgz -C $Q/kit; }
    D=$V/recipe_dev
    if [ ! -f $Q/dev/dev_rows.jsonl ]; then
      aws s3 cp --quiet $D/dev_rows.jsonl $Q/dev/
      aws s3 cp --quiet $D/dev_rows.header.json $Q/dev/
      aws s3 cp --quiet $D/img_dev.tar $Q/dev/
      tar xf $Q/dev/img_dev.tar -C $Q/dev && rm -f $Q/dev/img_dev.tar
    fi
    if [ ! -f $Q/public/suite/vision-rows-noWG.jsonl ]; then
      aws s3 cp --quiet $V/suite/suite_noWG.tar $Q/public/
      aws s3 cp --quiet $V/suite/suite_noWG.tar.md5 $Q/public/
      [ "$(awk '{print $1}' $Q/public/suite_noWG.tar.md5)" = "$(md5sum < $Q/public/suite_noWG.tar | cut -d' ' -f1)" ] && tar xf $Q/public/suite_noWG.tar -C $Q/public && rm -f $Q/public/suite_noWG.tar
      [ -f $Q/public/suite/vision-rows-noWG.jsonl ] || aws s3 cp --quiet $V/dormice/gate/vision-rows-noWG.jsonl $Q/public/suite/vision-rows-noWG.jsonl
    fi
  ) & pids+=($!)
  (
    if [ ! -f $Q/models/$MKEY/MODEL_OK ]; then
      python3 -m pip install -q --user "huggingface_hub>=1.0" > /dev/null 2>&1 || true
      python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MREPO', revision='$MREV', local_dir='$Q/models/$MKEY', token=False, max_workers=16)" > $UW/logs/model.log 2>&1
      python3 - $Q/models/$MKEY >> $UW/logs/model.log 2>&1 <<'PY' && touch $Q/models/$MKEY/MODEL_OK
import json, os, struct, sys
d = sys.argv[1]
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
by = {}
for n, f in idx.items():
    by.setdefault(f, set()).add(n)
for f, names in by.items():
    p = os.path.join(d, f)
    with open(p, "rb") as fh:
        hl = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(hl))
    end = max(v["data_offsets"][1] for k, v in h.items() if k != "__metadata__")
    assert os.path.getsize(p) == 8 + hl + end, f"{f}: size mismatch"
    assert names <= set(h), f"{f}: missing tensors"
print("MODEL_OK", len(by), "shards", len(idx), "tensors")
PY
    fi
  ) & pids+=($!)
  for pid in "${pids[@]}"; do wait "$pid" || true; done

  M_OK=$([ -f $Q/models/$MKEY/MODEL_OK ] && echo 1 || echo 0)
  D_OK=$([ "$(sha256sum $Q/dev/dev_rows.jsonl | cut -d' ' -f1)" = "082d1eceec1f34ce122871f133f07e1200807c3ad19d5a8fe9acd3f30a75a72d" ] && echo 1 || echo 0)
  P_OK=$([ "$(sha256sum $Q/public/suite/vision-rows-noWG.jsonl | cut -d' ' -f1)" = "0d44a006e575363aee592dd7a03a5ae45cf095c45eef6ce74079e8a040ef5d3c" ] && echo 1 || echo 0)
  E_OK=$([ -f $Q/venv.functional.ok ] && echo 1 || echo 0)
  log "RESTAGE check env=$E_OK model=$M_OK dev=$D_OK pub=$P_OK $(tail -1 $UW/logs/model.log 2>/dev/null)"
  if [ "$E_OK" = "1" ] && [ "$M_OK" = "1" ] && [ "$D_OK" = "1" ] && [ "$P_OK" = "1" ]; then
    touch $Q/V4B_RESTAGE_OK
  else
    log "RESTAGE_FAIL"; mirror; exit 4
  fi
fi

# 3. OCR caches (md5-checked) + HELD split
ocr_get() {
  [ -f "$2" ] && [ "$(md5sum < "$2" | cut -d' ' -f1)" = "$3" ] && return 0
  aws s3 cp --quiet "$1" "$2.dl" && [ "$(md5sum < "$2.dl" | cut -d' ' -f1)" = "$3" ] && mv "$2.dl" "$2" || { log "OCR_BAD $(basename "$2")"; return 1; }
}
ocr_get $A/ocr/ocr_dev.jsonl $Q/ocr/ocr_dev.jsonl 5b62c8a22424c56bae8f3cfb0375cbe9 \
  && ocr_get $A/ocr/ocr_public_EVAL_ONLY.jsonl $Q/ocr/ocr_public.jsonl 212232c925a9f94c9b4c6b0506d73db9 \
  && ocr_get $FG/ocr/ocr_train_ug1.jsonl $Q/ocr/ocr_train_ug1.jsonl 02603453ee741ea87ca6d6749e07f8a6 \
  && ocr_get $V/recipe_privlike/ocr/ocr_rbh.jsonl $Q/rbh_gate/ocr_rbh.jsonl ff5023125c676eed1f30524afa4c0e32 \
  && ocr_get $V/recipe_privlike/ocr/ocr_priv.jsonl $Q/priv/ocr_priv.jsonl 98776a8972ec6d019423092a9e73946b \
  && log "OCR_OK" || { log "OCR_FAIL"; mirror; exit 4; }

if [ ! -f $Q/priv/.ok ]; then
  PP=$V/recipe_privlike
  for f in priv_rows.jsonl MD5SUMS img_priv.tar; do aws s3 cp --quiet $PP/$f $Q/priv/; done
  for f in priv_rows.jsonl img_priv.tar; do
    [ "$(grep " $f\$" $Q/priv/MD5SUMS | tail -1 | cut -d' ' -f1)" = "$(md5sum < $Q/priv/$f | cut -d' ' -f1)" ] || { log "PRIV_MD5_BAD $f"; mirror; exit 4; }
  done
  tar xf $Q/priv/img_priv.tar -C $Q/priv && rm -f $Q/priv/img_priv.tar
  python3 - <<'PY'
import hashlib, json
F = "/data/local/qv/priv"
out = {0: open(f"{F}/priv_fit.jsonl", "w"), 1: open(f"{F}/priv_held.jsonl", "w")}
for l in open(f"{F}/priv_rows.jsonl"):
    if l.strip():
        out[int(hashlib.sha1(json.loads(l)["id"].encode()).hexdigest(), 16) % 2].write(l)
PY
  [ "$(sha256sum $Q/priv/priv_held.jsonl | cut -c1-16)" = "47d52acc7d9d0834" ] && touch $Q/priv/.ok && log "HELD_OK $(wc -l < $Q/priv/priv_held.jsonl)" || { log "PRIV_SPLIT_BAD"; mirror; exit 4; }
fi

# 4. Stage wxs1_nc manifest + ug1_nc image tars (md5-checked)
if [ ! -f $Q/data/$MAN.jsonl ]; then
  aws s3 cp --quiet $WXS/manifests/$MAN.jsonl $Q/data/$MAN.jsonl
  aws s3 cp --quiet $WXS/manifests/$MAN.header.json $Q/data/$MAN.header.json 2>/dev/null || true
fi
GOT_SHA=$(sha256sum $Q/data/$MAN.jsonl | cut -d' ' -f1)
[ "$GOT_SHA" = "$MSHA" ] || { log "MANIFEST_SHA_BAD $GOT_SHA"; mirror; exit 4; }

if [ ! -f $Q/data/img/.all_v4b_ok ]; then
  aws s3 cp --quiet $A/MD5SUMS $Q/data/MD5SUMS
  aws s3 cp --quiet $V/recipe_b/MD5SUMS $Q/data/MD5SUMS_b
  aws s3 cp --quiet $V/recipe_b3/MD5SUMS $Q/data/MD5SUMS_b3
  aws s3 cp --quiet $V/recipe_c/MD5SUMS $Q/data/MD5SUMS_c
  aws s3 cp --quiet $V/recipe_d/MD5SUMS $Q/data/MD5SUMS_d
  aws s3 cp --quiet $E/MD5SUMS $Q/data/MD5SUMS_e
  aws s3 cp --quiet $FF/MD5SUMS $Q/data/MD5SUMS_f
  aws s3 cp --quiet $FG/MD5SUMS $Q/data/MD5SUMS_g
  for fam in $(python3 -c "import json; print(' '.join(sorted({r['source'] for r in map(json.loads, open('$Q/data/$MAN.jsonl'))})))"); do
    [ -f $Q/data/img/.ok_$fam ] && continue
    PFX=""; M=""
    for pm in "$A MD5SUMS" "$V/recipe_b MD5SUMS_b" "$V/recipe_b3 MD5SUMS_b3" "$V/recipe_c MD5SUMS_c" "$V/recipe_d MD5SUMS_d" "$E MD5SUMS_e" "$FF MD5SUMS_f" "$FG MD5SUMS_g"; do
      set -- $pm
      grep -q " img_$fam.tar$" $Q/data/$2 && { PFX=$1; M=$Q/data/$2; break; }
    done
    [ -n "$PFX" ] || { log "IMG_NOT_IN_MD5SUMS $fam"; mirror; exit 5; }
    aws s3 cp --quiet $PFX/img/img_$fam.tar $Q/data/ || { log "IMG_FETCH_FAIL $fam"; mirror; exit 5; }
    want=$(grep " img_$fam.tar$" $M | cut -d' ' -f1)
    got=$(md5sum < $Q/data/img_$fam.tar | cut -d' ' -f1)
    [ "$want" = "$got" ] || { log "IMG_MD5_BAD $fam"; mirror; exit 6; }
    tar xf $Q/data/img_$fam.tar -C $Q/data/img && rm -f $Q/data/img_$fam.tar && touch $Q/data/img/.ok_$fam
  done
  MISSING=$(python3 -c "
import json, os
n = sum(not os.path.exists('$Q/data/img/' + p) for l in open('$Q/data/$MAN.jsonl') for p in json.loads(l)['images'])
print(n)")
  [ "$MISSING" = "0" ] || { log "IMAGES_MISSING $MISSING"; mirror; exit 10; }
  touch $Q/data/img/.all_v4b_ok
  log "IMAGES_OK missing=0"
fi
mirror

PY=$Q/venv/bin/python
TR=$Q/venv/bin/torchrun
cd $Q

# 5. Pre-smoke (once): 4 steps + S3 resume to step 6 + v4b_eval parity check on 65 dev rows
if ! aws s3 ls $U/state/${NAME}_smoke_ok > /dev/null 2>&1 && ! aws s3 ls $U/ckpt/ 2>/dev/null | grep -q ckpt_; then
  log "SMOKE_START"
  awk 'NR%46==1' $Q/dev/dev_rows.jsonl > $Q/logs/v4b_dev65.jsonl
  SN=_${NAME}_smoke
  rm -rf $Q/runs/$SN $Q/runs/${SN}_1
  $TR --nproc_per_node 8 --master_port 29661 $Q/code_v4b/vx_train_vl.py \
    --family qwen --model $Q/models/$MKEY --manifest $Q/data/$MAN.jsonl --expect-rows-sha256 $MSHA \
    --img-root $Q/data/img --out $Q/runs/$SN --s3 $U/smoke_ckpt \
    --steps 4 --rows-per-step 64 --lr $LR --head-lr $HEAD_LR --warmup 2 --floor $FLOOR --wd $WD --clip 1.0 \
    --lora-r $LORA_R --lora-alpha $LORA_A --merger-lr-mult $MERGER --label-smoothing 0.05 --max-tokens 8192 \
    --save-every 2 --seed $SEED --kit $Q/kit/kit_vis --layout d3 \
    --ocr-train $Q/ocr/ocr_train_ug1.jsonl --ocr-gate $Q/ocr/ocr_dev.jsonl \
    --gate-file $Q/logs/v4b_dev65.jsonl --gate-root $Q/dev/img_dev --gate-steps 4 --gate-continue 1000000 --gate-stop 0 \
    > $Q/logs/smoke1.out 2>&1
  RC1=$?
  sleep 10
  mv $Q/runs/$SN $Q/runs/${SN}_1
  mkdir -p $Q/runs/$SN/ckpt_00004
  aws s3 sync --quiet $U/smoke_ckpt/ckpt_00004/ $Q/runs/$SN/ckpt_00004/
  $TR --nproc_per_node 8 --master_port 29662 $Q/code_v4b/vx_train_vl.py \
    --family qwen --model $Q/models/$MKEY --manifest $Q/data/$MAN.jsonl --expect-rows-sha256 $MSHA \
    --img-root $Q/data/img --out $Q/runs/$SN --s3 $U/smoke_ckpt \
    --steps 6 --rows-per-step 64 --lr $LR --head-lr $HEAD_LR --warmup 2 --floor $FLOOR --wd $WD --clip 1.0 \
    --lora-r $LORA_R --lora-alpha $LORA_A --merger-lr-mult $MERGER --label-smoothing 0.05 --max-tokens 8192 \
    --save-every 2 --seed $SEED --kit $Q/kit/kit_vis --layout d3 \
    --ocr-train $Q/ocr/ocr_train_ug1.jsonl --ocr-gate $Q/ocr/ocr_dev.jsonl \
    --gate-file $Q/logs/v4b_dev65.jsonl --gate-root $Q/dev/img_dev --gate-steps 6 --gate-continue 1000000 --gate-stop 0 \
    --resume $Q/runs/$SN/ckpt_00004 > $Q/logs/smoke2.out 2>&1
  RC2=$?
  # Engine parity on 65 dev rows using ckpt_00006
  mkdir -p $Q/runs/$SN/par
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$Q/kit/kit_vis:$Q/code_v4b $PY $Q/code_v4b/v4b_eval.py smoke $Q/runs/$SN/ckpt_00006 1 $Q/logs/v4b_dev65.jsonl $Q/dev/img_dev $Q/runs/$SN/par/shard_0 $Q/models/$MKEY $Q/ocr/ocr_dev.jsonl > $Q/runs/$SN/par/eval.log 2>&1
  RC3=$?
  PYTHONPATH=$Q/kit/kit_vis $PY $Q/code_vl/vl_dev_parity.py $Q/logs/v4b_dev65.jsonl $Q/runs/$SN/par $Q/runs/$SN/gate_00006.json > $Q/runs/$SN/par/parity.json 2>&1
  PAR_OK=$(python3 -c "
import json
d = json.load(open('$Q/runs/$SN/par/parity.json'))
dp = d.get('max_abs_prob_diff') or {}
ok = bool(d.get('ok')) or (d.get('rows', 0) >= 60 and d.get('same_answer_frac', 0) >= 0.95 and dp.get('max', 1) <= 0.05 and dp.get('median', 1) <= 0.01)
print(int(ok))" 2>/dev/null || echo 0)
  HAS_MG=$([ "$MERGER" = "0" ] || [ "$MERGER" = "0.0" ] || [ -f $Q/runs/$SN/ckpt_00006/merger.safetensors ] && echo 1 || echo 0)
  log "SMOKE rc1=$RC1 rc2=$RC2 rc3=$RC3 merger_ok=$HAS_MG (mult=$MERGER) par_ok=$PAR_OK par=$(cat $Q/runs/$SN/par/parity.json 2>/dev/null | tr '\n' ' ' | cut -c1-400)"
  if [ "$RC1" = "0" ] && [ "$RC2" = "0" ] && [ "$RC3" = "0" ] && [ "$HAS_MG" = "1" ] && [ "$PAR_OK" = "1" ]; then
    echo "ok $(date -u +%FT%TZ)" | aws s3 cp --quiet - $U/state/${NAME}_smoke_ok
    log "SMOKE_OK"
  else
    log "SMOKE_FAIL tail1=$(tail -n 5 $Q/logs/smoke1.out 2>/dev/null | tr '\n' '|') tail2=$(tail -n 5 $Q/logs/smoke2.out 2>/dev/null | tr '\n' '|') tail3=$(tail -n 5 $Q/runs/$SN/par/eval.log 2>/dev/null | tr '\n' '|')"
    echo "smoke failed $(date -u +%FT%TZ)" | aws s3 cp --quiet - $U/state/${NAME}_fail
    mirror; exit 7
  fi
  rm -rf $Q/runs/$SN $Q/runs/${SN}_1
  mirror
fi

# 6. Pull latest checkpoint from S3 if resuming on a fresh node
RESUME=""
if ! ls -d $Q/runs/$NAME/ckpt_* >/dev/null 2>&1; then
  SL=$(aws s3 ls $U/ckpt/ 2>/dev/null | awk '{print $2}' | grep ckpt_ | sort | tail -1 | tr -d /)
  if [ -n "$SL" ]; then
    aws s3 sync --quiet $U/ckpt/$SL/ $Q/runs/$NAME/$SL/ && log "PULLED_S3_CKPT $SL"
    [ -f $Q/runs/$NAME/$SL/train_log.jsonl ] && cp $Q/runs/$NAME/$SL/train_log.jsonl $Q/runs/$NAME/train_log.jsonl
    aws s3 cp --quiet $U/train_log.jsonl $Q/runs/$NAME/train_log.s3.jsonl 2>/dev/null && \
      [ "$(wc -l < $Q/runs/$NAME/train_log.s3.jsonl)" -ge "$(wc -l < $Q/runs/$NAME/train_log.jsonl 2>/dev/null || echo 0)" ] && \
      cp $Q/runs/$NAME/train_log.s3.jsonl $Q/runs/$NAME/train_log.jsonl
    aws s3 sync --quiet $U/gates/ $Q/runs/$NAME/ --exclude "*" --include "gate_*.json" 2>/dev/null || true
  fi
fi
LAST=""
for d in $(ls -d $Q/runs/$NAME/ckpt_* 2>/dev/null | sort -r); do
  if [ -f "$d/state.json" ] && [ -f "$d/optim.pt" ] && [ -f "$d/adapter.safetensors" ] && [ -f "$d/head.pt" ] && ( [ "$MERGER" = "0" ] || [ "$MERGER" = "0.0" ] || [ -f "$d/merger.safetensors" ] ); then
    LAST="$d"; break
  fi
done
[ -n "$LAST" ] && RESUME="--resume $LAST" && log "RESUME_FROM $LAST"

# 7. Train v4b1 (with background S3 log mirror killed immediately when training finishes!)
FINAL=$(printf "ckpt_%05d" $STEPS)
DSHA=$(python3 -c "import json; print(json.load(open('$Q/dev/dev_rows.header.json'))['rows_sha256'])")
if [ ! -f $Q/runs/$NAME/$FINAL/state.json ] || ! grep -q "\"event\": \"gate\", \"step\": $STEPS," $Q/runs/$NAME/train_log.jsonl 2>/dev/null; then
  log "TRAIN_START steps=$STEPS r=$LORA_R a=$LORA_A lr=$LR head_lr=$HEAD_LR merger=$MERGER pubstep=$PUBSTEP"
  ( exec 9>&-; while sleep 300; do mirror; aws s3 cp --quiet $Q/runs/$NAME/train_log.jsonl $U/train_log.jsonl 2>/dev/null || true; done ) & MIR=$!
  $TR --nproc_per_node 8 --master_port 29671 $Q/code_v4b/vx_train_vl.py \
    --family qwen --model $Q/models/$MKEY --manifest $Q/data/$MAN.jsonl --expect-rows-sha256 $MSHA \
    --img-root $Q/data/img --out $Q/runs/$NAME --s3 $U/ckpt \
    --steps $STEPS --rows-per-step 64 --lr $LR --head-lr $HEAD_LR --warmup $WARMUP --floor $FLOOR --wd $WD --clip 1.0 \
    --lora-r $LORA_R --lora-alpha $LORA_A --merger-lr-mult $MERGER --label-smoothing 0.05 --max-tokens 8192 \
    --save-every $SAVE --seed $SEED --kit $Q/kit/kit_vis --layout d3 \
    --ocr-train $Q/ocr/ocr_train_ug1.jsonl --ocr-gate $Q/ocr/ocr_dev.jsonl --ocr-public $Q/ocr/ocr_public.jsonl \
    --gate-file $Q/dev/dev_rows.jsonl --gate-sha256 $DSHA --gate-root $Q/dev/img_dev \
    --gate-baseline --gate-steps $GSTEPS --gate-continue 1000000 --gate-continue-rel 1000000 --gate-stop $STOPB --gate-stop-rel -20.0 \
    --public-file $Q/public/suite/vision-rows-noWG.jsonl --public-root $Q/public/suite --public-step $PUBSTEP \
    $RESUME >> $Q/logs/train_$NAME.out 2>&1
  TRC=$?
  kill $MIR 2>/dev/null || true
  wait $MIR 2>/dev/null || true
  log "TRAIN_END rc=$TRC"
  aws s3 cp --quiet $Q/runs/$NAME/train_log.jsonl $U/train_log.jsonl 2>/dev/null || true
  aws s3 sync --quiet $Q/runs/$NAME $U/gates/ --exclude "*" --include "gate_*.json" --include "public_*.json" 2>/dev/null || true
  mirror
  [ -f $Q/runs/$NAME/$FINAL/state.json ] || { log "NO_FINAL_CKPT rc=$TRC (will resume)"; exit 20; }
fi

# Helper: run 8-GPU sharded V4BEngine eval with explicit PIDs (NO bare wait!)
run_sharded() {
  local lbl=$1 rows_file=$2 img_root=$3 out_dir=$4 ocr_file=$5
  mkdir -p $out_dir/rows
  python3 -c "
rows = [l for l in open('$rows_file') if l.strip()]
for k in range(8):
    open('$out_dir/rows/rows_%d.jsonl' % k, 'w').writelines(rows[k::8])"
  local pids=()
  for k in 0 1 2 3 4 5 6 7; do
    ( CUDA_VISIBLE_DEVICES=$k PYTHONPATH=$Q/kit/kit_vis:$Q/code_v4b $PY $Q/code_v4b/v4b_eval.py \
        "$lbl" "$Q/runs/$NAME/$FINAL" 1 "$out_dir/rows/rows_$k.jsonl" "$img_root" "$out_dir/shard_$k" "$Q/models/$MKEY" "$ocr_file" \
        > "$out_dir/shard_$k.log" 2>&1 ) &
    pids+=($!)
  done
  local fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail+1)); done
  log "SHARDED $lbl fail=$fail"
  return $fail
}

# 8. Dev Parity (2,967 rows)
DP=$Q/runs/$NAME/dev_parity
if [ ! -f $DP/parity.json ]; then
  run_sharded "${NAME}_devpar" "$Q/dev/dev_rows.jsonl" "$Q/dev/img_dev" "$DP" "$Q/ocr/ocr_dev.jsonl"
  PYTHONPATH=$Q/kit/kit_vis $PY $Q/code_vl/vl_dev_parity.py $Q/dev/dev_rows.jsonl $DP $Q/runs/$NAME/gate_$(printf %05d $STEPS).json > $DP/parity.json 2> $DP/parity.err
fi
DPOK=$(python3 -c "
import json
d = json.load(open('$DP/parity.json'))
dp = d.get('max_abs_prob_diff') or {}
ok = bool(d.get('ok')) or (d.get('rows', 0) >= 2900 and d.get('same_answer_frac', 0) >= 0.985 and dp.get('median', 1) <= 0.01 and dp.get('p95', 1) <= 0.03)
print(int(ok))" 2>/dev/null || echo 0)
log "DEV_PARITY ok=$DPOK $(cat $DP/parity.json 2>/dev/null | tr '\n' ' ' | cut -c1-400)"
if [ "$DPOK" != "1" ]; then
  echo "dev parity failed $(date -u +%FT%TZ)" | aws s3 cp --quiet - $U/state/${NAME}_fail
  mirror; exit 21
fi

# 9. RBH diagnostic (reported only)
RB=$Q/rbh_gate; RO=$Q/runs/$NAME/rbh_read
for f in rbh_rows.jsonl img_rbh.tar MD5SUMS; do [ -f $RB/$f ] || aws s3 cp --quiet $V/recipe_privlike/rbh/$f $RB/; done
[ -f $RB/rbh_gate.py ] || aws s3 cp --quiet $V/recipe_privlike/ue3/code/rbh_gate.py $RB/rbh_gate.py
[ -d $RB/img_rbh ] || tar xf $RB/img_rbh.tar -C $RB
if [ ! -f $RO/rbh_gate.json ]; then
  run_sharded "${NAME}_rbh" "$RB/rbh_rows.jsonl" "$RB/img_rbh" "$RO" "$RB/ocr_rbh.jsonl"
  python3 $RB/rbh_gate.py $RB/rbh_rows.jsonl $RO > $RO/rbh_gate.json 2> $RO/rbh_gate.err || true
fi
log "RBH $(cat $RO/rbh_gate.json 2>/dev/null | tr '\n' ' ' | cut -c1-300)"

# 10. HELD read (1,621 private-like rows)
if [ ! -f $Q/runs/$NAME/held.json ]; then
  run_sharded "${NAME}_held" "$Q/priv/priv_held.jsonl" "$Q/priv/img_priv" "$Q/runs/$NAME/held_read" "$Q/priv/ocr_priv.jsonl"
  python3 - $Q/priv/priv_held.jsonl $Q/runs/$NAME/held_read > $Q/runs/$NAME/held.json <<'PY'
import collections, glob, json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
rid = lambda r: (r.get("_evaluation") or {}).get("run_id") or r.get("id")
cnt = collections.Counter(rid(r) for r in rows)
rows = [r for r in rows if cnt[rid(r)] == 1]
p = {}
for f in sorted(glob.glob(sys.argv[2] + "/shard_*/results.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if r.get("status") == "ok":
            p[r["run_id"]] = r["response"]["answers"]["answer"]["choice"]
fam = collections.defaultdict(lambda: [0, 0, 0.0])
for r in rows:
    f = fam[r.get("benchmark") or r.get("family") or "all"]
    f[0] += 1; f[1] += (p.get(rid(r)) == r["expected"]["answer"]); f[2] += 1.0 / len(r["questions"]["answer"]["criteria"])
sk = {b: round(100 * max(0.0, (c / n - ch / n) / (1 - ch / n)), 2) for b, (n, c, ch) in fam.items()}
print(json.dumps({"arm": {"rows": len(rows), "read": sum(rid(r) in p for r in rows),
                          "acc": round(100 * sum(c for _, c, _ in fam.values()) / len(rows), 2),
                          "index_like": round(sum(sk.values()) / len(sk), 2), "skill": sk}}))
PY
fi
log "HELD $(cat $Q/runs/$NAME/held.json | tr '\n' ' ' | cut -c1-500)"

# 11. ONE Public no-WG read (11,410 rows; NO Winoground!)
PUBO=$Q/runs/$NAME/public_read
if [ ! -f $Q/runs/$NAME/${NAME}_result.json ]; then
  run_sharded "${NAME}_pub" "$Q/public/suite/vision-rows-noWG.jsonl" "$Q/public/suite" "$PUBO" "$Q/ocr/ocr_public.jsonl"
  PYTHONPATH=$Q/kit/kit_vis $PY $Q/code_v4b/v4b_result.py $NAME > $UW/logs/result.log 2>&1
fi
log "RESULT $(cat $Q/runs/$NAME/${NAME}_result.json 2>/dev/null | tr '\n' ' ' | cut -c1-1200)"

# 12. Upload final artifacts & mark CHAIN_DONE
aws s3 cp --quiet $Q/runs/$NAME/${NAME}_result.json $U/${NAME}_result.json
aws s3 cp --quiet $Q/runs/$NAME/train_log.jsonl $U/train_log.jsonl
aws s3 sync --quiet $Q/runs/$NAME/public_read $U/public_read/ --exclude "rows/*"
aws s3 sync --quiet $Q/runs/$NAME/held_read $U/held_read/ --exclude "rows/*"
aws s3 cp --quiet $Q/runs/$NAME/held.json $U/held.json
aws s3 cp --quiet $DP/parity.json $U/dev_parity.json
aws s3 sync --quiet $Q/runs/$NAME/$FINAL/ $U/ckpt/$FINAL/
echo "done $(date -u +%FT%TZ)" | aws s3 cp --quiet - $U/state/${NAME}_chain_done
log "CHAIN_DONE"
mirror
