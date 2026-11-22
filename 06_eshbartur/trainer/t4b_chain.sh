#!/bin/bash
# t4b_chain.sh <ws_id> <tag> <arm_name>
# Isolated Qwen/Qwen3.5-4B (4.21B causal text) karpo-31B distillation chain (ops/t4b).
# Zero writes outside s3://.../engineering/text/t4b/ and local /data/local/t4b/.
# Never uses bare `wait`. Evaluates on BOTH 4,400-row Text screen AND unseen private-like proxies (gde1v3s + dev).
set -u
unset ASCALE_RUN_ID HF_TOKEN HUGGING_FACE_HUB_TOKEN
export AWS_DEFAULT_REGION=us-east-2 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WS=$1; TAG=$2; NAME=$3
B=s3://YOUR_BUCKET/jev-race
E=$B/engineering/esbar
ER=$E/runs
TZ=$B/engineering/text/tz
T4B=$B/engineering/text/t4b
U=$T4B/arms/$NAME

W=/data/local/t4b
V=$W/venv_t4b
C=$W/code
R=$W/kroot
M=$W/models/qwen35_4b
DATA=$W/data
RUN=$W/runs/$NAME
EVAL=$W/eval/$NAME
L=$W/logs/chain_$NAME.log
mkdir -p $W/logs $C $R/src $M $DATA/src $RUN $EVAL/screen $EVAL/unseen

log() { echo "[$NAME $(date -u +%H:%M:%SZ)] $*" >> $L; }
mirror() { aws s3 cp --quiet $L $U/logs/chain_$NAME.$TAG.log 2>/dev/null || true; }

exec 9> $W/chain.lock
flock -n 9 || { log "another chain holds the lock"; exit 0; }
log "START ws=$WS tag=$TAG host=$(hostname) gpus=$(nvidia-smi -L 2>/dev/null | wc -l)"

# 0. Check S3 markers & GPUs
aws s3 ls $U/state/${NAME}_chain_done > /dev/null 2>&1 && { log "CHAIN_DONE already on S3"; exit 0; }
aws s3 ls $U/state/${NAME}_fail > /dev/null 2>&1 && { log "FAIL marker on S3: holding"; exit 5; }
[ "$(nvidia-smi -L 2>/dev/null | wc -l)" = "8" ] || { log "NOT_8_GPUS"; mirror; exit 6; }

NL=$(aws s3 cp --quiet $U/state/${NAME}_launches - 2>/dev/null | tr -dc 0-9 | cut -c1-3)
NL=$(( ${NL:-0} + 1 ))
echo "$NL" | aws s3 cp --quiet - $U/state/${NAME}_launches
log "launch_number=$NL"
[ "$NL" -le 12 ] || { echo "too many launches $(date -u +%FT%TZ)" | aws s3 cp --quiet - $U/state/${NAME}_fail; mirror; exit 5; }

# 1. Fetch our t4b code bundle from S3 and verify MD5SUMS
aws s3 cp --quiet --recursive $T4B/code/ $C/ && (cd $C && md5sum -c --quiet MD5SUMS) || { log "T4B_CODE_BAD"; mirror; exit 3; }

# Dispatch zero-training soup/ckpt sweep cleanly if NAME=t4b3s (keep t4b_chain.sh alive for guard pgrep & flock)
if [ "$NAME" = "t4b3s" ]; then
  bash $C/t4b_sweep_chain.sh "$WS" "$TAG" "$NAME"
  exit $?
fi

MREPO=Qwen/Qwen3.5-4B
MREV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
if [ "$NAME" = "t4b2" ]; then
  LORA_R=${LORA_R:-16}
  LORA_A=${LORA_A:-32}
  LR=${LR:-4e-5}
  HEAD_LR=${HEAD_LR:-2e-4}
  WD=${WD:-0.02}
  DIAG_FIRST=${DIAG_FIRST:-1}
elif [ "$NAME" = "t4b3" ]; then
  LORA_R=${LORA_R:-32}
  LORA_A=${LORA_A:-64}
  LR=${LR:-4e-5}
  HEAD_LR=${HEAD_LR:-2e-4}
  WD=${WD:-0.02}
  BIDIR=${BIDIR:-1}
  LAYOUT=${LAYOUT:-t4b_bidir_re2_v1}
  READ_TWICE=${READ_TWICE:-2048}
  DIAG_FIRST=${DIAG_FIRST:-0}
fi
STEPS=${STEPS:-3516}
RPS=${RPS:-128}
MBSZ=${MBSZ:-4}
LR=${LR:-2e-4}
HEAD_LR=${HEAD_LR:-1e-3}
LORA_R=${LORA_R:-64}
LORA_A=${LORA_A:-128}
WD=${WD:-0.01}
BIDIR=${BIDIR:-0}
LAYOUT=${LAYOUT:-t4b_sandwich_v1}
READ_TWICE=${READ_TWICE:-0}
DIAG_FIRST=${DIAG_FIRST:-0}
WARMUP=${WARMUP:-100}
FLOOR=${FLOOR:-0.05}
SAVE=${SAVE:-250}
EVAL_EVERY=${EVAL_EVERY:-500}
SEED=${SEED:-20261119}
EXTRA_FLAGS="--layout $LAYOUT --read-twice-tokens $READ_TWICE"
[ "$BIDIR" = "1" ] && EXTRA_FLAGS="$EXTRA_FLAGS --bidir"

# 2. Restage venv, kit/eval refs, Qwen3.5-4B, and raw S3 text datasets in parallel (explicit PIDs)
if [ ! -f $W/RESTAGE_OK ]; then
  pids=()
  # (a) Isolated venv matching TX's tested Qwen3.5 stack + peft + flash-linear-attention
  (
    probe() {
      $V/bin/python -c "
import torch, transformers, peft, fla, safetensors, accelerate, numpy, httpx
assert torch.__version__.startswith('2.14.1') and transformers.__version__ == '5.23.0', (torch.__version__, transformers.__version__)
torch.zeros(1).cuda(); print('VENV_OK', torch.__version__, transformers.__version__, peft.__version__)"
    }
    if ! probe > $W/logs/venv_probe.log 2>&1; then
      python3 -m venv --clear $V
      $V/bin/pip install -q "torch==2.14.1+cu130" --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple > $W/logs/venv.log 2>&1
      $V/bin/pip install -q "transformers==5.23.0" "peft==0.21.3" "flash-linear-attention==0.5.2" "fla-core==0.5.2" "safetensors>=0.6" "accelerate>=1.10" numpy httpx pillow sentencepiece protobuf tiktoken jinja2 einops "huggingface_hub>=1.0" >> $W/logs/venv.log 2>&1
      probe > $W/logs/venv_probe.log 2>&1
    fi
    grep -q "^VENV_OK " $W/logs/venv_probe.log && touch $W/VENV_OK
  ) & pids+=($!)

  # (b) Decision Index kit 9eb2dbe2, suite-0.3, sample4400, gate_index, reference runs & TZ unseen refs
  (
    if [ ! -f $R/.ok ]; then
      aws s3 cp --quiet $B/fleetq/code/di03/kit_9eb2dbe2a358004c8782c66e40a83ac07b953fec.tgz $R/src/kit.tgz
      aws s3 cp --quiet $E/heldout/rebuilt_2026-11-02/suite-0.3.tgz $R/src/suite.tgz
      mkdir -p $R/kit && tar xzf $R/src/kit.tgz -C $R/kit && find $R/kit -name '._*' -delete && tar xzf $R/src/suite.tgz -C $R
      aws s3 cp --quiet $ER/esbar_v8/public_di_full/code/v8f_code_r2.tgz $R/src/v8f_code_r2.tgz
      mkdir -p $R/gc && tar xzf $R/src/v8f_code_r2.tgz -C $R/gc gate_index.py
      aws s3 cp --quiet $ER/esbar_tb1/sample/sample4400-rows.jsonl.gz $R/src/s4400_rows.jsonl.gz
      aws s3 cp --quiet $E/public_E1/e1_20261104T1251Z/final/results.jsonl.gz $R/src/e1_pub.jsonl.gz
      aws s3 cp --quiet $ER/esbar_wi_fs1f50/public_di_full/wi_fs1f50_full_20261112/final/results.jsonl.gz $R/src/munku_pub.jsonl.gz
      aws s3 cp --quiet $ER/esbar_tx/screens/kudhab_t5/results.jsonl.gz $R/src/kudhab_t5.jsonl.gz 2>/dev/null || true
      aws s3 cp --quiet $ER/esbar_tx/screens/q38p1/results.jsonl.gz $R/src/q38p1.jsonl.gz 2>/dev/null || true
      # TZ unseen sets (gde1v3s 10,899 ESCI-cleaned + dev 7,551 unseen families) + reference reads
      mkdir -p $R/unseen
      for s in gde1v3s dev; do
        aws s3 cp --quiet $TZ/unseen/$s.req.jsonl.gz $R/unseen/$s.req.jsonl.gz
        for m in karpo kudhab lg; do
          aws s3 cp --quiet $TZ/unseen/reads/$m/$s.p.jsonl.gz $R/unseen/$s.$m.p.jsonl.gz 2>/dev/null || true
        done
      done
      [ "$(sha256sum < $R/src/s4400_rows.jsonl.gz | cut -c1-16)" = "6296ba7ae4f9269f" ] && [ -s $R/unseen/gde1v3s.req.jsonl.gz ] && [ -s $R/unseen/dev.req.jsonl.gz ] && touch $R/.ok
    fi
  ) & pids+=($!)

  # (c) Qwen/Qwen3.5-4B pinned snapshot
  (
    if [ ! -f $M/MODEL_OK ]; then
      python3 -m pip install -q --user "huggingface_hub>=1.0" > /dev/null 2>&1 || true
      python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MREPO', revision='$MREV', local_dir='$M', token=False, max_workers=16)" > $W/logs/model.log 2>&1
      python3 - $M >> $W/logs/model.log 2>&1 <<'PY' && touch $M/MODEL_OK
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

  # (d) Clean karpo-31B training sources from S3
  (
    if [ ! -f $DATA/src/.ok ]; then
      TL=$ER/esbar_mx/tl
      aws s3 cp --quiet $TL/mxa.jsonl $DATA/src/mxa.jsonl
      aws s3 cp --quiet $ER/esbar_dh/pools/fs5h_rows.jsonl $DATA/src/fs5h_rows.jsonl
      aws s3 cp --quiet $ER/esbar_dh/droplist_roundH.txt $DATA/src/droplist_roundH.txt
      for f in fs5.jsonl.gz teacher_karpo_fs5.jsonl.gz fs5h0.jsonl.gz teacher_karpo_fs5h0.jsonl.gz fs5h.jsonl.gz teacher_karpo_fs5h.jsonl.gz fs5h_multi.jsonl.gz teacher_karpo_fs5h_multi.jsonl.gz; do
        aws s3 cp --quiet $TL/$f $DATA/src/$f
      done
      aws s3 cp --quiet $ER/esbar_kl2_data/native_ns.jsonl.gz $DATA/src/native_ns.jsonl.gz
      aws s3 cp --quiet $ER/esbar_kl2_data/teacher_karpo_native_ns.jsonl.gz $DATA/src/teacher_karpo_native_ns.jsonl.gz
      [ -s $DATA/src/fs5.jsonl.gz ] && [ -s $DATA/src/teacher_karpo_fs5.jsonl.gz ] && [ -s $DATA/src/native_ns.jsonl.gz ] && [ -s $DATA/src/fs5h_multi.jsonl.gz ] && touch $DATA/src/.ok
    fi
  ) & pids+=($!)

  for pid in "${pids[@]}"; do wait "$pid" || true; done
  E_OK=$([ -f $W/VENV_OK ] && echo 1 || echo 0)
  K_OK=$([ -f $R/.ok ] && echo 1 || echo 0)
  M_OK=$([ -f $M/MODEL_OK ] && echo 1 || echo 0)
  D_OK=$([ -f $DATA/src/.ok ] && echo 1 || echo 0)
  log "RESTAGE check venv=$E_OK kit=$K_OK model=$M_OK data_src=$D_OK ($(cat $W/logs/venv_probe.log 2>/dev/null | tail -1))"
  if [ "$E_OK" = "1" ] && [ "$K_OK" = "1" ] && [ "$M_OK" = "1" ] && [ "$D_OK" = "1" ]; then
    touch $W/RESTAGE_OK
  else
    log "RESTAGE_FAIL"; mirror; exit 4
  fi
fi

# 3. Build clean training JSONL (~450k rows, native noul/score, zero DMCA/NC-SA/droplist)
if [ ! -f $DATA/t4b_train.ok ]; then
  $V/bin/python $C/t4b_prep_data.py --src-dir $DATA/src --out $DATA/t4b_train.jsonl --stats $DATA/t4b_data_stats.json --multi-per-req 2 > $W/logs/prep_data.log 2>&1 \
    && touch $DATA/t4b_train.ok \
    || { log "PREP_DATA_FAIL $(tail -5 $W/logs/prep_data.log | tr '\n' ' ' | cut -c1-400)"; mirror; exit 4; }
  aws s3 cp --quiet $DATA/t4b_data_stats.json $U/t4b_data_stats.json
  log "PREP_DATA $(grep T4B_DATA_OK $W/logs/prep_data.log | tail -1)"
fi

# Compute exact 1-epoch step count if STEPS=auto or keep default STEPS
NROWS=$(wc -l < $DATA/t4b_train.jsonl)
if [ "$STEPS" = "auto" ]; then
  STEPS=$(( (NROWS + RPS - 1) / RPS ))
fi
log "CONFIG rows=$NROWS steps=$STEPS rps=$RPS micro_batch=$MBSZ lora_r=$LORA_R lora_a=$LORA_A lr=$LR head_lr=$HEAD_LR wd=$WD bidir=$BIDIR layout=$LAYOUT read_twice=$READ_TWICE"
mirror

export PYTHONPATH=$C:$R/kit
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 3b. Optional 6-min 5-probe diagnostic on stratified 1,100 rows (s4400[::4]) before training
if [ "$DIAG_FIRST" = "1" ] && ! aws s3 ls $T4B/diag/diag_done > /dev/null 2>&1; then
  log "DIAG_START (5 probes on 1,100 stratified s4400 rows)"
  DDIR=$W/diag
  mkdir -p $DDIR/src $DDIR/t4b1_ck3516 $DDIR/t4b1_ck250
  aws s3 sync --quiet $T4B/arms/t4b1/ckpt/ckpt_03516/ $DDIR/t4b1_ck3516/ --exclude "optim.pt"
  aws s3 sync --quiet $T4B/arms/t4b1/ckpt/ckpt_00250/ $DDIR/t4b1_ck250/ --exclude "optim.pt"
  aws s3 cp --quiet $T4B/arms/t4b1/screen/results.jsonl.gz $DDIR/t4b1_full.jsonl.gz
  $V/bin/python - $R/src/s4400_rows.jsonl.gz $DDIR/src 8 <<'PY'
import gzip, sys
rows = [r for r in gzip.open(sys.argv[1], "rt", encoding="utf-8").read().split("\n") if r][::4]
with gzip.open(f"{sys.argv[2]}/s1100.jsonl.gz", "wt", encoding="utf-8") as f:
    for r in rows: f.write(r + "\n")
n = int(sys.argv[3])
fs = [gzip.open(f"{sys.argv[2]}/s_{k}.jsonl.gz", "wt", encoding="utf-8") for k in range(n)]
for i, r in enumerate(rows): fs[i % n].write(r + "\n")
for f in fs: f.close()
print("DIAG_ROWS", len(rows))
PY
  run_diag_arm() {
    local albl=$1 amode=$2 ackpt=$3 ascl=$4
    [ -s $DDIR/$albl.jsonl.gz ] && return 0
    local pids=()
    for g in 0 1 2 3 4 5 6 7; do
      (
        mkdir -p $DDIR/run/$albl/$g
        cd $DDIR/run/$albl/$g
        CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$C:$R/kit HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
          $V/bin/python -m decision_index run --engine t4b_diag:DiagEngine \
          --option model=$M --option mode=$amode --option ckpt=$ackpt --option scale_mult=$ascl \
          --edition 0.3 --suite-dir $R/suite-0.3 --rows $DDIR/src/s_$g.jsonl.gz \
          --out $DDIR/run/$albl/$g/out --compact > $DDIR/run/$albl/$g/kit.log 2>&1
      ) & pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" || true; done
    cat $DDIR/run/$albl/*/out/results.jsonl | gzip -n > $DDIR/$albl.jsonl.gz
    log "DIAG_ARM $albl rows=$(zcat $DDIR/$albl.jsonl.gz | wc -l)"
  }
  run_diag_arm stock_causal stock_causal none 1.0
  run_diag_arm stock_bidir  stock_bidir  none 1.0
  run_diag_arm stock_plain  stock_plain  none 1.0
  run_diag_arm t4b1_s35     stock_causal $DDIR/t4b1_ck3516 0.35
  run_diag_arm t4b1_ck250   stock_causal $DDIR/t4b1_ck250  1.0
  DARMS="kudhab=$R/src/kudhab_t5.jsonl.gz E1=$R/src/e1_pub.jsonl.gz t4b1=$DDIR/t4b1_full.jsonl.gz stock_causal=$DDIR/stock_causal.jsonl.gz stock_bidir=$DDIR/stock_bidir.jsonl.gz stock_plain=$DDIR/stock_plain.jsonl.gz t4b1_s35=$DDIR/t4b1_s35.jsonl.gz t4b1_ck250=$DDIR/t4b1_ck250.jsonl.gz"
  PYTHONPATH=$R/kit $V/bin/python $R/gc/gate_index.py $R/suite-0.3 $DDIR/src/s1100.jsonl.gz $DDIR/w_score $DDIR/diag.json $DARMS > $DDIR/diag_raw.txt 2>&1 || true
  $V/bin/python - $DDIR/diag.json > $DDIR/diag_summary.txt <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
arms = d["arms"]
print("=== 1,100-ROW STRATIFIED DIAGNOSTIC ===")
for name, v in arms.items():
    print(f"  {name:14s}: index={v['index']:6.3f}  areas={v['areas']}")
print("\nKEY BENCHMARKS (43=CRUXEval, 56=Phish, 30=GSM8K, 44=CLadder, 20=BPoMP, 25=GPQA, 57=MMLUPro, 58=BBH, 59=RAGTruth, 5=CLINC150):")
keys = ["43", "56", "30", "44", "20", "25", "57", "58", "59", "5"]
hdr = f"{'arm':14s} " + " ".join(f"{k:>6s}" for k in keys)
print(hdr)
for name, v in arms.items():
    b = v.get("bench", {})
    vals = " ".join(f"{(b.get(k) or 0)*100:6.1f}" for k in keys)
    print(f"{name:14s} {vals}")
PY
  aws s3 cp --quiet $DDIR/diag.json $T4B/diag/diag.json
  aws s3 cp --quiet $DDIR/diag_summary.txt $T4B/diag/diag_summary.txt
  date -u +%FT%TZ | aws s3 cp --quiet - $T4B/diag/diag_done
  log "DIAG_DONE $(grep 'index=' $DDIR/diag_summary.txt | tr '\n' ' ' | cut -c1-400)"
  rm -rf $DDIR/run $DDIR/t4b1_ck3516 $DDIR/t4b1_ck250
  mirror
fi

# 4. Pre-training smoke: step-0 head & right-pad parity, 4-step train + S3 save/resume, T4BEngine merge & prefix-cache parity
if ! aws s3 ls $U/state/${NAME}_smoke_ok > /dev/null 2>&1; then
  log "SMOKE_START"
  rm -rf $W/smoke
  mkdir -p $W/smoke/run
  # (a) Step-0 head + right-pad micro-batch parity
  $V/bin/torchrun --nproc-per-node 8 --master-port 29541 $C/t4b_train.py \
    --model $M --rows $DATA/t4b_train.jsonl --out $W/smoke/parity --smoke-parity --lora-r $LORA_R --lora-alpha $LORA_A --micro-batch $MBSZ $EXTRA_FLAGS \
    > $W/logs/smoke_parity.log 2>&1 || { log "SMOKE_PARITY_FAIL $(tail -5 $W/logs/smoke_parity.log | tr '\n' ' ' | cut -c1-400)"; mirror; exit 7; }
  log "SMOKE_HEAD $(cat $W/smoke/parity/smoke_parity.json | tr -d '\n' | cut -c1-300)"

  # (b) 4-step train + save + 2-step resume smoke
  $V/bin/torchrun --nproc-per-node 8 --master-port 29542 $C/t4b_train.py \
    --model $M --rows $DATA/t4b_train.jsonl --out $W/smoke/run --steps 2 --rows-per-step 64 --micro-batch $MBSZ \
    --lr $LR --head-lr $HEAD_LR --warmup 10 --lora-r $LORA_R --lora-alpha $LORA_A --save-every 2 --seed $SEED $EXTRA_FLAGS \
    > $W/logs/smoke_tr1.log 2>&1 || { aws s3 cp --quiet $W/logs/smoke_tr1.log $U/logs/smoke_tr1.log 2>/dev/null || true; log "SMOKE_TR1_FAIL $(grep -E 'Error:|Exception:|assert|Checkpoint' $W/logs/smoke_tr1.log | tail -5 | tr '\n' ' ' | cut -c1-400)"; mirror; exit 7; }
  $V/bin/torchrun --nproc-per-node 8 --master-port 29542 $C/t4b_train.py \
    --model $M --rows $DATA/t4b_train.jsonl --out $W/smoke/run --steps 4 --rows-per-step 64 --micro-batch $MBSZ \
    --lr $LR --head-lr $HEAD_LR --warmup 10 --lora-r $LORA_R --lora-alpha $LORA_A --save-every 2 --seed $SEED $EXTRA_FLAGS \
    --resume $W/smoke/run/ckpt_00002 \
    > $W/logs/smoke_tr2.log 2>&1 || { aws s3 cp --quiet $W/logs/smoke_tr2.log $U/logs/smoke_tr2.log 2>/dev/null || true; log "SMOKE_TR2_FAIL $(grep -E 'Error:|Exception:|assert|Checkpoint' $W/logs/smoke_tr2.log | tail -5 | tr '\n' ' ' | cut -c1-400)"; mirror; exit 7; }

  # (c) T4BEngine merged-weight parity & multi-question prefix-cache parity
  CUDA_VISIBLE_DEVICES=0 $V/bin/python $C/t4b_eval.py --mode parity --model $M --ckpt $W/smoke/run/ckpt_00004 \
    --s4400 $R/src/s4400_rows.jsonl.gz --out $W/smoke/engine_parity.json \
    > $W/logs/smoke_eng.log 2>&1 || { log "SMOKE_ENG_FAIL $(tail -5 $W/logs/smoke_eng.log | tr '\n' ' ' | cut -c1-400)"; mirror; exit 7; }
  log "SMOKE_ENGINE $(cat $W/smoke/engine_parity.json | tr -d '\n' | cut -c1-350)"

  aws s3 cp --quiet $W/smoke/parity/smoke_parity.json $U/smoke_parity.json
  aws s3 cp --quiet $W/smoke/engine_parity.json $U/smoke_engine_parity.json
  rm -rf $W/smoke
  date -u +%FT%TZ | aws s3 cp --quiet - $U/state/${NAME}_smoke_ok
  log "SMOKE_OK"
  mirror
fi

# 5. Full 1-epoch training with S3 checkpoint resume & background log mirror
S3CK=$U/ckpt
FINAL=$(printf "ckpt_%05d" "$STEPS")
if ! aws s3 ls $S3CK/$FINAL/state.json > /dev/null 2>&1; then
  # Background log mirror (explicit PID, killed cleanly after training)
  (
    while :; do
      sleep 120
      [ -f $RUN/train_log.jsonl ] && aws s3 cp --quiet $RUN/train_log.jsonl $U/logs/train_log.jsonl 2>/dev/null || true
      mirror
    done
  ) &
  MIRROR_PID=$!
  trap 'kill $MIRROR_PID 2>/dev/null || true' EXIT

  LAST=$(aws s3 ls $S3CK/ | awk '/PRE ckpt_/{print $2}' | tr -d '/' | sort | tail -1)
  RES=""
  if [ -n "$LAST" ]; then
    mkdir -p $RUN/$LAST
    aws s3 sync --quiet $S3CK/$LAST/ $RUN/$LAST/
    [ -s $RUN/$LAST/state.json ] && RES="--resume $RUN/$LAST"
    log "RESUMING from $LAST"
  else
    log "TRAIN_START fresh steps=$STEPS"
  fi
  mirror

  $V/bin/torchrun --nproc-per-node 8 --master-port 29543 $C/t4b_train.py \
    --model $M --rows $DATA/t4b_train.jsonl --out $RUN --s3 $S3CK \
    --dev-req $R/unseen/dev.req.jsonl.gz --dev-n 512 --eval-every $EVAL_EVERY \
    --steps $STEPS --rows-per-step $RPS --micro-batch $MBSZ \
    --lr $LR --head-lr $HEAD_LR --warmup $WARMUP --floor $FLOOR --wd $WD --clip 1.0 \
    --lora-r $LORA_R --lora-alpha $LORA_A --max-tokens 8192 --save-every $SAVE --seed $SEED $EXTRA_FLAGS $RES \
    > $W/logs/train_stdout.log 2>&1
  TR_RC=$?
  kill $MIRROR_PID 2>/dev/null || true
  wait $MIRROR_PID 2>/dev/null || true
  [ -f $RUN/train_log.jsonl ] && aws s3 cp --quiet $RUN/train_log.jsonl $U/logs/train_log.jsonl 2>/dev/null || true
  log "TRAIN_EXIT rc=$TR_RC last=$(grep '"event": "step"' $RUN/train_log.jsonl 2>/dev/null | tail -1 | cut -c1-200)"
  mirror
  [ "$TR_RC" = "0" ] || exit $TR_RC
fi

# Ensure final checkpoint is present locally
[ -s $RUN/$FINAL/readout.pt ] || { mkdir -p $RUN/$FINAL && aws s3 sync --quiet $S3CK/$FINAL/ $RUN/$FINAL/; }
[ -s $RUN/$FINAL/readout.pt ] || { log "NO_FINAL_CKPT $FINAL"; mirror; exit 8; }
log "FINAL_CKPT_OK $FINAL"
mirror

# 6. Eval A: 4,400-row Text Screen (sharded across 8 GPUs via Decision Index kit)
SDIR=$EVAL/screen
if [ ! -s $SDIR/results.jsonl.gz ]; then
  log "SCREEN_4400_START"
  mkdir -p $SDIR/src $SDIR/run
  $V/bin/python - $R/src/s4400_rows.jsonl.gz $SDIR/src 8 <<'PY'
import gzip, sys
rows = [r for r in gzip.open(sys.argv[1], "rt", encoding="utf-8").read().split("\n") if r]
n = int(sys.argv[3])
fs = [gzip.open(f"{sys.argv[2]}/s_{k}.jsonl.gz", "wt", encoding="utf-8") for k in range(n)]
for i, r in enumerate(rows):
    fs[i % n].write(r + "\n")
for f in fs:
    f.close()
PY
  pids=()
  for g in 0 1 2 3 4 5 6 7; do
    (
      mkdir -p $SDIR/run/$g
      cd $SDIR/run/$g
      CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$C:$R/kit HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        $V/bin/python -m decision_index run --engine t4b_engine:T4BEngine \
        --option model=$M --option ckpt=$RUN/$FINAL --option max_len=65536 \
        --edition 0.3 --suite-dir $R/suite-0.3 --rows $SDIR/src/s_$g.jsonl.gz \
        --out $SDIR/run/$g/out --compact > $SDIR/run/$g/kit.log 2>&1
    ) & pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
  cat $SDIR/run/*/out/results.jsonl > $SDIR/results.jsonl 2>/dev/null || true
  N_SCR=$(wc -l < $SDIR/results.jsonl)
  log "SCREEN_4400_ROWS $N_SCR statuses $(grep -o '"status": *"[a-z]*"' $SDIR/results.jsonl | sort | uniq -c | tr '\n' ' ')"
  [ "$N_SCR" = "4400" ] || { log "SCREEN_4400_INCOMPLETE $(tail -5 $SDIR/run/0/kit.log | tr '\n' ' ' | cut -c1-300)"; mirror; exit 9; }
  gzip -n -c $SDIR/results.jsonl > $SDIR/results.jsonl.gz
  aws s3 cp --quiet $SDIR/results.jsonl.gz $U/screen/results.jsonl.gz

  # Score full + halves h0/h1 with gate_index.py alongside E1, munku, kudhab_t5, q38p1
  $V/bin/python - $R/src/s4400_rows.jsonl.gz $SDIR <<'PY'
import gzip, hashlib, json, sys
ids = [json.loads(l)["_evaluation"]["run_id"] for l in gzip.open(sys.argv[1], "rt")]
for h in (0, 1):
    open(f"{sys.argv[2]}/h{h}.txt", "w").write("".join(i + "\n" for i in ids if int(hashlib.sha1(i.encode()).hexdigest(), 16) % 2 == h))
PY
  ARMS="E1=$R/src/e1_pub.jsonl.gz munku=$R/src/munku_pub.jsonl.gz"
  [ -s $R/src/kudhab_t5.jsonl.gz ] && ARMS="$ARMS kudhab=$R/src/kudhab_t5.jsonl.gz"
  [ -s $R/src/q38p1.jsonl.gz ] && ARMS="$ARMS q38p1=$R/src/q38p1.jsonl.gz"
  ARMS="$ARMS $NAME=$SDIR/results.jsonl.gz"
  : > $SDIR/cmp.txt
  for sp in full h0 h1; do
    IDS=$R/src/s4400_rows.jsonl.gz
    [ "$sp" = "full" ] || IDS=$SDIR/$sp.txt
    PYTHONPATH=$R/kit $V/bin/python $R/gc/gate_index.py $R/suite-0.3 $IDS $SDIR/w_$sp $SDIR/$sp.json $ARMS 2>&1 \
      | grep -E '^(INDEX|DIFF)' | sed "s/^/$sp /" >> $SDIR/cmp.txt || true
  done
  aws s3 cp --quiet $SDIR/cmp.txt $U/screen/cmp.txt
  for sp in full h0 h1; do [ -s $SDIR/$sp.json ] && aws s3 cp --quiet $SDIR/$sp.json $U/screen/$sp.json; done
  log "SCREEN_4400_DONE $(grep "full INDEX $NAME " $SDIR/cmp.txt | head -1)"
  mirror
fi

# 7. Eval B: Unseen Private-Like Proxies (gde1v3s 10,899 ESCI-cleaned + dev 7,551 unseen families) across 8 GPUs
UDIR=$EVAL/unseen
if [ ! -s $UDIR/unseen_score.json ]; then
  log "UNSEEN_START"
  mkdir -p $UDIR/work
  for s in gde1v3s dev; do
    cp $R/unseen/$s.req.jsonl.gz $UDIR/work/$s.req.jsonl.gz
    for m in karpo kudhab lg; do
      [ -s $R/unseen/$s.$m.p.jsonl.gz ] && cp $R/unseen/$s.$m.p.jsonl.gz $UDIR/work/$s.$m.p.jsonl.gz
    done
    $V/bin/python - $UDIR/work/$s.req.jsonl.gz $UDIR/work $s 8 <<'PY'
import gzip, sys
rows = [r for r in gzip.open(sys.argv[1], "rt", encoding="utf-8").read().split("\n") if r]
n = int(sys.argv[4])
fs = [gzip.open(f"{sys.argv[2]}/{sys.argv[3]}_{k}.req.jsonl.gz", "wt", encoding="utf-8") for k in range(n)]
for i, r in enumerate(rows):
    fs[i % n].write(r + "\n")
for f in fs:
    f.close()
PY
  done
  pids=()
  for g in 0 1 2 3 4 5 6 7; do
    (
      for s in gde1v3s dev; do
        CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$C:$R/kit HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
          $V/bin/python $C/t4b_eval.py --mode unseen --model $M --ckpt $RUN/$FINAL \
          --req $UDIR/work/${s}_$g.req.jsonl.gz --out $UDIR/work/${s}_$g.$NAME.jsonl \
          > $UDIR/work/${s}_$g.log 2>&1
      done
    ) & pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
  for s in gde1v3s dev; do
    cat $UDIR/work/${s}_*.$NAME.jsonl | gzip -n > $UDIR/work/$s.$NAME.p.jsonl.gz
    aws s3 cp --quiet $UDIR/work/$s.$NAME.p.jsonl.gz $U/unseen/$s.$NAME.p.jsonl.gz
  done
  $V/bin/python $C/tz_unseen_score.py $UDIR/work $UDIR/unseen_score.json > $UDIR/unseen_score.txt 2>&1 || true
  aws s3 cp --quiet $UDIR/unseen_score.json $U/unseen/unseen_score.json
  aws s3 cp --quiet $UDIR/unseen_score.txt $U/unseen/unseen_score.txt
  log "UNSEEN_DONE $(grep -E "^(==|   $NAME |   karpo |   lg )" $UDIR/unseen_score.txt | tr '\n' ' ' | cut -c1-450)"
  mirror
fi

# 8. Final Result JSON + Done Marker (triggers ctl2 guard auto-termination)
$V/bin/python $C/t4b_eval.py --mode summarize --arm $NAME \
  --data-stats $DATA/t4b_data_stats.json \
  --train-log $RUN/train_log.jsonl \
  --screen-dir $SDIR \
  --unseen-json $UDIR/unseen_score.json \
  --out $EVAL/${NAME}_result.json > $W/logs/summary.log 2>&1
aws s3 cp --quiet $EVAL/${NAME}_result.json $U/${NAME}_result.json
log "RESULT $(cat $W/logs/summary.log | tr '\n' ' ' | cut -c1-450)"
date -u +%FT%TZ | aws s3 cp --quiet - $U/state/${NAME}_chain_done
log "CHAIN_DONE"
mirror
