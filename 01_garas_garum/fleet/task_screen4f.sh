#!/bin/sh
# task_screen4f.sh mixp | run l|p|lp 0|1 [tryN] | read [tryN]
#   -- (SCREEN4F) the registered SCREEN4 follow-up after GUARD_BLOCKED (2026-10-13): arm C's caldev NLL harm vs Garas native.
# Pre-registration tmp/screen4f/prereg.md (design + diagnosis there). A 2 x 2 on top of SCREEN4's arm C (C itself = SCREEN4's c0 / c1,
# re-read, not re-run): every cell is C's recipe (claude_lmtune e2c6100a; (99)-init LoRA r16 attn + MLP; warmup 40, cosine; --rps-weight
# 1.0; max-len 2048; global 8 = 2 x 4; 1,500 steps = 12,000 units; gemma fast path; the residual VBLL head grid) on C's records, except:
#   l   peak lr 1.5e-5 (C: 5e-5)                       data = SCREEN4's arm c (s4_mixed_v1, unchanged)
#   p   Garas-parent preservation weight 1.0 (C: 0.3)  data = arm c re-mixed (mixp; the EG teacher weight stays 0.3)
#   lp  both                                           data = p's
# Hardware: SINGLE-GPU H100 for run (bounce elsewhere); mixp / read are CPU.
#   mixp       CPU: build_screen4f.py mixp on SCREEN4's prep tarball + teach npz files (NOT re-taught) with the record-by-record check
#              against SCREEN4's arm c -> results_screen4f/data/s4f_mixp_v1.tgz (.md5 LAST)
#   run A S    H100: train (gates as SCREEN4: 12,000 units; step-0 identity; init applied; no NC; VBLL head trained; + the arm's lr) ->
#              mean-head checkpoint -> DEV scores (didev / egdev / caldev; both checkpoints) -> run record LAST
#   read       CPU: analysis/screen4f_read.py (imports SCREEN4's registered reader) -> results_screen4f/read/verdict.txt LAST
# Inputs (read-only): results_screen4/ (prep tarball, mixed tarball, teach npz, ref I scores, C's run records and scores).
# Code: ONLY md5-pinned files from fleetq/code/pc/, fleetq/code/ext5/ and fleetq/code/screen4/ (SCREEN4's deployed set + the two
# SCREEN4F files). Never touches results_screen4/ for writing; CONFIRM-2610 is not read.
MODE=${1:?usage: task_screen4f.sh mixp | run A S | read}
J=${JEV_J:-/data/shared/jev}; D=${JEV_D:-/data/local}; FL=${JEV_FLEET:-/data/shared/fleet}
S3=s3://YOUR_BUCKET/jev-race; RR4=$S3/results_screen4; RR=$S3/results_screen4f; CB=$S3/fleetq/code; W=$D/screen4
[ -f $FL/env.sh ] && . $FL/env.sh
export HF_HOME=${JEV_HF:-$D/hf} HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $W/data $W/runs || exit 1
GM=google/gemma-4-31B-it
C99_MD5=a4ee3b50c1a578c6536fc3727170fd0f         # results99/bbtune_gemma31it_99.pt (Garas / (99))
RP_MD5=67359006eb3f51aa27dd210df90c0c23          # results_pc/data/pc_replay_v1.tgz (R, R2, mix2 val)
FAST_MD5=433ac777db0a365121589ea59b12be58        # fleetq/scripts/gemma_fast_h100.sh
PREP_MD5=f4bb81bc9dadc2892e3f44c0479b4a02        # results_screen4/data/s4_prep_v1.tgz (SCREEN4 prep try1, A2 builder)
MIXED_MD5=9a54394e6345ea7b56d8899a6a77966c       # results_screen4/data/s4_mixed_v1.tgz (SCREEN4 mix try1, A3 builder; arm c = cell C / L data)
TEACH_MD5=15334d81a6457b6114b930b3d529a814       # results_screen4/teach/teach.done.json (the three teacher npz md5s)
S4_BUILD=d1a0ed10f963be7c01619d49684521ed                   # fleetq/code/screen4/build_screen4.py (A3, deployed)
S4_LOSS=0388f88d7915b022b0e44634d4a581d7                     # fleetq/code/screen4/trainers/screen4_loss.py
S4_HEAD=0df767bcd3dbd83d8fd7a57d0effed6c                     # fleetq/code/screen4/trainers/screen4_head.py
S4_READ=607234f61d30baea9f462e80a5620b45                     # fleetq/code/screen4/analysis/screen4_read.py (SCREEN4's registered reader)
S4F_BUILD=TBD                                               # fleetq/code/screen4/build_screen4f.py
S4F_READ=TBD                                                # fleetq/code/screen4/analysis/screen4f_read.py
HEAD_FLAGS="--aux-vbll --aux-vbll-prior 1 --aux-vbll-kl 0,1,10000,1000000 --aux-vbll-cov diag,full --aux-vbll-dim 128 --aux-vbll-samples 64 --aux-vbll-lr 0.001"
have() { aws s3 ls $1 > /dev/null 2>&1; }
fetch() {   # fetch DIR KEY MD5: fleetq/code/KEY -> DIR/<KEY without its first component>, md5-checked
  dst=$1/${2#*/}; mkdir -p $(dirname $dst) && aws s3 cp --only-show-errors $CB/$2 $dst || { echo "NO_CODE $2"; exit 1; }
  got=$(md5sum < $dst | cut -c1-32); [ "$got" = "$3" ] || { echo "CODE_MD5 $2 $got != $3"; exit 1; }; }
getd() {    # getd S3KEY MD5 NAME CHECKPATH: fetch + md5 + extract into $W/data/NAME, once (shares SCREEN4's cache names)
  [ -f $W/data/$3/.ok ] && [ "$(cat $W/data/$3/.ok)" = $2 ] && [ -e $W/data/$3/$4 ] && return 0
  aws s3 cp --only-show-errors $S3/$1 $W/data/$3.tgz || { echo "NO_DATA $1"; return 1; }
  [ "$(md5sum < $W/data/$3.tgz | cut -c1-32)" = $2 ] || { echo "DATA_MD5 $1"; return 1; }
  x=$W/data/.x_$3_$$; mkdir -p $x && tar xzf $W/data/$3.tgz -C $x && [ -e $x/$4 ] || { echo "UNTAR_FAIL $1 (no $4)"; return 1; }
  [ -e $W/data/$3 ] && mv $W/data/$3 $W/data/$3.stale.$(date +%s)
  mv $x $W/data/$3 && echo $2 > $W/data/$3/.ok && rm -f $W/data/$3.tgz; }
put() {     # put LOCAL KEY: upload under results_screen4f/ + read-back md5
  aws s3 cp --only-show-errors $1 $RR/$2 || { echo "UPLOAD_FAIL $2"; exit 1; }
  [ "$(aws s3 cp --only-show-errors $RR/$2 - | md5sum | cut -c1-32)" = "$(md5sum < $1 | cut -c1-32)" ] || { echo "READBACK_MISMATCH $2"; exit 1; }; }
need_h100() {
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  case "$GPU_NAME|${JEV_SLOT:-}" in *H100*"|") echo "affinity ok: $GPU_NAME";; *) echo "BOUNCE_NEEDED: '$GPU_NAME' slot '${JEV_SLOT:-}' is not a single-GPU H100 (re-queue this line on g8 / g9 / g10)"; exit 3;; esac; }
prod_code() {   # PC's pinned production set into $1 (trainer + scorer dir)
  for p in "pc/trainers/claude_lmtune.py e2c6100a1c0651804613da6aaa477fb0" "pc/trainers/claude_lmscore.py e0672bd0f515ccb17d97035c84c3d339" \
           "pc/trainers/cache_features.py 49613a40faed8cd612175e81c3f3925f" "pc/trainers/claude_ckpt.py fc271c212f77821a72c98a8b3dd4fa03" \
           "pc/trainers/train_e2e.py 971983177fe857f576e3e72531360558" "pc/trainers/train_head.py 30171b53b75bad8419dcf717b2d7ea77" \
           "pc/trainers/claude_ordinal.py 4726629353c91cda55c485756c8454bc" "pc/vbll.py 2ce7d97a3e2bd109606f9a602ce554b8"; do
    set -- $1 $p; fetch $1 $2 $3; done; }
s4_code() {     # SCREEN4's deployed files + the two SCREEN4F files + the overlap tools into $1
  fetch $1 screen4/build_screen4.py $S4_BUILD; fetch $1 screen4/trainers/screen4_loss.py $S4_LOSS
  fetch $1 screen4/trainers/screen4_head.py $S4_HEAD; fetch $1 screen4/analysis/screen4_read.py $S4_READ
  fetch $1 screen4/build_screen4f.py $S4F_BUILD; fetch $1 screen4/analysis/screen4f_read.py $S4F_READ
  for p in "ext5/build_jev_ext2.py 8ac5de094bdc7747e47f229e9b968176" "ext5/build_jev_ext3.py 61585da5ee80bc2d3267aa0efbd952a2" \
           "ext5/build_jev_ext.py 41271b13bca9c87550cd791ea16f0f07" "ext5/build_jev_data.py c9d4570b000aa39b6aca792947c6a145" \
           "ext5/analysis/claude_overlap.py 0bcf9b1ff942fd62808f646627da8e59" "ext5/trainers/convert_extevals.py 23cce7db3ad527fe57195a1191bb5558" \
           "pc/trainers/cache_features.py 49613a40faed8cd612175e81c3f3925f"; do set -- $1 $p; fetch $1 $2 $3; done; }
score() {       # score DIR OUT ARGS...: the pinned production scorer (PYTHONPATH = its dir only); HAVE-skips from results_screen4f/score
  sd=$1; out=$2; shift 2
  if have $RR/score/$(basename $out); then aws s3 cp --only-show-errors $RR/score/$(basename $out) $out; echo "HAVE $(basename $out)"; return 0; fi
  env PYTHONPATH=$sd python3 $sd/trainers/claude_lmscore.py --model $GM --exact-path --out $out "$@" > $W/runs/$(basename $out .npz).log 2>&1
  rc=$?; echo "SCORE $(basename $out) EXIT=$rc $(grep '^wrote' $W/runs/$(basename $out .npz).log | cut -c1-140)"
  [ $rc -eq 0 ] && [ -s $out ] || { echo "SCORE_FAILED $(basename $out)"; exit 1; }
  put $out score/$(basename $out); put $W/runs/$(basename $out .npz).log score/$(basename $out .npz).log; }
prep_data() {   # SCREEN4's published prep tarball (pinned md5), extracted once (SCREEN4's cache name)
  getd results_screen4/data/s4_prep_v1.tgz $PREP_MD5 s4prep s4_prep_v1/arms/arms_plain_manifest.json || exit 1
  P=$W/data/s4prep/s4_prep_v1; }
echo "== (SCREEN4F $*) start $(date -u +%FT%TZ) on $(hostname)"
case "$S4F_BUILD$S4F_READ" in *TBD*) echo "NOT_STAGED: the SCREEN4F code pins are unset"; exit 1;; esac

# ============================================================================================================ mixp (CPU)
if [ "$MODE" = mixp ]; then
  prep_data
  have $RR/data/s4f_mixp_v1.md5 && { echo "ALREADY_MIXED $(aws s3 cp --only-show-errors $RR/data/s4f_mixp_v1.md5 -)"; exit 0; }
  C=$W/code_mixp_$$; mkdir -p $C; s4_code $C
  getd results_pc/data/pc_replay_v1.tgz $RP_MD5 pcreplay pc_replay_v1/manifest.json || exit 1
  getd results_screen4/data/s4_mixed_v1.tgz $MIXED_MD5 s4mixed s4_mixed_v1/arms/arms_mixed_manifest.json || exit 1
  T=$W/runs/teach_f; mkdir -p $T
  aws s3 cp --only-show-errors $RR4/teach/teach.done.json $T/teach.done.json && [ "$(md5sum < $T/teach.done.json | cut -c1-32)" = $TEACH_MD5 ] || { echo "TEACH_RECORD_MD5"; exit 1; }
  for f in parent_99 teacher_gemma teacher_gemma_dev; do aws s3 cp --only-show-errors $RR4/teach/score/$f.npz $T/$f.npz || exit 1
    [ "$(md5sum < $T/$f.npz | cut -c1-32)" = "$(python3 -c "import json,sys; print(json.load(open('$T/teach.done.json'))['$f.npz'])")" ] || { echo "TEACHER_MD5 $f"; exit 1; }; done
  O=$W/s4f_mixp_v1; [ -e $O ] && mv $O $W/s4f_mixp_v1.stale.$(date +%s); mkdir -p $O; cp -r $P/arms $O/arms
  ( cd $C && PYTHONPATH=$C:$C/analysis:$C/trainers python3 build_screen4f.py --self-test ) > $W/runs/mixp_selftest.log 2>&1 || { echo "MIXP_SELF_TEST_FAILED"; exit 1; }
  ( cd $C && PYTHONPATH=$C:$C/analysis:$C/trainers python3 build_screen4f.py mixp --replay $W/data/pcreplay/pc_replay_v1 --eg $P/eg \
      --parent-npz $T/parent_99.npz --teacher-npz $T/teacher_gemma.npz --teacher-dev-npz $T/teacher_gemma_dev.npz --dev $P/didev --dev $P/caldev \
      --reference $W/data/s4mixed/s4_mixed_v1/arms --out $O/arms ) > $W/runs/mixp.log 2>&1 || { echo "MIXP_FAILED: $(tail -3 $W/runs/mixp.log)"; exit 1; }
  tail -1 $W/runs/mixp.log | cut -c1-300
  COPYFILE_DISABLE=1 tar czf $W/s4f_mixp_v1.tgz -C $W s4f_mixp_v1 || exit 1
  M5=$(md5sum < $W/s4f_mixp_v1.tgz | cut -c1-32); put $W/s4f_mixp_v1.tgz data/s4f_mixp_v1.tgz; put $O/arms/arms_mixed_manifest.json data/s4f_mixp_manifest.json
  echo "$M5  s4f_mixp_v1.tgz" > $W/s4f_mixp_v1.md5; put $W/s4f_mixp_v1.md5 data/s4f_mixp_v1.md5
  echo "MIXP_PUBLISHED $M5"; echo "ALL_DONE_SCREEN4F mixp $(date -u +%FT%TZ)"; exit 0
fi

# ============================================================================================================ run (H100)
dev_scores() {  # dev_scores SCOREDIR TAG CKPT-ARGS...: the three DEV panels (SCREEN4's invocation)
  sd=$1; tag=$2; shift 2
  score $sd $W/runs/sc_$tag/didev_$tag.npz "$@" --data $P/didev --splits test --n-perm 1 --exclude-identity --max-len 4096
  score $sd $W/runs/sc_$tag/egdev_$tag.npz "$@" --data $P/eg --splits val --n-perm 1 --exclude-identity --max-len 4096
  score $sd $W/runs/sc_$tag/caldev_$tag.npz "$@" --data $P/caldev --splits test --n-perm 1 --exclude-identity --max-len 4096; }
if [ "$MODE" = run ]; then
  ARM=${2:?run needs ARM}; SEED=${3:?run needs SEED}
  case "$ARM" in l) LR=1.5e-5; LAMP=0.3;; p) LR=5e-5; LAMP=1.0;; lp) LR=1.5e-5; LAMP=1.0;; *) echo "ARM must be l|p|lp (C is SCREEN4's)"; exit 1;; esac
  case "$SEED" in 0|1) ;; *) echo "SEED must be 0 or 1 (registered)"; exit 1;; esac
  need_h100; prep_data
  V=s4f${ARM}s$SEED
  have $RR/run_$V.done.json && { echo "ALREADY_DONE $V"; exit 0; }
  [ -n "$JEV_SLOT" ] || pkill -f "sglang serve"; sleep 10
  TR=$W/code_tr_$$; SC=$W/code_sc_$$; C=$W/code_s4f_$$; mkdir -p $TR $SC $C; prod_code $TR; prod_code $SC; s4_code $C
  aws s3 cp --only-show-errors $S3/fleetq/scripts/gemma_fast_h100.sh $W/runs/gemma_fast_s4f.sh && [ "$(md5sum < $W/runs/gemma_fast_s4f.sh | cut -c1-32)" = $FAST_MD5 ] || { echo "FAST_HELPER_MD5"; exit 1; }
  case $ARM in
    l)   getd results_screen4/data/s4_mixed_v1.tgz $MIXED_MD5 s4mixed s4_mixed_v1/arms/arms_mixed_manifest.json || exit 1
         X=$W/data/s4mixed/s4_mixed_v1/arms/arm_c;;
    p|lp) have $RR/data/s4f_mixp_v1.md5 || { echo "NOT_READY: mixp not done"; exit 1; }
         PM5=$(aws s3 cp --only-show-errors $RR/data/s4f_mixp_v1.md5 - | cut -c1-32)
         getd results_screen4f/data/s4f_mixp_v1.tgz $PM5 s4fmixp s4f_mixp_v1/arms/arms_mixed_manifest.json || exit 1
         X=$W/data/s4fmixp/s4f_mixp_v1/arms/arm_c
         python3 -c "import json,sys; m=json.load(open('$X/../arms_mixed_manifest.json')); sys.exit(0 if abs(m['lam_parent']-1.0)<1e-12 and abs(m['lam_teacher']-0.3)<1e-12 else 1)" \
           || { echo "MIXP_WEIGHTS_WRONG"; exit 1; };;
  esac
  DMD5=$(cd $X && find . -name '*.jsonl' | LC_ALL=C sort | xargs md5sum | md5sum | cut -c1-32)
  NU=$(env PYTHONPATH=$TR python3 - $X $TR/trainers/cache_features.py <<'PY'
import contextlib, importlib.util, io, sys
s = importlib.util.spec_from_file_location("cf", sys.argv[2]); cf = importlib.util.module_from_spec(s); s.loader.exec_module(cf)
with contextlib.redirect_stdout(io.StringIO()):
    us = [u for u in cf.iter_units(sys.argv[1], ["train"], set(), None) if not cf.is_noncommercial(u) and len(u["option_keys"]) <= 52]
print(len(us))
PY
)
  [ "$NU" = 12000 ] || { echo "ARM_UNITS $NU != 12000"; exit 1; }
  C99=$W/data/bbtune_gemma31it_99.pt; [ -s $C99 ] || aws s3 cp --only-show-errors $S3/results99/bbtune_gemma31it_99.pt $C99
  [ "$(md5sum < $C99 | cut -c1-32)" = $C99_MD5 ] || { echo "C99_MD5"; exit 1; }
  CK=$W/runs/bbtune_gemma31it_$V.pt; LOG=$W/runs/bbtune_gemma31it_$V.log; ID=$W/runs/identity_$V.txt
  TOL=0.02; [ $SEED = 1 ] && TOL=0.25
  if have $RR/bbtune_gemma31it_$V.pt.md5; then
    aws s3 cp --only-show-errors $RR/bbtune_gemma31it_$V.pt $CK; aws s3 cp --only-show-errors $RR/bbtune_gemma31it_$V.pt.md5 $CK.md5
    [ "$(md5sum < $CK | cut -c1-32)" = "$(cut -c1-32 $CK.md5)" ] || { echo "CKPT_MD5_MISMATCH (S3 copy)"; exit 1; }
    aws s3 cp --only-show-errors $RR/identity_$V.txt $ID; aws s3 cp --only-show-errors $RR/bbtune_gemma31it_$V.log $LOG
  else
    . $W/runs/gemma_fast_s4f.sh
    have $RR/identity_$V.txt && aws s3 cp --only-show-errors $RR/identity_$V.txt $ID
    env PYTHONPATH=$TR timeout 21600 python3 $TR/trainers/claude_lmtune.py --model $GM --exact-path $GEMMA_FAST_ARGS --gc-budget 24576 --lora-mlp \
      --data $X --init-lora $C99 --steps 1500 --batch-size 2 --grad-accum 4 --lr $LR --warmup 40 --max-len 2048 --eval-every 250 \
      --save-every 100 --rps-weight 1.0 $HEAD_FLAGS --seed $SEED --resume-s3 $RR/resume_$V --out $CK > $LOG.attempt 2>&1; rc=$?
    cat $LOG.attempt >> $LOG; aws s3 cp --only-show-errors $LOG $RR/bbtune_gemma31it_$V.log
    tu=$(grep -oE "^train units [0-9]+" $LOG.attempt | head -1 | awk '{print $3}')
    [ "$tu" = 12000 ] || { echo "TRAIN_UNITS_GATE_FAILED ($tu)"; exit 1; }
    if grep -q "^RESUMED at step" $LOG.attempt; then head -1 $ID 2>/dev/null | grep -q '^PASS' || { echo "IDENTITY_UNKNOWN_ON_RESUME"; exit 1; }
    else
      v0=$(grep -oE "step +0 +val soft-CE +[0-9.]+" $LOG.attempt | head -1 | awk '{print $NF}')
      python3 -c "import sys; sys.exit(0 if abs(float('${v0:-9}') - 0.4894) <= $TOL else 1)" && echo "PASS identity gate: step-0 val soft-CE $v0 vs (99) 0.4894 (tol $TOL)" > $ID \
        || echo "FAIL identity gate: step-0 val soft-CE $v0 vs (99) 0.4894 (tol $TOL)" > $ID
      put $ID identity_$V.txt; head -1 $ID | grep -q '^PASS' || { echo "IDENTITY_GATE_FAILED"; exit 1; }
    fi
    [ $rc -eq 0 ] && [ -s $CK ] || { echo "TRAIN_FAILED rc=$rc"; exit 1; }
    grep -q "LoRA initialised from $C99" $LOG || { echo "INIT_LORA_NOT_APPLIED"; exit 1; }
    grep -q "aux-vbll val head" $LOG || { echo "VBLL_HEAD_NOT_TRAINED"; exit 1; }
    grep -q "NC FILTER (train)" $LOG && { echo "NC_UNITS_IN_TRAIN"; exit 1; }
    md5sum < $CK | cut -c1-32 > $CK.md5; put $CK bbtune_gemma31it_$V.pt; put $CK.md5 bbtune_gemma31it_$V.pt.md5
  fi
  CKM=$(cut -c1-32 $CK.md5)
  CKH=$W/runs/bbtune_gemma31it_${V}_meanhead.pt
  env PYTHONPATH=$TR python3 $C/trainers/screen4_head.py meanhead --in $CK --out $CKH > $W/runs/meanhead_$V.log 2>&1 || { echo "MEANHEAD_FAILED"; exit 1; }
  mkdir -p $W/runs/sc_$V $W/runs/sc_${V}m
  dev_scores $SC $V --lora-ckpt $CK --aux-vbll --aux-vbll-head 0
  dev_scores $SC ${V}m --lora-ckpt $CKH --aux-vbll --aux-vbll-head 0
  python3 - $V $ARM $SEED "$GPU_NAME" "$CKM" "$DMD5" $LR $LAMP $LOG $ID $W/runs/sc_$V $W/runs/sc_${V}m > $W/runs/run_$V.done.json <<'PY' || exit 1
import glob, hashlib, json, os, re, socket, sys, time
V, arm, seed, gpu, ckm, dmd5, lr, lamp, log, idf, A, Am = sys.argv[1:13]
t = open(log).read()
tu = re.findall(r"^train units (\d+)", t, re.M)
v0 = re.findall(r"step-0 val soft-CE ([0-9.]+)", open(idf).read())
lrs = [float(x) for x in re.findall(r"^step +\d+ +loss [0-9.]+ +lr ([0-9.e+-]+)", t, re.M)]
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
print(json.dumps({"run": V, "arm": arm, "seed": int(seed), "gpu": gpu, "host": socket.gethostname(), "ckpt_md5": ckm, "data_md5": dmd5,
                  "preservation": True, "lr": float(lr), "lam_parent": float(lamp), "max_logged_lr": max(lrs) if lrs else None,
                  "aux_vbll": "aux-vbll val head" in t, "trainer_md5": "e2c6100a1c0651804613da6aaa477fb0",
                  "scorer_md5": "e0672bd0f515ccb17d97035c84c3d339", "steps": 1500, "train_units": int(tu[-1]) if tu else None,
                  "step0_val_softce": float(v0[0]) if v0 else None, "init_lora_applied": "LoRA initialised from" in t,
                  "resumed": "RESUMED at step" in t, "val_softce": [float(x) for x in re.findall(r"val soft-CE +([0-9.]+)", t)],
                  "dev_scores": {os.path.basename(p): md5(p) for p in sorted(glob.glob(A + "/*dev*.npz"))},
                  "meanhead_scores": {os.path.basename(p): md5(p) for p in sorted(glob.glob(Am + "/*.npz"))},
                  "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
PY
  python3 -c "import json,sys; r=json.load(open('$W/runs/run_$V.done.json')); m=r['max_logged_lr']; sys.exit(0 if m is not None and m <= r['lr'] * 1.0001 and m >= r['lr'] * 0.9 else 1)" \
    || { echo "LR_NOT_APPLIED (max logged lr vs $LR)"; exit 1; }
  put $W/runs/run_$V.done.json run_$V.done.json && echo "RUN_DONE $V ckpt $CKM"
  echo "ALL_DONE_SCREEN4F run $V $(date -u +%FT%TZ)"; exit 0
fi

# ============================================================================================================ read (CPU)
if [ "$MODE" = read ]; then
  prep_data
  have $RR/read/verdict.txt && { echo "ALREADY_READ: $(aws s3 cp --only-show-errors $RR/read/verdict.txt - | head -1)"; exit 0; }
  for a in l p lp; do for s in 0 1; do have $RR/run_s4f${a}s$s.done.json || { echo "NOT_READY: run $a $s"; exit 1; }; done; done
  C=$W/code_read4f_$$; mkdir -p $C; s4_code $C
  IN=$W/read4f_$$; mkdir -p $IN/sc
  for p in didev egdev caldev; do for t in I s4cs0 s4cs1 s4cs0m s4cs1m; do aws s3 cp --only-show-errors $RR4/score/${p}_$t.npz $IN/sc/ || exit 1; done; done
  aws s3 cp --only-show-errors --recursive $RR/score/ $IN/sc/ --exclude "*" --include "*dev_*.npz" || exit 1
  for s in 0 1; do aws s3 cp --only-show-errors $RR4/run_s4cs$s.done.json $IN/ || exit 1
    for a in l p lp; do aws s3 cp --only-show-errors $RR/run_s4f${a}s$s.done.json $IN/ || exit 1; done; done
  python3 - $IN $P > $IN/inputs.json <<'PY' || exit 1
import glob, json, sys
IN, P = sys.argv[1], sys.argv[2]
blk = {}
for f in glob.glob(P + "/didev/test/*.jsonl"):
    for l in open(f):
        r = json.loads(l); blk[r["source"]] = r["meta"]["screen4"]["block"]
panels = {"didev": {"data": P + "/didev", "splits": ["test"], "block_of_source": blk}, "egdev": {"data": P + "/eg", "splits": ["val"]},
          "caldev": {"data": P + "/caldev", "splits": ["test"]}}
f = lambda tag: {p: f"{IN}/sc/{p}_{tag}.npz" for p in panels}
files, mean, done = {"I": f("I")}, {}, {}
for cell, pre in (("c", "s4c"), ("l", "s4fl"), ("p", "s4fp"), ("lp", "s4flp")):
    for s in (0, 1):
        V = f"{pre}s{s}"
        files[f"{cell}{s}"], mean[f"{cell}{s}"] = f(V), f(V + "m")
        done[f"{cell}{s}"] = json.load(open(f"{IN}/run_{V}.done.json"))
print(json.dumps({"panels": panels, "files": files, "meanhead_files": mean, "done": done}))
PY
  ( cd $C && python3 analysis/screen4f_read.py --self-test ) > $IN/selftest.log 2>&1 || { echo "READ_SELF_TEST_FAILED"; exit 1; }
  ( cd $C && PYTHONPATH=$C:$C/trainers python3 analysis/screen4f_read.py --manifest $IN/inputs.json --base $C/analysis/screen4_read.py \
      --cf $C/trainers/cache_features.py --head $C/trainers/screen4_head.py --out $IN/verdict.txt --json $IN/verdict.json ) > $IN/read.log 2>&1 \
      || { echo "READ_FAILED"; tail -5 $IN/read.log; exit 1; }
  cat $IN/verdict.txt
  put $IN/verdict.json read/verdict.json; put $IN/inputs.json read/inputs.json; put $IN/read.log read/read.log
  put $IN/verdict.txt read/verdict.txt && echo "VERDICT_PUBLISHED $(head -1 $IN/verdict.txt | cut -d' ' -f1)"
  echo "ALL_DONE_SCREEN4F read $(date -u +%FT%TZ)"; exit 0
fi
echo "unknown mode $MODE"; exit 1
