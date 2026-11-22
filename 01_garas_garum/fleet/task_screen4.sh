#!/bin/sh
# task_screen4.sh prep | teach | mix | run a|b|c|d 0|1 [tryN] | ref [tryN] | read | memsmoke LEN [offload]
#   -- (SCREEN4) EVIDENCE-GROUNDED, CAPABILITY-PRESERVING continuation of Garas / (99) with a RESIDUAL VBLL decision head. DRAFT (not registered).
# Design tmp/screen4/design.md; pre-registration draft tmp/screen4/prereg_draft.md. Every arm is a (99)-init LoRA continuation with PC's
# recipe (claude_lmtune e2c6100a; lr 5e-5, warmup 40, cosine; --rps-weight 1.0; LoRA r16 attn + MLP; max-len 2048; global 8 = 2 x 4;
# 1,500 steps = 12,000 units = one pass; gemma fast path) PLUS the residual VBLL head grid (screen4_head.GRID; detached, so the LoRA
# trains as without it):
#   a  R + R2                                   (PC's arm r data; the broad-only control)
#   b  R + R2[2,000] + EG[4,000]                (+ evidence-grounded data: claim verification, support, relevance, answerability)
#   c  b with PRESERVATION targets              (replay -> (99)'s order-marginal prediction; EG -> stock Gemma's tempered prediction
#                                                where it agrees with gold; lam 0.3)
#   d  R + R2[500] + EG[4,000] + Bayes[1,500]   (c + the Bayesian-evidence families: 1,000 retrieval-style A2 + 500 numeric A1, AMEND A1)
# Hardware: SINGLE-GPU H100 for run / teach / ref (bounce elsewhere, p8 slots included); memsmoke ONLY as the p8 whole-node hold.
#   prep        CPU: tasksource shards at the pinned revision (HF), eval / corpus fingerprints and sealed + CONFIRMATION indexes (S3), PC's
#               replay -> EG pool, DI-family dev, caldev, arms a / b + teacher dirs -> results_screen4/data/s4_prep_v1.tgz (.md5 LAST)
#   teach       H100: (99) on the replay teacher dir, stock Gemma on the EG teacher dir (n-perm 4, max-len 2048) -> teacher npz files
#   mix         CPU: arms c / d with the preservation targets -> results_screen4/data/s4_mixed_v1.tgz (.md5 LAST)
#   run A S     H100: train (gates: 12,000 units; step-0 val soft-CE vs (99)'s 0.4894; init applied; no NC; the VBLL head trained) ->
#               mean-head checkpoint -> DEV scores (--aux-vbll; both checkpoints) -> run record LAST (exposed sets: never scored here)
#   ref         H100: Garas / (99) (native readout) and stock Gemma on the DEV panels (the reference + a descriptive anchor)
#   read        CPU: analysis/screen4_read.py on the DEV scores -> results_screen4/read/verdict.txt LAST
#   memsmoke L  p8 (8 x H100): trainers/screen4_memsmoke.py under torchrun, one length per call (timeout: an OOM is a result)
# Code: ONLY md5-pinned files from fleetq/code/pc/ (PC's production set), fleetq/code/ext5/ (the overlap tools) and fleetq/code/screen4/.
MODE=${1:?usage: task_screen4.sh prep | teach | mix | run A S | ref | read | memsmoke LEN [offload]}
J=${JEV_J:-/data/shared/jev}; D=${JEV_D:-/data/local}; FL=${JEV_FLEET:-/data/shared/fleet}
S3=s3://YOUR_BUCKET/jev-race; RR=$S3/results_screen4; CB=$S3/fleetq/code; W=$D/screen4
[ -f $FL/env.sh ] && . $FL/env.sh
export HF_HOME=${JEV_HF:-$D/hf} HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $W/data $W/runs || exit 1
GM=google/gemma-4-31B-it
TS_REV=f8ff7e81ed820eeac1ad7afd63449fe99c8a76c3
C99_MD5=a4ee3b50c1a578c6536fc3727170fd0f         # results99/bbtune_gemma31it_99.pt (Garas / (99))
RP_MD5=67359006eb3f51aa27dd210df90c0c23          # results_pc/data/pc_replay_v1.tgz (R, R2, mix2 val)
FAST_MD5=433ac777db0a365121589ea59b12be58        # fleetq/scripts/gemma_fast_h100.sh
CONFIRM_KEY=sealed/confirm_2610/index/confirm_2610_exclusion_index.json.gz; CONFIRM_MD5=06f9215999e6a51991a94120b1744671   # AMEND A1: CONFIRM-2610 (ids, state_sha1, clusters)
BF_KEY=fleetq/data/bayes_families_v1.tgz; BF_MD5=395c88a148614a4cc5cabaa453e9d867   # AMEND A1: the Bayesian-evidence families v1 (arm d)
S4_BUILD=d1a0ed10f963be7c01619d49684521ed                   # fleetq/code/screen4/build_screen4.py
S4_LOSS=0388f88d7915b022b0e44634d4a581d7                     # fleetq/code/screen4/trainers/screen4_loss.py
S4_HEAD=0df767bcd3dbd83d8fd7a57d0effed6c                     # fleetq/code/screen4/trainers/screen4_head.py
S4_READ=607234f61d30baea9f462e80a5620b45                     # fleetq/code/screen4/analysis/screen4_read.py
S4_SMOKE=4d2e68b7e4b85a8341b972c2d134b697                # fleetq/code/screen4/trainers/screen4_memsmoke.py
S4_PINS=e4d0bf710fd0a229032c5a5c2a17158a                          # fleetq/code/screen4/ts_pins_f8ff7e81.json (tasksource shard sha256s)
HEAD_FLAGS="--aux-vbll --aux-vbll-prior 1 --aux-vbll-kl 0,1,10000,1000000 --aux-vbll-cov diag,full --aux-vbll-dim 128 --aux-vbll-samples 64 --aux-vbll-lr 0.001"
have() { aws s3 ls $1 > /dev/null 2>&1; }
fetch() {   # fetch DIR KEY MD5: fleetq/code/KEY -> DIR/<KEY without its first component>, md5-checked
  dst=$1/${2#*/}; mkdir -p $(dirname $dst) && aws s3 cp --only-show-errors $CB/$2 $dst || { echo "NO_CODE $2"; exit 1; }
  got=$(md5sum < $dst | cut -c1-32); [ "$got" = "$3" ] || { echo "CODE_MD5 $2 $got != $3"; exit 1; }; }
getd() {    # getd S3KEY MD5 NAME CHECKPATH: fetch + md5 + extract into $W/data/NAME, once
  [ -f $W/data/$3/.ok ] && [ "$(cat $W/data/$3/.ok)" = $2 ] && [ -e $W/data/$3/$4 ] && return 0
  aws s3 cp --only-show-errors $S3/$1 $W/data/$3.tgz || { echo "NO_DATA $1"; return 1; }
  [ "$(md5sum < $W/data/$3.tgz | cut -c1-32)" = $2 ] || { echo "DATA_MD5 $1"; return 1; }
  x=$W/data/.x_$3_$$; mkdir -p $x && tar xzf $W/data/$3.tgz -C $x && [ -e $x/$4 ] || { echo "UNTAR_FAIL $1 (no $4)"; return 1; }
  [ -e $W/data/$3 ] && mv $W/data/$3 $W/data/$3.stale.$(date +%s)
  mv $x $W/data/$3 && echo $2 > $W/data/$3/.ok && rm -f $W/data/$3.tgz; }
put() {     # put LOCAL KEY: upload + read-back md5
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
s4_code() {     # the screen's own files + the overlap tools into $1
  fetch $1 screen4/build_screen4.py $S4_BUILD; fetch $1 screen4/trainers/screen4_loss.py $S4_LOSS
  fetch $1 screen4/trainers/screen4_head.py $S4_HEAD; fetch $1 screen4/analysis/screen4_read.py $S4_READ
  fetch $1 screen4/ts_pins_f8ff7e81.json $S4_PINS
  for p in "ext5/build_jev_ext2.py 8ac5de094bdc7747e47f229e9b968176" "ext5/build_jev_ext3.py 61585da5ee80bc2d3267aa0efbd952a2" \
           "ext5/build_jev_ext.py 41271b13bca9c87550cd791ea16f0f07" "ext5/build_jev_data.py c9d4570b000aa39b6aca792947c6a145" \
           "ext5/analysis/claude_overlap.py 0bcf9b1ff942fd62808f646627da8e59" "ext5/trainers/convert_extevals.py 23cce7db3ad527fe57195a1191bb5558" \
           "pc/trainers/cache_features.py 49613a40faed8cd612175e81c3f3925f"; do set -- $1 $p; fetch $1 $2 $3; done; }
score() {       # score DIR OUT ARGS...: the pinned production scorer (PYTHONPATH = its dir only); HAVE-skips from S3
  sd=$1; out=$2; shift 2
  if have $RR/score/$(basename $out); then aws s3 cp --only-show-errors $RR/score/$(basename $out) $out; echo "HAVE $(basename $out)"; return 0; fi
  env PYTHONPATH=$sd python3 $sd/trainers/claude_lmscore.py --model $GM --exact-path --out $out "$@" > $W/runs/$(basename $out .npz).log 2>&1
  rc=$?; echo "SCORE $(basename $out) EXIT=$rc $(grep '^wrote' $W/runs/$(basename $out .npz).log | cut -c1-140)"
  [ $rc -eq 0 ] && [ -s $out ] || { echo "SCORE_FAILED $(basename $out)"; exit 1; }
  put $out score/$(basename $out); put $W/runs/$(basename $out .npz).log score/$(basename $out .npz).log; }
echo "== (SCREEN4 $*) start $(date -u +%FT%TZ) on $(hostname)"

# ============================================================================================================ prep (CPU)
if [ "$MODE" = prep ]; then
  have $RR/data/s4_prep_v1.md5 && { echo "ALREADY_PREPPED $(aws s3 cp --only-show-errors $RR/data/s4_prep_v1.md5 -)"; exit 0; }
  [ "$CONFIRM_KEY" = TBD ] && { echo "NO_CONFIRMATION_INDEX: the fresh confirmation set must be sealed and its exclusion index pinned (amendment) before any screen data exists"; exit 1; }
  C=$W/code_prep_$$; mkdir -p $C; s4_code $C
  getd results_pc/data/pc_replay_v1.tgz $RP_MD5 pcreplay pc_replay_v1/manifest.json || exit 1
  getd fleetq/data/decision_index_v021.tgz d023674a8e8c654e650cc5d34a0e7079 di . || exit 1
  getd data/gde1_v3_gated.tgz 7bd4f39c86e67dfad7206494fa23d094 gde1v3 . || exit 1
  getd data/rw_eval_v1.tgz a700f892335fd2246c2f78abf124083b rw_eval physics_jev/eval || exit 1
  getd fleetq/data/bayes_consistency_v2.tgz 90416ce3df51800b6a2be6daa31479e6 bc_v2 bayes_consistency_v2/test || exit 1
  getd fleetq/data/numeric_v2.tgz 5b7c4cb5e1eb168ee1f4dbfadcefceb0 numeric_v2 numeric_v2/test || exit 1
  getd data/jev_data_ref.tgz 8459f2d4c09ad25567ffe09665e9789f jev_data jev_data/train || exit 1
  getd data/jev_ext_strict.tgz a3369fcde32a97039d1ccfe7734a017a jev_ext_strict jev_ext_strict/train || exit 1
  getd data/jev_ext2.tgz ca43f76a09b4d0071f7d23c105aa714d jev_ext2 jev_ext2/train || exit 1
  mkdir -p $W/data/excl
  for p in "rd_exclusion_index.json.gz 28a13a92edc55b38da3db5d4f5f93ef5" "hl_exclusion_index.json.gz f047836e7cf6be1f9dcde484d1318676"; do
    set -- $p; k=$(aws s3 ls --recursive $S3/sealed/final_v1/ | awk '{print $4}' | grep "/$1$" | head -1)
    [ -n "$k" ] && aws s3 cp --only-show-errors s3://YOUR_BUCKET/$k $W/data/excl/$1 && [ "$(md5sum < $W/data/excl/$1 | cut -c1-32)" = $2 ] \
      || { echo "NO_SEALED_INDEX $1"; exit 1; }; done
  aws s3 cp --only-show-errors $S3/$CONFIRM_KEY $W/data/excl/confirm_index.json.gz && [ "$(md5sum < $W/data/excl/confirm_index.json.gz | cut -c1-32)" = $CONFIRM_MD5 ] \
    || { echo "CONFIRM_INDEX_MD5"; exit 1; }
  BFARG=""
  if [ "$BF_KEY" != TBD ]; then getd $BF_KEY $BF_MD5 bf bayes_families_v1/train || exit 1; BFARG="--bayes $W/data/bf/bayes_families_v1"; fi
  TS=$W/data/ts_$TS_REV; mkdir -p $TS
  python3 - $TS $C/ts_pins_f8ff7e81.json $TS_REV <<'PY' || { echo "TS_FETCH_FAILED"; exit 1; }
import hashlib, json, sys
from huggingface_hub import hf_hub_download
d, pins, rev = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3]
for rel, want in sorted(pins.items()):
    p = hf_hub_download("tasksource/tasksource-jev-typed-decisions", rel, repo_type="dataset", revision=rev, local_dir=d)
    got = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if got != want:
        sys.exit(f"{rel}: sha256 {got} != pin {want}")
print("TS_SHARDS_OK", len(pins))
PY
  N2=$W/data/numeric_v2/numeric_v2; O=$W/s4_prep_v1; [ -e $O ] && mv $O $W/s4_prep_v1.stale.$(date +%s); mkdir -p $O
  EV="--eval di021=$W/data/di --eval gde1v3=$W/data/gde1v3 --eval rw_eval_v1=$W/data/rw_eval --eval bc_v2=$W/data/bc_v2/bayes_consistency_v2/test --eval numeric_v2_test=$N2/test --eval numeric_v2_val=$N2/val --fetch-evals"
  CO="--corpus jev_data=$W/data/jev_data/jev_data --corpus jev_ext_strict=$W/data/jev_ext_strict/jev_ext_strict --corpus jev_ext2=$W/data/jev_ext2/jev_ext2"
  IX="--sealed $W/data/excl/rd_exclusion_index.json.gz --sealed $W/data/excl/hl_exclusion_index.json.gz --confirm-index $W/data/excl/confirm_index.json.gz"
  ( cd $C && PYTHONPATH=$C:$C/analysis:$C/trainers python3 build_screen4.py --self-test ) > $W/runs/prep_selftest.log 2>&1 || { echo "BUILD_SELF_TEST_FAILED"; exit 1; }
  for step in eg didev; do
    ( cd $C && PYTHONPATH=$C:$C/analysis:$C/trainers nice -n 10 python3 build_screen4.py $step --ts $TS --shard-pins $C/ts_pins_f8ff7e81.json \
        $EV $CO $IX --out $O/$step ) > $W/runs/prep_$step.log 2>&1 || { echo "PREP_${step}_FAILED: $(tail -3 $W/runs/prep_$step.log)"; exit 1; }
    tail -1 $W/runs/prep_$step.log | cut -c1-300
  done
  ( cd $C && PYTHONPATH=$C:$C/analysis:$C/trainers python3 build_screen4.py caldev --replay $W/data/pcreplay/pc_replay_v1 --out $O/caldev ) \
    > $W/runs/prep_caldev.log 2>&1 || { echo "PREP_CALDEV_FAILED"; exit 1; }
  ( cd $C && PYTHONPATH=$C:$C/analysis:$C/trainers python3 build_screen4.py arms --phase plain --replay $W/data/pcreplay/pc_replay_v1 --eg $O/eg \
      $BFARG $IX --dev $O/didev --dev $O/caldev --out $O/arms ) > $W/runs/prep_arms.log 2>&1 || { echo "PREP_ARMS_FAILED: $(tail -3 $W/runs/prep_arms.log)"; exit 1; }
  tail -1 $W/runs/prep_arms.log | cut -c1-300
  COPYFILE_DISABLE=1 tar czf $W/s4_prep_v1.tgz -C $W s4_prep_v1 || exit 1
  M5=$(md5sum < $W/s4_prep_v1.tgz | cut -c1-32); put $W/s4_prep_v1.tgz data/s4_prep_v1.tgz
  for f in eg/manifest.json didev/manifest.json arms/arms_plain_manifest.json; do put $O/$f data/s4_prep_$(echo $f | tr / _); done
  echo "$M5  s4_prep_v1.tgz" > $W/s4_prep_v1.md5; put $W/s4_prep_v1.md5 data/s4_prep_v1.md5
  echo "PREP_PUBLISHED $M5"; echo "ALL_DONE_SCREEN4 prep $(date -u +%FT%TZ)"; exit 0
fi

prep_data() {   # the published prep tarball (by its md5), extracted once
  have $RR/data/s4_prep_v1.md5 || { echo "NOT_READY: results_screen4/data/s4_prep_v1.md5 (prep not done)"; exit 1; }
  PM5=$(aws s3 cp --only-show-errors $RR/data/s4_prep_v1.md5 - | cut -c1-32)
  getd results_screen4/data/s4_prep_v1.tgz $PM5 s4prep s4_prep_v1/arms/arms_plain_manifest.json || exit 1
  P=$W/data/s4prep/s4_prep_v1; }

# ============================================================================================================ teach (H100)
if [ "$MODE" = teach ]; then
  need_h100; prep_data
  have $RR/teach/teach.done.json && { echo "ALREADY_TAUGHT"; exit 0; }
  [ -n "$JEV_SLOT" ] || pkill -f "sglang serve"; sleep 10
  SC=$W/code_sc_$$; mkdir -p $SC; prod_code $SC
  C99=$W/data/bbtune_gemma31it_99.pt; [ -s $C99 ] || aws s3 cp --only-show-errors $S3/results99/bbtune_gemma31it_99.pt $C99
  [ "$(md5sum < $C99 | cut -c1-32)" = $C99_MD5 ] || { echo "C99_MD5"; exit 1; }
  T=$W/runs/teach; mkdir -p $T
  RR0=$RR; RR=$RR/teach
  score $SC $T/parent_99.npz --lora-ckpt $C99 --data $P/arms/teach_replay --splits test --n-perm 4 --max-len 2048
  score $SC $T/teacher_gemma.npz --data $P/arms/teach_eg --splits test --n-perm 4 --max-len 2048
  score $SC $T/teacher_gemma_dev.npz --data $P/arms/teach_eg --splits dev --n-perm 4 --max-len 2048
  RR=$RR0
  python3 - $T > $T/teach.done.json <<'PY' || exit 1
import glob, hashlib, json, os, sys
print(json.dumps({os.path.basename(p): hashlib.md5(open(p, "rb").read()).hexdigest() for p in sorted(glob.glob(sys.argv[1] + "/*.npz"))}))
PY
  put $T/teach.done.json teach/teach.done.json; echo "ALL_DONE_SCREEN4 teach $(date -u +%FT%TZ)"; exit 0
fi

# ============================================================================================================ mix (CPU)
if [ "$MODE" = mix ]; then
  prep_data
  have $RR/data/s4_mixed_v1.md5 && { echo "ALREADY_MIXED"; exit 0; }
  have $RR/teach/teach.done.json || { echo "NOT_READY: teach not done"; exit 1; }
  C=$W/code_mix_$$; mkdir -p $C; s4_code $C
  T=$W/runs/teach_in; mkdir -p $T
  aws s3 cp --only-show-errors $RR/teach/teach.done.json $T/
  for f in parent_99 teacher_gemma teacher_gemma_dev; do aws s3 cp --only-show-errors $RR/teach/score/$f.npz $T/$f.npz || exit 1
    [ "$(md5sum < $T/$f.npz | cut -c1-32)" = "$(python3 -c "import json,sys; print(json.load(open('$T/teach.done.json'))['$f.npz'])")" ] || { echo "TEACHER_MD5 $f"; exit 1; }; done
  O=$W/s4_mixed_v1; [ -e $O ] && mv $O $W/s4_mixed_v1.stale.$(date +%s); mkdir -p $O; cp -r $P/arms $O/arms
  ( cd $C && PYTHONPATH=$C:$C/analysis:$C/trainers python3 build_screen4.py arms --phase mixed --replay $W/data/pcreplay/pc_replay_v1 --eg $P/eg \
      --parent-npz $T/parent_99.npz --teacher-npz $T/teacher_gemma.npz --teacher-dev-npz $T/teacher_gemma_dev.npz --dev $P/didev --dev $P/caldev \
      --out $O/arms ) > $W/runs/mix.log 2>&1 || { echo "MIX_FAILED: $(tail -3 $W/runs/mix.log)"; exit 1; }
  tail -1 $W/runs/mix.log | cut -c1-300
  COPYFILE_DISABLE=1 tar czf $W/s4_mixed_v1.tgz -C $W s4_mixed_v1 || exit 1
  M5=$(md5sum < $W/s4_mixed_v1.tgz | cut -c1-32); put $W/s4_mixed_v1.tgz data/s4_mixed_v1.tgz; put $O/arms/arms_mixed_manifest.json data/s4_arms_mixed_manifest.json
  echo "$M5  s4_mixed_v1.tgz" > $W/s4_mixed_v1.md5; put $W/s4_mixed_v1.md5 data/s4_mixed_v1.md5
  echo "MIXED_PUBLISHED $M5"; echo "ALL_DONE_SCREEN4 mix $(date -u +%FT%TZ)"; exit 0
fi

# ============================================================================================================ run (H100)
dev_scores() {  # dev_scores SCOREDIR TAG CKPT-ARGS...: the three DEV panels
  sd=$1; tag=$2; shift 2
  score $sd $W/runs/sc_$tag/didev_$tag.npz "$@" --data $P/didev --splits test --n-perm 1 --exclude-identity --max-len 4096
  score $sd $W/runs/sc_$tag/egdev_$tag.npz "$@" --data $P/eg --splits val --n-perm 1 --exclude-identity --max-len 4096
  score $sd $W/runs/sc_$tag/caldev_$tag.npz "$@" --data $P/caldev --splits test --n-perm 1 --exclude-identity --max-len 4096; }
if [ "$MODE" = run ]; then
  ARM=${2:?run needs ARM}; SEED=${3:?run needs SEED}
  case "$ARM" in a|b|c) ;; d) [ "$BF_KEY" = TBD ] && { echo "ARM_D_NOT_LICENSED: the Bayesian-evidence families are not pinned (amendment)"; exit 1; };; *) echo "ARM must be a|b|c|d"; exit 1;; esac
  case "$SEED" in 0|1) ;; *) echo "SEED must be 0 or 1 (registered); a third seed needs an amendment"; exit 1;; esac
  need_h100; prep_data
  V=s4${ARM}s$SEED
  have $RR/run_$V.done.json && { echo "ALREADY_DONE $V"; exit 0; }
  [ -n "$JEV_SLOT" ] || pkill -f "sglang serve"; sleep 10
  TR=$W/code_tr_$$; SC=$W/code_sc_$$; C=$W/code_s4_$$; mkdir -p $TR $SC $C; prod_code $TR; prod_code $SC; s4_code $C
  aws s3 cp --only-show-errors $S3/fleetq/scripts/gemma_fast_h100.sh $W/runs/gemma_fast_s4.sh && [ "$(md5sum < $W/runs/gemma_fast_s4.sh | cut -c1-32)" = $FAST_MD5 ] || { echo "FAST_HELPER_MD5"; exit 1; }
  case $ARM in
    a|b) X=$P/arms/arm_$ARM; PRES=0;;
    c|d) have $RR/data/s4_mixed_v1.md5 || { echo "NOT_READY: mix not done"; exit 1; }
         MM5=$(aws s3 cp --only-show-errors $RR/data/s4_mixed_v1.md5 - | cut -c1-32)
         getd results_screen4/data/s4_mixed_v1.tgz $MM5 s4mixed s4_mixed_v1/arms/arms_mixed_manifest.json || exit 1
         X=$W/data/s4mixed/s4_mixed_v1/arms/arm_$ARM; PRES=1;;
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
    . $W/runs/gemma_fast_s4.sh
    have $RR/identity_$V.txt && aws s3 cp --only-show-errors $RR/identity_$V.txt $ID
    env PYTHONPATH=$TR timeout 21600 python3 $TR/trainers/claude_lmtune.py --model $GM --exact-path $GEMMA_FAST_ARGS --gc-budget 24576 --lora-mlp \
      --data $X --init-lora $C99 --steps 1500 --batch-size 2 --grad-accum 4 --lr 5e-5 --warmup 40 --max-len 2048 --eval-every 250 \
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
  python3 - $V $ARM $SEED "$GPU_NAME" "$CKM" "$DMD5" $PRES $LOG $ID $W/runs/sc_$V $W/runs/sc_${V}m > $W/runs/run_$V.done.json <<'PY' || exit 1
import glob, hashlib, json, os, re, socket, sys, time
V, arm, seed, gpu, ckm, dmd5, pres, log, idf, A, Am = sys.argv[1:12]
t = open(log).read()
tu = re.findall(r"^train units (\d+)", t, re.M)
v0 = re.findall(r"step-0 val soft-CE ([0-9.]+)", open(idf).read())
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
print(json.dumps({"run": V, "arm": arm, "seed": int(seed), "gpu": gpu, "host": socket.gethostname(), "ckpt_md5": ckm, "data_md5": dmd5,
                  "preservation": pres == "1", "aux_vbll": "aux-vbll val head" in t, "trainer_md5": "e2c6100a1c0651804613da6aaa477fb0",
                  "scorer_md5": "e0672bd0f515ccb17d97035c84c3d339", "steps": 1500, "train_units": int(tu[-1]) if tu else None,
                  "step0_val_softce": float(v0[0]) if v0 else None, "init_lora_applied": "LoRA initialised from" in t,
                  "resumed": "RESUMED at step" in t, "val_softce": [float(x) for x in re.findall(r"val soft-CE +([0-9.]+)", t)],
                  "dev_scores": {os.path.basename(p): md5(p) for p in sorted(glob.glob(A + "/*dev*.npz"))},
                  "meanhead_scores": {os.path.basename(p): md5(p) for p in sorted(glob.glob(Am + "/*.npz"))},
                  "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
PY
  put $W/runs/run_$V.done.json run_$V.done.json && echo "RUN_DONE $V ckpt $CKM"
  echo "ALL_DONE_SCREEN4 run $V $(date -u +%FT%TZ)"; exit 0
fi

# ============================================================================================================ ref (H100)
if [ "$MODE" = ref ]; then
  need_h100; prep_data
  have $RR/ref.done.json && { echo "ALREADY_DONE ref"; exit 0; }
  [ -n "$JEV_SLOT" ] || pkill -f "sglang serve"; sleep 10
  SC=$W/code_sc_$$; mkdir -p $SC; prod_code $SC
  C99=$W/data/bbtune_gemma31it_99.pt; [ -s $C99 ] || aws s3 cp --only-show-errors $S3/results99/bbtune_gemma31it_99.pt $C99
  [ "$(md5sum < $C99 | cut -c1-32)" = $C99_MD5 ] || { echo "C99_MD5"; exit 1; }
  mkdir -p $W/runs/sc_I $W/runs/sc_G
  dev_scores $SC I --lora-ckpt $C99
  dev_scores $SC G
  python3 -c "import glob,hashlib,json,os; print(json.dumps({os.path.basename(p): hashlib.md5(open(p,'rb').read()).hexdigest() for p in sorted(glob.glob('$W/runs/sc_[IG]/*.npz'))}))" > $W/runs/ref.done.json
  put $W/runs/ref.done.json ref.done.json; echo "ALL_DONE_SCREEN4 ref $(date -u +%FT%TZ)"; exit 0
fi

# ============================================================================================================ read (CPU)
if [ "$MODE" = read ]; then
  prep_data
  have $RR/read/verdict.txt && { echo "ALREADY_READ: $(aws s3 cp --only-show-errors $RR/read/verdict.txt - | head -1)"; exit 0; }
  ARMS_READ=${S4_ARMS:-"a b c d"}
  have $RR/ref.done.json || { echo "NOT_READY: ref"; exit 1; }
  for a in $ARMS_READ; do for s in 0 1; do have $RR/run_s4${a}s$s.done.json || { echo "NOT_READY: run $a $s"; exit 1; }; done; done
  C=$W/code_read_$$; mkdir -p $C; s4_code $C
  IN=$W/read_$$; mkdir -p $IN/sc
  aws s3 cp --only-show-errors --recursive $RR/score/ $IN/sc/ --exclude "*" --include "*dev_*.npz" || exit 1
  for a in $ARMS_READ; do for s in 0 1; do aws s3 cp --only-show-errors $RR/run_s4${a}s$s.done.json $IN/; done; done
  python3 - $IN $P "$ARMS_READ" > $IN/inputs.json <<'PY' || exit 1
import glob, json, os, sys
IN, P, arms = sys.argv[1], sys.argv[2], sys.argv[3].split()
blk = {}
for f in glob.glob(P + "/didev/test/*.jsonl"):
    for l in open(f):
        r = json.loads(l); blk[r["source"]] = r["meta"]["screen4"]["block"]
panels = {"didev": {"data": P + "/didev", "splits": ["test"], "block_of_source": blk}, "egdev": {"data": P + "/eg", "splits": ["val"]},
          "caldev": {"data": P + "/caldev", "splits": ["test"]}}
f = lambda tag: {p: f"{IN}/sc/{p}_{tag}.npz" for p in panels}
files, mean, done = {"I": f("I")}, {}, {}
for a in arms:
    for s in (0, 1):
        V = f"s4{a}s{s}"
        files[f"{a}{s}"], mean[f"{a}{s}"] = f(V), f(V + "m")
        done[f"{a}{s}"] = json.load(open(f"{IN}/run_{V}.done.json"))
print(json.dumps({"panels": panels, "files": files, "meanhead_files": mean, "done": done}))
PY
  ( cd $C && python3 analysis/screen4_read.py --self-test ) > $IN/selftest.log 2>&1 || { echo "READ_SELF_TEST_FAILED"; exit 1; }
  ( cd $C && PYTHONPATH=$C:$C/trainers python3 analysis/screen4_read.py --manifest $IN/inputs.json --cf $C/trainers/cache_features.py \
      --head $C/trainers/screen4_head.py --out $IN/verdict.txt --json $IN/verdict.json ) > $IN/read.log 2>&1 || { echo "READ_FAILED"; tail -5 $IN/read.log; exit 1; }
  cat $IN/verdict.txt
  put $IN/verdict.json read/verdict.json; put $IN/inputs.json read/inputs.json; put $IN/read.log read/read.log
  put $IN/verdict.txt read/verdict.txt && echo "VERDICT_PUBLISHED $(head -1 $IN/verdict.txt | cut -d' ' -f1)"
  echo "ALL_DONE_SCREEN4 read $(date -u +%FT%TZ)"; exit 0
fi

# ============================================================================================================ memsmoke (p8)
if [ "$MODE" = memsmoke ]; then
  LEN=${2:?memsmoke needs LEN}; OFF=${3:-}
  NG=${JEV_NODE_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)}
  [ "$NG" -ge 8 ] || { echo "memsmoke needs the whole 8-GPU node (got $NG): queue it only through fleetq/hold/p8"; exit 1; }
  case "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" in *H100*) ;; *) echo "not H100"; exit 1;; esac
  TAG=L$LEN${OFF:+_offload}; have $RR/memsmoke/$TAG.json && { echo "ALREADY_DONE memsmoke $TAG"; exit 0; }
  C=$W/code_ms_$$; mkdir -p $C; fetch $C screen4/trainers/screen4_memsmoke.py $S4_SMOKE
  OUT=$W/runs/memsmoke_$TAG.json; rm -f $OUT
  timeout 2400 torchrun --standalone --nproc-per-node 8 $C/trainers/screen4_memsmoke.py --model $GM --length $LEN ${OFF:+--offload-activations} \
    --out $OUT > $W/runs/memsmoke_$TAG.log 2>&1; rc=$?
  if [ ! -s $OUT ]; then
    python3 -c "import json; print(json.dumps({'length': $LEN, 'offload': '$OFF' != '', 'oom': True, 'exit': $rc, 'note': 'no record: OOM / timeout / crash on some rank (log uploaded)'}))" > $OUT
  fi
  put $W/runs/memsmoke_$TAG.log memsmoke/$TAG.log; put $OUT memsmoke/$TAG.json
  cat $OUT; echo "ALL_DONE_SCREEN4 memsmoke $TAG $(date -u +%FT%TZ)"; exit 0
fi
echo "unknown MODE $MODE"; exit 1
