#!/bin/bash
# FS1 step-2,400 export + head for the prefix read (TB1 fork, 2026-11-12; prereg AGENT_NOTES "PREREG FS1-S2400-PREFIX") = fs1_build.sh
# with the weights source esbar_fs1/export_s2400 (fork E's from-stock pilot, leg 1) on p8: venv, pull + md5-verify the export against its
# READBACK_MD5.txt, head refit with the v7.1 stage-2 procedure, head -> S3 with read-back, then exec fs1_gate.sh. Idempotent steps.
set -u
unset ASCALE_RUN_ID
export AWS_DEFAULT_REGION=us-east-2 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_AVOID_RECORD_STREAMS=1 NCCL_RUNTIME_CONNECT=0
D=/data/local; B=s3://YOUR_BUCKET/jev-race; E=$B/engineering/esbar; S3H=$E/runs/esbar_fs1/export_s2400
O=$E/runs/esbar_fs1/public_di_s2400
W=$D/fs1pub; X=$W/fs1_export
V=$D/venv_t2141; PY=$V/bin/python; FF=4e5b8440e46f9e777b06c6b11c1a42f7
C=$D/esbar_code_v2e; MAN=$D/esbar_v5/manifests; FV=$W/features_fs1; HD=$W/head_fs1
mkdir -p $W/logs $W/src $W/tmp $FV $HD
L=$W/logs/build.log
log() { echo "[fs1 $(date -u +%FT%TZ)] $*" | tee -a $L; }
die() { log "DIE $*"; aws s3 cp --only-show-errors $L $O/logs/build.log 2>/dev/null; exit 1; }
md5() { md5sum < "$1" | cut -c1-32; }
fetch() { [ -s $2 ] || { mkdir -p $(dirname $2) && aws s3 cp --only-show-errors $1 $2; } || die "NO_INPUT $1"
          [ -z "${3:-}" ] || [ "$(md5sum < $2 | cut -c1-${#3})" = "$3" ] || die "MD5 $1 $(md5sum < $2 | cut -c1-8) != $3"; }
ident() { (cd $1 && ls model*.safetensors config.json | sort | xargs -P 13 -n 1 sha256sum | sort -k2) | sha256sum | cut -c1-64; }
log "start host $(hostname)"
# ---- 0. venv (pub71_driver step 1, functional probe)
probe() { $V/bin/python -c "import torch, transformers, safetensors; assert torch.__version__ == '2.14.1+cu130', torch.__version__; torch.zeros(1).cuda(); print('PROBE torch', torch.__version__, 'transformers', transformers.__version__, 'gpus', torch.cuda.device_count())"; }
if ! probe >> $L 2>&1; then
  fetch $B/fleetq/code/fullft43/env/torch2141_lock.txt $W/src/lock.txt $FF
  python3 -m venv --clear --system-site-packages $V && $V/bin/pip install -q --no-deps -r $W/src/lock.txt \
    --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple > $W/logs/venv.log 2>&1 || die VENV
  grep -vE '^(-e |#|$)' $W/src/lock.txt > $W/src/constraints.txt
  $V/bin/pip install -q "torch==2.14.1+cu130" "transformers==5.12.1" "accelerate==1.10.1" -c $W/src/constraints.txt \
    --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple >> $W/logs/venv.log 2>&1 || die VENV_DEPS
  probe >> $L 2>&1 || die VENV_PROBE
  $V/bin/pip freeze --local > $W/logs/pip_freeze.txt
  echo "$FF $(date -u +%FT%TZ) freeze=$(md5 $W/logs/pip_freeze.txt)" > $V/.ok
  aws s3 cp --only-show-errors $W/logs/pip_freeze.txt $O/env/pip_freeze_$(hostname).txt || true
fi
log "env ok: $(probe 2>&1 | tail -1)"
export PATH=$V/bin:$PATH
# ---- 1. wait for export_s2400 (READBACK_MD5.txt = complete), pull, verify every pulled file against it
if [ ! -f $X/.pulled ]; then
  n=0; until aws s3 ls $S3H/READBACK_MD5.txt > /dev/null 2>&1; do
    [ $((n % 10)) = 0 ] && log "waiting for export_s2400/READBACK_MD5.txt"; n=$((n + 1)); [ $n -gt 360 ] && die "NO_EXPORT_FULL after 6 h"; sleep 60; done
  aws s3 cp --only-show-errors $S3H/READBACK_MD5.txt $W/export_s2400_READBACK_MD5.txt || die READBACK_FETCH
  grep -q '^BAD ' $W/export_s2400_READBACK_MD5.txt && die "READBACK_HAS_BAD $(grep -c '^BAD ' $W/export_s2400_READBACK_MD5.txt)"
  log "export_s2400 complete: $(grep -c '^OK ' $W/export_s2400_READBACK_MD5.txt) files OK in READBACK_MD5.txt"
  mkdir -p $X && aws s3 sync --only-show-errors --exclude 'vbll_head*' --exclude 'READBACK_MD5.txt' $S3H/ $X/ || die SYNC_FULL
  [ "$(ls $X/model-*.safetensors | wc -l)" = 12 ] || die FULL_SHARDS
  bad=0; while read st f m; do case $f in vbll_head*) continue;; esac
    [ -f $X/$f ] || { log "MISSING $f"; bad=$((bad + 1)); continue; }
    [ "$(md5 $X/$f)" = "$m" ] || { log "MD5_MISMATCH $f"; bad=$((bad + 1)); }; done < <(grep '^OK ' $W/export_s2400_READBACK_MD5.txt)
  [ $bad = 0 ] || die "FULL_VERIFY bad=$bad"
  extra=$(cd $X && ls | grep -vxF -f <(awk '{print $2}' $W/export_s2400_READBACK_MD5.txt) | tr '\n' ' '); [ -z "$extra" ] || log "files not in readback list: $extra"
  touch $X/.pulled
fi
XID=$(ident $X); echo $XID > $W/fs1_identity.txt
log "FS1 s2400 export pulled + md5-verified; identity $XID; config trainer $(python3 -c "import json; c=json.load(open('$X/config.json')); print(c.get('jev_requires_head',{}).get('trainer'))")"
# ---- 3. head refit (v7.1's stage-2 procedure, v5 role manifests headfit + cal)
if [ ! -f $C/.ok ]; then
  fetch $E/code/esbar_code_v2e_full.tgz $W/tmp/code2e.tgz e6bd8a5aa97693be477ff50b5234f7ae
  fetch $E/code/esbar_code_v2e_full.md5.txt $W/tmp/code2e.md5.txt ead5408b79fb0ccc8723a753bf2d57fd
  (cd $D && tar xzf $W/tmp/code2e.tgz) || die UNTAR_CODE2E
  (cd $D && md5sum -c --quiet $W/tmp/code2e.md5.txt) > $W/logs/code2e_md5.txt 2>&1 || die "CODE2E_MD5 $(head -3 $W/logs/code2e_md5.txt)"
  touch $C/.ok
fi
mkdir -p $MAN
for m in headfit cal; do for x in $m.jsonl $m.jsonl.header.json; do fetch $B/engineering/coteach_wave4_20261028/p9_stage2_manifests/$x $MAN/$x; done; done
for x in "headfit f3cc7e116c8d264a181da7354e44e8234aab8cf8296aaf20f3e7c73a0b342980" "cal 95f14043c4bbf0524d22b05f12c11461d0d6618ffbbf64e4e756cbba61a97322"; do
  set -- $x
  g=$(sha256sum < $MAN/$1.jsonl | cut -c1-64); w=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['rows_sha256'])" $MAN/$1.jsonl.header.json)
  [ "$g" = "$2" ] && [ "$w" = "$2" ] || die "MANIFEST_SHA $1 $g"
done
log "code v2e + role manifests ok"
export PYTHONPATH=$C/engine:$C/trainers:$C
TR="$PY -m torch.distributed.run --nnodes 1 --nproc-per-node ${NPROC:-8} --rdzv-backend c10d --local-addr 127.0.0.1"
EXTRACTOR="$(md5sum $C/trainers/fullft_train.py | cut -c1-32)+$(md5sum $C/trainers/fullft_mem.py | cut -c1-32)"
port=${PORT0:-29890}
for role in headfit cal; do
  man=$MAN/$role.jsonl; dig=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['rows_sha256'])" $man.header.json)
  Fe=$FV/$role.pt; FM=$FV/$role.stamp.json
  EXP="{\"identity_sha256\": \"esbar_fs1:s2400\", \"export_sha256\": \"$XID\", \"policy\": \"B\", \"role\": \"$role\", \"manifest_rows_sha256\": \"$dig\", \"extractor_md5\": \"$EXTRACTOR\"}"
  echo "$EXP" > $FV/$role.expect.json
  if $PY $C/trainers/sdv1_head.py check-features --features $Fe --meta $FM --manifest $man --expect "$EXP" > /dev/null 2>&1; then log "FEATURES_VALID $role (cached)"; continue; fi
  port=$((port + 1)); t0=$(date +%s)
  $TR --rdzv-endpoint 127.0.0.1:$port $C/trainers/fullft_train.py --init-from $X --attention-policy B --seed 0 \
    --param-dtype bf16 --reduce-dtype fp32 --pack-tokens 8192 --ac 0 --offload-ac 0 --mlp-chunk 4096 --mlp-chunk-min 16384 \
    --pinned-code $C/trainers --manifest $man --expect-rows-sha256 $dig --extract-features $Fe \
    --out $FV/tmp_$role > $W/logs/features_$role.log 2>&1
  rc=$?; log "FEATURES $role rc=$rc wall_s=$(( $(date +%s) - t0 ))"
  [ $rc -eq 0 ] || die "FEATURES_FAILED $role: $(tail -3 $W/logs/features_$role.log)"
  $PY $C/trainers/sdv1_head.py stamp-features --features $Fe --meta $FM --manifest $man --expect "$EXP" >> $L 2>&1 || die "STAMP_FAILED $role"
done
if [ ! -s $HD/serving/vbll_head.json ]; then
  SDH="$PY $C/trainers/sdv1_head.py"; t0=$(date +%s)
  { OMP_NUM_THREADS=48 $SDH refit --features $FV/headfit.pt --manifest $MAN/headfit.jsonl --expect "$(cat $FV/headfit.expect.json)" --pinned-code $C/trainers --out $HD \
    && OMP_NUM_THREADS=48 $SDH fit-t --features $FV/cal.pt --manifest $MAN/cal.jsonl --expect "$(cat $FV/cal.expect.json)" --pinned-code $C/trainers --heads $HD --rule l01post_di; } > $W/logs/head.log 2>&1
  rc=$?; log "HEAD rc=$rc wall_s=$(( $(date +%s) - t0 ))"
  [ $rc -eq 0 ] || die "HEAD_FAILED: $(tail -3 $W/logs/head.log)"
fi
cp $HD/serving/vbll_head.safetensors $HD/serving/vbll_head.json $X/ || die HEAD_COPY
log "$(grep -E '^FIT_T' $W/logs/head.log | tail -1 | cut -c1-200)"
log "HEAD in FS1 s2400 export T=$(python3 -c "import json; print(json.load(open('$X/vbll_head.json'))['temperature'])") json md5 $(md5 $X/vbll_head.json)"
touch $X/.ok
# ---- 4. refit head + identity -> S3 (the export itself is already on S3 with its own read-back)
if [ ! -f $W/.head_s3_ok ]; then
  for f in vbll_head.json vbll_head.safetensors; do aws s3 cp --only-show-errors $X/$f $O/head/$f && \
    [ "$(aws s3 cp --only-show-errors $O/head/$f - | md5sum | cut -c1-32)" = "$(md5 $X/$f)" ] || die "HEAD_S3 $f"; done
  for f in fs1_identity.txt export_s2400_READBACK_MD5.txt; do aws s3 cp --only-show-errors $W/$f $O/head/$f || die "META_S3 $f"; done
  for f in head.log features_headfit.log features_cal.log; do aws s3 cp --only-show-errors $W/logs/$f $O/logs/$f; done
  touch $W/.head_s3_ok
fi
log "HEAD_UPLOADED (read-back md5 MATCH)"
aws s3 cp --only-show-errors $L $O/logs/build.log
log "FS1S_BUILD_OK -> driver"
export FS1S_HM=$(md5 $X/vbll_head.json)
exec bash $W/fs1_gate.sh
