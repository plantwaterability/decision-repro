#!/bin/bash
# FS1 step-2,400 MEASUREMENT on the fixed public prefix (prereg AGENT_NOTES "PREREG FS1-S2400-PREFIX"; it decides nothing by itself).
# Steps 1-6 = fs1_gate_r4.sh (code, export + refit head, bundle FS1S = E1's package, public root, inputs, runset with the newline-only
# splitter). Then: M1 the fixed 31,198-row prefix, answered-rows index vs E1's stored rows; M2 pooled ho8 + hoL + hoF (diagnostic).
# Every chunk -> S3 when it finishes; a relaunch skips what is on S3. No stop / continue rule and no full read.
#   CODEMD5=<md5 of fs1_code.tgz> FS1S_HM=<md5 of the refit vbll_head.json> PREFIX_SHA=<ids sha256> bash fs1_gate.sh
set -u
unset ASCALE_RUN_ID; export AWS_DEFAULT_REGION=us-east-2 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
D=/data/local; S3=s3://YOUR_BUCKET/jev-race; E=$S3/engineering/esbar
O=$E/runs/esbar_fs1/public_di_s2400; RUN=fs1s2400_20261112
W=$D/fs1pub; P=$W/pub; C=$P/rc_code; X=$W/fs1_export
mkdir -p $W/logs $W/src $W/scorer/analysis $W/scorer/trainers $W/integrate $C
L=$W/logs/driver.log
log() { echo "[readFS1S $(date -u +%FT%TZ)] $*" | tee -a $L; }
die() { log "DIE $*"; exit 1; }
md5() { md5sum < "$1" | cut -c1-32; }
fetch() { [ -s "$2" ] || aws s3 cp --only-show-errors "$1" "$2" || die "NO_INPUT $1"
          [ -z "${3:-}" ] || [ "$(md5 "$2" | cut -c1-${#3})" = "$3" ] || die "MD5 $1 $(md5 "$2" | cut -c1-8) != $3"; }
put() { aws s3 cp --only-show-errors "$1" "$2" && [ "$(aws s3 cp --only-show-errors "$2" - | md5sum | cut -c1-32)" = "$(md5 "$1")" ]; }
log "start host $(hostname) run $RUN"
# ---- 1. venv (built + functionally probed by the v7.1 public run on this worker)
V=$D/venv_t2141
probe() { $V/bin/python -c "import torch, transformers, safetensors; assert torch.__version__ == '2.14.1+cu130', torch.__version__; torch.zeros(1).cuda(); print('PROBE torch', torch.__version__, 'transformers', transformers.__version__, 'gpus', torch.cuda.device_count())"; }
probe >> $L 2>&1 || die VENV_PROBE
log "env ok: $(probe 2>&1 | tail -1)"
export PATH=$V/bin:$PATH
# ---- 2. run code (md5-checked on this node)
fetch $O/code/fs1_code.tgz $W/src/fs1_code.tgz ${CODEMD5:?}
tar xzf $W/src/fs1_code.tgz -C $C || die UNTAR_CODE
for kv in public_run_rc.sh:0fd24ec7 stage_public_rc.sh:a4766453 merge_score_rc.sh:f0b1c9e5 report_public_rc.py:448b599b; do
  f=${kv%%:*}; m=${kv#*:}; [ "$(md5 $C/$f | cut -c1-8)" = $m ] || die "CODE_MD5 $f"; done
log "code ok"
# ---- 3. the FS1 s2400 export (pulled + verified by fs1_build.sh; head refit in it)
[ -f $X/.ok ] && [ -f $W/.head_s3_ok ] || die NO_FS1S_EXPORT
[ "$(ls $X/model-*.safetensors | wc -l)" = 12 ] || die EXPORT_SHARDS
HM=$(md5 $X/vbll_head.json); [ "$HM" = "${FS1S_HM:?}" ] || die "HEAD_JSON $HM"
log "export ok ($(du -sh $X | cut -f1)), head json $HM"
# ---- 4. bundle FS1S (built once; md5s persisted to S3)
if aws s3 cp --only-show-errors $O/bundle_FS1S/bundle.env $W/bundle.env 2>/dev/null; then
  . $W/bundle.env; log "bundle from S3: $BMD5 pins $PMD5"
else
  [ -e $W/bundle_FS1S ] && die "bundle_FS1S dir exists but no bundle.env on S3 (partial earlier build): inspect, then remove $W/bundle_FS1S and $W/tree_FS1S"
  fetch $E/v6rule/code/integrate/build_bundle.py $W/integrate/build_bundle.py efd134ba901a67a29450d4ff24d9b856
  fetch $E/v6rule/code/integrate/verify_pkg.py $W/integrate/verify_pkg.py dbc7f880ceaac467e3e7bb6f854afff2
  (cd $X && ls model-*.safetensors | sort | xargs -P 12 -n 1 sha256sum | sort -k2) > $W/fs1_weights.sha256
  XID=$( (cat $W/fs1_weights.sha256; cd $X && sha256sum config.json) | sort -k2 | sha256sum | cut -c1-64)
  log "FS1 s2400 export identity (raw config) $XID"
  V6X=$X V6ID=$XID V6HEAD=$HM ARM=FS1S bash $C/phaseFS1S_bundle.sh >> $W/logs/bundle.log 2>&1 || { tail -20 $W/logs/bundle.log; die BUNDLE; }
  tail -4 $W/logs/bundle.log | tee -a $L
  grep -q "^PACKAGE_EQUALS_E1" $W/logs/bundle.log || die PACKAGE_NOT_E1
  BMD5=$(md5 $W/bundle_FS1S/eshbar_FS1S.tgz); PMD5=$(md5 $W/tree_FS1S/PINS.json)
  printf 'BMD5=%s\nPMD5=%s\nXID=%s\n' $BMD5 $PMD5 $XID > $W/bundle.env
  put $W/bundle.env $O/bundle_FS1S/bundle.env || die BUNDLE_ENV_S3
  log "bundle built: $BMD5 pins $PMD5"
fi
# ---- 5. public root: kit 9eb2dbe2 (1ec1274c), suite-0.3 (02a6ec13), 24 shards (order 838b0d71ca8e), scorer 155fbfdb, converter dc22d683
if [ ! -f $P/shards/.ok ] || [ ! -f $P/suite-0.3/.ok ]; then
  fetch $S3/fleetq/code/di03/kit_9eb2dbe2a358004c8782c66e40a83ac07b953fec.tgz $W/src/kit.tgz 1ec1274c4adccb55b25c2ba88fd87c75
  fetch $E/heldout/rebuilt_2026-11-02/suite-0.3.tgz $W/src/suite.tgz 02a6ec13
  fetch $S3/fleetq/code/di03/score_decision_index.py $W/scorer/analysis/score_decision_index.py 155fbfdb
  fetch $S3/fleetq/code/trainers/convert_decision_index.py $W/scorer/trainers/convert_decision_index.py dc22d683
  WROOT=$P SRC=$W/src SCORER=$W/scorer bash $C/stage_public_rc.sh >> $W/logs/stage.log 2>&1 || { tail -5 $W/logs/stage.log; die STAGE; }
fi
log "public root ok: $(tail -1 $W/logs/stage.log 2>/dev/null)"
# ---- 6. gate inputs
G=$W/gate; OG=$O/gate; KIT=$P/kit; S03=$P/suite-0.3; GC=$C; mkdir -p $G/src $G/run $G/score $W/logs
ver() { echo "$(date -u +%FT%TZ) $*" >> $G/VERDICT.txt; log "VERDICT $*"; put $G/VERDICT.txt $OG/VERDICT.txt || true; put $L $OG/logs/driver.log || true; }
fetch $E/v6rule/code/integrate/verify_pkg.py $W/integrate/verify_pkg.py dbc7f880ceaac467e3e7bb6f854afff2
T=$G/tree
if [ ! -f $T/.verified ]; then
  [ -e $T ] && die "gate tree exists unverified: inspect $T"
  mkdir -p $T
  for f in $X/model-*.safetensors $X/model.safetensors.index.json $X/tokenizer.json $X/tokenizer_config.json $X/chat_template.jinja $X/generation_config.json $X/processor_config.json; do
    ln -s $f $T/$(basename $f) || die "LINK $f"; done
  fetch $O/bundle_FS1S/eshbar_FS1S.tgz $G/src/bundle_FS1S.tgz $BMD5
  tar xzf $G/src/bundle_FS1S.tgz -C $T 2>/dev/null && (cd $T && md5sum -c --quiet BUNDLE_MD5SUMS) && [ "$(md5 $T/PINS.json)" = $PMD5 ] || die GATE_TREE
  python3 -I $W/integrate/verify_pkg.py $T --pins $T/PINS.json --export-identity > $W/logs/gate_verify.txt 2>&1 || { cat $W/logs/gate_verify.txt; die GATE_VERIFY; }
  grep -q "^VERIFY_PKG PASS arm=FS1S " $W/logs/gate_verify.txt || die GATE_VERIFY_ARM
  touch $T/.verified
fi
fetch $E/runs/esbar_tb1/sample/sample4400-rows.jsonl.gz $G/src/sample4400.jsonl.gz
[ "$(sha256sum < $G/src/sample4400.jsonl.gz | cut -c1-16)" = 6296ba7ae4f9269f ] || die SAMPLE_SHA
fetch $E/rtxrig/samples/ho8_rows.jsonl.gz $G/src/ho8_rows.jsonl.gz 751763f3c35793161f044a3c756fdc1f
fetch $E/rtxrig/samples/ho8_manifest.json $G/src/ho8_manifest.json
fetch $E/forkE_oorder/rows/hoL_rows.jsonl.gz $G/src/hoL_rows.jsonl.gz
fetch $E/forkE_oorder/rows/hoL_manifest.json $G/src/hoL_manifest.json
fetch $E/forkE_oorder/v2/rows/hoF_rows.jsonl.gz $G/src/hoF_rows.jsonl.gz c2f6e92ffd72006a07cfc6b66df19937
fetch $E/forkE_oorder/v2/rows/hoF_manifest.json $G/src/hoF_manifest.json
fetch $E/runs/esbar_soupB/reads/E1R/ho8_results.jsonl $G/src/base_ho8.jsonl
fetch $E/forkE_oorder/run/K4_results.jsonl.gz $G/src/base_hoL.jsonl.gz
fetch $E/forkE_oorder/v2/run/SH_results.jsonl.gz $G/src/base_hoF.jsonl.gz
for s in hoL hoF; do [ -s $G/src/base_$s.jsonl ] || zcat $G/src/base_$s.jsonl.gz > $G/src/base_$s.jsonl; done
fetch $E/public_E1/e1_20261104T1251Z/final/results.jsonl.gz $G/src/e1_public_results.jsonl.gz
for d in eval_v5 eval_pools_v5; do [ -d $G/rec/$d ] || { mkdir -p $G/rec/$d; aws s3 sync --only-show-errors --exclude '*' --include 'shard*.jsonl' $E/runs/esbar_v5/$d/ $G/rec/$d/ || die REC; }; done
[ -s $G/src/ho_rows.jsonl.gz ] || { zcat $G/src/ho8_rows.jsonl.gz $G/src/hoL_rows.jsonl.gz $G/src/hoF_rows.jsonl.gz | gzip -n > $G/src/ho_rows.jsonl.gz; }
[ "$(zcat $G/src/ho_rows.jsonl.gz | wc -l)" = 8000 ] || die "HO_ROWS $(zcat $G/src/ho_rows.jsonl.gz | wc -l)"
if [ ! -s $G/prefix/prefix_ids.txt ]; then
  PYTHONPATH=$KIT python3 $GC/gate_prefix.py $S03 $G/src/sample4400.jsonl.gz $P/shards $G/prefix > $W/logs/gate_prefix.log 2>&1 || { tail -5 $W/logs/gate_prefix.log; die PREFIX; }
fi
PSHA=$(sha256sum < $G/prefix/prefix_ids.txt | cut -c1-64); [ -z "${PREFIX_SHA:-}" ] || [ "$PSHA" = "$PREFIX_SHA" ] || die "PREFIX_SHA $PSHA"
for f in prefix_ids.txt prefix_meta.json prefix_rows.jsonl.gz; do put $G/prefix/$f $OG/prefix/$f || die "S3 prefix $f"; done
log "gate inputs ok: prefix $(wc -l < $G/prefix/prefix_ids.txt) rows ids sha256 $PSHA; $(tail -1 $W/logs/gate_prefix.log 2>/dev/null)"
# ---- runset NAME ROWS NCHUNKS STRICT: chunks i % NCHUNKS, chunk k on GPU k % 8, each chunk's results + route -> S3 when it finishes
runset() { name=$1; rows=$2; nch=$3; strict=$4
  python3 - $rows $G/src $name $nch <<'PY' || die "SPLIT $name"
import gzip, sys
rows = [r for r in gzip.open(sys.argv[1], "rt", newline="\n").read().split("\n") if r]; n = int(sys.argv[4])   # r4: newline only (splitlines also cuts at U+2028 etc. inside JSON strings)
fs = [gzip.open(f"{sys.argv[2]}/{sys.argv[3]}_{k}_of_{n}.jsonl.gz", "wt") for k in range(n)]
for i, r in enumerate(rows): fs[i % n].write(r + "\n")
for f in fs: f.close()
PY
  pids=""
  for g in 0 1 2 3 4 5 6 7; do
    ( for k in $(seq 0 $((nch - 1))); do [ $((k % 8)) = $g ] || continue
        od=$G/run/${name}_$k; mkdir -p $od; src=$G/src/${name}_${k}_of_${nch}.jsonl.gz; want=$(zcat $src | wc -l)
        if aws s3 cp --only-show-errors $OG/$name/chunk_${k}_of_${nch}.jsonl.gz $od/results.jsonl.gz 2>/dev/null && [ "$(zcat $od/results.jsonl.gz | wc -l)" = "$want" ]; then
          zcat $od/results.jsonl.gz > $od/results.jsonl; log "set $name chunk $k from S3"; continue; fi
        t0=$(date +%s)
        (cd $T && PYTHONPATH=$KIT:$T CUDA_VISIBLE_DEVICES=$g STAGE2_ACTIVITY=$od/route.jsonl python3 -m decision_index run --engine eshbar_lulilla:Engine \
           --option model=$T --edition 0.3 --suite-dir $S03 --rows $src --out $od/run --compact) > $od/kit.log 2>&1
        rc=$?; n=$(wc -l < $od/run/results.jsonl 2>/dev/null); ne=$(grep -c '"status": *"error"' $od/run/results.jsonl 2>/dev/null)
        log "set $name chunk $k gpu $g rc=$rc rows=$n/$want errors=$ne wall_s=$(( $(date +%s) - t0 ))"
        [ $rc = 0 ] && [ "$n" = "$want" ] || { log "set $name chunk $k FAILED"; continue; }
        [ $strict = 0 ] || [ "$ne" = 0 ] || { log "set $name chunk $k FAILED (errors)"; continue; }
        cp $od/run/results.jsonl $od/results.jsonl; gzip -n -c $od/results.jsonl > $od/results.jsonl.gz; gzip -n -c $od/route.jsonl > $od/route.jsonl.gz 2>/dev/null || : > $od/route.jsonl.gz
        put $od/route.jsonl.gz $OG/$name/route_${k}_of_${nch}.jsonl.gz || log "S3 route $name $k"
        [ -f $od/run/environment.json ] && aws s3 cp --only-show-errors $od/run/environment.json $OG/$name/environment.json
        put $od/results.jsonl.gz $OG/$name/chunk_${k}_of_${nch}.jsonl.gz || { log "S3 FAILED $name $k"; rm -f $od/results.jsonl; }
      done ) & pids="$pids $!"
  done
  for p in $pids; do wait $p; done
  : > $G/${name}_results.jsonl
  for k in $(seq 0 $((nch - 1))); do [ -s $G/run/${name}_$k/results.jsonl ] || die "SET_INCOMPLETE $name chunk $k (relaunch resumes)"; cat $G/run/${name}_$k/results.jsonl >> $G/${name}_results.jsonl; done
  log "set $name done: $(wc -l < $G/${name}_results.jsonl) rows"; }
num() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); x=d
for k in sys.argv[2:]: x=x[int(k)] if isinstance(x, list) else x[k]
print(x)" "$@"; }
ge() { python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)" "$1" "$2"; }
# ---- M1 the fixed public prefix (errors must be 0)
runset prefix $G/prefix/prefix_rows.jsonl.gz 24 1
[ "$(wc -l < $G/prefix_results.jsonl)" = "$(wc -l < $G/prefix/prefix_ids.txt)" ] || die "PREFIX_ROWS $(wc -l < $G/prefix_results.jsonl)"
PYTHONPATH=$KIT python3 $GC/gate_index.py $S03 $G/prefix/prefix_ids.txt $G/score/prefix $G/prefix_compare.json E1=$G/src/e1_public_results.jsonl.gz FS1S=$G/prefix_results.jsonl > $W/logs/gate_prefix_score.txt 2>&1 || { tail -5 $W/logs/gate_prefix_score.txt; die SCORE_PREFIX; }
grep -E "^INDEX|^DIFF" $W/logs/gate_prefix_score.txt | cut -c1-200 | tee -a $L; put $G/prefix_compare.json $OG/prefix_compare.json || die S3_PREFIX
gzip -n -c $G/prefix_results.jsonl > $G/prefix_results.jsonl.gz; put $G/prefix_results.jsonl.gz $OG/prefix_results.jsonl.gz || die S3_PREFIX_RESULTS
D3=$(num $G/prefix_compare.json diff_vs_E1 FS1S)
ver "M1 prefix MEASUREMENT (decides nothing): FS1S(step 2,400) - E1 = $D3 on $(wc -l < $G/prefix/prefix_ids.txt) rows; FS1S $(num $G/prefix_compare.json arms FS1S index) vs E1 $(num $G/prefix_compare.json arms E1 index)"
# ---- M2 pooled non-public diagnostic
runset ho $G/src/ho_rows.jsonl.gz 8 0
python3 $GC/gate_ho.py $G/rec/eval_v5,$G/rec/eval_pools_v5 $G/ho_compare.json ho8=$G/src/ho8_manifest.json,$G/src/base_ho8.jsonl,$G/ho_results.jsonl,$G/src/ho8_rows.jsonl.gz \
  hoL=$G/src/hoL_manifest.json,$G/src/base_hoL.jsonl,$G/ho_results.jsonl,$G/src/hoL_rows.jsonl.gz hoF=$G/src/hoF_manifest.json,$G/src/base_hoF.jsonl,$G/ho_results.jsonl,$G/src/hoF_rows.jsonl.gz > $W/logs/gate_ho.txt 2>&1 || { tail -5 $W/logs/gate_ho.txt; die SCORE_HO; }
grep -E "^SET|^POOLED" $W/logs/gate_ho.txt | cut -c1-260 | tee -a $L; put $G/ho_compare.json $OG/ho_compare.json || die S3_HO
ver "M2 pooled ho8+hoL+hoF DIAGNOSTIC: FS1S - E1 = $(num $G/ho_compare.json pooled paired 0) [$(num $G/ho_compare.json pooled paired 1), $(num $G/ho_compare.json pooled paired 2)]"
aws s3 cp --only-show-errors $W/logs/launch.out $OG/logs/launch.out
log "READS_DONE"
