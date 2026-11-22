#!/bin/bash
# kl_parity.sh TAG INIT_DIR REF_DIR (KL agent): the parity gate. The 4,400 screen rows through kl_train.py's own sharded forward (predict mode,
# 8 ranks, bf16 over the fp32 master, no training) against karpo's engine's stored answers on the same weights (REF_DIR/prefix_results.jsonl.gz),
# then the kit's scorer on both. Result: s3 .../esbar_kl_par/TAG/{parity.json, score.txt, log}.
set -u; source /data/local/kl/scripts/kl_env.sh
TAG=${1:?}; INIT=${2:?}; REF=${3:?}; W=$K/par/$TAG; O=$R/esbar_kl_par/$TAG; mkdir -p $W; L=$W/parity.log
log() { echo "[kl_parity $TAG $(date -u +%FT%TZ)] $*" | tee -a $L; aws s3 cp --only-show-errors $L $O/parity.log 2>/dev/null; }
exec 8> $K/node.lock; flock -n 8 || { log "NODE_LOCK held: another kl job runs"; exit 3; }
ROWS=$KR/src/s4400_rows.jsonl.gz
$V/bin/python $SC/kl_parity_cmp.py make $ROWS $W/reqs.jsonl || { log "DIE make"; exit 1; }
log "start init $INIT ref $REF requests $(wc -l < $W/reqs.jsonl)"
t0=$(date +%s)
[ "${KL_SKIP_PREDICT:-0}" = 1 ] && [ -s $W/pred.jsonl ] || $TR --rdzv-endpoint 127.0.0.1:29801 $SC/kl_train.py --mode predict --init-from $INIT --out $W/run --pinned-code $C/trainers \
   --predict-in $W/reqs.jsonl --predict-out $W/pred.jsonl ${KL_EVAL_TOKENS:+--eval-tokens $KL_EVAL_TOKENS} > $W/predict.log 2>&1; rc=$?
log "predict rc=$rc $(( $(date +%s) - t0 )) s: $(grep -E 'PREDICT_DONE|BUILD ' $W/predict.log | cut -c1-260 | tr '\n' ' ')"
[ $rc = 0 ] && [ -s $W/pred.jsonl ] || { tail -30 $W/predict.log | cut -c1-400 > $W/tail.txt; aws s3 cp --only-show-errors $W/tail.txt $O/predict_tail.txt; log "DIE predict: $(grep -E 'Error|error' $W/predict.log | tail -3 | cut -c1-300 | tr '\n' ' ')"; exit 1; }
$V/bin/python $SC/kl_parity_cmp.py cmp $ROWS $REF/prefix_results.jsonl.gz $W/pred.jsonl $W/parity.json $W/kl_results.jsonl > $W/cmp.out 2>&1 || { log "DIE cmp: $(tail -2 $W/cmp.out | cut -c1-300)"; exit 1; }
zcat $REF/prefix_results.jsonl.gz > $W/ref_results.jsonl
PYTHONPATH=$KR/karpo/src:$KR/kit $V/bin/python $KR/gc/gate_index.py $KR/suite-0.3 $ROWS $W/score $W/compare.json KARPO=$KR/src/karpo_pub_results.jsonl.gz ENGINE=$W/ref_results.jsonl KLFWD=$W/kl_results.jsonl > $W/score.txt 2>&1 || log "SCORE_FAILED $(tail -2 $W/score.txt | cut -c1-200)"
for f in parity.json score.txt compare.json predict.log; do aws s3 cp --only-show-errors $W/$f $O/$f 2>/dev/null; done
if [ -n "${KL_ENGINE_HERE:-}" ]; then      # karpo's engine on THIS node, same weights: engine-vs-engine noise (reference run vs this node) and forward vs this node's engine
  [ -s $M/work/$KL_ENGINE_HERE/kown_s4400/prefix_results.jsonl ] || KSET=s4400 bash $M/scripts/mx_kown.sh $KL_ENGINE_HERE $INIT > $W/engine_here.out 2>&1
  EH=$M/work/$KL_ENGINE_HERE/kown_s4400/prefix_results.jsonl
  log "engine here: $(grep -E 'KOWN_DONE|DIE' $M/work/$KL_ENGINE_HERE/kown_s4400/driver.log | tail -1 | cut -c1-400)"
  $V/bin/python $SC/kl_parity_cmp.py cmp $ROWS $REF/prefix_results.jsonl.gz $EH $W/engine_vs_engine.json > $W/cmp_ee.out 2>&1; log "ENGINE(ref node) vs ENGINE(this node): $(tail -1 $W/cmp_ee.out | cut -c1-900)"
  $V/bin/python $SC/kl_parity_cmp.py cmp $ROWS $EH $W/pred.jsonl $W/parity_vs_engine_here.json > $W/cmp_eh.out 2>&1; log "FORWARD vs ENGINE(this node): $(tail -1 $W/cmp_eh.out | cut -c1-900)"
  for f in engine_vs_engine.json parity_vs_engine_here.json; do aws s3 cp --only-show-errors $W/$f $O/$f 2>/dev/null; done
fi
log "KL_PARITY_DONE $(cut -c1-700 $W/cmp.out | tail -1) | $(grep -E '^INDEX|^DIFF' $W/score.txt | cut -c1-160 | tr '\n' ';')"
