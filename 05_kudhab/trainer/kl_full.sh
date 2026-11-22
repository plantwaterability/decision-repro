#!/bin/bash
# kl_full.sh (KL agent; coordinator order 22:05Z 11-15, item 6): the FULL public DI 0.3 run in karpo's engine (the text agent's mx_kfull.sh: 24 shards, kit
# 9eb2dbe2, board-rule scorer) of the export named in s3 .../esbar_kl_code/full/REQUEST ("ARM TAG S3_EXPORT IDENTITY prefix=X time"), on this node. Nothing public:
# rows and scores go to s3 .../esbar_<TAG>/kown_full/ only. Writes full/DONE (rc + summary) at the end, which lets kl_guard terminate this workspace.
set -u; unset ASCALE_RUN_ID; export AWS_DEFAULT_REGION=us-east-2
D=/data/local; K=$D/kl; M=$D/mx; R=s3://YOUR_BUCKET/jev-race/engineering/esbar/runs; F=$R/esbar_kl_code/full; mkdir -p $K/logs $K/src $M/scripts $M/logs; L=$K/logs/full.log
log() { echo "[kl_full $(date -u +%FT%TZ)] $*" | tee -a $L; aws s3 cp --only-show-errors $L $F/full_$(hostname).log 2>/dev/null; }
exec 8> $K/node.lock; flock -n 8 || { echo "NODE_LOCK held"; exit 3; }
if [ $# -ge 4 ]; then ARM=$1; TAG=$2; SRC=$3; WANT=$4; REST="(by hand: $*)"; DM=DONE_$TAG      # a named run (not the REQUEST one): its own DONE marker, so the klf rule is not touched
else read ARM TAG SRC WANT REST < <(aws s3 cp --only-show-errors $F/REQUEST -); DM=DONE; fi
[ ${#WANT} = 64 ] || { log "bad request / arguments"; exit 1; }
echo "whole node: KL agent FULL public run of $TAG in karpo's engine (nothing public), since $(date -u +%FT%TZ)" > /tmp/claim.txt
log "start host $(hostname) gpus $(nvidia-smi -L | wc -l) arm $ARM tag $TAG src $SRC identity $(echo $WANT | cut -c1-16) $REST"
[ "$(nvidia-smi -L | wc -l)" = 8 ] || { log "FAILED not 8 GPUs"; exit 1; }
aws s3 sync --only-show-errors $R/esbar_mx/scripts/ $M/scripts/ || { log "FAILED mx scripts"; exit 1; }
X=$K/src/${ARM}_full
if [ ! -f $X/.ok ]; then mkdir -p $X && aws s3 sync --only-show-errors --exclude READBACK_MD5.txt $SRC/ $X/ || { log "FAILED export sync"; exit 1; }
  I=$( (cd $X && ls model*.safetensors config.json | sort | xargs -P 13 -n 1 sha256sum | sort -k2) | sha256sum | cut -c1-64); [ "$I" = "$WANT" ] || { log "FAILED identity $I"; exit 1; }; echo $I > $X/.identity; touch $X/.ok; fi
if [ -n "${KL_FULL_PRETEST:-}" ]; then      # shake-out of mx_kfull.sh's merge / score tail on karpo's PUBLISHED full rows (CPU, in the background): must give 75.05
  ( T=$K/fulltest; mkdir -p $T/merge $T/trainers; cd $T/merge; KR=$M/kroot; V=$D/venv_mxk; S3=s3://YOUR_BUCKET/jev-race
    until [ -f $KR/.ok2 ] && [ -x $V/bin/python ]; do sleep 20; done; export PYTHONPATH=$KR/karpo/src:$KR/kit
    zcat $KR/src/karpo_pub_results.jsonl.gz > results.jsonl
    $V/bin/python -m decision_index score --suite $KR/suite-0.3 --edition 0.3 --results results.jsonl --out as_scored > score_as.log 2>&1; r1=$?
    aws s3 cp --only-show-errors $S3/fleetq/code/di03/score_decision_index.py score_decision_index.py; aws s3 cp --only-show-errors $S3/fleetq/code/trainers/convert_decision_index.py $T/trainers/convert_decision_index.py
    aws s3 cp --only-show-errors $S3/results_di/overlap/audit/wrong_ids/d82_exact_near.txt d82_exact_near.txt; r2=$?
    $V/bin/python score_decision_index.py --kit $KR/kit --suite-dir $KR/suite-0.3 --edition 0.3 --exclude-ids d82_exact_near.txt kit --results results.jsonl --out br19 > score_br.log 2>&1; r3=$?
    { echo "PRETEST $(date -u +%FT%TZ) $(hostname): kit score rc=$r1 (karpo published rows; expect index 75.05): $($V/bin/python -c "import json; v=json.load(open('as_scored/index.json')); print('index', v['index'], 'raw', v.get('raw_index'), 'areas', [(a['id'], round(a['skill'], 4)) for a in v['areas']])" 2>&1 | cut -c1-300)"
      echo "scorer md5 $(md5sum < score_decision_index.py | cut -c1-8) (want 155fbfdb) convert md5 $(md5sum < $T/trainers/convert_decision_index.py | cut -c1-8) (want dc22d683) d82 fetch rc=$r2 lines $(wc -l < d82_exact_near.txt)"
      echo "board-rule scorer rc=$r3: $(head -c 400 br19/scores.json 2>/dev/null | tr '\n' ' ') $(tail -2 score_br.log | cut -c1-300 | tr '\n' ' ')"; echo "score_as tail: $(tail -2 score_as.log | cut -c1-300 | tr '\n' ' ')"; } > $T/pretest.txt 2>&1
    aws s3 cp --only-show-errors $T/pretest.txt $F/pretest_$(hostname).txt ) > /dev/null 2>&1 &
fi
log "export staged (identity ok); mx_kfull.sh $(md5sum < $M/scripts/mx_kfull.sh | cut -c1-8) starts"
bash $M/scripts/mx_kfull.sh $TAG $X > $K/logs/kfull_$TAG.out 2>&1; rc=$?
S="$(grep -E 'KFULL|INDEX|DIE' $M/work/$TAG/kfull/driver.log 2>/dev/null | tail -4 | cut -c1-300 | tr '\n' ';')"
log "mx_kfull.sh rc=$rc: $S"; aws s3 cp --only-show-errors $K/logs/kfull_$TAG.out $F/kfull_$TAG.out 2>/dev/null
echo "$(date -u +%FT%TZ) $(hostname) rc=$rc $TAG: $S | results s3 .../esbar_$TAG/kown_full/" | aws s3 cp --only-show-errors - $F/$DM
echo "FREE: full public run of $TAG finished rc=$rc $(date -u +%FT%TZ)" > /tmp/claim.txt
