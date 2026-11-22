#!/bin/bash
# dh_tlm.sh TAG REQ [LIMIT] (DH fork: copy of the MX fork's mx_tl_label.sh for GROUPED multi-question requests; labeller dh_tl_label.py writes
# {"id", "p": {question_key: {option_key: prob}}}; mean_max_prob is over all questions). Original header: mx_tl_label.sh TAG REQ [LIMIT] (MX fork, distillation line): karpo-own-engine teacher probabilities for the structured requests file
# s3 .../esbar_mx/tl/REQ.jsonl.gz (mx_tl_convert.py's output), 8 GPU shards, parts mirrored to S3 every 5 minutes (a fresh disk pulls them and
# continues), merged file -> s3 .../esbar_TAG/tl/labels.jsonl.gz + stats. Teacher = the released karpo checkpoint as it ships (its engine code a6fa38bf).
set -u; unset ASCALE_RUN_ID; export AWS_DEFAULT_REGION=us-east-2 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
TAG=${1:?}; REQ=${2:?}; LIMIT=${3:-0}
D=/data/local; M=$D/mx; E=s3://YOUR_BUCKET/jev-race/engineering/esbar; MX=$E/runs/esbar_mx; O=$E/runs/esbar_$TAG/tl
W=$M/work/$TAG/tl; V=$D/venv_mxk; R=$M/kroot; CK=$M/src/karpo; mkdir -p $W/parts; L=$W/driver.log
log() { echo "[dh_tlm $TAG $(date -u +%FT%TZ)] $*" | tee -a $L; aws s3 cp --only-show-errors $L $O/driver_$(hostname).log 2>/dev/null; }
aws s3 cp --only-show-errors s3://YOUR_BUCKET/jev-race/engineering/esbar/runs/esbar_dh/code/dh_tl_label.py $M/scripts/dh_tl_label.py || { echo "no dh_tl_label.py"; exit 1; }; bash $M/scripts/mx_kown.sh __setup__ /nonexistent > $M/logs/kown_setup.out 2>&1; [ -f $R/.ok ] && [ -x $V/bin/python ] && [ -f $CK/.ok ] || { log "DIE setup / karpo not staged"; exit 1; }
[ -s $W/req.jsonl.gz ] || aws s3 cp --only-show-errors $MX/tl/$REQ.jsonl.gz $W/req.jsonl.gz || { log "DIE no requests file"; exit 1; }
aws s3 sync --only-show-errors $O/parts/ $W/parts/ 2>/dev/null
NG=$(nvidia-smi -L | wc -l); log "start host $(hostname) requests $REQ ($(zcat $W/req.jsonl.gz | wc -l) rows, limit $LIMIT) parts on disk $(cat $W/parts/*.jsonl 2>/dev/null | wc -l) rows"
( while sleep 300; do aws s3 sync --only-show-errors $W/parts/ $O/parts/; done ) & SYNCER=$!; trap 'kill $SYNCER 2>/dev/null' EXIT
pids=""; for g in $(seq 0 $((NG - 1))); do
  ( cd $W && PYTHONPATH=$R/karpo/src CUDA_VISIBLE_DEVICES=$g $V/bin/python $M/scripts/dh_tl_label.py $CK $W/req.jsonl.gz $W/parts/part_${g}_of_$NG.jsonl $g $NG $LIMIT > $W/label_$g.log 2>&1 ) & pids="$pids $!"; done
for p in $pids; do wait $p; done
aws s3 sync --only-show-errors $W/parts/ $O/parts/
[ "$(grep -l LABEL_DONE $W/label_*.log | wc -l)" = $NG ] || { log "DIE a shard did not finish: $(grep -L LABEL_DONE $W/label_*.log | head -2 | tr '\n' ' ') $(tail -1 $(grep -L LABEL_DONE $W/label_*.log | head -1) | cut -c1-200)"; exit 1; }
$V/bin/python - $W $LIMIT <<'PY' > $W/merge.txt 2>&1 || { log "DIE merge: $(tail -1 $W/merge.txt | cut -c1-200)"; exit 1; }
import glob, gzip, json, sys, collections
W, limit = sys.argv[1], int(sys.argv[2]); want = []
for i, l in enumerate(gzip.open(f"{W}/req.jsonl.gz", "rt")):
    if limit and i >= limit: break
    want.append(json.loads(l)["id"])
got = {}; err = collections.Counter()
for f in glob.glob(f"{W}/parts/part_*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if "p" in r: got[r["id"]] = r["p"]
        else: err[r["err"].split(":")[0]] += 1
miss = [i for i in want if i not in got]
with gzip.open(f"{W}/labels.jsonl.gz", "wt") as g:
    for i in want:
        if i in got: g.write(json.dumps({"id": i, "p": got[i]}) + "\n")
st = {"requests": len(want), "labelled": len(want) - len(miss), "missing": len(miss), "errors": dict(err), "questions": sum(len(got[i]) if isinstance(next(iter(got[i].values())), dict) else 1 for i in got),
      "mean_max_prob": (lambda v: sum(v) / max(1, len(v)))([max(q.values()) for i in want if i in got for q in (got[i].values() if isinstance(next(iter(got[i].values())), dict) else [got[i]])])}
json.dump(st, open(f"{W}/labels_stats.json", "w"), indent=1); print("TL_MERGED", json.dumps(st))
PY
aws s3 cp --only-show-errors $W/labels.jsonl.gz $O/labels.jsonl.gz && aws s3 cp --only-show-errors $W/labels_stats.json $O/labels_stats.json || { log "DIE s3"; exit 1; }
log "TLAB_DONE $(grep TL_MERGED $W/merge.txt | cut -c11-400) | $(grep -h LABEL_DONE $W/label_*.log | head -2 | tr '\n' ' ')"
