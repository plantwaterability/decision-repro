#!/bin/bash
# kl_kfullx.sh (TA agent, 11-16) = the MX fork's mx_kfull.sh 8629dd07 with three additions and nothing else changed: (1) engine option max_length rides on the tag
# (`_ml<N>` anywhere in TAG, e.g. kl_k24_full_ml98304 or kl_k24_full_ml98304_rev; no `_ml` = the engine's default 32,768, i.e. mx_kfull.sh exactly); (2) report lines OPTIONS /
# WALL_MS / LENGTH (request wall ms from the kit's model_request_wall_ms; unsupported rows by track) and final/benchmark-summary.json beside the results; (3) the determinism
# check may read another tag's prefix rows (KFULL_PREFIX_RESULTS). The original header follows.
# mx_kfull.sh TAG (MX fork, 2026-11-15): FULL public DI 0.3 run of a karpo-engine probe (a tag read by mx_kown.sh): the 24 standard shards
# (stage_public_rc.sh's rule: in-edition rows in suite order, row i -> shard i % 24; order sha 838b0d71ca8e, 140,620 requests) through kit
# 9eb2dbe2 with karpo_decision.serve.engine:DecisionIndexEngine (a6fa38bf) over the SAME checkpoint dir the prefix read used (rebuilt from
# the merge spec on S3 on a fresh disk), view by the tag suffix; merged in suite order (each request once, 0 error rows; unsupported rows
# stay and count as wrong) and scored by the kit's `score` (as_scored) + the board-rule scorer 155fbfdb under d82_exact_near.txt, as
# merge_score_rc.sh does. Shards -> S3 with DONE markers; a relaunch skips them. Determinism check against the prefix run. Nothing public.
#   bash mx_kfull.sh TAG CKPT_SRC      -> s3 .../esbar_TAG/kown_full/{shard_i_of_24/, final/}, KFULL.txt
set -u; unset ASCALE_RUN_ID; export AWS_DEFAULT_REGION=us-east-2 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
TAG=${1:?}; SRC=${2:?}; VIEW=joint; case $TAG in *_rev) VIEW=reversed;; *_single) VIEW=single;; esac
ML=$(echo "$TAG" | grep -oE '_ml[0-9]+' | head -1 | tr -dc 0-9); MLOPT=""; [ -n "$ML" ] && MLOPT="--option max_length=$ML"      # TA: max_length by tag
D=/data/local; M=$D/mx; S3=s3://YOUR_BUCKET/jev-race; E=$S3/engineering/esbar; O=$E/runs/esbar_$TAG/kown_full
W=$M/work/$TAG/kfull; V=$D/venv_mxk; R=$M/kroot; mkdir -p $W/run $W/merge; L=$W/driver.log
log() { echo "[kfull $TAG $(date -u +%FT%TZ)] $*" | tee -a $L; aws s3 cp --only-show-errors $L $O/logs/driver_$(hostname).log 2>/dev/null; }
die() { log "DIE $*"; exit 1; }
md5() { md5sum < "$1" | cut -c1-32; }
fetch() { [ -s "$2" ] || aws s3 cp --only-show-errors "$1" "$2" || die "NO_INPUT $1"; [ -z "${3:-}" ] || [ "$(md5 "$2" | cut -c1-${#3})" = "$3" ] || die "MD5 $1"; }
put() { aws s3 cp --only-show-errors "$1" "$2" && [ "$(aws s3 cp --only-show-errors "$2" - | md5sum | cut -c1-32)" = "$(md5 "$1")" ]; }
on_s3() { aws s3 ls "$1" > /dev/null 2>&1; }
bash $M/scripts/mx_kown.sh __setup__ /nonexistent > $M/logs/kown_setup.out 2>&1; [ -f $R/.ok ] && [ -x $V/bin/python ] || die SETUP
KIT=$R/kit; S03=$R/suite-0.3; export PYTHONPATH=$R/karpo/src:$KIT
if [ -s $SRC/decision_config.json ] && [ -s $SRC/readout.safetensors ]; then CK=$SRC
else CK=$M/work/$TAG/kown/ckpt; [ -s $CK/decision_config.json ] || { mkdir -p $M/work/$TAG/kown; (cd $M/work/$TAG/kown && $V/bin/python -m karpo_decision.checkpoint $SRC $CK bidirectional) >> $L 2>&1 || die CKPT_INIT; }; fi
if [ ! -f $R/shards/.ok ]; then      # the 24 standard shards, stage_public_rc.sh's python verbatim
  mkdir -p $R/shards
  $V/bin/python - $S03 $R/shards <<'PY' || die SHARDS
import gzip, hashlib, json, sys
from decision_index.runner import iter_rows
from decision_index.suite.io import Suite
s03, out = sys.argv[1], sys.argv[2]
suite = Suite(s03, "0.3")
rows = list(iter_rows(suite.row_paths, suite.in_edition))
fs = [gzip.open(f"{out}/shard_{i}_of_24_rows.jsonl.gz", "wt", encoding="utf-8") for i in range(24)]
for i, r in enumerate(rows):
    fs[i % 24].write(json.dumps(r, ensure_ascii=False) + "\n")
for f in fs:
    f.close()
order = [r["_evaluation"]["run_id"] for r in rows]
open(f"{out}/order.txt", "w").write("\n".join(order) + "\n")
print(json.dumps({"rows": len(rows), "order_sha256": hashlib.sha256("\n".join(order).encode()).hexdigest()}))
PY
  [ "$(sha256sum < $R/shards/order.txt | cut -c1-12)" = 838b0d71ca8e ] && [ "$(wc -l < $R/shards/order.txt)" = 140620 ] && touch $R/shards/.ok || die SHARD_ORDER
fi
CKID=$( (cd $CK && sha256sum decision_config.json readout.safetensors | sha256sum | cut -c1-16) ); WID=$(cat $SRC/.identity 2>/dev/null | cut -c1-16)
log "start host $(hostname) view $VIEW max_length ${ML:-default} checkpoint $CK (config+readout $CKID, weights identity ${WID:-?}) karpo code a7bd160d kit 1ec1274c"
NG=$(nvidia-smi -L | wc -l); pids=""
for g in $(seq 0 $((NG - 1))); do
  ( for i in $(seq 0 23); do [ $((i % NG)) = $g ] || continue
      on_s3 $O/shard_${i}_of_24/DONE && continue
      od=$W/run/shard_${i}_of_24; mkdir -p $od; t0=$(date +%s); want=$(zcat $R/shards/shard_${i}_of_24_rows.jsonl.gz | wc -l)
      (cd $W && CUDA_VISIBLE_DEVICES=$g $V/bin/python -m decision_index run --engine karpo_decision.serve.engine:DecisionIndexEngine \
         --option checkpoint=$CK --option view=$VIEW $MLOPT --edition 0.3 --suite-dir $S03 --rows $R/shards/shard_${i}_of_24_rows.jsonl.gz --out $od/run --compact) > $od/kit.log 2>&1
      rc=$?; n=$(wc -l < $od/run/results.jsonl 2>/dev/null); ne=$(grep -c '"status": *"error"' $od/run/results.jsonl 2>/dev/null)
      log "shard $i gpu $g rc=$rc rows=$n/$want errors=$ne wall_s=$(( $(date +%s) - t0 ))"
      [ $rc = 0 ] && [ "$n" = "$want" ] && [ "$ne" = 0 ] || { log "shard $i FAILED: $(tail -1 $od/kit.log | cut -c1-200)"; continue; }
      gzip -n -c $od/run/results.jsonl > $od/results.jsonl.gz
      put $od/results.jsonl.gz $O/shard_${i}_of_24/results.jsonl.gz && put $od/run/environment.json $O/shard_${i}_of_24/environment.json || { log "S3 shard $i"; continue; }
      echo "$(date -u +%FT%TZ) rows=$n errors=0 ckpt=$CKID weights=${WID:-?} view=$VIEW max_length=${ML:-default} host=$(hostname) gpu=$g wall_s=$(( $(date +%s) - t0 )) results_md5=$(md5 $od/results.jsonl.gz)" > $od/DONE
      put $od/DONE $O/shard_${i}_of_24/DONE
    done ) & pids="$pids $!"
done
for p in $pids; do wait $p; done
n=0; for i in $(seq 0 23); do on_s3 $O/shard_${i}_of_24/DONE && n=$((n + 1)); done
log "shards DONE on S3: $n/24"; [ $n = 24 ] || die "INCOMPLETE (a relaunch resumes)"
# ---- merge (suite order, each request once, 0 error rows) + scores
X=$W/merge; mkdir -p $X/shards
for i in $(seq 0 23); do d=$X/shards/shard_${i}_of_24; mkdir -p $d; for f in DONE results.jsonl.gz; do aws s3 cp --only-show-errors $O/shard_${i}_of_24/$f $d/$f || die "fetch shard $i $f"; done
  grep -q "ckpt=$CKID " $d/DONE && [ "$(md5 $d/results.jsonl.gz)" = "$(sed 's/.*results_md5=//' $d/DONE)" ] || die "shard $i checkpoint id / md5: $(cat $d/DONE)"; done
$V/bin/python - $R/shards/order.txt $X <<'PY' || die MERGE
import gzip, json, sys, collections
order = [l.strip() for l in open(sys.argv[1]) if l.strip()]
M = sys.argv[2]
by, dup, st, err = {}, 0, collections.Counter(), []
for i in range(24):
    for l in gzip.open(f"{M}/shards/shard_{i}_of_24/results.jsonl.gz", "rt"):
        x = json.loads(l)
        if x["run_id"] in by:
            dup += 1
        by[x["run_id"]] = l if l.endswith("\n") else l + "\n"
        st[x["status"]] += 1
        if x["status"] == "error":
            err.append(x["run_id"])
miss = [r for r in order if r not in by]
extra = set(by) - set(order)
with open(f"{M}/results.jsonl", "w") as f:
    for r in order:
        if r in by:
            f.write(by[r])
res = {"rows": len(by), "order": len(order), "missing": len(miss), "extra": len(extra), "duplicates": dup, "status": dict(st), "errors": len(err)}
json.dump(res, open(f"{M}/merge.json", "w"), indent=1)
print(json.dumps(res))
sys.exit(0 if len(by) == len(order) == 140620 and not miss and not extra and not dup and not err else 1)
PY
cd $X || die CD
$V/bin/python -m decision_index score --suite $S03 --edition 0.3 --results results.jsonl --out as_scored > score_as.log 2>&1 || { tail -3 score_as.log; die SCORE_AS; }
fetch $S3/fleetq/code/di03/score_decision_index.py $X/score_decision_index.py 155fbfdb
mkdir -p $W/trainers; fetch $S3/fleetq/code/trainers/convert_decision_index.py $W/trainers/convert_decision_index.py dc22d683
aws s3 cp --only-show-errors $S3/results_di/overlap/audit/wrong_ids/d82_exact_near.txt d82_exact_near.txt || die D82
$V/bin/python score_decision_index.py --kit $KIT --suite-dir $S03 --edition 0.3 --exclude-ids d82_exact_near.txt kit --results results.jsonl --out br19 > score_br.log 2>&1 || log "board-rule scorer failed: $(tail -1 score_br.log | cut -c1-200)"
$V/bin/python - $X ${KFULL_PREFIX_RESULTS:-$M/work/$TAG/kown/prefix_results.jsonl} $TAG $VIEW ${ML:-default} > $X/report.txt 2>&1 <<'PY'
import json, os, statistics as st, sys
X, PREF, TAG = sys.argv[1:4]; VIEW, ML = sys.argv[4:6]
v = json.load(open(f"{X}/as_scored/index.json")); m = json.load(open(f"{X}/merge.json"))
print(f"HEADLINE {TAG} FULL PUBLIC (karpo engine) as_scored index {v['index']} raw {v.get('raw_index')} | statuses {m['status']} | vs karpo published 75.05 {v['index'] - 75.05:+.2f} | vs munku 71.10 {v['index'] - 71.10:+.2f}")
try:
    b = json.load(open(f"{X}/br19/scores.json")); print("BOARD_RULE", json.dumps(b)[:400])
except Exception as e: print("board-rule scores:", e)
for a in v["areas"]: print(f"AREA {a['id']:10s} {a['skill']:.4f}")
lat = []; wall = []; uns = {}
ans = lambda r: json.dumps((r.get("response") or {}).get("answers"), sort_keys=True); full = {}
for l in open(f"{X}/results.jsonl"):
    r = json.loads(l); full[r["run_id"]] = ans(r)
    x = r.get("latency_ms")
    if isinstance(x, (int, float)): lat.append(x)
    x = r.get("model_request_wall_ms")
    if isinstance(x, (int, float)): wall.append(x)
    if r.get("status") != "ok": uns[r.get("track")] = uns.get(r.get("track"), 0) + 1
print(f"OPTIONS view {VIEW} max_length {ML}")
if wall:
    wall.sort(); print(f"WALL_MS n {len(wall)} median {st.median(wall):.1f} mean {sum(wall) / len(wall):.1f} p80 {wall[int(0.8 * len(wall))]:.1f} p95 {wall[int(0.95 * len(wall))]:.1f} p99 {wall[int(0.99 * len(wall))]:.1f} max {wall[-1]:.0f} over_1000 {sum(1 for x in wall if x > 1000)} over_10000 {sum(1 for x in wall if x > 10000)} sum_gpu_h {sum(wall) / 3.6e6:.2f}")
print(f"LENGTH rows not ok by track: {uns}")
if lat:
    lat.sort(); print(f"LATENCY_THIS_GPU n {len(lat)} median {st.median(lat):.1f} mean {sum(lat) / len(lat):.1f} p80 {lat[int(0.8 * len(lat))]:.1f} p95 {lat[int(0.95 * len(lat))]:.1f} max {lat[-1]:.0f} over_1000 {sum(1 for x in lat if x > 1000)}")
if os.path.exists(PREF):
    seen = same = 0
    for l in open(PREF):
        r = json.loads(l)
        if r["run_id"] in full: seen += 1; same += ans(r) == full[r["run_id"]]
    print(f"DETERMINISM prefix rows found in the full run {seen}; identical answers {same} ({100 * same / max(seen, 1):.2f}%)")
PY
gzip -n -c results.jsonl > results.jsonl.gz
for f in results.jsonl.gz merge.json report.txt score_as.log; do put $f $O/final/$f || die "S3 $f"; done
for f in as_scored/* br19/*; do [ -f $f ] && put $f $O/final/$f; done
[ -f as_scored/benchmark-summary.json ] && put as_scored/benchmark-summary.json $O/final/benchmark-summary.json      # TA (coordinator A4): the kit's per-benchmark summary beside the results
cp $CK/decision_config.json $X/decision_config.used.json; put $X/decision_config.used.json $O/final/decision_config.used.json
grep -E "^HEADLINE|^LATENCY|^DETERMINISM|^OPTIONS|^WALL_MS|^LENGTH" report.txt > KFULL.txt; put KFULL.txt $E/runs/esbar_$TAG/kown_full/KFULL.txt
date -u +%FT%TZ > DONE; put DONE $O/final/DONE
log "KFULL_DONE $(grep -E '^HEADLINE' report.txt | cut -c1-260)"
