#!/bin/bash
# eshbartur_rel_chain.sh <ws_id> <tag> <rel_name>
# Standalone Apache-2.0 HF release staging & complete 140,620-row (140,178 scored) Text 0.3 evaluation for eshbartur (t4b3 ckpt_03516 + T=1.40).
# Strictly isolated to s3://.../engineering/text/t4b/release/eshbartur/ and /data/local/eshbartur_rel/.
# Spot-resumable: mirrors all 8 shard results.jsonl to S3 every 5 minutes and restores them on restart.
# Publishes NOTHING to Hugging Face or GitHub.
set -u
unset ASCALE_RUN_ID HF_TOKEN HUGGING_FACE_HUB_TOKEN
export AWS_DEFAULT_REGION=us-east-2 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WS=$1; TAG=$2; NAME=$3
B=s3://YOUR_BUCKET/jev-race
E=$B/engineering/esbar
T4B=$B/engineering/text/t4b
CKPT_S3=$T4B/arms/t4b3/ckpt/ckpt_03516
P=$T4B/release/$NAME

U=/data/local/eshbartur_rel
C=$U/code
R=$U/repo
D=$U/dataset
O=$U/out
S=$U/shards
SO=$U/shards_out
L=$U/logs/chain_${NAME}.log
mkdir -p $U/logs $C $U/base $U/ckpt $U/t4b_code $U/kit $U/ref $S $SO $O $R $D/runs/$NAME

log() { echo "[$NAME $(date -u +%H:%M:%SZ)] $*" >> $L; }
mirror() { aws s3 cp --quiet $L $P/logs/chain_${NAME}.${TAG}.log 2>/dev/null || true; }
up() { aws s3 cp --quiet "$1" "$2"; }

exec 9> $U/chain.lock
flock -n 9 || { log "another chain holds the lock"; exit 0; }
log "START ws=$WS tag=$TAG host=$(hostname) gpus=$(nvidia-smi -L 2>/dev/null | wc -l)"

# 0. Markers & GPU count
aws s3 ls $P/state/${NAME}_chain_done > /dev/null 2>&1 && { log "CHAIN_DONE already on S3"; exit 0; }
aws s3 ls $P/state/${NAME}_fail > /dev/null 2>&1 && { log "FAIL marker on S3: holding"; exit 5; }
[ "$(nvidia-smi -L 2>/dev/null | wc -l)" = "8" ] || { log "NOT_8_GPUS"; mirror; exit 6; }

NL=$(aws s3 cp --quiet $P/state/${NAME}_launches - 2>/dev/null | tr -dc 0-9 | cut -c1-3)
NL=$(( ${NL:-0} + 1 ))
echo "$NL" | aws s3 cp --quiet - $P/state/${NAME}_launches
log "launch_number=$NL"
[ "$NL" -le 12 ] || { echo "too many launches $(date -u +%FT%TZ)" | aws s3 cp --quiet - $P/state/${NAME}_fail; mirror; exit 5; }

# 1. Fetch our release code bundle from S3 and verify MD5SUMS
aws s3 cp --quiet --recursive $P/code/ $C/ && (cd $C && md5sum -c --quiet MD5SUMS) || { log "CODE_BAD"; mirror; exit 3; }

# Ensure uv is available in $U/bin
mkdir -p $U/bin
if ! command -v uv >/dev/null 2>&1 && [ ! -x $U/bin/uv ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$U/bin sh > $U/logs/uv_install.log 2>&1
fi
UV_BIN=$(command -v uv 2>/dev/null || echo $U/bin/uv)
[ -x "$UV_BIN" ] || { log "UV_MISSING"; mirror; exit 4; }
XP="$(dirname "$UV_BIN"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
log "UV=$UV_BIN ($($UV_BIN --version))"

MREPO=Qwen/Qwen3.5-4B
MREV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a

# 2. Restage base model, ckpt_03516, t4b ref code, suite-0.3, s4400, and 1000-row latency sample in parallel
if [ ! -f $U/RESTAGE_OK ]; then
  pids=()
  # (a) Base Qwen/Qwen3.5-4B pinned snapshot
  (
    if [ ! -f $U/base/MODEL_OK ]; then
      python3 -m pip install -q --user "huggingface_hub>=1.0" >/dev/null 2>&1 || true
      python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MREPO', revision='$MREV', local_dir='$U/base', token=False, max_workers=16)" > $U/logs/base.log 2>&1
      python3 - $U/base >> $U/logs/base.log 2>&1 <<'PY' && touch $U/base/MODEL_OK
import json, os, struct, sys
d = sys.argv[1]
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
by = {}
for n, f in idx.items(): by.setdefault(f, set()).add(n)
for f, names in by.items():
    p = os.path.join(d, f)
    with open(p, "rb") as fh:
        hl = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(hl))
    end = max(v["data_offsets"][1] for k, v in h.items() if k != "__metadata__")
    assert os.path.getsize(p) == 8 + hl + end, f
    assert names <= set(h), f
print("BASE_OK", len(by), "shards", len(idx), "tensors")
PY
    fi
  ) & pids+=($!)
  # (b) t4b3 ckpt_03516 + t4b reference engine code (for parity check only)
  (
    aws s3 sync --quiet $CKPT_S3/ $U/ckpt/
    aws s3 cp --quiet $T4B/code/t4b_layout.py $U/t4b_code/t4b_layout.py
    aws s3 cp --quiet $T4B/code/t4b_engine.py $U/t4b_code/t4b_engine.py
    [ -s $U/ckpt/adapter.safetensors ] && [ -s $U/ckpt/readout.pt ] && [ -s $U/ckpt/state.json ] && touch $U/ckpt/.ok
  ) & pids+=($!)
  # (c) Official suite-0.3 + sample4400 + 1000-row public latency sample
  (
    if [ ! -f $U/kit/suite-0.3/.ok ]; then
      aws s3 cp --quiet $E/heldout/rebuilt_2026-11-02/suite-0.3.tgz $U/kit/suite.tgz
      [ "$(md5sum < $U/kit/suite.tgz | cut -c1-8)" = "02a6ec13" ] || { echo "SUITE_MD5_BAD"; exit 1; }
      tar xzf $U/kit/suite.tgz -C $U/kit && rm -f $U/kit/suite.tgz
      touch $U/kit/suite-0.3/.ok
    fi
    [ -s $U/ref/s4400_rows.jsonl.gz ] || aws s3 cp --quiet $E/runs/esbar_tb1/sample/sample4400-rows.jsonl.gz $U/ref/s4400_rows.jsonl.gz
    [ -s $U/ref/lat_sample_1000.jsonl.gz ] || aws s3 cp --quiet $P/ref/lat_sample_1000.jsonl.gz $U/ref/lat_sample_1000.jsonl.gz
    [ "$(sha256sum < $U/ref/lat_sample_1000.jsonl.gz | cut -c1-16)" = "5548d2d2d9a4cb6d" ] \
      && [ -f $U/kit/suite-0.3/.ok ] && [ -s $U/ref/s4400_rows.jsonl.gz ] && touch $U/kit/.all_ok
  ) & pids+=($!)
  for pid in "${pids[@]}"; do wait "$pid" || true; done

  [ -f $U/base/MODEL_OK ] && [ -f $U/ckpt/.ok ] && [ -f $U/kit/.all_ok ] \
    && touch $U/RESTAGE_OK && log "RESTAGE_OK $(tail -1 $U/logs/base.log)" \
    || { log "RESTAGE_FAIL base=$([ -f $U/base/MODEL_OK ] && echo 1 || echo 0) ckpt=$([ -f $U/ckpt/.ok ] && echo 1 || echo 0) kit=$([ -f $U/kit/.all_ok ] && echo 1 || echo 0)"; mirror; exit 4; }
  mirror
fi

# 3. Copy source files into $R, generate frozen uv.lock on Linux x86_64, install uv env
cp $C/src/eshbartur.py $R/eshbartur.py
cp $C/src/pyproject.toml $R/pyproject.toml
cp $C/src/.python-version $R/.python-version
cp $C/src/install.sh $R/install.sh && chmod 755 $R/install.sh
cp $C/src/selfcheck.py $R/selfcheck.py
cp $C/src/NOTICE $R/NOTICE
cp $C/src/LICENSE $R/LICENSE

uvenv() { env -i PATH="$XP" HOME="$U/home_inst" UV_CACHE_DIR="$U/uvcache" UV_NO_CONFIG=1 HF_HOME="$U/hf_rel" CUDA_VISIBLE_DEVICES=0 "$@"; }
mkdir -p $U/home_inst $U/hf_rel

if [ ! -f $R/uv.lock ] || [ ! -x $R/.venv/bin/python ]; then
  t0=$(date +%s)
  ( cd $R && uvenv uv lock ) > $U/logs/uv_lock.out 2>&1; rc=$?
  log "UV_LOCK rc=$rc $(( $(date +%s) - t0 ))s $(tail -n 2 $U/logs/uv_lock.out | tr '\n' ' ' | cut -c1-300)"
  [ $rc = 0 ] || { mirror; exit 4; }
  ( cd $R && uvenv bash ./install.sh --no-check ) > $U/logs/install_nocheck.out 2>&1; rc=$?
  log "INSTALL_NOCHECK rc=$rc $(( $(date +%s) - t0 ))s $(tail -n 2 $U/logs/install_nocheck.out | tr '\n' ' ' | cut -c1-300)"
  [ $rc = 0 ] || { mirror; exit 4; }
  mirror
fi
PY=$R/.venv/bin/python

# 4. Export standalone merged pure-text HF weights + readout.safetensors (T=1.40 folded) + eshbartur.json
if [ ! -f $O/build.json ] || ! grep -q '"BUILD_OK": true' $O/build.json; then
  rm -f $R/model-*.safetensors $R/model.safetensors.index.json $R/readout.safetensors $R/eshbartur.json $R/config.json
  CUDA_VISIBLE_DEVICES=0 $PY $C/eshbartur_export.py $U/base $U/ckpt $R $O/build.json cuda:0 > $U/logs/export.out 2>&1; rc=$?
  log "EXPORT rc=$rc $(tail -n 1 $U/logs/export.out | cut -c1-400)"
  [ $rc = 0 ] || { echo "export failed" | aws s3 cp --quiet - $P/state/${NAME}_fail; mirror; exit 5; }
  mirror
fi

# 5. Full shipped selfcheck + uv lock --check
( cd $R && uvenv bash ./install.sh ) > $U/logs/install_full.out 2>&1; rc=$?
log "INSTALL_FULL rc=$rc $(tail -n 1 $U/logs/install_full.out | cut -c1-300)"
[ $rc = 0 ] || { echo "selfcheck failed" | aws s3 cp --quiet - $P/state/${NAME}_fail; mirror; exit 5; }
( cd $R && uvenv uv lock --check ) > $U/logs/lock_check.out 2>&1; rc=$?
log "UV_LOCK_CHECK rc=$rc"
[ $rc = 0 ] || { echo "uv lock check failed" | aws s3 cp --quiet - $P/state/${NAME}_fail; mirror; exit 5; }
( cd $R && uvenv uv pip freeze ) > $O/uv_pip_freeze.txt 2>/dev/null || true

# 6. Independent bit-exact merge check + live engine parity check vs T4BEngine(T=1.40) (parallel on GPUs 0 & 1)
pids=()
if [ ! -f $O/merge_check.json ] || ! grep -q '"ALL_IDENTICAL": true' $O/merge_check.json; then
  ( CUDA_VISIBLE_DEVICES=0 $PY $C/eshbartur_check_merge.py $U/base $U/ckpt $R $O/merge_check.json cuda:0 > $U/logs/merge_check.out 2>&1; log "MERGE_CHECK rc=$? $(tail -n 1 $U/logs/merge_check.out | cut -c1-400)" ) & pids+=($!)
fi
if [ ! -f $O/parity_check.json ] || ! grep -q '"PARITY_OK": true' $O/parity_check.json; then
  ( CUDA_VISIBLE_DEVICES=1 $PY $C/eshbartur_verify.py parity $R $U/base $U/ckpt $U/t4b_code $U/ref/s4400_rows.jsonl.gz $O/parity_check.json > $U/logs/parity.out 2>&1; log "PARITY_CHECK rc=$? $(tail -n 1 $U/logs/parity.out | cut -c1-400)" ) & pids+=($!)
fi
# Also run 1-GPU 1,000-row public board latency sample on GPU 2 in parallel!
nv_poll() {
  local outf=$1 gpu=$2
  ( while :; do echo "$(date -u +%H:%M:%SZ) $(nvidia-smi -i $gpu --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ',')"; sleep 5; done ) > "$outf" &
  echo $!
}
if [ ! -f $O/latency_h200_1000.json ]; then
  (
    mkdir -p $U/lat_h200 $U/home_lat
    mp=$(nv_poll $U/logs/gpumem_lat.log 2)
    ( cd $R && env -i PATH="$XP" HOME="$U/home_lat" UV_CACHE_DIR="$U/uvcache" UV_NO_CONFIG=1 HF_HOME="$U/hf_rel" HF_HUB_OFFLINE=1 \
        OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 \
        uv run --frozen python -m decision_index run --rows $U/ref/lat_sample_1000.jsonl.gz --engine eshbartur:Engine --compact --out $U/lat_h200 ) \
        > $U/logs/lat_h200.out 2>&1; rc=$?
    kill "$mp" 2>/dev/null || true
    [ $rc = 0 ] && $PY $C/eshbartur_verify.py lat $U/lat_h200/results.jsonl $U/lat_h200/environment.json $U/logs/gpumem_lat.log $O/latency_h200_1000.json h200_1000 >> $L 2>&1
    log "LAT_1000 rc=$rc $(cat $O/latency_h200_1000.json 2>/dev/null | tr '\n' ' ' | cut -c1-350)"
  ) & pids+=($!)
fi
# And upload the verified model repo right now so it's safe on S3 even before the 140k run finishes!
for pid in "${pids[@]}"; do wait "$pid" || true; done
mirror

grep -q '"ALL_IDENTICAL": true' $O/merge_check.json || { echo "merge check failed" | aws s3 cp --quiet - $P/state/${NAME}_fail; mirror; exit 5; }
grep -q '"PARITY_OK": true' $O/parity_check.json || { echo "parity check failed" | aws s3 cp --quiet - $P/state/${NAME}_fail; mirror; exit 5; }
[ -s $O/latency_h200_1000.json ] || { echo "latency check failed" | aws s3 cp --quiet - $P/state/${NAME}_fail; mirror; exit 5; }
log "PRE_140K_GATES_PASSED"

# Upload verified standalone model repo to S3 now (excluding .venv, __pycache__)
aws s3 sync --quiet --exclude ".venv/*" --exclude "__pycache__/*" $R/ $P/repo/
aws s3 sync --quiet $O/ $P/checks/
log "MODEL_REPO_UPLOADED_TO_S3"
mirror

# 7. Partition official suite-0.3 (140,620 rows) round-robin into 8 shards + restore any prior shard progress from S3
if [ ! -f $S/shard_manifest.json ]; then
  $PY $C/eshbartur_verify.py shard $U/kit/suite-0.3 $S 8 >> $L 2>&1; rc=$?
  [ $rc = 0 ] || { echo "shard failed" | aws s3 cp --quiet - $P/state/${NAME}_fail; mirror; exit 5; }
fi

for i in 0 1 2 3 4 5 6 7; do
  mkdir -p $SO/shard_$i
  if [ ! -s $SO/shard_$i/results.jsonl ]; then
    aws s3 cp --quiet $P/work/shards/shard_$i/results.jsonl $SO/shard_$i/results.jsonl 2>/dev/null || true
    aws s3 cp --quiet $P/work/shards/shard_$i/environment.json $SO/shard_$i/environment.json 2>/dev/null || true
  fi
done

# Background S3 shard sync every 5 min (so spot preemption loses at most 5 min!)
(
  while :; do
    sleep 300
    tot=0
    for i in 0 1 2 3 4 5 6 7; do
      f=$SO/shard_$i/results.jsonl
      if [ -s "$f" ]; then
        c=$(wc -l < "$f" 2>/dev/null || echo 0)
        tot=$((tot + c))
        aws s3 cp --quiet "$f" "$P/work/shards/shard_$i/results.jsonl" 2>/dev/null || true
        [ -s "$SO/shard_$i/environment.json" ] && aws s3 cp --quiet "$SO/shard_$i/environment.json" "$P/work/shards/shard_$i/environment.json" 2>/dev/null || true
      fi
    done
    log "PROGRESS_140K done=$tot/140620"
    mirror
  done
) &
SYNC_PID=$!

# 8. Launch all 8 H200 workers running shipped eshbartur:Engine via uv run --frozen python -m decision_index run
pids=()
for i in 0 1 2 3 4 5 6 7; do
  (
    want=$(wc -l < $S/shard_$i.jsonl)
    got=$(grep -c '"status": *"ok"' $SO/shard_$i/results.jsonl 2>/dev/null || echo 0)
    if [ "$got" = "$want" ] && [ -s $SO/shard_$i/environment.json ]; then
      log "SHARD_$i already complete ($got/$want)"
      exit 0
    fi
    mkdir -p $U/home_s$i
    ( cd $R && env -i PATH="$XP" HOME="$U/home_s$i" UV_CACHE_DIR="$U/uvcache" UV_NO_CONFIG=1 HF_HOME="$U/hf_rel" HF_HUB_OFFLINE=1 \
        OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$i \
        uv run --frozen python -m decision_index run --rows $S/shard_$i.jsonl --engine eshbartur:Engine --compact --out $SO/shard_$i ) \
        > $U/logs/shard_$i.out 2>&1; rc=$?
    got=$(grep -c '"status": *"ok"' $SO/shard_$i/results.jsonl 2>/dev/null || echo 0)
    log "SHARD_$i finished rc=$rc ok=$got/$want $(tail -n 1 $U/logs/shard_$i.out | cut -c1-200)"
    aws s3 cp --quiet $SO/shard_$i/results.jsonl $P/work/shards/shard_$i/results.jsonl 2>/dev/null || true
    aws s3 cp --quiet $SO/shard_$i/environment.json $P/work/shards/shard_$i/environment.json 2>/dev/null || true
    [ "$rc" = "0" ] && [ "$got" = "$want" ]
  ) & pids+=($!)
done
log "LAUNCHED_8_SHARDS pids=${pids[*]}"
mirror

fail_n=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail_n=$((fail_n+1))
done
kill "$SYNC_PID" 2>/dev/null || true
log "ALL_8_SHARDS_DONE fail_n=$fail_n"
mirror
[ "$fail_n" = "0" ] || { echo "shard run failed ($fail_n)" | aws s3 cp --quiet - $P/state/${NAME}_fail; mirror; exit 5; }

# 9. Merge all 140,620 rows + official kit score --edition 0.3
FINAL=$U/final
rm -rf $FINAL && mkdir -p $FINAL
$PY $C/eshbartur_verify.py merge $U/kit/suite-0.3 $SO 8 $FINAL >> $L 2>&1; rc=$?
[ $rc = 0 ] || { echo "merge failed" | aws s3 cp --quiet - $P/state/${NAME}_fail; mirror; exit 5; }

( cd $R && env -i PATH="$XP" HOME="$U/home_inst" UV_CACHE_DIR="$U/uvcache" UV_NO_CONFIG=1 \
    uv run --frozen python -m decision_index score --edition 0.3 --suite-dir $U/kit/suite-0.3 \
    --results $FINAL/results.jsonl --engine eshbartur:Engine --out $FINAL/scored ) > $U/logs/score.out 2>&1; rc=$?
log "OFFICIAL_SCORE rc=$rc $(cat $U/logs/score.out | tr '\n' ' ' | cut -c1-600)"
[ $rc = 0 ] || { echo "official score failed" | aws s3 cp --quiet - $P/state/${NAME}_fail; mirror; exit 5; }

# Stage official dataset tree: dataset/runs/eshbartur/{results.jsonl.gz, scores.json, index.json, benchmark-summary.json, environment.json, latency.json}
rm -rf $D/runs/$NAME && mkdir -p $D/runs/$NAME
cp $FINAL/results.jsonl.gz $FINAL/environment.json $D/runs/$NAME/
cp $FINAL/scored/scores.json $FINAL/scored/index.json $FINAL/scored/benchmark-summary.json $D/runs/$NAME/
cp $O/latency_h200_1000.json $D/runs/$NAME/latency.json

# 10. Strict leak scan across BOTH repo/ and dataset/ (decompressed results.jsonl.gz included!) + release summary
$PY - $R $D $O $FINAL <<'PY' >> $L 2>&1
import gzip, json, os, sys
repo, ds, out_d, fin = sys.argv[1:5]
bad_all = [b"t4b", b"plantwaterer", b"trainers", b"jev-race", b"claude", b"anthropic", b"co-authored", b"s3://", b"/data/local"]
bad_tok = [b"t4b", b"plantwaterer", b"trainers", b"jev-race", b"co-authored", b"s3://", b"/data/local"]
leaks = []
for base_label, root_dir in (("repo", repo), ("dataset", ds)):
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in (".venv", "__pycache__")]
        for f in files:
            if f.endswith(".safetensors"):
                continue
            p = os.path.join(root, f)
            rel = os.path.relpath(p, root_dir)
            data = gzip.open(p, "rb").read() if f.endswith(".gz") else open(p, "rb").read()
            low = data.lower()
            words = bad_tok if (f.startswith("results.jsonl") or f in ("vocab.json", "tokenizer.json", "merges.txt")) else bad_all
            for w in words:
                if w in low:
                    leaks.append((base_label, rel, w.decode()))
assert not leaks, f"LEAKS_FOUND: {leaks[:10]}"

mc = json.load(open(f"{out_d}/merge_check.json"))
pc = json.load(open(f"{out_d}/parity_check.json"))
lat = json.load(open(f"{out_d}/latency_h200_1000.json"))
mg = json.load(open(f"{fin}/merge.json"))
sc = json.load(open(f"{fin}/scored/scores.json"))
assert sc["complete"] is True and sc["completed"] == 140178 and mg["rows"] == 140620 and mg["errors"] == 0, (sc["complete"], sc["completed"], mg)

summary = {
    "name": "eshbartur",
    "base": "Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "calibration_temperature_folded": 1.40,
    "merge_bit_identical": mc["ALL_IDENTICAL"],
    "parity_vs_t4b3_T140": pc,
    "latency_h200_1000": lat,
    "merge": mg,
    "official_text_03": {
        "edition": sc.get("edition", "0.3"),
        "decision_index": sc["decision_index"],
        "scores": sc["scores"],
        "completed": sc["completed"],
        "complete": sc["complete"],
        "areas": [{k: a[k] for k in ("id", "skill", "raw", "coverage", "n")} for a in sc["areas"]],
    },
    "leaks": leaks,
    "PASS": bool(mc["ALL_IDENTICAL"] and pc["PARITY_OK"] and sc["complete"] is True and sc["completed"] == 140178 and mg["errors"] == 0 and not leaks),
}
open(f"{out_d}/eshbartur_release_summary.json", "w").write(json.dumps(summary, indent=2) + "\n")
print("LEAK_AND_GATE_SCAN", json.dumps({"PASS": summary["PASS"], "decision_index": sc["decision_index"], "scores": sc["scores"], "completed": sc["completed"], "lat_med": lat["median_ms"]}))
assert summary["PASS"], "RELEASE_GATE_FAILED"
PY
rc=$?
[ $rc = 0 ] || { echo "leak or gate check failed" | aws s3 cp --quiet - $P/state/${NAME}_fail; mirror; exit 5; }

# 11. Compute SHA256SUMS, upload everything to S3, verify S3 sizes, emit chain_done
( cd $R && find . -type f ! -path './.venv/*' ! -path './__pycache__/*' ! -name '.DS_Store' | sed 's|^\./||' | LC_ALL=C sort | xargs -r sha256sum ) > $O/SHA256SUMS_model
( cd $D && find . -type f ! -name '.DS_Store' | sed 's|^\./||' | LC_ALL=C sort | xargs -r sha256sum ) > $O/SHA256SUMS_dataset
aws s3 sync --quiet --exclude ".venv/*" --exclude "__pycache__/*" $R/ $P/repo/
aws s3 sync --quiet $D/ $P/dataset/
aws s3 sync --quiet $O/ $P/checks/
up $O/SHA256SUMS_model $P/SHA256SUMS_model
up $O/SHA256SUMS_dataset $P/SHA256SUMS_dataset
up $O/eshbartur_release_summary.json $P/eshbartur_release_summary.json

python3 - $R $P/repo $O/SHA256SUMS_model $D $P/dataset $O/SHA256SUMS_dataset >> $L 2>&1 <<'PY'
import os, subprocess, sys
for local_dir, s3_uri, sums_f in ((sys.argv[1], sys.argv[2], sys.argv[3]), (sys.argv[4], sys.argv[5], sys.argv[6])):
    want = {}
    for l in open(sums_f):
        if l.strip():
            _, f = l.strip().split(None, 1)
            want[f] = os.path.getsize(os.path.join(local_dir, f))
    out = subprocess.check_output(["aws", "s3", "ls", "--recursive", s3_uri + "/"]).decode()
    pfx = s3_uri.split("/", 3)[3].rstrip("/") + "/"
    got = {}
    for l in out.splitlines():
        p = l.split(None, 3)
        if len(p) == 4 and p[3].startswith(pfx):
            got[p[3][len(pfx):]] = int(p[2])
    bad = [f for f, sz in want.items() if got.get(f) != sz]
    assert not bad, f"S3_SIZE_MISMATCH {s3_uri} {bad[:5]}"
    print(f"S3_VERIFY_OK {s3_uri} files={len(want)} total_bytes={sum(want.values())}")
PY
rc=$?
[ $rc = 0 ] || { echo "s3 verify failed" | aws s3 cp --quiet - $P/state/${NAME}_fail; mirror; exit 5; }

echo "done $(date -u +%FT%TZ)" | aws s3 cp --quiet - $P/state/${NAME}_chain_done
log "ESHBARTUR_RELEASE_STAGE_DONE"
mirror
