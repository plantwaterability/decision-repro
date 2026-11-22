#!/bin/bash
# peshtur_rel_chain.sh <ws_id> <tag> <rel_name>
# Standalone Apache-2.0 HF release staging & complete 12,210-row Vision 0.3 evaluation for peshtur (v4b2 ckpt_05051).
# Strictly isolated to s3://.../engineering/vision/v4b/release/peshtur/ and /data/local/peshtur_rel/.
# Winoground raw rows live strictly under chmod-700 /data/local/eval_restricted/ and are wiped on exit.
# Publishes NOTHING to Hugging Face or GitHub.
set -u
unset ASCALE_RUN_ID HF_TOKEN HUGGING_FACE_HUB_TOKEN
export AWS_DEFAULT_REGION=us-east-2 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1

WS=$1; TAG=$2; NAME=$3
B=s3://YOUR_BUCKET/jev-race
V=$B/engineering/vision
A=$V/recipe_a
V4B=$V/v4b
CKPT_S3=$V4B/arms/v4b2/ckpt/ckpt_05051
REF_S3=$V4B/arms/v4b2/public_read
P=$V4B/release/$NAME
SR=$V/eval_restricted/winoground_research_only

U=/data/local/peshtur_rel
E=/data/local/eval_restricted
C=$U/code
R=$U/repo
O=$U/out
L=$U/logs/chain_${NAME}.log
mkdir -p $U/logs $C $U/src $U/base $U/ckpt $U/ref $U/suite $U/kit $U/ocr $U/reads $U/h200 $O $R

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
[ "$NL" -le 8 ] || { echo "too many launches $(date -u +%FT%TZ)" | aws s3 cp --quiet - $P/state/${NAME}_fail; mirror; exit 5; }

# 1. Fetch our release code bundle from S3 and verify MD5SUMS
aws s3 cp --quiet --recursive $P/code/ $C/ && (cd $C && md5sum -c --quiet MD5SUMS) || { log "CODE_BAD"; mirror; exit 3; }

# Ensure OpenCV system libraries (libgl1, libglib2.0-0) are present on fresh workers
[ -e /usr/lib/x86_64-linux-gnu/libGL.so.1 ] || {
  sudo -n env DEBIAN_FRONTEND=noninteractive apt-get update -q > $U/logs/apt.out 2>&1
  sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -q libgl1 libglib2.0-0 >> $U/logs/apt.out 2>&1
  log "APT libgl1/libglib2.0-0 rc=$?"
}

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
WH=paddlepaddle_gpu-3.2.0-cp312-cp312-linux_x86_64.whl
WSHA=127f7f1afbc7e268080f733718e3c667ebb06d1152dd32861a6201836c8ddba0
PUBSHA=0d44a006e575363aee592dd7a03a5ae45cf095c45eef6ce74079e8a040ef5d3c
WGSHA=63119d5b7d663b8c4f254c8995cbf6d082e4005f0a67867830b8f4934945d8ec
OCRMD5=212232c925a9f94c9b4c6b0506d73db9

# 2. Restage base model, ckpt, suite, OCR cache, kit_vis, ref read, paddle wheel in parallel (explicit PIDs)
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
  # (b) v4b2 ckpt_05051 + v4b2 public_read reference shards
  (
    aws s3 sync --quiet $CKPT_S3/ $U/ckpt/
    aws s3 sync --quiet $REF_S3/ $U/ref/ --exclude "*.log"
    [ -s $U/ckpt/adapter.safetensors ] && [ -s $U/ckpt/head.pt ] && [ -s $U/ckpt/state.json ] && touch $U/ckpt/.ok
  ) & pids+=($!)
  # (c) Public no-WG suite + OCR cache + kit_vis
  (
    [ -d $U/kit/kit_vis ] || { aws s3 cp --quiet $A/code/kit_vis.tgz $U/kit/ && tar xzf $U/kit/kit_vis.tgz -C $U/kit; }
    if [ ! -f $U/suite/suite/vision-rows-noWG.jsonl ]; then
      aws s3 cp --quiet $V/suite/suite_noWG.tar $U/suite/
      aws s3 cp --quiet $V/suite/suite_noWG.tar.md5 $U/suite/
      [ "$(awk '{print $1}' $U/suite/suite_noWG.tar.md5)" = "$(md5sum < $U/suite/suite_noWG.tar | cut -d' ' -f1)" ] && tar xf $U/suite/suite_noWG.tar -C $U/suite && rm -f $U/suite/suite_noWG.tar
      [ -f $U/suite/suite/vision-rows-noWG.jsonl ] || aws s3 cp --quiet $V/dormice/gate/vision-rows-noWG.jsonl $U/suite/suite/vision-rows-noWG.jsonl
    fi
    if [ ! -f $U/ocr/ocr_public.jsonl ] || [ "$(md5sum < $U/ocr/ocr_public.jsonl | cut -d' ' -f1)" != "$OCRMD5" ]; then
      aws s3 cp --quiet $A/ocr/ocr_public_EVAL_ONLY.jsonl $U/ocr/ocr_public.jsonl
    fi
    [ "$(sha256sum $U/suite/suite/vision-rows-noWG.jsonl | cut -d' ' -f1)" = "$PUBSHA" ] \
      && [ "$(md5sum < $U/ocr/ocr_public.jsonl | cut -d' ' -f1)" = "$OCRMD5" ] \
      && touch $U/suite/.ok
  ) & pids+=($!)
  # (d) Pre-cache PaddlePaddle GPU wheel from S3 (exact sha256 checked)
  (
    mkdir -p $R/wheels
    if [ ! -f $R/wheels/$WH ] || [ "$(sha256sum $R/wheels/$WH | cut -d' ' -f1)" != "$WSHA" ]; then
      aws s3 cp --quiet $V/usu_release/wheels/$WH $R/wheels/$WH
    fi
    [ "$(sha256sum $R/wheels/$WH | cut -d' ' -f1)" = "$WSHA" ] && touch $U/wheel.ok
  ) & pids+=($!)
  for pid in "${pids[@]}"; do wait "$pid" || true; done

  [ -f $U/base/MODEL_OK ] && [ -f $U/ckpt/.ok ] && [ -f $U/suite/.ok ] && [ -f $U/wheel.ok ] \
    && touch $U/RESTAGE_OK && log "RESTAGE_OK $(tail -1 $U/logs/base.log)" \
    || { log "RESTAGE_FAIL base=$([ -f $U/base/MODEL_OK ] && echo 1 || echo 0) ckpt=$([ -f $U/ckpt/.ok ] && echo 1 || echo 0) suite=$([ -f $U/suite/.ok ] && echo 1 || echo 0) wheel=$([ -f $U/wheel.ok ] && echo 1 || echo 0)"; mirror; exit 4; }
  mirror
fi

# 3. Copy source files into $R, generate frozen uv.lock on Linux x86_64, run shipped bash install.sh (without selfcheck yet until weights exported)
cp $C/src/peshtur.py $R/peshtur.py
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

# 4. Export standalone merged HF weights + readout.safetensors + peshtur.json (GPU 0)
if [ ! -f $O/build.json ] || ! grep -q '"BUILD_OK": true' $O/build.json; then
  rm -f $R/model-*.safetensors $R/model.safetensors.index.json $R/readout.safetensors $R/peshtur.json
  CUDA_VISIBLE_DEVICES=0 $PY $C/peshtur_export.py $U/base $U/ckpt $R $O/build.json cuda:0 > $U/logs/export.out 2>&1; rc=$?
  log "EXPORT rc=$rc $(tail -n 1 $U/logs/export.out | cut -c1-400)"
  [ $rc = 0 ] || { echo "export failed" | aws s3 cp --quiet - $P/state/${NAME}_fail; mirror; exit 5; }
  mirror
fi

# 5. Now run FULL shipped selfcheck (verifies torch+paddle GPU, all files present, small sha256, live GPU OCR) + uv lock --check
( cd $R && uvenv bash ./install.sh ) > $U/logs/install_full.out 2>&1; rc=$?
log "INSTALL_FULL rc=$rc $(tail -n 1 $U/logs/install_full.out | cut -c1-300)"
[ $rc = 0 ] || { echo "selfcheck failed" | aws s3 cp --quiet - $P/state/${NAME}_fail; mirror; exit 5; }
( cd $R && uvenv uv lock --check ) > $U/logs/lock_check.out 2>&1; rc=$?
log "UV_LOCK_CHECK rc=$rc"
[ $rc = 0 ] || { echo "uv lock check failed" | aws s3 cp --quiet - $P/state/${NAME}_fail; mirror; exit 5; }
( cd $R && uvenv uv pip freeze ) > $O/uv_pip_freeze.txt 2>/dev/null || true

# 6. Independent bit-exact merge check (GPU 0)
if [ ! -f $O/merge_check.json ] || ! grep -q '"ALL_IDENTICAL": true' $O/merge_check.json; then
  CUDA_VISIBLE_DEVICES=0 $PY $C/peshtur_check_merge.py $U/base $U/ckpt $R $O/merge_check.json cuda:0 > $U/logs/merge_check.out 2>&1; rc=$?
  log "MERGE_CHECK rc=$rc $(tail -n 1 $U/logs/merge_check.out | cut -c1-400)"
  [ $rc = 0 ] || { echo "merge check failed" | aws s3 cp --quiet - $P/state/${NAME}_fail; mirror; exit 5; }
  mirror
fi

# 7. Prepare 547-row latency/parity sample + 4-way no-WG shards + Winoground suite (eval_restricted)
python3 - $U/suite/suite/vision-rows-noWG.jsonl $U/reads/rows_lat.jsonl $U/h200 <<'PY'
import hashlib, json, os, sys
rows = [l for l in open(sys.argv[1]) if l.strip()]
sel = [l for l in rows if int(hashlib.sha1(json.loads(l)["id"].encode()).hexdigest(), 16) % 20 == 0]
os.makedirs(os.path.dirname(sys.argv[2]), exist_ok=True)
open(sys.argv[2], "w").writelines(sel)
for k in range(4):
    d = os.path.join(sys.argv[3], f"s{k}")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "vision-rows.jsonl"), "w").writelines(rows[k::4])
print("SPLITS", len(sel), [len(rows[k::4]) for k in range(4)])
PY
mkdir -p $U/reads/lat
cp $U/reads/rows_lat.jsonl $U/reads/lat/vision-rows.jsonl
ln -sfn $U/suite/suite/full $U/reads/lat/full
for k in 0 1 2 3; do ln -sfn $U/suite/suite/full $U/h200/s$k/full; done

# Rebuild Winoground strictly inside chmod 700 /data/local/eval_restricted
if [ "$(sha256sum $E/wg_suite/wg_rows.jsonl 2>/dev/null | cut -d' ' -f1)" != "$WGSHA" ]; then
  mkdir -p $E/raw/data $E/build $E/wg_run $E/wg_out; chmod 700 $E
  for f in data/examples.jsonl data/images.zip; do
    [ -f $E/raw/$f ] || aws s3 cp --quiet $SR/$f $E/raw/$f || { log "WG_RAW_FAIL $f"; mirror; exit 5; }
  done
  cp $C/wg_rebuild.py $E/build/wg_rebuild.py
  BW=$E/wg_suite.building_$(date -u +%H%M%S); mkdir -p $BW
  env -i PATH="$XP" HOME="$U/home_inst" $PY $E/build/wg_rebuild.py $E/raw $BW $WGSHA > $E/wg_rebuild.out 2>&1; rc=$?
  log "WG_REBUILD rc=$rc $(tail -n 2 $E/wg_rebuild.out | tr '\n' ' ' | cut -c1-300)"
  [ $rc = 0 ] || { echo "wg rebuild failed" | aws s3 cp --quiet - $P/state/${NAME}_fail; mirror; exit 5; }
  rm -rf $E/wg_suite && mv $BW $E/wg_suite
fi
[ "$(sha256sum $E/wg_suite/wg_rows.jsonl | cut -d' ' -f1)" = "$WGSHA" ] || { log "WG_SHA_BAD"; mirror; exit 5; }
mkdir -p $E/wg_run $E/wg_out
cp $E/wg_suite/wg_rows.jsonl $E/wg_run/vision-rows.jsonl
ln -sfn $E/wg_suite/full $E/wg_run/full
log "WG_SUITE_OK rows=$(wc -l < $E/wg_run/vision-rows.jsonl)"

# Helper: GPU memory logger (closes flock fd 9 and stdio so command substitution returns immediately)
nv_poll() {
  local outf=$1 gpu=$2
  ( exec 9>&- </dev/null >/dev/null 2>&1; while :; do echo "$(date -u +%H:%M:%SZ) $(nvidia-smi -i $gpu --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ',')" >> "$outf"; sleep 5; done ) &
  echo $!
}

# Helper: run shipped engine via uv run --frozen python -m decision_index vision run (live PaddleOCR, no options)
shipped() {
  local lab=$1 gpu=$2 dir=$3 outd=$4 compact=$5
  local want=$(wc -l < "$dir/vision-rows.jsonl")
  if [ "$(grep -c '"status": *"ok"' "$outd/results.jsonl" 2>/dev/null || echo 0)" = "$want" ]; then
    log "SHIPPED $lab already done ($want rows)"
    return 0
  fi
  mkdir -p "$U/home_$lab" "$outd"
  local cf=""; [ "$compact" = "1" ] && cf="--compact"
  local mp; mp=$(nv_poll "$U/logs/gpumem_$lab.log" "$gpu")
  ( cd $R && env -i PATH="$XP" HOME="$U/home_$lab" UV_CACHE_DIR="$U/uvcache" UV_NO_CONFIG=1 HF_HOME="$U/hf_rel" HF_HUB_OFFLINE=1 \
      OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu \
      uv run --frozen python -m decision_index vision run --dir "$dir" --engine peshtur:Engine --workers 1 --chunk 50 $cf --no-verify --out "$outd" ) \
      > "$U/logs/run_$lab.out" 2>&1
  local rc=$?
  kill "$mp" 2>/dev/null || true
  local got=$(grep -c '"status": *"ok"' "$outd/results.jsonl" 2>/dev/null || echo 0)
  log "SHIPPED $lab gpu=$gpu rc=$rc ok=$got/$want $(tail -n 1 "$U/logs/run_$lab.out" | cut -c1-200)"
  [ "$rc" = "0" ] && [ "$got" = "$want" ]
}

# 8. Launch ALL 8 GPUs in parallel!
#   GPUs 0, 1, 2, 7: 11,410 public no-WG rows (shipped path, live OCR, compact)
#   GPU 3: r1 (cached OCR, 547-row sample, parity vs v4b2)
#   GPU 4: r1b (cached OCR, 547-row sample, determinism vs r1)
#   GPU 5: r2 (shipped path, live OCR, non-compact, 547-row latency sample)
#   GPU 6: Winoground 800 rows (shipped path, live OCR, compact)
pids=()
G=(0 1 2 7)
for k in 0 1 2 3; do
  shipped "h200_$k" "${G[$k]}" "$U/h200/s$k" "$U/h200/out_$k" 1 & pids+=($!)
done
(
  [ "$(grep -c '"status": *"ok"' $U/reads/r1/results.jsonl 2>/dev/null || echo 0)" = "$(wc -l < $U/reads/rows_lat.jsonl)" ] || \
    HF_HOME="$U/hf_rel" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 $PY $C/peshtur_verify_reads.py read_cached $R $U/reads/rows_lat.jsonl $U/suite/suite $U/reads/r1 $U/ocr/ocr_public.jsonl > $U/logs/run_r1.out 2>&1
  log "CACHED r1 $(tail -n 1 $U/logs/run_r1.out 2>/dev/null | cut -c1-200)"
) & pids+=($!)
(
  [ "$(grep -c '"status": *"ok"' $U/reads/r1b/results.jsonl 2>/dev/null || echo 0)" = "$(wc -l < $U/reads/rows_lat.jsonl)" ] || \
    HF_HOME="$U/hf_rel" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4 $PY $C/peshtur_verify_reads.py read_cached $R $U/reads/rows_lat.jsonl $U/suite/suite $U/reads/r1b $U/ocr/ocr_public.jsonl > $U/logs/run_r1b.out 2>&1
  log "CACHED r1b $(tail -n 1 $U/logs/run_r1b.out 2>/dev/null | cut -c1-200)"
) & pids+=($!)
shipped "r2" 5 "$U/reads/lat" "$U/reads/r2" 0 & pids+=($!)
shipped "wg" 6 "$E/wg_run" "$E/wg_out" 1 & pids+=($!)

log "LAUNCHED_8_GPUS pids=${pids[*]}"
mirror
fail_n=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail_n=$((fail_n+1))
done
log "ALL_8_GPUS_FINISHED fail_n=$fail_n"
mirror
[ "$fail_n" = "0" ] || { echo "one or more GPU runs failed ($fail_n)" | aws s3 cp --quiet - $P/state/${NAME}_fail; mirror; exit 5; }

# 9. Parity checks + latency summary + official 12,210-row assembly
cat $U/h200/out_0/results.jsonl $U/h200/out_1/results.jsonl $U/h200/out_2/results.jsonl $U/h200/out_3/results.jsonl > $U/h200/all_nowg.jsonl
$PY $C/peshtur_verify_reads.py cmp $U/reads/rows_lat.jsonl $U/reads/r1/results.jsonl "$U/ref/**/results.jsonl" $O/cmp_r1_vs_v4b2.json r1_vs_v4b2 >> $L 2>&1
$PY $C/peshtur_verify_reads.py cmp $U/reads/rows_lat.jsonl $U/reads/r1b/results.jsonl $U/reads/r1/results.jsonl $O/cmp_r1b_vs_r1.json r1b_vs_r1 >> $L 2>&1
$PY $C/peshtur_verify_reads.py cmp $U/reads/rows_lat.jsonl $U/reads/r2/results.jsonl $U/reads/r1/results.jsonl $O/cmp_r2_vs_r1.json r2_vs_r1 >> $L 2>&1
$PY $C/peshtur_verify_reads.py cmp $U/suite/suite/vision-rows-noWG.jsonl $U/h200/all_nowg.jsonl "$U/ref/**/results.jsonl" $O/cmp_h200_vs_v4b2.json h200_vs_v4b2 >> $L 2>&1
$PY $C/peshtur_verify_reads.py lat $U/reads/r2/results.jsonl $U/reads/r2/environment.json $U/logs/gpumem_r2.log $O/latency_h200_r2.json h200_r2_547 >> $L 2>&1

PARTS="h200=$U/h200/out_0@$U/h200/s0/vision-rows.jsonl+$U/h200/out_1@$U/h200/s1/vision-rows.jsonl+$U/h200/out_2@$U/h200/s2/vision-rows.jsonl+$U/h200/out_3@$U/h200/s3/vision-rows.jsonl"
rm -rf $R/runs
$PY $C/peshtur_verify_reads.py assemble peshtur $E/asm_tmp $R "$PARTS" "h200-winoground=$E/wg_out@$E/wg_suite/wg_rows.jsonl" >> $L 2>&1; rc=$?
[ $rc = 0 ] || { echo "assemble failed" | aws s3 cp --quiet - $P/state/${NAME}_fail; mirror; exit 5; }
cp $E/asm_tmp/assemble_summary.json $O/assemble_summary.json

# 10. Strict leak scan & release summary
$PY - $R $O <<'PY' >> $L 2>&1
import json, os, sys
repo, out_d = sys.argv[1:3]
bad_all = [b"v4b", b"plantwaterer", b"trainers", b"jev-race", b"claude", b"anthropic", b"co-authored", b"s3://", b"/data/local"]
bad_tok = [b"v4b", b"plantwaterer", b"trainers", b"jev-race", b"co-authored", b"s3://", b"/data/local"]
leaks = []
for root, dirs, files in os.walk(repo):
    dirs[:] = [d for d in dirs if d not in (".venv", "wheels", "__pycache__")]
    for f in files:
        if f.endswith(".safetensors"):
            continue
        p = os.path.join(root, f)
        rel = os.path.relpath(p, repo)
        data = open(p, "rb").read().lower()
        words = bad_tok if f in ("results.jsonl", "vocab.json", "tokenizer.json", "merges.txt") else bad_all
        for w in words:
            if w in data:
                leaks.append((rel, w.decode()))
assert not leaks, f"LEAKS_FOUND: {leaks[:10]}"

# Gate check
mc = json.load(open(f"{out_d}/merge_check.json"))
r1 = json.load(open(f"{out_d}/cmp_r1_vs_v4b2.json"))
det = json.load(open(f"{out_d}/cmp_r1b_vs_r1.json"))
h2 = json.load(open(f"{out_d}/cmp_h200_vs_v4b2.json"))
lat = json.load(open(f"{out_d}/latency_h200_r2.json"))
asm = json.load(open(f"{out_d}/assemble_summary.json"))

summary = {
    "name": "peshtur",
    "base": "Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "merge_bit_identical": mc["ALL_IDENTICAL"],
    "leaks": leaks,
    "parity_r1_vs_v4b2": r1,
    "determinism_r1b_vs_r1": det,
    "parity_h200_live_ocr_vs_v4b2": h2,
    "latency_h200_547": lat,
    "official_vision_03": asm,
    "PASS": bool(
        mc["ALL_IDENTICAL"]
        and r1["same_answer_frac"] >= 0.98
        and det["answers_differ"] == 0
        and asm["complete"] is True
        and asm["rows"] == 12210
        and abs(h2["index_diff"]) <= 0.5
    ),
}
open(f"{out_d}/peshtur_release_summary.json", "w").write(json.dumps(summary, indent=2) + "\n")
print("LEAK_AND_GATE_SCAN", json.dumps({"PASS": summary["PASS"], "full_index": asm["full_index"], "nowg_index": asm["nowg_index"], "wg": asm["winoground_skill"], "r1_same": r1["same_answer_frac"], "det_diff": det["answers_differ"], "h200_diff": h2["index_diff"], "lat_med": lat["median_ms"]}))
assert summary["PASS"], "RELEASE_GATE_FAILED"
PY
rc=$?
[ $rc = 0 ] || { echo "leak or gate check failed" | aws s3 cp --quiet - $P/state/${NAME}_fail; mirror; exit 5; }

# 11. Compute SHA256SUMS of the entire publishable repo tree and upload to S3 (excluding .venv, wheels, __pycache__)
( cd $R && find . -type f ! -path './.venv/*' ! -path './wheels/*' ! -path './__pycache__/*' ! -name '.DS_Store' | sed 's|^\./||' | LC_ALL=C sort | xargs -r sha256sum ) > $O/SHA256SUMS
aws s3 sync --quiet --exclude ".venv/*" --exclude "wheels/*" --exclude "__pycache__/*" $R/ $P/repo/
aws s3 sync --quiet $O/ $P/checks/
up $O/SHA256SUMS $P/SHA256SUMS
up $O/peshtur_release_summary.json $P/peshtur_release_summary.json

# Verify S3 file count & sizes match local repo
python3 - $R $P/repo $O/SHA256SUMS >> $L 2>&1 <<'PY'
import os, subprocess, sys
repo, s3_uri, sums_f = sys.argv[1:4]
want = {}
for l in open(sums_f):
    if l.strip():
        _, f = l.strip().split(None, 1)
        want[f] = os.path.getsize(os.path.join(repo, f))
out = subprocess.check_output(["aws", "s3", "ls", "--recursive", s3_uri + "/"]).decode()
pfx = s3_uri.split("/", 3)[3].rstrip("/") + "/"
got = {}
for l in out.splitlines():
    p = l.split(None, 3)
    if len(p) == 4 and p[3].startswith(pfx):
        got[p[3][len(pfx):]] = int(p[2])
bad = [f for f, sz in want.items() if got.get(f) != sz]
assert not bad, f"S3_SIZE_MISMATCH {bad[:5]}"
print(f"S3_VERIFY_OK files={len(want)} total_bytes={sum(want.values())}")
PY
rc=$?
[ $rc = 0 ] || { echo "s3 verify failed" | aws s3 cp --quiet - $P/state/${NAME}_fail; mirror; exit 5; }

# 12. Clean up restricted Winoground temp files and emit chain_done
rm -rf $E
echo "done $(date -u +%FT%TZ)" | aws s3 cp --quiet - $P/state/${NAME}_chain_done
log "PESHTUR_RELEASE_STAGE_DONE"
mirror
