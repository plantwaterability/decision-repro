#!/bin/bash
# rtx_both_lat.sh <ws_id> <tag>
# Official 1 x NVIDIA RTX PRO 6000 Blackwell latency & clean-install verification for BOTH peshtur (Vision) & eshbartur (Text).
# Runs shipped bash install.sh (empty uv cache, empty HF cache, empty ~/.triton) + sequential 1-GPU latency samples + H200 parity check.
# Publishes NOTHING publicly; writes strictly to S3 .../peshtur/rtx/ and .../eshbartur/rtx/.
set -u
unset ASCALE_RUN_ID HF_TOKEN HUGGING_FACE_HUB_TOKEN
export AWS_DEFAULT_REGION=us-east-2 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1

WS=$1; TAG=$2
B=s3://YOUR_BUCKET/jev-race
V=$B/engineering/vision
VP=$V/v4b/release/peshtur
EP=$B/engineering/text/t4b/release/eshbartur
VR=$VP/rtx
ER=$EP/rtx

D=/data/local/agyrtx
L=$D/logs/rtx_both.log
mkdir -p $D/logs $D/bin $D/peshtur $D/eshbartur $D/suite

log() { echo "[agyrtx $(date -u +%H:%M:%SZ)] $*" >> $L; }
mirror() {
  aws s3 cp --quiet $L $VR/rtx_both.log 2>/dev/null || true
  aws s3 cp --quiet $L $ER/rtx_both.log 2>/dev/null || true
}
fail() {
  log "FAIL: $*"
  echo "$* $(date -u +%FT%TZ)" | aws s3 cp --quiet - $VR/agyrtx_fail
  mirror
  exit 5
}

exec 9> $D/rtx.lock
flock -n 9 || { log "another rtx_both_lat holds lock"; exit 0; }
log "START ws=$WS tag=$TAG host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tr '\n' ' ') disk=$(df -h /data/local | tail -1 | awk '{print $4}')"

aws s3 ls $VR/agyrtx_done > /dev/null 2>&1 && { log "ALREADY_DONE"; exit 0; }
aws s3 ls $VR/agyrtx_fail > /dev/null 2>&1 && { log "FAIL_MARKER_PRESENT"; exit 5; }
[ "$(nvidia-smi -L 2>/dev/null | wc -l)" = "1" ] || fail "expected 1 GPU, saw $(nvidia-smi -L 2>/dev/null | wc -l)"

# 0. OpenCV system deps + uv
[ -e /usr/lib/x86_64-linux-gnu/libGL.so.1 ] || {
  sudo -n env DEBIAN_FRONTEND=noninteractive apt-get update -q > $D/logs/apt.out 2>&1
  sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -q libgl1 libglib2.0-0 >> $D/logs/apt.out 2>&1
  log "APT libgl1/libglib2.0-0 rc=$?"
}
if ! command -v uv >/dev/null 2>&1 && [ ! -x $D/bin/uv ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$D/bin sh > $D/logs/uv_install.log 2>&1
fi
UV_BIN=$(command -v uv 2>/dev/null || echo $D/bin/uv)
[ -x "$UV_BIN" ] || fail "uv missing"
XP="$(dirname "$UV_BIN"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
log "UV=$UV_BIN ($($UV_BIN --version))"

nv_poll() {
  local outf=$1
  ( exec 9>&- </dev/null >/dev/null 2>&1; while sleep 1; do echo "$(date -u +%H:%M:%SZ) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ',')" >> "$outf"; done ) &
  echo $!
}

# ==============================================================================
# PART A: peshtur (Vision <= 4B) on 1 x RTX PRO 6000 (547-row fixed sample)
# ==============================================================================
if ! aws s3 ls $VR/peshtur_rtx_latency.json > /dev/null 2>&1; then
  log "=== PART A: PESHTUR RTX PRO 6000 ==="
  PR=$D/peshtur/repo; mkdir -p $PR $D/peshtur/out
  aws s3 sync --quiet $VP/repo/ $PR/ || fail "peshtur repo sync failed"
  aws s3 cp --quiet $VP/SHA256SUMS $D/peshtur/SHA256SUMS || fail "peshtur SHA256SUMS fetch failed"
  ( cd $PR && sha256sum -c --quiet $D/peshtur/SHA256SUMS ) || fail "peshtur repo sha256 check failed"
  chmod 755 $PR/install.sh
  log "PESHTUR_REPO_SHA_OK ($(wc -l < $D/peshtur/SHA256SUMS) files)"

  # Fetch Vision no-WG suite + build exact 547-row sample (sha1(id) % 20 == 0)
  NOWGSHA=0d44a006e575363aee592dd7a03a5ae45cf095c45eef6ce74079e8a040ef5d3c
  if [ ! -f $D/suite/suite/vision-rows-noWG.jsonl ]; then
    aws s3 cp --quiet $V/suite/suite_noWG.tar $D/suite/
    aws s3 cp --quiet $V/suite/suite_noWG.tar.md5 $D/suite/
    [ "$(awk '{print $1}' $D/suite/suite_noWG.tar.md5)" = "$(md5sum < $D/suite/suite_noWG.tar | cut -d' ' -f1)" ] || fail "suite_noWG md5 bad"
    tar xf $D/suite/suite_noWG.tar -C $D/suite && rm -f $D/suite/suite_noWG.tar
    [ -f $D/suite/suite/vision-rows-noWG.jsonl ] || aws s3 cp --quiet $V/dormice/gate/vision-rows-noWG.jsonl $D/suite/suite/vision-rows-noWG.jsonl
  fi
  [ "$(sha256sum $D/suite/suite/vision-rows-noWG.jsonl | cut -d' ' -f1)" = "$NOWGSHA" ] || fail "vision-rows-noWG sha bad"
  mkdir -p $D/peshtur/lat_in
  python3 - $D/suite/suite/vision-rows-noWG.jsonl $D/peshtur/lat_in/vision-rows.jsonl <<'PY'
import hashlib, json, sys
rows = [l for l in open(sys.argv[1]) if l.strip()]
sel = [l for l in rows if int(hashlib.sha1(json.loads(l)["id"].encode()).hexdigest(), 16) % 20 == 0]
open(sys.argv[2], "w").writelines(sel)
assert len(sel) == 547, len(sel)
PY
  ln -sfn $D/suite/suite/full $D/peshtur/lat_in/full

  # Shipped install.sh from empty uv cache + empty HF cache + empty HOME
  rm -rf $D/peshtur/uvcache $D/peshtur/hf $D/peshtur/home_inst $D/peshtur/home_run
  mkdir -p $D/peshtur/uvcache $D/peshtur/hf $D/peshtur/home_inst $D/peshtur/home_run
  t0=$(date +%s)
  ( cd $PR && env -i PATH="$XP" HOME="$D/peshtur/home_inst" UV_CACHE_DIR="$D/peshtur/uvcache" UV_NO_CONFIG=1 HF_HOME="$D/peshtur/hf" CUDA_VISIBLE_DEVICES=0 bash ./install.sh ) > $D/logs/peshtur_install.out 2>&1; rc=$?
  if [ $rc != 0 ]; then
    log "PESHTUR_INSTALL_RETRY_S3_WHL rc=$rc $(tail -n 2 $D/logs/peshtur_install.out | tr '\n' ' ' | cut -c1-250)"
    WH=paddlepaddle_gpu-3.2.0-cp312-cp312-linux_x86_64.whl
    mkdir -p $PR/wheels && aws s3 cp --quiet $V/usu_release/wheels/$WH $PR/wheels/$WH
    ( cd $PR && env -i PATH="$XP" HOME="$D/peshtur/home_inst" UV_CACHE_DIR="$D/peshtur/uvcache" UV_NO_CONFIG=1 HF_HOME="$D/peshtur/hf" CUDA_VISIBLE_DEVICES=0 bash ./install.sh ) > $D/logs/peshtur_install2.out 2>&1; rc=$?
    log "PESHTUR_INSTALL2 rc=$rc $(( $(date +%s) - t0 ))s $(tail -n 1 $D/logs/peshtur_install2.out | cut -c1-250)"
    [ $rc = 0 ] || fail "peshtur install.sh failed on RTX PRO 6000"
  else
    log "PESHTUR_INSTALL_OK $(( $(date +%s) - t0 ))s $(grep -a 'install.sh:' $D/logs/peshtur_install.out | tr '\n' ';' | cut -c1-250) | $(tail -n 1 $D/logs/peshtur_install.out | cut -c1-250)"
  fi
  mirror

  # Sequential 1-GPU latency run on 547-row Vision sample (fresh HOME -> cold Triton cache!)
  rm -rf $D/peshtur/lat_out $D/logs/gpumem_peshtur.log
  mkdir -p $D/peshtur/lat_out
  MP=$(nv_poll $D/logs/gpumem_peshtur.log)
  ( cd $PR && env -i PATH="$XP" HOME="$D/peshtur/home_run" UV_CACHE_DIR="$D/peshtur/uvcache" UV_NO_CONFIG=1 HF_HOME="$D/peshtur/hf" HF_HUB_OFFLINE=1 \
      OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
      uv run --frozen python -m decision_index vision run --dir $D/peshtur/lat_in --engine peshtur:Engine --workers 1 --chunk 50 --no-verify --out $D/peshtur/lat_out ) \
      > $D/logs/peshtur_lat.out 2>&1; rc=$?
  kill "$MP" 2>/dev/null || true
  log "PESHTUR_LAT_RUN rc=$rc $(tail -n 1 $D/logs/peshtur_lat.out | cut -c1-250)"
  [ $rc = 0 ] || fail "peshtur RTX latency run failed"

  # Summarize latency + compare RTX PRO 6000 answers vs H200 run
  $PR/.venv/bin/python - $D/peshtur/lat_out/results.jsonl $D/peshtur/lat_out/environment.json $D/logs/gpumem_peshtur.log $PR/runs/peshtur-vision-0.3/results.jsonl $D/peshtur/lat_in/vision-rows.jsonl $D/peshtur/out <<'PY' >> $L 2>&1
import json, os, statistics, sys
from pathlib import Path
from decision_index.vision.score import score_run

res_f, env_f, mem_f, h200_f, rows_f, out_d = sys.argv[1:7]
rows = [json.loads(l) for l in open(res_f) if l.strip()]
ok = [r for r in rows if r.get("status") == "ok"]
assert len(ok) == 547 and len(rows) == 547, (len(ok), len(rows))
ms = sorted(float(r["total_wall_ms"]) for r in ok)
n = len(ms)
q = lambda f: ms[min(n - 1, int(f * n))]
env = json.load(open(env_f))
mem = [int(l.split()[1]) for l in open(mem_f) if len(l.split()) >= 2 and l.split()[1].isdigit()] if os.path.exists(mem_f) else []
raw = [r.get("raw_output") or {} for r in ok]
by = {}
for r in ok:
    by.setdefault(r["benchmark"], []).append(float(r["total_wall_ms"]))
med = lambda v: statistics.median(v) if v else None
lat = {
    "model": "peshtur",
    "hardware": env.get("gpu"),
    "n": n,
    "errors": 0,
    "median_ms": round(statistics.median(ms), 2),
    "mean_ms": round(sum(ms) / n, 2),
    "p80_ms": round(q(0.8), 2),
    "p95_ms": round(q(0.95), 2),
    "p99_ms": round(q(0.99), 2),
    "max_ms": round(ms[-1], 2),
    "over_1000_ms": sum(x > 1000 for x in ms),
    "board_rule_each_le_1000": statistics.median(ms) <= 1000 and (sum(ms) / n) <= 1000 and q(0.8) <= 1000,
    "ocr_ms_median": round(med([x["ocr_ms"] for x in raw if "ocr_ms" in x]), 2),
    "engine_ms_median": round(med([x["engine_ms"] for x in raw if "engine_ms" in x]), 2),
    "loaded_seconds": round(env.get("loaded_seconds", 0.0), 2),
    "gpu_mem_max_mib": max(mem) if mem else None,
    "by_benchmark_median_ms": {b: round(statistics.median(v), 1) for b, v in sorted(by.items())},
}
Path(f"{out_d}/peshtur_rtx_latency.json").write_text(json.dumps(lat, indent=2) + "\n")

# Compare answers vs H200 scored run
rtx_m = {r["run_id"]: r for r in ok}
h200_m = {}
for l in open(h200_f):
    if l.strip():
        r = json.loads(l)
        if r["run_id"] in rtx_m:
            h200_m[r["run_id"]] = r
diff, dps = 0, []
for rid, rn in rtx_m.items():
    rh = h200_m[rid]
    an = rn["response"]["answers"]["answer"]
    ah = rh["response"]["answers"]["answer"]
    if an["choice"] != ah["choice"]:
        diff += 1
    dps.append(max(abs(an["probabilities"][k] - ah["probabilities"][k]) for k in an["probabilities"]))
dps.sort()
Path(f"{out_d}/h200_sub.jsonl").write_text("\n".join(json.dumps(h200_m[rid]) for rid in rtx_m) + "\n")
s_rtx = score_run(Path(rows_f), Path(res_f))["public_vision_index"]
s_h200 = score_run(Path(rows_f), Path(f"{out_d}/h200_sub.jsonl"))["public_vision_index"]
cmp = {
    "common": len(rtx_m),
    "answers_differ": diff,
    "same_answer_frac": round((len(rtx_m) - diff) / len(rtx_m), 6),
    "med_abs_dp": round(dps[len(dps) // 2], 6),
    "p95_abs_dp": round(dps[int(0.95 * len(dps))], 6),
    "max_abs_dp": round(dps[-1], 6),
    "index_rtx_547": round(s_rtx, 4),
    "index_h200_547": round(s_h200, 4),
    "index_diff": round(s_rtx - s_h200, 4),
}
Path(f"{out_d}/peshtur_rtx_vs_h200.json").write_text(json.dumps(cmp, indent=2) + "\n")
print("PESHTUR_RTX_LAT", json.dumps(lat))
print("PESHTUR_RTX_VS_H200", json.dumps(cmp))
PY
  rc=$?; [ $rc = 0 ] || fail "peshtur RTX summary failed"

  aws s3 cp --quiet $D/peshtur/out/peshtur_rtx_latency.json $VR/peshtur_rtx_latency.json
  aws s3 cp --quiet $D/peshtur/out/peshtur_rtx_vs_h200.json $VR/peshtur_rtx_vs_h200.json
  aws s3 cp --quiet $D/peshtur/lat_out/environment.json $VR/environment.json
  aws s3 cp --quiet $D/peshtur/lat_out/results.jsonl $VR/results_547.jsonl
  log "PART_A_PESHTUR_RTX_UPLOADED"
  mirror
  # Free disk before Part B
  rm -rf $D/peshtur/repo/.venv $D/peshtur/uvcache $D/suite
fi

# ==============================================================================
# PART B: eshbartur (Text <= 4B) on 1 x RTX PRO 6000 (1,000-row public sample)
# ==============================================================================
if ! aws s3 ls $ER/eshbartur_rtx_latency.json > /dev/null 2>&1; then
  log "=== PART B: ESHBARTUR RTX PRO 6000 ==="
  ER_REPO=$D/eshbartur/repo; mkdir -p $ER_REPO $D/eshbartur/out $D/eshbartur/ref
  aws s3 sync --quiet $EP/repo/ $ER_REPO/ || fail "eshbartur repo sync failed"
  aws s3 cp --quiet $EP/SHA256SUMS_model $D/eshbartur/SHA256SUMS_model || fail "eshbartur SHA256SUMS_model fetch failed"
  ( cd $ER_REPO && sha256sum -c --quiet $D/eshbartur/SHA256SUMS_model ) || fail "eshbartur repo sha256 check failed"
  chmod 755 $ER_REPO/install.sh
  log "ESHBARTUR_REPO_SHA_OK ($(wc -l < $D/eshbartur/SHA256SUMS_model) files)"

  aws s3 cp --quiet $EP/ref/lat_sample_1000.jsonl.gz $D/eshbartur/ref/lat_sample_1000.jsonl.gz || fail "lat_sample_1000 fetch failed"
  [ "$(sha256sum < $D/eshbartur/ref/lat_sample_1000.jsonl.gz | cut -c1-16)" = "5548d2d2d9a4cb6d" ] || fail "lat_sample_1000 sha bad"
  aws s3 cp --quiet $EP/dataset/runs/eshbartur/results.jsonl.gz $D/eshbartur/ref/h200_results.jsonl.gz || fail "h200 results fetch failed"

  # Shipped install.sh from empty uv cache + empty HF cache + empty HOME
  rm -rf $D/eshbartur/uvcache $D/eshbartur/hf $D/eshbartur/home_inst $D/eshbartur/home_run
  mkdir -p $D/eshbartur/uvcache $D/eshbartur/hf $D/eshbartur/home_inst $D/eshbartur/home_run
  t0=$(date +%s)
  ( cd $ER_REPO && env -i PATH="$XP" HOME="$D/eshbartur/home_inst" UV_CACHE_DIR="$D/eshbartur/uvcache" UV_NO_CONFIG=1 HF_HOME="$D/eshbartur/hf" CUDA_VISIBLE_DEVICES=0 bash ./install.sh ) > $D/logs/eshbartur_install.out 2>&1; rc=$?
  log "ESHBARTUR_INSTALL rc=$rc $(( $(date +%s) - t0 ))s $(tail -n 1 $D/logs/eshbartur_install.out | cut -c1-250)"
  [ $rc = 0 ] || fail "eshbartur install.sh failed on RTX PRO 6000"
  mirror

  # Sequential 1-GPU latency run on 1,000-row public board Text sample (fresh HOME -> cold Triton cache!)
  rm -rf $D/eshbartur/lat_out $D/logs/gpumem_eshbartur.log
  mkdir -p $D/eshbartur/lat_out
  MP=$(nv_poll $D/logs/gpumem_eshbartur.log)
  ( cd $ER_REPO && env -i PATH="$XP" HOME="$D/eshbartur/home_run" UV_CACHE_DIR="$D/eshbartur/uvcache" UV_NO_CONFIG=1 HF_HOME="$D/eshbartur/hf" HF_HUB_OFFLINE=1 \
      OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
      uv run --frozen python -m decision_index run --rows $D/eshbartur/ref/lat_sample_1000.jsonl.gz --engine eshbartur:Engine --compact --out $D/eshbartur/lat_out ) \
      > $D/logs/eshbartur_lat.out 2>&1; rc=$?
  kill "$MP" 2>/dev/null || true
  log "ESHBARTUR_LAT_RUN rc=$rc $(tail -n 1 $D/logs/eshbartur_lat.out | cut -c1-250)"
  [ $rc = 0 ] || fail "eshbartur RTX latency run failed"

  # Summarize latency + compare RTX PRO 6000 answers vs H200 140k scored run
  $ER_REPO/.venv/bin/python - $D/eshbartur/lat_out/results.jsonl $D/eshbartur/lat_out/environment.json $D/logs/gpumem_eshbartur.log $D/eshbartur/ref/h200_results.jsonl.gz $D/eshbartur/out <<'PY' >> $L 2>&1
import gzip, json, os, statistics, sys
from pathlib import Path

res_f, env_f, mem_f, h200_gz, out_d = sys.argv[1:6]
rows = [json.loads(l) for l in open(res_f) if l.strip()]
ok = [r for r in rows if r.get("status") == "ok"]
assert len(ok) == 1000 and len(rows) == 1000, (len(ok), len(rows))
ms = sorted(float(r["total_wall_ms"]) for r in ok)
n = len(ms)
q = lambda f: ms[min(n - 1, int(f * n))]
env = json.load(open(env_f))
mem = [int(l.split()[1]) for l in open(mem_f) if len(l.split()) >= 2 and l.split()[1].isdigit()] if os.path.exists(mem_f) else []
by_fam = {}
for r in ok:
    fam = r.get("dataset") or r.get("benchmark") or r.get("family") or str(r.get("run_id", "")).split(":")[0]
    by_fam.setdefault(fam, []).append(float(r["total_wall_ms"]))
lat = {
    "model": "eshbartur",
    "hardware": env.get("gpu"),
    "n": n,
    "errors": 0,
    "median_ms": round(statistics.median(ms), 2),
    "mean_ms": round(sum(ms) / n, 2),
    "p80_ms": round(q(0.8), 2),
    "p95_ms": round(q(0.95), 2),
    "p99_ms": round(q(0.99), 2),
    "max_ms": round(ms[-1], 2),
    "over_1000_ms": sum(x > 1000 for x in ms),
    "loaded_seconds": round(env.get("loaded_seconds", 0.0), 2),
    "gpu_mem_max_mib": max(mem) if mem else None,
    "by_family_median_ms": {b: round(statistics.median(v), 1) for b, v in sorted(by_fam.items())},
}
Path(f"{out_d}/eshbartur_rtx_latency.json").write_text(json.dumps(lat, indent=2) + "\n")

# Compare answers vs H200 full run
rtx_m = {r["run_id"]: r["response"]["answers"] for r in ok}
h200_m = {}
with gzip.open(h200_gz, "rt", encoding="utf-8") as fh:
    for l in fh:
        r = json.loads(l)
        if r["run_id"] in rtx_m:
            h200_m[r["run_id"]] = r["response"]["answers"]
n_q, n_ch, ch_diff, dps = 0, 0, 0, []
for rid, an in rtx_m.items():
    if rid not in h200_m:
        continue
    ah = h200_m[rid]
    for qk, a in an.items():
        b = ah[qk]
        n_q += 1
        if a["type"] == "choice":
            n_ch += 1
            if a["choice"] != b["choice"]:
                ch_diff += 1
            dps.append(max(abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["probabilities"]))
        elif a["type"] == "noul":
            dps.append(abs(a["noul"] - b["noul"]))
        elif a["type"] == "score":
            dps.append(max(abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["probabilities"]))
dps.sort()
cmp = {
    "rows_matched": len(h200_m),
    "questions": n_q,
    "choices": n_ch,
    "choices_differ": ch_diff,
    "choice_same_frac": round((n_ch - ch_diff) / n_ch, 6) if n_ch else 1.0,
    "med_abs_dp": round(dps[len(dps) // 2], 6) if dps else 0.0,
    "p95_abs_dp": round(dps[int(0.95 * len(dps))], 6) if dps else 0.0,
    "max_abs_dp": round(dps[-1], 6) if dps else 0.0,
}
Path(f"{out_d}/eshbartur_rtx_vs_h200.json").write_text(json.dumps(cmp, indent=2) + "\n")
print("ESHBARTUR_RTX_LAT", json.dumps(lat))
print("ESHBARTUR_RTX_VS_H200", json.dumps(cmp))
PY
  rc=$?; [ $rc = 0 ] || fail "eshbartur RTX summary failed"

  aws s3 cp --quiet $D/eshbartur/out/eshbartur_rtx_latency.json $ER/eshbartur_rtx_latency.json
  aws s3 cp --quiet $D/eshbartur/out/eshbartur_rtx_vs_h200.json $ER/eshbartur_rtx_vs_h200.json
  aws s3 cp --quiet $D/eshbartur/lat_out/environment.json $ER/environment.json
  aws s3 cp --quiet $D/eshbartur/lat_out/results.jsonl $ER/results_1000.jsonl
  log "PART_B_ESHBARTUR_RTX_UPLOADED"
  mirror
fi

echo "done $(date -u +%FT%TZ)" | aws s3 cp --quiet - $VR/agyrtx_done
log "ALL_RTX_LATENCY_DONE"
mirror
