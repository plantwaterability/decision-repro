#!/bin/bash
# t4b_sweep_chain.sh <ws_id> <tag> <arm_name>
# Fast (~25 min) zero-training checkpoint/soup + temperature calibration sweep on t4b3.
# Stage 1 (~12 min): 4 orthogonal candidates (ck1500, ck2000, soup_20_35, wise_080) on full unseen gde1v3s (10,899) + dev (7,551).
# Stage 2 (~10 min): Top 2 candidates on full 4,400 Text screen scored alongside t4b3 (ckpt_03516), q38p1, kudhab, munku, E1.
# Stage 3 (~5 sec): Fit scalar readout temperature T on dev half0 -> verify NLL & ECE on dev half1 + gde1v3s for every candidate.
# Lock fd 9 is already held by t4b_chain.sh via exec. Zero bare `wait`.
set -u
unset ASCALE_RUN_ID HF_TOKEN HUGGING_FACE_HUB_TOKEN
export AWS_DEFAULT_REGION=us-east-2 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WS=$1; TAG=$2; NAME=$3
B=s3://YOUR_BUCKET/jev-race
E=$B/engineering/esbar
ER=$E/runs
TZ=$B/engineering/text/tz
T4B=$B/engineering/text/t4b
U=$T4B/arms/$NAME
SRC_ARM=$T4B/arms/t4b3

W=/data/local/t4b
V=$W/venv_t4b
C=$W/code
R=$W/kroot
M=$W/models/qwen35_4b
CK=$W/sweep_ckpts/t4b3
OUT=$W/sweep/$NAME
L=$W/logs/chain_$NAME.log
mkdir -p $W/logs $C $R/src $M $CK $OUT/unseen $OUT/screen

log() { echo "[$NAME $(date -u +%H:%M:%SZ)] $*" >> $L; }
mirror() { aws s3 cp --quiet $L $U/logs/chain_$NAME.$TAG.log 2>/dev/null || true; }

log "SWEEP_START ws=$WS tag=$TAG host=$(hostname) gpus=$(nvidia-smi -L 2>/dev/null | wc -l)"
mirror

MREPO=Qwen/Qwen3.5-4B
MREV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a

# 1. Restage venv, kit/eval refs, Qwen3.5-4B, and the 3 small t4b3 checkpoints (NO 445k train data needed!)
if [ ! -f $W/SWEEP_RESTAGE_OK ]; then
  pids=()
  (
    probe() {
      $V/bin/python -c "
import torch, transformers, peft, fla, safetensors, accelerate, numpy, httpx
assert torch.__version__.startswith('2.14.1') and transformers.__version__ == '5.23.0'
torch.zeros(1).cuda(); print('VENV_OK', torch.__version__, transformers.__version__)"
    }
    if ! probe > $W/logs/venv_probe.log 2>&1; then
      python3 -m venv --clear $V
      $V/bin/pip install -q "torch==2.14.1+cu130" --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple > $W/logs/venv.log 2>&1
      $V/bin/pip install -q "transformers==5.23.0" "peft==0.21.3" "flash-linear-attention==0.5.2" "fla-core==0.5.2" "safetensors>=0.6" "accelerate>=1.10" numpy httpx pillow sentencepiece protobuf tiktoken jinja2 einops "huggingface_hub>=1.0" >> $W/logs/venv.log 2>&1
      probe > $W/logs/venv_probe.log 2>&1
    fi
    grep -q "^VENV_OK " $W/logs/venv_probe.log && touch $W/VENV_OK
  ) & pids+=($!)

  (
    if [ ! -f $R/.ok ]; then
      aws s3 cp --quiet $B/fleetq/code/di03/kit_9eb2dbe2a358004c8782c66e40a83ac07b953fec.tgz $R/src/kit.tgz
      aws s3 cp --quiet $E/heldout/rebuilt_2026-11-02/suite-0.3.tgz $R/src/suite.tgz
      mkdir -p $R/kit && tar xzf $R/src/kit.tgz -C $R/kit && find $R/kit -name '._*' -delete && tar xzf $R/src/suite.tgz -C $R
      aws s3 cp --quiet $ER/esbar_v8/public_di_full/code/v8f_code_r2.tgz $R/src/v8f_code_r2.tgz
      mkdir -p $R/gc && tar xzf $R/src/v8f_code_r2.tgz -C $R/gc gate_index.py
      aws s3 cp --quiet $ER/esbar_tb1/sample/sample4400-rows.jsonl.gz $R/src/s4400_rows.jsonl.gz
      aws s3 cp --quiet $E/public_E1/e1_20261104T1251Z/final/results.jsonl.gz $R/src/e1_pub.jsonl.gz
      aws s3 cp --quiet $ER/esbar_wi_fs1f50/public_di_full/wi_fs1f50_full_20261112/final/results.jsonl.gz $R/src/munku_pub.jsonl.gz
      aws s3 cp --quiet $ER/esbar_tx/screens/kudhab_t5/results.jsonl.gz $R/src/kudhab_t5.jsonl.gz 2>/dev/null || true
      aws s3 cp --quiet $ER/esbar_tx/screens/q38p1/results.jsonl.gz $R/src/q38p1.jsonl.gz 2>/dev/null || true
      mkdir -p $R/unseen
      for s in gde1v3s dev; do
        aws s3 cp --quiet $TZ/unseen/$s.req.jsonl.gz $R/unseen/$s.req.jsonl.gz
        for m in karpo kudhab lg; do
          aws s3 cp --quiet $TZ/unseen/reads/$m/$s.p.jsonl.gz $R/unseen/$s.$m.p.jsonl.gz 2>/dev/null || true
        done
      done
      [ "$(sha256sum < $R/src/s4400_rows.jsonl.gz | cut -c1-16)" = "6296ba7ae4f9269f" ] && [ -s $R/unseen/gde1v3s.req.jsonl.gz ] && [ -s $R/unseen/dev.req.jsonl.gz ] && touch $R/.ok
    fi
  ) & pids+=($!)

  (
    if [ ! -f $M/MODEL_OK ]; then
      python3 -m pip install -q --user "huggingface_hub>=1.0" > /dev/null 2>&1 || true
      python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MREPO', revision='$MREV', local_dir='$M', token=False, max_workers=16)" > $W/logs/model.log 2>&1
      [ -s $M/model.safetensors.index.json ] && touch $M/MODEL_OK
    fi
  ) & pids+=($!)

  (
    for c in ckpt_01500 ckpt_02000 ckpt_03516; do
      mkdir -p $CK/$c
      aws s3 sync --quiet $SRC_ARM/ckpt/$c/ $CK/$c/ --exclude "optim.pt"
    done
    aws s3 cp --quiet $SRC_ARM/screen/results.jsonl.gz $OUT/screen/t4b3_ck3516.results.jsonl.gz
    aws s3 cp --quiet $SRC_ARM/unseen/gde1v3s.t4b3.p.jsonl.gz $OUT/unseen/gde1v3s.t4b3.p.jsonl.gz
    aws s3 cp --quiet $SRC_ARM/unseen/dev.t4b3.p.jsonl.gz $OUT/unseen/dev.t4b3.p.jsonl.gz
    [ -s $CK/ckpt_01500/readout.pt ] && [ -s $CK/ckpt_02000/readout.pt ] && [ -s $CK/ckpt_03516/readout.pt ] && touch $CK/.ok
  ) & pids+=($!)

  for pid in "${pids[@]}"; do wait "$pid" || true; done
  E_OK=$([ -f $W/VENV_OK ] && echo 1 || echo 0)
  K_OK=$([ -f $R/.ok ] && echo 1 || echo 0)
  M_OK=$([ -f $M/MODEL_OK ] && echo 1 || echo 0)
  C_OK=$([ -f $CK/.ok ] && echo 1 || echo 0)
  log "SWEEP_RESTAGE venv=$E_OK kit=$K_OK model=$M_OK ckpts=$C_OK"
  mirror
  [ "$E_OK$K_OK$M_OK$C_OK" = "1111" ] || exit 4
  touch $W/SWEEP_RESTAGE_OK
fi

export PYTHONPATH=$C:$R/kit
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Candidate definitions: name|ckpt_spec
CANDS=(
  "ck1500|$CK/ckpt_01500:1.0"
  "ck2000|$CK/ckpt_02000:1.0"
  "soup_20_35|$CK/ckpt_02000:0.5;$CK/ckpt_03516:0.5"
  "wise_080|$CK/ckpt_03516:0.8"
)

# 2. Stage 1 (~12 min): Run all 4 candidates on FULL unseen gde1v3s (10,899) + dev (7,551)
UDIR=$OUT/unseen
mkdir -p $UDIR/work
for s in gde1v3s dev; do
  cp $R/unseen/$s.req.jsonl.gz $UDIR/work/$s.req.jsonl.gz
  for m in karpo kudhab lg; do
    [ -s $R/unseen/$s.$m.p.jsonl.gz ] && cp $R/unseen/$s.$m.p.jsonl.gz $UDIR/work/$s.$m.p.jsonl.gz
  done
  cp $UDIR/$s.t4b3.p.jsonl.gz $UDIR/work/$s.t4b3.p.jsonl.gz
  if [ ! -s $UDIR/work/${s}_0.req.jsonl.gz ]; then
    $V/bin/python - $UDIR/work/$s.req.jsonl.gz $UDIR/work $s 8 <<'PY'
import gzip, sys
rows = [r for r in gzip.open(sys.argv[1], "rt", encoding="utf-8").read().split("\n") if r]
n = int(sys.argv[4])
fs = [gzip.open(f"{sys.argv[2]}/{sys.argv[3]}_{k}.req.jsonl.gz", "wt", encoding="utf-8") for k in range(n)]
for i, r in enumerate(rows): fs[i % n].write(r + "\n")
for f in fs: f.close()
PY
  fi
done

for entry in "${CANDS[@]}"; do
  CN="${entry%%|*}"
  CS="${entry#*|}"
  if aws s3 ls $U/unseen/dev.$CN.p.jsonl.gz > /dev/null 2>&1 && aws s3 ls $U/unseen/gde1v3s.$CN.p.jsonl.gz > /dev/null 2>&1; then
    aws s3 cp --quiet $U/unseen/gde1v3s.$CN.p.jsonl.gz $UDIR/work/gde1v3s.$CN.p.jsonl.gz
    aws s3 cp --quiet $U/unseen/dev.$CN.p.jsonl.gz $UDIR/work/dev.$CN.p.jsonl.gz
    log "STAGE1_CACHED $CN"
    continue
  fi
  log "STAGE1_START $CN spec=$CS"
  mirror
  pids=()
  for g in 0 1 2 3 4 5 6 7; do
    (
      for s in gde1v3s dev; do
        CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$C:$R/kit HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
          $V/bin/python $C/t4b_eval.py --mode unseen --model $M --ckpt-spec "$CS" \
          --req $UDIR/work/${s}_$g.req.jsonl.gz --out $UDIR/work/${s}_$g.$CN.jsonl \
          > $UDIR/work/${s}_$g.$CN.log 2>&1
      done
    ) & pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
  for s in gde1v3s dev; do
    cat $UDIR/work/${s}_*.$CN.jsonl | gzip -n > $UDIR/work/$s.$CN.p.jsonl.gz
    aws s3 cp --quiet $UDIR/work/$s.$CN.p.jsonl.gz $U/unseen/$s.$CN.p.jsonl.gz
  done
  log "STAGE1_DONE $CN gde=$(zcat $UDIR/work/gde1v3s.$CN.p.jsonl.gz | wc -l) dev=$(zcat $UDIR/work/dev.$CN.p.jsonl.gz | wc -l)"
  mirror
done

$V/bin/python $C/tz_unseen_score.py $UDIR/work $UDIR/unseen_score.json > $UDIR/unseen_score.txt 2>&1 || true
aws s3 cp --quiet $UDIR/unseen_score.json $U/unseen/unseen_score.json
aws s3 cp --quiet $UDIR/unseen_score.txt $U/unseen/unseen_score.txt
log "UNSEEN_SCORED $(grep -E '^(==|   ck|   soup|   wise|   t4b3|   lg )' $UDIR/unseen_score.txt | tr '\n' ' ' | cut -c1-450)"
mirror

# Pick top 2 candidates by combined unseen micro accuracy (gde1v3s + dev)
TOP2=$($V/bin/python - $UDIR/unseen_score.json <<'PY'
import json, sys
u = json.load(open(sys.argv[1]))
cands = ["ck1500", "ck2000", "soup_20_35", "wise_080"]
scored = []
for c in cands:
    g = u["gde1v3s"]["table"].get(c, {}).get("micro", 0.0)
    d = u["dev"]["table"].get(c, {}).get("micro", 0.0)
    scored.append(((g + d) / 2.0, c))
scored.sort(reverse=True)
print(" ".join(c for _, c in scored[:2]))
PY
)
log "STAGE2_TOP2 $TOP2"
mirror

# 3. Stage 2 (~10 min): Run Top 2 candidates on full 4,400 Text screen & score together with t4b3 + refs
SDIR=$OUT/screen
mkdir -p $SDIR/src $SDIR/run
if [ ! -s $SDIR/src/s_0.jsonl.gz ]; then
  $V/bin/python - $R/src/s4400_rows.jsonl.gz $SDIR/src 8 <<'PY'
import gzip, sys
rows = [r for r in gzip.open(sys.argv[1], "rt", encoding="utf-8").read().split("\n") if r]
n = int(sys.argv[3])
fs = [gzip.open(f"{sys.argv[2]}/s_{k}.jsonl.gz", "wt", encoding="utf-8") for k in range(n)]
for i, r in enumerate(rows): fs[i % n].write(r + "\n")
for f in fs: f.close()
PY
fi

for CN in $TOP2; do
  CS=""
  for entry in "${CANDS[@]}"; do
    [ "${entry%%|*}" = "$CN" ] && CS="${entry#*|}"
  done
  if aws s3 ls $U/screen/$CN.results.jsonl.gz > /dev/null 2>&1; then
    aws s3 cp --quiet $U/screen/$CN.results.jsonl.gz $SDIR/$CN.results.jsonl.gz
    log "STAGE2_CACHED $CN"
    continue
  fi
  log "STAGE2_START $CN spec=$CS"
  mirror
  pids=()
  for g in 0 1 2 3 4 5 6 7; do
    (
      mkdir -p $SDIR/run/$CN/$g
      cd $SDIR/run/$CN/$g
      CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$C:$R/kit HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        $V/bin/python -m decision_index run --engine t4b_engine:T4BEngine \
        --option model=$M --option ckpt_spec="$CS" --option max_len=65536 \
        --edition 0.3 --suite-dir $R/suite-0.3 --rows $SDIR/src/s_$g.jsonl.gz \
        --out $SDIR/run/$CN/$g/out --compact > $SDIR/run/$CN/$g/kit.log 2>&1
    ) & pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
  cat $SDIR/run/$CN/*/out/results.jsonl | gzip -n > $SDIR/$CN.results.jsonl.gz
  aws s3 cp --quiet $SDIR/$CN.results.jsonl.gz $U/screen/$CN.results.jsonl.gz
  log "STAGE2_DONE $CN rows=$(zcat $SDIR/$CN.results.jsonl.gz | wc -l)"
  mirror
done

# Score all together in ONE gate_index.py pass (full + h0 + h1)
$V/bin/python - $R/src/s4400_rows.jsonl.gz $SDIR <<'PY'
import gzip, hashlib, json, sys
ids = [json.loads(l)["_evaluation"]["run_id"] for l in gzip.open(sys.argv[1], "rt")]
for h in (0, 1):
    open(f"{sys.argv[2]}/h{h}.txt", "w").write("".join(i + "\n" for i in ids if int(hashlib.sha1(i.encode()).hexdigest(), 16) % 2 == h))
PY
ARMS="E1=$R/src/e1_pub.jsonl.gz munku=$R/src/munku_pub.jsonl.gz"
[ -s $R/src/kudhab_t5.jsonl.gz ] && ARMS="$ARMS kudhab=$R/src/kudhab_t5.jsonl.gz"
[ -s $R/src/q38p1.jsonl.gz ] && ARMS="$ARMS q38p1=$R/src/q38p1.jsonl.gz"
ARMS="$ARMS t4b3=$SDIR/t4b3_ck3516.results.jsonl.gz"
for CN in $TOP2; do
  [ -s $SDIR/$CN.results.jsonl.gz ] && ARMS="$ARMS $CN=$SDIR/$CN.results.jsonl.gz"
done
: > $SDIR/cmp.txt
for sp in full h0 h1; do
  IDS=$R/src/s4400_rows.jsonl.gz
  [ "$sp" = "full" ] || IDS=$SDIR/$sp.txt
  PYTHONPATH=$R/kit $V/bin/python $R/gc/gate_index.py $R/suite-0.3 $IDS $SDIR/w_$sp $SDIR/$sp.json $ARMS 2>&1 \
    | grep -E '^(INDEX|DIFF)' | sed "s/^/$sp /" >> $SDIR/cmp.txt || true
done
aws s3 cp --quiet $SDIR/cmp.txt $U/screen/cmp.txt
for sp in full h0 h1; do [ -s $SDIR/$sp.json ] && aws s3 cp --quiet $SDIR/$sp.json $U/screen/$sp.json; done
log "STAGE2_SCORED $(grep '^full INDEX ' $SDIR/cmp.txt | tr '\n' ' ' | cut -c1-450)"
mirror

# 4. Stage 3 (~5s CPU): Fit scalar readout temperature T on dev half0 -> verify NLL & 15-bin ECE on dev half1 & gde1v3s + aggregate t4b3s_result.json
$V/bin/python - $UDIR/work $UDIR/unseen_score.json $SDIR $OUT/${NAME}_result.json "$TOP2" <<'PY'
import gzip, json, math, os, sys, time
from pathlib import Path

udir, ujson, sdir, outp, top2_s = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
top2 = top2_s.split()
uj = json.load(open(ujson))

def load_gz(p):
    return [json.loads(l) for l in gzip.open(p, "rt", encoding="utf-8")]

def eval_cal(req_map, probs_map, ids, T):
    nll, brier, hit, n = 0.0, 0.0, 0, 0
    bins = [[0, 0.0, 0.0] for _ in range(15)]
    for i in ids:
        if i not in probs_map:
            continue
        r = req_map[i]
        ks = r["keys"]
        g = r["gold"]
        gold_idx = max(range(len(ks)), key=lambda j: (g.get(ks[j], 0.0), -j))
        p0 = [max(1e-12, float(probs_map[i][k])) for k in ks]
        # Temperature scaling on log-probabilities: p_T \propto p0^(1/T)
        inv_t = 1.0 / T
        lps = [math.log(x) * inv_t for x in p0]
        m = max(lps)
        ex = [math.exp(x - m) for x in lps]
        s = sum(ex)
        pt = [x / s for x in ex]
        nll += -math.log(max(1e-12, pt[gold_idx]))
        brier += sum((pt[j] - (1.0 if j == gold_idx else 0.0)) ** 2 for j in range(len(ks)))
        pred_idx = max(range(len(ks)), key=lambda j: (pt[j], -j))
        ok = 1 if pred_idx == gold_idx else 0
        hit += ok
        conf = pt[pred_idx]
        bi = min(14, int(conf * 15))
        bins[bi][0] += 1
        bins[bi][1] += conf
        bins[bi][2] += ok
        n += 1
    ece = sum(abs(c[2] - c[1]) for c in bins if c[0] > 0) / max(1, n)
    return {
        "n": n,
        "acc": round(100.0 * hit / max(1, n), 3),
        "nll": round(nll / max(1, n), 4),
        "brier": round(brier / max(1, n), 4),
        "ece": round(ece, 4),
    }

req_dev = {r["id"]: r for r in load_gz(f"{udir}/dev.req.jsonl.gz")}
req_gde = {r["id"]: r for r in load_gz(f"{udir}/gde1v3s.req.jsonl.gz")}
dev_h0 = [i for i, r in req_dev.items() if r["half"] == 0]
dev_h1 = [i for i, r in req_dev.items() if r["half"] == 1]
gde_all = list(req_gde.keys())

cal_summary = {}
for m in ["t4b3", "ck1500", "ck2000", "soup_20_35", "wise_080"]:
    pd_f = f"{udir}/dev.{m}.p.jsonl.gz"
    pg_f = f"{udir}/gde1v3s.{m}.p.jsonl.gz"
    if not (os.path.exists(pd_f) and os.path.exists(pg_f)):
        continue
    pd = {r["id"]: r["p"] for r in load_gz(pd_f) if "p" in r}
    pg = {r["id"]: r["p"] for r in load_gz(pg_f) if "p" in r}
    best_t, best_nll = 1.0, 999.0
    for k in range(70, 202, 2):
        T = k / 100.0
        st = eval_cal(req_dev, pd, dev_h0, T)
        if st["nll"] < best_nll:
            best_nll, best_t = st["nll"], T
    cal_summary[m] = {
        "fit_T_dev_h0": best_t,
        "dev_h1_T1": eval_cal(req_dev, pd, dev_h1, 1.0),
        "dev_h1_Tfit": eval_cal(req_dev, pd, dev_h1, best_t),
        "gde_T1": eval_cal(req_gde, pg, gde_all, 1.0),
        "gde_Tfit": eval_cal(req_gde, pg, gde_all, best_t),
    }

scr = {}
for sp in ("full", "h0", "h1"):
    p = Path(sdir) / f"{sp}.json"
    if p.exists():
        scr[sp] = json.load(open(p))

out = {
    "arm": "t4b3s",
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "top2_screened": top2,
    "unseen": {s: {"common": uj[s]["common"], "table": uj[s]["table"]} for s in ("gde1v3s", "dev")},
    "screen_4400": scr,
    "temperature_calibration": cal_summary,
}
Path(outp).write_text(json.dumps(out, indent=2))
print("SWEEP_SUMMARY_OK", outp)
PY

aws s3 cp --quiet $OUT/${NAME}_result.json $U/${NAME}_result.json
date -u +%FT%TZ | aws s3 cp --quiet - $U/state/${NAME}_chain_done
log "SWEEP_CHAIN_DONE"
mirror
