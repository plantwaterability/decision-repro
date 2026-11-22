#!/bin/bash
# Arm UE2 (plain arm on the new data; prereg in AGENT_NOTES 2026-11-10): Arm U's recipe unchanged (Qwen3.8 BASE, no init, no OCR
# block, no perception facts, stock code/qv_train.py, same seed and settings) on fork B's ue2_nc_noade manifest (236,800 rows,
# 3,700 steps at 64). Dev reads at 1,000 / 2,000 / 3,000 / 3,700 (mid-run collapse stop: dev < STOP_BELOW). After the final step:
# if the step-3,700 dev index >= FINAL_BAR, ONE public read without facts or OCR on the reference path (soup_eval.py, as SU50 /
# dormice): noWG rows on 8 GPUs, then the 800 Winoground rows (eval_restricted), then the noWG + full indexes. Outputs to S3.
# ue3 variant (fork A): same script as qv_arm_ue.sh (md5 18aa705a) with (1) the manifest given by env MAN / SHA and (2) the two-number RBH gate
#   between the final dev bar and the one public read. Nothing else changed.
# usage: MAN=ue3_nc_noade SHA=<sha> STEPS=3700 GSTEPS=1000,2000,3000,3700 qv_arm_ue3.sh <run name> <dev_stop_below> <final_dev_bar>
# smoke overrides (env): STEPS SAVE GSTEPS GATE_FILE NPROC PORT PUBLIC=0 (smoke never touches a public row)
set -u
export AWS_DEFAULT_REGION=us-east-2 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
NAME=$1; STOPB=$2; FBAR=$3
MAN=${MAN:?set MAN=ue3_nc_noade or ue3s}; SHA=${SHA:?set SHA=<rows_sha256 from the manifest header>}  # ue3: manifest by env; STEPS=3700 (ue3_nc_noade) or 3325 (ue3s)
STEPS=${STEPS:-3700}; SAVE=${SAVE:-100}; GSTEPS=${GSTEPS:-1000,2000,3000,3700}; NPROC=${NPROC:-8}; PORT=${PORT:-29626}; PUBLIC=${PUBLIC:-1}; WG=${WG:-1}  # WG=0 (helpers fork, p8): the node has no Winoground suite (restricted copies live on p8e only): noWG read + noWG index here, Winoground rows + full index on p8e afterwards (ue3_wg_p8e.sh)
Q=/data/local/qv; L=$Q/logs/arm_${NAME}.log; TR=$Q/venv/bin/torchrun; PY=$Q/venv/bin/python
CAPW=""; [ -n "${READ_CAP:-}" ] && CAPW=/data/local/vh/code2/capwrap.py
# OCR=<train ocr cache jsonl> (fork A, arm uof1 = Arm UO's recipe): the PP-OCRv5 block before the question in training, the dev reads and every read below
# (RBH, HELD incl. the reference, public), exactly as qv_arm_ue3o.sh (helpers fork, Arm UO4) does it. Without OCR this script is qv_arm_uf.sh unchanged.
OCR=${OCR:-}; EV=code/soup_eval.py; OTR=""; O_RBH=""; O_PUB=""; O_HELD=""
if [ -n "$OCR" ]; then EV=code/soup_eval_ocr.py; OTR="--ocr-train $OCR --ocr-gate ocr/ocr_dev.jsonl"; O_RBH=$Q/rbh_gate/ocr_rbh.jsonl; O_PUB=ocr/ocr_public.jsonl; O_HELD=$Q/priv/ocr_priv.jsonl; fi  # READ_CAP=0.45: each read process is capped at that share of its GPU
V=s3://YOUR_BUCKET/jev-race/engineering/vision; S3=$V/recipe_a; E=$V/recipe_e; FF=$V/recipe_f; FG=$V/recipe_g  # round F / G (data fork): manifests uf* / ug* + their new image tars
SE=$V/eval_restricted/winoground_research_only; ER=$Q/eval_restricted
log(){ echo "[arm $(date -u +%H:%M:%SZ)] $*" >> $L; }
mkdir -p $Q/logs $Q/data/img $Q/code $Q/kit $Q/runs/$NAME
cd $Q
log "START $NAME manifest=$MAN expect=$SHA steps=$STEPS gates=$GSTEPS stop_below=$STOPB final_bar=$FBAR nproc=$NPROC public=$PUBLIC"
[ -f $Q/venv.functional.ok ] || { log NO_FUNCTIONAL_ENV; exit 2; }
aws s3 sync --quiet $S3/code/qv/ $Q/code/ && (cd $Q/code && md5sum -c MD5SUMS >> $L 2>&1) || { log CODE_MD5_BAD; exit 3; }
[ -d $Q/kit/kit_vis ] || { aws s3 cp --quiet $S3/code/kit_vis.tgz $Q/kit/ && tar xzf $Q/kit/kit_vis.tgz -C $Q/kit; }
[ -f $Q/data/$MAN.jsonl ] || { aws s3 cp --quiet $FG/manifests/$MAN.jsonl $Q/data/ && aws s3 cp --quiet $FG/manifests/$MAN.header.json $Q/data/; }
GOT=$(sha256sum $Q/data/$MAN.jsonl | cut -d" " -f1); [ "$GOT" = "$SHA" ] || { log "MANIFEST_SHA_BAD $GOT"; exit 4; }
aws s3 cp --quiet $S3/MD5SUMS $Q/data/MD5SUMS && aws s3 cp --quiet $V/recipe_b/MD5SUMS $Q/data/MD5SUMS_b && aws s3 cp --quiet $V/recipe_b3/MD5SUMS $Q/data/MD5SUMS_b3 && aws s3 cp --quiet $V/recipe_c/MD5SUMS $Q/data/MD5SUMS_c && aws s3 cp --quiet $V/recipe_d/MD5SUMS $Q/data/MD5SUMS_d && aws s3 cp --quiet $E/MD5SUMS $Q/data/MD5SUMS_e && aws s3 cp --quiet $FF/MD5SUMS $Q/data/MD5SUMS_f && aws s3 cp --quiet $FG/MD5SUMS $Q/data/MD5SUMS_g
for fam in $(python3 -c "import json; print(' '.join(sorted({r['source'] for r in map(json.loads, open('$Q/data/$MAN.jsonl'))})))"); do
  [ -f $Q/data/img/.ok_$fam ] && continue
  PFX=""; for pm in "$S3 MD5SUMS" "$V/recipe_b MD5SUMS_b" "$V/recipe_b3 MD5SUMS_b3" "$V/recipe_c MD5SUMS_c" "$V/recipe_d MD5SUMS_d" "$E MD5SUMS_e" "$FF MD5SUMS_f" "$FG MD5SUMS_g"; do set -- $pm; grep -q " img_$fam.tar$" $Q/data/$2 && { PFX=$1; M=$Q/data/$2; break; }; done
  [ -n "$PFX" ] || { log "IMG_NOT_IN_ANY_MD5SUMS $fam"; exit 5; }
  aws s3 cp --quiet $PFX/img/img_$fam.tar $Q/data/ || { log "IMG_FETCH_FAIL $fam"; exit 5; }
  want=$(grep " img_$fam.tar$" $M | cut -d" " -f1); got=$(md5sum < $Q/data/img_$fam.tar | cut -d" " -f1)
  [ "$want" = "$got" ] || { log "IMG_MD5_BAD $fam"; exit 6; }
  tar xf $Q/data/img_$fam.tar -C $Q/data/img && touch $Q/data/img/.ok_$fam && log "IMG_OK $fam"
done
MISSING=$(python3 -c "
import json, os
n = 0
for l in open('$Q/data/$MAN.jsonl'):
    for p in json.loads(l)['images']:
        n += not os.path.exists('$Q/data/img/' + p)
print(n)")
[ "$MISSING" = "0" ] || { log "MANIFEST_IMAGES_MISSING $MISSING"; exit 10; }
log "INPUTS_OK manifest=$GOT images_missing=0"
[ "${STAGE_ONLY:-0}" = "1" ] && { log "STAGE_ONLY: inputs staged, stopping before training"; exit 0; }
RESUME=""
if ! ls -d $Q/runs/$NAME/ckpt_* >/dev/null 2>&1; then  # fresh node: pull the newest mirrored checkpoint, md5-checked file by file
  SL=$(aws s3 ls $S3/runs/$NAME/ckpt/ | awk '{print $2}' | grep ckpt_ | sort | tail -1 | tr -d /)
  if [ -n "$SL" ]; then
    aws s3 sync --quiet $S3/runs/$NAME/ckpt/$SL/ $Q/runs/$NAME/$SL/ && log "PULLED_S3_CKPT $SL"
    for f in adapter.safetensors head.pt optim.pt state.json; do
      a=$(md5sum < $Q/runs/$NAME/$SL/$f | cut -d" " -f1); b=$(aws s3 cp --quiet $S3/runs/$NAME/ckpt/$SL/$f - | md5sum | cut -d" " -f1)
      [ "$a" = "$b" ] || { log "S3_CKPT_MD5_BAD $SL/$f"; exit 9; }
    done
    [ -f $Q/runs/$NAME/$SL/train_log.jsonl ] && cp $Q/runs/$NAME/$SL/train_log.jsonl $Q/runs/$NAME/train_log.jsonl
    # the checkpoint's copy of the run log is written before the same step's dev read: take the run-level log + gate files when they are fuller
    aws s3 cp --quiet $S3/runs/$NAME/train_log.jsonl $Q/runs/$NAME/train_log.s3.jsonl 2>/dev/null && [ "$(wc -l < $Q/runs/$NAME/train_log.s3.jsonl)" -ge "$(wc -l < $Q/runs/$NAME/train_log.jsonl 2>/dev/null || echo 0)" ] && cp $Q/runs/$NAME/train_log.s3.jsonl $Q/runs/$NAME/train_log.jsonl && log "RUN_LOG_FROM_S3 $(wc -l < $Q/runs/$NAME/train_log.jsonl) lines"
    aws s3 sync --quiet $S3/runs/$NAME/ $Q/runs/$NAME/ --exclude "*" --include "gate_*.json"
  fi
fi
LAST=""  # newest COMPLETE local checkpoint only (a kill mid-save can leave a partial dir)
for d in $(ls -d $Q/runs/$NAME/ckpt_* 2>/dev/null | sort -r); do
  [ -f $d/state.json ] && [ -f $d/optim.pt ] && [ -f $d/adapter.safetensors ] && [ -f $d/head.pt ] && { LAST=$d; break; }
done
[ -n "$LAST" ] && RESUME="--resume $LAST" && log "RESUME_FROM $LAST"
if [ -n "${GATE_FILE:-}" ]; then GF=$GATE_FILE; GSHA=""; else
  GF=dev/dev_rows.jsonl; DSHA=$(python3 -c "import json; print(json.load(open('$Q/dev/dev_rows.header.json'))['rows_sha256'])")
  [ "$(sha256sum $Q/dev/dev_rows.jsonl | cut -d' ' -f1)" = "$DSHA" ] || { log DEV_ROWS_SHA_BAD; exit 13; }
  GSHA="--gate-sha256 $DSHA"; log "DEV_OK $DSHA"
fi
FINAL=$(printf "ckpt_%05d" $STEPS)
if [ ! -f $Q/runs/$NAME/$FINAL/state.json ] || ! grep -q "\"event\": \"gate\", \"step\": $STEPS," $Q/runs/$NAME/train_log.jsonl 2>/dev/null; then
  $TR --nproc_per_node $NPROC --master_port $PORT code/qv_train.py --model $Q/models/qwen38_27b --manifest data/$MAN.jsonl \
     --expect-rows-sha256 $SHA --img-root data/img --out runs/$NAME --s3 $S3/runs/$NAME/ckpt \
     --steps $STEPS --rows-per-step 64 --lr 1e-4 --head-lr 1e-3 --warmup 50 --floor 0.1 --clip 1.0 --lora-r ${LORA_R:-16} --lora-alpha ${LORA_A:-32} \
     --label-smoothing 0.05 --max-tokens 8192 --save-every $SAVE --seed ${SEED:-20261114} --kit kit/kit_vis \
     --gate-file $GF $GSHA --gate-root dev/img_dev \
     $OTR --gate-steps $GSTEPS --gate-continue 1000000 --gate-stop $STOPB $RESUME >> $Q/logs/train_$NAME.out 2>&1
  RC=$?; log "TRAIN_RC=$RC"
fi
aws s3 sync --quiet $Q/runs/$NAME $S3/runs/$NAME/ --exclude "ckpt_*" && aws s3 cp --quiet $L $S3/runs/$NAME/arm.log
# final dev bar, then the one public read
G=$(python3 -c "
import json
g = [json.loads(l) for l in open('$Q/runs/$NAME/train_log.jsonl') if '\"event\": \"gate\"' in l]
g = [x for x in g if x['step'] == $STEPS]
print(g[-1]['index'] if g else -1)")
[ -f $Q/runs/$NAME/$FINAL/state.json ] || { log "NO_FINAL_CKPT (resume by re-running this script)"; exit 20; }
OKBAR=$(python3 -c "print(int($G >= $FBAR))")
log "FINAL_DEV index=$G bar=$FBAR pass=$OKBAR"
[ "$OKBAR" = "1" ] || { log "FINAL_DEV_BELOW_BAR: no public read"; aws s3 cp --quiet $L $S3/runs/$NAME/arm.log; exit 0; }
[ "$PUBLIC" = "1" ] || [ "${RBH_SMOKE:-0}" = "1" ] || { log "PUBLIC=0: stopping before the public read"; exit 0; }  # RBH_SMOKE=1 (smoke only): run the RBH gate path, then stop before any public row
# --- fork A two-number RBH gate (prereg AGENT_NOTES 2026-11-10; RBH = held-out MathVision check, never a public row): no public read unless it passes
RB=$Q/rbh_gate; PR=$V/recipe_privlike/rbh; RO=$Q/runs/$NAME/rbh_read
mkdir -p $RB $RO
for f in rbh_rows.jsonl img_rbh.tar MD5SUMS; do [ -f $RB/$f ] || aws s3 cp --quiet $PR/$f $RB/ || { log "RBH_FETCH_FAIL $f"; exit 16; }; done
aws s3 cp --quiet $V/recipe_privlike/ue3/code/rbh_gate.py $RB/rbh_gate.py || { log RBH_GATE_CODE_MISSING; exit 16; }
[ -z "$OCR" ] || [ -f $RB/ocr_rbh.jsonl ] || aws s3 cp --quiet $V/recipe_privlike/ocr/ocr_rbh.jsonl $RB/ocr_rbh.jsonl || { log RBH_OCR_MISSING; exit 16; }
for f in rbh_rows.jsonl img_rbh.tar; do [ "$(grep " $f\$" $RB/MD5SUMS | cut -d' ' -f1)" = "$(md5sum < $RB/$f | cut -d' ' -f1)" ] || { log "RBH_MD5_BAD $f"; exit 16; }; done
[ -d $RB/img_rbh ] || tar xf $RB/img_rbh.tar -C $RB
if [ ! -f $RO/rbh_gate.json ]; then
  python3 -c "
rows=[l for l in open('$RB/rbh_rows.jsonl') if l.strip()]
for k in range($NPROC): open('$RB/rows_%d.jsonl'%k,'w').writelines(rows[k::$NPROC])"
  for k in $(seq 0 $((NPROC-1))); do
    ( CUDA_VISIBLE_DEVICES=$k PYTHONPATH=kit/kit_vis:code $PY $CAPW $EV $NAME runs/$NAME/$FINAL 1 $RB/rows_$k.jsonl $RB/img_rbh $RO/shard_$k models/qwen38_27b $O_RBH > $RO/shard_$k.log 2>&1; log "RBH shard $k RC=$?" ) &
  done
  wait
  python3 $RB/rbh_gate.py $RB/rbh_rows.jsonl $RO > $RO/rbh_gate.json 2> $RO/rbh_gate.err
fi
GATE=$(python3 -c "import json; print(int(json.load(open('$RO/rbh_gate.json'))['pass']))" 2>/dev/null || echo 0)
log "RBH_GATE pass=$GATE $(cat $RO/rbh_gate.json 2>/dev/null) $(tail -1 $RO/rbh_gate.err 2>/dev/null)"
aws s3 sync --quiet $Q/runs/$NAME $S3/runs/$NAME/ --exclude "ckpt_*" && aws s3 cp --quiet $L $S3/runs/$NAME/arm.log
[ "$GATE" = "1" ] || [ "${RBH_DIAG:-0}" = "1" ] || { log "RBH_GATE_FAIL: no public read"; exit 0; }
[ "${RBH_DIAG:-0}" = "1" ] && log "RBH is a reported diagnostic for this arm (not gating)"
# --- end of the RBH gate
# --- private-like proxy HELD read (HELD=1; not public; fork A's v1 rows, HELD half by sha1 parity of the id), for this arm and for REF (run/ckpt on S3) as the paired reference
held_block(){
  local PV=$Q/priv PP=$V/recipe_privlike f k; mkdir -p $PV
  if [ ! -f $PV/.ok ]; then
    for f in priv_rows.jsonl MD5SUMS img_priv.tar; do aws s3 cp --quiet $PP/$f $PV/ || return 1; done
    for f in priv_rows.jsonl img_priv.tar; do [ "$(grep " $f\$" $PV/MD5SUMS | tail -1 | cut -d' ' -f1)" = "$(md5sum < $PV/$f | cut -d' ' -f1)" ] || { log "PRIV_MD5_BAD $f"; return 1; }; done
    tar xf $PV/img_priv.tar -C $PV && rm -f $PV/img_priv.tar
    python3 - <<'PY'
import hashlib, json
F = "/data/local/qv/priv"
out = {0: open(f"{F}/priv_fit.jsonl", "w"), 1: open(f"{F}/priv_held.jsonl", "w")}
for l in open(f"{F}/priv_rows.jsonl"):
    if l.strip():
        out[int(hashlib.sha1(json.loads(l)["id"].encode()).hexdigest(), 16) % 2].write(l)
PY
    [ "$(sha256sum $PV/priv_held.jsonl | cut -c1-16)" = "47d52acc7d9d0834" ] && touch $PV/.ok || { log PRIV_SPLIT_BAD; return 1; }
    python3 -c "
rows=[l for l in open('$PV/priv_held.jsonl') if l.strip()]
for k in range($NPROC): open('$PV/rows_%d.jsonl'%k,'w').writelines(rows[k::$NPROC])"
  fi
  one(){ # label ckpt outdir
    ls $3/shard_0/results.jsonl >/dev/null 2>&1 && [ -f $3/.read_ok ] && return 0; mkdir -p $3
    for k in $(seq 0 $((NPROC-1))); do
      ( CUDA_VISIBLE_DEVICES=$k PYTHONPATH=kit/kit_vis:code $PY $CAPW $EV $1 $2 1 $PV/rows_$k.jsonl $PV/img_priv $3/shard_$k models/qwen38_27b $O_HELD > $3/shard_$k.log 2>&1; log "HELD $1 shard $k RC=$?" ) &
    done; wait; touch $3/.read_ok
  }
  [ -z "$OCR" ] || [ -f $O_HELD ] || aws s3 cp --quiet $PP/ocr/ocr_priv.jsonl $O_HELD || { log HELD_OCR_MISSING; return 1; }
  one $NAME runs/$NAME/$FINAL $Q/runs/$NAME/held_read
  local RD=""
  if [ -n "${REF:-}" ]; then
    [ -f $Q/runs/$REF/head.pt ] || aws s3 sync --quiet $S3/runs/${REF%/*}/ckpt/${REF#*/}/ $Q/runs/$REF/ --exclude "optim.pt"
    [ -f $Q/runs/$REF/head.pt ] && { one ref runs/$REF $Q/runs/$NAME/held_ref; RD=$Q/runs/$NAME/held_ref; } || log "HELD_REF_MISSING $REF"
  fi
  python3 - $PV/priv_held.jsonl $Q/runs/$NAME/held_read "$RD" > $Q/runs/$NAME/held.json 2> $Q/runs/$NAME/held.err <<'PY'
import collections, glob, json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
rid = lambda r: (r.get("_evaluation") or {}).get("run_id") or r.get("id")
cnt = collections.Counter(rid(r) for r in rows); rows = [r for r in rows if cnt[rid(r)] == 1]
def preds(d):
    p = {}
    for f in sorted(glob.glob(d + "/shard_*/results.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r.get("status") == "ok": p[r["run_id"]] = r["response"]["answers"]["answer"]["choice"]
    return p
def summ(p):
    fam = collections.defaultdict(lambda: [0, 0, 0.0])
    for r in rows:
        f = fam[r.get("benchmark") or r.get("family") or "all"]; f[0] += 1; f[1] += p.get(rid(r)) == r["expected"]["answer"]; f[2] += 1 / len(r["questions"]["answer"]["criteria"])
    sk = {b: round(100 * max(0.0, (c / n - ch / n) / (1 - ch / n)), 2) for b, (n, c, ch) in fam.items()}
    return {"rows": len(rows), "read": sum(rid(r) in p for r in rows), "acc": round(100 * sum(c for _, c, _ in fam.values()) / len(rows), 2),
            "index_like": round(sum(sk.values()) / len(sk), 2), "skill": sk, "n": {b: v[0] for b, v in fam.items()}}
a = preds(sys.argv[2]); out = {"arm": summ(a), "unique_id_rows_only": True}
if sys.argv[3]:
    b = preds(sys.argv[3]); out["ref"] = summ(b)
    ok = lambda p, r: p.get(rid(r)) == r["expected"]["answer"]
    out["arm_vs_ref"] = {"gained": sum(ok(a, r) and not ok(b, r) for r in rows), "lost": sum(ok(b, r) and not ok(a, r) for r in rows), "index_like_delta": round(out["arm"]["index_like"] - out["ref"]["index_like"], 2)}
print(json.dumps(out))
PY
  log "HELD $(cut -c1-1500 $Q/runs/$NAME/held.json) $(tail -1 $Q/runs/$NAME/held.err 2>/dev/null | cut -c1-200)"
}
if [ "${HELD:-0}" = "1" ]; then held_block || log "HELD_FAIL (reported only; continuing)"; aws s3 sync --quiet $Q/runs/$NAME $S3/runs/$NAME/ --exclude "ckpt_*"; fi
[ "$PUBLIC" = "1" ] || { log "PUBLIC=0 after the RBH gate: stopping before the public read"; exit 0; }
O=$Q/runs/$NAME/public_read; W=$ER/wg_reads/$NAME
[ -f $O/full_index.json ] && { log "PUBLIC_READ_EXISTS - refusing a second public read"; exit 1; }
aws s3 ls $S3/runs/$NAME/public_read/full_index.json > /dev/null 2>&1 && { log "PUBLIC_READ_EXISTS on S3 - refusing a second public read"; exit 1; }
[ "$(sha256sum public/suite/vision-rows-noWG.jsonl | cut -d' ' -f1)" = "0d44a006e575363aee592dd7a03a5ae45cf095c45eef6ce74079e8a040ef5d3c" ] || { log ROWS_SHA_BAD; exit 4; }
[ "$WG" = "0" ] || [ "$(sha256sum $ER/wg_suite/wg_rows.jsonl | cut -d' ' -f1)" = "63119d5b7d663b8c4f254c8995cbf6d082e4005f0a67867830b8f4934945d8ec" ] || { log WG_ROWS_SHA_BAD; exit 4; }
mkdir -p $O/rows $ER/wg_reads
python3 -c "
rows=[l for l in open('public/suite/vision-rows-noWG.jsonl') if l.strip()]
for k in range(8): open('$O/rows/rows_%d.jsonl'%k,'w').writelines(rows[k::8])"
log "PUBLIC_READ_START ckpt=runs/$NAME/$FINAL"
for k in 0 1 2 3 4 5 6 7; do
  ( CUDA_VISIBLE_DEVICES=$k PYTHONPATH=kit/kit_vis:code $PY $CAPW $EV $NAME runs/$NAME/$FINAL 1 $O/rows/rows_$k.jsonl public/suite $O/shard_$k models/qwen38_27b $O_PUB > $O/shard_$k.log 2>&1; log "noWG shard $k RC=$?" ) &
done
wait
[ "$WG" = "0" ] || CUDA_VISIBLE_DEVICES=0 PYTHONPATH=kit/kit_vis:code $PY $CAPW code/soup_eval.py $NAME runs/$NAME/$FINAL 1 $ER/wg_suite/wg_rows.jsonl $ER/wg_suite $W models/qwen38_27b > $ER/wg_reads/$NAME.log 2>&1; log "WG RC=$?"
NAME=$NAME WG=$WG PYTHONPATH=kit/kit_vis $PY - > $O/score.log 2>&1 <<'PY'
import json, glob, os, statistics
from decision_index.vision.score import score
Q = "/data/local/qv"; N = os.environ["NAME"]; O = f"{Q}/runs/{N}/public_read"
nowg = [json.loads(l) for l in open(f"{Q}/public/suite/vision-rows-noWG.jsonl") if l.strip()]
WG = os.environ.get("WG", "1") == "1"
wg = [json.loads(l) for l in open(f"{Q}/eval_restricted/wg_suite/wg_rows.jsonl") if l.strip()] if WG else []
preds, lat = {}, []
for f in sorted(glob.glob(f"{O}/shard_*/results.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if r["status"] == "ok":
            preds[r["run_id"]] = r["response"]["answers"]["answer"]["choice"]
            if "total_wall_ms" in r: lat.append(r["total_wall_ms"])
a = score(nowg, dict(preds))
json.dump({"index": a["public_vision_index"], "benchmarks": a["benchmarks"], "preds": preds}, open(f"{O}/public_noWG.json", "w"))
allp = dict(preds)
for l in (open(f"{Q}/eval_restricted/wg_reads/{N}/results.jsonl") if WG else []):
    r = json.loads(l)
    if r["status"] == "ok":
        allp[r["run_id"]] = r["response"]["answers"]["answer"]["choice"]
b = score(nowg + wg, allp)
out = {"nowg_index": round(a["public_vision_index"], 2), "full_index": round(b["public_vision_index"], 2) if WG else None, "wg_pending": not WG, "n_nowg": len(preds), "missing_nowg": len(nowg) - len(preds),
       "rows_full": len(nowg) + len(wg), "missing_full": len(nowg) + len(wg) - len(allp),
       "skill": {k: round(x["skill_clipped"] * 100, 2) for k, x in b["benchmarks"].items()}, "acc": {k: round(x["accuracy"] * 100, 2) for k, x in b["benchmarks"].items()}}
if lat:
    lat.sort(); out["latency_ms"] = {"median": round(statistics.median(lat)), "p80": round(lat[int(.8 * len(lat))])}
json.dump(out, open(f"{O}/full_index.json" if WG else f"{O}/nowg_index.json", "w")); json.dump(out, open(f"{O}/full_index.json", "w")) if not WG else json.dump(out, open(f"{Q}/eval_restricted/wg_reads/full_index_{N}.json", "w"))
print(json.dumps(out))
PY
log "SCORE RC=$? $(tail -c 2500 $O/score.log)"
aws s3 sync --quiet $Q/runs/$NAME $S3/runs/$NAME/ --exclude "ckpt_*" --exclude "public_read/rows/*" && log S3_PUBLIC_SYNCED
[ "$WG" = "0" ] && log "WG_PENDING: Winoground rows + full index to be read on p8e from the S3 checkpoint" || { aws s3 sync --quiet $W $SE/reads/wg_reads_1110/$NAME/ && aws s3 cp --quiet $ER/wg_reads/full_index_$NAME.json $SE/reads/wg_reads_1110/full_index_$NAME.json && log S3_WG_SYNCED; }
aws s3 cp --quiet $L $S3/runs/$NAME/arm.log
log DONE
