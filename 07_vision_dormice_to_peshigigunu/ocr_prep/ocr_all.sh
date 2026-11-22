#!/bin/bash
# OCR pass for the OCR-trained arm (UO): PP-OCRv5 mobile (Arm 0's settings, same pinned pip set) over the gate images (identity
# check against Arm 0's cache), the public noWG suite, the dev set, Winoground (eval_restricted only) and every u_union training
# image. Caches -> S3 with md5 read-back. Resumable (shards skip done images).
set -u
export AWS_DEFAULT_REGION=us-east-2 OMP_NUM_THREADS=2 PADDLE_PDX_MODEL_SOURCE=huggingface
Q=/data/local/qv; O=$Q/ocr; E=$Q/eval_restricted; L=$Q/logs/ocr_all.log
S3=s3://YOUR_BUCKET/jev-race/engineering/vision/recipe_a; SE=s3://YOUR_BUCKET/jev-race/engineering/vision/eval_restricted/winoground_research_only
log(){ echo "[ocr $(date -u +%H:%M:%SZ)] $*" >> $L; }
mkdir -p $O/lists $O/logs $E/ocr; cd $Q
log START
if [ ! -f $Q/ocrvenv.ok ]; then
  python3 -m venv $Q/ocrvenv && $Q/ocrvenv/bin/pip install -q --upgrade pip >> $L 2>&1
  $Q/ocrvenv/bin/pip install -q paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ >> $L 2>&1 || { log PADDLE_GPU_INSTALL_FAIL; exit 2; }
  sudo -n env DEBIAN_FRONTEND=noninteractive apt-get update -q >> $L 2>&1  # p8e has no apt lists (libgl1 not found without this)
  sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -q libgl1 libglib2.0-0 >> $L 2>&1  # OpenCV GUI build needs libGL
  $Q/ocrvenv/bin/pip install -q "paddleocr>=3.1" "paddlex[ocr]==3.7.2" >> $L 2>&1 && touch $Q/ocrvenv.ok || { log OCR_PIP_FAIL; exit 2; }
fi
[ -e /usr/lib/x86_64-linux-gnu/libGL.so.1 ] || { sudo -n env DEBIAN_FRONTEND=noninteractive apt-get update -q >> $L 2>&1; sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -q libgl1 libglib2.0-0 >> $L 2>&1; }
$Q/ocrvenv/bin/pip freeze 2>/dev/null | grep -iE "^(paddle|opencv|numpy|shapely|pyclipper)" > $O/pip_freeze.txt
log "OCR_ENV $(grep -iE '^paddle' $O/pip_freeze.txt | tr '\n' ' ')"
python3 - <<'PY' >> $L 2>&1
import json
Q = "/data/local/qv"
for name, rows in [("gate", "gate/gate_rows.jsonl"), ("public", "public/suite/vision-rows-noWG.jsonl"), ("dev", "dev/dev_rows.jsonl"),
                   ("wg", "eval_restricted/wg_suite/wg_rows.jsonl"), ("train", "data/u_union.jsonl")]:
    s = set()
    for l in open(f"{Q}/{rows}"):
        s.update(json.loads(l)["images"])
    open(f"{Q}/ocr/lists/{name}.txt", "w").write("\n".join(sorted(s)) + "\n")
    print(f"[ocr list] {name} {len(s)} images")
PY
# run_set <name> <image_root> <procs per GPU>
run_set(){
  local name=$1 root=$2 per=$3 n=$((8 * $3)) bad=0
  mkdir -p $O/$name
  for k in $(seq 0 $((n - 1))); do
    CUDA_VISIBLE_DEVICES=$((k % 8)) $Q/ocrvenv/bin/python $Q/code/ocr_pass.py $O/lists/$name.txt $root $O/$name/shard_$k.jsonl $k $n gpu:0 > $O/logs/${name}_$k.log 2>&1 &
  done
  for j in $(jobs -p); do wait $j || bad=$((bad + 1)); done
  [ $bad = 0 ] || { log "OCR_SHARD_FAIL $name $bad: $(tail -2 $O/logs/${name}_0.log | tr '\n' ' ' | cut -c1-300)"; return 1; }
  local dst=$O/ocr_$name.jsonl; [ $name = wg ] && dst=$E/ocr/ocr_wg.jsonl
  python3 $Q/code/ocr_merge.py $O/lists/$name.txt $O/$name $dst > $O/logs/merge_$name.json 2>&1 || { log "OCR_MERGE_FAIL $name $(cat $O/logs/merge_$name.json | cut -c1-300)"; return 1; }
  log "OCR_OK $name $(cat $O/logs/merge_$name.json) md5 $(md5sum < $dst | cut -d' ' -f1)"
}
# 1) gate images on ONE process first (downloads the model archives once), then the identity check against Arm 0's cache
mkdir -p $O/gate
CUDA_VISIBLE_DEVICES=0 $Q/ocrvenv/bin/python $Q/code/ocr_pass.py $O/lists/gate.txt gate $O/gate/shard_0.jsonl 0 1 gpu:0 > $O/logs/gate_0.log 2>&1 || { log "OCR_GATE_FAIL $(tail -3 $O/logs/gate_0.log | tr '\n' ' ' | cut -c1-400)"; exit 3; }
python3 $Q/code/ocr_merge.py $O/lists/gate.txt $O/gate $O/ocr_gate.jsonl > $O/logs/merge_gate.json 2>&1 || { log "OCR_MERGE_FAIL gate"; exit 3; }
log "OCR_OK gate $(cat $O/logs/merge_gate.json)"
aws s3 cp --quiet $S3/arm0/ocr_cache.jsonl $O/arm0_ocr_cache.jsonl
[ "$(md5sum < $O/arm0_ocr_cache.jsonl | cut -d' ' -f1)" = cc4e6ef232737f5d80505417be493962 ] || { log ARM0_CACHE_MD5_BAD; exit 4; }
SAME=$(python3 - <<'PY'
import json
Q = "/data/local/qv/ocr"
def load(f):
    d = {}
    for l in open(f):
        r = json.loads(l)
        ls = [x for x in r["lines"] if x["score"] >= 0.5]
        ls.sort(key=lambda x: (round(x["box"][1] / 12), x["box"][0]))
        d[r["image"]] = [x["text"] for x in ls]
    return d
a, b = load(f"{Q}/arm0_ocr_cache.jsonl"), load(f"{Q}/ocr_gate.jsonl")
same = sum(a[k] == b.get(k) for k in a)
print(f"{same}/{len(a)} {same / len(a):.4f} keys_equal={set(a) == set(b)}")
PY
)
log "ARM0_IDENTITY same_text_images $SAME"
FR=$(echo "$SAME" | awk '{print $2}')
python3 -c "import sys; sys.exit(0 if float('$FR') >= 0.97 else 1)" || { log "ARM0_IDENTITY_BELOW_0.97 stop for inspection"; exit 5; }
# pin the model archives actually used
MD=$(ls -d ~/.paddlex/official_models 2>/dev/null)
[ -n "$MD" ] && (cd $MD && find . -type f | sort | xargs md5sum > $O/ocr_models.md5 && tar cf $O/ocr_models.tar .) && log "MODELS_PINNED $(wc -l < $O/ocr_models.md5) files tar md5 $(md5sum < $O/ocr_models.tar | cut -d' ' -f1)"
# 2) eval sets at one process per GPU (cleaner per-image ms), then the training images at 3 per GPU
run_set public public/suite 1 || exit 6
run_set dev dev/img_dev 1 || exit 6
run_set wg eval_restricted/wg_suite 1 || exit 6
run_set train data/img 3 || exit 7
# 3) S3 with md5 read-back (Winoground OCR text only under the eval_restricted prefix)
BAD=0
up(){ aws s3 cp --quiet $1 $2 && [ "$(md5sum < $1 | cut -d' ' -f1)" = "$(aws s3 cp --quiet $2 - | md5sum | cut -d' ' -f1)" ] && log "S3_READBACK_OK $2 $(md5sum < $1 | cut -d' ' -f1)" || { log "S3_READBACK_BAD $2"; BAD=1; }; }
up $O/ocr_train.jsonl $S3/ocr/ocr_train.jsonl
up $O/ocr_dev.jsonl $S3/ocr/ocr_dev.jsonl
up $O/ocr_public.jsonl $S3/ocr/ocr_public_EVAL_ONLY.jsonl
up $O/ocr_gate.jsonl $S3/ocr/ocr_gate_EVAL_ONLY.jsonl
up $E/ocr/ocr_wg.jsonl $SE/ocr/ocr_wg.jsonl
[ -f $O/ocr_models.tar ] && up $O/ocr_models.tar $S3/ocr/ocr_models.tar && up $O/ocr_models.md5 $S3/ocr/ocr_models.md5
up $O/pip_freeze.txt $S3/ocr/pip_freeze.txt
aws s3 cp --quiet $L $S3/ocr/ocr_all.log
[ $BAD = 0 ] && log ALL_DONE || { log DONE_WITH_S3_ERRORS; exit 8; }
