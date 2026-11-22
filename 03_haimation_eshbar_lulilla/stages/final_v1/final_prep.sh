#!/bin/bash
# ESBAR-FINAL prep on p8: new run root, init dir (export config + symlinked liquamen weights, identity check), manifests copied from the
# pilot dir (sha256 vs headers; the guard manifest is NOT copied), final manifest sha256 + stats + overlap.
set -u
export PATH=/data/local/venv_t2141/bin:$PATH OMP_NUM_THREADS=8
F=/data/local/esbar_final_v1b
PP=/data/local/esbar_pilot_v1
mkdir -p $F/init $F/manifests $F/logs $F/scripts || exit 1
M=/data/local/heldout_loom/model
for f in $M/*; do b=$(basename $f); case $b in config.json|vbll_head*) continue;; esac; [ -e $F/init/$b ] || ln -s $f $F/init/$b; done
[ -e $F/init/config.json ] || cp $PP/config.export.json $F/init/config.json
md5sum $F/init/config.json
( cd $F/init && ls model*.safetensors config.json | sort | xargs -P 13 -n 1 sha256sum | sort -k2 ) > $F/init_sha256.txt
echo "INIT_IDENTITY $(sha256sum < $F/init_sha256.txt | cut -c1-64)"
for m in headfit cal dev when2call_4opt_clean_dev; do
  cp $PP/manifests/$m.jsonl $PP/manifests/$m.jsonl.header.json $F/manifests/
  got=$(sha256sum < $F/manifests/$m.jsonl | cut -c1-64); want=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['rows_sha256'])" $F/manifests/$m.jsonl.header.json)
  echo "$m $([ "$got" = "$want" ] && echo OK || echo MISMATCH) $got"
done
PM=/data/local/esbar_data/out/esbar_stage1_final_v1.jsonl
echo "final manifest sha256 $(sha256sum < $PM | cut -c1-64) md5 $(md5sum < $PM | cut -c1-32)"
cat /data/local/esbar_data/out/final_v1_build.log
md5sum /data/local/esbar_code_v1/engine/*.py /data/local/esbar_code_v1/engine/*.json /data/local/esbar_code_v1/trainers/fullft_train.py /data/local/esbar_code_v1/trainers/sdv1_head.py /data/local/esbar_code_v1/trainers/fullft_mem.py
echo "(stats: see esbar_final_v1/logs/prep.log, same manifest)"
date -u
