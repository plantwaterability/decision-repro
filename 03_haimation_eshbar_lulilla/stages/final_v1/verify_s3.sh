#!/bin/bash
# ESBAR-FINAL: read the S3 export back and compare sha256 with the local export (weights + config + served head files).
unset ASCALE_RUN_ID
export OMP_NUM_THREADS=1 PATH=/data/local/venv_t2141/bin:$PATH
P=/data/local/esbar_final_v1b
S3R=s3://YOUR_BUCKET/jev-race/engineering/esbar/runs/stage1_final_v1
cd $P/run_final/export || exit 1
ls model*.safetensors config.json vbll_head.safetensors vbll_head.json | xargs -P 14 -I{} sh -c \
  'a=$(sha256sum < {} | cut -c1-64); b=$(aws s3 cp --only-show-errors '$S3R'/export/{} - | sha256sum | cut -c1-64); [ "$a" = "$b" ] && echo "OK {} $a" || echo "MISMATCH {} local $a s3 $b"' \
  | sort -k2 > $P/logs/verify_s3.txt
echo "$(grep -c ^OK $P/logs/verify_s3.txt) OK, $(grep -c ^MISMATCH $P/logs/verify_s3.txt) MISMATCH" >> $P/logs/verify_s3.txt
