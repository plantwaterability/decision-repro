#!/bin/bash
# usage: ug_halves.sh <arm: ug1|uog1> <reference: uf2|uof2>   (CPU, any node with S3; run AFTER the arm's one public no-WG read; makes no model call)
# The arm's stored public answers against the reference's stored ones on the two fixed halves (sha1(row id) % 2), the merge agent's mx_halves.py unchanged.
export AWS_DEFAULT_REGION=us-east-2; V=s3://YOUR_BUCKET/jev-race/engineering/vision; A=$V/recipe_a
G=/data/shared/visdata_g; W=$G/halves; mkdir -p $W; bash $G/scripts/vg_setup.sh
aws s3 cp --quiet $V/work/vision-rows-noWG.jsonl $W/rows.jsonl && [ "$(sha256sum < $W/rows.jsonl | cut -c1-16)" = "0d44a006e575363a" ] || { echo ROWS_SHA_BAD; exit 2; }
for n in $1 $2; do aws s3 cp --quiet $A/runs/$n/public_read/public_noWG.json $W/$n.json || { echo "NO_PUBLIC_READ $n"; exit 3; }; done
cd /tmp && PYTHONPATH=/data/local/visdata/kit/kit_vis python3 $G/scripts/mx_halves.py $W/rows.jsonl $W/$2.json $1=$W/$1.json > $W/halves_$1_vs_$2.out 2>&1; rc=$?
aws s3 cp --quiet $W/halves_$1_vs_$2.out $A/runs/$1/halves_vs_$2.out; echo "rc=$rc"; tail -c 1800 $W/halves_$1_vs_$2.out
