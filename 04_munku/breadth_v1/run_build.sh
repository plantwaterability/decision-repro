#!/bin/bash
# breadth_v1 wave-1 build on ctl2 (CPU). Usage: run_build.sh <decisions.json> <outdir>
D=/data/local/breadth_v1
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
HO=$(ls /data/local/v8build/heldout/official_dev_splits_eval.jsonl /data/local/v8build/heldout/official_dev_splits_eval_v2.jsonl /data/local/v8build/heldout/libeval_clean.jsonl /data/local/v8build/heldout/v7/HO_*.jsonl /data/local/hoscore/ho8_rows.jsonl.gz /tmp/trainers/data/shared/fullft_test/materialize_r2/diag/sd2_dev_select_r2.jsonl | paste -sd,)
HX=$(find $D/gate/gde1 -name "*.jsonl" | sort | paste -sd,)
python3 $D/scripts/build.py --kit /data/local/phish_v73h/kit --code /tmp/c2e/esbar_code_v2e/trainers --scripts /data/local/v8build/scripts --tok /data/local/v8build/tok \
  --suite /data/local/phish_v73h/suite/suite-0.3 --heldout "$HO" --heldout_exact "$HX" --v8 /data/local/v8build/final/esbar_stage1_v8.jsonl \
  --cand $D/cand --decisions "$1" --out "$2"
