#!/bin/bash
# es-bar Stage-1 v2 manifest (continuation from final v1): base (all) + the new arithmetic / STEM pools + a replay share of every v1 pool,
# with the program library's eval-split groups excluded (build_manifest --exclude-groups). New seed, so replayed pools draw a fresh mix.
#   bash build_v2.sh EXCLUDE_GROUPS_FILE
unset ASCALE_RUN_ID; export OMP_NUM_THREADS=1
EXCL=${1:?library eval-split group file}
D=/data/local; E=$D/esbar_data; H=$D/heldout_loom
export PATH=$D/venv_t2141/bin:$PATH TOKENIZERS_PARALLELISM=false
P=$E/pools_v2; mkdir -p $P
for f in pools/tracking pools/shapes pools/chess_train incoming/causal incoming/navigate incoming/belief incoming/formulas incoming/boolean incoming/lies \
         incoming_v2b/arith_gsm8k incoming_v2b/arith_aqua incoming_stem/stem_math; do
  ln -sfn $E/$f.jsonl $P/$(basename $f).jsonl
done
ln -sfn $E/incoming_stem/stem_mds_f3k2.jsonl $P/stem_mds.jsonl      # K=2 rows only at correct_samples >= 3
cp $EXCL $E/out/final_v2_exclude_groups.txt
cd $E && md5sum build_manifest_v2.py out/final_v2_exclude_groups.txt $P/*.jsonl > out/final_v2_build.log
python3 build_manifest_v2.py --code $D/esbar_code_v2/trainers --model $H/model --suite $H/suite-0.3 --heldout $H --pools $P \
  --base $E/base/coteach_wave7_ppfixed_v1_25k_l4096_gemma.jsonl --out $E/out/esbar_stage1_final_v2.jsonl --seed 20261102 \
  --exclude-groups $E/out/final_v2_exclude_groups.txt \
  --per-pool arith_gsm8k=13000,arith_aqua=11000,stem_mds=14000,stem_math=5100,chess_train=5000,tracking=3000,shapes=1500,navigate=2000,causal=2000,belief=2000,formulas=3000,boolean=1000,lies=1000 \
  --bigk-max 2000 >> out/final_v2_build.log 2>&1
echo "exit $?" >> out/final_v2_build.log
B=s3://YOUR_BUCKET/jev-race/engineering/esbar/manifests
for f in esbar_stage1_final_v2.jsonl esbar_stage1_final_v2.jsonl.header.json final_v2_exclude_groups.txt; do
  aws s3 cp --only-show-errors out/$f $B/$f && \
  echo "$f s3 $(aws s3api head-object --bucket YOUR_BUCKET --key jev-race/engineering/esbar/manifests/$f --query ContentLength) local $(stat -c %s out/$f)" >> out/final_v2_build.log
done
