#!/bin/bash
# pixmo_cnt v2 (counting-method rows only): keep the first pool under prov/ (S3 has no versioning), re-plan, fetch what is new, rebuild, package
set -u
export AWS_DEFAULT_REGION=us-east-2 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
G=/data/shared/visdata_g; S=$G/scripts; L=$G/logs; SG=s3://YOUR_BUCKET/jev-race/engineering/vision/recipe_g
log(){ echo "[pixmo2 $(date -u +%H:%M:%SZ)] $*" >> $L/g_build.log; }
bash $S/vg_setup.sh; cd $L
if [ ! -f $G/prov_v1/pixmo_cnt_v1.jsonl ]; then
  mkdir -p $G/prov_v1 $G/trash
  aws s3 cp --quiet $SG/items/pixmo_cnt.jsonl $SG/prov/pixmo_cnt_v1.jsonl && aws s3 cp --quiet $SG/img/img_pixmo_cnt.tar $SG/prov/img_pixmo_cnt_v1.tar || { log "PROV_COPY_FAIL"; exit 2; }
  a=$(md5sum < $G/items_g/pixmo_cnt.jsonl | cut -d" " -f1); b=$(aws s3 cp --quiet $SG/prov/pixmo_cnt_v1.jsonl - | md5sum | cut -d" " -f1)
  c=$(grep " img_pixmo_cnt.tar$" $G/pkg/MD5SUMS_pixmo | cut -d" " -f1); d=$(aws s3 cp --quiet $SG/prov/img_pixmo_cnt_v1.tar - | md5sum | cut -d" " -f1)
  [ "$a" = "$b" ] && [ "$c" = "$d" ] || { log "PROV_MD5_BAD $a $b $c $d"; exit 2; }
  printf '%s  pixmo_cnt_v1.jsonl\n%s  img_pixmo_cnt_v1.tar\n' "$a" "$c" > $G/pkg/MD5SUMS_pixmo_v1; aws s3 cp --quiet $G/pkg/MD5SUMS_pixmo_v1 $SG/MD5SUMS_pixmo_v1
  mv $G/items_g/pixmo_cnt.jsonl $G/prov_v1/pixmo_cnt_v1.jsonl; mv $G/img_g/pixmo_cnt $G/trash/pixmo_cnt_v1_img
  cp $G/dl/pixmo/plan.jsonl $G/prov_v1/pixmo_plan_v1.jsonl; cp $G/dl/pixmo/plan_summary.json $G/prov_v1/pixmo_plan_summary_v1.json; cp $G/summ/pixmo.json $G/prov_v1/summ_pixmo_v1.json
  log "v1 kept: prov/pixmo_cnt_v1.jsonl $a, prov/img_pixmo_cnt_v1.tar $c"
fi
python3 -P $S/vg_build.py pixmo_plan > $L/g_step_pixmo_plan2.log 2>&1 || { log "PLAN2_FAIL"; exit 3; }; log "plan2 $(tail -c 400 $L/g_step_pixmo_plan2.log | tr '\n' ' ')"
python3 -P $S/vg_build.py pixmo_fetch > $L/g_step_pixmo_fetch2.log 2>&1 || { log "FETCH2_FAIL"; exit 4; }; log "fetch2 $(tail -c 300 $L/g_step_pixmo_fetch2.log | tr '\n' ' ')"
python3 -P $S/vg_build.py pixmo > $L/g_build_pixmo2.log 2>&1 || { log "BUILD2_FAIL"; exit 5; }; log "build2 $(tail -c 300 $L/g_build_pixmo2.log | tr '\n' ' ')"
bash $S/vg_pkgimg.sh pixmo2 pixmo_cnt; log "G_PIXMO2_DONE $(tail -1 $L/g_package.log)"
