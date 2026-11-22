#!/bin/bash
# usage: vg_run.sh <builder>[:<source>,<source>...] [...]   sequential; one log per builder; each built source is packed and backed up to
# s3 recipe_g straight after its build (items + per-source image tar, md5 read-back), as round F's vf_run.sh + vf_pkgimg.sh
G=/data/shared/visdata_g; S=$G/scripts; L=$G/logs
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
bash $S/vg_setup.sh
mkdir -p $L; cd $L
for sb in "$@"; do
  IFS=: read -r b srcs <<< "$sb"
  echo "[$(date -u +%H:%M:%SZ)] start $b" >> $L/g_build.log
  python3 -P $S/vg_build.py $b > $L/g_build_$b.log 2>&1; rc=$?
  echo "[$(date -u +%H:%M:%SZ)] end $b rc=$rc $(tail -c 300 $L/g_build_$b.log | tr '\n' ' ' | tail -c 200)" >> $L/g_build.log
  if [ $rc -eq 0 ] && [ -n "$srcs" ]; then bash $S/vg_pkgimg.sh $b $(echo $srcs | tr ',' ' '); fi
done
echo "[$(date -u +%H:%M:%SZ)] G_RUN_DONE $*" >> $L/g_build.log
