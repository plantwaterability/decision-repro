#!/bin/bash
# usage: vg_job.sh <log tag> <python script> <args...> : one step of round G with setup and a log line in g_build.log
G=/data/shared/visdata_g; S=$G/scripts; L=$G/logs
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
bash $S/vg_setup.sh; mkdir -p $L; cd $L
tag=$1; shift
echo "[$(date -u +%H:%M:%SZ)] start $tag" >> $L/g_build.log
python3 -P $S/"$@" > $L/g_step_$tag.log 2>&1; rc=$?
echo "[$(date -u +%H:%M:%SZ)] end $tag rc=$rc $(tail -c 300 $L/g_step_$tag.log | tr '\n' ' ' | tail -c 200)" >> $L/g_build.log
