#!/bin/bash
# round G: (re)start whatever is unfinished after a head replacement; every step skips what is complete
G=/data/shared/visdata_g; S=$G/scripts
bash $S/vg_setup.sh
echo "[$(date -u +%H:%M:%SZ)] resume start" >> $G/logs/g_build.log
( { [ -s $G/items_g/gqa_rw.jsonl ] && grep -q " img_gqa_rw.tar$" $G/pkg/MD5SUMS_gqa 2>/dev/null; } || { bash $S/vg_job.sh gqa_build vg_gqa.py build && bash $S/vg_pkgimg.sh gqa gqa_rw; } ) &
( { [ -s $G/items_g/hs_cnt.jsonl ] && grep -q " img_hs_cnt.tar$" $G/pkg/MD5SUMS_hs_cnt 2>/dev/null; } || bash $S/vg_run.sh hs_cnt:hs_cnt ) &
( [ -s $G/nevertrain_hashes.jsonl ] || bash $S/vg_job.sh nevertrain vg_mix.py nevertrain ) &
( n=$(grep -c . $G/dl/pixmo/fetch_status.jsonl 2>/dev/null || echo 0); [ "$n" -ge "$(grep -c . $G/dl/pixmo/plan.jsonl)" ] || bash $S/vg_job.sh pixmo_fetch vg_build.py pixmo_fetch
  { [ -s $G/items_g/pixmo_cnt.jsonl ] && grep -q " img_pixmo_cnt.tar$" $G/pkg/MD5SUMS_pixmo 2>/dev/null; } || bash $S/vg_run.sh pixmo:pixmo_cnt ) &
wait
echo "[$(date -u +%H:%M:%SZ)] G_RESUME_DONE" >> $G/logs/g_build.log
