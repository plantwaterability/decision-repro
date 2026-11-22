#!/bin/bash
# DH round H: build the single-question pools on ctl2 (CPU), make the karpo requests file, upload everything (md5 read-back), queue labels on p10.
set -u; unset ASCALE_RUN_ID; export AWS_DEFAULT_REGION=us-east-2 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
W=/data/shared/dh; R=s3://YOUR_BUCKET/jev-race/engineering/esbar/runs; MX=$R/esbar_mx; O=$R/esbar_dh
cd $W; L=$W/logs/build_h.log
log() { echo "[run_h $(date -u +%FT%TZ)] $*" >> $L; aws s3 cp --only-show-errors $L $O/logs/build_h.log 2>/dev/null; }
put() { aws s3 cp --only-show-errors "$1" "$2" && [ "$(aws s3 cp --only-show-errors "$2" - | md5sum | cut -c1-32)" = "$(md5sum < "$1" | cut -c1-32)" ] || { log "PUT_FAIL $1 -> $2"; return 1; }; }
: > $L; log "start builder $(md5sum < scripts/fs5h_build.py | cut -c1-8) ag $(md5sum < scripts/gsm8k_ag_kit.py | cut -c1-8)"
python3 -I $W/scripts/fs5h_build.py >> $L 2>&1; rc=$?; log "BUILD_RC $rc"
[ $rc = 0 ] || exit 1
: | gzip > $W/empty.jsonl.gz
python3 $W/scripts/dh_h0.py $W/out/fs5h_rows.jsonl $W/empty.jsonl.gz $W/out/fs5h.jsonl.gz >> $L 2>&1 || { log "REQUESTS_FAIL"; exit 1; }
cd $W/out; sha256sum fs5h_rows.jsonl fs5h_rows.meta.jsonl fs5h_report.json fs5h.jsonl.gz fs5h.stats.json pools/* ../scripts/fs5h_build.py ../scripts/gsm8k_ag_kit.py ../scripts/dh_h0.py > fs5h.SHA256SUMS
(cd $W && sha256sum lic/* src/dl_sha256.txt > out/fs5h_lic.SHA256SUMS)
for f in fs5h_rows.jsonl fs5h_rows.meta.jsonl fs5h_report.json fs5h.SHA256SUMS fs5h_lic.SHA256SUMS; do put $f $O/pools/$f || exit 1; done
for f in pools/*; do put $f $O/pools/$f || exit 1; done
for f in ../lic/* ../src/dl_sha256.txt; do put $f $O/lic/$(basename $f) || exit 1; done
for f in ../scripts/fs5h_build.py ../scripts/gsm8k_ag_kit.py ../scripts/dh_h0.py ../scripts/run_h.sh; do put $f $O/code/$(basename $f) || exit 1; done
for f in fs5h.jsonl.gz fs5h.stats.json fs5h.SHA256SUMS; do put $f $MX/tl/$f || exit 1; done
log "UPLOADED rows $(wc -l < fs5h_rows.jsonl) requests $(zcat fs5h.jsonl.gz | wc -l) sha $(sha256sum < fs5h.jsonl.gz | cut -c1-16)"
aws s3 cp --only-show-errors $MX/queue/p10.txt /tmp/p10_dh.txt
grep -q '^tlab mx_tl_karpo_fs5h fs5h$' /tmp/p10_dh.txt || { printf '# DH round H single-question pools (cfcolor / HellaSwag / BANKING77 / GSM8K answer-groups / SATA wording): karpo own-engine labels\ntlab mx_tl_karpo_fs5h fs5h\n' >> /tmp/p10_dh.txt; aws s3 cp --only-show-errors /tmp/p10_dh.txt $MX/queue/p10.txt && log "QUEUED tlab mx_tl_karpo_fs5h on p10"; }
log RUN_H_DONE
