#!/bin/bash
# kl_databuild.sh (KL agent): FS5 (mxa.jsonl = FS5's rows + karpo's teacher targets) and dev.jsonl -> karpo-layout token data. CPU only. Backed up to S3.
set -u; source /data/local/kl/scripts/kl_env.sh
W=$K/data; O=$R/esbar_kl_data; L=$K/logs/databuild.log
log() { echo "[kl_databuild $(date -u +%FT%TZ)] $*" | tee -a $L; aws s3 cp --only-show-errors $L $O/databuild.log 2>/dev/null; }
if [ ! -s $W/fs5k.info.json ]; then
  $V/bin/python $SC/kl_data.py $W/mxa.jsonl $W/fs5k $MUNKU $C/trainers --requests $W/fs5.jsonl.gz --teacher $W/teacher_karpo_fs5.jsonl.gz --workers 96 > $K/logs/data_fs5k.out 2>&1 || { log "DIE fs5k: $(tail -2 $K/logs/data_fs5k.out | cut -c1-300)"; exit 1; }
fi
log "fs5k: $(grep KLDATA_DONE $K/logs/data_fs5k.out | cut -c1-1500)"
if [ ! -s $W/devk.info.json ]; then
  $V/bin/python $SC/kl_data.py $D/esbar_v5/manifests/dev.jsonl $W/devk $MUNKU $C/trainers --workers 32 > $K/logs/data_devk.out 2>&1 || { log "DIE devk: $(tail -2 $K/logs/data_devk.out | cut -c1-300)"; exit 1; }
fi
log "devk: $(grep KLDATA_DONE $K/logs/data_devk.out | cut -c1-900)"
(cd $W && sha256sum fs5k.* devk.* > kl_data.SHA256SUMS)
for f in fs5k.ids.npy fs5k.meta.npz fs5k.rowids.txt fs5k.info.json devk.ids.npy devk.meta.npz devk.rowids.txt devk.info.json kl_data.SHA256SUMS; do
  aws s3 cp --only-show-errors $W/$f $O/$f && [ "$(aws s3 cp --only-show-errors $O/$f - | md5sum | cut -c1-32)" = "$(md5sum < $W/$f | cut -c1-32)" ] || { log "DIE s3 $f"; exit 1; }; done
aws s3 cp --only-show-errors $K/logs/data_fs5k.out $O/data_fs5k.out; aws s3 cp --only-show-errors $K/logs/data_devk.out $O/data_devk.out
log "KL_DATABUILD_DONE (S3 copy md5-verified)"
