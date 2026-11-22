#!/bin/bash
# breadth_v1 delivery: stage -> s3 -> md5 read-back of every object (streamed, no temp files).
D=/data/local/breadth_v1; S=$D/deliver; P=s3://YOUR_BUCKET/jev-race/engineering/esbar/breadth_v1
mkdir -p $S/tables $S/pools $S/reports $S/scripts $S/logs $S/licence_upstream
tar xzf $D/xfer/deliver_tables.tgz -C $S/tables
for l in clean nc clean_tierB nc_tierB; do mkdir -p $S/pools/$l; cp -a $D/out/$l/pools $D/out/$l/di_rows $S/pools/$l/; done
for l in clean_src nc_src; do mkdir -p $S/pools/$l; cp -a $D/out_src/$l/pools $D/out_src/$l/di_rows $S/pools/$l/; done
cp $D/out/pools_report.json $S/reports/pools_report_wave1.json; cp $D/out_src/pools_report.json $S/reports/pools_report_wave1b.json
cp $D/cand/sample_stats.json $D/cand/sample_src_stats.json $D/ts_raw/fetch_log.json $S/reports/ 2>/dev/null
cp $D/scripts/* $S/scripts/; cp $D/logs/*.log $S/logs/; cp $D/dec/*.json $S/reports/
cp $D/lic/upstream.jsonl $S/licence_upstream/; cp -a $D/lic/text $S/licence_upstream/
cd $S && find . -type f ! -name MD5SUMS -print0 | sort -z | xargs -0 md5sum > $D/xfer/MD5SUMS && cp $D/xfer/MD5SUMS $S/MD5SUMS
echo "staged $(wc -l < $S/MD5SUMS) files $(du -sh $S | cut -f1)"
echo "existing objects under prefix: $(aws s3 ls $P/ --recursive | wc -l)"
aws s3 cp $S $P/ --recursive --only-show-errors; echo "cp exit $?"
echo "objects after upload: $(aws s3 ls $P/ --recursive | wc -l)"
bad=0; n=0
while read -r m f; do
  f=${f#./}; r=$(aws s3 cp "$P/$f" - | md5sum | cut -d" " -f1); n=$((n+1))
  if [ "$r" != "$m" ]; then bad=$((bad+1)); echo "MISMATCH $f $m $r"; fi
done < $D/xfer/MD5SUMS
echo "READBACK files $n mismatches $bad $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo UPLOAD_DONE
