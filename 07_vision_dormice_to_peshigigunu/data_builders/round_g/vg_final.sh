#!/bin/bash
# last pass: keep the prov copies' md5 file out of the aggregate (their tar lives under prov/, not img/), verify from S3 again, upload late files, report
G=/data/shared/visdata_g; S=$G/scripts; L=$G/logs
log(){ echo "[final $(date -u +%H:%M:%SZ)] $*" >> $L/g_build.log; }
[ -f $G/pkg/MD5SUMS_pixmo_v1 ] && mv $G/pkg/MD5SUMS_pixmo_v1 $G/pkg/PROV_MD5SUMS_pixmo_v1
bash $S/vg_pkgextra.sh extra; log "extra $(tail -2 $L/g_package.log | head -1)"
bash $S/vg_verify.sh ug1_nc ug1c_nc > $L/g_verify.log 2>&1; log "verify $(tail -1 $L/g_verify.log)"
cd /tmp && python3 -I $S/vg_report.py > $L/g_report.out 2>&1; grep "^REPORT_JSON " $L/g_report.out | cut -c13- > $L/g_report.json; log "report $(wc -c < $L/g_report.json) bytes"
bash $S/vg_pkgextra.sh extra2; log "G_FINAL_DONE $(tail -2 $L/g_package.log | head -1)"
