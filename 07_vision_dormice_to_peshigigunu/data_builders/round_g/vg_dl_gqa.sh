#!/bin/bash
# round G: GQA questions (balanced train only is used) + scene graphs v1.1 + Visual Genome image_data (for COCO ids); tokenless; sha256 recorded
G=/data/shared/visdata_g; D=$G/dl/gqa; L=$G/logs/g_download.log; mkdir -p $D $G/logs
log(){ echo "[dl $(date -u +%H:%M:%SZ)] $*" >> $L; }
get(){ local u=$1 o=$2; for t in 1 2 3 4 5; do [ -s "$o" ] && break; curl -sS -L --fail -C - --max-time 3600 -o "$o.part" "$u" && mv "$o.part" "$o" && break; log "retry $t $u"; sleep 20; done; log "GOT $(stat -c %s $o 2>/dev/null) $o"; }
log "start gqa"
get https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip $D/sceneGraphs.zip
get https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip $D/vg_image_data.json.zip
get https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip $D/questions1.2.zip
cd $G/dl && sha256sum gqa/sceneGraphs.zip gqa/vg_image_data.json.zip gqa/questions1.2.zip > $D/SHA256SUMS_gqa && cat $D/SHA256SUMS_gqa >> $L
python3 -I -c "
import zipfile
for f in ('sceneGraphs.zip', 'questions1.2.zip', 'vg_image_data.json.zip'):
    z = zipfile.ZipFile('$D/' + f); print(f, [(i.filename, i.file_size) for i in z.infolist()][:40])
" >> $L 2>&1
log "GQA_DL_DONE"
