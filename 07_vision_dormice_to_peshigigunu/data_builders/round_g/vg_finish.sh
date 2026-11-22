#!/bin/bash
# round G finish: dedupe gate -> ug1_nc -> ug1c_nc -> checks -> OCR coverage -> package (lists, manifests, code, prov) -> S3-only verify
set -u
export AWS_DEFAULT_REGION=us-east-2 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
G=/data/shared/visdata_g; S=$G/scripts; L=$G/logs; V=s3://YOUR_BUCKET/jev-race/engineering/vision
bash $S/vg_setup.sh; cd $L
log(){ echo "[finish $(date -u +%H:%M:%SZ)] $*" >> $L/g_build.log; }
[ -s $G/nevertrain_hashes.jsonl ] || { log "NEVERTRAIN_MISSING"; exit 2; }
if [ "${SKIP_POOL:-0}" != "1" ]; then python3 -P $S/vg_mix.py pool > $L/g_pool.log 2>&1 || { log "POOL_FAIL"; exit 3; }; log "pool done $(tail -c 200 $L/g_pool.log | tr '\n' ' ')"; fi
python3 -P $S/vg_mix.py $S/cfg_ug1_nc.json > $L/g_mix_ug1_nc.log 2>&1 || { log "MIX_UG1_FAIL $(tail -c 300 $L/g_mix_ug1_nc.log | tr '\n' ' ')"; exit 4; }
SHA1=$(python3 -I -c "import json; print(json.load(open('$G/manifests_g/ug1_nc.header.json'))['rows_sha256'])")
python3 -I - $SHA1 <<'PY'
import json, sys
json.dump({"name": "ug1c_nc", "base_path": "/data/shared/visdata_g/manifests_g/ug1_nc.jsonl", "base_sha256": sys.argv[1], "line": "nc",
           "add": {"chartqa_h": 6080}}, open("/data/shared/visdata_g/scripts/cfg_ug1c_nc.json", "w"), indent=1)
PY
python3 -P $S/vg_mix.py $S/cfg_ug1c_nc.json > $L/g_mix_ug1c_nc.log 2>&1 || { log "MIX_UG1C_FAIL $(tail -c 300 $L/g_mix_ug1c_nc.log | tr '\n' ' ')"; exit 5; }
log "mix done $(grep -o '"rows_sha256": "[0-9a-f]*"' $L/g_mix_ug1_nc.log $L/g_mix_ug1c_nc.log | tr '\n' ' ')"
k=$(aws s3 ls --recursive $V/helpers/ | grep "vh_facts.py$" | sort | tail -1 | awk '{print $4}')
VH=-; [ -n "$k" ] && aws s3 cp --quiet s3://YOUR_BUCKET/$k $G/base/vh_facts.py && VH=$G/base/vh_facts.py && log "vh_facts $k $(md5sum < $VH | cut -c1-8)"
python3 -I $S/vg_chk.py $G/base/qv_train.py $VH ug1_nc ug1c_nc > $L/g_chk_manifest.log 2>&1; log "chk rc=$? $(cut -c1-600 $L/g_chk_manifest.log | tr '\n' ' ')"
[ -s $G/ocr/ocr_train_uf1.jsonl ] || { mkdir -p $G/ocr; aws s3 cp --quiet $V/recipe_f/ocr/ocr_train_uf1.jsonl $G/ocr/ocr_train_uf1.jsonl; }
python3 -I $S/vg_ocrneed.py ug1c_nc $G/ocr/ocr_train_uf1.jsonl > $L/g_ocrneed.log 2>&1; log "ocrneed rc=$? $(cut -c1-700 $L/g_ocrneed.log)"
python3 -I - <<'PY' >> $L/g_build.log 2>&1
import json  # ug1_nc's images are a subset of ug1c_nc's: write its own list too (same cache)
G = "/data/shared/visdata_g/manifests_g/"
imgs = {p for l in open(G + "ug1_nc.jsonl") for p in json.loads(l)["images"]}
rows = [json.loads(l) for l in open(G + "ocr_needed_ug1c_nc.jsonl")]
keep = [r for r in rows if r["image"] in imgs]
open(G + "ocr_needed_ug1_nc.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in keep))
print("[finish] ocr_needed_ug1_nc", len(keep), "of", len(rows))
PY
bash $S/vg_pkgmeta.sh meta; log "pkgmeta $(tail -2 $L/g_package.log | tr '\n' ' ')"
bash $S/vg_verify.sh ug1_nc ug1c_nc > $L/g_verify.log 2>&1; log "verify $(tail -1 $L/g_verify.log)"
bash $S/vg_pkgextra.sh extra; log "G_FINISH_DONE $(tail -2 $L/g_package.log | tr "\n" " ")"
