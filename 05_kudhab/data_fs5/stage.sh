#!/bin/bash
# FS5 supplement staging on ctl2 (CPU): source downloads at pinned revisions (tokenless), licence evidence, gate inputs, code.
set -u; unset ASCALE_RUN_ID; export AWS_DEFAULT_REGION=us-east-2
W=/data/shared/fs5; S=$W/src; G=$W/gate; L=$W/lic; mkdir -p $S $G $L $W/logs
B=s3://YOUR_BUCKET/jev-race; E=$B/engineering/esbar
LOG=$W/logs/stage.log
log() { echo "[stage $(date -u +%FT%TZ)] $*" >> $LOG; }
get() { [ -s "$2" ] || curl -sSL --fail --retry 3 -o "$2" "$1" || { log "FAIL $1"; rm -f "$2"; return 1; }
        echo "$(sha256sum < "$2" | cut -c1-64)  $(basename $2)  $1" >> $S/dl_sha256.txt; log "got $(basename $2) $(stat -c %s "$2")"; }
lic() { curl -sSL --retry 3 -o "$L/$2" "$1"; echo "$(sha256sum < $L/$2 | cut -c1-64)  $2  $1  http=$(curl -s -o /dev/null -w '%{http_code}' "$1")" >> $L/LIC_SHA256.txt; }
log start
# ---- VAST (kit pin e7c47751)
VR=e7c4775182b184730f350995f8260579c9e066fe; GH=https://raw.githubusercontent.com/emilyallaway/zero-shot-stance/$VR
get $GH/data/VAST/vast_train.csv $S/vast_train.csv
lic $GH/README.md vast_README.md; lic $GH/LICENSE vast_LICENSE; lic https://api.github.com/repos/emilyallaway/zero-shot-stance/license vast_ghlicense.json
# ---- ACOS (kit pin 45d179a3); train files used for rows, dev only for the category inventory
AR=45d179a3dcc6a3dedd848d81b16f2552454805fe; GH=https://raw.githubusercontent.com/NUSTM/ACOS/$AR
for f in Laptop-ACOS/laptop_quad_train.tsv Laptop-ACOS/laptop_quad_dev.tsv Restaurant-ACOS/rest16_quad_train.tsv Restaurant-ACOS/rest16_quad_dev.tsv; do get $GH/data/$f $S/acos_$(basename $f); done
lic $GH/README.md acos_README.md; lic $GH/LICENSE acos_LICENSE; lic https://api.github.com/repos/NUSTM/ACOS/license acos_ghlicense.json
# ---- RAGTruth (kit pin c103204b)
RR=c103204b9ce28d6bbad859304bf30de72b8ed8fe; GH=https://raw.githubusercontent.com/ParticleMedia/RAGTruth/$RR
get $GH/dataset/response.jsonl $S/ragtruth_response.jsonl; get $GH/dataset/source_info.jsonl $S/ragtruth_source_info.jsonl
lic $GH/README.md ragtruth_README.md; lic $GH/LICENSE ragtruth_LICENSE
# ---- When2Call train_pref (kit pin 0582f774)
WR=0582f7749df63a96fdc3070932e83e72396ace53; HF=https://huggingface.co/datasets/nvidia/When2Call/resolve/$WR
get $HF/train/when2call_train_pref.jsonl $S/when2call_train_pref.jsonl
get $HF/test/when2call_test_mcq.jsonl $S/when2call_test_mcq.jsonl     # test: id ban + format reference only, never a row
lic $HF/README.md when2call_README.md
# ---- Habermas Machine comparisons (kit pin: repo 7923b719, parquet sha 7cf8d5ce)
HR=7923b71966c14136077c78d7841bc9e1a182dfe0; GH=https://raw.githubusercontent.com/google-deepmind/habermas_machine/$HR
get https://storage.googleapis.com/habermas_machine/datasets/hm_all_candidate_comparisons.parquet $S/hm_all_candidate_comparisons.parquet
[ "$(sha256sum < $S/hm_all_candidate_comparisons.parquet | cut -c1-64)" = 7cf8d5ce3fce8853b36f0ffe1158424f7813867e422e313db4df0a5f9e03e4a4 ] && log "habermas sha ok" || log "HABERMAS_SHA_MISMATCH"
lic $GH/README.md habermas_README.md; lic $GH/LICENSE habermas_LICENSE
# ---- SuperGPQA (rev 4430d445) + MMLU-CF (filter only, never rows)
SR=4430d4458112c7d4497fdcf94d7cc223313d6acf; HF=https://huggingface.co/datasets/m-a-p/SuperGPQA/resolve/$SR
get $HF/SuperGPQA-all.jsonl $S/SuperGPQA-all.jsonl; lic $HF/README.md supergpqa_README.md
curl -s https://huggingface.co/api/datasets/microsoft/MMLU-CF > $L/mmlucf_api.json
MR=$(python3 -c "import json; print(json.load(open('$L/mmlucf_api.json'))['sha'])"); log "mmlu-cf sha $MR"
for p in $(python3 -c "import json; print(' '.join(s['rfilename'] for s in json.load(open('$L/mmlucf_api.json'))['siblings'] if s['rfilename'].endswith(('.parquet','.json','.jsonl','.csv')) and 'test' not in s['rfilename'].lower()))"); do
  mkdir -p $S/mmlucf/$(dirname $p); get https://huggingface.co/datasets/microsoft/MMLU-CF/resolve/$MR/$p $S/mmlucf/$p; done
lic https://huggingface.co/datasets/microsoft/MMLU-CF/resolve/$MR/README.md mmlucf_README.md
# ---- gate inputs: FS1's list + v8's held-out files
for m in headfit cal dev when2call_4opt_clean_dev; do aws s3 cp --only-show-errors $B/engineering/coteach_wave4_20261028/p9_stage2_manifests/$m.jsonl $G/$m.jsonl || log "FAIL gate $m"; done
aws s3 cp --only-show-errors $E/manifests/esbar_eval_fresh_v1.jsonl $G/eval_fresh.jsonl || log "FAIL eval_fresh"
for h in aqua boardgame folio medmcqa medqa pubmedqa_art ruletaker stepgame; do aws s3 cp --only-show-errors $E/teacher_distill/pools_v4/heldout/HO_${h}_v9.jsonl.gz $G/HO_${h}_v9.jsonl.gz; done
aws s3 cp --only-show-errors $E/v7/data/ref/cov2_v6_heldout_manifest.jsonl.gz $G/cov2_heldout_excl.jsonl.gz
[ "$(md5sum < $G/cov2_heldout_excl.jsonl.gz | cut -c1-8)" = 81245eeb ] || log "COV2_MD5_MISMATCH"
cp /data/local/v8build/heldout/*.jsonl $G/; mkdir -p $G/v7; cp /data/local/v8build/heldout/v7/HO_*.jsonl $G/v7/
(cd $G && sha256sum dev.jsonl cal.jsonl headfit.jsonl when2call_4opt_clean_dev.jsonl eval_fresh.jsonl) >> $LOG
# ---- code (pinned esbar_code_v2e, the FS1 trainer text) + tokenizer
if [ ! -s $W/code/esbar_code_v2e/trainers/fullft_train.py ]; then
  aws s3 cp --only-show-errors $E/code/esbar_code_v2e_full.tgz $W/code2e.tgz; mkdir -p $W/code; (cd $W/code && tar xzf $W/code2e.tgz); fi
log "code md5 tgz $(md5sum < $W/code2e.tgz | cut -c1-8) fullft $(md5sum < $W/code/esbar_code_v2e/trainers/fullft_train.py | cut -c1-8)"
mkdir -p $W/tok; cp -n /data/local/v8build/tok/* $W/tok/
log STAGE_DONE
