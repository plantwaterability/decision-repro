#!/bin/bash
# DH round H staging on ctl2 (CPU): tokenless downloads at pinned revisions, sha256 of every download, licence evidence.
set -u; unset ASCALE_RUN_ID; export AWS_DEFAULT_REGION=us-east-2
W=/data/shared/dh; S=$W/src; L=$W/lic; mkdir -p $S $L $W/logs $W/out; LOG=$W/logs/stage.log
log() { echo "[stage_h $(date -u +%FT%TZ)] $*" >> $LOG; }
get() { [ -s "$2" ] || curl -sSL --fail --retry 3 -o "$2" "$1" || { log "FAIL $1"; rm -f "$2"; return 1; }
        echo "$(sha256sum < "$2" | cut -c1-64)  $(basename $2)  $1" >> $S/dl_sha256.txt; log "got $(basename $2) $(stat -c %s "$2")"; }
lic() { curl -sSL --retry 3 -o "$L/$2" "$1"; echo "$(sha256sum < $L/$2 | cut -c1-64)  $2  $1  http=$(curl -s -o /dev/null -w '%{http_code}' "$1")" >> $L/LIC_SHA256.txt; }
log start
# cfcolor (kit pin: zip sha256 47c07095...)
get https://www.dgp.toronto.edu/~donovan/cfcolor/cfcolor.zip $S/cfcolor.zip
[ "$(sha256sum < $S/cfcolor.zip | cut -c1-64)" = 47c07095642cfab3c2eeab366a5d152b07783cbb5af390cbfda7d7c13db4b54c ] && log "cfcolor sha ok" || log CFCOLOR_SHA_MISMATCH
lic https://www.dgp.toronto.edu/~donovan/color/ cfcolor_project_page.html; lic https://www.dgp.toronto.edu/~donovan/cfcolor/ cfcolor_dir_page.html
mkdir -p $S/cfcolor; (cd $S/cfcolor && python3 -c "
import zipfile; z=zipfile.ZipFile('$S/cfcolor.zip'); print([n for n in z.namelist()][:60]); z.extractall('.')" > $W/logs/cfcolor_names.txt 2>&1)
# HellaSwag train (kit pin 218ec52)
HR=218ec52e09a7e7462a5400043bb9a69a41d06b76; HF=https://huggingface.co/datasets/Rowan/hellaswag/resolve/$HR
get $HF/data/train-00000-of-00001.parquet $S/hellaswag_train.parquet; lic $HF/README.md hellaswag_README.md
# BANKING77 train (PolyAI; pin the commit that master points to now)
curl -sSL -o $L/polyai_master.json https://api.github.com/repos/PolyAI-LDN/task-specific-datasets/commits/master
BR=$(python3 -c "import json; print(json.load(open('$L/polyai_master.json'))['sha'])"); log "polyai master $BR"; echo $BR > $S/banking77_commit.txt
GH=https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/$BR
get $GH/banking_data/train.csv $S/banking77_train.csv; get $GH/banking_data/categories.json $S/banking77_categories.json
[ "$(sha256sum < $S/banking77_categories.json | cut -c1-64)" = 53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63 ] && log "banking categories sha = kit pin" || log BANKING_CATEGORIES_DIFFER
lic $GH/README.md polyai_README.md; lic $GH/LICENSE polyai_LICENSE; lic $GH/banking_data/README.md polyai_banking_README.md
# GSM8K train (kit pin 740312a)
GR=740312add88f781978c0658806c59bc2815b9866; HF=https://huggingface.co/datasets/openai/gsm8k/resolve/$GR
get $HF/main/train-00000-of-00001.parquet $S/gsm8k_train.parquet; lic $HF/README.md gsm8k_README.md
# MultiRC train (SuperGLUE v2 distribution)
get https://dl.fbaipublicfiles.com/glue/superglue/data/v2/MultiRC.zip $S/MultiRC.zip
mkdir -p $S/multirc; (cd $S/multirc && python3 -c "
import zipfile; z=zipfile.ZipFile('$S/MultiRC.zip'); print(z.namelist()); z.extract('MultiRC/train.jsonl', '.')" > $W/logs/multirc_names.txt 2>&1)
lic https://cogcomp.seas.upenn.edu/multirc/ multirc_page.html
# GoEmotions train (pin the revision the Hub reports now)
curl -sSL -o $L/goemotions_api.json https://huggingface.co/api/datasets/google-research-datasets/go_emotions
ER=$(python3 -c "import json; print(json.load(open('$L/goemotions_api.json'))['sha'])"); log "go_emotions sha $ER"; echo $ER > $S/goemotions_rev.txt
HF=https://huggingface.co/datasets/google-research-datasets/go_emotions/resolve/$ER
get $HF/simplified/train-00000-of-00001.parquet $S/goemotions_train.parquet; lic $HF/README.md goemotions_README.md
ls -la $S >> $LOG; log STAGE_DONE; tail -30 $LOG; cat $W/logs/cfcolor_names.txt | cut -c1-600; cat $W/logs/multirc_names.txt
grep -i -E 'licen|copyright|terms|permission|research|commercial' $L/cfcolor_project_page.html | cut -c1-400 | head -12
grep -i -E '^license|licen' $L/hellaswag_README.md $L/gsm8k_README.md $L/goemotions_README.md | head -8 | cut -c1-200; head -c 400 $L/polyai_LICENSE; grep -i -E 'licen' $L/polyai_README.md $L/polyai_banking_README.md | head -5 | cut -c1-300
grep -i -E 'licen|terms|copyright' $L/multirc_page.html | head -5 | cut -c1-300
