#!/bin/bash
# v8 final: joined new rows -> esbar_stage1_v8.jsonl (assemble_v8.py) -> reader check -> stage1_v8.sh / resume_v8.sh (staged, NOT launched)
# -> S3 (md5 read-back, new keys only) -> BUILD_V8_OK (the only marker the chain trusts).
set -u; export AWS_DEFAULT_REGION=us-east-2
V=/data/local/v8build; S3R=s3://YOUR_BUCKET/jev-race/engineering/esbar/runs/esbar_v8; O=$V/final; mkdir -p $O
md5() { md5sum < "$1" | cut -c1-32; }
die() { echo "DIE $*"; exit 1; }
put() { aws s3 cp --only-show-errors "$1" "$2" && [ "$(aws s3 cp --only-show-errors "$2" - | md5sum | cut -c1-32)" = "$(md5 "$1")" ] && echo "OK $(basename $2) $(md5 $1 | cut -c1-8)" || die "PUT $1"; }
aws s3 cp --only-show-errors $S3R/parent/V8PREP_OK $O/V8PREP_OK || die NO_V8PREP_OK
JM=$(sed -n 's/.*joined_md5=\([0-9a-f]*\).*/\1/p' $O/V8PREP_OK)
aws s3 cp --only-show-errors $S3R/parent/v8_new_joined.jsonl $O/v8_new_joined.jsonl && [ "$(md5 $O/v8_new_joined.jsonl)" = "$JM" ] || die JOINED_MD5
NB=$(python3 -c "import json; r=json.load(open('$V/out/v8_merge_report.json')); print(r['base_rows_v75'] - r['base_removed'])")
python3 $V/scripts/assemble_v8.py $V/out/v8_prejoin.jsonl $O/v8_new_joined.jsonl $NB $O/esbar_stage1_v8.jsonl /data/local/v75build/esbar_stage1_v75.jsonl.header.json > $O/assemble.json 2> $O/assemble.err || { tail -5 $O/assemble.err; die ASSEMBLE; }
cat $O/assemble.json
python3 - $O/esbar_stage1_v8.jsonl > $O/yn_share.json <<'PY'
import json, re, sys
y = n = 0.0; ry = rn = 0
for l in open(sys.argv[1]):
    r = json.loads(l)
    if len(r["option_order"]) != 2: continue
    p = r["prompt"]; blk = p[p.rfind("\n\nOptions:\n") + 11:p.rfind("\n\nAnswer with")]
    opts = re.findall(r"^[A-Za-z]\) (.*)$", blk, re.M)
    gk = max(r["gold"], key=r["gold"].get); gtxt = opts[r["option_order"].index(gk)] if len(opts) == 2 else None
    if gtxt == "Yes": y += r.get("weight", 1.0); ry += 1
    elif gtxt == "No": n += r.get("weight", 1.0); rn += 1
print(json.dumps({"rows_yes": ry, "rows_no": rn, "weighted_yes_share": round(y / (y + n), 4), "row_yes_share": round(ry / (ry + rn), 4)}))
PY
cat $O/yn_share.json
SHA=$(python3 -c "import json; print(json.load(open('$O/esbar_stage1_v8.jsonl.header.json'))['rows_sha256'])"); [ "$(sha256sum < $O/esbar_stage1_v8.jsonl | cut -c1-64)" = "$SHA" ] || die FILE_SHA
ROWS=$(python3 -c "import json; print(json.load(open('$O/esbar_stage1_v8.jsonl.header.json'))['rows'])"); STEPS=$(( (ROWS + 63) / 64 )); HALF=$(( STEPS / 2 ))
echo $STEPS > $O/steps.txt
sed -e "s/^# ESBAR-V75 stage 1 (PREREG v7.5 = v7.4 + AMENDMENT 1 minus the 914 phish_v5 generic-webmail shared-document other-domain rows): init = E1 (v5 export, identity c92e70ea, head excluded); ONE schedule of 3,392 steps in two legs:/# ESBAR-V8 stage 1 (PREREG v8 = v7.5 + clean official train splits + ESCI train, minus 43 suite-exact rows): init = E1 (v5 export, identity c92e70ea, head excluded); ONE schedule of $STEPS steps in two legs:/" \
    -e "s/^# leg 1 --max-steps 1696 -> export_half, leg 2 resumes -> 3392 -> export_full./# leg 1 --max-steps $HALF -> export_half, leg 2 resumes -> $STEPS -> export_full./" \
    -e 's#F=$D/esbar_v75; RUN=$F/run_v75;#F=$D/esbar_v8; RUN=$F/run_v8;#' \
    -e "s#PM=\$F/manifests/esbar_stage1_v75.jsonl; PMSHA_EXPECT=7146db56f3aa07522f839a86e5a5f98cc488d7a38ca3278a8ae992bbf0823862#PM=\$F/manifests/esbar_stage1_v8.jsonl; PMSHA_EXPECT=$SHA#" \
    -e 's#runs/esbar_v75$#runs/esbar_v8#' -e 's#\[stage1_v75 #[stage1_v8 #' -e 's#STAGE1_V75_FAILED#STAGE1_V8_FAILED#' -e 's#STAGE1_V75_DONE#STAGE1_V8_DONE#' \
    -e 's#for x in esbar_stage1_v75.jsonl esbar_stage1_v75.jsonl.header.json; do#for x in esbar_stage1_v8.jsonl esbar_stage1_v8.jsonl.header.json; do#' \
    -e "s#\[ \"\$STEPS\" = 3392 \] || die \"STEPS \$STEPS\"#[ \"\$STEPS\" = $STEPS ] || die \"STEPS \$STEPS\"#" \
    -e 's#--rdzv-endpoint 127.0.0.1:29785#--rdzv-endpoint 127.0.0.1:29787#' \
    -e "s#log \"LEG1 start: max-steps 1696 of schedule 3392\"; t0=\$(date +%s); train 1696;#log \"LEG1 start: max-steps $HALF of schedule $STEPS\"; t0=\$(date +%s); train $HALF;#" \
    -e "s#log \"LEG2 start: resume -> 3392\"; t0=\$(date +%s); train 3392;#log \"LEG2 start: resume -> $STEPS\"; t0=\$(date +%s); train $STEPS;#" $V/stage1_v75.sh > $O/stage1_v8.sh
bash -n $O/stage1_v8.sh || die STAGE1_SYNTAX
grep -nE "v75|3392|1696|7146db56|29785" $O/stage1_v8.sh && die STAGE1_LEFTOVER
M8=$(md5 $O/stage1_v8.sh | cut -c1-8)
sed -e 's#esbar_v75#esbar_v8#g' -e 's#stage1_v75.sh#stage1_v8.sh#g' -e "s#= c02189dc ]#= $M8 ]#" -e 's#\[resume_v75 #[resume_v8 #' -e 's#(p9 after v7.3)#(p9 after v7.5)#' $V/resume_v75.sh > $O/resume_v8.sh
sed -i "s#\[ \"\$(md5sum < /data/local/esbar_v8/scripts/stage1_v8.sh | cut -c1-8)\" = [0-9a-f]* \]#[ \"\$(md5sum < /data/local/esbar_v8/scripts/stage1_v8.sh | cut -c1-8)\" = $M8 ]#" $O/resume_v8.sh
bash -n $O/resume_v8.sh || die RESUME_SYNTAX; grep -c "$M8" $O/resume_v8.sh
for x in "$O/esbar_stage1_v8.jsonl manifests/esbar_stage1_v8.jsonl" "$O/esbar_stage1_v8.jsonl.header.json manifests/esbar_stage1_v8.jsonl.header.json" \
         "$O/steps.txt registered/steps.txt" "$O/stage1_v8.sh scripts/stage1_v8.sh" "$O/resume_v8.sh scripts/resume_v8.sh" "$O/assemble.json logs/assemble.json" \
         "$O/yn_share.json logs/yn_share.json" "$V/scripts/final_v8.sh scripts/final_v8.sh"; do set -- $x; put $1 $S3R/$2; done
echo "$(date -u +%FT%TZ) rows=$ROWS steps=$STEPS half=$HALF rows_sha256=$SHA manifest_md5=$(md5 $O/esbar_stage1_v8.jsonl) stage1_v8=$M8 resume_v8=$(md5 $O/resume_v8.sh | cut -c1-8)" > $O/BUILD_V8_OK
put $O/BUILD_V8_OK $S3R/BUILD_V8_OK
cat $O/BUILD_V8_OK
