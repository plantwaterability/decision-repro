R="s/\[f\]s4/[f]s$N/g; s/fs4/fs$N/g; s/FS4/FS$N/g"
sed -e "$R" -e "s/--mlp-chunk-min 16384 --seed 3 /--mlp-chunk-min 16384 --seed $SEED /; s/with --seed 3 (the trainer/with --seed $SEED (the trainer/" stage1_fs4.sh > stage1_fs$N.sh
if [ "$N" = 5 ]; then python3 - stage1_fs5.sh <<'PY' || exit 1
import sys
p = sys.argv[1]; s = open(p).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:60]; s = s.replace(a, b)
rep("PM=$F/manifests/esbar_stage1_fs1.jsonl; PMSHA_EXPECT=8461365da84aaf5e0f3528900b314755b080212d6ea2cda9ceb0f41111cad644; STEPS_EXPECT=6011; LEG1=2400;", "LEG1=2400;")
rep("S3M=$S3/engineering/esbar/runs/esbar_fs1/manifests", "S3M=$S3/engineering/esbar/runs/esbar_fs5/manifests")
rep("for x in esbar_stage1_fs1.jsonl esbar_stage1_fs1.jsonl.header.json; do", '''# FS5: the data agent's manifest, discovered: exactly one *.jsonl.header.json in esbar_fs5/manifests/ (its .jsonl must hash to rows_sha256)
HN=$(aws s3 ls $S3M/ | awk '{print $4}' | grep '[.]jsonl[.]header[.]json$'); [ "$(echo "$HN" | grep -c .)" = 1 ] || die "MANIFEST_DISCOVERY: $(echo $HN)"
MN=${HN%.header.json}; PM=$F/manifests/$MN
for x in $MN $HN; do''')
rep('[ "$HS" = "$PMSHA_EXPECT" ] && [ "$FS" = "$PMSHA_EXPECT" ] || die "MANIFEST_SHA header=$HS file=$FS expect=$PMSHA_EXPECT"',
    '[ -n "$HS" ] && [ "$HS" = "$FS" ] || { mv $PM $PM.bad_$(date +%s); mv $PM.header.json $PM.header.json.bad_$(date +%s); die "MANIFEST_SHA header=$HS file=$FS (incomplete or changed upload; local copies set aside)"; }; PMSHA_EXPECT=$HS')
rep('[ "$STEPS" = "$STEPS_EXPECT" ] || die "STEPS $STEPS"', '[ "$STEPS" -gt "$LEG1" ] || die "STEPS $STEPS"')
open(p, "w").write(s)
PY
fi
