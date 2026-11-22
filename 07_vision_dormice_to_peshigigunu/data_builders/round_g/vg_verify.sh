#!/bin/bash
# Round G delivery check, from S3 only (what a trainer node will see): (1) every round G image tar on S3 has the md5 in recipe_g/MD5SUMS and
# holds every image the UG manifests name (ade_cnt rows: recipe_e's img_ade_cnt.tar); (2) each manifest on S3 has the rows_sha256 of its header
# and the md5 in recipe_g/MD5SUMS; (3) the arm script's image-tar lookup, with recipe_g added to its list, finds a tar for every source.
set -u
export AWS_DEFAULT_REGION=us-east-2
V=s3://YOUR_BUCKET/jev-race/engineering/vision; SG=$V/recipe_g
W=/data/shared/visdata_g/verify; mkdir -p $W/md5 $W/names $W/man; cd $W
aws s3 cp --quiet $SG/MD5SUMS $W/md5/recipe_g
for r in recipe_a recipe_b recipe_b3 recipe_c recipe_d recipe_e recipe_f; do aws s3 cp --quiet $V/$r/MD5SUMS $W/md5/$r; done
bad=0
for t in $(grep -o " img_[a-z0-9_]*\.tar$" $W/md5/recipe_g | tr -d ' '); do
  s=${t#img_}; s=${s%.tar}
  got=$(aws s3 cp --quiet $SG/img/$t - | tee >(tar t > $W/names/$s.txt) | md5sum | cut -d" " -f1)
  want=$(grep " $t$" $W/md5/recipe_g | cut -d" " -f1)
  [ "$got" = "$want" ] && echo "TAR_OK $t $got $(grep -c . $W/names/$s.txt) entries" || { echo "TAR_BAD $t got=$got want=$want"; bad=$((bad+1)); }
done
if [ ! -s $W/names/ade_cnt.txt ]; then
  got=$(aws s3 cp --quiet $V/recipe_e/img/img_ade_cnt.tar - | tee >(tar t > $W/names/ade_cnt.txt) | md5sum | cut -d" " -f1)
  want=$(grep " img_ade_cnt.tar$" $W/md5/recipe_e | cut -d" " -f1); [ "$got" = "$want" ] && echo "TAR_OK recipe_e img_ade_cnt.tar $got" || { echo "TAR_BAD recipe_e img_ade_cnt.tar"; bad=$((bad+1)); }
fi
for m in "$@"; do
  aws s3 cp --quiet $SG/manifests/$m.jsonl $W/man/ && aws s3 cp --quiet $SG/manifests/$m.header.json $W/man/
  a=$(sha256sum < $W/man/$m.jsonl | cut -d" " -f1); b=$(python3 -I -c "import json; print(json.load(open('$W/man/$m.header.json'))['rows_sha256'])")
  c=$(md5sum < $W/man/$m.jsonl | cut -d" " -f1); d=$(grep " $m.jsonl$" $W/md5/recipe_g | cut -d" " -f1)
  [ "$a" = "$b" ] && [ "$c" = "$d" ] && echo "MANIFEST_OK $m rows_sha256=$a md5=$c rows=$(wc -l < $W/man/$m.jsonl)" || { echo "MANIFEST_BAD $m $a $b $c $d"; bad=$((bad+1)); }
  : > $W/man/$m.lookup
  for fam in $(python3 -I -c "import json; print(' '.join(sorted({json.loads(l)['source'] for l in open('$W/man/$m.jsonl')})))"); do
    PFX=""; for r in recipe_a recipe_b recipe_b3 recipe_c recipe_d recipe_e recipe_f recipe_g; do grep -q " img_$fam.tar$" $W/md5/$r && { PFX=$r; break; }; done
    [ -n "$PFX" ] || { echo "LOOKUP_BAD $m $fam: in no MD5SUMS"; bad=$((bad+1)); }
    echo "$fam $PFX" >> $W/man/$m.lookup
  done
  echo "LOOKUP $m: $(sort -u $W/man/$m.lookup | awk '{print $2}' | sort | uniq -c | tr '\n' ' ')"
  for fam in $(python3 -I -c "import json; print(' '.join(json.load(open('$W/man/$m.header.json'))['new_sources']))"); do
    got=$(grep "^$fam " $W/man/$m.lookup | awk '{print $2}'); want=recipe_g; [ "$fam" = "ade_cnt" ] && want=recipe_e
    [ "$got" = "$want" ] && echo "LOOKUP_NEW_OK $fam -> $got" || { echo "LOOKUP_NEW_BAD $fam -> $got (want $want)"; bad=$((bad+1)); }
  done
  python3 -I - $m <<'PY'
import json, os, sys
W = "/data/shared/visdata_g/verify"; m = sys.argv[1]
names = {f[:-4]: {l.strip() for l in open(f"{W}/names/{f}")} for f in os.listdir(W + "/names")}
hdr = json.load(open(f"{W}/man/{m}.header.json")); new = set(hdr["new_sources"])
need, miss = 0, 0
for l in open(f"{W}/man/{m}.jsonl"):
    r = json.loads(l)
    if r["source"] in new:
        for p in r["images"]:
            need += 1; miss += p not in names[r["source"]]
print(f"IMAGES_IN_TARS {m}: round G image references {need} (sources {sorted(new)}), missing from the S3 tars {miss}")
sys.exit(1 if miss else 0)
PY
  [ $? -eq 0 ] || bad=$((bad+1))
done
echo "G_VERIFY_DONE bad=$bad"
