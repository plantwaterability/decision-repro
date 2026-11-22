#!/bin/sh
# ss_corpus.sh PACK [SPEC]: build the GPU SCREENSAVER's "ALL OUR DATA" corpus ON A NODE (never the laptop), once, and upload it pinned:
#   s3://.../jev-race/screensaver/data/PACK.tgz + PACK.manifest.json + PACK.verify.json + PACK.spec.tsv + PACK.md5 (the md5 is written
#   LAST with a conditional put, so a pack is complete or absent; an existing PACK.md5 is never replaced: a new build = a new name).
# SPEC (fleetq/screensaver/corpus_spec.tsv when present, else fleet/ss_corpus_spec.tsv) lists every corpus the fleet has, one per line:
#   name  s3_key[|alt_key]  md5  top  role  line  train_dirs  val_dirs  note
#   md5: 32 hex = pinned (a mismatch refuses the build); "-" = unpinned (recorded as fetched); "?" = unpinned and OPTIONAL (skipped
#        with a note when no key exists)
#   role train:      train_dirs (relative to top) are a GROUP of the pack; val_dirs its val group (monitoring only); the source's OWN
#                    eval splits (dirs named val* dev* test* eval* heldout* novel* confirm* sealed* bench* probe* extended* overlap*
#                    ood* unlabel* teacher_fill* gde1* meragpt*, outside train*) join the exclusion sets automatically
#   role evalsplits: nothing trained; only its eval-named splits are exclusion sets (raw / duplicate / superseded sources)
#   role eval:       the whole extracted tree is an exclusion set (held-out corpora, eval suites, a stack pool whose train/ is held out)
#   role index:      a sealed exclusion index (json.gz; research/final_suite.md 3.4): ids, state sha1s, entity keys, clusters
#   role td:         typed-decisions data: excluded from every pack but td1, the FITTED variant's
#   role skip:       never fetched (sealed items, confirm splits, internal-only or do-not-train sets); listed for the inventory
#   line clean / nc / mixed: nc = every record tagged NC (the trainer drops it unless --allow-nc); mixed = per-record licence fields
# Always excluded as well: jev_real val + test (every jev-bench config and typed-decisions; built by convert_jevbench.py when missing),
# and the TD train split (except in td1: its train is TD train minus a 1-in-20 hashed holdout = its val; jev_real has no TD val).
# Every record also passes the trainer's own loader (cache_features.question_options) or is dropped as "schema". Exclusions are by
# normalised state, state-less id and the sealed indexes (ss_lib.py pack), re-checked on the written pack, then re-verified from the
# TARBALL before upload; the upload is read back and its md5 compared (backup rule).
# Run it as a queue job (CPU work; ~2-3x the corpora's size of local disk):  ss_corpus.sh all1 | ss_corpus.sh td1
# MEMORISATION SUBSET (review LATE P2-3): PACK = mem<N>k (e.g. mem20k) is not built from the spec but drawn from a built parent pack
# ($SS_MEM_PARENT, default all1; md5-checked against its PACK.md5): N x 1000 CLEAN train units (or $SS_MEM_UNITS), whole records, a
# seeded ($SS_MEM_SEED, default 20261008) draw PROPORTIONAL to each source's size with a small-source floor (ss_lib.py subset:
# per-source quota = max(min(available, 100), proportional share) -- not even across sources: mem20k has courts 2,188, clinvar 1,244,
# typed_synth 1,240, Bayesian worlds 422 = 2.1%); val = the same rule over the parent's val (2,000 units). Its manifest is in the
# regular pack schema (ss_lib normalize_pack) with the selection rule / seed / quotas under subset, the parent's md5 / inputs /
# exclusion history / own drops under parent_pack. The parent's exclusion record is inherited and the
# subset is re-verified against the sealed indexes; its manifest records the parent md5, the rule, per-source quotas and draws.
PACK=${1:?usage: ss_corpus.sh PACK [SPEC]}; SPEC=${2:-}
J=${JEV_J:-/data/shared/jev}; D=${JEV_D:-/data/local}; FL=${JEV_FLEET:-/data/shared/fleet}
BK=${SS_BUCKET:-YOUR_BUCKET}; S3=s3://$BK/jev-race; FQ=$S3/fleetq; SR=$S3/screensaver; PY=${SS_PY:-python3}
W=$D/ss_corpus/$PACK; X=$D/ss_src; mkdir -p $W $X
say() { echo "$(date -u +%FT%TZ) [ss_corpus $PACK] $*"; }
die() { say "FAILED: $*"; exit 1; }
case "$PACK" in *[!a-z0-9_]*|"") die "PACK must be [a-z0-9_]";; esac
aws s3api head-object --bucket $BK --key jev-race/screensaver/data/$PACK.md5 > /dev/null 2>&1 && { say "already built: $(aws s3 cp --only-show-errors $SR/data/$PACK.md5 - | cut -c1-32)"; exit 0; }
L=$W/ss_lib.py
aws s3 cp --only-show-errors $FQ/scripts/ss_lib.py $L 2>/dev/null || cp $FL/ss_lib.py $L || die "no ss_lib.py"
if [ -z "$SPEC" ]; then SPEC=$W/spec.tsv
  aws s3 cp --only-show-errors $FQ/screensaver/corpus_spec.tsv $SPEC 2>/dev/null || cp $FL/ss_corpus_spec.tsv $SPEC || die "no spec"; fi
cd $J || die "no repo at $J"
case "$PACK" in mem[0-9]*k)   # ------------------------------------------- the memorisation subset of a built parent pack
  PAR=${SS_MEM_PARENT:-all1}; NU=${SS_MEM_UNITS:-$(( $(echo $PACK | sed 's/^mem\([0-9]*\)k$/\1/') * 1000 ))}; SEED=${SS_MEM_SEED:-20261008}
  PM=$(aws s3 cp --only-show-errors $SR/data/$PAR.md5 - 2>/dev/null | cut -c1-32); [ -n "$PM" ] || die "parent pack $PAR is not built (no $PAR.md5)"
  P0=$W/parent; mkdir -p $P0
  aws s3 cp --only-show-errors $SR/data/$PAR.tgz $P0/p.tgz || die "parent download"
  [ "$(md5sum < $P0/p.tgz | cut -c1-32)" = "$PM" ] || die "parent $PAR md5 differs from its PACK.md5"
  tar xzf $P0/p.tgz -C $P0 && rm -f $P0/p.tgz || die "parent extract"
  B=$W/build; [ -e $B ] && mv $B $W/build.old.$$; mkdir -p $B
  $PY $L subset --pack $P0/$PAR --out $B/$PACK --units $NU --seed $SEED --parent-md5 $PM > $W/subset.log 2>&1 || die "subset: $(tail -2 $W/subset.log | tr '\n' ' ')"
  say "$(tail -1 $W/subset.log)"
  IX=""; for n in rd_exclusion_index:${SS_RD_INDEX_MD5:-28a13a92edc55b38da3db5d4f5f93ef5} hl_exclusion_index:${SS_HL_INDEX_MD5:-f047836e7cf6be1f9dcde484d1318676}; do
    f=$W/${n%%:*}.json.gz; aws s3 cp --only-show-errors $S3/sealed/final_v1/${n%%:*}.json.gz $f && [ "$(md5sum < $f | cut -c1-32)" = "${n#*:}" ] \
      || die "sealed index ${n%%:*} missing or md5 differs"; IX="$IX --exclude ${n%%:*}=index:$f"; done
  M=$($PY $L tar --dir $B/$PACK --out $B/$PACK.tgz) || die "tar"
  V=$W/verify; mkdir -p $V && tar xzf $B/$PACK.tgz -C $V || die "re-extract"
  $PY $L verify --pack $V/$PACK $IX > $W/verify.json || die "VERIFY FAILED on the tarball: $(cut -c1-300 $W/verify.json)"
  for x in tgz:$B/$PACK.tgz manifest.json:$B/$PACK/manifest.json verify.json:$W/verify.json; do
    aws s3 cp --only-show-errors ${x#*:} $SR/data/$PACK.${x%%:*} || die "upload ${x%%:*}"; done
  aws s3 cp --only-show-errors $SR/data/$PACK.tgz $W/readback.tgz && [ "$(md5sum < $W/readback.tgz | cut -c1-32)" = "$M" ] || die "READ-BACK md5 differs"
  echo "$M" > $W/md5
  aws s3api put-object --bucket $BK --key jev-race/screensaver/data/$PACK.md5 --body $W/md5 --if-none-match '*' > /dev/null 2>&1 || die "PACK.md5 exists (another build won): not replaced"
  say "PACK_DONE $PACK md5 $M (subset of $PAR $PM), read back OK"
  find $P0 $V $W/readback.tgz -type f -delete 2>/dev/null; find $P0 $V -depth -type d -empty -delete 2>/dev/null
  exit 0;;
esac
LC=""; [ -f $J/trainers/cache_features.py ] && LC="--loader-check $J/trainers"
[ -n "$LC" ] || [ -n "$SS_NO_LOADER_CHECK" ] || die "no $J/trainers/cache_features.py for the loader check"
# ------------------------------------------------------------------ fetch every listed source (md5-checked when pinned)
G=""; E=""; IN=""; TDG=""; NOTES=""
while IFS='	' read -r name key md5 top role line trd vad note; do
  case "$name" in ""|\#*|name) continue;; esac
  [ "$role" = skip ] && { say "skip $name: $note"; continue; }
  dst=$X/$name
  if [ ! -f $dst/.ss_ok ]; then
    src=""; for k in $(echo "$key" | tr '|' ' '); do aws s3api head-object --bucket $BK --key jev-race/$k > /dev/null 2>&1 && { src=$k; break; }; done
    if [ -z "$src" ]; then
      [ "$md5" = "?" ] && { say "optional $name: no key ($key): skipped"; NOTES="$NOTES --input $name=ABSENT"; continue; }
      die "NO_KEY $key"
    fi
    t=$X/.$name.dl; aws s3 cp --only-show-errors $S3/$src $t || die "download $src"
    got=$(md5sum < $t | cut -c1-32)
    case "$md5" in -|"?") ;; *) [ "$md5" = "$got" ] || die "TGZ_MD5 $src: $got, pinned $md5";; esac
    x=$X/.x_$name; mkdir -p $x
    if [ "$role" = index ]; then mv $t $x/index.json.gz
    else tar xzf $t -C $x || die "UNTAR $src"; find $x -name '._*' -delete; rm -f $t; fi
    [ -e $dst ] && mv $dst $dst.old.$$
    mv $x $dst && echo "$got" > $dst/.ss_ok
  fi
  got=$(cat $dst/.ss_ok); IN="$IN --input $name=$got"; root=$dst/$top
  case "$role" in
    index) E="$E --exclude $name=index:$dst/index.json.gz"; continue;;
  esac
  [ -d "$root" ] || die "$name: no $top inside its tarball"
  case "$role" in
    train)
      tr=""; for d in $(echo "$trd" | tr ',' ' '); do [ -d $root/$d ] || die "$name: no train dir $top/$d"; tr="$tr${tr:+,}$root/$d"; done
      va=""; [ "$vad" = - ] || for d in $(echo "$vad" | tr ',' ' '); do [ -d $root/$d ] && va="$va${va:+,}$root/$d"; done
      fl=""; [ "$line" = nc ] && fl=@nc
      [ "$PACK" = td1 ] || G="$G --group $name$fl=$tr${va:+:$va}"
      E="$E --exclude ${name}_evalsplits=evalsplits:$root";;
    evalsplits) E="$E --exclude ${name}_evalsplits=evalsplits:$root";;
    eval) E="$E --exclude $name=$root";;
    td) TDG="$TDG${TDG:+,}$root";;
    *) die "$name: unknown role $role";;
  esac
done < $SPEC
# ------------------------------------------------------------------ jev_real (jev-bench + typed-decisions): its val / test always excluded
JR=$D/jev_real
{ [ -f $JR/train/real.jsonl ] && [ -f $JR/val/real.jsonl ] && [ -f $JR/test/real.jsonl ]; } || { JR=$X/jev_real
  [ -f $JR/test/real.jsonl ] || $PY trainers/convert_jevbench.py --out $JR > $W/convert_jevbench.log 2>&1 || die "convert_jevbench (see $W/convert_jevbench.log)"; }
E="$E --exclude jev_real_valtest=$JR/val,$JR/test"
mkdir -p $W/td/train $W/td/val
$PY - $JR $W/td <<'PY' || die "TD split"
import hashlib, json, sys            # typed-decisions rows of jev_real train (and val, if it ever has any); td1's val = a hashed 1-in-20
n = {"train": 0, "val": 0}            # of TD train when jev_real val has no TD rows (it has none: AGENT_NOTES:396)
rows = {s: [l for l in open(f"{sys.argv[1]}/{s}/real.jsonl") if json.loads(l).get("source") == "typed-decisions"] for s in ("train", "val")}
hold = not rows["val"]
with open(f"{sys.argv[2]}/train/td.jsonl", "w") as tr, open(f"{sys.argv[2]}/val/td.jsonl", "w") as va:
    for l in rows["train"]:
        r = json.loads(l)
        if hold and int(hashlib.sha1(str(r.get("base_id") or r["id"]).encode()).hexdigest(), 16) % 20 == 0:
            va.write(l); n["val"] += 1
        else:
            tr.write(l); n["train"] += 1
    for l in rows["val"]:
        va.write(l); n["val"] += 1
assert n["train"] > 0, "no typed-decisions rows in jev_real train"
print("TD", n, "val = hashed holdout of TD train" if hold else "val = jev_real val TD rows")
PY
if [ "$PACK" = td1 ]; then   # the FITTED variant's pack: TD train only (+ any role=td corpus), val = its holdout; everything else is eval
  G="--group td=$W/td/train${TDG:+,$TDG}:$W/td/val"
else
  E="$E --exclude td_train=$W/td/train,$W/td/val${TDG:+ --exclude td_corpus=$TDG}"
fi
[ -n "$G" ] || die "no training groups"
# ------------------------------------------------------------------ build, verify from the tarball, upload, read back
B=$W/build; [ -e $B ] && mv $B $W/build.old.$$; mkdir -p $B
say "PACK groups:$(echo "$G" | grep -o -- '--group [^=]*' | sed 's/--group//' | tr '\n' ' ')"
$PY $L pack --out $B/$PACK $G $E $IN $NOTES $LC > $W/pack.log 2>&1 || die "pack: $(tail -3 $W/pack.log | tr '\n' ' ')"
tail -1 $W/pack.log
M=$($PY $L tar --dir $B/$PACK --out $B/$PACK.tgz) || die "tar"
V=$W/verify; mkdir -p $V && tar xzf $B/$PACK.tgz -C $V || die "re-extract"
$PY $L verify --pack $V/$PACK $E > $W/verify.json || die "VERIFY FAILED on the tarball: $(cut -c1-300 $W/verify.json)"
say "VERIFY_OK $(cut -c1-200 $W/verify.json)"
aws s3 cp --only-show-errors $B/$PACK.tgz $SR/data/$PACK.tgz || die "upload tgz"
aws s3 cp --only-show-errors $B/$PACK/manifest.json $SR/data/$PACK.manifest.json || die "upload manifest"
aws s3 cp --only-show-errors $W/verify.json $SR/data/$PACK.verify.json || die "upload verify"
aws s3 cp --only-show-errors $SPEC $SR/data/$PACK.spec.tsv || die "upload spec"
aws s3 cp --only-show-errors $SR/data/$PACK.tgz $W/readback.tgz || die "read back"
[ "$(md5sum < $W/readback.tgz | cut -c1-32)" = "$M" ] || die "READ-BACK md5 differs"
echo "$M" > $W/md5
aws s3api put-object --bucket $BK --key jev-race/screensaver/data/$PACK.md5 --body $W/md5 --if-none-match '*' > /dev/null 2>&1 || die "PACK.md5 exists (another build won): not replaced"
say "PACK_DONE $PACK md5 $M ($(du -k $B/$PACK.tgz | cut -f1) KiB), read back OK"
find $V $W/readback.tgz -type f -delete 2>/dev/null; find $V -depth -type d -empty -delete 2>/dev/null
exit 0
