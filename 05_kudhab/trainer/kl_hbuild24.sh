#!/bin/bash
# kl_hbuild24.sh NAME (TA agent; coordinator order ~22:30Z 11-16): make $K/data/NAME present for NAME = fs5hc (K24) = K23's set (fs5hn) without the HellaSwag pool:
# H0 rows (chess / bv1_discovery left out) + round-H pools sata_mrc, sata_goemo, gsm8k_ag, banking77 (NOT hellaswag, NOT cfcolor), DH's drop list left out, + an equal-size
# replay sample of fs5kt (seed 1). Fetched from s3 .../esbar_kl_data/ when built before, else picked here from fs5hh (kl_hbuild.sh's build, on S3) by kl_pick24.py, which
# stops unless the built set holds 0 hellaswag / cfcolor rows and the exact expected pool counts. kl_hbuild.sh and kl_pick.py are untouched. Exit 0 = $K/data/NAME.info.json present.
set -u; source /data/local/kl/scripts/kl_env.sh
NAME=${1:?}; [ "$NAME" = fs5hc ] || { echo "NAME must be fs5hc"; exit 1; }
W=$K/data; O=$R/esbar_kl_data; T=$R/esbar_mx/tl; P=$R/esbar_dh/pools; L=$K/logs/hbuild24.log; mkdir -p $W $K/logs
log() { echo "[kl_hbuild24 $(date -u +%FT%TZ)] $*" | tee -a $L; aws s3 cp --only-show-errors $L $O/hbuild24_$(hostname).log 2>/dev/null; }
die() { log "KL_HBUILD24_FAILED $*"; exit 1; }
[ -s $W/$NAME.info.json ] && exit 0
exec 9> $K/hbuild.lock; flock 9; [ -s $W/$NAME.info.json ] && exit 0
getset() {   # $1 name: from S3 into a side dir, sha256-checked, then moved into place (info.json last)
  local s=$1 D2=$W/dl_$1 f; mkdir -p $D2
  for f in ids.npy meta.npz info.json rowids.txt SHA256SUMS; do aws s3 cp --only-show-errors $O/$s.$f $D2/$s.$f || return 1; done
  (cd $D2 && sha256sum -c --quiet $s.SHA256SUMS) || return 1
  for f in ids.npy meta.npz rowids.txt SHA256SUMS info.json; do mv $D2/$s.$f $W/$s.$f; done; }
if aws s3 ls $O/$NAME.SHA256SUMS > /dev/null 2>&1; then getset $NAME || die "FETCH / SHA256 $NAME"; log "$NAME fetched from S3: $(tr -d '\n ' < $W/$NAME.info.json | cut -c1-300)"; exit 0; fi
log "building $NAME on $(hostname) (kl_pick24.py $(md5sum < $SC/kl_pick24.py | cut -c1-8))"
if [ -s $W/fs5hh.info.json ]; then aws s3 cp --only-show-errors $O/fs5hh.SHA256SUMS $W/fs5hh.SHA256SUMS.s3 && (cd $W && sha256sum -c --quiet fs5hh.SHA256SUMS.s3) || die "local fs5hh differs from S3's SHA256SUMS"
else getset fs5hh || die "FETCH / SHA256 fs5hh"; fi
fetch() { local u=$1 f=$2 s=$3; if [ -s $W/$f ] && [ "$(sha256sum < $W/$f | cut -c1-64)" = "$s" ]; then return 0; fi
  aws s3 cp --only-show-errors $u $W/$f.tmp && mv $W/$f.tmp $W/$f || die "FETCH $f"; [ "$(sha256sum < $W/$f | cut -c1-64)" = "$s" ] || die "SHA256 $f"; }
fetch $P/fs5h_rows.jsonl fs5h_rows.jsonl 5bcabd65de07e29a4eaa5c9a136886cb960c16467cd0faf17301e576f5aedbae
fetch $P/fs5h_rows.meta.jsonl fs5h_rows.meta.jsonl de0e2a72ce7de53348db6e99ecff3547633603ecd78d204620d532c68b4d87e9
fetch $T/mxa.jsonl mxa.jsonl e96f3489fed1f9f4517720c1270aad7bd70e5ba0e7a1e6a43e3f7a60e5507b52
fetch $R/esbar_dh/droplist_roundH.txt droplist_roundH.txt 6e10a9d5fdb34ccefc128fe8d1128747d3a8bf8da2a801a59cf5cc41d2516ab7
for f in fs5k.rowids.txt fs5k.meta.npz; do [ -s $W/$f ] || aws s3 cp --only-show-errors $O/$f $W/$f || die "FETCH $f"; done
aws s3 cp --only-show-errors $O/fs5hh_build.hh_exclude_ids.txt $W/hc_exclude_ids.txt || die "FETCH the fs5hh build's exclusion list"
ne=$(grep -c . $W/hc_exclude_ids.txt); [ "$ne" = 31947 ] || die "exclusion list has $ne ids, not 31,947 (H0 chess 26,586 + bv1_discovery 400 + drop list 4,961)"
nd=$(grep -v '^#' $W/droplist_roundH.txt | grep -c .); [ "$nd" = 4961 ] && [ "$(grep -v '^#' $W/droplist_roundH.txt | grep . | grep -Fxvf $W/hc_exclude_ids.txt | wc -l)" = 0 ] || die "DH's drop list ($nd ids) is not inside the exclusion list"
$V/bin/python $SC/kl_pick24.py $W/fs5hh $W/fs5k.rowids.txt $W/fs5k.meta.npz $W/fs5h_rows.meta.jsonl $W/mxa.jsonl,$W/fs5h_rows.jsonl $W/hc_exclude_ids.txt 1 $W/$NAME.tmp > $K/logs/hc_pick.out 2>&1 || die "PICK24: $(tail -1 $K/logs/hc_pick.out | cut -c1-400)"
log "$(grep -E '^KLPICK24' $K/logs/hc_pick.out | cut -c1-1800)"
[ "$($V/bin/python -c "import json; d = json.load(open('$W/$NAME.tmp.info.json')); p = d['pools']; c = d['checks']; print(p['hellaswag'] + p['cfcolor'] + c['ids_matching_hellaswag_or_cfcolor'] + c['manifest_sources_matching_hellaswag_or_cfcolor'] + c['excluded_ids_present'])")" = 0 ] || die "hellaswag / cfcolor / excluded rows in the built set"
mkdir -p $W/hc_sum && mv $W/$NAME.tmp.info.json $W/hc_sum/$NAME.info.json || die MV_INFO
for f in ids.npy meta.npz rowids.txt; do mv $W/$NAME.tmp.$f $W/$NAME.$f || die "MV $f"; done
{ (cd $W && sha256sum $NAME.ids.npy $NAME.meta.npz $NAME.rowids.txt); (cd $W/hc_sum && sha256sum $NAME.info.json); } > $W/$NAME.SHA256SUMS
for f in ids.npy meta.npz rowids.txt; do aws s3 cp --only-show-errors $W/$NAME.$f $O/$NAME.$f || die "UPLOAD $NAME.$f"; done
aws s3 cp --only-show-errors $W/hc_sum/$NAME.info.json $O/$NAME.info.json || die "UPLOAD $NAME.info.json"
for f in ids.npy meta.npz rowids.txt info.json; do [ "$(aws s3 cp --only-show-errors $O/$NAME.$f - | sha256sum | cut -c1-64)" = "$(grep " $NAME.$f\$" $W/$NAME.SHA256SUMS | cut -c1-64)" ] || die "READBACK $NAME.$f"; done
aws s3 cp --only-show-errors $K/logs/hc_pick.out $O/${NAME}_build.hc_pick.out 2>/dev/null; aws s3 cp --only-show-errors $W/hc_exclude_ids.txt $O/${NAME}_build.hc_exclude_ids.txt 2>/dev/null
aws s3 cp --only-show-errors $W/$NAME.SHA256SUMS $O/$NAME.SHA256SUMS || die "UPLOAD $NAME.SHA256SUMS"      # last on S3: its presence there means "built"
mv $W/hc_sum/$NAME.info.json $W/$NAME.info.json || die MV_INFO_FINAL      # last on disk: its presence here means "present"
log "KL_HBUILD24_DONE $NAME uploaded to $O/ (sha256 read-back ok)"
[ -s $W/$NAME.info.json ] || die "no $NAME after the build"
