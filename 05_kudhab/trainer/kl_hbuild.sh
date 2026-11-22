#!/bin/bash
# kl_hbuild.sh NAME (KL agent; coordinator order 17:45Z 11-16): make $K/data/NAME (fs5hh = K21 | fs5hn = K23 | fs5hnc = K22) present. Fetched from s3 .../esbar_kl_data/ when
# it was built before (any node), else built here and uploaded: one kl_data.py build of FS5 (MX requests + karpo targets) + round H0 (DH requests + karpo targets) + round H
# (DH's pool rows, requests, karpo targets), every input sha256-checked; left out: H0 LOOMD-chess_train / bv1_discovery (exact counts), DH's round-H drop list (4,961 ids);
# kl_pick.py then makes the three sets (NC = the cfcolor pool, meta line NC-SA, exactly 4,841 rows; sata_mrc, line "states no licence", stays in as ordered). Exit 0 = $K/data/NAME.info.json present.
set -u; source /data/local/kl/scripts/kl_env.sh
NAME=${1:?}; W=$K/data; O=$R/esbar_kl_data; T=$R/esbar_mx/tl; P=$R/esbar_dh/pools; L=$K/logs/hbuild.log; mkdir -p $W $K/logs
log() { echo "[kl_hbuild $(date -u +%FT%TZ)] $*" | tee -a $L; aws s3 cp --only-show-errors $L $O/hbuild_$(hostname).log 2>/dev/null; }
die() { log "KL_HBUILD_FAILED $*"; exit 1; }
[ -s $W/$NAME.info.json ] && exit 0
exec 9> $K/hbuild.lock; flock 9; [ -s $W/$NAME.info.json ] && exit 0
if aws s3 ls $O/$NAME.SHA256SUMS > /dev/null 2>&1; then      # built before: fetch into a side dir, check, then move into place (info.json last)
  D2=$W/dl_$NAME; mkdir -p $D2
  for f in ids.npy meta.npz info.json rowids.txt SHA256SUMS; do aws s3 cp --only-show-errors $O/$NAME.$f $D2/$NAME.$f || die "FETCH $NAME.$f"; done
  (cd $D2 && sha256sum -c --quiet $NAME.SHA256SUMS) || die "SHA256 $NAME"
  for f in ids.npy meta.npz rowids.txt SHA256SUMS info.json; do mv $D2/$NAME.$f $W/$NAME.$f; done; log "$NAME fetched from S3: $(tr -d '\n ' < $W/$NAME.info.json | cut -c1-300)"; exit 0
fi
log "building fs5hh / fs5hn / fs5hnc on $(hostname)"
fetch() { local u=$1 f=$2 s=$3; if [ -s $W/$f ] && [ "$(sha256sum < $W/$f | cut -c1-64)" = "$s" ]; then return 0; fi
  aws s3 cp --only-show-errors $u $W/$f.tmp && mv $W/$f.tmp $W/$f || die "FETCH $f"; [ "$(sha256sum < $W/$f | cut -c1-64)" = "$s" ] || die "SHA256 $f"; }
fetch $T/fs5h0.jsonl.gz h0_requests.jsonl.gz 51ca6ea1f34761a41b483dd78d6dfd42b4fdec02f6d0929ad3cebd4cf4119dda
fetch $T/teacher_karpo_fs5h0.jsonl.gz h0_teacher.jsonl.gz 12bb07c8a7bb26ee96303390a7d3864871e475327decbd17cca7ce860d4d6607
fetch $P/fs5h_rows.jsonl fs5h_rows.jsonl 5bcabd65de07e29a4eaa5c9a136886cb960c16467cd0faf17301e576f5aedbae
fetch $P/fs5h_rows.meta.jsonl fs5h_rows.meta.jsonl de0e2a72ce7de53348db6e99ecff3547633603ecd78d204620d532c68b4d87e9
fetch $T/fs5h.jsonl.gz h_requests.jsonl.gz afc36766fc0fdb733c64a69ad9c562c8a52e9dd96244436eebf9e8589996cd1a
fetch $T/teacher_karpo_fs5h.jsonl.gz h_teacher.jsonl.gz ae83846ab84f8dad3b8d9d1c8f1394cbfd42809cb5fddd5fd28a439b61458138
fetch $R/esbar_dh/droplist_roundH.txt droplist_roundH.txt 6e10a9d5fdb34ccefc128fe8d1128747d3a8bf8da2a801a59cf5cc41d2516ab7
fetch $T/mxa.jsonl mxa.jsonl e96f3489fed1f9f4517720c1270aad7bd70e5ba0e7a1e6a43e3f7a60e5507b52
for f in fs5.jsonl.gz teacher_karpo_fs5.jsonl.gz; do [ -s $W/$f ] || aws s3 cp --only-show-errors $T/$f $W/$f || die "FETCH $f"; done
for f in fs5k.rowids.txt fs5k.meta.npz; do [ -s $W/$f ] || aws s3 cp --only-show-errors $O/$f $W/$f || die "FETCH $f"; done
$V/bin/python $SC/kl_h0sel.py $W/mxa.jsonl $W/h0_requests.jsonl.gz $W/hh_h0excl.txt $W/hh_h0sel.json "LOOMD-chess_train:26586,bv1_discovery:400" > $K/logs/hh_h0sel.out 2>&1 || die "H0SEL: $(tail -1 $K/logs/hh_h0sel.out | cut -c1-300)"
grep -v '^#' $W/droplist_roundH.txt | grep . > $W/hh_droplist_ids.txt; nd=$(wc -l < $W/hh_droplist_ids.txt); [ "$nd" = 4961 ] || die "DROPLIST $nd ids, not 4,961"
cat $W/hh_h0excl.txt $W/hh_droplist_ids.txt > $W/hh_exclude_ids.txt
$V/bin/python - $W/fs5h_rows.meta.jsonl $W/hh_nc_ids.txt > $K/logs/hh_nc.out 2>&1 <<'PY' || die "NC ids: $(tail -1 $K/logs/hh_nc.out | cut -c1-300)"
import collections, json, sys
c = collections.Counter(); ids = []; allc = collections.Counter()
for l in open(sys.argv[1]):
    r = json.loads(l); allc[(r.get("pool"), r.get("line"))] += 1
    if r.get("pool") == "cfcolor": ids.append(r["exposure_id"]); c[(r.get("pool"), r.get("line"))] += 1      # NC = the cfcolor pool (coordinator 17:45Z); sata_mrc's line is "states no licence" (as FS1's train_sata_multirc) and stays in, as ordered
assert len(ids) == 4841 and set(c) == {("cfcolor", "NC-SA")}, (len(ids), c)
open(sys.argv[2], "w").write("\n".join(ids) + "\n"); print("NC", {f"{k[0]}|{k[1]}": v for k, v in c.items()}, "ALL", {f"{k[0]}|{k[1]}": v for k, v in allc.items()})
PY
log "exclusions: H0 $(wc -l < $W/hh_h0excl.txt) (chess + bv1_discovery), drop list $nd; NC ids: $(tail -1 $K/logs/hh_nc.out | cut -c1-400)"
tail -c 1 $W/mxa.jsonl | od -c | grep -q '\\n' || die "mxa.jsonl does not end with a newline"
cat $W/mxa.jsonl $W/fs5h_rows.jsonl > $W/hh_manifest.jsonl
$V/bin/python $SC/kl_data.py $W/hh_manifest.jsonl $W/fs5hhall $MUNKU $C/trainers --requests $W/fs5.jsonl.gz,$W/h0_requests.jsonl.gz,$W/h_requests.jsonl.gz \
  --teacher $W/teacher_karpo_fs5.jsonl.gz,$W/h0_teacher.jsonl.gz,$W/h_teacher.jsonl.gz --exclude-ids $W/hh_exclude_ids.txt --workers 96 > $K/logs/hh_kldata.out 2>&1 || die "KLDATA: $(tail -1 $K/logs/hh_kldata.out | cut -c1-240)"
log "kl_data: $(grep -E '^KLDATA parsed' $K/logs/hh_kldata.out | cut -c1-500) | $(grep -E '^KLDATA_DONE' $K/logs/hh_kldata.out | cut -c1-600)"
$V/bin/python $SC/kl_pick.py $W/fs5hhall $W/fs5k.rowids.txt $W/fs5k.meta.npz $W/hh_nc_ids.txt 1 $W > $K/logs/hh_pick.out 2>&1 || die "PICK: $(tail -1 $K/logs/hh_pick.out | cut -c1-240)"
log "$(tail -1 $K/logs/hh_pick.out | cut -c1-1500)"
for s in fs5hh fs5hn fs5hnc; do (cd $W && sha256sum $s.ids.npy $s.meta.npz $s.rowids.txt $s.info.json > $s.SHA256SUMS)
  for f in ids.npy meta.npz rowids.txt info.json SHA256SUMS; do aws s3 cp --only-show-errors $W/$s.$f $O/$s.$f || die "UPLOAD $s.$f"; done; done
for f in hh_kldata.out hh_pick.out hh_nc.out hh_h0sel.out; do aws s3 cp --only-show-errors $K/logs/$f $O/fs5hh_build.$f 2>/dev/null; done
for f in fs5hhall.info.json hh_exclude_ids.txt hh_nc_ids.txt hh_h0sel.json; do aws s3 cp --only-show-errors $W/$f $O/fs5hh_build.$f 2>/dev/null; done
log "KL_HBUILD_DONE fs5hh / fs5hn / fs5hnc uploaded to $O/"
[ -s $W/$NAME.info.json ] || die "no $NAME after the build"
