#!/bin/bash
# kl_mscr.sh SRC_S3_EXPORT SRC_LABEL (KL agent; coordinator order 19:05Z 11-15): merge screens in karpo's engine of karpo with one of our karpo-layout
# exports (e.g. K1's mid export). KARPO DERIVATIVES, measurement only (nothing uploaded but the reads and the merge logs). On this node, in order:
# karpo as it ships (the same-node reference), then  a karpo + 0.3 (X - stock) | b karpo + 0.5 (X - stock) | c 0.5 karpo + 0.5 X | d 0.7 karpo + 0.3 X,
# each on the 4,400 screen; any merge >= the same-node karpo read + 0.4 gets its full 31,198-row prefix read at once. Resumable (S3 chunks / local .ok).
set -u; source /data/local/kl/scripts/kl_env.sh
SRC=${1:?}; LAB=${2:?}; W=$K/ms/$LAB; O=$R/esbar_kl_ms_$LAB; mkdir -p $W; L=$W/mscr.log
log() { echo "[kl_mscr $LAB $(date -u +%FT%TZ)] $*" | tee -a $L; aws s3 cp --only-show-errors $L $O/mscr.log 2>/dev/null; }
die() { log "KL_MSCR_FAILED $*"; exit 1; }
exec 8> $K/node.lock; flock -n 8 || { echo "NODE_LOCK held"; exit 3; }
echo "whole node: KL agent merge screens ($LAB) in karpo's engine, since $(date -u +%FT%TZ)" > /tmp/claim.txt
ident() { (cd $1 && ls model*.safetensors config.json | sort | xargs -P 13 -n 1 sha256sum | sort -k2) | sha256sum | cut -c1-64; }
[ -f $KARPO/.ok ] || bash $M/scripts/mx_get.sh karpo > $W/get_karpo.out 2>&1; [ "$(cut -c1-16 $KARPO/.identity)" = 70f785365c81a4d8 ] || die KARPO
E1D=$D/soupB_in/e1_export; ST=$D/soup75_in/stock; X=$K/src/$LAB
( [ -f $E1D/.ok ] || { mkdir -p $E1D && aws s3 sync --only-show-errors $E/runs/esbar_v5/export/ $E1D/ && [ "$(ident $E1D)" = c92e70ea0dcbfb4d62e04164e7fe48cff4b292b6b47175ae94b4f5987f63eeb3 ] && touch $E1D/.ok; } ) & p1=$!
( [ -f $ST/.ok ] || { mkdir -p $ST && aws s3 sync --only-show-errors $S3/models/gemma-4-31B-it_842da379/ $ST/ && i=$(ident $ST) && case $i in 4d59c3ca50908a67*) echo $i > $ST/.identity; touch $ST/.ok;; esac; } ) & p2=$!
WANT=$(aws s3 cp --only-show-errors $SRC.identity.txt - | cut -c1-64); [ ${#WANT} = 64 ] || die "no identity file for $SRC"
[ -f $X/.ok ] || { mkdir -p $X && aws s3 sync --only-show-errors --exclude READBACK_MD5.txt $SRC/ $X/ && i=$(ident $X) && [ "$i" = "$WANT" ] && echo $i > $X/.identity && touch $X/.ok; }
wait $p1; wait $p2; [ -f $E1D/.ok ] && [ -f $ST/.ok ] && [ -f $X/.ok ] || die "STAGE e1 $([ -f $E1D/.ok ] && echo ok) stock $([ -f $ST/.ok ] && echo ok) src $([ -f $X/.ok ] && echo ok)"
log "staged: karpo 70f78536, stock 4d59c3ca, E1 layout c92e70ea, $LAB $(cut -c1-16 $X/.identity) from $SRC"
idx() { grep -E '^INDEX KOWN' $M/work/$1/kown_s4400/score.txt 2>/dev/null | tail -1 | awk '{print $3}'; }
scr() { [ -n "$(idx $1)" ] || KSET=s4400 bash $M/scripts/mx_kown.sh $1 $2 > $W/kown_$1.out 2>&1; [ -n "$(idx $1)" ] || die "SCREEN $1: $(grep -E 'DIE' $M/work/$1/kown_s4400/driver.log | tail -1 | cut -c1-200)"; }
KT=kl_ms_${LAB}_karpo; scr $KT $KARPO; KREF=$(idx $KT); log "REFERENCE karpo as it ships, this node: $KREF"
: > $W/results.txt; echo "karpo (same node, as it ships) $KREF" >> $W/results.txt
for job in "a karpo:1,X:0.3,stock:-0.3" "b karpo:1,X:0.5,stock:-0.5" "c karpo:0.5,X:0.5" "d karpo:0.7,X:0.3"; do set -- $job; t=kl_ms_${LAB}_$1; spec=$2; G=$W/$1
  if [ ! -f $G/.ok ]; then
    arg=$(echo $spec | tr ',' '\n' | while IFS=: read m w; do case $m in karpo) echo "karpo=$KARPO:$w";; X) echo "$LAB=$X:$w";; stock) echo "stock=$ST:$w";; esac; done | paste -sd, -)
    [ -e $G ] && mv $G $G.partial_$(date -u +%d%H%M%S)
    $V/bin/python $M/scripts/mx_merge.py $E1D $G $W/merge_$1.json "$arg" > $W/merge_$1.log 2>&1 || die "MERGE $1: $(grep -E 'STRUCT|Error|assert' $W/merge_$1.log | tail -1 | cut -c1-200)"
    [ "$(ls $G/model-*.safetensors | wc -l)" = 12 ] && [ "$(grep -c SPOT_OK $W/merge_$1.log)" = 3 ] || die "MERGE_INCOMPLETE $1"
    rm -f $G/vbll_head.json $G/vbll_head.safetensors; ident $G > $G/.identity; touch $G/.ok; aws s3 cp --only-show-errors $W/merge_$1.log $O/merge_$1.log
  fi
  scr $t $G; I=$(idx $t); dlt=$(python3 -c "print('%+.3f' % ($I - $KREF))")
  log "SCREEN $1 $(echo $spec | sed "s/X/$LAB/") identity $(cut -c1-16 $G/.identity): $I ($dlt vs same-node karpo $KREF)"; echo "$1 $spec (X = $LAB) $I $dlt" >> $W/results.txt
  if python3 -c "import sys; sys.exit(0 if $I >= $KREF + 0.4 else 1)"; then
    log "$1 is >= karpo + 0.4: full prefix read now"; KSET=prefix bash $M/scripts/mx_kown.sh $t $G > $W/kownp_$t.out 2>&1
    P=$(grep -E '^INDEX KOWN' $M/work/$t/kown/score.txt 2>/dev/null | tail -1 | awk '{print $3}'); log "PREFIX $1: ${P:-FAILED} (karpo 75.301 on the prefix)"; echo "$1 PREFIX ${P:-FAILED}" >> $W/results.txt
  fi
  aws s3 cp --only-show-errors $W/results.txt $O/results.txt
done
echo "$(date -u +%FT%TZ) $(hostname): merge screens $LAB done; node idle, waiting for the coordinator (results s3 .../esbar_kl_ms_$LAB/results.txt)" | aws s3 cp --only-show-errors - $R/esbar_kl_code/guard/$(cat $K/NODE).IDLE
echo "FREE: KL merge screens ($LAB) finished $(date -u +%FT%TZ); node idle" > /tmp/claim.txt
log "KL_MSCR_DONE $(tr '\n' ';' < $W/results.txt)"
