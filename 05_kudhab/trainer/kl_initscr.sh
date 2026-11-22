#!/bin/bash
# kl_initscr.sh (KL agent; coordinator order 05:15Z 11-16): UNTRAINED merge inits in karpo's engine on the 4,400 screen, on this node, measurement only. Same-node reference
# first (msj33), then the listed merges, each built with the text agent's mx_merge.py over E1's layout from the MX source copies (s3 .../esbar_mx_src/<label>, fetched and
# sha256-checked by mx_get.sh; munku from its export). One line per result appended to s3 .../esbar_kl_ms_init/results.txt as it lands. Nothing is started after 07:42Z and
# whatever runs is stopped at 07:55Z (its own process group only). No merged weights leave the node. Licences of copernicus / knowline / onejev are NOT checked here.
set -u; source /data/local/kl/scripts/kl_env.sh
W=$K/ms_init; O=$R/esbar_kl_ms_init; mkdir -p $W; L=$W/initscr.log
log() { echo "[kl_initscr $(date -u +%FT%TZ)] $*" | tee -a $L; aws s3 cp --only-show-errors $L $O/initscr.log 2>/dev/null; }
exec 8> $K/node.lock; flock -n 8 || { echo "NODE_LOCK held"; exit 3; }
echo "whole node: KL agent untrained merge-init screens (until 07:55Z), since $(date -u +%FT%TZ)" > /tmp/claim.txt
ident() { (cd $1 && ls model*.safetensors config.json | sort | xargs -P 13 -n 1 sha256sum | sort -k2) | sha256sum | cut -c1-64; }
hhmm() { date -u +%H%M | sed 's/^0*//;s/^$/0/'; }
E1D=$D/soupB_in/e1_export
[ -f $E1D/.ok ] || { mkdir -p $E1D && aws s3 sync --only-show-errors $E/runs/esbar_v5/export/ $E1D/ && [ "$(ident $E1D)" = c92e70ea0dcbfb4d62e04164e7fe48cff4b292b6b47175ae94b4f5987f63eeb3 ] && touch $E1D/.ok; } || { log "FAILED E1 layout"; exit 1; }
aws s3 sync --only-show-errors $R/esbar_mx/scripts/ $M/scripts/; aws s3 sync --only-show-errors $R/esbar_mx/manifests/ $M/manifests/
mdir() { case $1 in munku) echo $D/soup75_in/munku;; *) echo $M/src/$1;; esac; }
stage() { local d=$(mdir $1); [ -f $d/.ok ] && return 0
  if [ $1 = munku ]; then mkdir -p $d && aws s3 sync --only-show-errors --exclude 'vbll_head*' --exclude '*.pt' --exclude READBACK_MD5.txt $R/esbar_wi_fs1f50/export/ $d/ && i=$(ident $d) && [ "$i" = 6a0b6e48b55a950ee1f739af7c938cea5b16ec01ef2b233a033640f9a15f8db9 ] && echo $i > $d/.identity && touch $d/.ok
  else bash $M/scripts/mx_get.sh $1 > $W/get_$1.out 2>&1; [ -f $d/.ok ]; fi; }
idx() { grep -E '^INDEX KOWN' $M/work/$1/kown_s4400/score.txt 2>/dev/null | tail -1 | awk '{print $3}'; }
scr() {   # $1 tag, $2 dir: the screen in its own process group, stopped at 07:55Z
  [ -n "$(idx $1)" ] && return 0; [ $(hhmm) -ge 742 ] && [ $(hhmm) -lt 815 ] && { log "$1: not started (after 07:42Z)"; return 2; }
  KSET=s4400 setsid bash $M/scripts/mx_kown.sh $1 $2 > $W/kown_$1.out 2>&1 & local p=$!
  while kill -0 $p 2>/dev/null; do if [ $(hhmm) -ge 755 ] && [ $(hhmm) -lt 815 ]; then kill -- -$p 2>/dev/null; log "$1: stopped at 07:55Z (process group $p)"; return 2; fi; sleep 10; done
  [ -n "$(idx $1)" ]; }
: >> $W/results.txt
REF=$D/soup75_in/mx_o_msj33
if [ ! -f $REF/.ok ] || [ "$(cat $REF/.identity 2>/dev/null)" != 518ed8debb39f5d78669c111eb7b9f85881473cee7f6590992157da20b4f7e9b ]; then REF=$K/src/msj33
  [ -f $REF/.ok ] || { mkdir -p $REF && aws s3 sync --only-show-errors --exclude 'vbll_head*' --exclude '*.pt' --exclude READBACK_MD5.txt $R/esbar_mx_o_msj33/export/ $REF/ && [ "$(ident $REF)" = 518ed8debb39f5d78669c111eb7b9f85881473cee7f6590992157da20b4f7e9b ] && echo 518ed8debb39f5d78669c111eb7b9f85881473cee7f6590992157da20b4f7e9b > $REF/.identity && touch $REF/.ok; } || { log "FAILED msj33 reference"; exit 1; }; fi
log "start host $(hostname) gpus $(nvidia-smi -L | wc -l); reference msj33 518ed8de at $REF"
scr kl_ms_init_ref_msj33 $REF || { log "FAILED reference screen"; [ $(hhmm) -ge 742 ] && exit 0; exit 1; }
KREF=$(idx kl_ms_init_ref_msj33); log "REFERENCE msj33 (munku:0.3333334,sieve:0.3333333,jeffan:0.3333333) on this node: $KREF (p10: 73.196)"
grep -q "^ref " $W/results.txt || echo "ref msj33 munku:0.3333334,sieve:0.3333333,jeffan:0.3333333 screen $KREF (same node $(hostname), 8x H100; p10 read 73.196) identity 518ed8de" >> $W/results.txt; aws s3 cp --only-show-errors $W/results.txt $O/results.txt
for job in "a munku:0.25,sieve:0.25,jeffan:0.5" "b munku:0.4,sieve:0.2,jeffan:0.4" "c munku:0.25,sieve:0.25,jeffan:0.25,copernicus:0.25" "d munku:0.25,sieve:0.25,jeffan:0.25,knowline:0.25" "e munku:0.2,sieve:0.2,jeffan:0.2,copernicus:0.2,knowline:0.2" "f sieve:0.5,jeffan:0.5" "g munku:0.25,sieve:0.25,jeffan:0.25,onejev:0.25" "h munku:0.5,sieve:0.5"; do set -- $job; x=$1; spec=$2; t=kl_ms_init_$x; G=$W/$x
  grep -q "^$x " $W/results.txt && continue
  [ $(hhmm) -ge 742 ] && [ $(hhmm) -lt 815 ] && { log "$x: not started (after 07:42Z)"; break; }
  if [ ! -f $G/.ok ]; then arg=""; bad=""
    for mw in $(echo $spec | tr ',' ' '); do m=${mw%%:*}; stage $m || { bad="$bad $m"; continue; }; arg="$arg${arg:+,}$m=$(mdir $m):${mw#*:}"; done
    [ -z "$bad" ] || { log "$x $spec: member(s) not staged:$bad: skipped"; echo "$x $spec FAILED members not staged:$bad" >> $W/results.txt; aws s3 cp --only-show-errors $W/results.txt $O/results.txt; continue; }
    [ -e $G ] && mv $G $G.partial_$(date -u +%d%H%M%S)
    if ! $V/bin/python $M/scripts/mx_merge.py $E1D $G $W/merge_$x.json "$arg" > $W/merge_$x.log 2>&1 || [ "$(ls $G/model-*.safetensors 2>/dev/null | wc -l)" != 12 ] || [ "$(grep -c SPOT_OK $W/merge_$x.log)" != 3 ]; then
      r="$(grep -E 'STRUCT|Error|assert' $W/merge_$x.log | tail -1 | cut -c1-200)"; log "$x $spec: merge failed: $r"; echo "$x $spec FAILED merge: $r" >> $W/results.txt; aws s3 cp --only-show-errors $W/results.txt $O/results.txt; aws s3 cp --only-show-errors $W/merge_$x.log $O/merge_$x.log; continue; fi
    rm -f $G/vbll_head.json $G/vbll_head.safetensors; ident $G > $G/.identity; touch $G/.ok; aws s3 cp --only-show-errors $W/merge_$x.log $O/merge_$x.log
  fi
  scr $t $G; rc=$?; [ $rc = 2 ] && break
  I=$(idx $t); if [ -n "$I" ]; then dlt=$(python3 -c "print('%+.3f' % ($I - $KREF))"); log "SCREEN $x $spec identity $(cut -c1-16 $G/.identity): $I ($dlt vs same-node msj33 $KREF)"; echo "$x $spec screen $I delta_vs_msj33 $dlt identity $(cut -c1-16 $G/.identity) members $(for mw in $(echo $spec | tr ',' ' '); do m=${mw%%:*}; echo -n "$m=$(cut -c1-8 $(mdir $m)/.identity) "; done)" >> $W/results.txt
  else log "$x $spec: screen failed: $(grep -E 'DIE' $M/work/$t/kown_s4400/driver.log | tail -1 | cut -c1-200)"; echo "$x $spec FAILED screen" >> $W/results.txt; fi
  aws s3 cp --only-show-errors $W/results.txt $O/results.txt
  for old in $W/[a-h]; do [ "$old" != "$G" ] && [ -f $old/.ok ] && [ -n "$(idx kl_ms_init_$(basename $old))" ] && rm -f $old/model-*.safetensors; done      # a screened merge's weights are not needed again (single files only)
done
echo "FREE: KL merge-init screens finished / stopped $(date -u +%FT%TZ)" > /tmp/claim.txt; log "KL_INITSCR_DONE $(tr '\n' ';' < $W/results.txt | cut -c1-900)"
