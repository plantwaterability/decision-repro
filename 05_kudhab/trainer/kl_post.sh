#!/bin/bash
# kl_post.sh post1 (KL agent; coordinator orders 22:05Z 11-15, items 1 and 7): soups of the pure-distillation finals in karpo's engine, on this node after K8.
#  A. 0.5 K7 + 0.5 K8 (4,400 screen); then the full prefix read of whichever of K7 / K8 / that soup screens best.   B. (K7 + K8 + K12) / 3.
#  C. K10 + K7 only if K10's mid screen is >= 74.3 (01:45Z 11-16).   Every member enters with the better of its mid / final export. A soup >= its best member + 0.2 gets a prefix read.
# All members are Apache-clean (init mx_mj50; no karpo weights). mx_merge.py over E1's layout as the text agent uses it. Idempotent (S3 markers / chunks).
set -u; source /data/local/kl/scripts/kl_env.sh
P=${1:?}; W=$K/post/$P; O=$R/esbar_kl_$P; mkdir -p $W; L=$W/post.log
log() { echo "[kl_post $P $(date -u +%FT%TZ)] $*" | tee -a $L; aws s3 cp --only-show-errors $L $O/post.log 2>/dev/null; }
die() { log "KL_POST_FAILED $*"; exit 1; }
mark() { echo "$(date -u +%FT%TZ) $(hostname) ${2:-}" | aws s3 cp --only-show-errors - $O/markers/$1; }
has() { aws s3 ls $O/markers/$1 > /dev/null 2>&1; }
exec 8> $K/node.lock; flock -n 8 || { echo "NODE_LOCK held"; exit 3; }
echo "whole node: KL agent $P (soup screens / prefix reads), since $(date -u +%FT%TZ)" > /tmp/claim.txt
ident() { (cd $1 && ls model*.safetensors config.json | sort | xargs -P 13 -n 1 sha256sum | sort -k2) | sha256sum | cut -c1-64; }
hhmm() { date -u +%H%M | sed 's/^0*//;s/^$/0/'; }
late() { local t=$(hhmm); [ $t -ge $1 ] && [ $t -lt 815 ]; }      # after $1 (HHMM UTC) and before the 08:05Z pass: do not start this
scr() { aws s3 cp --only-show-errors $R/esbar_kl_${1}_$2/kown_s4400/logs/driver.log - 2>/dev/null | grep -E '^INDEX KOWN' | tail -1 | awk '{print $3}'; }      # screen of arm $1's export $2 (s2811 | full)
fin() { scr $1 full; }
waitfin() { until [ -n "$(fin $1)" ] && aws s3 ls $R/esbar_kl_$1/export_full.identity.txt > /dev/null 2>&1; do late $2 && return 1; sleep 60; done; }
pick() {   # coordinator 01:45Z 11-16: each member enters with the better of its mid / final export (K7's mid reads above its final) -> PL (s2811 | full), PV
  local m=$(scr $1 s2811) f=$(scr $1 full); PL=full; PV=$f
  [ -z "$f" ] && { PL=s2811; PV=$m; }; [ -n "$m" ] && [ -n "$f" ] && python3 -c "import sys; sys.exit(0 if float('$m') > float('$f') else 1)" && { PL=s2811; PV=$m; }; [ -n "$PV" ]; }
getexp() { local a=$1 l=$2 x=$K/src/${1}_$2 want=$(aws s3 cp --only-show-errors $R/esbar_kl_$1/export_$2.identity.txt - | cut -c1-64)
  [ ${#want} = 64 ] || return 1; [ -f $x/.ok ] && [ "$(cat $x/.identity)" = "$want" ] && return 0
  mkdir -p $x && aws s3 sync --only-show-errors --exclude READBACK_MD5.txt $R/esbar_kl_$a/export_$l/ $x/ && [ "$(ident $x)" = "$want" ] && echo $want > $x/.identity && touch $x/.ok; }
idx() { grep -E '^INDEX KOWN' $M/work/$1/kown_s4400/score.txt 2>/dev/null | tail -1 | awk '{print $3}'; }
pidx() { grep -E '^INDEX KOWN' $M/work/$1/kown/score.txt 2>/dev/null | tail -1 | awk '{print $3}'; }
E1D=$D/soupB_in/e1_export
[ -f $E1D/.ok ] || { mkdir -p $E1D && aws s3 sync --only-show-errors $E/runs/esbar_v5/export/ $E1D/ && [ "$(ident $E1D)" = c92e70ea0dcbfb4d62e04164e7fe48cff4b292b6b47175ae94b4f5987f63eeb3 ] && touch $E1D/.ok; } || die E1_LAYOUT
soup() {   # $1 name, $2.. member arms (equal weights, each with its better export) -> screen; BESTM / BESTL / BESTV = best member; SV = the soup's screen
  local name=$1; shift; local n=$# a arg="" i=0 w G=$W/$name t=kl_soup_$name desc=""
  for a in $K/post/*/$name; do [ -f $a/.ok ] && G=$a; done; SG=$G      # a soup built by an earlier post job on this disk is reused
  BESTM=""; BESTV=0; for a in "$@"; do pick $a || die "no screen of $a"; desc="$desc$a:$PL=$PV "; python3 -c "import sys; sys.exit(0 if float('$PV') > float('$BESTV') else 1)" && { BESTM=$a; BESTL=$PL; BESTV=$PV; }; done
  if [ -z "$(idx $t)" ]; then
    late 745 && { log "$name: too late for a screen"; return 1; }
    if [ ! -f $G/.ok ]; then
      for a in "$@"; do pick $a; getexp $a $PL || die "EXPORT $a $PL"; i=$((i + 1)); w=$(python3 -c "n=$n; i=$i; print(round(1.0 - (n - 1) * round(1.0 / n, 7), 7) if i == n else round(1.0 / n, 7))"); arg="$arg${arg:+,}$a=$K/src/${a}_$PL:$w"; done
      [ -e $G ] && mv $G $G.partial_$(date -u +%d%H%M%S)
      $V/bin/python $M/scripts/mx_merge.py $E1D $G $W/merge_$name.json "$arg" > $W/merge_$name.log 2>&1 || die "MERGE $name: $(grep -E 'STRUCT|Error|assert' $W/merge_$name.log | tail -1 | cut -c1-200)"
      [ "$(ls $G/model-*.safetensors | wc -l)" = 12 ] && [ "$(grep -c SPOT_OK $W/merge_$name.log)" = 3 ] || die "MERGE_INCOMPLETE $name"
      rm -f $G/vbll_head.json $G/vbll_head.safetensors; ident $G > $G/.identity; touch $G/.ok; echo "$desc" > $G/members.txt; aws s3 cp --only-show-errors $W/merge_$name.log $O/merge_$name.log
    fi
    KSET=s4400 bash $M/scripts/mx_kown.sh $t $G > $W/kown_$t.out 2>&1
  fi
  SV=$(idx $t); [ -n "$SV" ] || die "SCREEN $name"
  log "SOUP $name screen $SV | members (export=screen): ${desc}| best member $BESTM:$BESTL $BESTV | $(python3 -c "print('%+.3f' % ($SV - $BESTV))")"
  echo "soup $name $SV members ${desc}best_member $BESTM:$BESTL $BESTV" >> $W/results.txt; aws s3 cp --only-show-errors $W/results.txt $O/results.txt
}
prefix() {   # $1 tag, $2 dir, $3 label
  [ -n "$(pidx $1)" ] && return 0; late 720 && { log "$3: too late for a prefix read"; return 1; }
  KSET=prefix bash $M/scripts/mx_kown.sh $1 $2 > $W/kownp_$1.out 2>&1; local p=$(pidx $1)
  log "PREFIX $3: ${p:-FAILED} (karpo 75.301; K7 see its own read)"; echo "prefix $3 ${p:-FAILED}" >> $W/results.txt; aws s3 cp --only-show-errors $W/results.txt $O/results.txt
}
: >> $W/results.txt
if [ $P = post1 ]; then
# ---- A: K7 + K8 (each with its better export), then the prefix read of the best of K7 / K8 / soup
if ! has A_DONE; then
  waitfin k8 700 || die "A: no K8 final in time"
  soup k7k8 k7 k8 || die A_SOUP
  if python3 -c "import sys; sys.exit(0 if float('$SV') > float('$BESTV') else 1)"; then log "A: the soup screens best ($SV): its prefix read"; prefix kl_soup_k7k8 $W/k7k8 "soup k7k8"
  elif [ $BESTM = k8 ]; then log "A: K8 ($BESTL) screens best ($BESTV): its prefix read"; getexp k8 $BESTL && prefix kl_k8_$BESTL $K/src/k8_$BESTL "k8 $BESTL" && [ $BESTL = full ] && echo "$(date -u +%FT%TZ) $(hostname) THIS=$(pidx kl_k8_full) (read by kl_post.sh)" | aws s3 cp --only-show-errors - $R/esbar_kl_k8/markers/READ_prefix_full
  else log "A: K7 ($BESTL) screens best ($BESTV): no prefix read here (its mid export has the full public run, 73.38; its final's prefix is read on kl2)"; fi
  mark A_DONE "$(tail -2 $W/results.txt | tr '\n' ';')"
fi
# ---- B: K7 + K8 + K12
if ! has B_DONE; then
  if waitfin k12 645; then soup k7k8k12 k7 k8 k12 && { python3 -c "import sys; sys.exit(0 if float('$SV') >= float('$BESTV') + 0.2 else 1)" && prefix kl_soup_k7k8k12 $W/k7k8k12 "soup k7k8k12"; mark B_DONE "$(tail -2 $W/results.txt | tr '\n' ';')"; }
  else log "B: no K12 final in time: skipped"; mark B_DONE "skipped: no K12 final by 06:45Z"; fi
fi
# ---- C (coordinator 01:45Z 11-16): K10 (lr 5e-7) + K7 only if K10's MID screen is >= 74.3 (K9, lr 2.5e-6, is stopped)
if ! has C_DONE; then
  until [ -n "$(scr k10 s2811)" ]; do late 650 && break; sleep 60; done; m10=$(scr k10 s2811)
  if [ -n "$m10" ] && python3 -c "import sys; sys.exit(0 if float('$m10') >= 74.3 else 1)"; then
    waitfin k10 650; log "C: K10 mid screen $m10 >= 74.3: soup K10 + K7"; soup k10k7 k10 k7 && { python3 -c "import sys; sys.exit(0 if float('$SV') >= float('$BESTV') + 0.2 else 1)" && prefix kl_soup_k10k7 $W/k10k7 "soup k10k7"; mark C_DONE "$(tail -2 $W/results.txt | tr '\n' ';')"; }
  else log "C: K10 mid screen ${m10:-none} is not >= 74.3: the lr soup is dropped"; mark C_DONE "dropped: K10 mid screen ${m10:-none} < 74.3"; fi
fi
has A_DONE && has B_DONE && has C_DONE && { mark DONE "$(tr '\n' ';' < $W/results.txt | cut -c1-600)"; echo "FREE: KL $P finished $(date -u +%FT%TZ); node idle" > /tmp/claim.txt; log "KL_POST_DONE $(tr '\n' ';' < $W/results.txt)"; }
fi
if [ $P = post2 ]; then      # coordinator 04:55Z 11-16: the screen cannot rank these soups: screens AND prefix reads; the best soup by prefix gets the full public run on klf (no 74.3 gate)
  bestsoup() { BN=""; BP=0; local nm p; for nm in "$@"; do p=$(pidx kl_soup_$nm); [ -n "$p" ] && python3 -c "import sys; sys.exit(0 if float('$p') > float('$BP') else 1)" && { BN=$nm; BP=$p; }; done; [ -n "$BN" ]; }
  if ! has S12_DONE; then
    soup k7k8k12 k7 k8 k12 || die S1; G1=$SG; prefix kl_soup_k7k8k12 $G1 "soup k7k8k12 (K7mid + K8 + K12)" || die S1_PREFIX
    soup k7k8k12k3 k7 k8 k12 k3 || die S2; G2=$SG; prefix kl_soup_k7k8k12k3 $G2 "soup k7k8k12k3 (+ K3 final)" || die S2_PREFIX
    mark S12_DONE "k7k8k12 screen $(idx kl_soup_k7k8k12) prefix $(pidx kl_soup_k7k8k12) | k7k8k12k3 screen $(idx kl_soup_k7k8k12k3) prefix $(pidx kl_soup_k7k8k12k3)"
  fi
  if ! has FULL_REQUESTED; then bestsoup k7k8k12 k7k8k12k3 || die "no soup prefix"; BG=""; for a in $K/post/*/$BN; do [ -f $a/.ok ] && BG=$a; done; [ -n "$BG" ] || die "soup dir $BN"
    log "best soup by prefix: $BN $BP -> its weights to S3 and the full public run request (klf)"
    aws s3 sync --only-show-errors --exclude '.*' $BG/ $R/esbar_kl_soup_$BN/export/ || die SOUP_UPLOAD; cut -c1-64 $BG/.identity | aws s3 cp --only-show-errors - $R/esbar_kl_soup_$BN/export.identity.txt
    [ "$(aws s3 ls $R/esbar_kl_soup_$BN/export/ | grep -c 'model-.*safetensors')" = 12 ] || die SOUP_UPLOAD_COUNT
    if aws s3 ls $R/esbar_kl_code/full/REQUEST > /dev/null 2>&1; then log "a full/REQUEST exists already: $(aws s3 cp --only-show-errors $R/esbar_kl_code/full/REQUEST - | cut -c1-120): not replaced"
    else echo "soup_$BN kl_soup_$BN $R/esbar_kl_soup_$BN/export $(cut -c1-64 $BG/.identity) prefix=$BP $(date -u +%FT%TZ) (kl_post.sh post2: best soup by prefix; coordinator order 04:55Z 11-16, no 74.3 gate; members $(cat $BG/members.txt))" | aws s3 cp --only-show-errors - $R/esbar_kl_code/full/REQUEST && log "FULL_RUN_REQUESTED soup $BN prefix $BP"; fi
    mark FULL_REQUESTED "$BN prefix $BP"
  fi
  if ! has S34_DONE; then      # + the two short seeds (and K10's final): after their exports and final screens exist (after the 08:05Z pass at the earliest)
    waitfin k17 2359; waitfin k18 2359; bestsoup k7k8k12 k7k8k12k3; base="k7 k8 k12"; [ "$BN" = k7k8k12k3 ] && base="k7 k8 k12 k3"; nm=$(echo $base | tr -d ' ')k17k18
    soup $nm $base k17 k18 && prefix kl_soup_$nm $SG "soup $nm (+ K17 + K18)"
    if [ -n "$(fin k10)" ] && aws s3 ls $R/esbar_kl_k10/export_full.identity.txt > /dev/null 2>&1; then soup ${nm}k10 $base k17 k18 k10 && prefix kl_soup_${nm}k10 $SG "soup ${nm}k10 (+ K10)"; fi
    mark S34_DONE "$(tail -4 $W/results.txt | tr '\n' ';' | cut -c1-500)"
  fi
  has S12_DONE && has FULL_REQUESTED && has S34_DONE && { mark DONE "$(tr '\n' ';' < $W/results.txt | cut -c1-700)"; echo "FREE: KL $P finished $(date -u +%FT%TZ); node idle" > /tmp/claim.txt; log "KL_POST_DONE $(tr '\n' ';' < $W/results.txt)"; }
fi
if [ $P = post3 ]; then      # coordinator 08:25Z 11-16: prefix reads of the two soups with the short seeds; one that beats soup k7k8k12k3's prefix (73.646) by >= 0.2 is queued for a full public run
  for job in "k7k8k12k3k17k18 k7 k8 k12 k3 k17 k18" "k7k8k12k3k17k18k10 k7 k8 k12 k3 k17 k18 k10"; do set -- $job; nm=$1; shift
    soup $nm "$@" || die "SOUP $nm"; prefix kl_soup_$nm $SG "soup $nm" || die "PREFIX $nm"; p=$(pidx kl_soup_$nm)
    if python3 -c "import sys; sys.exit(0 if float('$p') >= 73.646 + 0.2 else 1)"; then log "$nm prefix $p >= 73.846: its weights to S3 and a full-run request (runs after the first on klf)"
      aws s3 sync --only-show-errors --exclude '.*' $SG/ $R/esbar_kl_soup_$nm/export/ && cut -c1-64 $SG/.identity | aws s3 cp --only-show-errors - $R/esbar_kl_soup_$nm/export.identity.txt || die SOUP_UPLOAD
      line="soup_$nm kl_soup_$nm $R/esbar_kl_soup_$nm/export $(cut -c1-64 $SG/.identity) prefix=$p $(date -u +%FT%TZ) (kl_post.sh post3: beats soup k7k8k12k3 73.646 by >= 0.2)"
      if aws s3 ls $R/esbar_kl_code/full/ | awk '{print $4}' | grep -qx REQUEST; then echo "$line" | aws s3 cp --only-show-errors - $R/esbar_kl_code/full/QUEUED_$nm; log "a REQUEST is active: queued as full/QUEUED_$nm (move it to full/REQUEST when the slot is free)"
      else echo "$line" | aws s3 cp --only-show-errors - $R/esbar_kl_code/full/REQUEST && log "FULL_RUN_REQUESTED soup $nm prefix $p"; fi
    else log "$nm prefix $p: not >= 73.846, no full run"; fi
  done
  mark DONE "$(tr '\n' ';' < $W/results.txt | cut -c1-600)"; echo "$(date -u +%FT%TZ) $(hostname) post3 done: $(tr '\n' ';' < $W/results.txt | cut -c1-300)" | aws s3 cp --only-show-errors - $R/esbar_kl_code/full/DONE_post3_kl3
  echo "FREE: KL $P finished $(date -u +%FT%TZ); node idle" > /tmp/claim.txt; log "KL_POST_DONE $(tr '\n' ';' < $W/results.txt)"
fi
