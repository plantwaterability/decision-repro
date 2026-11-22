#!/bin/bash
# kl_k24post.sh (TA agent; coordinator order ~22:30Z 11-16): K24 on its own node, after its final screen + prefix reads and BEFORE its DONE marker (kl_run.sh POSTSTEP; the
# script is fetched fresh from S3 each time, so it may change while K24 trains). Idempotent; a relaunch resumes.
#  (1) s3 .../esbar_kl_k24/k24_vs_soup.txt = kl_cmp.py as for K23: K24's prefix per benchmark against the soup's reads on kls (73.744) and kl3 (73.646);
#  (2) K24's FULL PUBLIC run with the serving options named by the pointer s3 .../esbar_kl_code/guard/k24_full.txt (written by the TA agent after the soup measurements):
#        tag <TAG>   kl_k24_full (plain) | kl_k24_full_ml<N> | kl_k24_full_rev | kl_k24_full_ml<N>_rev   (kl_kfullx.sh reads view and max_length off the tag)
#        run no      no full run here (the TA agent runs it elsewhere)
#      no pointer 20 min after this step first started -> tag kl_k24_full (plain). Results s3 .../esbar_<TAG>/kown_full/{KFULL.txt, final/}; marker esbar_kl_k24/markers/FULL_DONE. Nothing public.
set -u; source /data/local/kl/scripts/kl_env.sh
ARM=${KL_ARM:-k24}; RUN=${KL_RUN:?}; O=$R/esbar_kl_$ARM; F=$K/run/$ARM; L=$F/logs/post.log; mkdir -p $F/cmp $F/logs
log() { echo "[kl_k24post $(date -u +%FT%TZ)] $*" | tee -a $L; aws s3 cp --only-show-errors $L $O/logs/post_$ARM.log 2>/dev/null; }
fail() { log "FAILED $*"; exit 1; }
PT=kl_${ARM}_full; PW=$M/work/$PT/kown
# ---- 1. the comparison with the soup
if ! aws s3 ls $O/${ARM}_vs_soup.txt > /dev/null 2>&1; then
  for x in "$PW/prefix_compare.json $R/esbar_$PT/kown_prefix/prefix_compare.json $F/cmp/a.json" "/nonexistent $R/esbar_kl_soup_k7k8k12k3_j2/kown_prefix/prefix_compare.json $F/cmp/b.json" "/nonexistent $R/esbar_kl_soup_k7k8k12k3/kown_prefix/prefix_compare.json $F/cmp/c.json"; do read -r xl xs xo <<< "$x"
    if [ -s $xl ]; then cp $xl $xo; else aws s3 cp --only-show-errors $xs $xo || fail "no $xs"; fi; done
  $V/bin/python $SC/kl_cmp.py ${ARM^^} $F/cmp/a.json soup_kls $F/cmp/b.json soup_kl3 $F/cmp/c.json $KR/src/prefix_rows.jsonl.gz > $F/cmp/${ARM}_vs_soup.txt 2>&1 || fail "kl_cmp: $(tail -1 $F/cmp/${ARM}_vs_soup.txt | cut -c1-200)"
  aws s3 cp --only-show-errors $F/cmp/${ARM}_vs_soup.txt $O/${ARM}_vs_soup.txt || fail "upload ${ARM}_vs_soup.txt"
  log "CMP $(head -1 $F/cmp/${ARM}_vs_soup.txt | cut -c1-260)"
fi
# ---- 2. the full public run
P=$R/esbar_kl_code/guard/k24_full.txt
aws s3 ls $O/markers/POST_STARTED > /dev/null 2>&1 || date +%s | aws s3 cp --only-show-errors - $O/markers/POST_STARTED
t0=$(aws s3 cp --only-show-errors $O/markers/POST_STARTED - 2>/dev/null | tr -dc 0-9); [ -n "$t0" ] || t0=$(date +%s); w=0
until aws s3 ls $P > /dev/null 2>&1; do [ $(( $(date +%s) - t0 )) -gt 1200 ] && break; [ $w = 0 ] && log "waiting up to 20 min for the pointer guard/k24_full.txt (then: plain)"; w=1; sleep 30; done
ptr=$(aws s3 cp --only-show-errors $P - 2>/dev/null | grep -v '^#'); TAG=$(echo "$ptr" | awk '$1 == "tag" {print $2; exit}'); RUNIT=$(echo "$ptr" | awk '$1 == "run" {print $2; exit}')
[ -n "$TAG" ] || TAG=$PT
case $TAG in kl_${ARM}_full|kl_${ARM}_full_*) ;; *) fail "pointer tag [$TAG] is not kl_${ARM}_full[_...]";; esac
if [ "$RUNIT" = no ]; then log "POST_DONE comparison written; no full public run here (pointer: run no)"; exit 0; fi
if aws s3 ls $R/esbar_$TAG/kown_full/KFULL.txt > /dev/null 2>&1; then
  aws s3 ls $O/markers/FULL_DONE > /dev/null 2>&1 || echo "$(date -u +%FT%TZ) $(hostname) $TAG $(aws s3 cp --only-show-errors $R/esbar_$TAG/kown_full/KFULL.txt - | head -1 | cut -c1-300)" | aws s3 cp --only-show-errors - $O/markers/FULL_DONE
  log "POST_DONE full public run $TAG exists already: $(aws s3 cp --only-show-errors $R/esbar_$TAG/kown_full/KFULL.txt - | head -1 | cut -c1-220)"; exit 0; fi
ident() { (cd $1 && ls model*.safetensors config.json | sort | xargs -P 13 -n 1 sha256sum | sort -k2) | sha256sum | cut -c1-64; }
PD=$RUN/export_full; WANT=$(aws s3 cp --only-show-errors $O/export_full.identity.txt - | cut -c1-64); [ ${#WANT} = 64 ] || fail "no export_full.identity.txt"
[ -s $PD/model.safetensors.index.json ] || { mkdir -p $PD; aws s3 sync --only-show-errors --exclude READBACK_MD5.txt $O/export_full/ $PD/ || fail "export fetch"; }
[ "$(cat $PD/.identity 2>/dev/null)" = "$WANT" ] || { [ "$(ident $PD)" = "$WANT" ] || fail "identity of $PD"; echo $WANT > $PD/.identity; }
aws s3 cp --only-show-errors $R/esbar_kl_code/scripts/kl_kfullx.sh $SC/kl_kfullx.sh.new && mv $SC/kl_kfullx.sh.new $SC/kl_kfullx.sh || fail "no kl_kfullx.sh on S3"
log "FULL PUBLIC run of $ARM (final prefix ${KL_PX:-?}) on $(hostname): tag $TAG (pointer: $(echo "$ptr" | tr '\n' ';' | cut -c1-200)); kl_kfullx.sh $(md5sum < $SC/kl_kfullx.sh | cut -c1-8); nothing public"
KFULL_PREFIX_RESULTS=$PW/prefix_results.jsonl bash $SC/kl_kfullx.sh $TAG $PD > $F/logs/full_$TAG.out 2>&1; rc=$?
log "kl_kfullx.sh rc=$rc: $(grep -E 'KFULL_DONE|DIE' $M/work/$TAG/kfull/driver.log 2>/dev/null | tail -1 | cut -c1-300)"
[ $rc = 0 ] || exit 1
echo "$(date -u +%FT%TZ) $(hostname) $TAG $(grep -E '^HEADLINE' $M/work/$TAG/kfull/merge/report.txt | cut -c1-300)" | aws s3 cp --only-show-errors - $O/markers/FULL_DONE
log "POST_DONE $TAG $(grep -E '^HEADLINE' $M/work/$TAG/kfull/merge/report.txt | cut -c1-240)"
