#!/bin/bash
# kl_keeper.sh (KL agent): keeps kl_chain.sh going on this node. Every 120 s: chain not running and no KL_CHAIN_DONE -> start it (at most 8 starts). Exits on KL_CHAIN_DONE.
unset ASCALE_RUN_ID; export AWS_DEFAULT_REGION=us-east-2
D=/data/local; K=$D/kl; R=s3://YOUR_BUCKET/jev-race/engineering/esbar/runs; mkdir -p $K/logs $K/scripts; KL=$K/logs/kl_keeper.log; starts=0
log() { echo "[kl_keeper $(date -u +%FT%TZ)] $*" | tee -a $KL; aws s3 cp --only-show-errors $KL $R/esbar_kl_code/logs/kl_keeper_$(hostname).log 2>/dev/null; }
exec 6> $K/keeper.lock; flock -n 6 || { echo "another kl_keeper runs"; exit 3; }
log "start host $(hostname) order ${KL_ORDER:-k1 k2}"
while true; do
  grep -q "KL_CHAIN_DONE ${KL_ORDER:-k1 k2}\$" $K/logs/kl_chain.log 2>/dev/null && { log "EXIT KL_CHAIN_DONE ${KL_ORDER:-k1 k2}"; exit 0; }      # this order's own end line (an earlier chain with another order on this node does not count)
  if [ "$(pgrep -fc '[k]l_chain.sh')" = 0 ]; then
    [ $starts -ge 8 ] && { log "EXIT gave up after $starts chain starts: $(tail -1 $K/logs/kl_chain.log 2>/dev/null | cut -c1-200)"; exit 1; }
    sleep 20; [ "$(pgrep -fc '[k]l_chain.sh')" = 0 ] || continue
    aws s3 cp --only-show-errors $R/esbar_kl_code/scripts/kl_chain.sh $K/scripts/kl_chain.sh || { log "chain fetch failed"; sleep 120; continue; }
    starts=$((starts + 1)); log "chain not running -> start $starts (last: $(tail -1 $K/logs/kl_chain.log 2>/dev/null | cut -c1-160))"
    bash $K/scripts/kl_chain.sh >> $K/logs/kl_chain.out 2>&1 &
  fi
  sleep 120
done
