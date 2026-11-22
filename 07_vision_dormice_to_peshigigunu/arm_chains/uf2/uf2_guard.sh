#!/bin/bash
# data fork (round F): the ONE laptop guard for the uf2 arm on p12 (worker-node). Every ~2 min:
#   1. "finished" comes ONLY from the S3 marker recipe_f/uf2/state/uf2_chain_done, read through the CPU control node (ctl2), never from the
#      training node's disk. Marker there -> exit. A fail marker (smoke failed) or 3 failed restages -> stop relaunching, keep watching.
#   2. p12 TERMINATED (spot loss / the 08:05Z pass) -> `ascale start --yes`, but only after a marker read that SUCCEEDED and said "not done".
#   3. an alive 8-GPU worker with no uf2 chain on it -> launch uf2_chain.sh from S3 (it restages, checks the model shards, resumes from S3).
#   4. webterminal keepalive about every 30 min while the run is unfinished.
# Start ONLY with: python3 ~/zsc/jev/ops/daemonize.py ~/zsc/jev/ops/recipe_f/uf2/uf2_guard.out /bin/bash ~/zsc/jev/ops/recipe_f/uf2/uf2_guard.sh
# Restartable from a fresh shell with that one command (it keeps no state of its own: everything it decides on is read from S3 and from
# the workspace). A second copy exits at once. Needs only bash + a logged-in `ascale`; if this file is missing on a new laptop, fetch it:
#   mkdir -p ~/zsc/jev/ops/recipe_f/uf2 && ascale run WORKSPACE_ID -- "aws s3 cp --quiet s3://YOUR_BUCKET/jev-race/engineering/vision/recipe_f/uf2/code/uf2_guard.sh -" > ~/zsc/jev/ops/recipe_f/uf2/uf2_guard.sh
unset ASCALE_RUN_ID; cd "$(dirname "$0")" || exit 1; L=uf2_guard.log
if [ -f uf2_guard.pid ] && kill -0 "$(cat uf2_guard.pid)" 2>/dev/null && ps -o command= -p "$(cat uf2_guard.pid)" | grep -q "uf2_guard.sh"; then
  echo "[uf2-guard $(date -u +%FT%TZ)] a guard is already running (pid $(cat uf2_guard.pid)): this copy exits" >> $L; exit 0; fi
echo $$ > uf2_guard.pid
WS=WORKSPACE_ID; CTL=WORKSPACE_ID; TAG=p12; NAME=uf2
U=s3://YOUR_BUCKET/jev-race/engineering/vision/recipe_f/uf2
log(){ echo "[uf2-guard $(date -u +%FT%TZ)] $*" >> $L; }
st(){ ascale status $1 2>/dev/null | awk '$1=="state"{print $2}'; }
worker(){ ascale now $1 2>/dev/null | awk 'NR==1{for(i=1;i<=NF;i++){if($i=="HEAD")h=i; if($i=="STATE")t=i}} $1=="node" && $t=="ALIVE" && $h=="false" {print $3; exit}'; }
marker(){ # -> "MK done=<0|1> fail=<0|1> rf=<n>" from S3 via ctl2, or empty if the read itself failed
  ascale run $CTL -- "export AWS_DEFAULT_REGION=us-east-2; S=\$(aws s3 ls $U/state/ 2>&1); rc=\$?; [ \$rc -le 1 ] && echo \"\$S\" | grep -q -E 'AccessDenied|Could not|error' && rc=9; [ \$rc -le 1 ] && echo \"MK done=\$(echo \"\$S\" | grep -c ' ${NAME}_chain_done\$') fail=\$(echo \"\$S\" | grep -c ' ${NAME}_fail\$') rf=\$(aws s3 cp --quiet $U/state/${NAME}_restage_fails - 2>/dev/null | tr -dc 0-9 | cut -c1-3)\"" 2>/dev/null | grep "^MK " | tail -1
}
log "armed pid=$$"; n=0; held=0
while :; do
  n=$((n+1))
  M=$(marker)
  if [ -z "$M" ]; then [ $((n % 5)) = 1 ] && log "marker read failed (ctl2 unreachable?): doing nothing this round"; sleep 110; continue; fi
  case "$M" in *"done=1"*) log "S3 marker ${NAME}_chain_done exists ($M): exit"; exit 0;; esac
  rf=$(echo "$M" | sed -n 's/.* rf=\([0-9]*\).*/\1/p'); rf=${rf:-0}
  if echo "$M" | grep -q "fail=1" || [ "$rf" -ge 3 ]; then
    [ $held = 0 ] && log "HOLD: $M (smoke fail marker or 3 failed restages): no relaunch, no workspace start until the marker is removed"; held=1
    sleep 110; continue
  fi
  held=0
  s=$(st $WS); w=$(worker $WS)
  if [ "$s" = "TERMINATED" ]; then log "$TAG state TERMINATED, run unfinished ($M) -> start: $(ascale start $WS --yes 2>&1 | tail -1 | cut -c1-120)"; sleep 110; continue; fi
  if [ -z "$w" ]; then [ $((n % 10)) = 1 ] && log "$TAG state $s, no alive worker yet"; sleep 110; continue; fi
  [ $((n % 15)) = 2 ] && ascale run $WS --backend webterminal -- 'echo KEEPALIVE' > /dev/null 2>&1
  O=$(ascale run $WS --node $w --backend ws -- "LG=/data/local/uf2/logs/uf2_chain_$NAME.log; echo \"ST chain=\$(pgrep -fc '[u]f2_chain.sh') gpus=\$(nvidia-smi -L 2>/dev/null | wc -l) :: \$(grep '\\\"event\\\": \\\"step\\\"' /data/local/qv/runs/$NAME/train_log.jsonl 2>/dev/null | tail -1 | grep -o '\\\"step\\\": [0-9]*') :: \$(tail -n 1 \$LG 2>/dev/null | cut -c1-200)\"" 2>/dev/null | grep "^ST " | tail -1)
  if [ -z "$O" ]; then sleep 110; continue; fi
  [ $((n % 5)) = 1 ] && log "$TAG $w $O"
  if echo "$O" | grep -q "gpus=8" && echo "$O" | grep -q "chain=0 "; then
    log "$TAG launch chain on $w: $(ascale run $WS --node $w --backend ws --detach --name uf2_${NAME}_$(date -u +%d%H%M) -- "export AWS_DEFAULT_REGION=us-east-2; mkdir -p /data/local/uf2/code; aws s3 cp --quiet $U/code/uf2_chain.sh /data/local/uf2/code/uf2_chain.sh && exec bash /data/local/uf2/code/uf2_chain.sh $TAG $NAME" 2>&1 | tail -3 | tr '\n' ' ' | cut -c1-200)"
  fi
  sleep 110
done
