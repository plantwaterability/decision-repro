#!/bin/bash
W=/data/shared/fs5
python3 $W/scripts/fix_acos_meta.py $W/out/pools/fs5_acos.meta.jsonl >> $W/logs/upload.log 2>&1
head -c 400 $W/out/pools/fs5_acos.meta.jsonl >> $W/logs/upload.log; echo >> $W/logs/upload.log
bash $W/scripts/upload.sh
