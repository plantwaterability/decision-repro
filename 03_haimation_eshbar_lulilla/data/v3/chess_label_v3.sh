#!/bin/bash
# es-bar v3 chess pool: 80,000 fresh Stockfish-16-labelled positions (50 ms per action, mover win prob, soft targets tau 0.02) from
# searchless_chess TRAIN shard 00001 + a fresh 2,000-position held-out (DI scoring fields); every test position (FEN or placement + side)
# and every placement already in the v1/v2 pool (30,000) or held-out (1,000) excluded. p8 CPU (180 workers). Outputs -> S3 pools/.
unset ASCALE_RUN_ID; export OMP_NUM_THREADS=1
D=/data/local; L=$D/chess_v3; mkdir -p $L/pylib $L/sfx $L/sfbin; cd $L || exit 1
S3=s3://YOUR_BUCKET/jev-race/engineering/esbar
log() { echo "[chess_v3 $(date -u +%FT%TZ)] $*" | tee -a $L/run.log; }
PY=$D/venv_t2141/bin/python
[ -d pylib/chess ] || $PY -m pip install -q --target $L/pylib chess==1.11.2 > pip.log 2>&1 || { log PIP; exit 1; }
if [ ! -x sfbin/stockfish ]; then
  [ -s sf.tar ] || curl -sfL -o sf.tar https://github.com/official-stockfish/Stockfish/releases/download/sf_16/stockfish-ubuntu-x86-64-avx2.tar
  tar xf sf.tar -C sfx && cp "$(find sfx -type f -name 'stockfish-ubuntu-x86-64-avx2' | head -1)" sfbin/stockfish && chmod +x sfbin/stockfish || { log STOCKFISH; exit 1; }
fi
log "stockfish $(md5sum < sfbin/stockfish | cut -c1-8) $(echo quit | sfbin/stockfish | head -1)"
[ -s train1.bag ] || curl -sfL -o train1.bag https://storage.googleapis.com/searchless_chess/data/train/action_value-00001-of-02148_data.bag || { log TRAIN_BAG; exit 1; }
[ -s test.bag ] || curl -sfL -o test.bag https://storage.googleapis.com/searchless_chess/data/test/action_value_data.bag || { log TEST_BAG; exit 1; }
log "bags: train1 $(md5sum < train1.bag | cut -c1-8) test $(md5sum < test.bag | cut -c1-8)"
aws s3 cp --only-show-errors $S3/pools/chess_train.jsonl.gz pool_v12.jsonl.gz
$PY - <<'PY' > exclude_fens.txt
import gzip, json, ast
for p in ("pool_v12.jsonl.gz",):
    for l in gzip.open(p, "rt"):
        st = json.loads(l)["state"]
        st = ast.literal_eval(st) if isinstance(st, str) else st
        print(st["fen"])
for l in open("/data/local/esbar_data/chess/heldout_chess.jsonl"):
    st = json.loads(l)["state"]
    st = ast.literal_eval(st) if isinstance(st, str) else st
    print(st["fen"])
PY
log "exclude fens $(wc -l < exclude_fens.txt)"
t0=$(date +%s)
PYTHONPATH=$L/pylib $PY sf_label_chess_v3.py --kit $D/public_v2/kit --test test.bag --train train1.bag --stockfish $L/sfbin/stockfish \
  --out chess_train_v3_sf16_80k.jsonl --heldout-out heldout_chess_v3.jsonl --n 80000 --heldout-n 2000 --workers 180 --ms 50 --tau 0.02 \
  --seed 20261102 --exclude exclude_fens.txt > label.log 2>&1 || { log "LABEL_FAILED $(tail -2 label.log)"; exit 1; }
log "labelled: $(tail -1 label.log) wall_s=$(( $(date +%s) - t0 ))"
gzip -kf chess_train_v3_sf16_80k.jsonl
for f in chess_train_v3_sf16_80k.jsonl.gz heldout_chess_v3.jsonl sf_label_chess_v3.py label.log exclude_fens.txt; do
  aws s3 cp --only-show-errors $f $S3/pools/chess_v3/$f && [ "$(aws s3 cp --only-show-errors $S3/pools/chess_v3/$f - | md5sum | cut -c1-32)" = "$(md5sum < $f | cut -c1-32)" ] && log "S3 ok $f $(md5sum < $f | cut -c1-8)" || log "S3_FAILED $f"
done
touch DONE
