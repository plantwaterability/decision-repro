#!/bin/bash
# kl_smoke.sh (KL agent): before any real training step. (a) memory: 2 real steps on the 128 LONGEST rows; (b) resume: steps 0-5 with an S3 save,
# a fresh process resumes from S3 and runs 5-10 + export; an uninterrupted 0-10 run for comparison (losses of steps 5-9 must agree);
# (c) the export through karpo's own engine on the 4,400 screen (mx_kown.sh: checkpoint.initialize + kit). Init munku, lr 2.5e-6 / warm-up 100 (10 steps move ~nothing).
set -u; source /data/local/kl/scripts/kl_env.sh
W=$K/smoke; O=$R/esbar_kl_smoke; mkdir -p $W; L=$W/smoke.log
log() { echo "[kl_smoke $(date -u +%FT%TZ)] $*" | tee -a $L; aws s3 cp --only-show-errors $L $O/smoke.log 2>/dev/null; }
exec 8> $K/node.lock; flock -n 8 || { log "NODE_LOCK held"; exit 3; }
COMMON="--init-from $MUNKU --manifest $K/data/fs5k --val $K/data/devk --val-max 500 --pinned-code $C/trainers --alpha 0.8 --lr 2.5e-6 --warmup 100 --lr-floor 0.1 --sched-steps 6574 --seed 1 --eval-every 1000000 --spot-watch off ${KL_PACK:+--pack-tokens $KL_PACK}"
steps() { grep '^STEP' $1 | cut -c1-330; }
if [ ! -f $W/a.ok ]; then
  $TR --rdzv-endpoint 127.0.0.1:29802 $SC/kl_train.py --mode smoke --smoke-steps 2 --out $W/a $COMMON > $W/a.log 2>&1; rc=$?
  log "(a) memory rc=$rc: $(grep -E '^DATA|GRAD_CHECK' $W/a.log | cut -c1-300 | tr '\n' ' ') $(steps $W/a.log | tr '\n' ' ')"
  [ $rc = 0 ] || { log "DIE (a): $(grep -E 'Error|error|OOM' $W/a.log | tail -4 | cut -c1-300 | tr '\n' ' ')"; aws s3 cp --only-show-errors $W/a.log $O/a.log; exit 1; }; touch $W/a.ok
fi
[ "${KL_SMOKE_ONLY_A:-0}" = 1 ] && { log "KL_SMOKE_A_DONE"; exit 0; }
if [ ! -f $W/b.ok ]; then
  RS=$O/resume_$(date -u +%Y%m%dT%H%M%S)      # a fresh S3 resume prefix per attempt (nothing is deleted)
  $TR --rdzv-endpoint 127.0.0.1:29802 $SC/kl_train.py --out $W/b1 $COMMON --max-steps 5 --save-every 5 --resume-s3 $RS --no-export > $W/b1.log 2>&1; rc=$?
  log "(b1) 0-5 rc=$rc: $(grep -E 'SAVE_DONE' $W/b1.log | tail -1 | cut -c1-200)"; [ $rc = 0 ] || { log "DIE b1: $(grep -E 'Error|error' $W/b1.log | tail -3 | cut -c1-300 | tr '\n' ' ')"; exit 1; }
  $TR --rdzv-endpoint 127.0.0.1:29802 $SC/kl_train.py --out $W/b2 $COMMON --max-steps 10 --save-every 5 --resume-s3 $RS > $W/b2.log 2>&1; rc=$?
  log "(b2) resume 5-10 rc=$rc: $(grep -E '^RESUMED|^EXPORTED' $W/b2.log | cut -c1-200 | tr '\n' ' ')"; [ $rc = 0 ] && [ -s $W/b2/export/model.safetensors.index.json ] || { log "DIE b2: $(grep -E 'Error|error' $W/b2.log | tail -3 | cut -c1-300 | tr '\n' ' ')"; exit 1; }
  $TR --rdzv-endpoint 127.0.0.1:29802 $SC/kl_train.py --out $W/c $COMMON --max-steps 10 --save-every 0 --no-export > $W/c.log 2>&1; rc=$?
  log "(c) uninterrupted 0-10 rc=$rc"; [ $rc = 0 ] || { log "DIE c"; exit 1; }
  $V/bin/python - $W/b1.log $W/b2.log $W/c.log > $W/resume_cmp.txt <<'PY'
import json, sys
def rows(f): return {r["step"]: r for r in (json.loads(l[5:]) for l in open(f) if l.startswith("STEP "))}
b1, b2, c = (rows(f) for f in sys.argv[1:4])
for s in range(10):
    x = (b1 if s < 5 else b2).get(s); y = c.get(s)
    print(f"step {s} interrupted loss {x['loss']:.9f} gnorm {x['gnorm']} | uninterrupted loss {y['loss']:.9f} gnorm {y['gnorm']} | dloss {x['loss'] - y['loss']:+.2e}")
d1 = max(abs(b1[s]["loss"] - c[s]["loss"]) for s in range(5)); d2 = max(abs(b2[s]["loss"] - c[s]["loss"]) for s in range(5, 10))
print(f"RESUME_CMP max |dloss| steps 0-4 (same process path, run-to-run noise) {d1:.3e}; steps 5-9 (after the S3 resume) {d2:.3e}; sec/step {sum(c[s]['sec'] for s in range(2, 10)) / 8:.2f}; peak {max(c[s]['peak_gib'] for s in range(10))} GiB")
PY
  log "$(tail -1 $W/resume_cmp.txt)"; for f in resume_cmp.txt a.log b1.log b2.log c.log; do aws s3 cp --only-show-errors $W/$f $O/$f; done; touch $W/b.ok
fi
if [ ! -f $W/e.ok ]; then
  KSET=s4400 bash $M/scripts/mx_kown.sh kl_smoke_exp $W/b2/export > $W/e.out 2>&1; rc=$?
  log "(e) export in karpo's engine, 4,400 screen rc=$rc: $(grep -E 'KOWN_DONE|DIE' $M/work/kl_smoke_exp/kown_s4400/driver.log | tail -1 | cut -c1-500)"; [ $rc = 0 ] && touch $W/e.ok
fi
log "KL_SMOKE_DONE"
