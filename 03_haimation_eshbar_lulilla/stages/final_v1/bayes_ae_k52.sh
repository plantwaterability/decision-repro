#!/bin/bash
# ESBAR-FINAL: bayes_ae.py on the final run, with the fresh eval restricted to K <= 52 rows. The B / C / D readouts come from the pinned
# AuxVBLL (claude_lmtune e2c6100a), whose letter codes are 52 wide: its moments() fails on K > 52 rows (CLINC150 / BANKING77 / 23 chess
# rows, 1,223 of 19,737). The subset file keeps every per-row field of the K <= 52 rows and trims the option columns to 52.
unset ASCALE_RUN_ID
export OMP_NUM_THREADS=48 PATH=/data/local/venv_t2141/bin:$PATH
C=/data/local/esbar_code_v2
F=/data/local/esbar_final_v1b
FF=$F/features_final
S3R=s3://YOUR_BUCKET/jev-race/engineering/esbar/runs/stage1_final_v1
log() { echo "[bayes_ae_k52 $(date -u +%FT%TZ)] $*" | tee -a $F/logs/stage3.log; }
cd /tmp || exit 1
python - $FF/eval_fresh.pt $FF/eval_fresh_k52.pt <<'PY' >> $F/logs/stage3.log 2>&1 || { log "SUBSET_FAILED"; exit 1; }
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
f = torch.load(src, map_location="cpu", weights_only=False)
n = len(f["decision_uid"])
keep = (f["k"] <= 52).nonzero().flatten()
idx = keep.tolist()
out = {}
for k, v in f.items():
    if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == n:
        v = v[keep]
        if k in ("ell", "t", "loss_by_option") and v.dim() == 2:
            v = v[:, :52].contiguous()
        out[k] = v
    elif isinstance(v, list) and len(v) == n:
        out[k] = [v[i] for i in idx]
    else:
        out[k] = v
out["meta"] = dict(f.get("meta") or {}, subset="K <= 52 rows of eval_fresh.pt (option columns trimmed to 52)", n_full=n)
torch.save(out, dst)
print(f"subset: {len(idx)} of {n} rows (K <= 52), max k {int(out['k'].max())}, ell {tuple(out['ell'].shape)}")
PY
t0=$(date +%s)
python3 /data/local/esbar_data/evalr/bayes_ae.py --code $C/trainers --heads $F/head_final --cal $FF/cal.pt \
  --eval fresh_k52=$FF/eval_fresh_k52.pt dev=$FF/dev.pt w2c=$FF/when2call_4opt_clean_dev.pt --out $F/bayes_ae_final.json > $F/logs/bayes_ae_final_k52.log 2>&1
rc=$?
log "BAYES_AE final (fresh K <= 52) rc=$rc wall_s=$(( $(date +%s) - t0 ))"
for f in $F/bayes_ae_final.json $F/logs/bayes_ae_final_k52.log $F/logs/bayes_ae_final.log $F/logs/stage3.log $F/head_read_fresh.json $F/logs/head_read_fresh.log; do
  [ -f $f ] && aws s3 cp --only-show-errors $f $S3R/report/$(basename $f); done
aws s3 cp --only-show-errors $FF/eval_fresh_k52.pt $S3R/features_final/eval_fresh_k52.pt
log "DONE"
