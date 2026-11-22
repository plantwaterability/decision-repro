#!/bin/sh
# task_ddp.sh gate|93 (pre-registered 2026-10-04, AGENT_NOTES.md; user 2026-10-03: "can we run a much larger training job on p8? maybe
# all GPUs with DDP?"): WHOLE-NODE Gemma runs on p8's 8 H100s through claude_lmtune.py's data parallelism (fleet/ddp_train.sh,
# research/ddp.md). Run ONLY as a node-hold job (fleetq/hold/p8 + gq_runner.sh v2, which sets CUDA_VISIBLE_DEVICES to every GPU and
# JEV_NODE_GPUS=8). Fast path (pack + compile, gemma_fast_h100.sh) everywhere; scoring on the plain exact path.
#   gate  F2 for DDP on the (90f) recipe (mix2 + rd1), 300 steps, eval every 50: two one-GPU references (2 x 4, seed 0, GPUs 0 / 1 in
#         parallel), then ONE 8-rank run at per-rank 1 x 1 = the same example stream. PASS iff at eval steps 200 / 250 / 300
#         |ddp - mean(ref)| <= max(|ref0 - ref1|, 0.015). Verdict -> results93/ddp_gate.txt (exit 0 once measured, PASS or FAIL).
#   93    (93) = (91)'s recipe at 8x the batch: mix2 + rd2 + flights, global 64 (per-rank 2 x 4), 12,000 steps, lr 1e-4, warmup 40,
#         seed 0, save every 50 (a preempted run resumes from S3 at 8 GPUs). Refuses unless the gate said PASS. Scores in parallel on
#         the 8 GPUs, then queues task_67 / 67b / 70 d93.
#   99    (99) = (93)'s recipe on MORE DATA (user 2026-10-04: "a run trained on more data, as well as gdelt"): mix2 + rd3 (nearly all
#         gated real data, per-source cap; the rd3 STACKPOOL is NOT in it) + flights + GDELT v1 train (all 150k), global 64, seed 0,
#         + the auxiliary Bayesian last layer trained alongside (--aux-vbll grid; detached, so the LoRA is the same with or without it)
#         + --gc-budget 24576 (speed only). Refuses unless the DDP gate and the auxsmoke gate said PASS and fleetq/code has --aux-vbll.
#         Scores with --aux-vbll (test_p unchanged), plus GDELT (eval_gdelt in-stage) and the stack pool; queues task_67b / 67 / 70 / di d99.
#   93x   (93x) EXPOSURE-MATCHED BATCH CONTROL (audit 2026-10-05; user OK): (93)'s recipe at global 64 for 1,500 steps = 96,000 examples,
#         the SAME exposure as (91) (12,000 x 8). Same lr 1e-4, warmup 40, cosine over its own 1,500 steps, seed 0. (91) vs (93x) isolates
#         batch size at fixed examples; (93x) vs (93) isolates exposure (8x) at fixed batch.
STAGE=${1:?usage: task_ddp.sh gate|93|93s<SEED>|93x|99|99s<SEED>}
case "$STAGE" in gate|93|93s[0-9]|93s[0-9][0-9]|93x|99|99s[0-9]|99s[0-9][0-9]) ;; *) echo "stage must be gate, 93, 93s<SEED>, 93x, 99 or 99s<SEED>"; exit 1;; esac
SEED=0; case "$STAGE" in 93s*) SEED=${STAGE#93s};; 99s*) SEED=${STAGE#99s};; esac   # 2026-10-06: 99s1 = seed replicate of (99) (V=99s1), same recipe and code   # 2026-10-04: 93s1 = seed replicate of (93) (V=93s1), same recipe and code
STEPS93=12000; [ "$STAGE" = 93x ] && STEPS93=1500
NG=${JEV_NODE_GPUS:-0}
[ "$NG" -ge 8 ] || { echo "task_ddp.sh needs the whole node (JEV_NODE_GPUS=$NG): queue it only through fleetq/hold/<node>"; exit 1; }
case "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" in *H100*) ;; *) echo "not H100"; exit 1;; esac
J=/data/shared/jev; D=${JEV_D:-/data/local}; S3=s3://YOUR_BUCKET/jev-race; A=analysis; R=$D/runs
RR=$S3/results93; case $STAGE in 99|99s*) RR=$S3/results99;; esac; M=google/gemma-4-31B-it
[ -f /data/shared/fleet/env.sh ] && . /data/shared/fleet/env.sh
export HF_HOME=${JEV_HF:-$D/hf} HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$J
cd $J || exit 1; mkdir -p $R $A
for f in trainers/claude_lmtune.py trainers/claude_lmscore.py trainers/cache_features.py; do aws s3 cp --only-show-errors $S3/fleetq/code/$f $J/$f || { echo "NO_CODE $f"; exit 1; }; done
grep -q 'WORLD_SIZE' $J/trainers/claude_lmtune.py || { echo "NO_DDP_IN_TRAINER (fleetq/code is not the DDP build)"; exit 1; }
aws s3 cp --only-show-errors $S3/fleetq/scripts/ddp_train.sh $R/ddp_train.sh || { echo "NO_DDP_LAUNCHER"; exit 1; }
echo "== (ddp $STAGE) start $(date -u +%FT%TZ) on $(hostname), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, trainer md5 $(md5sum < $J/trainers/claude_lmtune.py | cut -c1-12)"
have() { aws s3 ls $RR/$1 > /dev/null 2>&1; }
[ -d $D/jev_real/test ] && [ -d $D/jev_data/train ] || sh /data/shared/fleet/setup_node.sh
[ -d $D/jev_ext_strict/train ] || { aws s3 cp --only-show-errors $S3/data/jev_ext_strict.tgz $D/ && tar xzf $D/jev_ext_strict.tgz -C $D; }
[ -d $D/jev_ext2/train ] || { aws s3 cp --only-show-errors $S3/data/jev_ext2.tgz $D/ && tar xzf $D/jev_ext2.tgz -C $D; }
[ -d $D/jev_mix2/train ] || python3 - <<PY
import glob, json, os, sys
sys.path.insert(0, "analysis")
from claude_overlap import norm
D = "$D"
test = {norm(json.loads(l)["state"]) for l in open(f"{D}/jev_real/test/real.jsonl")}
for split in ("train", "val"):
    os.makedirs(f"{D}/jev_mix2/{split}", exist_ok=True)
    for p in sorted(glob.glob(f"{D}/jev_data/{split}/*.jsonl")):
        with open(p) as f, open(f"{D}/jev_mix2/{split}/{os.path.basename(p)}", "w") as g:
            for l in f:
                if norm(json.loads(l)["state"]) not in test: g.write(l)
    for src, pre in (("jev_ext_strict", "ext_"), ("jev_ext2", "ext2_")):
        for p in sorted(glob.glob(f"{D}/{src}/{split}/**/*.jsonl", recursive=True)):
            q = f"{D}/jev_mix2/{split}/{pre}{os.path.relpath(p, f'{D}/{src}/{split}').replace('/', '__')}"
            if not os.path.exists(q): os.symlink(p, q)
print("MIX2_DONE", len(os.listdir(D + "/jev_mix2/train")), "train files")
PY
[ -f $D/jev_real_typed/test/real.jsonl ] || python3 - <<'PY'
import json, os
D = os.environ.get("JEV_D", "/data/local")
os.makedirs(f"{D}/jev_real_typed/test", exist_ok=True)
with open(f"{D}/jev_real_typed/test/real.jsonl", "w") as g:
    for l in open(f"{D}/jev_real/test/real.jsonl"):
        if all("/" not in k for k in json.loads(l)["questions"]): g.write(l)
PY

aws s3 cp --only-show-errors $S3/fleetq/scripts/gemma_fast_h100.sh $R/gemma_fast_h100.sh || { echo "NO_FAST_HELPER"; exit 1; }
. $R/gemma_fast_h100.sh
ALL=$CUDA_VISIBLE_DEVICES
if [ $STAGE = gate ]; then
  have ddp_gate.txt && { echo "HAVE gate verdict: $(aws s3 cp --only-show-errors $RR/ddp_gate.txt - | head -1)"; exit 0; }
  [ -d $D/jev_rd1/train ] || { aws s3 cp --only-show-errors $S3/data/jev_rd1.tgz $D/ && tar xzf $D/jev_rd1.tgz -C $D; }
  [ -d $D/jev_rd1/train ] || { echo "NO_RD1"; exit 1; }
  X=$D/jev_x90f; mkdir -p $X/train $X/val          # = task_90.sh f's training set (mix2 + rd1)
  for p in $D/jev_mix2/train/*.jsonl; do q=$X/train/$(basename $p); [ -e $q ] || ln -s $(readlink -f $p) $q; done
  for p in $D/jev_mix2/val/*.jsonl; do q=$X/val/$(basename $p); [ -e $q ] || ln -s $(readlink -f $p) $q; done
  for p in $D/jev_rd1/train/*.jsonl; do q=$X/train/rd1_$(basename $p); [ -e $q ] || ln -s $p $q; done
  G="--model $M --exact-path $GEMMA_FAST_ARGS --lora-mlp --data $X --steps 300 --lr 1e-4 --warmup 40 --max-len 2048 --eval-every 50 --seed 0 --resume off"
  for i in 0 1; do
    CUDA_VISIBLE_DEVICES=$i python3 trainers/claude_lmtune.py $G --batch-size 2 --grad-accum 4 --out $R/ddpgate_ref$i.pt > $R/ddpgate_ref$i.log 2>&1 &
  done; wait
  CUDA_VISIBLE_DEVICES=$ALL sh $R/ddp_train.sh 8 -- $G --batch-size 1 --grad-accum 1 --out $R/ddpgate_ddp.pt > $R/ddpgate_ddp.log 2>&1
  echo "DDP_EXIT=$?"
  python3 - $R > $R/ddp_gate.txt <<'PYG'
import re, sys
R = sys.argv[1]
def vals(n):
    t = open(f"{R}/ddpgate_{n}.log").read()
    v = {int(s): float(x) for s, x in re.findall(r"step\s+(\d+)\s+val soft-CE\s+([0-9.]+)", t)}
    sp = re.findall(r"([0-9.]+)s/step", t)
    return v, (float(sp[-1]) if sp else float("nan")), ("DDP world 8" in t)
r0, s0, _ = vals("ref0"); r1, s1, _ = vals("ref1"); dd, sd, w8 = vals("ddp")
ok, rows = w8, []
for st in (200, 250, 300):
    if st not in r0 or st not in r1 or st not in dd: ok = False; rows.append(f"step {st}: MISSING"); continue
    m, band = (r0[st] + r1[st]) / 2, max(abs(r0[st] - r1[st]), 0.015)
    good = abs(dd[st] - m) <= band; ok &= good
    rows.append(f"step {st}: ref {r0[st]:.4f} / {r1[st]:.4f} ddp {dd[st]:.4f} |d| {abs(dd[st] - m):.4f} band {band:.4f} {'ok' if good else 'OUT'}")
print(("PASS" if ok else "FAIL") + f" ddp-gate (F2, 8 x (1x1) vs one-GPU 2x4); world8={w8}; s/step ref {s0:.2f} ddp {sd:.2f} ({s0 / sd if sd else 0:.2f}x)")
print("\n".join(rows))
print("all evals ref0", {k: round(v, 4) for k, v in sorted(r0.items())}); print("all evals ddp ", {k: round(v, 4) for k, v in sorted(dd.items())})
PYG
  cat $R/ddp_gate.txt
  for f in ddpgate_ref0.log ddpgate_ref1.log ddpgate_ddp.log; do aws s3 cp --only-show-errors $R/$f $RR/$f; done
  aws s3 cp --only-show-errors $R/ddp_gate.txt $RR/ddp_gate.txt && echo "GATE_DONE $(head -1 $R/ddp_gate.txt | cut -c1-4)"
  exit 0
fi
# ---- (99)
if [ $STAGE = 99 ] || [ "${STAGE#99s}" != "$STAGE" ]; then
RD3_MD5=32677581cc3e4b74f10c89108e44d94c; SP_MD5=3c52345eb0d0c04794d29f66723945a7; STEPS=18000   # rd3 agent 2026-10-04: 1,153,401 train units ~ 1 epoch at global 64
aws s3 cp --only-show-errors $S3/results93/ddp_gate.txt $R/ddp_gate.txt 2>/dev/null; head -1 $R/ddp_gate.txt 2>/dev/null | grep -q '^PASS' || { echo "GATE_NOT_PASSED: $(head -1 $R/ddp_gate.txt 2>/dev/null)"; exit 1; }
aws s3 cp --only-show-errors $S3/results_auxsmoke/verdict.txt $R/auxsmoke_verdict.txt 2>/dev/null; head -1 $R/auxsmoke_verdict.txt 2>/dev/null | grep -q '^VERDICT PASS' || { echo "AUXSMOKE_NOT_PASSED: $(head -1 $R/auxsmoke_verdict.txt 2>/dev/null)"; exit 1; }
grep -q -- '--aux-vbll' $J/trainers/claude_lmtune.py && grep -q -- '--aux-vbll' $J/trainers/claude_lmscore.py || { echo "NO_AUX_VBLL_IN_FLEETQ_CODE (promote fleetq/code/auxvbll first)"; exit 1; }
getd() {   # getd KEY MD5 DIR: fetch + md5-check + extract a corpus tarball (extracted aside, moved into place)
  [ -d $D/$3 ] && return 0
  aws s3 cp --only-show-errors $S3/$1 $D/$(basename $1) || { echo "NO_DATA $1"; return 1; }
  [ "$(md5sum < $D/$(basename $1) | cut -c1-32)" = $2 ] || { echo "MD5_MISMATCH $1"; return 1; }
  mkdir -p $D/.x99_tmp_$$ && tar xzf $D/$(basename $1) -C $D/.x99_tmp_$$ || { echo "UNTAR_FAIL $1"; return 1; }
  [ -d $D/.x99_tmp_$$/$3 ] || { echo "NO_DIR $3 in $1: $(ls $D/.x99_tmp_$$)"; return 1; }
  mv $D/.x99_tmp_$$/$3 $D/$3
}
getd data/jev_rd3.tgz $RD3_MD5 jev_rd3 || exit 1
getd data/jev_rd3_stackpool.tgz $SP_MD5 jev_rd3_stackpool || exit 1
getd data/gdelt/jev_gdelt_v1.tgz b77bb5ae946b7f4b7456ddb73fea06ce jev_gdelt_v1 || exit 1
[ -d $D/bp_train_v1/exact/train ] || { aws s3 cp --only-show-errors $S3/fleetq/data/bp_train_v1.tgz $D/ && tar xzf $D/bp_train_v1.tgz -C $D; }
[ -d $D/bp_train_v1/exact/train ] || { echo "NO_BP_TRAIN"; exit 1; }
X=$D/jev_x99; mkdir -p $X/train $X/val
for p in $D/jev_mix2/train/*.jsonl; do q=$X/train/$(basename $p); [ -e $q ] || ln -s $(readlink -f $p) $q; done
for p in $D/jev_mix2/val/*.jsonl; do q=$X/val/$(basename $p); [ -e $q ] || ln -s $(readlink -f $p) $q; done
for p in $D/jev_rd3/train/*.jsonl; do q=$X/train/rd3_$(basename $p); [ -e $q ] || ln -s $p $q; done
for p in $D/bp_train_v1/exact/train/*.jsonl; do q=$X/train/bp_$(basename $p); [ -e $q ] || ln -s $p $q; done
for p in $D/jev_gdelt_v1/train/*.jsonl; do q=$X/train/gdelt_$(basename $p); [ -e $q ] || ln -s $p $q; done
# the stack pool must not be in the training set: no record of it may appear in any train file (normalised state)
python3 - $X $D/jev_rd3_stackpool <<'PYS' || { echo "STACKPOOL_LEAK_CHECK_FAILED"; exit 1; }
import glob, json, os, sys
sys.path.insert(0, "analysis")
from claude_overlap import norm
X, SP = sys.argv[1], sys.argv[2]
pool = {norm(json.loads(l)["state"]) for p in glob.glob(f"{SP}/**/*.jsonl", recursive=True) for l in open(p) if l.strip()}
n = leak = 0; units = 0
for p in sorted(glob.glob(f"{X}/train/*.jsonl")):
    for l in open(p):
        if not l.strip(): continue
        r = json.loads(l); n += 1; units += len(r["questions"])
        if norm(r["state"]) in pool: leak += 1
print(f"X99 train records {n} units {units}; stack-pool states {len(pool)}; leaked {leak}")
sys.exit(1 if leak or not pool else 0)
PYS
echo "X99_FILES train $(ls $X/train | wc -l) val $(ls $X/val | wc -l)"
V=99; [ "$SEED" != 0 ] && V=99s$SEED; C=$R/bbtune_gemma31it_$V.pt
AUX="--aux-vbll --aux-vbll-kl 1,1e4,1e6 --aux-vbll-prior 1 --aux-vbll-cov diag,full --aux-vbll-dim 128 --aux-vbll-lr 1e-3 --aux-vbll-samples 64"
if have bbtune_gemma31it_$V.pt; then aws s3 cp --only-show-errors $RR/bbtune_gemma31it_$V.pt $C; else
  CUDA_VISIBLE_DEVICES=$ALL sh $R/ddp_train.sh 8 -- --model $M --exact-path $GEMMA_FAST_ARGS --gc-budget 24576 --lora-mlp --data $X --steps $STEPS \
    --batch-size 2 --grad-accum 4 --lr 1e-4 --warmup 40 --max-len 2048 --eval-every 250 --save-every 50 --seed $SEED --out $C $AUX >> $R/bbtune_gemma31it_$V.log 2>&1
  rc=$?; echo "TUNE99_EXIT=$rc $(grep -E 'DDP world|val soft-CE' $R/bbtune_gemma31it_$V.log | tail -3 | tr '\n' ' ' | cut -c1-300)"
  [ $rc -eq 0 ] && [ -s $C ] || { aws s3 cp --only-show-errors $R/bbtune_gemma31it_$V.log $RR/bbtune_gemma31it_$V.fail.log; exit 1; }
  grep -q "NC FILTER (train)" $R/bbtune_gemma31it_$V.log || { echo "NC_FILTER_NOT_APPLIED"; exit 1; }
  grep -q "aux-vbll val head" $R/bbtune_gemma31it_$V.log || { echo "AUX_VBLL_NOT_TRAINED"; exit 1; }
  aws s3 cp --only-show-errors $C $RR/ && aws s3 cp --only-show-errors $R/bbtune_gemma31it_$V.log $RR/ && echo "CKPT99_UPLOADED $(date -u +%FT%TZ)"
fi
E=$D/bayes_pref_v1; [ -d $E/bayes_pref/eval ] || { mkdir -p $E && aws s3 cp --only-show-errors $S3/fleetq/data/bayes_pref_v1.tgz $D/ && tar xzf $D/bayes_pref_v1.tgz -C $E; }
GG=$D/gde1v3; [ -d $GG/gde1/sys1cal/eval ] || { mkdir -p $GG && aws s3 cp --only-show-errors $S3/data/gde1_v3_gated.tgz $D/ && tar xzf $D/gde1_v3_gated.tgz -C $GG; }
SPS=$(ls -d $D/jev_rd3_stackpool/*/ | xargs -n1 basename | tr '\n' ',' | sed 's/,$//')
i=0
for spec in "sp1_gemma31it${V}_typed_zeroshot_units:--data $D/jev_real_typed --n-perm 1" "gemma31it${V}_typed_zeroshot_units:--data $D/jev_real_typed --n-perm 4" \
            "sp1_gemma31it${V}_zeroshot_units:--data $D/jev_real --n-perm 1" "bp_d$V:--data $E/bayes_pref --splits eval --n-perm 1" \
            "gde1_d${V}_sys1cal:--data $GG/gde1/sys1cal --splits eval --n-perm 1" "gdelt_d$V:--data $D/jev_gdelt_v1 --splits val,test,test_recent --n-perm 1" \
            "stackpool_d$V:--data $D/jev_rd3_stackpool --splits $SPS --n-perm 1"; do
  n=${spec%%:*}; args=${spec#*:}; out=$A/claude_$n.npz
  if have $(basename $out); then echo "HAVE $n"; else
    ( CUDA_VISIBLE_DEVICES=$i python3 trainers/claude_lmscore.py --model $M --exact-path --lora-ckpt $C $args --exclude-identity --aux-vbll --out $out > $R/$n.log 2>&1
      echo "SCORE99_${n}_EXIT=$? $(grep wrote $R/$n.log | cut -c1-150)"; [ -f $out ] && aws s3 cp --only-show-errors $out $RR/ && aws s3 cp --only-show-errors ${out%.npz}_auxvbll.npz $RR/ 2>/dev/null ) &
  fi; i=$((i + 1))
done; wait
aws s3 cp --only-show-errors $S3/fleetq/code/analysis/eval_gdelt.py $J/analysis/eval_gdelt.py && [ -f $A/claude_gdelt_d$V.npz ] && \
  python3 analysis/eval_gdelt.py --data $D/jev_gdelt_v1 --npz $A/claude_gdelt_d$V.npz --splits val,test,test_recent --boot 2000 --out $R/gdelt_eval_d$V.json > $R/gdelt_eval_d$V.txt 2>&1 && \
  aws s3 cp --only-show-errors $R/gdelt_eval_d$V.txt $RR/ && aws s3 cp --only-show-errors $R/gdelt_eval_d$V.json $RR/ && grep -E '^(test|test_recent) +ALL +model' $R/gdelt_eval_d$V.txt
for t in 1 2 3 4 5 6; do
  E=$(aws s3api get-object --bucket YOUR_BUCKET --key jev-race/fleetq/queue.txt $R/q99.txt --query ETag --output text 2>/dev/null) || { sleep 20; continue; }
  for l in "task_67b.sh d$V" "task_67.sh d$V" "task_70.sh d$V v3" "task_di.sh d$V"; do grep -qxF "$l" $R/q99.txt || echo "$l"; done > $R/q99_new.txt
  [ -s $R/q99_new.txt ] || { echo "EVALS_ALREADY_QUEUED"; break; }
  cat $R/q99_new.txt $R/q99.txt > $R/q99_2.txt && aws s3api put-object --bucket YOUR_BUCKET --key jev-race/fleetq/queue.txt --body $R/q99_2.txt --if-match "$E" > /dev/null 2>&1 && { echo "EVALS_QUEUED (front)"; break; }
  sleep $((t * 5))
done
echo "ALL_DONE_99 $(date -u +%FT%TZ)"
exit 0
fi
# ---- (93)
aws s3 cp --only-show-errors $RR/ddp_gate.txt $R/ddp_gate.txt 2>/dev/null; head -1 $R/ddp_gate.txt 2>/dev/null | grep -q '^PASS' || { echo "GATE_NOT_PASSED: $(head -1 $R/ddp_gate.txt 2>/dev/null)"; exit 1; }
n=0; until aws s3 ls $S3/data/jev_rd2.tgz > /dev/null 2>&1; do n=$((n + 1)); [ $n -gt 6 ] && { echo "NO_RD2"; exit 1; }; sleep 300; done
[ -d $D/jev_rd2/train ] || { aws s3 cp --only-show-errors $S3/data/jev_rd2.tgz $D/ && tar xzf $D/jev_rd2.tgz -C $D; }
[ -d $D/jev_rd2/train ] || { echo "NO_RD2"; exit 1; }
X=$D/jev_x93; mkdir -p $X/train $X/val
for p in $D/jev_mix2/train/*.jsonl; do q=$X/train/$(basename $p); [ -e $q ] || ln -s $(readlink -f $p) $q; done
for p in $D/jev_mix2/val/*.jsonl; do q=$X/val/$(basename $p); [ -e $q ] || ln -s $(readlink -f $p) $q; done
for p in $D/jev_rd2/train/*.jsonl; do q=$X/train/rd1_$(basename $p); [ -e $q ] || ln -s $p $q; done   # (84)'s naming
# (82) / (90g)'s exact-posterior Bayesian-teaching flights corpus
[ -d $D/bp_train_v1/exact/train ] || { aws s3 cp --only-show-errors $S3/fleetq/data/bp_train_v1.tgz $D/ && tar xzf $D/bp_train_v1.tgz -C $D; }
[ -d $D/bp_train_v1/exact/train ] || { echo "NO_BP_TRAIN"; exit 1; }
for p in $D/bp_train_v1/exact/train/*.jsonl; do q=$X/train/bp_$(basename $p); [ -e $q ] || ln -s $p $q; done
echo "X93_FILES train $(ls $X/train | wc -l) val $(ls $X/val | wc -l)"
V=93; [ "$SEED" != 0 ] && V=93s$SEED; [ "$STAGE" = 93x ] && V=93x; C=$R/bbtune_gemma31it_$V.pt
if have bbtune_gemma31it_$V.pt; then aws s3 cp --only-show-errors $RR/bbtune_gemma31it_$V.pt $C; else
  CUDA_VISIBLE_DEVICES=$ALL sh $R/ddp_train.sh 8 -- --model $M --exact-path $GEMMA_FAST_ARGS --lora-mlp --data $X --steps $STEPS93 --batch-size 2 --grad-accum 4 \
    --lr 1e-4 --warmup 40 --max-len 2048 --eval-every 250 --save-every 50 --seed $SEED --out $C >> $R/bbtune_gemma31it_$V.log 2>&1
  rc=$?; echo "TUNE93_EXIT=$rc $(grep -E 'DDP world|val soft-CE' $R/bbtune_gemma31it_$V.log | tail -3 | tr '\n' ' ' | cut -c1-300)"
  [ $rc -eq 0 ] && [ -s $C ] || { aws s3 cp --only-show-errors $R/bbtune_gemma31it_$V.log $RR/bbtune_gemma31it_$V.fail.log; exit 1; }
  grep -q "NC FILTER (train)" $R/bbtune_gemma31it_$V.log || { echo "NC_FILTER_NOT_APPLIED"; exit 1; }
  aws s3 cp --only-show-errors $C $RR/ && aws s3 cp --only-show-errors $R/bbtune_gemma31it_$V.log $RR/ && echo "CKPT93_UPLOADED $(date -u +%FT%TZ)"
fi
# score in parallel, one spec per GPU (plain exact path)
E=$D/bayes_pref_v1; [ -d $E/bayes_pref/eval ] || { mkdir -p $E && aws s3 cp --only-show-errors $S3/fleetq/data/bayes_pref_v1.tgz $D/ && tar xzf $D/bayes_pref_v1.tgz -C $E; }
GG=$D/gde1v3; [ -d $GG/gde1/sys1cal/eval ] || { mkdir -p $GG && aws s3 cp --only-show-errors $S3/data/gde1_v3_gated.tgz $D/ && tar xzf $D/gde1_v3_gated.tgz -C $GG; }
i=0
for spec in "sp1_gemma31it${V}_typed_zeroshot_units:--data $D/jev_real_typed --n-perm 1" "gemma31it${V}_typed_zeroshot_units:--data $D/jev_real_typed --n-perm 4" \
            "sp1_gemma31it${V}_zeroshot_units:--data $D/jev_real --n-perm 1" "bp_d$V:--data $E/bayes_pref --splits eval --n-perm 1" \
            "gde1_d${V}_sys1cal:--data $GG/gde1/sys1cal --splits eval --n-perm 1"; do
  n=${spec%%:*}; args=${spec#*:}; out=$A/claude_$n.npz
  if have $(basename $out); then echo "HAVE $n"; else
    ( CUDA_VISIBLE_DEVICES=$i python3 trainers/claude_lmscore.py --model $M --exact-path --lora-ckpt $C $args --exclude-identity --out $out > $R/$n.log 2>&1
      echo "SCORE93_${n}_EXIT=$? $(grep wrote $R/$n.log | cut -c1-150)"; [ -f $out ] && aws s3 cp --only-show-errors $out $RR/ ) &
  fi; i=$((i + 1))
done; wait
Q=$S3/fleetq
for t in 1 2 3 4 5 6; do
  E=$(aws s3api get-object --bucket YOUR_BUCKET --key jev-race/fleetq/queue.txt $R/q93.txt --query ETag --output text 2>/dev/null) || { sleep 20; continue; }
  for l in "task_67b.sh d$V" "task_67.sh d$V" "task_70.sh d$V v3"; do grep -qxF "$l" $R/q93.txt || echo "$l"; done > $R/q93_new.txt
  [ -s $R/q93_new.txt ] || { echo "EVALS_ALREADY_QUEUED"; break; }
  cat $R/q93_new.txt $R/q93.txt > $R/q93_2.txt && aws s3api put-object --bucket YOUR_BUCKET --key jev-race/fleetq/queue.txt --body $R/q93_2.txt --if-match "$E" > /dev/null 2>&1 && { echo "EVALS_QUEUED (front)"; break; }
done
echo "ALL_DONE_93 $(date -u +%FT%TZ)"
