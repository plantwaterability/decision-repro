#!/bin/bash
# soup75 (PREREG AGENT_NOTES "PREREG soup75", 2026-11-10): soup = A*E1 + (1-A)*X0, X0 = a v7.5 export (X0SRC on S3, pulled against its
# READBACK_MD5.txt, identity gate X0ID); every tensor, fp32 math, E1's dtype out (bf16), E1's 12-shard layout / index / non-weight files;
# head refit with soupA's procedure; export -> S3 with per-file md5 read-back. = soupB_build_v2.sh with ONLY the second parent, paths /
# labels and the rendezvous port changed. Structural mismatch = DIE. Idempotent (.ok stamps).
#   TAG=soup75h50 ALPHA=0.5 X0LABEL=v75h X0SRC=s3://.../esbar_v75/export_half X0ID=<sha256> NPROC=2 CUDA_VISIBLE_DEVICES=4,5 bash soup75_build.sh
set -u
unset ASCALE_RUN_ID
export AWS_DEFAULT_REGION=us-east-2 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_AVOID_RECORD_STREAMS=1 NCCL_RUNTIME_CONNECT=0
TAG=${TAG:?}; A=${ALPHA:?}; X0SRC=${X0SRC:?}; X0ID=${X0ID:?}; X0LABEL=${X0LABEL:?}
D=/data/local; B=s3://YOUR_BUCKET/jev-race; E=$B/engineering/esbar; S3R=$E/runs/esbar_$TAG
W=$D/$TAG; X1=$D/soupB_in/e1_export; X0=$D/soup75_in/$X0LABEL; XS=$W/soup_export
V=$D/venv_t2141; PY=$V/bin/python; export PATH=$V/bin:$PATH
C=$D/esbar_code_v2e; MAN=$D/esbar_v5/manifests; FV=$W/features; HD=$W/head
mkdir -p $W/logs $W/tmp $FV $HD $D/soupB_in $D/soup75_in
L=$W/logs/soup.log
log() { echo "[$TAG $(date -u +%FT%TZ)] $*" | tee -a $L; }
die() { log "DIE $*"; aws s3 cp --only-show-errors $L $S3R/logs/soup.log 2>/dev/null; exit 1; }
md5() { md5sum < "$1" | cut -c1-32; }
fetch() { [ -s $2 ] || { mkdir -p $(dirname $2) && aws s3 cp --only-show-errors $1 $2; } || die "NO_INPUT $1"
          [ -z "${3:-}" ] || [ "$(md5sum < $2 | cut -c1-${#3})" = "$3" ] || die "MD5 $1 $(md5sum < $2 | cut -c1-8) != $3"; }
ident() { (cd $1 && ls model*.safetensors config.json | sort | xargs -P 13 -n 1 sha256sum | sort -k2) | sha256sum | cut -c1-64; }
log "start host $(hostname) alpha_E1 $A"
$PY -c "import torch, safetensors; torch.zeros(1).cuda(); print('torch', torch.__version__)" >> $L 2>&1 || die ENV
# ---- 1. inputs: E1 (v5 export incl. its own head files, identity c92e70ea) + stock base (s3 models/, per-file sha256 = MANIFEST.json)
if [ ! -f $X1/.ok ]; then
  mkdir -p $X1 && aws s3 sync --only-show-errors $E/runs/esbar_v5/export/ $X1/ || die SYNC_E1
  [ "$(ls $X1/model-*.safetensors | wc -l)" = 12 ] || die E1_SHARDS
  I1=$(ident $X1); [ "$I1" = c92e70ea0dcbfb4d62e04164e7fe48cff4b292b6b47175ae94b4f5987f63eeb3 ] || die "E1_IDENTITY $I1"
  touch $X1/.ok
fi
log "E1 export ok (identity c92e70ea; head json $(md5 $X1/vbll_head.json))"
if [ ! -f $X0/.ok ]; then
  aws s3 cp --only-show-errors $X0SRC/READBACK_MD5.txt $W/x0_READBACK_MD5.txt || die X0_READBACK_FETCH
  grep -q '^BAD ' $W/x0_READBACK_MD5.txt && die "X0_READBACK_HAS_BAD"
  mkdir -p $X0 && aws s3 sync --only-show-errors --exclude 'vbll_head*' --exclude 'READBACK_MD5.txt' $X0SRC/ $X0/ || die SYNC_X0
  [ "$(ls $X0/model-*.safetensors | wc -l)" = 12 ] || die X0_SHARDS
  bad=0; while read st f m; do case $f in vbll_head*) continue;; esac
    [ -f $X0/$f ] || { log "MISSING $f"; bad=$((bad + 1)); continue; }
    [ "$(md5 $X0/$f)" = "$m" ] || { log "MD5_MISMATCH $f"; bad=$((bad + 1)); }; done < <(grep '^OK ' $W/x0_READBACK_MD5.txt)
  [ $bad = 0 ] || die "X0_VERIFY bad=$bad"
  I0=$(ident $X0); [ "$I0" = "$X0ID" ] || die "X0_IDENTITY $I0 != $X0ID"
  touch $X0/.ok
fi
log "X0 ok ($X0LABEL from $X0SRC, md5 = READBACK_MD5.txt, identity $X0ID)"
# ---- 2. structure gate + soup (12 workers, one per E1 shard)
if [ ! -f $XS/.soup_ok ]; then
  mkdir -p $XS
  $PY - $X1 $X0 $XS $A > $W/logs/soup_build.log 2>&1 <<'PY' || { tail -20 $W/logs/soup_build.log | tee -a $L; die SOUP_BUILD; }
import json, sys, multiprocessing as mp
x1, x0, xs, A = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
i1 = json.load(open(f"{x1}/model.safetensors.index.json"))["weight_map"]; i0 = json.load(open(f"{x0}/model.safetensors.index.json"))["weight_map"]
assert set(i1) == set(i0), f"STRUCT key sets differ: only_E1 {sorted(set(i1) - set(i0))[:5]} only_base {sorted(set(i0) - set(i1))[:5]}"
shards = sorted(set(i1.values())); assert len(shards) == 12

def work(sh):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    out, st = {}, {"n": 0, "equal": 0, "rel": []}
    keys = sorted(k for k, v in i1.items() if v == sh)
    opened = {}
    with safe_open(f"{x1}/{sh}", "pt") as a:
        meta = a.metadata()
        for k in keys:
            f0 = i0[k]
            if f0 not in opened: opened[f0] = safe_open(f"{x0}/{f0}", "pt").__enter__()
            t1, t0 = a.get_tensor(k), opened[f0].get_tensor(k)
            assert t1.shape == t0.shape, f"STRUCT {k} shape {t1.shape} vs {t0.shape}"
            st["n"] += 1
            if t1.dtype == t0.dtype and torch.equal(t1, t0):
                st["equal"] += 1; out[k] = t1.clone(); continue
            if not (t1.dtype.is_floating_point and t0.dtype.is_floating_point):
                raise AssertionError(f"STRUCT non-float tensor differs: {k} {t1.dtype}/{t0.dtype}")
            g1, g0 = t1.float(), t0.float()
            out[k] = (A * g1 + (1.0 - A) * g0).to(t1.dtype)
            n = g1.norm().item(); st["rel"].append((g1 - g0).norm().item() / n if n > 0 else 0.0)
    save_file(out, f"{xs}/{sh}", metadata=meta)
    rel = st.pop("rel"); st["rel_E1_base_mean"] = sum(rel) / len(rel) if rel else 0.0; st["rel_E1_base_max"] = max(rel) if rel else 0.0
    return sh, st

with mp.get_context("fork").Pool(12) as pool:
    res = dict(pool.map(work, shards))
tot = {"alpha_E1": A, "tensors": sum(v["n"] for v in res.values()), "identical_E1_base": sum(v["equal"] for v in res.values()),
       "rel_E1_vs_base_mean": sum(v["rel_E1_base_mean"] for v in res.values()) / 12, "rel_E1_vs_base_max": max(v["rel_E1_base_max"] for v in res.values())}
json.dump({"shards": res, "total": tot}, open(f"{xs}/../soup_stats.json", "w"), indent=1)
print("SOUP_STATS", json.dumps(tot))
PY
  for f in config.json tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json processor_config.json model.safetensors.index.json; do cp $X1/$f $XS/$f; done
  $PY - $X1 $X0 $XS $A >> $W/logs/soup_build.log 2>&1 <<'PY' || { tail -5 $W/logs/soup_build.log | tee -a $L; die SOUP_SPOTCHECK; }
import json, random, sys, torch
from safetensors import safe_open
x1, x0, xs, A = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
w1 = json.load(open(f"{x1}/model.safetensors.index.json"))["weight_map"]; w0 = json.load(open(f"{x0}/model.safetensors.index.json"))["weight_map"]
random.seed(7); picks = random.sample([k for k in sorted(w1) if "layers" in k], 3)
for k in picks:
    with safe_open(f"{x1}/{w1[k]}", "pt") as a, safe_open(f"{x0}/{w0[k]}", "pt") as b, safe_open(f"{xs}/{w1[k]}", "pt") as s:
        t1, t0, ts = a.get_tensor(k), b.get_tensor(k), s.get_tensor(k)
        ref = (A * t1.float() + (1.0 - A) * t0.float()).to(t1.dtype)
        assert torch.equal(ref, ts), f"SPOT mismatch {k}"
        print("SPOT_OK", k, "eq_E1", torch.equal(ts, t1), "eq_base", torch.equal(ts, t0))
PY
  touch $XS/.soup_ok
fi
log "soup built: $(grep SOUP_STATS $W/logs/soup_build.log | tail -1)"
XSID=$(ident $XS); echo $XSID > $W/soup_identity.txt
log "soup export identity $XSID"
# ---- 3. head refit (soupA's procedure: v5 role manifests headfit + cal, fullft_train v8 --extract-features, sdv1_head refit + fit-t)
if [ ! -f $C/.ok ]; then
  fetch $E/code/esbar_code_v2e_full.tgz $W/tmp/code2e.tgz e6bd8a5aa97693be477ff50b5234f7ae
  fetch $E/code/esbar_code_v2e_full.md5.txt $W/tmp/code2e.md5.txt ead5408b79fb0ccc8723a753bf2d57fd
  (cd $D && tar xzf $W/tmp/code2e.tgz) || die UNTAR_CODE2E
  (cd $D && md5sum -c --quiet $W/tmp/code2e.md5.txt) > $W/logs/code2e_md5.txt 2>&1 || die "CODE2E_MD5 $(head -3 $W/logs/code2e_md5.txt)"
  touch $C/.ok
fi
mkdir -p $MAN
for m in headfit cal; do for x in $m.jsonl $m.jsonl.header.json; do fetch $B/engineering/coteach_wave4_20261028/p9_stage2_manifests/$x $MAN/$x; done; done
for x in "headfit f3cc7e116c8d264a181da7354e44e8234aab8cf8296aaf20f3e7c73a0b342980" "cal 95f14043c4bbf0524d22b05f12c11461d0d6618ffbbf64e4e756cbba61a97322"; do
  set -- $x
  g=$(sha256sum < $MAN/$1.jsonl | cut -c1-64); w=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['rows_sha256'])" $MAN/$1.jsonl.header.json)
  [ "$g" = "$2" ] && [ "$w" = "$2" ] || die "MANIFEST_SHA $1 $g"
done
log "code v2e + role manifests ok"
export PYTHONPATH=$C/engine:$C/trainers:$C
TR="$PY -m torch.distributed.run --nnodes 1 --nproc-per-node ${NPROC:-8} --rdzv-backend c10d --local-addr 127.0.0.1"
EXTRACTOR="$(md5sum $C/trainers/fullft_train.py | cut -c1-32)+$(md5sum $C/trainers/fullft_mem.py | cut -c1-32)"
port=${PORT0:-29790}
for role in headfit cal; do
  man=$MAN/$role.jsonl; dig=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['rows_sha256'])" $man.header.json)
  Fe=$FV/$role.pt; FM=$FV/$role.stamp.json
  EXP="{\"identity_sha256\": \"esbar_$TAG:$TAG\", \"export_sha256\": \"$XSID\", \"policy\": \"B\", \"role\": \"$role\", \"manifest_rows_sha256\": \"$dig\", \"extractor_md5\": \"$EXTRACTOR\"}"
  echo "$EXP" > $FV/$role.expect.json
  if $PY $C/trainers/sdv1_head.py check-features --features $Fe --meta $FM --manifest $man --expect "$EXP" > /dev/null 2>&1; then log "FEATURES_VALID $role (cached)"; continue; fi
  port=$((port + 1)); t0=$(date +%s)
  $TR --rdzv-endpoint 127.0.0.1:$port $C/trainers/fullft_train.py --init-from $XS --attention-policy B --seed 0 \
    --param-dtype bf16 --reduce-dtype fp32 --pack-tokens 8192 --ac 0 --offload-ac 0 --mlp-chunk 4096 --mlp-chunk-min 16384 \
    --pinned-code $C/trainers --manifest $man --expect-rows-sha256 $dig --extract-features $Fe \
    --out $FV/tmp_$role > $W/logs/features_$role.log 2>&1
  rc=$?; log "FEATURES $role rc=$rc wall_s=$(( $(date +%s) - t0 ))"
  [ $rc -eq 0 ] || die "FEATURES_FAILED $role: $(tail -3 $W/logs/features_$role.log)"
  $PY $C/trainers/sdv1_head.py stamp-features --features $Fe --meta $FM --manifest $man --expect "$EXP" >> $L 2>&1 || die "STAMP_FAILED $role"
done
if [ ! -s $HD/serving/vbll_head.json ]; then
  SDH="$PY $C/trainers/sdv1_head.py"; t0=$(date +%s)
  { OMP_NUM_THREADS=48 $SDH refit --features $FV/headfit.pt --manifest $MAN/headfit.jsonl --expect "$(cat $FV/headfit.expect.json)" --pinned-code $C/trainers --out $HD \
    && OMP_NUM_THREADS=48 $SDH fit-t --features $FV/cal.pt --manifest $MAN/cal.jsonl --expect "$(cat $FV/cal.expect.json)" --pinned-code $C/trainers --heads $HD --rule l01post_di; } > $W/logs/head.log 2>&1
  rc=$?; log "HEAD rc=$rc wall_s=$(( $(date +%s) - t0 ))"
  [ $rc -eq 0 ] || die "HEAD_FAILED: $(tail -3 $W/logs/head.log)"
fi
cp $HD/serving/vbll_head.safetensors $HD/serving/vbll_head.json $XS/ || die HEAD_COPY
log "$(grep -E '^FIT_T' $W/logs/head.log | tail -1 | cut -c1-200)"
log "HEAD in soup export T=$(python3 -c "import json; print(json.load(open('$XS/vbll_head.json'))['temperature'])") json md5 $(md5 $XS/vbll_head.json)"
touch $XS/.ok
# ---- 4. soup export -> S3, per-file md5 read-back (BEFORE any eval)
if [ ! -f $W/.export_s3_ok ]; then
  (cd $XS && ls | grep -v '^\.' | sort | xargs -P 13 -n 1 sha256sum) > $W/soup_export.sha256
  (cd $XS && ls | grep -v '^\.' | sort | xargs -P 13 -n 1 md5sum) > $W/soup_export.md5
  aws s3 sync --only-show-errors $XS/ $S3R/export/ --exclude '.*' || die EXPORT_UPLOAD
  bad=$(cat $W/soup_export.md5 | xargs -P 13 -n 2 sh -c 'r=$(aws s3 cp --only-show-errors '"$S3R"'/export/$1 - | md5sum | cut -c1-32); [ "$r" = "$0" ] || echo "BAD $1"' | wc -l)
  [ "$bad" = 0 ] || die "EXPORT_READBACK bad=$bad"
  for f in soup_export.sha256 soup_export.md5 soup_identity.txt soup_stats.json; do aws s3 cp --only-show-errors $W/$f $S3R/export_meta/$f || die "META_UPLOAD $f"; done
  for f in soup_build.log head.log features_headfit.log features_cal.log; do aws s3 cp --only-show-errors $W/logs/$f $S3R/logs/$f; done
  touch $W/.export_s3_ok
fi
log "EXPORT_UPLOADED files=$(wc -l < $W/soup_export.md5) read-back md5 all MATCH"
aws s3 cp --only-show-errors $L $S3R/logs/soup.log
log "SOUP_BUILD_OK"
