#!/bin/bash
# kl_single.sh (KL agent; coordinator order 17:35Z 11-16): the full PREFIX read (31,198 rows) of soup k7k8k12k3 in karpo's engine with view "single" (each question in a
# request of its own) and, on the SAME node, the joint view again as the reference; per-benchmark deltas single minus joint; if single beats joint by >= 0.3 on the prefix,
# the full public run with view single right after (mx_kfull.sh, tag suffix _single). Measurement only; nothing public. Marker full/DONE_single_view at the end.
set -u; unset ASCALE_RUN_ID; export AWS_DEFAULT_REGION=us-east-2
D=/data/local; K=$D/kl; M=$D/mx; R=s3://YOUR_BUCKET/jev-race/engineering/esbar/runs; F=$R/esbar_kl_code/full; O=$R/esbar_kl_single; mkdir -p $K/logs $K/src $K/single $M/scripts $M/logs; L=$K/single/single.log
log() { echo "[kl_single $(date -u +%FT%TZ)] $*" | tee -a $L; aws s3 cp --only-show-errors $L $O/single.log 2>/dev/null; }
exec 8> $K/node.lock; flock -n 8 || { echo "NODE_LOCK held"; exit 3; }
echo "whole node: KL agent single-view prefix read of soup k7k8k12k3 (measurement), since $(date -u +%FT%TZ)" > /tmp/claim.txt
[ "$(nvidia-smi -L | wc -l)" = 8 ] || { log "FAILED not 8 GPUs"; exit 1; }
aws s3 sync --only-show-errors $R/esbar_mx/scripts/ $M/scripts/ || { log "FAILED mx scripts"; exit 1; }
X=$K/src/soup_k7k8k12k3; WANT=7ecdd654795a46805c58f1a2eed4539a937ae56f88700ae9c0b9733142d22ed8
if [ ! -f $X/.ok ]; then mkdir -p $X && aws s3 sync --only-show-errors --exclude READBACK_MD5.txt $R/esbar_kl_soup_k7k8k12k3/export/ $X/ || { log "FAILED export sync"; exit 1; }
  I=$( (cd $X && ls model*.safetensors config.json | sort | xargs -P 13 -n 1 sha256sum | sort -k2) | sha256sum | cut -c1-64); [ "$I" = "$WANT" ] || { log "FAILED identity $I"; exit 1; }; echo $I > $X/.identity; touch $X/.ok; fi
log "start host $(hostname): soup k7k8k12k3 identity $(cut -c1-16 $X/.identity) staged"
pidx() { grep -E '^INDEX KOWN' $M/work/$1/kown/score.txt 2>/dev/null | tail -1 | awk '{print $3}'; }
TS=kl_soup_k7k8k12k3_single; TJ=kl_soup_k7k8k12k3_j2      # _single = mx_kown's view switch; j2 = the joint view again on this node (own tag: the stored read's chunks are not reused)
[ -n "$(pidx $TS)" ] || KSET=prefix bash $M/scripts/mx_kown.sh $TS $X > $K/single/kown_single.out 2>&1; PS=$(pidx $TS); log "SINGLE view prefix: ${PS:-FAILED} ($(grep -o 'view [a-z]*' $M/work/$TS/kown/driver.log | head -1))"
[ -n "$(pidx $TJ)" ] || KSET=prefix bash $M/scripts/mx_kown.sh $TJ $X > $K/single/kown_joint.out 2>&1; PJ=$(pidx $TJ); log "JOINT view prefix, same node: ${PJ:-FAILED} (stored read on kl3: 73.646)"
[ -n "$PS" ] && [ -n "$PJ" ] || { log "FAILED a read is missing"; echo "$(date -u +%FT%TZ) FAILED single ${PS:-none} joint ${PJ:-none}" | aws s3 cp --only-show-errors - $F/DONE_single_view; exit 1; }
$D/venv_mxk/bin/python - $M/work/$TS/kown/prefix_compare.json $M/work/$TJ/kown/prefix_compare.json $M/kroot/src/prefix_rows.jsonl.gz $M/work/$TS/kown/prefix_results.jsonl $M/work/$TJ/kown/prefix_results.jsonl > $K/single/deltas.txt 2>&1 <<'PY'
import gzip, json, sys, collections
cs, cj, rows, rs, rj = sys.argv[1:6]
S = json.load(open(cs))["arms"]["KOWN"]; J = json.load(open(cj))["arms"]["KOWN"]
names = {}; nq = collections.defaultdict(list)
for l in gzip.open(rows, "rt"):
    r = json.loads(l); e = r["_evaluation"]; names.setdefault(str(e["catalog_id"]), f"{e.get('dataset')} / {r.get('family')}"); nq[str(e["catalog_id"])].append(len(r["questions"]))
print(f"INDEX single {S['index']:.3f} joint {J['index']:.3f} delta {S['index'] - J['index']:+.3f} (prefix, 31,198 rows, same node)")
print("AREAS single-joint: " + ", ".join(f"{a} {S['areas'][a] - J['areas'][a]:+.4f}" for a in S["areas"]))
want = ("sata", "acos", "bfcl", "contract", "toolret", "phish", "appliance", "api-bank", "apibank", "api_bank")
out = []
for b in sorted(S["bench"], key=lambda b: int(b) if b.isdigit() else 0):
    d = S["bench"][b] - J["bench"].get(b, float("nan")); nm = names.get(b, "?"); q = nq.get(b, [0]); out.append((b, nm, S["bench"][b], J["bench"].get(b), d, len(q), sum(q) / max(len(q), 1)))
print("NAMED BENCHMARKS (catalog id, name, single, joint, delta in points of that benchmark's score, rows in prefix, mean questions per row):")
for b, nm, s, j, d, n, mq in out:
    if any(w in nm.lower() for w in want): print(f"  {b:>3} {nm[:44]:44s} single {100 * s:6.2f} joint {100 * j:6.2f} delta {100 * d:+6.2f} rows {n} q/row {mq:.1f}")
print("ALL MULTI-QUESTION BENCHMARKS (mean questions per row > 1.05):")
for b, nm, s, j, d, n, mq in out:
    if mq > 1.05 and not any(w in nm.lower() for w in want): print(f"  {b:>3} {nm[:44]:44s} single {100 * s:6.2f} joint {100 * j:6.2f} delta {100 * d:+6.2f} rows {n} q/row {mq:.1f}")
one = [d for b, nm, s, j, d, n, mq in out if mq <= 1.05]
print(f"SINGLE-QUESTION BENCHMARKS: {len(one)}; max |delta| {100 * max(abs(x) for x in one):.2f} points (the two views are the same request there; this is run-to-run noise)")
lat = lambda f: sorted(json.loads(l).get("model_request_wall_ms", 0) for l in open(f))
a, b = lat(rs), lat(rj); print(f"LATENCY this GPU (ms) single median {a[len(a) // 2]:.0f} p80 {a[int(.8 * len(a))]:.0f} mean {sum(a) / len(a):.0f} | joint median {b[len(b) // 2]:.0f} p80 {b[int(.8 * len(b))]:.0f} mean {sum(b) / len(b):.0f}")
PY
aws s3 cp --only-show-errors $K/single/deltas.txt $O/deltas.txt; log "DELTAS $(head -2 $K/single/deltas.txt | tr '\n' ';' | cut -c1-400)"
FULLNOTE="no full run (single - joint < 0.3)"
if python3 -c "import sys; sys.exit(0 if float('$PS') - float('$PJ') >= 0.3 else 1)"; then log "single beats joint by >= 0.3 on the prefix: FULL PUBLIC run with view single now"
  bash $M/scripts/mx_kfull.sh $TS $X > $K/single/kfull_single.out 2>&1; rc=$?; FULLNOTE="full public run view single rc=$rc: $(grep -E '^HEADLINE' $M/work/$TS/kfull/merge/report.txt 2>/dev/null | cut -c1-200) (s3 .../esbar_$TS/kown_full/KFULL.txt)"; log "$FULLNOTE"; fi
echo "$(date -u +%FT%TZ) $(hostname) single $PS joint(same node) $PJ | $FULLNOTE | deltas s3 .../esbar_kl_single/deltas.txt" | aws s3 cp --only-show-errors - $F/DONE_single_view
echo "FREE: KL single-view read finished $(date -u +%FT%TZ)" > /tmp/claim.txt; log "KL_SINGLE_DONE"
