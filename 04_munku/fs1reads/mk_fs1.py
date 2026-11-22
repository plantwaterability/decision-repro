#!/usr/bin/env python3
"""Builds fs1_build.sh and fs1_gate.sh from the v8 read chain (ops/v8reads): same steps, the weights source is fork E's FS1 leg-1 export
(esbar_fs1/export_s2400), outputs under esbar_fs1/public_di_s2400, arm FS1S. The gate part is replaced by two measurements: the fixed
public prefix (scored vs E1's stored rows) and the pooled held-out diagnostic. No stop / continue rule, no full read."""
import re

SUB = [("v8fpub", "fs1pub"), ("v8f_", "fs1_"), ("_v8f", "_fs1"), ("esbar_v8/export_full", "esbar_fs1/export_s2400"),
       ("esbar_v8/public_di_full", "esbar_fs1/public_di_s2400"), ("esbar_v8:full", "esbar_fs1:s2400"), ("export_full", "export_s2400"),
       ("V8F", "FS1S"), ("v8 full", "FS1 s2400"), ("[v8f ", "[fs1 "), ("[gateFS1S ", "[readFS1S "), ("v8full_20261110", "fs1s2400_20261112"),
       ("fs1_code_r2.tgz", "fs1_code.tgz")]


def sub(s):
    for a, b in SUB:
        s = s.replace(a, b)
    return s


# ---- build
b = open("../v8reads/v8f_build.sh").read().split("\n")
i = next(k for k, l in enumerate(b) if l.startswith("set -u"))
head = ["#!/bin/bash",
        "# FS1 step-2,400 export + head for the prefix read (TB1 fork, 2026-11-12; prereg AGENT_NOTES \"PREREG FS1-S2400-PREFIX\") = v8f_build.sh",
        "# with the weights source esbar_fs1/export_s2400 (fork E's from-stock pilot, leg 1) on p8: venv, pull + md5-verify the export against its",
        "# READBACK_MD5.txt, head refit with the v7.1 stage-2 procedure, head -> S3 with read-back, then exec fs1_gate.sh. Idempotent steps."]
out = sub("\n".join(head + b[i:]))
assert "exec bash $W/fs1_gate.sh" in out and "v8" not in out.replace("stage1_v8", "").replace("fullft_train v8", ""), [l for l in out.split("\n") if "v8" in l][:5]
open("fs1_build.sh", "w").write(out)

# ---- gate -> measurement driver
g = open("../v8reads/v8f_gate_r4.sh").read().split("\n")
i = next(k for k, l in enumerate(g) if l.startswith("set -u"))
j = next(k for k, l in enumerate(g) if l.startswith("# ---- G1 harm screen"))
head = ["#!/bin/bash",
        "# FS1 step-2,400 MEASUREMENT on the fixed public prefix (prereg AGENT_NOTES \"PREREG FS1-S2400-PREFIX\"; it decides nothing by itself).",
        "# Steps 1-6 = v8f_gate_r4.sh (code, export + refit head, bundle FS1S = E1's package, public root, inputs, runset with the newline-only",
        "# splitter). Then: M1 the fixed 31,198-row prefix, answered-rows index vs E1's stored rows; M2 pooled ho8 + hoL + hoF (diagnostic).",
        "# Every chunk -> S3 when it finishes; a relaunch skips what is on S3. No stop / continue rule and no full read.",
        "#   CODEMD5=<md5 of fs1_code.tgz> FS1S_HM=<md5 of the refit vbll_head.json> PREFIX_SHA=<ids sha256> bash fs1_gate.sh"]
tail = r'''# ---- M1 the fixed public prefix (errors must be 0)
runset prefix $G/prefix/prefix_rows.jsonl.gz 24 1
[ "$(wc -l < $G/prefix_results.jsonl)" = "$(wc -l < $G/prefix/prefix_ids.txt)" ] || die "PREFIX_ROWS $(wc -l < $G/prefix_results.jsonl)"
PYTHONPATH=$KIT python3 $GC/gate_index.py $S03 $G/prefix/prefix_ids.txt $G/score/prefix $G/prefix_compare.json E1=$G/src/e1_public_results.jsonl.gz FS1S=$G/prefix_results.jsonl > $W/logs/gate_prefix_score.txt 2>&1 || { tail -5 $W/logs/gate_prefix_score.txt; die SCORE_PREFIX; }
grep -E "^INDEX|^DIFF" $W/logs/gate_prefix_score.txt | cut -c1-200 | tee -a $L; put $G/prefix_compare.json $OG/prefix_compare.json || die S3_PREFIX
gzip -n -c $G/prefix_results.jsonl > $G/prefix_results.jsonl.gz; put $G/prefix_results.jsonl.gz $OG/prefix_results.jsonl.gz || die S3_PREFIX_RESULTS
D3=$(num $G/prefix_compare.json diff_vs_E1 FS1S)
ver "M1 prefix MEASUREMENT (decides nothing): FS1S(step 2,400) - E1 = $D3 on $(wc -l < $G/prefix/prefix_ids.txt) rows; FS1S $(num $G/prefix_compare.json arms FS1S index) vs E1 $(num $G/prefix_compare.json arms E1 index)"
# ---- M2 pooled non-public diagnostic
runset ho $G/src/ho_rows.jsonl.gz 8 0
python3 $GC/gate_ho.py $G/rec/eval_v5,$G/rec/eval_pools_v5 $G/ho_compare.json ho8=$G/src/ho8_manifest.json,$G/src/base_ho8.jsonl,$G/ho_results.jsonl,$G/src/ho8_rows.jsonl.gz \
  hoL=$G/src/hoL_manifest.json,$G/src/base_hoL.jsonl,$G/ho_results.jsonl,$G/src/hoL_rows.jsonl.gz hoF=$G/src/hoF_manifest.json,$G/src/base_hoF.jsonl,$G/ho_results.jsonl,$G/src/hoF_rows.jsonl.gz > $W/logs/gate_ho.txt 2>&1 || { tail -5 $W/logs/gate_ho.txt; die SCORE_HO; }
grep -E "^SET|^POOLED" $W/logs/gate_ho.txt | cut -c1-260 | tee -a $L; put $G/ho_compare.json $OG/ho_compare.json || die S3_HO
ver "M2 pooled ho8+hoL+hoF DIAGNOSTIC: FS1S - E1 = $(num $G/ho_compare.json pooled paired 0) [$(num $G/ho_compare.json pooled paired 1), $(num $G/ho_compare.json pooled paired 2)]"
aws s3 cp --only-show-errors $W/logs/launch.out $OG/logs/launch.out
log "READS_DONE"'''
out = sub("\n".join(head + g[i:j])) + "\n" + tail + "\n"
assert "fetch $O/code/fs1_code.tgz $W/src/fs1_code.tgz ${CODEMD5:?}" in out, [l for l in out.split("\n") if "code" in l and "fetch" in l]
assert "phaseFS1S_bundle.sh" in out and "bundle_FS1S/eshbar_FS1S.tgz" in out and 'arm=FS1S ' in out
left = [l for l in out.split("\n") if re.search(r"v8|V8", l) and "v8f_gate_r4" not in l]
assert not left, left[:5]
open("fs1_gate.sh", "w").write(out)
print("ok")
