"""Round G arms ug1 (plain) and uog1 (OCR): chain, launcher, companions, MD5SUMS and the ctl guard, generated from the files uf2 / uof2 ran with.
usage: vg_genarm.py <name: ug1|uog1> <outdir>
Sources (md5 asserted): uf2_chain.sh 35edce68 (structure of both chains), qv_arm_uf.sh 4ce4807c -> qv_arm_ug.sh e74cd64c (plain launcher),
qv_arm_uof.sh 350c4a73 -> qv_arm_uog.sh (OCR launcher; same four recipe_g edits), uf2_{modelcheck,rows_check,result}.py, ctl's uf2_guard.sh 823e25f3.
What changes against uf2: names, manifest ug1_nc / sha / 5,051 steps / gates, state dir recipe_g/<name>, HELD + public reference = uf2 (ug1) or
uof2 (uog1) with the preregistered bands, a hold after three unfinished runs in a row; uog1 only: the OCR stage (every OCR file checksum-checked
on every start), the one-time OCR pass for the images recipe_f's cache lacks (identity gate >= 0.97), OCR=<cache> to the launcher, the smoke's
dev-OCR-flag clause, torchrun port 29633."""
import hashlib
import re
import sys
from pathlib import Path

name, outd = sys.argv[1], Path(sys.argv[2])
OPS = Path(__file__).resolve().parents[2]
ocr = name == "uog1"
N, NU = name, name.upper()
MAN, SHA, STEPS, ROWS = "ug1_nc", "65e7224c571fa5c814630229ca234e68c96ba4f78fd52796f3bae90dcfaad20b", 5051, 323264
REFRUN, REFCK, REFPUB = ("uof2", "uof2/ckpt_04519", 78.80) if ocr else ("uf2", "uf2/ckpt_04519", 78.98)
GAIN, LOW = round(REFPUB + 0.30, 2), round(REFPUB - 0.30, 2)
LAUNCHER = "qv_arm_uog.sh" if ocr else "qv_arm_ug.sh"
PORT = "29633" if ocr else "29641"
md5 = lambda s: hashlib.md5(s.encode()).hexdigest()  # noqa: E731


def rep(s, old, new, n=1):
    assert s.count(old) == n, (old[:80], s.count(old))
    return s.replace(old, new)


# ---------------------------------------------------------------- launcher
if ocr:
    L = (OPS / "forkA" / "uof2" / "qv_arm_uof.sh").read_text()
    assert md5(L).startswith("350c4a73")
else:
    L = (OPS / "recipe_g" / "arms" / "qv_arm_uf.sh").read_text()
    assert md5(L).startswith("4ce4807c")
L = rep(L, "E=$V/recipe_e; FF=$V/recipe_f  # round F (data fork): manifests uf1_* + their new image tars\n",
        "E=$V/recipe_e; FF=$V/recipe_f; FG=$V/recipe_g  # round F / G (data fork): manifests uf* / ug* + their new image tars\n")
L = rep(L, "aws s3 cp --quiet $FF/manifests/$MAN.jsonl $Q/data/ && aws s3 cp --quiet $FF/manifests/$MAN.header.json $Q/data/;",
        "aws s3 cp --quiet $FG/manifests/$MAN.jsonl $Q/data/ && aws s3 cp --quiet $FG/manifests/$MAN.header.json $Q/data/;")
L = rep(L, "aws s3 cp --quiet $FF/MD5SUMS $Q/data/MD5SUMS_f\n", "aws s3 cp --quiet $FF/MD5SUMS $Q/data/MD5SUMS_f && aws s3 cp --quiet $FG/MD5SUMS $Q/data/MD5SUMS_g\n")
L = rep(L, '"$FF MD5SUMS_f"; do set -- $pm;', '"$FF MD5SUMS_f" "$FG MD5SUMS_g"; do set -- $pm;')
if not ocr:
    assert md5(L) == "e74cd64c05b16793e84fc716268e8e9f", md5(L)  # the launcher delivered with round G

# ---------------------------------------------------------------- chain
C = (OPS / "recipe_f" / "uf2" / "uf2_chain.sh").read_text()
assert md5(C).startswith("35edce68")
gs = ",".join(str(x) for x in range(1000, STEPS - 499, 1000)) + f",{STEPS}"
C = rep(C, "MANIFEST=uf2_nc_ade; MSHA=df059f4d8aa72f000eabe808025ef89bfb9f3c94fc6a0e93aada174578df15d2; STEPS=4519; GSTEPS=1000,2000,3000,4000,4519",
        f"MANIFEST={MAN}; MSHA={SHA}; STEPS={STEPS}; GSTEPS={gs}")
C = rep(C, "U=$V/recipe_f/uf2", f"U=$V/recipe_g/{N}")
C = rep(C, "aws s3 sync --quiet $U/code/ $UW/code/ --exclude uf2_chain.sh", "aws s3 cp --quiet --recursive $U/code/ $UW/code/ --exclude uf2_chain.sh")  # sync kept a stale same-size MD5SUMS (11-15)
C = rep(C, "AR=$UW/code/qv_arm_uf.sh", f"AR=$UW/code/{LAUNCHER}")
C = rep(C, "$(md5sum < $UW/code/qv_arm_uf.sh | cut -c1-8)", f"$(md5sum < $UW/code/{LAUNCHER} | cut -c1-8)")
C = rep(C, "${TAG}__dataF-${NAME}__", "${TAG}__dataG-${NAME}__", 2)
C = rep(C, "data fork (round F): arm $NAME = UE2's recipe on uf2_nc_ade (289,200 rows, 4,519 steps)",
        f"data fork (round G): arm $NAME = {'uof2' if ocr else 'uf2'}'s recipe on {MAN} ({ROWS:,} rows, {STEPS:,} steps)")
C = rep(C, "node-side chain uf2_chain.sh, laptop guard ops/recipe_f/uf2/uf2_guard.sh", f"node-side chain {N}_chain.sh, guard on worker-node ops/recipe_g/{N}/{N}_guard.sh")
C = rep(C, "REF=ue2_noade/ckpt_03700", f"REF={REFCK}")
C = C.replace("PORT=29641", f"PORT={PORT}")
# hold after three unfinished runs in a row (uof2's rule)
C = rep(C, 'else log "ARM_NOT_FINISHED (a re-run of this chain resumes it)"; fi',
        'else log "ARM_NOT_FINISHED (a re-run of this chain resumes it) :: $(grep -E "Error|Traceback" $Q/logs/train_$NAME.out 2>/dev/null | tail -1 | cut -c1-200)"\n'
        '  [ "$(tail -n 40 $L | grep -c ARM_NOT_FINISHED)" -ge 3 ] && { echo "three unfinished arm runs in a row $(date -u +%FT%TZ) on $TAG" | aws s3 cp --quiet - $U/state/${NAME}_fail; log "HELD after 3 unfinished runs in a row: ${NAME}_fail written"; }\nfi')
C = rep(C, "  ( bash $Q/code/qv_env.sh; log \"ENV $(tail -2 $Q/logs/env.log | tr '\\n' ' ')\" ) &\n",
        "  L0=$(wc -l < $L); E0=$(wc -l < $Q/logs/env.log 2>/dev/null || echo 0)  # this start's log lines only: an earlier failed restage on the same disk is not held against this one\n"
        "  ( bash $Q/code/qv_env.sh; cc1d_fix; log \"ENV $(tail -2 $Q/logs/env.log | tr '\\n' ' ')\" ) &\n")
C = rep(C, '  bad=$(cat $L $Q/logs/env.log | grep -c -E "_BAD|ENV_FUNCTIONAL_FAIL|MODEL_FAIL")\n',
        '  bad=$( (tail -n +$((L0 + 1)) $L; tail -n +$((E0 + 1)) $Q/logs/env.log) | grep -c -E "_BAD|ENV_FUNCTIONAL_FAIL|MODEL_FAIL")\n')
C = rep(C, "if [ ! -f $Q/UF2_RESTAGE_OK ]; then\n", (Path(__file__).parent / "cc1d_fix.sh.txt").read_text() + "if [ ! -f $Q/UF2_RESTAGE_OK ]; then\n")
if ocr:
    OCRB = r'''# --- OCR (uog1). Every OCR file is fetched and checksum-checked on EVERY chain start (a replaced spot worker has an empty disk).
FO=$V/recipe_f/ocr; GO=$V/recipe_g/ocr; OCRF=ocr/ocr_train_ug1.jsonl
ocr_get(){ # <s3 url> <local path> <md5>: keep a local file only if its md5 is right, else fetch and check
  [ -n "$3" ] || return 1
  [ -f $2 ] && [ "$(md5sum < $2 | cut -d' ' -f1)" = "$3" ] && return 0
  aws s3 cp --quiet $1 $2.dl && [ "$(md5sum < $2.dl | cut -d' ' -f1)" = "$3" ] && mv $2.dl $2 || { log "OCR_STAGE_BAD $(basename $2)"; return 1; }
}
ocr_stage(){
  mkdir -p $Q/ocr $Q/rbh_gate $Q/priv
  ocr_get $A/ocr/ocr_dev.jsonl $Q/ocr/ocr_dev.jsonl 5b62c8a22424c56bae8f3cfb0375cbe9 && ocr_get $A/ocr/ocr_public_EVAL_ONLY.jsonl $Q/ocr/ocr_public.jsonl 212232c925a9f94c9b4c6b0506d73db9 \
    && ocr_get $FO/ocr_train_uf1.jsonl $Q/ocr/ocr_train_uf1.jsonl 1232cbfffd9809ed9abaaa9376c93b7f && ocr_get $V/recipe_privlike/ocr/ocr_rbh.jsonl $Q/rbh_gate/ocr_rbh.jsonl ff5023125c676eed1f30524afa4c0e32 \
    && ocr_get $V/recipe_privlike/ocr/ocr_priv.jsonl $Q/priv/ocr_priv.jsonl 98776a8972ec6d019423092a9e73946b
}
ocr_stage && log "OCR_STAGE_OK dev / public / round F train cache / rbh / priv present, md5 ok" || { log "OCR_STAGE_FAIL: not starting the arm"; mirror; exit 4; }
stage_images(){ env MAN=$MANIFEST SHA=$MSHA STAGE_ONLY=1 bash $AR _${NAME}_stage 0 0 && tail -n 1 $Q/logs/arm__${NAME}_stage.log | grep -q "STAGE_ONLY: inputs staged"; }
if aws s3 ls $GO/MD5SUMS_ocr_ug1 > /dev/null 2>&1; then
  aws s3 cp --quiet $GO/MD5SUMS_ocr_ug1 $Q/ocr/MD5SUMS_ocr_ug1 && ocr_get $GO/ocr_train_ug1.jsonl $Q/$OCRF "$(grep ' ocr_train_ug1.jsonl$' $Q/ocr/MD5SUMS_ocr_ug1 | cut -d' ' -f1)" \
    && log "OCR_CACHE_FROM_S3 $(wc -l < $Q/$OCRF) images" || { log OCR_CACHE_S3_BAD; mirror; exit 4; }
else
  # one-time OCR pass: PP-OCRv5 mobile det + en mobile rec, Arm UO's ocr_pass.py unchanged, Arm UO's PINNED model files; images the round F cache lacks
  stage_images || { log "IMAGE_STAGE_FAIL $(tail -2 $Q/logs/arm__${NAME}_stage.log | tr '\n' ' ' | cut -c1-300)"; mirror; exit 5; }
  log "IMAGES_STAGED for the OCR pass"
  aws s3 cp --quiet $A/code/qv_extra/ocr_pass.py $UW/code/ocr_pass.py && aws s3 cp --quiet $A/code/qv_extra/ocr_merge.py $UW/code/ocr_merge.py || { log OCR_CODE_FAIL; mirror; exit 3; }
  OL=$Q/logs/ocr_ug1.log; OD=$Q/ocr/ug1; mkdir -p $OD/lists $OD/logs
  if [ ! -f $Q/ocrvenv.ok ]; then
    python3 -m venv $Q/ocrvenv && $Q/ocrvenv/bin/pip install -q --upgrade pip >> $OL 2>&1
    $Q/ocrvenv/bin/pip install -q paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ >> $OL 2>&1 || { log PADDLE_GPU_INSTALL_FAIL; mirror; exit 2; }
    sudo -n env DEBIAN_FRONTEND=noninteractive apt-get update -q >> $OL 2>&1; sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -q libgl1 libglib2.0-0 >> $OL 2>&1
    $Q/ocrvenv/bin/pip install -q "paddleocr>=3.1" "paddlex[ocr]==3.7.2" >> $OL 2>&1 && touch $Q/ocrvenv.ok || { log OCR_PIP_FAIL; mirror; exit 2; }
  fi
  [ -e /usr/lib/x86_64-linux-gnu/libGL.so.1 ] || { sudo -n env DEBIAN_FRONTEND=noninteractive apt-get update -q >> $OL 2>&1; sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -q libgl1 libglib2.0-0 >> $OL 2>&1; }
  $Q/ocrvenv/bin/pip freeze 2>/dev/null | grep -iE "^(paddle|opencv|numpy|shapely|pyclipper)" > $Q/ocr/pip_freeze_ug1.txt
  MD=$HOME/.paddlex/official_models; mkdir -p $MD
  aws s3 cp --quiet $A/ocr/ocr_models.tar $Q/ocr/ocr_models.tar && aws s3 cp --quiet $A/ocr/ocr_models.md5 $Q/ocr/ocr_models.md5 && tar xf $Q/ocr/ocr_models.tar -C $MD && (cd $MD && md5sum -c --quiet $Q/ocr/ocr_models.md5) || { log OCR_MODELS_PIN_FAIL; mirror; exit 4; }
  log "OCR_ENV $(grep -iE '^paddle' $Q/ocr/pip_freeze_ug1.txt | tr '\n' ' ') models pinned $(wc -l < $Q/ocr/ocr_models.md5) files; gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  python3 - <<'PY' >> $L 2>&1 || { log OCR_LISTS_FAIL; mirror; exit 6; }
import json, random
Q = "/data/local/qv"
need = set()
for l in open(f"{Q}/data/ug1_nc.jsonl"):
    need.update(json.loads(l)["images"])
have = {json.loads(l)["image"] for l in open(f"{Q}/ocr/ocr_train_uf1.jsonl")}
new, old = sorted(need - have), sorted(need & have)
open(f"{Q}/ocr/ug1/lists/all.txt", "w").write("\n".join(sorted(need)) + "\n")
open(f"{Q}/ocr/ug1/lists/new.txt", "w").write("\n".join(new) + "\n")
random.seed(20261115)
open(f"{Q}/ocr/ug1/lists/ident.txt", "w").write("\n".join(sorted(random.sample(old, 400))) + "\n")
print(f"[uog1 ocr] images in ug1_nc {len(need)}; already in the round F cache {len(old)}; new {len(new)}")
PY
  # identity gate: 400 images that already have OCR text, read again on this node, must give the same text on >= 97%
  for k in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$k $Q/ocrvenv/bin/python $UW/code/ocr_pass.py $OD/lists/ident.txt $Q/data/img $OD/ident_$k.jsonl $k 8 gpu:0 > $OD/logs/ident_$k.log 2>&1 & done; wait
  SAME=$(python3 - <<'PY'
import glob, json
Q = "/data/local/qv/ocr"
def norm(r):
    ls = [x for x in r["lines"] if x["score"] >= 0.5]
    ls.sort(key=lambda x: (round(x["box"][1] / 12), x["box"][0]))
    return [x["text"] for x in ls]
want = {l.strip() for l in open(f"{Q}/ug1/lists/ident.txt") if l.strip()}
old = {}
for l in open(f"{Q}/ocr_train_uf1.jsonl"):
    r = json.loads(l)
    if r["image"] in want:
        old[r["image"]] = norm(r)
new = {}
for f in glob.glob(f"{Q}/ug1/ident_*.jsonl"):
    for l in open(f):
        r = json.loads(l); new[r["image"]] = norm(r)
same = sum(new[k] == old[k] for k in new)
print(f"{same}/{len(new)} {same / max(1, len(want)):.4f}")
PY
)
  log "OCR_IDENTITY same_text $SAME (denominator: the 400 listed images)"
  python3 -c "import sys; sys.exit(0 if float('$(echo $SAME | awk '{print $2}')') >= 0.97 else 1)" || { log "OCR_IDENTITY_BELOW_0.97: stop for inspection"; echo "ocr identity gate $SAME $(date -u +%FT%TZ) on $TAG" | aws s3 cp --quiet - $U/state/${NAME}_fail; aws s3 cp --quiet --recursive $OD/logs $GO/ident_fail_logs_$TAG/ 2>/dev/null; mirror; exit 5; }
  NNEW=$(grep -c . $OD/lists/new.txt)
  PIDS=""; for k in $(seq 0 23); do CUDA_VISIBLE_DEVICES=$((k % 8)) $Q/ocrvenv/bin/python $UW/code/ocr_pass.py $OD/lists/new.txt $Q/data/img $OD/shard_$k.jsonl $k 24 gpu:0 > $OD/logs/new_$k.log 2>&1 & PIDS="$PIDS $!"; done
  bad=0; for j in $PIDS; do wait $j || bad=$((bad + 1)); done
  [ $bad = 0 ] || { log "OCR_SHARD_FAIL $bad: $(tail -2 $OD/logs/new_0.log | tr '\n' ' ' | cut -c1-300)"; mirror; exit 6; }
  python3 $UW/code/ocr_merge.py $OD/lists/new.txt $OD $Q/ocr/ocr_train_ug1_new.jsonl > $OD/merge_new.json 2>&1 || { log "OCR_MERGE_FAIL $(cut -c1-300 $OD/merge_new.json)"; mirror; exit 6; }
  log "OCR_NEW_OK $NNEW images: $(cat $OD/merge_new.json)"
  python3 - <<'PY' > $Q/ocr/ug1/merge_all.json 2>&1 || { log "OCR_FINAL_MERGE_FAIL $(cut -c1-300 $Q/ocr/ug1/merge_all.json)"; mirror; exit 6; }
import json
Q = "/data/local/qv/ocr"
want = [l.rstrip("\n") for l in open(f"{Q}/ug1/lists/all.txt") if l.strip()]
got = {}
for f in ("ocr_train_uf1.jsonl", "ocr_train_ug1_new.jsonl"):  # the new pass covers only images the round F cache lacks; old lines verbatim
    for l in open(f"{Q}/{f}"):
        got[json.loads(l)["image"]] = l if l.endswith("\n") else l + "\n"
miss = [p for p in want if p not in got]
assert not miss, ("missing", len(miss), miss[:3])
with open(f"{Q}/ocr_train_ug1.jsonl.tmp", "w") as f:
    for p in want:
        f.write(got[p])
print(json.dumps({"images": len(want)}))
PY
  mv $Q/ocr/ocr_train_ug1.jsonl.tmp $Q/$OCRF
  (cd $Q/ocr && md5sum ocr_train_ug1.jsonl ocr_train_ug1_new.jsonl > MD5SUMS_ocr_ug1)
  ok=0; for f in ocr_train_ug1.jsonl ocr_train_ug1_new.jsonl pip_freeze_ug1.txt; do aws s3 cp --quiet $Q/ocr/$f $GO/$f && [ "$(md5sum < $Q/ocr/$f)" = "$(aws s3 cp --quiet $GO/$f - | md5sum)" ] && ok=$((ok+1)); done
  aws s3 sync --quiet $OD/lists $GO/lists/; aws s3 cp --quiet $OD/merge_new.json $GO/merge_new.json; aws s3 cp --quiet $OL $GO/ocr_ug1.log 2>/dev/null
  echo "$SAME identity re-read on $TAG ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)), $(date -u +%FT%TZ)" | aws s3 cp --quiet - $GO/identity_ug1.txt
  [ $ok = 3 ] && aws s3 cp --quiet $Q/ocr/MD5SUMS_ocr_ug1 $GO/MD5SUMS_ocr_ug1 && [ "$(md5sum < $Q/ocr/MD5SUMS_ocr_ug1)" = "$(aws s3 cp --quiet $GO/MD5SUMS_ocr_ug1 - | md5sum)" ] \
    && log "OCR_CACHE_BUILT $(cat $OD/merge_all.json) S3_OK 4/4 $GO/" || { log "OCR_CACHE_S3_BAD $ok/3"; mirror; exit 7; }
fi
stage_images || { log "IMAGE_STAGE_FAIL $(tail -2 $Q/logs/arm__${NAME}_stage.log | tr '\n' ' ' | cut -c1-300)"; mirror; exit 5; }
python3 - <<'PY' >> $L 2>&1 || { log "OCR_COVERAGE_FAIL: not starting the arm"; mirror; exit 4; }
import json
Q = "/data/local/qv"
have = {json.loads(l)["image"] for l in open(f"{Q}/ocr/ocr_train_ug1.jsonl")}
need = set()
for l in open(f"{Q}/data/ug1_nc.jsonl"):
    need.update(json.loads(l)["images"])
print(f"[uog1 ocr] manifest images {len(need)}, missing from the OCR cache {len(need - have)}")
assert not (need - have)
PY
'''
    a = C.index('UA="MAN=$MANIFEST SHA=$MSHA')
    C = C[:a] + OCRB + C[a:]
    C = rep(C, 'UA="MAN=$MANIFEST SHA=$MSHA LORA_R=16', 'UA="MAN=$MANIFEST SHA=$MSHA OCR=$OCRF LORA_R=16')
    C = rep(C, "  b8=$(ls -d $Q/eval_restricted/wg_suite 2>/dev/null | wc -l)\n",
            "  b8=$(ls -d $Q/eval_restricted/wg_suite 2>/dev/null | wc -l)\n"
            "  b9=$(python3 -c \"import json; print(int(json.load(open('$Q/runs/$N/gate_00006.json')).get('ocr') is True))\" 2>/dev/null)\n")
    C = rep(C, 'winoground_dirs=$b8 peak_train_mib', 'winoground_dirs=$b8 dev_ocr_flag=${b9:-0} peak_train_mib')
    C = rep(C, '[ "$b7" = "1" ] && [ "$b8" = "0" ]; then', '[ "$b7" = "1" ] && [ "$b8" = "0" ] && [ "${b9:-0}" = "1" ]; then')
C = C.replace("UF2_RESTAGE_OK", f"{NU}_RESTAGE_OK").replace("uf2_chain", f"{N}_chain").replace("uf2_modelcheck", f"{N}_modelcheck") \
     .replace("uf2_rows_check", f"{N}_rows_check").replace("uf2_result", f"{N}_result").replace("uf2_smoke_mem", f"{N}_smoke_mem") \
     .replace("/data/local/uf2", f"/data/local/{N}").replace("[uf2 ", f"[{N} ").replace("another uf2 chain", f"another {N} chain") \
     .replace("uf_dev65", f"{N}_dev65")
lines = C.split("\n")
i = next(k for k, l in enumerate(lines) if l.startswith("# usage:"))
head = [f"# data fork (round G): arm `{N}` = {'uof2' if ocr else 'uf2'}'s recipe ({'PP-OCRv5 block in every prompt' if ocr else 'plain, no OCR block'}) on manifest {MAN} ({ROWS:,} rows; {STEPS:,} steps).",
        "# Node-side chain, re-runnable, generated by vg_genarm.py from round F's chain (md5 35edce68): restage from S3 (18 base-model shards checked against",
        "# the Hub's sha256" + ("; every OCR file checksum-checked on every start; one-time OCR pass for the images round F's cache lacks" if ocr else "") + ") -> smoke once -> train with S3",
        f"# checkpoints every 100 steps (resumes) -> dev gates {gs.replace(',', ' / ')} (stop only on collapse, dev < 85) -> final dev >= 89.93 or no public read ->",
        f"# RBH two numbers (reported) -> proxy HELD with the {REFRUN} reference ({REFCK}) in the same processes -> ONE public no-WG read. No soup sweep.",
        "# No Winoground on this node (WG=0: the launcher never touches eval_restricted). Prereg: AGENT_NOTES 2026-11-15 (PREREG Arms ug1 + uog1).",
        f"# usage: {N}_chain.sh <node tag: p12 | p4de> <run name: {N}>"]
C = "\n".join([lines[0]] + head + lines[i + 1:])
left = [m.group(0) for m in re.finditer(r"(?i)u\[?f\]?2\b(?!/ckpt|'s| reference)|uf1(?!\.jsonl|_| cache)|recipe_f/uf|qv_arm_uf\b|4519(?!\b.{0,3}$)|df059f4d|289,200|dataF|ue2_noade", C)]
left = [x for x in left if x]
# ---------------------------------------------------------------- companions
src = OPS / "recipe_f" / "uf2"
mc = (src / "uf2_modelcheck.py").read_text().replace("uf2", N)
rc = (src / "uf2_rows_check.py").read_text().replace("uf2", N)
R = (src / "uf2_result.py").read_text()
R = rep(R, 'Q = os.environ.get("UF2_Q", "/data/local/qv")', f'Q = os.environ.get("{NU}_Q", "/data/local/qv")')
R = rep(R, '"reference": "ue2_noade/ckpt_03700 read on this node in the same processes"', f'"reference": "{REFCK} read on this node in the same processes"')
R = rep(R, 'f"{A}/runs/ue2_noade/public_read/full_index.json"', f'f"{{A}}/runs/{REFRUN}/public_read/full_index.json"')
R = rep(R, '"ue2_index_reference": 78.15', f'"ue2_index_reference": {REFPUB}')
R = rep(R, '"verdict_by_prereg": "GAIN (>= 78.45)" if x >= 78.45 else ("NO CHANGE (77.85 .. 78.45)" if x >= 77.85 else "WORSE (< 77.85)")',
        f'"verdict_by_prereg": "GAIN (>= {GAIN})" if x >= {GAIN} else ("NO CHANGE ({LOW} .. {GAIN})" if x >= {LOW} else "WORSE (< {LOW})")')
R = R.replace("the UE2 reference", f"the {REFRUN} reference").replace("against UE2's stored one", f"against {REFRUN}'s stored one").replace("vs UE2's stored read", f"vs {REFRUN}'s stored read") \
     .replace("no new read of UE2", f"no new read of {REFRUN}").replace("ue2", "ref").replace("uf2", N)
left += [f"result:{m.group(0)}" for m in re.finditer(r"(?i)uf2|ue2|78\.15|78\.45|77\.85", R)] + [f"mc:{m.group(0)}" for m in re.finditer(r"(?i)uf2", mc + rc)]
# ---------------------------------------------------------------- guard (runs on worker-node; S3 markers read locally)
Gd = (OPS / "recipe_g" / "arms" / "ctl_tpl" / "uf2_guard.sh").read_text()
assert md5(Gd).startswith("823e25f3")
a = Gd.index("unset ASCALE_RUN_ID; cd")
ghead = (f"#!/bin/bash\n# data fork (round G): the ONE guard for arm {N}, on worker-node (/workspace/jev_guards/ops/recipe_g/{N}/{N}_guard.sh; line in loops.ctl.txt).\n"
         f"# usage: {N}_guard.sh   with the node in ./{N}_node.txt = \"<workspace id> <tag>\" (one line). Generated by vg_genarm.py from uf2's cloud guard (823e25f3). Every ~2 min:\n"
         f"#   1. \"finished\" comes ONLY from the S3 marker recipe_g/{N}/state/{N}_chain_done (read here, never from the training node's disk) -> exit.\n"
         f"#      A fail marker or 3 failed restages -> stop relaunching, keep watching.\n"
         "#   2. workspace TERMINATED (spot loss / the 08:05Z pass) -> `ascale start --yes`, only after a marker read that SUCCEEDED and said \"not done\".\n"
         f"#   3. an alive 8-GPU worker with no {N} chain on it -> launch {N}_chain.sh from S3 (it restages, checks every checksum, resumes from S3).\n"
         "#   4. webterminal keepalive about every 30 min. It kills nothing, ever. It keeps no state of its own. A second copy exits at once (flock in start_all.sh too).\n")
Gd = ghead + Gd[a:]
Gd = rep(Gd, "WS=WORKSPACE_ID; CTL=WORKSPACE_ID; TAG=p12; NAME=uf2",
         f"read -r WS TAG < {N}_node.txt; NAME={N}; [ -n \"$WS\" ] && [ -n \"$TAG\" ] || {{ echo \"no {N}_node.txt\" >> $L; exit 1; }}")
Gd = rep(Gd, "U=s3://YOUR_BUCKET/jev-race/engineering/vision/recipe_f/uf2", f"U=s3://YOUR_BUCKET/jev-race/engineering/vision/recipe_g/{N}")
Gd = rep(Gd, "pgrep -fc '[u]f2_chain.sh'", f"pgrep -fc '[{N[0]}]{N[1:]}_chain.sh'")
Gd = rep(Gd, "--detach --name uf2_${NAME}_$(date -u +%d%H%M)", f"--detach --name {N}_chain_$(date -u +%d%H%M%S)")
Gd = Gd.replace("/data/local/uf2/", f"/data/local/{N}/").replace("uf2_chain", f"{N}_chain").replace("uf2_guard", f"{N}_guard").replace("uf2-guard", f"{N}-guard") \
       .replace("(ctl2 unreachable?)", "(S3 read failed)").replace("# -> \"MK done=<0|1> fail=<0|1> rf=<n>\" from S3 via ctl2", "# -> \"MK done=<0|1> fail=<0|1> rf=<n>\" from S3")
Gd = rep(Gd, "\\\"MK done=", "\\\"MK st=\\$(echo \\\"\\$S\\\" | grep -c ' ${NAME}_started\\$') done=")
Gd = rep(Gd, 'log "armed pid=$$"; n=0; held=0\n', (Path(__file__).parent / "guard_gate.sh.txt").read_text())
Gd = rep(Gd, '  if echo "$O" | grep -q "gpus=8" && echo "$O" | grep -q "chain=0 "; then\n',
         '  if echo "$O" | grep -q "gpus=8" && echo "$O" | grep -q "chain=0 "; then\n    if ! mine; then sleep 110; continue; fi\n')
left += [f"guard:{m.group(0)}" for m in re.finditer(r"(?i)u\[?f\]?2|recipe_f|WORKSPACE_ID|CTL=", Gd)]
outd.mkdir(parents=True, exist_ok=True)
files = {LAUNCHER: L, f"{N}_chain.sh": C, f"{N}_modelcheck.py": mc, f"{N}_rows_check.py": rc, f"{N}_result.py": R}
for f, s in files.items():
    (outd / f).write_text(s)
(outd / "MD5SUMS").write_text("".join(f"{md5(s)}  {f}\n" for f, s in files.items()))
(outd / f"{N}_guard.sh").write_text(Gd)
print(N, {f: md5(s)[:8] for f, s in files.items()}, "guard", md5(Gd)[:8], "leftovers", left)
