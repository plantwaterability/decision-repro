"""ug1 / ug1c chains generated from round F's uf2_chain.sh (35edce68) by exact replacements, then a leftover check (old names in any case and
with the bracket trick). usage: vg_genchain.py <uf2_chain.sh> <name> <manifest> <rows_sha256> <steps> <rows> <outdir>   (NOT run on a GPU node)"""
import hashlib, re, sys
from pathlib import Path
src, name, man, sha, steps, rows, outd = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), Path(sys.argv[7])
s = Path(src).read_text()
assert hashlib.md5(s.encode()).hexdigest().startswith("35edce68"), "not uf2_chain.sh 35edce68"
N, NU = name, name.upper()
gs = ",".join(str(x) for x in range(1000, steps - 499, 1000)) + f",{steps}"  # no thousand-step gate within 500 steps of the end
reps = [
    ("MANIFEST=uf2_nc_ade; MSHA=df059f4d8aa72f000eabe808025ef89bfb9f3c94fc6a0e93aada174578df15d2; STEPS=4519; GSTEPS=1000,2000,3000,4000,4519",
     f"MANIFEST={man}; MSHA={sha}; STEPS={steps}; GSTEPS={gs}"),
    ("U=$V/recipe_f/uf2", f"U=$V/recipe_g/{N}"),
    ("AR=$UW/code/qv_arm_uf.sh", "AR=$UW/code/qv_arm_ug.sh"),
    ("$(md5sum < $UW/code/qv_arm_uf.sh | cut -c1-8)", "$(md5sum < $UW/code/qv_arm_ug.sh | cut -c1-8)"),
    ("${TAG}__dataF-${NAME}__", "${TAG}__dataG-${NAME}__"),
    ("data fork (round F): arm $NAME = UE2's recipe on uf2_nc_ade (289,200 rows, 4,519 steps)",
     f"data fork (round G): arm $NAME = UE2's recipe on {man} ({rows:,} rows, {steps:,} steps)"),
    ("node-side chain uf2_chain.sh, laptop guard ops/recipe_f/uf2/uf2_guard.sh", f"node-side chain {N}_chain.sh (no guard written by the data fork)"),
    ("# usage: uf2_chain.sh <node tag: p12> <run name: uf2>", f"# usage: {N}_chain.sh <node tag> <run name: {N}>"),
]
for o, n in reps:
    assert s.count(o) >= 1, o
    s = s.replace(o, n)
s = s.replace("UF2_RESTAGE_OK", f"{NU}_RESTAGE_OK").replace("uf2_chain", f"{N}_chain").replace("uf2_modelcheck", f"{N}_modelcheck") \
     .replace("uf2_rows_check", f"{N}_rows_check").replace("uf2_result", f"{N}_result").replace("uf2_smoke_mem", f"{N}_smoke_mem") \
     .replace("/data/local/uf2", f"/data/local/{N}").replace("[uf2 ", f"[{N} ").replace("another uf2 chain", f"another {N} chain")
head = (f"# data fork (round G): plain arm `{N}` = UE2's recipe on manifest {man} ({rows:,} rows; {steps:,} steps at 64 rows). GENERATED from round F's\n"
        f"# uf2_chain.sh (35edce68) by vg_genchain.py: names, manifest / sha / steps / gates, recipe_g state dir, launcher qv_arm_ug.sh. NOT RUN ANYWHERE.\n"
        f"# The HELD reference (REF=ue2_noade/ckpt_03700 below, inherited) and the result script's reference numbers are the arm owner's call.\n")
lines = s.split("\n")
i = next(k for k, l in enumerate(lines) if l.startswith("# usage:"))
lines = [lines[0]] + head.rstrip("\n").split("\n") + [l for l in lines[i:]]
s = "\n".join(lines)
left = [m.group(0) for m in re.finditer(r"(?i)u\[?f\]?2|uf1|recipe_f/|qv_arm_uf|4519|4,519|df059f4d|289,200|dataF", s)]
outd.mkdir(parents=True, exist_ok=True)
(outd / f"{N}_chain.sh").write_text(s)
for comp in ("modelcheck", "rows_check", "result"):  # the chain's three companions: uf2's files with the name changed (contents otherwise unchanged)
    c = (Path(src).parent / f"uf2_{comp}.py").read_text().replace("uf2", N)
    (outd / f"{N}_{comp}.py").write_text(c)
    left += [f"{comp}:{m.group(0)}" for m in re.finditer(r"(?i)u\[?f\]?2|uf1|recipe_f/", c)]
print(N, "md5", hashlib.md5(s.encode()).hexdigest(), "leftovers", left)
