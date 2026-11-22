"""qv_arm_ug.sh = round F's qv_arm_uf.sh (md5 4ce4807c) with the four data-location edits a recipe_g manifest needs; nothing else changes."""
import hashlib, subprocess
from pathlib import Path
B = Path("/data/shared/visdata_g/base")
W = Path("/data/shared/visdata_g/launcher"); W.mkdir(parents=True, exist_ok=True)
src = (B / "qv_arm_uf.sh").read_text()
assert hashlib.md5(src.encode()).hexdigest() == "4ce4807ca21044925e8b0f9a46678c08"
s = src

def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (old, s.count(old))
    s = s.replace(old, new)

rep("E=$V/recipe_e; FF=$V/recipe_f  # round F (data fork): manifests uf1_* + their new image tars\n",
    "E=$V/recipe_e; FF=$V/recipe_f; FG=$V/recipe_g  # round F / G (data fork): manifests uf* / ug* + their new image tars\n")
rep("aws s3 cp --quiet $FF/manifests/$MAN.jsonl $Q/data/ && aws s3 cp --quiet $FF/manifests/$MAN.header.json $Q/data/;",
    "aws s3 cp --quiet $FG/manifests/$MAN.jsonl $Q/data/ && aws s3 cp --quiet $FG/manifests/$MAN.header.json $Q/data/;")
rep("aws s3 cp --quiet $FF/MD5SUMS $Q/data/MD5SUMS_f\n", "aws s3 cp --quiet $FF/MD5SUMS $Q/data/MD5SUMS_f && aws s3 cp --quiet $FG/MD5SUMS $Q/data/MD5SUMS_g\n")
rep('"$FF MD5SUMS_f"; do set -- $pm;', '"$FF MD5SUMS_f" "$FG MD5SUMS_g"; do set -- $pm;')
(W / "qv_arm_ug.sh").write_text(s)
print("md5", hashlib.md5(s.encode()).hexdigest())
print(subprocess.run(["bash", "-n", str(W / "qv_arm_ug.sh")], capture_output=True, text=True))
print(subprocess.run(["diff", str(B / "qv_arm_uf.sh"), str(W / "qv_arm_ug.sh")], capture_output=True, text=True).stdout[:4000])
