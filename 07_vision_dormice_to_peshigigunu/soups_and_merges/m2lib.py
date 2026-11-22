"""Small helpers shared by the m2rel node scripts (no torch import here)."""
import hashlib
import json
import os
import shlex
from concurrent.futures import ThreadPoolExecutor


def read_env(path):
    """KEY=value lines of a shell env file (merge.env / release.env); values unquoted with shlex, comments skipped."""
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = " ".join(shlex.split(v)) if v.strip() else ""
    return out


def spec_members(env):
    """[(label, kind, coef, src, hub)] in spec order."""
    ms = []
    for lab in env["MEMBERS"].split():
        kind = env[f"M_{lab}_KIND"]
        assert kind in ("full", "lowrank", "peft"), (lab, kind)
        ms.append((lab, kind, float(env[f"M_{lab}_COEF"]), env.get(f"M_{lab}_SRC", ""), env.get(f"M_{lab}_HUB", "")))
    assert ms, "no members"
    return ms


def sha256_file(p, bs=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def sha256_many(paths, threads=16):
    with ThreadPoolExecutor(threads) as ex:
        return dict(zip(paths, ex.map(sha256_file, paths)))


def dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)
