#!/usr/bin/env python3
"""(PCD AMEND A1) the length / schedule-matched control arm L: a THIRD disjoint replay draw R3, and arm l.

Design: tmp/prior_cond_dose/amend_a1.md. Arm l = R + R3 + N = 6,000 + 6,000 + 6,000 = 18,000 units, 2,250 steps (D's length and
cosine schedule; D's replay base R and numeric base N), so D - L isolates numeric amount (6,000 fresh numeric vs 6,000 fresh replay)
at matched steps, schedule and bases.

R3 = PC's replay draw (build_prior_cond.replay, e24fdd0d) CONTINUED: the same eligibility rules (state not in the rd3 stack pool nor in
jev_real test; not non-commercial by the ORIGINAL file stem; every question <= 52 options), the same single seeded shuffle
("prior_cond/v1|replay") and the same greedy exact-fill rule, with a third bin filled only after R and R2 are full. R and R2 are a
prefix of the same process, so the build REPRODUCES PC's published R / R2 files byte for byte (checked against pc_replay_v1's manifest
and files; fail-closed) and R3 is disjoint from both by construction (record ids checked too). Verbatim lines; file stems r3_<X99 name>.

    python3 build_pcd2.py replay3 --bpc build_prior_cond.py --jev-data D ... --jev-real DIR --ref-replay DIR/pc_replay_v1 --out DIR \
        --cf cache_features.py --overlap-py claude_overlap.py [--inputs-json J]
    python3 build_pcd2.py check-r3 DIR/pcd_r3_v1 --ref-replay DIR/pc_replay_v1
    python3 build_pcd2.py arm-l --replay DIR/pc_replay_v1 --r3 DIR/pcd_r3_v1 --pcnum DIR/pc_numeric_v1 --out ARMDIR
    python3 build_pcd2.py self-test
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

VERSION = "build_pcd2.py v1 (2026-10-12)"
UNITS = 6000
L_UNITS = 18000
FAMS = ("coin", "rate", "normal_mean")
PCB_MD5 = "e24fdd0d3b772ee794c0e1ec501499b7"       # build_prior_cond.py (deployed; x99_files / NS_PC / K_MAX come from it)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_pcb(path, check_md5=True):
    if check_md5 and hashlib.md5(Path(path).read_bytes()).hexdigest() != PCB_MD5:
        raise SystemExit(f"{path} is not build_prior_cond.py {PCB_MD5[:8]} (PC's replay rules)")
    return load_path("build_prior_cond_pcd2", path)


def fill(order, sizes, units, bins):
    """PC's greedy exact fill (build_prior_cond.replay) generalised to `bins` bins: walk the shuffled order; a record goes to the FIRST
    bin that is not yet full if it fits, else it is skipped; stop when every bin is full. With bins = 2 this is PC's loop verbatim, and
    bins 0 / 1 of a 3-bin fill are identical to it (bin 2 only receives records after bins 0 and 1 are both full)."""
    picks, got = tuple([] for _ in range(bins)), [0] * bins
    for k in order:
        n = sizes[k]
        for j in range(bins):
            if got[j] < units:
                if got[j] + n <= units:
                    picks[j].append(k); got[j] += n
                break
        if all(g == units for g in got):
            break
    return picks, got


def eligible(pcb, src, jev_real, cf, ov):
    """PC's eligibility pass, verbatim (build_prior_cond.replay): -> recs [(file index, line, units)], drop, train_files, states"""
    real = Path(jev_real) / "test" / "real.jsonl"
    real_states = {ov.norm(json.loads(l)["state"]) for l in open(real, encoding="utf-8") if l.strip()}
    pool_states = set()
    for p in sorted(Path(src["jev_rd3_stackpool"]).glob("**/*.jsonl")):
        pool_states.update(ov.norm(json.loads(l)["state"]) for l in open(p, encoding="utf-8") if l.strip())
    if not pool_states:
        raise SystemExit("empty stack pool: wrong directory")
    train_files, _ = pcb.x99_files(src)
    recs, drop = [], Counter()
    for fi, (name, p, filt) in enumerate(train_files):
        stem = Path(name).stem
        for l in open(p, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            st = ov.norm(r["state"])
            if st in real_states:
                drop["jev_real_test_state" + ("" if filt else "_nonmix2")] += 1; continue
            if st in pool_states:
                drop["stackpool_state"] += 1; continue
            lic = str(r.get("license") or (r.get("meta") or {}).get("license") or "")
            if cf.is_noncommercial({"source": r.get("source") or stem, "license": lic}):
                drop["noncommercial"] += 1; continue
            ks = [len(cf.question_options(q, r["labels"][qid])[0]) for qid, q in r["questions"].items()]
            if not ks or max(ks) > pcb.K_MAX:
                drop["k_gt_52_or_empty"] += 1; continue
            recs.append((fi, l if l.endswith("\n") else l + "\n", len(ks)))
    return recs, drop, train_files, real_states, pool_states, real


def texts_for(sel, recs, train_files, pre):
    by = defaultdict(list)
    for k in sorted(sel):
        by[recs[k][0]].append(recs[k][1])
    return {pre + train_files[fi][0]: "".join(ls) for fi, ls in sorted(by.items())}


def replay3(pcb_path, src, out, jev_real, cf_path, ov_path, ref_replay, units=UNITS, inputs=None, log=print, check_pcb=True):
    t0 = time.time()
    pcb = load_pcb(pcb_path, check_pcb)
    cf, ov = load_path("cf_pcd2", cf_path), load_path("ov_pcd2", ov_path)
    out, ref = Path(out), Path(ref_replay)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty: refusing to overwrite a build")
    refman = json.loads((ref / "manifest.json").read_text())
    if refman.get("units_each") != units:
        raise SystemExit(f"the reference replay holds {refman.get('units_each')} units per bin, not {units}")
    recs, drop, train_files, real_states, pool_states, real = eligible(pcb, src, jev_real, cf, ov)
    order = list(range(len(recs)))
    random.Random(f"{pcb.NS_PC}|replay").shuffle(order)
    picks, got = fill(order, [r[2] for r in recs], units, 3)
    if got != [units] * 3:
        raise SystemExit(f"could not fill R / R2 / R3 exactly: {got}")
    # R and R2 must be PC's published files, byte for byte (the same process; fail-closed)
    rep = {}
    for tag, sel, pre in (("R", picks[0], "r_"), ("R2", picks[1], "r2_")):
        want = {k: v for k, v in refman["files"].items() if k.startswith(f"{tag}/train/")}
        mine = {f"{tag}/train/{n}": sha256_bytes(t.encode("utf-8")) for n, t in texts_for(sel, recs, train_files, pre).items()}
        disk = {k: sha256_file(ref / k) for k in want}
        if mine != want or disk != want:
            bad = sorted(set(mine.items()) ^ set(want.items()))[:3]
            raise SystemExit(f"{tag} is not reproduced: this build differs from PC's published replay ({bad}); refusing to draw R3")
        rep[tag] = {"files": len(want), "identical": True}
    d = out / "R3" / "train"; d.mkdir(parents=True)
    for n, t in texts_for(picks[2], recs, train_files, "r3_").items():
        (d / n).write_text(t, encoding="utf-8")
    us = list(cf.iter_units(str(out / "R3"), ["train"], set(), None))
    ver = {"units": len(us), "nc_by_trainer_rule": sum(bool(cf.is_noncommercial(u)) for u in us),
           "k_gt_52": sum(len(u["option_keys"]) > pcb.K_MAX for u in us),
           "stackpool_or_real_states": sum(ov.norm(u["state"]) in pool_states or ov.norm(u["state"]) in real_states for u in us)}
    if ver != {"units": units, "nc_by_trainer_rule": 0, "k_gt_52": 0, "stackpool_or_real_states": 0}:
        raise SystemExit(f"R3 failed verification: {ver}")
    ids = ids_of(out / "R3/train")
    for tag in ("R", "R2"):
        shared = ids & ids_of(ref / tag / "train")
        if shared:
            raise SystemExit(f"R3 shares {len(shared)} record ids with {tag}")
    cls = lambda name: {"rd3_": "rd3", "bp_": "flights", "gdelt_": "gdelt"}.get(next((x for x in ("rd3_", "bp_", "gdelt_")
                                                                                   if name.startswith(x)), ""), "mix2")
    comp = dict(Counter(cls(train_files[recs[k][0]][0]) for k in picks[2] for _ in range(recs[k][2])))
    man = {"version": VERSION, "rules": f"build_prior_cond.replay {PCB_MD5} continued (third bin)", "namespace": pcb.NS_PC,
           "units": units, "inputs": inputs or {}, "jev_real_sha256": sha256_file(real), "pool_records": len(recs),
           "dropped": dict(drop), "x99_train_files": len(train_files), "reproduce": rep, "verify": {"R3": ver},
           "records": {"R": len(picks[0]), "R2": len(picks[1]), "R3": len(picks[2])}, "composition_units": {"R3": comp},
           "ref_replay_manifest_sha256": sha256_file(ref / "manifest.json"), "files": {}, "build_secs": round(time.time() - t0, 1)}
    for f in sorted(out.glob("R3/train/*.jsonl")):
        man["files"][str(f.relative_to(out))] = sha256_file(f)
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    log(f"replay3: R / R2 reproduce PC's published files ({rep['R']['files']} + {rep['R2']['files']} files); R3 {len(us)} units "
        f"({len(picks[2])} records) disjoint from both; composition {comp}; {time.time() - t0:.0f} s")
    return man


def ids_of(d):
    return {json.loads(l)["id"] for p in Path(d).glob("*.jsonl") for l in open(p, encoding="utf-8") if l.strip()}


def check_r3(r3, ref_replay, log=print):
    """the run's pre-training check of a fetched R3: manifest flags, file sha256s, disjointness from R / R2, the unit count by lines"""
    r3, ref = Path(r3), Path(ref_replay)
    E = []
    man = json.loads((r3 / "manifest.json").read_text())
    if man.get("reproduce", {}).get("R", {}).get("identical") is not True or man.get("reproduce", {}).get("R2", {}).get("identical") is not True:
        E.append("R3 manifest: R / R2 not recorded as reproduced")
    if man.get("verify", {}).get("R3") != {"units": UNITS, "nc_by_trainer_rule": 0, "k_gt_52": 0, "stackpool_or_real_states": 0}:
        E.append(f"R3 manifest: verification {man.get('verify')}")
    if man.get("ref_replay_manifest_sha256") != sha256_file(ref / "manifest.json"):
        E.append("R3 was drawn against another reference replay")
    files = sorted(str(p.relative_to(r3)) for p in r3.glob("R3/train/*.jsonl"))
    if files != sorted(man.get("files", {})):
        E.append("R3 files differ from the manifest list")
    for k in files:
        if sha256_file(r3 / k) != man["files"].get(k):
            E.append(f"{k}: sha256 differs from the manifest")
    ids = ids_of(r3 / "R3/train")
    for tag in ("R", "R2"):
        if ids & ids_of(ref / tag / "train"):
            E.append(f"R3 shares record ids with {tag}")
    log(f"check-r3: {len(E)} errors" + (": " + "; ".join(E[:4]) if E else ""))
    return E


def arm_l(replay, r3, pcnum, out, log=print):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    (out / "train").mkdir(parents=True, exist_ok=True); (out / "val").mkdir(parents=True, exist_ok=True)
    srcs = sorted(Path(replay, "R/train").glob("*.jsonl")) + sorted(Path(r3, "R3/train").glob("*.jsonl"))
    srcs += [Path(pcnum, "N/train", f"numeric_{f}.jsonl") for f in FAMS]
    names = [p.name for p in srcs]
    if len(set(names)) != len(names):
        raise SystemExit("file-name collision in arm l")
    for p in srcs:
        if not p.exists():
            raise SystemExit(f"arm l: missing {p}")
        os.symlink(os.path.abspath(p), out / "train" / p.name)
    for p in sorted(Path(replay, "val").glob("*.jsonl")):
        os.symlink(os.path.abspath(p), out / "val" / p.name)
    fam = Counter()
    for p in srcs[-3:]:
        for l in open(p, encoding="utf-8"):
            fam[json.loads(l)["meta"]["family"]] += 1
    s = {"arm": "l", "files": len(srcs), "replay_files": len(srcs) - 3, "numeric_units_by_family": dict(sorted(fam.items())),
         "numeric_units": sum(fam.values()), "expected_units": L_UNITS}
    if s["numeric_units"] != 6000 or any(v != 2000 for v in fam.values()):
        raise SystemExit(f"arm l: N is not 2,000 per family: {s}")
    log(json.dumps(s))
    return s


def self_test():
    rng = random.Random(11)
    for trial in range(200):
        sizes = [rng.choice([1, 1, 1, 2, 3, 5]) for _ in range(400)]
        order = list(range(400)); rng.shuffle(order)
        u = rng.choice([40, 57, 100, 131])
        p2, g2 = fill(order, sizes, u, 2)
        p3, g3 = fill(order, sizes, u, 3)
        assert p3[0] == p2[0] and p3[1] == p2[1] and g3[:2] == g2, trial          # R / R2 = PC's 2-bin fill, exactly
        assert not (set(p3[2]) & (set(p3[0]) | set(p3[1])))                         # R3 disjoint
        assert all(g <= u for g in g3) and sum(sizes[k] for k in p3[2]) == g3[2]
        if g3 == [u] * 3:
            first2 = max(order.index(k) for k in p3[0] + p3[1])
            assert all(order.index(k) > first2 for k in p3[2])                       # R3 only after R and R2 are full
    print("self-test passed (3-bin fill = PC's fill on bins 1-2; bin 3 disjoint and later)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay3")
    for k in ("bpc", "jev-data", "jev-ext-strict", "jev-ext2", "jev-rd3", "jev-rd3-stackpool", "jev-gdelt-v1", "bp-train-v1", "jev-real",
              "ref-replay", "out", "cf", "overlap-py"):
        r.add_argument("--" + k, required=True)
    r.add_argument("--inputs-json", default="{}")
    c = sub.add_parser("check-r3"); c.add_argument("r3"); c.add_argument("--ref-replay", required=True)
    a = sub.add_parser("arm-l")
    for k in ("replay", "r3", "pcnum", "out"):
        a.add_argument("--" + k, required=True)
    sub.add_parser("self-test")
    x = ap.parse_args(argv)
    if x.cmd == "replay3":
        src = {"jev_data": x.jev_data, "jev_ext_strict": x.jev_ext_strict, "jev_ext2": x.jev_ext2, "jev_rd3": x.jev_rd3,
               "jev_rd3_stackpool": x.jev_rd3_stackpool, "jev_gdelt_v1": x.jev_gdelt_v1, "bp_train_v1": x.bp_train_v1}
        replay3(x.bpc, src, x.out, x.jev_real, x.cf, x.overlap_py, x.ref_replay, inputs=json.loads(x.inputs_json))
        return 0
    if x.cmd == "check-r3":
        return 1 if check_r3(x.r3, x.ref_replay) else 0
    if x.cmd == "arm-l":
        arm_l(x.replay, x.r3, x.pcnum, x.out)
        return 0
    return self_test()


if __name__ == "__main__":
    raise SystemExit(main())
