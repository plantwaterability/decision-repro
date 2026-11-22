#!/usr/bin/env python3
"""(SCREEN4M) Data for arm M of the SCREEN4 line (tmp/screen4m/prereg.md): arm LP of SCREEN4F (lr 1.5e-5, Garas-parent weight 1.0) with
its R2[2,000] replay block REPLACED by 2,000 units of (99)'s own mix2 TRAIN data drawn at mix2 val's family (file) shares, i.e. the
families caldev measures, rehearsed with Garas / (99)'s own order-marginal predictions as targets (t' = (g + q) / 2).

  sample   X99 mix2 train (jev_data train + jev_ext_strict / jev_ext2 train under their mix2 names; build_prior_cond.x99_files' naming)
           -> 2,000 units, quotas = mix2 val's NC-free unit share per X99 file name (largest remainder), seeded record order, greedy
           exact fill per file, shortfalls refilled from the files with the largest val share. A record is ELIGIBLE iff (PC's replay
           rules) its state is not in jev_real test nor the rd3 stack pool, it is not non-commercial (cache_features rule, judged with the
           X99 file stem), every question has <= 52 options; AND (new) its id / state is not in PC's R or R2 and its state is not in
           mix2 val nor any SCREEN4 dev panel (didev, egdev, caldev). Writes OUT/m.jsonl and the teacher dir OUT/teach_m/test/m.jsonl
           (byte-identical) + OUT/sample_manifest.json.
  mix      the (99) teacher npz on teach_m -> arm M = LP's replay_R.jsonl and eg.jsonl (byte copies of SCREEN4F's arm c) + mix2_M.jsonl
           (gold mixed with the teacher row, weight 1.0; exact uid join with K / target checks) + PC's mix2 val (byte copies); 12,000
           units checked.

  python3 build_screen4m.py --self-test
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE / "analysis", HERE / "trainers"):
    sys.path.insert(0, str(_p))

VERSION = "build_screen4m.py v1 (2026-10-13)"
N_M = 2000
LAM_M = 1.0
K_MAX = 52
NS = "screen4m/v1"


def mix2_files(jev_data, ext_strict, ext2, split):
    """[(X99 name, path)] as build_prior_cond.x99_files names mix2 (jev_data files by name; ext_ / ext2_ + relative path, '/' -> '__')"""
    out = [(p.name, p) for p in sorted((Path(jev_data) / split).glob("*.jsonl"))]
    for root, pre in ((ext_strict, "ext_"), (ext2, "ext2_")):
        base = Path(root) / split
        out += [(pre + str(p.relative_to(base)).replace("/", "__"), p) for p in sorted(base.glob("**/*.jsonl"))]
    return sorted(out)


def quotas(shares, n):
    """largest-remainder integer quotas summing to n; ties by name"""
    tot = sum(shares.values())
    raw = {k: n * v / tot for k, v in shares.items()}
    q = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - q[k]), k))[: n - sum(q.values())]:
        q[k] += 1
    return q


def sample(a):
    import build_screen4 as bs
    cf = _load(Path(a.cf))
    norm = bs.norm
    rp = Path(a.replay)
    real_state, pool_state, drop = set(), set(), collections.Counter()
    if a.jev_real:
        real_state = {norm(json.loads(l)["state"]) for l in open(Path(a.jev_real) / "test" / "real.jsonl", encoding="utf-8") if l.strip()}
    if a.stackpool:
        for p in sorted(Path(a.stackpool).glob("**/*.jsonl")):
            pool_state |= {norm(json.loads(l)["state"]) for l in open(p, encoding="utf-8") if l.strip()}
    used_state, used_id = set(), set()
    for tag in ("R", "R2"):
        for r in bs.replay_records(rp / tag / "train"):
            used_id.add(str(r["id"]))
            used_state.add(norm(r["state"]))
    dev_state = set()
    for d in a.dev:
        for f in sorted(Path(d).rglob("*.jsonl")):
            dev_state |= {norm(json.loads(l)["state"]) for l in open(f, encoding="utf-8") if l.strip()}
    # val shares (NC-free units per X99 file name) and val states
    shares, val_state = collections.Counter(), set()
    for f in sorted((rp / "val").glob("*.jsonl")):
        for l in open(f, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            val_state.add(norm(r["state"]))
            lic = str(r.get("license") or (r.get("meta") or {}).get("license") or "")
            if not cf.is_noncommercial({"source": r.get("source") or f.stem, "license": lic}):
                shares[f.name] += len(r["questions"])
    train = dict(mix2_files(a.jev_data, a.jev_ext_strict, a.jev_ext2, "train"))
    q = quotas({k: v for k, v in shares.items() if k in train}, a.n)
    missing = sorted(k for k in shares if k not in train)
    # eligible records per file, seeded order
    elig = {}
    for name in sorted(q):
        stem, recs = Path(name).stem, []
        for l in open(train[name], encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            st = norm(r["state"])
            if st in real_state:
                drop["jev_real_state"] += 1; continue
            if st in pool_state:
                drop["stackpool_state"] += 1; continue
            if str(r["id"]) in used_id or st in used_state:
                drop["in_R_or_R2"] += 1; continue
            if st in val_state:
                drop["mix2_val_state"] += 1; continue
            if st in dev_state:
                drop["dev_state"] += 1; continue
            lic = str(r.get("license") or (r.get("meta") or {}).get("license") or "")
            if cf.is_noncommercial({"source": r.get("source") or stem, "license": lic}):
                drop["noncommercial"] += 1; continue
            ks = [len(cf.question_options(qq, r["labels"][qid])[0]) for qid, qq in r["questions"].items()]
            if not ks or max(ks) > K_MAX:
                drop["k_gt_52_or_empty"] += 1; continue
            recs.append(r)
        recs.sort(key=lambda r: bs.sha1(f"{NS}|{a.seed}|{name}|{r['id']}"))
        elig[name] = recs
    picked, got, seen_st, seen_id = {k: [] for k in q}, collections.Counter(), set(), set()

    def fill(name, want):
        for r in elig[name]:
            if got[name] >= want:
                break
            u, st, rid = len(r["questions"]), norm(r["state"]), str(r["id"])
            if st in seen_st or rid in seen_id:
                continue
            if got[name] + u <= want:
                picked[name].append(r); got[name] += u; seen_st.add(st); seen_id.add(rid)

    for name in sorted(q):
        fill(name, q[name])
    short = a.n - sum(got.values())
    for name in sorted(q, key=lambda k: (-shares[k], k)):      # refill from the largest-share files
        if short <= 0:
            break
        before = got[name]
        fill(name, got[name] + short)
        short -= got[name] - before
    if sum(got.values()) != a.n:
        raise SystemExit(f"could not fill {a.n} units exactly: {sum(got.values())}")
    out = Path(a.out)
    recs = []
    for name in sorted(picked):
        for r in picked[name]:
            r = json.loads(json.dumps(r))
            r.setdefault("meta", {}).setdefault("screen4", {}).update({"role": "mix2m", "x99_file": name})
            recs.append(r)
    bs.write_jsonl(out / "m.jsonl", recs)
    bs.write_jsonl(out / "teach_m" / "test" / "m.jsonl", recs)
    man = {"version": VERSION, "n": a.n, "seed": a.seed, "quotas": q, "units_by_file": dict(got), "records": len(recs),
           "val_shares_units": dict(shares), "val_files_without_train": missing, "drops": dict(drop),
           "jev_real_filter": bool(a.jev_real), "stackpool_filter": bool(a.stackpool), "dev_dirs": [str(d) for d in a.dev],
           "files": {"m.jsonl": bs.sha256_file(out / "m.jsonl")}}
    (out / "sample_manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    print(f"SAMPLE_OK {a.n} units, {len(recs)} records, {len(q)} files; shortfall-refilled {sum(max(0, got[k] - q[k]) for k in q)}; "
          f"drops {dict(drop)}")
    return man


def mix(a):
    import build_screen4 as bs
    import screen4_loss as sl
    sd, lp, out = Path(a.sample), Path(a.lp_arm), Path(a.out)
    recs = bs.read_jsonl(sd / "m.jsonl")
    if (sd / "teach_m" / "test" / "m.jsonl").read_bytes() != (sd / "m.jsonl").read_bytes():
        raise SystemExit("teach_m (what the teacher scored) is not byte-identical to m.jsonl")
    units = bs.unit_list(recs)
    gold = [bs.gold_vector(r, q) for _, r, q in units]
    parent = sl.teacher_rows(a.parent_npz, [u for u, _, _ in units], gold, [len(g) for g in gold])
    new = {u: sl.mix(g, parent[u], LAM_M) for (u, _, _), g in zip(units, gold)}
    mixed = bs.apply_mixed(recs, new)
    d = out / "arm_m"
    (d / "train").mkdir(parents=True, exist_ok=True)
    (d / "val").mkdir(parents=True, exist_ok=True)
    for stem in ("replay_R.jsonl", "eg.jsonl"):
        (d / "train" / stem).write_bytes((lp / "train" / stem).read_bytes())
    bs.write_jsonl(d / "train" / "mix2_M.jsonl", mixed)
    for f in sorted((lp / "val").glob("*.jsonl")):
        (d / "val" / f.name).write_bytes(f.read_bytes())
    n = {f.name: sum(map(bs.units_of, bs.read_jsonl(f))) for f in sorted((d / "train").glob("*.jsonl"))}
    sm = json.load(open(sd / "sample_manifest.json"))
    if sorted(n) != ["eg.jsonl", "mix2_M.jsonl", "replay_R.jsonl"] or n["mix2_M.jsonl"] != sm["n"] or sum(n.values()) != a.expect_units:
        raise SystemExit(f"arm M units {n} are not R + M ({sm['n']}) + EG = {a.expect_units}")
    lpm = json.load(open(lp.parent / "arms_mixed_manifest.json"))
    if (lpm.get("lam_parent"), lpm.get("lam_teacher")) != (1.0, 0.3):
        raise SystemExit(f"--lp-arm is not SCREEN4F's mixp arm c (weights {lpm.get('lam_parent')} / {lpm.get('lam_teacher')})")
    moved = sum(1 for (u, r, q), g in zip(units, gold) if max(abs(x - y) for x, y in zip(new[u], [v / sum(g) for v in g])) > 1e-9)
    man = {"version": VERSION, "lam_m": LAM_M, "units": n, "m_units_moved": moved, "lp_manifest_weights": [1.0, 0.3],
           "parent_npz_sha256": bs.sha256_file(a.parent_npz), "sample_manifest_sha256": bs.sha256_file(sd / "sample_manifest.json"),
           "files": bs.file_table(d)}
    (out / "arm_m_manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    print(f"MIX_M_OK units {n} moved {moved}")
    return man


def _load(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("cf_m", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("sample", "mix"))
    ap.add_argument("--jev-data"); ap.add_argument("--jev-ext-strict"); ap.add_argument("--jev-ext2")
    ap.add_argument("--jev-real", help="dir holding test/real.jsonl (its states are never sampled)")
    ap.add_argument("--stackpool", help="the rd3 stack pool dir (its states are never sampled)")
    ap.add_argument("--replay", help="PC's pc_replay_v1 dir (R, R2, val)")
    ap.add_argument("--dev", action="append", default=[], help="dev panel dirs whose states are never sampled")
    ap.add_argument("--cf", default=str(HERE / "trainers" / "cache_features.py"))
    ap.add_argument("--n", type=int, default=N_M)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", help="mix: the sample dir")
    ap.add_argument("--parent-npz")
    ap.add_argument("--lp-arm", help="mix: SCREEN4F's mixp arms/arm_c dir (LP's data)")
    ap.add_argument("--expect-units", type=int, default=12000, help="mix: arm M's total units (12,000 registered; tests use less)")
    ap.add_argument("--allow-no-filters", action="store_true", help="tests / dry runs only: sample without jev_real / stack pool")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.cmd == "sample":
        if not (a.jev_real and a.stackpool) and not a.allow_no_filters:
            raise SystemExit("sample needs --jev-real and --stackpool (PC's replay eligibility; --allow-no-filters is for tests only)")
        sample(a)
    elif a.cmd == "mix":
        mix(a)
    return 0


def self_test():
    q = quotas({"a": 3, "b": 1, "c": 1}, 7)
    assert q == {"a": 4, "b": 2, "c": 1}, q
    assert sum(quotas({"x": 0.11, "y": 0.103, "z": 0.087}, 2000).values()) == 2000
    print("SELF_TEST_OK build_screen4m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
