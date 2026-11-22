#!/usr/bin/env python3
"""build_grok1_subset.py -- the (G1) grokking experiment's data (AGENT_NOTES 2026-10-09 04:35Z registration, 04:41Z amendment A6).

From an extracted jev_bayes_worlds_v1_2h corpus (md5-pinned tarball on the node; the laptop build for tests) it writes, with every
line copied BYTE-IDENTICAL from its source file (source line order kept, one output file per source file name):
  train/     whole GROUPS (meta.group_id) of train_aug: per stratum (family, structure) the quota is TRAIN x the stratum's share of
             units; the stratum's groups are taken in sha256('grok1|train|' + group_id) order while each addition reduces
             |selected - quota| (the first group always: >= 1 group per stratum)
  val/       whole GROUPS of the val split (unseen worlds), the same rule with tag 'val' and target VAL (amendment A6: invariance pairs
             stay together; the read's bootstrap resamples groups)
  novel/     NOVEL units of test_novel_wording: largest-remainder quotas per stratum, sha256('grok1|novel|' + record id) order
  trainfit/  TRAINFIT units of train/ (the selected subset), the same unit rule with tag 'trainfit'
  manifest.json  rule, seed, the source files' md5s / line counts, per-stratum quotas / selections / availability, the selected group
             ids, the checks, and the platform-independent DIGESTS (md5 of the sorted record ids per output split)
Only train_aug/, val/ and test_novel_wording/ are ever opened. test_confirm is never read (a path containing 'confirm' is refused).
Checks (exit 1 on failure): one question per record, choice-type with 2..52 options, labels over exactly the criteria keys summing to
1 (+-1e-3); no meta.fingerprint and no state string shared between train/ and val/ or novel/.

  python3 build_grok1_subset.py build --src ROOT --out OUT [--train-units 2000 --val-units 800 --novel-units 500 --trainfit-units 500]
  python3 build_grok1_subset.py digest OUT      # the digests line (what fleet/task_grok.sh pins)
  python3 build_grok1_subset.py md5 DIR         # content md5 of a split directory (task_105e.sh's tmd5)
NEVER a candidate's data: G1 outputs are never candidates, teachers or zero-shot rows.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys

SEED = "grok1"
READ_SPLITS = ("train_aug", "val", "test_novel_wording")
OUT_SPLITS = ("train", "val", "novel", "trainfit")
MAX_OPTIONS = 52


def die(msg):
    print(f"GROK1_SUBSET_FAIL {msg}", file=sys.stderr)
    raise SystemExit(1)


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def tmd5(d):
    """content md5 of a split directory: the *.jsonl files in byte order of their names, each as name + newline + bytes"""
    h = hashlib.md5()
    for n in sorted((x for x in os.listdir(d) if x.endswith(".jsonl")), key=lambda x: x.encode()):
        h.update(n.encode() + b"\n")
        with open(os.path.join(d, n), "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def file_md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def read_split(root, split):
    """[(file name, raw line bytes, record)] of ROOT/SPLIT/*.jsonl in file-name then line order; only READ_SPLITS, never 'confirm'."""
    if split not in READ_SPLITS:
        die(f"split {split!r} is not one this builder reads {READ_SPLITS}")
    d = os.path.join(root, split)
    if "confirm" in os.path.realpath(d).lower():
        die(f"refusing a path that names 'confirm': {d}")
    if not os.path.isdir(d):
        die(f"no split directory {d}")
    out = []
    for fn in sorted((x for x in os.listdir(d) if x.endswith(".jsonl") and not x.startswith("._")), key=lambda x: x.encode()):
        with open(os.path.join(d, fn), "rb") as f:
            for ln in f:
                if not ln.strip():
                    continue
                if not ln.endswith(b"\n"):
                    ln += b"\n"
                out.append((fn, ln, json.loads(ln)))
    if not out:
        die(f"split {split} is empty at {d}")
    return out


def check_record(rec, where):
    qs = rec.get("questions") or {}
    if len(qs) != 1:
        die(f"{where} {rec.get('id')}: {len(qs)} questions (one expected)")
    qid, q = next(iter(qs.items()))
    crit = q.get("criteria") or {}
    if q.get("type") != "choice" or not 2 <= len(crit) <= MAX_OPTIONS:
        die(f"{where} {rec.get('id')}: type {q.get('type')} with {len(crit)} options")
    lab = (rec.get("labels") or {}).get(qid) or {}
    if set(lab) != set(crit) or abs(sum(float(x) for x in lab.values()) - 1.0) > 1e-3:
        die(f"{where} {rec.get('id')}: labels do not cover the options or do not sum to 1")
    m = rec.get("meta") or {}
    for k in ("family", "structure", "group_id"):
        if not m.get(k):
            die(f"{where} {rec.get('id')}: meta.{k} missing")


def stratum(rec):
    m = rec["meta"]
    return f"{m['family']}/{m['structure']}"


def pick_groups(rows, target, tag):
    """whole groups per stratum: quota = target x the stratum's unit share; groups in sha256(seed|tag|group) order while each
    addition reduces |selected - quota| (the first always). -> (selected group ids, per-stratum report)"""
    size, gstr = collections.Counter(), {}
    for _, _, r in rows:
        g = r["meta"]["group_id"]
        size[g] += 1
        s = stratum(r)
        if gstr.setdefault(g, s) != s:
            die(f"group {g} spans strata {gstr[g]} and {s}")
    total = sum(size.values())
    by = collections.defaultdict(list)
    for g, s in gstr.items():
        by[s].append(g)
    chosen, rep = set(), {}
    for s in sorted(by):
        gs = sorted(by[s], key=lambda g: sha(f"{SEED}|{tag}|{g}"))
        avail = sum(size[g] for g in gs)
        q = target * avail / total
        n, sel = 0, []
        for g in gs:
            if not sel or abs(n + size[g] - q) < abs(n - q):
                sel.append(g)
                n += size[g]
            else:
                break
        chosen.update(sel)
        rep[s] = {"quota": round(q, 3), "units": n, "groups": len(sel), "avail_units": avail, "avail_groups": len(gs)}
    return chosen, rep


def pick_units(rows, target, tag):
    """units per stratum by largest-remainder quotas (ties: stratum name), sha256(seed|tag|record id) order within a stratum"""
    by = collections.defaultdict(list)
    for i, (_, _, r) in enumerate(rows):
        by[stratum(r)].append(i)
    total = len(rows)
    if target > total:
        die(f"{tag}: {target} units requested, {total} available")
    exact = {s: target * len(ix) / total for s, ix in by.items()}
    quota = {s: int(x) for s, x in exact.items()}
    for s in sorted(by, key=lambda s: (-(exact[s] - quota[s]), s))[: target - sum(quota.values())]:
        quota[s] += 1
    chosen, rep = set(), {}
    for s in sorted(by):
        ix = sorted(by[s], key=lambda i: sha(f"{SEED}|{tag}|{rows[i][2]['id']}"))[: quota[s]]
        chosen.update(ix)
        rep[s] = {"quota": quota[s], "units": len(ix), "avail_units": len(by[s])}
    return chosen, rep


def write_split(out, name, rows, keep):
    """rows[i] for i in keep, byte-identical, in source order, one output file per source file name"""
    d = os.path.join(out, name)
    os.makedirs(d)
    files = collections.defaultdict(list)
    for i, (fn, ln, _) in enumerate(rows):
        if i in keep:
            files[fn].append(ln)
    for fn, lns in files.items():
        with open(os.path.join(d, fn), "wb") as f:
            f.write(b"".join(lns))
    return sum(len(v) for v in files.values())


def digests(out):
    """platform-independent: per output split, n and md5 of the sorted record ids (the SELECTION, not the float bytes)"""
    res = {}
    for name in OUT_SPLITS:
        d = os.path.join(out, name)
        ids = []
        for fn in sorted(x for x in os.listdir(d) if x.endswith(".jsonl")):
            with open(os.path.join(d, fn), "rb") as f:
                ids += [json.loads(ln)["id"] for ln in f if ln.strip()]
        if len(ids) != len(set(ids)):
            die(f"{name}: duplicate record ids")
        res[name] = {"n": len(ids), "ids_md5": hashlib.md5("\n".join(sorted(ids)).encode()).hexdigest()}
    return res


def digest_line(dg):
    return " ".join(f"{k}:{dg[k]['n']}:{dg[k]['ids_md5']}" for k in OUT_SPLITS)


def build(a):
    if os.path.exists(a.out) and os.listdir(a.out):
        die(f"--out {a.out} exists and is not empty (never overwritten)")
    src = {s: read_split(a.src, s) for s in READ_SPLITS}
    for s, rows in src.items():
        for _, _, r in rows:
            check_record(r, s)
    tr, va, nv = src["train_aug"], src["val"], src["test_novel_wording"]
    tg, trep = pick_groups(tr, a.train_units, "train")
    tkeep = {i for i, (_, _, r) in enumerate(tr) if r["meta"]["group_id"] in tg}
    vg, vrep = pick_groups(va, a.val_units, "val")
    vkeep = {i for i, (_, _, r) in enumerate(va) if r["meta"]["group_id"] in vg}
    nkeep, nrep = pick_units(nv, a.novel_units, "novel")
    sub = [tr[i] for i in sorted(tkeep)]
    fkeep, frep = pick_units(sub, a.trainfit_units, "trainfit")
    # leakage checks: the training subset shares no world fingerprint and no state text with the eval samples
    tfp = {r["meta"].get("fingerprint") for _, _, r in sub} - {None}
    tst = {r["state"] for _, _, r in sub}
    chk = {}
    for name, rows, keep in (("val", va, vkeep), ("novel", nv, nkeep)):
        sel = [rows[i][2] for i in keep]
        chk[f"fingerprint_overlap_{name}"] = sum(1 for r in sel if r["meta"].get("fingerprint") in tfp)
        chk[f"state_overlap_{name}"] = sum(1 for r in sel if r["state"] in tst)
    bad = {k: v for k, v in chk.items() if v}
    if bad:
        die(f"train / eval overlap: {bad}")
    os.makedirs(a.out, exist_ok=True)
    n = {"train": write_split(a.out, "train", tr, tkeep), "val": write_split(a.out, "val", va, vkeep),
         "novel": write_split(a.out, "novel", nv, nkeep), "trainfit": write_split(a.out, "trainfit", sub, fkeep)}
    dg = digests(a.out)
    man = {
        "experiment": "G1 grokking (AGENT_NOTES 2026-10-09 04:35Z + 04:41Z amendment)", "seed": SEED,
        "role": "research only: never a candidate, teacher, stack member or zero-shot row", "test_confirm": "never read",
        "rule": {"train": f"whole meta.group_id groups of train_aug per family/structure stratum; quota = {a.train_units} x stratum unit "
                          "share; groups in sha256('grok1|train|' + group_id) order while each addition reduces |selected - quota| "
                          "(the first always)",
                 "val": f"the same group rule on the val split, tag 'val', target {a.val_units}",
                 "novel": f"{a.novel_units} units of test_novel_wording, largest-remainder quotas per stratum, sha256('grok1|novel|' + id)",
                 "trainfit": f"{a.trainfit_units} units of train/, largest-remainder quotas per stratum, sha256('grok1|trainfit|' + id)"},
        "targets": {"train": a.train_units, "val": a.val_units, "novel": a.novel_units, "trainfit": a.trainfit_units},
        "inputs": {s: {"lines": len(src[s]), "files": {fn: file_md5(os.path.join(a.src, s, fn))
                                                       for fn in sorted({x[0] for x in src[s]})}} for s in READ_SPLITS},
        "units": n, "groups": {"train": len(tg), "val": len(vg)},
        "by_stratum": {"train": trep, "val": vrep, "novel": nrep, "trainfit": frep},
        "group_ids": {"train": sorted(tg), "val": sorted(vg)},
        "checks": chk, "digests": dg, "digest_line": digest_line(dg),
        "content_md5": {s: tmd5(os.path.join(a.out, s)) for s in OUT_SPLITS}}
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1, sort_keys=True)
    print(f"GROK1_SUBSET_BUILT units {json.dumps(n, sort_keys=True)} groups train {len(tg)} val {len(vg)}; checks {json.dumps(chk)}")
    print(f"DIGESTS {man['digest_line']}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--src", required=True, help="extracted jev_bayes_worlds_v1_2h root (train_aug/, val/, test_novel_wording/)")
    b.add_argument("--out", required=True)
    b.add_argument("--train-units", type=int, default=2000)
    b.add_argument("--val-units", type=int, default=800)
    b.add_argument("--novel-units", type=int, default=500)
    b.add_argument("--trainfit-units", type=int, default=500)
    d = sp.add_parser("digest")
    d.add_argument("out")
    m = sp.add_parser("md5")
    m.add_argument("dir")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        return build(a)
    if a.cmd == "digest":
        print(digest_line(digests(a.out)))
        return 0
    print(tmd5(a.dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
