#!/usr/bin/env python3
"""(PCD) dose / held-out-family follow-up to (PC): the extra numeric_v2 blocks B3 / B4 and the two new arms' train dirs.

Design: tmp/prior_cond_dose/design.md; pre-registration draft: tmp/prior_cond_dose/prereg_draft.md. Registered origin: the (PC) s01
read, row C_PARTIAL (AGENT_NOTES 2026-10-12 00:29Z): "a dose arm (2x numeric exposure) ...; nothing adopted".

Blocks (numeric_v2 = fleetq/data/numeric_v2.tgz 5b7c4cb5; families coin / rate / normal_mean, 20,000 train items each):
  B1, B2   (PC, registered) the first 2 x 1,000 picks per family of the seeded stream random.Random("prior_cond/v1|select|<fam>")
           .sample(range(20000), .); PC's arm N = B1 + B2 (pc_numeric_v1.tgz 5cab5ec0, N/train/numeric_<fam>.jsonl).
  B3, B4   (NEW) picks 2,001-3,000 and 3,001-4,000 of the SAME stream: disjoint from B1 / B2 and from each other by construction
           (sampling without replacement), never dependent on the data. Records copied VERBATIM (byte-identical lines, index order).
           The stream is extended with ONE call, sample(range(20000), 4000); its first 2,000 picks must equal PC's registered
           sample(range(20000), 2000) (CPython's set-based branch is prefix-stable for these sizes; checked, fail-closed).
Arms (each = PC's replay R, 6,000 units of (99)'s own training set, results_pc/data/pc_replay_v1.tgz 67359006, byte for byte):
  d   R + N + B3 + B4 (all three families)        = 6,000 + 6,000 + 6,000 = 18,000 units: 2x PC's numeric exposure, fresh items
  h   R + N[coin, rate] + B3[coin, rate]          = 6,000 + 4,000 + 2,000 = 12,000 units: no normal_mean item at all (the
                                                     held-out family), numeric units matched to N
val = PC's mix2 val ((99)'s X99/val), as PC.

    python3 build_pcd.py build --numeric DIR/numeric_v2 --pcnum DIR/pc_numeric_v1 --out DIR/pcd_numeric_v1
    python3 build_pcd.py check DIR/pcd_numeric_v1 --numeric DIR/numeric_v2 --pcnum DIR/pc_numeric_v1 [--suite DIR] [--json OUT]
    python3 build_pcd.py arm d|h --replay DIR/pc_replay_v1 --pcnum DIR/pc_numeric_v1 --pcd DIR/pcd_numeric_v1 --out ARMDIR
    python3 build_pcd.py self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

VERSION = "build_pcd.py v1 (2026-10-12)"
NS_PC = "prior_cond/v1"              # PC's selection namespace, CONTINUED (not a new stream)
FAMS = ("coin", "rate", "normal_mean")
N_PER = 1000                         # per family and block
N_TRAIN = 20000                      # numeric_v2 train items per family
BLOCKS = ("b1", "b2", "b3", "b4")
NEW_BLOCKS = ("b3", "b4")
# numeric_v2 train files inside fleetq/data/numeric_v2.tgz (md5 5b7c4cb5): sha256, read back from S3 on 2026-10-12 (ctl2)
NUMERIC_TRAIN_SHA256 = {
    "coin": "253d5ea44db44028e3404dee915a70457c8253cbede895214376c78ef884269a",
    "rate": "57f260ed28b1948c55517134aba82ff011f466ca01341a78cf785ae768927ef8",
    "normal_mean": "c49ca2d046a49c3c995932f72a6ab883ed280048ab8e25c7b6b458d8ca18a8f5"}
ARM_SPEC = {   # arm -> (families taken from PC's N files, families x blocks taken from this build, expected units incl. replay R)
    "d": (FAMS, tuple((f, b) for f in FAMS for b in NEW_BLOCKS), 18000),
    "h": (("coin", "rate"), (("coin", "b3"), ("rate", "b3")), 12000)}
HELD_OUT = {"d": (), "h": ("normal_mean",)}
REPLAY_UNITS = 6000


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stream(fam, n_train=N_TRAIN):
    """The PC selection stream extended to 4 blocks; fails closed unless its first 2,000 picks are PC's registered draw."""
    pick = random.Random(f"{NS_PC}|select|{fam}").sample(range(n_train), 4 * N_PER)
    ref = random.Random(f"{NS_PC}|select|{fam}").sample(range(n_train), 2 * N_PER)       # = build_prior_cond.select_indices
    if pick[:2 * N_PER] != ref:
        raise SystemExit(f"{fam}: the extended draw does not continue PC's registered selection on this Python "
                         f"({sys.version.split()[0]}): refusing")
    return pick


def block_indices(fam, n_train=N_TRAIN):
    p = stream(fam, n_train)
    return {b: sorted(p[k * N_PER:(k + 1) * N_PER]) for k, b in enumerate(BLOCKS)}


def file_name(fam, block):
    return f"numeric_{fam}__pcd_{block}.jsonl"


def read_train(numeric, verify_sha=True):
    out = {}
    for fam in FAMS:
        p = Path(numeric) / "train" / f"numeric_{fam}.jsonl"
        raw = p.read_bytes()
        if verify_sha and sha256_bytes(raw) != NUMERIC_TRAIN_SHA256[fam]:
            raise SystemExit(f"{p}: sha256 {sha256_bytes(raw)[:12]} is not numeric_v2 5b7c4cb5's train file "
                             f"({NUMERIC_TRAIN_SHA256[fam][:12]})")
        out[fam] = raw.decode("utf-8").splitlines()
        if len(out[fam]) != N_TRAIN:
            raise SystemExit(f"{p}: {len(out[fam])} lines, expected {N_TRAIN}")
    return out


def qtext(rec):
    """the record's prompt text (state + every question's instructions): equal texts = the same training example"""
    return rec["state"] + "\n" + "\n".join(q.get("instructions", "") for q in rec["questions"].values())


def coin_key(rec):
    st, d = rec["meta"]["prior"]["stated"], rec["meta"]["data"]
    return (float(st["a"]), float(st["b"]), int(d["n"]), int(d["h"]))


def suite_coin_keys(suite):
    """coin problem keys (stated prior a, b; n, h) of the BC suite (PC's suite_index rule), or set()"""
    if not suite:
        return set()
    fps = set()
    for p in sorted(Path(suite).glob("test/*.jsonl")):
        for l in open(p, encoding="utf-8"):
            fp = (json.loads(l).get("meta") or {}).get("fingerprint") or ""
            if fp.startswith("coin|"):
                _, pr, da = fp.split("|", 2)
                pr, da = json.loads(pr), json.loads(da)
                if {"a", "b"} <= set(pr) and {"n", "h"} <= set(da):
                    fps.add((float(pr["a"]), float(pr["b"]), int(da["n"]), int(da["h"])))
    return fps


def check_pc_selection(pcnum, lines, idx):
    """PC's N files must be the registered B1 + B2 (manifest indices AND verbatim lines): B3 / B4 are then disjoint from N."""
    E = []
    man = json.loads((Path(pcnum) / "manifest.json").read_text())
    for fam in FAMS:
        fm = man["families"][fam]
        if fm["B1"] != idx[fam]["b1"] or fm["B2"] != idx[fam]["b2"]:
            E.append(f"{fam}: pc_numeric_v1's B1 / B2 are not the first 2,000 picks of the stream")
        want = "\n".join([lines[fam][i] for i in idx[fam]["b1"]] + [lines[fam][i] for i in idx[fam]["b2"]]) + "\n"
        got = (Path(pcnum) / "N/train" / f"numeric_{fam}.jsonl").read_text(encoding="utf-8")
        if got != want:
            E.append(f"{fam}: pc_numeric_v1 N/train/numeric_{fam}.jsonl is not B1 + B2 verbatim")
        rel = f"N/train/numeric_{fam}.jsonl"
        if man["files"][rel]["sha256"] != sha256_file(Path(pcnum) / rel):
            E.append(f"{rel}: sha256 differs from pc_numeric_v1's manifest")
    return E


def diagnostics(lines, idx, suite_keys):
    """descriptive: truth model, folded share, coin problems that recur in the suite, exact-text repeats across blocks"""
    out = {}
    for fam in FAMS:
        d = {}
        texts = {b: Counter(qtext(json.loads(lines[fam][i])) for i in idx[fam][b]) for b in BLOCKS}
        for b in BLOCKS:
            recs = [json.loads(lines[fam][i]) for i in idx[fam][b]]
            broad = sum(r["meta"]["truth"].get("source") == "broad" for r in recs)
            folded = sum(max(r["meta"]["exact"].get("folded_low", 0) or 0, r["meta"]["exact"].get("folded_high", 0) or 0) > 0.25
                         for r in recs)
            ent = {"items": len(recs), "truth_broad_share": round(broad / len(recs), 4), "folded_gt_25pct": folded}
            if fam == "coin":
                ent["coin_problems_in_suite"] = sum(coin_key(r) in suite_keys for r in recs) if suite_keys else None
            others = Counter()
            for b2 in BLOCKS:
                if b2 != b:
                    others.update(texts[b2])
            ent["texts_repeated_in_other_blocks"] = sum(c for t, c in texts[b].items() if t in others)
            d[b] = ent
        out[fam] = d
    return out


def build(numeric, pcnum, out, suite=None, log=print):
    t0 = time.time()
    lines = read_train(numeric)
    idx = {fam: block_indices(fam) for fam in FAMS}
    E = check_pc_selection(pcnum, lines, idx)
    if E:
        raise SystemExit("pc_numeric_v1 is not PC's registered N: " + "; ".join(E))
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty: refusing to overwrite a build")
    (out / "X/train").mkdir(parents=True, exist_ok=True)
    man = {"version": VERSION, "namespace": NS_PC, "stream": "random.Random(f'{ns}|select|{fam}').sample(range(20000), 4000)",
           "python": sys.version.split()[0], "numeric_v2_train_sha256": dict(NUMERIC_TRAIN_SHA256), "n_per_block": N_PER,
           "blocks": {fam: {b: idx[fam][b] for b in BLOCKS} for fam in FAMS}, "files": {}}
    for fam in FAMS:
        for b in NEW_BLOCKS:
            p = out / "X/train" / file_name(fam, b)
            p.write_text("\n".join(lines[fam][i] for i in idx[fam][b]) + "\n", encoding="utf-8")
            man["files"][f"X/train/{file_name(fam, b)}"] = {"lines": N_PER, "sha256": sha256_file(p)}
    man["diagnostics"] = diagnostics(lines, idx, suite_coin_keys(suite))
    man["build_secs"] = round(time.time() - t0, 1)
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    log(f"built {len(man['files'])} files ({time.time() - t0:.0f} s)")
    return man


def check(out, numeric, pcnum, expect=None, log=print):
    """Every invariant, recomputed from the inputs: -> report dict; errors listed (empty = pass)."""
    out = Path(out)
    E = []
    lines = read_train(numeric)
    idx = {fam: block_indices(fam) for fam in FAMS}
    E += check_pc_selection(pcnum, lines, idx)
    man = json.loads((out / "manifest.json").read_text())
    for fam in FAMS:
        allb = [i for b in BLOCKS for i in idx[fam][b]]
        if len(set(allb)) != 4 * N_PER:
            E.append(f"{fam}: blocks overlap")
        for b in BLOCKS:
            if man["blocks"][fam][b] != idx[fam][b]:
                E.append(f"{fam} {b}: manifest indices are not the stream's")
        for b in NEW_BLOCKS:
            rel = f"X/train/{file_name(fam, b)}"
            p = out / rel
            if not p.exists():
                E.append(f"{rel}: missing"); continue
            got = p.read_text(encoding="utf-8")
            if got != "\n".join(lines[fam][i] for i in idx[fam][b]) + "\n":
                E.append(f"{rel}: not the numeric_v2 train lines of {b} verbatim (index order)")
            sh = sha256_file(p)
            if man["files"].get(rel, {}).get("sha256") != sh:
                E.append(f"{rel}: sha256 differs from the manifest")
            if expect is not None and expect.get(rel) != sh:
                E.append(f"{rel}: sha256 {sh[:12]} != the pinned {str(expect.get(rel))[:12]}")
            for l in got.splitlines():
                r = json.loads(l)
                if r["meta"]["family"] != fam or r["meta"]["split"] != "train" or not str(r["license"]).startswith("synthetic"):
                    E.append(f"{rel}: a record that is not a {fam} numeric_v2 train item"); break
    extra = sorted(str(p.relative_to(out)) for p in out.glob("X/train/*") if str(p.relative_to(out)) not in man["files"])
    if extra:
        E.append(f"unexpected files {extra}")
    if expect is not None and set(expect) != set(man["files"]):
        E.append(f"the pinned file set {sorted(expect)} != the build's {sorted(man['files'])}")
    rep = {"version": VERSION, "errors": E, "files": man["files"], "python": sys.version.split()[0]}
    log(f"check: {len(E)} errors" + (": " + "; ".join(E[:5]) if E else ""))
    return rep


def arm(name, replay, pcnum, pcd, out, log=print):
    """the arm's train / val dirs (symlinks to the verified files) -> summary dict"""
    if name not in ARM_SPEC:
        raise SystemExit(f"arm must be one of {sorted(ARM_SPEC)}")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    (out / "train").mkdir(parents=True, exist_ok=True); (out / "val").mkdir(parents=True, exist_ok=True)
    nfams, blocks, want = ARM_SPEC[name]
    srcs = sorted(Path(replay, "R/train").glob("*.jsonl"))
    srcs += [Path(pcnum, "N/train", f"numeric_{f}.jsonl") for f in nfams]
    srcs += [Path(pcd, "X/train", file_name(f, b)) for f, b in blocks]
    names = [p.name for p in srcs]
    if len(set(names)) != len(names):
        raise SystemExit(f"file-name collision in arm {name}")
    for p in srcs:
        if not p.exists():
            raise SystemExit(f"arm {name}: missing {p}")
        os.symlink(os.path.abspath(p), out / "train" / p.name)
    for p in sorted(Path(replay, "val").glob("*.jsonl")):
        os.symlink(os.path.abspath(p), out / "val" / p.name)
    fam_units, lines_total = Counter(), 0
    for p in srcs:
        for l in open(p, encoding="utf-8"):
            lines_total += 1
            r = json.loads(l)
            if str(r.get("source", "")).startswith("numeric/"):
                fam_units[(r.get("meta") or {}).get("family")] += 1
    held = {f: fam_units.get(f, 0) for f in HELD_OUT[name]}
    if any(held.values()):
        raise SystemExit(f"arm {name}: held-out family items present {held}")
    s = {"arm": name, "files": len(srcs), "numeric_units_by_family": dict(sorted(fam_units.items())),
         "numeric_units": sum(fam_units.values()), "expected_units": want, "held_out": list(HELD_OUT[name])}
    log(json.dumps(s))
    return s


# ============================================================================ self-test (no data)
def self_test():
    for fam in FAMS:
        b = block_indices(fam)
        allb = [i for k in BLOCKS for i in b[k]]
        assert len(allb) == len(set(allb)) == 4 * N_PER and all(0 <= i < N_TRAIN for i in allb)
        assert all(b[k] == sorted(b[k]) and len(b[k]) == N_PER for k in BLOCKS)
        ref = random.Random(f"{NS_PC}|select|{fam}").sample(range(N_TRAIN), 2 * N_PER)
        assert b["b1"] == sorted(ref[:N_PER]) and b["b2"] == sorted(ref[N_PER:])          # = PC's select_indices
        assert block_indices(fam) == b                                                   # deterministic
    for name, (nf, bl, want) in ARM_SPEC.items():     # replay + 2,000 per N family + 1,000 per (family, new block) = the expected units
        assert want == REPLAY_UNITS + 2 * N_PER * len(nf) + N_PER * len(bl), name
    assert ARM_SPEC["d"][2] == REPLAY_UNITS + 12 * N_PER and ARM_SPEC["h"][2] == REPLAY_UNITS + 6 * N_PER
    assert sum(1 for f, _ in ARM_SPEC["h"][1] if f == "normal_mean") == 0 and "normal_mean" not in ARM_SPEC["h"][0]
    print(f"self-test passed (stream prefix = PC's B1 / B2 on Python {sys.version.split()[0]}; blocks disjoint; arm specs)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--numeric", required=True); b.add_argument("--pcnum", required=True)
    b.add_argument("--out", required=True); b.add_argument("--suite")
    c = sub.add_parser("check"); c.add_argument("out"); c.add_argument("--numeric", required=True); c.add_argument("--pcnum", required=True)
    c.add_argument("--expect", help="json {relative file: sha256} (the task script's pins)"); c.add_argument("--json")
    a = sub.add_parser("arm"); a.add_argument("name"); a.add_argument("--replay", required=True); a.add_argument("--pcnum", required=True)
    a.add_argument("--pcd", required=True); a.add_argument("--out", required=True)
    sub.add_parser("self-test")
    x = ap.parse_args(argv)
    if x.cmd == "build":
        build(x.numeric, x.pcnum, x.out, x.suite)
        rep = check(x.out, x.numeric, x.pcnum)
        return 1 if rep["errors"] else 0
    if x.cmd == "check":
        exp = json.loads(x.expect) if x.expect else None
        rep = check(x.out, x.numeric, x.pcnum, exp)
        if x.json:
            Path(x.json).write_text(json.dumps(rep, indent=1))
        return 1 if rep["errors"] else 0
    if x.cmd == "arm":
        arm(x.name, x.replay, x.pcnum, x.pcd, x.out)
        return 0
    return self_test()


if __name__ == "__main__":
    raise SystemExit(main())
