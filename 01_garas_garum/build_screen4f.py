#!/usr/bin/env python3
"""(SCREEN4F) Data for the registered SCREEN4 follow-up (tmp/screen4f/prereg.md): arm C with the GARAS-PARENT preservation weight raised
from 0.3 to 1.0 (arms P and LP). A thin wrapper around the deployed SCREEN4 builder (build_screen4.py, md5-pinned by the task script):
the same records, R2 subsets, teacher npz files (SCREEN4's teach pass; NOT re-run), the same EG teacher weight (0.3) and temperature fit;
only the replay units' targets change, from (t + 0.3 q) / 1.3 to (t + 1.0 q) / 2.

  mixp   build_screen4.build_mixed_arms(..., lam_parent=1.0, lam_teacher=0.3) into OUT (a copy of the prep tarball's arms dir), then
         build_screen4.check_arms; with --reference (SCREEN4's published mixed arms dir) also verifies, unit by unit, that
           * every non-replay record of arm c is byte-identical to SCREEN4's arm c (EG targets unchanged), and
           * every replay unit moved from SCREEN4's target exactly as the weight change implies: t_P - g = (1.3 / 0.6) (t_C - g)
             (same gold g, same parent row q; t_C = (g + 0.3 q) / 1.3, t_P = (g + q) / 2).

  python3 build_screen4f.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE / "analysis", HERE / "trainers"):
    sys.path.insert(0, str(_p))

VERSION = "build_screen4f.py v1 (2026-10-13)"
LAM_PARENT_P = 1.0
LAM_TEACHER_P = 0.3
LAM_C = 0.3
TOL = 1e-6


def reference_problems(out, ref, lam_p=LAM_PARENT_P, lam_c=LAM_C):
    """-> problems (empty = OK): arm c of OUT (weight lam_p) vs arm c of the SCREEN4 reference arms dir REF (weight lam_c), with the gold
    targets from OUT's arm b (the same records, plain): EG records byte-identical; every replay unit t_P - g == k (t_C - g),
    k = (1 + lam_c) / lam_c * lam_p / (1 + lam_p) (the same parent row q behind both)."""
    import build_screen4 as bs
    probs = []
    fo, fr = Path(out) / "arm_c" / "train" / "eg.jsonl", Path(ref) / "arm_c" / "train" / "eg.jsonl"
    if not fo.exists() or not fr.exists() or fo.read_bytes() != fr.read_bytes():
        probs.append("arm c EG records differ from SCREEN4's (the EG teacher weight must stay 0.3)")
    k = (1 + lam_c) / lam_c * lam_p / (1 + lam_p)
    moved = 0
    for stem in ("replay_R.jsonl", "replay_R2.jsonl"):
        fs = [Path(out) / "arm_c" / "train" / stem, Path(ref) / "arm_c" / "train" / stem, Path(out) / "arm_b" / "train" / stem]
        if not all(f.exists() for f in fs):
            probs.append(f"{stem}: missing ({[str(f) for f in fs if not f.exists()]})")
            continue
        ro, rr, rb = (bs.read_jsonl(f) for f in fs)
        if not len(ro) == len(rr) == len(rb):
            probs.append(f"{stem}: record counts {len(ro)} / {len(rr)} / {len(rb)} differ")
            continue
        bad = 0
        for x, y, b in zip(ro, rr, rb):
            if not bs._without_targets(x) == bs._without_targets(y) == bs._without_targets(b):
                bad += 1
                continue
            for q in b["questions"]:
                g, tc, tp = bs.gold_vector(b, q), bs.gold_vector(y, q), bs.gold_vector(x, q)
                s = sum(g)
                g = [v / s for v in g]
                if max(abs((p - gg) - k * (c - gg)) for p, c, gg in zip(tp, tc, g)) > 1e-6:
                    bad += 1
                    break
                moved += max(abs(p - c) for p, c in zip(tp, tc)) > 1e-6
        if bad:
            probs.append(f"{stem}: {bad} records are not SCREEN4's arm c with the parent weight {lam_c} -> {lam_p}")
    if not moved:
        probs.append("no replay unit moved: the parent weight change had no effect")
    return probs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("mixp",))
    ap.add_argument("--replay")
    ap.add_argument("--eg")
    ap.add_argument("--parent-npz")
    ap.add_argument("--teacher-npz")
    ap.add_argument("--teacher-dev-npz")
    ap.add_argument("--dev", action="append", default=[])
    ap.add_argument("--reference", help="SCREEN4's published mixed arms dir (s4_mixed_v1/arms): the record-by-record cross-check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    import build_screen4 as bs
    out = Path(a.out)
    if not (out / "arms_plain_manifest.json").exists():
        raise SystemExit("OUT must be a copy of the prep tarball's arms dir (arms_plain_manifest.json, teacher dirs)")
    man = bs.build_mixed_arms(a.replay, a.eg, out, a.parent_npz, a.teacher_npz, a.teacher_dev_npz, a.seed,
                              lam_parent=LAM_PARENT_P, lam_teacher=LAM_TEACHER_P)
    if (man["lam_parent"], man["lam_teacher"]) != (LAM_PARENT_P, LAM_TEACHER_P):
        raise SystemExit(f"weights {man['lam_parent']} / {man['lam_teacher']} are not the registered {LAM_PARENT_P} / {LAM_TEACHER_P}")
    probs = bs.check_arms(out, a.dev)
    if a.reference:
        probs += reference_problems(out, a.reference)
    if probs:
        raise SystemExit(f"MIXP_CHECK_FAILED {probs}")
    m = json.load(open(out / "arms_mixed_manifest.json"))
    m["screen4f"] = {"version": VERSION, "lam_parent": LAM_PARENT_P, "lam_teacher": LAM_TEACHER_P, "reference_checked": bool(a.reference)}
    (out / "arms_mixed_manifest.json").write_text(json.dumps(m, indent=1, sort_keys=True))
    print(f"MIXP_OK lam_parent {man['lam_parent']} lam_teacher {man['lam_teacher']} T {man['teacher_T']:.4f} stats {man['stats']} "
          f"reference_checked {bool(a.reference)}")
    return 0


def self_test():
    k = (1 + LAM_C) / LAM_C * LAM_PARENT_P / (1 + LAM_PARENT_P)
    g, q = [1.0, 0.0, 0.0], [0.5, 0.3, 0.2]
    tc = [(x + LAM_C * y) / (1 + LAM_C) for x, y in zip(g, q)]
    tp = [(x + LAM_PARENT_P * y) / (1 + LAM_PARENT_P) for x, y in zip(g, q)]
    assert max(abs((p - x) - k * (c - x)) for p, c, x in zip(tp, tc, g)) < 1e-12
    gr = [(c - p / k) / (1 - 1 / k) for c, p in zip(tc, tp)]
    assert max(abs(a - b) for a, b in zip(gr, g)) < 1e-12
    print("SELF_TEST_OK build_screen4f")
    return 0


if __name__ == "__main__":
    sys.exit(main())
