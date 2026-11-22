#!/usr/bin/env python3
"""(89) CROSS-BACKBONE DISTILLATION helpers (fleet/task_89.sh; pre-registration: AGENT_NOTES.md, 2026-10-03 00:29Z).

    pool     P_real and rdval, the two real-data corpora (89) adds to the (46) recipe:
               P_real  11,000 train + 800 val units from jev_rd2 TRAIN minus jev_rd1 TRAIN. rd2 samples newest-first per stratum
                       like rd1, so the remainder is OLDER real decisions no teacher trained on. A candidate record is dropped if
                       its record id OR its normalised state (analysis/claude_overlap.norm) is in rd1 train / val or in any
                       --against set (the real-world / GDE-1 eval splits, jev_real test, the teachers' benchmark training data,
                       P_bench), if it is non-commercial (cache_features.is_noncommercial: the clean line), or if a state already
                       taken repeats. Per-source unit quotas mirror rd1 TRAIN's source mix (water-filled where a source runs
                       short); inside a source the records come in seeded-hash order, val first, then train. Written in the
                       jev_dist46pool layout ({train,val}/pool.jsonl; qids made globally unique as "record_id|qid", so a
                       scorer row maps back to its record; the per-question fields rewritten with them).
               rdval   the stack's fourth held-out set: jev_rd2 VAL, deduped the same way (plus against P_real), then a seeded
                       1,400-unit sample drawn exactly as claude_lmscore.py --sample 1400 --seed 300 draws the other three
                       held-out sets (np.random.default_rng(300 + 7) over the units in iter_units order), written as
                       val/<source>.jsonl with the records cut to their sampled questions (original qids). Scoring it with
                       --sample 1400 --seed 300 is then a no-op that keeps every unit.
    summary  the P89-1 readout: per stack (S_x, S_q, S_g) the stack weights and temperature (posterior mean, 90% interval, ML)
             and the held-out log score per set (jdval, extval, ext2val, rdval, the three bench sets pooled, all four), plus the
             paired S_x - S_q and S_x - S_g log-score differences with unit-bootstrap 95% intervals.

    python3 analysis/claude_d89.py pool --rd1 D/jev_rd1 --rd2 D/jev_rd2 --against jev_real_test=D/jev_real/test ... \\
        --out-real D/jev_d89real --out-rdval D/jev_d89rdval --report analysis/claude_d89_pool.json
    python3 analysis/claude_d89.py summary --stack x=claude_d89_stack_x.json --stack q=... --stack g=... \\
        --npz-fmt 'analysis/claude_d89_hv_{set}_{t}.npz' --out analysis/claude_d89_stacks.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from claude_overlap import norm  # noqa: E402  the eval-overlap normalisation, as everywhere else


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


cf = _load("cf", os.path.join(os.path.dirname(HERE), "trainers", "cache_features.py"))
KMAX_LETTERS = 52
PER_Q = ("questions", "labels", "label_kind", "label_source")   # record fields keyed by question id


def shash(state):
    return hashlib.sha1(norm(state).encode("utf-8")).digest()


def rhash(seed, s):
    return hashlib.sha1(f"{seed}|{s}".encode()).hexdigest()


def jsonl_files(path):
    if os.path.isfile(path):
        return [path]
    fs = sorted(glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True))
    return [f for f in fs if not os.path.basename(f).startswith("._")]


def read_jsonl(p, bad):
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                bad[p] += 1


def nq(r):
    return len(r["questions"])


def lic(r):
    return str(r.get("license") or (r.get("meta") or {}).get("license") or "")


# --------------------------------------------------------------------------- pool
class Against:
    """Record ids and normalised-state hashes of the sets candidates must not overlap, one pair of sets per name."""

    def __init__(self):
        self.ids, self.states, self.rep = {}, {}, {}

    def add(self, name, path):
        ids, sts, n, nostate, bad = set(), set(), 0, 0, Counter()
        files = jsonl_files(path)
        if not files:
            sys.exit(f"--against {name}={path}: no *.jsonl there")
        for p in files:
            for r in read_jsonl(p, bad):
                n += 1
                if "id" in r:
                    ids.add(str(r["id"]))
                if "state" in r:
                    sts.add(shash(r["state"]))
                else:
                    nostate += 1
        self.ids[name], self.states[name] = ids, sts
        self.rep[name] = {"path": path, "files": len(files), "records": n, "ids": len(ids), "states": len(sts),
                          "records_without_state": nostate, "bad_lines": sum(bad.values())}
        print(f"against {name:28s} {n:7d} records, {len(ids):7d} ids, {len(sts):7d} states  ({len(files)} files: {path})",
              flush=True)

    def hit(self, r, sh):
        rid = str(r["id"])
        for name in self.ids:
            if rid in self.ids[name]:
                return f"id:{name}"
            if sh in self.states[name]:
                return f"state:{name}"
        return None


def load_split(root, split):
    """{source stem: [records]} of <root>/<split>/<stem>.jsonl, in file order."""
    out, bad = {}, Counter()
    for p in sorted(glob.glob(os.path.join(root, split, "*.jsonl"))):
        if os.path.basename(p).startswith("._"):
            continue
        out[os.path.basename(p)[: -len(".jsonl")]] = list(read_jsonl(p, bad))
    if bad:
        sys.exit(f"unparseable lines in {root}/{split}: {dict(bad)}")
    return out


def k_filter(r):
    """The record with every question whose option count exceeds the letter set dropped (claude_distill46 pool's rule), or
    None if none is left. Returns (record, n_questions_dropped)."""
    keep = []
    for qid, q in r["questions"].items():
        keys, _, _ = cf.question_options(q, r["labels"][qid])
        if len(keys) <= KMAX_LETTERS:
            keep.append(qid)
    if len(keep) == len(r["questions"]):
        return r, 0
    if not keep:
        return None, len(r["questions"])
    r2 = dict(r)
    for f in PER_Q:
        if isinstance(r.get(f), dict):
            r2[f] = {q: r[f][q] for q in keep if q in r[f]}
    return r2, len(r["questions"]) - len(keep)


def clean(cands, rd1_ids, rd1_states, against, taken_states, rep_drop):
    """Filter {stem: [records]} in place order; returns {stem: [kept records]}. Drop reasons are counted per source (records,
    units); the first reason that applies is the one counted."""
    out = {}
    for stem, recs in cands.items():
        kept = []
        for r in recs:
            sh = shash(r["state"])
            why = None
            if str(r["id"]) in rd1_ids:
                why = "rd1_id"
            elif sh in rd1_states:
                why = "rd1_state"
            else:
                why = against.hit(r, sh)
            if why is None and cf.is_noncommercial({"source": r.get("source") or stem, "license": lic(r)}):
                why = "noncommercial"
            if why is None:
                r2, nk = k_filter(r)
                if r2 is None:
                    why = "k_gt_52_all"
                else:
                    if nk:
                        rep_drop[stem]["units:k_gt_52_questions"] += nk
                    r = r2
            if why is None and sh in taken_states:
                why = "dup_state"
            if why is not None:
                rep_drop[stem][f"records:{why}"] += 1
                rep_drop[stem][f"units:{why}"] += nq(r)
                continue
            taken_states.add(sh)
            kept.append(r)
        out[stem] = kept
    return out


def water_fill(budget, avail, weight):
    """Integer shares of `budget` over keys, proportional to weight[k], none above avail[k] (excess redistributed;
    build_jev_rd1.water_fill verbatim)."""
    out, left, active = {}, budget, {k for k in avail if avail[k] > 0}
    while active and left > 0:
        tot = sum(weight[k] for k in active)
        share = {k: left * weight[k] / tot for k in active}
        full = {k for k in active if share[k] >= avail[k]}
        if full:
            for k in full:
                out[k] = avail[k]
                left -= avail[k]
            active -= full
            continue
        base = {k: int(share[k]) for k in active}
        rem = left - sum(base.values())
        for k in sorted(active, key=lambda k: (-(share[k] - base[k]), str(k)))[:rem]:
            base[k] += 1
        out.update(base)
        left = 0
    for k in avail:
        out.setdefault(k, 0)
    return out


def take(rs, budget, got=0):
    """build_jev_rd1._take: records in order while adding one brings the unit total closer to the budget."""
    out = []
    for r in rs:
        if got >= budget:
            break
        n = nq(r)
        if got + n <= budget or (budget - got) >= (got + n - budget):
            out.append(r)
            got += n
    return out, got


def rewrite_qids(r, stem):
    """jev_dist46pool's record layout: qids made globally unique ("record_id|qid") in every per-question field, so a scorer
    row (test_qid) maps back to its record; `source` defaults to the file stem (claude_distill46.py pool's rule)."""
    other = [f for f, v in r.items() if f not in PER_Q and isinstance(v, dict) and v and set(v) == set(r["questions"])]
    if other:
        sys.exit(f"record {r['id']} has per-question fields {other} this rewrite does not know")
    r = dict(r)
    r.setdefault("source", stem)
    new = {}
    for f in PER_Q:
        if not isinstance(r.get(f), dict):
            continue
        new[f] = {f"{r['id']}|{qid}": r[f][qid] for qid in r["questions"] if qid in r[f]}
    for f in ("questions", "labels", "label_kind"):
        if len(new.get(f, {})) != len(r["questions"]):
            sys.exit(f"record {r['id']}: field {f} does not cover every question")
    r.update(new)
    return r


def date_of(r):
    m = r.get("meta") or {}
    d = (m.get("rd1") or {}).get("date") if isinstance(m, dict) else None
    return str(d) if d else None


def date_range(recs):
    ds = sorted(d for d in (date_of(r) for r in recs) if d)
    return [ds[0], ds[-1], ds[len(ds) // 2]] if ds else None


def validate(root, splits):
    """The trainer's loading path (cache_features.iter_units) over what was written: units per split, NC units, K > 52,
    invalid targets, duplicate qids."""
    lm_labels = KMAX_LETTERS
    res = {}
    for split in splits:
        units = list(cf.iter_units(root, [split], set(), None))
        qids = Counter(str(u["qid"]) for u in units)
        res[split] = {"units": len(units),
                      "noncommercial": sum(1 for u in units if cf.is_noncommercial(u)),
                      "k_gt_52": sum(1 for u in units if len(u["option_keys"]) > lm_labels),
                      "bad_target": sum(1 for u in units if not (sum(u["target"]) > 0 and min(u["target"]) >= 0)),
                      "duplicate_qids": sum(c - 1 for c in qids.values() if c > 1),
                      "per_source": dict(Counter(str(u["source"]).split("/")[0] for u in units).most_common())}
    return res


def cmd_pool(a):
    rep = {"version": "d89 pool 1.0 (2026-10-03)", "seed": a.seed, "rdval_seed": a.rdval_seed,
           "n_train": a.n_train, "n_val": a.n_val, "n_rdval": a.n_rdval}
    # ---- what candidates must not overlap
    rd1 = {s: load_split(a.rd1, s) for s in ("train", "val")}
    rd1_ids = {str(r["id"]) for s in rd1.values() for rs in s.values() for r in rs}
    rd1_states = {shash(r["state"]) for s in rd1.values() for rs in s.values() for r in rs}
    rd1_units = {stem: sum(nq(r) for r in rs) for stem, rs in rd1["train"].items()}
    rep["rd1"] = {"train_units": rd1_units, "train_records": {k: len(v) for k, v in rd1["train"].items()},
                  "val_units": {stem: sum(nq(r) for r in rs) for stem, rs in rd1["val"].items()},
                  "ids": len(rd1_ids), "states": len(rd1_states),
                  "train_dates": {k: date_range(v) for k, v in rd1["train"].items()}}
    print(f"rd1: {sum(rd1_units.values())} train units over {len(rd1_units)} sources; {len(rd1_ids)} ids, "
          f"{len(rd1_states)} states (train + val)", flush=True)
    for nm, got, want in (("rd1 train", sum(rd1_units.values()), a.expect_rd1_train_units),):
        if want and got != want:
            sys.exit(f"{nm}: {got} units, expected {want} (a stale or partial copy?)")
    against = Against()
    for spec in a.against:
        name, path = spec.split("=", 1)
        against.add(name, path)
    rep["against"] = against.rep

    # ---- P_real: rd2 TRAIN minus rd1, deduped
    rd2_train = load_split(a.rd2, "train")
    rd2_val = load_split(a.rd2, "val")
    for nm, got, want in (("rd2 train", sum(nq(r) for rs in rd2_train.values() for r in rs), a.expect_rd2_train_units),
                          ("rd2 val", sum(nq(r) for rs in rd2_val.values() for r in rs), a.expect_rd2_val_units)):
        if want and got != want:
            sys.exit(f"{nm}: {got} units, expected {want} (a stale or partial copy?)")
        print(f"{nm}: {got} units", flush=True)
    missing = sorted(set(rd1_units) - set(rd2_train))
    if missing:
        sys.exit(f"rd1 sources missing from rd2 train: {missing}")
    drop = defaultdict(Counter)
    taken_states = set()
    cand = clean(rd2_train, rd1_ids, rd1_states, against, taken_states, drop)
    avail = {s: sum(nq(r) for r in rs) for s, rs in cand.items()}
    weight = {s: rd1_units.get(s, 0) for s in cand}
    total = a.n_train + a.n_val
    quota = water_fill(total, avail, weight)                       # rd1's TRAIN source mix, capped by what is left
    vq = water_fill(a.n_val, quota, quota)                         # val's share of each source's quota (same mix)
    tq = {s: quota[s] - vq[s] for s in quota}
    chosen = {"train": [], "val": []}
    per = {}
    for s in sorted(cand):
        order = sorted(cand[s], key=lambda r: rhash(a.seed, r["id"]))
        v, vu = take(order, vq[s])
        vids = {r["id"] for r in v}
        t, tu = take([r for r in order if r["id"] not in vids], tq[s])
        chosen["val"] += [rewrite_qids(r, s) for r in v]
        chosen["train"] += [rewrite_qids(r, s) for r in t]
        per[s] = {"rd2_train_records": len(rd2_train[s]), "rd2_train_units": sum(nq(r) for r in rd2_train[s]),
                  "dropped": dict(drop[s]), "available_units": avail[s], "available_records": len(cand[s]),
                  "rd1_train_units": rd1_units.get(s, 0), "quota_units": quota[s], "quota_train": tq[s], "quota_val": vq[s],
                  "train_units": tu, "train_records": len(t), "val_units": vu, "val_records": len(v),
                  "train_dates": date_range(t), "short": quota[s] < round(total * weight[s] / max(sum(weight.values()), 1))}
        print(f"  {s:14s} rd2 train {per[s]['rd2_train_units']:6d} u -> available {avail[s]:6d} u  quota {quota[s]:4d} "
              f"(train {tq[s]}, val {vq[s]})  took train {tu} / val {vu}  dropped {dict(drop[s])}", flush=True)
    rep["p_real"] = {"per_source": per,
                     "dropped_total": dict(sum((drop[s] for s in drop), Counter())),
                     "train_units": sum(p["train_units"] for p in per.values()),
                     "val_units": sum(p["val_units"] for p in per.values()),
                     "train_records": len(chosen["train"]), "val_records": len(chosen["val"])}
    rng = random.Random(a.seed)
    for split in ("train", "val"):
        rng.shuffle(chosen[split])
        os.makedirs(os.path.join(a.out_real, split), exist_ok=True)
        with open(os.path.join(a.out_real, split, "pool.jsonl"), "w", encoding="utf-8") as f:
            for r in chosen[split]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tr_states = {shash(r["state"]) for r in chosen["train"]}
    rep["p_real"]["val_states_in_train"] = sum(1 for r in chosen["val"] if shash(r["state"]) in tr_states)
    print(f"P_REAL train {rep['p_real']['train_units']} units / {len(chosen['train'])} records, val "
          f"{rep['p_real']['val_units']} / {len(chosen['val'])}; dropped {rep['p_real']['dropped_total']}", flush=True)

    # ---- rdval: rd2 VAL, deduped the same way and against P_real, then the scorer's seeded sample
    pr_ids = {str(r["id"]) for sp in chosen.values() for r in sp}
    pr_states = {shash(r["state"]) for sp in chosen.values() for r in sp}
    against.ids["p_real"], against.states["p_real"] = pr_ids, pr_states
    vdrop = defaultdict(Counter)
    vcand = clean(rd2_val, rd1_ids, rd1_states, against, set(), vdrop)
    units = [(s, i, qid) for s in sorted(vcand) for i, r in enumerate(vcand[s]) for qid in r["questions"]]
    if len(units) < a.n_rdval:
        sys.exit(f"rdval: only {len(units)} candidate units for {a.n_rdval}")
    pick = sorted(np.random.default_rng(a.rdval_seed + 7).choice(len(units), a.n_rdval, replace=False))
    keep = defaultdict(lambda: defaultdict(list))
    for j in pick:
        s, i, qid = units[j]
        keep[s][i].append(qid)
    os.makedirs(os.path.join(a.out_rdval, "val"), exist_ok=True)
    vper = {}
    for s in sorted(vcand):
        out = []
        for i, qs in sorted(keep[s].items()):
            r = dict(vcand[s][i])
            for f in PER_Q:
                if isinstance(r.get(f), dict):
                    r[f] = {q: r[f][q] for q in r["questions"] if q in qs and q in r[f]}
            out.append(r)
        if out:
            with open(os.path.join(a.out_rdval, "val", f"{s}.jsonl"), "w", encoding="utf-8") as f:
                for r in out:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        vper[s] = {"rd2_val_units": sum(nq(r) for r in rd2_val.get(s, [])), "dropped": dict(vdrop[s]),
                   "available_units": sum(nq(r) for r in vcand[s]), "sampled_units": sum(len(q) for q in keep[s].values()),
                   "sampled_records": len(out)}
    rep["rdval"] = {"per_source": vper, "candidate_units": len(units), "sampled_units": len(pick),
                    "dropped_total": dict(sum((vdrop[s] for s in vdrop), Counter()))}
    print(f"RDVAL {len(pick)} units sampled (seed {a.rdval_seed} + 7) of {len(units)} candidates; dropped "
          f"{rep['rdval']['dropped_total']}", flush=True)
    print("  " + ", ".join(f"{s} {v['sampled_units']}" for s, v in vper.items()), flush=True)

    # ---- the trainer's / scorer's own view of what was written
    rep["validation"] = {"p_real": validate(a.out_real, ("train", "val")), "rdval": validate(a.out_rdval, ("val",))}
    v = rep["validation"]
    ok = (v["p_real"]["train"]["noncommercial"] == v["p_real"]["val"]["noncommercial"] == v["rdval"]["val"]["noncommercial"] == 0
          and v["p_real"]["train"]["k_gt_52"] == v["p_real"]["val"]["k_gt_52"] == v["rdval"]["val"]["k_gt_52"] == 0
          and v["p_real"]["train"]["bad_target"] == v["p_real"]["val"]["bad_target"] == v["rdval"]["val"]["bad_target"] == 0
          and v["p_real"]["train"]["duplicate_qids"] == v["p_real"]["val"]["duplicate_qids"] == 0
          and v["rdval"]["val"]["units"] == a.n_rdval and rep["p_real"]["val_states_in_train"] == 0)
    rep["validation_ok"] = bool(ok)
    with open(a.report, "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print(f"VALIDATE p_real train {v['p_real']['train']['units']} / val {v['p_real']['val']['units']} units, rdval "
          f"{v['rdval']['val']['units']}; NC {v['p_real']['train']['noncommercial']}/{v['p_real']['val']['noncommercial']}/"
          f"{v['rdval']['val']['noncommercial']}, K>52 0?, bad targets 0?, dup qids "
          f"{v['p_real']['train']['duplicate_qids']}/{v['p_real']['val']['duplicate_qids']} -> {'OK' if ok else 'FAILED'}",
          flush=True)
    if not ok:
        sys.exit("POOL_VALIDATION_FAILED (see the report)")


# --------------------------------------------------------------------------- summary (P89-1)
def cmd_summary(a):
    d46 = _load("claude_distill46", os.path.join(HERE, "claude_distill46.py"))
    sets = a.sets.split(",")
    bench = a.bench_sets.split(",")
    stacks, per_unit = {}, {}
    for spec in a.stack:
        nm, path = spec.split("=", 1)
        st = json.load(open(path))
        w, tau = np.array(st["w"]), float(st["tau"])
        teachers = st["teachers"]
        row = {k: st[k] for k in ("teachers", "w", "w_q05", "w_q95", "w_ml", "tau", "tau_q05", "tau_q95", "tau_ml", "units")}
        row["weights"] = dict(zip(teachers, st["w"]))
        row["held_out"] = {}
        per_unit[nm] = {}
        for s in sets:
            P, T, mask = d46.load_set([a.npz_fmt.format(set=s, t=t) for t in teachers])
            lq = d46.pool_logq(np.log(np.clip(P, 1e-12, 1)), mask, w, tau)
            ls = (T * np.where(mask, lq, 0)).sum(1)
            per_unit[nm][s] = ls
            row["held_out"][s] = d46.metrics(lq, T, mask)
            ref = st["rows"]["teacher"].get(s, {}).get("log_score")
            if ref is not None and abs(ref - row["held_out"][s]["log_score"]) > 1e-6:
                sys.exit(f"{nm} {s}: recomputed log score {row['held_out'][s]['log_score']} != stack json {ref}")
        for pooled, members in (("bench", bench), ("all", sets)):
            row["held_out"][pooled] = {"log_score": float(np.concatenate([per_unit[nm][s] for s in members]).mean()),
                                       "units": int(sum(len(per_unit[nm][s]) for s in members))}
        stacks[nm] = row
    rng = np.random.default_rng(a.seed)
    comp = {}
    if a.cross in stacks:
        for other in stacks:
            if other == a.cross:
                continue
            c = {}
            for key, members in [(s, [s]) for s in sets] + [("bench", bench), ("all", sets)]:
                dx = np.concatenate([per_unit[a.cross][s] for s in members])
                do = np.concatenate([per_unit[other][s] for s in members])
                if dx.shape != do.shape:
                    sys.exit(f"{a.cross} vs {other} on {key}: different unit counts {dx.shape} {do.shape}")
                d = dx - do
                bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(a.boot)])
                c[key] = {"diff": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                          "units": int(len(d))}
            comp[f"{a.cross}-{other}"] = c
    out = {"stacks": stacks, "paired_log_score_diff": comp,
           "note": "log score = mean over units of sum_k t_k log q_k (higher is better); held-out = the sets the stacks are "
                   "fitted on (in-sample for the stack's 1-2 weights + temperature); CIs: unit bootstrap, " f"{a.boot} draws"}
    singles = [o for o in stacks if o != a.cross]
    if a.cross in stacks and singles:
        w_ok = all(v >= 0.25 for v in stacks[a.cross]["w"])
        beat = {k: all(comp[f"{a.cross}-{o}"][k]["diff"] > 0 for o in singles)
                for k in ("bench", "rdval") if k in comp[f"{a.cross}-{singles[0]}"]}
        out["P89_1"] = {"each_weight_ge_0.25": w_ok, "beats_every_single_stack": beat,
                        "holds_point_estimates": bool(w_ok and len(beat) == 2 and all(beat.values()))}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    for nm, r in stacks.items():
        print(f"STACK {nm}: w {dict(zip(r['teachers'], np.round(r['w'], 3)))} (90% {np.round(r['w_q05'], 3)} .. "
              f"{np.round(r['w_q95'], 3)})  tau {r['tau']:.3f} [{r['tau_q05']:.3f}, {r['tau_q95']:.3f}]  log score "
              + "  ".join(f"{k} {v['log_score']:+.4f}" for k, v in r["held_out"].items()), flush=True)
    for k, c in comp.items():
        print(f"DIFF {k}: " + "  ".join(f"{s} {v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" for s, v in c.items()),
              flush=True)
    if "P89_1" in out:
        print(f"P89-1 (point estimates): {out['P89_1']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--rd1", required=True)
    p.add_argument("--rd2", required=True)
    p.add_argument("--against", action="append", default=[], help="NAME=PATH (a .jsonl or a directory, read recursively)")
    p.add_argument("--out-real", required=True)
    p.add_argument("--out-rdval", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--n-train", type=int, default=11000)
    p.add_argument("--n-val", type=int, default=800)
    p.add_argument("--n-rdval", type=int, default=1400)
    p.add_argument("--seed", type=int, default=89)
    p.add_argument("--rdval-seed", type=int, default=300, help="as the scorer's --seed for the other held-out samples")
    p.add_argument("--expect-rd1-train-units", type=int, default=0, help="fail unless rd1 train has this many units (0 = off)")
    p.add_argument("--expect-rd2-train-units", type=int, default=0)
    p.add_argument("--expect-rd2-val-units", type=int, default=0)
    p = sub.add_parser("summary")
    p.add_argument("--stack", action="append", required=True, help="NAME=stack.json (claude_distill46.py stack output)")
    p.add_argument("--npz-fmt", required=True, help="held-out teacher npz path, with {set} and {t}")
    p.add_argument("--sets", default="jdval,extval,ext2val,rdval")
    p.add_argument("--bench-sets", default="jdval,extval,ext2val")
    p.add_argument("--cross", default="x", help="the cross-backbone stack's NAME")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=89)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"pool": cmd_pool, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
