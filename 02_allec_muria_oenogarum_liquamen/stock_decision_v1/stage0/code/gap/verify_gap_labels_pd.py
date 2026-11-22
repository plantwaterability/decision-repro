#!/usr/bin/env python3
"""Independent label re-derivation for the PATCHED gap_supply_v1 (sdv1_gap_generators_pd). Re-derives every gold from the VISIBLE
state / options / question text only (never meta, never generator code paths) on a seeded sample per family.
Tool diagnosis (v1.1, review 2026-10-14 17:00 finding 4): every option of the diagnosis question is evaluated as an INDEPENDENT
predicate over the visible state, read from the option's own wording (the v1 "ready" wording -- arguments present and no lock --
is evaluated as written); a one-hot diagnosis gold must have EXACTLY ONE true option and the gold must be it. --all-tool checks
every tool record (not a sample) and reports the records with zero or several true options.
  python3 verify_gap_labels_pd.py DIR [--n 400] [--all-tool]"""
import argparse
import collections
import json
import random
import re
import sys
from fractions import Fraction

PITCH = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


def argmax(d):
    m = max(d.values())
    return [k for k, v in d.items() if v == m]


def tool_facts(r):
    """visible facts of a tool record: target tool, its required args / state_changing flag, provided args, lock, cached result."""
    s = r["state"]
    tgt = re.search(r"User request: run `([^`]+)` with arguments (\{.*?\})\n", s)
    name, prov = tgt.group(1), json.loads(tgt.group(2))
    req = mut = None
    for line in s.splitlines():
        m = re.match(r"- `([^`]+)`: .*\| required=(\[.*?\]) \| state_changing=(\w+)", line)
        if m and m.group(1) == name:
            req, mut = json.loads(m.group(2).replace("'", '"')), m.group(3) == "True"
    lock_line = re.search(r"Governance status: (.*)\n", s).group(1)
    lock_active = not lock_line.startswith("no active locks")
    ctx = re.search(r"Session context results: (.*)\n", s).group(1)
    cm = re.match(r"Verified result already in context \(\d+ s old\) for (\S+) with (\{.*?\}): status=OK", ctx)
    cached = bool(cm) and cm.group(1) == name and json.loads(cm.group(2)) == prov
    return {"name": name, "required": req, "state_changing": mut, "provided": prov, "missing": [x for x in req if x not in prov],
            "lock_applies": lock_active and mut, "cached": cached}


def diagnosis_truth(r):
    """{option key: bool} -- each diagnosis option evaluated independently from its own wording."""
    f = tool_facts(r)
    out = {}
    for k, text in r["questions"]["diagnosis"]["criteria"].items():
        t = text.lower()
        if k == "status_ready":
            v = not f["missing"] and not f["lock_applies"]
            if "not already verified" in t or "not yet verified" in t:
                v = v and not f["cached"]
            elif "no lock applies" not in t:
                raise ValueError(f"unknown ready wording: {text}")
        elif k == "status_missing":
            v = bool(f["missing"])
        elif k == "status_cached":
            v = f["cached"]
        elif k == "status_blocked":
            v = f["lock_applies"]
        else:
            raise ValueError(f"unknown diagnosis option {k}")
        out[k] = v
    return out


def tool(r):
    s = r["state"]
    tgt = re.search(r"User request: run `([^`]+)` with arguments (\{.*?\})\n", s)
    name, prov = tgt.group(1), json.loads(tgt.group(2))
    req = None
    for line in s.splitlines():
        m = re.match(r"- `([^`]+)`: .*\| required=(\[.*?\]) \| state_changing=(\w+)", line)
        if m and m.group(1) == name:
            req, mut = json.loads(m.group(2).replace("'", '"')), m.group(3) == "True"
    locked = "Governance status: no active locks" not in s
    cached = "Session context results: none relevant" not in s
    missing = [a for a in req if a not in prov]
    if locked and mut:
        return {"policy": "refuse", "diagnosis": "status_blocked"}
    if missing:
        return {"policy": "clarify", "diagnosis": "status_missing"}
    if cached:
        return {"policy": "answer_direct", "diagnosis": "status_cached"}
    return {"policy": "call_target", "diagnosis": "status_ready"}


def appliance(r):
    s = r["state"]
    g = lambda p: re.search(p, s).group(1)
    cap = int(g(r"Main breaker cap: (\d+) kW"))
    peak = "Tariff period: PEAK" in s
    occ = "Occupancy: OCCUPIED" in s
    t_in = int(g(r"Indoor temperature: (-?\d+) C"))
    lo, hi = map(int, re.search(r"comfort band when occupied: (\d+)\.\.(\d+) C", s).groups())
    t_w, tmin = int(g(r"Water tank temperature: (\d+) C")), int(g(r"minimum setpoint: (\d+) C"))
    soc = int(g(r"EV state of charge: (\d+)%"))
    tgt, urg = int(g(r"\(target (\d+)%")), int(g(r"urgent below (\d+)%"))
    laundry = "Laundry queue: QUEUED" in s
    kw = {d: int(re.search(rf"`{d}` \((\d+) kW\): wants ON", s).group(1)) for d in ("hvac", "wh", "ev", "dryer")}
    want = {"hvac": occ and not (lo <= t_in <= hi), "wh": t_w < tmin, "ev": soc < tgt and ((not peak) or soc < urg), "dryer": laundry and not peak}
    on, tot = [], 0
    for d in ("hvac", "wh", "ev", "dryer"):
        if want[d] and tot + kw[d] <= cap:
            on.append(d)
            tot += kw[d]
    out = {f"dev_{d}": ("on" if d in on else "off") for d in kw}
    target = f"ON={on} ({tot} kW total)" if on else "NONE (all four appliances OFF; 0 kW total)"
    out["case_joint"] = [k for k, v in r["questions"]["case_joint"]["criteria"].items() if v == target]
    return out


def causal(r):
    s = r["state"]
    pats = {"qz": r"P\(Z = 1\) = [0-9.]+ \(([0-9/]+)\)", "px0": r"P\(X = 1 \| Z = 0\) = [0-9.]+ \(([0-9/]+)\)",
            "px1": r"P\(X = 1 \| Z = 1\) = [0-9.]+ \(([0-9/]+)\)", "y00": r"X = 0, Z = 0\) = [0-9.]+ \(([0-9/]+)\)",
            "y10": r"X = 1, Z = 0\) = [0-9.]+ \(([0-9/]+)\)", "y01": r"X = 0, Z = 1\) = [0-9.]+ \(([0-9/]+)\)",
            "y11": r"X = 1, Z = 1\) = [0-9.]+ \(([0-9/]+)\)"}
    f = {k: Fraction(re.search(p, s).group(1)) for k, p in pats.items()}
    qz, q0 = f["qz"], 1 - f["qz"]
    px1 = q0 * f["px0"] + qz * f["px1"]
    px0 = q0 * (1 - f["px0"]) + qz * (1 - f["px1"])
    o1 = (q0 * f["px0"] * f["y10"] + qz * f["px1"] * f["y11"]) / px1
    o0 = (q0 * (1 - f["px0"]) * f["y00"] + qz * (1 - f["px1"]) * f["y01"]) / px0
    d1, d0 = q0 * f["y10"] + qz * f["y11"], q0 * f["y00"] + qz * f["y01"]
    od, ad = o1 - o0, d1 - d0
    c = "pos_both" if od > 0 and ad > 0 else "neg_both" if od < 0 and ad < 0 else "obs_pos_do_neg" if od > 0 else "obs_neg_do_pos"
    return {"rung1_obs": float(o1), "rung2_do": float(d1), "causal_contrast": c}


def music(r):
    dur = {}
    for m in re.finditer(r"onset [0-9.]+, dur ([0-9.]+), vel \d+: MIDI (\d+)", r["state"]):
        pc = int(m.group(2)) % 12
        dur[pc] = dur.get(pc, 0) + float(m.group(1))
    pcs = {PITCH[p] for p, d in dur.items() if d >= 1.0 - 1e-9}
    hits = []
    for k, v in r["questions"]["chord"]["criteria"].items():
        opc = {x.strip() for x in re.search(r"pitch classes \{(.*?)\}", v).group(1).split(",")}
        if opc == pcs:
            hits.append(k)
    return {"chord": hits}


def pyexec(r):
    s = r["state"]
    code = re.search(r"```python\n(.*?)\n```", s, re.S).group(1)
    call = re.search(r"Target call: `(.*?)`", s).group(1)
    ns = {}
    exec(code, ns)
    val = repr(eval(call, ns))
    return {"output": [k for k, v in r["questions"]["output"]["criteria"].items() if v == val]}


FN = {"synth_tool_policy": tool, "synth_appliance_sata_case": appliance, "synth_causal_scm_cladder": causal,
      "synth_music_harmony_pop909": music, "synth_pyexec_cruxeval": pyexec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--all-tool", action="store_true")
    a = ap.parse_args()
    res, bad = {}, []
    import os
    for fam, fn in FN.items():
        if not os.path.exists(f"{a.dir}/{fam}.jsonl"):
            continue
        rows = [json.loads(l) for l in open(f"{a.dir}/{fam}.jsonl")]
        smp = random.Random(20261014).sample(rows, min(a.n, len(rows)))
        nq = ok_n = 0
        for r in smp:
            for q, w in fn(r).items():
                g = r["labels"][q]
                nq += 1
                if isinstance(w, float):
                    ok = abs(g["y1"] - w) <= 1e-12 and abs(g["y0"] - (1 - w)) <= 1e-12
                elif isinstance(w, list):
                    ok = len(w) == 1 and argmax(g) == w
                else:
                    ok = argmax(g) == [w]
                ok_n += ok
                if not ok and len(bad) < 10:
                    bad.append({"family": fam, "id": r["id"], "q": q, "derived": w, "stored": g})
            for q, loss in ((r.get("meta") or {}).get("loss_by_option") or {}).items():
                nq += 1
                ok = argmax(r["labels"][q]) == [min(loss, key=loss.get)] and sorted(loss.values()).count(0.0) == 1
                ok_n += ok
        res[fam] = {"sampled_records": len(smp), "questions_checked": nq, "agree": ok_n}
        if fam == "synth_tool_policy":
            ex = collections.Counter()
            badx = []
            for r in (rows if a.all_tool else smp):
                tr = diagnosis_truth(r)
                true = sorted(k for k, v in tr.items() if v)
                gold = argmax(r["labels"]["diagnosis"])
                ok = len(true) == 1 and true == gold
                ex["exclusive_and_gold" if ok else ("several_true" if len(true) > 1 else ("none_true" if not true else "gold_not_true"))] += 1
                if not ok and len(badx) < 5:
                    badx.append({"id": r["id"], "true_options": true, "gold": gold})
            res[fam]["diagnosis_exclusivity"] = {"records": sum(ex.values()), **dict(ex), "examples": badx}
    print(json.dumps({"results": res, "disagreements": bad}, indent=1))
    excl_ok = all(v["diagnosis_exclusivity"]["records"] == v["diagnosis_exclusivity"].get("exclusive_and_gold", 0)
                  for v in res.values() if "diagnosis_exclusivity" in v)
    sys.exit(0 if all(v["agree"] == v["questions_checked"] for v in res.values()) and excl_ok else 1)


if __name__ == "__main__":
    main()
