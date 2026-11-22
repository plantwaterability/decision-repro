#!/usr/bin/env python3
"""Independent label re-derivation for gap_supply_v1 (PILOT-DATA audit). Re-derives every gold from the VISIBLE state / options
text only (never from meta or the generator's code path), on a seeded sample per family, and compares with the stored labels.
  python3 verify_gap_labels.py DIR [--n 300]"""
import argparse
import json
import random
import re
import sys
from fractions import Fraction

PITCH = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


def argmax(d):
    m = max(d.values())
    return [k for k, v in d.items() if v == m]


def tool(r):
    s = r["state"]
    out = {}
    lock = "freeze_window=ACTIVE" in s
    cached = "Cached observation in context: None" not in s
    prov = json.loads(re.search(r"Provided arguments from user request: (\{.*?\})\n", s).group(1))
    tgt = re.search(r"Invoke `([^`]+)` now", r["questions"]["policy"]["criteria"]["call_target"]).group(1)
    req = None
    for line in s.splitlines():
        m = re.match(r"- `([^`]+)`: .*\| required=(\[.*?\]), optional", line)
        if m and m.group(1) == tgt:
            req = json.loads(m.group(2).replace("'", '"'))
    missing = [a for a in req if a not in prov]
    if lock:
        pol, dia = "refuse", "status_blocked"
    elif cached:
        pol, dia = "answer_direct", "status_cached"
    elif missing:
        pol, dia = "clarify", "status_missing"
    else:
        pol, dia = "call_target", "status_ready"
    out["policy"], out["diagnosis"] = pol, dia
    return out


def appliance(r):
    s = r["state"]
    cap = int(re.search(r"Breaker Cap: (\d+) kW", s).group(1))
    peak = "Tariff Period: PEAK" in s
    occ = "Occupancy: OCCUPIED" in s
    t_in = int(re.search(r"Indoor Temperature: (-?\d+) C", s).group(1))
    t_w = int(re.search(r"Tank Temperature: (\d+) C", s).group(1))
    soc = int(re.search(r"State of Charge: (\d+)%", s).group(1))
    laundry = "Laundry Queue: QUEUED" in s
    want = {"hvac": occ and not (18 <= t_in <= 25), "wh": t_w < 45, "ev": soc < 80 and ((not peak) or soc < 30), "dryer": laundry and not peak}
    kw = {"hvac": 4, "wh": 3, "ev": 7, "dryer": 5}
    on, tot = [], 0
    for d in ("hvac", "wh", "ev", "dryer"):
        if want[d] and tot + kw[d] <= cap:
            on.append(d)
            tot += kw[d]
    out = {f"dev_{d}": ("on" if d in on else "off") for d in kw}
    crit = r["questions"]["case_joint"]["criteria"]
    target = f"ON={on} ({tot} kW total)" if on else "NONE (all four appliances OFF; 0 kW total)"
    out["case_joint"] = [k for k, v in crit.items() if v == target]
    return out


def causal(r):
    s = r["state"]
    fr = {}
    for name, pat in (("qz", r"P\(Z = 1\) = [0-9.]+ \(([0-9/]+)\)"), ("px0", r"P\(X = 1 \| Z = 0\) = [0-9.]+ \(([0-9/]+)\)"),
                      ("px1", r"P\(X = 1 \| Z = 1\) = [0-9.]+ \(([0-9/]+)\)"), ("y00", r"X = 0, Z = 0\) = [0-9.]+ \(([0-9/]+)\)"),
                      ("y10", r"X = 1, Z = 0\) = [0-9.]+ \(([0-9/]+)\)"), ("y01", r"X = 0, Z = 1\) = [0-9.]+ \(([0-9/]+)\)"),
                      ("y11", r"X = 1, Z = 1\) = [0-9.]+ \(([0-9/]+)\)")):
        fr[name] = Fraction(re.search(pat, s).group(1))
    qz, q0 = fr["qz"], 1 - fr["qz"]
    px1 = q0 * fr["px0"] + qz * fr["px1"]
    px0 = q0 * (1 - fr["px0"]) + qz * (1 - fr["px1"])
    obs1 = (q0 * fr["px0"] * fr["y10"] + qz * fr["px1"] * fr["y11"]) / px1
    obs0 = (q0 * (1 - fr["px0"]) * fr["y00"] + qz * (1 - fr["px1"]) * fr["y01"]) / px0
    do1 = q0 * fr["y10"] + qz * fr["y11"]
    do0 = q0 * fr["y00"] + qz * fr["y01"]
    od, ad = obs1 - obs0, do1 - do0
    c = "pos_both" if od > 0 and ad > 0 else "neg_both" if od < 0 and ad < 0 else "obs_pos_do_neg" if od > 0 else "obs_neg_do_pos"
    return {"rung1_obs": float(obs1), "rung2_do": float(do1), "causal_contrast": c}


def music(r):
    s = r["state"]
    root = re.search(r"Bass Root Pitch Class on Downbeat: (\S+) ", s).group(1)
    pcs = set(x.strip() for x in re.search(r"Sustained Harmonic Pitch-Class Set \(.*?\): \{(.*?)\}", s).group(1).split(","))
    hits = []
    for k, v in r["questions"]["chord"]["criteria"].items():
        lab = v.split(" -- ")[0]
        opc = set(x.strip() for x in re.search(r"chord pitch classes: \{(.*?)\}", v).group(1).split(","))
        if lab.split(":")[0] == root and opc == pcs:
            hits.append(k)
    return {"chord": hits}


def pyexec(r):
    s = r["state"]
    code = re.search(r"```python\n(.*?)\n```", s, re.S).group(1)
    call = re.search(r"Target Call: `(.*?)`", s).group(1)
    ns = {}
    exec(code, {}, ns)
    val = repr(eval(call, {}, ns))
    return {"output": [k for k, v in r["questions"]["output"]["criteria"].items() if v == val]}


FN = {"synth_tool_policy": tool, "synth_appliance_sata_case": appliance, "synth_causal_scm_cladder": causal,
      "synth_music_harmony_pop909": music, "synth_pyexec_cruxeval": pyexec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    res, bad = {}, []
    for fam, fn in FN.items():
        rows = [json.loads(l) for l in open(f"{a.dir}/{fam}.jsonl")]
        smp = random.Random(20261014).sample(rows, min(a.n, len(rows)))
        n_q = n_ok = 0
        for r in smp:
            want = fn(r)
            for q, w in want.items():
                g = r["labels"][q]
                n_q += 1
                if isinstance(w, float):                      # soft target: exact probability of the first key (y1)
                    ok = abs(g["y1"] - w) <= 5e-7 and abs(g["y0"] - (1 - w)) <= 5e-7
                elif isinstance(w, list):                     # option keys whose text the independent derivation matches
                    ok = len(w) == 1 and argmax(g) == w
                else:
                    ok = argmax(g) == [w]
                n_ok += ok
                if not ok and len(bad) < 10:
                    bad.append({"family": fam, "id": r["id"], "q": q, "derived": w, "stored": argmax(g) if not isinstance(w, float) else g})
            # losses must agree with the policy gold where present
            lbo = (r.get("meta") or {}).get("loss_by_option") or {}
            for q, loss in lbo.items():
                n_q += 1
                lo = min(loss, key=loss.get)
                ok = argmax(r["labels"][q]) == [lo]
                n_ok += ok
        res[fam] = {"sampled_records": len(smp), "questions_checked": n_q, "agree": n_ok}
    print(json.dumps({"results": res, "disagreements": bad}, indent=1))
    sys.exit(0 if all(v["agree"] == v["questions_checked"] for v in res.values()) else 1)


if __name__ == "__main__":
    main()
