#!/usr/bin/env python3
"""sdv1_gap_generators_pd.py -- PILOT-DATA patched copy of the SOTA pack's sdv1_gap_generators.py (the pack file is untouched).

Why a patched copy (gap_supply_v1 audit, ledger seq 33):
  * the pack derives every record's content from the record INDEX, not the seed: 1,500 records gave 20 distinct causal worlds and
    132 music measures, and the val split (another seed) was contained in train;
  * two families exposed the answer in the state: tool_policy printed a per-situation loss table whose only zero entry was the
    gold; music printed the sustained pitch-class set and the bass root.
What changed (same five families, same record / question / label schema, same target semantics and label sources):
  * every parameter comes from a per-record RNG seeded by sha256(seed | family | idx); records whose content (state, questions,
    labels) OR state repeats an earlier record of this run -- or of any --exclude-dir split -- are skipped and redrawn, so a split
    never repeats a state and val / discovery share no state with train (or with each other when chained);
  * synth_tool_policy: six domains, random argument values, random missing argument, phrasing variants, irrelevant context; ONE
    fixed, situation-independent penalty schedule is stated in every state, so the model must recognise the situation (missing
    argument / cached verified result / policy lock / ready) before choosing; loss_by_option is the schedule applied to the
    actual situation (argmin = gold). Every record carries meta.td_workflow_imitation = "agent_tool_routing" (Typed-Decisions
    agent tool-routing workflow domain: NOT zero-shot for TD tool routing; excluded from the strict zero-shot line);
  * synth_appliance_sata_case: all thresholds, device ratings, breaker cap and conditions are sampled; the 8 candidate joint
    schedules are shuffled so the gold key varies;
  * synth_causal_scm_cladder: sampled rational SCMs (parameters on a 1/20 grid), about half Simpson reversals by construction
    (rejection sampling to the drawn type), random scenario wording;
  * synth_music_harmony_pop909: random voicings, octaves, inversions (root position kept for pitch-symmetric / sus chords so the
    label stays determined), random rhythms, 1-3 short non-chord tones; the state lists note events only (no pitch-class set, no
    histogram, no root line); the question states the labelling rule (pitch classes sounding >= 1 beat in total);
  * synth_pyexec_cruxeval: six program templates with sampled arguments; distractors are the outputs of executed buggy variants.
v1.1 (2026-10-14, review 2026-10-14 17:00 finding 4): the tool diagnosis question was not exclusive -- with every argument
  present, no lock and a verified cached result, both "ready" ('all required arguments present and no lock applies') and
  "cached" were true while the one-hot gold gave all mass to cached (351 / 1,500 train records). "ready" now states all three
  conditions (arguments present, no lock, result NOT already verified in the session context), so exactly one diagnosis option
  is true in every situation. Only that option text changes; the RNG stream, states, other questions and labels are unchanged.

  python3 sdv1_gap_generators_pd.py --out-dir DIR --n-per-family N --seed S --split train|val|discovery [--exclude-dir TRAIN_DIR]
        [--families a,b]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from fractions import Fraction
from typing import Any

VERSION = "sdv1_gap_generators_pd v1 (2026-10-14, PILOT-DATA patch of sdv1_gap_generators v1)"
TOOL_VERSION = "sdv1_gap_generators_pd v1.1 (2026-10-14, exclusive tool diagnosis: ready = arguments present, no lock, not already verified)"
# VERSION is kept for the four unchanged families so their records stay byte-identical; synth_tool_policy records carry TOOL_VERSION
LICENSE = "CC0-1.0 (synthetic deterministic oracle; first-party code, no third-party data)"
PITCH = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
QUAL = {
    "maj": ((0, 4, 7), "Major triad"), "min": ((0, 3, 7), "Minor triad"), "dim": ((0, 3, 6), "Diminished triad"),
    "aug": ((0, 4, 8), "Augmented triad"), "sus2": ((0, 2, 7), "Suspended 2nd triad"), "sus4": ((0, 5, 7), "Suspended 4th triad"),
    "7": ((0, 4, 7, 10), "Dominant seventh"), "maj7": ((0, 4, 7, 11), "Major seventh"), "min7": ((0, 3, 7, 10), "Minor seventh"),
    "dim7": ((0, 3, 6, 9), "Fully diminished seventh"), "hdim7": ((0, 3, 6, 10), "Half-diminished seventh"),
}
ROOT_POSITION_ONLY = {"aug", "dim7", "sus2", "sus4"}       # pitch-class sets shared across roots: inversion would make the label ambiguous


def rng_for(seed, family, idx, attempt=0):
    return random.Random(int(hashlib.sha256(f"{seed}|{family}|{idx}|{attempt}".encode()).hexdigest()[:16], 16))


def one_hot(keys, winner):
    assert winner in keys
    return {k: (1.0 if k == winner else 0.0) for k in keys}


def qc(prompt, options):
    return {"type": "choice", "instructions": prompt, "criteria": options, "prompt": prompt, "options": options}


def content_hash(rec):
    return hashlib.sha256(json.dumps([rec["state"], rec["questions"], rec["labels"]], sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------------------- 1. tool policy
TOOL_DOMAINS = [
    ("cloud_ops", [("scale_deployment", "Scale a Kubernetes deployment to a target replica count.", ("cluster", "namespace", "deployment", "replicas"), True),
                   ("fetch_pod_logs", "Read recent log lines from a pod.", ("cluster", "namespace", "pod_name"), False),
                   ("list_cluster_quotas", "Read CPU and memory quota usage for a namespace.", ("cluster", "namespace"), False)]),
    ("clinical_ehr", [("order_lab_panel", "Submit a signed laboratory order for a verified encounter.", ("patient_id", "encounter_id", "panel_code", "ordering_npi"), True),
                      ("lookup_drug_interaction", "Check contraindications between two RxNorm codes.", ("rxnorm_a", "rxnorm_b"), False),
                      ("read_vital_trends", "Fetch the last 24 h of vital signs for an encounter.", ("patient_id", "encounter_id"), False)]),
    ("fin_treasury", [("initiate_wire_transfer", "Submit an outbound wire from a treasury account.", ("source_acct", "beneficiary_iban", "amount_usd", "approval_token"), True),
                      ("query_fx_spot_rate", "Retrieve the spot quote for a currency pair.", ("base_ccy", "quote_ccy"), False),
                      ("check_sanctions_list", "Screen an entity name and country against sanctions lists.", ("entity_name", "country_iso2"), False)]),
    ("logistics_dispatch", [("reroute_shipment", "Change the destination depot of an in-transit shipment.", ("shipment_id", "new_depot", "carrier_code"), True),
                            ("track_shipment", "Read the latest scan events of a shipment.", ("shipment_id",), False),
                            ("estimate_transit_days", "Estimate transit days between two depots.", ("origin_depot", "dest_depot"), False)]),
    ("grid_operations", [("dispatch_battery", "Command a grid battery to discharge at a set power for a window.", ("site_id", "power_mw", "window_min"), True),
                         ("read_feeder_load", "Read the current load on a distribution feeder.", ("feeder_id",), False),
                         ("forecast_solar_output", "Forecast solar output for a site and hour.", ("site_id", "hour_utc"), False)]),
    ("lab_inventory", [("reserve_reagent_lot", "Reserve a reagent lot for an experiment.", ("lot_id", "experiment_id", "quantity_ml"), True),
                       ("check_freezer_temp", "Read the temperature log of a freezer.", ("freezer_id",), False),
                       ("lookup_sds_sheet", "Retrieve the safety data sheet of a chemical.", ("cas_number",), False)]),
]
LOCKS = ["Runtime policy lock: freeze_window=ACTIVE (mutating calls are prohibited without an emergency override ticket).",
         "Change-control lock: CAB_HOLD=ACTIVE (no state-changing operations until the hold is lifted).",
         "Maintenance mode: WRITE_LOCK=ACTIVE for this session (read-only operations still permitted for read tools).",
         "Compliance lock: LEGAL_HOLD=ACTIVE (state-changing actions on this account are blocked)."]
PENALTY = ("Fixed penalty schedule (identical for every session): executing a state-changing tool call while a policy lock forbids it = 12; "
           "calling a tool with a required argument missing = 8; calling the wrong tool = 9.5; calling a tool although the verified result "
           "is already in the session context = 3.5; asking for clarification when nothing is missing = 4; answering directly without the "
           "needed data = 7 (6 when a policy lock is active); refusing a permitted request = 5.5; the correct action costs 0.")


def _val(rng, name):
    r = rng.randrange
    table = {"cluster": f"prod-{rng.choice(['use1', 'use2', 'euw1', 'apse2'])}", "namespace": f"{rng.choice(['payments', 'search', 'ledger', 'auth'])}-v{r(1, 9)}",
             "deployment": f"{rng.choice(['checkout', 'indexer', 'gateway', 'scorer'])}-worker-{r(0, 40)}", "replicas": str(r(2, 64)),
             "pod_name": f"{rng.choice(['checkout', 'indexer', 'gateway'])}-{r(1000, 9999):x}", "patient_id": f"PT-{r(10000, 99999)}",
             "encounter_id": f"ENC-{r(100000, 999999)}", "panel_code": rng.choice(["BMP-8", "CMP-14", "CBC-D", "LFT-7", "TSH-1"]),
             "ordering_npi": f"1{r(100000000, 999999999)}", "rxnorm_a": str(r(100000, 999999)), "rxnorm_b": str(r(100000, 999999)),
             "source_acct": f"TR-{r(100, 999)}", "beneficiary_iban": f"DE{r(10, 99)}{r(10**17, 10**18 - 1)}", "amount_usd": str(250 * r(20, 4000)),
             "approval_token": f"APR-{r(1000, 99999)}", "base_ccy": rng.choice(["EUR", "GBP", "JPY", "CHF"]), "quote_ccy": rng.choice(["USD", "CAD", "AUD"]),
             "entity_name": f"{rng.choice(['Meridian', 'Halcyon', 'Northgate', 'Corvex'])} {rng.choice(['Logistics', 'Trading', 'Holdings'])} {r(1, 99)}",
             "country_iso2": rng.choice(["DE", "FR", "BR", "SG", "AE"]), "shipment_id": f"SH{r(10**7, 10**8 - 1)}", "new_depot": f"DEP-{r(10, 99)}",
             "carrier_code": rng.choice(["UPSN", "FDEG", "DHLX", "MAEU"]), "origin_depot": f"DEP-{r(10, 99)}", "dest_depot": f"DEP-{r(10, 99)}",
             "site_id": f"SITE-{r(100, 999)}", "power_mw": str(r(1, 60)), "window_min": str(15 * r(1, 12)), "feeder_id": f"FDR-{r(1000, 9999)}",
             "hour_utc": str(r(0, 24)), "lot_id": f"LOT-{r(10000, 99999)}", "experiment_id": f"EXP-{r(100, 999)}", "quantity_ml": str(5 * r(1, 200)),
             "freezer_id": f"FZ-{r(1, 60)}", "cas_number": f"{r(50, 9999)}-{r(10, 99)}-{r(0, 9)}"}
    return table[name]


def gen_tool(rng, idx, seed, split):
    dname, tools = rng.choice(TOOL_DOMAINS)
    situation = rng.choice(["ready", "missing", "cached", "locked"])
    cand = [i for i, t in enumerate(tools) if t[3]] if situation == "locked" else list(range(len(tools)))   # a lock forbids state-changing calls
    ti = rng.choice(cand)
    tname, tdesc, treq, tmut = tools[ti]
    oname = tools[(ti + rng.randrange(1, len(tools))) % len(tools)][0]
    provided = {k: _val(rng, k) for k in treq}
    missing = None
    if situation == "missing":
        missing = rng.choice(treq)
        del provided[missing]
    cached = (f"Verified result already in context ({rng.randrange(5, 120)} s old) for {tname} with {json.dumps(provided, sort_keys=True)}: "
              f"status=OK, ref={rng.randrange(10**5, 10**6):x}.") if situation == "cached" else None
    lock = rng.choice(LOCKS) if situation == "locked" else None
    noise = rng.choice(["", f"- Unrelated note: the on-call engineer is {rng.choice(['Ana', 'Raj', 'Mei', 'Tom'])}.\n",
                        f"- Unrelated note: dashboard refresh interval is {rng.randrange(10, 90)} s.\n"])
    lines = [f"- `{n}`: {d} | required={list(r_)} | state_changing={m}" for n, d, r_, m in tools]
    rng.shuffle(lines)
    state = (f"Domain: {dname}\nAvailable tools:\n" + "\n".join(lines) + "\n\nSession context:\n"
             f"- User request: run `{tname}` with arguments {json.dumps(provided, sort_keys=True)}\n"
             f"- Session context results: {cached or 'none relevant to this request'}\n"
             f"- Governance status: {lock or 'no active locks; standard operations permitted'}\n" + noise + f"- {PENALTY}")
    loss = {"call_target": {"ready": 0.0, "missing": 8.0, "cached": 3.5, "locked": 12.0}[situation], "call_distractor": 9.5,
            "clarify": 0.0 if situation == "missing" else 4.0,
            "answer_direct": 0.0 if situation == "cached" else (6.0 if situation == "locked" else 7.0),
            "refuse": 0.0 if situation == "locked" else 5.5}
    q1 = {"call_target": f"Invoke `{tname}` now with the provided arguments", "call_distractor": f"Invoke `{oname}` instead",
          "clarify": "Ask the user for the missing required argument before invoking any tool",
          "answer_direct": "Answer directly from the verified session context without calling a tool",
          "refuse": "Refuse because an active policy lock forbids the requested call"}
    q2 = {"status_ready": (f"All required arguments of `{tname}` are present, no lock applies, and the requested result is not "
                            "already verified in the session context"),
          "status_missing": f"A required argument of `{tname}` is missing from the request",
          "status_cached": "The requested result is already verified in the session context",
          "status_blocked": "The requested call is blocked by an active policy lock"}
    w2 = {"ready": "status_ready", "missing": "status_missing", "cached": "status_cached", "locked": "status_blocked"}[situation]
    w1 = min(loss, key=loss.get)
    return {"id": f"synth_tool_policy_{seed}_{idx:05d}", "base_id": f"synth_tool_policy:{split}:s{seed}:i{idx:05d}", "source": "synth_tool_policy",
            "license": LICENSE, "split": split, "state": state,
            "questions": {"policy": qc("Given the tool schemas, the session context and the fixed penalty schedule, which action has the lowest penalty?", q1),
                          "diagnosis": qc("What is the precondition status of the user's request?", q2)},
            "labels": {"policy": one_hot(list(q1), w1), "diagnosis": one_hot(list(q2), w2)},
            "label_kind": {"policy": "hard", "diagnosis": "hard"},
            "meta": {"generator": TOOL_VERSION, "family": "synth_tool_policy", "situation": situation, "domain": dname,
                     "td_workflow_imitation": "agent_tool_routing",
                     "target_semantics": {"policy": "determinate_comparative_decision", "diagnosis": "determinate_evidence_selection"},
                     "label_source": {"policy": "rule_exact", "diagnosis": "rule_exact"}, "loss_by_option": {"policy": loss}}}


# ------------------------------------------------------------------------------------------------------------- 2. appliance
SUBSETS = [tuple(d for d, b in zip(("hvac", "wh", "ev", "dryer"), bits) if b) for bits in
           [(i >> 3 & 1, i >> 2 & 1, i >> 1 & 1, i & 1) for i in range(16)]]


def gen_appliance(rng, idx, seed, split):
    kw = {"hvac": rng.choice([3, 4, 5]), "wh": rng.choice([2, 3, 4]), "ev": rng.choice([6, 7, 9, 11]), "dryer": rng.choice([4, 5, 6])}
    cap = rng.randrange(6, 19)
    peak, occ, laundry = rng.random() < 0.4, rng.random() < 0.6, rng.random() < 0.5
    lo, hi = rng.choice([(18, 25), (19, 24), (17, 26)])
    t_in = rng.randrange(12, 33)
    tank_min = rng.choice([42, 45, 48])
    t_w = rng.randrange(30, 66)
    ev_target, ev_urgent = rng.choice([(80, 30), (90, 25), (75, 35)])
    soc = rng.randrange(5, 101)
    want = {"hvac": occ and not (lo <= t_in <= hi), "wh": t_w < tank_min, "ev": soc < ev_target and ((not peak) or soc < ev_urgent),
            "dryer": laundry and not peak}
    on, tot = [], 0
    for d in ("hvac", "wh", "ev", "dryer"):
        if want[d] and tot + kw[d] <= cap:
            on.append(d)
            tot += kw[d]
    gold = tuple(on)
    others = [s for s in SUBSETS if s != gold]
    rng.shuffle(others)
    cands = [gold] + others[:7]
    rng.shuffle(cands)

    def fmt(s):
        return f"ON={list(s)} ({sum(kw[k] for k in s)} kW total)" if s else "NONE (all four appliances OFF; 0 kW total)"
    jo = {f"s{j}": fmt(s) for j, s in enumerate(cands)}
    gk = f"s{cands.index(gold)}"
    titles = {"hvac": "Heat Pump HVAC", "wh": "Heat-Pump Water Heater", "ev": "EV Charger", "dryer": "Electric Dryer"}
    state = (f"Household controller state:\n- Main breaker cap: {cap} kW maximum simultaneous load\n- Tariff period: {'PEAK' if peak else 'OFF_PEAK'}\n"
             f"- Occupancy: {'OCCUPIED' if occ else 'UNOCCUPIED'}\n- Indoor temperature: {t_in} C (comfort band when occupied: {lo}..{hi} C)\n"
             f"- Water tank temperature: {t_w} C (minimum setpoint: {tank_min} C)\n- EV state of charge: {soc}% (target {ev_target}%; urgent below {ev_urgent}%)\n"
             f"- Laundry queue: {'QUEUED' if laundry else 'EMPTY'}\n\nRules (strict priority 1 -> 4 under the {cap} kW cap):\n"
             f"1. `hvac` ({kw['hvac']} kW): wants ON iff OCCUPIED and indoor temperature is outside {lo}..{hi} C.\n"
             f"2. `wh` ({kw['wh']} kW): wants ON iff tank temperature < {tank_min} C.\n"
             f"3. `ev` ({kw['ev']} kW): wants ON iff SoC < {ev_target}% AND (tariff is OFF_PEAK or SoC < {ev_urgent}%).\n"
             f"4. `dryer` ({kw['dryer']} kW): wants ON iff laundry is QUEUED and tariff is OFF_PEAK.\n"
             f"Load shedding: go through priorities 1 -> 4; a device turns ON iff it wants ON and adding its kW keeps the total <= {cap} kW.")
    qs, labs, lk, ts, ls = {}, {}, {}, {}, {}
    for d in ("hvac", "wh", "ev", "dryer"):
        q = f"dev_{d}"
        qs[q] = qc(f"In the resulting schedule, what is the state of `{d}` ({titles[d]})?",
                   {"on": f"`{d}` ({kw[d]} kW) is ON", "off": f"`{d}` is OFF"})
        labs[q], lk[q], ts[q], ls[q] = one_hot(["on", "off"], "on" if d in gold else "off"), "hard", "determinate_grounded_composition", "rule_exact"
    qs["case_joint"] = qc("Select the exact set of appliances that are ON together after applying all rules:", jo)
    labs["case_joint"], lk["case_joint"], ts["case_joint"], ls["case_joint"] = one_hot(list(jo), gk), "hard", "determinate_grounded_composition", "rule_exact"
    return {"id": f"synth_appliance_sata_{seed}_{idx:05d}", "base_id": f"synth_appliance_sata_case:{split}:s{seed}:i{idx:05d}",
            "source": "synth_appliance_sata_case", "license": LICENSE, "split": split, "state": state, "questions": qs, "labels": labs,
            "label_kind": lk, "meta": {"generator": VERSION, "family": "synth_appliance_sata_case", "target_semantics": ts, "label_source": ls}}


# ------------------------------------------------------------------------------------------------------------- 3. causal SCM
SCEN = [("Z", "X", "Y"), ("age group", "treatment", "recovery"), ("region", "ad exposure", "purchase"), ("soil type", "fertiliser", "high yield"),
        ("prior plan", "upgrade offer", "renewal"), ("shift", "new checklist", "zero defects")]


def gen_causal(rng, idx, seed, split):
    want_rev = rng.random() < 0.5
    g = lambda lo=1, hi=19: Fraction(rng.randrange(lo, hi + 1), 20)
    for _ in range(10000):
        qz, px0, px1 = g(2, 18), g(), g()
        y00, y10, y01, y11 = g(), g(), g(), g()
        q0 = 1 - qz
        p_x1 = q0 * px0 + qz * px1
        p_x0 = q0 * (1 - px0) + qz * (1 - px1)
        o1 = (q0 * px0 * y10 + qz * px1 * y11) / p_x1
        o0 = (q0 * (1 - px0) * y00 + qz * (1 - px1) * y01) / p_x0
        d1 = q0 * y10 + qz * y11
        d0 = q0 * y00 + qz * y01
        od, ad = o1 - o0, d1 - d0
        if od == 0 or ad == 0 or abs(od) < Fraction(1, 100) or abs(ad) < Fraction(1, 100):
            continue
        rev = (od > 0) != (ad > 0)
        if rev == want_rev:
            break
    else:
        raise RuntimeError("causal rejection sampling exhausted")
    zn, xn, yn = rng.choice(SCEN)
    f = lambda v: f"{float(v):.4f} ({v})"
    state = (f"Causal model (Z -> X, Z -> Y, X -> Y; no unmeasured confounders). Z = {zn}, X = {xn}, Y = {yn} (all binary).\n"
             f"- P(Z = 1) = {f(qz)}, P(Z = 0) = {f(q0)}.\n- Treatment assignment P(X = 1 | Z):\n    P(X = 1 | Z = 0) = {f(px0)}\n    P(X = 1 | Z = 1) = {f(px1)}\n"
             f"- Outcome mechanism P(Y = 1 | do(X = x), Z = z):\n    P(Y = 1 | X = 0, Z = 0) = {f(y00)}\n    P(Y = 1 | X = 1, Z = 0) = {f(y10)}\n"
             f"    P(Y = 1 | X = 0, Z = 1) = {f(y01)}\n    P(Y = 1 | X = 1, Z = 1) = {f(y11)}")
    cw = "pos_both" if od > 0 and ad > 0 else "neg_both" if od < 0 and ad < 0 else "obs_pos_do_neg" if od > 0 else "obs_neg_do_pos"
    co = {"pos_both": "Observational difference P(Y=1|X=1) - P(Y=1|X=0) > 0 and causal effect P(Y=1|do(X=1)) - P(Y=1|do(X=0)) > 0",
          "neg_both": "Observational difference < 0 and causal effect < 0",
          "obs_pos_do_neg": "Reversal: observational difference > 0 but causal effect < 0",
          "obs_neg_do_pos": "Reversal: observational difference < 0 but causal effect > 0"}
    sp = lambda p: {"y1": float(p), "y0": float(1 - p)}                 # exact rationals as floats (no 6-dp rounding)
    return {"id": f"synth_causal_scm_{seed}_{idx:05d}", "base_id": f"synth_causal_scm:{split}:s{seed}:i{idx:05d}", "source": "synth_causal_scm_cladder",
            "license": LICENSE, "split": split, "state": state,
            "questions": {"rung1_obs": qc("Observational query: among units observed with X = 1 (no intervention), what is the distribution of Y?",
                                          {"y1": "Y = 1", "y0": "Y = 0"}),
                          "rung2_do": qc("Interventional query: if X is set to 1 for everyone, do(X = 1), what is the distribution of Y?",
                                         {"y1": "Y = 1 under do(X = 1)", "y0": "Y = 0 under do(X = 1)"}),
                          "causal_contrast": qc("Compare the observational association with the average causal effect of X on Y:", co)},
            "labels": {"rung1_obs": sp(o1), "rung2_do": sp(d1), "causal_contrast": one_hot(list(co), cw)},
            "label_kind": {"rung1_obs": "soft", "rung2_do": "soft", "causal_contrast": "hard"},
            "meta": {"generator": VERSION, "family": "synth_causal_scm_cladder", "simpson_reversal": rev,
                     "exact": {"obs": str(o1), "do": str(d1), "obs_diff": str(od), "ate": str(ad)},
                     "target_semantics": {"rung1_obs": "exact_outcome_predictive_probability", "rung2_do": "exact_outcome_predictive_probability",
                                          "causal_contrast": "determinate_grounded_composition"},
                     "label_source": {"rung1_obs": "exact:posterior_predictive", "rung2_do": "exact:posterior_predictive", "causal_contrast": "rule_exact"}}}


# ------------------------------------------------------------------------------------------------------------- 4. music harmony
CHORDS = [(f"{PITCH[r]}:{q}", r, q, frozenset((r + i) % 12 for i in iv)) for r in range(12) for q, (iv, _) in QUAL.items()]


def gen_music(rng, idx, seed, split):
    ci = rng.randrange(len(CHORDS))
    label, root, qual, pcs = CHORDS[ci]
    ivs = QUAL[qual][0]
    tones = [(root + i) % 12 for i in ivs]
    bass_pc = root if qual in ROOT_POSITION_ONLY else rng.choice(tones[:3])     # inversion: bass is the root, third or fifth
    events = []
    bass_oct = rng.choice([2, 3])
    t = 0.0
    events.append((1.0, rng.choice([2.0, 4.0, 3.0]), 36 + 12 * (bass_oct - 2) + bass_pc, "bass"))
    pattern = rng.choice([[(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (4.0, 1.0)], [(1.0, 1.5), (2.5, 1.5), (4.0, 1.0)],
                          [(1.0, 2.0), (3.0, 2.0)], [(1.0, 0.5), (1.5, 1.5), (3.0, 0.5), (3.5, 1.5)]])
    for onset, dur in pattern:
        voiced = list(tones)
        if rng.random() < 0.5:
            voiced.append(rng.choice(tones))                           # doubling
        for pc in voiced:
            octv = rng.choice([4, 4, 5, 3])
            events.append((onset, dur, 12 * (octv + 1) + pc, "chord"))
    nct_pcs = [pc for pc in range(12) if pc not in pcs]
    for _ in range(rng.randrange(1, 4)):                                # short non-chord tones: total < 1 beat per pitch class
        pc = rng.choice(nct_pcs)
        on = rng.choice([1.25, 1.75, 2.25, 2.75, 3.25, 3.75])
        events.append((on, rng.choice([0.25, 0.5]) * 0.5, 12 * (rng.choice([4, 5]) + 1) + pc, "nct"))
    dur_by_pc = {}
    for _, d, m, _k in events:
        dur_by_pc[m % 12] = dur_by_pc.get(m % 12, 0) + d
    sounding = frozenset(pc for pc, d in dur_by_pc.items() if d >= 1.0)
    if sounding != pcs:
        return None                                                     # redraw (a non-chord tone or thin voicing broke the rule)
    events.sort(key=lambda e: (e[0], e[2]))
    lines = [f"onset {on:.2f}, dur {d:.3f}, vel {rng.randrange(40, 110)}: MIDI {m} ({PITCH[m % 12]}{m // 12 - 1})" for on, d, m, _k in events]
    K = rng.choice([16, 32, 48, 52])
    conf = [c for c in range(len(CHORDS)) if c != ci and CHORDS[c][3] != pcs and len(CHORDS[c][3] & pcs) >= 2]
    rest = [c for c in range(len(CHORDS)) if c != ci and CHORDS[c][3] != pcs and c not in conf]
    rng.shuffle(conf)
    rng.shuffle(rest)
    menu = [ci] + (conf + rest)[:K - 1]
    rng.shuffle(menu)
    opts, gk = {}, None
    for j, c in enumerate(menu):
        lab, _r, q, cp = CHORDS[c]
        opts[f"c{j:02d}"] = f"{lab} -- {QUAL[q][1]}; pitch classes {{{', '.join(PITCH[p] for p in sorted(cp))}}}"
        if c == ci:
            gk = f"c{j:02d}"
    state = "Symbolic measure (4/4, quarter note = 1.0 beat). Note events:\n  - " + "\n  - ".join(lines)
    return {"id": f"synth_music_harmony_{seed}_{idx:05d}", "base_id": f"synth_music_harmony:{split}:s{seed}:i{idx:05d}", "source": "synth_music_harmony_pop909",
            "license": LICENSE, "split": split, "state": state,
            "questions": {"chord": qc(f"Which chord label governs this measure? Rule: the chord's pitch classes are exactly the pitch classes that sound for "
                                      f"at least 1.0 beat in total; shorter tones are non-chord tones; the root is the chord's root, not necessarily the "
                                      f"bass. Choose from {K} candidates.", opts)},
            "labels": {"chord": one_hot(list(opts), gk)}, "label_kind": {"chord": "hard"},
            "meta": {"generator": VERSION, "family": "synth_music_harmony_pop909", "menu_k": K, "true_chord": label, "inversion": bass_pc != root,
                     "target_semantics": {"chord": "determinate_grounded_composition"}, "label_source": {"chord": "rule_exact"}}}


# ------------------------------------------------------------------------------------------------------------- 5. python execution
PY_TEMPLATES = {
    "alias": ("def solve(x, y, z):\n    grid = [[x, y], [y, z]]\n    alias = grid[0]\n    shallow = list(grid)\n    alias.append(x + z)\n"
              "    shallow[1] = [z, x]\n    return (len(grid[0]), grid[1][0], shallow[0][-1])",
              ["def solve(x, y, z):\n    grid = [[x, y], [y, z]]\n    alias = list(grid[0])\n    shallow = list(grid)\n    alias.append(x + z)\n"
               "    shallow[1] = [z, x]\n    return (len(grid[0]), grid[1][0], shallow[0][-1])",
               "def solve(x, y, z):\n    grid = [[x, y], [y, z]]\n    alias = grid[0]\n    shallow = grid\n    alias.append(x + z)\n"
               "    shallow[1] = [z, x]\n    return (len(grid[0]), grid[1][0], shallow[0][-1])"], 3),
    "dict_accum": ("def solve(nums):\n    seen = {}\n    total = 0\n    for idx, val in enumerate(nums):\n        prev = seen.get(val, -1)\n"
                   "        if prev >= 0:\n            total += (idx - prev) * val\n        else:\n            total -= val\n        seen[val] = idx\n"
                   "    return (total, len(seen))",
                   ["def solve(nums):\n    seen = {}\n    total = 0\n    for idx, val in enumerate(nums):\n        prev = seen.get(val, -1)\n"
                    "        if prev >= 0:\n            total += (idx - prev) * val\n        seen[val] = idx\n    return (total, len(seen))",
                    "def solve(nums):\n    seen = {}\n    total = 0\n    for idx, val in enumerate(nums):\n        prev = seen.get(val, -1)\n"
                    "        if prev >= 0:\n            total += (idx - prev) * val\n        else:\n            total -= val\n            seen[val] = idx\n"
                    "    return (total, len(seen))"], 1),
    "mutable_default": ("def add(item, bucket=[]):\n    bucket.append(item)\n    return bucket\n\ndef solve(a, b, c):\n    first = add(a)\n    second = add(b)\n"
                        "    third = add(c, [])\n    return (len(first), second[-1], len(third), first is second)",
                        ["def add(item, bucket=None):\n    bucket = [] if bucket is None else bucket\n    bucket.append(item)\n    return bucket\n\n"
                         "def solve(a, b, c):\n    first = add(a)\n    second = add(b)\n    third = add(c, [])\n    return (len(first), second[-1], len(third), first is second)"], 3),
    "slice_step": ("def solve(xs, k):\n    head = xs[:k]\n    tail = xs[k::2]\n    rev = xs[::-1][:k]\n    return (sum(head) - sum(tail), rev[-1], len(tail))",
                   ["def solve(xs, k):\n    head = xs[:k + 1]\n    tail = xs[k::2]\n    rev = xs[::-1][:k]\n    return (sum(head) - sum(tail), rev[-1], len(tail))",
                    "def solve(xs, k):\n    head = xs[:k]\n    tail = xs[k:]\n    rev = xs[::-1][:k]\n    return (sum(head) - sum(tail), rev[-1], len(tail))"], 2),
    "floor_mod": ("def solve(a, b, c):\n    q = a // b\n    r = a % b\n    t = -a // b\n    return (q, r, t, (c - a) % b)",
                  ["def solve(a, b, c):\n    q = int(a / b)\n    r = a % b\n    t = int(-a / b)\n    return (q, r, t, (c - a) % b)",
                   "def solve(a, b, c):\n    q = a // b\n    r = a - b * q\n    t = -(a // b)\n    return (q, r, t, (c - a) % b)"], 3),
    "str_ops": ("def solve(s, n):\n    parts = s.split('-')\n    joined = ''.join(p[::-1] for p in parts)\n    return (joined[:n], len(parts), s.count('a'), s.find('b'))",
                ["def solve(s, n):\n    parts = s.split('-')\n    joined = ''.join(parts)[::-1]\n    return (joined[:n], len(parts), s.count('a'), s.find('b'))",
                 "def solve(s, n):\n    parts = s.split('-')\n    joined = ''.join(p[::-1] for p in parts)\n    return (joined[:n + 1], len(parts), s.count('a'), s.index('a'))"], 2),
}


def _run(code, call_args):
    ns = {}
    exec(code, ns)
    return ns["solve"](*call_args)


def gen_pyexec(rng, idx, seed, split):
    name = rng.choice(sorted(PY_TEMPLATES))
    code, bugs, _ = PY_TEMPLATES[name]
    r = rng.randrange
    if name in ("alias", "mutable_default"):
        args = (r(-20, 60), r(-20, 60), r(-20, 60))
    elif name == "dict_accum":
        pool = [r(-9, 30) for _ in range(3)]
        args = ([rng.choice(pool) for _ in range(r(5, 10))],)
    elif name == "slice_step":
        xs = [r(-15, 40) for _ in range(r(5, 11))]
        args = (xs, r(1, len(xs) - 1))
    elif name == "floor_mod":
        args = (r(-90, 91), rng.choice([d for d in range(-9, 10) if d not in (0,)]), r(-50, 51))
    else:
        s = "-".join("".join(rng.choice("abcdeloprst") for _ in range(r(2, 6))) for _ in range(r(2, 5)))
        args = (s, r(1, 7))
    call = f"solve({', '.join(repr(x) for x in args)})"
    ans = repr(_run(code, args))
    cands = [ans]
    for b in bugs:
        try:
            v = repr(_run(b, args))
        except Exception:                                                 # noqa: BLE001 -- a crashing bug variant is no distractor
            continue
        if v not in cands:
            cands.append(v)
    val = _run(code, args)
    tries = 0
    while len(cands) < 5 and tries < 50:                               # perturb one element of the true tuple
        tries += 1
        if isinstance(val, tuple):
            lst = list(val)
            j = rng.randrange(len(lst))
            if isinstance(lst[j], bool):
                lst[j] = not lst[j]
            elif isinstance(lst[j], int):
                lst[j] = lst[j] + rng.choice([-2, -1, 1, 2])
            elif isinstance(lst[j], str):
                lst[j] = lst[j][::-1] if len(lst[j]) > 1 else lst[j] + "a"
            v = repr(tuple(lst))
        else:
            v = repr(val) + "0"
        if v not in cands:
            cands.append(v)
    if len(cands) < 5:
        return None
    order = list(range(len(cands)))
    rng.shuffle(order)
    opts, gk = {}, None
    for slot, oi in enumerate(order):
        opts[f"o{slot + 1}"] = cands[oi]
        if oi == 0:
            gk = f"o{slot + 1}"
    state = f"Python function:\n```python\n{code}\n```\nTarget call: `{call}`"
    return {"id": f"synth_pyexec_cruxeval_{seed}_{idx:05d}", "base_id": f"synth_pyexec_cruxeval:{split}:s{seed}:i{idx:05d}", "source": "synth_pyexec_cruxeval",
            "license": LICENSE, "split": split, "state": state,
            "questions": {"output": qc(f"What is the exact return value repr(...) of `{call}`?", opts)},
            "labels": {"output": one_hot(list(opts), gk)}, "label_kind": {"output": "hard"},
            "meta": {"generator": VERSION, "family": "synth_pyexec_cruxeval", "template": name, "execution_verified": True,
                     "target_semantics": {"output": "determinate_grounded_composition"}, "label_source": {"output": "rule_exact"}}}


GENERATORS = {"synth_tool_policy": gen_tool, "synth_appliance_sata_case": gen_appliance, "synth_causal_scm_cladder": gen_causal,
              "synth_music_harmony_pop909": gen_music, "synth_pyexec_cruxeval": gen_pyexec}
DI_TARGETS = {"synth_tool_policy": ["When2Call", "BFCL", "API-Bank"], "synth_appliance_sata_case": ["Home Appliance Simulator", "SATA-Bench"],
              "synth_causal_scm_cladder": ["CLadder", "GPQA", "BBH"], "synth_music_harmony_pop909": ["POP909-CL", "high-K menus 33-52"],
              "synth_pyexec_cruxeval": ["CRUXEval", "GSM8K"]}


def check(rec):
    """schema + semantics checks (as the pack's verify_qsemantics_record, plus loss argmin = gold)"""
    m = rec["meta"]
    for q, qq in rec["questions"].items():
        lab = rec["labels"][q]
        assert set(lab) == set(qq["criteria"]) and abs(sum(lab.values()) - 1) < 1e-5 and min(lab.values()) >= 0
        sem = m["target_semantics"][q]
        hard = sem in ("determinate_comparative_decision", "determinate_evidence_selection", "determinate_grounded_composition")
        assert (rec["label_kind"][q] == "hard") == hard and (not hard or max(lab.values()) >= 0.999999)
        lb = (m.get("loss_by_option") or {}).get(q)
        if lb is not None:
            assert set(lb) == set(lab) and max(lab, key=lab.get) == min(lb, key=lb.get)


def build(out_dir, n, seed, split, exclude_dir=None, families=None):
    os.makedirs(out_dir, exist_ok=True)
    excl = set()
    for xd in (exclude_dir.split(",") if exclude_dir else []):
        for fam in GENERATORS:
            p = os.path.join(xd, f"{fam}.jsonl")
            if os.path.exists(p):
                for l in open(p):
                    r_ = json.loads(l)
                    excl.add(content_hash(r_))
                    excl.add("state:" + hashlib.sha256(r_["state"].encode()).hexdigest())
    summ = {"version": VERSION, "tool_version": TOOL_VERSION, "seed": seed, "split": split, "exclude_dir": exclude_dir, "excluded_content_hashes": len(excl), "families": {}}
    for fam, fn in GENERATORS.items():
        if families and fam not in families:
            continue
        seen, recs, skipped, attempt_total = set(excl), [], {"duplicate": 0, "redraw": 0}, 0
        idx = 0
        while len(recs) < n:
            rec, att = None, 0
            while rec is None:
                rec = fn(rng_for(seed, fam, idx, att), idx, seed, split)
                att += 1
                attempt_total += 1
                if rec is None:
                    skipped["redraw"] += 1
                if att > 200:
                    raise RuntimeError(f"{fam} idx {idx}: no valid draw")
            h = content_hash(rec)
            hs = "state:" + hashlib.sha256(rec["state"].encode()).hexdigest()
            if h in seen or hs in seen:                     # no repeated content and no repeated state (within the split or vs --exclude-dir)
                skipped["duplicate"] += 1
            else:
                check(rec)
                rec["meta"]["di_target_benchmarks"] = DI_TARGETS[fam]
                rec["meta"]["content_sha256"] = h
                seen.add(h)
                seen.add(hs)
                recs.append(rec)
            idx += 1
            if idx > 50 * n:
                raise RuntimeError(f"{fam}: could not reach {n} distinct records ({len(recs)})")
        path = os.path.join(out_dir, f"{fam}.jsonl")
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        states = {hashlib.sha256(r["state"].encode()).hexdigest() for r in recs}
        summ["families"][fam] = {"records": len(recs), "decision_units": sum(len(r["questions"]) for r in recs), "distinct_states": len(states),
                                 "indices_used": idx, "skipped": skipped, "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                                 "di_target_benchmarks": DI_TARGETS[fam]}
    with open(os.path.join(out_dir, "gap_supply_manifest.json"), "w") as f:
        json.dump(summ, f, indent=1, sort_keys=True)
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-per-family", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--split", default="train")
    ap.add_argument("--exclude-dir", default="", help="comma-separated split dirs whose contents and states must not recur")
    ap.add_argument("--families", default="")
    a = ap.parse_args(argv)
    s = build(a.out_dir, a.n_per_family, a.seed, a.split, a.exclude_dir or None, set(a.families.split(",")) if a.families else None)
    print(json.dumps({k: v for k, v in s.items() if k != "families"}))
    for fam, v in s["families"].items():
        print(fam, json.dumps({k: v[k] for k in ("records", "decision_units", "distinct_states", "indices_used", "skipped")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
