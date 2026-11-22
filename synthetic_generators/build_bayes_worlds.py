#!/usr/bin/env python3
"""
build_bayes_worlds.py -- finite-hypothesis BAYESIAN WORLDS: synthetic inference and decision problems whose every target is EXACT
(enumeration over a finite latent grid in float64), built to add kinds of supervision the numeric corpora lack: copied /
correlated sources, latent source reliability, missingness mechanisms, loss matrices with abstention, costly evidence acquisition
(exact one- and two-step value of information), multimodal / weakly identified and hierarchical posteriors. Design, formulas,
schema, validation and the DRAFT pre-registration of (104): research/bayes_worlds.md.

A WORLD = a finite hypothesis set H (2-12), a stated prior over H, a stated generative model for the evidence (sources with stated
likelihoods / reliabilities / dependence / missingness), the observed evidence, and optionally a loss matrix and priced information
sources. Every number is printed on a 1% grid (probabilities) or in whole money units, and the solver uses exactly the printed
numbers. The hidden truth (hypothesis, nuisance latents, the next observation) is drawn from the STATED model, so the exact posterior
is calibrated against it (checked at build time).

A UNIT = one record, ONE question (build_jev_data schema; a `choice` question; soft target):
  posterior   the exact posterior over H (or over a hierarchical member's status)
  latent      the exact posterior over a nuisance latent (a source's reliability type; a shared hierarchical grade)
  predictive  the exact posterior predictive of a stated next observation
  action      one-hot (ties split) on the exact Bayes action (min expected loss); every action's expected loss in meta
  acquire     one-hot (ties split) on the exact optimal choice among act now / abstain / buy information then act
A GROUP = an anchor world plus linked worlds (prior swap, evidence rounds, dependence swap, mechanism swap, loss swap) and re-
renderings of the same world (order, units, paraphrase). base_id = the group; a whole group is in exactly one split.

LEAKAGE RULE (research/numeric_v2_axes.md): nothing visible (options, option order, which hypotheses are listed, wording, number
formats, lengths) is computed from a target or from the hidden truth. Separate RNG streams: `world` (stated model, then truth, then
evidence), `plan` (which questions / variants), `render` (templates, formats, orders). The renderer receives the visible spec only
(`visible(w)` drops the truth). No rejection sampling on anything derived from a posterior.

v1.2 (`build --version v1.2`, the CLI default; research/bayes_worlds.md section 16): (A) train_aug = K = 3 equivalent renderings
of every train world (orders, number formats, templates, a semantics-preserving skin) with identical exact targets, and the
matched control train_plain = the same worlds rendered once (byte-identical to rendering 1); (B) every train group carries its
intervention siblings in every rendering (prior / dependence / mechanism / reliability swaps; loss swaps for the decision
families); (C) the missingness2 family (hypothesis-dependent absence AND present observations; MAR / hypothesis-dependent /
result-dependent recording paired on the same data; a latent joint recording rule; an explicit-likelihood-factor rendering; a
held-out categorical structure); (D) the sealed test_confirm split. The shared eval splits are the v1.1 eval records verbatim
plus v1.2 supplements. `--version v1` / `--version v1.1` rebuild those corpora byte for byte.
Opt-in train views (defaults unchanged): --hint-per-group / --plain-k-view ((105b)); --extra-train-view
NAME:k=K:units=N[:drop=F:refill=G][:sib=v11] ((105d) recipe ablation: whole groups of the same train stream, e.g. missingness2
swapped for belief-family groups, or rendering 1 over more worlds; (105e) sib=v11: each group cut to its v1.1 sibling plan).

Usage (.venv python: numpy; scipy only for the evaluator):
  .venv/bin/python3 build_bayes_worlds.py build  [--version v1 | v1.1 | v1.2] [--out DIR]
  .venv/bin/python3 build_bayes_worlds.py check  --out DIR
  .venv/bin/python3 build_bayes_worlds.py report --out DIR
  .venv/bin/python3 build_bayes_worlds.py --self-test
Evaluation (KL / log score / regret / calibration / paired interventions / invariances, baselines and broken predictors):
  analysis_bayes_worlds_eval.py (repo root); v1.2's per-stratum mechanism and paired-contrast readouts:
  analysis_bayes_worlds_eval_v12.py.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import itertools
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SET = "bayes_worlds"
VERSION = "bayes_worlds v1 (2026-10-05)"
NS = "bayes_worlds/v1"
SPLITS = ("train", "val", "test", "test_heldout_structure")
DEFAULT_UNITS = {"train": 60000, "val": 4000, "test": 4000, "test_heldout_structure": 4000}
OUT = str(HERE / "tmp" / "bayes_worlds" / "out" / "jev_bayes_worlds_v1")
# ---- CORPUS VERSIONS. v1 (2026-10-05) is reproduced byte for byte by `build --version v1`. v1.1 (the default) adds, behind this
# switch only: three VALUE question kinds (expected regret of a named action, best-vs-second-best gap, net value of information;
# one-hot on a bin whose edges come from VISIBLE quantities), a decision-heavier family mix (~40% decision-type units), a
# test_novel_wording split (in-family structures rendered ONLY with reserved template variants and reserved skins never used in
# train / val / test), and meta.difficulty / meta.wording. Every v1.1 draw uses its own RNG stream or key, so v1 is untouched.
VERSIONS = {"v1": "bayes_worlds v1 (2026-10-05)", "v1.1": "bayes_worlds v1.1 (2026-10-05)", "v1.2": "bayes_worlds v1.2 (2026-10-06)"}
NS_BY_VER = {"v1": "bayes_worlds/v1", "v1.1": "bayes_worlds/v1_1", "v1.2": "bayes_worlds/v1_2"}
OUT_BY_VER = {"v1": OUT, "v1.1": str(HERE / "tmp" / "bayes_worlds" / "out" / "jev_bayes_worlds_v1_1"),
              "v1.2": str(HERE / "tmp" / "bayes_worlds" / "out" / "jev_bayes_worlds_v1_2")}
SPLITS_V11 = SPLITS + ("test_novel_wording",)
CFG = {"ver": "v1.1"}          # the module default stays v1.1 (the CLI default for `build` is v1.2); v1.2 is selected explicitly
# ---- v1.2 (2026-10-06; research/bayes_worlds.md section 16). Everything v1.2 adds sits behind ver() == "v1.2" on its own RNG streams,
# keys and namespaces, so v1 / v1.1 builds stay byte-identical. Layout of a v1.2 build: train_aug/ (K equivalent renderings of every
# train world, each carrying the same exact targets) and train_plain/ (the SAME worlds, rendering 1 only: byte-identical to the
# rendering-1 lines of train_aug), the shared eval splits val / test / test_heldout_structure / test_novel_wording (= the v1.1 eval
# records VERBATIM, regenerated by the v1.1 code path, plus v1.2 supplement files bayes_worlds_v12_<family>.jsonl), and the sealed
# test_confirm split (fresh seed namespace, held-out templates and skins, unseen compositions; never scored until a candidate exists).
TRAIN_VIEWS = ("train_aug", "train_plain")
V12_EVAL = ("val", "test", "test_heldout_structure", "test_novel_wording")
CONFIRM = "test_confirm"
SPLITS_V12 = TRAIN_VIEWS + V12_EVAL + (CONFIRM,)
V11_UNIQUE_TRAIN_WORLDS = 22032          # distinct world_ids in the v1.1 train split (measured on jev_bayes_worlds_v1_1)
V12_K = 3                                # renderings per train world in train_aug
HINT_PER_GROUP = False                   # (105b, opt-in) the missingness2 explicit-factor flag drawn once per train GROUP
                                         # (shared by every rendering and sibling), not per rendering; off = ac91ccad builds
V12_SUPPLEMENT_GROUPS = {               # v1.2 supplement groups added to each shared eval split: (family, structure source, groups)
    "val": [("missingness2", "train", 150), ("reliability", "train", 75)],
    "test": [("missingness2", "train", 300), ("reliability", "train", 150)],
    "test_novel_wording": [("missingness2", "train", 300), ("reliability", "train", 150)],
    "test_heldout_structure": [("missingness2", "heldout", 300), ("reliability", "heldout", 150)],
}
CONFIRM_UNITS = 6000
CONFIRM_SALT = "confirm-2026-10-06"


def ver():
    return CFG["ver"]


def set_version(v):
    assert v in VERSIONS, v
    CFG["ver"] = v


def splits():
    return SPLITS if ver() == "v1" else SPLITS_V12 if ver() == "v1.2" else SPLITS_V11


def default_units():
    d = dict(DEFAULT_UNITS)
    if ver() in ("v1.1", "v1.2"):
        d["test_novel_wording"] = 4000
    return d


def gen_version():
    return VERSIONS[ver()]


def ns_root():
    return NS_BY_VER[ver()]


LICENSE = "synthetic (ours), commercial-OK: generated by build_bayes_worlds.py, no third-party data or text"
QKINDS = ("posterior", "latent", "predictive", "action", "acquire", "value_regret", "value_gap", "value_voi")   # v1: the first 5
VALUE_KINDS = ("value_regret", "value_gap", "value_voi")
DECISION_KINDS = ("action", "acquire") + VALUE_KINDS
TARGET_KIND = {
    "posterior": "posterior probability of each hypothesis (exact, by enumeration)",
    "latent": "posterior probability of each level of a nuisance latent (exact, by enumeration)",
    "predictive": "posterior predictive probability of each outcome of a stated next observation (exact)",
    "action": "Bayes action under the stated loss matrix: one-hot on the minimum-expected-loss action (ties split)",
    "acquire": "optimal choice among acting now / abstaining / buying information then acting, by exact expected total cost "
               "including fees (one-hot, ties split)",
    "value_regret": "bin of the exact expected regret of a NAMED action (its expected loss minus the minimum over the world's "
                    "options); bins: 0, then edges at 5 / 15 / 30 / 45 / 70% of the largest STATED loss (one-hot)",
    "value_gap": "bin of the exact gap between the best and second-best options' expected losses; edges at 1 / 2.5 / 5 / 10 / 20% "
                 "of the largest STATED loss (one-hot)",
    "value_voi": "bin of the exact NET value of a named information option (expected loss acting now minus its expected total "
                 "cost, fee included; signed): exactly -fee (cannot change the decision), then edges at -0.5 / 0 / 1 / 3 times "
                 "its STATED fee (one-hot)",
}

# --------------------------------------------------------------------------- families, structures, held-out structures
FAMILIES = {
    "diagnosis":    {"train": {"targeted": 4, "signature": 2, "features": 2}, "heldout": {}},
    "copied":       {"train": {"shared_upstream": 3, "noisy_copy": 3, "independent": 2},
                     "heldout": {"chain_copy": 1, "common_cause": 1}},
    "reliability":  {"train": {"track_record": 3, "latent_sources": 3}, "heldout": {"shared_reliability": 1}},
    "missingness":  {"train": {"mar": 1, "informative": 2}, "heldout": {"mnar_hypothesis": 1}},
    "decision":     {"train": {"asymmetric": 1, "abstain": 2, "matrix": 1}, "heldout": {"avoid": 1}},
    "acquisition":  {"train": {"one_test": 2, "test_or_consult": 2, "collect_more": 1}, "heldout": {"two_step": 1}},
    "hierarchical": {"train": {"member": 1}, "heldout": {"two_factor": 1}},
    "multimodal":   {"train": {"ring_distance": 1, "unidentified_group": 1}, "heldout": {"grid_distance": 1}},
}
FAMILY_WEIGHTS = {"diagnosis": 0.16, "copied": 0.16, "reliability": 0.14, "missingness": 0.11, "decision": 0.12,
                  "acquisition": 0.14, "hierarchical": 0.08, "multimodal": 0.09}
HELDOUT = sorted((f, s) for f, d in FAMILIES.items() for s in d["heldout"])
FAMILY_WEIGHTS_V11 = {"diagnosis": 0.141, "copied": 0.141, "reliability": 0.121, "missingness": 0.097, "decision": 0.175,
                      "acquisition": 0.18, "hierarchical": 0.068, "multimodal": 0.077}
# v1.2: the missingness2 family (hypothesis-dependent absence AND present observations; MAR / hypothesis-dependent / result-dependent
# recording paired on the same data; a latent joint rate regime; a held-out categorical-reading structure) and two CONFIRM-ONLY
# compositions (correlated sources whose reports are selectively retained). FAMILIES / HELDOUT above stay the v1 / v1.1 tables.
FAMILIES_V12 = dict(FAMILIES)
FAMILIES_V12["missingness2"] = {"train": {"tests": 3, "regime": 1}, "heldout": {"signature": 1}}
COMPOSED = {"composed": {"train": {}, "heldout": {"cc_retained": 1, "relay_retained": 1}}}
HELDOUT_V12 = sorted(HELDOUT + [("missingness2", "signature")])
CONFIRM_ONLY = [("composed", "cc_retained"), ("composed", "relay_retained")]
# train mix: GROUP weights, set once from a 300-400-group pilot of units / decision-type units / worlds per group so that the
# unit shares are acquisition 0.31, decision 0.31 (decision-type share ~0.38, v1.1 0.397), missingness2 0.11, diagnosis 0.065,
# copied 0.065, reliability 0.05, missingness 0.03, multimodal 0.03, hierarchical 0.03 (distributions, not items)
FAMILY_WEIGHTS_V12 = {"diagnosis": 0.110, "copied": 0.097, "reliability": 0.062, "missingness": 0.043, "missingness2": 0.118,
                      "decision": 0.233, "acquisition": 0.235, "hierarchical": 0.050, "multimodal": 0.052}


def heldout_set():
    return set(HELDOUT_V12 + CONFIRM_ONLY) if ver() == "v1.2" else set(HELDOUT)


def families():
    return dict(FAMILIES_V12, **COMPOSED) if ver() == "v1.2" else FAMILIES

# --------------------------------------------------------------------------- skins (all text ours; names are invented)
LETTERS = "ABCDEFGHIJKL"
SKINS = {
    "factory": dict(
        setting="A bottling plant runs {n} filling machines on one line, and the line keeps jamming.",
        one="Exactly one machine is causing the jams.", noun="machine", nouns="machines", adj="faulty", isp="is faulty",
        names=[f"Machine {c}" for c in LETTERS], test="a vibration test",
        sig="the error code on the control panel", cats=["code E1", "code E2", "code E3", "code E4"],
        features=["a burning smell", "a rattling noise", "an oil leak", "slow cycle times"],
        reporters=["the shift supervisor", "the night operator", "the quality inspector", "a line technician",
                   "the plant manager", "an apprentice"],
        act="Shut down {h} for repair", abstain="Call in the manufacturer's field engineer",
        rely="Run tomorrow's urgent order on {h} alone",
        consult="Ask the manufacturer's remote support desk to name the faulty machine", agent="the maintenance lead",
        channel="readings from the line's vibration logger", ccats=["low", "medium", "high"], unit="usd"),
    "network": dict(
        setting="A web service runs on {n} servers behind one load balancer, and some requests are failing.",
        one="Exactly one server is at fault.", noun="server", nouns="servers", adj="failing", isp="is failing",
        names=[f"Server {i}" for i in range(1, 13)], test="a memory diagnostic",
        sig="the type of error in the gateway log", cats=["timeout", "reset", "checksum error", "refused connection"],
        features=["latency alerts", "disk warnings", "fan alarms", "clock drift"],
        reporters=["the monitoring dashboard", "the duty engineer", "a customer support summary", "the database team",
                   "an external uptime checker", "a junior administrator"],
        act="Take {h} out of service and replace it", abstain="Fail the whole service over to the backup site",
        rely="Route all priority traffic through {h} alone",
        consult="Ask the hardware vendor's support line to name the failing server", agent="the operations lead",
        channel="packet traces", ccats=["clean", "noisy", "corrupt"], unit="usd"),
    "water": dict(
        setting="A village draws water from {n} wells, and a stomach bug is spreading.",
        one="Exactly one well is contaminated.", noun="well", nouns="wells", adj="contaminated", isp="is contaminated",
        names=[f"Well {i}" for i in range(1, 13)], test="a lab assay",
        sig="the colour of a test strip dipped in the shared tank", cats=["clear", "yellow", "brown", "green"],
        features=["a metallic taste", "cloudy water", "a sulphur smell", "foam on the surface"],
        reporters=["a resident", "the district inspector", "the school nurse", "a farmer", "the village council",
                   "a visiting engineer"],
        act="Close {h} and disinfect it", abstain="Close every well and truck in water for a week",
        rely="Supply the school from {h} alone",
        consult="Ask the regional water board's hydrologist to name the contaminated well",
        agent="the village water officer", channel="turbidity readings at the treatment tank",
        ccats=["low", "medium", "high"], unit="usd"),
    "supply": dict(
        setting="A workshop buys the same bracket from {n} suppliers, and a batch of cracked brackets has turned up.",
        one="Exactly one supplier shipped the bad batch.", noun="supplier", nouns="suppliers", adj="responsible",
        isp="shipped the bad batch",
        names=["Ardent Metals", "Brightforge", "Cobalt Works", "Delta Castings", "Everline Parts", "Fulcrum Fittings",
               "Granite Alloys", "Harbor Steelworks", "Ironvale", "Juniper Fabrication", "Kestrel Forge", "Lumen Hardware"],
        test="a metallurgy assay", sig="the crack pattern under the microscope",
        cats=["hairline", "branching", "radial", "flaking"],
        features=["a matte finish", "burrs on the edges", "a faint maker's stamp", "an unusual weight"],
        reporters=["the receiving clerk", "a customer complaint", "the purchasing assistant", "the external auditor",
                   "a warehouse worker", "the quality engineer"],
        act="Recall every bracket bought from {h}", abstain="Quarantine the whole bracket stock",
        rely="Order the next urgent batch from {h} alone",
        consult="Ask an independent testing lab to name the responsible supplier", agent="the workshop owner",
        channel="hardness readings on fresh samples", ccats=["soft", "normal", "hard"], unit="usd"),
    "orchard": dict(
        setting="An orchard's trees show leaf damage, and {n} blights are known in the area.",
        one="Exactly one of these blights is causing the damage.", noun="blight", nouns="blights", adj="responsible",
        isp="is the cause", names=[f"Blight {c}" for c in LETTERS], test="a lab culture",
        sig="the shape of the leaf spots", cats=["round", "ring-shaped", "streaked", "blotchy"],
        features=["yellow leaf edges", "sticky sap", "curled leaves", "black specks"],
        reporters=["the orchard manager", "a fruit picker", "the extension officer", "a neighbouring grower",
                   "a seasonal worker", "the farm's agronomist"],
        act="Spray the treatment for {h}", abstain="Cut back every affected tree", rely=None,
        consult="Send leaf samples to the plant clinic to name the blight", agent="the grower",
        channel="spore-trap counts", ccats=["low", "medium", "high"], unit="usd"),
    "wildlife": dict(
        setting="A survey team found tracks by a river, made by one of {n} animal species known in the valley.",
        one="Exactly one of these species made the tracks.", noun="species", nouns="species", adj="responsible",
        isp="made the tracks", names=[f"Species {c}" for c in LETTERS], test="a hair-trap genetic test",
        sig="the width class of the prints", cats=["narrow", "medium", "wide", "very wide"],
        features=["claw marks", "a tail drag", "five-toed prints", "droppings nearby"],
        reporters=["a volunteer", "the ranger", "a hiker's note", "a photographer", "a local guide", "a student"],
        act="Record the tracks as {h}", abstain="Record the tracks as unidentified", rely=None,
        consult="Send the photos to the national tracking panel to name the species", agent="the survey leader",
        channel="camera-trap frames", ccats=["blurred", "partial", "clear"], unit="points"),
    "routes": dict(
        setting="A supply convoy must cross the mountains by one of {n} roads, and a rockslide has closed one of them.",
        one="Exactly one road is blocked.", noun="road", nouns="roads", adj="blocked", isp="is blocked",
        names=["North Pass", "South Pass", "River Road", "Ridge Road", "Old Mill Road", "Quarry Road", "Forest Road",
               "Lake Road", "Summit Road", "Valley Road", "Mine Road", "Canyon Road"],
        test="a drone flight", sig="the traffic pattern at the regional junction",
        cats=["light", "normal", "heavy", "stopped"], features=["dust clouds", "rerouted trucks", "a power cut",
                                                                "radio silence"],
        reporters=["a truck driver", "the radio operator", "a park ranger", "a bus company", "a local shopkeeper",
                   "a cyclist"],
        act="Send the road crew to clear {h}", abstain="Hold the convoy for a day", rely="Send the convoy over {h}",
        consult="Ask the regional road authority to name the blocked road", agent="the convoy commander",
        channel="seismic sensor readings", ccats=["quiet", "rumbling", "loud"], unit="usd"),
}
SKIN_ORDER = sorted(SKINS)                       # the in-family skins (v1 and every v1.1 split but test_novel_wording)
NOVEL_SKINS = {
    "lab": dict(
        setting="A research lab keeps its cultures in {n} incubators, and one batch of cultures has been spoiled.",
        one="Exactly one incubator caused the spoilage.", noun="incubator", nouns="incubators", adj="defective",
        isp="is defective", names=[f"Incubator {i}" for i in range(1, 13)], test="a temperature log audit",
        sig="the colour of the spoiled cultures", cats=["pink", "grey", "orange", "white"],
        features=["condensation on the door", "a humming fan", "a drifting thermostat", "a cracked seal"],
        reporters=["the lab supervisor", "a postdoctoral researcher", "the safety officer", "a visiting student",
                   "the facilities technician", "a graduate student"],
        act="Replace the controller in {h}", abstain="Discard every culture and sterilise all incubators",
        rely="Store tomorrow's samples in {h} alone",
        consult="Ask the incubator maker's service centre to name the defective incubator", agent="the lab director",
        channel="readings from the door sensors", ccats=["steady", "flickering", "erratic"], unit="usd"),
    "transit": dict(
        setting="A city runs its buses from {n} depots, and a batch of faulty brake pads was fitted at one of them.",
        one="Exactly one depot fitted the faulty pads.", noun="depot", nouns="depots", adj="responsible",
        isp="fitted the faulty pads",
        names=["Northgate Depot", "Eastfield Depot", "Harbourside Depot", "Millbrook Depot", "Westbury Depot", "Southport Depot",
               "Riverside Depot", "Oakmont Depot", "Ashford Depot", "Brookvale Depot", "Cedar Hill Depot", "Elmstead Depot"],
        test="a brake bench test", sig="the wear pattern on returned pads", cats=["even", "patchy", "glazed", "scored"],
        features=["squealing brakes", "a burning smell on hills", "longer stopping distances", "dashboard warnings"],
        reporters=["a bus driver", "the fleet engineer", "a passenger complaint", "the depot foreman", "a mechanic",
                   "the route supervisor"],
        act="Recall every bus serviced at {h}", abstain="Pull the whole fleet off the road for a day",
        rely="Run the airport route from {h} alone",
        consult="Ask the pad manufacturer's lab to name the responsible depot", agent="the transit operations chief",
        channel="brake temperature readings from the buses", ccats=["cool", "warm", "hot"], unit="usd"),
}
NOVEL_SKIN_ORDER = sorted(NOVEL_SKINS)
SKINS.update(NOVEL_SKINS)                         # every lookup works; SKIN_ORDER (the sampling pool) is unchanged
CONFIRM_SKINS = {   # v1.2 test_confirm only (sealed split): never used in train / val / test / held-out / novel wording
    "bakery": dict(
        setting="A bakery bakes its bread in {n} ovens, and one oven has been scorching the loaves.",
        one="Exactly one oven is scorching the bread.", noun="oven", nouns="ovens", adj="overheating",
        isp="is overheating", names=[f"Oven {i}" for i in range(1, 13)], test="a thermocouple probe",
        sig="the colour of the scorch marks", cats=["amber", "dark brown", "black", "grey"],
        features=["a flickering pilot light", "a warped door", "uneven crusts", "smoke at the vents"],
        reporters=["the head baker", "a pastry cook", "the morning porter", "a delivery driver", "the bakery owner",
                   "a trainee baker"],
        act="Switch off {h} and recalibrate it", abstain="Stop all baking until a technician inspects every oven",
        rely="Bake the wedding order in {h} alone",
        consult="Ask the oven maker's helpline to name the overheating oven", agent="the bakery manager",
        channel="temperature readings from the oven sensors", ccats=["cool", "normal", "hot"], unit="usd"),
    "greenhouse": dict(
        setting="A nursery waters its seedlings with {n} pumps, and one pump has been flooding the trays.",
        one="Exactly one pump is flooding the trays.", noun="pump", nouns="pumps", adj="leaking", isp="is leaking",
        names=["Pump North", "Pump South", "Pump East", "Pump West", "Pump Alpha", "Pump Bravo", "Pump Delta", "Pump Echo",
               "Pump Kilo", "Pump Lima", "Pump Oscar", "Pump Tango"],
        test="a pressure gauge check", sig="the pattern of wet patches", cats=["spots", "rings", "pools", "streaks"],
        features=["a hissing valve", "a damp floor", "algae on the pipes", "a dripping joint"],
        reporters=["the head gardener", "a seasonal picker", "the irrigation fitter", "a visiting buyer", "the nursery owner",
                   "a volunteer"],
        act="Shut off {h} and replace its seals", abstain="Turn off the water to every tray and hand-water for a day",
        rely="Water the rare orchids from {h} alone",
        consult="Ask the pump supplier's engineer to name the leaking pump", agent="the nursery manager",
        channel="flow-meter readings", ccats=["low", "steady", "surging"], unit="usd"),
}
CONFIRM_SKIN_ORDER = sorted(CONFIRM_SKINS)
SKINS.update(CONFIRM_SKINS)
HIER_SKINS = {
    "resin": dict(setting="{G} moulding lines all use plastic resin from the same delivery lot.",
                  snoun="resin lot", grades2=["good", "poor"], grades3=["good", "fair", "poor"],
                  member="Line", thing="mould", bad="failing", good="working", check="a pressure check",
                  factor2="crew", factor2s="crews", f2names=["Crew X", "Crew Y"], f2levels=["steady", "rushed"]),
    "hives": dict(setting="{G} beehives all forage on the same field.", snoun="field",
                  grades2=["clean", "sprayed"], grades3=["clean", "lightly sprayed", "heavily sprayed"],
                  member="Hive", thing="colony", bad="sick", good="healthy", check="a mite count",
                  factor2="keeper", factor2s="keepers", f2names=["Keeper X", "Keeper Y"], f2levels=["careful", "hurried"]),
}
MM_SKINS = {
    "huts": dict(ring="A lost dog is sheltering in one of {N} huts spaced evenly around a lake; Hut {N} stands next to Hut 1.",
                 pos="Hut", sensor="tracker", sensor_desc="a tracker with a scent dog", unit="huts", target="the dog",
                 gtarget="the drone", grtarget="the dog",
                 groups="A lost dog is sheltering in one of {k} huts, which stand in clusters around a lake: {glist}.",
                 gtest="a scent sweep", gword="cluster",
                 grid="A lost drone came down in one of {RC} fields laid out in {R} rows and {C} columns, named by row "
                      "letter and column number.", gpos="Field", gsensor="radio beacon"),
    "valves": dict(ring="A leak is at one of {N} valves spaced evenly around a ring main; Valve {N} sits next to Valve 1.",
                   pos="Valve", sensor="acoustic sensor", sensor_desc="an acoustic sensor", unit="valves",
                   target="the leak", gtarget="the fault", grtarget="the leak",
                   groups="A gas smell comes from one of {k} rooms, which are grouped into wings: {glist}.",
                   gtest="a pressure test of the wing's pipes", gword="wing",
                   grid="A buried cable fault lies under one of {RC} plots laid out in {R} rows and {C} columns, named by "
                        "row letter and column number.", gpos="Plot", gsensor="pulse locator"),
}
LEVEL_NAMES = {2: ["careless", "careful"], 3: ["careless", "average", "careful"],
               4: ["careless", "average", "careful", "meticulous"]}
ACC_MENU = [50, 60, 70, 80, 90, 95]

# --------------------------------------------------------------------------- templates (paraphrase sets; all ours)
TPL = {
    "prior": ["Before any of the evidence below, the chances are: {items}.",
              "Prior chances, from past records and before the evidence below: {items}.",
              "Without the evidence below, the chances would be: {items}."],
    "ev_head": ["Evidence (once you know which {noun} {isp}, each result below is independent of the others):",
                "What has been observed so far (independent results, given which {noun} {isp}):",
                "Evidence collected so far; given the true {noun}, the results are independent of each other:"],
    "ev_plain": ["Evidence:", "What has been reported:", "Reports received so far:"],
    "no_ev": ["No evidence has been collected yet.", "Nothing has been observed yet.",
              "There is no evidence yet beyond the chances above."],
    "tgt": ["{Test} on {name}: it flags with chance {sens} if {name} {isp}, and with chance {fpr} if not. Result: {res}.",
            "{Test} was run on {name}. It flags with chance {sens} when {name} {isp} and with chance {fpr} otherwise. "
            "It {resv}.",
            "Check of {name} by {test}: chance of a flag {sens} if {name} {isp}, otherwise {fpr}. Outcome: {res}."],
    "tgt_miss": ["{Test} on {name} (flags with chance {sens} if {name} {isp}, {fpr} if not): result missing.",
                 "{Test} was run on {name} (chance of a flag {sens} if {name} {isp}, {fpr} otherwise), but its result "
                 "is missing.",
                 "Check of {name} by {test} (flag chance {sens} if {name} {isp}, otherwise {fpr}): no result available."],
    "sig_head": ["{Sig} depends on which {noun} {isp}:", "The chances for {sig}, by which {noun} {isp}:",
                 "{Sig} has these chances:"],
    "sig_row": ["- if {name} {isp}: {items}", "- {name}: {items}", "- when it is {name}: {items}"],
    "sig_obs": ["Readings of {sig} so far (independent given which {noun} {isp}): {items}.",
                "Independent readings of {sig}, given the true {noun}, came out: {items}.",
                "Observed {sig}, reading by reading: {items} (independent given which {noun} {isp})."],
    "sig_none": ["No reading of {sig} has been taken yet.", "{Sig} has not been read yet.",
                 "There are no readings of {sig} yet."],
    "feat": ["{Feature} appears with chance {items}. It is {res}.",
             "Chance of {feature}: {items}. Observed: {res}.",
             "{Feature} shows up with chance {items}; it was found to be {res}."],
    "rep": ["{Rep} names the {adj} {noun} correctly with chance {acc} and otherwise names one of the other {nouns} at "
            "random. {Rep} says it is {claim}.",
            "{Rep} (right with chance {acc}; when wrong, any other {noun} is equally likely) reports: {claim}.",
            "{Rep} reports {claim}. {Rep} is right with chance {acc}, and when wrong names one of the other {nouns}, "
            "each equally likely."],
    "indep_note": ["Unless stated otherwise, sources make their mistakes independently of each other.",
                   "Sources err independently of each other except where a link between them is described.",
                   "Apart from any copying described below, the sources' mistakes are independent."],
    "relay": ["{Rlist} all say it is {claim}. None of them checked anything: each is repeating, word for word, what {O} "
              "told them. {O_cap} is right with chance {acc} and, when wrong, names one of the other {nouns} at random.",
              "Reports from {rlist} all say {claim}. These reports are not independent: every one is a verbatim copy of a "
              "single account by {O}, who is right with chance {acc} (otherwise naming one of the other {nouns} at "
              "random).",
              "{Rlist} each pass on the same claim, {claim}, which all of them got from {O} and repeated unchanged. {O_cap} "
              "is right with chance {acc}; when wrong, any other {noun} is equally likely."],
    "relay_ind": ["{Rlist} all say it is {claim}. Each of them checked independently of the others; each is right with "
                  "chance {acc} and, when wrong, names one of the other {nouns} at random.",
                  "Reports from {rlist} all say {claim}. These reports are independent: each source checked for itself and "
                  "is right with chance {acc} (otherwise naming one of the other {nouns} at random).",
                  "{Rlist} each report the same claim, {claim}, having checked separately. Each is right with chance {acc}; "
                  "when wrong, any other {noun} is equally likely."],
    "copy": ["{A_cap} (right with chance {a}; when wrong, any other {noun} is equally likely) says it is {ca}. {B_cap} says "
             "it is {cb}. {B_cap} sometimes just passes on {A}'s claim: with chance {c}, {B} repeats {A}'s claim without "
             "checking; otherwise {B} checks independently and is right with chance {b} (when wrong, any other {noun} is "
             "equally likely).",
             "{A_cap} says {ca} and {B} says {cb}. {A_cap} is right with chance {a}. {B_cap} copies {A}'s claim unchecked "
             "with chance {c}; otherwise {B} checks independently and is right with chance {b}. Whoever is wrong names "
             "one of the other {nouns} at random.",
             "Two reports: {A} says {ca}, {B} says {cb}. With chance {c}, {B}'s report is simply {A}'s claim passed on "
             "unchecked; otherwise {B} checked independently. {A_cap} is right with chance {a}, {B} (when checking) with "
             "chance {b}; a wrong report names one of the other {nouns} at random."],
    "copy_ind": ["{A_cap} (right with chance {a}; when wrong, any other {noun} is equally likely) says it is {ca}. {B_cap} "
                 "says it is {cb}. {B_cap} checked independently of {A} and is right with chance {b} (when wrong, any "
                 "other {noun} is equally likely).",
                 "{A_cap} says {ca} and {B} says {cb}. {A_cap} is right with chance {a}. {B_cap} checked independently of "
                 "{A} and is right with chance {b}. Whoever is wrong names one of the other {nouns} at random.",
                 "Two reports: {A} says {ca}, {B} says {cb}; they checked independently of each other. {A_cap} is right "
                 "with chance {a}, {B} with chance {b}; a wrong report names one of the other {nouns} at random."],
    "chain_head": ["{O_cap} saw which {noun} {isp} and is right with chance {a} (when wrong, any other {noun} is equally "
                   "likely), but {O}'s own account is not available. Every relay below passes on what it heard correctly "
                   "with the stated chance and otherwise substitutes one of the other {nouns} at random.",
                   "The only first-hand witness was {O}, who is right with chance {a} (otherwise naming one of the other "
                   "{nouns} at random); {O}'s account itself is lost. Each person below heard it second-hand; each passes "
                   "on what they heard correctly with the stated chance, and otherwise names one of the other {nouns} at "
                   "random.",
                   "All reports trace back to {O}, right with chance {a} (when wrong, any other {noun} equally likely), "
                   "whose own account is unavailable. A relay repeats what it heard correctly with its stated chance; "
                   "otherwise it names one of the other {nouns} at random."],
    "chain_direct": ["{R_cap} heard it directly from {O} (passes it on correctly with chance {f}) and says it is {claim}.",
                     "{R_cap}, told directly by {O} (correct relay with chance {f}), says {claim}.",
                     "{R_cap} got it straight from {O}, relays correctly with chance {f}, and reports {claim}."],
    "chain_two": ["{R_cap} heard it from {M}, who heard it from {O}; {M} relays correctly with chance {fm} and {R} with "
                  "chance {f}. {R_cap} says it is {claim}.",
                  "{R_cap} was told by {M} (correct relay with chance {fm}), who was told by {O}; {R} relays correctly "
                  "with chance {f}. {R_cap} says {claim}.",
                  "{R_cap} reports {claim}, having heard it from {M}, who had it from {O}. Correct relay chances: {M} "
                  "{fm}, {R} {f}."],
    "cc": ["{slist} each name the {adj} {noun}. How reliable they are depends on visibility, which nobody recorded: with "
           "chance {pf} it was foggy, and otherwise clear. In clear conditions each is right with chance {ac}; in fog each "
           "is right with chance {af} (when wrong, any other {noun} is equally likely). Given the conditions, their "
           "mistakes are independent. {claims}.",
           "{slist} each name the {adj} {noun}. Their accuracy depends on the weather, which was not recorded: foggy with "
           "chance {pf}, clear otherwise. Each is right with chance {ac} when clear and {af} in fog, erring independently "
           "given the weather (a wrong answer names one of the other {nouns} at random). {claims}.",
           "{slist} name the {adj} {noun}. With chance {pf} the day was foggy (it was not recorded); each is right with "
           "chance {af} in fog and {ac} in clear weather, and given the weather their errors are independent, a wrong "
           "answer naming one of the other {nouns} at random. {claims}."],
    "cc_ind": ["{slist} each name the {adj} {noun}. Each is right with chance {am} and, when wrong, names one of the other "
               "{nouns} at random, independently of the others. {claims}.",
               "{slist} each name the {adj} {noun}, independently of each other; each is right with chance {am} "
               "(otherwise any other {noun} equally likely). {claims}.",
               "{slist} name the {adj} {noun}; each is right with chance {am}, and their mistakes are independent, a "
               "wrong answer naming one of the other {nouns} at random. {claims}."],
    "lat_src": ["{Src} is one of {L} kinds of source, and it is not known which: {levels}. Before any evidence, the chances "
                "are {lprior}.",
                "It is not known how reliable {src} is. It is one of these kinds: {levels}; beforehand the chances are "
                "{lprior}.",
                "{Src}'s reliability is uncertain: {levels}, with prior chances {lprior}."],
    "lat_shared": ["{Slist} all work for the same agency and share its reliability, which is not known: {levels}. Before "
                   "any evidence, the chances are {lprior}.",
                   "{Slist} come from one agency, so all of them have the agency's reliability, one of: {levels}; prior "
                   "chances {lprior}.",
                   "The agency that employs {slist} is one of these kinds (all its staff share it): {levels}. Beforehand "
                   "the chances are {lprior}."],
    "lat_rule": ["A source of any kind, when wrong, names one of the other {nouns} at random. Given which {noun} {isp} and "
                 "the sources' kinds, sources err independently of each other.",
                 "Wrong answers name one of the other {nouns}, each equally likely; once the true {noun} and every source's "
                 "kind are fixed, the sources' mistakes are independent.",
                 "When a source is wrong it names one of the other {nouns} at random, and given the truth and the kinds, "
                 "different sources (and different claims) err independently."],
    "track": ["Of {n} earlier claims by {src} that could later be checked, {s} were right.",
              "{Src} has a track record: {s} right out of {n} checkable earlier claims.",
              "Earlier, {src} made {n} claims that were later checked; {s} of them were right."],
    "track0": ["None of {src}'s earlier claims could be checked.", "{Src} has no checkable track record.",
               "There is no track record for {src}."],
    "track_ag": ["Of {n} earlier claims by the agency's staff that could later be checked, {s} were right.",
                 "The agency's track record: {s} right out of {n} checkable earlier claims.",
                 "Earlier, the agency's staff made {n} claims that were later checked; {s} were right."],
    "lrep": ["{Src} says it is {claim}.", "{Src} reports: {claim}.", "According to {src}, it is {claim}."],
    "mar": ["Some results are missing because a data logger failed at random; whether a result went missing had nothing to "
            "do with what it showed or with which {noun} {isp}.",
            "A few results were lost in a random computer crash, unrelated to what they showed.",
            "Some results are missing by accident: the losses happened at random, independently of the results and of "
            "which {noun} {isp}."],
    "inf": ["Results do not always get written down: a result that flagged is recorded with chance {qf}, and one that did "
            "not flag is recorded with chance {qc}.",
            "Not every result is logged. A flag is logged with chance {qf}; a result without a flag is logged with chance "
            "{qc}.",
            "Whether a result is recorded depends on what it showed: with chance {qf} if it flagged and with chance {qc} "
            "if it did not."],
    "mnar": ["Each {noun}'s own log can go missing: if a {noun} {isp}, its log is missing with chance {m1}; otherwise with "
             "chance {m0}. A log that is present flags with chance {sens} if its {noun} {isp} and with chance {fpr} "
             "otherwise.",
             "Logs go missing more often for the {adj} {noun}: chance {m1} for it, {m0} for any other. A present log flags "
             "with chance {sens} for the {adj} {noun} and {fpr} for any other.",
             "A {noun}'s log is missing with chance {m1} if that {noun} {isp} and {m0} if not; when present, it flags "
             "with chance {sens} for the {adj} {noun} and {fpr} otherwise."],
    "mnar_mar": ["Some logs are missing because of a random fault, unrelated to which {noun} {isp}. A log that is present "
                 "flags with chance {sens} if its {noun} {isp} and with chance {fpr} otherwise.",
                 "A few logs were lost at random, independently of which {noun} {isp}. A present log flags with chance {sens} "
                 "for the {adj} {noun} and {fpr} for any other.",
                 "Logs went missing by accident, whatever the cause of the problem; when present, a log flags with chance "
                 "{sens} if its {noun} {isp} and {fpr} otherwise."],
    "hier_prior2": ["Beforehand, the chances for the {snoun} are {sprior}.", "Prior chances for the {snoun}: {sprior}.",
                    "The {snoun}'s grade has prior chances {sprior}."],
    "log_line": ["Log of {name}: {res}.", "{name}'s log: {res}.", "Log from {name}: {res}."],
    "cost_head": ["Costs of the possible actions:", "The actions available and what they cost:", "What each action costs:"],
    "fix": ["- {act}: costs {F}; if {name} is not the {adj} {noun}, a further {M} is lost.",
            "- {act}: {F} up front, plus {M} more if {name} turns out not to be the {adj} {noun}.",
            "- {act}: costs {F}, and another {M} if it is the wrong {noun}."],
    "abst": ["- {abstain}: costs {E}, whichever {noun} {isp}.", "- {abstain}: {E} in total, regardless of the cause.",
             "- {abstain}: a flat {E}."],
    "mat": ["- {act}: costs {items}.", "- {act}: {items}.", "- {act}: total cost {items}."],
    "avoid": ["- {act}: costs {B}; if {name} {isp}, a further {P} is lost.",
              "- {act}: {B}, plus {P} more if {name} turns out to be the {adj} {noun}.",
              "- {act}: costs {B}, and another {P} if {name} {isp}."],
    "info_head": ["Before acting, {agent} may first pay for more information and then choose in light of the result:",
                  "{Agent} can also buy information first, and then decide what to do:",
                  "Information that {agent} can pay for before deciding (any action is then chosen after seeing the "
                  "result):"],
    "info_tgt": ["- {Test} on {name}, fee {fee}: flags with chance {sens} if {name} {isp}, and with chance {fpr} if not.",
                 "- {Test} on {name} (fee {fee}): chance of a flag {sens} if {name} {isp}, {fpr} otherwise.",
                 "- For a fee of {fee}, {test} on {name}, which flags with chance {sens} if {name} {isp} and {fpr} if not."],
    "info_con": ["- {Consult}, fee {fee}: the answer names the {adj} {noun} correctly with chance {acc}; when wrong, it "
                 "names one of the other {nouns} at random.",
                 "- {Consult} (fee {fee}): right with chance {acc}, otherwise naming one of the other {nouns} at random.",
                 "- For a fee of {fee}: {consult_l}; the answer is right with chance {acc} and otherwise names one of the "
                 "other {nouns} at random."],
    "info_col": ["- More {channel}, fee {fee} per reading: {agent} can take 1, 2 or 3 readings, independent given which "
                 "{noun} {isp}, with these chances:",
                 "- Further {channel} at {fee} each (1, 2 or 3 of them, independent given the true {noun}); chances:",
                 "- {Agent} may buy 1, 2 or 3 more {channel} for {fee} each; given which {noun} {isp} they are "
                 "independent, with these chances:"],
    "two_note": ["After seeing the first test's result, {agent} may decide whether to run the other test as well.",
                 "Once the first result is in, {agent} can choose to run the other test too, or not.",
                 "Whichever test is run first, {agent} may run the other one afterwards in light of its result."],
    "hier_model": ["Beforehand, the chances for the {snoun} are {sprior}. Given its grade, each {member_l}'s {thing} is "
                   "{bad} independently, with chance {fails}.",
                   "Prior chances for the {snoun}: {sprior}. Each {member_l}'s {thing} is {bad} with chance {fails}, "
                   "independently of the others once the grade is known.",
                   "The {snoun}'s grade has prior chances {sprior}; given the grade, every {member_l}'s {thing} is "
                   "{bad} independently with chance {fails}."],
    "hier_crew": ["Each {member_l} is also looked after by one of two {factor2s}, whose state is not known: {crewmap}. Each "
                  "{factor2} is {lv0} or {lv1}; beforehand, the chance of {lv1} is {cp} for each, independently. Given "
                  "the {snoun}'s grade and the {factor2}'s state, a {member_l}'s {thing} is {bad} independently, with "
                  "chance {table}.",
                  "Two {factor2s} look after the {member_l}s ({crewmap}); each {factor2} is {lv0} or {lv1}, {lv1} with "
                  "prior chance {cp}, independently. Chance that a {member_l}'s {thing} is {bad}, independently given "
                  "both: {table}.",
                  "The {member_l}s are split between two {factor2s}: {crewmap}. A {factor2} is {lv1} with prior chance {cp} "
                  "(else {lv0}), independently of the other. Given the grade and its {factor2}'s state, a {member_l}'s "
                  "{thing} is {bad} with chance {table}, independently."],
    "hier_check": ["{Check} on {member}: flags with chance {sens} if its {thing} is {bad} and {fpr} if {good}. Result: "
                   "{res}.",
                   "{member} got {check} (flag chance {sens} if {bad}, {fpr} if {good}); it {resv}.",
                   "{Check} of {member} (flags with chance {sens} when {bad}, {fpr} when {good}): {res}."],
    "hier_none": ["{member} has not been checked.", "No check has been run on {member}.", "{member} is unchecked."],
    "ring_read": ["The {sensor} at {pos} indicated a distance of {r} (in {unit}, counting the shorter way round).",
                  "From {pos}, the {sensor} gave a reading of {r} {unit} away (the shorter way round).",
                  "Reading of the {sensor} placed at {pos}: {r}."],
    "ring_rule": ["{Sensor_desc} reports how far away {target} is (the shorter way round): exactly right with chance {pe}; "
                  "otherwise one too low or one too high, each equally likely, except that an impossible reading is "
                  "replaced by the exact one. Readings are independent given the position of {target}.",
                  "A reading is the exact distance with chance {pe}; otherwise it is off by one in either direction with "
                  "equal chance, an impossible value being replaced by the exact distance. Separate readings are "
                  "independent given where {target} is.",
                  "Each reading equals the true distance with chance {pe}, or else is one less or one more (equally "
                  "likely; if that value is impossible, the exact distance is reported instead), independently from "
                  "reading to reading."],
    "grid_rule": ["The {gsensor} reports the distance to {target} in rows apart plus columns apart: exactly right with "
                  "chance {pe}; otherwise one too low or one too high, each equally likely, except that an impossible "
                  "reading is replaced by the exact one. Readings are independent given the position of {target}.",
                  "A {gsensor} reading is the true (rows plus columns) distance with chance {pe}; otherwise it is off by "
                  "one in either direction with equal chance, an impossible value being replaced by the exact distance; "
                  "readings are independent given the position of {target}.",
                  "Each {gsensor} reading equals the rows-plus-columns distance with chance {pe}, or else is one less or "
                  "one more (equally likely; an impossible value is replaced by the exact distance), independently."],
    "grid_read": ["The {gsensor} at {pos} gave a distance of {r}.", "From {pos}, the {gsensor} read {r}.",
                  "Reading of the {gsensor} at {pos}: {r}."],
    "grp_read": ["{Gtest} on the {gname} {gword}: flags with chance {sens} if {target} is in that {gword} and {fpr} "
                 "if not. Result: {res}.",
                 "{Gtest} was run on the {gname} {gword} (chance of a flag {sens} if {target} is there, {fpr} "
                 "otherwise); it {resv}.",
                 "{Gtest} of the {gname} {gword}, which flags with chance {sens} if {target} is in it and {fpr} "
                 "otherwise: {res}."],
    "q_post": ["Posterior question: given the prior chances and all the evidence above, which {noun} {isp}? Give the "
               "posterior probability of each option.",
               "Which {noun} {isp}? Answer with your posterior probabilities, taking into account the stated prior and "
               "every piece of evidence.",
               "Posterior question: after all the evidence above, how probable is each {noun}? (Exactly one {isp}.)"],
    "q_hpost": ["Posterior question: given everything above, is {member}'s {thing} {bad}? Give the posterior probability of "
                "each option.",
                "What is the posterior probability that {member}'s {thing} is {bad}, given all of the above?",
                "Posterior question about {member}: taking every check above into account, is its {thing} {bad} or "
                "{good}?"],
    "q_pred": ["Predictive question: {desc} What will it show? Give the predictive probability of each possible result, "
               "given everything above.",
               "{desc} Predict the result: give the probability of each possible outcome, averaging over everything "
               "that is still uncertain.",
               "Predictive question. {desc} How likely is each result, given everything above?"],
    "q_lat": ["Posterior question about a source: which kind of source is {src}? Give the posterior probability of each "
              "kind, given everything above.",
              "How reliable is {src}? Give the posterior probability of each kind, taking all of the above into account.",
              "Posterior question: given everything above, what kind of source is {src}?"],
    "q_hlat": ["Posterior question about the shared {snoun}: which grade is it? Give the posterior probability of each "
               "grade, given everything above.",
               "What grade is the {snoun}? Give the posterior probability of each grade, taking every check into account.",
               "Posterior question: given all the checks above, how probable is each grade of the {snoun}?"],
    "q_act": ["Decision question: given the prior, the evidence and the costs above, which action has the lowest expected "
              "cost? Choose one action.",
              "Which action should be taken now to make the expected cost as low as possible? Choose one.",
              "Decision question: taking the chances and the costs above into account, which action minimises the "
              "expected cost?"],
    "q_acq": ["Decision question: what should {agent} do now to make the expected total cost (including any fee) as low as "
              "possible? Choose one option.",
              "Counting both the fees and the expected cost of the final action, which option should {agent} choose now?",
              "Decision question: which choice now gives the lowest expected total cost, fees included?"],
}
TPL_KEYS = sorted(TPL)
TPL11 = {   # v1.1 value questions (kept apart so v1's template draws are unchanged)
    "q_regret": ["Value question: compared with {best}, how much more would choosing '{act}' cost in expectation, given "
                 "everything above? Choose the range that contains this expected extra cost (its expected regret).",
                 "Expected-regret question: how much higher is the expected cost of '{act}' than that of {best}? Choose the "
                 "range that contains it.",
                 "Value question: by how much does the expected cost of '{act}' exceed that of {best}, given all of the above? "
                 "Pick the range."],
    "q_gap": ["Value question: given everything above, by how much does the expected cost of the second-best {nc} exceed that "
              "of the best {nc}? Choose the range that contains this gap.{note}",
              "How large is the gap in expected cost between the best and the second-best {ncs} available? Choose the "
              "range.{note}",
              "Value question: what is the difference in expected cost between the best and the second-best {nc} above? Pick "
              "the range.{note}"],
    "q_voi": ["Value question: what is the net value of the option '{opt}', that is, the expected cost of the best action taken "
              "now minus the expected total cost of taking that option (fee included)? A negative value means it is not "
              "worth its fee. Choose the range.",
              "Net-value-of-information question: compared with acting now, how much does '{opt}' save in expectation after "
              "paying for it (negative if it costs more than it saves)? Choose the range.",
              "Value question: taking '{opt}' instead of the best action now changes the expected total cost by how much, fee "
              "included? Choose the range of the net saving (negative if it is a net loss)."],
}
TPL11_KEYS = sorted(TPL11)
# v1.2: template variant 3 of every v1 / v1.1 key, used ONLY in the sealed test_confirm split (held-out wording). Kept apart from TPL /
# TPL11 because v1's template draws depend on the length of those lists.
TPL_CONFIRM = {
    "prior": "Going by past records alone, before the evidence below, the chances are: {items}.",
    "ev_head": "The evidence so far (given which {noun} {isp}, the results below are independent of one another):",
    "ev_plain": "Accounts received:",
    "no_ev": "No evidence is available yet; only the chances above.",
    "tgt": "{Test} applied to {name} (a flag comes up with chance {sens} if {name} {isp} and with chance {fpr} otherwise) gave: "
           "{res}.",
    "tgt_miss": "{Test} applied to {name} (a flag comes up with chance {sens} if {name} {isp} and with chance {fpr} otherwise) "
                "gave no recorded result.",
    "sig_head": "Chances of each value of {sig}, depending on which {noun} {isp}:",
    "sig_row": "- supposing {name} {isp}: {items}",
    "sig_obs": "One reading after another, {sig} came out: {items} (independent given which {noun} {isp}).",
    "sig_none": "Nobody has read {sig} yet.",
    "feat": "{Feature}: chance {items}. Finding: {res}.",
    "rep": "{Rep} names {claim}. {Rep} gets it right with chance {acc}; a wrong answer would name one of the other {nouns}, each "
           "equally likely.",
    "indep_note": "Except for any copying described below, each source's mistakes are independent of every other source's.",
    "relay": "{Rlist} each say {claim}, but none of them checked: they are all repeating word for word what {O} told them. {O_cap} "
             "gets it right with chance {acc} and otherwise names one of the other {nouns} at random.",
    "relay_ind": "{Rlist} each say {claim}, and each of them checked separately from the others; each gets it right with chance "
                 "{acc} and otherwise names one of the other {nouns} at random.",
    "copy": "{A_cap} says {ca}; {A} is right with chance {a}, and a wrong answer names one of the other {nouns} at random. {B_cap} "
            "says {cb}. With chance {c}, {B} simply repeated {A}'s claim without checking; otherwise {B} checked independently, "
            "being right with chance {b} (a wrong answer names one of the other {nouns} at random).",
    "copy_ind": "{A_cap} says {ca}; {A} is right with chance {a}, and a wrong answer names one of the other {nouns} at random. "
                "{B_cap} says {cb}, having checked independently of {A}; {B} is right with chance {b}, with the same rule for wrong "
                "answers.",
    "chain_head": "Everything traces back to {O}, who saw which {noun} {isp} and is right with chance {a} (otherwise naming one of "
                  "the other {nouns} at random); {O}'s own account cannot be found. Each relay below passes on what it heard "
                  "correctly with the chance given, and otherwise substitutes one of the other {nouns} at random.",
    "chain_direct": "{R_cap} says {claim}, having heard it from {O} in person (correct relay chance {f}).",
    "chain_two": "{R_cap} says {claim}, having heard it from {M}, who heard it from {O} (correct relay chances: {M} {fm}, then {R} "
                 "{f}).",
    "cc": "{slist} each name the {adj} {noun}. The weather that day was not recorded: it was foggy with chance {pf}, otherwise "
          "clear. Each is right with chance {ac} in clear weather and {af} in fog (a wrong answer names one of the other {nouns} at "
          "random), and given the weather their errors are independent. {claims}.",
    "cc_ind": "{slist} each name the {adj} {noun}, each right with chance {am}; their mistakes are independent, and a wrong answer "
              "names one of the other {nouns} at random. {claims}.",
    "lat_src": "How reliable {src} is remains unknown. {Src} is one of these kinds: {levels}; before any evidence the chances are "
               "{lprior}.",
    "lat_shared": "{Slist} are all staff of one agency and share its reliability, which is unknown: {levels}. The chances before "
                  "any evidence: {lprior}.",
    "lat_rule": "Any source that is wrong names one of the other {nouns}, each equally likely; and with the true {noun} and every "
                "source's kind fixed, mistakes by different sources (or by one source on different claims) are independent.",
    "track": "{Src}'s earlier claims that could be checked later: {n}, of which {s} were right.",
    "track0": "{Src} made no earlier claims that could be checked.",
    "track_ag": "The agency's staff made {n} earlier claims that could be checked later; {s} of those were right.",
    "lrep": "{Src} names {claim}.",
    "mar": "Some results went missing by pure chance (a corrupted file), unconnected with what they showed or with which {noun} "
           "{isp}.",
    "inf": "Results are only sometimes written up, depending on the outcome: a flag gets written up with chance {qf}, a result "
           "without a flag with chance {qc}.",
    "mnar": "A {noun}'s own log goes missing with chance {m1} if that {noun} {isp}, and with chance {m0} otherwise; a log that "
            "survives flags with chance {sens} if its {noun} {isp}, and with chance {fpr} otherwise.",
    "mnar_mar": "Logs went missing because of a random fault that has nothing to do with which {noun} {isp}; a log that survives "
                "flags with chance {sens} if its {noun} {isp}, and with chance {fpr} otherwise.",
    "hier_prior2": "Chances for the {snoun} before any check: {sprior}.",
    "log_line": "{name}, log: {res}.",
    "cost_head": "Each available action, with its cost:",
    "fix": "- {act}: {F}, and if {name} is not in fact the {adj} {noun}, an extra {M}.",
    "abst": "- {abstain}: {E}, the same whichever {noun} {isp}.",
    "mat": "- {act}: the cost is {items}.",
    "avoid": "- {act}: {B}, and an extra {P} if {name} {isp}.",
    "info_head": "{Agent} may first buy information and only then pick an action, knowing the result:",
    "info_tgt": "- {Test} on {name} for a fee of {fee}: a flag comes up with chance {sens} if {name} {isp} and with chance {fpr} "
                "otherwise.",
    "info_con": "- {Consult}, for a fee of {fee}: the answer picks out the {adj} {noun} with chance {acc} and otherwise names one "
                "of the other {nouns} at random.",
    "info_col": "- Extra {channel}, {fee} for each reading: {agent} may take 1, 2 or 3 of them, independent given which {noun} "
                "{isp}, with these chances:",
    "two_note": "Having seen the first result, {agent} can still decide whether to run the remaining test.",
    "hier_model": "Chances for the {snoun} before any check: {sprior}. Given the grade, each {member_l}'s {thing} is {bad} with "
                  "chance {fails}, independently of every other {member_l}.",
    "hier_crew": "Each {member_l} is also tended by one of two {factor2s} ({crewmap}), whose state is unknown: each {factor2} is "
                 "{lv0} or {lv1}, {lv1} with chance {cp} beforehand, independently of the other. Given the {snoun}'s grade and "
                 "its {factor2}'s state, a {member_l}'s {thing} is {bad} with chance {table}, independently.",
    "hier_check": "{Check} of {member} (a flag comes up with chance {sens} if its {thing} is {bad} and with chance {fpr} if "
                  "{good}) gave: {res}.",
    "hier_none": "Nobody has checked {member}.",
    "ring_read": "The {sensor} placed at {pos} read {r} (in {unit}, along the shorter way round).",
    "ring_rule": "{Sensor_desc} gives the distance to {target} along the shorter way round: the exact distance with chance {pe}, "
                 "otherwise one less or one more with equal chance (an impossible value is replaced by the exact distance); "
                 "separate readings are independent given where {target} is.",
    "grid_rule": "The {gsensor} gives the distance to {target} as rows apart plus columns apart: the exact distance with chance "
                 "{pe}, otherwise one less or one more with equal chance (an impossible value is replaced by the exact "
                 "distance); separate readings are independent given where {target} is.",
    "grid_read": "The {gsensor} placed at {pos} read {r}.",
    "grp_read": "{Gtest} of the {gname} {gword} (a flag comes up with chance {sens} if {target} is in that {gword} and {fpr} "
                "otherwise) gave: {res}.",
    "q_post": "Posterior question: taking the prior chances and every piece of evidence into account, what is the probability "
              "that each {noun} {isp}?",
    "q_hpost": "Posterior question: in light of everything above, how probable is it that {member}'s {thing} is {bad}, and how "
               "probable that it is {good}?",
    "q_pred": "Predictive question. {desc} Give the probability of each possible result, given all of the above.",
    "q_lat": "Posterior question: in light of everything above, how probable is each kind for {src}?",
    "q_hlat": "Posterior question: in light of all the checks, how probable is each grade of the shared {snoun}?",
    "q_act": "Decision question: with the chances and costs above, which single action gives the lowest expected cost?",
    "q_acq": "Decision question: to keep the expected total cost (fees included) as low as possible, which option should {agent} "
             "take now?",
    "q_regret": "Value question: given all of the above, how much more is '{act}' expected to cost than {best}? Pick the range "
                "containing this expected extra cost (its expected regret).",
    "q_gap": "Value question: how far apart are the expected costs of the best and the second-best {ncs}, given all of the above? "
             "Pick the range.{note}",
    "q_voi": "Value question: compared with the best action available now, how much does '{opt}' save in expectation once its fee "
             "is paid (a negative amount if it costs more than it saves)? Pick the range.",
}
assert set(TPL_CONFIRM) == set(TPL) | set(TPL11)
# v1.2 keys (missingness2 and the confirm-only compositions): variants 0-1 = standard wording, 2 = novel wording, 3 = confirm.
TPL12 = {
    "m2_mar": ["Some results are missing because a data logger failed at random: each result went missing with chance {rate}, "
               "whatever it showed and whichever {noun} {isp}.",
               "Results were lost at random, each with chance {rate}; a loss had nothing to do with what the result showed or "
               "with which {noun} {isp}.",
               "Any result could have been dropped by a random fault, with chance {rate} each, regardless of its outcome and of "
               "which {noun} {isp}.",
               "Each result had a {rate} chance of being lost to a random glitch, unrelated both to the result itself and to "
               "which {noun} {isp}."],
    "m2_hyp": ["Whether a result reaches the log depends on which {noun} {isp}, not on what the test showed: the result of a test "
               "on the {adj} {noun} goes missing with chance {m1}, and the result of a test on any other {noun} goes missing with "
               "chance {m0}, independently from test to test.",
               "Losses depend on the truth, not on the outcome: a test result about the {adj} {noun} is lost with chance {m1}, a "
               "result about any other {noun} with chance {m0}; each result is lost or kept independently of the others and of "
               "what it showed.",
               "How often a result goes missing depends only on whether the tested {noun} is the {adj} one: chance {m1} if it is "
               "and {m0} if it is not, regardless of the result, independently for each test.",
               "A result is dropped with chance {m1} if the {noun} it tested {isp}, and with chance {m0} otherwise; what the test "
               "showed plays no part, and drops happen independently from test to test."],
    "m2_res": ["Whether a result is recorded depends only on what it showed: a result that flagged is recorded with chance {qf}, "
               "and one that did not flag with chance {qc}, whichever {noun} {isp}, independently from test to test.",
               "Recording depends on the outcome alone: a flag gets logged with chance {qf} and a result without a flag with "
               "chance {qc}; given its outcome, whether a result is logged has nothing to do with which {noun} {isp}.",
               "A flagged result is kept with chance {qf} and an unflagged one with chance {qc}; once the outcome is fixed, "
               "keeping it does not depend on which {noun} {isp}, and each result is kept or lost independently.",
               "What a test showed decides how likely its result is to be filed: chance {qf} for a flag, {qc} for no flag, "
               "independently for each test and, given the outcome, regardless of which {noun} {isp}."],
    "m2_reg_head": ["Not every result reaches the log, and it is not known which recording rule applies; one rule applies to "
                    "every test. The possible rules and their chances:",
                    "The recording rule is uncertain; it is the same for all the tests and is one of the following (with the "
                    "chances given):",
                    "Which of these recording rules is in force is unknown, but a single rule governs every test; the candidates "
                    "and their chances are:",
                    "One recording rule governs all of the tests, but nobody knows which one; it is one of these, with the "
                    "stated chances:"],
    "m2_reg_hyp": ["- rule {R} (chance {p}): the result of a test on the {adj} {noun} goes missing with chance {m1}, and the "
                   "result of a test on any other {noun} with chance {m0}, whatever the test showed.",
                   "- rule {R}, chance {p}: a result about the {adj} {noun} is lost with chance {m1}, a result about any other "
                   "{noun} with chance {m0}, regardless of the outcome.",
                   "- rule {R} (prior chance {p}): results on the {adj} {noun} vanish with chance {m1} and results on other "
                   "{nouns} with chance {m0}, whatever they showed.",
                   "- rule {R}, with chance {p}: drop chance {m1} for a test on the {adj} {noun} and {m0} for a test on any "
                   "other {noun}, whatever the outcome."],
    "m2_reg_res": ["- rule {R} (chance {p}): a result that flagged is recorded with chance {qf}, and one that did not flag with "
                   "chance {qc}.",
                   "- rule {R}, chance {p}: a flag is logged with chance {qf} and a result without a flag with chance {qc}.",
                   "- rule {R} (prior chance {p}): flagged results are kept with chance {qf} and unflagged ones with chance "
                   "{qc}.",
                   "- rule {R}, with chance {p}: filing chance {qf} for a flag and {qc} for no flag."],
    "m2_reg_tail": ["Given the rule, each result is recorded or lost independently of the others.",
                    "Once the rule is fixed, results are logged or lost independently.",
                    "Under whichever rule applies, each result's fate is independent of the others.",
                    "Given the rule in force, every result is filed or dropped independently."],
    "m2s_mar": ["Some readings of {sig} are missing because the recorder failed at random: each reading went missing with chance "
                "{rate}, whatever it showed and whichever {noun} {isp}.",
                "Readings of {sig} were lost at random, each with chance {rate}, unrelated to what they showed and to which "
                "{noun} {isp}.",
                "Any reading of {sig} could have been dropped by a random fault (chance {rate} each), regardless of its value "
                "and of which {noun} {isp}.",
                "Each reading of {sig} had a {rate} chance of being lost to a random glitch, unrelated to the reading and to "
                "which {noun} {isp}."],
    "m2s_hyp": ["Readings of {sig} go missing more or less often depending on the cause: each reading is lost with chance {m1} if "
                "{name} {isp}, and with chance {m0} otherwise, whatever the reading showed, independently from reading to "
                "reading.",
                "How often a reading of {sig} is lost depends on the truth, not on its value: chance {m1} if {name} {isp}, "
                "chance {m0} if not; readings are lost or kept independently.",
                "Whether a reading of {sig} survives depends only on whether {name} {isp}: it is lost with chance {m1} if so and "
                "{m0} if not, regardless of its value, independently for each reading.",
                "A reading of {sig} is dropped with chance {m1} if {name} {isp} and with chance {m0} otherwise; its value plays "
                "no part, and drops happen independently."],
    "m2s_res": ["Whether a reading of {sig} is recorded depends only on what it showed: {items}; given what it showed, this does "
                "not depend on which {noun} {isp}, and readings are recorded or lost independently.",
                "Recording depends on the value read alone ({items}); given the value, it has nothing to do with which {noun} "
                "{isp}, and each reading is kept or lost independently.",
                "A reading of {sig} is kept with a chance that depends on its value: {items}; once the value is fixed, keeping it "
                "does not depend on which {noun} {isp}, independently from reading to reading.",
                "The value of a reading decides how likely it is to be filed: {items}; independently for each reading and, given "
                "the value, regardless of which {noun} {isp}."],
    "m2s_head": ["Readings of {sig} taken so far (independent given which {noun} {isp}):",
                 "The readings of {sig}, one by one (independent given the true {noun}):",
                 "Each reading of {sig} that was taken (independent given which {noun} {isp}):",
                 "Readings of {sig}, in the order they were taken (independent given which {noun} {isp}):"],
    "m2s_read": ["- Reading {i}: {res}.", "- Reading number {i}: {res}.", "- Reading {i} of {sig}: {res}.",
                 "- The reading taken {ordinal}: {res}."],
    "m2_x_head": ["How each result line above enters the calculation. Given which {noun} {isp}, each line contributes the factor "
                  "below (the chance of what was seen, including whether the result was recorded):",
                  "The likelihood factor of each result line, given which {noun} {isp} (the recording process is already "
                  "included):",
                  "Each result line multiplies the chances by a factor that depends on which {noun} {isp}; with the recording "
                  "rule taken into account, the factors are:",
                  "Spelled out, the evidence (with the recording rule built in) contributes these factors, depending on which "
                  "{noun} {isp}:"],
    "m2_x_line": ["- {what}: {a} if {name} {isp}, {b} otherwise.", "- {what}: factor {a} if {name} {isp}, {b} if not.",
                  "- {what}: {a} when {name} {isp}; {b} when it is another {noun}.",
                  "- {what}: multiply by {a} if {name} {isp} and by {b} otherwise."],
    "m2_x_vec": ["- {what}: {items}.", "- {what}: factor {items}.", "- {what}: {items}, by cause.",
                 "- {what}: multiply by {items}."],
    "m2_x_reg": ["- {what}: {parts}.", "- {what}: {parts}.", "- {what} -- {parts}.", "- {what}: {parts}."],
    "m2_x_tail": ["The posterior is proportional to the prior chance times the product of all these factors.",
                  "Multiply the prior chance of each {noun} by all of its factors and renormalise to get the posterior.",
                  "Posterior chances are proportional to prior chance times the product of the factors.",
                  "To get the posterior, multiply each prior chance by every factor that applies and rescale so the results sum "
                  "to one."],
    "m2_x_tail_reg": ["For each rule, multiply its factors; weight each product by the rule's chance and add them; the posterior is "
                      "proportional to the prior chance times that sum.",
                      "Take the product of the factors under each rule, weight it by that rule's chance, add over the rules, "
                      "multiply by the prior chance of each {noun} and renormalise.",
                      "The posterior is proportional to the prior chance times the chance-weighted sum, over the rules, of the "
                      "product of each rule's factors.",
                      "Combine the rules by adding their factor products, each weighted by the rule's chance; multiply by the "
                      "prior chance and rescale to get the posterior."],
    "q_reg": ["Posterior question about the recording: which rule applies? Give the posterior probability of each rule, given "
              "everything above.",
              "Which recording rule is in force? Answer with your posterior probabilities, taking all of the above into account.",
              "Posterior question: after everything above, how probable is each recording rule?",
              "Given all of the above, what is the posterior probability of each candidate recording rule?"],
    "ccr": ["{slist} each name the {adj} {noun}. How reliable they are depends on visibility, which nobody recorded: with chance "
            "{pf} it was foggy, and otherwise clear. In clear conditions each is right with chance {ac}; in fog each is right with "
            "chance {af} (when wrong, any other {noun} is equally likely). Given the conditions, their mistakes are independent.",
            "{slist} each name the {adj} {noun}. Their accuracy depends on the weather, which was not recorded: foggy with chance "
            "{pf}, clear otherwise. Each is right with chance {ac} when clear and {af} in fog, erring independently given the "
            "weather (a wrong answer names one of the other {nouns} at random).",
            "{slist} name the {adj} {noun}. With chance {pf} the day was foggy (it was not recorded); each is right with chance "
            "{af} in fog and {ac} in clear weather, and given the weather their errors are independent, a wrong answer naming "
            "one of the other {nouns} at random.",
            "{slist} each name the {adj} {noun}. The weather that day was not recorded: foggy with chance {pf}, otherwise clear. "
            "Each is right with chance {ac} in clear weather and {af} in fog (a wrong answer names one of the other {nouns} at "
            "random), and given the weather their errors are independent."],
    "ccr_fog": ["Their reports travel over a radio link that is also affected by the weather: each report is lost with chance "
                "{mf} if it was foggy and with chance {mc} if it was clear, independently given the weather and whatever the "
                "report said.",
                "The radio link also depends on the weather: a report is lost with chance {mf} in fog and {mc} in clear "
                "weather, independently given the weather, whatever it said.",
                "Reports can be lost on the way, more or less often depending on the weather: chance {mf} if foggy, {mc} if "
                "clear, independently given the weather and regardless of the claim.",
                "Fog also hurts the radio link: each report is lost with chance {mf} if it was foggy and {mc} if it was clear, "
                "independently once the weather is fixed, whatever the report named."],
    "ccr_mar": ["Some reports were lost to a random radio fault: each report went missing with chance {rate}, whatever the weather "
                "and whatever it said.",
                "A few reports were lost at random, each with chance {rate}, unrelated to the weather and to what they said.",
                "Any report could have been dropped by a random fault (chance {rate} each), regardless of the weather and of "
                "its content.",
                "Each report had a {rate} chance of being lost to a random glitch, unrelated to the weather and to the claim."],
    "ccr_obs": ["Reports received: {claims}.", "What arrived: {claims}.", "The reports: {claims}.",
                "Here is what came through: {claims}."],
    "rlr": ["{O_cap} checked which {noun} {isp} and is right with chance {acc} (when wrong, any other {noun} is equally likely), "
            "but {O}'s own account is not available. Each person below heard {O}'s claim directly and passes it on correctly "
            "with the stated chance, otherwise naming one of the other {nouns} at random.",
            "The only first-hand check was made by {O}, who is right with chance {acc} (otherwise naming one of the other "
            "{nouns} at random); {O}'s account itself is lost. Each person below heard it from {O} and relays it correctly with "
            "the stated chance, otherwise naming one of the other {nouns} at random.",
            "All reports below trace back to {O}, right with chance {acc} (when wrong, any other {noun} equally likely), whose "
            "own account is unavailable. A relay repeats what it heard correctly with its stated chance; otherwise it names one "
            "of the other {nouns} at random.",
            "Everything below comes from {O}, who saw which {noun} {isp} and is right with chance {acc} (otherwise naming one of "
            "the other {nouns} at random); {O}'s own account cannot be found. Each relay passes it on correctly with the chance "
            "given, and otherwise substitutes one of the other {nouns} at random."],
    "rlr_ind": ["Each person below checked independently which {noun} {isp} and is right with chance {acc} (when wrong, naming "
                "one of the other {nouns} at random); nobody copied anyone.",
                "The people below each checked for themselves, independently of each other; each is right with chance {acc} "
                "(otherwise naming one of the other {nouns} at random).",
                "Every person below made an independent check and is right with chance {acc}; a wrong answer names one of the "
                "other {nouns} at random.",
                "Each of the people below checked separately, being right with chance {acc} (a wrong answer names one of the "
                "other {nouns} at random), and none of them copied another."],
    "rlr_rep": ["- {R_cap} (passes the claim on correctly with chance {f}): {res}.",
                "- {R_cap}, correct relay chance {f}: {res}.",
                "- {R_cap} (relays correctly with chance {f}): {res}.",
                "- {R_cap}, who passes it on correctly with chance {f}: {res}."],
    "rlr_rep_ind": ["- {R_cap}: {res}.", "- {R_cap}: {res}.", "- From {R}: {res}.", "- {R_cap}: {res}."],
    "rlr_claim": ["Not every report reaches {agent}: a report naming {S} is passed on with chance {qa}, and a report naming any "
                  "other {noun} with chance {qb}, independently of the other reports.",
                  "Reports are forwarded selectively: one naming {S} gets through with chance {qa}, one naming any other {noun} "
                  "with chance {qb}, independently.",
                  "Whether a report arrives depends on what it says: chance {qa} if it names {S}, chance {qb} if it names "
                  "another {noun}, independently for each report.",
                  "A report naming {S} reaches {agent} with chance {qa}; a report naming any other {noun} with chance {qb}; "
                  "reports arrive or go missing independently."],
    "rlr_mar": ["Some reports went astray at random: each was lost with chance {rate}, whatever it said.",
                "A few reports were lost at random, each with chance {rate}, unrelated to what they said.",
                "Any report could have been dropped by a random fault (chance {rate} each), regardless of what it named.",
                "Each report had a {rate} chance of going missing, unrelated to its content."],
}
TPL12_KEYS = sorted(TPL12)
assert all(len(v) == 4 for v in TPL12.values())


# --------------------------------------------------------------------------- small helpers
def seed_of(*parts):
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def rng_of(*parts):
    return np.random.default_rng(seed_of(*parts))


def sha16(s):
    return hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:16]


def ri(rng, a, b):
    """Integer uniform on [a, b] inclusive."""
    return int(rng.integers(a, b + 1))


def pct(v):
    return np.asarray(v, dtype=np.float64) / 100.0


def grid_dist(w, total=100, lo=1):
    """Positive weights -> integers >= lo summing to `total` (largest remainder; deterministic)."""
    x = np.asarray(w, float)
    k = len(x)
    rem = total - lo * k
    assert rem >= 0
    x = x / x.sum() * rem
    fl = np.floor(x).astype(int)
    r = int(rem - fl.sum())
    order = np.argsort(-(x - fl), kind="stable")
    fl[order[:r]] += 1
    out = [int(v + lo) for v in fl]
    assert sum(out) == total
    return out


def draw_prior(rng, k):
    sigma = float(rng.choice([0.25, 0.6, 1.0, 1.5]))
    return grid_dist(np.exp(sigma * rng.standard_normal(k)))


def draw_k(rng, lo, hi):
    return ri(rng, lo, hi)


def money(rng, lo, hi, step):
    v = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    return int(max(step, round(v / step) * step))


def lab_dict(keys, vals):
    """Rounded to 6 dp, summing to exactly 1 (the largest entry absorbs the rounding), as build_bayes_pref.lab_dict."""
    v = [max(0.0, float(x)) for x in vals]
    s = sum(v)
    v = [round(x / s, 6) for x in v]
    j = max(range(len(v)), key=lambda i: v[i])
    v[j] = round(v[j] + (1.0 - sum(v)), 6)
    return dict(zip(keys, v))


def onehot_set(n, best):
    t = np.zeros(n)
    t[list(best)] = 1.0 / len(best)
    return t


def argmin_set(el, rel=1e-9):
    el = np.asarray(el, float)
    m = el.min()
    return [int(i) for i in np.flatnonzero(el <= m + rel * max(1.0, abs(m)))]


def acc_lik(k, acc, claim):
    """P(claim | h) for a source right with prob acc, otherwise uniform over the other k-1: vector over h."""
    a = acc / 100.0
    v = np.full(k, (1.0 - a) / (k - 1))
    v[claim] = a
    return v


def relay_mat(k, fid):
    """T[y, x] = P(passes on y | heard x)."""
    f = fid / 100.0
    T = np.full((k, k), (1.0 - f) / (k - 1))
    np.fill_diagonal(T, f)
    return T


def ring_d(N, i, j):
    d = abs(i - j)
    return min(d, N - d)


def read_lik(dmax, pe):
    """M[r, d] = P(reading r | true distance d): exact w.p. pe, else +-1 equally; an impossible value -> exact."""
    p = pe / 100.0
    M = np.zeros((dmax + 1, dmax + 1))
    for d in range(dmax + 1):
        M[d, d] += p
        for r in (d - 1, d + 1):
            if 0 <= r <= dmax:
                M[r, d] += (1 - p) / 2
            else:
                M[d, d] += (1 - p) / 2
    return M


def compositions(n, m):
    """All count vectors of length m summing to n (exchangeable readings)."""
    if m == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for rest in compositions(n - i, m - 1):
            yield (i,) + rest


def multinom_coef(c):
    return math.factorial(sum(c)) / math.prod(math.factorial(x) for x in c)


# --------------------------------------------------------------------------- world sampling (stated model, then truth, then evidence)
def base_world(family, structure, skin, k, names, prior):
    return {"family": family, "structure": structure, "skin": skin, "k": k, "names": names, "prior": prior,
            "parts": {}, "ev": [], "next": None, "loss": None, "info": None, "latq": None,
            "truth": {}, "from_model": True}


def pick_names(rng, pool, k):
    idx = sorted(int(i) for i in rng.choice(len(pool), size=k, replace=False))
    return [pool[i] for i in idx]


def sample_tgt_params(rng, k):
    return {"t": "tgt", "on": ri(rng, 0, k - 1), "sens": ri(rng, 55, 99), "fpr": ri(rng, 1, 40)}


def draw_truth_h(rng, prior):
    return int(rng.choice(len(prior), p=pct(prior) / pct(prior).sum()))


def obs_tgt(rng, it, h):
    p = it["sens"] if h == it["on"] else it["fpr"]
    return int(rng.random() < p / 100.0)


def next_tgt(rng, k, sk):
    it = sample_tgt_params(rng, k)
    return {"t": "tgt", "on": it["on"], "sens": it["sens"], "fpr": it["fpr"]}


def sim_next(rng, w, h, latent):
    """The realised next observation (for realised log scores), drawn from the stated model given the truth."""
    nx = w["next"]
    if nx is None:
        return None
    k = w["k"]
    t = nx["t"]
    if t == "tgt":
        return 0 if obs_tgt(rng, nx, h) else 1                     # outcome 0 = flags, 1 = does not flag
    if t == "sig":
        return int(rng.choice(len(w["parts"]["sig"]["cats"]), p=pct(w["parts"]["sig"]["table"][h])))
    if t == "feat":
        return 0 if rng.random() < nx["p"][h] / 100.0 else 1
    if t == "rep_new":
        return int(rng.choice(k, p=acc_lik(k, nx["acc"], h)))
    if t == "right_next":
        r = w["parts"]["rv"][nx["rv"]]["levels"][latent["rv"][nx["rv"]]]
        return 0 if rng.random() < r / 100.0 else 1
    if t == "line_check":
        failing = h == 0
        return 0 if rng.random() < (nx["sens"] if failing else nx["fpr"]) / 100.0 else 1
    if t == "dist":
        P = w["parts"]
        dmax = P["dmax"]
        d = P["dist"][nx["at"]][h]
        return int(rng.choice(dmax + 1, p=read_lik(dmax, P["pe"])[:, d]))
    if t == "grp":
        inside = P_in_group(w, nx["g"], h)
        return 0 if rng.random() < (nx["sens"] if inside else nx["fpr"]) / 100.0 else 1
    raise ValueError(t)


def P_in_group(w, g, h):
    return h in w["parts"]["groups"][g]


def sample_diagnosis(rng, structure, skin, family="diagnosis"):
    sk = SKINS[skin]
    if structure == "targeted":
        k = draw_k(rng, 2, 12)
    else:
        k = draw_k(rng, 2, 6)
    names = pick_names(rng, sk["names"], k)
    prior = draw_prior(rng, k)
    w = base_world(family, structure, skin, k, names, prior)
    if structure == "targeted":
        n = ri(rng, 1, 6)
        items = [sample_tgt_params(rng, k) for _ in range(n)]
        w["next"] = next_tgt(rng, k, sk)
    elif structure == "signature":
        m = ri(rng, 2, 4)
        table = [grid_dist(np.exp(1.0 * rng.standard_normal(m))) for _ in range(k)]
        w["parts"]["sig"] = {"table": table, "cats": sk["cats"][:m]}
        n = ri(rng, 1, 4)
        items = [{"t": "sig"} for _ in range(n)]
        w["next"] = {"t": "sig"}
    elif structure == "features":
        nf = ri(rng, 2, 4)
        fidx = sorted(int(i) for i in rng.choice(len(sk["features"]), size=min(nf + 1, len(sk["features"])), replace=False))
        nf = min(nf, len(fidx) - 1)
        items = [{"t": "feat", "f": fidx[i], "p": [ri(rng, 5, 95) for _ in range(k)]} for i in range(nf)]
        w["next"] = {"t": "feat", "f": fidx[nf], "p": [ri(rng, 5, 95) for _ in range(k)]}
    else:
        raise ValueError(structure)
    h = draw_truth_h(rng, prior)
    for it in items:
        if it["t"] == "tgt":
            it["y"] = obs_tgt(rng, it, h)
        elif it["t"] == "sig":
            it["y"] = int(rng.choice(len(w["parts"]["sig"]["cats"]), p=pct(w["parts"]["sig"]["table"][h])))
        else:
            it["y"] = int(rng.random() < it["p"][h] / 100.0)
    w["ev"] = items
    w["truth"] = {"h": h, "latent": {}}
    w["truth"]["next"] = sim_next(rng, w, h, {})
    return w


def sample_copied(rng, structure, skin):
    sk = SKINS[skin]
    k = draw_k(rng, 3, 8)
    names = pick_names(rng, sk["names"], k)
    prior = draw_prior(rng, k)
    w = base_world("copied", structure, skin, k, names, prior)
    pool = [int(i) for i in rng.permutation(len(sk["reporters"]))]
    take = lambda: pool.pop(0)                                       # noqa: E731
    blocks = []
    if structure == "shared_upstream":
        m = ri(rng, 2, 3)
        blocks.append({"t": "relay", "origin": take(), "acc": ri(rng, 55, 95),
                       "reps": [{"src": take(), "path": [100]} for _ in range(m)], "stmt": "copied"})
    elif structure == "noisy_copy":
        blocks.append({"t": "copy", "A": take(), "B": take(), "a": ri(rng, 55, 95), "b": ri(rng, 55, 95),
                       "c": ri(rng, 20, 90), "stmt": "copied"})
    elif structure == "chain_copy":
        o, r1, mm, r2 = take(), take(), take(), take()
        reps = [{"src": r1, "path": [ri(rng, 60, 99)]}, {"src": r2, "via": mm, "path": [ri(rng, 60, 99), ri(rng, 60, 99)]}]
        if rng.random() < 0.4:
            reps.append({"src": take(), "path": [ri(rng, 60, 99)]})
        blocks.append({"t": "relay", "origin": o, "acc": ri(rng, 55, 95), "reps": reps, "stmt": "chain"})
    elif structure == "common_cause":
        ns = ri(rng, 2, 3)
        blocks.append({"t": "cc", "sens": list(range(ns)), "pf": ri(rng, 10, 60), "ac": ri(rng, 75, 98),
                       "af": ri(rng, 35, 65), "stmt": "copied"})
    n_ind = {"independent": ri(rng, 2, 4), "shared_upstream": ri(rng, 0, 1), "noisy_copy": ri(rng, 0, 1),
             "chain_copy": ri(rng, 0, 1), "common_cause": ri(rng, 0, 1)}[structure]
    for _ in range(n_ind):
        blocks.append({"t": "rep", "src": take(), "acc": ri(rng, 45, 90)})
    w["next"] = {"t": "rep_new", "acc": ri(rng, 55, 95)}
    h = draw_truth_h(rng, prior)
    lat = {}
    for bi, b in enumerate(blocks):                                   # forward simulation of the stated process
        if b["t"] == "rep":
            b["c"] = int(rng.choice(k, p=acc_lik(k, b["acc"], h)))
        elif b["t"] == "relay":
            x = int(rng.choice(k, p=acc_lik(k, b["acc"], h)))            # the origin's own claim (never shown)
            lat[f"x:{bi}"] = x
            for rp in b["reps"]:
                y = x
                for f in rp["path"]:
                    y = int(rng.choice(k, p=relay_mat(k, f)[:, y]))
                rp["c"] = y
        elif b["t"] == "copy":
            b["ca"] = int(rng.choice(k, p=acc_lik(k, b["a"], h)))
            copied = bool(rng.random() < b["c"] / 100.0)
            lat[f"c:{bi}"] = int(copied)
            b["cb"] = b["ca"] if copied else int(rng.choice(k, p=acc_lik(k, b["b"], h)))
        elif b["t"] == "cc":
            fog = bool(rng.random() < b["pf"] / 100.0)
            lat[f"z:{bi}"] = int(fog)
            a = b["af"] if fog else b["ac"]
            b["cs"] = [int(rng.choice(k, p=acc_lik(k, a, h))) for _ in b["sens"]]
    w["ev"] = blocks
    w["truth"] = {"h": h, "latent": lat}
    w["truth"]["next"] = sim_next(rng, w, h, lat)
    return w


def draw_levels(rng):
    L = ri(rng, 2, 4)
    lv = sorted(int(x) for x in rng.choice(ACC_MENU, size=L, replace=False))
    return lv, draw_prior(rng, L)


def sample_reliability(rng, structure, skin):
    sk = SKINS[skin]
    k = draw_k(rng, 2, 6)
    names = pick_names(rng, sk["names"], k)
    prior = draw_prior(rng, k)
    w = base_world("reliability", structure, skin, k, names, prior)
    pool = [int(i) for i in rng.permutation(len(sk["reporters"]))]
    rv, items, srcs = {}, [], {}
    if structure == "track_record":
        lv, lp = draw_levels(rng)
        rv["0"] = {"levels": lv, "prior": lp}
        s0 = pool.pop(0)
        srcs[s0] = "0"
        n = 0 if rng.random() < 0.15 else ri(rng, 1, 20)
        items.append({"t": "track", "src": s0, "rv": "0", "n": n})
        items.append({"t": "lrep", "src": s0, "rv": "0"})
        if rng.random() < 0.5:
            items.append({"t": "rep", "src": pool.pop(0), "acc": ri(rng, 50, 95)})
    elif structure == "latent_sources":
        ns = ri(rng, 2, 3)
        shared_menu = rng.random() < 0.5
        lv0, lp0 = draw_levels(rng)
        for j in range(ns):
            lv, lp = (lv0, lp0) if shared_menu else draw_levels(rng)
            rv[str(j)] = {"levels": list(lv), "prior": list(lp)}
            s = pool.pop(0)
            srcs[s] = str(j)
            if rng.random() < 0.5:
                items.append({"t": "track", "src": s, "rv": str(j), "n": ri(rng, 1, 12)})
            items.append({"t": "lrep", "src": s, "rv": str(j)})
    elif structure == "shared_reliability":
        lv, lp = draw_levels(rng)
        rv["0"] = {"levels": lv, "prior": lp}
        ns = ri(rng, 2, 3)
        members = [pool.pop(0) for _ in range(ns)]
        n = 0 if rng.random() < 0.2 else ri(rng, 1, 15)
        items.append({"t": "track", "src": None, "rv": "0", "n": n, "agency": True})
        for s in members:
            srcs[s] = "0"
            items.append({"t": "lrep", "src": s, "rv": "0"})
        w["parts"]["agency"] = members
    w["parts"]["rv"] = rv
    w["parts"]["srcs"] = {str(s): r for s, r in srcs.items()}
    ask_rv = sorted(rv)[ri(rng, 0, len(rv) - 1)]
    ask_src = [int(s) for s, r in w["parts"]["srcs"].items() if r == ask_rv][0]
    w["latq"] = {"axis": f"rv:{ask_rv}", "src": ask_src}
    w["next"] = {"t": "right_next", "rv": ask_rv, "src": ask_src}
    h = draw_truth_h(rng, prior)
    lat = {"rv": {}}
    for r in sorted(rv):
        lat["rv"][r] = int(rng.choice(len(rv[r]["levels"]), p=pct(rv[r]["prior"])))
    for it in items:
        if it["t"] == "track":
            acc = rv[it["rv"]]["levels"][lat["rv"][it["rv"]]] / 100.0
            it["s"] = int(rng.binomial(it["n"], acc)) if it["n"] else 0
        elif it["t"] == "lrep":
            acc = rv[it["rv"]]["levels"][lat["rv"][it["rv"]]]
            it["c"] = int(rng.choice(k, p=acc_lik(k, acc, h)))
        elif it["t"] == "rep":
            it["c"] = int(rng.choice(k, p=acc_lik(k, it["acc"], h)))
    w["ev"] = items
    w["truth"] = {"h": h, "latent": lat}
    w["truth"]["next"] = sim_next(rng, w, h, lat)
    return w


def sample_missingness(rng, structure, skin):
    sk = SKINS[skin]
    k = draw_k(rng, 3, 8)
    names = pick_names(rng, sk["names"], k)
    prior = draw_prior(rng, k)
    w = base_world("missingness", structure, skin, k, names, prior)
    if structure == "mnar_hypothesis":
        nl = ri(rng, 2, k)
        on = sorted(int(i) for i in rng.choice(k, size=nl, replace=False))
        miss = {"mech": "mnar", "m1": ri(rng, 30, 80), "m0": ri(rng, 2, 25), "sens": ri(rng, 55, 99), "fpr": ri(rng, 1, 40)}
        items = [{"t": "log", "on": j} for j in on]
    else:
        n = ri(rng, 3, 7)
        items = [sample_tgt_params(rng, k) for _ in range(n)]
        if structure == "mar":
            miss = {"mech": "mar", "rate": ri(rng, 15, 50)}
        else:
            hi_, lo_ = ri(rng, 60, 100), ri(rng, 5, 50)
            miss = {"mech": "informative", "qf": hi_, "qc": lo_} if rng.random() < 0.7 else \
                {"mech": "informative", "qf": lo_, "qc": hi_}
    w["parts"]["miss"] = miss
    w["next"] = next_tgt(rng, k, sk)
    h = draw_truth_h(rng, prior)
    for it in items:
        if it["t"] == "log":
            m = miss["m1"] if h == it["on"] else miss["m0"]
            missing = bool(rng.random() < m / 100.0)
            yv = int(rng.random() < (miss["sens"] if h == it["on"] else miss["fpr"]) / 100.0)
        else:
            yv = obs_tgt(rng, it, h)
            if miss["mech"] == "mar":
                missing = bool(rng.random() < miss["rate"] / 100.0)
            else:
                q = miss["qf"] if yv else miss["qc"]
                missing = not bool(rng.random() < q / 100.0)
        it["y"] = None if missing else yv
        it["_y"] = yv                                                # hidden (moved to truth below)
    w["truth"] = {"h": h, "latent": {"hidden_results": [it.pop("_y") for it in items]}}
    w["ev"] = items
    w["truth"]["next"] = sim_next(rng, w, h, {})
    return w


# ---- v1.2 missingness2 (GUIDANCE.md contract 3). A test result's recording process, given h and its result y:
#   MAR          P(recorded) = 1 - rate                                  (a missing result carries a constant)
#   hyp          P(recorded | h) = 1 - m(h), m(h) = m1 if h is the tested unit else m0   (hypothesis-dependent absence)
#   result       P(recorded | y) = q_flag or q_noflag                      (result-dependent recording)
# so a PRESENT result contributes P(y|h) P(recorded | h, y) and a MISSING one sum_y P(y|h) (1 - P(recorded | h, y)).
# Rates come from a JOINT support (one high and one low rate, either orientation; never independent marginals): M2_RATE_SUPPORT.
M2_P_NONE_MISSING = 0.4      # design: the share of worlds whose observed pattern has NO missing result (the stratum where only
M2_MAX_TRIES = 5000          # presence can be informative); the forward draw repeats until its observed missing count falls in
#                              the drawn stratum: selection on VISIBLE data only, so the exact posterior stays calibrated
M2_RATE_SUPPORT = {
    "mar": "rate in 15-50 (the stated loss chance; it carries no information either way)",
    "hyp": "(m_tested_unit_is_the_one, m_other): one in 35-85 and the other in 2-25; the high one is m_tested_unit_is_the_one with "
           "chance 0.6",
    "result": "(q_flag, q_noflag): one in 60-100 and the other in 5-50; the high one is q_flag with chance 0.6",
    "regime": "two candidate rules of opposite orientation: (high, low) as drawn above and a fresh (low, high), prior on the 1% grid",
    "signature_result": "one recording chance per reading value: at least one value in 60-100 and at least one in 5-50",
    "signature_hyp": "one named hypothesis: m1 for it and m0 otherwise, (m1, m0) as for hyp",
    "composed_fog": "(loss in fog, loss in clear): one in 35-80 and the other in 2-25, the high one in fog with chance 0.6",
    "composed_claim": "(pass-on chance for reports naming the named unit, for any other): one in 60-100, the other in 5-50",
}


def draw_m2_rates(rng, mech):
    """Stated recording rates for the tests structure, from the JOINT support (M2_RATE_SUPPORT)."""
    if mech == "mar":
        return {"mech": "mar", "rate": ri(rng, 15, 50)}
    if mech == "hyp":
        hi, lo = ri(rng, 35, 85), ri(rng, 2, 25)
        return {"mech": "hyp", "m1": hi, "m0": lo} if rng.random() < 0.6 else {"mech": "hyp", "m1": lo, "m0": hi}
    hi, lo = ri(rng, 60, 100), ri(rng, 5, 50)
    return {"mech": "result", "qf": hi, "qc": lo} if rng.random() < 0.6 else {"mech": "result", "qf": lo, "qc": hi}


def draw_m2_regime(rng, mech):
    """A LATENT recording rule: two joint rate pairs of opposite orientation, with a stated prior (enumerated jointly with h)."""
    a = draw_m2_rates(rng, mech)
    keys = ("m1", "m0") if mech == "hyp" else ("qf", "qc")
    hi_r, lo_r = ((35, 85), (2, 25)) if mech == "hyp" else ((60, 100), (5, 50))
    if a[keys[0]] > a[keys[1]]:
        b = {keys[0]: ri(rng, *lo_r), keys[1]: ri(rng, *hi_r)}
    else:
        b = {keys[0]: ri(rng, *hi_r), keys[1]: ri(rng, *lo_r)}
    return {"mech": mech, "regimes": [{kk: a[kk] for kk in keys}, b], "rprior": draw_prior(rng, 2)}


def draw_m2_sig_rates(rng, mech, k, m):
    """Recording rates for the held-out signature structure (categorical readings)."""
    if mech == "mar":
        return {"mech": "mar", "rate": ri(rng, 15, 50)}
    if mech == "hyp":
        hi, lo = ri(rng, 35, 85), ri(rng, 2, 25)
        on = ri(rng, 0, k - 1)
        return {"mech": "hyp", "on": on, "m1": hi, "m0": lo} if rng.random() < 0.6 else \
            {"mech": "hyp", "on": on, "m1": lo, "m0": hi}
    nh = ri(rng, 1, m - 1)
    hi_set = {int(i) for i in rng.choice(m, size=nh, replace=False)}
    return {"mech": "result", "q": [ri(rng, 60, 100) if c in hi_set else ri(rng, 5, 50) for c in range(m)]}


def m2_lost_prob(w, it, rates, h, y):
    """P(the result is NOT recorded | h, y) under `rates` (one rate set; a regime world passes one rule)."""
    mech = rates["mech"]
    if mech == "mar":
        return rates["rate"] / 100.0
    if mech == "hyp":
        on = it["on"] if it["kind"] == "tgt" else rates["on"]
        return (rates["m1"] if h == on else rates["m0"]) / 100.0
    if it["kind"] == "tgt":
        return 1.0 - (rates["qf"] if y else rates["qc"]) / 100.0
    return 1.0 - rates["q"][y] / 100.0


def m2_draw_y(rng, w, it, h):
    if it["kind"] == "tgt":
        return obs_tgt(rng, it, h)
    return int(rng.choice(len(w["parts"]["sig"]["cats"]), p=pct(w["parts"]["sig"]["table"][h])))


def sample_missingness2(rng, structure, skin):
    sk = SKINS[skin]
    k = draw_k(rng, 3, 8)
    names = pick_names(rng, sk["names"], k)
    prior = draw_prior(rng, k)
    w = base_world("missingness2", structure, skin, k, names, prior)
    none_missing = bool(rng.random() < M2_P_NONE_MISSING)
    if structure == "signature":
        m = ri(rng, 3, 4)
        w["parts"]["sig"] = {"table": [grid_dist(np.exp(1.0 * rng.standard_normal(m))) for _ in range(k)], "cats": sk["cats"][:m]}
        n = ri(rng, 2, 4) if none_missing else ri(rng, 3, 6)
        items = [{"t": "m2", "kind": "sig"} for _ in range(n)]
        miss2 = draw_m2_sig_rates(rng, ("mar", "hyp", "result")[ri(rng, 0, 2)], k, m)
    else:
        n = ri(rng, 2, 5) if none_missing else ri(rng, 3, 7)
        items = []
        for _ in range(n):
            it = sample_tgt_params(rng, k)
            items.append({"t": "m2", "kind": "tgt", "on": it["on"], "sens": it["sens"], "fpr": it["fpr"]})
        if structure == "tests":
            miss2 = draw_m2_rates(rng, ("mar", "hyp", "result")[ri(rng, 0, 2)])
        else:                                                          # regime: the recording rule itself is latent
            miss2 = draw_m2_regime(rng, ("hyp", "result")[ri(rng, 0, 1)])
            w["latq"] = {"axis": "reg"}
    w["parts"]["miss2"] = miss2
    w["next"] = next_tgt(rng, k, sk)
    for _ in range(M2_MAX_TRIES):                                      # forward draws from the stated model; keep the first
        h = draw_truth_h(rng, prior)                                   # whose OBSERVED missing count is in the drawn stratum
        reg = int(rng.choice(2, p=pct(miss2["rprior"]))) if "regimes" in miss2 else None
        rates = dict(miss2["regimes"][reg], mech=miss2["mech"]) if reg is not None else miss2
        ys, lost = [], []
        for it in items:
            y = m2_draw_y(rng, w, it, h)
            ys.append(y)
            lost.append(bool(rng.random() < m2_lost_prob(w, it, rates, h, y)))
        if (sum(lost) == 0) == none_missing:
            break
    for it, y, ls in zip(items, ys, lost):
        it["y"] = None if ls else y
    lat = {"hidden_results": ys}
    if reg is not None:
        lat.update(reg=reg, s=reg)              # 's' mirrors the asked latent for analysis_bayes_worlds_eval.truth_index
    w["ev"] = items
    w["truth"] = {"h": h, "latent": lat, "stratum_target": "none_missing" if none_missing else "some_missing",
                  "stratum_met": (sum(lost) == 0) == none_missing}
    w["truth"]["next"] = sim_next(rng, w, h, lat)
    return w


def ret_keep(it, k, y):
    """Composed worlds: P(a report naming y / a sensor's report is passed on), for the retention rule `it['ret']` (z-free part)."""
    r = it["ret"]
    if r["mech"] == "mar":
        return 1.0 - r["rate"] / 100.0
    if r["mech"] == "claim":
        return (r["qa"] if y == r["S"] else r["qb"]) / 100.0
    raise ValueError(r["mech"])


def sample_composed(rng, structure, skin):
    """CONFIRM-ONLY compositions: correlated sources whose reports are selectively retained."""
    sk = SKINS[skin]
    k = draw_k(rng, 3, 6)
    names = pick_names(rng, sk["names"], k)
    prior = draw_prior(rng, k)
    w = base_world("composed", structure, skin, k, names, prior)
    none_missing = bool(rng.random() < 0.3)
    pool = [int(i) for i in rng.permutation(len(sk["reporters"]))]
    if structure == "cc_retained":
        ns = ri(rng, 3, 4)
        hi, lo = ri(rng, 35, 80), ri(rng, 2, 25)
        ret = {"mech": "fog", "mf": hi, "mc": lo} if rng.random() < 0.6 else {"mech": "fog", "mf": lo, "mc": hi}
        b = {"t": "ccr", "sens": list(range(ns)), "pf": ri(rng, 10, 60), "ac": ri(rng, 75, 98), "af": ri(rng, 35, 65),
             "stmt": "copied", "ret": ret}
    else:
        m = ri(rng, 2, 4)
        hi, lo = ri(rng, 60, 100), ri(rng, 5, 50)
        S = ri(rng, 0, k - 1)
        ret = {"mech": "claim", "S": S, "qa": hi, "qb": lo} if rng.random() < 0.5 else {"mech": "claim", "S": S, "qa": lo, "qb": hi}
        b = {"t": "rlr", "origin": pool.pop(0), "acc": ri(rng, 55, 95), "stmt": "copied", "ret": ret,
             "reps": [{"src": pool.pop(0), "path": [ri(rng, 60, 99)]} for _ in range(m)]}
    w["next"] = {"t": "rep_new", "acc": ri(rng, 55, 95)}
    for _ in range(M2_MAX_TRIES):
        h = draw_truth_h(rng, prior)
        lat = {}
        if structure == "cc_retained":
            fog = bool(rng.random() < b["pf"] / 100.0)
            lat["z:0"] = int(fog)
            a = b["af"] if fog else b["ac"]
            lost_p = (ret["mf"] if fog else ret["mc"]) / 100.0
            cs = [int(rng.choice(k, p=acc_lik(k, a, h))) for _ in b["sens"]]
            lost = [bool(rng.random() < lost_p) for _ in cs]
            vals = cs
        else:
            x = int(rng.choice(k, p=acc_lik(k, b["acc"], h)))
            lat["x:0"] = x
            vals = [int(rng.choice(k, p=relay_mat(k, rp["path"][0])[:, x])) for rp in b["reps"]]
            lost = [bool(rng.random() >= ret_keep(b, k, y)) for y in vals]
        if (sum(lost) == 0) == none_missing:
            break
    if structure == "cc_retained":
        b["cs"] = [None if ls else c for c, ls in zip(vals, lost)]
    else:
        for rp, y, ls in zip(b["reps"], vals, lost):
            rp["c"] = None if ls else y
    lat["hidden_reports"] = vals
    w["ev"] = [b]
    w["truth"] = {"h": h, "latent": lat, "stratum_target": "none_missing" if none_missing else "some_missing",
                  "stratum_met": (sum(lost) == 0) == none_missing}
    w["truth"]["next"] = sim_next(rng, w, h, lat)
    return w


def draw_loss(rng, k, form, unit, abstain):
    """Loss spec (integers). fix: L(fix_j, h) = F_j + M_j [h != j]; avoid: L(rely_j, h) = B_j + P_j [h == j];
    matrix: explicit L(a_j, h); abstain: a flat E."""
    if unit == "points":
        S, st = money(rng, 20, 400, 1), 1
    else:
        S, st = money(rng, 1000, 40000, 50), 50
    rnd = lambda v: int(max(st, round(v / st) * st))               # noqa: E731
    L = {"form": form}
    if form == "fix":
        L["F"] = [rnd(S * rng.uniform(0.0, 0.25)) if rng.random() < 0.8 else 0 for _ in range(k)]
        L["M"] = [rnd(S * math.exp(rng.uniform(math.log(0.3), math.log(3.0)))) for _ in range(k)]
    elif form == "avoid":
        L["B"] = [rnd(S * rng.uniform(0.02, 0.3)) for _ in range(k)]
        L["P"] = [rnd(S * math.exp(rng.uniform(math.log(0.5), math.log(3.0)))) for _ in range(k)]
    elif form == "matrix":
        M = []
        for a in range(k):
            row = []
            for hh in range(k):
                base = rng.uniform(0.0, 0.3) if a == hh else math.exp(rng.uniform(math.log(0.2), math.log(3.0)))
                row.append(rnd(S * base) if not (a == hh and rng.random() < 0.3) else 0)
            M.append(row)
        L["L"] = M
    L["E"] = rnd(S * rng.uniform(0.15, 0.9)) if abstain else None
    L["scale"] = S
    return L


def loss_matrix(w, scale_losses=1.0):
    """(nA, k) float matrix; rows = the k actions in canonical order, then abstain if present."""
    L, k = w["loss"], w["k"]
    rows = []
    for a in range(k):
        if L["form"] == "fix":
            rows.append([L["F"][a] + (L["M"][a] if hh != a else 0) for hh in range(k)])
        elif L["form"] == "avoid":
            rows.append([L["B"][a] + (L["P"][a] if hh == a else 0) for hh in range(k)])
        else:
            rows.append(list(L["L"][a]))
    if L["E"] is not None:
        rows.append([L["E"]] * k)
    return np.asarray(rows, dtype=np.float64) * scale_losses


def loss_scale(w):
    return float(loss_matrix(w).max())


def sample_decision(rng, structure, skin):
    base = "targeted" if rng.random() < 0.6 else "features"
    w = sample_decision_base(rng, structure, skin, base)
    return w


def sample_decision_base(rng, structure, skin, base):
    sk = SKINS[skin]
    if structure == "asymmetric":
        k = draw_k(rng, 2, 3)
    elif structure == "matrix":
        k = draw_k(rng, 2, 3)
    else:
        k = draw_k(rng, 3, 6)
    names = pick_names(rng, sk["names"], k)
    prior = draw_prior(rng, k)
    w = base_world("decision", structure, skin, k, names, prior)
    if base == "targeted":
        items = [sample_tgt_params(rng, k) for _ in range(ri(rng, 1, 4))]
    else:
        nf = ri(rng, 1, 3)
        fidx = sorted(int(i) for i in rng.choice(len(sk["features"]), size=nf, replace=False))
        items = [{"t": "feat", "f": fi, "p": [ri(rng, 5, 95) for _ in range(k)]} for fi in fidx]
    form = {"asymmetric": "fix", "abstain": "fix", "matrix": "matrix", "avoid": "avoid"}[structure]
    abst = {"asymmetric": False, "abstain": True, "matrix": rng.random() < 0.3, "avoid": rng.random() < 0.5}[structure]
    w["loss"] = draw_loss(rng, k, form, sk["unit"], abst)
    if structure == "asymmetric":                                      # strongly unequal miss penalties
        M = w["loss"]["M"]
        j = ri(rng, 0, k - 1)
        M[j] = int(M[j] * ri(rng, 3, 12))
    w["next"] = next_tgt(rng, k, sk)
    h = draw_truth_h(rng, prior)
    for it in items:
        it["y"] = obs_tgt(rng, it, h) if it["t"] == "tgt" else int(rng.random() < it["p"][h] / 100.0)
    w["ev"] = items
    w["truth"] = {"h": h, "latent": {}}
    w["truth"]["next"] = sim_next(rng, w, h, {})
    return w


def sample_acquisition(rng, structure, skin):
    sk = SKINS[skin]
    k = draw_k(rng, 2, 5)
    names = pick_names(rng, sk["names"], k)
    prior = draw_prior(rng, k)
    w = base_world("acquisition", structure, skin, k, names, prior)
    items = [sample_tgt_params(rng, k) for _ in range(ri(rng, 0, 3))]
    w["loss"] = draw_loss(rng, k, "fix", sk["unit"], rng.random() < 0.5)
    S, st = w["loss"]["scale"], (1 if sk["unit"] == "points" else 10)
    fee = lambda: money(rng, max(st, 0.005 * S), max(2 * st, 0.3 * S), st)   # noqa: E731
    info = []
    if structure in ("one_test", "test_or_consult"):
        t = sample_tgt_params(rng, k)
        t.update({"t": "test", "fee": fee(), "sens": ri(rng, 60, 99), "fpr": ri(rng, 1, 35)})
        info.append(t)
        if structure == "test_or_consult":
            info.append({"t": "consult", "acc": ri(rng, 55, 95), "fee": fee()})
    elif structure == "collect_more":
        m = ri(rng, 2, 3)
        table = [grid_dist(np.exp(1.2 * rng.standard_normal(m))) for _ in range(k)]
        f = fee()
        for n in (1, 2, 3):
            info.append({"t": "collect", "n": n, "fee": f, "table": table, "cats": sk["ccats"][:m]})
    elif structure == "two_step":
        on = [int(i) for i in rng.choice(k, size=2, replace=False)]
        tests = [{"on": on[j], "sens": ri(rng, 60, 99), "fpr": ri(rng, 1, 35), "fee": fee()} for j in range(2)]
        info.append({"t": "two", "first": 0, "tests": tests})
        info.append({"t": "two", "first": 1, "tests": tests})
    w["info"] = info
    nx = info[0]
    if nx["t"] == "test":
        w["next"] = {"t": "tgt", "on": nx["on"], "sens": nx["sens"], "fpr": nx["fpr"]}
    elif nx["t"] == "two":
        t0 = nx["tests"][0]
        w["next"] = {"t": "tgt", "on": t0["on"], "sens": t0["sens"], "fpr": t0["fpr"]}
    else:
        w["next"] = None
    h = draw_truth_h(rng, prior)
    for it in items:
        it["y"] = obs_tgt(rng, it, h)
    w["ev"] = items
    w["truth"] = {"h": h, "latent": {}}
    w["truth"]["next"] = sim_next(rng, w, h, {})
    return w


def sample_hierarchical(rng, structure, skin):
    hs = HIER_SKINS[skin]
    G = ri(rng, 3, 5)
    L = 2 if structure == "two_factor" else ri(rng, 2, 3)
    grades = hs["grades2"] if L == 2 else hs["grades3"]
    target = ri(rng, 0, G - 1)
    w = base_world("hierarchical", structure, skin, 2, [hs["bad"], hs["good"]], None)
    P = w["parts"]
    P.update({"G": G, "L": L, "grades": grades, "sprior": draw_prior(rng, L), "target": target})
    if structure == "member":                                        # grade-separated failure rates (strong pooling)
        P["fail"] = [ri(rng, 2, 25), ri(rng, 35, 90)] if L == 2 else [ri(rng, 2, 20), ri(rng, 21, 50), ri(rng, 51, 90)]
    else:
        crew = [int(rng.integers(0, 2)) for _ in range(G)]
        P["crew"] = crew
        P["cprior"] = ri(rng, 15, 60)                                  # chance a crew is in its second (worse) state
        tab = [[ri(rng, 2, 25), ri(rng, 15, 50)], [ri(rng, 30, 60), ri(rng, 50, 95)]]
        P["fail2"] = tab                                               # [grade][crew state]
    checks = []
    for g in range(G):
        nc = (0 if rng.random() < 0.85 else 1) if g == target else (1 if rng.random() < 0.6 else 2)
        for _ in range(nc):
            checks.append({"t": "line", "g": g, "sens": ri(rng, 60, 98), "fpr": ri(rng, 2, 30)})
    w["next"] = {"t": "line_check", "g": target, "sens": ri(rng, 60, 98), "fpr": ri(rng, 2, 30)}
    w["latq"] = {"axis": "s"}
    s = int(rng.choice(L, p=pct(P["sprior"])))
    lat = {"s": s}
    if structure == "member":
        status = [int(rng.random() < P["fail"][s] / 100.0) for _ in range(G)]
    else:
        cs = [int(rng.random() < P["cprior"] / 100.0) for _ in range(2)]
        lat["crew_state"] = cs
        status = [int(rng.random() < P["fail2"][s][cs[P["crew"][g]]] / 100.0) for g in range(G)]
    lat["status"] = status
    for c in checks:
        c["y"] = int(rng.random() < (c["sens"] if status[c["g"]] else c["fpr"]) / 100.0)
    h = 0 if status[target] else 1                                     # names[0] = failing, names[1] = fine
    w["ev"] = checks
    w["truth"] = {"h": h, "latent": lat}
    w["truth"]["next"] = sim_next(rng, w, h, lat)
    return w


def sample_multimodal(rng, structure, skin):
    ms = MM_SKINS[skin]
    if structure == "ring_distance":
        N = ri(rng, 8, 12)
        names = [f"{ms['pos']} {i + 1}" for i in range(N)]
        P = {"N": N, "dmax": N // 2, "pe": ri(rng, 50, 95)}
        P["dist"] = [[ring_d(N, a, b) for b in range(N)] for a in range(N)]
        k = N
    elif structure == "grid_distance":
        R, C = (3, 3) if rng.random() < 0.5 else (3, 4)
        names = [f"{ms['gpos']} {'ABC'[r]}{c + 1}" for r in range(R) for c in range(C)]
        k = R * C
        P = {"R": R, "C": C, "pe": ri(rng, 50, 95)}
        cells = [(r, c) for r in range(R) for c in range(C)]
        P["dist"] = [[abs(a[0] - b[0]) + abs(a[1] - b[1]) for b in cells] for a in cells]
        P["dmax"] = R - 1 + C - 1
    else:
        k = ri(rng, 4, 8)
        names = [f"{'Hut' if skin == 'huts' else 'Room'} {i + 1}" for i in range(k)]
        sizes, left = [], k
        while left > 0:
            s = left if left <= 3 else ri(rng, 2, 3)
            if left - s == 1:
                s = 2
            sizes.append(s)
            left -= s
        perm = [int(i) for i in rng.permutation(k)]
        groups, at = [], 0
        for s in sizes:
            groups.append(sorted(perm[at:at + s]))
            at += s
        P = {"groups": groups, "gnames": ["north", "south", "east", "west", "central"][:len(groups)]}
    prior = draw_prior(rng, k)
    w = base_world("multimodal", structure, skin, k, names, prior)
    w["parts"] = P
    if structure in ("ring_distance", "grid_distance"):
        nr = int(rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15]))
        items = [{"t": "dist", "at": ri(rng, 0, k - 1)} for _ in range(nr)]
        w["next"] = {"t": "dist", "at": ri(rng, 0, k - 1)}
    else:
        items = [{"t": "grp", "g": ri(rng, 0, len(P["groups"]) - 1), "sens": ri(rng, 60, 98), "fpr": ri(rng, 2, 35)}
                 for _ in range(ri(rng, 1, 4))]
        w["next"] = {"t": "grp", "g": ri(rng, 0, len(P["groups"]) - 1), "sens": ri(rng, 60, 98), "fpr": ri(rng, 2, 35)}
    h = draw_truth_h(rng, prior)
    for it in items:
        if it["t"] == "dist":
            d = P["dist"][it["at"]][h]
            it["y"] = int(rng.choice(P["dmax"] + 1, p=read_lik(P["dmax"], P["pe"])[:, d]))
        else:
            inside = h in P["groups"][it["g"]]
            it["y"] = int(rng.random() < (it["sens"] if inside else it["fpr"]) / 100.0)
    w["ev"] = items
    w["truth"] = {"h": h, "latent": {}}
    w["truth"]["next"] = sim_next(rng, w, h, {})
    return w


def sample_world(family, structure, rng, novel=False):
    if family == "hierarchical":
        skin = sorted(HIER_SKINS)[ri(rng, 0, len(HIER_SKINS) - 1)]
        return sample_hierarchical(rng, structure, skin)
    if family == "multimodal":
        skin = sorted(MM_SKINS)[ri(rng, 0, len(MM_SKINS) - 1)]
        return sample_multimodal(rng, structure, skin)
    base = NOVEL_SKIN_ORDER if novel else SKIN_ORDER
    pool = base if not (family == "decision" and structure == "avoid") else [s for s in base if SKINS[s]["rely"]]
    skin = pool[ri(rng, 0, len(pool) - 1)]
    fn = {"diagnosis": sample_diagnosis, "copied": sample_copied, "reliability": sample_reliability,
          "missingness": sample_missingness, "decision": sample_decision, "acquisition": sample_acquisition}[family]
    return fn(rng, structure, skin)


def visible(w):
    """The world as the renderer and the solver may see it: no truth."""
    v = copy.deepcopy(w)
    v.pop("truth", None)
    v.pop("from_model", None)
    return v


# --------------------------------------------------------------------------- the exact solver (enumeration over a latent grid)
MODES = ("exact", "prior_only", "flat_prior", "ignore_dependence", "ignore_reliability", "ignore_missingness", "no_pooling")


def bc(axes, names, arr):
    """Broadcast `arr` (dims in the order of `names`) to the joint table's axes."""
    order = [a for a, _ in axes]
    arr = np.asarray(arr, dtype=np.float64)
    perm = sorted(range(len(names)), key=lambda i: order.index(names[i]))
    arr = np.transpose(arr, perm)
    shape = [1] * len(axes)
    for nm in names:
        shape[order.index(nm)] = axes[order.index(nm)][1]
    return arr.reshape(shape)


def rv_mean(rvd):
    return float(np.dot(pct(rvd["levels"]), pct(rvd["prior"])))


def tgt_lik(k, on, sens, fpr, y):
    p = np.where(np.arange(k) == on, sens / 100.0, fpr / 100.0)
    return p if y == 1 else 1.0 - p


M2_MODES = ("ignore_presence", "missing_only", "present_only")     # v1.2 broken predictors (the half-right missingness heuristics)


def m2_obs_table(w, it):
    """(k, n_out) table P(y | h) of a missingness2 line's underlying result; tgt columns: 0 = no flag, 1 = flag (= it['y'])."""
    k = w["k"]
    if it["kind"] == "tgt":
        p = np.where(np.arange(k) == it["on"], it["sens"] / 100.0, it["fpr"] / 100.0)
        return np.stack([1 - p, p], axis=1)
    return pct(w["parts"]["sig"]["table"])


def m2_factor(w, it, rates, mode="exact"):
    """Likelihood over h of one missingness2 line under ONE rate set: a present result y contributes P(y|h) P(rec | h, y); a
    missing one sum_y P(y|h) (1 - P(rec | h, y)). Modes: ignore_missingness = every mechanism as MAR; ignore_presence = a present
    result carries P(y|h) only (the recording factor of presence dropped); missing_only = present results unused; present_only =
    missing results carry nothing."""
    k = w["k"]
    T = m2_obs_table(w, it)
    if mode == "ignore_missingness":
        rates = {"mech": "mar", "rate": 50}
    lost = np.array([[m2_lost_prob(w, it, rates, h, y) for y in range(T.shape[1])] for h in range(k)])
    y = it.get("y")
    if y is None:
        return np.ones(k) if mode == "present_only" else (T * lost).sum(axis=1)
    if mode == "missing_only":
        return np.ones(k)
    if mode == "ignore_presence":
        return T[:, y].copy()
    return T[:, y] * (1.0 - lost[:, y])


def ccr_factors(w, i, it, axes, mode):
    """Composed cc_retained (common-cause sensors whose reports are lost with a weather-dependent chance): factors."""
    k = w["k"]
    tag = ("ev", i)
    out = []
    ret = it["ret"] if mode != "ignore_missingness" else {"mech": "mar", "rate": 50}
    if it["stmt"] == "copied" and mode != "ignore_dependence":
        pf = it["pf"] / 100.0
        out.append((tag, bc(axes, (f"z:{i}",), [1 - pf, pf])))
        if ret["mech"] == "fog":
            lost = np.array([ret["mc"], ret["mf"]]) / 100.0                 # index 0 = clear, 1 = fog
        else:
            lost = np.full(2, ret["rate"] / 100.0)
        for c in it["cs"]:
            if c is None:
                if mode != "present_only":
                    out.append((tag, bc(axes, (f"z:{i}",), lost)))
                continue
            if mode == "missing_only":
                continue
            M = np.stack([acc_lik(k, it["ac"], c), acc_lik(k, it["af"], c)], axis=1)
            if mode != "ignore_presence":
                M = M * (1 - lost)[None, :]
            out.append((tag, bc(axes, ("h", f"z:{i}"), M)))
        return out
    am = it["am"] if it["stmt"] == "independent" else (1 - it["pf"] / 100.0) * it["ac"] + (it["pf"] / 100.0) * it["af"]
    for c in it["cs"]:                                                  # independent sensors (or the dependence-blind rival):
        if c is None or mode == "missing_only":                          # the loss is constant in h
            continue
        out.append((tag, bc(axes, ("h",), acc_lik(k, am, c))))
    return out


def rlr_factors(w, i, it, axes, mode):
    """Composed relay_retained (relays of one unobserved origin; a report is passed on with a claim-dependent chance)."""
    k = w["k"]
    tag = ("ev", i)
    out = []
    keep = np.array([ret_keep(it, k, y) for y in range(k)]) if mode != "ignore_missingness" else np.full(k, 0.5)
    A = np.stack([acc_lik(k, it["acc"], hh) for hh in range(k)], axis=1)         # A[x, h] = P(origin says x | h)
    copied = it["stmt"] == "copied"
    if copied and mode != "ignore_dependence":
        out.append((tag, bc(axes, (f"x:{i}", "h"), A)))
    for rp in it["reps"]:
        if copied:
            T = relay_mat(k, rp["path"][0])                                      # T[y, x]
            L = T if mode != "ignore_dependence" else T @ A                       # rows y; columns x (or h: marginal)
        else:
            L = A                                                                 # independent reporter: L[y, h]
        ax = (f"x:{i}",) if (copied and mode != "ignore_dependence") else ("h",)
        c = rp.get("c")
        if c is None:
            if mode != "present_only":
                out.append((tag, bc(axes, ax, ((1 - keep)[:, None] * L).sum(axis=0))))
            continue
        if mode == "missing_only":
            continue
        out.append((tag, bc(axes, ax, L[c, :] * (1.0 if mode == "ignore_presence" else keep[c]))))
    return out


def model(w, mode="exact"):
    """(axes, factors): axes = [(name, size)], axis 0 = 'h'; factors = [(tag, array broadcastable to the joint)]."""
    if w["family"] == "hierarchical":
        return model_hier(w, mode)
    k, P = w["k"], w["parts"]
    axes = [("h", k)]
    rvs = P.get("rv", {})
    use_rv = mode != "ignore_reliability"
    if use_rv:
        for r in sorted(rvs):
            axes.append((f"rv:{r}", len(rvs[r]["levels"])))
    for i, it in enumerate(w["ev"]):
        if it["t"] == "relay" and it["stmt"] != "independent" and mode != "ignore_dependence":
            axes.append((f"x:{i}", k))
        if it["t"] == "copy" and it["stmt"] == "copied" and mode != "ignore_dependence":
            axes.append((f"c:{i}", 2))
        if it["t"] == "cc" and it["stmt"] == "copied" and mode != "ignore_dependence":
            axes.append((f"z:{i}", 2))
        if it["t"] == "ccr" and it["stmt"] == "copied" and mode != "ignore_dependence":     # v1.2 composed (confirm only)
            axes.append((f"z:{i}", 2))
        if it["t"] == "rlr" and it["stmt"] == "copied" and mode != "ignore_dependence":
            axes.append((f"x:{i}", k))
    m2 = P.get("miss2")                                                # v1.2 missingness2: a latent recording rule
    use_reg = m2 is not None and "regimes" in m2 and mode != "ignore_missingness"
    if use_reg:
        axes.append(("reg", len(m2["regimes"])))
    facs = []
    pr = np.full(k, 1.0 / k) if mode == "flat_prior" else pct(w["prior"])
    facs.append((("prior",), bc(axes, ("h",), pr)))
    if use_rv:
        for r in sorted(rvs):
            facs.append((("lat", r), bc(axes, (f"rv:{r}",), pct(rvs[r]["prior"]))))
    if use_reg:
        facs.append((("lat", "reg"), bc(axes, ("reg",), pct(m2["rprior"]))))
    miss = P.get("miss")
    for i, it in enumerate(w["ev"]):
        tag = ("ev", i)
        t = it["t"]
        if t == "tgt":
            if it.get("y") is None:                                    # a missing result
                if miss["mech"] == "informative" and mode not in ("ignore_missingness", "present_only"):
                    pf = tgt_lik(k, it["on"], it["sens"], it["fpr"], 1)
                    facs.append((tag, bc(axes, ("h",), pf * (1 - miss["qf"] / 100.0) + (1 - pf) * (1 - miss["qc"] / 100.0))))
                continue
            if mode == "missing_only" and miss is not None:            # v1.2 broken mode: present results unused
                continue
            f = tgt_lik(k, it["on"], it["sens"], it["fpr"], it["y"])
            if miss is not None and miss["mech"] == "informative" and mode not in ("ignore_missingness", "ignore_presence"):
                f = f * ((miss["qf"] if it["y"] else miss["qc"]) / 100.0)
            facs.append((tag, bc(axes, ("h",), f)))
        elif t == "log":
            mvec = np.where(np.arange(k) == it["on"], miss["m1"] / 100.0, miss["m0"] / 100.0)
            use_m = mode != "ignore_missingness" and miss["mech"] == "mnar"
            if it.get("y") is None:
                if use_m and mode != "present_only":
                    facs.append((tag, bc(axes, ("h",), mvec)))
                continue
            if mode == "missing_only":                                 # v1.2 broken mode: present logs unused
                continue
            f = tgt_lik(k, it["on"], miss["sens"], miss["fpr"], it["y"])
            if use_m and mode != "ignore_presence":
                f = f * (1 - mvec)
            facs.append((tag, bc(axes, ("h",), f)))
        elif t == "m2":                                                # v1.2 missingness2
            if use_reg:
                M = np.stack([m2_factor(w, it, dict(r, mech=m2["mech"]), mode) for r in m2["regimes"]], axis=1)
                facs.append((tag, bc(axes, ("h", "reg"), M)))
            else:
                rates = m2 if "regimes" not in m2 else {"mech": "mar", "rate": 50}
                facs.append((tag, bc(axes, ("h",), m2_factor(w, it, rates, mode))))
        elif t == "ccr":
            facs += ccr_factors(w, i, it, axes, mode)
        elif t == "rlr":
            facs += rlr_factors(w, i, it, axes, mode)
        elif t == "sig":
            facs.append((tag, bc(axes, ("h",), pct([row[it["y"]] for row in P["sig"]["table"]]))))
        elif t == "feat":
            p = pct(it["p"])
            facs.append((tag, bc(axes, ("h",), p if it["y"] else 1 - p)))
        elif t == "rep":
            facs.append((tag, bc(axes, ("h",), acc_lik(k, it["acc"], it["c"]))))
        elif t == "lrep":
            rvd = rvs[it["rv"]]
            if use_rv:
                lv = pct(rvd["levels"])
                M = np.empty((k, len(lv)))
                for j, a in enumerate(lv):
                    M[:, j] = np.where(np.arange(k) == it["c"], a, (1 - a) / (k - 1))
                facs.append((tag, bc(axes, ("h", f"rv:{it['rv']}"), M)))
            else:
                a = rv_mean(rvd)
                facs.append((tag, bc(axes, ("h",), np.where(np.arange(k) == it["c"], a, (1 - a) / (k - 1)))))
        elif t == "track":
            if use_rv and it["n"]:
                lv = pct(rvs[it["rv"]]["levels"])
                facs.append((tag, bc(axes, (f"rv:{it['rv']}",), lv ** it["s"] * (1 - lv) ** (it["n"] - it["s"]))))
        elif t == "relay":
            A = np.empty((k, k))                                       # A[x, h] = P(origin says x | h)
            for hh in range(k):
                A[:, hh] = acc_lik(k, it["acc"], hh)
            if it["stmt"] == "independent":
                for rp in it["reps"]:
                    facs.append((tag, bc(axes, ("h",), acc_lik(k, it["acc"], rp["c"]))))
            elif mode == "ignore_dependence":
                for rp in it["reps"]:
                    T = np.eye(k)
                    for f in rp["path"]:
                        T = relay_mat(k, f) @ T
                    facs.append((tag, bc(axes, ("h",), (T @ A)[rp["c"], :])))
            else:
                facs.append((tag, bc(axes, (f"x:{i}", "h"), A)))
                for rp in it["reps"]:
                    T = np.eye(k)
                    for f in rp["path"]:
                        T = relay_mat(k, f) @ T
                    facs.append((tag, bc(axes, (f"x:{i}",), T[rp["c"], :])))
        elif t == "copy":
            fa = acc_lik(k, it["a"], it["ca"])
            fb = acc_lik(k, it["b"], it["cb"])
            facs.append((tag, bc(axes, ("h",), fa)))
            if it["stmt"] != "copied" or mode == "ignore_dependence":
                facs.append((tag, bc(axes, ("h",), fb)))
            else:
                c = it["c"] / 100.0
                facs.append((tag, bc(axes, (f"c:{i}",), [1 - c, c])))     # index 1 = copied
                M = np.empty((k, 2))
                M[:, 0] = fb
                M[:, 1] = 1.0 if it["cb"] == it["ca"] else 0.0
                facs.append((tag, bc(axes, ("h", f"c:{i}"), M)))
        elif t == "cc":
            if it["stmt"] == "independent":
                for c in it["cs"]:
                    facs.append((tag, bc(axes, ("h",), acc_lik(k, it["am"], c))))
            elif mode == "ignore_dependence":
                am = (1 - it["pf"] / 100.0) * it["ac"] + (it["pf"] / 100.0) * it["af"]
                for c in it["cs"]:
                    facs.append((tag, bc(axes, ("h",), acc_lik(k, am, c))))
            else:
                pf = it["pf"] / 100.0
                facs.append((tag, bc(axes, (f"z:{i}",), [1 - pf, pf])))   # index 1 = fog
                for c in it["cs"]:
                    M = np.stack([acc_lik(k, it["ac"], c), acc_lik(k, it["af"], c)], axis=1)
                    facs.append((tag, bc(axes, ("h", f"z:{i}"), M)))
        elif t == "dist":
            Mr = read_lik(P["dmax"], P["pe"])
            facs.append((tag, bc(axes, ("h",), np.array([Mr[it["y"], P["dist"][it["at"]][hh]] for hh in range(k)]))))
        elif t == "grp":
            inside = np.array([hh in P["groups"][it["g"]] for hh in range(k)])
            p = np.where(inside, it["sens"] / 100.0, it["fpr"] / 100.0)
            facs.append((tag, bc(axes, ("h",), p if it["y"] else 1 - p)))
        else:
            raise ValueError(t)
    return axes, facs


def hier_fail_vec(w, g):
    """P(member g failing | the shared latents it depends on): (names, array)."""
    P = w["parts"]
    if w["structure"] == "member":
        return ("s",), pct(P["fail"])
    c = P["crew"][g]
    return ("s", f"cr:{c}"), pct(P["fail2"])


def model_hier(w, mode):
    P = w["parts"]
    G, L, tgt = P["G"], P["L"], P["target"]
    others = [g for g in range(G) if g != tgt]
    if mode == "no_pooling":                                           # members independent, each with its marginal rate
        axes = [("h", 2)] + [(f"x:{g}", 2) for g in others]
        facs = []
        for g in [tgt] + others:
            if w["structure"] == "member":
                pb = float(np.dot(pct(P["sprior"]), pct(P["fail"])))
            else:
                cp = P["cprior"] / 100.0
                pb = float(sum(pct(P["sprior"])[s] * ((1 - cp) * P["fail2"][s][0] + cp * P["fail2"][s][1]) / 100.0
                               for s in range(L)))
            ax = "h" if g == tgt else f"x:{g}"
            facs.append((("prior",) if g == tgt else ("model",), bc(axes, (ax,), [pb, 1 - pb])))
        for i, c in enumerate(w["ev"]):
            ax = "h" if c["g"] == tgt else f"x:{c['g']}"
            p = np.array([c["sens"], c["fpr"]]) / 100.0
            facs.append((("ev", i), bc(axes, (ax,), p if c["y"] else 1 - p)))
        return axes, facs
    axes = [("h", 2), ("s", L)]
    if w["structure"] == "two_factor":
        axes += [("cr:0", 2), ("cr:1", 2)]
    axes += [(f"x:{g}", 2) for g in others]
    facs = []
    sp = np.full(L, 1.0 / L) if mode == "flat_prior" else pct(P["sprior"])
    facs.append((("prior",), bc(axes, ("s",), sp)))
    if w["structure"] == "two_factor":
        cp = P["cprior"] / 100.0
        for c in (0, 1):
            facs.append((("model",), bc(axes, (f"cr:{c}",), [1 - cp, cp])))
    for g in [tgt] + others:
        nm, fv = hier_fail_vec(w, g)
        ax = "h" if g == tgt else f"x:{g}"
        arr = np.stack([fv, 1 - fv], axis=0)                           # status axis first: 0 = failing, 1 = fine
        facs.append((("model",), bc(axes, (ax,) + nm, arr)))
    for i, c in enumerate(w["ev"]):
        ax = "h" if c["g"] == tgt else f"x:{c['g']}"
        p = np.array([c["sens"], c["fpr"]]) / 100.0
        facs.append((("ev", i), bc(axes, (ax,), p if c["y"] else 1 - p)))
    return axes, facs


def solve(w, mode="exact", double=None):
    """Normalised joint over the latent grid. mode 'prior_only' drops every evidence factor. `double`: the index of an
    evidence item whose factors are applied twice (the order-sensitive broken predictor)."""
    m = "exact" if mode == "prior_only" else mode
    axes, facs = model(w, m)
    J = np.ones(tuple(n for _, n in axes))
    for tag, f in facs:
        if mode == "prior_only" and tag[0] == "ev":
            continue
        J = J * f
        if double is not None and tag == ("ev", double):
            J = J * f
    Z = J.sum()
    assert Z > 0 and np.isfinite(Z), (w["family"], w["structure"], Z)
    J = J / Z
    assert abs(J.sum() - 1.0) < 1e-12
    return axes, J


def marg(axes, J, name):
    i = [a for a, _ in axes].index(name)
    out = J.sum(axis=tuple(j for j in range(J.ndim) if j != i))
    assert abs(out.sum() - 1.0) < 1e-12
    return out


def posterior_h(w, mode="exact", double=None):
    axes, J = solve(w, mode, double)
    return marg(axes, J, "h")


def next_outcomes(w):
    nx = w["next"]
    t = nx["t"]
    if t in ("tgt", "line_check", "grp"):
        return 2
    if t == "sig":
        return len(w["parts"]["sig"]["cats"])
    if t == "feat":
        return 2
    if t == "rep_new":
        return w["k"]
    if t == "right_next":
        return 2
    if t == "dist":
        return w["parts"]["dmax"] + 1
    raise ValueError(t)


def next_table(w, axes):
    """P(next outcome | latents), broadcastable to the joint with the outcome as a trailing axis."""
    nx, k, P = w["next"], w["k"], w["parts"]
    names = [a for a, _ in axes]
    t = nx["t"]
    if t == "tgt":
        p = np.where(np.arange(k) == nx["on"], nx["sens"] / 100.0, nx["fpr"] / 100.0)
        T, on = np.stack([p, 1 - p], 1), ("h",)
    elif t == "sig":
        T, on = pct(P["sig"]["table"]), ("h",)
    elif t == "feat":
        p = pct(nx["p"])
        T, on = np.stack([p, 1 - p], 1), ("h",)
    elif t == "rep_new":
        T = np.stack([acc_lik(k, nx["acc"], hh) for hh in range(k)], 0)      # T[h, claim]
        on = ("h",)
    elif t == "right_next":
        ax = f"rv:{nx['rv']}"
        if ax in names:
            lv = pct(P["rv"][nx["rv"]]["levels"])
            T, on = np.stack([lv, 1 - lv], 1), (ax,)
        else:
            a = rv_mean(P["rv"][nx["rv"]])
            T, on = np.array([[a, 1 - a]] * axes[0][1]), ("h",)
    elif t == "line_check":
        tgt = P["target"]
        ax = "h" if nx["g"] == tgt else f"x:{nx['g']}"
        p = np.array([nx["sens"], nx["fpr"]]) / 100.0
        T, on = np.stack([p, 1 - p], 1), (ax,)
    elif t == "dist":
        Mr = read_lik(P["dmax"], P["pe"])
        T = np.stack([Mr[:, P["dist"][nx["at"]][hh]] for hh in range(k)], 0)
        on = ("h",)
    elif t == "grp":
        inside = np.array([hh in P["groups"][nx["g"]] for hh in range(k)])
        p = np.where(inside, nx["sens"] / 100.0, nx["fpr"] / 100.0)
        T, on = np.stack([p, 1 - p], 1), ("h",)
    else:
        raise ValueError(t)
    n_out = T.shape[-1]
    ext = axes + [("_y", n_out)]
    return bc(ext, on + ("_y",), T)


def predictive(w, mode="exact", double=None, plugin=False):
    axes, J = solve(w, mode, double)
    if plugin:                                                         # point mass at the joint posterior mode
        J2 = np.zeros_like(J)
        J2[np.unravel_index(int(np.argmax(J)), J.shape)] = 1.0
        J = J2
    T = next_table(w, axes)
    p = (J[..., None] * T).reshape(-1, T.shape[-1]).sum(0)
    assert abs(p.sum() - 1.0) < 1e-12
    return p


def latent_post(w, mode="exact", double=None):
    axes, J = solve(w, mode, double)
    ax = w["latq"]["axis"]
    if ax not in [a for a, _ in axes]:                                 # a mode without that latent: its prior
        if ax.startswith("rv:"):
            return pct(w["parts"]["rv"][ax[3:]]["prior"])
        if ax == "reg":
            return pct(w["parts"]["miss2"]["rprior"])
        return pct(w["parts"]["sprior"])
    return marg(axes, J, ax)


def latent_levels(w):
    ax = w["latq"]["axis"]
    if ax.startswith("rv:"):
        return w["parts"]["rv"][ax[3:]]["levels"]
    if ax == "reg":
        return [f"Rule {c}" for c in "AB"[:len(w["parts"]["miss2"]["regimes"])]]
    return w["parts"]["grades"]


# --------------------------------------------------------------------------- decisions and value of information
def act_value(Lm, q):
    return float((Lm @ q).min())


def info_lik(w, opt):
    """Outcome likelihood table (k, n_out) of a one-shot information option."""
    k = w["k"]
    if opt["t"] in ("test",):
        p = np.where(np.arange(k) == opt["on"], opt["sens"] / 100.0, opt["fpr"] / 100.0)
        return np.stack([p, 1 - p], 1)
    if opt["t"] == "consult":
        return np.stack([acc_lik(k, opt["acc"], hh) for hh in range(k)], 0)
    raise ValueError(opt["t"])


def expect_after(Lm, q, lik):
    """E_y[ min_a E[L | y] ] for one observation with likelihood table lik (k, n_out)."""
    tot = 0.0
    for y in range(lik.shape[1]):
        u = q * lik[:, y]
        s = u.sum()
        if s > 0:
            tot += s * act_value(Lm, u / s)
    return tot


def option_values(w, q, fees=1.0, scale_losses=1.0, scale_fees=1.0, allow_info=True):
    """Expected total cost of every option, canonical order: the k actions, abstain (if any), the info options."""
    Lm = loss_matrix(w, scale_losses)
    vals = list(Lm @ q)
    if w["info"] is None:
        return np.asarray(vals)
    k = w["k"]
    for opt in w["info"]:
        if not allow_info:
            vals.append(np.inf)
            continue
        t = opt["t"]
        if t in ("test", "consult"):
            v = fees * scale_fees * opt["fee"] + expect_after(Lm, q, info_lik(w, opt))
        elif t == "collect":
            tab = pct(opt["table"])
            v = fees * scale_fees * opt["fee"] * opt["n"]
            for c in compositions(opt["n"], tab.shape[1]):
                lik = multinom_coef(c) * np.prod(tab ** np.asarray(c)[None, :], axis=1)
                u = q * lik
                s = u.sum()
                if s > 0:
                    v += s * act_value(Lm, u / s)
        elif t == "two":
            a, b = opt["tests"][opt["first"]], opt["tests"][1 - opt["first"]]
            la = info_lik(w, {"t": "test", **a})
            lb = info_lik(w, {"t": "test", **b})
            v = fees * scale_fees * a["fee"]
            for y in range(2):
                u = q * la[:, y]
                s = u.sum()
                if s <= 0:
                    continue
                qy = u / s
                stop = act_value(Lm, qy)
                cont = fees * scale_fees * b["fee"] + expect_after(Lm, qy, lb)
                v += s * min(stop, cont)
        else:
            raise ValueError(t)
        vals.append(v)
    assert len(vals) == n_options(w, "acquire") or w["info"] is None
    return np.asarray(vals, dtype=np.float64)


# Value-bin edges (fixed per kind; chosen once from the pooled TRAIN distribution of a v1.1 pilot, as numeric v2's zoom rule, so
# that bins are roughly equally used; never per item): regret / gap as fractions of the largest STATED loss; net VOI as
# multiples of the named option's STATED fee, with "exactly minus the fee" (the result cannot change the decision) its own bin.
VALUE_FR = {"value_regret": (0.05, 0.15, 0.3, 0.45, 0.7), "value_gap": (0.01, 0.025, 0.05, 0.1, 0.2)}
VOI_MULT = (-1.0, -0.5, 0.0, 1.0, 3.0)


def round2sig(v):
    e = math.floor(math.log10(v))
    q = 10.0 ** (e - 1)
    return float(round(round(v / q) * q, 6))


def option_fee(opt):
    """The stated fee of an information option (collect: n readings; two-step: the first test's fee)."""
    t = opt["t"]
    if t in ("test", "consult"):
        return opt["fee"]
    if t == "collect":
        return opt["fee"] * opt["n"]
    return opt["tests"][opt["first"]]["fee"]


def value_edges(w, qk):
    """Bin edges in money units from VISIBLE quantities only (the largest stated loss; the named option's stated fee), fixed
    before anything is solved."""
    if qk == "value_voi":
        fee = option_fee(w["info"][w["vq"]["voi"]])
        return [m * fee for m in VOI_MULT]
    S = loss_scale(w)
    out = []
    for f in VALUE_FR[qk]:
        e = round2sig(f * S)
        if e > 0 and (not out or e > out[-1]):
            out.append(e)
    return out


def decision_values(w, q, fees=1.0, scale_losses=1.0):
    """Expected (total) cost of every option of the world's decision question under belief q."""
    if w["info"] is not None:
        return option_values(w, q, fees=fees, scale_losses=scale_losses)
    return loss_matrix(w, scale_losses) @ q


def value_quantity(w, qk, q, fees=1.0, scale_losses=1.0):
    """(the value, the best expected loss) for a value question under belief q."""
    el = decision_values(w, q, fees, scale_losses)
    if qk == "value_regret":
        return float(el[w["vq"]["regret"]] - el.min()), float(el.min())
    if qk == "value_gap":
        srt = np.sort(el)
        return float(srt[1] - srt[0]), float(srt[0])
    nA = w["k"] + (1 if w["loss"]["E"] is not None else 0)
    return float(el[:nA].min() - el[nA + w["vq"]["voi"]]), float(el.min())


def value_bin(w, qk, x, ref, edge_scale=1.0):
    """Bin index of a value: regret 0 = 'zero (a best option)', then (0, e1), [e1, e2), ...; gap [0, e1), ...; net VOI
    0 = exactly -fee (cannot change the decision), (-fee, -fee/2), [-fee/2, 0), [0, fee), [fee, 3 fee), [3 fee, inf). A value
    within 1e-9 (relative) of an edge counts as on it."""
    edges = [e * edge_scale for e in value_edges(w, qk)]
    tol = 1e-9 * max(1.0, abs(ref), max(abs(e) for e in edges))
    if qk == "value_regret":
        if x <= tol:
            return 0
        return 1 + sum(1 for e in edges if x >= e - tol)
    if qk == "value_voi":
        if x <= edges[0] + tol:
            return 0
        return 1 + sum(1 for e in edges[1:] if x >= e - tol)
    return sum(1 for e in edges if x >= e - tol)


def n_value_bins(w, qk):
    return len(value_edges(w, qk)) + (2 if qk == "value_regret" else 1)


def difficulty(w):
    axes, _ = model(w, "exact")
    depth = (2 if w["structure"] == "two_step" else 1) if w["family"] == "acquisition" else 0
    return {"k": w["k"], "grid": int(np.prod([n for _, n in axes])), "acq_depth": depth, "n_evidence": len(w["ev"])}


def n_options(w, qk):
    if qk in VALUE_KINDS:
        return n_value_bins(w, qk)
    k = w["k"]
    if qk in ("posterior",):
        return k
    if qk == "latent":
        return len(latent_levels(w))
    if qk == "predictive":
        return next_outcomes(w)
    nA = k + (1 if w["loss"]["E"] is not None else 0)
    if qk == "action":
        return nA
    return nA + len(w["info"])


# --------------------------------------------------------------------------- exact targets
def exact_target(w, qk):
    """(target vector in CANONICAL option order, extras)."""
    if qk == "posterior":
        p = posterior_h(w)
        return p, {}
    if qk == "latent":
        return latent_post(w), {}
    if qk == "predictive":
        return predictive(w), {}
    q = posterior_h(w)
    if qk in VALUE_KINDS:
        x, ref = value_quantity(w, qk, q)
        b = value_bin(w, qk, x, ref)
        t = np.zeros(n_value_bins(w, qk))
        t[b] = 1.0
        named = w["vq"]["voi"] if qk == "value_voi" else (w["vq"]["regret"] if qk == "value_regret" else None)
        return t, {"value": x, "bin": b, "edges": value_edges(w, qk), "named": named, "loss_scale": loss_scale(w),
                   "fee": option_fee(w["info"][named]) if qk == "value_voi" else None}
    el = option_values(w, q) if qk == "acquire" else loss_matrix(w) @ q
    best = argmin_set(el)
    ex = {"expected_loss": [float(x) for x in el], "bayes_set": best, "min_expected_loss": float(el.min()),
          "loss_scale": loss_scale(w)}
    if qk == "acquire":
        nA = w["k"] + (1 if w["loss"]["E"] is not None else 0)
        act_now = float(np.min(el[:nA]))
        ex["act_now_value"] = act_now
        ex["voi_net"] = [float(act_now - x) for x in el[nA:]]
        ex["buy_is_optimal"] = bool(min(best) >= nA)
    return onehot_set(len(el), best), ex


# --------------------------------------------------------------------------- rendering (visible spec + render config only)
class Fmt:
    def __init__(self, prob, money, unit):
        self.prob, self.money, self.unit = prob, money, unit

    def p(self, v):
        v = int(v)
        if self.prob == "pct":
            return f"{v}%"
        if self.prob == "dec":
            return f"{v / 100:.2f}"
        return f"{v} in 100"

    def m(self, v, fee=False):
        v = int(v)
        if self.unit == "points":
            return f"{v:,} points"
        if v == 0:
            return "$0"
        if self.money == "usd" or (self.money == "mixed" and fee):
            return f"${v:,}"
        return f"${v / 1000:g} thousand"

    def mv(self, v, fee=False):
        """A money amount that may be fractional (value-question bin edges)."""
        if abs(v - round(v)) < 1e-9:
            return self.m(int(round(v)), fee)
        if self.unit == "points":
            return f"{v:g} points"
        if self.money == "usd" or (self.money == "mixed" and fee):
            return f"${v:,.2f}"
        return f"${v / 1000:g} thousand"


PROB_FMTS = ("pct", "dec", "in100")
MONEY_FMTS = ("usd", "kusd", "mixed")


def cap(s):
    return s[:1].upper() + s[1:]


def draw_fmt(rr):
    return {"prob": PROB_FMTS[int(rr.choice(3, p=[0.5, 0.3, 0.2]))],
            "money": MONEY_FMTS[int(rr.choice(3, p=[0.6, 0.2, 0.2]))]}


def draw_tpl(rr, mode="v1"):
    """v1: any variant. v1.1: variants 0-1 in every split but test_novel_wording ('base11'); variant 2 is reserved for it."""
    if mode == "v1":
        return {key: int(rr.integers(len(TPL[key]))) for key in TPL_KEYS}
    keys = TPL_KEYS + TPL11_KEYS
    if mode == "novel":
        return {key: 2 for key in keys}
    return {key: int(rr.integers(2)) for key in keys}


def draw_render(rr, w, mode="v1"):
    """A render config: templates, number formats, display orders. Reads only sizes of the visible spec."""
    R = {"fmt": draw_fmt(rr), "tpl": draw_tpl(rr, mode)}
    k = w["k"]
    natural = w["family"] == "multimodal" and w["structure"] in ("ring_distance", "grid_distance")
    R["hyp"] = list(range(k)) if natural else [int(i) for i in rr.permutation(k)]
    R["ev"] = [int(i) for i in rr.permutation(len(w["ev"]))]
    R["opt_seed"] = int(rr.integers(2 ** 31))
    return R


def opt_perm(R, qk, n):
    if qk == "posterior":
        return None
    return [int(i) for i in np.random.default_rng([R["opt_seed"], QKINDS.index(qk)]).permutation(n)]


def T_(_rc, _key, **kw):
    v = _rc["tpl"][_key]
    if _key in TPL12:                                                  # v1.2 keys (4 variants)
        return TPL12[_key][v].format(**kw)
    if v == 3:                                                         # v1.2 confirm-only wording
        return TPL_CONFIRM[_key].format(**kw)
    return (TPL[_key] if _key in TPL else TPL11[_key])[v].format(**kw)


def items_join(parts):
    return ", ".join(parts)


def render_state(w, R):
    fam = w["family"]
    if fam == "hierarchical":
        return render_hier(w, R)
    if fam == "multimodal":
        return render_mm(w, R)
    if fam == "missingness2":
        return render_m2(w, R)
    if fam == "composed":
        return render_composed(w, R)
    sk = SKINS[w["skin"]]
    fmt = Fmt(R["fmt"]["prob"], R["fmt"]["money"], sk["unit"])
    k, names, hyp = w["k"], w["names"], R["hyp"]
    S = dict(noun=sk["noun"], nouns=sk["nouns"], adj=sk["adj"], isp=sk["isp"])
    lines = [sk["setting"].format(n=k) + " " + sk["one"] + " The possibilities are: " + ", ".join(names[j] for j in hyp) + "."]
    lines.append(T_(R, "prior", items=items_join(f"{names[j]} {fmt.p(w['prior'][j])}" for j in hyp)))
    P = w["parts"]
    rep_name = lambda i: sk["reporters"][i]                           # noqa: E731
    if fam in ("diagnosis", "missingness", "decision", "acquisition"):
        if P.get("sig"):
            lines.append(T_(R, "sig_head", Sig=cap(sk["sig"]), sig=sk["sig"], **S))
            for j in hyp:
                lines.append(T_(R, "sig_row", name=names[j], isp=sk["isp"],
                                items=items_join(f"{c} {fmt.p(v)}" for c, v in zip(P["sig"]["cats"], P["sig"]["table"][j]))))
        miss = P.get("miss")
        if miss:
            key = {"mar": "mar", "informative": "inf", "mnar": "mnar", "mnar_mar": "mnar_mar"}[miss["mech"]]
            lines.append(T_(R, key, qf=fmt.p(miss.get("qf", 0)), qc=fmt.p(miss.get("qc", 0)), m1=fmt.p(miss.get("m1", 0)),
                            m0=fmt.p(miss.get("m0", 0)), sens=fmt.p(miss.get("sens", 0)), fpr=fmt.p(miss.get("fpr", 0)), **S))
        order = R["ev"]
        if not w["ev"]:
            lines.append(T_(R, "no_ev"))
        else:
            lines.append(T_(R, "ev_head", **S))
            sig_ys = [P["sig"]["cats"][w["ev"][i]["y"]] for i in order if w["ev"][i]["t"] == "sig"]
            if sig_ys:
                lines.append("- " + T_(R, "sig_obs", sig=sk["sig"], items=items_join(sig_ys), **S))
            for i in order:
                it = w["ev"][i]
                if it["t"] == "tgt":
                    tk = "tgt" if it["y"] is not None else "tgt_miss"
                    lines.append("- " + T_(R, tk, Test=cap(sk["test"]), test=sk["test"], name=names[it["on"]],
                                           sens=fmt.p(it["sens"]), fpr=fmt.p(it["fpr"]),
                                           res=("flagged" if it["y"] else "not flagged"),
                                           resv=("flagged" if it["y"] else "did not flag"), **S))
                elif it["t"] == "log":
                    lines.append("- " + T_(R, "log_line", name=names[it["on"]],
                                           res=("missing" if it["y"] is None else "flagged" if it["y"] else "no flag")))
                elif it["t"] == "feat":
                    feat = sk["features"][it["f"]]
                    lines.append("- " + T_(R, "feat", Feature=cap(feat), feature=feat,
                                           items=items_join(f"{fmt.p(it['p'][j])} if {names[j]} {sk['isp']}" for j in hyp),
                                           res=("present" if it["y"] else "absent")))
        if w["loss"] is not None:
            lines += render_loss(w, R, fmt, sk)
        if w["info"] is not None:
            lines += render_info(w, R, fmt, sk)
        return "\n".join(lines)
    if fam == "copied":
        lines.append(T_(R, "indep_note"))
        lines.append(T_(R, "ev_plain"))
        for i in R["ev"]:
            b = w["ev"][i]
            if b["t"] == "rep":
                rn = rep_name(b["src"])
                lines.append("- " + T_(R, "rep", Rep=cap(rn), acc=fmt.p(b["acc"]), claim=names[b["c"]], **S))
            elif b["t"] == "relay" and b["stmt"] in ("copied", "independent"):
                rl = [rep_name(rp["src"]) for rp in b["reps"]]
                rlist = ", ".join(rl[:-1]) + " and " + rl[-1]
                key = "relay" if b["stmt"] == "copied" else "relay_ind"
                lines.append("- " + T_(R, key, rlist=rlist, Rlist=cap(rlist), claim=names[b["reps"][0]["c"]], O=rep_name(b["origin"]),
                                       O_cap=cap(rep_name(b["origin"])), acc=fmt.p(b["acc"]), **S))
            elif b["t"] == "relay":                                     # chain
                O = rep_name(b["origin"])
                parts = [T_(R, "chain_head", O=O, O_cap=cap(O), a=fmt.p(b["acc"]), **S)]
                for rp in b["reps"]:
                    rn = rep_name(rp["src"])
                    if len(rp["path"]) == 1:
                        parts.append(T_(R, "chain_direct", R_cap=cap(rn), O=O, f=fmt.p(rp["path"][0]), claim=names[rp["c"]]))
                    else:
                        mn = rep_name(rp["via"])
                        parts.append(T_(R, "chain_two", R_cap=cap(rn), R=rn, M=mn, O=O, fm=fmt.p(rp["path"][0]),
                                        f=fmt.p(rp["path"][1]), claim=names[rp["c"]]))
                lines.append("- " + " ".join(parts))
            elif b["t"] == "copy":
                A, B = rep_name(b["A"]), rep_name(b["B"])
                key = "copy" if b["stmt"] == "copied" else "copy_ind"
                lines.append("- " + T_(R, key, A=A, A_cap=cap(A), B=B, B_cap=cap(B), a=fmt.p(b["a"]), b=fmt.p(b["b"]),
                                       c=fmt.p(b["c"]), ca=names[b["ca"]], cb=names[b["cb"]], **S))
            elif b["t"] == "cc":
                sn = [f"Sensor {j + 1}" for j in b["sens"]]
                slist = ", ".join(sn[:-1]) + " and " + sn[-1]
                claims = "; ".join(f"{s} says {names[c]}" for s, c in zip(sn, b["cs"]))
                if b["stmt"] == "copied":
                    lines.append("- " + T_(R, "cc", slist=slist, pf=fmt.p(b["pf"]), ac=fmt.p(b["ac"]), af=fmt.p(b["af"]),
                                           claims=cap(claims), **S))
                else:
                    lines.append("- " + T_(R, "cc_ind", slist=slist, am=fmt.p(b["am"]), claims=cap(claims), **S))
        return "\n".join(lines)
    if fam == "reliability":
        rv = P["rv"]
        lines.append(T_(R, "lat_rule", **S))
        if P.get("agency"):
            sl = [rep_name(s) for s in P["agency"]]
            slist = ", ".join(sl[:-1]) + " and " + sl[-1]
            lines.append(T_(R, "lat_shared", slist=slist, Slist=cap(slist), levels=levels_text(rv["0"], fmt),
                            lprior=lprior_text(rv["0"], fmt)))
        else:
            for s, r in sorted(P["srcs"].items(), key=lambda kv: kv[1]):
                rn = rep_name(int(s))
                lines.append(T_(R, "lat_src", Src=cap(rn), src=rn, L=len(rv[r]["levels"]), levels=levels_text(rv[r], fmt),
                                lprior=lprior_text(rv[r], fmt)))
        lines.append(T_(R, "ev_plain"))
        for i in R["ev"]:
            it = w["ev"][i]
            if it["t"] == "track":
                if it.get("agency"):
                    if it["n"]:
                        lines.append("- " + T_(R, "track_ag", n=it["n"], s=it["s"]))
                    continue
                rn = rep_name(it["src"])
                lines.append("- " + (T_(R, "track", n=it["n"], s=it["s"], src=rn, Src=cap(rn)) if it["n"]
                                     else T_(R, "track0", src=rn, Src=cap(rn))))
            elif it["t"] == "lrep":
                rn = rep_name(it["src"])
                lines.append("- " + T_(R, "lrep", Src=cap(rn), src=rn, claim=names[it["c"]]))
            elif it["t"] == "rep":
                rn = rep_name(it["src"])
                lines.append("- " + T_(R, "rep", Rep=cap(rn), acc=fmt.p(it["acc"]), claim=names[it["c"]], **S))
        return "\n".join(lines)
    raise ValueError(fam)


def levels_text(rvd, fmt):
    nm = LEVEL_NAMES[len(rvd["levels"])]
    return ", ".join(f"{n} (right with chance {fmt.p(a)})" for n, a in zip(nm, rvd["levels"]))


def lprior_text(rvd, fmt):
    nm = LEVEL_NAMES[len(rvd["levels"])]
    return ", ".join(f"{n} {fmt.p(p)}" for n, p in zip(nm, rvd["prior"]))


def action_text(w, a, sk, now=False):
    L = w["loss"]
    k = w["k"]
    if a < k:
        t = (sk["rely"] if L["form"] == "avoid" else sk["act"]).format(h=w["names"][a])
    else:
        t = sk["abstain"]
    return t + (" now" if now else "")


def render_loss(w, R, fmt, sk):
    L, k, names = w["loss"], w["k"], w["names"]
    S = dict(noun=sk["noun"], nouns=sk["nouns"], adj=sk["adj"], isp=sk["isp"])
    out = [T_(R, "cost_head")]
    order = [int(i) for i in np.random.default_rng([R["opt_seed"], 99]).permutation(k)]
    for a in order:
        if L["form"] == "fix":
            out.append(T_(R, "fix", act=action_text(w, a, sk), F=fmt.m(L["F"][a]), M=fmt.m(L["M"][a]), name=names[a], **S))
        elif L["form"] == "avoid":
            out.append(T_(R, "avoid", act=action_text(w, a, sk), B=fmt.m(L["B"][a]), P=fmt.m(L["P"][a]), name=names[a], **S))
        else:
            items = "; ".join(f"{fmt.m(L['L'][a][hh])} if {names[hh]} {sk['isp']}" for hh in R["hyp"])
            out.append(T_(R, "mat", act=action_text(w, a, sk), items=items))
    if L["E"] is not None:
        out.append(T_(R, "abst", abstain=sk["abstain"], E=fmt.m(L["E"]), **S))
    return out


def render_info(w, R, fmt, sk):
    names = w["names"]
    S = dict(noun=sk["noun"], nouns=sk["nouns"], adj=sk["adj"], isp=sk["isp"])
    out = [T_(R, "info_head", agent=sk["agent"], Agent=cap(sk["agent"]))]
    seen_col = False
    for opt in w["info"]:
        t = opt["t"]
        if t == "test":
            out.append(T_(R, "info_tgt", Test=cap(sk["test"]), test=sk["test"], name=names[opt["on"]], fee=fmt.m(opt["fee"], True),
                          sens=fmt.p(opt["sens"]), fpr=fmt.p(opt["fpr"]), **S))
        elif t == "consult":
            out.append(T_(R, "info_con", Consult=cap(sk["consult"]), consult_l=sk["consult"][:1].lower() + sk["consult"][1:],
                          fee=fmt.m(opt["fee"], True), acc=fmt.p(opt["acc"]), **S))
        elif t == "collect" and not seen_col:
            seen_col = True
            out.append(T_(R, "info_col", channel=sk["channel"], fee=fmt.m(opt["fee"], True), agent=sk["agent"],
                          Agent=cap(sk["agent"]), **S))
            for j in R["hyp"]:
                out.append(f"  - if {names[j]} {sk['isp']}: " + items_join(f"{c} {fmt.p(v)}" for c, v in zip(opt["cats"], opt["table"][j])))
        elif t == "two" and opt["first"] == 0:
            for tst in opt["tests"]:
                out.append(T_(R, "info_tgt", Test=cap(sk["test"]), test=sk["test"], name=names[tst["on"]],
                              fee=fmt.m(tst["fee"], True), sens=fmt.p(tst["sens"]), fpr=fmt.p(tst["fpr"]), **S))
            out.append(T_(R, "two_note", agent=sk["agent"]))
    return out


def render_hier(w, R):
    hs = HIER_SKINS[w["skin"]]
    P = w["parts"]
    fmt = Fmt(R["fmt"]["prob"], R["fmt"]["money"], "usd")
    G = P["G"]
    mem = lambda g: f"{hs['member']} {g + 1}"                         # noqa: E731
    lines = [hs["setting"].format(G=G) + f" The {hs['snoun']} is one of these grades: " + ", ".join(P["grades"]) + "."]
    sprior = items_join(f"{gname} {fmt.p(p)}" for gname, p in zip(P["grades"], P["sprior"]))
    if w["structure"] == "member":
        fails = items_join(f"{fmt.p(f)} if the {hs['snoun']} is {gname}" for gname, f in zip(P["grades"], P["fail"]))
        lines.append(T_(R, "hier_model", snoun=hs["snoun"], sprior=sprior, member_l=hs["member"].lower(), thing=hs["thing"],
                        bad=hs["bad"], fails=fails))
    else:
        lines.append(T_(R, "hier_prior2", snoun=hs["snoun"], sprior=sprior))
        crewmap = "; ".join(f"{hs['f2names'][c]}: " + ", ".join(mem(g) for g in range(G) if P["crew"][g] == c)
                            for c in (0, 1) if any(P["crew"][g] == c for g in range(G)))
        table = "; ".join(f"{fmt.p(P['fail2'][s][c])} if the {hs['snoun']} is {P['grades'][s]} and the {hs['factor2']} is "
                          f"{hs['f2levels'][c]}" for s in range(2) for c in range(2))
        lines.append(T_(R, "hier_crew", member_l=hs["member"].lower(), factor2=hs["factor2"], factor2s=hs["factor2s"],
                        crewmap=crewmap,
                        lv0=hs["f2levels"][0], lv1=hs["f2levels"][1], cp=fmt.p(P["cprior"]), snoun=hs["snoun"],
                        thing=hs["thing"], bad=hs["bad"], table=table))
    lines.append(T_(R, "ev_plain"))
    checked = {c["g"] for c in w["ev"]}
    for i in R["ev"]:
        c = w["ev"][i]
        lines.append("- " + T_(R, "hier_check", Check=cap(hs["check"]), check=hs["check"], member=mem(c["g"]), thing=hs["thing"],
                               sens=fmt.p(c["sens"]), fpr=fmt.p(c["fpr"]), bad=hs["bad"], good=hs["good"],
                               res=("flagged" if c["y"] else "not flagged"), resv=("flagged" if c["y"] else "did not flag")))
    for g in range(G):
        if g not in checked:
            lines.append("- " + T_(R, "hier_none", member=mem(g)))
    return "\n".join(lines)


def render_mm(w, R):
    ms = MM_SKINS[w["skin"]]
    P = w["parts"]
    fmt = Fmt(R["fmt"]["prob"], R["fmt"]["money"], "usd")
    k, names = w["k"], w["names"]
    st = w["structure"]
    if st == "ring_distance":
        lines = [ms["ring"].format(N=P["N"])]
    elif st == "grid_distance":
        lines = [ms["grid"].format(RC=k, R=P["R"], C=P["C"])]
    else:
        glist = "; ".join(f"{gn} {ms['gword']}: " + ", ".join(names[j] for j in g) for gn, g in zip(P["gnames"], P["groups"]))
        lines = [ms["groups"].format(k=k, glist=glist)]
    lines.append(T_(R, "prior", items=items_join(f"{names[j]} {fmt.p(w['prior'][j])}" for j in R["hyp"])))
    if st == "ring_distance":
        lines.append(T_(R, "ring_rule", Sensor_desc=cap(ms["sensor_desc"]), pe=fmt.p(P["pe"]), target=ms["target"]))
    elif st == "grid_distance":
        lines.append(T_(R, "grid_rule", gsensor=ms["gsensor"], pe=fmt.p(P["pe"]), target=ms["gtarget"]))
    if not w["ev"]:
        lines.append(T_(R, "no_ev"))
    else:
        lines.append(T_(R, "ev_plain"))
    for i in R["ev"]:
        it = w["ev"][i]
        if it["t"] == "dist" and st == "ring_distance":
            lines.append("- " + T_(R, "ring_read", sensor=ms["sensor"], pos=names[it["at"]], r=it["y"], unit=ms["unit"]))
        elif it["t"] == "dist":
            lines.append("- " + T_(R, "grid_read", gsensor=ms["gsensor"], pos=names[it["at"]], r=it["y"]))
        else:
            lines.append("- " + T_(R, "grp_read", Gtest=cap(ms["gtest"]), gname=P["gnames"][it["g"]], gword=ms["gword"],
                                   target=ms["grtarget"],
                                   sens=fmt.p(it["sens"]), fpr=fmt.p(it["fpr"]), res=("flagged" if it["y"] else "not flagged"),
                                   resv=("flagged" if it["y"] else "did not flag")))
    return "\n".join(lines)


ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]


def x4(num):
    """An exact likelihood factor given as an integer numerator over 10,000 (products of two printed percentages), 4 dp."""
    return f"{num / 10000:.4f}"


def m2_factor_num(w, it, rates, h):
    """The explicit-factor text's numbers: the factor of line `it` for hypothesis h as an INTEGER over 10,000 (exact)."""
    if it["kind"] == "tgt":
        p1 = it["sens"] if h == it["on"] else it["fpr"]
        probs = [100 - p1, p1]
    else:
        probs = list(w["parts"]["sig"]["table"][h])
    mech = rates["mech"]

    def rec(y):                                                        # P(recorded | h, y) in percent
        if mech == "mar":
            return 100 - rates["rate"]
        if mech == "hyp":
            on = it["on"] if it["kind"] == "tgt" else rates["on"]
            return 100 - (rates["m1"] if h == on else rates["m0"])
        return (rates["qf"] if y else rates["qc"]) if it["kind"] == "tgt" else rates["q"][y]
    y = it.get("y")
    if y is None:
        return sum(probs[v] * (100 - rec(v)) for v in range(len(probs)))
    return probs[y] * rec(y)


def m2_line_desc(w, it, sk, idx):
    if it["kind"] == "tgt":
        res = "result missing" if it["y"] is None else ("flagged" if it["y"] else "not flagged")
        return f"{cap(sk['test'])} on {w['names'][it['on']]}, {res}"
    res = "missing" if it["y"] is None else w["parts"]["sig"]["cats"][it["y"]]
    return f"Reading {idx} of {sk['sig']}, {res}"


def render_m2_explicit(w, R, sk, S, hyp):
    """The explicit-factor variant: every result line's likelihood factor, as exact numbers implied by the stated model."""
    m2 = w["parts"]["miss2"]
    names = w["names"]
    out = [T_(R, "m2_x_head", **S)]
    disp = {i: n + 1 for n, i in enumerate(R["ev"])}
    for i in R["ev"]:
        it = w["ev"][i]
        what = m2_line_desc(w, it, sk, disp[i])
        if "regimes" in m2:
            on = it["on"]
            parts = []
            for c, rg in zip("AB", m2["regimes"]):
                rates = dict(rg, mech=m2["mech"])
                alt = (on + 1) % w["k"]
                parts.append(f"under rule {c}, {x4(m2_factor_num(w, it, rates, on))} if {names[on]} {sk['isp']} and "
                             f"{x4(m2_factor_num(w, it, rates, alt))} otherwise")
            out.append(T_(R, "m2_x_reg", what=what, parts="; ".join(parts)))
        elif it["kind"] == "tgt":
            on = it["on"]
            alt = (on + 1) % w["k"]
            out.append(T_(R, "m2_x_line", what=what, name=names[on], a=x4(m2_factor_num(w, it, m2, on)),
                          b=x4(m2_factor_num(w, it, m2, alt)), **S))
        else:
            items = ", ".join(f"{x4(m2_factor_num(w, it, m2, j))} if {names[j]} {sk['isp']}" for j in hyp)
            out.append(T_(R, "m2_x_vec", what=what, items=items))
    out.append(T_(R, "m2_x_tail_reg" if "regimes" in m2 else "m2_x_tail", **S))
    return out


def render_m2(w, R):
    sk = SKINS[w["skin"]]
    fmt = Fmt(R["fmt"]["prob"], R["fmt"]["money"], sk["unit"])
    k, names, hyp = w["k"], w["names"], R["hyp"]
    S = dict(noun=sk["noun"], nouns=sk["nouns"], adj=sk["adj"], isp=sk["isp"])
    lines = [sk["setting"].format(n=k) + " " + sk["one"] + " The possibilities are: " + ", ".join(names[j] for j in hyp) + "."]
    lines.append(T_(R, "prior", items=items_join(f"{names[j]} {fmt.p(w['prior'][j])}" for j in hyp)))
    P = w["parts"]
    m2 = P["miss2"]
    if w["structure"] == "signature":
        lines.append(T_(R, "sig_head", Sig=cap(sk["sig"]), sig=sk["sig"], **S))
        for j in hyp:
            lines.append(T_(R, "sig_row", name=names[j], isp=sk["isp"],
                            items=items_join(f"{c} {fmt.p(v)}" for c, v in zip(P["sig"]["cats"], P["sig"]["table"][j]))))
        if m2["mech"] == "mar":
            lines.append(T_(R, "m2s_mar", sig=sk["sig"], rate=fmt.p(m2["rate"]), **S))
        elif m2["mech"] == "hyp":
            lines.append(T_(R, "m2s_hyp", sig=sk["sig"], name=names[m2["on"]], m1=fmt.p(m2["m1"]), m0=fmt.p(m2["m0"]), **S))
        else:
            items = ", ".join(f"chance {fmt.p(q)} if it reads {c}" for c, q in zip(P["sig"]["cats"], m2["q"]))
            lines.append(T_(R, "m2s_res", sig=sk["sig"], items=items, **S))
        lines.append(T_(R, "m2s_head", sig=sk["sig"], **S))
        for n, i in enumerate(R["ev"]):
            it = w["ev"][i]
            res = "missing" if it["y"] is None else P["sig"]["cats"][it["y"]]
            lines.append(T_(R, "m2s_read", i=n + 1, ordinal=ORDINALS[n], sig=sk["sig"], res=res))
    else:
        if "regimes" in m2:
            lines.append(T_(R, "m2_reg_head", **S))
            for c, rg, p in zip("AB", m2["regimes"], m2["rprior"]):
                if m2["mech"] == "hyp":
                    lines.append(T_(R, "m2_reg_hyp", R=c, p=fmt.p(p), m1=fmt.p(rg["m1"]), m0=fmt.p(rg["m0"]), **S))
                else:
                    lines.append(T_(R, "m2_reg_res", R=c, p=fmt.p(p), qf=fmt.p(rg["qf"]), qc=fmt.p(rg["qc"]), **S))
            lines.append(T_(R, "m2_reg_tail", **S))
        elif m2["mech"] == "mar":
            lines.append(T_(R, "m2_mar", rate=fmt.p(m2["rate"]), **S))
        elif m2["mech"] == "hyp":
            lines.append(T_(R, "m2_hyp", m1=fmt.p(m2["m1"]), m0=fmt.p(m2["m0"]), **S))
        else:
            lines.append(T_(R, "m2_res", qf=fmt.p(m2["qf"]), qc=fmt.p(m2["qc"]), **S))
        lines.append(T_(R, "ev_head", **S))
        for i in R["ev"]:
            it = w["ev"][i]
            tk = "tgt" if it["y"] is not None else "tgt_miss"
            lines.append("- " + T_(R, tk, Test=cap(sk["test"]), test=sk["test"], name=names[it["on"]], sens=fmt.p(it["sens"]),
                                   fpr=fmt.p(it["fpr"]), res=("flagged" if it["y"] else "not flagged"),
                                   resv=("flagged" if it["y"] else "did not flag"), **S))
    if R.get("explicit"):
        lines += render_m2_explicit(w, R, sk, S, hyp)
    return "\n".join(lines)


def render_composed(w, R):
    sk = SKINS[w["skin"]]
    fmt = Fmt(R["fmt"]["prob"], R["fmt"]["money"], sk["unit"])
    k, names, hyp = w["k"], w["names"], R["hyp"]
    S = dict(noun=sk["noun"], nouns=sk["nouns"], adj=sk["adj"], isp=sk["isp"])
    lines = [sk["setting"].format(n=k) + " " + sk["one"] + " The possibilities are: " + ", ".join(names[j] for j in hyp) + "."]
    lines.append(T_(R, "prior", items=items_join(f"{names[j]} {fmt.p(w['prior'][j])}" for j in hyp)))
    b = w["ev"][0]
    ret = b["ret"]
    sub = R["sub"]
    rep_name = lambda i: sk["reporters"][i]                           # noqa: E731
    if b["t"] == "ccr":
        sn = [f"Sensor {j + 1}" for j in b["sens"]]
        order = [j for j in sub if j < len(sn)]
        slist = ", ".join(sn[j] for j in order[:-1]) + " and " + sn[order[-1]]
        if b["stmt"] == "copied":
            lines.append(T_(R, "ccr", slist=slist, pf=fmt.p(b["pf"]), ac=fmt.p(b["ac"]), af=fmt.p(b["af"]), **S))
        else:
            lines.append(T_(R, "cc_ind", slist=slist, am=fmt.p(b["am"]), claims="Each report is listed below", **S))
        if ret["mech"] == "fog":
            lines.append(T_(R, "ccr_fog", mf=fmt.p(ret["mf"]), mc=fmt.p(ret["mc"])))
        else:
            lines.append(T_(R, "ccr_mar", rate=fmt.p(ret["rate"])))
        claims = "; ".join(f"{sn[j]} says {names[b['cs'][j]]}" if b["cs"][j] is not None else f"{sn[j]}: no report received"
                           for j in order)
        lines.append(T_(R, "ccr_obs", claims=claims))
        return "\n".join(lines)
    O = rep_name(b["origin"])
    if b["stmt"] == "copied":
        lines.append(T_(R, "rlr", O=O, O_cap=cap(O), acc=fmt.p(b["acc"]), **S))
    else:
        lines.append(T_(R, "rlr_ind", acc=fmt.p(b["acc"]), **S))
    if ret["mech"] == "claim":
        lines.append(T_(R, "rlr_claim", S=names[ret["S"]], qa=fmt.p(ret["qa"]), qb=fmt.p(ret["qb"]), agent=sk["agent"], **S))
    else:
        lines.append(T_(R, "rlr_mar", rate=fmt.p(ret["rate"])))
    lines.append(T_(R, "ev_plain"))
    for j in [j for j in sub if j < len(b["reps"])]:
        rp = b["reps"][j]
        rn = rep_name(rp["src"])
        res = f"reports {names[rp['c']]}" if rp.get("c") is not None else "no report received"
        if b["stmt"] == "copied":
            lines.append(T_(R, "rlr_rep", R_cap=cap(rn), f=fmt.p(rp["path"][0]), res=res))
        else:
            lines.append(T_(R, "rlr_rep_ind", R_cap=cap(rn), R=rn, res=res))
    return "\n".join(lines)


def skin_words(w):
    fam = w["family"]
    if fam == "hierarchical":
        hs = HIER_SKINS[w["skin"]]
        return dict(noun=hs["member"].lower(), isp=f"is {hs['bad']}")
    if fam == "multimodal":
        ms = MM_SKINS[w["skin"]]
        nn = {"ring_distance": ms["pos"].lower(), "grid_distance": ms["gpos"].lower(),
              "unidentified_group": "hut" if w["skin"] == "huts" else "room"}[w["structure"]]
        tg = {"ring_distance": ms["target"], "grid_distance": ms["gtarget"], "unidentified_group": ms["grtarget"]}
        return dict(noun=nn, isp=f"has {tg[w['structure']]}")
    sk = SKINS[w["skin"]]
    return dict(noun=sk["noun"], isp=sk["isp"], adj=sk["adj"], nouns=sk["nouns"])


def render_question(w, qk, R):
    """(instructions, [(canonical option index, text)] in display order)."""
    fmt_unit = SKINS[w["skin"]]["unit"] if w["skin"] in SKINS else "usd"
    fmt = Fmt(R["fmt"]["prob"], R["fmt"]["money"], fmt_unit)
    k, names = w["k"], w["names"]
    S = skin_words(w)
    if qk == "posterior":
        if w["family"] == "hierarchical":
            hs = HIER_SKINS[w["skin"]]
            member = f"{hs['member']} {w['parts']['target'] + 1}"
            instr = T_(R, "q_hpost", member=member, thing=hs["thing"], bad=hs["bad"], good=hs["good"])
            texts = [f"{member}'s {hs['thing']} is {hs['bad']}", f"{member}'s {hs['thing']} is {hs['good']}"]
            order = R["hyp"]
            return instr, [(j, texts[j]) for j in order]
        instr = T_(R, "q_post", noun=S["noun"], isp=S["isp"])
        return instr, [(j, names[j]) for j in R["hyp"]]
    if qk == "latent":
        if w["family"] == "hierarchical":
            hs = HIER_SKINS[w["skin"]]
            instr = T_(R, "q_hlat", snoun=hs["snoun"])
            texts = [f"The {hs['snoun']} is {g}" for g in w["parts"]["grades"]]
        elif w["latq"]["axis"] == "reg":                               # v1.2 missingness2 regime: which recording rule
            instr = T_(R, "q_reg")
            texts = list(latent_levels(w))
        else:
            rn = SKINS[w["skin"]]["reporters"][w["latq"]["src"]]
            instr = T_(R, "q_lat", src=rn)
            rvd = w["parts"]["rv"][w["latq"]["axis"][3:]]
            texts = [f"{cap(n)} (right with chance {fmt.p(a)})" for n, a in zip(LEVEL_NAMES[len(rvd["levels"])], rvd["levels"])]
        perm = opt_perm(R, qk, len(texts))
        return instr, [(j, texts[j]) for j in perm]
    if qk == "predictive":
        desc, texts = next_desc(w, fmt)
        instr = T_(R, "q_pred", desc=desc)
        perm = opt_perm(R, qk, len(texts))
        return instr, [(j, texts[j]) for j in perm]
    sk = SKINS[w["skin"]]
    nA = k + (1 if w["loss"]["E"] is not None else 0)
    if qk in VALUE_KINDS:
        instr, texts = value_question(w, qk, R, fmt, sk, nA)
        perm = opt_perm(R, qk, len(texts))
        return instr, [(j, texts[j]) for j in perm]
    if qk == "action":
        texts = [action_text(w, a, sk) for a in range(nA)]
        instr = T_(R, "q_act")
    else:
        texts = [action_text(w, a, sk, now=True) for a in range(nA)]
        for opt in w["info"]:
            texts.append(info_option_text(w, opt, sk, fmt))
        instr = T_(R, "q_acq", agent=sk["agent"])
    perm = opt_perm(R, qk, len(texts))
    return instr, [(j, texts[j]) for j in perm]


def value_question(w, qk, R, fmt, sk, nA):
    """(instructions, bin texts in canonical bin order) of a value question; the bins are printed from value_edges."""
    acq = w["info"] is not None
    edges = value_edges(w, qk)
    M = lambda v: fmt.mv(v, fee=(qk == "value_voi"))                  # noqa: E731
    mid = [f"{M(a)} to less than {M(b)}" for a, b in zip(edges, edges[1:])]
    if qk == "value_regret":
        best = ("the best available choice (acting now, or first paying for information as described above)" if acq
                else "the best available action")
        instr = T_(R, "q_regret", act=action_text(w, w["vq"]["regret"], sk, now=acq), best=best)
        texts = ["0 (it is one of the best choices)" if acq else "0 (it is one of the best actions)",
                 f"more than {M(0)} but less than {M(edges[0])}"] + mid + [f"{M(edges[-1])} or more"]
    elif qk == "value_gap":
        note = " A choice is acting now, or first paying for information as described above." if acq else ""
        instr = T_(R, "q_gap", nc="choice" if acq else "action", ncs="choices" if acq else "actions", note=note)
        texts = [f"less than {M(edges[0])}"] + mid + [f"{M(edges[-1])} or more"]
    else:
        opt = w["info"][w["vq"]["voi"]]
        f = option_fee(opt)
        instr = T_(R, "q_voi", opt=info_option_text(w, opt, sk, fmt))
        texts = [f"a net loss of exactly the fee, {M(f)}: no result could change the decision",
                 f"a net loss of more than {M(f / 2)} but less than {M(f)}", f"a net loss of at most {M(f / 2)}",
                 f"a net saving of less than {M(f)}", f"a net saving of {M(f)} to less than {M(3 * f)}",
                 f"a net saving of {M(3 * f)} or more"]
    assert len(texts) == n_value_bins(w, qk) and len(set(texts)) == len(texts)
    return instr, texts


def info_option_text(w, opt, sk, fmt):
    names = w["names"]
    t = opt["t"]
    if t == "test":
        return f"Run {sk['test']} on {names[opt['on']]} first (fee {fmt.m(opt['fee'], True)}), then decide"
    if t == "consult":
        return f"{sk['consult']} first (fee {fmt.m(opt['fee'], True)}), then decide"
    if t == "collect":
        n = opt["n"]
        if n == 1:
            return f"Take 1 more reading first (fee {fmt.m(opt['fee'], True)}), then decide"
        return f"Take {n} more readings first (fee {fmt.m(opt['fee'], True)} each, {fmt.m(n * opt['fee'], True)} in total), then decide"
    if t == "two":
        a = opt["tests"][opt["first"]]
        return (f"Start with {sk['test']} on {names[a['on']]} (fee {fmt.m(a['fee'], True)}), then decide whether to run "
                f"the other test")
    raise ValueError(t)


def next_desc(w, fmt):
    nx, k, names = w["next"], w["k"], w["names"]
    t = nx["t"]
    flags = ["It flags", "It does not flag"]
    if w["family"] == "hierarchical":
        hs = HIER_SKINS[w["skin"]]
        member = f"{hs['member']} {nx['g'] + 1}"
        return (f"{cap(hs['check'])} (it flags with chance {fmt.p(nx['sens'])} if the {hs['thing']} is {hs['bad']}, and with "
                f"chance {fmt.p(nx['fpr'])} if it is {hs['good']}) is about to be run on {member}."), flags
    if w["family"] == "multimodal":
        ms = MM_SKINS[w["skin"]]
        if t == "dist":
            sensor = ms["sensor"] if w["structure"] == "ring_distance" else ms["gsensor"]
            texts = [f"A reading of {r}" for r in range(w["parts"]["dmax"] + 1)]
            return f"The {sensor} is about to be read once more, this time at {names[nx['at']]}.", texts
        gn = w["parts"]["gnames"][nx["g"]]
        return (f"{cap(ms['gtest'])} (flags with chance {fmt.p(nx['sens'])} if {ms['grtarget']} is in that {ms['gword']}, "
                f"{fmt.p(nx['fpr'])} if not) is about to be run on the {gn} {ms['gword']}."), flags
    sk = SKINS[w["skin"]]
    if t == "tgt":
        return (f"One more check, {sk['test']} (it flags with chance {fmt.p(nx['sens'])} if {names[nx['on']]} {sk['isp']}, "
                f"and with chance {fmt.p(nx['fpr'])} if not), is about to be run on {names[nx['on']]}."), flags
    if t == "sig":
        return f"One more reading of {sk['sig']} is about to be taken.", list(w["parts"]["sig"]["cats"])
    if t == "feat":
        feat = sk["features"][nx["f"]]
        items = items_join(f"{fmt.p(nx['p'][j])} if {names[j]} {sk['isp']}" for j in range(k))
        return f"Someone is about to check for {feat}, which appears with chance {items}.", ["Present", "Absent"]
    if t == "rep_new":
        return (f"A new source, who checks independently of everyone above and is right with chance {fmt.p(nx['acc'])} "
                f"(otherwise naming one of the other {sk['nouns']} at random), is about to name the {sk['adj']} "
                f"{sk['noun']}."), list(names)
    if t == "right_next":
        rn = sk["reporters"][nx["src"]]
        return (f"{cap(rn)} is about to make another claim that can be checked later."), ["It will be right",
                                                                                           "It will be wrong"]
    raise ValueError(t)


# --------------------------------------------------------------------------- units (records)
def world_fingerprint(w):
    v = visible(w)
    for kk in ("skin", "names"):
        v.pop(kk, None)
    return sha16(json.dumps(v, sort_keys=True))


def make_record(w, qk, R, ctx):
    """One record: state + ONE choice question + exact soft target; the world spec and truth in meta (hidden)."""
    vis = visible(w)
    state = render_state(vis, R)
    instr, opts = render_question(vis, qk, R)
    tgt, extra = exact_target(vis, qk)
    assert abs(tgt.sum() - 1.0) < 1e-12 and (tgt >= 0).all()
    canon = [j for j, _ in opts]
    assert sorted(canon) == list(range(len(tgt)))
    keys = [f"{qk[0]}{j}" for j in canon]
    qid = f"{SET}/{qk}/{sha16(ctx['rid_seed'])}"
    q = {"type": "choice", "instructions": instr, "criteria": {kk: t for kk, (_, t) in zip(keys, opts)}}
    lab = lab_dict(keys, [tgt[j] for j in canon])
    kind = "hard" if sum(1 for v in lab.values() if v > 0) == 1 else "soft"
    truth = dict(w["truth"])
    meta = {
        "set": SET, "generator": gen_version(), "seed_namespace": ctx["ns"], "split": ctx["split"],
        "family": w["family"], "structure": w["structure"], "heldout_structure": (w["family"], w["structure"]) in heldout_set(),
        "skin": w["skin"], "group_id": ctx["group_id"], "world_id": ctx["world_id"], "world_role": ctx["role"],
        "role_info": ctx.get("role_info"), "render_variant": ctx["rvar"], "qkind": qk, "target_kind": TARGET_KIND[qk],
        "option_canon": canon, "exact_target": [float(x) for x in tgt], "exact_extra": extra,
        "posterior_h": [float(x) for x in posterior_h(vis)],
        "truth": truth, "truth_from_stated_model": bool(w["from_model"]),
        "spec": vis, "render": R, "fingerprint": world_fingerprint(w),
        "label_source": {qid: f"exact:{qk}"}, "eval_only": False, "td_workflow": None,
        "exclusion_index": "synthetic: no id / state can match rd_ / hl_exclusion_index (generated text only)",
    }
    if ver() == "v1.1":
        meta["difficulty"] = difficulty(vis)
        meta["wording"] = "novel" if ctx["split"] == "test_novel_wording" else "standard"
    elif ver() == "v1.2":
        meta["difficulty"] = difficulty(vis)
        meta["wording"] = ctx["wording"]
        meta["rendering_index"] = ctx.get("rendering", 1)               # train: 1..K equivalent renderings (1 = train_plain)
        meta["explicit_factors"] = bool(R.get("explicit"))
        mp = missing_pattern(vis)
        if mp is not None:
            meta["missing_pattern"] = mp
    rec = {"id": f"{SET}-{ctx['split']}-{sha16(ctx['rid_seed'])}", "source": f"{SET}/{w['family']}", "license": LICENSE,
           "base_id": f"{SET}/{ctx['group_id']}", "state": state, "questions": {qid: q}, "labels": {qid: lab},
           "label_kind": {qid: kind}, "meta": meta}
    return rec


def draw_plan(pr, w):
    fam, st = w["family"], w["structure"]
    plan = {"predictive": w["next"] is not None and pr.random() < 0.35,
            "latent": w["latq"] is not None and pr.random() < 0.5,
            "action": fam == "decision", "acquire": fam == "acquisition"}
    plan["prior_swap"] = fam != "acquisition" and pr.random() < 0.22
    nev = len(w["ev"])
    plan["rounds"] = []
    if fam in ("diagnosis", "multimodal") and nev >= 2 and pr.random() < 0.35:
        cand = list(range(0, nev))
        m = 1 if pr.random() < 0.6 else 2
        plan["rounds"] = sorted(int(x) for x in pr.choice(cand, size=min(m, len(cand)), replace=False))
    plan["dep_swap"] = fam == "copied" and st in ("shared_upstream", "noisy_copy", "common_cause") and pr.random() < 0.6
    plan["mech_swap"] = fam == "missingness" and pr.random() < 0.6
    plan["loss_swap"] = fam in ("decision", "acquisition") and pr.random() < 0.4
    for v in ("order", "units", "paraphrase"):
        plan[v] = pr.random() < 0.18
    return plan


def draw_plan11(pr, w):
    """v1.1 value questions (own stream): which action / information option is NAMED, drawn uniformly, before solving."""
    if w["loss"] is None:
        return {}
    nA = w["k"] + (1 if w["loss"]["E"] is not None else 0)
    if w["family"] == "decision":
        return {"value_regret": int(pr.integers(nA)), "value_gap": bool(pr.random() < 0.7)}
    out = {"value_voi": int(pr.integers(len(w["info"])))}
    out["value_regret"] = int(pr.integers(nA)) if pr.random() < 0.5 else None
    out["value_gap"] = bool(pr.random() < 0.4)
    return out


def sibling(w, role, sr, extra=None):
    """A linked world: the same evidence with one intervention (drawn from its own stream `sr`)."""
    w2 = copy.deepcopy(w)
    info = {}
    if role == "prior_swap":
        if w["family"] == "hierarchical":
            w2["parts"]["sprior"] = draw_prior(sr, w["parts"]["L"])
            info = {"old": w["parts"]["sprior"], "new": w2["parts"]["sprior"]}
        else:
            w2["prior"] = draw_prior(sr, w["k"])
            info = {"old": w["prior"], "new": w2["prior"]}
        w2["from_model"] = False
    elif role == "round":
        r = extra
        w2["ev"] = w["ev"][:r]
        info = {"n_items": r, "of": len(w["ev"])}
    elif role == "dep_swap":
        for b in w2["ev"]:
            if b["t"] in ("relay", "copy", "cc", "rlr") and b["stmt"] == "copied":
                b["stmt"] = "independent"
                if b["t"] == "cc":
                    b["am"] = int(round((1 - b["pf"] / 100.0) * b["ac"] + (b["pf"] / 100.0) * b["af"]))
        w2["from_model"] = False
        info = {"statement": "independent"}
    elif role == "rel_swap":                                           # v1.2: the same reports, a different track record
        i = rel_swap_item(w)
        it = w2["ev"][i]
        n, s = it["n"], it["s"]
        if n >= 1:
            r = ri(sr, 0, n - 1)                                       # uniform over {0..n} minus the observed s
            n2, s2 = n, (r if r < s else r + 1)
        else:
            n2 = ri(sr, 1, 12)
            s2 = ri(sr, 0, n2)
        it["n"], it["s"] = n2, s2
        w2["from_model"] = False
        info = {"item": i, "old": [n, s], "new": [n2, s2]}
    elif role == "mech2_swap":                                         # v1.2: the same observed data, another recording mechanism
        to = extra
        if w["family"] == "missingness2":
            old = w["parts"]["miss2"]
            if w["structure"] == "signature":
                new = draw_m2_sig_rates(sr, to, w["k"], len(w["parts"]["sig"]["cats"]))
            elif w["structure"] == "regime" and to != "mar":
                new = draw_m2_regime(sr, to)
            else:
                new = draw_m2_rates(sr, to)
            w2["parts"]["miss2"] = new
            if "regimes" not in new:
                w2["latq"] = None
            info = {"from": old["mech"] + ("_regime" if "regimes" in old else ""),
                    "to": new["mech"] + ("_regime" if "regimes" in new else "")}
        else:                                                          # composed: the retention rule becomes MAR
            b = w2["ev"][0]
            info = {"from": b["ret"]["mech"], "to": "mar"}
            b["ret"] = {"mech": "mar", "rate": ri(sr, 15, 50)}
        w2["from_model"] = False
    elif role == "mech_swap":
        miss = w["parts"]["miss"]
        if miss["mech"] == "informative":
            w2["parts"]["miss"] = {"mech": "mar", "rate": ri(sr, 15, 50)}
        elif miss["mech"] == "mnar":
            w2["parts"]["miss"] = dict(miss, mech="mnar_mar")
        else:
            hi_, lo_ = ri(sr, 60, 100), ri(sr, 5, 50)
            w2["parts"]["miss"] = {"mech": "informative", "qf": hi_, "qc": lo_}
        w2["from_model"] = False
        info = {"from": miss["mech"], "to": w2["parts"]["miss"]["mech"]}
    elif role == "loss_swap":
        sk = SKINS[w["skin"]]
        form = w["loss"]["form"]
        w2["loss"] = draw_loss(sr, w["k"], form, sk["unit"], w["loss"]["E"] is not None)
        if w["structure"] == "asymmetric":
            j = ri(sr, 0, w["k"] - 1)
            w2["loss"]["M"][j] = int(w2["loss"]["M"][j] * ri(sr, 3, 12))
        if w["info"] is not None:
            S, st = w2["loss"]["scale"], (1 if sk["unit"] == "points" else 10)
            for opt in w2["info"]:
                if "fee" in opt:
                    opt["fee"] = money(sr, max(st, 0.005 * S), max(2 * st, 0.3 * S), st)
                for t in opt.get("tests", []):
                    t["fee"] = money(sr, max(st, 0.005 * S), max(2 * st, 0.3 * S), st)
            if w2["info"] and w2["info"][0]["t"] == "collect":
                f = w2["info"][0]["fee"]
                for opt in w2["info"]:
                    opt["fee"] = f
            if w2["info"] and w2["info"][0]["t"] == "two":
                w2["info"][1]["tests"] = w2["info"][0]["tests"]
        info = {"loss_scale_old": w["loss"]["scale"], "loss_scale_new": w2["loss"]["scale"]}
    return w2, info


def rerender(R, kind, rr, w, mode="v1"):
    """An invariance re-rendering of the SAME world: only one aspect of the render config changes."""
    R2 = copy.deepcopy(R)
    if kind == "order":
        R3 = draw_render(rr, w)
        R2["hyp"], R2["ev"], R2["opt_seed"] = R3["hyp"], R3["ev"], R3["opt_seed"]
    elif kind == "units":
        probs = [f for f in PROB_FMTS if f != R["fmt"]["prob"]]
        monies = [f for f in MONEY_FMTS if f != R["fmt"]["money"]]
        R2["fmt"] = {"prob": probs[int(rr.integers(len(probs)))], "money": monies[int(rr.integers(len(monies)))]}
    elif kind == "paraphrase":
        for _ in range(20):
            t = draw_tpl(rr, mode)
            if t != R["tpl"]:
                break
        R2["tpl"] = t
    return R2


def make_group(split, idx, ns, family, structure):
    """All records of one group (anchor world + linked worlds + re-renderings), deterministic in (ns, idx)."""
    gid = f"{split}/{family}/{structure}/g{idx:06d}"
    v11 = ver() == "v1.1"
    novel = v11 and split == "test_novel_wording"
    w = sample_world(family, structure, rng_of(ns, "world", idx), novel=novel)
    pr = rng_of(ns, "plan", idx)
    plan = draw_plan(pr, w)
    p11 = draw_plan11(rng_of(ns, "plan11", idx), w) if v11 else {}
    if p11:
        w["vq"] = {"regret": p11.get("value_regret"), "voi": p11.get("value_voi")}
    if novel:
        plan["paraphrase"] = False                     # every template is already the reserved variant
    mode = "v1" if not v11 else ("novel" if novel else "base11")
    R0 = draw_render(rng_of(ns, "render", idx, "base"), visible(w), mode)
    recs = []

    def emit(wx, qk, R, role, rvar, wid, rinfo=None):
        ctx = {"ns": ns, "split": split, "group_id": gid, "world_id": f"{gid}/{wid}", "role": role, "rvar": rvar,
               "role_info": rinfo, "rid_seed": f"{ns}|{gid}|{wid}|{qk}|{rvar}"}
        recs.append(make_record(wx, qk, R, ctx))

    anchor_q = ["posterior"]
    for qk in ("latent", "predictive", "action", "acquire"):
        if plan.get(qk):
            anchor_q.append(qk)
    if p11.get("value_regret") is not None:
        anchor_q.append("value_regret")
    if p11.get("value_gap"):
        anchor_q.append("value_gap")
    if p11.get("value_voi") is not None:
        anchor_q.append("value_voi")
    for qk in anchor_q:
        emit(w, qk, R0, "anchor", "base", "w0")
    sr = rng_of(ns, "sibling", idx)
    wn = 1
    if plan["prior_swap"]:
        w2, info = sibling(w, "prior_swap", sr)
        emit(w2, "posterior", R0, "prior_swap", "base", f"w{wn}", info)
        wn += 1
    for r in plan["rounds"]:
        w2, info = sibling(w, "round", sr, r)
        R2 = copy.deepcopy(R0)
        R2["ev"] = [i for i in R0["ev"] if i < r]
        emit(w2, "posterior", R2, "round", "base", f"w{wn}", info)
        wn += 1
    if plan["dep_swap"]:
        w2, info = sibling(w, "dep_swap", sr)
        emit(w2, "posterior", R0, "dep_swap", "base", f"w{wn}", info)
        wn += 1
    if plan["mech_swap"]:
        w2, info = sibling(w, "mech_swap", sr)
        emit(w2, "posterior", R0, "mech_swap", "base", f"w{wn}", info)
        wn += 1
    if plan["loss_swap"]:
        w2, info = sibling(w, "loss_swap", sr)
        qd = "action" if w["family"] == "decision" else "acquire"
        emit(w2, "posterior", R0, "loss_swap", "base", f"w{wn}", info)
        emit(w2, qd, R0, "loss_swap", "base", f"w{wn}", info)
        wn += 1
    for kind in ("order", "units", "paraphrase"):
        if plan[kind]:
            R2 = rerender(R0, kind, rng_of(ns, "render", idx, kind), visible(w), mode)
            for qk in anchor_q:
                if qk in ("posterior", "action", "acquire") or qk in VALUE_KINDS:
                    emit(w, qk, R2, "anchor", kind, "w0")
    return recs


# --------------------------------------------------------------------------- v1.2 groups: K renderings, forced contrasts, missingness2
def rel_swap_item(w):
    """The track item a rel_swap changes (the first, by evidence index); None if the world has no track record item."""
    for i, it in enumerate(w["ev"]):
        if it["t"] == "track":
            return i
    return None


def missing_pattern(w):
    """(v1.2 meta) how many results / reports are missing, for the per-stratum mechanism readouts; None if not applicable."""
    fam = w["family"]
    if fam == "missingness":
        ys = [it.get("y") for it in w["ev"]]
    elif fam == "missingness2":
        ys = [it.get("y") for it in w["ev"]]
    elif fam == "composed":
        b = w["ev"][0]
        ys = b["cs"] if b["t"] == "ccr" else [rp.get("c") for rp in b["reps"]]
    else:
        return None
    return {"n_missing": sum(1 for y in ys if y is None), "n_items": len(ys)}


def world_mech(w):
    """The recording mechanism of a missingness / missingness2 / composed world (for strata): mar / hyp / result / ..."""
    fam = w["family"]
    if fam == "missingness":
        return {"mar": "mar", "informative": "result", "mnar": "hyp", "mnar_mar": "mar"}[w["parts"]["miss"]["mech"]]
    if fam == "missingness2":
        return w["parts"]["miss2"]["mech"]
    if fam == "composed":
        return {"fog": "hyp_latent", "claim": "result", "mar": "mar"}[w["ev"][0]["ret"]["mech"]]
    return None


def draw_plan12(pr, w, kind):
    """v1.2 plan. kind: 'train' (every applicable intervention sibling FORCED; single-aspect re-renderings replaced by the K
    equivalent renderings), 'eval' (v1.1 plan + the missingness2 siblings / rel_swap / explicit-factor rendering) or 'confirm'
    (forced siblings + the v1.1 re-rendering plan)."""
    plan = draw_plan(pr, w)
    fam, st = w["family"], w["structure"]
    if kind in ("train", "confirm"):                                 # decision / acquisition: the forced contrast is the loss swap
        plan["prior_swap"] = plan["prior_swap"] if fam == "decision" else fam not in ("acquisition", "composed")
        plan["loss_swap"] = fam in ("decision", "acquisition")
        plan["dep_swap"] = (fam == "copied" and st in ("shared_upstream", "noisy_copy", "common_cause")) or \
            (fam == "composed" and st == "relay_retained")
        plan["mech_swap"] = fam == "missingness"
    plan["rel_swap"] = fam == "reliability" and rel_swap_item(w) is not None
    plan["mech2"] = []
    if fam == "missingness2":
        cur = w["parts"]["miss2"]["mech"]
        pool = ("mar", "hyp", "result")
        plan["mech2"] = [m for m in pool if m != cur]
    elif fam == "composed":
        plan["mech2"] = ["mar"]
    plan["explicit"] = fam == "missingness2" and kind != "train"
    if kind == "train":
        plan["order"] = plan["units"] = plan["paraphrase"] = False
    return plan


def draw_render12(rr, w, wording, explicit=False):
    """A v1.2 render config: the v1.1 draw, plus the v1.2 template keys (from a stream derived from opt_seed, so the v1.1 draw
    sequence is unchanged), the explicit-factor flag and (composed) a sub-order of the sensors / relays."""
    R = draw_render(rr, w, "base11" if wording == "standard" else "novel")
    sub = np.random.default_rng([R["opt_seed"], 12])
    if wording == "standard":
        R["tpl"].update({key: int(sub.integers(2)) for key in TPL12_KEYS})
    elif wording == "novel":
        R["tpl"].update({key: 2 for key in TPL12_KEYS})
    else:                                                              # confirm: the held-out variant 3 of every key
        R["tpl"] = {key: 3 for key in list(R["tpl"]) + TPL12_KEYS}
    R["explicit"] = bool(explicit)
    if w["family"] == "composed":
        n = len(w["ev"][0]["sens"]) if w["ev"][0]["t"] == "ccr" else len(w["ev"][0]["reps"])
        R["sub"] = [int(i) for i in np.random.default_rng([R["opt_seed"], 77]).permutation(n)]
    return R


def rerender12(R, kind, rr, w, wording):
    """A v1.2 single-aspect re-rendering (eval / confirm splits): order, units, paraphrase (confirm: variants 2 / 3 mixed), or
    explicit (the likelihood factors stated)."""
    if kind == "explicit":
        R2 = copy.deepcopy(R)
        R2["explicit"] = True
        return R2
    if kind in ("order", "units"):
        R2 = rerender(R, kind, rr, w, "base11")
        if kind == "order" and "sub" in R:
            n = len(R["sub"])
            R2["sub"] = [int(i) for i in np.random.default_rng([R2["opt_seed"], 77]).permutation(n)]
        return R2
    R2 = copy.deepcopy(R)
    for _ in range(50):
        if wording == "confirm":
            t = {key: (2 if rr.random() < 0.5 else 3) for key in R["tpl"]}
        else:
            t = draw_tpl(rr, "base11")
            t.update({key: int(rr.integers(2)) for key in TPL12_KEYS})
        if t != R["tpl"]:
            break
    R2["tpl"] = t
    return R2


def compatible_skins(w, pool):
    """Skins a world can be re-rendered in without changing its meaning: same money unit, and an alternative-supplier phrase
    when the loss form needs one; hierarchical / multimodal keep to their own skin sets."""
    fam = w["family"]
    if fam == "hierarchical":
        return sorted(HIER_SKINS)
    if fam == "multimodal":
        return sorted(MM_SKINS)
    unit = SKINS[w["skin"]]["unit"]
    need_rely = w.get("loss") is not None and w["loss"]["form"] == "avoid"
    return [s for s in pool if SKINS[s]["unit"] == unit and (SKINS[s]["rely"] or not need_rely)]


def reskin(w, skin):
    """The same world told in another skin: names, categories, grades follow the skin; every number is unchanged (the solver
    never reads skin text, so every exact target is unchanged; tested)."""
    w2 = copy.deepcopy(w)
    if skin == w["skin"]:
        return w2
    fam = w["family"]
    P = w2["parts"]
    w2["skin"] = skin
    if fam == "hierarchical":
        hs = HIER_SKINS[skin]
        w2["names"] = [hs["bad"], hs["good"]]
        P["grades"] = hs["grades2"] if P["L"] == 2 else hs["grades3"]
        return w2
    if fam == "multimodal":
        ms = MM_SKINS[skin]
        if w["structure"] == "ring_distance":
            w2["names"] = [f"{ms['pos']} {i + 1}" for i in range(w["k"])]
        elif w["structure"] == "grid_distance":
            w2["names"] = [f"{ms['gpos']} {'ABC'[r]}{c + 1}" for r in range(P["R"]) for c in range(P["C"])]
        else:
            w2["names"] = [f"{'Hut' if skin == 'huts' else 'Room'} {i + 1}" for i in range(w["k"])]
        return w2
    old, new = SKINS[w["skin"]], SKINS[skin]
    w2["names"] = [new["names"][old["names"].index(n)] for n in w["names"]]
    if P.get("sig"):
        P["sig"]["cats"] = new["cats"][:len(P["sig"]["cats"])]
    for opt in (w2.get("info") or []):
        if opt.get("cats"):
            opt["cats"] = new["ccats"][:len(opt["cats"])]
    return w2


def sample_world12(family, structure, rng, pool):
    """v1.2 sampler: the v1.1 samplers (fix-form skins from `pool`), missingness2 and the composed worlds."""
    if family in ("hierarchical", "multimodal"):
        return sample_world(family, structure, rng)
    pool2 = pool if not (family == "decision" and structure == "avoid") else [s for s in pool if SKINS[s]["rely"]]
    skin = pool2[ri(rng, 0, len(pool2) - 1)]
    fn = {"diagnosis": sample_diagnosis, "copied": sample_copied, "reliability": sample_reliability,
          "missingness": sample_missingness, "decision": sample_decision, "acquisition": sample_acquisition,
          "missingness2": sample_missingness2, "composed": sample_composed}[family]
    return fn(rng, structure, skin)


SKIN_POOL = {"standard": SKIN_ORDER, "novel": NOVEL_SKIN_ORDER, "confirm": CONFIRM_SKIN_ORDER}


def make_group12(split, idx, ns, family, structure, kind, wording="standard", K=1):
    """All records of one v1.2 group. Train: K equivalent renderings (rendering 1 = the world as sampled; 2..K re-draw the
    hypothesis / evidence / option orders, the number formats, the templates and a semantics-preserving skin), EACH carrying the
    anchor's questions AND every intervention sibling, so the group holds both 'equivalent text -> same answer' and 'changed
    mechanism -> changed answer'. Eval / confirm: one rendering + the single-aspect re-renderings (order / units / paraphrase /
    explicit factors) as in v1.1."""
    gid = f"{split}/v12/{family}/{structure}/g{idx:06d}"
    w = sample_world12(family, structure, rng_of(ns, "world", idx), SKIN_POOL[wording])
    plan = draw_plan12(rng_of(ns, "plan", idx), w, kind)
    p11 = draw_plan11(rng_of(ns, "plan11", idx), w)
    if p11:
        w["vq"] = {"regret": p11.get("value_regret"), "voi": p11.get("value_voi")}
    if wording == "novel":
        plan["paraphrase"] = False
    vis = visible(w)

    x_group = bool(rng_of(ns, "xflag", idx).random() < 0.25)           # (105b) one draw per group, used iff HINT_PER_GROUP

    def x_flag(R):                                                     # train: 1 in 4 missingness2 renderings states the factors
        if family != "missingness2" or kind != "train":
            return False
        if HINT_PER_GROUP:
            return x_group
        return bool(np.random.default_rng([R["opt_seed"], 13]).random() < 0.25)
    R1 = draw_render12(rng_of(ns, "render", idx, "base"), vis, wording)
    R1["explicit"] = x_flag(R1)
    renders = [(None, R1, "base", 1)]
    for r in range(2, K + 1):
        pool = compatible_skins(vis, SKIN_POOL[wording])
        skin = pool[int(rng_of(ns, "skin", idx, r).integers(len(pool)))]
        Rr = draw_render12(rng_of(ns, "render", idx, f"multi{r - 1}"), reskin(vis, skin), wording)
        Rr["explicit"] = x_flag(Rr)
        renders.append((skin, Rr, f"multi{r - 1}", r))
    sibs = []                                                          # (role, world, info, rounds-prefix or None)
    sr = rng_of(ns, "sibling", idx)
    if plan["prior_swap"]:
        sibs.append(("prior_swap",) + sibling(w, "prior_swap", sr) + (None,))
    for r_ in plan["rounds"]:
        sibs.append(("round",) + sibling(w, "round", sr, r_) + (r_,))
    if plan["dep_swap"]:
        sibs.append(("dep_swap",) + sibling(w, "dep_swap", sr) + (None,))
    if plan["mech_swap"]:
        sibs.append(("mech_swap",) + sibling(w, "mech_swap", sr) + (None,))
    if plan["rel_swap"]:
        sibs.append(("rel_swap",) + sibling(w, "rel_swap", sr) + (None,))
    for to in plan["mech2"]:
        sibs.append(("mech2_swap",) + sibling(w, "mech2_swap", sr, to) + (None,))
    if plan["loss_swap"]:
        sibs.append(("loss_swap",) + sibling(w, "loss_swap", sr) + (None,))
    anchor_q = ["posterior"]
    for qk in ("latent", "predictive", "action", "acquire"):
        if plan.get(qk):
            anchor_q.append(qk)
    if p11.get("value_regret") is not None:
        anchor_q.append("value_regret")
    if p11.get("value_gap"):
        anchor_q.append("value_gap")
    if p11.get("value_voi") is not None:
        anchor_q.append("value_voi")
    recs = []

    def emit(wx, qk, R, role, rvar, wid, rinfo, rendering):
        ctx = {"ns": ns, "split": split, "group_id": gid, "world_id": f"{gid}/{wid}", "role": role, "rvar": rvar,
               "role_info": rinfo, "rid_seed": f"{ns}|{gid}|{wid}|{qk}|{rvar}", "wording": wording, "rendering": rendering}
        recs.append(make_record(wx, qk, R, ctx))

    for skin, R, rvar, rnum in renders:
        wa = w if skin is None else dict(reskin(vis, skin), truth=w["truth"], from_model=w["from_model"])
        for qk in anchor_q:
            emit(wa, qk, R, "anchor", rvar, "w0", None, rnum)
        for n, (role, w2, info, rpre) in enumerate(sibs):
            wx = w2 if skin is None else dict(reskin(visible(w2), skin), truth=w2["truth"], from_model=w2["from_model"])
            Rx = R
            if role == "round":
                Rx = copy.deepcopy(R)
                Rx["ev"] = [i for i in R["ev"] if i < rpre]
            emit(wx, "posterior", Rx, role, rvar, f"w{n + 1}", info, rnum)
            if role == "loss_swap":
                emit(wx, "action" if w["family"] == "decision" else "acquire", Rx, role, rvar, f"w{n + 1}", info, rnum)
    if kind != "train":                                                # single-aspect re-renderings of rendering 1
        for rk in ("order", "units", "paraphrase"):
            if plan[rk]:
                R2 = rerender12(R1, rk, rng_of(ns, "render", idx, rk), vis, wording)
                for qk in anchor_q:
                    if qk in ("posterior", "action", "acquire") or qk in VALUE_KINDS:
                        emit(w, qk, R2, "anchor", rk, "w0", None, 1)
        if plan["explicit"]:                                           # the likelihood factors stated: anchor + mechanism siblings
            R2 = rerender12(R1, "explicit", None, vis, wording)
            emit(w, "posterior", R2, "anchor", "explicit", "w0", None, 1)
            for n, (role, w2, info, rpre) in enumerate(sibs):
                if role == "mech2_swap":
                    emit(w2, "posterior", R2, role, "explicit", f"w{n + 1}", info, 1)
    return recs


# --------------------------------------------------------------------------- build
def structure_schedule(split, idx, ns):
    r = rng_of(ns, "structure", idx)
    if split == "test_heldout_structure":
        f, s = HELDOUT[int(r.integers(len(HELDOUT)))]
        return f, s
    FW = FAMILY_WEIGHTS if ver() == "v1" else FAMILY_WEIGHTS_V11
    fams = sorted(FW)
    p = np.array([FW[f] for f in fams])
    f = fams[int(r.choice(len(fams), p=p / p.sum()))]
    sts = sorted(FAMILIES[f]["train"])
    pw = np.array([FAMILIES[f]["train"][s] for s in sts], float)
    return f, sts[int(r.choice(len(sts), p=pw / pw.sum()))]


def norm_state(s):
    s = re.sub(r"\s+", " ", str(s)).strip()
    if len(s) > 1 and s[0] == s[-1] == '"':
        s = s[1:-1]
    return s.strip().lower()


def gen_split(split, n_units, ns, forbid_fp=None, forbid_state=None, log=print):
    recs, idx, stats = [], 0, Counter()
    while len(recs) < n_units:
        f, s = structure_schedule(split, idx, ns)
        grp = make_group(split, idx, ns, f, s)
        idx += 1
        if forbid_fp is not None:
            if any(r["meta"]["fingerprint"] in forbid_fp for r in grp):
                stats["groups_dropped_world_shared_with_eval"] += 1
                continue
            if any(norm_state(r["state"]) in forbid_state for r in grp):
                stats["groups_dropped_state_shared_with_eval"] += 1
                continue
        recs += grp
        stats["groups"] += 1
        if stats["groups"] % 2000 == 0:
            log(f"  {split}: {stats['groups']} groups, {len(recs)} units")
    return recs, dict(stats, group_indices_drawn=idx)


def load_exclusion(path):
    if not path:
        return None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        idx = json.load(f)
    return {"ids": set(idx.get("ids", [])) | set(idx.get("unassigned_eval_only_ids", [])),
            "state_sha1": set(idx.get("state_sha1", []))}


def apply_exclusion(recs, idxs):
    out, drop = [], Counter()
    for r in recs:
        hit = False
        for name, ix in idxs.items():
            if ix is None:
                continue
            if r["id"] in ix["ids"] or hashlib.sha1(norm_state(r["state"]).encode()).hexdigest() in ix["state_sha1"]:
                drop[name] += 1
                hit = True
        if not hit:
            out.append(r)
    return out, dict(drop)


def write_split(root, split, recs):
    d = Path(root) / split
    d.mkdir(parents=True, exist_ok=True)
    by = defaultdict(list)
    for r in recs:
        by[r["meta"]["family"]].append(r)
    for fam, v in sorted(by.items()):
        with (d / f"{SET}_{fam}.jsonl").open("w", encoding="utf-8") as f:
            for r in v:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_split(root, split):
    recs = []
    for p in sorted((Path(root) / split).glob("*.jsonl")):
        with p.open(encoding="utf-8") as f:
            recs += [json.loads(line) for line in f]
    return recs


def prompt_chars(r):
    q = next(iter(r["questions"].values()))
    opts = "\n".join(f"A) {t}" for t in q["criteria"].values())
    return len(r["state"]) + len(q["instructions"]) + len(opts) + 120     # + the scorer's fixed PROMPT text


def summarize(recs):
    pc = np.array([prompt_chars(r) for r in recs])
    tok = pc / 4.0 + 30                                               # ~4 chars / token + the chat template
    return {
        "units": len(recs), "groups": len({r["base_id"] for r in recs}),
        "by_family": dict(sorted(Counter(r["meta"]["family"] for r in recs).items())),
        "by_structure": dict(sorted(Counter(f"{r['meta']['family']}/{r['meta']['structure']}" for r in recs).items())),
        "by_qkind": dict(sorted(Counter(r["meta"]["qkind"] for r in recs).items())),
        "by_role": dict(sorted(Counter(r["meta"]["world_role"] for r in recs).items())),
        "by_render_variant": dict(sorted(Counter(r["meta"]["render_variant"] for r in recs).items())),
        "label_kind": dict(Counter(next(iter(r["label_kind"].values())) for r in recs)),
        "k_options": dict(sorted(Counter(len(next(iter(r["questions"].values()))["criteria"]) for r in recs).items())),
        "prompt_tokens_approx_4cpt": {"median": float(np.median(tok)), "p95": float(np.percentile(tok, 95)),
                                      "p99": float(np.percentile(tok, 99)), "max": float(tok.max()),
                                      "over_2048": int((tok > 2048).sum())},
    }


# --------------------------------------------------------------------------- v1.2 build
B_ROLES = ("prior_swap", "dep_swap", "mech_swap", "mech2_swap", "rel_swap", "loss_swap")   # the contrasts forced in train
ALL_SIB_ROLES = B_ROLES + ("round",)


def structure_schedule12(idx, ns, mixture):
    """v1.2 train mix (FAMILY_WEIGHTS_V12 over the in-family structures) or the confirm mix (50% in-family, 30% held-out
    structures incl. missingness2/signature, 20% the confirm-only compositions)."""
    r = rng_of(ns, "structure", idx)
    if mixture == "confirm":
        u = r.random()
        if u >= 0.8:
            return CONFIRM_ONLY[int(r.integers(len(CONFIRM_ONLY)))]
        if u >= 0.5:
            return HELDOUT_V12[int(r.integers(len(HELDOUT_V12)))]
    fams = sorted(FAMILY_WEIGHTS_V12)
    p = np.array([FAMILY_WEIGHTS_V12[f] for f in fams])
    f = fams[int(r.choice(len(fams), p=p / p.sum()))]
    sts = sorted(FAMILIES_V12[f]["train"])
    pw = np.array([FAMILIES_V12[f]["train"][s] for s in sts], float)
    return f, sts[int(r.choice(len(sts), p=pw / pw.sum()))]


def supplement_structure(idx, ns, fam, src):
    r = rng_of(ns, "structure", idx)
    d = FAMILIES_V12[fam][src]
    sts = sorted(d)
    pw = np.array([d[s] for s in sts], float)
    return fam, sts[int(r.choice(len(sts), p=pw / pw.sum()))]


def gen_groups12(split, ns, kind, wording, K, schedule, stop, owner, log=print):
    """Groups until stop(recs, stats, worlds). A group is dropped if any of its world fingerprints or normalised states already
    belongs to ANOTHER split (owner: {'fp': {fp: split}, 'state': {state: split}}); kept groups claim theirs."""
    recs, idx, stats, worlds = [], 0, Counter(), set()
    while not stop(recs, stats, worlds):
        f, s = schedule(idx)
        grp = make_group12(split, idx, ns, f, s, kind, wording, K)
        idx += 1
        fps = {r["meta"]["fingerprint"] for r in grp}
        sts = {norm_state(r["state"]) for r in grp}
        if any(owner["fp"].get(x, split) != split for x in fps):
            stats["groups_dropped_world_shared_with_another_split"] += 1
            continue
        if any(owner["state"].get(x, split) != split for x in sts):
            stats["groups_dropped_state_shared_with_another_split"] += 1
            continue
        for x in fps:
            owner["fp"][x] = split
        for x in sts:
            owner["state"][x] = split
        recs += grp
        stats["groups"] += 1
        worlds |= {r["meta"]["world_id"] for r in grp}
        if stats["groups"] % 2000 == 0:
            log(f"  {split}: {stats['groups']} groups, {len(worlds)} worlds, {len(recs)} units")
    return recs, dict(stats, group_indices_drawn=idx, unique_worlds=len(worlds))


def plain_key(r):
    """(105b) the identity of a train world-question across renderings: rendering r of it in train_aug <-> its train_plain line."""
    m = r["meta"]
    return (m["world_id"], m["world_role"], json.dumps(m["role_info"], sort_keys=True), m["qkind"])


def write_files(d, recs, prefix=f"{SET}_"):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    by = defaultdict(list)
    for r in recs:
        by[r["meta"]["family"]].append(r)
    for fam, v in sorted(by.items()):
        with (d / f"{prefix}{fam}.jsonl").open("w", encoding="utf-8") as f:
            for r in v:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def world_counts(recs):
    """Worlds AND presentations: units = presentations; unique worlds = distinct world ids (anchor + every linked world);
    unique world-question units = distinct (world, question kind) pairs, i.e. presentations without the equivalent renderings."""
    wq = Counter((r["meta"]["world_id"], r["meta"]["qkind"]) for r in recs)
    return {"presentations": len(recs), "groups": len({r["base_id"] for r in recs}),
            "unique_worlds": len({r["meta"]["world_id"] for r in recs}), "unique_world_question_units": len(wq),
            "renderings_per_world_question": dict(sorted(Counter(wq.values()).items())),
            "by_family_unique_worlds": dict(sorted(Counter(f for f, _ in {(r["meta"]["family"], r["meta"]["world_id"])
                                                                           for r in recs}).items()))}


def contrast_stats(recs):
    """Share of units in intervention contrast pairs: a sibling unit and the anchor unit of the same question kind, group and
    rendering (posterior units for every role; also the action / acquire units of loss swaps), by role, and whether the exact
    answer changes (TV > 1e-9) or not (e.g. MAR <-> result-dependent with nothing missing, or the belief under a loss swap)."""
    G = defaultdict(lambda: defaultdict(list))
    for r in recs:
        m = r["meta"]
        if m["qkind"] in ("posterior", "action", "acquire"):
            G[(r["base_id"], m["render_variant"], m["qkind"])][m["world_role"]].append((r["id"], m))
    in_pair, in_pair_b = set(), set()
    by_role, changed = Counter(), Counter()
    for (gid, rv, qk), d in G.items():
        if not d.get("anchor"):
            continue
        aid, a = d["anchor"][0]
        ea = np.asarray(a["exact_target"])
        for role, ms in d.items():
            if role == "anchor":
                continue
            for rid, m in ms:
                tv = 0.5 * float(np.abs(np.asarray(m["exact_target"]) - ea).sum())
                key = role if qk == "posterior" else f"{role}:{qk}"
                by_role[key] += 1
                changed[(key, tv > 1e-9)] += 1
                in_pair |= {rid, aid}
                if role in B_ROLES:
                    in_pair_b |= {rid, aid}
    n = len(recs)
    return {"units": n, "share_of_units_in_forced_contrast_pairs": round(len(in_pair_b) / n, 4),
            "share_of_units_in_any_contrast_pair": round(len(in_pair) / n, 4),
            "pairs_by_role": dict(sorted(by_role.items())),
            "pairs_answer_changed": {role: changed[(role, True)] for role in sorted(by_role)},
            "pairs_answer_unchanged": {role: changed[(role, False)] for role in sorted(by_role)}}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def confirm_manifest(root, recs, ns, salt, stats):
    """The sealed confirm split's manifest: file sha256 / bytes / lines and COUNTS only (no label, target or answer statistic)."""
    d = Path(root) / CONFIRM
    files = {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size, "lines": sum(1 for _ in p.open())}
             for p in sorted(d.glob("*.jsonl"))}
    C = lambda key: dict(sorted(Counter(key(r) for r in recs).items()))                      # noqa: E731
    return {"split": CONFIRM, "status": "SEALED: built 2026-10-06, never scored; the coordinator seals it (hashes below); "
                                        "unseal only when a candidate exists and its protocol is registered",
            "builder": gen_version(), "builder_md5": hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
            "namespace": ns, "salt": salt, "files": files, "units": len(recs), "generation": stats,
            "counts": {"groups": len({r["base_id"] for r in recs}), "unique_worlds": len({r["meta"]["world_id"] for r in recs}),
                       "by_family": C(lambda r: r["meta"]["family"]),
                       "by_structure": C(lambda r: f"{r['meta']['family']}/{r['meta']['structure']}"),
                       "by_qkind": C(lambda r: r["meta"]["qkind"]), "by_role": C(lambda r: r["meta"]["world_role"]),
                       "by_render_variant": C(lambda r: r["meta"]["render_variant"]),
                       "heldout_structure_units": sum(1 for r in recs if r["meta"]["heldout_structure"])},
            "design": "fresh seed namespace; template variant 3 of every key (paraphrase re-renderings mix variants 2 / 3, both "
                      "never in train); confirm-only skins bakery / greenhouse for the fix-form families; 50% in-family structures "
                      "(missingness2 included), 30% held-out structures (v1.1's eight + missingness2/signature), 20% confirm-only "
                      "compositions (composed/cc_retained: common-cause sensors whose reports are lost with a weather-dependent "
                      "chance; composed/relay_retained: relays of one origin passed on with a claim-dependent chance); every "
                      "applicable intervention sibling forced; explicit-factor renderings of every missingness2 world"}


EXTRA_VIEW_NAME = re.compile(r"^train_[a-z0-9_]+$")
EXTRA_VIEW_RESERVED = TRAIN_VIEWS + ("train_plain_k", "train")
EXTRA_VIEW_MIN_GROUP = 2          # the smallest train group has 2 plain units (measured on v1_2h): first-fit never leaves 1 behind
EXTRA_VIEW_MAX_SCAN = 200000      # kept groups after the --train-worlds prefix before a view counts as unfillable
EXTRA_VIEW_SIB = ("forced", "v11")   # (105e) sibling plan of a view: v1.2 train (forced) or each group cut to its v1.1 draw
V11_PLAN_ROLES = ("prior_swap", "dep_swap", "mech_swap", "loss_swap")   # (105e) the roles v1.2 train forces and v1.1 draws
V12_TRAIN_GID = re.compile(r"^train/v12/([A-Za-z0-9_]+)/([A-Za-z0-9_]+)/g(\d+)$")


def parse_extra_view(s, K):
    """(105d) 'NAME:k=1|K:units=N[:drop=FAM+FAM[:refill=FAM+FAM]][:sib=v11]' -> {'name', 'k', 'units', 'drop', 'refill', 'sib',
    'spec'}. units = lines in the view (k x plain units); refill defaults to every train family not dropped; (105e) sib = 'forced'
    (default: the v1.2 train plan, every applicable intervention sibling) or 'v11' (each group cut to its v1.1 sibling plan)."""
    parts = s.split(":")
    name, kv = parts[0], {}
    for p in parts[1:]:
        if "=" not in p:
            raise SystemExit(f"--extra-train-view {s!r}: '{p}' is not key=value")
        kk, v = p.split("=", 1)
        if kk in kv:
            raise SystemExit(f"--extra-train-view {s!r}: '{kk}' given twice")
        kv[kk] = v
    if not EXTRA_VIEW_NAME.match(name) or name in EXTRA_VIEW_RESERVED:
        raise SystemExit(f"--extra-train-view {s!r}: the name must be train_<a-z0-9_> and not one of {EXTRA_VIEW_RESERVED}")
    if set(kv) - {"k", "units", "drop", "refill", "sib"} or not {"k", "units"} <= set(kv):
        raise SystemExit(f"--extra-train-view {s!r}: keys are k, units and optionally drop, refill, sib")
    if kv.get("sib", "forced") not in EXTRA_VIEW_SIB:
        raise SystemExit(f"--extra-train-view {s!r}: sib must be one of {EXTRA_VIEW_SIB}")
    k, units = int(kv["k"]), int(kv["units"])
    fams = sorted(FAMILY_WEIGHTS_V12)
    drop = tuple(sorted(x for x in kv.get("drop", "").split("+") if x))
    refill = tuple(sorted(x for x in kv.get("refill", "").split("+") if x)) or tuple(f for f in fams if f not in drop)
    if k not in (1, K):
        raise SystemExit(f"--extra-train-view {s!r}: k must be 1 or --k ({K})")
    if units <= 0 or units % k:
        raise SystemExit(f"--extra-train-view {s!r}: units must be a positive multiple of k")
    if any(f not in fams for f in drop + refill) or set(drop) & set(refill) or not refill:
        raise SystemExit(f"--extra-train-view {s!r}: drop / refill must name disjoint train families of {fams}")
    if "refill" in kv and not drop:
        raise SystemExit(f"--extra-train-view {s!r}: refill needs drop")
    return {"name": name, "k": k, "units": units, "drop": drop, "refill": refill, "sib": kv.get("sib", "forced"), "spec": s}


def tree_md5(d):
    """(105d) content md5 of a split directory: its *.jsonl files in byte order of their names, each as name + newline + bytes."""
    h = hashlib.md5()
    for p in sorted(Path(d).glob("*.jsonl"), key=lambda q: q.name.encode()):
        h.update(p.name.encode() + b"\n")
        h.update(p.read_bytes())
    return h.hexdigest()


def contiguous_groups(recs):
    """Split a generation-ordered record list into its groups (gen_groups12 appends each group's records contiguously)."""
    out = []
    for r in recs:
        if not out or out[-1][0]["base_id"] != r["base_id"]:
            out.append([])
        out[-1].append(r)
    return out


def v11_sibling_plan(m):
    """(105e) the v1.1 plan's sibling flags of a v1.2 TRAIN group, from any of its records' meta. draw_plan12 draws
    draw_plan(pr, w) first and only then forces prior / dep / mech / loss swaps for train, so re-drawing the group's world and plan
    streams gives the flags the group has without forcing (the eval splits' plan for these roles). Returns (world, {role: bool})."""
    mt = V12_TRAIN_GID.match(m["group_id"])
    if not mt or (mt.group(1), mt.group(2)) != (m["family"], m["structure"]):
        raise SystemExit(f"v11 sibling plan: {m['group_id']!r} is not a v1.2 train group id of {m['family']}/{m['structure']}")
    idx, ns = int(mt.group(3)), m["seed_namespace"]
    w = sample_world12(m["family"], m["structure"], rng_of(ns, "world", idx), SKIN_POOL["standard"])
    p = draw_plan(rng_of(ns, "plan", idx), w)
    p11 = draw_plan11(rng_of(ns, "plan11", idx), w)                   # as make_group12 (the world's value questions)
    if p11:
        w["vq"] = {"regret": p11.get("value_regret"), "voi": p11.get("value_voi")}
    return w, {role: bool(p[role]) for role in V11_PLAN_ROLES}


def v11_filter(grp):
    """(105e) a forced-plan v1.2 train group cut to its v1.1 sibling plan: every record (all K renderings) of a forced role its
    v1.1 draw does not give is dropped; anchors, evidence rounds and the rel_swap / mech2_swap siblings (in every v1.2 plan) are
    kept, each line unchanged. Fails if the re-drawn world is not the group's or a v1.1 role is missing from the forced group.
    Returns (records, flags)."""
    a1 = [r for r in grp if r["meta"]["world_role"] == "anchor" and r["meta"]["render_variant"] == "base"]
    if not a1:
        raise SystemExit(f"v11 sibling plan: group {grp[0]['base_id']} has no rendering-1 anchor")
    w, flags = v11_sibling_plan(a1[0]["meta"])
    if a1[0]["meta"]["fingerprint"] != world_fingerprint(w):
        raise SystemExit(f"v11 sibling plan: the re-drawn world is not group {grp[0]['base_id']}'s")
    have = {r["meta"]["world_role"] for r in grp}
    for role, on in flags.items():
        if on and role not in have:
            raise SystemExit(f"v11 sibling plan: group {grp[0]['base_id']} lacks {role}, which its v1.1 draw gives")
    return [r for r in grp if r["meta"]["world_role"] not in V11_PLAN_ROLES or flags[r["meta"]["world_role"]]], flags


def extra_train_views(views, tr, st_train, ns, K, owner, excl, log=print):
    """(105d, opt-in) Extra train views from ONE train group stream in generation order: the groups of train_aug (the
    --train-worlds prefix), then the next groups of the SAME stream (the group index continues; gen_groups12's rule drops a group
    whose world or state another split owns). Groups with an exclusion-index hit are never used (so the counts stay exact).
      1. PARENT: whole groups in stream order, first-fit to exactly units / k plain units (a group is skipped if it does not fit or
         would leave 1 plain unit, which no group can fill). With units / k = the prefix's plain units the parent IS the prefix.
      2. DROP: the parent's groups of the dropped families are removed (D plain units).
      3. REFILL: D plain units from groups of the refill families that are not in the parent, in stream order, first-fit.
    The view = (parent - dropped) + refill, in stream order. k = 1: the rendering-1 lines (for prefix groups: the train_plain lines
    byte for byte); k = K: every rendering (the train_aug lines). So a drop view differs from its parent only by the swap of the
    dropped families' groups for refill-family groups (refill = the belief families keeps the question-kind mix).
    (105e) sib = 'v11': every group of the view is first cut to its v1.1 sibling plan (v11_filter; each kept line unchanged), and
    plain units are counted on the cut groups (a cut group can hold 1 plain unit, so first-fit only needs the group to fit)."""
    stream = [(g, True) for g in contiguous_groups(tr)]               # (records, from the prefix)
    gen = Counter()
    nxt = [st_train["group_indices_drawn"]]
    idx0 = nxt[0]

    def get(i):
        while len(stream) <= i:
            if len(stream) - n_prefix > EXTRA_VIEW_MAX_SCAN:
                raise SystemExit("extra train views: the group stream is exhausted (EXTRA_VIEW_MAX_SCAN)")
            idx = nxt[0]
            nxt[0] += 1
            f, s_ = structure_schedule12(idx, ns, "train")
            grp = make_group12("train", idx, ns, f, s_, "train", "standard", K)
            fps = {r["meta"]["fingerprint"] for r in grp}
            sts = {norm_state(r["state"]) for r in grp}
            if any(owner["fp"].get(x, "train") != "train" for x in fps):
                gen["groups_dropped_world_shared_with_another_split"] += 1
                continue
            if any(owner["state"].get(x, "train") != "train" for x in sts):
                gen["groups_dropped_state_shared_with_another_split"] += 1
                continue
            for x in fps:
                owner["fp"][x] = "train"
            for x in sts:
                owner["state"][x] = "train"
            stream.append((grp, False))
            gen["groups"] += 1
            if gen["groups"] % 2000 == 0:
                log(f"  extra views: {gen['groups']} groups generated after the prefix")
        return stream[i]

    excluded, plain = {}, {}

    def usable(i):
        if i not in excluded:
            grp = get(i)[0]
            excluded[i] = len(apply_exclusion(grp, excl)[0]) != len(grp)
            plain[i] = sum(1 for r in grp if r["meta"]["render_variant"] == "base")
        return not excluded[i]

    n_prefix = len(stream)

    v11 = {}                                                           # (105e) stream position -> (cut records, v1.1 flags)

    def members(i, sib):
        if sib == "forced":
            return get(i)[0]
        if i not in v11:
            v11[i] = v11_filter(get(i)[0])
        return v11[i][0]

    def n_plain(i, sib):
        return plain[i] if sib == "forced" else sum(1 for r in members(i, sib) if r["meta"]["render_variant"] == "base")

    def first_fit(rem, ok, st, sib="forced"):
        min_left = EXTRA_VIEW_MIN_GROUP if sib == "forced" else 1     # (105e) a cut group can hold 1 plain unit
        took, i = [], 0
        while rem:
            if i >= n_prefix + EXTRA_VIEW_MAX_SCAN:
                raise SystemExit(f"extra train views: could not fill {rem} plain units")
            if not ok(i):
                i += 1
                continue
            if not usable(i):
                st["skip_exclusion"] += 1
            elif n_plain(i, sib) > rem or 0 < rem - n_plain(i, sib) < min_left:
                st["skip_fit"] += 1
            else:
                took.append(i)
                rem -= n_plain(i, sib)
            i += 1
        return took

    out, stats = {}, {}
    for v in views:
        st = Counter()
        sib = v.get("sib", "forced")
        parent = first_fit(v["units"] // v["k"], lambda i: True, st, sib)
        fam = lambda i: get(i)[0][0]["meta"]["family"]                  # noqa: E731
        dropped = [i for i in parent if fam(i) in v["drop"]]
        pset = set(parent)
        refill = first_fit(sum(n_plain(i, sib) for i in dropped), lambda i: i not in pset and fam(i) in v["refill"], st, sib) \
            if dropped else []
        keep = sorted(set(parent) - set(dropped) | set(refill))
        recs = []
        for i in keep:
            grp = members(i, sib)
            recs += grp if v["k"] > 1 else [r for r in grp if r["meta"]["render_variant"] == "base"]
        out[v["name"]] = recs
        stats[v["name"]] = dict(st, parent_groups=len(parent), parent_prefix_groups=sum(stream[i][1] for i in parent),
                                dropped_groups=len(dropped), dropped_plain_units=sum(n_plain(i, sib) for i in dropped),
                                refill_groups=len(refill), groups=len(keep), prefix_groups=sum(stream[i][1] for i in keep),
                                last_stream_position=max(keep) if keep else None)
        if sib == "v11":                                               # (105e) what the v1.1 plan kept and cut, on the view's groups
            sel = lambda rs: [r for r in rs if v["k"] > 1 or r["meta"]["render_variant"] == "base"]   # noqa: E731
            stats[v["name"]]["v11_plan"] = {
                "groups_with_forced_role": dict(sorted(Counter(ro for i in keep for ro in V11_PLAN_ROLES if any(
                    r["meta"]["world_role"] == ro for r in get(i)[0])).items())),
                "groups_keeping_role": dict(sorted(Counter(ro for i in keep for ro in V11_PLAN_ROLES if v11[i][1][ro]).items())),
                "lines_cut_from_view_groups": dict(sorted(Counter(r["meta"]["world_role"] for i in keep for r in sel(get(i)[0])
                                                                  if r["meta"]["world_role"] in V11_PLAN_ROLES and
                                                                  not v11[i][1][r["meta"]["world_role"]]).items())),
                "lines_forced_plan_of_view_groups": sum(len(sel(get(i)[0])) for i in keep)}
    gstats = dict(gen, continued_from_group_index=idx0, group_indices_drawn=nxt[0], stream_groups=len(stream))
    return out, stats, gstats


def cmd_build12(a):
    """v1.2: train_aug / train_plain + the shared eval splits (v1.1 records verbatim + v1.2 supplements) + test_confirm."""
    t0 = time.time()
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    sc = a.scale
    a.k = getattr(a, "k", V12_K)
    a.confirm_salt = getattr(a, "confirm_salt", CONFIRM_SALT)
    NSR = NS_BY_VER["v1.2"]
    owner = {"fp": {}, "state": {}}
    v11, v12, stats = {}, {}, {}
    set_version("v1.1")                                                # 1. the v1.1 eval records, regenerated verbatim
    try:
        sizes11 = default_units()
        if sc != 1.0:
            sizes11 = {kk: max(50, int(v * sc)) for kk, v in sizes11.items()}
        for split in V12_EVAL:
            v11[split], st = gen_split(split, sizes11[split], f"{NS_BY_VER['v1.1']}/{split}")
            stats[split] = {"v1_1_records": st}
    finally:
        set_version("v1.2")
    for split in V12_EVAL:
        for r in v11[split]:
            owner["fp"].setdefault(r["meta"]["fingerprint"], split)
            owner["state"].setdefault(norm_state(r["state"]), split)
        print(f"[{time.strftime('%H:%M:%S')}] {split}: {len(v11[split])} v1.1 records (verbatim)", flush=True)
    for split in V12_EVAL:                                             # 2. v1.2 supplements (missingness2 + rel_swap pairs)
        wording = "novel" if split == "test_novel_wording" else "standard"
        v12[split] = []
        for fam, src, ng in V12_SUPPLEMENT_GROUPS[split]:
            ng = max(6, int(round(ng * sc)))
            ns = f"{NSR}/{split}/supp/{fam}"
            recs, st = gen_groups12(split, ns, "eval", wording, 1,
                                    lambda i, ns=ns, fam=fam, src=src: supplement_structure(i, ns, fam, src),
                                    lambda rs, st_, wd, ng=ng: st_["groups"] >= ng, owner)
            v12[split] += recs
            stats[split][f"v1_2_{fam}"] = st
        print(f"[{time.strftime('%H:%M:%S')}] {split}: {len(v12[split])} v1.2 supplement records", flush=True)
    ns_c = f"bayes_worlds/v1_2_confirm/{a.confirm_salt}/{CONFIRM}"     # 3. the sealed confirm split
    n_conf = max(300, int(CONFIRM_UNITS * sc))
    conf, stats[CONFIRM] = gen_groups12(CONFIRM, ns_c, "confirm", "confirm", 1, lambda i: structure_schedule12(i, ns_c, "confirm"),
                                        lambda rs, st_, wd: len(rs) >= n_conf, owner)
    print(f"[{time.strftime('%H:%M:%S')}] {CONFIRM}: {len(conf)} records (sealed; counts only)", flush=True)
    target = getattr(a, "train_worlds", 0) or (V11_UNIQUE_TRAIN_WORLDS if sc == 1.0 else
                                               max(60, int(round(V11_UNIQUE_TRAIN_WORLDS * sc))))
    ns_t = f"{NSR}/train"                                              # 4. train: K renderings, the v1.1 number of unique worlds
    tr, stats["train"] = gen_groups12("train", ns_t, "train", "standard", a.k, lambda i: structure_schedule12(i, ns_t, "train"),
                                      lambda rs, st_, wd: len(wd) >= target, owner,
                                      log=lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True))
    plain = [r for r in tr if r["meta"]["render_variant"] == "base"]
    print(f"[{time.strftime('%H:%M:%S')}] train: {stats['train']['unique_worlds']} unique worlds; train_aug {len(tr)} / "
          f"train_plain {len(plain)} presentations", flush=True)
    excl = {"rd_exclusion_index": load_exclusion(a.rd_exclusion), "hl_exclusion_index": load_exclusion(a.hl_exclusion)}
    excl_rep = {}
    allsets = {"train_aug": tr, "train_plain": plain, CONFIRM: conf}
    allsets.update({f"{s}:v1_1": v11[s] for s in V12_EVAL})
    allsets.update({f"{s}:v1_2": v12[s] for s in V12_EVAL})
    for key in list(allsets):
        allsets[key], excl_rep[key] = apply_exclusion(allsets[key], excl)
    write_files(root / "train_aug", allsets["train_aug"])
    write_files(root / "train_plain", allsets["train_plain"])
    if getattr(a, "plain_k_view", False):                              # (105b) train_plain_k: line j of each file = the
        base = {plain_key(r): r for r in allsets["train_plain"]}       # rendering-1 record of line j of train_aug, i.e. the
        allsets["train_plain_k"] = [base[plain_key(r)] for r in allsets["train_aug"]]   # plain worlds repeated K times, paired
        write_files(root / "train_plain_k", allsets["train_plain_k"])  # position by position with train_aug
    views = [parse_extra_view(s, a.k) for s in (getattr(a, "extra_train_view", None) or [])]   # (105d, opt-in)
    if len({v["name"] for v in views}) != len(views):
        raise SystemExit("--extra-train-view: duplicate view name")
    vstate, vgen = {}, {}
    if views:
        vrecs, vstate, vgen = extra_train_views(views, tr, stats["train"], ns_t, a.k, owner, excl,
                                                log=lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True))
        for v in views:
            allsets[v["name"]] = vrecs[v["name"]]
            write_files(root / v["name"], vrecs[v["name"]])
            print(f"[{time.strftime('%H:%M:%S')}] {v['name']}: {len(vrecs[v['name']])} lines, {vstate[v['name']]['groups']} groups "
                  f"({vstate[v['name']]['prefix_groups']} from the prefix)", flush=True)
    for s in V12_EVAL:
        write_files(root / s, allsets[f"{s}:v1_1"])
        write_files(root / s, allsets[f"{s}:v1_2"], prefix=f"{SET}_v12_")
    write_files(root / CONFIRM, allsets[CONFIRM])
    trp, tra = allsets["train_plain"], allsets["train_aug"]
    man = {
        "set": SET, "builder": gen_version(), "built": datetime.now().isoformat(timespec="seconds"), "license": LICENSE,
        "namespace": NSR, "build_seconds": round(time.time() - t0, 1), "scale": sc, "K_renderings_train_aug": a.k,
        "train_unique_world_target": target, "hint_per_group": HINT_PER_GROUP,
        "train_plain_k": len(allsets.get("train_plain_k", [])), "generation": {kk: v for kk, v in stats.items() if kk != CONFIRM},
        "layout": "train_aug/ (K renderings per train world) and train_plain/ (rendering 1 only = the same worlds; every line "
                  "byte-identical to the rendering-1 line of train_aug); val / test / test_heldout_structure / test_novel_wording "
                  "= the v1.1 eval records VERBATIM (bayes_worlds_<family>.jsonl, byte-identical to jev_bayes_worlds_v1_1) + "
                  "v1.2 supplements (bayes_worlds_v12_<family>.jsonl); test_confirm/ SEALED (manifest_confirm.json: hashes and "
                  "counts only)",
        "splits": {"train_aug": dict(summarize(tra), worlds=world_counts(tra)),
                   "train_plain": dict(summarize(trp), worlds=world_counts(trp))},
        "train_contrasts": {"train_plain": contrast_stats(trp), "train_aug": contrast_stats(tra)},
        "decision_type_share": {"train_plain": round(sum(r["meta"]["qkind"] in DECISION_KINDS for r in trp) / len(trp), 4),
                                "train_aug": round(sum(r["meta"]["qkind"] in DECISION_KINDS for r in tra) / len(tra), 4)},
        "families": FAMILIES_V12, "family_weights": FAMILY_WEIGHTS_V12, "heldout_structures": [f"{f}/{s}" for f, s in HELDOUT_V12],
        "confirm_only_structures": [f"{f}/{s}" for f, s in CONFIRM_ONLY], "missingness2_rate_support": M2_RATE_SUPPORT,
        "missingness2_none_missing_share": M2_P_NONE_MISSING,
        "exclusion_indices": {"applied": {kk: (v is not None) for kk, v in excl.items()}, "dropped": excl_rep,
                              "statement": "as v1.1: synthetic, generated text only; applied by id and normalised-state sha1 "
                                           "when the index files are given"},
    }
    for s in V12_EVAL:
        a11, a12 = allsets[f"{s}:v1_1"], allsets[f"{s}:v1_2"]
        man["splits"][s] = {"all": dict(summarize(a11 + a12), worlds=world_counts(a11 + a12)),
                            "v1_1_records": len(a11), "v1_2_supplement": dict(summarize(a12), worlds=world_counts(a12))}
    man["splits"][CONFIRM] = "SEALED: see manifest_confirm.json (file sha256 + counts only)"
    if views:                                                          # (105d) only when requested: the default manifest is unchanged
        man["extra_train_views"] = {
            v["name"]: dict(spec=v["spec"], k=v["k"], units=v["units"], drop=list(v["drop"]), refill=list(v["refill"]),
                            selection=vstate[v["name"]],
                            content_md5=tree_md5(root / v["name"]), summary=summarize(allsets[v["name"]]),
                            worlds=world_counts(allsets[v["name"]]), contrasts=contrast_stats(allsets[v["name"]]),
                            decision_type_share=round(sum(r["meta"]["qkind"] in DECISION_KINDS for r in allsets[v["name"]]) /
                                                      len(allsets[v["name"]]), 4),
                            **({"sib": v["sib"]} if v["sib"] != "forced" else {}))       # (105e) only for a non-default plan
            for v in views}
        man["extra_train_views_generation"] = dict(vgen, rule="one train group stream in generation order (the --train-worlds "
                                                   "prefix, then the continued group index); parent = first-fit to units / k "
                                                   "plain units; the parent's dropped-family groups are swapped for refill-family "
                                                   "groups outside the parent (first-fit, same plain units); groups with an "
                                                   "exclusion-index hit never used; k = 1: rendering-1 lines, k = K: every "
                                                   "rendering")
    (root / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    cm = confirm_manifest(root, allsets[CONFIRM], ns_c, a.confirm_salt, stats[CONFIRM])
    (root / "manifest_confirm.json").write_text(json.dumps(cm, indent=1, default=str))
    print(json.dumps({"train_aug": len(tra), "train_plain": len(trp), **{s: len(allsets[f'{s}:v1_1']) + len(allsets[f'{s}:v1_2'])
                                                                         for s in V12_EVAL}, CONFIRM: len(allsets[CONFIRM])}),
          flush=True)
    return man


def cmd_build(a):
    if ver() == "v1.2":
        return cmd_build12(a)
    t0 = time.time()
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    sizes = default_units()
    if a.scale != 1.0:
        sizes = {k: max(50, int(v * a.scale)) for k, v in sizes.items()}
    NSR = ns_root()
    out, stats = {}, {}
    forbid_fp, forbid_state = set(), set()
    for split in [x for x in splits() if x != "train"]:
        recs, st = gen_split(split, sizes[split], f"{NSR}/{split}")
        out[split], stats[split] = recs, st
        forbid_fp |= {r["meta"]["fingerprint"] for r in recs}
        forbid_state |= {norm_state(r["state"]) for r in recs}
        print(f"[{time.strftime('%H:%M:%S')}] {split}: {len(recs)} units, {st}", flush=True)
    recs, st = gen_split("train", sizes["train"], f"{NSR}/train", forbid_fp, forbid_state,
                         log=lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True))
    out["train"], stats["train"] = recs, st
    print(f"[{time.strftime('%H:%M:%S')}] train: {len(recs)} units, {st}", flush=True)
    excl = {"rd_exclusion_index": load_exclusion(a.rd_exclusion), "hl_exclusion_index": load_exclusion(a.hl_exclusion)}
    excl_rep = {}
    for split in splits():
        out[split], drop = apply_exclusion(out[split], excl)
        excl_rep[split] = drop
    for split in splits():
        write_split(root, split, out[split])
    v11 = ver() == "v1.1"
    manifest = {
        "set": SET, "builder": gen_version(), "built": datetime.now().isoformat(timespec="seconds"), "license": LICENSE,
        "namespace": NSR, "requested_units": sizes, "build_seconds": round(time.time() - t0, 1),
        "splits": {s: summarize(out[s]) for s in splits()}, "generation": stats,
        "families": FAMILIES, "family_weights": FAMILY_WEIGHTS_V11 if v11 else FAMILY_WEIGHTS,
        "heldout_structures": [f"{f}/{s}" for f, s in HELDOUT],
        "exclusion_indices": {
            "applied": {k: (v is not None) for k, v in excl.items()}, "dropped": excl_rep,
            "statement": "rd_exclusion_index.json.gz / hl_exclusion_index.json.gz (research/final_suite.md 3.4) apply to every "
                         "generator. This corpus is synthetic: every id is bayes_worlds-*, every state is generated from this "
                         "file's templates (check: vocabulary test), so no record can match; when the index files are given "
                         "(--rd-exclusion / --hl-exclusion) they are applied by id and normalised-state sha1. They are not on "
                         "the laptop (S3 only), so the laptop build applies them trivially."},
        "reserved_names": "no Dirichlet-multinomial / categorical-share model and no dirichlet_share family (final_suite.md 3.4); "
                          "no final_v1 namespace",
        "schema": {
            "record": "build_jev_data schema: id, source (bayes_worlds/<family>), license, base_id (= the group: every linked "
                      "world and re-rendering), state, questions {qid: choice}, labels {qid: {key: p}}, label_kind, meta",
            "question": "ONE `choice` question; criteria dict order = display order; keys <q><canonical index> (never shown)",
            "meta": "family, structure, heldout_structure, skin, group_id, world_id, world_role (anchor / prior_swap / round / "
                    "dep_swap / mech_swap / loss_swap) + role_info, render_variant (base / order / units / paraphrase), qkind, "
                    "target_kind, option_canon (canonical index of each option, display order), exact_target (canonical, full "
                    "precision), exact_extra (expected losses, Bayes set, VOI), posterior_h, truth (hidden: h, latents, the "
                    "realised next observation), truth_from_stated_model, spec (the visible world), render (templates, "
                    "formats, orders), fingerprint",
            "qkinds": TARGET_KIND if v11 else {k: TARGET_KIND[k] for k in QKINDS[:5]},
            "files": "<split>/bayes_worlds_<family>.jsonl; splits train / val / test / test_heldout_structure"},
    }
    if v11:
        tr = out["train"]
        manifest["v1_1"] = {
            "changes": "value question kinds (value_regret / value_gap / value_voi); decision-heavier family mix; the "
                       "test_novel_wording split (in-family structures, reserved template variant 2 of every template, reserved "
                       "skins lab / transit for the fix-form families; hierarchical / multimodal keep their skins); "
                       "meta.difficulty (k, grid = enumeration size, acq_depth, n_evidence) and meta.wording",
            "decision_type_share_train": round(sum(r["meta"]["qkind"] in DECISION_KINDS for r in tr) / len(tr), 4),
            "qkind_share_train": {k: round(v / len(tr), 4) for k, v in sorted(Counter(r["meta"]["qkind"] for r in tr).items())},
            "novel_skins": NOVEL_SKIN_ORDER, "reserved_template_variant": 2,
            "value_bins": {"regret_gap_fractions_of_largest_stated_loss": VALUE_FR, "voi_multiples_of_stated_fee": VOI_MULT}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(json.dumps({s: manifest["splits"][s]["units"] for s in splits()}), flush=True)
    return manifest


# --------------------------------------------------------------------------- check
def source_vocab():
    return {t.lower() for t in re.findall(r"[A-Za-z]{3,}", Path(__file__).read_text(encoding="utf-8"))}


def recompute(r):
    m = r["meta"]
    tgt, extra = exact_target(m["spec"], m["qkind"])
    return tgt


def detect_version(root):
    try:
        b = json.loads((Path(root) / "manifest.json").read_text())["builder"]
    except (OSError, KeyError, ValueError):
        return ver()
    return "v1.2" if "v1.2" in b else "v1.1" if "v1.1" in b else "v1"


def required_roles(m):
    """v1.2 train: the intervention siblings every rendering of a group must carry (section 16 B)."""
    fam, st, spec = m["family"], m["structure"], m["spec"]
    req = Counter()
    if fam not in ("acquisition", "composed", "decision"):
        req["prior_swap"] = 1
    if fam in ("decision", "acquisition"):
        req["loss_swap"] = 1
    if fam == "copied" and st in ("shared_upstream", "noisy_copy", "common_cause"):
        req["dep_swap"] = 1
    if fam == "missingness":
        req["mech_swap"] = 1
    if fam == "reliability" and rel_swap_item(spec) is not None:
        req["rel_swap"] = 1
    if fam == "missingness2":
        req["mech2_swap"] = 2
    return req


def check_extra_views12(root, xv, full, bad, fp_split, state_split, vocab):
    """(105d) the manifest's extra train views: line count = units; content md5 = the manifest's; no dropped family; k = 1 holds
    only rendering-1 lines; every world-question has exactly k renderings with one exact target; every group rendering carries its
    required siblings; groups of the --train-worlds prefix are byte-identical to their train_aug (k = K) / train_plain (k = 1)
    lines; no world or state owned by an eval / confirm split; standard templates and skins only; in-family structures only;
    the other records: schema, foreign tokens and (full) labels recomputed.
    (105e) a sib = 'v11' view: every group = its forced-plan group (prefix: the train_aug / train_plain lines; later groups:
    regenerated with the build's K and hint flag) cut to its v1.1 sibling plan, line for line; every group rendering carries
    exactly its v1.1-plan prior / dep / mech / loss swaps and every rel_swap / mech2_swap sibling of the v1.2 plan."""
    from build_jev_data import validate
    held, conf_only = set(HELDOUT_V12), set(CONFIRM_ONLY)
    man = json.loads((root / "manifest.json").read_text())
    K_build, hpg_build = int(man.get("K_renderings_train_aug", V12_K)), bool(man.get("hint_per_group"))

    def v11_expected(gid, lns, ref_lns, k):
        """The group's forced-plan lines (ref_lns, or regenerated), cut to its v1.1 plan and to rendering 1 if k = 1."""
        global HINT_PER_GROUP
        m = json.loads(lns[0])["meta"]
        if ref_lns is None:
            mt = V12_TRAIN_GID.match(m["group_id"])
            if not mt:
                return None, None
            hpg0, HINT_PER_GROUP = HINT_PER_GROUP, hpg_build
            try:
                grp = make_group12("train", int(mt.group(3)), m["seed_namespace"], m["family"], m["structure"], "train",
                                   "standard", K_build)
            finally:
                HINT_PER_GROUP = hpg0
            if k == 1:
                grp = [r for r in grp if r["meta"]["render_variant"] == "base"]
            ref_lns = [json.dumps(r, ensure_ascii=False) for r in grp]
        recs = [json.loads(ln) for ln in ref_lns]
        a1 = [r for r in recs if r["meta"]["world_role"] == "anchor" and r["meta"]["render_variant"] == "base"]
        if not a1:
            return None, None
        w, flags = v11_sibling_plan(a1[0]["meta"])
        if a1[0]["meta"]["fingerprint"] != world_fingerprint(w):
            return None, None
        return [ln for ln, r in zip(ref_lns, recs) if r["meta"]["world_role"] not in V11_PLAN_ROLES or
                flags[r["meta"]["world_role"]]], flags

    def by_group(split):
        g = defaultdict(list)
        for p in sorted((root / split).glob("*.jsonl")):
            for ln in p.read_text(encoding="utf-8").splitlines():
                g[json.loads(ln)["base_id"]].append(ln)
        return g
    aug, pln = by_group("train_aug"), by_group("train_plain")
    out = {}
    for name, info in sorted(xv.items()):
        if not (root / name).is_dir():
            bad[f"{name}: missing"] += 1
            continue
        k, drop = info["k"], set(info["drop"])
        sib = info.get("sib", "forced")
        ref = aug if k > 1 else pln
        groups = by_group(name)
        n = sum(len(v) for v in groups.values())
        if n != info["units"]:
            bad[f"{name}: {n} lines != units {info['units']}"] += 1
        md5 = tree_md5(root / name)
        if md5 != info["content_md5"]:
            bad[f"{name}: content md5 != manifest"] += 1
        wq, roles, ids, prefix = defaultdict(list), defaultdict(Counter), set(), 0
        v11_flags = {}
        for gid, lns in groups.items():
            in_prefix = gid in ref
            if in_prefix:
                prefix += 1
            if sib == "v11":                                           # (105e) the forced group, cut to its v1.1 plan
                want, v11_flags[gid] = v11_expected(gid, lns, ref.get(gid), k)
                if want is None or lns != want:
                    bad[f"{name}: a group is not its forced-plan group cut to its v1.1 sibling plan"] += 1
            elif in_prefix and lns != ref[gid]:
                bad[f"{name}: a prefix group differs from its {'train_aug' if k > 1 else 'train_plain'} lines"] += 1
            for ln in lns:
                r = json.loads(ln)
                m = r["meta"]
                if r["id"] in ids:
                    bad[f"{name}: duplicate id"] += 1
                ids.add(r["id"])
                if m["family"] in drop:
                    bad[f"{name}: dropped family present"] += 1
                if k == 1 and m["render_variant"] != "base":
                    bad[f"{name}: k = 1 view with a rendering other than 1"] += 1
                if (m["family"], m["structure"]) in held | conf_only:
                    bad[f"{name}: held-out / confirm-only structure"] += 1
                if any(x != "train" for x in fp_split.get(m["fingerprint"], ())) or \
                        any(x != "train" for x in state_split.get(norm_state(r["state"]), ())):
                    bad[f"{name}: world or state owned by an eval / confirm split"] += 1
                tp = m["render"]["tpl"]
                if any(v not in (0, 1) for v in tp.values()) or m["skin"] in NOVEL_SKINS or m["skin"] in CONFIRM_SKINS:
                    bad[f"{name}: reserved template or skin"] += 1
                wq[(m["world_id"], m["qkind"])].append(json.dumps(m["exact_target"]))
                if m["qkind"] == "posterior":
                    roles[(gid, m["render_variant"])][m["world_role"]] += 1
                    if m["world_role"] == "anchor":
                        req = required_roles(m)
                        if sib == "v11":                               # (105e) exactly the v1.1 draw for the four roles
                            fl = v11_flags.get(gid) or {}
                            req = Counter({ro: nn for ro, nn in req.items() if ro not in V11_PLAN_ROLES})
                            req["_v11"] = tuple(sorted(ro for ro in V11_PLAN_ROLES if fl.get(ro)))
                        roles[(gid, m["render_variant"])]["_req"] = req
                if in_prefix:
                    continue
                try:
                    validate(r)
                except AssertionError as e:
                    bad[f"{name}: schema: {e}"] += 1
                q = next(iter(r["questions"].values()))
                txt = r["state"] + " " + q["instructions"] + " " + " ".join(q["criteria"].values())
                if any(tok.lower() not in vocab for tok in re.findall(r"[A-Za-z]{3,}", txt)):
                    bad[f"{name}: foreign tokens"] += 1
                if full:
                    tgt = recompute(r)
                    qid = next(iter(r["questions"]))
                    d = max(abs(tgt[j] - r["labels"][qid][kk]) for j, kk in zip(m["option_canon"], list(q["criteria"])))
                    d2 = max(abs(a_ - b_) for a_, b_ in zip(tgt, m["exact_target"]))
                    if d > 1e-5 or d2 > 1e-12:
                        bad[f"{name}: label != recomputed exact target"] += 1
        if {len(v) for v in wq.values()} != {k}:
            bad[f"{name}: world-questions without exactly {k} renderings"] += 1
        if any(len(set(v)) > 1 for v in wq.values()):
            bad[f"{name}: renderings of one world-question with different exact targets"] += 1
        for cnt in roles.values():
            req = cnt.pop("_req", Counter())
            v11_on = req.pop("_v11", None)
            if any(cnt[role] < nn for role, nn in req.items()):
                bad[f"{name}: group rendering without its required intervention siblings"] += 1
            if v11_on is not None and any(cnt[ro] != (1 if ro in v11_on else 0) for ro in V11_PLAN_ROLES):
                bad[f"{name}: group rendering whose prior / dep / mech / loss swaps are not its v1.1 plan"] += 1
        out[name] = {"lines": n, "groups": len(groups), "prefix_groups": prefix, "content_md5": md5,
                     "unique_worlds": len({w for w, _ in wq}), "world_question_units": len(wq)}
    return out


def check12(root, verbose=True, full=True):
    """v1.2 integrity: everything check() does for v1.1, on the v1.2 layout, plus: train_plain = the rendering-1 lines of
    train_aug byte for byte; every train world-question has K renderings with the SAME exact target; every train group carries
    its required intervention siblings in every rendering; wording / skin reservations (standard 0-1 / novel 2 / confirm 3, and
    confirm skins only in test_confirm); held-out structures only in test_heldout_structure / test_confirm; the confirm-only
    compositions only in test_confirm. test_confirm is checked for integrity (labels recomputed = generator correctness) but no
    statistic of its labels is computed or printed."""
    from build_jev_data import validate
    set_version("v1.2")
    root = Path(root)
    rep, bad = {}, Counter()
    vocab = source_vocab()
    group_split, fp_split, state_split, ids, qids = {}, defaultdict(set), defaultdict(set), set(), set()
    nc = re.compile(r"\bNC\b|non[- ]?commercial|research[- ]only|research use only|academic use only|cc[- ]by[- ]nc|\bby-nc\b", re.I)
    calib, foreign, maxdiff = [], Counter(), 0.0
    held, conf_only = set(HELDOUT_V12), set(CONFIRM_ONLY)
    view = (root / "train").is_dir() and not (root / "train_aug").is_dir()   # a per-arm corpus view (train/ = one arm)
    if not view:
        aug = {p.name: p.read_text(encoding="utf-8").splitlines() for p in sorted((root / "train_aug").glob("*.jsonl"))}
        pln = {p.name: p.read_text(encoding="utf-8").splitlines() for p in sorted((root / "train_plain").glob("*.jsonl"))}
        if sorted(aug) != sorted(pln):
            bad["train_plain files != train_aug files"] += 1
        for name, lines in aug.items():
            base = [ln for ln in lines if json.loads(ln)["meta"]["render_variant"] == "base"]
            if base != pln.get(name):
                bad["train_plain != the rendering-1 lines of train_aug"] += 1
        rep["train_plain"] = sum(len(v) for v in pln.values())
        if (root / "train_plain_k").is_dir():                          # (105b) the paired, K-times repeated plain view
            plk = {p.name: p.read_text(encoding="utf-8").splitlines() for p in sorted((root / "train_plain_k").glob("*.jsonl"))}
            if sorted(plk) != sorted(aug):
                bad["train_plain_k files != train_aug files"] += 1
            for name, lines in aug.items():
                base = {plain_key(json.loads(ln)): ln for ln in pln.get(name, [])}
                want = [base.get(plain_key(json.loads(ln))) for ln in lines]
                if want != plk.get(name):
                    bad["train_plain_k line != the train_plain line of the paired train_aug line"] += 1
            rep["train_plain_k"] = sum(len(v) for v in plk.values())
        man_p = root / "manifest.json"
        if man_p.exists() and json.loads(man_p.read_text()).get("hint_per_group"):
            xg = defaultdict(set)
            for lines in aug.values():
                for ln in lines:
                    m = json.loads(ln)["meta"]
                    if m["family"] == "missingness2":
                        xg[m["group_id"]].add(bool(m.get("explicit_factors")))
            rep["hint_groups"] = {"groups": len(xg), "explicit": sum(v == {True} for v in xg.values())}
            if any(len(v) > 1 for v in xg.values()):
                bad["hint_per_group: a missingness2 group mixes explicit and implicit renderings"] += 1
        del aug, pln
    K = None
    present = [s for s in (("train",) if view else ("train_aug",)) + V12_EVAL + (CONFIRM,) if (root / s).is_dir()]
    for split in present:
        sp = "train" if split in ("train_aug", "train") else split
        recs = read_split(root, split)
        rep[split] = len(recs)
        wq_targets, grp_roles = defaultdict(list), defaultdict(Counter)
        for r in recs:
            try:
                validate(r)
            except AssertionError as e:
                bad[f"schema: {e}"] += 1
            m = r["meta"]
            if r["id"] in ids:
                bad["duplicate id"] += 1
            ids.add(r["id"])
            qid = next(iter(r["questions"]))
            if qid in qids:
                bad["duplicate qid"] += 1
            qids.add(qid)
            if len(r["questions"]) != 1:
                bad["not one question"] += 1
            if group_split.setdefault(r["base_id"], sp) != sp:
                bad["group in two splits"] += 1
            fp_split[m["fingerprint"]].add(sp)
            state_split[norm_state(r["state"])].add(sp)
            fs = (m["family"], m["structure"])
            if fs in conf_only and sp != CONFIRM:
                bad["confirm-only composition outside test_confirm"] += 1
            elif fs in held and sp not in ("test_heldout_structure", CONFIRM):
                bad["held-out structure outside its splits"] += 1
            elif fs not in held and fs not in conf_only and sp == "test_heldout_structure":
                bad["in-family structure in the held-out split"] += 1
            if nc.search(r["license"]):
                bad["licence trips the NC filter"] += 1
            tp = m["render"]["tpl"]
            fixform = m["family"] not in ("hierarchical", "multimodal")
            if sp == CONFIRM:
                ok = all(v in (2, 3) for v in tp.values()) and (m["render_variant"] == "paraphrase" or
                                                                all(v == 3 for v in tp.values()))
                if not ok or (fixform and m["skin"] not in CONFIRM_SKINS):
                    bad["confirm unit with a non-confirm template or skin"] += 1
            elif sp == "test_novel_wording":
                if any(v != 2 for v in tp.values()) or (fixform and m["skin"] not in NOVEL_SKINS):
                    bad["novel-wording unit with a standard template or skin"] += 1
            elif any(v not in (0, 1) for v in tp.values()) or m["skin"] in NOVEL_SKINS or m["skin"] in CONFIRM_SKINS:
                bad["standard unit with a reserved template or skin"] += 1
            if "dirichlet" in json.dumps(r).lower():
                bad["reserved name"] += 1
            q = r["questions"][qid]
            txt = r["state"] + " " + q["instructions"] + " " + " ".join(q["criteria"].values())
            for tok in re.findall(r"[A-Za-z]{3,}", txt):
                if tok.lower() not in vocab:
                    foreign[tok] += 1
            if full:
                tgt = recompute(r)
                lab = r["labels"][qid]
                keys = list(q["criteria"])
                d = max(abs(tgt[j] - lab[kk]) for j, kk in zip(m["option_canon"], keys))
                d2 = max(abs(a_ - b_) for a_, b_ in zip(tgt, m["exact_target"]))
                maxdiff = max(maxdiff, d2)
                if d > 1e-5 or d2 > 1e-12:
                    bad["label != recomputed exact target"] += 1
            if sp != CONFIRM and m["qkind"] == "posterior" and m["truth_from_stated_model"] and m["render_variant"] == "base" \
                    and m["world_role"] in ("anchor", "round"):
                calib.append((m["posterior_h"], m["truth"]["h"]))
            if prompt_chars(r) / 4.0 + 30 > 2048:
                bad["prompt over 2048 approx tokens"] += 1
            if sp == "train":
                wq_targets[(m["world_id"], m["qkind"])].append((m["render_variant"], m["exact_target"]))
                if m["qkind"] == "posterior":
                    grp_roles[(r["base_id"], m["render_variant"])][m["world_role"]] += 1
                    if m["world_role"] == "anchor":
                        grp_roles[(r["base_id"], m["render_variant"])]["_req"] = required_roles(m)
        if sp == "train":
            ks = Counter(len(v) for v in wq_targets.values())
            K = sorted(ks)
            if len(ks) != 1:
                bad["train world-questions with different rendering counts"] += 1
            for v in wq_targets.values():
                if any(t != v[0][1] for _, t in v):
                    bad["train renderings of one world-question with different exact targets"] += 1
            for key, cnt in grp_roles.items():
                req = cnt.pop("_req", Counter())
                if any(cnt[role] < n for role, n in req.items()):
                    bad["train group rendering without its required intervention siblings"] += 1
    man_x = root / "manifest.json"                                     # (105d) extra train views, only when the manifest has them
    xv = json.loads(man_x.read_text()).get("extra_train_views") if (man_x.exists() and not view) else None
    if xv:
        rep["extra_train_views"] = check_extra_views12(root, xv, full, bad, fp_split, state_split, vocab)
    shared_fp = sum(1 for v in fp_split.values() if len(v) > 1)
    shared_state = sum(1 for v in state_split.values() if len(v) > 1)
    if shared_fp:
        bad["world fingerprint in two splits"] += shared_fp
    if shared_state:
        bad["state text in two splits"] += shared_state
    if foreign:
        bad["foreign tokens (not in this file's text)"] += sum(foreign.values())
    pr = np.concatenate([np.asarray(p) for p, _ in calib])
    y = np.concatenate([np.eye(len(p))[h] for p, h in calib])
    bins = np.clip((pr * 10).astype(int), 0, 9)
    rel = []
    for b in range(10):
        s = bins == b
        if s.sum() >= 20:
            rel.append({"bin": b, "n": int(s.sum()), "mean_p": round(float(pr[s].mean()), 4), "freq": round(float(y[s].mean()), 4)})
    ece = float(sum(abs(x["mean_p"] - x["freq"]) * x["n"] for x in rel) / max(1, sum(x["n"] for x in rel)))
    rep.update({"problems": dict(bad) or "none", "max_abs_target_diff": maxdiff, "foreign_tokens": dict(foreign.most_common(20)),
                "worlds_shared_across_splits": shared_fp, "states_shared_across_splits": shared_state,
                "train_renderings_per_world_question": K,
                "exact_calibration_vs_truth (test_confirm excluded)": {"units": len(calib), "ece_all_options": round(ece, 4),
                                                                        "reliability": rel}})
    if verbose:
        print(json.dumps(rep, indent=1))
    return rep


def check(root, verbose=True, full=True):
    """Schema, exact labels recomputed from meta.spec, split integrity, held-out structures, vocabulary (no foreign text),
    lengths, licence, calibration of the exact posterior against the hidden truth; v1.1: the novel-wording split uses only the
    reserved template variant and skins, and no other split uses them."""
    from build_jev_data import validate
    if detect_version(root) == "v1.2":
        return check12(root, verbose, full)
    set_version(detect_version(root))
    rep, bad = {}, Counter()
    vocab = source_vocab()
    seen_ids, group_split, fp_split, state_split, qids = {}, {}, defaultdict(set), defaultdict(set), set()
    nc = re.compile(r"\bNC\b|non[- ]?commercial|research[- ]only|research use only|academic use only|cc[- ]by[- ]nc|\bby-nc\b", re.I)
    calib = []
    foreign = Counter()
    maxdiff = 0.0
    for split in splits():
        recs = read_split(root, split)
        rep[split] = len(recs)
        for r in recs:
            try:
                validate(r)
            except AssertionError as e:
                bad[f"schema: {e}"] += 1
            m = r["meta"]
            if r["id"] in seen_ids:
                bad["duplicate id"] += 1
            seen_ids[r["id"]] = split
            qid = next(iter(r["questions"]))
            if qid in qids:
                bad["duplicate qid"] += 1
            qids.add(qid)
            if len(r["questions"]) != 1:
                bad["not one question"] += 1
            g = r["base_id"]
            if group_split.setdefault(g, split) != split:
                bad["group in two splits"] += 1
            fp_split[m["fingerprint"]].add(split)
            state_split[norm_state(r["state"])].add(split)
            ho = (m["family"], m["structure"]) in HELDOUT
            if ho != (split == "test_heldout_structure"):
                bad["held-out structure outside its split" if ho else "in-family structure in the held-out split"] += 1
            if nc.search(r["license"]):
                bad["licence trips the NC filter"] += 1
            if ver() == "v1.1":
                tp = m["render"]["tpl"]
                fixform = m["family"] not in ("hierarchical", "multimodal")
                if split == "test_novel_wording":
                    if any(v != 2 for v in tp.values()) or (fixform and m["skin"] not in NOVEL_SKINS):
                        bad["novel-wording unit with a standard template or skin"] += 1
                elif any(v == 2 for v in tp.values()) or m["skin"] in NOVEL_SKINS:
                    bad["standard unit with a reserved (novel) template or skin"] += 1
            if "dirichlet" in json.dumps(r).lower():
                bad["reserved name"] += 1
            q = r["questions"][qid]
            txt = r["state"] + " " + q["instructions"] + " " + " ".join(q["criteria"].values())
            for tok in re.findall(r"[A-Za-z]{3,}", txt):
                if tok.lower() not in vocab:
                    foreign[tok] += 1
            if full:
                tgt = recompute(r)
                lab = r["labels"][qid]
                keys = list(q["criteria"])
                d = max(abs(tgt[j] - lab[kk]) for j, kk in zip(m["option_canon"], keys))
                d2 = max(abs(a - b) for a, b in zip(tgt, m["exact_target"]))
                maxdiff = max(maxdiff, d2)
                if d > 1e-5 or d2 > 1e-12:                  # labels: 6 dp, the largest entry absorbs the rounding
                    bad["label != recomputed exact target"] += 1
            if m["qkind"] == "posterior" and m["truth_from_stated_model"] and m["render_variant"] == "base" \
                    and m["world_role"] in ("anchor", "round"):
                calib.append((m["posterior_h"], m["truth"]["h"]))
            if prompt_chars(r) / 4.0 + 30 > 2048:
                bad["prompt over 2048 approx tokens"] += 1
    shared_fp = sum(1 for v in fp_split.values() if len(v) > 1)
    shared_state = sum(1 for v in state_split.values() if len(v) > 1)
    if shared_fp:
        bad["world fingerprint in two splits"] += shared_fp
    if shared_state:
        bad["state text in two splits"] += shared_state
    if foreign:
        bad["foreign tokens (not in this file's text)"] += sum(foreign.values())
    # calibration of the exact posterior against the truth (truth drawn from the stated model)
    pr = np.concatenate([np.asarray(p) for p, _ in calib])
    y = np.concatenate([np.eye(len(p))[h] for p, h in calib])
    bins = np.clip((pr * 10).astype(int), 0, 9)
    rel = []
    for b in range(10):
        s = bins == b
        if s.sum() >= 20:
            rel.append({"bin": b, "n": int(s.sum()), "mean_p": round(float(pr[s].mean()), 4), "freq": round(float(y[s].mean()), 4)})
    ece = float(sum(abs(x["mean_p"] - x["freq"]) * x["n"] for x in rel) / max(1, sum(x["n"] for x in rel)))
    top = np.array([max(p) for p, _ in calib])
    topc = np.array([float(np.argmax(p) == h) for p, h in calib])
    rep.update({"problems": dict(bad) or "none", "max_abs_target_diff": maxdiff, "foreign_tokens": dict(foreign.most_common(20)),
                "worlds_shared_across_splits": shared_fp, "states_shared_across_splits": shared_state,
                "exact_calibration_vs_truth": {"units": len(calib), "ece_all_options": round(ece, 4),
                                               "top_label_mean_p": round(float(top.mean()), 4),
                                               "top_label_accuracy": round(float(topc.mean()), 4), "reliability": rel}})
    if verbose:
        print(json.dumps(rep, indent=1))
    return rep


def cmd_check(a):
    rep = check(a.out, full=not a.fast)
    (Path(a.out) / "check_report.json").write_text(json.dumps(rep, indent=1))
    if rep["problems"] != "none":
        raise SystemExit("check FAILED")


def cmd_report(a):
    man = json.loads((Path(a.out) / "manifest.json").read_text())
    set_version(detect_version(a.out))
    if ver() == "v1.2":
        for s in TRAIN_VIEWS:
            print(f"== {s}: {json.dumps(man['splits'][s]['worlds'])}")
            print("   by_qkind", man["splits"][s]["by_qkind"])
        for s in V12_EVAL:
            print(f"== {s}: {man['splits'][s]['all']['units']} units ({man['splits'][s]['v1_1_records']} v1.1 verbatim)")
        print("contrasts", json.dumps(man["train_contrasts"]))
        print("decision-type share", man["decision_type_share"])
        return
    for s in man["splits"]:
        sm = man["splits"][s]
        print(f"== {s}: {sm['units']} units, {sm['groups']} groups; tokens {sm['prompt_tokens_approx_4cpt']}")
        print("   by_qkind", sm["by_qkind"])
        print("   by_structure", sm["by_structure"])
    recs = read_split(a.out, "test")
    acq = [r for r in recs if r["meta"]["qkind"] == "acquire"]
    if acq:
        print("acquire: buy optimal share", round(float(np.mean([r["meta"]["exact_extra"]["buy_is_optimal"] for r in acq])), 3))
    act = [r for r in recs if r["meta"]["qkind"] == "action"]
    if act:
        k_ab = [r for r in act if r["meta"]["spec"]["loss"]["E"] is not None]
        print("action: abstain optimal share (worlds with abstain)",
              round(float(np.mean([r["meta"]["spec"]["k"] in r["meta"]["exact_extra"]["bayes_set"] for r in k_ab])), 3))
    for qk in VALUE_KINDS:
        v = [r for r in recs if r["meta"]["qkind"] == qk]
        if v:
            print(qk, len(v), "bin shares", dict(sorted(Counter(r["meta"]["exact_extra"]["bin"] for r in v).items())))
    if "v1_1" in man:
        print("v1.1:", json.dumps({k: man["v1_1"][k] for k in ("decision_type_share_train", "qkind_share_train")}))


# --------------------------------------------------------------------------- self-test
def self_test():
    # copied: two verbatim relays of one origin count once (double counting would differ)
    w = base_world("copied", "shared_upstream", "factory", 3, SKINS["factory"]["names"][:3], [34, 33, 33])
    w["ev"] = [{"t": "relay", "origin": 0, "acc": 80, "reps": [{"src": 1, "path": [100], "c": 1},
                                                                {"src": 2, "path": [100], "c": 1}], "stmt": "copied"}]
    p = posterior_h(w)
    one = pct([34, 33, 33]) * acc_lik(3, 80, 1)
    assert np.allclose(p, one / one.sum(), atol=1e-14)
    dbl = posterior_h(w, "ignore_dependence")
    two = pct([34, 33, 33]) * acc_lik(3, 80, 1) ** 2
    assert np.allclose(dbl, two / two.sum(), atol=1e-14) and not np.allclose(p, dbl)
    # VOI: binary, prior 50/50, fix losses 0 / 1000, test sens 90 fpr 10, fee 100
    w = base_world("acquisition", "one_test", "factory", 2, SKINS["factory"]["names"][:2], [50, 50])
    w["loss"] = {"form": "fix", "F": [0, 0], "M": [1000, 1000], "E": None, "scale": 1000}
    w["info"] = [{"t": "test", "on": 0, "sens": 90, "fpr": 10, "fee": 100}]
    v = option_values(w, np.array([0.5, 0.5]))
    assert np.allclose(v, [500, 500, 100 + 100]), v                 # after the test: error 0.1 -> 100 expected loss
    # every family/structure samples, solves, renders deterministically, in both versions (and the novel-wording split)
    for v in ("v1", "v1.1"):
        set_version(v)
        for fam, d in FAMILIES.items():
            for st in list(d["train"]) + list(d["heldout"]):
                for split in (("selftest",) if v == "v1" else ("selftest", "test_novel_wording")):
                    g1 = make_group(split, 0, f"selftest/{v}/{fam}/{st}", fam, st)
                    assert json.dumps(g1) == json.dumps(make_group(split, 0, f"selftest/{v}/{fam}/{st}", fam, st))
                    if v == "v1":
                        assert all("difficulty" not in r["meta"] and r["meta"]["qkind"] in QKINDS[:5] for r in g1)
    set_version("v1.1")
    for fam, d in FAMILIES.items():
        for st in list(d["train"]) + list(d["heldout"]):
            for i in range(3):
                g1 = make_group("selftest", i, f"selftest/{fam}/{st}", fam, st)
                g2 = make_group("selftest", i, f"selftest/{fam}/{st}", fam, st)
                assert json.dumps(g1) == json.dumps(g2)
    # v1.2: every family / structure (incl. missingness2 and the confirm-only compositions), in train (K = 3), eval and confirm
    set_version("v1.2")
    try:
        for fam, d in dict(FAMILIES_V12, **COMPOSED).items():
            for st in list(d["train"]) + list(d["heldout"]):
                for kind, wording, K in (("train", "standard", 3), ("eval", "standard", 1), ("eval", "novel", 1),
                                         ("confirm", "confirm", 1)):
                    if kind == "train" and st not in d["train"]:
                        continue
                    for i in range(2):
                        g1 = make_group12("selftest", i, f"selftest/v12/{fam}/{st}/{kind}", fam, st, kind, wording, K)
                        g2 = make_group12("selftest", i, f"selftest/v12/{fam}/{st}/{kind}", fam, st, kind, wording, K)
                        assert json.dumps(g1) == json.dumps(g2)
    finally:
        set_version("v1.1")
    print("self-test OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("build")
    p.add_argument("--version", choices=sorted(VERSIONS), default="v1.2",
                   help="v1 / v1.1 = the 2026-10-05 corpora, byte for byte; v1.2 (default) = section 16 of research/bayes_worlds.md")
    p.add_argument("--out", default="", help="default: tmp/bayes_worlds/out/jev_bayes_worlds_v1 / _v1_1 / _v1_2")
    p.add_argument("--k", type=int, default=V12_K, help="v1.2: equivalent renderings per train world in train_aug")
    p.add_argument("--confirm-salt", default=CONFIRM_SALT, help="v1.2: seed salt of the sealed test_confirm namespace")
    p.add_argument("--train-worlds", type=int, default=0, help="v1.2: unique train worlds (default: v1.1's 22,032); a smaller "
                   "number gives a prefix of the same train groups (e.g. an exposure-matched arm)")
    p.add_argument("--scale", type=float, default=1.0, help="multiply every split's unit target")
    p.add_argument("--rd-exclusion", default="", help="rd_exclusion_index.json.gz (S3 sealed/final_v1/), if available")
    p.add_argument("--hl-exclusion", default="", help="hl_exclusion_index.json.gz, if available")
    p.add_argument("--hint-per-group", action="store_true", help="v1.2 (105b): draw the missingness2 explicit-factor flag once "
                   "per train group (all renderings and siblings share it) instead of per rendering")
    p.add_argument("--plain-k-view", action="store_true", help="v1.2 (105b): also write train_plain_k/ = train_plain repeated "
                   "K times, line j paired with line j of train_aug (same world-question, rendering 1)")
    p.add_argument("--extra-train-view", action="append", default=[],
                   metavar="NAME:k=K:units=N[:drop=FAM+FAM[:refill=FAM+FAM]][:sib=v11]",
                   help="v1.2 (105d), repeatable: also write NAME/ = whole groups of the train stream in generation order (the "
                   "--train-worlds prefix, then the continued group index): first-fit to N / k plain units, the dropped "
                   "families' groups swapped for refill-family groups (default refill: every other family); k = 1 keeps "
                   "rendering 1 (= train_plain lines), k = --k every rendering (= train_aug lines). E.g. "
                   "train_uq_plain:k=1:units=18366 or train_nm_aug:k=3:units=18366:drop=missingness2:refill=diagnosis+copied; "
                   "(105e) sib=v11 cuts every group to its v1.1 sibling plan first (train_nofc_aug:k=3:units=18366:sib=v11)")
    p = sub.add_parser("check")
    p.add_argument("--out", default=OUT_BY_VER["v1.2"])
    p.add_argument("--fast", action="store_true", help="skip recomputing every label")
    p = sub.add_parser("report")
    p.add_argument("--out", default=OUT_BY_VER["v1.2"])
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if a.cmd == "build":
        global HINT_PER_GROUP
        HINT_PER_GROUP = bool(a.hint_per_group)
        set_version(a.version)
        a.out = a.out or OUT_BY_VER[a.version]
        if a.extra_train_view and a.version != "v1.2":
            raise SystemExit("--extra-train-view needs --version v1.2")
        cmd_build(a)
    elif a.cmd == "check":
        cmd_check(a)
    elif a.cmd == "report":
        cmd_report(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
