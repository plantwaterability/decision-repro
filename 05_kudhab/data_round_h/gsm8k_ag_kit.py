"""answer-groups-v1 option construction, copied from the Decision Index kit (decision_index/suite/build/gsm8k_v3.py, MIT) for use on TRAIN golds; functions unchanged."""
import collections
import gzip
import hashlib
import json
import random
from decimal import Decimal
from pathlib import Path



SEED = 20261003
K = 10
WINDOW = 4
METHOD = "answer-groups-v1"
RUN_PREFIX = "v0.3-gsm8k:"
MODEL = "jev-1.13.0"
TEST_SHA256 = "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"
SOURCE_PATH = "gsm8k-rows.jsonl"


def gold_of(answer):
    return Decimal(answer.split("####")[-1].strip().replace(",", ""))


def num(n):
    return format(Decimal(n).normalize(), "f")


def tiebreak(i):
    return hashlib.sha256(f"{SEED}:order:{i}".encode()).hexdigest()


def max_run(vals):
    best = run = 1
    for a, b in zip(vals, vals[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def bands(vals, k):
    out, s, n = [], 0, len(vals)
    while s < n:
        m = 1
        while s + k * m <= n and max_run(vals[s:s + k * m]) > m:
            m += 1
        if s + k * m > n:
            while True:
                s0 = out.pop()[0] if out else 0
                m = (n - s0) // k
                if max_run(vals[s0:n]) <= m:
                    out.append((s0, m))
                    return out
                if s0 == 0:
                    raise ValueError("a value occurs more than n/k times")
        out.append((s, m))
        s += k * m
    return out


def leftover(order, golds, r):
    if not r:
        return [], []
    start = random.Random(f"{SEED}:leftover").randrange(len(order))
    scan = order[start:] + order[:start][::-1]
    pick, seen = [], set()
    for i in scan:
        if golds[i] not in seen:
            pick.append(i)
            seen.add(golds[i])
        if len(pick) == r:
            break
    top = max(golds[i] for i in pick)
    bottom = min(golds[i] for i in pick)
    above = sorted({g for g in golds.values() if g > top})
    below = sorted({g for g in golds.values() if g < bottom}, reverse=True)
    return pick, (above + below)[: K - r]


def groups(golds):
    order = sorted(golds, key=lambda i: (golds[i], tiebreak(i)))
    left, extra = leftover(order, golds, len(order) % K)
    taken = set(left)
    rest = [i for i in order if i not in taken]
    vals = [golds[i] for i in rest]
    out = []
    for s, m in bands(vals, K):
        for j in range(m):
            out.append({"cases": [rest[s + t * m + j] for t in range(K)], "extra": [], "band": [s, m]})
    if left:
        out.append({"cases": sorted(left, key=lambda i: golds[i]), "extra": extra, "band": None})
    for n, g in enumerate(out):
        g["id"] = f"G{n:03d}"
        g["values"] = [golds[i] for i in g["cases"]] + g["extra"]
        assert len(g["values"]) == K and len(set(g["values"])) == K
    return out


def four_sets(group, golds, shift):
    ring = sorted(group["values"])
    n = len(ring)
    pos = {v: p for p, v in enumerate(ring)}
    return {i: [ring[(pos[golds[i]] - shift + t) % n] for t in range(WINDOW)] for i in group["cases"]}


def option_sets(golds):
    sets = {}
    gs = groups(golds)
    shifts = []
    for b in range(0, len(gs), WINDOW):
        block = list(range(WINDOW))
        random.Random(f"{SEED}:shifts:{b // WINDOW}").shuffle(block)
        shifts.extend(block[: len(gs) - b])
    for g, shift in zip(gs, shifts):
        four = four_sets(g, golds, shift)
        for i in g["cases"]:
            assert golds[i] in four[i] and set(four[i]) <= set(g["values"])
            sets[i] = {"group": g["id"], "band": g["band"], 10: g["values"], 4: four[i], "members": sorted(g["cases"])}
    return sets


def options(gold, values, count, i):
    opts = [num(v) for v in values]
    random.Random(f"{SEED}:GSM:{i}:{count}").shuffle(opts)
    return opts, opts.index(num(gold))


