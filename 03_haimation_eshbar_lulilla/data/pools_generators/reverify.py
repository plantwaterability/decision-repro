"""Rerun a pool's Loom verdicts on a given Loom tree: every verified program
must still return the row's gold, and every distractor program the option
it was recorded for. Distractor programs stored without their value (the
formulas pool) are run on --baseline too, and their value must not change;
--sample-unvalued caps how many. Reports each verdict that changed.

  python reverify.py --loom /tmp/loom_fix --baseline . out/tracking.jsonl ... --out changes.json
"""
import argparse
import json
import os
import random
import sys
from collections import Counter
from multiprocessing import Pool

LOOM = None


def _init(loom):
    global LOOM
    LOOM = loom
    sys.path.insert(0, loom)


def _run(chunk):
    import loom_py
    assert loom_py.__file__.startswith(LOOM), loom_py.__file__
    out = []
    for N in sorted({j[1] for j in chunk}):
        part = [j for j in chunk if j[1] == N]
        res = loom_py.score_python_batch([{'src': s, 'inputs': {}, 'option_texts': opts} for s, _, opts in part],
                                         N=N, numbers='rational', backend='interpreter', max_steps=2_000_000)
        out += [(j, r['status'] if r['status'] == 'ok' else r['abstain_reason'], r.get('value'))
                for j, r in zip(part, res)]
    return out


def jobs_for(path):
    """(src, N, option_texts, what, group, expect) for every program a pool row vouches for."""
    seen, jobs = set(), []
    for l in open(path):
        r = json.loads(l)
        N = r['loom'].get('N', 32)
        gold = r['loom']['value']
        crit = r['questions']['answer']['criteria']
        wrong = [t for k, t in crit.items() if k != r['expected']['answer']]
        progs = r.get('programs_verified') or [r['program']]
        for p in progs:
            key = ('gold', p)
            if (r['group'], key) not in seen:
                seen.add((r['group'], key))
                jobs.append((p, N, {'g': gold}, 'verified', r['group'], 'g'))
        dp = r.get('distractor_programs') or {}
        items = dp.items() if isinstance(dp, dict) else [(None, p) for p in dp]
        for text, p in items:
            if (r['group'], ('d', p, text)) in seen:
                continue
            seen.add((r['group'], ('d', p, text)))
            if text is not None:
                jobs.append((p, N, {'d': text}, 'distractor', r['group'], 'd'))
            else:
                jobs.append((p, N, {'x': '0'}, 'distractor_unvalued', r['group'], 'same'))
    return jobs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('pools', nargs='+')
    ap.add_argument('--loom', required=True)
    ap.add_argument('--procs', type=int, default=12)
    ap.add_argument('--out', default=None)
    ap.add_argument('--baseline', default=None)
    ap.add_argument('--sample-unvalued', type=int, default=5000)
    a = ap.parse_args()
    report, changes = {}, []
    for path in a.pools:
        jobs = jobs_for(path)
        unvalued = [j for j in jobs if j[3] == 'distractor_unvalued']
        if len(unvalued) > a.sample_unvalued:
            keep = set(map(id, random.Random(0).sample(unvalued, a.sample_unvalued)))
            jobs = [j for j in jobs if j[3] != 'distractor_unvalued' or id(j) in keep]

        def run_on(loom, js):
            chunks = [[(s, N, o) for s, N, o, *_ in js[i:i + 300]] for i in range(0, len(js), 300)]
            with Pool(a.procs, initializer=_init, initargs=(os.path.abspath(loom),)) as pool:
                return [x for c in pool.imap(_run, chunks, chunksize=1) for x in c]
        res = run_on(a.loom, jobs)
        base = {}
        if a.baseline:
            uj = [j for j in jobs if j[3] == 'distractor_unvalued']
            base = {id(j): (st, v) for j, (_, st, v) in zip(uj, run_on(a.baseline, uj))}
        c = Counter()
        for j, (_, status, value) in zip(jobs, res):
            src, N, opts, what, group, expect = j
            if what == 'distractor_unvalued':
                if id(j) not in base:
                    continue
                held = base[id(j)] == (status, value)
            else:
                held = status == 'ok'
            c[(what, 'held' if held else 'CHANGED')] += 1
            if not held:
                changes.append({'pool': os.path.basename(path), 'group': group, 'what': what, 'status': status,
                                'expected': opts, 'program': src})
        report[os.path.basename(path)] = {f'{w} {h}': n for (w, h), n in sorted(c.items())}
        print(json.dumps({os.path.basename(path): report[os.path.basename(path)]}), flush=True)
    if a.out:
        json.dump({'loom': a.loom, 'report': report, 'changes': changes}, open(a.out, 'w'), indent=1)
