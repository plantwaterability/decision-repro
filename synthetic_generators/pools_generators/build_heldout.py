"""Held-out splits of the symbolic pools, for in-distribution checks: a
seed disjoint from the training build, and no question text or program that
appears in the training file. Rows have split 'heldout' and ids/groups
LOOMD-<family>-heldout-<i>.

  python build_heldout.py causal:500 navigate:500  ->  out/heldout/<family>.jsonl
"""
import argparse
import importlib
import json
import os

from build_symbolic import GENERATORS, SEEDS
from common import OUT, build

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('jobs', nargs='+', help='family:items')
    p.add_argument('--backend', default='auto')
    a = p.parse_args()
    for job in a.jobs:
        fam, n = job.split(':')
        mod = importlib.import_module(GENERATORS[fam])
        with open(os.path.join(OUT, f'{fam}.jsonl')) as f:
            rows = [json.loads(l) for l in f]
        train, programs = {r['state']['question'] for r in rows}, {r['program'] for r in rows}
        stats = build(fam, mod.TASK, mod.make, int(n), 9000 + SEEDS[fam], backend=a.backend, N=getattr(mod, 'N', 32),
                      out=os.path.join(OUT, 'heldout', f'{fam}.jsonl'), exclude=train, exclude_programs=programs, split='heldout',
                      log=lambda m: None)
        stats['excluded_training_questions'] = len(train)
        print(json.dumps({fam: stats}), flush=True)
