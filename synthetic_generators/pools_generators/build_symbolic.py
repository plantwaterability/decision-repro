"""Build the symbolic pools: every item's program runs on Loom and must
return the generator's own answer.

  python build_symbolic.py tracking:15000 shapes:10000 [--backend auto]
"""
import argparse
import importlib
import json
import os
import time

from common import OUT, build

GENERATORS = {'tracking': 'gen_tracking', 'shapes': 'gen_shapes', 'causal': 'gen_causal', 'navigate': 'gen_navigate',
              'boolean': 'gen_boolean', 'lies': 'gen_lies', 'belief': 'gen_belief',
              'formulas': 'gen_formulas'}
SEEDS = {'tracking': 101, 'shapes': 102, 'causal': 103, 'navigate': 104, 'boolean': 105, 'lies': 106, 'belief': 107, 'formulas': 108}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('jobs', nargs='+', help='family:count')
    p.add_argument('--backend', default='auto')
    p.add_argument('--workers', type=int, default=0)
    a = p.parse_args()
    summary = {}
    for job in a.jobs:
        fam, n = job.split(':')
        mod = importlib.import_module(GENERATORS[fam])
        t0 = time.time()
        stats = build(fam, mod.TASK, mod.make, int(n), SEEDS[fam], backend=a.backend, workers=a.workers,
                      N=getattr(mod, 'N', 32))
        stats['seconds'] = round(time.time() - t0)
        summary[fam] = stats
        print(json.dumps({fam: stats}), flush=True)
    with open(os.path.join(OUT, 'build_symbolic.log'), 'a') as f:
        f.write(json.dumps(summary) + '\n')
