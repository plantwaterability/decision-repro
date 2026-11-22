"""Problem lists for the STEM-calculation pool (train data only):
- DeepMind mathematics_dataset (Apache-2.0) generated afresh with new seeds,
  medium train difficulty, modules with a numeric or True/False answer;
- MATH train (EleutherAI/hendrycks_math, all subjects) whose boxed answer
  is a plain number (integer, decimal or fraction).
No SciBench, TheoremQA, MATH test or MMLU-Pro.

  python prep_stem.py --per-module 1500  ->  out/stem_problems.jsonl
"""
import argparse
import importlib
import json
import os
import random
import re
from fractions import Fraction

import numpy as np

from common import OUT, number_text

MODULES = ['algebra__linear_1d', 'algebra__linear_1d_composed', 'algebra__linear_2d', 'algebra__linear_2d_composed',
           'algebra__sequence_next_term', 'arithmetic__add_or_sub', 'arithmetic__add_sub_multiple', 'arithmetic__div',
           'arithmetic__mixed', 'arithmetic__mul', 'arithmetic__mul_div_multiple', 'arithmetic__nearest_integer_root',
           'comparison__closest', 'comparison__kth_biggest', 'comparison__pair', 'measurement__conversion',
           'numbers__div_remainder', 'numbers__div_remainder_composed', 'numbers__gcd', 'numbers__gcd_composed',
           'numbers__is_factor', 'numbers__is_factor_composed', 'numbers__is_prime', 'numbers__is_prime_composed',
           'numbers__lcm', 'numbers__lcm_composed', 'numbers__place_value', 'numbers__place_value_composed',
           'polynomials__evaluate', 'polynomials__evaluate_composed', 'probability__swr_p_level_set',
           'probability__swr_p_sequence']


def gold_of(text):
    """A plain-number or True/False answer as normalized text, else None."""
    t = text.strip().replace('$', '').replace('\\!', '').replace(',', '')
    t = re.sub(r'\\[dt]?frac\{(-?\d+)\}\{(\d+)\}', r'\1/\2', t)
    t = re.sub(r'^\\left\(|\\right\)$', '', t)
    if t in ('True', 'False'):
        return t
    if re.fullmatch(r'-?\d+(\.\d+)?(/\d+)?', t) or re.fullmatch(r'-\\?frac\{\d+\}\{\d+\}', t):
        try:
            v = Fraction(t)
        except (ValueError, ZeroDivisionError):
            return None
        d = v.denominator
        for q in (2, 5):
            while d % q == 0:
                d //= q
        return number_text(v) if d == 1 else f"{v.numerator}/{v.denominator}"
    return None


def boxed(solution):
    i = solution.rfind('\\boxed{')
    if i < 0:
        return None
    depth, j = 0, i + len('\\boxed{')
    for k in range(j, len(solution)):
        if solution[k] == '{':
            depth += 1
        elif solution[k] == '}':
            if depth == 0:
                return solution[j:k]
            depth -= 1
    return None


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--per-module', type=int, default=1500)
    p.add_argument('--seed', type=int, default=20261101)
    a = p.parse_args()
    dmod = importlib.import_module('sympy.solvers.diophantine.diophantine')     # mathematics_dataset 1.0.1
    importlib.import_module('sympy.solvers.diophantine').base_solution_linear = dmod.base_solution_linear
    from mathematics_dataset.modules import modules
    random.seed(a.seed)
    np.random.seed(a.seed % 2**32)
    mods = modules.train(lambda r: (r[0] + (r[1] - r[0]) * 0.5, r[0] + (r[1] - r[0]) * 0.8))
    rows, seen, stats = [], set(), {}
    for name in MODULES:
        group, sub = name.split('__')
        fn, kept, tries = mods[group][sub], 0, 0
        while kept < a.per_module and tries < 4 * a.per_module:
            tries += 1
            try:
                prob = fn()
            except Exception:
                continue
            q, g = str(prob.question).strip(), gold_of(str(prob.answer))
            if g is None or q in seen:
                continue
            seen.add(q)
            rows.append({'pid': f'mds-{name}-{kept}', 'source': 'mds', 'module': name, 'question': q, 'gold': g})
            kept += 1
        stats[name] = kept
    from datasets import get_dataset_config_names, load_dataset
    n_math = 0
    for cfg in get_dataset_config_names('EleutherAI/hendrycks_math'):
        for i, r in enumerate(load_dataset('EleutherAI/hendrycks_math', cfg, split='train')):
            b = boxed(r['solution'])
            g = gold_of(b) if b else None
            if g is None or g in ('True', 'False'):
                continue
            rows.append({'pid': f'math-train-{cfg}-{i}', 'source': 'math', 'module': f"math_{cfg}", 'level': r['level'],
                         'question': r['problem'].strip(), 'gold': g})
            n_math += 1
    with open(os.path.join(OUT, 'stem_problems.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(json.dumps({'problems': len(rows), 'mds': len(rows) - n_math, 'math_train_numeric': n_math, 'per_module': stats}))
