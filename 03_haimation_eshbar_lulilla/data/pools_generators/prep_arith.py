"""Problem lists for the arithmetic pools (train splits only): all of GSM8K
train, and AQuA-RAT train deduplicated, keeping items whose labeled option
states one plain number. SVAMP, ASDiv and MAWPS are never used (SVAMP is a
held-out set and was built from ASDiv-A), nor any test split.

  python prep_arith.py --aqua 40000  ->  out/arith_problems.jsonl
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset  # noqa: E402

import loom_gemma as lg  # noqa: E402
from common import OUT  # noqa: E402


def norm(q):
    return re.sub(r'[^a-z0-9]+', ' ', q.lower()).strip()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--aqua', type=int, default=40000)
    a = p.parse_args()
    rows, seen = [], set()
    for i, r in enumerate(load_dataset('openai/gsm8k', 'main', split='train')):
        seen.add(norm(r['question']))
        rows.append({'pid': f'gsm8k-train-{i}', 'source': 'gsm8k', 'question': r['question'],
                     'gold': lg.gsm8k_gold(r['answer']).replace(',', '')})
    aqua = load_dataset('deepmind/aqua_rat', 'raw', split='train')
    order = list(range(len(aqua)))
    random.Random(7).shuffle(order)
    stats = {'aqua_seen': 0, 'dup': 0, 'gold_not_number': 0}
    n_aqua = 0
    for i in order:
        if n_aqua >= a.aqua:
            break
        r = aqua[i]
        stats['aqua_seen'] += 1
        key = norm(r['question'])
        if key in seen:
            stats['dup'] += 1
            continue
        seen.add(key)
        gold_opt = next((o for o in r['options'] if lg.option_letter(o) == r['correct'].strip().upper()), None)
        gold = lg.option_number(gold_opt) if gold_opt else None
        if gold is None:
            stats['gold_not_number'] += 1
            continue
        others = [lg.option_number(o) for o in r['options'] if o is not gold_opt]
        rows.append({'pid': f'aqua-train-{i}', 'source': 'aqua', 'question': r['question'].strip(),
                     'gold': gold.replace(',', ''), 'aqua_other_options': [x for x in others if x]})
        n_aqua += 1
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'arith_problems.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(json.dumps({'problems': len(rows), 'gsm8k': sum(r['source'] == 'gsm8k' for r in rows), 'aqua': n_aqua, **stats}))
