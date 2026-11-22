"""Shared pieces of the Loom-verified DI training pools: run each program on
Loom (exact, N=32), keep it only when Loom's result equals the generator's
own answer, and write DI-format multiple-choice rows.

A generator item is a dict:
  question  the text that goes in state.question
  program   a def solution() program (Loom's Python subset) that returns the
            answer's code: an option index, a class id, 1/0, or a number
  answer    the generator's own value for that code (int or Fraction)
  options   option texts; the gold text is options[gold]
  gold      index of the gold option
  variants  option counts to emit, e.g. [5] or [4, 10]; a variant keeps the
            gold and samples the rest from the other options
  meta      generator parameters (depth, entities, family, ...)
  distractors (optional) [(program, value, text), ...]: wrong programs whose
            values are distractors. Each runs on Loom too; only those whose
            Loom value equals the generator's are kept, then options become
            [gold text] + their texts and variants needing more are dropped.
"""
import ast
import json
import os
import random
import re
import sys
from decimal import Decimal
from fractions import Fraction
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loom_py import score_python_batch  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
INSTRUCTIONS = "Choose the answer to the question in state. Do not provide reasoning."


class _Untemp(ast.NodeTransformer):
    """t1, t2, ... -> t1_, t2_, ...: loom_py names its expression temporaries
    py_<fn>__t<n>, the same C name it gives a local t<n>, so such a local can
    be overwritten mid-expression (600 / (v - 200) came out 1)."""
    def visit_Name(self, n):
        if re.fullmatch(r't\d+', n.id):
            n.id += '_'
        return n

    def visit_arg(self, n):
        if re.fullmatch(r't\d+', n.arg):
            n.arg += '_'
        return n


def bare(src: str) -> str:
    """The program without comments, as ast.unparse writes it, with locals
    that would collide with loom_py's temporaries renamed."""
    return ast.unparse(_Untemp().visit(ast.parse(src))) + '\n'


def number_text(v) -> str:
    """A number as the DI renders it: Decimal.normalize() without exponents
    ("18", "2.5", "-3", "180"). Fractions must be terminating decimals."""
    if isinstance(v, Fraction):
        d = Decimal(v.numerator) / Decimal(v.denominator)
        if Fraction(d) != v:
            raise ValueError(f"{v} is not a terminating decimal")
        v = d
    text = format(Decimal(str(v)).normalize(), 'f')
    return '0' if text in ('-0', '') else text


def exact_text(v) -> str:
    """An exact value as an option text Loom's matcher reads (n/d if needed)."""
    v = Fraction(v)
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def verify(items: List[Dict], backend: str = 'auto', workers: int = 0, max_steps: int = 10_000_000,
           N: int = 32) -> List[Dict]:
    """Run every item's program on Loom (exact, N-bit words); keep those
    whose result equals the item's own answer. Adds item['loom'] = {value,
    steps, backend, N}. Returns (kept, stats) where stats counts the reasons
    for dropping."""
    calls = [{'src': it['program'], 'inputs': {}, 'option_texts': {'a': exact_text(it['answer'])}} for it in items]
    res = score_python_batch(calls, N=N, numbers='rational', backend=backend, compile_workers=workers,
                             max_steps=max_steps)
    kept, stats = [], {'total': len(items), 'kept': 0}
    for it, r in zip(items, res):
        if r['status'] == 'ok':
            it['loom'] = {'value': str(r['value']), 'steps': r['steps'], 'backend': r.get('backend'), 'N': N}
            kept.append(it)
            stats['kept'] += 1
        else:
            why = r['abstain_reason'] or 'unknown'
            stats[why] = stats.get(why, 0) + 1
            if why not in ('no_match', 'overflow'):          # anything else is a generator bug: show one
                stats.setdefault('example_' + why, (str(r['fault_detail'])[:200], it['program'][:300]))
    return kept, stats


def verify_distractors(items: List[Dict], backend: str = 'auto', workers: int = 0, N: int = 32) -> Dict:
    """Run every distractor program; keep the ones Loom agrees with."""
    flat = [(i, d) for i, it in enumerate(items) for d in it.get('distractors', [])]
    if not flat:
        return {}
    calls = [{'src': d[0], 'inputs': {}, 'option_texts': {'a': exact_text(d[1])}} for _, d in flat]
    res = score_python_batch(calls, N=N, numbers='rational', backend=backend, compile_workers=workers,
                             max_steps=10_000_000)
    good: Dict[int, List] = {}
    stats = {'distractor_programs': len(flat), 'distractors_kept': 0}
    for (i, d), r in zip(flat, res):
        if r['status'] == 'ok':
            good.setdefault(i, []).append(d)
            stats['distractors_kept'] += 1
    for i, it in enumerate(items):
        if 'distractors' not in it:
            continue
        ds = good.get(i, [])
        it['options'] = [it['options'][it['gold']]] + [d[2] for d in ds]
        it['gold'] = 0
        it['distractor_programs'] = [bare(d[0]) for d in ds]
        it['variants'] = [k for k in it['variants'] if k <= len(it['options'])]
        del it['distractors']
    return stats


def rows_for(item: Dict, family: str, task: str, idx: int, rng: random.Random,
             instructions: str = INSTRUCTIONS, key: str = 'answer', split: str = 'train') -> List[Dict]:
    """DI rows for one verified item, one per option-count variant."""
    rows, tag = [], family if split == 'train' else f'{family}-{split}'
    gold_text = item['options'][item['gold']]
    others = [o for i, o in enumerate(item['options']) if i != item['gold']]
    for k in item['variants']:
        if k - 1 > len(others):
            continue
        opts = [gold_text] + (others if k - 1 == len(others) else rng.sample(others, k - 1))
        rng.shuffle(opts)
        criteria = {f'option_{i}': o for i, o in enumerate(opts)}
        rows.append({
            'id': f'LOOMD-{tag}-{idx}-{k}',
            'family': f'LOOMD-{family}',
            'split': split,
            'state': {'question': item['question'], 'task': task.format(k=k)},
            'questions': {key: {'type': 'choice', 'instructions': instructions, 'criteria': criteria}},
            'expected': {key: f'option_{opts.index(gold_text)}'},
            'program': bare(item['program']),
            'loom': item['loom'],
            'meta': item.get('meta', {}),
            'group': f'LOOMD-{tag}-{idx}',
        })
        if 'distractor_programs' in item:
            rows[-1]['distractor_programs'] = item['distractor_programs']
    return rows


def build(family: str, task: str, make, n: int, seed: int, backend: str = 'auto', workers: int = 0,
          batch: int = 2000, out: str = None, log=print, N: int = 32, exclude=(), exclude_programs=(), split: str = 'train') -> Dict:
    """Generate n verified items with make(rng) and write their rows to
    OUT/<family>.jsonl. Question texts in exclude and programs in
    exclude_programs are skipped (a held-out split excludes the training
    ones). Returns the drop statistics."""
    rng = random.Random(seed)
    path = out or os.path.join(OUT, f'{family}.jsonl')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    total: Dict[str, object] = {}
    done, rows_written, questions = 0, 0, set(exclude)
    with open(path, 'w') as f:
        while done < n:
            items = [make(rng) for _ in range(min(batch, 2 * (n - done) + 8))]
            kept, stats = verify(items, backend=backend, workers=workers, N=N)
            stats.update(verify_distractors(kept, backend=backend, workers=workers, N=N))
            kept = [it for it in kept if it['variants']]
            for k, v in stats.items():
                total[k] = total.get(k, 0) + v if isinstance(v, int) else v
            fresh = []
            for it in kept:                       # one item per question text: no duplicates across groups
                if it['question'] not in questions and bare(it['program']) not in exclude_programs:
                    questions.add(it['question'])
                    fresh.append(it)
            total['duplicate_questions'] = total.get('duplicate_questions', 0) + len(kept) - len(fresh)
            for it in fresh[:n - done]:
                for row in rows_for(it, family, task, done, rng, split=split):
                    f.write(json.dumps(row) + '\n')
                    rows_written += 1
                done += 1
            log(f"{family}: {done}/{n} items, {rows_written} rows, drops so far {total}")
    total['rows'] = rows_written
    return total


def pick(rng: random.Random, xs: Sequence, k: int) -> List:
    return rng.sample(list(xs), k)
