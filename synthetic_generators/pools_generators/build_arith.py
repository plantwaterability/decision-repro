"""Arithmetic pools from sampled programs (GSM8K train, AQuA-RAT train).

Every sampled program runs on Loom (exact, N=64). A problem is kept when a
program's result equals the dataset's label. Its distractors are the values
of wrong programs: first the model's own wrong samples, then mutants of the
shortest verified program (an operator swapped, a constant changed, a step
dropped, a wrong operand), each run on Loom. A distractor has the gold's
type (integer gold, integer distractors), its sign, a terminating decimal of
at most 4 places, and a distinct text. 4- and 10-option rows; a 10-option
row only when 9 such distractors exist.

  python build_arith.py raw_0.jsonl ... --procs 12
"""
import argparse
import ast
import copy
import json
import os
import random
import sys
from fractions import Fraction
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loom_gemma as lg  # noqa: E402
from loom_api import parse_option_number  # noqa: E402
from common import OUT, bare, number_text  # noqa: E402

N = 64
MAX_STEPS = 2_000_000
NAMES = {'gsm8k': ('GSM8K-train', 'GSM8K'), 'aqua': ('AQuA-train', 'AQuA-RAT'),
         'mds': ('DMMath-train', 'DeepMind Mathematics'), 'math': ('MATH-train', 'MATH')}


def _run(chunk):
    from loom_py import score_python_batch
    res = score_python_batch([{'src': s, 'inputs': {}, 'option_texts': {'g': g}} for s, g in chunk], N=N,
                             numbers='rational', backend='interpreter', max_steps=MAX_STEPS)
    out = []
    for r in res:
        v = None
        if r['status'] == 'ok' or r['abstain_reason'] == 'no_match':
            v = Fraction(*r['value_raw']) if r['value_raw'] else Fraction(r['value'])
        out.append((r['status'], r['abstain_reason'], None if v is None else str(v), r['steps']))
    return out


CACHE = os.path.join(OUT, 'arith_loom_cache.jsonl')


def run_all(pairs, procs):
    """Loom results for (program, gold) pairs; cached across runs."""
    import hashlib
    key = lambda pr: hashlib.sha1(f"{N}\0{pr[0]}\0{pr[1]}".encode()).hexdigest()
    cache = {}
    if os.path.exists(CACHE):
        for l in open(CACHE):
            k, v = json.loads(l)
            cache[k] = tuple(v)
    todo = sorted({key(pr): pr for pr in pairs if key(pr) not in cache}.items())
    chunks = [[pr for _, pr in todo[i:i + 400]] for i in range(0, len(todo), 400)]
    if chunks:
        with Pool(procs) as pool, open(CACHE, 'a') as f:
            for chunk, part in zip(chunks, pool.imap(_run, chunks, chunksize=1)):
                for pr, v in zip(chunk, part):
                    cache[key(pr)] = v
                    f.write(json.dumps([key(pr), v]) + '\n')
    print(json.dumps({'loom_runs': len(todo), 'cached': len(pairs) - len(todo)}), flush=True)
    return [cache[key(pr)] for pr in pairs]


SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult, ast.FloorDiv: ast.Div}


def mutants(src, rng, limit=28):
    """Wrong but plausible programs: one local change each."""
    tree = ast.parse(src)
    out = set()

    def nth(t, kind, i):
        return [m for m in ast.walk(t) if isinstance(m, kind)][i]

    binops = [n for n in ast.walk(tree) if isinstance(n, ast.BinOp)]
    for i, n in enumerate(binops):
        if type(n.op) in SWAP:
            t = copy.deepcopy(tree)
            nth(t, ast.BinOp, i).op = SWAP[type(n.op)]()
            out.add(ast.unparse(t))
    consts = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and type(n.value) in (int, float)]
    for i, n in enumerate(consts):
        for f in (lambda v: v + 1, lambda v: v - 1, lambda v: v * 2, lambda v: v * 10, lambda v: v / 2):
            v = f(n.value)
            if v == n.value or v < 0 <= n.value:
                continue
            t = copy.deepcopy(tree)
            nth(t, ast.Constant, i).value = int(v) if float(v).is_integer() and type(n.value) is int else v
            out.add(ast.unparse(t))
    fns = [f for f in tree.body if isinstance(f, ast.FunctionDef)]
    for fi, f in enumerate(fns):
        for si, st in enumerate(f.body[:-1]):
            if isinstance(st, (ast.Assign, ast.AugAssign)):
                t = copy.deepcopy(tree)
                del [g for g in t.body if isinstance(g, ast.FunctionDef)][fi].body[si]
                out.add(ast.unparse(t))
        names = sorted({tg.id for st in ast.walk(f) if isinstance(st, ast.Assign) for tg in st.targets
                        if isinstance(tg, ast.Name)})
        loads = [n for n in ast.walk(f) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in names]
        for li in range(len(loads)):
            for other in names:
                if other != loads[li].id:
                    t = copy.deepcopy(tree)
                    g = [x for x in t.body if isinstance(x, ast.FunctionDef)][fi]
                    [n for n in ast.walk(g) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                     and n.id in names][li].id = other
                    out.add(ast.unparse(t))
    out.discard(ast.unparse(tree))
    out = sorted(out)
    rng.shuffle(out)
    return [s + '\n' for s in out[:limit]]


def terminating(v: Fraction) -> bool:
    d = v.denominator
    for p in (2, 5):
        while d % p == 0:
            d //= p
    return d == 1


def fraction_text(v: Fraction) -> str:
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def renderable(v: Fraction, gold: Fraction):
    """The option text for a distractor value, or None if it is not a
    plausible distractor for this gold. A gold that is not a terminating
    decimal (2/35) gets fraction options; otherwise options are decimals."""
    if v == gold or (gold.denominator == 1) != (v.denominator == 1) or (gold > 0 > v) or (gold < 0 < v):
        return None
    if not terminating(gold):
        return fraction_text(v) if v.denominator < 10 ** 6 and abs(v.numerator) < 10 ** 9 else None
    d = v.denominator
    for p in (2, 5):
        while d % p == 0:
            d //= p
    if d != 1 or len(str(v.denominator)) > 5:
        return None
    text = number_text(v)
    return text if len(text.split('.')[-1]) <= 4 or '.' not in text else None


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('raw', nargs='+')
    p.add_argument('--procs', type=int, default=12)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--problems', default='arith_problems.jsonl')
    p.add_argument('--prefix', default='arith')
    a = p.parse_args()
    rng = random.Random(a.seed)
    probs = {json.loads(l)['pid']: json.loads(l) for l in open(os.path.join(OUT, a.problems))}
    loom_gold = lambda g: {'True': '1', 'False': '0'}.get(g, g)       # True/False answers run as 1/0
    raw = {}
    for path in a.raw:
        for l in open(path):
            r = json.loads(l)
            raw[r['pid']] = r
    stats = {'problems': len(raw), 'samples': 0, 'no_program': 0, 'truncated': 0}
    pairs, index = [], []
    for pid, r in raw.items():
        gold = loom_gold(probs[pid]['gold'])
        for k, (text, fin) in enumerate(zip(r['replies'], r['finish'])):
            stats['samples'] += 1
            stats['truncated'] += fin == 'length'
            prog = lg.extract_program(text)
            if prog is None:
                stats['no_program'] += 1
                continue
            try:
                prog = bare(prog)
            except SyntaxError:
                stats['no_program'] += 1
                continue
            pairs.append((prog, gold))
            index.append((pid, k))
    print(json.dumps({'stage': 'samples', **stats, 'programs': len(pairs)}), flush=True)
    res = run_all(pairs, a.procs)
    per = {}
    reasons = {}
    for (pid, k), (prog, _), (status, reason, value, steps) in zip(index, pairs, res):
        e = per.setdefault(pid, {'ok': [], 'wrong': []})
        if status == 'ok':
            e['ok'].append((prog, value, steps))
        else:
            reasons[reason] = reasons.get(reason, 0) + 1
            if value is not None:
                e['wrong'].append((prog, value))
    solved = {pid: e for pid, e in per.items() if e['ok']}
    print(json.dumps({'stage': 'verified', 'problems_solved': len(solved), 'sample_fail_reasons': reasons}), flush=True)

    # mutants of the shortest verified program of each solved problem
    mut_pairs, mut_index = [], []
    for pid, e in solved.items():
        canon = min(e['ok'], key=lambda x: (len(x[0]), x[0]))[0]
        e['canon'] = canon
        try:
            ms = mutants(canon, random.Random(pid))     # per problem, so reruns hit the Loom cache
        except Exception:
            ms = []
        if probs[pid]['gold'] in ('True', 'False'):
            ms = []                                 # the other truth value is the only distractor
        for m in ms:
            mut_pairs.append((m, loom_gold(probs[pid]['gold'])))
            mut_index.append(pid)
    print(json.dumps({'stage': 'mutants', 'programs': len(mut_pairs)}), flush=True)
    mres = run_all(mut_pairs, a.procs)
    for pid, (prog, _), (status, reason, value, steps) in zip(mut_index, mut_pairs, mres):
        if status != 'ok' and value is not None:
            solved[pid].setdefault('mutant', []).append((prog, value))

    sources = sorted({probs[pid]['source'] for pid in raw})
    files = {src: open(os.path.join(OUT, f'{a.prefix}_{src}.jsonl'), 'w') for src in sources}
    counts = {src: {'problems': 0, 'rows_2': 0, 'rows_4': 0, 'rows_10': 0, 'too_few_distractors': 0,
                    'gold_not_decimal': 0} for src in sources}
    for pid, e in solved.items():
        pr = probs[pid]
        src = pr['source']
        c = counts[src]
        family, label = NAMES[src]
        tail = pid.split('-', 1)[1].replace('train-', '', 1)     # 123, algebra__linear_1d-17, algebra-5
        canon_steps = next(s_ for p_, v, s_ in e['ok'] if p_ == e['canon'])
        verified = sorted({p_ for p_, _, _ in e['ok']}, key=len)[:4]
        base = {'family': f'LOOMD-{family}', 'split': 'train', 'program': e['canon'], 'programs_verified': verified,
                'meta': {'source': src, 'pid': pid, 'module': pr.get('module'), 'correct_samples': len(e['ok']),
                         'samples': len(raw[pid]['replies'])},
                'group': f"LOOMD-{family}-{tail}"}
        if pr['gold'] in ('True', 'False'):
            opts = ['True', 'False']
            files[src].write(json.dumps({**base, 'id': f"LOOMD-{family}-{tail}-2",
                'state': {'question': pr['question'], 'task': f"{label} deterministic 2-choice selection"},
                'questions': {'answer': {'type': 'choice', 'instructions':
                              "Choose the answer to the problem in state. Do not provide reasoning. "
                              f"This is a named multiple-choice adaptation of {label}.",
                              'criteria': {'option_0': 'True', 'option_1': 'False'}}},
                'expected': {'answer': f"option_{opts.index(pr['gold'])}"},
                'loom': {'value': pr['gold'], 'steps': canon_steps, 'N': N}}) + '\n')
            c['problems'] += 1
            c['rows_2'] += 1
            continue
        g = parse_option_number(pr['gold'])          # dataset golds like "$24" or "1,200"
        if g is None:
            c['gold_not_decimal'] += 1
            continue
        gold = Fraction(g)
        gold_text = number_text(gold) if terminating(gold) else fraction_text(gold)
        dis = {}
        for origin, pool in (('sample', e['wrong']), ('mutant', e.get('mutant', []))):
            for prog, value in pool:
                t = renderable(Fraction(value), gold)
                if t and t != gold_text and t not in dis:
                    dis[t] = (prog, origin)
        if len(dis) < 3:
            c['too_few_distractors'] += 1
            continue
        c['problems'] += 1
        texts = list(dis)
        for k in (4, 10):
            if k - 1 > len(texts):
                continue
            chosen = texts[:k - 1]                         # model mistakes first, then mutants
            opts = [gold_text] + chosen
            rng.shuffle(opts)
            row = {
                **base, 'id': f"LOOMD-{family}-{tail}-{k}",
                'state': {'question': pr['question'], 'task': f"{label} deterministic {k}-choice numeric selection"},
                'questions': {'answer': {'type': 'choice', 'instructions':
                              "Choose the numeric answer to the problem in state. Do not provide reasoning. "
                              f"This is a named multiple-choice adaptation of {label}.",
                              'criteria': {f'option_{i}': o for i, o in enumerate(opts)}}},
                'expected': {'answer': f'option_{opts.index(gold_text)}'},
                'distractor_programs': {t: dis[t][0] for t in chosen},
                'distractor_origin': {t: dis[t][1] for t in chosen},
                'loom': {'value': gold_text, 'steps': canon_steps, 'N': N},
            }
            files[src].write(json.dumps(row) + '\n')
            c[f'rows_{k}'] += 1
    for f in files.values():
        f.close()
    print(json.dumps({'stage': 'rows', **counts}), flush=True)
