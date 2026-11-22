"""Library mining over Loom-verified programs (DreamCoder / LILO style).
Straight-line programs are inlined into one expression tree for the
returned value; calls to straight-line helper functions (the formulas pool's
kinetic_energy, posterior, ...) are inlined too, and a mined fragment that is
exactly a helper's body takes the helper's name when most of its uses come
from that helper. Helpers with loops (choose, power) leave the program
opaque. Candidate library functions are tree fragments: every depth-<=2
pattern and the full skeleton of every subtree, with numbers as parameters
(common unit constants such as 60 or 100 may be built in; the exponent of **
and round's digits always are, since Loom needs them constant). Fragments
are ranked by (programs using it) x (operations it saves); the library of
size N is the top N, except that a fragment no training program's tiling
picks (a larger one always beats it) is replaced by the next candidates
until every function is used.

Before mining, problems are split into train and eval by what must stay
together: a formulas generator family, a DeepMind module (with its _composed
twin), else the number-masked question template. Held-out problems given
with --exclude are dropped from both. The library is mined and frozen on
train only (helper names included); eval is only tiled against it. A program
is tiled with the fewest calls: library fragments, plus + - * / // % ** and
the builtins as single-operation primitives (each also a call). Reports, for
N = 50 / 200 / 1000, separately for train (in-sample) and eval:
  - (a) oracle coverage: the share of programs expressible in <= 1 / 2 / 3
    calls whose composition (the library functions plus
    `def solution(): return <calls>`) returns the verified answer on Loom,
    and Loom's failure statuses for the rest,
  - tokens of the call expression vs the full program (Gemma tokenizer),
  - the share of those whose arguments are all numbers stated in the question,
  - (b) a BM25 baseline over function name + description + signature:
    Recall@k of the oracle tiling's functions from the question text alone.

  python mine_library.py out/arith_gsm8k.jsonl [...] --exclude heldout.jsonl
writes out/library.json ('libraries': the function names of library N, each
picked by some training tiling unless --no-prune; 'functions': their
definitions, with no eval ids or question words in descriptions), library_eval_manifest.jsonl,
library_calls_N<n>[_eval].jsonl and library_retrieval_eval_N<n>.jsonl
(queries, gold function sets, oracle calls, BM25 top-20).
"""
import argparse
import ast
import hashlib
import json
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OPS = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/', ast.FloorDiv: '//', ast.Mod: '%', ast.Pow: '**'}
FUNCS = {'round', 'abs', 'min', 'max', 'int', 'float', 'floor', 'ceil'}
COMMUTATIVE = {'+', '*', 'min', 'max'}
UNITS = {2, 3, 4, 5, 7, 10, 12, 24, 30, 52, 60, 100, 365, 1000, 0.5, 0.25}
HOLE = ('?',)
MAX_NODES = 120


class Opaque(Exception):
    pass




def tree_of(src, names=None):
    """The returned value of a straight-line solution() as a tuple tree:
    ('num', v) leaves, (op, child, ...) nodes. Raises Opaque otherwise."""
    mod = ast.parse(src)
    fns = {f.name: f for f in mod.body if isinstance(f, ast.FunctionDef)}
    if 'solution' not in fns or fns['solution'].args.args:
        raise Opaque('no solution()')
    return body_tree(fns['solution'], {}, fns, names)


def body_tree(fn, env, fns, names=None):
    """The returned value of a straight-line function body, its parameters
    bound in env. Each helper inlined is counted in names (pattern -> name)."""
    env = dict(env)

    def ex(e):
        if isinstance(e, ast.Constant) and type(e.value) in (int, float):
            return ('num', e.value)
        if isinstance(e, ast.Name):
            if e.id not in env:
                raise Opaque('name')
            return env[e.id]
        if isinstance(e, ast.BinOp) and type(e.op) in OPS:
            return norm((OPS[type(e.op)], ex(e.left), ex(e.right)))
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub):
            c = ex(e.operand)
            return ('num', -c[1]) if c[0] == 'num' else ('neg', c)
        if isinstance(e, ast.Call):
            name = e.func.id if isinstance(e.func, ast.Name) else e.func.attr if isinstance(e.func, ast.Attribute) else None
            if name in FUNCS and not e.keywords:
                return norm((name,) + tuple(ex(a) for a in e.args))
            if name in fns and name != 'solution' and not e.keywords:
                h = fns[name]
                if len(h.args.args) != len(e.args):
                    raise Opaque('helper arity')
                pat = body_tree(h, {a.arg: HOLE for a in h.args.args}, fns)
                if names is not None:
                    names[full_skeleton(pat, True)][name] += 1
                return body_tree(h, {a.arg: ex(v) for a, v in zip(h.args.args, e.args)}, fns, names)
        raise Opaque(type(e).__name__)

    for st in fn.body:
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            env[st.targets[0].id] = ex(st.value)
        elif isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name) and type(st.op) in OPS:
            env[st.target.id] = norm((OPS[type(st.op)], ex(st.target), ex(st.value)))
        elif isinstance(st, ast.Return) and st.value is not None:
            t = ex(st.value)
            if size(t) > MAX_NODES:
                raise Opaque('too large')
            return t
        elif isinstance(st, (ast.Import, ast.ImportFrom)):
            continue
        else:
            raise Opaque(type(st).__name__)
    raise Opaque('no return')


def norm(t):
    if t[0] in COMMUTATIVE:
        return (t[0],) + tuple(sorted(t[1:], key=repr))
    return t


def size(t):
    return 0 if t[0] in ('num', '?') else 1 + sum(size(c) for c in t[1:])



def fixed(op, i):
    """Argument positions Loom needs as constants: the exponent of ** and
    round's digits. Patterns keep them verbatim, never as parameters."""
    return i == 1 and op in ('**', 'round')


def skeletons(t, depth):
    """Patterns rooted at t: children cut to holes or expanded to depth."""
    if t[0] == 'num':
        return [HOLE] + ([t] if t[1] in UNITS else [])
    if depth == 0:
        return [HOLE]
    opts = [[c] if fixed(t[0], i) else skeletons(c, depth - 1) for i, c in enumerate(t[1:])]
    out = [(t[0],)]
    for o in opts:
        out = [p + (c,) for p in out for c in o][:64]
    return [norm(p) for p in out]


def full_skeleton(t, bake):
    if t[0] == 'num':
        return t if bake and t[1] in UNITS else HOLE
    return norm((t[0],) + tuple(c if fixed(t[0], i) else full_skeleton(c, bake) for i, c in enumerate(t[1:])))


def subtrees(t):
    if t[0] != 'num':
        yield t
        for c in t[1:]:
            yield from subtrees(c)


def match(p, t, binds):
    if p == HOLE:
        binds.append(t)
        return True
    if p[0] == 'num':
        return t == p
    if t[0] != p[0] or len(t) != len(p):
        return False
    return all(match(pc, tc, binds) for pc, tc in zip(p[1:], t[1:]))


def tile(t, lib_by_op, memo):
    """Fewest calls covering t: (calls, rendering, args) or None for a leaf."""
    if t[0] == 'num':
        return None
    key = id(t)
    if key in memo:
        return memo[key]
    best = None
    for name, p in lib_by_op.get(t[0], []) + [(None, (t[0],) + (HOLE,) * (len(t) - 1))]:
        binds = []
        if not match(p, t, binds):
            continue
        calls, parts, args = 1, [], []
        for b in binds:
            sub = tile(b, lib_by_op, memo)
            if sub is None:
                parts.append(num_text(b[1]))
                args.append(b[1])
            else:
                calls += sub[0]
                parts.append(sub[1])
                args += sub[2]
        render = f"{name}({', '.join(parts)})" if name else prim(t[0], parts)
        if best is None or calls < best[0]:
            best = (calls, render, args, name)
    memo[key] = best
    return best


def prim(op, parts):
    if op in FUNCS:
        return f"{'math.' + op if op in ('floor', 'ceil') else op}({', '.join(parts)})"
    if op == 'neg':
        return f"-({parts[0]})"
    return f"({parts[0]} {op} {parts[1]})"


def num_text(v):
    return f"({v!r})" if v < 0 else repr(v)      # -6 ** 2 would read as -(6 ** 2)


def py_of(name, p):
    """A library fragment as a Loom-subset Python function."""
    n = [0]

    def ren(q):
        if q == HOLE:
            n[0] += 1
            return f"a{n[0] - 1}"
        if q[0] == 'num':
            return num_text(q[1])
        return prim(q[0], [ren(c) for c in q[1:]])
    body = ren(p)
    return f"def {name}({', '.join(f'a{i}' for i in range(n[0]))}):\n    return {body}\n"


def stated_numbers(q):
    vals = set()
    for m in re.findall(r'\d[\d,]*(?:\.\d+)?', q):
        try:
            vals.add(Fraction(m.replace(',', '')))
        except ValueError:
            pass
    pct = {v / 100 for v in vals}
    return vals, vals | pct



OPNAME = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div', '//': 'idiv', '%': 'mod', '**': 'pow', 'neg': 'neg'}


def short_name(p):
    """A readable name from the fragment's shape: add_mul_mul, div_60, ..."""
    def walk(q):
        if q == HOLE:
            return []
        if q[0] == 'num':
            return [str(q[1]).replace('.', 'p').replace('-', 'm')]
        return [OPNAME.get(q[0], q[0])] + [w for c in q[1:] for w in walk(c)]
    return '_'.join(walk(p))[:40]


def formula(p):
    n = [0]

    def ren(q, top=False):
        if q == HOLE:
            n[0] += 1
            return f"a{n[0] - 1}"
        if q[0] == 'num':
            return num_text(q[1])
        if q[0] in FUNCS:
            return f"{q[0]}({', '.join(ren(c, True) for c in q[1:])})"
        if q[0] == 'neg':
            return f"-{ren(q[1])}"
        body = f"{ren(q[1])} {q[0]} {ren(q[2])}"
        return body if top else f"({body})"
    return ren(p, True), n[0]


def hardcodes(src, q):
    """Whether a program writes its answer down instead of computing it: it
    returns a constant, or a literal in it equals the value it returns
    (beyond -1..2) while the question does not state that number."""
    try:
        t = tree_of(src)
    except (Opaque, SyntaxError, RecursionError):
        return False
    if t[0] == 'num':
        return True
    lits = [n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant)
            and type(n.value) in (int, float)]
    try:
        v = value_of(t)
    except (ZeroDivisionError, OverflowError, ValueError, TypeError):
        return False
    strict, _ = stated_numbers(q)
    return v is not None and v not in strict and not -1 <= v <= 2 and \
        any(Fraction(str(x)) in (v, -v) for x in lits)


def value_of(t):
    """The exact value of a tree, or None if it uses something other than
    + - * / and integer powers."""
    if t[0] == 'num':
        return Fraction(str(t[1]))
    if t[0] == 'neg':
        x = value_of(t[1])
        return None if x is None else -x
    xs = [value_of(c) for c in t[1:]]
    if any(x is None for x in xs):
        return None
    if t[0] == '+':
        return sum(xs, Fraction(0))
    if t[0] == '*':
        out = Fraction(1)
        for x in xs:
            out *= x
        return out
    if t[0] == '-':
        return xs[0] - xs[1]
    if t[0] == '/':
        return xs[0] / xs[1]
    if t[0] == '**' and xs[1].denominator == 1 and abs(xs[1]) <= 64:
        return xs[0] ** int(xs[1])
    return None


def qnorm(q):
    return re.sub(r'\s+', ' ', q.lower()).strip()


def split_key(r):
    """What a library-eval split must keep together: a generator family
    (formulas), a DeepMind module with its _composed twin, else the question
    with its numbers masked (AQuA's templated near-duplicates)."""
    meta = r.get('meta', {})
    if r['family'] == 'LOOMD-formulas':
        return f"formulas:{meta['family']}"
    if meta.get('source') == 'mds':
        return f"mds:{meta['module'].removesuffix('_composed')}"
    tpl = re.sub(r'\d[\d,]*(?:\.\d+)?', '#', qnorm(r['state']['question']))
    return 'tpl:' + hashlib.sha256(tpl.encode()).hexdigest()[:16]


def eval_keys(keys, share):
    """Template keys go to eval by hash; family keys (formulas, mds) are few,
    so the first round(share * n) of each kind by hash go."""
    h = lambda k: hashlib.sha256(f'libsplit:{k}'.encode()).hexdigest()
    out = {k for k in keys if k.startswith('tpl:') and int(h(k)[:8], 16) / 16 ** 8 < share}
    for kind in ('formulas:', 'mds:'):
        fam = sorted((k for k in keys if k.startswith(kind)), key=h)
        out.update(fam[:max(1, round(share * len(fam)))] if fam else [])
    return out


WORD = re.compile(r'[a-z]+|\d+(?:\.\d+)?')


def doc_text(f):
    return f"{f['name'].replace('_', ' ')} {f['description']} {f['signature']}"


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        import math
        self.k1, self.b = k1, b
        self.tf = [Counter(WORD.findall(d.lower())) for d in docs]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / max(len(docs), 1)
        df = Counter(w for t in self.tf for w in t)
        self.idf = {w: math.log(1 + (len(docs) - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.post = defaultdict(list)
        for i, t in enumerate(self.tf):
            for w, c in t.items():
                self.post[w].append((i, c))

    def top(self, query, k):
        score = defaultdict(float)
        for w in set(WORD.findall(query.lower())):
            for i, c in self.post.get(w, ()):
                score[i] += self.idf[w] * c * (self.k1 + 1) / (c + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return [i for i, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def chain(expr, names):
    """Library calls in evaluation order (innermost first)."""
    out = []

    def walk(n):
        for c in ast.iter_child_nodes(n):
            walk(c)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in names:
            out.append(n.func.id)
    walk(ast.parse(expr, mode='eval'))
    return out


def arg_spans(q, args):
    """Each literal argument with the first span of the question that states
    it, or states it as a percent (value * 100), else None."""
    found = defaultdict(list)
    for m in re.finditer(r'\d[\d,]*(?:\.\d+)?', q):
        try:
            found[Fraction(m.group().replace(',', ''))].append([m.start(), m.end()])
        except ValueError:
            pass
    out = []
    for x in args:
        v = Fraction(str(x))
        out.append({'value': repr(x), 'span': (found.get(v) or found.get(-v) or [None])[0],
                    'percent_span': (found.get(v * 100) or [None])[0]})
    return out


_LIB = None


def _init(lib_items):
    global _LIB
    by_op = defaultdict(list)
    for name, p in sorted(lib_items, key=lambda kv: -size(kv[1])):
        by_op[p[0]].append((name, p))
    _LIB = by_op


def _tile_one(item):
    q, tree = item
    t = tile(tree, _LIB, {})
    calls, expr, args = (t[0], t[1], t[2]) if t else (0, num_text(tree[1]), [tree[1]])
    strict, lenient = stated_numbers(q)
    fa = [Fraction(str(x)) for x in args]
    return calls, expr, all(x in strict for x in fa), all(x in lenient for x in fa), args


def _loom(chunk):
    from loom_py import score_python_batch
    return [r['status'] if r['status'] == 'ok' else r['abstain_reason'] or 'abstain'
            for r in score_python_batch(chunk, N=64, numbers='rational', max_steps=2_000_000)]


def coverage(tiles, statuses, ntok, progs, rng_seed):
    row = {'programs': len(tiles), 'loom_same_answer': statuses.count('ok'),
           'loom_failures': dict(Counter(s for s in statuses if s != 'ok')),
           'calls_median': statistics.median(t[0] for t in tiles) if tiles else None}
    for k in (1, 2, 3):
        sel = [t for t, s in zip(tiles, statuses) if t[0] <= k and s == 'ok']
        row[f'<= {k} calls'] = {'share': round(len(sel) / max(len(tiles), 1), 4),
                                'args_all_stated': round(sum(t[2] for t in sel) / max(len(sel), 1), 4),
                                'args_all_stated_or_percent': round(sum(t[3] for t in sel) / max(len(sel), 1), 4)}
    rs = random.Random(rng_seed).sample(range(len(tiles)), min(3000, len(tiles)))
    sel3 = [i for i in rs if tiles[i][0] <= 3]
    if sel3:
        row['tokens_call_median'] = statistics.median(ntok(tiles[i][1]) for i in sel3)
        row['tokens_program_median'] = statistics.median(ntok(progs[i]['src']) for i in sel3)
    return row


if __name__ == '__main__':
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument('pools', nargs='+')
    ap.add_argument('--sizes', default='50,200,1000')
    ap.add_argument('--procs', type=int, default=12)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--exclude', action='append', default=[],
                    help='jsonl of held-out problems ({question} or {sha256_norm}): left out of mining and eval')
    ap.add_argument('--eval-share', type=float, default=0.2)
    ap.add_argument('--seen', action='append', default=[],
                    help='file of groups some model already trained on: eval rows get seen_by_final_v1')
    ap.add_argument('--ks', default='1,3,5,10,20')
    ap.add_argument('--no-prune', action='store_true',
                    help='library N = the first N ranked fragments, even those no tiling ever picks')
    a = ap.parse_args()
    outdir = a.outdir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
    excluded = set()
    for path in a.exclude:
        for l in open(path):
            m = json.loads(l)
            excluded.add(m.get('sha256_norm') or hashlib.sha256(qnorm(m['question']).encode()).hexdigest())

    seen_groups = {l.strip() for path in a.seen for l in open(path) if l.strip()}
    seen, progs, n_excluded = set(), [], 0
    for path in a.pools:
        for l in open(path):
            r = json.loads(l)
            if r['group'] in seen:
                continue
            seen.add(r['group'])
            q = r['state']['question']
            if hashlib.sha256(qnorm(q).encode()).hexdigest() in excluded:
                n_excluded += 1
                continue
            cands = [r['program']] + [x for x in r.get('programs_verified', []) if x != r['program']]
            src = next((x for x in cands if not hardcodes(x, q)), None)
            progs.append({'group': r['group'], 'family': r['family'], 'q': q, 'src': src or r['program'],
                          'hardcoded': src is None, 'gold': r['loom']['value'], 'key': split_key(r),
                          'seen_by_final_v1': r['group'] in seen_groups})
    ev = eval_keys({p['key'] for p in progs}, a.eval_share)
    for p in progs:
        p['split'] = 'eval' if p['key'] in ev else 'train'
    with open(os.path.join(outdir, 'library_eval_manifest.jsonl'), 'w') as f, \
            open(os.path.join(outdir, 'library_eval_groups.txt'), 'w') as g:
        for p in progs:
            if p['split'] == 'eval':
                f.write(json.dumps({'group': p['group'], 'family': p['family'], 'split_key': p['key'],
                                    'hardcoded': p['hardcoded'], 'seen_by_final_v1': p['seen_by_final_v1']}) + '\n')
                g.write(p['group'] + '\n')
    split_summary = {'excluded_heldout': n_excluded, 'eval_share': a.eval_share,
                     'eval_seen_by_final_v1': sum(p['split'] == 'eval' and p['seen_by_final_v1'] for p in progs),
                     'answer_written_as_literal': dict(Counter(p['split'] for p in progs if p['hardcoded'])),
                     'eval_family_keys': sorted(k for k in ev if not k.startswith('tpl:')),
                     'programs': dict(Counter((p['family'], p['split']) for p in progs).most_common())}
    split_summary['programs'] = {f'{f} {s}': n for (f, s), n in split_summary['programs'].items()}
    print(json.dumps(split_summary), flush=True)

    # trees: training programs first, so helper names come from training programs only
    opaque = {'train': Counter(), 'eval': Counter()}
    train_helper_names = defaultdict(Counter)     # helper body pattern -> helper names, training programs only
    for part in ('train', 'eval'):
        for p in progs:
            if p['split'] != part:
                continue
            if p['hardcoded']:
                opaque[part]['answer written as a literal'] += 1
                continue
            try:
                p['tree'] = tree_of(p['src'], train_helper_names if part == 'train' else None)
            except (Opaque, SyntaxError, RecursionError) as exc:
                opaque[part][str(exc)[:30]] += 1
    lin = {part: [p for p in progs if p['split'] == part and 'tree' in p] for part in ('train', 'eval')}
    print(json.dumps({'straight_line': {k: len(v) for k, v in lin.items()},
                      'opaque': {k: dict(v.most_common(8)) for k, v in opaque.items()}}), flush=True)

    # mine on training programs only
    df = Counter()
    for p in lin['train']:
        cands = set()
        for s_ in subtrees(p['tree']):
            cands.update(skeletons(s_, 2))
            cands.add(full_skeleton(s_, False))
            cands.add(full_skeleton(s_, True))
        df.update(c for c in cands if c != HOLE and size(c) >= 2)
    sizes = [int(x) for x in a.sizes.split(',')]
    ranked = sorted(df, key=lambda c: (-df[c] * (size(c) - 1), repr(c)))[:max(sizes) * (1 if a.no_prune else 3)]
    names, functions = set(), []
    for rank, p in enumerate(ranked):
        base, helper = short_name(p), None
        if p in train_helper_names:            # a helper's name, if that helper is where most uses come from
            hn, hc = train_helper_names[p].most_common(1)[0]
            if hc >= 0.5 * df[p] and hc >= 0.8 * sum(train_helper_names[p].values()):
                base = helper = hn
        name = base if base not in names else f"{base}_{rank}"
        names.add(name)
        expr, arity = formula(p)
        functions.append({
            'rank': rank, 'name': name, 'uses': df[p], 'ops': size(p), 'arity': arity,
            'signature': f"{name}({', '.join(f'a{i}: number' for i in range(arity))}) -> number",
            'description': f"returns {expr}" + (f" (the {helper.replace('_', ' ')} formula)" if helper else '')
                           + f"; used in {df[p]} training programs",
            'code': py_of(name, p)})
    pattern_of = {f['name']: p for f, p in zip(functions, ranked)}
    top, fams, keys = set(ranked), defaultdict(Counter), defaultdict(set)
    for p in lin['train']:                     # where each function was mined from: families and split keys
        cands = set()
        for s_ in subtrees(p['tree']):
            cands.update(skeletons(s_, 2))
            cands.add(full_skeleton(s_, False))
            cands.add(full_skeleton(s_, True))
        label = p['family'] if p['key'].startswith('tpl:') else f"{p['family']}/{p['key']}"
        for c in cands & top:
            fams[c][label] += 1
            keys[c].add(p['key'])
    for f, p in zip(functions, ranked):
        f['mined_from'] = dict(fams[p].most_common())
        f['split_keys'] = len(keys[p])
        f['eval_split_keys'] = len(keys[p] & ev)

    # library N: N ranked fragments, each picked by some training program's tiling. A fragment a larger one
    # always beats is dropped and the next candidates fill its place, until all N are used (or none are left).
    chosen = {}
    for n_lib in sizes:
        pick, nxt = list(range(min(n_lib, len(ranked)))), n_lib
        while True:
            items = [(functions[i]['name'], ranked[i]) for i in pick]
            with Pool(a.procs, initializer=_init, initargs=(items,)) as pool:
                tiles = pool.map(_tile_one, [(p['q'], p['tree']) for p in lin['train']], chunksize=200)
            used = set().union(*(set(re.findall(r'\b[a-z][a-z0-9_]*(?=\()', t[1])) for t in tiles)) & \
                {nm for nm, _ in items}
            keep = [i for i in pick if functions[i]['name'] in used]
            if a.no_prune or len(keep) == len(pick) or nxt >= len(ranked):
                break
            pick = keep + list(range(nxt, min(len(ranked), nxt + n_lib - len(keep))))
            nxt += n_lib - len(keep)
        chosen[n_lib] = sorted(pick)
        print(json.dumps({'library': n_lib, 'functions': len(pick), 'tiled_by_training': len(keep),
                          'candidates_tried': nxt}), flush=True)

    report = {'programs': len(progs), 'split': split_summary,
              'straight_line': {k: len(v) for k, v in lin.items()},
              'opaque': {k: dict(v) for k, v in opaque.items()}, 'by_library_size': {}}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('google/gemma-4-26B-A4B-it')
    ntok = lambda s_: len(tok(s_, add_special_tokens=False)['input_ids'])
    ks = [int(x) for x in a.ks.split(',')]
    for n_lib in sizes:
        lib_fns = [functions[i] for i in chosen[n_lib]]
        lib_items = [(f['name'], pattern_of[f['name']]) for f in lib_fns]
        code = {f['name']: f['code'] for f in lib_fns}
        row = {'functions': len(lib_fns)}
        for part in ('train', 'eval'):
            ps = lin[part]
            with Pool(a.procs, initializer=_init, initargs=(lib_items,)) as pool:
                tiles = pool.map(_tile_one, [(p['q'], p['tree']) for p in ps], chunksize=200)
            used = [sorted(set(re.findall(r'\b[a-z][a-z0-9_]*(?=\()', t[1])) & set(code), key=lambda u: t[1].index(u))
                    for t in tiles]
            items = [{'src': "import math\n" + ''.join(code[u] for u in us) + f"def solution():\n    return {t[1]}\n",
                      'inputs': {}, 'option_texts': {'g': p['gold']}} for p, t, us in zip(ps, tiles, used)]
            chunks = [items[i:i + 300] for i in range(0, len(items), 300)]
            with Pool(a.procs) as pool:
                statuses = [x for c in pool.map(_loom, chunks, chunksize=1) for x in c]
            row[part] = coverage(tiles, statuses, ntok, ps, n_lib)
            if part == 'eval':
                row['eval_by_family'] = {}
                for fam in sorted({p['family'] for p in ps}):
                    idx = [i for i, p in enumerate(ps) if p['family'] == fam]
                    row['eval_by_family'][fam] = {'programs': len(idx), **{
                        f'<= {k} calls': round(sum(tiles[i][0] <= k and statuses[i] == 'ok' for i in idx) / len(idx), 4)
                        for k in (1, 2, 3)}}
                # (b) baseline: BM25 over name + description + signature; gold = the oracle tiling's functions
                bm = BM25([doc_text(f) for f in lib_fns])
                need = [i for i, us in enumerate(used) if us and statuses[i] == 'ok']
                tops = {i: bm.top(ps[i]['q'], max(ks)) for i in need}
                rec = {}
                for k in ks:
                    hit_all = sum(set(used[i]) <= {lib_fns[j]['name'] for j in tops[i][:k]} for i in need)
                    mean = sum(len(set(used[i]) & {lib_fns[j]['name'] for j in tops[i][:k]}) / len(used[i]) for i in need)
                    rec[f'@{k}'] = {'all_gold_in_top_k': round(hit_all / max(len(need), 1), 4),
                                    'mean_recall': round(mean / max(len(need), 1), 4)}
                row['bm25_retrieval'] = {'queries': len(need), 'needing_no_library_function':
                                         sum(1 for i, us in enumerate(used) if not us and statuses[i] == 'ok'),
                                         'recall': rec}
                with open(os.path.join(outdir, f'library_retrieval_eval_N{n_lib}.jsonl'), 'w') as f:
                    for i in need:
                        f.write(json.dumps({'group': ps[i]['group'], 'family': ps[i]['family'],
                                            'split_key': ps[i]['key'], 'seen_by_final_v1': ps[i]['seen_by_final_v1'], 'query': ps[i]['q'], 'gold': ps[i]['gold'],
                                            'gold_functions': used[i], 'gold_chain': chain(tiles[i][1], set(code)),
                                            'oracle_call': tiles[i][1],
                                            'bm25_top': [lib_fns[j]['name'] for j in tops[i]]}) + '\n')
            with open(os.path.join(outdir, f'library_calls_N{n_lib}{"" if part == "train" else "_eval"}.jsonl'), 'w') as f:
                for p, (calls, expr, st, stp, args), us, status in zip(ps, tiles, used, statuses):
                    f.write(json.dumps({'group': p['group'], 'family': p['family'], 'split': part,
                                        'split_key': p['key'], 'seen_by_final_v1': p['seen_by_final_v1'], 'question': p['q'], 'gold': p['gold'], 'calls': calls,
                                        'call': expr, 'functions': us, 'chain': chain(expr, set(code)),
                                        'args': arg_spans(p['q'], args), 'args_all_stated': st,
                                        'args_all_stated_or_percent': stp, 'loom_status': status,
                                        'loom_same_answer': status == 'ok'}) + '\n')
        report['by_library_size'][n_lib] = row
        print(json.dumps({n_lib: row}), flush=True)
    in_lib = sorted(set().union(*chosen.values()))
    json.dump({**report, 'libraries': {n: [functions[i]['name'] for i in chosen[n]] for n in sizes},
               'functions': [functions[i] for i in in_lib]}, open(os.path.join(outdir, 'library.json'), 'w'), indent=1)
    print(json.dumps({'top': [(f['name'], f['uses'], f['description']) for f in functions[:15]]}, indent=0))
