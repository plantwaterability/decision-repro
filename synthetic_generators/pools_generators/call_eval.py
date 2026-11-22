"""(c) and (d) for the mined library: a model writes one call expression per
library-eval problem, and Loom runs it; it must return the verified answer.

  (c) oracle:   the model is offered exactly the functions the oracle tiling
                uses (items whose tiling uses at least one).
  (d) retrieve: the model is offered the top-k functions a retriever
                returned for the question: BM25 over name + description +
                signature by default, or another retriever's lists
                ({group, functions} jsonl, e.g. KGMem's) with --retrieved.
  none:         no functions at all, only the primitives: the control that
                says whether the library helps.

Three steps, so generation can run wherever the model is:
  python call_eval.py prompts --library out/library.json --n 200 \\
      --items out/library_calls_N200_eval.jsonl --mode oracle|retrieve [--k 5] [--retrieved R.jsonl] --out P.jsonl
  python call_eval.py generate --prompts P.jsonl --endpoint http://host:8000/v1 --model NAME --out R.jsonl
      (any OpenAI-compatible server, greedy; sample_arith.py --prompt raw reads P.jsonl too)
  python call_eval.py score --prompts P.jsonl --replies R.jsonl --library out/library.json --out S.jsonl

score reports execution accuracy and counts retrieval misses (a gold
function not offered), replies that are not one expression, unknown
functions or names, signature errors (wrong arity), Loom abstains by reason
(compile_error = an unsupported op, overflow, step_limit, no_match, ...),
and verified matches whose call states the gold value as a literal the
question does not (the model computed the answer itself, so the match says
nothing about the call). Matching is exact: Fractions, and integers are
never compared through floats. Per-item results let net flips against
another system's answers be counted.
"""
import argparse
import ast
import json
import os
import re
import sys
from collections import Counter
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mine_library import BM25, FUNCS, doc_text, stated_numbers

PROMPT = """Answer the problem by writing one Python expression that computes the answer. Arithmetic is exact (decimals and fractions are exact).

You may call these library functions:

{functions}
You may also use numbers, + - * / // %, ** with a constant exponent, round, abs, min, max, int, float, math.floor and math.ceil. Prefer the library functions. Do not write the final answer as a number: compute it from the numbers in the problem.

Reply with only the expression in one ```python block.

Problem: {question}"""


# The pool-sampling prompt (sample_arith.py's STEM variant): a whole program, loops allowed.
PROGRAM_PROMPT = """Solve the problem by writing a Python function `solution()` that takes no arguments and returns the final answer: a number, or True/False if the question asks yes/no or true/false. It runs with exact arithmetic (decimals and fractions are exact), so do not add small epsilons or round to guard against float error.

Use only: integers and decimal numbers; + - * / // % and ** with a constant exponent; comparisons; and/or/not; if/elif/else; for loops over range(...); while loops; break/continue; lists of numbers (indexing, len, sum, min, max); helper functions (no recursion); round, abs, min, max, int, float, math.floor, math.ceil.
Do not use strings, dictionaries, sets, imports other than math, input() or print().

If the answer is a fraction, return it as an exact division such as 3 / 7 (not a rounded decimal). Do not write comments or docstrings. Use short, clear variable names. Reply with only the code in one ```python block.

Problem: {question}"""

# The same, asking for the shortest program that computes the answer (decode tokens are the latency budget).
SHORT_PROMPT = PROGRAM_PROMPT.replace(
    "Use short, clear variable names.",
    "Write the shortest correct program: no comments, no prints, no unused variables, a single return "
    "expression when that is clear. Compute the answer from the numbers in the problem; do not return it "
    "as a number you worked out yourself.")
assert SHORT_PROMPT != PROGRAM_PROMPT


def load_library(path, n):
    """Library n: the listed names when library.json has 'libraries' (pruned
    cuts), else the first n functions."""
    d = json.load(open(path))
    by_name = {f['name']: f for f in d['functions']}
    names = d.get('libraries', {}).get(str(n)) or [f['name'] for f in d['functions'][:n]]
    return {x: by_name[x] for x in names}


def cmd_prompts(a):
    lib = load_library(a.library, a.n)
    names = list(lib)
    items = [json.loads(l) for l in open(a.items)]
    items = [it for it in items if it.get('loom_same_answer', True)]
    retrieved = {}
    if a.retrieved:
        retrieved = {r['group']: r['functions'] for r in map(json.loads, open(a.retrieved))}
    bm = BM25([doc_text(lib[x]) for x in names]) if a.mode == 'retrieve' and not a.retrieved else None
    n = 0
    with open(a.out, 'w') as f:
        for it in items:
            gold_fns = it.get('functions') or it.get('gold_functions') or []
            if a.mode == 'oracle':
                if not gold_fns:
                    continue
                offered = list(gold_fns)
            elif a.mode == 'none':
                offered = []
            elif a.retrieved:
                offered = [x for x in retrieved.get(it['group'], [])[:a.k] if x in lib]
            else:
                offered = [names[j] for j in bm.top(it['question'], a.k)]
            listing = ''.join(lib[x]['code'] + '\n' for x in offered) if offered else '(none)\n\n'
            q = it['question']
            f.write(json.dumps({'pid': it['group'], 'group': it['group'], 'family': it['family'], 'mode': a.mode,
                                'k': a.k, 'question_text': q, 'gold': it['gold'], 'gold_functions': gold_fns,
                                'offered': offered, 'question': PROMPT.format(functions=listing, question=q)}) + '\n')
            n += 1
    print(json.dumps({'prompts': n, 'mode': a.mode, 'n_library': a.n, 'k': a.k}))


def cmd_generate(a):
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    prompts = [json.loads(l) for l in open(a.prompts)]
    done = {json.loads(l)['pid'] for l in open(a.out)} if os.path.exists(a.out) else set()
    todo = [p for p in prompts if p['pid'] not in done]

    def one(p):
        body = json.dumps({'model': a.model, 'messages': [{'role': 'user', 'content': p['question']}],
                           'temperature': 0, 'max_tokens': a.max_tokens}).encode()
        req = urllib.request.Request(a.endpoint.rstrip('/') + '/chat/completions', body,
                                     {'Content-Type': 'application/json',
                                      **({'Authorization': f'Bearer {a.api_key}'} if a.api_key else {})})
        r = json.load(urllib.request.urlopen(req, timeout=600))
        c = r['choices'][0]
        return {'pid': p['pid'], 'replies': [c['message']['content']], 'finish': [c.get('finish_reason')]}

    with open(a.out, 'a') as f, ThreadPoolExecutor(a.threads) as ex:
        for res in ex.map(one, todo):
            f.write(json.dumps(res) + '\n')
    print(json.dumps({'generated': len(todo), 'already': len(done)}))


def extract(reply):
    """The reply's one expression, or (None, why)."""
    blocks = re.findall(r'```(?:python)?[ \t]*\n(.*?)```', reply, re.S)
    text = (blocks[-1] if blocks else reply).strip()
    try:
        return ast.unparse(ast.parse(text, mode='eval').body), None
    except SyntaxError:
        pass
    try:
        mod = ast.parse(text)
    except SyntaxError:
        return None, 'syntax_error'
    body = [s for s in mod.body if not isinstance(s, (ast.Import, ast.ImportFrom))]
    if len(body) == 1 and isinstance(body[0], (ast.Expr, ast.Return, ast.Assign)) and body[0].value is not None:
        return ast.unparse(body[0].value), None
    return None, 'not_one_expression'


def check_names(expr, offered, lib):
    """None, or why the expression cannot be run against the offered functions."""
    tree = ast.parse(expr, mode='eval')
    callees = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == 'math' \
                    and fn.attr in ('floor', 'ceil'):
                continue
            if not isinstance(fn, ast.Name):
                return 'unknown_function'
            if fn.id in offered:
                if node.keywords or len(node.args) != lib[fn.id]['arity']:
                    return 'signature_error'
            elif fn.id in lib:
                return 'function_not_offered'
            elif fn.id not in FUNCS:
                return 'unknown_function'
        elif isinstance(node, ast.Name) and node.id != 'math' and id(node) not in callees:
            return 'unknown_name'
    return None


def literals(expr):
    out = []
    for node in ast.walk(ast.parse(expr, mode='eval')):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            out.append(Fraction(repr(node.value)) if isinstance(node.value, float) else Fraction(node.value))
    return out


def cmd_score(a):
    from loom_py import score_python_batch
    lib = load_library(a.library, a.n)
    prompts = {p['pid']: p for p in map(json.loads, open(a.prompts))}
    replies = {r['pid']: r for r in map(json.loads, open(a.replies))}
    res, jobs = [], []
    for pid, p in prompts.items():
        r = {'group': pid, 'family': p['family'], 'mode': p['mode'], 'gold': p['gold'],
             'gold_functions': p['gold_functions'], 'offered': p['offered'],
             'retrieval_miss': not set(p['gold_functions']) <= set(p['offered'])}
        res.append(r)
        if pid not in replies:
            r['status'] = 'no_reply'
            continue
        reply = replies[pid]['replies'][0] or ''
        r['finish'] = (replies[pid].get('finish') or [None])[0]
        expr, why = extract(reply)
        r['call'] = expr
        if expr is None:
            r['status'] = why
            continue
        why = check_names(expr, set(p['offered']), lib)
        if why:
            r['status'] = why
            continue
        used = [x for x in p['offered'] if re.search(rf'\b{re.escape(x)}\(', expr)]
        r['functions_used'] = used
        src = "import math\n" + ''.join(lib[x]['code'] for x in used) + f"def solution():\n    return {expr}\n"
        jobs.append((r, {'src': src, 'inputs': {}, 'option_texts': {'g': p['gold']}}))
        strict, _ = stated_numbers(p['question_text'])
        g = Fraction(p['gold'])
        r['gold_literal'] = any(x == g for x in literals(expr)) and g not in strict
    for i in range(0, len(jobs), 500):
        chunk = jobs[i:i + 500]
        for (r, _), out in zip(chunk, score_python_batch([j for _, j in chunk], N=64, numbers='rational',
                                                         max_steps=2_000_000)):
            r['status'] = 'ok' if out['status'] == 'ok' else out['abstain_reason'] or 'abstain'
            r['value'] = out.get('value')
    with open(a.out, 'w') as f:
        for r in res:
            r['correct'] = r['status'] == 'ok'
            f.write(json.dumps(r) + '\n')
    n = len(res)
    ok = [r for r in res if r['correct']]
    summary = {'items': n, 'mode': res[0]['mode'] if res else None, 'n_library': a.n,
               'accuracy': round(len(ok) / max(n, 1), 4),
               'accuracy_without_gold_literal_matches': round(sum(not r.get('gold_literal') for r in ok) / max(n, 1), 4),
               'gold_literal_matches': sum(bool(r.get('gold_literal')) for r in ok),
               'retrieval_misses': sum(r['retrieval_miss'] for r in res),
               'correct_despite_retrieval_miss': sum(r['retrieval_miss'] for r in ok),
               'statuses': dict(Counter(r['status'] for r in res).most_common()),
               'by_family': {fam: round(sum(r['correct'] for r in res if r['family'] == fam) /
                                        sum(r['family'] == fam for r in res), 4)
                             for fam in sorted({r['family'] for r in res})}}
    print(json.dumps(summary))
    json.dump(summary, open(a.out.removesuffix('.jsonl') + '_summary.json', 'w'), indent=1)


TRUTH = {'true', 'false', 'yes', 'no'}
COMPUTES = (ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.Call, ast.For, ast.While, ast.If, ast.IfExp,
            ast.AugAssign, ast.Subscript, ast.ListComp, ast.GeneratorExp)


def literal_only(program):
    """Whether a program computes nothing: it only returns a literal, directly
    or through names bound to literals (`return True`, the reasoning in
    comments). -2 is a literal; 3 > 2 is a computation."""
    tree = ast.parse(program)
    return not any(isinstance(n, COMPUTES) and not (isinstance(n, ast.UnaryOp) and isinstance(n.operand, ast.Constant))
                   for n in ast.walk(tree))


def cmd_rowprompts(a):
    """Primitives-only prompts (the control's) for DI rows: one row per group,
    the K=4 variant when there is one. With --groups, only those groups, and
    --sample draws that many, split as evenly as the families allow (a
    family smaller than its share gives the rest to the others)."""
    import random
    want = {l.strip() for l in open(a.groups) if l.strip()} if a.groups else None
    rows = {}
    for path in a.rows:
        for l in open(path):
            r = json.loads(l)
            g = r['group']
            if want is not None and g not in want:
                continue
            k = len(r['questions']['answer']['criteria'])
            if g not in rows or (k == 4) > (len(rows[g]['questions']['answer']['criteria']) == 4):
                rows[g] = r
    groups = sorted(rows)
    if a.sample and a.sample < len(groups):
        rng = random.Random(a.seed)
        fam = {}
        for g in groups:
            fam.setdefault(rows[g]['family'], []).append(g)
        quota, left = {}, a.sample
        for f in sorted(fam, key=lambda f: len(fam[f])):      # smallest first; leftovers roll to the rest
            quota[f] = min(len(fam[f]), left // (len(fam) - len(quota)))
            left -= quota[f]
        groups = sorted(g for f in fam for g in rng.sample(fam[f], quota[f]))
        print(json.dumps({'sample': {f: quota[f] for f in sorted(fam)}}))
    with open(a.out, 'w') as f:
        for g in groups:
            r = rows[g]
            f.write(json.dumps({'pid': f"{a.tag}|{g}", 'group': g, 'family': r['family'],
                                'track': (r.get('metadata') or {}).get('track'), 'id': r.get('id'),
                                'question_text': r['state']['question'],
                                'options': r['questions']['answer']['criteria'],
                                'expected': r['expected']['answer'],
                                'template': a.template,
                                'question': ({'program': PROGRAM_PROMPT, 'short': SHORT_PROMPT}[a.template].format(
                                                 question=r['state']['question'])
                                             if a.template in ('program', 'short') else
                                             PROMPT.format(functions='(none)\n\n', question=r['state']['question']))})
                    + '\n')
    print(json.dumps({'prompts': len(groups)}))


def cmd_route(a):
    """The program route on DI rows: the reply's expression runs on Loom, and
    its value must equal exactly one of the row's options (truth words read
    as 1/0). Per item: program, status, value, matched option, correct and
    the reply's new tokens."""
    from loom_py import score_python_batch
    prompts = [json.loads(l) for l in open(a.prompts)]
    replies = {r['pid']: r for r in map(json.loads, open(a.replies))}
    res, jobs = [], []
    for p in prompts:
        r = {'group': p['group'], 'family': p['family'], 'track': p.get('track'), 'id': p.get('id'),
             'options': p['options'], 'expected': p['expected']}
        res.append(r)
        rep = replies.get(p['pid'])
        if rep is None:
            r['status'] = 'no_reply'
            continue
        r['new_tokens'] = (rep.get('new_tokens') or [None])[0]
        r['finish'] = (rep.get('finish') or [None])[0]
        if p.get('template') in ('program', 'short'):
            import loom_gemma
            from common import bare
            prog = loom_gemma.extract_program(rep['replies'][0] or '')
            try:
                r['program'] = bare(prog) if prog else None
            except SyntaxError:
                r['program'] = None
            if r['program'] is None:
                r['status'] = 'no_program' if prog is None else 'syntax_error'
                continue
        else:
            expr, why = extract(rep['replies'][0] or '')
            r['call'] = expr
            if expr is None:
                r['status'] = why
                continue
            why = check_names(expr, set(), {})
            if why:
                r['status'] = why
                continue
            r['program'] = f"import math\ndef solution():\n    return {expr}\n"
        truth = all(str(t).strip().rstrip('.').lower() in TRUTH for t in p['options'].values())
        jobs.append((r, truth, {'src': r['program'], 'inputs': {}, 'option_texts': p['options']}))
    for truth in (False, True):
        part = [j for j in jobs if j[1] == truth]
        for i in range(0, len(part), 500):
            chunk = part[i:i + 500]
            outs = score_python_batch([j for _, _, j in chunk], N=64, numbers='rational', max_steps=2_000_000,
                                      boolean_options=truth)
            for (r, _, _), out in zip(chunk, outs):
                r['status'] = 'ok' if out['status'] == 'ok' else out['abstain_reason'] or 'abstain'
                r['value'] = out.get('value')
                hits = [k for k, v in (out.get('match') or {}).items() if v]
                r['matched_option'] = hits[0] if len(hits) == 1 else None
    with open(a.out, 'w') as f:
        for r in res:
            r['literal_only'] = bool(r.get('program')) and literal_only(r['program'])
            if r['literal_only']:               # not an answer: the program computes nothing
                r['matched_option_literal'], r['matched_option'] = r.get('matched_option'), None
            r['matches_an_option'] = r.get('matched_option') is not None
            r['correct'] = r.get('matched_option') == r['expected']
            f.write(json.dumps(r) + '\n')
    n = len(res)
    by = lambda key: {v: {'items': sum(r.get(key) == v for r in res),
                          'answered': round(sum(r['matches_an_option'] for r in res if r.get(key) == v) /
                                            max(sum(r.get(key) == v for r in res), 1), 4),
                          'correct': round(sum(r['correct'] for r in res if r.get(key) == v) /
                                           max(sum(r.get(key) == v for r in res), 1), 4)}
                      for v in sorted({r.get(key) for r in res}, key=str)}
    toks = sorted(r['new_tokens'] for r in res if r.get('new_tokens') is not None)
    answered = [r for r in res if r['matches_an_option']]
    summary = {'items': n, 'answered (value equals one option)': round(len(answered) / max(n, 1), 4),
               'literal_only (not counted as answered)': sum(r['literal_only'] for r in res),
               'correct': round(sum(r['correct'] for r in res) / max(n, 1), 4),
               'precision when answered': round(sum(r['correct'] for r in answered) / max(len(answered), 1), 4),
               'statuses': dict(Counter(r['status'] for r in res).most_common()),
               'new_tokens': {'median': toks[len(toks) // 2] if toks else None,
                              'p80': toks[int(len(toks) * 0.8)] if toks else None,
                              'max': toks[-1] if toks else None},
               'by_family': by('family'), 'by_track': by('track')}
    print(json.dumps(summary))
    json.dump(summary, open(a.out.removesuffix('.jsonl') + '_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prompts')
    p.add_argument('--library', required=True)
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--items', required=True)
    p.add_argument('--mode', choices=['oracle', 'retrieve', 'none'], required=True)
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--retrieved')
    p.add_argument('--out', required=True)
    p = sub.add_parser('generate')
    p.add_argument('--prompts', required=True)
    p.add_argument('--endpoint', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--api-key', default=os.environ.get('OPENAI_API_KEY', ''))
    p.add_argument('--max-tokens', type=int, default=256)
    p.add_argument('--threads', type=int, default=32)
    p.add_argument('--out', required=True)
    p = sub.add_parser('score')
    p.add_argument('--prompts', required=True)
    p.add_argument('--replies', required=True)
    p.add_argument('--library', required=True)
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--out', required=True)
    p = sub.add_parser('rowprompts')
    p.add_argument('--rows', nargs='+', required=True)
    p.add_argument('--groups')
    p.add_argument('--sample', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tag', default='route')
    p.add_argument('--template', choices=['expression', 'program', 'short'], default='expression',
                   help='expression: the control prompt (one expression, primitives only); '
                        'program: the pool-sampling prompt (a whole def solution(), loops allowed); '
                        'short: program, asking for the shortest program that computes the answer')
    p.add_argument('--out', required=True)
    p = sub.add_parser('route')
    p.add_argument('--prompts', required=True)
    p.add_argument('--replies', required=True)
    p.add_argument('--out', required=True)
    a = ap.parse_args()
    {'prompts': cmd_prompts, 'generate': cmd_generate, 'score': cmd_score, 'rowprompts': cmd_rowprompts,
     'route': cmd_route}[a.cmd](a)
