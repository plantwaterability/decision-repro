"""Differential audit of the pools' verified programs: every program Loom
verified is run again in CPython with exact arithmetic (float literals and
float() as Fractions, / as exact division), and its value must equal the
gold Loom matched. A disagreement is a false verified match (a Loom or
compiler bug, or a Loom/Python semantic difference such as rounding).

CPython runs only in worker processes, each program under a wall-time alarm
and a CPU-time limit, with a memory watchdog: a worker past the limit exits,
and its chunk is retried one program per fresh worker, so a runaway
sum(range(...)) or comprehension is counted, not hung on.

  python audit_verified.py out/arith_gsm8k.jsonl ... [--out out/audit_verified.json]
"""
import argparse
import ast
import json
import os
import resource
import signal
import threading
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from fractions import Fraction

WALL_S, CPU_S, MEM_BYTES = 5, 10, 1 << 30


class _Exact(ast.NodeTransformer):
    def visit_Constant(self, n):
        if isinstance(n.value, float):
            return ast.Call(ast.Name('Fr', ast.Load()), [ast.Constant(repr(n.value))], [])
        return n

    def visit_BinOp(self, n):
        self.generic_visit(n)
        if isinstance(n.op, ast.Div):
            n.left = ast.Call(ast.Name('Fr', ast.Load()), [n.left], [])
        return n

    def visit_AugAssign(self, n):
        self.generic_visit(n)
        if isinstance(n.op, ast.Div):
            n.value = ast.BinOp(ast.Call(ast.Name('Fr', ast.Load()), [ast.Constant(1)], []), ast.Mult(), n.value)
            n.value = ast.Call(ast.Name('Fr', ast.Load()), [n.value], [])
        return n

    def visit_Call(self, n):
        self.generic_visit(n)
        if isinstance(n.func, ast.Name) and n.func.id == 'float':
            n.func = ast.Name('Fr', ast.Load())
        return n


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_S, CPU_S + 1))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEM_BYTES, MEM_BYTES))
    except (ValueError, OSError):
        pass                                   # macOS does not enforce RLIMIT_AS; the watchdog does

    def watchdog():
        import time
        scale = 1 if os.uname().sysname == 'Darwin' else 1024      # ru_maxrss: bytes on macOS, KB on Linux
        while True:
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale > MEM_BYTES:
                os._exit(3)
            time.sleep(0.05)
    threading.Thread(target=watchdog, daemon=True).start()


def _alarm(signum, frame):
    raise TimeoutError


def run_one(src):
    import math
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(WALL_S)
    try:
        ns = {'Fr': Fraction, 'math': math}
        exec(compile(ast.fix_missing_locations(_Exact().visit(ast.parse(src))), '<prog>', 'exec'), ns)
        v = ns['solution']()
        return 'ok', str(Fraction(v)) if isinstance(v, (int, Fraction, bool)) else repr(v)
    except TimeoutError:
        return 'timeout', None
    except MemoryError:
        return 'memory', None
    except Exception as exc:
        return 'python_error', f'{type(exc).__name__}: {str(exc)[:80]}'
    finally:
        signal.alarm(0)


def run_chunk(srcs):
    return [run_one(s) for s in srcs]


def run_all(srcs, procs, chunk=200):
    out = [None] * len(srcs)
    chunks = [list(range(i, min(i + chunk, len(srcs)))) for i in range(0, len(srcs), chunk)]
    retry = []
    with ProcessPoolExecutor(procs, initializer=_limits) as ex:
        futs = [(c, ex.submit(run_chunk, [srcs[i] for i in c])) for c in chunks]
        for c, f in futs:
            try:
                for i, r in zip(c, f.result()):
                    out[i] = r
            except BrokenProcessPool:
                retry += c
    for i in retry:                            # one program per fresh worker: find the one that died
        if out[i] is not None:
            continue
        try:
            with ProcessPoolExecutor(1, initializer=_limits) as ex:
                out[i] = ex.submit(run_one, srcs[i]).result()
        except BrokenProcessPool:
            out[i] = ('killed', None)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('pools', nargs='+')
    ap.add_argument('--procs', type=int, default=12)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    report, bad = {}, []
    for path in a.pools:
        seen, jobs = set(), []
        for l in open(path):
            r = json.loads(l)
            if r['group'] in seen:
                continue
            seen.add(r['group'])
            gold = r['loom']['value']
            for i, p in enumerate(r.get('programs_verified') or [r['program']]):
                jobs.append((r['group'], i, p, gold))
        res = run_all([j[2] for j in jobs], a.procs)
        c = Counter()
        for (g, i, p, gold), (status, v) in zip(jobs, res):
            if status != 'ok':
                c[status] += 1
                bad.append({'pool': os.path.basename(path), 'group': g, 'program_index': i, 'status': status,
                            'detail': v, 'gold': gold})
                continue
            try:
                agree = Fraction(v) == Fraction(gold)
            except (ValueError, ZeroDivisionError):
                agree = False
            c['agree' if agree else 'disagree'] += 1
            if not agree:
                bad.append({'pool': os.path.basename(path), 'group': g, 'program_index': i, 'status': 'disagree',
                            'python': v, 'gold': gold, 'program': p})
        report[os.path.basename(path)] = {'problems': len(seen), 'programs': len(jobs), **dict(c)}
        print(json.dumps({os.path.basename(path): report[os.path.basename(path)]}), flush=True)
    if a.out:
        json.dump({'report': report, 'limits': {'wall_s': WALL_S, 'cpu_s': CPU_S, 'mem_bytes': MEM_BYTES},
                   'not_agreeing': bad}, open(a.out, 'w'), indent=1)
