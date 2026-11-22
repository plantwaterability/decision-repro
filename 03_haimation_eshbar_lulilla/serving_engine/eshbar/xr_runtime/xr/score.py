# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.

from fractions import Fraction
from functools import lru_cache
from typing import Dict, List, Optional, Union
from xr.cc import ExecutionFault, compile_c, fix_value
from xr.isa import ISAInterpreter
Value = Union[int, float, List[int]]

def _store(memory: List[int], addr: int, value: int, N: int) -> None:
    value &= (1 << N) - 1
    memory[addr] = value - (1 << N) if value >= 1 << N - 1 else value

def _fix_in(memory: List[int], addr: int, value, N: int, numbers: str) -> bool:
    q = Fraction(repr(value)) if isinstance(value, float) else Fraction(value)
    if numbers == 'rational':
        if max(abs(q.numerator), q.denominator) >= 1 << N - 1:
            return False
        memory[addr], memory[addr + 1] = (q.numerator, q.denominator)
        return True
    raw = q * (1 << N)
    if raw.denominator != 1 or not -(1 << 2 * N - 1) <= raw < 1 << 2 * N - 1:
        return False
    raw = int(raw)
    _store(memory, addr, raw >> N, N)
    _store(memory, addr + 1, raw & (1 << N) - 1, N)
    return True

def _fix_out(final: List[int], addr: int, N: int, numbers: str) -> Fraction:
    if numbers == 'rational':
        return Fraction(final[addr], final[addr + 1])
    return Fraction(final[addr]) + Fraction(final[addr + 1] & (1 << N) - 1, 1 << N)

def run_function(source: str, inputs: Optional[Dict[str, Value]]=None, N: int=16, backend: str='interpreter', s: int=64, m: int=128, max_steps: int=1000000, trace: bool=False, numbers: str='binary', device: str='cpu') -> Dict[str, object]:
    raise ValueError('run_function is not included in this runtime')
import re
from fractions import Fraction
_NUMBER = re.compile('\n    \\s*\n    (?P<sign>[+-])?\n    \\s*\n    (?P<cur>\\$)?\n    (?P<int>\\d{1,3}(?:,\\d{3})+|\\d+)       # 1200 or 1,200 (well-formed groups only)\n    (?:\\.(?P<frac>\\d+))?\n    (?:\\s*/\\s*(?P<den>\\d+))?               # a/b\n    \\s*$\n', re.VERBOSE)

def parse_option_number(text: str) -> Optional[Union[int, Fraction]]:
    if not isinstance(text, str):
        return None
    m = _NUMBER.fullmatch(text)
    if m is None:
        return None
    value = Fraction(int(m['int'].replace(',', '')))
    if m['frac'] is not None:
        if m['den'] is not None:
            return None
        value += Fraction(int(m['frac']), 10 ** len(m['frac']))
    if m['den'] is not None:
        if int(m['den']) == 0 or ',' in m['int']:
            return None
        value /= int(m['den'])
    if m['sign'] == '-':
        value = -value
    return int(value) if value.denominator == 1 else value
_TRUTH = {'true': 1, 'yes': 1, 'false': 0, 'no': 0}

def parse_option_truth(text: str) -> Optional[int]:
    if not isinstance(text, str):
        return None
    return _TRUTH.get(text.strip().rstrip('.').strip().lower())
RUNTIME_VERSION, RUNTIME_DIFF_SHA256 = ('1.1.0', None)
_PREPARED: Dict[tuple, tuple] = {}
_PREPARED_PROGRAM: Dict[tuple, object] = {}

@lru_cache(maxsize=1024)
def _compiled_checked(source: str, s: int, m: int, N: int, numbers: str='binary'):
    hit = _PREPARED.pop((source, s, m, N, numbers), None)
    if hit is not None:
        return hit
    cfg, memory, commands, meta = compile_c(source, s=s, m=m, n=4096, N=N, checked=True, numbers=numbers)
    return (cfg, tuple(memory), tuple(commands), meta)

def _prepare_remote(kind: str, src: str, entry: Optional[str], s: int, m: int, N: int, numbers: str, pack: bool):
    c_src = src
    if kind == 'python':
        from xr.front import PythonSubsetError, python_to_c
        try:
            c_src = python_to_c(src, entry)[0]
        except PythonSubsetError:
            return ('redo', None, None, None)
    try:
        compiled = _compiled_checked(c_src, s, m, N, numbers)
    except Exception:
        return ('redo', c_src, None, None)
    program = None
    if pack:
        import xr.gpu
        program = xr.gpu.Program(compiled[2], compiled[3]['checks'])
    return ('ok', c_src, compiled, program)
_POOL = None

def precompile(jobs: List[tuple], workers: int, pack: bool) -> List[Optional[str]]:
    global _POOL
    import concurrent.futures
    import multiprocessing
    if _POOL is None or _POOL._max_workers != workers:
        _POOL = concurrent.futures.ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('forkserver'))
    out: List[Optional[str]] = []
    for job, (status, c_src, compiled, program) in zip(jobs, _POOL.map(_prepare_remote, *zip(*[(*j, pack) for j in jobs]))):
        kind, src, entry, s, m, N, numbers = job
        out.append(c_src)
        if status == 'ok':
            _PREPARED[c_src, s, m, N, numbers] = compiled
            if program is not None:
                _PREPARED_PROGRAM[c_src, s, m, N, numbers] = program
    return out

def score_options(program: str, inputs: Dict[str, Union[int, float]], option_texts: Dict[str, str], *, N: int, result: str='return', backend: str='interpreter', s: int=64, m: int=128, max_steps: int=1000000, rounding_slack: Optional[Union[int, float, str, Fraction]]=None, numbers: str='binary', boolean_options: bool=False) -> Dict[str, object]:
    backend, why = _resolve_backend(backend)
    job = _Scoring(program, inputs, option_texts, N=N, result=result, backend=backend, s=s, m=m, max_steps=max_steps, rounding_slack=rounding_slack, numbers=numbers, boolean_options=boolean_options)
    if job.done:
        return _note(job.out, why)
    if job.backend in ('cuda', 'cuda-transformer'):
        return _note(_run_cuda([job])[0], why)
    if job.backend not in ('interpreter', 'transformer'):
        raise ValueError(f"unknown backend '{job.backend}'")
    return _note(job.execute(), why)
_AUTO: Optional[tuple] = None

def _resolve_backend(backend: str) -> tuple:
    global _AUTO
    if backend != 'auto':
        return (backend, None)
    if _AUTO is None:
        try:
            import xr.gpu
            if not xr.gpu.available():
                raise RuntimeError('no CUDA device')
            xr.gpu._module()
            _AUTO = ('cuda', None)
        except Exception as exc:
            _AUTO = ('interpreter', f'{type(exc).__name__}: {str(exc)[:300]}')
    return _AUTO

def _note(result: Dict[str, object], why: Optional[str]) -> Dict[str, object]:
    if why is not None:
        result['fallback_reason'] = why
    return result

def score_options_batch(calls: List[Dict[str, object]], **common) -> List[Dict[str, object]]:
    workers = common.pop('compile_workers', 0)
    common['backend'], why = _resolve_backend(common.get('backend', 'interpreter'))
    if workers and workers > 1 and (len(calls) > 1):
        keys = []
        for call in calls:
            kw = {**common, **call}
            keys.append(('c', kw['program'], None, kw.get('s', 64), kw.get('m', 128), kw['N'], kw.get('numbers', 'binary')))
        precompile(keys, workers, pack=common.get('backend') == 'cuda')
    jobs = []
    for call in calls:
        kw = {**common, **call}
        program, inputs, options = (kw.pop('program'), kw.pop('inputs', {}), kw.pop('option_texts'))
        jobs.append(_Scoring(program, inputs or {}, options, **kw))
    results = [job.out if job.done else None for job in jobs]
    cuda = [k for k, job in enumerate(jobs) if not job.done and job.backend in ('cuda', 'cuda-transformer')]
    for k, r in zip(cuda, _run_cuda([jobs[k] for k in cuda])):
        results[k] = r
    for k, job in enumerate(jobs):
        if results[k] is None:
            if job.backend not in ('interpreter', 'transformer'):
                raise ValueError(f"unknown backend '{job.backend}'")
            results[k] = job.execute()
    return [_note(r, why) for r in results]

@lru_cache(maxsize=1024)
def _cuda_program(source: str, s: int, m: int, N: int, numbers: str):
    import xr.gpu
    hit = _PREPARED_PROGRAM.pop((source, s, m, N, numbers), None)
    if hit is not None:
        return hit
    cfg, memory, commands, meta = _compiled_checked(source, s, m, N, numbers)
    return xr.gpu.Program(commands, meta['checks'])

def _run_cuda(jobs: List['_Scoring']) -> List[Dict[str, object]]:
    import xr.gpu
    groups: Dict[tuple, List[int]] = {}
    for k, job in enumerate(jobs):
        groups.setdefault((job.cfg.s, job.cfg.m, job.cfg.N, job.max_steps, job.cfg.log_frac_bits), []).append(k)
    results: List[Optional[Dict[str, object]]] = [None] * len(jobs)
    for (s, m, N, max_steps, F), ks in groups.items():
        runs = xr.gpu.run_batch(s, m, N, [(jobs[k].memory, _cuda_program(jobs[k].program, jobs[k].s, jobs[k].m, N, jobs[k].numbers)) for k in ks], max_steps=max_steps, log_frac_bits=F)
        for k, run in zip(ks, runs):
            job = jobs[k]
            if run['status'] == 'fault':
                job.out['steps'] = run['steps']
                results[k] = job.finish(run['fault'].reason, run['fault'].detail)
            elif run['status'] == 'step_limit':
                job.out['steps'] = max_steps
                results[k] = job.finish('step_limit', f'no HALT within {max_steps} steps')
            else:
                results[k] = job.complete(run['memory'], run['steps'], run['inexact'])
    return results

class _Scoring:

    def __init__(self, program: str, inputs, option_texts, *, N: int, result: str='return', backend: str='interpreter', s: int=64, m: int=128, max_steps: int=1000000, rounding_slack=None, numbers: str='binary', boolean_options: bool=False):
        import hashlib
        import time
        if backend == 'cuda-transformer' and N > 16:
            raise ValueError('the transformer is exact only up to N=16')
        fallback = None
        if backend == 'cuda' and N > 32 and (numbers != 'rational'):
            backend, fallback = ('interpreter', 'the GPU backend checks binary fix arithmetic only up to N=32')
        self.program, self.inputs, self.option_texts = (program, inputs, option_texts)
        self.N, self.result, self.backend, self.s, self.m = (N, result, backend, s, m)
        self.max_steps, self.rounding_slack, self.numbers = (max_steps, rounding_slack, numbers)
        self.done = True
        self.out: Dict[str, object] = {'status': 'abstain', 'abstain_reason': None, 'fault_detail': None, 'value': None, 'value_raw': None, 'parsed': {}, 'match': {}, 'match_kind': None, 'steps': 0, 'latency_s': 0.0, 'inexact_ops': {'int_div_remainder': 0, 'fix_rounding': 0}, 'backend': backend, 'N': N, 'program_sha256': hashlib.sha256(program.encode()).hexdigest(), 'runtime_version': RUNTIME_VERSION, 'runtime_diff_sha256': RUNTIME_DIFF_SHA256}
        if fallback:
            self.out['fallback_reason'] = fallback
        self.t0 = time.perf_counter()
        out = self.out
        parsed = {key: parse_option_number(text) for key, text in option_texts.items()}
        truth = {key: parse_option_truth(text) for key, text in option_texts.items()}
        self.booleans = boolean_options and bool(truth) and all((v is not None for v in truth.values()))
        if self.booleans:
            parsed = truth
        self.parsed = parsed
        out['parsed'] = {k: None if v is None else int(v) if isinstance(v, int) else float(v) for k, v in parsed.items()}
        try:
            self.cfg, memory, self.commands, self.meta = _compiled_checked(program, s, m, N, numbers)
        except ExecutionFault as fault:
            self.finish(fault.reason, fault.detail)
            return
        except (SyntaxError, NameError, KeyError, ValueError, MemoryError, RuntimeError) as exc:
            self.finish('compile_error', f'{type(exc).__name__}: {exc}')
            return
        memory = list(memory)
        variables, fixes = (self.meta['variables'], self.meta['fix_variables'])
        lo, hi = (-(1 << N - 1), (1 << N - 1) - 1)
        for name, value in inputs.items():
            if name not in variables or (name not in fixes and self.meta['sizes'].get(name, 1) != 1):
                self.finish('compile_error', f"no scalar global '{name}'")
                return
            addr = variables[name]
            if name in fixes:
                if not _fix_in(memory, addr, value, N, numbers):
                    self.finish('input_out_of_domain', f'{name} = {value}')
                    return
            else:
                if isinstance(value, bool) or not isinstance(value, int) or (not lo <= value <= hi):
                    self.finish('input_out_of_domain', f'{name} = {value!r}')
                    return
                memory[addr] = value
        if result != 'return' and result not in variables:
            self.finish('compile_error', f"no global '{result}'")
            return
        self.memory = memory
        self.done = False

    def finish(self, reason=None, detail=None) -> Dict[str, object]:
        import time
        self.out['abstain_reason'] = reason
        self.out['fault_detail'] = detail
        self.out['status'] = 'ok' if reason is None else 'abstain'
        self.out['latency_s'] = time.perf_counter() - self.t0
        self.done = True
        return self.out

    def execute(self) -> Dict[str, object]:
        cfg, max_steps = (self.cfg, self.max_steps)
        vm = ISAInterpreter(cfg.s, cfg.m, cfg.N, self.memory, list(self.commands), cfg.log_frac_bits, checks=self.meta['checks'])
        try:
            for steps in range(max_steps):
                if vm.step():
                    break
            else:
                self.out['steps'] = max_steps
                return self.finish('step_limit', f'no HALT within {max_steps} steps')
        except ExecutionFault as fault:
            self.out['steps'] = vm.executed
            return self.finish(fault.reason, fault.detail)
        return self.complete(vm.memory, steps + 1, vm.inexact)

    def complete(self, final, steps: int, inexact: Dict[str, int]) -> Dict[str, object]:
        out, N, numbers, meta = (self.out, self.N, self.numbers, self.meta)
        variables, fixes, result = (meta['variables'], meta['fix_variables'], self.result)
        parsed, option_texts = (self.parsed, self.option_texts)
        final = [int(v) for v in final]
        out['steps'] = steps
        out['inexact_ops'] = {'int_div_remainder': inexact['int_div_remainder'], 'fix_rounding': inexact['fix_rounding'] + meta['fix_literal_rounding']}
        if self.backend in ('transformer', 'cuda-transformer'):
            hw = run_function(self.program, self.inputs, N=N, backend='transformer', s=self.s, m=self.m, max_steps=self.max_steps, numbers=numbers, device='cuda' if self.backend == 'cuda-transformer' else 'cpu')
            names = [k for k in variables if not k.startswith('__')]
            if any((hw[k] != ((_fix_out(final, variables[k], N, numbers) if numbers == 'rational' else fix_value(final[variables[k]], final[variables[k] + 1], N)) if k in fixes else final[variables[k]:variables[k] + meta['sizes'][k]] if meta['sizes'].get(k, 1) > 1 else final[variables[k]]) for k in names)):
                raise RuntimeError('transformer and interpreter disagree')
        if result == 'return':
            value = Fraction(final[meta['retval_addr']])
        elif result in fixes:
            a = variables[result]
            out['value_raw'] = (final[a], final[a + 1])
            value = _fix_out(final, a, N, numbers)
        else:
            value = Fraction(final[variables[result]])
        out['value'] = int(value) if value.denominator == 1 else float(value)
        if self.booleans and value not in (0, 1):
            out['match'] = {key: False for key in parsed}
            return self.finish('no_match', f"result {out['value']} is not a truth value (0 or 1)")
        if any((v is None for v in parsed.values())):
            return self.finish('unparseable_options', ', '.join(sorted((repr(t) for k, t in option_texts.items() if parsed[k] is None))))
        out['match'] = {key: v == value for key, v in parsed.items()}
        kind = 'exact'
        if self.rounding_slack is not None and out['inexact_ops']['fix_rounding'] > 0:
            rs = self.rounding_slack
            slack = Fraction(repr(rs) if isinstance(rs, float) else rs)
            out['match'] = {key: abs(v - value) <= slack for key, v in parsed.items()}
            kind = 'within_slack'
        hits = sum(out['match'].values())
        if hits == 0:
            return self.finish('no_match')
        if hits > 1:
            return self.finish('multi_match')
        out['match_kind'] = kind
        return self.finish()
