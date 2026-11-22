from __future__ import annotations
import ast
import json
import os
import re
import time
from . import balanced as _base
TRUTH = {'true', 'false', 'yes', 'no'}
PROGRAM_PROMPT = 'Solve the problem by writing a Python function `solution()` that takes no arguments and returns the final answer: a number, or True/False if the question asks yes/no or true/false. It runs with exact arithmetic (decimals and fractions are exact), so do not add small epsilons or round to guard against float error.\n\nUse only: integers and decimal numbers; + - * / // % and ** with a constant exponent; comparisons; and/or/not; if/elif/else; for loops over range(...); while loops; break/continue; lists of numbers (indexing, len, sum, min, max); helper functions (no recursion); round, abs, min, max, int, float, math.floor, math.ceil.\nDo not use strings, dictionaries, sets, imports other than math, input() or print().\n\nIf the answer is a fraction, return it as an exact division such as 3 / 7 (not a rounded decimal). Do not write comments or docstrings. Use short, clear variable names. Reply with only the code in one ```python block.\n\nProblem: {question}'
_CODE = re.compile('```(?:python)?\\s*\\n(.*?)```', re.S)
CACHE_LEN = 4096
MAX_STEPS = 150000
STORE_BLOCK, STORE_TABLE = ('store_block.safetensors', 'store_table.json')
EXEC_BACKEND = 'interpreter'
STORE_WARM_LENGTHS = (8, 257)
MEM_TAU = 0.18

class _Untemp(ast.NodeTransformer):

    def visit_Name(self, n):
        if re.fullmatch('t\\d+', n.id):
            n.id += '_'
        return n

    def visit_arg(self, n):
        if re.fullmatch('t\\d+', n.arg):
            n.arg += '_'
        return n

def bare(src):
    return ast.unparse(_Untemp().visit(ast.parse(src))) + '\n'

def extract_program(text):
    blocks = _CODE.findall(text or '')
    return blocks[-1].strip() + '\n' if blocks else None

def literal_only(src):
    try:
        m = ast.parse(src)
    except SyntaxError:
        return True
    fns = [n for n in m.body if isinstance(n, ast.FunctionDef) and n.name == 'solution']
    if not fns:
        return True
    body = [n for n in fns[0].body if not (isinstance(n, ast.Expr) and isinstance(getattr(n, 'value', None), ast.Constant))]
    if len(body) == 1 and isinstance(body[0], ast.Return) and (body[0].value is not None):
        v = body[0].value
        return isinstance(v, ast.Constant) or (isinstance(v, ast.UnaryOp) and isinstance(v.operand, ast.Constant))
    return False
from .text import FRAME, problem_text

class RouteEngine(_base.BalancedEngine):

    def __init__(self, model='', route_tau=0.95, route_tau_truth=0.7, route_max_new=320, route='on', mem='on', mem_block=STORE_BLOCK, mem_graph=STORE_TABLE, exec_backend=EXEC_BACKEND, **options):
        super().__init__(model=model, **options)
        self.mem_on = mem == 'on'
        self._mem = None
        if self.mem_on:
            mdir = model if os.path.isdir(model) else os.path.dirname(os.path.abspath(__file__))
            mb, mg = (p if os.path.isabs(p) else os.path.join(mdir, p) for p in (mem_block, mem_graph))
            t0 = time.perf_counter()
            from .store import reader as _store_reader
            self._mem = _store_reader
            self._mem.load(mb, mg)
            self._mem.attach(self.model, self.tok)
            self._mem_warm = self._mem.warm(self.model, self.tok, lengths=range(*STORE_WARM_LENGTHS))
        self.route_tau, self.route_tau_truth, self.route_max_new, self.route_on = (float(route_tau), float(route_tau_truth), int(route_max_new), route == 'on')
        from .runtime import parse_option_number
        self._num = parse_option_number
        self.exec_backend = str(exec_backend)
        self._warm_s = None
        self._s2_log = os.environ.get('STAGE2_ACTIVITY') or None
        if self.route_on and os.environ.get('STAGE2_NO_WARM') != '1':
            self.warm()

    def _eligible(self, q):
        if q.get('type', 'choice') != 'choice':
            return None
        vals = [str(v) for v in q['criteria'].values()]
        if all((v.strip().rstrip('.').lower() in TRUTH for v in vals)):
            return 'truth'
        if all((self._num(v) is not None for v in vals)):
            return 'num'
        return None

    def _decoder(self):
        if getattr(self, '_cache', None) is None:
            from transformers import StaticCache
            m = self.model
            cfg = m.config.get_text_config() if hasattr(m.config, 'get_text_config') else m.config
            self._cache = StaticCache(config=cfg, max_cache_len=CACHE_LEN)
        return self._cache

    def _generate(self, texts):
        import torch
        m, tok = (self.model, self.tok)
        prev_attn = m.config._attn_implementation
        m.set_attn_implementation('sdpa')
        res = []
        try:
            for t in texts:
                chat = tok.apply_chat_template([{'role': 'user', 'content': PROGRAM_PROMPT.format(question=t)}], tokenize=False, add_generation_prompt=True)
                enc = tok(chat, return_tensors='pt', add_special_tokens=False).to(m.device)
                L = enc['input_ids'].shape[1]
                if L + self.route_max_new > CACHE_LEN:
                    res.append((None, 0))
                    continue
                cache = self._decoder()
                cache.reset()
                with torch.no_grad():
                    out = m.generate(**enc, past_key_values=cache, max_new_tokens=self.route_max_new, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
                ids = out[0, L:].tolist()
                n = len(ids)
                while n and ids[n - 1] == (tok.pad_token_id or tok.eos_token_id):
                    n -= 1
                res.append((tok.decode(ids[:n], skip_special_tokens=True), n))
            return res
        finally:
            m.set_attn_implementation(prev_attn)

    def warm(self):
        t0 = time.perf_counter()
        self._generate(['What is 12 times 7, plus 5?'])
        self._warm_s = round(time.perf_counter() - t0, 1)

    def __call__(self, state, questions):
        resp, raw = super().__call__(state, questions)
        if not self.route_on or len(questions or {}) != 1:
            return (resp, raw)
        cand = []
        for qk, q in (questions or {}).items():
            kind = self._eligible(q)
            a = (resp.get('answers') or {}).get(qk)
            if not kind or not a or (not a.get('probabilities')):
                continue
            if max(a['probabilities'].values()) < (self.route_tau if kind == 'num' else self.route_tau_truth):
                cand.append((qk, kind))
        if not cand:
            return (resp, raw)
        tm = time.perf_counter()
        mem_log = []
        if self._mem is not None:
            keep = []
            for qk, kind in cand:
                if kind != 'num':
                    keep.append((qk, kind))
                    continue
                keys = list(questions[qk]['criteria'])
                out = self._mem.read(self.model, self.tok, problem_text(state, questions[qk]), [str(questions[qk]['criteria'][k]) for k in keys])
                rec = {'q': qk, 'abstain': round(float(out.get('abstain', 1.0)), 4)}
                if out.get('choice') is not None and float(out.get('abstain', 1.0)) < MEM_TAU:
                    hit = keys[int(out['choice'])]
                    a = resp['answers'][qk]
                    native = dict(a['probabilities'])
                    rest = sum((v for k, v in native.items() if k != hit))
                    a['choice'], a['probabilities'] = (hit, {k: 0.99 if k == hit else 0.01 * (v / rest if rest > 0 else 1.0 / max(len(native) - 1, 1)) for k, v in native.items()})
                    rec.update(status='mem', choice=hit)
                else:
                    rec['status'] = 'abstain'
                    keep.append((qk, kind))
                mem_log.append(rec)
            cand = keep
        t0 = time.perf_counter()
        if not cand:
            info = {'routed': [], 'mem': mem_log, 'mem_s': round(t0 - tm, 3), 'gen_s': 0.0, 'exec_s': 0.0}
            if isinstance(raw, dict):
                raw['route'] = info
            if self._s2_log:
                with open(self._s2_log, 'a') as f:
                    f.write(json.dumps(info) + '\n')
            return (resp, raw)
        gens = self._generate([problem_text(state, questions[qk]) for qk, _ in cand])
        t1 = time.perf_counter()
        from .runtime import score_python_batch
        jobs, meta = ([], [])
        for (qk, kind), (text, ntok) in zip(cand, gens):
            prog = extract_program(text) if text is not None else None
            try:
                prog = bare(prog) if prog else None
            except SyntaxError:
                prog = None
            rec = {'q': qk, 'kind': kind, 'new_tokens': ntok, 'program': prog}
            if prog is None or literal_only(prog):
                rec['status'] = 'no_program' if prog is None else 'literal'
                meta.append((rec, None))
                continue
            meta.append((rec, len(jobs)))
            jobs.append(({'src': prog, 'inputs': {}, 'option_texts': dict(questions[qk]['criteria'])}, kind == 'truth'))
        outs = [None] * len(jobs)
        for truth in (False, True):
            idx = [i for i, (_, t) in enumerate(jobs) if t == truth]
            if idx:
                r = score_python_batch([jobs[i][0] for i in idx], N=64, numbers='rational', max_steps=MAX_STEPS, boolean_options=truth, backend=self.exec_backend)
                for i, o in zip(idx, r):
                    outs[i] = o
        t2 = time.perf_counter()
        log = []
        for rec, j in meta:
            if j is not None:
                o = outs[j]
                hits = [k for k, v in (o.get('match') or {}).items() if v] if o['status'] == 'ok' else []
                rec['status'] = o['status'] if o['status'] != 'ok' else 'match' if len(hits) == 1 else 'no_match'
                rec['backend'] = o.get('backend')
                if o.get('fallback_reason'):
                    rec['fallback_reason'] = o['fallback_reason']
                if len(hits) == 1:
                    a = resp['answers'][rec['q']]
                    native = dict(a['probabilities'])
                    rest = sum((v for k, v in native.items() if k != hits[0]))
                    probs = {k: 0.99 if k == hits[0] else 0.01 * (v / rest if rest > 0 else 1.0 / max(len(native) - 1, 1)) for k, v in native.items()}
                    rec.update(native_choice=a.get('choice'), route_choice=hits[0])
                    a['choice'], a['probabilities'] = (hits[0], probs)
            log.append(rec)
        info = {'routed': log, 'mem': mem_log, 'mem_s': round(t0 - tm, 3), 'gen_s': round(t1 - t0, 3), 'exec_s': round(t2 - t1, 3)}
        if isinstance(raw, dict):
            raw['route'] = info
        if self._s2_log:
            with open(self._s2_log, 'a') as f:
                f.write(json.dumps(info) + '\n')
        return (resp, raw)
