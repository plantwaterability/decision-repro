from __future__ import annotations
import collections
import hashlib
import json
import numpy as np
from . import render as core
from decision_index.engines.base import Engine, Unsupported
TEST_BACKENDS: dict = {}

def request_id(state, questions):
    return 'eng-' + hashlib.sha1(json.dumps({'state': state, 'questions': questions}, sort_keys=True, ensure_ascii=False, default=str).encode('utf-8')).hexdigest()[:16]

def question_seed(state_text, qid_key, keys):
    h = hashlib.sha256(json.dumps([state_text, qid_key, list(keys)], ensure_ascii=False).encode('utf-8')).hexdigest()
    return int(h[:16], 16)

def option_order(qtype, k, seed):
    if qtype != 'choice' or k < 2:
        return list(range(k))
    rng = np.random.default_rng(seed)
    while True:
        o = [int(x) for x in rng.permutation(k)]
        if o != list(range(k)):
            return o

def state_shown(state, text):
    return state.lstrip() in text

def check_state_shown(state, text, key, max_len):
    if text is None:
        raise Unsupported(f'maximum context length: question {key!r}: the prompt without its state does not fit {max_len} tokens')
    if not state_shown(state, text):
        raise Unsupported(f'maximum context length: question {key!r} does not fit {max_len} tokens without cutting the state (no truncation)')

class PromptEngine(Engine):

    def synchronize(self):
        self._sync()

    def runtime(self):
        try:
            import torch
            return {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        except Exception:
            return {}

    def unit(self, rec, qid):
        q = rec['questions'][qid]
        keys, texts, _ = core.question_options(q, rec['labels'][qid])
        return {'state': rec['state'], 'instructions': q['instructions'], 'qtype': q['type'], 'option_keys': keys, 'option_texts': texts, 'noul_criteria': q.get('criteria') if q['type'] == 'noul' else None}

    def prompt(self, u, qid):
        order = option_order(u['qtype'], len(u['option_keys']), question_seed(u['state'], qid.split('|', 1)[1], u['option_keys']))
        text = core.build_prompt(self.tok, u, order, self.max_len, chat=self.chat, noul_defs=True)
        check_state_shown(u['state'], text, qid.split('|', 1)[1], self.max_len)
        return (order, text)

    def __call__(self, state, questions):
        for k, q in questions.items():
            if q.get('type') not in ('choice', 'noul'):
                raise Unsupported(f"question type {q.get('type')!r}")
            if q['type'] == 'choice' and len(q.get('criteria') or {}) < 2:
                raise Unsupported('a choice needs at least two options')
        rid = request_id(state, questions)
        stats = {'k_hist': collections.Counter(), 'units': 0}
        row = {'state': state, 'questions': questions, 'expected': {}, '_evaluation': {'catalog_id': 57, 'run_id': rid}}
        rec = core.convert_row(row, None, stats, 'text')
        items = []
        for qid in rec['questions']:
            u = self.unit(rec, qid)
            order, text = self.prompt(u, qid)
            items.append((qid, order, text))
        probs = self.run_prompts(items)
        st = collections.Counter()
        answers = {}
        for qkey, q in questions.items():
            a = core.answer_for(rec, qkey, q, probs, st)
            if a is None:
                raise RuntimeError(f'no prediction for question {qkey!r}')
            answers[qkey] = a
        return ({'model': self.name, 'answers': answers}, None)
