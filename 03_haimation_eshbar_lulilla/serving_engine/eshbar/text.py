import json
GENERIC = 'Which option is the correct answer?'
FRAME = '(The question was posed with these instructions for its multiple-choice form: {ins} Follow any part of them that changes what is being asked; the program only computes the value.)'

def problem_text(state, q):
    ins = q.get('instructions')
    ins = json.dumps(ins, ensure_ascii=False) if isinstance(ins, dict) else ins
    if isinstance(state, dict) and isinstance(state.get('question'), str):
        extra = [f'{k}: {(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))}' for k, v in state.items() if k not in ('question', 'task') and v not in (None, '', [], {})]
        t = '\n\n'.join(extra + [state['question']])
        if ins and ins.strip() != GENERIC:
            t += '\n\n' + FRAME.format(ins=ins.strip())
        return t
    parts = [state] if isinstance(state, str) else [v if isinstance(v, str) else json.dumps(v) for k, v in (state or {}).items() if k != 'task']
    if ins and ins.strip() != GENERIC:
        parts.append(ins)
    return '\n\n'.join((p for p in parts if p))
