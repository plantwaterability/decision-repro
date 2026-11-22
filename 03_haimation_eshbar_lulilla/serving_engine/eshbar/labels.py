import hashlib
import json
import os
LIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'labels.json')
LIST_SHA16 = '30435b1faa49af2a'
_LABELS = None

def labels():
    global _LABELS
    if _LABELS is None:
        with open(LIST_FILE, encoding='utf-8') as f:
            lab = json.load(f)
        got = hashlib.sha256('\n'.join(lab).encode()).hexdigest()[:16]
        if got != LIST_SHA16:
            raise SystemExit(f'labels: {LIST_FILE} sha256[:16] {got} != {LIST_SHA16}')
        _LABELS = lab
    return _LABELS

def for_k(k):
    lab = labels()
    if k > len(lab):
        raise SystemExit(f'labels: K = {k} options but {len(lab)} single-token labels; extend the list (same rule)')
    return lab[:k]

def token_ids(tok, k=None):
    lab = labels() if k is None else for_k(k)
    ids = [tok(x, add_special_tokens=False)['input_ids'] for x in lab]
    bad = [x for x, i in zip(lab, ids) if len(i) != 1]
    if bad:
        raise SystemExit(f'labels: {len(bad)} labels are not single tokens for this tokenizer (first: {bad[:5]})')
    out = [i[0] for i in ids]
    if len(set(out)) != len(out):
        raise SystemExit('labels: two labels share a token id')
    return out
