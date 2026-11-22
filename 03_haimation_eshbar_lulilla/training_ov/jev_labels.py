"""Option labels for any number of options: one ordered list, one mechanism, no K cap.

A-Z, a-z, then the two-letter and three-letter uppercase strings that the Gemma-4 tokenizer encodes as one token
(2,591 labels in all; jev_labels_gemma4.json, sha256[:16] 30435b1faa49af2a). Option j of a rendered question is
labels()[j] for every K; the answer is read from those labels' logits at the answer position. The first 52 are the
letters every earlier jev model was trained on, so K <= 52 renders are unchanged.
"""
import hashlib
import json
import os

LIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jev_labels_gemma4.json")
LIST_SHA16 = "30435b1faa49af2a"
# The VBLL head's reference subspace is spanned by the embedding rows of the first BASIS_LABELS labels (A-Z a-z, the
# letters every head so far was fitted on). It is a basis dimension, not an option limit: any label's row projects onto it,
# and options past it are scored by the same readout.
BASIS_LABELS = 52

_LABELS = None


def labels():
    global _LABELS
    if _LABELS is None:
        with open(LIST_FILE, encoding="utf-8") as f:
            lab = json.load(f)
        got = hashlib.sha256("\n".join(lab).encode()).hexdigest()[:16]
        if got != LIST_SHA16:
            raise SystemExit(f"jev_labels: {LIST_FILE} sha256[:16] {got} != {LIST_SHA16}")
        _LABELS = lab
    return _LABELS


def for_k(k):
    lab = labels()
    if k > len(lab):
        raise SystemExit(f"jev_labels: K = {k} options but {len(lab)} single-token labels; extend the list (same rule)")
    return lab[:k]


def token_ids(tok, k=None):
    """the labels' token ids under `tok`; refuses a tokenizer that splits any of them"""
    lab = labels() if k is None else for_k(k)
    ids = [tok(x, add_special_tokens=False)["input_ids"] for x in lab]
    bad = [x for x, i in zip(lab, ids) if len(i) != 1]
    if bad:
        raise SystemExit(f"jev_labels: {len(bad)} labels are not single tokens for this tokenizer (first: {bad[:5]})")
    out = [i[0] for i in ids]
    if len(set(out)) != len(out):
        raise SystemExit("jev_labels: two labels share a token id")
    return out
