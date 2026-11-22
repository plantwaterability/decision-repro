"""Boolean expressions, from a fresh generator: nested not/and/or over
True/False (and sometimes integer comparisons), 1-5 levels deep, symbolic or
in words. Questions: the value of one expression (True/False), or how many of
k expressions are true (options 0..k)."""
import random


def _expr(rng, depth, words):
    if depth == 0 or rng.random() < 0.25:
        if rng.random() < 0.2:
            a, b = rng.randint(0, 9), rng.randint(0, 9)
            op = rng.choice(['<', '>', '==', '!=', '<=', '>='])
            txt = f"{a} {op} {b}" if not words else f"{a} {dict(zip(['<', '>', '==', '!=', '<=', '>='], ['is less than', 'is greater than', 'equals', 'does not equal', 'is at most', 'is at least']))[op]} {b}"
            return f"({a} {op} {b})", f"({txt})" if words else f"( {txt} )"
        v = rng.choice(['True', 'False'])
        return v, (v.lower() if words else v)
    r = rng.random()
    if r < 0.25:
        p, t = _expr(rng, depth - 1, words)
        return f"(not {p})", (f"not {t}" if words else f"not {t}")
    op = 'and' if r < 0.62 else 'or'
    p1, t1 = _expr(rng, depth - 1, words)
    p2, t2 = _expr(rng, depth - 1, words)
    t = f"({t1} {op} {t2})" if words else f"( {t1} {op} {t2} )"
    return f"({p1} {op} {p2})", t


def make(rng: random.Random) -> dict:
    words = rng.random() < 0.3
    if rng.random() < 0.65:
        p, t = _expr(rng, rng.randint(2, 5), words)
        if p.count('(') < 2:                       # too short to be worth a question (and they repeat)
            return make(rng)
        ans = 1 if eval(p) else 0
        q = rng.choice([f"{t} is", f"Evaluate the boolean expression: {t}", f"What is the value of {t}?"])
        program = f"def solution():\n    return 1 if {p} else 0\n"
        return dict(question=q, program=program, answer=ans, options=['True', 'False'], gold=0 if ans else 1,
                    variants=[2], meta={'kind': 'value', 'words': words, 'length': len(p)})
    k = rng.randint(3, 9)
    exprs = [_expr(rng, rng.randint(1, 4), words) for _ in range(k)]
    ans = sum(1 for p, _ in exprs if eval(p))
    listing = ' '.join(f"({i + 1}) {t}" for i, (_, t) in enumerate(exprs))
    program = "def solution():\n    n = 0\n" + ''.join(f"    if {p}:\n        n += 1\n" for p, _ in exprs) + "    return n\n"
    return dict(question=f"How many of the following {k} boolean expressions are true? {listing}", program=program,
                answer=ans, options=[str(i) for i in range(k + 1)], gold=ans,
                variants=sorted({4, k + 1} if k + 1 > 4 else {k + 1}), meta={'kind': 'count', 'k': k, 'words': words})


TASK = "Boolean expressions deterministic {k}-choice selection"

if __name__ == '__main__':
    rng = random.Random(3)
    for _ in range(4):
        it = make(rng)
        print(it['question'], '->', it['options'][it['gold']], it['meta'])
