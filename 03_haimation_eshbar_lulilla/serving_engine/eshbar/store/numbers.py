import re
from fractions import Fraction
_S = {'thousand': 1000.0, 'k': 1000.0, 'million': 1000000.0, 'm': 1000000.0, 'mn': 1000000.0, 'billion': 1000000000.0, 'bn': 1000000000.0, 'b': 1000000000.0, 'trillion': 1000000000000.0, 'tn': 1000000000000.0}
_N = re.compile('\\s*(?P<sign>[-+\\u2212])?\\s*[$\\u20ac\\u00a3]?\\s*(?P<int>\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.(?P<frac>\\d+))?\n                    (?:\\s*/\\s*(?P<den>\\d+))?\\s*(?P<scale>%|percent|thousand|million|billion|trillion|k|m|mn|bn|b|tn)?\\s*', re.X | re.I)

def val(o):
    m = _N.fullmatch(str(o))
    if not m:
        return float('nan')
    x = Fraction(int(m['int'].replace(',', '')))
    if m['frac'] is not None:
        if m['den'] is not None:
            return float('nan')
        x += Fraction(int(m['frac']), 10 ** len(m['frac']))
    if m['den'] is not None:
        if int(m['den']) == 0 or ',' in m['int']:
            return float('nan')
        x /= int(m['den'])
    x *= _S.get((m['scale'] or '').lower(), 1)
    return float(-x if m['sign'] in ('-', '−') else x)

def sim(v, o):
    if v == 0 or o == 0:
        return 1.0 if v == o else 0.0
    if (v > 0) != (o > 0):
        return None
    a, b = (abs(v), abs(o))
    return min(a, b) / max(a, b)

def pick(vals, options, tie=1e-09):
    if not vals:
        return None
    sc = []
    for o in map(val, options):
        if o != o:
            sc.append(float('-inf'))
            continue
        sc.append(sum((w * s for w, s in ((w, sim(v, o)) for v, w in vals) if s is not None)))
    order = sorted(range(len(sc)), key=lambda i: -sc[i])
    if not order or sc[order[0]] <= 0:
        return None
    if len(order) > 1 and sc[order[0]] - sc[order[1]] <= tie:
        return None
    return order[0]
