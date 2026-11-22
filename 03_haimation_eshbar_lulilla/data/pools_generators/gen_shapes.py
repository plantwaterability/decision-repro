"""SVG path shapes, from a fresh generator: polygons with 3 and 5-8 vertices,
rectangles (also rotated), kites, trapezoids, lines, sectors, circles and
ellipses. Shapes are built exactly on a grid of hundredths; the program
lists the path's points (in hundredths) and classifies the shape from its
distinct vertices, side lengths, right angles, parallel sides and arcs."""
import math
import random

CLASSES = ['circle', 'ellipse', 'heptagon', 'hexagon', 'kite', 'line', 'octagon', 'pentagon', 'rectangle', 'sector',
           'trapezoid', 'triangle']
C = {c: i for i, c in enumerate(CLASSES)}
POLY = {3: 'triangle', 5: 'pentagon', 6: 'hexagon', 7: 'heptagon', 8: 'octagon'}
DIRS = [(1, 0), (0, 1), (1, 1), (1, 2), (2, 1), (3, 4), (4, 3), (1, 3), (3, 1), (2, 3), (3, 2), (5, 12), (12, 5)]


def classify(x, y, lines, arcs, rx, ry):
    """The program's rule, in Python: the generator's own answer."""
    if arcs > 0:
        if lines > 0:
            return C['sector']
        return C['circle'] if rx == ry else C['ellipse']
    pts = []
    for p in zip(x, y):
        if p not in pts:
            pts.append(p)
    n = len(pts)
    if n == 2:
        return C['line']
    if n == 4:
        e = [(pts[(i + 1) % 4][0] - pts[i][0], pts[(i + 1) % 4][1] - pts[i][1]) for i in range(4)]
        if all(e[i][0] * e[(i + 1) % 4][0] + e[i][1] * e[(i + 1) % 4][1] == 0 for i in range(4)):
            return C['rectangle']
        s = [a * a + b * b for a, b in e]
        if (s[0] == s[1] and s[2] == s[3]) or (s[1] == s[2] and s[3] == s[0]):
            return C['kite']
        if e[0][0] * e[2][1] - e[0][1] * e[2][0] == 0 or e[1][0] * e[3][1] - e[1][1] * e[3][0] == 0:
            return C['trapezoid']
        return -1
    return C[POLY[n]] if n in POLY else -1


PROGRAM = """def solution():
    x = {x}
    y = {y}
    lines = {lines}
    arcs = {arcs}
    rx = {rx}
    ry = {ry}
    if arcs > 0:
        if lines > 0:
            return {sector}
        if rx == ry:
            return {circle}
        return {ellipse}
    ux = {zeros}
    uy = {zeros}
    n = 0
    for i in range(len(x)):
        new = 1
        for j in range(n):
            if ux[j] == x[i] and uy[j] == y[i]:
                new = 0
        if new == 1:
            ux[n] = x[i]
            uy[n] = y[i]
            n += 1
    if n == 2:
        return {line}
    if n == 3:
        return {triangle}
    if n == 4:
        right = 1
        s = [0, 0, 0, 0]
        for i in range(4):
            ax = ux[(i + 1) % 4] - ux[i]
            ay = uy[(i + 1) % 4] - uy[i]
            bx = ux[(i + 2) % 4] - ux[(i + 1) % 4]
            by = uy[(i + 2) % 4] - uy[(i + 1) % 4]
            if ax * bx + ay * by != 0:
                right = 0
            s[i] = ax * ax + ay * ay
        if right == 1:
            return {rectangle}
        if s[0] == s[1] and s[2] == s[3] or (s[1] == s[2] and s[3] == s[0]):
            return {kite}
        c02 = (ux[1] - ux[0]) * (uy[3] - uy[2]) - (uy[1] - uy[0]) * (ux[3] - ux[2])
        c13 = (ux[2] - ux[1]) * (uy[0] - uy[3]) - (uy[2] - uy[1]) * (ux[0] - ux[3])
        if c02 == 0 or c13 == 0:
            return {trapezoid}
        return -1
    if n == 5:
        return {pentagon}
    if n == 6:
        return {hexagon}
    if n == 7:
        return {heptagon}
    if n == 8:
        return {octagon}
    return -1
"""


def _fmt(v):
    return f"{v / 100:.2f}"


def _pt(p):
    return f"{_fmt(p[0])},{_fmt(p[1])}"


def _inside(pts, lo=200, hi=9800):
    return all(lo <= a <= hi and lo <= b <= hi for a, b in pts)


def _simple_turns(pts):
    n = len(pts)
    for i in range(n):
        a, b, c = pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) == 0:
            return False
    return len(set(pts)) == n


def _polygon(rng, n):
    while True:
        cx, cy = rng.randint(3000, 7000), rng.randint(3000, 7000)
        r = rng.randint(1500, 3000)
        angles = sorted(rng.uniform(0, 2 * math.pi) for _ in range(n))
        gaps = [(angles[(i + 1) % n] - angles[i]) % (2 * math.pi) for i in range(n)]
        if min(gaps) < 0.35 * 2 * math.pi / n:
            continue
        pts = [(round(cx + r * rng.uniform(0.6, 1.3) * math.cos(t)), round(cy + r * rng.uniform(0.6, 1.3) * math.sin(t)))
               for t in angles]
        if _inside(pts) and _simple_turns(pts):
            return pts


def _frame(rng):
    dx, dy = rng.choice(DIRS)
    if rng.random() < 0.5:
        dx, dy = -dy, dx
    if rng.random() < 0.5:
        dx, dy = -dx, -dy
    return (dx, dy), (-dy, dx), math.hypot(dx, dy)


def _quad(rng, kind):
    for _ in range(1000):
        d, nrm, ln = _frame(rng)
        P0 = (rng.randint(1500, 8500), rng.randint(1500, 8500))
        add = lambda p, v, k: (p[0] + k * v[0], p[1] + k * v[1])
        if kind == 'rectangle':
            a, b = rng.randint(int(800 / ln) + 1, int(4000 / ln)), rng.randint(int(800 / ln) + 1, int(4000 / ln))
            if a == b:
                continue
            pts = [P0, add(P0, d, a), add(add(P0, d, a), nrm, b), add(P0, nrm, b)]
        elif kind == 'kite':
            a, b, w = (rng.randint(int(600 / ln) + 1, int(3500 / ln)) for _ in range(3))
            if a == b:
                continue
            pts = [add(P0, d, a), add(P0, nrm, w), add(P0, d, -b), add(P0, nrm, -w)]
        else:
            L1, L2 = rng.randint(int(1000 / ln) + 1, int(4500 / ln)), rng.randint(int(600 / ln) + 1, int(3500 / ln))
            h = rng.randint(int(800 / ln) + 1, int(3500 / ln))
            s = rng.randint(-L2 // 2, max(L1 - L2 // 2, 1))
            if L1 == L2:
                continue
            P3 = add(add(P0, d, s), nrm, h)
            pts = [P0, add(P0, d, L1), add(P3, d, L2), P3]
        if rng.random() < 0.5:
            pts = pts[::-1]
        r = rng.randrange(4)
        pts = pts[r:] + pts[:r]
        if _inside(pts):
            return pts
    raise RuntimeError(kind)


def _path(rng, pts, closed=True):
    style = rng.random()
    seq = pts + [pts[0]] if closed else pts
    if style < 0.45 or not closed:
        return 'M ' + ' L '.join(_pt(p) for p in seq), seq
    if style < 0.75:                                         # BBH style: one move per segment
        out, xs = [], []
        for a, b in zip(seq, seq[1:]):
            out.append(f"M {_pt(a)} L {_pt(b)}")
            xs += [a, b]
        return ' '.join(out), xs
    return 'M ' + ' L '.join(_pt(p) for p in pts) + ' Z', list(pts)


def make(rng: random.Random) -> dict:
    cls = rng.choices(CLASSES, weights=[1, 1, 1.2, 1.2, 1.5, 0.6, 1.2, 1.2, 1.5, 1.2, 1.5, 1.2])[0]
    lines, arcs, rx, ry = 0, 0, 0, 0
    if cls in ('circle', 'ellipse', 'sector'):
        cx, cy = rng.randint(3000, 7000), rng.randint(3000, 7000)
        r1 = rng.randint(800, 2500)
        r2 = r1 if cls != 'ellipse' else r1 + rng.choice([-1, 1]) * rng.randint(200, 700)
        if cls == 'sector':
            t1 = rng.uniform(0, 2 * math.pi)
            t2 = t1 + rng.uniform(0.4, 5.2)
            c, p1 = (cx, cy), (round(cx + r1 * math.cos(t1)), round(cy + r1 * math.sin(t1)))
            p2 = (round(cx + r1 * math.cos(t2)), round(cy + r1 * math.sin(t2)))
            large = 1 if (t2 - t1) > math.pi else 0
            d = f"M {_pt(c)} L {_pt(p1)} A {_fmt(r1)},{_fmt(r1)} 0.00 {large},1 {_pt(p2)} L {_pt(c)}"
            seq, lines, arcs, rx, ry = [c, p1, p2, c], 2, 1, r1, r1
        else:
            rot = 0 if cls == 'circle' else rng.choice([0, 0, 15, 30, 45, 60, 90, 120])
            t = rng.uniform(0, 2 * math.pi)
            p = (round(cx + r1 * math.cos(t)), round(cy + r2 * math.sin(t)))
            q = (round(cx - r1 * math.cos(t)), round(cy - r2 * math.sin(t)))
            d = (f"M {_pt(p)} A {_fmt(r1)},{_fmt(r2)} {rot:.2f} 1,0 {_pt(q)} "
                 f"A {_fmt(r1)},{_fmt(r2)} {rot:.2f} 1,0 {_pt(p)}")
            seq, arcs, rx, ry = [p, q, p], 2, r1, r2
    elif cls == 'line':
        while True:
            pts = [(rng.randint(500, 9500), rng.randint(500, 9500)) for _ in range(2)]
            if pts[0] != pts[1]:
                break
        d, seq = _path(rng, pts, closed=False)
    elif cls in ('rectangle', 'kite', 'trapezoid'):
        d, seq = _path(rng, _quad(rng, cls))
    else:
        n = {v: k for k, v in POLY.items()}[cls]
        d, seq = _path(rng, _polygon(rng, n))
    x, y = [p[0] for p in seq], [p[1] for p in seq]
    answer = classify(x, y, lines, arcs, rx, ry)
    if answer != C[cls]:                                      # an accidental square, rhombus or parallelogram
        return make(rng)
    program = PROGRAM.format(x=x, y=y, lines=lines, arcs=arcs, rx=rx, ry=ry, zeros=[0] * max(len(x), 4), **C)
    q = rng.choice([
        f'This SVG path element <path d="{d}"/> draws a',
        f'What shape does the SVG path <path d="{d}"/> draw?',
        f'An SVG image contains the element <path d="{d}"/>. Which shape does it draw?',
        f'Render the path data "{d}" as an SVG path. What figure does it trace?',
        f'<svg><path d="{d}"/></svg> Which shape is drawn?',
    ])
    return dict(question=q, program=program, answer=answer, options=list(CLASSES), gold=answer, variants=[4, 10],
                meta={'shape': cls, 'points': len(seq), 'path_style': 'arc' if arcs else
                      ('split' if d.count('M') > 1 else 'Z' if d.endswith('Z') else 'polyline')})


TASK = "Geometric shapes deterministic {k}-choice selection"

if __name__ == '__main__':
    rng = random.Random(1)
    for _ in range(4):
        it = make(rng)
        print(it['question'], '->', CLASSES[it['answer']], it['meta'])
