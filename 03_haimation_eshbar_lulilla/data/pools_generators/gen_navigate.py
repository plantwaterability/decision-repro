"""Navigation, from a fresh generator: turn-based walks, always-face-forward
walks and compass walks of 1-12 instructions. Questions: do you return to the
start (yes/no; returning paths, near misses one step off, and random ones),
which way do you face at the end (turn-based), and how far from the start
(Manhattan distance; distractors are Loom-run wrong programs: turns ignored,
left and right swapped, backward read as forward, the last instruction
dropped, one axis only)."""
import random

DIRS = ['north', 'east', 'south', 'west']
STEP = ["Take {k} step{s}{w}.", "Walk {k} pace{s}{w}.", "Move {k} step{s}{w}.", "Go {k} step{s}{w}."]
TURN = {1: ["Turn right.", "Make a right turn."], 3: ["Turn left.", "Make a left turn."],
        2: ["Turn around.", "Turn to face the opposite way."]}
REL = {0: 'forward', 2: 'backward', 3: 'left', 1: 'right'}


def _walk(ops, mode, mutate=None):
    """ops: ('turn', q) or ('step', k, rel). Returns x, y, d. mutate: how a
    wrong program misreads the walk."""
    x = y = d = 0
    dx, dy = [0, 1, 0, -1], [1, 0, -1, 0]
    seq = ops[:-1] if mutate == 'drop_last' else ops
    for op in seq:
        if op[0] == 'turn':
            q = op[1]
            if mutate == 'ignore_turns':
                continue
            if mutate == 'swap_lr' and q in (1, 3):
                q = 4 - q
            d = (d + q) % 4
        else:
            k, rel = op[1], op[2]
            if mutate == 'backward_forward' and mode != 'compass' and rel == 2:
                rel = 0
            if mutate == 'swap_lr' and rel in (1, 3):
                rel = 4 - rel
            h = rel if mode == 'compass' else (d + rel) % 4
            x, y = x + k * dx[h], y + k * dy[h]
    return x, y, d


def _program(ops, mode, ret, mutate=None):
    lines = ["def solution():", "    x = 0", "    y = 0", "    d = 0", "    dx = [0, 1, 0, -1]", "    dy = [1, 0, -1, 0]"]
    seq = ops[:-1] if mutate == 'drop_last' else ops
    for op in seq:
        if op[0] == 'turn':
            q = op[1]
            if mutate == 'ignore_turns':
                continue
            if mutate == 'swap_lr' and q in (1, 3):
                q = 4 - q
            lines.append(f"    d = (d + {q}) % 4")
        else:
            k, rel = op[1], op[2]
            if mutate == 'backward_forward' and mode != 'compass' and rel == 2:
                rel = 0
            if mutate == 'swap_lr' and rel in (1, 3):
                rel = 4 - rel
            h = str(rel) if mode == 'compass' else ("d" if rel == 0 else f"(d + {rel}) % 4")
            lines += [f"    x += {k} * dx[{h}]", f"    y += {k} * dy[{h}]"]
    lines.append({'return': "    return 1 if x == 0 and y == 0 else 0", 'facing': "    return d",
                  'distance': "    return abs(x) + abs(y)", 'x_only': "    return abs(x)",
                  'y_only': "    return abs(y)"}[ret])
    return '\n'.join(lines) + '\n'


def _text(op, mode, rng):
    if op[0] == 'turn':
        return rng.choice(TURN[op[1]])
    k, rel = op[1], op[2]
    w = (' ' + DIRS[rel]) if mode == 'compass' else ('' if mode == 'turns' and rel == 0 else ' ' + REL[rel])
    return rng.choice(STEP).format(k=k, s='' if k == 1 else 's', w=w)


def make(rng: random.Random) -> dict:
    mode = rng.choice(['turns', 'forward', 'compass'])
    n = rng.randint(1, 12)
    ops = []
    for _ in range(n):
        if mode == 'turns' and rng.random() < 0.4:
            ops.append(('turn', rng.choice([1, 3, 1, 3, 2])))
        else:
            rel = 0 if mode == 'turns' else rng.randrange(4)
            ops.append(('step', rng.randint(1, 10), rel))
    kind = rng.random()
    if kind < 0.6:                                           # walk back to the start
        x, y, d = _walk(ops, mode)
        back = []
        if mode == 'turns':
            for target, amount in ((0 if y < 0 else 2, abs(y)), (1 if x < 0 else 3, abs(x))):
                if amount:
                    q = (target - d) % 4
                    if q:
                        back.append(('turn', q))
                    back.append(('step', amount, 0))
                    d = target
        else:
            if y:
                back.append(('step', abs(y), 0 if y < 0 else 2))
            if x:
                back.append(('step', abs(x), 1 if x < 0 else 3))
            if mode == 'compass':
                rng.shuffle(back)
        ops += back
        if kind < 0.12 and back:                             # a near miss: one step count off by one
            i = rng.choice([j for j, o in enumerate(ops) if o[0] == 'step'])
            k = ops[i][1] + rng.choice([-1, 1])
            if k >= 1:
                ops[i] = ('step', k, ops[i][2])
    if len(ops) > 16:
        return make(rng)
    head = "Always face forward. " if mode == 'forward' else ("You start facing north. " if mode == 'turns' and rng.random() < 0.5 else "")
    body = ' '.join(_text(op, mode, rng) for op in ops)
    x, y, d = _walk(ops, mode)
    qtype = rng.choices(['return', 'facing', 'distance'], weights=[0.55, 0.2 if mode == 'turns' else 0, 0.25])[0]
    meta = {'mode': mode, 'instructions': len(ops), 'question': qtype}
    if qtype == 'return':
        q = rng.choice(["If you follow these instructions, do you return to the starting point?",
                        "After following these instructions, are you back where you started?"])
        ans = 1 if x == 0 and y == 0 else 0
        return dict(question=f"{q} {head}{body}" if rng.random() < 0.5 else f"{head}{body} {q}",
                    program=_program(ops, mode, 'return'), answer=ans, options=['Yes', 'No'], gold=0 if ans else 1,
                    variants=[2], meta=meta)
    if qtype == 'facing':
        q = "You start facing north. " if not head else ""
        return dict(question=f"{q}{head}{body} Which direction are you facing at the end?",
                    program=_program(ops, mode, 'facing'), answer=d, options=list(DIRS), gold=d, variants=[4], meta=meta)
    dist = abs(x) + abs(y)
    dis, seen = [], {dist}
    for mut, ret in (('ignore_turns', 'distance'), ('swap_lr', 'distance'), ('backward_forward', 'distance'),
                     ('drop_last', 'distance'), (None, 'x_only'), (None, 'y_only')):
        if mut == 'ignore_turns' and mode != 'turns':
            continue
        mx, my, _ = _walk(ops, mode, mut)
        v = abs(mx) + abs(my) if ret == 'distance' else abs(x) if ret == 'x_only' else abs(y)
        if v not in seen:
            seen.add(v)
            dis.append((_program(ops, mode, ret, mut), v, str(v)))
    if len(dis) < 3:
        return make(rng)
    q = rng.choice(["How many steps from the starting point do you end up, counting north-south and east-west "
                    "distance separately and adding them?", "What is the Manhattan distance between where you start "
                    "and where you end up, in steps?"])
    return dict(question=f"{head}{body} {q}", program=_program(ops, mode, 'distance'), answer=dist, options=[str(dist)],
                gold=0, variants=[4, 10], distractors=dis, meta=meta)


TASK = "Navigation deterministic {k}-choice selection"

if __name__ == '__main__':
    rng = random.Random(2)
    for _ in range(4):
        it = make(rng)
        print(it['question'], '->', it['options'][it['gold']], it['meta'])
