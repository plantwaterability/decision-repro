"""Belief tracking (theory of mind) for object placement, from a fresh
generator (not MuSR, ToMi or Hi-ToM text). 3-6 agents share a room with 3-8
places and 1-4 objects; everyone starts in the room and sees where things
are. 4-15 events: an agent leaves or comes back, an agent moves an object,
and (for first-order and real-place questions) someone present is
distracted and misses a move. Whoever is in the room and paying attention
sees a move. Questions: where an object really is; where an agent thinks it
is (first order); where an agent thinks another agent thinks it is (second
order: A updates its picture of B only for moves both saw); who else saw a
given move. About two thirds of the belief questions are false beliefs (the
answer is not the object's real place). The program tracks presence,
attention and each agent's beliefs as integer lists; its answer is
cross-checked against the generator's own simulation."""
import random

from gen_tracking import NAMES

SCENES = [
    ('kitchen', ['drawer', 'pantry', 'fridge', 'cupboard', 'fruit bowl', 'oven', 'bread bin', 'sink cabinet', 'spice rack'],
     ['keys', 'phone', 'wallet', 'lemon', 'recipe card', 'mug', 'lunchbox', 'sunglasses', 'charger']),
    ('office', ['desk drawer', 'filing cabinet', 'bookshelf', 'supply closet', 'safe', 'briefcase', 'printer tray',
                'mail tray', 'coat closet'],
     ['contract', 'stapler', 'USB drive', 'badge', 'notebook', 'calculator', 'envelope', 'headphones', 'pen case']),
    ('workshop', ['toolbox', 'workbench', 'pegboard', 'parts bin', 'shelf', 'cabinet', 'tackle box', 'bucket', 'crate'],
     ['hammer', 'wrench', 'tape measure', 'drill bit', 'level', 'flashlight', 'chisel', 'glue gun', 'screwdriver']),
    ('library', ['returns cart', 'reference desk', 'card catalog', 'locker', 'reading table', 'archive box',
                 'display case', 'book drop', 'back office'],
     ['atlas', 'rare map', 'logbook', 'magnifier', 'reading lamp', 'manuscript', 'stamp', 'folder', 'bookmark']),
    ('backstage area', ['prop table', 'costume rack', 'trunk', 'dressing room', 'wings', 'cue desk', 'wardrobe',
                        'stage box', 'basket'],
     ['crown', 'script', 'fake sword', 'wig', 'lantern', 'cape', 'scroll', 'mask', 'goblet']),
    ('dorm room', ['backpack', 'closet', 'desk', 'laundry basket', 'nightstand', 'shoebox', 'bookcase', 'window sill',
                   'drawer'],
     ['textbook', 'laptop', 'scarf', 'headphones', 'water bottle', 'ID card', 'snack bag', 'camera', 'umbrella']),
    ('greenhouse', ['potting bench', 'seed tray', 'watering can', 'shed', 'compost bin', 'shelf', 'wheelbarrow', 'pot',
                    'cold frame'],
     ['trowel', 'gloves', 'seed packet', 'pruning shears', 'label maker', 'twine', 'sprayer', 'notebook', 'thermometer']),
]
MOVE = ["{a} moves the {x} to the {l}.", "{a} puts the {x} in the {l}.", "{a} takes the {x} and stashes it in the {l}.",
        "{a} slips the {x} into the {l}.", "{a} relocates the {x} to the {l}."]
MOVE_PROSE = ["{a} quietly tucked the {x} into the {l}.", "Wanting it out of the way, {a} carried the {x} over to the {l}.",
              "Thinking it would be safer there, {a} moved the {x} to the {l}.",
              "{a} picked up the {x} and set it down in the {l}."]
LEAVE = ["{a} leaves the room.", "{a} steps out.", "{a} goes outside."]
LEAVE_PROSE = ["{a} stepped out to take a phone call.", "{a} headed off to grab something from the car.",
               "{a} excused themselves and left for a few minutes.", "{a} wandered out to get some air."]
BACK = ["{a} comes back.", "{a} returns to the room.", "{a} walks back in."]
BACK_PROSE = ["A little later, {a} came back in.", "{a} returned, none the wiser.", "{a} walked back into the {room}."]
MISS = ["{a} is in the room but is on the phone and does not see this.", "{a} is there but has their back turned and misses it.",
        "{a} is reading and does not notice."]
MISS_PROSE = ["{a} was right there, but buried in a phone call and never noticed.",
              "{a} had their back turned the whole time and missed it.", "{a}, absorbed in a book, did not see a thing."]
PLURAL = {'keys', 'sunglasses', 'headphones', 'pruning shears', 'gloves'}
FILLER = ["It had been a long day.", "Outside, it had started to rain.", "Someone put the kettle on.",
          "Everyone was getting ready for the evening.", "There was a lot to do before the deadline."]


def _join(xs):
    return ', '.join(xs[:-1]) + ', and ' + xs[-1] if len(xs) > 2 else ' and '.join(xs)


def make(rng: random.Random) -> dict:
    room, places, things = rng.choice(SCENES)
    NA, NL = rng.randint(3, 6), rng.randint(3, 8)
    NO = rng.randint(1, 4)
    qkind = rng.choices(['actual', 'first', 'second', 'saw'], weights=[0.15, 0.35, 0.3, 0.2])[0]
    if qkind == 'second':
        NO = min(NO, 96 // (NA * NA))          # the agent x agent x object table must fit Loom's memory
    agents, locs, objs = rng.sample(NAMES, NA), rng.sample(places, NL), rng.sample(things, NO)
    track1, track2 = qkind in ('first', 'second'), qkind == 'second'
    distract_ok = qkind in ('first', 'actual')
    prose = rng.random() < 0.6
    init = [rng.randrange(NL) for _ in range(NO)]
    loc, here = list(init), [1] * NA
    bel = [list(init) for _ in range(NA)]
    bel2 = [[list(init) for _ in range(NA)] for _ in range(NA)]
    lines, code, moves = [], [], []
    for _ in range(rng.randint(4, 15)):
        r = rng.random()
        present = [a for a in range(NA) if here[a]]
        absent = [a for a in range(NA) if not here[a]]
        if r < 0.22 and len(present) > 1:
            a = rng.choice(present)
            here[a] = 0
            lines.append(rng.choice(LEAVE_PROSE if prose else LEAVE).format(a=agents[a]))
            code.append(f"    here[{a}] = 0")
        elif r < 0.36 and absent:
            a = rng.choice(absent)
            here[a] = 1
            lines.append(rng.choice(BACK_PROSE if prose else BACK).format(a=agents[a], room=room))
            code.append(f"    here[{a}] = 1")
        else:
            a = rng.choice(present)
            x = rng.randrange(NO)
            l = rng.choice([k for k in range(NL) if k != loc[x]])
            others = [b for b in present if b != a]
            missed = [rng.choice(others)] if distract_ok and others and rng.random() < 0.3 else []
            loc[x] = l
            watch = [1 if here[b] and b not in missed else 0 for b in range(NA)]
            for b in range(NA):
                if watch[b]:
                    bel[b][x] = l
                    for c in range(NA):
                        if watch[c]:
                            bel2[b][c][x] = l
            text = rng.choice(MOVE_PROSE if prose else MOVE).format(a=agents[a], x=objs[x], l=locs[l])
            if missed:
                text += ' ' + rng.choice(MISS_PROSE if prose else MISS).format(a=agents[missed[0]])
            lines.append(text)
            code.append(f"    loc[{x}] = {l}")
            if track1 or qkind == 'saw':
                code.append("    see = [" + ', '.join("0" if b in missed else f"here[{b}]" for b in range(NA)) + "]")
            if track1:
                code += [f"    for b in range({NA}):", "        if see[b] == 1:", f"            bel[b * {NO} + {x}] = {l}"]
            if track2:
                code += [f"            for c in range({NA}):", "                if see[c] == 1:",
                         f"                    bel2[(b * {NA} + c) * {NO} + {x}] = {l}"]
            moves.append({'mover': a, 'x': x, 'l': l, 'watch': watch, 'code_at': len(code)})
        if prose and rng.random() < 0.15:
            lines.append(rng.choice(FILLER))
    if not moves:
        return make(rng)

    # the question
    meta = {'scene': room, 'agents': NA, 'places': NL, 'objects': NO, 'events': len(lines), 'question': qkind,
            'prose': prose}
    options, variants = list(locs), sorted({NL, 4} if NL > 4 else {NL})
    false_belief = rng.random() < 0.67
    if qkind == 'actual':
        x = rng.randrange(NO)
        q = rng.choice([f"Where is the {objs[x]} really?", f"Where is the {objs[x]} at the end?"])
        ret, answer, gold = f"loc[{x}]", loc[x], loc[x]
    elif qkind == 'first':
        cands = [(a, x) for a in range(NA) for x in range(NO) if (bel[a][x] != loc[x]) == false_belief]
        if not cands:
            return make(rng)
        a, x = rng.choice(cands)
        q = rng.choice([f"Where does {agents[a]} think the {objs[x]} is?", f"Where will {agents[a]} look for the {objs[x]}?"])
        ret, answer, gold = f"bel[{a} * {NO} + {x}]", bel[a][x], bel[a][x]
        meta['false_belief'] = false_belief
    elif qkind == 'second':
        cands = [(a, b, x) for a in range(NA) for b in range(NA) for x in range(NO)
                 if a != b and (bel2[a][b][x] != loc[x]) == false_belief]
        if not cands:
            return make(rng)
        a, b, x = rng.choice(cands)
        q = rng.choice([f"Where does {agents[a]} think {agents[b]} thinks the {objs[x]} is?",
                        f"Where does {agents[a]} think {agents[b]} will look for the {objs[x]}?"])
        ret, answer, gold = f"bel2[({a} * {NA} + {b}) * {NO} + {x}]", bel2[a][b][x], bel2[a][b][x]
        meta['false_belief'] = false_belief
    else:
        single = [m for m in moves if sum(m['watch']) - m['watch'][m['mover']] == 1]
        if not single:
            return make(rng)
        m = rng.choice(single)
        w = next(b for b in range(NA) if m['watch'][b] and b != m['mover'])
        q = (f"Besides {agents[m['mover']]}, who saw the {objs[m['x']]} being moved to the {locs[m['l']]}?")
        code = code[:m['code_at']] + [f"    saw = -1", f"    for b in range({NA}):",
                                      f"        if see[b] == 1 and b != {m['mover']}:", "            saw = b"] + code[m['code_at']:]
        ret, answer = "saw", w
        options = [agents[b] for b in range(NA) if b != m['mover']]
        gold = options.index(agents[w])
        variants = sorted({len(options), 4} if len(options) > 4 else {len(options)})
    intro = (f"{_join(agents)} are in the {room}. " + ' '.join(
        f"The {objs[x]} {'are' if objs[x] in PLURAL else 'is'} in the {locs[init[x]]}." for x in range(NO))
        + " Everyone can see where everything is.")
    body = ' '.join(lines) if prose else ' '.join(f"{i + 1}. {l}" for i, l in enumerate(lines))
    head = ["def solution():", f"    loc = {init}", f"    here = {[1] * NA}"]
    if track1:
        head.append(f"    bel = {[init[x] for _ in range(NA) for x in range(NO)]}")
    if track2:
        head.append(f"    bel2 = {[init[x] for _ in range(NA * NA) for x in range(NO)]}")
    program = '\n'.join(head + code + [f"    return {ret}"]) + '\n'
    return dict(question=f"{intro} {body} {q}", program=program, answer=answer, options=options, gold=gold,
                variants=variants, meta=meta)


TASK = "Belief tracking deterministic {k}-choice selection"

if __name__ == '__main__':
    rng = random.Random(6)
    for _ in range(3):
        it = make(rng)
        print(it['question'], '\n ->', it['options'][it['gold']], it['meta'])
    print(it['program'])
