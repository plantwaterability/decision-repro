"""Tracking shuffled objects, from a fresh generator: 3-10 entities, 1-12
updates (pair swaps and three-way rotations), many scenario families and
phrasings. The question asks what an entity ends up with, or who ends up
with an item. The program tracks a list of item ids."""
import random

NAMES = """Alice Bob Claire Dave Eve Fred Gertrude Helen Izzi Jamie Karl Lola Mohammed Nadia Ophelia Pat Quentin
Rodrigo Sam Tara Umar Vera Wei Xavier Yuki Zoe Amara Bilal Chen Dmitri Esther Farah Gustavo Hiro Ines Jonas Kemal Leila
Mateo Nkechi Oskar Priya Rania Soren Tomas Uma Viktor Wanjiru Yusuf Zara Aiko Bruno Carmen Diego Elif Femi Greta Hassan
Ingrid Javier Keiko Lars Maya Niamh Omar Paolo Rosa Sanjay Talia Ugo Valeria Willa Ximena Yara Zhen Anika Boris Celine
Darius Emeka Fatima Gideon Hana Ivan Jasmin Kofi Lucia Malik Noor Olga Pedro Rui Saoirse Theo Ulla Vikram Wren""".split()

COLORS = """red blue green yellow purple orange pink white black brown gold silver teal maroon turquoise lavender
crimson olive navy beige""".split()

# family: (intro, holder noun, item list, item phrase, swap phrasings, rotate phrasing, forward q, inverse q, option form)
FAMILIES = {
    'presents': dict(
        intro="{names} are friends at a white elephant gift swap. At the start, each holds one present: {assign}.",
        assign="{p} has {a} {i} present", items=COLORS, opt="{i} present", thing="present",
        swap=["{a} and {b} swap their presents.", "{a} and {b} trade presents.", "{a} exchanges presents with {b}."],
        fwd=["At the end of the swap, which present does {p} have?", "Which present does {p} end up holding?"],
        inv=["Who ends up with the {i} present?", "At the end, who is holding the {i} present?"]),
    'balls': dict(
        intro="{names} are playing a game in the park. At the start of the game, each has a ball: {assign}.",
        assign="{p} has {a} {i} ball", items=COLORS, opt="{i} ball", thing="ball",
        swap=["{a} and {b} swap balls.", "{a} and {b} trade balls.", "{a} switches balls with {b}."],
        fwd=["At the end of the game, which ball does {p} have?", "Which ball is {p} holding when the game ends?"],
        inv=["Who has the {i} ball at the end of the game?"]),
    'books': dict(
        intro="{names} are in a book club. Each starts the month with a new book: {assign}.",
        assign="{p} gets {i}",
        items="""Frankenstein|Moby Dick|Ulysses|The Odyssey|Hamlet|Catch-22|Lolita|The Great Gatsby|Dune|Emma|Beloved|
Middlemarch|Dracula|Persuasion|The Hobbit|Walden|Jane Eyre|Don Quixote|Things Fall Apart|The Trial""".replace('\n', '').split('|'),
        opt="{i}", thing="book",
        swap=["{a} and {b} swap books.", "{a} and {b} trade books.", "{a} lends {b} their book and borrows {b}'s in return."],
        fwd=["At the end of the month, which book is {p} reading?", "Which book does {p} end up with?"],
        inv=["Who ends up with {i}?", "At the end of the month, who has {i}?"]),
    'dance': dict(
        intro="{names} are dancers at a square dance. At the start of a song, each has a partner: {assign}.",
        assign="{p} is dancing with {i}", items=None, opt="{i}", thing="partner",
        swap=["{a} and {b} switch partners.", "{a} and {b} trade partners.", "{a} and {b} swap partners."],
        rotate="{a} takes {b}'s partner, {b} takes {c}'s partner, and {c} takes {a}'s original partner.",
        fwd=["At the end of the dance, who is {p} dancing with?", "Who is {p}'s partner when the song ends?"],
        inv=["Who is dancing with {i} at the end of the dance?"]),
    'soccer': dict(
        intro="{names} are on the same team in a soccer match. At the start of the match, each has a position: {assign}.",
        assign="{p} is playing {i}",
        items="""goalkeeper|left winger|right winger|striker|center midfielder|left midfielder|right midfielder|
fullback|sweeper|center back|left back|right back|attacking midfielder|defensive midfielder""".replace('\n', '').split('|'),
        opt="{i}", thing="position",
        swap=["{a} and {b} trade positions.", "{a} and {b} swap positions.", "{a} switches positions with {b}."],
        fwd=["At the end of the match, what position is {p} playing?", "Which position does {p} finish the match in?"],
        inv=["Who is playing {i} at the end of the match?"]),
    'boxes': dict(
        intro="There are {k} boxes, labeled {names}. Each box holds one thing: {assign}.",
        assign="box {p} holds the {i}", labels=True,
        items="""apple|key|coin|ring|marble|feather|shell|pebble|button|stamp|whistle|compass|thimble|acorn|bell|dice""".split('|'),
        opt="{i}", thing="item",
        swap=["The contents of boxes {a} and {b} are exchanged.", "Someone swaps what is in box {a} with what is in box {b}.",
              "The items in box {a} and box {b} trade places."],
        rotate="The item in box {a} moves to box {b}, the item in box {b} moves to box {c}, and the item in box {c} moves to box {a}.",
        fwd=["What is in box {p} at the end?", "Which item ends up in box {p}?"],
        inv=["Which box holds the {i} at the end?"]),
    'roles': dict(
        intro="{names} are rehearsing a play. At the first rehearsal, each is cast in a role: {assign}.",
        assign="{p} plays {i}",
        items="""the king|the queen|the narrator|the villain|the jester|the knight|the wizard|the ghost|the baker|the sailor|
the detective|the gardener|the princess|the dragon""".replace('\n', '').split('|'),
        opt="{i}", thing="role",
        swap=["{a} and {b} swap roles.", "The director has {a} and {b} trade roles.", "{a} and {b} exchange parts."],
        fwd=["By opening night, which role does {p} play?", "Which part does {p} end up playing?"],
        inv=["Who plays {i} on opening night?"]),
    'dogs': dict(
        intro="{names} are dog walkers. At the start of the day, each is walking a dog: {assign}.",
        assign="{p} is walking {i}",
        items="Rex|Biscuit|Luna|Pepper|Milo|Daisy|Bruno|Juniper|Otis|Maple|Ziggy|Nova|Waffles|Scout".split('|'),
        opt="{i}", thing="dog",
        swap=["{a} and {b} swap dogs.", "{a} and {b} trade leashes.", "{a} hands their dog to {b} and takes {b}'s dog."],
        fwd=["Which dog is {p} walking at the end of the day?", "At the end of the day, whose leash does {p} hold? Name the dog."],
        inv=["Who is walking {i} at the end of the day?"]),
    'hats': dict(
        intro="{names} arrive at a costume party, each wearing a hat: {assign}.",
        assign="{p} wears {a} {i}",
        items="fedora|beret|sombrero|top hat|beanie|cowboy hat|bowler|baseball cap|panama hat|trilby|pith helmet|fez".split('|'),
        opt="{i}", thing="hat",
        swap=["{a} and {b} swap hats.", "{a} and {b} trade hats.", "{a} puts on {b}'s hat and {b} puts on {a}'s."],
        fwd=["At the end of the party, which hat is {p} wearing?"],
        inv=["Who is wearing the {i} at the end of the party?"]),
    'seats': dict(
        intro="{names} are seated on a train. At departure, each has a seat: {assign}.",
        assign="{p} is in seat {i}", items=[str(n) for n in range(1, 41)], opt="seat {i}", thing="seat",
        swap=["{a} and {b} switch seats.", "{a} and {b} trade seats.", "{a} moves to {b}'s seat and {b} moves to {a}'s."],
        fwd=["At the last stop, which seat is {p} in?"],
        inv=["Who is sitting in seat {i} at the last stop?"]),
    'shifts': dict(
        intro="{names} work at a bakery. This week each was assigned one shift: {assign}.",
        assign="{p} has the {i} shift",
        items="Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|early Monday|late Friday|Sunday night".split('|'),
        opt="{i} shift", thing="shift",
        swap=["{a} and {b} swap shifts.", "{a} trades shifts with {b}.", "{a} and {b} exchange shifts."],
        fwd=["After all the trades, which shift does {p} have?"],
        inv=["After all the trades, who works the {i} shift?"]),
}

CONNECT = [
    ("As the {event} goes on, they make these changes. ", ["First, ", "Then, ", "Next, ", "After that, ", "Finally, "]),
    ("Over the course of the {event}: ", ["", "", "", "", ""]),
    ("The following happens, in order. ", ["", "", "", "", ""]),
    ("During the {event}, ", ["first ", "then ", "then ", "then ", "and finally "]),
]
EVENT = {'presents': 'swap', 'balls': 'game', 'books': 'month', 'dance': 'song', 'soccer': 'match', 'boxes': 'shuffle',
         'roles': 'rehearsal period', 'dogs': 'day', 'hats': 'party', 'seats': 'trip', 'shifts': 'week'}


def _join(xs):
    return ', '.join(xs[:-1]) + ', and ' + xs[-1] if len(xs) > 2 else ' and '.join(xs)


def make(rng: random.Random) -> dict:
    fam = rng.choice(list(FAMILIES))
    F = FAMILIES[fam]
    k = rng.randint(3, 10)
    depth = rng.randint(1, 12)
    if F.get('labels'):
        people = [chr(ord('A') + i) for i in range(k)]
    else:
        people = rng.sample(NAMES, k)
    pool = F['items'] if F['items'] is not None else [n for n in NAMES if n not in people]
    if len(pool) < k:
        return make(rng)
    items = rng.sample(pool, k)
    if fam == 'seats':
        items = sorted(items, key=int)
    has = list(range(k))                                     # has[person] = item id
    ops, lines = [], []
    for _ in range(depth):
        if k >= 3 and rng.random() < 0.25:
            a, b, c = rng.sample(range(k), 3)
            rot = F.get('rotate', "{a} gives their " + F['thing'] + " to {b}, {b} gives theirs to {c}, and {c} gives theirs to {a}.")
            lines.append(rot.format(a=people[a], b=people[b], c=people[c]))
            has[b], has[c], has[a] = has[a], has[b], has[c]
            ops.append(f"    has[{b}], has[{c}], has[{a}] = has[{a}], has[{b}], has[{c}]")
        else:
            a, b = rng.sample(range(k), 2)
            lines.append(rng.choice(F['swap']).format(a=people[a], b=people[b]))
            has[a], has[b] = has[b], has[a]
            ops.append(f"    has[{a}], has[{b}] = has[{b}], has[{a}]")
    intro = F['intro'].format(names=_join([p if not F.get('labels') else p for p in people]), k=k,
                              assign='; '.join(F['assign'].format(p=p, i=i, a='an' if i[0] in 'aeiou' else 'a')
                                               for p, i in zip(people, items)))
    lead, words = rng.choice(CONNECT)
    lead = lead.format(event=EVENT[fam])
    if depth == 1:
        lead, body = rng.choice(["Then ", "Later, ", "During the " + EVENT[fam] + ", ", ""]), lines[0]
        body = body[0].lower() + body[1:] if lead and not body.startswith(tuple(people)) else body
    elif rng.random() < 0.2:
        body = ' '.join(f"{n + 1}. {l}" for n, l in enumerate(lines))
    else:
        body = ' '.join((words[0] if n == 0 else words[4] if n == depth - 1 and depth > 1 else words[min(n, 3)])
                        + (l[0].lower() + l[1:] if words[0] and not l.startswith(tuple(people)) else l)
                        for n, l in enumerate(lines))
    forward = rng.random() < 0.7
    prog = [f"def solution():", f"    has = {list(range(k))}"] + ops
    if forward:
        p = rng.randrange(k)
        q = rng.choice(F['fwd']).format(p=people[p])
        prog.append(f"    return has[{p}]")
        answer, options = has[p], [F['opt'].format(i=i) for i in items]
    else:
        it = rng.randrange(k)
        q = rng.choice(F['inv']).format(i=items[it])
        prog += [f"    for person in range({k}):", f"        if has[person] == {it}:", "            return person",
                 "    return -1"]
        answer = has.index(it)
        options = [f"box {p}" if F.get('labels') else p for p in people]
    question = f"{intro} {lead}{body} {q}"
    variants = sorted({k, 4} if k > 4 else {k})
    return dict(question=question, program='\n'.join(prog) + '\n', answer=answer, options=options, gold=answer,
                variants=variants, meta={'scenario': fam, 'entities': k, 'updates': depth,
                                         'direction': 'forward' if forward else 'inverse'})


TASK = "Tracking shuffled objects deterministic {k}-choice selection"

if __name__ == '__main__':
    rng = random.Random(0)
    for _ in range(3):
        it = make(rng)
        print(it['question'], '\n  options:', it['options'], 'gold:', it['options'][it['gold']], it['meta'])
        print(it['program'])
