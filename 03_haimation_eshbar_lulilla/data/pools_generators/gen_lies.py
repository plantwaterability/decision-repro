"""Web of lies, from a fresh generator: a chain of 1-12 people, each saying
whether earlier people tell the truth, sometimes about two of them at once
("both", "at least one"). Questions: does the last person tell the truth
(yes/no), or how many of the people tell the truth (options 0..n)."""
import random

from gen_tracking import NAMES

TRUE = ["tells the truth", "is honest", "is truthful", "speaks the truth"]
FALSE = ["lies", "is a liar", "is lying", "does not tell the truth"]


def make(rng: random.Random) -> dict:
    n = rng.randint(2, 13)
    people = rng.sample(NAMES, n)
    t = [rng.random() < 0.5]
    lines = [f"{people[0]} {rng.choice(TRUE if t[0] else FALSE)}."]
    prog = ["def solution():", f"    t = [{', '.join(['0'] * n)}]", f"    t[0] = {1 if t[0] else 0}"]
    for i in range(1, n):
        r = rng.random()
        if r < 0.75 or i < 2:
            j = i - 1 if rng.random() < 0.8 else rng.randrange(i)
            claim = rng.random() < 0.5                        # claims j tells the truth
            lines.append(f"{people[i]} says {people[j]} {rng.choice(TRUE if claim else FALSE)}.")
            t.append(t[j] == claim)
            prog.append(f"    t[{i}] = 1 if t[{j}] == {1 if claim else 0} else 0")
        else:
            j, k = rng.sample(range(i), 2)
            if r < 0.88:
                lines.append(f"{people[i]} says both {people[j]} and {people[k]} {rng.choice(['tell the truth', 'are honest'])}.")
                t.append(t[j] and t[k])
                prog.append(f"    t[{i}] = 1 if t[{j}] == 1 and t[{k}] == 1 else 0")
            else:
                lines.append(f"{people[i]} says at least one of {people[j]} and {people[k]} lies.")
                t.append(not (t[j] and t[k]))
                prog.append(f"    t[{i}] = 1 if t[{j}] == 0 or t[{k}] == 0 else 0")
    story = ' '.join(lines)
    if rng.random() < 0.7:
        ans = 1 if t[-1] else 0
        q = rng.choice([f"Does {people[-1]} tell the truth?", f"Is {people[-1]} telling the truth?"])
        return dict(question=f"Question: {story} {q}", program='\n'.join(prog + [f"    return t[{n - 1}]"]) + '\n',
                    answer=ans, options=['Yes', 'No'], gold=0 if ans else 1, variants=[2],
                    meta={'kind': 'last', 'people': n})
    ans = sum(t)
    return dict(question=f"Question: {story} How many of these {n} people tell the truth?",
                program='\n'.join(prog + ["    return sum(t)"]) + '\n', answer=ans,
                options=[str(i) for i in range(n + 1)], gold=ans,
                variants=sorted({4, min(n + 1, 10)}) if n + 1 > 4 else [n + 1], meta={'kind': 'count', 'people': n})


TASK = "Web of lies deterministic {k}-choice selection"

if __name__ == '__main__':
    rng = random.Random(5)
    for _ in range(3):
        it = make(rng)
        print(it['question'], '->', it['options'][it['gold']], it['meta'])
        print(it['program'])
