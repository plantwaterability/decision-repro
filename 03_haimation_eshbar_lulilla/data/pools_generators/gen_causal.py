"""Causal stories with stated probabilities, from a fresh generator (CLadder
style, not CLadder items): chain, confounder, mediator and collider graphs
over everyday and nonsense variables, arbitrary two-digit probabilities.
Yes/no questions (marginal, correlation, ATE, NDE, NIE, effect existence,
collider-conditioned correlation), and numeric questions whose distractors
are the values of wrong formulas (correlation for causation, one stratum,
unweighted strata, NDE for NIE, ...), each its own Loom-run program.
Runs at N=64: Bayes ratios of two-digit probabilities outgrow 32 bits."""
import random
from fractions import Fraction as Fr

N = 64

# variable: (name, yes, no, event); topic: subject, Z, X, M, Y, C
TOPICS = [
    ('people', ('gender', 'are male', 'are not male', 'being male'),
     ('smoking', 'smoke', 'do not smoke', 'smoking'),
     ('tar deposit', 'have high tar deposit', 'do not have high tar deposit', 'high tar deposit'),
     ('lung cancer', 'have lung cancer', 'do not have lung cancer', 'lung cancer'),
     ('hospitalization', 'are hospitalized', 'are not hospitalized', 'hospitalization')),
    ('students', ('family income', 'come from high-income families', 'come from low-income families', 'high family income'),
     ('tutoring', 'attend tutoring', 'do not attend tutoring', 'attending tutoring'),
     ('study time', 'study over ten hours a week', 'study ten hours a week or less', 'studying over ten hours a week'),
     ('exam success', 'pass the exam', 'fail the exam', 'passing the exam'),
     ('a scholarship', 'win a scholarship', 'do not win a scholarship', 'winning a scholarship')),
    ('patients', ('age', 'are over sixty', 'are sixty or younger', 'being over sixty'),
     ('the drug', 'take the drug', 'do not take the drug', 'taking the drug'),
     ('blood pressure', 'have lowered blood pressure', 'do not have lowered blood pressure', 'lowered blood pressure'),
     ('recovery', 'recover', 'do not recover', 'recovery'),
     ('follow-up visits', 'return for follow-up', 'do not return for follow-up', 'returning for follow-up')),
    ('plants', ('soil quality', 'grow in rich soil', 'grow in poor soil', 'growing in rich soil'),
     ('fertilizer', 'get fertilizer', 'get no fertilizer', 'getting fertilizer'),
     ('root depth', 'develop deep roots', 'develop shallow roots', 'deep roots'),
     ('flowering', 'flower', 'do not flower', 'flowering'),
     ('pest damage', 'show pest damage', 'show no pest damage', 'pest damage')),
    ('companies', ('market size', 'operate in a large market', 'operate in a small market', 'a large market'),
     ('advertising', 'advertise', 'do not advertise', 'advertising'),
     ('brand awareness', 'have high brand awareness', 'have low brand awareness', 'high brand awareness'),
     ('profit growth', 'grow their profits', 'do not grow their profits', 'profit growth'),
     ('media coverage', 'get media coverage', 'get no media coverage', 'media coverage')),
    ('employees', ('experience', 'are experienced', 'are new', 'being experienced'),
     ('training', 'take the training', 'skip the training', 'taking the training'),
     ('skills', 'gain new skills', 'gain no new skills', 'gaining new skills'),
     ('promotion', 'get promoted', 'do not get promoted', 'getting promoted'),
     ('an award', 'win an award', 'do not win an award', 'winning an award')),
    ('days', ('season', 'fall in summer', 'fall outside summer', 'summer'),
     ('ice cream sales', 'have high ice cream sales', 'have low ice cream sales', 'high ice cream sales'),
     ('crowding', 'have crowded beaches', 'have quiet beaches', 'crowded beaches'),
     ('sunburn reports', 'have many sunburn reports', 'have few sunburn reports', 'many sunburn reports'),
     ('news stories', 'make the news', 'do not make the news', 'making the news')),
    ('households', ('neighborhood', 'live in the city', 'live outside the city', 'living in the city'),
     ('solar panels', 'install solar panels', 'do not install solar panels', 'installing solar panels'),
     ('energy bills', 'have low energy bills', 'have high energy bills', 'low energy bills'),
     ('savings', 'build savings', 'do not build savings', 'building savings'),
     ('a rebate', 'receive a rebate', 'receive no rebate', 'receiving a rebate')),
]
SYL = ['ka', 'zo', 'vu', 'pe', 'ri', 'lo', 'mi', 'xa', 'ty', 'qe', 'no', 'bu', 'fi', 'gr', 'sl', 'wy', 'jo', 'de']


def nonsense(rng):
    words = set()
    while len(words) < 5:
        words.add(''.join(rng.choice(SYL) for _ in range(rng.randint(2, 3))))
    w = list(words)
    var = lambda x: (x, f'are {x}', f'are not {x}', f'being {x}')
    return ('individuals', var(w[0]), var(w[1]), var(w[2]), var(w[3]), var(w[4]))


PREAMBLES = ["Imagine a self-contained, hypothetical world with only the following conditions, and without any "
             "unmentioned factors or causal relationships: ",
             "Consider a closed world in which only the following causal relationships hold: ",
             "In this hypothetical setting, nothing else matters beyond the following: ", ""]


def pr(rng, lo=5, hi=95):
    return Fr(rng.randint(lo, hi), 100)


def ptxt(p, rng):
    return f"{int(p * 100)}%" if rng.random() < 0.7 else f"{float(p):.2f}"


def lit(p):
    return f"{float(p):.2f}"


def round4(v):
    """round(v, 4), ties away from zero (Loom's round)."""
    q = abs(v) * 10000
    r = int(q) + (1 if q - int(q) >= Fr(1, 2) else 0)
    return Fr(r if v >= 0 else -r, 10000)


def among(subj, *conds):
    rel = 'that' if subj in ('plants', 'companies', 'days', 'households') else 'who'
    return f"Among {subj} {rel} " + ' and '.join(conds)


def make(rng: random.Random) -> dict:
    topic = rng.choice(TOPICS) if rng.random() < 0.75 else nonsense(rng)
    subj, Z, X, M, Y, C = topic
    graph = rng.choice(['chain', 'confounder', 'confounder', 'mediator', 'mediator', 'collider'])
    text, lines = [], {}
    if graph == 'chain':
        px, a1, a0 = pr(rng), pr(rng), pr(rng)
        edges = f"{X[0].capitalize()} has a direct effect on {Y[0]}."
        text = [f"The overall probability of {X[3]} is {ptxt(px, rng)}.",
                f"{among(subj, X[1])}, the probability of {Y[3]} is {ptxt(a1, rng)}.",
                f"{among(subj, X[2])}, the probability of {Y[3]} is {ptxt(a0, rng)}."]
        lines = {'px': px, 'py_x1': a1, 'py_x0': a0}
        Q = {'marginal': px * a1 + (1 - px) * a0, 'corr': a1 - a0, 'ate': a1 - a0}
        prog = {'marginal': "px * py_x1 + (1 - px) * py_x0", 'corr': "py_x1 - py_x0", 'ate': "py_x1 - py_x0"}
        wrong = {'marginal': ["py_x1", "py_x0", "(py_x1 + py_x0) / 2", "px * py_x0 + (1 - px) * py_x1"],
                 'ate': ["px * py_x1 - (1 - px) * py_x0", "py_x0 - py_x1", "py_x1", "px * (py_x1 - py_x0)"]}
    elif graph == 'confounder':
        pz, b1, b0 = pr(rng), pr(rng), pr(rng)
        c = {(x, z): pr(rng) for x in (1, 0) for z in (1, 0)}
        edges = (f"{Z[0].capitalize()} has a direct effect on {X[0]} and {Y[0]}. "
                 f"{X[0].capitalize()} has a direct effect on {Y[0]}.")
        text = [f"The overall probability of {Z[3]} is {ptxt(pz, rng)}.",
                f"{among(subj, Z[1])}, the probability of {X[3]} is {ptxt(b1, rng)}.",
                f"{among(subj, Z[2])}, the probability of {X[3]} is {ptxt(b0, rng)}."]
        for (x, z), v in c.items():
            text.append(f"{among(subj, (X[1] if x else X[2]), (Z[1] if z else Z[2]))}, the probability of {Y[3]} "
                        f"is {ptxt(v, rng)}.")
        lines = {'pz': pz, 'px_z1': b1, 'px_z0': b0, 'py_x1z1': c[1, 1], 'py_x1z0': c[1, 0], 'py_x0z1': c[0, 1],
                 'py_x0z0': c[0, 0]}
        p1 = pz * b1 + (1 - pz) * b0
        py1 = (pz * b1 * c[1, 1] + (1 - pz) * b0 * c[1, 0]) / p1
        py0 = (pz * (1 - b1) * c[0, 1] + (1 - pz) * (1 - b0) * c[0, 0]) / (1 - p1)
        Q = {'marginal': p1 * py1 + (1 - p1) * py0, 'corr': py1 - py0,
             'ate': pz * (c[1, 1] - c[0, 1]) + (1 - pz) * (c[1, 0] - c[0, 0])}
        prog_corr = ("(pz * px_z1 * py_x1z1 + (1 - pz) * px_z0 * py_x1z0) / (pz * px_z1 + (1 - pz) * px_z0) - "
                     "(pz * (1 - px_z1) * py_x0z1 + (1 - pz) * (1 - px_z0) * py_x0z0) / (pz * (1 - px_z1) + (1 - pz) * (1 - px_z0))")
        prog = {'marginal': "pz * px_z1 * py_x1z1 + pz * (1 - px_z1) * py_x0z1 + (1 - pz) * px_z0 * py_x1z0 + "
                            "(1 - pz) * (1 - px_z0) * py_x0z0",
                'corr': prog_corr, 'ate': "pz * (py_x1z1 - py_x0z1) + (1 - pz) * (py_x1z0 - py_x0z0)"}
        wrong = {'ate': [prog_corr, "py_x1z1 - py_x0z1", "py_x1z0 - py_x0z0",
                         "(py_x1z1 - py_x0z1 + py_x1z0 - py_x0z0) / 2",
                         "(1 - pz) * (py_x1z1 - py_x0z1) + pz * (py_x1z0 - py_x0z0)",
                         "pz * py_x1z1 + (1 - pz) * py_x1z0", "py_x0z1 - py_x1z1", "pz * (py_x1z1 - py_x0z0)",
                         "px_z1 * (py_x1z1 - py_x0z1) + (1 - px_z1) * (py_x1z0 - py_x0z0)",
                         "pz * (py_x1z1 - py_x1z0) + (1 - pz) * (py_x0z1 - py_x0z0)"],
                 'corr': ["pz * (py_x1z1 - py_x0z1) + (1 - pz) * (py_x1z0 - py_x0z0)", "py_x1z1 - py_x0z1",
                          "py_x1z0 - py_x0z0", "(py_x1z1 + py_x1z0) / 2 - (py_x0z1 + py_x0z0) / 2"]}
    elif graph == 'mediator':
        px, m1, m0 = pr(rng), pr(rng), pr(rng)
        d = {(x, m): pr(rng) for x in (1, 0) for m in (1, 0)}
        edges = (f"{X[0].capitalize()} has a direct effect on {M[0]} and {Y[0]}. "
                 f"{M[0].capitalize()} has a direct effect on {Y[0]}.")
        text = [f"The overall probability of {X[3]} is {ptxt(px, rng)}.",
                f"{among(subj, X[1])}, the probability of {M[3]} is {ptxt(m1, rng)}.",
                f"{among(subj, X[2])}, the probability of {M[3]} is {ptxt(m0, rng)}."]
        for (x, m), v in d.items():
            text.append(f"{among(subj, (X[1] if x else X[2]), (M[1] if m else M[2]))}, the probability of {Y[3]} "
                        f"is {ptxt(v, rng)}.")
        lines = {'px': px, 'pm_x1': m1, 'pm_x0': m0, 'py_x1m1': d[1, 1], 'py_x1m0': d[1, 0], 'py_x0m1': d[0, 1],
                 'py_x0m0': d[0, 0]}
        y1 = m1 * d[1, 1] + (1 - m1) * d[1, 0]
        y0 = m0 * d[0, 1] + (1 - m0) * d[0, 0]
        Q = {'marginal': px * y1 + (1 - px) * y0, 'ate': y1 - y0, 'corr': y1 - y0,
             'nde': m0 * (d[1, 1] - d[0, 1]) + (1 - m0) * (d[1, 0] - d[0, 0]),
             'nie': (m1 - m0) * (d[0, 1] - d[0, 0])}
        ate = "pm_x1 * py_x1m1 + (1 - pm_x1) * py_x1m0 - pm_x0 * py_x0m1 - (1 - pm_x0) * py_x0m0"
        nde = "pm_x0 * (py_x1m1 - py_x0m1) + (1 - pm_x0) * (py_x1m0 - py_x0m0)"
        nie = "(pm_x1 - pm_x0) * (py_x0m1 - py_x0m0)"
        prog = {'marginal': f"px * (pm_x1 * py_x1m1 + (1 - pm_x1) * py_x1m0) + (1 - px) * (pm_x0 * py_x0m1 + (1 - pm_x0) * py_x0m0)",
                'ate': ate, 'corr': ate, 'nde': nde, 'nie': nie}
        alt = ["pm_x1 * (py_x1m1 - py_x0m1) + (1 - pm_x1) * (py_x1m0 - py_x0m0)", "(pm_x1 - pm_x0) * (py_x1m1 - py_x1m0)",
               "py_x1m1 - py_x0m1", "py_x1m0 - py_x0m0", "pm_x1 - pm_x0", "py_x1m1 - py_x0m0"]
        wrong = {'ate': [nde, nie] + alt, 'nde': [ate, nie] + alt, 'nie': [ate, nde] + alt}
    else:                                                    # collider: X -> C <- Y, X and Y independent
        px, py = pr(rng), pr(rng)
        e = {(x, y): pr(rng) for x in (1, 0) for y in (1, 0)}
        edges = f"{X[0].capitalize()} has a direct effect on {C[0]}. {Y[0].capitalize()} has a direct effect on {C[0]}."
        text = [f"The overall probability of {X[3]} is {ptxt(px, rng)}.",
                f"The overall probability of {Y[3]} is {ptxt(py, rng)}."]
        for (x, y), v in e.items():
            text.append(f"{among(subj, (X[1] if x else X[2]), (Y[1] if y else Y[2]))}, the probability of {C[3]} "
                        f"is {ptxt(v, rng)}.")
        lines = {'px': px, 'py': py, 'pc_x1y1': e[1, 1], 'pc_x1y0': e[1, 0], 'pc_x0y1': e[0, 1], 'pc_x0y0': e[0, 0]}
        q1 = py * e[1, 1] / (py * e[1, 1] + (1 - py) * e[1, 0])
        q0 = py * e[0, 1] / (py * e[0, 1] + (1 - py) * e[0, 0])
        Q = {'collider_corr': q1 - q0, 'ate': Fr(0)}
        prog = {'collider_corr': "py * pc_x1y1 / (py * pc_x1y1 + (1 - py) * pc_x1y0) - "
                                 "py * pc_x0y1 / (py * pc_x0y1 + (1 - py) * pc_x0y0)", 'ate': "0"}
        wrong = {'collider_corr': ["pc_x1y1 - pc_x0y1", "py * (pc_x1y1 - pc_x0y1)", "0",
                                   "pc_x1y1 / (pc_x1y1 + pc_x1y0) - pc_x0y1 / (pc_x0y1 + pc_x0y0)"]}
    rng.shuffle(text)
    story = rng.choice(PREAMBLES) + edges + ' ' + ' '.join(text)

    queries = [q for q in Q if not (graph == 'collider' and q == 'ate')] + (['effect'] if graph in ('chain', 'collider') else [])
    query = rng.choice(queries)
    numeric = query in wrong and rng.random() < 0.35
    assigns = [f"    {k} = {lit(v)}" for k, v in lines.items()]
    if numeric:
        name = {'marginal': f"the overall probability of {Y[3]}", 'corr': f"the difference in the probability of {Y[3]} "
                f"between {subj} who {X[1]} and those who {X[2]}" if subj not in ('plants', 'companies', 'days', 'households')
                else f"the difference in the probability of {Y[3]} between {subj} that {X[1]} and those that {X[2]}",
                'ate': f"the average treatment effect of {X[0]} on {Y[3]}",
                'nde': f"the natural direct effect of {X[0]} on {Y[3]}",
                'nie': f"the natural indirect effect of {X[0]} on {Y[3]} through {M[0]}",
                'collider_corr': f"the difference in the probability of {Y[3]} between those with and without {X[3]}, "
                                 f"among those with {C[3]}"}[query]
        question = f"{story} What is {name}? Give it to four decimal places."
        value = round4(Q[query])
        program = "def solution():\n" + '\n'.join(assigns) + f"\n    v = {prog[query]}\n    return round(v, 4)\n"
        ctx = dict(lines)
        dis, seen = [], {value}
        for w in wrong[query]:
            for sign, expr in ((1, w), (-1, f"-({w})")):
                v = round4(sign * eval(w, {}, ctx))
                if v in seen or (sign < 0 and v == 0):
                    continue
                seen.add(v)
                src = "def solution():\n" + '\n'.join(assigns) + f"\n    v = {expr}\n    return round(v, 4)\n"
                dis.append((src, v, f"{float(v):.4f}"))
        texts = [f"{float(value):.4f}"] + [d[2] for d in dis]
        if len(dis) < 3 or len(set(texts)) != len(texts):
            return make(rng)
        return dict(question=question, program=program, answer=value, options=[texts[0]], gold=0,
                    variants=[4, 10], distractors=dis,
                    meta={'graph': graph, 'query': query, 'kind': 'numeric', 'topic': subj})
    # yes/no
    if query == 'effect':
        truth = Q['ate'] != 0
        if graph == 'chain' and abs(Q['ate']) < Fr(2, 100):
            return make(rng)
        question = f"{story} Does {X[3]} have any causal effect on {Y[3]}?"
        body = f"    v = {prog['ate']}\n    return 1 if v != 0 else 0\n"
    else:
        v = Q[query]
        margin = v - Fr(1, 2) if query == 'marginal' else v
        if abs(margin) < Fr(2, 100):
            return make(rng)
        up = rng.random() < 0.5
        rel_x1 = ('that ' if subj in ('plants', 'companies', 'days', 'households') else 'who ') + X[1]
        if query == 'marginal':
            question = f"{story} Is {Y[3]} more likely than not overall?"
            truth = v > Fr(1, 2)
            test = "v > 0.5"
        elif query == 'corr':
            question = (f"{story} Is the probability of {Y[3]} {'higher' if up else 'lower'} among {subj} {rel_x1} "
                        f"than among the others?")
            truth, test = (v > 0, "v > 0") if up else (v < 0, "v < 0")
        elif query == 'ate':
            question = f"{story} Will {X[3]} {'increase' if up else 'decrease'} the chance of {Y[3]}?"
            truth, test = (v > 0, "v > 0") if up else (v < 0, "v < 0")
        elif query == 'nde':
            question = (f"{story} If we disregard the mediation effect through {M[0]}, would {X[3]} still "
                        f"{'increase' if up else 'decrease'} the chance of {Y[3]}?")
            truth, test = (v > 0, "v > 0") if up else (v < 0, "v < 0")
        elif query == 'nie':
            question = f"{story} Does {X[3]} {'increase' if up else 'decrease'} the chance of {Y[3]} through {M[0]}?"
            truth, test = (v > 0, "v > 0") if up else (v < 0, "v < 0")
        else:
            question = (f"{story} Among {subj} with {C[3]}, is {Y[3]} {'more' if up else 'less'} likely for those with "
                        f"{X[3]} than for those without?")
            truth, test = (v > 0, "v > 0") if up else (v < 0, "v < 0")
        body = f"    v = {prog[query]}\n    return 1 if {test} else 0\n"
    program = "def solution():\n" + '\n'.join(assigns) + "\n" + body
    answer = 1 if truth else 0
    return dict(question=question, program=program, answer=answer, options=['Yes', 'No'], gold=0 if answer else 1,
                variants=[2], meta={'graph': graph, 'query': query, 'kind': 'yes/no', 'topic': subj})


TASK = "Causal reasoning deterministic {k}-choice selection"

if __name__ == '__main__':
    rng = random.Random(4)
    for _ in range(3):
        it = make(rng)
        print(it['question'], '\n ->', it['options'][it['gold']], it['meta'], '\n', it['program'])
