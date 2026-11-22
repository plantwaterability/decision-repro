"""STEM formula problems, from a fresh generator: kinematics, forces and
energy, momentum, electricity, gases, heat, waves, density, chemistry
(moles, molarity, dilution), probability (binomial, Bayes), finance
(compound interest), geometry and unit conversion. Each program calls named
helper functions (kinetic_energy, kmh_to_ms, ideal_gas_moles, ...) so a
library miner sees named formulas. Answers are exact (N=64 rational) and
rounded to 3 significant figures (at most 4 decimals). Distractors are the
values of Loom-run wrong programs: a wrong formula (missing 1/2, v instead
of v^2, P(Y|X) for P(X|Y)) or a unit slip (km/h not converted, degrees C
for kelvin, grams for kilograms)."""
import ast
import math
import random
from fractions import Fraction as Fr

N = 64


def ndp_for(v):
    v = abs(float(v))
    if v == 0:
        return 2
    return max(0, min(4, 2 - math.floor(math.log10(v))))


def rnd(v, n):
    q = abs(v) * 10 ** n
    r = int(q) + (1 if q - int(q) >= Fr(1, 2) else 0)
    return Fr(r if v >= 0 else -r, 10 ** n)


def lit(v):
    v = Fr(v)
    if v.denominator == 1:
        return str(v.numerator)
    s = f"{float(v):.10f}".rstrip('0')
    assert Fr(s) == v, v
    return s


def dec(rng, lo, hi, places=1):
    return Fr(rng.randint(int(lo * 10 ** places), int(hi * 10 ** places)), 10 ** places)


# family -> function(rng) returning (question, helpers {name: src}, args {var: value}, body expr,
#                                    [(label, body expr) wrong], unit text)
def kinematics(rng):
    u, a, t = dec(rng, 0, 30), dec(rng, 0.5, 6), rng.randint(2, 20)
    kmh = rng.random() < 0.5
    u_in = rng.randint(10, 108) if kmh else u
    helpers = {'displacement': "def displacement(u, a, t):\n    return u * t + 0.5 * a * t ** 2\n"}
    if kmh:
        helpers['kmh_to_ms'] = "def kmh_to_ms(v):\n    return v * 1000 / 3600\n"
    uq = f"{lit(u_in)} km/h" if kmh else f"{lit(u_in)} m/s"
    q = (f"A cyclist moving at {uq} accelerates uniformly at {lit(a)} m/s^2 for {t} s. How far does the cyclist travel "
         f"in that time, in meters?")
    u_expr = "kmh_to_ms(u)" if kmh else "u"
    wrong = [('no half', f"{u_expr} * t + a * t ** 2"), ('t not squared', f"{u_expr} * t + 0.5 * a * t"),
             ('ignores u', "0.5 * a * t ** 2"), ('final speed', f"{u_expr} + a * t")]
    if kmh:
        wrong.append(('km/h not converted', "displacement(u, a, t)"))
    return q, helpers, {'u': u_in, 'a': a, 't': t}, f"displacement({u_expr}, a, t)", wrong


def energy(rng):
    m, v = rng.choice([rng.randint(1, 90), dec(rng, 0.2, 5, 1)]), rng.randint(2, 40)
    grams = rng.random() < 0.35
    m_in = m * 1000 if grams else m
    helpers = {'kinetic_energy': "def kinetic_energy(m, v):\n    return 0.5 * m * v ** 2\n"}
    if grams:
        helpers['g_to_kg'] = "def g_to_kg(x):\n    return x / 1000\n"
    mq = f"{lit(m_in)} g" if grams else f"{lit(m_in)} kg"
    q = f"An object of mass {mq} moves at {v} m/s. What is its kinetic energy, in joules?"
    m_expr = "g_to_kg(m)" if grams else "m"
    wrong = [('no half', f"{m_expr} * v ** 2"), ('v not squared', f"0.5 * {m_expr} * v"),
             ('momentum', f"{m_expr} * v"), ('half of mv', f"0.25 * {m_expr} * v ** 2")]
    if grams:
        wrong.append(('grams as kg', "kinetic_energy(m, v)"))
    return q, helpers, {'m': m_in, 'v': v}, f"kinetic_energy({m_expr}, v)", wrong


def potential(rng):
    m, h = rng.randint(1, 80), dec(rng, 0.5, 60)
    g = rng.choice([Fr(98, 10), Fr(981, 100)])
    helpers = {'potential_energy': "def potential_energy(m, g, h):\n    return m * g * h\n"}
    q = f"How much gravitational potential energy does {'an' if str(m)[0] == '8' or m in (11, 18) else 'a'} {m} kg mass gain when lifted {lit(h)} m? Use g = {lit(g)} m/s^2. Give joules."
    wrong = [('g = 10', "potential_energy(m, 10, h)"), ('no g', "m * h"), ('half', "0.5 * m * g * h"),
             ('h squared', "m * g * h ** 2")]
    return q, helpers, {'m': m, 'g': g, 'h': h}, "potential_energy(m, g, h)", wrong


def ohm(rng):
    R1, R2, V = rng.randint(2, 200), rng.randint(2, 200), rng.randint(3, 240)
    parallel = rng.random() < 0.5
    helpers = {'parallel': "def parallel(r1, r2):\n    return r1 * r2 / (r1 + r2)\n",
               'current': "def current(v, r):\n    return v / r\n"}
    how = 'in parallel' if parallel else 'in series'
    q = f"Two resistors of {R1} ohm and {R2} ohm are connected {how} across a {V} V source. What current flows from the source, in amperes?"
    req = "parallel(r1, r2)" if parallel else "r1 + r2"
    other = "r1 + r2" if parallel else "parallel(r1, r2)"
    wrong = [('other connection', f"current(v, {other})"), ('one resistor', "current(v, r1)"),
             ('V times R', f"v * {req}"), ('R over V', f"{req} / v")]
    return q, helpers, {'r1': R1, 'r2': R2, 'v': V}, f"current(v, {req})", wrong


def power(rng):
    I, R = dec(rng, 0.1, 12), rng.randint(2, 150)
    helpers = {'power_dissipated': "def power_dissipated(i, r):\n    return i ** 2 * r\n"}
    q = f"A current of {lit(I)} A flows through a {R} ohm resistor. How much power does it dissipate, in watts?"
    wrong = [('I not squared', "i * r"), ('voltage', "i * r"), ('I over R', "i / r"), ('half', "0.5 * i ** 2 * r"),
             ('R squared', "i * r ** 2")]
    return q, helpers, {'i': I, 'r': R}, "power_dissipated(i, r)", wrong


def gas(rng):
    P, V, Tc = rng.randint(80, 500), dec(rng, 0.5, 50), rng.randint(-20, 150)
    helpers = {'celsius_to_kelvin': "def celsius_to_kelvin(t):\n    return t + 273.15\n",
               'ideal_gas_moles': "def ideal_gas_moles(p, v, t):\n    return p * v / (8.314 * t)\n"}
    q = (f"A gas occupies {lit(V)} L at {P} kPa and {Tc} degrees C. How many moles of gas are there? "
         f"Use R = 8.314 J/(mol K).")
    wrong = [('celsius as kelvin', "ideal_gas_moles(p, v, t)"), ('kelvin by 273', "ideal_gas_moles(p, v, t + 273)"),
             ('litres as m^3 x1000', "ideal_gas_moles(p * 1000, v, celsius_to_kelvin(t))"),
             ('inverted', "8.314 * celsius_to_kelvin(t) / (p * v)")]
    if Tc == 0:
        return gas(rng)
    return q, helpers, {'p': P, 'v': V, 't': Tc}, "ideal_gas_moles(p, v, celsius_to_kelvin(t))", wrong


def heat(rng):
    m, dT = rng.randint(50, 3000), rng.randint(2, 80)
    helpers = {'heat_energy': "def heat_energy(m, c, dt):\n    return m * c * dt\n"}
    q = (f"How much heat, in joules, raises {m} g of water by {dT} degrees C? The specific heat of water is "
         f"4.186 J/(g C).")
    wrong = [('c per kg', "heat_energy(m, 4186, dt)"), ('kg', "heat_energy(m / 1000, 4.186, dt)"),
             ('no dT', "m * 4.186"), ('c = 4.2', "heat_energy(m, 4.2, dt)")]
    return q, helpers, {'m': m, 'dt': dT}, "heat_energy(m, 4.186, dt)", wrong


def wave(rng):
    f, lam = rng.randint(20, 2000), dec(rng, 0.05, 20, 2)
    helpers = {'wave_speed': "def wave_speed(f, lam):\n    return f * lam\n"}
    cm = rng.random() < 0.4
    lam_in = lam * 100 if cm else lam
    unit = 'cm' if cm else 'm'
    q = f"A wave has frequency {f} Hz and wavelength {lit(lam_in)} {unit}. What is its speed, in m/s?"
    body = "wave_speed(f, lam / 100)" if cm else "wave_speed(f, lam)"
    wrong = [('f over lambda', "f / lam"), ('lambda over f', "lam / f"), ('period', "1 / f")]
    if cm:
        wrong.append(('cm as m', "wave_speed(f, lam)"))
    return q, helpers, {'f': f, 'lam': lam_in}, body, wrong


def density(rng):
    mass, vol = rng.randint(10, 5000), rng.randint(2, 900)
    helpers = {'density': "def density(m, v):\n    return m / v\n"}
    q = f"A block has a mass of {mass} g and a volume of {vol} cm^3. What is its density in kg/m^3?"
    wrong = [('g/cm^3', "density(m, v)"), ('inverted', "1000 * v / m"), ('x100', "density(m, v) * 100"),
             ('product', "m * v / 1000")]
    return q, helpers, {'m': mass, 'v': vol}, "density(m, v) * 1000", wrong


MOLAR = {'water': ('H2O', Fr(18015, 1000)), 'carbon dioxide': ('CO2', Fr(4401, 100)), 'sodium chloride': ('NaCl', Fr(5844, 100)),
         'glucose': ('C6H12O6', Fr(18016, 100)), 'ammonia': ('NH3', Fr(17031, 1000)), 'calcium carbonate': ('CaCO3', Fr(10009, 100)),
         'sulfuric acid': ('H2SO4', Fr(9808, 100)), 'methane': ('CH4', Fr(16043, 1000))}


def molarity(rng):
    name = rng.choice(sorted(MOLAR))
    formula, mm = MOLAR[name]
    grams, ml = dec(rng, 1, 200), rng.randint(50, 2000)
    helpers = {'moles': "def moles(mass, molar_mass):\n    return mass / molar_mass\n",
               'molarity': "def molarity(n, litres):\n    return n / litres\n"}
    q = (f"{lit(grams)} g of {name} ({formula}, molar mass {lit(mm)} g/mol) is dissolved to make {ml} mL of solution. "
         f"What is the molarity, in mol/L?")
    wrong = [('mL as L', "molarity(moles(g, mm), ml)"), ('mass as moles', "g / (ml / 1000)"),
             ('inverted', "molarity(mm / g, ml / 1000)"), ('x molar mass', "g * mm / (ml / 1000)")]
    return q, helpers, {'g': grams, 'mm': mm, 'ml': ml}, "molarity(moles(g, mm), ml / 1000)", wrong


def dilution(rng):
    c1, v1, v2 = dec(rng, 0.1, 12, 2), rng.randint(5, 500), None
    v2 = v1 + rng.randint(5, 2000)
    helpers = {'dilute': "def dilute(c1, v1, v2):\n    return c1 * v1 / v2\n"}
    q = f"{v1} mL of a {lit(c1)} M solution is diluted to {v2} mL. What is the new concentration, in M?"
    wrong = [('inverted', "c1 * v2 / v1"), ('added volume', f"c1 * v1 / (v2 - v1)"), ('no change', "c1"),
             ('difference', "c1 * (v2 - v1) / v2")]
    return q, helpers, {'c1': c1, 'v1': v1, 'v2': v2}, "dilute(c1, v1, v2)", wrong


def binomial(rng):
    n = rng.randint(3, 10)
    k = rng.randint(0, n)
    p = rng.choice([Fr(1, 2), Fr(1, 4), Fr(3, 4), Fr(1, 5), Fr(2, 5), Fr(3, 10), Fr(7, 10), Fr(1, 10), Fr(9, 10)])
    helpers = {'choose': "def choose(n, k):\n    c = 1\n    for i in range(k):\n        c = c * (n - i) // (i + 1)\n    return c\n",
               'power': "def power(x, n):\n    r = 1\n    for i in range(n):\n        r = r * x\n    return r\n",
               'binomial_prob': "def binomial_prob(n, k, p):\n    return choose(n, k) * power(p, k) * power(1 - p, n - k)\n"}
    q = (f"A test question is answered correctly with probability {lit(p)}, independently across {n} questions. "
         f"What is the probability of exactly {k} correct answers?")
    wrong = [('no coefficient', "power(p, k) * power(1 - p, n - k)"), ('p^k only', "choose(n, k) * power(p, k)"),
             ('swapped', "choose(n, k) * power(1 - p, k) * power(p, n - k)"), ('k/n', f"{k} / {n}")]
    return q, helpers, {'n': n, 'k': k, 'p': p}, "binomial_prob(n, k, p)", wrong


def bayes(rng):
    prior, sens, fpr = dec(rng, 0.01, 0.3, 2), dec(rng, 0.7, 0.99, 2), dec(rng, 0.01, 0.2, 2)
    helpers = {'posterior': "def posterior(prior, sens, fpr):\n    return sens * prior / (sens * prior + fpr * (1 - prior))\n"}
    q = (f"A condition affects {lit(prior * 100)}% of people. A test detects it {lit(sens * 100)}% of the time and gives "
         f"a false positive {lit(fpr * 100)}% of the time. If a person tests positive, what is the probability they "
         f"have the condition?")
    wrong = [('sensitivity', "sens"), ('no prior', "sens / (sens + fpr)"), ('joint', "sens * prior"),
             ('ignores complement', "sens * prior / (sens * prior + fpr)")]
    return q, helpers, {'prior': prior, 'sens': sens, 'fpr': fpr}, "posterior(prior, sens, fpr)", wrong


def compound(rng):
    P, r, years = rng.randint(100, 20000), Fr(rng.randint(1, 9)), rng.randint(1, 6)
    n = rng.choice([1, 2])
    per = {1: 'annually', 2: 'semiannually', 4: 'quarterly', 12: 'monthly'}[n]
    helpers = {'compound': f"def compound(p, rate, n, years):\n    return p * (1 + rate / 100 / n) ** ({n * years})\n"}
    q = f"${P} is invested at {lit(r)}% annual interest, compounded {per}, for {years} years. What is the final amount, in dollars?"
    wrong = [('simple', "p * (1 + rate / 100 * years)"), ('annual compounding', f"p * (1 + rate / 100) ** {years}"),
             ('interest only', f"compound(p, rate, n, years) - p"), ('rate not percent', f"p * (1 + rate / n) ** {n * years}")]
    if n * years > 6:                          # exact powers outgrow 64 bits beyond this
        return compound(rng)
    return q, helpers, {'p': P, 'rate': r, 'n': n, 'years': years}, "compound(p, rate, n, years)", wrong


def geometry(rng):
    r, h = rng.randint(1, 30), rng.randint(1, 40)
    shape = rng.choice(['cylinder', 'cone', 'sphere'])
    helpers = {'cylinder_volume': "def cylinder_volume(r, h):\n    return 3.14 * r ** 2 * h\n",
               'cone_volume': "def cone_volume(r, h):\n    return 3.14 * r ** 2 * h / 3\n",
               'sphere_volume': "def sphere_volume(r):\n    return 4 * 3.14 * r ** 3 / 3\n"}
    if shape == 'sphere':
        q = f"What is the volume of a sphere of radius {r} cm, in cm^3? Use pi = 3.14."
        body, wrong = "sphere_volume(r)", [('r squared', "4 * 3.14 * r ** 2 / 3"), ('no 4/3', "3.14 * r ** 3"),
                                             ('surface area', "4 * 3.14 * r ** 2"), ('diameter', "4 * 3.14 * (2 * r) ** 3 / 3")]
    else:
        q = f"What is the volume of a {shape} of radius {r} cm and height {h} cm, in cm^3? Use pi = 3.14."
        body = f"{shape}_volume(r, h)"
        other = 'cone' if shape == 'cylinder' else 'cylinder'
        wrong = [('other solid', f"{other}_volume(r, h)"), ('r not squared', "3.14 * r * h"),
                 ('diameter', f"{shape}_volume(2 * r, h)"), ('no pi', f"{shape}_volume(r, h) / 3.14")]
    return q, helpers, {'r': r, 'h': h}, body, wrong


def convert(rng):
    kind = rng.choice(['f', 'mph', 'in', 'lb', 'kmh'])
    if kind == 'f':
        c = rng.randint(-40, 200)
        helpers = {'celsius_to_fahrenheit': "def celsius_to_fahrenheit(c):\n    return c * 9 / 5 + 32\n"}
        return (f"Convert {c} degrees Celsius to degrees Fahrenheit.", helpers, {'c': c}, "celsius_to_fahrenheit(c)",
                [('no offset', "c * 9 / 5"), ('inverted ratio', "c * 5 / 9 + 32"), ('offset first', "(c + 32) * 9 / 5"),
                 ('subtract', "(c - 32) * 5 / 9")])
    if kind == 'mph':
        v = rng.randint(5, 120)
        helpers = {'mph_to_kmh': "def mph_to_kmh(v):\n    return v * 1.609344\n"}
        return (f"Convert {v} miles per hour to kilometres per hour.", helpers, {'v': v}, "mph_to_kmh(v)",
                [('divided', "v / 1.609344"), ('1.6', "v * 1.6"), ('to m/s', "mph_to_kmh(v) * 1000 / 3600")])
    if kind == 'in':
        x = dec(rng, 1, 300)
        helpers = {'inches_to_cm': "def inches_to_cm(x):\n    return x * 2.54\n"}
        return (f"How many centimetres are in {lit(x)} inches?", helpers, {'x': x}, "inches_to_cm(x)",
                [('divided', "x / 2.54"), ('to metres', "inches_to_cm(x) / 100"), ('2.5', "x * 2.5")])
    if kind == 'lb':
        x = dec(rng, 1, 500)
        helpers = {'lb_to_kg': "def lb_to_kg(x):\n    return x * 0.45359237\n"}
        return (f"How many kilograms are in {lit(x)} pounds?", helpers, {'x': x}, "lb_to_kg(x)",
                [('multiplied by 2.2', "x * 2.2"), ('divided', "x / 0.45359237"), ('half', "x / 2")])
    v = rng.randint(5, 300)
    helpers = {'kmh_to_ms': "def kmh_to_ms(v):\n    return v * 1000 / 3600\n"}
    return (f"Convert {v} km/h to m/s.", helpers, {'v': v}, "kmh_to_ms(v)",
            [('multiplied by 3.6', "v * 3.6"), ('divided by 1000', "v / 1000"), ('x 1000/60', "v * 1000 / 60")])


FAMILIES = [kinematics, energy, potential, ohm, power, gas, heat, wave, density, molarity, dilution, binomial, bayes,
            compound, geometry, convert]


def program(helpers, args, body, ndp):
    lines = [src.rstrip('\n') for _, src in sorted(helpers.items())] + ["def solution():"]
    lines += [f"    {k} = {lit(v)}" for k, v in args.items()]
    lines += [f"    v = {body}", f"    return round(v, {ndp})"]
    return '\n'.join(lines) + '\n'


class _Exact(ast.NodeTransformer):
    """Float literals as exact fractions, as Loom's rational mode reads them."""
    def visit_Constant(self, node):
        if isinstance(node.value, float):
            return ast.Call(ast.Name('Fr', ast.Load()), [ast.Constant(repr(node.value))], [])
        return node


def evaluate(helpers, args, body):
    """The exact value of body, computed in Python with fractions."""
    ns = {'Fr': Fr}
    for src in helpers.values():
        exec(compile(ast.fix_missing_locations(_Exact().visit(ast.parse(src))), '<helper>', 'exec'), ns)
    expr = ast.fix_missing_locations(ast.Expression(_Exact().visit(ast.parse(body, mode='eval').body)))
    env = {k: v if isinstance(v, int) else Fr(v) for k, v in args.items()}
    return Fr(eval(compile(expr, '<body>', 'eval'), ns, env))


def make(rng: random.Random) -> dict:
    fam = rng.choice(FAMILIES)
    q, helpers, args, body, wrong = fam(rng)
    try:
        exact = evaluate(helpers, args, body)
    except ZeroDivisionError:
        return make(rng)
    ndp = ndp_for(exact)
    value = rnd(exact, ndp)
    dis, seen = [], {value}
    for label, w in wrong + [('x10 slip', f"({body}) * 10"), ('/10 slip', f"({body}) / 10"),
                             ('x1000 slip', f"({body}) * 1000"), ('/1000 slip', f"({body}) / 1000"),
                             ('x100 slip', f"({body}) * 100"), ('/100 slip', f"({body}) / 100"),
                             ('x2 slip', f"({body}) * 2"), ('/2 slip', f"({body}) / 2"), ('sign', f"-({body})")]:
        try:
            v = rnd(evaluate(helpers, args, w), ndp)
        except (ZeroDivisionError, OverflowError):
            continue
        if v in seen or abs(v) > 10 ** 12 or (v == 0 and value != 0):
            continue
        seen.add(v)
        dis.append((program(helpers, args, w, ndp), v, f"{float(v):.{ndp}f}"))
    texts = [f"{float(value):.{ndp}f}"] + [d[2] for d in dis]
    if len(dis) < 3 or len(set(texts)) != len(texts):
        return make(rng)
    unit_note = f" Round to {ndp} decimal place{'s' if ndp != 1 else ''}." if ndp else " Round to the nearest whole number."
    return dict(question=q + unit_note, program=program(helpers, args, body, ndp), answer=value, options=[texts[0]],
                gold=0, variants=[4, 10], distractors=dis, meta={'family': fam.__name__, 'decimals': ndp})


TASK = "STEM calculation deterministic {k}-choice numeric selection"

if __name__ == '__main__':
    rng = random.Random(8)
    for _ in range(3):
        it = make(rng)
        print(it['question'], '->', it['options'][0], [d[2] for d in it['distractors']], it['meta'])
    print(it['program'])
