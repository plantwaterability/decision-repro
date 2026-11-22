"""synth_workflows.base -- the interface every workflow module implements, plus shared rendering helpers.

A WORKFLOW is a decision family (IT change approval, insurance claims triage, ...). It owns several SCHEMAS.
A SCHEMA is one case template:

    sample_factors(rng) -> dict          latent factors, JSON-serialisable, deliberately borderline / conflicting /
                                         missing a fair share of the time
    render(f, rng, surface) -> str       the state a model (and the teacher) reads; one of several surfaces
    questions() -> {qid: question}       exactly five typed questions, FIXED per schema (not per case)

Question dicts use the corpus shapes that trainers/cache_features.question_options reads:

    choice  {"type": "choice", "instructions": "...?", "criteria": {option_key: description, ...}}
    noul    {"type": "noul",   "instructions": "... .",  "criteria": {"yes": definition, "no": definition}}  (criteria optional)
    score   {"type": "score",  "instructions": "...?", "levels": [lowest, ..., highest]}

Standard library only (no numpy): the generator must run anywhere, deterministically under a seed.
See synth_workflows/README.md for how to add a workflow and research/typed_synth_spec.md for the contract.
"""
from __future__ import annotations

import json
import random
import re
from datetime import date, datetime, timedelta

GENERATOR_VERSION = "typed_synth-0.1.0"

QTYPES = ("choice", "noul", "score")
N_QUESTIONS = 5
# Option-count bounds. The benchmark uses choice K 4-5, score K 4-5, noul 2; 3 and 6 are allowed for variety.
K_BOUNDS = {"choice": (3, 6), "score": (3, 6), "noul": (2, 2)}
# Hard state-length bounds in characters. The benchmark's states run ~230-1,840 chars; aim for 250-2,000.
STATE_CHARS = (150, 4000)
SOFT_STATE_CHARS = (200, 2600)
IDENT = re.compile(r"^[a-z][a-z0-9_]{1,47}$")
QUESTION_FIELDS = {"type", "instructions", "criteria", "levels"}
# the surface that must exist in every schema: compact, key-sorted JSON (the benchmark's own state convention)
BENCH_SURFACE = "json"
MIN_SURFACES = 3
MIN_BENCH_SURFACE_SHARE = 0.3


# =========================================================================== question builders
def choice(instructions: str, criteria: dict) -> dict:
    """A `choice` question. Option keys are snake_case and stored alphabetically (the benchmark's order);
    every description must stand on its own, because the scored prompt shows the DESCRIPTION, not the key."""
    return {"type": "choice", "instructions": instructions.strip(),
            "criteria": {k: str(criteria[k]).strip() for k in sorted(criteria)}}


def noul(instructions: str, yes: str | None = None, no: str | None = None) -> dict:
    """A `noul` (yes/no) question: a declarative PROPOSITION, optionally with a definition of when each answer
    is right. Definitions are stored as criteria {"yes": ..., "no": ...}; question_options ignores them, the
    labeller shows them to its teachers."""
    q = {"type": "noul", "instructions": instructions.strip()}
    if yes is not None or no is not None:
        if not (yes and no):
            raise ValueError("noul definitions come in pairs (yes AND no)")
        q["criteria"] = {"yes": yes.strip(), "no": no.strip()}
    return q


def score(instructions: str, levels: list) -> dict:
    """A `score` question: an ordered rubric, LOWEST level first."""
    return {"type": "score", "instructions": instructions.strip(), "levels": [str(x).strip() for x in levels]}


def option_keys(q: dict) -> list:
    """Option keys in stored order, exactly as trainers/cache_features.question_options derives them."""
    if q["type"] == "choice":
        return list(q["criteria"])
    if q["type"] == "noul":
        return ["yes", "no"]
    if q["type"] == "score":
        return list(q["levels"])
    raise ValueError(f"unknown question type {q['type']!r}")


# =========================================================================== validation
def question_errors(qid: str, q: dict) -> list:
    """Every way one question dict breaks the contract (empty list = fine)."""
    e = []
    if not IDENT.match(qid):
        e.append(f"{qid}: qid must be snake_case [a-z][a-z0-9_]*")
    t = q.get("type")
    if t not in QTYPES:
        return e + [f"{qid}: unknown type {t!r}"]
    extra = set(q) - QUESTION_FIELDS
    if extra:
        e.append(f"{qid}: unexpected fields {sorted(extra)}")
    ins = str(q.get("instructions", "")).strip()
    if not ins:
        e.append(f"{qid}: empty instructions")
    elif t in ("choice", "score") and not ins.endswith("?"):
        e.append(f"{qid}: {t} instructions are a question ending in '?'")
    elif t == "noul" and not ins.endswith("."):
        e.append(f"{qid}: noul instructions are a declarative statement ending in '.'")
    lo, hi = K_BOUNDS[t]
    if t == "choice":
        c = q.get("criteria")
        if not isinstance(c, dict):
            return e + [f"{qid}: choice criteria must be a dict"]
        if not lo <= len(c) <= hi:
            e.append(f"{qid}: choice has {len(c)} options, want {lo}-{hi}")
        if list(c) != sorted(c):
            e.append(f"{qid}: choice criteria keys must be sorted (use base.choice)")
        for k, v in c.items():
            if not IDENT.match(k):
                e.append(f"{qid}: option key {k!r} must be snake_case")
            if len(str(v).strip()) < 8:
                e.append(f"{qid}: option {k!r} needs a real description")
        if len({str(v).strip().lower() for v in c.values()}) != len(c):
            e.append(f"{qid}: duplicate option descriptions")
        if "levels" in q:
            e.append(f"{qid}: choice must not carry levels")
    elif t == "score":
        lv = q.get("levels")
        if not isinstance(lv, list):
            return e + [f"{qid}: score levels must be a list"]
        if not lo <= len(lv) <= hi:
            e.append(f"{qid}: score has {len(lv)} levels, want {lo}-{hi}")
        if len({str(x).strip().lower() for x in lv}) != len(lv):
            e.append(f"{qid}: duplicate levels")
        if any(len(str(x).strip()) < 3 for x in lv):
            e.append(f"{qid}: empty level text")
        if any(str(x).strip() in {str(i) for i in range(10)} for x in lv):
            e.append(f"{qid}: a level text may not be a bare digit (it collides with positional label keys)")
        if "criteria" in q:
            e.append(f"{qid}: score keeps its rubric in `levels`, not `criteria`")
    else:  # noul
        if "levels" in q:
            e.append(f"{qid}: noul must not carry levels")
        c = q.get("criteria")
        if c is not None:
            if not isinstance(c, dict) or set(c) != {"yes", "no"}:
                e.append(f"{qid}: noul criteria must be exactly {{'yes': ..., 'no': ...}}")
            elif any(len(str(v).strip()) < 8 for v in c.values()):
                e.append(f"{qid}: noul definitions need real text")
    return e


def questions_errors(qs: dict) -> list:
    """Case-level checks: five questions, every type present, no type used more than three times."""
    e = []
    if not isinstance(qs, dict):
        return ["questions must be a dict {qid: question}"]
    if len(qs) != N_QUESTIONS:
        e.append(f"{len(qs)} questions, want exactly {N_QUESTIONS}")
    for qid, q in qs.items():
        e.extend(question_errors(qid, q))
    n = {t: sum(1 for q in qs.values() if q.get("type") == t) for t in QTYPES}
    for t in QTYPES:
        if n[t] < 1:
            e.append(f"no {t} question (every case mixes all three types)")
        if n[t] > 3:
            e.append(f"{n[t]} {t} questions (at most 3 of one type)")
    return e


def state_errors(state, surface: str) -> list:
    e = []
    if not isinstance(state, str) or not state.strip():
        return ["empty state"]
    lo, hi = STATE_CHARS
    if not lo <= len(state) <= hi:
        e.append(f"state is {len(state)} chars, want {lo}-{hi}")
    if surface == BENCH_SURFACE:
        try:
            obj = json.loads(state)
        except ValueError:
            return e + [f"surface {BENCH_SURFACE!r} must be JSON"]
        if not isinstance(obj, dict):
            e.append(f"surface {BENCH_SURFACE!r} must be a JSON object")
        elif state != dumps(obj):
            e.append(f"surface {BENCH_SURFACE!r} must be rendered with base.dumps (compact, sorted keys)")
    return e


# =========================================================================== schema / workflow
class Schema:
    """One case template. Subclass it and set `name`, `description`, `surfaces`; implement the three methods.

    `surfaces` maps a surface name to a sampling weight. It must contain "json" (compact sorted-key JSON, the
    benchmark's convention, rendered with `dumps`) at a share >= 0.3, and at least three surfaces in all.
    """

    name: str = ""
    description: str = ""
    surfaces: dict = {}

    def sample_factors(self, rng: random.Random) -> dict:
        raise NotImplementedError

    def render(self, f: dict, rng: random.Random, surface: str) -> str:
        raise NotImplementedError

    def questions(self) -> dict:
        raise NotImplementedError

    # ------------------------------------------------------------------ helpers the base provides
    def pick_surface(self, rng: random.Random) -> str:
        names = list(self.surfaces)
        return rng.choices(names, weights=[self.surfaces[n] for n in names])[0]

    def schema_errors(self) -> list:
        e = []
        if not IDENT.match(self.name or ""):
            e.append(f"schema name {self.name!r} must be snake_case")
        if len((self.description or "").strip()) < 10:
            e.append(f"{self.name}: needs a one-line description")
        if len(self.surfaces) < MIN_SURFACES:
            e.append(f"{self.name}: {len(self.surfaces)} surfaces, want >= {MIN_SURFACES}")
        if BENCH_SURFACE not in self.surfaces:
            e.append(f"{self.name}: no {BENCH_SURFACE!r} surface")
        else:
            tot = sum(self.surfaces.values())
            if tot <= 0 or self.surfaces[BENCH_SURFACE] / tot < MIN_BENCH_SURFACE_SHARE:
                e.append(f"{self.name}: {BENCH_SURFACE!r} surface share must be >= {MIN_BENCH_SURFACE_SHARE}")
        try:
            e.extend(f"{self.name}: {x}" for x in questions_errors(self.questions()))
        except Exception as ex:  # noqa: BLE001
            e.append(f"{self.name}: questions() raised {type(ex).__name__}: {ex}")
        return e


class Workflow:
    """A decision family. Instantiate one per module as the module-level name `WORKFLOW`."""

    def __init__(self, name: str, title: str, description: str, domain: str, schemas, notes: str = ""):
        self.name, self.title, self.description = name, title, description
        self.domain, self.notes = domain, notes
        self.schemas = tuple(schemas)

    def schema(self, name: str) -> Schema:
        for s in self.schemas:
            if s.name == name:
                return s
        raise KeyError(f"{self.name} has no schema {name!r}")

    def workflow_errors(self) -> list:
        e = []
        if not IDENT.match(self.name or ""):
            e.append(f"workflow name {self.name!r} must be snake_case")
        if len(self.schemas) < 5:
            e.append(f"{self.name}: {len(self.schemas)} schemas, want >= 5")
        names = [s.name for s in self.schemas]
        if len(set(names)) != len(names):
            e.append(f"{self.name}: duplicate schema names")
        for s in self.schemas:
            e.extend(f"{self.name}/{x}" for x in s.schema_errors())
        return e


def case_rng(seed, workflow: str, schema: str, index: int) -> random.Random:
    """The per-case RNG. String seeding is sha512-based: stable across processes and PYTHONHASHSEED."""
    return random.Random(f"typed_synth|{seed}|{workflow}|{schema}|{index}")


def make_case(schema: Schema, rng: random.Random, surface: str | None = None) -> dict:
    """factors -> (surface, state) plus the schema's five questions. Raises if anything breaks the contract."""
    f = schema.sample_factors(rng)
    surface = surface or schema.pick_surface(rng)
    state = schema.render(f, rng, surface)
    qs = schema.questions()
    errs = state_errors(state, surface) + questions_errors(qs)
    if errs:
        raise ValueError(f"{schema.name}/{surface}: " + "; ".join(errs))
    factors = jsonable(f)
    json.dumps(factors)  # must serialise
    return {"state": state, "questions": qs, "factors": factors, "surface": surface}


# =========================================================================== sampling helpers
def chance(rng: random.Random, p: float) -> bool:
    return rng.random() < p


def pick(rng: random.Random, options, weights=None):
    options = list(options)
    return rng.choices(options, weights=weights)[0] if weights else rng.choice(options)


def pick_w(rng: random.Random, weighted: dict):
    """Pick a key of {value: weight}."""
    return pick(rng, list(weighted), list(weighted.values()))


def regime(rng: random.Random, clear=0.3, borderline=0.45, conflicting=0.25) -> str:
    """How hard this case should be. Use it to steer the sampler: 'clear' cases have aligned signals,
    'borderline' cases sit near a policy threshold, 'conflicting' cases carry signals that point opposite ways
    (often with a key fact missing). The benchmark's gold is soft most of the time (mean max-p ~0.66), so
    keep 'clear' a minority."""
    return pick(rng, ["clear", "borderline", "conflicting"], [clear, borderline, conflicting])


def near(rng: random.Random, threshold: float, spread: float, lo=None, hi=None, integer=False, digits=2):
    """A value close to `threshold` (triangular, +-spread): the borderline cases a careful judge hesitates on."""
    v = rng.triangular(threshold - spread, threshold + spread, threshold)
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return int(round(v)) if integer else round(v, digits)


def between(rng: random.Random, lo: float, hi: float, integer=False, digits=2):
    v = rng.uniform(lo, hi)
    return int(round(v)) if integer else round(v, digits)


def maybe(rng: random.Random, value, p_missing: float):
    """`value`, or None (missing information) with probability p_missing."""
    return None if rng.random() < p_missing else value


def lognormal_amount(rng: random.Random, median: float, sigma: float = 0.8, digits=2):
    import math
    return round(median * math.exp(rng.gauss(0.0, sigma)), digits)


# =========================================================================== rendering helpers
def dumps(obj) -> str:
    """The benchmark's state convention: compact JSON (', ' / ': ' separators), sorted keys, UTF-8 kept."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def dumps_pretty(obj, rng: random.Random) -> str:
    """Indented JSON with keys in a shuffled (not sorted) order: a second JSON surface."""
    return json.dumps(shuffle_keys(obj, rng), ensure_ascii=False, indent=rng.choice([2, 2, 4]))


def shuffle_keys(obj, rng: random.Random):
    if isinstance(obj, dict):
        items = list(obj.items())
        rng.shuffle(items)
        return {k: shuffle_keys(v, rng) for k, v in items}
    if isinstance(obj, list):
        return [shuffle_keys(v, rng) for v in obj]
    return obj


def prune(obj):
    """Drop None-valued keys recursively: missing information rendered as an ABSENT field. A section left
    empty by the pruning disappears too (an artefact does not print an empty block)."""
    if isinstance(obj, dict):
        out = {k: prune(v) for k, v in obj.items() if v is not None}
        return {k: v for k, v in out.items() if v != {}}
    if isinstance(obj, list):
        return [prune(v) for v in obj if v is not None]
    return obj


MISSING_WORDS = ["unknown", "not provided", "TBD", "pending", "n/a", "-", "see attachment", "not recorded"]


def missing_text(rng: random.Random) -> str:
    return rng.choice(MISSING_WORDS)


def show(value, rng: random.Random, fmt=str) -> str:
    """Text for a possibly-missing value: formatted, or a missing-value word."""
    return missing_text(rng) if value is None else fmt(value)


def kv_lines(pairs, rng: random.Random, style: str | None = None, skip_none: bool = True) -> str:
    """'Label: value' lines in one of several house styles. `pairs` is a list of (label, value);
    None values are skipped (skip_none) or shown as a missing-value word."""
    style = style or rng.choice(["colon", "colon", "aligned", "dash", "equals"])
    rows = []
    for k, v in pairs:
        if v is None:
            if skip_none:
                continue
            v = missing_text(rng)
        rows.append((str(k), str(v)))
    if not rows:
        return ""
    if style == "aligned":
        w = max(len(k) for k, _ in rows) + 1
        return "\n".join(f"{(k + ':').ljust(w)} {v}" for k, v in rows)
    if style == "dash":
        return "\n".join(f"- {k}: {v}" for k, v in rows)
    if style == "equals":
        return "\n".join(f"{k.lower().replace(' ', '_')} = {v}" for k, v in rows)
    return "\n".join(f"{k}: {v}" for k, v in rows)


def bullets(items, rng: random.Random, marker: str | None = None) -> str:
    marker = marker or rng.choice(["-", "*", "•"])
    return "\n".join(f"{marker} {x}" for x in items if x)


def numbered(items) -> str:
    return "\n".join(f"{i}. {x}" for i, x in enumerate([x for x in items if x], 1))


def table(headers, rows, rng: random.Random, style: str | None = None) -> str:
    """A small markdown or fixed-width table."""
    style = style or rng.choice(["markdown", "fixed"])
    rows = [[("" if c is None else str(c)) for c in r] for r in rows]
    if style == "markdown":
        out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
        out += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(out)
    w = [max(len(str(h)), *(len(r[i]) for r in rows)) if rows else len(str(h)) for i, h in enumerate(headers)]
    out = ["  ".join(str(h).ljust(w[i]) for i, h in enumerate(headers))]
    out += ["  ".join(c.ljust(w[i]) for i, c in enumerate(r)) for r in rows]
    return "\n".join(x.rstrip() for x in out)


def money(x, rng: random.Random | None = None, cur: str = "USD", style: str | None = None) -> str:
    if x is None:
        return "unknown"
    style = style or (rng.choice(["sym", "sym", "code", "plain"]) if rng else "sym")
    sym = {"USD": "$", "EUR": "€", "GBP": "£", "CAD": "CA$", "AUD": "A$"}.get(cur, "")
    whole = abs(x) >= 1000 and float(x).is_integer()
    body = f"{abs(x):,.0f}" if whole else f"{abs(x):,.2f}"
    neg = "-" if x < 0 else ""
    if style == "code" or not sym:
        return f"{neg}{cur} {body}"
    if style == "plain":
        return f"{neg}{abs(x):.2f} {cur}"
    return f"{neg}{sym}{body}"


def pct(x, digits: int = 1) -> str:
    return "unknown" if x is None else f"{x:.{digits}f}%"


DATE_STYLES = ["iso", "iso", "us", "long", "short", "eu"]


def fmt_date(d: date, rng: random.Random | None = None, style: str | None = None) -> str:
    if d is None:
        return "unknown"
    style = style or (rng.choice(DATE_STYLES) if rng else "iso")
    if style == "us":
        return f"{d.month:02d}/{d.day:02d}/{d.year}"
    if style == "eu":
        return f"{d.day:02d}.{d.month:02d}.{d.year}"
    if style == "long":
        return d.strftime("%B ") + str(d.day) + d.strftime(", %Y")
    if style == "short":
        return d.strftime("%b ") + str(d.day)
    return d.isoformat()


def fmt_ts(t: datetime, rng: random.Random | None = None, style: str | None = None) -> str:
    style = style or (rng.choice(["iso", "iso", "log", "human"]) if rng else "iso")
    if style == "log":
        return t.strftime("%Y-%m-%d %H:%M:%S")
    if style == "human":
        return t.strftime("%a %b ") + str(t.day) + t.strftime(" %H:%M")
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def rand_date(rng: random.Random, start: date = date(2025, 1, 6), end: date = date(2027, 6, 30)) -> date:
    return start + timedelta(days=rng.randrange((end - start).days + 1))


def rand_ts(rng: random.Random, day: date | None = None, hours=(0, 23)) -> datetime:
    day = day or rand_date(rng)
    return datetime(day.year, day.month, day.day, rng.randint(*hours), rng.randrange(60), rng.randrange(60))


def ident(rng: random.Random, prefix: str, digits: int = 5, sep: str = "-") -> str:
    lo = 10 ** (digits - 1)
    return f"{prefix}{sep}{rng.randrange(lo, lo * 10)}"


def hexid(rng: random.Random, n: int = 8) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


FIRST = ["Amara", "Bogdan", "Chiara", "Dmitri", "Elif", "Farid", "Greta", "Hiroshi", "Ines", "Jonas", "Kavya",
         "Lukas", "Mei", "Nnamdi", "Olga", "Pedro", "Quinn", "Rosa", "Sanjay", "Tamsin", "Uriel", "Vera", "Wes",
         "Ximena", "Yusuf", "Zofia", "Aiden", "Beatriz", "Callum", "Daria", "Emeka", "Fiona", "Gustavo", "Hana",
         "Idris", "Julia", "Kofi", "Leilani", "Marek", "Noor", "Oskar", "Priya", "Rafael", "Sade", "Tomas",
         "Ulla", "Viktor", "Wen", "Yara", "Zane", "Anouk", "Brendan", "Carmen", "Deshawn", "Esther", "Felix",
         "Gideon", "Harriet", "Ivo", "Jun"]
LAST = ["Abara", "Bergström", "Castellano", "Duarte", "Eklund", "Fernandes", "Gallagher", "Haddad", "Ivanova",
        "Jansen", "Kowalczyk", "Lindqvist", "Mbeki", "Nakamura", "Okafor", "Petrov", "Quintero", "Rahman",
        "Sørensen", "Takahashi", "Umarov", "Varga", "Whitfield", "Xu", "Yilmaz", "Zielinski", "Ashworth",
        "Brennan", "Chowdhury", "Delacroix", "Esposito", "Fitzgerald", "Grünewald", "Holloway", "Iqbal",
        "Jovanović", "Kaplan", "Lowery", "Moreau", "Nguyen", "Oyelaran", "Pereira", "Rasmussen", "Silva",
        "Tan", "Underwood", "Vasquez", "Weber", "Yamada", "Zhou", "Achterberg", "Brandt", "Coleman", "Dvořák",
        "Ellison", "Fontaine", "Gupta", "Horvath", "Ibáñez", "Kim"]
ORG_A = ["Alder", "Basalt", "Cobalt", "Driftwood", "Ember", "Fernhill", "Granite", "Harbor", "Ironbark", "Juniper",
         "Keystone", "Lakeshore", "Meridian", "Northgate", "Oakridge", "Pinecrest", "Quarry", "Redfern", "Saltmarsh",
         "Tidewater", "Upland", "Valewood", "Westbrook", "Yarrow", "Zephyr", "Ashgrove", "Bluestem", "Copperline",
         "Dunmore", "Elmstead"]
ORG_B = {
    "company": ["Industries", "Logistics", "Systems", "Holdings", "Works", "Labs", "Partners", "Supply Co.",
                "Manufacturing", "Group", "Technologies", "Foods", "Outfitters", "Freight", "Analytics"],
    "clinic": ["Medical Group", "Health", "Clinic", "Family Practice", "Hospital", "Care Partners"],
    "school": ["University", "College", "Institute", "Academy"],
    "bank": ["Savings", "Credit Union", "Bank", "Mortgage", "Lending"],
    "insurer": ["Mutual", "Assurance", "Insurance", "Casualty"],
}


def first_name(rng: random.Random) -> str:
    return rng.choice(FIRST)


def last_name(rng: random.Random) -> str:
    return rng.choice(LAST)


def person(rng: random.Random) -> str:
    return f"{first_name(rng)} {last_name(rng)}"


def initials_name(rng: random.Random) -> str:
    return f"{first_name(rng)[0]}. {last_name(rng)}"


# Pool combinations that name a well-known real business. org() swaps such a suffix for the next one in the pool,
# deterministically and WITHOUT an extra rng draw, so every other case and every rng stream is unchanged.
REAL_NAME_CLASHES = {"Harbor Freight", "Keystone Foods", "Keystone Logistics", "Meridian Health", "Juniper Systems",
                     "Northgate Group", "Tidewater Logistics", "Granite Group"}


def org(rng: random.Random, kind: str = "company") -> str:
    a, suffixes = rng.choice(ORG_A), ORG_B.get(kind, ORG_B["company"])
    i = rng.randrange(len(suffixes))
    for k in range(len(suffixes)):
        name = f"{a} {suffixes[(i + k) % len(suffixes)]}"
        if name not in REAL_NAME_CLASHES:
            return name
    return name


def email_addr(name: str, domain: str) -> str:
    parts = re.sub(r"[^a-z ]", "", name.lower().replace("ø", "o").replace("ü", "u")).split()
    return f"{parts[0]}.{parts[-1]}@{domain}" if parts else f"user@{domain}"


def domain_of(org_name: str) -> str:
    return re.sub(r"[^a-z]", "", org_name.lower().split()[0]) + rng_tld(org_name)


def rng_tld(seed_text: str) -> str:
    return [".com", ".io", ".net", ".co", ".org"][sum(map(ord, seed_text)) % 5]


TYPO = [("the ", "teh "), ("receive", "recieve"), ("separate", "seperate"), ("until", "untill"),
        ("definitely", "definately"), ("tomorrow", "tommorow"), ("which", "wich"), ("because", "becuase")]


def noisy(text: str, rng: random.Random, p: float = 0.2) -> str:
    """Light human noise for FREE-TEXT fields only (never inside JSON keys or numbers): a typo, a doubled space,
    a lower-cased sentence start, a dropped final period. Applied with probability p per kind."""
    if not text:
        return text
    if rng.random() < p:
        for a, b in rng.sample(TYPO, len(TYPO)):
            if a in text:
                text = text.replace(a, b, 1)
                break
    if rng.random() < p and ". " in text:
        i = text.index(". ")
        text = text[: i + 1] + "  " + text[i + 2:]
    if rng.random() < p / 2:
        text = text[0].lower() + text[1:]
    if rng.random() < p and text.endswith("."):
        text = text[:-1]
    return text


def join_sentences(parts) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip())


def jsonable(obj):
    """Factors -> JSON-safe values (dates to ISO strings, tuples/sets to lists, floats rounded)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, float):
        return round(obj, 4)
    return obj
