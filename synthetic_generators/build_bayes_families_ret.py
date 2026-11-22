"""build_bayes_families_ret.py -- RETRIEVAL-STYLE Bayesian-evidence families (imported by build_bayes_families.py; design:
tmp/bayes_families/design.md section R; interface: tmp/bayes_families/INTERFACE.md).

Built to transfer to the Decision Index "retrieval & classification" competencies (claim verification over several evidence documents
as HoVer; per-candidate relevance as BRIGHT) WITHOUT using or imitating any DI item: every entity, name, value and sentence is
generated here from syllables and templates (no real entity, no Wikipedia / StackExchange / product / intent text).

A RETRIEVAL WORLD: a fictional entity E of a domain kind (bridge, company, ship, mountain, festival; held-out domains medicine,
software, club) with attributes (a year, a town, a person), and for two-hop claims the person P = E's person attribute with P's own
birthplace. EVIDENCE DOCUMENTS {title, text}: the KEY document (E's document stating the asked attribute, plus filler facts about E),
NEAR-MISS documents (hn_entity: a similarly named other entity with the asked attribute quoting a listed / claimed value;
hn_attribute: E's document WITHOUT the asked attribute, quoting a listed / claimed value for another attribute), and DISTRACTORS
(another entity of another kind).

Record kinds (one question each; build_jev_data schema; the DI converter's object-state style or prose, chosen per group):
  verify     claim + evidence documents -> SUPPORTED / NOT SUPPORTED (choice; deterministic: SUPPORTED iff the documents state every
             fact the claim needs; refuted and not-enough-information are both NOT SUPPORTED, as HoVer). hard target
  relevance  query + ONE candidate document -> relevant yes / no (choice; deterministic: the document states the asked fact about the
             asked entity). hard target
  answer     question + passages from archives with STATED accuracy (verified 95%, unverified 70%; a wrong archive's value is uniform
             over the other listed answers; independent errors; uniform prior over the K listed answers) -> exact posterior over the
             answers (choice, soft); no key passage -> exactly the prior (the calibrated abstention target)
Families (meta.bf.family; members differ in ONE factor)
  ret_hard_negatives     base / hn_entity / hn_attribute / hn_both (easy distractors replaced IN PLACE by near-misses): verify and
                         answer targets unchanged; plus relevance members rel_key (yes) / rel_distractor / rel_hn_entity /
                         rel_hn_attribute (no)
  ret_answerability      answerable / unans_removed (key -> distractor) / unans_attribute / unans_entity (key -> near-miss): verify
                         SUPPORTED -> NOT SUPPORTED; answer -> the prior
  ret_missing_evidence   variant "multikey" (answer): full (2-3 key passages) / drop / retract (shown + "must be ignored") / drop_all;
                         variant "twohop" (verify): full / drop_hop1 / drop_hop2 / retract_hop2 / hn_hop2 (the hop-2 document
                         replaced by a near-miss person): SUPPORTED -> NOT SUPPORTED
  ret_distractors        base / plus2 / plus5 / shuffled: verify and answer unchanged
  ret_evidence_strength  k1 / k2 / k3 (agreeing passages from independent unverified archives) / strong (one verified): answer only
Generators: ret_fact (trained); ret_table (HELD OUT: test_gen only; the same worlds rendered as field records). Regime: train <= 7
documents, test_reg 10-16 documents with K 6-8 listed answers.
"""
from __future__ import annotations

import json
import random
from fractions import Fraction

import build_bayes_families as bf

RET_FAMILIES = ("ret_hard_negatives", "ret_answerability", "ret_missing_evidence", "ret_distractors", "ret_evidence_strength")
RET_TRAIN_GENS = ("ret_fact",)
RET_HELDOUT_GENS = ("ret_table",)
ACC = {"verified": Fraction(95, 100), "unverified": Fraction(70, 100)}
TRAIN_MAX_M, REG_MIN_M = 7, 10

SYL_A = ("Var", "Cor", "Hal", "Ost", "Bren", "Lis", "Mor", "Tal", "Wen", "Kel", "Dra", "Fen", "Gor", "Ith", "Jun", "Lor", "Nes",
         "Pel", "Quen", "Rav", "Sel", "Tor", "Ul", "Vel", "Yar", "Zen", "Ard", "Bel", "Cal", "Dun", "Esk", "Fal", "Hav", "Kor")
SYL_B = ("holt", "wen", "mere", "dale", "ford", "ick", "ora", "an", "eth", "isk", "ova", "ard", "ane", "bury", "stead", "wick",
         "oth", "ella", "und", "rin", "ast", "ock", "ley", "mar")
FIRST = ("Ada", "Bram", "Cleo", "Dov", "Edda", "Finn", "Greta", "Hugo", "Ilse", "Joran", "Kaia", "Lev", "Mira", "Nils", "Oona",
         "Pavel", "Rhea", "Soren", "Tilda", "Uri", "Vera", "Wim", "Yusuf", "Zora")


def name(R):
    return R.choice(SYL_A) + R.choice(SYL_B)


def city(R):
    return name(R) + R.choice(("", "", "ton", "by", "burg", "port"))


def person(R):
    return f"{R.choice(FIRST)} {name(R)}"


def value(R, vk):
    if vk == "year":
        return str(1850 + R.randrange(171))
    if vk == "city":
        return city(R)
    if vk == "person":
        return person(R)
    raise ValueError(vk)


def T(*a):
    return a


# attribute: (value kind, question, statement templates, field name, near-miss templates (same value kind, another attribute),
#             claim template, query template)
DOMAINS = {
    "bridge": {"make": lambda R: f"{name(R)} Bridge", "similar": ("East {n}", "Old {n}", "{n} Viaduct"), "person": "designer", "attrs": {
        "opened": T("year", "In which year did the {e} open?", ("The {e} opened to traffic in {v}.", "Traffic first crossed the {e} in {v}."),
                    "opened", ("The {e} was closed for repairs in {v}.", "Work on the {e} began in {v}."), "The {e} opened in {v}.",
                    "opening year of the {e}"),
        "designer": T("person", "Who designed the {e}?", ("The {e} was designed by {v}.", "{v} drew up the design of the {e}."), "designer",
                      ("The {e} was inspected in its first year by {v}.",), "The {e} was designed by {v}.", "designer of the {e}"),
        "town": T("city", "In which town does the {e} stand?", ("The {e} stands in {v}.", "{v} is home to the {e}."), "town",
                  ("The {e}'s steel was shipped from {v}.",), "The {e} stands in {v}.", "town where the {e} stands")}},
    "company": {"make": lambda R: f"{name(R)} {R.choice(('Works', 'Holdings', 'Textiles', 'Instruments'))}",
                "similar": ("{n} Group", "New {n}", "{n} International"), "person": "founder", "attrs": {
        "founded": T("year", "In which year was {e} founded?", ("{e} was founded in {v}.", "{e} began trading in {v}."), "founded",
                     ("{e} moved to a new head office in {v}.", "{e} listed its shares in {v}."), "{e} was founded in {v}.",
                     "founding year of {e}"),
        "hq": T("city", "In which city is {e} headquartered?", ("{e} is headquartered in {v}.", "The head office of {e} is in {v}."),
                "headquarters", ("{e} opened a sales office in {v}.",), "{e} is headquartered in {v}.", "headquarters city of {e}"),
        "founder": T("person", "Who founded {e}?", ("{e} was founded by {v}.", "{v} started {e}."), "founder",
                     ("{e}'s first chief engineer was {v}.",), "{e} was founded by {v}.", "founder of {e}")}},
    "ship": {"make": lambda R: f"the {R.choice(('MV', 'SS', 'RV'))} {name(R)}", "similar": ("{n} II", "{n} Star"), "person": "captain",
             "attrs": {
        "launched": T("year", "In which year was {e} launched?", ("{e} was launched in {v}.", "{e} first took to the water in {v}."),
                      "launched", ("{e} was refitted in {v}.", "{e} changed owners in {v}."), "{e} was launched in {v}.",
                      "launch year of {e}"),
        "port": T("city", "What is the home port of {e}?", ("{e} is registered at the port of {v}.", "{e}'s home port is {v}."),
                  "home port", ("{e} was built in a yard near {v}.",), "{e}'s home port is {v}.", "home port of {e}"),
        "captain": T("person", "Who was the first captain of {e}?", ("{e}'s first captain was {v}.", "{v} commanded {e} on her first voyage."),
                     "first captain", ("{e} was named by {v}.",), "{e}'s first captain was {v}.", "first captain of {e}")}},
    "mountain": {"make": lambda R: f"Mount {name(R)}", "similar": ("{n} North", "Little {n}"), "person": "climber", "attrs": {
        "ascent": T("year", "In which year was {e} first climbed?", ("{e} was first climbed in {v}.", "The first ascent of {e} was made in {v}."),
                    "first ascent", ("{e} was first surveyed in {v}.", "A hut was built below {e} in {v}."), "{e} was first climbed in {v}.",
                    "year of the first ascent of {e}"),
        "climber": T("person", "Who first climbed {e}?", ("{e} was first climbed by {v}.", "{v} reached the top of {e} first."),
                     "first climber", ("{e} was first mapped by {v}.",), "{e} was first climbed by {v}.", "first climber of {e}"),
        "village": T("city", "Which village lies at the foot of {e}?", ("The village of {v} lies at the foot of {e}.",
                                                                       "{e} rises above the village of {v}."), "village",
                     ("Climbers bound for {e} often stay in {v}.",), "The village of {v} lies at the foot of {e}.",
                     "village at the foot of {e}")}},
    "festival": {"make": lambda R: f"the {name(R)} {R.choice(('Folk', 'Film', 'Lantern', 'Harvest'))} Festival",
                 "similar": ("{n} Winter Fair", "{n} Youth Festival"), "person": "founder", "attrs": {
        "first": T("year", "In which year was {e} first held?", ("{e} was first held in {v}.", "{e} began in {v}."), "first held",
                   ("{e} moved to a larger site in {v}.", "{e} was cancelled in {v}."), "{e} was first held in {v}.",
                   "year {e} was first held"),
        "host": T("city", "Which town hosts {e}?", ("{e} is held every year in {v}.", "{v} hosts {e}."), "host town",
                  ("{e}'s organisers are based in {v}.",), "{e} is held every year in {v}.", "host town of {e}"),
        "founder": T("person", "Who founded {e}?", ("{e} was founded by {v}.", "{v} started {e}."), "founder",
                     ("{e}'s first poster was drawn by {v}.",), "{e} was founded by {v}.", "founder of {e}")}},
}
HELDOUT_DOMAINS = {
    "medicine": {"make": lambda R: name(R) + R.choice(("ex", "ol", "ide", "ab")), "similar": ("{n} XR", "{n}-B"), "person": "inventor",
                 "attrs": {
        "approved": T("year", "In which year was {e} approved?", ("{e} was approved for use in {v}.", "Regulators cleared {e} in {v}."),
                      "approved", ("A trial of {e} ended in {v}.", "{e}'s patent was filed in {v}."), "{e} was approved in {v}.",
                      "approval year of {e}"),
        "inventor": T("person", "Who discovered {e}?", ("{e} was discovered by {v}.", "{v} first made {e}."), "discoverer",
                      ("{e}'s first trial was led by {v}.",), "{e} was discovered by {v}.", "discoverer of {e}"),
        "plant": T("city", "In which city is {e} manufactured?", ("{e} is made in a plant in {v}.", "The factory that produces {e} is in {v}."),
                   "manufactured in", ("{e} is packaged in {v}.",), "{e} is manufactured in {v}.", "city where {e} is made")}},
    "software": {"make": lambda R: f"{name(R)}{R.choice(('DB', 'Kit', 'JS', 'Flow'))}", "similar": ("{n} Lite", "{n}2"), "person": "author",
                 "attrs": {
        "released": T("year", "In which year was {e} first released?", ("{e} was first released in {v}.", "Version 1.0 of {e} came out in {v}."),
                      "first release", ("{e} was rewritten in {v}.", "{e} changed its licence in {v}."), "{e} was first released in {v}.",
                      "first release year of {e}"),
        "author": T("person", "Who wrote {e}?", ("{e} was written by {v}.", "{v} is the original author of {e}."), "author",
                    ("{e}'s logo was designed by {v}.",), "{e} was written by {v}.", "original author of {e}"),
        "office": T("city", "In which city is the team behind {e} based?", ("The team behind {e} is based in {v}.", "{e} is developed in {v}."),
                    "team city", ("{e}'s first conference was held in {v}.",), "The team behind {e} is based in {v}.",
                    "city where {e} is developed")}},
    "club": {"make": lambda R: f"{name(R)} {R.choice(('Rovers', 'Athletic', 'United', 'Wanderers'))}", "similar": ("{n} Reserves", "{n} Women"),
             "person": "coach", "attrs": {
        "founded": T("year", "In which year was {e} founded?", ("{e} was founded in {v}.", "{e} played its first match in {v}."), "founded",
                     ("{e} moved to a new ground in {v}.", "{e} won its first title in {v}."), "{e} was founded in {v}.",
                     "founding year of {e}"),
        "ground": T("city", "In which town does {e} play its home games?", ("{e} plays its home games in {v}.", "{e}'s ground is in {v}."),
                    "home town", ("{e} trains in {v}.",), "{e} plays its home games in {v}.", "home town of {e}"),
        "coach": T("person", "Who was the first coach of {e}?", ("{e}'s first coach was {v}.", "{v} coached {e} first."), "first coach",
                   ("{e}'s first captain was {v}.",), "{e}'s first coach was {v}.", "first coach of {e}")}},
}
PERSON_HOP2 = ("city", ("{p} was born in {v}.", "{p} was born in the town of {v}."), "birthplace",
               ("{p} retired to {v}.", "{p} later taught in {v}."))
RET_HELDOUT_DOMAIN_IDS = tuple(HELDOUT_DOMAINS)
VERIFY_OPTS = {"supported": "SUPPORTED: the evidence documents establish every part of the claim",
               "not_supported": "NOT SUPPORTED: the evidence refutes the claim or does not establish all of it"}
REL_OPTS = {"yes": "Relevant: the document states what the query asks for", "no": "Not relevant"}


def options(R, vk, K, must=()):
    out = list(must)
    while len(out) < K:
        v = value(R, vk)
        if v not in out:
            out.append(v)
    R.shuffle(out)
    if vk == "year":
        out.sort()
    return out


class World:
    def __init__(self, G):
        R = G.Rw
        doms = HELDOUT_DOMAINS if bf.slice_of(G.split) == "dom" else DOMAINS
        self.dom_id = sorted(doms)[G.Rr.randrange(len(doms))]
        self.dom = doms[self.dom_id]
        self.attr_id = sorted(self.dom["attrs"])[G.Rr.randrange(len(self.dom["attrs"]))]
        self.attr = self.dom["attrs"][self.attr_id]
        self.vk = self.attr[0]
        self.entity = self.dom["make"](R)
        self.K = (6 + R.randrange(3)) if G.reg == "heldout" else (3 + R.randrange(3))
        self.opts = options(R, self.vk, self.K)
        self.truth = R.randrange(self.K)
        self.facts = {a: value(R, self.dom["attrs"][a][0]) for a in self.dom["attrs"]}   # E's other attributes (filler facts)
        self.facts[self.attr_id] = self.opts[self.truth]
        self.ri = {"i_tpl": G.Rr.randrange(2), "style": G.Rr.choice(("json", "prose")) if G.gen_name == "ret_fact" else "table"}
        self.R = G.Rp

    def report(self, R, acc):
        if R.random() < float(acc):
            return self.truth
        o = R.randrange(self.K - 1)
        return o if o < self.truth else o + 1

    def _sent(self, attr_id, v, tpl=0, e=None):
        a = self.dom["attrs"][attr_id]
        s = a[2][(self.ri["i_tpl"] + tpl) % len(a[2])].format(e=e or self.entity, v=v)
        return s[0].upper() + s[1:]

    def fillers(self, R, exclude, n):
        others = [a for a in sorted(self.dom["attrs"]) if a not in exclude]
        R.shuffle(others)
        return [(a, self._sent(a, self.facts[a])) for a in others[:n]]

    # documents: {kind, title, sents [(attr, text)], src, opt (answer index of the asked attribute, if stated)}
    def key(self, R, src="verified", opt=None, tpl=0, value_override=None):
        o = self.report(R, ACC[src]) if opt is None else opt
        v = self.opts[o] if value_override is None else value_override
        sents = [(self.attr_id, self._sent(self.attr_id, v, tpl))] + self.fillers(R, {self.attr_id}, 1 + R.randrange(2))
        R.shuffle(sents)
        return {"kind": "key", "title": self.entity, "sents": sents, "src": src, "opt": o if value_override is None else None,
                "value": v}

    def hn_attribute(self, R, trap):
        tpls = self.attr[4]
        s = tpls[R.randrange(len(tpls))].format(e=self.entity, v=trap)
        sents = [("other", s[0].upper() + s[1:])] + self.fillers(R, {self.attr_id}, 1)
        R.shuffle(sents)
        return {"kind": "hn_attribute", "title": self.entity, "sents": sents, "src": R.choice(("verified", "unverified")), "opt": None,
                "value": trap}

    def similar_name(self, R):
        core = self.entity[4:] if self.entity.startswith("the ") else self.entity
        sim = self.dom["similar"][R.randrange(len(self.dom["similar"]))].format(n=core)
        return ("the " + sim) if self.entity.startswith("the ") else sim

    def hn_entity(self, R, trap):
        sim = self.similar_name(R)
        sents = [(self.attr_id, self._sent(self.attr_id, trap, 1, e=sim))]
        return {"kind": "hn_entity", "title": sim, "sents": sents, "src": R.choice(("verified", "unverified")), "opt": None,
                "value": trap}

    def distractor(self, R):
        doms = {**DOMAINS, **HELDOUT_DOMAINS}
        others = sorted(k for k in doms if k != self.dom_id)
        d = doms[others[R.randrange(len(others))]]
        e = d["make"](R)
        attrs = sorted(d["attrs"])
        R.shuffle(attrs)
        sents = []
        for a in attrs[:1 + R.randrange(2)]:
            t = d["attrs"][a][2][R.randrange(2)].format(e=e, v=value(R, d["attrs"][a][0]))
            sents.append(("other", t[0].upper() + t[1:]))
        return {"kind": "distractor", "title": e[0].upper() + e[1:], "sents": sents, "src": R.choice(("verified", "unverified")),
                "opt": None, "value": None}


# ============================================================================ rendering
def doc_text(d):
    return " ".join(s for _, s in d["sents"])


def cap(s):
    return s[0].upper() + s[1:]


def render_docs(W, docs, with_src):
    st = W.ri["style"]
    if st == "json":
        ev = [{"title": cap(d["title"]), "text": doc_text(d), **({"archive": d["src"]} if with_src else {})} for d in docs]
        return ev
    lines = []
    for i, d in enumerate(docs):
        if st == "table":
            fields = " | ".join(s for _, s in d["sents"])
            lines.append(f"[{i + 1}] record | title: {cap(d['title'])} | " + (f"archive: {d['src']} | " if with_src else "") + f"facts: {fields}")
        else:
            lines.append(f"[{i + 1}] " + (f"({d['src']} archive) " if with_src else "") + f"{cap(d['title'])}: {doc_text(d)}")
    return lines


def state_of(W, head, docs, with_src, extra=()):
    ev = render_docs(W, docs, with_src)
    if W.ri["style"] == "json":
        obj = dict(head)
        obj["evidence" if not with_src else "passages"] = ev
        if extra:
            obj["note"] = " ".join(extra)
        return json.dumps(obj, ensure_ascii=False)
    lines = [f"{k.capitalize()}: {v}" for k, v in head.items()]
    lines.append("Evidence documents:" if not with_src else "Retrieved passages:")
    return "\n".join(lines + ev + list(extra))


MODEL_LINE = ("Each passage comes from a separate archive. Verified archives state the correct value 95% of the time, unverified ones 70% "
              "of the time; when an archive is wrong, its value is equally likely to be any of the other listed answers, and archives err "
              "independently. A passage is evidence only for the facts it states about the entity it names. Before the passages are read, "
              "each listed answer is equally likely.")
VERIFY_Q = "Is the claim supported by the evidence documents? Choose one."
REL_TASK = "Decide whether the candidate document is relevant to the query."
REL_Q = "Is this candidate document relevant to the query?"
RETRACT = "Passage [{j}] was later found to come from a mislabelled file and must be ignored."
RETRACT_DOC = "Document [{j}] was later found to describe a different {kind}, so it must be ignored."


# ============================================================================ exact targets
def posterior(W, keys):
    w = [Fraction(1)] * W.K
    for p in keys:
        a = ACC[p["src"]]
        w = [x * (a if o == p["opt"] else (1 - a) / (W.K - 1)) for o, x in enumerate(w)]
    s = sum(w)
    return [x / s for x in w]


def round_vec(v, dp=bf.LABEL_DP):
    return [float(bf.round_exact(x, dp)) for x in v]


def doc_json(d):
    return {"kind": d["kind"], "title": d["title"], "src": d["src"], "opt": d["opt"], "value": d["value"],
            "attrs": [a for a, _ in d["sents"]]}


class RetGroup:
    def __init__(self, ns, split, family, gen, gi, attempt):
        key = f"{split}|{family}|{gen}|{gi}|{attempt}"
        self.ns, self.split, self.family, self.gen_name, self.gi, self.attempt = ns, split, family, gen, gi, attempt
        self.Rw = random.Random(bf.seed_int(ns, key, "world"))
        self.Rr = random.Random(bf.seed_int(ns, key, "render"))
        self.Rp = random.Random(bf.seed_int(ns, key, "passages"))
        self.gid = f"{split}/{family}/{gen}/{gi:05d}"
        self.reg = "heldout" if bf.slice_of(split) == "reg" else "train"
        self.members = []


def emit(B, G, W, member, kind, state, instr, crit, target, problem, factor, anchor=False, label_kind="soft", n_docs=0):
    keys = list(crit)
    values = round_vec(target)
    key = state + "\n" + instr
    if key in B.seen or key in B.forbid_text or any(key == m["state"] + "\n" + next(iter(m["questions"].values()))["instructions"]
                                                     for m in G.members):
        raise bf.Dup(key[:60])
    reg = "heldout" if n_docs >= REG_MIN_M else "train"
    fp = "ret|" + bf.sha16(json.dumps({"p": problem, "kind": kind, "m": member}, sort_keys=True))
    prior = [Fraction(1, len(keys))] * len(keys) if kind == "answer" else [Fraction(1, 2)] * 2
    meta = {"family": G.gen_name, "split": G.split, "format": kind, "context": W.dom_id, "quantity": "posterior" if kind == "answer" else "label",
            "generator": bf.VERSION, "seed_namespace": B.ns_public, "fingerprint": fp,
            "truth": {"value": W.opts[W.truth], "index": W.truth,
                      "source": "conditioned" if G.family == "ret_evidence_strength" else "model"},
            "exact": {"target": [bf.exact_str(x) for x in target]},
            "baselines": {"prior": round_vec(prior)},
            "bf": {"group": G.gid, "family": G.family, "generator": G.gen_name, "member": member, "record_kind": kind,
                   "anchor": bool(anchor), "holdout": bf.HOLDOUT_OF[G.split], "split": G.split,
                   "context_set": "heldout" if W.dom_id in HELDOUT_DOMAINS else "train", "regime": reg, "attempt": G.attempt,
                   "factor": factor, "style": W.ri["style"], "problem": problem}}
    rid = bf.sha16(f"{B.ns}|{G.split}|{G.gid}|{member}|{kind}|{state}|{instr}")
    qid = f"bf_{G.family}/{rid}"
    rec = {"id": f"bf-{G.family}-{G.split}-{rid}", "source": f"bayes_families/{G.family}", "license": bf.LICENSE,
           "base_id": f"bf/{G.gid}", "state": state, "questions": {qid: {"type": "choice", "instructions": instr, "criteria": crit}},
           "labels": {qid: dict(zip(keys, values))}, "label_kind": {qid: label_kind}, "meta": meta}
    G.members.append(rec)
    return rec


def answer_record(B, G, W, member, docs, factor, anchor=False, valid=None, extra=()):
    valid = [d for d in docs if d["kind"] == "key"] if valid is None else valid
    head = {"question": W.attr[1].format(e=W.entity)[0].upper() + W.attr[1].format(e=W.entity)[1:],
            "listed answers": "; ".join(W.opts), "reliability": MODEL_LINE}
    state = state_of(W, head, docs, True, extra)
    crit = {f"o{i + 1}": W.opts[i] for i in range(W.K)}
    tv = posterior(W, valid)
    problem = {"domain": W.dom_id, "attribute": W.attr_id, "entity": W.entity, "options": W.opts, "prior": "uniform",
               "accuracy": {k: bf.exact_str(v) for k, v in ACC.items()}, "docs": [doc_json(d) for d in docs],
               "valid_keys": [i for i, d in enumerate(docs) if any(d is v for v in valid)]}
    return emit(B, G, W, member, "answer", state, "Which listed answer is correct? Choose one.", crit, tv, problem, factor, anchor,
                n_docs=len(docs))


def verify_record(B, G, W, member, docs, claim, supported, factor, anchor=False, extra=(), needs=None, claim_value=None):
    head = {"claim": claim}
    state = state_of(W, head, docs, False, extra)
    tv = [Fraction(1), Fraction(0)] if supported else [Fraction(0), Fraction(1)]
    problem = {"domain": W.dom_id, "attribute": W.attr_id, "entity": W.entity, "claim": claim, "claim_value": claim_value,
               "docs": [doc_json(d) for d in docs], "needs": needs or [], "supported": bool(supported)}
    return emit(B, G, W, member, "verify", state, VERIFY_Q, dict(VERIFY_OPTS), tv, problem, factor, anchor, label_kind="hard",
                n_docs=len(docs))


def relevance_record(B, G, W, member, doc, relevant, factor, anchor=False):
    q = W.attr[6].format(e=W.entity)
    head = {"query": cap(q), "task": REL_TASK}
    cand = f"{cap(doc['title'])}: {doc_text(doc)}"
    if W.ri["style"] == "json":
        state = json.dumps(head, ensure_ascii=False)
        instr = json.dumps({"task": REL_Q, "candidate": cand}, ensure_ascii=False)
    elif W.ri["style"] == "table":
        state = f"Query: {cap(q)}\nTask: {REL_TASK}"
        instr = REL_Q + f" Candidate record | title: {cap(doc['title'])} | facts: " + " | ".join(s for _, s in doc["sents"])
    else:
        state = f"Query: {cap(q)}\nTask: {REL_TASK}"
        instr = f"{REL_Q} Candidate document: {cand}"
    tv = [Fraction(1), Fraction(0)] if relevant else [Fraction(0), Fraction(1)]
    problem = {"domain": W.dom_id, "attribute": W.attr_id, "entity": W.entity, "doc": doc_json(doc), "relevant": bool(relevant)}
    return emit(B, G, W, member, "relevance", state, instr, dict(REL_OPTS), tv, problem, factor, anchor, label_kind="hard", n_docs=1)


def n_easy(G):
    return (9 + G.Rp.randrange(7)) if G.reg == "heldout" else (1 + G.Rp.randrange(4))


def _place(R, items, x):
    i = R.randrange(len(items) + 1)
    return items[:i] + [x] + items[i:], i


def _claim(W, v):
    return W.attr[5].format(e=W.entity, v=v)[0].upper() + W.attr[5].format(e=W.entity, v=v)[1:]


# ============================================================================ families
def fam_ret_hard_negatives(B, G):
    W = World(G)
    R = G.Rp
    src = R.choice(("verified", "unverified"))
    key = W.key(G.Rw, src)
    ds = [W.distractor(R) for _ in range(max(2, n_easy(G)))]
    docs, ki = _place(R, ds, key)
    slots = [i for i in range(len(docs)) if i != ki]
    s1, s2 = R.sample(slots, 2)
    supported = R.random() < 0.5
    stated = key["value"]
    claim_v = stated if supported else next(o for o in W.opts if o != stated)
    # the near-misses quote the CLAIMED value (the trap): a reader matching surface strings is pulled towards SUPPORTED
    he, ha = W.hn_entity(R, claim_v), W.hn_attribute(R, claim_v)
    rep = lambda ps, i, p: ps[:i] + [p] + ps[i + 1:]
    mem = {"base": docs, "hn_entity": rep(docs, s1, he), "hn_attribute": rep(docs, s1, ha), "hn_both": rep(rep(docs, s1, he), s2, ha)}
    claim = _claim(W, claim_v)
    for name_, ps in mem.items():
        fac = {"replaced": [] if name_ == "base" else [s1 + 1] + ([s2 + 1] if name_ == "hn_both" else []), "key_pos": ki + 1,
               "claim_true_in_docs": supported}
        verify_record(B, G, W, name_, ps, claim, supported, fac, anchor=(name_ == "base"), needs=[ki], claim_value=claim_v)
        answer_record(B, G, W, name_, ps, fac, anchor=False)
    for name_, d, rel in (("rel_key", key, True), ("rel_distractor", docs[s1], False), ("rel_hn_entity", he, False),
                          ("rel_hn_attribute", ha, False)):
        relevance_record(B, G, W, name_, d, rel, {"candidate": name_[4:]}, anchor=(name_ == "rel_key"))


def fam_ret_answerability(B, G):
    W = World(G)
    R = G.Rp
    key = W.key(G.Rw, R.choice(("verified", "unverified")))
    docs, ki = _place(R, [W.distractor(R) for _ in range(n_easy(G))], key)
    claim = _claim(W, key["value"])
    rep = lambda p: docs[:ki] + [p] + docs[ki + 1:]
    mem = {"answerable": docs, "unans_removed": rep(W.distractor(R)), "unans_attribute": rep(W.hn_attribute(R, key["value"])),
           "unans_entity": rep(W.hn_entity(R, key["value"]))}
    for name_, ps in mem.items():
        fac = {"key_pos": ki + 1, "answerable": name_ == "answerable"}
        verify_record(B, G, W, name_, ps, claim, name_ == "answerable", fac, anchor=(name_ == "answerable"),
                      needs=[ki] if name_ == "answerable" else [], claim_value=key["value"])
        answer_record(B, G, W, name_, ps, fac)


def fam_ret_missing(B, G):
    W = World(G)
    R = G.Rp
    if R.random() < 0.5:                                         # variant multikey: the answer moves toward the prior
        nk = 2 + R.randrange(2)
        keys = [W.key(G.Rw, R.choice(("verified", "unverified")), tpl=i) for i in range(nk)]
        full = [W.distractor(R) for _ in range(n_easy(G))]
        for k in keys:
            full, _ = _place(R, full, k)
        j = R.randrange(nk)
        dropped = keys[j]
        jpos = full.index(dropped)
        fac = {"variant": "multikey", "keys": nk, "dropped_pos": jpos + 1}
        answer_record(B, G, W, "full", full, fac, anchor=True)
        answer_record(B, G, W, "drop", [d for d in full if d is not dropped], fac)
        answer_record(B, G, W, "retract", full, fac, valid=[k for k in keys if k is not dropped], extra=[RETRACT.format(j=jpos + 1)])
        answer_record(B, G, W, "drop_all", [d for d in full if d["kind"] != "key"], fac)
        return
    # variant twohop (verify): "The <person attribute> of E was born in C" needs E's document (E -> P) and P's document (P -> C)
    pa = W.dom["person"]
    P = W.facts[pa] if W.attr_id != pa else W.opts[W.truth]
    born = city(G.Rw)
    s1 = W._sent(pa, P)
    d1 = {"kind": "key", "title": W.entity, "sents": [(pa, s1)] + W.fillers(R, {pa}, 1), "src": "verified", "opt": None, "value": P}
    hop2 = PERSON_HOP2[1][R.randrange(2)].format(p=P, v=born)
    d2 = {"kind": "key", "title": P, "sents": [("birthplace", hop2), ("other", PERSON_HOP2[3][R.randrange(2)].format(p=P, v=city(R)))],
          "src": "verified", "opt": None, "value": born}
    sim_p = f"{R.choice([f for f in FIRST if not P.startswith(f + ' ')])} {P.split(' ', 1)[1]}"
    d2n = {"kind": "hn_entity", "title": sim_p, "sents": [("birthplace", PERSON_HOP2[1][0].format(p=sim_p, v=born))], "src": "verified",
           "opt": None, "value": born}
    role = W.dom["attrs"][pa][6].format(e=W.entity)
    claim = f"The {role} was born in {born}."
    full = [W.distractor(R) for _ in range(max(1, n_easy(G) - 1))]
    full, _ = _place(R, full, d1)
    full, _ = _place(R, full, d2)
    p1, p2 = full.index(d1), full.index(d2)
    fac = {"variant": "twohop", "hop1_pos": p1 + 1, "hop2_pos": p2 + 1}
    verify_record(B, G, W, "full", full, claim, True, fac, anchor=True, needs=[p1, p2])
    verify_record(B, G, W, "drop_hop1", [d for d in full if d is not d1], claim, False, fac)
    verify_record(B, G, W, "drop_hop2", [d for d in full if d is not d2], claim, False, fac)
    verify_record(B, G, W, "retract_hop2", full, claim, False, fac,
                  extra=[RETRACT_DOC.format(j=p2 + 1, kind="person")])
    verify_record(B, G, W, "hn_hop2", [d2n if d is d2 else d for d in full], claim, False, fac)


def fam_ret_distractors(B, G):
    W = World(G)
    R = G.Rp
    key = W.key(G.Rw, R.choice(("verified", "unverified")))
    nd = 1 if G.reg == "train" else 9
    base, ki = _place(R, [W.distractor(R) for _ in range(nd)], key)
    p2 = list(base)
    for _ in range(2):
        p2, _ = _place(R, p2, W.distractor(R))
    p5 = list(base)
    for _ in range(5):
        p5, _ = _place(R, p5, W.distractor(R))
    sh = list(base)
    for _ in range(20):
        R.shuffle(sh)
        if sh.index(key) != ki:
            break
    else:
        raise bf.Dup("shuffle")
    supported = R.random() < 0.5
    cv = key["value"] if supported else next(o for o in W.opts if o != key["value"])
    claim = _claim(W, cv)
    for name_, ps in (("base", base), ("plus2", p2), ("plus5", p5), ("shuffled", sh)):
        fac = {"n_docs": len(ps), "key_pos": ps.index(key) + 1}
        verify_record(B, G, W, name_, ps, claim, supported, fac, anchor=(name_ == "base"), needs=[ps.index(key)], claim_value=cv)
        answer_record(B, G, W, name_, ps, fac)


def fam_ret_strength(B, G):
    W = World(G)
    R = G.Rp
    o = W.report(G.Rw, ACC["unverified"])
    ks = [W.key(G.Rw, "unverified", opt=o, tpl=i) for i in range(3)]
    strong = W.key(G.Rw, "verified", opt=o)
    ds = [W.distractor(R) for _ in range(n_easy(G))]
    slots = sorted(R.sample(range(len(ds) + 3), 3))

    def with_keys(kp):
        out, it = [], iter(ds)
        for i in range(len(ds) + len(kp)):
            out.append(kp[slots.index(i)] if i in slots[:len(kp)] else next(it))
        return out
    mem = {"k1": with_keys(ks[:1]), "k2": with_keys(ks[:2]), "k3": with_keys(ks[:3]), "strong": with_keys([strong])}
    for name_, ps in mem.items():
        answer_record(B, G, W, name_, ps, {"agreeing_keys": sum(p["kind"] == "key" for p in ps),
                                            "archive": "verified" if name_ == "strong" else "unverified", "reported": W.opts[o]},
                      anchor=(name_ == "k1"))


RET_FAM_FN = {"ret_hard_negatives": fam_ret_hard_negatives, "ret_answerability": fam_ret_answerability,
              "ret_missing_evidence": fam_ret_missing, "ret_distractors": fam_ret_distractors,
              "ret_evidence_strength": fam_ret_strength}


def build_ret_group(B, split, family, gen, gi, max_attempts=200):
    last = None
    for att in range(max_attempts):
        G = RetGroup(B.ns, split, family, gen, gi, att)
        try:
            RET_FAM_FN[family](B, G)
        except bf.Dup as e:
            last = e
            bf.STATS[f"redraw/{family}/{gen}"] += 1
            continue
        fps = {r["meta"]["fingerprint"] for r in G.members}
        if split == "train" and fps & B.eval_fp:
            bf.STATS[f"redraw_overlap/{family}/{gen}"] += 1
            continue
        if split == "train" and any(r["meta"]["bf"]["regime"] != "train" for r in G.members):
            bf.STATS[f"redraw_regime/{family}/{gen}"] += 1
            continue
        if split in bf.CONFIRM_SPLITS and fps & B.public_fp:
            bf.STATS[f"redraw_public/{family}/{gen}"] += 1
            continue
        for r in G.members:
            B.seen.add(r["state"] + "\n" + next(iter(r["questions"].values()))["instructions"])
            if split != "train":
                B.eval_fp.add(r["meta"]["fingerprint"])
        return G
    raise RuntimeError(f"cannot build {split}/{family}/{gen}/{gi}: {last}")


# ============================================================================ check: independent recomputation + invariants
def recompute(m):
    """The exact target from meta.bf.problem alone."""
    b = m["bf"]
    pr = b["problem"]
    kind = b["record_kind"]
    if kind == "answer":
        K = len(pr["options"])
        acc = {k: bf.parse_exact(v) for k, v in pr["accuracy"].items()}
        w = [Fraction(1, K)] * K
        for i in pr["valid_keys"]:
            d = pr["docs"][i]
            a = acc[d["src"]]
            w = [x * (a if o == d["opt"] else (1 - a) / (K - 1)) for o, x in enumerate(w)]
        s = sum(w)
        return [x / s for x in w]
    if kind == "verify":
        # SUPPORTED iff every needed document is present, is a key document, and (single-hop) states the claimed value
        docs = pr["docs"]
        ok = bool(pr["needs"]) and all(i < len(docs) and docs[i]["kind"] == "key" for i in pr["needs"])
        if ok and len(pr["needs"]) == 1:
            ok = docs[pr["needs"][0]]["value"] == pr["claim_value"]
        return [Fraction(1), Fraction(0)] if ok else [Fraction(0), Fraction(1)]
    d = pr["doc"]
    rel = d["kind"] == "key" and d["title"] == pr["entity"] and pr["attribute"] in d["attrs"]
    return [Fraction(1), Fraction(0)] if rel else [Fraction(0), Fraction(1)]


def ret_invariants(fam, mem):
    e = []
    lab = lambda r: list(r["labels"][next(iter(r["labels"]))].values())
    by = {}
    for (m, k), r in mem.items():
        by.setdefault(k, {})[m] = r
    ans, ver, rel = by.get("answer", {}), by.get("verify", {}), by.get("relevance", {})
    if fam == "ret_hard_negatives":
        for kind, d in (("answer", ans), ("verify", ver)):
            for m, r in d.items():
                if lab(r) != lab(d["base"]):
                    e.append(f"{kind} {m}: target differs from base")
        if [lab(rel[m])[0] for m in ("rel_key", "rel_distractor", "rel_hn_entity", "rel_hn_attribute")] != [1.0, 0.0, 0.0, 0.0]:
            e.append("relevance targets")
    elif fam == "ret_answerability":
        K = len(ans["answerable"]["meta"]["bf"]["problem"]["options"])
        for m in ("unans_removed", "unans_attribute", "unans_entity"):
            if lab(ans[m]) != [float(bf.round_exact(Fraction(1, K), bf.LABEL_DP))] * K:
                e.append(f"{m}: answer is not the prior")
            if lab(ver[m]) != [0.0, 1.0]:
                e.append(f"{m}: verify is not NOT SUPPORTED")
        if lab(ver["answerable"]) != [1.0, 0.0]:
            e.append("answerable: verify is not SUPPORTED")
    elif fam == "ret_missing_evidence":
        if ans:
            if lab(ans["drop"]) != lab(ans["retract"]):
                e.append("drop / retract answers differ")
            K = len(ans["full"]["meta"]["bf"]["problem"]["options"])
            if lab(ans["drop_all"]) != [float(bf.round_exact(Fraction(1, K), bf.LABEL_DP))] * K:
                e.append("drop_all is not the prior")
        if ver:
            if lab(ver["full"]) != [1.0, 0.0] or any(lab(ver[m]) != [0.0, 1.0] for m in ver if m != "full"):
                e.append("two-hop verify labels")
    elif fam == "ret_distractors":
        for kind, d in (("answer", ans), ("verify", ver)):
            for m, r in d.items():
                if lab(r) != lab(d["base"]):
                    e.append(f"{kind} {m}: target differs from base")
    elif fam == "ret_evidence_strength":
        rep = ans["k1"]["meta"]["bf"]["factor"]["reported"]
        j = ans["k1"]["meta"]["bf"]["problem"]["options"].index(rep)
        p = [lab(ans[m])[j] for m in ("k1", "k2", "k3")]
        if not (p[0] < p[1] < p[2] and lab(ans["strong"])[j] > p[0]):
            e.append(f"agreeing passages do not concentrate the answer: {p}")
    return e
