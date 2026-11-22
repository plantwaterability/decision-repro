#!/usr/bin/env python3
"""long_knowledge_v1 (stock_decision_v1): verified long-context supply for the gaps the 2026-10-14 17:00 review named (2,049-8,192
token states, verified knowledge, contradiction / absence, retrieval grounding, large menus). Separate pool; NOT in any mixture
(candidate supply for M3 after the 31B bridge reads). Families (one jsonl per family = root):

  lk_records_<src>  multi-record comparisons over our public-domain record families (ClinicalTrials.gov registrations, NHTSA
                    defect investigations, NTSB aviation reports, OSHA severe-injury reports, federal appeals (FJC IDB), state supreme
                    court cases (SSCDP)). 4-8 FULL records per state, drawn only from Stage 0 pilot-reserve records whose ancestry
                    components no selection / panel uses. Questions over fields VISIBLE in the records: extreme value (which record
                    has the largest / smallest ...), count (how many records have ... = v), find-or-none (which record has ... = v;
                    'None of the records' when no record does). Gold computed from the visible fields (rule_exact).
  lk_cuad           CUAD v1 (CC BY 4.0) contract excerpts (whole contract when it fits, else <= 3 contiguous windows): clause-type
                    presence / absence (noul) and 'which of these clause types appears' (choice, exactly one present). Gold = the
                    CUAD expert span annotations (a type is present iff an annotated span lies inside the excerpt; absent iff no
                    annotated span of that type touches it; partial overlaps are never asked).
  lk_maud           MAUD v1 (CC BY 4.0) merger agreements: the deal-point excerpt that answers a MAUD question, hidden among other
                    excerpts of the SAME agreement; the MAUD multiple-choice question over that question's answer set. Gold = MAUD
                    expert answer (main rows of MAUD_train only).
  lk_musique        MuSiQue v1.0 full train (CC BY 4.0; paragraph text from Wikipedia: CC BY-SA, share-alike flagged): 20
                    paragraphs; answerability (noul, answerable vs unanswerable contrast) and, when answerable, which paragraph
                    supports the final reasoning step (choice, K = 20).
  lk_tydi           TyDi QA primary task train (Apache-2.0; Wikipedia article text: CC BY-SA, share-alike flagged), 11 languages:
                    the whole article (or a contiguous passage window containing the annotated passage) and 'which passage answers
                    the question, or none' (choice, K = passages + 1 <= 52). Null-annotated articles give the 'none' rows.

Every record: id, base_id (parent: contract / article / question pair / record group), split (train / val by parent hash, 10% val),
license, state, questions, labels (one-hot), label_kind hard, meta {generator, family, label_source per question (rule_exact |
expert), share_alike, source_ref, sources}. States are sized with the pinned Gemma tokenizer to <= 7,000 tokens.

  python3 build_long_knowledge.py --raw DIR(downloads.json) --rd3 POOLDIR/emo_pool_v1 --free-rk free_rk.txt --tokenizer tok/tokenizer.json
        --out pools/long_knowledge_v1 [--families a,b]
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re

import pandas as pd
from tokenizers import Tokenizer

VERSION = "build_long_knowledge v1 (2026-10-14, PILOT-DATA)"
SEED = 20261015
MAX_STATE_TOK = 7000
VAL_FRAC = 0.10
TOK = None


def ntok(s):
    return len(TOK.encode(s, add_special_tokens=False).ids)


def H(*x):
    return hashlib.sha256("|".join(map(str, x)).encode()).hexdigest()


def rng(*x):
    return random.Random(int(H(SEED, *x)[:16], 16))


def split_of(parent):
    return "val" if int(H("split", parent)[:8], 16) / 16 ** 8 < VAL_FRAC else "train"


def one_hot(keys, gold):
    assert gold in keys
    return {k: (1.0 if k == gold else 0.0) for k in keys}


def qc(instr, crit, kind="choice"):
    return {"type": kind, "instructions": instr, "criteria": crit}


YESNO = lambda yes, no: {"yes": yes, "no": no}


def rec_out(fam, rid, parent, state, qs, labels, lsrc, lic, share_alike, extra):
    return {"id": rid, "base_id": parent, "source": fam, "split": split_of(parent), "license": lic, "state": state, "questions": qs,
            "labels": labels, "label_kind": {q: "hard" for q in qs},
            "meta": {"generator": VERSION, "family": fam, "label_source": lsrc, "share_alike": share_alike, "group_id": parent, **extra}}


# =============================================================================================== 1. multi-record comparisons
SUB = [("ctgov", "jev_rd3__ctgov", lambda m: True, "ClinicalTrials.gov trial registrations (U.S. National Library of Medicine)"),
       ("nhtsa", "jev_rd3__safety", lambda m: "nhtsa" in str(m.get("dataset")), "NHTSA Office of Defects Investigation case files (U.S. federal work)"),
       ("ntsb", "jev_rd3__safety", lambda m: "ntsb" in str(m.get("dataset")), "NTSB aviation accident reports (U.S. federal work)"),
       ("osha_sir", "jev_rd3__safety", lambda m: "January2015" in str(m.get("dataset")) or "severe-injury" in str(m.get("dataset")),
        "OSHA severe injury reports (U.S. federal work)"),
       ("fjc", "jev_rd3__courts", lambda m: str(m.get("dataset", "")).startswith("fjc"), "federal appeals (FJC Integrated Database + CourtListener; public domain)"),
       ("sscdp", "jev_rd3__courts", lambda m: str(m.get("dataset", "")).startswith("sscdp"), "state supreme court cases (State Supreme Court Data Project, CC0)")]
NUM = re.compile(r"^-?\d+(\.\d+)?$")
YM = re.compile(r"^(19|20)\d\d-\d\d$")
LINE = re.compile(r"^\s*([A-Z][A-Za-z0-9 /()'-]{1,40}): (.{1,90})$")


def fields_json(state):
    try:
        o = json.loads(state)
    except Exception:
        return None
    if not isinstance(o, dict):
        return None
    out = {}

    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, path + [k])
        elif isinstance(x, bool):
            out[".".join(path)] = ("bool", "true" if x else "false")
        elif isinstance(x, (int, float)):
            out[".".join(path)] = ("num", x)
        elif isinstance(x, str) and 0 < len(x) <= 60 and "\n" not in x:
            out[".".join(path)] = ("num", float(x)) if NUM.match(x) else (("ym", x) if YM.match(x) else ("cat", x))
    walk(o, [])
    return out


def fields_text(state):
    out, seen = {}, collections.Counter()
    for line in state.splitlines():
        m = LINE.match(line)
        if m:
            seen[m.group(1)] += 1
            v = m.group(2).strip()
            out[m.group(1)] = ("num", float(v)) if NUM.match(v) else (("ym", v) if YM.match(v) else ("cat", v))
    return {k: v for k, v in out.items() if seen[k] == 1}


def fdesc(sub, path):
    if sub == "ctgov":
        return f"`{path}`"
    parts = path.split(".")
    return f"`{parts[-1]}`" + (f" (inside `{'.'.join(parts[:-1])}`)" if len(parts) > 1 else "")


def build_records(a, out):
    free = {x.strip() for x in open(a.free_rk) if x.strip()}
    stats = collections.Counter()
    for sub, stem, sel, desc in SUB:
        recs = []
        with open(os.path.join(a.rd3, "train", f"{stem}.jsonl"), encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                rk = f"emo_pool_v1|train/{stem}.jsonl|{line_no}"
                if rk not in free:
                    continue
                r = json.loads(line)
                if not sel(r.get("meta") or {}) or not isinstance(r.get("state"), str):
                    continue
                fl = fields_json(r["state"]) if r["state"].lstrip().startswith("{") else fields_text(r["state"])
                if not fl:
                    continue
                recs.append({"rk": rk, "state": r["state"], "fields": fl, "tok": None, "id": r.get("id")})
        R = rng("records", sub)
        R.shuffle(recs)
        vocab = collections.defaultdict(collections.Counter)
        for x in recs:
            for k, (t, v) in x["fields"].items():
                if t in ("cat", "bool"):
                    vocab[k][v] += 1
        fam = f"lk_records_{sub}"
        rows, i = [], 0
        while i < len(recs) and len(rows) < a.records_per_sub:
            want = 8 if sub in ("osha_sir", "fjc", "sscdp") else R.randint(4, 8)
            group, tok = [], 0
            while i < len(recs) and len(group) < want:
                x = recs[i]
                i += 1
                x["tok"] = x["tok"] or ntok(x["state"])
                if tok + x["tok"] + 40 > MAX_STATE_TOK:
                    continue
                group.append(x)
                tok += x["tok"] + 8
            if len(group) < 4:
                continue
            n = len(group)
            state = (f"The following {n} records come from {desc}. Every record is shown in full, as stored.\n\n"
                     + "\n\n".join(f"=== Record {j + 1} ===\n{x['state']}" for j, x in enumerate(group)))
            qs, labs, ls = {}, {}, {}
            common = set.intersection(*[set(x["fields"]) for x in group])
            rr = rng("q", sub, group[0]["rk"])
            recopts = {f"r{j + 1}": f"Record {j + 1}" for j in range(n)}
            # extreme
            ext = []
            for k in sorted(common):
                tv = [x["fields"][k] for x in group]
                if all(t == "num" for t, _ in tv) or all(t == "ym" for t, _ in tv):
                    vals = [v for _, v in tv]
                    for d in ("largest", "smallest"):
                        best = max(vals) if d == "largest" else min(vals)
                        if vals.count(best) == 1 and len(set(vals)) >= 3:
                            ext.append((k, d, vals.index(best), tv[0][0]))
            if ext:
                k, d, j, t = rr.choice(ext)
                word = {"largest": "latest" if t == "ym" else "largest", "smallest": "earliest" if t == "ym" else "smallest"}[d]
                qs["extreme"] = qc(f"Which record has the {word} value of the field {fdesc(sub, k)}?", recopts)
                labs["extreme"], ls["extreme"] = one_hot(list(recopts), f"r{j + 1}"), "rule_exact"
            # count
            cats = [k for k in sorted(common) if all(x["fields"][k][0] in ("cat", "bool") for x in group) and len(vocab[k]) >= 2]
            if cats:
                k = rr.choice(cats)
                present = [x["fields"][k][1] for x in group]
                v = rr.choice(present) if rr.random() < 0.85 else rr.choice([w for w in vocab[k] if w not in present] or present)
                c = sum(1 for p in present if p == v)
                opts = {f"n{c_}": str(c_) for c_ in range(n + 1)}
                qs["count"] = qc(f"How many of the {n} records have the field {fdesc(sub, k)} equal to \"{v}\"?", opts)
                labs["count"], ls["count"] = one_hot(list(opts), f"n{c}"), "rule_exact"
            # find or none
            finds = []
            for k in cats:
                present = [x["fields"][k][1] for x in group]
                uniq = [p for p in present if present.count(p) == 1]
                absent = [w for w, cnt in vocab[k].items() if w not in present and cnt >= 2]
                if uniq or absent:
                    finds.append((k, uniq, absent))
            if finds:
                k, uniq, absent = rr.choice(finds)
                if absent and (not uniq or rr.random() < 0.35):
                    v, gold = rr.choice(sorted(absent)), "none"
                else:
                    v = rr.choice(uniq)
                    gold = f"r{[x['fields'][k][1] for x in group].index(v) + 1}"
                opts = {**recopts, "none": "None of the records"}
                qs["find"] = qc(f"Which record has the field {fdesc(sub, k)} equal to \"{v}\"? If no record does, choose \"None of the records\".", opts)
                labs["find"], ls["find"] = one_hot(list(opts), gold), "rule_exact"
            if not qs:
                continue
            parent = f"{fam}:{H(*[x['rk'] for x in group])[:16]}"
            rows.append(rec_out(fam, f"{fam}_{len(rows):05d}", parent, state, qs, labs, ls,
                                "public domain / CC0 records (US federal works; SSCDP CC0; ClinicalTrials.gov NLM data) as in the source family",
                                False, {"source_records": [x["rk"] for x in group], "n_records": n, "sub_source": sub}))
            stats[fam] += 1
        out[fam] = rows
    return stats


# =============================================================================================== 2. CUAD
CUAD_SKIP = {"Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date", "Renewal Term",
             "Notice Period To Terminate Renewal", "Governing Law"}


def windows_by_tokens(text, max_tok, max_windows, R):
    paras = re.split(r"(\n+)", text)
    chunks, cur, start, pos = [], "", 0, 0
    spans = []
    for p in paras:
        spans.append((pos, pos + len(p)))
        pos += len(p)
    # greedy contiguous windows on paragraph boundaries
    out, s = [], 0
    while s < len(spans):
        e, tot = s, 0
        while e < len(spans):
            t = ntok(text[spans[e][0]:spans[e][1]]) if spans[e][1] > spans[e][0] else 0
            if tot + t > max_tok:
                break
            tot += t
            e += 1
        if e == s:
            s += 1
            continue
        out.append((spans[s][0], spans[e - 1][1], tot))
        s = e
    if len(out) > max_windows:
        out = sorted(R.sample(out, max_windows))
    return out


def build_cuad(a, out, dl):
    c = json.load(open(dl["theatticusproject/cuad/CUAD_v1/CUAD_v1.json"]["path"]))
    rows = []
    all_ctx = [x["paragraphs"][0]["context"] for x in c["data"]]
    for di, doc in enumerate(c["data"]):
        ctx = doc["paragraphs"][0]["context"]
        title = doc["title"]
        spans, defs = collections.defaultdict(list), {}
        for q in doc["paragraphs"][0]["qas"]:
            cat = q["id"].split("__")[-1]
            m = re.search(r"Details: (.*)$", q["question"])
            defs[cat] = (m.group(1).strip() if m else cat)
            for an in q["answers"]:
                spans[cat].append((an["answer_start"], an["answer_start"] + len(an["text"])))
        R = rng("cuad", title)
        for wi, (ws, we, wt) in enumerate(windows_by_tokens(ctx, MAX_STATE_TOK - 300, 3, R)):
            if wt < 400:
                continue
            pres, absn = [], []
            for cat in sorted(defs):
                if cat in CUAD_SKIP:
                    continue
                inside = [s for s in spans[cat] if s[0] >= ws and s[1] <= we]
                touch = [s for s in spans[cat] if s[0] < we and s[1] > ws]
                if inside:
                    pres.append(cat)
                elif not touch:
                    absn.append(cat)
            if not pres and not absn:
                continue
            body = ctx[ws:we].strip()
            if sum(1 for cx in all_ctx if body in cx) != 1:      # the shown text must identify one contract (CUAD has duplicates)
                continue
            whole = ws == 0 and we >= len(ctx.rstrip())
            state = (f"Contract: {title.split('-EX-')[-1].replace('-', ' ').title() if '-EX-' in title else 'commercial agreement'}"
                     f" ({'complete text' if whole else 'contiguous excerpt'}; source: CUAD v1, EDGAR filing exhibit)\n\n{ctx[ws:we].strip()}")
            qs, labs, ls = {}, {}, {}
            asked = R.sample(pres, min(2, len(pres))) + R.sample(absn, min(2, len(absn)))
            for cat in asked:
                qid = "has_" + re.sub(r"[^a-z0-9]+", "_", cat.lower()).strip("_")
                qs[qid] = qc(f"The {'contract' if whole else 'excerpt'} contains a clause of the type \"{cat}\" ({defs[cat].rstrip('.')}).",
                             YESNO(f"At least one passage of the text is a \"{cat}\" clause.", f"No passage of the text is a \"{cat}\" clause."), "noul")
                labs[qid], ls[qid] = one_hot(["yes", "no"], "yes" if cat in pres else "no"), "expert:cuad_annotation"
            if pres and len(absn) >= 4:
                tgt = R.choice(pres)
                opts_c = [tgt] + R.sample(absn, min(len(absn), R.randint(4, 7)))
                R.shuffle(opts_c)
                crit = {f"t{j}": f"{cat} -- {defs[cat].rstrip('.')}" for j, cat in enumerate(opts_c)}
                qs["which_type"] = qc("Exactly one of these clause types appears in the text. Which one?", crit)
                labs["which_type"], ls["which_type"] = one_hot(list(crit), f"t{opts_c.index(tgt)}"), "expert:cuad_annotation"
            parent = f"cuad:{title}"
            rows.append(rec_out("lk_cuad", f"lk_cuad_{di:03d}_{wi}", parent, state, qs, labs, ls,
                                "CC BY 4.0 (CUAD v1, The Atticus Project; contract text from SEC EDGAR filings)", False,
                                {"cuad_title": title, "window_chars": [ws, we], "whole_contract": whole, "window_tokens": wt}))
    out["lk_cuad"] = rows
    return {"lk_cuad": len(rows)}


# =============================================================================================== 3. MAUD
def build_maud(a, out, dl):
    tr = pd.read_csv(dl["theatticusproject/maud/MAUD_v1/MAUD_train.csv"]["path"])
    dv = pd.read_csv(dl["theatticusproject/maud/MAUD_v1/MAUD_dev.csv"]["path"])
    allr = pd.concat([tr, dv])
    allr = allr[allr.answer.notna() & (allr.answer.astype(str).str.strip() != "") & (allr.answer.astype(str) != "nan")]
    opts = allr.groupby(["question", "subquestion"]).answer.apply(lambda s: sorted(set(map(str, s)))).to_dict()
    main = tr[(tr.data_type == "main") & tr.answer.notna() & (tr.answer.astype(str) != "nan")].copy()
    main["text"] = main.text.astype(str)
    rows = []
    texts = main.drop_duplicates(["contract_name", "text"])
    by_contract = {k: list(g.text) for k, g in texts.groupby("contract_name")}
    tokc = {}
    main = main.drop_duplicates(["contract_name", "question", "subquestion"]).sample(frac=1.0, random_state=SEED % (2 ** 31))
    per_c = collections.Counter()
    for r in main.itertuples():
        o = opts.get((r.question, r.subquestion)) or []
        if len(o) < 2 or len(o) > 52 or per_c[r.contract_name] >= a.maud_per_contract:
            continue
        R = rng("maud", r.id, r.contract_name)
        t0 = tokc.setdefault(r.text, ntok(r.text))
        if t0 > MAX_STATE_TOK - 600:
            continue
        others = [t for t in by_contract[r.contract_name] if t != r.text]
        R.shuffle(others)
        chosen, tot = [r.text], t0
        for t in others:
            tt = tokc.setdefault(t, ntok(t))
            if tot + tt > a.maud_target_tok:
                continue
            chosen.append(t)
            tot += tt
            if tot >= a.maud_target_tok * 0.8:
                break
        R.shuffle(chosen)
        state = ("Excerpts from one merger agreement (MAUD v1; the excerpts are deal-point passages of the same agreement, shown in "
                 "random order):\n\n" + "\n\n".join(f"--- Excerpt {j + 1} ---\n{t.strip()}" for j, t in enumerate(chosen)))
        q = str(r.question).replace("-Answer", "").strip()
        sq = "" if str(r.subquestion) in ("<NONE>", "nan") else f" ({r.subquestion})"
        crit = {f"a{j}": ans for j, ans in enumerate(o)}
        gold = f"a{o.index(str(r.answer))}"
        qs = {"deal_point": qc(f"Merger agreement question -- {q}{sq}: which answer describes this agreement? Use the excerpt that "
                               f"governs this point.", crit)}
        parent = f"maud:{r.contract_name}"
        rows.append(rec_out("lk_maud", f"lk_maud_{r.id}", parent, state, qs, {"deal_point": one_hot(list(crit), gold)},
                            {"deal_point": "expert:maud_annotation"}, "CC BY 4.0 (MAUD v1, The Atticus Project; agreements from SEC EDGAR)",
                            False, {"maud_id": str(r.id), "maud_question": str(r.question), "maud_subquestion": str(r.subquestion),
                                    "maud_category": str(r.category), "n_excerpts": len(chosen), "state_tokens_est": tot}))
        per_c[r.contract_name] += 1
    out["lk_maud"] = rows
    return {"lk_maud": len(rows)}


# =============================================================================================== 4. MuSiQue
def build_musique(a, out, dl):
    rows = []
    items = []
    with open(dl["bdsaglam/musique/musique_full_v1.0_train.jsonl"]["path"], encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    bases = sorted({it["id"] for it in items})
    R = rng("musique")
    keep = set(R.sample(bases, min(a.musique_bases, len(bases))))
    for it in items:
        if it["id"] not in keep:
            continue
        paras = it["paragraphs"]
        state = ("Paragraphs (Wikipedia text):\n\n" + "\n\n".join(f"[Paragraph {j + 1}] {p['title']}\n{p['paragraph_text'].strip()}"
                                                                   for j, p in enumerate(paras)))
        if ntok(state) > MAX_STATE_TOK:
            continue
        qq = it["question"].strip()
        qs = {"answerable": qc(f"The paragraphs above contain all the information needed to answer the question: \"{qq}\"",
                               YESNO("Yes: chaining facts stated in the paragraphs answers the question.",
                                     "No: at least one needed fact is missing from the paragraphs."), "noul")}
        labs = {"answerable": one_hot(["yes", "no"], "yes" if it["answerable"] else "no")}
        ls = {"answerable": "expert:musique_construction"}
        if it["answerable"]:
            last = it["question_decomposition"][-1].get("paragraph_support_idx")
            if last is not None and 0 <= int(last) < len(paras) and str(paras[int(last)].get("is_supporting")) in ("True", "true", "1"):
                crit = {f"p{j + 1}": f"Paragraph {j + 1}: {p['title']}" for j, p in enumerate(paras)}
                if len(set(crit.values())) == len(crit):
                    qs["final_hop"] = qc(f"To answer \"{qq}\", one paragraph supplies the last fact in the chain (the answer itself). Which paragraph?", crit)
                    labs["final_hop"], ls["final_hop"] = one_hot(list(crit), f"p{int(last) + 1}"), "expert:musique_annotation"
        parent = f"musique:{it['id']}"
        rows.append(rec_out("lk_musique", f"lk_musique_{it['id']}_{'a' if it['answerable'] else 'u'}", parent, state, qs, labs, ls,
                            "CC BY 4.0 (MuSiQue v1.0, Trivedi et al. 2022; paragraph text from Wikipedia: CC BY-SA 3.0/4.0 + GFDL, share-alike)",
                            True, {"musique_id": it["id"], "answerable": bool(it["answerable"]), "hops": len(it["question_decomposition"])}))
    out["lk_musique"] = rows
    return {"lk_musique": len(rows)}


# =============================================================================================== 5. TyDi QA
def build_tydi(a, out, dl):
    rows = []
    paths = [v["path"] for k, v in dl.items() if "tydiqa/primary_task/train" in k]
    df = pd.concat([pd.read_parquet(p) for p in sorted(paths)], ignore_index=True)
    R0 = rng("tydi")
    order = list(range(len(df)))
    R0.shuffle(order)
    per_lang = collections.Counter()
    nulls = 0
    for i in order:
        if len(rows) >= a.tydi_n:
            break
        x = df.iloc[i]
        lang = x.language
        if per_lang[lang] >= a.tydi_n // 11 + 20:
            continue
        gi = int(x.annotations["passage_answer_candidate_index"][0])
        if gi < 0 and nulls >= int(a.tydi_n * 0.3):
            continue
        doc = x.document_plaintext.encode("utf-8")
        st, en = x.passage_answer_candidates["plaintext_start_byte"], x.passage_answer_candidates["plaintext_end_byte"]
        ps = [doc[int(s):int(e)].decode("utf-8", errors="replace").strip() for s, e in zip(st, en)]
        if len(ps) < 3 or (gi >= len(ps)):
            continue
        pt = [ntok(p) + 6 for p in ps]
        R = rng("tydi", x.question_text, x.document_title)
        budget = MAX_STATE_TOK - 200
        if sum(pt) <= budget and len(ps) <= 51:
            s, e = 0, len(ps)
        else:
            anchor = gi if gi >= 0 else R.randrange(len(ps))
            if pt[anchor] > budget:
                continue
            s, e, tot = anchor, anchor + 1, pt[anchor]
            while e - s < 51:
                sides = []
                if s > 0 and tot + pt[s - 1] <= budget:
                    sides.append("L")
                if e < len(ps) and tot + pt[e] <= budget:
                    sides.append("R")
                if not sides:
                    break
                if R.choice(sides) == "L":
                    s -= 1
                    tot += pt[s]
                else:
                    tot += pt[e]
                    e += 1
        win = ps[s:e]
        if len(win) < 3:
            continue
        whole = s == 0 and e == len(ps)
        state = (f"Wikipedia article \"{x.document_title}\" ({lang}; {'complete article' if whole else f'passages {s + 1}-{e} of {len(ps)}'}):\n\n"
                 + "\n\n".join(f"[Passage {j + 1}] {p}" for j, p in enumerate(win)))
        crit = {f"p{j + 1}": f"Passage {j + 1}" for j in range(len(win))}
        crit["none"] = "None of the passages answers the question"
        gold = "none" if gi < 0 else f"p{gi - s + 1}"
        qs = {"answer_passage": qc(f"Question: \"{x.question_text.strip()}\". Which passage answers this question? If none of the shown "
                                   f"passages does, choose \"None of the passages answers the question\".", crit)}
        parent = f"tydi:{H(x.document_url, x.question_text)[:16]}"
        rows.append(rec_out("lk_tydi", f"lk_tydi_{len(rows):05d}", parent, state, qs, {"answer_passage": one_hot(list(crit), gold)},
                            {"answer_passage": "expert:tydi_annotation"},
                            "Apache-2.0 (TyDi QA, Clark et al. 2020; article text from Wikipedia: CC BY-SA 3.0 + GFDL, share-alike)", True,
                            {"language": lang, "document_url": x.document_url, "whole_article": whole, "passages_total": len(ps),
                             "window": [s, e], "null_annotation": gi < 0, "group_id": f"tydi_doc:{H(x.document_url)[:16]}"}))
        per_lang[lang] += 1
        nulls += gi < 0
    out["lk_tydi"] = rows
    return {"lk_tydi": len(rows), "per_language": dict(per_lang), "null_rows": nulls}


def main():
    global TOK
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--rd3", required=True)
    ap.add_argument("--free-rk", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", default="records,cuad,maud,musique,tydi")
    ap.add_argument("--records-per-sub", type=int, default=900)
    ap.add_argument("--maud-per-contract", type=int, default=40)
    ap.add_argument("--maud-target-tok", type=int, default=5000)
    ap.add_argument("--musique-bases", type=int, default=2600)
    ap.add_argument("--tydi-n", type=int, default=4000)
    a = ap.parse_args()
    TOK = Tokenizer.from_file(a.tokenizer)
    dl = json.load(open(os.path.join(a.raw, "downloads.json")))
    out, summ = {}, {"version": VERSION, "seed": SEED, "downloads": {k: {"revision": v["revision"], "bytes": v["bytes"]} for k, v in dl.items()}}
    fams = set(a.families.split(","))
    for name, fn in (("records", build_records), ("cuad", build_cuad), ("maud", build_maud), ("musique", build_musique), ("tydi", build_tydi)):
        if name in fams:
            s = fn(a, out, dl) if name != "records" else fn(a, out)
            summ[name] = s
            print(name, json.dumps(s), flush=True)
    for sp in ("train", "val"):
        os.makedirs(os.path.join(a.out, sp), exist_ok=True)
    fam_s = {}
    for fam, rows in out.items():
        for sp in ("train", "val"):
            rs = [r for r in rows if r["split"] == sp]
            if not rs:
                continue
            p = os.path.join(a.out, sp, f"{fam}.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                for r in rs:
                    f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            fam_s[f"{sp}/{fam}"] = {"records": len(rs), "decisions": sum(len(r["questions"]) for r in rs),
                                    "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
    summ["files"] = fam_s
    json.dump(summ, open(os.path.join(a.out, "long_knowledge_manifest.json"), "w"), indent=1, sort_keys=True)
    print(json.dumps(fam_s, indent=1))


if __name__ == "__main__":
    main()
