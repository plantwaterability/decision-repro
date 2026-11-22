#!/usr/bin/env python3
"""Independent label verification for long_knowledge_v1. Never reads the builder's meta / offsets for the answer; every gold is
re-derived from the VISIBLE state + question text, and for annotated sources from the RAW source file matched by CONTENT:
  lk_records_*  re-parse every '=== Record n ===' block of the state (JSON blocks: json.loads + path walk; text blocks: line regex),
                parse the question text (field, value, direction) and recompute extreme / count / find-or-none.
  lk_cuad       locate the shown text inside the raw CUAD contract by string search (not the stored window), recompute every clause
                type's presence from the raw expert spans (inside = present, untouched = absent, partial = must not be asked).
  lk_maud       find the raw MAUD main row(s) whose excerpt text is shown verbatim in the state and whose question / subquestion match
                the question text; the gold must be that row's expert answer (and the answer set the question's raw answer set).
  lk_musique    find the raw item whose question and paragraph titles match; recompute answerability and the final hop's
                supporting paragraph (by matching paragraph text, not index).
  lk_tydi       find the raw example by (question, title); rebuild passages from the byte offsets; the gold passage must be the
                shown passage whose text equals the annotated passage (or 'none' for a null annotation).
Reports agree / checked per family and per question kind; exit 1 on any disagreement.

  python3 verify_long_knowledge.py --pool pools/long_knowledge_v1 --raw RAWDIR [--splits train,val]"""
import argparse
import collections
import json
import os
import re

import pandas as pd

LINE = re.compile(r"^\s*([A-Z][A-Za-z0-9 /()'-]{1,40}): (.{1,90})$")


def argmax(d):
    m = max(d.values())
    return [k for k, v in d.items() if v == m]


def blocks(state):
    parts = re.split(r"(?m)^=== Record (\d+) ===\n", state)
    return [parts[i + 1].rstrip("\n") for i in range(1, len(parts), 2)]


def get_field(block, path, is_text):
    if not is_text:
        o = json.loads(block)
        for k in path.split("."):
            if not isinstance(o, dict) or k not in o:
                return None
            o = o[k]
        if isinstance(o, bool):
            return "true" if o else "false"
        return o
    hits = [m.group(2).strip() for m in (LINE.match(x) for x in block.splitlines()) if m and m.group(1) == path]
    return hits[0] if len(hits) == 1 else None


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def path_of(q):
    m = re.search(r"the field `([^`]+)`(?: \(inside `([^`]+)`\))?", q)
    return (m.group(2) + "." + m.group(1)) if m and m.group(2) else (m.group(1) if m else None)


def records(r):
    st = r["state"]
    bl = blocks(st)
    is_text = not bl[0].lstrip().startswith("{")
    res = {}
    for qid, q in r["questions"].items():
        ins = q["instructions"]
        path = path_of(ins)
        vals = [get_field(b, path, is_text) for b in bl]
        if qid == "extreme":
            big = re.search(r"has the (largest|latest|smallest|earliest) value", ins).group(1) in ("largest", "latest")
            vs = [num(v) for v in vals]
            best = max(vs) if big else min(vs)
            res[qid] = [f"r{i + 1}" for i, v in enumerate(vs) if v == best]
        elif qid == "count":
            v = re.search(r'equal to "(.*)"\?$', ins).group(1)
            res[qid] = [f"n{sum(1 for x in vals if str(x) == v or (isinstance(x, (int, float)) and num(v) == x))}"]
        elif qid == "find":
            v = re.search(r'equal to "(.*)"\? If no record', ins).group(1)
            idx = [i for i, x in enumerate(vals) if str(x) == v]
            res[qid] = [f"r{idx[0] + 1}"] if len(idx) == 1 else (["none"] if not idx else ["AMBIGUOUS"])
    return res


class Raw:
    def __init__(self, raw):
        dl = json.load(open(os.path.join(raw, "downloads.json")))
        self.dl = {k: v["path"] for k, v in dl.items()}
        self._cuad = self._maud = self._mus = self._tydi = None

    def cuad(self):
        if self._cuad is None:
            c = json.load(open(self.dl["theatticusproject/cuad/CUAD_v1/CUAD_v1.json"]))
            self._cuad = {}
            for doc in c["data"]:
                ctx = doc["paragraphs"][0]["context"]
                sp = collections.defaultdict(list)
                for q in doc["paragraphs"][0]["qas"]:
                    for an in q["answers"]:
                        sp[q["id"].split("__")[-1]].append((an["answer_start"], an["answer_start"] + len(an["text"])))
                self._cuad[doc["title"]] = (ctx, sp)
        return self._cuad

    def maud(self):
        if self._maud is None:
            tr = pd.read_csv(self.dl["theatticusproject/maud/MAUD_v1/MAUD_train.csv"])
            self._maud = tr[tr.data_type == "main"].assign(text=lambda x: x.text.astype(str))
        return self._maud

    def musique(self):
        if self._mus is None:
            self._mus = collections.defaultdict(list)
            for line in open(self.dl["bdsaglam/musique/musique_full_v1.0_train.jsonl"], encoding="utf-8"):
                it = json.loads(line)
                self._mus[it["question"].strip()].append(it)
        return self._mus

    def tydi(self):
        if self._tydi is None:
            ps = [v for k, v in self.dl.items() if "tydiqa/primary_task/train" in k]
            df = pd.concat([pd.read_parquet(p, columns=["question_text", "document_title", "document_plaintext", "passage_answer_candidates", "annotations"]) for p in sorted(ps)])
            self._tydi = collections.defaultdict(list)
            for x in df.itertuples():
                self._tydi[(x.question_text.strip(), x.document_title)].append(x)
        return self._tydi


def cuad(r, raw):
    body = r["state"].split("\n\n", 1)[1]
    hits = [(t, ctx.find(body), sp) for t, (ctx, sp) in raw.cuad().items() if body in ctx]
    if len(hits) != 1:
        return {q: ["SOURCE_NOT_FOUND"] for q in r["questions"]}
    title, ws, sp = hits[0]
    we = ws + len(body)
    res = {}
    for qid, q in r["questions"].items():
        if q["type"] == "noul":
            cat = re.search(r'clause of the type "(.*?)" \(', q["instructions"]).group(1)
            inside = [s for s in sp[cat] if s[0] >= ws and s[1] <= we]
            touch = [s for s in sp[cat] if s[0] < we and s[1] > ws]
            res[qid] = ["yes"] if inside else (["no"] if not touch else ["PARTIAL"])
        else:
            pres = []
            for k, v in q["criteria"].items():
                cat = v.split(" -- ")[0]
                inside = [s for s in sp[cat] if s[0] >= ws and s[1] <= we]
                touch = [s for s in sp[cat] if s[0] < we and s[1] > ws]
                if inside:
                    pres.append(k)
                elif touch:
                    pres.append("PARTIAL:" + k)
            res[qid] = pres
    return res


def maud(r, raw):
    m = raw.maud()
    exc = re.split(r"(?m)^--- Excerpt \d+ ---\n", r["state"])[1:]
    exc = {e.strip() for e in exc}
    ins = r["questions"]["deal_point"]["instructions"]
    qtxt = re.match(r"Merger agreement question -- (.*?): which answer", ins).group(1)
    cand = m[m.text.str.strip().isin(exc)]
    rows = [x for x in cand.itertuples() if (str(x.question).replace("-Answer", "").strip()
                                             + ("" if str(x.subquestion) in ("<NONE>", "nan") else f" ({x.subquestion})")) == qtxt]
    answers = {str(x.answer) for x in rows}
    crit = r["questions"]["deal_point"]["criteria"]
    return {"deal_point": [k for k, v in crit.items() if v in answers] if len(answers) == 1 else ["AMBIGUOUS_OR_MISSING"]}


def musique(r, raw):
    ins = r["questions"]["answerable"]["instructions"]
    qq = re.search(r'answer the question: "(.*)"$', ins, re.S).group(1)
    body = r["state"].split("\n\n", 1)[1]
    parts = re.split(r"(?m)^\[Paragraph \d+\] ", body)[1:]
    shown = []
    for part in parts:
        t, _, x = part.partition("\n")
        shown.append((t, x.strip()))
    match = [it for it in raw.musique().get(qq, []) if [(p["title"], p["paragraph_text"].strip()) for p in it["paragraphs"]] == shown]
    if len(match) != 1:
        return {q: ["SOURCE_NOT_FOUND"] for q in r["questions"]}
    it = match[0]
    res = {"answerable": ["yes" if it["answerable"] else "no"]}
    if "final_hop" in r["questions"]:
        idx = it["question_decomposition"][-1]["paragraph_support_idx"]
        tgt = (it["paragraphs"][idx]["title"], it["paragraphs"][idx]["paragraph_text"].strip())
        res["final_hop"] = [f"p{i + 1}" for i, s in enumerate(shown) if s == tgt]
    return res


def tydi(r, raw):
    ins = r["questions"]["answer_passage"]["instructions"]
    qq = re.match(r'Question: "(.*)"\. Which passage answers', ins, re.S).group(1)
    title = re.match(r'Wikipedia article "(.*?)" \(', r["state"]).group(1)
    shown = [x.strip() for x in re.split(r"(?m)^\[Passage \d+\] ", r["state"].split("\n\n", 1)[1])[1:]]
    out = set()
    for x in raw.tydi().get((qq, title), []):
        doc = x.document_plaintext.encode("utf-8")
        st, en = x.passage_answer_candidates["plaintext_start_byte"], x.passage_answer_candidates["plaintext_end_byte"]
        ps = [doc[int(s):int(e)].decode("utf-8", errors="replace").strip() for s, e in zip(st, en)]
        if not all(p in ps for p in shown):
            continue
        gi = int(x.annotations["passage_answer_candidate_index"][0])
        if gi < 0:
            out.add("none")
        else:
            hit = [i for i, p in enumerate(shown) if p == ps[gi]]
            out.add(f"p{hit[0] + 1}" if len(hit) == 1 else "GOLD_NOT_SHOWN")
    return {"answer_passage": sorted(out) if out else ["SOURCE_NOT_FOUND"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = Raw(a.raw)
    res = collections.defaultdict(collections.Counter)
    bad = []
    for sp in a.splits.split(","):
        for fn in sorted(os.listdir(os.path.join(a.pool, sp))):
            if not fn.endswith(".jsonl"):
                continue
            fam = fn[:-6]
            f = records if fam.startswith("lk_records_") else {"lk_cuad": cuad, "lk_maud": maud, "lk_musique": musique, "lk_tydi": tydi}[fam]
            for line in open(os.path.join(a.pool, sp, fn), encoding="utf-8"):
                r = json.loads(line)
                der = f(r) if f is records else f(r, raw)
                for qid in r["questions"]:
                    g = argmax(r["labels"][qid])
                    d = der.get(qid)
                    ok = d == g and len(d) == 1
                    kind = qid if not qid.startswith("has_") else "has_<clause>"
                    res[f"{fam}/{kind}"]["checked"] += 1
                    res[f"{fam}/{kind}"]["agree"] += ok
                    if not ok and len(bad) < 30:
                        bad.append({"id": r["id"], "q": qid, "gold": g, "derived": d})
    rep = {"results": {k: dict(v) for k, v in sorted(res.items())}, "disagreements": bad,
           "all_agree": all(v["agree"] == v["checked"] for v in res.values())}
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps(rep)[:3000])
    raise SystemExit(0 if rep["all_agree"] else 1)


if __name__ == "__main__":
    main()
