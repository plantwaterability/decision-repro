#!/usr/bin/env python3
"""Content-addressed immutable evaluation indexes and grouped split plans.

Index schema includes exact/normalized state, question, canonical option text
multisets, joint identity, lineage, approximate lexical and template identities.
Near/template flags are review candidates, NOT proof of semantic contamination.
Labels are never stored in the evaluation index or overlap report.

Registry: {"files":[{"population":str,"path":str,"required":bool,
"expected_sha256":str?,"expected_md5":str?,"sealed":bool?,"scope":str}]}.
Read-only existing sealed confirmation populations must have sealed=true; they
are not indexed. Their omission is explicitly reported, never silently cleared.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import heapq
import json
import os
import re
import sqlite3
import unicodedata
import zlib
from pathlib import Path

SCHEMA = "garas-decontam-sqlite-1"
NORMALIZATION = "nfkc-casefold-whitespace-outerquotes-1; compatible-claude-norm-1"


def dump(x): return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
def sha(x): return hashlib.sha256(x.encode() if isinstance(x, str) else x).hexdigest()


def file_hash(path, kind="sha256"):
    h = hashlib.new(kind)
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def compatible_norm(text):
    s = re.sub(r"\s+", " ", str(text)).strip()
    if len(s) > 1 and s[0] == s[-1] == '"': s = s[1:-1]
    return s.strip().lower()


def norm(text):
    s = unicodedata.normalize("NFKC", str(text)).casefold()
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 1 and s[0] == s[-1] and s[0] in ('"', "'", "“", "‘"): s = s[1:-1].strip()
    return s


def skeleton(text):
    s = norm(text)
    s = re.sub(r"\b[0-9a-f]{7,}\b", "<id>", s)
    s = re.sub(r"\b[a-z]+-[a-z0-9-]*[0-9][a-z0-9-]*\b", "<entity>", s)
    s = re.sub(r"\b\d+(?:[./:]\d+)*\b", "<number>", s)
    return s


def shingles(text, n=5):
    words = re.findall(r"[\w<>]+", norm(text))
    return {" ".join(words[i:i+n]) for i in range(max(1, len(words)-n+1))}


def signatures(text):
    # KMV candidate proposal: retain 8 minimum 64-bit shingle hashes. A lexical
    # threshold is computed on actual sets after proposal; this is heuristic.
    return [f"{v:016x}" for v in heapq.nsmallest(8, {int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")
                                                 for s in shingles(text)})]


def options(q):
    typ = q["type"]
    if typ == "choice": return [(str(k), str(v) or str(k)) for k, v in q["criteria"].items()]
    if typ == "noul": return [("yes", "Yes"), ("no", "No")]
    if typ == "score": return [(str(k), str(k)) for k in q["levels"]]
    raise ValueError(f"unsupported question type {typ}")


def identity(rec, q):
    state = rec["state"] if isinstance(rec["state"], str) else dump(rec["state"])
    question = q.get("instructions", "")
    texts = [text for _, text in options(q)]
    # Ordered score levels have semantics; sorting would conflate reversed
    # scales. Choice menus use canonical multisets for variant grouping only,
    # without claiming that option order never changes interpretation.
    canonical = dump([q["type"], texts if q["type"] == "score" else sorted(texts)])
    normalized_texts = [norm(v) for v in texts]
    normalized = dump([q["type"], normalized_texts if q["type"] == "score" else sorted(normalized_texts)])
    return {"state_exact": sha(state), "state_normalized": sha(norm(state)), "state_compatible": sha(compatible_norm(state)),
            "question_exact": sha(str(question)), "question_normalized": sha(norm(question)),
            "options_exact": sha(canonical), "options_normalized": sha(normalized),
            "joint_exact": sha(dump([state, str(question), canonical])),
            "joint_normalized": sha(dump([norm(state), norm(question), normalized])),
            "question_options": sha(dump([norm(question), normalized])), "template": sha(skeleton(state)),
            "normstate": norm(state)}


def roots(rec):
    m = rec.get("meta") or {}
    candidates = [rec.get("base_id"), m.get("variant_of"), m.get("group_id"), (m.get("bf") or {}).get("group"),
                  (m.get("lineage") or {}).get("root")]
    candidates.extend((m.get("lineage") or {}).get("parents") or [])
    if m.get("source_case_id") is not None:
        candidates.append(f"sourcecase:{rec.get('source', 'unknown')}:{m['source_case_id']}")
    return sorted({str(v) for v in candidates if v is not None and str(v)})


def records(path):
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip(): raise ValueError(f"blank JSONL line {path}:{line_no}")
            r = json.loads(line)
            if not r.get("questions") or "state" not in r: raise ValueError(f"not JEV schema: {path}:{line_no}")
            yield line_no, r


def atomic_json(path, obj):
    p = Path(path)
    if p.exists():
        if json.loads(p.read_text()) != obj: raise ValueError(f"immutable output differs: {p}")
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".partial")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, p); p.chmod(0o444)


def registry_inputs(registry, allow_partial):
    data = json.loads(Path(registry).read_text())
    available, unavailable = [], []
    for row in data["files"]:
        entry = dict(row)
        if entry.get("sealed"):
            unavailable.append({**entry, "reason": "sealed_confirmation_not_indexed"}); continue
        if entry.get("kind") == "opaque_index":
            unavailable.append({**entry, "reason": "opaque_exclusion_index_requires_schema_adapter"}); continue
        p = Path(entry["path"])
        if not p.is_file():
            unavailable.append({**entry, "reason": "file_unavailable"}); continue
        entry["sha256"] = file_hash(p)
        if entry.get("expected_sha256") and entry["sha256"] != entry["expected_sha256"]:
            raise ValueError(f"registry SHA256 mismatch: {p}")
        if entry.get("expected_md5"):
            actual_md5 = file_hash(p, "md5")
            if actual_md5 != entry["expected_md5"]: raise ValueError(f"registry MD5 mismatch: {p}")
            entry["verified_md5"] = actual_md5
        available.append(entry)
    blocked = [x for x in unavailable if x.get("required", True) and not x.get("sealed")]
    if blocked and not allow_partial: raise ValueError(f"{len(blocked)} required populations unavailable; use --allow-partial for explicitly incomplete research checks")
    return available, unavailable, data


def index(args):
    available, unavailable, registry = registry_inputs(args.registry, args.allow_partial)
    spec = {"schema": SCHEMA, "normalization": NORMALIZATION, "code_sha256": file_hash(__file__),
            "input_files": [{"population": x["population"], "path": x["path"], "sha256": x["sha256"],
                             "scope": x.get("scope", "unspecified"), "expected_md5": x.get("expected_md5")} for x in available],
            "unavailable": unavailable, "registry_sha256": file_hash(args.registry)}
    key = sha(dump(spec)); root = Path(args.cache) / key; root.mkdir(parents=True, exist_ok=True)
    db = root / "index.sqlite"; meta = root / "manifest.json"
    if db.exists():
        if not meta.exists(): raise ValueError("cached index has no immutable manifest")
        previous = json.loads(meta.read_text())
        if previous["spec"] != spec or previous["sqlite_sha256"] != file_hash(db): raise ValueError("cache identity/integrity mismatch")
        print(dump({"index": str(db.resolve()), "cache_hit": True, "status": previous["status"], "units": previous["units"]})); return
    partial = root / "index.partial.sqlite"
    if partial.exists(): partial.unlink()
    con = sqlite3.connect(partial)
    con.executescript("""PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
    CREATE TABLE state(id INTEGER PRIMARY KEY,exact TEXT,normalized TEXT,compatible TEXT,template TEXT,text BLOB);
    CREATE UNIQUE INDEX unique_norm_state ON state(normalized);
    CREATE INDEX state_exact ON state(exact); CREATE INDEX state_compatible ON state(compatible); CREATE INDEX state_template ON state(template);
    CREATE TABLE band(signature TEXT,state_id INTEGER);
    CREATE INDEX state_bands ON band(signature);
    CREATE TABLE unit(id INTEGER PRIMARY KEY,state_id INTEGER,state_exact TEXT,population TEXT,path TEXT,line_no INTEGER,record_id TEXT,qid TEXT,
                      question_exact TEXT,question_normalized TEXT,options_exact TEXT,options_normalized TEXT,joint_exact TEXT,joint_normalized TEXT,question_options TEXT);
    CREATE INDEX units_by_state ON unit(state_id);
    CREATE INDEX units_state_exact ON unit(state_exact);
    CREATE INDEX units_question_normalized ON unit(question_normalized); CREATE INDEX units_options_normalized ON unit(options_normalized);
    CREATE INDEX units_exact ON unit(joint_exact); CREATE INDEX units_normalized ON unit(joint_normalized);
    CREATE INDEX units_question_options ON unit(question_options);
    CREATE TABLE lineage(root TEXT,population TEXT,record_id TEXT);
    CREATE INDEX lineage_roots ON lineage(root);
    """)
    counts, unitn, recn = collections.Counter(), 0, 0
    for entry in available:
        n = 0
        for line, rec in records(entry["path"]):
            recn += 1; n += 1
            for root_id in roots(rec): con.execute("INSERT INTO lineage VALUES(?,?,?)", (root_id, entry["population"], rec["id"]))
            for qid, q in rec["questions"].items():
                i = identity(rec, q)
                cur = con.execute("SELECT id FROM state WHERE normalized=?", (i["state_normalized"],)).fetchone()
                if cur: sid = cur[0]
                else:
                    cur = con.execute("INSERT INTO state(exact,normalized,compatible,template,text) VALUES(?,?,?,?,?)",
                                      (i["state_exact"],i["state_normalized"],i["state_compatible"],i["template"],zlib.compress(i["normstate"].encode())))
                    sid = cur.lastrowid
                    con.executemany("INSERT INTO band VALUES(?,?)", [(sig,sid) for sig in signatures(i["normstate"])])
                con.execute("INSERT INTO unit(state_id,state_exact,population,path,line_no,record_id,qid,question_exact,question_normalized,options_exact,options_normalized,joint_exact,joint_normalized,question_options) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (sid,i["state_exact"],entry["population"],entry["path"],line,str(rec["id"]),qid,
                             i["question_exact"],i["question_normalized"],i["options_exact"],i["options_normalized"],i["joint_exact"],i["joint_normalized"],i["question_options"]))
                counts[entry["population"]] += 1; unitn += 1
            if recn % 5000 == 0: con.commit()
        if entry.get("expected_records") is not None and n != entry["expected_records"]:
            raise ValueError(f"registry record count mismatch {entry['path']}: {n}")
    con.commit()
    states = con.execute("SELECT count(*) FROM state").fetchone()[0]
    con.close(); os.replace(partial, db); db.chmod(0o444)
    # Full means all registry files checked, not all world's evaluation data.
    complete = not [x for x in unavailable if x.get("required", True) and not x.get("sealed")]
    full_scope = registry.get("coverage_complete", False)
    status = "registry_complete" if complete and full_scope else "partial_population_coverage"
    atomic_json(meta, {"spec": spec, "status": status, "records": recn, "units": unitn, "unique_states": states,
                       "units_per_population": dict(counts), "sqlite_sha256": file_hash(db),
                       "limits": "Exact/hash checks do not establish semantic non-overlap. Approximate proposal can miss near duplicates; template flags need adjudication."})
    print(dump({"index": str(db.resolve()), "cache_hit": False, "status": status, "units": unitn}))


def scan(args):
    db = Path(args.index).resolve(); manifest = json.loads(db.with_name("manifest.json").read_text())
    if file_hash(db) != manifest["sqlite_sha256"]: raise ValueError("index file integrity mismatch")
    spec=manifest["spec"]
    if spec.get("schema")!=SCHEMA or spec.get("normalization")!=NORMALIZATION or spec.get("code_sha256")!=file_hash(__file__):
        raise ValueError("index algorithm/schema/code identity differs; rebuild once under the current immutable input hashes")
    con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    files = []
    for spec in args.input:
        p = Path(spec)
        files.extend(sorted(p.glob("*.jsonl")) if p.is_dir() else [p])
    if not files: raise ValueError("no candidate inputs")
    identity_counts, flags, uncertain, scanned = collections.Counter(), [], 0, 0
    for path in files:
        for line, rec in records(path):
            if (rec.get("meta") or {}).get("eval_only") and not args.include_eval_only: continue
            for qid, q in rec["questions"].items():
                scanned += 1; i = identity(rec, q); events = []
                for kind, table, field, value in (("joint_exact","unit","joint_exact",i["joint_exact"]),
                                                  ("joint_normalized","unit","joint_normalized",i["joint_normalized"]),
                                                  ("state_exact","unit","state_exact",i["state_exact"]),
                                                  ("state_normalized","state","normalized",i["state_normalized"]),
                                                  ("state_compatible","state","compatible",i["state_compatible"]),
                                                  ("question_options_weak","unit","question_options",i["question_options"]),
                                                  ("question_normalized_weak","unit","question_normalized",i["question_normalized"]),
                                                  ("options_normalized_weak","unit","options_normalized",i["options_normalized"]),
                                                  ("template_heuristic","state","template",i["template"])):
                    query = f"SELECT id FROM {table} WHERE {field}=? LIMIT 2"
                    found = con.execute(query, (value,)).fetchall()
                    if found:
                        identity_counts[kind] += 1
                        interpretation = "literal_state_identity" if kind == "state_exact" else "exact_canonical_joint_identity_order_review_required" if kind == "joint_exact" else "normalized_identity_semantic_review_required"
                        if kind.endswith("_weak") or kind == "template_heuristic": interpretation="weak_identity_review_required"
                        events.append({"kind": kind, "index_ids": [r["id"] for r in found], "interpretation": interpretation})
                for root_id in roots(rec):
                    found = con.execute("SELECT population,record_id FROM lineage WHERE root=? LIMIT 2", (root_id,)).fetchall()
                    if found:
                        identity_counts["lineage"] += 1; events.append({"kind":"lineage","roots":[root_id]})
                sig = signatures(i["normstate"])
                placeholders = ",".join("?" for _ in sig)
                candidate_ids = con.execute(f"SELECT DISTINCT state_id FROM band WHERE signature IN ({placeholders}) LIMIT ?", (*sig,args.near_candidates+1)).fetchall()
                if len(candidate_ids) > args.near_candidates:
                    uncertain += 1; events.append({"kind":"near_candidate_cap","interpretation":"uncertain_semantic_overlap"})
                candidate_set = shingles(i["normstate"])
                for row in candidate_ids[:args.near_candidates]:
                    st = con.execute("SELECT normalized,text FROM state WHERE id=?", (row["state_id"],)).fetchone()
                    if st["normalized"] == i["state_normalized"]: continue
                    other = shingles(zlib.decompress(st["text"]).decode())
                    sim = len(candidate_set & other) / max(1,len(candidate_set | other))
                    if sim >= args.near_threshold:
                        identity_counts["near_lexical"] += 1
                        events.append({"kind":"near_lexical","index_state_id":row["state_id"],"jaccard5":round(sim,6),"interpretation":"uncertain_semantic_overlap"})
                if events:
                    flags.append({"candidate_path":str(path),"line":line,"record_id":rec["id"],"qid":qid,
                                  "base_id":rec.get("base_id"),"events":events})
    con.close()
    strong = any(k in identity_counts for k in ("joint_exact","joint_normalized","state_exact","state_normalized","state_compatible","lineage"))
    report = {"schema":SCHEMA,"index_sha256":manifest["sqlite_sha256"],"index_status":manifest["status"],
              "input_hashes":{str(p):file_hash(p) for p in files},"scanned_units":scanned,
              "identity_counts":dict(identity_counts),"flags":flags,"near_candidate_cap_units":uncertain,
              "training_release":"withheld" if strong or manifest["status"] != "registry_complete" else "requires_semantic_adjudication_and_source_checks",
              "interpretation":"No matching exact content within indexed populations is not proof of no contamination. Casefold/NFKC can change code/math meaning (e.g. t_unit vs T_unit); normalized identities need semantic review. Canonical choice menus do not assert option-order semantic equivalence. Weak/template/near hits need adjudication; unindexed populations remain unknown.",
              "parameters":{"near_threshold":args.near_threshold,"near_candidates":args.near_candidates}}
    atomic_json(args.out, report)
    print(dump({"scanned_units":scanned,"identity_counts":dict(identity_counts),"index_status":manifest["status"],"training_release":report["training_release"]}))


class UnionFind:
    def __init__(self): self.parent=[]
    def add(self): self.parent.append(len(self.parent)); return len(self.parent)-1
    def find(self,i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]; i = self.parent[i]
        return i
    def union(self,a,b): self.parent[self.find(b)] = self.find(a)


def splitplan(args):
    uf, rows, seen, ids = UnionFind(), [], {}, {}
    files = [Path(p) for p in args.input]
    for path in files:
        for line, rec in records(path):
            i = uf.add(); rows.append({"path":str(path),"line":line,"record_id":rec["id"],
                                      "existing_split":rec.get("split") or getattr(args,"unassigned_split",None),
                                      "lineage":roots(rec),"units":len(rec["questions"]),"source":rec.get("source"),
                                      "stable_base":rec.get("base_id") or (rec.get("meta",{}).get("lineage") or {}).get("root")
                                                    or (rec.get("meta",{}).get("bf") or {}).get("group") or rec.get("meta",{}).get("group_id"),
                                      "parent":rec.get("meta",{}).get("variant_of"),"state_hash":sha(norm(rec["state"]))})
            tokens = ["state:"+sha(norm(rec["state"]))] + ["lineage:"+r for r in roots(rec)]
            tokens += ["joint:"+identity(rec,q)["joint_normalized"] for q in rec["questions"].values()]
            tokens += ["record:"+str(rec["id"])]
            for token in tokens:
                if token in seen: uf.union(i,seen[token])
                else: seen[token]=i
    groups = collections.defaultdict(list)
    for i,row in enumerate(rows): groups[uf.find(i)].append(row)
    output, conflicts, counts = [], [], collections.Counter()
    for members in groups.values():
        stable = sorted({str(x["stable_base"]) for x in members if x["stable_base"]})
        if stable:
            assignment_key = "lineage:"+stable[0]
        else:
            by_id={str(x["record_id"]):x for x in members}
            original=[]
            for row in members:
                cur=row; visited=set()
                while cur.get("parent") in by_id and cur["parent"] not in visited:
                    visited.add(cur["parent"]);cur=by_id[cur["parent"]]
                original.append(cur["state_hash"])
            assignment_key="content:"+min(original)
        group_key = sha(assignment_key)
        existing = {x["existing_split"] for x in members if x["existing_split"]}
        if len(existing)>1: conflicts.append({"group":group_key,"existing_splits":sorted(existing),"members":len(members)})
        # Stable split of connected lineage/content group. Absolute source paths
        # must stay fixed in registry; changing inputs means a new versioned plan.
        fraction = int(sha(args.seed+group_key)[:16],16)/2**64
        split = "train" if fraction < .8 else "val" if fraction < .9 else "confirmation"
        if args.preserve_existing and len(existing)==1: split = next(iter(existing))
        counts[split] += len(members)
        output.append({"group":group_key,"assignment_key":assignment_key,"split":split,"members":members})
    atomic_json(args.out,{"schema":SCHEMA,"input_hashes":{str(p):file_hash(p) for p in files},"groups":output,
                          "cross_split_conflicts":conflicts,"record_counts":dict(counts),
                          "unassigned_split_policy":getattr(args,"unassigned_split",None),
                          "policy":"lineage + normalized state + canonical joint identities connected before splitting. Assignment uses stable lineage root, falling back to normalized root-state content; path/order/line numbers are provenance only. Generic question/options are flags, not edges. Missing lineage that merges formerly independent worlds requires reviewing the immutable previous plan; whole domain/generator holdouts require explicit policy."})
    print(dump({"groups":len(output),"records":len(rows),"cross_split_conflicts":len(conflicts),"record_counts":dict(counts)}))
    if conflicts: raise ValueError("related source states/lineage already cross splits; repair using the written plan before release")


def adapt_registry(args):
    """Translate audit exclusion metadata without reading benchmark item text."""
    source=json.loads(Path(args.known).read_text())
    candidate_by_md5={}
    for name in args.candidate:
        path=Path(name).resolve()
        candidate_by_md5.setdefault(file_hash(path,"md5"),str(path))
    rows=[]
    for population,data in source["populations"].items():
        hashes=data.get("md5",{})
        opaque=isinstance(hashes,str)
        if opaque: hashes={str(data.get("spec", "")).removeprefix("index:"):hashes}
        for original,md5 in hashes.items():
            available=candidate_by_md5.get(md5)
            if available is None and args.mount_root and "/pool/src/" in original:
                available=str(Path(args.mount_root)/original.split("/pool/src/",1)[1])
            rows.append({"population":population,"path":available or original,"original_path":original,
                         "expected_md5":md5,"required":True,"sealed":"confirm" in population.lower(),
                         "kind":"opaque_index" if opaque else "jev_jsonl",
                         "scope":"exact byte identity against 2026-10-10 exclusion source manifest"})
    # The registry describes a historical inventory. New v0.3/private board
    # populations were not proven present, so never call this exhaustive.
    atomic_json(args.out,{"schema":SCHEMA,"coverage_complete":False,"files":rows,
                          "source_registry_sha256":file_hash(args.known),
                          "notes":["Historical 47-population exclusion inventory. Availability and matching hashes must be checked.",
                                   "Current Decision Index v0.3 public/private coverage unresolved; original sealed confirmation inputs excluded."]})
    print(dump({"registry":str(Path(args.out).resolve()),"files":len(rows),"local_hash_matches":sum(x["path"] in candidate_by_md5.values() for x in rows),"coverage_complete":False}))


def main():
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest="command",required=True)
    x=sub.add_parser("index");x.add_argument("--registry",required=True);x.add_argument("--cache",required=True);x.add_argument("--allow-partial",action="store_true")
    x=sub.add_parser("scan");x.add_argument("--index",required=True);x.add_argument("--input",nargs="+",required=True);x.add_argument("--out",required=True)
    x.add_argument("--near-threshold",type=float,default=.82);x.add_argument("--near-candidates",type=int,default=200);x.add_argument("--include-eval-only",action="store_true")
    x=sub.add_parser("splitplan");x.add_argument("--input",nargs="+",required=True);x.add_argument("--out",required=True)
    x.add_argument("--seed",default="garas-curation-split-v1");x.add_argument("--preserve-existing",action="store_true")
    x.add_argument("--unassigned-split",choices=("train","val","development","confirmation"),help="force previously inspected unassigned candidate samples to development/val; never relabel exposed examples as fresh confirmation")
    x=sub.add_parser("adapt-registry");x.add_argument("--known",required=True);x.add_argument("--candidate",nargs="*",default=[])
    x.add_argument("--mount-root");x.add_argument("--out",required=True)
    args=ap.parse_args();globals()[args.command.replace("-","_")](args)


if __name__ == "__main__": main()
