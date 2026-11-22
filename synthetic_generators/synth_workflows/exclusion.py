"""synth_workflows.exclusion -- what the generator may never produce, and the audits that police it.

The zero-shot claim rests on this file. `LocalLLaMA/typed-decisions` has four workflows and twenty question
schemas (five per workflow). Nothing generated here may be one of them or a thin reskin of one. Names and
one-line descriptions below come from the public card (workflow table) and the TEST split's FORMAT (question
ids, types, option counts); descriptions are paraphrased, never copied. No benchmark state, question or
criterion text is stored in this repository: wording overlap is checked on a node against an eval jsonl
(`audit_states`, `audit_question_texts`), which build_jev_typed_synth.py runs under --exclude-states.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# --------------------------------------------------------------------------- the exclusion list
EXCLUDED_WORKFLOWS = {
    "agent_trace_observability": "whether an AI agent's run needs human review, and how urgently (card)",
    "customer_service": "the right response and action for a customer thread and its account (card)",
    "invoice_processing": "pay, hold or reject a vendor bill against its purchase order and delivery (card)",
    "security_incidents": "close, investigate or contain a security alert given the machine's history (card)",
}

# (workflow, qid, type, K, paraphrased description). K = option count; noul is always 2.
EXCLUDED_SCHEMAS = [
    ("agent_trace_observability", "action", "choice", 4, "what the monitoring system does with the run"),
    ("agent_trace_observability", "needs_review", "noul", 2, "the run needs a human to look at it"),
    ("agent_trace_observability", "outcome", "choice", 4, "how the agent's run ended"),
    ("agent_trace_observability", "risk", "score", 4, "how risky the agent's actions were"),
    ("agent_trace_observability", "urgency", "score", 4, "how soon the run needs attention"),
    ("customer_service", "action", "choice", 5, "the assistant's next step in the conversation"),
    ("customer_service", "category", "choice", 5, "what the customer conversation is mainly about"),
    ("customer_service", "churn_risk", "score", 4, "how likely the customer is to leave"),
    ("customer_service", "needs_human", "noul", 2, "a human agent must take the conversation over"),
    ("customer_service", "urgency", "score", 4, "how time-sensitive the conversation is"),
    ("invoice_processing", "discrepancy_severity", "score", 4, "how material the invoice/order/delivery gap is"),
    ("invoice_processing", "disposition", "choice", 4, "how the vendor bill is dispositioned"),
    ("invoice_processing", "duplicate", "noul", 2, "the bill repeats one already submitted"),
    ("invoice_processing", "matches_order", "noul", 2, "the bill reconciles with order and delivery"),
    ("invoice_processing", "urgency", "score", 4, "how time-sensitive processing the bill is"),
    ("security_incidents", "credential_compromise", "noul", 2, "an account or credential is compromised"),
    ("security_incidents", "disposition", "choice", 4, "how the security alert is dispositioned"),
    ("security_incidents", "severity", "score", 5, "potential impact if the alert is real"),
    ("security_incidents", "true_positive", "noul", 2, "the alert reflects real malicious activity"),
    ("security_incidents", "urgency", "score", 4, "how fast a responder must act"),
]

EXCLUDED_QIDS = sorted({q for _, q, _, _, _ in EXCLUDED_SCHEMAS})

# Near neighbours: families too close to the four to generate at all ...
FORBIDDEN_FAMILIES = [
    "customer support / service desk threads, chat support, complaint handling, churn or retention triage",
    "security alert / SIEM / SOC / EDR triage, login-anomaly or credential-compromise review, phishing triage",
    "accounts payable: vendor invoices, PO / goods-receipt three-way match, duplicate-bill detection, "
    "employee expense-claim approval",
    "AI agent / LLM run review, agent trace or tool-call monitoring, chatbot transcript QA",
]
# ... and families that are allowed only under a stated constraint (checked by review, not code).
CONSTRAINED_FAMILIES = {
    "fraud alerts": "payments/transactions, not machine security; no true-positive + close/monitor/"
                    "investigate/contain pairing",
    "sre incident severity": "service outages and degradations, not security alerts; no 'is it malicious' question",
    "procurement exceptions": "pre-purchase approvals (sole source, off-contract, budget); never reconcile a bill "
                              "against a PO or delivery, never pay/hold/reject a bill",
    "data-access requests": "approving access to datasets; no compromise or alert questions",
    "warranty / returns claims": "product-defect adjudication from the service side; no conversation-reply or "
                                 "churn questions",
    "loan servicing hardship": "a hardship application and account data; no support-thread reply questions",
}

# words a workflow NAME may not contain (a renamed copy of a benchmark workflow is still a copy)
FORBIDDEN_NAME_PARTS = ["customer_service", "customer_support", "invoice", "accounts_payable", "security_incident",
                        "security_alert", "siem", "soc_", "agent_trace", "agent_run", "llm_trace"]


def exclusion_errors(wf) -> list:
    """Code-checkable exclusion rules for one Workflow (the rest is review; see research/typed_synth_spec.md)."""
    e = []
    if wf.name in EXCLUDED_WORKFLOWS:
        e.append(f"{wf.name}: is a benchmark workflow")
    for part in FORBIDDEN_NAME_PARTS:
        if part in wf.name:
            e.append(f"{wf.name}: name contains excluded family marker {part!r}")
    bench_sets = defaultdict(set)
    for w, q, _, _, _ in EXCLUDED_SCHEMAS:
        bench_sets[w].add(q)
    for s in wf.schemas:
        try:
            qs = s.questions()
        except Exception:  # noqa: BLE001  (reported by the contract check)
            continue
        for qid in qs:
            if qid in EXCLUDED_QIDS:
                e.append(f"{wf.name}/{s.name}: qid {qid!r} is a benchmark question id; name it for your domain")
        for w, ids in bench_sets.items():
            if len(ids & set(qs)) >= 2:
                e.append(f"{wf.name}/{s.name}: shares {sorted(ids & set(qs))} with benchmark workflow {w}")
    return e


# --------------------------------------------------------------------------- n-gram audits
WORD = re.compile(r"\w+")


def words(s: str) -> list:
    return WORD.findall(str(s).lower())


def ngrams(ws: list, n: int) -> set:
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def flat_text(state: str) -> str:
    """A JSON state's string leaves joined (keys dropped), else the text itself. Shingles are counted on this, so
    two states that merely share a JSON key skeleton do not look like copies (build_jev_ext2's 'flattened' form)."""
    try:
        obj = json.loads(state)
    except (ValueError, TypeError):
        return state
    if not isinstance(obj, (dict, list)):
        return state
    out = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            out.append(x)
    walk(obj)
    return " ".join(out)


def read_eval(path) -> tuple:
    """(states, question_texts) from a jsonl file or a directory of *.jsonl (one level, plus split subdirs).

    Accepts our corpus format (questions = {qid: {instructions, criteria | levels}}) and the raw typed-decisions
    row format (questions as a JSON string, criteria as a list for score). Labels are never read.
    """
    p = Path(path)
    files = [p] if p.is_file() else sorted(list(p.glob("*.jsonl")) + list(p.glob("*/*.jsonl")))
    if not files:
        raise SystemExit(f"--exclude-states {path}: no jsonl files")
    states, qtexts = [], set()
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("state"):
                    states.append(str(r["state"]))
                qs = r.get("questions") or {}
                if isinstance(qs, str):
                    qs = json.loads(qs)
                for q in qs.values() if isinstance(qs, dict) else []:
                    if not isinstance(q, dict):
                        continue
                    qtexts.add(str(q.get("instructions", "")))
                    c = q.get("criteria")
                    vals = list(c.values()) if isinstance(c, dict) else list(c or [])
                    vals += list(q.get("levels") or [])
                    qtexts.update(str(v) for v in vals)
    qtexts.discard("")
    # dedupe states (a per-question corpus repeats each state once per question)
    return list(dict.fromkeys(states)), sorted(qtexts)


class StateIndex:
    """An n-gram index over eval states, for 'does this generated state copy an eval state?'."""

    def __init__(self, states, n: int = 13, shingle: int = 8):
        self.n, self.shingle = n, shingle
        self.big, self.small = defaultdict(set), defaultdict(set)
        for i, s in enumerate(states):
            for g in ngrams(words(s), n):
                self.big[g].add(i)
            for g in ngrams(words(flat_text(s)), shingle):
                self.small[g].add(i)
        self.n_states = len(states)

    def check(self, state: str) -> dict:
        big, small = ngrams(words(state), self.n), ngrams(words(flat_text(state)), self.shingle)
        hit_big, hit_small = Counter(), Counter()
        for g in big:
            for i in self.big.get(g, ()):
                hit_big[i] += 1
        for g in small:
            for i in self.small.get(g, ()):
                hit_small[i] += 1
        frac = (max(hit_big.values()) / len(big)) if (big and hit_big) else 0.0
        shared = max(hit_small.values()) if hit_small else 0
        return {"max_frac_ngram": round(frac, 4), "max_shared_shingles": shared,
                "any_ngram": bool(hit_big), "n_ngrams": len(big)}


def audit_states(records, eval_states, n: int = 13, frac_threshold: float = 0.5, shingle: int = 8,
                 shingle_min: int = 3) -> dict:
    """Flag generated records whose state copies an eval state.

    A record is FLAGGED when, against any ONE eval state, >= frac_threshold of its word n-grams (n=13 over the
    raw text, od1's audit) occur in it, or it shares >= shingle_min distinct 8-word shingles with it (over the
    flattened text: JSON string leaves, keys dropped; build_jev_ext2's rule).
    Returns {flagged: [ids], stats: {...}, worst: [...]}; the build drops flagged records.
    """
    idx = StateIndex(eval_states, n=n, shingle=shingle)
    flagged, worst, any_hit = [], [], 0
    per = Counter()
    for r in records:
        c = idx.check(r["state"])
        any_hit += c["any_ngram"]
        if c["max_frac_ngram"] >= frac_threshold or c["max_shared_shingles"] >= shingle_min:
            flagged.append(r["id"])
            per[r["meta"]["workflow"] + "/" + r["meta"]["schema"]] += 1
        worst.append((c["max_frac_ngram"], c["max_shared_shingles"], r["id"]))
    worst.sort(reverse=True)
    return {"n_eval_states": idx.n_states, "n_checked": len(records), "n_flagged": len(flagged),
            "n_with_any_13gram": any_hit, "flagged_by_schema": dict(per), "flagged": flagged,
            "rule": f">= {frac_threshold} of {n}-grams in one eval state, or >= {shingle_min} shared "
                    f"{shingle}-word shingles",
            "worst": [{"frac_13gram": a, "shared_8gram": b, "id": i} for a, b, i in worst[:10]]}


def audit_question_texts(workflows, eval_qtexts, n: int = 6) -> dict:
    """Flag any instruction / criterion / level text of ours that equals, or shares an n-word run (n=6) with,
    an eval question text. These are schema-level: a hit means REWORD the schema, so the build refuses."""
    ev_norm = {" ".join(words(t)) for t in eval_qtexts}
    ev_grams = set()
    for t in eval_qtexts:
        ev_grams |= ngrams(words(t), n)
    hits = []
    for wf in workflows:
        for s in wf.schemas:
            for qid, q in s.questions().items():
                texts = [q["instructions"]]
                c = q.get("criteria")
                texts += list(c.values()) if isinstance(c, dict) else []
                texts += list(q.get("levels") or [])
                for t in texts:
                    ws = words(t)
                    if " ".join(ws) in ev_norm or (ngrams(ws, n) & ev_grams):
                        hits.append({"workflow": wf.name, "schema": s.name, "qid": qid, "text": t})
    return {"n_eval_texts": len(eval_qtexts), "n_hits": len(hits), "hits": hits,
            "rule": f"exact normalised match or a shared {n}-word run"}
