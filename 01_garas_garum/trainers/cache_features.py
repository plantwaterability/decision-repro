#!/usr/bin/env python3
"""Stage 0a: cache backbone readout features for every (record, question) decision unit.

For each unit we run ONE forward pass over
    <state> ... </state> <question> ... </question> <options> <option>..</option> * K </options> <decision>
and save:
    phis        [K, 5120]  hidden at each `</option>`   -> the VBLL features
    h_decision  [5120]     hidden at `<decision>`        -> the state-level / aleatoric head
    target      [K]        the label distribution over options

Nothing is trained here. This is a pure inference pass whose only job is to make every later
experiment cheap: the head is 6.1M parameters on top of these features, so once the cache
exists, head training costs seconds and needs no GPU.

WHY THIS IS SAFE TO CACHE (and where the subtlety is)
    The head is permutation-equivariant in phi, but the backbone is NOT order-invariant --
    measured worst cosine 0.41 for the same option under a different permutation
    (trainers/check_order_invariance.py). So a cached phi is valid ONLY for the exact option
    order it was computed in. That order is recorded per unit (option_keys) precisely so a
    downstream consumer can never silently reuse it for a different ordering. This is also
    why the loader's reshuffle variants are useful rather than redundant.

Layout, one shard file per `--shard-size` units (flat, not padded -- K varies 2..255):

    shard_000123.pt = {
        phis:       Float16/BFloat16Tensor [T, 5120]
        h_decision: Float16/BFloat16Tensor [U, 5120]
        offsets:    int32  [U+1]            index into phis
        targets:    float32[T]
        source:     list[str]  len U
        qid:        list[str]  len U
        label_kind: list[str]  len U
        option_keys:list[list[str]] len U   the order phis rows are in
        split:      str
    }

  python3 cache_features.py --data jev_data --out jev_feats --limit 200   # smoke test
  nohup python3 cache_features.py --data jev_data --out jev_feats > /tmp/cache.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

OPTION_OPEN, OPTION_CLOSE, DECISION = "<option>", "</option>", "<decision>"

PROMPT = """<state>
{state}
</state>
<question>
{instructions}
</question>
<options>
{options}
</options>
{DECISION}"""


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Non-commercial / research-only licence strings. The trainer drops units that match (claude_lmtune.py --allow-nc overrides);
# NC_SOURCES catches known NC families even if a record lost its licence string on the way through a derived corpus.
NC_LICENCE = re.compile(r"\bNC\b|non[- ]?commercial|research[- ]only|research use only|academic use only|cc[- ]by[- ]nc|\bby-nc\b", re.I)
NC_SOURCES = {"ms_marco"}


def is_noncommercial(unit):
    """True if a unit comes from non-commercial / research-only data (by licence string or known source family)."""
    fam = str(unit.get("source") or "").split("/")[0].split("__")[0]
    return bool(NC_LICENCE.search(unit.get("license") or "")) or fam in NC_SOURCES


# --------------------------------------------------------------------------- units
def question_options(q, labels):
    """(option_keys, rendered_lines) in the record's stored order.

    Order matters twice over: the target vector is aligned to it, and the backbone's readout
    is order-dependent, so the saved `option_keys` is the contract for the cached rows.
    """
    if q["type"] == "choice":
        keys = list(q["criteria"])
        # An EMPTY description must fall back to the option key. jev-bench's go_emotions config
        # ships `criteria` as {emotion: ''} -- the label IS the key -- and an empty render is not
        # merely uninformative: "<option></option>" BPE-merges to [13886,1419,1979,29], swallowing
        # the '</' token, so the marker scan finds ZERO options and every unit is dropped with
        # `marker_mismatch(K=28,found=0)`. The two symptoms have one cause.
        texts = [str(q["criteria"][k]) or str(k) for k in keys]
    elif q["type"] == "noul":
        keys = ["yes", "no"]
        texts = ["Yes", "No"]
    elif q["type"] == "score":
        keys = list(q["levels"])
        texts = [str(x) for x in keys]
    else:
        raise ValueError(f"unknown question type {q['type']!r}")
    lab = labels
    # Label keys are NOT uniform across corpus builds. `score` questions carry their level TEXTS
    # in `q["levels"]`, and `build_jev_data.py` keyed its labels by that text -- but the jev-bench
    # converter keyed them by level INDEX ('0','1','2'). Accepting either, positionally, is the
    # only thing that serves both; picking one silently breaks the other corpus.
    vals = []
    for i, k in enumerate(keys):
        if k in lab:
            vals.append(float(lab[k]))
        elif str(i) in lab:
            vals.append(float(lab[str(i)]))
        else:
            raise KeyError(
                f"no label for option {k!r} (index {i}); available keys: {sorted(lab)[:6]}"
            )
    return keys, texts, vals


def char_end_token(offsets, char_idx):
    """Token index whose character span contains `char_idx`, or None if none does.

    Matching a marker as an exact token-id subsequence is NOT robust, and the failure is silent.
    `</option>` BPE-merges with whatever precedes it, so the ids the tokenizer produces for the
    marker in isolation ([510, 1979, 29] for Qwen3.8) do not appear in the stream when the option
    text is a phrase. Measured on jev_real:

        K=28, one-word options   28 markers in text -> 28 in the token stream
        K=2,  "Yes"/"No"          2                 ->  2
        K=3,  sentence options    3                 ->  0     (MHS score, ChaosNLI)
        K=4/5, phrase options     4/5               ->  0     (typed-decisions)

    That silently dropped 16,699 of 31,681 units -- 53% of the benchmark, including every
    ChaosNLI record, which is the sharpest external milestone. Mapping character offsets to
    tokens has no such failure mode.
    """
    for j in range(len(offsets)):
        s, e = int(offsets[j][0]), int(offsets[j][1])
        if s <= char_idx < e:
            return j
    return None


def marker_token_positions(text, offsets, marker):
    """Every token index that ends an occurrence of `marker`, in order."""
    out = []
    for m in re.finditer(re.escape(marker), text):
        j = char_end_token(offsets, m.end() - 1)
        if j is not None:
            out.append(j)
    return out


def iter_units(data_dir, splits, sources, limit=None):
    """Yield decision units from the jsonl corpus, breadth-first across sources."""
    n = 0
    per_source = Counter()
    files = []
    for split in splits:
        for p in sorted(Path(data_dir, split).glob("*.jsonl")):
            if sources and p.stem not in sources:
                continue
            files.append((split, p.stem, p))
    # Interleave sources so a --limit smoke test samples every corpus family.
    for split, source, path in files:
        with path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                for qid, q in rec["questions"].items():
                    keys, texts, target = question_options(q, rec["labels"][qid])
                    yield {
                        "split": split,
                        # Prefer the RECORD's own `source` over the filename stem. jev_real keeps
                        # every jev-bench config in a single `real.jsonl`, so the stem collapses
                        # go_emotions, measuring_hate_speech, civil_comments and ChaosNLI into one
                        # source called "real" -- which made the per-source table useless exactly
                        # where it matters most. The stem remains the fallback for corpora that
                        # do not carry the field.
                        "source": rec.get("source") or source,
                        # the record's licence string, so the TRAINER can refuse non-commercial data (user, 2026-10-01:
                        # never train on NC sets; NC is fine for the public non-commercial eval). Scoring ignores it.
                        "license": str(rec.get("license") or (rec.get("meta") or {}).get("license") or ""),
                        "record_id": rec["id"],
                        # `base_id` groups a record with all its option-reshuffled variants,
                        # so downstream can cluster a bootstrap on it (variants are near
                        # duplicates, not independent evidence) and count T correctly.
                        # Prefer an EXPLICIT top-level `base_id` when the corpus provides one.
                        # jev_real carries it on every record; jev_data does not and uses
                        # `meta.variant_of`. Ignoring the explicit field would silently treat
                        # variants as independent whenever a corpus groups them that way --
                        # which understates every clustered CI downstream.
                        "base_id": (rec.get("base_id")
                                    or rec.get("meta", {}).get("variant_of")
                                    or rec["id"]),
                        "n_raters": (rec.get("meta", {}).get("n_raters")
                                     or rec.get("meta", {}).get("n_annotators")),
                        "qid": qid,
                        "qtype": q["type"],
                        "state": rec["state"],
                        "instructions": q["instructions"],
                        "option_keys": keys,
                        "option_texts": texts,
                        # a noul's written yes/no (or true/false) definitions, as stored; None when absent or
                        # not a noul. question_options shows Yes/No regardless; claude_lmscore.py --noul-defs
                        # renders these (backlog 53).
                        "noul_criteria": q.get("criteria") if q["type"] == "noul" else None,
                        "target": target,
                        # TRUST THE VALUES OVER THE TAG. jev-bench's ChaosNLI ships the full
                        # 100-annotator distribution -- {entailment: 0.19, neutral: 0.72,
                        # contradiction: 0.09} -- but is tagged 'hard'. Taking the tag at face
                        # value excluded the single best disagreement dataset in the corpus from
                        # the aleatoric head's supervision (entropy_mask = soft), made
                        # `corr_conf_agreement` uncomputable for it, and left the sharpest
                        # external milestone (JEV's r = 0.046) out of reach. A label that is not
                        # one-hot IS a distribution, whatever the tag says.
                        "label_kind": ("soft"
                                       if (rec["label_kind"][qid] == "hard"
                                           and sum(1 for v in target if v > 0) > 1)
                                       else rec["label_kind"][qid]),
                    }
                    n += 1
                    per_source[source] += 1
                    if limit and n >= limit:
                        return
    log(f"units enumerated: {n}  by source: {dict(per_source)}")


def render(unit, max_state_chars=None):
    """Render the prompt, truncating the STATE rather than the whole sequence.

    `tokenizer(..., truncation=True, max_length=N)` cuts from the END -- which is exactly
    where `</options>` and `<decision>` live. An over-length unit therefore loses its
    markers, fails the marker check, and is silently dropped. That is the likely cause of
    most of the tail losses, and it eats precisely the longest states. Truncating the state
    string before rendering keeps every marker intact.
    """
    state = unit["state"]
    if max_state_chars and len(state) > max_state_chars:
        state = state[:max_state_chars]
    opts = "\n".join(f"{OPTION_OPEN}{t}{OPTION_CLOSE}" for t in unit["option_texts"])
    return PROMPT.format(
        state=state, instructions=unit["instructions"],
        options=opts, DECISION=DECISION,
    )


# --------------------------------------------------------------------------- extraction
class ShardWriter:
    def __init__(self, out, shard_size, dtype, start_index=0):
        self.out, self.shard_size, self.dtype = Path(out), shard_size, dtype
        self.out.mkdir(parents=True, exist_ok=True)
        self.buf, self.index, self.total = [], int(start_index), 0

    def add(self, rec):
        self.buf.append(rec)
        if len(self.buf) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        phis, hd, offsets, targets = [], [], [0], []
        source, qid, kind, keys, split = [], [], [], [], []
        record_id, base_id, n_raters = [], [], []
        for r in self.buf:
            phis.append(r["phis"])
            hd.append(r["h_decision"])
            offsets.append(offsets[-1] + r["phis"].shape[0])
            targets.append(r["target"])
            source.append(r["source"]); qid.append(r["qid"])
            kind.append(r["label_kind"]); keys.append(r["option_keys"])
            split.append(r["split"])
            record_id.append(r.get("record_id")); base_id.append(r.get("base_id"))
            n_raters.append(r.get("n_raters"))
        payload = {
            "phis": torch.cat(phis).to(self.dtype).cpu(),
            "h_decision": torch.stack(hd).to(self.dtype).cpu(),
            "offsets": torch.tensor(offsets, dtype=torch.int32),
            "targets": torch.cat(targets).float().cpu(),
            "source": source, "qid": qid, "label_kind": kind,
            "option_keys": keys, "split": split,
            # `base_id` for a clustered bootstrap (variants are near-duplicates, so an
            # item-level bootstrap is ~sqrt(2) too narrow) and `n_raters` for the
            # Miller-Madow correction of a small-sample entropy estimate.
            "record_id": record_id, "base_id": base_id, "n_raters": n_raters,
        }
        path = self.out / f"shard_{self.index:05d}.pt"
        torch.save(payload, path)
        n_opt = payload["phis"].shape[0]
        self.total += len(self.buf)
        log(f"wrote {path.name}: {len(self.buf)} units, {n_opt} option rows, "
            f"{path.stat().st_size / 1e6:.1f} MB")
        self.buf, self.index = [], self.index + 1

    def close(self):
        self.flush()


def find_text_stack(model, tok, dev):
    """The inner text stack, checked against the full model's hidden_states[-1].

    Calling the text stack instead of `model(...)` skips the LM head (B*L*248320 logits) and the
    retained per-layer hidden states. Shared with train_e2e.py so the fine-tune reads the same
    features this script caches.
    """
    base = getattr(model, "model", None)
    text_stack = None
    for attr in ("language_model", "text_model", "model"):
        cand = getattr(base, attr, None) if base is not None else None
        if cand is not None and hasattr(cand, "embed_tokens"):
            text_stack = cand
            break
    # AutoModelForCausalLM puts the decoder itself at `model.model`.
    if text_stack is None and base is not None and hasattr(base, "embed_tokens"):
        text_stack = base
    if text_stack is None and hasattr(model, "embed_tokens"):
        text_stack = model
    if text_stack is None:
        raise SystemExit("no language-model submodule reachable for hidden states")

    # with the tokenizer's own special tokens (Gemma's BOS: without it a base model's activations are ill-conditioned and the
    # probe compares bf16 noise); Qwen adds none, so its probe is unchanged
    probe_in = tok("The capital of France is Paris. The capital of Italy is", return_tensors="pt", add_special_tokens=True)
    probe_in = {k: v.to(dev) for k, v in probe_in.items()}
    with torch.no_grad():
        full = model(**probe_in, output_hidden_states=True, use_cache=False)
        ref = full.hidden_states[-1].float()
        ref_logits = full.logits[:, -1].float()
        del full
        got = text_stack(**probe_in, output_hidden_states=False,
                         use_cache=False).last_hidden_state.float()
    if ref.shape != got.shape:
        raise SystemExit(f"shape mismatch: {tuple(ref.shape)} vs {tuple(got.shape)}")
    delta = float((ref - got).abs().max())
    log(f"text stack {type(text_stack).__name__}: last_hidden_state vs hidden_states[-1] "
        f"max|delta| = {delta:.3e}")
    if delta > 1e-2:
        # Some architectures (Gemma 4) report hidden_states[-1] BEFORE the final norm while last_hidden_state is after it.
        # What the letter-scoring fast path needs is the LOGITS: head(last_hidden_state) must equal the full model's logits
        # at the last position (final_logit_softcapping included, see head_logits). Accept the stack only if that holds.
        with torch.no_grad():
            h = text_stack(**probe_in, use_cache=False).last_hidden_state[:, -1]
            got_logits = head_logits(model, h)
        ld = float((ref_logits - got_logits).abs().max())
        pd = float((torch.softmax(ref_logits, -1) - torch.softmax(got_logits, -1)).abs().max())
        same_top = bool((ref_logits.argmax(-1) == got_logits.argmax(-1)).all())
        log(f"  hidden-state probe failed; logits probe head(last_hidden_state) vs full logits: max|dlogit| = {ld:.3e} "
            f"(bf16 over the whole vocab), max|dprob| = {pd:.3e}, same top-1 = {same_top}")
        if pd > 5e-3 or not same_top:
            raise SystemExit(
                "last_hidden_state does NOT match hidden_states[-1] and head(last_hidden_state) does NOT match the model's "
                "logits. The fast path would score different distributions from the model's, silently. Fix first."
            )
    # PADDING INVARIANCE. Every caller batches right-padded prompts and reads the last real token, which is exact only under a
    # correct causal mask. Gemma-4-12B-it's stack failed this (a short prompt batched with a long one moved by 0.26) while
    # passing both probes above, so check it directly: a short and a long prompt, alone vs in one padded batch.
    if getattr(tok, "padding_side", "right") == "right" and tok.pad_token is not None:
        body = ("A customer writes: my order has not arrived after nine days and I need it by Friday. ")
        texts = [body + "Is the customer upset? Answer:", body * 12 + "Is the customer upset? Answer:"]
        with torch.no_grad():
            single = []
            for t in texts:
                e = tok(t, return_tensors="pt", add_special_tokens=True).to(dev)
                single.append(torch.softmax(head_logits(model, text_stack(**e, use_cache=False).last_hidden_state[:, -1]), -1))
            e = tok(texts, return_tensors="pt", padding=True, add_special_tokens=True).to(dev)
            h = text_stack(**e, use_cache=False).last_hidden_state
            last = e["attention_mask"].sum(1) - 1
            batch = torch.softmax(head_logits(model, h[torch.arange(len(texts), device=h.device), last]), -1)
        pad_d = max(float((batch[i] - single[i][0]).abs().max()) for i in range(len(texts)))
        log(f"  padding probe: max|dprob| batched vs alone = {pad_d:.3e}")
        if pad_d > 3e-2:
            raise SystemExit(f"the text stack is NOT padding-invariant (max|dprob| {pad_d:.3f}): batched right-padded scoring would "
                             "change each prompt's distribution. Use an exact path (unbatched / equal-length batches).")
    return text_stack


def head_logits(model, h):
    """The model's output logits from final hidden states h: the LM head, then final_logit_softcapping if the config sets
    it (Gemma: cap * tanh(z / cap)). Qwen has no softcap, so this is exactly lm_head(h) there. (66) A text config with an
    `output_multiplier` (Muse-Glimmer: logits * 0.196 BEFORE its softcap of 20) gets it first, as the model's own forward does;
    no other backbone's config has one, so their logits are unchanged."""
    z = model.get_output_embeddings()(h).float()
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    mult = getattr(cfg, "output_multiplier", None)
    if mult:
        z = z * mult
    cap = getattr(cfg, "final_logit_softcapping", None)
    return cap * torch.tanh(z / cap) if cap else z


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="jev_data")
    ap.add_argument("--out", default="jev_feats")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--splits", default="train,val,test,ood",
                    help="splits to cache; 'ood' is only present if build_jev_data.py ran "
                         "with --holdout-source. Missing dirs glob to nothing, so this is safe.")
    ap.add_argument("--sources", default="", help="comma-separated subset; empty = all")
    ap.add_argument("--limit", type=int, default=None, help="units, for a smoke test")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--max-state-chars", type=int, default=12000,
                    help="truncate the STATE to this many characters (not the whole "
                         "sequence, which would cut off the option markers)")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--skip", type=int, default=0,
                    help="skip the first N units AFTER the length sort. Use to finish a run "
                         "that died partway: the sort is deterministic, so unit N is the same "
                         "unit next time. Pair with --shard-start to avoid clobbering shards.")
    ap.add_argument("--shard-start", type=int, default=0,
                    help="first shard index to write, so a resumed run appends instead of "
                         "overwriting shard_00000.pt")
    ap.add_argument("--resume", action="store_true",
                    help="derive --skip/--shard-start from the shards already on disk. Assumes "
                         "full shards contain exactly --shard-size units, which holds unless a "
                         "unit was dropped; check the DROPPED line in the previous log.")
    a = ap.parse_args()

    splits = [s.strip() for s in a.splits.split(",") if s.strip()]
    sources = {s.strip() for s in a.sources.split(",") if s.strip()}
    dtype = getattr(torch, a.dtype)

    log(f"model {a.model} dtype {a.dtype}")
    tok = AutoTokenizer.from_pretrained(a.model)
    # `ids[:L]` slicing below assumes right padding. With left padding the pad tokens would
    # also flow into the Gated DeltaNet recurrent state and corrupt every readout.
    if getattr(tok, "padding_side", "right") != "right":
        raise SystemExit(f"tokenizer padding_side is {tok.padding_side!r}, expected 'right'")
    # Keep the ids only for the diagnostic below: the marker is now located via character
    # offsets (see `char_end_token`), because id-subsequence matching silently fails whenever
    # `</option>` BPE-merges with a preceding phrase.
    close_ids = tok(OPTION_CLOSE, add_special_tokens=False)["input_ids"]
    dec_ids = tok(DECISION, add_special_tokens=False)["input_ids"]
    log(f"marker token counts (diagnostic only): </option>={len(close_ids)} "
        f"<decision>={len(dec_ids)}")

    # Load order and the hidden-state extraction mirror probe_qwen38.py, which proved them
    # against the real checkpoint: the declared architecture first, then text-only fallbacks.
    from transformers import Qwen3_5ForConditionalGeneration

    t0 = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        a.model, dtype=dtype, device_map="auto", low_cpu_mem_usage=True
    )
    model.eval()
    dev = next(model.parameters()).device
    log(f"model loaded in {time.time() - t0:.1f}s on {dev}")

    # Find the inner text stack. It is the whole OOM fix: calling `model(...)` runs the LM
    # head (B*L*248320 logits) and, with output_hidden_states=True, RETAINS all 65 hidden
    # states. At batch 16 x 4096 tokens that is ~32 GB of logits plus ~43 GB of hidden states
    # -- ~75 GB of waste on a 96 GB card already holding 54 GB of weights. We only ever read
    # the last layer, so call the text stack with output_hidden_states=False instead.
    text_stack = find_text_stack(model, tok, dev)

    # A run that died partway can be continued, because the length sort below is
    # deterministic: unit N is the same unit next time. Derive the skip from what is on disk.
    if a.resume and not (a.skip or a.shard_start):
        existing = sorted(Path(a.out).glob("shard_*.pt"))
        a.shard_start = len(existing)
        a.skip = a.shard_start * a.shard_size
        log(f"--resume: {len(existing)} shards on disk -> skip {a.skip} units, "
            f"resume at shard {a.shard_start}")

    writer = ShardWriter(a.out, a.shard_size, dtype, start_index=a.shard_start)
    units = list(iter_units(a.data, splits, sources, a.limit))
    log(f"units enumerated: {len(units)}")

    # Length-sort so each batch pads to a similar width; the head-marker scan is unaffected.
    # This also puts the longest sequences LAST, which is why the first pass OOM'd at 86%.
    order = sorted(range(len(units)), key=lambda i: len(units[i]["state"]) + 20 * len(units[i]["option_texts"]))
    units = [units[i] for i in order]
    if a.skip:
        units = units[a.skip :]
        log(f"skipped {a.skip} already-cached units; {len(units)} remain")

    counts = Counter()
    dropped = Counter()
    done = 0
    t_start = time.time()
    for start in range(0, len(units), a.batch_size):
        batch = units[start : start + a.batch_size]
        texts = [render(u, a.max_state_chars) for u in batch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=a.max_len, add_special_tokens=False, return_offsets_mapping=True)
        # Offsets are pulled out BEFORE the device move: they are only needed on CPU, and
        # `offset_mapping` is not a model input.
        offs_batch = enc.pop("offset_mapping")
        enc = {k: v.to(dev) for k, v in enc.items() if k != "token_type_ids"}
        lengths = enc["attention_mask"].sum(dim=1).tolist()

        with torch.no_grad():
            # No LM head, no retained intermediates. See the equivalence gate above.
            hidden_all = text_stack(**enc, output_hidden_states=False,
                                    use_cache=False).last_hidden_state.float()

        for b, u in enumerate(batch):
            L = int(lengths[b])
            K = len(u["option_keys"])
            offs = offs_batch[b][:L]
            close_pos = marker_token_positions(texts[b], offs, OPTION_CLOSE)
            dec_pos = marker_token_positions(texts[b], offs, DECISION)
            if len(close_pos) != K or len(dec_pos) != 1:
                dropped[f"marker_mismatch(K={K},found={len(close_pos)},dec={len(dec_pos)})"] += 1
                continue
            h = hidden_all[b, :L]
            writer.add({
                "phis": h[close_pos].to(torch.float32),
                "h_decision": h[dec_pos[0]].to(torch.float32),
                "target": torch.tensor(u["target"], dtype=torch.float32),
                "option_keys": u["option_keys"], "source": u["source"],
                "qid": u["qid"], "label_kind": u["label_kind"], "split": u["split"],
                "record_id": u.get("record_id"), "base_id": u.get("base_id"),
                "n_raters": u.get("n_raters"),
            })
            counts[u["source"]] += 1
            done += 1

        if (start // a.batch_size) % 25 == 0:
            el = time.time() - t_start
            rate = done / el if el > 0 else 0
            log(f"{done}/{len(units)} units  {rate:.1f} u/s  "
                f"eta {(len(units) - done) / rate / 60:.1f} min" if rate else f"{done} units")

    writer.close()
    log(f"DONE {done} units cached, {writer.total} written to shards")
    if counts:
        log(f"by source: {dict(counts)}")
    if dropped:
        log(f"DROPPED: {dict(dropped)}")
    log(f"output: {a.out}")


if __name__ == "__main__":
    main()
