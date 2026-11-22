# synth_workflows

Latent-factor case generators for NEW workflow families, in the style of `LocalLLaMA/typed-decisions`
(backlog item 48). Each case is one state plus five typed questions. `build_jev_typed_synth.py` turns the cases
into unlabelled corpus records, and `trainers/claude_verbal_label.py` labels them with open ~4B teachers that
verbalise distributions, K samples averaged. The contract, the benchmark format and the exclusion list are in
`research/typed_synth_spec.md`. Read its sections 2 to 4 before you write a workflow.

```
synth_workflows/
  __init__.py          registry: load_all() imports every workflow module and checks it
  base.py              Schema / Workflow interface, question builders, sampling + rendering helpers
  exclusion.py         the benchmark exclusion list, the reskin checks, and the n-gram overlap audits
  contract.py          the unlabelled-record contract (validate_unlabelled / attach_labels / validate_labelled)
  it_change_approval.py   the REFERENCE workflow: copy its structure
  <workflow>.py        one file per workflow
```

## Adding a workflow (one file, nothing else to edit)

1. Pick a family from the proposed list in `research/typed_synth_spec.md` section 8, or a new one. Check it
   against the exclusion list first (section 3). The four benchmark workflows and their near neighbours are
   forbidden: customer support threads, security alert triage, vendor invoices / AP, and AI-agent run review.
   Families on the "constrained" list are allowed only under the stated constraint.
2. Create `synth_workflows/<workflow_name>.py`. The file name is the workflow name (snake_case). Define at
   least 5 `Schema` subclasses and a module-level `WORKFLOW = Workflow(name="<workflow_name>", ...)`. The
   registry finds the module by itself.
3. Preview the cases and run the self-test:
   ```
   python3 build_jev_typed_synth.py --preview <workflow_name> --n 2      # print cases for every schema x surface
   PYTHONPATH=/tmp/stubs python3 build_jev_typed_synth.py --self-test --workflows <workflow_name>
   ```
   On a laptop without torch or transformers, `/tmp/stubs` holds `torch.py` (`class Tensor: pass`) and
   `transformers.py` (`AutoTokenizer = None`). The self-test also stubs them in memory if they are missing.
4. Read 10 or more previewed cases as a careful reviewer would. Would you really hesitate on each question for
   a good share of the cases? Does every case read like a real artefact?

## A schema

```python
class MySchema(Schema):
    name = "snake_case_name"                 # unique within the workflow
    description = "One line: the decision this case asks for."
    surfaces = {"json": 0.5, "email": 0.25, "form": 0.25}   # >= 3 surfaces; "json" share >= 0.3

    def sample_factors(self, rng):          # -> dict of latent factors (JSON-serialisable after base.jsonable)
    def render(self, f, rng, surface):      # -> state string for that surface
    def questions(self):                    # -> {qid: choice(...) | noul(...) | score(...)}, exactly 5, FIXED
```

### Rules the code checks (`load_all`, `make_case` and the self-test fail on them)

- Exactly 5 questions per schema, with at least one each of `choice`, `noul` and `score`, and at most 3 of
  any one type. Aim for the benchmark mix: 2 score + 2 choice + 1 noul, or 2 score + 1 choice + 2 noul.
  Across a workflow that gives roughly 40% score, 30% choice and 30% noul.
- `choice`: 3 to 6 options (prefer 4 or 5). Keys are snake_case. Build with `base.choice`, which sorts the keys.
  The instructions are a question ending in `?`.
- `score`: 3 to 6 levels (prefer 4, sometimes 5), listed LOWEST first. The instructions are a question ending
  in `?`. A level text may not be a bare digit.
- `noul`: the instructions are a declarative proposition ending in `.`, for example "The claimant's policy
  was in force on the date of loss." Give about two thirds of your noul questions definitions:
  `noul(text, yes="when yes is right", no="when no is right")`. They are stored as
  `criteria: {"yes": ..., "no": ...}`.
- No qid may be a benchmark question id (`action`, `category`, `churn_risk`, `credential_compromise`,
  `discrepancy_severity`, `disposition`, `duplicate`, `matches_order`, `needs_human`, `needs_review`, `outcome`,
  `risk`, `severity`, `true_positive`, `urgency`). Name questions for your domain, for example
  `claim_routing` or `fraud_indicator_strength`.
- Every schema has a `"json"` surface rendered with `base.dumps(obj)`, which gives compact JSON with sorted
  keys (the benchmark's own state convention). Its sampling share must be at least 0.3. Add at least two
  other surfaces: email, ticket or form text, log lines, chat, memo, markdown table, pretty JSON with
  shuffled keys (`base.dumps_pretty`), and so on.
- States run 150 to 4,000 characters. Aim for 250 to 2,000; the benchmark's run from about 230 to 1,840.

### Rules for review (not machine-checked; they matter more)

- **Spread by design.** Draw a `regime(rng)` first ('clear', 'borderline' or 'conflicting') and let it steer
  the factors. Borderline cases sit near a stated threshold (`near(rng, threshold, spread)`). Conflicting cases
  carry signals that point opposite ways: a clean record with one alarming fact, or a self-assessment that
  the evidence contradicts. Some cases leave a key fact out (`maybe(rng, v, p)`; render it as absent, "TBD",
  "see attachment" and so on). Keep 'clear' a minority. The benchmark's gold is soft most of the time
  (mean top probability about 0.66).
- **Latent factors.** Use 8 to 15 factors per schema, mixing categorical, numeric, boolean and nullable ones.
  Some are rendered as fields; some only colour the free text (tone, a hedge in a note). Keep every
  factor, including hidden ones such as `regime`, in the returned dict; it lands in `meta.factors` and is
  never shown to the model. Never render a factor NAME that gives the answer away ("regime": "conflicting",
  "should_reject": true).
- **Questions are answerable but not trivial.** Each question should depend on two or three factors and not
  be read straight off a field. Do not put the verdict in the state: no "recommendation: approve". Opinions
  from people inside the artefact ("reviewer thinks it's fine") are fine and useful, because they conflict
  with evidence.
- **Standalone option text.** The scored prompt shows the DESCRIPTION of each option, not its key, so every
  criterion must read on its own and differ clearly from its siblings.
- **Realistic artefacts.** Use the house styles real systems produce: ITSM exports, claim notes, CI logs,
  review forms, emails with a subject line. Vary field order, date and money formats (`fmt_date`, `money`),
  and optional fields. Put light noise (`noisy`) in free text only, never in JSON keys or numbers.
- **Your own words.** Never copy benchmark state, question or criterion wording (see
  `research/typed_synth_spec.md` section 4). At most one time-pressure question per schema, and anchor its
  levels in your domain ("before the next board meeting", "before the shipment clears port"), not in a
  generic "routine / elevated / critical" scale.
- **Fictional and safe.** Use names from `base.person` and `base.org`, example.com-style domains, invented ids,
  and no real people or companies. Do not make a protected attribute (race, religion, disability,
  nationality, age, sex) the deciding factor in a decision.
- **Deterministic.** Use only the `rng` you are given. Do not use the module-level `random`, `time.now()` or
  `uuid`, and do not iterate over a `set` while sampling, because set order changes between processes. The
  self-test checks this: the same seed must give byte-identical output.

## Helpers in base.py

Sampling: `regime`, `chance`, `pick`, `pick_w({value: weight})`, `near`, `between`, `maybe`,
`lognormal_amount`.

Rendering: `dumps` (the benchmark JSON), `dumps_pretty`, `prune` (drop None fields and emptied sections),
`kv_lines` (several "Label: value" house styles), `bullets`, `numbered`, `table` (markdown or fixed width),
`money`, `pct`, `fmt_date`, `fmt_ts`, `rand_date`, `rand_ts`, `ident`, `hexid`, `person`, `initials_name`,
`org(rng, kind)`, `email_addr`, `show` (a value or a missing-value word), `noisy`, `join_sentences`.

## Interface changes (kept backward-compatible; re-read before finishing a module)

- 2026-09-30: `base.org` never returns a pool combination that names a well-known real business ("Harbor
  Freight", and the others in `base.REAL_NAME_CLASHES`). Its signature and rng use are unchanged:
  `rng.choice` for the prefix, then `rng.randrange` for the suffix, which draws exactly as `rng.choice` did.
  Output differs only in the swapped combinations.
