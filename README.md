# Star Labs — TikTok TechJam 2026 · Track 4: Shopping Copilot

A conversational shopping agent that finds one hidden product among 50,000 in
about two turns. On the 200 public sessions, scored by the organizer's
unmodified evaluator:

| TechnicalScore | hit@10 | MRR | MTTC | tokens | dependencies |
|---|---|---|---|---|---|
| **0.980000** | 1.000 | **1.0000** | 2.00 | 0 | none — Python stdlib |

Every one of the 200 sessions converts at **rank 1**. The baseline starter
agent scores 0.10671. The agent is byte-deterministic: identical output for
any `PYTHONHASHSEED`, guaranteed by a full ordering on every sort.

## How it works

**Offline — built once at startup (3.5 s):**

![Offline build: the catalog feeds an intent_card replay, which builds five in-memory indexes plus an optional dense encoder](submission/pipeline_offline.svg)

What each build step does for robustness, in plain terms:

- **Card replay** — the agent indexes the exact sentences a shopper could say
  about each product, so matching never depends on guessing how a description
  might be phrased.
- **Clue index** — whole preferences are matched exactly; a damaged phrase
  simply fails to match and falls through to the next step, instead of
  matching the wrong product.
- **Category buckets** — the product type is locked from the first message;
  if it doesn't match a known type exactly, it is treated as a guess so later
  steps are allowed to look outside it.
- **BM25 word index** — a keyword safety net that still works when
  preferences arrive reworded or with words missing.
- **Popularity + profile priors** — a sensible ordering for turns where the
  conversation has given nothing to filter on yet.
- **Canonicalizer** — translates casual words into catalog words (violet →
  purple), consulted only when nothing matched.
- **Dense encoder (optional)** — meaning-based matching for heavy rewording;
  off by default so scoring never depends on extra software or a network.

**Online — every turn (0.06 ms median):**

![Per-turn flow: parse, remember, narrow through a seven-route cascade, rank, schedule, ask, reply](submission/pipeline_online.svg)

What each turn step does for robustness, in plain terms:

- **Understand** — anything not recognized with certainty is recorded as a
  guess, not a fact, so one wrong reading can never lock the agent in.
- **Remember** — nothing the shopper said is thrown away casually; a
  preference is dropped only when the catalog proves no product can satisfy
  both it and the newer one, and a mind-change downgrades the old product
  type back to a guess.
- **Shortlist** — the strictest filter runs first and falls back one level at
  a time; there is always a next step, so a misleading message can't leave
  the agent stuck or empty-handed.
- **Order** — the same input always produces the same list (a fixed
  tie-break), so behaviour is reproducible and testable.
- **Choose how many to show** — the agent shows only what it is confident
  about, saves strong candidates for the top spot next turn, and never
  repeats itself, so every turn adds new coverage.
- **Ask one question** — a question every turn keeps new information flowing,
  and the asking stops once the shopper has nothing left, so no turn is
  wasted on a dead end.
- **Answer** — every step runs inside a safety net: if anything fails, the
  agent returns its last good list instead of an error, because an error
  would forfeit the turn.

The theme across all of it: **the expensive or looser step runs only after
the simple, precise one has provably failed.**

## Setup and reproduction

1. Python **>= 3.10** (verified on CPython 3.11.12). No packages to install.
2. Download `catalog.jsonl.gz` from the organizer's participant-kit release,
   verify it against the release `SHA256SUMS`, then:
   ```bash
   gzip -dk catalog.jsonl.gz && mv catalog.jsonl data/catalog.jsonl
   ```
3. Run the official harness:
   ```bash
   python3 -m evaluator.local_evaluator
   ```
   Expected: `technical_score 0.980000` (hit@10 1.000, MRR 1.0000, MTTC 2.00).
   Contract check: `PYTHONPATH=. python3 tools/smoke_test.py submission.agent`

Package details, environment switches, and the feasibility disclosure are in
[submission/README.md](submission/README.md).

## Robustness — measured, not asserted

The private set uses unseen users and targets, so every mechanism was adopted
only after measurement on shifted inputs. The evidence lives in `tools/`:

- `verify_agent.py` — the official 200 under six corruption conditions
  (word drops up to 50%, category damage, template rephrasings).
- `stress_suite.py` — 500 sessions across four shift axes (paraphrase, false
  drift, cold targets, foreign-generator cards).
- `stress10k.py` — 10,000 sessions across five axes the agent was never built
  for, with transform vocabularies disjoint from the agent's own tables.
- `turn1_suite.py` — 1,000 provable turn-1 singletons: the agent converts
  1000/1000 at rank 1 on turn 1.

Components the track brief names were built and measured either way: dense
retrieval ships resolution-gated behind `TECHJAM_DENSE=1` (byte-identical
clean, +0.036 under paraphrase); cross-encoder reranking and time-based slot
decay were implemented, measured, and rejected on the numbers — the probe
scripts and their results are retained in `tools/`.

## Limitations

- The turn-1 parser keys on the evaluator's message templates. Heavy rewrites
  of *both* templates degrade the score to 0.77 (single-template rewrites cost
  about 0.03–0.05); the cascade's BM25 fallback bounds the damage.
- The popularity prior assumes targets are real purchases. On uniformly
  resampled cold targets the score is 0.946 — the elimination cascade, not
  popularity, carries the result.
- Colloquial vocabulary is handled by a fixed 160-entry canonicalizer plus the
  optional dense lane; an open-vocabulary paraphraser would need the latter
  enabled.

## Repository layout

| Path | What it is |
|---|---|
| `submission/` | The agent package: `agent.py`, `config.py`, `tracelog.py`, README, diagrams |
| `evaluator/`, `docs/`, `data/` | The organizer's harness, spec, and public sessions (unmodified) |
| `starter/` | Harness entry shim → `submission.agent`; original BM25 baseline preserved |
| `tools/` | Verification harnesses, stress suites, and the measurement probes behind every design decision |
| `tests/` | Unit tests for the evaluator contract and suite generation |
| `traces/` | A full demonstrated session (intent override, hit at rank 1) |

## Team — Star Labs

- **Jovan** — architecture and the retrieval cascade; intent-override handling
  (category demotion + catalog-wide contradiction proof); canonicalizer; the
  10k and turn-1 stress suites; integration and verification.
- **Germaine** — slot-value scheduler; category-free intersection; BM25-trust
  on unresolved clues; turn-1 one-card cap; the 8.5k-session hardening suite.
- **Ben** — paraphrase-recovery research (constraint recovery, denormalizer,
  category word-count matching) and the unit-test suite.
- **Marcus** — evidence-free page discipline and candidate-pool analysis.
