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

The build replays the evaluator's own `intent_card()` over the catalog, so the
indexes hold what each product *would say in a conversation*, not just what
its description contains. That one decision is why exact constraint
intersection resolves most sessions in two turns.

**Online — every turn (0.06 ms median):**

![Per-turn flow: parse, remember, narrow through a seven-route cascade, rank, schedule, ask, reply](submission/pipeline_online.svg)

One discipline organizes the whole cascade: **escalate only on a failure
signature.** A category that had to be fuzzily repaired re-arms the wide
routes; a constraint that resolves nowhere in the exact index opens the
canonicalizer → BM25-trust → dense lane; an intent override demotes the
pre-override category to a guess and erases a remembered slot only when the
catalog proves no product satisfies both values. Nothing expensive runs while
the cheap path is winning.

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
