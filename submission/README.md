# Star Labs — TechJam 2026 Track 4: Shopping Copilot (submission package)

Entry point: `submission/agent.py` exports `Agent` (also re-exported through
`starter/agent.py` for the starter-path harness). Helper modules:
`config.py` (every tunable constant), `tracelog.py` (optional per-turn JSONL
trace). The scoring path is **pure Python standard library** — no model, no
network, no API keys, zero tokens.

## Setup (one-time)

1. Python **>= 3.10** (PEP 604 annotations). Verified on CPython 3.11.12.
2. Dependencies: **none** — see `requirements.txt` (there is nothing to install).
3. Catalog (not in the repo): download `catalog.jsonl.gz` (18.3 MB) from the
   organizer's `participant-kit` release, verify against the release
   `SHA256SUMS`, then:

   ```bash
   gzip -dk catalog.jsonl.gz && mv catalog.jsonl data/catalog.jsonl
   ```

## Run (one command, from the repo root)

```bash
python3 -m evaluator.local_evaluator
```

Expected on the 200 public sessions: **TechnicalScore 0.980000**
(hit@10 1.000, MRR 1.0000, MTTC 2.00). The agent is byte-deterministic:
identical output for any `PYTHONHASHSEED`. Contract check:

```bash
PYTHONPATH=. python3 tools/smoke_test.py submission.agent
```

`Agent()` also resolves the catalog module-relatively, so constructing it from
any working directory works; an explicit path argument is honoured verbatim.

## Environment variables and flags (all optional; defaults are the submission)

- `TECHJAM_DENSE=1` — enables a dense-retrieval escalation lane
  (`sentence-transformers`, bge-small) that fires only when a constraint
  resolves nowhere in the exact index. OFF by default: the official run may be
  network-disabled and its timeout is unknown, and the clean score is
  byte-identical either way (0.9800). Any failure (missing package, missing
  weights, no network) falls back silently to the stdlib cascade.
- `TRACE_PATH` in `submission/config.py` — set to a file path to log one JSONL
  record per turn (route taken, pool size, state). Empty (off) by default.

## Feasibility disclosure (model policy)

No LLM is called at inference time. Token usage: **0 prompt / 0 completion**
per turn (reported as such in every response). API cost: **$0**; no keys used
or required. Latency: ~3.5 s one-time index build (~341 MiB in-memory),
**0.06 ms median per turn**. Fallback behaviour: every stage is wrapped
fail-safe — on any internal fault the agent returns its last known-good page
rather than raising, since an exception would score the turn empty.

## Layout

```
submission/agent.py      the Agent (parse -> remember -> narrow -> rank ->
                         schedule -> ask -> reply cascade)
submission/config.py     all constants, each annotated with its measurement
submission/tracelog.py   never-raises JSONL tracer (off unless TRACE_PATH set)
submission/requirements.txt
```

Method, measurements, and limitations are covered in the project report and
the repository root README.
