# Final evaluation run — procedure

Per `docs/final_evaluation_faq.md` §1, the 800 final sessions are released
**after** the Devpost deadline and **we** run the official evaluator ourselves,
using the exact commit submitted before the deadline. Nothing may be modified
after the package is released.

## Before the deadline

- [ ] Submitted commit is tagged `submission` and pushed.
- [ ] Devpost links the public repository and the demo video.
- [x] Submission commit: `2a74e47c97dbebacaa01bc250671b874439aaa77` (tag `submission`)
- [ ] Record that hash on Devpost

## When the final package is released

**Do not modify the Agent, config, indexes or any solution component.** The
frozen commit is what runs.

```bash
git clone <repo-url> techjam-final && cd techjam-final
git checkout submission                 # the frozen tag, NOT main
# place the released final catalog/sessions exactly as the package instructs
python3 -m evaluator.local_evaluator    # unmodified official evaluator
```

Sanity check before trusting the run:

```bash
git status                              # must be clean
git rev-parse HEAD                      # must equal the submitted hash
PYTHONPATH=. python3 tools/smoke_test.py submission.agent   # must PASS
```

## Evidence to retain (the organizer may request it)

Keep all of the following together, unedited:

1. `results.json` — including per-session results. **Copy it aside immediately**;
   re-running the evaluator overwrites it.
2. The submitted commit hash (`git rev-parse HEAD`).
3. Environment details:

```bash
python3 -VV; uname -a; python3 -c "import platform;print(platform.platform())"
python3 -m pip freeze          # expected: empty / irrelevant, we use no deps
```

4. Wall-clock start and end time of the run, and the machine used.

Optional supporting evidence, if logs are requested: set `TRACE_PATH` in
`submission/config.py` to a file path and re-run — this writes one JSONL record
per turn (route taken, candidate count, state, emitted IDs). **Only do this on a
separate copy after the official run is complete and saved**, since editing
config is a modification of the frozen solution.

## Environment used for the reported public-set results

| Item | Value |
|---|---|
| Python | CPython 3.11.12 |
| OS / hardware | macOS, MacBook Pro (Apple M4), 16 GB RAM |
| GPU / MPS | Not used |
| Dependencies | None (standard library only) |
| Network | Not required, none contacted |
| Startup | ~6 s (measured 6.03 s) |
| Memory | ~724 MiB after build; ~938 MiB peak across a 200-session run |
| Latency | 0.26 ms median per turn (p95 2.9 ms, worst 96.8 ms) |
| Full run | build + 200 sessions in 6.8 s wall clock |
| Tokens / cost | 0 / $0 |
| Determinism | `results.json` byte-identical across `PYTHONHASHSEED` 0 and 1337 |
