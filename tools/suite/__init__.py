"""Adversarial test suite for the Track 4 shopping agent.

Design: `TEST_MATRIX.md`.  Measurements it rests on: `MEASUREMENT_LOG.md`.

Hard rules this package obeys, in every module:

* It NEVER imports from, mutates, or monkeypatches `submission/` at import time.
  Counterfactual agent behaviour lives only inside the harness, applied
  explicitly and reverted in a `finally`.
* Every product statistic is derived by calling the ORGANIZER'S OWN
  `evaluator.local_evaluator.intent_card` / `coarse_category`, never a local
  re-implementation.  A drifting replica is the exact failure mode the suite
  exists to detect, so the suite must not contain one.
* Determinism is load-bearing.  Every iteration order is explicitly sorted and
  every sort ends in `parent_asin`.  No `set` is ever iterated in an order that
  reaches an output.
"""
from __future__ import annotations

SUITE_VERSION = "1.0.0"

__all__ = ["SUITE_VERSION"]
