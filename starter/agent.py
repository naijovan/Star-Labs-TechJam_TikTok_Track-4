"""Entry point for the official harness.

`evaluator/local_evaluator.py` imports `Agent` from here (its line 12), so this
module re-exports the real implementation, which lives in `submission/agent.py`
with its configuration in `submission/config.py`.

The organizer's original weak-BM25 starter is preserved unchanged in
`starter/_original_bm25_agent.py` for reference and for baseline reproduction.
"""
from submission.agent import Agent  # noqa: F401
