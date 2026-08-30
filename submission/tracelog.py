"""Structured per-turn trace log (JSONL) — diagnostics and the demo transcript.

Named `tracelog` rather than the design doc's `trace` because `trace` collides
with the standard-library module of that name when the agent is imported as a
bare file.

OFF by default (`config.TRACE_PATH = ""`). Tracing must never be able to cost a
turn, so every operation is wrapped: a write failure silently disables further
writes rather than raising into respond().

Each record is one turn:
    session, turn, msg          what arrived
    new_clues, cat, cat_sure    what stage 1-2 extracted
    route, cand                 which retrieval route answered, pool size
    gated, emitted, ask         what went back to the customer
    dead, error                 exhausted attributes; exception if one fired
Read it with jq, e.g. route health:  jq -r .route trace.jsonl | sort | uniq -c
"""
from __future__ import annotations

import json


class Tracer:
    def __init__(self, path: str = "") -> None:
        self._fh = None
        if path:
            try:
                self._fh = open(path, "a", encoding="utf-8")
            except Exception:
                self._fh = None

    def write(self, record: dict) -> None:
        if self._fh is None:
            return
        try:
            self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()
        except Exception:
            self._fh = None
