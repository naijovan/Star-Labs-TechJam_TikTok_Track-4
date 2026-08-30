"""Stage 3 evaluation harness.

Runs the UNCHANGED production agent through the UNCHANGED organizer evaluator.

Every counterfactual lives here and nowhere else:
  * message transformations are applied by monkeypatching
    `local_evaluator.initial_message` / `customer_reply` / `behavior_for`
    for the duration of ONE session, restored in a `finally`;
  * F3's turn-1 clip wraps `Agent.respond` for one arm only;
  * F4's config sweep sets module attributes on `submission.config` and restores
    them afterwards.

`submission/` and `evaluator/` files are never written to.  Scoring arithmetic is
always the organizer's own `LE.evaluate`, never a re-implementation, so a result
cannot silently diverge from the real scoring path.
"""
from __future__ import annotations

import collections
import contextlib
import copy
import hashlib
import json
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import evaluator.local_evaluator as LE
import submission.config as CONFIG
from submission.agent import Agent, FILLER

from tools.suite import SUITE_VERSION, transforms as TR

FROZEN_SUITE_SHA256 = "9bb68c91547ae96febfcfbe459c9036b69d505fbe3d28202ed45fcb181e99bdb"

FAILURE_STAGES = ("none", "scenario_misclassified", "category_extraction",
                  "clue_extraction", "filler_leaked", "override_gate_stuck", "retrieval")


# ---------------------------------------------------------------- environment

def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def agent_source_sha256() -> str:
    """Hash the agent + config source so a run is pinned even on a dirty tree."""
    digest = hashlib.sha256()
    for name in ("submission/agent.py", "submission/config.py", "submission/tracelog.py"):
        digest.update(Path(name).read_bytes())
    return digest.hexdigest()


def config_snapshot() -> dict:
    return {k: getattr(CONFIG, k) for k in sorted(dir(CONFIG))
            if k.isupper() and not k.startswith("_")}


def environment() -> dict:
    import os
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "<unset>"),
        "sqlite_version": __import__("sqlite3").sqlite_version,
    }


# ---------------------------------------------------------------- transforms

def _damage_category(text: str, generator: str) -> str:
    """F4 category damage.  D1 reproduces tools/verify_agent.py's cat_noise;
    D2 is an INDEPENDENT generator so the thresholds are not swept against the
    one condition they were tuned under."""
    if generator == "D1_word_reversal":
        words = text.split()
        return " ".join(reversed(words)) if len(words) > 1 else (text[:-1] if len(text) > 4 else text)
    if generator == "D2_char_corruption":
        rng = random.Random(hashlib.md5(text.encode()).hexdigest())
        chars = [c for c in text if rng.random() > 0.18]
        out = "".join(chars).strip()
        return out if len(out) >= 3 else text[: max(3, len(text) - 2)]
    return text


class _Patcher:
    """Applies one test case's transformation to the evaluator, then restores."""

    def __init__(self, case: dict) -> None:
        self.case = case
        self.transformation = case.get("transformation") or {}
        self.surface = self.transformation.get("surface")
        self.variant: Dict[str, str] = self.transformation.get("template_variant") or {}
        self.cmap: Dict[str, str] = self.transformation.get("constraint_map") or {}
        self.scripted: Dict[str, str] = self.transformation.get("scripted_turns") or {}
        self.kind = self.transformation.get("kind")
        self.reply_calls = 0
        self._saved: Dict[str, Any] = {}

    # -- message rewriters -------------------------------------------------

    def _sub_constraints(self, message: str) -> str:
        for original, paraphrase in sorted(self.cmap.items(), key=lambda kv: -len(kv[0])):
            message = message.replace(original, paraphrase)
        return message

    def _rewrite_opening(self, message: str, template_name: str, cat: str) -> str:
        if self.cmap:
            return self._sub_constraints(message)
        if template_name not in self.variant:
            return message
        canonical = TR.CANONICAL[template_name]
        marker = canonical.replace("<CAT>", cat)
        if template_name == "opening_buying":
            head = "I'm looking for " + cat + ". A key requirement is: "
            constraint = message[len(head):].rstrip(".") if message.startswith(head) else ""
            return TR.render(self.variant[template_name], cat=cat, constraint=constraint)
        if template_name == "opening_browsing":
            return TR.render(self.variant[template_name], cat=cat)
        head = "I'm looking for " + cat + ". "
        old = message[len(head):] if message.startswith(head) else ""
        return TR.render(self.variant[template_name], cat=cat, old=old)

    def _rewrite_reply(self, message: str) -> str:
        if "For that, what matters is: " in message:
            joined = message.split("For that, what matters is: ", 1)[1]
            joined = joined[:-1] if joined.endswith(".") else joined
            if self.cmap:
                return "For that, what matters is: " + self._sub_constraints(joined) + "."
            if "payout" in self.variant:
                return TR.render(self.variant["payout"], joined=joined)
            return message
        for name, canonical in (("filler_none", TR.CANONICAL["filler_none"]),
                                ("filler_drained", TR.CANONICAL["filler_drained"]),
                                ("filler_boundary", TR.CANONICAL["filler_boundary"])):
            if name == "filler_none":
                if message == canonical and name in self.variant:
                    return TR.render(self.variant[name])
                continue
            prefix = canonical.split("<ATTR>")[0]
            suffix = canonical.split("<ATTR>")[1]
            if message.startswith(prefix) and message.endswith(suffix) and name in self.variant:
                attr = message[len(prefix): len(message) - len(suffix)]
                return TR.render(self.variant[name], attr=attr)
        return message

    # -- context manager ---------------------------------------------------

    def __enter__(self):
        case, self.reply_calls = self.case, 0
        original_initial, original_reply = LE.initial_message, LE.customer_reply
        original_behavior = LE.behavior_for
        self._saved = {"initial_message": original_initial, "customer_reply": original_reply,
                       "behavior_for": original_behavior}
        template_name = case["clean_inputs"]["opening_template"]

        def initial_message(sample, category, disclosed):
            if self.kind in ("D1_word_reversal", "D2_char_corruption"):
                category = _damage_category(category, self.kind)
            message = original_initial(sample, category, disclosed)
            message = self._rewrite_opening(message, template_name, category)
            if self.scripted.get("1"):
                message = self.scripted["1"]
            return message

        def customer_reply(sample, ask_attribute, disclosed, boundary_used):
            message, boundary = original_reply(sample, ask_attribute, disclosed, boundary_used)
            self.reply_calls += 1
            turn = self.reply_calls + 1
            message = self._rewrite_reply(message)
            if self.scripted.get(str(turn)):
                message = self.scripted[str(turn)]
            return message, boundary

        def behavior_for(scenario, card, rng):
            behavior = original_behavior(scenario, card, rng)
            override = behavior.get("override")
            if override:
                new_value = str(override.get("new_value", ""))
                if self.cmap:
                    override["new_value"] = self._sub_constraints(new_value)
                    override["message"] = ("Actually, ignore my earlier preference. "
                                           f"What I need is: {override['new_value']}.")
                elif "override" in self.variant:
                    override["message"] = TR.render(self.variant["override"], constraint=new_value)
            return behavior

        LE.initial_message, LE.customer_reply, LE.behavior_for = (
            initial_message, customer_reply, behavior_for)
        return self

    def __exit__(self, *exc):
        LE.initial_message = self._saved["initial_message"]
        LE.customer_reply = self._saved["customer_reply"]
        LE.behavior_for = self._saved["behavior_for"]
        return False


@contextlib.contextmanager
def config_override(overrides: Dict[str, Any]):
    saved = {k: getattr(CONFIG, k) for k in overrides}
    try:
        for key, value in overrides.items():
            setattr(CONFIG, key, value)
        yield
    finally:
        for key, value in saved.items():
            setattr(CONFIG, key, value)


@contextlib.contextmanager
def turn1_clip(agent: Agent):
    """F3 diagnostic arm.  Clips turn 1 to a single card and rewinds `seen`."""
    original = agent.respond

    def clipped(session_id, user_message, turn, top_k):
        response = original(session_id, user_message, turn, top_k)
        if turn == 1 and len(response["recommendations"]) > 1:
            keep = response["recommendations"][:1]
            state = agent.S.get(session_id)
            if state is not None:
                for item in response["recommendations"][1:]:
                    state["seen"].discard(item["parent_asin"])
                state["last"] = [d["parent_asin"] for d in keep]
            response = dict(response, recommendations=keep)
        return response

    agent.respond = clipped
    try:
        yield
    finally:
        agent.respond = original


# ---------------------------------------------------------------- diagnosis

def diagnose(case: dict, state: Optional[dict], hit: bool) -> str:
    """Attribute a failure to a pipeline stage.  Deliberately reports a STAGE,
    not a retrieval route -- `route` labels are not outcome labels (TEST_MATRIX
    section 0.4): `weak` fires on 156 of 200 clean sessions while doing nothing."""
    if state is None:
        return "retrieval" if not hit else "none"
    scenario = case["scenario_type"]
    is_override = bool(state.get("is_override"))
    if is_override != (scenario == "intent_override"):
        return "scenario_misclassified"
    if is_override and not state.get("override_fired"):
        return "override_gate_stuck"
    true_category = case["clean_inputs"]["coarse_category"]
    if state.get("cat") != true_category:
        return "category_extraction"
    free_text = " ".join(state.get("free") or [])
    if any(marker in free_text for marker in FILLER):
        return "filler_leaked"
    if case["family"] == "F2A":
        disclosed = set(case["clean_inputs"]["constraint_pool"])
        if state.get("clues") and not (set(state["clues"]) & disclosed):
            return "clue_extraction"
    if hit:
        return "none"
    return "retrieval"


# ---------------------------------------------------------------- execution

@dataclass
class SessionOutcome:
    hit: bool
    first_hit_turn: Optional[int]
    best_rank: Optional[int]
    reciprocal_rank: float
    turns: List[dict]
    failure_stage: str
    state: dict


def _sample_for(case: dict) -> dict:
    return {
        "sample_id": case["sample_id"],
        "scenario_type": case["scenario_type"],
        "category_bucket": "clothing",
        "difficulty_bucket": "medium",
        "user_profile": case["user_profile"],
        "ground_truth": {"parent_asin": case["target_parent_asin"]},
    }


def run_session(agent: Agent, case: dict, catalog, *, clip: bool = False,
                config: Optional[dict] = None) -> SessionOutcome:
    """One session through the organizer's own evaluate()."""
    ids, cats, prods = catalog
    turns: List[dict] = []
    captured: Dict[str, Any] = {}
    original_respond = agent.respond

    def spy(session_id, user_message, turn, top_k):
        response = spy.__wrapped_target__(session_id, user_message, turn, top_k)
        captured["session_id"] = session_id
        turns.append({
            "turn": turn,
            "user_message": user_message,
            "ask_attribute": response.get("ask_attribute"),
            "recommendations": [d.get("parent_asin") for d in response.get("recommendations", [])],
        })
        return response

    # The spy must wrap the OUTERMOST respond so it logs what the evaluator
    # actually received.  Installing it before the clip context logged the
    # pre-clip page -- a diagnostic-only bug, but it made `turns` unusable for
    # exactly the mechanism F3 exists to measure.
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(_Patcher(case))
            if config:
                stack.enter_context(config_override(config))
            if clip:
                stack.enter_context(turn1_clip(agent))
            inner = agent.respond
            spy.__wrapped_target__ = inner
            agent.respond = spy
            try:
                result = LE.evaluate(agent, [_sample_for(case)], ids, cats, prods)
            finally:
                agent.respond = inner
    finally:
        agent.respond = original_respond

    session = result["sessions"][0]
    state = agent.S.get(captured.get("session_id")) or {}
    serialisable = {
        "cat": state.get("cat"), "cat_sure": state.get("cat_sure"),
        "clues": list(state.get("clues") or []), "free": list(state.get("free") or []),
        "is_override": state.get("is_override"), "override_fired": state.get("override_fired"),
        "dead": sorted(state.get("dead") or []), "n_seen": len(state.get("seen") or []),
    }
    return SessionOutcome(
        hit=bool(session["hit"]),
        first_hit_turn=session["first_hit_turn"],
        best_rank=session["best_rank"],
        reciprocal_rank=session["reciprocal_rank"],
        turns=turns,
        failure_stage=diagnose(case, state, bool(session["hit"])),
        state=serialisable,
    )


def technical_contribution(outcome: SessionOutcome) -> dict:
    """Per-session decomposition of TechnicalScore.  Efficiency is linear in MTTC
    and never clips, so a session's contribution is exact (TRICKS.md section 0)."""
    turn = outcome.first_hit_turn if outcome.first_hit_turn is not None else LE.MAX_TURNS + 1
    return {
        "hit": 1.0 if outcome.hit else 0.0,
        "reciprocal_rank": outcome.reciprocal_rank,
        "mttc_contribution": float(turn),
        "technical_score_contribution": (0.50 * (1.0 if outcome.hit else 0.0)
                                         + 0.30 * outcome.reciprocal_rank
                                         + 0.20 * max(0.0, min(1.0, (11.0 - turn) / 10.0))),
    }


def evaluate_case(agent: Agent, case: dict, catalog) -> List[dict]:
    """Run every arm / config point a case declares.  Returns one row per session."""
    rows: List[dict] = []
    base = {
        "test_id": case["test_id"], "suite_version": case["suite_version"],
        "family": case["family"], "cell_id": case["cell_id"], "tier": case["tier"],
        "class": case["class"], "severity": case["severity"],
        "requested_severity": case.get("requested_severity"),
        "achieved_severity": case.get("achieved_severity"),
        "scenario_type": case["scenario_type"],
        "target_parent_asin": case["target_parent_asin"],
        "scheme": case.get("scheme"), "sentinel": case.get("sentinel", False),
        "stress_bound": case.get("stress_bound", False), "ood": case.get("ood", False),
        "diagnostic_case": case.get("diagnostic", False),
        "excluded_from_headline": case.get("excluded_from_headline", False),
    }

    def record(outcome: SessionOutcome, arm: str, *, diagnostic: bool,
               config: Optional[dict] = None) -> dict:
        row = dict(base)
        row.update({
            "arm": arm, "diagnostic_arm": diagnostic, "config_override": config or {},
            "hit": outcome.hit, "first_hit_turn": outcome.first_hit_turn,
            "best_rank": outcome.best_rank, "reciprocal_rank": outcome.reciprocal_rank,
            "failure_stage": outcome.failure_stage, "turns": outcome.turns,
            "agent_state": outcome.state,
        })
        row.update(technical_contribution(outcome))
        return row

    if case["family"] == "F4" and case.get("config_grid"):
        for index, overrides in enumerate(case["config_grid"]):
            outcome = run_session(agent, case, catalog, config=overrides)
            arm = "config_control" if not overrides else "config_" + "_".join(
                f"{k}={v}" for k, v in sorted(overrides.items()))
            rows.append(record(outcome, arm, diagnostic=True, config=overrides))
        return rows

    control = run_session(agent, case, catalog)
    rows.append(record(control, "as_ships", diagnostic=False))

    if "turn1_clipped" in (case.get("arms") or []):
        clipped = run_session(agent, case, catalog, clip=True)
        row = record(clipped, "turn1_clipped", diagnostic=True)
        row["paired_delta"] = {
            "reciprocal_rank": clipped.reciprocal_rank - control.reciprocal_rank,
            "first_hit_turn": ((clipped.first_hit_turn or 11) - (control.first_hit_turn or 11)),
            "technical_score_contribution": (technical_contribution(clipped)["technical_score_contribution"]
                                             - technical_contribution(control)["technical_score_contribution"]),
        }
        row["control_outcome"] = {"hit": control.hit, "best_rank": control.best_rank,
                                  "first_hit_turn": control.first_hit_turn,
                                  "reciprocal_rank": control.reciprocal_rank}
        rows.append(row)
    return rows


def load_cases(path: str | Path) -> List[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
