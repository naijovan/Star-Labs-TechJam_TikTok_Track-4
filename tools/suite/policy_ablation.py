"""Harness-only ablation of turn-1 page-size policies (Finding #2).

    PYTHONHASHSEED=0 python3 -m tools.suite.policy_ablation --policies A B C

NOTHING in submission/ is modified.  Each policy is applied by wrapping
`Agent.respond` for the duration of a run and reverted in a `finally`.

The decision variable is `nc` -- the candidate count `_retrieve` already returns
and `respond` already holds in a local.  That is deliberate: any policy that
looks good here is implementable in production as a one-line guard, with no new
state and no new computation.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import json
import time
from pathlib import Path

import evaluator.local_evaluator as LE
from submission.agent import Agent

from tools.suite.harness import _Patcher, load_cases

OUT = Path("results/ablation")

# name -> (nc_threshold, page).  Clip turn 1 to `page` cards when nc > threshold.
# threshold -1 means "always".  None means "leave the agent alone".
POLICIES = {
    "A_current":        None,
    "B_always_1":       (-1, 1),
    "C_gt10_1":         (10, 1),
    "D_gt20_1":         (20, 1),
    "E_gt40_1":         (40, 1),
    "F_gt100_1":        (100, 1),
    "G_gt40_3":         (40, 3),
}


@contextlib.contextmanager
def turn1_policy(agent: Agent, policy):
    """Clip turn 1 to `page` cards when the candidate pool exceeds `threshold`."""
    if policy is None:
        yield
        return
    threshold, page = policy
    original_respond, original_retrieve = agent.respond, agent._retrieve
    captured = {}

    def retrieve(state):
        ranked, nc, route = original_retrieve(state)
        captured["nc"] = nc if nc is not None else len(ranked)
        return ranked, nc, route

    def respond(session_id, user_message, turn, top_k):
        captured.pop("nc", None)
        response = original_respond(session_id, user_message, turn, top_k)
        if turn == 1 and len(response["recommendations"]) > page:
            nc = captured.get("nc", 0)
            if nc > threshold:
                keep = response["recommendations"][:page]
                state = agent.S.get(session_id)
                if state is not None:
                    for item in response["recommendations"][page:]:
                        state["seen"].discard(item["parent_asin"])
                    state["last"] = [d["parent_asin"] for d in keep]
                response = dict(response, recommendations=keep)
        return response

    agent.respond, agent._retrieve = respond, retrieve
    try:
        yield
    finally:
        agent.respond, agent._retrieve = original_respond, original_retrieve


def _sample(case):
    return {"sample_id": case["sample_id"], "scenario_type": case["scenario_type"],
            "category_bucket": "clothing", "difficulty_bucket": "medium",
            "user_profile": case["user_profile"],
            "ground_truth": {"parent_asin": case["target_parent_asin"]}}


def run_case(agent, case, catalog, policy):
    """One session; records the turn-1 diagnostics the ablation needs."""
    ids, cats, prods = catalog
    log, captured = [], {}
    original_respond, original_retrieve = agent.respond, agent._retrieve

    def retrieve(state):
        ranked, nc, route = original_retrieve(state)
        captured.setdefault("nc_by_call", []).append(nc if nc is not None else len(ranked))
        return ranked, nc, route

    def spy(session_id, user_message, turn, top_k):
        response = spy.__wrapped_target__(session_id, user_message, turn, top_k)
        log.append((turn, [d["parent_asin"] for d in response["recommendations"]]))
        return response

    agent._retrieve = retrieve
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(_Patcher(case))
            stack.enter_context(turn1_policy(agent, policy))
            # Spy OUTSIDE the policy so the log records the post-clip page.
            inner = agent.respond
            spy.__wrapped_target__ = inner
            agent.respond = spy
            try:
                result = LE.evaluate(agent, [_sample(case)], ids, cats, prods)
            finally:
                agent.respond = inner
    finally:
        agent.respond, agent._retrieve = original_respond, original_retrieve

    session = result["sessions"][0]
    turn1 = next((recs for t, recs in log if t == 1), [])
    turn = session["first_hit_turn"] if session["first_hit_turn"] is not None else 11
    return {
        "test_id": case["test_id"], "family": case["family"], "cell_id": case["cell_id"],
        "scenario_type": case["scenario_type"], "scheme": case.get("scheme"),
        "hit": bool(session["hit"]), "first_hit_turn": session["first_hit_turn"],
        "best_rank": session["best_rank"], "rr": session["reciprocal_rank"],
        "mttc": float(turn), "turn1_cards": len(turn1),
        "turn1_nc": (captured.get("nc_by_call") or [0])[0],
        "target_in_turn1": case["target_parent_asin"] in turn1,
        "score": (0.5 * (1.0 if session["hit"] else 0.0) + 0.3 * session["reciprocal_rank"]
                  + 0.2 * max(0.0, min(1.0, (11.0 - turn) / 10.0))),
    }


def run_public(agent, catalog, policy):
    ids, cats, prods = catalog
    samples = LE.load_jsonl("data/public_set.jsonl")
    log = []
    original = agent.respond

    def spy(session_id, user_message, turn, top_k):
        response = original(session_id, user_message, turn, top_k)
        if turn == 1:
            log.append(len(response["recommendations"]))
        return response

    agent.respond = spy
    try:
        with turn1_policy(agent, policy):
            result = LE.evaluate(agent, samples, ids, cats, prods)
    finally:
        agent.respond = original
    rows = []
    for session in result["sessions"]:
        turn = session["first_hit_turn"] if session["first_hit_turn"] is not None else 11
        rows.append({"scenario_type": session["scenario_type"], "hit": bool(session["hit"]),
                     "rr": session["reciprocal_rank"], "mttc": float(turn),
                     "first_hit_turn": session["first_hit_turn"],
                     "best_rank": session["best_rank"]})
    return {"overall": result["recommended_technical_score"], "sessions": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+", required=True)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cases = load_cases("results/test_cases.jsonl")
    f3 = [c for c in cases if c["family"] == "F3"]
    f1a = [c for c in cases if c["family"] == "F1a"]
    ids, cats, prods = LE.catalog_index(args.catalog)
    agent = Agent(args.catalog)
    catalog = (ids, cats, prods)

    for name in args.policies:
        key = next(k for k in POLICIES if k.startswith(name))
        policy = POLICIES[key]
        start = time.time()
        payload = {
            "policy": key, "spec": policy,
            "f3": [run_case(agent, c, catalog, policy) for c in f3],
            "f1a": [run_case(agent, c, catalog, policy) for c in f1a],
            "public": run_public(agent, catalog, policy),
        }
        path = OUT / f"policy_{key}.json"
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        print(f"{key:14s} {len(payload['f3'])} F3 + {len(payload['f1a'])} F1a + 200 public "
              f"-> {path}  ({time.time()-start:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
