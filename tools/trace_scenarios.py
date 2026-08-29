"""Print the exact turn-by-turn customer script for each scenario type.

Runs against a one-row synthetic catalog, so it needs no data/catalog.jsonl.
Use it to confirm how many asks actually pay out before tuning anything.

    python3 -m tools.trace_scenarios
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate

# A stand-in intent card. Two hard constraints, two soft preferences: the
# maximum the simulator ever holds.
CARD = {
    "hard_constraints": ["cotton", "color: blue"],
    "soft_preferences": ["Machine wash cold", "Imported lightweight fabric"],
}
PAYOUT_PREFIX = "For that, what matters is:"


class Probe:
    """Asks 'other' every turn and never guesses, so all 10 turns run."""

    def __init__(self) -> None:
        self.log: list[tuple[int, str]] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.log = []

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        self.log.append((turn, user_message))
        return {"message": "", "ask_attribute": "other", "recommendations": []}


def _synthetic_catalog(directory: Path) -> Path:
    path = directory / "catalog.jsonl"
    path.write_text(
        json.dumps({
            "parent_asin": "A", "title": "placeholder", "features": [], "details": {},
            "description": [], "categories": ["Clothing", "Tops"], "store": "store",
            "average_rating": 4.0, "rating_number": 9, "price": 10.0,
        }) + "\n",
        encoding="utf-8",
    )
    return path


def _sample(scenario: str, override_turn: int | None) -> dict:
    behavior: dict = {"scenario_type": scenario}
    if scenario == "intent_override":
        new_value = CARD["hard_constraints"][0]
        behavior["override"] = {
            "turn": override_turn,
            "old_value": CARD["soft_preferences"][-1],
            "new_value": new_value,
            "message": f"Actually, ignore my earlier preference. What I need is: {new_value}.",
        }
    return {
        "sample_id": "trace_0001",
        "scenario_type": scenario,
        "user_profile": {"summary": "synthetic"},
        "ground_truth": {"parent_asin": "A"},
        "intent_card": dict(CARD),
        "behavior": behavior,
    }


def main() -> None:
    cases = [
        ("browsing", None), ("buying", None), ("boundary", None),
        ("intent_override", 3), ("intent_override", 4),
    ]
    with tempfile.TemporaryDirectory() as directory:
        catalog_ids, categories, products = catalog_index(_synthetic_catalog(Path(directory)))
        for scenario, override_turn in cases:
            agent = Probe()
            evaluate(agent, [_sample(scenario, override_turn)], catalog_ids, categories, products)
            label = scenario + (f"  (override fires on turn {override_turn})" if override_turn else "")
            print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
            payouts = 0
            for turn, message in agent.log:
                paid = message.startswith(PAYOUT_PREFIX)
                payouts += paid
                print(f"  turn {turn:>2}: {message[:90]}{'   <== PAYOUT' if paid else ''}")
            print(f"  --> productive asks: {payouts}")


if __name__ == "__main__":
    main()
