"""Stage 3 harness tests.

Regression guard for the turn-1 logging bug: the spy that records emitted cards
must wrap the OUTERMOST respond, so `turns` reflects what the evaluator actually
received rather than the page before a counterfactual clip was applied.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import evaluator.local_evaluator as LE
from submission.agent import Agent

from tools.suite.harness import evaluate_case, load_cases, run_session
from tools.suite.policy_ablation import POLICIES, run_case

CATALOG = Path("data/catalog.jsonl")
CASES = Path("results/test_cases.jsonl")

_AGENT = None
_CATALOG = None
_F3 = None


def setUpModule() -> None:
    global _AGENT, _CATALOG, _F3
    if not (CATALOG.exists() and CASES.exists()):
        raise unittest.SkipTest("catalogue or frozen suite missing")
    _CATALOG = LE.catalog_index(str(CATALOG))
    _AGENT = Agent(str(CATALOG))
    cases = load_cases(CASES)
    # Large-pool buying cases: the regime where turn 1 emits a full page.
    _F3 = [c for c in cases if c["cell_id"] == "F3/pool_gt200"][:6]


class TestTurn1LoggingReflectsEmittedCards(unittest.TestCase):
    """The logged page must equal what the evaluator received.

    NOTE. Since Remediation #3 (TURN1_PAGE=1) the PRODUCTION agent already emits
    at most one card on turn 1, so the harness clip is a no-op against it. These
    tests therefore verify the logging invariant directly -- logged page ==
    response page -- rather than by contrasting a clipped and unclipped arm,
    which would silently pass for the wrong reason.
    """

    def test_logged_page_equals_response_page(self):
        """The spy must record the OUTERMOST response, post-policy."""
        for case in _F3:
            for policy in (POLICIES["A_current"], POLICIES["B_always_1"]):
                row = run_case(_AGENT, case, _CATALOG, policy)
                self.assertLessEqual(row["turn1_cards"], 1,
                                     f"{case['test_id']}: logged {row['turn1_cards']} cards")

    def test_clip_arm_is_now_a_noop_against_production(self):
        """Production implements policy B, so the diagnostic clip changes nothing.
        If this ever fails, either TURN1_PAGE moved or the clip regressed."""
        for case in _F3:
            plain = run_session(_AGENT, case, _CATALOG)
            clipped = run_session(_AGENT, case, _CATALOG, clip=True)
            for a, b in zip(plain.turns, clipped.turns):
                self.assertEqual(a["recommendations"], b["recommendations"],
                                 f"{case['test_id']} turn {a['turn']} differs")

    def test_harness_clip_still_works_on_a_wide_page(self):
        """Guards the harness itself: with TURN1_PAGE raised, the clip must bite
        and the log must show the clipped page, not the pre-clip one."""
        import submission.config as CONFIG
        from tools.suite.harness import config_override
        case = _F3[0]
        with config_override({"TURN1_PAGE": 10}):
            wide = run_session(_AGENT, case, _CATALOG)
            clipped = run_session(_AGENT, case, _CATALOG, clip=True)
        wide_turn1 = next(t for t in wide.turns if t["turn"] == 1)
        clip_turn1 = next(t for t in clipped.turns if t["turn"] == 1)
        self.assertGreater(len(wide_turn1["recommendations"]), 1,
                           "TURN1_PAGE=10 should produce a wide page")
        self.assertEqual(len(clip_turn1["recommendations"]), 1,
                         "clip arm must log exactly one card")

    def test_evaluate_case_emits_both_arms(self):
        rows = evaluate_case(_AGENT, _F3[0], _CATALOG)
        by_arm = {r["arm"]: r for r in rows}
        self.assertEqual(set(by_arm), {"as_ships", "turn1_clipped"})
        self.assertTrue(by_arm["turn1_clipped"]["diagnostic_arm"])
        self.assertFalse(by_arm["as_ships"]["diagnostic_arm"])

    def test_spy_does_not_leak_between_runs(self):
        """agent.respond must be restored after every run."""
        before = _AGENT.respond
        run_session(_AGENT, _F3[0], _CATALOG, clip=True)
        run_case(_AGENT, _F3[0], _CATALOG, POLICIES["B_always_1"])
        self.assertIs(_AGENT.respond, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
