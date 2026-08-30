"""Harness-only measurement of JACCARD_MIN across independent damage generators.

    PYTHONHASHSEED=0 python3 -m tools.suite.jaccard_experiment --probe
    PYTHONHASHSEED=0 python3 -m tools.suite.jaccard_experiment --generators D1 --values 0.40 0.50

NOTHING in submission/ is modified.  JACCARD_MIN is set via `config_override`,
which restores the shipped value in a `finally`.

Design notes that matter for the conclusion:

* **Paired.**  The same targets, scenarios, seeds and damage are used at every
  JACCARD_MIN value within a generator, so the per-session delta cancels target
  variance entirely.  The aggregate TechnicalScore is exactly the mean of the
  per-session contributions (efficiency is linear in MTTC and never clips), so a
  bootstrap over per-session deltas is a valid CI for the aggregate delta.
* **Three independent generators.**  D1 and D2 already exist and D2 is where the
  original 15-session signal appeared, so D3 is NEW and was not consulted while
  choosing any threshold.
* **Sensitivity is recorded.**  `_nearest_bucket` only runs when the leaked
  category missed its bucket, so most sessions are structurally unable to
  respond to the threshold.  Counting those separately is what turns "no
  difference" into "no difference among the sessions that could differ".
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import json
import math
import random
import re
import statistics
import time
from pathlib import Path

import evaluator.local_evaluator as LE
import submission.config as CONFIG
from submission.agent import Agent

from tools.suite.harness import _Patcher, config_override, load_cases

OUT = Path("results/jaccard")
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

VALUES = (0.40, 0.50, 0.60, 0.70, 0.75, 0.80)
SHIPPED = 0.50


# ----------------------------------------------------------------- generators

def damage_D1(text: str) -> str:
    """Word reversal -- the generator tools/verify_agent.py already used."""
    words = text.split()
    return " ".join(reversed(words)) if len(words) > 1 else (text[:-1] if len(text) > 4 else text)


def damage_D2(text: str) -> str:
    """Character deletion, seeded per string."""
    import hashlib
    rng = random.Random(hashlib.md5(text.encode()).hexdigest())
    out = "".join(c for c in text if rng.random() > 0.18).strip()
    return out if len(out) >= 3 else text[: max(3, len(text) - 2)]


def damage_D3(text: str) -> str:
    """NEW, independent of D1 and D2: casual-typing morphology.

    Lowercases, drops punctuation and '&', and toggles a trailing 's' on every
    other token.  Word ORDER and word IDENTITY are preserved, so the intended
    category is unchanged -- what breaks is exact string equality and part of
    the token set, which is precisely the regime the Jaccard repair exists for.
    Deterministic: no RNG at all.
    """
    tokens = TOKEN_RE.findall(text.lower())
    out = []
    for index, token in enumerate(tokens):
        if index % 2 == 0 and len(token) > 3:
            token = token[:-1] if token.endswith("s") else token + "s"
        out.append(token)
    return " ".join(out) if out else text


GENERATORS = {"D1": damage_D1, "D2": damage_D2, "D3": damage_D3}


@contextlib.contextmanager
def damaged_category(generator):
    """Damage only the leaked category in the turn-1 opening."""
    original = LE.initial_message

    def initial_message(sample, category, disclosed):
        return original(sample, generator(category), disclosed)

    LE.initial_message = initial_message
    try:
        yield
    finally:
        LE.initial_message = original


# ----------------------------------------------------------------- execution

def _sample(case):
    return {"sample_id": case["sample_id"], "scenario_type": case["scenario_type"],
            "category_bucket": "clothing", "difficulty_bucket": "medium",
            "user_profile": case["user_profile"],
            "ground_truth": {"parent_asin": case["target_parent_asin"]}}


def run_case(agent, case, catalog, generator, jaccard, no_override=False):
    ids, cats, prods = catalog
    probe = {"nearest_calls": 0, "best_jaccard": None}
    original_nearest = agent._nearest_bucket

    def nearest(q):
        probe["nearest_calls"] += 1
        qs = set(TOKEN_RE.findall(q.lower()))
        best = 0.0
        for word in qs:
            for key in agent.cat_tok.get(word, ()):  # same candidate set the agent uses
                ks = agent.cat_words[key]
                best = max(best, len(qs & ks) / len(qs | ks))
        probe["best_jaccard"] = max(probe["best_jaccard"] or 0.0, best)
        return original_nearest(q)

    agent._nearest_bucket = nearest
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(_Patcher(case))
            stack.enter_context(damaged_category(GENERATORS[generator]))
            if not no_override:
                stack.enter_context(config_override({"JACCARD_MIN": jaccard}))
            result = LE.evaluate(agent, [_sample(case)], ids, cats, prods)
    finally:
        agent._nearest_bucket = original_nearest

    session = result["sessions"][0]
    turn = session["first_hit_turn"] if session["first_hit_turn"] is not None else 11
    return {
        "test_id": case["test_id"], "scenario_type": case["scenario_type"],
        "scheme": case.get("scheme"), "hit": bool(session["hit"]),
        "first_hit_turn": session["first_hit_turn"], "best_rank": session["best_rank"],
        "rr": session["reciprocal_rank"], "mttc": float(turn),
        "nearest_calls": probe["nearest_calls"], "best_jaccard": probe["best_jaccard"],
        "contribution": (0.5 * (1.0 if session["hit"] else 0.0) + 0.3 * session["reciprocal_rank"]
                         + 0.2 * max(0.0, min(1.0, (11.0 - turn) / 10.0))),
    }


def aggregate(rows):
    n = len(rows)
    if not n:
        return {"n": 0, "score": 0.0, "hit": 0.0, "mrr": 0.0, "mttc": 0.0}
    hit = sum(r["hit"] for r in rows) / n
    mrr = sum(r["rr"] for r in rows) / n
    mttc = sum(r["mttc"] for r in rows) / n
    return {"n": n, "hit": hit, "mrr": mrr, "mttc": mttc,
            "score": 0.5 * hit + 0.3 * mrr + 0.2 * max(0.0, min(1.0, (11.0 - mttc) / 10.0))}


def paired_ci(deltas, iterations=10000, seed=7):
    """Bootstrap CI for the MEAN paired delta, which equals the aggregate
    TechnicalScore delta because the score is linear in the per-session terms."""
    if not deltas:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(iterations):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (means[int(0.025 * iterations)], means[int(0.975 * iterations)])


def main() -> int:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--generators", nargs="+", default=["D1", "D2", "D3"])
    parser.add_argument("--values", nargs="+", type=float, default=list(VALUES))
    parser.add_argument("--cases", type=int, default=900)
    parser.add_argument("--probe", action="store_true",
                        help="small run to size the experiment: how many sessions can respond?")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--no-override", action="store_true",
                        help="use the SHIPPED config value instead of overriding it -- "
                             "this is what verifies the production change end to end")
    parser.add_argument("--smart", action="store_true",
                        help="EXACT optimisation: a session can only differ across values "
                             "if its best Jaccard lies in [min(values), max(values)); outside "
                             "that band every value makes the same accept/reject decision, so "
                             "the baseline row is reused verbatim.")
    args = parser.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    cases = [c for c in load_cases("results/test_cases.jsonl") if c["family"] == "F1a"]
    n = 120 if args.probe else args.cases
    cases = cases[:n]
    catalog = LE.catalog_index(args.catalog)
    agent = Agent(args.catalog)

    if args.smart:
        lo, hi = min(args.values), max(args.values)
        for generator in args.generators:
            base_path = OUT / f"{generator}_{SHIPPED:.2f}.json"
            if base_path.exists():
                base = json.loads(base_path.read_text())["rows"]
            else:
                start = time.time()
                base = [run_case(agent, c, catalog, generator, SHIPPED) for c in cases]
                base_path.write_text(json.dumps({"generator": generator,
                                                 "jaccard_min": SHIPPED, "rows": base}) + "\n")
                print(f"{generator} {SHIPPED:.2f}  baseline n={len(base)} "
                      f"({time.time()-start:.0f}s)", flush=True)
            by_id = {c["test_id"]: c for c in cases}
            band = [r["test_id"] for r in base
                    if r["best_jaccard"] is not None and lo <= r["best_jaccard"] < hi]
            print(f"{generator}: {len(band)}/{len(base)} sessions in the decision band "
                  f"[{lo:.2f},{hi:.2f}) -- only these can differ", flush=True)
            for value in args.values:
                if abs(value - SHIPPED) < 1e-9:
                    continue
                path = OUT / f"{generator}_{value:.2f}.json"
                if path.exists():
                    continue
                start = time.time()
                redone = {t: run_case(agent, by_id[t], catalog, generator, value) for t in band}
                rows = [redone.get(r["test_id"], r) for r in base]
                path.write_text(json.dumps({"generator": generator, "jaccard_min": value,
                                            "rows": rows, "recomputed": len(band)}) + "\n")
                agg = aggregate(rows)
                print(f"{generator} {value:.2f}  n={agg['n']:4d} score {agg['score']:.5f} "
                      f"hit {agg['hit']:.3f} mrr {agg['mrr']:.4f} mttc {agg['mttc']:.2f}  "
                      f"recomputed {len(band)}  ({time.time()-start:.0f}s)", flush=True)
        return 0

    for generator in args.generators:
        for value in args.values:
            path = OUT / f"{generator}_{value:.2f}.json"
            if path.exists():
                print(f"{generator} {value:.2f}  cached", flush=True)
                continue
            start = time.time()
            rows = [run_case(agent, c, catalog, generator, value, args.no_override)
                    for c in cases]
            path.write_text(json.dumps({"generator": generator, "jaccard_min": value,
                                        "rows": rows}) + "\n", encoding="utf-8")
            agg = aggregate(rows)
            sensitive = sum(1 for r in rows if r["nearest_calls"] > 0)
            print(f"{generator} {value:.2f}  n={agg['n']:4d} score {agg['score']:.5f} "
                  f"hit {agg['hit']:.3f} mrr {agg['mrr']:.4f} mttc {agg['mttc']:.2f}  "
                  f"repair-eligible {sensitive}  ({time.time()-start:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
