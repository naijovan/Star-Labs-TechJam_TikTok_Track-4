"""Measure what the leaked leaf category is worth ON ITS OWN.

No BM25, no constraints, no ranking beyond one optional prior: parse the
category out of turn 1, then page through that bucket ten at a time.

    python3 -m tools.floor_probe
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

from evaluator.local_evaluator import catalog_index, coarse_category, evaluate

CATEGORY_RE = re.compile(r"I'm looking for (.+?)[.,]")


class CategoryOnlyAgent:
    """Category bucket + blind paging. Deliberately has no retrieval at all."""

    def __init__(self, catalog_path: str, order: str = "arbitrary") -> None:
        self.buckets: dict[str, list[str]] = defaultdict(list)
        products = []
        with open(catalog_path, encoding="utf-8") as handle:
            for line in handle:
                products.append(json.loads(line))
        if order == "popular":
            products.sort(key=lambda p: (-(p.get("rating_number") or 0),
                                         -(p.get("average_rating") or 0)))
        for product in products:
            self.buckets[coarse_category(product.get("categories") or [])].append(
                str(product["parent_asin"])
            )
        self.bucket: list[str] = []
        self.cursor = 0

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.bucket = []
        self.cursor = 0
        self.blocked = False        # intent_override: cannot convert until it fires

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        match = CATEGORY_RE.match(user_message)
        if match:                                   # only turn 1 carries it
            self.bucket = self.buckets.get(match.group(1), [])
            self.cursor = 0
            # turn-1 shape identifies the scenario with certainty:
            #   "...but I'm still exploring."   -> browsing / boundary
            #   "... A key requirement is: X."  -> buying
            #   anything else                   -> intent_override
            self.blocked = not (
                user_message.endswith("but I'm still exploring.")
                or ". A key requirement is: " in user_message
            )
        if user_message.startswith("Actually, ignore my earlier preference"):
            self.blocked = False                    # the gate just opened
            self.cursor = 0                         # restart the sweep from the top
        page = self.bucket[self.cursor:self.cursor + top_k]
        if not self.blocked:                        # don't burn pages on dead turns
            self.cursor += top_k
        return {
            "message": "Anything else that matters?",
            "ask_attribute": "other",
            "recommendations": [{"parent_asin": a} for a in page],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


def main() -> None:
    samples = [json.loads(l) for l in open("data/public_set.jsonl", encoding="utf-8") if l.strip()]
    ids, cats, prods = catalog_index("data/catalog.jsonl")
    for order in ("arbitrary", "popular"):
        result = evaluate(CategoryOnlyAgent("data/catalog.jsonl", order), samples, ids, cats, prods)
        print(f"\n=== category bucket, paged 10/turn, order={order} ===")
        print(f"  hit_rate@10 {result['hit_rate_at_10']:.4f}   mrr {result['mrr']:.4f}"
              f"   mttc {result['mttc']:.2f}   eff {result['efficiency']:.3f}"
              f"   SCORE {result['recommended_technical_score']:.5f}")
        for name, m in result["scenario_metrics"].items():
            print(f"    {name:<16} n={m['sample_count']:<4} hit={m['hit_rate_at_10']:.3f} mrr={m['mrr']:.3f}")


if __name__ == "__main__":
    main()
