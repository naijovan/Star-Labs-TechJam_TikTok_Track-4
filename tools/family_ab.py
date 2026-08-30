"""Score selected suite families with the CURRENT working tree, for A/B iteration.

    PYTHONHASHSEED=0 PYTHONPATH=. python tools/family_ab.py --families F2A F2B F6
    PYTHONHASHSEED=0 PYTHONPATH=. python tools/family_ab.py --families F2A --baseline \
        results/runs/base_8c3ef91/test_results.jsonl

`tools/suite/run_eval.py --full` remains the authority: it gates on the clean
public baseline, runs every family, and writes a manifest. This is a development
loop for one family at a time, so a paraphrase change can be measured in a minute
instead of the better part of an hour. It reuses the suite's own frozen cases and
the suite's own `evaluate_case`, so a number here means the same thing it does there.

Headline exclusions are honoured: sentinel, stress_bound, ood, excluded_from_headline
and every diagnostic arm are dropped unless --include-excluded is passed.
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import evaluator.local_evaluator as LE
from submission.agent import Agent

from tools.suite.harness import evaluate_case, load_cases

CASES = "results/test_cases.jsonl"
CATALOG = "data/catalog.jsonl"


def headline(row: dict) -> bool:
    return not (row.get("sentinel") or row.get("stress_bound") or row.get("ood")
                or row.get("excluded_from_headline") or row.get("diagnostic_arm")
                or row.get("diagnostic_case"))


def aggregate(rows) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0, "hit": 0.0, "mrr": 0.0, "mttc": 0.0, "score": 0.0}
    hit = sum(r["hit"] for r in rows) / n
    mrr = sum(r["reciprocal_rank"] for r in rows) / n
    mttc = sum(r["mttc_contribution"] for r in rows) / n
    eff = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return {"n": n, "hit": hit, "mrr": mrr, "mttc": mttc,
            "score": 0.50 * hit + 0.30 * mrr + 0.20 * eff}


def fmt(label: str, agg: dict, ref: dict | None = None) -> str:
    line = (f"{label:<28}{agg['n']:>6}  hit {agg['hit']:.4f}  mrr {agg['mrr']:.4f}  "
            f"mttc {agg['mttc']:.3f}  score {agg['score']:.5f}")
    if ref and ref.get("n"):
        line += f"   delta {agg['score'] - ref['score']:+.5f}"
    return line


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", required=True)
    ap.add_argument("--cases", default=CASES)
    ap.add_argument("--catalog", default=CATALOG)
    ap.add_argument("--baseline", default=None,
                    help="a test_results.jsonl to diff against, restricted to the "
                         "test_ids actually run here")
    ap.add_argument("--limit", type=int, default=0, help="cases per family (0 = all)")
    ap.add_argument("--include-excluded", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    wanted = set(args.families)
    cases = [c for c in load_cases(args.cases) if c["family"] in wanted]
    if args.limit:
        seen: collections.Counter = collections.Counter()
        kept = []
        for case in cases:
            if seen[case["family"]] < args.limit:
                kept.append(case)
                seen[case["family"]] += 1
        cases = kept
    if not cases:
        print(f"no cases matched {sorted(wanted)}")
        return 1

    agent = Agent(args.catalog)
    catalog = LE.catalog_index(args.catalog)

    rows, start = [], time.time()
    for index, case in enumerate(cases, 1):
        rows.extend(evaluate_case(agent, case, catalog))
        if index % 250 == 0:
            rate = index / (time.time() - start)
            print(f"    {index}/{len(cases)} cases  {rate:.1f}/s  "
                  f"eta {(len(cases) - index) / rate / 60:.1f} min", flush=True)
    print(f"\nran {len(cases)} cases -> {len(rows)} sessions in {time.time() - start:.0f}s\n")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.out).open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    scored = rows if args.include_excluded else [r for r in rows if headline(r)]
    ref_rows = {}
    if args.baseline:
        ids = {(r["test_id"], r["arm"]) for r in scored}
        with Path(args.baseline).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if (row["test_id"], row["arm"]) in ids:
                    ref_rows.setdefault(row["family"], []).append(row)

    by_family = collections.defaultdict(list)
    for row in scored:
        by_family[row["family"]].append(row)
    for family in sorted(by_family):
        ref = aggregate(ref_rows[family]) if ref_rows.get(family) else None
        if ref:
            print(fmt(f"{family} baseline", ref))
        print(fmt(f"{family} current", aggregate(by_family[family]), ref))

    by_cell = collections.defaultdict(list)
    for row in scored:
        by_cell[(row["family"], row["cell_id"])].append(row)
    worst = sorted(by_cell.items(), key=lambda kv: aggregate(kv[1])["score"])[:12]
    print("\nweakest cells:")
    for (family, cell), cell_rows in worst:
        print("  " + fmt(cell, aggregate(cell_rows)))

    stages = collections.Counter(r["failure_stage"] for r in scored if not r["hit"])
    if stages:
        print("\nfailure stages on misses: " + ", ".join(
            f"{k}={v}" for k, v in stages.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
