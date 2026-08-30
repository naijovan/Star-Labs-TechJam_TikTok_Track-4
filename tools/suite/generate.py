"""CLI: generate results/test_cases.jsonl.

    # validation batch (small, every family/cell type represented)
    PYTHONHASHSEED=0 python3 -m tools.suite.generate --validation

    # the full approved 8,520-session suite
    PYTHONHASHSEED=0 python3 -m tools.suite.generate --full

Refuses to overwrite an existing file unless --overwrite is passed.
Generation is agent-independent: nothing here imports submission/.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from tools.suite import SUITE_VERSION
from tools.suite.sessions import CaseBuilder, SuiteGenerator, write_cases
from tools.suite.strata import CatalogFeatures


def summarise(cases) -> None:
    by_family = collections.Counter(c.family for c in cases)
    sessions = collections.Counter()
    for c in cases:
        sessions[c.family] += c.n_sessions
    print(f"\n{'family':<8}{'cases':>8}{'sessions':>10}  cells")
    print("-" * 46)
    for family in sorted(by_family):
        cells = len({c.cell_id for c in cases if c.family == family})
        print(f"{family:<8}{by_family[family]:>8}{sessions[family]:>10}  {cells}")
    print("-" * 46)
    print(f"{'TOTAL':<8}{len(cases):>8}{sum(sessions.values()):>10}")
    flags = collections.Counter()
    for c in cases:
        for name in ("sentinel", "stress_bound", "ood", "excluded_from_headline"):
            if getattr(c, name):
                flags[name] += 1
    print("flags: " + ", ".join(f"{k}={v}" for k, v in sorted(flags.items())))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate test_cases.jsonl")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validation", action="store_true", help="small representative batch")
    mode.add_argument("--full", action="store_true", help="the full approved suite")
    parser.add_argument("--out", default=None)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--master-seed", type=int, default=20260830)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    features = CatalogFeatures(args.catalog)
    builder = CaseBuilder(features, args.catalog, args.dataset)
    generator = SuiteGenerator(features, builder, master_seed=args.master_seed)

    sizes = SuiteGenerator.VALIDATION if args.validation else SuiteGenerator.FULL
    out = args.out or ("results/validation_cases.jsonl" if args.validation
                       else "results/test_cases.jsonl")

    cases = generator.generate(sizes)
    path = write_cases(cases, out, overwrite=args.overwrite)

    print(f"suite_version {SUITE_VERSION}  master_seed {args.master_seed}")
    print(f"wrote {len(cases)} cases -> {path}")
    summarise(cases)
    return 0


if __name__ == "__main__":
    sys.exit(main())
