"""Stage 1 verification: reproduce the TEST_MATRIX.md section 0.5 measurements.

    PYTHONHASHSEED=0 python3 -m tools.suite.verify_stage1            # diagnostics only
    PYTHONHASHSEED=0 python3 -m tools.suite.verify_stage1 --score    # + evaluator runs

`--score` runs the UNMODIFIED production agent through the UNMODIFIED evaluator.
Nothing is monkeypatched and no production file is touched.  The tiny session
builder below exists only so Stage 1 can be verified end to end; it is replaced
by `tools/suite/sessions.py` in Stage 2 and must not grow features here.
"""
from __future__ import annotations

import argparse
import statistics
import sys

from tools.suite.pool import (
    ACCEPTED_SCHEMES,
    CoarseJointScheme,
    FineJointScheme,
    NearestNeighbourScheme,
    T1Mixture,
)
from tools.suite.strata import CatalogFeatures, load_public_targets

CATALOG = "data/catalog.jsonl"
DATASET = "data/public_set.jsonl"


def _print_diagnostics_table(features, targets) -> dict:
    donors = features.donor_pool(exclude=targets, strict=False)
    print(f"Donor pool (public-unseen, has_features & nc==4 & avg>=3.5): {len(donors)}\n")
    header = f"{'scheme':<26}{'occ':>5}{'0-donor':>9}{'unmatched':>11}{'min/med/p90':>16}{'ESS/400':>9}{'leave-out':>11}  verdict"
    print(header)
    print("-" * len(header))
    out = {}
    for cls in (FineJointScheme, CoarseJointScheme, NearestNeighbourScheme):
        scheme = cls(features, targets, donors)
        d = scheme.diagnostics()
        lo = f"{d.leave_out_mean:.0f}/100" + ("" if d.leave_out_informative else "*")
        verdict = "ACCEPT" if d.accepted else "REJECT"
        print(
            f"{d.scheme:<26}{d.occupied_cells:>5}{d.zero_donor_cells:>9}"
            f"{d.unmatched_mass:>10.1%}{f'{d.donors_min}/{d.donors_median:g}/{d.donors_p90:g}':>16}"
            f"{d.ess:>9.1f}{lo:>11}  {verdict}"
        )
        for reason in d.reject_reasons:
            print(f"{'':<26}   -> rejected: {reason}")
        if not d.leave_out_informative:
            print(f"{'':<26}   *  {d.coverage_note}")
        out[cls.__name__] = d
    return out


def _print_core_tail(features, targets) -> None:
    core = [a for a in targets if features.is_strict(a)]
    tail = [a for a in targets if not features.is_strict(a)]
    print(f"\ncore {len(core)}/{len(targets)} ({len(core)/len(targets):.1%})   "
          f"tail {len(tail)}/{len(targets)} ({len(tail)/len(targets):.1%})")
    print(f"  tail failing has_price          : {sum(1 for a in tail if not features.has_price(a))}")
    print(f"  tail failing rating_number>=100 : {sum(1 for a in tail if features.rating_number(a) < 100)}")
    print(f"  tail failing has_features       : {sum(1 for a in tail if not features.has_features(a))}")
    print(f"  tail failing n_constraints==4   : {sum(1 for a in tail if features.n_constraints(a) != 4)}")
    print(f"  core median rating_number {statistics.median([features.rating_number(a) for a in core]):.0f}"
          f" | median bucket rank {statistics.median([features.bucket_rank(a) for a in core]):.1f}")
    print(f"  tail median rating_number {statistics.median([features.rating_number(a) for a in tail]):.0f}"
          f" | median bucket rank {statistics.median([features.bucket_rank(a) for a in tail]):.1f}")


def _score_draws(features, targets, seeds=(11, 22, 33), n=400) -> None:
    import evaluator.local_evaluator as LE
    from submission.agent import Agent

    samples = LE.load_jsonl(DATASET)
    ids, cats, prods = LE.catalog_index(CATALOG)
    agent = Agent(CATALOG)
    profiles = [s["user_profile"] for s in samples]
    scenarios = [s["scenario_type"] for s in samples]

    def build(asins, seed):
        import random

        rng = random.Random(seed)
        return [
            {
                "sample_id": f"stage1_{seed}_{i}",
                "scenario_type": scenarios[i % len(scenarios)],
                "category_bucket": "clothing",
                "difficulty_bucket": "medium",
                "user_profile": profiles[rng.randrange(len(profiles))],
                "ground_truth": {"parent_asin": a},
            }
            for i, a in enumerate(asins)
        ]

    def run(asins, seed):
        r = LE.evaluate(agent, build(asins, seed), ids, cats, prods)
        return (r["recommended_technical_score"], r["hit_rate_at_10"], r["mrr"], r["mttc"])

    print(f"\n{'draw':<40}{'score':>9}{'hit':>7}{'MRR':>8}{'MTTC':>7}")
    print("-" * 71)
    print(f"{'public 200 (reference, from MEASUREMENT_LOG)':<40}{0.97870:>9.5f}{1.000:>7.3f}{0.9933:>8.4f}{1.965:>7.2f}")
    for key, cls in ACCEPTED_SCHEMES.items():
        mixture = T1Mixture(features, targets, cls)
        scores = []
        for seed in seeds:
            draw = mixture.draw(n, seed)
            v = run(draw.asins, seed)
            scores.append(v[0])
            print(f"{f'T1 mixture {key}  seed {seed}':<40}{v[0]:>9.5f}{v[1]:>7.3f}{v[2]:>8.4f}{v[3]:>7.2f}")
        print(f"{f'  -> {key} mean':<40}{statistics.mean(scores):>9.5f}"
              f"   spread {max(scores)-min(scores):.5f}")
    print("\nNOTE: B and C are reported separately and MUST NOT be averaged.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1 verification")
    parser.add_argument("--score", action="store_true", help="also run the evaluator on T1 draws")
    parser.add_argument("--catalog", default=CATALOG)
    parser.add_argument("--dataset", default=DATASET)
    args = parser.parse_args()

    print("STAGE 1 VERIFICATION -- sampling foundation\n")
    features = CatalogFeatures(args.catalog)
    targets = load_public_targets(args.dataset)
    print(f"catalogue {len(features.asins)} products | {len(features.bucket_members)} buckets "
          f"| {len(targets)} public targets\n")

    diagnostics = _print_diagnostics_table(features, targets)
    _print_core_tail(features, targets)

    if args.score:
        _score_draws(features, targets)

    rejected = [d.scheme for d in diagnostics.values() if not d.accepted]
    accepted = [d.scheme for d in diagnostics.values() if d.accepted]
    print(f"\nACCEPTED: {', '.join(accepted)}")
    print(f"REJECTED: {', '.join(rejected) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
