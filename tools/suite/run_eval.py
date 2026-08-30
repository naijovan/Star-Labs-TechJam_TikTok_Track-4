"""Stage 3 runner: preflight, smoke, full evaluation.

    PYTHONHASHSEED=0 python3 -m tools.suite.run_eval --preflight
    PYTHONHASHSEED=0 python3 -m tools.suite.run_eval --smoke
    PYTHONHASHSEED=0 python3 -m tools.suite.run_eval --full

Results land in results/runs/<agent_sha>/ and an existing run is never
overwritten silently.  The production agent and the evaluator are imported
unmodified; every counterfactual lives in tools/suite/harness.py.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import evaluator.local_evaluator as LE
from submission.agent import Agent

from tools.suite import SUITE_VERSION
from tools.suite.harness import (
    FROZEN_SUITE_SHA256, agent_source_sha256, config_snapshot, environment,
    evaluate_case, file_sha256, git_sha, load_cases,
)

CASES = "results/test_cases.jsonl"
CATALOG = "data/catalog.jsonl"
DATASET = "data/public_set.jsonl"
# Was 0.97870, recorded against the remediation2_pathfix agent. The canonicalizer
# and scheduling work since then took MTTC 2.065 -> 2.000, so the clean public score
# is now exactly 0.50 + 0.30 + 0.20*0.9. Pass --expect-baseline to gate on another.
PUBLIC_BASELINE = 0.98000
BASELINE_TOLERANCE = 1e-5


def preflight(cases_path=CASES, catalog_path=CATALOG, expect_baseline=PUBLIC_BASELINE) -> dict:
    print("=" * 74)
    print("STAGE 3A - PREFLIGHT")
    print("=" * 74)
    ok = True

    actual = file_sha256(cases_path)
    match = actual == FROZEN_SUITE_SHA256
    ok &= match
    print(f"\n[1] frozen suite SHA256")
    print(f"    expected {FROZEN_SUITE_SHA256}")
    print(f"    actual   {actual}   {'MATCH' if match else 'MISMATCH -- ABORT'}")
    if not match:
        return {"ok": False, "reason": "suite hash mismatch"}

    cases = load_cases(cases_path)
    manifest = {
        "suite_version": SUITE_VERSION,
        "test_cases_path": cases_path,
        "test_cases_sha256": actual,
        "n_test_cases": len(cases),
        "n_sessions_declared": sum(c["n_sessions"] for c in cases),
        "agent_git_sha": git_sha(),
        "agent_source_sha256": agent_source_sha256(),
        "catalog_sha256": file_sha256(catalog_path),
        "public_set_sha256": file_sha256(DATASET),
        "config": config_snapshot(),
        "environment": environment(),
    }
    print(f"\n[2] run identity")
    for key in ("suite_version", "n_test_cases", "n_sessions_declared",
                "agent_git_sha", "agent_source_sha256", "catalog_sha256"):
        print(f"    {key:24s} {manifest[key]}")
    for key, value in manifest["environment"].items():
        print(f"    {key:24s} {value}")
    print(f"    config                   {json.dumps(manifest['config'], sort_keys=True)}")

    print(f"\n[3] clean public baseline (gate: {expect_baseline:.5f})")
    build_start = time.time()
    agent = Agent(catalog_path)
    build_seconds = time.time() - build_start
    ids, cats, prods = LE.catalog_index(catalog_path)
    samples = LE.load_jsonl(DATASET)
    result = LE.evaluate(agent, samples, ids, cats, prods)
    score = result["recommended_technical_score"]
    baseline_ok = abs(score - expect_baseline) < BASELINE_TOLERANCE
    ok &= baseline_ok
    print(f"    measured {score:.5f}  hit {result['hit_rate_at_10']:.3f} "
          f"mrr {result['mrr']:.4f} mttc {result['mttc']:.3f}   "
          f"{'OK' if baseline_ok else 'DRIFT -- ABORT'}")
    print(f"    agent build {build_seconds:.1f}s")
    manifest["public_baseline"] = {"score": score, "hit_rate_at_10": result["hit_rate_at_10"],
                                   "mrr": result["mrr"], "mttc": result["mttc"],
                                   "build_seconds": round(build_seconds, 2)}
    manifest["ok"] = bool(ok)
    return {"ok": ok, "manifest": manifest, "cases": cases,
            "agent": agent, "catalog": (ids, cats, prods)}


def smoke(state: dict) -> bool:
    print("\n" + "=" * 74)
    print("STAGE 3A - SMOKE EVALUATION")
    print("=" * 74)
    cases, agent, catalog = state["cases"], state["agent"], state["catalog"]
    by_family = collections.defaultdict(list)
    for case in cases:
        by_family[case["family"]].append(case)

    picked = []
    for family in ("F1a", "F2A", "F2B", "F3", "F4", "F5", "F6", "F7", "F8"):
        picked.extend(by_family[family][:2])

    frozen = {c["test_id"]: json.dumps(c, sort_keys=True) for c in picked}
    rows, start = [], time.time()
    for case in picked:
        rows.extend(evaluate_case(agent, case, catalog))
    elapsed = time.time() - start

    checks = []
    ids_in = {c["test_id"] for c in picked}
    ids_out = {r["test_id"] for r in rows}
    checks.append(("every test_id joins", ids_in == ids_out))

    f3 = [c for c in picked if c["family"] == "F3"]
    f3_rows = [r for r in rows if r["family"] == "F3"]
    checks.append(("F3 paired arms join",
                   all({r["arm"] for r in f3_rows if r["test_id"] == c["test_id"]}
                       == {"as_ships", "turn1_clipped"} for c in f3)))
    checks.append(("F3 paired delta present",
                   all("paired_delta" in r for r in f3_rows if r["arm"] == "turn1_clipped")))

    expected_sessions = sum(c["n_sessions"] for c in picked)
    checks.append((f"session count expands ({len(rows)} == {expected_sessions})",
                   len(rows) == expected_sessions))

    maths_ok = True
    for row in rows:
        turn = row["first_hit_turn"] if row["first_hit_turn"] is not None else 11
        expected = (0.50 * (1.0 if row["hit"] else 0.0) + 0.30 * row["reciprocal_rank"]
                    + 0.20 * max(0.0, min(1.0, (11.0 - turn) / 10.0)))
        if abs(expected - row["technical_score_contribution"]) > 1e-12:
            maths_ok = False
        if row["hit"] and abs(row["reciprocal_rank"] - 1.0 / row["best_rank"]) > 1e-12:
            maths_ok = False
    checks.append(("scores reproduce evaluator mathematics", maths_ok))

    stages = {r["failure_stage"] for r in rows}
    checks.append((f"failure_stage populated {sorted(stages)}",
                   bool(stages) and all(s for s in stages)))
    checks.append(("turn logs captured", all(r["turns"] for r in rows)))

    mutated = [c["test_id"] for c in picked
               if json.dumps(c, sort_keys=True) != frozen[c["test_id"]]]
    checks.append(("no test definition mutated", not mutated))

    f4_rows = [r for r in rows if r["family"] == "F4"]
    checks.append(("F4 config arms all diagnostic",
                   all(r["diagnostic_arm"] for r in f4_rows)))
    checks.append(("F3 clip arm marked diagnostic",
                   all(r["diagnostic_arm"] for r in f3_rows if r["arm"] == "turn1_clipped")))

    print(f"\nran {len(picked)} cases -> {len(rows)} sessions in {elapsed:.1f}s\n")
    for label, passed in checks:
        print(f"    [{'PASS' if passed else 'FAIL'}] {label}")
    return all(passed for _, passed in checks)


def full(state: dict, out_dir: Path, overwrite: bool = False,
         resume: bool = False, budget_seconds: float = 0.0) -> Path:
    """Evaluate every case.  `resume` appends, skipping test_ids already present,
    so a long run can be completed across several bounded invocations without
    ever regenerating or mutating the frozen suite."""
    cases, agent, catalog = state["cases"], state["agent"], state["catalog"]
    results_path = out_dir / "test_results.jsonl"
    manifest_path = out_dir / "run_manifest.json"
    if results_path.exists() and not (overwrite or resume):
        raise FileExistsError(f"{results_path} exists. Refusing to overwrite a run silently.")
    out_dir.mkdir(parents=True, exist_ok=True)

    done: set = set()
    if resume and results_path.exists():
        with results_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done.add(json.loads(line)["test_id"])
        cases = [c for c in cases if c["test_id"] not in done]
        print(f"\nRESUME: {len(done)} test_ids already complete, {len(cases)} remaining")

    print("\n" + "=" * 74)
    print(f"STAGE 3B - FULL EVALUATION -> {out_dir}")
    print("=" * 74)
    start, written = time.time(), 0
    mode = "a" if (resume and results_path.exists()) else "w"
    with results_path.open(mode, encoding="utf-8") as handle:
        for index, case in enumerate(cases, 1):
            for row in evaluate_case(agent, case, catalog):
                row["agent_git_sha"] = state["manifest"]["agent_git_sha"]
                row["agent_source_sha256"] = state["manifest"]["agent_source_sha256"]
                row["test_cases_sha256"] = state["manifest"]["test_cases_sha256"]
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                written += 1
            if index % 500 == 0:
                handle.flush()
                rate = index / (time.time() - start)
                print(f"    {index}/{len(cases)} cases  {written} sessions  "
                      f"{rate:.1f} cases/s  eta {(len(cases)-index)/rate/60:.1f} min", flush=True)
            if budget_seconds and (time.time() - start) > budget_seconds:
                handle.flush()
                print(f"    time budget reached at {index}/{len(cases)} cases -- "
                      f"rerun with --resume", flush=True)
                break

    elapsed = time.time() - start
    total_rows = sum(1 for line in results_path.open(encoding="utf-8") if line.strip())
    complete = total_rows >= state["manifest"]["n_sessions_declared"]
    if not complete:
        print(f"\nPARTIAL: {total_rows}/{state['manifest']['n_sessions_declared']} sessions "
              f"written. Rerun with --resume to continue. Manifest NOT finalised.")
        return results_path
    written = total_rows
    manifest = dict(state["manifest"])
    manifest.update({
        "run_completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(elapsed, 1),
        "n_result_rows": written,
        "results_sha256": file_sha256(results_path),
        "reporting_rules": [
            "The 8,520-session suite is an ADVERSARIAL EXPERIMENTAL SUITE, not a "
            "representative private-evaluator sample.",
            "Never report one unweighted overall 8,520 score as a private-score estimate.",
            "F1a schemes B and C are the private surrogate; report separately, never averaged.",
            "T3 (stress_bound) is a bound, never an estimate.",
            "F7 is OOD; F1g is sentinel; both are excluded from headline figures.",
            "F3 turn1_clipped and all F4 config arms are DIAGNOSTIC and never enter the "
            "shipped-agent headline score.",
        ],
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {written} rows in {elapsed/60:.1f} min")
    print(f"  {results_path}  sha256 {manifest['results_sha256']}")
    print(f"  {manifest_path}")
    return results_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 3 evaluation")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--cases", default=CASES)
    parser.add_argument("--catalog", default=CATALOG)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--budget-seconds", type=float, default=0.0)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--expect-baseline", type=float, default=PUBLIC_BASELINE,
                        help="clean-public gate; pass the new value after an accepted "
                             "remediation so drift is still caught")
    parser.add_argument("--run-id", default=None,
                        help="output dir name; defaults to the agent git sha. Use this "
                             "when the tree is dirty so runs are not conflated.")
    args = parser.parse_args()

    state = preflight(args.cases, args.catalog, args.expect_baseline)
    if not state["ok"]:
        print("\nPREFLIGHT FAILED -- aborting.")
        return 1
    print("\nPREFLIGHT OK")

    if (args.smoke or args.full) and not args.skip_smoke:
        if not smoke(state):
            print("\nSMOKE FAILED -- aborting.")
            return 1
        print("\nSMOKE OK")

    if args.full:
        out_dir = Path("results/runs") / (args.run_id or state["manifest"]["agent_git_sha"][:12])
        full(state, out_dir, overwrite=args.overwrite, resume=args.resume,
             budget_seconds=args.budget_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
