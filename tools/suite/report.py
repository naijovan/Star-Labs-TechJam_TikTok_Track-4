"""Stage 3D reporting: aggregate raw results into the approved matrix.

Reporting rules enforced here, not left to the reader:
  * no unweighted "overall 8,520" score is ever computed;
  * F1a schemes B and C are reported separately and never averaged;
  * T3 stress bounds, F1g sentinels, F7 OOD and every diagnostic arm are
    excluded from headline figures by construction.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
from pathlib import Path
from typing import Dict, List, Sequence

PUBLIC_REFERENCE = 0.98000  # keep in step with run_eval.PUBLIC_BASELINE


def load(path: str | Path) -> List[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def aggregate(rows: Sequence[dict]) -> dict:
    if not rows:
        return {"n": 0, "hit": 0.0, "mrr": 0.0, "mttc": 0.0, "score": 0.0}
    n = len(rows)
    hit = sum(1 for r in rows if r["hit"]) / n
    mrr = sum(r["reciprocal_rank"] for r in rows) / n
    mttc = sum(r["mttc_contribution"] for r in rows) / n
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return {"n": n, "hit": hit, "mrr": mrr, "mttc": mttc,
            "score": 0.50 * hit + 0.30 * mrr + 0.20 * efficiency}


def bootstrap_ci(rows: Sequence[dict], iterations: int = 10000, seed: int = 20260830) -> tuple:
    """Percentile bootstrap over SESSIONS for the aggregate TechnicalScore."""
    if len(rows) < 2:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(rows)
    hits = [1.0 if r["hit"] else 0.0 for r in rows]
    rrs = [r["reciprocal_rank"] for r in rows]
    turns = [r["mttc_contribution"] for r in rows]
    scores = []
    for _ in range(iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        h = sum(hits[i] for i in idx) / n
        m = sum(rrs[i] for i in idx) / n
        t = sum(turns[i] for i in idx) / n
        scores.append(0.50 * h + 0.30 * m + 0.20 * max(0.0, min(1.0, (11.0 - t) / 10.0)))
    scores.sort()
    return (scores[int(0.025 * iterations)], scores[int(0.975 * iterations)])


def shipped(rows: Sequence[dict]) -> List[dict]:
    """Only the shipped-agent arm: no diagnostic arms, no config sweeps."""
    return [r for r in rows if r["arm"] == "as_ships" and not r["diagnostic_arm"]]


def fmt(agg: dict) -> str:
    return (f"n={agg['n']:5d}  score {agg['score']:.5f}  hit {agg['hit']:.3f}  "
            f"mrr {agg['mrr']:.4f}  mttc {agg['mttc']:.2f}")


def line(char="-", width=88):
    print(char * width)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=None)
    parser.add_argument("--run-dir", default="results/runs/de61dda7dc0e")
    args = parser.parse_args()
    path = args.results or str(Path(args.run_dir) / "test_results.jsonl")
    rows = load(path)
    by_family = collections.defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)

    print("=" * 88)
    print("STAGE 3D - INITIAL REPORTING")
    print("=" * 88)
    print(f"rows {len(rows)}   agent {rows[0]['agent_git_sha'][:12]}   "
          f"suite {rows[0]['suite_version']}   cases_sha {rows[0]['test_cases_sha256'][:16]}")
    print("\nThe 8,520-session suite is an ADVERSARIAL EXPERIMENTAL SUITE.")
    print("No unweighted overall score is reported: it would not mean anything.")

    # ---------------- 1. F1a ----------------
    print("\n" + "=" * 88)
    print("1. F1a - PRIVATE SURROGATE (schemes reported separately, never averaged)")
    line()
    f1a = shipped(by_family["F1a"])
    envelope = []
    for scheme in ("B", "C"):
        srows = [r for r in f1a if r["scheme"] == scheme]
        agg = aggregate(srows)
        lo, hi = bootstrap_ci(srows)
        envelope += [lo, hi]
        print(f"  Scheme {scheme}   {fmt(agg)}")
        print(f"              95% CI [{lo:.5f}, {hi:.5f}]")
        for cell in sorted({r['cell_id'] for r in srows}):
            print(f"                {cell:24s} {fmt(aggregate([r for r in srows if r['cell_id']==cell]))}")
    print(f"\n  SURROGATE ENVELOPE (union of the two CIs): "
          f"[{min(envelope):.5f}, {max(envelope):.5f}]")
    print(f"  public reference (200 real targets)      : {PUBLIC_REFERENCE:.5f}")
    print("\n  T3 STRESS BOUNDS (bounds only -- never an estimate):")
    for cell in sorted({r["cell_id"] for r in by_family["F1f"]}):
        print(f"    {cell:24s} {fmt(aggregate(shipped([r for r in by_family['F1f'] if r['cell_id']==cell])))}")
    print("\n  F1g SENTINELS (excluded from every headline figure):")
    for cell in sorted({r["cell_id"] for r in by_family["F1g"]}):
        print(f"    {cell:24s} {fmt(aggregate(shipped([r for r in by_family['F1g'] if r['cell_id']==cell])))}")
    print("\n  F1b/F1c/F1d/F1e strata:")
    for fam in ("F1b", "F1c", "F1d", "F1e"):
        for cell in sorted({r["cell_id"] for r in by_family[fam]}):
            print(f"    {cell:24s} {fmt(aggregate(shipped([r for r in by_family[fam] if r['cell_id']==cell])))}")

    # ---------------- 2. F2A ----------------
    print("\n" + "=" * 88)
    print("2. F2A - SURFACE PARAPHRASE (constraints byte-identical => parser failures)")
    line()
    f2a = shipped(by_family["F2A"])
    def cellparts(r): return r["cell_id"].split("/")
    print("  by SURFACE:")
    for surface in sorted({cellparts(r)[1] for r in f2a}):
        print(f"    {surface:16s} {fmt(aggregate([r for r in f2a if cellparts(r)[1]==surface]))}")
    print("  by KIND:")
    for kind in sorted({cellparts(r)[2] for r in f2a}):
        print(f"    {kind:16s} {fmt(aggregate([r for r in f2a if cellparts(r)[2]==kind]))}")
    print("  by SEVERITY:")
    for sev in ("S1", "S2", "S3"):
        print(f"    {sev:16s} {fmt(aggregate([r for r in f2a if r['severity']==sev]))}")
    print("  by SCENARIO:")
    for scen in sorted({r["scenario_type"] for r in f2a}):
        print(f"    {scen:16s} {fmt(aggregate([r for r in f2a if r['scenario_type']==scen]))}")
    print("  SURFACE x SEVERITY (score):")
    surfaces = sorted({cellparts(r)[1] for r in f2a})
    print(f"    {'surface':16s} " + "".join(f"{s:>10s}" for s in ("S1", "S2", "S3")))
    for surface in surfaces:
        cells = [aggregate([r for r in f2a if cellparts(r)[1] == surface and r["severity"] == s])
                 for s in ("S1", "S2", "S3")]
        print(f"    {surface:16s} " + "".join(f"{c['score']:10.5f}" for c in cells))
    print("  FAILURE-STAGE histogram:")
    total = len(f2a)
    for stage, count in collections.Counter(r["failure_stage"] for r in f2a).most_common():
        print(f"    {stage:24s} {count:5d}  {100*count/total:5.1f}%")
    print("  failure stage x surface (non-'none' only):")
    for surface in surfaces:
        sub = [r for r in f2a if cellparts(r)[1] == surface and r["failure_stage"] != "none"]
        if sub:
            hist = collections.Counter(r["failure_stage"] for r in sub)
            print(f"    {surface:16s} " + ", ".join(f"{k}={v}" for k, v in hist.most_common()))

    # ---------------- 3. F2B ----------------
    print("\n" + "=" * 88)
    print("3. F2B - EVIDENCE PARAPHRASE (templates byte-identical => retrieval failures)")
    line()
    f2b = shipped(by_family["F2B"])
    main = [r for r in f2b if not r["diagnostic_case"]]
    diag = [r for r in f2b if r["diagnostic_case"]]
    print("  MAIN MATRIX by achieved severity:")
    for sev in ("E1", "E2", "E3"):
        print(f"    {sev:16s} {fmt(aggregate([r for r in main if r['achieved_severity']==sev]))}")
    print("  MAIN MATRIX by kind x severity (score):")
    kinds = sorted({r["cell_id"].split("/")[1] for r in main})
    print(f"    {'kind':16s} " + "".join(f"{s:>10s}" for s in ("E1", "E2", "E3")))
    for kind in kinds:
        cells = []
        for sev in ("E1", "E2", "E3"):
            sub = [r for r in main if r["cell_id"].split("/")[1] == kind and r["severity"] == sev]
            cells.append(f"{aggregate(sub)['score']:10.5f}" if sub else f"{'-':>10s}")
        print(f"    {kind:16s} " + "".join(cells))
    print("  FAILURE-STAGE (main):")
    for stage, count in collections.Counter(r["failure_stage"] for r in main).most_common():
        print(f"    {stage:24s} {count:5d}  {100*count/len(main):5.1f}%")
    print(f"  DIAGNOSTICS (t2_reorder ceiling, excluded from aggregates): "
          f"{fmt(aggregate(diag))}")

    # ---------------- 4. F3 ----------------
    print("\n" + "=" * 88)
    print("4. F3 - BUYING EARLY-PAGE (paired; clip arm is DIAGNOSTIC)")
    line()
    f3_ship = [r for r in by_family["F3"] if r["arm"] == "as_ships"]
    f3_clip = [r for r in by_family["F3"] if r["arm"] == "turn1_clipped"]
    print(f"  as-ships   {fmt(aggregate(f3_ship))}")
    print(f"  clipped    {fmt(aggregate(f3_clip))}   [DIAGNOSTIC]")
    deltas = [r["paired_delta"]["technical_score_contribution"] for r in f3_clip]
    rr_deltas = [r["paired_delta"]["reciprocal_rank"] for r in f3_clip]
    print(f"  paired delta per session: mean {statistics.mean(deltas):+.5f}  "
          f"median {statistics.median(deltas):+.5f}")
    print(f"  sessions improved {sum(1 for d in deltas if d > 1e-9)}  "
          f"worsened {sum(1 for d in deltas if d < -1e-9)}  "
          f"unchanged {sum(1 for d in deltas if abs(d) <= 1e-9)}")
    print(f"  aggregate score delta: {aggregate(f3_clip)['score'] - aggregate(f3_ship)['score']:+.5f}")
    print("  by candidate-pool band:")
    for cell in sorted({r["cell_id"] for r in f3_ship}):
        ship = aggregate([r for r in f3_ship if r["cell_id"] == cell])
        clip = aggregate([r for r in f3_clip if r["cell_id"] == cell])
        band_deltas = [r["paired_delta"]["technical_score_contribution"]
                       for r in f3_clip if r["cell_id"] == cell]
        print(f"    {cell:20s} ships {ship['score']:.5f}  clipped {clip['score']:.5f}  "
              f"delta {clip['score']-ship['score']:+.5f}  mean/session {statistics.mean(band_deltas):+.5f}")
    turn1 = [r for r in f3_ship if r["first_hit_turn"] == 1]
    print(f"  turn-1 hits as-ships: {len(turn1)}/{len(f3_ship)}  "
          f"of which rank>1: {sum(1 for r in turn1 if r['best_rank'] and r['best_rank']>1)}")

    # ---------------- 5. F4 ----------------
    print("\n" + "=" * 88)
    print("5. F4 - THRESHOLD RESPONSE (ALL arms DIAGNOSTIC)")
    line()
    f4 = by_family["F4"]
    for generator in sorted({r["cell_id"] for r in f4}):
        print(f"  {generator}")
        sub = [r for r in f4 if r["cell_id"] == generator]
        control = aggregate([r for r in sub if r["arm"] == "config_control"])
        print(f"    {'shipped config':32s} score {control['score']:.5f}  hit {control['hit']:.3f}")
        by_const = collections.defaultdict(list)
        for row in sub:
            if row["config_override"]:
                key = list(row["config_override"])[0]
                by_const[key].append(row)
        for const in sorted(by_const):
            points = collections.defaultdict(list)
            for row in by_const[const]:
                points[row["config_override"][const]].append(row)
            rendered = "  ".join(f"{v}:{aggregate(points[v])['score']:.5f}" for v in sorted(points))
            print(f"    {const:32s} {rendered}")

    # ---------------- 6. F5 ----------------
    print("\n" + "=" * 88)
    print("6. F5 - DEEP-TARGET REACHABILITY")
    line()
    for cell in sorted({r["cell_id"] for r in by_family["F5"]},
                       key=lambda c: int(c.split("_")[-1]) if c.split("_")[-1].isdigit() else 999):
        sub = shipped([r for r in by_family["F5"] if r["cell_id"] == cell])
        print(f"  {cell:20s} {fmt(aggregate(sub))}")
    print("  by scenario within rank_gt100:")
    deep = shipped([r for r in by_family["F5"] if r["cell_id"] == "F5/rank_gt100"])
    for scen in sorted({r["scenario_type"] for r in deep}):
        print(f"    {scen:16s} {fmt(aggregate([r for r in deep if r['scenario_type']==scen]))}")

    # ---------------- 7. F6 ----------------
    print("\n" + "=" * 88)
    print("7. F6 - SEMICOLON PARSING (ENRICHED sampling; base rate 232/50000 = 0.464%)")
    line()
    f6 = shipped(by_family["F6"])
    for cell in sorted({r["cell_id"] for r in f6}):
        print(f"  {cell:28s} {fmt(aggregate([r for r in f6 if r['cell_id']==cell]))}")
    enriched = aggregate(f6)
    known = aggregate([r for r in f6 if r["cell_id"] == "F6/known_excluding"])
    baseline_gap = PUBLIC_REFERENCE - known["score"]
    print(f"\n  enriched overall           {fmt(enriched)}")
    print(f"  BASE-RATE-WEIGHTED effect on 800 private sessions:")
    print(f"    known-excluding cell gap vs public reference : {baseline_gap:+.5f}")
    print(f"    base rate 0.464% -> expected sessions in 800 : {800*0.00464:.1f}")
    print(f"    expected TechnicalScore impact               : {baseline_gap*0.00464:+.6f}")

    # ---------------- 8. F7 ----------------
    print("\n" + "=" * 88)
    print("8. F7 - OOD SHOPPER LANGUAGE (class C -- NOT a private-evaluator projection)")
    line()
    f7 = shipped(by_family["F7"])
    print(f"  overall OOD {fmt(aggregate(f7))}    [floor probe reference 0.83199]")
    for cell in sorted({r["cell_id"] for r in f7},
                       key=lambda c: aggregate([r for r in f7 if r["cell_id"] == c])["score"]):
        sub = [r for r in f7 if r["cell_id"] == cell]
        stages = collections.Counter(r["failure_stage"] for r in sub)
        top = ", ".join(f"{k}={v}" for k, v in stages.most_common(2))
        print(f"    {cell:30s} {fmt(aggregate(sub))}   {top}")

    # ---------------- 9. F8 ----------------
    print("\n" + "=" * 88)
    print("9. F8 - 800-SESSION SOAK")
    line()
    f8 = shipped(by_family["F8"])
    print(f"  soak {fmt(aggregate(f8))}")
    half = len(f8) // 2
    print(f"  first half  {fmt(aggregate(f8[:half]))}")
    print(f"  second half {fmt(aggregate(f8[half:]))}  (drift would indicate state leakage)")
    print(f"  failure stages: {dict(collections.Counter(r['failure_stage'] for r in f8))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
