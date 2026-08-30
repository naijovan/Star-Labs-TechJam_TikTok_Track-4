"""Deterministic test-case (session definition) generation.

A ROW IN `test_cases.jsonl` IS A SESSION DEFINITION, not a result.  It is
agent-independent by design: a future agent version must be runnable against the
exact same file.  That is why a row stores the *transformation spec* rather than
a full dialogue -- the customer's later replies depend on what the agent asks,
so they cannot be frozen without freezing the agent too.

What IS frozen per row:
  * the target and its stratum
  * the scenario, profile and sample_id (which fix the simulator's RNG)
  * the intent card the evaluator will derive
  * the rendered clean opening and the rendered transformed opening
  * the template variant and/or constraint map the harness must apply
  * the hypothesis and the invariant the case is asserting

Families that reuse a session across arms (F3) or configurations (F4) carry
`arms` / `config_grid`, and `n_sessions` records the evaluator cost.  The sum of
`n_sessions` over the file is the suite's session budget.
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from evaluator.local_evaluator import behavior_for, coarse_category, intent_card

from tools.suite import SUITE_VERSION
from tools.suite.pool import (
    ACCEPTED_SCHEMES,
    CoarseJointScheme,
    NearestNeighbourScheme,
    T1Mixture,
    stress_pool,
)
from tools.suite.strata import CatalogFeatures, load_public_targets
from tools.suite import transforms as TR

SCENARIOS = ("buying", "browsing", "intent_override", "boundary")
SCENARIO_MIX = (("buying", 40), ("browsing", 40), ("intent_override", 15), ("boundary", 5))


# ---------------------------------------------------------------- helpers

def scenario_cycle(n: int) -> List[str]:
    """Deterministic 40/40/15/5 mix -- a repeating 20-slot pattern, no RNG."""
    pattern = (["buying"] * 8 + ["browsing"] * 8 + ["intent_override"] * 3 + ["boundary"] * 1)
    return [pattern[i % len(pattern)] for i in range(n)]


def _stable_id(*parts) -> str:
    raw = "\0".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class TestCase:
    test_id: str
    suite_version: str
    family: str
    cell_id: str
    klass: str                       # "A" | "B" | "C"
    tier: str                        # "T0" public | "T1" | "T2" | "T3" | "n/a"
    severity: str
    scheme: Optional[str]            # T1 calibration scheme label, else None
    target_parent_asin: str
    target_stratum: dict
    scenario_type: str
    user_profile: dict
    sample_id: str
    seed: int
    clean_inputs: dict
    transformed_inputs: dict
    transformation: dict
    hypothesis: str
    expected_invariant: str
    sentinel: bool = False
    excluded_from_headline: bool = False
    stress_bound: bool = False
    ood: bool = False
    arms: List[str] = field(default_factory=lambda: ["as_ships"])
    config_grid: List[dict] = field(default_factory=list)
    n_sessions: int = 1
    requested_severity: Optional[str] = None
    achieved_severity: Optional[str] = None
    generation_attempts: int = 1
    downgraded: bool = False
    diagnostic: bool = False

    def as_row(self) -> dict:
        row = asdict(self)
        row["class"] = row.pop("klass")
        return row


class CaseBuilder:
    """Builds the deterministic per-target facts a row needs."""

    def __init__(self, features: CatalogFeatures, catalog_path: str, dataset_path: str) -> None:
        self.features = features
        self.public_targets = load_public_targets(dataset_path)
        self.public_target_set = set(self.public_targets)
        self._products: Dict[str, dict] = {}
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    product = json.loads(line)
                    self._products[str(product["parent_asin"])] = product
        self.profiles = []
        with Path(dataset_path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    self.profiles.append(json.loads(line)["user_profile"])

    def card(self, asin: str) -> dict:
        return intent_card(self._products[asin])

    def category(self, asin: str) -> str:
        return coarse_category([str(v) for v in self._products[asin].get("categories") or []])

    def clean_inputs(self, asin: str, scenario: str, sample_id: str) -> dict:
        card = self.card(asin)
        cat = self.category(asin)
        behavior = behavior_for(scenario, card, random.Random(f"{sample_id}\0{scenario}"))
        override = behavior.get("override") or {}
        hard = card["hard_constraints"]
        if scenario == "buying" and hard:
            opening = TR.render(TR.CANONICAL["opening_buying"], cat=cat, constraint=str(hard[0]))
            opening_template = "opening_buying"
        elif scenario == "intent_override":
            opening = TR.render(TR.CANONICAL["opening_override"], cat=cat,
                                old=str(override.get("old_value", "")))
            opening_template = "opening_override"
        else:
            opening = TR.render(TR.CANONICAL["opening_browsing"], cat=cat)
            opening_template = "opening_browsing"
        return {
            "coarse_category": cat,
            "opening_template": opening_template,
            "opening_message": opening,
            "intent_card": {"hard_constraints": card["hard_constraints"],
                            "soft_preferences": card["soft_preferences"]},
            "constraint_pool": list(dict.fromkeys(
                [*card["hard_constraints"], *card["soft_preferences"]])),
            "override": {"turn": override.get("turn"), "old_value": override.get("old_value"),
                         "new_value": override.get("new_value"),
                         "message": override.get("message")} if override else None,
        }

    def profile(self, index: int) -> dict:
        return self.profiles[index % len(self.profiles)]


# ---------------------------------------------------------------- families

class SuiteGenerator:
    def __init__(self, features: CatalogFeatures, builder: CaseBuilder, master_seed: int = 20260830) -> None:
        self.f = features
        self.b = builder
        self.master_seed = master_seed
        self.targets = builder.public_targets
        self._t1: Dict[str, T1Mixture] = {}

    def t1(self, scheme_key: str) -> T1Mixture:
        if scheme_key not in self._t1:
            self._t1[scheme_key] = T1Mixture(self.f, self.targets, ACCEPTED_SCHEMES[scheme_key])
        return self._t1[scheme_key]

    def _mk(self, *, family, cell_id, klass, tier, severity, asin, scenario, index, seed,
            hypothesis, invariant, scheme=None, transformation=None, transformed=None,
            **flags) -> TestCase:
        sample_id = f"{family}_{cell_id}_{index}"
        clean = self.b.clean_inputs(asin, scenario, sample_id)
        return TestCase(
            test_id=_stable_id(SUITE_VERSION, family, cell_id, index, asin, scenario, seed),
            suite_version=SUITE_VERSION,
            family=family, cell_id=cell_id, klass=klass, tier=tier, severity=severity,
            scheme=scheme,
            target_parent_asin=asin,
            target_stratum=self.f.stratum(asin).as_dict(),
            scenario_type=scenario,
            user_profile=self.b.profile(seed + index),
            sample_id=sample_id,
            seed=seed,
            clean_inputs=clean,
            transformed_inputs=transformed if transformed is not None else dict(clean),
            transformation=transformation or {"surface": None, "kind": None, "severity": "none"},
            hypothesis=hypothesis,
            expected_invariant=invariant,
            **flags,
        )

    # ---- F1: target draw -------------------------------------------------

    def f1a(self, per_draw=250, seeds=(11, 22, 33)) -> List[TestCase]:
        out = []
        for key in sorted(ACCEPTED_SCHEMES):
            mixture = self.t1(key)
            for seed in seeds:
                draw = mixture.draw(per_draw, seed)
                core_cut = draw.n_core
                scen = scenario_cycle(len(draw.asins))
                for i, asin in enumerate(draw.asins):
                    out.append(self._mk(
                        family="F1a", cell_id=f"F1a/{key}/seed{seed}", klass="B", tier="T1",
                        severity="none", asin=asin, scenario=scen[i], index=i, seed=seed,
                        scheme=key,
                        hypothesis="A target draw calibrated to the observed public-target "
                                   "signature transfers better than a uniform draw.",
                        invariant="TechnicalScore within the surrogate envelope; "
                                  "HitRate@10 >= 0.99",
                    ))
        return out

    def _stratified(self, family, cells, klass="B", tier="T1", hypothesis="", invariant="",
                    stress=False) -> List[TestCase]:
        out = []
        for cell_name, (asins, n, seed) in sorted(cells.items()):
            if not asins:
                continue
            rng = random.Random(seed)
            picked = [rng.choice(asins) for _ in range(n)]
            scen = scenario_cycle(n)
            for i, asin in enumerate(picked):
                out.append(self._mk(
                    family=family, cell_id=f"{family}/{cell_name}", klass=klass, tier=tier,
                    severity="none", asin=asin, scenario=scen[i], index=i, seed=seed,
                    hypothesis=hypothesis, invariant=invariant, stress_bound=stress,
                ))
        return out

    def _donors(self) -> List[str]:
        return self.f.donor_pool(exclude=self.b.public_targets, strict=False)

    def f1b(self, n_per=40) -> List[TestCase]:
        donors = self._donors()
        ordered = sorted(donors, key=lambda a: (self.f.rating_number(a), a))
        size = len(ordered) // 10
        cells = {f"popdecile{d}": (ordered[d * size:(d + 1) * size], n_per, 1000 + d) for d in range(10)}
        return self._stratified("F1b", cells,
                                hypothesis="Score degrades monotonically as the popularity prior weakens.",
                                invariant="No decile below 0.90; response is a plateau, not a collapse.")

    def f1c(self, n_per=30) -> List[TestCase]:
        donors = self._donors()
        ordered = sorted(donors, key=lambda a: (self.f.bucket_rank(a), a))
        size = len(ordered) // 10
        cells = {f"rankdecile{d}": (ordered[d * size:(d + 1) * size], n_per, 2000 + d) for d in range(10)}
        return self._stratified("F1c", cells,
                                hypothesis="Within-bucket rank, not raw popularity, drives the floor route.",
                                invariant="HitRate@10 >= 0.98 for deciles whose rank <= 100.")

    def f1d(self, n_per=30) -> List[TestCase]:
        donors = self._donors()
        bands = {"tiny_le10": lambda s: s <= 10, "small_11_70": lambda s: 11 <= s <= 70,
                 "mid_71_234": lambda s: 71 <= s <= 234, "large_235_769": lambda s: 235 <= s <= 769,
                 "xl_ge770": lambda s: s >= 770}
        cells = {name: ([a for a in donors if test(self.f.bucket_size(a))], n_per, 3000 + i)
                 for i, (name, test) in enumerate(sorted(bands.items()))}
        wrapper = [a for a in donors if self.f.bucket(a).startswith("Shoes & Jewelry")]
        cells["wrapper_leak"] = (wrapper, n_per, 3099)
        return self._stratified("F1d", cells,
                                hypothesis="Bucket geometry unexercised by the public 200 (88.97% of "
                                           "buckets) behaves like the exercised geometry.",
                                invariant="No bucket band below 0.95 TechnicalScore.")

    def f1e(self, n_per=30) -> List[TestCase]:
        donors = self._donors()
        cells = {
            "df_unique": ([a for a in donors if self.f.rarest_df(a) == 1], n_per, 4001),
            "df_2_10": ([a for a in donors if 2 <= self.f.rarest_df(a) <= 10], n_per, 4002),
            "df_common_only": ([a for a in donors if self.f.rarest_df(a) > 1000], n_per, 4003),
            "unpriced": ([a for a in donors if not self.f.has_price(a)], n_per, 4004),
        }
        return self._stratified("F1e", cells,
                                hypothesis="Targets owning no rare constraint fall back to popularity "
                                           "alone, so rarity and popularity failures compound.",
                                invariant="df_common_only HitRate@10 >= 0.95.")

    def f1f(self, n_per=50) -> List[TestCase]:
        cells = {
            "uniform_50k": (stress_pool(self.f, self.b.public_targets, "uniform_50k"), n_per, 5001),
            "unpopular_half": (stress_pool(self.f, self.b.public_targets, "unpopular_half"), n_per, 5002),
        }
        return self._stratified("F1f", cells, klass="B", tier="T3", stress=True,
                                hypothesis="Uniform and unpopular-half draws bound the worst case.",
                                invariant="STRESS BOUND ONLY -- never quoted as a private estimate.")

    def f1g(self, n_per=40) -> List[TestCase]:
        donors = [a for a in self.f.asins if a not in self.b.public_target_set]
        cells = {
            "empty_features": ([a for a in donors if not self.f.has_features(a)], n_per, 6001),
            "lt4_constraints": ([a for a in donors if self.f.n_constraints(a) < 4], n_per, 6002),
            "old_eq_new": ([a for a in donors if self.f.n_constraints(a) <= 2], n_per, 6003),
        }
        out = []
        for cell_name, (asins, n, seed) in sorted(cells.items()):
            if not asins:
                continue
            rng = random.Random(seed)
            picked = [rng.choice(asins) for _ in range(n)]
            scen = (["intent_override"] * n if cell_name == "old_eq_new" else scenario_cycle(n))
            for i, asin in enumerate(picked):
                out.append(self._mk(
                    family="F1g", cell_id=f"F1g/{cell_name}", klass="B", tier="T2",
                    severity="none", asin=asin, scenario=scen[i], index=i, seed=seed,
                    sentinel=True, excluded_from_headline=True,
                    hypothesis="Very unlikely under observed target selection (0/200 public "
                               "targets); retained as a tripwire, not an estimate.",
                    invariant="TRIPWIRE: HitRate@10 >= 0.90 within the cell. Never enters "
                              "any headline private-score figure.",
                ))
        return out

    # ---- F2A: surface paraphrase ----------------------------------------

    _F2A_SURFACE_TEMPLATES = {
        "opening": ["opening_buying", "opening_browsing", "opening_override"],
        "payout": ["payout"],
        "filler": ["filler_none", "filler_drained", "filler_boundary"],
        "override": ["override"],
        "opening+payout": ["opening_buying", "opening_browsing", "opening_override", "payout"],
        "all": list(TR.CANONICAL),
    }

    def f2a(self, total=1700) -> List[TestCase]:
        cells = [(s, k, sev) for s in TR.SURFACES for k in TR.F2A_KINDS for sev in TR.F2A_SEVERITIES]
        base, extra = divmod(total, len(cells))
        mixture = self.t1("C")
        pool = mixture.draw(max(total, 400), 909).asins
        out, cursor = [], 0
        for cell_index, (surface, kind, sev) in enumerate(cells):
            n = base + (1 if cell_index < extra else 0)
            for i in range(n):
                asin = pool[cursor % len(pool)]
                cursor += 1
                scenario = ("intent_override" if surface == "override"
                            else scenario_cycle(cursor)[-1])
                clean = self.b.clean_inputs(asin, scenario, f"F2A_{surface}/{kind}/{sev}_{i}")
                variant = {t: TR.template_for(t, kind, sev, i)
                           for t in self._F2A_SURFACE_TEMPLATES[surface]}
                transformed = dict(clean)
                ot = clean["opening_template"]
                if ot in variant:
                    transformed["opening_message"] = TR.render(
                        variant[ot], cat=clean["coarse_category"],
                        constraint=str(clean["intent_card"]["hard_constraints"][0])
                        if clean["intent_card"]["hard_constraints"] else "",
                        old=str((clean["override"] or {}).get("old_value") or ""))
                out.append(self._mk(
                    family="F2A", cell_id=f"F2A/{surface}/{kind}/{sev}", klass="B", tier="T1",
                    severity=sev, asin=asin, scenario=scenario, index=i, seed=909,
                    scheme="C", transformed=transformed,
                    transformation={"surface": surface, "kind": kind, "severity": sev,
                                    "template_variant": variant, "constraint_map": {},
                                    "variant_index": i, "scripted_turns": {}},
                    hypothesis="Paraphrasing the prose AROUND verbatim constraints breaks the "
                               "parser, not retrieval; loss should concentrate in a few stages.",
                    invariant="F2A INVARIANT: every constraint string is byte-identical to the "
                              "clean card. Any clue_extraction failure means contamination.",
                ))
        return out

    # ---- F2B: evidence paraphrase ---------------------------------------

    _SEV_RANK = {"E0": 0, "E1": 1, "E2": 2, "E3": 3}

    # t2_reorder E2/E3 are NOT in the main matrix: structural reordering cannot
    # reliably reach those severities (measured), so those cells could not support
    # a severity comparison.  A small labelled diagnostic set records the ceiling.
    _F2B_MAIN_CELLS = [
        ("t1_pattern", "E1"), ("t1_pattern", "E2"), ("t1_pattern", "E3"),
        ("t2_reorder", "E1"),
        ("t3_carrier", "E1"), ("t3_carrier", "E2"), ("t3_carrier", "E3"),
        ("mixed", "E1"), ("mixed", "E2"), ("mixed", "E3"),
        ("auto", "E1"), ("auto", "E2"), ("auto", "E3"),
        ("shape_mixed", "E3"),
    ]
    _F2B_DIAG_CELLS = [("t2_reorder", "E2"), ("t2_reorder", "E3")]
    _F2B_DIAG_N = 4

    @staticmethod
    def _weighted_alloc(total: int, weights: List[int]) -> List[int]:
        """Deterministic allocation summing exactly to `total`, remainder to the
        heaviest cells first."""
        weight_sum = sum(weights)
        base = [total * w // weight_sum for w in weights]
        order = sorted(range(len(weights)), key=lambda i: (-weights[i], i))
        for j in range(total - sum(base)):
            base[order[j % len(order)]] += 1
        return base

    def _f2b_case(self, kind, sev, i, pool, cursor, max_attempts, diagnostic):
        sample_id = f"F2B_F2B{'_DIAG' if diagnostic else ''}/{kind}/{sev}_{i}"
        chosen, best, attempts = None, None, 0
        while attempts < max_attempts:
            asin = pool[cursor % len(pool)]
            cursor += 1
            attempts += 1
            scenario = scenario_cycle(cursor)[-1]
            clean = self.b.clean_inputs(asin, scenario, sample_id)
            result = TR.build_constraint_map(clean["constraint_pool"], kind, sev)
            candidate = (asin, scenario, clean, result, attempts)
            if best is None or (self._SEV_RANK[result["achieved_severity"]]
                                > self._SEV_RANK[best[3]["achieved_severity"]]):
                best = candidate
            if result["achieved_severity"] == sev:
                chosen = candidate
                break
        return (chosen or best), cursor

    def f2b(self, total=500) -> List[TestCase]:
        """Main matrix + a small, separately labelled t2_reorder ceiling diagnostic."""
        diag_total = len(self._F2B_DIAG_CELLS) * self._F2B_DIAG_N
        main_total = max(total - diag_total, len(self._F2B_MAIN_CELLS))
        # Prioritise E3: those cells carry double weight.
        weights = [2 if sev == "E3" else 1 for _, sev in self._F2B_MAIN_CELLS]
        counts = self._weighted_alloc(main_total, weights)
        pool = self.t1("C").draw(max(total * 8, 4000), 707).asins
        out, cursor = [], 0

        for (kind, sev), n in zip(self._F2B_MAIN_CELLS, counts):
            for i in range(n):
                (asin, scenario, clean, result, attempts), cursor = self._f2b_case(
                    kind, sev, i, pool, cursor, 60, False)
                out.append(self._f2b_row(kind, sev, i, asin, scenario, clean, result,
                                         attempts, diagnostic=False))

        for kind, sev in self._F2B_DIAG_CELLS:
            for i in range(self._F2B_DIAG_N):
                (asin, scenario, clean, result, attempts), cursor = self._f2b_case(
                    kind, sev, i, pool, cursor, 25, True)
                out.append(self._f2b_row(kind, sev, i, asin, scenario, clean, result,
                                         attempts, diagnostic=True))
        return out

    def _f2b_row(self, kind, sev, i, asin, scenario, clean, result, attempts, diagnostic):
        cmap = result["constraint_map"]
        transformed = dict(clean)
        transformed["opening_message"] = self._reopen(clean, cmap)
        transformed["constraint_pool"] = [cmap.get(c, c) for c in clean["constraint_pool"]]
        downgraded = result["achieved_severity"] != sev
        prefix = "F2B_DIAG" if diagnostic else "F2B"
        if diagnostic:
            hypothesis = ("DIAGNOSTIC: pure structural reordering has a natural severity "
                          "ceiling -- most constraints have nothing safe to reorder.")
            invariant = ("DIAGNOSTIC ONLY. Excluded from F2B severity aggregates and from "
                         "every headline figure. Records the ceiling, does not measure it.")
        else:
            hypothesis = ("With the parser intact, paraphrased constraints miss clue_to "
                          "entirely and the agent must fall back to BM25 and the floor.")
            invariant = ("F2B INVARIANT: every template is byte-identical to canonical. "
                         "achieved_severity equals requested_severity in the main matrix.")
        return self._mk(
            family="F2B", cell_id=f"{prefix}/{kind}/{sev}", klass="B", tier="T1",
            severity=sev, asin=asin, scenario=scenario, index=i, seed=707,
            scheme="C", transformed=transformed,
            requested_severity=sev,
            achieved_severity=result["achieved_severity"],
            generation_attempts=attempts,
            downgraded=downgraded,
            diagnostic=diagnostic,
            excluded_from_headline=diagnostic,
            transformation={"surface": "evidence", "kind": kind, "severity": sev,
                            "requested_severity": sev,
                            "achieved_severity": result["achieved_severity"],
                            "template_variant": {}, "constraint_map": cmap,
                            "constraint_records": result["constraint_records"],
                            "unparaphrased": result["unparaphrased"],
                            "rejected": result["rejected"],
                            "coverage": result["coverage"],
                            "diagnostic": diagnostic,
                            "scripted_turns": {}},
            hypothesis=hypothesis, invariant=invariant)

    def _reopen(self, clean: dict, cmap: Dict[str, str]) -> str:
        """Re-render the canonical opening with paraphrased slot values."""
        template = TR.CANONICAL[clean["opening_template"]]
        hard = clean["intent_card"]["hard_constraints"]
        first = str(hard[0]) if hard else ""
        old = str((clean["override"] or {}).get("old_value") or "")
        return TR.render(template, cat=clean["coarse_category"],
                         constraint=cmap.get(first, first), old=cmap.get(old, old))

    # ---- F3-F8 -----------------------------------------------------------

    def f3(self, n_per=100) -> List[TestCase]:
        donors = self._donors()
        buying = sorted(donors, key=lambda a: (self.f.bucket_size(a), a))
        bands = {"pool_le10": [a for a in buying if self.f.bucket_size(a) <= 10],
                 "pool_11_40": [a for a in buying if 11 <= self.f.bucket_size(a) <= 40],
                 "pool_41_200": [a for a in buying if 41 <= self.f.bucket_size(a) <= 200],
                 "pool_gt200": [a for a in buying if self.f.bucket_size(a) > 200]}
        out = []
        for band_index, (name, asins) in enumerate(sorted(bands.items())):
            rng = random.Random(7000 + band_index)
            for i in range(n_per):
                asin = rng.choice(asins)
                out.append(self._mk(
                    family="F3", cell_id=f"F3/{name}", klass="A", tier="T1",
                    severity="none", asin=asin, scenario="buying", index=i, seed=7000 + band_index,
                    scheme="C", arms=["as_ships", "turn1_clipped"], n_sessions=2,
                    hypothesis="NOEVID_PAGE cannot fire for buying, so turn 1 emits a full "
                               "10-card page above ~40 candidates and can lock a bad rank.",
                    invariant="Paired: clipping turn 1 to one card must not reduce HitRate@10.",
                ))
        return out

    def f4(self, n_base=15) -> List[TestCase]:
        grid = ([{"JACCARD_MIN": v} for v in (0.30, 0.40, 0.50, 0.60, 0.75)]
                + [{"FUZZY_CATEGORY": v} for v in (0.45, 0.60, 0.75, 0.90)]
                + [{"WEAK_ABS": v} for v in (0.5, 1.0, 1.5, 2.5, 4.0)]
                + [{"WEAK_RATIO": v} for v in (1.00, 1.05, 1.15, 1.35, 1.75)]
                + [{}])
        pool = self.t1("C").draw(n_base, 8080).asins
        out = []
        for gen_index, generator in enumerate(("D1_word_reversal", "D2_char_corruption")):
            scen = scenario_cycle(n_base)
            for i, asin in enumerate(pool):
                out.append(self._mk(
                    family="F4", cell_id=f"F4/{generator}", klass="B", tier="T1",
                    severity="none", asin=asin, scenario=scen[i], index=i, seed=8080 + gen_index,
                    scheme="C", config_grid=grid, n_sessions=len(grid),
                    transformation={"surface": "opening", "kind": generator,
                                    "severity": "category_damage", "template_variant": {},
                                    "constraint_map": {}, "scripted_turns": {}},
                    hypothesis="The four thresholds are inert on clean data and may be fitted to "
                               "the single cat_noise generator they have ever run under.",
                    invariant="Response curve peaks at the shipped value under BOTH generators, "
                              "or the constants are fitted to D1.",
                ))
        return out

    def f5(self, n_per=100) -> List[TestCase]:
        donors = self._donors()
        bands = {"rank_1_10": (1, 10), "rank_11_25": (11, 25), "rank_26_50": (26, 50),
                 "rank_51_100": (51, 100), "rank_gt100": (101, 10 ** 9)}
        out = []
        for band_index, (name, (lo, hi)) in enumerate(sorted(bands.items())):
            asins = [a for a in donors if lo <= self.f.bucket_rank(a) <= hi]
            rng = random.Random(9000 + band_index)
            scen = [("browsing", "boundary", "buying")[i % 3] for i in range(n_per)]
            for i in range(n_per):
                out.append(self._mk(
                    family="F5", cell_id=f"F5/{name}", klass="B", tier="T1",
                    severity="none", asin=rng.choice(asins), scenario=scen[i], index=i,
                    seed=9000 + band_index, scheme="C",
                    hypothesis="The scheduler exposes exactly 100 (turn, rank) slots, so a target "
                               "that never rises above rank 100 is positionally unreachable.",
                    invariant="HitRate@10 >= 0.98 for bands up to rank 100.",
                ))
        return out

    def f6(self, n_per=30) -> List[TestCase]:
        donors = self._donors()
        semi = [a for a in donors if any(";" in c for c in self.f.constraints(a))]
        multi = [a for a in semi if any(c.count(";") >= 2 for c in self.f.constraints(a))]
        prefix_amb = [a for a in semi
                      if any(";" in c and c.split(";")[0].strip() in self.f.document_frequency
                             for c in self.f.constraints(a))]
        dup = [a for a in self.f.asins
               if a not in self.b.public_target_set and self.f.n_constraints(a) == 1]
        excluding = [a for a in prefix_amb
                     if any(";" in c and self.f.document_frequency.get(c.split(";")[0].strip(), 0) > 0
                            and c.split(";")[0].strip() not in self.f.constraints(a)
                            for c in self.f.constraints(a))]
        cells = {"single_semicolon": (semi, n_per, 10001), "multi_semicolon": (multi, n_per, 10002),
                 "ambiguous_prefix": (prefix_amb, n_per, 10003), "duplicate_reply": (dup, n_per, 10004),
                 "known_excluding": (excluding, n_per, 10005)}
        return self._stratified("F6", cells, klass="A",
                                hypothesis="_split_clues keeps every individually-valid reading "
                                           "without requiring one consistent partition.",
                                invariant="Base-rate-weighted TechnicalScore impact < 0.003. "
                                          "ENRICHED SAMPLING -- report enriched AND base-rate figures.")

    def f7(self, total=750) -> List[TestCase]:
        base, extra = divmod(total, len(TR.OOD_CATEGORIES))
        pool = self.t1("C").draw(max(total, 400), 606).asins
        out, cursor = [], 0
        for cat_index, category in enumerate(TR.OOD_CATEGORIES):
            n = base + (1 if cat_index < extra else 0)
            for i in range(n):
                asin = pool[cursor % len(pool)]
                cursor += 1
                scenario = scenario_cycle(cursor)[-1]
                clean = self.b.clean_inputs(asin, scenario, f"F7_{category}_{i}")
                turn = TR.ood_turn(category)
                text = TR.ood_message(category, i, clean["coarse_category"])
                transformed = dict(clean)
                if turn == 1:
                    transformed["opening_message"] = text
                out.append(self._mk(
                    family="F7", cell_id=f"F7/{category}", klass="C", tier="T1",
                    severity="ood", asin=asin, scenario=scenario, index=i, seed=606,
                    scheme="C", ood=True, excluded_from_headline=True, transformed=transformed,
                    transformation={"surface": "ood", "kind": category, "severity": "ood",
                                    "template_variant": {}, "constraint_map": {},
                                    "scripted_turns": {str(turn): text}},
                    hypothesis="The category+popularity floor needs no constraint parsing, so "
                               "language the simulator cannot produce degrades gracefully.",
                    invariant="OOD -- MUST NOT be used to project private-evaluator performance. "
                              "HitRate@10 should stay above the 0.832 floor probe.",
                ))
        return out

    def f8(self, n=800) -> List[TestCase]:
        draw = self.t1("C").draw(n, 8008)
        scen = scenario_cycle(n)
        return [self._mk(
            family="F8", cell_id="F8/soak", klass="B", tier="T1", severity="none",
            asin=asin, scenario=scen[i], index=i, seed=8008, scheme="C",
            hypothesis="One Agent instance serves all sessions; state, memory and determinism "
                       "must hold across 800 sequential resets.",
            invariant="Byte-identical score across PYTHONHASHSEED values; no memory growth; "
                      "no cross-session leakage.",
        ) for i, asin in enumerate(draw.asins)]

    # ---- orchestration ---------------------------------------------------

    FULL = {"F1a": 250, "F1b": 40, "F1c": 30, "F1d": 30, "F1e": 30, "F1f": 50, "F1g": 40,
            "F2A": 1700, "F2B": 500, "F3": 100, "F4": 15, "F5": 100, "F6": 30, "F7": 750, "F8": 800}
    VALIDATION = {"F1a": 4, "F1b": 2, "F1c": 2, "F1d": 2, "F1e": 2, "F1f": 2, "F1g": 2,
                  "F2A": 108, "F2B": 30, "F3": 2, "F4": 2, "F5": 2, "F6": 2, "F7": 16, "F8": 4}

    def generate(self, sizes: Dict[str, int]) -> List[TestCase]:
        cases: List[TestCase] = []
        cases += self.f1a(per_draw=sizes["F1a"])
        cases += self.f1b(sizes["F1b"])
        cases += self.f1c(sizes["F1c"])
        cases += self.f1d(sizes["F1d"])
        cases += self.f1e(sizes["F1e"])
        cases += self.f1f(sizes["F1f"])
        cases += self.f1g(sizes["F1g"])
        cases += self.f2a(sizes["F2A"])
        cases += self.f2b(sizes["F2B"])
        cases += self.f3(sizes["F3"])
        cases += self.f4(sizes["F4"])
        cases += self.f5(sizes["F5"])
        cases += self.f6(sizes["F6"])
        cases += self.f7(sizes["F7"])
        cases += self.f8(sizes["F8"])
        # Deterministic row order, independent of construction order.
        cases.sort(key=lambda c: (c.family, c.cell_id, c.sample_id, c.test_id))
        return cases


def write_cases(cases: Sequence[TestCase], path: str | Path, overwrite: bool = False) -> Path:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"{target} already exists. Refusing to overwrite silently -- pass --overwrite "
            f"or move the existing suite aside."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" is load-bearing: the frozen SHA256 is over LF-terminated bytes,
    # and the default text mode rewrites "\n" to "\r\n" on Windows, so the same
    # deterministic cases hash differently there and preflight aborts.
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(json.dumps(case.as_row(), ensure_ascii=False, sort_keys=True) + "\n")
    return target
