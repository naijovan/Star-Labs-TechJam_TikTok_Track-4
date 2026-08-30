"""T1 calibration schemes, diagnostics, the rejection rule, and the core/tail draw.

Two schemes are implemented and BOTH are always reported.  They must never be
averaged: the spread between them is the calibration uncertainty, and collapsing
it to one number is the specific failure this module exists to prevent.

    B  coarse joint       rank-tercile x pop-tercile x df-binary x has_price
    C  nearest-neighbour  k=25, price-matched, standardised log-space

Scheme A (rank-decile x pop-decile x bsize-tercile x df-stratum x price) is
implemented ONLY so the rejection rule can be demonstrated to fire on it.  It is
not usable: 23% of target mass falls in zero-donor cells and leave-out
generalisation is 29/100.  See TEST_MATRIX.md §0.5.
"""
from __future__ import annotations

import collections
import random
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

from tools.suite.strata import (
    CatalogFeatures,
    _bin_of,
    _quantile_cuts,
    rarest_df_binary,
    rarest_df_stratum,
)

# Rejection rule, exactly as approved.
MAX_UNMATCHED_MASS = 0.05
MIN_LEAVE_OUT = 80
LEAVE_OUT_SPLITS = 5
LEAVE_OUT_BASE_SEED = 100
DEFAULT_ESS_DRAW = 400

# Observed core/tail mass is computed from the data, never hardcoded; this is
# only the value we expect to see (169/200).
EXPECTED_CORE_MASS = 0.845


@dataclass
class Diagnostics:
    scheme: str
    n_targets: int
    n_donors: int
    occupied_cells: int
    zero_donor_cells: int
    unmatched_mass: float
    donors_min: int
    donors_median: float
    donors_p90: float
    ess: float
    ess_draw: int
    leave_out_scores: List[int] = field(default_factory=list)
    leave_out_mean: float = 0.0
    leave_out_informative: bool = True
    coverage_note: str = ""
    accepted: bool = True
    reject_reasons: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _kish_ess(draw: Sequence[str]) -> float:
    """Effective sample size of a draw, penalising donor reuse."""
    counts = collections.Counter(draw)
    total = sum(counts.values())
    return total * total / sum(v * v for v in counts.values())


class Scheme(ABC):
    """A calibration scheme maps each observed target to a donor set."""

    name = "abstract"

    def __init__(self, features: CatalogFeatures, targets: Sequence[str], donors: Sequence[str]) -> None:
        self.features = features
        self.targets = list(targets)
        self.donors = list(donors)

    @abstractmethod
    def donors_for(self, target: str) -> List[str]:
        """Donors matched to one target.  May be empty (that is the failure the
        unmatched-mass check measures)."""

    @abstractmethod
    def _structural_diagnostics(self) -> dict:
        ...

    @abstractmethod
    def _leave_out(self) -> Tuple[List[int], bool, str]:
        ...

    def draw(self, n: int, seed: int) -> List[str]:
        """Draw n donors reproducing the target distribution.  Deterministic in
        (n, seed) given the same catalogue and target list."""
        rng = random.Random(seed)
        matched = [t for t in self.targets if self.donors_for(t)]
        if not matched:
            raise ValueError(f"scheme {self.name}: no target has a donor")
        return [rng.choice(self.donors_for(rng.choice(matched))) for _ in range(n)]

    def diagnostics(self, ess_draw: int = DEFAULT_ESS_DRAW, ess_seed: int = 7) -> Diagnostics:
        structural = self._structural_diagnostics()
        leave_out_scores, informative, note = self._leave_out()
        leave_out_mean = statistics.mean(leave_out_scores) if leave_out_scores else 0.0
        ess = _kish_ess(self.draw(ess_draw, ess_seed))

        reasons: List[str] = []
        if structural["unmatched_mass"] > MAX_UNMATCHED_MASS:
            reasons.append(
                f"unmatched target mass {structural['unmatched_mass']:.1%} > {MAX_UNMATCHED_MASS:.0%}"
            )
        if informative and leave_out_mean < MIN_LEAVE_OUT:
            reasons.append(f"leave-out {leave_out_mean:.0f}/100 < {MIN_LEAVE_OUT}/100")

        return Diagnostics(
            scheme=self.name,
            n_targets=len(self.targets),
            ess=ess,
            ess_draw=ess_draw,
            leave_out_scores=leave_out_scores,
            leave_out_mean=leave_out_mean,
            leave_out_informative=informative,
            coverage_note=note,
            accepted=not reasons,
            reject_reasons=reasons,
            **structural,
        )


class CellScheme(Scheme):
    """Shared machinery for the cell-based schemes (A and B)."""

    def __init__(self, features, targets, donors) -> None:
        super().__init__(features, targets, donors)
        self._cuts = self._build_cuts()
        self._target_cells = collections.Counter(self.cell(t) for t in self.targets)
        donors_by_cell: Dict[tuple, List[str]] = collections.defaultdict(list)
        for asin in self.donors:  # self.donors is already sorted -> deterministic
            donors_by_cell[self.cell(asin)].append(asin)
        self._donors_by_cell = donors_by_cell

    @abstractmethod
    def _build_cuts(self) -> dict:
        ...

    @abstractmethod
    def cell(self, asin: str) -> tuple:
        ...

    def donors_for(self, target: str) -> List[str]:
        return self._donors_by_cell.get(self.cell(target), [])

    def _structural_diagnostics(self) -> dict:
        occupied = sorted(self._target_cells)
        zero = [c for c in occupied if not self._donors_by_cell.get(c)]
        unmatched = sum(self._target_cells[c] for c in zero) / max(len(self.targets), 1)
        counts = sorted(len(self._donors_by_cell[c]) for c in occupied if self._donors_by_cell.get(c))
        if not counts:
            counts = [0]
        return {
            "n_donors": len(self.donors),
            "occupied_cells": len(occupied),
            "zero_donor_cells": len(zero),
            "unmatched_mass": unmatched,
            "donors_min": counts[0],
            "donors_median": statistics.median(counts),
            "donors_p90": counts[min(len(counts) - 1, int(0.9 * len(counts)))],
        }

    def _leave_out(self) -> Tuple[List[int], bool, str]:
        """Fit the cell structure on half the targets; count how many held-out
        targets land in a cell the fitting half occupied."""
        scores: List[int] = []
        half = len(self.targets) // 2
        for trial in range(LEAVE_OUT_SPLITS):
            shuffled = list(self.targets)
            random.Random(LEAVE_OUT_BASE_SEED + trial).shuffle(shuffled)
            fit, held = shuffled[:half], shuffled[half:]
            occupied = {self.cell(a) for a in fit}
            hits = sum(1 for a in held if self.cell(a) in occupied)
            scores.append(round(100 * hits / max(len(held), 1)))
        return scores, True, "cell-structure generalisation across target halves"


class FineJointScheme(CellScheme):
    """Scheme A -- retained only to demonstrate the rejection rule firing."""

    name = "A fine joint"

    def _build_cuts(self) -> dict:
        f = self.features
        return {
            "rank": _quantile_cuts([f.bucket_rank(a) for a in self.targets], 10),
            "pop": _quantile_cuts([f.rating_number(a) for a in self.targets], 10),
            "bsize": _quantile_cuts([f.bucket_size(a) for a in self.targets], 3),
        }

    def cell(self, asin: str) -> tuple:
        f, c = self.features, self._cuts
        return (
            _bin_of(f.bucket_rank(asin), c["rank"]),
            _bin_of(f.rating_number(asin), c["pop"]),
            _bin_of(f.bucket_size(asin), c["bsize"]),
            rarest_df_stratum(f.rarest_df(asin)),
            int(f.has_price(asin)),
        )


class CoarseJointScheme(CellScheme):
    """Scheme B -- rank-tercile x pop-tercile x df-binary x has_price."""

    name = "B coarse joint"

    def _build_cuts(self) -> dict:
        f = self.features
        return {
            "rank": _quantile_cuts([f.bucket_rank(a) for a in self.targets], 3),
            "pop": _quantile_cuts([f.rating_number(a) for a in self.targets], 3),
        }

    def cell(self, asin: str) -> tuple:
        f, c = self.features, self._cuts
        return (
            _bin_of(f.bucket_rank(asin), c["rank"]),
            _bin_of(f.rating_number(asin), c["pop"]),
            rarest_df_binary(f.rarest_df(asin)),
            int(f.has_price(asin)),
        )


class NearestNeighbourScheme(Scheme):
    """Scheme C -- k nearest donors per target in standardised log-space,
    matched on has_price.

    Leave-out is NON-INFORMATIVE for this scheme by construction: every target
    always has k donors, so a cell-occupancy check trivially returns 100/100.
    We report that honestly and add a real diagnostic instead -- the coverage
    radius, i.e. how far a held-out target sits from the nearest fitting target
    relative to the scheme's own matching radius.  A coverage ratio far above 1
    means the scheme is extrapolating rather than interpolating.
    """

    name = "C nearest-neighbour k=25"

    def __init__(self, features, targets, donors, k: int = 25) -> None:
        super().__init__(features, targets, donors)
        self.k = k
        self._z = features.standardiser(self.donors)
        by_price: Dict[bool, List[str]] = collections.defaultdict(list)
        for asin in self.donors:
            by_price[features.has_price(asin)].append(asin)
        self._by_price = by_price
        self._cache: Dict[str, List[str]] = {}
        self._radius: Dict[str, float] = {}

    def _dist(self, a: str, b: str) -> float:
        return sum((p - q) ** 2 for p, q in zip(self._z(a), self._z(b)))

    def donors_for(self, target: str) -> List[str]:
        if target not in self._cache:
            candidates = self._by_price[self.features.has_price(target)]
            scored = sorted((self._dist(target, d), d) for d in candidates)[: self.k]
            self._cache[target] = [d for _, d in scored]
            self._radius[target] = scored[-1][0] if scored else float("inf")
        return self._cache[target]

    def _structural_diagnostics(self) -> dict:
        matched = [t for t in self.targets if self.donors_for(t)]
        counts = sorted(len(self.donors_for(t)) for t in matched) or [0]
        return {
            "n_donors": len(self.donors),
            "occupied_cells": len(self.targets),
            "zero_donor_cells": len(self.targets) - len(matched),
            "unmatched_mass": (len(self.targets) - len(matched)) / max(len(self.targets), 1),
            "donors_min": counts[0],
            "donors_median": statistics.median(counts),
            "donors_p90": counts[min(len(counts) - 1, int(0.9 * len(counts)))],
        }

    def _leave_out(self) -> Tuple[List[int], bool, str]:
        ratios: List[float] = []
        half = len(self.targets) // 2
        for trial in range(LEAVE_OUT_SPLITS):
            shuffled = list(self.targets)
            random.Random(LEAVE_OUT_BASE_SEED + trial).shuffle(shuffled)
            fit, held = shuffled[:half], shuffled[half:]
            for target in held:
                nearest = min(self._dist(target, f) for f in fit)
                radius = self._radius.get(target) or self.donors_for(target) and self._radius[target]
                ratios.append(nearest / radius if radius else float("inf"))
        median_ratio = statistics.median(ratios) if ratios else float("inf")
        note = (
            f"coverage ratio (median nearest-fit-target distance / own k={self.k} matching "
            f"radius) = {median_ratio:.2f}; <=1 means interpolation, >>1 extrapolation"
        )
        # Cell-occupancy leave-out is vacuous here; report 100 but flag it.
        return [100] * LEAVE_OUT_SPLITS, False, note


@dataclass
class MixtureDraw:
    asins: List[str]
    core_mass: float
    n_core: int
    n_tail: int
    scheme: str
    seed: int


class T1Mixture:
    """Core/tail mixture at the OBSERVED mass split.

    Core targets are modelled with strict donors; tail targets with relaxed
    donors (price and popularity floors dropped -- the only two criteria any
    public target violates).  The mass split is measured from the targets, not
    hardcoded, so it stays correct if the definition of `is_strict` changes.
    """

    def __init__(
        self,
        features: CatalogFeatures,
        targets: Sequence[str],
        scheme_cls,
        exclude_targets: bool = True,
        **scheme_kwargs,
    ) -> None:
        self.features = features
        self.targets = list(targets)
        excluded = self.targets if exclude_targets else []

        self.core_targets = [t for t in self.targets if features.is_strict(t)]
        self.tail_targets = [t for t in self.targets if not features.is_strict(t)]
        self.core_mass = len(self.core_targets) / max(len(self.targets), 1)

        core_donors = features.donor_pool(exclude=excluded, strict=True)
        tail_donors = features.donor_pool(exclude=excluded, strict=False)

        self.core_scheme = scheme_cls(features, self.core_targets, core_donors, **scheme_kwargs)
        self.tail_scheme = (
            scheme_cls(features, self.tail_targets, tail_donors, **scheme_kwargs)
            if self.tail_targets
            else None
        )
        self.scheme_name = self.core_scheme.name

    def draw(self, n: int, seed: int) -> MixtureDraw:
        n_core = round(n * self.core_mass)
        n_tail = n - n_core
        asins = self.core_scheme.draw(n_core, seed)
        if n_tail and self.tail_scheme is not None:
            asins = asins + self.tail_scheme.draw(n_tail, seed + 1)
        elif n_tail:
            asins = asins + self.core_scheme.draw(n_tail, seed + 1)
        random.Random(seed).shuffle(asins)
        return MixtureDraw(
            asins=asins,
            core_mass=self.core_mass,
            n_core=n_core,
            n_tail=n_tail,
            scheme=self.scheme_name,
            seed=seed,
        )

    def diagnostics(self, **kwargs) -> Dict[str, Diagnostics]:
        out = {"core": self.core_scheme.diagnostics(**kwargs)}
        if self.tail_scheme is not None:
            out["tail"] = self.tail_scheme.diagnostics(**kwargs)
        return out


ACCEPTED_SCHEMES = {"B": CoarseJointScheme, "C": NearestNeighbourScheme}
ALL_SCHEMES = {"A": FineJointScheme, **ACCEPTED_SCHEMES}


def stress_pool(features: CatalogFeatures, exclude: Sequence[str], kind: str) -> List[str]:
    """T3 stress bounds.  NEVER an estimate -- callers must label these."""
    excluded = set(exclude)
    eligible = [a for a in features.asins if a not in excluded]
    if kind == "uniform_50k":
        return eligible
    if kind == "unpopular_half":
        ordered = sorted(eligible, key=lambda a: (-features.rating_number(a), a))
        return sorted(ordered[len(ordered) // 2 :])
    raise ValueError(f"unknown stress pool {kind!r}")
