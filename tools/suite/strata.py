"""Per-product features and stratum assignment over the frozen catalogue.

Everything here is READ-ONLY with respect to the catalogue and the production
agent.  Product statistics are computed by calling the organizer's own
`intent_card()` and `coarse_category()` so that the suite can never drift from
the evaluator the way a re-implementation would.

Definitions used throughout (they are the ones `TEST_MATRIX.md` §0 measured):

    rating_number   popularity prior the agent sorts by
    average_rating  eligibility-ish signal; no public target is below 3.5
    n_constraints   |set(hard_constraints + soft_preferences)| -- the number of
                    distinct things the shopper can ever disclose
    rarest_df       min over the product's emitted constraints of how many
                    catalogue products emit that same string.  rarest_df == 1
                    means one harvested clue identifies the product outright.
    bucket          coarse_category(categories) -- leaked verbatim on turn 1
    bucket_rank     1-based position inside the bucket sorted by
                    (-rating_number, parent_asin), i.e. exactly the ordering the
                    agent's popularity floor produces
"""
from __future__ import annotations

import collections
import json
import math
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from evaluator.local_evaluator import coarse_category, intent_card

# Criteria that define the "core" of the observed target population.  Measured in
# MEASUREMENT_LOG.md: 169 of the 200 public targets satisfy all five.  Of the 31
# that do not, 22 fail only `has_price` and 10 only `rating_number >= 100`; ZERO
# fail has_features, n_constraints == 4, or average_rating >= 3.5.
STRICT_MIN_AVG_RATING = 3.5
STRICT_MIN_RATING_NUMBER = 100

# Relaxed donor criteria for the tail: drop exactly the two conditions that
# public targets are observed to violate, keep the three that none violate.
RELAXED_MIN_AVG_RATING = 3.5


@dataclass(frozen=True)
class Stratum:
    """The stratum coordinates of one product.  Serialised into every test case."""

    parent_asin: str
    rating_number: int
    average_rating: float
    n_constraints: int
    rarest_df: int
    bucket: str
    bucket_size: int
    bucket_rank: int
    has_features: bool
    has_price: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _quantile_cuts(values: Sequence[float], k: int) -> List[float]:
    """k-quantile cut points.  Deterministic; ties collapse cuts, which is fine
    (a collapsed cut just yields an empty bin, never a wrong assignment)."""
    ordered = sorted(values)
    return [ordered[min(len(ordered) - 1, int(round(len(ordered) * i / k)))] for i in range(1, k)]


def _bin_of(value: float, cuts: Sequence[float]) -> int:
    for index, cut in enumerate(cuts):
        if value < cut:
            return index
    return len(cuts)


def rarest_df_stratum(rarest_df: int) -> int:
    """4-level rarity stratum: unique / near-unique / mid / common-only."""
    if rarest_df == 1:
        return 0
    if rarest_df <= 10:
        return 1
    if rarest_df <= 1000:
        return 2
    return 3


def rarest_df_binary(rarest_df: int) -> int:
    return 0 if rarest_df <= 10 else 1


class CatalogFeatures:
    """Loads the catalogue once and exposes per-product strata.

    Deliberately does NOT cache to disk.  The team rejected a pickle cache for
    the agent on stale-cache grounds (`PEER_REVIEW.md`, Ben's `.agent_cache.pkl`)
    and the same argument applies here: a suite that silently strata-fies against
    a stale catalogue would invalidate every downstream number.  Build time is
    ~25 s; callers should construct once and pass the instance around.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = str(catalog_path)
        products: Dict[str, dict] = {}
        constraints: Dict[str, frozenset] = {}
        buckets: Dict[str, List[str]] = collections.defaultdict(list)

        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                asin = str(product["parent_asin"])
                products[asin] = product
                card = intent_card(product)
                constraints[asin] = frozenset(card["hard_constraints"]) | frozenset(card["soft_preferences"])
                buckets[coarse_category([str(v) for v in product.get("categories") or []])].append(asin)

        self.asins: List[str] = sorted(products)
        self._products = products
        self._constraints = constraints

        document_frequency: collections.Counter = collections.Counter()
        for asin in self.asins:
            document_frequency.update(constraints[asin])
        self.document_frequency = document_frequency

        self.bucket_members: Dict[str, List[str]] = {}
        self._bucket_of: Dict[str, str] = {}
        for name in sorted(buckets):
            # Exactly the ordering the agent's popularity floor produces.
            members = sorted(
                buckets[name],
                key=lambda a: (-(products[a].get("rating_number") or 0), a),
            )
            self.bucket_members[name] = members
            for asin in members:
                self._bucket_of[asin] = name

        self._bucket_rank: Dict[str, int] = {}
        for name, members in self.bucket_members.items():
            for position, asin in enumerate(members, start=1):
                self._bucket_rank[asin] = position

    # ---------------- per-product accessors ----------------

    def rating_number(self, asin: str) -> int:
        return self._products[asin].get("rating_number") or 0

    def average_rating(self, asin: str) -> float:
        return self._products[asin].get("average_rating") or 0.0

    def constraints(self, asin: str) -> frozenset:
        return self._constraints[asin]

    def n_constraints(self, asin: str) -> int:
        return len(self._constraints[asin])

    def rarest_df(self, asin: str) -> int:
        return min(self.document_frequency[c] for c in self._constraints[asin])

    def bucket(self, asin: str) -> str:
        return self._bucket_of[asin]

    def bucket_size(self, asin: str) -> int:
        return len(self.bucket_members[self._bucket_of[asin]])

    def bucket_rank(self, asin: str) -> int:
        return self._bucket_rank[asin]

    def has_features(self, asin: str) -> bool:
        return bool(self._products[asin].get("features"))

    def has_price(self, asin: str) -> bool:
        return self._products[asin].get("price") not in (None, "")

    def stratum(self, asin: str) -> Stratum:
        return Stratum(
            parent_asin=asin,
            rating_number=self.rating_number(asin),
            average_rating=self.average_rating(asin),
            n_constraints=self.n_constraints(asin),
            rarest_df=self.rarest_df(asin),
            bucket=self.bucket(asin),
            bucket_size=self.bucket_size(asin),
            bucket_rank=self.bucket_rank(asin),
            has_features=self.has_features(asin),
            has_price=self.has_price(asin),
        )

    # ---------------- population membership ----------------

    def is_strict(self, asin: str) -> bool:
        """The 'core' target signature (169/200 public targets)."""
        return (
            self.has_features(asin)
            and self.has_price(asin)
            and self.n_constraints(asin) == 4
            and self.average_rating(asin) >= STRICT_MIN_AVG_RATING
            and self.rating_number(asin) >= STRICT_MIN_RATING_NUMBER
        )

    def is_relaxed(self, asin: str) -> bool:
        """Donor eligibility for the tail: the three criteria NO public target
        violates.  Price and the popularity floor are deliberately absent."""
        return (
            self.has_features(asin)
            and self.n_constraints(asin) == 4
            and self.average_rating(asin) >= RELAXED_MIN_AVG_RATING
        )

    def donor_pool(self, exclude: Iterable[str], strict: bool) -> List[str]:
        """Public-target exclusion is applied here and nowhere else, so it cannot
        be forgotten by a caller."""
        excluded = set(exclude)
        test = self.is_strict if strict else self.is_relaxed
        return [a for a in self.asins if a not in excluded and test(a)]

    # ---------------- standardised feature space (scheme C) ----------------

    def log_features(self, asin: str) -> tuple:
        return (
            math.log1p(self.rating_number(asin)),
            math.log(self.bucket_rank(asin)),
            math.log(self.bucket_size(asin)),
            math.log(self.rarest_df(asin)),
        )

    def standardiser(self, asins: Sequence[str]):
        """Mean/pstdev over a reference population; returns a z() function."""
        columns = list(zip(*[self.log_features(a) for a in asins]))
        means = [statistics.mean(c) for c in columns]
        stdevs = [statistics.pstdev(c) or 1.0 for c in columns]

        def z(asin: str) -> tuple:
            return tuple((v - m) / s for v, m, s in zip(self.log_features(asin), means, stdevs))

        return z


def load_public_targets(dataset_path: str | Path = "data/public_set.jsonl") -> List[str]:
    """Order preserved as it appears in the file -- leave-out splits are seeded
    against this order, so it must not be sorted."""
    targets: List[str] = []
    with Path(dataset_path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                targets.append(str(json.loads(line)["ground_truth"]["parent_asin"]))
    return targets
