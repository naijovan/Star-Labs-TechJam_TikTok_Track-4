"""Stage 1 tests: tools/suite/strata.py and tools/suite/pool.py.

The catalogue is built ONCE for the whole module (~25 s) and shared.

These tests assert two different kinds of thing and it is worth keeping them
straight:

  * CONTRACT tests -- determinism, exclusion, mass preservation, the rejection
    rule.  These must hold for any catalogue.
  * REPRODUCTION tests -- the specific numbers recorded in TEST_MATRIX.md §0.5.
    These pin the suite to the measurements the design was approved against, so
    that a silent change to strata definitions is caught immediately.
"""
from __future__ import annotations

import math
import random
import unittest
from pathlib import Path

from tools.suite import SUITE_VERSION
from tools.suite.pool import (
    ACCEPTED_SCHEMES,
    CoarseJointScheme,
    FineJointScheme,
    MAX_UNMATCHED_MASS,
    MIN_LEAVE_OUT,
    NearestNeighbourScheme,
    T1Mixture,
    _kish_ess,
    stress_pool,
)
from tools.suite.strata import CatalogFeatures, load_public_targets

CATALOG = Path("data/catalog.jsonl")
DATASET = Path("data/public_set.jsonl")

_FEATURES = None
_TARGETS = None


def setUpModule() -> None:
    global _FEATURES, _TARGETS
    if not CATALOG.exists():
        raise unittest.SkipTest(f"{CATALOG} not present -- download it before running Stage 1 tests")
    _FEATURES = CatalogFeatures(CATALOG)
    _TARGETS = load_public_targets(DATASET)


class TestCatalogFeatures(unittest.TestCase):
    def test_catalog_shape(self):
        self.assertEqual(len(_FEATURES.asins), 50000)
        self.assertEqual(len(_TARGETS), 200)
        self.assertEqual(len(set(_TARGETS)), 200, "public targets must be distinct")

    def test_asins_sorted_for_determinism(self):
        self.assertEqual(_FEATURES.asins, sorted(_FEATURES.asins))

    def test_bucket_rank_matches_agent_floor_ordering(self):
        """bucket_rank must reproduce sorted(bucket, key=(-rating_number, asin)),
        because that is exactly what the agent's popularity floor emits."""
        name = max(_FEATURES.bucket_members, key=lambda k: len(_FEATURES.bucket_members[k]))
        members = _FEATURES.bucket_members[name]
        expected = sorted(members, key=lambda a: (-_FEATURES.rating_number(a), a))
        self.assertEqual(members, expected)
        self.assertEqual(_FEATURES.bucket_rank(members[0]), 1)
        self.assertEqual(_FEATURES.bucket_rank(members[-1]), len(members))

    def test_measured_catalog_statistics(self):
        """Reproduces MEASUREMENT_LOG.md section 1 (catalogue structure)."""
        self.assertEqual(len(_FEATURES.bucket_members), 1115)
        no_features = sum(1 for a in _FEATURES.asins if not _FEATURES.has_features(a))
        self.assertEqual(no_features, 5219)
        priced = sum(1 for a in _FEATURES.asins if _FEATURES.has_price(a))
        self.assertEqual(priced, 10527)
        self.assertEqual(len(_FEATURES.document_frequency), 60670)
        unique = sum(1 for v in _FEATURES.document_frequency.values() if v == 1)
        self.assertEqual(unique, 55411)
        self.assertEqual(_FEATURES.document_frequency["Imported"], 13633)

    def test_measured_target_population(self):
        """Reproduces TEST_MATRIX.md section 0.1 -- the narrow target signature."""
        self.assertTrue(all(_FEATURES.has_features(a) for a in _TARGETS))
        self.assertTrue(all(_FEATURES.n_constraints(a) == 4 for a in _TARGETS))
        self.assertEqual(sum(1 for a in _TARGETS if _FEATURES.has_price(a)), 178)

    def test_core_tail_split(self):
        """Reproduces TEST_MATRIX.md section 0.6."""
        core = [a for a in _TARGETS if _FEATURES.is_strict(a)]
        tail = [a for a in _TARGETS if not _FEATURES.is_strict(a)]
        self.assertEqual(len(core), 169)
        self.assertEqual(len(tail), 31)
        self.assertEqual(sum(1 for a in tail if not _FEATURES.has_price(a)), 22)
        self.assertEqual(sum(1 for a in tail if _FEATURES.rating_number(a) < 100), 10)
        # No target fails the three criteria the relaxed donor pool keeps.
        for a in tail:
            self.assertTrue(_FEATURES.is_relaxed(a), f"{a} should satisfy relaxed criteria")

    def test_donor_pool_excludes_public_targets(self):
        pool = set(_FEATURES.donor_pool(exclude=_TARGETS, strict=False))
        self.assertEqual(len(pool & set(_TARGETS)), 0, "public targets must never be donors")
        self.assertEqual(len(pool), 38020)

    def test_strict_donor_pool_size(self):
        pool = _FEATURES.donor_pool(exclude=[], strict=True)
        self.assertEqual(len(pool), 2551)

    def test_stratum_is_serialisable(self):
        d = _FEATURES.stratum(_TARGETS[0]).as_dict()
        for key in ("parent_asin", "rating_number", "bucket", "bucket_rank", "rarest_df"):
            self.assertIn(key, d)


class TestSchemeDiagnostics(unittest.TestCase):
    """Reproduces the TEST_MATRIX.md section 0.5 diagnostics table."""

    @classmethod
    def setUpClass(cls):
        donors = _FEATURES.donor_pool(exclude=_TARGETS, strict=False)
        cls.donors = donors
        cls.A = FineJointScheme(_FEATURES, _TARGETS, donors)
        cls.B = CoarseJointScheme(_FEATURES, _TARGETS, donors)
        cls.C = NearestNeighbourScheme(_FEATURES, _TARGETS, donors)

    def test_scheme_a_is_rejected(self):
        d = self.A.diagnostics()
        self.assertGreater(d.unmatched_mass, MAX_UNMATCHED_MASS)
        self.assertLess(d.leave_out_mean, MIN_LEAVE_OUT)
        self.assertFalse(d.accepted)
        self.assertEqual(len(d.reject_reasons), 2, d.reject_reasons)

    def test_scheme_a_reproduces_measured_sparsity(self):
        d = self.A.diagnostics()
        self.assertEqual(d.occupied_cells, 151)
        self.assertEqual(d.zero_donor_cells, 38)
        self.assertAlmostEqual(d.unmatched_mass, 0.230, places=3)

    def test_scheme_b_is_accepted_and_reproduces(self):
        d = self.B.diagnostics()
        self.assertEqual(d.occupied_cells, 20)
        self.assertEqual(d.zero_donor_cells, 0)
        self.assertEqual(d.unmatched_mass, 0.0)
        self.assertGreaterEqual(d.leave_out_mean, MIN_LEAVE_OUT)
        self.assertTrue(d.accepted, d.reject_reasons)

    def test_scheme_c_is_accepted_and_flags_its_own_leave_out(self):
        d = self.C.diagnostics()
        self.assertEqual(d.zero_donor_cells, 0)
        self.assertEqual(d.unmatched_mass, 0.0)
        self.assertEqual(d.donors_min, 25)
        self.assertFalse(d.leave_out_informative, "C's cell leave-out must be flagged vacuous")
        self.assertIn("coverage ratio", d.coverage_note)
        self.assertTrue(d.accepted)

    def test_accepted_schemes_registry(self):
        self.assertEqual(set(ACCEPTED_SCHEMES), {"B", "C"})
        self.assertNotIn("A", ACCEPTED_SCHEMES)

    def test_draws_are_deterministic(self):
        for scheme in (self.B, self.C):
            self.assertEqual(scheme.draw(200, 11), scheme.draw(200, 11))
            self.assertNotEqual(scheme.draw(200, 11), scheme.draw(200, 12))

    def test_draws_never_contain_public_targets(self):
        for scheme in (self.B, self.C):
            self.assertEqual(set(scheme.draw(400, 5)) & set(_TARGETS), set())

    def test_ess_bounds(self):
        self.assertAlmostEqual(_kish_ess(["a"] * 10), 1.0)
        self.assertAlmostEqual(_kish_ess([str(i) for i in range(10)]), 10.0)


class TestT1Mixture(unittest.TestCase):
    def test_mass_is_measured_not_hardcoded(self):
        m = T1Mixture(_FEATURES, _TARGETS, CoarseJointScheme)
        self.assertAlmostEqual(m.core_mass, 169 / 200)
        self.assertEqual(len(m.core_targets), 169)
        self.assertEqual(len(m.tail_targets), 31)

    def test_draw_preserves_tail_mass(self):
        for cls in (CoarseJointScheme, NearestNeighbourScheme):
            m = T1Mixture(_FEATURES, _TARGETS, cls)
            d = m.draw(400, 42)
            self.assertEqual(len(d.asins), 400)
            self.assertEqual(d.n_core, round(400 * 169 / 200))
            self.assertEqual(d.n_tail, 400 - d.n_core)
            self.assertGreater(d.n_tail, 0, "the 15.5% tail must never be dropped")

    def test_tail_donors_are_relaxed_not_strict(self):
        m = T1Mixture(_FEATURES, _TARGETS, CoarseJointScheme)
        strict_only = set(_FEATURES.donor_pool(exclude=_TARGETS, strict=True))
        tail_donors = set(m.tail_scheme.donors)
        self.assertTrue(tail_donors - strict_only, "tail donors must include non-strict products")

    def test_mixture_draw_deterministic_and_excludes_targets(self):
        m = T1Mixture(_FEATURES, _TARGETS, NearestNeighbourScheme)
        a, b = m.draw(300, 7), m.draw(300, 7)
        self.assertEqual(a.asins, b.asins)
        self.assertEqual(set(a.asins) & set(_TARGETS), set())

    def test_b_and_c_produce_different_draws(self):
        """The two schemes must not silently collapse to the same thing --
        their disagreement is the calibration uncertainty we report."""
        b = T1Mixture(_FEATURES, _TARGETS, CoarseJointScheme).draw(400, 3).asins
        c = T1Mixture(_FEATURES, _TARGETS, NearestNeighbourScheme).draw(400, 3).asins
        overlap = len(set(b) & set(c)) / len(set(b) | set(c))
        self.assertLess(overlap, 0.5, f"schemes suspiciously similar (Jaccard {overlap:.2f})")


class TestStressPools(unittest.TestCase):
    def test_stress_pools_exclude_targets_and_are_deterministic(self):
        for kind in ("uniform_50k", "unpopular_half"):
            p1 = stress_pool(_FEATURES, _TARGETS, kind)
            p2 = stress_pool(_FEATURES, _TARGETS, kind)
            self.assertEqual(p1, p2)
            self.assertEqual(set(p1) & set(_TARGETS), set())

    def test_unpopular_half_is_actually_less_popular(self):
        half = stress_pool(_FEATURES, _TARGETS, "unpopular_half")
        allp = stress_pool(_FEATURES, _TARGETS, "uniform_50k")
        med_half = sorted(_FEATURES.rating_number(a) for a in half)[len(half) // 2]
        med_all = sorted(_FEATURES.rating_number(a) for a in allp)[len(allp) // 2]
        self.assertLess(med_half, med_all)

    def test_unknown_stress_pool_raises(self):
        with self.assertRaises(ValueError):
            stress_pool(_FEATURES, _TARGETS, "nope")


class TestSuiteMetadata(unittest.TestCase):
    def test_version_pinned(self):
        self.assertEqual(SUITE_VERSION, "1.0.0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
