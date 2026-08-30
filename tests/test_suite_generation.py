"""Stage 2 tests: deterministic test-case generation.

Built once against the VALIDATION batch (269 cases), which contains at least one
case from every family and every cell type.
"""
from __future__ import annotations

import collections
import json
import tempfile
import unittest
from pathlib import Path

from tools.suite import SUITE_VERSION, transforms as TR
from tools.suite.sessions import CaseBuilder, SuiteGenerator, scenario_cycle, write_cases
from tools.suite.strata import CatalogFeatures, load_public_targets

CATALOG = Path("data/catalog.jsonl")
DATASET = Path("data/public_set.jsonl")

_CASES = None
_ROWS = None
_FEATURES = None
_TARGETS = None

REQUIRED_KEYS = {
    "test_id", "suite_version", "family", "cell_id", "class", "tier", "severity",
    "scheme", "target_parent_asin", "target_stratum", "scenario_type", "user_profile",
    "sample_id", "seed", "clean_inputs", "transformed_inputs", "transformation",
    "hypothesis", "expected_invariant", "sentinel", "excluded_from_headline",
    "stress_bound", "ood", "arms", "config_grid", "n_sessions",
    "requested_severity", "achieved_severity", "generation_attempts", "downgraded",
    "diagnostic",
}

# Families drawn from public-unseen donors only.
PUBLIC_UNSEEN_FAMILIES = {"F1a", "F1b", "F1c", "F1d", "F1e", "F1f", "F1g",
                          "F2A", "F2B", "F3", "F4", "F5", "F6", "F7", "F8"}


def setUpModule() -> None:
    global _CASES, _ROWS, _FEATURES, _TARGETS
    if not CATALOG.exists():
        raise unittest.SkipTest(f"{CATALOG} not present")
    _FEATURES = CatalogFeatures(CATALOG)
    _TARGETS = load_public_targets(DATASET)
    builder = CaseBuilder(_FEATURES, str(CATALOG), str(DATASET))
    generator = SuiteGenerator(_FEATURES, builder)
    _CASES = generator.generate(SuiteGenerator.VALIDATION)
    _ROWS = [c.as_row() for c in _CASES]


class TestSchema(unittest.TestCase):
    def test_every_row_has_required_keys(self):
        for row in _ROWS:
            self.assertEqual(REQUIRED_KEYS - set(row), set(), f"missing keys in {row['test_id']}")

    def test_rows_are_json_serialisable(self):
        for row in _ROWS:
            json.loads(json.dumps(row, ensure_ascii=False, sort_keys=True))

    def test_test_ids_unique(self):
        ids = [r["test_id"] for r in _ROWS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_suite_version_pinned_on_every_row(self):
        self.assertEqual({r["suite_version"] for r in _ROWS}, {SUITE_VERSION})

    def test_class_and_tier_values_are_legal(self):
        for row in _ROWS:
            self.assertIn(row["class"], {"A", "B", "C"})
            self.assertIn(row["tier"], {"T1", "T2", "T3"})

    def test_every_family_and_all_cells_represented(self):
        families = {r["family"] for r in _ROWS}
        self.assertEqual(families, PUBLIC_UNSEEN_FAMILIES)
        self.assertEqual(len({r["cell_id"] for r in _ROWS if r["family"] == "F2A"}), 108)
        f2b_cells = {r["cell_id"] for r in _ROWS if r["family"] == "F2B"}
        main = {c for c in f2b_cells if not c.startswith("F2B_DIAG")}
        diag = {c for c in f2b_cells if c.startswith("F2B_DIAG")}
        self.assertEqual(len(main), 14, sorted(main))
        self.assertEqual(len(diag), 2, sorted(diag))
        self.assertNotIn("F2B/t2_reorder/E2", main)
        self.assertNotIn("F2B/t2_reorder/E3", main)
        self.assertEqual(len({r["cell_id"] for r in _ROWS if r["family"] == "F7"}), 16)


class TestTargets(unittest.TestCase):
    def test_targets_are_real_catalog_products(self):
        catalog = set(_FEATURES.asins)
        for row in _ROWS:
            self.assertIn(row["target_parent_asin"], catalog)

    def test_public_target_exclusion(self):
        """Requirement 2: leakage must be impossible for public-unseen families."""
        public = set(_TARGETS)
        leaked = [r["test_id"] for r in _ROWS
                  if r["family"] in PUBLIC_UNSEEN_FAMILIES
                  and r["target_parent_asin"] in public]
        self.assertEqual(leaked, [], f"public targets leaked into {len(leaked)} cases")

    def test_stratum_matches_the_catalogue(self):
        for row in _ROWS[::7]:
            asin = row["target_parent_asin"]
            self.assertEqual(row["target_stratum"]["rating_number"], _FEATURES.rating_number(asin))
            self.assertEqual(row["target_stratum"]["bucket_rank"], _FEATURES.bucket_rank(asin))
            self.assertEqual(row["target_stratum"]["rarest_df"], _FEATURES.rarest_df(asin))


class TestFlags(unittest.TestCase):
    def test_f1g_is_sentinel_and_excluded(self):
        rows = [r for r in _ROWS if r["family"] == "F1g"]
        self.assertTrue(rows)
        for row in rows:
            self.assertTrue(row["sentinel"])
            self.assertTrue(row["excluded_from_headline"])

    def test_f1f_is_stress_bound(self):
        rows = [r for r in _ROWS if r["family"] == "F1f"]
        self.assertTrue(rows)
        for row in rows:
            self.assertTrue(row["stress_bound"])
            self.assertEqual(row["tier"], "T3")

    def test_f7_is_ood_and_excluded(self):
        rows = [r for r in _ROWS if r["family"] == "F7"]
        self.assertTrue(rows)
        for row in rows:
            self.assertTrue(row["ood"])
            self.assertTrue(row["excluded_from_headline"])
            self.assertEqual(row["class"], "C")

    def test_no_other_family_is_flagged(self):
        for row in _ROWS:
            if row["family"] not in {"F1f", "F1g", "F7", "F2B"}:
                self.assertFalse(row["sentinel"] or row["stress_bound"] or row["ood"],
                                 f"{row['test_id']} ({row['family']}) wrongly flagged")

    def test_t3_never_labelled_t1(self):
        for row in _ROWS:
            if row["stress_bound"]:
                self.assertEqual(row["tier"], "T3")


class TestSchemeLabels(unittest.TestCase):
    def test_f1a_retains_bc_labels(self):
        """Requirement 4: T1 B and C must retain their scheme label."""
        rows = [r for r in _ROWS if r["family"] == "F1a"]
        self.assertEqual({r["scheme"] for r in rows}, {"B", "C"})
        for row in rows:
            self.assertIn(f"/{row['scheme']}/", row["cell_id"])

    def test_f1a_never_mixes_schemes_within_a_cell(self):
        by_cell = collections.defaultdict(set)
        for row in _ROWS:
            if row["family"] == "F1a":
                by_cell[row["cell_id"]].add(row["scheme"])
        for cell, schemes in by_cell.items():
            self.assertEqual(len(schemes), 1, f"{cell} mixes schemes {schemes}")


class TestCoreTailProportion(unittest.TestCase):
    def test_f1a_draw_preserves_core_tail_mass(self):
        """Each F1a draw is a core/tail mixture at the observed 84.5/15.5 split."""
        by_cell = collections.defaultdict(list)
        for row in _ROWS:
            if row["family"] == "F1a":
                by_cell[row["cell_id"]].append(row["target_parent_asin"])
        for cell, asins in sorted(by_cell.items()):
            core = sum(1 for a in asins if _FEATURES.is_strict(a))
            expected = round(len(asins) * 169 / 200)
            self.assertLessEqual(abs(core - expected), max(2, len(asins) // 5),
                                 f"{cell}: {core}/{len(asins)} core, expected about {expected}")


class TestScenarioMix(unittest.TestCase):
    def test_scenario_cycle_matches_official_mix(self):
        counts = collections.Counter(scenario_cycle(2000))
        self.assertAlmostEqual(counts["buying"] / 2000, 0.40, places=2)
        self.assertAlmostEqual(counts["browsing"] / 2000, 0.40, places=2)
        self.assertAlmostEqual(counts["intent_override"] / 2000, 0.15, places=2)
        self.assertAlmostEqual(counts["boundary"] / 2000, 0.05, places=2)

    def test_all_scenarios_legal(self):
        legal = {"buying", "browsing", "intent_override", "boundary"}
        self.assertTrue({r["scenario_type"] for r in _ROWS} <= legal)

    def test_f3_is_buying_only(self):
        rows = [r for r in _ROWS if r["family"] == "F3"]
        self.assertEqual({r["scenario_type"] for r in rows}, {"buying"})

    def test_f1g_old_eq_new_cell_is_override_only(self):
        rows = [r for r in _ROWS if r["cell_id"] == "F1g/old_eq_new"]
        if rows:
            self.assertEqual({r["scenario_type"] for r in rows}, {"intent_override"})


class TestF2AInvariant(unittest.TestCase):
    """Requirement 3a: F2A constraint strings must be byte-identical."""

    def test_constraint_pool_untouched(self):
        rows = [r for r in _ROWS if r["family"] == "F2A"]
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row["clean_inputs"]["constraint_pool"],
                             row["transformed_inputs"]["constraint_pool"],
                             f"{row['test_id']} altered a constraint")
            self.assertEqual(row["transformation"]["constraint_map"], {},
                             "F2A must never carry a constraint map")

    def test_constraint_appears_verbatim_in_transformed_opening(self):
        for row in _ROWS:
            if row["family"] != "F2A":
                continue
            hard = row["clean_inputs"]["intent_card"]["hard_constraints"]
            if row["clean_inputs"]["opening_template"] == "opening_buying" and hard:
                self.assertIn(str(hard[0]), row["transformed_inputs"]["opening_message"],
                              f"{row['test_id']} lost its verbatim constraint")

    def test_template_variant_present(self):
        for row in _ROWS:
            if row["family"] == "F2A":
                self.assertTrue(row["transformation"]["template_variant"])


class TestF2BInvariant(unittest.TestCase):
    """Requirement 3b: F2B templates must be byte-identical to canonical."""

    def test_template_variant_empty(self):
        rows = [r for r in _ROWS if r["family"] == "F2B"]
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row["transformation"]["template_variant"], {},
                             "F2B must never vary a template")

    def test_transformed_opening_is_canonical_template_with_paraphrased_slots(self):
        for row in _ROWS:
            if row["family"] != "F2B":
                continue
            clean, cmap = row["clean_inputs"], row["transformation"]["constraint_map"]
            template = TR.CANONICAL[clean["opening_template"]]
            hard = clean["intent_card"]["hard_constraints"]
            first = str(hard[0]) if hard else ""
            old = str((clean["override"] or {}).get("old_value") or "")
            expected = TR.render(template, cat=clean["coarse_category"],
                                 constraint=cmap.get(first, first), old=cmap.get(old, old))
            self.assertEqual(row["transformed_inputs"]["opening_message"], expected,
                             f"{row['test_id']} is not a canonical template")

    def test_paraphrases_are_recorded_with_their_rule(self):
        for row in _ROWS:
            if row["family"] != "F2B":
                continue
            cmap = row["transformation"]["constraint_map"]
            records = row["transformation"]["constraint_records"]
            self.assertEqual(set(cmap), {r["original_constraint"] for r in records},
                             "every paraphrase needs a record")
            for rec in records:
                self.assertTrue(rec["transformation_rule"])
            for original, paraphrase in cmap.items():
                self.assertNotEqual(original, paraphrase)
                self.assertTrue(paraphrase.strip())

    def test_severity_budget_respected(self):
        for row in _ROWS:
            if row["family"] != "F2B":
                continue
            n = len(row["transformation"]["constraint_map"])
            total = len(row["clean_inputs"]["constraint_pool"])
            cap = total if row["severity"] == "E3" else {"E1": 1, "E2": 2}[row["severity"]]
            self.assertLessEqual(n, cap)

    def test_f2a_and_f2b_are_mechanically_distinguishable(self):
        for row in _ROWS:
            if row["family"] == "F2A":
                self.assertTrue(row["transformation"]["template_variant"])
                self.assertFalse(row["transformation"]["constraint_map"])
            elif row["family"] == "F2B":
                self.assertFalse(row["transformation"]["template_variant"])


class TestOODAndArms(unittest.TestCase):
    def test_f7_scripts_a_turn(self):
        for row in _ROWS:
            if row["family"] == "F7":
                self.assertTrue(row["transformation"]["scripted_turns"])

    def test_f3_has_two_arms(self):
        for row in _ROWS:
            if row["family"] == "F3":
                self.assertEqual(row["arms"], ["as_ships", "turn1_clipped"])
                self.assertEqual(row["n_sessions"], 2)

    def test_f4_carries_a_config_grid(self):
        for row in _ROWS:
            if row["family"] == "F4":
                self.assertEqual(len(row["config_grid"]), 20)
                self.assertEqual(row["n_sessions"], 20)


class TestDeterminism(unittest.TestCase):
    """Requirement 1: same version + seed => byte-identical file."""

    def test_regeneration_is_byte_identical(self):
        builder = CaseBuilder(_FEATURES, str(CATALOG), str(DATASET))
        again = SuiteGenerator(_FEATURES, builder).generate(SuiteGenerator.VALIDATION)
        first = "\n".join(json.dumps(c.as_row(), ensure_ascii=False, sort_keys=True) for c in _CASES)
        second = "\n".join(json.dumps(c.as_row(), ensure_ascii=False, sort_keys=True) for c in again)
        self.assertEqual(first, second)

    def test_row_order_is_deterministic(self):
        keys = [(c.family, c.cell_id, c.sample_id, c.test_id) for c in _CASES]
        self.assertEqual(keys, sorted(keys))

    def test_refuses_silent_overwrite(self):
        """Requirement 9."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            write_cases(_CASES[:5], path)
            with self.assertRaises(FileExistsError):
                write_cases(_CASES[:5], path)
            write_cases(_CASES[:5], path, overwrite=True)


class TestBudget(unittest.TestCase):
    def test_full_suite_totals_8520_sessions(self):
        """The FULL size table must add up to the approved budget."""
        builder = CaseBuilder(_FEATURES, str(CATALOG), str(DATASET))
        generator = SuiteGenerator(_FEATURES, builder)
        sizes = SuiteGenerator.FULL
        expected = (
            2 * 3 * sizes["F1a"] + 10 * sizes["F1b"] + 10 * sizes["F1c"] + 6 * sizes["F1d"]
            + 4 * sizes["F1e"] + 2 * sizes["F1f"] + 3 * sizes["F1g"] + sizes["F2A"]
            + sizes["F2B"] + 4 * sizes["F3"] * 2 + 2 * sizes["F4"] * 20 + 5 * sizes["F5"]
            + 5 * sizes["F6"] + sizes["F7"] + sizes["F8"]
        )
        self.assertEqual(expected, 8520, f"budget table sums to {expected}, not 8520")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestF2BSemanticSafety(unittest.TestCase):
    """Automated guards. These catch MECHANICAL breakage only and do not
    replace human review of meaning preservation."""

    @classmethod
    def setUpClass(cls):
        cls.rows = [r for r in _ROWS if r["family"] == "F2B"]
        cls.records = [rec for r in cls.rows
                       for rec in r["transformation"]["constraint_records"]]

    def test_there_are_records(self):
        self.assertTrue(self.rows)
        self.assertTrue(self.records)

    def test_balanced_brackets(self):
        for rec in self.records:
            self.assertTrue(TR._balanced(rec["transformed_constraint"]),
                            f"unbalanced: {rec['transformed_constraint']!r}")

    def test_non_empty_and_changed(self):
        for rec in self.records:
            new = rec["transformed_constraint"]
            self.assertTrue(new and new.strip())
            self.assertNotEqual(new.strip(), rec["original_constraint"].strip())

    def test_numeric_groups_preserved_outside_explicit_reformat(self):
        for rec in self.records:
            if rec["transformation_rule"] in ("t1_pct_all", "t1_pct_full"):
                continue
            self.assertEqual(TR._numbers(rec["original_constraint"]),
                             TR._numbers(rec["transformed_constraint"]),
                             f"numbers altered by {rec['transformation_rule']}")

    def test_percentage_rule_transforms_every_group(self):
        for rec in self.records:
            if rec["transformation_rule"] == "t1_pct_all":
                self.assertNotIn("%", rec["transformed_constraint"],
                                 "t1_pct_all left a raw percentage behind")

    def test_multi_percentage_string_directly(self):
        new, rule, shape, tier = TR.paraphrase_constraint("82% Nylon, 18% Spandex", "t1_pattern")
        self.assertEqual(shape, "percentage")
        self.assertNotIn("%", new)
        self.assertIn("82 percent", new)
        self.assertIn("18 percent", new)
        new2, *_ = TR.paraphrase_constraint("60% Cotton, 40% Polyester", "t1_pattern")
        self.assertIn("60 percent", new2)
        self.assertIn("40 percent", new2)

    def test_content_token_preservation_for_t2_and_t3(self):
        for rec in self.records:
            original, new = rec["original_constraint"], rec["transformed_constraint"]
            if rec["tier"] == "t2":
                self.assertEqual(TR._content_tokens(original), TR._content_tokens(new))
            elif rec["tier"] == "t3":
                new_tokens = TR._content_tokens(new)
                for token, count in TR._content_tokens(original).items():
                    self.assertGreaterEqual(new_tokens[token], count,
                                            f"t3 lost token {token!r}")

    def test_clause_swap_never_splits_a_parenthetical(self):
        hostile = "Women silver necklace （!!！Please know our package with brand X）"
        self.assertIsNone(TR._t2(hostile), "unsafe clause swap was not rejected")

    def test_unsafe_constraints_are_preserved_not_forced(self):
        long_junk = "A" * 300 + " with " + "B" * 300
        self.assertIsNone(TR._t2(long_junk))

    def test_target_asin_unchanged_by_transformation(self):
        for row in self.rows:
            self.assertEqual(row["target_parent_asin"],
                             row["target_stratum"]["parent_asin"])

    def test_canonical_templates_byte_identical(self):
        for row in self.rows:
            self.assertEqual(row["transformation"]["template_variant"], {})
            self.assertIn(row["clean_inputs"]["opening_template"], TR.CANONICAL)

    def test_requested_vs_achieved_recorded_and_consistent(self):
        for row in self.rows:
            self.assertEqual(row["requested_severity"], row["severity"])
            self.assertIn(row["achieved_severity"], ("E0", "E1", "E2", "E3"))
            self.assertEqual(row["transformation"]["achieved_severity"],
                             row["achieved_severity"])
            self.assertEqual(row["downgraded"],
                             row["achieved_severity"] != row["requested_severity"])
            for rec in row["transformation"]["constraint_records"]:
                self.assertEqual(rec["requested_severity"], row["requested_severity"])
                self.assertEqual(rec["achieved_severity"], row["achieved_severity"])

    def test_achieved_severity_matches_the_count(self):
        for row in self.rows:
            n = len(row["transformation"]["constraint_map"])
            total = len(row["clean_inputs"]["constraint_pool"])
            self.assertEqual(row["achieved_severity"], TR.severity_for(n, total))

    def test_e3_paraphrases_every_available_constraint(self):
        for row in self.rows:
            if row["achieved_severity"] == "E3":
                self.assertEqual(len(row["transformation"]["constraint_map"]),
                                 len(row["clean_inputs"]["constraint_pool"]),
                                 "E3 must paraphrase ALL available constraints")

    def test_e1_behaviour_never_labelled_e2_or_e3(self):
        for row in self.rows:
            n = len(row["transformation"]["constraint_map"])
            if row["achieved_severity"] in ("E2", "E3"):
                self.assertGreaterEqual(n, 2)

    def test_every_record_has_the_required_fields(self):
        required = {"original_constraint", "transformed_constraint", "transformation_rule",
                    "shape", "tier", "requested_severity", "achieved_severity"}
        for rec in self.records:
            self.assertEqual(required - set(rec), set())

    def test_shape_classification_is_total(self):
        for rec in self.records:
            self.assertIn(rec["shape"], TR.SHAPES)


class TestF2BMatrixIntegrity(unittest.TestCase):
    """Post-revision guarantees: no unreachable severity cells in the main matrix."""

    @classmethod
    def setUpClass(cls):
        rows = [r for r in _ROWS if r["family"] == "F2B"]
        cls.main = [r for r in rows if not r["diagnostic"]]
        cls.diag = [r for r in rows if r["diagnostic"]]

    def test_t2_reorder_e2_e3_absent_from_main_matrix(self):
        offenders = [r["cell_id"] for r in self.main
                     if r["transformation"]["kind"] == "t2_reorder"
                     and r["severity"] in ("E2", "E3")]
        self.assertEqual(offenders, [])

    def test_t2_reorder_e1_retained(self):
        self.assertTrue([r for r in self.main
                         if r["transformation"]["kind"] == "t2_reorder"
                         and r["severity"] == "E1"])

    def test_no_downgraded_case_inside_main_e2_e3_cells(self):
        """The core guarantee: E2/E3 aggregates contain no downgraded rows."""
        bad = [(r["cell_id"], r["requested_severity"], r["achieved_severity"])
               for r in self.main if r["severity"] in ("E2", "E3") and r["downgraded"]]
        self.assertEqual(bad, [], f"downgraded rows inside main E2/E3 cells: {bad}")

    def test_main_matrix_has_no_downgrades_at_all(self):
        bad = [(r["cell_id"], r["achieved_severity"]) for r in self.main if r["downgraded"]]
        self.assertEqual(bad, [])

    def test_diagnostics_are_labelled_and_excluded(self):
        self.assertTrue(self.diag)
        for row in self.diag:
            self.assertTrue(row["diagnostic"])
            self.assertTrue(row["excluded_from_headline"])
            self.assertTrue(row["cell_id"].startswith("F2B_DIAG"))
            self.assertTrue(row["transformation"]["diagnostic"])

    def test_shape_mixed_e3_exists_and_paraphrases_all_four(self):
        rows = [r for r in self.main if r["transformation"]["kind"] == "shape_mixed"]
        self.assertTrue(rows, "shape_mixed E3 mode missing")
        for row in rows:
            self.assertEqual(row["severity"], "E3")
            self.assertEqual(row["achieved_severity"], "E3")
            pool = row["clean_inputs"]["constraint_pool"]
            self.assertEqual(len(row["transformation"]["constraint_map"]), len(pool))
            rules = {rec["transformation_rule"]
                     for rec in row["transformation"]["constraint_records"]}
            self.assertTrue(rules, "per-constraint rules must be recorded")

    def test_shape_mixed_dispatches_per_constraint_shape(self):
        pool = ["cotton", "85% Polyester, 15% Cotton", "Imported", "Zipper closure"]
        result = TR.build_constraint_map(pool, "shape_mixed", "E3")
        self.assertEqual(result["achieved_severity"], "E3")
        self.assertEqual(len(result["constraint_map"]), 4)
        shapes = {rec["shape"] for rec in result["constraint_records"]}
        self.assertGreaterEqual(len(shapes), 3, "shape dispatch collapsed to one shape")

    def test_imported_renders_naturally(self):
        """Regression: 'Imported' must not become 'it has Imported'."""
        new, rule, shape, tier = TR.paraphrase_constraint("Imported", "t3_carrier")
        self.assertEqual(new, "it is imported")
        self.assertEqual(rule, "t3_fragment_single")
        self.assertNotIn("it has", new)

    def test_multiword_fragment_still_uses_has(self):
        new, *_ = TR.paraphrase_constraint("Rubber sole", "t3_carrier")
        self.assertEqual(new, "it has Rubber sole")

    def test_e3_cells_are_prioritised_in_allocation(self):
        counts = collections.Counter(r["cell_id"] for r in self.main)
        e3 = [n for cell, n in counts.items() if cell.endswith("/E3")]
        e1 = [n for cell, n in counts.items() if cell.endswith("/E1")]
        if len(_ROWS) > 500:  # only meaningful on the full suite
            self.assertGreater(min(e3), max(e1))
