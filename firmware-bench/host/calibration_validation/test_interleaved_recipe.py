import unittest

import numpy as np

from calibration_corpus import Corpus, Cue
from calibration_fit import checkpoint_inputs
from experiment_14_interleaved_matrix import cue_order_digest
from interleaved_recipe import (
    FIT_ORDER_NAMES, ORDER_NAMES, checkpoint_groups, cue_limits, fit_order_cues,
    order_cues, selected_cues, with_bootstrapped_commands,
)
from experiment_15_interleaved_search import (
    acceptance_vector, pareto_frontier, search_cells,
)
from experiment_16_interleaved_robust_search import (
    aggregate_frontier, aggregate_results, order_scenarios, structural_cells,
)
from experiment_17_fit_order_search import (
    fit_search_cells, output_document, parse_cells,
)
from experiment_18_sequential_assignment_search import (
    assignment_families, assignment_search_cells,
)


def cue(group, label):
    return Cue(group, label, np.full((1, 64), group + label, dtype=np.float32))


def fixture_corpus(command_reps=10, no_op_reps=12):
    commands = [cue(rep * 5 + label, label)
                for rep in range(command_reps) for label in range(5)]
    no_ops = [cue(rep * 5 + label, label + 5)
              for rep in range(no_op_reps) for label in range(5)]
    return Corpus(np.zeros((2, 64), dtype=np.float32), np.array([10, 11]),
                  commands, no_ops)


class InterleavedRecipeTests(unittest.TestCase):
    def setUp(self):
        self.corpus = fixture_corpus()
        self.limits = cue_limits(10, 12)
        self.selected = selected_cues(
            self.corpus, self.corpus.all_command_groups,
            self.corpus.all_no_op_groups, self.limits)

    def identities(self, cues):
        return sorted((item.label, item.group) for item in cues)

    def test_every_order_preserves_the_selected_multiset(self):
        expected = self.identities(self.selected)
        for order in ORDER_NAMES:
            with self.subTest(order=order):
                self.assertEqual(
                    self.identities(order_cues(self.selected, order, 17)),
                    expected,
                )

    def test_named_order_invariants(self):
        chronological = order_cues(self.selected, "chronological", 17)
        reversed_order = order_cues(self.selected, "reversed", 17)
        self.assertEqual([item.label for item in reversed_order],
                         list(reversed([item.label for item in chronological])))

        class_clustered = order_cues(self.selected, "class_clustered", 17)
        self.assertEqual([item.label for item in class_clustered],
                         sorted(item.label for item in class_clustered))

        thumb_clustered = order_cues(self.selected, "thumb_clustered", 17)
        states = [item.label >= 5 for item in thumb_clustered]
        self.assertEqual(states, sorted(states))

    def test_song_order_is_seeded_and_deterministic(self):
        first = order_cues(self.selected, "song_like", 17)
        self.assertEqual(first, order_cues(self.selected, "song_like", 17))
        self.assertNotEqual(first, order_cues(self.selected, "song_like", 18))

    def test_song_orders_preserve_recorded_order_inside_each_class(self):
        for order in ("reversed", "song_like"):
            with self.subTest(order=order):
                ordered = order_cues(self.selected, order, 17)
                for label in range(10):
                    groups = [item.group for item in ordered
                              if item.label == label]
                    self.assertEqual(groups, sorted(groups))

    def test_sequential_assignment_cycles_and_preserves_class_chronology(self):
        canonical = order_cues(self.selected, "sequential_canonical", 99)
        self.assertEqual([cue.label for cue in canonical[:20]],
                         list(range(10)) * 2)

        paired = order_cues(self.selected, "sequential_paired", 99)
        self.assertEqual([cue.label for cue in paired[:10]],
                         [0, 5, 1, 6, 2, 7, 3, 8, 4, 9])

        rotated = order_cues(self.selected, "sequential_rotated", 3)
        self.assertEqual([cue.label for cue in rotated[:10]],
                         [3, 4, 5, 6, 7, 8, 9, 0, 1, 2])
        for ordered in (canonical, paired, rotated):
            for label in range(10):
                groups = [cue.group for cue in ordered if cue.label == label]
                self.assertEqual(groups, sorted(groups))

    def test_sequential_assignment_ignores_prior_song_semantics(self):
        first_song = order_cues(self.selected, "song_like", 17)
        second_song = order_cues(self.selected, "song_like", 18)
        self.assertNotEqual(first_song, second_song)
        self.assertEqual(
            order_cues(first_song, "sequential_canonical", 0),
            order_cues(second_song, "sequential_canonical", 0),
        )

    def test_fit_order_normalization_does_not_mutate_collection_order(self):
        first_collection = order_cues(self.selected, "song_like", 17)
        second_collection = order_cues(self.selected, "song_like", 18)
        self.assertNotEqual(first_collection, second_collection)
        first_snapshot = list(first_collection)
        second_snapshot = list(second_collection)
        for policy in FIT_ORDER_NAMES[1:]:
            with self.subTest(policy=policy):
                self.assertEqual(
                    fit_order_cues(first_collection, policy, 23),
                    fit_order_cues(second_collection, policy, 23),
                )
        self.assertEqual(first_collection, first_snapshot)
        self.assertEqual(second_collection, second_snapshot)
        self.assertNotEqual(cue_order_digest(first_collection),
                            cue_order_digest(second_collection))
        self.assertEqual(
            cue_order_digest(fit_order_cues(first_collection,
                                            "round_robin", 23)),
            cue_order_digest(fit_order_cues(second_collection,
                                            "round_robin", 23)),
        )

    def test_fit_order_policies_have_canonical_invariants(self):
        round_robin = fit_order_cues(self.selected, "round_robin", 23)
        self.assertEqual([cue.label for cue in round_robin[:10]], list(range(10)))

        class_balanced = fit_order_cues(self.selected, "class_balanced", 23)
        self.assertEqual(self.identities(class_balanced),
                         self.identities(self.selected))

        shuffled = fit_order_cues(self.selected, "deterministic_shuffled", 23)
        self.assertEqual(shuffled,
                         fit_order_cues(list(reversed(self.selected)),
                                        "deterministic_shuffled", 23))
        self.assertNotEqual(shuffled,
                            fit_order_cues(self.selected,
                                           "deterministic_shuffled", 24))

    def test_checkpoint_batches_flatten_back_to_the_order(self):
        groups = checkpoint_groups(
            self.corpus, self.corpus.all_command_groups,
            self.corpus.all_no_op_groups, self.limits,
            "song_like", 5, 17,
        )
        self.assertTrue(all(0 < len(group) <= 5 for group in groups))
        flattened = [item for group in groups for item in group]
        self.assertEqual(flattened, order_cues(self.selected, "song_like", 17))

    def test_count_limits_are_asymmetric(self):
        counts = np.bincount([item.label for item in self.selected], minlength=10)
        np.testing.assert_array_equal(counts[:5], np.full(5, 10))
        np.testing.assert_array_equal(counts[5:10], np.full(5, 12))

    def test_eleven_command_cell_repeats_groups_without_new_fold_identity(self):
        expanded = with_bootstrapped_commands(self.corpus, 11)
        counts = np.bincount([item.label for item in expanded.command_cues], minlength=5)
        np.testing.assert_array_equal(counts, np.full(5, 11))
        for label in range(5):
            groups = [item.group for item in expanded.command_cues
                      if item.label == label]
            self.assertEqual(len(set(groups)), 10)

    def test_checkpoint_weighting_keeps_no_op_scale_at_point_four(self):
        labels = np.arange(10).repeat(2)
        _, weights = checkpoint_inputs(labels, convention="checkpoint_counts")
        totals = np.bincount(labels, weights=weights, minlength=10)
        np.testing.assert_allclose(totals[:5], np.ones(5))
        np.testing.assert_allclose(totals[5:10], np.full(5, 0.4))

    def test_search_seeds_expand_only_song_like_orders(self):
        cells = search_cells(((10, 12),), (5,), (8,), (10,),
                             ("chronological", "song_like"), (17, 18))
        self.assertEqual(len(cells), 3)
        self.assertEqual([cell[3] for cell in cells], [17, 17, 18])

    def test_acceptance_distance_and_frontier_name_the_blocking_metrics(self):
        passing = {
            "false_negatives": 4, "misclassified": 0, "false_fires": 3,
            "static_rest_commits": 0, "moving_rest_commits": 0,
        }
        confused = {**passing, "misclassified": 2}
        false_firing = {**passing, "false_fires": 5}
        both = {**confused, "false_fires": 5}
        self.assertEqual(acceptance_vector(passing), (0, 0, 0, 0))
        self.assertEqual(acceptance_vector(both), (0, 2, 2, 0))
        self.assertEqual(pareto_frontier([confused, false_firing, both]),
                         [false_firing, confused])

    def test_robust_scenarios_and_structural_grid_are_deterministic(self):
        scenarios = order_scenarios(
            ("chronological", "reversed", "song_like"), (17, 18))
        self.assertEqual(scenarios, [
            ("chronological", 17), ("reversed", 17),
            ("song_like", 17), ("song_like", 18),
        ])
        self.assertEqual(
            structural_cells(((9, 16), (10, 16)), (4,), (8,), (10, 12)),
            [(9, 16, 4, 8, 10), (9, 16, 4, 8, 12),
             (10, 16, 4, 8, 10), (10, 16, 4, 8, 12)],
        )

    def test_aggregate_scoring_uses_component_worst_cases(self):
        base = {
            "false_negatives": 4, "misclassified": 0, "false_fires": 3,
            "static_rest_commits": 0, "moving_rest_commits": 0,
            "in_legacy_region": True,
        }
        confusion = {**base, "misclassified": 2, "in_legacy_region": False}
        false_fire = {**base, "false_fires": 5, "in_legacy_region": False}
        aggregate = aggregate_results([base, confusion, false_fire])
        self.assertEqual(aggregate["worst_case_distance"], (0, 2, 2, 0))
        self.assertEqual(aggregate["passing_order_count"], 1)
        self.assertFalse(aggregate["all_orders_in_legacy_region"])

    def test_aggregate_frontier_can_include_fit_cost(self):
        def cell(distance, passes):
            return {"aggregate": {"worst_case_distance": distance},
                    "fit_passes": passes}

        cheap_worse = cell((0, 1, 1, 0), 100)
        costly_better = cell((0, 1, 0, 0), 200)
        dominated = cell((0, 2, 1, 0), 250)
        self.assertEqual(aggregate_frontier(
            [cheap_worse, costly_better, dominated], include_cost=True),
            [cheap_worse, costly_better])

    def test_fit_search_grid_and_cell_parser_are_deterministic(self):
        cells = parse_cells("10/16:4:8:10,9/16:5:8:10")
        self.assertEqual(cells, ((10, 16, 4, 8, 10),
                                 (9, 16, 5, 8, 10)))
        self.assertEqual(
            fit_search_cells(cells, ("collection", "round_robin")),
            [(10, 16, 4, 8, 10, "collection"),
             (10, 16, 4, 8, 10, "round_robin"),
             (9, 16, 5, 8, 10, "collection"),
             (9, 16, 5, 8, 10, "round_robin")],
        )

    def test_fit_search_output_records_partial_progress(self):
        document = output_document(
            [], [("chronological", 17), ("song_like", 17)],
            ("round_robin",), 23, 10, False,
        )
        self.assertFalse(document["complete"])
        self.assertEqual(document["planned_structural_policy_cells"], 5)
        self.assertEqual(document["completed_order_evaluations"], 0)

    def test_sequential_family_search_expansion_is_deterministic(self):
        families = assignment_families((0, 3, 7))
        self.assertEqual(families["rotated"], (
            ("sequential_rotated", 0),
            ("sequential_rotated", 3),
            ("sequential_rotated", 7),
        ))
        cells = assignment_search_cells(((10, 16, 5, 8, 10),), families)
        self.assertEqual([family for _, family, _ in cells],
                         ["canonical", "paired", "rotated"])
        self.assertEqual(list(assignment_families((0, 1), ("rotated",))),
                         ["rotated"])


if __name__ == "__main__":
    unittest.main()
