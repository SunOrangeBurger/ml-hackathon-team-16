import unittest
import numpy as np

from route_decoder import decode, normalized_log_transitions


class DecoderJitterTests(unittest.TestCase):
    def test_hmm_transitions_normalize_reachable_rows_and_keep_forbidden_edges(self):
        weights = np.array([[-1000., -1001., -np.inf], [-np.inf, -np.inf, -np.inf]])
        probabilities = np.exp(normalized_log_transitions(weights))
        self.assertAlmostEqual(probabilities[0].sum(), 1)
        self.assertEqual(probabilities[0, 2], 0)
        self.assertEqual(probabilities[1].sum(), 0)

    def test_hmm_uses_geometric_likelihood_with_uniform_initial_state(self):
        observations = [{
            "longitude": .0005, "latitude": 0., "timestamp_s": 0,
            "candidates": [
                {"arc_id": 0, "arc_fraction": .5, "probability": .01},
                {"arc_id": 1, "arc_fraction": .5, "probability": .99},
            ],
        }]
        nodes = [[0., 0.], [.001, 0.], [0., .001], [.001, .001]]
        assignments, _, _ = decode(observations, nodes, [[0, 1], [2, 3]], probabilistic_hmm=True)
        self.assertEqual(assignments, [0])

    def test_uturn_penalty_prefers_forward_arc_when_gps_direction_is_ambiguous(self):
        nodes = [[0.0, 0.0], [0.001, 0.0]]
        arcs = [[0, 1], [1, 0]]
        observations = [
            {
                "longitude": 0.00095, "latitude": 0.0, "timestamp_s": 0,
                "candidates": [{"arc_id": 0, "arc_fraction": .95, "probability": 1}],
            },
            {
                "longitude": 0.00095, "latitude": 0.0, "timestamp_s": 1,
                "candidates": [
                    {"arc_id": 0, "arc_fraction": .95, "probability": .1},
                    {"arc_id": 1, "arc_fraction": .05, "probability": .9},
                ],
            },
        ]
        baseline, _, _ = decode(observations, nodes, arcs, uturn_penalty=0)
        penalized, _, _ = decode(observations, nodes, arcs, uturn_penalty=6)
        self.assertEqual(baseline, [0, 1])
        self.assertEqual(penalized, [0, 0])

    def test_small_backward_projection_stays_in_one_section(self):
        nodes = [[0.0, 0.0], [0.001, 0.0]]
        arcs = [[0, 1]]
        observations = [
            {
                "longitude": 0.00060,
                "latitude": 0.0,
                "timestamp_s": 0,
                "candidates": [{"arc_id": 0, "arc_fraction": 0.60, "probability": 1.0}],
            },
            {
                "longitude": 0.00055,
                "latitude": 0.0,
                "timestamp_s": 1,
                "candidates": [{"arc_id": 0, "arc_fraction": 0.55, "probability": 1.0}],
            },
        ]

        assignments, sections, warnings = decode(observations, nodes, arcs)

        self.assertEqual(assignments, [0, 0])
        self.assertEqual(len(sections), 1)
        self.assertEqual(warnings, [])

    def test_backward_projection_beyond_tolerance_restarts(self):
        nodes = [[0.0, 0.0], [0.001, 0.0]]
        arcs = [[0, 1]]
        observations = [
            {
                "longitude": 0.00080,
                "latitude": 0.0,
                "timestamp_s": 0,
                "candidates": [{"arc_id": 0, "arc_fraction": 0.80, "probability": 1.0}],
            },
            {
                "longitude": 0.00020,
                "latitude": 0.0,
                "timestamp_s": 1,
                "candidates": [{"arc_id": 0, "arc_fraction": 0.20, "probability": 1.0}],
            },
        ]

        assignments, sections, warnings = decode(observations, nodes, arcs)

        self.assertEqual(assignments, [0, 0])
        self.assertEqual(len(sections), 2)
        self.assertEqual(warnings, ["impossible_transition: 1"])


if __name__ == "__main__":
    unittest.main()
