import json
import unittest
from pathlib import Path

from route_reconstruction import DirectedRoadGraph, decode


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "fixtures" / "route-reconstruction" / "synthetic_decoder_fixture.json"
CONFIG_PATH = ROOT / "fixtures" / "route-reconstruction" / "decoder_config.json"


def load_synthetic():
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    graph = DirectedRoadGraph(fixture["graph"]["nodes"], fixture["graph"]["arcs"])
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return fixture, graph, config


class RouteReconstructionTests(unittest.TestCase):
    def test_viterbi_replaces_disconnected_pointwise_winner_with_feasible_route(self):
        fixture, graph, config = load_synthetic()
        pointwise = [
            max(item["candidates"], key=lambda candidate: candidate["probability"])["arc_id"]
            for item in fixture["observations"]
        ]

        result = decode(fixture["observations"], graph, config)

        self.assertEqual(pointwise, [0, 2])
        self.assertEqual([item["arc_id"] for item in result["assignments"]], [0, 1])
        self.assertEqual(
            [step["arc_id"] for step in result["route_sections"][0]["arc_traversals"]],
            [0, 1],
        )
        self.assertEqual(
            [item["observation_index"] for item in result["assignments"]], [0, 1]
        )

    def test_empty_candidates_are_unmatched_and_reported(self):
        _, graph, config = load_synthetic()
        observations = [
            {"observation_index": 4, "timestamp_s": 1, "candidates": []},
            {"observation_index": 5, "timestamp_s": 2, "candidates": [
                {"arc_id": 0, "arc_fraction": 0.5, "probability": 1.0}
            ]},
        ]

        result = decode(observations, graph, config)

        self.assertIsNone(result["assignments"][0]["arc_id"])
        self.assertEqual(result["assignments"][1]["observation_index"], 5)
        self.assertEqual([warning["code"] for warning in result["warnings"]], ["empty_candidates"])
        self.assertEqual(len(result["route_sections"]), 1)

    def test_long_timestamp_gap_restarts_and_keeps_route_sections_separate(self):
        _, graph, config = load_synthetic()
        observations = [
            {"observation_index": 0, "timestamp_s": 0, "candidates": [
                {"arc_id": 0, "arc_fraction": 0.5, "probability": 1.0}
            ]},
            {"observation_index": 1, "timestamp_s": 61, "candidates": [
                {"arc_id": 2, "arc_fraction": 0.5, "probability": 1.0}
            ]},
        ]

        result = decode(observations, graph, config)

        self.assertEqual(len(result["route_sections"]), 2)
        self.assertEqual([warning["code"] for warning in result["warnings"]], ["timestamp_gap"])

    def test_impossible_transition_restarts_instead_of_joining_disconnected_arcs(self):
        _, graph, config = load_synthetic()
        observations = [
            {"observation_index": 0, "timestamp_s": 0, "candidates": [
                {"arc_id": 0, "arc_fraction": 0.5, "probability": 1.0}
            ]},
            {"observation_index": 1, "timestamp_s": 1, "candidates": [
                {"arc_id": 2, "arc_fraction": 0.5, "probability": 1.0}
            ]},
        ]

        result = decode(observations, graph, config)

        self.assertEqual(len(result["route_sections"]), 2)
        self.assertEqual([warning["code"] for warning in result["warnings"]], ["no_legal_transition"])

    def test_duplicate_timestamps_allow_only_zero_distance_transitions(self):
        _, graph, config = load_synthetic()
        observations = [
            {"observation_index": 0, "timestamp_s": 1, "candidates": [
                {"arc_id": 0, "arc_fraction": 0.5, "probability": 1.0}
            ]},
            {"observation_index": 1, "timestamp_s": 1, "candidates": [
                {"arc_id": 0, "arc_fraction": 0.5, "probability": 1.0}
            ]},
        ]

        result = decode(observations, graph, config)

        self.assertEqual(len(result["route_sections"]), 1)
        self.assertEqual(result["warnings"], [])

    def test_one_way_arc_has_no_reverse_transition(self):
        graph = DirectedRoadGraph([[0, 0], [0.001, 0]], [[0, 1]])

        distance, traversals = graph.transition((0, 0.8), (0, 0.2))

        self.assertEqual(distance, float("inf"))
        self.assertEqual(traversals, [])

    def test_repeated_arc_can_be_reached_again_through_a_directed_loop(self):
        graph = DirectedRoadGraph(
            [[0, 0], [0.001, 0], [0.001, 0.001]],
            [[0, 1], [1, 2], [2, 0]],
        )

        distance, traversals = graph.transition((0, 0.8), (0, 0.2))

        self.assertGreater(distance, graph.arc_lengths[0])
        self.assertEqual([arc_id for arc_id, _, _ in traversals], [0, 1, 2, 0])

    def test_invalid_probability_is_rejected(self):
        _, graph, config = load_synthetic()
        observations = [{
            "observation_index": 0,
            "timestamp_s": 0,
            "candidates": [{"arc_id": 0, "arc_fraction": 0.5, "probability": 1.1}],
        }]

        with self.assertRaisesRegex(ValueError, "probability"):
            decode(observations, graph, config)


if __name__ == "__main__":
    unittest.main()
