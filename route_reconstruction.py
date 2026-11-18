"""Directed-road HMM/Viterbi decoder for PathSnap candidate scores."""

import argparse
import heapq
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path


EARTH_RADIUS_M = 6_371_008.8
DEFAULT_CONFIG = {
    "schema_version": 1,
    "probability_floor": 1e-12,
    "max_gap_s": 60.0,
    "max_speed_mps": 55.0,
    "transition_speed_scale_mps": 15.0,
    "distance_tolerance_m": 1.0,
}


@dataclass(frozen=True)
class DecoderConfig:
    schema_version: int = 1
    probability_floor: float = 1e-12
    max_gap_s: float = 60.0
    max_speed_mps: float = 55.0
    transition_speed_scale_mps: float = 15.0
    distance_tolerance_m: float = 1.0

    @classmethod
    def from_mapping(cls, values):
        config = cls(**values)
        if config.schema_version != 1:
            raise ValueError("Unsupported decoder config schema_version")
        if not 0 < config.probability_floor <= 1:
            raise ValueError("probability_floor must be in (0, 1]")
        if not math.isfinite(config.max_gap_s) or config.max_gap_s <= 0:
            raise ValueError("max_gap_s must be finite and positive")
        if not math.isfinite(config.max_speed_mps) or config.max_speed_mps <= 0:
            raise ValueError("max_speed_mps must be finite and positive")
        if (
            not math.isfinite(config.transition_speed_scale_mps)
            or config.transition_speed_scale_mps <= 0
        ):
            raise ValueError("transition_speed_scale_mps must be finite and positive")
        if (
            not math.isfinite(config.distance_tolerance_m)
            or config.distance_tolerance_m < 0
        ):
            raise ValueError("distance_tolerance_m must be finite and nonnegative")
        return config


class DirectedRoadGraph:
    """Directed graph whose arc IDs are the zero-based input row indices."""

    def __init__(self, nodes, arcs):
        if not nodes or not arcs:
            raise ValueError("Graph must contain at least one node and one directed arc")
        self.nodes = []
        for node in nodes:
            if len(node) != 2 or not all(math.isfinite(float(value)) for value in node):
                raise ValueError("Each graph node must contain finite longitude and latitude")
            lon, lat = map(float, node)
            if not -180 <= lon <= 180 or not -90 <= lat <= 90:
                raise ValueError("Graph node coordinate is outside longitude/latitude bounds")
            self.nodes.append((lon, lat))
        self.arcs = []
        for arc in arcs:
            if len(arc) != 2 or any(isinstance(v, bool) or int(v) != v for v in arc):
                raise ValueError("Each directed arc must contain two integer node IDs")
            source, target = map(int, arc)
            if not 0 <= source < len(nodes) or not 0 <= target < len(nodes):
                raise ValueError("A directed arc references a nonexistent node")
            self.arcs.append((source, target))

        lat0 = math.radians(sum(lat for _, lat in self.nodes) / len(self.nodes))
        lon0 = math.radians(sum(lon for lon, _ in self.nodes) / len(self.nodes))
        self.xy = [
            (
                EARTH_RADIUS_M * (math.radians(lon) - lon0) * math.cos(lat0),
                EARTH_RADIUS_M * (math.radians(lat) - lat0),
            )
            for lon, lat in self.nodes
        ]
        self.arc_lengths = [
            math.hypot(
                self.xy[target][0] - self.xy[source][0],
                self.xy[target][1] - self.xy[source][1],
            )
            for source, target in self.arcs
        ]
        adjacency = [[] for _ in self.nodes]
        for arc_id, (source, target) in enumerate(self.arcs):
            adjacency[source].append((target, arc_id))
        self.adjacency = [tuple(sorted(edges, key=lambda edge: edge[1])) for edges in adjacency]

    @lru_cache(maxsize=8192)
    def shortest_path(self, source, target):
        if source == target:
            return 0.0, ()
        distances = {source: 0.0}
        previous = {}
        queue = [(0.0, source)]
        settled = set()
        while queue:
            distance, node = heapq.heappop(queue)
            if node in settled:
                continue
            settled.add(node)
            if node == target:
                path = []
                while node != source:
                    prior, arc_id = previous[node]
                    path.append(arc_id)
                    node = prior
                return distance, tuple(reversed(path))
            for next_node, arc_id in self.adjacency[node]:
                next_distance = distance + self.arc_lengths[arc_id]
                old_distance = distances.get(next_node, math.inf)
                if next_distance < old_distance - 1e-9 and next_node not in settled:
                    distances[next_node] = next_distance
                    previous[next_node] = (node, arc_id)
                    heapq.heappush(queue, (next_distance, next_node))
        return math.inf, ()

    def transition(self, source, target):
        """Return network distance and traversed arc fractions, or an unreachable result."""
        source_id, source_fraction = source
        target_id, target_fraction = target
        source_node = self.arcs[source_id][1]
        target_node = self.arcs[target_id][0]

        if source_id == target_id and target_fraction >= source_fraction:
            traversals = [(source_id, source_fraction, target_fraction)]
            distance = (target_fraction - source_fraction) * self.arc_lengths[source_id]
            return distance, traversals

        middle_distance, middle_arcs = self.shortest_path(source_node, target_node)
        if not math.isfinite(middle_distance):
            return math.inf, []
        traversals = [(source_id, source_fraction, 1.0)]
        traversals.extend((arc_id, 0.0, 1.0) for arc_id in middle_arcs)
        traversals.append((target_id, 0.0, target_fraction))
        distance = (
            (1 - source_fraction) * self.arc_lengths[source_id]
            + middle_distance
            + target_fraction * self.arc_lengths[target_id]
        )
        return distance, traversals

    def traversal_geometry(self, traversals):
        geometry = []
        for arc_id, start_fraction, end_fraction in traversals:
            source, target = self.arcs[arc_id]
            start = _interpolate(self.nodes[source], self.nodes[target], start_fraction)
            end = _interpolate(self.nodes[source], self.nodes[target], end_fraction)
            if not geometry or geometry[-1] != list(start):
                geometry.append(list(start))
            if geometry[-1] != list(end):
                geometry.append(list(end))
        return geometry


def _interpolate(start, end, fraction):
    return (
        start[0] + (end[0] - start[0]) * fraction,
        start[1] + (end[1] - start[1]) * fraction,
    )


def _is_integer(value):
    return not isinstance(value, bool) and isinstance(value, int)


def _prepare_observations(observations, graph):
    if not isinstance(observations, list):
        raise ValueError("observations must be a list")
    prepared = []
    previous_index = None
    previous_time = None
    for item in observations:
        if not isinstance(item, dict):
            raise ValueError("Each observation must be an object")
        index = item.get("observation_index")
        timestamp = item.get("timestamp_s")
        if not _is_integer(index) or index < 0:
            raise ValueError("observation_index must be a nonnegative integer")
        if previous_index is not None and index <= previous_index:
            raise ValueError("observations must have strictly increasing observation_index")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise ValueError("timestamp_s must be finite")
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        if previous_time is not None and timestamp < previous_time:
            raise ValueError("timestamps must be nondecreasing")
        candidates = item.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("Each observation candidates field must be a list")
        states = []
        seen = set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError("Each candidate must be an object")
            arc_id = candidate.get("arc_id")
            fraction = candidate.get("arc_fraction")
            probability = candidate.get("probability")
            if not _is_integer(arc_id) or not 0 <= arc_id < len(graph.arcs):
                raise ValueError("candidate arc_id must identify a directed graph arc")
            if (
                isinstance(fraction, bool)
                or not isinstance(fraction, (int, float))
                or not math.isfinite(float(fraction))
                or not 0 <= fraction <= 1
            ):
                raise ValueError("candidate arc_fraction must be finite and in [0, 1]")
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(float(probability))
                or not 0 <= probability <= 1
            ):
                raise ValueError("candidate probability must be finite and in [0, 1]")
            state = (arc_id, float(fraction))
            if state in seen:
                raise ValueError("An observation cannot contain duplicate candidate states")
            seen.add(state)
            states.append({
                "arc_id": arc_id,
                "arc_fraction": float(fraction),
                "probability": float(probability),
            })
        states.sort(key=lambda candidate: (candidate["arc_id"], candidate["arc_fraction"]))
        prepared.append({
            "observation_index": index,
            "timestamp_s": timestamp,
            "candidates": states,
        })
        previous_index, previous_time = index, timestamp
    return prepared


def _emission(candidate, config):
    return math.log(max(candidate["probability"], config.probability_floor))


def _transition_score(distance, elapsed, config):
    if not math.isfinite(distance):
        return -math.inf
    if distance > config.max_speed_mps * elapsed + config.distance_tolerance_m:
        return -math.inf
    duration = max(elapsed, 1.0)
    normalized_speed = distance / (config.transition_speed_scale_mps * duration)
    return -0.5 * normalized_speed * normalized_speed


def _append_traversals(output, traversals):
    for arc_id, start, end in traversals:
        if (
            output
            and output[-1]["arc_id"] == arc_id
            and abs(output[-1]["end_fraction"] - start) <= 1e-12
        ):
            output[-1]["end_fraction"] = end
        else:
            output.append({
                "arc_id": arc_id,
                "start_fraction": start,
                "end_fraction": end,
            })


def _decode_connected_block(observations, config, graph, warnings, route_sections):
    if not observations:
        return []
    section_results = []
    segment_start = 0
    while segment_start < len(observations):
        dynamic = [_emission(c, config) for c in observations[segment_start]["candidates"]]
        backpointers = []
        segment_end = segment_start + 1
        while segment_end < len(observations):
            previous = observations[segment_end - 1]
            current = observations[segment_end]
            elapsed = current["timestamp_s"] - previous["timestamp_s"]
            previous_candidates = previous["candidates"]
            current_candidates = current["candidates"]
            next_dynamic = [-math.inf] * len(current_candidates)
            pointers = [None] * len(current_candidates)
            for current_i, current_candidate in enumerate(current_candidates):
                for previous_i, previous_candidate in enumerate(previous_candidates):
                    if not math.isfinite(dynamic[previous_i]):
                        continue
                    distance, _ = graph.transition(
                        (previous_candidate["arc_id"], previous_candidate["arc_fraction"]),
                        (current_candidate["arc_id"], current_candidate["arc_fraction"]),
                    )
                    score = dynamic[previous_i] + _transition_score(distance, elapsed, config)
                    if score > next_dynamic[current_i] + 1e-12:
                        next_dynamic[current_i] = score
                        pointers[current_i] = previous_i
            if not any(math.isfinite(score) for score in next_dynamic):
                warnings.append({
                    "code": "no_legal_transition",
                    "observation_index": current["observation_index"],
                    "message": "No candidate transition is reachable within the speed limit; decoding restarted.",
                })
                break
            dynamic = [
                next_dynamic[i] + _emission(candidate, config)
                for i, candidate in enumerate(current_candidates)
            ]
            backpointers.append(pointers)
            segment_end += 1

        last_candidates = observations[segment_end - 1]["candidates"]
        best_score = max(dynamic)
        best_state = next(
            i for i, score in enumerate(dynamic) if abs(score - best_score) <= 1e-12
        )
        selected = [best_state]
        for pointers in reversed(backpointers):
            selected.append(pointers[selected[-1]])
        selected.reverse()

        segment = observations[segment_start:segment_end]
        selected_candidates = [
            segment[offset]["candidates"][candidate_index]
            for offset, candidate_index in enumerate(selected)
        ]
        section_traversals = []
        first = selected_candidates[0]
        _append_traversals(section_traversals, [
            (first["arc_id"], first["arc_fraction"], first["arc_fraction"])
        ])
        for previous, current in zip(selected_candidates, selected_candidates[1:]):
            _, traversals = graph.transition(
                (previous["arc_id"], previous["arc_fraction"]),
                (current["arc_id"], current["arc_fraction"]),
            )
            _append_traversals(section_traversals, traversals)
        section = {
            "observation_indices": [item["observation_index"] for item in segment],
            "arc_traversals": section_traversals,
            "geometry": graph.traversal_geometry([
                (step["arc_id"], step["start_fraction"], step["end_fraction"])
                for step in section_traversals
            ]),
        }
        route_sections.append(section)
        for item, candidate in zip(segment, selected_candidates):
            section_results.append({
                "observation_index": item["observation_index"],
                "arc_id": candidate["arc_id"],
                "arc_fraction": candidate["arc_fraction"],
                "probability": candidate["probability"],
            })
        segment_start = segment_end
    return section_results


def decode(observations, graph, config=None):
    """Decode ordered observations into assignments, connected route sections and warnings."""
    config = DecoderConfig.from_mapping(
        asdict(config) if isinstance(config, DecoderConfig)
        else (DEFAULT_CONFIG if config is None else config)
    )
    prepared = _prepare_observations(observations, graph)
    warnings = []
    route_sections = []
    assignments = []
    block = []

    def finish_block():
        if block:
            assignments.extend(_decode_connected_block(
                block, config, graph, warnings, route_sections
            ))
            block.clear()

    for item in prepared:
        if not item["candidates"]:
            finish_block()
            assignments.append({
                "observation_index": item["observation_index"],
                "arc_id": None,
                "arc_fraction": None,
                "probability": None,
            })
            warnings.append({
                "code": "empty_candidates",
                "observation_index": item["observation_index"],
                "message": "Observation has no candidate states and is unmatched.",
            })
            continue
        if block:
            elapsed = item["timestamp_s"] - block[-1]["timestamp_s"]
            if elapsed > config.max_gap_s:
                finish_block()
                warnings.append({
                    "code": "timestamp_gap",
                    "observation_index": item["observation_index"],
                    "gap_s": elapsed,
                    "message": "Timestamp gap exceeds max_gap_s; decoding restarted.",
                })
        block.append(item)
    finish_block()
    assignments.sort(key=lambda assignment: assignment["observation_index"])
    return {
        "schema_version": 1,
        "decoder_config": asdict(config),
        "assignments": assignments,
        "route_sections": route_sections,
        "warnings": warnings,
    }


def _read_whitespace_table(path, columns):
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            values = line.split()
            if len(values) != columns:
                raise ValueError(f"{path}:{line_number} must have {columns} columns")
            rows.append([float(value) for value in values])
    return rows


def load_fixture_graph(fixture, data_dir):
    graph_data = fixture.get("graph")
    if not isinstance(graph_data, dict):
        raise ValueError("fixture graph must be an object")
    if "nodes" in graph_data and "arcs" in graph_data:
        return DirectedRoadGraph(graph_data["nodes"], graph_data["arcs"])
    nodes_file, arcs_file = graph_data.get("nodes_file"), graph_data.get("arcs_file")
    if not isinstance(nodes_file, str) or not isinstance(arcs_file, str):
        raise ValueError("graph must contain inline nodes/arcs or nodes_file/arcs_file paths")
    return DirectedRoadGraph(
        _read_whitespace_table(data_dir / nodes_file, 2),
        _read_whitespace_table(data_dir / arcs_file, 2),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("fixtures/route-reconstruction/synthetic_decoder_fixture.json"),
        help="JSON fixture containing ordered candidates and an inline or file-backed graph",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Student folder/Student folder"),
        help="Base directory for graph paths in a file-backed fixture",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("fixtures/route-reconstruction/decoder_config.json"),
    )
    parser.add_argument("--output", type=Path, help="Write result JSON to this path instead of stdout")
    args = parser.parse_args()
    try:
        fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(fixture, dict):
            raise ValueError("fixture root must be a JSON object")
        graph = load_fixture_graph(fixture, args.data_dir)
        result = decode(fixture.get("observations"), graph, config)
        rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
            print(f"Wrote decoded assignments and route sections to {args.output}")
        else:
            print(rendered, end="")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        parser.exit(1, f"route reconstruction failed: {error}\n")


if __name__ == "__main__":
    main()
