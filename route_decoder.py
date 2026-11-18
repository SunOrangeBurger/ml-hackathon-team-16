"""Decode the Review 1 fixture with directed, distance-aware Viterbi."""

import heapq
import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np


def load_fixture(path, data_dir):
    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    if fixture["schema_version"] != 1:
        raise ValueError("Unsupported decoder fixture schema")
    graph = fixture["graph"]
    if "nodes_file" in graph:
        nodes = np.loadtxt(data_dir / graph["nodes_file"], ndmin=2)
        arcs = np.loadtxt(data_dir / graph["arcs_file"], dtype=int, ndmin=2)
    else:
        nodes, arcs = np.asarray(graph["nodes"]), np.asarray(graph["arcs"])
    return fixture, nodes, arcs


def decode(
    observations,
    nodes,
    arcs,
    max_gap_s=30,
    backward_tolerance_m=30,
    transition_scale_m=15,
    max_speed_mps=55,
    distance_slack_m=25,
):
    if not observations:
        return [], [], []
    nodes = np.asarray(nodes, dtype=float)
    arcs = np.asarray(arcs, dtype=int)
    gps = np.asarray([[row["longitude"], row["latitude"]] for row in observations])
    lat = np.deg2rad(gps[:, 1].mean())
    metres = 111_195 * np.array([np.cos(lat), 1])
    lengths = np.linalg.norm((nodes[arcs[:, 1]] - nodes[arcs[:, 0]]) * metres, axis=1)
    outgoing = [[] for _ in nodes]
    for arc_id, (source, target) in enumerate(arcs):
        outgoing[source].append((int(target), float(lengths[arc_id]), arc_id))

    @lru_cache(maxsize=256)
    def shortest_paths(source):
        distances = {source: 0.0}
        incoming = {}
        queue = [(0.0, source)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances[node]:
                continue
            for target, length, arc_id in outgoing[node]:
                new_distance = distance + length
                if new_distance <= 1_500 and new_distance < distances.get(target, math.inf):
                    distances[target] = new_distance
                    incoming[target] = (node, arc_id)
                    heapq.heappush(queue, (new_distance, target))
        return distances, incoming

    def transition(previous, current):
        a, b = previous["arc_id"], current["arc_id"]
        fa, fb = previous["arc_fraction"], current["arc_fraction"]
        if a == b:
            signed_distance = (fb - fa) * lengths[a]
            if signed_distance >= -backward_tolerance_m:
                return abs(signed_distance), True
        distances, _ = shortest_paths(int(arcs[a, 1]))
        return (1 - fa) * lengths[a] + distances.get(int(arcs[b, 0]), math.inf) + fb * lengths[b], False

    def snap(candidate):
        source, target = nodes[arcs[candidate["arc_id"]]]
        return (source + candidate["arc_fraction"] * (target - source)).tolist()

    def link_geometry(previous, current):
        a, b = previous["arc_id"], current["arc_id"]
        _, direct = transition(previous, current)
        if direct:
            return [snap(current)]
        source, target = int(arcs[a, 1]), int(arcs[b, 0])
        _, incoming = shortest_paths(source)
        middle = [target]
        while target != source:
            target, _ = incoming[target]
            middle.append(target)
        return [*nodes[middle[::-1]].tolist(), snap(current)]

    assignments = [None] * len(observations)
    sections = []
    warnings = Counter()
    active = []
    scores = []

    def finish_section():
        if not active:
            return
        selected = []
        state = int(np.argmax(scores))
        for index, group, parents in reversed(active):
            candidate = group[state]
            assignments[index] = candidate["arc_id"]
            selected.append(candidate)
            state = parents[state]
        selected.reverse()
        line = [snap(selected[0])]
        for previous, current in zip(selected, selected[1:]):
            line.extend(link_geometry(previous, current))
        if len(line) == 1:
            line.append(line[0])
        sections.append(line)
        active.clear()

    for index, observation in enumerate(observations):
        group = observation["candidates"]
        if not group:
            finish_section()
            warnings["no_candidates"] += 1
            continue
        emissions = [math.log(max(candidate["probability"], 1e-12)) for candidate in group]
        parents = [-1] * len(group)
        if active:
            previous_index, previous_group, _ = active[-1]
            gap = observation["timestamp_s"] - observations[previous_index]["timestamp_s"]
            if gap <= 0 or gap > max_gap_s:
                finish_section()
                warnings["timestamp_gap"] += 1
            else:
                displacement = np.linalg.norm((gps[index] - gps[previous_index]) * metres)
                next_scores = [-math.inf] * len(group)
                for j, candidate in enumerate(group):
                    for k, previous in enumerate(previous_group):
                        if not math.isfinite(scores[k]):
                            continue
                        distance, _ = transition(previous, candidate)
                        if distance > min(1_500, max_speed_mps * gap + distance_slack_m):
                            continue
                        value = (
                            scores[k]
                            + emissions[j]
                            - abs(distance - displacement) / transition_scale_m
                        )
                        if value > next_scores[j]:
                            next_scores[j], parents[j] = value, k
                if any(parent >= 0 for parent in parents):
                    active.append((index, group, parents))
                    scores = next_scores
                    continue
                finish_section()
                warnings["impossible_transition"] += 1
        active.append((index, group, parents))
        scores = emissions
    finish_section()
    return assignments, sections, [f"{reason}: {count}" for reason, count in warnings.items()]
