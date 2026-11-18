"""Build the Workstream 01 Review 1 audit and decoder fixtures."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import softmax

from inspect_trip import (
    load_table,
    load_trip,
    project_trip,
    rank_candidates,
    save_inspection,
    segment_distances,
)


TABLES = {
    "track": (3, float),
    "nodes": (2, float),
    "arcs": (2, np.int64),
    "route": (1, np.int64),
}


def audit_trip(folder):
    result = {"trip_id": folder.name}
    tables = {}
    issues = []
    for extension, (columns, dtype) in TABLES.items():
        path = folder / f"{folder.name}.{extension}"
        result[f"{extension}_bytes"] = path.stat().st_size if path.is_file() else None
        if not path.is_file():
            issues.append(f"missing {extension}")
            continue
        try:
            tables[extension] = load_table(folder, extension, columns, dtype)
            result[f"{extension}_rows"] = len(tables[extension])
        except (OSError, ValueError) as error:
            issues.append(f"invalid {extension}: {error}")

    for name in ("track", "nodes"):
        if name in tables:
            coordinates = tables[name][:, :2]
            result[f"{name}_invalid_coordinate_rows"] = int(
                np.count_nonzero((np.abs(coordinates) > [180, 90]).any(axis=1))
            )
            for axis, column in enumerate(("longitude", "latitude")):
                result[f"{name}_min_{column}"] = float(coordinates[:, axis].min())
                result[f"{name}_max_{column}"] = float(coordinates[:, axis].max())

    if "track" in tables:
        steps = np.diff(tables["track"][:, 2])
        result.update(
            decreasing_timestamp_steps=int(np.count_nonzero(steps < 0)),
            duplicate_timestamp_steps=int(np.count_nonzero(steps == 0)),
            steps_over_one_second=int(np.count_nonzero(steps > 1)),
            max_gap_s=float(steps.max(initial=0)),
        )

    if "nodes" in tables and "arcs" in tables:
        n = len(tables["nodes"])
        arcs = tables["arcs"]
        valid = ((arcs >= 0) & (arcs < n)).all(axis=1)
        result["arcs_with_invalid_endpoints"] = int(np.count_nonzero(~valid))
        sources, targets = arcs[valid].T
        graph = coo_matrix((np.ones(len(sources), dtype=bool), (sources, targets)), shape=(n, n)).tocsr()
        result["isolated_nodes"] = n - len(np.unique(arcs[valid]))
        result["weak_components"] = int(connected_components(graph, directed=True, connection="weak", return_labels=False))
        result["strong_components"] = int(connected_components(graph, directed=True, connection="strong", return_labels=False))
        if "route" in tables:
            route = tables["route"].ravel()
            valid_ids = (route >= 0) & (route < len(arcs))
            result["invalid_route_arc_ids"] = int(np.count_nonzero(~valid_ids))
            if valid_ids.all():
                result["reference_arc_discontinuities"] = int(
                    np.count_nonzero(arcs[route[:-1], 1] != arcs[route[1:], 0])
                )
    result["file_issues"] = " | ".join(issues)
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def align_reference(trip):
    """Find the minimum-distance nondecreasing sequence of reference positions."""
    points, nodes = project_trip(trip)
    starts, ends = nodes[trip.arcs[trip.route]].transpose(1, 0, 2)
    vectors = ends - starts
    lengths_squared = np.einsum("ij,ij->i", vectors, vectors)
    positions = np.arange(len(trip.route))
    back = np.empty((len(points), len(positions)), dtype=np.int32)

    costs = segment_distances(points[0], starts, vectors, lengths_squared)[0] ** 2
    for index, point in enumerate(points[1:], 1):
        prefix_costs = np.minimum.accumulate(costs)
        improves = np.r_[True, prefix_costs[1:] < prefix_costs[:-1]]
        back[index] = np.maximum.accumulate(np.where(improves, positions, 0))
        costs = prefix_costs + segment_distances(point, starts, vectors, lengths_squared)[0] ** 2

    selected = np.empty(len(points), dtype=np.int32)
    selected[-1] = np.argmin(costs)
    for index in range(len(points) - 1, 0, -1):
        selected[index - 1] = back[index, selected[index]]
    offsets = points - starts[selected]
    selected_vectors = vectors[selected]
    selected_lengths_squared = lengths_squared[selected]
    fractions = np.divide(
        np.einsum("ij,ij->i", offsets, selected_vectors),
        selected_lengths_squared,
        out=np.zeros(len(points)),
        where=selected_lengths_squared > 0,
    ).clip(0, 1)
    distances = np.linalg.norm(offsets - fractions[:, None] * selected_vectors, axis=1)
    return selected, distances, fractions


def score_candidates(group, distance_scale_m):
    distances = np.array([distance for _, distance, _ in group])
    probabilities = softmax(-0.5 * (distances / distance_scale_m) ** 2) if group else []
    return [
        {"arc_id": arc, "arc_fraction": fraction, "probability": float(probability)}
        for (arc, _, fraction), probability in zip(group, probabilities)
    ]


def build_example(folder, output, radius_m, limit, distance_scale_m):
    trip = load_trip(folder)
    groups = rank_candidates(trip, radius_m, limit)
    positions, distances, fractions = align_reference(trip)
    labels = []
    observations = []
    for index, ((lon, lat, time), group, position, distance, fraction) in enumerate(
        zip(trip.track, groups, positions, distances, fractions)
    ):
        arc_id = int(trip.route[position])
        rank = next((rank for rank, candidate in enumerate(group, 1) if candidate[0] == arc_id), None)
        labels.append({
            "observation_index": index,
            "reference_route_index": int(position),
            "reference_arc_id": arc_id,
            "reference_distance_m": float(distance),
            "reference_arc_fraction": float(fraction),
            "reference_candidate_rank": rank,
        })
        observations.append({
            "observation_index": index, "longitude": lon, "latitude": lat, "timestamp_s": time,
            "candidates": score_candidates(group, distance_scale_m),
        })
    example = save_inspection(folder.name, output, trip, groups, radius_m, limit)
    prefix = f"train/{folder.name}/{folder.name}"
    write_json(output / "decoder_fixture.json", {
        "schema_version": 1,
        "trip_id": folder.name,
        "score_method": "normalized Gaussian distance weights",
        "distance_scale_m": distance_scale_m,
        "graph": {"nodes_file": f"{prefix}.nodes", "arcs_file": f"{prefix}.arcs"},
        "observations": observations,
    })
    write_csv(output / "aligned_reference.csv", labels)
    misses = sum(label["reference_candidate_rank"] is None for label in labels)
    hits_at_one = sum(label["reference_candidate_rank"] == 1 for label in labels)
    write_json(output / "baseline_metrics.json", {
        "trip_id": folder.name,
        "label_method": "minimum squared GPS distance over nondecreasing reference-route positions",
        "observations": len(labels),
        "radius_m": radius_m,
        "max_candidates": limit,
        "aligned_candidate_misses": misses,
        "aligned_candidate_miss_rate": misses / len(labels),
        "aligned_candidate_coverage": 1 - misses / len(labels),
        "aligned_distance_baseline_top_1": hits_at_one / len(labels),
        "aligned_distances_over_candidate_radius": sum(
            label["reference_distance_m"] > radius_m for label in labels
        ),
        "max_timestamp_gap_s": example["max_gap_s"],
    })


def write_synthetic_fixture(path):
    # Arc 0 leads to arc 1. Arc 2 is a separate road with the best second score.
    observations = []
    for index, (lon, lat, scores) in enumerate([
        (8.00005, 49.0, [(0, 1.0)]),
        (8.00015, 49.00003, [(2, 0.7), (1, 0.3)]),
    ]):
        observations.append({
            "observation_index": index,
            "longitude": lon,
            "latitude": lat,
            "timestamp_s": index + 1,
            "candidates": [
                {"arc_id": arc, "arc_fraction": 0.5, "probability": score}
                for arc, score in scores
            ],
        })
    write_json(path, {
        "schema_version": 1,
        "trip_id": "synthetic",
        "score_method": "fixed synthetic scores",
        "graph": {
            "nodes": [[8.0, 49.0], [8.0001, 49.0], [8.0002, 49.0],
                      [8.0001, 49.00004], [8.0002, 49.00004]],
            "arcs": [[0, 1], [1, 2], [3, 4]],
        },
        "observations": observations,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("Student folder/Student folder"))
    parser.add_argument("--output", type=Path, default=Path("output/review-1"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ("train", "student_test"):
        for trip in sorted((args.data_dir / split).iterdir()):
            if trip.is_dir():
                rows.append({"split": split, **audit_trip(trip)})
    write_csv(args.output / "dataset_inventory.csv", rows)
    build_example(args.data_dir / "train/00000001", args.output / "00000001", 50, 10, 15)
    write_synthetic_fixture(args.output / "synthetic_decoder_fixture.json")
    print(f"Audited {len(rows)} trips and wrote the training example to {args.output}")


if __name__ == "__main__":
    main()
