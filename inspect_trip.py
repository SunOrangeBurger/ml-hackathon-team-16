"""Inspect one PathSnap training trip and export a distance-ranked baseline."""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Trip:
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray
    route: np.ndarray


def load_table(folder, extension, columns, dtype):
    path = folder / f"{folder.name}.{extension}"
    table = np.loadtxt(path, dtype=dtype, ndmin=2)
    if not table.size or table.shape[1] != columns or not np.isfinite(table).all():
        raise ValueError(f"{path} must contain {columns} finite columns and at least one row")
    return table


def load_trip(folder):
    track = load_table(folder, "track", 3, float)
    nodes = load_table(folder, "nodes", 2, float)
    arcs = load_table(folder, "arcs", 2, np.int64)
    route = load_table(folder, "route", 1, np.int64).ravel()
    for coordinates in (track[:, :2], nodes):
        if (np.abs(coordinates) > [180, 90]).any():
            raise ValueError("Longitude or latitude is outside its valid range")
    if (arcs < 0).any() or (arcs >= len(nodes)).any():
        raise ValueError("An arc references a nonexistent node")
    if (route < 0).any() or (route >= len(arcs)).any():
        raise ValueError("The reference route contains a nonexistent arc")
    if (np.diff(track[:, 2]) < 0).any():
        raise ValueError("GPS timestamps decrease in file order")
    return Trip(track, nodes, arcs, route)


def segment_distances(point, starts, vectors, lengths_squared):
    offsets = point - starts
    fractions = np.divide(
        np.einsum("ij,ij->i", offsets, vectors),
        lengths_squared,
        out=np.zeros(len(starts)),
        where=lengths_squared > 0,
    ).clip(0, 1)
    distances = np.linalg.norm(offsets - fractions[:, None] * vectors, axis=1)
    return distances, fractions


def project_trip(trip):
    # Local equirectangular approximation for this city-scale dataset.
    origin = np.mean(trip.track[:, :2], axis=0)
    metres_per_degree = np.pi * 6_371_008.8 / 180
    scale = metres_per_degree * np.array([np.cos(np.deg2rad(origin[1])), 1])
    return (trip.track[:, :2] - origin) * scale, (trip.nodes - origin) * scale


def rank_candidates(trip, radius_m, limit):
    points, nodes = project_trip(trip)
    starts, ends = nodes[trip.arcs].transpose(1, 0, 2)
    vectors = ends - starts
    lengths_squared = np.einsum("ij,ij->i", vectors, vectors)
    tree = cKDTree(starts + vectors / 2)
    # A segment within the radius has a midpoint within radius + half its length.
    search_radius = radius_m + np.sqrt(lengths_squared.max()) / 2

    candidates = []
    for point in points:
        ids = np.asarray(tree.query_ball_point(point, search_radius), dtype=np.int64)
        distances, fractions = segment_distances(
            point, starts[ids], vectors[ids], lengths_squared[ids]
        )
        order = np.lexsort((ids, distances))
        nearby = order[distances[order] <= radius_m][:limit]
        candidates.append([(int(ids[i]), float(distances[i]), float(fractions[i])) for i in nearby])
    return candidates


def road_geometry(trip, arc_ids):
    return [trip.nodes[trip.arcs[arc_id], ::-1].tolist() for arc_id in arc_ids]


def save_map(path, trip, candidates):
    import folium

    points = trip.track[:, 1::-1]
    result = folium.Map(location=points.mean(axis=0).tolist(), control_scale=True)
    nearby_ids = sorted({arc for group in candidates for arc, _, _ in group})
    layers = (
        ("Candidate roads", nearby_ids, "#64748b", 2),
        ("Reference route", trip.route, "#15803d", 4),
    )
    for name, arc_ids, color, weight in layers:
        if len(arc_ids):
            layer = folium.FeatureGroup(name=name).add_to(result)
            folium.PolyLine(road_geometry(trip, arc_ids), color=color, weight=weight).add_to(layer)
    gps = folium.FeatureGroup(name="GPS observations").add_to(result)
    for index, (point, group) in enumerate(zip(points, candidates)):
        folium.CircleMarker(
            point.tolist(), radius=3, color="#1d4ed8" if group else "#b91c1c",
            fill=True, fill_opacity=0.85,
            tooltip=f"Observation {index} | {trip.track[index, 2]:g} s | {len(group)} candidates",
        ).add_to(gps)
    result.fit_bounds([points.min(axis=0).tolist(), points.max(axis=0).tolist()])
    folium.LayerControl().add_to(result)
    result.get_root().html.add_child(folium.Element(
        '<aside style="position:fixed;bottom:24px;left:12px;z-index:9999;background:white;'
        'padding:8px;border:1px solid #aaa">Blue: GPS · Red: GPS without candidates<br>'
        'Green: reference route · Gray: candidate roads</aside>'
    ))
    result.save(str(path))


def inspect_trip(folder, output, radius_m, limit):
    trip = load_trip(folder)
    candidates = rank_candidates(trip, radius_m, limit)
    return save_inspection(folder.name, output, trip, candidates, radius_m, limit)


def save_inspection(trip_id, output, trip, candidates, radius_m, limit):
    output.mkdir(parents=True, exist_ok=True)
    reference_arcs = set(trip.route.tolist())
    with (output / "candidates.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "observation_index", "timestamp_s", "longitude", "latitude",
            "candidate_rank", "arc_id", "distance_m", "arc_fraction", "appears_in_reference_route",
        ])
        for index, (observation, group) in enumerate(zip(trip.track, candidates)):
            lon, lat, timestamp = observation
            if not group:
                writer.writerow([index, timestamp, lon, lat, "", "", "", "", ""])
            for rank, (arc, distance, fraction) in enumerate(group, 1):
                writer.writerow([
                    index, timestamp, lon, lat, rank, arc, f"{distance:.3f}",
                    f"{fraction:.6f}", int(arc in reference_arcs),
                ])
    gaps = np.diff(trip.track[:, 2])
    summary = {
        "trip_id": trip_id,
        "observations": len(trip.track),
        "nodes": len(trip.nodes),
        "directed_arcs": len(trip.arcs),
        "reference_arc_steps": len(trip.route),
        "reference_arc_discontinuities": int(np.count_nonzero(
            trip.arcs[trip.route[:-1], 1] != trip.arcs[trip.route[1:], 0]
        )),
        "duplicate_timestamp_steps": int(np.count_nonzero(gaps == 0)),
        "steps_over_one_second": int(np.count_nonzero(gaps > 1)),
        "max_gap_s": float(gaps.max(initial=0)),
        "radius_m": radius_m,
        "max_candidates_per_observation": limit,
        "observations_without_candidates": sum(not group for group in candidates),
        "observations_with_any_reference_arc_candidate": sum(
            any(arc in reference_arcs for arc, _, _ in group) for group in candidates
        ),
        "observations_with_reference_arc_as_nearest_candidate": sum(
            bool(group) and group[0][0] in reference_arcs for group in candidates
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    save_map(output / "map.html", trip, candidates)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=Path("Student folder/Student folder/train"))
    parser.add_argument("--trip", default="00000001", help="Eight-digit training trip ID")
    parser.add_argument("--output", type=Path, help="Defaults to output/trips/TRIP_ID")
    parser.add_argument("--radius-m", type=float, default=50)
    parser.add_argument("--max-candidates", type=int, default=10)
    args = parser.parse_args()
    if len(args.trip) != 8 or not args.trip.isascii() or not args.trip.isdecimal():
        parser.error("--trip must be an eight-digit training trip ID")
    if not np.isfinite(args.radius_m) or args.radius_m <= 0 or args.max_candidates < 1:
        parser.error("Candidate radius and limit must be positive")
    output = args.output or Path("output/trips") / args.trip
    try:
        summary = inspect_trip(args.training_dir / args.trip, output, args.radius_m, args.max_candidates)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote map.html, candidates.csv and summary.json to {output}")


if __name__ == "__main__":
    main()
