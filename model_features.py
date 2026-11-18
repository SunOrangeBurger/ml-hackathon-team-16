"""Shared candidate features and weak supervision for PathSnap."""

import numpy as np

from inspect_trip import project_trip, rank_candidates
from review_one import align_reference


FEATURE_NAMES = ("distance_m", "arc_fraction", "rank", "heading_agreement", "heading_available", "speed_mps")
RADIUS_M = 75
MAX_CANDIDATES = 16


def candidate_features(trip, radius_m=RADIUS_M, limit=MAX_CANDIDATES):
    groups = rank_candidates(trip, radius_m, limit)
    points, nodes = project_trip(trip)
    dt = np.diff(trip.track[:, 2], prepend=trip.track[0, 2])
    movement = np.diff(points, axis=0, prepend=points[:1])
    speed_distance = np.linalg.norm(movement, axis=1)
    usable = (dt > 0) & (dt <= 10) & (speed_distance >= 2)
    starts, ends = nodes[trip.arcs].transpose(1, 0, 2)
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    rows = []
    indices = []
    arc_ids = []
    for i, group in enumerate(groups):
        if not group:
            continue
        ids, distances, fractions = map(np.asarray, zip(*group))
        ids = ids.astype(int)
        heading_available = usable[i] & (lengths[ids] > 0)
        agreement = np.divide(
            vectors[ids] @ movement[i], lengths[ids] * speed_distance[i],
            out=np.zeros(len(ids)), where=heading_available,
        ).clip(-1, 1)
        rows.append(np.column_stack([
            distances, fractions, np.arange(1, len(ids) + 1), agreement,
            heading_available, np.full(len(ids), speed_distance[i] / dt[i] if usable[i] else 0),
        ]))
        indices.extend([i] * len(ids))
        arc_ids.extend(ids)
    features = np.concatenate(rows).astype(np.float32) if rows else np.empty((0, len(FEATURE_NAMES)), np.float32)
    return groups, features, np.asarray(indices, dtype=int), np.asarray(arc_ids, dtype=int)


def aligned_labels(trip):
    positions, distances, _ = align_reference(trip)
    labels = trip.route[positions].copy()
    labels[distances > 50] = -1
    return labels


def score_groups(model, groups, features):
    probabilities = model.predict_proba(features)[:, 1] if len(features) else []
    offset = 0
    scored = []
    for group in groups:
        values = np.asarray(probabilities[offset:offset + len(group)], dtype=float).clip(1e-12, 1)
        if len(values):
            values /= values.sum()
        scored.append([
            {"arc_id": arc, "arc_fraction": fraction, "probability": float(score)}
            for (arc, _, fraction), score in zip(group, values)
        ])
        offset += len(group)
    return scored


def decoder_observations(track, scored):
    return [
        {
            "observation_index": i, "longitude": lon, "latitude": lat, "timestamp_s": time,
            "candidates": candidates,
        }
        for i, ((lon, lat, time), candidates) in enumerate(zip(track, scored))
    ]
