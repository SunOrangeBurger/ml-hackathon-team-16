"""Tune decoder continuity settings on development trips without using evaluator data."""

import argparse
import json
from difflib import SequenceMatcher
from itertools import groupby, product
from pathlib import Path

import numpy as np

from inspect_trip import load_trip
from learned_inference import load_artifact
from model_features import aligned_labels, candidate_features, decoder_observations, score_groups
from route_decoder import decode


def similarity(expected, actual):
    expected = [value for value, _ in groupby(expected)]
    actual = [value for value, _ in groupby(actual)]
    return SequenceMatcher(None, expected, actual, autojunk=False).ratio()


def prepare(data_dir, artifact):
    prepared = []
    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue
        trip = load_trip(folder)
        groups, features, _, _ = candidate_features(
            trip, artifact["radius_m"], artifact["max_candidates"]
        )
        observations = decoder_observations(
            trip.track, score_groups(artifact["model"], groups, features)
        )
        labels = aligned_labels(trip)
        prepared.append((folder.name, trip, observations, labels))
    return prepared


def measure(prepared, config):
    rows = []
    for trip_id, trip, observations, labels in prepared:
        assignments, sections, warnings = decode(observations, trip.nodes, trip.arcs, **config)
        predictions = np.array([-1 if value is None else value for value in assignments])
        eligible = labels >= 0
        rows.append({
            "trip_id": trip_id,
            "eligible": int(eligible.sum()),
            "correct": int(np.count_nonzero((predictions == labels) & eligible)),
            "sequence_similarity": similarity(labels[eligible], predictions[eligible]),
            "sections": len(sections),
            "impossible_transitions": next((
                int(warning.split(": ")[1])
                for warning in warnings
                if warning.startswith("impossible_transition:")
            ), 0),
        })
    eligible = sum(row["eligible"] for row in rows)
    return {
        "config": config,
        "pointwise_accuracy": sum(row["correct"] for row in rows) / eligible,
        "mean_sequence_similarity": float(np.mean([row["sequence_similarity"] for row in rows])),
        "sections": sum(row["sections"] for row in rows),
        "impossible_transitions": sum(row["impossible_transitions"] for row in rows),
        "trips": rows,
    }


def run(args):
    artifact = load_artifact(args.model)
    prepared = prepare(args.data, artifact)
    if len(prepared) != 10:
        raise ValueError("Decoder tuning expects ten development trips")
    results = []
    settings = product(
        args.backward_tolerance,
        args.transition_scale,
        args.distance_slack,
    )
    for backward, scale, slack in settings:
        config = {
            "backward_tolerance_m": backward,
            "transition_scale_m": scale,
            "distance_slack_m": slack,
        }
        result = measure(prepared, config)
        results.append(result)
        print(
            f"accuracy={result['pointwise_accuracy']:.4%} "
            f"sequence={result['mean_sequence_similarity']:.4f} "
            f"sections={result['sections']} {config}",
            flush=True,
        )
    selected = max(
        results,
        key=lambda row: (row["pointwise_accuracy"], row["mean_sequence_similarity"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"selected": selected, "results": results}, indent=2), encoding="utf-8")
    print(f"selected={json.dumps(selected['config'])}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("Student folder/Student folder/student_test"),
    )
    parser.add_argument("--output", type=Path, default=Path("output/decoder-tuning/results.json"))
    parser.add_argument("--backward-tolerance", type=float, nargs="+", default=[0, 5, 15, 30])
    parser.add_argument("--transition-scale", type=float, nargs="+", default=[15, 30, 60, 120])
    parser.add_argument("--distance-slack", type=float, nargs="+", default=[25, 50, 100])
    run(parser.parse_args())
