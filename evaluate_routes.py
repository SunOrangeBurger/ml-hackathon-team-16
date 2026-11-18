"""Compare independent model predictions and Viterbi on student-testing trips."""

import argparse
from difflib import SequenceMatcher
from itertools import groupby
from pathlib import Path

import numpy as np

from inspect_trip import save_map
from learned_inference import evaluate_assignments, load_artifact, predict_trip
from review_one import write_csv, write_json
from route_decoder import decode


def sequence_similarity(labels, predictions):
    expected = [arc for arc, _ in groupby(labels)]
    actual = [arc for arc, _ in groupby(predictions)]
    return SequenceMatcher(None, expected, actual, autojunk=False).ratio()


def evaluate(artifact_path, data_dir, output):
    artifact = load_artifact(artifact_path)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    folders = [folder for folder in sorted(data_dir.iterdir()) if folder.is_dir()]
    if len(folders) != 10:
        raise ValueError("Route evaluation expects the ten student-testing trips")
    for folder in folders:
        trip, groups, observations, decoded, sections, warnings = predict_trip(folder, artifact)
        labels, _ = evaluate_assignments(folder, trip, decoded)
        eligible = labels >= 0
        independent = [
            {**row, "candidates": sorted(
                row["candidates"], key=lambda candidate: candidate["probability"], reverse=True
            )[:1]}
            for row in observations
        ]
        top_1, top_sections, top_warnings = decode(independent, trip.nodes, trip.arcs)
        row = {"trip_id": folder.name, "observations": len(labels), "eligible": int(eligible.sum())}
        for name, assignments, lines, reasons in (
            ("classifier", top_1, top_sections, top_warnings),
            ("viterbi", decoded, sections, warnings),
        ):
            predictions = np.array([-1 if arc is None else arc for arc in assignments])
            row[f"{name}_correct"] = int(np.count_nonzero((predictions == labels) & eligible))
            row[f"{name}_accuracy"] = row[f"{name}_correct"] / row["eligible"] if row["eligible"] else None
            row[f"{name}_sequence_similarity"] = sequence_similarity(labels[eligible], predictions[eligible])
            row[f"{name}_sections"] = len(lines)
            row[f"{name}_impossible_transitions"] = next(
                (int(reason.split(": ")[1]) for reason in reasons if reason.startswith("impossible_transition:")),
                0,
            )
        results.append(row)
        print(folder.name, row["classifier_accuracy"], row["viterbi_accuracy"], flush=True)
        if folder == folders[0]:
            save_map(output / "student-test-map.html", trip, groups, sections)
    write_csv(output / "route_comparison.csv", results)
    eligible_total = sum(row["eligible"] for row in results)
    write_json(output / "route_summary.json", {
        "student_test_trips": len(results), "eligible_observations": eligible_total,
        **{
            name: {
                "pointwise_accuracy": sum(row[f"{name}_correct"] for row in results) / eligible_total,
                "mean_sequence_similarity": float(np.mean([row[f"{name}_sequence_similarity"] for row in results])),
                "sections": sum(row[f"{name}_sections"] for row in results),
                "impossible_transitions": sum(row[f"{name}_impossible_transitions"] for row in results),
            }
            for name in ("classifier", "viterbi")
        },
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--student-test", type=Path, default=Path("Student folder/Student folder/student_test"))
    parser.add_argument("--output", type=Path, default=Path("output/evaluation"))
    args = parser.parse_args()
    evaluate(args.model, args.student_test, args.output)
