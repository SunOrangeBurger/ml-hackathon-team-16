"""Export the five compared classifiers and measure both evaluation splits."""

import json
from pathlib import Path

import joblib
import numpy as np

from model_features import FEATURE_NAMES, MAX_CANDIDATES, RADIUS_M
from train_models import Dataset, MODELS
from tune_decoder import measure, prepare


def main():
    destination = Path("artifacts/models")
    destination.mkdir(parents=True, exist_ok=True)
    training = Dataset(Path("output/features/train"))
    packed = training.pack(range(len(training.trips)))
    catalog = []
    for name, (factory, parameters, _) in MODELS.items():
        path = destination / f"{name}.joblib"
        if path.exists():
            artifact = joblib.load(path)
        else:
            model = factory(**parameters).fit(packed[0], packed[1])
            artifact = {
                "model": model, "name": name, "features": FEATURE_NAMES,
                "radius_m": RADIUS_M, "max_candidates": MAX_CANDIDATES,
            }
            joblib.dump(artifact, path, compress=3)
        splits = {}
        for split, folder in (
            ("student_test", Path("Student folder/Student folder/student_test")),
            ("evaluator_test", Path("evaluator_test")),
        ):
            results = measure(prepare(folder, artifact), {})
            count = sum(row["eligible"] for row in results["trips"])
            correct = sum(row["correct"] for row in results["trips"])
            splits[split] = {
                "accuracy": correct / count, "eligible": count,
                "correct": correct, "trips": results["trips"],
            }
        catalog.append({
            "id": name, "label": name.replace("_", " ").title(),
            "selection_accuracy": splits["student_test"]["accuracy"],
            "splits": splits,
            "combined_accuracy": sum(x["correct"] for x in splits.values())
                / sum(x["eligible"] for x in splits.values()),
        })
        print(name, {k: v["accuracy"] for k, v in splits.items()}, flush=True)
        ordered = sorted(catalog, key=lambda row: row["selection_accuracy"], reverse=True)
        (destination / "catalog.json").write_text(json.dumps(ordered, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
