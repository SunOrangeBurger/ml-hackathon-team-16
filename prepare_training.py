"""Cache reproducible features from the two supplied development splits."""

import argparse
from pathlib import Path

import numpy as np

from inspect_trip import load_trip
from model_features import FEATURE_NAMES, aligned_labels, candidate_features


def prepare(data, output):
    for split in ("train", "student_test"):
        destination = output / split
        destination.mkdir(parents=True, exist_ok=True)
        for folder in sorted((data / split).iterdir()):
            if not folder.is_dir():
                continue
            trip = load_trip(folder)
            _, features, indices, arcs = candidate_features(trip)
            labels = aligned_labels(trip)
            np.savez_compressed(destination / f"{folder.name}.npz", features=features,
                                indices=indices, arcs=arcs, labels=labels, schema=FEATURE_NAMES)
            print(f"{split}/{folder.name}: {len(labels)} observations, {len(features)} candidates", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("Student folder/Student folder"))
    parser.add_argument("--output", type=Path, default=Path("output/features"))
    args = parser.parse_args()
    prepare(args.data, args.output)
