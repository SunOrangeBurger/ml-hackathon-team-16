"""Run the deployed model and decoder without loading reference labels."""

import numpy as np
import joblib
import os

from inspect_trip import load_reference, load_trip
from model_features import FEATURE_NAMES, aligned_labels, candidate_features, decoder_observations, score_groups
from route_decoder import decode


def load_artifact(path):
    # Artifact paths are controlled by the backend operator.
    artifact = joblib.load(path)
    if tuple(artifact["features"]) != FEATURE_NAMES:
        raise ValueError("Model feature schema does not match inference code")
    return artifact


def predict_trip(folder, artifact):
    trip = load_trip(folder, include_reference=False)
    groups, features, _, _ = candidate_features(
        trip, artifact["radius_m"], artifact["max_candidates"]
    )
    mode = os.environ.get("PATHSNAP_DECODER", "weighted")
    if mode not in {"weighted", "hmm"}:
        raise ValueError("PATHSNAP_DECODER must be weighted or hmm")
    if mode == "hmm":
        scored = [
            [{"arc_id": arc, "arc_fraction": fraction, "probability": 1 / len(group)}
             for arc, _, fraction in group]
            for group in groups
        ]
    else:
        scored = score_groups(artifact["model"], groups, features)
    observations = decoder_observations(trip.track, scored)
    assignments, sections, warnings = decode(
        observations, trip.nodes, trip.arcs, probabilistic_hmm=mode == "hmm",
        emission_sigma_m=float(os.environ.get("PATHSNAP_HMM_SIGMA_M", "5")),
    )
    return trip, groups, observations, assignments, sections, warnings


def evaluate_assignments(folder, trip, assignments):
    trip.route = load_reference(folder, len(trip.arcs))
    labels = aligned_labels(trip)
    predictions = np.array([-1 if arc is None else arc for arc in assignments])
    eligible = labels >= 0
    count = int(eligible.sum())
    accuracy = float(np.mean(predictions[eligible] == labels[eligible])) if count else None
    return labels, {"eligible_observations": count, "provisional_accuracy": accuracy}
