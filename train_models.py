"""Compare five models by trip, then tune the winner with Bayesian search."""

import argparse
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from model_features import FEATURE_NAMES, MAX_CANDIDATES, RADIUS_M
from review_one import write_csv, write_json


SEED = 42
MODELS = {
    "logistic": (lambda **p: make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, **p)),
                 {"C": 1.0}, {"C": np.logspace(-3, 2, 15)}),
    "naive_bayes": (GaussianNB, {"var_smoothing": 1e-9},
                    {"var_smoothing": np.logspace(-12, -1, 16)}),
    "decision_tree": (lambda **p: DecisionTreeClassifier(random_state=SEED, **p),
                      {"max_depth": 8, "min_samples_leaf": 20},
                      {"max_depth": [4, 8, 12, 16], "min_samples_leaf": [10, 20, 50, 100]}),
    "random_forest": (lambda **p: RandomForestClassifier(n_estimators=32, n_jobs=2, random_state=SEED, **p),
                      {"max_depth": 10, "min_samples_leaf": 20},
                      {"max_depth": [6, 10, 14, 18], "min_samples_leaf": [10, 20, 50, 100]}),
    "hist_gradient_boosting": (lambda **p: HistGradientBoostingClassifier(max_iter=75, random_state=SEED, **p),
                               {"max_leaf_nodes": 15, "learning_rate": 0.1},
                               {"max_leaf_nodes": [7, 15, 31, 63], "learning_rate": [.025, .05, .1, .2]}),
}


class Dataset:
    def __init__(self, folder):
        self.trips = []
        for file in sorted(folder.glob("*.npz")):
            with np.load(file) as cached:
                if tuple(cached["schema"]) != FEATURE_NAMES:
                    raise ValueError(f"Feature schema mismatch in {file}")
                self.trips.append((file.stem, cached["features"], cached["indices"],
                                   cached["arcs"], cached["labels"]))

    def pack(self, trip_indices, sample=0):
        rng = np.random.default_rng(SEED)
        features, targets, observations = [], [], []
        offset = 0
        for i in trip_indices:
            _, x, indices, arcs, labels = self.trips[i]
            selected = labels >= 0
            eligible = np.flatnonzero(selected)
            if sample and len(eligible) > sample:
                selected[:] = False
                selected[rng.choice(eligible, sample, replace=False)] = True
            include = selected[indices]
            features.append(x[include])
            targets.append(arcs[include] == labels[indices[include]])
            observations.append(indices[include] + offset)
            offset += len(labels)
        return np.concatenate(features), np.concatenate(targets), np.concatenate(observations), sum(
            np.count_nonzero(self.trips[i][4] >= 0) for i in trip_indices
        ), offset


def evaluate(model, packed):
    x, y, observations, labeled, total = packed
    start = perf_counter()
    p = model.predict_proba(x)[:, 1]
    inference = perf_counter() - start
    precisions, recalls, f1, _ = precision_recall_fscore_support(
        y, p >= .5, average="macro", zero_division=0
    )
    highest = np.full(total, -np.inf)
    np.maximum.at(highest, observations, p)
    tied = np.flatnonzero(p == highest[observations])
    first = np.full(total, len(y))
    np.minimum.at(first, observations[tied], tied)
    hits = int(np.count_nonzero(y[first[first < len(y)]]))
    return {
        "macro_precision": precisions, "macro_recall": recalls, "macro_f1": f1,
        "top_1_accuracy": hits / labeled, "candidate_miss_rate": 1 - int(y.sum()) / labeled,
        "log_loss": log_loss(y, p, labels=[False, True]), "inference_s": inference,
        "labeled_observations": labeled, "excluded_observations": total - labeled,
        "confusion_matrix": confusion_matrix(y, p >= .5, labels=[False, True]).tolist(),
    }


def fit(factory, params, packed):
    start = perf_counter()
    model = factory(**params).fit(packed[0], packed[1])
    return model, perf_counter() - start


def cross_validate(factory, params, folds):
    results = []
    for train, validation in folds:
        model, seconds = fit(factory, params, train)
        results.append({**evaluate(model, validation), "training_s": seconds})
    return results


def bayesian_search(factory, default, grid, folds, trials):
    options = list(ParameterGrid(grid))
    keys = sorted(grid)
    encoded = np.array([[float(option[key]) for key in keys] for option in options])
    for column, key in enumerate(keys):
        if key in {"C", "var_smoothing", "learning_rate"}:
            encoded[:, column] = np.log10(encoded[:, column])
    encoded = (encoded - encoded.min(axis=0)) / np.maximum(np.ptp(encoded, axis=0), 1e-12)
    seen = [options.index(default)] if default in options else [0]
    history = []
    rng = np.random.default_rng(SEED)
    for step in range(min(trials, len(options))):
        if step == len(seen):
            remaining = sorted(set(range(len(options))) - set(seen))
            if len(seen) < 3:
                selected = int(rng.choice(remaining))
            else:
                objective = [row["cv_macro_f1"] for row in history]
                surrogate = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6,
                                                      normalize_y=True, optimizer=None)
                surrogate.fit(encoded[seen], objective)
                mean, deviation = surrogate.predict(encoded[remaining], return_std=True)
                improvement = mean - max(objective) - .001
                z = improvement / np.maximum(deviation, 1e-12)
                acquisition = improvement * ndtr(z) + deviation * np.exp(-z**2 / 2) / np.sqrt(2 * np.pi)
                selected = remaining[int(acquisition.argmax())]
            seen.append(selected)
        params = options[seen[step]]
        results = cross_validate(factory, params, folds)
        history.append({"trial": step + 1, "params": params, "fold_metrics": results,
                        "cv_macro_f1": float(np.mean([r["macro_f1"] for r in results]))})
        print(f"Bayesian trial {step + 1}: {history[-1]['cv_macro_f1']:.4f} {params}", flush=True)
    return history


def train(feature_dir, output, sample, trials):
    training = Dataset(feature_dir / "train")
    testing = Dataset(feature_dir / "student_test")
    if (len(training.trips), len(testing.trips)) != (80, 10):
        raise ValueError("Expected feature caches for 80 training and 10 student-testing trips")
    output.mkdir(parents=True, exist_ok=True)
    trip_indices = np.arange(len(training.trips))
    splits = list(GroupKFold(n_splits=3).split(trip_indices, groups=trip_indices))
    folds = [(training.pack(a, sample), training.pack(b)) for a, b in splits]
    full_train = training.pack(trip_indices, sample)
    test = testing.pack(range(len(testing.trips)))
    comparison = []
    for name, (factory, default, _) in MODELS.items():
        cv = cross_validate(factory, default, folds)
        model, seconds = fit(factory, default, full_train)
        row = {"model": name, "params": default, "fold_metrics": cv,
               "cv_macro_f1": float(np.mean([r["macro_f1"] for r in cv])),
               "student_test": evaluate(model, test), "training_s": seconds}
        comparison.append(row)
        print(f"{name}: CV macro F1 {row['cv_macro_f1']:.4f}", flush=True)
    winner = max(comparison, key=lambda row: row["cv_macro_f1"])
    name = winner["model"]
    factory, default, grid = MODELS[name]
    history = bayesian_search(factory, default, grid, folds, trials)
    best = max(history, key=lambda row: row["cv_macro_f1"])
    all_eligible = training.pack(trip_indices) if sample else full_train
    untuned, untuned_seconds = fit(factory, default, all_eligible)
    untuned_final = {
        **winner, "student_test": evaluate(untuned, test), "training_s": untuned_seconds,
    }
    model, seconds = fit(factory, best["params"], all_eligible)
    tuned = {**best, "student_test": evaluate(model, test), "training_s": seconds}
    write_json(output / "experiments.json", {
        "seed": SEED, "fit_observation_cap_per_trip": sample,
        "selection_objective": "mean train-only grouped CV candidate macro F1",
        "validation_trip_ids": [[training.trips[i][0] for i in b] for _, b in splits],
        "comparison": comparison, "selected_model": name, "full_refit_untuned": untuned_final,
        "bayesian_search_space": {key: list(values) for key, values in grid.items()},
        "bayesian_trials": history, "selected_configuration": tuned,
    })
    write_csv(output / "model_comparison.csv", [
        {"model": row["model"], "cv_macro_f1": row["cv_macro_f1"], **row["student_test"]}
        for row in [*comparison, {**untuned_final, "model": name + "_untuned_full"},
                    {"model": name + "_tuned_full", **tuned}]
    ])
    joblib.dump({"model": model, "name": name, "features": FEATURE_NAMES,
                 "radius_m": RADIUS_M, "max_candidates": MAX_CANDIDATES},
                output / "model.joblib", compress=3)
    print(f"Exported {name} to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("output/features"))
    parser.add_argument("--output", type=Path, default=Path("output/model"))
    parser.add_argument("--sample", type=int, default=300, help="Fit observations per trip, or 0 for all")
    parser.add_argument("--trials", type=int, default=6)
    args = parser.parse_args()
    if args.sample < 0 or args.trials < 4:
        parser.error("--sample must be nonnegative and --trials must be at least 4")
    train(args.features, args.output, args.sample, args.trials)
