"""Train a CUDA listwise candidate ranker and report grouped validation metrics."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import joblib
from sklearn.model_selection import GroupKFold

from model_features import FEATURE_NAMES, MAX_CANDIDATES


SEED = 42


@dataclass
class TripGroups:
    trip_id: str
    features: np.ndarray
    mask: np.ndarray
    targets: np.ndarray


def load_trips(folder):
    trips = []
    for path in sorted(folder.glob("*.npz")):
        with np.load(path) as cached:
            if tuple(cached["schema"]) != FEATURE_NAMES:
                raise ValueError(f"Feature schema mismatch in {path}")
            rows = cached["features"].astype(np.float32)
            indices = cached["indices"].astype(int)
            arcs = cached["arcs"].astype(int)
            labels = cached["labels"].astype(int)
        eligible = np.flatnonzero(labels >= 0)
        features = np.zeros((len(eligible), MAX_CANDIDATES, len(FEATURE_NAMES)), np.float32)
        mask = np.zeros((len(eligible), MAX_CANDIDATES), bool)
        targets = np.full(len(eligible), -1, np.int64)
        starts = np.searchsorted(indices, eligible, side="left")
        ends = np.searchsorted(indices, eligible, side="right")
        for output_index, (observation, start, end) in enumerate(zip(eligible, starts, ends)):
            count = min(end - start, MAX_CANDIDATES)
            if not count:
                continue
            features[output_index, :count] = rows[start:start + count]
            mask[output_index, :count] = True
            matches = np.flatnonzero(arcs[start:start + count] == labels[observation])
            if len(matches):
                targets[output_index] = matches[0]
        trips.append(TripGroups(path.stem, features, mask, targets))
    return trips


class Ranker(torch.nn.Module):
    def __init__(self, width, depth, dropout):
        super().__init__()
        layers = []
        size = len(FEATURE_NAMES)
        for _ in range(depth):
            layers.extend((torch.nn.Linear(size, width), torch.nn.SiLU(), torch.nn.Dropout(dropout)))
            size = width
        layers.append(torch.nn.Linear(size, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, values):
        return self.network(values).squeeze(-1)


def concatenate(trips, indices):
    return tuple(np.concatenate([getattr(trips[index], field) for index in indices]) for field in (
        "features", "mask", "targets"
    ))


def normalization(features, mask):
    rows = features[mask]
    mean = rows.mean(axis=0, dtype=np.float64).astype(np.float32)
    deviation = rows.std(axis=0, dtype=np.float64).astype(np.float32)
    deviation[deviation < 1e-6] = 1
    return mean, deviation


@torch.inference_mode()
def evaluate(model, data, mean, deviation, device, batch_size):
    features, mask, targets = data
    correct = 0
    observed = len(targets)
    scored = 0
    model.eval()
    for start in range(0, observed, batch_size):
        stop = min(start + batch_size, observed)
        x = torch.from_numpy((features[start:stop] - mean) / deviation).to(device)
        valid = torch.from_numpy(mask[start:stop]).to(device)
        scores = model(x).masked_fill(~valid, -torch.inf)
        prediction = scores.argmax(dim=1).cpu().numpy()
        expected = targets[start:stop]
        correct += int(np.count_nonzero((expected >= 0) & (prediction == expected)))
        scored += int(np.count_nonzero(expected >= 0))
    return {
        "top_1_accuracy": correct / observed,
        "candidate_coverage": scored / observed,
        "correct": correct,
        "observations": observed,
    }


def evaluate_sklearn(model, data):
    features, mask, targets = data
    scores = np.full(mask.shape, -np.inf, dtype=float)
    scores[mask] = model.predict_proba(features[mask])[:, 1]
    prediction = scores.argmax(axis=1)
    correct = int(np.count_nonzero((targets >= 0) & (prediction == targets)))
    covered = int(np.count_nonzero(targets >= 0))
    return {
        "top_1_accuracy": correct / len(targets),
        "candidate_coverage": covered / len(targets),
        "correct": correct,
        "observations": len(targets),
    }


def train_fold(train_data, validation_data, args, device):
    mean, deviation = normalization(train_data[0], train_data[1])
    model = Ranker(args.width, args.depth, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = np.random.default_rng(SEED)
    trainable = np.flatnonzero(train_data[2] >= 0)
    best = None
    started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        generator.shuffle(trainable)
        losses = []
        for offset in range(0, len(trainable), args.batch_size):
            selected = trainable[offset:offset + args.batch_size]
            x = torch.from_numpy((train_data[0][selected] - mean) / deviation).to(device)
            valid = torch.from_numpy(train_data[1][selected]).to(device)
            target = torch.from_numpy(train_data[2][selected]).to(device)
            logits = model(x).masked_fill(~valid, -1e4)
            loss = torch.nn.functional.cross_entropy(logits, target, label_smoothing=args.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        metrics = evaluate(model, validation_data, mean, deviation, device, args.batch_size)
        metrics.update(epoch=epoch, loss=float(np.mean(losses)))
        if best is None or metrics["top_1_accuracy"] > best["metrics"]["top_1_accuracy"]:
            best = {
                "metrics": metrics,
                "state": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                "mean": mean,
                "deviation": deviation,
            }
        print(
            f"epoch={epoch:02d} loss={metrics['loss']:.4f} "
            f"validation_top1={metrics['top_1_accuracy']:.4%}",
            flush=True,
        )
    best["training_s"] = perf_counter() - started
    return best


def run(args):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training command")
    device = torch.device("cuda")
    training = load_trips(args.features / "train")
    testing = load_trips(args.features / "student_test")
    if (len(training), len(testing)) != (80, 10):
        raise ValueError("Expected 80 training and 10 student-test feature files")
    trip_indices = np.arange(len(training))
    folds = list(GroupKFold(n_splits=3).split(trip_indices, groups=trip_indices))
    selected_folds = folds if args.fold < 0 else [folds[args.fold]]
    baseline = joblib.load(args.baseline_model)["model"]
    output = {
        "device": torch.cuda.get_device_name(0),
        "parameters": vars(args) | {
            "features": str(args.features),
            "output": str(args.output),
            "baseline_model": str(args.baseline_model),
        },
        "folds": [],
    }
    for fold_index, (train_indices, validation_indices) in enumerate(selected_folds):
        print(f"training fold {fold_index + 1}/{len(selected_folds)} on {device}", flush=True)
        validation_data = concatenate(training, validation_indices)
        trained = train_fold(
            concatenate(training, train_indices),
            validation_data,
            args,
            device,
        )
        output["folds"].append({
            "validation_trip_ids": [training[index].trip_id for index in validation_indices],
            "metrics": trained["metrics"],
            "training_s": trained["training_s"],
            "deployed_baseline": evaluate_sklearn(baseline, validation_data),
        })
    output["mean_cv_top_1"] = float(np.mean([
        fold["metrics"]["top_1_accuracy"] for fold in output["folds"]
    ]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"mean_cv_top1={output['mean_cv_top_1']:.4%}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("output/features"))
    parser.add_argument("--baseline-model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--output", type=Path, default=Path("output/gpu/ranker-results.json"))
    parser.add_argument("--fold", type=int, choices=(-1, 0, 1, 2), default=0,
                        help="Grouped fold to run, or -1 for all three")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    run(parser.parse_args())
