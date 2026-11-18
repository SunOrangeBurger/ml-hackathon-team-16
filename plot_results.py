"""Generate comparison figures from measured experiment logs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(log, output):
    experiments = json.loads(log.read_text(encoding="utf-8"))
    rows = experiments["comparison"]
    names = ["Logistic regression", "Gaussian naive Bayes", "Decision tree",
             "Random forest", "Histogram gradient boosting"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4.2), layout="constrained", sharey=True)
    panels = (
        ("Grouped CV macro F1", [row["cv_macro_f1"] for row in rows]),
        ("Student-test top-1", [row["student_test"]["top_1_accuracy"] for row in rows]),
        ("Student-test macro F1", [row["student_test"]["macro_f1"] for row in rows]),
        ("Student-test log loss", [row["student_test"]["log_loss"] for row in rows]),
    )
    for axis, (title, values) in zip(axes, panels):
        bars = axis.barh(names, values, color="#167368")
        axis.set_title(title)
        axis.set_xlim(0, max(values) * 1.20)
        axis.bar_label(bars, fmt="%.3f", padding=3)
        axis.grid(axis="x", alpha=.2)
        axis.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.savefig(output, dpi=155)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=Path("output/model/experiments.json"))
    parser.add_argument("--output", type=Path, default=Path("output/model/model_comparison.png"))
    args = parser.parse_args()
    plot(args.log, args.output)
