"""Relate out-of-sample forecast quality to paired economic uplift."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "data" / "misspecification_grid.csv"
    if not source.exists():
        raise FileNotFoundError("run experiments/misspecification.py first")
    frame = pd.read_csv(source)
    correlation = spearmanr(frame["brier_score"], frame["mean_uplift"])
    correct = (frame.assumed_stay == 0.94) & (frame.assumed_beta == 0.20)

    fig, axis = plt.subplots(figsize=(7.5, 5.2), layout="constrained")
    scatter = axis.scatter(
        frame.brier_score,
        frame.mean_uplift,
        c=frame.assumed_beta,
        cmap="viridis",
        s=55,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )
    axis.scatter(
        frame.loc[correct, "brier_score"],
        frame.loc[correct, "mean_uplift"],
        marker="*",
        s=240,
        color="#e45756",
        edgecolor="black",
        linewidth=0.7,
        label="Correct model",
    )
    axis.axhline(0.0, color="#444444", linewidth=0.8)
    axis.set_xlabel("Next-trade Brier score (lower is better)")
    axis.set_ylabel("Mean paired P&L uplift over inventory-only")
    axis.set_title("Inference quality and market-making value need not rank identically")
    axis.legend()
    fig.colorbar(scatter, ax=axis, label="Assumed emission strength β")
    fig.savefig(root / "figures" / "inference_vs_decision.png", dpi=200)
    plt.close(fig)

    summary = {
        "spearman_brier_vs_uplift": float(correlation.statistic),
        "models": len(frame),
    }
    (root / "data" / "inference_vs_decision.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
