"""Evaluate transition, emission, and economic-map misspecification."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from order_flow_mm.backtest import BacktestResult, run_backtest
from order_flow_mm.config import FilterParams, MarketParams, symmetric_transition
from order_flow_mm.metrics import inference_metrics, paired_uplift
from order_flow_mm.policies import BayesianPolicy, InventoryPolicy
from order_flow_mm.simulator import MarketPath, generate_market_path

STAY_GRID = (0.70, 0.80, 0.86, 0.90, 0.94, 0.97, 0.99)
BETA_GRID = (0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
MU_GRID = (-0.12, 0.00, 0.06, 0.12, 0.18, 0.30)


def evaluate_beliefs(
    beliefs: FilterParams,
    market: MarketParams,
    paths: list[MarketPath],
    inventory_results: list[BacktestResult],
) -> dict[str, float]:
    bayesian_results = [run_backtest(BayesianPolicy(market, beliefs), path, market) for path in paths]
    belief_metrics = [
        inference_metrics(result, path) for result, path in zip(bayesian_results, paths, strict=True)
    ]
    return {
        **paired_uplift(bayesian_results, inventory_results),
        "brier_score": float(np.mean([item.brier_score for item in belief_metrics])),
        "log_loss": float(np.mean([item.log_loss for item in belief_metrics])),
        "map_accuracy": float(np.mean([item.map_accuracy for item in belief_metrics])),
    }


def plot_heatmap(frame: pd.DataFrame, destination: Path, true_stay: float, true_beta: float) -> None:
    pivot = frame.pivot(index="assumed_beta", columns="assumed_stay", values="mean_uplift")
    fig, axis = plt.subplots(figsize=(8.2, 5.6), layout="constrained")
    maximum = float(np.max(np.abs(pivot.to_numpy())))
    image = axis.imshow(
        pivot.to_numpy(),
        origin="lower",
        aspect="auto",
        cmap="RdYlGn",
        vmin=-maximum,
        vmax=maximum,
    )
    axis.set_xticks(np.arange(len(pivot.columns)), [f"{value:.2f}" for value in pivot.columns])
    axis.set_yticks(np.arange(len(pivot.index)), [f"{value:.2f}" for value in pivot.index])
    axis.set_xlabel("Assumed pressure-state persistence")
    axis.set_ylabel("Assumed emission strength β")
    axis.set_title("Bayesian P&L uplift over inventory-only under misspecification")
    true_x = int(np.argmin(np.abs(pivot.columns.to_numpy() - true_stay)))
    true_y = int(np.argmin(np.abs(pivot.index.to_numpy() - true_beta)))
    axis.scatter(true_x, true_y, marker="*", s=180, color="black", label="Correct model")
    axis.legend(loc="upper left", fontsize=8)
    fig.colorbar(image, ax=axis, label="Mean paired terminal-P&L uplift")
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=80)
    parser.add_argument("--horizon", type=int, default=1_000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data_dir, figure_dir = root / "data", root / "figures"
    data_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)

    market = MarketParams(horizon=args.horizon)
    paths = [generate_market_path(market, seed) for seed in range(30_000, 30_000 + args.paths)]
    inventory_results = [run_backtest(InventoryPolicy(market), path, market) for path in paths]
    rows = []
    for stay in STAY_GRID:
        for beta in BETA_GRID:
            beliefs = FilterParams(symmetric_transition(stay), beta, market.mu)
            rows.append(
                {"assumed_stay": stay, "assumed_beta": beta, **evaluate_beliefs(
                    beliefs, market, paths, inventory_results
                )}
            )
    grid = pd.DataFrame(rows)
    grid.to_csv(data_dir / "misspecification_grid.csv", index=False)

    economic_rows = []
    for assumed_mu in MU_GRID:
        beliefs = FilterParams(market.transition, market.beta, assumed_mu)
        economic_rows.append(
            {"assumed_mu": assumed_mu, **evaluate_beliefs(
                beliefs, market, paths, inventory_results
            )}
        )
    pd.DataFrame(economic_rows).to_csv(
        data_dir / "economic_map_misspecification.csv", index=False
    )
    plot_heatmap(
        grid,
        figure_dir / "misspecification_heatmap.png",
        float(market.transition[0, 0]),
        market.beta,
    )
    print(grid.to_string(index=False))
    print("\nEconomic-map misspecification:\n", pd.DataFrame(economic_rows).to_string(index=False))


if __name__ == "__main__":
    main()
