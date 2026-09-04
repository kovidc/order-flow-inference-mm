"""Run the main five-policy benchmark, sensitivity sweeps, nulls, and two figures."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from order_flow_mm.backtest import BacktestResult, run_backtest
from order_flow_mm.config import FilterParams, MarketParams, symmetric_transition
from order_flow_mm.metrics import inference_metrics, paired_uplift, summarize_results
from order_flow_mm.policies import (
    BayesianPolicy,
    FixedPolicy,
    InventoryPolicy,
    OraclePolicy,
    fit_rolling_policy,
)
from order_flow_mm.simulator import MarketPath, generate_market_path

POLICY_COLORS = {
    "Fixed": "#7f7f7f",
    "Inventory": "#4c78a8",
    "Rolling": "#f58518",
    "Bayesian": "#54a24b",
    "Oracle": "#b279a2",
}


def paths_for(params: MarketParams, seeds: range) -> list[MarketPath]:
    return [generate_market_path(params, seed) for seed in seeds]


def evaluate_main(
    market: MarketParams, evaluation_paths: list[MarketPath], training_paths: list[MarketPath]
) -> tuple[dict[str, list[BacktestResult]], dict[str, float]]:
    rolling = fit_rolling_policy(training_paths, market)
    beliefs = FilterParams(market.transition, market.beta, market.mu)
    all_results: dict[str, list[BacktestResult]] = {
        "Fixed": [],
        "Inventory": [],
        "Rolling": [],
        "Bayesian": [],
        "Oracle": [],
    }
    for path in evaluation_paths:
        policies = (
            FixedPolicy(1.0 / market.k_fill),
            InventoryPolicy(market),
            rolling,
            BayesianPolicy(market, beliefs),
            OraclePolicy(market),
        )
        for policy in policies:
            all_results[policy.name].append(run_backtest(policy, path, market))
    rolling_fit = {
        "rolling_window": float(rolling.window),
        "rolling_intercept": rolling.intercept,
        "rolling_slope": rolling.slope,
    }
    return all_results, rolling_fit


def benchmark_frame(results: dict[str, list[BacktestResult]]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    baseline = results["Inventory"]
    for name, policy_results in results.items():
        row: dict[str, float | str] = {"policy": name, **summarize_results(policy_results)}
        row.update(paired_uplift(policy_results, baseline))
        rows.append(row)
    return pd.DataFrame(rows)


def inference_summary(
    results: list[BacktestResult], paths: list[MarketPath]
) -> dict[str, float]:
    metrics = [inference_metrics(result, path) for result, path in zip(results, paths, strict=True)]
    return {
        "mean_brier_score": float(np.mean([item.brier_score for item in metrics])),
        "mean_log_loss": float(np.mean([item.log_loss for item in metrics])),
        "mean_map_accuracy": float(np.mean([item.map_accuracy for item in metrics])),
    }


def sensitivity_point(market: MarketParams, seeds: range) -> dict[str, float]:
    beliefs = FilterParams(market.transition, market.beta, market.mu)
    inventory_results: list[BacktestResult] = []
    bayesian_results: list[BacktestResult] = []
    inference_rows = []
    for path in paths_for(market, seeds):
        inventory_results.append(run_backtest(InventoryPolicy(market), path, market))
        result = run_backtest(BayesianPolicy(market, beliefs), path, market)
        bayesian_results.append(result)
        inference_rows.append(inference_metrics(result, path))
    return {
        **paired_uplift(bayesian_results, inventory_results),
        "brier_score": float(np.mean([row.brier_score for row in inference_rows])),
        "log_loss": float(np.mean([row.log_loss for row in inference_rows])),
        "map_accuracy": float(np.mean([row.map_accuracy for row in inference_rows])),
    }


def run_sweeps(base: MarketParams, seeds: range) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    grids = {
        "beta": (0.0, 0.05, 0.10, 0.20, 0.30, 0.40),
        "mu": (0.0, 0.04, 0.08, 0.12, 0.20),
        "persistence": (0.65, 0.75, 0.85, 0.94, 0.98),
    }
    for experiment, values in grids.items():
        for value in values:
            if experiment == "persistence":
                market = replace(base, transition=symmetric_transition(value))
            else:
                market = replace(base, **{experiment: value})
            rows.append({"experiment": experiment, "value": value, **sensitivity_point(market, seeds)})
    return pd.DataFrame(rows)


def plot_example(
    path: MarketPath, result: BacktestResult, destination: Path, events: int = 300
) -> None:
    if result.posteriors is None:
        raise ValueError("example path figure requires Bayesian posterior history")
    count = min(events, path.horizon)
    time = np.arange(count)
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, layout="constrained")
    axes[0].step(time, path.regimes[:count], where="post", color="#222222", linewidth=1.2)
    axes[0].set_ylabel("Regime")
    axes[0].set_yticks([-1, 0, 1])
    labels = ("P(sell regime)", "P(neutral)", "P(buy regime)")
    colors = ("#e45756", "#9d9d9d", "#4c78a8")
    for index, (label, color) in enumerate(zip(labels, colors, strict=True)):
        axes[1].plot(time, result.posteriors[:count, index], label=label, color=color)
    axes[1].set_ylabel("Posterior")
    axes[1].legend(ncol=3, fontsize=8, loc="upper center")
    axes[2].plot(time, result.ask_distances[:count], label="Ask distance", color="#e45756")
    axes[2].plot(time, result.bid_distances[:count], label="Bid distance", color="#4c78a8")
    axes[2].set_ylabel("Quote distance")
    axes[2].legend(ncol=2, fontsize=8)
    axes[3].plot(np.arange(count + 1), result.wealth[: count + 1], color="#54a24b")
    axes[3].axhline(0.0, color="#888888", linewidth=0.7)
    axes[3].set_ylabel("Cumulative P&L")
    axes[3].set_xlabel("Event")
    fig.suptitle("Bayesian belief, quotes, and marked-to-market P&L on one held-out path")
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def plot_policy_comparison(frame: pd.DataFrame, destination: Path) -> None:
    order = ["Fixed", "Inventory", "Rolling", "Bayesian", "Oracle"]
    indexed = frame.set_index("policy").loc[order]
    means = indexed["mean_pnl"].to_numpy()
    errors = np.vstack(
        (means - indexed["pnl_ci_low"].to_numpy(), indexed["pnl_ci_high"].to_numpy() - means)
    )
    fig, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
    axis.bar(
        order,
        means,
        yerr=errors,
        capsize=4,
        color=[POLICY_COLORS[name] for name in order],
        edgecolor="white",
    )
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_ylabel("Mean terminal P&L (normalized units)")
    axis.set_title("Held-out paired policy comparison (95% confidence intervals)")
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=300, help="held-out paths in main benchmark")
    parser.add_argument("--sweep-paths", type=int, default=80, help="held-out paths per sweep point")
    parser.add_argument("--horizon", type=int, default=1_000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data_dir, figure_dir = root / "data", root / "figures"
    data_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)

    market = MarketParams(horizon=args.horizon)
    training = paths_for(market, range(1_000, 1_040))
    evaluation = paths_for(market, range(10_000, 10_000 + args.paths))
    results, rolling_fit = evaluate_main(market, evaluation, training)
    frame = benchmark_frame(results)
    frame.to_csv(data_dir / "benchmark.csv", index=False)
    bayesian_inference = inference_summary(results["Bayesian"], evaluation)
    inventory_mean = float(frame.loc[frame.policy == "Inventory", "mean_pnl"].iloc[0])
    bayesian_mean = float(frame.loc[frame.policy == "Bayesian", "mean_pnl"].iloc[0])
    oracle_mean = float(frame.loc[frame.policy == "Oracle", "mean_pnl"].iloc[0])
    denominator = oracle_mean - inventory_mean
    metadata = {
        "market": {
            "beta": market.beta,
            "mu": market.mu,
            "sigma": market.sigma,
            "pressure_stay": float(market.transition[0, 0]),
            "horizon": market.horizon,
        },
        **rolling_fit,
        **bayesian_inference,
        "value_of_information_recovery": (
            (bayesian_mean - inventory_mean) / denominator if denominator != 0.0 else None
        ),
    }
    (data_dir / "benchmark_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    sweep_frame = run_sweeps(market, range(20_000, 20_000 + args.sweep_paths))
    sweep_frame.to_csv(data_dir / "sensitivity_sweeps.csv", index=False)
    nulls = sweep_frame[
        ((sweep_frame.experiment == "beta") & (sweep_frame.value == 0.0))
        | ((sweep_frame.experiment == "mu") & (sweep_frame.value == 0.0))
    ]
    nulls.to_csv(data_dir / "null_controls.csv", index=False)

    plot_example(evaluation[0], results["Bayesian"][0], figure_dir / "example_path.png")
    plot_policy_comparison(frame, figure_dir / "policy_comparison.png")
    print(frame.to_string(index=False))
    print(json.dumps(metadata, indent=2))
    print("\nSensitivity sweeps:\n", sweep_frame.to_string(index=False))


if __name__ == "__main__":
    main()
