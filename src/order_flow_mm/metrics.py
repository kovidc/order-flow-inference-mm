"""Inference, path-level trading, and paired Monte Carlo metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from .backtest import BacktestResult
from .simulator import MarketPath

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TradingMetrics:
    terminal_pnl: float
    maximum_drawdown: float
    fill_rate: float
    rms_inventory: float
    maximum_absolute_inventory: int
    adverse_selection_markout: float


@dataclass(frozen=True)
class InferenceMetrics:
    brier_score: float
    log_loss: float
    map_accuracy: float


def path_trading_metrics(result: BacktestResult) -> TradingMetrics:
    peaks = np.maximum.accumulate(result.wealth)
    maximum_drawdown = float(np.max(peaks - result.wealth))
    executed_markouts = result.adverse_markouts[result.fills]
    markout = float(executed_markouts.mean()) if len(executed_markouts) else 0.0
    return TradingMetrics(
        terminal_pnl=result.terminal_pnl,
        maximum_drawdown=maximum_drawdown,
        fill_rate=float(result.fills.mean()),
        rms_inventory=float(np.sqrt(np.mean(result.inventory.astype(float) ** 2))),
        maximum_absolute_inventory=int(np.max(np.abs(result.inventory))),
        adverse_selection_markout=markout,
    )


def inference_metrics(result: BacktestResult, path: MarketPath) -> InferenceMetrics:
    if result.predicted_buy is None or result.posteriors is None:
        raise ValueError("inference metrics require a Bayesian backtest result")
    outcomes = (path.trade_signs == 1).astype(float)
    probabilities = np.clip(result.predicted_buy, 1e-12, 1.0 - 1e-12)
    brier = float(np.mean((probabilities - outcomes) ** 2))
    log_loss = float(
        -np.mean(outcomes * np.log(probabilities) + (1.0 - outcomes) * np.log(1 - probabilities))
    )
    state_values = np.array([-1, 0, 1])
    map_states = state_values[np.argmax(result.posteriors, axis=1)]
    accuracy = float(np.mean(map_states == path.regimes))
    return InferenceMetrics(brier, log_loss, accuracy)


def lower_tail_cvar(values: FloatArray, alpha: float = 0.05) -> float:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")
    count = max(1, int(np.ceil(alpha * len(values))))
    return float(np.sort(values)[:count].mean())


def confidence_interval(values: FloatArray, level_multiplier: float = 1.96) -> tuple[float, float]:
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean
    half_width = level_multiplier * float(np.std(values, ddof=1)) / np.sqrt(len(values))
    return mean - half_width, mean + half_width


def summarize_results(results: list[BacktestResult]) -> dict[str, float]:
    """Aggregate path metrics, keeping terminal-P&L dispersion and tail risk explicit."""
    if not results:
        raise ValueError("cannot summarize an empty result list")
    path_metrics = [asdict(path_trading_metrics(result)) for result in results]
    pnl = np.asarray([row["terminal_pnl"] for row in path_metrics])
    ci_low, ci_high = confidence_interval(pnl)
    summary = {
        "paths": float(len(results)),
        "mean_pnl": float(pnl.mean()),
        "pnl_std": float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0,
        "pnl_ci_low": ci_low,
        "pnl_ci_high": ci_high,
        "pnl_cvar_05": lower_tail_cvar(pnl),
    }
    for name in (
        "maximum_drawdown",
        "fill_rate",
        "rms_inventory",
        "maximum_absolute_inventory",
        "adverse_selection_markout",
    ):
        summary[f"mean_{name}"] = float(np.mean([row[name] for row in path_metrics]))
    return summary


def paired_uplift(
    candidate: list[BacktestResult], baseline: list[BacktestResult]
) -> dict[str, float]:
    candidate_by_seed = {result.seed: result.terminal_pnl for result in candidate}
    baseline_by_seed = {result.seed: result.terminal_pnl for result in baseline}
    if candidate_by_seed.keys() != baseline_by_seed.keys():
        raise ValueError("paired comparisons require identical seed sets")
    differences = np.asarray(
        [candidate_by_seed[seed] - baseline_by_seed[seed] for seed in sorted(candidate_by_seed)]
    )
    ci_low, ci_high = confidence_interval(differences)
    return {
        "mean_uplift": float(differences.mean()),
        "uplift_std": float(differences.std(ddof=1)) if len(differences) > 1 else 0.0,
        "uplift_ci_low": ci_low,
        "uplift_ci_high": ci_high,
    }
