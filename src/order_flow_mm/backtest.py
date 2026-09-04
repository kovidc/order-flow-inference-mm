"""One execution and accounting engine shared by every policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import MarketParams
from .policies import BayesianPolicy, OraclePolicy, Policy, Quote
from .simulator import MarketPath

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class BacktestResult:
    policy_name: str
    seed: int
    terminal_pnl: float
    cash: FloatArray
    inventory: IntArray
    wealth: FloatArray
    bid_distances: FloatArray
    ask_distances: FloatArray
    fills: BoolArray
    adverse_markouts: FloatArray
    predicted_buy: FloatArray | None = None
    posteriors: FloatArray | None = None


def execution_probability(distance: float, params: MarketParams) -> float:
    """Reduced-form fill probability at a nonnegative quote distance."""
    return float(params.p0_fill * np.exp(-params.k_fill * distance))


def run_backtest(
    policy: Policy | OraclePolicy,
    path: MarketPath,
    params: MarketParams,
) -> BacktestResult:
    """Evaluate one policy without allowing it to influence the market path."""
    if path.horizon != params.horizon:
        raise ValueError("market path and parameter horizons differ")

    horizon = path.horizon
    cash = np.zeros(horizon + 1)
    inventory = np.zeros(horizon + 1, dtype=np.int64)
    wealth = np.zeros(horizon + 1)
    bid_distances = np.empty(horizon)
    ask_distances = np.empty(horizon)
    fills = np.zeros(horizon, dtype=bool)
    adverse_markouts = np.full(horizon, np.nan)
    is_bayesian = isinstance(policy, BayesianPolicy)
    predicted_buy = np.empty(horizon) if is_bayesian else None
    posteriors = np.empty((horizon, 3)) if is_bayesian else None

    for time in range(horizon):
        current_inventory = int(inventory[time])
        if isinstance(policy, OraclePolicy):
            quote = policy.quote_with_regime(current_inventory, int(path.regimes[time]))
        else:
            quote = policy.quote(current_inventory, path.trade_signs[:time])
        _validate_quote(quote)
        bid_distances[time] = quote.bid_distance
        ask_distances[time] = quote.ask_distance
        if predicted_buy is not None:
            predicted_buy[time] = policy.last_predicted_buy

        cash[time + 1] = cash[time]
        inventory[time + 1] = current_inventory
        sign = int(path.trade_signs[time])
        distance = quote.ask_distance if sign == 1 else quote.bid_distance
        if path.fill_uniforms[time] < execution_probability(distance, params):
            fills[time] = True
            if sign == 1:
                cash[time + 1] += path.prices[time] + quote.ask_distance
                inventory[time + 1] -= 1
            else:
                cash[time + 1] -= path.prices[time] - quote.bid_distance
                inventory[time + 1] += 1
            adverse_markouts[time] = sign * (path.prices[time + 1] - path.prices[time])

        policy.observe(sign)
        if posteriors is not None:
            posteriors[time] = policy.filter.posterior
        wealth[time + 1] = cash[time + 1] + inventory[time + 1] * path.prices[time + 1]

    wealth[-1] -= params.liquidation_cost * abs(inventory[-1])
    return BacktestResult(
        policy_name=policy.name,
        seed=path.seed,
        terminal_pnl=float(wealth[-1]),
        cash=cash,
        inventory=inventory,
        wealth=wealth,
        bid_distances=bid_distances,
        ask_distances=ask_distances,
        fills=fills,
        adverse_markouts=adverse_markouts,
        predicted_buy=predicted_buy,
        posteriors=posteriors,
    )


def _validate_quote(quote: Quote) -> None:
    distances = (quote.bid_distance, quote.ask_distance)
    if any(not np.isfinite(distance) or distance < 0.0 for distance in distances):
        raise ValueError("quote distances must be finite and nonnegative")
