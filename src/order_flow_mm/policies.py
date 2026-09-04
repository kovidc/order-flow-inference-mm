"""Interpretable market-making policies sharing a common quote interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .config import FilterParams, MarketParams
from .filters import BayesianFilter
from .simulator import MarketPath

IntArray = NDArray[np.int8]


@dataclass(frozen=True)
class Quote:
    bid_distance: float
    ask_distance: float


def closed_form_quote(
    inventory: int,
    ask_cost: float,
    bid_cost: float,
    k_fill: float,
    inventory_penalty: float,
    minimum_distance: float = 0.0,
) -> Quote:
    """Return the constrained myopic optimum for both quote distances."""
    ask = 1.0 / k_fill + ask_cost - inventory_penalty * (2 * inventory - 1)
    bid = 1.0 / k_fill + bid_cost + inventory_penalty * (2 * inventory + 1)
    return Quote(bid_distance=max(minimum_distance, bid), ask_distance=max(minimum_distance, ask))


class Policy(Protocol):
    name: str

    def quote(self, inventory: int, trade_history: IntArray) -> Quote: ...

    def observe(self, trade_sign: int) -> None: ...


class BasePolicy:
    name = "base"

    def observe(self, trade_sign: int) -> None:
        del trade_sign


@dataclass
class FixedPolicy(BasePolicy):
    distance: float
    name: str = "Fixed"

    def quote(self, inventory: int, trade_history: IntArray) -> Quote:
        del inventory, trade_history
        return Quote(self.distance, self.distance)


@dataclass
class InventoryPolicy(BasePolicy):
    market: MarketParams
    name: str = "Inventory"

    def quote(self, inventory: int, trade_history: IntArray) -> Quote:
        del trade_history
        return closed_form_quote(
            inventory,
            0.0,
            0.0,
            self.market.k_fill,
            self.market.inventory_penalty,
            self.market.min_quote_distance,
        )


@dataclass
class RollingImbalancePolicy(BasePolicy):
    market: MarketParams
    window: int
    intercept: float
    slope: float
    name: str = "Rolling"

    def quote(self, inventory: int, trade_history: IntArray) -> Quote:
        recent = trade_history[-self.window :]
        imbalance = float(recent.mean()) if len(recent) else 0.0
        expected_move = self.intercept + self.slope * imbalance
        return closed_form_quote(
            inventory,
            expected_move,
            -expected_move,
            self.market.k_fill,
            self.market.inventory_penalty,
            self.market.min_quote_distance,
        )


class BayesianPolicy(BasePolicy):
    name = "Bayesian"

    def __init__(self, market: MarketParams, beliefs: FilterParams) -> None:
        self.market = market
        self.beliefs = beliefs
        self.filter = BayesianFilter(beliefs)
        self.last_predicted_buy = 0.5

    def quote(self, inventory: int, trade_history: IntArray) -> Quote:
        del trade_history
        self.last_predicted_buy = self.filter.predicted_buy_probability()
        ask_cost = self.beliefs.assumed_mu * self.filter.conditional_state_mean(1)
        bid_cost = -self.beliefs.assumed_mu * self.filter.conditional_state_mean(-1)
        return closed_form_quote(
            inventory,
            ask_cost,
            bid_cost,
            self.market.k_fill,
            self.market.inventory_penalty,
            self.market.min_quote_distance,
        )

    def observe(self, trade_sign: int) -> None:
        self.filter.update(trade_sign)


@dataclass
class OraclePolicy(BasePolicy):
    """Impossible benchmark; only this class accepts the current latent regime."""

    market: MarketParams
    name: str = "Oracle"

    def quote_with_regime(self, inventory: int, regime: int) -> Quote:
        expected_move = self.market.mu * regime
        return closed_form_quote(
            inventory,
            expected_move,
            -expected_move,
            self.market.k_fill,
            self.market.inventory_penalty,
            self.market.min_quote_distance,
        )


def fit_rolling_policy(
    paths: list[MarketPath],
    market: MarketParams,
    windows: tuple[int, ...] = (5, 10, 20, 50, 100),
) -> RollingImbalancePolicy:
    """Select a window and OLS map using training paths only."""
    best: tuple[float, int, float, float] | None = None
    for window in windows:
        predictors: list[float] = []
        targets: list[float] = []
        for path in paths:
            increments = np.diff(path.prices)
            signs = path.trade_signs
            for time in range(window, path.horizon):
                predictors.append(float(signs[time - window : time].mean()))
                targets.append(float(increments[time]))
        if not predictors:
            continue
        design = np.column_stack((np.ones(len(predictors)), predictors))
        coefficients, *_ = np.linalg.lstsq(design, np.asarray(targets), rcond=None)
        residuals = np.asarray(targets) - design @ coefficients
        mse = float(np.mean(residuals**2))
        candidate = (mse, window, float(coefficients[0]), float(coefficients[1]))
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("at least one training path and candidate window are required")
    _, window, intercept, slope = best
    return RollingImbalancePolicy(market, window, intercept, slope)
