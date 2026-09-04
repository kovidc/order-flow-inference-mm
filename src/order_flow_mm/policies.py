"""Interpretable market-making policies sharing a common quote interface."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    side_coefficient: float
    interaction_coefficient: float
    name: str = "Rolling"
    validation_mse: dict[int, float] = field(default_factory=dict)

    def quote(self, inventory: int, trade_history: IntArray) -> Quote:
        recent = trade_history[-self.window :]
        imbalance = float(recent.mean()) if len(recent) else 0.0
        common_move = self.intercept + self.slope * imbalance
        side_move = self.side_coefficient + self.interaction_coefficient * imbalance
        return closed_form_quote(
            inventory,
            common_move + side_move,
            -(common_move - side_move),
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
    """Full-information benchmark using the current latent regime."""

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


def _rolling_observations(
    paths: list[MarketPath], window: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build [1, trailing imbalance, current sign, interaction] training rows."""
    rows = []
    targets = []
    for path in paths:
        increments = np.diff(path.prices)
        for time in range(window, path.horizon):
            imbalance = float(path.trade_signs[time - window : time].mean())
            sign = int(path.trade_signs[time])
            rows.append((1.0, imbalance, sign, imbalance * sign))
            targets.append(increments[time])
    design = np.asarray(rows, dtype=float).reshape(-1, 4)
    if any(np.count_nonzero(design[:, 2] == sign) < 2 for sign in (-1, 1)):
        raise ValueError(f"window {window} requires at least two observations per trade side")
    return design, np.asarray(targets)


def _fit_rolling_coefficients(
    design: NDArray[np.float64], targets: NDArray[np.float64], window: int
) -> NDArray[np.float64]:
    coefficients, _, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
    if rank < 4:
        raise ValueError(f"window {window} has a rank-deficient conditional OLS design")
    return coefficients


def fit_rolling_policy(
    paths: list[MarketPath],
    market: MarketParams,
    windows: tuple[int, ...] = (5, 10, 20, 50, 100),
) -> RollingImbalancePolicy:
    """Select W on a 75/25 ordered training-path split, then refit on all paths."""
    if len(paths) < 2 or not windows or any(window <= 0 for window in windows):
        raise ValueError("at least two training paths and positive candidate windows are required")
    split = 3 * len(paths) // 4
    validation_mse = {}
    for window in windows:
        design, targets = _rolling_observations(paths[:split], window)
        coefficients = _fit_rolling_coefficients(design, targets, window)
        validation_design, validation_targets = _rolling_observations(paths[split:], window)
        residuals = validation_targets - validation_design @ coefficients
        validation_mse[window] = float(np.mean(residuals**2))
    window = min(validation_mse, key=lambda candidate: (validation_mse[candidate], candidate))
    design, targets = _rolling_observations(paths, window)
    coefficients = _fit_rolling_coefficients(design, targets, window)
    return RollingImbalancePolicy(
        market, window, *map(float, coefficients), validation_mse=validation_mse
    )
