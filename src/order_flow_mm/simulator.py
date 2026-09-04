"""Deterministic pre-generation of common-random-number market paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import STATES, MarketParams

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int8]


def stationary_distribution(transition: FloatArray) -> FloatArray:
    """Compute a stationary row distribution by solving a small linear system."""
    system = np.vstack((transition.T - np.eye(3), np.ones(3)))
    target = np.append(np.zeros(3), 1.0)
    distribution, *_ = np.linalg.lstsq(system, target, rcond=None)
    distribution = np.maximum(distribution, 0.0)
    return distribution / distribution.sum()


@dataclass(frozen=True)
class MarketPath:
    """All exogenous randomness needed to evaluate every policy on one path."""

    regimes: IntArray
    trade_signs: IntArray
    price_shocks: FloatArray
    fill_uniforms: FloatArray
    prices: FloatArray
    seed: int

    def __post_init__(self) -> None:
        specifications = {
            "regimes": np.int8,
            "trade_signs": np.int8,
            "price_shocks": float,
            "fill_uniforms": float,
            "prices": float,
        }
        for name, dtype in specifications.items():
            value = np.array(getattr(self, name), dtype=dtype, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        horizon = len(self.trade_signs)
        event_arrays = (self.regimes, self.price_shocks, self.fill_uniforms)
        if any(len(array) != horizon for array in event_arrays) or len(self.prices) != horizon + 1:
            raise ValueError("market path arrays have inconsistent lengths")
        if not np.all(np.isin(self.regimes, (-1, 0, 1))):
            raise ValueError("regimes must be drawn from {-1, 0, +1}")
        if not np.all(np.isin(self.trade_signs, (-1, 1))):
            raise ValueError("trade signs must be -1 or +1")
        if np.any((self.fill_uniforms < 0.0) | (self.fill_uniforms >= 1.0)):
            raise ValueError("fill uniforms must lie in [0, 1)")

    @property
    def horizon(self) -> int:
        return len(self.trade_signs)


def generate_market_path(params: MarketParams, seed: int) -> MarketPath:
    """Generate latent states, observations, shocks, prices, and fill uniforms."""
    rng = np.random.default_rng(seed)
    horizon = params.horizon
    regime_indices = np.empty(horizon, dtype=np.int8)
    initial = stationary_distribution(params.transition)
    regime_indices[0] = rng.choice(3, p=initial)
    for time in range(1, horizon):
        regime_indices[time] = rng.choice(3, p=params.transition[regime_indices[time - 1]])

    regimes = STATES[regime_indices].astype(np.int8)
    buy_probabilities = 0.5 + params.beta * regimes
    trade_signs = np.where(rng.random(horizon) < buy_probabilities, 1, -1).astype(np.int8)
    price_shocks = rng.standard_normal(horizon)
    fill_uniforms = rng.random(horizon)
    increments = params.mu * regimes + params.sigma * price_shocks
    prices = np.empty(horizon + 1, dtype=float)
    prices[0] = params.initial_price
    prices[1:] = params.initial_price + np.cumsum(increments)

    return MarketPath(
        regimes=regimes,
        trade_signs=trade_signs,
        price_shocks=price_shocks,
        fill_uniforms=fill_uniforms,
        prices=prices,
        seed=seed,
    )
