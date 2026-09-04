"""Validated parameter objects for market truth and a maker's beliefs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
STATES: FloatArray = np.array([-1.0, 0.0, 1.0])
STATES.setflags(write=False)


def symmetric_transition(
    pressure_stay: float = 0.94,
    neutral_stay: float = 0.90,
    direct_switch: float = 0.01,
) -> FloatArray:
    """Return a symmetric transition matrix ordered as ``(-1, 0, +1)``."""
    if not 0.0 <= direct_switch < 1.0:
        raise ValueError("direct_switch must lie in [0, 1)")
    if not direct_switch <= pressure_stay <= 1.0 - direct_switch:
        raise ValueError("pressure_stay leaves an invalid probability for neutral")
    if not 0.0 <= neutral_stay <= 1.0:
        raise ValueError("neutral_stay must lie in [0, 1]")
    leave_neutral = (1.0 - neutral_stay) / 2.0
    transition = np.array(
        [
            [pressure_stay, 1.0 - pressure_stay - direct_switch, direct_switch],
            [leave_neutral, neutral_stay, leave_neutral],
            [direct_switch, 1.0 - pressure_stay - direct_switch, pressure_stay],
        ],
        dtype=float,
    )
    transition.setflags(write=False)
    return transition


def _immutable_transition(value: FloatArray) -> FloatArray:
    transition = np.array(value, dtype=float, copy=True)
    if transition.shape != (3, 3):
        raise ValueError("transition must have shape (3, 3)")
    if np.any(transition < 0.0) or not np.allclose(transition.sum(axis=1), 1.0):
        raise ValueError("transition rows must be nonnegative and sum to one")
    transition.setflags(write=False)
    return transition


@dataclass(frozen=True)
class MarketParams:
    """Parameters of the data-generating process (the market truth)."""

    transition: FloatArray = field(default_factory=symmetric_transition)
    beta: float = 0.20
    mu: float = 0.12
    sigma: float = 0.35
    p0_fill: float = 0.90
    k_fill: float = 1.0
    inventory_penalty: float = 0.02
    initial_price: float = 100.0
    horizon: int = 1_000
    min_quote_distance: float = 0.0
    liquidation_cost: float = 0.10

    def __post_init__(self) -> None:
        object.__setattr__(self, "transition", _immutable_transition(self.transition))
        if not 0.0 <= self.beta <= 0.5:
            raise ValueError("beta must lie in [0, 0.5]")
        if self.sigma < 0.0:
            raise ValueError("sigma must be nonnegative")
        if not 0.0 < self.p0_fill <= 1.0:
            raise ValueError("p0_fill must lie in (0, 1]")
        if self.k_fill <= 0.0:
            raise ValueError("k_fill must be positive")
        if self.inventory_penalty < 0.0 or self.min_quote_distance < 0.0:
            raise ValueError("penalties and quote floor must be nonnegative")
        if self.liquidation_cost < 0.0:
            raise ValueError("liquidation_cost must be nonnegative")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")


@dataclass(frozen=True)
class FilterParams:
    """A maker's assumed transition, emission, and economic mapping."""

    transition: FloatArray = field(default_factory=symmetric_transition)
    beta: float = 0.20
    assumed_mu: float = 0.12

    def __post_init__(self) -> None:
        object.__setattr__(self, "transition", _immutable_transition(self.transition))
        if not 0.0 <= self.beta <= 0.5:
            raise ValueError("beta must lie in [0, 0.5]")


TRAINING_SEEDS: tuple[int, ...] = tuple(range(1_000, 1_040))
EVALUATION_SEEDS: tuple[int, ...] = tuple(range(10_000, 10_300))
