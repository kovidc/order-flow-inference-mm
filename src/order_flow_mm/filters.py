"""A direct NumPy implementation of the three-state Bayesian filter."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import STATES, FilterParams
from .simulator import stationary_distribution

FloatArray = NDArray[np.float64]


class BayesianFilter:
    """Filter latent regimes using observed public trade signs only."""

    def __init__(self, params: FilterParams, initial: FloatArray | None = None) -> None:
        self.params = params
        posterior = stationary_distribution(params.transition) if initial is None else initial
        self.posterior = self._normalize(np.asarray(posterior, dtype=float))

    @staticmethod
    def _normalize(weights: FloatArray) -> FloatArray:
        total = float(weights.sum())
        if total <= 0.0 or not np.isfinite(total):
            raise ValueError("posterior weights must have a positive finite sum")
        return weights / total

    def emission(self, sign: int) -> FloatArray:
        """Return P(Y=sign | Z) for states ordered (-1, 0, +1)."""
        if sign not in (-1, 1):
            raise ValueError("sign must be -1 or +1")
        buy_probability = 0.5 + self.params.beta * STATES
        return buy_probability if sign == 1 else 1.0 - buy_probability

    def predict(self) -> FloatArray:
        """Return the prior for the next latent state without mutating the filter."""
        return self._normalize(self.posterior @ self.params.transition)

    def posterior_if(self, sign: int) -> FloatArray:
        """Return the hypothetical next posterior conditional on a trade sign."""
        return self._normalize(self.predict() * self.emission(sign))

    def predicted_buy_probability(self) -> float:
        """Return P(next trade is BUY | observed trade history)."""
        return float(self.predict() @ self.emission(1))

    def conditional_state_mean(self, sign: int) -> float:
        return float(self.posterior_if(sign) @ STATES)

    def update(self, actual_sign: int) -> FloatArray:
        """Assimilate one observed sign and return the new posterior."""
        self.posterior = self.posterior_if(actual_sign)
        return self.posterior.copy()
