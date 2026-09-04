from dataclasses import FrozenInstanceError

import pytest

from order_flow_mm.config import FilterParams, MarketParams


def test_truth_and_belief_parameters_are_separate_and_immutable() -> None:
    market = MarketParams()
    beliefs = FilterParams()
    assert market.transition is not beliefs.transition
    assert not market.transition.flags.writeable
    assert not beliefs.transition.flags.writeable
    with pytest.raises(FrozenInstanceError):
        market.beta = 0.1  # type: ignore[misc]
    with pytest.raises(ValueError):
        beliefs.transition[0, 0] = 0.0
