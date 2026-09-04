import numpy as np
from scipy.optimize import minimize_scalar

from order_flow_mm.policies import closed_form_quote


def test_closed_form_quote_matches_numerical_objective() -> None:
    k_fill = 1.3
    cost = 0.27
    numerical = minimize_scalar(
        lambda distance: -np.exp(-k_fill * distance) * (distance - cost),
        bounds=(0.0, 5.0),
        method="bounded",
    )
    quote = closed_form_quote(0, cost, cost, k_fill, 0.0)
    assert np.isclose(quote.ask_distance, numerical.x, atol=1e-5)
    assert np.isclose(quote.bid_distance, numerical.x, atol=1e-5)


def test_positive_inventory_makes_ask_more_aggressive_and_bid_less_aggressive() -> None:
    flat = closed_form_quote(0, 0.0, 0.0, 1.0, 0.03)
    long = closed_form_quote(3, 0.0, 0.0, 1.0, 0.03)
    assert long.ask_distance < flat.ask_distance
    assert long.bid_distance > flat.bid_distance
