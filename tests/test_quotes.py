import numpy as np
from scipy.optimize import minimize_scalar

from order_flow_mm.config import MarketParams
from order_flow_mm.policies import RollingImbalancePolicy, closed_form_quote


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


def test_rolling_quotes_evaluate_both_hypothetical_sides() -> None:
    market = MarketParams()
    policy = RollingImbalancePolicy(market, 4, 0.01, 0.10, 0.03, 0.02)
    history = np.array([-1, 1, 1, 1], dtype=np.int8)
    # x=0.5 gives BUY prediction 0.10 and SELL prediction 0.02.
    quote = policy.quote(0, history)
    assert np.isclose(quote.ask_distance, 1.12)
    assert np.isclose(quote.bid_distance, 1.00)
    assert policy.quote(0, np.r_[[-1, -1], history]) == quote
    assert policy.quote(0, -history) != quote
    assert policy.quote(2, history) != quote
    empty_quote = policy.quote(0, np.array([], dtype=np.int8))
    assert np.isclose(empty_quote.ask_distance, 1.06)
    assert np.isclose(empty_quote.bid_distance, 1.04)
