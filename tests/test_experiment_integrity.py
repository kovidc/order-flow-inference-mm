from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from order_flow_mm.backtest import execution_probability, run_backtest
from order_flow_mm.config import FilterParams, MarketParams, symmetric_transition
from order_flow_mm.metrics import confidence_interval, lower_tail_cvar
from order_flow_mm.policies import (
    BasePolicy,
    BayesianPolicy,
    FixedPolicy,
    InventoryPolicy,
    Quote,
    fit_rolling_policy,
)
from order_flow_mm.simulator import MarketPath, generate_market_path


@dataclass
class HistorySpyPolicy(BasePolicy):
    seen_histories: list[np.ndarray] = field(default_factory=list)
    name: str = "HistorySpy"

    def quote(self, inventory: int, trade_history: np.ndarray) -> Quote:
        del inventory
        self.seen_histories.append(trade_history.copy())
        return Quote(1.0, 1.0)


def test_policy_receives_only_strictly_prior_trade_signs() -> None:
    market = MarketParams(horizon=3)
    path = MarketPath(
        regimes=np.array([1, -1, 0]),
        trade_signs=np.array([1, -1, 1]),
        price_shocks=np.zeros(3),
        fill_uniforms=np.full(3, 0.99),
        prices=np.full(4, 100.0),
        seed=1,
    )
    policy = HistorySpyPolicy()
    run_backtest(policy, path, market)
    assert [history.tolist() for history in policy.seen_histories] == [[], [1], [1, -1]]


@pytest.mark.parametrize(
    ("market", "beliefs"),
    [
        (MarketParams(beta=0.0, horizon=200), FilterParams(beta=0.0)),
        (MarketParams(mu=0.0, horizon=200), FilterParams(assumed_mu=0.0)),
    ],
)
def test_null_bayesian_and_inventory_policies_are_pathwise_identical(
    market: MarketParams, beliefs: FilterParams
) -> None:
    path = generate_market_path(market, seed=20_123)
    inventory = run_backtest(InventoryPolicy(market), path, market)
    bayesian = run_backtest(BayesianPolicy(market, beliefs), path, market)
    for name in ("bid_distances", "ask_distances", "fills", "cash", "inventory", "wealth"):
        np.testing.assert_array_equal(getattr(bayesian, name), getattr(inventory, name))


def test_bayesian_quotes_depend_on_beliefs_not_truth_signal_parameters() -> None:
    beliefs = FilterParams(symmetric_transition(0.80), beta=0.10, assumed_mu=0.07)
    first = BayesianPolicy(
        MarketParams(transition=symmetric_transition(0.70), beta=0.0, mu=-2.0), beliefs
    )
    second = BayesianPolicy(
        MarketParams(transition=symmetric_transition(0.99), beta=0.49, mu=3.0), beliefs
    )
    history = np.array([1, 1, -1, 1], dtype=np.int8)
    for sign in history:
        np.testing.assert_equal(first.quote(3, history), second.quote(3, history))
        first.observe(int(sign))
        second.observe(int(sign))


def test_rolling_fit_cannot_depend_on_latent_regime_labels() -> None:
    market = MarketParams(horizon=6)
    common = {
        "trade_signs": np.array([1, 1, -1, -1, 1, -1]),
        "price_shocks": np.zeros(6),
        "fill_uniforms": np.full(6, 0.5),
        "prices": np.array([100.0, 100.2, 100.1, 99.9, 100.0, 100.3, 100.1]),
        "seed": 5,
    }
    first = MarketPath(regimes=np.array([-1, -1, 0, 0, 1, 1]), **common)
    second = MarketPath(regimes=np.array([1, 0, -1, 1, 0, -1]), **common)
    first_fit = fit_rolling_policy([first], market, windows=(2,))
    second_fit = fit_rolling_policy([second], market, windows=(2,))
    assert (first_fit.window, first_fit.intercept, first_fit.slope) == pytest.approx(
        (second_fit.window, second_fit.intercept, second_fit.slope)
    )


def test_common_uniforms_create_monotone_fill_coupling() -> None:
    market = MarketParams(horizon=1_000)
    path = generate_market_path(market, seed=41_001)
    aggressive = run_backtest(FixedPolicy(0.5), path, market)
    passive = run_backtest(FixedPolicy(1.5), path, market)
    assert execution_probability(0.5, market) > execution_probability(1.5, market)
    assert np.all(~passive.fills | aggressive.fills)


def test_accounting_decomposes_into_carry_spread_markout_and_liquidation() -> None:
    market = MarketParams(horizon=100, liquidation_cost=0.17)
    path = generate_market_path(market, seed=42_002)
    result = run_backtest(InventoryPolicy(market), path, market)
    active_distance = np.where(
        path.trade_signs == 1, result.ask_distances, result.bid_distances
    )
    spread = float(active_distance[result.fills].sum())
    adverse_loss = float(np.nansum(result.adverse_markouts))
    inventory_carry = float(np.sum(result.inventory[:-1] * np.diff(path.prices)))
    liquidation = market.liquidation_cost * abs(result.inventory[-1])
    assert result.terminal_pnl == pytest.approx(
        spread - adverse_loss + inventory_carry - liquidation
    )


def test_lower_tail_cvar_and_normal_confidence_interval_sign_conventions() -> None:
    pnl = np.array([-10.0, -4.0, 1.0, 5.0])
    assert lower_tail_cvar(pnl, alpha=0.5) == -7.0
    low, high = confidence_interval(np.array([1.0, 2.0, 3.0]))
    assert low < 2.0 < high
    assert 2.0 - low == pytest.approx(high - 2.0)
