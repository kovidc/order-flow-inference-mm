import inspect

import numpy as np

from order_flow_mm.backtest import run_backtest
from order_flow_mm.config import MarketParams
from order_flow_mm.policies import (
    BayesianPolicy,
    FixedPolicy,
    InventoryPolicy,
    RollingImbalancePolicy,
)
from order_flow_mm.simulator import MarketPath, generate_market_path


def test_market_path_is_deterministic_and_immutable() -> None:
    params = MarketParams(horizon=50)
    first = generate_market_path(params, seed=17)
    second = generate_market_path(params, seed=17)
    for field in ("regimes", "trade_signs", "price_shocks", "fill_uniforms", "prices"):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
        assert not getattr(first, field).flags.writeable


def test_non_oracle_quote_interfaces_cannot_receive_regime() -> None:
    for policy_type in (FixedPolicy, InventoryPolicy, RollingImbalancePolicy, BayesianPolicy):
        parameters = inspect.signature(policy_type.quote).parameters
        assert "regime" not in parameters


def test_backtest_is_deterministic_under_fixed_seed() -> None:
    params = MarketParams(horizon=100)
    path = generate_market_path(params, seed=44)
    first = run_backtest(InventoryPolicy(params), path, params)
    second = run_backtest(InventoryPolicy(params), path, params)
    np.testing.assert_array_equal(first.wealth, second.wealth)
    np.testing.assert_array_equal(first.fills, second.fills)


def test_cash_inventory_and_markout_accounting() -> None:
    arrays = {
        "regimes": np.array([1, -1], dtype=np.int8),
        "trade_signs": np.array([1, -1], dtype=np.int8),
        "price_shocks": np.zeros(2),
        "fill_uniforms": np.zeros(2),
        "prices": np.array([100.0, 101.0, 100.0]),
    }
    path = MarketPath(**arrays, seed=1)
    params = MarketParams(horizon=2, p0_fill=1.0, liquidation_cost=0.0)
    result = run_backtest(FixedPolicy(0.0), path, params)
    np.testing.assert_array_equal(result.inventory, [0, -1, 0])
    np.testing.assert_allclose(result.cash, [0.0, 100.0, -1.0])
    np.testing.assert_allclose(result.adverse_markouts, [1.0, 1.0])
    assert result.terminal_pnl == -1.0
