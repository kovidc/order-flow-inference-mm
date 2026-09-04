from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

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
    RollingImbalancePolicy,
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
    market = MarketParams(horizon=150)
    first = [generate_market_path(market, seed) for seed in range(4)]
    second = [replace(path, regimes=-path.regimes) for path in first]
    first_fit = fit_rolling_policy(first, market)
    second_fit = fit_rolling_policy(second, market)
    for name in (
        "window", "intercept", "slope", "side_coefficient", "interaction_coefficient", "validation_mse"
    ):
        assert getattr(first_fit, name) == getattr(second_fit, name)


def test_rolling_receives_only_prefix_and_ignores_current_and_future_signs() -> None:
    market = MarketParams(horizon=30)
    path = generate_market_path(market, seed=63)
    time = 12
    signs = path.trade_signs.copy()
    signs[time:] *= -1
    changed = replace(path, trade_signs=signs)
    seen = []

    class RecordingRolling(RollingImbalancePolicy):
        def quote(self, inventory: int, trade_history: np.ndarray) -> Quote:
            seen.append(trade_history.copy())
            return super().quote(inventory, trade_history)

    policy = RecordingRolling(market, 5, 0.01, 0.10, 0.03, 0.02)
    first = run_backtest(policy, path, market)
    for t, history in enumerate(seen):
        np.testing.assert_array_equal(history, path.trade_signs[:t])
    second = run_backtest(policy, changed, market)
    np.testing.assert_array_equal(first.ask_distances[:time + 1], second.ask_distances[:time + 1])
    np.testing.assert_array_equal(first.bid_distances[:time + 1], second.bid_distances[:time + 1])
    np.testing.assert_array_equal(first.inventory[:time + 1], second.inventory[:time + 1])


def test_rolling_window_validation_and_full_training_refit() -> None:
    market = MarketParams(horizon=150)
    paths = [generate_market_path(market, seed) for seed in range(8)]

    def observations(subset: list[MarketPath], window: int) -> tuple[np.ndarray, np.ndarray]:
        rows, targets = [], []
        for path in subset:
            for t in range(window, path.horizon):
                x = np.mean(path.trade_signs[t - window:t])
                y = path.trade_signs[t]
                rows.append([1, x, y, x * y])
                targets.append(path.prices[t + 1] - path.prices[t])
        return np.array(rows), np.array(targets)

    expected_mse = {}
    for window in (5, 10, 20, 50, 100):
        design, targets = observations(paths[:6], window)
        coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
        validation_design, validation_targets = observations(paths[6:], window)
        expected_mse[window] = np.mean((validation_targets - validation_design @ coefficients)**2)
    selected = min(expected_mse, key=lambda w: (expected_mse[w], w))
    design, targets = observations(paths, selected)
    expected_coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
    policy = fit_rolling_policy(paths, market)
    assert policy.window == selected
    assert policy.validation_mse == pytest.approx(expected_mse)
    np.testing.assert_allclose(
        [policy.intercept, policy.slope, policy.side_coefficient, policy.interaction_coefficient],
        expected_coefficients,
    )


def test_benchmark_fits_and_selects_using_only_supplied_training_paths(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from experiments import benchmark

    market = MarketParams(horizon=150)
    training = [generate_market_path(market, seed) for seed in range(4)]
    evaluation = [generate_market_path(market, seed=99)]
    calls = []

    def record_fit(paths, params):
        assert paths is training
        calls.append(paths)
        return fit_rolling_policy(paths, params)

    monkeypatch.setattr(benchmark, "fit_rolling_policy", record_fit)
    _, first_fit = benchmark.evaluate_main(market, evaluation, training)
    changed = [replace(evaluation[0], trade_signs=-evaluation[0].trade_signs,
                       prices=-evaluation[0].prices)]
    _, second_fit = benchmark.evaluate_main(market, changed, training)
    assert len(calls) == 2
    assert first_fit == second_fit


def test_rolling_rejects_insufficient_training_observations() -> None:
    market = MarketParams(horizon=150)
    path = generate_market_path(market, seed=63)
    with pytest.raises(ValueError, match="at least two training paths"):
        fit_rolling_policy([path], market)
    with pytest.raises(ValueError, match="positive candidate windows"):
        fit_rolling_policy([path, path], market, windows=())
    with pytest.raises(ValueError, match="window 150.*two observations per trade side"):
        fit_rolling_policy([path, path], market, windows=(150,))
    one_side = replace(path, trade_signs=np.ones(150, dtype=np.int8))
    with pytest.raises(ValueError, match="two observations per trade side"):
        fit_rolling_policy([path, one_side], market)
    alternating = replace(path, trade_signs=np.tile([-1, 1], 75))
    with pytest.raises(ValueError, match="rank-deficient"):
        fit_rolling_policy([alternating, alternating], market, windows=(10,))


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
