import numpy as np

from order_flow_mm.backtest import run_backtest
from order_flow_mm.config import FilterParams, MarketParams
from order_flow_mm.metrics import inference_metrics, lower_tail_cvar, paired_uplift
from order_flow_mm.policies import BayesianPolicy, InventoryPolicy
from order_flow_mm.simulator import generate_market_path


def test_inference_metrics_are_finite_and_bounded() -> None:
    market = MarketParams(horizon=100)
    path = generate_market_path(market, 9)
    result = run_backtest(BayesianPolicy(market, FilterParams()), path, market)
    metrics = inference_metrics(result, path)
    assert 0.0 <= metrics.brier_score <= 1.0
    assert metrics.log_loss >= 0.0
    assert 0.0 <= metrics.map_accuracy <= 1.0


def test_paired_uplift_and_lower_tail_cvar() -> None:
    market = MarketParams(horizon=25)
    paths = [generate_market_path(market, seed) for seed in (1, 2, 3)]
    baseline = [run_backtest(InventoryPolicy(market), path, market) for path in paths]
    candidate = [run_backtest(InventoryPolicy(market), path, market) for path in paths]
    assert paired_uplift(candidate, baseline)["mean_uplift"] == 0.0
    assert lower_tail_cvar(np.array([-3.0, -1.0, 2.0, 4.0]), alpha=0.5) == -2.0
