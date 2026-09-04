import numpy as np

from order_flow_mm.config import FilterParams, symmetric_transition
from order_flow_mm.filters import BayesianFilter


def test_posterior_sums_to_one_after_many_updates() -> None:
    hmm = BayesianFilter(FilterParams())
    for sign in [1, 1, -1, 1, -1, -1] * 20:
        posterior = hmm.update(sign)
        assert np.isclose(posterior.sum(), 1.0)
        assert np.all(posterior >= 0.0)


def test_beta_zero_observation_adds_no_information() -> None:
    hmm = BayesianFilter(FilterParams(beta=0.0), initial=np.array([0.7, 0.2, 0.1]))
    prediction = hmm.predict()
    np.testing.assert_allclose(hmm.posterior_if(1), prediction)
    np.testing.assert_allclose(hmm.posterior_if(-1), prediction)


def test_symmetric_model_has_balanced_prediction_and_mirrored_posteriors() -> None:
    params = FilterParams(transition=symmetric_transition(), beta=0.2)
    hmm = BayesianFilter(params, initial=np.array([0.25, 0.5, 0.25]))
    assert np.isclose(hmm.predicted_buy_probability(), 0.5)
    buy = hmm.posterior_if(1)
    sell = hmm.posterior_if(-1)
    np.testing.assert_allclose(buy, sell[::-1])
