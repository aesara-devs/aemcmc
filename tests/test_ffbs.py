import aesara
import aesara.tensor as at
import aesara.tensor.random as atr
import numpy as np
import pytest
import scipy.stats as sp

from aemcmc.ffbs import ffbs_step
from tests.utils import assert_consistent_rng_updates


@pytest.mark.parametrize("first_is_nonzero", [True, False])
def test_ffbs_one_state_likelihood(first_is_nonzero):
    """Test for categorical emissions that assign non-zero likelihood to only one state."""

    Gammas = np.array([[[0.9, 0.1], [0.1, 0.9]]])
    gamma_0 = np.r_[0.5, 0.5]

    N = 10

    if first_is_nonzero:
        log_lik = at.stack([at.broadcast_to(0.0, (N,)), at.broadcast_to(-np.inf, (N,))])
    else:
        log_lik = at.stack([at.broadcast_to(-np.inf, (N,)), at.broadcast_to(0.0, (N,))])

    srng = atr.RandomStream(seed=2032)
    step_at, updates = ffbs_step(gamma_0, Gammas, log_lik, srng)

    assert_consistent_rng_updates(step_at)

    step_fn = aesara.function([], step_at, updates=updates)
    res = step_fn()

    if first_is_nonzero:
        assert np.all(res == 0)
    else:
        assert np.all(res == 1)


def test_ffbs_poisson_states():
    """Test well-separated Poisson emissions."""
    N = 10
    Gammas = np.array([[[0.9, 0.1], [0.1, 0.9]]])
    gamma_0 = np.r_[0.5, 0.5]

    rng = np.random.default_rng(2032)
    seq = rng.choice(2, size=N)
    obs = np.where(
        np.logical_not(seq),
        np.random.poisson(1, N),
        np.random.poisson(50, N),
    )
    log_lik = np.stack(
        [sp.poisson.logpmf(obs, 1), sp.poisson.logpmf(obs, 50)],
    )

    # TODO FIXME: This is a fairly arbitrary check
    assert np.mean(np.abs(log_lik.argmax(0) - seq)) < 1e-2

    srng = atr.RandomStream(seed=2032)
    step_at, updates = ffbs_step(gamma_0, Gammas, log_lik, srng)

    assert_consistent_rng_updates(step_at)

    step_fn = aesara.function([], step_at, updates=updates)
    res = step_fn()

    # TODO FIXME: This is a fairly arbitrary check
    assert np.mean(np.abs(res - seq)) < 1e-2


def test_ffbs_strict_alternating_transitions():
    """Test time-varying transition matrices that specify strictly alternating states--except for the second-to-last one."""
    Gammas = np.stack(
        [
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        ],
        axis=0,
    )

    gamma_0 = np.r_[1.0, 0.0]

    log_lik = np.tile(np.r_[np.log(0.9), np.log(0.1)], (4, 1))
    log_lik[::2] = log_lik[::2][:, ::-1]
    log_lik = log_lik.T

    srng = atr.RandomStream(seed=2032)
    step_at, updates = ffbs_step(gamma_0, Gammas, log_lik, srng)

    assert_consistent_rng_updates(step_at)

    step_fn = aesara.function([], step_at, updates=updates)
    res = step_fn()

    assert np.array_equal(res, np.r_[1, 0, 0, 1])
