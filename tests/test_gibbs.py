import aesara
import aesara.tensor as at
import numpy as np
import pytest
from aesara.tensor.random.utils import RandomStream
from scipy.linalg import toeplitz

from aemcmc.gibbs import (
    bernoulli_horseshoe_gibbs,
    bernoulli_horseshoe_match,
    bernoulli_horseshoe_model,
    horseshoe_match,
    horseshoe_model,
    nbinom_horseshoe_gibbs,
    nbinom_horseshoe_match,
    nbinom_horseshoe_model,
)


@pytest.fixture
def srng():
    return RandomStream(1234)


def test_horseshoe_match(srng):

    size = at.lscalar("size")
    # Vector tau
    tau_rv = srng.halfcauchy(0, 1, size=1, name="tau")
    lmbda_rv = srng.halfcauchy(0, 1, size=size, name="lambda")
    beta_rv = srng.normal(0, lmbda_rv * tau_rv, size=size, name="beta")

    lambda_res, tau_res = horseshoe_match(beta_rv)

    assert lambda_res is lmbda_rv
    assert tau_res is tau_rv

    # Scalar tau
    tau_rv = srng.halfcauchy(0, 1, name="tau")
    lmbda_rv = srng.halfcauchy(0, 1, size=size, name="lambda")
    beta_rv = srng.normal(0, lmbda_rv * tau_rv, size=size, name="beta")

    lambda_res, tau_res = horseshoe_match(beta_rv)

    assert lambda_res is lmbda_rv
    # `tau_res` should've had its `DimShuffle` lifted, so it's not identical to `tau_rv`
    assert isinstance(tau_res.owner.op, type(tau_rv.owner.op))
    assert tau_res.type.ndim == 1


def test_horseshoe_match_wrong_graph(srng):
    beta_rv = srng.normal(0, 1)
    with pytest.raises(ValueError):
        horseshoe_match(beta_rv)


def test_horseshoe_match_wrong_local_scale_dist(srng):
    size = at.scalar("size", dtype="int32")
    tau_rv = srng.halfcauchy(0, 1, size=1)
    lmbda_rv = srng.normal(0, 1, size=size)
    beta_rv = srng.normal(0, tau_rv * lmbda_rv, size=size)
    with pytest.raises(ValueError):
        horseshoe_match(beta_rv)


def test_horseshoe_match_wrong_global_scale_dist(srng):
    size = at.scalar("size", dtype="int32")
    tau_rv = srng.normal(0, 1, size=1)
    lmbda_rv = srng.halfcauchy(0, 1, size=size)
    beta_rv = srng.normal(0, tau_rv * lmbda_rv, size=size)
    with pytest.raises(ValueError):
        horseshoe_match(beta_rv)


def test_horseshoe_match_wrong_dimensions(srng):
    size = at.scalar("size", dtype="int32")
    tau_rv = srng.halfcauchy(0, 1, size=size)
    lmbda_rv = srng.halfcauchy(0, 1, size=size)

    beta_rv = srng.normal(0, tau_rv * lmbda_rv, size=size)
    with pytest.raises(ValueError):
        horseshoe_match(beta_rv)


def test_match_nbinom_horseshoe(srng):
    nbinom_horseshoe_match(nbinom_horseshoe_model(srng))


def test_match_binom_horseshoe_wrong_graph(srng):
    beta = at.vector("beta")
    X = at.matrix("X")
    Y = X @ beta

    with pytest.raises(ValueError):
        nbinom_horseshoe_match(Y)


def test_match_nbinom_horseshoe_wrong_sign(srng):
    X = at.matrix("X")
    h = at.scalar("h")

    beta_rv = horseshoe_model(srng)
    eta = X @ beta_rv
    p = at.sigmoid(2 * eta)
    Y_rv = srng.nbinom(h, p)

    with pytest.raises(ValueError):
        nbinom_horseshoe_match(Y_rv)


def test_horseshoe_nbinom(srng):
    """
    This test example is modified from section 3.2 of Makalic & Schmidt (2016)
    """
    h = 2
    p = 10
    N = 50

    # generate synthetic data
    true_beta = np.array([5, 3, 3, 1, 1] + [0] * (p - 5))
    S = toeplitz(0.5 ** np.arange(p))
    X = srng.multivariate_normal(np.zeros(p), cov=S, size=N)
    y = srng.nbinom(h, at.sigmoid(-(X.dot(true_beta))))

    # build the model
    tau_rv = srng.halfcauchy(0, 1, size=1)
    lambda_rv = srng.halfcauchy(0, 1, size=p)
    beta_rv = srng.normal(0, tau_rv * lambda_rv, size=p)

    eta_tt = X @ beta_rv
    p_tt = at.sigmoid(-eta_tt)
    Y_rv = srng.nbinom(h, p_tt)

    # sample from the posterior distributions
    num_samples = at.scalar("num_samples", dtype="int32")
    outputs, updates = nbinom_horseshoe_gibbs(srng, Y_rv, y, num_samples)
    sample_fn = aesara.function((num_samples,), outputs, updates=updates)

    beta, lmbda, tau = sample_fn(2000)

    assert beta.shape == (2000, p)
    assert lmbda.shape == (2000, p)
    assert tau.shape == (2000, 1)

    # test distribution domains
    assert np.all(tau > 0)
    assert np.all(lmbda > 0)


def test_match_bernoulli_horseshoe(srng):
    bernoulli_horseshoe_match(bernoulli_horseshoe_model(srng))


def test_match_bernoulli_horseshoe_wrong_graph(srng):
    beta = at.vector("beta")
    X = at.matrix("X")
    Y = X @ beta

    with pytest.raises(ValueError):
        bernoulli_horseshoe_match(Y)


def test_match_bernoulli_horseshoe_wrong_sign(srng):
    X = at.matrix("X")

    beta_rv = horseshoe_model(srng)
    eta = X @ beta_rv
    p = at.sigmoid(2 * eta)
    Y_rv = srng.bernoulli(p)

    with pytest.raises(ValueError):
        bernoulli_horseshoe_match(Y_rv)


def test_bernoulli_horseshoe(srng):
    p = 10
    N = 50

    # generate synthetic data
    true_beta = np.array([5, 3, 3, 1, 1] + [0] * (p - 5))
    S = toeplitz(0.5 ** np.arange(p))
    X = srng.multivariate_normal(np.zeros(p), cov=S, size=N)
    y = srng.bernoulli(at.sigmoid(-X.dot(true_beta)))

    # build the model
    tau_rv = srng.halfcauchy(0, 1, size=1)
    lambda_rv = srng.halfcauchy(0, 1, size=p)
    beta_rv = srng.normal(0, tau_rv * lambda_rv, size=p)

    eta_tt = X @ beta_rv
    p_tt = at.sigmoid(-eta_tt)
    Y_rv = srng.bernoulli(p_tt)

    # sample from the posterior distributions
    num_samples = at.scalar("num_samples", dtype="int32")
    outputs, updates = bernoulli_horseshoe_gibbs(srng, Y_rv, y, num_samples)
    sample_fn = aesara.function((num_samples,), outputs, updates=updates)

    beta, lmbda, tau = sample_fn(2000)

    assert beta.shape == (2000, p)
    assert lmbda.shape == (2000, p)
    assert tau.shape == (2000, 1)

    # test distribution domains
    assert np.all(tau > 0)
    assert np.all(lmbda > 0)
