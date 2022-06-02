import aesara
import aesara.tensor as at
import numpy as np
import pytest
import scipy.special
from aesara.graph.basic import equal_computations
from aesara.tensor.random.utils import RandomStream
from scipy.linalg import toeplitz

from aemcmc.gibbs import (
    bernoulli_horseshoe_gibbs,
    bernoulli_horseshoe_match,
    bernoulli_horseshoe_model,
    gamma_match,
    h_step,
    horseshoe_match,
    horseshoe_model,
    nbinom_horseshoe_gibbs,
    nbinom_horseshoe_gibbs_with_dispersion,
    nbinom_horseshoe_match,
    nbinom_horseshoe_model,
    nbinom_horseshoe_with_dispersion_match,
    sample_CRT,
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


@pytest.mark.parametrize(
    "h_val, y_val",
    [
        (1.0, 0),
        (1.0, 1),
        (1.0, 20),
        (10.0, 20),
    ],
)
def test_sample_CRT_mean(srng, h_val, y_val):
    y = at.lvector("y")
    h = at.scalar("h")

    crt_res, updates = sample_CRT(srng, y, h)

    crt_fn = aesara.function([y, h], crt_res, updates=updates)

    y_vals = np.repeat(y_val, 5000).astype(np.int64)

    crt_vals = crt_fn(y_vals, h_val)

    crt_mean_val = crt_vals.mean()
    crt_exp_val = h_val * (
        scipy.special.digamma(h_val + y_val) - scipy.special.digamma(h_val)
    )

    assert np.allclose(crt_mean_val, crt_exp_val, rtol=1e-1)


def test_h_step(srng):

    true_h = 10
    M = 10
    N = 50

    true_beta = np.array([2, 0.02, 0.2, 0.1, 1] + [0.1] * (M - 5))
    S = toeplitz(0.5 ** np.arange(M))
    X = srng.multivariate_normal(np.zeros(M), cov=S, size=N)
    p_at = at.sigmoid(-(X.dot(true_beta)))
    p, y = aesara.function([], [p_at, srng.nbinom(true_h, p_at)])()

    a_val = 1.0
    b_val = 2e-1
    a = at.as_tensor(a_val)
    b = at.as_tensor(b_val)

    h_samples, h_updates = aesara.scan(
        lambda: h_step(srng, at.as_tensor(true_h), p, a, b, y), n_steps=1000
    )

    h_mean_fn = aesara.function([], h_samples.mean(), updates=h_updates)

    h_mean_val = h_mean_fn()

    # Make sure that the posterior `h` values aren't right around the prior
    # mean
    assert not np.allclose(h_mean_val, a_val / b_val, rtol=2e-1)

    # Make sure the posterior values are near the "true" value
    assert np.allclose(h_mean_val, true_h, rtol=2e-1)


def test_horseshoe_nbinom_w_dispersion(srng):
    """
    This test example is modified from section 3.2 of Makalic & Schmidt (2016)
    """
    true_h = 10
    M = 10
    N = 50

    # generate synthetic data
    true_beta = np.array([2, 0.02, 0.2, 0.1, 1] + [0.1] * (M - 5))
    S = toeplitz(0.5 ** np.arange(M))
    X_at = srng.multivariate_normal(np.zeros(M), cov=S, size=N)
    X, y = aesara.function(
        [], [X_at, srng.nbinom(true_h, at.sigmoid(-(X_at.dot(true_beta))))]
    )()
    X = at.as_tensor(X)
    y = at.as_tensor(y)

    # build the model
    tau_rv = srng.halfcauchy(0, 1, name="tau")
    lambda_rv = srng.halfcauchy(0, 1, size=M, name="lambda")
    beta_rv = srng.normal(0, tau_rv * lambda_rv, size=M, name="beta")

    eta_tt = X @ beta_rv
    p_tt = at.sigmoid(-eta_tt)
    p_tt.name = "p"

    h_rv = srng.gamma(100, 1, name="h")

    Y_rv = srng.nbinom(h_rv, p_tt, name="Y")

    # sample from the posterior distributions
    num_samples = at.lscalar("num_samples")
    outputs, updates = nbinom_horseshoe_gibbs_with_dispersion(
        srng, Y_rv, y, num_samples
    )

    sample_fn = aesara.function((num_samples,), outputs, updates=updates)

    sample_num = 2000
    beta, lmbda, tau, h = sample_fn(sample_num)

    assert beta.shape == (sample_num, M)
    assert lmbda.shape == (sample_num, M)
    assert tau.shape == (sample_num, 1)
    assert h.shape == (sample_num,)

    assert np.all(tau > 0)
    assert np.all(lmbda > 0)
    assert np.all(h > 0)

    assert np.allclose(h.mean(), true_h, rtol=1e-1)


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


def test_gamma_match(srng):
    beta_rv = srng.normal(0, 1)
    with pytest.raises(ValueError):
        gamma_match(beta_rv)

    a = at.scalar("a")
    b = at.scalar("b")
    beta_rv = srng.gamma(a, b)
    a_m, b_m = gamma_match(beta_rv)

    assert a_m is a

    b_exp = at.as_tensor(1.0, dtype="floatX") / b
    assert equal_computations([b_m], [b_exp])


def test_nbinom_horseshoe_with_dispersion_match(srng):
    a = at.scalar("a")
    b = at.scalar("b")
    X = at.matrix("X")

    beta_rv = horseshoe_model(srng)
    eta = X @ beta_rv
    p = at.sigmoid(-eta)
    h = srng.gamma(a, b)
    Y_rv = srng.nbinom(h, p)

    X_m, beta_m, lmbda_m, tau_m, h_m, a_m, b_m = nbinom_horseshoe_with_dispersion_match(
        Y_rv
    )

    assert a_m is a
    assert X_m is X
