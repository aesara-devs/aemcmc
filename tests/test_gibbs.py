import aesara
import aesara.tensor as at
import numpy as np
import pytest
import scipy.special
from aesara.graph.basic import equal_computations
from aesara.graph.rewriting.utils import rewrite_graph
from aesara.tensor.random.utils import RandomStream
from scipy.linalg import toeplitz

from aemcmc.gibbs import (
    bern_normal_posterior,
    bern_sigmoid_dot_match,
    gamma_match,
    horseshoe_match,
    horseshoe_posterior,
    nbinom_dispersion_posterior,
    nbinom_normal_posterior,
    normal_regression_posterior,
    sample_CRT,
)
from aemcmc.rewriting import SamplerTracker, construct_ir_fgraph, sampler_rewrites_db


@pytest.fixture
def srng():
    return RandomStream(1234)


def test_horseshoe_match(srng):
    size = at.lscalar("size")
    # Vector tau
    tau_rv = srng.halfcauchy(0, 1, size=1, name="tau")
    lmbda_rv = srng.halfcauchy(0, 1, size=size, name="lambda")
    beta_rv = srng.normal(0, lmbda_rv * tau_rv, size=size, name="beta")

    fgraph, _, memo, _ = construct_ir_fgraph({beta_rv: beta_rv})
    beta_rv = fgraph.outputs[-1]

    lambda_res, tau_res = horseshoe_match(beta_rv)

    assert lambda_res is memo[lmbda_rv]
    assert tau_res is memo[tau_rv]

    # Scalar tau
    tau_rv = srng.halfcauchy(0, 1, name="tau")
    lmbda_rv = srng.halfcauchy(0, 1, size=size, name="lambda")
    beta_rv = srng.normal(0, lmbda_rv * tau_rv, size=size, name="beta")

    fgraph, _, memo, _ = construct_ir_fgraph({beta_rv: beta_rv})
    beta_rv = fgraph.outputs[-1]

    lambda_res, tau_res = horseshoe_match(beta_rv)

    assert lambda_res is memo[lmbda_rv]
    # `tau_res` should've had its `DimShuffle` lifted, so it's not identical to `tau_rv`
    assert isinstance(tau_res.owner.op, type(tau_rv.owner.op))
    assert tau_res.type.ndim == 0

    beta_rv = srng.normal(0, 1)
    with pytest.raises(ValueError):
        horseshoe_match(beta_rv)

    size = at.scalar("size", dtype="int32")
    tau_rv = srng.halfcauchy(0, 1, size=1)
    lmbda_rv = srng.normal(0, 1, size=size)
    beta_rv = srng.normal(0, tau_rv * lmbda_rv, size=size)
    with pytest.raises(ValueError):
        horseshoe_match(beta_rv)

    size = at.scalar("size", dtype="int32")
    tau_rv = srng.normal(0, 1, size=1)
    lmbda_rv = srng.halfcauchy(0, 1, size=size)
    beta_rv = srng.normal(0, tau_rv * lmbda_rv, size=size)
    with pytest.raises(ValueError):
        horseshoe_match(beta_rv)

    size = at.scalar("size", dtype="int32")
    tau_rv = srng.halfcauchy(0, 1, size=size)
    lmbda_rv = srng.halfcauchy(0, 1, size=size)

    beta_rv = srng.normal(0, tau_rv * lmbda_rv, size=size)
    with pytest.raises(ValueError):
        horseshoe_match(beta_rv)


@pytest.mark.parametrize(
    "N, p, rtol",
    [
        (50, 10, 0.5),
        (50, 75, 0.5),
    ],
)
def test_normal_horseshoe_sampler(srng, N, p, rtol):
    """Check the results of a normal regression model with a Horseshoe prior.

    This test example is modified from section 3.2 of Makalic & Schmidt (2016)

    """
    rng = np.random.default_rng(9420)

    true_beta = np.array([5, 3, 3, 1, 1] + [0] * (p - 5))
    S = toeplitz(0.5 ** np.arange(p))
    X = rng.multivariate_normal(np.zeros(p), cov=S, size=N)
    y = rng.normal(X @ true_beta, np.ones(N))

    tau_rv = srng.halfcauchy(0, 1)
    lambda_rv = srng.halfcauchy(0, 1, size=p)

    tau_vv = tau_rv.clone()
    tau_vv.name = "tau"
    lambda_vv = lambda_rv.clone()
    lambda_vv.name = "lambda"

    beta_post = normal_regression_posterior(
        srng, np.ones(N), at.reciprocal(tau_vv * lambda_vv), at.as_tensor(X), y
    )

    lambda_post, tau_post = horseshoe_posterior(srng, beta_post, 1.0, lambda_vv, tau_vv)

    outputs = (beta_post, lambda_post, tau_post)
    sample_fn = aesara.function((tau_vv, lambda_vv), outputs)

    beta_post_vals = []
    lambda_post_val, tau_post_val = np.ones(p), 1.0
    for i in range(3000):
        beta_post_val, lambda_post_val, tau_post_val = sample_fn(
            tau_post_val, lambda_post_val
        )
        beta_post_vals += [beta_post_val]
        assert np.all(tau_post_val >= 0)
        assert np.all(lambda_post_val >= 0)

    beta_post_median = np.median(beta_post_vals[100::2], axis=0)
    assert np.allclose(beta_post_median[:5], true_beta[:5], atol=1e-1, rtol=rtol)


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


def test_nbinom_normal_posterior(srng):
    M = 10
    N = 50

    true_h = 100
    true_beta = np.array([2, 0.02, 0.2, 0.1, 1] + [0.0] * (M - 5))
    S = toeplitz(0.5 ** np.arange(M))
    X_at = srng.multivariate_normal(np.zeros(M), cov=S, size=N)
    p_at = at.sigmoid(-(X_at.dot(true_beta)))
    X, p, y = aesara.function([], [X_at, p_at, srng.nbinom(true_h, p_at)])()

    beta_vv = at.vector("beta")
    beta_post = nbinom_normal_posterior(
        srng, beta_vv, 200 * np.ones(M), at.as_tensor(X), true_h, y
    )

    beta_post_fn = aesara.function([beta_vv], beta_post)

    beta_post_vals = []
    beta_post_val = np.zeros(M)
    for i in range(1000):
        beta_post_val = beta_post_fn(beta_post_val)
        beta_post_vals += [beta_post_val]

    beta_post_mean = np.mean(beta_post_vals, axis=0)
    assert np.allclose(beta_post_mean, true_beta, atol=1e-1)


def test_nbinom_dispersion_posterior(srng):
    M = 10
    N = 100

    true_h = 100
    true_beta = np.array([2, 0.02, 0.2, 0.1, 1] + [0.1] * (M - 5))
    S = toeplitz(0.5 ** np.arange(M))
    X = srng.multivariate_normal(np.zeros(M), cov=S, size=N)
    p_at = at.sigmoid(-(X.dot(true_beta)))
    p, y = aesara.function([], [p_at, srng.nbinom(true_h, p_at)])()

    a_val = 70.0
    b_val = 0.9
    a = at.as_tensor(a_val)
    b = at.as_tensor(b_val)

    h_samples, h_updates = aesara.scan(
        lambda last_h: nbinom_dispersion_posterior(srng, last_h, p, a, b, y),
        outputs_info=[at.as_tensor(90.0, dtype=np.float64)],
        n_steps=1000,
    )

    h_mean_fn = aesara.function([], h_samples.mean(), updates=h_updates)

    h_mean_val = h_mean_fn()

    # Make sure that the average posterior `h` values have increased relative
    # to the prior mean
    assert h_mean_val > a_val / b_val

    # Make sure the posterior values are near the "true" value
    assert np.allclose(h_mean_val, true_h, rtol=1e-1)


def test_bern_sigmoid_dot_match(srng):
    X = at.matrix("X")

    beta_rv = srng.normal(0, 1, size=X.shape[1], name="beta")
    eta = X @ beta_rv
    p = at.sigmoid(-eta)
    Y_rv = srng.bernoulli(p)

    Y_rv = rewrite_graph(Y_rv)

    assert bern_sigmoid_dot_match(Y_rv)

    beta = at.vector("beta")
    X = at.matrix("X")
    Y = X @ beta

    with pytest.raises(ValueError):
        bern_sigmoid_dot_match(Y)

    X = at.matrix("X")
    beta_rv = srng.normal(0, 1, name="beta")
    eta = X @ beta_rv
    p = at.sigmoid(2 * eta)
    Y_rv = srng.bernoulli(p)

    with pytest.raises(ValueError):
        bern_sigmoid_dot_match(Y_rv)


def test_bern_normal_posterior(srng):
    M = 10
    N = 100

    true_beta = np.array([3, 2, 1, 0.5, 0.05] + [0.0] * (M - 5))
    S = toeplitz(0.5 ** np.arange(M))
    X_at = srng.multivariate_normal(np.zeros(M), cov=S, size=N)
    p_at = at.sigmoid(X_at.dot(true_beta))
    X, p, y = aesara.function([], [X_at, p_at, srng.bernoulli(p_at)])()

    beta_vv = at.vector("beta")
    beta_post = bern_normal_posterior(srng, beta_vv, np.ones(M), at.as_tensor(X), y)

    beta_post_fn = aesara.function([beta_vv], beta_post)

    beta_post_vals = []
    beta_post_val = np.zeros(M)
    for i in range(1000):
        beta_post_val = beta_post_fn(beta_post_val)
        beta_post_vals += [beta_post_val]

    beta_post_mean = np.mean(beta_post_vals, axis=0)
    assert np.allclose(beta_post_mean, true_beta, atol=0.5)


def test_gamma_match(srng):
    beta_rv = srng.normal(0, 1)

    with pytest.raises(ValueError):
        gamma_match(beta_rv)

    a = at.scalar("a")
    b = at.scalar("b")
    beta_rv = srng.gamma(a, b)

    beta_rv = rewrite_graph(beta_rv)

    a_m, b_m = gamma_match(beta_rv)

    assert a_m is a

    b_exp = at.as_tensor(1.0, dtype="floatX") / b
    assert equal_computations([b_m], [b_exp])


def test_nbinom_logistic_horseshoe_finders():
    """Make sure `nbinom_logistic_finder` and `normal_horseshoe_finder` work."""
    srng = RandomStream(0)

    X = at.matrix("X")

    # Horseshoe `beta_rv`
    tau_rv = srng.halfcauchy(0, 1, name="tau")
    lmbda_rv = srng.halfcauchy(0, 1, size=X.shape[1], name="lambda")
    beta_rv = srng.normal(0, lmbda_rv * tau_rv, size=X.shape[1], name="beta")

    a = at.scalar("a")
    b = at.scalar("b")
    h_rv = srng.gamma(a, b, name="h")

    # Negative-binomial regression
    eta = X @ beta_rv
    p = at.sigmoid(-eta)
    Y_rv = srng.nbinom(h_rv, p, name="Y")

    y_vv = Y_rv.clone()
    y_vv.name = "y"

    fgraph, obs_rvs_to_values, memo, new_to_old_rvs = construct_ir_fgraph({Y_rv: y_vv})

    fgraph.attach_feature(SamplerTracker(srng))

    _ = sampler_rewrites_db.query("+basic").rewrite(fgraph)

    discovered_samplers = fgraph.sampler_mappings.rvs_to_samplers
    discovered_samplers = {
        new_to_old_rvs[rv]: discovered_samplers.get(rv)
        for rv in fgraph.outputs
        if rv not in obs_rvs_to_values
    }

    assert len(discovered_samplers) == 4

    assert discovered_samplers[tau_rv][0][0] == "normal_horseshoe_finder"
    assert discovered_samplers[lmbda_rv][0][0] == "normal_horseshoe_finder"
    assert discovered_samplers[beta_rv][0][0] == "nbinom_logistic_finder"
    assert discovered_samplers[h_rv][0][0] == "nbinom_logistic_finder"


def test_bern_logistic_horseshoe_finders():
    """Make sure `bern_logistic_finder` and `normal_horseshoe_finder` work."""
    srng = RandomStream(0)

    X = at.matrix("X")

    # Horseshoe `beta_rv`
    tau_rv = srng.halfcauchy(0, 1, name="tau")
    lmbda_rv = srng.halfcauchy(0, 1, size=X.shape[1], name="lambda")
    beta_rv = srng.normal(0, lmbda_rv * tau_rv, size=X.shape[1], name="beta")

    # Negative-binomial regression
    eta = X @ beta_rv
    p = at.sigmoid(-eta)
    Y_rv = srng.bernoulli(p, name="Y")

    y_vv = Y_rv.clone()
    y_vv.name = "y"

    fgraph, obs_rvs_to_values, memo, new_to_old_rvs = construct_ir_fgraph({Y_rv: y_vv})

    fgraph.attach_feature(SamplerTracker(srng))

    _ = sampler_rewrites_db.query("+basic").rewrite(fgraph)

    discovered_samplers = fgraph.sampler_mappings.rvs_to_samplers
    discovered_samplers = {
        new_to_old_rvs[rv]: discovered_samplers.get(rv)
        for rv in fgraph.outputs
        if rv not in obs_rvs_to_values
    }

    assert len(discovered_samplers) == 3

    assert discovered_samplers[tau_rv][0][0] == "normal_horseshoe_finder"
    assert discovered_samplers[lmbda_rv][0][0] == "normal_horseshoe_finder"
    assert discovered_samplers[beta_rv][0][0] == "bern_logistic_finder"
