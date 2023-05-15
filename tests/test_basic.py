import aesara
import aesara.tensor as at
import numpy as np
import pytest
from aesara.graph.basic import graph_inputs, io_toposort
from aesara.tensor.random import RandomStream
from aesara.tensor.random.basic import BetaRV, GammaRV
from scipy.linalg import toeplitz

from aemcmc.basic import construct_sampler
from aemcmc.gibbs import (
    DispersionGibbsKernel,
    HorseshoeGibbsKernel,
    NBRegressionGibbsKernel,
)
from aemcmc.rewriting import SubsumingElemwise
from tests.utils import assert_consistent_rng_updates


def test_closed_form_posterior_beta_binomial():
    srng = RandomStream(0)

    alpha_tt = at.scalar("alpha")
    beta_tt = at.scalar("beta")
    p_rv = srng.beta(alpha_tt, beta_tt, name="p")

    n_tt = at.iscalar("n")
    Y_rv = srng.binomial(n_tt, p_rv, name="Y")

    y_vv = Y_rv.clone()
    y_vv.name = "y"

    sampler, initial_values = construct_sampler({Y_rv: y_vv}, srng)

    p_posterior_step = sampler.sample_steps[p_rv]
    assert len(sampler.parameters) == 0
    assert len(sampler.stages) == 1
    assert isinstance(p_posterior_step.owner.op, BetaRV)
    assert_consistent_rng_updates(p_posterior_step)


def test_closed_form_posterior_gamma_poisson():
    srng = RandomStream(0)

    alpha_tt = at.scalar("alpha")
    beta_tt = at.scalar("beta")
    l_rv = srng.gamma(alpha_tt, beta_tt, name="p")

    Y_rv = srng.poisson(l_rv, name="Y")

    y_vv = Y_rv.clone()
    y_vv.name = "y"

    sampler, initial_values = construct_sampler({Y_rv: y_vv}, srng)

    p_posterior_step = sampler.sample_steps[l_rv]
    assert len(sampler.parameters) == 0
    assert len(sampler.stages) == 1
    assert isinstance(p_posterior_step.owner.op, GammaRV)
    assert_consistent_rng_updates(p_posterior_step)


def test_closed_form_posterior_beta_nbinom():
    srng = RandomStream(0)

    alpha_tt = at.scalar("alpha")
    beta_tt = at.scalar("beta")

    p_rv = srng.beta(alpha_tt, beta_tt, name="p")

    n_tt = at.scalar("n")
    Y_rv = srng.negative_binomial(n_tt, p_rv, name="Y")

    y_vv = Y_rv.clone()
    y_vv.name = "y"

    sampler, initial_values = construct_sampler({Y_rv: y_vv}, srng)

    p_posterior_step = sampler.sample_steps[p_rv]
    assert len(sampler.parameters) == 0
    assert len(sampler.stages) == 1
    assert isinstance(p_posterior_step.owner.op, BetaRV)
    assert_consistent_rng_updates(p_posterior_step)


@pytest.mark.parametrize("size", [1, (1,), (2, 3)])
def test_nuts_sampler_single_variable(size):
    """We make sure that the NUTS sampler compiles and updates the chains for
    different sizes of the random variable.

    """
    srng = RandomStream(0)

    tau_rv = srng.halfcauchy(0, 1, size=size, name="tau")
    Y_rv = srng.halfcauchy(0, tau_rv, name="Y")

    y_vv = Y_rv.clone()
    y_vv.name = "y"

    sampler, initial_values = construct_sampler({Y_rv: y_vv}, srng)
    assert len(sampler.sample_steps) == 1
    assert len(sampler.stages) == 1

    tau_post_step = sampler.sample_steps[tau_rv]
    assert_consistent_rng_updates(tau_post_step)
    nuts = tau_post_step.owner.op
    assert y_vv in graph_inputs([tau_post_step])
    assert len(sampler.parameters[nuts]) == 2

    inputs = [
        initial_values[tau_rv],
        y_vv,
        sampler.parameters[nuts][0],
        sampler.parameters[nuts][1],
    ]
    output = tau_post_step
    sample_step = aesara.function(inputs, output)

    tau_val = np.ones(shape=size)
    y_val = np.ones(shape=size)
    step_size = 1e-1
    inverse_mass_matrix = np.ones(shape=size).flatten()
    res = sample_step(tau_val, y_val, step_size, inverse_mass_matrix)
    with pytest.raises(AssertionError):
        np.testing.assert_equal(tau_val, res)


def test_nuts_with_closed_form():
    """Make sure that the NUTS sampler works in combination with closed-form posteriors."""
    srng = RandomStream(0)

    alpha_tt = at.scalar("alpha")
    beta_rv = srng.halfnormal(1.0, name="beta")
    l_rv = srng.gamma(alpha_tt, beta_rv, name="p")

    Y_rv = srng.poisson(l_rv, name="Y")

    y_vv = Y_rv.clone()
    y_vv.name = "y"

    sampler, initial_values = construct_sampler({Y_rv: y_vv}, srng)

    assert len(sampler.stages) == 2

    l_posterior_step = sampler.sample_steps[l_rv]
    assert_consistent_rng_updates(l_posterior_step)
    assert y_vv in graph_inputs([l_posterior_step])
    assert len(initial_values) == 2
    assert isinstance(l_posterior_step.owner.op, GammaRV)

    assert beta_rv in sampler.sample_steps


def test_create_gibbs():
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

    sample_vars = [tau_rv, lmbda_rv, beta_rv, h_rv]

    sampler, initial_values = construct_sampler({Y_rv: y_vv}, srng)

    assert len(sampler.sample_steps) == 4
    assert len(sampler.stages) == 3
    assert sampler.updates

    tau_post_step = sampler.sample_steps[tau_rv]
    # These are *very* rough checks of the resulting graphs
    assert isinstance(tau_post_step.owner.op, HorseshoeGibbsKernel)
    assert_consistent_rng_updates(tau_post_step)

    lmbda_post_step = sampler.sample_steps[lmbda_rv]
    assert isinstance(lmbda_post_step.owner.op, HorseshoeGibbsKernel)
    assert_consistent_rng_updates(lmbda_post_step)

    beta_post_step = sampler.sample_steps[beta_rv]
    assert isinstance(beta_post_step.owner.op, NBRegressionGibbsKernel)
    assert_consistent_rng_updates(beta_post_step)

    h_post_step = sampler.sample_steps[h_rv]
    assert isinstance(h_post_step.owner.op, DispersionGibbsKernel)
    assert_consistent_rng_updates(h_post_step)

    inputs = [X, a, b, y_vv] + [initial_values[rv] for rv in sample_vars]
    outputs = [sampler.sample_steps[rv] for rv in sample_vars]

    subsuming_elemwises = [
        n for n in io_toposort([], outputs) if isinstance(n.op, SubsumingElemwise)
    ]
    assert not any(subsuming_elemwises)

    sample_step = aesara.function(
        inputs,
        outputs,
        updates=sampler.updates,
        on_unused_input="ignore",
    )

    rng = np.random.default_rng(2309)

    N = 100
    M = 10
    S = toeplitz(0.5 ** np.arange(M))
    X_val = rng.multivariate_normal(np.zeros(M), S, size=N)

    a_val, b_val = 100.0, 1.0
    beta_true = np.array([2, 0.02, 0.2, 0.1, 1] + [0.0] * (M - 5))
    tau_val, lmbda_val, h_val = 1.0, np.ones(M), 10.0

    y_fn = aesara.function([X, a, b, beta_rv], Y_rv)
    y_val = y_fn(X_val, a_val, b_val, beta_true)

    beta_pst_vals = []
    tau_pst_val, lmbda_pst_val, beta_pst_val, h_pst_val = (
        tau_val,
        lmbda_val,
        np.zeros(M),
        h_val,
    )
    for i in range(100):
        tau_pst_val, lmbda_pst_val, beta_pst_val, h_pst_val = sample_step(
            X_val,
            a_val,
            b_val,
            y_val,
            tau_pst_val,
            lmbda_pst_val,
            beta_pst_val,
            h_pst_val,
        )
        beta_pst_vals += [beta_pst_val]

    beta_pst_mean = np.mean(beta_pst_vals, axis=0)
    assert np.allclose(beta_pst_mean, beta_true, atol=1e-1, rtol=1e-1)
