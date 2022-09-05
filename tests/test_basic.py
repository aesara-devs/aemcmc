import aesara
import aesara.tensor as at
import numpy as np
import pytest
from aesara.graph.basic import graph_inputs, io_toposort
from aesara.ifelse import IfElse
from aesara.tensor.random import RandomStream
from aesara.tensor.random.basic import BetaRV, GammaRV
from scipy.linalg import toeplitz

from aemcmc.basic import construct_sampler
from aemcmc.rewriting import SubsumingElemwise


def test_closed_form_posterior_beta_binomial():
    srng = RandomStream(0)

    alpha_tt = at.scalar("alpha")
    beta_tt = at.scalar("beta")
    p_rv = srng.beta(alpha_tt, beta_tt, name="p")

    n_tt = at.iscalar("n")
    Y_rv = srng.binomial(n_tt, p_rv, name="Y")

    y_vv = Y_rv.clone()
    y_vv.name = "y"

    sample_steps, updates, initial_values = construct_sampler({Y_rv: y_vv}, srng)

    p_posterior_step = sample_steps[p_rv]
    assert isinstance(p_posterior_step.owner.op, BetaRV)


def test_closed_form_posterior_gamma_poisson():
    srng = RandomStream(0)

    alpha_tt = at.scalar("alpha")
    beta_tt = at.scalar("beta")
    l_rv = srng.gamma(alpha_tt, beta_tt, name="p")

    Y_rv = srng.poisson(l_rv, name="Y")

    y_vv = Y_rv.clone()
    y_vv.name = "y"

    sample_steps, updates, initial_values = construct_sampler({Y_rv: y_vv}, srng)

    p_posterior_step = sample_steps[l_rv]
    assert isinstance(p_posterior_step.owner.op, GammaRV)


def test_no_samplers():
    srng = RandomStream(0)

    size = at.lscalar("size")
    tau_rv = srng.halfcauchy(0, 1, name="tau")
    Y_rv = srng.halfcauchy(0, tau_rv, size=size, name="Y")

    y_vv = Y_rv.clone()
    y_vv.name = "y"

    with pytest.raises(NotImplementedError):
        construct_sampler({Y_rv: y_vv}, srng)


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

    sample_steps, updates, initial_values = construct_sampler({Y_rv: y_vv}, srng)

    assert len(sample_steps) == 4
    assert updates

    tau_post_step = sample_steps[tau_rv]
    # These are *very* rough checks of the resulting graphs
    assert tau_post_step.owner.op == at.reciprocal

    lmbda_post_step = sample_steps[lmbda_rv]
    assert lmbda_post_step.owner.op == at.reciprocal

    beta_post_step = sample_steps[beta_rv]
    assert isinstance(beta_post_step.owner.op, IfElse)

    assert y_vv in graph_inputs([beta_post_step])

    inputs = [X, a, b, y_vv] + [initial_values[rv] for rv in sample_vars]
    outputs = [sample_steps[rv] for rv in sample_vars]

    subsuming_elemwises = [
        n for n in io_toposort([], outputs) if isinstance(n.op, SubsumingElemwise)
    ]
    assert not any(subsuming_elemwises)

    sample_step = aesara.function(
        inputs,
        outputs,
        updates=updates,
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
