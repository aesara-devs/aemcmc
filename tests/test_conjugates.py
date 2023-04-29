import aesara
import aesara.tensor as at
import pytest
from aesara.graph.rewriting.unify import eval_if_etuple
from aesara.tensor.random import RandomStream
from etuples import etuple, etuplize
from kanren import run
from unification import var

from aemcmc.conjugates import (
    beta_bernoulli_conjugateo,
    beta_binomial_conjugateo,
    beta_negative_binomial_conjugateo,
    gamma_poisson_conjugateo,
    uniform_pareto_conjugateo,
)


def test_gamma_poisson_conjugate_contract():
    """Produce the closed-form posterior for the poisson observation model with
    a gamma prior.

    """
    srng = RandomStream(0)

    alpha_tt = at.scalar("alpha")
    beta_tt = at.scalar("beta")
    z_rv = srng.gamma(alpha_tt, beta_tt)

    Y_rv = srng.poisson(z_rv)
    y_vv = Y_rv.clone()
    y_vv.name = "y"

    q_lv = var()
    (posterior_expr,) = run(1, q_lv, gamma_poisson_conjugateo(y_vv, Y_rv, q_lv))
    posterior = eval_if_etuple(posterior_expr)

    assert isinstance(posterior.owner.op, type(at.random.gamma))


@pytest.mark.xfail(
    reason="Op.__call__ does not dispatch to Op.make_node for some RandomVariable and etuple evaluation returns an error"
)
def test_gamma_poisson_conjugate_expand():
    """Expand a contracted beta-binomial observation model."""

    srng = RandomStream(0)

    alpha_tt = at.scalar("alpha")
    beta_tt = at.scalar("beta")
    y_vv = at.iscalar("y")
    Y_rv = srng.gamma(alpha_tt + y_vv, beta_tt + 1)

    e_lv = var()
    (expanded_expr,) = run(1, e_lv, gamma_poisson_conjugateo(e_lv, y_vv, Y_rv))
    expanded = eval_if_etuple(expanded_expr)

    assert isinstance(expanded.owner.op, type(at.random.gamma))


def test_beta_binomial_conjugate_contract():
    """Produce the closed-form posterior for the binomial observation model with
    a beta prior.

    """
    srng = RandomStream(0)

    alpha_tt = at.scalar("alpha")
    beta_tt = at.scalar("beta")
    p_rv = srng.beta(alpha_tt, beta_tt, name="p")

    n_tt = at.iscalar("n")
    Y_rv = srng.binomial(n_tt, p_rv)
    y_vv = Y_rv.clone()
    y_vv.tag.name = "y"

    q_lv = var()
    (posterior_expr,) = run(1, q_lv, beta_binomial_conjugateo(y_vv, Y_rv, q_lv))
    posterior = eval_if_etuple(posterior_expr)

    assert isinstance(posterior.owner.op, type(at.random.beta))

    # Build the sampling function and check the results on limiting cases.
    sample_fn = aesara.function((alpha_tt, beta_tt, y_vv, n_tt), posterior)
    assert sample_fn(1.0, 1.0, 1000, 1000) == pytest.approx(
        1.0, abs=0.01
    )  # only successes
    assert sample_fn(1.0, 1.0, 0, 1000) == pytest.approx(0.0, abs=0.01)  # zero success


@pytest.mark.xfail(
    reason="Op.__call__ does not dispatch to Op.make_node for some RandomVariable and etuple evaluation returns an error"
)
def test_beta_binomial_conjugate_expand():
    """Expand a contracted beta-binomial observation model."""

    srng = RandomStream(0)

    alpha_tt = at.scalar("alpha")
    beta_tt = at.scalar("beta")
    y_vv = at.iscalar("y")
    n_tt = at.iscalar("n")
    Y_rv = srng.beta(alpha_tt + y_vv, beta_tt + n_tt - y_vv)

    e_lv = var()
    (expanded_expr,) = run(1, e_lv, beta_binomial_conjugateo(e_lv, y_vv, Y_rv))
    expanded = eval_if_etuple(expanded_expr)

    assert isinstance(expanded.owner.op, type(at.random.beta))


def test_beta_negative_binomial_conjugate_contract():
    """Produce the closed-form posterior for the binomial observation model with
    a beta prior.

    """
    srng = RandomStream(0)

    alpha_tt = at.scalar("alpha")
    beta_tt = at.scalar("beta")
    p_rv = srng.beta(alpha_tt, beta_tt, name="p")

    n_tt = at.iscalar("n")
    Y_rv = srng.negative_binomial(n_tt, p_rv)
    y_vv = Y_rv.clone()
    y_vv.tag.name = "y"

    q_lv = var()
    (posterior_expr,) = run(
        1, q_lv, beta_negative_binomial_conjugateo(y_vv, Y_rv, q_lv)
    )
    posterior = eval_if_etuple(posterior_expr)

    assert isinstance(posterior.owner.op, type(at.random.beta))

    # Build the sampling function and check the results on limiting cases.
    sample_fn = aesara.function((alpha_tt, beta_tt, y_vv, n_tt), posterior)
    assert sample_fn(1.0, 1.0, 1000, 0) == pytest.approx(
        1.0, abs=0.01
    )  # only successes
    assert sample_fn(1.0, 1.0, 0, 1000) == pytest.approx(0.0, abs=0.01)  # no success


@pytest.mark.xfail(
    reason="Op.__call__ does not dispatch to Op.make_node for some RandomVariable and etuple evaluation returns an error"
)
def test_beta_negative_binomial_conjugate_expand():
    """Expand a contracted beta-binomial observation model."""

    srng = RandomStream(0)

    alpha_tt = at.scalar("alpha")
    beta_tt = at.scalar("beta")
    y_vv = at.iscalar("y")
    n_tt = at.iscalar("n")
    Y_rv = srng.beta(alpha_tt + y_vv, beta_tt + n_tt)

    e_lv = var()
    (expanded_expr,) = run(1, e_lv, beta_negative_binomial_conjugateo(e_lv, y_vv, Y_rv))
    expanded = eval_if_etuple(expanded_expr)

    assert isinstance(expanded.owner.op, type(at.random.beta))


def test_beta_bernoulli_conjugate_contract():
    """Produce the closed-form posterior for the binomial observation model with
    a beta prior.

    """
    srng = RandomStream(0)

    alpha_tt = at.scalar("alpha")
    beta_tt = at.scalar("beta")
    p_rv = srng.beta(alpha_tt, beta_tt, name="p")

    Y_rv = srng.bernoulli(p_rv)
    y_vv = Y_rv.clone()
    y_vv.tag.name = "y"

    q_lv = var()
    (posterior_expr,) = run(1, q_lv, beta_bernoulli_conjugateo(y_vv, Y_rv, q_lv))
    posterior = eval_if_etuple(posterior_expr)

    assert isinstance(posterior.owner.op, type(at.random.beta))

    # Build the sampling function and check the results on limiting cases.
    sample_fn = aesara.function((alpha_tt, beta_tt, y_vv), posterior)
    assert sample_fn(1.0, 1.0, 1) == pytest.approx(1.0, abs=0.3)  # only successes
    assert sample_fn(1.0, 1.0, 0) == pytest.approx(0.0, abs=0.3)  # no success


@pytest.mark.xfail(
    reason="Op.__call__ does not dispatch to Op.make_node for some RandomVariable and etuple evaluation returns an error"
)
def test_beta_bernoulli_conjugate_expand():
    """Expand a contracted beta-binomial observation model."""

    srng = RandomStream(0)

    alpha_tt = at.scalar("alpha")
    beta_tt = at.scalar("beta")
    y_vv = at.iscalar("y")
    n_tt = at.iscalar("n")
    Y_rv = srng.beta(alpha_tt + y_vv, beta_tt + n_tt - y_vv)

    e_lv = var()
    (expanded_expr,) = run(1, e_lv, beta_bernoulli_conjugateo(e_lv, y_vv, Y_rv))
    expanded = eval_if_etuple(expanded_expr)

    assert isinstance(expanded.owner.op, type(at.random.beta))


def test_uniform_pareto_conjugate_contract():
    """Produce the closed-form posterior for the uniform observation model with
    a pareto prior.

    """
    srng = RandomStream(0)

    xm_tt = at.scalar("xm")
    k_tt = at.scalar("k")
    theta_rv = srng.pareto(k_tt, xm_tt, name="theta")

    # zero = at.iscalar("zero")
    Y_rv = srng.uniform(0, theta_rv)
    y_vv = Y_rv.clone()
    y_vv.tag.name = "y"

    q_lv = var()
    (posterior_expr,) = run(1, q_lv, uniform_pareto_conjugateo(y_vv, Y_rv, q_lv))
    posterior = eval_if_etuple(posterior_expr)

    assert isinstance(posterior.owner.op, type(at.random.pareto))

    # Build the sampling function and check the results on limiting cases.
    sample_fn = aesara.function((xm_tt, k_tt, y_vv), posterior)
    assert sample_fn(1.0, 1000, 1) == pytest.approx(1.0, abs=0.01)  # k = 1000
    assert sample_fn(1.0, 1, 0) == pytest.approx(0.0, abs=0.01)  # all zeros


def test_uniform_pareto_binomial_conjugate_expand():
    """Expand a contracted beta-binomial observation model."""

    srng = RandomStream(0)

    k_tt = at.scalar("k")
    y_vv = at.iscalar("y")
    n_tt = at.scalar("n")

    Y_rv = srng.pareto(at.max(y_vv), k_tt + n_tt)
    etuplize(Y_rv)

    # e_lv = var()
    # (expanded_expr,) = run(1, e_lv, uniform_pareto_conjugateo(e_lv, y_vv, Y_rv))
    # expanded = eval_if_etuple(expanded_expr)

    # assert isinstance(expanded.owner.op, type(at.random.pareto))
    from aesara.tensor.math import MaxAndArgmax
    from kanren import eq, run
    from unification import var

    observed_val = var()
    axis_lv = var()
    new_x_et = etuple(etuple(MaxAndArgmax, axis_lv), observed_val)

    k_lv, n_lv = var(), var()
    new_k_et = etuple(etuplize(at.add), k_lv, n_lv)

    theta_rng_lv = var()
    theta_size_lv = var()
    theta_type_idx_lv = var()
    theta_posterior_et = etuple(
        etuplize(at.random.pareto),
        theta_rng_lv,
        theta_size_lv,
        theta_type_idx_lv,
        new_x_et,
        new_k_et,
    )

    run(0, (new_x_et, new_k_et), eq(Y_rv, theta_posterior_et))
