import aesara
import aesara.tensor as at
import pytest
from aesara.tensor.random import RandomStream
from etuples import etuplize
from kanren import run
from unification import var

from aemcmc.conjugates import (
    beta_binomial_conjugateo,
    beta_negative_binomial_conjugateo,
    gamma_poisson_conjugateo,
)


def test_gamma_poisson_conjugate_contract():
    """Produce the closed-form posterior for the poisson observation model with a gamma prior."""
    srng = RandomStream(0)

    alpha_tt = at.scalar("alpha")
    beta_tt = at.scalar("beta")
    z_rv = srng.gamma(alpha_tt, beta_tt)
    Y_rv = srng.poisson(z_rv)

    q_lv = var()
    (posterior_expr,) = run(1, q_lv, gamma_poisson_conjugateo(srng, Y_rv, q_lv))
    posterior = posterior_expr.evaled_obj

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
    Y_et = etuplize(Y_rv)

    e_lv = var()
    (expanded_expr,) = run(1, e_lv, gamma_poisson_conjugateo(srng, e_lv, Y_et))
    expanded = expanded_expr.evaled_obj

    assert isinstance(expanded.owner.op, type(at.random.gamma))


def test_beta_binomial_conjugate_contract():
    """Produce the closed-form posterior for the binomial observation model with a beta prior."""
    srng = RandomStream(0)

    alpha_tt = at.scalar("alpha")
    beta_tt = at.scalar("beta")
    p_rv = srng.beta(alpha_tt, beta_tt, name="p")

    n_tt = at.iscalar("n")
    Y_rv = srng.binomial(n_tt, p_rv)

    q_lv = var()
    (posterior_expr,) = run(1, q_lv, beta_binomial_conjugateo(srng, Y_rv, q_lv))
    posterior = posterior_expr.evaled_obj

    assert isinstance(posterior.owner.op, type(at.random.beta))

    # Build the sampling function and check the results on limiting cases.
    sample_fn = aesara.function((alpha_tt, beta_tt, Y_rv, n_tt), posterior)
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
    (expanded_expr,) = run(1, e_lv, beta_binomial_conjugateo(srng, e_lv, Y_rv))
    expanded = expanded_expr.evaled_obj

    assert isinstance(expanded.owner.op, type(at.random.beta))


def test_beta_negative_binomial_conjugate_contract():
    """Produce the closed-form posterior for the binomial observation model with a beta prior."""
    srng = RandomStream(0)

    alpha_tt = at.scalar("alpha")
    beta_tt = at.scalar("beta")
    p_rv = srng.beta(alpha_tt, beta_tt, name="p")

    n_tt = at.iscalar("n")
    Y_rv = srng.negative_binomial(n_tt, p_rv)

    q_lv = var()
    (posterior_expr,) = run(
        1, q_lv, beta_negative_binomial_conjugateo(srng, Y_rv, q_lv)
    )
    posterior = posterior_expr.evaled_obj

    assert isinstance(posterior.owner.op, type(at.random.beta))

    # Build the sampling function and check the results on limiting cases.
    sample_fn = aesara.function((alpha_tt, beta_tt, Y_rv, n_tt), posterior)
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
    (expanded_expr,) = run(1, e_lv, beta_negative_binomial_conjugateo(srng, e_lv, Y_rv))
    expanded = expanded_expr

    assert isinstance(expanded.owner.op, type(at.random.beta))
