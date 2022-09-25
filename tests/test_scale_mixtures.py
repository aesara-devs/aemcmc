import aesara.tensor as at
import pytest
from aesara.graph.rewriting.unify import eval_if_etuple
from kanren import run, var

from aemcmc.scale_mixtures import halfcauchy_inverse_gamma


def test_halfcauchy_to_inverse_gamma_mixture():

    srng = at.random.RandomStream(0)
    A = at.scalar("A")
    X_rv = srng.halfcauchy(0, A)

    q_lv = var()
    results = run(0, q_lv, halfcauchy_inverse_gamma(X_rv, q_lv))

    found_mixture = False
    for res in results:
        try:
            mixture = eval_if_etuple(res)
            found_mixture = True
        except (AttributeError, TypeError):
            continue

    assert found_mixture is True
    assert isinstance(mixture.owner.op, type(at.sqrt))
    assert isinstance(mixture.owner.inputs[0].owner.op, type(at.random.invgamma))


@pytest.mark.xfail(
    reason="Op.__call__ does not dispatch to Op.make_node for some RandomVariable and etuple evaluation returns an error"
)
def test_halfcauchy_from_inverse_gamma_mixture():

    srng = at.random.RandomStream(0)
    A = at.scalar("A")
    a_rv = srng.invgamma(0.5, 1.0 / A**2)
    X_rv = at.sqrt(srng.invgamma(0.5, 1.0 / a_rv))

    q_lv = var()
    run(0, q_lv, halfcauchy_inverse_gamma(q_lv, X_rv))
