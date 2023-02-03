import aesara.tensor as at
import pytest
from aesara.graph.fg import FunctionGraph
from aesara.graph.kanren import KanrenRelationSub

from aemcmc.transforms import invgamma_exponential, location_scale_transform


def test_normal_scale_loc_transform_lift():
    srng = at.random.RandomStream(0)
    mu_rv = srng.halfnormal(1.0)
    sigma_rv = srng.halfcauchy(1)
    Y_rv = srng.normal(mu_rv, sigma_rv)

    fgraph = FunctionGraph(outputs=[Y_rv], clone=False)
    res = KanrenRelationSub(location_scale_transform).transform(
        fgraph, fgraph.outputs[0].owner
    )[0]

    # Make sure that Y_rv gets replaced with an addition
    assert res.owner.op == at.add
    lhs = res.owner.inputs[0]
    assert isinstance(lhs.owner.op, type(at.random.halfnormal))
    rhs = res.owner.inputs[1]
    assert rhs.owner.op == at.mul
    assert isinstance(rhs.owner.inputs[0].owner.op, type(at.random.halfcauchy))
    assert isinstance(rhs.owner.inputs[1].owner.op, type(at.random.normal))


@pytest.mark.xfail(
    reason="Op.__call__ does not dispatch to Op.make_node for some RandomVariable and etuple evaluation returns an error"
)
def test_normal_scale_loc_transform_sink():
    srng = at.random.RandomStream(0)
    mu_rv = srng.halfnormal(1.0)
    sigma_rv = srng.halfcauchy(1)
    std_normal_rv = srng.normal(0, 1)
    Y_at = mu_rv + sigma_rv * std_normal_rv

    fgraph = FunctionGraph(outputs=[Y_at], clone=False)
    res = KanrenRelationSub(lambda x, y: location_scale_transform(y, x)).transform(
        fgraph, fgraph.outputs[0].owner
    )[0]

    assert isinstance(res.owner.op, type(at.random.normal))


def test_invgamma_to_exp():
    srng = at.random.RandomStream(0)
    c_at = at.scalar()
    X_rv = srng.invgamma(1.0, c_at)

    fgraph = FunctionGraph(outputs=[X_rv], clone=False)
    res = KanrenRelationSub(invgamma_exponential).transform(
        fgraph, fgraph.outputs[0].owner
    )[0]

    Y_rv = 1.0 / srng.exponential(c_at)

    assert res.owner.op == Y_rv.owner.op
    assert isinstance(res.owner.inputs[1].owner.op, type(Y_rv.owner.inputs[1].owner.op))
    assert res.owner.inputs[1].owner.inputs[-1] == c_at


@pytest.mark.xfail(
    reason="Op.__call__ does not dispatch to Op.make_node for some RandomVariable and etuple evaluation returns an error"
)
def test_invgamma_from_exp():
    srng = at.random.RandomStream(0)
    c_at = at.scalar()
    X_rv = 1.0 / srng.exponential(c_at)

    fgraph = FunctionGraph(outputs=[X_rv], clone=False)
    res = KanrenRelationSub(lambda x, y: invgamma_exponential(y, x)).transform(
        fgraph, fgraph.outputs[0].owner
    )[0]

    Y_rv = srng.invgamma(1.0, c_at)

    assert isinstance(res.owner.op, type(Y_rv.owner.op))
    assert res.owner.inputs[-1] == c_at
