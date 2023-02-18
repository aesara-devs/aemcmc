import aesara.tensor as at
import numpy as np
from aesara.graph.basic import equal_computations
from aesara.tensor.elemwise import DimShuffle, Elemwise
from cons import car, cdr
from etuples import etuple, etuplize
from unification import unify

from aemcmc.rewriting import (
    SubsumingElemwise,
    construct_ir_fgraph,
    local_elemwise_dimshuffle_subsume,
)


def test_SubsumingElemwise_basics():
    a = at.vector("a")
    b = at.scalar("b")

    x = a * b

    assert isinstance(x.owner.op, Elemwise)
    b_ds = x.owner.inputs[1].owner.op
    assert isinstance(b_ds, DimShuffle)

    ee_mul_op = SubsumingElemwise([a, b], [x])

    assert ee_mul_op != ee_mul_op.clone()
    assert str(ee_mul_op) == "SubsumingElemwise{mul}"

    s = unify(at.mul, ee_mul_op)
    assert s is not False

    assert car(ee_mul_op) == car(x.owner.op)
    assert cdr(ee_mul_op) == cdr(x.owner.op)

    s = unify(etuplize(at.mul), etuplize(ee_mul_op))
    assert s is not False

    ee_et = etuplize(ee_mul_op(a, b))
    x_et = etuple(etuplize(at.mul), a, b)

    s = unify(ee_et, x_et)
    assert s is not False

    # TODO: Consider making this possible
    # s = unify(ee_mul(a, b), x)
    # assert s is not False


def test_local_elemwise_dimshuffle_subsume_basic():
    srng = at.random.RandomStream(2398)

    a = at.vector("a")
    b = srng.normal(0, 1, name="b")

    x = a * b

    node = x.owner
    assert isinstance(node.op, Elemwise)
    b_ds = node.inputs[1].owner.op
    assert isinstance(b_ds, DimShuffle)

    (res,) = local_elemwise_dimshuffle_subsume.transform(None, node)
    assert isinstance(res.owner.op, SubsumingElemwise)
    assert equal_computations(
        [res.owner.op.inner_outputs[0]], [x], res.owner.op.inner_inputs[:2], [a, b]
    )
    assert res.owner.inputs == [a, b]


def test_local_elemwise_dimshuffle_subsume_transpose():
    """Make sure that `local_elemwise_dimshuffle_subsume` is applied selectively."""
    srng = at.random.RandomStream(2398)

    a = at.vector("a")
    # This transpose shouldn't be subsumed, but the one applied to `a` by
    # `Elemwise.make_node` should
    b = srng.normal(at.arange(4).reshape((2, 2)), 1, name="b").T

    x = a * b

    node = x.owner
    assert isinstance(node.op, Elemwise)
    b_ds = node.inputs[1].owner.op
    assert isinstance(b_ds, DimShuffle)

    (res,) = local_elemwise_dimshuffle_subsume.transform(None, node)
    assert isinstance(res.owner.op, SubsumingElemwise)
    assert equal_computations(
        [res.owner.op.inner_outputs[0]], [x], res.owner.op.inner_inputs[:2], [a, b]
    )
    assert res.owner.inputs == [a, b]

    a = at.tensor(np.float64, shape=(None, None, None), name="a")
    # Again, the transpose part shouldn't be subsumed, but the added broadcast
    # dimension should
    b = srng.normal(at.arange(4).reshape((2, 2)), 1, name="b")
    b_ds = b.dimshuffle(("x", 1, 0))

    x = a * b_ds

    node = x.owner
    assert isinstance(node.op, Elemwise)
    b_ds = node.inputs[1].owner.op
    assert isinstance(b_ds, DimShuffle)

    (res,) = local_elemwise_dimshuffle_subsume.transform(None, node)
    assert isinstance(res.owner.op, SubsumingElemwise)
    assert res.owner.inputs[0] == a
    # The input corresponding to `b`/`b_ds` should be equivalent to `b.T`
    assert isinstance(res.owner.inputs[1].owner.op, DimShuffle)
    assert equal_computations([b.T], [res.owner.inputs[1]])


def test_SubsumingElemwise_constant_inputs():
    """Make sure constant inputs are handled correctly by `SubsumingElemwise`."""

    srng = at.random.RandomStream(0)

    s = at.lscalar("s")
    # The `1` is the constant input to a `true_divide` `Elemwise` that should be
    # "subsumed"
    Z = srng.exponential(1, size=s, name="Z")
    mu = 1 / Z
    Y = srng.normal(mu, name="Y")
    y = Y.clone()
    y.name = "y"

    res, *_ = construct_ir_fgraph({Y: y}, clone=False)

    normal_node = res.outputs[1].owner
    subelem_node = normal_node.inputs[3].owner
    assert isinstance(subelem_node.op, SubsumingElemwise)
    assert subelem_node.inputs == [Z]
