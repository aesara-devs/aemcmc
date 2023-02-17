import aesara
import aesara.tensor as at
import numpy as np
from aesara.compile.sharedvalue import SharedVariable

from aemcmc.sample import sample_prior


def test_sample_prior():
    srng = at.random.RandomStream(123)

    mu_rv = srng.normal(0, 1, name="mu")
    Y_rv = srng.normal(mu_rv, 1.0, name="Y")
    Z_rv = srng.gamma(0.5, 0.5, name="Z")

    samples, updates = sample_prior(srng, 10, Y_rv)
    fn = aesara.function([], samples, updates=updates)

    # Make sure that `Z_rv` doesn't sneak into our prior sampling.
    rng_objects = set(
        var.get_value(borrow=True)
        for var in fn.maker.fgraph.variables
        if isinstance(var, SharedVariable)
    )

    assert mu_rv.owner.inputs[0].get_value(borrow=True) in rng_objects
    assert Y_rv.owner.inputs[0].get_value(borrow=True) in rng_objects
    assert Z_rv.owner.inputs[0].get_value(borrow=True) not in rng_objects

    samples_vals = fn()
    assert np.shape(np.unique(samples_vals)) == (10,)

    # Try it again, but without a default update
    Y_rv.owner.inputs[0].default_update = None

    samples, updates = sample_prior(srng, 10, Y_rv)
    fn = aesara.function([], samples, updates=updates)

    rng_objects = set(
        var.get_value(borrow=True)
        for var in fn.maker.fgraph.variables
        if isinstance(var, SharedVariable)
    )

    assert mu_rv.owner.inputs[0].get_value(borrow=True) in rng_objects
    assert Y_rv.owner.inputs[0].get_value(borrow=True) in rng_objects
    assert Z_rv.owner.inputs[0].get_value(borrow=True) not in rng_objects

    samples_vals = fn()
    assert np.shape(np.unique(samples_vals)) == (10,)
