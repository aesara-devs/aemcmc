from typing import Callable, Dict, Tuple

import aesara
import aesara.tensor as at
from aehmc import nuts as aehmc_nuts
from aehmc.utils import RaveledParamsMap
from aeppl import joint_logprob
from aeppl.transforms import (
    RVTransform,
    TransformValuesRewrite,
    _default_transformed_rv,
)
from aesara import config
from aesara.tensor.random import RandomStream
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.var import TensorVariable

NUTSStateType = Tuple[TensorVariable, TensorVariable, TensorVariable]
NUTSKernelType = Callable[
    [NUTSStateType],
    Tuple[
        Tuple[
            NUTSStateType,
            Dict[TensorVariable, TensorVariable],
            Dict[TensorVariable, TensorVariable],
        ],
        Dict,
    ],
]


def construct_nuts_sampler(
    srng: RandomStream,
    to_sample_rvs,  # RVs to sample
    rvs_to_values,  # All RVs to values
) -> Tuple[Dict[RandomVariable, TensorVariable], Dict, Dict[str, TensorVariable]]:
    """Build a NUTS kernel and the initial state.

    This function currently assumes that we will update the value of all of the
    model's variables with the NUTS sampler.

    Parameters
    ----------
    rvs_to_samples
        A sequence that contains the random variables whose posterior
        distribution we wish to sample from.
    rvs_to_values
        A dictionary that maps all random variables in the model (including
        those not sampled with NUTS) to their value variable.

    Returns
    -------
    A NUTS sampling step for each variable.

    """

    # Algorithms in the HMC family can more easily explore the posterior distribution
    # when the support of each random variable's distribution is unconstrained.
    # First we build the logprob graph in the transformed space.
    transforms = {
        vv: get_transform(rv) for rv, vv in rvs_to_values.items() if rv in to_sample_rvs
    }

    logprob_sum = joint_logprob(
        rvs_to_values, extra_rewrites=TransformValuesRewrite(transforms)
    )

    # Then we transform the value variables.
    transformed_vvs = {
        vv: transform_forward(rv, vv, transforms[vv])
        for rv, vv in rvs_to_values.items()
        if rv in to_sample_rvs
    }

    # Algorithms in `aehmc` work with flat arrays and we need to ravel parameter
    # values to use them as an input to the NUTS kernel.
    rp_map = RaveledParamsMap(tuple(transformed_vvs.values()))
    rp_map.ref_params = tuple(transformed_vvs.keys())

    # Make shared variables for all the non-NUTS sampled terms
    non_nuts_vals = {
        vv: vv for rv, vv in rvs_to_values.items() if rv not in to_sample_rvs
    }

    # We can now write the logprob function associated with the model and build
    # the NUTS kernel.
    def logprob_fn(q):
        unraveled_q = rp_map.unravel_params(q)
        unraveled_q.update(non_nuts_vals)
        memo = aesara.graph.basic.clone_get_equiv(
            [], [logprob_sum], copy_inputs=False, copy_orphans=False, memo=unraveled_q
        )

        return memo[logprob_sum]

    # Finally we build the NUTS sampling step
    nuts_kernel = aehmc_nuts.new_kernel(srng, logprob_fn)

    initial_q = rp_map.ravel_params(tuple(transformed_vvs.values()))
    initial_state = aehmc_nuts.new_state(initial_q, logprob_fn)

    # Initialize the parameter values
    step_size = at.scalar("step_size", dtype=config.floatX)
    inverse_mass_matrix = at.tensor(
        name="inverse_mass_matrix", shape=initial_q.type.shape, dtype=config.floatX
    )

    # TODO: Does that lead to wasteful computation? Or is it handled by Aesara?
    (new_q, *_), updates = nuts_kernel(*initial_state, step_size, inverse_mass_matrix)
    transformed_params = rp_map.unravel_params(new_q)
    params = {
        rv: transform_backward(rv, transformed_params[vv], transforms[vv])
        for rv, vv in rvs_to_values.items()
        if rv in to_sample_rvs
    }

    return (
        params,
        updates,
        {"step_size": step_size, "inverse_mass_matrix": inverse_mass_matrix},
    )


def get_transform(rv: TensorVariable):
    """Get the default transform associated with the random variable."""
    transform = _default_transformed_rv(rv.owner.op, rv.owner)
    if transform:
        return transform.op.transform
    else:
        return None


def transform_forward(rv: TensorVariable, vv: TensorVariable, transform: RVTransform):
    """Push variables to the transformed space."""
    if transform:
        res = transform.forward(vv, *rv.owner.inputs)
        if vv.name:
            res.name = f"{vv.name}_trans"
        return res
    else:
        return vv


def transform_backward(rv: TensorVariable, vv: TensorVariable, transform: RVTransform):
    """Pull variables back from the transformed space."""
    if transform:
        res = transform.backward(vv, *rv.owner.inputs)
        return res
    else:
        return vv
