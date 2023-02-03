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
from aesara.compile.sharedvalue import SharedVariable
from aesara.graph.basic import Apply, graph_inputs
from aesara.graph.type import Constant
from aesara.tensor.random import RandomStream
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.var import TensorVariable

from aemcmc.types import SamplingStep

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


class NUTSKernel(SamplingStep):
    """An `Op` that represents the update of one or many random variables
    with the NUTS sampling algorithm.

    """


def step(
    srng: RandomStream,
    to_sample_rvs: Dict[RandomVariable, TensorVariable],
    realized_rvs_to_values: Dict[RandomVariable, TensorVariable],
) -> Tuple[
    Dict[RandomVariable, TensorVariable], Dict, Tuple[TensorVariable, TensorVariable]
]:
    """Build a NUTS sampling step and its initial state.

    This sampling step works with variables in their original space, to create
    the sampling step we thus need to:

    1. Create the initial value for the variables to sample;
    2. Create a log-density graph that works in the transformed space, and
       build a NUTS kernel that uses this graph;
    3. Apply the default transformations to the initial values;
    4. Apply the NUTS kernel to the transformed initial values;
    5. Apply the backward transformation to the updated values.

    Parameters
    ----------
    rvs_to_samples
        A dictionary that maps the random variables whose posterior
        distribution we wish to sample from to their initial values.
    realized_rvs_to_values
        A dictionary that maps the random variables not sampled by NUTS to their
        realized value. These variables can either correpond to observations, or
        to variables whose value is set by a different sampler.

    Returns
    -------
    A NUTS sampling step for each random variable, their initial values, the
    shared variable updates and the NUTS parameters.

    """

    # Get the initial values for the random variables that are assigned this
    # sampling step.
    initial_values = to_sample_rvs.values()

    # Algorithms in the HMC family can more easily explore the posterior distribution
    # when the support of each random variable's distribution is unconstrained. Get
    # the default transform that corresponds to each random variable.
    transforms = {rv: get_transform(rv) for rv in to_sample_rvs}
    transformed_values = [
        transform_forward(rv, vv, transforms[rv])
        for rv, vv in zip(to_sample_rvs, initial_values)
    ]

    # Build the graph for the joint log-density of the model, setting the value
    # of the realized variables. The placeholder nodes are defined in the
    # transformed space and will be replaced by their actual value when building
    # the NUTS kernel.
    logprob, placeholder_values = joint_logprob(
        *to_sample_rvs.keys(),
        realized=realized_rvs_to_values,
        extra_rewrites=TransformValuesRewrite(transforms),
    )

    # Algorithms in `AeHMC` work with flat arrays so we need to ravel parameter
    # values to use them as an input to the NUTS kernel. The following creates
    # an object that can ravel the `transformed_values` and can unravel
    # flattened values in a dictionary that maps placeholder values to the
    # unraveled values.
    rp_map = RaveledParamsMap(transformed_values)
    rp_map.ref_params = placeholder_values

    # Within the NUTS kernel we unravel the flat position and replace the
    # placeholder values by their current value in the logprob graph.
    def logprob_fn(q):
        unraveled_q = rp_map.unravel_params(q)
        memo = aesara.graph.basic.clone_get_equiv(
            [], [logprob], copy_inputs=False, copy_orphans=False, memo=unraveled_q
        )
        return memo[logprob]

    # Build the NUTS kernel and initialize the state
    nuts_kernel = aehmc_nuts.new_kernel(srng, logprob_fn)

    initial_q = rp_map.ravel_params(transformed_values)
    initial_state = aehmc_nuts.new_state(initial_q, logprob_fn)

    # Initialize the parameter values
    step_size = at.scalar("step_size", dtype=config.floatX)
    inverse_mass_matrix = at.tensor(
        name="inverse_mass_matrix", shape=initial_q.type.shape, dtype=config.floatX
    )

    # Apply the NUTS kernel to the initial state, unravel and transform the updated
    # values back to the original space.
    (new_q, *_), updates = nuts_kernel(*initial_state, step_size, inverse_mass_matrix)
    transformed_params = rp_map.unravel_params(new_q)
    results = {
        rv: transform_backward(rv, transformed_params[pv], transforms[rv])
        for rv, pv in zip(to_sample_rvs, placeholder_values)
    }

    return (
        results,
        updates,
        (step_size, inverse_mass_matrix),
    )


def construct_sampler(
    srng: RandomStream,
    to_sample_rvs: Dict[RandomVariable, TensorVariable],
    realized_rvs_to_values: Dict[RandomVariable, TensorVariable],
) -> Tuple[Dict[RandomVariable, TensorVariable], Dict, Dict[Apply, TensorVariable]]:
    results, updates, parameters = step(srng, to_sample_rvs, realized_rvs_to_values)

    # Build an `Op` that represents the NUTS sampling step
    update_outputs = list(updates.values())
    outputs = list(results.values()) + update_outputs
    inputs = [
        var_in
        for var_in in graph_inputs(outputs)
        if not isinstance(var_in, Constant) and not isinstance(var_in, SharedVariable)
    ]
    nuts_op = NUTSKernel(inputs, outputs)

    posterior = nuts_op(*inputs)
    results = {rv: posterior[i] for i, rv in enumerate(to_sample_rvs)}

    updates_input = posterior[0].owner.inputs[len(inputs) :]
    updates_output = posterior[len(results) :]
    updates = {
        updates_input[i]: update_out for i, update_out in enumerate(updates_output)
    }

    return results, updates, {nuts_op: parameters}


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
