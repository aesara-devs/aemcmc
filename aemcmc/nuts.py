from typing import Callable, Dict, Tuple

import aesara
from aehmc import nuts as aehmc_nuts
from aehmc.utils import RaveledParamsMap
from aeppl import joint_logprob
from aeppl.transforms import RVTransform, TransformValuesOpt, _default_transformed_rv
from aesara.tensor.random import RandomStream
from aesara.tensor.var import TensorVariable

from aemcmc.utils import ModelInfo

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


def nuts(
    srng: RandomStream,
    model: ModelInfo,
    inverse_mass_matrix: TensorVariable,
    step_size: TensorVariable,
) -> Tuple[NUTSStateType, NUTSKernelType]:
    """Build a NUTS kernel and the initial state.

    This function currently assumes that we will update the value of all of the
    model's variables with the NUTS sampler.

    Parameters
    ----------
    model
        The Aesara model whose posterior distribution we wish to sample from
        passed as a `ModelInfo` instance.
    step_size
        The step size used in the symplectic integrator.
    inverse_mass_matrix
        One or two-dimensional array used as the inverse mass matrix that
        defines the euclidean metric.

    """
    unobserved_rvs = tuple(
        rv for rv in model.rvs_to_values.keys() if rv not in model.observed_rvs
    )
    unobserved_rvs_to_values = {rv: model.rvs_to_values[rv] for rv in unobserved_rvs}
    observed_vvs = tuple(model.rvs_to_values[rv] for rv in model.observed_rvs)

    # Algorithms in the HMC family can more easily explore the posterior distribution
    # when the support of each random variable's distribution is unconstrained.
    # First we build the logprob graph in the transformed space.
    transforms = {
        vv: get_transform(rv) if rv in unobserved_rvs else None
        for rv, vv in model.rvs_to_values.items()
    }

    logprob_sum = joint_logprob(
        model.rvs_to_values, extra_rewrites=TransformValuesOpt(transforms)
    )

    # Then we transform the value variables.
    transformed_vvs = {
        vv: transform_forward(rv, vv, transforms[vv])
        for rv, vv in unobserved_rvs_to_values.items()
    }

    # Algorithms in `aehmc` work with flat arrays and we need to ravel parameter
    # values to use them as an input to the NUTS kernel.
    rp_map = RaveledParamsMap(tuple(transformed_vvs.values()))
    rp_map.ref_params = tuple(transformed_vvs.keys())

    # We can now write the logprob function associated with the model and build
    # the NUTS kernel.
    def logprob_fn(q):
        unraveled_q = rp_map.unravel_params(q)
        unraveled_q.update({vv: vv for vv in observed_vvs})

        memo = aesara.graph.basic.clone_get_equiv(
            [], [logprob_sum], copy_inputs=False, copy_orphans=False, memo=unraveled_q
        )

        return memo[logprob_sum]

    nuts_kernel = aehmc_nuts.new_kernel(srng, logprob_fn)

    def step_fn(state):
        """Take one step with the NUTS kernel.

        The NUTS kernel works with a state that contains the current value of
        the variables, but also the current value of the potential and its
        gradient, and we need to carry this state forward. We also return the
        unraveled parameter values in both the original and transformed space.

        """
        (new_q, new_pe, new_peg, *_), updates = nuts_kernel(
            *state, step_size, inverse_mass_matrix
        )
        new_state = (new_q, new_pe, new_peg)
        transformed_params = rp_map.unravel_params(new_q)
        params = {
            vv: transform_backward(rv, transformed_params[vv], transforms[vv])
            for rv, vv in unobserved_rvs_to_values.items()
        }
        return (new_state, params, transformed_params), updates

    # Finally we build the initial state
    initial_q = rp_map.ravel_params(tuple(transformed_vvs.values()))
    initial_state = aehmc_nuts.new_state(initial_q, logprob_fn)

    return initial_state, step_fn


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
