import aesara
import aesara.tensor as at
import numpy as np
import pytest
from aeppl import joint_logprob
from aesara.tensor.random import RandomStream

from aemcmc.nuts import nuts
from aemcmc.utils import ModelInfo


def test_nuts():
    srng = RandomStream(0)
    mu_rv = srng.normal(0, 1, name="mu")
    sigma_rv = srng.halfnormal(0.0, 1.0, name="sigma")
    Y_rv = srng.normal(mu_rv, sigma_rv, name="Y")

    mu_vv = mu_rv.clone()
    mu_vv.name = "mu_vv"
    sigma_vv = sigma_rv.clone()
    sigma_vv.name = "sigma_vv"
    Y_at = at.scalar(name="Y_at")

    rvs_to_values = {mu_rv: mu_vv, sigma_rv: sigma_vv, Y_rv: Y_at}
    model = ModelInfo((Y_rv,), rvs_to_values, (), ())

    inverse_mass_matrix = at.as_tensor([1.0, 1.0])
    step_size = at.as_tensor(0.1)
    state_at, step_fn = nuts(srng, model, inverse_mass_matrix, step_size)

    # Make sure that the state is properly initialized
    state_fn = aesara.function((mu_vv, sigma_vv, Y_at), state_at)
    state = state_fn(1.0, 1.0, 1.0)

    position = state[0]
    assert position[0] == 1.0
    assert position[1] != 1.0  # The state is in the transformed space
    logprob = joint_logprob(rvs_to_values)
    assert state[1] == -1 * logprob.eval({Y_at: 1.0, mu_vv: 1.0, sigma_vv: 1.0})

    # Make sure that the step function updates the state
    (new_state, params, transformed_params), updates = step_fn(state_at)
    update_fn = aesara.function(
        (mu_vv, sigma_vv, Y_at),
        (
            params[mu_vv],
            params[sigma_vv],
            transformed_params[mu_vv],
            transformed_params[sigma_vv],
        ),
        updates=updates,
    )

    new_position = update_fn(1.0, 1.0, 1.0)
    untransformed_position = np.array([new_position[0], new_position[1]])
    transformed_position = np.array([new_position[2], new_position[3]])

    # Did the chain advance?
    np.testing.assert_raises(
        AssertionError, np.testing.assert_equal, untransformed_position, [1.0, 1.0]
    )

    # Are the transformations applied correctly?
    assert untransformed_position[0] == pytest.approx(transformed_position[0])  # mu
    assert untransformed_position[1] != transformed_position[1]  # sigma
