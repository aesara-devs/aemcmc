import aesara
import aesara.tensor as at
from aesara.tensor.random import RandomStream

from aemcmc.nuts import construct_nuts_sampler


def test_nuts():
    srng = RandomStream(0)
    mu_rv = srng.normal(0, 1, name="mu")
    sigma_rv = srng.halfnormal(0.0, 1.0, name="sigma")
    Y_rv = srng.normal(mu_rv, sigma_rv, name="Y")

    mu_vv = mu_rv.clone()
    mu_vv.name = "mu_vv"
    sigma_vv = sigma_rv.clone()
    sigma_vv.name = "sigma_vv"
    y_vv = Y_rv.clone()

    to_sample_rvs = [mu_rv, sigma_rv]
    rvs_to_values = {mu_rv: mu_vv, sigma_rv: sigma_vv, Y_rv: y_vv}

    inverse_mass_matrix = at.as_tensor([1.0, 1.0])
    step_size = at.as_tensor(0.1)
    state_at, step_fn = construct_nuts_sampler(
        srng, to_sample_rvs, rvs_to_values, inverse_mass_matrix, step_size
    )

    # Make sure that the state is properly initialized
    sample_steps = [state_at[rv] for rv in to_sample_rvs]
    state_fn = aesara.function((mu_vv, sigma_vv, y_vv), sample_steps)
    new_state = state_fn(1.0, 1.0, 1.0)

    # Make sure that the state has advanced
    assert new_state[0] != 1.0
    assert new_state[1] != 1.0
