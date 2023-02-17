import aesara
import aesara.tensor as at
import aesara.tensor.random as ar

from aemcmc.utils import get_rv_updates


def sample_prior(
    srng: ar.RandomStream, num_samples: at.TensorVariable, *rvs: at.TensorVariable
) -> at.TensorVariable:
    """Sample from a model's prior distributions.

    Parameters
    ----------
    srng:
        `RandomStream` instance with which the model was defined.
    num_samples:
        The number of prior samples to generate.
    rvs:
        The random variables whose prior distribution we want to sample.

    """

    rv_updates = get_rv_updates(srng, *rvs)

    def step_fn():
        return rvs, rv_updates

    samples, updates = aesara.scan(step_fn, n_steps=num_samples)

    return samples, updates
