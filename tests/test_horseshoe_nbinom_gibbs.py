import aesara
import aesara.tensor as at
import numpy as np
import pytest
from aesara.tensor.random.utils import RandomStream
from scipy.linalg import toeplitz

from aemcmc.negative_binomial_horseshoe_gibbs import horseshoe_nbinom_gibbs


def test_horseshoe_nbinom_gibbs():
    """
    This test example is modified from section 3.2 of Makalic & Schmidt (2016)
    """
    h = 2
    p = 10
    N = 50
    srng = RandomStream(seed=12345)
    true_beta0 = srng.uniform(-1, 1)
    true_beta = np.array([5, 3, 3, 1, 1] + [0] * (p - 5))
    S = toeplitz(0.5 ** np.arange(p))
    X = srng.multivariate_normal(np.zeros(p), cov=S, size=N)
    y = srng.nbinom(h, at.sigmoid(-(X.dot(true_beta) + true_beta0)))

    beta_init, lambda2_init, tau2_init, n_samples = (
        at.vector("beta_init"),
        at.vector("lambda2_init"),
        at.scalar("tau2_init"),
        at.iscalar("n_samples"),
    )
    outputs, updates = horseshoe_nbinom_gibbs(
        srng, X, y, h, beta_init, lambda2_init, tau2_init, n_samples
    )
    sample_fn = aesara.function(
        [beta_init, lambda2_init, tau2_init, n_samples], outputs, updates=updates
    )

    rng = np.random.default_rng(54321)
    posterior_samples = 2000
    beta0, beta, lambda2_inv, tau2_inv = sample_fn(
        rng.normal(0, 5, size=p), np.ones(p), 1, posterior_samples
    )

    assert beta0.shape == (posterior_samples,)
    assert beta.shape == (posterior_samples, p)
    assert lambda2_inv.shape == (posterior_samples, p)
    assert tau2_inv.shape == (posterior_samples,)

    # test distribution domains
    assert np.all(tau2_inv > 0)
    assert np.all(lambda2_inv > 0)

    # test if the sampler fails with wrong input
    with pytest.raises(ValueError, match="`beta_init` must be a vector "):
        wrong_beta_init = at.tensor3("wrong_beta_init")
        horseshoe_nbinom_gibbs(
            srng, X, y, h, wrong_beta_init, lambda2_init, tau2_init, n_samples
        )

    with pytest.raises(
        ValueError, match="The local shrinkage initial value must be a vector"
    ):
        wrong_lambda2_init = at.scalar("wrong_lambda2_init")
        horseshoe_nbinom_gibbs(
            srng, X, y, h, beta_init, wrong_lambda2_init, tau2_init, n_samples
        )

    with pytest.raises(
        ValueError, match="The global shrinkage initial value must be a scalar"
    ):
        wrong_tau2_init = at.matrix("wrong_tau2_init")
        horseshoe_nbinom_gibbs(
            srng, X, y, h, beta_init, lambda2_init, wrong_tau2_init, n_samples
        )

    with pytest.raises(
        ValueError, match="The number of samples should be an integer scalar"
    ):
        wrong_n_samples = at.vector("wrong_n_samples")
        horseshoe_nbinom_gibbs(
            srng, X, y, h, beta_init, lambda2_init, tau2_init, wrong_n_samples
        )
