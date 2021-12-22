import aesara
import aesara.tensor as at
from aesara.ifelse import ifelse
from aesara.tensor.var import TensorVariable

from aemcmc.dists import (
    multivariate_normal_cong2017,
    multivariate_normal_rue2005,
    polyagamma,
)


def update_beta_low_dimension(rng, omega, lambda2tau2_inv, X, z):
    Q = X.T @ (omega[:, None] * X)
    indices = at.arange(Q.shape[1])
    Q = at.subtensor.set_subtensor(
        Q[indices, indices],
        at.diag(Q) + lambda2tau2_inv,
    )
    return multivariate_normal_rue2005(rng, X.T @ (omega * z), Q)


def update_beta_high_dimension(rng, omega, lambda2tau2_inv, X, z):
    return multivariate_normal_cong2017(rng, lambda2tau2_inv, omega, X, z)


def update_beta(rng, omega, lambda2tau2_inv, X, z):
    return ifelse(
        X.shape[1] > X.shape[0],
        update_beta_high_dimension(rng, omega, lambda2tau2_inv, X, z),
        update_beta_low_dimension(rng, omega, lambda2tau2_inv, X, z),
    )


# NOTE: invGamma(1, a) === 1 / Exp(a), where a is the scale/mean of the exponential
def horseshoe_step(srng, beta, sigma2, lambda2_inv, tau2_inv):
    upsilon_inv = srng.exponential(1 + lambda2_inv)
    zeta_inv = srng.exponential(1 + tau2_inv)

    beta2 = beta * beta
    lambda2_inv_new = srng.exponential(upsilon_inv + 0.5 * beta2 * tau2_inv / sigma2)
    tau2_inv_new = srng.gamma(
        0.5 * (beta.shape[0] + 1),
        zeta_inv + 0.5 * (beta2 * lambda2_inv_new).sum() / sigma2,
    )
    return lambda2_inv_new, tau2_inv_new, upsilon_inv, zeta_inv


def horseshoe_nbinom_gibbs(
    srng: at.random.RandomStream,
    X: TensorVariable,
    y: TensorVariable,
    h: TensorVariable,
    beta_init: TensorVariable,
    local_shrinkage_init: TensorVariable,
    global_shrinkage_init: TensorVariable,
    n_samples: TensorVariable,
):
    r"""
    Compose a symbolic graph that describes the gibbs sampler of the negative
    binomial regression with a HorseShoe prior on the regression coefficients.

    The implemenation follows the sampler desribed in [1].

    Parameters
    ----------
    srng: symbolic random number generator
        The random number generating object to be used during sampling.
    X: TensorVariable
        The covariate matrix.
    y: TensorVariable
        The observed count data.
    h: TensorVariable
        The "number of successes" parameter of the negative binomial disribution
        used to model the data.
    beta_init: TensorVariable
        A tensor describing the starting values of the posterior samples of
        the regression coefficients.
    local_shrinkage_init: TensorVariable
        A tensor describing the starting values of the local shrinkage
        parameter of the horseshoe prior.
    global_shrinkage_init: TensorVariable
        A tensor describing the starting values of the global shrinkage
        parameter of the horseshoe prior.
    n_samples: TensorVariable
        A tensor describing the number of posterior samples to generate.

    Returns
    -------
    (outputs, updates): tuple
        A symbolic sescription of the sampling result to be used when
        compiling a function to be used for sampling.

    Notes
    -----
    [2] mention that the intercept term should not be subject to any
    regularization and thus needs to be explicitely modelled. As recommented by
    the authors, we use a uniform prior on the intercept of the regression
    coefficients and its posterior conditional distribution is shown in
    equation (14) of [2]. The ``z`` expression in section 2.2 of [1] seems to
    omit division by the Polya-Gamma auxilliary variables whereas [2] and [3]
    explicitely include it. We found that including the division results in
    accurate posterior samples for the regression coefficients. It is also
    worth noting that the :math:`\sigma^2` parameter is not sampled directly
    in the negative binomial regression problem and thus set to 1 [2].

    References
    ----------
    ..[1] Makalic, Enes & Schmidt, Daniel. (2015). A Simple Sampler for the
          Horseshoe Estimator. 10.1109/LSP.2015.2503725.
    ..[2] Makalic, Enes & Schmidt, Daniel. (2016). High-Dimensional Bayesian
          Regularised Regression with the BayesReg Package.
    ..[3] Neelon, Brian. (2019). Bayesian Zero-Inflated Negative Binomial
          Regression Based on Pólya-Gamma Mixtures. Bayesian Anal.
          2019 September ; 14(3): 829–855. doi:10.1214/18-ba1132.
    """
    beta = at.as_tensor_variable(beta_init, "beta", dtype=aesara.config.floatX)
    if beta.ndim != 1:
        raise ValueError(
            "`beta_init` must be a vector whose length is equal to the "
            "number of columns in the covariate matrix"
        )

    lambda2 = at.as_tensor_variable(
        local_shrinkage_init, name="lambda2", dtype=aesara.config.floatX
    )
    if lambda2.ndim != 1:
        raise ValueError(
            "The local shrinkage initial value must be a vector whose size "
            "is equal to the number of columns in the covariate matrix"
        )

    tau2 = at.as_tensor_variable(
        global_shrinkage_init, name="tau2", dtype=aesara.config.floatX
    )
    if tau2.ndim != 0:
        raise ValueError("The global shrinkage initial value must be a scalar")

    n_samples = at.as_tensor_variable(n_samples, name="n_samples", dtype=int)
    if n_samples.ndim != 0:
        raise ValueError("The number of samples should be an integer scalar")

    # sigma2 must be set to 1 during sampling for this sampler and the
    # logistic regression version. It is only required if the data is
    # modelled using linear regression.
    sigma2 = at.constant(1, "sigma2")

    # set the initial value of the intercept term to a random uniform value.
    beta0 = at.empty_like(tau2)
    beta0.name = "intercept"
    beta0 = srng.uniform(-10, 10)

    def step_fn(
        beta0: TensorVariable,
        beta: TensorVariable,
        lambda2_inv: TensorVariable,
        tau2_inv: TensorVariable,
        sigma2: TensorVariable,
        X: TensorVariable,
        y: TensorVariable,
        h: TensorVariable,
    ):
        """
        Complete one full update of the gibbs sampler and return the new state
        of the posterior conditional parameters.
        """
        xb = X @ beta
        w = srng.gen(polyagamma, y + h, beta0 + xb)

        z = 0.5 * (y - h) / w
        beta0_var = 1 / w.sum()
        beta0_mean = beta0_var * (w @ (z - xb))
        beta0_new = srng.normal(beta0_mean, at.sqrt(beta0_var))

        beta_new = update_beta(srng, w, lambda2_inv * tau2_inv, X, z - beta0_new)

        lambda2_inv_new, tau2_inv_new, _, _ = horseshoe_step(
            srng, beta_new, sigma2, lambda2_inv, tau2_inv
        )

        return beta0_new, beta_new, lambda2_inv_new, tau2_inv_new

    outputs, updates = aesara.scan(
        step_fn,
        outputs_info=[beta0, beta, 1 / lambda2, 1 / tau2],
        non_sequences=[sigma2, X, y, h],
        n_steps=n_samples,
        strict=True,
    )
    return outputs, updates
