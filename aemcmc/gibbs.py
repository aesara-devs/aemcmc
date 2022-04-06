from typing import Callable, Dict, List, NamedTuple, Tuple

import aesara
import aesara.tensor as at
from aesara.graph.unify import eval_if_etuple
from aesara.ifelse import ifelse
from aesara.tensor.random import RandomStream
from aesara.tensor.var import TensorVariable, Variable
from etuples import etuple, etuplize
from etuples.core import ExpressionTuple
from unification import unify, var

from aemcmc.dists import (
    multivariate_normal_cong2017,
    multivariate_normal_rue2005,
    polyagamma,
)


class Sampler(NamedTuple):
    pattern: ExpressionTuple
    match_fn: Callable
    step_fn: Callable


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


halfcauchy_1_lv, halfcauchy_2_lv = var(), var()
zero_lv = var()
horsehoe_pattern = etuple(
    etuplize(at.random.normal),
    var(),
    var(),
    var(),
    zero_lv,
    etuple(etuplize(at.mul), halfcauchy_1_lv, halfcauchy_2_lv),
)


def horseshoe_match(graph):
    graph_et = etuplize(graph)

    s = unify(graph_et, horsehoe_pattern)
    if s is False:
        raise ValueError("Not a horseshoe prior")

    halfcauchy_1 = eval_if_etuple(s[halfcauchy_1_lv])

    if halfcauchy_1.owner is None or not isinstance(
        halfcauchy_1.owner.op, type(at.random.halfcauchy)
    ):
        raise ValueError("Input graph does not match")

    halfcauchy_2 = eval_if_etuple(s[halfcauchy_2_lv])

    if halfcauchy_2.owner is None or not isinstance(
        halfcauchy_2.owner.op, type(at.random.halfcauchy)
    ):
        raise ValueError("Input graph does not match")

    if halfcauchy_1.type.shape == (1,):
        lambda_rv = halfcauchy_2
        tau_rv = halfcauchy_1
    elif halfcauchy_2.type.shape == (1,):
        lambda_rv = halfcauchy_1
        tau_rv = halfcauchy_2
    else:
        raise ValueError("Input graph does not match")

    return (lambda_rv, tau_rv)


def horseshoe_step(
    srng: RandomStream,
    beta: TensorVariable,
    sigma2: TensorVariable,
    lambda2_inv: TensorVariable,
    tau2_inv: TensorVariable,
) -> Tuple[TensorVariable, TensorVariable]:
    r"""Gibbs kernel to sample from the posterior distribution of a horsehoe prior.

    This kernel generates samples from the posterior distribution of the local
    and global shrinkage parameters of a horsehoe prior, respectively :math:`\lambda`
    and :math:`\tau` in the following model:

    .. math::

        \begin{align*}
            \beta_j &\sim \operatorname{Normal}(0, \lambda_j^2\;\tau^2\;\sigma^2)\\
            \sigma^2 &\sim \sigma^{-2} \mathrm{d} \sigma\\
            \lambda_j &\sim \operatorname{HalfCauchy}(0, 1)\\
            \tau &\sim \operatorname{HalfCauchy}(0, 1)
        \end{align*}

    We use the following observations [1]_ to sample from the posterior
    conditional probability of :math:`\tau^{-2}` and :math:`\lambda^{-2}`:

    1. The half-Cauchy distribution can be intepreted as a mixture of inverse-gamma
    distributions;
    2. If :math:` Y \sim InverseGamma(1, a)`, :math:`Y \sim 1 / \operatorname{Exp}(a)`.

    Parameters
    ----------
    srng
        The random number generating object to be used during sampling.
    beta
        Regression coefficients.
    sigma2
        Variance of the regression coefficients.
    lambda2_inv
        Square inverse of the local shrinkage parameters.
    tau2_inv
        Square inverse of the global shrinkage parameters.

    References
    ----------
    ..[1] Makalic, Enes & Schmidt, Daniel. (2016). High-Dimensional Bayesian
          Regularised Regression with the BayesReg Package.
    """
    upsilon_inv = srng.exponential(1 + lambda2_inv)
    zeta_inv = srng.exponential(1 + tau2_inv)

    beta2 = beta * beta
    lambda2_inv_new = srng.exponential(upsilon_inv + 0.5 * beta2 * tau2_inv / sigma2)
    tau2_inv_new = srng.gamma(
        0.5 * (beta.shape[0] + 1),
        zeta_inv + 0.5 * (beta2 * lambda2_inv_new).sum() / sigma2,
    )
    return lambda2_inv_new, tau2_inv_new


horseshoe = Sampler(horsehoe_pattern, horseshoe_match, horseshoe_step)


def horseshoe_nbinom(
    srng: at.random.RandomStream,
    X: TensorVariable,
    y: TensorVariable,
    h: TensorVariable,
    beta_init: TensorVariable,
    local_shrinkage_init: TensorVariable,
    global_shrinkage_init: TensorVariable,
    n_samples: TensorVariable,
) -> Tuple[List[Variable], Dict]:
    r"""
    Build a symbolic graph that describes the gibbs sampler of the negative
    binomial regression with a HorseShoe prior on the regression coefficients.

    The implementation follows the sampler described in [1]. It is designed to
    sample efficiently from the following negative binomial regression model:

    .. math::

        \begin{align*}
            y_i &\sim \operatorname{NegativeBinomial}\left(\pi_i, h\right)\\
            h &\sim \pi_h(h) \mathrm{d}h\\
            \pi_i &= \frac{\exp(\psi_i)}{1 + \exp(\psi_i)}\\
            \psi_i &= x^T \beta\\
            \beta_j &\sim \operatorname{Normal}(0, \lambda_j^2\;\tau^2\;\sigma^2)\\
            \sigma^2 &\sim \sigma^{-2} \mathrm{d} \sigma\\
            \lambda_j &\sim \operatorname{HalfCauchy}(0, 1)\\
            \tau &\sim \operatorname{HalfCauchy}(0, 1)
        \end{align*}


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
    # TODO: Set the prior value of the intercept outside of the function
    beta0 = srng.uniform(-10, 10)
    beta0.name = "intercept"

    def step_fn(
        beta0: TensorVariable,
        beta: TensorVariable,
        lambda2_inv: TensorVariable,
        tau2_inv: TensorVariable,
        sigma2: TensorVariable,
        X: TensorVariable,
        y: TensorVariable,
        h: TensorVariable,
    ) -> Tuple[TensorVariable, TensorVariable, TensorVariable, TensorVariable]:
        """
        Complete one full update of the gibbs sampler and return the new state
        of the posterior conditional parameters.

        Parameters
        ----------
        beta0: TensorVariable
            The intercept coefficient of the regression model.
        beta: Tensorvariable
            Coefficients (other than intercept) of the regression model.
        lambda2_inv
            Square inverse of the local shrinkage parameter of the horseshoe prior.
        tau2_inv
            Square inverse of the global shrinkage parameters of the horseshoe prior.
        sigma2
            Variance of the regression coefficients.
        X: TensorVariable
            The covariate matrix.
        y: TensorVariable
            The observed count data.
        h: TensorVariable
            The "number of successes" parameter of the negative binomial disribution
            used to model the data.
        """
        xb = X @ beta
        w = srng.gen(polyagamma, y + h, beta0 + xb)

        z = 0.5 * (y - h) / w
        beta0_var = 1 / w.sum()
        beta0_mean = beta0_var * (w @ (z - xb))
        beta0_new = srng.normal(beta0_mean, at.sqrt(beta0_var))

        beta_new = update_beta(srng, w, lambda2_inv * tau2_inv, X, z - beta0_new)

        lambda2_inv_new, tau2_inv_new = horseshoe_step(
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


def horseshoe_logistic(
    srng: RandomStream,
    X: TensorVariable,
    y: TensorVariable,
    beta_init: TensorVariable,
    local_shrinkage_init: TensorVariable,
    global_shrinkage_init: TensorVariable,
    n_samples: TensorVariable,
) -> Tuple[List[Variable], Dict]:
    r"""

    Build a symbolic graph that describes the gibbs sampler of the binary
    logistic regression with a Horseshoe prior on the regression coefficients.

    The implementation follows the sampler described in [1]. It is designed to
    sample efficiently from the following binary logistic regression model:

    .. math::

        \begin{align*}
            y_i &\sim \operatorname{Benoulli}\left(\pi_i\right)\\
            \pi &= \frac{1}{1 + \exp\left(-(\beta_0 + x_i^T\,\beta)\right)}\\
            \beta_j &\sim \operatorname{Normal}(0, \lambda_j^2\;\tau^2\;\sigma^2)\\
            \sigma^2 &\sim \sigma^{-2} \mathrm{d} \sigma\\
            \lambda_j^2 &\sim \operatorname{HalfCauchy}(0, 1)\\
            \tau &\sim \operatorname{HalfCauchy}(0, 1)
        \end{align*}

    Parameters
    ----------
    srng: symbolic random number generator
        The random number generating object to be used during sampling.
    X: TensorVariable
        The covariate matrix.
    y: TensorVariable
        The observed count data.
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

    References
    ----------
    ..[1] Makalic, Enes & Schmidt, Daniel. (2015). A Simple Sampler for the
          Horseshoe Estimator. 10.1109/LSP.2015.2503725.
    ..[2] Makalic, Enes & Schmidt, Daniel. (2016). High-Dimensional Bayesian
          Regularised Regression with the BayesReg Package.

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

    # sigma2 must be set to 1 during sampling for this sampler and the negative
    # binomial regression version. It is only required if the data is modelled
    # using linear regression.
    sigma2 = at.constant(1, "sigma2")

    # set the initial value of the intercept term to a random uniform value.
    # TODO: Set the prior value of the intercept outside of the function
    beta0 = srng.uniform(-10, 10)
    beta0.name = "intercept"

    def step_fn(
        beta0: TensorVariable,
        beta: TensorVariable,
        lambda2_inv: TensorVariable,
        tau2_inv: TensorVariable,
        sigma2: TensorVariable,
        X: TensorVariable,
        y: TensorVariable,
    ) -> Tuple[TensorVariable, TensorVariable, TensorVariable, TensorVariable]:
        """
        Complete one full update of the gibbs sampler and return the new state
        of the posterior conditional parameters.

        Parameters
        ----------
        beta0: TensorVariable
            The intercept coefficient of the regression model.
        beta: Tensorvariable
            Coefficients (other than intercept) of the regression model.
        lambda2_inv
            Square inverse of the local shrinkage parameter of the horseshoe prior.
        tau2_inv
            Square inverse of the global shrinkage parameters of the horseshoe prior.
        X: TensorVariable
            The covariate matrix.
        y: TensorVariable
            The observed binary data
        """

        xb = X @ beta
        w = srng.gen(polyagamma, 1, beta0 + xb)

        z = (y - 0.5) / w
        beta0_var = 1 / w.sum()
        beta0_mean = beta0_var * (w @ (z - xb))
        beta0_new = srng.normal(beta0_mean, at.sqrt(beta0_var))

        beta_new = update_beta(srng, w, lambda2_inv * tau2_inv, X, z - beta0_new)

        lambda2_inv_new, tau2_inv_new = horseshoe_step(
            srng, beta_new, sigma2, lambda2_inv, tau2_inv
        )

        return beta0_new, beta_new, lambda2_inv_new, tau2_inv_new

    outputs, updates = aesara.scan(
        step_fn,
        outputs_info=[beta0, beta, 1 / lambda2, 1 / tau2],
        non_sequences=[sigma2, X, y],
        n_steps=n_samples,
        strict=True,
    )
    return outputs, updates
