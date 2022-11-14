from typing import List, Mapping, Optional, Tuple

import aesara
import aesara.tensor as at
from aesara.graph.basic import Variable
from aesara.graph.rewriting.basic import in2out
from aesara.graph.rewriting.db import LocalGroupDB
from aesara.graph.rewriting.unify import eval_if_etuple
from aesara.graph.type import Constant
from aesara.ifelse import ifelse
from aesara.tensor.math import Dot
from aesara.tensor.random import RandomStream
from aesara.tensor.random.basic import BernoulliRV, NegBinomialRV, NormalRV
from aesara.tensor.var import TensorVariable
from etuples import etuple, etuplize
from unification import unify, var

from aemcmc.dists import (
    multivariate_normal_cong2017,
    multivariate_normal_rue2005,
    polyagamma,
)
from aemcmc.rewriting import sampler_finder, sampler_finder_db
from aemcmc.types import SamplingStep

gibbs_db = LocalGroupDB(apply_all_rewrites=True)
gibbs_db.name = "gibbs_db"


def remove_constants(inputs):
    inputs_at = [at.as_tensor_variable(x) for x in inputs]
    return [x for x in inputs_at if not isinstance(x, Constant)]


def normal_regression_overdetermined_posterior(
    srng: RandomStream,
    omega: TensorVariable,
    lmbdatau_inv: TensorVariable,
    X: TensorVariable,
    z: TensorVariable,
) -> TensorVariable:
    """Sample from the posterior of a normal prior and normal observation model.

    This version handles ``X.shape[1] <= X.shape[0]``.

    See `update_beta` for a description of the parameters and return value.

    """
    Q = X.T @ (omega[:, None] * X)
    indices = at.arange(Q.shape[1])
    Q = at.subtensor.set_subtensor(
        Q[indices, indices],
        at.diag(Q) + lmbdatau_inv,
    )
    return multivariate_normal_rue2005(srng, X.T @ (omega * z), Q)


def normal_regression_underdetermined_posterior(
    srng: RandomStream,
    omega: TensorVariable,
    lmbdatau_inv: TensorVariable,
    X: TensorVariable,
    z: TensorVariable,
) -> TensorVariable:
    """Sample from the posterior of a normal prior and normal observation model.

    This version handles ``X.shape[1] > X.shape[0]``.

    See `update_beta` for a description of the parameters and return value.

    """
    return multivariate_normal_cong2017(srng, lmbdatau_inv, omega, X, z)


def normal_regression_posterior(
    srng: RandomStream,
    omega: TensorVariable,
    lmbdatau_inv: TensorVariable,
    X: TensorVariable,
    z: TensorVariable,
) -> TensorVariable:
    r"""Sample from the posterior of a normal prior and normal observation model.

    .. math::

        \begin{align*}
            Z &\sim \operatorname{N}\left( X \beta, \Omega^{-1} \right) \\
            \beta &\sim \operatorname{N}\left( 0, \tau^2 \Lambda \right)
        \end{align*}

    where :math:`X \in \mathbb{R}^{n \times p}`,
    :math:`\Lambda = \operatorname{diag}\left(\lambda^2_1, \dots, \lambda^2_p)`, and
    :math:`\Omega^{-1} = \operatorname{diag}\left(\omega_1, \dots, \omega_n\right)`.

    The posterior distribution of :math:`\beta` is given by

    .. math::

        \begin{align*}
            \left( \beta \mid Z = z \right) &\sim
                \operatorname{N}\left( A^{-1} X^{\top} \Omega z, A^{-1} \right) \\
            A &= X^{\top} X + \Lambda^{-1}_{*} \\
            \Lambda_{*} &= \tau^2 \Lambda
        \end{align*}

    This function chooses the best sampler for :math:`\beta \mid z` based on
    the dimensions of :math:`X`.

    Parameters
    ----------
    srng
        The random number generator used to draw samples.
    omega
        The observation model diagonal std. dev. values :math:`\omega_i`.
        In other words, :math:`\operatorname{diag}\left(\Omega\right)`.
    lmbdatau_inv
        The inverse :math:`beta` std. dev. values :math:`\tau^{-1} \lambda^{-1}_i`.
        In other words, :math:`\operatorname{diag}\left(\Lambda^{-1/2}_{*}\right)`.
    X
        Regression matrix :math:`X`.
    z
        Observed values :math:`z \sim Z`.


    Returns
    -------
    A sample from :math:`\beta \mid z`.

    """
    beta_posterior = ifelse(
        X.shape[1] > X.shape[0],
        normal_regression_underdetermined_posterior(srng, omega, lmbdatau_inv, X, z),
        normal_regression_overdetermined_posterior(srng, omega, lmbdatau_inv, X, z),
    )

    return beta_posterior


halfcauchy_1_lv, halfcauchy_2_lv = var(), var()
zero_lv = var()
horseshoe_pattern = etuple(
    etuplize(at.random.normal),
    var(),
    var(),
    var(),
    zero_lv,
    etuple(etuplize(at.mul), halfcauchy_1_lv, halfcauchy_2_lv),
)


def horseshoe_match(graph: TensorVariable) -> Tuple[TensorVariable, TensorVariable]:

    graph_et = etuplize(graph)

    s = unify(graph_et, horseshoe_pattern)
    if s is False:
        raise ValueError("Not a horseshoe prior.")

    halfcauchy_1 = eval_if_etuple(s[halfcauchy_1_lv])
    if halfcauchy_1.owner is None or not isinstance(
        halfcauchy_1.owner.op, type(at.random.halfcauchy)
    ):
        raise ValueError(
            "Not a horseshoe prior. One of the shrinkage parameters "
            + "in your model is not half-Cauchy distributed."
        )

    halfcauchy_2 = eval_if_etuple(s[halfcauchy_2_lv])

    if halfcauchy_2.owner is None or not isinstance(
        halfcauchy_2.owner.op, type(at.random.halfcauchy)
    ):
        raise ValueError(
            "Not a horseshoe prior. One of the shrinkage parameters "
            + "in your model is not half-Cauchy distributed."
        )

    if halfcauchy_1.type.ndim == 0 or all(s == 1 for s in halfcauchy_1.type.shape):
        lmbda_rv = halfcauchy_2
        tau_rv = halfcauchy_1
    elif halfcauchy_2.type.ndim == 0 or all(s == 1 for s in halfcauchy_2.type.shape):
        lmbda_rv = halfcauchy_1
        tau_rv = halfcauchy_2
    else:
        raise ValueError(
            "Not a horseshoe prior. The global shrinkage parameter "
            + "in your model must be one-dimensional."
        )

    return (lmbda_rv, tau_rv)


def horseshoe_posterior(
    srng: RandomStream,
    beta: TensorVariable,
    sigma2: TensorVariable,
    lambda2: TensorVariable,
    tau2: TensorVariable,
) -> Tuple[TensorVariable, TensorVariable]:
    r"""Gibbs kernel to sample from the posterior distributions of the horseshoe prior shrinkage parameters.

    This kernel generates samples from the posterior distribution of the local
    and global shrinkage parameters of a horseshoe prior, respectively :math:`\lambda`
    and :math:`\tau` in the following model:

    .. math::

        \begin{align*}
            \beta_j &\sim \operatorname{N}\left(0, \lambda_j^2 \tau^2 \sigma^2\right) \\
            \lambda_j &\sim \operatorname{C}^{+}(0, 1) \\
            \tau &\sim \operatorname{C}^{+}(0, 1)
        \end{align*}

    The graphs constructed by this function are :math:`\lambda \mid \beta, \tau` and
    :math:`\tau \mid \lambda`, respectively.

    We use the following observations [1]_ to sample from the posterior
    conditional probability of :math:`\tau` and :math:`\lambda`:

    1. The half-Cauchy distribution can be intepreted as a mixture of inverse-gamma
    distributions;
    2. If :math:`Z \sim \operatorname{InvGamma}(1, a)`, :math:`Z \sim 1 / \operatorname{Exp}(a)`.

    Parameters
    ----------
    srng
        The random number generating object to be used during sampling.
    beta
        Regression coefficients.
    sigma2
        Variance of the regression coefficients.
    lambda2
        Square of the local shrinkage parameters.
    tau2
        Square of the global shrinkage parameters.

    Returns
    -------
    Posteriors for :math:`lambda` and :math:`tau`, respectively.

    References
    ----------
    .. [1] Makalic, Enes & Schmidt, Daniel. (2016). High-Dimensional Bayesian
          Regularised Regression with the BayesReg Package.

    """
    lmbda2_inv = at.reciprocal(lambda2)
    tau2_inv = at.reciprocal(tau2)

    upsilon_inv = srng.exponential(1 + lmbda2_inv)
    zeta_inv = srng.exponential(1 + tau2_inv)

    beta2 = beta**2
    lambda2_inv_new = srng.exponential(upsilon_inv + 0.5 * beta2 * tau2_inv / sigma2)
    tau2_inv_new = srng.gamma(
        0.5 * (beta.shape[0] + 1),
        zeta_inv + 0.5 * (beta2 * lambda2_inv_new).sum() / sigma2,
    )

    lambda2_update = at.reciprocal(at.sqrt(lambda2_inv_new))
    tau2_update = at.reciprocal(at.sqrt(tau2_inv_new))

    return lambda2_update, tau2_update


class HorseshoeGibbsKernel(SamplingStep):
    """An `Op` that represents a state update with the FFBS sampler."""


@sampler_finder([NormalRV])
def normal_horseshoe_finder(fgraph, node, srng):
    r"""Find and construct a Gibbs sampler for the normal-Horseshoe model.

    The implementation follows the sampler described in [1]_. It is designed to
    sample efficiently from the following model:

    .. math::

        \begin{align*}
            \beta_i &\sim \operatorname{N}(0, \lambda_i^2 \tau^2) \\
            \lambda_i &\sim \operatorname{C}^{+}(0, 1) \\
            \tau &\sim \operatorname{C}^{+}(0, 1)
        \end{align*}

    References
    ----------
    .. [1] Makalic, Enes & Schmidt, Daniel. (2015). A Simple Sampler for the
          Horseshoe Estimator. 10.1109/LSP.2015.2503725.

    """

    rv_var = node.outputs[1]

    try:
        lambda_rv, tau_rv = horseshoe_match(node.outputs[1])
    except ValueError:  # pragma: no cover
        return None

    tau2 = tau_rv**2
    lambda2 = lambda_rv**2
    lambda_posterior, tau_posterior = horseshoe_posterior(
        srng, rv_var, at.as_tensor(1.0), lambda2, tau2
    )

    # Build an `Op` for the sampling kernel
    outputs = [lambda_posterior, tau_posterior]
    inputs = remove_constants([rv_var, lambda2, tau2])
    gibbs = HorseshoeGibbsKernel(inputs, outputs)

    lambda_posterior, tau_posterior = gibbs(*inputs)
    lambda_posterior.name = f"{lambda_rv.name or 'lambda'}_posterior"
    tau_posterior.name = f"{tau_rv.name or 'tau'}_posterior"

    return [(lambda_rv, lambda_posterior, None), (tau_rv, tau_posterior, None)]


gibbs_db.register("normal_horseshoe", normal_horseshoe_finder, "basic")


X_lv = var()
beta_lv = var()
neg_one_lv = var()

sigmoid_dot_pattern = etuple(
    etuplize(at.sigmoid),
    etuple(etuplize(at.mul), neg_one_lv, etuple(etuple(Dot), X_lv, beta_lv)),
)

a_lv = var()
b_lv = var()
gamma_pattern = etuple(etuplize(at.random.gamma), var(), var(), var(), a_lv, b_lv)


def gamma_match(graph: TensorVariable) -> Tuple[TensorVariable, TensorVariable]:

    graph_et = etuplize(graph)
    s = unify(graph_et, gamma_pattern)
    if s is False:
        raise ValueError("Not a gamma prior.")

    a = eval_if_etuple(s[a_lv])
    b = eval_if_etuple(s[b_lv])

    return a, b


h_lv = var()
nbinom_sigmoid_dot_pattern = etuple(
    etuplize(at.random.nbinom), var(), var(), var(), h_lv, sigmoid_dot_pattern
)


def nbinom_sigmoid_dot_match(
    graph: TensorVariable,
) -> Tuple[TensorVariable, TensorVariable, TensorVariable]:

    graph_et = etuplize(graph)
    s = unify(graph_et, nbinom_sigmoid_dot_pattern)
    if s is False:
        raise ValueError("Not a negative binomial regression.")

    if all(s[neg_one_lv].data != -1):
        raise ValueError(
            "Not a negative binomial regression. The argument to "
            + "the sigmoid must be minus the dot product."
        )

    h = eval_if_etuple(s[h_lv])
    beta_rv = eval_if_etuple(s[beta_lv])
    X = eval_if_etuple(s[X_lv])

    return X, h, beta_rv


def sample_CRT(
    srng: RandomStream, y: TensorVariable, h: TensorVariable
) -> Tuple[TensorVariable, Mapping[Variable, Variable]]:
    r"""Sample a Chinese Restaurant Process value: :math:`l \sim \operatorname{CRT}(y, h)`.

    Sampling is performed according to the following:

    .. math::

        \begin{gather*}
            l = \sum_{n=1}^{y} b_n, \quad
            b_n \sim \operatorname{Bern}\left(\frac{h}{n - 1 + h}\right)
        \end{gather*}

    References
    ----------
    .. [1] Zhou, Mingyuan, and Lawrence Carin. 2012. “Augment-and-Conquer Negative Binomial Processes.” Advances in Neural Information Processing Systems 25.

    """

    def single_sample_CRT(y_i: TensorVariable, h: TensorVariable):
        n_i = at.arange(2, y_i + 1)
        return at.switch(y_i < 1, 0, 1 + srng.bernoulli(h / (n_i - 1 + h)).sum())

    res, updates = aesara.scan(
        single_sample_CRT,
        sequences=[y.ravel()],
        non_sequences=[h],
        strict=True,
    )
    res = res.reshape(y.shape)
    res.name = "CRT sample"

    return res, updates


def nbinom_dispersion_posterior(
    srng: RandomStream,
    h: TensorVariable,
    p: TensorVariable,
    a: TensorVariable,
    b: TensorVariable,
    y: TensorVariable,
) -> Tuple[TensorVariable, Mapping[Variable, Variable]]:
    r"""Sample the conditional posterior for the dispersion parameter under a negative-binomial and gamma prior.

    In other words, this draws a sample from :math:`h \mid Y = y` per

    .. math::

        \begin{align*}
            Y_i &\sim \operatorname{NB}(h, p_i) \\
            h &\sim \operatorname{Gamma}(a, b)
        \end{align*}

    where :math:`\operatorname{NB}` is a negative-binomial distribution.

    The conditional posterior sample step is derived from the following decomposition:

    .. math::
        \begin{gather*}
            Y_i = \sum_{j=1}^{l_i} u_{i j}, \quad u_{i j} \sim \operatorname{Log}(p_i), \quad
            l_i \sim \operatorname{Pois}\left(-h \log(1 - p_i)\right)
        \end{gather*}

    where :math:`\operatorname{Log}` is the logarithmic distribution.  Under a
    gamma prior, :math:`h` is conjugate to :math:`l`.  We draw samples from
    :math:`l` according to :math:`l \sim \operatorname{CRT(y, h)}`, where
    :math:`y` is a sample from :math:`y \sim Y`.

    The resulting posterior is

    .. math::

        \begin{gather*}
            \left(h \mid Y = y\right) \sim \operatorname{Gamma}\left(a + \sum_{i=1}^N l_i, \frac{1}{1/b + \sum_{i=1}^N \log(1 - p_i)} \right)
        \end{gather*}

    Parameters
    ----------
    srng
        The random number generator from which samples are drawn.
    h
        The  value of :math:`h`.
    p
        The success probability parameter in the negative-binomial distribution of :math:`Y`.
    a
        The shape parameter in the :math:`\operatorname{Gamma}` prior on :math:`h`.
    b
        The rate parameter in the :math:`\operatorname{Gamma}` prior on :math:`h`.
    y
        A sample from :math:`Y`.

    Returns
    -------
    A sample from the posterior :math:`h \mid y`.

    References
    ----------
    .. [1] Zhou, Mingyuan, Lingbo Li, David Dunson, and Lawrence Carin. 2012.
        “Lognormal and Gamma Mixed Negative Binomial Regression.”
        Proceedings of the International Conference on Machine Learning.
        2012: 1343–50. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4180062/.

    """
    Ls, updates = sample_CRT(srng, y, h)
    L_sum = Ls.sum(axis=-1)
    h_posterior = srng.gamma(
        a + L_sum, at.reciprocal(b) - at.sum(at.log(1 - p), axis=-1)
    )

    return h_posterior, updates


def nbinom_normal_posterior(srng, beta, beta_std, X, h, y):
    r"""Produce a Gibbs sample step for a negative binomial logistic-link regression with a normal prior.

    The implementation follows the sampler described in [1]_. It is designed to
    sample efficiently from the following negative binomial regression model:

    .. math::

        \begin{align*}
            Y &\sim \operatorname{NB}\left(h, p\right) \\
            p &= \frac{\exp(\psi)}{1 + \exp(\psi)} \\
            \psi &= X^\top \beta \\
            \beta &\sim \operatorname{N}(0, \lambda^2) \\
        \end{align*}

    where :math:`\operatorname{NB}` is a negative-binomial distribution.

    Parameters
    ----------
    srng
        The random number generator from which samples are drawn.
    beta
        The current/previous value of the regression parameter :math:`beta`.
    beta_std
        The std. dev. of the regression parameter :math:`beta`.
    X
        The regression matrix.
    h
        The :math:`h` parameter in the negative-binomial distribution of :math:`Y`.
    y
        A sample from the observation distribution :math:`y \sim Y`.

    Returns
    -------
    A sample from the posterior :math:`\beta \mid y`.

    Notes
    -----
    The :math:`z` expression in Section 2.2 of [1]_ seems to
    omit division by the Polya-Gamma auxilliary variables whereas [2]_ and [3]_
    explicitly include it. We found that including the division results in
    accurate posterior samples for the regression coefficients. It is also
    worth noting that the :math:`\sigma^2` parameter is not sampled directly
    in the negative binomial regression problem and thus set to 1 [2]_.

    References
    ----------
    .. [1] Makalic, Enes & Schmidt, Daniel. (2015). A Simple Sampler for the
          Horseshoe Estimator. 10.1109/LSP.2015.2503725.
    .. [2] Makalic, Enes & Schmidt, Daniel. (2016). High-Dimensional Bayesian
          Regularised Regression with the BayesReg Package.
    .. [3] Neelon, Brian. (2019). Bayesian Zero-Inflated Negative Binomial
          Regression Based on Pólya-Gamma Mixtures. Bayesian Anal.
          2019 September ; 14(3): 829–855. doi:10.1214/18-ba1132.

    """

    # This effectively introduces a new term, `w`, to the model.
    # TODO: We could/should generate a graph that uses this scale-mixture
    # "expanded" form and find/create the posteriors from there
    w = srng.gen(polyagamma, y + h, X @ beta)
    z = 0.5 * (y - h) / w

    tau_beta = at.reciprocal(beta_std)

    beta_posterior = normal_regression_posterior(srng, w, tau_beta, X, z)

    return beta_posterior


class NBRegressionGibbsKernel(SamplingStep):
    """An `Op` that represents the update of the regression parameter of
    a negative binomial regression.

    """

    default_output = 0


class DispersionGibbsKernel(SamplingStep):
    """An `Op` that represents the state update for the dispersion parameter
    of a negative binomial in a negative binomial regression.

    """

    default_output = 0


@sampler_finder([NegBinomialRV])
def nbinom_logistic_finder(fgraph, node, srng):
    r"""Find and construct a Gibbs sampler for a negative-binomial logistic-link regression.

    The implementation follows the sampler described in `nbinom_normal_posterior`. It is designed to
    sample efficiently from the following negative binomial regression model:

    .. math::

        \begin{align*}
            Y &\sim \operatorname{NB}\left(h, p\right) \\
            p &= \frac{\exp(\psi)}{1 + \exp(\psi)} \\
            \psi &= X^\top \beta \\
            \beta_j &\sim \operatorname{N}(0, \lambda_j^2) \\
            h \sim \operatorname{Gamma}\left(a, b\right)
        \end{align*}

    If :math:`h` doesn't take the above form, a sampler is produced with steps
    for all the other terms; otherwise, sampling for :math:`h` is performed
    in accordance with [1]_.

    References
    ----------
    .. [1] Zhou, Mingyuan, Lingbo Li, David Dunson, and Lawrence Carin. 2012.
          Lognormal and Gamma Mixed Negative Binomial Regression.
          Proceedings of the International Conference on Machine Learning.
          2012: 1343–50. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4180062/.

    """

    y = node.outputs[1]

    try:
        X, h, beta_rv = nbinom_sigmoid_dot_match(node.outputs[1])
    except ValueError:  # pragma: no cover
        return None

    beta_std = beta_rv.owner.inputs[4]
    beta_posterior = nbinom_normal_posterior(srng, beta_rv, beta_std, X, h, y)

    # Build the `Op` corresponding to the sampling step
    outputs = [beta_posterior]
    inputs = remove_constants([beta_rv, beta_std, X, h, y])
    gibbs = NBRegressionGibbsKernel(inputs, outputs)

    beta_posterior = gibbs(*inputs)
    beta_posterior.name = f"{beta_rv.name or 'beta'}_posterior"

    res: List[
        Tuple[TensorVariable, TensorVariable, Optional[Mapping[Variable, Variable]]]
    ] = [(beta_rv, beta_posterior, None)]

    # TODO: Should this be in a separate rewriter?
    try:
        a, b = gamma_match(h)
    except ValueError:  # pragma: no cover
        return res

    p = at.sigmoid(-X @ beta_posterior)

    h_posterior, updates = nbinom_dispersion_posterior(srng, h, p, a, b, y)

    # Build the `Op` corresponding to the sampling step
    update_outputs = [h_posterior.owner.inputs[0].default_update]
    update_outputs.extend(updates.values())

    outputs = [h_posterior] + update_outputs
    inputs = remove_constants([h, p, a, b, y])
    gibbs = DispersionGibbsKernel(inputs, outputs)

    h_posterior = gibbs(*inputs)
    h_posterior.name = f"{h.name or 'h'}_posterior"

    updates_offset = len(inputs)
    updates = {
        h_posterior.owner.inputs[updates_offset]: h_posterior.owner.outputs[1],
        h_posterior.owner.inputs[updates_offset + 1]: h_posterior.owner.outputs[2],
    }

    res.append((h, h_posterior, updates))

    return res


gibbs_db.register("nbinom_logistic_regression", nbinom_logistic_finder, "basic")


bernoulli_sigmoid_dot_pattern = etuple(
    etuplize(at.random.bernoulli), var(), var(), var(), sigmoid_dot_pattern
)


def bern_sigmoid_dot_match(
    graph: TensorVariable,
) -> Tuple[TensorVariable, TensorVariable]:

    graph_et = etuplize(graph)

    s = unify(graph_et, bernoulli_sigmoid_dot_pattern)
    if s is False:
        raise ValueError("Not a Bernoulli regression.")

    if all(s[neg_one_lv].data != -1):
        raise ValueError(
            "Not a Bernoulli regression. The argument to the sigmoid "
            + "must be minus the dot product."
        )

    beta_rv = eval_if_etuple(s[beta_lv])
    X = eval_if_etuple(s[X_lv])

    return X, beta_rv


def bern_normal_posterior(
    srng: RandomStream,
    beta: TensorVariable,
    beta_std: TensorVariable,
    X: TensorVariable,
    y: TensorVariable,
) -> Tuple[TensorVariable, TensorVariable, TensorVariable]:
    r"""Produce a Gibbs sample step for a bernoulli logistic-link regression with a normal prior.

    The implementation follows the sampler described in [1]_. It is designed to
    sample efficiently from the following binary logistic regression model:

    .. math::

        \begin{align*}
            Y &\sim \operatorname{Bern}\left( p \right) \\
            p &= \frac{1}{1 + \exp\left( -X^\top \beta\right)} \\
            \beta_j &\sim \operatorname{N}\right( 0, \lambda_j^2 \right)
        \end{align*}


    Parameters
    ----------
    beta
        The current/previous value of the regression parameter :math:`beta`.
    beta_std
        The std. dev. of the regression parameter :math:`beta`.
    X
        The regression matrix.
    y
        A sample from the observation distribution :math:`y \sim Y`.

    Returns
    -------
    A sample from :math:`\beta \mid y`.

    References
    ----------
    .. [1] Makalic, Enes & Schmidt, Daniel. (2016). High-Dimensional Bayesian
          Regularised Regression with the BayesReg Package.

    """
    w = srng.gen(polyagamma, 1, X @ beta)
    z = (y - 0.5) / w

    tau_beta = at.reciprocal(beta_std)

    beta_posterior = normal_regression_posterior(srng, w, tau_beta, X, z)

    return beta_posterior


class BernoulliRegressionGibbsKernel(SamplingStep):
    """An `Op` that represents the update of the regression parameter of
    a Bernoulli regression.

    """

    default_output = 0


@sampler_finder([BernoulliRV])
def bern_logistic_finder(fgraph, node, srng):
    r"""Find and construct a Gibbs sampler for a negative binomial logistic-link regression."""

    y = node.outputs[1]

    try:
        X, beta_rv = bern_sigmoid_dot_match(node.outputs[1])
    except ValueError:  # pragma: no cover
        return None

    beta_std = beta_rv.owner.inputs[4]
    beta_posterior = bern_normal_posterior(srng, beta_rv, beta_std, X, y)

    # Build the `Op` corresponding to the sampling step
    outputs = [beta_posterior]
    inputs = remove_constants([beta_rv, beta_std, X, y])
    gibbs = BernoulliRegressionGibbsKernel(inputs, outputs)

    beta_posterior: TensorVariable = gibbs(*inputs)  # type: ignore
    beta_posterior.name = f"{beta_rv.name or 'beta'}_posterior"  # type: ignore

    return [(beta_rv, beta_posterior, None)]


gibbs_db.register("bern_logistic_finder", bern_logistic_finder, "basic")

sampler_finder_db.register(
    "gibbs_db", in2out(gibbs_db.query("+basic"), name="gibbs"), "basic"
)
