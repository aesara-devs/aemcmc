from typing import Dict, List, Mapping, Tuple, Union

import aesara
import aesara.tensor as at
from aesara.graph import optimize_graph
from aesara.graph.basic import Variable
from aesara.graph.opt import EquilibriumOptimizer
from aesara.graph.unify import eval_if_etuple
from aesara.ifelse import ifelse
from aesara.tensor.math import Dot
from aesara.tensor.random import RandomStream
from aesara.tensor.random.opt import local_dimshuffle_rv_lift
from aesara.tensor.var import TensorVariable
from etuples import etuple, etuplize
from etuples.core import ExpressionTuple
from unification import unify, var

from aemcmc.dists import (
    multivariate_normal_cong2017,
    multivariate_normal_rue2005,
    polyagamma,
)


def canonicalize_and_tuplize(graph: TensorVariable) -> ExpressionTuple:
    """Canonicalize and etuple-ize a graph."""
    graph_opt = optimize_graph(
        graph,
        custom_opt=EquilibriumOptimizer(
            [local_dimshuffle_rv_lift], max_use_ratio=aesara.config.optdb__max_use_ratio
        ),
    )
    graph_et = etuplize(graph_opt)
    return graph_et


def update_beta_low_dimension(
    srng: RandomStream,
    omega: TensorVariable,
    lmbdatau_inv: TensorVariable,
    X: TensorVariable,
    z: TensorVariable,
) -> TensorVariable:
    Q = X.T @ (omega[:, None] * X)
    indices = at.arange(Q.shape[1])
    Q = at.subtensor.set_subtensor(
        Q[indices, indices],
        at.diag(Q) + lmbdatau_inv,
    )
    return multivariate_normal_rue2005(srng, X.T @ (omega * z), Q)


def update_beta_high_dimension(
    srng: RandomStream,
    omega: TensorVariable,
    lmbdatau_inv: TensorVariable,
    X: TensorVariable,
    z: TensorVariable,
) -> TensorVariable:
    return multivariate_normal_cong2017(srng, lmbdatau_inv, omega, X, z)


def update_beta(
    srng: RandomStream,
    omega: TensorVariable,
    lmbdatau_inv: TensorVariable,
    X: TensorVariable,
    z: TensorVariable,
) -> TensorVariable:
    return ifelse(
        X.shape[1] > X.shape[0],
        update_beta_high_dimension(srng, omega, lmbdatau_inv, X, z),
        update_beta_low_dimension(srng, omega, lmbdatau_inv, X, z),
    )


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


def horseshoe_model(srng: TensorVariable) -> TensorVariable:
    """Horseshoe shrinkage prior [1]_.

    References
    ----------
    .. [1]: Carvalho, C. M., Polson, N. G., & Scott, J. G. (2010).
            The horseshoe estimator for sparse signals.
            Biometrika, 97(2), 465-480.

    """
    size = at.scalar("size", dtype="int32")
    tau_rv = srng.halfcauchy(0, 1, size=1)
    lmbda_rv = srng.halfcauchy(0, 1, size=size)
    beta_rv = srng.normal(0, tau_rv * lmbda_rv, size=size)
    return beta_rv


def horseshoe_match(graph: TensorVariable) -> Tuple[TensorVariable, TensorVariable]:

    graph_et = canonicalize_and_tuplize(graph)

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

    if halfcauchy_1.type.shape == (1,):
        lmbda_rv = halfcauchy_2
        tau_rv = halfcauchy_1
    elif halfcauchy_2.type.shape == (1,):
        lmbda_rv = halfcauchy_1
        tau_rv = halfcauchy_2
    else:
        raise ValueError(
            "Not a horseshoe prior. The global shrinkage parameter "
            + "in your model must be one-dimensional."
        )

    return (lmbda_rv, tau_rv)


def horseshoe_step(
    srng: RandomStream,
    beta: TensorVariable,
    sigma: TensorVariable,
    lmbda_inv: TensorVariable,
    tau_inv: TensorVariable,
) -> Tuple[TensorVariable, TensorVariable]:
    r"""Gibbs kernel to sample from the posterior distribution of a horseshoe prior.

    This kernel generates samples from the posterior distribution of the local
    and global shrinkage parameters of a horseshoe prior, respectively :math:`\lambda`
    and :math:`\tau` in the following model:

    .. math::

        \begin{align*}
            \beta_j &\sim \operatorname{N}\left(0, \lambda_j^2 \tau^2 \sigma^2\right) \\
            \lambda_j &\sim \operatorname{HalfCauchy}(0, 1) \\
            \tau &\sim \operatorname{HalfCauchy}(0, 1)
        \end{align*}

    We use the following observations [1]_ to sample from the posterior
    conditional probability of :math:`\tau` and :math:`\lambda`:

    1. The half-Cauchy distribution can be intepreted as a mixture of inverse-gamma
    distributions;
    2. If :math:` Y \sim \operatorname{InvGamma}(1, a)`, :math:`Y \sim 1 / \operatorname{Exp}(a)`.

    Parameters
    ----------
    srng
        The random number generating object to be used during sampling.
    beta
        Regression coefficients.
    sigma2
        Variance of the regression coefficients.
    lmbda2_inv
        Square inverse of the local shrinkage parameters.
    tau2_inv
        Square inverse of the global shrinkage parameters.

    References
    ----------
    .. [1] Makalic, Enes & Schmidt, Daniel. (2016). High-Dimensional Bayesian
          Regularised Regression with the BayesReg Package.

    """
    upsilon_inv = srng.exponential(1 + lmbda_inv)
    zeta_inv = srng.exponential(1 + tau_inv)

    beta2 = beta * beta
    lmbda_inv_new = srng.exponential(upsilon_inv + 0.5 * beta2 * tau_inv / sigma)
    tau_inv_new = srng.gamma(
        0.5 * (beta.shape[0] + 1),
        zeta_inv + 0.5 * (beta2 * lmbda_inv_new).sum() / sigma,
    )
    return lmbda_inv_new, tau_inv_new


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


def gamma_match(
    graph: TensorVariable,
) -> Tuple[TensorVariable, TensorVariable]:
    graph_opt = optimize_graph(graph)
    graph_et = etuplize(graph_opt)
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
    graph_opt = optimize_graph(graph)
    graph_et = etuplize(graph_opt)
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


def nbinom_horseshoe_model(srng: RandomStream) -> TensorVariable:
    """Negative binomial regression model with a horseshoe shrinkage prior."""
    X = at.matrix("X")
    h = at.scalar("h")

    beta_rv = horseshoe_model(srng)
    eta = X @ beta_rv
    p = at.sigmoid(-eta)
    Y_rv = srng.nbinom(h, p)

    return Y_rv


def nbinom_horseshoe_match(
    Y_rv: TensorVariable,
) -> Tuple[
    TensorVariable, TensorVariable, TensorVariable, TensorVariable, TensorVariable
]:
    X, h, beta_rv = nbinom_sigmoid_dot_match(Y_rv)
    lmbda_rv, tau_rv = horseshoe_match(beta_rv)
    return h, X, beta_rv, lmbda_rv, tau_rv


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


def h_step(
    srng: RandomStream,
    h_last: TensorVariable,
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

    where `y` is a sample from :math:`y \sim Y`.

    The conditional posterior sample step is derived from the following decomposition:

    .. math::
        \begin{gather*}
            Y_i = \sum_{j=1}^{l_i} u_{i j}, \quad u_{i j} \sim \operatorname{Log}(p_i), \quad
            l_i \sim \operatorname{Pois}\left(-h \log(1 - p_i)\right)
        \end{gather*}

    where :math:`\operatorname{Log}` is the logarithmic distribution.  Under a
    gamma prior, :math:`h` is conjugate to :math:`l`.  We draw samples from
    :math:`l` according to :math:`l \sim \operatorname{CRT(y, h)}`.

    The resulting posterior is

    .. math::

        \begin{gather*}
            \left(h \mid Y = y\right) \sim \operatorname{Gamma}\left(a + \sum_{i=1}^N l_i, \frac{1}{1/b + \sum_{i=1}^N \log(1 - p_i)} \right)
        \end{gather*}


    References
    ----------
    .. [1] Zhou, Mingyuan, Lingbo Li, David Dunson, and Lawrence Carin. 2012. “Lognormal and Gamma Mixed Negative Binomial Regression.” Proceedings of the International Conference on Machine Learning. International Conference on Machine Learning 2012: 1343–50. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4180062/.

    """
    Ls, updates = sample_CRT(srng, y, h_last)
    L_sum = Ls.sum(axis=-1)
    h = srng.gamma(a + L_sum, at.reciprocal(b) - at.sum(at.log(1 - p), axis=-1))
    h.name = f"{h_last.name or 'h'}-posterior"
    return h, updates


def nbinom_horseshoe_with_dispersion_match(
    Y_rv: TensorVariable,
) -> Tuple[
    TensorVariable,
    TensorVariable,
    TensorVariable,
    TensorVariable,
    TensorVariable,
    TensorVariable,
    TensorVariable,
]:
    X, h_rv, beta_rv = nbinom_sigmoid_dot_match(Y_rv)
    lmbda_rv, tau_rv = horseshoe_match(beta_rv)
    a, b = gamma_match(h_rv)
    return X, beta_rv, lmbda_rv, tau_rv, h_rv, a, b


def nbinom_horseshoe_gibbs(
    srng: RandomStream, Y_rv: TensorVariable, y: TensorVariable, num_samples: int
) -> Tuple[Union[TensorVariable, List[TensorVariable]], Dict]:
    r"""Build a Gibbs sampler for the negative binomial regression with a horseshoe prior.

    The implementation follows the sampler described in [1]_. It is designed to
    sample efficiently from the following negative binomial regression model:

    .. math::

        \begin{align*}
            Y_i &\sim \operatorname{NB}\left(h, p_i\right) \\
            p_i &= \frac{\exp(\psi_i)}{1 + \exp(\psi_i)} \\
            \psi_i &= x_i^\top \beta \\
            \beta_j &\sim \operatorname{N}(0, \lambda_j^2 \tau^2) \\
            \lambda_j &\sim \operatorname{HalfCauchy}(0, 1) \\
            \tau &\sim \operatorname{HalfCauchy}(0, 1)
        \end{align*}


    Parameters
    ----------
    srng: symbolic random number generator
        The random number generating object to be used during sampling.
    Y_rv
        Model graph.
    y: TensorVariable
        The observed count data.
    n_samples: TensorVariable
        A tensor describing the number of posterior samples to generate.

    Returns
    -------
    (outputs, updates): tuple
        A symbolic description of the sampling result to be used to
        compile a sampling function.

    Notes
    -----
    The :math:`z` expression in Section 2.2 of [1]_ seems to
    omit division by the Polya-Gamma auxilliary variables whereas [2]_ and [3]_
    explicitely include it. We found that including the division results in
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

    def nbinom_horseshoe_step(
        beta: TensorVariable,
        lmbda: TensorVariable,
        tau: TensorVariable,
        y: TensorVariable,
        X: TensorVariable,
        h: TensorVariable,
    ) -> Tuple[TensorVariable, TensorVariable, TensorVariable]:
        """Complete one full update of the Gibbs sampler and return the new state
        of the posterior conditional parameters.

        Parameters
        ----------
        beta: Tensorvariable
            Coefficients (other than intercept) of the regression model.
        lmbda
            Inverse of the local shrinkage parameter of the horseshoe prior.
        tau
            Inverse of the global shrinkage parameters of the horseshoe prior.
        y: TensorVariable
            The observed count data.
        X: TensorVariable
            The covariate matrix.
        h: TensorVariable
            The "number of successes" parameter of the negative binomial disribution
            used to model the data.

        """
        xb = X @ beta
        w = srng.gen(polyagamma, y + h, xb)
        z = 0.5 * (y - h) / w

        lmbda_inv = 1.0 / lmbda
        tau_inv = 1.0 / tau
        beta_new = update_beta(srng, w, lmbda_inv * tau_inv, X, z)

        lmbda_inv_new, tau_inv_new = horseshoe_step(
            srng, beta_new, 1.0, lmbda_inv, tau_inv
        )
        return beta_new, 1.0 / lmbda_inv_new, 1.0 / tau_inv_new

    h, X, beta_rv, lmbda_rv, tau_rv = nbinom_horseshoe_match(Y_rv)

    outputs, updates = aesara.scan(
        nbinom_horseshoe_step,
        outputs_info=[beta_rv, lmbda_rv, tau_rv],
        non_sequences=[y, X, h],
        n_steps=num_samples,
        strict=True,
    )

    return outputs, updates


def nbinom_horseshoe_gibbs_with_dispersion(
    srng: RandomStream,
    Y_rv: TensorVariable,
    y: TensorVariable,
    num_samples: TensorVariable,
) -> Tuple[Union[TensorVariable, List[TensorVariable]], Mapping[Variable, Variable]]:
    r"""Build a Gibbs sampler for the negative binomial regression with a horseshoe prior and gamma prior dispersion.

    This is a direct extension of `nbinom_horseshoe_gibbs_with_dispersion` that
    adds a gamma prior assumption to the :math:`h` parameter in the
    negative-binomial and samples according to [1]_.

    In other words, this model is the same as `nbinom_horseshoe_gibbs` except
    for the addition assumption:

    .. math::

        \begin{gather*}
            h \sim \operatorname{Gamma}\left(a, b\right)
        \end{gather*}


    References
    ----------
    .. [1] Zhou, Mingyuan, Lingbo Li, David Dunson, and Lawrence Carin. 2012. “Lognormal and Gamma Mixed Negative Binomial Regression.” Proceedings of the International Conference on Machine Learning. International Conference on Machine Learning 2012: 1343–50. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4180062/.

    """

    def nbinom_horseshoe_step(
        beta: TensorVariable,
        lmbda: TensorVariable,
        tau: TensorVariable,
        h: TensorVariable,
        y: TensorVariable,
        X: TensorVariable,
        a: TensorVariable,
        b: TensorVariable,
    ):
        """Complete one full update of the Gibbs sampler and return the new state
        of the posterior conditional parameters.

        Parameters
        ----------
        beta
            Coefficients (other than intercept) of the regression model.
        lmbda
            Inverse of the local shrinkage parameter of the horseshoe prior.
        tau
            Inverse of the global shrinkage parameters of the horseshoe prior.
        h
            The "number of successes" parameter of the negative-binomial distribution
            used to model the data.
        y
            The observed count data.
        X
            The covariate matrix.
        a
            The shape parameter for the :math:`h` gamma prior.
        b
            The rate parameter for the :math:`h` gamma prior.

        """
        xb = X @ beta
        w = srng.gen(polyagamma, y + h, xb)
        z = 0.5 * (y - h) / w

        lmbda_inv = 1.0 / lmbda
        tau_inv = 1.0 / tau
        beta_new = update_beta(srng, w, lmbda_inv * tau_inv, X, z)

        lmbda_inv_new, tau_inv_new = horseshoe_step(
            srng, beta_new, 1.0, lmbda_inv, tau_inv
        )
        eta = X @ beta_new
        p = at.sigmoid(-eta)
        h_new, h_updates = h_step(srng, h, p, a, b, y)

        return (beta_new, 1.0 / lmbda_inv_new, 1.0 / tau_inv_new, h_new), h_updates

    X, beta_rv, lmbda_rv, tau_rv, h_rv, a, b = nbinom_horseshoe_with_dispersion_match(
        Y_rv
    )

    outputs, updates = aesara.scan(
        nbinom_horseshoe_step,
        outputs_info=[beta_rv, lmbda_rv, tau_rv, h_rv],
        non_sequences=[y, X, a, b],
        n_steps=num_samples,
        strict=True,
    )

    return outputs, updates


bernoulli_sigmoid_dot_pattern = etuple(
    etuplize(at.random.bernoulli), var(), var(), var(), sigmoid_dot_pattern
)


def bernoulli_sigmoid_dot_match(
    graph: TensorVariable,
) -> Tuple[TensorVariable, TensorVariable]:
    graph_opt = optimize_graph(graph)
    graph_et = etuplize(graph_opt)
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


def bernoulli_horseshoe_model(srng: RandomStream) -> TensorVariable:
    """Bernoulli regression model with a horseshoe shrinkage prior."""
    X = at.matrix("X")

    beta_rv = horseshoe_model(srng)
    eta = X @ beta_rv
    p = at.sigmoid(-eta)
    Y_rv = srng.bernoulli(p)

    return Y_rv


def bernoulli_horseshoe_match(
    Y_rv: TensorVariable,
) -> Tuple[TensorVariable, TensorVariable, TensorVariable, TensorVariable]:
    X, beta_rv = bernoulli_sigmoid_dot_match(Y_rv)
    lmbda_rv, tau_rv = horseshoe_match(beta_rv)

    return X, beta_rv, lmbda_rv, tau_rv


def bernoulli_horseshoe_gibbs(
    srng: RandomStream, Y_rv: TensorVariable, y: TensorVariable, num_samples: int
) -> Tuple[Union[TensorVariable, List[TensorVariable]], Dict]:
    r"""Build a Gibbs sampler for Bernoulli (logistic) regression with a horseshoe prior.

    The implementation follows the sampler described in [1]_. It is designed to
    sample efficiently from the following binary logistic regression model:

    .. math::

        \begin{align*}
            Y_i &\sim \operatorname{Bern}\left(p_i\right) \\
            p_i &= \frac{1}{1 + \exp\left(-(\beta_0 + x_i^\top \beta)\right)} \\
            \beta_j &\sim \operatorname{N}(0, \lambda_j^2 \tau^2) \\
            \lambda_j &\sim \operatorname{HalfCauchy}(0, 1) \\
            \tau &\sim \operatorname{HalfCauchy}(0, 1)
        \end{align*}

    Parameters
    ----------
    srng
        The random number generating object to be used during sampling.
    Y_rv
        Model graph.
    y
        The observed binary data.
    X
        The covariate matrix.
    n_samples
        A tensor describing the number of posterior samples to generate.

    Returns
    -------
    (outputs, updates): tuple
        A symbolic description of the sampling result to be used to
        compile a sampling function.


    References
    ----------
    .. [1] Makalic, Enes & Schmidt, Daniel. (2015). A Simple Sampler for the
          Horseshoe Estimator. 10.1109/LSP.2015.2503725.
    .. [2] Makalic, Enes & Schmidt, Daniel. (2016). High-Dimensional Bayesian
          Regularised Regression with the BayesReg Package.

    """

    def bernoulli_horseshoe_step(
        beta: TensorVariable,
        lmbda: TensorVariable,
        tau: TensorVariable,
        y: TensorVariable,
        X: TensorVariable,
    ) -> Tuple[TensorVariable, TensorVariable, TensorVariable]:
        """Complete one full update of the Gibbs sampler and return the new
        state of the posterior conditional parameters.

        Parameters
        ----------
        beta
            Coefficients (other than intercept) of the regression model.
        lmbda
            Square of the local shrinkage parameter of the horseshoe prior.
        tau
            Square of the global shrinkage parameters of the horseshoe prior.
        y
            The observed binary data
        X
            The covariate matrix.

        """
        xb = X @ beta
        w = srng.gen(polyagamma, 1, xb)
        z = 0.5 * y / w

        lmbda_inv = 1.0 / lmbda
        tau_inv = 1.0 / tau
        beta_new = update_beta(srng, w, lmbda_inv * tau_inv, X, z)

        lmbda_inv_new, tau_inv_new = horseshoe_step(
            srng, beta_new, 1.0, lmbda_inv, tau_inv
        )

        return beta_new, 1 / lmbda_inv_new, 1.0 / tau_inv_new

    X, beta_rv, lmbda_rv, tau_rv = bernoulli_horseshoe_match(Y_rv)

    outputs, updates = aesara.scan(
        bernoulli_horseshoe_step,
        outputs_info=[beta_rv, lmbda_rv, tau_rv],
        non_sequences=[y, X],
        n_steps=num_samples,
        strict=True,
    )

    return outputs, updates
