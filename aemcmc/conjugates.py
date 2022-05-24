import aesara.tensor as at
from aesara.graph.kanren import KanrenRelationSub
from aesara.graph.optdb import OptimizationDatabase
from etuples import etuple, etuplize
from kanren import eq, lall
from unification import var

conjugatesdb = OptimizationDatabase()


def beta_binomial_conjugateo(observed_rv_expr, posterior_expr):
    r"""Produce a goal that represents the application of Bayes theorem
    for a beta prior with a binomial observation model.

    .. math::

        \begin{align*}
            p &\sim \operatorname{Beta}\left(\alpha, \beta\right)\\
            y &\sim \operatorname{Binomial}\left(n, p\right)
        \end{align*}

    If we observe :math:`y=Y`, then :math:`p` follows a beta distribution:

    .. math::

        p \sim \operatorname{Beta}\left(\alpha + Y, \beta + n - Y\right)

    Parameters
    ----------
    observed_rv_expr
        A tuple that contains expressions that represent the observed variable
        and it observed value respectively.
    posterior_exp
        An expression that represents the posterior distribution of the latent
        variable.

    """

    # Beta-binomial observation model
    alpha_lv, beta_lv = var(), var()
    p_rng_lv = var()
    p_size_lv = var()
    p_type_idx_lv = var()
    p_et = etuple(
        etuplize(at.random.beta), p_rng_lv, p_size_lv, p_type_idx_lv, alpha_lv, beta_lv
    )
    n_lv = var()
    Y_et = etuple(etuplize(at.random.binomial), var(), var(), var(), n_lv, p_et)

    y_lv = var()  # observation

    # Posterior distribution for p
    new_alpha_et = etuple(etuplize(at.add), alpha_lv, y_lv)
    new_beta_et = etuple(
        etuplize(at.sub), etuple(etuplize(at.add), beta_lv, n_lv), y_lv
    )
    p_posterior_et = etuple(
        etuplize(at.random.beta),
        new_alpha_et,
        new_beta_et,
        rng=p_rng_lv,
        size=p_size_lv,
        dtype=p_type_idx_lv,
    )

    return lall(
        eq(observed_rv_expr[0], Y_et),
        eq(observed_rv_expr[1], y_lv),
        eq(posterior_expr, p_posterior_et),
    )


conjugatesdb.register("beta_binomial", KanrenRelationSub(beta_binomial_conjugateo))
