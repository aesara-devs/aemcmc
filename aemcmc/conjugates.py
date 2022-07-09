import aesara.tensor as at
from aesara.graph.opt import in2out, local_optimizer
from aesara.graph.optdb import LocalGroupDB
from aesara.graph.unify import eval_if_etuple
from aesara.tensor.random.basic import BinomialRV
from etuples import etuple, etuplize
from kanren import eq, lall, run
from unification import var

from aemcmc.opt import sampler_finder_db


def beta_binomial_conjugateo(observed_val, observed_rv_expr, posterior_expr):
    r"""Produce a goal that represents the application of Bayes theorem
    for a beta prior with a binomial observation model.

    .. math::

        \frac{
            Y \sim \operatorname{Binom}\left(N, p\right), \quad
            p \sim \operatorname{Beta}\left(\alpha, \beta\right)
        }{
            \left(p|Y=y\right) \sim \operatorname{Beta}\left(\alpha+y, \beta+N-y\right)
        }


    Parameters
    ----------
    observed_val
        The observed value.
    observed_rv_expr
        An expression that represents the observed variable.
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

    # Posterior distribution for p
    new_alpha_et = etuple(etuplize(at.add), alpha_lv, observed_val)
    new_beta_et = etuple(
        etuplize(at.sub), etuple(etuplize(at.add), beta_lv, n_lv), observed_val
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
        eq(observed_rv_expr, Y_et),
        eq(posterior_expr, p_posterior_et),
    )


@local_optimizer([BinomialRV])
def local_beta_binomial_posterior(fgraph, node):

    sampler_mappings = getattr(fgraph, "sampler_mappings", None)

    rv_var = node.outputs[1]
    key = ("local_beta_binomial_posterior", rv_var)

    if sampler_mappings is None or key in sampler_mappings.rvs_seen:
        return None  # pragma: no cover

    q = var()

    rv_et = etuplize(rv_var)

    res = run(None, q, beta_binomial_conjugateo(rv_var, rv_et, q))
    res = next(res, None)

    if res is None:
        return None  # pragma: no cover

    beta_rv = rv_et[-1].evaled_obj
    beta_posterior = eval_if_etuple(res)

    sampler_mappings.rvs_to_samplers.setdefault(beta_rv, []).append(
        ("local_beta_binomial_posterior", beta_posterior, None)
    )
    sampler_mappings.rvs_seen.add(key)

    return rv_var.owner.outputs


conjugates_db = LocalGroupDB(apply_all_opts=True)
conjugates_db.name = "conjugates_db"
conjugates_db.register("beta_binomial", local_beta_binomial_posterior, "basic")

sampler_finder_db.register(
    "conjugates", in2out(conjugates_db.query("+basic"), name="gibbs"), "basic"
)
