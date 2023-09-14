import aesara.tensor as at
from etuples import etuple, etuplize
from kanren import eq, lall, lany, var
from kanren.core import succeed


def halfcauchy_inverse_gamma(in_expr, out_expr):
    r"""Produce a goal that represents the fact that a half-Cauchy distribution can be
    expressed as a scale mixture of inverse gamma distributions.

    .. math::

        \begin{equation*}
            \frac{
                X^2 | a \sim \operatorname{Gamma^{-1}}\left(1/2, a), \quad
                a \sim \operatorname{Gamma^{-1}}\left(1/2, 1/A^2\right), \quad
            }{
                X \sim \operatorname{C^{+}}\left(0, A)
            }
        \end{equation*}

    TODO: This relation is a particular case of a similar relation for the
    half-t distribution [1]_ which does not have an implementation yet in Aesara.
    When it becomes available we should replace this relation with the more
    general one, and implement the relation between the half-t and half-Cauchy
    distributions.

    Parameters
    ----------
    in_expr
        An expression that represents a half-Cauchy distribution.
    out_expr
        An expression that represents the square root of the inverse gamma scale
        mixture.

    References
    ----------
    .. [1]:  Wand, M. P., Ormerod, J. T., Padoan, S. A., & Frühwirth, R. (2011).
             Mean field variational Bayes for elaborate distributions. Bayesian
             Analysis, 6(4), 847-900.

    """

    # Half-Cauchy distribution
    rng_lv, size_lv, type_idx_lv = var(), var(), var()
    loc_at = at.as_tensor(0)
    scale_lv = var()
    X_halfcauchy_et = etuple(
        etuplize(at.random.halfcauchy), rng_lv, size_lv, type_idx_lv, loc_at, scale_lv
    )

    # Inverse-Gamma scale mixture
    rng_inner_lv, size_inner_lv, type_idx_inner_lv = var(), var(), var()
    rng_outer_lv, size_outer_lv, type_idx_outer_lv = var(), var(), var()
    a_et = etuple(
        etuplize(at.random.invgamma),
        at.as_tensor(0.5),
        etuple(
            etuplize(at.true_div),
            at.as_tensor(1.0),
            etuple(etuplize(at.pow), scale_lv, at.as_tensor(2)),
        ),
        rng=rng_inner_lv,
        size=size_inner_lv,
        dtype=type_idx_inner_lv,
    )
    X_scale_mixture_square_et = etuple(
        etuplize(at.random.invgamma),
        at.as_tensor(0.5),
        etuple(
            etuplize(at.true_div),
            at.as_tensor(1.0),
            a_et,
        ),
        rng=rng_outer_lv,
        size=size_outer_lv,
        dtype=type_idx_outer_lv,
    )
    X_scale_mixture_et = etuple(etuplize(at.sqrt), X_scale_mixture_square_et)

    return lall(
        eq(in_expr, X_halfcauchy_et),
        eq(out_expr, X_scale_mixture_et),
        eq(rng_inner_lv, rng_lv),
        eq(type_idx_inner_lv, type_idx_lv),
        eq(size_inner_lv, size_lv),
        lany(
            eq(rng_outer_lv, rng_lv),
            succeed,
        ),
        lany(
            eq(size_outer_lv, size_lv),
            succeed,
        ),
        lany(
            eq(type_idx_outer_lv, type_idx_lv),
            succeed,
        ),
    )
