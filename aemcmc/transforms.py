import aesara.tensor as at
from etuples import etuple, etuplize
from kanren import eq, lall
from kanren.facts import Relation, fact
from unification import var

location_scale_family = Relation("location-scale-family")
fact(location_scale_family, at.random.cauchy)
fact(location_scale_family, at.random.gumbel)
fact(location_scale_family, at.random.laplace)
fact(location_scale_family, at.random.logistic)
fact(location_scale_family, at.random.normal)


def location_scale_transform(in_expr, out_expr):
    r"""Produce a goal that represents the action of lifting and sinking
    the scale and location parameters of distributions in the location-scale
    family.

    For instance

    .. math::

        \begin{equation*}
            \frac{
                X \sim \operatorname{N}\left(\mu_x, \sigma_x^2\right), \quad
                Y \sim \operatorname{N}\left(\mu_y, \sigma_y^2\right), \quad
                X + Y = Z
            }{
                Z \sim \operatorname{N}\left(\mu_x + \mu_y, \sigma_x^2 + \sigma_y^2\right)
            }
        \end{equation*}

    Parameters
    ----------
    in_expr
        An expression that represents a random variable whose distribution belongs
        to the location-scale family.
    out_expr
        An expression for the non-centered representation of this random variable.

    """

    # Centered representation
    rng_lv, size_lv, type_idx_lv = var(), var(), var()
    mu_lv, sd_lv = var(), var()
    distribution_lv = var()
    centered_et = etuple(distribution_lv, rng_lv, size_lv, type_idx_lv, mu_lv, sd_lv)

    # Non-centered representation
    noncentered_et = etuple(
        etuplize(at.add),
        mu_lv,
        etuple(
            etuplize(at.mul),
            sd_lv,
            etuple(
                distribution_lv,
                0,
                1,
                rng=rng_lv,
                size=size_lv,
                dtype=type_idx_lv,
            ),
        ),
    )

    return lall(
        eq(in_expr, centered_et),
        eq(out_expr, noncentered_et),
        location_scale_family(distribution_lv),
    )
