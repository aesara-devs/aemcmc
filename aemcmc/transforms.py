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
fact(location_scale_family, at.random.t)


def location_scale_transform(in_expr, out_expr):
    r"""Produce a goal that represents the action of lifting and sinking
    the scale and location parameters of distributions in the location-scale
    family.

    For instance

    .. math::

        \begin{equation*}
           \frac{
                Y \sim \operatorname{P}(0, 1), \quad
                X = \mu + \sigma\,Y
            }{
                X \sim \operatorname{P}\left(\mu, \sigma\right)
            }
        \end{equation*}

    where `P` is any distribution in the location-scale family.

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


def invgamma_exponential(invgamma_expr, invexponential_expr):
    r"""Produce a goal that represents the relation between the inverse gamma distribution
    and the inverse of an exponential distribution.

    .. math::

        \begin{equation*}
            \frac{
                X \sim \operatorname{Gamma^{-1}}\left(1, c\right)
            }{
                Y = 1 / X, \quad
                Y \sim \operatorname{Exp}\left(c\right)
            }
        \end{equation*}

    TODO: This is a particular case of a more general relation between the inverse gamma
    and the gamma distribution (of which the exponential distribution is a special case).
    We should implement this more general relation, and the special case separately in the
    future.

    Parameters
    ----------
    invgamma_expr
        An expression that represents a random variable with an inverse gamma
        distribution with a shape parameter equal to 1.
    invexponential_expr
        An expression that represents the inverse of a random variable with an
        exponential distribution.

    """
    c_lv = var()
    rng_lv, size_lv, dtype_lv = var(), var(), var()

    invgamma_et = etuple(
        etuplize(at.random.invgamma), rng_lv, size_lv, dtype_lv, at.as_tensor(1.0), c_lv
    )

    exponential_et = etuple(
        etuplize(at.random.exponential),
        c_lv,
        rng=rng_lv,
        size=size_lv,
        dtype=dtype_lv,
    )
    invexponential_et = etuple(at.true_div, at.as_tensor(1.0), exponential_et)

    return lall(
        eq(invgamma_expr, invgamma_et), eq(invexponential_expr, invexponential_et)
    )
