import aesara
import aesara.tensor as at
import numpy as np
from aesara.sparse.basic import _is_sparse_variable, dense_from_sparse
from aesara.tensor.random.op import RandomVariable
from polyagamma import random_polyagamma


class PolyaGammaRV(RandomVariable):
    name = "polyagamma"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("PG", "\\operatorname{PG}")

    def __call__(self, h=1.0, z=0.0, size=None, **kwargs):
        return super().__call__(h, z, size=size, **kwargs)

    @classmethod
    def rng_fn(cls, rng, h, z, size=None):
        bg = rng._bit_generator if isinstance(rng, np.random.RandomState) else rng
        return random_polyagamma(h, z, size=size, random_state=bg).astype(
            aesara.config.floatX
        )


polyagamma = PolyaGammaRV()


def multivariate_normal_rue2005(rng, b, Q):
    """
    Sample from a multivariate normal distribution of the form N(Qinv * b, Qinv).

    We use the algorithm described in [1]. This algorithm is suitable for when
    the number of regression coefficients is significantly less than the number
    of data points.

    References
    ----------
    ..[1] Rue, H. and Held, L. (2005), Gaussian Markov Random Fields, Boca
          Raton: Chapman & Hall/CRC.
    """
    if _is_sparse_variable(Q):
        Q = dense_from_sparse(Q)
    L = at.linalg.cholesky(Q)
    w = at.slinalg.solve_triangular(L, b, lower=True)
    u = at.slinalg.solve_triangular(L.T, w, lower=False)
    z = rng.standard_normal(size=L.shape[0])
    v = at.slinalg.solve_triangular(L.T, z, lower=False)
    return u + v


# A and omega are assumed to be diagonal
def multivariate_normal_cong2017(rng, A, omega, phi, t):
    """
    Sample from a multivariate normal distribution with a structured mean
    and covariance matrix, as described in Example 4 [page 17] of [1]. This
    algorithm is suitable for high-dimensional regression problems and the
    runtime scales linearly with the number of regression coefficients.

    This implementation assumes that A and omega are  diagonal matrices and
    the parameters ``A`` and ``omega`` are expected to be an arrays containing
    diagonal entries of the matrices.

    References
    ----------
    ..[1] Cong, Yulai; Chen, Bo; Zhou, Mingyuan. Fast Simulation of Hyperplane-
          Truncated Multivariate Normal Distributions. Bayesian Anal. 12 (2017),
          no. 4, 1017--1037. doi:10.1214/17-BA1052.
    """
    A_inv = 1 / A
    a_rows = A.shape[0]
    z = rng.standard_normal(size=a_rows + omega.shape[0])
    y1 = at.sqrt(A_inv) * z[:a_rows]
    y2 = (1 / at.sqrt(omega)) * z[a_rows:]
    Ainv_phi = A_inv[:, None] * phi.T
    B = phi @ Ainv_phi
    indices = at.arange(B.shape[0])
    B = at.subtensor.set_subtensor(
        B[indices, indices],
        at.diag(B) + 1.0 / omega,
    )
    alpha = at.linalg.solve(B, t - phi @ y1 - y2, assume_a="pos")
    return y1 + Ainv_phi @ alpha


def multivariate_normal_bhattacharya2016(rng, D, phi, alpha):
    """
    Sample from a multivariable normal distribution of the form:
        N(Sigma * phi.T * alpha, Sigma),
        where Sigma = inv(phi.T * phi + inv(D)) and D is positive definite.

    This implementation assumes that D is a symmetric matrix and the parameter
    ``D`` is expected to be an array containing diagonal entries of the D matrix.

    References
    ----------
    ..[1] Bhattacharya, A., Chakraborty, A., and Mallick, B. K. (2016).
          “Fast sampling with Gaussian scale mixture priors in high-dimensional
          regression.” Biometrika, 103(4): 985.033
    """
    D_phi = D[:, None] * phi.T
    B = phi @ D_phi
    indices = at.arange(B.shape[0])
    B = at.subtensor.set_subtensor(
        B[indices, indices],
        at.diag(B) + 1.0,
    )
    u = rng.normal(0, at.sqrt(D))
    v = phi @ u + rng.standard_normal(size=phi.shape[0])
    w = at.linalg.solve(B, alpha - v, assume_a="pos")
    return u + D_phi @ w
