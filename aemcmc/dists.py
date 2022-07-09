import aesara
import aesara.tensor as at
import numpy as np
from aesara.sparse.basic import _is_sparse_variable, dense_from_sparse
from aesara.tensor.random import RandomStream
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.var import TensorVariable
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
    r"""Sample from a multivariate normal distribution.

    More specifically, this function draws a sample from the following distribution:

    .. math::

        \operatorname{N}\left( Q^{-1} b, Q^{-1} \right)

    It uses the algorithm described in [1]_, which is suitable for cases in
    which the number of regression coefficients is significantly less than the
    number of data points.

    References
    ----------
    .. [1] Rue, H. and Held, L. (2005), Gaussian Markov Random Fields, Boca
          Raton: Chapman & Hall/CRC.
    """
    if _is_sparse_variable(Q):
        Q = dense_from_sparse(Q)
    L = at.linalg.cholesky(Q)
    w = at.slinalg.solve_triangular(L, b, lower=True)
    u = at.slinalg.solve_triangular(L.T, w, lower=False)
    z = rng.standard_normal(size=L.shape[0])
    z.owner.outputs[0].name = "z_rng"
    v = at.slinalg.solve_triangular(L.T, z, lower=False)
    return u + v


def multivariate_normal_cong2017(
    rng: RandomStream,
    A: TensorVariable,
    omega: TensorVariable,
    phi: TensorVariable,
    t: TensorVariable,
) -> TensorVariable:
    r"""Sample from a multivariate normal distribution with a structured mean and covariance.

    As described in Example 4 [page 17] of [1]_, The covariance of this normal
    distribution should be decomposable into a sum of a positive-definite matrix
    and a low-rank symmetric matrix such that:

    .. math::

         \operatorname{N}\left(\Lambda^{-1} \Phi^{\top} \Omega t, \Lambda^{-1}\right)

    where

    .. math::

        \Lambda = A + \Phi^{\top} \Omega \Phi

    and :math:`A` is the positive-definite part and
    :math:`\Phi^{\top} \Omega \Phi` is the eigen-factorization of
    the low-rank part of the "structured" covariance.

    Parameters
    ----------
    rng
        The random number generating object to be used during sampling.
    A
        The entries of the diagonal elements of the positive-definite part of
        the structured covariance.
    omega
        The elements of the diagonal matrix in the eigen-decomposition of the
        low-rank part of the structured covariance of the multivariate normal
        distribution.
    phi
        A matrix containing the eigenvectors of the eigen-decomposition of the
        low-rank part of the structured covariance of the normal distribution.
    t
        A 1D array whose length is the number of eigenvalues of the low-rank
        part of the structured covariance.

    Notes
    -----
    This algorithm is suitable for high-dimensional regression problems and the
    runtime scales linearly with the number of regression coefficients. This
    implementation assumes that `A` and `omega` are diagonal matrices and
    the parameters `A` and `omega` are expected to be vectors that contain
    diagonal entries of the respective matrices they represent.

    Note the the algorithm described in [2]_ is a special case when `omega` is
    the identity matrix. Samples from the algorithm described in [2]_ can be
    generated using the following:

    .. code::

        rng = at.random.RandomStream(12345)
        X = rng.standard_normal(size=(50, 100))
        phi = at.linalg.cholesky(X @ X.T).T
        D = rng.random(phi.shape[1])
        alpha = rng.random(phi.shape[0])
        multivariate_normal_cong2017(rng, 1 / D, at.ones(phi.shape[0]), phi, alpha)

    References
    ----------
    .. [1] Cong, Yulai; Chen, Bo; Zhou, Mingyuan. Fast Simulation of Hyperplane-
          Truncated Multivariate Normal Distributions. Bayesian Anal. 12 (2017),
          no. 4, 1017--1037. doi:10.1214/17-BA1052.
    .. [2] Bhattacharya, A., Chakraborty, A., and Mallick, B. K. (2016).
          “Fast sampling with Gaussian scale mixture priors in high-dimensional
          regression.” Biometrika, 103(4): 985.033
    """
    A_inv = 1 / A
    a_rows = A.shape[0]
    z = rng.standard_normal(size=a_rows + omega.shape[0])
    z.owner.outputs[0].name = "z_rng"
    y1 = at.sqrt(A_inv) * z[:a_rows]
    y2 = (1 / at.sqrt(omega)) * z[a_rows:]
    Ainv_phi = A_inv[:, None] * phi.T
    # NOTE: B is equivalent to B = phi @ Ainv_phi + at.diag(1 / omega), but
    # may be sligthly more efficient since we do not allocate memory for the
    # full inverse of omega but instead use just the diagonal elements.
    B = phi @ Ainv_phi
    indices = at.arange(B.shape[0])
    B = at.subtensor.set_subtensor(
        B[indices, indices],
        at.diag(B) + 1.0 / omega,
    )
    alpha = at.linalg.solve(B, t - phi @ y1 - y2, assume_a="pos")
    return y1 + Ainv_phi @ alpha
