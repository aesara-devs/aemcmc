import aesara
import aesara.tensor as at
import numpy as np
import pytest
from aesara.sparse.basic import as_sparse
from scipy.sparse import csc_matrix

from aemcmc.dists import (
    multivariate_normal_cong2017,
    multivariate_normal_rue2005,
    polyagamma,
)


def test_polyagamma():
    assert polyagamma(size=(5, 4)).eval().shape == (5, 4)

    with pytest.raises(ValueError, match="values of `h` must be positive"):
        polyagamma([1, -0]).eval()

    default_dtype = aesara.config.floatX
    aesara.config.floatX = "float32"
    assert polyagamma().eval().dtype == "float32"
    aesara.config.floatX = default_dtype

    rng1 = at.random.RandomStream(12345)
    res1 = rng1.gen(polyagamma)
    rng2 = at.random.RandomStream(12345)
    res2 = rng2.gen(polyagamma)
    assert res1.eval() == res2.eval()


def test_multivariate_normal_rue2005():
    b = np.array([0, -1, 5])
    var_inv = np.array([1.0, 2.0, 4.0])
    Q = csc_matrix(np.diag(var_inv))

    var = 1.0 / var_inv
    mean = b * var

    srng = at.random.RandomStream(12345)

    def update():
        return multivariate_normal_rue2005(srng, at.as_tensor(b), as_sparse(Q))

    samples_out, updates = aesara.scan(update, n_steps=10000)
    sampling_fn = aesara.function((), samples_out, updates=updates)
    samples = sampling_fn()

    np.testing.assert_allclose(np.mean(samples, axis=0), mean, atol=0.1)
    np.testing.assert_allclose(np.var(samples, axis=0), var, atol=0.1)


def test_multivariate_normal_cong2017():
    nrng = np.random.default_rng(54321)
    X = nrng.standard_normal(size=(50, 100))
    omega, phi = np.linalg.eigh(X @ X.T)
    A = nrng.random(phi.shape[1])
    t = nrng.random(phi.shape[1])

    phi_omega = phi * omega
    cov = np.linalg.inv(A + phi_omega @ phi.T)
    mean = cov @ phi_omega @ t

    srng = at.random.RandomStream(12345)

    def update():
        return multivariate_normal_cong2017(
            srng,
            at.as_tensor(A),
            at.as_tensor(omega),
            at.as_tensor(phi.T),
            at.as_tensor(t),
        )

    samples_out, updates = aesara.scan(update, n_steps=10000)
    sampling_fn = aesara.function((), samples_out, updates=updates)
    samples = sampling_fn()

    np.testing.assert_allclose(np.mean(samples, axis=0), mean, atol=0.05)
    np.testing.assert_allclose(np.var(samples, axis=0, ddof=1), np.diag(cov), atol=0.05)


def test_multivariate_normal_bhattacharya2016_via_cong2017():
    """This test case is to test if the algorithm from Cong et al (2017) can
    be used realiably to implement one described in Bhattacharya (2016) as a
    special case when ``omega`` is the identity matrix.
    """
    nrng = np.random.default_rng(54321)
    X = nrng.standard_normal(size=(50, 100))
    phi = np.linalg.cholesky(X @ X.T).T
    D = nrng.random(phi.shape[1])
    alpha = nrng.random(phi.shape[0])
    precision = phi.T @ phi
    precision[np.diag_indices_from(precision)] += 1.0 / D
    cov = np.linalg.inv(precision)
    mean = cov @ phi.T @ alpha

    srng = at.random.RandomStream(12345)

    def update():
        def multivariate_normal_bhattacharya2016(rng, D, phi, alpha):
            return multivariate_normal_cong2017(
                rng, 1 / D, at.ones(phi.shape[0]), phi, alpha
            )

        return multivariate_normal_bhattacharya2016(
            srng, at.as_tensor(D), at.as_tensor(phi), at.as_tensor(alpha)
        )

    samples_out, updates = aesara.scan(update, n_steps=10000)
    sampling_fn = aesara.function((), samples_out, updates=updates)
    samples = sampling_fn()

    np.testing.assert_allclose(np.mean(samples, axis=0), mean, atol=0.05)
    np.testing.assert_allclose(np.var(samples, axis=0, ddof=1), np.diag(cov), atol=0.05)
