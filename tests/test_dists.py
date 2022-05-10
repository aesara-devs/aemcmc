import aesara
import aesara.tensor as at
import numpy as np
import pytest
from aesara.sparse.basic import as_sparse
from scipy.sparse import csc_matrix

from aemcmc.dists import (
    multivariate_normal_bhattacharya2016,
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


def test_multivariate_normal_bhattacharya2016():
    nrng = np.random.default_rng(54321)
    X = nrng.standard_normal(size=10 * 5)
    X.resize((10, 5))
    XX = X.T @ X
    D, phi = np.linalg.eigh(XX)
    alpha = nrng.random(phi.shape[1])

    srng = at.random.RandomStream(12345)
    got = multivariate_normal_bhattacharya2016(
        srng, at.as_tensor(D), at.as_tensor(phi), at.as_tensor(alpha)
    )
    expected_shape = (5,)
    np.testing.assert_allclose(np.shape(got.eval()), expected_shape)


def test_multivariate_normal_cong2017():
    nrng = np.random.default_rng(54321)
    X = nrng.standard_normal(size=10 * 5)
    X.resize((10, 5))
    XX = X.T @ X
    omega, phi = np.linalg.eigh(XX)
    A = nrng.random(phi.shape[1])
    t = nrng.random(phi.shape[1])

    srng = at.random.RandomStream(12345)
    got = multivariate_normal_cong2017(
        srng, at.as_tensor(A), at.as_tensor(omega), at.as_tensor(phi), at.as_tensor(t)
    )
    expected_shape = (5,)
    np.testing.assert_allclose(np.shape(got.eval()), expected_shape)
