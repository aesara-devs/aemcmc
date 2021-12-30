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
    nrng = np.random.default_rng(54321)
    b = np.array([0.5, -0.2, 0.75, 1.0, -2.22])
    Q = csc_matrix(np.diag(nrng.random(5)))

    srng = at.random.RandomStream(12345)
    got = multivariate_normal_rue2005(srng, at.as_tensor(b), as_sparse(Q))
    expected = np.array(
        [-0.87260997, 0.24812936, -0.14312798, 30.57354048, -6.83054447]
    )
    np.testing.assert_allclose(got.eval(), expected)


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
    expected = np.array([0.13220936, 0.20621965, -2.98777855, -2.35904856, -0.19972386])
    np.testing.assert_allclose(got.eval(), expected)


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
    expected = np.array([0.79532198, 0.54771371, 0.42505174, -0.33428737, -0.74749463])
    np.testing.assert_allclose(got.eval(), expected)
