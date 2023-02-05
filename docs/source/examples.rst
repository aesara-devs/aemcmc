Examples
========

AeMCMC is capable of identifying conjugacy relations and simplify the model graph before building a sampler:

.. code-block:: python

    import aesara
    import aemcmc
    import aesara.tensor as at

    srng = at.random.RandomStream(0)

    alpha_tt = at.scalar("alpha")
    beta_rv = srng.halfnormal(1.0)
    l_rv = srng.gamma(alpha_tt, beta_rv)

    Y_rv = srng.poisson(l_rv)

    y_vv = Y_rv.clone()

    sampler, initial_values = construct_sampler({Y_rv: y_vv}, srng)


AeMCMC can recognize some models like the following sparse regression and construct efficient samplers:

.. code-block:: python

    srng = at.random.RandomStream(0)

    X = at.matrix("X")

    tau_rv = srng.halfcauchy(0, 1)
    lmbda_rv = srng.halfcauchy(0, 1, size=X.shape[1])
    beta_rv = srng.normal(0, lmbda_rv * tau_rv, size=X.shape[1])

    a = at.scalar("a")
    b = at.scalar("b")
    h_rv = srng.gamma(a, b)

    # Negative-binomial regression
    eta = X @ beta_rv
    p = at.sigmoid(-eta)
    Y_rv = srng.nbinom(h_rv, p)

    y_vv = Y_rv.clone()

    sampler, initial_values = construct_sampler({Y_rv: y_vv}, srng)
