|Tests Status| |Coverage| |Gitter|

AeMCMC is a Python library that automates the construction of samplers for `Aesara <https://github.com/pymc-devs/aesara>`_ graphs that represent statistical models.

Features
========

This project is currently in an alpha state, but the basic features/objectives are currently as follows:

- Provide utilities that simplify the process of constructing Aesara graphs/functions for posterior and posterior predictive sampling
- Host a wide array of "exact" posterior sampling steps (e.g. Gibbs steps, scale-mixture/decomposition-based conditional samplers, etc.)
- Build a framework for identifying and composing said sampler steps and enumerating the possible samplers for an arbitrary model

Overall, we would like this project to serve as a hub for community-sourced specialized samplers and facilitate their general use.

Getting started
===============

Using AeMCMC, one can construct sampling steps from a graph containing Aesara
`RandomVariable`\s. AeMCMC analyzes the model graph and possibly rewrites it
to find the most suitable sampler.

AeMCMC can recognize closed-form posteriors; for instance the following
Beta-Binomial model amounts to sampling from a Beta distribution:

.. code-block:: python

    import aesara
    import aemcmc
    import aesara.tensor as at

    srng = at.random.RandomStream(0)

    p_rv = srng.beta(1., 1., name="p")
    Y_rv = srng.binomial(10, p_rv, name="Y")

    y_vv = Y_rv.clone()
    y_vv.name = "y"

    sample_steps, _, initial_values, _ = aemcmc.construct_sampler(
        {Y_rv: y_vv}, srng
    )

    p_posterior_step = sample_steps[p_rv]
    aesara.dprint(p_posterior_step)
    # beta_rv{0, (0, 0), floatX, False}.1 [id A]
    #  |RandomGeneratorSharedVariable(<Generator(PCG64) at 0x7F77B2831200>) [id B]
    #  |TensorConstant{[]} [id C]
    #  |TensorConstant{11} [id D]
    #  |Elemwise{add,no_inplace} [id E]
    #  | |TensorConstant{1.0} [id F]
    #  | |y [id G]
    #  |Elemwise{sub,no_inplace} [id H]
    #    |Elemwise{add,no_inplace} [id I]
    #    | |TensorConstant{1.0} [id F]
    #    | |TensorConstant{10} [id J]
    #    |y [id G]

    sample_fn = aesara.function([y_vv], p_posterior_step)

AeMCMC also contains a database of Gibbs samplers that can be used to sample
some models more efficiently than a general-purpose sampler like NUTS
would:

.. code-block:: python

    import aemcmc
    import aesara.tensor as at

    srng = at.random.RandomStream(0)

    X = at.matrix("X")

    # Horseshoe prior for `beta_rv`
    tau_rv = srng.halfcauchy(0, 1, name="tau")
    lmbda_rv = srng.halfcauchy(0, 1, size=X.shape[1], name="lambda")
    beta_rv = srng.normal(0, lmbda_rv * tau_rv, size=X.shape[1], name="beta")

    a = at.scalar("a")
    b = at.scalar("b")
    h_rv = srng.gamma(a, b, name="h")

    # Negative-binomial regression
    eta = X @ beta_rv
    p = at.sigmoid(-eta)
    Y_rv = srng.nbinom(h_rv, p, name="Y")

    y_vv = Y_rv.clone()
    y_vv.name = "y"

    sample_steps, updates, initial_values, parameters = aemcmc.construct_sampler(
        {Y_rv: y_vv}, srng
    )
    print(sample_steps.keys())
    # dict_keys([tau, lambda, beta, h])


In case no specialized sampler is found, AeMCMC assigns the NUTS sampler to the
remaining variables. AeMCMC reparametrizes the model automatically to improve
sampling if needed:

.. code-block:: python

    import aemcmc
    import aesara.tensor as at

    srng = at.random.RandomStream(0)
    mu_rv = srng.normal(0, 1, name="mu")
    sigma_rv = srng.halfnormal(0.0, 1.0, name="sigma")
    Y_rv = srng.normal(mu_rv, sigma_rv, name="Y")

    y_vv = Y_rv.clone()

    sample_steps, updates, initial_values, parameters = aemcmc.construct_sampler(
        {Y_rv: y_vv}, srng
    )
    print(sample_steps.keys())
    # dict_keys([sigma, mu])
    print(parameters.keys())
    # dict_keys(['step_size', 'inverse_mass_matrix'])


Installation
============

The latest release of AeMCMC can be installed from PyPI using ``pip``:

::

    pip install aemcmc


Or via conda-forge:

::

    conda install -c conda-forge aemcmc


The current development branch of AeMCMC can be installed from GitHub, also using ``pip``:

::

    pip install git+https://github.com/aesara-devs/aemcmc



.. |Tests Status| image:: https://github.com/aesara-devs/aemcmc/workflows/Tests/badge.svg
  :target: https://github.com/aesara-devs/aemcmc/actions?query=workflow%3ATests
.. |Coverage| image:: https://codecov.io/gh/aesara-devs/aemcmc/branch/main/graph/badge.svg?token=45nKZ7fDG5
  :target: https://codecov.io/gh/aesara-devs/aemcmc
.. |Gitter| image:: https://badges.gitter.im/aesara-devs/aesara.svg
  :target: https://gitter.im/aesara-devs/aesara?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
