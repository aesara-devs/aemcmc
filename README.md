<div align="center">


<img src="./docs/source/_static/aemcmc_logo.png#gh-light-mode-only" alt="AeMCMC Logo Dark" width=400></img>
<img src="./docs/source/_static/aemcmc_logo_dark.png#gh-dark-mode-only" alt="AeMCMC Logo Dark" width=400></img>

# AeMCMC

[![Pypi][pypi-badge]][pypi]
[![Gitter][gitter-badge]][gitter]
[![Discord][discord-badge]][discord]
[![Twitter][twitter-badge]][twitter]

AeMCMC automatically constructs samplers for probabilistic models written in [Aesara](https://github.com/aesara-devs/aesara).

*A compiler for Bayesian inference.*

[Features](#features) •
[Get started](#get-started) •
[Install](#install) •
[Get help](#get-help) •
[Contribute](#contribute)

</div>

## Features

This project is currently in an alpha state, but the core objectives are as follows:

- Provide utilities that simplify the process of constructing Aesara graphs/functions for posterior and posterior predictive sampling
- Host a wide array of "exact" posterior sampling steps (e.g. Gibbs steps, scale-mixture/decomposition-based conditional samplers, etc.)
- Build a framework for identifying and composing said sampler steps and enumerating the possible samplers for an arbitrary model

Overall, we would like this project to serve as a hub for community-sourced specialized samplers and facilitate their general use.

## Get started

Using AeMCMC, one can construct sampling steps from a graph containing Aesara
`RandomVariable`s. AeMCMC analyzes the model graph and possibly rewrites it
to find the most suitable sampler.

AeMCMC can recognize closed-form posteriors; for instance the following
Beta-Binomial model amounts to sampling from a Beta distribution:

``` python
import aesara
import aemcmc
import aesara.tensor as at

srng = at.random.RandomStream(0)

p_rv = srng.beta(1., 1., name="p")
Y_rv = srng.binomial(10, p_rv, name="Y")

y_vv = Y_rv.clone()
y_vv.name = "y"

sampler, initial_values = aemcmc.construct_sampler({Y_rv: y_vv}, srng)

p_posterior_step = sampler.sample_steps[p_rv]
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
```


AeMCMC also contains a database of Gibbs samplers that can be used to sample
some models more efficiently than a general-purpose sampler like NUTS
would:

``` python
import aemcmc
import aesara
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

sampler, initial_values = aemcmc.construct_sampler({Y_rv: y_vv}, srng)

# `sampler.sample_steps` contains the sample step for each random variable
print(sampler.sample_steps[h_rv])
# h_posterior

# `sampler.stages` contains the sampling kernels sorted by scan order
print(sampler.stages)
# {HorseshoeGibbsKernel: [tau, lambda], NBRegressionGibbsKernel: [beta], DispersionGibbsKernel: [h]}

# Build a function that returns new samples
to_sample_rvs = [tau_rv, lmbda_rv, beta_rv, h_rv]
inputs = [a, b, X, y_vv] + [initial_values[rv] for rv in to_sample_rvs]
outputs = [sampler.sample_steps[rv] for rv in to_sample_rvs]
sample_fn = aesara.function(inputs, outputs, updates=sampler.updates)
```

In case no specialized sampler is found, AeMCMC assigns the NUTS sampler to the
remaining variables. AeMCMC reparametrizes the model automatically to improve
sampling if needed:

``` python
import aemcmc
import aesara
import aesara.tensor as at

srng = at.random.RandomStream(0)
mu_rv = srng.normal(0, 1, name="mu")
sigma_rv = srng.halfnormal(0.0, 1.0, name="sigma")
Y_rv = srng.normal(mu_rv, sigma_rv, name="Y")

y_vv = Y_rv.clone()

sampler, initial_values = aemcmc.construct_sampler({Y_rv: y_vv}, srng)

print(sampler.sample_steps.keys())
# dict_keys([sigma, mu])
print(sampler.stages)
# {NUTSKernel: [sigma, mu]}
print(sampler.parameters)
# {NUTSKernel: (step_size, inverse_mass_matrix)}

# Build a function that returns new samples
step_size, inverse_mass_matrix = list(sampler.parameters.values())[0]
inputs = [
    initial_values[mu_rv],
    initial_values[sigma_rv],
    y_vv,
    step_size,
    inverse_mass_matrix
]
outputs = [sampler.sample_steps[mu_rv], sampler.sample_steps[sigma_rv]]
sample_fn = aesara.function(inputs, outputs, updates=sampler.updates)
```



## Install

The latest release of AeMCMC can be installed from PyPI using `pip`:

``` bash
pip install aemcmc
```

Or via conda-forge:

``` bash
conda install -c conda-forge aemcmc
```

The nightly (bleeding edge) version of `aemcmc` can be installed using `pip`:

``` bash
pip install aemcmc-nightly
```

## Get help

Report bugs by opening an [issue][issues]. If you have a question regarding the usage of AeMCMC, start a [discussion][discussions]. For real-time feedback or more general chat about AeMCMC use our [Discord server][discord] or [Gitter room][gitter].

## Contribute

AeMCMC welcomes contributions. A good place to start contributing is by looking at the [issues][issues].

If you want to implement a new feature, open a [discussion][discussions] or come chat with us on [Discord][discord] or [Gitter][gitter].


[contributors]: https://github.com/aesara-devs/aemcmc/graphs/contributors
[contributors-badge]: https://img.shields.io/github/contributors/aesara-devs/aemcmc?style=flat-square&logo=github&logoColor=white&color=ECEFF4
[discussions]: https://github.com/aesara-devs/aemcmc/discussions
[documentation-examples]: https://aemcmc.readthedocs.io/en/latest/examples.html
[downloads-badge]: https://img.shields.io/pypi/dm/aemcmc?style=flat-square&logo=pypi&logoColor=white&color=8FBCBB
[discord]: https://discord.gg/h3sjmPYuGJ
[discord-badge]: https://img.shields.io/discord/1072170173785723041?color=81A1C1&logo=discord&logoColor=white&style=flat-square
[gitter]: https://gitter.im/aesara-devs/aemcmc
[gitter-badge]: https://img.shields.io/gitter/room/aesara-devs/aemcmc?color=81A1C1&logo=matrix&logoColor=white&style=flat-square
[issues]: https://github.com/aesara-devs/aemcmc/issues
[releases]: https://github.com/aesara-devs/aemcmc/releases
[twitter]: https://twitter.com/AesaraDevs
[twitter-badge]: https://img.shields.io/twitter/follow/AesaraDevs?style=social
[pypi]: https://pypi.org/project/aemcmc/
[pypi-badge]: https://img.shields.io/pypi/v/aemcmc?color=ECEFF4&logo=python&logoColor=white&style=flat-square
