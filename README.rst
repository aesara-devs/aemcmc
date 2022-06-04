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

TODO

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
