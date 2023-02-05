Build a sampler for your model
==============================

.. autofunction:: aemcmc.basic.construct_sampler

The Sampler object
------------------

``construct_sampler`` returns a ``Sampler`` object that contains the graphs for the variables' sampling steps and the updates to pass to `aesara.function`:

.. autoclass:: aemcmc.types.Sampler
   :members:

