from . import _version

__version__ = _version.get_versions()["version"]


from aemcmc.basic import construct_sampler
from aemcmc.sample import sample_prior

# isort: off
# Register rewrite databases
import aemcmc.conjugates
import aemcmc.gibbs

# isort: on
