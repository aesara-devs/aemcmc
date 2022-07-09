from . import _version

__version__ = _version.get_versions()["version"]

# Register rewrite databases
import aemcmc.conjugates
import aemcmc.gibbs
