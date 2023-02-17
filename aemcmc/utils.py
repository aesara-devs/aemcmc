from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from aesara.compile.sharedvalue import SharedVariable
from aesara.graph.basic import Constant, Variable, ancestors
from aesara.tensor import as_tensor_variable
from aesara.tensor.random.type import RandomType
from aesara.tensor.var import TensorVariable

if TYPE_CHECKING:
    from aesara.tensor.random.utils import RandomStream


@dataclass(frozen=True)
class ModelInfo:
    """A class that tracks sampler-specific variable types, names, and updates for a model graph."""

    observed_rvs: Tuple[TensorVariable, ...]
    """Observed random/measurable variables."""
    rvs_to_values: Dict[TensorVariable, TensorVariable]
    """A map between random/measurable variables and their value variables."""
    deterministic_vars: Tuple[TensorVariable, ...] = field(default_factory=tuple)
    """Stochastic variables that are tracked but not sampled directly/explicitly."""
    updates: Optional[Dict[Variable, TensorVariable]] = field(default_factory=dict)
    """Updates to be passed to `aesara.function`."""

    values_to_rvs: Dict[TensorVariable, TensorVariable] = field(init=False)
    """The inverse of `rvs_to_values`."""
    names_to_vars: Dict[str, TensorVariable] = field(init=False)
    observed_values: Tuple[TensorVariable, ...] = field(init=False)
    unobserved_rvs: Tuple[TensorVariable] = field(init=False)
    """Random/measurable variables that are neither observable nor deterministic."""
    unobserved_values: Tuple[TensorVariable] = field(init=False)
    """The value variables associated with `unobserved_rvs`."""

    def __post_init__(self):
        super().__setattr__(
            "values_to_rvs", {v: k for k, v in self.rvs_to_values.items()}
        )

        all_rvs = set(self.observed_rvs)
        all_rvs.update(self.rvs_to_values.keys())

        super().__setattr__(
            "unobserved_rvs",
            tuple(k for k in self.rvs_to_values.keys() if k not in self.observed_rvs),
        )
        super().__setattr__(
            "unobserved_values",
            tuple(self.rvs_to_values[k] for k in self.unobserved_rvs),
        )
        super().__setattr__(
            "observed_values",
            tuple(self.rvs_to_values[k] for k in self.observed_rvs),
        )

        all_vars = (
            all_rvs
            | set(self.deterministic_vars)
            | set(self.observed_values)
            | set(self.unobserved_values)
        )

        if not all(v.name for v in all_vars):
            raise ValueError("All variables in the model must have non-empty names")

        super().__setattr__("names_to_vars", {v.name: v for v in all_vars})

        if len(self.names_to_vars) != len(all_vars):
            raise ValueError("All variables in the model must have unique names")


def remove_constants(inputs):
    res = []
    for inp in inputs:
        inp_t = as_tensor_variable(inp)
        if not isinstance(inp_t, Constant):
            res.append(inp_t)

    return res


def get_rv_updates(
    srng: "RandomStream", *rvs: TensorVariable
) -> Dict[SharedVariable, "Variable"]:
    r"""Get the updates needed to update RNG objects during sampling of `rvs`.

    A search is performed over `rvs` for `SharedVariable`\s with default
    updates and the updates stored in `srng`.

    Parameters
    ----------
    srng:
        `RandomStream` instance with which the model was defined.
    rvs:
        The random variables whose prior distribution we want to sample.

    Returns
    -------
    A dict containing the updates needed to sample from the models given by
    `rvs`.

    """
    # TODO: It's kind of weird that this is an alist-like data structure; we
    # should revisit this in `RandomStream`
    srng_updates = dict(srng.state_updates)
    rv_updates = {}

    for var in ancestors(rvs):
        if not isinstance(var, SharedVariable) and not isinstance(var.type, RandomType):
            continue

        # TODO: Consider making sure the updates correspond to "in-place"
        # updates of the RNGs for relevant `RandomVariable`s?
        # More generally, a function like this could be used to determine the
        # consistency of `RandomVariable` updates in general (e.g. find
        # bad/disassociated updates).
        srng_update = srng_updates.get(var)

        if var.default_update:
            if srng_update:
                assert srng_update == var.default_update

            # We prefer the default update (for no particular reason)
            rv_updates[var] = var.default_update
        elif srng_update:
            rv_updates[var] = srng_update

    return rv_updates
