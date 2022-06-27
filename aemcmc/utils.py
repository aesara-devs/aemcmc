from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from aesara.graph.basic import Variable
from aesara.tensor.var import TensorVariable


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
