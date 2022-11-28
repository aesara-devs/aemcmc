from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from aesara.compile.builders import OpFromGraph
from aesara.graph.basic import Variable
from aesara.tensor.var import TensorVariable


class SamplingStep(OpFromGraph):
    """An AeMCMC sampling step.

    Sampling steps represent the computation necessary to update the value of a
    random variables with a Markov kernel. They are represented as `Op`s with an
    inner graph.

    """

    def __init__(self, inputs: List[TensorVariable], outputs: List[TensorVariable]):
        super().__init__(inputs, outputs, inline=True)


@dataclass(frozen=True)
class Sampler:
    """A class that tracks sampling steps and their parameters."""

    sample_steps: Dict[TensorVariable, TensorVariable]
    """A map between measures and their updated value under the current sampling scheme."""
    updates: Optional[Dict[Variable, TensorVariable]] = field(default_factory=dict)
    """Updates to be passed to `aesara.function`"""
    parameters: Dict[SamplingStep, Tuple[TensorVariable]] = field(default_factory=dict)
    """Parameters needed by the sampling steps."""

    stages: Dict[SamplingStep, List[TensorVariable]] = field(init=False)
    """A list of the sampling stages sorted in scan order."""

    def __post_init__(self):

        # The scan order corresponds to the order in which the sampling
        # steps were found, and thus the order in which elements were
        # inserted in `sample_steps`.
        stages = defaultdict(list)
        for rv, updated_rv in self.sample_steps.items():
            kernel = updated_rv.owner.op
            stages[kernel].append(rv)
        super().__setattr__("stages", stages)
