from typing import List

from aesara.compile.builders import OpFromGraph
from aesara.tensor.var import TensorVariable


class SamplingStep(OpFromGraph):
    """An AeMCMC sampling step.

    Sampling steps represent the computation necessary to update the value of a
    random variables with a Markov kernel. They are represented as `Op`s with an
    inner graph.

    """

    def __init__(self, inputs: List[TensorVariable], outputs: List[TensorVariable]):
        super().__init__(inputs, outputs, inline=True)
