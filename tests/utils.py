from aesara.compile.ops import UpdatePlaceholder
from aesara.compile.sharedvalue import SharedVariable
from aesara.graph.basic import walk
from aesara.graph.op import HasInnerGraph
from aesara.tensor.random.op import RandomVariable


def assert_consistent_rng_updates(var):
    r"""Assert that `RandomTypeSharedVariable`\s updates are associated with their assigned `RandomVariable`\s."""

    def expand(r):
        owner = r.owner
        if owner:
            inputs = list(reversed(owner.inputs))

            if isinstance(owner.op, HasInnerGraph):
                inputs += owner.op.inner_outputs

            return inputs

    for v in walk([var], expand, False):
        if v.owner and isinstance(v.owner.op, RandomVariable):
            rng = v.owner.inputs[0]
            if (
                isinstance(rng, SharedVariable)
                and rng.default_update
                and (
                    # We ignore `OpFromGraph`/inner-graph update placeholders
                    not rng.default_update.owner
                    or not isinstance(rng.default_update.owner.op, UpdatePlaceholder)
                )
            ):
                assert rng.default_update == v.owner.outputs[0]
