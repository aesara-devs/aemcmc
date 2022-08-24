from collections.abc import Mapping
from functools import wraps
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from aeppl.rewriting import PreserveRVMappings
from aesara.compile.builders import OpFromGraph
from aesara.compile.mode import optdb
from aesara.graph.basic import Apply, Variable, clone_replace, io_toposort
from aesara.graph.features import AlreadyThere, Feature
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op
from aesara.graph.rewriting.basic import in2out, node_rewriter
from aesara.graph.rewriting.db import SequenceDB
from aesara.tensor.elemwise import DimShuffle, Elemwise
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.random.utils import RandomStream
from aesara.tensor.rewriting.shape import ShapeFeature
from aesara.tensor.var import TensorVariable
from cons.core import _car
from unification.core import _unify

SamplerFunctionReturnType = Optional[
    Iterable[Tuple[Variable, Variable, Union[Dict[Variable, Variable]]]]
]
SamplerFunctionType = Callable[
    [FunctionGraph, Apply, RandomStream], SamplerFunctionReturnType
]
LocalRewriterReturnType = Optional[Union[Dict[Variable, Variable], Sequence[Variable]]]

sampler_ir_db = SequenceDB()
sampler_ir_db.name = "sampler_ir_db"
sampler_ir_db.register(
    "sampler_canonicalize",
    optdb.query("+canonicalize"),
    "basic",
)

sampler_rewrites_db = SequenceDB()
sampler_rewrites_db.name = "sampler_rewrites_db"

sampler_finder_db = SequenceDB()
sampler_finder_db.name = "sampler_finder_db"

sampler_rewrites_db.register(
    "sampler_finders",
    sampler_finder_db,
    "basic",
    position=0,
)


def construct_ir_fgraph(
    obs_rvs_to_values: Dict[Variable, Variable]
) -> Tuple[
    FunctionGraph,
    Dict[Variable, Variable],
    Dict[Variable, Variable],
    Dict[Variable, Variable],
]:
    r"""Construct a `FunctionGraph` in measurable IR form for the keys in `obs_rvs_to_values`.

    Returns
    -------
    A `FunctionGraph` of the measurable IR, a copy of `obs_rvs_to_values` containing
    the new, cloned versions of the original variables in `obs_rvs_to_values`, and
    a ``dict`` mapping all the original variables to their cloned values in
    `FunctionGraph`.
    """
    memo = {v: v for v in obs_rvs_to_values.values()}

    rv_outputs = tuple(
        node.outputs[1]
        for node in io_toposort([], list(obs_rvs_to_values.keys()))
        if isinstance(node.op, RandomVariable)
    )

    observed_vars = tuple(obs_rvs_to_values.keys())

    assert all(obs_rv in obs_rvs_to_values for obs_rv in observed_vars)

    fgraph = FunctionGraph(
        outputs=rv_outputs,
        clone=True,
        memo=memo,
        copy_orphans=False,
        copy_inputs=False,
        features=[ShapeFeature(), PreserveRVMappings(obs_rvs_to_values)],
    )

    # Update `obs_rvs_to_values` so that it uses the new cloned variables
    obs_rvs_to_values = {memo[k]: v for k, v in obs_rvs_to_values.items()}

    sampler_ir_db.query("+basic").rewrite(fgraph)

    new_to_old_rvs = {
        new_rv: old_rv for old_rv, new_rv in zip(rv_outputs, fgraph.outputs)
    }

    return fgraph, obs_rvs_to_values, memo, new_to_old_rvs


class SamplerTracker(Feature):
    """A `Feature` that tracks potential sampler steps in a graph."""

    def __init__(self, srng: RandomStream):
        self.srng: RandomStream = srng
        # Maps variables to a list of tuples that each provide a description of
        # the posterior step, the posterior step's graph/output variable, and
        # any updates generated for the posterior step
        self.rvs_to_samplers: Dict[
            TensorVariable,
            List[Tuple[str, TensorVariable, Optional[Dict[Variable, Variable]]]],
        ] = {}
        self.rvs_seen: Set[TensorVariable] = set()

    def on_attach(self, fgraph: FunctionGraph):
        if hasattr(fgraph, "sampler_mappings"):  # pragma: no cover
            raise AlreadyThere(
                f"{fgraph} already has the `SamplerTracker` feature attached."
            )

        fgraph.sampler_mappings = self


def sampler_finder(tracks: Optional[Sequence[Union[Op, type]]]):
    """Construct a `NodeRewriter` that identifies sample steps.

    This is a decorator that is used as follows:

        @sampler_finder([NormalRV])
        def local_horseshoe_posterior(fgraph, node, srng):
            # Determine if this normal is the root of a Horseshoe
            # prior graph.
            ...
            # If it is, construct the posterior steps for its parameters and
            # return them as a list of tuples like `(rv, posterior_rv, updates)`.
            ...
            return [(lambda_rv, lambda_posterior, None), (tau_rv, tau_posterior, None)]

    """

    def decorator(f: SamplerFunctionType):
        @node_rewriter(tracks)
        @wraps(f)
        def sampler_finder(
            fgraph: FunctionGraph, node: Apply
        ) -> LocalRewriterReturnType:
            sampler_mappings = getattr(fgraph, "sampler_mappings", None)

            # TODO: This assumes that `node` is a `RandomVariable`-generated `Apply` node
            rv_var = node.outputs[1]
            key = (f.__name__, rv_var)

            if sampler_mappings is None or key in sampler_mappings.rvs_seen:
                return None  # pragma: no cover

            srng = sampler_mappings.srng

            rvs_and_posteriors: SamplerFunctionReturnType = f(fgraph, node, srng)

            if not rvs_and_posteriors:
                return None  # pragma: no cover

            for rv, posterior_rv, updates in rvs_and_posteriors:
                sampler_mappings.rvs_to_samplers.setdefault(rv, []).append(
                    (f.__name__, posterior_rv, updates)
                )
            sampler_mappings.rvs_seen.add(key)

            return rv_var.owner.outputs

        return sampler_finder

    return decorator


class SubsumingElemwise(OpFromGraph, Elemwise):
    r"""A class representing an `Elemwise` with `DimShuffle`\ed arguments."""

    def __init__(self, inputs, outputs, *args, **kwargs):
        # TODO: Mock the `Elemwise` interface just enough for our purposes
        self.elemwise_op = outputs[0].owner.op
        self.scalar_op = self.elemwise_op.scalar_op
        self.nfunc_spec = self.elemwise_op.nfunc_spec
        self.inplace_pattern = self.elemwise_op.inplace_pattern
        # self.destroy_map = self.elemwise_op.destroy_map
        self.ufunc = None
        self.nfunc = None
        OpFromGraph.__init__(self, inputs, outputs, *args, **kwargs)

    def make_node(self, *inputs):
        node = super().make_node(*inputs)
        # Remove shared variable inputs.  We aren't going to compute anything
        # with this `Op`, so they're not needed
        real_inputs = node.inputs[: len(node.inputs) - len(self.shared_inputs)]
        return Apply(self, real_inputs, [o.clone() for o in node.outputs])

    def perform(self, *args, **kwargs):
        raise NotImplementedError(  # pragma: no cover
            "This `OpFromGraph` should have been in-line expanded."
        )

    def clone(self):
        res = OpFromGraph.clone(self)
        res.elemwise_op = self.elemwise_op
        res.scalar_op = self.scalar_op
        res.nfunc_spec = self.nfunc_spec
        res.inplace_pattern = self.inplace_pattern
        # res.destroy_map = self.destroy_map
        res.ufunc = self.ufunc
        res.nfunc = self.nfunc
        return res

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"{type(self).__name__}{{{self.scalar_op}}}"

    def __eq__(self, other):
        return OpFromGraph.__eq__(self, other)

    def __hash__(self):
        return OpFromGraph.__hash__(self)


def _unify_SubsumingElemwise(u: Elemwise, v: SubsumingElemwise, s: Mapping):
    yield _unify(u, v.elemwise_op, s)


_unify.add(
    (Elemwise, SubsumingElemwise, Mapping),
    lambda u, v, s: _unify_SubsumingElemwise(u, v, s),
)
_unify.add(
    (SubsumingElemwise, Elemwise, Mapping),
    lambda u, v, s: _unify_SubsumingElemwise(v, u, s),
)
_unify.add(
    (SubsumingElemwise, SubsumingElemwise, Mapping),
    lambda u, v, s: _unify(v.elemwise_op, u.elemwise_op, s),
)


def car_SubsumingElemwise(x):
    return type(x.elemwise_op)


_car.add((SubsumingElemwise,), car_SubsumingElemwise)


@node_rewriter([Elemwise])
def local_elemwise_dimshuffle_subsume(fgraph, node):
    r"""This rewrite converts `DimShuffle`s in the `Elemwise` inputs into a single `Op`.

    The replacement rule is

    .. math:

        \frac{
            \operatorname{Elemwise}_{o}\left(
                \operatorname{DimShuffle}_{z_i}(x_i), \dots
            \right)
        }{
            \operatorname{OpFromGraph}_{\operatorname{Elemwise}_{o}\left(
                \operatorname{DimShuffle}_{z_i}(y_i), \dots
            \right)}\left(
                x_i, \dots
            \right)
        }
        //, \quad
        // x_i \text{ is a } \operatorname{RandomVariable}

    where :math:`o` is a scalar `Op`, :math:`z_i` are the `DimShuffle` settings
    for the inputs at index :math:`i`.

    """

    if isinstance(node.op, SubsumingElemwise):
        return None

    new_inputs = []
    subsumed_inputs = []

    out_ndim = node.outputs[0].type.ndim

    found_subsumable_ds = False
    for i in node.inputs:
        if i.owner and isinstance(i.owner.op, DimShuffle):
            # TODO FIXME: Only do this when the `DimShuffle`s are adding
            # broadcastable dimensions.  If they're doing more
            # (e.g. transposing), separate the broadcasting from everything
            # else.
            ds_order = i.owner.op.new_order
            dim_shuffle_input = i.owner.inputs[0]

            ndim_diff = out_ndim - dim_shuffle_input.type.ndim

            # The `DimShuffle`ing added by `Elemwise`
            el_order = ds_order[:ndim_diff]
            # The remaining `DimShuffle`ing that was added by something else
            new_ds_order = ds_order[ndim_diff:]

            # Only consider broadcast dimensions added on the left as
            # having come from `Elemwise.make_node`
            if len(el_order) == 0 or not all(d == "x" for d in el_order):
                # In this case, the necessary broadcast elements were most
                # likely not added by `Elemwise.make_node` (e.g. broadcasts are
                # interspersed with transposes, or there are none at all), so
                # we don't want to mess with them.
                # TODO: We could still subsume some of these `DimShuffle`s,
                # though
                subsumed_inputs.append(i)
                new_inputs.append(i)
                continue

            # if dim_shuffle_input.owner and isinstance(
            #     dim_shuffle_input.owner.op, RandomVariable
            # ):
            found_subsumable_ds = True

            if new_ds_order and not new_ds_order == tuple(range(len(new_ds_order))):
                # The remaining `DimShuffle`ing is substantial, so we need to
                # apply it separately
                new_dim_shuffle_input = dim_shuffle_input.dimshuffle(new_ds_order)
                new_subsumed_input = new_dim_shuffle_input.dimshuffle(
                    el_order + tuple(range(new_dim_shuffle_input.type.ndim))
                )

                subsumed_inputs.append(new_subsumed_input)
                new_inputs.append(new_dim_shuffle_input)
            else:
                subsumed_inputs.append(i)
                new_inputs.append(dim_shuffle_input)

        else:
            subsumed_inputs.append(i)
            new_inputs.append(i)

    if not found_subsumable_ds:
        return None  # pragma: no cover

    assert len(subsumed_inputs) == len(node.inputs)
    new_outputs = node.op.make_node(*subsumed_inputs).outputs
    new_op = SubsumingElemwise(new_inputs, new_outputs, inline=True)

    new_out = new_op(*new_inputs)

    assert len(new_out.owner.inputs) == len(node.inputs)

    return new_out.owner.outputs


sampler_ir_db.register(
    "elemwise_dimshuffle_subsume",
    in2out(local_elemwise_dimshuffle_subsume),
    "basic",
    position=-10,
)


@node_rewriter([Elemwise])
def inline_SubsumingElemwise(fgraph, node):

    op = node.op

    if not isinstance(op, SubsumingElemwise):
        return False

    if not op.is_inline:
        return False  # pragma: no cover

    res = clone_replace(
        op.inner_outputs, {u: v for u, v in zip(op.inner_inputs, node.inputs)}
    )

    return res


expand_subsumptions = in2out(inline_SubsumingElemwise)

# This step undoes `elemwise_dimshuffle_subsume`
sampler_rewrites_db.register(
    "expand_subsumptions",
    expand_subsumptions,
    "basic",
    position="last",
)
