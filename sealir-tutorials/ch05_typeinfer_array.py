# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Chapter 5: Type Inference for Array Operations
#
# This chapter extends the type inference system to handle array operations,
# including array indexing, broadcasting, and shape inference. We show how
# to encode array metadata in the e-graph and implement type inference rules
# for array operations.
#
# The chapter covers:
# - How to represent array types with dimensions and shapes
# - How to implement array indexing operations
# - How to handle array broadcasting and shape inference

# ## Imports and Setup

from __future__ import annotations

import ctypes

import numpy as np
import sealir.rvsdg.grammar as rg
from egglog import (
    BoolLike,
    EGraph,
    Expr,
    String,
    StringLike,
    Unit,
    Vec,
    delete,
    function,
    i64,
    i64Like,
    method,
    rewrite,
    rule,
    ruleset,
    set_,
    subsume,
    union,
)
from llvmlite import ir
from sealir.eqsat.py_eqsat import (
    Py_SubscriptIO,
)
from sealir.eqsat.rvsdg_eqsat import (
    Term,
)

from ch04_1_typeinfer_ifelse import (
    Grammar,
    NbOp_Type,
)
from ch04_2_typeinfer_loops import Backend as _ch04_2_Backend
from ch04_2_typeinfer_loops import (
    ExtendEGraphToRVSDG as _ch04_2_ExtendEGraphToRVSDG,
)
from ch04_2_typeinfer_loops import (
    Int64,
    MyCostModel,
    NbOp_Base,
    SExpr,
    Type,
    TypeInt64,
    TypeVar,
    base_ruleset,
)
from ch04_2_typeinfer_loops import finalize_ruleset as _ch04_2_finalize_ruleset
from ch04_2_typeinfer_loops import (
    jit_compiler,
    setup_argtypes,
)
from utils import IN_NOTEBOOK

# ## Array Type Definitions
#
# Define the `ArrayDesc` to describe metadata for an Array type. The Array
# type is more interesting because it is not a simple scalar values. The
# array type has attributes like data-type, number of dimensions, shape and
# data-layout. Shape of an array can be statically known to be a fixed
# integer, or it can be symbolic.

# ### Dimension Representation
#
# Define Dim for the shape info at each dimension


class Dim(Expr):
    @classmethod
    def fixed(self, size: i64Like) -> Dim: ...
    @classmethod
    def symbolic(self, unque_id: StringLike) -> Dim: ...


# ### Data Layout Representation
#
# Define DataLayout for array memory layout


class DataLayout(Expr):
    @classmethod
    def c_contiguous(cls) -> DataLayout: ...
    @classmethod
    def fortran_contiguous(cls) -> DataLayout: ...
    @classmethod
    def strided(cls) -> DataLayout: ...


# ### Array Description
#
# Define ArrayDesc to represent array metadata. Note that `ArrayDesc` is
# convertible to `Type`.


class ArrayDesc(Expr):
    @method(cost=10000)
    def __init__(self, uid: StringLike): ...

    @property
    def dtype(self) -> Type: ...

    @property
    def ndim(self) -> i64: ...

    def dim(self, idx: i64Like) -> Dim: ...

    @property
    def dataLayout(self) -> DataLayout: ...

    def toType(self) -> Type: ...


# ## Array Type Examples
#
# Demonstrate how to set up array types with different properties.

# Example: set the dtype

if __name__ == "__main__":

    array0 = ArrayDesc(uid="array0")
    eg = EGraph()
    eg.let("array0", array0)
    eg.register(set_(array0.dtype).to(TypeInt64))
    if IN_NOTEBOOK:
        eg.display(graphviz=True)

# Example: set the shape

if __name__ == "__main__":
    # array0 is M x N x 4
    eg.register(
        set_(array0.ndim).to(3),
        set_(array0.dim(0)).to(Dim.symbolic("M")),
        set_(array0.dim(1)).to(Dim.symbolic("N")),
        set_(array0.dim(2)).to(Dim.fixed(4)),
    )
    if IN_NOTEBOOK:
        eg.display(graphviz=True)

# Example: set the data-layout

if __name__ == "__main__":
    eg.register(
        set_(array0.dataLayout).to(DataLayout.c_contiguous()),
    )
    if IN_NOTEBOOK:
        eg.display(graphviz=True)

# ## Symbolic Dimension Merging
#
# Demonstrate how symbolic dimensions can be merged and resolved.

# introduce a new array `array1`

if __name__ == "__main__":
    # array1 is 10 x 24 x K
    array1 = ArrayDesc(uid="array1")
    eg.register(
        set_(array1.ndim).to(3),
        set_(array1.dim(0)).to(Dim.fixed(10)),
        set_(array1.dim(1)).to(Dim.fixed(24)),
        set_(array1.dim(2)).to(Dim.symbolic("K")),
    )
    if IN_NOTEBOOK:
        eg.display(graphviz=True)

# Merging `array0` with `array1` will also propagate equivalences to the
# `.dim()`. This will make shape inference trivial to implement.

if __name__ == "__main__":
    eg.register(union(array0).with_(array1))
    eg.run(1)
    if IN_NOTEBOOK:
        eg.display(graphviz=True)

    # check that the Dim merged
    eg.check(array0.dim(0) == array1.dim(0))
    eg.check(array0.dim(1) == array1.dim(1))
    eg.check(array0.dim(2) == array1.dim(2))

    # Now we know the symbolic shape
    eg.check(Dim.symbolic("M") == Dim.fixed(10))
    eg.check(Dim.symbolic("N") == Dim.fixed(24))
    eg.check(Dim.symbolic("K") == Dim.fixed(4))

# ## Compiler Extensions for Arrays
#
# Extend the compiler for Array implementation


class NbOp_ArrayDimFixed(NbOp_Base):
    size: int


class NbOp_ArrayDimSymbolic(NbOp_Base):
    name: str


class NbOp_ArrayType(NbOp_Base):
    dtype: NbOp_Type
    ndim: int
    datalayout: str
    shape: tuple[SExpr, ...]


# ## Example 1: Array Indexing
#
# Implement `Array.__getitem__` operation


def example_1(ary, idx):
    return ary[idx]


array_1d_symbolic = NbOp_ArrayType(
    dtype=Int64,
    ndim=1,
    datalayout="c_contiguous",
    shape=(NbOp_ArrayDimSymbolic("m"),),
)

# ### E-Graph Rules for Array Operations
#
# Define egraph rules for the array operation


def array_desc_rules(
    uid: str, shape: tuple[int | str, ...], dtype: Type, layout: str
):
    desc = ArrayDesc(uid=uid)
    rules = []
    rules.append(set_(desc.ndim).to(i64(len(shape))))
    for i, d in enumerate(shape):
        match d:
            case str(k):
                dim = Dim.symbolic(k)
            case int(n):
                dim = Dim.fixed(n)
            case _:
                raise ValueError
        rules.append(set_(desc.dim(i)).to(dim))

    match layout.lower():
        case "c":
            dl = DataLayout.c_contiguous()
        case "f":
            dl = DataLayout.fortran_contiguous()
        case "s":
            dl = DataLayout.strided()
        case _:
            raise ValueError
    rules.append(set_(desc.dataLayout).to(dl))
    rules.append(set_(desc.dtype).to(dtype))

    the_rule = rule(desc).then(*rules)
    return desc, [the_rule]


@ruleset
def ruleset_typeinfer_array_getitem(
    getitem: Term,
    io: Term,
    ary: Term,
    index: Term,
    ty: Type,
    ary_uid: String,
    arydesc: ArrayDesc,
    itemty: Type,
):
    yield rule(
        # Implement getitem(int)->scalar
        getitem == Py_SubscriptIO(io, ary, index),
        # ary is array type
        ty == TypeVar(ary).getType(),
        ty == arydesc.toType(),
        # index is int type
        TypeVar(index).getType() == TypeInt64,
        # then ary must be 1D
        arydesc.ndim == i64(1),
        # get item type
        itemty == arydesc.dtype,
    ).then(
        # shortcut IO
        union(getitem.getPort(0)).with_(io),
        # Rewrite operation
        union(getitem.getPort(1)).with_(
            Nb_Array_1D_Getitem_Scalar(io, ary, index, itemty)
        ),
        # Return type is int64
        set_(TypeVar(getitem.getPort(1)).getType()).to(itemty),
    )


@function
def FlattenArrayType(ndim: i64, dtype: Type, dataLayout: DataLayout) -> Type: ...


@ruleset
def ruleset_finalize_arraydesc(
    ad: ArrayDesc,
    ndim: i64,
    dtype: Type,
    datalayout: DataLayout,
):
    yield rewrite(ad.toType(), subsume=True).to(
        FlattenArrayType(ndim, dtype, datalayout),
        ndim == ad.ndim,
        dtype == ad.dtype,
        datalayout == ad.dataLayout,
    )


@function
def Nb_Array_1D_Getitem_Scalar(
    io: Term, ary: Term, index: Term, dtype: Type
) -> Term: ...


class NbOp_Array_1D_Getitem_Scalar(NbOp_Base):
    io: SExpr
    ary: SExpr
    index: SExpr
    attr: SExpr


# ### Extend E-Graph Extraction
#
# Extend egraph extraction to handle array operations


class ExtendEGraphToRVSDG(_ch04_2_ExtendEGraphToRVSDG):
    def handle_Term(self, op: str, children: dict | list, grm: Grammar):
        match op, children:
            case "Nb_Array_1D_Getitem_Scalar", {
                "io": io,
                "ary": ary,
                "index": index,
                "dtype": dtype,
            }:
                return grm.write(
                    NbOp_Array_1D_Getitem_Scalar(
                        io=io,
                        ary=ary,
                        index=index,
                        attr=grm.write(rg.Attrs(dtype)),
                    )
                )
        return super().handle_Term(op, children, grm)

    def handle_Type(
        self, key: str, op: str, children: dict | list, grm: Grammar
    ):
        match op, children:
            case "FlattenArrayType", {
                "ndim": ndim,
                "dtype": dtype,
                "dataLayout": dataLayout,
            }:
                return grm.write(
                    NbOp_ArrayType(
                        ndim=ndim,
                        dtype=dtype,
                        datalayout=dataLayout.name,
                        shape=(-1,) * ndim,
                    )
                )
            case _:
                return super().handle_Type(key, op, children, grm)


# ### Extend the LLVM Backend
#
# Extend the LLVM backend for array operations


class Backend(_ch04_2_Backend):

    def lower_type(self, ty: NbOp_Type):
        match ty:
            case NbOp_ArrayType(
                dtype=dtype,
                ndim=int(ndim),
                datalayout=str(datalayout),
                shape=shape,
            ):
                ll_dtype = self.lower_type(dtype)
                ptr = ll_dtype.as_pointer()
                shape_array = ir.ArrayType(ir.IntType(64), ndim)
                return ir.LiteralStructType([ptr, shape_array]).as_pointer()

        return super().lower_type(ty)

    def lower_expr(self, expr, state):
        builder = state.builder
        match expr:
            case NbOp_Array_1D_Getitem_Scalar(
                io=io, ary=ary, index=index, attr=attr
            ):
                io = yield io
                ary = yield ary
                index = yield index
                match attr:
                    case rg.Attrs((NbOp_Type(str(typename)),)):
                        pass
                    case _:
                        assert False, attr
                arystruct = builder.load(ary)
                dataptr = builder.extract_value(arystruct, 0)
                ptr_offset = builder.gep(dataptr, [index])
                return builder.load(ptr_offset)

        return (yield from super().lower_expr(expr, state))

    def get_ctype(self, lltype: ir.Type):
        match lltype:
            case ir.PointerType():
                # pointer will be void*
                return ctypes.c_void_p()

        return super().get_ctype(lltype)


# ### C-Types Definition for Array
#
# Define ctypes for array handling


class CtypeInt64Array1D(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_void_p), ("shape", (ctypes.c_uint64 * 1))]


array_int64_1d, array_infos = array_desc_rules(
    "array_int64_1d", shape=("n",), dtype=TypeInt64, layout="c"
)


finalize_ruleset = _ch04_2_finalize_ruleset | ruleset_finalize_arraydesc

compiler_config = dict(
    converter_class=ExtendEGraphToRVSDG,
    backend=Backend(),
    cost_model=MyCostModel(),
    verbose=True,
)

if __name__ == "__main__":
    # compile
    cres = jit_compiler(
        fn=example_1,
        argtypes=(array_1d_symbolic, Int64),
        # Rule schedule combines array type inference rules:
        # - Base rules with argument types
        # - Array metadata and shape information
        # - Array getitem operations
        # First saturate array inference, then finalize
        rule_schedule=(
            base_ruleset
            | setup_argtypes(array_int64_1d.toType(), TypeInt64)
            | ruleset(*array_infos)
            | ruleset_typeinfer_array_getitem
        ).saturate()
        + finalize_ruleset.saturate(),
        **compiler_config,
    )
    jit_func = cres.jit_func
    # create array
    ary = np.arange(10, dtype=np.int64)
    # prepare array for passing to C-API
    param_ary = CtypeInt64Array1D()
    param_ary.ptr = ary.ctypes.data
    param_ary.shape[0] = ary.shape[0]
    # call the compiled function
    got = jit_func(ctypes.byref(param_ary), 3)
    print("got", got)
    # compare the result
    expect = example_1(ary, 3)
    assert got == expect

# ## Example 2: 1D Array Summation
#
# This example works without any new extension


def example_2(ary, size):
    i = 0
    c = 0
    while i < size:
        c = c + ary[i]
        i = i + 1
    return c


if __name__ == "__main__":
    cres = jit_compiler(
        fn=example_2,
        argtypes=(array_1d_symbolic, Int64),
        rule_schedule=(
            base_ruleset
            | setup_argtypes(array_int64_1d.toType(), TypeInt64)
            | ruleset(*array_infos)
            | ruleset_typeinfer_array_getitem
        ).saturate()
        + finalize_ruleset.saturate(),
        **compiler_config,
    )
    jit_func = cres.jit_func

    ary = np.arange(10, dtype=np.int64)
    param_ary = CtypeInt64Array1D()
    param_ary.ptr = ary.ctypes.data
    param_ary.shape[0] = ary.shape[0]

    got = jit_func(ctypes.byref(param_ary), ary.size)
    print("got", got)
    expect = example_2(ary, ary.size)
    assert got == expect

# ## Broadcasting Logic
#
# Broadcasting can be implemented as declarative logic in the egraph. Let's
# start with an example:

if __name__ == "__main__":
    eg = EGraph()

    # array0 is M x N x 10 x 4
    array0 = ArrayDesc(uid="array0")
    eg.register(
        set_(array0.dtype).to(TypeInt64),
        set_(array0.ndim).to(4),
        set_(array0.dim(0)).to(Dim.symbolic("M")),
        set_(array0.dim(1)).to(Dim.symbolic("N")),
        set_(array0.dim(2)).to(Dim.fixed(10)),
        set_(array0.dim(3)).to(Dim.fixed(4)),
    )

    # array1 is 1 x 4
    array1 = ArrayDesc(uid="array1")
    eg.let("array1", array1)
    eg.register(
        set_(array1.dtype).to(TypeInt64),
        set_(array1.ndim).to(2),
        set_(array1.dim(0)).to(Dim.fixed(1)),
        set_(array1.dim(1)).to(Dim.fixed(4)),
    )

    if IN_NOTEBOOK:
        eg.display(graphviz=True)

# ### Define Broadcast Function
#
# Define the Broadcast function for array broadcasting


@function
def Broadcast(x: ArrayDesc, y: ArrayDesc) -> ArrayDesc: ...


if __name__ == "__main__":
    broadcasted = Broadcast(array0, array1)
    eg.let("broadcasted", broadcasted)

    if IN_NOTEBOOK:
        eg.display(graphviz=True)

# ### Define Broadcasting Logic
#
# Two arrays can be broadcasted together when:
#
# - The corresponding dimensions are either the same or are both one.
# - If number of dimensions mismatch, the lesser one gets new dimensions of
#   shape 1 added to the left.


@function
def ArrayAddDim(x: ArrayDesc, nd_diff: i64) -> ArrayDesc:
    "Creates a new ArrayDesc with `nd_diff` new dimension on the left."
    ...


@function
def AddLeftDim(x: ArrayDesc, dim: Dim) -> ArrayDesc:
    "Create a new ArrayDesc with one dimension specified by `dim`."
    ...


@function
def CopyDim(
    src: ArrayDesc, dst: ArrayDesc, start: i64Like, offset: i64Like
) -> Unit:
    "Set dst.dim(start) to src.dim(start - offset)"
    ...


@function
def CheckBroadcast(x: ArrayDesc, y: ArrayDesc, res: ArrayDesc) -> Unit:
    """Apply CheckBroadcastDim to all dimensions
    Require x.ndim == y.ndim
    """
    ...


@function
def CheckBroadcastDim(
    x: ArrayDesc, y: ArrayDesc, res: ArrayDesc, i: i64Like
) -> Unit:
    "Check x.dim(i) can be broadcasted to y.dim(i)"
    ...


@ruleset
def ruleset_broadcasting(
    x: ArrayDesc,
    y: ArrayDesc,
    z: ArrayDesc,
    nd: i64,
    dim: Dim,
    offset: i64,
    start: i64,
    nd_diff: i64,
):
    yield rule(
        # X has more dimension
        z == (bc := Broadcast(x, y)),
        nd == x.ndim,
        nd > y.ndim,
        nd_diff == nd - y.ndim,
    ).then(
        # subsume(bc),
        union(z).with_(Broadcast(x, ArrayAddDim(y, nd_diff))),
    )

    yield rewrite(
        # Swap left right argument
        Broadcast(x, y)
    ).to(Broadcast(y, x))

    yield rule(
        # X and Y has same ndim
        z == Broadcast(x, y),
        y.ndim == x.ndim,
        nd == x.ndim,
    ).then(
        CheckBroadcast(x, y, z),
        set_(z.ndim).to(nd),
    )

    yield rewrite(
        CheckBroadcast(x, y, z),
        subsume=True,
    ).to(
        # Start check at dim(0)
        CheckBroadcastDim(x, y, z, 0)
    )

    yield rule(
        CheckBroadcastDim(x, y, z, offset),
        offset + 1 < z.ndim,  # in range?
    ).then(
        # Advance to the next dim
        CheckBroadcastDim(x, y, z, offset + 1)
    )

    # Dimension checks
    yield rule(
        # same dim
        delme := CheckBroadcastDim(x, y, z, offset),
        x.dim(offset) == y.dim(offset),
        dim == x.dim(offset),
    ).then(delete(delme), set_(z.dim(offset)).to(dim))
    yield rule(
        # not the same dim (left is 1)
        delme := CheckBroadcastDim(x, y, z, offset),
        x.dim(offset) == Dim.fixed(1),
        dim == y.dim(offset),
    ).then(delete(delme), set_(z.dim(offset)).to(dim))

    # Logic to add dimensions
    yield rewrite(
        ArrayAddDim(x, nd_diff),
        subsume=True,
    ).to(
        # Add one dimension at a time.
        ArrayAddDim(AddLeftDim(x, Dim.fixed(1)), nd_diff - 1),
        nd_diff > 0,
    )

    yield rewrite(
        ArrayAddDim(x, nd_diff),
        subsume=True,
    ).to(
        # Reached the end
        x,
        nd_diff == i64(0),
    )

    yield rule(
        y == AddLeftDim(x, dim),
        nd == x.ndim,
    ).then(
        # New array has leftmost dimension as `dim`
        set_(y.dim(0)).to(dim),
        # has ndim incremented
        set_(y.ndim).to(nd + 1),
        # has remaiing dimensions copied from the source.
        CopyDim(src=x, dst=y, start=1, offset=1),
    )

    # Logic to copy dimensions
    yield rule(
        delme := CopyDim(src=x, dst=y, start=start, offset=offset),
        start < y.ndim,  # in range?
    ).then(
        # delete the node
        delete(delme),
        # copy the dimension
        set_(y.dim(start)).to(x.dim(start - offset)),
        # advance
        CopyDim(src=x, dst=y, start=start + 1, offset=offset),
    )

    yield rule(
        # rule to delete if out-of-bound
        delme := CopyDim(src=x, dst=y, offset=offset, start=start),
        start >= y.ndim,
    ).then(delete(delme))


# Here, we run the broadcasting rules and check the results:

if __name__ == "__main__":
    eg.run(ruleset_broadcasting.saturate())
    if IN_NOTEBOOK:
        eg.display(graphviz=True)

    # Verify
    eg.check(broadcasted.dim(0) == Dim.symbolic("M"))
    eg.check(broadcasted.dim(1) == Dim.symbolic("N"))
    eg.check(broadcasted.dim(2) == Dim.fixed(10))
    eg.check(broadcasted.dim(3) == Dim.fixed(4))

# ## Broadcasting Error Detection
#
# Now, we add the logic to detect broadcasting error. Starting with a
# failing example:

if __name__ == "__main__":
    eg = EGraph()

    # array0 is 10 x 4
    array0 = ArrayDesc(uid="array0")
    eg.register(
        set_(array0.dtype).to(TypeInt64),
        set_(array0.ndim).to(2),
        set_(array0.dim(0)).to(Dim.fixed(10)),
        set_(array0.dim(1)).to(Dim.fixed(4)),
    )

    # array1 is 2
    array1 = ArrayDesc(uid="array1")
    eg.let("array1", array1)
    eg.register(
        set_(array1.dtype).to(TypeInt64),
        set_(array1.ndim).to(1),
        set_(array1.dim(0)).to(Dim.fixed(2)),
    )

    if IN_NOTEBOOK:
        eg.display(graphviz=True)

    broadcasted = Broadcast(array0, array1)
    eg.let("broadcasted", broadcasted)

    eg.run(ruleset_broadcasting.saturate())
    # Cannot determine dimension 1 of the broadcasted array
    assert len(eg.extract_multiple(broadcasted.dim(1), 10)) == 1

# ### Define Error Handling Logic
#
# Broadcasting fails when the dimensions are mismatching and neither is one.


@function
def DimBroadcastFailed(dim: i64Like) -> Dim:
    "Mark the failed `dim`."
    ...


@ruleset
def ruleset_broadcasting_error(
    x: ArrayDesc,
    y: ArrayDesc,
    z: ArrayDesc,
    offset: i64,
    m: i64,
    n: i64,
):

    yield rule(
        # mismatch in dim
        CheckBroadcastDim(x, y, z, offset),
        x.dim(offset) == Dim.fixed(m),
        y.dim(offset) == Dim.fixed(n),
        m != 1,  # not one
        n != 1,  # not one
        m != n,  # not equal
    ).then(
        # Mark the dimension as a failed broadcast
        set_(z.dim(offset)).to(DimBroadcastFailed(offset))
    )


if __name__ == "__main__":
    eg.run((ruleset_broadcasting | ruleset_broadcasting_error).saturate())
    if IN_NOTEBOOK:
        eg.display(graphviz=True)

    # Verify
    eg.check(broadcasted.dim(0) == Dim.fixed(10))
    eg.check(broadcasted.dim(1) == DimBroadcastFailed(1))

# ### Implement CanBroadcast
#
# To implement `CanBroadcast` to determine whether a broadcasting is legal,
# we'll need do Boolean expression. `CanBroadcast(x, y)` is checking each
# dimension of `Broadcast(x, y)` to make sure they are valid `Dim`.


class BoolExpr(Expr):
    def __init__(self, val: BoolLike): ...
    def __and__(self, other: BoolExpr) -> BoolExpr: ...


@function
def ValidDim(desc: ArrayDesc, dim: i64Like) -> BoolExpr:
    "Is desc.dim(dim) valid?"
    ...


@function
def NextValidDim(desc: ArrayDesc, dim: i64Like) -> BoolExpr:
    """Rewrite to ValidDim(desc, dim) & NextValidDim(desc, dim + 1)
    when dim < desc.ndim
    Otherwise, this becomes True.
    """
    ...


@function
def CanBroadcast(x: ArrayDesc, y: ArrayDesc) -> BoolExpr:
    "Can x broadcast with y?"
    ...


@ruleset
def ruleset_can_broadcast(
    x: ArrayDesc,
    y: ArrayDesc,
    offset: i64,
    n: i64,
    sym: String,
):
    # Can broadcast?
    yield rewrite(CanBroadcast(x, y)).to(
        NextValidDim(Broadcast(x, y), 0)
        # given
    )

    # Logic to check if an ArrayDesc has invalid dimension
    yield rewrite(
        # Invalid dimension?
        ValidDim(x, offset),
        subsume=True,
    ).to(
        BoolExpr(False),
        # given
        x.dim(offset) == DimBroadcastFailed(offset),
    )
    yield rewrite(
        # Valid fixed dimension?
        ValidDim(x, offset),
        subsume=True,
    ).to(
        BoolExpr(True),
        # given
        x.dim(offset) == Dim.fixed(n),
    )
    yield rewrite(
        # Valid symbolic dimension?
        ValidDim(x, offset),
        subsume=True,
    ).to(
        BoolExpr(True),
        # given
        x.dim(offset) == Dim.symbolic(sym),
    )
    yield rewrite(
        # Expand the expressions
        NextValidDim(x, offset),
        subsume=True,
    ).to(
        ValidDim(x, offset) & NextValidDim(x, offset + 1),
        # given
        offset < x.ndim,
    )
    yield rewrite(
        # Out-of-bound check resolve to True
        NextValidDim(x, offset),
        subsume=True,
    ).to(
        BoolExpr(True),
        # given
        offset >= x.ndim,
    )


@ruleset
def ruleset_condition(x: BoolExpr, y: BoolExpr):
    yield rewrite(
        # False & y is False
        BoolExpr(False) & y,
        subsume=True,
    ).to(BoolExpr(False))
    yield rewrite(
        # True & True is True
        BoolExpr(True) & BoolExpr(True),
        subsume=True,
    ).to(BoolExpr(True))
    # Commutative
    yield rewrite(x & y).to(y & x)


# -

# Test

if __name__ == "__main__":
    # Case 1: broadcasting is illegal
    case1 = CanBroadcast(array0, array1)
    eg.let("can_broadcast_1", case1)
    # Case 2: broadcasting is legal
    case2 = CanBroadcast(array0, array0)
    eg.let("can_broadcast_2", case2)
    eg.run(
        (
            ruleset_broadcasting
            | ruleset_broadcasting_error
            | ruleset_can_broadcast
            | ruleset_condition
        ).saturate()
    )
    if IN_NOTEBOOK:
        eg.display(graphviz=True)
    # Verify
    eg.check(case1 == BoolExpr(False))
    eg.check(case2 == BoolExpr(True))
