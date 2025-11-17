from __future__ import annotations
from tracemalloc import start
import pytest
import inspect
import numpy as np
import operator
import math
from functools import reduce
from pprint import pprint
from dataclasses import dataclass
from types import FunctionType
import time

from typeinfer_ifelse import Type, TypeFloat64, TypeVar, Nb_CastI64ToF64
from typeinfer_array import (
    ArrayDesc,
    Broadcast,
    Dim,
    ExtendEGraphToRVSDG as _ExtendEGraphToRVSDG,
    MyCostModel as _MyCostModel,
    Grammar,
    Int64,
    NbOp_Type,
    NbOp_Base,
    TypeInt64,
    compiler_config as _compiler_config,
    DataLayout,
)
from call_graph import CallGraphVisitor
from mlir_backend import Backend as _MlirBackend, LowerStates

from egglog import (
    Bool,
    BoolLike,
    Expr,
    String,
    StringLike,
    Unit,
    Vec,
    function,
    i64,
    i64Like,
    rewrite,
    birewrite,
    rule,
    ruleset,
    set_,
    union,
    var,
    method,
    Ruleset,
)
from sealir.eqsat.py_eqsat import (
    Py_AttrIO,
    Py_Call,
    Py_CallKwargs,
    Py_DivIO,
    Py_LoadGlobal,
    Py_NegIO,
    Py_SubIO,
    Py_MulIO,
    Py_SliceIO,
    Py_SubscriptIO,
    Py_Tuple,
    Py_AddIO,
    Py_SetitemIO,
    Py_FloorDivIO,
)
from sealir.eqsat.py_eqsat import make_rules as py_eqsat_rules
from sealir.eqsat.rvsdg_eqsat import Term, TermDict, TermList, termlist, DynInt
from sealir.rvsdg.grammar import SExpr
import sealir.rvsdg.grammar as rg
from sealir import ase
from sealir.rvsdg import format_rvsdg
from utils import Report

from pathlib import Path
import os.path

from sealir.eqsat.rvsdg_eqsat import wildcard as _wc


class TodoException(NotImplementedError): ...


source_filename = (
    Path(os.path.dirname(__file__)) / ".." / "examples" / "llama3" / "llama3.py"
)
with open(source_filename, "r") as fin:
    source_code = fin.read()

cgv = CallGraphVisitor(source_code, source_filename)
cgv.visit_all()


#######################################
class Module(Expr):
    def __init__(self, name: StringLike): ...

    def toType(self) -> Type: ...


@function
def ModuleGetAttr(mod: Module, attrname: StringLike) -> Term: ...


# Install module rules


def make_module_rule(gv_name: str, mod_name):
    def ruleset_module_getattr(
        mod: Module,
        attrname: String,
        io: Term,
        name: String,
        op: Term,
        args: Vec[Term],
        obj: Term,
    ):
        # When loading a global variable that matches the module name,
        # set its type to be the corresponding module type
        yield rule(
            op == Py_LoadGlobal(io, name),
            name == String(gv_name),
        ).then(set_(TypeVar(op).getType()).to(Module(mod_name).toType()))

        # Module getattr
        yield rule(
            op == Py_AttrIO(io, obj, attrname),
            TypeVar(obj).getType() == Module(name).toType(),
        ).then(
            # Shortcut io
            union(op.getPort(0)).with_(io),
            # Setup getattr
            union(op.getPort(1)).with_(ModuleGetAttr(Module(name), attrname)),
        )

    return ruleset_module_getattr


def make_module_fact_ruleset(module_mapping: dict[str, str]):
    rs = Ruleset(name="module_fact_ruleset")
    for gv_name, mod_name in module_mapping.items():
        rs.register(make_module_rule(gv_name, mod_name))
    return rs


module_rules = make_module_fact_ruleset(cgv.imported)


#######################################


@function
def TypeTuple(size: i64Like) -> Type: ...


@function
def tuple_slice_upper(tup: Term, upper_amt: i64Like) -> Term: ...


@function
def tuple_add(lhs: Term, rhs: Term) -> Term: ...


@ruleset
def ruleset_tuple(
    tuptype: Type,
    expr: Term,
    io: Term,
    amt: i64,
    target: Term,
    obj: Term,
    size: i64,
    elems: TermList,
    termVec: Vec[Term],
    termVec2: Vec[Term],
    slice: Term,
    io2: Term,
    lhs: Term,
    rhs: Term,
    lhs_size: i64,
    rhs_size: i64,
):
    yield rule(
        # handle value_tuple[:-1]
        slice
        == Py_SliceIO(io, Term.LiteralNone(), Term.LiteralI64(amt), Term.LiteralNone()),
        expr == Py_SubscriptIO(io2, obj, slice.getPort(1)),
        TypeVar(obj).getType() == TypeTuple(size),
    ).then(
        union(expr.getPort(0)).with_(io),
        union(expr.getPort(0)).with_(io2),
        union(expr.getPort(1)).with_(tuple_slice_upper(obj, amt)),
    )
    # tuple building
    yield rule(
        expr == Py_Tuple(TermList(termVec)),
        size == termVec.length(),
    ).then(set_(TypeVar(expr).getType()).to(TypeTuple(size)))
    # tuple adding
    yield rule(
        expr == Py_AddIO(io, lhs, rhs),
        TypeVar(lhs).getType() == TypeTuple(lhs_size),
        TypeVar(rhs).getType() == TypeTuple(rhs_size),
    ).then(
        union(expr.getPort(1)).with_(tuple_add(lhs, rhs)),
        union(expr.getPort(0)).with_(io),
    )
    # tuple __getitem__
    yield rule(
        target == Py_SubscriptIO(io, expr, Term.LiteralI64(amt)),
        expr == Py_Tuple(TermList(termVec)),
    ).then(
        union(target.getPort(1)).with_(termVec[amt]),
        # io
        union(io).with_(target.getPort(0)),
    )
    yield rewrite(
        tuple_add(Py_Tuple(TermList(termVec)), Py_Tuple(TermList(termVec2)))
    ).to(Py_Tuple(TermList(termVec.append(termVec2))))


class Slice(Expr):
    @classmethod
    def from_term(cls, term: Term) -> Slice: ...

    @classmethod
    def from_args(cls, lower: Term, upper: Term, step: Term) -> Slice: ...


@ruleset
def ruleset_slice(
    py_slice: Term, io: Term, lower: Term, upper: Term, step: Term, slice1: Slice
):
    # slice building are pure
    yield rewrite(Py_SliceIO(io, lower, upper, step).getPort(0)).to(io)

    # Slice.from_term(Py_SliceIO) to Slice.from_args
    yield rewrite(Slice.from_term(Py_SliceIO(io, lower, upper, step).getPort(1))).to(
        Slice.from_args(lower, upper, step)
    )


@ruleset
def ruleset_more_constant_folding(x: i64, y: i64, io: Term, res: Term):
    yield rule(res == Py_AddIO(io, Term.LiteralI64(x), Term.LiteralI64(y))).then(
        union(res.getPort(1)).with_(Term.LiteralI64(x + y)),
        union(res.getPort(0)).with_(io),
    )

    # FloorDiv

    yield rule(res == Py_FloorDivIO(io, Term.LiteralI64(x), Term.LiteralI64(y))).then(
        union(res.getPort(1)).with_(Term.LiteralI64(x / y)),
        union(res.getPort(0)).with_(io),
    )


@ruleset
def ruleset_more_typing(term: Term):
    # literal int is int64
    yield rule(term == Term.LiteralI64(_wc(i64))).then(
        set_(TypeVar(term).getType()).to(TypeInt64),
    )


#######################################


# Install numpy function rules
@function(cost=1000)
def NpyOp_AsArray(io: Term, operand: Term) -> Term: ...


@function
def NpyOp_AsArray_F64(scalar: Term) -> Term: ...


@function(cost=1000)
def NpyOp_Sum(io: Term, operand: Term, axis: i64Like, keepdims: BoolLike) -> Term: ...


@function
def NpyOp_Sum_Shaped(
    io: Term,
    operand: Term,
    axis: i64Like,
    keepdims: BoolLike,
    inshape: Shape,
    outshape: Shape,
) -> Term: ...


@function(cost=1000)
def NpyOp_Max(io: Term, operand: Term, axis: i64Like, keepdims: BoolLike) -> Term: ...


@function
def NpyOp_Max_Shaped(
    io: Term,
    operand: Term,
    axis: i64Like,
    keepdims: BoolLike,
    inshape: Shape,
    outshape: Shape,
) -> Term: ...


@function(cost=1000)
def NpyOp_Exp(io: Term, operand: Term) -> Term: ...


@function
def NpyOp_Exp_Shaped(
    io: Term, operand: Term, inshape: Shape, outshape: Shape
) -> Term: ...


@function(cost=1000)
def NpyOp_Add(io: Term, lhs: Term, rhs: Term) -> Term: ...


@function
def NpyOp_Add_Shaped(
    io: Term, lhs: Term, rhs: Term, lhs_shape: Shape, rhs_shape: Shape, outshape: Shape
) -> Term: ...


@function(cost=1000)
def NpyOp_Subtract(io: Term, lhs: Term, rhs: Term) -> Term: ...


@function
def NpyOp_Subtract_Shaped(
    io: Term, lhs: Term, rhs: Term, lhs_shape: Shape, rhs_shape: Shape, outshape: Shape
) -> Term: ...


@function(cost=1000)
def NpyOp_Multiply(io: Term, lhs: Term, rhs: Term) -> Term: ...


@function
def NpyOp_Multiply_Shaped(
    io: Term, lhs: Term, rhs: Term, lhs_shape: Shape, rhs_shape: Shape, outshape: Shape
) -> Term: ...


@function(cost=1000)
def NpyOp_Divide(io: Term, lhs: Term, rhs: Term) -> Term: ...


@function
def NpyOp_Divide_Shaped(
    io: Term, lhs: Term, rhs: Term, lhs_shape: Shape, rhs_shape: Shape, outshape: Shape
) -> Term: ...


@function(cost=1000)
def NpyOp_Reshape(io: Term, ary: Term, src_nd: i64Like, new_shape: Term) -> Term: ...


@function
def NpyOp_Reshape_Shaped(
    io: Term, ary: Term, src_nd: i64Like, inshape: Shape, outshape: Shape
) -> Term: ...


@function(cost=1000)
def NpyOp_Take_one_index(
    io: Term, ary: Term, index: i64Like, axis: i64Like
) -> Term: ...


@function
def NpyOp_Take_Shaped_one_index(
    io: Term,
    ary: Term,
    index: i64Like,
    axis: i64Like,
    src_nd: i64Like,
    inshape: Shape,
    outshape: Shape,
) -> Term: ...


@function(cost=1000)
def NpyOp_Broadcast_To(io: Term, ary: Term, shape: Term) -> Term: ...


@function
def NpyOp_Broadcast_To_Shaped(
    io: Term, ary: Term, inshape: Shape, outshape: Shape
) -> Term: ...


@function(cost=10000)
def NpyOp_Stack_2(io: Term, ary1: Term, ary2: Term, axis: i64Like) -> Term: ...


@function
def NpyOp_Stack_2_Shaped(
    io: Term, ary1: Term, ary2: Term, axis: i64Like, inshape: Shape, outshape: Shape
) -> Term: ...


@function(cost=1000)
def NpyOp_Transpose_simple(io: Term, array: Term) -> Term: ...


@function
def NpyOp_Transpose_Shaped_simple(
    io: Term, array: Term, inshape: Shape, outshape: Shape
) -> Term: ...


@function(cost=1000)
def NpyOp_Transpose_explicit(
    io: Term, array: Term, reorder: TermList, outshape: Shape
) -> Term: ...


@function
def NpyOp_Transpose_Shaped_explicit(
    io: Term, array: Term, inshape: Shape, reorder: TermList, outshape: Shape
) -> Term: ...


@function(cost=1000)
def NpyOp_MatMul(io: Term, lhs: Term, rhs: Term) -> Term: ...


@function
def NpyOp_MatMul_Shaped(
    io: Term,
    lhs: Term,
    rhs: Term,
    lhs_shape: Shape,
    rhs_shape: Shape,
    out_shape: Shape,
) -> Term: ...


@function
def NpyOp_SetitemIO_Shaped_2d_index(
    io: Term, ary: Term, value: Term, ishape: Shape, index0: Slice, index1: Slice
) -> Term: ...


@function
def NpyOp_GetitemIO_Shaped_2d_index(
    io: Term, ary: Term, ishape: Shape, index0: Slice, index1: Slice, oshape: Shape
) -> Term: ...


@function(cost=1000)
def NpyOp_Copy(io: Term, ary: Term) -> Term: ...


@function
def NpyOp_Copy_Shaped(io: Term, ary: Term, shape: Shape) -> Term: ...


@function
def Shape_Broadcast(lhs: Shape, rhs: Shape) -> Shape: ...


@function
def get_ufunc_reduce_array_desc(
    in_array: ArrayDesc, axis: i64Like, keepdims: BoolLike
) -> ArrayDesc: ...


@function(cost=1000)
def _shape_from_tuple(termVec: Vec[Term]) -> Shape: ...


@ruleset
def ruleset_ufunc_reduce_array_desc(
    in_array: ArrayDesc,
    out_array: ArrayDesc,
    axis: i64,
    keepdims: Bool,
    ndim: i64,
    idx: i64,
    dim: Dim,
):
    @function
    def _ad_reduce_keepdims_not_normed(
        in_array: ArrayDesc,
        out_array: ArrayDesc,
        axis: i64Like,
        ndim: i64Like,
    ) -> Unit: ...

    @function
    def _ad_reduce_keepdims(
        in_array: ArrayDesc,
        out_array: ArrayDesc,
        axis: i64Like,
        ndim: i64Like,
        i: i64Like,
    ) -> Unit: ...

    # get_ufunc_reduce_array_desc
    yield rule(
        out_array == get_ufunc_reduce_array_desc(in_array, axis, keepdims),
        ndim == in_array.ndim,
        keepdims == Bool(True),
    ).then(
        set_(out_array.dataLayout).to(in_array.dataLayout),
        set_(out_array.ndim).to(ndim),
        set_(out_array.dtype).to(in_array.dtype),
        _ad_reduce_keepdims_not_normed(in_array, out_array, axis, ndim),
    )

    # _ad_reduce_keepdims_not_normed
    #   normalize axis
    yield rewrite(
        _ad_reduce_keepdims_not_normed(in_array, out_array, axis, ndim),
        subsume=True,
    ).to(
        _ad_reduce_keepdims(in_array, out_array, ndim + axis, ndim, 0),
        axis < i64(0),
    )
    yield rewrite(
        _ad_reduce_keepdims_not_normed(in_array, out_array, axis, ndim),
        subsume=True,
    ).to(
        _ad_reduce_keepdims(in_array, out_array, axis, ndim, 0),
        axis >= i64(0),
    )
    #   out_dim[idx]=in_dim[idx] if idx != axis
    yield rule(
        _ad_reduce_keepdims(in_array, out_array, axis, ndim, idx),
        0 <= idx,
        idx < ndim,
        idx != axis,
        axis < ndim,  # valid
    ).then(
        union(out_array.dim(idx)).with_(in_array.dim(idx)),
    )
    #   out_dim[idx]=1 if idx == axis
    yield rule(
        _ad_reduce_keepdims(in_array, out_array, axis, ndim, idx),
        0 <= idx,
        idx < ndim,
        idx == axis,
    ).then(
        union(out_array.dim(idx)).with_(Dim.fixed(1)),
    )
    #  idx+=1
    yield rule(
        _ad_reduce_keepdims(in_array, out_array, axis, ndim, idx),
        0 <= idx,
        idx < ndim - 1,
    ).then(
        _ad_reduce_keepdims(in_array, out_array, axis, ndim, idx + 1),
    )


@function
def DEBUG(operand: Term) -> Term: ...


class Shape(Expr):
    @classmethod
    def from_list(cls, shape: Vec[Dim]) -> Shape: ...

    @classmethod
    def from_arraydesc(cls, ad: ArrayDesc) -> Shape: ...

    @method(cost=1000)
    def __init__(self): ...

    def to_append(self, ad: ArrayDesc, start: i64Like, nd: i64Like) -> Shape: ...

    def append(self, dim: Dim) -> Shape: ...

    def toTuple(self) -> Term: ...

    @property
    def size(self) -> i64: ...

    def __add__(self, rhs: Shape) -> Shape: ...


@function(cost=1)
def ArrayType(
    ndim: i64Like, dtype: Type, shape: Shape, layout: DataLayout
) -> ArrayDesc: ...


@function
def math_unary_func(fname: StringLike) -> Term: ...


@function
def MathOp_Sqrt_F64(x: Term) -> Term: ...


class MathRules:
    module_name = "math"

    @staticmethod
    def sqrt(orig: Term):
        yield rewrite(orig, subsume=True).to(math_unary_func("sqrt"))

        @ruleset
        def handle_sqrt(
            io: Term,
            argVec: Vec[Term],
            arg: Term,
            res: Term,
        ):
            callee = Py_Call(func=math_unary_func("sqrt"), io=io, args=TermList(argVec))
            yield rewrite(callee.getPort(1)).to(
                MathOp_Sqrt_F64(arg),
                # when
                arg == argVec[0],
                argVec.length() == i64(1),
                TypeVar(arg).getType() == TypeFloat64,  # argument must be float64
            )
            yield rewrite(callee.getPort(0)).to(io)  # Pure op

            # Casting
            yield rewrite(callee.getPort(1)).to(
                MathOp_Sqrt_F64(Nb_CastI64ToF64(arg)),
                # when
                arg == argVec[0],
                argVec.length() == i64(1),
                TypeVar(arg).getType() == TypeInt64,
            )

            # Typing
            yield rule(
                res == MathOp_Sqrt_F64(arg),
            ).then(set_(TypeVar(res).getType()).to(TypeFloat64))

        yield handle_sqrt


class NumPyRules:
    module_name = "numpy"

    @staticmethod
    def _make_reduce_op_rules(op_name: str, op_constructor, op_con_special=None):
        """
        Common logic for numpy reduce operations (max, sum, etc.) that handle
        keepdims and axis parameters.
        """
        io = var("io", Term)
        obj = var("obj", Term)
        res = var("res", Term)
        args = var("args", TermList)
        kwargs = var("kwargs", TermDict)

        intype = var("intype", Type)
        arrdesc = var("arrdesc", ArrayDesc)
        shape = var("shape", Shape)
        in_ad = var("in_ad", ArrayDesc)

        nd = var("nd", i64)

        callee = Py_CallKwargs(
            func=npy_reduce(op_name), io=io, args=args, kwargs=kwargs
        )
        keepdims_val = var("keepdims_val", Bool)
        axis_val = var("axis_val", i64)
        yield rule(callee).then(kwargs.lookup("keepdims"), kwargs.lookup("axis"))
        yield rewrite(callee.getPort(1)).to(
            op_constructor(io, args[0], axis=axis_val, keepdims=keepdims_val),
            # conditions
            Term.LiteralI64(axis_val) == kwargs.get("axis"),
            Term.LiteralBool(keepdims_val) == kwargs.get("keepdims"),
        )
        # No effect on the output
        yield rewrite(callee.getPort(0)).to(io)
        yield rule(
            res == op_constructor(io, obj, axis=axis_val, keepdims=keepdims_val),
            intype == TypeVar(obj).getType(),
            intype == arrdesc.toType(),
        ).then(
            set_(TypeVar(res).getType()).to(
                get_ufunc_reduce_array_desc(arrdesc, axis_val, keepdims_val).toType()
            )
        )
        # make it shape specialized
        if op_con_special is not None:
            yield rule(
                res == op_constructor(io, obj, axis=axis_val, keepdims=keepdims_val),
                arrdesc.toType() == TypeVar(res).getType(),
                in_ad.toType() == TypeVar(obj).getType(),
                nd == arrdesc.ndim,
                shape == Shape().to_append(arrdesc, 0, nd),
            ).then(
                union(res).with_(
                    op_con_special(
                        io,
                        obj,
                        axis=axis_val,
                        keepdims=keepdims_val,
                        outshape=Shape.from_arraydesc(arrdesc),
                        inshape=Shape.from_arraydesc(in_ad),
                    )
                )
            )

    @staticmethod
    def _make_binary_rules(op_name: str, op_constructor, op_con_special=None):
        io = var("io", Term)
        lhs = var("lhs", Term)
        rhs = var("rhs", Term)
        res = var("res", Term)
        lhs_arraydesc = var("lhs_arraydesc", ArrayDesc)
        rhs_arraydesc = var("rhs_arraydesc", ArrayDesc)
        res_arraydesc = var("res_arraydesc", ArrayDesc)
        arg_vector = var("arg_vector", Vec[Term])
        nd = var("nd", i64)
        shape = var("shape", Shape)

        the_call = Py_Call(
            func=npy_binary_ufunc(op_name),
            io=io,
            args=TermList(arg_vector),
        )
        yield rule(
            the_call,
            arg_vector.length() == i64(2),
            lhs == arg_vector[0],
            rhs == arg_vector[1],
        ).then(
            # No output effect
            union(the_call.getPort(0)).with_(io),
            union(the_call.getPort(1)).with_(op_constructor(io, lhs, rhs)),
        )
        # Typing and broadcasting
        yield rule(
            res == op_constructor(io, lhs, rhs),
            lhs_arraydesc.toType() == TypeVar(lhs).getType(),
            rhs_arraydesc.toType() == TypeVar(rhs).getType(),
        ).then(
            set_(TypeVar(res).getType()).to(
                Broadcast(lhs_arraydesc, rhs_arraydesc).toType()
            )
        )
        yield rule(
            res == op_constructor(io, lhs, rhs),
            lhs_arraydesc.toType() == TypeVar(lhs).getType(),
            rhs_arraydesc.toType() == TypeVar(rhs).getType(),
            res_arraydesc.toType() == TypeVar(res).getType(),
            # if the dtype matches TODO: fix type promotion
            lhs_arraydesc.dtype == rhs_arraydesc.dtype,
        ).then(
            # set dtype
            set_(res_arraydesc.dtype).to(lhs_arraydesc.dtype),
            set_(res_arraydesc.dataLayout).to(
                DataLayout.strided()
            ),  # TODO improve this
        )
        if op_con_special is not None:
            yield rule(
                res == op_constructor(io, lhs, rhs),
                res_arraydesc.toType() == TypeVar(res).getType(),
                lhs_arraydesc.toType() == TypeVar(lhs).getType(),
                rhs_arraydesc.toType() == TypeVar(rhs).getType(),
                nd == res_arraydesc.ndim,
                shape == Shape().to_append(res_arraydesc, 0, nd),
            ).then(
                union(res).with_(
                    op_con_special(
                        io,
                        lhs,
                        rhs,
                        lhs_shape=Shape.from_arraydesc(lhs_arraydesc),
                        rhs_shape=Shape.from_arraydesc(rhs_arraydesc),
                        outshape=Shape.from_arraydesc(res_arraydesc),
                    )
                )
            )

    @staticmethod
    def _make_unary_rules(op_name: str, op_constructor, op_con_special=None):
        io = var("io", Term)
        operand = var("operand", Term)
        res = var("res", Term)
        operand_arraydesc = var("operand_arraydesc", ArrayDesc)
        res_arraydesc = var("res_arraydesc", ArrayDesc)
        in_ad = var("in_ad", ArrayDesc)
        arg_vector = var("arg_vector", Vec[Term])
        nd = var("nd", i64)
        shape = var("shape", Shape)

        the_call = Py_Call(
            func=npy_unary_ufunc(op_name),
            io=io,
            args=TermList(arg_vector),
        )
        yield rule(
            the_call,
            arg_vector.length() == i64(1),
            operand == arg_vector[0],
        ).then(
            # No output effect
            union(the_call.getPort(0)).with_(io),
            union(the_call.getPort(1)).with_(op_constructor(io, operand)),
        )
        # Typing and broadcasting
        yield rule(
            res == op_constructor(io, operand),
            operand_arraydesc.toType() == TypeVar(operand).getType(),
        ).then(set_(TypeVar(res).getType()).to(operand_arraydesc.toType()))

        if op_con_special is not None:
            yield rule(
                res == op_constructor(io, operand),
                res_arraydesc.toType() == TypeVar(res).getType(),
                in_ad.toType() == TypeVar(operand).getType(),
                nd == res_arraydesc.ndim,
                shape == Shape().to_append(res_arraydesc, 0, nd),
            ).then(
                union(res).with_(
                    op_con_special(
                        io,
                        operand,
                        Shape.from_arraydesc(in_ad),
                        outshape=Shape.from_arraydesc(res_arraydesc),
                    )
                )
            )

    @staticmethod
    def exp(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_unary_ufunc("exp"))
        yield from NumPyRules._make_unary_rules("exp", NpyOp_Exp, NpyOp_Exp_Shaped)

    @staticmethod
    def max(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_reduce("max"))
        yield from NumPyRules._make_reduce_op_rules("max", NpyOp_Max, NpyOp_Max_Shaped)

    @staticmethod
    def sum(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_reduce("sum"))
        yield from NumPyRules._make_reduce_op_rules("sum", NpyOp_Sum, NpyOp_Sum_Shaped)

    @staticmethod
    def add(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_binary_ufunc("add"))
        yield from NumPyRules._make_binary_rules("add", NpyOp_Add, NpyOp_Add_Shaped)

    @staticmethod
    def subtract(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_binary_ufunc("subtract"))
        yield from NumPyRules._make_binary_rules(
            "subtract", NpyOp_Subtract, NpyOp_Subtract_Shaped
        )

    @staticmethod
    def multiply(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_binary_ufunc("multiply"))
        yield from NumPyRules._make_binary_rules(
            "multiply", NpyOp_Multiply, NpyOp_Multiply_Shaped
        )

    @staticmethod
    def divide(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_binary_ufunc("divide"))
        yield from NumPyRules._make_binary_rules(
            "divide", NpyOp_Divide, NpyOp_Divide_Shaped
        )

    @staticmethod
    def take(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_take())

        io = var("io", Term)
        obj = var("obj", Term)
        res = var("res", Term)
        args = var("args", TermList)
        kwargs = var("kwargs", TermDict)
        callee = Py_CallKwargs(func=npy_take(), io=io, args=args, kwargs=kwargs)
        index_val = var("index", i64)
        axis_val = var("axis_val", i64)
        ndim = var("ndim", i64)
        dtype = var("dtype", Type)
        layout = var("layout", DataLayout)
        shape = var("shape", Shape)
        inshape = var("inshape", Shape)
        dimVec = var("dimVec", Vec[Dim])

        yield rule(callee).then(kwargs.lookup("axis"), args[1])
        yield rewrite(callee.getPort(1)).to(
            NpyOp_Take_one_index(io, args[0], index_val, axis=axis_val),
            # when
            Term.LiteralI64(axis_val) == kwargs.get("axis"),
            Term.LiteralI64(index_val) == args[1],
        )
        # No output effect.
        # e.g. np.take() returns a copy
        yield rewrite(callee.getPort(0)).to(io)

        # Typing & Shaping
        yield rule(
            res == NpyOp_Take_one_index(io, obj, index_val, axis=-1),
            TypeVar(obj).getType() == ArrayType(ndim, dtype, shape, layout).toType(),
            shape == Shape.from_list(dimVec),
        ).then(
            set_(TypeVar(res).getType()).to(
                ArrayType(
                    ndim - 1, dtype, Shape.from_list(dimVec.pop()), layout
                ).toType()
            )
        )
        # promote
        yield rule(
            res == NpyOp_Take_one_index(io, obj, index_val, axis_val),
            TypeVar(res).getType()
            == ArrayType(_wc(i64), _wc(Type), shape, _wc(DataLayout)).toType(),
            TypeVar(obj).getType()
            == ArrayType(ndim, _wc(Type), inshape, _wc(DataLayout)).toType(),
        ).then(
            union(res).with_(
                NpyOp_Take_Shaped_one_index(
                    io, obj, index_val, axis_val, ndim, inshape, shape
                )
            )
        )

    @staticmethod
    def broadcast_to(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_broadcast_to())

        io = var("io", Term)
        call = var("call", Term)
        obj = var("obj", Term)
        shape_tup = var("shape_tup", Term)
        res = var("res", Term)
        argVec = var("argVec", Vec[Term])
        ndim = var("ndim", i64)
        shape = var("shape", Shape)
        dimVec = var("dimVec", Vec[Dim])
        termVec = var("termVec", Vec[Term])
        ad = var("ad", ArrayDesc)
        in_ad = var("in_ad", ArrayDesc)

        yield rule(
            call == Py_Call(func=npy_broadcast_to(), io=io, args=TermList(argVec)),
            argVec.length() == i64(2),
        ).then(
            # No output effect
            union(call.getPort(0)).with_(io),
            union(call.getPort(1)).with_(
                NpyOp_Broadcast_To(io, argVec[0], argVec[1]),
            ),
        )

        # Typing & Shaping

        yield rewrite(
            Py_Tuple(TermList(termVec)),
        ).to(
            _shape_from_tuple(termVec).toTuple(),
        )

        yield rule(
            res == NpyOp_Broadcast_To(io, obj, shape_tup),
            TypeVar(obj).getType() == ad.toType(),  # obj is an array
            shape == Shape.from_list(dimVec),
            shape_tup == shape.toTuple(),
            ndim == dimVec.length(),
        ).then(
            set_(TypeVar(res).getType()).to(
                ArrayType(
                    ndim,
                    ad.dtype,
                    shape,
                    ad.dataLayout,
                ).toType()
            )
        )
        # promote
        yield rewrite(
            NpyOp_Broadcast_To(io, obj, shape_tup),
        ).to(
            NpyOp_Broadcast_To_Shaped(io, obj, Shape.from_arraydesc(in_ad), shape),
            # when
            shape == Shape.from_list(dimVec),
            in_ad.toType() == TypeVar(obj).getType(),
            shape_tup == shape.toTuple(),
        )

    @staticmethod
    def stack(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_stack())

        io = var("io", Term)
        res = var("res", Term)
        ary1 = var("ary1", Term)
        ary2 = var("ary2", Term)
        arrayVec = var("arrayVec", Vec[Term])
        args = var("args", TermList)
        kwargs = var("kwargs", TermDict)
        callee = Py_CallKwargs(func=npy_stack(), io=io, args=args, kwargs=kwargs)
        axis_val = var("axis_val", i64)
        ndim = var("ndim", i64)
        dtype = var("dtype", Type)
        layout = var("layout", DataLayout)
        shape1 = var("shape1", Shape)
        shape2 = var("shape2", Shape)
        dimVec1 = var("dimVec1", Vec[Dim])
        dimVec2 = var("dimVec2", Vec[Dim])
        ad1 = var("ad1", ArrayDesc)

        yield rule(callee).then(kwargs.lookup("axis"), args[0])
        yield rewrite(callee.getPort(1)).to(
            NpyOp_Stack_2(io, ary1, ary2, axis=axis_val),
            # when
            Term.LiteralI64(axis_val) == kwargs.get("axis"),
            args[0] == Py_Tuple(TermList(arrayVec)),
            ary1 == arrayVec[0],
            ary2 == arrayVec[1],
        )
        # no output effect
        yield rewrite(callee.getPort(0)).to(io)

        # Typing & Shaping
        # FIXME: only does stack of two arrays using recursion

        @function(cost=1000)
        def _shape_stack_at_axis(
            shape1: Shape, shape2: Shape, axis_val: i64Like, ndim: i64Like
        ) -> Shape: ...
        @function(cost=1000)
        def _shape_stack_at_axis_normalized(
            dimVec1: Vec[Dim], dimVec2: Vec[Dim], axis_val: i64Like
        ) -> Shape: ...
        @function
        def _shape_stack_copy_tail(dimVec1: Vec[Dim], dimVec2: Vec[Dim]) -> Shape: ...

        yield rule(
            res == NpyOp_Stack_2(io, ary1, ary2, axis=axis_val),
            TypeVar(ary1).getType() == ArrayType(ndim, dtype, shape1, layout).toType(),
            TypeVar(ary2).getType() == ArrayType(ndim, dtype, shape2, layout).toType(),
            shape1 == Shape.from_list(dimVec1),
            shape2 == Shape.from_list(dimVec2),
            shape1 == shape2,
        ).then(
            set_(TypeVar(res).getType()).to(
                ArrayType(
                    ndim + 1,
                    dtype,
                    _shape_stack_at_axis(shape1, shape2, axis_val, ndim),
                    layout,
                ).toType()
            )
        )
        yield rewrite(res).to(
            NpyOp_Stack_2_Shaped(
                io, ary1, ary2, axis_val, Shape.from_arraydesc(ad1), shape1
            ),
            # when
            res == NpyOp_Stack_2(io, ary1, ary2, axis_val),
            ad1.toType() == TypeVar(ary1).getType(),
            TypeVar(res).getType()
            == ArrayType(_wc(i64), _wc(Type), shape1, _wc(DataLayout)).toType(),
        )

        yield rewrite(_shape_stack_at_axis(shape1, shape2, axis_val, ndim)).to(
            _shape_stack_at_axis_normalized(dimVec1, dimVec2, axis_val + ndim),
            # when
            axis_val < 0,
            shape1 == Shape.from_list(dimVec1),
            shape2 == Shape.from_list(dimVec2),
        )

        yield rewrite(_shape_stack_at_axis(shape1, shape2, axis_val, ndim)).to(
            _shape_stack_at_axis_normalized(dimVec1, dimVec2, axis_val),
            # when
            axis_val >= 0,
            shape1 == Shape.from_list(dimVec1),
            shape2 == Shape.from_list(dimVec2),
        )

        yield rewrite(
            # axis_val != 0
            _shape_stack_at_axis_normalized(dimVec1, dimVec2, axis_val)
        ).to(
            Shape.from_list(Vec[Dim](dimVec1[0]))
            + _shape_stack_at_axis_normalized(
                dimVec1.remove(0), dimVec2.remove(0), axis_val - 1
            ),
            # when
            dimVec1[0] == dimVec2[0],
            axis_val != i64(0),
            dimVec1.length() > 0,
        )
        yield rewrite(
            # axis_val == 0
            _shape_stack_at_axis_normalized(dimVec1, dimVec2, axis_val)
        ).to(
            Shape.from_list(Vec[Dim](dimVec1[0], Dim.fixed(2)))
            + _shape_stack_copy_tail(dimVec1.remove(0), dimVec2.remove(0)),
            # when
            dimVec1[0] == dimVec2[0],
            axis_val == i64(0),
        )

        yield rewrite(
            # empty
            _shape_stack_at_axis_normalized(dimVec1, dimVec2, axis_val)
        ).to(
            Shape(),
            # when
            dimVec1.length() == i64(0),
        )

        # _shape_stack_copy_tail
        yield rewrite(_shape_stack_copy_tail(dimVec1, dimVec2)).to(
            Shape.from_list(Vec[Dim](dimVec1[0]))
            + _shape_stack_copy_tail(dimVec1.remove(0), dimVec2.remove(0)),
            # when
            dimVec1[0] == dimVec2[0],  # TODO: mark error when mismatch
        )

        yield rewrite(_shape_stack_copy_tail(dimVec1, dimVec2)).to(
            Shape(),
            dimVec1.length() == i64(0),
            dimVec2.length() == i64(0),
        )

    @staticmethod
    def transpose(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_transpose())

        @ruleset
        def handle_simple_transpose(
            io: Term,
            argVec: Vec[Term],
            arg_array: Term,
            res: Term,
            dtype: Type,
            shape: Shape,
            in_shape: Shape,
            out_shape: Shape,
            layout: DataLayout,
            ndim: i64,
            dimVec: Vec[Dim],
        ):
            callee = Py_Call(func=npy_transpose(), io=io, args=TermList(argVec))

            yield rewrite(callee.getPort(1)).to(
                NpyOp_Transpose_simple(io, arg_array),
                # when
                arg_array == argVec[0],
                i64(1) == argVec.length(),
            )
            # No output effect
            yield rewrite(callee.getPort(0)).to(io)

            # Typing & Shaping

            @function(cost=1000)
            def _shape_transpose(dimVec: Vec[Dim]) -> Shape: ...

            yield rule(
                res == NpyOp_Transpose_simple(io, arg_array),
                TypeVar(arg_array).getType()
                == ArrayType(ndim, dtype, shape, layout).toType(),
                shape == Shape.from_list(dimVec),
            ).then(
                set_(TypeVar(res).getType()).to(
                    ArrayType(ndim, dtype, _shape_transpose(dimVec), layout).toType()
                )
            )

            # Specialization
            yield rewrite(res).to(
                NpyOp_Transpose_Shaped_simple(io, arg_array, in_shape, out_shape),
                # when
                res == NpyOp_Transpose_simple(io, arg_array),
                TypeVar(arg_array).getType()
                == ArrayType(ndim, dtype, in_shape, layout).toType(),
                TypeVar(res).getType()
                == ArrayType(ndim, dtype, out_shape, layout).toType(),
            )

            # _shape_transpose
            yield rewrite(
                _shape_transpose(dimVec),
                subsume=True,
            ).to(
                Shape().append(dimVec[ndim - 1])
                + _shape_transpose(dimVec.remove(ndim - 1)),
                # when
                ndim == dimVec.length(),
                ndim > 0,
            )

            yield rewrite(
                _shape_transpose(dimVec),
                subsume=True,
            ).to(
                Shape(),
                # when
                ndim == dimVec.length(),
                ndim == i64(0),
            )

            # _transpose_reoder_simple

        yield handle_simple_transpose

        @ruleset
        def handle_explicit_transpose(
            io: Term,
            argVec: Vec[Term],
            arg_array: Term,
            arg_shape: Term,
            res: Term,
            dtype: Type,
            shape: Shape,
            in_shape: Shape,
            out_shape: Shape,
            layout: DataLayout,
            ndim: i64,
            dimVec: Vec[Dim],
            termVec: Vec[Term],
            x: i64,
        ):
            callee = Py_Call(func=npy_transpose(), io=io, args=TermList(argVec))

            @function(cost=1000)
            def _shape_reorder(dimVec: Vec[Dim], termVec: Vec[Term]) -> Shape: ...

            yield rewrite(callee.getPort(1)).to(
                NpyOp_Transpose_explicit(
                    io, arg_array, TermList(termVec), _shape_reorder(dimVec, termVec)
                ),
                # when
                TypeVar(arg_array).getType()
                == ArrayType(_wc(i64), _wc(Type), in_shape, _wc(DataLayout)).toType(),
                arg_array == argVec[0],
                argVec[1] == Py_Tuple(TermList(termVec)),
                i64(2) == argVec.length(),
                in_shape == Shape.from_list(dimVec),
            )

            # Typing & Shaping
            yield rule(
                res
                == NpyOp_Transpose_explicit(
                    io, arg_array, TermList(termVec), out_shape
                ),
                TypeVar(arg_array).getType()
                == ArrayType(ndim, dtype, in_shape, _wc(DataLayout)).toType(),
            ).then(
                set_(TypeVar(res).getType()).to(
                    ArrayType(ndim, dtype, out_shape, DataLayout.strided()).toType()
                ),
                union(res).with_(
                    NpyOp_Transpose_Shaped_explicit(
                        io, arg_array, in_shape, TermList(termVec), out_shape
                    )
                ),
            )

            # _shape_reorder
            yield rewrite(
                _shape_reorder(dimVec, termVec),
                subsume=True,
            ).to(
                Shape().append(dimVec[x]) + _shape_reorder(dimVec, termVec.remove(0)),
                # when
                Term.LiteralI64(x) == termVec[0],
            )
            yield rewrite(
                _shape_reorder(dimVec, Vec[Term].empty()),
                subsume=True,
            ).to(Shape())

        yield handle_explicit_transpose

    @staticmethod
    def matmul(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_matmul())

        @ruleset
        def handle_matmul(
            io: Term,
            argVec: Vec[Term],
            arg_lhs: Term,
            arg_rhs: Term,
            res: Term,
            dtype: Type,
            shape: Shape,
            shape_lhs: Shape,
            shape_rhs: Shape,
            ndim_lhs: i64,
            ndim_rhs: i64,
            layout_lhs: DataLayout,
            layout_rhs: DataLayout,
            dimVec_lhs: Vec[Dim],
            dimVec_rhs: Vec[Dim],
            dimM: Dim,
            dimN: Dim,
            dimK: Dim,
        ):
            callee = Py_Call(func=npy_matmul(), io=io, args=TermList(argVec))

            yield rewrite(callee.getPort(1)).to(
                NpyOp_MatMul(io, arg_lhs, arg_rhs),
                # when
                arg_lhs == argVec[0],
                arg_rhs == argVec[1],
                i64(2) == argVec.length(),
            )
            # no output effect
            yield rewrite(callee.getPort(0)).to(io)

            # Typing & Shaping

            @function(cost=1000)
            def _shape_matmul_broadcast(lhs: Shape, rhs: Shape) -> Shape: ...

            @function(cost=1000)
            def _shape_matmul_broadcast_step(
                lhs: Shape,
                rhs: Shape,
                dimM: Dim,
                dimN: Dim,
                dimK: Dim,
            ) -> Shape: ...

            @function(cost=10)
            def _shape_matmul_batch(
                batch_shape: Shape,
                dimM: Dim,
                dimN: Dim,
                dimK: Dim,
            ) -> Shape: ...

            yield rule(
                res == NpyOp_MatMul(io, arg_lhs, arg_rhs),
                TypeVar(arg_lhs).getType()
                == ArrayType(ndim_lhs, dtype, shape_lhs, layout_lhs).toType(),
                TypeVar(arg_rhs).getType()
                == ArrayType(ndim_rhs, dtype, shape_rhs, layout_rhs).toType(),
            ).then(
                set_(TypeVar(res).getType()).to(
                    ArrayType(
                        ndim=ndim_lhs.max(ndim_rhs),
                        dtype=dtype,
                        shape=_shape_matmul_broadcast(shape_lhs, shape_rhs),
                        layout=DataLayout.c_contiguous(),
                    ).toType()
                )
            )

            # _shape_matmul_broadcast
            yield rewrite(
                _shape_matmul_broadcast(
                    Shape.from_list(dimVec_lhs),
                    Shape.from_list(dimVec_rhs),
                ),
                subsume=True,
            ).to(
                _shape_matmul_broadcast_step(
                    Shape.from_list(dimVec_lhs.pop().pop()),
                    Shape.from_list(dimVec_rhs.pop().pop()),
                    dimM,
                    dimN,
                    dimK,
                ),
                # when
                ndim_lhs == dimVec_lhs.length(),
                ndim_rhs == dimVec_rhs.length(),
                ndim_lhs >= 2,
                ndim_rhs >= 2,
                dimM == dimVec_lhs[ndim_lhs - 2],
                dimN == dimVec_lhs[ndim_lhs - 1],
                dimN == dimVec_rhs[ndim_rhs - 2],
                dimK == dimVec_rhs[ndim_rhs - 1],
            )

            yield rewrite(
                _shape_matmul_broadcast_step(
                    shape_lhs,
                    shape_rhs,
                    dimM,
                    dimN,
                    dimK,
                )
            ).to(
                _shape_matmul_batch(
                    Shape_Broadcast(shape_lhs, shape_rhs), dimM, dimN, dimK
                )
            )

            yield rewrite(
                _shape_matmul_batch(shape, dimM, dimN, dimK),
                subsume=True,
            ).to(shape + Shape().append(dimM).append(dimK))

            # Specialize
            yield rewrite(res).to(
                NpyOp_MatMul_Shaped(io, arg_lhs, arg_rhs, shape_lhs, shape_rhs, shape),
                # when
                res == NpyOp_MatMul(io, arg_lhs, arg_rhs),
                TypeVar(res).getType()
                == ArrayType(_wc(i64), _wc(Type), shape, _wc(DataLayout)).toType(),
                TypeVar(arg_lhs).getType()
                == ArrayType(_wc(i64), _wc(Type), shape_lhs, _wc(DataLayout)).toType(),
                TypeVar(arg_rhs).getType()
                == ArrayType(_wc(i64), _wc(Type), shape_rhs, _wc(DataLayout)).toType(),
            )

        yield handle_matmul

    @staticmethod
    def copy(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_copy())

        @ruleset
        def handle_copy(
            io: Term,
            argVec: Vec[Term],
            ary: Term,
            res: Term,
            dtype: Type,
            shape: Shape,
            ndim: i64,
        ):
            callee = Py_Call(func=npy_copy(), io=io, args=TermList(argVec))

            yield rewrite(callee).to(
                NpyOp_Copy(io, ary),
                # when
                ary == argVec[0],
                argVec.length() == i64(1),
            )

            # Typing & shaping
            # Input is the same as output
            yield rule(
                res == NpyOp_Copy(io, ary),
                TypeVar(ary).getType()
                == ArrayType(
                    ndim=ndim,
                    dtype=dtype,
                    shape=shape,
                    layout=_wc(DataLayout),
                ).toType(),
            ).then(
                union(res).with_(NpyOp_Copy_Shaped(io, ary, shape)),
                set_(TypeVar(res.getPort(1)).getType()).to(
                    ArrayType(
                        ndim=ndim,
                        dtype=dtype,
                        shape=shape,
                        layout=DataLayout.c_contiguous(),
                    ).toType()
                ),
            )

        yield handle_copy

    @staticmethod
    def asarray(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_asarray())

        @ruleset
        def handle_asarray(
            io: Term,
            argVec: Vec[Term],
            arg: Term,
            res: Term,
        ):
            callee = Py_Call(npy_asarray(), io, TermList(argVec))

            # arity=1
            yield rewrite(callee.getPort(1)).to(
                NpyOp_AsArray(io, arg),
                # when
                argVec.length() == i64(1),
                arg == argVec[0],
            )
            # no output effect
            yield rewrite(callee.getPort(0)).to(io)

            # Typing for asarray(scalar)
            yield rewrite(res).to(
                NpyOp_AsArray_F64(arg),
                # when
                res == NpyOp_AsArray(io, arg),
                TypeVar(arg).getType() == TypeFloat64,
            )

            @function(cost=1000)
            def _get_arraydesc_from_asarray(op: Term) -> ArrayDesc: ...

            yield rule(
                res == NpyOp_AsArray_F64(arg),
            ).then(
                set_(TypeVar(res).getType()).to(
                    (_ad := _get_arraydesc_from_asarray(res)).toType()
                ),
                set_(_ad.ndim).to(i64(1)),
                set_(_ad.dim(0)).to(Dim.fixed(1)),
                set_(_ad.dataLayout).to(DataLayout.c_contiguous()),
                set_(_ad.dtype).to(TypeFloat64),
            )

        yield handle_asarray


@ruleset
def ruleset_numpy_promote_binop(
    op: Term, lhs: Term, rhs: Term, io: Term, arraydesc: ArrayDesc
):
    def promote_ops(operand, opname, py_op):
        return rewrite(py_op(io, lhs, rhs)).to(
            Py_Call(ModuleGetAttr(Module("numpy"), opname), io, termlist(lhs, rhs)),
            # when
            # any operand is a ndarray
            TypeVar(operand).getType() == arraydesc.toType(),
        )

    for operand in [lhs, rhs]:
        yield promote_ops(operand, "add", Py_AddIO)
        yield promote_ops(operand, "subtract", Py_SubIO)
        yield promote_ops(operand, "multiply", Py_MulIO)
        yield promote_ops(operand, "divide", Py_DivIO)


@ruleset
def ruleset_numpy_reshape(
    ary: Term,
    io: Term,
    args: Vec[Term],
    callee: Term,
    new_shape: Shape,
    old_shape: Shape,
    shape: Shape,
    target: Term,
    dtype: Type,
    ndim: i64,
    size: i64,
    n: i64,
    layout: DataLayout,
    dimVec: Vec[Dim],
    termVec: Vec[Term],
    ad: ArrayDesc,
):
    # match operation
    yield rule(
        target
        == Py_Call(func=callee.getPort(1), io=callee.getPort(0), args=TermList(args)),
        callee == Py_AttrIO(io, ary, "reshape"),
        args.length() == i64(1),  # expect one argument
        TypeVar(ary).getType() == ad.toType(),
        ndim == ad.ndim,
    ).then(
        union(target.getPort(1)).with_(NpyOp_Reshape(io, ary, ndim, args[0])),
        # has output effect because it returns a view
        union(target.getPort(0)).with_(io),
        union(target.getPort(0)).with_(callee.getPort(0)),
    )
    # promote
    yield rewrite(target).to(
        NpyOp_Reshape_Shaped(io, ary, ndim, Shape.from_arraydesc(ad), shape),
        # when
        target == NpyOp_Reshape(io, ary, ndim, _wc(Term)),
        TypeVar(target).getType()
        == ArrayType(_wc(i64), _wc(Type), shape, _wc(DataLayout)).toType(),
        TypeVar(ary).getType() == ad.toType(),
    )
    # type & shape inference

    @function
    def _normalize_shape_for_reshape(new_shape: Shape, size: i64Like) -> Shape: ...
    @function
    def _norm_shape_step(shape: Shape, size: i64Like, out_shape: Shape) -> Shape: ...

    yield rule(
        target == NpyOp_Reshape(io, ary, _wc(i64), new_shape.toTuple()),
        TypeVar(new_shape.toTuple()).getType() == TypeTuple(ndim),
        ArrayType(_wc(i64), dtype, old_shape, layout).toType()
        == TypeVar(ary).getType(),
        size == old_shape.size,
        new_shape.size >= 0,
    ).then(
        set_(TypeVar(target).getType()).to(
            ArrayType(ndim, dtype, new_shape, layout).toType()
        ),
    )
    yield rule(
        target == NpyOp_Reshape(io, ary, _wc(i64), new_shape.toTuple()),
        TypeVar(new_shape.toTuple()).getType() == TypeTuple(ndim),
        ArrayType(_wc(i64), dtype, old_shape, layout).toType()
        == TypeVar(ary).getType(),
        size == old_shape.size,
        new_shape.size < 0,
    ).then(
        set_(TypeVar(target).getType()).to(
            ArrayType(
                ndim, dtype, _normalize_shape_for_reshape(new_shape, size), layout
            ).toType()
        ),
    )

    yield rule(
        NpyOp_Reshape(io, ary, _wc(i64), Py_Tuple(TermList(termVec))),
    ).then(
        union(Py_Tuple(TermList(termVec))).with_(_shape_from_tuple(termVec).toTuple())
    )

    yield rewrite(_shape_from_tuple(termVec)).to(
        Shape().append(Dim.fixed(n)) + _shape_from_tuple(termVec.remove(0)),
        # when
        termVec[0] == Term.LiteralI64(n),
    )
    yield rewrite(_shape_from_tuple(termVec)).to(
        Shape(),
        # when
        termVec.length() == i64(0),
    )

    yield rewrite(_normalize_shape_for_reshape(shape, size)).to(
        _norm_shape_step(shape, size, Shape())
    )
    yield rewrite(
        _norm_shape_step(Shape.from_list(dimVec), size, new_shape),
    ).to(
        _norm_shape_step(
            Shape.from_list(dimVec.remove(0)), size / n, new_shape.append(dimVec[0])
        ),
        # when
        dimVec[0] == Dim.fixed(n),
        n > 0,  # positive
        dimVec.length() > 0,
    )

    yield rule(
        shape == _norm_shape_step(Shape.from_list(dimVec), size, new_shape),
        # when
        dimVec[0] == Dim.fixed(-1),  # dynamic size
        dimVec.length() > 0,
        n == Shape.from_list(dimVec).size,
    ).then(
        union(shape).with_(
            new_shape.append(Dim.fixed(size / (i64(0) - n)))
            + Shape.from_list(dimVec.remove(0))
        )
    )


@ruleset
def ruleset_numpy_shape(
    ary: Term,
    shape: Shape,
    shape1: Shape,
    shape2: Shape,
    io: Term,
    ad: ArrayDesc,
    ndim: i64,
    dtype: Type,
    layout: DataLayout,
    target: Term,
    dimVec: Vec[Dim],
    dimVec2: Vec[Dim],
    amt: i64,
    idx: i64,
    dim: Dim,
    elems: Vec[Term],
    termlist: TermList,
):
    yield rule(
        target == Py_AttrIO(io, ary, "shape"),
        ad.toType() == TypeVar(ary).getType(),
        ad == ArrayType(ndim, dtype, shape, layout),
    ).then(
        union(target.getPort(1)).with_(shape.toTuple()),
        # shortcut io
        union(io).with_(target.getPort(0)),
    )
    # shape.toTuple typing
    yield rule(
        target == shape.toTuple(),
        shape == Shape.from_list(dimVec),
        ndim == dimVec.length(),
    ).then(set_(TypeVar(target).getType()).to(TypeTuple(ndim)))
    # slicing shape[:-1]
    yield rewrite(tuple_slice_upper(Shape.from_list(dimVec).toTuple(), -1)).to(
        Shape.from_list(dimVec.pop()).toTuple()
    )

    # shape + shape
    @function(cost=1000000)
    def _convert_term_tuple_to_shape(elems: Vec[Term]) -> Term: ...

    yield rule(
        target == Py_Tuple(TermList(elems)),
        tuple_add(Shape.from_list(dimVec).toTuple(), target),
    ).then(
        union(target).with_(_convert_term_tuple_to_shape(elems)),
    )

    yield rewrite(
        # non-zero lengths
        _convert_term_tuple_to_shape(elems)
    ).to(
        tuple_add(
            _convert_term_tuple_to_shape(elems.pop()),
            Shape().append(Dim.fixed(amt)).toTuple(),
        ),
        # where
        elems.length() > i64(0),
        Term.LiteralI64(amt) == elems[elems.length() - 1],
    )

    yield rewrite(
        # zero length
        _convert_term_tuple_to_shape(elems)
    ).to(
        Shape().toTuple(),
        # where
        elems.length() == i64(0),
    )

    yield rewrite(
        # add two tuple of shape
        tuple_add(shape1.toTuple(), shape2.toTuple())
    ).to((shape1 + shape2).toTuple())

    # empty shape is the same as shape from_list([])
    yield birewrite(Shape()).to(Shape.from_list(Vec[Dim].empty()))

    # shape.size

    @function(cost=10000)
    def _shape_compute_size(dimVec: Vec[Dim]) -> DynInt: ...

    yield rule(shape == Shape.from_list(dimVec)).then(_shape_compute_size(dimVec))

    yield rule(
        shape == Shape.from_list(dimVec),
        amt == _shape_compute_size(dimVec).get(),
    ).then(
        set_(shape.size).to(amt),
    )
    yield rewrite(
        _shape_compute_size(dimVec),
    ).to(
        DynInt(amt) * _shape_compute_size(dimVec.remove(0)),
        # when
        Dim.fixed(amt) == dimVec[0],
        dimVec.length() > i64(1),
    )

    yield rewrite(
        _shape_compute_size(dimVec),
    ).to(
        amt,
        # when
        Dim.fixed(amt) == dimVec[0],
        dimVec.length() == i64(1),
    )
    yield rewrite(
        _shape_compute_size(dimVec),
    ).to(
        DynInt(0),
        # when
        dimVec.length() == i64(0),
    )

    # Shape building
    yield rewrite(shape.to_append(ad, idx, ndim)).to(
        shape.append(dim).to_append(ad, idx + 1, ndim),
        dim == ad.dim(idx),
        idx < ndim,
    )
    yield rewrite(shape.to_append(ad, ndim, ndim)).to(shape)
    yield rewrite(Shape.from_arraydesc(ad), subsume=True).to(
        Shape().to_append(ad, 0, ndim), ndim == ad.ndim
    )
    yield rewrite(Shape().append(dim)).to(Shape.from_list(Vec[Dim](dim)))
    yield rewrite(Shape.from_list(dimVec).append(dim)).to(
        Shape.from_list(dimVec.append(Vec[Dim](dim)))
    )

    # shape + shape
    yield rewrite(Shape.from_list(dimVec) + Shape.from_list(dimVec2)).to(
        Shape.from_list(dimVec.append(dimVec2))
    )

    # shape.toTuple()
    yield rule(
        target == Shape.from_list(dimVec).toTuple(),
        Dim.fixed(amt) == dimVec[0],
    ).then(
        union(target).with_(
            tuple_add(
                Py_Tuple(TermList(Vec[Term](Term.LiteralI64(amt)))),
                Shape.from_list(dimVec.remove(0)).toTuple(),
            )
        ),
    )
    yield rule(
        target == Shape().toTuple(),
    ).then(
        union(target).with_(Py_Tuple(TermList(Vec[Term].empty()))),
    )


@ruleset_numpy_shape.register
def _ruleset_shape_broadcas(
    dimVec1: Vec[Dim],
    dimVec2: Vec[Dim],
    nd: i64,
    ndim1: i64,
    ndim2: i64,
    dim1: Dim,
    dim2: Dim,
    m: i64,
    n: i64,
):
    # broadcast shape
    # TODO: duplicating logic from ch5 because a shape based broadcast is cleaner

    @function(cost=1000)
    def _broadcast_match_nd(dimVec1: Vec[Dim], dimVec2: Vec[Dim], nd: i64) -> Shape: ...

    @function(cost=1000)
    def _broadcast_match_dimensions(dimVec1: Vec[Dim], dimVec2: Vec[Dim]) -> Shape: ...

    @function(cost=1000)
    def _broadcast_dim(dimL: Dim, dimR: Dim) -> Dim: ...

    yield rewrite(
        Shape_Broadcast(Shape.from_list(dimVec1), Shape.from_list(dimVec2))
    ).to(
        _broadcast_match_nd(dimVec1, dimVec2, ndim1.max(ndim2)),
        # when
        ndim1 == dimVec1.length(),
        ndim2 == dimVec2.length(),
    )

    yield rewrite(
        _broadcast_match_nd(dimVec1, dimVec2, nd),
        subsume=True,
    ).to(
        _broadcast_match_nd(dimVec1, Vec[Dim](Dim.fixed(1)).append(dimVec2), nd),
        # when
        nd != dimVec2.length(),
    )

    yield rewrite(
        _broadcast_match_nd(dimVec1, dimVec2, nd),
        subsume=True,
    ).to(
        _broadcast_match_nd(Vec[Dim](Dim.fixed(1)).append(dimVec1), dimVec2, nd),
        # when
        nd != dimVec1.length(),
    )

    yield rewrite(
        _broadcast_match_nd(dimVec1, dimVec2, nd),
        subsume=True,
    ).to(
        _broadcast_match_dimensions(dimVec1, dimVec2),
        # when
        nd == dimVec1.length(),
        nd == dimVec2.length(),
    )

    yield rewrite(
        _broadcast_match_dimensions(dimVec1, dimVec2),
        subsume=True,
    ).to(
        Shape().append(_broadcast_dim(dim1, dim2))
        + _broadcast_match_dimensions(dimVec1.remove(0), dimVec2.remove(0)),
        # when
        dim1 == dimVec1[0],
        dim2 == dimVec2[0],
    )

    yield rewrite(
        _broadcast_match_dimensions(dimVec1, dimVec2),
        subsume=True,
    ).to(
        Shape(),
        # when
        dimVec1.length() == i64(0),
        dimVec2.length() == i64(0),
    )

    # broadcast Dim
    yield rewrite(
        # dim==1 can broadcast to anything
        _broadcast_dim(dim1, Dim.fixed(1)),
        subsume=True,
    ).to(dim1)
    yield rewrite(
        # commutative: flip left and right
        _broadcast_dim(dim1, dim2),
    ).to(
        _broadcast_dim(dim2, dim1),
    )

    yield rewrite(
        # matching dimension
        _broadcast_dim(dim1, dim1),
        subsume=True,
    ).to(dim1)


@ruleset
def ruleset_numpy_setitem(
    res: Term,
    io: Term,
    ary: Term,
    indices: Term,
    index0: Term,
    index1: Term,
    val: Term,
    ad: ArrayDesc,
    idxVec: Vec[Term],
):
    yield rewrite(
        Py_SetitemIO(io=io, obj=ary, index=indices, val=val), subsume=True
    ).to(
        NpyOp_SetitemIO_Shaped_2d_index(
            io=io,
            ary=ary,
            value=val,
            ishape=Shape.from_arraydesc(ad),
            index0=Slice.from_term(index0),
            index1=Slice.from_term(index1),
        ),
        # when
        TypeVar(ary).getType() == ad.toType(),
        indices == Py_Tuple(TermList(idxVec)),
        idxVec.length() == i64(2),
        idxVec[0] == index0,
        idxVec[1] == index1,
    )


@ruleset
def ruleset_numpy_getitem(
    res: Term,
    io: Term,
    ary: Term,
    indices: Term,
    shape: Shape,
    upper0: i64,
    upper1: i64,
    ndim: i64,
    dtype: Type,
    dimVec: Vec[Dim],
    index0: Term,
    index1: Term,
    idxVec: Vec[Term],
):
    @function
    def _shape_getitem_2d(shape: Shape, index0: Slice, index1: Slice) -> Shape: ...

    yield rule(
        res == Py_SubscriptIO(io=io, obj=ary, index=indices),
        TypeVar(ary).getType()
        == ArrayType(
            ndim=ndim, dtype=dtype, shape=shape, layout=_wc(DataLayout)
        ).toType(),
        indices == Py_Tuple(TermList(idxVec)),
        idxVec.length() == i64(2),
        idxVec[0] == index0,
        idxVec[1] == index1,
    ).then(
        union(res.getPort(0)).with_(io),  # consume but no new effect
        union(res.getPort(1)).with_(
            NpyOp_GetitemIO_Shaped_2d_index(
                io=io,
                ary=ary,
                ishape=shape,
                index0=(_slice0 := Slice.from_term(index0)),
                index1=(_slice1 := Slice.from_term(index1)),
                oshape=(_oshape := _shape_getitem_2d(shape, _slice0, _slice1)),
            )
        ),
        set_(TypeVar(res.getPort(1)).getType()).to(
            ArrayType(
                ndim=ndim, dtype=dtype, shape=_oshape, layout=DataLayout.strided()
            ).toType()
        ),
    )

    # _shape_getitem_2d
    none = Term.LiteralNone()
    yield rewrite(
        # support slice with only upper
        _shape_getitem_2d(
            Shape.from_list(dimVec),
            Slice.from_args(none, Term.LiteralI64(upper0), none),
            Slice.from_args(none, Term.LiteralI64(upper1), none),
        )
    ).to(
        Shape().append(Dim.fixed(upper0)).append(Dim.fixed(upper1))
        + Shape.from_list(dimVec.remove(0).remove(0)),
        # when
        dimVec.length() >= 2,
    )


numpy_rulesset = (
    ruleset_numpy_reshape
    | ruleset_numpy_promote_binop
    | ruleset_numpy_shape
    | ruleset_numpy_setitem
    | ruleset_numpy_getitem
)


@function
def npy_unary_ufunc(name: StringLike) -> Term: ...


@function
def npy_binary_ufunc(name: StringLike) -> Term: ...


@function
def npy_reduce(name: StringLike) -> Term: ...


@function
def npy_take() -> Term: ...


@function
def npy_broadcast_to() -> Term: ...


@function
def npy_stack() -> Term: ...


@function
def npy_transpose() -> Term: ...


@function
def npy_matmul() -> Term: ...


@function
def npy_copy() -> Term: ...


@function
def npy_asarray() -> Term: ...


loaded_module = {
    "numpy": NumPyRules,
    "math": MathRules,
}


def module_function_rule_lookup(modname: str):
    modrule = loaded_module[modname]
    for fname in dir(modrule):
        if not fname.startswith("_"):
            fn = getattr(modrule, fname)
            if isinstance(fn, FunctionType):
                base_node = ModuleGetAttr(Module(modname), fname)
                yield from fn(base_node)


def make_function_rule(module_mapping: dict[str, str]):
    for modname in module_mapping.values():
        if modname in loaded_module:
            print("SETUP module rules", modname)
            rules = module_function_rule_lookup(modname)
            others = []
            for rule in rules:
                if isinstance(rule, Ruleset):
                    yield rule
                else:
                    others.append(rule)

            if others:
                yield ruleset(*others, name=f"ruleset_module_{modname}")


module_rulesets = (
    reduce(operator.or_, make_function_rule(cgv.imported)) | numpy_rulesset
)

######################################
# Explain array desc


@ruleset
def ruleset_explain_array_desc(
    ad: ArrayDesc,
    ndim: i64,
    dtype: Type,
    shape: Shape,
    dim: Dim,
    idx: i64,
    layout: DataLayout,
    dimVec: Vec[Dim],
    dimVec2: Vec[Dim],
):
    # ArrayType spelling
    yield rule(
        ndim == ad.ndim,
        dtype == ad.dtype,
        layout == ad.dataLayout,
    ).then(
        union(ad).with_(
            ArrayType(
                ndim=ndim,
                dtype=dtype,
                shape=Shape().to_append(ad, 0, ndim),
                layout=layout,
            )
        )
    )
    # inverse

    @function
    def _array_desc_set_dim(ad: ArrayDesc, dimVec: Vec[Dim], i: i64Like) -> Unit: ...

    yield rule(
        ad
        == ArrayType(
            ndim=ndim,
            dtype=dtype,
            shape=Shape.from_list(dimVec),
            layout=layout,
        ),
        dimVec.length() == ndim,
    ).then(
        set_(ad.ndim).to(ndim),
        set_(ad.dtype).to(dtype),
        set_(ad.dataLayout).to(layout),
        _array_desc_set_dim(ad, dimVec, 0),
    )

    yield rule(
        _array_desc_set_dim(ad, dimVec, idx),
        idx < dimVec.length(),
    ).then(
        _array_desc_set_dim(ad, dimVec, idx + 1),
        set_(ad.dim(idx)).to(dimVec[idx]),
    )


class Annotate(Expr):
    def __init__(self, term: Term, ty: Type): ...


class ArgFact(Expr):
    def __init__(self, i: i64Like, ty: Type): ...


@ruleset
def ruleset_typevar_annotate(term: Term, tv: TypeVar, typ: Type):
    yield rule(
        typ == TypeVar(term).getType(),
    ).then(Annotate(term, typ))


#######################################
# Extra operator rules


@function
def Nb_Neg_Int64(operand: Term) -> Term: ...


@ruleset
def ruleset_type_infer_negate(
    io: Term,
    x: Term,
    op: Term,
):
    yield rule(
        op == Py_NegIO(io, x),
        TypeVar(x).getType() == TypeInt64,
    ).then(
        union(op.getPort(1)).with_(Nb_Neg_Int64(x)),
        union(op.getPort(0)).with_(io),
    )

    yield rule(op == Nb_Neg_Int64(x)).then(set_(TypeVar(op).getType()).to(TypeInt64))


ruleset_extra_builtin_operations = ruleset_type_infer_negate

#######################################


class LLM_generic(NbOp_Base):
    desc: str
    operands: tuple[SExpr, ...]


class CostModel(_MyCostModel):
    def get_cost_function(self, nodename, op, ty, cost, children):
        match op:
            case "·.getType" if nodename.endswith("-TypeVar_getType"):
                return self.get_simple(float("inf"))
        cost = super().get_cost_function(nodename, op, ty, cost, children)
        return cost


class ExtendEGraphToRVSDG(_ExtendEGraphToRVSDG):
    def handle_Term(self, op: str, children: dict | list, grm: Grammar):
        parent_output = super().handle_Term(op, children, grm)
        if parent_output is NotImplemented:
            assert isinstance(children, dict)
            # flatten children
            # XXX: improve handling of Vec[]
            values = []
            for v in children.values():
                if isinstance(v, tuple):
                    v = grm.write(rg.GenericList(name="tuple", children=v))
                values.append(v)
            return grm.write(
                LLM_generic(
                    desc=op + f"<{', '.join(children)}>",
                    operands=tuple(values),
                )
            )
        return parent_output

    def is_type_from_egraph(self, node) -> bool:
        return node["op"] == "·.toType"

    def handle_Type(self, key: str, op: str, children: dict | list, grm: Grammar):
        try:
            return super().handle_Type(key, op, children, grm)
        except NotImplementedError:
            assert isinstance(children, dict)
            return self.handle_generic(key, op, children, grm)

    def handle_generic(self, key: str, op: str, children: dict | list, grm: Grammar):
        assert isinstance(children, dict)
        return super().handle_generic(str, op, children, grm)

    handle_ArrayDesc = handle_generic
    handle_Dim = handle_generic
    handle_TypedOuts = handle_generic
    handle_TypedIns = handle_generic
    handle_TypeVar = handle_generic
    handle_Module = handle_generic
    handle_ErrorMsg = handle_generic
    handle_DataLayout = handle_generic
    handle_Annotate = handle_generic
    handle_ArgFact = handle_generic
    handle_Shape = handle_generic
    handle_Slice = handle_generic


compiler_config = _compiler_config.copy()
compiler_config["converter_class"] = ExtendEGraphToRVSDG
compiler_config["cost_model"] = CostModel()


@dataclass
class BeArrayType:
    ndim: int
    dtype: str
    shape: tuple[int, ...]
    layout: str


class TypeSpeller(ase.TreeVisitor):

    def __init__(self):
        self.memo = {}

    def visit(self, expr: SExpr):
        memo = self.memo
        match expr:
            case rg.Generic(op, children):
                memo[expr] = r = self.visit_Generic(
                    op, tuple(map(lambda x: memo.get(x, x), children))
                )
                assert r is not None
            case rg.GenericList(op, children):
                memo[expr] = r = list(map(lambda x: memo.get(x, x), children))
                assert r is not None
            # case LLM_generic(desc, operands):
            #     memo[expr] = r = self.visit_Generic(desc, tuple(map(lambda x: memo.get(x, x), operands)))
            #     assert r is not None
            case NbOp_Type(str("Int64")):
                memo[expr] = Int64
            case NbOp_Type(str(name)):
                memo[expr] = name
            case rg.PyNone():
                memo[expr] = None
            case rg.PyInt(int(ival)):
                memo[expr] = ival
            case _:
                print("HAR?", ase.pretty_str(expr), type(expr))
                return None
        return expr

    def visit_Generic(self, op, children):
        match op, children:
            case "Shape", ():
                return ()
            case "Dim.symbolic", (name,):
                return name
            case "Dim.fixed", (name,):
                return name
            case "·.append", (asb, dim):
                return asb + [dim]
            case "DataLayout.strided", ():
                return "A"
            case "DataLayout.c_contiguous", ():
                return "C"
            case "ArrayType", (
                nd,
                dtype,
                shape,
                layout,
            ):
                return BeArrayType(nd, dtype, tuple(shape), layout)
            case "·.toType", (ad,):
                return ad
            case "Shape.from_list", (vec,):
                return vec
            case "Slice.from_args", (lower, upper, step):
                return slice(lower, upper, step)
            # case "·.toTuple<self>", (vec,):
            #     return vec
            case _:
                raise ValueError(f"{op} {children}")

    @classmethod
    def apply(cls, expr: SExpr):
        visitor = cls()
        ase.apply_bottomup(expr, visitor, reachable="compute")
        return visitor.memo[expr]


class StubBackend:
    def lower(self, root, argtypes):
        [func] = [child for child in root._args if isinstance(child, rg.Func)]

        # root._tape.render_dot(only_reachable=True).view()

        fname = func.fname
        beginnode = func.body.begin
        intypes = {}
        for argport in ase.search_parents(
            beginnode, lambda x: isinstance(x, rg.Unpack)
        ):
            # print('   .parent', argport)
            idx = argport._args[1]
            annos = list(
                ase.search_parents(
                    argport,
                    lambda x: isinstance(x, rg.Generic) and x._args[0] == "Annotate",
                )
            )
            if annos:
                intypes[idx] = TypeSpeller.apply(annos[0]._args[2])
        # print(intypes)

        # outtypes
        outtypes = {}
        for port in func.body.ports:
            annos = list(
                ase.search_parents(
                    port.value,
                    lambda x: isinstance(x, rg.Generic) and x._args[0] == "Annotate",
                )
            )
            if annos:
                outtypes[port.name] = TypeSpeller.apply(annos[0]._args[2])
        retty = outtypes["!ret"]

        # attrs = Attributes(func.body.begin.attrs)
        # retty = attrs.get_return_type(func.body)
        # print("ARGS", intypes)
        # print("RETURN TYPE", retty)
        return format_rvsdg(func)

    def jit_compile(self, module, extracted, export_name):
        return module  # TODO


def _mlir_location_from_frame(info: str = ""):
    """Helper to create MLIR location from Python frame information"""
    from mlir.ir import Location
    import inspect
    import os

    frame = inspect.currentframe().f_back
    if frame is None:
        return Location.name(info)

    # Get method/function name
    method_name = frame.f_code.co_name

    # Get source filename and line number
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno

    # Use just the basename for cleaner output
    basename = os.path.basename(filename)

    # Create location string in format "method_name@filename:lineno"
    location_str = f"{info}[{method_name}@{basename}:{lineno}]"

    return Location.name(location_str)


_DEBUG = False

import sys


def read_parent_source_line() -> int:
    caller_frame = sys._getframe(1)
    return caller_frame.f_lineno


class MlirBackend(_MlirBackend):
    tmp_dir: str | None = None

    def __init__(self):
        super().__init__()

    def run_passes(self, module):
        from mlir import ir
        from mlir.passmanager import PassManager

        if _DEBUG:
            module.dump()

        pass_man = PassManager(context=module.context)
        module.context.emit_error_diagnostics = True

        if _DEBUG:
            module.context.enable_multithreading(False)
        if _DEBUG:
            # notebook may hang if ir_printing is enabled and and MLIR failed.
            pass_man.enable_ir_printing()

        passes = [
            # Phase 1: Clean up and canonicalize
            "canonicalize",
            "cse",
            "symbol-dce",
            # Phase 2: Linalg optimizations
            "linalg-fold-unit-extent-dims",
            "linalg-fuse-elementwise-ops",
            "canonicalize",
            # Phase 3: Bufferization
            "one-shot-bufferize{bufferize-function-boundaries}",
            "canonicalize",
            "cse",
            # Phase 4: Convert to loops
            "linalg-generalize-named-ops",
            "convert-linalg-to-affine-loops",
            # Phase 5: Affine optimizations
            # "affine-loop-fusion='mode=greedy'",
            # "affine-scalrep",
            "func.func(affine-loop-invariant-code-motion)",
            "func.func(affine-simplify-structures)",
            "func.func(affine-loop-coalescing)",
            "func.func(affine-loop-tile{tile-size=64})",
            # "affine-loop-unroll='unroll-factor=4 unroll-up-to-factor'",
            # "affine-super-vectorize='vectorize-reductions'",
            "func.func(affine-parallelize{parallel-reductions='true'})",
            "func.func(affine-loop-normalize)",
            # Phase 6: Memory optimizations
            "normalize-memrefs",
            "fold-memref-alias-ops",
            "canonicalize",
            "expand-strided-metadata",
            "memref-expand",
            "lower-affine",
            # === SCF OPTIMIZATIONS ===
            # Phase 7: SCF-level optimizations (after lowering!)
            "scf-parallel-loop-fusion",
            "scf-for-loop-peeling",
            "scf-for-loop-specialization",
            # "for-loop-invariant-code-motion",
            "canonicalize",
            "cse",
            #     --buffer-deallocation                                  -   Adds all required dealloc operations for all allocations in the input program
            #   --buffer-deallocation-simplification                   -   Optimizes `bufferization.dealloc` operation for more efficient codegen
            #   --buffer-hoisting                                      -   Optimizes placement of allocation operations by moving them into common dominators and out of nested regions
            #   --buffer-loop-hoisting
            "func.func(promote-buffers-to-stack)",
            "func.func(mem2reg)",
            # "buffer-deallocation",
            # "convert-scf-to-openmp",
            # Phase 8: Lower to LLVM
            "convert-scf-to-cf",
            "convert-cf-to-llvm",
            "convert-vector-to-llvm",
            "finalize-memref-to-llvm",
            "convert-openmp-to-llvm",
            "convert-math-to-libm",
            "convert-math-to-llvm",
            "convert-arith-to-llvm",
            "convert-func-to-llvm",
            "reconcile-unrealized-casts",
        ]

        if self.tmp_dir is not None:
            tmp_dir = "tmp"
            os.makedirs(tmp_dir, exist_ok=True)
            self.tmp_dir = tmp_dir

        self.curr_dir_name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        os.makedirs(f"{self.tmp_dir}/{self.curr_dir_name}", exist_ok=True)
        
        for i, pass_name in enumerate(passes):
            pass_man = PassManager(context=module.context)
            pass_man.add(pass_name)
            pass_man.enable_verifier(True)
            pass_man.run(module.operation)
            asm = module.operation.get_asm(enable_debug_info=True)
            # Open a file in write mode
            pass_name = str(i) + "_" + pass_name
            with open(f"{self.tmp_dir}/{self.curr_dir_name}/{pass_name}.mlir", "w") as file:
                file.write(asm)

        # Output LLVM-dialect MLIR
        if _DEBUG:
            module.dump()
        return module

    def get_last_compiled_return_type(self):
        return self._retty

    def _cast_return_value(self, val):
        from mlir.dialects import memref

        resty = self.lower_type_return(self._retty)
        return memref.CastOp(resty, val)

    def lower(self, root, argtypes):
        self._retty = None  # reset
        [func] = [child for child in root._args if isinstance(child, rg.Func)]

        # HACK
        # Find arguments
        argfacts = [
            child
            for child in root._args
            if isinstance(child, rg.Generic) and child.name == "ArgFact"
        ]

        # print(format_rvsdg(func))
        fname = func.fname
        beginnode = func.body.begin
        intypes = {}

        for argfact in argfacts:
            [arg_idx, ty] = argfact.children
            intypes[arg_idx] = TypeSpeller.apply(ty)

        # pprint(intypes)
        ninports = len(beginnode.inports)
        assert len(intypes) == ninports - 1  # one extra for the IO
        self._argtys = tuple([intypes[i] for i in range(len(argfacts))])

        # outtypes
        outtypes = {}
        for port in func.body.ports:
            annos = list(
                ase.search_parents(
                    port.value,
                    lambda x: isinstance(x, rg.Generic) and x._args[0] == "Annotate",
                )
            )
            if annos:
                outtypes[port.name] = TypeSpeller.apply(annos[0]._args[2])
        retty = outtypes["!ret"]
        self._retty = retty  # TODO XXX ugly smelly code

        # attrs = Attributes(func.body.begin.attrs)
        # retty = attrs.get_return_type(func.body)
        # print("ARGS", self._argtys)
        # print("RETURN TYPE", retty)

        argtypes = self._argtys

        super().lower(func, argtypes)

        # print(self.module.dump())
        return self.module

    def lower_type_return(self, ty):
        from mlir.ir import MemRefType, F64Type, StridedLayoutAttr, ShapedType

        with self.context, _mlir_location_from_frame():
            ty = self._retty
            element_type = F64Type.get()
            nd = len(ty.shape)
            dyn = ShapedType.get_dynamic_stride_or_offset()
            return MemRefType.get(
                ty.shape, element_type, layout=StridedLayoutAttr.get(dyn, [dyn] * nd)
            )

    def get_return_types(self, root):
        return [self.lower_type_return(self._retty)]

    def lower_type(self, ty):
        from mlir.ir import MemRefType, F64Type, IntegerType

        if isinstance(ty, BeArrayType):
            # TODO: make this use static shape
            assert ty.dtype == "Float64"
            with self.context, _mlir_location_from_frame():
                element_type = F64Type.get()
                return MemRefType.get(ty.shape, element_type)
        elif ty == Int64:
            with self.context, _mlir_location_from_frame():
                return IntegerType.get_signless(64)
        else:
            return super().lower_type(ty)

    def lower_expr(self, expr: SExpr, state: LowerStates):
        from mlir import ir

        with _mlir_location_from_frame(str(expr)):
            match expr:
                case LLM_generic(desc=str(op), operands=tuple(operands)):
                    return self._lower_llm_ops(op, operands, state)
                case _:
                    return super().lower_expr(expr, state)

    def _get_func_by_name(self, fname: str):
        for decl in self.module.body:
            if decl.sym_name.value == fname:
                return decl

    def _gen_reshape(self, ary_val, in_shapes=(), out_shape=()):
        from mlir.dialects import memref, arith, bufferization, tensor
        from mlir import ir

        element_type = ir.F64Type.get()
        index_type = ir.IndexType.get()
        if in_shapes == out_shape:
            return ary_val

        with ir.Location.file(__file__, read_parent_source_line() + 1, 1):
            array_val_tensor = bufferization.to_tensor(
                ir.RankedTensorType.get(in_shapes[0], element_type),
                ary_val,
                restrict=True,
            )

        shape_tensor = tensor.empty([len(out_shape)], index_type)
        memref_type_res = ir.RankedTensorType.get(out_shape, element_type)
        for idx, i in enumerate(out_shape):
            tensor.insert(
                arith.constant(index_type, i),
                shape_tensor,
                [arith.constant(index_type, idx)],
            )
        with ir.Location.file(__file__, read_parent_source_line() + 1, 1):
            out = tensor.reshape(memref_type_res, array_val_tensor, shape=shape_tensor)

        return bufferization.to_buffer(ir.MemRefType.get(out_shape, element_type), out)

    def _gen_static_broadcast(self, array_val, in_shapes=(), out_shape=()):
        from mlir.dialects import memref, linalg, tensor, bufferization
        from mlir import ir

        (in_shape,) = in_shapes

        if in_shape == out_shape:
            return array_val

        element_type = ir.F64Type.get()

        array_val_tensor = bufferization.to_tensor(
            ir.RankedTensorType.get(in_shape, element_type), array_val, restrict=True
        )

        if tuple(in_shape) == tuple(out_shape):
            return array_val

        bc_out = tensor.empty(out_shape, element_type)

        # Create affine expressions that map to the broadcasted dimensions
        # For a 2x1 -> 2x12 broadcast, you want (d0, d1) -> (d0, 0)
        input_exprs = []
        for i, (in_dim, out_dim) in enumerate(zip(in_shape, out_shape)):
            if in_dim == 1 and out_dim > 1:
                # This dimension is being broadcasted - use constant 0
                input_exprs.append(ir.AffineConstantExpr.get(0))
            else:
                # This dimension is not broadcasted - use the dimension variable
                input_exprs.append(ir.AffineDimExpr.get(i))

        input_map = ir.AffineMap.get(len(out_shape), 0, input_exprs)
        # Output map: identity mapping for output dimensions
        output_map = ir.AffineMap.get_identity(len(out_shape))

        # Create iterator types - one parallel iterator for each output dimension
        iterator_types = [
            ir.Attribute.parse("#linalg.iterator_type<parallel>") for _ in out_shape
        ]
        with ir.Location.file(__file__, read_parent_source_line() + 1, 1):
            bc_op = linalg.GenericOp(
                [bc_out.type],
                inputs=[array_val_tensor],
                outputs=[bc_out],
                indexing_maps=[input_map, output_map],
                iterator_types=ir.ArrayAttr.get(iterator_types),
            )
            body = bc_op.regions[0].blocks.append(element_type, element_type)
            with ir.InsertionPoint(body):
                linalg.YieldOp([body.arguments[0]])

        return bufferization.to_buffer(
            ir.MemRefType.get(out_shape, element_type), bc_op
        )

    def _gen_take_shaped(self, ary_val, index, src_nd, in_shapes=(), out_shape=()):
        indices = [slice(None)] * (src_nd - 1) + [index]
        return self._gen_array_getitem_shaped(ary_val, indices, in_shapes, out_shape)

    def _gen_array_stack_shaped(self, input_args, axis, in_shapes=(), out_shape=()):
        from mlir.dialects import memref, tensor, bufferization
        from mlir import ir

        if axis == -1:
            axis = len(out_shape) - 1

        num_inputs = len(input_args)

        ndim = len(out_shape) - 1
        element_type = ir.F64Type.get()

        memref_type_out = ir.MemRefType.get(out_shape, element_type)
        with ir.Location.file(__file__, read_parent_source_line() + 1, 1):
            bc_out = tensor.empty(out_shape, element_type)

        curr_offset = 0
        strides = [1] * (ndim + 1)
        strides[axis] = num_inputs
        out_shape_inner = list(out_shape)
        out_shape_inner[axis] = 1

        for input_arg in input_args:

            offsets = [0] * (ndim + 1)
            offsets[axis] = curr_offset
            with ir.Location.file(__file__, read_parent_source_line() + 1, 1):
                bc_out = tensor.insert_slice(
                    bufferization.to_tensor(
                        ir.RankedTensorType.get(in_shapes[0], element_type),
                        input_arg,
                        restrict=True,
                    ),
                    bc_out,
                    offsets=[],
                    sizes=[],
                    strides=[],
                    static_offsets=ir.DenseI64ArrayAttr.get(offsets),
                    static_sizes=ir.DenseI64ArrayAttr.get(out_shape_inner),
                    static_strides=ir.DenseI64ArrayAttr.get(strides),
                )
            curr_offset += 1

        return bufferization.to_buffer(memref_type_out, bc_out)

    @classmethod
    def build_mlir_reassociation(cls, ndim, new_axes):
        from mlir import ir

        new_axes = sorted(
            [ax if ax >= 0 else ndim + len(new_axes) + ax for ax in new_axes]
        )
        final_ndim = ndim + len(new_axes)
        is_new_dim = [False] * final_ndim
        for ax in new_axes:
            is_new_dim[ax] = True

        reassociation = []
        current_group = []
        for output_pos in range(final_ndim):
            current_group.append(output_pos)
            if not is_new_dim[output_pos]:
                reassociation.append(current_group)
                current_group = []

        # Convert to MLIR ArrayAttr
        attr_groups = []
        for group in reassociation:
            group_attrs = [
                ir.IntegerAttr.get(ir.IntegerType.get_signless(64), idx)
                for idx in group
            ]
            attr_groups.append(ir.ArrayAttr.get(group_attrs))

        return ir.ArrayAttr.get(attr_groups)

    def _gen_inline_array_transpose(self, ary_val, permutation=None, dtype=None):
        """Inline version of transpose that returns the transposed memref directly."""
        # Get input shape information
        from mlir import ir

        input_type = ary_val.type
        if not isinstance(input_type, ir.MemRefType):
            raise TypeError("Input must be a MemRef type")

        in_shape = input_type.shape
        dims = len(in_shape)

        if permutation is None:
            permutation = [i for i in range(dims).__reversed__()]

        return self._gen_inline_array_transpose_shaped(
            ary_val, permutation=permutation, dtype=dtype, in_shapes=(in_shape,)
        )

    def _gen_inline_array_transpose_shaped(
        self, ary_val, permutation=None, dtype=None, in_shapes=(), out_shape=()
    ):
        """Inline version of transpose_shaped that returns the transposed memref directly."""
        from mlir import ir
        from mlir.dialects import memref, linalg, bufferization, tensor

        (in_shape,) = in_shapes

        if permutation is None:
            permutation = [i for i in range(len(in_shape)).__reversed__()]

        element_type = ir.F64Type.get()
        permutation = list(permutation)
        out_shape = [in_shape[i] for i in permutation]

        # Verify permutation dimensions match
        for i, j in enumerate(permutation):
            assert out_shape[i] == in_shape[j]

        ary_val = bufferization.to_tensor(
            ir.RankedTensorType.get(in_shape, element_type), ary_val, restrict=True
        )

        bc_out = tensor.empty(out_shape, element_type)

        input_exprs = []
        for i, (in_dim, out_dim) in enumerate(zip(in_shape, out_shape)):
            input_exprs.append(ir.AffineDimExpr.get(i))

        input_exprs = [input_exprs[perm] for perm in permutation]

        input_map = ir.AffineMap.get(len(out_shape), 0, input_exprs)
        # Output map: identity mapping for output dimensions
        output_map = ir.AffineMap.get_identity(len(out_shape))

        # Create iterator types - one parallel iterator for each output dimension
        iterator_types = [
            ir.Attribute.parse("#linalg.iterator_type<parallel>") for _ in out_shape
        ]
        with ir.Location.file(__file__, read_parent_source_line() + 1, 1):

            bc_op = linalg.GenericOp(
                [bc_out.type],
                inputs=[ary_val],
                outputs=[bc_out],
                indexing_maps=[input_map, output_map],
                iterator_types=ir.ArrayAttr.get(iterator_types),
            )
            body = bc_op.regions[0].blocks.append(element_type, element_type)
            with ir.InsertionPoint(body):
                linalg.YieldOp([body.arguments[0]])

        memref_type = ir.MemRefType.get(out_shape, element_type)
        return bufferization.to_buffer(memref_type, bc_op)

    def _gen_array_matmul_shaped(
        self, lhs_matrix, rhs_matrix, in_shapes=(), out_shape=()
    ):
        from mlir import ir
        from mlir.dialects import memref, linalg, arith, tensor, bufferization

        lhs_shape, rhs_shape = in_shapes

        if len(lhs_shape) != len(rhs_shape):
            if len(lhs_shape) > len(rhs_shape):
                rhs_matrix = self._gen_array_expand_dims_shaped(
                    rhs_matrix,
                    [i for i in range(len(lhs_shape) - len(rhs_shape))],
                    in_shapes=(rhs_shape,),
                )
                rhs_shape = tuple(
                    [1 for _ in range(len(lhs_shape) - len(rhs_shape))]
                    + list(rhs_shape)
                )
            else:
                lhs_matrix = self._gen_array_expand_dims_shaped(
                    lhs_matrix,
                    [i for i in range(len(rhs_shape) - len(lhs_shape))],
                    in_shapes=(lhs_shape,),
                )
                lhs_shape = tuple(
                    [1 for _ in range(len(rhs_shape) - len(lhs_shape))]
                    + list(lhs_shape)
                )

        element_type = ir.F64Type.get()
        memref_type_out = ir.MemRefType.get(out_shape, element_type)

        dims = len(out_shape)

        total_dims = dims + 1

        batch_exprs = [ir.AffineExpr.get_dim(i) for i in range(dims - 2)]
        m_expr = ir.AffineExpr.get_dim(dims - 2)  # M dimension
        n_expr = ir.AffineExpr.get_dim(dims - 1)  # N dimension
        k_expr = ir.AffineExpr.get_dim(dims)  # K dimension (reduction)

        map_a = ir.AffineMap.get(total_dims, 0, batch_exprs + [m_expr, k_expr])
        map_b = ir.AffineMap.get(total_dims, 0, batch_exprs + [k_expr, n_expr])
        map_c = ir.AffineMap.get(total_dims, 0, batch_exprs + [m_expr, n_expr])

        indexing_maps = ir.ArrayAttr.get(
            [
                ir.AffineMapAttr.get(map_a),
                ir.AffineMapAttr.get(map_b),
                ir.AffineMapAttr.get(map_c),
            ]
        )

        iterator_types = ir.ArrayAttr.get(
            [
                ir.Attribute.parse("#linalg.iterator_type<parallel>")
                for _ in range(dims)  # batch + M + N
            ]
            + [ir.Attribute.parse("#linalg.iterator_type<reduction>")]
        )  # K

        result_tensor_type = ir.RankedTensorType.get(out_shape, element_type)
        bc_out = tensor.empty(out_shape, element_type)

        zero = arith.ConstantOp(element_type, 0.0)
        bc_out = linalg.fill(zero, outs=[bc_out])
        lhs_matrix = bufferization.to_tensor(
            ir.RankedTensorType.get(lhs_shape, element_type), lhs_matrix, restrict=True
        )
        rhs_matrix = bufferization.to_tensor(
            ir.RankedTensorType.get(rhs_shape, element_type), rhs_matrix, restrict=True
        )
        with ir.Location.file(__file__, read_parent_source_line() + 1, 1):
            generic_op = linalg.GenericOp(
                result_tensors=[result_tensor_type],
                inputs=[lhs_matrix, rhs_matrix],
                outputs=[bc_out],
                indexing_maps=indexing_maps,
                iterator_types=iterator_types,
            )

            block = generic_op.regions[0].blocks.append(
                element_type, element_type, element_type
            )
            with ir.InsertionPoint(block):
                a_val, b_val, acc_val = block.arguments
                mul = arith.mulf(a_val, b_val)
                add = arith.addf(acc_val, mul)
                linalg.yield_([add])

        return bufferization.to_buffer(memref_type_out, generic_op)

    def _gen_array_expand_dims_shaped(self, ary_val, axes, in_shapes=(), out_shape=()):
        from mlir import ir
        from mlir.dialects import memref, bufferization, tensor

        (in_shape,) = in_shapes

        element_type = ir.F64Type.get()

        result_shape = list(in_shape)
        for i in sorted(axes):
            result_shape.insert(i, 1)
        dims = len(in_shape)

        with ir.Location.file(__file__, read_parent_source_line() + 1, 1):
            ary_val_tensor = bufferization.to_tensor(
                ir.RankedTensorType.get(in_shape, element_type), ary_val, restrict=True
            )

        result_tensor_type = ir.RankedTensorType.get(result_shape, element_type)
        with ir.Location.file(__file__, read_parent_source_line() + 1, 1):
            result_tensor = tensor.expand_shape(
                result_tensor_type,
                ary_val_tensor,
                reassociation=self.build_mlir_reassociation(dims, axes),
                output_shape=[],  # [2, 3, 4]
                static_output_shape=ir.DenseI64ArrayAttr.get(result_shape),
            )

            memref_type = ir.MemRefType.get(result_shape, element_type)
        return bufferization.to_buffer(memref_type, result_tensor)

    def _gen_array_getitem_shaped(
        self, input_memref, indices, in_shapes=(), out_shape=()
    ):
        from mlir import ir
        from mlir.dialects import memref, bufferization, tensor

        (in_shape,) = in_shapes

        dims = len(in_shape)
        out_shape_list = list(out_shape)

        assert (
            len(indices) <= dims
        ), "Number of indices should be less than or equal to number of dimensions"
        if len(indices) < dims:
            indices = list(indices) + ([None] * (dims - len(indices)))

        offsets = [0] * dims
        strides = [1] * dims
        modified_axes = []

        for axis, index in enumerate(indices):
            if index is None:
                continue
            elif isinstance(index, int):
                modified_axes.append(axis)
                offsets[axis] = index
            elif isinstance(index, slice):
                index_start = index.start if index.start is not None else 0
                offsets[axis] = index_start
            else:
                raise TypeError(f"Unknown index type {type(index)}")

        element_type = ir.F64Type.get()
        input_memref_tensor = bufferization.to_tensor(
            ir.RankedTensorType.get(in_shapes[0], element_type),
            input_memref,
            restrict=True,
        )

        for axis in modified_axes:
            out_shape_list.insert(axis, 1)
        with ir.Location.file(__file__, read_parent_source_line() + 1, 1):

            result_tensor_type = ir.RankedTensorType.get(out_shape, element_type)
            result_tensor = tensor.extract_slice(
                result_tensor_type,
                input_memref_tensor,
                offsets=[],
                sizes=[],
                strides=[],
                static_offsets=ir.DenseI64ArrayAttr.get(offsets),
                static_sizes=ir.DenseI64ArrayAttr.get(out_shape_list),
                static_strides=ir.DenseI64ArrayAttr.get(strides),
            )

        memref_type = ir.MemRefType.get(out_shape, element_type)
        return bufferization.to_buffer(memref_type, result_tensor)

    def _gen_array_setitem_shaped(
        self, arr_memref, value_memref, indices, in_shapes=(), out_shape=()
    ):
        from mlir import ir
        from mlir.dialects import memref, bufferization, tensor

        (in_shape,) = in_shapes

        dims = len(in_shape)
        assert (
            len(indices) <= dims
        ), "Number of indices should be less than or equal to number of dimensions"

        if len(indices) < dims:
            indices = list(indices) + ([None] * (dims - len(indices)))

        val_shape = in_shape.copy()
        element_type = ir.F64Type.get()
        offsets = [0] * dims
        strides = [1] * dims

        modified_axes = []

        for axis, index in enumerate(indices):
            if index is None:
                continue
            elif isinstance(index, int):
                val_shape[index] = 1
                offsets[axis] = index
                modified_axes.append(axis)
            elif isinstance(index, slice):
                index_start = index.start if index.start is not None else 0
                index_step = index.step if index.step is not None else 1
                index_stop = 0
                if index.stop is None:
                    raise ValueError("Slices with undefined stop not supported")
                else:
                    index_stop = index.stop

                val_shape[axis] = (index_stop - index_start) // index_step
                offsets[axis] = index_start
                modified_axes.append(axis)
            else:
                raise TypeError(f"Unknown index type {type(index)}")

        out_strides = [1] * dims

        for i in range(dims - 1, 0, -1):
            out_strides[i - 1] = out_strides[i] * in_shape[i]
        with ir.Location.file(__file__, read_parent_source_line() + 1, 1):
            result_slice = tensor.insert_slice(
                bufferization.to_tensor(
                    ir.RankedTensorType.get(val_shape, element_type),
                    value_memref,
                    restrict=True,
                ),
                bufferization.to_tensor(
                    ir.RankedTensorType.get(in_shape, element_type),
                    arr_memref,
                    restrict=True,
                ),
                offsets=[],
                sizes=[],
                strides=[],
                static_offsets=ir.DenseI64ArrayAttr.get(offsets),
                static_sizes=ir.DenseI64ArrayAttr.get(val_shape),
                static_strides=ir.DenseI64ArrayAttr.get(strides),
            )

        with ir.Location.file(__file__, read_parent_source_line() + 1, 1):
            memref.copy(
                bufferization.to_buffer(
                    ir.MemRefType.get(in_shape, element_type), result_slice
                ),
                arr_memref,
            )

    def _gen_binop_ufunc(self, lhs_val, rhs_val, op, in_shapes=(), out_shape=()):
        from mlir.dialects import arith, func, memref, linalg, bufferization, tensor
        from mlir import ir

        lhs_shape, rhs_shape = in_shapes

        with _mlir_location_from_frame(f"binop({op})"):
            element_type = ir.F64Type.get()

            lhs_val = bufferization.to_tensor(
                ir.RankedTensorType.get(lhs_shape, element_type), lhs_val, restrict=True
            )
            rhs_val = bufferization.to_tensor(
                ir.RankedTensorType.get(rhs_shape, element_type), rhs_val, restrict=True
            )

            def do_broadcast(tensor_val, in_shape, out_shape):
                if tuple(in_shape) == tuple(out_shape):
                    return tensor_val
                bc_out = tensor.empty(out_shape, element_type)

                # Create affine expressions that map to the broadcasted dimensions
                # For a 2x1 -> 2x12 broadcast, you want (d0, d1) -> (d0, 0)
                input_exprs = []
                for i, (in_dim, out_dim) in enumerate(zip(in_shape, out_shape)):
                    if in_dim == 1 and out_dim > 1:
                        # This dimension is being broadcasted - use constant 0
                        input_exprs.append(ir.AffineConstantExpr.get(0))
                    else:
                        # This dimension is not broadcasted - use the dimension variable
                        input_exprs.append(ir.AffineDimExpr.get(i))

                input_map = ir.AffineMap.get(len(out_shape), 0, input_exprs)
                # Output map: identity mapping for output dimensions
                output_map = ir.AffineMap.get_identity(len(out_shape))

                # Create iterator types - one parallel iterator for each output dimension
                iterator_types = [
                    ir.Attribute.parse("#linalg.iterator_type<parallel>")
                    for _ in out_shape
                ]
                with ir.Location.file(__file__, read_parent_source_line() + 1, 1):

                    bc_op = linalg.GenericOp(
                        [bc_out.type],
                        inputs=[tensor_val],
                        outputs=[bc_out],
                        indexing_maps=[input_map, output_map],
                        iterator_types=ir.ArrayAttr.get(iterator_types),
                    )
                    body = bc_op.regions[0].blocks.append(element_type, element_type)
                    with ir.InsertionPoint(body):
                        linalg.YieldOp([body.arguments[0]])

                return bc_op

            # broadcast
            print(in_shapes, "->", out_shape)
            bc_lhs = do_broadcast(lhs_val, in_shapes[0], out_shape)
            bc_rhs = do_broadcast(rhs_val, in_shapes[1], out_shape)

            # Do binop
            nd = len(out_shape)

            result = tensor.empty(out_shape, element_type)
            generic_op = linalg.GenericOp(
                result_tensors=[result.type],
                inputs=[bc_lhs, bc_rhs],
                outputs=[result],
                indexing_maps=[
                    ir.AffineMap.get_identity(nd),
                    ir.AffineMap.get_identity(nd),
                    ir.AffineMap.get_identity(nd),
                ],
                iterator_types=ir.ArrayAttr.get(
                    [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * nd
                ),
            )
            with ir.Location.file(__file__, read_parent_source_line() + 1, 1):
                body = generic_op.regions[0].blocks.append(
                    element_type, element_type, element_type
                )

                with ir.InsertionPoint(body):
                    linalg.YieldOp([op(body.arguments[0], body.arguments[1])])

            memref_type = ir.MemRefType.get(out_shape, element_type)
            return bufferization.to_buffer(memref_type, generic_op)

    def _gen_unary_ufunc(self, operand, op, in_shapes=(), out_shape=()):
        from mlir.dialects import memref, linalg, bufferization, tensor, arith
        from mlir import ir

        nd = len(out_shape)
        element_type = ir.F64Type.get()
        (inshape,) = in_shapes

        (base, offset, *shapes_strides) = memref.extract_strided_metadata(operand)
        strides = shapes_strides[nd:]
        assert len(strides) == nd
        operand_tensor_type = ir.RankedTensorType.get(inshape, element_type)
        operand_tensor = bufferization.to_tensor(
            result=operand_tensor_type, buffer=operand, restrict=True
        )

        result_tensor_type = ir.RankedTensorType.get(out_shape, element_type)
        result_tensor = tensor.empty(out_shape, element_type)
        with ir.Location.file(__file__, read_parent_source_line() + 1, 1):
            generic_op = linalg.GenericOp(
                result_tensors=[result_tensor_type],
                inputs=[operand_tensor],
                outputs=[result_tensor],
                indexing_maps=[
                    ir.AffineMap.get_identity(nd),
                    ir.AffineMap.get_identity(nd),
                ],
                iterator_types=ir.ArrayAttr.get(
                    [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * nd
                ),
            )

            body = generic_op.regions[0].blocks.append(element_type, element_type)

            with ir.InsertionPoint(body):
                linalg.YieldOp([op(body.arguments[0])])

        memref_type = ir.MemRefType.get(out_shape, element_type)
        return bufferization.to_buffer(memref_type, generic_op.result_tensors)

    def _gen_reduce_ufunc(
        self, opval, axis, op, in_shapes=(), out_shape=(), *, initializer=0.0
    ):
        from mlir.dialects import arith, func, memref, linalg, bufferization, tensor
        from mlir import ir

        with _mlir_location_from_frame(f"reduceop({op})"):
            (inshape,) = in_shapes
            nd = len(inshape)
            if axis < 0:
                axis = nd + axis

            # Extract input dimensions for reduced result (all dims except the reduced one)
            element_type = ir.F64Type.get()
            reduced_shape = list(inshape)
            reduced_shape.pop(axis)

            tensor_type = ir.RankedTensorType.get(reduced_shape, element_type)

            result_reduced = tensor.empty(reduced_shape, element_type)

            opval_tensor = bufferization.to_tensor(
                ir.RankedTensorType.get(inshape, element_type), opval, restrict=True
            )

            # Necessary to fill zeros
            zero = arith.ConstantOp(element_type, initializer)
            result_reduced = linalg.fill(zero, outs=[result_reduced])
            with ir.Location.file(__file__, read_parent_source_line() + 1, 1):
                reduce_op = linalg.ReduceOp(
                    result=[tensor_type],
                    inputs=[opval_tensor],
                    inits=[result_reduced],
                    dimensions=[axis],
                )

                body = reduce_op.regions[0].blocks.append(element_type, element_type)
                with ir.InsertionPoint(body):
                    linalg.YieldOp([op(body.arguments[0], body.arguments[1])])

            # broadcast for keepdims
            bc_tensor = tensor.empty(out_shape, element_type)
            broadcasted = linalg.broadcast(
                reduce_op.results[0], outs=[bc_tensor], dimensions=[axis]
            )
            memref_type = ir.MemRefType.get(out_shape, element_type)
            return bufferization.to_buffer(memref_type, broadcasted)

    def _lower_llm_ops(self, op: str, operands: tuple, state: LowerStates):
        from mlir.dialects import arith, memref, linalg, math as mlir_math
        from mlir import ir

        print("_____", op)

        match op, operands:
            case "MathOp_Sqrt_F64<x>", (operand,):
                operand = yield operand
                return mlir_math.sqrt(operand)

            case "NpyOp_Exp_Shaped<io, operand, inshape, outshape>", (
                io,
                operand,
                inshape,
                outshape,
            ):
                (yield io)
                operand = yield operand
                oshape = TypeSpeller.apply(outshape)
                ishape = TypeSpeller.apply(inshape)
                result = self._gen_unary_ufunc(
                    operand, op=mlir_math.exp, in_shapes=(ishape,), out_shape=oshape
                )
                return result

            case "NpyOp_Add_Shaped<io, lhs, rhs, lhs_shape, rhs_shape, outshape>", (
                io,
                lhs,
                rhs,
                lhs_shape,
                rhs_shape,
                outshape,
            ):
                (yield io)
                lhs_shape = TypeSpeller.apply(lhs_shape)
                rhs_shape = TypeSpeller.apply(rhs_shape)
                outshape = TypeSpeller.apply(outshape)

                return self._gen_binop_ufunc(
                    (yield lhs),
                    (yield rhs),
                    op=arith.addf,
                    in_shapes=(lhs_shape, rhs_shape),
                    out_shape=outshape,
                )

            case (
                "NpyOp_Subtract_Shaped<io, lhs, rhs, lhs_shape, rhs_shape, outshape>",
                (io, lhs, rhs, lhs_shape, rhs_shape, outshape),
            ):
                (yield io)
                lhs_shape = TypeSpeller.apply(lhs_shape)
                rhs_shape = TypeSpeller.apply(rhs_shape)
                outshape = TypeSpeller.apply(outshape)
                return self._gen_binop_ufunc(
                    (yield lhs),
                    (yield rhs),
                    op=arith.subf,
                    in_shapes=(lhs_shape, rhs_shape),
                    out_shape=outshape,
                )

            case (
                "NpyOp_Multiply_Shaped<io, lhs, rhs, lhs_shape, rhs_shape, outshape>",
                (io, lhs, rhs, lhs_shape, rhs_shape, outshape),
            ):
                (yield io)
                lhs_shape = TypeSpeller.apply(lhs_shape)
                rhs_shape = TypeSpeller.apply(rhs_shape)
                outshape = TypeSpeller.apply(outshape)
                return self._gen_binop_ufunc(
                    (yield lhs),
                    (yield rhs),
                    op=arith.mulf,
                    in_shapes=(lhs_shape, rhs_shape),
                    out_shape=outshape,
                )

            case "NpyOp_Divide_Shaped<io, lhs, rhs, lhs_shape, rhs_shape, outshape>", (
                io,
                lhs,
                rhs,
                lhs_shape,
                rhs_shape,
                outshape,
            ):
                (yield io)
                lhs_shape = TypeSpeller.apply(lhs_shape)
                rhs_shape = TypeSpeller.apply(rhs_shape)
                outshape = TypeSpeller.apply(outshape)
                return self._gen_binop_ufunc(
                    (yield lhs),
                    (yield rhs),
                    op=arith.divf,
                    in_shapes=(lhs_shape, rhs_shape),
                    out_shape=outshape,
                )

            case "NpyOp_Max_Shaped<io, operand, axis, keepdims, inshape, outshape>", (
                io,
                operand,
                axis,
                True,
                inshape,
                outshape,
            ):
                # Implements np.max(operand, axis, keepdims=True)
                (yield io)
                op = arith.maximumf
                opval = yield operand
                oshape = TypeSpeller.apply(outshape)
                ishape = TypeSpeller.apply(inshape)
                return self._gen_reduce_ufunc(
                    opval,
                    axis,
                    op=op,
                    in_shapes=(ishape,),
                    out_shape=oshape,
                    initializer=float("-inf"),
                )

            case "NpyOp_Sum_Shaped<io, operand, axis, keepdims, inshape, outshape>", (
                io,
                operand,
                axis,
                True,
                inshape,
                outshape,
            ):
                op = arith.addf
                opval = yield operand
                oshape = TypeSpeller.apply(outshape)
                ishape = TypeSpeller.apply(inshape)
                return self._gen_reduce_ufunc(
                    opval,
                    axis,
                    op=op,
                    in_shapes=(ishape,),
                    out_shape=oshape,
                    initializer=0.0,
                )

            case "NpyOp_Reshape_Shaped<io, ary, src_nd, inshape, outshape>", (
                io,
                ary,
                nd,
                inshape,
                outshape,
            ):
                (yield io)
                ishape = TypeSpeller.apply(inshape)
                oshape = TypeSpeller.apply(outshape)
                ary_val = yield ary
                return self._gen_reshape(ary_val, in_shapes=(ishape,), out_shape=oshape)
            case (
                "NpyOp_Take_Shaped_one_index<io, ary, index, axis, src_nd, inshape, outshape>",
                (io, ary, index, -1, src_nd, inshape, outshape),
            ):
                # This is implementing np.take(ary, index, axis=-1)
                (yield io)
                ary_val = yield ary
                ishape = TypeSpeller.apply(inshape)
                oshape = TypeSpeller.apply(outshape)
                return self._gen_take_shaped(
                    ary_val, index, src_nd, in_shapes=(ishape,), out_shape=oshape
                )

            case "NpyOp_Broadcast_To_Shaped<io, ary, inshape, outshape>", (
                io,
                ary,
                inshape,
                outshape,
            ):
                # This is implementing np.broacast_to(ary, outshape)
                # outshape is the shape of the output
                (yield io)
                in_shape = TypeSpeller.apply(inshape)
                out_shape = TypeSpeller.apply(outshape)
                ary_val = yield ary

                return self._gen_static_broadcast(
                    ary_val, in_shapes=(in_shape,), out_shape=out_shape
                )

            case "NpyOp_Stack_2_Shaped<io, ary1, ary2, axis, inshape, outshape>", (
                io,
                ary1,
                ary2,
                axis,
                inshape,
                outshape,
            ):
                # This is implementing np.broacast_to(ary, outshape)
                (yield io)
                in_shape = TypeSpeller.apply(inshape)
                out_shape = TypeSpeller.apply(outshape)
                ary_val_1 = yield ary1
                ary_val_2 = yield ary2

                return self._gen_array_stack_shaped(
                    [ary_val_1, ary_val_2],
                    axis,
                    in_shapes=(in_shape,),
                    out_shape=out_shape,
                )

            case "NpyOp_Transpose_Shaped_simple<io, array, inshape, outshape>", (
                io,
                ary,
                inshape,
                outshape,
            ):
                # This is implementing np.transpose(ary)
                (yield io)
                in_shape = TypeSpeller.apply(inshape)
                out_shape = TypeSpeller.apply(outshape)
                ary_val = yield ary

                return self._gen_inline_array_transpose(ary_val)

            case (
                "NpyOp_Transpose_Shaped_explicit<io, array, inshape, reorder, outshape>",
                (io, ary, inshape, reorder, outshape),
            ):
                # This is implementing np.transpose(ary)
                (yield io)
                in_shape = TypeSpeller.apply(inshape)
                out_shape = TypeSpeller.apply(outshape)
                reorder = TypeSpeller.apply(reorder)
                ary_val = yield ary

                element_type = ir.F64Type.get()

                permutation = list(reorder)
                out_shape = out_shape
                for i, j in enumerate(permutation):
                    assert out_shape[i] == in_shape[j]

                return self._gen_inline_array_transpose_shaped(
                    ary_val,
                    permutation,
                    dtype=element_type,
                    in_shapes=(in_shape,),
                    out_shape=out_shape,
                )

            case "NpyOp_MatMul_Shaped<io, lhs, rhs, lhs_shape, rhs_shape, out_shape>", (
                io,
                lhs,
                rhs,
                lhs_shape,
                rhs_shape,
                out_shape,
            ):
                # np.matmul
                (yield io)
                lhs_val = yield lhs
                rhs_val = yield rhs
                lhs_shape = TypeSpeller.apply(lhs_shape)
                rhs_shape = TypeSpeller.apply(rhs_shape)
                out_shape = TypeSpeller.apply(out_shape)

                return self._gen_array_matmul_shaped(
                    lhs_val,
                    rhs_val,
                    in_shapes=(lhs_shape, rhs_shape),
                    out_shape=out_shape,
                )

            case (
                "NpyOp_SetitemIO_Shaped_2d_index<io, ary, value, ishape, index0, index1>",
                (io, ary, value, ishape, index0, index1),
            ):
                io_val = yield io
                index0 = TypeSpeller().apply(index0)
                index1 = TypeSpeller().apply(index1)
                ishape = TypeSpeller().apply(ishape)
                ary_val = yield ary
                value_val = yield value

                self._gen_array_setitem_shaped(
                    ary_val,
                    value_val,
                    (index0, index1),
                    in_shapes=(ishape,),
                    out_shape=(),
                )

                return (io_val,)

            case (
                "NpyOp_GetitemIO_Shaped_2d_index<io, ary, ishape, index0, index1, oshape>",
                (_, ary, ishape, index0, index1, oshape),
            ):
                index0 = TypeSpeller().apply(index0)
                index1 = TypeSpeller().apply(index1)
                ishape = TypeSpeller().apply(ishape)
                oshape = TypeSpeller().apply(oshape)
                ary_val = yield ary

                return self._gen_array_getitem_shaped(
                    ary_val, (index0, index1), in_shapes=(ishape,), out_shape=oshape
                )

            case "NpyOp_Copy_Shaped<io, ary, shape>", (io, ary, shape):
                io_val = yield io
                ary_val = yield ary
                element_type = ir.F64Type.get()
                shape = TypeSpeller().apply(shape)
                result = memref.alloc(ir.MemRefType.get(shape, element_type), [], [])
                memref.copy(ary_val, result)
                return io_val, result

            case "NpyOp_AsArray_F64<scalar>", (scalar,):
                scalar_val = yield scalar
                element_type = ir.F64Type.get()
                memref_type = ir.MemRefType.get([1], element_type)
                result = memref.alloc(memref_type, [], [])
                linalg.fill(scalar_val, outs=[result])
                return result

            case _:
                raise NotImplementedError(f"_lower_llm_ops | {op} | {operands}")

    def jit_compile(self, llmod, func_node: rg.Func, func_name="func"):
        from mlir import ir
        from mlir.dialects import func

        # Add an empty invocation function
        with ir.InsertionPoint(self.module.body), _mlir_location_from_frame():

            func_type = ir.FunctionType.get([], [])

            func_op = func.FuncOp("global_init", func_type)
            func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

            with ir.InsertionPoint(func_op.add_entry_block()):
                func.ReturnOp([])

        optimized = self.run_passes(llmod)

        in_types, out_types = [], []

        from ctypes.util import find_library

        needed_shared_libs = ("mlir_c_runner_utils", "mlir_runner_utils", "omp")
        shared_libs = [find_library(x) for x in needed_shared_libs]

        module = self.module

        with ir.InsertionPoint(module.body), _mlir_location_from_frame():
            element_type = ir.F64Type.get()
            # self._argtys and self._retty are from `.lower()`
            # TODO: ^ not good
            for aty in self._argtys:
                match aty:
                    case BeArrayType(dtype="Float64") as aty:
                        in_types.append(ir.MemRefType.get(aty.shape, element_type))
                    case NbOp_Type("Int64"):
                        in_types.append(ir.IntegerType.get_signless(64))
                    case _:
                        assert False

            aty = self._retty
            assert aty.dtype == "Float64"
            out_types.append(ir.MemRefType.get(aty.shape, element_type))

        fn_jitted = self.jit_compile_extra(
            optimized, in_types, out_types, func_name, shared_libs=shared_libs
        )
        return fn_jitted

    def jit_compile_extra(
        self,
        llmod,
        input_types,
        output_types,
        function_name="func",
        exec_engine=None,
        **execution_engine_params,
    ):
        asm = llmod.operation.get_asm(enable_debug_info=True)
        # Open a file in write mode
        with open(os.path.join(self.tmp_dir, self.curr_dir_name, "output.mlir"), "w") as file:
            file.write(asm)

        import subprocess

        subprocess.run(
            ["mlir-translate", "--mlir-to-llvmir", os.path.join(self.tmp_dir, self.curr_dir_name, "output.mlir"), "-o", os.path.join(self.tmp_dir, self.curr_dir_name, "output.ll")]
        )
        subprocess.run(
            [
                "llc",
                "-filetype=obj",
                "--relocation-model=pic",
                os.path.join(self.tmp_dir, self.curr_dir_name, "output.ll"),
                "-o",
                os.path.join(self.tmp_dir, self.curr_dir_name, "output.o"),
            ]
        )
        # TODO: The clib detection should be dynamic
        subprocess.run(
            [
                "gcc",
                "-shared",
                "-fPIC",
                os.path.join(self.tmp_dir, self.curr_dir_name, "output.o"),
                "-o",
                os.path.join(self.tmp_dir, self.curr_dir_name, "output.so"),
                "-L/home/kc611/miniconda3/envs/mlir21/lib/",
                "-lmlir_c_runner_utils",
            ]
        )
        import ctypes

        module = ctypes.CDLL(os.path.join(self.tmp_dir, self.curr_dir_name, "output.so"))

        # Build a wrapper function
        def jit_func(*args):
            import time

            def as_memref_descriptor(arr, ty):
                intptr_t = getattr(ctypes, f"c_int{8 * ctypes.sizeof(ctypes.c_void_p)}")
                ty_ptr = ctypes.POINTER(ty)

                arg0 = ctypes.cast(arr.ctypes.data, ty_ptr)
                arg1 = arg0
                arg2 = intptr_t(0)

                shapes_arg = [intptr_t(x) for x in arr.shape]
                strides_arg = [intptr_t(x // arr.itemsize) for x in arr.strides]

                return arg0, arg1, arg2, *shapes_arg, *strides_arg

            res_val = np.zeros(output_types[0].shape)

            all_args = list(args) + [res_val]
            all_arrs_as_descriptors = [
                as_memref_descriptor(arr, ctypes.c_double) for arr in all_args
            ]

            func_argtypes = [*[type(x) for arr in all_arrs_as_descriptors for x in arr]]

            module.attention.argtypes = func_argtypes

            module.attention.restype = ctypes.c_void_p

            final_args = []
            for x in all_arrs_as_descriptors:
                final_args.extend(x)

            for _ in range(10):
                pstart = time.time_ns()
                input_args = args

                assert len(input_args) == len(input_types)

                pend = time.time_ns()
                jstart = time.time_ns()
                module.attention(*final_args)
                jend = time.time_ns()

                tstart = time.time_ns()
                tend = time.time_ns()
                print(
                    f"MLIRGen: To Memref {(pend - pstart)/1000} microseconds, Exec {(jend-jstart)/1000} microseconds, To NumPy {(tend-tstart)/1000} microseconds"
                )

            return res_val

        return jit_func


compiler_config["backend"] = MlirBackend()
