from __future__ import annotations
import pytest
import inspect
import numpy as np
import operator
from functools import reduce
from pprint import pprint
from dataclasses import dataclass
from types import FunctionType

from ch04_1_typeinfer_ifelse import Type, TypeFloat64, TypeVar
from ch05_typeinfer_array import ArrayDesc, Broadcast, Dim, ruleset_broadcasting
from ch05_typeinfer_array import (
    ExtendEGraphToRVSDG as _ch05_ExtendEGraphToRVSDG,
    MyCostModel as _ch05_MyCostModel,
)
from ch05_typeinfer_array import (
    Grammar,
    Int64,
    NbOp_Base,
    TypeInt64,
    array_desc_rules,
    base_ruleset,
)
from ch05_typeinfer_array import compiler_config as _compiler_config
from ch05_typeinfer_array import jit_compiler, setup_argtypes, DataLayout
from ch09_whole_program_compiler_driver import CallGraphVisitor
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
    subsume,
    union,
    var,
    method,
    Ruleset,
    panic,
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


source_filename = Path(os.path.dirname(__file__)) / '..' / "examples" / "llama3" / "llama3.py"
with open(source_filename, "r") as fin:
    source_code = fin.read()

cgv = CallGraphVisitor(source_code, source_filename)
cgv.visit_all()

print("########## Symbol Table ##########")
pprint(cgv.functions)
print("########## ------------ ##########")
print("########## Global Calls ##########")
pprint(cgv.global_calls)
print("########## ------------ ##########")
print("########## Call Graph   ##########")
pprint(cgv.get_call_graph())
print("########## ------------ ##########")
print("########## Module Imported   ##########")
pprint(cgv.imported)
print("########## ------------ ##########")


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
def ruleset_tuple(tuptype: Type, expr: Term, io: Term, amt: i64,
                  target: Term,
                  obj: Term, size: i64, elems: TermList,
                  termVec: Vec[Term],
                  termVec2: Vec[Term],
                  slice: Term, io2: Term,
                  lhs: Term, rhs: Term,
                  lhs_size: i64, rhs_size: i64):
    from egglog import panic
    yield rule(
        # handle value_tuple[:-1]
        slice == Py_SliceIO(io, Term.LiteralNone(), Term.LiteralI64(amt), Term.LiteralNone()),
        expr == Py_SubscriptIO(io2, obj, slice.getPort(1)),
        TypeVar(obj).getType() == TypeTuple(size),
    ).then(
        union(expr.getPort(0)).with_(io),
        union(expr.getPort(0)).with_(io2),
        union(expr.getPort(1)).with_(tuple_slice_upper(obj, amt))
    )
    # tuple building
    yield rule(
        expr == Py_Tuple(TermList(termVec)),
        size == termVec.length(),
    ).then(
        set_(TypeVar(expr).getType()).to(TypeTuple(size))
    )
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
    ).to(
        Py_Tuple(TermList(termVec.append(termVec2)))
    )

#######################################

# Install numpy function rules


@function(cost=1000)
def NpyOp_Sum(operand: Term, axis: i64Like, keepdims: BoolLike) -> Term: ...

@function
def NpyOp_Sum_Shaped(operand: Term, axis: i64Like, keepdims: BoolLike,
                     inshape: Shape, outshape: Shape) -> Term: ...


@function(cost=1000)
def NpyOp_Max(operand: Term, axis: i64Like, keepdims: BoolLike) -> Term: ...

@function
def NpyOp_Max_Shaped(operand: Term, axis: i64Like, keepdims: BoolLike,
                     inshape: Shape, outshape: Shape) -> Term: ...


@function(cost=1000)
def NpyOp_Exp(operand: Term) -> Term: ...

@function
def NpyOp_Exp_Shaped(operand: Term, inshape: Shape, outshape: Shape) -> Term: ...


@function(cost=1000)
def NpyOp_Add(lhs: Term, rhs: Term) -> Term: ...


@function
def NpyOp_Add_Shaped(lhs: Term, rhs: Term,
                     lhs_shape: Shape, rhs_shape: Shape,
                     outshape: Shape) -> Term: ...



@function(cost=1000)
def NpyOp_Subtract(lhs: Term, rhs: Term) -> Term: ...


@function
def NpyOp_Subtract_Shaped(lhs: Term, rhs: Term,
                          lhs_shape: Shape, rhs_shape: Shape,
                          outshape: Shape) -> Term: ...


@function(cost=1000)
def NpyOp_Multiply(lhs: Term, rhs: Term) -> Term: ...


@function
def NpyOp_Multiply_Shaped(lhs: Term, rhs: Term,
                          lhs_shape: Shape, rhs_shape: Shape,
                          outshape: Shape) -> Term: ...


@function(cost=1000)
def NpyOp_Divide(lhs: Term, rhs: Term) -> Term: ...

@function
def NpyOp_Divide_Shaped(lhs: Term, rhs: Term,
                        lhs_shape: Shape, rhs_shape: Shape,
                        outshape: Shape) -> Term: ...


@function(cost=1000)
def NpyOp_Reshape(ary: Term, src_nd: i64Like, new_shape: Term) -> Term: ...


@function
def NpyOp_Reshape_Shaped(ary: Term, src_nd: i64Like, inshape: Shape, outshape: Shape) -> Term: ...


@function(cost=1000)
def NpyOp_Take_one_index(ary: Term, index: i64Like, axis: i64Like) -> Term: ...

@function
def NpyOp_Take_Shaped_one_index(ary: Term, index: i64Like, axis: i64Like, src_nd: i64Like, inshape: Shape, outshape: Shape) -> Term: ...


@function(cost=1000)
def NpyOp_Broadcast_To(ary: Term, shape: Term) -> Term: ...


@function
def NpyOp_Broadcast_To_Shaped(ary: Term, inshape: Shape, outshape: Shape) -> Term: ...


@function(cost=10000)
def NpyOp_Stack_2(ary1: Term, ary2: Term, axis: i64Like) -> Term: ...


@function
def NpyOp_Stack_2_Shaped(ary1: Term, ary2: Term, axis: i64Like, inshape: Shape, outshape: Shape) -> Term: ...


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
        _ad_reduce_keepdims_not_normed(
            in_array, out_array, axis, ndim
        ),
        subsume=True,
    ).to(
        _ad_reduce_keepdims(
            in_array, out_array, ndim + axis, ndim, 0
        ),
        axis < i64(0),
    )
    yield rewrite(
        _ad_reduce_keepdims_not_normed(
            in_array, out_array, axis, ndim
        ),
        subsume=True,
    ).to(
        _ad_reduce_keepdims(
            in_array, out_array, axis, ndim, 0
        ),
        axis >= i64(0),
    )
    #   out_dim[idx]=in_dim[idx] if idx != axis
    yield rule(
        _ad_reduce_keepdims(in_array, out_array, axis, ndim, idx),
        0 <= idx,
        idx < ndim,
        idx != axis,
        axis < ndim, # valid
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
        _ad_reduce_keepdims(
            in_array, out_array, axis, ndim, idx + 1
        ),
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
def ArrayType(ndim: i64Like, dtype: Type, shape: Shape, layout: DataLayout) -> ArrayDesc:
    ...


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
        yield rule(callee).then(
            kwargs.lookup("keepdims"), kwargs.lookup("axis")
        )
        yield rewrite(callee.getPort(1)).to(
            op_constructor(args[0], axis=axis_val, keepdims=keepdims_val),
            # conditions
            Term.LiteralI64(axis_val) == kwargs.get("axis"),
            Term.LiteralBool(keepdims_val) == kwargs.get("keepdims"),
        )
        # make it pure
        yield rewrite(callee.getPort(0)).to(io)
        yield rule(
            res == op_constructor(obj, axis=axis_val, keepdims=keepdims_val),
            intype == TypeVar(obj).getType(),
            intype == arrdesc.toType(),
        ).then(
            set_(TypeVar(res).getType()).to(
                get_ufunc_reduce_array_desc(
                    arrdesc, axis_val, keepdims_val
                ).toType()
            )
        )
        # make it shape specialized
        if op_con_special is not None:
            yield rule(
                res == op_constructor(obj, axis=axis_val, keepdims=keepdims_val),
                arrdesc.toType() == TypeVar(res).getType(),
                in_ad.toType() == TypeVar(obj).getType(),
                nd == arrdesc.ndim,
                shape == Shape().to_append(arrdesc, 0, nd)
            ).then(
                union(res).with_(
                    op_con_special(
                        obj,
                        axis=axis_val,
                        keepdims=keepdims_val,
                        outshape=Shape.from_arraydesc(arrdesc),
                        inshape=Shape.from_arraydesc(in_ad)
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
            union(the_call.getPort(0)).with_(io),
            union(the_call.getPort(1)).with_(op_constructor(lhs, rhs)),
        )
        # Typing and broadcasting
        yield rule(
            res == op_constructor(lhs, rhs),
            lhs_arraydesc.toType() == TypeVar(lhs).getType(),
            rhs_arraydesc.toType() == TypeVar(rhs).getType(),
        ).then(
            set_(TypeVar(res).getType()).to(
                Broadcast(lhs_arraydesc, rhs_arraydesc).toType()
            )
        )
        yield rule(
            res == op_constructor(lhs, rhs),
            lhs_arraydesc.toType() == TypeVar(lhs).getType(),
            rhs_arraydesc.toType() == TypeVar(rhs).getType(),
            res_arraydesc.toType() == TypeVar(res).getType(),
            # if the dtype matches TODO: fix type promotion
            lhs_arraydesc.dtype == rhs_arraydesc.dtype,
        ).then(
            # set dtype
            set_(res_arraydesc.dtype).to(lhs_arraydesc.dtype),
            set_(res_arraydesc.dataLayout).to(DataLayout.strided()),  # TODO improve this
        )
        if op_con_special is not None:
            yield rule(
                res == op_constructor(lhs, rhs),
                res_arraydesc.toType() == TypeVar(res).getType(),
                lhs_arraydesc.toType() == TypeVar(lhs).getType(),
                rhs_arraydesc.toType() == TypeVar(rhs).getType(),
                nd == res_arraydesc.ndim,
                shape == Shape().to_append(res_arraydesc, 0, nd)
            ).then(
                union(res).with_(
                    op_con_special(
                        lhs, rhs,
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
            union(the_call.getPort(0)).with_(io),
            union(the_call.getPort(1)).with_(op_constructor(operand)),
        )
        # Typing and broadcasting
        yield rule(
            res == op_constructor(operand),
            operand_arraydesc.toType() == TypeVar(operand).getType(),
        ).then(set_(TypeVar(res).getType()).to(operand_arraydesc.toType()))

        if op_con_special is not None:
            yield rule(
                res == op_constructor(operand),
                res_arraydesc.toType() == TypeVar(res).getType(),
                in_ad.toType() == TypeVar(operand).getType(),
                nd == res_arraydesc.ndim,
                shape == Shape().to_append(res_arraydesc, 0, nd)
            ).then(
                union(res).with_(
                    op_con_special(
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
        yield from NumPyRules._make_binary_rules("subtract", NpyOp_Subtract, NpyOp_Subtract_Shaped)


    @staticmethod
    def multiply(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_binary_ufunc("multiply"))
        yield from NumPyRules._make_binary_rules("multiply", NpyOp_Multiply, NpyOp_Multiply_Shaped)

    @staticmethod
    def divide(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_binary_ufunc("divide"))
        yield from NumPyRules._make_binary_rules("divide", NpyOp_Divide, NpyOp_Divide_Shaped)

    @staticmethod
    def take(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_take())

        io = var("io", Term)
        obj = var("obj", Term)
        res = var("res", Term)
        args = var("args", TermList)
        kwargs = var("kwargs", TermDict)
        callee = Py_CallKwargs(
            func=npy_take(), io=io, args=args, kwargs=kwargs
        )
        index_val = var("index", i64)
        axis_val = var("axis_val", i64)
        ndim = var("ndim", i64)
        dtype = var("dtype", Type)
        layout = var("layout", DataLayout)
        shape = var("shape", Shape)
        inshape = var("inshape", Shape)
        dimVec = var("dimVec", Vec[Dim])

        yield rule(callee).then(
            kwargs.lookup("axis"), args[1]
        )
        yield rewrite(callee.getPort(1)).to(
            NpyOp_Take_one_index(args[0], index_val, axis=axis_val),
            # when
            Term.LiteralI64(axis_val) == kwargs.get("axis"),
            Term.LiteralI64(index_val) == args[1],
        )
        # make it pure
        yield rewrite(callee.getPort(0)).to(io)

        # Typing & Shaping
        yield rule(
            res == NpyOp_Take_one_index(obj, index_val, axis=-1),
            TypeVar(obj).getType() == ArrayType(ndim, dtype, shape, layout).toType(),
            shape == Shape.from_list(dimVec)
        ).then(
            set_(TypeVar(res).getType()).to(
                ArrayType(
                    ndim - 1,
                    dtype,
                    Shape.from_list(dimVec.pop()),
                    layout
                ).toType()
            )
        )
        # promote
        yield rule(
            res == NpyOp_Take_one_index(obj, index_val, axis_val),
            TypeVar(res).getType() == ArrayType(_wc(i64), _wc(Type), shape, _wc(DataLayout)).toType(),
            TypeVar(obj).getType() == ArrayType(ndim, _wc(Type), inshape, _wc(DataLayout)).toType(),
        ).then(
            union(res).with_(NpyOp_Take_Shaped_one_index(obj, index_val, axis_val, ndim, inshape, shape))
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
            call == Py_Call(
                func=npy_broadcast_to(), io=io, args=TermList(argVec)
            ),
            argVec.length() == i64(2),
        ).then(
            union(call.getPort(0)).with_(io),
            union(call.getPort(1)).with_(
                NpyOp_Broadcast_To(argVec[0], argVec[1]),
            ),
        )

        # Typing & Shaping

        yield rewrite(
            Py_Tuple(TermList(termVec)),
        ).to(
            _shape_from_tuple(termVec).toTuple(),
            # when
            NpyOp_Broadcast_To(obj, shape_tup),
        )

        yield rule(
            res == NpyOp_Broadcast_To(obj, shape_tup),
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
            NpyOp_Broadcast_To(obj, shape_tup),
        ).to(
            NpyOp_Broadcast_To_Shaped(obj, Shape.from_arraydesc(in_ad), shape),
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
        callee = Py_CallKwargs(
            func=npy_stack(), io=io, args=args, kwargs=kwargs
        )
        axis_val = var("axis_val", i64)
        ndim = var("ndim", i64)
        dtype = var("dtype", Type)
        layout = var("layout", DataLayout)
        shape1 = var("shape1", Shape)
        shape2 = var("shape2", Shape)
        dimVec1 = var("dimVec1", Vec[Dim])
        dimVec2 = var("dimVec2", Vec[Dim])
        ad1 = var("ad1", ArrayDesc)

        yield rule(callee).then(
            kwargs.lookup("axis"), args[0]
        )
        yield rewrite(callee.getPort(1)).to(
            NpyOp_Stack_2(ary1, ary2, axis=axis_val),
            # when
            Term.LiteralI64(axis_val) == kwargs.get("axis"),
            args[0] == Py_Tuple(TermList(arrayVec)),
            ary1 == arrayVec[0],
            ary2 == arrayVec[1],
        )
        # make it pure
        yield rewrite(callee.getPort(0)).to(io)

        # Typing & Shaping
        # FIXME: only does stack of two arrays using recursion

        @function(cost=1000)
        def _shape_stack_at_axis(shape1: Shape, shape2: Shape, axis_val: i64Like, ndim: i64Like) -> Shape: ...
        @function(cost=1000)
        def _shape_stack_at_axis_normalized(dimVec1: Vec[Dim], dimVec2: Vec[Dim], axis_val: i64Like) -> Shape: ...

        yield rule(
            res == NpyOp_Stack_2(ary1, ary2, axis=axis_val),
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
                    layout
                ).toType()
            )
        )
        yield rewrite(res).to(
            NpyOp_Stack_2_Shaped(ary1, ary2, axis_val,
                                 Shape.from_arraydesc(ad1),
                                 shape1),
            # when
            res == NpyOp_Stack_2(ary1, ary2, axis_val),
            ad1.toType() == TypeVar(ary1).getType(),
            TypeVar(res).getType() == ArrayType(_wc(i64), _wc(Type), shape1, _wc(DataLayout)).toType(),
        )

        yield rewrite(
            _shape_stack_at_axis(shape1, shape2, axis_val, ndim)
        ).to(
            _shape_stack_at_axis_normalized(dimVec1, dimVec2, axis_val + ndim),
            # when
            axis_val < 0,
            shape1 == Shape.from_list(dimVec1),
            shape2 == Shape.from_list(dimVec2),
        )

        yield rewrite(
            _shape_stack_at_axis(shape1, shape2, axis_val, ndim)
        ).to(
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
            Shape.from_list(Vec[Dim](dimVec1[0])) + _shape_stack_at_axis_normalized(dimVec1.remove(0), dimVec2.remove(0), axis_val-1),
            # when
            dimVec1[0] == dimVec2[0],
            axis_val != i64(0),
            dimVec1.length() > 0,
        )
        yield rewrite(
            # axis_val == 0
            _shape_stack_at_axis_normalized(dimVec1, dimVec2, axis_val)
        ).to(
            Shape.from_list(Vec[Dim](dimVec1[0], Dim.fixed(2))) +  _shape_stack_at_axis_normalized(dimVec1.remove(0), dimVec2.remove(0), axis_val),
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

@ruleset
def ruleset_numpy_promote_binop(
    op: Term, lhs: Term, rhs: Term, io: Term, arraydesc: ArrayDesc
):
    def promote_ops(operand, opname, py_op):
        return rewrite(py_op(io, lhs, rhs)).to(
            Py_Call(
                ModuleGetAttr(Module("numpy"), opname), io, termlist(lhs, rhs)
            ),
            # when
            TypeVar(operand).getType() == arraydesc.toType(),
        )

    for operand in [lhs, rhs]:
        yield promote_ops(operand, "add", Py_AddIO)
        yield promote_ops(operand, "subtract", Py_SubIO)
        yield promote_ops(operand, "multiply", Py_MulIO)
        yield promote_ops(operand, "divide", Py_DivIO)


@ruleset
def ruleset_numpy_reshape(
    ary: Term, io: Term, args: Vec[Term],
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
        target == Py_Call(func=callee.getPort(1),
                          io=callee.getPort(0),
                          args=TermList(args)),
        callee == Py_AttrIO(io, ary, "reshape"),
        args.length() == i64(1),  # expect one argument
        TypeVar(ary).getType() == ad.toType(),
        ndim == ad.ndim,
    ).then(
        union(target.getPort(1)).with_(NpyOp_Reshape(ary, ndim, args[0])),
        # shortcut io
        union(target.getPort(0)).with_(io),
        union(target.getPort(0)).with_(callee.getPort(0)),
    )
    # promote
    yield rewrite(
        target
    ).to(
        NpyOp_Reshape_Shaped(ary, ndim, Shape.from_arraydesc(ad), shape),
        # when
        target == NpyOp_Reshape(ary, ndim, _wc(Term)),
        TypeVar(target).getType() == ArrayType(_wc(i64), _wc(Type), shape, _wc(DataLayout)).toType(),
        TypeVar(ary).getType() == ad.toType(),
    )
    # type & shape inference

    @function
    def _normalize_shape_for_reshape(new_shape: Shape, size: i64Like) -> Shape: ...
    @function
    def _norm_shape_step(shape: Shape, size: i64Like, out_shape: Shape) -> Shape: ...

    yield rule(
        target == NpyOp_Reshape(ary, _wc(i64), new_shape.toTuple()),
        TypeVar(new_shape.toTuple()).getType() == TypeTuple(ndim),
        ArrayType(_wc(i64), dtype, old_shape, layout).toType() == TypeVar(ary).getType(),
        size == old_shape.size,
        new_shape.size >= 0,
    ).then(
        set_(TypeVar(target).getType()).to(
            ArrayType(
                ndim,
                dtype,
                new_shape,
                layout
            ).toType()
        ),
    )
    yield rule(
        target == NpyOp_Reshape(ary, _wc(i64), new_shape.toTuple()),
        TypeVar(new_shape.toTuple()).getType() == TypeTuple(ndim),
        ArrayType(_wc(i64), dtype, old_shape, layout).toType() == TypeVar(ary).getType(),
        size == old_shape.size,
        new_shape.size < 0,
    ).then(
        set_(TypeVar(target).getType()).to(
            ArrayType(
                ndim,
                dtype,
                _normalize_shape_for_reshape(new_shape, size),
                layout
            ).toType()
        ),
    )

    yield rule(
        NpyOp_Reshape(ary, _wc(i64), Py_Tuple(TermList(termVec))),
    ).then(
        union(Py_Tuple(TermList(termVec))).with_(_shape_from_tuple(termVec).toTuple())
    )

    yield rewrite(
        _shape_from_tuple(termVec)
    ).to(
        Shape().append(Dim.fixed(n)) + _shape_from_tuple(termVec.remove(0)),
        # when
        termVec[0] == Term.LiteralI64(n)
    )
    yield rewrite(
        _shape_from_tuple(termVec)
    ).to(
        Shape(),
        # when
        termVec.length() == i64(0)
    )


    yield rewrite(
        _normalize_shape_for_reshape(shape, size)
    ).to(
        _norm_shape_step(shape, size, Shape())
    )
    yield rewrite(
        _norm_shape_step(Shape.from_list(dimVec), size, new_shape),
    ).to(
        _norm_shape_step(Shape.from_list(dimVec.remove(0)), size / n, new_shape.append(dimVec[0])),
        # when
        dimVec[0] == Dim.fixed(n),
        n > 0,  # positive
        dimVec.length() > 0,
    )

    yield rule(
        shape == _norm_shape_step(Shape.from_list(dimVec), size, new_shape),
        # when
        dimVec[0] == Dim.fixed(-1), # dynamic size
        dimVec.length() > 0,
        n == Shape.from_list(dimVec).size,
    ).then(
        union(shape).with_(new_shape.append(Dim.fixed(size / (i64(0)-n))) + Shape.from_list(dimVec.remove(0)))
    )



@ruleset
def ruleset_numpy_shape(
    ary: Term,
    shape: Shape,
    shape1: Shape,
    shape2: Shape,
    io: Term, ad: ArrayDesc,
    ndim: i64, dtype: Type, layout: DataLayout,
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
        union(io).with_(target.getPort(0))
    )
    # shape.toTuple typing
    yield rule(
        target == shape.toTuple(),
        shape == Shape.from_list(dimVec),
        ndim == dimVec.length(),
    ).then(
        set_(TypeVar(target).getType()).to(TypeTuple(ndim))
    )
    # slicing shape[:-1]
    yield rewrite(
        tuple_slice_upper(Shape.from_list(dimVec).toTuple(), -1)
    ).to(
        Shape.from_list(dimVec.pop()).toTuple()
    )
    # shape + shape
    @function(cost=1000000)
    def _convert_term_tuple_to_shape(elems: Vec[Term]) -> Term: ...

    yield rule(
        target == Py_Tuple(TermList(elems)),
        tuple_add(Shape.from_list(dimVec).toTuple(),
                  target),
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
    ).to(
        (shape1 + shape2).toTuple()
    )

    # empty shape is the same as shape from_list([])
    yield birewrite(Shape()).to(Shape.from_list(Vec[Dim].empty()))

    # shape.size

    @function(cost=10000)
    def _shape_compute_size(dimVec: Vec[Dim]) -> DynInt: ...


    yield rule(
        shape == Shape.from_list(dimVec)
    ).then(
        _shape_compute_size(dimVec)
    )

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
        dimVec.length() > i64(1)
    )

    yield rewrite(
        _shape_compute_size(dimVec),
    ).to(
        amt,
        # when
        Dim.fixed(amt) == dimVec[0],
        dimVec.length() == i64(1)
    )
    yield rewrite(
        _shape_compute_size(dimVec),
    ).to(
        DynInt(0),
        # when
        dimVec.length() == i64(0)
    )

    # Shape building
    yield rewrite(
        shape.to_append(ad, idx, ndim)
    ).to(
        shape.append(dim).to_append(ad, idx + 1, ndim),
        dim == ad.dim(idx),
        idx < ndim,
    )
    yield rewrite(
        shape.to_append(ad, ndim, ndim)
    ).to(
        shape
    )
    yield rewrite(Shape.from_arraydesc(ad), subsume=True).to(
        Shape().to_append(ad, 0, ndim),
        ndim == ad.ndim
    )
    yield rewrite(Shape().append(dim)).to(
        Shape.from_list(Vec[Dim](dim))
    )
    yield rewrite(
        Shape.from_list(dimVec).append(dim)
    ).to(
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
                Py_Tuple(
                    TermList(Vec[Term](Term.LiteralI64(amt)))
                ),
                Shape.from_list(dimVec.remove(0)).toTuple(),
            )
        ),
    )
    yield rule(
        target == Shape().toTuple(),
    ).then(
        union(target).with_(
            Py_Tuple(TermList(Vec[Term].empty()))
        ),
    )



numpy_rulesset = (
    ruleset_numpy_reshape
    | ruleset_numpy_promote_binop
    | ruleset_numpy_shape
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



loaded_module = {
    "numpy": NumPyRules,
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
            rules = module_function_rule_lookup(modname)
            yield ruleset(*rules, name=f"ruleset_module_{modname}")


module_rulesets = (
    reduce(operator.or_, make_function_rule(cgv.imported))
    | numpy_rulesset
)

######################################
# Explain array desc


@ruleset
def ruleset_explain_array_desc(ad: ArrayDesc, ndim: i64, dtype: Type, shape: Shape, dim: Dim, idx: i64,
                               layout: DataLayout, dimVec: Vec[Dim], dimVec2: Vec[Dim]):
    # ArrayType spelling
    yield rule(
        ndim == ad.ndim,
        dtype == ad.dtype,
        layout == ad.dataLayout,
    ).then(
        union(ad).with_(ArrayType(
            ndim=ndim, dtype=dtype,
            shape=Shape().to_append(ad, 0, ndim),
            layout=layout,
        ))
    )
    # inverse

    @function
    def _array_desc_set_dim(ad: ArrayDesc, dimVec: Vec[Dim], i: i64Like) -> Unit:
        ...

    yield rule(
        ad == ArrayType(
            ndim=ndim, dtype=dtype,
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
    ).then(
        Annotate(term, typ)
    )


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

    yield rule(op == Nb_Neg_Int64(x)).then(
        set_(TypeVar(op).getType()).to(TypeInt64)
    )


ruleset_extra_builtin_operations = ruleset_type_infer_negate

#######################################

class LLM_generic(NbOp_Base):
    desc: str
    operands: tuple[SExpr, ...]


class CostModel(_ch05_MyCostModel):
    def get_cost_function(self, nodename, op, ty, cost, children):
        match op:
            case "·.getType" if nodename.endswith("-TypeVar_getType"):
                return self.get_simple(float('inf'))
        cost = super().get_cost_function(nodename, op, ty, cost, children)
        return cost


class ExtendEGraphToRVSDG(_ch05_ExtendEGraphToRVSDG):
    def handle_Term(self, op: str, children: dict | list, grm: Grammar):
        parent_output = super().handle_Term(op, children, grm)
        if parent_output is NotImplemented:
            assert isinstance(children, dict)
            return grm.write(
                LLM_generic(
                    desc=op + f"<{', '.join(children)}>",
                    operands=tuple(children.values()),
                )
            )
        return parent_output

    def is_type_from_egraph(self, node) -> bool:
        return node["op"] == "·.toType"



    def handle_Type(
        self, key: str, op: str, children: dict | list, grm: Grammar
    ):
        try:
            return super().handle_Type(key, op, children, grm)
        except NotImplementedError:
            assert isinstance(children, dict)
            return self.handle_generic(key, op, children, grm)

    def handle_generic(
        self, key: str, op: str, children: dict | list, grm: Grammar
    ):
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
        from ch04_1_typeinfer_ifelse import NbOp_Type
        memo = self.memo
        match expr:
            case rg.Generic(op, children):
                memo[expr] = r = self.visit_Generic(op, tuple(map(lambda x: memo.get(x, x), children)))
                assert r is not None
            case rg.GenericList(op, children):
                memo[expr] = r = list(map(lambda x: memo.get(x, x), children))
                assert r is not None
            # case LLM_generic(desc, operands):
            #     memo[expr] = r = self.visit_Generic(desc, tuple(map(lambda x: memo.get(x, x), operands)))
            #     assert r is not None
            case NbOp_Type(str(name)):
                memo[expr] = name
            case _:
                print("HAR?", ase.pretty_str(expr), type(expr))
                return None
        return expr

    def visit_Generic(self, op, children):
        match op, children:
            case "Shape", ():
                return ()
            case 'Dim.symbolic', (name,):
                return name
            case 'Dim.fixed', (name,):
                return name
            case '·.append', (asb, dim):
                return asb + [dim]
            case 'DataLayout.strided', ():
                return 'A'
            case 'DataLayout.c_contiguous', ():
                return 'C'
            case 'ArrayType', (nd, dtype, shape, layout,):
                return BeArrayType(nd, dtype, tuple(shape), layout)
            case '·.toType', (ad,):
                return ad
            case "Shape.from_list", (vec,):
                return vec
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
        [func] = [child for child in root._args
                  if isinstance(child, rg.Func)]

        # root._tape.render_dot(only_reachable=True).view()

        fname = func.fname
        beginnode = func.body.begin
        intypes = {}
        for argport in ase.search_parents(beginnode, lambda x: isinstance(x, rg.Unpack)):
            print('   .parent', argport)
            idx = argport._args[1]
            annos = list(ase.search_parents(argport, lambda x: isinstance(x, rg.Generic) and x._args[0]=='Annotate'))
            if annos:
                intypes[idx] = TypeSpeller.apply(annos[0]._args[2])
        print(intypes)

        # outtypes
        outtypes = {}
        for port in func.body.ports:
            annos = list(ase.search_parents(port.value, lambda x: isinstance(x, rg.Generic) and x._args[0]=='Annotate'))
            if annos:
                outtypes[port.name] = TypeSpeller.apply(annos[0]._args[2])
        retty = outtypes['!ret']

        # attrs = Attributes(func.body.begin.attrs)
        # retty = attrs.get_return_type(func.body)
        print("ARGS", intypes)
        print("RETURN TYPE", retty)
        return format_rvsdg(func)

    def jit_compile(self, module, extracted, export_name):
        return module  # TODO


def _mlir_location_from_frame(info: str=""):
    """Helper to create MLIR location from Python frame information
    """
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


from llama_functions import Backend as LlamaBackend
from ch06_mlir_backend import Backend as _ch06_MlirBackend, LowerStates

class MlirBackend(_ch06_MlirBackend):
    def __init__(self):
        super().__init__()

        self.codegen = LlamaBackend()


    def get_last_compiled_return_type(self):
        return self._retty

    def _cast_return_value(self, val):
        from mlir.dialects import memref
        resty =  self.lower_type_return(self._retty)
        return memref.CastOp(resty, val)

    def lower(self, root, argtypes):
        self._retty = None # reset
        [func] = [child for child in root._args
                  if isinstance(child, rg.Func)]

        # HACK
        # Find arguments
        argfacts = [child for child in root._args if isinstance(child, rg.Generic) and child.name=="ArgFact"]

        print(format_rvsdg(func))
        fname = func.fname
        beginnode = func.body.begin
        intypes = {}

        for argfact in argfacts:
            [arg_idx, ty] = argfact.children
            intypes[arg_idx] = TypeSpeller.apply(ty)

        print(intypes)
        ninports = len(beginnode.inports)
        assert len(intypes) == ninports - 1  # one extra for the IO
        self._argtys = tuple([intypes[i] for i in range(len(argfacts))])

        # outtypes
        outtypes = {}
        for port in func.body.ports:
            annos = list(ase.search_parents(port.value, lambda x: isinstance(x, rg.Generic) and x._args[0]=='Annotate'))
            if annos:
                outtypes[port.name] = TypeSpeller.apply(annos[0]._args[2])
        retty = outtypes['!ret']
        self._retty = retty  # TODO XXX ugly smelly code

        # attrs = Attributes(func.body.begin.attrs)
        # retty = attrs.get_return_type(func.body)
        print("ARGS", self._argtys)
        print("RETURN TYPE", retty)

        argtypes = self._argtys
        print(argtypes)

        super().lower(func, argtypes)

        print(self.module.dump())
        return self.module

    def lower_type_return(self, ty):
        from mlir.ir import MemRefType, F64Type, StridedLayoutAttr, ShapedType
        with self.context, _mlir_location_from_frame():
            ty = self._retty
            element_type = F64Type.get()
            nd = len(ty.shape)
            dyn = ShapedType.get_dynamic_stride_or_offset()
            return MemRefType.get(ty.shape, element_type, layout=StridedLayoutAttr.get(dyn, [dyn] * nd))

    def get_return_types(self, root):
        return [self.lower_type_return(self._retty)]

    def lower_type(self, ty):
        from mlir.ir import MemRefType, F64Type
        if isinstance(ty, BeArrayType):
            # TODO: make this use static shape
            assert ty.dtype == "Float64"
            with self.context:
                with _mlir_location_from_frame():
                    element_type = F64Type.get()
                    return MemRefType.get(ty.shape, element_type)
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

    def _gen_reshape(self, ary_val, inshape: list[int], outshape: list[int]):
        from mlir.dialects import memref, arith
        from mlir import ir
        element_type = ir.F64Type.get()
        index_type = ir.IndexType.get()

        shape_memref = memref.AllocOp(ir.MemRefType.get([len(outshape)], index_type), [], [])
        memref_type_res = ir.MemRefType.get(outshape, element_type)
        for idx, i in enumerate(outshape):
            memref.store(arith.constant(index_type, i), shape_memref, [arith.constant(index_type, idx)])
        out = memref.reshape(memref_type_res, ary_val, shape=shape_memref)
        return out

    def _gen_static_broadcast(self, array_val, in_shape: list[int], out_shape: list[int]):
        from mlir.dialects import memref
        from mlir import ir

        if in_shape == out_shape:
            return array_val

        element_type = ir.F64Type.get()

        # Use SubView-based broadcasting for numpy broadcast semantics
        static_offsets = []
        static_sizes = []
        calculated_strides = []
        static_strides = []

        # Calculate broadcasting dimensions with proper strides
        # For n-dimensional arrays, stride[i] = product of dimensions[i+1:]
        current_stride = 1
        for i in reversed(range(len(in_shape))):
            rhs_dim = in_shape[i]
            out_dim = out_shape[i]

            if rhs_dim == 1 and out_dim > 1:
                # Broadcast dimension: stride = 0 to repeat the single element
                static_offsets.insert(0, 0)
                static_sizes.insert(0, out_dim)
                calculated_strides.insert(0, 0)
                static_strides.insert(0, 0)
            else:
                # Non-broadcast dimension: use calculated stride
                static_offsets.insert(0, 0)
                static_sizes.insert(0, rhs_dim)
                calculated_strides.insert(0, current_stride)
                static_strides.insert(0, 1)
                # Update stride for next outer dimension
                current_stride *= rhs_dim

        with _mlir_location_from_frame():
            # Create the broadcasted memref type
            layout = ir.StridedLayoutAttr.get(0, calculated_strides)
            bc_memref_type = ir.MemRefType.get(out_shape, element_type, layout=layout)

            # Create subview for broadcasting
            return memref.SubViewOp(
                bc_memref_type,
                array_val,
                offsets=[],
                sizes=[],
                strides=[],
                static_offsets=static_offsets,
                static_sizes=static_sizes,
                static_strides=static_strides
            )


    def _gen_binop_ufunc(self, lhs_val, rhs_val, lhs_shape, rhs_shape, outshape, op):
        from mlir.dialects import arith, func, memref, linalg, math
        from mlir import ir

        with _mlir_location_from_frame(f"binop({op})"):
            shape = TypeSpeller.apply(outshape)
            lhs_shape = TypeSpeller.apply(lhs_shape)
            rhs_shape = TypeSpeller.apply(rhs_shape)
            element_type = ir.F64Type.get()
            # broadcast
            bc_lhs = self._gen_static_broadcast(lhs_val, lhs_shape, shape)
            bc_rhs = self._gen_static_broadcast(rhs_val, rhs_shape, shape)
            # Do binop
            nd = len(shape)

            result = memref.AllocOp(ir.MemRefType.get(shape, element_type), [], [])
            generic_op = linalg.GenericOp(
                result_tensors=[],
                inputs=[bc_lhs, bc_rhs],
                outputs=[result],
                indexing_maps=[
                    ir.AffineMap.get_identity(nd),
                    ir.AffineMap.get_identity(nd),
                    ir.AffineMap.get_identity(nd)
                ],
                iterator_types=ir.ArrayAttr.get([ir.Attribute.parse("#linalg.iterator_type<parallel>")]*2)
            )

            body = generic_op.regions[0].blocks.append(
                element_type, element_type, element_type
            )

            with ir.InsertionPoint(body):
                linalg.YieldOp([op(body.arguments[0], body.arguments[1])])

            return result

    def _gen_unary_ufunc(self, operand, inshape, outshape, op):
        from mlir.dialects import memref, linalg
        from mlir import ir

        oshape = TypeSpeller.apply(outshape)
        nd = len(oshape)
        element_type = operand.result.type.element_type

        (base, offset, *shapes_strides) = memref.extract_strided_metadata(operand)
        strides = shapes_strides[nd:]
        assert len(strides) == nd

        result = memref.AllocOp(
            ir.MemRefType.get(oshape, element_type),
            [], []
        )

        generic_op = linalg.GenericOp(
            result_tensors=[],
            inputs=[operand],
            outputs=[result],
            indexing_maps=[
                ir.AffineMap.get_identity(nd),
                ir.AffineMap.get_identity(nd)
            ],
            iterator_types=ir.ArrayAttr.get([ir.Attribute.parse("#linalg.iterator_type<parallel>")])
        )

        body = generic_op.regions[0].blocks.append(
            element_type, element_type
        )

        with ir.InsertionPoint(body):
            linalg.YieldOp([op(body.arguments[0])])

        return result

    def _gen_reduce_ufunc(self, opval, axis, inshape: list[int], outshape: list[int], op):
        from mlir.dialects import arith, func, memref, linalg, math
        from mlir import ir

        nd = len(inshape)
        if axis < 0:
            axis = nd + axis

        # Extract input dimensions for reduced result (all dims except the reduced one)
        element_type = ir.F64Type.get()
        reduced_shape = list(inshape)
        reduced_shape.pop(axis)
        memref_type = ir.MemRefType.get(reduced_shape, element_type)
        result_reduced = memref.AllocOp(memref_type, [], [])

        # Necessary to fill zeros
        zero = arith.ConstantOp(element_type, 0.0)
        linalg.fill(zero, outs=[result_reduced])

        reduce_op = linalg.ReduceOp(
            result=[],
            inputs=[opval],
            inits=[result_reduced],
            dimensions=[axis]
        )

        body = reduce_op.regions[0].blocks.append(
            element_type, element_type
        )

        with ir.InsertionPoint(body):
            linalg.YieldOp([op(body.arguments[0], body.arguments[1])])

        # broadcast for keepdims
        memref_type = ir.MemRefType.get(outshape, element_type)
        result = memref.AllocOp(memref_type, [], [])
        linalg.broadcast(
            result_reduced,
            outs=[result],
            dimensions=[axis]
        )
        return result

    def _lower_llm_ops(self, op: str, operands: tuple, state: LowerStates):
        from mlir.dialects import arith, func, memref, linalg, math
        from mlir import ir
        be: LlamaBackend = self.codegen
        match op, operands:
            case "NpyOp_Exp_Shaped<operand, inshape, outshape>", (operand, inshape, outshape):
                operand = (yield operand)
                result = self._gen_unary_ufunc(operand, inshape, outshape, op=math.exp)
                return result

            case "NpyOp_Add_Shaped<lhs, rhs, lhs_shape, rhs_shape, outshape>", (lhs, rhs, lhs_shape, rhs_shape, outshape):
                return self._gen_binop_ufunc((yield lhs), (yield rhs), lhs_shape, rhs_shape, outshape, op=arith.addf)

            case "NpyOp_Subtract_Shaped<lhs, rhs, lhs_shape, rhs_shape, outshape>", (lhs, rhs, lhs_shape, rhs_shape, outshape):
                return self._gen_binop_ufunc((yield lhs), (yield rhs), lhs_shape, rhs_shape, outshape, op=arith.subf)

            case "NpyOp_Multiply_Shaped<lhs, rhs, lhs_shape, rhs_shape, outshape>", (lhs, rhs, lhs_shape, rhs_shape, outshape):
                return self._gen_binop_ufunc((yield lhs), (yield rhs), lhs_shape, rhs_shape, outshape, op=arith.mulf)

            case "NpyOp_Divide_Shaped<lhs, rhs, lhs_shape, rhs_shape, outshape>", (lhs, rhs, lhs_shape, rhs_shape, outshape):
                return self._gen_binop_ufunc((yield lhs), (yield rhs), lhs_shape, rhs_shape, outshape, op=arith.divf)

            case "NpyOp_Max_Shaped<operand, axis, keepdims, inshape, outshape>", (operand, axis, True, inshape, outshape):
                # Implements np.max(operand, axis, keepdims=True)
                op = arith.maximumf
                opval = (yield operand)
                oshape = TypeSpeller.apply(outshape)
                ishape = TypeSpeller.apply(inshape)
                return self._gen_reduce_ufunc(opval, axis, ishape, oshape, op=op)

            case "NpyOp_Sum_Shaped<operand, axis, keepdims, inshape, outshape>", (operand, axis, True, inshape, outshape):
                op = arith.addf
                opval = (yield operand)
                oshape = TypeSpeller.apply(outshape)
                ishape = TypeSpeller.apply(inshape)
                return self._gen_reduce_ufunc(opval, axis, ishape, oshape, op=op)

            case "NpyOp_Reshape_Shaped<ary, src_nd, inshape, outshape>", (ary, nd, inshape, outshape):
                ishape = TypeSpeller.apply(inshape)
                oshape = TypeSpeller.apply(outshape)
                ary_val = (yield ary)
                return self._gen_reshape(ary_val, ishape, oshape)
            case "NpyOp_Take_Shaped_one_index<ary, index, axis, src_nd, inshape, outshape>", (ary, index, -1, src_nd, inshape, outshape):
                # This is implementing np.take(ary, index, axis=-1)
                axis = -1
                element_type = ir.F64Type.get()
                ary_val = (yield ary)
                oshape = TypeSpeller.apply(outshape)
                ishape = TypeSpeller.apply(inshape)

                sub_shape = list(ishape)
                sub_shape[axis] = 1

                offsets = [0] * src_nd
                offsets[axis] = index

                strides = [1] * src_nd

                calculated_strides = []
                current_stride = 1
                for i in reversed(range(src_nd)):
                    dim = ishape[i]

                    calculated_strides.insert(0, current_stride)
                    # Update stride for next outer dimension
                    current_stride *= dim


                out_layout = ir.StridedLayoutAttr.get(index, calculated_strides)
                memref_type_out = ir.MemRefType.get(sub_shape, element_type, layout=out_layout)


                subview = memref.SubViewOp(
                    memref_type_out,
                    ary_val,
                    offsets=[],
                    sizes=[],
                    strides=[],
                    static_offsets=ir.DenseI64ArrayAttr.get(offsets),
                    static_sizes=ir.DenseI64ArrayAttr.get(sub_shape),
                    static_strides=ir.DenseI64ArrayAttr.get(strides)
                ).result

                result = ir.MemRefType.get(oshape, element_type, layout=ir.StridedLayoutAttr.get(index, calculated_strides[:-1]))

                # collapse the axis
                reassoc = [[x] for x in range(len(sub_shape))]
                reassoc_target = reassoc.pop(axis)
                reassoc[axis].extend(reassoc_target)
                return memref.CollapseShapeOp(
                    src=subview,
                    result=result,
                    reassociation=reassoc,
                ).result

            case "NpyOp_Broadcast_To_Shaped<ary, inshape, outshape>", (ary, inshape, outshape):
                # This is implementing np.broacast_to(ary, outshape)
                # outshape is the shape of the output
                in_shape = TypeSpeller.apply(inshape)
                out_shape = TypeSpeller.apply(outshape)
                ary_val = (yield ary)

                return self._gen_static_broadcast(ary_val, in_shape, out_shape)

            case "NpyOp_Stack_2_Shaped<ary1, ary2, axis, inshape, outshape>", (ary1, ary2, axis, inshape, outshape):
                # This is implementing np.broacast_to(ary, outshape)
                in_shape = TypeSpeller.apply(inshape)
                out_shape = TypeSpeller.apply(outshape)
                ary_val_1 = (yield ary1)
                ary_val_2 = (yield ary2)

                fname_stack = be.gen_array_stack(self.module, 2, in_shape, out_shape, axis)
                fn_stack = self._get_func_by_name(fname_stack)

                [src_type_1, src_type_2] = fn_stack.type.inputs

                element_type = src_type_1.element_type
                memref_type_out = ir.MemRefType.get(out_shape, element_type)

                result = func.call((memref_type_out,), fname_stack, [ary_val_1, ary_val_2])

                return result
            case _:
                raise NotImplementedError(f"_lower_llm_ops | {op} | {operands}")

    def jit_compile(self, llmod, func_node: rg.Func, func_name="func"):
        from mlir import ir
        optimized = self.codegen.run_passes(llmod)

        in_types, out_types = [], []

        from ctypes.util import find_library
        needed_shared_libs = ("mlir_c_runner_utils", "mlir_runner_utils")
        shared_libs = [find_library(x) for x in needed_shared_libs]

        module = self.module


        with ir.InsertionPoint(module.body), _mlir_location_from_frame():
            element_type = ir.F64Type.get()
            # self._argtys and self._retty are from `.lower()`
            # TODO: ^ not good
            for aty in self._argtys:
                assert aty.dtype == "Float64"
                in_types.append(ir.MemRefType.get(aty.shape, element_type))

            aty = self._retty
            assert aty.dtype == "Float64"
            out_types.append(ir.MemRefType.get(aty.shape, element_type))

        fn_jitted = self.jit_compile_extra(optimized, in_types, out_types, func_name, shared_libs=shared_libs)
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
        from mlir import execution_engine, runtime, ir
        import ctypes
        # Converts the MLIR module into a JIT-callable function.
        # Use MLIR's own internal execution engine
        if exec_engine is None:
            engine = execution_engine.ExecutionEngine(
                llmod, **execution_engine_params
            )
        else:
            engine = exec_engine

        assert (
            len(output_types) == 1
        ), "Execution of functions with output arguments > 1 not supported"
        [out_type] = output_types

        # Build a wrapper function
        def jit_func(*args):

            input_args = args

            assert len(input_args) == len(input_types)

            input_exec_ptrs = [
                self.get_exec_ptr(ty, val)[0]
                for ty, val in zip(input_types, input_args)
            ]

            with self.context:
                assert out_type.element_type == ir.F64Type.get()
            res_val = runtime.make_nd_memref_descriptor(
                rank=out_type.rank, dtype=ctypes.c_double
            )()
            res_ptr = ctypes.pointer(res_val)
            engine.invoke(function_name, ctypes.byref(res_ptr), *input_exec_ptrs)

            out = runtime.ranked_memref_to_numpy(res_ptr)
            return out


        return jit_func

compiler_config["backend"] = MlirBackend()

##########COMPILER##############

def run_compiler(target_function, args):
    input_shapes = []
    input_types = []
    input_type_rules = []

    for i, a in enumerate(args):
        assert a.dtype == np.float64
        assert a.flags.c_contiguous
        input_shapes.append(a.shape)
        desc, eg_facts = array_desc_rules(
            f"array_{i}", shape=a.shape, dtype=TypeFloat64, layout="c"
        )
        input_types.append(desc.toType())
        input_type_rules.extend(eg_facts)

        # HACK
        input_type_rules.append(rule(
            desc.toType()
        ).then(
            ArgFact(i, desc.toType())
        ))

    ruleset_array_facts = ruleset(*input_type_rules)

    # FIXME: egraph function parameters are sorted.
    #        they don't match the ordering of the actual parameters.
    #        this should be handled elsewhere.
    #        for now we just reorder it here.
    argnames = list(inspect.signature(target_function).parameters.keys())
    arg_ordered = sorted([(v, k) for k, v in enumerate(argnames)])
    input_types = [input_types[i] for k, i in arg_ordered]

    report = Report(default_expanded=True, enable_nested_metadata=True)
    try:
        out = jit_compiler(
            fn=target_function,
            argtypes=tuple(input_types),
            ruleset=(
                base_ruleset
                | py_eqsat_rules()
                | ruleset_broadcasting
                | setup_argtypes(*input_types)
                | ruleset_array_facts
                | module_rules
                | module_rulesets
                | ruleset_extra_builtin_operations
                | ruleset_ufunc_reduce_array_desc
                | ruleset_explain_array_desc
                | ruleset_typevar_annotate
                | ruleset_tuple
            ),
            pipeline_report=report,
            # pipeline_debug=True,
            # display_egraph=True,
            **compiler_config,
        )

    finally:
        pass
        # print(report.display())
        # report.display(view_html=True)

    return out

###########TESTING###############


def softmax_max(x):
    return np.max(x, axis=-1, keepdims=True)


def test_softmax_max():
    np.random.seed(0)
    _run_array_unary_test(softmax_max, np.random.random((3, 5)))


def softmax_x_minus_max(x):
    return x - np.max(x, axis=-1, keepdims=True)


def test_softmax_x_minux_max_1d():
    np.random.seed(0)
    _run_array_unary_test(softmax_x_minus_max, np.random.random(4))


def test_softmax_x_minux_max():
    np.random.seed(0)
    _run_array_unary_test(softmax_x_minus_max, np.random.random((1, 4)))
    _run_array_unary_test(softmax_x_minus_max, np.random.random((1, 2, 6, 4)))


def softmax_sum(x):
    return np.sum(x, axis=-1, keepdims=True)


def test_softmax_sum():
    np.random.seed(0)
    _run_array_unary_test(softmax_sum, np.random.random((1, 4)))
    _run_array_unary_test(softmax_sum, np.random.random((1, 2, 6, 4)))


def softmax_full(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def test_softmax_full():
    np.random.seed(0)
    _run_array_unary_test(softmax_full, np.random.random((1, 4)))
    _run_array_unary_test(softmax_full, np.random.random((1, 2, 6, 4)))


def apply_rotary_emb_reshape(xq):
    xqri = xq.reshape(xq.shape[:-1] + (-1, 2))
    return xqri


def test_apply_rotary_emb_reshape():
    np.random.seed(0)
    _run_array_unary_test(apply_rotary_emb_reshape, np.random.random((1, 2, 3, 4)))



def test_apply_rotary_emb_fancy_index_equiv():
    np.random.seed(0)
    arr = np.random.random((1, 2, 3, 4))
    np.testing.assert_equal(arr[..., 0], np.take(arr, 0, axis=-1))

def test_apply_rotary_emb_broadcast_to_expanddims_equiv():
    seq_len, head_dim = 5, 24
    freqs_cos = np.random.random((seq_len, head_dim))
    desired = np.broadcast_to(np.expand_dims(freqs_cos, axis=(0, 2)), (1, 5, 6, 24))
    got = np.broadcast_to(freqs_cos.reshape(1, freqs_cos.shape[0], 1, freqs_cos.shape[1]), (1, 5, 6, 24))
    np.testing.assert_equal(got, desired)


def apply_rotary_emb_fancy_index_0(xqri):
    # xq_r = xqri[..., 0]
    xq_r = np.take(xqri, 0, axis=-1)
    return xq_r

def apply_rotary_emb_fancy_index_1(xqri):
    # xq_r = xqri[..., 1]
    xq_r = np.take(xqri, 1, axis=-1)
    return xq_r


def test_apply_rotary_emb_fancy_index():
    np.random.seed(0)
    _run_array_unary_test(apply_rotary_emb_fancy_index_0,
                          np.random.random((1, 2, 3, 4)))
    _run_array_unary_test(apply_rotary_emb_fancy_index_1,
                          np.random.random((1, 2, 3, 4)))


def apply_rotary_emb_expand_dims(freqs_cos):
    # np.expand_dims(freqs_cos, axis=(0, 2))
    # TODO actually support np.expand_dims
    return freqs_cos.reshape((1,) + (freqs_cos.shape[0],) + (1,) + (freqs_cos.shape[1],))


def test_apply_rotary_emb_expand_dims():
    np.random.seed(0)
    seq_len, head_dim = 5, 24
    freqs_cos = np.random.random((seq_len, head_dim))
    _run_array_unary_test(apply_rotary_emb_expand_dims, freqs_cos)


def apply_rotary_emb_broadcast_to(freqs_cos_expanded):
    return np.broadcast_to(freqs_cos_expanded, (1, 5, 6, 24))


def test_apply_rotary_emb_broadcast_to():
    np.random.seed(0)
    seq_len, head_dim = 5, 24
    freqs_cos_expanded = np.random.random((1, seq_len, 1, head_dim))
    # print("???", apply_rotary_emb_broadcast_to(freqs_cos_expanded).shape)
    _run_array_unary_test(apply_rotary_emb_broadcast_to, freqs_cos_expanded)


def apply_rotary_emb_ufuncs(xq_r, xq_i, freqs_cos, freqs_sin):
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin  # adjusted
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    return xq_out_r + xq_out_i

def test_apply_rotary_emb_ufuncs():
    np.random.seed(0)
    shape = 1, 5, 6, 24
    xq_r = np.random.random(shape)
    xq_i = np.random.random(shape)
    freqs_cos = np.random.random(shape)
    freqs_sin = np.random.random(shape)
    _run_array_test(apply_rotary_emb_ufuncs, (xq_r, xq_i, freqs_cos, freqs_sin))


def apply_rotary_emb_stack(xq_out_r, xq_out_i):
    return np.stack((xq_out_r, xq_out_i), axis=-1)


def test_apply_rotary_emb_stack():
    np.random.seed(0)
    shape = 1, 5, 6
    xq_out_r = np.random.random(shape)
    xq_out_i = np.random.random(shape)
    _run_array_test(apply_rotary_emb_stack, (xq_out_r, xq_out_i))


def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    xqri = xq.reshape(xq.shape[:-1] + (-1, 2))
    xkri = xk.reshape(xk.shape[:-1] + (-1, 2))
    # xq_r = xqri[..., 0]
    xq_r = np.take(xqri, 0, axis=-1)
    # xq_i = xqri[..., 1]
    xq_i = np.take(xqri, 1, axis=-1)
    # xk_r = xkri[..., 0]
    xk_r = np.take(xkri, 0, axis=-1)
    # xk_i = xkri[..., 1]
    xk_i = np.take(xkri, 1, axis=-1)

    # freqs_cos = np.broadcast_to(np.expand_dims(freqs_cos, axis=(0, 2)), (1, 5, 6, 24))
    freqs_cos = np.broadcast_to(freqs_cos.reshape((1,) + (freqs_cos.shape[0],) + (1,) + (freqs_cos.shape[1],)), (1, 5, 6, 24))
    # freqs_sin = np.broadcast_to(np.expand_dims(freqs_sin, axis=(0, 2)), (1, 5, 6, 24))
    freqs_sin = np.broadcast_to(freqs_sin.reshape((1,) + (freqs_sin.shape[0],) + (1,) + (freqs_sin.shape[1],)), (1, 5, 6, 24))

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # Combine real and imaginary parts
    xq_out = np.stack((xq_out_r, xq_out_i), axis=-1).reshape(
        xq_out_r.shape[:-1] + (-1,)
    )
    xk_out = np.stack((xk_out_r, xk_out_i), axis=-1).reshape(
        xk_out_r.shape[:-1] + (-1,)
    )

    return np.stack((xq_out, xk_out), axis=-1)



def test_apply_rotary_emb():
    np.random.seed(0)

    batch_size, seq_len, n_heads, dims = 1, 5, 6, 288
    n_local_heads, head_dim = n_heads, dims // n_heads

    xq = np.random.random((batch_size, seq_len, n_local_heads, head_dim))
    xk = np.random.random((batch_size, seq_len, n_local_heads, head_dim))
    freqs_cos = np.random.random((seq_len, head_dim // 2))
    freqs_sin = np.random.random((seq_len, head_dim // 2))
    _run_array_test(apply_rotary_emb, (xq, xk, freqs_cos, freqs_sin))



#######################################

DEBUG = True

def _run_array_unary_test(target_function, inary):
    return _run_array_test(target_function, [inary])


def _run_array_test(target_function, args):


    desired = target_function(*args)

    try:
        cres = run_compiler(target_function, args)
    finally:
        # still try to test the shape output
        be = compiler_config["backend"]
        retty = be.get_last_compiled_return_type()

        assert desired.shape == retty.shape
        assert desired.ndim == retty.ndim

    jit_func = cres.jit_func
    got = jit_func(*args)
    if DEBUG:
        print("GOT".center(80, '-'))
        print(got)
        print("DESIRED".center(80, '-'))
        print(desired)
    np.testing.assert_allclose(got, desired)


def expected_func(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def main():
    ### Use the commented out code to compile from `llama3.py`
    # target_function = cgv.functions["softmax"]
    # pprint(target_function)
    target_function = expected_func

    batch_size, seq_len, n_heads, dims, cache_size = 1, 5, 6, 288, 256
    n_local_heads, head_dim = n_heads, dims // n_heads
    softmax_input_shape = (batch_size, n_local_heads, seq_len, seq_len)
    print("softmax_input_shape", softmax_input_shape)

    inary = np.arange(np.prod(softmax_input_shape), dtype=np.float64).reshape(softmax_input_shape)
    cres = run_compiler(target_function, [inary])

    jf = cres.jit_func
    print('jitfunc', jf)
    res = jf(inary)

    desired = target_function(inary)
    if DEBUG:
        print("GOT".center(80, '-'))
        print(res)
        print("DESIRED".center(80, '-'))
        print(desired)
    np.testing.assert_allclose(res, desired)

if __name__ == "__main__":
    # test_apply_rotary_emb()
    main()
