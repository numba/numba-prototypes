from __future__ import annotations
import pytest
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
                  obj: Term, size: i64, elems: TermList, termVec: Vec[Term],
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

#######################################

# Install numpy function rules


@function(cost=1000)
def NpyOp_Sum(operand: Term, axis: i64Like, keepdims: BoolLike) -> Term: ...

@function
def NpyOp_Sum_Shaped(operand: Term, axis: i64Like, keepdims: BoolLike, outshape: Shape) -> Term: ...


@function(cost=1000)
def NpyOp_Max(operand: Term, axis: i64Like, keepdims: BoolLike) -> Term: ...

@function
def NpyOp_Max_Shaped(operand: Term, axis: i64Like, keepdims: BoolLike, outshape: Shape) -> Term: ...


@function(cost=1000)
def NpyOp_Exp(operand: Term) -> Term: ...

@function
def NpyOp_Exp_Shaped(operand: Term, outshape: Shape) -> Term: ...


@function(cost=1000)
def NpyOp_Subtract(lhs: Term, rhs: Term) -> Term: ...


@function
def NpyOp_Subtract_Shaped(lhs: Term, rhs: Term, outshape: Shape) -> Term: ...


@function(cost=1000)
def NpyOp_Divide(lhs: Term, rhs: Term) -> Term: ...

@function
def NpyOp_Divide_Shaped(lhs: Term, rhs: Term, outshape: Shape) -> Term: ...


@function(cost=1000)
def NpyOp_Reshape(ary: Term, src_nd: i64Like, new_shape: Term) -> Term: ...


@function
def NpyOp_Reshape_Shaped(ary: Term, src_nd: i64Like, outshape: Shape) -> Term: ...


@function
def get_ufunc_reduce_array_desc(
    in_array: ArrayDesc, axis: i64Like, keepdims: BoolLike
) -> ArrayDesc: ...




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
                nd == arrdesc.ndim,
                shape == Shape().to_append(arrdesc, 0, nd)
            ).then(
                union(res).with_(
                    op_con_special(
                        obj,
                        axis=axis_val,
                        keepdims=keepdims_val,
                        outshape=Shape.from_arraydesc(arrdesc),
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
                nd == res_arraydesc.ndim,
                shape == Shape().to_append(res_arraydesc, 0, nd)
            ).then(
                union(res).with_(
                    op_con_special(
                        lhs, rhs,
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
                nd == res_arraydesc.ndim,
                shape == Shape().to_append(res_arraydesc, 0, nd)
            ).then(
                union(res).with_(
                    op_con_special(
                        operand,
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
    def subtract(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_binary_ufunc("subtract"))
        yield from NumPyRules._make_binary_rules("subtract", NpyOp_Subtract, NpyOp_Subtract_Shaped)

    @staticmethod
    def divide(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_binary_ufunc("divide"))
        yield from NumPyRules._make_binary_rules("divide", NpyOp_Divide, NpyOp_Divide_Shaped)


@ruleset
def ruleset_numpy_promote_binop(
    op: Term, lhs: Term, rhs: Term, io: Term, arraydesc: ArrayDesc
):
    def promote_subtract(operand, opname, py_op):
        return rewrite(py_op(io, lhs, rhs)).to(
            Py_Call(
                ModuleGetAttr(Module("numpy"), opname), io, termlist(lhs, rhs)
            ),
            # when
            TypeVar(operand).getType() == arraydesc.toType(),
        )

    for operand in [lhs, rhs]:
        yield promote_subtract(operand, "subtract", Py_SubIO)
        yield promote_subtract(operand, "divide", Py_DivIO)


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
    ad: ArrayDesc,
):
    from sealir.eqsat.rvsdg_eqsat import wildcard as _wc

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
        NpyOp_Reshape_Shaped(ary, ndim, shape),
        # when
        target == NpyOp_Reshape(ary, ndim, _wc(Term)),
        TypeVar(target).getType() == ArrayType(_wc(i64), _wc(Type), shape, _wc(DataLayout)).toType(),
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
    elems: Vec[Term]):
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



@function(cost=1)
def ArrayType(ndim: i64Like, dtype: Type, shape: Shape, layout: DataLayout) -> ArrayDesc:
    ...


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


class Annotate(Expr):
    def __init__(self, term: Term, ty: Type): ...

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


from llama_functions import Backend as LlamaBackend
from ch06_mlir_backend import Backend as _ch06_MlirBackend, LowerStates

class MlirBackend(_ch06_MlirBackend):
    def __init__(self):
        super().__init__()

        self.codegen = LlamaBackend()

    def lower(self, root, argtypes):
        [func] = [child for child in root._args
                  if isinstance(child, rg.Func)]

        print(format_rvsdg(func))
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
        self._argtys = tuple(intypes.values())

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
        print("ARGS", intypes)
        print("RETURN TYPE", retty)

        argtypes = tuple(intypes.values())
        print(argtypes)

        super().lower(func, argtypes)
        return self.module

    def get_return_types(self, root):
        return (self.lower_type(self._retty),)

    def lower_type(self, ty):
        from mlir.ir import MemRefType, ShapedType, F64Type, Location
        if isinstance(ty, BeArrayType):
            # TODO: make this use static shape
            assert ty.dtype == "Float64"
            with self.context:
                with Location.unknown():
                    element_type = F64Type.get()
                    return MemRefType.get([ShapedType.get_dynamic_size()] * ty.ndim, element_type)
        else:
            return super().lower_type(ty)

    def lower_expr(self, expr: SExpr, state: LowerStates):
        match expr:
            case LLM_generic(desc=str(op), operands=tuple(operands)):
                return self._lower_llm_ops(op, operands, state)
            case _:
                return super().lower_expr(expr, state)

    def _get_func_by_name(self, fname: str):
        for decl in self.module.body:
            if decl.sym_name.value == fname:
                return decl

    def _lower_llm_ops(self, op: str, operands: tuple, state: LowerStates):
        from mlir.dialects import arith, func, memref, linalg, math
        from mlir import ir
        be: LlamaBackend = self.codegen
        match op, operands:
            case "NpyOp_Exp_Shaped<operand, outshape>", (operand, outshape):
                shape = TypeSpeller.apply(outshape)
                nd = len(shape)
                fname_exp = be.gen_array_unary(self.module, nd, math.exp, None)
                f_exp = self._get_func_by_name(fname_exp)
                [op_ty, result_type] = f_exp.type.inputs
                operand_val = (yield operand)

                dynshape = ir.ShapedType.get_dynamic_size()
                final_shape = [dynshape] * nd
                memref_type = ir.MemRefType.get(final_shape, result_type.element_type)
                result = memref.AllocOp(memref_type, [memref.DimOp(operand_val, arith.ConstantOp(ir.IndexType.get(), i)) for i in range(nd)], [])
                func.call((), fname_exp, [operand_val, result])

                return result
            case "NpyOp_Divide_Shaped<lhs, rhs, outshape>", (lhs, rhs, outshape):
                # Implements np.max(operand, axis, keepdims=True)
                shape = TypeSpeller.apply(outshape)
                nd = len(shape)
                fname_div = be.gen_array_binary(self.module, nd, nd, arith.divf, None)
                f_div = self._get_func_by_name(fname_div)
                [lhs_ty, rhs_ty, result_type] = f_div.type.inputs
                lhsval = (yield lhs)
                rhsval = (yield rhs)

                dynshape = ir.ShapedType.get_dynamic_size()
                final_shape = [dynshape] * nd
                memref_type = ir.MemRefType.get(final_shape, result_type.element_type)
                result = memref.AllocOp(memref_type, [memref.DimOp(lhsval, arith.ConstantOp(ir.IndexType.get(), i)) for i in range(nd)], [])
                func.call((), fname_div, [lhsval, rhsval, result])

                return result
            case "NpyOp_Subtract_Shaped<lhs, rhs, outshape>", (lhs, rhs, outshape):
                shape = TypeSpeller.apply(outshape)
                nd = len(shape)
                fname_sub = be.gen_array_binary(self.module, nd, nd, arith.subf, None)
                f_sub = self._get_func_by_name(fname_sub)
                [lhs_ty, rhs_ty, result_type] = f_sub.type.inputs
                lhsval = (yield lhs)
                rhsval = (yield rhs)

                dynshape = ir.ShapedType.get_dynamic_size()
                final_shape = [dynshape] * nd
                memref_type = ir.MemRefType.get(final_shape, result_type.element_type)
                result = memref.AllocOp(memref_type, [memref.DimOp(lhsval, arith.ConstantOp(ir.IndexType.get(), i)) for i in range(nd)], [])
                func.call((), fname_sub, [lhsval, rhsval, result])

                return result
            case "NpyOp_Max_Shaped<operand, axis, keepdims, outshape>", (operand, axis, True, outshape):
                # Implements np.max(operand, axis, keepdims=True)
                shape = TypeSpeller.apply(outshape)
                nd = len(shape)
                if axis < 0:
                    axis = nd + axis
                fname_reduce = be.gen_array_reduce(self.module, nd, (axis,), arith.maximumf, None)
                fn_reduce = self._get_func_by_name(fname_reduce)
                [operand_type, result_type] = fn_reduce.type.inputs
                opval = (yield operand)

                # Extract input dimensions for reduced result (all dims except the reduced one)
                reduced_dims = []
                for i in range(nd):
                    if axis != i:
                        idx = arith.ConstantOp(ir.IndexType.get(), i)
                        reduced_dims.append(memref.DimOp(opval, idx))

                dynshape = ir.ShapedType.get_dynamic_size()
                reduced_shape = [dynshape] * (nd - 1)
                memref_type = ir.MemRefType.get(reduced_shape, result_type.element_type)
                result_reduced = memref.AllocOp(memref_type, reduced_dims, [])
                func.call((), fname_reduce, [opval, result_reduced])

                # broadcast - need to add dimension back at the axis position
                print(result_type)
                # Build final shape with keepdims=True (reduced dim becomes dynamic size)
                # Need to add the size-1 dimension back for allocation
                print("***HACK***")
                axis_shape = memref.DimOp(opval, arith.ConstantOp(ir.IndexType.get(), axis))
                final_dims = reduced_dims.copy()
                final_dims.insert(axis, axis_shape)

                final_shape = [dynshape] * nd
                memref_type = ir.MemRefType.get(final_shape, result_type.element_type)
                result = memref.AllocOp(memref_type, final_dims, [])
                linalg.broadcast(
                    result_reduced,
                    outs=[result],
                    dimensions=[axis],
                )
                return result
            case "NpyOp_Sum_Shaped<operand, axis, keepdims, outshape>", (operand, axis, True, outshape):
                shape = TypeSpeller.apply(outshape)
                nd = len(shape)
                if axis < 0:
                    axis = nd + axis
                fname_reduce = be.gen_array_reduce(self.module, nd, (axis,), arith.addf, None)
                fn_reduce = self._get_func_by_name(fname_reduce)
                [operand_type, result_type] = fn_reduce.type.inputs
                opval = (yield operand)

                # Extract input dimensions for reduced result (all dims except the reduced one)
                reduced_dims = []
                for i in range(nd):
                    if axis != i:
                        idx = arith.ConstantOp(ir.IndexType.get(), i)
                        reduced_dims.append(memref.DimOp(opval, idx))

                dynshape = ir.ShapedType.get_dynamic_size()
                reduced_shape = [dynshape] * (nd - 1)
                memref_type = ir.MemRefType.get(reduced_shape, result_type.element_type)
                result_reduced = memref.AllocOp(memref_type, reduced_dims, [])
                func.call((), fname_reduce, [opval, result_reduced])

                print("***HACK***")
                axis_shape = memref.DimOp(opval, arith.ConstantOp(ir.IndexType.get(), axis))
                final_dims = reduced_dims.copy()
                final_dims.insert(axis, axis_shape)

                final_shape = [dynshape] * nd
                memref_type = ir.MemRefType.get(final_shape, result_type.element_type)
                result = memref.AllocOp(memref_type, final_dims, [])
                linalg.broadcast(
                    result_reduced,
                    outs=[result],
                    dimensions=[axis],
                )
                return result
            case "NpyOp_Subtract_Shaped<lhs, rhs, outshape>", (lhs, rhs, outshape):
                # This implements np.subtract(lhs, rhs) with broadcasting
                shape = TypeSpeller.apply(outshape)
                nd = len(shape)
                fname_binary = be.gen_array_binary(self.module, nd, nd, arith.subf, None)
                fn_binary = self._get_func_by_name(fname_binary)
                [lhs_type, rhs_type, result_type] = fn_binary.type.inputs
                lhs_val = (yield lhs)
                rhs_val = (yield rhs)

                # Get output dimensions from outshape - use the largest operand to determine target size
                dynshape = ir.ShapedType.get_dynamic_size()
                result_shape = [dynshape] * nd
                memref_type = ir.MemRefType.get(result_shape, result_type.element_type)

                # Get target dimensions from outshape
                target_dims = []
                for i in range(nd):
                    # Convert each dimension from the outshape to an MLIR dimension value
                    dim_size = shape[i]
                    if isinstance(dim_size, int):
                        # Static dimension
                        target_dims.append(arith.ConstantOp(ir.IndexType.get(), dim_size))
                    else:
                        assert False, 'not supported' # TODO

                # Allocate result memref
                result = memref.AllocOp(memref_type, target_dims, [])

                # Call the generated binary function with broadcasted operands
                func.call((), fname_binary, [lhs_val, rhs_val, result])
                return result
            case "NpyOp_Reshape_Shaped<ary, src_nd, outshape>", (ary, nd, outshape):
                shape = TypeSpeller.apply(outshape)
                out_nd = len(shape)
                fname_reshape = be.gen_array_reshape(self.module, nd, shape)
                fn_reshape = self._get_func_by_name(fname_reshape)
                [src_type, result_type] = fn_reshape.type.inputs
                ary_val = (yield ary)

                # Get output dimensions from outshape - use fixed shape
                result_shape = shape  # Use the actual shape values, not dynamic
                memref_type = ir.MemRefType.get(result_shape, result_type.element_type)

                # Allocate result memref
                result = memref.AllocOp(memref_type, [], [])

                # Call the generated reshape function

                func.call((), fname_reshape, [ary_val, result])

                # Cast result to dynamic shaped type to match function signature
                out_type = ir.MemRefType.get(
                    [ir.ShapedType.get_dynamic_size()] * out_nd,
                    result_type.element_type
                )
                return memref.CastOp(out_type, result)
            case _:
                raise NotImplementedError(f"_lower_llm_ops | {op} | {operands}")

    def jit_compile(self, llmod, func_node: rg.Func, func_name="func"):
        from mlir import ir
        optimized = self.codegen.run_passes(llmod)

        in_types, out_types = [], []
        shared_libs = []

        module = self.module


        with ir.InsertionPoint(module.body), ir.Location.unknown():
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
    assert len(args) == 1
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

    ruleset_array_facts = ruleset(*input_type_rules)

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
            pipeline_debug=False,
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


@pytest.mark.xfail(reason="reduce broadcast is a hack",
                   raises=AssertionError)
def test_softmax_max():
    np.random.seed(0)
    _run_array_unary_test(softmax_max, np.random.random((3, 5)))



def softmax_x_minus_max(x):
    return x - np.max(x, axis=-1, keepdims=True)


@pytest.mark.xfail(reason="1d error",
                   raises=AssertionError)
def test_softmax_x_minux_max_1d():
    np.random.seed(0)
    _run_array_unary_test(softmax_x_minus_max, np.random.random(4))


def test_softmax_x_minux_max():
    np.random.seed(0)
    _run_array_unary_test(softmax_x_minus_max, np.random.random((1, 4)))
    _run_array_unary_test(softmax_x_minus_max, np.random.random((1, 2, 6, 4)))


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


#######################################

DEBUG = True

def _run_array_unary_test(target_function, inary):
    cres = run_compiler(target_function, [inary])
    jit_func = cres.jit_func

    got = jit_func(inary)
    desired = target_function(inary)
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
    main()
