from __future__ import annotations
import operator
from functools import reduce
from pprint import pprint
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
    rule,
    ruleset,
    set_,
    subsume,
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
)
from sealir.eqsat.py_eqsat import make_rules as py_eqsat_rules
from sealir.eqsat.rvsdg_eqsat import Term, TermDict, TermList, termlist
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
    | ruleset_numpy_promote_binop
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
target_function = cgv.functions["softmax"]
pprint(target_function)

array_x_desc, array_x_infos = array_desc_rules(
    "array_x", shape=("M", "N",), dtype=TypeFloat64, layout="c"
)
ruleset_array_facts = ruleset(
    *array_x_infos,
)


class LLM_generic(NbOp_Base):
    desc: str
    operands: tuple[SExpr, ...]


class CostModel(_ch05_MyCostModel):
    def get_cost_function(self, nodename, op, ty, cost, children):
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
                return asb + (dim,)
            case 'DataLayout.strided', ():
                return 'A'
            case 'DataLayout.c_contiguous', ():
                return 'C'
            case 'ArrayType', (*args,):
                return f'ArrayType({args})'
            case '·.toType', (ad,):
                return ad
            case "Shape.from_list", (vec,):
                return vec
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


compiler_config["backend"] = StubBackend()

report = Report(default_expanded=True, enable_nested_metadata=True)
try:
    out = jit_compiler(
        fn=target_function.ast,
        argtypes=(array_x_desc.toType(),),
        ruleset=(
            base_ruleset
            | py_eqsat_rules()
            | ruleset_broadcasting
            | setup_argtypes(array_x_desc.toType())
            | ruleset_array_facts
            | module_rules
            | module_rulesets
            | ruleset_extra_builtin_operations
            | ruleset_ufunc_reduce_array_desc
            | ruleset_explain_array_desc
            | ruleset_typevar_annotate
        ),
        pipeline_report=report,
        pipeline_debug=False,
        **compiler_config,
    )
finally:
    pass
    # print(report.display())
    # report.display(view_html=True)

print("OUTPUT".center(80, "-"))
print(out.jit_func)
